"""U-Net denoiser for astronomical image patches.

A standard encoder-decoder U-Net with skip connections operating on
single-channel patches. Patches are normalized per-image with robust
statistics (median / MAD) before entering the network and denormalized
on the way out, so the model is insensitive to the arbitrary dynamic
range of astronomical data.

Training is residual: the network predicts the noise, and the denoised
image is ``input - predicted_noise`` (DnCNN-style), which converges
faster for denoising than predicting the clean image directly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .interface import DenoiseMethod


# ---------------------------------------------------------------------------
# architecture
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2, the basic U-Net block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Encoder-decoder U-Net with skip connections for 1-channel denoising.

    Parameters
    ----------
    base_channels : int
        Channels of the first encoder level; doubles at each level.
    depth : int
        Number of down/upsampling levels (input sides should be
        divisible by ``2**depth``; the denoiser pads automatically).
    """

    def __init__(self, base_channels=32, depth=3):
        super().__init__()
        self.depth = depth
        channels = [base_channels * 2**i for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        in_ch = 1
        for out_ch in channels[:-1]:
            self.encoders.append(DoubleConv(in_ch, out_ch))
            in_ch = out_ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(channels[-2], channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for skip_ch in reversed(channels[:-1]):
            self.upconvs.append(
                nn.ConvTranspose2d(skip_ch * 2, skip_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(skip_ch * 2, skip_ch))
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders,
                                         reversed(skips)):
            x = upconv(x)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.head(x)  # predicted noise residual


# ---------------------------------------------------------------------------
# training configuration and data
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 8
    loss: str = "l1"          # "l1" | "l2" (a.k.a. "mse")
    weight_decay: float = 0.0
    seed: int = 42

    def make_loss(self):
        if self.loss in ("l2", "mse"):
            return nn.MSELoss()
        if self.loss == "l1":
            return nn.L1Loss()
        raise ValueError(f"unsupported loss {self.loss!r} (use 'l1' or 'l2')")


class _PairDataset(Dataset):
    """Normalized (noisy, noise-residual) tensor pairs."""

    def __init__(self, noisy_patches, clean_patches):
        if len(noisy_patches) != len(clean_patches):
            raise ValueError("noisy and clean patch lists differ in length")
        self.items = []
        for noisy, clean in zip(noisy_patches, clean_patches):
            noisy_n, offset, scale = robust_normalize(noisy)
            clean_n = (np.asarray(clean, dtype=np.float32) - offset) / scale
            residual = noisy_n - clean_n
            self.items.append((
                torch.from_numpy(noisy_n[None]),
                torch.from_numpy(residual[None]),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def robust_normalize(patch):
    """Median/MAD normalization; returns (normalized, offset, scale)."""
    patch = np.asarray(patch, dtype=np.float32)
    finite = patch[np.isfinite(patch)]
    offset = float(np.median(finite))
    mad = float(np.median(np.abs(finite - offset)))
    scale = max(1.4826 * mad, 1e-6)
    normalized = (np.nan_to_num(patch, nan=offset) - offset) / scale
    return normalized.astype(np.float32), offset, scale


# ---------------------------------------------------------------------------
# denoiser
# ---------------------------------------------------------------------------

class UNetDenoiser(DenoiseMethod):
    """U-Net denoising method implementing the astro-denoise contract.

    Parameters
    ----------
    weights : str or Path, optional
        Checkpoint produced by :meth:`save`; loaded on construction.
    device : str, optional
        Torch device; defaults to CUDA when available.
    """

    def __init__(self, weights=None, device=None, base_channels=32, depth=3):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = UNet(base_channels=base_channels, depth=depth)
        self.trained = False
        if weights is not None:
            self.load(weights)
        self.model.to(self.device)

    def name(self) -> str:
        return "unet"

    # -- inference ----------------------------------------------------------

    @torch.no_grad()
    def denoise(self, patch: np.ndarray) -> np.ndarray:
        patch = np.asarray(patch, dtype=np.float32)
        if patch.ndim != 2:
            raise ValueError(f"expected a 2-D patch, got shape {patch.shape}")
        self.model.eval()

        normalized, offset, scale = robust_normalize(patch)
        padded, pads = self._pad(normalized)
        tensor = torch.from_numpy(padded[None, None]).to(self.device)
        residual = self.model(tensor)[0, 0].cpu().numpy()
        residual = self._unpad(residual, pads, patch.shape)
        denoised = normalized - residual
        return (denoised * scale + offset).astype(np.float64)

    def _pad(self, image):
        factor = 2**self.model.depth
        pad_y = (-image.shape[0]) % factor
        pad_x = (-image.shape[1]) % factor
        if pad_y or pad_x:
            image = np.pad(image, ((0, pad_y), (0, pad_x)), mode="reflect")
        return image, (pad_y, pad_x)

    @staticmethod
    def _unpad(image, pads, shape):
        return image[: shape[0], : shape[1]]

    # -- training -----------------------------------------------------------

    def train(self, noisy_patches, clean_patches, config: TrainConfig | None = None,
              log=None):
        """Supervised training on (noisy, clean) patch pairs.

        Returns the list of mean epoch losses. ``log`` is an optional
        callback ``(epoch, epochs, loss)``.
        """
        config = config or TrainConfig()
        torch.manual_seed(config.seed)
        dataset = _PairDataset(noisy_patches, clean_patches)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        criterion = config.make_loss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.model.train()
        history = []
        for epoch in range(1, config.epochs + 1):
            total, count = 0.0, 0
            for noisy, residual in loader:
                noisy = noisy.to(self.device)
                residual = residual.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(noisy), residual)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(noisy)
                count += len(noisy)
            epoch_loss = total / max(count, 1)
            history.append(epoch_loss)
            if log:
                log(epoch, config.epochs, epoch_loss)
        self.trained = True
        return history

    def fit(self, patches):
        """DenoiseMethod hook: unsupervised fit is not defined for U-Net."""
        raise NotImplementedError(
            "UNetDenoiser requires (noisy, clean) pairs; use train()"
        )

    # -- persistence --------------------------------------------------------

    def save(self, path, config: TrainConfig | None = None):
        payload = {
            "state_dict": self.model.state_dict(),
            "trained": self.trained,
            "train_config": asdict(config) if config else None,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(payload["state_dict"])
        self.trained = bool(payload.get("trained", True))
        self.model.to(self.device)
