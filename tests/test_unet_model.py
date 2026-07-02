"""Unit tests for the U-Net architecture, normalization and denoiser."""

import numpy as np
import pytest
import torch

from astro_denoise_unet.unet_model import (
    TrainConfig,
    UNet,
    UNetDenoiser,
    robust_normalize,
)

# Small network so tests stay fast on CPU.
TINY = dict(base_channels=4, depth=2, device="cpu")


class TestRobustNormalize:
    def test_centers_and_scales(self):
        rng = np.random.default_rng(0)
        patch = rng.normal(500.0, 20.0, size=(64, 64))
        normalized, offset, scale = robust_normalize(patch)
        assert offset == pytest.approx(500.0, rel=0.02)
        assert scale == pytest.approx(20.0, rel=0.1)
        assert abs(float(np.median(normalized))) < 0.05

    def test_handles_nan(self):
        patch = np.ones((8, 8))
        patch[0, 0] = np.nan
        normalized, _, _ = robust_normalize(patch)
        assert np.all(np.isfinite(normalized))

    def test_constant_patch_does_not_divide_by_zero(self):
        normalized, _, scale = robust_normalize(np.full((8, 8), 7.0))
        assert scale > 0
        assert np.all(np.isfinite(normalized))


class TestUNet:
    def test_forward_preserves_shape(self):
        model = UNet(base_channels=4, depth=2)
        x = torch.randn(2, 1, 32, 32)
        assert model(x).shape == (2, 1, 32, 32)


class TestUNetDenoiser:
    def test_denoise_preserves_shape_with_padding(self):
        # 50 is not divisible by 2**depth: exercises reflect padding.
        denoiser = UNetDenoiser(**TINY)
        patch = np.random.default_rng(1).normal(100.0, 5.0, size=(50, 70))
        out = denoiser.denoise(patch)
        assert out.shape == (50, 70)
        assert out.dtype == np.float64
        assert np.all(np.isfinite(out))

    def test_non_2d_patch_raises(self):
        with pytest.raises(ValueError):
            UNetDenoiser(**TINY).denoise(np.zeros((2, 8, 8)))

    def test_fit_is_not_supported(self):
        with pytest.raises(NotImplementedError):
            UNetDenoiser(**TINY).fit([np.zeros((8, 8))])

    def test_name(self):
        assert UNetDenoiser(**TINY).name() == "unet"

    def test_training_reduces_loss(self):
        rng = np.random.default_rng(2)
        clean = [np.full((32, 32), 100.0) for _ in range(4)]
        noisy = [c + rng.normal(0, 5.0, c.shape) for c in clean]

        denoiser = UNetDenoiser(**TINY)
        config = TrainConfig(epochs=8, lr=1e-2, batch_size=2, seed=0)
        history = denoiser.train(noisy, clean, config)

        assert len(history) == 8
        assert history[-1] < history[0]
        assert denoiser.trained

    def test_save_load_roundtrip(self, tmp_path):
        denoiser = UNetDenoiser(**TINY)
        denoiser.trained = True
        path = tmp_path / "weights.pt"
        denoiser.save(path, config=TrainConfig(epochs=1))

        restored = UNetDenoiser(weights=path, **TINY)
        assert restored.trained
        patch = np.random.default_rng(3).normal(size=(32, 32))
        np.testing.assert_allclose(
            restored.denoise(patch), denoiser.denoise(patch), rtol=1e-5
        )

    def test_mismatched_pairs_raise(self):
        with pytest.raises(ValueError):
            UNetDenoiser(**TINY).train([np.zeros((8, 8))], [])


class TestTrainConfig:
    def test_loss_selection(self):
        assert isinstance(TrainConfig(loss="l1").make_loss(), torch.nn.L1Loss)
        assert isinstance(TrainConfig(loss="l2").make_loss(), torch.nn.MSELoss)
        assert isinstance(TrainConfig(loss="mse").make_loss(), torch.nn.MSELoss)
        with pytest.raises(ValueError):
            TrainConfig(loss="huber").make_loss()
