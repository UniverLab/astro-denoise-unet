"""Build (noisy, clean) training pairs from a curated DC2 block via noise injection.

Real coadd regions have no clean counterpart, so supervised targets are
manufactured from the data itself:

- ``clean``  = the real region lightly smoothed with a Gaussian kernel,
  which suppresses per-pixel noise while keeping the real source
  morphology (galaxies, stars, blends) of the survey.
- ``noisy``  = ``clean`` + Gaussian noise drawn per pixel from the
  region's own VARIANCE plane, so the network learns to remove exactly
  the noise the survey reports for each pixel.

Regions come from the curated-block ``manifest.json`` written by
``astro-denoise-bench --curate`` (FITS cutouts with a VARIANCE HDU).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits


@dataclass
class PairConfig:
    """Recipe for manufacturing training pairs from curated regions.

    Parameters
    ----------
    smooth_sigma : float
        Gaussian sigma (pixels) of the smoothing that defines the clean
        target.
    noise_scale : float
        Multiplier on ``sqrt(variance)`` for the injected noise.
    patch_size : int or None
        Tile each region into non-overlapping ``patch_size`` squares
        (remainders are dropped); ``None`` keeps whole regions.
    seed : int
        Seed of the noise-injection RNG (recorded for reproducibility).
    bands : list[str] or None
        Restrict to these bands; ``None`` uses every region.
    tracts : list[int] or None
        Restrict to regions from these tracts; ``None`` uses every
        region. Training and evaluation must use disjoint tracts so the
        benchmark only scores regions the network never saw.
    """

    smooth_sigma: float = 1.5
    noise_scale: float = 1.0
    patch_size: int | None = 128
    seed: int = 42
    bands: list[str] | None = None
    tracts: list[int] | None = None


def load_region(path):
    """Read a curated-region FITS; returns ``(image, variance_or_None)``."""
    with fits.open(path) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
        variance = None
        for hdu in hdul[1:]:
            if str(hdu.header.get("EXTNAME", "")).upper() == "VARIANCE":
                variance = np.asarray(hdu.data, dtype=np.float64)
    return image, variance


def injection_pair(image, variance, rng, smooth_sigma=1.5, noise_scale=1.0):
    """Manufacture one (noisy, clean) pair from a real region.

    ``clean`` is the Gaussian-smoothed region; ``noisy`` adds zero-mean
    Gaussian noise with per-pixel sigma ``sqrt(variance) * noise_scale``.
    """
    clean = np.asarray(
        convolve(image, Gaussian2DKernel(smooth_sigma), boundary="extend"),
        dtype=np.float64,
    )
    sigma = np.sqrt(np.clip(np.nan_to_num(variance, nan=0.0), 0.0, None)) * noise_scale
    noisy = clean + rng.standard_normal(clean.shape) * sigma
    return noisy, clean


def tiles(image, patch_size):
    """Non-overlapping ``patch_size`` tiles of ``image`` (remainder dropped)."""
    if patch_size is None:
        return [image]
    ny, nx = image.shape[0] // patch_size, image.shape[1] // patch_size
    if ny == 0 or nx == 0:
        raise ValueError(f"patch_size {patch_size} exceeds image shape {image.shape}")
    return [
        image[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
        for i in range(ny)
        for j in range(nx)
    ]


def pairs_from_manifest(manifest_path, config: PairConfig | None = None):
    """Build training pairs from every curated region with a variance plane.

    Returns ``(noisy_patches, clean_patches, info)`` where ``info``
    records the recipe and the regions used (for the training report).
    """
    config = config or PairConfig()
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    rng = np.random.default_rng(config.seed)

    noisy_patches, clean_patches, used, skipped = [], [], [], []
    for region in manifest["regions"]:
        if config.bands and region["band"] not in config.bands:
            continue
        if config.tracts and region.get("tract") not in config.tracts:
            continue
        image, variance = load_region(manifest_path.parent / region["path"])
        if variance is None:
            skipped.append(region["id"])
            continue
        noisy, clean = injection_pair(
            image,
            variance,
            rng,
            smooth_sigma=config.smooth_sigma,
            noise_scale=config.noise_scale,
        )
        noisy_patches.extend(tiles(noisy, config.patch_size))
        clean_patches.extend(tiles(clean, config.patch_size))
        used.append(region["id"])

    if not noisy_patches:
        raise ValueError(f"no usable regions (with VARIANCE HDU) in {manifest_path}")

    info = {
        "manifest": str(manifest_path),
        "manifest_seed": manifest.get("seed"),
        "n_regions": len(used),
        "n_patches": len(noisy_patches),
        "skipped_regions": skipped,
        "smooth_sigma": config.smooth_sigma,
        "noise_scale": config.noise_scale,
        "patch_size": config.patch_size,
        "pair_seed": config.seed,
        "bands": config.bands or manifest.get("bands"),
        "tracts": config.tracts or manifest.get("tracts"),
    }
    return noisy_patches, clean_patches, info
