"""Noise-injection training pairs built from a curated-block manifest."""

import json

import numpy as np
import pytest
from astropy.io import fits

from astro_denoise_unet.__main__ import main
from astro_denoise_unet.training_data import (
    PairConfig,
    injection_pair,
    load_region,
    pairs_from_manifest,
    tiles,
)


def write_region(path, rng, size=32, with_variance=True):
    """Curated-style region FITS: noisy image + VARIANCE HDU."""
    image = rng.normal(0.0, 0.05, (size, size))
    image[size // 2, size // 2] = 5.0  # a bright "source"
    hdus = [fits.PrimaryHDU(image.astype(np.float32))]
    if with_variance:
        variance = np.full((size, size), 0.05**2, dtype=np.float32)
        hdus.append(fits.ImageHDU(variance, name="VARIANCE"))
    fits.HDUList(hdus).writeto(path)


@pytest.fixture
def curated_block(tmp_path):
    """Tiny curated block: manifest + 3 regions (bands z, z, r)."""
    rng = np.random.default_rng(7)
    regions_dir = tmp_path / "regions"
    regions_dir.mkdir()
    regions = []
    for i, band in enumerate(["z", "z", "r"]):
        name = f"t0-p00-{band}-r{i}.fits"
        write_region(regions_dir / name, rng)
        regions.append({"id": name[:-5], "band": band, "path": f"regions/{name}"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"seed": 123, "bands": ["r", "z"], "regions": regions}))
    return manifest


def test_load_region_returns_image_and_variance(curated_block):
    image, variance = load_region(curated_block.parent / "regions/t0-p00-z-r0.fits")
    assert image.shape == (32, 32)
    assert variance is not None
    assert np.allclose(variance, 0.05**2)


def test_injection_pair_statistics():
    rng = np.random.default_rng(0)
    image = rng.normal(0.0, 0.1, (64, 64))
    variance = np.full((64, 64), 0.04**2)
    noisy, clean = injection_pair(image, variance, np.random.default_rng(1))
    # the clean target is smoother than the raw region
    assert clean.std() < image.std()
    # the injected noise follows the variance plane
    residual = noisy - clean
    assert abs(residual.mean()) < 0.01
    assert residual.std() == pytest.approx(0.04, rel=0.15)


def test_tiles_splits_and_drops_remainder():
    image = np.arange(32 * 32, dtype=float).reshape(32, 32)
    patches = tiles(image, 16)
    assert len(patches) == 4
    assert all(p.shape == (16, 16) for p in patches)
    assert tiles(image, None) == [image]
    with pytest.raises(ValueError):
        tiles(image, 64)


def test_pairs_from_manifest_with_band_filter(curated_block):
    noisy, clean, info = pairs_from_manifest(
        curated_block, PairConfig(patch_size=16, bands=["z"], seed=5)
    )
    assert info["n_regions"] == 2
    assert len(noisy) == len(clean) == 2 * 4
    assert info["patch_size"] == 16
    assert info["pair_seed"] == 5
    assert info["manifest_seed"] == 123


def test_pairs_from_manifest_skips_regions_without_variance(tmp_path):
    rng = np.random.default_rng(3)
    (tmp_path / "regions").mkdir()
    write_region(tmp_path / "regions/a.fits", rng, with_variance=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"regions": [{"id": "a", "band": "z", "path": "regions/a.fits"}]})
    )
    with pytest.raises(ValueError, match="no usable regions"):
        pairs_from_manifest(manifest)


def test_cli_trains_from_manifest(curated_block, tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "base_channels": 4,
                "depth": 2,
                "device": "cpu",
                "epochs": 2,
                "batch_size": 4,
                "patch_size": 16,
                "smooth_sigma": 1.0,
            }
        )
    )
    weights = tmp_path / "weights.pt"
    rc = main(
        [
            "--train",
            "--manifest",
            str(curated_block),
            "--weights",
            str(weights),
            "--config",
            str(config),
        ]
    )
    assert rc == 0
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["status"] == "ok"
    assert status["mode"] == "train"
    assert status["n_pairs"] == 3 * 4
    assert status["source"]["n_regions"] == 3
    assert status["source"]["smooth_sigma"] == 1.0
    assert weights.exists()
