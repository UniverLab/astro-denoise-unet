"""End-to-end tests of the U-Net CLI: training then inference."""

import json

import numpy as np
import pytest
from astropy.io import fits

from astro_denoise_unet.__main__ import main

TINY_CONFIG = {"base_channels": 4, "depth": 2, "device": "cpu"}


@pytest.fixture
def pair_dataset(tmp_path):
    """Directory with two (noisy, clean) FITS pairs."""
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(2):
        clean = np.full((32, 32), 100.0)
        noisy = clean + rng.normal(0, 5.0, clean.shape)
        fits.PrimaryHDU(clean.astype(np.float32)).writeto(
            data_dir / f"p{i}_clean.fits"
        )
        fits.PrimaryHDU(noisy.astype(np.float32)).writeto(
            data_dir / f"p{i}_noisy.fits"
        )
    return data_dir


def last_json(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_train_then_infer(pair_dataset, tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**TINY_CONFIG, "epochs": 2, "batch_size": 2}))
    weights = tmp_path / "weights.pt"

    rc = main([
        "--train", "--data-dir", str(pair_dataset),
        "--weights", str(weights), "--config", str(config),
    ])
    assert rc == 0
    status = last_json(capsys)
    assert status["status"] == "ok"
    assert status["mode"] == "train"
    assert status["n_pairs"] == 2
    assert weights.exists()

    infer_config = tmp_path / "infer.json"
    infer_config.write_text(json.dumps({**TINY_CONFIG, "weights": str(weights)}))
    output = tmp_path / "denoised.fits"
    rc = main([
        "--input", str(pair_dataset / "p0_noisy.fits"),
        "--output", str(output), "--config", str(infer_config),
    ])
    assert rc == 0
    status = last_json(capsys)
    assert status["status"] == "ok"
    assert status["trained_weights"] is True

    with fits.open(output) as hdul:
        denoised = np.asarray(hdul[0].data)
    assert denoised.shape == (32, 32)
    assert np.all(np.isfinite(denoised))


def test_inference_without_weights_warns_but_runs(pair_dataset, tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(TINY_CONFIG))
    rc = main([
        "--input", str(pair_dataset / "p0_noisy.fits"),
        "--output", str(tmp_path / "out.fits"), "--config", str(config),
    ])
    assert rc == 0
    assert last_json(capsys)["trained_weights"] is False


def test_train_without_pairs_is_structured_error(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = main([
        "--train", "--data-dir", str(empty),
        "--weights", str(tmp_path / "w.pt"),
    ])
    assert rc == 1
    assert last_json(capsys)["status"] == "error"
