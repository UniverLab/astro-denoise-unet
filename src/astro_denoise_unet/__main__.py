"""CLI: U-Net denoising following the astro-denoise bench protocol.

Inference (default):
    python -m astro_denoise_unet --input input.fits --output output.fits \
        [--config config.json]

Training:
    python -m astro_denoise_unet --train --data-dir /path/to/pairs \
        --weights model.pt [--config config.json]

    where the data dir contains ``*_noisy.fits`` / ``*_clean.fits`` pairs, or

    python -m astro_denoise_unet --train --manifest data/curated/manifest.json \
        --weights model.pt [--config config.json]

    which manufactures pairs from the curated DC2 block by noise injection
    (clean = smoothed real region, noisy = clean + noise drawn from the
    VARIANCE plane; see ``training_data.py``).

stdout emits a single JSON line, e.g.:
    {"algorithm": "unet", "version": "0.1.0", "status": "ok"}

Config keys (all optional):
    weights          checkpoint path used for inference
    base_channels    first-level channels of the U-Net (default 32)
    depth            encoder depth (default 3)
    device           "cpu" | "cuda"
    epochs, lr, batch_size, loss ("l1"|"l2"), weight_decay, seed
                     training hyperparameters
    smooth_sigma, noise_scale, patch_size, pair_seed, bands, tracts
                     noise-injection recipe (manifest training only);
                     "tracts" restricts training to those tracts so the
                     benchmark can evaluate on held-out ones
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from rich.console import Console

from . import __version__
from .training_data import PairConfig, pairs_from_manifest
from .unet_model import TrainConfig, UNetDenoiser


def load_config(path):
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def load_image(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
                return (np.asarray(hdu.data, dtype=np.float64), hdu.header.copy())
    raise ValueError(f"{path}: no 2-D image HDU found")


def build_denoiser(config, console):
    weights = config.get("weights")
    denoiser = UNetDenoiser(
        weights=weights,
        device=config.get("device"),
        base_channels=config.get("base_channels", 32),
        depth=config.get("depth", 3),
    )
    if weights is None:
        console.print(
            "[yellow]warning:[/yellow] no weights in config — "
            "running with untrained (randomly initialized) U-Net"
        )
    return denoiser


def run_inference(args, config, console):
    denoiser = build_denoiser(config, console)
    image, header = load_image(args.input)
    start = time.perf_counter()
    denoised = denoiser.denoise(image)
    elapsed = time.perf_counter() - start

    header["HISTORY"] = f"astro-denoise-unet {__version__}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=denoised.astype(np.float32), header=header).writeto(
        args.output, overwrite=True
    )
    console.print(
        f"[green]ok[/green] {args.input.name} -> {args.output.name} "
        f"({elapsed:.2f}s, device={denoiser.device})"
    )
    return {
        "algorithm": "unet",
        "version": __version__,
        "status": "ok",
        "input": str(args.input),
        "output": str(args.output),
        "trained_weights": bool(config.get("weights")),
        "device": str(denoiser.device),
        "elapsed_s": round(elapsed, 4),
    }


def load_pair_dir(data_dir):
    """(noisy, clean) patch lists from ``*_noisy.fits`` / ``*_clean.fits`` pairs."""
    pairs = []
    for noisy_path in sorted(data_dir.glob("*_noisy.fits")):
        clean_path = noisy_path.with_name(noisy_path.name.replace("_noisy.fits", "_clean.fits"))
        if clean_path.exists():
            pairs.append((noisy_path, clean_path))
    if not pairs:
        raise FileNotFoundError(f"no *_noisy.fits / *_clean.fits pairs found in {data_dir}")
    noisy = [load_image(n)[0] for n, _ in pairs]
    clean = [load_image(c)[0] for _, c in pairs]
    return noisy, clean, {"data_dir": str(data_dir), "n_patches": len(pairs)}


def run_training(args, config, console):
    if args.manifest:
        bands = config.get("bands")
        tracts = config.get("tracts")
        pair_config = PairConfig(
            smooth_sigma=config.get("smooth_sigma", 1.5),
            noise_scale=config.get("noise_scale", 1.0),
            patch_size=config.get("patch_size", 128),
            seed=config.get("pair_seed", 42),
            bands=list(bands) if bands else None,
            tracts=[int(t) for t in tracts] if tracts else None,
        )
        noisy, clean, source = pairs_from_manifest(args.manifest, pair_config)
        console.print(
            f"noise-injection pairs from {source['n_regions']} regions "
            f"(tracts={source['tracts']}, smooth_sigma={source['smooth_sigma']}, "
            f"noise_scale={source['noise_scale']}, patch_size={source['patch_size']}, "
            f"pair_seed={source['pair_seed']})"
        )
    else:
        noisy, clean, source = load_pair_dir(args.data_dir)

    train_config = TrainConfig(
        epochs=config.get("epochs", 20),
        lr=config.get("lr", 1e-3),
        batch_size=config.get("batch_size", 8),
        loss=config.get("loss", "l1"),
        weight_decay=config.get("weight_decay", 0.0),
        seed=config.get("seed", 42),
    )
    denoiser = UNetDenoiser(
        device=config.get("device"),
        base_channels=config.get("base_channels", 32),
        depth=config.get("depth", 3),
    )
    console.print(
        f"training on {len(noisy)} patches, {train_config.epochs} epochs, "
        f"loss={train_config.loss}, lr={train_config.lr}, "
        f"device={denoiser.device}"
    )

    def log(epoch, epochs, loss):
        console.print(f"epoch {epoch}/{epochs} loss={loss:.6f}")

    history = denoiser.train(noisy, clean, train_config, log=log)
    denoiser.save(args.weights, config=train_config)
    console.print(f"[green]saved weights:[/green] {args.weights}")
    return {
        "algorithm": "unet",
        "version": __version__,
        "status": "ok",
        "mode": "train",
        "n_pairs": len(noisy),
        "epochs": train_config.epochs,
        "final_loss": history[-1],
        "weights": str(args.weights),
        "source": source,
    }


def main(argv=None):
    console = Console(stderr=True)
    parser = argparse.ArgumentParser(
        prog="python -m astro_denoise_unet",
        description="U-Net deep-learning denoiser for astronomical FITS images.",
    )
    parser.add_argument("--input", type=Path, help="Input FITS image")
    parser.add_argument("--output", type=Path, help="Output FITS image")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train", action="store_true", help="Train instead of running inference")
    parser.add_argument(
        "--data-dir", type=Path, help="Training data dir with *_noisy/*_clean pairs"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Curated-block manifest.json: train on noise-injection pairs "
        "built from the real regions (alternative to --data-dir)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("unet_weights.pt"),
        help="Checkpoint path written by --train",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.train:
            if not args.data_dir and not args.manifest:
                parser.error("--train requires --data-dir or --manifest")
            payload = run_training(args, config, console)
        else:
            if not args.input or not args.output:
                parser.error("inference requires --input and --output")
            payload = run_inference(args, config, console)
        print(json.dumps(payload))
        return 0
    except Exception as exc:
        console.print(f"[red]error:[/red] {exc}")
        print(
            json.dumps(
                {"algorithm": "unet", "version": __version__, "status": "error", "error": str(exc)}
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
