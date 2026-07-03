# astro-denoise-unet

> Baseline deep learning (U-Net) para la plataforma astro-denoise.

U-Net encoder-decoder con skip connections para denoising de parches
astronómicos de un canal. El entrenamiento es residual (estilo DnCNN: la red
predice el ruido y la imagen limpia es `input - ruido`), con normalización
robusta mediana/MAD por patch para tolerar el rango dinámico arbitrario de
imágenes astronómicas.

## Instalación

```bash
uv sync --all-extras
```

Esto crea `.venv/` e instala el paquete en modo editable (`src/astro_denoise_unet/`),
resolviendo `torch` contra el índice CPU de PyTorch (`[tool.uv.sources]` en
`pyproject.toml`), y registra el entry point `astro_denoise.methods/unet` junto
con las dependencias de desarrollo (pytest, ruff, mypy).

## API

```python
from astro_denoise_unet import UNetDenoiser, TrainConfig

denoiser = UNetDenoiser(weights="model.pt")   # o sin pesos para entrenar
clean = denoiser.denoise(patch)               # ndarray 2-D -> ndarray 2-D

config = TrainConfig(epochs=30, lr=1e-3, loss="l1", batch_size=8)
history = denoiser.train(noisy_patches, clean_patches, config)
denoiser.save("model.pt", config=config)
```

## CLI (protocolo bench)

Inferencia:

```bash
python -m astro_denoise_unet \
    --input input.fits \
    --output output.fits \
    --config config.json
```

Entrenamiento (pares `*_noisy.fits` / `*_clean.fits` en `--data-dir`):

```bash
python -m astro_denoise_unet --train \
    --data-dir /path/to/pairs \
    --weights model.pt \
    --config config.json
```

Entrenamiento con datos DC2 reales (bloque curado de bench, vía **inyección
de ruido**: el target limpio es la región real suavizada con una Gaussiana y
la entrada ruidosa es ese target más ruido gaussiano por píxel extraído del
plano VARIANCE de la propia región):

```bash
python -m astro_denoise_unet --train \
    --manifest data/curated/manifest.json \
    --weights weights/unet_dc2_v1.pt \
    --config config.json
```

`config.json` (todas las claves opcionales):

```json
{
  "weights": "model.pt",
  "base_channels": 32,
  "depth": 3,
  "device": "cpu",
  "epochs": 20,
  "lr": 0.001,
  "batch_size": 8,
  "loss": "l1",
  "patch_size": 128,
  "smooth_sigma": 1.5,
  "noise_scale": 1.0,
  "pair_seed": 42,
  "bands": ["z", "r"],
  "tracts": [3828]
}
```

Las últimas seis claves solo aplican al modo `--manifest` (receta de
inyección de ruido; queda registrada en el JSON que emite el entrenamiento).
`tracts` restringe el entrenamiento a esos tracts: entrenamiento y
evaluación deben usar tracts **disjuntos** para que el benchmark solo
puntúe regiones que la red nunca vio (p. ej. entrenar con 3828 y evaluar
con 3829).

stdout devuelve una línea JSON con los metadatos mínimos del protocolo:

```json
{"algorithm": "unet", "version": "0.1.0", "status": "ok"}
```

Sin `weights` en el config, la inferencia corre con la red sin entrenar y el
CLI lo advierte por stderr (útil solo para smoke tests).

## Notas

- Los parches se acolchan por reflexión hasta múltiplos de `2**depth`, así
  que cualquier tamaño de entrada es válido.
- Los pesos entrenados en DC2 viven en `weights/` (gitignored vía `*.pt`);
  la receta y el log de cada corrida quedan junto a los pesos
  (`train_config_*.json`, `train_report_*.json`). Se distribuirán vía
  Hugging Face Hub cuando estén validados.

## Tests

```bash
uv run pytest
```

Usan una red diminuta (`base_channels=4, depth=2`) en CPU: normalización
robusta, padding, entrenamiento que reduce la pérdida, roundtrip de pesos y
el flujo CLI completo train → inferencia.

## Desarrollo

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src/
```

Las cuatro comprobaciones (format, lint, mypy, pytest) corren en CI vía el
workflow reusable [`UniverLab/workflows`](https://github.com/UniverLab/workflows).

## Estructura del proyecto

```
astro-denoise-unet/
├── pyproject.toml
├── src/
│   └── astro_denoise_unet/
│       ├── __init__.py
│       ├── __main__.py       # CLI entry point
│       ├── interface.py      # DenoiseMethod ABC
│       ├── training_data.py  # pares por inyección de ruido desde el manifest
│       └── unet_model.py
├── tests/
└── README.md
```

---
Este submódulo forma parte de la plataforma `astro-denoise`; véase
[`CONTEXT_v2.md`](../../CONTEXT_v2.md) para arquitectura y motivación.
