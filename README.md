# astro-denoise-unet

> Baseline deep learning (U-Net) para la plataforma astro-denoise.

U-Net encoder-decoder con skip connections para denoising de parches
astronómicos de un canal. El entrenamiento es residual (estilo DnCNN: la red
predice el ruido y la imagen limpia es `input - ruido`), con normalización
robusta mediana/MAD por patch para tolerar el rango dinámico arbitrario de
imágenes astronómicas.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pip install -e .   # registra el entry point astro_denoise.methods/unet
```

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
  "loss": "l1"
}
```

stdout devuelve una línea JSON con los metadatos mínimos del protocolo:

```json
{"algorithm": "unet", "version": "0.1.0", "status": "ok"}
```

Sin `weights` en el config, la inferencia corre con la red sin entrenar y el
CLI lo advierte por stderr (útil solo para smoke tests).

## Notas

- Los parches se acolchan por reflexión hasta múltiplos de `2**depth`, así
  que cualquier tamaño de entrada es válido.
- Los pesos entrenados en DC2 se distribuirán vía Hugging Face Hub cuando
  estén disponibles.

---
Este submódulo forma parte de la plataforma `astro-denoise`; véase
[`CONTEXT_v2.md`](../../CONTEXT_v2.md) para arquitectura y motivación.
