"""Standard denoising-method contract of the astro-denoise platform.

Every algorithm module ships (a copy of) this ABC so methods stay
importable without a shared runtime dependency; `astro-denoise-bench`
only relies on the duck-typed ``denoise`` / ``name`` protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DenoiseMethod(ABC):
    """A denoising algorithm operating on 2-D image patches."""

    @abstractmethod
    def denoise(self, patch: np.ndarray) -> np.ndarray:
        """Return the denoised version of ``patch`` (same shape/dtype float)."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier of the method (e.g. ``"unet"``)."""

    def fit(self, patches: list[np.ndarray]) -> None:
        """Optional training/calibration step."""
