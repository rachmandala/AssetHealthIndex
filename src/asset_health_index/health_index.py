from __future__ import annotations

import numpy as np


def calculate_hi(md: np.ndarray, b: float) -> np.ndarray:
    return 1.0 - (2.0 / np.pi) * np.arctan(b * md)


def calculate_ahi(md: np.ndarray, b: float) -> np.ndarray:
    hi = calculate_hi(md, b)
    return np.clip(hi * 100.0, 0.0, 100.0)
