"""Health Index (HI) and Asset Health Index (AHI) calculation.

Implements the Han et al. (2023) health scoring formulas that convert a
Mahalanobis Distance into a bounded, interpretable health score.

Formulas
--------
    HI  = 1 - (2 / pi) * arctan(b * MD)
    AHI = HI * 100

Where:
    MD = Mahalanobis Distance of the current observation from the
         healthy baseline.
    b  = configurable sensitivity parameter (``health_index.b`` in
         ``config/config.yaml``); larger ``b`` makes the index more
         sensitive to small deviations in MD.

As MD -> 0 (observation matches the healthy baseline), HI -> 1 and
AHI -> 100. As MD -> infinity, HI -> 0 and AHI -> 0.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

from src.config.settings import AppSettings, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HealthIndexCalculator:
    """Converts Mahalanobis Distance values into Health Index / AHI scores."""

    def __init__(self, settings: Optional[AppSettings] = None, b: Optional[float] = None) -> None:
        """Initialize the calculator.

        Args:
            settings: Application settings; defaults to :func:`get_settings`.
            b: Optional override for the sensitivity parameter ``b``. When
                omitted, the value is read from ``health_index.b`` in
                configuration.
        """
        self._settings = settings or get_settings()
        self.b: float = b if b is not None else self._settings.health_index.b

    def calculate_health_index(self, mahalanobis_distance: float) -> float:
        """Calculate the Health Index (HI) for a single MD value.

        Args:
            mahalanobis_distance: Non-negative Mahalanobis Distance.

        Returns:
            Health Index in the range ``(0, 1]``.

        Raises:
            ValueError: If ``mahalanobis_distance`` is negative.
        """
        if mahalanobis_distance < 0:
            raise ValueError("mahalanobis_distance must be non-negative.")
        hi = 1.0 - (2.0 / math.pi) * math.atan(self.b * mahalanobis_distance)
        return hi

    def calculate_ahi_score(self, mahalanobis_distance: float) -> float:
        """Calculate the Asset Health Index (AHI) score for a single MD value.

        Args:
            mahalanobis_distance: Non-negative Mahalanobis Distance.

        Returns:
            AHI score on a 0-100 scale.
        """
        return self.calculate_health_index(mahalanobis_distance) * 100.0

    def calculate_batch(
        self, mahalanobis_distances: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized HI/AHI calculation for a batch of MD values.

        Args:
            mahalanobis_distances: Iterable of non-negative MD values.

        Returns:
            A structured array-like tuple of ``(health_index, ahi_score)``
            arrays, each of shape ``(n_samples,)``.
        """
        md_array = np.asarray(list(mahalanobis_distances), dtype=float)
        if np.any(md_array < 0):
            raise ValueError("All mahalanobis_distances must be non-negative.")
        hi = 1.0 - (2.0 / math.pi) * np.arctan(self.b * md_array)
        ahi = hi * 100.0
        return hi, ahi
