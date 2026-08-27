"""Health classification engine.

Classifies an Asset Health Index (AHI) score into a business-friendly
health status using configurable thresholds.

Default thresholds (upper-exclusive on the lower band, inclusive on the
upper band):

    AHI > 90            -> Healthy
    60 < AHI <= 90       -> Sub-Healthy
    30 < AHI <= 60       -> Abnormal
    AHI <= 30            -> Fault
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config.settings import AppSettings, ThresholdSettings, get_settings
from src.utils.constants import (
    HEALTH_STATUS_ABNORMAL,
    HEALTH_STATUS_FAULT,
    HEALTH_STATUS_HEALTHY,
    HEALTH_STATUS_SUB_HEALTHY,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassificationThresholds:
    """Immutable threshold configuration for health classification."""

    healthy: float = 90.0
    subhealthy: float = 60.0
    abnormal: float = 30.0

    def __post_init__(self) -> None:
        if not (self.healthy > self.subhealthy > self.abnormal):
            raise ValueError(
                "Thresholds must satisfy healthy > subhealthy > abnormal "
                f"(got healthy={self.healthy}, subhealthy={self.subhealthy}, "
                f"abnormal={self.abnormal})."
            )

    @classmethod
    def from_settings(cls, thresholds: ThresholdSettings) -> "ClassificationThresholds":
        return cls(
            healthy=thresholds.healthy,
            subhealthy=thresholds.subhealthy,
            abnormal=thresholds.abnormal,
        )


class HealthClassificationEngine:
    """Classifies AHI scores into health status bands."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        thresholds: Optional[ClassificationThresholds] = None,
    ) -> None:
        """Initialize the classification engine.

        Args:
            settings: Application settings; defaults to :func:`get_settings`.
            thresholds: Optional explicit thresholds, overriding configuration.
        """
        self._settings = settings or get_settings()
        self.thresholds = thresholds or ClassificationThresholds.from_settings(
            self._settings.thresholds
        )

    def classify(self, ahi_score: float) -> str:
        """Classify a single AHI score into a health status label.

        Args:
            ahi_score: Asset Health Index score, expected in ``[0, 100]``.

        Returns:
            One of ``"Healthy"``, ``"Sub-Healthy"``, ``"Abnormal"``, ``"Fault"``.
        """
        if ahi_score > self.thresholds.healthy:
            status = HEALTH_STATUS_HEALTHY
        elif ahi_score > self.thresholds.subhealthy:
            status = HEALTH_STATUS_SUB_HEALTHY
        elif ahi_score > self.thresholds.abnormal:
            status = HEALTH_STATUS_ABNORMAL
        else:
            status = HEALTH_STATUS_FAULT

        logger.debug("Classified AHI score %.2f as '%s'.", ahi_score, status)
        return status

    def classify_batch(self, ahi_scores: list[float]) -> list[str]:
        """Classify a batch of AHI scores.

        Args:
            ahi_scores: List of AHI scores.

        Returns:
            List of health status labels, in the same order as input.
        """
        return [self.classify(score) for score in ahi_scores]
