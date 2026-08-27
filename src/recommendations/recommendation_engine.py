"""Maintenance recommendation engine.

Translates a health status label into an actionable maintenance
recommendation, per the business rules defined by Han et al. (2023) and
plant reliability engineering practice.

    Healthy      -> Continue Normal Operation
    Sub-Healthy  -> Increase Monitoring Frequency
    Abnormal     -> Schedule Inspection
    Fault        -> Immediate Maintenance Assessment
"""

from __future__ import annotations

from src.utils.constants import RECOMMENDATION_BY_HEALTH_STATUS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationEngine:
    """Maps health status labels to maintenance recommendations."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        """Initialize the engine.

        Args:
            mapping: Optional override mapping of health status -> recommendation
                text. Defaults to ``RECOMMENDATION_BY_HEALTH_STATUS``.
        """
        self._mapping = mapping or dict(RECOMMENDATION_BY_HEALTH_STATUS)

    def recommend(self, health_status: str) -> str:
        """Return the recommendation text for a given health status.

        Args:
            health_status: One of "Healthy", "Sub-Healthy", "Abnormal", "Fault".

        Returns:
            The recommendation text associated with the status.

        Raises:
            ValueError: If ``health_status`` is not a recognized status.
        """
        try:
            recommendation = self._mapping[health_status]
        except KeyError as exc:
            raise ValueError(
                f"Unrecognized health status '{health_status}'. "
                f"Expected one of {list(self._mapping.keys())}."
            ) from exc

        logger.debug(
            "Recommendation for status '%s': '%s'.", health_status, recommendation
        )
        return recommendation

    def recommend_batch(self, health_statuses: list[str]) -> list[str]:
        """Return recommendations for a batch of health statuses."""
        return [self.recommend(status) for status in health_statuses]
