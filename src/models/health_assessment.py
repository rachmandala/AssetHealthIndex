"""HealthAssessment domain model.

Represents the outcome of running the AHI pipeline (BP Neural Network +
Mahalanobis Distance + Health Index) against one operational measurement.
Maps to the future ``HealthAssessment`` database table and is the primary
source record for ``Fact_AssetHealthAssessment`` in the Power BI star
schema.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BPPredictionClass(str, Enum):
    """BP Neural Network predicted condition class."""

    HEALTHY = "Healthy"
    SUB_HEALTHY = "Sub-Healthy"
    ABNORMAL = "Abnormal"
    FAULT = "Fault"


class HealthStatus(str, Enum):
    """Final health status derived from the AHI score."""

    HEALTHY = "Healthy"
    SUB_HEALTHY = "Sub-Healthy"
    ABNORMAL = "Abnormal"
    FAULT = "Fault"


class HealthAssessment(BaseModel):
    """Represents one point-in-time health assessment for an asset.

    Attributes:
        assessment_id: Unique identifier of the assessment record.
        asset_id: Foreign key reference to the assessed :class:`Asset`.
        timestamp: UTC timestamp the assessment corresponds to.
        bp_prediction: BP Neural Network predicted condition class.
        bp_class_probabilities: Optional per-class predicted probabilities.
        mahalanobis_distance: Mahalanobis Distance (MD) from healthy baseline.
        health_index: Health Index (HI), a value between 0 and 1.
        ahi_score: Asset Health Index (AHI), ``HI * 100`` (0-100 scale).
        health_status: Classified health status derived from ``ahi_score``.
        recommendation: Maintenance recommendation text for this status.
        model_version: Version identifier of the model(s) used to produce
            this assessment.
    """

    model_config = ConfigDict(from_attributes=True)

    assessment_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1)
    timestamp: datetime

    bp_prediction: BPPredictionClass
    bp_class_probabilities: Optional[dict[str, float]] = Field(
        default=None,
        description="Predicted probability per BP condition class, if available.",
    )

    mahalanobis_distance: float = Field(..., ge=0.0)
    health_index: float = Field(..., ge=0.0, le=1.0)
    ahi_score: float = Field(..., ge=0.0, le=100.0)

    health_status: HealthStatus
    recommendation: str = Field(..., min_length=1)
    model_version: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_ahi_consistency(self) -> "HealthAssessment":
        """Ensure ``ahi_score`` is consistent with ``health_index`` (HI * 100)."""
        expected = self.health_index * 100.0
        if abs(expected - self.ahi_score) > 1e-6:
            raise ValueError(
                f"ahi_score ({self.ahi_score}) must equal health_index * 100 "
                f"({expected})."
            )
        return self


class HealthAssessmentCreate(BaseModel):
    """Payload model used to persist a newly computed health assessment."""

    assessment_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1)
    timestamp: datetime
    bp_prediction: BPPredictionClass
    bp_class_probabilities: Optional[dict[str, float]] = None
    mahalanobis_distance: float = Field(..., ge=0.0)
    health_index: float = Field(..., ge=0.0, le=1.0)
    ahi_score: float = Field(..., ge=0.0, le=100.0)
    health_status: HealthStatus
    recommendation: str = Field(..., min_length=1)
    model_version: str = Field(..., min_length=1)
