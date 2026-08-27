"""ModelMetadata domain model.

Tracks metadata about a trained model version (BP Neural Network and/or
Mahalanobis baseline) for reproducibility, auditability, and Power BI
``Dim_ModelVersion`` reporting. Maps to the future ``ModelMetadata``
database table.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelMetadata(BaseModel):
    """Represents metadata for one trained model version.

    Attributes:
        model_version: Unique version identifier (e.g., "v1.0.0").
        training_date: Date the model was trained.
        parameters: Hyperparameters/configuration used for training.
        model_accuracy: Overall accuracy (or another primary metric) on
            the held-out test set.
        notes: Free-text notes about this model version.
    """

    model_config = ConfigDict(from_attributes=True)

    model_version: str = Field(..., min_length=1)
    training_date: date
    parameters: dict[str, Any] = Field(default_factory=dict)
    model_accuracy: float = Field(..., ge=0.0, le=1.0)
    notes: Optional[str] = None


class ModelMetadataCreate(BaseModel):
    """Payload model used to register a newly trained model version."""

    model_version: str = Field(..., min_length=1)
    training_date: date
    parameters: dict[str, Any] = Field(default_factory=dict)
    model_accuracy: float = Field(..., ge=0.0, le=1.0)
    notes: Optional[str] = None
