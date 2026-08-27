"""Abstract repository interfaces for data access.

These interfaces decouple the rest of the application from the concrete
data source. Today only mock (in-memory) repositories exist; future
phases will add SQL/SQLAlchemy-backed implementations without requiring
changes to calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from src.models.asset import Asset
from src.models.health_assessment import HealthAssessment
from src.models.model_metadata import ModelMetadata
from src.models.operational_measurement import OperationalMeasurement


class AssetRepository(ABC):
    """Repository interface for :class:`Asset` records."""

    @abstractmethod
    def get_by_id(self, asset_id: str) -> Optional[Asset]:
        """Return the asset with the given ID, or ``None`` if not found."""

    @abstractmethod
    def list_all(self) -> list[Asset]:
        """Return all known assets."""

    @abstractmethod
    def add(self, asset: Asset) -> Asset:
        """Persist a new asset record and return it."""

    @abstractmethod
    def update(self, asset: Asset) -> Asset:
        """Persist changes to an existing asset record and return it."""

    @abstractmethod
    def delete(self, asset_id: str) -> bool:
        """Delete an asset record. Returns ``True`` if a record was removed."""


class OperationalMeasurementRepository(ABC):
    """Repository interface for :class:`OperationalMeasurement` records."""

    @abstractmethod
    def get_by_id(self, measurement_id: str) -> Optional[OperationalMeasurement]:
        """Return the measurement with the given ID, or ``None`` if not found."""

    @abstractmethod
    def list_by_asset(
        self,
        asset_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[OperationalMeasurement]:
        """Return measurements for an asset, optionally filtered by time range."""

    @abstractmethod
    def add(self, measurement: OperationalMeasurement) -> OperationalMeasurement:
        """Persist a new measurement record and return it."""

    @abstractmethod
    def add_batch(
        self, measurements: list[OperationalMeasurement]
    ) -> list[OperationalMeasurement]:
        """Persist multiple measurement records and return them."""


class HealthAssessmentRepository(ABC):
    """Repository interface for :class:`HealthAssessment` records."""

    @abstractmethod
    def get_by_id(self, assessment_id: str) -> Optional[HealthAssessment]:
        """Return the assessment with the given ID, or ``None`` if not found."""

    @abstractmethod
    def list_by_asset(
        self,
        asset_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[HealthAssessment]:
        """Return assessments for an asset, optionally filtered by time range."""

    @abstractmethod
    def list_latest_per_asset(self) -> list[HealthAssessment]:
        """Return the most recent assessment for every asset."""

    @abstractmethod
    def add(self, assessment: HealthAssessment) -> HealthAssessment:
        """Persist a new assessment record and return it."""

    @abstractmethod
    def add_batch(self, assessments: list[HealthAssessment]) -> list[HealthAssessment]:
        """Persist multiple assessment records and return them."""


class ModelMetadataRepository(ABC):
    """Repository interface for :class:`ModelMetadata` records."""

    @abstractmethod
    def get_by_version(self, model_version: str) -> Optional[ModelMetadata]:
        """Return metadata for the given model version, or ``None``."""

    @abstractmethod
    def list_all(self) -> list[ModelMetadata]:
        """Return metadata for all registered model versions."""

    @abstractmethod
    def get_latest(self) -> Optional[ModelMetadata]:
        """Return metadata for the most recently trained model version."""

    @abstractmethod
    def add(self, metadata: ModelMetadata) -> ModelMetadata:
        """Persist a new model metadata record and return it."""
