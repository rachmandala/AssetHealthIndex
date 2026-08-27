"""In-memory mock repository implementations.

These implementations satisfy the repository interfaces in
``src/repositories/interfaces.py`` using simple in-memory dictionaries.
They are intended for development, unit testing, and demos before a real
database connection is introduced.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.models.asset import Asset
from src.models.health_assessment import HealthAssessment
from src.models.model_metadata import ModelMetadata
from src.models.operational_measurement import OperationalMeasurement
from src.repositories.interfaces import (
    AssetRepository,
    HealthAssessmentRepository,
    ModelMetadataRepository,
    OperationalMeasurementRepository,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MockAssetRepository(AssetRepository):
    """In-memory implementation of :class:`AssetRepository`."""

    def __init__(self, initial_assets: Optional[list[Asset]] = None) -> None:
        self._assets: dict[str, Asset] = {
            asset.asset_id: asset for asset in (initial_assets or [])
        }

    def get_by_id(self, asset_id: str) -> Optional[Asset]:
        return self._assets.get(asset_id)

    def list_all(self) -> list[Asset]:
        return list(self._assets.values())

    def add(self, asset: Asset) -> Asset:
        if asset.asset_id in self._assets:
            raise ValueError(f"Asset with ID '{asset.asset_id}' already exists.")
        self._assets[asset.asset_id] = asset
        logger.debug("Added asset '%s' to mock repository.", asset.asset_id)
        return asset

    def update(self, asset: Asset) -> Asset:
        if asset.asset_id not in self._assets:
            raise KeyError(f"Asset with ID '{asset.asset_id}' does not exist.")
        self._assets[asset.asset_id] = asset
        logger.debug("Updated asset '%s' in mock repository.", asset.asset_id)
        return asset

    def delete(self, asset_id: str) -> bool:
        if asset_id in self._assets:
            del self._assets[asset_id]
            logger.debug("Deleted asset '%s' from mock repository.", asset_id)
            return True
        return False


class MockOperationalMeasurementRepository(OperationalMeasurementRepository):
    """In-memory implementation of :class:`OperationalMeasurementRepository`."""

    def __init__(
        self, initial_measurements: Optional[list[OperationalMeasurement]] = None
    ) -> None:
        self._measurements: dict[str, OperationalMeasurement] = {
            m.measurement_id: m for m in (initial_measurements or [])
        }

    def get_by_id(self, measurement_id: str) -> Optional[OperationalMeasurement]:
        return self._measurements.get(measurement_id)

    def list_by_asset(
        self,
        asset_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[OperationalMeasurement]:
        results = [m for m in self._measurements.values() if m.asset_id == asset_id]
        if start_time is not None:
            results = [m for m in results if m.timestamp >= start_time]
        if end_time is not None:
            results = [m for m in results if m.timestamp <= end_time]
        return sorted(results, key=lambda m: m.timestamp)

    def add(self, measurement: OperationalMeasurement) -> OperationalMeasurement:
        if measurement.measurement_id in self._measurements:
            raise ValueError(
                f"Measurement with ID '{measurement.measurement_id}' already exists."
            )
        self._measurements[measurement.measurement_id] = measurement
        return measurement

    def add_batch(
        self, measurements: list[OperationalMeasurement]
    ) -> list[OperationalMeasurement]:
        return [self.add(m) for m in measurements]


class MockHealthAssessmentRepository(HealthAssessmentRepository):
    """In-memory implementation of :class:`HealthAssessmentRepository`."""

    def __init__(
        self, initial_assessments: Optional[list[HealthAssessment]] = None
    ) -> None:
        self._assessments: dict[str, HealthAssessment] = {
            a.assessment_id: a for a in (initial_assessments or [])
        }

    def get_by_id(self, assessment_id: str) -> Optional[HealthAssessment]:
        return self._assessments.get(assessment_id)

    def list_by_asset(
        self,
        asset_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[HealthAssessment]:
        results = [a for a in self._assessments.values() if a.asset_id == asset_id]
        if start_time is not None:
            results = [a for a in results if a.timestamp >= start_time]
        if end_time is not None:
            results = [a for a in results if a.timestamp <= end_time]
        return sorted(results, key=lambda a: a.timestamp)

    def list_latest_per_asset(self) -> list[HealthAssessment]:
        latest_by_asset: dict[str, HealthAssessment] = {}
        for assessment in self._assessments.values():
            current = latest_by_asset.get(assessment.asset_id)
            if current is None or assessment.timestamp > current.timestamp:
                latest_by_asset[assessment.asset_id] = assessment
        return list(latest_by_asset.values())

    def add(self, assessment: HealthAssessment) -> HealthAssessment:
        if assessment.assessment_id in self._assessments:
            raise ValueError(
                f"Assessment with ID '{assessment.assessment_id}' already exists."
            )
        self._assessments[assessment.assessment_id] = assessment
        return assessment

    def add_batch(self, assessments: list[HealthAssessment]) -> list[HealthAssessment]:
        return [self.add(a) for a in assessments]


class MockModelMetadataRepository(ModelMetadataRepository):
    """In-memory implementation of :class:`ModelMetadataRepository`."""

    def __init__(self, initial_metadata: Optional[list[ModelMetadata]] = None) -> None:
        self._metadata: dict[str, ModelMetadata] = {
            m.model_version: m for m in (initial_metadata or [])
        }

    def get_by_version(self, model_version: str) -> Optional[ModelMetadata]:
        return self._metadata.get(model_version)

    def list_all(self) -> list[ModelMetadata]:
        return list(self._metadata.values())

    def get_latest(self) -> Optional[ModelMetadata]:
        if not self._metadata:
            return None
        return max(self._metadata.values(), key=lambda m: m.training_date)

    def add(self, metadata: ModelMetadata) -> ModelMetadata:
        if metadata.model_version in self._metadata:
            raise ValueError(
                f"Model version '{metadata.model_version}' already registered."
            )
        self._metadata[metadata.model_version] = metadata
        return metadata
