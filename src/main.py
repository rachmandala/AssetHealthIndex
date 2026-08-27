"""Application entry point for the Asset Health Index (AHI) platform.

Wires together configuration, mock repositories, and the AHI pipeline
stages (validation -> preprocessing -> BP Neural Network ->
Mahalanobis Distance -> Health Index -> classification ->
recommendation -> Power BI export). No real data or trained models are
used yet; this module demonstrates the intended orchestration flow and
serves as the integration point for future phases.
"""

from __future__ import annotations

from src.config.settings import get_settings
from src.exports.powerbi_export import PowerBIExporter
from src.recommendations.recommendation_engine import RecommendationEngine
from src.repositories.mock_repository import (
    MockAssetRepository,
    MockHealthAssessmentRepository,
    MockModelMetadataRepository,
    MockOperationalMeasurementRepository,
)
from src.rules.health_classification import HealthClassificationEngine
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def build_application_context() -> dict:
    """Construct the core application components used by the AHI pipeline.

    Returns:
        A dictionary of initialized components: settings, repositories,
        classification engine, recommendation engine, and exporter. This
        acts as a lightweight, explicit dependency container until a
        real database and model artifacts are introduced.
    """
    settings = get_settings()

    context = {
        "settings": settings,
        "asset_repository": MockAssetRepository(),
        "measurement_repository": MockOperationalMeasurementRepository(),
        "assessment_repository": MockHealthAssessmentRepository(),
        "model_metadata_repository": MockModelMetadataRepository(),
        "classification_engine": HealthClassificationEngine(settings=settings),
        "recommendation_engine": RecommendationEngine(),
        "powerbi_exporter": PowerBIExporter(settings=settings),
    }
    return context


def main() -> None:
    """Application entry point.

    Initializes logging and the application context, then logs a summary
    of the loaded configuration. Real pipeline execution (data ingestion,
    model inference, exports) will be added once training data and
    trained model artifacts are available.
    """
    setup_logging()
    context = build_application_context()
    settings = context["settings"]

    logger.info(
        "Asset Health Index platform initialized: project='%s', version='%s', "
        "environment='%s'.",
        settings.project.name,
        settings.project.version,
        settings.project.environment,
    )
    logger.info(
        "BP Neural Network classes: %s | Health thresholds: healthy>%.0f, "
        "subhealthy>%.0f, abnormal>%.0f.",
        settings.bp_network.classes,
        settings.thresholds.healthy,
        settings.thresholds.subhealthy,
        settings.thresholds.abnormal,
    )
    logger.info(
        "No trained models or operational data are loaded yet. "
        "This phase establishes the platform foundation only."
    )


if __name__ == "__main__":
    main()
