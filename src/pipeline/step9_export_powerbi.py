"""Phase 2 - Step 9: Power BI star-schema dataset export.

Converts the final health assessment records, asset master data, and model
metadata into the Power BI star schema (fact + dimension tables) and writes
them as CSV files under ``data/exports/``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.exports.powerbi_export import PowerBIExporter
from src.models.asset import Asset
from src.models.health_assessment import HealthAssessment
from src.models.model_metadata import ModelMetadata
from src.pipeline.common import DIM_ASSET_PATH, EXPORTS_DIR, MODEL_METRICS_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_assets() -> list[Asset]:
    df = pd.read_csv(DIM_ASSET_PATH)
    return [Asset(**row) for row in df.to_dict(orient="records")]


def _load_health_assessments(assessments_df: pd.DataFrame) -> list[HealthAssessment]:
    assessments: list[HealthAssessment] = []
    for row in assessments_df.to_dict(orient="records"):
        probabilities = {
            "Healthy": row.get("healthy_probability"),
            "Sub-Healthy": row.get("subhealthy_probability"),
            "Abnormal": row.get("abnormal_probability"),
            "Fault": row.get("fault_probability"),
        }
        assessments.append(
            HealthAssessment(
                assessment_id=row["assessment_id"],
                asset_id=row["asset_id"],
                timestamp=row["timestamp"],
                bp_prediction=row["bp_prediction"],
                bp_class_probabilities=probabilities,
                mahalanobis_distance=row["mahalanobis_distance"],
                health_index=row["health_index"],
                ahi_score=row["ahi_score"],
                health_status=row["health_status"],
                recommendation=row["recommendation"],
                model_version=row["model_version"],
            )
        )
    return assessments


def _load_model_metadata(model_version: str) -> list[ModelMetadata]:
    import json

    accuracy = 0.0
    if MODEL_METRICS_PATH.exists():
        with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        accuracy = float(metrics.get("accuracy", 0.0))

    return [
        ModelMetadata(
            model_version=model_version,
            training_date=date.today(),
            parameters={},
            model_accuracy=accuracy,
            notes="BP Neural Network + Mahalanobis Distance (Phase 2 pipeline).",
        )
    ]


def run_powerbi_export(
    assessments_df: pd.DataFrame, model_version: str
) -> dict:
    """Build and write all Power BI star-schema CSV exports.

    Args:
        assessments_df: Output of Step 8 (final health assessment records).
        model_version: Model version identifier used across assessments.

    Returns:
        Mapping of logical dataset name to the written file path.
    """
    exporter = PowerBIExporter()

    assets = _load_assets()
    assessments = _load_health_assessments(assessments_df)
    model_metadata = _load_model_metadata(model_version)

    written_paths = exporter.export_all(
        assessments=assessments,
        assets=assets,
        model_metadata=model_metadata,
        output_dir=EXPORTS_DIR,
    )
    logger.info("Power BI exports written: %s", {k: str(v) for k, v in written_paths.items()})
    return written_paths


if __name__ == "__main__":
    from src.pipeline.step8_classify_and_recommend import (
        MODEL_VERSION,
        run_classification_and_recommendation,
    )
    from src.pipeline.step6_calculate_health_index import run_health_index_calculation
    from src.pipeline.step5_score_mahalanobis import run_mahalanobis_scoring
    from src.pipeline.step7_predict_bp_classes import run_bp_predictions

    _md_scores_df = run_mahalanobis_scoring()
    _ahi_scores_df = run_health_index_calculation(_md_scores_df)
    _bp_predictions_df = run_bp_predictions()
    _assessments_df = run_classification_and_recommendation(
        _ahi_scores_df, _bp_predictions_df
    )
    run_powerbi_export(_assessments_df, MODEL_VERSION)
