"""Phase 2 - Step 8: Health classification and maintenance recommendation.

Combines the AHI scores (Step 6) and BP predictions (Step 7) into complete
health assessment records, classifies each AHI score into a health status,
and attaches a maintenance recommendation.
"""

from __future__ import annotations

import uuid

import pandas as pd

from src.pipeline.common import (
    AHI_SCORES_PATH,
    BP_PREDICTIONS_PATH,
    HEALTH_ASSESSMENTS_PATH,
)
from src.recommendations.recommendation_engine import RecommendationEngine
from src.rules.health_classification import HealthClassificationEngine
from src.utils.constants import DEFAULT_MODEL_VERSION
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_VERSION = "phase2-v1.0.0"


def run_classification_and_recommendation(
    ahi_scores_df: pd.DataFrame,
    bp_predictions_df: pd.DataFrame,
    model_version: str = MODEL_VERSION,
) -> pd.DataFrame:
    """Classify AHI scores and attach recommendations for each assessment.

    Args:
        ahi_scores_df: Output of Step 6 (``asset_id``, ``timestamp``,
            ``mahalanobis_distance``, ``health_index``, ``ahi_score``).
        bp_predictions_df: Output of Step 7 (``asset_id``, ``timestamp``,
            ``bp_prediction``, per-class probabilities).
        model_version: Version identifier recorded on every assessment.

    Returns:
        DataFrame matching the ``HealthAssessment`` domain model, one row
        per (asset_id, timestamp) scored record.
    """
    classification_engine = HealthClassificationEngine()
    recommendation_engine = RecommendationEngine()

    merged = ahi_scores_df.merge(
        bp_predictions_df, on=["asset_id", "timestamp"], how="left"
    )

    merged["health_status"] = merged["ahi_score"].apply(classification_engine.classify)
    merged["recommendation"] = merged["health_status"].apply(
        recommendation_engine.recommend
    )
    merged["model_version"] = model_version or DEFAULT_MODEL_VERSION
    merged["assessment_id"] = [str(uuid.uuid4()) for _ in range(len(merged))]

    column_order = [
        "assessment_id",
        "asset_id",
        "timestamp",
        "bp_prediction",
        "healthy_probability",
        "subhealthy_probability",
        "abnormal_probability",
        "fault_probability",
        "mahalanobis_distance",
        "health_index",
        "ahi_score",
        "health_status",
        "recommendation",
        "model_version",
    ]
    merged = merged[[c for c in column_order if c in merged.columns]]

    HEALTH_ASSESSMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(HEALTH_ASSESSMENTS_PATH, index=False)
    logger.info(
        "Wrote %d health assessment records to '%s'.",
        len(merged),
        HEALTH_ASSESSMENTS_PATH,
    )
    logger.info(
        "Health status distribution: %s",
        merged["health_status"].value_counts().to_dict(),
    )

    return merged


if __name__ == "__main__":
    from src.pipeline.step6_calculate_health_index import run_health_index_calculation
    from src.pipeline.step5_score_mahalanobis import run_mahalanobis_scoring
    from src.pipeline.step7_predict_bp_classes import run_bp_predictions

    _md_scores_df = run_mahalanobis_scoring()
    _ahi_scores_df = run_health_index_calculation(_md_scores_df)
    _bp_predictions_df = run_bp_predictions()
    run_classification_and_recommendation(_ahi_scores_df, _bp_predictions_df)
