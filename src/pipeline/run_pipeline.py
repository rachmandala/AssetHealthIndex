"""Phase 2 - End-to-end AHI pipeline orchestrator.

Runs the full Han et al. (2023) methodology over the raw datasets in
``data/raw/``: validation -> preprocessing -> healthy baseline -> BP
Neural Network training -> Mahalanobis Distance scoring -> Health Index /
AHI calculation -> BP predictions -> health classification and
recommendation -> Power BI export.

Usage:
    python -m src.pipeline.run_pipeline
"""

from __future__ import annotations

from src.pipeline.step1_validate_data import run_validation
from src.pipeline.step2_preprocess_data import run_preprocessing
from src.pipeline.step3_build_healthy_baseline import run_baseline_creation
from src.pipeline.step4_train_bp_network import run_bp_training
from src.pipeline.step5_score_mahalanobis import run_mahalanobis_scoring
from src.pipeline.step6_calculate_health_index import run_health_index_calculation
from src.pipeline.step7_predict_bp_classes import run_bp_predictions
from src.pipeline.step8_classify_and_recommend import (
    MODEL_VERSION,
    run_classification_and_recommendation,
)
from src.pipeline.step9_export_powerbi import run_powerbi_export
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def run_full_pipeline() -> dict:
    """Execute all Phase 2 pipeline steps in order and return a summary."""
    setup_logging()

    logger.info("=== Step 1: Data validation ===")
    validation_report = run_validation(raise_on_error=True)

    logger.info("=== Step 2: Data preprocessing ===")
    clean_df, preprocessor = run_preprocessing()

    logger.info("=== Step 3: Healthy baseline creation ===")
    run_baseline_creation(clean_df, preprocessor)

    logger.info("=== Step 4: BP Neural Network training ===")
    _, metrics = run_bp_training(clean_df, preprocessor)

    logger.info("=== Step 5: Mahalanobis Distance scoring ===")
    md_scores_df = run_mahalanobis_scoring()

    logger.info("=== Step 6: Health Index / AHI calculation ===")
    ahi_scores_df = run_health_index_calculation(md_scores_df)

    logger.info("=== Step 7: BP predictions on new measurements ===")
    bp_predictions_df = run_bp_predictions()

    logger.info("=== Step 8: Health classification and recommendation ===")
    assessments_df = run_classification_and_recommendation(
        ahi_scores_df, bp_predictions_df, model_version=MODEL_VERSION
    )

    logger.info("=== Step 9: Power BI dataset export ===")
    exported_paths = run_powerbi_export(assessments_df, MODEL_VERSION)

    logger.info("=== Phase 2 pipeline complete ===")
    return {
        "validation_report": validation_report,
        "model_metrics": metrics,
        "assessments": assessments_df,
        "exported_paths": exported_paths,
    }


if __name__ == "__main__":
    run_full_pipeline()
