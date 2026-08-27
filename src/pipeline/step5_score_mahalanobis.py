"""Phase 2 - Step 5: Mahalanobis Distance scoring.

Scores new (unlabeled) operational measurements against the healthy
baseline built in Step 3, using the previously fitted scaler for
standardization.

    MD = sqrt((X - mu)^T * Sigma^-1 * (X - mu))
"""

from __future__ import annotations

import pandas as pd

from src.ml.mahalanobis import MahalanobisHealthModel
from src.pipeline.common import (
    COVARIANCE_ARTIFACT_PATH,
    HEALTHY_MEAN_ARTIFACT_PATH,
    INV_COVARIANCE_ARTIFACT_PATH,
    MD_SCORES_PATH,
    NEW_MEASUREMENTS_PATH,
    SCALER_ARTIFACT_PATH,
)
from src.preprocessing.preprocessor import Preprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_baseline_model() -> MahalanobisHealthModel:
    """Reconstruct a fitted MahalanobisHealthModel from the saved artifacts."""
    import joblib

    model = MahalanobisHealthModel()
    model._mean = joblib.load(HEALTHY_MEAN_ARTIFACT_PATH)
    model._covariance = joblib.load(COVARIANCE_ARTIFACT_PATH)
    model._inv_covariance = joblib.load(INV_COVARIANCE_ARTIFACT_PATH)
    model._is_fitted = True
    return model


def run_mahalanobis_scoring() -> pd.DataFrame:
    """Score ``new_measurements.csv`` and write ``md_scores.csv``.

    Returns:
        DataFrame with columns ``asset_id``, ``timestamp``,
        ``mahalanobis_distance``.
    """
    new_df = pd.read_csv(NEW_MEASUREMENTS_PATH, parse_dates=["timestamp"])
    logger.info("Loaded %d new measurement records for scoring.", len(new_df))

    preprocessor = Preprocessor().load_artifacts(SCALER_ARTIFACT_PATH)
    X_new_scaled = preprocessor.transform(new_df)

    baseline_model = _load_baseline_model()
    distances = baseline_model.calculate_batch_distance(X_new_scaled)

    result_df = pd.DataFrame(
        {
            "asset_id": new_df["asset_id"],
            "timestamp": new_df["timestamp"],
            "mahalanobis_distance": distances,
        }
    )
    MD_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(MD_SCORES_PATH, index=False)
    logger.info("Wrote %d Mahalanobis Distance scores to '%s'.", len(result_df), MD_SCORES_PATH)

    return result_df


if __name__ == "__main__":
    run_mahalanobis_scoring()
