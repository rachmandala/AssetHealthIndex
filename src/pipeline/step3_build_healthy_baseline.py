"""Phase 2 - Step 3: Healthy baseline creation.

Filters the cleaned historical dataset to ``label == "Healthy"`` records,
standardizes them with the fitted scaler, and fits the Mahalanobis healthy
baseline (mean vector, covariance matrix, inverse covariance matrix).
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.pipeline.common import (
    COVARIANCE_ARTIFACT_PATH,
    FEATURE_COLUMNS,
    HEALTHY_BASELINE_PATH,
    HEALTHY_MEAN_ARTIFACT_PATH,
    INV_COVARIANCE_ARTIFACT_PATH,
)
from src.ml.mahalanobis import MahalanobisHealthModel
from src.preprocessing.preprocessor import Preprocessor
from src.utils.constants import CLASS_HEALTHY
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_baseline_creation(
    clean_df: pd.DataFrame, preprocessor: Preprocessor
) -> MahalanobisHealthModel:
    """Build and persist the healthy operating baseline.

    Args:
        clean_df: Cleaned historical operational measurements (output of
            Step 2), including the ``label`` column.
        preprocessor: The fitted :class:`Preprocessor` from Step 2, used to
            standardize the healthy subset consistently with training/scoring.

    Returns:
        The fitted :class:`MahalanobisHealthModel`.
    """
    healthy_df = clean_df[clean_df["label"] == CLASS_HEALTHY].reset_index(drop=True)
    if healthy_df.empty:
        raise ValueError(
            f"No records with label == '{CLASS_HEALTHY}' found; cannot build "
            "a healthy baseline."
        )
    logger.info("Building healthy baseline from %d healthy records.", len(healthy_df))

    X_healthy_scaled = preprocessor.transform(healthy_df)

    baseline_model = MahalanobisHealthModel()
    baseline_model.fit_healthy_baseline(X_healthy_scaled)

    HEALTHY_MEAN_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline_model.mean_vector, HEALTHY_MEAN_ARTIFACT_PATH)
    joblib.dump(baseline_model.covariance_matrix, COVARIANCE_ARTIFACT_PATH)
    joblib.dump(baseline_model.inv_covariance_matrix, INV_COVARIANCE_ARTIFACT_PATH)
    logger.info(
        "Saved baseline artifacts: '%s', '%s', '%s'.",
        HEALTHY_MEAN_ARTIFACT_PATH,
        COVARIANCE_ARTIFACT_PATH,
        INV_COVARIANCE_ARTIFACT_PATH,
    )

    baseline_export_df = pd.DataFrame(X_healthy_scaled, columns=list(FEATURE_COLUMNS))
    baseline_export_df.insert(0, "timestamp", healthy_df["timestamp"].values)
    baseline_export_df.insert(1, "asset_id", healthy_df["asset_id"].values)
    HEALTHY_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline_export_df.to_csv(HEALTHY_BASELINE_PATH, index=False)
    logger.info(
        "Wrote %d standardized healthy baseline records to '%s'.",
        len(baseline_export_df),
        HEALTHY_BASELINE_PATH,
    )

    return baseline_model


if __name__ == "__main__":
    from src.pipeline.step2_preprocess_data import run_preprocessing

    _clean_df, _preprocessor = run_preprocessing()
    run_baseline_creation(_clean_df, _preprocessor)
