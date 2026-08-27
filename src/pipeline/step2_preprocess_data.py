"""Phase 2 - Step 2: Data preprocessing.

Cleans the historical operational measurement dataset (missing values,
outliers) and fits a ``StandardScaler`` over the seven input features,
persisting the fitted scaler for reuse during baseline creation, training,
and scoring.
"""

from __future__ import annotations

import pandas as pd

from src.pipeline.common import (
    CLEAN_OPERATIONAL_MEASUREMENTS_PATH,
    FEATURE_COLUMNS,
    OPERATIONAL_MEASUREMENTS_PATH,
    SCALER_ARTIFACT_PATH,
)
from src.preprocessing.preprocessor import Preprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_preprocessing() -> tuple[pd.DataFrame, Preprocessor]:
    """Clean and standardize the historical operational measurements.

    Returns:
        A tuple of ``(clean_dataframe, fitted_preprocessor)``. The clean
        dataframe retains original (unscaled) feature values; standardized
        features are produced on demand via ``preprocessor.transform()``.
    """
    df = pd.read_csv(OPERATIONAL_MEASUREMENTS_PATH, parse_dates=["timestamp"])
    logger.info("Loaded %d raw operational measurement records.", len(df))

    preprocessor = Preprocessor(feature_columns=FEATURE_COLUMNS)
    clean_df = preprocessor.handle_missing_values(df)
    clean_df = preprocessor.handle_outliers(clean_df)

    preprocessor.fit(clean_df)
    preprocessor.save_artifacts(SCALER_ARTIFACT_PATH)

    CLEAN_OPERATIONAL_MEASUREMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_OPERATIONAL_MEASUREMENTS_PATH, index=False)
    logger.info(
        "Wrote %d cleaned records to '%s'.",
        len(clean_df),
        CLEAN_OPERATIONAL_MEASUREMENTS_PATH,
    )

    return clean_df, preprocessor


if __name__ == "__main__":
    run_preprocessing()
