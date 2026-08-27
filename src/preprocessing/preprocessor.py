"""Preprocessing pipeline for operational measurement data.

Implements missing value handling, outlier handling, and feature
normalization/standardization using ``sklearn.preprocessing.StandardScaler``.
Fitted preprocessing artifacts are persisted with ``joblib`` so the exact
same transformation can be reapplied at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config.settings import AppSettings, get_settings
from src.utils.constants import INPUT_FEATURE_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Cleans and standardizes operational measurement features.

    Typical usage::

        preprocessor = Preprocessor()
        clean_df = preprocessor.handle_missing_values(df)
        clean_df = preprocessor.handle_outliers(clean_df)
        scaled = preprocessor.fit_transform(clean_df)
        preprocessor.save_artifacts("artifacts/feature_scaler.joblib")
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        feature_columns: tuple[str, ...] = INPUT_FEATURE_NAMES,
    ) -> None:
        self._settings = settings or get_settings()
        self._feature_columns = list(feature_columns)
        self._scaler: StandardScaler = StandardScaler()
        self._is_fitted: bool = False

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in feature columns.

        Strategy is controlled by ``preprocessing.missing_value_strategy``
        in configuration (``"median"``, ``"mean"``, or ``"drop"``).
        """
        strategy = self._settings.preprocessing.missing_value_strategy
        result = df.copy()
        present_columns = [c for c in self._feature_columns if c in result.columns]

        if strategy == "drop":
            before = len(result)
            result = result.dropna(subset=present_columns)
            logger.info(
                "Dropped %d rows with missing values (strategy=drop).",
                before - len(result),
            )
            return result

        for col in present_columns:
            if result[col].isna().any():
                if strategy == "mean":
                    fill_value = result[col].mean()
                else:  # default: median
                    fill_value = result[col].median()
                n_missing = int(result[col].isna().sum())
                result[col] = result[col].fillna(fill_value)
                logger.info(
                    "Imputed %d missing value(s) in '%s' using %s=%.4f.",
                    n_missing,
                    col,
                    strategy,
                    fill_value,
                )
        return result

    # ------------------------------------------------------------------
    # Outlier handling
    # ------------------------------------------------------------------
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers in feature columns using the configured method.

        Currently supports the IQR (interquartile range) method, which
        clips values outside ``[Q1 - k*IQR, Q3 + k*IQR]`` to the nearest
        bound, where ``k`` is ``preprocessing.outlier_iqr_multiplier``.
        """
        method = self._settings.preprocessing.outlier_method
        result = df.copy()
        present_columns = [c for c in self._feature_columns if c in result.columns]

        if method != "iqr":
            logger.warning("Unsupported outlier method '%s'; skipping.", method)
            return result

        multiplier = self._settings.preprocessing.outlier_iqr_multiplier
        for col in present_columns:
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            n_outliers = int(
                ((result[col] < lower_bound) | (result[col] > upper_bound)).sum()
            )
            if n_outliers:
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(
                    "Clipped %d outlier(s) in '%s' to range [%.4f, %.4f].",
                    n_outliers,
                    col,
                    lower_bound,
                    upper_bound,
                )
        return result

    # ------------------------------------------------------------------
    # Standardization / normalization
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Fit the internal :class:`StandardScaler` on the given features."""
        matrix = self._to_feature_matrix(df)
        self._scaler.fit(matrix)
        self._is_fitted = True
        logger.info("Fitted StandardScaler on %d records.", len(df))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Standardize features using the previously fitted scaler."""
        if not self._is_fitted:
            raise RuntimeError(
                "Preprocessor.transform() called before fit(). "
                "Call fit() or fit_transform() first, or load_artifacts()."
            )
        matrix = self._to_feature_matrix(df)
        return self._scaler.transform(matrix)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit the scaler and transform in one step."""
        return self.fit(df).transform(df)

    def _to_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")
        return df[self._feature_columns].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------
    def save_artifacts(self, path: str | Path) -> None:
        """Persist the fitted scaler (and feature column order) to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted Preprocessor.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"scaler": self._scaler, "feature_columns": self._feature_columns}, path
        )
        logger.info("Saved preprocessing artifacts to '%s'.", path)

    def load_artifacts(self, path: str | Path) -> "Preprocessor":
        """Load a previously fitted scaler (and feature column order)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessing artifact not found: {path}")
        payload = joblib.load(path)
        self._scaler = payload["scaler"]
        self._feature_columns = payload["feature_columns"]
        self._is_fitted = True
        logger.info("Loaded preprocessing artifacts from '%s'.", path)
        return self
