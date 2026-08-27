"""Mahalanobis Distance health model.

Implements the Mahalanobis Distance (MD) methodology from Han et al.
(2023) for measuring how far a current operational observation deviates
from a "healthy" baseline distribution.

Formula
-------
    MD = sqrt((X - mu)^T * Sigma^-1 * (X - mu))

Where:
    X     = current observation (feature vector)
    mu    = healthy baseline mean vector
    Sigma = healthy baseline covariance matrix

A small regularization term (``covariance_regularization``) is added to
the diagonal of the covariance matrix before inversion to keep it
well-conditioned/invertible, which is standard practice when the healthy
baseline sample size is small relative to the number of features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.config.settings import AppSettings, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MahalanobisHealthModel:
    """Computes Mahalanobis Distance against a healthy operating baseline.

    Typical usage::

        model = MahalanobisHealthModel()
        model.fit_healthy_baseline(healthy_feature_matrix)
        md = model.calculate_distance(new_observation)
        model.save_baseline("artifacts/mahalanobis_baseline.joblib")
    """

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        self._settings = settings or get_settings()
        self._regularization = self._settings.mahalanobis.covariance_regularization
        self._mean: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._inv_covariance: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def mean_vector(self) -> np.ndarray:
        self._ensure_fitted()
        assert self._mean is not None
        return self._mean.copy()

    @property
    def covariance_matrix(self) -> np.ndarray:
        self._ensure_fitted()
        assert self._covariance is not None
        return self._covariance.copy()

    @property
    def inv_covariance_matrix(self) -> np.ndarray:
        self._ensure_fitted()
        assert self._inv_covariance is not None
        return self._inv_covariance.copy()

    def fit_healthy_baseline(self, X_healthy: np.ndarray) -> "MahalanobisHealthModel":
        """Fit the healthy baseline mean vector and covariance matrix.

        Args:
            X_healthy: Standardized feature matrix of healthy observations,
                shape ``(n_samples, n_features)``. ``n_samples`` should
                comfortably exceed ``n_features`` for a stable covariance
                estimate.

        Returns:
            ``self``, to allow method chaining.
        """
        if X_healthy.ndim != 2:
            raise ValueError("X_healthy must be a 2D array of shape (n_samples, n_features).")
        if X_healthy.shape[0] < 2:
            raise ValueError("At least 2 healthy observations are required to fit a baseline.")

        self._mean = X_healthy.mean(axis=0)
        covariance = np.cov(X_healthy, rowvar=False)
        covariance = np.atleast_2d(covariance)

        # Regularize the covariance matrix to guarantee invertibility.
        n_features = covariance.shape[0]
        regularized = covariance + self._regularization * np.eye(n_features)

        self._covariance = regularized
        self._inv_covariance = np.linalg.inv(regularized)
        self._is_fitted = True

        logger.info(
            "Fitted Mahalanobis healthy baseline on %d samples, %d features "
            "(regularization=%.6f).",
            X_healthy.shape[0],
            n_features,
            self._regularization,
        )
        return self

    def calculate_distance(self, x: np.ndarray) -> float:
        """Calculate the Mahalanobis Distance for a single observation.

        Args:
            x: Feature vector of shape ``(n_features,)``, standardized the
                same way as the healthy baseline data.

        Returns:
            The scalar Mahalanobis Distance ``MD = sqrt((x-mu)^T Sigma^-1 (x-mu))``.
        """
        self._ensure_fitted()
        assert self._mean is not None and self._inv_covariance is not None

        diff = np.asarray(x, dtype=float) - self._mean
        squared_distance = float(diff.T @ self._inv_covariance @ diff)
        # Guard against tiny negative values caused by floating point error.
        squared_distance = max(squared_distance, 0.0)
        return float(np.sqrt(squared_distance))

    def calculate_batch_distance(self, X: np.ndarray) -> np.ndarray:
        """Calculate Mahalanobis Distance for a batch of observations.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of Mahalanobis Distances, shape ``(n_samples,)``.
        """
        self._ensure_fitted()
        return np.array([self.calculate_distance(row) for row in np.atleast_2d(X)])

    def save_baseline(self, path: str | Path) -> None:
        """Persist the fitted healthy baseline (mean + covariance) to disk."""
        self._ensure_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "mean": self._mean,
                "covariance": self._covariance,
                "inv_covariance": self._inv_covariance,
                "regularization": self._regularization,
            },
            path,
        )
        logger.info("Saved Mahalanobis baseline to '%s'.", path)

    def load_baseline(self, path: str | Path) -> "MahalanobisHealthModel":
        """Load a previously fitted healthy baseline from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Mahalanobis baseline file not found: {path}")
        payload = joblib.load(path)
        self._mean = payload["mean"]
        self._covariance = payload["covariance"]
        self._inv_covariance = payload["inv_covariance"]
        self._regularization = payload["regularization"]
        self._is_fitted = True
        logger.info("Loaded Mahalanobis baseline from '%s'.", path)
        return self

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "MahalanobisHealthModel has not been fitted or loaded. "
                "Call fit_healthy_baseline() or load_baseline() first."
            )
