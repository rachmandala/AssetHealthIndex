from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


@dataclass
class HealthyBaseline:
    mean_vector: np.ndarray
    covariance_matrix: np.ndarray
    inverse_covariance_matrix: np.ndarray

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "HealthyBaseline":
        return joblib.load(path)


def build_healthy_baseline(x_healthy: np.ndarray, regularization: float = 1e-6) -> HealthyBaseline:
    if x_healthy.size == 0:
        raise ValueError("No healthy records available to build baseline")
    mean_vector = np.mean(x_healthy, axis=0)
    covariance = np.cov(x_healthy, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.array([[float(covariance)]])
    covariance = covariance + np.eye(covariance.shape[0]) * regularization
    inv_covariance = np.linalg.pinv(covariance)
    return HealthyBaseline(mean_vector=mean_vector, covariance_matrix=covariance, inverse_covariance_matrix=inv_covariance)
