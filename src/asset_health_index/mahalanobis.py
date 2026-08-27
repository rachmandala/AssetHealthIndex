from __future__ import annotations

import numpy as np

from .baseline import HealthyBaseline


def mahalanobis_distance_batch(x: np.ndarray, baseline: HealthyBaseline) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("Input x must be 2-dimensional")
    if x.shape[1] != baseline.mean_vector.shape[0]:
        raise ValueError("Input feature dimension does not match baseline mean vector")

    delta = x - baseline.mean_vector
    md_sq = np.einsum("ij,jk,ik->i", delta, baseline.inverse_covariance_matrix, delta)
    md_sq = np.maximum(md_sq, 0.0)
    return np.sqrt(md_sq)
