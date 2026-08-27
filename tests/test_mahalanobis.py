"""Unit tests for the Mahalanobis Distance health model (src/ml/mahalanobis.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.mahalanobis import MahalanobisHealthModel


def test_model_creation_is_not_fitted() -> None:
    model = MahalanobisHealthModel()
    assert model.is_fitted is False


def test_calculate_distance_before_fit_raises() -> None:
    model = MahalanobisHealthModel()
    with pytest.raises(RuntimeError):
        model.calculate_distance(np.zeros(7))


def test_fit_healthy_baseline_requires_at_least_two_samples() -> None:
    model = MahalanobisHealthModel()
    with pytest.raises(ValueError):
        model.fit_healthy_baseline(np.zeros((1, 7)))


def test_fit_and_distance_of_mean_is_zero() -> None:
    rng = np.random.default_rng(0)
    X_healthy = rng.normal(loc=0.0, scale=1.0, size=(200, 7))
    model = MahalanobisHealthModel()
    model.fit_healthy_baseline(X_healthy)

    assert model.is_fitted is True
    mean_vector = model.mean_vector
    distance_at_mean = model.calculate_distance(mean_vector)
    assert distance_at_mean == pytest.approx(0.0, abs=1e-6)


def test_distance_increases_further_from_baseline() -> None:
    rng = np.random.default_rng(1)
    X_healthy = rng.normal(loc=0.0, scale=1.0, size=(200, 7))
    model = MahalanobisHealthModel()
    model.fit_healthy_baseline(X_healthy)

    mean_vector = model.mean_vector
    near_point = mean_vector + 0.1
    far_point = mean_vector + 5.0

    d_near = model.calculate_distance(near_point)
    d_far = model.calculate_distance(far_point)
    assert d_far > d_near


def test_calculate_batch_distance_matches_single(tmp_path) -> None:
    rng = np.random.default_rng(2)
    X_healthy = rng.normal(size=(100, 7))
    model = MahalanobisHealthModel()
    model.fit_healthy_baseline(X_healthy)

    batch = rng.normal(size=(5, 7))
    batch_distances = model.calculate_batch_distance(batch)
    single_distances = np.array([model.calculate_distance(row) for row in batch])
    np.testing.assert_allclose(batch_distances, single_distances)


def test_save_and_load_baseline_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(3)
    X_healthy = rng.normal(size=(100, 7))
    model = MahalanobisHealthModel()
    model.fit_healthy_baseline(X_healthy)

    path = tmp_path / "baseline.joblib"
    model.save_baseline(path)
    assert path.exists()

    loaded_model = MahalanobisHealthModel().load_baseline(path)
    assert loaded_model.is_fitted is True
    np.testing.assert_allclose(loaded_model.mean_vector, model.mean_vector)
