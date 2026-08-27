"""Unit tests for the BP Neural Network model class (src/ml/bp_network.py).

No real training data is used yet; these tests verify class construction,
configuration wiring, and error handling for calling inference methods
before training.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.bp_network import BPNeuralNetworkModel


def test_model_initializes_with_configured_classes() -> None:
    model = BPNeuralNetworkModel()
    assert model.classes == ["Healthy", "Sub-Healthy", "Abnormal", "Fault"]
    assert model.is_trained is False


def test_predict_before_training_raises() -> None:
    model = BPNeuralNetworkModel()
    X = np.zeros((2, 7))
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_predict_proba_before_training_raises() -> None:
    model = BPNeuralNetworkModel()
    X = np.zeros((2, 7))
    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_evaluate_before_training_raises() -> None:
    model = BPNeuralNetworkModel()
    X = np.zeros((2, 7))
    y = np.array(["Healthy", "Fault"])
    with pytest.raises(RuntimeError):
        model.evaluate(X, y)


def test_save_model_before_training_raises(tmp_path) -> None:
    model = BPNeuralNetworkModel()
    with pytest.raises(RuntimeError):
        model.save_model(tmp_path / "model.joblib")


def test_load_model_missing_file_raises(tmp_path) -> None:
    model = BPNeuralNetworkModel()
    with pytest.raises(FileNotFoundError):
        model.load_model(tmp_path / "does_not_exist.joblib")


def test_train_predict_save_load_round_trip(tmp_path) -> None:
    """Smoke test the full lifecycle using small synthetic data."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 7))
    y = np.array(["Healthy", "Sub-Healthy", "Abnormal", "Fault"] * 10)

    model = BPNeuralNetworkModel()
    model.train(X, y)
    assert model.is_trained is True

    predictions = model.predict(X)
    assert predictions.shape == (40,)

    probabilities = model.predict_proba(X)
    assert probabilities.shape[0] == 40

    evaluation = model.evaluate(X, y)
    assert "accuracy" in evaluation

    model_path = tmp_path / "bp_model.joblib"
    model.save_model(model_path)
    assert model_path.exists()

    loaded_model = BPNeuralNetworkModel().load_model(model_path)
    assert loaded_model.is_trained is True
    assert loaded_model.predict(X).shape == (40,)
