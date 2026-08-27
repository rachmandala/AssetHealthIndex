"""BP Neural Network model for asset condition classification.

Wraps ``sklearn.neural_network.MLPClassifier`` to classify operational
measurements into one of four condition classes: Healthy, Sub-Healthy,
Abnormal, Fault, per the Han et al. (2023) methodology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from src.config.settings import AppSettings, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BPNeuralNetworkModel:
    """BP (Back Propagation) Neural Network condition classifier.

    This class is a thin, production-oriented wrapper around
    :class:`sklearn.neural_network.MLPClassifier` that standardizes how
    the AHI platform trains, persists, and serves the BP Neural Network
    described in Han et al. (2023).

    Attributes:
        classes: Ordered list of condition class labels the model predicts.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        classifier: Optional[MLPClassifier] = None,
    ) -> None:
        """Initialize the model.

        Args:
            settings: Application settings; defaults to :func:`get_settings`.
            classifier: Optional pre-configured ``MLPClassifier`` instance,
                primarily for testing. When omitted, a classifier is built
                from ``bp_network`` settings in configuration.
        """
        self._settings = settings or get_settings()
        self.classes: list[str] = list(self._settings.bp_network.classes)
        self._model: MLPClassifier = classifier or self._build_classifier()
        self._is_trained: bool = False

    def _build_classifier(self) -> MLPClassifier:
        cfg = self._settings.bp_network
        return MLPClassifier(
            hidden_layer_sizes=tuple(cfg.hidden_layer_sizes),
            activation=cfg.activation,
            solver=cfg.solver,
            alpha=cfg.alpha,
            learning_rate_init=cfg.learning_rate_init,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
            early_stopping=cfg.early_stopping,
        )

    @property
    def is_trained(self) -> bool:
        """Whether :meth:`train` (or :meth:`load_model`) has been called."""
        return self._is_trained

    def train(self, X: np.ndarray, y: np.ndarray) -> "BPNeuralNetworkModel":
        """Train the BP Neural Network on standardized feature data.

        Args:
            X: Standardized feature matrix of shape ``(n_samples, n_features)``.
            y: Target condition class labels of shape ``(n_samples,)``.

        Returns:
            ``self``, to allow method chaining.
        """
        logger.info(
            "Training BP Neural Network on %d samples, %d features.",
            X.shape[0],
            X.shape[1],
        )
        self._model.fit(X, y)
        self._is_trained = True
        logger.info("BP Neural Network training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most likely condition class for each sample.

        Args:
            X: Standardized feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of predicted class labels, shape ``(n_samples,)``.
        """
        self._ensure_trained()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for each sample.

        Args:
            X: Standardized feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, n_classes)`` with per-class
            probabilities in the order given by ``self._model.classes_``.
        """
        self._ensure_trained()
        return self._model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
        """Evaluate the trained model against labeled test data.

        Args:
            X: Standardized feature matrix for the evaluation set.
            y_true: Ground-truth condition class labels.

        Returns:
            A dictionary containing ``accuracy``, a per-class
            ``classification_report`` (as a dict), and the ``confusion_matrix``.
        """
        self._ensure_trained()
        y_pred = self._model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred, labels=self._model.classes_)

        logger.info("Evaluation accuracy: %.4f", accuracy)
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "labels": list(self._model.classes_),
        }

    def save_model(self, path: str | Path) -> None:
        """Persist the trained model to disk using ``joblib``.

        Args:
            path: Destination file path (e.g., ``artifacts/bp_model_v1.joblib``).
        """
        self._ensure_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "classes": self.classes}, path)
        logger.info("Saved BP Neural Network model to '%s'.", path)

    def load_model(self, path: str | Path) -> "BPNeuralNetworkModel":
        """Load a previously trained model from disk.

        Args:
            path: Source file path produced by :meth:`save_model`.

        Returns:
            ``self``, to allow method chaining.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BP Neural Network model file not found: {path}")
        payload = joblib.load(path)
        self._model = payload["model"]
        self.classes = payload["classes"]
        self._is_trained = True
        logger.info("Loaded BP Neural Network model from '%s'.", path)
        return self

    def _ensure_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "BPNeuralNetworkModel has not been trained or loaded. "
                "Call train() or load_model() first."
            )
