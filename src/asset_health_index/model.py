from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class BPNeuralNetwork:
    def __init__(self, model_cfg: dict[str, Any], random_state: int = 42) -> None:
        self.model_cfg = model_cfg
        self.random_state = random_state
        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(model_cfg.get("hidden_layer_sizes", [32, 16])),
            activation=model_cfg.get("activation", "relu"),
            solver=model_cfg.get("solver", "adam"),
            alpha=model_cfg.get("alpha", 0.0001),
            learning_rate=model_cfg.get("learning_rate", "adaptive"),
            max_iter=model_cfg.get("max_iter", 500),
            random_state=random_state,
        )

    def train(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        test_size = float(self.model_cfg.get("test_size", 0.4))
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state,
        )
        self.model.fit(x_train, y_train)
        pred = self.model.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=self.model.classes_).tolist(),
            "labels": self.model.classes_.tolist(),
            "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True),
        }
        return metrics

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str | Path) -> "BPNeuralNetwork":
        obj = BPNeuralNetwork({})
        obj.model = joblib.load(path)
        return obj
