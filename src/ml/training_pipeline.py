"""Training pipeline orchestration for the BP Neural Network.

Coordinates stratified train/test splitting, model training, evaluation,
and artifact persistence. This module defines the pipeline structure and
interfaces; it is not invoked with real data until a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from src.config.settings import AppSettings, get_settings
from src.ml.bp_network import BPNeuralNetworkModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Result of running :class:`BPNetworkTrainingPipeline.run`."""

    model: BPNeuralNetworkModel
    evaluation: dict[str, Any]
    train_size: int
    test_size: int


class BPNetworkTrainingPipeline:
    """Orchestrates BP Neural Network training end-to-end.

    Steps:
        1. Stratified train/test split (default 60% train / 40% test,
           configurable via ``model.train_split`` / ``model.test_split``).
        2. Model training via :class:`BPNeuralNetworkModel`.
        3. Evaluation against the held-out test set.
        4. Model persistence via ``save_model``.
    """

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        self._settings = settings or get_settings()

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform a stratified train/test split per configuration.

        Args:
            X: Standardized feature matrix, shape ``(n_samples, n_features)``.
            y: Condition class labels, shape ``(n_samples,)``.

        Returns:
            ``(X_train, X_test, y_train, y_test)``.
        """
        cfg = self._settings.model
        test_size = cfg.test_split
        stratify = y if cfg.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=cfg.random_state,
            stratify=stratify,
        )
        logger.info(
            "Split data into %d training and %d testing samples "
            "(test_split=%.2f, stratify=%s).",
            len(X_train),
            len(X_test),
            test_size,
            cfg.stratify,
        )
        return X_train, X_test, y_train, y_test

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Optional[BPNeuralNetworkModel] = None,
        save_path: Optional[str | Path] = None,
    ) -> TrainingResult:
        """Execute the full training pipeline.

        Args:
            X: Standardized feature matrix.
            y: Condition class labels.
            model: Optional pre-configured :class:`BPNeuralNetworkModel`.
                A new instance is created from configuration if omitted.
            save_path: Optional path to persist the trained model.

        Returns:
            A :class:`TrainingResult` containing the trained model and
            evaluation metrics.
        """
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        bp_model = model or BPNeuralNetworkModel(settings=self._settings)
        bp_model.train(X_train, y_train)

        evaluation = bp_model.evaluate(X_test, y_test)

        if save_path is not None:
            bp_model.save_model(save_path)

        return TrainingResult(
            model=bp_model,
            evaluation=evaluation,
            train_size=len(X_train),
            test_size=len(X_test),
        )
