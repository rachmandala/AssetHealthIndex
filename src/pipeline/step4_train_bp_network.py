"""Phase 2 - Step 4: BP Neural Network training and evaluation.

Trains the BP Neural Network (MLPClassifier) on standardized historical
operational measurements using a stratified 60/40 train/test split, then
evaluates it and persists the model, metrics, and a confusion matrix plot.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # Headless rendering; no display server required.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

from src.ml.bp_network import BPNeuralNetworkModel
from src.ml.training_pipeline import BPNetworkTrainingPipeline
from src.pipeline.common import (
    BP_MODEL_ARTIFACT_PATH,
    CONFUSION_MATRIX_IMAGE_PATH,
    MODEL_METRICS_PATH,
)
from src.preprocessing.preprocessor import Preprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_bp_training(
    clean_df: pd.DataFrame, preprocessor: Preprocessor
) -> tuple[BPNeuralNetworkModel, dict, dict]:
    """Train and evaluate the BP Neural Network condition classifier.

    Args:
        clean_df: Cleaned historical operational measurements with a
            ``label`` column (output of Step 2).
        preprocessor: Fitted :class:`Preprocessor` from Step 2.

    Returns:
        A tuple of ``(trained_model, metrics_dict, split_info)``. ``split_info``
        contains the stratified train/test arrays and predicted labels for
        both splits, for use by downstream visualization steps.
    """
    X = preprocessor.transform(clean_df)
    y = clean_df["label"].to_numpy()

    pipeline = BPNetworkTrainingPipeline()
    result = pipeline.run(X, y, save_path=BP_MODEL_ARTIFACT_PATH)
    model = result.model
    evaluation = result.evaluation

    X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": evaluation["accuracy"],
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": evaluation["confusion_matrix"],
        "labels": evaluation["labels"],
        "classification_report": evaluation["classification_report"],
        "train_size": result.train_size,
        "test_size": result.test_size,
    }

    MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Wrote model metrics to '%s'.", MODEL_METRICS_PATH)
    logger.info(
        "BP Neural Network evaluation: accuracy=%.4f, balanced_accuracy=%.4f, "
        "f1_macro=%.4f",
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["f1_macro"],
    )

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=evaluation["labels"], xticks_rotation=45
    )
    disp.figure_.tight_layout()
    CONFUSION_MATRIX_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    disp.figure_.savefig(CONFUSION_MATRIX_IMAGE_PATH, dpi=150)
    plt.close(disp.figure_)
    logger.info("Saved confusion matrix image to '%s'.", CONFUSION_MATRIX_IMAGE_PATH)

    split_info = {
        "labels": evaluation["labels"],
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "y_test": y_test,
        "y_test_pred": y_pred,
    }

    return model, metrics, split_info


if __name__ == "__main__":
    from src.pipeline.step2_preprocess_data import run_preprocessing

    _clean_df, _preprocessor = run_preprocessing()
    run_bp_training(_clean_df, _preprocessor)
