"""Phase 2 - Step 7: BP Neural Network predictions on new measurements.

Runs the trained BP Neural Network against ``new_measurements.csv`` to
produce a condition class prediction and per-class probabilities.
"""

from __future__ import annotations

import pandas as pd

from src.ml.bp_network import BPNeuralNetworkModel
from src.pipeline.common import (
    BP_MODEL_ARTIFACT_PATH,
    BP_PREDICTIONS_PATH,
    NEW_MEASUREMENTS_PATH,
    SCALER_ARTIFACT_PATH,
)
from src.preprocessing.preprocessor import Preprocessor
from src.utils.constants import (
    CLASS_ABNORMAL,
    CLASS_FAULT,
    CLASS_HEALTHY,
    CLASS_SUB_HEALTHY,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROBABILITY_COLUMN_BY_CLASS = {
    CLASS_HEALTHY: "healthy_probability",
    CLASS_SUB_HEALTHY: "subhealthy_probability",
    CLASS_ABNORMAL: "abnormal_probability",
    CLASS_FAULT: "fault_probability",
}


def run_bp_predictions() -> pd.DataFrame:
    """Predict BP condition classes for ``new_measurements.csv``.

    Returns:
        DataFrame with ``asset_id``, ``timestamp``, ``bp_prediction``, and
        one probability column per condition class.
    """
    new_df = pd.read_csv(NEW_MEASUREMENTS_PATH, parse_dates=["timestamp"])

    preprocessor = Preprocessor().load_artifacts(SCALER_ARTIFACT_PATH)
    X_new_scaled = preprocessor.transform(new_df)

    model = BPNeuralNetworkModel().load_model(BP_MODEL_ARTIFACT_PATH)
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)

    # model.classes gives the original config order, but predict_proba columns
    # follow the underlying classifier's sorted classes_ attribute.
    class_order = sorted(model.classes)

    result_df = pd.DataFrame(
        {
            "asset_id": new_df["asset_id"],
            "timestamp": new_df["timestamp"],
            "bp_prediction": predictions,
        }
    )
    for class_name, column_name in PROBABILITY_COLUMN_BY_CLASS.items():
        if class_name in class_order:
            idx = class_order.index(class_name)
            result_df[column_name] = probabilities[:, idx]
        else:
            result_df[column_name] = 0.0

    BP_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(BP_PREDICTIONS_PATH, index=False)
    logger.info("Wrote %d BP predictions to '%s'.", len(result_df), BP_PREDICTIONS_PATH)

    return result_df


if __name__ == "__main__":
    run_bp_predictions()
