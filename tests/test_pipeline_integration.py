"""Integration tests for the Phase 2 data processing pipeline.

Runs the full pipeline against the real raw datasets in ``data/raw/`` and
checks that every expected artifact/output file is produced and internally
consistent. These are integration tests (not isolated unit tests) since the
pipeline is explicitly designed to operate on the checked-in raw CSVs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline import common
from src.pipeline.run_pipeline import run_full_pipeline


@pytest.fixture(scope="module")
def pipeline_result():
    return run_full_pipeline()


def test_validation_report_is_valid(pipeline_result) -> None:
    report = pipeline_result["validation_report"]
    assert report["operational_measurements"]["is_valid"] is True
    assert report["new_measurements"]["is_valid"] is True
    assert common.VALIDATION_REPORT_PATH.exists()


def test_preprocessing_artifacts_exist() -> None:
    assert common.SCALER_ARTIFACT_PATH.exists()
    assert common.CLEAN_OPERATIONAL_MEASUREMENTS_PATH.exists()


def test_healthy_baseline_artifacts_exist() -> None:
    assert common.HEALTHY_MEAN_ARTIFACT_PATH.exists()
    assert common.COVARIANCE_ARTIFACT_PATH.exists()
    assert common.INV_COVARIANCE_ARTIFACT_PATH.exists()
    assert common.HEALTHY_BASELINE_PATH.exists()


def test_bp_model_and_metrics(pipeline_result) -> None:
    assert common.BP_MODEL_ARTIFACT_PATH.exists()
    assert common.CONFUSION_MATRIX_IMAGE_PATH.exists()
    metrics = pipeline_result["model_metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["train_size"] + metrics["test_size"] == 1400


def test_md_and_ahi_scores_consistent() -> None:
    md_df = pd.read_csv(common.MD_SCORES_PATH)
    ahi_df = pd.read_csv(common.AHI_SCORES_PATH)
    assert len(md_df) == len(ahi_df) == 10
    assert (md_df["mahalanobis_distance"] >= 0).all()
    assert ahi_df["ahi_score"].between(0, 100).all()


def test_bp_predictions_have_all_probability_columns() -> None:
    bp_df = pd.read_csv(common.BP_PREDICTIONS_PATH)
    for col in [
        "healthy_probability",
        "subhealthy_probability",
        "abnormal_probability",
        "fault_probability",
    ]:
        assert col in bp_df.columns
    prob_sum = bp_df[
        [
            "healthy_probability",
            "subhealthy_probability",
            "abnormal_probability",
            "fault_probability",
        ]
    ].sum(axis=1)
    assert (prob_sum - 1.0).abs().max() < 1e-6


def test_health_assessments_have_valid_status(pipeline_result) -> None:
    assessments_df = pipeline_result["assessments"]
    assert len(assessments_df) == 10
    assert set(assessments_df["health_status"]).issubset(
        {"Healthy", "Sub-Healthy", "Abnormal", "Fault"}
    )
    assert assessments_df["recommendation"].notna().all()


def test_powerbi_exports_written(pipeline_result) -> None:
    exported_paths = pipeline_result["exported_paths"]
    for path in exported_paths.values():
        assert path.exists()

    fact_df = pd.read_csv(common.EXPORTS_DIR / "fact_asset_health_assessment.csv")
    assert len(fact_df) == 10
