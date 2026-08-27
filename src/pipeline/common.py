"""Shared paths and constants for the Phase 2 data processing pipeline."""

from __future__ import annotations

from pathlib import Path

from src.utils.constants import INPUT_FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
EXPORTS_DIR = REPO_ROOT / "data" / "exports"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

DIM_ASSET_PATH = RAW_DIR / "dim_asset.csv"
OPERATIONAL_MEASUREMENTS_PATH = RAW_DIR / "operational_measurements.csv"
NEW_MEASUREMENTS_PATH = RAW_DIR / "new_measurements.csv"

VALIDATION_REPORT_PATH = PROCESSED_DIR / "validation_report.json"
CLEAN_OPERATIONAL_MEASUREMENTS_PATH = (
    PROCESSED_DIR / "operational_measurements_clean.csv"
)
HEALTHY_BASELINE_PATH = PROCESSED_DIR / "healthy_baseline.csv"
MODEL_METRICS_PATH = PROCESSED_DIR / "model_metrics.json"
CONFUSION_MATRIX_IMAGE_PATH = PROCESSED_DIR / "confusion_matrix.png"
MD_SCORES_PATH = PROCESSED_DIR / "md_scores.csv"
AHI_SCORES_PATH = PROCESSED_DIR / "ahi_scores.csv"
BP_PREDICTIONS_PATH = PROCESSED_DIR / "bp_predictions.csv"
HEALTH_ASSESSMENTS_PATH = PROCESSED_DIR / "health_assessments.csv"

VISUALIZATIONS_HTML_PATH = EXPORTS_DIR / "visualizations.html"
VISUALIZATIONS_EXCEL_PATH = EXPORTS_DIR / "visualizations.xlsx"

SCALER_ARTIFACT_PATH = ARTIFACTS_DIR / "scaler.pkl"
HEALTHY_MEAN_ARTIFACT_PATH = ARTIFACTS_DIR / "healthy_mean.pkl"
COVARIANCE_ARTIFACT_PATH = ARTIFACTS_DIR / "covariance.pkl"
INV_COVARIANCE_ARTIFACT_PATH = ARTIFACTS_DIR / "inv_covariance.pkl"
BP_MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "bp_model.pkl"

FEATURE_COLUMNS: tuple[str, ...] = INPUT_FEATURE_NAMES
