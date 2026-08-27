"""Unit tests for the data validation engine (src/validation/validator.py)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.validation.validator import (
    DataValidationError,
    OperationalMeasurementValidator,
)


def _valid_row(measurement_id: str = "M1", asset_id: str = "A1") -> dict:
    return {
        "measurement_id": measurement_id,
        "asset_id": asset_id,
        "timestamp": datetime(2026, 1, 1, 0, 0, 0),
        "speed": 3000.0,
        "frequency": 50.0,
        "voltage": 11000.0,
        "current": 500.0,
        "active_power": 10.0,
        "reactive_power": 2.0,
        "power_factor": 0.95,
    }


def test_validate_schema_detects_missing_columns() -> None:
    df = pd.DataFrame([{"measurement_id": "M1"}])
    validator = OperationalMeasurementValidator()
    issues = validator.validate_schema(df)
    assert any(issue.check == "schema" for issue in issues)


def test_validate_valid_dataframe_has_no_errors() -> None:
    df = pd.DataFrame([_valid_row()])
    validator = OperationalMeasurementValidator()
    report = validator.validate(df)
    assert report.is_valid
    assert report.total_records == 1


def test_check_missing_values_detects_nan() -> None:
    row = _valid_row()
    row["speed"] = None
    df = pd.DataFrame([row])
    validator = OperationalMeasurementValidator()
    issues = validator.check_missing_values(df)
    assert len(issues) == 1
    assert issues[0].column == "speed"


def test_check_duplicates_detects_repeated_measurement_id() -> None:
    df = pd.DataFrame([_valid_row(), _valid_row()])
    validator = OperationalMeasurementValidator()
    issues = validator.check_duplicates(df)
    assert len(issues) == 1


def test_check_data_types_detects_non_numeric() -> None:
    row = _valid_row()
    row["speed"] = "not-a-number"
    df = pd.DataFrame([row])
    validator = OperationalMeasurementValidator()
    issues = validator.check_data_types(df)
    assert len(issues) == 1
    assert issues[0].check == "invalid_data_type"


def test_check_ranges_flags_out_of_range_as_warning() -> None:
    row = _valid_row()
    row["power_factor"] = -5.0  # outside configured [-1.0, 1.0]
    df = pd.DataFrame([row])
    validator = OperationalMeasurementValidator()
    issues = validator.check_ranges(df)
    assert len(issues) == 1
    assert issues[0].severity == "warning"


def test_validate_raises_when_raise_on_error_true() -> None:
    row = _valid_row()
    row["speed"] = None
    df = pd.DataFrame([row])
    validator = OperationalMeasurementValidator()
    with pytest.raises(DataValidationError):
        validator.validate(df, raise_on_error=True)
