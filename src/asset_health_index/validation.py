from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    issues: list[dict[str, Any]] = field(default_factory=list)

    def add_issue(self, check: str, message: str, count: int = 0) -> None:
        payload = {"check": check, "message": message, "count": int(count)}
        self.issues.append(payload)
        logger.warning("%s | %s | count=%s", check, message, count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_issues": len(self.issues),
            "issues": self.issues,
            "status": "pass" if not self.issues else "fail",
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class DataValidator:
    def __init__(self, features: list[str], label_col: str, ranges: dict[str, list[float]]) -> None:
        self.features = features
        self.label_col = label_col
        self.ranges = ranges
        self.required = ["timestamp", "asset_id", *features, label_col]

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        report = DataQualityReport()
        self._check_schema(df, report)
        self._check_missing(df, report)
        self._check_duplicates(df, report)
        self._check_datatypes(df, report)
        self._check_ranges(df, report)
        return report

    def _check_schema(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        missing_cols = [col for col in self.required if col not in df.columns]
        if missing_cols:
            report.add_issue("schema_validation", f"Missing columns: {missing_cols}", len(missing_cols))

    def _check_missing(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        available = [col for col in self.required if col in df.columns]
        missing_counts = df[available].isna().sum()
        for col, count in missing_counts.items():
            if count > 0:
                report.add_issue("missing_value", f"Missing values in {col}", int(count))

    def _check_duplicates(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            report.add_issue("duplicate_record", "Duplicate rows detected", dup_count)

    def _check_datatypes(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        for col in self.features:
            if col not in df.columns:
                continue
            converted = pd.to_numeric(df[col], errors="coerce")
            invalid = int(converted.isna().sum() - df[col].isna().sum())
            if invalid > 0:
                report.add_issue("invalid_datatype", f"Non-numeric values in {col}", invalid)

    def _check_ranges(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        for col, bounds in self.ranges.items():
            if col not in df.columns:
                continue
            low, high = bounds
            numeric = pd.to_numeric(df[col], errors="coerce")
            invalid = int(((numeric < low) | (numeric > high)).sum())
            if invalid > 0:
                report.add_issue("range_check", f"Out-of-range values in {col} [{low}, {high}]", invalid)
