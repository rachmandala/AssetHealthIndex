from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class PreprocessingArtifacts:
    features: list[str]
    normalize: bool
    normalizer: MinMaxScaler | None
    scaler: StandardScaler
    clip_bounds: dict[str, tuple[float, float]]


class Preprocessor:
    def __init__(self, features: list[str], outlier_quantiles: tuple[float, float] = (0.01, 0.99), normalize: bool = True) -> None:
        self.features = features
        self.outlier_quantiles = outlier_quantiles
        self.normalize = normalize
        self.normalizer: MinMaxScaler | None = MinMaxScaler() if normalize else None
        self.scaler = StandardScaler()
        self.clip_bounds: dict[str, tuple[float, float]] = {}

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        clean = self._impute_and_clip(df.copy())
        x = clean[self.features].to_numpy(dtype=float)
        if self.normalizer is not None:
            x = self.normalizer.fit_transform(x)
        return self.scaler.fit_transform(x)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        clean = self._impute_and_clip(df.copy(), fit=False)
        x = clean[self.features].to_numpy(dtype=float)
        if self.normalizer is not None:
            x = self.normalizer.transform(x)
        return self.scaler.transform(x)

    def _impute_and_clip(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        for col in self.features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median = df[col].median()
            df[col] = df[col].fillna(median)
            if fit:
                low_q, high_q = self.outlier_quantiles
                low = float(df[col].quantile(low_q))
                high = float(df[col].quantile(high_q))
                self.clip_bounds[col] = (low, high)
            low, high = self.clip_bounds[col]
            df[col] = df[col].clip(lower=low, upper=high)
        return df

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        artifacts = PreprocessingArtifacts(
            features=self.features,
            normalize=self.normalize,
            normalizer=self.normalizer,
            scaler=self.scaler,
            clip_bounds=self.clip_bounds,
        )
        joblib.dump(artifacts, path)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        artifacts: PreprocessingArtifacts = joblib.load(path)
        pre = Preprocessor(artifacts.features, normalize=artifacts.normalize)
        pre.normalizer = artifacts.normalizer
        pre.scaler = artifacts.scaler
        pre.clip_bounds = artifacts.clip_bounds
        return pre
