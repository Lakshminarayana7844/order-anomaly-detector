"""IsolationForest anomaly model with a calibrated 0-1 score and threshold."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyModel:
    """Wraps IsolationForest and remembers the score calibration from training.

    anomaly_score is min-max scaled from the training decision scores so a
    score near 1.0 means "as unusual as the most unusual training order".
    The threshold defaults to the 97th percentile of training scores, which
    matches the contamination the generator injects.
    """

    def __init__(self, contamination: float = 0.03, threshold_quantile: float = 0.97, seed: int = 42):
        self.contamination = contamination
        self.threshold_quantile = threshold_quantile
        self.seed = seed
        self.forest = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=seed
        )
        self.score_min: float = 0.0
        self.score_max: float = 1.0
        self.threshold: float = 0.0
        self._fitted = False

    def fit(self, features: pd.DataFrame) -> "AnomalyModel":
        self.forest.fit(features)
        raw = -self.forest.decision_function(features)
        self.score_min, self.score_max = float(raw.min()), float(raw.max())
        scaled = self._scale(raw)
        self.threshold = float(np.quantile(scaled, self.threshold_quantile))
        self._fitted = True
        return self

    def _scale(self, raw: np.ndarray) -> np.ndarray:
        span = max(self.score_max - self.score_min, 1e-9)
        return np.clip((raw - self.score_min) / span, 0.0, 1.5)

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        raw = -self.forest.decision_function(features)
        scaled = self._scale(raw)
        return pd.DataFrame(
            {
                "anomaly_score": np.round(scaled, 4),
                "is_anomaly": scaled >= self.threshold,
            },
            index=features.index,
        )

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "AnomalyModel":
        model = joblib.load(path)
        if not isinstance(model, AnomalyModel):
            raise TypeError(f"{path} does not contain an AnomalyModel")
        return model
