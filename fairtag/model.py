"""Load the trained LightGBM model, price cars, give each price a range and break it down into euros per feature.
No Streamlit here."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fairtag.schema import CATEGORICAL, FEATURES

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_FILE = "fairtag_lightgbm.txt"
FLOOR = 100  # the lowest asking price in the training data

VERDICTS = ["Below the range", "Within the range", "Above the range"]


class PriceModel:
    """Gradient-boosted trees (LightGBM) that predict the asking price in euros from nine car features. The range is
    the band that held 80% of real asking prices around similar predictions in cross-validation."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.booster = lgb.Booster(model_file=str(model_dir / MODEL_FILE))
        self.meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.categories: dict[str, list[str]] = self.meta["categories"]
        rng = self.meta["range"]
        self._edges = np.asarray(rng["edges"], dtype=float)
        self._lo = np.asarray(rng["low_ratio"], dtype=float)
        self._hi = np.asarray(rng["high_ratio"], dtype=float)

    def frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Model input: the nine features, categories in the training order (an unseen value counts as missing)."""
        x = rows[FEATURES].copy()
        for c in CATEGORICAL:
            v = x[c].astype(str)
            x[c] = pd.Categorical(v.where(v.isin(self.categories[c])), categories=self.categories[c])
        for c in ("Age", "Power", "Mileage"):
            x[c] = x[c].astype(float)
        return x

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """Fair asking price in euros: what similar cars were listed for."""
        return np.maximum(self.booster.predict(self.frame(rows)), FLOOR)

    def price_range(self, fair: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Low and high end of the 80% range for each predicted price."""
        fair = np.asarray(fair, dtype=float)
        band = np.clip(np.searchsorted(self._edges, fair, side="right") - 1, 0, len(self._lo) - 1)
        return np.maximum(fair * self._lo[band], FLOOR), fair * self._hi[band]

    def explain(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Euros each feature adds to or takes from the average asking price (SHAP values from LightGBM). The column
        'base' is the starting point; base + the nine features = the raw prediction, exactly."""
        c = self.booster.predict(self.frame(rows), pred_contrib=True)
        return pd.DataFrame(c, columns=FEATURES + ["base"], index=rows.index)

    @staticmethod
    def verdict(ask: np.ndarray | float, low: np.ndarray | float, high: np.ndarray | float) -> np.ndarray:
        """Where an asking price sits against the range."""
        ask, low, high = (np.asarray(v, dtype=float) for v in (ask, low, high))
        return np.where(ask < low, VERDICTS[0], np.where(ask > high, VERDICTS[2], VERDICTS[1]))


def round_price(x: float, step: int = 50) -> int:
    return int(round(float(x) / step) * step)
