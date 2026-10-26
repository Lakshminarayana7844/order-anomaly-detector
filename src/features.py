"""Feature engineering for order anomaly detection.

Features are built from fields a SAP SD extract already carries, plus a
customer-relative value feature (this order's value vs the customer's median)
that needs training-set statistics. The builder is fitted on training data and
reused at scoring time so train and inference can never drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMMON_TERMS = ["NT30", "NT45", "NT60", "Z001"]
CATEGORIES = ["RAW", "FINISHED", "TRADING"]

NUMERIC_FEATURES = [
    "log_net_value",
    "log_quantity",
    "unit_price",
    "delivery_days",
    "customer_rel_value",
    "order_hour",
    "order_dow",
]
CATEGORICAL_FEATURES = ["category_code", "plant_code", "rare_payment_terms"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class FeatureBuilder:
    def __init__(self) -> None:
        self.customer_median: dict[str, float] = {}
        self.global_median: float = 1.0
        self.plant_map: dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        self.customer_median = df.groupby("customer_id")["net_value"].median().to_dict()
        self.global_median = float(df["net_value"].median())
        plants = sorted(df["plant"].astype(str).unique())
        self.plant_map = {p: i for i, p in enumerate(plants)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["log_net_value"] = np.log1p(df["net_value"].clip(lower=0))
        out["log_quantity"] = np.log1p(df["quantity"].clip(lower=0))
        out["unit_price"] = df["unit_price"].astype(float)
        out["delivery_days"] = df["delivery_days"].astype(float)
        med = df["customer_id"].map(self.customer_median).fillna(self.global_median)
        out["customer_rel_value"] = df["net_value"] / med.clip(lower=1.0)
        created = pd.to_datetime(df["created_at"])
        out["order_hour"] = created.dt.hour
        out["order_dow"] = created.dt.dayofweek
        out["category_code"] = df["category"].map({c: i for i, c in enumerate(CATEGORIES)}).fillna(-1)
        out["plant_code"] = df["plant"].astype(str).map(self.plant_map).fillna(-1)
        out["rare_payment_terms"] = (~df["payment_terms"].isin(COMMON_TERMS)).astype(int)
        return out[FEATURES].astype(float)
