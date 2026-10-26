"""Plain-language reason codes for flagged orders.

This is deliberately a heuristic, not SHAP: each numeric feature is compared
against the training distribution (median + MAD) and the features with the
largest robust deviation are reported. Honest, fast, and easy to audit, at
the cost of not being a formal feature-attribution method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES

DISPLAY = {
    "log_net_value": ("net value", np.expm1, "EUR"),
    "log_quantity": ("quantity", np.expm1, "units"),
    "unit_price": ("unit price", lambda x: x, "EUR"),
    "delivery_days": ("delivery time", lambda x: x, "days"),
    "customer_rel_value": ("value vs this customer's median order", lambda x: x, "x"),
    "order_hour": ("order hour", lambda x: x, "h"),
    "order_dow": ("day of week (0=Mon)", lambda x: x, ""),
    "category_code": ("product category", lambda x: x, ""),
    "plant_code": ("plant", lambda x: x, ""),
    "rare_payment_terms": ("unusual payment terms", lambda x: x, ""),
}


class ReasonEngine:
    def __init__(self, train_features: pd.DataFrame):
        self.median = train_features.median()
        mad = (train_features - self.median).abs().median()
        self.mad = mad.replace(0.0, 1e-6)

    def top_reasons(self, feature_row: pd.Series, top_k: int = 3) -> list[dict]:
        z = ((feature_row[FEATURES] - self.median) / (self.mad + 1e-9)).abs()
        order = z.sort_values(ascending=False)
        reasons = []
        for feat in order.index[:top_k]:
            if order[feat] < 3.0:  # not meaningfully unusual
                continue
            name, conv, unit = DISPLAY[feat]
            val = conv(feature_row[feat])
            if feat == "rare_payment_terms":
                msg = "payment terms are not in the common set (NT30/NT45/NT60/Z001)"
            elif feat == "customer_rel_value":
                msg = f"order value is {val:.1f}x this customer's median order"
            else:
                direction = "above" if feature_row[feat] > self.median[feat] else "below"
                msg = f"{name} of {val:,.0f} {unit} is far {direction} the typical range".strip()
            reasons.append({"feature": feat, "message": msg, "magnitude": round(float(order[feat]), 2)})
        return reasons
