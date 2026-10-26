import pandas as pd

from src.features import FeatureBuilder
from src.reasons import ReasonEngine


def test_reasons_sorted_and_nonempty(small_df):
    builder = FeatureBuilder().fit(small_df)
    feats = builder.transform(small_df)
    engine = ReasonEngine(feats)
    # Use a genuinely anomalous injected order.
    row = small_df[small_df["is_anomaly"]].iloc[0:1]
    frow = builder.transform(row).iloc[0]
    reasons = engine.top_reasons(frow)
    assert reasons, "an injected anomaly should produce at least one reason"
    mags = [r["magnitude"] for r in reasons]
    assert mags == sorted(mags, reverse=True)


def test_big_order_flags_customer_relative(small_df):
    builder = FeatureBuilder().fit(small_df)
    feats = builder.transform(small_df)
    engine = ReasonEngine(feats)
    cust = small_df["customer_id"].iloc[0]
    median_val = builder.customer_median[cust]
    big = pd.DataFrame([{
        "order_id": "SO-X", "created_at": "2025-06-01T10:00:00",
        "customer_id": cust, "material": "MAT-2001", "category": "FINISHED",
        "plant": "1000", "sales_org": "DE01", "quantity": 50,
        "unit_price": median_val, "net_value": median_val * 50,
        "currency": "EUR", "payment_terms": "NT30", "delivery_days": 7,
        "is_anomaly": True,
    }])
    frow = builder.transform(big).iloc[0]
    reasons = engine.top_reasons(frow)
    assert any(r["feature"] == "customer_rel_value" for r in reasons)


def test_normal_order_may_have_no_reasons(small_df):
    builder = FeatureBuilder().fit(small_df)
    feats = builder.transform(small_df)
    engine = ReasonEngine(feats)
    # The median training row should produce no reasons.
    frow = feats.median()
    assert engine.top_reasons(frow) == []
