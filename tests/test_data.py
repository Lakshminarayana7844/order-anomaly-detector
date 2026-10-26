from src.generate_data import generate


def test_deterministic():
    a = generate(200, seed=1)
    b = generate(200, seed=1)
    assert a.equals(b)


def test_shape_and_columns():
    df = generate(1000, seed=2)
    assert len(df) == 1000
    for col in ["order_id", "customer_id", "net_value", "quantity",
                "payment_terms", "delivery_days", "is_anomaly"]:
        assert col in df.columns
    rate = df["is_anomaly"].mean()
    assert 0.02 <= rate <= 0.04


def test_values_sane():
    df = generate(500, seed=3)
    assert (df["net_value"] > 0).all()
    assert (df["quantity"] >= 1).all()
    assert df["order_id"].is_unique
