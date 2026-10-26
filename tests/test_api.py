from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_ORDER = {
    "created_at": "2025-06-01T10:30:00",
    "customer_id": "C0001",
    "category": "FINISHED",
    "plant": "1000",
    "quantity": 5,
    "unit_price": 140.0,
    "net_value": 700.0,
    "payment_terms": "NT30",
    "delivery_days": 7,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_valid():
    r = client.post("/score", json={"orders": [VALID_ORDER]})
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert 0.0 <= res["anomaly_score"] <= 1.5
    assert isinstance(res["is_anomaly"], bool)
    assert isinstance(res["reasons"], list)


def test_score_flags_extreme_order():
    extreme = dict(VALID_ORDER, quantity=2000, unit_price=140.0,
                   net_value=280000.0, delivery_days=120)
    r = client.post("/score", json={"orders": [extreme]})
    res = r.json()["results"][0]
    assert res["is_anomaly"]
    assert res["reasons"]


def test_score_validation_error():
    bad = dict(VALID_ORDER)
    del bad["net_value"]
    r = client.post("/score", json={"orders": [bad]})
    assert r.status_code == 422


def test_anomalies_endpoint_sorted():
    r = client.get("/orders/anomalies?top=5")
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 5
    scores = [x["anomaly_score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True)
