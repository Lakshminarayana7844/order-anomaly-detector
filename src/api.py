"""FastAPI service for scoring sales orders.

On startup the service loads model.pkl if present; otherwise it trains on the
bundled synthetic CSV so the container works out of the box. Endpoints:

  GET  /health                    liveness
  POST /score                     score one or more orders, with reason codes
  GET  /orders/anomalies?top=N    top anomalies in the bundled training data
"""
from __future__ import annotations

import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .features import FeatureBuilder
from .model import AnomalyModel
from .reasons import ReasonEngine

DATA_PATH = os.environ.get("ORDERS_CSV", "data/orders_train.csv")
MODEL_PATH = os.environ.get("MODEL_PKL", "model.pkl")


class Order(BaseModel):
    order_id: Optional[str] = None
    created_at: str = Field(..., examples=["2025-06-01T10:30:00"])
    customer_id: str
    material: Optional[str] = None
    category: str = "FINISHED"
    plant: str = "1000"
    sales_org: Optional[str] = None
    quantity: float = Field(..., gt=0)
    unit_price: float = Field(..., ge=0)
    net_value: float = Field(..., ge=0)
    currency: str = "EUR"
    payment_terms: str = "NT30"
    delivery_days: float = Field(..., ge=0)


class ScoreRequest(BaseModel):
    orders: list[Order]


def _load_or_train() -> tuple[AnomalyModel, FeatureBuilder, ReasonEngine, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    if os.path.exists(MODEL_PATH):
        model = AnomalyModel.load(MODEL_PATH)
        builder = joblib.load(MODEL_PATH + ".features")
        engine = joblib.load(MODEL_PATH + ".reasons")
    else:
        builder = FeatureBuilder().fit(df)
        feats = builder.transform(df)
        model = AnomalyModel().fit(feats)
        engine = ReasonEngine(feats)
    return model, builder, engine, df


app = FastAPI(title="SAP Order Anomaly Detector", version="1.0.0")
MODEL, BUILDER, ENGINE, TRAIN_DF = _load_or_train()


def _score_df(df: pd.DataFrame) -> list[dict]:
    feats = BUILDER.transform(df)
    scored = MODEL.score(feats)
    results = []
    for (idx, row), (_, srow) in zip(df.iterrows(), scored.iterrows()):
        reasons = ENGINE.top_reasons(feats.loc[idx]) if srow["is_anomaly"] else []
        results.append(
            {
                "order_id": row.get("order_id"),
                "anomaly_score": float(srow["anomaly_score"]),
                "is_anomaly": bool(srow["is_anomaly"]),
                "reasons": reasons,
            }
        )
    return results


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "threshold": MODEL.threshold}


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    df = pd.DataFrame([o.model_dump() for o in req.orders])
    return {"results": _score_df(df)}


@app.get("/orders/anomalies")
def anomalies(top: int = 10) -> dict:
    results = _score_df(TRAIN_DF)
    flagged = [r for r in results if r["is_anomaly"]]
    flagged.sort(key=lambda r: r["anomaly_score"], reverse=True)
    return {"count": len(flagged), "results": flagged[: max(1, min(top, 100))]}
