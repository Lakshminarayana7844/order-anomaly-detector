"""Train the anomaly model on the bundled (or a real) order CSV.

Usage: python -m src.train --data data/orders_train.csv --out model.pkl
Prints precision/recall against the labels when the CSV carries them.
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd

from .features import FeatureBuilder
from .model import AnomalyModel
from .reasons import ReasonEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/orders_train.csv")
    parser.add_argument("--out", default="model.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    builder = FeatureBuilder().fit(df)
    feats = builder.transform(df)
    model = AnomalyModel().fit(feats)

    if "is_anomaly" in df.columns:
        scored = model.score(feats)
        truth = df["is_anomaly"].astype(bool)
        tp = int((scored["is_anomaly"] & truth).sum())
        fp = int((scored["is_anomaly"] & ~truth).sum())
        fn = int((~scored["is_anomaly"] & truth).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        print(f"labels found: precision={precision:.2f} recall={recall:.2f} "
              f"(threshold={model.threshold:.3f})")

    model.save(args.out)
    joblib.dump(builder, args.out + ".features")
    engine = ReasonEngine(feats)
    joblib.dump(engine, args.out + ".reasons")
    print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
