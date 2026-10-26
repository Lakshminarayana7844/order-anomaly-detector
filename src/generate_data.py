"""Generate synthetic SAP SD-style sales order data with labeled anomalies.

The data is fully synthetic. It mimics the shape of an SAP SD VBAK/VBAP extract
(order header + item fields) so the pipeline can later be pointed at a real
extract without code changes. Labels are kept in the CSV so model quality can
be evaluated honestly; a real deployment would not have them.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

CUSTOMERS = [f"C{i:04d}" for i in range(1, 51)]
MATERIALS = [
    ("MAT-1001", "RAW", 12.0),
    ("MAT-1002", "RAW", 8.5),
    ("MAT-2001", "FINISHED", 140.0),
    ("MAT-2002", "FINISHED", 220.0),
    ("MAT-2003", "FINISHED", 95.0),
    ("MAT-3001", "TRADING", 450.0),
    ("MAT-3002", "TRADING", 780.0),
]
PLANTS = ["1000", "1100", "2000", "3000"]
SALES_ORGS = ["DE01", "EU01", "US01"]
COMMON_TERMS = ["NT30", "NT45", "NT60", "Z001"]


def generate(n_rows: int = 2000, seed: int = 42, anomaly_rate: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Each customer has a typical order-size factor (lognormal).
    cust_factor = {c: float(rng.lognormal(0.0, 0.5)) for c in CUSTOMERS}
    rows = []
    n_anom = int(n_rows * anomaly_rate)
    labels = np.array([False] * (n_rows - n_anom) + [True] * n_anom)
    rng.shuffle(labels)

    start = np.datetime64("2025-01-01")
    for i in range(n_rows):
        cust = CUSTOMERS[rng.integers(0, len(CUSTOMERS))]
        mat, cat, price = MATERIALS[rng.integers(0, len(MATERIALS))]
        qty = max(1, int(rng.lognormal(2.2, 0.7)))
        net = qty * price * cust_factor[cust] * float(rng.lognormal(0.0, 0.15))
        payterms = COMMON_TERMS[rng.integers(0, len(COMMON_TERMS))]
        delivery = int(np.clip(rng.normal(7, 3), 1, 30))
        day = start + rng.integers(0, 300).astype("timedelta64[D]")
        hour = int(rng.integers(6, 19))

        if labels[i]:
            kind = rng.integers(0, 4)
            if kind == 0:
                net *= float(rng.uniform(15, 40))  # extreme value
            elif kind == 1:
                qty = int(rng.uniform(500, 3000))  # extreme quantity
                net = qty * price * cust_factor[cust]
            elif kind == 2:
                payterms = "Z999"  # unheard-of payment terms
            else:
                delivery = int(rng.uniform(90, 180))  # extreme delivery time

        rows.append(
            {
                "order_id": f"SO-{100000 + i}",
                "created_at": f"{day}T{hour:02d}:{int(rng.integers(0,60)):02d}:00",
                "customer_id": cust,
                "material": mat,
                "category": cat,
                "plant": PLANTS[rng.integers(0, len(PLANTS))],
                "sales_org": SALES_ORGS[rng.integers(0, len(SALES_ORGS))],
                "quantity": qty,
                "unit_price": round(price * cust_factor[cust], 2),
                "net_value": round(net, 2),
                "currency": "EUR",
                "payment_terms": payterms,
                "delivery_days": delivery,
                "is_anomaly": bool(labels[i]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/orders_train.csv")
    args = parser.parse_args()
    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {len(df)} rows ({df['is_anomaly'].sum()} anomalies) to {args.out}")


if __name__ == "__main__":
    main()
