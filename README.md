# SAP Order Anomaly Detector

A small, self-contained ML service that flags unusual sales orders from an
SAP SD-style extract (VBAK/VBAP shape). It trains an IsolationForest on
historical orders, scores new orders over a REST API, and explains every flag
with plain-language reason codes, so a sales-ops or finance user can see
*why* an order looks wrong instead of just getting a score.

Built as a portfolio project. The bundled data is fully synthetic.

## What it does

- Trains on the bundled synthetic extract (2,000 orders, 3% labeled anomalies)
  or on a real CSV you point it at.
- Scores orders via `POST /score` and returns a calibrated 0-1 anomaly score,
  a flag, and reason codes such as "order value is 24.7x this customer's
  median order" or "delivery time of 152 days is far above the typical range".
- Ships with tests, a Dockerfile, and GitHub Actions CI.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                      # 15 tests
uvicorn src.api:app --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/score -H 'Content-Type: application/json' -d '{
  "orders": [{
    "created_at": "2025-06-01T10:30:00",
    "customer_id": "C0001",
    "category": "FINISHED",
    "plant": "1000",
    "quantity": 2000,
    "unit_price": 140.0,
    "net_value": 280000.0,
    "payment_terms": "Z999",
    "delivery_days": 150
  }]
}'
```

With Docker:

```bash
docker build -t order-anomaly-detector .
docker run -p 8000:8000 order-anomaly-detector
```

The API trains on the bundled CSV at startup if no `model.pkl` is mounted, so
the container works out of the box.

## Training on your own data

```bash
python -m src.train --data your_orders.csv --out model.pkl
```

Required columns: `order_id, created_at, customer_id, material, category,
plant, sales_org, quantity, unit_price, net_value, currency, payment_terms,
delivery_days`. An `is_anomaly` label column is optional; when present, the
trainer prints precision/recall.

To regenerate the synthetic data:

```bash
python -m src.generate_data --rows 2000 --out data/orders_train.csv
```

## Model quality (honest numbers)

Evaluated against the labels in the synthetic training data
(`python -m src.train`):

| Metric    | Value |
| --------- | ----- |
| Precision | 0.67  |
| Recall    | 0.67  |

The threshold flags about 3% of orders, matching the injected anomaly rate.
The anomalies it catches cleanly are extreme values, quantities, payment
terms, and delivery times. Subtle fraud or pricing errors inside the normal
range are **not** detected; IsolationForest only finds orders that look
unusual relative to the training distribution.

## Design notes and limits

- **Features**: log net value, log quantity, unit price, delivery days,
  order hour/weekday, category and plant codes, a rare-payment-terms flag,
  and a customer-relative value feature (order value divided by the
  customer's median order value). The customer-relative feature is what
  catches "normal amount, wrong customer" cases.
- **Reason codes are a heuristic, not SHAP**: each feature is compared to the
  training median with a robust (MAD-based) deviation and the largest
  deviations are reported. Fast and easy to audit, but it is not a formal
  attribution method and can over-credit correlated features.
- **No drift handling**: the model is static once trained. A real deployment
  would retrain on a schedule and monitor score distributions.
- **Synthetic data**: the generator mimics the shape of an SD extract. Real
  SAP data needs its own calibration (currencies, returns, order types).

## Project layout

```
src/generate_data.py   synthetic SAP SD-style data generator (labeled)
src/features.py        fitted feature builder (train/inference parity)
src/model.py           IsolationForest wrapper with calibrated score
src/reasons.py         plain-language reason codes
src/api.py             FastAPI service (/score, /orders/anomalies, /health)
src/train.py           training CLI with precision/recall output
data/orders_train.csv  2,000 synthetic orders (3% anomalies)
tests/                 15 tests: data, model, reasons, API
```

## License

MIT
