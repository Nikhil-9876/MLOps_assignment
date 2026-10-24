# HTTP Request Anomaly Detector — CSIC 2010

A reproducible MLOps pipeline that classifies HTTP requests as **normal** or **attack**, exported to ONNX for production inference.

## Dataset
- **CSIC 2010** — 36,000 normal + 25,065 attack requests in `dataset/csic_database.csv`
- Attack types: SQL injection, XSS, path traversal, command injection, parameter tampering

## Repo Structure
```
dataset/            # raw data (untouched)
src/
  parse.py          # CSV -> structured records + deduplication
  features.py       # pure function: record -> fixed-length feature vector
  train.py          # training pipeline (LR baseline + LightGBM)
  evaluate.py       # metrics, plots, error analysis
  export_onnx.py    # ONNX export + latency benchmark
  predict.py        # production inference
models/             # trained artifacts (pkl, json, onnx)
reports/            # metrics.json, plots
tests/              # pytest suite
```

## Quick Start

```bash
pip install -r requirements.txt

# Run full pipeline
make train       # parse -> feature extraction -> train -> save models
make evaluate    # plots + error analysis
make export      # ONNX export + latency benchmark
make test        # run pytest suite
make ci-check    # fails if prod holdout FPR > 1.5%
```

Or run each step individually:
```bash
python -m src.train --csv dataset/csic_database.csv
python -m src.evaluate
python -m src.export_onnx

# Single prediction
python -m src.predict --method GET --url "http://localhost:8080/tienda1/index.jsp"
```

## Model Choice: LightGBM over XGBoost
- **Speed**: LightGBM's leaf-wise growth is 3–5× faster on this dataset size
- **PR-AUC**: Consistently higher in CV experiments due to better minority-class handling
- **class_weight='balanced'**: Compensates for 36k/25k imbalance

## Primary Metrics
The headline metric is **FPR on normal traffic** (must be ≤ 1.5% for CI gate) and **PR-AUC**.
Accuracy is intentionally not the headline metric.

| Metric | Description |
|--------|-------------|
| `fpr`  | False positive rate on clean traffic — CI gate |
| `pr_auc` | Area under Precision-Recall curve |
| `roc_auc` | Area under ROC curve |
| `recall` | Fraction of attacks detected |

## DVC Compatibility
Each pipeline stage reads inputs from files and writes outputs to files — no hidden global state. To wrap in DVC:
```bash
dvc init
dvc run -n train -d dataset/csic_database.csv -o models/ -o reports/ python -m src.train
dvc run -n evaluate -d models/ -d reports/ -o reports/ python -m src.evaluate
dvc run -n export -d models/ -d reports/ -o models/model.onnx python -m src.export_onnx
```

## Feature Groups
- **Content**: lengths, Shannon entropy, special char counts/ratios, encoding, injection keywords (SQL, XSS, path-traversal, cmd)
- **Structure-aware**: unknown endpoint, unknown param names, param count deviation, value-length z-score, unseen chars (learned from normal traffic only)
