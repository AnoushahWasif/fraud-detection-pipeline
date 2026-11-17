# fraud-detection-pipeline

Flagging fraudulent card transactions with a small pipeline built the way a team would run it: config-driven ingestion, schema validation, reproducible training with metadata, experiment tracking, a scoring API in Docker, and CI that runs the whole pipeline.

The model is the less interesting part. The point is that every number can be traced to the data, code and config that produced it, and that the alert threshold is set by what mistakes cost, not by 0.5.

**Status:** in progress (day 7 of 12). Results will go here once there are any.

## Planned approach

1. Pinned, hash-checked download of OpenML dataset 1597, validated with a Pandera schema
2. EDA: imbalance (0.17% fraud), amounts, time of day
3. Split by time, with leakage checks as tests; resampling only inside training folds
4. Class weights against under- and oversampling, judged by PR-AUC
5. An alert threshold chosen by cost: a missed fraud costs its amount, a false alarm costs a review
6. MLflow tracking, a FastAPI scoring service in Docker, and `make all`

## Stack

Python 3.12, uv, pandas, scikit-learn, matplotlib, pytest, ruff. CI: GitHub Actions with ruff, pytest, pip-audit and Gitleaks.

## Data

[OpenML dataset 1597](https://www.openml.org/d/1597) (`creditcard`): card transactions from September 2013 by European cardholders, collected by Worldline and the ULB Machine Learning Group (Dal Pozzolo et al., 2015). It isn't committed: a script downloads a pinned version and checks its hash.

## Quick start

```bash
uv sync
make data    # downloads the pinned 73 MB Parquet once, checks its SHA-256 and the schema
make lint test
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT
