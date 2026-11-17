"""Day 4: reproducible training with metadata (decision 8).

    uv run python -m fraud_pipeline.train

Fits the baseline (class-weighted logistic regression) on the training split and saves it to
models/baseline with everything needed to trace it: data and config hashes, git commit and
whether the tree had uncommitted changes, library versions, seed, features and validation
metrics.
"""

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud_pipeline import config as config_module
from fraud_pipeline import ingest, split
from fraud_pipeline.evaluate import summary
from fraud_pipeline.model_store import save_model
from fraud_pipeline.schema import FEATURES, TARGET


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    try:
        return {
            "commit": run("rev-parse", "HEAD").strip(),
            "dirty": bool(run("status", "--porcelain").strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def baseline(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed),
    )


def fit(model, train: pd.DataFrame):
    return model.fit(train[FEATURES], train[TARGET])


def scores(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, 1]


def metadata(cfg, config_sha: str, extra: dict) -> dict:
    return {
        "data_sha256": cfg.data.sha256,
        "config_sha256": config_sha,
        "git": git_state(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "seed": cfg.seed,
        "features": FEATURES,
        **extra,
    }


def main(out_dir: Path = Path("models/baseline"), report: Path = Path("reports/baseline.json")):
    cfg, config_sha = config_module.load()
    splits = split.by_time(ingest.load(cfg), cfg.split)
    split.check_no_leakage(splits, FEATURES)
    model = fit(baseline(cfg.seed), splits.train)
    val = summary(splits.validation[TARGET], scores(model, splits.validation))
    meta = metadata(
        cfg, config_sha, {"model": "logistic regression, class_weight=balanced", "validation": val}
    )
    # Model first, so the report never names a hash that wasn't saved.
    meta["model_sha256"] = save_model(model, out_dir, meta)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
