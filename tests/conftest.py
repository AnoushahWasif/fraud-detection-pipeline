import hashlib
import io

import numpy as np
import pandas as pd
import pytest

from fraud_pipeline.schema import FEATURES, TARGET


def transactions(n=200, frauds=5, seed=0) -> pd.DataFrame:
    """Made-up transactions with the real column layout. Not the real data."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"V{i}": rng.normal(size=n) for i in range(1, 29)})
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_000, n)))
    df["Amount"] = rng.gamma(2.0, 40.0, n).round(2)
    df[TARGET] = 0
    df.loc[rng.choice(n, frauds, replace=False), TARGET] = 1
    return df[[*FEATURES, TARGET]]


def parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def table():
    return transactions()


@pytest.fixture
def config_text():
    def make(sha="0" * 64, **overrides):
        v = {
            "url": "https://example.org/dataset.pq",
            "sha256": sha,
            "max_bytes": 1_000_000,
            "timeout_seconds": 5,
            "train": 0.7,
            "validation": 0.15,
            "test": 0.15,
            "review_eur": 5.0,
            "seed": 0,
        } | overrides
        return (
            "data:\n"
            f"  url: {v['url']}\n"
            f'  sha256: "{v["sha256"]}"\n'
            f"  max_bytes: {v['max_bytes']}\n"
            f"  timeout_seconds: {v['timeout_seconds']}\n"
            "split:\n"
            f"  train: {v['train']}\n"
            f"  validation: {v['validation']}\n"
            f"  test: {v['test']}\n"
            "costs:\n"
            f"  review_eur: {v['review_eur']}\n"
            f"seed: {v['seed']}\n"
        )

    return make
