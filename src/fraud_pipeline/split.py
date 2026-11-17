"""Day 3: train, validation and test by time (decision 2).

The cut points are timestamps, not row positions, so transactions sharing a timestamp always
land in the same split. Exact duplicate rows share their timestamp, so a duplicate can never
straddle a boundary, and `report` checks that rather than assuming it.
"""

from dataclasses import dataclass

import pandas as pd

from fraud_pipeline.config import Split
from fraud_pipeline.schema import TARGET


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def items(self):
        return {"train": self.train, "validation": self.validation, "test": self.test}.items()


def by_time(df: pd.DataFrame, fractions: Split) -> Splits:
    if df.empty:
        raise ValueError("nothing to split")
    first = df["Time"].quantile(fractions.train)
    second = df["Time"].quantile(fractions.train + fractions.validation)
    if not first < second:
        raise ValueError("too few distinct timestamps to split by time")
    early, late = df["Time"] < first, df["Time"] >= second
    out = Splits(df[early], df[~early & ~late], df[late])
    if any(part.empty for _, part in out.items()):
        raise ValueError("a split came out empty")
    return out


def report(splits: Splits) -> dict:
    """Size, frauds, time range and duplicates per split, and duplicates shared across splits."""
    rows = {
        name: {
            "transactions": len(part),
            "frauds": int(part[TARGET].sum()),
            "fraud_rate": round(float(part[TARGET].mean()), 5),
            "hours": [round(part["Time"].min() / 3600, 2), round(part["Time"].max() / 3600, 2)],
            "duplicate_rows": int(part.duplicated().sum()),
        }
        for name, part in splits.items()
    }
    keys = {name: set(map(tuple, part.to_numpy())) for name, part in splits.items()}
    shared = len(keys["train"] & keys["validation"]) + len(keys["train"] & keys["test"])
    shared += len(keys["validation"] & keys["test"])
    return {"splits": rows, "rows_shared_across_splits": shared}
