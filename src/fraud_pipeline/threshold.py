"""Day 7: the alert threshold, chosen by cost on validation (decision 4).

    uv run python -m fraud_pipeline.threshold

Fits the chosen model (gradient boosting with class weights, day 6) on the training split,
picks the threshold with the lowest expected cost per 1,000 validation transactions, and saves
the model to models/fraud with the threshold in its metadata. This is what the API serves.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_pipeline import compare, ingest, split, train
from fraud_pipeline import config as config_module
from fraud_pipeline.evaluate import expected_cost, summary
from fraud_pipeline.model_store import save_model
from fraud_pipeline.schema import FEATURES, TARGET

CHOSEN = "boosting, class weights"


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Every distinct score is a possible cut (at most 2,000, spread by quantile), plus infinity:
    flagging nothing has to be an option, since reviews can cost more than the frauds."""
    unique = np.unique(scores)
    if len(unique) > 2000:
        unique = np.unique(np.quantile(scores, np.linspace(0, 1, 2001)))
    return np.append(unique, np.inf)


def sweep(y, scores, amount, review_eur: float) -> pd.DataFrame:
    rows = [
        {"threshold": t, **expected_cost(y, scores, amount, t, review_eur)}
        for t in candidate_thresholds(scores)
    ]
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame) -> pd.Series:
    """The cheapest threshold; among equally cheap ones, the highest (fewest alerts)."""
    best = table["cost_per_1000"].min()
    return table[np.isclose(table["cost_per_1000"], best)].iloc[-1]


def main(out_dir: Path = Path("models/fraud"), report: Path = Path("reports/threshold.json")):
    cfg, config_sha = config_module.load()
    splits = split.by_time(ingest.load(cfg), cfg.split)
    split.check_no_leakage(splits, FEATURES)
    model = compare.candidates(cfg.seed)[CHOSEN].fit(splits.train[FEATURES], splits.train[TARGET])
    val = splits.validation
    y, amount = val[TARGET].to_numpy(), val["Amount"].to_numpy()
    scores = model.predict_proba(val[FEATURES])[:, 1]
    table = sweep(y, scores, amount, cfg.costs.review_eur)
    best = choose(table)
    nothing = expected_cost(y, scores, amount, np.inf, cfg.costs.review_eur)
    half = expected_cost(y, scores, amount, 0.5, cfg.costs.review_eur)
    flagged = scores >= best["threshold"]
    missed = val[(y == 1) & ~flagged]
    result = {
        "model": CHOSEN,
        "threshold": float(best["threshold"]),
        "validation": {
            **{k: round(v, 4) for k, v in summary(y, scores).items()},
            "at_threshold": {k: round(float(v), 2) for k, v in best.drop("threshold").items()},
            "flag_nothing": {k: round(float(v), 2) for k, v in nothing.items()},
            "threshold_0.5": {k: round(float(v), 2) for k, v in half.items()},
            "missed_fraud_amounts": sorted(round(float(a), 2) for a in missed["Amount"]),
        },
    }
    meta = train.metadata(cfg, config_sha, result)
    # Model first, so the report never names a hash that wasn't saved.
    meta["model_sha256"] = save_model(model, out_dir, meta)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(meta, indent=2))
    table.to_csv(report.with_suffix(".csv"), index=False)
    return meta


if __name__ == "__main__":
    print(json.dumps(main()["validation"], indent=2))
