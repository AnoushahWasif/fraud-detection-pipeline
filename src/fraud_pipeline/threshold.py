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


def sweep(y, scores, amount, review_eur: float) -> pd.DataFrame:
    """Expected cost per 1,000 transactions at every distinct score as the cut (flag score >= cut),
    plus infinity, since flagging nothing has to be an option when reviews cost more than the
    frauds. One sort and running totals, so every score is tried, not a sample of them: the
    decision happens in the top fraction of a percent, where a sample has few points."""
    y = np.asarray(y, int)
    scores, amount = np.asarray(scores, float), np.asarray(amount, float)
    order = np.argsort(-scores, kind="stable")
    s, fraud_eur, is_fraud = scores[order], (amount * y)[order], y[order]
    # Last position of each distinct score, so ties are flagged together.
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    flagged = ends + 1
    caught_eur = np.cumsum(fraud_eur)[ends]
    caught = np.cumsum(is_fraud)[ends]
    per = 1000 / len(y)
    total_eur, frauds = fraud_eur.sum(), int(y.sum())
    table = pd.DataFrame(
        {
            "threshold": np.r_[s[ends], np.inf],
            "missed_eur_per_1000": np.r_[total_eur - caught_eur, total_eur] * per,
            "review_eur_per_1000": np.r_[flagged * review_eur, 0.0] * per,
            "alerts_per_1000": np.r_[flagged, 0] * per,
            "frauds_caught": np.r_[caught, 0],
            "frauds": frauds,
        }
    )
    table["cost_per_1000"] = table["missed_eur_per_1000"] + table["review_eur_per_1000"]
    return table.sort_values("threshold").reset_index(drop=True)


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
