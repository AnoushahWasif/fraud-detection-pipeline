"""Day 12: the saved model on the test split, scored once (decision 2).

    uv run python -m fraud_pipeline.holdout

Uses the exact model and threshold the API serves (models/fraud, hash-checked). A second run is
refused while reports/holdout.json exists: tuning after seeing these numbers would make the test
split one more validation set.
"""

import json
from pathlib import Path

import numpy as np

from fraud_pipeline import config as config_module
from fraud_pipeline import ingest, split, train
from fraud_pipeline.evaluate import expected_cost, pr_auc, recall_at_budget
from fraud_pipeline.model_store import load_model
from fraud_pipeline.schema import FEATURES, TARGET

N_BOOT = 2000


def bootstrap(y, scores, amount, threshold, review_eur, n=N_BOOT, seed=0) -> dict:
    """95% intervals, resampling transactions. With about 50 frauds they're wide, on purpose."""
    rng = np.random.default_rng(seed)
    y, scores, amount = np.asarray(y), np.asarray(scores), np.asarray(amount)
    stats = {"pr_auc": [], "recall_at_1_per_1000": [], "cost_per_1000": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if not y[i].any():
            continue
        stats["pr_auc"].append(pr_auc(y[i], scores[i]))
        stats["recall_at_1_per_1000"].append(recall_at_budget(y[i], scores[i]))
        stats["cost_per_1000"].append(
            expected_cost(y[i], scores[i], amount[i], threshold, review_eur)["cost_per_1000"]
        )
    return {
        k: [round(float(v), 4) for v in np.percentile(s, [2.5, 97.5])] for k, s in stats.items()
    }


def main(out: Path = Path("reports/holdout.json"), model_dir: Path = Path("models/fraud")):
    if out.exists():
        raise RuntimeError(
            f"{out} exists, so the test split has been scored already. Delete it only to score "
            "a model chosen without looking at these results."
        )
    cfg, config_sha = config_module.load()
    loaded = load_model(model_dir)
    if loaded.manifest["data_sha256"] != cfg.data.sha256:
        raise RuntimeError("the saved model was trained on different data than the config names")
    splits = split.by_time(ingest.load(cfg), cfg.split)
    split.check_no_leakage(splits, FEATURES)
    test = splits.test
    y, amount = test[TARGET].to_numpy(), test["Amount"].to_numpy()
    scores = loaded.model.predict_proba(test[FEATURES])[:, 1]
    threshold = float(loaded.manifest["threshold"])
    at = expected_cost(y, scores, amount, threshold, cfg.costs.review_eur)
    nothing = expected_cost(y, scores, amount, np.inf, cfg.costs.review_eur)
    report = {
        "split": "test",
        "transactions": len(test),
        "frauds": int(y.sum()),
        "model_sha256": loaded.manifest["sha256"],
        "threshold": threshold,
        "pr_auc": round(pr_auc(y, scores), 4),
        "recall_at_1_per_1000": round(recall_at_budget(y, scores), 4),
        "at_threshold": {k: round(float(v), 2) for k, v in at.items()},
        "flag_nothing": {k: round(float(v), 2) for k, v in nothing.items()},
        "intervals_95": bootstrap(y, scores, amount, threshold, cfg.costs.review_eur),
        "missed_fraud_amounts": sorted(
            round(float(a), 2) for a in amount[(y == 1) & (scores < threshold)]
        ),
        "run": {
            "git": train.git_state(),
            "config_sha256": config_sha,
            "data_sha256": cfg.data.sha256,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(out)
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
