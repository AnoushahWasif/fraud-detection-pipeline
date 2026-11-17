"""Day 6: class weights against resampling, by time-ordered cross-validation (decision 5).

    uv run python -m fraud_pipeline.compare

Resamplers sit inside an imblearn Pipeline, which applies them only when fitting, so they only
ever see the training rows of each fold, never the rows being scored.
"""

import json
import time
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from fraud_pipeline import config as config_module
from fraud_pipeline import ingest, split, tracking, train
from fraud_pipeline.evaluate import summary
from fraud_pipeline.schema import FEATURES, TARGET

FOLDS = 5
# Undersampling keeps 10 legitimate transactions per fraud; SMOTE raises frauds to 10% of
# legitimate ones. Both are common starting points, not tuned.
UNDER_RATIO, SMOTE_RATIO = 0.1, 0.1


def logistic(weight=None):
    return LogisticRegression(class_weight=weight, max_iter=5000)


def boosting(weight=None):
    return HistGradientBoostingClassifier(
        class_weight=weight, learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
        early_stopping=False,
    )  # fmt: skip


def candidates(seed: int) -> dict:
    def pipe(model, sampler=None):
        steps = [("scale", StandardScaler())]
        if sampler is not None:
            steps.append(("resample", sampler))
        return Pipeline([*steps, ("model", model)])

    under = RandomUnderSampler(sampling_strategy=UNDER_RATIO, random_state=seed)
    smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=seed)
    out = {}
    for family, make in (("logistic", logistic), ("boosting", boosting)):
        out[f"{family}, no weighting"] = pipe(make())
        out[f"{family}, class weights"] = pipe(make("balanced"))
        out[f"{family}, undersampling"] = pipe(make(), under)
        out[f"{family}, SMOTE"] = pipe(make(), smote)
    for model in out.values():
        model.set_params(model__random_state=seed)
    return out


def cross_validate(model, x, y, folds: int = FOLDS) -> list[float]:
    scores = []
    for fit_idx, score_idx in TimeSeriesSplit(n_splits=folds).split(x):
        model.fit(x.iloc[fit_idx], y.iloc[fit_idx])
        p = model.predict_proba(x.iloc[score_idx])[:, 1]
        scores.append(summary(y.iloc[score_idx], p)["pr_auc"])
    return scores


def main(out: Path = Path("reports/compare.json")) -> dict:
    cfg, config_sha = config_module.load()
    splits = split.by_time(ingest.load(cfg), cfg.split)
    split.check_no_leakage(splits, FEATURES)
    x, y = splits.train[FEATURES], splits.train[TARGET]
    xv, yv = splits.validation[FEATURES], splits.validation[TARGET]
    results = {}
    for name, model in candidates(cfg.seed).items():
        start = time.perf_counter()
        folds = cross_validate(model, x, y)
        model.fit(x, y)
        val = summary(yv, model.predict_proba(xv)[:, 1])
        results[name] = {
            "cv_pr_auc_folds": [round(s, 4) for s in folds],
            "cv_pr_auc_mean": round(float(np.mean(folds)), 4),
            "cv_pr_auc_sd": round(float(np.std(folds)), 4),
            "validation": {k: round(v, 4) for k, v in val.items()},
            "seconds": round(time.perf_counter() - start, 1),
        }
        print(name, results[name]["cv_pr_auc_mean"], results[name]["validation"])
    report = {"config_sha256": config_sha, "folds": FOLDS, "results": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    tags = tracking.tags_from(train.metadata(cfg, config_sha, {}))
    for name, result in results.items():
        params = {"candidate": name, "folds": FOLDS, "seed": cfg.seed}
        tracking.log_run(f"compare: {name}", params, result, tags | {"stage": "compare"})
    return report


if __name__ == "__main__":
    main()
