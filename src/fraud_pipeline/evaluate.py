"""Metrics for a fraud queue (decision 3) and the cost model (decision 4)."""

import numpy as np
from sklearn.metrics import average_precision_score

ALERTS_PER_1000 = 1.0


def _arrays(y, score):
    y, score = np.asarray(y, int), np.asarray(score, float)
    if len(y) == 0 or len(y) != len(score):
        raise ValueError("need one score per transaction")
    return y, score


def pr_auc(y, score) -> float:
    y, score = _arrays(y, score)
    return float(average_precision_score(y, score)) if y.any() else float("nan")


def recall_at_budget(y, score, alerts_per_1000: float = ALERTS_PER_1000) -> float:
    """Share of frauds among the top-scored transactions when only `alerts_per_1000` of every
    1,000 can be reviewed."""
    y, score = _arrays(y, score)
    if not y.any():
        return float("nan")
    k = max(1, int(round(len(y) * alerts_per_1000 / 1000)))
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def expected_cost(y, score, amount, threshold: float, review_eur: float) -> dict:
    """Money lost to missed frauds plus reviews, per 1,000 transactions, at one threshold."""
    y, score = _arrays(y, score)
    amount = np.asarray(amount, float)
    flagged = score >= threshold
    missed = float(amount[(y == 1) & ~flagged].sum())
    reviews = float(flagged.sum() * review_eur)
    per = 1000 / len(y)
    return {
        "cost_per_1000": (missed + reviews) * per,
        "missed_eur_per_1000": missed * per,
        "review_eur_per_1000": reviews * per,
        "alerts_per_1000": float(flagged.sum()) * per,
        "frauds_caught": int(((y == 1) & flagged).sum()),
        "frauds": int(y.sum()),
    }


def summary(y, score) -> dict:
    return {"pr_auc": pr_auc(y, score), "recall_at_1_per_1000": recall_at_budget(y, score)}
