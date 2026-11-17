import numpy as np
import pytest
from conftest import transactions

from fraud_pipeline import evaluate, train


def test_pr_auc_and_recall_on_a_perfect_ranking():
    y = np.array([0] * 990 + [1] * 10)
    score = y.astype(float)
    assert evaluate.pr_auc(y, score) == 1.0
    assert evaluate.recall_at_budget(y, score, alerts_per_1000=5) == 0.5  # 5 alerts, 10 frauds


def test_recall_at_budget_reviews_at_least_one():
    y = np.array([0] * 99 + [1])
    assert evaluate.recall_at_budget(y, y.astype(float)) == 1.0


def test_metrics_without_frauds_are_nan_not_crashes():
    y = np.zeros(50, int)
    assert np.isnan(evaluate.pr_auc(y, np.random.default_rng(0).uniform(size=50)))
    assert np.isnan(evaluate.recall_at_budget(y, np.zeros(50)))


@pytest.mark.parametrize("y, s", [([], []), ([0, 1], [0.5])])
def test_mismatched_inputs_are_refused(y, s):
    with pytest.raises(ValueError):
        evaluate.pr_auc(y, s)


def test_expected_cost_counts_missed_money_and_reviews():
    y = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.1, 0.8, 0.2])
    amount = np.array([100.0, 40.0, 7.0, 3.0])
    c = evaluate.expected_cost(y, score, amount, threshold=0.5, review_eur=5)
    # flagged: the 100 fraud and the 7 legit -> 2 reviews (10), missed the 40 fraud
    assert c["missed_eur_per_1000"] == 40 * 250
    assert c["review_eur_per_1000"] == 10 * 250
    assert c["frauds_caught"] == 1 and c["alerts_per_1000"] == 500


def test_training_is_reproducible():
    t = transactions(2000, frauds=40)
    a = train.scores(train.fit(train.baseline(0), t), t)
    b = train.scores(train.fit(train.baseline(0), t), t)
    assert np.array_equal(a, b)


def test_git_state_reports_a_commit_or_unknown():
    state = train.git_state()
    assert set(state) == {"commit", "dirty"}
    assert state["commit"] == "unknown" or len(state["commit"]) == 40
