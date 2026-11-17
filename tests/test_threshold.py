import numpy as np
import pandas as pd
import pytest

from fraud_pipeline import threshold
from fraud_pipeline.evaluate import expected_cost


def test_the_cheapest_threshold_is_chosen():
    y = np.array([1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    amount = np.array([500.0, 10, 10, 2, 10])
    table = threshold.sweep(y, scores, amount, review_eur=5)
    best = threshold.choose(table)
    # Flag only the 500 fraud (one review, 5) and miss the 2 one: 7 in total, the minimum.
    assert best["threshold"] == 0.9
    assert best["cost_per_1000"] == 7 * 200


def test_ties_go_to_the_higher_threshold():
    table = pd.DataFrame({"threshold": [0.1, 0.5, 0.9], "cost_per_1000": [3.0, 1.0, 1.0]})
    assert threshold.choose(table)["threshold"] == 0.9


def test_small_frauds_are_not_worth_a_review():
    y = np.array([1, 1, 0])
    scores = np.array([0.9, 0.8, 0.1])
    amount = np.array([1.0, 2.0, 50.0])
    best = threshold.choose(threshold.sweep(y, scores, amount, review_eur=5))
    assert best["alerts_per_1000"] == 0  # reviewing costs more than the frauds


def test_the_sweep_matches_the_direct_cost_at_every_cut():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=3000) < 0.01).astype(int)
    scores = np.round(rng.uniform(size=3000), 3)  # rounded, so there are ties
    amount = rng.gamma(2, 40, 3000)
    table = threshold.sweep(y, scores, amount, review_eur=5)
    for _, row in table.sample(40, random_state=1).iterrows():
        direct = expected_cost(y, scores, amount, row["threshold"], 5)
        assert row["cost_per_1000"] == pytest.approx(direct["cost_per_1000"])
        assert row["frauds_caught"] == direct["frauds_caught"]
    assert table["threshold"].iloc[-1] == np.inf
    assert len(table) == len(np.unique(scores)) + 1
