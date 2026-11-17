import numpy as np
import pandas as pd

from fraud_pipeline import threshold


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


def test_candidate_thresholds_are_capped():
    scores = np.random.default_rng(0).uniform(size=50_000)
    cuts = threshold.candidate_thresholds(scores)
    assert len(cuts) <= 2002 and cuts.min() == scores.min() and cuts[-2] == scores.max()
    assert cuts[-1] == np.inf
