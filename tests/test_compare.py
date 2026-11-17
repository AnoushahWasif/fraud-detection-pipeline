import numpy as np
import pandas as pd
from conftest import transactions
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from fraud_pipeline import compare
from fraud_pipeline.schema import FEATURES


class SpySampler:
    """Records which rows the resampling step is given, and resamples nothing."""

    def __init__(self):
        self.seen = []

    def fit_resample(self, x, y):
        self.seen.append(set(np.asarray(x)[:, FEATURES.index("V1")]))
        return x, y

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def test_resampling_only_sees_training_fold_rows():
    t = transactions(1500, frauds=60)
    t["V1"] = np.arange(len(t), dtype=float)  # a unique id per row, unscaled below
    spy = SpySampler()
    model = Pipeline([("resample", spy), ("model", LogisticRegression(max_iter=2000))])
    x, y = t[FEATURES], t["Class"]
    folds = list(compare.TimeSeriesSplit(n_splits=3).split(x))
    compare.cross_validate(model, x, y, folds=3)
    assert len(spy.seen) == 3
    for seen, (fit_idx, score_idx) in zip(spy.seen, folds, strict=True):
        assert seen == set(x.iloc[fit_idx]["V1"])
        assert not seen & set(x.iloc[score_idx]["V1"])


def test_every_candidate_fits_and_scores():
    t = transactions(1500, frauds=60)
    for name, model in compare.candidates(seed=0).items():
        model.fit(t[FEATURES], t["Class"])
        p = model.predict_proba(t[FEATURES])[:, 1]
        assert ((p >= 0) & (p <= 1)).all(), name


def test_cross_validation_returns_one_score_per_fold():
    t = transactions(2000, frauds=120)
    scores = compare.cross_validate(
        compare.candidates(0)["logistic, class weights"], t[FEATURES], t["Class"]
    )
    assert len(scores) == compare.FOLDS and all(isinstance(s, float) for s in scores)


def test_candidates_are_reproducible():
    t = transactions(1500, frauds=60)
    a = compare.candidates(0)["logistic, SMOTE"].fit(t[FEATURES], t["Class"])
    b = compare.candidates(0)["logistic, SMOTE"].fit(t[FEATURES], t["Class"])
    pd.testing.assert_series_equal(
        pd.Series(a.predict_proba(t[FEATURES])[:, 1]), pd.Series(b.predict_proba(t[FEATURES])[:, 1])
    )
