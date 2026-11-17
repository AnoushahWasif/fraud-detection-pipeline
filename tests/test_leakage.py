"""Leakage checks (decision 5 and day 5). The last test runs on the real data when it has been
downloaded, and is skipped otherwise (CI has no copy of it)."""

import numpy as np
import pandas as pd
import pytest
from conftest import transactions

from fraud_pipeline import config, ingest, train
from fraud_pipeline.config import Split
from fraud_pipeline.schema import FEATURES
from fraud_pipeline.split import LeakageError, Splits, by_time, check_no_leakage

FRACTIONS = Split(train=0.7, validation=0.15, test=0.15)


@pytest.fixture
def splits():
    return by_time(transactions(3000, frauds=60), FRACTIONS)


def test_a_clean_split_passes(splits):
    check_no_leakage(splits, FEATURES)


def test_the_label_as_a_feature_is_caught(splits):
    with pytest.raises(LeakageError, match="label"):
        check_no_leakage(splits, [*FEATURES, "Class"])


def test_overlapping_time_is_caught(splits):
    shifted = splits.test.assign(Time=splits.test["Time"] - 1e5)
    with pytest.raises(LeakageError, match="time"):
        check_no_leakage(Splits(splits.train, splits.validation, shifted), FEATURES)


def test_a_shared_transaction_is_caught(splits):
    # The same row in train and test, with test's own times otherwise intact.
    copied = splits.train.iloc[[0]].assign(Time=splits.test["Time"].max())
    leaky_train = pd.concat([splits.train, copied])
    test = pd.concat([splits.test, copied])
    with pytest.raises(LeakageError):
        check_no_leakage(Splits(leaky_train, splits.validation, test), FEATURES)


def test_the_model_never_sees_validation_or_test(splits):
    a = train.fit(train.baseline(0), splits.train)
    scrambled = Splits(
        splits.train,
        splits.validation.assign(Class=1 - splits.validation["Class"], V14=0.0),
        splits.test.sample(frac=1, random_state=1),
    )
    b = train.fit(train.baseline(0), scrambled.train)
    probe = splits.test
    assert np.array_equal(train.scores(a, probe), train.scores(b, probe))


def test_the_label_never_reaches_the_model(splits):
    model = train.fit(train.baseline(0), splits.train)
    assert list(model.feature_names_in_) == FEATURES
    flipped = splits.test.assign(Class=1 - splits.test["Class"])
    assert np.array_equal(train.scores(model, splits.test), train.scores(model, flipped))


def test_the_real_split_is_clean():
    cfg, _ = config.load()
    if not ingest.raw_path(cfg).exists():
        pytest.skip("real data not downloaded")
    s = by_time(ingest.load(cfg), cfg.split)
    check_no_leakage(s, FEATURES)
