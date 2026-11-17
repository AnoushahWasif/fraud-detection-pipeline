import pandas as pd
import pytest
from conftest import transactions

from fraud_pipeline.config import Split
from fraud_pipeline.split import by_time, report

FRACTIONS = Split(train=0.7, validation=0.15, test=0.15)


def test_splits_are_in_time_order_and_cover_everything(table):
    s = by_time(table, FRACTIONS)
    assert s.train["Time"].max() < s.validation["Time"].min()
    assert s.validation["Time"].max() < s.test["Time"].min()
    assert len(s.train) + len(s.validation) + len(s.test) == len(table)
    assert abs(len(s.train) / len(table) - 0.7) < 0.02


def test_equal_timestamps_never_straddle_a_boundary():
    t = transactions(300)
    t["Time"] = (t.index // 10) * 60.0  # ten transactions share each timestamp
    s = by_time(t, FRACTIONS)
    assert not set(s.train["Time"]) & set(s.validation["Time"])
    assert not set(s.validation["Time"]) & set(s.test["Time"])


def test_duplicates_stay_in_one_split_and_are_counted():
    t = transactions(300)
    t = pd.concat([t, t.iloc[[5, 6, 290]]], ignore_index=True).sort_values("Time")
    r = report(by_time(t, FRACTIONS))
    assert r["rows_shared_across_splits"] == 0
    assert sum(v["duplicate_rows"] for v in r["splits"].values()) == 3


def test_report_counts_frauds(table):
    r = report(by_time(table, FRACTIONS))
    assert sum(v["frauds"] for v in r["splits"].values()) == int(table["Class"].sum())


@pytest.mark.parametrize(
    "make",
    [
        lambda t: t.iloc[:0],
        lambda t: t.assign(Time=100.0),  # a single timestamp
    ],
)
def test_tables_that_cannot_be_split_by_time_are_refused(table, make):
    with pytest.raises(ValueError):
        by_time(make(table), FRACTIONS)
