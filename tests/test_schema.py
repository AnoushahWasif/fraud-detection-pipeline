import numpy as np
import pytest

from fraud_pipeline.schema import SchemaError, validate


def test_good_table_passes_and_label_is_coerced(table):
    table["Class"] = table["Class"].astype(str)
    out = validate(table)
    assert out["Class"].dtype.kind == "i" and set(out["Class"]) == {0, 1}


def broken(table, change):
    change(table)
    return table


BAD = {
    "missing column": lambda t: t.pop("V7"),
    "extra column": lambda t: t.insert(1, "card_number", "4111"),
    "negative amount": lambda t: t.__setitem__("Amount", -t["Amount"]),
    "missing value": lambda t: t.loc.__setitem__((3, "V2"), np.nan),
    "label 2": lambda t: t.loc.__setitem__((0, "Class"), 2),
    "time past two days": lambda t: t.loc.__setitem__((0, "Time"), 200_000.0),
    "text in a feature": lambda t: t.__setitem__("V1", "x"),
    "absurd component": lambda t: t.loc.__setitem__((0, "V5"), 1e6),
    "columns out of order": lambda t: t.insert(0, "Amount", t.pop("Amount")),
}


@pytest.mark.parametrize("name", BAD)
def test_bad_tables_are_refused(table, name):
    with pytest.raises(SchemaError):
        validate(broken(table, BAD[name]))


def test_an_empty_table_is_refused(table):
    with pytest.raises(SchemaError):
        validate(table.iloc[:0])
