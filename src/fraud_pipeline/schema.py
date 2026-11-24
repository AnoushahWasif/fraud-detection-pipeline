"""What a transactions table must look like before anything trains on it or scores it
(decision 7). A table that fails stops the pipeline."""

import pandas as pd
import pandera.pandas as pa

FEATURES = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]
TARGET = "Class"
# Two days of transactions: Time is seconds since the first one.
MAX_TIME = 2 * 24 * 3600

_number = {"dtype": float, "nullable": False, "coerce": True}
TRANSACTIONS = pa.DataFrameSchema(
    {
        "Time": pa.Column(checks=pa.Check.in_range(0, MAX_TIME), **_number),
        **{
            f"V{i}": pa.Column(checks=pa.Check(lambda s: s.abs() < 1e3), **_number)
            for i in range(1, 29)
        },
        "Amount": pa.Column(checks=pa.Check.ge(0), **_number),
        TARGET: pa.Column(int, pa.Check.isin([0, 1]), nullable=False, coerce=True),
    },
    strict=True,  # no extra columns, so nothing unexpected can reach the model as a feature
    ordered=True,
    checks=pa.Check(lambda df: len(df) > 0, error="the table is empty"),
)


class SchemaError(ValueError):
    pass


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """The table with dtypes coerced, or SchemaError listing every failed check."""
    try:
        return TRANSACTIONS.validate(df, lazy=True)
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as exc:
        cases = getattr(exc, "failure_cases", None)
        summary = (
            cases[["column", "check"]].drop_duplicates().head(10).to_dict("records")
            if cases is not None
            else str(exc)
        )
        raise SchemaError(f"transactions failed validation: {summary}") from exc
