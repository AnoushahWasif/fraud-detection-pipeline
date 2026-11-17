import numpy as np
import pytest

from fraud_pipeline import holdout


def test_bootstrap_intervals_bracket_sensible_values():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=5000) < 0.01).astype(int)
    scores = np.clip(y * 0.6 + rng.uniform(0, 0.5, 5000), 0, 1)
    amount = rng.gamma(2, 40, 5000)
    ci = holdout.bootstrap(y, scores, amount, threshold=0.55, review_eur=5, n=200)
    for name in ("pr_auc", "recall_at_1_per_1000", "cost_per_1000"):
        low, high = ci[name]
        assert low <= high
    assert 0 <= ci["pr_auc"][0] <= ci["pr_auc"][1] <= 1


def test_a_second_run_is_refused(tmp_path):
    out = tmp_path / "holdout.json"
    out.write_text("{}")
    with pytest.raises(RuntimeError, match="scored already"):
        holdout.main(out)
