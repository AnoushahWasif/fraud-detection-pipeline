import pytest

from fraud_pipeline import tracking


def test_flat_turns_nested_metrics_into_keys_and_drops_non_numbers():
    out = tracking.flat(
        "", {"validation": {"pr_auc": 0.8, "at": {"cost": 9}}, "name": "x", "ok": True}
    )
    assert out == {"validation.pr_auc": 0.8, "validation.at.cost": 9.0}


def test_tracking_can_be_switched_off(monkeypatch, capsys):
    monkeypatch.setenv("FRAUD_MLFLOW", "off")
    assert tracking.log_run("x", {}, {}, {}) is None
    assert "not logged" in capsys.readouterr().out


def test_a_run_is_logged_to_a_local_store(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    import mlflow

    monkeypatch.delenv("FRAUD_MLFLOW", raising=False)
    store = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    report = tmp_path / "r.json"
    report.write_text("{}")
    run_id = tracking.log_run(
        "test", {"model": "m"}, {"validation": {"pr_auc": 0.5}}, {"stage": "t"}, report, store
    )
    run = mlflow.MlflowClient(store).get_run(run_id)
    assert run.data.metrics["validation.pr_auc"] == 0.5
    assert run.data.params["model"] == "m" and run.data.tags["stage"] == "t"
