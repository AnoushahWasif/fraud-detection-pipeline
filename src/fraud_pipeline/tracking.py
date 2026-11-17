"""Day 8: MLflow tracking on a local SQLite store (decision 9). No server, nothing leaves the
machine. MLflow is an optional extra (`uv sync --extra tracking`), so the scoring image doesn't
carry it; without it, or with FRAUD_MLFLOW=off, runs still work and just aren't logged.
"""

import os
from pathlib import Path

EXPERIMENT = "fraud-detection"
STORE = "sqlite:///mlflow.db"


def enabled() -> bool:
    if os.environ.get("FRAUD_MLFLOW", "on").lower() in {"off", "0", "false"}:
        return False
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def flat(prefix: str, values: dict) -> dict:
    """Nested metrics as MLflow keys: {"validation": {"pr_auc": 0.8}} -> validation.pr_auc."""
    out = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out |= flat(name, value)
        elif isinstance(value, int | float) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def log_run(
    name: str, params: dict, metrics: dict, tags: dict, artifact: Path | None = None,
    store: str | None = None,
) -> str | None:  # fmt: skip
    """One MLflow run; returns its id, or None when tracking is off."""
    if not enabled():
        print("MLflow tracking off (not installed or FRAUD_MLFLOW=off); run not logged")
        return None
    import mlflow

    mlflow.set_tracking_uri(store or os.environ.get("FRAUD_MLFLOW_URI", STORE))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.log_metrics(flat("", metrics))
        mlflow.set_tags({k: str(v) for k, v in tags.items()})
        if artifact is not None and artifact.exists():
            mlflow.log_artifact(str(artifact))
        return run.info.run_id


def tags_from(meta: dict) -> dict:
    return {
        "data_sha256": meta["data_sha256"],
        "config_sha256": meta["config_sha256"],
        "git_commit": meta["git"]["commit"],
        "git_dirty": meta["git"]["dirty"],
        "sklearn": meta["sklearn"],
    }
