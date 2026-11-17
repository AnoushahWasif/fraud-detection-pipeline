import json
import logging

import pytest
from conftest import transactions
from fastapi.testclient import TestClient

from fraud_pipeline import service, train
from fraud_pipeline.model_store import MODEL_FILE, save_model
from fraud_pipeline.schema import FEATURES

KEY = "k" * 40


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    t = transactions(3000, frauds=80)
    model = train.fit(train.baseline(0), t)
    folder = tmp_path_factory.mktemp("model")
    save_model(model, folder, {"threshold": 0.5})
    return folder


@pytest.fixture
def client(model_dir):
    return TestClient(service.create_app(service.Settings(model_dir, KEY)))


def body(n=2):
    rows = transactions(n, frauds=1, seed=3)[FEATURES].to_dict("records")
    return {"transactions": rows}


def post(client, payload, key=KEY, **kw):
    headers = {"X-API-Key": key} if key else {}
    return client.post("/score", json=payload, headers=headers, **kw)


def test_scores_and_alerts(client):
    r = post(client, body(3))
    assert r.status_code == 200
    out = r.json()
    assert len(out["results"]) == 3 and out["threshold"] == 0.5
    for item in out["results"]:
        assert 0 <= item["score"] <= 1 and item["alert"] == (item["score"] >= 0.5)
    assert len(out["model_version"]) == 12
    assert "X-Request-ID" in r.headers


def test_health_needs_no_key(client):
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("key", [None, "wrong", "k" * 39])
def test_bad_keys_are_refused(client, key):
    r = post(client, body(), key=key)
    assert r.status_code == 401 and "API key" in r.json()["detail"]


def bad(change):
    b = body()
    change(b["transactions"][0])
    return b


BAD = {
    "extra field": bad(lambda t: t.__setitem__("card_number", "4111")),
    "missing field": bad(lambda t: t.pop("V3")),
    "negative amount": bad(lambda t: t.__setitem__("Amount", -1)),
    "time past two days": bad(lambda t: t.__setitem__("Time", 1e7)),
    "absurd component": bad(lambda t: t.__setitem__("V2", 5e3)),
    "text": bad(lambda t: t.__setitem__("V1", "abc")),
    "empty batch": {"transactions": []},
    "too many": {"transactions": body(1)["transactions"] * (service.MAX_BATCH + 1)},
    "extra top-level key": {**body(), "debug": True},
}


@pytest.mark.parametrize("name", BAD)
def test_bad_input_is_refused(client, name):
    assert post(client, BAD[name]).status_code == 422


def test_non_finite_numbers_are_refused(client):
    text = json.dumps(body()).replace(str(body()["transactions"][0]["V4"]), "NaN", 1)
    r = client.post(
        "/score", content=text, headers={"X-API-Key": KEY, "Content-Type": "application/json"}
    )
    assert r.status_code == 422


def test_large_bodies_are_refused_before_parsing(client):
    big = b"{" + b" " * (service.MAX_BODY_BYTES + 1) + b"}"
    r = client.post(
        "/score", content=big, headers={"X-API-Key": KEY, "Content-Type": "application/json"}
    )
    assert r.status_code == 413


def test_a_body_without_a_declared_length_is_refused(client):
    def chunks():
        yield b'{"transactions": []}'

    r = client.post("/score", content=chunks(), headers={"X-API-Key": KEY})
    assert r.status_code == 411


def test_logs_never_contain_transaction_values(client, caplog):
    payload = body(1)
    marker = payload["transactions"][0]["V7"]
    with caplog.at_level(logging.INFO, logger="fraud_pipeline.service"):
        assert post(client, payload).status_code == 200
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "scored" in text and str(marker) not in text and KEY not in text


def test_refuses_to_start_with_a_short_key(model_dir):
    with pytest.raises(RuntimeError, match="at least"):
        service.Settings(model_dir, "short")


def test_refuses_to_start_with_a_swapped_model(model_dir, tmp_path):
    import shutil

    copy = tmp_path / "m"
    shutil.copytree(model_dir, copy)
    (copy / MODEL_FILE).write_bytes(b"not a model")
    with pytest.raises(Exception, match="hash"):
        service.Settings(copy, KEY)


def test_a_pinned_hash_beats_the_manifest(model_dir):
    with pytest.raises(Exception, match="hash"):
        service.Settings(model_dir, KEY, pinned_sha="0" * 64)


def test_from_environment_needs_a_key(monkeypatch, model_dir):
    monkeypatch.setenv("FRAUD_MODEL_DIR", str(model_dir))
    monkeypatch.delenv("FRAUD_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRAUD_API_KEY"):
        service.from_environment()
    monkeypatch.setenv("FRAUD_API_KEY", KEY)
    assert service.from_environment() is not None


def test_no_docs_or_schema_endpoints(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_errors_do_not_echo_transaction_values(client):
    payload = BAD["negative amount"]
    r = post(client, payload)
    assert r.status_code == 422
    assert "input" not in r.text and str(payload["transactions"][0]["V7"]) not in r.text


def test_a_full_batch_fits_under_the_size_cap(client):
    assert post(client, body(service.MAX_BATCH)).status_code == 200
