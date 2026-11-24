import http.client
import io

import pandas as pd
import pytest
from conftest import parquet_bytes, sha

from fraud_pipeline import config, ingest


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def serve(data):
    return lambda url, timeout: Response(data)


def serve_in_turn(*replies):
    """Each call gets the next reply: bytes to serve, or an exception to raise."""
    calls = []

    def opener(url, timeout):
        reply = replies[len(calls)]
        calls.append(url)
        if isinstance(reply, Exception):
            raise reply
        return Response(reply)

    opener.calls = calls
    return opener


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    waits = []
    monkeypatch.setattr(ingest.time, "sleep", waits.append)
    return waits


def cfg_for(tmp_path, config_text, data, **overrides):
    path = tmp_path / "pipeline.yaml"
    path.write_text(config_text(sha=sha(data), **overrides))
    return config.load(path)[0]


def test_fetch_and_load_round_trip(tmp_path, config_text, table):
    data = parquet_bytes(table)
    cfg = cfg_for(tmp_path, config_text, data)
    dest = ingest.fetch(cfg, tmp_path / "raw" / "d.pq", opener=serve(data))
    assert not list((tmp_path / "raw").glob("*.part"))
    assert len(ingest.load(cfg, dest)) == len(table)


def test_a_different_file_is_refused_and_not_kept(tmp_path, config_text, table):
    cfg = cfg_for(tmp_path, config_text, parquet_bytes(table))
    other = parquet_bytes(table.assign(Amount=table["Amount"] + 1))
    with pytest.raises(ingest.IngestError, match="SHA-256"):
        ingest.fetch(cfg, tmp_path / "d.pq", opener=serve(other))
    assert not list(tmp_path.glob("d.pq*"))


def test_the_size_cap_stops_a_large_download(tmp_path, config_text, table):
    data = parquet_bytes(table)
    cfg = cfg_for(tmp_path, config_text, data, max_bytes=100)
    with pytest.raises(ingest.IngestError, match="cap"):
        ingest.fetch(cfg, tmp_path / "d.pq", opener=serve(data))
    assert not list(tmp_path.glob("d.pq*"))


def test_network_errors_become_ingest_errors(tmp_path, config_text, table):
    cfg = cfg_for(tmp_path, config_text, parquet_bytes(table))

    def down(url, timeout):
        raise TimeoutError("timed out")

    with pytest.raises(ingest.IngestError, match="download failed"):
        ingest.fetch(cfg, tmp_path / "d.pq", opener=down)


def test_load_refuses_a_changed_or_missing_file(tmp_path, config_text, table):
    cfg = cfg_for(tmp_path, config_text, parquet_bytes(table))
    path = tmp_path / "d.pq"
    with pytest.raises(ingest.IngestError, match="missing"):
        ingest.load(cfg, path)
    path.write_bytes(parquet_bytes(table.iloc[:-1]))
    with pytest.raises(ingest.IngestError, match="no longer matches"):
        ingest.load(cfg, path)


def test_a_categorical_string_label_like_openmls_is_handled(tmp_path, config_text, table):
    t = table.copy()
    t["Class"] = pd.Categorical(t["Class"].astype(str), categories=["0", "1"], ordered=True)
    data = parquet_bytes(t)
    cfg = cfg_for(tmp_path, config_text, data)
    path = tmp_path / "d.pq"
    path.write_bytes(data)
    out = ingest.load(cfg, path)
    assert out["Class"].dtype.kind == "i" and out["Class"].sum() == table["Class"].sum()


def test_a_corrupted_copy_is_retried(tmp_path, config_text, table, no_waiting):
    data = parquet_bytes(table)
    cfg = cfg_for(tmp_path, config_text, data)
    opener = serve_in_turn(data[:-100], data)
    assert ingest.fetch(cfg, tmp_path / "d.pq", opener=opener).read_bytes() == data
    assert len(opener.calls) == 2 and no_waiting == [5]


def test_a_broken_connection_is_retried(tmp_path, config_text, table, no_waiting):
    data = parquet_bytes(table)
    cfg = cfg_for(tmp_path, config_text, data)
    opener = serve_in_turn(http.client.IncompleteRead(b"x", 10), TimeoutError(), data)
    assert ingest.fetch(cfg, tmp_path / "d.pq", opener=opener).exists()
    assert no_waiting == [5, 10]


def test_gives_up_after_three_tries_and_says_how_much_arrived(tmp_path, config_text, table):
    data = parquet_bytes(table)
    cfg = cfg_for(tmp_path, config_text, data)
    opener = serve_in_turn(b"abc", b"abc", b"abc", data)
    with pytest.raises(ingest.IngestError, match="3 bytes"):
        ingest.fetch(cfg, tmp_path / "d.pq", opener=opener)
    assert len(opener.calls) == 3 and not list(tmp_path.glob("d.pq*"))


def test_the_size_cap_is_never_retried(tmp_path, config_text, table):
    data = parquet_bytes(table)
    cfg = cfg_for(tmp_path, config_text, data, max_bytes=100)
    opener = serve_in_turn(data, data)
    with pytest.raises(ingest.IngestError, match="cap"):
        ingest.fetch(cfg, tmp_path / "d.pq", opener=opener)
    assert len(opener.calls) == 1
