"""Smoke test against a running scoring service (local uvicorn or the Docker container).

    FRAUD_API_KEY=... uv run python scripts/smoke.py http://127.0.0.1:8000

Sends 5 real validation frauds and 5 legitimate transactions and checks that the service
answers, refuses a request without the key, and alerts on more frauds than legitimate ones.
Exits non-zero on any failure, so CI fails with it.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

from fraud_pipeline import config, ingest, split
from fraud_pipeline.schema import FEATURES, TARGET


def call(url: str, body: dict | None = None, key: str | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    request.add_header("Content-Type", "application/json")
    if key:
        request.add_header("X-API-Key", key)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def wait_for(base: str, seconds: int = 60) -> None:
    for _ in range(seconds):
        try:
            if call(f"{base}/health")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(1)
    sys.exit(f"service at {base} didn't become healthy in {seconds} s")


def main(base: str) -> None:
    key = os.environ["FRAUD_API_KEY"]
    wait_for(base)
    cfg, _ = config.load()
    val = split.by_time(ingest.load(cfg), cfg.split).validation
    frauds = val[val[TARGET] == 1].head(5)
    legit = val[val[TARGET] == 0].head(5)
    rows = [*frauds[FEATURES].to_dict("records"), *legit[FEATURES].to_dict("records")]

    status, _ = call(f"{base}/score", {"transactions": rows})
    if status != 401:
        sys.exit(f"a request without the key got {status}, expected 401")
    status, out = call(f"{base}/score", {"transactions": rows}, key)
    if status != 200 or len(out.get("results", [])) != len(rows):
        sys.exit(f"scoring failed: {status} {out}")
    alerts = [r["alert"] for r in out["results"]]
    fraud_alerts, legit_alerts = sum(alerts[:5]), sum(alerts[5:])
    print(
        f"model {out['model_version']}: {fraud_alerts}/5 frauds and {legit_alerts}/5 legit alerted"
    )
    if fraud_alerts <= legit_alerts:
        sys.exit("the model doesn't separate real frauds from legitimate transactions")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
