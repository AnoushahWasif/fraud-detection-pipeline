"""Day 2: download the pinned dataset once, check it, and load it validated (decision 1).

uv run python -m fraud_pipeline.ingest
"""

import hashlib
import hmac
import os
import urllib.request
from pathlib import Path

import pandas as pd

from fraud_pipeline import config as config_module
from fraud_pipeline.schema import TARGET, validate

CHUNK = 1 << 20


class IngestError(RuntimeError):
    pass


def data_dir() -> Path:
    return Path(os.environ.get("FRAUD_DATA_DIR", "data"))


def raw_path(cfg: config_module.Config) -> Path:
    return data_dir() / "raw" / Path(cfg.data.url.path or "dataset").name


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def fetch(cfg: config_module.Config, dest: Path, opener=urllib.request.urlopen) -> Path:
    """Download to a .part file, refusing more than max_bytes, then keep it only if its hash
    matches the config."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    size = 0
    try:
        with (
            opener(str(cfg.data.url), timeout=cfg.data.timeout_seconds) as response,
            tmp.open("wb") as out,
        ):
            while block := response.read(CHUNK):
                size += len(block)
                if size > cfg.data.max_bytes:
                    raise IngestError(f"download passed the {cfg.data.max_bytes:,}-byte cap")
                out.write(block)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise IngestError(f"download failed: {exc}") from exc
    except IngestError:
        tmp.unlink(missing_ok=True)
        raise
    actual = sha256_of(tmp)
    if not hmac.compare_digest(actual, cfg.data.sha256):
        tmp.unlink(missing_ok=True)
        raise IngestError(f"downloaded file has SHA-256 {actual}, config expects {cfg.data.sha256}")
    tmp.replace(dest)
    return dest


def load(cfg: config_module.Config, path: Path | None = None) -> pd.DataFrame:
    """The cached table, only if its hash still matches, with the schema enforced."""
    path = path or raw_path(cfg)
    if not path.exists():
        raise IngestError(f"{path} is missing; run `make data`")
    if not hmac.compare_digest(sha256_of(path), cfg.data.sha256):
        raise IngestError(f"{path} no longer matches the SHA-256 in the config")
    df = pd.read_parquet(path)
    # OpenML stores the label as a categorical of the strings "0" and "1".
    if TARGET in df and isinstance(df[TARGET].dtype, pd.CategoricalDtype):
        df[TARGET] = df[TARGET].astype(str)
    return validate(df)


def main() -> None:
    cfg, _ = config_module.load()
    path = raw_path(cfg)
    if path.exists() and hmac.compare_digest(sha256_of(path), cfg.data.sha256):
        print(f"{path} already downloaded and verified")
    else:
        fetch(cfg, path)
        print(f"downloaded and verified {path}")
    df = load(cfg, path)
    print(f"{len(df):,} transactions, {int(df[TARGET].sum()):,} frauds, schema passed")


if __name__ == "__main__":
    main()
