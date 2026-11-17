"""The pipeline's settings, from config/pipeline.yaml (decision 6).

Unknown keys are refused, so a typo fails loudly instead of quietly falling back to a default.
"""

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

DEFAULT = Path("config/pipeline.yaml")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Data(Strict):
    url: HttpUrl
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_bytes: int = Field(gt=0, le=1024**3)
    timeout_seconds: float = Field(gt=0, le=600)

    @model_validator(mode="after")
    def https_only(self):
        if self.url.scheme != "https":
            raise ValueError("data.url must use https")
        return self


class Split(Strict):
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def sums_to_one(self):
        total = self.train + self.validation + self.test
        if abs(total - 1) > 1e-9:
            raise ValueError(f"split fractions add up to {total}, not 1")
        return self


class Costs(Strict):
    review_eur: float = Field(gt=0)


class Config(Strict):
    data: Data
    split: Split
    costs: Costs
    seed: int = Field(ge=0)


class ConfigError(ValueError):
    pass


def load(path: Path = DEFAULT) -> tuple[Config, str]:
    """The validated config and the SHA-256 of the file's bytes, saved with every model."""
    try:
        raw = path.read_bytes()
        parsed = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"can't read {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{path} must be a mapping of settings")
    try:
        return Config.model_validate(parsed), hashlib.sha256(raw).hexdigest()
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
