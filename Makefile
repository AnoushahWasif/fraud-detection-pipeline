.PHONY: install lint format test data train compare threshold

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

data:
	uv run python -m fraud_pipeline.ingest

train:
	uv run python -m fraud_pipeline.train

compare:
	uv run python -m fraud_pipeline.compare

threshold:
	uv run python -m fraud_pipeline.threshold
