.PHONY: install lint format test data train compare threshold all serve docker serve-docker

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

# The whole pipeline, in order: download and validate, baseline, comparison, final model.
all: data train compare threshold

serve:
	uv run uvicorn fraud_pipeline.service:app --host 127.0.0.1 --port 8000

docker:
	docker build -t fraud-scoring:local .

# Needs FRAUD_API_KEY in the environment. Published on localhost only.
serve-docker:
	docker run --rm -p 127.0.0.1:8000:8000 -e FRAUD_API_KEY fraud-scoring:local
