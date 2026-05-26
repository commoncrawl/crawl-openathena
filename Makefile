.PHONY: install sync test lint format format-check check clean

install:
	uv sync

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

format-check:
	uv run ruff format --check src tests

check: lint format-check test

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
