"""Shared pytest fixtures + options for the test suite."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--run-real` CLI flag that enables `real_model`-marked tests."""
    parser.addoption(
        "--run-real",
        action="store_true",
        default=False,
        help=(
            "Run tests marked `real_model` (downloads a real fasttext model "
            "from HuggingFace; takes a few minutes on first run)."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip `real_model`-marked tests unless `--run-real` is passed."""
    if config.getoption("--run-real"):
        return
    skip_real = pytest.mark.skip(reason="needs --run-real to enable the real-model end-to-end test")
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip_real)
