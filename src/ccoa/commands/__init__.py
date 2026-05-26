"""CLI commands for ccoa."""

from __future__ import annotations

import abc
import argparse


class BaseCommand(abc.ABC):
    """Base class for all CLI subcommands."""

    name: str
    help: str

    @abc.abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments to the subparser."""

    @abc.abstractmethod
    def run(self, args: argparse.Namespace) -> int:
        """Execute the command. Returns 0 on success, non-zero on failure."""
