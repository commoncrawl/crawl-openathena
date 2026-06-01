"""Unified CLI for ccoa."""

from __future__ import annotations

import argparse
import logging
import sys

from ccoa.commands.classify_warc import ClassifyWarcCommand
from ccoa.commands.tokenize import TokenizeCommand

LOG_LEVELS = ["debug", "info", "warning", "error", "critical"]


def main(argv: list[str] | None = None) -> int:
    """Build the argparse parser, dispatch to the chosen subcommand, return its exit code."""
    parser = argparse.ArgumentParser(prog="ccoa", description="CC-Open-Athena CLI tools.")
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="info",
        help="Logging verbosity (default: info).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = [ClassifyWarcCommand(), TokenizeCommand()]
    command_map: dict[str, object] = {}
    for cmd in commands:
        sub = subparsers.add_parser(cmd.name, help=cmd.help, description=cmd.help)
        cmd.add_arguments(sub)
        command_map[cmd.name] = cmd

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return command_map[args.command].run(args)


if __name__ == "__main__":
    sys.exit(main())
