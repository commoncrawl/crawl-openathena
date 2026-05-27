"""fsspec path/URI utilities shared by the CLI commands."""

from __future__ import annotations

import contextlib
import os
import random
import sys

import fsspec


def s3_storage_options(uri: str, anonymous: bool, requester_pays: bool) -> dict[str, object]:
    """Build the kwargs to pass to fsspec.open for an `s3://` URI."""
    if not uri.startswith("s3://"):
        return {}
    opts: dict[str, object] = {}
    if anonymous:
        opts["anon"] = True
    if requester_pays:
        opts["requester_pays"] = True
    return opts


def summary_path_for(output: str) -> str:
    """Return the sidecar summary path for an output file path or fsspec URL.

    Inserts `.summary` before the last extension. Examples:
        `foo.csv`              -> `foo.summary.csv`
        `data/out.tsv`         -> `data/out.summary.tsv`
        `s3://bucket/key.csv`  -> `s3://bucket/key.summary.csv`
        `foo`                  -> `foo.summary`
    """
    root, ext = os.path.splitext(output)
    return f"{root}.summary{ext}"


def output_exists(uri: str, storage_options: dict[str, object]) -> bool:
    """Return True if `uri` already exists (works for local paths and fsspec URLs)."""
    fs, fs_path = fsspec.url_to_fs(uri, **storage_options)
    return bool(fs.exists(fs_path))


def _is_glob(pattern: str) -> bool:
    """Return True if `pattern` contains any fsspec glob metacharacter."""
    return any(ch in pattern for ch in "*?[")


def resolve_warc_paths(
    patterns: list[str],
    *,
    anonymous: bool,
    requester_pays: bool,
    shuffle: bool,
    seed: int,
    files_limit: int,
) -> list[str]:
    """Expand glob patterns to a sorted (optionally shuffled, limited) URI list.

    Each pattern is either a literal URI (passes through) or a glob
    pattern containing `*`, `?`, or `[`. Globs are resolved via
    `fsspec.url_to_fs` + `fs.glob`, then re-stamped with their scheme
    via `fs.unstrip_protocol`. The combined matches are de-duplicated
    and sorted lexicographically; when `shuffle=True` they are then
    permuted by `random.Random(seed)`; finally `files_limit > 0`
    truncates to that many URIs.
    """
    resolved: list[str] = []
    for pattern in patterns:
        if _is_glob(pattern):
            opts = s3_storage_options(pattern, anonymous, requester_pays)
            fs, _ = fsspec.url_to_fs(pattern, **opts)
            has_scheme = "://" in pattern
            for match in fs.glob(pattern):
                resolved.append(fs.unstrip_protocol(match) if has_scheme else match)
        else:
            resolved.append(pattern)

    seen: set[str] = set()
    deduped: list[str] = []
    for uri in resolved:
        if uri not in seen:
            seen.add(uri)
            deduped.append(uri)
    deduped.sort()

    if shuffle:
        random.Random(seed).shuffle(deduped)
    if files_limit > 0:
        deduped = deduped[:files_limit]
    return deduped


@contextlib.contextmanager
def open_output_sink(output: str, storage_options: dict[str, object]):
    """Yield a text-mode file-like for the output.

    `"-"` yields `sys.stdout` without closing it. Any other path is
    opened via `fsspec.open`, so local paths, `s3://`, and any
    other fsspec-supported URL all work.
    """
    if output == "-":
        yield sys.stdout
        return
    with fsspec.open(
        output,
        mode="w",
        newline="",
        encoding="utf-8",
        **storage_options,
    ) as sink:
        yield sink
