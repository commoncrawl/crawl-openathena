"""Per-WARC text extraction cache.

The `classify-warc` command pays most of its wall-clock cost inside
`trafilatura.extract`. When the same WARC files are processed more than
once (different model, different label, parameter sweeps, retries), this
module lets the extraction step be skipped for previously-seen records.

Cache layout
------------
One file per input WARC, gzipped JSONL. Each line is
`{"index": N, "text": "..."}` where `index` is the 0-based ordinal of the
response record within the WARC (i.e. the index over the stream that
`iter_response_records` yields).

Cache files live under a user-supplied `--cache-dir`, which can be a
local path or any fsspec URI (e.g. ``s3://bucket/prefix``). The input
URI is mapped to a deterministic path under that directory using
`cache_path_for_warc`.
"""

from __future__ import annotations

import gzip
import json
import logging
import posixpath

import fsspec

logger = logging.getLogger(__name__)


def cache_path_for_warc(warc_uri: str, cache_dir: str) -> str:
    """Return the cache file path for `warc_uri` under `cache_dir`.

    The scheme of `warc_uri` is used as the first path segment so caches
    for different sources never collide (`local/...` vs `s3/bucket/...`).
    `..` segments are rejected to prevent escaping `cache_dir`.
    """
    if "://" in warc_uri:
        scheme, _, remainder = warc_uri.partition("://")
    else:
        scheme, remainder = "local", warc_uri

    if not scheme:
        raise ValueError(f"Empty scheme in WARC URI: {warc_uri!r}")

    remainder = remainder.replace("\\", "/")
    parts: list[str] = []
    for segment in remainder.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            raise ValueError(
                f"Parent-directory segment '..' is not allowed in WARC URI: {warc_uri!r}"
            )
        parts.append(segment)

    if not parts:
        raise ValueError(f"WARC URI has no path component: {warc_uri!r}")

    relative = posixpath.join(scheme, *parts) + ".jsonl.gz"
    return cache_dir.rstrip("/") + "/" + relative


def load_extraction_cache(cache_uri: str, storage_options: dict[str, object]) -> dict[int, str]:
    """Load `{record_index: text}` from a JSONL.gz cache file.

    Returns an empty dict when the file is missing. On any read or parse
    error, logs a warning and returns an empty dict — a broken cache must
    not break the pipeline.
    """
    try:
        with fsspec.open(cache_uri, "rb", **storage_options) as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                entries: dict[int, str] = {}
                for line in gz:
                    obj = json.loads(line)
                    entries[int(obj["index"])] = obj["text"]
                return entries
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning(
            "Could not read extraction cache %s (%s: %s); treating as empty.",
            cache_uri,
            type(exc).__name__,
            exc,
        )
        return {}


def save_extraction_cache(
    cache_uri: str, entries: dict[int, str], storage_options: dict[str, object]
) -> None:
    """Write `entries` as gzipped JSONL to `cache_uri`. No-op if empty."""
    if not entries:
        return

    with fsspec.open(cache_uri, "wb", **storage_options) as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb") as gz:
            for index in sorted(entries):
                line = (
                    json.dumps(
                        {"index": index, "text": entries[index]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                gz.write(line.encode("utf-8"))
