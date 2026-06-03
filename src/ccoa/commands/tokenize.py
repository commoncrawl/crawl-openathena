"""Tokenize text-cache records with a HuggingFace tokenizer.

Reads the per-WARC text-extraction cache produced by
`ccoa classify-warc --cache-dir <uri>` (gzipped JSONL with one
`{"index": N, "text": "..."}` line per response record), tokenizes each
text with a fast HuggingFace tokenizer, and writes a per-record parquet
of `(cache_path, record_index, n_tokens, token_ids)`. A sidecar
`<output>.summary.csv` captures CLI args, counters, and a token-count
distribution.

Example:
    ```bash
    uv sync --extra tokenize
    export HF_TOKEN=<gated-license token>
    uv run ccoa tokenize \\
      --cache-paths 's3://commoncrawl-dev/cc-focus-tools/warc-text-extract-cache/s3/.../*.warc.gz.jsonl.gz' \\
      --files-limit 1 --records-per-file-limit 100 \\
      --workers 4 --progress-every 25 \\
      --output /tmp/tokens.parquet
    ```

Each input cache file maps 1:1 to a WARC, so per-file parallelism is the
natural unit of work. HuggingFace Fast tokenizers (Rust) release the GIL
and are thread-safe — the default `--workers-mode thread` shares one
loaded tokenizer instance with no per-call lock.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field

import pyarrow as pa

from ccoa.commands import BaseCommand
from ccoa.extraction.cache import load_extraction_cache
from ccoa.tokenizer.hf import DEFAULT_TOKENIZER_REPO, load_tokenizer, tokenize_batch
from ccoa.utils.fs.paths import (
    output_exists,
    resolve_warc_paths,
    s3_storage_options,
)
from ccoa.utils.io.parquet import TOKENIZE_SCHEMA, open_parquet_writer
from ccoa.utils.reporting.summary import (
    format_duration,
    log_tokens_summary,
    write_tokenize_summary,
)

logger = logging.getLogger(__name__)


# Per-process tokenizer state. In thread mode the parent loads the
# tokenizer once and passes it to each worker; in process mode each
# worker rebuilds `_WORKER_TOKENIZER` via the pool initializer.
_WORKER_TOKENIZER: object | None = None


@dataclass
class FileResult:
    """Aggregated per-file outcome returned by `process_one_file`."""

    cache_uri: str
    rows: list[tuple[str, int, int, list[int]]] = field(default_factory=list)
    processed: int = 0
    skipped_empty: int = 0
    t_load: float = 0.0
    t_tokenize: float = 0.0


def process_one_file(
    cache_uri: str,
    tokenizer: object,
    args: argparse.Namespace,
) -> FileResult:
    """Tokenize all records in one cache file and return its `FileResult`.

    Loads the gzipped-JSONL cache into memory, applies
    `args.records_per_file_limit` (if set), drops empty texts, then
    batches the surviving texts through `tokenize_batch`. Records are
    emitted in ascending record-index order for determinism.
    """
    result = FileResult(cache_uri=cache_uri)
    storage_options = s3_storage_options(cache_uri, args.anonymous_s3, args.s3_requester_pays)

    logger.info("Loading cache %s", cache_uri)
    t0 = time.perf_counter()
    entries = load_extraction_cache(cache_uri, storage_options)
    result.t_load = time.perf_counter() - t0

    if not entries:
        logger.warning("Cache %s yielded no entries; skipping.", cache_uri)
        return result

    indices = sorted(entries)
    if args.records_per_file_limit > 0:
        indices = indices[: args.records_per_file_limit]

    # Drop empties up front so the per-batch arrays line up with the
    # tokenizer output.
    surviving: list[int] = []
    for idx in indices:
        text = entries[idx]
        if not text or not text.strip():
            result.skipped_empty += 1
            continue
        surviving.append(idx)

    progress_every = getattr(args, "progress_every", 0) or 0
    batch_size = max(1, args.batch_size)

    for i in range(0, len(surviving), batch_size):
        batch_idx = surviving[i : i + batch_size]
        batch_text = [entries[j] for j in batch_idx]
        t0 = time.perf_counter()
        ids_list = tokenize_batch(tokenizer, batch_text)
        result.t_tokenize += time.perf_counter() - t0
        for j, ids in zip(batch_idx, ids_list, strict=True):
            result.rows.append((cache_uri, j, len(ids), ids))
            result.processed += 1
            if progress_every and result.processed % progress_every == 0:
                logger.info(
                    "  %s: processed=%d skipped_empty=%d",
                    cache_uri,
                    result.processed,
                    result.skipped_empty,
                )

    return result


def _tokenize_summary_path(output: str) -> str:
    """Return the sidecar summary path for `output`.

    Always emits `.summary.csv` (the sidecar is CSV regardless of the
    main output's extension — typically `.parquet` for this command).
    """
    root, _ = os.path.splitext(output)
    return f"{root}.summary.csv"


def _process_pool_initializer(tokenizer_repo: str) -> None:
    """Per-worker setup for `ProcessPoolExecutor`: load the tokenizer once."""
    global _WORKER_TOKENIZER  # noqa: PLW0603
    _WORKER_TOKENIZER = load_tokenizer(tokenizer_repo)


def _process_pool_worker(
    payload: tuple[str, argparse.Namespace],
) -> FileResult:
    """Top-level pickleable adapter that runs `process_one_file` in a worker."""
    cache_uri, args = payload
    return process_one_file(cache_uri, _WORKER_TOKENIZER, args)


def _rows_to_table(rows: list[tuple[str, int, int, list[int]]]) -> pa.Table:
    """Convert `FileResult.rows` into a `pyarrow.Table` matching `TOKENIZE_SCHEMA`."""
    cache_paths = [r[0] for r in rows]
    record_indices = [r[1] for r in rows]
    n_tokens = [r[2] for r in rows]
    token_ids = [r[3] for r in rows]
    return pa.table(
        {
            "cache_path": pa.array(cache_paths, type=pa.string()),
            "record_index": pa.array(record_indices, type=pa.int32()),
            "n_tokens": pa.array(n_tokens, type=pa.int32()),
            "token_ids": pa.array(token_ids, type=pa.list_(pa.int32())),
        },
        schema=TOKENIZE_SCHEMA,
    )


class TokenizeCommand(BaseCommand):
    """Tokenize text-cache records with a HuggingFace tokenizer."""

    name = "tokenize"
    help = "Tokenize text-cache records with a HuggingFace tokenizer; write parquet."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register CLI flags for the tokenize subcommand."""
        parser.add_argument(
            "--cache-paths",
            nargs="+",
            required=True,
            metavar="URI",
            help=(
                "One or more text-cache URIs or glob patterns "
                "(e.g. 's3://my-bucket/cache/s3/commoncrawl/.../*.warc.gz.jsonl.gz'). "
                "Each match is a gzipped-JSONL cache file as produced by "
                "`classify-warc --cache-dir`. Globs are expanded via fsspec."
            ),
        )
        parser.add_argument(
            "--tokenizer",
            default=DEFAULT_TOKENIZER_REPO,
            metavar="REPO",
            help=(
                "HuggingFace repo id of the tokenizer "
                f"(default: {DEFAULT_TOKENIZER_REPO}). Must resolve to a fast "
                "(Rust) tokenizer for thread-mode safety. Gated repos require "
                "HF_TOKEN env var or `huggingface-cli login`."
            ),
        )
        parser.add_argument(
            "--records-limit",
            type=int,
            default=0,
            help=(
                "Max number of records to tokenize across all selected files "
                "(0 = unlimited). Incompatible with --workers > 1."
            ),
        )
        parser.add_argument(
            "--records-per-file-limit",
            type=int,
            default=0,
            help="Max number of records per cache file (0 = unlimited).",
        )
        parser.add_argument(
            "--files-limit",
            type=int,
            default=0,
            help="Max number of input cache files after glob expansion and shuffle (0 = unlimited).",
        )
        parser.add_argument(
            "--shuffle-files",
            action="store_true",
            help="Shuffle the resolved file list (deterministic via --seed) before --files-limit.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Seed for --shuffle-files (default: 42).",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help=(
                "Number of cache files tokenized concurrently (default: 1). "
                "Incompatible with --records-limit; use --records-per-file-limit "
                "for a per-file cap."
            ),
        )
        parser.add_argument(
            "--workers-mode",
            choices=["thread", "process"],
            default="thread",
            help=(
                "How to parallelise across files when --workers > 1. 'thread' "
                "(default) shares one fast tokenizer in this process; HF fast "
                "tokenizers release the GIL and are thread-safe. 'process' "
                "loads a separate tokenizer per worker process (heavier RAM)."
            ),
        )
        parser.add_argument(
            "--max-pool-restarts",
            type=int,
            default=10,
            metavar="N",
            help=(
                "In --workers-mode process, restart the pool up to N times when "
                "a worker dies (BrokenProcessPool). The suspect file is dropped "
                "from this run. 0 = unlimited; default 10."
            ),
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            metavar="N",
            help=(
                "Number of texts handed to the tokenizer per call (default: 64). "
                "HF fast tokenizers vectorize batches internally."
            ),
        )
        parser.add_argument(
            "--progress-every",
            type=int,
            default=1000,
            metavar="N",
            help=(
                "Log a per-file progress heartbeat every N tokenized records "
                "(default: 1000; 0 disables)."
            ),
        )
        parser.add_argument(
            "--output",
            required=True,
            metavar="PATH",
            help=(
                "Output parquet path. Local path or any fsspec URL "
                "(e.g. s3://bucket/key.parquet). Suggest the .parquet extension."
            ),
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help=(
                "Overwrite an existing --output (and its `.summary` sidecar) "
                "instead of failing fast. Off by default to protect prior runs."
            ),
        )
        parser.add_argument(
            "--anonymous-s3",
            action="store_true",
            help="Force anonymous S3 access (only for buckets that allow it).",
        )
        parser.add_argument(
            "--s3-requester-pays",
            action="store_true",
            help="Set RequestPayer=requester for S3 access.",
        )

    def run(self, args: argparse.Namespace) -> int:
        """Resolve cache files, tokenize each record, write parquet + summary."""
        logger.info("Running tokenize")

        if args.workers > 1 and args.records_limit > 0:
            logger.error(
                "--records-limit cannot be combined with --workers > 1; "
                "use --records-per-file-limit (works in parallel) or set --workers 1."
            )
            return 2

        output_storage_options = s3_storage_options(
            args.output, args.anonymous_s3, args.s3_requester_pays
        )
        summary_uri = _tokenize_summary_path(args.output)
        summary_storage_options = s3_storage_options(
            summary_uri, args.anonymous_s3, args.s3_requester_pays
        )
        for path, opts in (
            (args.output, output_storage_options),
            (summary_uri, summary_storage_options),
        ):
            if output_exists(path, opts):
                if args.overwrite:
                    logger.warning("Overwriting existing output: %s", path)
                else:
                    logger.error(
                        "Output already exists: %s. Refusing to overwrite; "
                        "delete the file(s), pick a fresh --output path, or "
                        "pass --overwrite to replace them.",
                        path,
                    )
                    return 2

        logger.info("Resolving cache paths: %s", args.cache_paths)
        resolved = resolve_warc_paths(
            args.cache_paths,
            anonymous=args.anonymous_s3,
            requester_pays=args.s3_requester_pays,
            shuffle=args.shuffle_files,
            seed=args.seed,
            files_limit=args.files_limit,
        )
        logger.info(
            "Resolved %d cache files (shuffle=%s files_limit=%d)",
            len(resolved),
            args.shuffle_files,
            args.files_limit,
        )
        for uri in resolved:
            logger.debug("  selected %s", uri)

        if not resolved:
            logger.warning("No cache files matched --cache-paths; nothing to tokenize.")
            log_tokens_summary([])
            return 0

        logger.info("Loading tokenizer %s", args.tokenizer)
        try:
            parent_tokenizer = load_tokenizer(args.tokenizer)
        except Exception as exc:
            logger.error(
                "Failed to load tokenizer %s: %s: %s", args.tokenizer, type(exc).__name__, exc
            )
            return 2

        use_process_pool = args.workers > 1 and args.workers_mode == "process"
        if use_process_pool:
            logger.info(
                "Worker mode 'process'; freeing parent-side tokenizer and deferring "
                "load to each of %d worker processes.",
                args.workers,
            )
            # Workers reload via the pool initializer; drop the parent copy.
            parent_tokenizer = None  # noqa: F841

        n_tokens_all: list[int] = []
        processed = 0
        skipped_empty = 0
        files_done = 0
        files_total = len(resolved)
        limit = args.records_limit
        t_load_total = 0.0
        t_tokenize_total = 0.0

        logger.info("Writing output to %s (workers=%d)", args.output, args.workers)
        logger.info("Summary will be written to %s", summary_uri)

        started_at = _dt.datetime.now(_dt.UTC).isoformat()
        t_processing_start = time.perf_counter()

        writer_lock = threading.Lock()
        # Inner-scope state so `_aggregate` can capture `writer` and the
        # totals via `nonlocal` (mirrors classify-warc's structure).
        with open_parquet_writer(args.output, TOKENIZE_SCHEMA, output_storage_options) as writer:

            def _aggregate(result: FileResult) -> None:
                """Append `result` to the parquet and fold its counters into totals."""
                nonlocal processed, skipped_empty, t_load_total, t_tokenize_total
                nonlocal files_done
                if result.rows:
                    table = _rows_to_table(result.rows)
                    with writer_lock:
                        writer.write_table(table)
                    n_tokens_all.extend(r[2] for r in result.rows)
                processed += result.processed
                skipped_empty += result.skipped_empty
                t_load_total += result.t_load
                t_tokenize_total += result.t_tokenize
                files_done += 1
                logger.info(
                    "Finished %s — processed=%d skipped_empty=%d",
                    result.cache_uri,
                    processed,
                    skipped_empty,
                )
                elapsed = time.perf_counter() - t_processing_start
                mean_per_file = elapsed / files_done
                eta = mean_per_file * (files_total - files_done)
                logger.info(
                    "progress — files=%d/%d elapsed=%s eta=~%s",
                    files_done,
                    files_total,
                    format_duration(elapsed),
                    format_duration(eta),
                )

            if args.workers == 1:
                for uri in resolved:
                    _aggregate(process_one_file(uri, parent_tokenizer, args))
                    if limit and processed >= limit:
                        logger.info("Reached --records-limit %d; stopping.", limit)
                        break
            elif args.workers_mode == "process":
                remaining_uris = list(resolved)
                restarts = 0
                max_restarts = args.max_pool_restarts
                while remaining_uris:
                    completed_in_pool = 0
                    try:
                        with ProcessPoolExecutor(
                            max_workers=args.workers,
                            initializer=_process_pool_initializer,
                            initargs=(args.tokenizer,),
                        ) as pool:
                            payloads = [(u, args) for u in remaining_uris]
                            for result in pool.map(_process_pool_worker, payloads):
                                _aggregate(result)
                                completed_in_pool += 1
                        break
                    except BrokenProcessPool:
                        suspect_idx = completed_in_pool
                        if suspect_idx >= len(remaining_uris):
                            logger.error(
                                "Process pool died after consuming all results; "
                                "aborting (cannot identify a culprit to drop)."
                            )
                            raise
                        suspect = remaining_uris[suspect_idx]
                        restarts += 1
                        if max_restarts and restarts > max_restarts:
                            logger.error(
                                "Process pool died %d times (limit %d); aborting. "
                                "Last suspect: %s.",
                                restarts,
                                max_restarts,
                                suspect,
                            )
                            raise
                        logger.error(
                            "Process pool worker died after %d files in this pool; "
                            "dropping suspect %s and restarting (restart %d/%s).",
                            completed_in_pool,
                            suspect,
                            restarts,
                            "∞" if max_restarts == 0 else str(max_restarts),
                        )
                        remaining_uris = remaining_uris[suspect_idx + 1 :]
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for result in pool.map(
                        lambda u: process_one_file(u, parent_tokenizer, args),
                        resolved,
                    ):
                        _aggregate(result)

        t_processing = time.perf_counter() - t_processing_start

        logger.info(
            "Total records tokenized: %d (skipped %d empty)",
            processed,
            skipped_empty,
        )
        log_tokens_summary(n_tokens_all)

        finished_at = _dt.datetime.now(_dt.UTC).isoformat()
        write_tokenize_summary(
            summary_uri,
            summary_storage_options,
            args=args,
            resolved_count=len(resolved),
            n_tokens=n_tokens_all,
            processed=processed,
            skipped_empty=skipped_empty,
            t_processing=t_processing,
            t_load_total=t_load_total,
            t_tokenize_total=t_tokenize_total,
            started_at=started_at,
            finished_at=finished_at,
        )
        logger.info("Wrote summary %s", summary_uri)

        return 0
