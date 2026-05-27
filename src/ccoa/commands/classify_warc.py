"""Classify WARC response records with a fasttext model.

Streams one or more WARC files from S3 (or any fsspec URL), extracts
plain text from each `response` record with trafilatura, and applies
a HuggingFace-hosted fasttext classifier. Per-record output is a CSV
`URL,prediction_score,warc_filename,warc_record_index` written to
stdout, a local path, or an `s3://` URL (anything `fsspec.open`
understands); a one-shot summary of the score distribution is logged
at the end and written to a `<output>.summary.csv` sidecar.

Example:
    ```bash
    uv run ccoa classify-warc \\
      --warc-paths 's3://commoncrawl/crawl-data/CC-MAIN-2025-51/segments/.../*.warc.gz' \\
      --shuffle-files --files-limit 8 \\
      --records-per-file-limit 50 \\
      --workers 4 \\
      --output data/classified.csv
    ```

The default model (`ibm-granite/GneissWeb.Sci_classifier`) is ~4 GB;
the first run will download it into the HuggingFace cache.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import logging
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

import fsspec

from ccoa.classifier.fasttext import (
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_REPO,
    DEFAULT_TARGET_LABEL,
    load_classifier,
    predict_target,
)
from ccoa.commands import BaseCommand
from ccoa.extraction.cache import (
    cache_path_for_warc,
    load_extraction_cache,
    save_extraction_cache,
)
from ccoa.extraction.text import TrafilaturaLogCounter, extract_text
from ccoa.utils.fs.paths import (
    open_output_sink,
    output_exists,
    resolve_warc_paths,
    s3_storage_options,
    summary_path_for,
)
from ccoa.utils.reporting.summary import (
    format_duration,
    log_summary,
    log_timing,
    write_run_summary,
)
from ccoa.utils.warc.reader import iter_response_records
from ccoa.utils.warc.urls import is_homepage_url

logger = logging.getLogger(__name__)


_WORKER_MODEL: object = None
_WORKER_LOCK = threading.Lock()


def load_resume_skipset(
    path: str,
    storage_options: dict[str, object],
) -> dict[str, frozenset[int]]:
    """Build `{warc_filename: frozenset(record_indices)}` from a prior output CSV.

    Used by `--resume-from-output` to skip records already classified. The
    input must have `warc_filename` and `warc_record_index` columns (which
    the current output schema always emits); a `ValueError` is raised
    otherwise to flag stale outputs from the pre-resume schema.
    """
    required = {"warc_filename", "warc_record_index"}
    buckets: dict[str, set[int]] = {}
    with fsspec.open(path, "r", encoding="utf-8", **storage_options) as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            missing = required - fields
            raise ValueError(
                f"{path} is missing required columns for --resume-from-output: "
                f"need {sorted(required)}, missing {sorted(missing)} "
                f"(found {sorted(fields)})."
            )
        for row in reader:
            warc = row["warc_filename"]
            try:
                idx = int(row["warc_record_index"])
            except (TypeError, ValueError):
                continue
            buckets.setdefault(warc, set()).add(idx)
    return {warc: frozenset(idxs) for warc, idxs in buckets.items()}


@dataclass
class FileResult:
    """Aggregated per-file outcome returned by `process_one_file`.

    Each entry in `rows` is `(url, score, warc_filename, warc_record_index)`,
    matching the output CSV schema.
    """

    uri: str
    rows: list[tuple[str, float, str, int]]
    processed: int
    skipped_empty: int
    skipped_homepage: int
    skipped_resume: int
    extract_errors: int
    cache_hits: int
    cache_misses: int
    t_extract: float
    t_predict: float


def process_one_file(
    uri: str,
    model,
    model_lock: threading.Lock,
    args: argparse.Namespace,
    skip_indices: frozenset[int] = frozenset(),
) -> FileResult:
    """Stream one WARC, classify each response record, return its `FileResult`.

    Owns its own fsspec input stream, its own per-WARC cache load/save
    (when `--cache-dir` is set), and its own `--records-per-file-limit`
    counter. Calls `predict_target` under `model_lock` since fasttext
    makes no thread-safety guarantees; the lock cost is sub-ms
    compared to the per-record extract cost.
    """
    rows: list[tuple[str, float, str, int]] = []
    processed = 0
    skipped_empty = 0
    skipped_homepage = 0
    skipped_resume = 0
    extract_errors = 0
    cache_hits = 0
    cache_misses = 0
    t_extract = 0.0
    t_predict = 0.0
    per_file_limit = args.records_per_file_limit
    progress_every = getattr(args, "progress_every", 0) or 0

    # Resume target semantics: --records-per-file-limit names the desired
    # total in the final output (resumed + new). Subtract the skip-set size
    # so we only process the remainder; skip the whole file when the target
    # is already met (no S3 stream, no cache load, no extract).
    if per_file_limit and skip_indices:
        remaining = per_file_limit - len(skip_indices)
        if remaining <= 0:
            logger.info(
                "Skipping %s — already at --records-per-file-limit (%d resumed).",
                uri,
                len(skip_indices),
            )
            return FileResult(
                uri=uri,
                rows=[],
                processed=0,
                skipped_empty=0,
                skipped_homepage=0,
                skipped_resume=len(skip_indices),
                extract_errors=0,
                cache_hits=0,
                cache_misses=0,
                t_extract=0.0,
                t_predict=0.0,
            )
        per_file_limit = remaining

    logger.info("Opening %s", uri)

    input_storage_options = s3_storage_options(uri, args.anonymous_s3, args.s3_requester_pays)
    cache_uri: str | None = None
    cache_storage_options: dict[str, object] = {}
    cache: dict[int, str] = {}
    cache_dirty = False
    if args.cache_dir:
        cache_uri = cache_path_for_warc(uri, args.cache_dir)
        cache_storage_options = s3_storage_options(
            cache_uri, args.anonymous_s3, args.s3_requester_pays
        )
        cache = load_extraction_cache(cache_uri, cache_storage_options)
        logger.info(
            "Cache for %s: %s (%d entries preloaded)",
            uri,
            cache_uri,
            len(cache),
        )

    try:
        with fsspec.open(uri, "rb", **input_storage_options) as stream:
            for record_index, record in enumerate(iter_response_records(stream)):
                if record_index in skip_indices:
                    skipped_resume += 1
                    continue
                url = record.rec_headers.get_header("WARC-Target-URI") or ""
                if args.skip_homepages and is_homepage_url(url):
                    skipped_homepage += 1
                    continue

                if record_index in cache:
                    text = cache[record_index]
                    cache_hits += 1
                else:
                    html = record.content_stream().read()
                    t0 = time.perf_counter()
                    try:
                        text = extract_text(html)
                    except Exception as exc:
                        t_extract += time.perf_counter() - t0
                        extract_errors += 1
                        logger.debug(
                            "extract_text failed for %s record %d: %s: %s",
                            uri,
                            record_index,
                            type(exc).__name__,
                            exc,
                        )
                        continue
                    t_extract += time.perf_counter() - t0
                    cache_misses += 1
                    if args.cache_dir:
                        cache[record_index] = text
                        cache_dirty = True

                cleaned = text.strip()
                if not cleaned:
                    skipped_empty += 1
                    continue

                t0 = time.perf_counter()
                with model_lock:
                    score = predict_target(model, cleaned, args.target_label)
                t_predict += time.perf_counter() - t0

                rows.append((url, score, uri, record_index))
                processed += 1

                if progress_every and processed % progress_every == 0:
                    logger.info(
                        "  %s: processed=%d cache_hits=%d cache_misses=%d "
                        "skipped_empty=%d skipped_homepage=%d",
                        uri,
                        processed,
                        cache_hits,
                        cache_misses,
                        skipped_empty,
                        skipped_homepage,
                    )

                if per_file_limit and processed >= per_file_limit:
                    logger.info(
                        "Reached --records-per-file-limit %d for %s; moving to next file.",
                        per_file_limit,
                        uri,
                    )
                    break
    finally:
        if cache_uri is not None and cache_dirty:
            save_extraction_cache(cache_uri, cache, cache_storage_options)
            logger.info("Wrote cache %s (%d entries)", cache_uri, len(cache))
        elif cache_uri is not None:
            logger.info("Cache %s already complete; no write needed.", cache_uri)

    return FileResult(
        uri=uri,
        rows=rows,
        processed=processed,
        skipped_empty=skipped_empty,
        skipped_homepage=skipped_homepage,
        skipped_resume=skipped_resume,
        extract_errors=extract_errors,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        t_extract=t_extract,
        t_predict=t_predict,
    )


def _process_pool_initializer(model_repo: str, model_file: str) -> None:
    """Per-worker setup for `ProcessPoolExecutor`: load the model once.

    Also silences trafilatura's chatty per-page WARN/ERROR logs in this
    worker. Each worker process has its own lxml + fasttext state, so a
    heap-corruption abort here cannot tear down the parent or its peers.
    """
    global _WORKER_MODEL  # noqa: PLW0603
    _WORKER_MODEL = load_classifier(model_repo, model_file)
    logging.getLogger("trafilatura").setLevel(logging.CRITICAL)


def _process_pool_worker(
    payload: tuple[str, argparse.Namespace, frozenset[int]],
) -> FileResult:
    """Top-level pickleable adapter that runs `process_one_file` in a worker."""
    uri, args, skip_indices = payload
    return process_one_file(uri, _WORKER_MODEL, _WORKER_LOCK, args, skip_indices)


class ClassifyWarcCommand(BaseCommand):
    """Classify WARC response records with a fasttext model."""

    name = "classify-warc"
    help = "Classify WARC response records with a HuggingFace-hosted fasttext model."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register CLI flags for the classify-warc subcommand."""
        parser.add_argument(
            "--warc-paths",
            nargs="+",
            required=True,
            metavar="URI",
            help=(
                "One or more WARC URIs or glob patterns "
                "(e.g. 's3://commoncrawl/.../*.warc.gz'). Globs are expanded "
                "via fsspec; quote them in the shell to prevent local expansion."
            ),
        )
        parser.add_argument(
            "--records-limit",
            type=int,
            default=0,
            help=(
                "Max number of response records to process across all selected "
                "files (0 = unlimited)."
            ),
        )
        parser.add_argument(
            "--records-per-file-limit",
            type=int,
            default=0,
            help=(
                "Max number of response records per WARC file (0 = unlimited). "
                "With --resume-from-output this is treated as the target total "
                "per file (resumed + new); files already at the target are "
                "skipped entirely. To process an additional N records on top of "
                "a prior run, set this to `prior_limit + N`."
            ),
        )
        parser.add_argument(
            "--skip-homepages",
            action="store_true",
            help=(
                "Skip response records whose URL is a site-root homepage "
                "(empty/root path, no query, no fragment)."
            ),
        )
        parser.add_argument(
            "--files-limit",
            type=int,
            default=0,
            help=(
                "Max number of input WARC files after glob expansion and shuffle (0 = unlimited)."
            ),
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
            "--model-repo",
            default=DEFAULT_MODEL_REPO,
            help=f"HuggingFace repo id of the fasttext model (default: {DEFAULT_MODEL_REPO}).",
        )
        parser.add_argument(
            "--model-file",
            default=DEFAULT_MODEL_FILE,
            help=f"Filename of the .bin model inside the repo (default: {DEFAULT_MODEL_FILE}).",
        )
        parser.add_argument(
            "--target-label",
            default=DEFAULT_TARGET_LABEL,
            help=(
                "Label whose probability is written as 'prediction_score' "
                f"(default: {DEFAULT_TARGET_LABEL})."
            ),
        )
        parser.add_argument(
            "--output",
            default="-",
            help=(
                "Output CSV path. Use '-' for stdout, a local path, or any fsspec URL "
                "(e.g. s3://bucket/key.csv). S3 outputs use the same --anonymous-s3 / "
                "--s3-requester-pays options as inputs (default: -)."
            ),
        )
        parser.add_argument(
            "--cache-dir",
            default=None,
            metavar="URI",
            help=(
                "Optional cache directory for extracted text (local path or fsspec URI, "
                "e.g. s3://bucket/prefix). When set, trafilatura output is cached per "
                "input WARC as gzipped JSONL keyed by record ordinal. Honors "
                "--anonymous-s3 / --s3-requester-pays for S3 access."
            ),
        )
        parser.add_argument(
            "--resume-from-output",
            default=None,
            metavar="PATH",
            help=(
                "Optional path/URI to a prior classify-warc output CSV. Records "
                "matching `(warc_filename, warc_record_index)` from that file are "
                "skipped on this run; the new --output gets only the missing rows. "
                "Concatenate the two CSVs (drop the second header) to get a "
                "complete result."
            ),
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help=(
                "Number of WARC files processed concurrently (default: 1). "
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
                "(default) shares one loaded model in this process; cheap, but "
                "trafilatura/lxml is called concurrently from multiple threads "
                "and can trigger glibc heap-corruption aborts on adversarial "
                "HTML. 'process' loads a separate model per worker process "
                "(~4 GB extra RAM each) and fully isolates lxml + fasttext C "
                "state — switch to this if you see 'corrupted size vs. "
                "prev_size' or similar aborts under thread mode. The sidecar "
                "summary's trafilatura counts are 0 in process mode (workers "
                "silence the logger locally and don't report counts back)."
            ),
        )
        parser.add_argument(
            "--max-pool-restarts",
            type=int,
            default=10,
            metavar="N",
            help=(
                "In --workers-mode process, restart the pool up to N times when "
                "a worker dies (BrokenProcessPool). Each restart drops the "
                "suspected culprit file and continues with the remaining ones "
                "(re-run with --resume-from-output later to retry the dropped "
                "files). 0 = unlimited; default 10."
            ),
        )
        parser.add_argument(
            "--progress-every",
            type=int,
            default=1000,
            metavar="N",
            help=(
                "Log a per-file progress heartbeat every N classified records "
                "(default: 1000; 0 disables). At a typical 1000 docs / 38s "
                "throughput this is roughly one line per ~40s of CPU work."
            ),
        )
        parser.add_argument(
            "--anonymous-s3",
            action="store_true",
            help="Force anonymous S3 access (only for buckets that allow it; not Common Crawl).",
        )
        parser.add_argument(
            "--s3-requester-pays",
            action="store_true",
            help=(
                "Set RequestPayer=requester for S3 access. Only needed for genuinely "
                "Requester Pays buckets — Common Crawl is NOT one (it just requires "
                "signed access, picked up from the default AWS credential chain)."
            ),
        )

    def run(self, args: argparse.Namespace) -> int:
        """Stream the WARCs, classify each response, and write CSV output."""
        logger.info("Running classify-warc")

        if args.workers > 1 and args.records_limit > 0:
            logger.error(
                "--records-limit cannot be combined with --workers > 1; "
                "use --records-per-file-limit (works in parallel) or set --workers 1."
            )
            return 2

        output_storage_options = s3_storage_options(
            args.output, args.anonymous_s3, args.s3_requester_pays
        )
        summary_uri: str | None = None
        summary_storage_options: dict[str, object] = {}
        if args.output != "-":
            summary_uri = summary_path_for(args.output)
            summary_storage_options = s3_storage_options(
                summary_uri, args.anonymous_s3, args.s3_requester_pays
            )
            for path, opts in (
                (args.output, output_storage_options),
                (summary_uri, summary_storage_options),
            ):
                if output_exists(path, opts):
                    logger.error(
                        "Output already exists: %s. Refusing to overwrite; "
                        "delete the file(s) or pick a fresh --output path.",
                        path,
                    )
                    return 2

        logger.info("Resolving WARC paths: %s", args.warc_paths)

        resolved = resolve_warc_paths(
            args.warc_paths,
            anonymous=args.anonymous_s3,
            requester_pays=args.s3_requester_pays,
            shuffle=args.shuffle_files,
            seed=args.seed,
            files_limit=args.files_limit,
        )
        logger.info(
            "Resolved %d WARC files (shuffle=%s files_limit=%d)",
            len(resolved),
            args.shuffle_files,
            args.files_limit,
        )
        for uri in resolved:
            logger.debug("  selected %s", uri)

        if not resolved:
            logger.warning("No WARC files matched --warc-paths; nothing to classify.")
            log_summary([])
            return 0

        skip_by_warc: dict[str, frozenset[int]] = {}
        if args.resume_from_output:
            resume_storage_options = s3_storage_options(
                args.resume_from_output, args.anonymous_s3, args.s3_requester_pays
            )
            logger.info("Loading resume skip-set from %s", args.resume_from_output)
            skip_by_warc = load_resume_skipset(args.resume_from_output, resume_storage_options)
            total_skips = sum(len(v) for v in skip_by_warc.values())
            logger.info(
                "Resume skip-set: %d records across %d distinct WARCs.",
                total_skips,
                len(skip_by_warc),
            )

        use_process_pool = args.workers > 1 and args.workers_mode == "process"
        if use_process_pool:
            logger.info(
                "Worker mode 'process'; deferring model load to each of "
                "%d worker processes (model: %s/%s).",
                args.workers,
                args.model_repo,
                args.model_file,
            )
            model = None
            model_lock = threading.Lock()
        else:
            logger.info("Loading model %s/%s", args.model_repo, args.model_file)
            model = load_classifier(args.model_repo, args.model_file)
            model_lock = threading.Lock()

        # Silence trafilatura's per-page WARN/ERROR noise (empty / non-HTML /
        # bad-encoding pages) — these outcomes already land in `skipped_empty`.
        # Count the suppressed records and report the totals at the end.
        traf_counter = TrafilaturaLogCounter()
        traf_logger = logging.getLogger("trafilatura")
        traf_logger.addHandler(traf_counter)
        traf_prev_propagate = traf_logger.propagate
        traf_logger.propagate = False

        scores: list[float] = []
        processed = 0
        skipped_empty = 0
        skipped_homepage = 0
        skipped_resume_total = 0
        extract_errors_total = 0
        files_done = 0
        files_total = len(resolved)
        limit = args.records_limit
        t_extract_total = 0.0
        t_predict_total = 0.0
        cache_hits_total = 0
        cache_misses_total = 0

        logger.info("Writing output to %s (workers=%d)", args.output, args.workers)
        if summary_uri:
            logger.info("Summary will be written to %s", summary_uri)

        started_at = _dt.datetime.now(_dt.UTC).isoformat()
        t_processing_start = time.perf_counter()
        with open_output_sink(args.output, output_storage_options) as sink:
            writer = csv.writer(sink)
            writer.writerow(["URL", "prediction_score", "warc_filename", "warc_record_index"])

            def _aggregate(result: FileResult) -> None:
                """Write `result` rows to the CSV and fold its counters into the totals."""
                nonlocal processed, skipped_empty, skipped_homepage
                nonlocal skipped_resume_total, extract_errors_total
                nonlocal cache_hits_total, cache_misses_total
                nonlocal t_extract_total, t_predict_total, files_done
                for url, score, warc_filename, record_index in result.rows:
                    writer.writerow([url, f"{score:.6f}", warc_filename, record_index])
                    scores.append(score)
                if sink is sys.stdout:
                    sink.flush()
                processed += result.processed
                skipped_empty += result.skipped_empty
                skipped_homepage += result.skipped_homepage
                skipped_resume_total += result.skipped_resume
                extract_errors_total += result.extract_errors
                cache_hits_total += result.cache_hits
                cache_misses_total += result.cache_misses
                t_extract_total += result.t_extract
                t_predict_total += result.t_predict
                files_done += 1
                logger.info(
                    "Finished %s — processed=%d skipped_empty=%d "
                    "skipped_homepage=%d skipped_resume=%d extract_errors=%d",
                    result.uri,
                    processed,
                    skipped_empty,
                    skipped_homepage,
                    skipped_resume_total,
                    extract_errors_total,
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

            empty_skip: frozenset[int] = frozenset()
            if args.workers == 1:
                for uri in resolved:
                    _aggregate(
                        process_one_file(
                            uri, model, model_lock, args, skip_by_warc.get(uri, empty_skip)
                        )
                    )
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
                            initargs=(args.model_repo, args.model_file),
                        ) as pool:
                            payloads = [
                                (u, args, skip_by_warc.get(u, empty_skip)) for u in remaining_uris
                            ]
                            for result in pool.map(_process_pool_worker, payloads):
                                _aggregate(result)
                                completed_in_pool += 1
                        break  # all remaining_uris finished
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
                                "Last suspect: %s. Use --resume-from-output to "
                                "continue from the partial output.",
                                restarts,
                                max_restarts,
                                suspect,
                            )
                            raise
                        logger.error(
                            "Process pool worker died after %d files in this pool; "
                            "dropping suspect %s and restarting (restart %d/%s). "
                            "The dropped file will be missing from this output; "
                            "rerun with --resume-from-output to retry it.",
                            completed_in_pool,
                            suspect,
                            restarts,
                            "∞" if max_restarts == 0 else str(max_restarts),
                        )
                        remaining_uris = remaining_uris[suspect_idx + 1 :]
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for result in pool.map(
                        lambda u: process_one_file(
                            u, model, model_lock, args, skip_by_warc.get(u, empty_skip)
                        ),
                        resolved,
                    ):
                        _aggregate(result)

        t_processing = time.perf_counter() - t_processing_start

        logger.info(
            "Total response records classified: %d "
            "(skipped %d empty extractions, %d homepages, %d resumed, %d extract errors)",
            processed,
            skipped_empty,
            skipped_homepage,
            skipped_resume_total,
            extract_errors_total,
        )
        if args.cache_dir:
            logger.info(
                "Extraction cache totals — hits=%d misses=%d",
                cache_hits_total,
                cache_misses_total,
            )
        log_summary(scores)
        log_timing(processed, t_processing, t_extract_total, t_predict_total)

        if traf_counter.warnings or traf_counter.errors:
            logger.info(
                "Suppressed trafilatura logs: warnings=%d errors=%d",
                traf_counter.warnings,
                traf_counter.errors,
            )

        if summary_uri:
            finished_at = _dt.datetime.now(_dt.UTC).isoformat()
            write_run_summary(
                summary_uri,
                summary_storage_options,
                args=args,
                resolved_count=len(resolved),
                scores=scores,
                processed=processed,
                skipped_empty=skipped_empty,
                skipped_homepage=skipped_homepage,
                skipped_resume=skipped_resume_total,
                extract_errors=extract_errors_total,
                cache_hits=cache_hits_total,
                cache_misses=cache_misses_total,
                trafilatura_warnings=traf_counter.warnings,
                trafilatura_errors=traf_counter.errors,
                t_processing=t_processing,
                t_extract_total=t_extract_total,
                t_predict_total=t_predict_total,
                started_at=started_at,
                finished_at=finished_at,
            )
            logger.info("Wrote summary %s", summary_uri)

        traf_logger.removeHandler(traf_counter)
        traf_logger.propagate = traf_prev_propagate

        return 0
