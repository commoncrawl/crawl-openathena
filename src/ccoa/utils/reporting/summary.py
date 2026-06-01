"""Score statistics, timing helpers, and the sidecar-summary CSV writer."""

from __future__ import annotations

import argparse
import csv
import logging
import shlex
import statistics
import sys

import fsspec

logger = logging.getLogger(__name__)


def format_seconds(seconds: float) -> str:
    """Format a duration as `Xs` (>= 1s) or `Yms` (< 1s)."""
    if seconds >= 1:
        return f"{seconds:.3f}s"
    return f"{seconds * 1000:.2f}ms"


def format_duration(seconds: float) -> str:
    """Format a longer duration as h/m/s (e.g. `1h 23m 45s`, `23m 45s`, `45.3s`)."""
    if seconds < 0:
        seconds = 0.0
    if seconds >= 3600:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    return f"{seconds:.1f}s"


def log_timing(
    processed: int,
    t_processing: float,
    t_extract_total: float,
    t_predict_total: float,
) -> None:
    """Log a one-line summary of wall-clock time, throughput, and per-step costs."""
    if processed == 0:
        logger.info("runtime — processed=0; no timing data to report.")
        return

    docs_per_sec = processed / t_processing if t_processing > 0 else float("inf")
    logger.info(
        "runtime — processed=%d total=%s throughput=%.2f docs/s mean_per_doc=%s "
        "extract_total=%s extract_mean=%s predict_total=%s predict_mean=%s",
        processed,
        format_seconds(t_processing),
        docs_per_sec,
        format_seconds(t_processing / processed),
        format_seconds(t_extract_total),
        format_seconds(t_extract_total / processed),
        format_seconds(t_predict_total),
        format_seconds(t_predict_total / processed),
    )


def format_score(value: float | str) -> str:
    """Format a score as a fixed-precision float, passing through `"n/a"`."""
    if isinstance(value, str):
        return value
    return f"{value:.6f}"


def compute_score_stats(scores: list[float]) -> dict[str, float | int | str]:
    """Return summary statistics over `scores`.

    Keys: `count`, `min`, `max`, `mean`, `median`, `stdev`, plus percentiles
    `p10/p25/p50/p75/p90/p95/p99`. `stdev` and percentiles are the string
    `"n/a"` when `count < 2`. An empty list returns `{"count": 0}`.
    """
    if not scores:
        return {"count": 0}

    count = len(scores)
    stats: dict[str, float | int | str] = {
        "count": count,
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
    }
    if count >= 2:
        quantiles = statistics.quantiles(scores, n=100, method="inclusive")
        stats["stdev"] = statistics.stdev(scores)
        stats["p10"] = quantiles[9]
        stats["p25"] = quantiles[24]
        stats["p50"] = quantiles[49]
        stats["p75"] = quantiles[74]
        stats["p90"] = quantiles[89]
        stats["p95"] = quantiles[94]
        stats["p99"] = quantiles[98]
    else:
        stats["stdev"] = "n/a"
        for key in ("p10", "p25", "p50", "p75", "p90", "p95", "p99"):
            stats[key] = "n/a"
    return stats


def log_summary(scores_by_column: dict[str, list[float]]) -> None:
    """Log one INFO line per score column with count/percentiles/mean.

    `scores_by_column` is keyed by output column name (e.g. `score___label__science`)
    and preserves insertion order. An empty dict, or one whose every value list
    is empty, logs a single warning instead.
    """
    if not scores_by_column or not any(scores_by_column.values()):
        logger.warning("No records classified.")
        return

    for column, scores in scores_by_column.items():
        if not scores:
            logger.warning("No scores collected for %s.", column)
            continue
        stats = compute_score_stats(scores)
        logger.info(
            "score stats [%s] — count=%d min=%s p10=%s p25=%s p50=%s p75=%s p90=%s p95=%s p99=%s max=%s mean=%s median=%s stdev=%s",
            column,
            stats["count"],
            format_score(stats["min"]),
            format_score(stats["p10"]),
            format_score(stats["p25"]),
            format_score(stats["p50"]),
            format_score(stats["p75"]),
            format_score(stats["p90"]),
            format_score(stats["p95"]),
            format_score(stats["p99"]),
            format_score(stats["max"]),
            format_score(stats["mean"]),
            format_score(stats["median"]),
            format_score(stats["stdev"]),
        )


def write_run_summary(
    summary_uri: str,
    storage_options: dict[str, object],
    *,
    args: argparse.Namespace,
    resolved_count: int,
    scores_by_column: dict[str, list[float]],
    processed: int,
    skipped_empty: int,
    skipped_homepage: int,
    skipped_resume: int,
    extract_errors: int,
    cache_hits: int,
    cache_misses: int,
    trafilatura_warnings: int,
    trafilatura_errors: int,
    t_processing: float,
    t_extract_total: float,
    t_predict_total: float,
    started_at: str,
    finished_at: str,
) -> None:
    """Write a sidecar two-column CSV (`key,value`) capturing inputs + results.

    Sections (by key prefix): `run.*` (cli + timestamps), `arg.*` (every CLI flag),
    `input.*` (resolved files), `count.*` (record counters),
    `score.<column>.*` (per-column stats from `compute_score_stats`),
    `time.*` (wall-clock / extract / predict). Designed to be parsed back with
    `csv.reader` or `pandas.read_csv`.
    """
    rows: list[tuple[str, str]] = []

    rows.append(("run.cli", shlex.join(sys.argv)))
    rows.append(("run.started_at", started_at))
    rows.append(("run.finished_at", finished_at))

    for key in sorted(vars(args)):
        rows.append((f"arg.{key}", _format_arg_value(getattr(args, key))))

    rows.append(("input.resolved_count", str(resolved_count)))

    rows.append(("count.processed", str(processed)))
    rows.append(("count.skipped_empty", str(skipped_empty)))
    rows.append(("count.skipped_homepage", str(skipped_homepage)))
    rows.append(("count.skipped_resume", str(skipped_resume)))
    rows.append(("count.extract_errors", str(extract_errors)))
    rows.append(("count.cache_hits", str(cache_hits)))
    rows.append(("count.cache_misses", str(cache_misses)))
    rows.append(("count.trafilatura_warnings", str(trafilatura_warnings)))
    rows.append(("count.trafilatura_errors", str(trafilatura_errors)))

    for column, scores in scores_by_column.items():
        stats = compute_score_stats(scores)
        for key in (
            "count",
            "min",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
            "max",
            "mean",
            "median",
            "stdev",
        ):
            if key in stats:
                value = stats[key]
                rows.append(
                    (
                        f"score.{column}.{key}",
                        format_score(value) if key != "count" else str(value),
                    )
                )

    rows.append(("time.total_seconds", f"{t_processing:.6f}"))
    rows.append(("time.extract_total_seconds", f"{t_extract_total:.6f}"))
    rows.append(("time.predict_total_seconds", f"{t_predict_total:.6f}"))
    if processed > 0:
        rows.append(
            (
                "time.throughput_docs_per_sec",
                f"{processed / t_processing:.6f}" if t_processing > 0 else "inf",
            )
        )
        rows.append(("time.extract_mean_seconds", f"{t_extract_total / processed:.6f}"))
        rows.append(("time.predict_mean_seconds", f"{t_predict_total / processed:.6f}"))

    with fsspec.open(
        summary_uri,
        mode="w",
        newline="",
        encoding="utf-8",
        **storage_options,
    ) as sink:
        writer = csv.writer(sink)
        writer.writerow(["key", "value"])
        for key, value in rows:
            writer.writerow([key, value])


def _format_arg_value(value: object) -> str:
    """Serialise an argparse value for the summary CSV (lists → `;`-joined)."""
    if value is None:
        return ""
    if isinstance(value, bool | int | float):
        return str(value)
    if isinstance(value, list | tuple):
        return ";".join(_format_arg_value(item) for item in value)
    return str(value)


def log_tokens_summary(n_tokens: list[int]) -> None:
    """Log one INFO line summarising the token-count distribution."""
    if not n_tokens:
        logger.warning("No records tokenized.")
        return
    stats = compute_score_stats([float(n) for n in n_tokens])
    logger.info(
        "tokens stats — count=%d min=%s p10=%s p25=%s p50=%s p75=%s p90=%s p95=%s p99=%s "
        "max=%s mean=%s median=%s stdev=%s total=%d",
        stats["count"],
        format_score(stats["min"]),
        format_score(stats["p10"]),
        format_score(stats["p25"]),
        format_score(stats["p50"]),
        format_score(stats["p75"]),
        format_score(stats["p90"]),
        format_score(stats["p95"]),
        format_score(stats["p99"]),
        format_score(stats["max"]),
        format_score(stats["mean"]),
        format_score(stats["median"]),
        format_score(stats["stdev"]),
        sum(n_tokens),
    )


def write_tokenize_summary(
    summary_uri: str,
    storage_options: dict[str, object],
    *,
    args: argparse.Namespace,
    resolved_count: int,
    n_tokens: list[int],
    processed: int,
    skipped_empty: int,
    t_processing: float,
    t_load_total: float,
    t_tokenize_total: float,
    started_at: str,
    finished_at: str,
) -> None:
    """Write a sidecar two-column CSV summarising a `ccoa tokenize` run.

    Sections (by key prefix): `run.*` (cli + timestamps), `arg.*` (every
    CLI flag), `input.*` (resolved files), `count.*` (record counters +
    total tokens), `tokens.*` (per-record token-count stats), `time.*`
    (wall-clock / load / tokenize). Parses back with `csv.reader` or
    `pandas.read_csv`.
    """
    rows: list[tuple[str, str]] = []

    rows.append(("run.cli", shlex.join(sys.argv)))
    rows.append(("run.started_at", started_at))
    rows.append(("run.finished_at", finished_at))

    for key in sorted(vars(args)):
        rows.append((f"arg.{key}", _format_arg_value(getattr(args, key))))

    rows.append(("input.resolved_count", str(resolved_count)))

    rows.append(("count.processed", str(processed)))
    rows.append(("count.skipped_empty", str(skipped_empty)))
    rows.append(("count.total_tokens", str(sum(n_tokens))))

    stats = compute_score_stats([float(n) for n in n_tokens])
    for key in (
        "count",
        "min",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
        "mean",
        "median",
        "stdev",
    ):
        if key in stats:
            value = stats[key]
            rows.append(
                (
                    f"tokens.{key}",
                    format_score(value) if key != "count" else str(value),
                )
            )

    rows.append(("time.total_seconds", f"{t_processing:.6f}"))
    rows.append(("time.load_total_seconds", f"{t_load_total:.6f}"))
    rows.append(("time.tokenize_total_seconds", f"{t_tokenize_total:.6f}"))
    if processed > 0:
        rows.append(
            (
                "time.throughput_docs_per_sec",
                f"{processed / t_processing:.6f}" if t_processing > 0 else "inf",
            )
        )
        rows.append(("time.load_mean_seconds", f"{t_load_total / processed:.6f}"))
        rows.append(("time.tokenize_mean_seconds", f"{t_tokenize_total / processed:.6f}"))

    with fsspec.open(
        summary_uri,
        mode="w",
        newline="",
        encoding="utf-8",
        **storage_options,
    ) as sink:
        writer = csv.writer(sink)
        writer.writerow(["key", "value"])
        for key, value in rows:
            writer.writerow([key, value])
