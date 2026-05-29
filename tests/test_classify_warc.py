from __future__ import annotations

import argparse
import csv
import gzip
import logging
import sys

import pytest

from ccoa.classifier.fasttext import FASTTEXT_MAX_INPUT_CHARS, clean_for_fasttext
from ccoa.commands.classify_warc import (
    ClassifyWarcCommand,
    _resolve_model_slots,
    load_resume_skipset,
)
from ccoa.extraction.cache import (
    cache_path_for_warc,
    load_extraction_cache,
    save_extraction_cache,
)
from ccoa.extraction.text import TrafilaturaLogCounter
from ccoa.utils.fs.paths import open_output_sink, resolve_warc_paths, summary_path_for
from ccoa.utils.reporting.summary import (
    compute_score_stats,
    format_duration,
    log_summary,
    log_timing,
    write_run_summary,
)
from ccoa.utils.warc.urls import is_homepage_url


def _make_warcs(tmp_path, names):
    """Create empty files named `names` under `tmp_path` and return their str paths."""
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"")
        paths.append(str(p))
    return paths


def test_summary_handles_empty(caplog):
    """An empty mapping logs a single warning and returns cleanly."""
    caplog.set_level(logging.WARNING, logger="ccoa.utils.reporting.summary")
    log_summary({})
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "No records classified" in warnings[0].getMessage()


def test_summary_computes_stats(caplog):
    """Each column emits an INFO line containing count/min/max/mean for its scores."""
    caplog.set_level(logging.INFO, logger="ccoa.utils.reporting.summary")
    log_summary(
        {
            "score___label__science": [0.1, 0.5, 0.9],
            "score___label__cc": [0.9, 0.5, 0.1],
        }
    )
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert messages, "expected INFO summary lines"
    text = "\n".join(messages)
    assert "[score___label__science]" in text
    assert "[score___label__cc]" in text
    assert text.count("count=3") == 2
    assert "min=0.100000" in text
    assert "max=0.900000" in text
    assert "mean=0.500000" in text


def test_timing_zero_processed(caplog):
    """With zero processed records, log_timing reports no timing data."""
    caplog.set_level(logging.INFO, logger="ccoa.utils.reporting.summary")
    log_timing(processed=0, t_processing=0.0, t_extract_total=0.0, t_predict_total=0.0)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("no timing data" in m for m in messages)


def test_timing_reports_throughput(caplog):
    """log_timing emits docs/s, total, and per-step costs in the right format."""
    caplog.set_level(logging.INFO, logger="ccoa.utils.reporting.summary")
    log_timing(processed=10, t_processing=2.0, t_extract_total=0.5, t_predict_total=1.0)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert messages
    line = "\n".join(messages)
    assert "processed=10" in line
    assert "throughput=5.00 docs/s" in line
    assert "extract_total=500.00ms" in line
    assert "predict_total=1.000s" in line


def test_open_output_sink_stdout_does_not_close(capsys):
    """`-` yields sys.stdout and the context manager must not close it."""
    with open_output_sink("-", {}) as sink:
        assert sink is sys.stdout
        csv.writer(sink).writerow(["a", "b"])
    captured = capsys.readouterr()
    assert "a,b" in captured.out
    assert not sys.stdout.closed


def test_open_output_sink_writes_local_file(tmp_path):
    """A local path goes through fsspec.open and round-trips CSV content."""
    target = tmp_path / "out.csv"
    with open_output_sink(str(target), {}) as sink:
        writer = csv.writer(sink)
        writer.writerow(["URL", "prediction_score"])
        writer.writerow(["http://example.com", "0.123456"])
    assert target.read_text().splitlines() == [
        "URL,prediction_score",
        "http://example.com,0.123456",
    ]


def test_resolve_paths_passes_literal_through(tmp_path):
    """A literal (non-glob) path is returned unchanged, in input order."""
    target = tmp_path / "only.warc.gz"
    target.write_bytes(b"")
    resolved = resolve_warc_paths(
        [str(target)],
        anonymous=False,
        requester_pays=False,
        shuffle=False,
        seed=42,
        files_limit=0,
    )
    assert resolved == [str(target)]


def test_resolve_paths_globs_and_sorts(tmp_path):
    """A glob pattern expands and the result is sorted lexicographically."""
    _make_warcs(tmp_path, ["c.warc.gz", "a.warc.gz", "b.warc.gz"])
    resolved = resolve_warc_paths(
        [str(tmp_path / "*.warc.gz")],
        anonymous=False,
        requester_pays=False,
        shuffle=False,
        seed=42,
        files_limit=0,
    )
    assert resolved == [
        str(tmp_path / "a.warc.gz"),
        str(tmp_path / "b.warc.gz"),
        str(tmp_path / "c.warc.gz"),
    ]


def test_resolve_paths_shuffle_deterministic(tmp_path):
    """Same seed → same shuffled order; different seed → different order."""
    _make_warcs(
        tmp_path,
        ["a.warc.gz", "b.warc.gz", "c.warc.gz", "d.warc.gz", "e.warc.gz"],
    )
    pattern = str(tmp_path / "*.warc.gz")

    def call(seed: int) -> list[str]:
        return resolve_warc_paths(
            [pattern],
            anonymous=False,
            requester_pays=False,
            shuffle=True,
            seed=seed,
            files_limit=0,
        )

    first = call(42)
    second = call(42)
    other = call(7)
    assert first == second
    assert sorted(first) == sorted(other)
    assert first != other


def test_resolve_paths_files_limit_truncates(tmp_path):
    """`files_limit` keeps only the first N URIs after sort/shuffle."""
    _make_warcs(
        tmp_path,
        ["a.warc.gz", "b.warc.gz", "c.warc.gz", "d.warc.gz", "e.warc.gz"],
    )
    resolved = resolve_warc_paths(
        [str(tmp_path / "*.warc.gz")],
        anonymous=False,
        requester_pays=False,
        shuffle=False,
        seed=42,
        files_limit=2,
    )
    assert resolved == [
        str(tmp_path / "a.warc.gz"),
        str(tmp_path / "b.warc.gz"),
    ]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://example.com", True),
        ("http://example.com/", True),
        ("https://www.example.com/", True),
        ("http://example.com/page", False),
        ("http://example.com/?q=1", False),
        ("http://example.com/#frag", False),
        ("http://example.com/index.html", False),
        ("", True),
    ],
)
def test_is_homepage_url(url: str, expected: bool) -> None:
    """Site-root URLs (no path/query/fragment) are homepages; everything else is not."""
    assert is_homepage_url(url) is expected


def _default_args(**overrides) -> argparse.Namespace:
    """Build a complete argparse.Namespace covering every classify-warc flag."""
    base = {
        "workers": 1,
        "records_limit": 0,
        "warc_paths": [],
        "records_per_file_limit": 0,
        "skip_homepages": False,
        "files_limit": 0,
        "shuffle_files": False,
        "seed": 42,
        "model_repo": [],
        "model_file": [],
        "labels": [],
        "output": "-",
        "cache_dir": None,
        "anonymous_s3": False,
        "s3_requester_pays": False,
        "progress_every": 1000,
        "workers_mode": "thread",
        "max_pool_restarts": 10,
        "resume_from_output": None,
        "overwrite": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_workers_with_records_limit_errors(caplog):
    """`--workers > 1` combined with `--records-limit > 0` exits early with code 2."""
    caplog.set_level(logging.ERROR, logger="ccoa.commands.classify_warc")
    rc = ClassifyWarcCommand().run(_default_args(workers=2, records_limit=10))
    assert rc == 2
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "expected an ERROR log line"
    message = errors[0].getMessage()
    assert "--records-limit" in message
    assert "--workers" in message


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("foo.csv", "foo.summary.csv"),
        ("data/out.tsv", "data/out.summary.tsv"),
        ("s3://bucket/key.csv", "s3://bucket/key.summary.csv"),
        ("path/to/output", "path/to/output.summary"),
    ],
)
def test_summary_path_for(output: str, expected: str) -> None:
    """`.summary` is inserted before the final extension."""
    assert summary_path_for(output) == expected


def test_run_aborts_when_output_exists(tmp_path, caplog):
    """A pre-existing --output path causes an early-return rc=2 before any processing."""
    caplog.set_level(logging.ERROR, logger="ccoa.commands.classify_warc")
    target = tmp_path / "out.csv"
    target.write_text("preexisting,data\n")
    rc = ClassifyWarcCommand().run(_default_args(output=str(target)))
    assert rc == 2
    assert any("already exists" in r.getMessage() for r in caplog.records)


def test_run_aborts_when_summary_exists(tmp_path, caplog):
    """A pre-existing summary sidecar path also causes an early-return rc=2."""
    caplog.set_level(logging.ERROR, logger="ccoa.commands.classify_warc")
    target = tmp_path / "out.csv"
    summary = tmp_path / "out.summary.csv"
    summary.write_text("stale,summary\n")
    rc = ClassifyWarcCommand().run(_default_args(output=str(target)))
    assert rc == 2
    assert not target.exists(), "output should not have been created"
    assert any("already exists" in r.getMessage() for r in caplog.records)


def test_run_overwrite_replaces_existing_output(monkeypatch, tmp_path, caplog):
    """`--overwrite` skips the exists-guard and warns; the run proceeds and rewrites the file."""
    warc_path = tmp_path / "empty.warc.gz"
    _write_synthetic_warc(warc_path, "http://example.com/", b"<html></html>")
    output_path = tmp_path / "out.csv"
    output_path.write_text("STALE,DATA\n")  # pre-existing
    summary_path = tmp_path / "out.summary.csv"
    summary_path.write_text("stale,summary\n")

    fake = _FakeModel(("__label__a",))
    _patch_models(monkeypatch, {("r/x", "x.bin"): fake})

    args = _default_args(
        warc_paths=[str(warc_path)],
        output=str(output_path),
        model_repo=["r/x"],
        model_file=["x.bin"],
        overwrite=True,
    )
    caplog.set_level(logging.WARNING, logger="ccoa.commands.classify_warc")
    rc = ClassifyWarcCommand().run(args)
    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Overwriting existing output" in m for m in warnings)
    # The stale content must be replaced — fresh header on the first line.
    assert output_path.read_text().splitlines()[0] == (
        "URL,score___label__a,warc_filename,warc_record_index"
    )


def test_compute_score_stats_empty_and_singleton() -> None:
    """Empty -> {count: 0}; singleton -> n/a percentiles + stdev."""
    assert compute_score_stats([]) == {"count": 0}
    one = compute_score_stats([0.5])
    assert one["count"] == 1
    assert one["min"] == one["max"] == one["mean"] == 0.5
    assert one["stdev"] == "n/a"
    assert one["p50"] == "n/a"


def test_write_run_summary_round_trip(tmp_path):
    """The sidecar CSV captures args, counters, per-column score stats, and timings."""
    summary_path = tmp_path / "out.summary.csv"
    args = _default_args(
        warc_paths=["s3://bucket/a.warc.gz", "s3://bucket/b.warc.gz"],
        records_per_file_limit=10,
        skip_homepages=True,
        workers=2,
    )
    write_run_summary(
        str(summary_path),
        {},
        args=args,
        resolved_count=2,
        scores_by_column={
            "score___label__science": [0.1, 0.5, 0.9],
            "score___label__cc": [0.9, 0.5, 0.1],
        },
        processed=3,
        skipped_empty=1,
        skipped_homepage=2,
        skipped_resume=11,
        extract_errors=8,
        cache_hits=4,
        cache_misses=5,
        trafilatura_warnings=17,
        trafilatura_errors=6,
        t_processing=1.5,
        t_extract_total=0.6,
        t_predict_total=0.3,
        started_at="2026-05-24T00:00:00+00:00",
        finished_at="2026-05-24T00:00:01+00:00",
    )
    with open(summary_path, newline="", encoding="utf-8") as fh:
        rows = dict(list(csv.reader(fh))[1:])
    assert rows["arg.workers"] == "2"
    assert rows["arg.records_per_file_limit"] == "10"
    assert rows["arg.skip_homepages"] == "True"
    assert rows["arg.warc_paths"] == "s3://bucket/a.warc.gz;s3://bucket/b.warc.gz"
    assert rows["arg.cache_dir"] == ""
    assert rows["input.resolved_count"] == "2"
    assert rows["count.processed"] == "3"
    assert rows["count.skipped_homepage"] == "2"
    assert rows["count.skipped_resume"] == "11"
    assert rows["count.extract_errors"] == "8"
    assert rows["count.cache_hits"] == "4"
    assert rows["count.trafilatura_warnings"] == "17"
    assert rows["count.trafilatura_errors"] == "6"
    assert rows["score.score___label__science.count"] == "3"
    assert rows["score.score___label__science.min"] == "0.100000"
    assert rows["score.score___label__science.max"] == "0.900000"
    assert rows["score.score___label__science.mean"] == "0.500000"
    assert rows["score.score___label__cc.count"] == "3"
    assert rows["score.score___label__cc.mean"] == "0.500000"
    assert rows["time.total_seconds"] == "1.500000"
    assert rows["time.throughput_docs_per_sec"] == "2.000000"
    assert rows["run.started_at"] == "2026-05-24T00:00:00+00:00"
    assert rows["run.finished_at"] == "2026-05-24T00:00:01+00:00"
    assert "run.cli" in rows


def test_trafilatura_log_counter():
    """Counter splits WARNING vs ERROR+ and never re-emits the record."""
    h = TrafilaturaLogCounter()
    for level in (logging.WARNING, logging.WARNING, logging.ERROR, logging.CRITICAL):
        h.handle(logging.LogRecord("trafilatura.core", level, "x.py", 1, "noisy", None, None))
    assert h.warnings == 2
    assert h.errors == 2


def test_cache_path_for_warc_maps_protocols(tmp_path):
    """Input URIs of any scheme map to <cache_dir>/<scheme>/<path>.jsonl.gz."""
    cache_dir = "/tmp/cache"
    assert (
        cache_path_for_warc("s3://commoncrawl/crawl-data/CC-MAIN/foo.warc.gz", cache_dir)
        == "/tmp/cache/s3/commoncrawl/crawl-data/CC-MAIN/foo.warc.gz.jsonl.gz"
    )
    assert (
        cache_path_for_warc("/data/foo.warc.gz", cache_dir)
        == "/tmp/cache/local/data/foo.warc.gz.jsonl.gz"
    )
    assert (
        cache_path_for_warc("data/foo.warc.gz", cache_dir)
        == "/tmp/cache/local/data/foo.warc.gz.jsonl.gz"
    )
    assert (
        cache_path_for_warc("gs://bucket/x.warc.gz", cache_dir)
        == "/tmp/cache/gs/bucket/x.warc.gz.jsonl.gz"
    )
    # Trailing slash on cache_dir is normalised.
    assert (
        cache_path_for_warc("s3://b/k.warc.gz", "/tmp/cache/")
        == "/tmp/cache/s3/b/k.warc.gz.jsonl.gz"
    )
    # S3 cache_dir is preserved verbatim.
    assert (
        cache_path_for_warc("s3://b/k.warc.gz", "s3://cache-bucket/prefix")
        == "s3://cache-bucket/prefix/s3/b/k.warc.gz.jsonl.gz"
    )


def test_cache_path_rejects_parent_traversal():
    """A `..` segment in the input must be rejected to prevent cache_dir escape."""
    with pytest.raises(ValueError, match=r"\.\."):
        cache_path_for_warc("../etc/foo.warc.gz", "/tmp/cache")
    with pytest.raises(ValueError, match=r"\.\."):
        cache_path_for_warc("s3://bucket/a/../../foo.warc.gz", "/tmp/cache")


def test_cache_roundtrip(tmp_path):
    """Save → load returns the identical dict, including empties, multi-line, non-ASCII."""
    target = str(tmp_path / "cache.jsonl.gz")
    entries = {
        0: "hello",
        1: "",  # negative caching
        2: "line one\nline two",
        3: "ümlaut and 漢字",
    }
    save_extraction_cache(target, entries, {})
    loaded = load_extraction_cache(target, {})
    assert loaded == entries
    # Sanity: keys come back as ints, not strs.
    assert all(isinstance(k, int) for k in loaded)


def test_load_missing_returns_empty_dict(tmp_path):
    """Loading from a path that does not exist returns {} without raising."""
    missing = str(tmp_path / "does_not_exist.jsonl.gz")
    assert load_extraction_cache(missing, {}) == {}


def test_load_malformed_returns_empty_and_warns(tmp_path, caplog):
    """A garbage file is treated as a cache miss and emits a WARNING."""
    target = tmp_path / "bad.jsonl.gz"
    target.write_bytes(b"this is not gzipped JSON")
    caplog.set_level(logging.WARNING, logger="ccoa.extraction.cache")
    result = load_extraction_cache(str(target), {})
    assert result == {}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a WARNING about the broken cache"
    assert "extraction cache" in warnings[0].getMessage().lower()


def test_save_skips_empty_dict(tmp_path):
    """save_extraction_cache must not create a file when the dict is empty."""
    target = tmp_path / "should_not_exist.jsonl.gz"
    save_extraction_cache(str(target), {}, {})
    assert not target.exists()


def test_save_writes_valid_gzip_jsonl(tmp_path):
    """Verify the on-disk format: gzip(jsonl) with index/text keys, sorted by index."""
    target = tmp_path / "cache.jsonl.gz"
    save_extraction_cache(str(target), {2: "c", 0: "a", 1: "b"}, {})
    with gzip.open(target, "rt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert lines == [
        '{"index": 0, "text": "a"}',
        '{"index": 1, "text": "b"}',
        '{"index": 2, "text": "c"}',
    ]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0.0s"),
        (12.3, "12.3s"),
        (59.9, "59.9s"),
        (60.0, "1m 0s"),
        (125.0, "2m 5s"),
        (3600.0, "1h 0m 0s"),
        (3725.0, "1h 2m 5s"),
        (-5.0, "0.0s"),
    ],
)
def test_format_duration(seconds: float, expected: str) -> None:
    """Sub-minute durations stay as Ns; minutes/hours roll up; negatives clamp to 0."""
    assert format_duration(seconds) == expected


def test_clean_for_fasttext_strips_newlines_and_nuls() -> None:
    """Newlines become spaces; NULs are deleted; other text passes through."""
    assert clean_for_fasttext("a\nb\rc\x00d") == "a b cd"
    assert clean_for_fasttext("hello world") == "hello world"


def test_clean_for_fasttext_clamps_length() -> None:
    """Long inputs are truncated to max_len characters."""
    big = "x" * (FASTTEXT_MAX_INPUT_CHARS + 50)
    assert len(clean_for_fasttext(big)) == FASTTEXT_MAX_INPUT_CHARS
    assert len(clean_for_fasttext(big, max_len=10)) == 10


def test_load_resume_skipset_groups_by_warc(tmp_path):
    """A prior output CSV with the expected header is grouped by warc_filename."""
    csv_path = tmp_path / "prior.csv"
    csv_path.write_text(
        "URL,score___label__science,warc_filename,warc_record_index\n"
        "http://a/,0.1,s3://bucket/a.warc.gz,0\n"
        "http://a/x,0.2,s3://bucket/a.warc.gz,3\n"
        "http://b/,0.3,s3://bucket/b.warc.gz,7\n"
    )
    expected = ["URL", "score___label__science", "warc_filename", "warc_record_index"]
    skipset = load_resume_skipset(str(csv_path), {}, expected)
    assert skipset == {
        "s3://bucket/a.warc.gz": frozenset({0, 3}),
        "s3://bucket/b.warc.gz": frozenset({7}),
    }


def test_load_resume_skipset_rejects_pre_multi_label_csv(tmp_path):
    """A pre-multi-label CSV (no `score_*` columns) is rejected with a clear message."""
    csv_path = tmp_path / "old.csv"
    csv_path.write_text("URL,prediction_score,warc_filename,warc_record_index\nhttp://a/,0.1,x,0\n")
    expected = ["URL", "score___label__science", "warc_filename", "warc_record_index"]
    with pytest.raises(ValueError, match="pre-multi-label"):
        load_resume_skipset(str(csv_path), {}, expected)


def test_load_resume_skipset_rejects_column_reorder(tmp_path):
    """Same columns in a different order: rejected (concat-friendliness)."""
    csv_path = tmp_path / "reordered.csv"
    csv_path.write_text(
        "URL,warc_filename,warc_record_index,score___label__science\nhttp://a/,x,0,0.1\n"
    )
    expected = ["URL", "score___label__science", "warc_filename", "warc_record_index"]
    with pytest.raises(ValueError, match="header does not match"):
        load_resume_skipset(str(csv_path), {}, expected)


def test_load_resume_skipset_rejects_extra_columns(tmp_path):
    """Prior CSV has columns the new run doesn't produce: rejected with named diff."""
    csv_path = tmp_path / "extra.csv"
    csv_path.write_text(
        "URL,score___label__science,score___label__cc,warc_filename,warc_record_index\n"
        "http://a/,0.1,0.9,x,0\n"
    )
    expected = ["URL", "score___label__science", "warc_filename", "warc_record_index"]
    with pytest.raises(ValueError, match="score___label__cc"):
        load_resume_skipset(str(csv_path), {}, expected)


def _write_synthetic_warc(warc_path, url: str, body: bytes) -> None:
    """Write a single-response-record .warc.gz at `warc_path`."""
    import io

    from warcio.statusandheaders import StatusAndHeaders
    from warcio.warcwriter import WARCWriter

    with open(warc_path, "wb") as out:
        writer = WARCWriter(out, gzip=True)
        http_headers = StatusAndHeaders(
            "200 OK",
            [("Content-Type", "text/html"), ("Content-Length", str(len(body)))],
            protocol="HTTP/1.0",
        )
        record = writer.create_warc_record(
            url,
            "response",
            payload=io.BytesIO(body),
            length=len(body),
            http_headers=http_headers,
        )
        writer.write_record(record)


@pytest.mark.real_model
def test_classify_warc_end_to_end(tmp_path):
    """End-to-end: synthetic local WARC -> classify-warc -> CSV + summary sidecar.

    Downloads `facebook/fasttext-language-identification` (~131 MB) into the
    HuggingFace cache on first run. Skipped by default; enable with
    `uv run pytest --run-real`.
    """
    warc_path = tmp_path / "test.warc.gz"
    html = (
        b"<!DOCTYPE html><html><head><title>Test</title></head><body>"
        b"<h1>About this Test Page</h1>"
        b"<p>This is an English language test page designed to provide enough text "
        b"content for trafilatura to extract and for a language identification "
        b"classifier to score with reasonable confidence. The page discusses nothing "
        b"in particular but uses common English vocabulary throughout.</p>"
        b"<p>The fasttext language identification model should identify this content "
        b"as English with high probability. Different languages would produce "
        b"different labels and lower confidence on this same text.</p>"
        b"</body></html>"
    )
    _write_synthetic_warc(warc_path, "http://example.com/en.html", html)

    output_path = tmp_path / "out.csv"
    args = _default_args(
        warc_paths=[str(warc_path)],
        output=str(output_path),
        model_repo=["facebook/fasttext-language-identification"],
        model_file=["model.bin"],
        labels=["__label__eng_Latn"],
    )
    rc = ClassifyWarcCommand().run(args)
    assert rc == 0

    with output_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "URL",
        "score___label__eng_Latn",
        "warc_filename",
        "warc_record_index",
    ]
    assert len(rows) == 2, f"expected 1 data row, got {len(rows) - 1}"

    url, score, warc_filename, record_index = rows[1]
    assert url == "http://example.com/en.html"
    assert 0.0 <= float(score) <= 1.0
    assert warc_filename == str(warc_path)
    assert record_index == "0"

    summary_path = tmp_path / "out.summary.csv"
    assert summary_path.exists(), "sidecar summary CSV should be written"
    with summary_path.open(newline="", encoding="utf-8") as fh:
        summary_rows = dict(list(csv.reader(fh))[1:])
    assert summary_rows["count.processed"] == "1"
    assert summary_rows["input.resolved_count"] == "1"
    assert summary_rows["score.score___label__eng_Latn.count"] == "1"


def test_resolve_model_slots_defaults_when_both_empty():
    """No CLI input → falls back to the default science model with '*' labels."""
    slots = _resolve_model_slots([], [], [])
    assert slots == [
        ("ibm-granite/GneissWeb.Sci_classifier", "fasttext_science.bin", ["*"]),
    ]


def test_resolve_model_slots_zips_parallel_lists():
    """Parallel `--model-repo`/`--model-file`/`--labels` lists are zipped positionally."""
    slots = _resolve_model_slots(
        ["r1/m", "r2/m"],
        ["f1.bin", "f2.bin"],
        ["__label__a,__label__b", "*"],
    )
    assert slots == [
        ("r1/m", "f1.bin", ["__label__a", "__label__b"]),
        ("r2/m", "f2.bin", ["*"]),
    ]


def test_resolve_model_slots_broadcasts_missing_labels():
    """Omitted `--labels` defaults to '*' for every model slot."""
    slots = _resolve_model_slots(["r1/m", "r2/m"], ["f1.bin", "f2.bin"], [])
    assert slots == [
        ("r1/m", "f1.bin", ["*"]),
        ("r2/m", "f2.bin", ["*"]),
    ]


def test_resolve_model_slots_rejects_length_mismatch():
    """`--model-repo` and `--model-file` of different lengths fails."""
    with pytest.raises(ValueError, match="same length"):
        _resolve_model_slots(["r1", "r2"], ["f1.bin"], [])


def test_resolve_model_slots_rejects_label_length_mismatch():
    """A `--labels` list whose length doesn't match the number of models fails."""
    with pytest.raises(ValueError, match="one entry per --model-repo"):
        _resolve_model_slots(["r1", "r2"], ["f1.bin", "f2.bin"], ["only_one"])


class _FakeModel:
    """Stand-in for a loaded fasttext model.

    `get_labels` returns the labels we want exposed to '*' expansion; `predict`
    deterministically returns a 1.0 for the first label and 0.0 for the rest,
    so the test can assert on output values without depending on the real model.
    """

    def __init__(self, labels: tuple[str, ...]):
        self._labels = labels

    def get_labels(self) -> tuple[str, ...]:
        return self._labels

    def predict(self, _text: str, k: int = -1):
        import numpy as np

        probs = np.zeros(len(self._labels), dtype=float)
        if probs.size:
            probs[0] = 1.0
        return (tuple(self._labels), probs)


def _patch_models(monkeypatch, model_for: dict[tuple[str, str], _FakeModel]) -> None:
    """Patch `load_classifier`/`get_model_labels` to return `_FakeModel`s by (repo, file)."""

    def fake_load(repo: str, file: str):
        try:
            return model_for[(repo, file)]
        except KeyError as exc:
            raise AssertionError(f"unexpected load_classifier({repo!r}, {file!r})") from exc

    def fake_get_labels(model):
        return tuple(model.get_labels())

    monkeypatch.setattr("ccoa.commands.classify_warc.load_classifier", fake_load)
    monkeypatch.setattr("ccoa.commands.classify_warc.get_model_labels", fake_get_labels)


def test_run_expands_star_labels_from_model(monkeypatch, tmp_path):
    """'*' expands to the model's full label set via `get_model_labels`."""
    warc_path = tmp_path / "empty.warc.gz"
    _write_synthetic_warc(warc_path, "http://example.com/", b"<html></html>")
    fake = _FakeModel(("__label__alpha", "__label__beta"))
    _patch_models(monkeypatch, {("r/x", "x.bin"): fake})

    output_path = tmp_path / "out.csv"
    args = _default_args(
        warc_paths=[str(warc_path)],
        output=str(output_path),
        model_repo=["r/x"],
        model_file=["x.bin"],
    )
    rc = ClassifyWarcCommand().run(args)
    assert rc == 0

    with output_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "URL",
        "score___label__alpha",
        "score___label__beta",
        "warc_filename",
        "warc_record_index",
    ]


def test_run_writes_multi_model_header(monkeypatch, tmp_path):
    """Two models: columns get a `m<idx>_` prefix so per-model labels never collide."""
    warc_path = tmp_path / "empty.warc.gz"
    _write_synthetic_warc(warc_path, "http://example.com/", b"<html></html>")
    fake_a = _FakeModel(("__label__sci", "__label__cc"))
    fake_b = _FakeModel(("__label__hq", "__label__lq"))
    _patch_models(
        monkeypatch,
        {
            ("rA/sci", "sci.bin"): fake_a,
            ("rB/q", "q.bin"): fake_b,
        },
    )

    output_path = tmp_path / "out.csv"
    args = _default_args(
        warc_paths=[str(warc_path)],
        output=str(output_path),
        model_repo=["rA/sci", "rB/q"],
        model_file=["sci.bin", "q.bin"],
    )
    rc = ClassifyWarcCommand().run(args)
    assert rc == 0

    with output_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "URL",
        "score_m0___label__sci",
        "score_m0___label__cc",
        "score_m1___label__hq",
        "score_m1___label__lq",
        "warc_filename",
        "warc_record_index",
    ]


def test_run_disambiguates_shared_label_across_models(monkeypatch, tmp_path):
    """Two models sharing a label name (e.g. `__label__cc`) coexist via the `m<idx>_` prefix."""
    warc_path = tmp_path / "empty.warc.gz"
    _write_synthetic_warc(warc_path, "http://example.com/", b"<html></html>")
    fake_a = _FakeModel(("__label__shared",))
    fake_b = _FakeModel(("__label__shared",))
    _patch_models(
        monkeypatch,
        {
            ("rA/m", "a.bin"): fake_a,
            ("rB/m", "b.bin"): fake_b,
        },
    )

    output_path = tmp_path / "out.csv"
    args = _default_args(
        warc_paths=[str(warc_path)],
        output=str(output_path),
        model_repo=["rA/m", "rB/m"],
        model_file=["a.bin", "b.bin"],
    )
    rc = ClassifyWarcCommand().run(args)
    assert rc == 0

    with output_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "URL",
        "score_m0___label__shared",
        "score_m1___label__shared",
        "warc_filename",
        "warc_record_index",
    ]


def test_run_rejects_mismatched_model_lengths(caplog):
    """--model-repo and --model-file of different lengths: rc=2, no work done."""
    caplog.set_level(logging.ERROR, logger="ccoa.commands.classify_warc")
    args = _default_args(
        model_repo=["r1/m", "r2/m"],
        model_file=["only_one.bin"],
        output="-",
    )
    rc = ClassifyWarcCommand().run(args)
    assert rc == 2


def test_run_scores_all_resolved_labels(monkeypatch, tmp_path):
    """Each record is scored against every requested label; outputs land in column order."""
    warc_path = tmp_path / "doc.warc.gz"
    html = (
        b"<!DOCTYPE html><html><head><title>T</title></head><body>"
        b"<p>Long enough body to survive trafilatura's minimum-content gates. "
        b"We just need a single response record to flow through the pipeline.</p>"
        b"</body></html>"
    )
    _write_synthetic_warc(warc_path, "http://example.com/doc", html)
    fake = _FakeModel(("__label__alpha", "__label__beta"))
    _patch_models(monkeypatch, {("r/x", "x.bin"): fake})

    output_path = tmp_path / "out.csv"
    args = _default_args(
        warc_paths=[str(warc_path)],
        output=str(output_path),
        model_repo=["r/x"],
        model_file=["x.bin"],
    )
    rc = ClassifyWarcCommand().run(args)
    assert rc == 0

    with output_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "URL",
        "score___label__alpha",
        "score___label__beta",
        "warc_filename",
        "warc_record_index",
    ]
    assert len(rows) == 2
    url, alpha, beta, warc_filename, record_index = rows[1]
    assert url == "http://example.com/doc"
    # `_FakeModel.predict` is rigged: 1.0 for the first label, 0.0 for the rest.
    assert float(alpha) == 1.0
    assert float(beta) == 0.0
    assert warc_filename == str(warc_path)
    assert record_index == "0"
