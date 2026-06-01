from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging

import pyarrow.parquet as pq
import pytest

from ccoa.commands import tokenize as tok_mod
from ccoa.commands.tokenize import TokenizeCommand, process_one_file
from ccoa.utils.io.parquet import TOKENIZE_SCHEMA


class _FakeTokenizer:
    """Fast-tokenizer stand-in: token IDs are deterministic per word."""

    is_fast = True

    def __call__(self, texts, **_):  # noqa: D401 - mirrors HF tokenizer signature
        return {"input_ids": [[hash(w) % 50_000 for w in t.split()] for t in texts]}


def _write_cache(path, entries):
    """Write a gzipped-JSONL extraction cache at `path` from `{index: text}`."""
    with gzip.open(path, "wb") as gz:
        for idx in sorted(entries):
            gz.write((json.dumps({"index": idx, "text": entries[idx]}) + "\n").encode("utf-8"))


def _default_args(**overrides):
    """Build an argparse.Namespace matching TokenizeCommand defaults."""
    base = {
        "cache_paths": [],
        "tokenizer": "fake-tokenizer",
        "records_limit": 0,
        "records_per_file_limit": 0,
        "files_limit": 0,
        "shuffle_files": False,
        "seed": 42,
        "workers": 1,
        "workers_mode": "thread",
        "max_pool_restarts": 10,
        "batch_size": 64,
        "progress_every": 0,
        "output": "",
        "overwrite": False,
        "anonymous_s3": False,
        "s3_requester_pays": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _patch_tokenizer(monkeypatch):
    monkeypatch.setattr(tok_mod, "load_tokenizer", lambda repo: _FakeTokenizer())


def test_process_one_file_emits_one_row_per_non_empty(tmp_path):
    """process_one_file reads cache, batches, skips empties, returns ordered rows."""
    cache = tmp_path / "foo.warc.gz.jsonl.gz"
    _write_cache(
        cache,
        {0: "alpha beta", 1: "   ", 2: "gamma delta epsilon", 3: ""},
    )
    args = _default_args(batch_size=2)
    result = process_one_file(str(cache), _FakeTokenizer(), args)

    assert result.processed == 2
    assert result.skipped_empty == 2
    assert [r[1] for r in result.rows] == [0, 2]
    assert [r[2] for r in result.rows] == [2, 3]
    assert all(
        isinstance(ids, list) and len(ids) == n
        for *_, n, ids in [(r[0], r[1], r[2], r[3]) for r in result.rows]
    )
    assert all(r[0] == str(cache) for r in result.rows)


def test_process_one_file_respects_per_file_limit(tmp_path):
    """records_per_file_limit caps the records taken from one cache file."""
    cache = tmp_path / "foo.warc.gz.jsonl.gz"
    _write_cache(cache, {i: f"word{i} more" for i in range(10)})
    args = _default_args(records_per_file_limit=3)
    result = process_one_file(str(cache), _FakeTokenizer(), args)
    assert result.processed == 3
    assert [r[1] for r in result.rows] == [0, 1, 2]


def test_progress_every_emits_heartbeat(tmp_path, caplog):
    """--progress-every N logs once every Nth record."""
    cache = tmp_path / "foo.warc.gz.jsonl.gz"
    _write_cache(cache, {i: "x y z" for i in range(5)})
    args = _default_args(progress_every=2, batch_size=1)
    caplog.set_level(logging.INFO, logger="ccoa.commands.tokenize")
    process_one_file(str(cache), _FakeTokenizer(), args)
    progress_lines = [r.getMessage() for r in caplog.records if "processed=" in r.getMessage()]
    assert any("processed=2" in m for m in progress_lines)
    assert any("processed=4" in m for m in progress_lines)


def test_cli_rejects_records_limit_with_workers(tmp_path, monkeypatch, caplog):
    """--records-limit + --workers > 1 errors fast with exit 2."""
    _patch_tokenizer(monkeypatch)
    args = _default_args(
        cache_paths=[str(tmp_path / "no-match-*.jsonl.gz")],
        output=str(tmp_path / "out.parquet"),
        records_limit=10,
        workers=2,
    )
    caplog.set_level(logging.ERROR, logger="ccoa.commands.tokenize")
    rc = TokenizeCommand().run(args)
    assert rc == 2
    assert any("--records-limit" in r.getMessage() for r in caplog.records)


def test_cli_writes_parquet_with_expected_schema(tmp_path, monkeypatch):
    """The end-to-end CLI run produces a parquet matching TOKENIZE_SCHEMA."""
    _patch_tokenizer(monkeypatch)
    cache = tmp_path / "alpha.warc.gz.jsonl.gz"
    _write_cache(cache, {0: "one two three", 1: "four", 2: "five six"})
    out = tmp_path / "tokens.parquet"

    args = _default_args(cache_paths=[str(cache)], output=str(out))
    rc = TokenizeCommand().run(args)
    assert rc == 0
    assert out.exists()

    table = pq.read_table(str(out))
    assert table.schema.equals(TOKENIZE_SCHEMA)
    assert table.num_rows == 3
    rows = table.to_pylist()
    assert [r["record_index"] for r in rows] == [0, 1, 2]
    assert [r["n_tokens"] for r in rows] == [3, 1, 2]
    assert all(r["cache_path"] == str(cache) for r in rows)
    assert all(len(r["token_ids"]) == r["n_tokens"] for r in rows)


def test_cli_overwrite_protects_then_replaces(tmp_path, monkeypatch, caplog):
    """Existing output errors without --overwrite; succeeds with it."""
    _patch_tokenizer(monkeypatch)
    cache = tmp_path / "alpha.warc.gz.jsonl.gz"
    _write_cache(cache, {0: "hello world"})
    out = tmp_path / "tokens.parquet"

    args = _default_args(cache_paths=[str(cache)], output=str(out))
    assert TokenizeCommand().run(args) == 0

    caplog.set_level(logging.ERROR, logger="ccoa.commands.tokenize")
    rc = TokenizeCommand().run(args)
    assert rc == 2
    assert any("already exists" in r.getMessage() for r in caplog.records)

    args.overwrite = True
    assert TokenizeCommand().run(args) == 0


def test_cli_writes_summary_sidecar(tmp_path, monkeypatch):
    """Sidecar CSV captures args, counters, and tokens.* distribution rows."""
    _patch_tokenizer(monkeypatch)
    cache = tmp_path / "alpha.warc.gz.jsonl.gz"
    _write_cache(cache, {0: "alpha", 1: "alpha beta", 2: "alpha beta gamma"})
    out = tmp_path / "tokens.parquet"
    summary = tmp_path / "tokens.summary.csv"

    args = _default_args(cache_paths=[str(cache)], output=str(out))
    assert TokenizeCommand().run(args) == 0
    assert summary.exists()

    rows = list(csv.reader(summary.open()))
    keys = {r[0] for r in rows[1:]}
    assert "run.cli" in keys
    assert "arg.tokenizer" in keys
    assert "count.processed" in keys
    assert "count.total_tokens" in keys
    assert "tokens.count" in keys
    assert "tokens.mean" in keys
    assert "time.total_seconds" in keys

    by_key = {r[0]: r[1] for r in rows[1:]}
    assert by_key["arg.tokenizer"] == "fake-tokenizer"
    assert by_key["count.processed"] == "3"
    assert by_key["count.total_tokens"] == "6"  # 1 + 2 + 3
    assert by_key["tokens.count"] == "3"


def test_load_tokenizer_rejects_slow(monkeypatch):
    """load_tokenizer raises when AutoTokenizer returns a slow tokenizer."""
    from ccoa.tokenizer import hf as hf_mod

    class _SlowTok:
        is_fast = False

    class _FakeAuto:
        @staticmethod
        def from_pretrained(*_, **__):
            return _SlowTok()

    import types

    fake = types.SimpleNamespace(AutoTokenizer=_FakeAuto)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake)
    with pytest.raises(RuntimeError, match="slow Python variant"):
        hf_mod.load_tokenizer("dummy/repo")


@pytest.mark.real_model
def test_tokenize_end_to_end_gpt2(tmp_path):
    """End-to-end with the public `gpt2` tokenizer. Skipped without --run-real."""
    cache = tmp_path / "real.warc.gz.jsonl.gz"
    _write_cache(
        cache,
        {0: "Hello, world!", 1: "The quick brown fox jumps over the lazy dog."},
    )
    out = tmp_path / "real-tokens.parquet"
    args = _default_args(
        cache_paths=[str(cache)],
        output=str(out),
        tokenizer="gpt2",
    )
    rc = TokenizeCommand().run(args)
    assert rc == 0
    table = pq.read_table(str(out))
    assert table.num_rows == 2
    n_tokens = table.column("n_tokens").to_pylist()
    assert all(n > 0 for n in n_tokens)
    token_ids = table.column("token_ids").to_pylist()
    assert all(len(ids) == n for ids, n in zip(token_ids, n_tokens, strict=True))
