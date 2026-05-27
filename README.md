# crawl-openathena

This repo contains the project plan and issues related to steering
Common Crawl's crawl using quality signals. The crawl is currently
mostly steered using search-engine-style ranking.


This project is currently in a pilot phase.


## Install

```bash
uv sync
```

The base install supports `ccoa classify-warc`. The Jupyter notebooks in `notebooks` folder
need an extra:

```bash
uv sync --extra notebooks
```

## CLI

```bash
uv run ccoa --help
```

### Classify WARC

`ccoa classify-warc` streams WARC files from S3 (or any fsspec URL),
extracts plain text from each response record with trafilatura, and
applies a HuggingFace-hosted fasttext classifier. Per-record output is a
CSV `URL,prediction_score,warc_filename,warc_record_index`; a one-shot score-distribution summary is
logged at the end and written to a `<output>.summary.csv` file.

```bash
uv run ccoa classify-warc \
  --warc-paths 's3://commoncrawl/crawl-data/CC-MAIN-2025-51/segments/1764871645602.73/warc/*.warc.gz' \
  --shuffle-files --seed 42 --files-limit 8 \
  --records-per-file-limit 50 \
  --skip-homepages \
  --workers 4 \
  --output data/classified.csv
```

When a `--warc-paths` value contains glob characters (`*`, `?`, `[`) it
is expanded via fsspec; matches are de-duplicated and sorted, then
optionally shuffled with `--seed` and truncated to `--files-limit`.
Quote the glob pattern in the shell to prevent local expansion.
`--records-limit` caps the total response records across all selected
files; `--records-per-file-limit` caps the records taken from each
individual file. Both default to `0` (unlimited).
`--skip-homepages` drops site-root URLs (empty/root path, no query, no
fragment) before extraction — useful when the classifier is meant to
score actual content pages, not link hubs.
`--workers N` (default `1`) processes that many WARC files concurrently;
CSV row order stays deterministic regardless of worker count. Combining
`--workers > 1` with `--records-limit` is rejected (the global cap
can't be enforced deterministically across parallel files); use
`--records-per-file-limit` instead.

`--workers-mode` picks the parallelism strategy when `--workers > 1`:
`thread` (default) shares one loaded model behind a lock — cheap, but
calls trafilatura/lxml concurrently and has been observed to hit glibc
heap-corruption aborts (`corrupted size vs. prev_size`) on adversarial
HTML. `process` loads a separate model per worker process (~4 GB extra
RAM each) and fully isolates lxml + fasttext C state — pick this if
thread mode crashes mid-run.

When `--output` is a file path (not `-`) the command also writes a
sidecar **summary** to `<output>.summary.<ext>` (e.g. `foo.csv` →
`foo.summary.csv`). It is a two-column `key,value` CSV containing the
exact CLI args, resolved input count, record counters, score stats
(min/max/mean/median/percentiles), wall-clock + per-step timings, and
start/finish timestamps — enough to reproduce the run. To avoid
clobbering past results, the command fails fast with a non-zero exit
if either the output **or** the summary file already exists.

The Common Crawl bucket no longer permits anonymous reads. The command
uses the default AWS credential chain (env / `~/.aws/credentials` /
instance profile) — any valid IAM identity works; the bucket owner pays
for requests (it is *not* Requester Pays). Alternatively, use the public
HTTPS gateway URL
(`https://data.commoncrawl.org/...`) with no credentials.

The default classifier is
[`ibm-granite/GneissWeb.Sci_classifier`](https://huggingface.co/ibm-granite/GneissWeb.Sci_classifier)
(`__label__science`). The first run downloads the ~4 GB model into the
HuggingFace cache. Override with `--model-repo`, `--model-file`, and
`--target-label`.

`--output` accepts `-` for stdout, any local path, or any fsspec URL —
including `s3://bucket/key.csv`. S3 outputs use the same `--anonymous-s3`
/ `--s3-requester-pays` options as inputs.

### Text extraction cache

Trafilatura is by far the most expensive step in the pipeline. When the
same WARCs are reprocessed (different model, different label, parameter
sweeps, retries), pass `--cache-dir` to skip re-extraction:

```bash
uv run ccoa classify-warc \
  --warc-paths s3://commoncrawl/crawl-data/CC-MAIN-2025-51/segments/.../foo.warc.gz \
  --limit 100 \
  --cache-dir s3://my-bucket/ccoa-cache/ \
  --output data/classified.csv
```

One gzipped JSONL file is written per input WARC, keyed by the 0-based
ordinal of the response record. Empty extractions are cached too
(negative caching — avoids re-running trafilatura on junk HTML).
`--cache-dir` may be a local path or any fsspec URI; S3 cache dirs honor
the same `--anonymous-s3` / `--s3-requester-pays` flags as inputs and
outputs. Input URIs are mirrored under the cache dir by scheme — e.g.
`s3://commoncrawl/.../foo.warc.gz` becomes
`<cache-dir>/s3/commoncrawl/.../foo.warc.gz.jsonl.gz`, so a single cache
dir can safely hold caches for many sources.

### Resuming an interrupted run

If a run crashes or is killed partway through, pass the partial output to
`--resume-from-output` on the next invocation to skip records already
classified:

```bash
uv run ccoa classify-warc \
  --warc-paths 's3://commoncrawl/crawl-data/CC-MAIN-2025-51/segments/.../*.warc.gz' \
  --records-per-file-limit 1000 \
  --resume-from-output data/classified.csv \
  --output data/classified__resume-2.csv
```

The resume CSV must include the `warc_filename` and `warc_record_index`
columns (the current output schema always does). Records matching that
`(warc_filename, record_index)` pair are skipped on the new run; the
new `--output` contains only the missing rows. Concatenate the two
CSVs (drop the second header) to get a complete result.

With `--records-per-file-limit N` the limit is interpreted as the
**target total** per file (resumed + new). Files already at the target
are skipped without opening the input stream; for files below the
target, only `N − resumed` more records are processed. To process an
additional `M` records on top of a prior run, set the limit to
`prior_limit + M`.

`--resume-from-output` is also useful with `--workers-mode process`:
when a worker dies on adversarial HTML the pool drops the suspect file
and continues; a follow-up resume run will retry the dropped files.


## Development

```bash
make test       # pytest
make lint       # ruff check
make format        # ruff format
make check      # lint + format-check + test
```

## License

Apache 2.0
