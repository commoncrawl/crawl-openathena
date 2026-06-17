# Overlap of URLs in focus crawl vs. main crawl archive

This document shows how to measure the URL overlap between the focus crawl and the whole main crawl archive.
The approach will utilize the URL index (parquet files) and AWS Athena.

References:

- https://commoncrawl.org/columnar-index
- https://github.com/commoncrawl/cc-index-table


## URL index

```
s3://commoncrawl-open-athena/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22/
```

### Setup

Tell duckdb to use AWS credentials (persistent across sessions):

```bash
INSTALL httpfs;
LOAD httpfs;

CREATE PERSISTENT SECRET aws_s3 (
    TYPE s3,
    PROVIDER credential_chain
);
```

### Look at the parquet schema

```bash
duckdb -c "DESCRIBE SELECT * FROM 's3://commoncrawl-open-athena/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22/index/table/cc-supplemental/warc/crawl=CC-SUPPLEMENTAL-2026-22/subset=warc/part-00000-8637f21e-a055-46d1-8233-990f59974248.c000.gz.parquet'"
```

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ s3://commoncrawl-open-athena/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22/index/table/cc-supplemental/warc/crawl=CC-SUPPLEMENTAL-2026-22/subset=warc/part-00000-8637f21e-a055-46d1-8233-990f59974248.c000.gz.parquet │
│                                                                                                                                                                                                                              │
│ url_surtkey                                                                         varchar                                                                              url                        varchar                  │
│ url_host_name                                                                       varchar                                                                              url_host_tld               varchar                  │
│ url_host_2nd_last_part                                                              varchar                                                                              url_host_3rd_last_part     varchar                  │
│ url_host_4th_last_part                                                              varchar                                                                              url_host_5th_last_part     varchar                  │
│ url_host_registry_suffix                                                            varchar                                                                              url_host_registered_domain varchar                  │
│ url_host_private_suffix                                                             varchar                                                                              url_host_private_domain    varchar                  │
│ url_host_name_reversed                                                              varchar                                                                              url_protocol               varchar                  │
│ url_port                                                                            integer                                                                              url_path                   varchar                  │
│ url_query                                                                           varchar                                                                              fetch_time                 timestamp with time zone │
│ fetch_status                                                                        smallint                                                                             fetch_redirect             varchar                  │
│ content_digest                                                                      varchar                                                                              content_mime_type          varchar                  │
│ content_mime_detected                                                               varchar                                                                              content_charset            varchar                  │
│ content_languages                                                                   varchar                                                                              content_truncated          varchar                  │
│ warc_filename                                                                       varchar                                                                              warc_record_offset         bigint                   │
│ warc_record_length                                                                  bigint                                                                               warc_segment               varchar                  │
│ crawl                                                                               varchar                                                                              subset                     varchar                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Compare against ccindex:

```bash
duckdb -c "DESCRIBE SELECT * FROM 's3://commoncrawl/cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-21/subset=warc/part-00000-0ddabc35-a36d-4326-aa24-6ee6775acdc0.c000.gz.parquet'"
```

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ s3://commoncrawl/cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-21/subset=warc/part-00000-0ddabc35-a36d-4326-aa24-6ee6775acdc0.c000.gz.parquet │
│                                                                                                                                                │
│ url_surtkey                                  varchar                                       url                        varchar                  │
│ url_host_name                                varchar                                       url_host_tld               varchar                  │
│ url_host_2nd_last_part                       varchar                                       url_host_3rd_last_part     varchar                  │
│ url_host_4th_last_part                       varchar                                       url_host_5th_last_part     varchar                  │
│ url_host_registry_suffix                     varchar                                       url_host_registered_domain varchar                  │
│ url_host_private_suffix                      varchar                                       url_host_private_domain    varchar                  │
│ url_host_name_reversed                       varchar                                       url_protocol               varchar                  │
│ url_port                                     integer                                       url_path                   varchar                  │
│ url_query                                    varchar                                       fetch_time                 timestamp with time zone │
│ fetch_status                                 smallint                                      fetch_redirect             varchar                  │
│ content_digest                               varchar                                       content_mime_type          varchar                  │
│ content_mime_detected                        varchar                                       content_charset            varchar                  │
│ content_languages                            varchar                                       content_truncated          varchar                  │
│ warc_filename                                varchar                                       warc_record_offset         integer                  │
│ warc_record_length                           integer                                       warc_segment               varchar                  │
│ crawl                                        varchar                                       subset                     varchar                  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Create Athena database

``bash
CREATE DATABASE ccoaindex;

USE ccoaindex;

CREATE EXTERNAL TABLE IF NOT EXISTS ccoaindex.ccoaindex (
      url_surtkey                   STRING,
      url                           STRING,
      url_host_name                 STRING,
      url_host_tld                  STRING,
      url_host_2nd_last_part        STRING,
      url_host_3rd_last_part        STRING,
      url_host_4th_last_part        STRING,
      url_host_5th_last_part        STRING,
      url_host_registry_suffix      STRING,
      url_host_registered_domain    STRING,
      url_host_private_suffix       STRING,
      url_host_private_domain       STRING,
      url_host_name_reversed        STRING,
      url_protocol                  STRING,
      url_port                      INT,
      url_path                      STRING,
      url_query                     STRING,
      fetch_time                    TIMESTAMP,
      fetch_status                  SMALLINT,
      fetch_redirect                STRING,
      content_digest                STRING,
      content_mime_type             STRING,
      content_mime_detected         STRING,
      content_charset               STRING,
      content_languages             STRING,
      content_truncated             STRING,
      warc_filename                 STRING,
      warc_record_offset            INT,
      warc_record_length            INT,
      warc_segment                  STRING)
    PARTITIONED BY (
      crawl                         STRING,
      subset                        STRING)
    STORED AS parquet
    LOCATION 's3://commoncrawl-open-athena/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22/index/table/cc-supplemental/warc/';
```

To make Athena recognise the data partitions on S3, and to update the table as new crawls are added, run the command:


```bash
MSCK REPAIR TABLE ccoaindex
```

```
Partitions not in metastore:	ccoaindex:crawl=CC-SUPPLEMENTAL-2026-22/subset=crawldiagnostics	ccoaindex:crawl=CC-SUPPLEMENTAL-2026-22/subset=robotstxt	ccoaindex:crawl=CC-SUPPLEMENTAL-2026-22/subset=warc
Repair: Added partition to metastore ccoaindex:crawl=CC-SUPPLEMENTAL-2026-22/subset=crawldiagnostics
Repair: Added partition to metastore ccoaindex:crawl=CC-SUPPLEMENTAL-2026-22/subset=robotstxt
Repair: Added partition to metastore ccoaindex:crawl=CC-SUPPLEMENTAL-2026-22/subset=warc
```


### Query the index

Make sure to select `ccoaindex` database in the Athena query editor.

```bash
SELECT COUNT(*) FROM ccoaindex.ccoaindex WHERE subset = 'warc';

# 48190281
```

Find overlapping URLs in both the focus crawl (ccoaindex) and the main crawl (ccindex):

1) Overlap between most recent main crawl: CC-MAIN-2026-21

```bash
  SELECT
    COUNT(*)                                                AS focus_urls,
    COUNT(CASE WHEN m.url_surtkey IS NOT NULL THEN 1 END)   AS focus_urls_also_in_main,
    ROUND(100.0 * COUNT(CASE WHEN m.url_surtkey IS NOT NULL THEN 1 END) / COUNT(*), 2) AS pct_focus_also_in_main
  FROM (
    SELECT DISTINCT url_surtkey
    FROM ccoaindex.ccoaindex
    WHERE crawl = 'CC-SUPPLEMENTAL-2026-22'
      AND subset = 'warc'
  ) f
  LEFT JOIN (
    SELECT DISTINCT url_surtkey
    FROM ccindex.ccindex
    WHERE crawl = 'CC-MAIN-2026-21'
      AND subset = 'warc'
  ) m
    ON f.url_surtkey = m.url_surtkey;

#	focus_urls	focus_urls_also_in_main
# 1	42464397	3191683
# 3191683 / 42464397 ≈ 7.52%
```

2) Overlap between full main crawl archive (all crawls), split by crawl

A single query over the whole archive hits Athena's 30-minute query timeout
(the global `DISTINCT url_surtkey` across 100+ crawls forces a huge shuffle).
Instead we split the work per crawl:

- Materialize the focus keys **once** (tiny, ~42M rows). We carry
  `url_host_name` and `url_host_registered_domain` so we can aggregate on them
  later — host is functionally determined by the surtkey, so the main side only
  ever needs to scan `url_surtkey` for the join.
- For each main crawl, semi-join against the small focus table and append the
  matched focus keys into a partitioned `matched_surt` table (one partition per
  main crawl). Each per-crawl query only scans that crawl's `url_surtkey`
  column (a few GB) and finishes in minutes.

**Step 1 — materialize the focus keys once:**

```sql
CREATE TABLE ccoaindex.focus_surt AS
SELECT DISTINCT url_surtkey, url_host_name, url_host_registered_domain
FROM ccoaindex.ccoaindex
WHERE crawl = 'CC-SUPPLEMENTAL-2026-22'
  AND subset = 'warc';
```

**Step 2 — create the (empty) partitioned results table once:**

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS ccoaindex.matched_surt (
  url_surtkey                STRING,
  url_host_name              STRING,
  url_host_registered_domain STRING
)
PARTITIONED BY (main_crawl STRING)
STORED AS PARQUET
LOCATION 's3://commoncrawl-dev/cc-focus-tools/athena/ccoaindex/matched_surt/';
```

**Step 3 — per-crawl insert (this is what the loop below runs for each crawl).**
Note `main_crawl` must be the **last** column for the partitioned INSERT:

```sql
INSERT INTO ccoaindex.matched_surt
SELECT DISTINCT
  f.url_surtkey,
  f.url_host_name,
  f.url_host_registered_domain,
  m.crawl AS main_crawl
FROM ccoaindex.focus_surt f
JOIN ccindex.ccindex m
  ON m.url_surtkey = f.url_surtkey
WHERE m.subset = 'warc'
  AND m.crawl = 'CC-MAIN-2026-21';   -- replaced per crawl by the loop
```

**Step 4 — iterate over the N most recent crawls (new → old), skipping any
already materialized.** The per-query submit/poll/fail logic lives in
[`scripts/run_athena.sh`](../scripts/run_athena.sh); the loop just calls it.
Run from a shell with the AWS CLI configured:

```bash
#!/usr/bin/env bash
set -euo pipefail

# --- config ---------------------------------------------------------------
N="${N:-10}"                                        # number of crawls to process
export ATHENA_DATABASE=ccoaindex
export ATHENA_WORKGROUP=primary                     # your Athena workgroup
export ATHENA_OUTPUT=s3://<your-results-bucket>/athena/   # Athena results location
# --------------------------------------------------------------------------

# Crawls already done = existing partitions of matched_surt (catalog-only, free).
done_crawls=$(aws glue get-partitions \
  --database-name "$ATHENA_DATABASE" --table-name matched_surt \
  --query 'Partitions[].Values[0]' --output text 2>/dev/null || true)

# Available main crawls, newest first, take N (zero-padded => lexical sort works).
available=$(aws s3 ls s3://commoncrawl/cc-index/table/cc-main/warc/ \
  | grep -oE 'CC-MAIN-[0-9]{4}-[0-9]{2}' \
  | sort -ru \
  | head -n "${N:-10}")

# `while read` (not `for crawl in $available`) so it splits per line in both
# bash and zsh — zsh does not word-split unquoted variables by default.
while IFS= read -r crawl; do
  [[ -z "$crawl" ]] && continue
  if grep -qw "$crawl" <<<"$done_crawls"; then
    echo "skip    $crawl (already in matched_surt)"
    continue
  fi
  echo "process $crawl ..."
  ./scripts/run_athena.sh "
    INSERT INTO ${ATHENA_DATABASE}.matched_surt
    SELECT DISTINCT
      f.url_surtkey,
      f.url_host_name,
      f.url_host_registered_domain,
      m.crawl AS main_crawl
    FROM ${ATHENA_DATABASE}.focus_surt f
    JOIN ccindex.ccindex m
      ON m.url_surtkey = f.url_surtkey
    WHERE m.subset = 'warc'
      AND m.crawl = '${crawl}';
  " && echo "  done   $crawl"
done <<< "$available"
```

The one-off setup queries (steps 1–3) and the read queries (step 5) can be run
the same way, e.g.:

```bash
export ATHENA_DATABASE=ccoaindex
export ATHENA_OUTPUT=s3://<your-results-bucket>/athena/
./scripts/run_athena.sh "$(cat <<'SQL'
CREATE TABLE ccoaindex.focus_surt AS
SELECT DISTINCT url_surtkey, url_host_name, url_host_registered_domain
FROM ccoaindex.ccoaindex
WHERE crawl = 'CC-SUPPLEMENTAL-2026-22' AND subset = 'warc';
SQL
)"
```

**Step 5 — read the results.** `matched_surt` may hold the same surtkey under
several `main_crawl` partitions, so always `COUNT(DISTINCT url_surtkey)`.

Overall overlap:

```sql
SELECT
  (SELECT COUNT(*) FROM ccoaindex.focus_surt)                       AS focus_urls,
  COUNT(DISTINCT url_surtkey)                                       AS focus_urls_also_in_main,
  ROUND(100.0 * COUNT(DISTINCT url_surtkey)
        / (SELECT COUNT(*) FROM ccoaindex.focus_surt), 2)           AS pct_focus_also_in_main
FROM ccoaindex.matched_surt;
```

Aggregated per host / registered domain (focus total vs. share also in main):

```sql
SELECT
  f.url_host_registered_domain,                          -- or f.url_host_name
  COUNT(DISTINCT f.url_surtkey)                            AS focus_urls,
  COUNT(DISTINCT m.url_surtkey)                            AS focus_urls_also_in_main,
  ROUND(100.0 * COUNT(DISTINCT m.url_surtkey)
        / COUNT(DISTINCT f.url_surtkey), 2)                AS pct_focus_also_in_main
FROM ccoaindex.focus_surt f
LEFT JOIN ccoaindex.matched_surt m
  ON m.url_surtkey = f.url_surtkey
GROUP BY f.url_host_registered_domain                     -- or f.url_host_name
ORDER BY focus_urls DESC;
```

Matches per main crawl (how many focus URLs were found in each crawl). Here a
URL is counted once per `main_crawl` it appears in, so the counts sum to more
than the distinct overall total — that is expected (a focus URL present in
several crawls is counted in each):

```sql
SELECT
  main_crawl,
  COUNT(DISTINCT url_surtkey)                                        AS focus_urls_in_crawl,
  ROUND(100.0 * COUNT(DISTINCT url_surtkey)
        / (SELECT COUNT(*) FROM ccoaindex.focus_surt), 2)            AS pct_of_focus
FROM ccoaindex.matched_surt
GROUP BY main_crawl
ORDER BY main_crawl ASC;
```

### Results

Total crawls 121 (CC-MAIN-2013-20 - CC-MAIN-2026-21)

- 182,123,820 URLs in crawldb
- 129,835,169 unfetched URLs (71.3% of crawldb)
- 52,288,651 other URLs (succesful fetches, not modified, gone, redirect, duplicate)
- 48,190,281 URLs in index (succesful fetches)
