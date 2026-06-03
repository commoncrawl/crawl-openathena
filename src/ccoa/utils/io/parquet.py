"""Streaming parquet writer over an fsspec-backed file handle.

Used by `ccoa tokenize` to write per-record token output to a local
path or `s3://...` URI in a single pass, without buffering the full
dataset in memory.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    pass

TOKENIZE_SCHEMA = pa.schema(
    [
        ("cache_path", pa.string()),
        ("record_index", pa.int32()),
        ("n_tokens", pa.int32()),
        ("token_ids", pa.list_(pa.int32())),
    ]
)


@contextlib.contextmanager
def open_parquet_writer(
    uri: str,
    schema: pa.Schema,
    storage_options: dict[str, object],
    *,
    compression: str = "zstd",
) -> Iterator[pq.ParquetWriter]:
    """Yield a `pyarrow.parquet.ParquetWriter` bound to `uri`.

    The underlying file handle is opened via `fsspec.open(..., 'wb')`, so
    local paths and any fsspec-supported URI (`s3://`, etc.) both work.
    Both the writer and the file handle are closed on exit.
    """
    with fsspec.open(uri, "wb", **storage_options) as raw:
        writer = pq.ParquetWriter(raw, schema, compression=compression)
        try:
            yield writer
        finally:
            writer.close()
