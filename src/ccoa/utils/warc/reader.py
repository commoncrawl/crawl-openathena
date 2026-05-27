"""Stream WARC files and yield response records."""

from __future__ import annotations

from collections.abc import Iterator

from warcio.archiveiterator import ArchiveIterator


def iter_response_records(stream) -> Iterator:
    """Yield `response` records from a WARC byte stream."""
    for record in ArchiveIterator(stream):
        if record.rec_type == "response":
            yield record
