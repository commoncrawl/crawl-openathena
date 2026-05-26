"""Plain-text extraction from raw HTML using trafilatura."""

from __future__ import annotations

import logging

import trafilatura


def extract_text(
    html: bytes,
    favour_precision: bool = True,
    include_images: bool = False,
    deduplicate: bool = True,
    **kwargs,
) -> str:
    """Extract plain text from raw HTML bytes using trafilatura.

    Settings like FineWeb: https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/extractors/trafilatura.py
    """
    text = trafilatura.extract(
        html,
        favor_precision=favour_precision,
        include_comments=False,
        deduplicate=deduplicate,
        include_images=include_images,
        **kwargs,
    )
    return text or ""


class TrafilaturaLogCounter(logging.Handler):
    """Counts trafilatura log records by severity instead of letting them propagate.

    Trafilatura is chatty on real Common Crawl content (empty payloads,
    misdetected encodings, non-HTML mislabelled as HTML). Callers that
    already treat those outcomes as "skipped" want the per-page log lines
    suppressed. Attaching an instance of this handler to the
    `trafilatura` logger with `propagate=False` swallows the records and
    keeps tallies (`warnings`, `errors`) the caller can report.

    The `logging.Handler` parent class wraps `emit()` in `self.acquire()`
    / `self.release()`, so the counters are safe under thread pools.
    """

    def __init__(self) -> None:
        """Initialise both severity counters to zero."""
        super().__init__()
        self.warnings = 0
        self.errors = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Increment the matching counter; never re-emit the record."""
        if record.levelno >= logging.ERROR:
            self.errors += 1
        else:
            self.warnings += 1
