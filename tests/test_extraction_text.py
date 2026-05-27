from __future__ import annotations

import logging

from ccoa.extraction.text import TrafilaturaLogCounter, extract_text

# Minimal but syntactically plausible PDF byte stream — enough that lxml
# definitively rejects it as HTML. Trafilatura's behaviour we care about
# is "non-HTML payloads don't crash the pipeline".
_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
    b"xref\n0 4\n0000000000 65535 f \n"
    b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n200\n%%EOF\n"
)


def test_extract_text_pdf_returns_empty_string():
    """PDF payloads must not crash trafilatura; `extract_text` returns ""."""
    result = extract_text(_PDF_BYTES)
    assert result == ""


def test_extract_text_pdf_is_captured_by_log_counter():
    """`TrafilaturaLogCounter` catches the rejection records for a PDF payload.

    Proves the pipeline can tally these "noisy non-HTML" records without
    leaking them to stderr.
    """
    counter = TrafilaturaLogCounter()
    traf_logger = logging.getLogger("trafilatura")
    traf_logger.addHandler(counter)
    prev_propagate = traf_logger.propagate
    traf_logger.propagate = False
    try:
        result = extract_text(_PDF_BYTES)
    finally:
        traf_logger.removeHandler(counter)
        traf_logger.propagate = prev_propagate

    assert result == ""
    # Trafilatura emits both an ERROR ("empty HTML tree") and a WARNING
    # ("discarding data") on this input; we only require that *something*
    # was captured, since the exact split is a trafilatura implementation
    # detail that could change between versions.
    assert counter.warnings + counter.errors >= 1


def test_extract_text_html_returns_text():
    """Sanity-check the happy path so the PDF assertions aren't vacuous."""
    html = (
        b"<!DOCTYPE html><html><head><title>T</title></head><body>"
        b"<p>Hello world. This sentence exists to give trafilatura "
        b"enough content to bother extracting.</p>"
        b"</body></html>"
    )
    result = extract_text(html)
    assert "Hello world" in result
