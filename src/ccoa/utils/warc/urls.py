"""URL-level filters for WARC records."""

from __future__ import annotations

from urllib.parse import urlparse


def is_homepage_url(url: str) -> bool:
    """Return True if `url` parses to a site-root URL (no path, no query, no fragment)."""
    parsed = urlparse(url)
    return parsed.path in ("", "/") and not parsed.query and not parsed.fragment
