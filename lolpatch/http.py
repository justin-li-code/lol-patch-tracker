"""Polite, disk-cached HTTP GET."""
import hashlib
import time
from pathlib import Path

import requests

from .config import RAW, REQUEST_DELAY, USER_AGENT

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last_request = 0.0


def _cache_path(url: str) -> Path:
    digest = hashlib.sha1(url.encode()).hexdigest()[:16]
    return RAW / f"{digest}.html"


def get(url: str, *, refresh: bool = False, retries: int = 6) -> str | None:
    """Return page text, or None on 404. Cached forever unless refresh=True."""
    global _last_request
    path = _cache_path(url)
    if path.exists() and not refresh:
        text = path.read_text(encoding="utf-8")
        return None if text == "__404__" else text

    for attempt in range(retries):
        wait = REQUEST_DELAY - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        try:
            resp = _session.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 404:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("__404__", encoding="utf-8")
            return None
        if resp.status_code in (403, 429, 500, 502, 503, 504):
            time.sleep(15 * (attempt + 1))
            continue
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text, encoding="utf-8")
        return resp.text
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")
