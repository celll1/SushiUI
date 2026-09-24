"""
Rate-limited Danbooru API client for in-memory image fetching.

Enforces:
  - At least 1.4 seconds after one JSON response before the next JSON request
  - Bandwidth cap (dl_speed_kbps KB/s) during image download
  - No disk writes — images returned as raw bytes

Usage:
    client = DanbooruClient(api_interval=1.4, dl_speed_kbps=500)
    posts = client.fetch_posts("hoshimachi_suisei", page=1, min_score=10)
    for post in posts:
        result = client.download_inmemory(post)
        if result:
            img_bytes, file_ext, tags = result
"""

from __future__ import annotations

import json
import time
import threading
from contextlib import contextmanager
from typing import List, Optional, Tuple
from urllib.parse import quote_plus

import requests


_USER_AGENT = "SushiUITaggerTraining/1.0"

_ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

_RATING_MAP = {
    "e": "explicit",
    "q": "questionable",
    "s": "sensitive",
    "g": "general",
}

_TAG_STRING_KEYS = [
    "tag_string_general",
    "tag_string_artist",
    "tag_string_copyright",
    "tag_string_character",
    "tag_string_meta",
]


class DanbooruApiError(RuntimeError):
    """A request failed; callers must not treat this as an exhausted page."""


class DanbooruClient:
    """Thread-safe, rate-limited Danbooru API client (anonymous access).

    The API rate limit is enforced globally across all instances via class-level
    lock and timestamp so that DanbooruSampleBuffer and DanbooruTagSurveyor (which
    each own a DanbooruClient) cannot interleave calls and violate the 1.4s interval.
    """

    _global_lock: threading.Lock = threading.Lock()
    _global_last_call: float = 0.0
    _minimum_api_interval: float = 1.4

    def __init__(self, api_interval: float = 1.4, dl_speed_kbps: int = 500) -> None:
        self._api_interval = api_interval
        self._dl_speed_bytes_per_sec = dl_speed_kbps * 1024


    @contextmanager
    def _api_slot(self):
        """Keep the gate locked through the HTTP call; the next call waits after its completion."""
        with DanbooruClient._global_lock:
            elapsed = time.monotonic() - DanbooruClient._global_last_call
            interval = max(self._api_interval, DanbooruClient._minimum_api_interval)
            if elapsed < interval:
                time.sleep(interval - elapsed)
            try:
                yield
            finally:
                DanbooruClient._global_last_call = time.monotonic()

    def _wait_for_api_rate(self) -> None:
        with self._api_slot():
            pass

    @staticmethod
    def _record_download_timeout() -> None:
        """Feed a timed-out/failed download to the speed monitor (a throttle
        commonly shows up as stalls/timeouts before a hard ban)."""
        try:
            from .download_speed_monitor import get_speed_monitor
            get_speed_monitor().record(0, 0.0, timed_out=True)
        except Exception:
            pass


    def fetch_posts(self, tags: str, page: int = 1, min_score: int = 0, *, strict: bool = False) -> List[dict]:
        """Fetch up to 200 posts from Danbooru matching ``tags`` on ``page``.

        Parameters
        ----------
        tags      : Space-joined Danbooru tag query (e.g. "hoshimachi_suisei solo").
                    Commas are treated as query separators by the caller; this method
                    receives a single query string.
        page      : 1-based page index.
        min_score : If > 0, appends ``score:>=N`` to the query automatically.

        Returns
        -------
        List of post dicts, or empty list on any error / rate-limit response.
        """
        tag_parts = [t.strip() for t in tags.split() if t.strip()]
        if min_score > 0:
            tag_parts.append(f"score:>={min_score}")

        encoded = "+".join(quote_plus(t) for t in tag_parts)
        url = (
            f"https://danbooru.donmai.us/posts.json"
            f"?tags={encoded}&limit=200&page={page}"
        )

        headers = {"User-Agent": _USER_AGENT}
        try:
            with self._api_slot():
                response = requests.get(url, headers=headers, timeout=30)
        except requests.exceptions.RequestException as exc:
            if strict:
                raise DanbooruApiError("posts request failed") from exc
            print(f"[DanbooruClient] fetch_posts network error: {exc}")
            return []

        if response.status_code in (429, 503):
            print(
                f"[DanbooruClient] Rate-limited (HTTP {response.status_code}). "
                "Waiting 10 s before continuing."
            )
            time.sleep(10.0)
            if strict:
                raise DanbooruApiError(f"posts request returned HTTP {response.status_code}")
            return []

        if response.status_code != 200:
            if strict:
                raise DanbooruApiError(f"posts request returned HTTP {response.status_code}")
            print(f"[DanbooruClient] fetch_posts HTTP {response.status_code} for {url!r}")
            return []

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            if strict:
                raise DanbooruApiError("posts response is not JSON") from exc
            print(f"[DanbooruClient] JSON decode error: {exc}")
            return []

        if isinstance(data, dict) and data.get("success") is False:
            if strict:
                raise DanbooruApiError("posts API rejected the request")
            print(f"[DanbooruClient] API error: {data.get('message')}")
            return []

        if not isinstance(data, list):
            if strict:
                raise DanbooruApiError("posts response is not a list")
            return []

        return data

    def fetch_artist_tags(self, min_count: int, page: int = 1) -> List[dict]:
        """Fetch one category-1 tag page without the new-tag survey's date filter."""
        if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
            raise ValueError("min_count must be a positive integer")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError("page must be a positive integer")
        url = (
            "https://danbooru.donmai.us/tags.json"
            f"?search[post_count]={min_count}..&search[category]=1"
            f"&search[order]=count&limit=200&page={page}"
        )
        try:
            with self._api_slot():
                response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
        except requests.exceptions.RequestException as exc:
            raise DanbooruApiError("artist tags request failed") from exc
        if response.status_code in (429, 503):
            time.sleep(10.0)
            raise DanbooruApiError(f"artist tags request returned HTTP {response.status_code}")
        if response.status_code != 200:
            raise DanbooruApiError(f"artist tags request returned HTTP {response.status_code}")
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise DanbooruApiError("artist tags response is not JSON") from exc
        if not isinstance(data, list):
            raise DanbooruApiError("artist tags response is not a list")
        return data

    def fetch_tags(
        self,
        created_after: str,
        min_count: int,
        category: int,
        page: int = 1,
    ) -> List[dict]:
        """Fetch tags created on or after ``created_after`` with post_count >= min_count.

        Parameters
        ----------
        created_after : ISO-8601 date string, e.g. ``"2026-03-01"``
        min_count     : minimum post_count threshold
        category      : Danbooru category code (0=General, 3=Copyright, 4=Character, …)
        page          : 1-based page index

        Returns
        -------
        List of tag dicts with keys ``name``, ``post_count``, ``created_at``,
        ``category``.  Empty list on any error or rate-limit response.
        """
        url = (
            f"https://danbooru.donmai.us/tags.json"
            f"?search[post_count]={min_count}.."
            f"&search[category]={category}"
            f"&search[created_at]={created_after}.."
            f"&search[order]=count"
            f"&limit=200&page={page}"
        )
        headers = {"User-Agent": _USER_AGENT}
        try:
            with self._api_slot():
                response = requests.get(url, headers=headers, timeout=30)
        except requests.exceptions.RequestException as exc:
            print(f"[DanbooruClient] fetch_tags network error: {exc}")
            return []

        if response.status_code in (429, 503):
            print(
                f"[DanbooruClient] fetch_tags rate-limited (HTTP {response.status_code}). "
                "Waiting 10 s."
            )
            time.sleep(10.0)
            return []

        if response.status_code != 200:
            print(f"[DanbooruClient] fetch_tags HTTP {response.status_code} for {url!r}")
            return []

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[DanbooruClient] fetch_tags JSON decode error: {exc}")
            return []

        if not isinstance(data, list):
            return []

        return data

    def fetch_tags_by_name(
        self,
        name_matches: str,
        min_count: int = 0,
        limit: int = 200,
        page: int = 1,
    ) -> List[dict]:
        """Resolve a tag name pattern (wildcards allowed) to concrete tags.

        Uses the Danbooru tags API ``search[name_matches]`` which accepts ``*``
        wildcards (e.g. ``"blue_*"``). Ordered by post_count descending so page 1
        already contains the most significant matches. Category is NOT filtered
        server-side (the caller filters by its eligible-category set), so a single
        request covers all categories for the pattern.

        Parameters
        ----------
        name_matches : tag name or wildcard pattern (underscore form, e.g. "blue_*")
        min_count    : minimum post_count threshold (server-side)
        limit        : max results per page (Danbooru caps at 1000; default 200)
        page         : 1-based page index

        Returns
        -------
        List of tag dicts with keys ``name``, ``post_count``, ``category``.
        Empty list on any error or rate-limit response.
        """
        import urllib.parse as _url
        _pat = _url.quote(name_matches, safe="*")
        url = (
            f"https://danbooru.donmai.us/tags.json"
            f"?search[name_matches]={_pat}"
            f"&search[post_count]={max(0, int(min_count))}.."
            f"&search[order]=count"
            f"&limit={int(limit)}&page={int(page)}"
        )
        headers = {"User-Agent": _USER_AGENT}
        try:
            with self._api_slot():
                response = requests.get(url, headers=headers, timeout=30)
        except requests.exceptions.RequestException as exc:
            print(f"[DanbooruClient] fetch_tags_by_name network error: {exc}")
            return []

        if response.status_code in (429, 503):
            print(
                f"[DanbooruClient] fetch_tags_by_name rate-limited (HTTP {response.status_code}). "
                "Waiting 10 s."
            )
            time.sleep(10.0)
            return []

        if response.status_code != 200:
            print(f"[DanbooruClient] fetch_tags_by_name HTTP {response.status_code} for {url!r}")
            return []

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[DanbooruClient] fetch_tags_by_name JSON decode error: {exc}")
            return []

        if not isinstance(data, list):
            return []

        return data

    def download_inmemory(
        self, post: dict
    ) -> Optional[Tuple[bytes, str, List[str]]]:
        """Download a Danbooru post image into memory without touching the disk.

        Parameters
        ----------
        post : A post dict returned by :meth:`fetch_posts`.

        Returns
        -------
        ``(img_bytes, file_ext, tags_list)`` on success, or ``None`` if the post
        should be skipped (missing URL, unsupported format, download error, etc.).
        """
        post_id = post.get("id")
        file_url = post.get("file_url")
        file_ext = (post.get("file_ext") or "").lower()

        if not file_url:
            return None

        if file_ext not in _ALLOWED_EXTENSIONS:
            return None

        headers = {"User-Agent": _USER_AGENT}
        response: Optional[requests.Response] = None
        try:
            response = requests.get(file_url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()

            # Bandwidth-limited streaming download. Track the throttle sleep
            # separately so the speed monitor sees the *actual* network speed
            # (bytes / non-sleeping time), not the artificially-capped rate.
            chunks: List[bytes] = []
            total_bytes = 0
            start_time = time.monotonic()
            sleep_total = 0.0

            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                chunks.append(chunk)
                total_bytes += len(chunk)

                if self._dl_speed_bytes_per_sec > 0:
                    elapsed = time.monotonic() - start_time
                    expected = total_bytes / self._dl_speed_bytes_per_sec
                    if expected > elapsed:
                        _s = expected - elapsed
                        time.sleep(_s)
                        sleep_total += _s

            img_bytes = b"".join(chunks)

            # Feed the actual network speed (excludes the throttle sleep above).
            try:
                from .download_speed_monitor import get_speed_monitor
                _net = (time.monotonic() - start_time) - sleep_total
                get_speed_monitor().record(total_bytes, _net)
            except Exception:
                pass

        except requests.exceptions.ChunkedEncodingError as exc:
            print(f"[DanbooruClient] Truncated response for post {post_id}: {exc}")
            self._record_download_timeout()
            return None
        except requests.exceptions.RequestException as exc:
            print(f"[DanbooruClient] Download error for post {post_id}: {exc}")
            self._record_download_timeout()
            return None
        except Exception as exc:
            print(f"[DanbooruClient] Unexpected error downloading post {post_id}: {exc}")
            return None
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

        # Extract tags from post metadata
        tags: List[str] = []

        rating_short = post.get("rating")
        if rating_short and rating_short in _RATING_MAP:
            tags.append(_RATING_MAP[rating_short])
        elif rating_short:
            tags.append(f"rating:{rating_short}")

        for key in _TAG_STRING_KEYS:
            val = post.get(key)
            if val:
                tags.extend(val.split())

        return img_bytes, file_ext, tags
