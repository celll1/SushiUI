"""
Rate-limited Danbooru API client for in-memory image fetching.

Enforces:
  - Minimum api_interval seconds between /posts.json calls (Danbooru TOS)
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


class DanbooruClient:
    """Thread-safe, rate-limited Danbooru API client (anonymous access)."""

    def __init__(self, api_interval: float = 1.4, dl_speed_kbps: int = 500) -> None:
        self._api_interval = api_interval
        self._dl_speed_bytes_per_sec = dl_speed_kbps * 1024
        self._last_api_call: float = 0.0
        self._api_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_api_rate(self) -> None:
        """Block until at least api_interval seconds have passed since the last API call."""
        with self._api_lock:
            elapsed = time.monotonic() - self._last_api_call
            if elapsed < self._api_interval:
                time.sleep(self._api_interval - elapsed)
            self._last_api_call = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_posts(self, tags: str, page: int = 1, min_score: int = 0) -> List[dict]:
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
        self._wait_for_api_rate()

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
            response = requests.get(url, headers=headers, timeout=30)
        except requests.exceptions.RequestException as exc:
            print(f"[DanbooruClient] fetch_posts network error: {exc}")
            return []

        if response.status_code in (429, 503):
            print(
                f"[DanbooruClient] Rate-limited (HTTP {response.status_code}). "
                "Waiting 10 s before continuing."
            )
            time.sleep(10.0)
            return []

        if response.status_code != 200:
            print(f"[DanbooruClient] fetch_posts HTTP {response.status_code} for {url!r}")
            return []

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[DanbooruClient] JSON decode error: {exc}")
            return []

        if isinstance(data, dict) and data.get("success") is False:
            print(f"[DanbooruClient] API error: {data.get('message')}")
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

            # Bandwidth-limited streaming download
            chunks: List[bytes] = []
            total_bytes = 0
            start_time = time.monotonic()

            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                chunks.append(chunk)
                total_bytes += len(chunk)

                if self._dl_speed_bytes_per_sec > 0:
                    elapsed = time.monotonic() - start_time
                    expected = total_bytes / self._dl_speed_bytes_per_sec
                    if expected > elapsed:
                        time.sleep(expected - elapsed)

            img_bytes = b"".join(chunks)

        except requests.exceptions.ChunkedEncodingError as exc:
            print(f"[DanbooruClient] Truncated response for post {post_id}: {exc}")
            return None
        except requests.exceptions.RequestException as exc:
            print(f"[DanbooruClient] Download error for post {post_id}: {exc}")
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
