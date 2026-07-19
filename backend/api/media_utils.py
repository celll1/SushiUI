"""Detail-view media helpers for the gallery.

Implements Phase 2 of the gallery performance redesign
(scratchpad/gallery_perf_redesign.md, Options C + D):

- Sized WebP preview cache for the detail/lightbox view (lazy-generated on
  first request, disk-cached, single-flighted, size-bounded).
- HTTP Range-aware file serving for /outputs (starlette 0.35.1's FileResponse
  has no Range support -- verified against venv/Lib/site-packages/starlette/
  responses.py; Range landed in starlette 0.36). Used for video/audio
  seeking and for the sized preview responses.

Both are CPU-only (Pillow + stdlib), zero VRAM.
"""
import asyncio
import mimetypes
import os
import re
import time
from typing import Dict, Optional

from fastapi import HTTPException
from PIL import Image
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from config.settings import settings

# ---------------------------------------------------------------------------
# Sized preview cache (Option C)
# ---------------------------------------------------------------------------

# Long-side width is snapped to one of these buckets so the cache can't grow
# one entry per distinct viewport/zoom size.
PREVIEW_WIDTH_BUCKETS = (1024, 1600, 2048)
PREVIEW_DEFAULT_WIDTH = 1024
PREVIEW_QUALITY = 80

# Per (cache_path) single-flight locks so concurrent requests for the same
# (file, width) don't double decode/resize/encode. Dict mutation is guarded
# separately since it isn't atomic across `await` points.
_preview_locks: Dict[str, asyncio.Lock] = {}
_preview_locks_guard = asyncio.Lock()


def snap_preview_width(width: Optional[int]) -> int:
    """Snap a requested width to the nearest supported bucket."""
    if not width:
        return PREVIEW_DEFAULT_WIDTH
    return min(PREVIEW_WIDTH_BUCKETS, key=lambda bucket: abs(bucket - width))


def _preview_cache_path(filename: str, width: int) -> str:
    base_name = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(settings.previews_dir, f"{base_name}_w{width}.webp")


def _generate_preview_sync(source_path: str, cache_path: str, width: int) -> None:
    """Decode + resize + re-encode as WebP. Called via asyncio.to_thread so it
    never blocks the event loop."""
    os.makedirs(settings.previews_dir, exist_ok=True)

    image = Image.open(source_path)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    long_side = max(image.size)
    if long_side > width:
        scale = width / long_side
        new_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Write to a temp file then atomically replace, so a concurrent reader
    # never observes a partially-written cache entry.
    tmp_path = f"{cache_path}.tmp-{os.getpid()}-{int(time.time() * 1000)}"
    image.save(tmp_path, format="WEBP", quality=PREVIEW_QUALITY)
    os.replace(tmp_path, cache_path)

    # Exclude the file we just wrote from eviction consideration: its mtime/
    # atime can tie (or lose, on coarse-grained filesystem clocks) against
    # older entries, and it's about to be served -- evicting it here would
    # make the very request that triggered generation 404 on read.
    _enforce_preview_cache_budget(exclude_path=cache_path)


def _enforce_preview_cache_budget(exclude_path: Optional[str] = None) -> None:
    """LRU-by-atime eviction: keep the preview cache directory under
    settings.preview_cache_max_bytes. Only runs right after a cache-miss
    write (cache hits never call this), bounding how often the directory is
    scanned. `exclude_path` (the file just written, if any) is counted
    towards the total but never selected for eviction."""
    try:
        entries = []
        total = 0
        with os.scandir(settings.previews_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                stat_result = entry.stat()
                total += stat_result.st_size
                if exclude_path is not None and entry.path == exclude_path:
                    continue
                entries.append((stat_result.st_atime, stat_result.st_size, entry.path))
        if total <= settings.preview_cache_max_bytes:
            return
        entries.sort(key=lambda e: e[0])  # oldest access time first
        for _atime, size, path in entries:
            if total <= settings.preview_cache_max_bytes:
                break
            try:
                os.remove(path)
                total -= size
            except OSError:
                pass
    except FileNotFoundError:
        pass


async def get_or_create_preview(source_path: str, filename: str, width: Optional[int] = None) -> str:
    """Return the cached preview path for (source file, width bucket),
    generating it on cache miss. Cache hit: a single os.path.isfile() check.
    Cache miss: single-flighted per (filename, width) key via asyncio.Lock,
    with a double-checked re-read after acquiring the lock."""
    snapped = snap_preview_width(width)
    cache_path = _preview_cache_path(filename, snapped)
    if os.path.isfile(cache_path):
        return cache_path

    async with _preview_locks_guard:
        lock = _preview_locks.get(cache_path)
        if lock is None:
            lock = asyncio.Lock()
            _preview_locks[cache_path] = lock

    async with lock:
        if not os.path.isfile(cache_path):
            await asyncio.to_thread(_generate_preview_sync, source_path, cache_path, snapped)

    async with _preview_locks_guard:
        if _preview_locks.get(cache_path) is lock:
            del _preview_locks[cache_path]

    return cache_path


# ---------------------------------------------------------------------------
# HTTP Range-aware file serving (Option D)
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_READ_CHUNK = 256 * 1024  # 256 KiB per yielded chunk

# mimetypes' built-in table is missing some formats used by ACE-Step audio
# outputs on some platforms/Python builds; register explicitly so
# Content-Type is always correct (browsers use it to pick the media element
# decode path).
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("video/webm", ".webm")


def _guess_media_type(path: str) -> str:
    media_type, _ = mimetypes.guess_type(path)
    return media_type or "application/octet-stream"


def _iter_file_range(path: str, start: int, length: int):
    with open(path, "rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            chunk = f.read(min(_READ_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _iter_file_full(path: str):
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_READ_CHUNK)
            if not chunk:
                break
            yield chunk


async def range_file_response(
    request: Request,
    file_path: str,
    media_type: Optional[str] = None,
    cache_control: str = "public, max-age=86400",
) -> Response:
    """HTTP Range-aware file responder.

    starlette 0.35.1's FileResponse has zero Range handling (grep confirms
    no "range" reference in responses.py; Range support landed in starlette
    0.36, which this project does not pin). Without it, a <video>/<audio>
    element's Range request gets a 200 + the full body: the whole file
    downloads before playback starts and seeking re-downloads from byte 0.

    Behavior:
    - No Range header: 200, full body, Content-Length set.
    - Valid `Range: bytes=start-end` (or open-ended / suffix form): 206
      Partial Content, Content-Range + Content-Length for the requested
      slice.
    - Unsatisfiable/malformed Range: 416, Content-Range: bytes */<size>.
    - HEAD: headers only, no body (mirrors starlette's own FileResponse).

    `Content-Encoding: identity` is set on every response from this function
    so the app-wide GZipMiddleware (main.py; compresses ANY response body
    over its size threshold with no content-type check) skips it entirely --
    GZip-ing a 206 Partial Content body would desync Content-Range/
    Content-Length from the bytes actually sent and break seeking. See
    starlette/middleware/gzip.py: a response is passed through untouched
    once it already carries a Content-Encoding header.
    """
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    file_size = os.path.getsize(file_path)
    if media_type is None:
        media_type = _guess_media_type(file_path)
    method = request.method.upper()

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Encoding": "identity",
        "Cache-Control": cache_control,
    }

    range_header = request.headers.get("range")
    if range_header:
        match = _RANGE_RE.match(range_header.strip())
        if not match or file_size == 0:
            headers["Content-Range"] = f"bytes */{file_size}"
            return Response(status_code=416, headers=headers)

        start_str, end_str = match.groups()
        if start_str == "" and end_str == "":
            headers["Content-Range"] = f"bytes */{file_size}"
            return Response(status_code=416, headers=headers)
        if start_str == "":
            # Suffix range: last N bytes.
            suffix_len = int(end_str)
            start = max(0, file_size - suffix_len)
            end = file_size - 1
        else:
            start = int(start_str)
            end = int(end_str) if end_str != "" else file_size - 1
        end = min(end, file_size - 1)

        if start > end or start >= file_size:
            headers["Content-Range"] = f"bytes */{file_size}"
            return Response(status_code=416, headers=headers)

        length = end - start + 1
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        headers["Content-Length"] = str(length)

        if method == "HEAD":
            return Response(status_code=206, headers=headers, media_type=media_type)
        return StreamingResponse(
            _iter_file_range(file_path, start, length),
            status_code=206,
            media_type=media_type,
            headers=headers,
        )

    headers["Content-Length"] = str(file_size)
    if method == "HEAD":
        return Response(status_code=200, headers=headers, media_type=media_type)
    return StreamingResponse(
        _iter_file_full(file_path),
        status_code=200,
        media_type=media_type,
        headers=headers,
    )
