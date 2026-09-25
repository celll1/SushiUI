"""Immutable, run-local Danbooru source for style training."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable

from PIL import Image

from core.tagger.danbooru_client import DanbooruApiError, DanbooruClient

from .style_danbooru_candidates import select_artist_tags
from .style_dataset import artist_split_unit
from .style_labels import StyleLabelResolver

_CATEGORIES = {
    "tag_string_general": "General",
    "tag_string_artist": "Artist",
    "tag_string_copyright": "Copyright",
    "tag_string_character": "Character",
    "tag_string_meta": "Meta",
}
_VERSION = 1


def post_tag_data(post: dict) -> list[dict[str, str]]:
    return [{"tag": tag, "category": category}
            for field, category in _CATEGORIES.items()
            for tag in str(post.get(field) or "").split()]


def _read_cache(path: Path, options: dict, excluded_hashes: set[str]) -> dict:
    with (path / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("version") != _VERSION or manifest.get("options") != options:
        raise ValueError("style Danbooru cache does not match this run's settings")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("style Danbooru cache has no usable items")
    for item in items:
        image = path / "images" / item["file"]
        if image.resolve().parent != (path / "images").resolve() or not image.is_file():
            raise ValueError("style Danbooru cache image is missing or outside the cache")
        if hashlib.sha256(image.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("style Danbooru cache image hash changed")
        if item["sha256"] in excluded_hashes:
            raise ValueError("style Danbooru cache image now overlaps local validation data")
    return manifest


def materialize_style_danbooru(
    cache_dir: str | os.PathLike, *, client: DanbooruClient,
    resolver: StyleLabelResolver, excluded_artists: set[str], excluded_hashes: set[str],
    min_post_count: int, order: str, seed: int, max_artists: int,
    max_tag_pages: int, max_posts_per_artist: int, max_post_pages_per_artist: int,
    min_images_per_artist: int, split_seed: int, val_artist_ratio: float,
    stop_requested: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Fetch once before optimization, then verify and reuse the immutable cache on resume."""
    cache = Path(cache_dir)
    options = {"min_post_count": min_post_count, "order": order, "seed": seed,
               "max_artists": max_artists, "max_tag_pages": max_tag_pages,
               "max_posts_per_artist": max_posts_per_artist,
               "max_post_pages_per_artist": max_post_pages_per_artist,
               "min_images_per_artist": min_images_per_artist,
               "split_seed": split_seed, "val_artist_ratio": val_artist_ratio}
    repo = Path(__file__).resolve().parents[3]
    try:
        relative = cache.resolve().relative_to(repo)
    except ValueError:
        pass
    else:
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", str(relative)], cwd=repo,
                                 capture_output=True, check=False)
        if ignored.returncode != 0:
            raise ValueError("style Danbooru cache location inside the repository must be Git-ignored")
    if cache.exists():
        return _read_cache(cache, options, excluded_hashes)
    if max_tag_pages < 1 or max_post_pages_per_artist < 1 or max_posts_per_artist < min_images_per_artist:
        raise ValueError("style Danbooru page and per-artist limits are too small")
    cache.parent.mkdir(parents=True, exist_ok=True)
    stage = cache.with_name(f"{cache.name}.partial-{uuid.uuid4().hex}")
    stage_images = stage / "images"
    stage_images.mkdir(parents=True)
    try:
        def fetch_page(call):
            for attempt in range(4):
                if stop_requested():
                    raise InterruptedError("style Danbooru collection stopped")
                try:
                    return call()
                except DanbooruApiError:
                    if attempt == 3:
                        raise
                    time.sleep(min(10.0 * (2 ** attempt), 60.0))

        pages = []
        candidate_limit = max_artists * 10
        for page in range(1, max_tag_pages + 1):
            if stop_requested():
                raise InterruptedError("style Danbooru collection stopped")
            if progress is not None:
                progress(page - 1, max_tag_pages, f"Scanning Artist tag page {page}")
            tags = fetch_page(lambda: client.fetch_artist_tags(min_post_count, page))
            pages.append(tags)
            if order == "count_desc" and len(select_artist_tags(
                    pages, min_post_count=min_post_count, order=order, seed=seed,
                    limit=candidate_limit)) >= candidate_limit:
                break
            if len(tags) < 200:
                break
        else:
            raise ValueError("style Danbooru tag-page cap reached before the scan ended")
        candidates = select_artist_tags(pages, min_post_count=min_post_count, order=order, seed=seed,
                                        limit=candidate_limit)
        entries: list[dict] = []
        accepted_artists: set[str] = set()
        used_post_ids: set[int] = set()
        used_hashes: set[str] = set()
        for candidate_index, name in enumerate(candidates, 1):
            if stop_requested():
                raise InterruptedError("style Danbooru collection stopped")
            if progress is not None:
                progress(candidate_index - 1, len(candidates),
                         f"Collecting Artist candidate {candidate_index}/{len(candidates)}")
            labels = resolver.resolve([{"tag": name, "category": "Artist"}], None)
            if len(labels.artists) != 1:
                continue
            (canonical_artist,) = labels.artists
            if (canonical_artist in accepted_artists or canonical_artist in excluded_artists
                    or artist_split_unit(canonical_artist, split_seed) < val_artist_ratio):
                continue
            artist_entries = []
            for page in range(1, max_post_pages_per_artist + 1):
                posts = fetch_page(lambda: client.fetch_posts(name, page=page, strict=True))
                for post in posts:
                    if stop_requested():
                        raise InterruptedError("style Danbooru collection stopped")
                    if str(post.get("tag_string_artist") or "").split() != [name]:
                        continue
                    post_id = post.get("id")
                    if isinstance(post_id, bool) or not isinstance(post_id, int) or post_id <= 0 \
                            or post_id in used_post_ids:
                        continue
                    downloaded = client.download_inmemory(post)
                    if downloaded is None:
                        continue
                    image_bytes, ext, _ = downloaded
                    try:
                        with Image.open(io.BytesIO(image_bytes)) as image:
                            image.load()
                    except (OSError, ValueError, Image.DecompressionBombError):
                        continue
                    digest = hashlib.sha256(image_bytes).hexdigest()
                    if digest in used_hashes or digest in excluded_hashes:
                        continue
                    data = post_tag_data(post)
                    resolved = resolver.resolve(data, None)
                    if resolved.artists != frozenset((canonical_artist,)):
                        continue
                    filename = f"{post_id}.{ext}"
                    (stage_images / filename).write_bytes(image_bytes)
                    artist_entries.append({"post_id": post_id, "file": filename, "sha256": digest,
                                           "artist": canonical_artist, "tag_data": data})
                    used_post_ids.add(post_id)
                    used_hashes.add(digest)
                    if len(artist_entries) >= max_posts_per_artist:
                        break
                if len(artist_entries) >= max_posts_per_artist or len(posts) < 200:
                    break
            if len(artist_entries) >= min_images_per_artist:
                entries.extend(artist_entries)
                accepted_artists.add(canonical_artist)
                if len(accepted_artists) >= max_artists:
                    break
            else:
                for item in artist_entries:
                    (stage_images / item["file"]).unlink()
                    used_post_ids.remove(item["post_id"])
                    used_hashes.remove(item["sha256"])
        if not entries:
            raise ValueError("style Danbooru collection found no eligible artist with enough usable images")
        manifest = {"version": _VERSION, "options": options, "items": entries}
        with (stage / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, separators=(",", ":"))
        os.replace(stage, cache)
        return manifest
    except BaseException:
        shutil.rmtree(stage)
        raise
