"""No network and no real artist names/images."""

from __future__ import annotations

import sys
import unittest
import io
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_danbooru_candidates import select_artist_tags
from core.style_encoder.style_danbooru_cache import materialize_style_danbooru
from core.tagger.danbooru_client import DanbooruApiError, DanbooruClient
from PIL import Image


class CandidateTests(unittest.TestCase):
    def test_count_threshold_category_dedup_and_tie_order(self):
        pages = [
            [{"name": "artist_b", "category": 1, "post_count": 10},
             {"name": "artist_a", "category": 1, "post_count": 10},
             {"name": "character_x", "category": 4, "post_count": 100}],
            [{"name": "artist_b", "category": 1, "post_count": 11},
             {"name": "artist_c", "category": 1, "post_count": 9},
             {"name": "artist_d", "category": 1, "post_count": 30, "is_deprecated": True}],
        ]
        self.assertEqual(select_artist_tags(pages, min_post_count=10, order="count_desc", seed=0, limit=10),
                         ("artist_b", "artist_a"))

    def test_random_is_seeded_and_independent_of_api_page_order(self):
        tags = [{"name": f"artist_{i}", "category": 1, "post_count": i + 10} for i in range(10)]
        first = select_artist_tags([tags[:5], tags[5:]], min_post_count=10, order="random", seed=7, limit=10)
        second = select_artist_tags([list(reversed(tags))], min_post_count=10, order="random", seed=7, limit=10)
        self.assertEqual(first, second)
        self.assertNotEqual(first, select_artist_tags([tags], min_post_count=10, order="random", seed=8, limit=10))

    def test_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            select_artist_tags([], min_post_count=0, order="random", seed=1, limit=10)
        with self.assertRaises(ValueError):
            select_artist_tags([], min_post_count=1, order="random", seed=1, limit=0)

    def test_cache_is_frozen_and_refuses_validation_overlap(self):
        class Resolver:
            def resolve(self, rows, _content):
                artists = frozenset(row["tag"] for row in rows if row["category"] == "Artist")
                return type("Labels", (), {"artists": artists})()

        class FakeClient:
            def __init__(self):
                self.calls = 0

            def fetch_artist_tags(self, _min_count, _page):
                self.calls += 1
                if self.calls == 1:
                    raise DanbooruApiError("temporary failure")
                return [{"name": "artist_remote", "category": 1, "post_count": 10}]

            def fetch_posts(self, _name, *, page, strict):
                self.calls += 1
                return [{"id": n, "tag_string_artist": "artist_remote", "tag_string_general": "1girl"}
                        for n in (1, 2)]

            def download_inmemory(self, post):
                self.calls += 1
                image = Image.new("RGB", (4, 4), (post["id"], 0, 0))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                return buffer.getvalue(), "png", []

        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "cache"
            client = FakeClient()
            kwargs = dict(client=client, resolver=Resolver(), excluded_artists=set(), excluded_hashes=set(),
                          min_post_count=10, order="count_desc", seed=0, max_artists=1, max_tag_pages=2,
                          max_posts_per_artist=2, max_post_pages_per_artist=1,
                          min_images_per_artist=2, split_seed=0, val_artist_ratio=0.0)
            with patch("core.style_encoder.style_danbooru_cache.time.sleep") as sleep:
                first = materialize_style_danbooru(cache, **kwargs)
            sleep.assert_called_once_with(10.0)
            self.assertEqual(len(first["items"]), 2)
            calls = client.calls
            self.assertEqual(materialize_style_danbooru(cache, **kwargs), first)
            self.assertEqual(client.calls, calls)
            with self.assertRaisesRegex(ValueError, "overlaps local validation"):
                materialize_style_danbooru(cache, **{**kwargs, "excluded_hashes": {first["items"][0]["sha256"]}})
            with self.assertRaisesRegex(ValueError, "settings"):
                materialize_style_danbooru(cache, **{**kwargs, "seed": 1})


class ApiIntervalTests(unittest.TestCase):
    def test_tags_then_posts_share_1_4_second_gate(self):
        client = DanbooruClient(api_interval=1.4)
        with DanbooruClient._global_lock:
            original = DanbooruClient._global_last_call
            DanbooruClient._global_last_call = -999.0
        try:
            class Clock:
                now = 0.0
                starts = []

                def monotonic(self):
                    return self.now

                def sleep(self, duration):
                    self.now += duration

                def get(self, *args, **kwargs):
                    self.starts.append(self.now)
                    return type("Response", (), {"status_code": 200, "json": lambda self: []})()

            clock = Clock()
            with patch("core.tagger.danbooru_client.time.monotonic", side_effect=clock.monotonic), \
                 patch("core.tagger.danbooru_client.time.sleep", side_effect=clock.sleep), \
                 patch("core.tagger.danbooru_client.requests.get", side_effect=clock.get):
                client.fetch_tags("2000-01-01", 10, 1)
                client.fetch_posts("artist_a")
            self.assertEqual(len(clock.starts), 2)
            self.assertGreaterEqual(clock.starts[1] - clock.starts[0], 1.4)
        finally:
            with DanbooruClient._global_lock:
                DanbooruClient._global_last_call = original

    def test_short_client_setting_cannot_bypass_global_floor(self):
        client = DanbooruClient(api_interval=0.0)
        with DanbooruClient._global_lock:
            original = DanbooruClient._global_last_call
            DanbooruClient._global_last_call = 10.0
        try:
            with patch("core.tagger.danbooru_client.time.monotonic", return_value=10.1), \
                 patch("core.tagger.danbooru_client.time.sleep") as sleep:
                client._wait_for_api_rate()
            sleep.assert_called_once()
            self.assertAlmostEqual(sleep.call_args.args[0], 1.3)
        finally:
            with DanbooruClient._global_lock:
                DanbooruClient._global_last_call = original

    def test_artist_query_has_no_date_filter_and_error_is_not_empty_page(self):
        client = DanbooruClient(api_interval=1.4)
        ok = type("Response", (), {"status_code": 200, "json": lambda self: []})()
        with patch.object(client, "_api_slot", return_value=nullcontext()), \
             patch("core.tagger.danbooru_client.requests.get", return_value=ok) as get:
            self.assertEqual(client.fetch_artist_tags(100, 2), [])
        url = get.call_args.args[0]
        self.assertIn("search[category]=1", url)
        self.assertIn("search[post_count]=100..", url)
        self.assertNotIn("created_at", url)
        error = type("Response", (), {"status_code": 503})()
        with patch.object(client, "_api_slot", return_value=nullcontext()), \
             patch("core.tagger.danbooru_client.requests.get", return_value=error), \
             patch("core.tagger.danbooru_client.time.sleep", return_value=None):
            with self.assertRaises(DanbooruApiError):
                client.fetch_artist_tags(100)

    def test_strict_post_error_is_not_empty_page(self):
        client = DanbooruClient(api_interval=1.4)
        error = type("Response", (), {"status_code": 503})()
        with patch.object(client, "_api_slot", return_value=nullcontext()), \
             patch("core.tagger.danbooru_client.requests.get", return_value=error), \
             patch("core.tagger.danbooru_client.time.sleep", return_value=None):
            with self.assertRaises(DanbooruApiError):
                client.fetch_posts("artist_a", strict=True)


if __name__ == "__main__":
    unittest.main()
