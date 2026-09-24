"""No network and no real artist names/images."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_danbooru_candidates import select_artist_tags
from core.tagger.danbooru_client import DanbooruClient


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


if __name__ == "__main__":
    unittest.main()
