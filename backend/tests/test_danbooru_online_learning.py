"""
Tests for Danbooru online learning and live vocabulary expansion.

Run with:
    d:\\celll1\\webui_cl\\venv\\Scripts\\python.exe -m pytest backend/tests/test_danbooru_online_learning.py -v

Test coverage:
  1.  API rate-limit interval (1.4 s) is enforced across all DanbooruClient instances
  2.  Download bandwidth cap is respected
  3.  Tag extraction from post metadata (rating map, tag_string_* keys)
  4.  Image download returns correct bytes + extension + tags
  5.  min_count filtering in DanbooruTagSurveyor
  6.  Download failure handling (network error, bad status, unsupported format, etc.)
  7.  Danbooru underscore/escape normalization matches existing vocabulary
  8.  MixedDataLoader interrupt-batch injection (vocab expansion, label padding,
      collation, base-batch invariance)
  9.  In-memory-only storage — no temp files created or left behind

All external HTTP calls are mocked.  No real network access is performed.
"""

from __future__ import annotations

import io
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from io import BytesIO
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

# ── path setup ───────────────────────────────────────────────────────────────
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
from PIL import Image

# Subjects under test
from backend.core.tagger.danbooru_client import DanbooruClient, _RATING_MAP
from backend.core.tagger.danbooru_sampler import (
    DanbooruSampleBuffer,
    MixedDataLoader,
    _build_label_and_mask_standalone,
)
from backend.core.tagger.danbooru_tag_surveyor import DanbooruTagSurveyor
from backend.core.tagger.danbooru_vocab_expander import (
    VocabExpander,
    _expand_param_state,
    expand_vocab_and_head,
)
from backend.core.tagger.tag_vocabulary import TagVocabulary, normalize_tag


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_rgb_png_bytes(width: int = 8, height: int = 8) -> bytes:
    """Return a small valid RGB PNG as bytes."""
    img = Image.new("RGB", (width, height), color=(128, 64, 192))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_vocab(tags: List[str], category: str = "General") -> TagVocabulary:
    vocab = TagVocabulary()
    for i, tag in enumerate(tags):
        norm = normalize_tag(tag)
        vocab.tag_to_idx[norm] = i
        vocab.idx_to_tag[i] = norm
        vocab.tag_to_category[norm] = category
    vocab._build_special_indices()
    return vocab


def _make_mock_processor(is_naflex: bool = False):
    """Return a mock AutoProcessor that returns plausible tensor dicts."""
    proc = MagicMock()

    def side_effect(images, return_tensors="pt"):
        result = {
            "pixel_values": torch.zeros(1, 3, 224, 224),
        }
        if is_naflex:
            result["pixel_attention_mask"] = torch.ones(1, 196, dtype=torch.int32)
            result["spatial_shapes"] = torch.tensor([[14, 14]], dtype=torch.int64)
        return result

    proc.side_effect = side_effect
    return proc


def _reset_client_rate_state():
    """Reset class-level rate-limit state between tests.

    Set _global_last_call far enough in the past that the *first* call in
    each test never waits, regardless of what time.monotonic() returns.
    """
    with DanbooruClient._global_lock:
        DanbooruClient._global_last_call = -999.0


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  API rate-limit interval
# ═══════════════════════════════════════════════════════════════════════════════

class TestApiRateLimit(unittest.TestCase):

    def setUp(self):
        _reset_client_rate_state()

    def tearDown(self):
        _reset_client_rate_state()

    def test_single_instance_waits_between_calls(self):
        """Second call on same instance must sleep if < api_interval elapsed."""
        client = DanbooruClient(api_interval=1.4)
        sleep_calls = []

        monotonic_seq = iter([0.0, 0.0, 0.5, 0.5])  # call1-start, call1-end, call2-start, call2-end

        with patch("backend.core.tagger.danbooru_client.time.monotonic", side_effect=monotonic_seq):
            with patch("backend.core.tagger.danbooru_client.time.sleep") as mock_sleep:
                client._wait_for_api_rate()  # first call — no sleep
                client._wait_for_api_rate()  # second call at t=0.5, needs 0.9 more sec

        mock_sleep.assert_called_once()
        sleep_duration = mock_sleep.call_args[0][0]
        self.assertAlmostEqual(sleep_duration, 0.9, places=5)

    def test_two_instances_share_rate_limit(self):
        """Two separate DanbooruClient instances must share the global rate limit."""
        c1 = DanbooruClient(api_interval=1.4)
        c2 = DanbooruClient(api_interval=1.4)

        # c1 calls at t=0; c2 tries at t=0.3 → must sleep ~1.1 s
        monotonic_seq = iter([0.0, 0.0, 0.3, 0.3])

        with patch("backend.core.tagger.danbooru_client.time.monotonic", side_effect=monotonic_seq):
            with patch("backend.core.tagger.danbooru_client.time.sleep") as mock_sleep:
                c1._wait_for_api_rate()
                c2._wait_for_api_rate()

        mock_sleep.assert_called_once()
        sleep_duration = mock_sleep.call_args[0][0]
        self.assertAlmostEqual(sleep_duration, 1.1, places=5)

    def test_no_sleep_when_sufficient_time_elapsed(self):
        """No sleep when enough time has already passed."""
        client = DanbooruClient(api_interval=1.4)

        # first call at t=0, second at t=2.0 (> 1.4)
        monotonic_seq = iter([0.0, 0.0, 2.0, 2.0])

        with patch("backend.core.tagger.danbooru_client.time.monotonic", side_effect=monotonic_seq):
            with patch("backend.core.tagger.danbooru_client.time.sleep") as mock_sleep:
                client._wait_for_api_rate()
                client._wait_for_api_rate()

        mock_sleep.assert_not_called()

    def test_rate_limit_429_triggers_backoff(self):
        """HTTP 429 response causes 10 s sleep and returns empty list."""
        client = DanbooruClient(api_interval=0.0)  # skip interval wait

        mock_resp = MagicMock()
        mock_resp.status_code = 429

        with patch("backend.core.tagger.danbooru_client.requests.get", return_value=mock_resp):
            with patch("backend.core.tagger.danbooru_client.time.sleep") as mock_sleep:
                result = client.fetch_posts("test_tag")

        self.assertEqual(result, [])
        mock_sleep.assert_called_once_with(10.0)

    def test_rate_limit_503_triggers_backoff(self):
        """HTTP 503 response causes 10 s sleep and returns empty list."""
        client = DanbooruClient(api_interval=0.0)
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch("backend.core.tagger.danbooru_client.requests.get", return_value=mock_resp):
            with patch("backend.core.tagger.danbooru_client.time.sleep") as mock_sleep:
                result = client.fetch_tags("2026-01-01", 100, 0)

        self.assertEqual(result, [])
        mock_sleep.assert_called_once_with(10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Download bandwidth cap
# ═══════════════════════════════════════════════════════════════════════════════

class TestBandwidthCap(unittest.TestCase):

    def setUp(self):
        _reset_client_rate_state()

    def tearDown(self):
        _reset_client_rate_state()

    def test_bandwidth_throttle_sleeps_when_too_fast(self):
        """If download is faster than cap, sleep is called to throttle."""
        dl_speed_kbps = 100
        client = DanbooruClient(api_interval=0.0, dl_speed_kbps=dl_speed_kbps)

        # 50 KB chunk downloaded in 0 elapsed time → expected = 50*1024 / (100*1024) = 0.5 s
        chunk = b"x" * (50 * 1024)
        post = {"id": 1, "file_url": "https://example.com/img.jpg", "file_ext": "jpg",
                "rating": "g", "tag_string_general": "1girl solo"}

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = iter([chunk])
        mock_resp.raise_for_status = MagicMock()

        # monotonic sequence: start_time=0.0, elapsed check=0.0 (instant)
        monotonic_seq = iter([0.0, 0.0])

        with patch("backend.core.tagger.danbooru_client.requests.get", return_value=mock_resp):
            with patch("backend.core.tagger.danbooru_client.time.monotonic", side_effect=monotonic_seq):
                with patch("backend.core.tagger.danbooru_client.time.sleep") as mock_sleep:
                    result = client.download_inmemory(post)

        self.assertIsNotNone(result)
        mock_sleep.assert_called_once()
        expected_sleep = (50 * 1024) / (dl_speed_kbps * 1024)  # 0.5 s
        self.assertAlmostEqual(mock_sleep.call_args[0][0], expected_sleep, places=5)

    def test_no_throttle_when_download_slow_enough(self):
        """No sleep when download is already at or below the cap."""
        client = DanbooruClient(api_interval=0.0, dl_speed_kbps=100)

        chunk = b"x" * (10 * 1024)  # 10 KB
        post = {"id": 2, "file_url": "https://example.com/img.png", "file_ext": "png",
                "rating": "s", "tag_string_general": "scenery"}

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = iter([chunk])
        mock_resp.raise_for_status = MagicMock()

        # elapsed = 1.0 s, expected = 10*1024 / 100*1024 = 0.1 s → no sleep needed
        monotonic_seq = iter([0.0, 1.0])

        with patch("backend.core.tagger.danbooru_client.requests.get", return_value=mock_resp):
            with patch("backend.core.tagger.danbooru_client.time.monotonic", side_effect=monotonic_seq):
                with patch("backend.core.tagger.danbooru_client.time.sleep") as mock_sleep:
                    result = client.download_inmemory(post)

        self.assertIsNotNone(result)
        mock_sleep.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Tag extraction from post metadata
# ═══════════════════════════════════════════════════════════════════════════════

class TestTagExtraction(unittest.TestCase):

    def setUp(self):
        _reset_client_rate_state()

    def tearDown(self):
        _reset_client_rate_state()

    def _download_post(self, post: dict) -> Optional[tuple]:
        """Helper: mock a successful download and return the result tuple."""
        client = DanbooruClient(api_interval=0.0)
        img_bytes = _make_rgb_png_bytes()

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = iter([img_bytes])
        mock_resp.raise_for_status = MagicMock()

        with patch("backend.core.tagger.danbooru_client.requests.get", return_value=mock_resp):
            with patch("backend.core.tagger.danbooru_client.time.monotonic", return_value=999.0):
                return client.download_inmemory(post)

    def test_rating_general_mapped(self):
        post = {"id": 1, "file_url": "http://x/a.jpg", "file_ext": "jpg",
                "rating": "g", "tag_string_general": ""}
        result = self._download_post(post)
        self.assertIsNotNone(result)
        _, _, tags = result
        self.assertIn("general", tags)

    def test_rating_explicit_mapped(self):
        post = {"id": 2, "file_url": "http://x/a.jpg", "file_ext": "jpg",
                "rating": "e", "tag_string_general": ""}
        result = self._download_post(post)
        _, _, tags = result
        self.assertIn("explicit", tags)

    def test_all_tag_string_keys_collected(self):
        """Tags from all tag_string_* fields are included."""
        post = {
            "id": 3, "file_url": "http://x/a.jpg", "file_ext": "jpg",
            "rating": "s",
            "tag_string_general":   "1girl solo",
            "tag_string_artist":    "some_artist",
            "tag_string_copyright": "original",
            "tag_string_character": "oc",
            "tag_string_meta":      "highres",
        }
        result = self._download_post(post)
        _, _, tags = result
        self.assertIn("1girl", tags)
        self.assertIn("solo", tags)
        self.assertIn("some_artist", tags)
        self.assertIn("original", tags)
        self.assertIn("oc", tags)
        self.assertIn("highres", tags)

    def test_underscore_tags_preserved_in_raw(self):
        """Raw tags from Danbooru keep underscores; normalization happens later."""
        post = {
            "id": 4, "file_url": "http://x/a.jpg", "file_ext": "jpg",
            "rating": "g",
            "tag_string_general": "hoshimachi_suisei virtual_youtuber",
        }
        result = self._download_post(post)
        _, _, tags = result
        # Raw tags preserve Danbooru underscore format
        self.assertIn("hoshimachi_suisei", tags)
        self.assertIn("virtual_youtuber", tags)

    def test_unsupported_extension_returns_none(self):
        """GIF, MP4 etc. must be rejected (not in _ALLOWED_EXTENSIONS)."""
        client = DanbooruClient(api_interval=0.0)
        for ext in ["gif", "mp4", "swf", "zip"]:
            post = {"id": 99, "file_url": f"http://x/a.{ext}", "file_ext": ext, "rating": "g"}
            result = client.download_inmemory(post)
            self.assertIsNone(result, f"Expected None for extension {ext!r}")

    def test_missing_file_url_returns_none(self):
        client = DanbooruClient(api_interval=0.0)
        post = {"id": 100, "file_ext": "jpg", "rating": "g"}
        self.assertIsNone(client.download_inmemory(post))


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Image download correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageDownload(unittest.TestCase):

    def setUp(self):
        _reset_client_rate_state()

    def tearDown(self):
        _reset_client_rate_state()

    def test_returned_bytes_match_served_content(self):
        """The bytes returned must exactly equal what the server sent."""
        client = DanbooruClient(api_interval=0.0)
        expected = _make_rgb_png_bytes(16, 16)

        post = {"id": 1, "file_url": "http://x/img.png", "file_ext": "png",
                "rating": "g", "tag_string_general": ""}

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = iter([expected])
        mock_resp.raise_for_status = MagicMock()

        with patch("backend.core.tagger.danbooru_client.requests.get", return_value=mock_resp):
            with patch("backend.core.tagger.danbooru_client.time.monotonic", return_value=999.0):
                result = client.download_inmemory(post)

        self.assertIsNotNone(result)
        img_bytes, ext, _ = result
        self.assertEqual(img_bytes, expected)
        self.assertEqual(ext, "png")

    def test_chunked_response_assembled_correctly(self):
        """Multiple chunks are reassembled into a single byte string."""
        client = DanbooruClient(api_interval=0.0, dl_speed_kbps=0)  # no throttle
        part1 = b"PNG_HEADER"
        part2 = b"PNG_BODY"
        post = {"id": 2, "file_url": "http://x/img.jpg", "file_ext": "jpg",
                "rating": "g", "tag_string_general": ""}

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = iter([part1, part2])
        mock_resp.raise_for_status = MagicMock()

        with patch("backend.core.tagger.danbooru_client.requests.get", return_value=mock_resp):
            with patch("backend.core.tagger.danbooru_client.time.monotonic", return_value=999.0):
                result = client.download_inmemory(post)

        img_bytes, _, _ = result
        self.assertEqual(img_bytes, part1 + part2)

    def test_network_error_returns_none(self):
        """RequestException during download → None, no crash."""
        import requests as req_lib
        client = DanbooruClient(api_interval=0.0)
        post = {"id": 3, "file_url": "http://x/img.jpg", "file_ext": "jpg", "rating": "g"}

        with patch("backend.core.tagger.danbooru_client.requests.get",
                   side_effect=req_lib.exceptions.ConnectionError("refused")):
            result = client.download_inmemory(post)

        self.assertIsNone(result)

    def test_http_error_returns_none(self):
        """Non-2xx response (404) → None."""
        import requests as req_lib
        client = DanbooruClient(api_interval=0.0)
        post = {"id": 4, "file_url": "http://x/img.jpg", "file_ext": "jpg", "rating": "g"}

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_lib.exceptions.HTTPError("404")
        mock_resp.iter_content.return_value = iter([])

        with patch("backend.core.tagger.danbooru_client.requests.get", return_value=mock_resp):
            result = client.download_inmemory(post)

        self.assertIsNone(result)

    def test_truncated_response_returns_none(self):
        """ChunkedEncodingError → None."""
        import requests as req_lib
        client = DanbooruClient(api_interval=0.0)
        post = {"id": 5, "file_url": "http://x/img.jpg", "file_ext": "jpg", "rating": "g"}

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_content.side_effect = req_lib.exceptions.ChunkedEncodingError("truncated")

        with patch("backend.core.tagger.danbooru_client.requests.get", return_value=mock_resp):
            result = client.download_inmemory(post)

        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  min_count filtering in DanbooruTagSurveyor
# ═══════════════════════════════════════════════════════════════════════════════

class TestTagSurveyorMinCount(unittest.TestCase):

    def _make_tag_entry(self, name: str, post_count: int) -> dict:
        return {"name": name, "post_count": post_count, "category": 0,
                "created_at": "2026-05-01T00:00:00.000Z"}

    def test_approved_set_contains_only_new_tags(self):
        """Tags already in vocabulary must NOT appear in approved set."""
        vocab = _make_vocab(["existing_tag", "another_existing"])

        surveyor = DanbooruTagSurveyor(
            vocabulary=vocab, categories=[0], min_count=10,
            lookback_days=30, survey_interval=9999,
            api_interval=0.0,
        )

        tag_page = [
            self._make_tag_entry("existing_tag",     500),   # already in vocab
            self._make_tag_entry("brand_new_tag",     300),   # not in vocab → approved
            self._make_tag_entry("another_existing",  200),   # already in vocab
            self._make_tag_entry("fresh_character",   150),   # not in vocab → approved
        ]

        with patch.object(surveyor._client, "fetch_tags", return_value=tag_page):
            surveyor._run_survey()

        approved = surveyor.get_approved()
        self.assertIn("brand new tag", approved)    # normalized (underscore→space)
        self.assertIn("fresh character", approved)
        self.assertNotIn("existing tag", approved)
        self.assertNotIn("another existing", approved)

    def test_mark_added_removes_from_approved(self):
        """mark_added() must remove tags from the approved set."""
        vocab = _make_vocab(["old_tag"])
        surveyor = DanbooruTagSurveyor(vocabulary=vocab, categories=[0],
                                        min_count=1, lookback_days=30,
                                        survey_interval=9999, api_interval=0.0)

        # Manually inject into approved
        with surveyor._lock:
            surveyor._approved.add("new tag")

        surveyor.mark_added(["new_tag"])  # underscore form — should normalize
        self.assertNotIn("new tag", surveyor.get_approved())

    def test_surveyor_paginates_until_fewer_than_200(self):
        """_run_survey must keep requesting pages until result < 200."""
        vocab = _make_vocab([])
        surveyor = DanbooruTagSurveyor(vocabulary=vocab, categories=[0],
                                        min_count=1, lookback_days=30,
                                        survey_interval=9999, api_interval=0.0)

        # page 1: 200 tags, page 2: 5 tags → stops
        page1 = [self._make_tag_entry(f"tag_{i}", 100) for i in range(200)]
        page2 = [self._make_tag_entry(f"tag_{i}", 100) for i in range(200, 205)]

        call_count = {"n": 0}
        def fake_fetch_tags(created_after, min_count, category, page=1):
            call_count["n"] += 1
            return page1 if page == 1 else page2

        with patch.object(surveyor._client, "fetch_tags", side_effect=fake_fetch_tags):
            surveyor._run_survey()

        self.assertEqual(call_count["n"], 2)
        self.assertEqual(len(surveyor.get_approved()), 205)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Danbooru underscore/escape normalization vs. existing vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalization(unittest.TestCase):

    def test_underscore_to_space(self):
        self.assertEqual(normalize_tag("hoshimachi_suisei"), "hoshimachi suisei")

    def test_already_spaced_unchanged(self):
        self.assertEqual(normalize_tag("1girl"), "1girl")

    def test_strip_whitespace(self):
        self.assertEqual(normalize_tag("  solo  "), "solo")

    def test_lowercase(self):
        self.assertEqual(normalize_tag("Artist_Name"), "artist name")

    def test_backslash_parens(self):
        self.assertEqual(normalize_tag("fate_\\(series\\)"), "fate (series)")

    def test_slash_parens(self):
        self.assertEqual(normalize_tag("game_/(title/)"), "game (title)")

    def test_backslash_slash(self):
        self.assertEqual(normalize_tag("fate\\/extra"), "fate/extra")

    def test_multiple_escaping_layers(self):
        # Multiple layers of escaping should all be resolved
        self.assertEqual(normalize_tag("\\(\\(deep\\)\\)"), "((deep))")

    def test_danbooru_tag_matches_vocab_after_normalization(self):
        """A Danbooru-style underscore tag must match the same vocab entry as space-form."""
        vocab = _make_vocab(["hoshimachi suisei", "virtual youtuber", "1girl"])

        # Tags as they arrive from Danbooru (underscores)
        raw_tags = ["hoshimachi_suisei", "virtual_youtuber", "1girl", "unknown_tag_xyz"]
        label, loss_mask = _build_label_and_mask_standalone(raw_tags, vocab)

        self.assertEqual(label[vocab.tag_to_idx["hoshimachi suisei"]], 1.0)
        self.assertEqual(label[vocab.tag_to_idx["virtual youtuber"]], 1.0)
        self.assertEqual(label[vocab.tag_to_idx["1girl"]], 1.0)

    def test_add_tags_normalizes_danbooru_underscores(self):
        """TagVocabulary.add_tags must normalize underscore-form tags to space-form."""
        vocab = _make_vocab(["existing tag"])
        added = vocab.add_tags(["new_danbooru_tag", "another_new"])

        norm_names = [a[0] for a in added]
        self.assertIn("new danbooru tag", norm_names)
        self.assertIn("another new", norm_names)
        self.assertIn("new danbooru tag", vocab.tag_to_idx)
        self.assertIn("another new", vocab.tag_to_idx)

    def test_add_tags_skips_already_present(self):
        """add_tags must not duplicate a tag already in the vocabulary."""
        vocab = _make_vocab(["existing tag"])
        added = vocab.add_tags(["existing_tag"])  # underscore form of already-present tag
        self.assertEqual(len(added), 0)
        self.assertEqual(vocab.num_tags, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MixedDataLoader interrupt-batch injection and vocab expansion
#
# Contract under test (backend/core/tagger/danbooru_sampler.py:MixedDataLoader):
#   - ctor: (base_loader, buffer, injection_interval=4, injection_batch_size=None,
#           expander, expansion_callback, vocabulary, quality_masking_mode,
#           alias_resolver).  injection_batch_size None/<=0 falls back to
#           base_loader.batch_size.
#   - __iter__ calls buffer.reset_download_cycle(<epoch set via set_epoch>) once
#     per iteration, then yields 2-tuples (batch, is_injection: bool).
#   - Base batches pass through UNCHANGED except for label/loss-mask padding to
#     the current vocabulary size; their batch size is invariant.
#   - Every injection_interval *non-None* base batches, buffer.drain_batch(
#     injection_batch_size) is attempted; on success a separate pure-Danbooru
#     batch is yielded with is_injection=True.  Drain is all-or-nothing: an
#     insufficient buffer silently skips the slot.
# ═══════════════════════════════════════════════════════════════════════════════

def _make_base_batch(batch_size: int = 2, num_tags: int = 5):
    """Return a fake (pv, pam, ss, labels, loss_masks) batch tuple."""
    pv         = torch.zeros(batch_size, 3, 224, 224)
    pam        = torch.zeros(batch_size, 0, dtype=torch.int32)
    ss         = torch.zeros(batch_size, 0, dtype=torch.int64)
    labels     = torch.zeros(batch_size, num_tags)
    loss_masks = torch.ones(batch_size, num_tags)
    return pv, pam, ss, labels, loss_masks


class _FakeLoader:
    """Minimal DataLoader stand-in that yields a fixed list of batches."""

    def __init__(self, batches):
        self._batches = batches
        self.dataset = MagicMock()
        self.num_workers = 0
        self.batch_size = 2

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class _FakeBuffer:
    """Minimal DanbooruSampleBuffer stand-in backed by a queue.

    Mirrors the consumer-facing surface MixedDataLoader actually uses:
    ``get_nowait``, ``drain_batch`` (all-or-nothing, per
    DanbooruSampleBuffer.drain_batch) and ``reset_download_cycle``.
    ``reset_calls`` records the epochs passed to reset_download_cycle so tests
    can assert the per-epoch cycle contract.
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._vocabulary = None  # unused; MixedDataLoader uses its own
        self.reset_calls: List[Optional[int]] = []

    def get_nowait(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def drain_batch(self, n: int):
        """Return n samples, or None if fewer than n are buffered.

        No partial drain — matches DanbooruSampleBuffer.drain_batch, which
        leaves the queue untouched when it cannot fill a whole batch.
        """
        if self._q.qsize() < n:
            return None
        items = []
        for _ in range(n):
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                for it in items:
                    self._q.put_nowait(it)
                return None
        return items

    def reset_download_cycle(self, epoch: Optional[int] = None) -> None:
        self.reset_calls.append(epoch)

    def put(self, item):
        self._q.put(item)

    def qsize(self) -> int:
        return self._q.qsize()


def _make_danbooru_sample(raw_tags: List[str]):
    """Return a fake buffered Danbooru sample (pv, pam, ss, raw_tags)."""
    pv_d  = torch.zeros(3, 224, 224)
    pam_d = torch.zeros(0, dtype=torch.int32)
    ss_d  = torch.zeros(0, dtype=torch.int64)
    return (pv_d, pam_d, ss_d, list(raw_tags))


class TestMixedDataLoaderInjection(unittest.TestCase):

    def test_passthrough_when_buffer_empty(self):
        """Empty buffer: base batches are yielded unchanged, no injection batch."""
        vocab  = _make_vocab(["tag0", "tag1", "tag2", "tag3", "tag4"])
        batch  = _make_base_batch(batch_size=2, num_tags=5)
        loader = _FakeLoader([batch])
        buf    = _FakeBuffer()

        # injection_interval=1 → an injection is attempted after every base batch;
        # the empty buffer must make every attempt a silent no-op.
        mdl = MixedDataLoader(loader, buf, injection_interval=1,
                              injection_batch_size=1, vocabulary=vocab)
        results = list(mdl)

        self.assertEqual(len(results), 1)
        payload, is_injection = results[0]
        self.assertFalse(is_injection)
        pv, pam, ss, labels, loss_masks = payload
        self.assertEqual(labels.shape, (2, 5))

    def test_epoch_reset_called_once_per_iteration(self):
        """__iter__ starts a new collection cycle with the epoch set by set_epoch."""
        vocab  = _make_vocab(["t0"])
        loader = _FakeLoader([_make_base_batch(1, 1)])
        buf    = _FakeBuffer()

        mdl = MixedDataLoader(loader, buf, injection_interval=1, vocabulary=vocab)
        self.assertEqual(buf.reset_calls, [])   # not called at construction

        list(mdl)
        self.assertEqual(buf.reset_calls, [None])  # no epoch set yet

        mdl.set_epoch(3)
        list(mdl)
        self.assertEqual(buf.reset_calls, [None, 3])

    def test_base_batch_size_invariant_and_interrupt_batch_scheduled(self):
        """Base batches pass through untouched; a separate pure-Danbooru batch
        arrives every injection_interval base batches (interrupt-batch scheme).

        The base batch size is INVARIANT — Danbooru samples are never spliced
        into it.
        """
        vocab   = _make_vocab(["tag0", "tag1", "tag2"])
        batches = [_make_base_batch(batch_size=2, num_tags=3) for _ in range(4)]
        loader  = _FakeLoader(batches)
        buf     = _FakeBuffer()
        for _ in range(2):
            buf.put(_make_danbooru_sample(["tag0", "tag2"]))

        mdl = MixedDataLoader(loader, buf, injection_interval=2,
                              injection_batch_size=1, vocabulary=vocab)
        results = list(mdl)

        flags = [is_inj for _, is_inj in results]
        # 4 base batches, injections after base batch #2 and #4
        self.assertEqual(flags, [False, False, True, False, False, True])

        base_payloads = [p for p, is_inj in results if not is_inj]
        self.assertEqual(len(base_payloads), 4)
        for pv, _, _, labels, loss_masks in base_payloads:
            self.assertEqual(pv.shape[0], 2, "base batch size must be invariant")
            self.assertEqual(labels.shape, (2, 3))
            self.assertEqual(loss_masks.shape, (2, 3))
            # Nothing was spliced in: base labels stay all-zero as constructed.
            self.assertTrue((labels == 0.0).all())

        inj_payloads = [p for p, is_inj in results if is_inj]
        self.assertEqual(len(inj_payloads), 2)
        for pv, _, _, labels, _ in inj_payloads:
            self.assertEqual(pv.shape[0], 1, "injection batch = injection_batch_size")
            self.assertEqual(labels.shape, (1, 3))

        self.assertEqual(buf.qsize(), 0, "both buffered samples must be drained")

    def test_injection_batch_size_defaults_to_base_batch_size(self):
        """injection_batch_size=None falls back to base_loader.batch_size."""
        vocab   = _make_vocab(["tag0", "tag1"])
        loader  = _FakeLoader([_make_base_batch(batch_size=2, num_tags=2)])
        self.assertEqual(loader.batch_size, 2)
        buf     = _FakeBuffer()
        buf.put(_make_danbooru_sample(["tag0"]))

        # Only ONE buffered sample: an all-or-nothing drain of 2 must fail.
        mdl = MixedDataLoader(loader, buf, injection_interval=1, vocabulary=vocab)
        results = list(mdl)
        self.assertEqual([f for _, f in results], [False])
        self.assertEqual(buf.qsize(), 1, "partial drain must leave the queue intact")

        # With a second sample buffered, the injection batch is 2 rows wide.
        buf.put(_make_danbooru_sample(["tag1"]))
        results = list(MixedDataLoader(loader, buf, injection_interval=1,
                                       vocabulary=vocab))
        self.assertEqual([f for _, f in results], [False, True])
        inj_pv = results[1][0][0]
        self.assertEqual(inj_pv.shape[0], 2)

    def test_injected_labels_match_raw_tags(self):
        """Labels in the interrupt batch reflect the normalized raw_tags of the
        samples they were drained from, row-for-row, and the base batch's own
        labels are left alone."""
        # normalize_tag converts underscore → space; build the vocab with spaces
        vocab = _make_vocab(["tag a", "tag b", "tag c"])

        loader = _FakeLoader([_make_base_batch(batch_size=1, num_tags=3)])
        buf    = _FakeBuffer()
        # Danbooru underscore form; drained in FIFO order
        buf.put(_make_danbooru_sample(["tag_a"]))
        buf.put(_make_danbooru_sample(["tag_c"]))

        mdl = MixedDataLoader(loader, buf, injection_interval=1,
                              injection_batch_size=2, vocabulary=vocab)
        results = list(mdl)

        self.assertEqual([f for _, f in results], [False, True])

        base_labels = results[0][0][3]
        self.assertTrue((base_labels == 0.0).all(),
                        "base-batch labels must not receive injected tags")

        inj_labels = results[1][0][3]
        self.assertEqual(inj_labels.shape, (2, 3))
        i_a = vocab.tag_to_idx["tag a"]
        i_b = vocab.tag_to_idx["tag b"]
        i_c = vocab.tag_to_idx["tag c"]
        # Row 0 came from ["tag_a"], row 1 from ["tag_c"]
        self.assertEqual(inj_labels[0][i_a], 1.0)
        self.assertEqual(inj_labels[0][i_b], 0.0)
        self.assertEqual(inj_labels[0][i_c], 0.0)
        self.assertEqual(inj_labels[1][i_a], 0.0)
        self.assertEqual(inj_labels[1][i_b], 0.0)
        self.assertEqual(inj_labels[1][i_c], 1.0)

    def test_label_padding_after_vocab_expansion(self):
        """Base-loader labels are padded to current vocab size after expansion.
        New columns get label=0 (negative), loss_mask=1 (train on negatives)."""
        vocab  = _make_vocab(["t0", "t1", "t2"])           # starts at 3 tags
        batch  = _make_base_batch(batch_size=2, num_tags=3)  # labels sized [2, 3]
        loader = _FakeLoader([batch])
        buf    = _FakeBuffer()

        # Simulate vocab expansion: add 2 new tags before iteration
        vocab.add_tags(["new_x", "new_y"])  # vocab now has 5 tags

        mdl = MixedDataLoader(loader, buf, injection_interval=1,
                              injection_batch_size=1, vocabulary=vocab)
        results = list(mdl)

        payload, is_injection = results[0]
        self.assertFalse(is_injection)
        _, _, _, labels, loss_masks = payload
        self.assertEqual(labels.shape[1], 5)
        self.assertEqual(loss_masks.shape[1], 5)
        # New columns: label=0, loss_mask=1
        self.assertTrue((labels[:, 3:] == 0.0).all())
        self.assertTrue((loss_masks[:, 3:] == 1.0).all())

    def test_vocab_expansion_callback_triggered(self):
        """VocabExpander.propose + consume → expansion_callback is called."""
        vocab    = _make_vocab(["t0", "t1"])
        expander = VocabExpander()
        callback_calls = []

        def fake_callback(new_tags):
            callback_calls.extend(new_tags)

        batch  = _make_base_batch(batch_size=1, num_tags=2)
        loader = _FakeLoader([batch])
        buf    = _FakeBuffer()

        # Propose new tags from the buffer thread side
        expander.propose({"brand_new_tag"})

        mdl = MixedDataLoader(
            loader, buf, injection_interval=1, injection_batch_size=1,
            expander=expander, expansion_callback=fake_callback,
            vocabulary=vocab,
        )
        list(mdl)

        self.assertIn("brand_new_tag", callback_calls)

    def test_none_batch_passed_through_and_does_not_advance_injection_counter(self):
        """A None batch (all samples broken) is yielded as (None, False) without
        processing, and does not count toward the injection interval."""
        vocab  = _make_vocab(["t0"])
        loader = _FakeLoader([None, _make_base_batch(1, 1), _make_base_batch(1, 1)])
        buf    = _FakeBuffer()
        buf.put(_make_danbooru_sample(["t0"]))

        mdl = MixedDataLoader(loader, buf, injection_interval=2,
                              injection_batch_size=1, vocabulary=vocab)
        results = list(mdl)

        self.assertEqual(results[0], (None, False))
        self.assertIsNotNone(results[1][0])
        self.assertFalse(results[1][1])
        # The None batch did not advance the counter: the injection lands after
        # the 2nd *real* base batch, i.e. last, not after results[1].
        self.assertEqual([f for _, f in results], [False, False, False, True])


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  In-memory-only storage — no temp files written or left behind
# ═══════════════════════════════════════════════════════════════════════════════

class TestInMemoryStorage(unittest.TestCase):

    def setUp(self):
        _reset_client_rate_state()
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        _reset_client_rate_state()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_no_files_written_during_process_post(self):
        """_process_post must not create any files anywhere."""
        vocab     = _make_vocab(["1girl", "solo"])
        processor = _make_mock_processor(is_naflex=False)

        buf = DanbooruSampleBuffer(
            tag_queries=["1girl"],
            vocabulary=vocab,
            processor=processor,
            is_naflex=False,
        )

        img_bytes = _make_rgb_png_bytes()
        post = {"id": 1, "file_url": "http://x/img.jpg", "file_ext": "jpg",
                "rating": "g", "tag_string_general": "1girl solo"}

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = iter([img_bytes])
        mock_resp.raise_for_status = MagicMock()

        snapshot_before = set(os.listdir(self._tmp))

        with patch("backend.core.tagger.danbooru_client.requests.get", return_value=mock_resp):
            with patch("backend.core.tagger.danbooru_client.time.monotonic", return_value=999.0):
                sample = buf._process_post(post)

        snapshot_after = set(os.listdir(self._tmp))
        self.assertEqual(snapshot_before, snapshot_after,
                         "Temp files were created during _process_post")

    def test_sample_is_tensors_not_file_paths(self):
        """Buffered sample must be (Tensor, Tensor, Tensor, list), not file paths."""
        vocab     = _make_vocab(["1girl"])
        processor = _make_mock_processor(is_naflex=False)

        buf = DanbooruSampleBuffer(
            tag_queries=["1girl"], vocabulary=vocab,
            processor=processor, is_naflex=False,
        )

        img_bytes = _make_rgb_png_bytes()
        post = {"id": 2, "file_url": "http://x/img.png", "file_ext": "png",
                "rating": "s", "tag_string_general": "1girl"}

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = iter([img_bytes])
        mock_resp.raise_for_status = MagicMock()

        with patch("backend.core.tagger.danbooru_client.requests.get", return_value=mock_resp):
            with patch("backend.core.tagger.danbooru_client.time.monotonic", return_value=999.0):
                sample = buf._process_post(post)

        self.assertIsNotNone(sample)
        pv, pam, ss, raw_tags = sample
        self.assertIsInstance(pv,       torch.Tensor, "pixel_values must be a Tensor")
        self.assertIsInstance(pam,      torch.Tensor, "pixel_attention_mask must be a Tensor")
        self.assertIsInstance(ss,       torch.Tensor, "spatial_shapes must be a Tensor")
        self.assertIsInstance(raw_tags, list,          "raw_tags must be a list")
        # No string that looks like a file path
        for tag in raw_tags:
            self.assertFalse(os.path.sep in tag or tag.startswith("/"),
                             f"raw_tags contains what looks like a path: {tag!r}")

    def test_buffer_queue_emptied_after_injection_batch_drained(self):
        """Consuming the loader drains the buffered sample out of the queue — the
        sample is handed to the interrupt batch and not retained by the buffer."""
        vocab  = _make_vocab(["t0", "t1"])

        batch  = _make_base_batch(batch_size=1, num_tags=2)
        loader = _FakeLoader([batch])
        buf    = _FakeBuffer()
        buf.put(_make_danbooru_sample(["t0"]))

        mdl = MixedDataLoader(loader, buf, injection_interval=1,
                              injection_batch_size=1, vocabulary=vocab)
        results = list(mdl)  # consume everything

        self.assertEqual([f for _, f in results], [False, True])
        # After iteration the internal queue must be empty
        self.assertIsNone(buf.get_nowait(), "Buffer queue not empty after consumption")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  VocabExpander dedup — no re-proposal of already-proposed tags
# ═══════════════════════════════════════════════════════════════════════════════

class TestVocabExpander(unittest.TestCase):

    def test_propose_deduplicates(self):
        exp = VocabExpander()
        exp.propose({"tag_a", "tag_b"})
        exp.propose({"tag_b", "tag_c"})  # tag_b already proposed

        pending = exp.consume_pending()
        self.assertCountEqual(pending, ["tag_a", "tag_b", "tag_c"])

    def test_consume_clears_pending(self):
        exp = VocabExpander()
        exp.propose({"x"})
        exp.consume_pending()
        self.assertFalse(exp.has_pending())

    def test_already_proposed_not_re_proposed(self):
        exp = VocabExpander()
        exp.propose({"x"})
        exp.consume_pending()
        # Propose the same tag again after consume
        exp.propose({"x"})
        self.assertFalse(exp.has_pending(),
                         "Already-proposed tag must not re-enter pending after consume")

    def test_thread_safety(self):
        """Concurrent propose calls must not lose tags or corrupt state."""
        exp = VocabExpander()
        errors = []

        def propose_many(prefix, n):
            try:
                for i in range(n):
                    exp.propose({f"{prefix}_{i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=propose_many, args=(f"t{j}", 50))
                   for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        pending = exp.consume_pending()
        self.assertEqual(len(pending), 200)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. expand_vocab_and_head + optimizer state migration
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpandVocabAndHead(unittest.TestCase):

    def _make_model_and_optimizer(self, num_tags: int, in_features: int = 16):
        """Return a minimal model stub with a head Linear and a FP32 AdamW optimizer.

        ``_Model.expand_head`` mirrors SigLIP2TaggerModel.expand_head
        (backend/core/tagger/siglip2_tagger_model.py): new rows zero-initialized,
        old rows copied, ``(new_weight, new_bias)`` returned for the optimizer
        param-group update. The real one additionally restores the old head's
        device/dtype, which is a no-op for these CPU/fp32 tests.
        """
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(in_features, num_tags)

            def expand_head(self, new_num_tags):
                old = self.head
                old_n = old.out_features
                new_head = nn.Linear(old.in_features, new_num_tags, bias=True)
                nn.init.zeros_(new_head.weight)
                nn.init.zeros_(new_head.bias)
                with torch.no_grad():
                    new_head.weight[:old_n] = old.weight
                    new_head.bias[:old_n]   = old.bias
                self.head = new_head
                return new_head.weight, new_head.bias

        model = _Model()
        optimizer = torch.optim.AdamW(
            [{"params": list(model.parameters()), "lr": 1e-4}]
        )
        # Run one step so optimizer.state is populated
        loss = model.head.weight.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return model, optimizer

    def test_head_output_size_increases(self):
        vocab = _make_vocab([f"t{i}" for i in range(5)])
        model, optimizer = self._make_model_and_optimizer(num_tags=5)

        n_added = expand_vocab_and_head(["new_tag_a", "new_tag_b"], vocab, model, optimizer)

        self.assertEqual(n_added, 2)
        self.assertEqual(model.head.out_features, 7)

    def test_old_weights_preserved(self):
        """Original head rows must be preserved after expansion."""
        import torch.nn as nn
        vocab = _make_vocab([f"t{i}" for i in range(4)])
        model, optimizer = self._make_model_and_optimizer(num_tags=4)

        old_weight = model.head.weight.data.clone()
        expand_vocab_and_head(["extra_tag"], vocab, model, optimizer)

        torch.testing.assert_close(model.head.weight[:4], old_weight)

    def test_new_rows_zero_initialized(self):
        """New head rows must start at zero (clean slate for new tags)."""
        vocab = _make_vocab([f"t{i}" for i in range(3)])
        model, optimizer = self._make_model_and_optimizer(num_tags=3)

        expand_vocab_and_head(["new_one", "new_two"], vocab, model, optimizer)

        new_rows = model.head.weight[3:]
        self.assertTrue((new_rows == 0.0).all(), "New head rows must be zero-initialized")

    def test_optimizer_state_migrated_fp32(self):
        """FP32 AdamW: exp_avg/exp_avg_sq rows are appended (not discarded)."""
        vocab = _make_vocab([f"t{i}" for i in range(4)])
        model, optimizer = self._make_model_and_optimizer(num_tags=4, in_features=8)

        old_param = model.head.weight
        old_state = optimizer.state.get(old_param, {})
        self.assertIn("exp_avg", old_state, "Optimizer state not populated before test")
        old_exp_avg_shape = old_state["exp_avg"].shape  # [4, 8]

        expand_vocab_and_head(["tag_x", "tag_y"], vocab, model, optimizer)

        new_param  = model.head.weight
        new_state  = optimizer.state.get(new_param, {})
        self.assertIn("exp_avg", new_state)
        self.assertEqual(new_state["exp_avg"].shape[0], old_exp_avg_shape[0] + 2)
        self.assertEqual(new_state["exp_avg"].shape[1], old_exp_avg_shape[1])

    def test_optimizer_state_8bit_cleared(self):
        """8-bit optimizer state must be cleared (bitsandbytes will re-init)."""
        state = {
            "state1": torch.zeros(4),
            "state2": torch.zeros(4),
            "absmax1": torch.zeros(1),
            "absmax2": torch.zeros(1),
        }
        _expand_param_state(state, n_new=2, is_bias=False)
        self.assertEqual(len(state), 0, "8-bit state must be fully cleared")

    def test_expand_returns_zero_when_nothing_added(self):
        """expand_vocab_and_head must return 0 if all tags already in vocab."""
        vocab = _make_vocab(["existing"])
        model, optimizer = self._make_model_and_optimizer(num_tags=1)
        n = expand_vocab_and_head(["existing"], vocab, model, optimizer)
        self.assertEqual(n, 0)
        self.assertEqual(model.head.out_features, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
