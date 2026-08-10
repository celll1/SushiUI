"""Video/audio gallery hashing (`image_hash` for media rows, `GET
/images/by-hash/{hash}`).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/media_hash_test.py -v

Background: every media `create_db_image_record` call site used to hardcode
`image_hash=""`, so a video/audio row could never be the TARGET of a
`source_image_hash` link -- clicking a gallery item's "source" link, when the
source was a video, could never resolve (0 of ~151 video rows had a hash).
Fixed by hashing the saved MASTER file's bytes at each call site, and adding
a `GET /images/by-hash/{hash}` lookup for when the source row is not on the
currently loaded gallery page.

THREE things are proven below, each with a NEGATIVE CONTROL where sensible
(a test that would also pass against the pre-fix code is worth nothing):

1. Every media `create_db_image_record` call site passes a real (non-empty-
   literal) hash, read off the LIVE SOURCE of `backend/api/routes.py` --
   the same "read the source, not just import behaviour" style as
   `attention_type_validation_test.py`'s `RouteValidationTest`, because the
   defect here was a hardcoded literal at each of 8 call sites, not a
   function that could be exercised without running an actual video/audio
   generation (GPU work, out of scope for this suite).
2. The by-hash route exists on the live FastAPI router AND is declared in
   openapi.yaml (OpenAPI-first: a route that exists in code but not in the
   spec, or vice versa, is itself the defect class this project's
   CLAUDE.md calls out).
3. The video source-side and target-side hash DEFINITIONS agree: hashing an
   uploaded clip's bytes directly (`calculate_bytes_hash`, used for
   `source_image_hash`) and hashing that same clip after it has been written
   to disk as the saved master (`calculate_file_hash`, used for
   `image_hash`) produce the IDENTICAL digest for identical bytes -- the
   precondition for the by-hash lookup to ever be able to match a video's
   source to the row it came from.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

_ROUTES_PATH = os.path.join(_BACKEND, "api", "routes.py")


def _routes_source() -> str:
    with open(_ROUTES_PATH, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1. No media call site still passes the "" sentinel
# ---------------------------------------------------------------------------
class MediaCallSiteHashTest(unittest.TestCase):
    """The 8 media generation routes: txt2vid, txt2aud, aud2aud, outpaint_aud,
    img2vid, ref2vid, outpaint_vid, inpaint_vid. Each calls
    `create_db_image_record(..., image_hash=<something>, ...)` once for its
    own gallery row."""

    GENERATION_TYPES = [
        "txt2vid",
        "txt2aud",
        # aud2aud's generation_type is a runtime variable (_generation_type),
        # not a literal -- located via its distinguishing db_image kwarg below.
        "outpaint_aud",
        "img2vid",
        "ref2vid",
        "outpaint_vid",
        "inpaint_vid",
    ]

    def test_no_media_call_site_hardcodes_the_empty_sentinel(self):
        source = _routes_source()
        # The pre-fix defect, verbatim. If this string is anywhere in the
        # file, at least one call site regressed back to the sentinel.
        self.assertNotIn(
            'image_hash=""', source,
            "a create_db_image_record call site still hardcodes the empty "
            "image_hash sentinel that made a media row unresolvable",
        )

    def test_each_named_generation_type_passes_a_computed_hash(self):
        """For each generation_type literal, the `create_db_image_record(...)`
        call it belongs to must set image_hash to a computed value, not a
        literal empty string. Anchored on the exact
        `generation_type="<type>",\\n            image_hash=` adjacency every
        one of these 8 call sites uses (verified against the file's actual
        formatting), rather than a loose substring window -- `find()`'s FIRST
        hit on a bare `generation_type="txt2vid"` can land on an unrelated
        earlier use of the same literal (e.g. a
        `process_controlnet_configs(..., generation_type="txt2vid")` call)."""
        source = _routes_source()
        for gtype in self.GENERATION_TYPES:
            needle = f'generation_type="{gtype}",\n            image_hash='
            with self.subTest(generation_type=gtype):
                self.assertIn(
                    needle, source,
                    f"no create_db_image_record(...) call found with "
                    f"generation_type={gtype!r} immediately followed by image_hash=",
                )
                self.assertNotIn(f'{needle}"",', source)

    def test_aud2aud_call_site_passes_a_computed_hash(self):
        """aud2aud's generation_type is the runtime variable
        `_generation_type`, not a literal, so it needs its own check."""
        source = _routes_source()
        needle = "generation_type=_generation_type,\n            image_hash="
        self.assertIn(needle, source, "aud2aud call site not found")
        self.assertNotIn(f'{needle}"",', source)

    def test_negative_control_the_pattern_would_have_caught_the_old_code(self):
        """Sanity check on the detector itself: a hand-built snippet shaped
        exactly like the OLD (broken) call site's formatting must trip
        the `assertNotIn` this class's real checks rely on."""
        broken_snippet = (
            'generation_type="txt2vid",\n'
            '            image_hash="",\n'
        )
        needle = 'generation_type="txt2vid",\n            image_hash='
        self.assertIn(needle, broken_snippet)
        self.assertIn(f'{needle}"",', broken_snippet)


# ---------------------------------------------------------------------------
# 2. GET /images/by-hash/{hash} exists in the live router AND in openapi.yaml
# ---------------------------------------------------------------------------
class ByHashRouteTest(unittest.TestCase):
    def test_route_registered_on_the_live_router(self):
        from api.routes import router

        paths = {getattr(route, "path", None) for route in router.routes}
        self.assertIn("/images/by-hash/{hash}", paths)

    def test_route_uses_not_found_error_not_a_bare_404(self):
        """Distinct from GET /images/{image_id}, which raises a bare
        HTTPException -- this endpoint follows the project's
        NotFoundError/ErrorResponse convention (see CLAUDE.md's error
        handling standard)."""
        import inspect

        from api.routes import get_image_by_hash

        source = inspect.getsource(get_image_by_hash)
        self.assertIn("NotFoundError", source)

    def test_route_returns_oldest_match_and_a_match_count(self):
        import inspect

        from api.routes import get_image_by_hash

        source = inspect.getsource(get_image_by_hash)
        self.assertIn("created_at.asc()", source)
        self.assertIn("match_count", source)

    def test_declared_in_openapi_yaml(self):
        with open(os.path.join(_REPO, "openapi.yaml"), encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        self.assertIn("/images/by-hash/{hash}", spec["paths"])
        path_item = spec["paths"]["/images/by-hash/{hash}"]
        self.assertIn("get", path_item)
        responses = path_item["get"]["responses"]
        self.assertIn("200", responses)
        self.assertIn("404", responses)
        # 404 must use the shared ErrorResponse schema, not an inline object
        # (see CLAUDE.md's unified error handling standard).
        ref = responses["404"]["content"]["application/json"]["schema"]["$ref"]
        self.assertEqual(ref, "#/components/schemas/ErrorResponse")

    def test_openapi_yaml_has_no_duplicate_path_keys(self):
        """YAML last-key-wins silently masks a duplicate `paths` entry from a
        concurrent edit (see CLAUDE.md's openapi-parity-maintenance note) --
        catch it with a duplicate-key-aware loader rather than plain
        yaml.safe_load, which would just silently keep the last one."""

        class _DupCheckLoader(yaml.SafeLoader):
            pass

        def _construct_mapping(loader, node, deep=False):
            mapping = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in mapping:
                    raise AssertionError(f"Duplicate key: {key!r}")
                mapping[key] = loader.construct_object(value_node, deep=deep)
            return mapping

        _DupCheckLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
        )
        with open(os.path.join(_REPO, "openapi.yaml"), encoding="utf-8") as f:
            yaml.load(f, Loader=_DupCheckLoader)  # raises on any duplicate key


# ---------------------------------------------------------------------------
# 3. Video source-side / target-side hash definitions agree
# ---------------------------------------------------------------------------
class VideoHashRoundTripTest(unittest.TestCase):
    """The design requires source_image_hash (hashed from the uploaded
    bytes, in memory, via `calculate_bytes_hash`) and a later row's
    image_hash (hashed from that same content once saved to disk as the
    MASTER file, via `calculate_file_hash`) to be defined identically --
    otherwise an uploaded clip's source link could never resolve to the row
    that produced the file, even when the bytes are literally the same."""

    def test_saved_master_hash_matches_the_reupload_source_hash(self):
        """Pins the ACTUAL invariant the by-hash lookup depends on, not just
        that the two hash functions are both sha256 wrappers (the previous
        version of this test built its own `hashlib.sha256(payload)` and
        compared BOTH functions to that -- a tautology that would pass
        against any correct-looking implementation, including a broken one,
        since it never proves the functions are called on what routes.py
        actually calls them on).

        The real sequence, across two requests: request A saves a MASTER
        file and hashes it (`image_hash`, via `calculate_file_hash`) only
        AFTER that file exists on disk; request B later uploads bytes
        identical to that master and hashes them in memory, before any file
        exists (`source_image_hash`/`source_audio_hash`, via
        `calculate_bytes_hash`). For `GET /images/by-hash/{hash}` to ever
        resolve B's source back to A's row, those two hashes -- computed at
        two different moments, by two different functions -- must agree."""
        from utils.image_utils import calculate_bytes_hash, calculate_file_hash

        payload = os.urandom(4096) + b"not a real mp4, just representative bytes"

        # "source" side: hashed from the bytes BEFORE any file is written.
        source_side_hash = calculate_bytes_hash(payload)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "master.mp4")
            # Hashing before the write proves the "after that file exists"
            # ordering is load-bearing: calculate_file_hash's own documented
            # not-found contract is "" for a path that doesn't exist yet.
            self.assertEqual(calculate_file_hash(path), "")

            with open(path, "wb") as f:
                f.write(payload)

            # "target" side: hashed from the file AFTER it exists -- the
            # exact moment routes.py's `_hash_saved_media` is called at
            # (once save_video_with_metadata/save_audio_with_metadata has
            # already returned the saved filename).
            target_side_hash = calculate_file_hash(path)

        self.assertEqual(
            source_side_hash, target_side_hash,
            "a clip uploaded, then later re-served as a saved master with "
            "byte-identical content, must hash identically on both sides or "
            "GET /images/by-hash/{hash} can never resolve the re-upload "
            "back to the row it came from",
        )

    def test_hash_saved_media_helper_runs_off_the_event_loop(self):
        """F4: `calculate_file_hash` reads+digests a file that can be a
        multi-gigabyte FFV1 master; routes.py must not do that synchronously
        inside an `async def` route. Reverting the fix (inlining
        `calculate_file_hash(...)` at each call site again, with no
        executor dispatch) must fail this test."""
        import inspect

        from api.routes import _hash_saved_media

        source = inspect.getsource(_hash_saved_media)
        self.assertIn("run_in_executor", source)
        self.assertIn("calculate_file_hash", source)

    def test_no_media_call_site_hashes_synchronously_inline(self):
        """All 8 media-save call sites (F4) must route through the shared
        off-loop helper rather than each inlining its own
        `calculate_file_hash(...)` call -- "ONE implementation ... not
        eight"."""
        source = _routes_source()
        self.assertEqual(
            source.count("_media_hash = await _hash_saved_media("), 8,
            "expected exactly 8 media-save call sites (one per media "
            "generation route) to hash through _hash_saved_media",
        )
        self.assertNotIn("_media_hash = calculate_file_hash(", source)

    def test_negative_control_pixel_decode_hash_would_not_agree(self):
        """The OLD (broken) design hashed a PNG re-encode of the ffmpeg-
        DECODED frame 0 for the source side. Decoding then re-encoding
        arbitrary bytes as a PNG does not reproduce those bytes -- this just
        documents that the two hash functions below (file-bytes vs an
        image-library round trip) do NOT define the same identity, which is
        exactly why the fix does not use `calculate_image_hash` for video."""
        from PIL import Image
        import numpy as np

        from utils.image_utils import calculate_bytes_hash, calculate_image_hash

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[0, 0] = [1, 2, 3]
        raw_bytes = frame.tobytes()

        pixel_hash = calculate_image_hash(Image.fromarray(frame))
        raw_hash = calculate_bytes_hash(raw_bytes)
        self.assertNotEqual(
            pixel_hash, raw_hash,
            "a pixel/PNG hash and a raw-bytes hash of the same frame data "
            "must not collide -- they are deliberately different identities",
        )


if __name__ == "__main__":
    unittest.main()
