"""Every `generation_type` the backend writes to gallery.db must be
reachable through SOME combination of the gallery's frontend filter
checkboxes -- otherwise rows of that type exist in the DB but can never be
displayed (the historical `inpaint_vid` bug, later repeated for `ref2vid`
and `upscale`).

This is a static cross-file consistency check, not a browser test: no JS
test runner exists in this repo (see `frontend/package.json`), so it reads
routes.py (the live write path) as text and compares its literal
generation_type set against ImageGrid.tsx's pushed literals.
`backend/restore_database.py` is a standalone disaster-recovery script, not
scanned here; its own literals are txt2img/img2img/inpaint plus a fallback
pinned to txt2img (not scanned but manually verified: no type outside this
set).

LIMITATION: this is a set-membership check only. It confirms every backend
type is pushed by SOME checkbox, not that it is pushed by the RIGHT one --
e.g. an edit that pushed "upscale" under filterTxt2Img instead of a
dedicated filterUpscale would still pass this test.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/gallery_type_visibility_test.py -v
"""

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROUTES_PATH = os.path.join(REPO_ROOT, "backend", "api", "routes.py")
IMAGE_GRID_PATH = os.path.join(REPO_ROOT, "frontend", "src", "components", "viewer", "ImageGrid.tsx")


def _backend_generation_types(source: str) -> set:
    """Every literal ever assigned to `generation_type=` in routes.py.

    Covers both `generation_type="literal"` call sites and any
    `..._generation_type = "a" if ... else "b"` local first, whose result
    later feeds a `generation_type=` kwarg (e.g. the txt2aud/repaint switch).
    """
    types = set(re.findall(r'generation_type\s*=\s*"([a-zA-Z0-9_]+)"', source))
    for line in re.findall(r'_generation_type\s*=.*', source):
        types.update(re.findall(r'"([a-zA-Z0-9_]+)"', line))
    return types


def _frontend_filtered_types(source: str) -> set:
    """Every literal ever pushed into the gallery's `types` filter array."""
    types = set()
    for call in re.findall(r"types\.push\(([^)]*)\)", source):
        types.update(re.findall(r'"([a-zA-Z0-9_]+)"', call))
    return types


def _missing_types(backend_source: str, frontend_source: str) -> set:
    backend_types = _backend_generation_types(backend_source)
    frontend_types = _frontend_filtered_types(frontend_source)
    return backend_types - frontend_types


def test_every_backend_generation_type_is_reachable_via_a_gallery_filter():
    with open(ROUTES_PATH, "r", encoding="utf-8") as f:
        backend_source = f.read()
    with open(IMAGE_GRID_PATH, "r", encoding="utf-8") as f:
        frontend_source = f.read()

    missing = _missing_types(backend_source, frontend_source)
    assert not missing, (
        f"generation_type(s) {sorted(missing)} are written by routes.py but "
        f"never appear in any ImageGrid.tsx types.push(...) call -- rows of "
        f"this type exist in gallery.db but can never be filtered into view "
        f"regardless of checkbox state (the inpaint_vid/ref2vid/upscale bug class)."
    )


def test_the_backend_extractor_actually_finds_the_known_types():
    """Sanity check the regex itself isn't silently matching nothing."""
    with open(ROUTES_PATH, "r", encoding="utf-8") as f:
        backend_source = f.read()
    backend_types = _backend_generation_types(backend_source)
    for expected in ("txt2img", "img2img", "inpaint", "outpaint", "upscale",
                     "txt2vid", "img2vid", "ref2vid", "inpaint_vid", "outpaint_vid",
                     "outpaint_aud", "txt2aud", "aud2aud", "repaint"):
        assert expected in backend_types, f"extractor missed known type {expected!r}"


# --------------------------------------------------------------------------
# Negative control: the historical bug and its later repeats, reproduced as
# in-memory source strings (never written to disk) to prove the checker
# above is sensitive to exactly this regression class.
# --------------------------------------------------------------------------

def test_checker_catches_the_inpaint_vid_class_regression():
    backend_source = 'generation_type="inpaint_vid"'
    frontend_source = 'if (filterInpaint) types.push("inpaint");'  # inpaint_vid omitted
    assert _missing_types(backend_source, frontend_source) == {"inpaint_vid"}


def test_checker_catches_the_ref2vid_and_upscale_regression():
    """MUTANT: the pre-fix ImageGrid.tsx -- img2vid checkbox pushes only
    "img2vid" (no ref2vid), and there is no upscale filter at all. Verified
    live by reverting ImageGrid.tsx's push lines to this shape: the main
    test above then failed listing exactly ['ref2vid', 'upscale']; restored
    after confirming."""
    backend_source = (
        'generation_type="img2vid"\n'
        'generation_type="ref2vid"\n'
        'generation_type="upscale"\n'
    )
    frontend_source = 'if (filterImg2Vid) types.push("img2vid");'  # pre-fix shape
    assert _missing_types(backend_source, frontend_source) == {"ref2vid", "upscale"}
