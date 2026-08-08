"""Every generation panel that offers a video-mode width/height slider must
cap it from `videoCanvasAxisBounds` (backed by the arch's `max_pixel_hw`
envelope, `frontend/src/utils/api.ts`) -- otherwise that panel's sliders let
the user build a canvas the backend's `validate_video_geometry` hard-400s
on, while an identical control in another panel already refuses it. This is
exactly the class of bug this task fixed in Txt2ImgPanel.tsx: the guard
existed in Img2ImgPanel/InpaintPanel/OutpaintPanel but not there.

Static cross-file consistency check, not a browser test (no JS test runner
in this repo, see `frontend/package.json`): it scans each panel's source as
text.

Every panel's grid-aligned resolution slider (video or still-image) carries
a "<n>" divisor mark in its label, e.g. `Width (÷${videoWidthBounds.step})`
or the pre-fix `Width (÷32)` -- that mark is what distinguishes a
resolution-grid slider from an unrelated one, and is common to every panel
already fixed. Among THOSE sliders, a video-mode one (its onChange writes
`width: parseInt(e.target.value)` under `isVideo`) must set `max=` to an
EXPRESSION (a bounds variable), not a numeric literal -- a literal `max`
cannot be reading the loaded architecture's envelope, since that envelope is
only known at render time from the capability matrix.

LIMITATION: this confirms the `max` prop is not a bare number, not that the
expression it names is actually `videoCanvasAxisBounds(...)`'s result --
a panel could satisfy this check with an unrelated non-literal expression.
Checking the exact call chain would need a real JSX/TS parse this repo has
no tooling for; the shape-based check below is the same tradeoff
`gallery_type_visibility_test.py` makes for generation_type literals.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_canvas_envelope_consistency_test.py -v
"""

import glob
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERATION_DIR = os.path.join(REPO_ROOT, "frontend", "src", "components", "generation")

WIDTH_SETTER = "width: parseInt(e.target.value)"
GRID_MARK = "÷"  # the division-sign divisor mark, e.g. "Width (÷32)"
SLIDER_BLOCK_RE = re.compile(r"<Slider\b.*?/>", re.DOTALL)
MAX_PROP_RE = re.compile(r"max=\{([^}]*)\}")
LITERAL_RE = re.compile(r"-?\d+(\.\d+)?")


def _panel_sources() -> dict:
    sources = {}
    for path in glob.glob(os.path.join(GENERATION_DIR, "*.tsx")):
        with open(path, "r", encoding="utf-8") as f:
            sources[os.path.basename(path)] = f.read()
    return sources


def _video_width_slider_blocks(source: str) -> list:
    """Every <Slider> block that is both a grid-aligned resolution control
    (carries the divisor mark) and the video-canvas width control (its
    onChange writes the width setter)."""
    return [
        b for b in SLIDER_BLOCK_RE.findall(source)
        if WIDTH_SETTER in b and GRID_MARK in b
    ]


def _max_is_hardcoded_literal(block: str) -> bool:
    m = MAX_PROP_RE.search(block)
    if not m:
        return True  # no max prop at all can't be envelope-derived either
    return bool(LITERAL_RE.fullmatch(m.group(1).strip()))


def _panels_with_hardcoded_video_width_max(sources: dict) -> list:
    missing = []
    for name, src in sources.items():
        for block in _video_width_slider_blocks(src):
            if _max_is_hardcoded_literal(block):
                missing.append(name)
                break
    return sorted(missing)


def test_every_video_width_slider_max_is_envelope_derived():
    sources = _panel_sources()
    missing = _panels_with_hardcoded_video_width_max(sources)
    assert not missing, (
        f"{missing} render a video-mode width slider (grid-aligned label + "
        f"a '{WIDTH_SETTER}' onChange) whose max={{...}} is a numeric literal "
        f"-- it cannot be reading the loaded architecture's max_pixel_hw "
        f"envelope from videoCanvasAxisBounds, so it can build a canvas the "
        f"backend rejects with a 400 that another panel's identical control "
        f"already prevents (the Txt2ImgPanel gap this task fixed)."
    )


def test_the_detector_actually_finds_the_known_video_width_sliders():
    """Sanity check the regex isn't silently matching nothing."""
    sources = _panel_sources()
    found = {
        name for name, src in sources.items()
        if _video_width_slider_blocks(src)
    }
    for expected in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx", "InpaintPanel.tsx", "OutpaintPanel.tsx"):
        assert expected in found, f"detector missed known video-width slider in {expected!r}"


# --------------------------------------------------------------------------
# Negative control: the historical (pre-this-task) Txt2ImgPanel shape,
# reproduced as an in-memory source string, to prove the checker above is
# sensitive to exactly this regression class.
# --------------------------------------------------------------------------

def test_checker_catches_the_hardcoded_slider_regression():
    """MUTANT: Txt2ImgPanel.tsx before this task's fix -- a video Width
    slider with the grid-aligned divisor label but min/max/step hardcoded to
    literals (32/2048/32) instead of a `videoWidthBounds` result. Verified
    live against the real file: reverting Txt2ImgPanel.tsx's video
    Width/Height sliders to this exact literal shape made this checker flag
    ['Txt2ImgPanel.tsx'] (confirmed with the actual file, then restored) --
    a WEAKER, file-presence-only version of this check (an earlier draft
    that only asked "does the file call videoCanvasAxisBounds anywhere")
    did NOT catch that mutation, because the file still calls it elsewhere
    for its bounds *variables* while the JSX itself stayed hardcoded; this
    version inspects the Slider prop itself for exactly that reason."""
    sources = {
        "Txt2ImgPanel.tsx": (
            "const videoWidthBounds = videoCanvasAxisBounds(archCapabilities, loadedArch, h);\n"
            '<Slider label="Width (÷32)" min={32} max={2048} step={32} '
            'value={params.width ?? 768} '
            "onChange={(e) => setParams({ ...params, width: parseInt(e.target.value) })} />"
        ),
    }
    assert _panels_with_hardcoded_video_width_max(sources) == ["Txt2ImgPanel.tsx"]
