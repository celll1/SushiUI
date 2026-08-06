"""The video-outpaint audio mode's default is PER-ARCHITECTURE.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/outpaint_video_audio_mode_default_test.py -v

WHY THIS FILE EXISTS
--------------------
``outpaint_video_audio_mode`` names two modes whose CONSEQUENCE differs by
architecture, which is why one default cannot serve both:

* LTX-2.3 hands back a generated audio track spanning the WHOLE output
  timeline, so ``regenerate`` still leaves the preserved span with sound. It is
  a harmless default there.
* MiniMax-H3 generates audio and video jointly for ONE span, so it produces
  audio only for the frames it generates. Under ``regenerate`` the preserved
  span is left SILENT -- correct for the mode ("do not carry the input's audio
  over"), but for the commonest request, extending a clip that has sound, it
  means the ORIGINAL audio vanishes from the output. It is therefore not that
  architecture's default; ``OUTPAINT_VIDEO_ARCH_OVERLAYS`` overlays
  ``preserve_input``.

Both modes stay selectable everywhere. What is pinned here is only which one a
request that says nothing resolves to, on each architecture, through the same
resolver the route uses -- plus a NEGATIVE CONTROL proving these assertions
actually depend on the overlay entry rather than passing for some other reason.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import param_defaults  # noqa: E402
from api.param_defaults import (  # noqa: E402
    OUTPAINT_VIDEO_ARCH_OVERLAYS,
    OUTPAINT_VIDEO_DEFAULTS,
    outpaint_video_defaults_for_arch,
)

KEY = "outpaint_video_audio_mode"


# --------------------------------------------------------------------------
# The resolved default, per architecture
# --------------------------------------------------------------------------

def test_minimax_h3_resolves_to_preserve_input():
    """An omitted field on MiniMax-H3 must not silence the preserved span."""
    assert outpaint_video_defaults_for_arch("minimax_h3")[KEY] == "preserve_input"


def test_ltx2_default_is_unchanged():
    """LTX-2.3's whole-timeline generated track keeps `regenerate`."""
    assert outpaint_video_defaults_for_arch("ltx2")[KEY] == "regenerate"


@pytest.mark.parametrize("arch", [None, "", "sdxl", "not_an_arch"])
def test_unknown_arch_takes_the_base_value(arch):
    """The base map is LTX-2.3's semantics and the answer for an unresolved model."""
    assert outpaint_video_defaults_for_arch(arch)[KEY] == "regenerate"
    assert OUTPAINT_VIDEO_DEFAULTS[KEY] == "regenerate"


def test_the_overlay_is_what_carries_it():
    """The difference lives in the overlay, not in a base-map edit or a branch."""
    assert OUTPAINT_VIDEO_ARCH_OVERLAYS["minimax_h3"][KEY] == "preserve_input"
    # The other overlaid key is still there -- this entry was added to an
    # existing map, not substituted for it.
    assert OUTPAINT_VIDEO_ARCH_OVERLAYS["minimax_h3"]["total_frames"] == 248
    assert KEY not in param_defaults.VIDEO_GEN_ARCH_OVERLAYS.get("minimax_h3", {})


def test_both_modes_remain_the_only_two_values():
    """The default moved; the vocabulary did not."""
    for arch in ("minimax_h3", "ltx2", None):
        assert outpaint_video_defaults_for_arch(arch)[KEY] in ("regenerate", "preserve_input")


# --------------------------------------------------------------------------
# Negative control: remove the overlay entry, the H3 assertion must fail
# --------------------------------------------------------------------------

def test_removing_the_overlay_entry_breaks_the_h3_default(monkeypatch):
    """Proof that the assertions above are load-bearing.

    Without this, `test_minimax_h3_resolves_to_preserve_input` could be passing
    because of some unrelated default and would go on passing if the overlay
    entry were deleted.
    """
    stripped = {
        arch: {k: v for k, v in overlay.items() if k != KEY}
        for arch, overlay in OUTPAINT_VIDEO_ARCH_OVERLAYS.items()
    }
    monkeypatch.setattr(param_defaults, "OUTPAINT_VIDEO_ARCH_OVERLAYS", stripped)

    assert param_defaults.outpaint_video_defaults_for_arch("minimax_h3")[KEY] == "regenerate"
    # total_frames' overlay is untouched by the removal, so the resolver itself
    # is demonstrably still working -- only the audio-mode entry went away.
    assert param_defaults.outpaint_video_defaults_for_arch("minimax_h3")["total_frames"] == 248
    # LTX-2.3 is unaffected either way, which is the point of it being an overlay.
    assert param_defaults.outpaint_video_defaults_for_arch("ltx2")[KEY] == "regenerate"


# --------------------------------------------------------------------------
# The route must reach the overlay: the form field has to be a SENTINEL
# --------------------------------------------------------------------------

def test_route_field_is_a_sentinel_not_the_base_value():
    """A `Form(OUTPAINT_VIDEO_DEFAULTS[...])` default would freeze `regenerate`.

    FastAPI materialises a non-sentinel form default for every request, so an
    omitted field would arrive as "regenerate" and never reach the per-arch
    resolution below it. Only `Form(None)` leaves the omission visible.
    """
    import inspect

    from api.routes import generate_outpaint_video

    param = inspect.signature(generate_outpaint_video).parameters[KEY]
    assert param.default.default is None, (
        "outpaint_video_audio_mode must be a Form(None) sentinel so the "
        "per-architecture default can be applied"
    )
