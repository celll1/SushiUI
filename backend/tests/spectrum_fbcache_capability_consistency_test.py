"""Spectrum/FBCache: the capability table and the pipeline code must agree, in
BOTH directions -- and where the table is right, the generation panels that
render a checkbox for the feature must actually consult it.

Two defect instances motivated this file, found in the same investigation:

1. **Offered but inert** (the defect this file was written for). Txt2ImgPanel,
   Img2ImgPanel, InpaintPanel and OutpaintPanel all render the Spectrum and
   FBCache checkboxes unconditionally. On MiniMax-H3 both are accepted by the
   route and stored in the generation record, and neither did anything at the
   time. MiniMax-H3 now has opt-in paired video/audio Spectrum and guarded
   FBCache paths. `cfg`/`negative_prompt`/
   `text_encoder_quantization` in the same files already gated on
   `archSupportsFeature`; Spectrum/FBCache did not.

2. **Falsely unsupported** (the mirror bug, found while fixing #1).
   `arch_capabilities.py`'s `_SPECTRUM_UNSUPPORTED` listed zimage, ideogram4,
   lens, minit2i, anima and ltx2 as not implementing Spectrum/FBCache. All six
   genuinely do (each arch's `*_pipeline_ops.py` / pipeline_backends file calls
   `build_output_forecaster(spectrum_params, ...)` and
   `fbcache_active(spectrum_params)` / `build_fbcache(...)` from real request
   params) -- the comment "implemented for the U-Net (and FLUX.2)" was simply
   never updated as each architecture gained the feature. A live request on any
   of those six with spectrum_enable/fbcache_enable set got a false
   "not supported ... and was ignored" warning while the feature actually ran.

Both directions are expressible here because both properties are STATIC:
"does this arch's implementation file contain the literal trigger key" and
"does this arch appear in ARCH_UNSUPPORTED" are both string/dict facts, no
model load or generation required. Direction 1 additionally needs no import at
all -- it is checked by reading the four generation panels' TypeScript source
as text, the same way `gallery_type_visibility_test.py` reads ImageGrid.tsx.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/spectrum_fbcache_capability_consistency_test.py -v
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BACKEND = os.path.join(REPO_ROOT, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def _read(relpath: str) -> str:
    with open(os.path.join(REPO_ROOT, *relpath.split("/")), "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Direction 2: falsely unsupported (table says inert, code proves it is not).
# ---------------------------------------------------------------------------

# Arch -> implementation file.
_COMMON_SPECTRUM_FBCACHE_IMPL_FILES = {
    "sd15": "backend/core/pipeline.py",
    "sdxl": "backend/core/pipeline.py",
    "zimage": "backend/core/pipeline_backends/zimage.py",
    "flux2": "backend/core/pipeline_backends/flux2.py",
    "ideogram4": "backend/core/models/ideogram4/ideogram4_pipeline_ops.py",
    "lens": "backend/core/models/lens/lens_pipeline_ops.py",
    "anima": "backend/core/models/anima/anima_pipeline_ops.py",
    "minit2i": "backend/core/models/minit2i/minit2i_pipeline_ops.py",
    "ltx2": "backend/core/pipeline_backends/ltx2.py",
}
_SPECTRUM_IMPL_FILES = {
    **_COMMON_SPECTRUM_FBCACHE_IMPL_FILES,
    "minimax_h3": "backend/core/models/minimax_h3/h3_pipeline_ops.py",
}
_FBCACHE_IMPL_FILES = {
    **_COMMON_SPECTRUM_FBCACHE_IMPL_FILES,
    "minimax_h3": "backend/core/models/minimax_h3/h3_pipeline_ops.py",
}


def _archs_implementing(needles, impl_files: dict) -> set:
    """Archs whose implementation file contains ANY of `needles`.

    Two call shapes exist in this codebase for the same feature: sd15/sdxl reads
    the literal request key directly (`params.get("spectrum_enable", ...)`,
    passed on as a kwarg); every DiT arch instead calls the SHARED helper
    (`build_output_forecaster(...)` / `fbcache_active(...)` in
    core/inference/spectrum_forecaster.py / fbcache.py) that reads the literal
    key itself -- so the literal string never appears in those files at all.
    Both shapes are genuine implementations; `needles` covers both so neither is
    missed.
    """
    if isinstance(needles, str):
        needles = (needles,)
    found = set()
    for arch, relpath in impl_files.items():
        source = _read(relpath)
        if any(needle in source for needle in needles):
            found.add(arch)
    return found


def _falsely_unsupported(unsupported: dict, implementing: set, feature: str) -> set:
    """Archs that implement `feature` but are declared unsupported for it."""
    return {a for a in implementing if feature in unsupported.get(a, {})}


def test_no_arch_that_implements_spectrum_is_declared_unsupported():
    from api.arch_capabilities import ARCH_UNSUPPORTED

    implementing = _archs_implementing(
        ('"spectrum_enable"', "build_output_forecaster("), _SPECTRUM_IMPL_FILES)
    bad = _falsely_unsupported(ARCH_UNSUPPORTED, implementing, "spectrum")
    assert not bad, (
        f"{sorted(bad)} genuinely read spectrum_enable (build_output_forecaster is "
        f"called from real request params in their pipeline_backends/*_pipeline_ops "
        f"file) but arch_capabilities.py lists them as not supporting Spectrum -- a "
        f"live request there gets a false 'not supported ... and was ignored' "
        f"warning while the feature actually runs.")


def test_no_arch_that_implements_fbcache_is_declared_unsupported():
    from api.arch_capabilities import ARCH_UNSUPPORTED

    implementing = _archs_implementing(
        ('"fbcache_enable"', "fbcache_active("), _FBCACHE_IMPL_FILES)
    bad = _falsely_unsupported(ARCH_UNSUPPORTED, implementing, "fbcache")
    assert not bad, (
        f"{sorted(bad)} genuinely read fbcache_enable (fbcache_active/build_fbcache "
        f"are called from real request params in their pipeline_backends/"
        f"*_pipeline_ops file) but arch_capabilities.py lists them as not "
        f"supporting FBCache -- a live request there gets a false 'not supported "
        f"... and was ignored' warning while the feature actually runs.")


def test_the_implementing_set_is_not_trivially_empty():
    """Sanity check: the extractor really finds the archs known to implement it."""
    implementing = _archs_implementing(
        ('"spectrum_enable"', "build_output_forecaster("), _SPECTRUM_IMPL_FILES)
    for arch in ("zimage", "ideogram4", "lens", "anima", "minit2i", "ltx2", "flux2",
                 "minimax_h3"):
        assert arch in implementing, f"extractor missed known implementer {arch!r}"


def test_checker_catches_the_pre_fix_table_regression():
    """MUTANT: arch_capabilities.py's shape before this fix -- zimage/ltx2 listed
    unsupported for spectrum despite both genuinely implementing it. Verified live
    by reverting _SPECTRUM_UNSUPPORTED to include them: this test then failed
    listing exactly {'zimage', 'ltx2'}; restored after confirming."""
    stale_unsupported = {
        "zimage": {"spectrum": "Spectral Feature Forecasting is not implemented for this architecture's sampler"},
        "ltx2": {"spectrum": "Spectral Feature Forecasting is not implemented for this architecture's sampler"},
        "krea2": {"spectrum": "Spectral Feature Forecasting is not implemented for this architecture's sampler"},
    }
    implementing = {"zimage", "ltx2", "flux2"}  # krea2 deliberately absent: no codepath
    assert _falsely_unsupported(stale_unsupported, implementing, "spectrum") == {"zimage", "ltx2"}


# ---------------------------------------------------------------------------
# Direction 1: offered but inert (a panel renders the checkbox without gating
# it on the capability the table -- now correctly -- describes).
# ---------------------------------------------------------------------------

_PANEL_FILES = [
    "frontend/src/components/generation/Txt2ImgPanel.tsx",
    "frontend/src/components/generation/Img2ImgPanel.tsx",
    "frontend/src/components/generation/InpaintPanel.tsx",
    "frontend/src/components/generation/OutpaintPanel.tsx",
]

_CHECKBOX_RE = {
    "spectrum": re.compile(r'id="[a-zA-Z0-9_]*spectrum_enable[a-zA-Z0-9_]*"'),
    "fbcache": re.compile(r'id="[a-zA-Z0-9_]*fbcache_enable[a-zA-Z0-9_]*"'),
}
_GATE_RE = {
    "spectrum": re.compile(r'archSupportsFeature\([^)]*"spectrum"\)'),
    "fbcache": re.compile(r'archSupportsFeature\([^)]*"fbcache"\)'),
}


def _panel_missing_gate(source: str, feature: str) -> bool:
    """True when `source` renders the feature's checkbox but never derives the
    matching archSupportsFeature(..., feature) gate anywhere in the file.

    LIMITATION (same shape as gallery_type_visibility_test.py): this is a
    set-membership/presence check, not a proof the gate variable actually wraps
    every render of the checkbox -- a gate computed but never used around the
    JSX would still pass. It is what caught the real regression (the gate was
    entirely absent, not merely misplaced).
    """
    if not _CHECKBOX_RE[feature].search(source):
        return False  # panel does not offer this control at all -- nothing to gate
    return not _GATE_RE[feature].search(source)


def test_every_panel_offering_the_spectrum_checkbox_derives_the_capability_gate():
    offenders = [p for p in _PANEL_FILES if _panel_missing_gate(_read(p), "spectrum")]
    assert not offenders, (
        f"{offenders} render a spectrum_enable checkbox with no "
        f"archSupportsFeature(..., \"spectrum\") gate anywhere in the file -- the "
        f"control is offered even on an architecture (MiniMax-H3) whose sampler "
        f"never reads spectrum_enable at all.")


def test_every_panel_offering_the_fbcache_checkbox_derives_the_capability_gate():
    offenders = [p for p in _PANEL_FILES if _panel_missing_gate(_read(p), "fbcache")]
    assert not offenders, (
        f"{offenders} render a fbcache_enable checkbox with no "
        f"archSupportsFeature(..., \"fbcache\") gate anywhere in the file -- the "
        f"control is offered even on an architecture whose sampler ignores it.")


def test_checker_catches_the_pre_fix_panel_regression():
    """MUTANT: Txt2ImgPanel.tsx's shape before this fix -- the checkbox rendered
    unconditionally, with no archSupportsFeature("spectrum") call in the file."""
    pre_fix_source = """
    acceleration: () => (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
        <div className="space-y-2">
          <input type="checkbox" id="spectrum_enable"
            checked={params.spectrum_enable || false}
            onChange={(e) => setParams({ ...params, spectrum_enable: e.target.checked })} />
        </div>
      </div>
    ),
    """
    assert _panel_missing_gate(pre_fix_source, "spectrum") is True


def test_checker_accepts_the_post_fix_panel_shape():
    """Positive control: the gated shape this fix produced must NOT be flagged."""
    post_fix_source = """
    const supportsSpectrum = archSupportsFeature(archCapabilities, loadedArch, "spectrum");
    acceleration: () => (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
        {supportsSpectrum && (
        <div className="space-y-2">
          <input type="checkbox" id="spectrum_enable"
            checked={params.spectrum_enable || false}
            onChange={(e) => setParams({ ...params, spectrum_enable: e.target.checked })} />
        </div>
        )}
      </div>
    ),
    """
    assert _panel_missing_gate(post_fix_source, "spectrum") is False
