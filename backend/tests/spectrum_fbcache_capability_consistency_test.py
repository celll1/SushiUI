"""Spectrum/FBCache implementations must not be declared unsupported."""

import os
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
