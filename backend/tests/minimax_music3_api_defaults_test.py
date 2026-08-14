"""MiniMax Music 3 API layer, part 1: per-architecture audio defaults overlay.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_api_defaults_test.py -v

Design doc phase plan item 4's structural prerequisite: audio defaults were a
flat, ACE-Step-shaped dict (`AUDIO_GEN_DEFAULTS`/`TXT2AUD_DEFAULTS`) because
ACE-Step was the only audio architecture. This introduces the per-arch overlay
mechanism (`AUDIO_GEN_ARCH_OVERLAYS` + `audio_defaults_for_arch`, and its
aud2aud/outpaint twins) that video already had. The load-bearing claim this
file exists to prove is the NO-OP: ACE-Step's resolved defaults must be
bit-identical to what they were before this mechanism existed -- proved by
comparing against `AUDIO_GEN_DEFAULTS` itself, not merely asserted.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.param_defaults import (
    AUD2AUD_DEFAULTS,
    AUD2AUD_GEN_ARCH_OVERLAYS,
    AUDIO_GEN_ARCH_OVERLAYS,
    AUDIO_GEN_DEFAULTS,
    OUTPAINT_AUDIO_ARCH_OVERLAYS,
    OUTPAINT_AUDIO_DEFAULTS,
    aud2aud_defaults_for_arch,
    audio_defaults_for_arch,
    outpaint_audio_defaults_for_arch,
)
from api.generation_utils import resolve_audio_defaults


def test_acestep_audio_defaults_are_a_noop():
    """The no-op proof: `audio_defaults_for_arch("acestep")` must equal
    `AUDIO_GEN_DEFAULTS` -- the exact dict ACE-Step's txt2aud route resolved
    to before this overlay mechanism existed -- key for key, value for value.
    Not merely "close" or "compatible": EQUAL.
    """
    resolved = audio_defaults_for_arch("acestep")
    assert resolved == AUDIO_GEN_DEFAULTS
    # And every individual ACE-Step-consumed field, spelled out, so a future
    # accidental overlay entry for "acestep" fails loudly and specifically
    # rather than only via the dict-equality assert above.
    assert resolved["prompt"] == ""
    assert resolved["lyrics"] == ""
    assert resolved["audio_duration"] == 30.0
    assert resolved["seed"] == -1
    assert resolved["inference_steps"] == 8
    assert resolved["guidance_scale"] == 1.0
    assert resolved["shift"] == 3.0
    assert resolved["sampler_mode"] == "euler"
    assert resolved["vocal_language"] == "en"
    assert resolved["loras"] == []


def test_audio_defaults_for_arch_returns_a_copy_not_the_same_object():
    resolved = audio_defaults_for_arch("acestep")
    assert resolved is not AUDIO_GEN_DEFAULTS
    resolved["prompt"] = "mutated"
    assert AUDIO_GEN_DEFAULTS["prompt"] == ""


def test_unknown_or_missing_arch_resolves_to_base_unchanged():
    assert audio_defaults_for_arch(None) == AUDIO_GEN_DEFAULTS
    assert audio_defaults_for_arch("") == AUDIO_GEN_DEFAULTS
    assert audio_defaults_for_arch("some_future_arch") == AUDIO_GEN_DEFAULTS


def test_minimax_music3_overlay_matches_the_design_doc_table():
    """Design doc "Generation parameter contract": audio_duration=60.0 (an
    upper bound), num_inference_steps=30 (per chunk), flow_guidance_scale=1.7.
    AR CFG (1.5) and top-k (50) are fixed and never appear here at all.
    """
    resolved = audio_defaults_for_arch("minimax_music3")
    assert resolved["audio_duration"] == 60.0
    assert resolved["num_inference_steps"] == 30
    assert resolved["flow_guidance_scale"] == 1.7
    assert "ar_guidance_scale" not in resolved
    assert "top_k" not in resolved
    # Fields the overlay does NOT touch stay ACE-Step-shaped -- MiniMax Music 3
    # simply never reads them (its pipeline backend reads different keys), so
    # this is inert, not a claim that MiniMax Music 3 "uses" inference_steps=8.
    assert resolved["prompt"] == AUDIO_GEN_DEFAULTS["prompt"]
    assert resolved["lyrics"] == AUDIO_GEN_DEFAULTS["lyrics"]
    assert resolved["seed"] == AUDIO_GEN_DEFAULTS["seed"]


def test_minimax_music3_is_the_only_overlay_entry_today():
    assert set(AUDIO_GEN_ARCH_OVERLAYS.keys()) == {"minimax_music3"}


def test_aud2aud_twin_is_still_a_noop_for_every_arch():
    """Design doc phase plan item 8 (repaint/cover) is what populates
    `AUD2AUD_GEN_ARCH_OVERLAYS`; until then every arch, including
    "minimax_music3", resolves unchanged -- `/generate/aud2aud` still
    hard-refuses a loaded MiniMax Music 3 model regardless of what a
    resolved default would say.
    """
    assert AUD2AUD_GEN_ARCH_OVERLAYS == {}
    for arch in ("acestep", "minimax_music3", None, "unknown"):
        assert aud2aud_defaults_for_arch(arch) == AUD2AUD_DEFAULTS


def test_outpaint_audio_twin_is_populated_only_for_minimax_music3():
    """Design doc phase plan item 7 ("Extend") populates
    `OUTPAINT_AUDIO_ARCH_OVERLAYS` with a "minimax_music3" entry
    (`extend_duration_sec`/`num_inference_steps`/`flow_guidance_scale`) now
    that `/generate/outpaint/audio` resumes the autoregressive stage from
    the frame-code sidecar. ACE-Step and any unrecognized/absent arch still
    resolve to `OUTPAINT_AUDIO_DEFAULTS` unchanged -- the no-op contract
    this overlay mechanism was introduced to preserve for every arch that
    has nothing to add.
    """
    assert set(OUTPAINT_AUDIO_ARCH_OVERLAYS.keys()) == {"minimax_music3"}
    for arch in ("acestep", None, "unknown"):
        assert outpaint_audio_defaults_for_arch(arch) == OUTPAINT_AUDIO_DEFAULTS
    _music3_resolved = outpaint_audio_defaults_for_arch("minimax_music3")
    assert _music3_resolved["extend_duration_sec"] == 30.0
    assert _music3_resolved["num_inference_steps"] == 30
    assert _music3_resolved["flow_guidance_scale"] == 1.7
    # Every base ACE-Step-shaped key is still present and unchanged for
    # "minimax_music3" -- the overlay adds keys, it does not replace the base.
    for key in OUTPAINT_AUDIO_DEFAULTS:
        assert _music3_resolved[key] == OUTPAINT_AUDIO_DEFAULTS[key]


def test_outpaint_audio_defaults_compose_the_aud2aud_overlay_first():
    """`outpaint_audio_defaults_for_arch` mirrors
    `outpaint_video_defaults_for_arch`'s two-overlay composition (shared
    audio overlay, then the outpaint-only one). Exercised against "acestep"
    here, whose `AUD2AUD_GEN_ARCH_OVERLAYS` entry is still empty (item 8,
    repaint/cover, has not populated it) -- `test_outpaint_audio_twin_is_
    populated_only_for_minimax_music3` above covers the composed result for
    the one arch (`minimax_music3`) that DOES have a populated
    `OUTPAINT_AUDIO_ARCH_OVERLAYS` entry today (item 7, extend).
    """
    resolved = outpaint_audio_defaults_for_arch("acestep")
    for key in ("prompt", "lyrics", "seed", "inference_steps", "guidance_scale", "shift", "vocal_language"):
        assert resolved[key] == AUD2AUD_DEFAULTS[key]
    for key in ("total_duration", "input_offset_sec", "input_trim_start_sec", "input_trim_end_sec"):
        assert resolved[key] == OUTPAINT_AUDIO_DEFAULTS[key]


def test_resolve_audio_defaults_fills_only_client_omitted_fields():
    """The route-facing contract: `resolve_audio_defaults` mutates `params`
    in place, overwriting a key only when the client did NOT explicitly send
    it (mirrors `resolve_video_defaults` exactly).
    """
    params = {
        "prompt": "",
        "lyrics": "[verse]\nla la la",
        "audio_duration": 30.0,  # client-declared default (Pydantic), not sent
        "num_inference_steps": None,
        "flow_guidance_scale": None,
        "seed": -1,
    }
    # Client explicitly set num_inference_steps=10; everything else omitted.
    provided_keys = {"lyrics", "num_inference_steps"}
    params["num_inference_steps"] = 10

    resolved = resolve_audio_defaults(params, provided_keys, "minimax_music3")

    assert resolved["audio_duration"] == 60.0
    assert resolved["num_inference_steps"] == 30
    assert resolved["flow_guidance_scale"] == 1.7

    # Omitted fields got the arch's resolved default...
    assert params["audio_duration"] == 60.0
    assert params["flow_guidance_scale"] == 1.7
    # ...but the explicitly-provided field is untouched.
    assert params["num_inference_steps"] == 10
    # A field with no entry in the resolved map (not in the overlay AND not
    # in the base) is never touched at all.
    assert params["lyrics"] == "[verse]\nla la la"


def test_resolve_audio_defaults_is_a_noop_for_acestep_when_nothing_is_provided():
    """The route-level analog of the module-level no-op proof: an ACE-Step
    request that sends nothing gets exactly its Pydantic-declared defaults
    back, unchanged, through this helper.
    """
    params = dict(AUDIO_GEN_DEFAULTS)
    before = dict(params)
    resolved = resolve_audio_defaults(params, provided_keys=set(), arch="acestep")
    assert resolved == AUDIO_GEN_DEFAULTS
    assert params == before
