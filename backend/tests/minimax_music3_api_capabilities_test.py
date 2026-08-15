"""MiniMax Music 3 API layer, part 2: arch_capabilities refusals.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_api_capabilities_test.py -v

Design doc "Capability verdict": the first three rows of that table are
properties of the RELEASED MODEL, not unimplemented features, and must be
encoded here with those reasons -- text-to-music (yes, not a refusal),
reference-audio conditioning of the autoregressive stage (no: the RVQ
tokenizer's encoder is unpublished), and negative prompt (no: the flow-stage
unconditional branch is zeros, the AR unconditional branch is the
token-masked prompt). This file checks the warn-table wiring for those two
refusals plus the ACE-Step-parity entries (advanced_cfg/nag/controlnets/lora/
unet_quantization/vae_override/text_encoder_quantization/cpu_text_encoding/
attention_impl/quantized_gemm) and the DIT_ARCHS/spectrum/FBCache membership
the design doc's item 4 also requires.

REACHABILITY (audit finding F6). `check_arch_capabilities` only fires when a
feature's trigger key is present in `params` with a non-default value, and
the only route reachable for this architecture today is `/generate/txt2aud`
(`Txt2AudRequest`'s DECLARED fields). Two different situations exist in this
table and this file is explicit about which is which, rather than exercising
every entry through one identical hand-built-dict pattern that cannot tell
them apart:

  - `lora` is REACHABLE: `loras` is a real `Txt2AudRequest` field.
    `test_lora_refusal_fires_through_a_real_txt2aud_request` below builds its
    params from an actual `Txt2AudRequest` instance, not a hand-built dict,
    so it would fail if that field were ever renamed.
  - `negative_prompt`/`audio_reference_conditioning` are NOT reachable on
    `/generate/txt2aud` today -- neither trigger key
    (`negative_prompt`/`reference_audio_path`/`reference_audio_enable`/
    `is_cover`) is a `Txt2AudRequest` field, which
    `test_negative_prompt_and_reference_audio_conditioning_trigger_keys_are_not_reachable_today`
    asserts explicitly. The two `test_check_arch_capabilities_warns_on_*`
    tests for them below feed a HAND-BUILT dict with the key force-inserted
    -- that only proves the table ENTRY and its trigger-key WIRING exist
    (`check_arch_capabilities`'s generic mechanism would fire if a future
    surface, e.g. an aud2aud "cover" request, ever carried that key under
    that exact name), not that any current request can reach it. Read them
    as "the entry is wired correctly", not "this refusal is live".
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.arch_capabilities import (
    _DIT_ARCHS,
    _FBCACHE_UNSUPPORTED,
    _SPECTRUM_UNSUPPORTED,
    ARCH_SUPPORTED_VALUES,
    ARCH_UNSUPPORTED,
    FEATURE_PARAMS,
    check_arch_capabilities,
)
from api.param_defaults import audio_defaults_for_arch


def _defaults():
    return audio_defaults_for_arch("minimax_music3")


def test_minimax_music3_is_in_dit_archs_and_spectrum_and_fbcache_unsupported():
    assert "minimax_music3" in _DIT_ARCHS
    assert "minimax_music3" in _SPECTRUM_UNSUPPORTED
    assert "minimax_music3" in _FBCACHE_UNSUPPORTED


def test_minimax_music3_has_the_same_shape_of_entries_acestep_has():
    """Not a value-for-value comparison (the reasons name the real mechanism
    per architecture), but the same FEATURE KEYS must be refused, since
    MiniMax Music 3 has no more of these mechanisms than ACE-Step does.
    """
    acestep_features = set(ARCH_UNSUPPORTED.get("acestep", {}))
    music3_features = set(ARCH_UNSUPPORTED.get("minimax_music3", {}))
    assert acestep_features.issubset(music3_features)


def test_negative_prompt_refusal_reason_matches_the_design_doc():
    reason = ARCH_UNSUPPORTED["minimax_music3"]["negative_prompt"]
    assert "zeros" in reason
    assert "token-masked" in reason


def test_reference_audio_conditioning_refusal_reason_matches_the_design_doc():
    reason = ARCH_UNSUPPORTED["minimax_music3"]["audio_reference_conditioning"]
    assert "RVQ" in reason
    assert "not published" in reason or "unpublished" in reason


def test_unet_quantization_is_fully_unsupported_with_no_exempt_values():
    """Unlike ACE-Step/Krea2/Ideogram4/LTX-2.3, MiniMax Music 3 has no
    per-generation `unet_quantization` value implemented AT ALL (phase 1
    loads BF16/FP16 only) -- so, unlike those architectures, it must have NO
    `ARCH_SUPPORTED_VALUES` exemption.
    """
    assert "unet_quantization" in ARCH_UNSUPPORTED["minimax_music3"]
    assert "unet_quantization" not in ARCH_SUPPORTED_VALUES.get("minimax_music3", {})


def test_quantized_gemm_is_unsupported_derived_from_dit_archs_membership():
    # Not an explicit `_add` call -- this is the DERIVED bulk-loop entry
    # (`_ALL_ARCHS = ["sd15", "sdxl"] + _DIT_ARCHS`), which only fires because
    # "minimax_music3" is in `_DIT_ARCHS` and absent from `QUANTIZED_LINEAR_ARCHS`.
    assert "quantized_gemm" in ARCH_UNSUPPORTED["minimax_music3"]


def test_negative_prompt_and_reference_audio_conditioning_trigger_keys_are_not_reachable_today():
    """The honesty check the two tests below need (audit finding F6): neither
    feature's trigger key is a real `Txt2AudRequest` field, so NEITHER
    refusal can fire on the only route reachable for this architecture. The
    table entries are still correct to keep (design doc: they document
    properties of the released model), but this pins that they are
    documentation-only today rather than letting the tests below imply
    otherwise by omission.
    """
    from api.routes import Txt2AudRequest

    request_fields = set(Txt2AudRequest.model_fields.keys())
    for feature in ("negative_prompt", "audio_reference_conditioning"):
        for trigger_key in FEATURE_PARAMS[feature]:
            assert trigger_key not in request_fields, (
                f"{trigger_key!r} (feature {feature!r}) is now a real Txt2AudRequest field -- "
                f"the refusal is reachable; update this test and the two warn-on-set tests' "
                f"docstrings to say so."
            )


def test_check_arch_capabilities_warns_on_negative_prompt_when_forced():
    """Proves the table ENTRY and its FEATURE_PARAMS wiring, not a live
    refusal -- see this module's docstring "REACHABILITY". `negative_prompt`
    is force-inserted into a hand-built dict because no real request can
    carry it today (pinned by the test above).
    """
    params = dict(_defaults())
    params["negative_prompt"] = "not silence"
    warnings = check_arch_capabilities(params, "minimax_music3", defaults=_defaults())
    messages = [w["message"] for w in warnings]
    assert any("negative_prompt" in m for m in messages)


def test_check_arch_capabilities_warns_on_reference_audio_conditioning_when_forced():
    """Proves the table ENTRY and its FEATURE_PARAMS wiring, not a live
    refusal -- see this module's docstring "REACHABILITY". `is_cover` is
    force-inserted into a hand-built dict because no real `Txt2AudRequest`/
    `Aud2AudRequest` field carries it today (pinned by the test above); its
    real surface is an aud2aud `mode="cover"` request (design doc phase plan
    item 8), which `MiniMaxMusic3Mixin._generate_aud2aud_minimax_music3`
    refuses outright for this architecture with the RVQ-tokenizer-encoder
    capability reason, at the mechanism layer rather than this warning table.
    """
    params = dict(_defaults())
    params["is_cover"] = True
    warnings = check_arch_capabilities(params, "minimax_music3", defaults=_defaults())
    messages = [w["message"] for w in warnings]
    assert any("reference-audio conditioning" in m for m in messages)


def test_lora_refusal_fires_through_a_real_txt2aud_request():
    """Unlike the two tests above, this one IS a live-refusal proof (audit
    finding F2): `loras` is a genuine `Txt2AudRequest` field, so this builds
    `params` from an actual model instance (`.dict()`), the same way
    `routes.generate_txt2aud` does, rather than a hand-built dict.
    """
    from api.routes import Txt2AudRequest

    request = Txt2AudRequest(
        prompt="ambient", lyrics="[verse]\nla",
        loras=[{"path": "some_lora.safetensors", "strength": 1.0}],
    )
    params = request.dict()
    warnings = check_arch_capabilities(params, "minimax_music3", defaults=_defaults())
    messages = [w["message"] for w in warnings]
    assert any("loras" in m or "LoRA" in m for m in messages)


def test_check_arch_capabilities_warns_on_advanced_cfg_nag_controlnets():
    params = dict(_defaults())
    params["cfg_schedule_type"] = "linear"
    params["nag_enable"] = True
    params["controlnets"] = [{"type": "canny"}]
    warnings = check_arch_capabilities(params, "minimax_music3", defaults=_defaults())
    features_warned = {w["message"] for w in warnings}
    assert any("advanced CFG" in m for m in features_warned)
    assert any("Normalized Attention Guidance" in m for m in features_warned)
    assert any("ControlNet" in m for m in features_warned)


def test_check_arch_capabilities_warns_on_unet_quantization_for_any_value():
    params = dict(_defaults())
    params["unet_quantization"] = "int8"
    warnings = check_arch_capabilities(params, "minimax_music3", defaults=_defaults())
    assert any("unet_quantization" in w["message"] for w in warnings)


def test_check_arch_capabilities_is_silent_when_nothing_unsupported_is_set():
    params = dict(_defaults())
    warnings = check_arch_capabilities(params, "minimax_music3", defaults=_defaults())
    assert warnings == []


def test_check_arch_capabilities_does_not_warn_for_the_music3_only_params():
    """`audio_duration`/`num_inference_steps`/`flow_guidance_scale` are real,
    honored MiniMax Music 3 parameters -- setting them must never trip a
    warning, even at a non-ACE-Step value, as long as `defaults` is the
    RESOLVED per-arch map (not the ACE-Step base or GENERATION_DEFAULTS).
    """
    params = dict(_defaults())
    params["audio_duration"] = 120.0
    params["num_inference_steps"] = 45
    params["flow_guidance_scale"] = 2.5
    warnings = check_arch_capabilities(params, "minimax_music3", defaults=_defaults())
    assert warnings == []
