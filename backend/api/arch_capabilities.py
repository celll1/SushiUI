"""Per-architecture generation-parameter capability table.

Several generation parameters are silently ignored by some model
architectures (e.g. ``use_torch_compile`` has no effect on the DiT archs,
the advanced-CFG block is a U-Net-only feature, etc.). Historically these
were dropped with no feedback to the API caller. This module records which
parameters/features a given architecture does NOT honor, together with a
short factual reason, so the generation routes can surface a warning when a
user explicitly set one of them.

The table doubles as documentation: each entry's value is a one-line,
factual explanation of why the parameter has no effect on that arch.

``check_arch_capabilities(params, arch)`` is called once per generation
(after ``start_generation``); for every feature the loaded architecture
ignores, it emits ONE ``add_warning`` when the user set a related parameter
to a non-default value (compared against ``GENERATION_DEFAULTS``).
"""
from typing import Any, Dict, List, Optional

from api.param_defaults import GENERATION_DEFAULTS
# Re-exported (and served by GET /schema/arch-capabilities) so the generation
# panels can offer unet_quantization="int8" exactly where the in-place converter
# is wired, without a second hardcoded arch list in the frontend. The tuple is
# owned by the module that implements the conversion.
from core.models.common.int8_runtime_quantize import (
    QUANTIZED_LINEAR_ARCHS, RUNTIME_INT8_ARCHS, arch_names,
)

# ---------------------------------------------------------------------------
# Feature -> the parameter keys that "arm" it. When any of these keys is set
# to a non-default value the feature is considered requested by the user.
# For enable-gated features only the enable flag is used as the trigger, so
# tweaking a sub-parameter without enabling the feature does not warn.
# ---------------------------------------------------------------------------
FEATURE_PARAMS: Dict[str, List[str]] = {
    "use_torch_compile": ["use_torch_compile"],
    "advanced_cfg": [
        "cfg_schedule_type", "cfg_schedule_min", "cfg_schedule_max",
        "cfg_schedule_power", "cfg_rescale_snr_alpha",
        "dynamic_threshold_percentile", "dynamic_threshold_mimic_scale",
    ],
    "spectrum": ["spectrum_enable"],
    "fbcache": ["fbcache_enable"],
    "nag": ["nag_enable"],
    "controlnets": ["controlnets"],
    "unet_quantization": ["unet_quantization"],
    "quantized_gemm": ["quantized_gemm_mode"],
    "text_encoder_quantization": ["text_encoder_quantization"],
    "cpu_text_encoding": ["cpu_text_encoding"],
    "attention_impl": ["attention_impl"],
    "vae_drift_correction": ["vae_drift_correction"],
    "flatten_in_loop": ["flatten_in_loop"],
    "te_override": ["text_encoder_path"],
    "vae_override": ["vae_path"],
    # Guidance. Split from `advanced_cfg` (which is the U-Net scheduling block)
    # because a guidance-DISTILLED architecture ignores the guidance scale
    # itself, not just its schedule.
    "cfg": ["guidance_scale", "cfg_scale"],
    "negative_prompt": ["negative_prompt"],
}

# Human-readable label used in the warning message for each feature.
FEATURE_LABELS: Dict[str, str] = {
    "use_torch_compile": "use_torch_compile",
    "advanced_cfg": "advanced CFG (cfg_schedule_*/dynamic_threshold_*/cfg_rescale_snr_alpha)",
    "spectrum": "spectrum_* (Spectral Feature Forecasting)",
    "fbcache": "fbcache_* (First Block Cache)",
    "nag": "nag_* (Normalized Attention Guidance)",
    "controlnets": "controlnets",
    "unet_quantization": "unet_quantization",
    "quantized_gemm": "quantized_gemm_mode (quantized GEMM path)",
    "text_encoder_quantization": "text_encoder_quantization",
    "cpu_text_encoding": "cpu_text_encoding",
    "attention_impl": "attention_impl",
    "vae_drift_correction": "vae_drift_correction (VAE DC-drift correction)",
    "flatten_in_loop": "flatten_in_loop (in-loop hard background flatten)",
    "te_override": "text_encoder_path (text-encoder override)",
    "vae_override": "vae_path (VAE override)",
    "cfg": "guidance_scale/cfg_scale (classifier-free guidance)",
    "negative_prompt": "negative_prompt",
}

# ---------------------------------------------------------------------------
# ARCH_UNSUPPORTED[arch][feature] = short factual reason the feature has no
# effect on that architecture.
# ---------------------------------------------------------------------------
_DIT_ARCHS = ["zimage", "flux2", "ideogram4", "lens", "minit2i", "anima", "krea2", "ltx2", "acestep",
              "minimax_h3"]
_SPECTRUM_UNSUPPORTED = ["zimage", "ideogram4", "lens", "minit2i", "anima", "krea2", "ltx2", "acestep",
                         "minimax_h3"]

ARCH_UNSUPPORTED: Dict[str, Dict[str, str]] = {}

# ---------------------------------------------------------------------------
# TRAINING_UNSUPPORTED[arch][training_method] = the factual reason that method
# cannot be run for that architecture.
#
# A DIFFERENT axis from ARCH_UNSUPPORTED, which is about generation parameters
# that are accepted and ignored. An entry here is a REFUSAL: the trainer raises
# rather than warns, and the table exists so the UI can filter its method
# dropdown from the same source instead of discovering the refusal after a run
# has been queued. Served as `training_unsupported` by
# GET /schema/arch-capabilities.
# ---------------------------------------------------------------------------
TRAINING_UNSUPPORTED: Dict[str, Dict[str, str]] = {}


def _add_training_unsupported(arch: str, method: str, reason: str) -> None:
    TRAINING_UNSUPPORTED.setdefault(arch, {})[method] = reason

# ---------------------------------------------------------------------------
# ARCH_SUPPORTED_VALUES[arch][feature] = the VALUES of the feature's arming
# parameter that the architecture DOES honor, even though the feature is listed
# unsupported above.
#
# Needed because a feature is not always all-or-nothing per architecture:
# `unet_quantization` on Krea 2 ignores the FP8 values (its FP8 story is
# checkpoint-format-driven) but applies `"int8"`, which converts an unquantized
# transformer in place. Recording that as a value exemption keeps the
# unsupported reason accurate for the values it really does ignore, instead of
# either warning wrongly on `int8` or going silent on the FP8 values.
#
# Only meaningful for single-value (string) parameters.
# ---------------------------------------------------------------------------
ARCH_SUPPORTED_VALUES: Dict[str, Dict[str, List[str]]] = {}


def _add(arch: str, feature: str, reason: str) -> None:
    ARCH_UNSUPPORTED.setdefault(arch, {})[feature] = reason


def _add_supported_values(arch: str, feature: str, values: List[str]) -> None:
    ARCH_SUPPORTED_VALUES.setdefault(arch, {})[feature] = list(values)


# use_torch_compile: only wired into the U-Net (SD1.5/SDXL) path; DiT archs
# run their own forward and never consult it.
for _a in _DIT_ARCHS:
    _add(_a, "use_torch_compile",
         "torch.compile is only applied to the SD1.5/SDXL U-Net; this DiT architecture ignores it")

# Advanced CFG block: implemented in the U-Net custom sampling loop; the
# flow-matching DiT samplers do not run it.
for _a in ["zimage", "flux2", "minit2i"]:
    _add(_a, "advanced_cfg",
         "CFG scheduling / dynamic thresholding / CFG-rescale run only in the U-Net sampling loop, not in this DiT sampler")

# Spectrum forecasting: implemented for the U-Net (and FLUX.2); not wired for
# these architectures.
for _a in _SPECTRUM_UNSUPPORTED:
    _add(_a, "spectrum",
         "Spectral Feature Forecasting is not implemented for this architecture's sampler")

# First Block Cache: same set as spectrum, minus flux2 (flux2 supports fbcache).
for _a in [a for a in _SPECTRUM_UNSUPPORTED if a != "flux2"]:
    _add(_a, "fbcache",
         "First Block Cache is not implemented for this architecture's sampler")

# NAG, ControlNet: not supported by Krea 2.
_add("krea2", "nag", "Normalized Attention Guidance is not implemented for Krea 2")
_add("krea2", "controlnets", "ControlNet is not supported for Krea 2")

# LTX-2.3 is a video model with its own flow-matching sampler; the image-oriented
# guidance/conditioning features do not apply.
_add("ltx2", "advanced_cfg",
     "CFG scheduling / dynamic thresholding / CFG-rescale run only in the U-Net sampling loop, not in the LTX-2.3 video sampler")
_add("ltx2", "nag", "Normalized Attention Guidance is not implemented for the LTX-2.3 video model")
_add("ltx2", "controlnets", "ControlNet is not supported for the LTX-2.3 video model")

# ACE-Step 1.5 is an audio model (own DiT + flow-matching turbo sampler, driven
# through /generate/txt2aud); none of the image-oriented guidance/conditioning
# features apply. Image endpoints reject an ACE-Step model outright (see
# _reject_if_audio_model), so these entries are defensive/documentation only.
_add("acestep", "advanced_cfg",
     "CFG scheduling / dynamic thresholding / CFG-rescale run only in the U-Net sampling loop, not in the ACE-Step turbo sampler")
_add("acestep", "nag", "Normalized Attention Guidance is not implemented for the ACE-Step audio model")
_add("acestep", "controlnets", "ControlNet is not supported for the ACE-Step audio model")

# U-Net/transformer quantization (per-generation unet_quantization parameter):
# not consumed by these architectures' pipeline backends. sd15/sdxl consume it
# via move_unet_to_gpu(); zimage/flux2/anima/lens consume it via their own
# transformer-quantization codepaths. FLUX.2 is absent from this table on
# purpose: it honours BOTH axes of the parameter -- the FP8 values through
# move_flux2_transformer_to_gpu, and "int8" through the in-place runtime
# conversion (Flux2Mixin._flux2_runtime_int8, RUNTIME_INT8_ARCHS) -- so there is
# nothing to warn about for any value.
_add("krea2", "unet_quantization",
     "FP8 quantization on this architecture is selected by checkpoint format at load time (bf16 or weight-only FP8 checkpoints); the only per-generation unet_quantization value applied is 'int8', which converts an unquantized transformer in place once per model load")
_add_supported_values("krea2", "unet_quantization", ["int8"])
_add("ideogram4", "unet_quantization",
     "FP8/nf4 quantization on this architecture is selected by checkpoint format at load time (FP8 or nf4 quantized checkpoints); the only per-generation unet_quantization value applied is 'int8', which converts an unquantized model's BOTH transformers in place once per model load")
_add_supported_values("ideogram4", "unet_quantization", ["int8"])
_add("minit2i", "unet_quantization",
     "unet_quantization is not implemented for this architecture")
_add("ltx2", "unet_quantization",
     "the FP8/nf4 values are not implemented for the LTX-2.3 video model; the only "
     "per-generation unet_quantization value applied is 'int8', which converts the "
     "unquantized video DiT (and only the DiT -- not the Gemma-3 text encoder or the text "
     "connectors) in place once per model load")
_add_supported_values("ltx2", "unet_quantization", ["int8"])
_add("acestep", "unet_quantization",
     "the FP8/nf4 values are not implemented for the ACE-Step audio model; the only "
     "per-generation unet_quantization value applied is 'int8', which converts the "
     "unquantized audio DiT (and only the DiT -- not the Oobleck VAE, which holds no "
     "2-D Linear weight at all, and not the Qwen3-Embedding text encoder) in place once "
     "per model load")
_add_supported_values("acestep", "unet_quantization", ["int8"])

# Quantized GEMM path (per-generation quantized_gemm_mode): only the
# architectures whose loaders swap in the weight-only quantized Linear classes
# (Fp8Linear / Int8Linear) have any GEMM to select -- Ideogram 4 (FP8/nf4),
# Krea 2 (FP8 or INT8), Anima (INT8) and FLUX.2 (INT8). Every other architecture
# stores plain floating-point Linear weights, so the two process flags govern
# nothing there.
# This table is what the generation panels read (via
# GET /schema/arch-capabilities) to decide whether to show the control at all,
# so the arch set is declared HERE and nowhere else.
#
# NOTE this is a different axis from `unet_quantization`, which quantizes an
# unquantized model's weights at load time to reduce VRAM. The two must not be
# merged: krea2/ideogram4 are listed as NOT applying `unet_quantization` just
# above, while they are exactly the archs that DO consume `quantized_gemm`.
#
# DERIVED from QUANTIZED_LINEAR_ARCHS rather than written out, because the set
# grows: FLUX.2 left this list the moment its loader gained the
# Int8Linear/Fp8Linear swap and it joined RUNTIME_INT8_ARCHS, and the next arch
# to be wired must not need an edit here to stop lying to the panels.
_ALL_ARCHS = ["sd15", "sdxl"] + _DIT_ARCHS
_QUANTIZED_GEMM_SUPPORTED = set(QUANTIZED_LINEAR_ARCHS)
for _a in [a for a in _ALL_ARCHS if a not in _QUANTIZED_GEMM_SUPPORTED]:
    _add(_a, "quantized_gemm",
         "this architecture's checkpoints hold plain floating-point Linear weights; the "
         "quantized-GEMM path selection applies only to the weight-only quantized Linear "
         f"layers used by {arch_names(QUANTIZED_LINEAR_ARCHS)}")

# Text-encoder quantization: not applied on these architectures' text-encoder paths.
for _a in ["sd15", "sdxl", "ideogram4", "minit2i", "krea2", "ltx2", "acestep"]:
    _add(_a, "text_encoder_quantization",
         "text-encoder quantization is not applied on this architecture's text-encoder path")

# CPU text encoding: not honored by these architectures' encode paths.
for _a in ["zimage", "flux2", "ideogram4", "minit2i", "krea2", "ltx2", "acestep"]:
    _add(_a, "cpu_text_encoding",
         "CPU text encoding is not honored by this architecture's encode path")

# attention_impl (generation side): only the FLUX.2 inference path consumes it;
# every other arch is conduit-only or ignores the selector.
# "deus" is intentionally omitted: model_loader never assigns arch type "deus".
for _a in ["sd15", "sdxl", "zimage", "ideogram4", "lens", "minit2i", "anima", "krea2", "ltx2", "acestep"]:
    _add(_a, "attention_impl",
         "attention_impl is only consumed by the FLUX.2 inference path; this architecture is conduit-only or ignores it")

# VAE DC-drift correction (img2img/inpaint): the reference round-trip decode is
# implemented in the SD1.5/SDXL custom img2img/inpaint sampling loops. The DiT
# archs use bespoke, PIL-returning decode funnels with arch-specific latent
# normalization, so the correction is accepted but not applied there.
for _a in _DIT_ARCHS:
    _add(_a, "vae_drift_correction",
         "VAE DC-drift correction is only implemented for the SD1.5/SDXL img2img/inpaint decode path")

# In-loop hard-flatten: the decode -> flat-region hard-replace -> encode -> latent
# injection is implemented in the SD1.5/SDXL custom sampling loops (Euler-validated
# x0 injection). The DiT archs use bespoke flow-matching samplers and PIL-returning
# decode funnels, so the flag is accepted but not applied there.
for _a in _DIT_ARCHS:
    _add(_a, "flatten_in_loop",
         "in-loop hard background flatten is only implemented for the SD1.5/SDXL sampling loops")

# TE override: only sound on SD1.5/SDXL, where either a custom-TE checkpoint's
# trained bridge adapters absorb the swap, or a stock CLIP encoder is substituted
# for a matching-hidden CLIP. Every DiT arch feeds its text encoder into
# arch-specific fusion / stacked-layer connectors trained for that exact geometry,
# with no adapter to absorb a raw TE-file swap.
for _a in _DIT_ARCHS:
    _add(_a, "te_override",
         "text-encoder override is only supported on SD1.5/SDXL; this architecture's text encoder feeds arch-specific fusion trained for that exact geometry")

# VAE override: unsupported on LTX-2.3 (a component swap invalidates the cpu-offload
# hook chain and there is no compatible 5D VAE), on MiniT2I (pixel-space, no VAE),
# and on ACE-Step (audio Oobleck VAE, not an image/video component override target).
_add("ltx2", "vae_override",
     "VAE override is not supported on the LTX-2.3 video model: a component swap invalidates the cpu-offload hook chain and there is no compatible 5D VAE")
_add("minit2i", "vae_override",
     "VAE override is not supported on this pixel-space architecture, which has no VAE")
_add("acestep", "vae_override",
     "VAE override is not supported on the ACE-Step audio model: its Oobleck VAE is audio-specific and not a per-generation image/video override target")

# ---------------------------------------------------------------------------
# MiniMax-H3 (joint video + audio DiT, driven through /generate/txt2vid).
# ---------------------------------------------------------------------------
# THE GUIDANCE PAIR. MiniMax-H3 is guidance-distilled: there is no guider, no
# unconditional branch and no guidance scale, so both the scale and the negative
# prompt are structurally absent rather than merely unused.
#
# WARNING CONTRACT (deliberate, not a limitation dodge): `_is_user_set` compares
# against the RESOLVED defaults, so an explicit `guidance_scale: 1.0` -- the
# default -- does not warn, while `guidance_scale: 5.0` does. The frontend
# always sends the full parameter object with defaults filled in, so a
# presence-based warning would fire on every UI-originated generation and mean
# nothing. The openapi descriptions state this contract.
_add("minimax_h3", "cfg",
     "guidance is distilled into the MiniMax-H3 weights: the sampler takes no guidance scale and runs exactly one forward pass per step")
_add("minimax_h3", "negative_prompt",
     "MiniMax-H3 is guidance-distilled and has no unconditional branch, so there is nothing for a negative prompt to steer away from")
_add("minimax_h3", "advanced_cfg",
     "CFG scheduling / dynamic thresholding / CFG-rescale run only in the U-Net sampling loop, and MiniMax-H3 has no guidance to schedule at all")
_add("minimax_h3", "nag",
     "Normalized Attention Guidance is not implemented for the MiniMax-H3 video model")
_add("minimax_h3", "controlnets",
     "ControlNet is not supported for the MiniMax-H3 video model")
# Quantization. The reason the generic `quantized_gemm` loop above would give is
# WRONG for this arch -- its released DiT is fp8-quantized -- so it is restated
# here with the real reason and overwrites the loop's text.
_add("minimax_h3", "quantized_gemm",
     "the released MiniMax-H3 DiT is weight-only FP8, but its scale sidecars are per-tensor scalars and 50 of its 200 quantized Linear layers are marked full_precision_matrix_mult, so every layer of this architecture is pinned to the dequantized path and there is no GEMM to select")
_add("minimax_h3", "unet_quantization",
     "the released MiniMax-H3 DiT already ships weight-only FP8-quantized, so there is no unquantized transformer for the per-generation converter to convert")
_add("minimax_h3", "text_encoder_quantization",
     "text-encoder quantization is not applied on this architecture's text-encoder path; its Qwen3-VL conditioner is streamed layer by layer from the memory-mapped bf16 file instead")
_add("minimax_h3", "cpu_text_encoding",
     "CPU text encoding is not honored by this architecture's encode path, which streams each decoder layer to the GPU and keeps the CPU weights memory-mapped")
_add("minimax_h3", "attention_impl",
     "attention_impl is only consumed by the FLUX.2 inference path; this architecture is conduit-only or ignores it")
_add("minimax_h3", "vae_override",
     "VAE override is not supported on MiniMax-H3: it owns two autoencoders (a 24-channel causal video VAE and a separate 32-channel audio VAE), its video VAE takes ImageNet-normalised RGB rather than [-1, 1], and its tiling policy is pinned because changing it changes the output")

# Training methods MiniMax-H3 does not offer. Enforced three ways: this table,
# the absence of a full-parameter adapter for the arch, and a hard refusal in
# the full_finetune trainer branch.
_add_training_unsupported(
    "minimax_h3", "full_finetune",
    "MiniMax-H3's DiT is a 33 B dense transformer; its parameters, gradients and optimizer state do not fit the single-GPU 48 GB envelope this integration targets, so only LoRA training is implemented")


def video_constraints_payload() -> Dict[str, Dict[str, Any]]:
    """The `video_constraints` block of GET /schema/arch-capabilities.

    Serialises each video architecture's ``TemporalSpec`` (the same table route
    validation snaps against) so a client can build a valid clip-length list
    from the backend's own rule instead of hardcoding one. Non-video
    architectures are absent from the map rather than present with nulls.
    """
    from core.models.components.wiring import TEMPORAL_SPECS

    payload: Dict[str, Dict[str, Any]] = {}
    for arch, spec in TEMPORAL_SPECS.items():
        payload[arch] = {
            "frame_multiple": spec.frame_multiple,
            "frame_offset": spec.frame_offset,
            "min_frames": spec.min_frames,
            "max_frames": spec.max_frames,
            "min_decodable_frames": spec.min_decodable_frames,
            "fps_fixed": spec.fps_fixed,
            "max_pixel_hw": list(spec.max_pixel_hw) if spec.max_pixel_hw else None,
            "pixel_align": spec.pixel_align,
            "suggested_frames": spec.suggested_lengths(),
        }
    return payload


def arch_supports_feature(arch: Optional[str], feature: str,
                          value: Any = None) -> bool:
    """True when ``arch`` honors ``feature`` (i.e. it is NOT in the unsupported
    table). An unknown/None arch is treated as supporting the feature so the
    override path is not silently dropped for an unrecognized model.

    ``value`` — when given, a value listed in ``ARCH_SUPPORTED_VALUES`` counts as
    supported even though the feature is otherwise unsupported on that arch
    (e.g. ``unet_quantization="int8"`` on Krea 2)."""
    if not arch:
        return True
    if feature not in ARCH_UNSUPPORTED.get(arch, {}):
        return True
    if value is None:
        return False
    return value in ARCH_SUPPORTED_VALUES.get(arch, {}).get(feature, [])


def _is_user_set(params: Dict[str, Any], key: str,
                 defaults: Optional[Dict[str, Any]] = None) -> bool:
    """True when the user set ``key`` to a non-default value.

    ``defaults`` overrides ``GENERATION_DEFAULTS`` for the keys it carries, and
    is how a non-image route supplies the defaults its own parameters were
    resolved against. Without it a video-only key (``num_frames``,
    ``frame_rate``, ``guidance_scale`` at the VIDEO default, ...) is compared
    against an IMAGE default it has nothing to do with -- usually absent
    entirely, which makes the default ``None`` and every value "user-set".
    """
    if defaults is not None and key in defaults:
        default = defaults.get(key)
    else:
        default = GENERATION_DEFAULTS.get(key, None)
    val = params.get(key, default)
    if isinstance(val, (list, tuple)):
        # Non-empty list (e.g. controlnets) counts as user-set.
        return bool(val)
    return val is not None and val != default


def check_arch_capabilities(params: Dict[str, Any], arch: str,
                            defaults: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Warn for each feature the loaded architecture ignores but the user set.

    Emits at most one warning per feature via ``add_warning`` and returns the
    list of warning dicts (for testing). Best-effort: never raises.

    ``defaults`` is the parameter-default map this request's values were
    resolved against -- the video routes pass the RESOLVED per-arch video
    defaults, so "non-default" means non-default for the loaded architecture.
    Omitted (the image routes) it falls back to ``GENERATION_DEFAULTS``, which
    is the historical behaviour.
    """
    emitted: List[Dict[str, Any]] = []
    if not arch:
        return emitted
    unsupported = ARCH_UNSUPPORTED.get(arch)
    if not unsupported:
        return emitted

    try:
        from api.generation_status import add_warning
    except ImportError:
        add_warning = None

    exempt = ARCH_SUPPORTED_VALUES.get(arch, {})
    for feature, reason in unsupported.items():
        trigger_keys = FEATURE_PARAMS.get(feature, [feature])
        if not any(_is_user_set(params, k, defaults) for k in trigger_keys):
            continue
        # A value this arch DOES honor (e.g. unet_quantization="int8" on Krea 2)
        # is not a reason to warn, even though other values of the same
        # parameter are ignored here.
        honored = exempt.get(feature)
        if honored and all(
            (not _is_user_set(params, k, defaults)) or params.get(k) in honored
            for k in trigger_keys
        ):
            continue
        label = FEATURE_LABELS.get(feature, feature)
        message = f"{label} is not supported on '{arch}' and was ignored: {reason}"
        warning = {"code": "unsupported_param", "message": message}
        emitted.append(warning)
        if add_warning is not None:
            add_warning(message, code="unsupported_param")
    return emitted
