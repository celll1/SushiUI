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
    "text_encoder_quantization": ["text_encoder_quantization"],
    "cpu_text_encoding": ["cpu_text_encoding"],
    "attention_impl": ["attention_impl"],
    "vae_drift_correction": ["vae_drift_correction"],
    "flatten_in_loop": ["flatten_in_loop"],
    "te_override": ["text_encoder_path"],
    "vae_override": ["vae_path"],
}

# Human-readable label used in the warning message for each feature.
FEATURE_LABELS: Dict[str, str] = {
    "use_torch_compile": "use_torch_compile",
    "advanced_cfg": "advanced CFG (cfg_schedule_*/dynamic_threshold_*/cfg_rescale_snr_alpha)",
    "spectrum": "spectrum_* (Spectral Feature Forecasting)",
    "fbcache": "fbcache_* (First Block Cache)",
    "nag": "nag_* (Normalized Attention Guidance)",
    "controlnets": "controlnets",
    "text_encoder_quantization": "text_encoder_quantization",
    "cpu_text_encoding": "cpu_text_encoding",
    "attention_impl": "attention_impl",
    "vae_drift_correction": "vae_drift_correction (VAE DC-drift correction)",
    "flatten_in_loop": "flatten_in_loop (in-loop hard background flatten)",
    "te_override": "text_encoder_path (text-encoder override)",
    "vae_override": "vae_path (VAE override)",
}

# ---------------------------------------------------------------------------
# ARCH_UNSUPPORTED[arch][feature] = short factual reason the feature has no
# effect on that architecture.
# ---------------------------------------------------------------------------
_DIT_ARCHS = ["zimage", "flux2", "ideogram4", "lens", "minit2i", "anima", "krea2", "ltx2"]
_SPECTRUM_UNSUPPORTED = ["zimage", "ideogram4", "lens", "minit2i", "anima", "krea2", "ltx2"]

ARCH_UNSUPPORTED: Dict[str, Dict[str, str]] = {}


def _add(arch: str, feature: str, reason: str) -> None:
    ARCH_UNSUPPORTED.setdefault(arch, {})[feature] = reason


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

# Text-encoder quantization: not applied on these architectures' text-encoder paths.
for _a in ["sd15", "sdxl", "ideogram4", "minit2i", "krea2", "ltx2"]:
    _add(_a, "text_encoder_quantization",
         "text-encoder quantization is not applied on this architecture's text-encoder path")

# CPU text encoding: not honored by these architectures' encode paths.
for _a in ["zimage", "flux2", "ideogram4", "minit2i", "krea2", "ltx2"]:
    _add(_a, "cpu_text_encoding",
         "CPU text encoding is not honored by this architecture's encode path")

# attention_impl (generation side): only the FLUX.2 inference path consumes it;
# every other arch is conduit-only or ignores the selector.
# "deus" is intentionally omitted: model_loader never assigns arch type "deus".
for _a in ["sd15", "sdxl", "zimage", "ideogram4", "lens", "minit2i", "anima", "krea2", "ltx2"]:
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
# hook chain and there is no compatible 5D VAE) and on MiniT2I (pixel-space, no VAE).
_add("ltx2", "vae_override",
     "VAE override is not supported on the LTX-2.3 video model: a component swap invalidates the cpu-offload hook chain and there is no compatible 5D VAE")
_add("minit2i", "vae_override",
     "VAE override is not supported on this pixel-space architecture, which has no VAE")


def arch_supports_feature(arch: Optional[str], feature: str) -> bool:
    """True when ``arch`` honors ``feature`` (i.e. it is NOT in the unsupported
    table). An unknown/None arch is treated as supporting the feature so the
    override path is not silently dropped for an unrecognized model."""
    if not arch:
        return True
    return feature not in ARCH_UNSUPPORTED.get(arch, {})


def _is_user_set(params: Dict[str, Any], key: str) -> bool:
    """True when the user set ``key`` to a non-default value."""
    default = GENERATION_DEFAULTS.get(key, None)
    val = params.get(key, default)
    if isinstance(val, (list, tuple)):
        # Non-empty list (e.g. controlnets) counts as user-set.
        return bool(val)
    return val is not None and val != default


def check_arch_capabilities(params: Dict[str, Any], arch: str) -> List[Dict[str, Any]]:
    """Warn for each feature the loaded architecture ignores but the user set.

    Emits at most one warning per feature via ``add_warning`` and returns the
    list of warning dicts (for testing). Best-effort: never raises.
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

    for feature, reason in unsupported.items():
        trigger_keys = FEATURE_PARAMS.get(feature, [feature])
        if not any(_is_user_set(params, k) for k in trigger_keys):
            continue
        label = FEATURE_LABELS.get(feature, feature)
        message = f"{label} is not supported on '{arch}' and was ignored: {reason}"
        warning = {"code": "unsupported_param", "message": message}
        emitted.append(warning)
        if add_warning is not None:
            add_warning(message, code="unsupported_param")
    return emitted
