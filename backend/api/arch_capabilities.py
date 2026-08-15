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
from typing import Any, Dict, List, Optional, Tuple

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
    # Which attention BACKEND runs the kernel (native/flash/sage/tq). Every
    # image architecture honors it, and so does MiniMax-H3 (its vendored
    # transformer routes attention through the unified conduit); an architecture
    # that drives diffusers' own attention dispatch instead declares it below.
    "attention_type": ["attention_type"],
    "vae_drift_correction": ["vae_drift_correction"],
    "flatten_in_loop": ["flatten_in_loop"],
    "te_override": ["text_encoder_path"],
    "vae_override": ["vae_path"],
    # Guidance. Split from `advanced_cfg` (which is the U-Net scheduling block)
    # because a guidance-DISTILLED architecture ignores the guidance scale
    # itself, not just its schedule.
    "cfg": ["guidance_scale", "cfg_scale"],
    "negative_prompt": ["negative_prompt"],
    # POST /generate/img2vid's optional SECOND keyframe (the last frame).
    # Meaningful only on an architecture that conditions on both ends of the
    # clip; the value carried in `params` is the uploaded filename.
    "last_frame_image": ["last_frame_image"],
    # POST /generate/img2vid's keyframe PLACEMENT fields: which frame the
    # uploaded image anchors, and any additional anchors with their own frames.
    # Separate from `last_frame_image` because they are a different claim -- one
    # says "there is a second anchor at the end", the other says "an anchor can
    # name any frame" -- and an architecture can honor the first without the
    # second. The frontend gates its keyframe timeline on this key.
    "keyframe_placement": ["input_image_frame_index", "keyframe_images",
                           "keyframe_frame_indices"],
    # POST /generate/img2vid's `input_audio`: an uploaded track the video is
    # generated AGAINST, pinned clean across the whole clip. A separate claim
    # again -- an architecture can place image keyframes without being able to
    # condition on audio at all -- and the frontend gates its audio lane on it.
    # The value carried in `params` is the uploaded filename.
    "audio_conditioning": ["input_audio"],
    # POST /generate/inpaint/video: regenerate one time range of a clip and
    # preserve the rest. A different claim again from `keyframe_placement` -- an
    # anchor is one conditioning frame outside the clip, this pins frames OF the
    # clip at their own positions -- and the frontend gates its video inpaint
    # surface on it.
    "temporal_inpaint": ["regenerate_start_frame", "regenerate_end_frame",
                         "inpaint_video_audio_mode"],
    # Generation-time LoRA on the video routes (txt2vid/img2vid/ref2vid/
    # outpaint/video/inpaint/video). Every image/audio architecture has its own
    # LoRA loader and is never listed here; this key exists for the video archs,
    # where only MiniMax-H3 has one.
    "lora": ["loras"],
    # Output-tail head fusion (AP3). MiniMax-H3 only -- see
    # `core.models.minimax_h3.adaln_chunking`'s "Head fusion" note.
    "fuse_output_proj": ["fuse_output_proj"],
    # Reference audio conditioning the autoregressive stage (voice/timbre/
    # instrument), i.e. an aud2aud "cover" request. The AUDIO_GEN_DEFAULTS
    # stub keys below predate any route that sends them (aud2aud instead
    # takes an uploaded reference clip, not a JSON path/flag), so this is
    # defensive/documentation only today -- same status as the ACE-Step
    # entries just below, which are unreachable because their routes reject
    # ACE-Step on the image endpoints outright.
    "audio_reference_conditioning": ["reference_audio_path", "reference_audio_enable", "is_cover"],
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
    "attention_type": "attention_type (attention backend)",
    "vae_drift_correction": "vae_drift_correction (VAE DC-drift correction)",
    "flatten_in_loop": "flatten_in_loop (in-loop hard background flatten)",
    "te_override": "text_encoder_path (text-encoder override)",
    "vae_override": "vae_path (VAE override)",
    "cfg": "guidance_scale/cfg_scale (classifier-free guidance)",
    "negative_prompt": "negative_prompt",
    "last_frame_image": "last_frame_image (last-frame keyframe)",
    "keyframe_placement": "input_image_frame_index/keyframe_images/keyframe_frame_indices (keyframe placement)",
    "audio_conditioning": "input_audio (audio-conditioned video)",
    "temporal_inpaint": "regenerate_start_frame/regenerate_end_frame (temporal inpaint)",
    "lora": "loras (LoRA)",
    "fuse_output_proj": "fuse_output_proj (output-tail head fusion)",
    "audio_reference_conditioning": "reference_audio_path/reference_audio_enable/is_cover (reference-audio conditioning)",
}

# ---------------------------------------------------------------------------
# ARCH_UNSUPPORTED[arch][feature] = short factual reason the feature has no
# effect on that architecture.
# ---------------------------------------------------------------------------
_DIT_ARCHS = ["zimage", "flux2", "ideogram4", "lens", "minit2i", "anima", "krea2", "ltx2", "acestep",
              "minimax_h3", "minimax_music3"]
# Both Spectrum and FBCache are wired for every image DiT arch through the
# same shared pattern (spectrum_params=params -> build_output_forecaster() /
# fbcache_active()+build_fbcache() inside each arch's *_pipeline_ops.py denoise
# loop): zimage, flux2, ideogram4, lens, minit2i, anima all genuinely consume
# spectrum_enable/fbcache_enable. ltx2 was wired in 444ebde5
# (_ltx2_build_spectrum / _ltx2_build_fbcache in
# core/pipeline_backends/ltx2.py). Only krea2, acestep and minimax_music3
# have no such codepath at all. MiniMax-H3 implements paired video/audio
# final-output forecasting and guarded whole-state FBCache.
_SPECTRUM_UNSUPPORTED = ["krea2", "acestep", "minimax_music3"]
_FBCACHE_UNSUPPORTED = ["krea2", "acestep", "minimax_music3"]

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

# Spectrum forecasting: implemented for the U-Net and every image/video DiT
# except krea2 and acestep (see _SPECTRUM_UNSUPPORTED above).
for _a in _SPECTRUM_UNSUPPORTED:
    _add(_a, "spectrum",
         "Spectral Feature Forecasting is not implemented for this architecture's sampler")

# First Block Cache remains unavailable on krea2/acestep.
for _a in _FBCACHE_UNSUPPORTED:
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
# The img2vid endpoint's optional SECOND keyframe. LTX-2.3's image-to-video
# pipeline pins the uploaded image as frame 0 and takes no last-frame condition,
# so a last frame sent here is accepted and dropped rather than refused (the
# endpoint serves two architectures and only one of them reads the field).
_add("ltx2", "last_frame_image",
     "LTX-2.3's image-to-video pipeline conditions on the first frame only, so a last-frame keyframe has nothing to attach to")
# Keyframe PLACEMENT on the same endpoint. LTX-2.3's img2vid pipeline pins the
# uploaded image as frame 0 and takes no frame index for it, so a placement is
# accepted and dropped rather than refused -- one endpoint, two architectures.
# (Its `LTX2VideoCondition.index` DOES address a latent frame, which is what
# /generate/outpaint/video's "free" placement uses; the img2vid pipeline simply
# does not expose it.)
_add("ltx2", "keyframe_placement",
     "LTX-2.3's image-to-video pipeline pins the uploaded image as frame 0 and takes no per-keyframe frame index, so a keyframe placement has nothing to apply to")
# Audio CONDITIONING on the same endpoint. LTX-2.3 generates a soundtrack
# jointly with the video, but its pipeline exposes no way to supply one: there
# is no audio conditioning input to pin an uploaded track to.
# Temporal inpaint (POST /generate/inpaint/video). Not a limitation discovered
# at the route: it is a permutation of MiniMax-H3's packed video rows, and
# LTX-2.3's conditions carry whole clips at latent indices rather than pinning
# frames of the target itself, so there is nothing here to implement it with.
_add("ltx2", "temporal_inpaint",
     "temporal inpaint is not implemented for LTX-2.3: it pins the kept frames' own latents inside one packed sequence, which is a MiniMax-H3 mechanism")
_add("ltx2", "audio_conditioning",
     "LTX-2.3 generates its soundtrack jointly with the video and its image-to-video pipeline takes no audio conditioning input, so an uploaded track has nothing to attach to")
# The video routes carry `attention_type` because MiniMax-H3 honors it; LTX-2.3
# runs the diffusers transformer's own attention dispatch and never consults it.
_add("ltx2", "attention_type",
     "LTX-2.3 runs diffusers' own attention dispatch rather than SushiUI's attention conduit, so the attention backend is not selectable per generation for this architecture")
# Generation-time LoRA: MiniMax-H3 has a loader (core.models.minimax_h3.minimax_h3_lora);
# LTX-2.3's video path has none at all.
_add("ltx2", "lora",
     "LoRA loading is not implemented for the LTX-2.3 video model's generation path")
_add("ltx2", "fuse_output_proj",
     "output-tail head fusion is a MiniMax-H3-specific chunking optimization (core.models.minimax_h3.adaln_chunking); LTX-2.3 has no equivalent output-head structure")

# ACE-Step 1.5 is an audio model (own DiT + flow-matching turbo sampler, driven
# through /generate/txt2aud); none of the image-oriented guidance/conditioning
# features apply. Image endpoints reject an ACE-Step model outright (see
# _reject_if_audio_model), so these entries are defensive/documentation only.
_add("acestep", "advanced_cfg",
     "CFG scheduling / dynamic thresholding / CFG-rescale run only in the U-Net sampling loop, not in the ACE-Step turbo sampler")
_add("acestep", "nag", "Normalized Attention Guidance is not implemented for the ACE-Step audio model")
_add("acestep", "controlnets", "ControlNet is not supported for the ACE-Step audio model")

# ---------------------------------------------------------------------------
# MiniMax Music 3 (lyrics- and caption-conditioned music generation, driven
# through /generate/txt2aud alongside ACE-Step; see
# docs/guides/MINIMAX_MUSIC3_DESIGN.md "Capability verdict"). Image endpoints
# reject a MiniMax Music 3 model outright (see _reject_if_audio_model).
#
# REACHABILITY, entry by entry (audit finding F6): `check_arch_capabilities`
# only warns when one of a feature's TRIGGER PARAMS (`FEATURE_PARAMS`) is a
# key `params` actually carries with a non-default value, and the only route
# reachable for this architecture is `/generate/txt2aud`
# (`Txt2AudRequest`'s declared fields) -- so an entry's status depends on
# whether that model declares the trigger key at all:
#   - `advanced_cfg`/`nag`/`controlnets` are UNREACHABLE today:
#     `Txt2AudRequest` has no `cfg_schedule_type`/`nag_enable`/`controlnets`
#     field (same status as ACE-Step's identical three entries just above).
#     Kept for documentation and so the warning fires the moment any future
#     shared-audio-endpoint change adds one of those fields.
#   - `lora` IS REACHABLE: `Txt2AudRequest.loras` is a real, live field this
#     architecture's backend
#     (`core.pipeline_backends.minimax_music3.MiniMaxMusic3Mixin.
#     _generate_txt2aud_minimax_music3`) never reads at all -- so a request
#     that selects a LoRA got a clean 200 with an empty `warnings[]` before
#     this entry existed (audit finding F2), which reads as "the LoRA had no
#     audible effect" rather than "it was never loaded".
#   - `negative_prompt`/`audio_reference_conditioning` are UNREACHABLE on
#     `/generate/txt2aud` today (neither is a `Txt2AudRequest` field; the
#     latter's real surface, an aud2aud "cover" request, IS reachable now
#     (design doc phase plan item 8) but is refused for this architecture at
#     the mechanism layer, inside `MiniMaxMusic3Mixin._generate_aud2aud_
#     minimax_music3`, with the RVQ-tokenizer-encoder capability reason --
#     not by a route-level gate). Both are properties of the RELEASED MODEL,
#     not unimplemented features -- design doc "Capability verdict", first
#     three rows -- kept for documentation and ready the moment either
#     surface exists.
# ---------------------------------------------------------------------------
_add("minimax_music3", "advanced_cfg",
     "CFG scheduling / dynamic thresholding / CFG-rescale run only in the U-Net sampling loop, not in MiniMax Music 3's autoregressive + flow-matching samplers")
_add("minimax_music3", "nag",
     "Normalized Attention Guidance is not implemented for MiniMax Music 3")
_add("minimax_music3", "controlnets",
     "ControlNet is not supported for MiniMax Music 3")
_add("minimax_music3", "lora",
     "generation-time LoRA is not implemented for MiniMax Music 3's pipeline backend (core.pipeline_backends.minimax_music3.MiniMaxMusic3Mixin._generate_txt2aud_minimax_music3 never reads params['loras']); a LoRA selected for this generation has no effect")
_add("minimax_music3", "negative_prompt",
     "the flow-stage unconditional branch conditions on zeros and the autoregressive stage's unconditional branch is the token-masked prompt itself, so there is no negative prompt anywhere in this model")
_add("minimax_music3", "audio_reference_conditioning",
     "the RVQ tokenizer's encoder is not published in this release, so no audio can be turned into the semantic codes needed to condition the autoregressive stage, and the flow-stage DiT conditions on the language model's hidden states rather than on audio directly")

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
_add("minimax_music3", "unet_quantization",
     "no per-generation unet_quantization value is implemented for MiniMax Music 3; phase 1 "
     "loads the official/ tree's BF16/FP16 weights only -- INT8 ConvRot on the flat tree's "
     "staged artifacts is a later phase (docs/guides/MINIMAX_MUSIC3_DESIGN.md \"Quantization\")")

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
for _a in ["sd15", "sdxl", "ideogram4", "minit2i", "krea2", "ltx2", "acestep", "minimax_music3"]:
    _add(_a, "text_encoder_quantization",
         "text-encoder quantization is not applied on this architecture's text-encoder path")

# CPU text encoding: not honored by these architectures' encode paths.
for _a in ["zimage", "flux2", "ideogram4", "minit2i", "krea2", "ltx2", "acestep", "minimax_music3"]:
    _add(_a, "cpu_text_encoding",
         "CPU text encoding is not honored by this architecture's encode path")

# attention_impl (generation side): only the FLUX.2 inference path consumes it;
# every other arch is conduit-only or ignores the selector.
# "deus" is intentionally omitted: model_loader never assigns arch type "deus".
for _a in ["sd15", "sdxl", "zimage", "ideogram4", "lens", "minit2i", "anima", "krea2", "ltx2", "acestep",
           "minimax_music3"]:
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
# and on ACE-Step / MiniMax Music 3 (audio-specific autoencoders, not an image/video
# component override target).
_add("ltx2", "vae_override",
     "VAE override is not supported on the LTX-2.3 video model: a component swap invalidates the cpu-offload hook chain and there is no compatible 5D VAE")
_add("minit2i", "vae_override",
     "VAE override is not supported on this pixel-space architecture, which has no VAE")
_add("acestep", "vae_override",
     "VAE override is not supported on the ACE-Step audio model: its Oobleck VAE is audio-specific and not a per-generation image/video override target")
_add("minimax_music3", "vae_override",
     "VAE override is not supported on MiniMax Music 3: its vocoder is the decoder half of a music-specific autoencoder (the DAV), not a per-generation image/video override target")

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
# Quantization. NOTE WHAT IS *NOT* DECLARED HERE: `quantized_gemm`.
#
# `minimax_h3` is in `QUANTIZED_LINEAR_ARCHS` (its loader really does swap 200
# `nn.Linear` for `Fp8Linear`), and that tuple is what grants the capability, so
# an entry here would contradict it -- `quantized_capability_parity_test`
# enforces exactly that, because "the table says unsupported while the loader
# owns quantized Linears" is how an architecture's LoRA target predicate stops
# being checked against the isinstance(nn.Linear) trap.
#
# The honest statement about `quantized_gemm_mode` on this architecture is not
# "unsupported" but "accepted and always resolves to dequant": the loader pins
# all 300 of the DiT's Fp8Linear modules to the dequantized path with
# `disable_scaled_mm`, which outranks the request, so `"w8a8"` produces a
# `quantization_fallback` warning naming the resolved path
# (`report_quantized_gemm_outcome` reads the real label out of
# `extract_fp8_gemm_info`) rather than silence. The reason is recorded in
# `ARCH_QUANT_POLICY["minimax_h3"]` and in
# `models/minimax_h3/loader.py::_dit_quantization_policy`; both scale sidecars
# being per-tensor scalars and 50 of the file's 200 quantized tensors being
# marked `full_precision_matrix_mult` are what make it permanent for this file.
_add("minimax_h3", "unet_quantization",
     "the released MiniMax-H3 DiT already ships weight-only FP8-quantized, so there is no unquantized transformer for the per-generation converter to convert")
_add("minimax_h3", "text_encoder_quantization",
     "there is no per-generation converter for this architecture's text encoder; the co-distributed qwen3vl_32b_minimax_h3_int8_convrot.safetensors is a separate file the loader selects and loads pre-quantized (ConvRot INT8, comfy-kitchen), same as the bf16 file, streamed layer by layer from the memory-mapped weights either way")
_add("minimax_h3", "cpu_text_encoding",
     "CPU text encoding is not honored by this architecture's encode path, which streams each decoder layer to the GPU and keeps the CPU weights memory-mapped")
_add("minimax_h3", "attention_impl",
     "attention_impl is only consumed by the FLUX.2 inference path; this architecture is conduit-only or ignores it")
_add("minimax_h3", "vae_override",
     "VAE override is not supported on MiniMax-H3: it owns two autoencoders (a 24-channel causal video VAE and a separate 32-channel audio VAE), its video VAE takes ImageNet-normalised RGB rather than [-1, 1], and its tiling policy is pinned because changing it changes the output")

# Training methods MiniMax-H3 does not offer. Refused in THREE layers, all of
# them live:
#   1. this table -- a client filters its training-method dropdown from it, so a
#      full fine-tune is not offerable in the first place;
#   2. the deliberate absence of a `MiniMaxH3FullParameterAdapter` class in
#      `core/training/adapters/minimax_h3_adapter.py` (that module exports the
#      LoRA adapter only);
#   3. a hard `ValueError` in `full_parameter_trainer._create_adapter`, which
#      fires before any model is loaded if a run is queued anyway.
# LoRA training IS implemented (arch handler + ops + adapter, Phase 6b).
_add_training_unsupported(
    "minimax_h3", "full_finetune",
    "MiniMax-H3's DiT is a 33 B dense transformer; its parameters, gradients and optimizer state do not fit the single-GPU 48 GB envelope this integration targets, so only LoRA training is implemented")
_add_training_unsupported(
    "minimax_h3", "relora",
    "MiniMax-H3 ReLoRA is not implemented; the supported training bases use weight-only FP8 or packed W4A8 Linears, which cannot accept dense LoRA merges without format-specific requantization. Use LoRA instead")


# ---------------------------------------------------------------------------
# CHAIN_CONTEXT[arch] = what a long-form video CHAIN's continuation segments can
# receive from their predecessor on that architecture (design §7.1), plus a
# `variants` map for the facts that differ per loaded transformer variant.
#
# Served as `chain_context` by GET /schema/arch-capabilities and used by
# POST /video-chain/plan|validate to refuse an unadvertised mode with a 400
# (design §7.7). It is deliberately NOT part of `video_constraints`: that block
# is a verbatim serialisation of `TemporalSpec`, which has no variant dimension,
# while `chain_supports_reference_video` is true for MiniMax-H3's `ref2va`
# transformer and false for `fl2va` with the same `TemporalSpec`.
#
# `chain_context_min_frames`/`chain_context_max_frames` count PIXEL frames of
# the preceding segment that the advertised modes condition on, and they bound
# the `continuation_overlap_frames` a mode that PINS an overlap accepts. They do
# not describe `boundary_frame`: its shared anchor is a first-frame
# conditioning, not an overlap request, and it takes no length at all. A value
# is only meaningful on a video-VAE group boundary, i.e. it must be a cumulative
# sum of `video_constraints[arch].latent_chunk_pattern` -- for MiniMax-H3's
# (1, 4, 4, 4, 4) those are 1, 5, 9, 13, 17, 18, 22, ... (the pattern CYCLES).
# The pattern is already served next to this block and is the ONE enumerator;
# this module does not restate the list.

# The shortest `pinned_tail` overlap MiniMax-H3 serves. MEASURED (P-VC-1,
# design §7.2b): a 1-frame pin hands the model a motionless still as "observed
# video", and it can read that as a static scene -- the generated span froze
# outright on 1 seed in 4, and diverged systematically less than the anchor even
# when it did not (24.1 vs 31.0). The collapse is gone by 5 (32.1) and 17 (29.6).
# A 1-frame pin is therefore NOT a cheap equivalent of `boundary_frame`, which
# stays its own mode; asking for one is refused rather than snapped up.
MINIMAX_H3_PINNED_TAIL_MIN_FRAMES = 5

# The longest `pinned_tail` overlap MiniMax-H3 serves: one full cycle of its
# chunk pattern, and the top of the 5/17 comparison the mode exists to make.
# It is a REFUSAL bound, not a clamp -- a longer pin is unmeasured, and it also
# keeps every advertised overlap far below the shortest generated span (124), so
# a pin can never claim the whole clip.
MINIMAX_H3_PINNED_TAIL_MAX_FRAMES = 17

# --- `motion_preroll` (design §7.3) -----------------------------------------
# The pre-roll is REGENERATED and discarded, and the model reads it through
# sparse keyframe ANCHORS rather than through pinned rows, so none of the
# pinned-tail numbers above transfer to it and it carries its own bounds.
#
# The pre-roll needs no VAE-group alignment: an anchor addresses a pixel frame
# directly (`h3_pipeline_ops._anchor_rotary_time`: "There is no grid to snap an
# anchor to"), unlike a pin, which conditions a latent frame whole.
#
# FLOOR 2, and it is STRUCTURAL, not the measured 5 above: two anchors need two
# distinct frames (`generation_utils.plan_keyframe_placements` refuses two
# anchors on one frame). The 5-frame pin floor is a property of PINNING -- the
# same P-VC-1 run is what showed a 1-frame pin is not an anchor -- so importing
# it here would be borrowing a measurement made about the other mechanism.
MINIMAX_H3_MOTION_PREROLL_MIN_FRAMES = 2
# CEILING 17: a REFUSAL bound. It matches `pinned_tail`'s top so the two
# comparison arms can be run over the same context lengths, and it keeps the
# discarded pre-roll under 17 of the shortest generated span this arch has
# (`MINIMAX_H3_TEMPORAL.min_frames` = 124). Longer is unmeasured.
MINIMAX_H3_MOTION_PREROLL_MAX_FRAMES = 17
# Two anchors is the point of the mode: one anchor carries no direction, and a
# request for one is `boundary_frame` plus frames thrown away.
MINIMAX_H3_MOTION_PREROLL_MIN_ANCHORS = 2
# Four is where this integration stops. Each anchor reserves `rows_per_frame`
# conditioning rows carried on EVERY denoise step
# (`h3_pipeline_ops.build_packed_layout`: `num_condition_rows =
# len(keyframe_anchors) * rows_per_frame`), so the cost is linear in the count,
# and the model card documents at most two anchors -- `plan_keyframe_placements`
# already reports anything above 2 as an undocumented shape. More than 4 is
# refused rather than accepted untested.
MINIMAX_H3_MOTION_PREROLL_MAX_ANCHORS = 4

CHAIN_CONTEXT: Dict[str, Dict[str, Any]] = {
    # MiniMax-H3: a continuation is POST /generate/outpaint/video with
    # `extend_forward`, which hands the model the preserved clip's last frame as
    # the generated span's first-frame anchor and concatenates the preserved
    # frames back untouched (core/pipeline_backends/minimax_h3.py:866-876). That
    # is exactly one frame of visual context, and one frame is latent frame 0's
    # whole coverage, so it is VAE-aligned by construction.
    #
    # `pinned_tail` widens that to `continuation_overlap_frames` frames by
    # pinning the preserved tail as the generated clip's own leading latent
    # frames -- the temporal-inpaint mechanism, reused rather than rebuilt. It
    # rides the arch-level entry because the pin and a reference block claim the
    # same conditioning prefix, so it is fl2va's and not ref2va's. Its min/max
    # bound the pin, not the anchor: `boundary_frame` is a separate mode and
    # keeps its single anchor frame whatever these say.
    #
    # `motion_preroll` shares the same overlap arithmetic but conditions on it
    # differently: the overlap is regenerated with several of the predecessor's
    # frames placed on it as keyframe anchors, and then discarded. It rides the
    # arch-level entry for the same reason `pinned_tail` does -- an anchor and a
    # reference block claim the same conditioning prefix.
    "minimax_h3": {
        "chain_continuation_modes": ["boundary_frame", "pinned_tail", "motion_preroll"],
        "chain_context_min_frames": MINIMAX_H3_PINNED_TAIL_MIN_FRAMES,
        "chain_context_max_frames": MINIMAX_H3_PINNED_TAIL_MAX_FRAMES,
        # TRUE since the outpaint route gained `continuation_mode:
        # motion_preroll`, which places the anchors through the same
        # index-addressable keyframe conditioning /generate/img2vid uses
        # (`keyframe_placement` above). The mode is unmeasured and opt-in; this
        # flag says the placement EXISTS here, not that it is better.
        "chain_supports_sparse_motion_anchors": True,
        # The pre-roll's own bounds. Separate from `chain_context_min/max_frames`
        # because a pre-roll is not a pin: it needs no VAE-group alignment (any
        # integer in range is addressable) and its floor is structural rather
        # than measured. Null on an architecture/variant that does not advertise
        # the mode.
        "chain_motion_preroll_min_frames": MINIMAX_H3_MOTION_PREROLL_MIN_FRAMES,
        "chain_motion_preroll_max_frames": MINIMAX_H3_MOTION_PREROLL_MAX_FRAMES,
        "chain_motion_preroll_min_anchors": MINIMAX_H3_MOTION_PREROLL_MIN_ANCHORS,
        "chain_motion_preroll_max_anchors": MINIMAX_H3_MOTION_PREROLL_MAX_ANCHORS,
        "chain_supports_reference_video": False,
        "chain_supports_exact_prefix": True,
        "variants": {
            # ref2va only: the preserved clip's own trailing
            # min(preserved, generated_span) frames become an automatic video
            # reference on top of the boundary anchor
            # (`build_outpaint_references`, core/pipeline_backends/
            # minimax_h3.py:88-147; the route forces the 22-frame reference
            # floor). fl2va was never trained to read reference rows
            # (routes.py:4815-4822), so it stays on the arch-level entry.
            #
            # No `pinned_tail` here: NOT because the builder has no opening for
            # a pin -- `build_ref2va_packed_layout` gained `pinned_video_frames`
            # / `pinned_audio_latents` (h3_pipeline_ops.py, MiniMax-H3 inpaint x
            # reference design, Option B) and does return a permutation for
            # them now, AND `resolve_minimax_h3_inpaint_reference_gate`'s
            # `ref2va` row is OPEN as of phase B-3-open (opened at the repo
            # owner's instruction ahead of Gate registration (B)'s §6.2 GPU
            # arms, which have not run) -- so `/generate/inpaint/video` itself
            # no longer refuses this partition. What remains true, and what
            # actually keeps `pinned_tail` off this list, is that this
            # partition's interior-pin hold is UNMEASURED (fl2va's own pin is
            # measured: preserved-span RMS 3.12, VAE floor 3.15, control 75.69,
            # `minimax_h3_ti_probe_results.md`; ref2va's is not, and every
            # generation that reaches it carries a
            # `minimax_h3_undocumented_conditioning` warning saying so) AND no
            # chaining wiring for it exists on this outpaint continuation path
            # regardless -- the open endpoint is `/generate/inpaint/video`
            # directly, not this chain mode. No `motion_preroll` either, for
            # the same reason as before: this partition's continuation already
            # spends its conditioning prefix on the automatic tail reference
            # (`build_outpaint_references`), and a pre-roll's anchors on top
            # of it is a shape nothing has measured.
            "ref2va": {
                "chain_continuation_modes": ["boundary_frame"],
                "chain_context_min_frames": 1,
                "chain_context_max_frames": 1,
                "chain_supports_sparse_motion_anchors": False,
                "chain_motion_preroll_min_frames": None,
                "chain_motion_preroll_max_frames": None,
                "chain_motion_preroll_min_anchors": None,
                "chain_motion_preroll_max_anchors": None,
                "chain_supports_reference_video": True,
                "chain_supports_exact_prefix": True,
            },
            # hybrid: an fl2va base carrying ref2va AdaLN blocks. It generates
            # on /generate/txt2vid and nowhere else -- the A/B that released it
            # compared single prompt-only clips -- so nothing about chaining it
            # is measured. It needs an entry of its own
            # BECAUSE the fallback in `chain_context_for` is the arch-level
            # entry, which would advertise fl2va's `pinned_tail` and
            # `motion_preroll` to it. `boundary_frame` is the floor of what an
            # entry can say -- `chain_context_payload` refuses an entry that
            # advertises no implemented mode -- not a claim that a chain works.
            "hybrid": {
                "chain_continuation_modes": ["boundary_frame"],
                "chain_context_min_frames": 1,
                "chain_context_max_frames": 1,
                "chain_supports_sparse_motion_anchors": False,
                "chain_motion_preroll_min_frames": None,
                "chain_motion_preroll_max_frames": None,
                "chain_motion_preroll_min_anchors": None,
                "chain_motion_preroll_max_anchors": None,
                "chain_supports_reference_video": False,
                "chain_supports_exact_prefix": True,
            },
        },
    },
    # LTX-2.3: a continuation places the whole accumulated clip as one
    # `LTX2VideoCondition` (core/pipeline_backends/ltx2.py:1151) and pastes the
    # input back frame-exact afterwards (:1250). The chain's shared-anchor
    # arithmetic still subtracts one frame per segment, so `boundary_frame` is
    # the mode the manifest is planned under, but the model is conditioned on
    # the ENTIRE preserved prefix rather than on one frame -- which is what the
    # unbounded `chain_context_max_frames` records. It is not selectable: this
    # architecture has no way to hand it less.
    "ltx2": {
        "chain_continuation_modes": ["boundary_frame"],
        "chain_context_min_frames": 1,
        "chain_context_max_frames": None,
        "chain_supports_sparse_motion_anchors": False,
        "chain_motion_preroll_min_frames": None,
        "chain_motion_preroll_max_frames": None,
        "chain_motion_preroll_min_anchors": None,
        "chain_motion_preroll_max_anchors": None,
        "chain_supports_reference_video": False,
        "chain_supports_exact_prefix": True,
        "variants": {},
    },
}


def chain_context_payload() -> Dict[str, Dict[str, Any]]:
    """The `chain_context` block of GET /schema/arch-capabilities.

    `chain_default_continuation_mode` is filled in here rather than written into
    the table: the default lives in `VIDEO_CHAIN_DEFAULTS["continuation_mode"]`
    (the single source of truth for API defaults), and an architecture that does
    not advertise it falls back to its own first advertised mode.
    """
    from api.param_defaults import VIDEO_CHAIN_DEFAULTS
    # The set of modes that EXIST. An architecture may advertise a subset of it
    # and never a value outside it -- `openapi.yaml`'s `continuation_mode` enum
    # is the wider wire vocabulary (it names the Phase-B candidates so they can
    # be refused by name), and advertising one of those here would be promising
    # a mode no code implements.
    from core.inference.video_chain_context import CONTINUATION_MODES

    requested_default = VIDEO_CHAIN_DEFAULTS["continuation_mode"]

    def _entry(spec: Dict[str, Any]) -> Dict[str, Any]:
        modes = [m for m in spec["chain_continuation_modes"] if m in CONTINUATION_MODES]
        if not modes:
            raise RuntimeError(
                "chain_context advertises no implemented continuation mode; "
                f"CHAIN_CONTEXT lists {spec['chain_continuation_modes']}, implemented: {CONTINUATION_MODES}"
            )
        out = {k: v for k, v in spec.items() if k != "variants"}
        out["chain_continuation_modes"] = modes
        out["chain_default_continuation_mode"] = (
            requested_default if requested_default in modes else modes[0]
        )
        return out

    payload: Dict[str, Dict[str, Any]] = {}
    for arch, spec in CHAIN_CONTEXT.items():
        entry = _entry(spec)
        entry["variants"] = {
            name: _entry(vspec) for name, vspec in (spec.get("variants") or {}).items()
        }
        payload[arch] = entry
    return payload


def chain_context_for(arch: Optional[str],
                      variant: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The chain-context capability of ``arch`` (``variant`` where it differs).

    None for an architecture that cannot be chained at all. A variant with no
    entry of its own answers with the ARCHITECTURE-LEVEL entry -- which is the
    widest one this architecture advertises, not a conservative default. A
    variant that serves less than the architecture (MiniMax-H3's `ref2va` and
    `hybrid`) must therefore carry its own entry; leaving it out advertises
    capabilities it does not have.
    """
    entry = chain_context_payload().get(arch or "")
    if entry is None:
        return None
    key = (variant or "").strip().lower()
    return entry["variants"].get(key, entry)


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
            # ADVISORY top of the arch's DOCUMENTED trained range, distinct
            # from `max_frames` (the ENFORCED ceiling, which may be None --
            # no enforced top at all). MiniMax-H3: 362, `max_frames` is None
            # (RoPE is computed on the fly, so nothing structural stops a
            # longer clip); a request past this is accepted and warned as
            # untested, never refused or clamped. None when the arch has
            # nothing narrower than `max_frames` to document (LTX-2.3).
            "trained_max_frames": spec.trained_max_frames,
            "min_decodable_frames": spec.min_decodable_frames,
            "fps_fixed": spec.fps_fixed,
            # ORIENTATION-AGNOSTIC `[short_edge, long_edge]`, NOT `[height,
            # width]`: route validation compares min(w, h) against the first
            # entry and max(w, h) against the second, so a client that read it
            # as height/width would build wrong portrait options.
            "max_pixel_hw": list(spec.max_pixel_hw) if spec.max_pixel_hw else None,
            "pixel_align": spec.pixel_align,
            # What an off-grid / out-of-range clip length does. The two video
            # archs differ on exactly this (LTX-2.3 answers 400, MiniMax-H3
            # snaps up and warns), and it cannot be derived from the fields
            # above, so a client cannot get it right without this flag.
            "snap_invalid_length": spec.snap_invalid_length,
            # 16, not the default 8: a client builds its clip-length control
            # from this list, and 8 entries stopped LTX-2.3's list at 65 --
            # dropping 81/97/121, all valid `8k+1` lengths that were offered
            # before this payload existed (121 is LTX-2.3's own default). 16
            # covers that and is the whole of MiniMax-H3's DOCUMENTED range
            # (124..362, stopping at `trained_max_frames` even though
            # `max_frames` no longer bounds it, is 15 lengths), so neither
            # arch's list is truncated.
            "suggested_frames": spec.suggested_lengths(16),
            # Step-count contract, for a client building a step-count control.
            # Neither value is derivable from the fields above and the two
            # video archs differ on both: LTX-2.3 runs N evaluations for N
            # steps (minimum 1), MiniMax-H3 counts sigma grid points and runs
            # N-1 (minimum 2). Below the minimum the route answers 400.
            "min_inference_steps": spec.min_inference_steps,
            "steps_are_sigma_grid_points": spec.steps_are_sigma_grid_points,
            # Which temporal-outpaint placements the arch's CONDITIONING can
            # serve. A client builds its placement control from this instead of
            # hardcoding an arch check: "free" means any offset (LTX-2.3),
            # while MiniMax-H3 lists only the boundary placements it can
            # actually anchor.
            "outpaint_placements": list(spec.outpaint_placements),
            # The video VAE's temporal chunking: pixel frames per latent frame,
            # cycled. It is the addressable unit of POST /generate/inpaint/video
            # (a requested range is expanded outward to these boundaries), so a
            # client that offers a range control needs it to show the range the
            # server will actually run. Empty = the arch declares none.
            "latent_chunk_pattern": list(spec.latent_chunk_pattern),
        }
    return payload


# ---------------------------------------------------------------------------
# Audio temporal-outpaint placements (POST /generate/outpaint/audio, design
# doc phase plan item 7 "Extend"). Mirrors video's
# `TEMPORAL_SPECS[...].outpaint_placements` (served via
# `video_constraints_payload`), but audio has no per-architecture
# `TemporalSpec` registry of its own -- only two audio architectures exist,
# and ACE-Step's placement is a continuous `total_duration`/`input_offset_sec`
# timeline offset, not an enumerated set at all -- so this is declared
# directly here rather than growing a second wiring registry for one entry.
# ---------------------------------------------------------------------------
AUDIO_OUTPAINT_PLACEMENTS: Dict[str, Tuple[str, ...]] = {
    # MiniMax Music 3's autoregressive stage is a causal language model: it
    # can only continue a song forward from its existing end
    # (`core.pipeline_backends.minimax_music3.MiniMaxMusic3Mixin.
    # _generate_audoutpaint_minimax_music3`'s "Placement" docstring). Backward
    # extension and mid-song infill are refused as a property of the released
    # model, not an unimplemented feature -- design doc
    # (docs/guides/MINIMAX_MUSIC3_DESIGN.md) "Capability verdict": "Mid-song
    # infill with a preserved tail -- No -- the global LM is causal; there is
    # no infilling contract."
    "minimax_music3": ("extend_forward",),
    # ACE-Step has no entry: its placement is a continuous offset a client
    # picks freely inside `total_duration`, not a value from an enumerated
    # set, so a single-entry tuple here would misrepresent it as one.
}


def audio_outpaint_placements(arch: Optional[str]) -> Tuple[str, ...]:
    """Placements `/generate/outpaint/audio`'s mechanism can serve for `arch`.

    Empty for an architecture with no entry (ACE-Step: a continuous offset,
    not an enum) or an unrecognized/None arch -- a client sees "no enumerated
    placement" and falls back to whatever free-offset UI it already has.
    """
    return AUDIO_OUTPAINT_PLACEMENTS.get(arch or "", ())


# ---------------------------------------------------------------------------
# MiniMax Music 3 aud2aud "repaint" sub-modes (POST /generate/aud2aud with
# mode=repaint, design doc phase plan item 8). Same rationale as
# AUDIO_OUTPAINT_PLACEMENTS just above -- a client builds its repaint-mode
# control from this instead of hardcoding an arch check. ACE-Step has no
# entry: its own aud2aud has no music3_repaint_mode concept at all (it uses
# mode=cover/repaint directly, with no further sub-mode).
# ---------------------------------------------------------------------------
AUD2AUD_MUSIC3_REPAINT_MODES: Dict[str, Tuple[str, ...]] = {
    "minimax_music3": ("regenerate", "rerender"),
}


def aud2aud_music3_repaint_modes(arch: Optional[str]) -> Tuple[str, ...]:
    """`music3_repaint_mode` values `/generate/aud2aud`'s mechanism can serve for `arch` (repaint mode only).

    Empty for an architecture with no entry (ACE-Step: no such sub-mode) or an unrecognized/None arch.
    """
    return AUD2AUD_MUSIC3_REPAINT_MODES.get(arch or "", ())


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
