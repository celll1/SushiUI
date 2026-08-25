"""
Single source of truth for all API parameter default values.

Backend Pydantic models and Form() defaults reference this module.
Frontend fetches these via /schema/* endpoints, eliminating manual sync.
"""

from typing import Any, Dict, Optional

# The only core import in this module. The hybrid block range is a CHECKPOINT
# fact the loader validates against (hybrid_spec.py:71), so the API default is
# imported rather than re-typed: two copies could disagree and the request would
# still load.
from core.models.minimax_h3.hybrid_spec import (
    DEFAULT_BLOCK_RANGE_END,
    DEFAULT_BLOCK_RANGE_START,
    DEFAULT_FINAL_ADALN_FROM_OVERLAY,
    PRESET_BLOCK_RANGE_ADALN,
)

# ---------------------------------------------------------------------------
# MiniMax-H3 prompt assistant
# ---------------------------------------------------------------------------

PROMPT_ASSIST_DEFAULTS: Dict[str, Any] = {
    "provider": "lm_studio",
    "lm_studio_base_url": "http://127.0.0.1:1234",
    "ollama_base_url": "http://127.0.0.1:11434",
    "base_url": "",
    "model": "",
    "api_key": "",
    "references": [],
    "reference_description": "",
    "temperature": 0.2,
    "top_p": 0.9,
    "max_output_tokens": 3072,
    "context_length": 8192,
    "timeout_seconds": 300,
    # `instruction` is what to change THIS TIME ("make the drop harder"),
    # sent as its own field rather than appended into `prompt` -- see
    # `core/extensions/minimax_h3_prompt_assistant.py`'s `revise` field for
    # why that separation matters. `revise` selects between the existing
    # expand behavior (default) and revise mode.
    "instruction": "",
    "revise": False,
    "force_refresh": False,
    "cache_max_entries": 256,
    "auto_on_generate": False,
}

# ---------------------------------------------------------------------------
# MiniMax Music 3 caption rewriter (AI rewrite) — /prompt-assist/music/*
# ---------------------------------------------------------------------------
# Sibling of PROMPT_ASSIST_DEFAULTS above, not an extension of it: the two
# domains share the local-LLM provider/cache layer in
# backend/core/extensions/minimax_h3_prompt_assistant.py (loopback URL
# enforcement, SQLite result cache, one self-repair retry) but the wire
# contract differs completely — a music caption has no mode, duration, or
# reference tokens, and its output is a fixed three-heading Structured
# Caption, not a shot-numbered video prompt. See
# docs/guides/MINIMAX_MUSIC3_DESIGN.md, "Caption rewriter (AI rewrite)".
# Provider base URLs and model listing reuse H3's keys/route unchanged — the
# same local LM Studio/Ollama server serves both rewriters. Deliberately NOT
# duplicated here: `lm_studio_base_url`/`ollama_base_url` live only in
# PROMPT_ASSIST_DEFAULTS above, and both `_prompt_assist_base_url()` (server)
# and the frontend's provider switch (client) read them from there. A second
# copy of those two keys here previously drifted from what the server
# actually used to resolve an empty base_url, silently, the moment someone
# repointed one but not the other.

MUSIC_PROMPT_ASSIST_DEFAULTS: Dict[str, Any] = {
    "provider": "lm_studio",
    "base_url": "",
    "model": "",
    "api_key": "",
    "lyrics": "",
    "constraints": "",
    # `instruction`/`revise`: same split as PROMPT_ASSIST_DEFAULTS above --
    # `constraints` is a standing rule for the piece; `instruction` is a
    # one-off edit, applied only when `revise` is True (in which case
    # `caption` itself is treated as the base text to preserve rather than
    # a short caption to expand). See
    # core/extensions/minimax_music3_caption_rewriter.py.
    "instruction": "",
    "revise": False,
    "temperature": 0.2,
    "top_p": 0.9,
    "max_output_tokens": 3072,
    "context_length": 8192,
    "timeout_seconds": 300,
    "force_refresh": False,
    "cache_max_entries": 256,
}

# ---------------------------------------------------------------------------
# MiniMax Music 3 lyrics assistant — /prompt-assist/music/lyrics/*
# ---------------------------------------------------------------------------
# Sibling of MUSIC_PROMPT_ASSIST_DEFAULTS above (the caption rewriter), not
# an extension of it: separate cache, separate GUIDE_VERSION, separate wire
# contract (`mode` + `theme` + `lyrics`, no `constraints`-only shape). See
# backend/core/extensions/minimax_music3_lyrics_assistant.py and
# docs/guides/MINIMAX_MUSIC3_DESIGN.md.
#
# `mode` here is a UI-ONLY initial-tab default (which of the three mode
# buttons `MusicLyricsAssist.tsx` starts on), NOT a request default: the two
# LLM-driven modes are a REQUIRED field on `MusicLyricsAssistTransformRequest`
# (no Pydantic default at all -- `structure`/`complete` must always be sent
# explicitly), and `format` is served by a wholly separate endpoint
# (`POST /prompt-assist/music/lyrics/format`) that has no `mode` field to
# default in the first place. The frontend reads this key via
# `GET /schema/prompt-assist-music-lyrics-defaults` to pick its initial tab
# rather than hardcoding the same literal a second time. `provider`/
# `base_url`/`model` below only matter for `structure`/`complete` (`format`
# needs no LLM fields at all). Provider base URLs are read from
# PROMPT_ASSIST_DEFAULTS (H3's), same reasoning as MUSIC_PROMPT_ASSIST_DEFAULTS.

MUSIC_LYRICS_ASSIST_DEFAULTS: Dict[str, Any] = {
    "mode": "format",
    "theme": "",
    "lyrics": "",
    "constraints": "",
    # `instruction`/`revise`: same split as the two DEFAULTS above. Applies
    # only to the two LLM-driven modes (`structure`/`complete`); `format`
    # is deterministic and has no revise concept at all. See
    # core/extensions/minimax_music3_lyrics_assistant.py.
    "instruction": "",
    "revise": False,
    "provider": "lm_studio",
    "base_url": "",
    "model": "",
    "api_key": "",
    "temperature": 0.2,
    "top_p": 0.9,
    "max_output_tokens": 3072,
    "context_length": 8192,
    "timeout_seconds": 300,
    "force_refresh": False,
    "cache_max_entries": 256,
}

# ---------------------------------------------------------------------------
# MiniMax-H3 hybrid DiT load (POST /models/load)
# ---------------------------------------------------------------------------
# Not generation parameters: these are load-time inputs, so they have their own
# bundle rather than keys in GENERATION_DEFAULTS (same separation as
# VIDEO_CHAIN_DEFAULTS). The keys are the loader's REQUEST field names
# (hybrid_spec.HYBRID_REQUEST_KEYS); the route spells them `hybrid_*` because
# `preset` alone would not say what it presets.
#
# Frontend delivery is `GET /models/minimax-h3/hybrid-overlays`, whose
# `defaults` object carries these values, rather than a `/schema/*` endpoint:
# the range is only meaningful next to the tree's block count, which that
# endpoint reads from the same headers (design doc section 6.2, option (a)).

H3_HYBRID_LOAD_DEFAULTS: Dict[str, Any] = {
    # No overlay = an ordinary single-checkpoint load, byte for byte.
    "overlay_file": None,
    "preset": PRESET_BLOCK_RANGE_ADALN,
    "block_range_start": DEFAULT_BLOCK_RANGE_START,
    "block_range_end": DEFAULT_BLOCK_RANGE_END,
    "final_adaln_from_overlay": DEFAULT_FINAL_ADALN_FROM_OVERLAY,
}

# ---------------------------------------------------------------------------
# SenseNova-U1.5-8B-MoT generation defaults
# ---------------------------------------------------------------------------
# `steps`/`cfg_scale` here are SenseNova's actual served defaults: the four
# image routes resolve an omitted `steps`/`cfg_scale` through
# `IMAGE_GEN_ARCH_OVERLAYS`/`image_defaults_for_arch` (below GENERATION_DEFAULTS),
# which overlays this dict's values on top of the shared 20/7.0 for a loaded
# SenseNova model. `SENSENOVA_GENERATION_DEFAULTS` is also consulted directly as
# `core.pipeline_backends.sensenova`'s internal fallback for a caller that
# builds a params dict without going through the route (mirrors MiniT2I's
# existing `params.get("cfg_scale", 6.0)` pattern in
# `core/pipeline_backends/minit2i.py`, sourced from here instead of a bare
# literal). `timestep_shift` is new to this architecture and collides with no
# other arch's key, so it is exposed as a real, shared `GENERATION_DEFAULTS`
# entry (below) even though the dict stays flat.
SENSENOVA_GENERATION_DEFAULTS: Dict[str, Any] = {
    # Upstream reference (examples/t2i/inference.py) default step count -- NOT
    # the 8-step distillation LoRA's count. Driving the LoRA path is the
    # caller's own choice (cfg_scale<=1 collapses CFG to single-branch); there
    # is no separate "8-step mode" flag anywhere in this integration.
    "steps": 50,
    # Also the shared default for real-negative_prompt use, not just the
    # empty-string-uncond case -- see docs/guides/MODEL_FACTS.md's sensenova
    # row for the measured cfg_scale/negative_prompt behavior.
    "cfg_scale": 4.0,
    # Upstream examples/t2i/inference.py default (NOT config.json's
    # timestep_shift: 1.0 -- see core/models/sensenova/sensenova_pipeline_ops.py's
    # module docstring for why the example script's value is the one used here).
    "timestep_shift": 3.0,
    # Second CFG scale used by SenseNova's it2i_generate (reference-image
    # editing) path alongside cfg_scale. 1.0 is upstream's documented
    # operating point (cfg_scale=4.0, img_cfg_scale=1.0) and keeps the
    # default at two-branch cost; inert without ref_images.
    "img_cfg_scale": 1.0,
    # CFG-overshoot clamp applied to the predicted velocity before the Euler
    # step (see sensenova_pipeline_ops.py's _cfg_combine): scales v_pred by
    # clamp(||v_cond||/||v_pred||, 0, 1), so it can only ever shrink an
    # overshoot. Defaulted ON because SenseNova is pixel-space and decodes no
    # VAE to absorb overshoot, unlike the archs served by
    # cfg_rescale_snr_alpha. The vendored generate() signatures default to
    # "none"; this is our own default -- see docs/guides/MODEL_FACTS.md's
    # sensenova row for the visual gate that measured it near-inert.
    "cfg_norm": "global",
    # Per-phase CPU eviction of the transformer's unused weight half
    # (understanding half idle during denoise, generation half idle during
    # prefix). Default False: the measured VRAM saving is real (-7.6 GB at
    # 9.05MP, -7.55 GB at 5-ref 2048^2) but the peaks fit a 48GB card without
    # it, and it costs ~21.7 GiB of host RAM that is never returned to the OS
    # (15.11 GiB pinned + pageable staging), persisting for the process
    # lifetime even for later generations run with the toggle off.
    "sensenova_mot_phase_eviction": False,
    # Streams the prefix-phase KV cache from pinned host memory per layer,
    # through a 2-slot GPU ring shared across layers and branches, instead of
    # keeping all 42 layers' KV buffers GPU-resident. Default False: the
    # measured peaks fit a 48GB card without it; the host-RAM cost (tens to
    # hundreds of MB transient, not the eviction toggle's tens of GiB) is
    # small but still unmeasured against a gate.
    "sensenova_kv_cache_streaming": False,
    # Reference-style K/V injected into every CFG branch, not cond only; ON for
    # SenseNova alone -- see StyleTransferConfig.inject_all_cfg_branches for why.
    # Internal wiring only -- not a route parameter, no user-facing knob.
    "style_inject_all_cfg_branches": True,
}

# ---------------------------------------------------------------------------
# Generation (txt2img / img2img / inpaint)
# ---------------------------------------------------------------------------
# Authoritative source: Form() defaults in generate_* route handlers.
# Pydantic GenerationParams defaults are aligned to these values.

GENERATION_DEFAULTS: Dict[str, Any] = {
    # Core
    "negative_prompt": "",
    "steps": 20,
    "cfg_scale": 7.0,
    # SenseNova U1.5 flow-matching time-shift (see SENSENOVA_GENERATION_DEFAULTS
    # above). Every other architecture has no equivalent knob and ignores it
    # (accepted and warned, api/arch_capabilities.py).
    "timestep_shift": SENSENOVA_GENERATION_DEFAULTS["timestep_shift"],
    # SenseNova it2i_generate second CFG scale (see SENSENOVA_GENERATION_DEFAULTS
    # above). Only meaningful with ref_images on SenseNova; every other
    # architecture ignores it.
    "img_cfg_scale": SENSENOVA_GENERATION_DEFAULTS["img_cfg_scale"],
    # SenseNova CFG-overshoot clamp (see SENSENOVA_GENERATION_DEFAULTS above).
    # Every other architecture ignores it (accepted and warned,
    # api/arch_capabilities.py).
    "cfg_norm": SENSENOVA_GENERATION_DEFAULTS["cfg_norm"],
    # SenseNova per-phase weight-half CPU eviction (see
    # SENSENOVA_GENERATION_DEFAULTS above). Every other architecture ignores
    # it (accepted and warned, api/arch_capabilities.py).
    "sensenova_mot_phase_eviction": SENSENOVA_GENERATION_DEFAULTS["sensenova_mot_phase_eviction"],
    # SenseNova per-phase KV cache CPU streaming (see
    # SENSENOVA_GENERATION_DEFAULTS above). Every other architecture ignores
    # it (accepted and warned, api/arch_capabilities.py).
    "sensenova_kv_cache_streaming": SENSENOVA_GENERATION_DEFAULTS["sensenova_kv_cache_streaming"],
    "sampler": "euler",
    "schedule_type": "uniform",
    "seed": -1,
    "ancestral_seed": -1,
    "width": 1024,
    "height": 1024,
    "batch_size": 1,
    "prompt_chunking_mode": "a1111",
    "max_prompt_chunks": 0,
    "developer_mode": False,
    # CFG scheduling
    "cfg_schedule_type": "constant",
    "cfg_schedule_min": 1.0,
    "cfg_schedule_max": None,
    "cfg_schedule_power": 2.0,
    "cfg_rescale_snr_alpha": 0.0,
    # Dynamic thresholding
    "dynamic_threshold_percentile": 0.0,
    "dynamic_threshold_mimic_scale": 7.0,   # Form() default; Pydantic had 1.0 (bug)
    # NAG
    "nag_enable": False,
    "nag_scale": 5.0,
    "nag_tau": 3.5,
    "nag_alpha": 0.25,
    "nag_sigma_end": 3.0,                   # Form() default; Pydantic had 0.0 (bug)
    "nag_negative_prompt": "",
    # Attention / quantization
    "attention_type": "normal",
    # Which attention IMPLEMENTATION runs the kernel (orthogonal to attention_type,
    # which selects the backend). "conduit" routes through core/attention (enables
    # conduit-only backends such as tq on FLUX.2); "diffusers" keeps diffusers' own
    # registry (byte-identical legacy path). Consumed by the FLUX.2 inference path;
    # other archs are conduit-only or ignore it.
    "attention_impl": "conduit",
    # U-Net/transformer FP8 quantization: a VRAM-reduction feature, not a speed
    # feature. Weights are dequantized back to full precision per operation
    # during inference, which measures slower than full precision on every
    # architecture where it does anything (sd15/sdxl, zimage/flux2, anima/lens).
    "unet_quantization": None,
    "text_encoder_quantization": None,
    # Quantized-GEMM path for checkpoints that ALREADY carry weight-only
    # quantized Linear weights (Ideogram 4: FP8/nf4; Krea 2: FP8 or INT8;
    # Anima: INT8). A different axis from `unet_quantization`, which quantizes
    # an unquantized model's weights at load time to reduce VRAM; this one
    # selects HOW already-quantized weights are multiplied.
    #   None      -> do not touch the process-level flags (the default). The
    #                process value set by SUSHI_FP8_SCALED_MM / SUSHI_INT8_MM,
    #                or by POST /system/fp8-scaled-mm | /system/int8-mm, stands.
    #   "w8a8"    -> force BOTH W8A8 paths on for this generation.
    #   "dequant" -> force BOTH W8A8 paths off for this generation.
    # One axis rather than two booleans: FP8-vs-INT8 is decided by the
    # checkpoint format, not by the caller, so an impossible request ("int8 on
    # an fp8 checkpoint") is unrepresentable. MUST stay None: a False/"dequant"
    # default here would override an env-var opt-in for every existing caller.
    "quantized_gemm_mode": None,
    "cpu_text_encoding": False,
    "use_torch_compile": False,
    # Keep-models-hot (queue optimization, SD1.5/SDXL only in this phase): when
    # generating several items back-to-back on the SAME loaded model (same
    # checkpoint + LoRA set + unet_quantization), skip the CPU offload at the end
    # of a successful generation (and the matching GPU stage at the start of the
    # next one) for components it is safe to keep resident, bounded by a VRAM
    # guard. Off by default (byte-identical staging/offload behavior). See
    # core/keep_hot.py.
    "keep_models_hot": False,
    # VAE tiling: decode the latent in overlapping tiles so the VAE decode peak is
    # bounded by the tile size, not the full image. Lets large images decode
    # without OOM. Off by default (not bit-identical to a full decode; small images
    # below the tile threshold are unaffected).
    "vae_tiling": False,
    # Image size (px) above which VAE tiling kicks in (and the tile size). 0 = auto
    # = VAE sample_size * 1.5 (e.g. ~1536px for SDXL). Below the threshold the decode
    # runs whole (no quality/speed cost); above it, split into threshold-sized tiles.
    "vae_tile_threshold": 0,
    # How tiled VAE decode joins its tiles.
    #   "blend"   = diffusers' own overlapping tiles + linear cross-fade of the
    #               overlap band (the historical behavior, and the default).
    #   "context" = each tile is decoded with a margin of real neighbouring
    #               latent cells which is then discarded, so tiles join without
    #               a cross-fade. The threshold above is the decode-area budget
    #               in this mode (output tile = threshold - 2*margin), which
    #               keeps the decode peak bounded by the same block size.
    # Default is "blend": measured on one 1536px image per family, the two modes
    # differ by <0.5/255 in mean, and on the GroupNorm-bearing decoders (SDXL
    # family) the dominant tiling artifact is a per-tile tint that blend's
    # cross-fade ramps rather than steps. See scratchpad/vae_training/results_4a1.md
    # and core/inference/context_tiled_decode.py.
    "vae_tile_mode": "blend",
    # Two-pass global GroupNorm statistics for a tiled VAE decode (opt-in).
    # A tiled decode normalises each tile with that tile's own per-group
    # statistics, which offsets whole tiles against each other. With this on, the
    # decode runs twice: pass 1 records every decoder GroupNorm's per-group
    # statistics across the tiles, pass 2 re-decodes forcing the accumulated
    # whole-image statistics. Independent of vae_tile_mode (it wraps the whole
    # decode, so it applies to both "blend" and "context").
    # Measured on SDXL, blend join, 512px budget: per-tile offset peak-to-peak
    # 1.32 -> 0.037 /255 (fp32) and 1.35 -> 0.038 (fp16); 1.36 -> 0.18 in bf16,
    # where the folded correction is limited by bf16's 8 mantissa bits.
    # Peak VRAM +0.00003 GB; decode wall time x2.
    # The x2 applies to EVERY decode in the request: on SD1.5/SDXL the override is
    # installed before the sampling loop, so the in-loop decodes of
    # flatten_in_loop (one per injected step) and vae_drift_correction are
    # doubled too.
    # No effect (and no second pass) when the decoder contains no GroupNorm --
    # which includes the Qwen-family autoencoder used by Anima and Krea2 -- or
    # when the image is below the tile threshold. Off by default.
    # See core/inference/global_group_norm.py.
    "vae_tile_global_norm": False,
    # Color Flatten (chroma smoothing): RGB-guided guided filter applied to the
    # decoded image's YCoCg chroma (luma untouched) right after VAE decode. Removes
    # low-frequency color mottling while preserving luminance detail. 0-100; 0 = off
    # (zero cost). All modes, all architectures.
    "color_flatten_strength": 0,
    # VAE DC-drift correction (img2img/inpaint only): subtract the per-channel DC
    # bias the VAE round-trip introduces (mean(decode(encode(input))) - mean(input))
    # from the final decode. Corrects a VAE property, so it is strength-independent.
    "vae_drift_correction": False,
    # In-loop hard-flatten (SD1.5/SDXL only): on the last N actual denoise steps,
    # decode the x0 prediction, detect the flat background region (largest connected
    # low-gradient component touching the border) and replace it with its dominant
    # colour, then re-encode and inject the correction back into the latents. Gated:
    # if no confident flat region (area < min_region) the step is a no-op, so
    # textured backgrounds are protected. DiT archs accept the flag and warn.
    "flatten_in_loop": False,       # master switch (default off = byte-identical loop)
    # Number of trailing ACTUAL denoise steps to inject on (relative to the real step
    # sequence, not a fixed fraction, so it is stable across step counts / accelerators).
    # Larger N = flatter background but more subject-detail cost (~6-7% wall per injection).
    "flatten_in_loop_last_steps": 3,
    # Flat-region area gate as a fraction of the frame. Below this the step is a no-op.
    "flatten_in_loop_min_region": 0.02,
    # Spectrum: Adaptive Spectral Feature Forecasting (training-free acceleration).
    # Skips U-Net forwards on selected steps by forecasting the output from a Chebyshev
    # fit over actual passes. Most useful at high step counts (>=30); little benefit on
    # low-step/distilled models. Auto-disabled with prompt-editing/ControlNet/DEUS.
    "spectrum_enable": False,
    "spectrum_w": 0.5,             # spectral/linear mix (1.0 = spectral only; lower = more linear/stable)
    "spectrum_w_decay": 0.0,       # OPT-IN per-step decay exponent for spectrum_w (0 = off, default)
    "spectrum_delta_cap": 0.0,     # OPT-IN trajectory speed limiter multiplier K (0 = off, default)
    "spectrum_m": 4,               # number of Chebyshev basis
    "spectrum_lam": 0.1,           # ridge regularization
    "spectrum_warmup_steps": 3,    # leading full-eval steps
    "spectrum_window_size": 4,     # initial skip interval
    "spectrum_flex_window": 0.75,  # skip damping (0 = max skip)
    "spectrum_tail": 0.12,         # fraction of final steps forced to actual passes (detail)
    "spectrum_feature_mode": "output",  # "output" (black-box) or "block" (deep-feature, paper-faithful)
    "spectrum_cache_branch": 1,    # block mode: down_blocks[branch:] + mid are forecast
    "spectrum_max_cache": 0,       # forecaster sliding window (0 = unlimited; block mode -> 6)
    # First Block Cache (FBCache): dynamic per-step caching. Mutually exclusive with Spectrum.
    "fbcache_enable": False,
    "fbcache_threshold": 0.12,     # relative-L1 first-block residual threshold (higher = more skips/faster)
    "fbcache_warmup_steps": 1,     # always compute the first N steps
    "fbcache_cache_branch": 1,     # U-Net block mode: indicator = down[branch]; reused region = down[branch+1:]+mid
    "use_tipo": False,
    "preview_predicted_x0": False,
    # Block swap (Form-only in original code)
    "enable_block_swap": False,
    "blocks_to_swap": 20,
    "use_pinned_memory": False,
    "block_swap_h2d_only": False,   # H2D-only swap (no device->host eviction of read-only weights)
    "block_swap_ring_size": 2,      # GPU weight-buffer ring slots (>=1; 2 double-buffers)
    # Vision encoder
    "vision_encoder_path": None,
    # Per-generation component overrides (RP2b). Both default to None = use the
    # loaded model's own component. VAE override supported on all image archs
    # except ltx2/minit2i; TE override supported on sd15/sdxl only.
    # NOT the same thing as POST /models/load's text_encoder_file /
    # clip_projection_file, which choose which encoder is BUILT at load time
    # (MiniMax-H3) and therefore have no entry here.
    "vae_path": None,
    "text_encoder_path": None,
    # PiD (Pixel Diffusion Decoder) VAE-override options — only consulted when
    # vae_path points at a PiD checkpoint (kind="pid_decoder"; see
    # api.generation_overrides.classify_vae_candidate); no effect otherwise.
    # "4x" = PiD's native super-resolution output (latent_h/w * 8 * 4); PiD's
    # sr_scale=4 is baked into the checkpoint so "original" still runs the full
    # 4x decode and downscales afterward (NOT a cheaper mode).
    "pid_sr_output": "4x",
    # False (default) = use the shipped null-caption embedding (no runtime
    # Gemma). True = load Efficient-Large-Model/gemma-2-2b-it (the ungated
    # mirror PiD was trained against) once per generation and encode the real
    # prompt for sharper, hallucination-reduced output (see phase 1b findings).
    "pid_use_gemma": False,
    # False (default) = PiTBlock/FinalLayer run their exact original,
    # unchunked forward pass (bit-identical output). True opts into a
    # row-chunked activation path that cuts the PiD decoder's per-block VRAM
    # peak (~6.6GB/42% measured at 4096px) at the cost of bf16 GEMM-tiling
    # rounding drift that is NOT bit-identical (verified bit-identical in
    # fp32; the drift is bf16-precision amplification through the 4-step SDE
    # sampler, not an implementation bug — see scratchpad/pid_vram_proposal.md).
    "pid_low_vram": False,
    # F9 — tiled large-output decode (default when native > native_cap; see
    # PidVaeWrapper's module docstring). Each tile's OWN native resolution is
    # capped at pid_tile_native (must stay <= the (currently hardcoded)
    # native_cap, else it is clamped with a warning); pid_tile_overlap_ratio
    # is the feather overlap as a fraction of the tile size (0.25 = R&D-
    # confirmed seam-free on both busy and smooth backgrounds).
    "pid_tile_native": 512,
    "pid_tile_overlap_ratio": 0.25,
    # False (default) = tiled decode (F9, true super-resolution detail at the
    # full requested output size). True opts back into the original
    # whole-latent downscale-then-decode-then-upscale (F7) — cheaper
    # (~6x fewer decode passes) but blurrier at large output sizes.
    "pid_fast_large_decode": False,
    # SDXL micro-conditioning override (inference). original_size for time_ids:
    # explicit w/h (0 = auto), else output size * scale. crop stays (0,0).
    "original_size_w": 0,
    "original_size_h": 0,
    "original_size_scale": 1.0,
    # img2img / inpaint shared
    "denoising_strength": 0.75,
    "img2img_fix_steps": True,
    "resize_mode": "image",
    "resampling_method": "lanczos",
    # inpaint only
    "mask_blur": 4,
    "inpaint_full_res": False,
    "inpaint_full_res_padding": 32,
    "inpaint_fill_mode": "original",
    "inpaint_fill_strength": 1.0,
    "inpaint_blur_strength": 1.0,
    # Regional additional prompt (SD/SDXL only, inpaint/outpaint): an
    # additional positive/negative prompt that conditions ONLY the generated
    # region (outpaint = mask_latent==1; inpaint = the repaint mask), leaving
    # the main whole-image prompt and the preserved region untouched. Active
    # iff region_prompt_strength > 0 AND (region_prompt or
    # region_negative_prompt) is non-empty. See
    # core.inference.custom_sampling.custom_inpaint_sampling_loop's REGIONAL
    # ADDITIONAL PROMPT block.
    "region_prompt": "",
    "region_negative_prompt": "",
    "region_prompt_strength": 1.0,   # 0-2; 0 = feature inactive
    "region_prompt_method": "cfg",   # "cfg" (spatial/masked CFG) | "attention" (not yet implemented -- falls back to "cfg" with a warning)
    "region_mask_feather": 0.0,      # Gaussian sigma (latent cells) for the region mask edge; 0 = hard mask
    # Seam Structure Continuity (SD/SDXL only, inpaint/outpaint): make thin
    # structures that cross the region boundary (a held rod/staff, a limb,
    # torso, architectural lines) CONTINUE coherently into the generated region.
    # Detects oriented structures crossing the boundary in the known latent
    # (structure tensor), affine-extrapolates their band-pass cross-section into
    # a shallow generate-side collar, and pulls the model's per-step x0
    # prediction toward that target -- gated per boundary position by a
    # structure-saliency map so it fires ONLY where a real crossing structure
    # touches the boundary (leaving the rest of the seam untouched). x0-space,
    # no extra U-Net forwards, known region bit-exact. See
    # core.inference.custom_sampling._ssc_precompute / _ssc_apply. 0 = off
    # (byte-identical). Active range ~0.25-0.6.
    "seam_structure_strength": 0.0,  # lambda; 0 = feature inactive
    "seam_structure_depth": 6.0,     # generate-side collar depth (latent cells)
    "seam_structure_end": 0.70,      # schedule progress at which the effect decays to 0 (full <= 0.45)
    "seam_structure_saliency": 2.0,  # saliency-gate midpoint as a multiple of the boundary-ribbon median (0 = whole seam)
    "seam_structure_max_area": 0.25, # safety cap: max fraction of the generate region the gate may cover
    # Boundary Determinism Relaxation (SD/SDXL only, inpaint/outpaint): the
    # "junction render" complement to Seam Structure Continuity. Instead of
    # HARD-pinning the known boundary each step, SOFT-pin a narrow SSC-saliency-
    # gated keep-side seam band (annealed from soft early to a full hard pin
    # late, with a small scheduled noise term), so the immediately-adjacent
    # known-side latent can bend to meet the continuation SSC indicates -- turning
    # a ~1-cell seam offset from a kink into a gentle bend. Deep known content
    # stays hard-pinned throughout. x0-space, no extra U-Net forwards. Most
    # effective WITH seam_structure_strength > 0 (SSC supplies the direction;
    # relaxation grants the permission). See core.inference.custom_sampling.
    # _bdr_precompute / _bdr_apply. 0 = off (byte-identical).
    "boundary_relax_strength": 0.0,   # 0 = feature inactive; active range ~0.2-0.35
    "boundary_relax_width": 3.0,      # keep-side band width (latent cells)
    "boundary_relax_noise": 0.35,     # band-noise fraction of the x0-posterior std (0-1)
    "boundary_relax_full_until": 0.37, # progress up to which the band is fully soft
    "boundary_relax_end": 0.55,       # progress by which the hard pin is fully restored
    "boundary_relax_paste": "feather", # Q3 paste variant: "feather" (thin model-rendered seam strip) | "exact" (full byte-exact rect)
    # Loop-generation decode mode (txt2img/img2img/inpaint, all steps of a
    # client-driven loop). "full" = decode with the active VAE (PiD if
    # overridden) + save + gallery (current/default behavior). "cheap" = if a
    # PidVaeWrapper override is active, decode with its EMBEDDED real SDXL VAE
    # instead of running the PiD student net (no-op -- identical to "full" --
    # when no PiD override is active); still saves + galleries. "none" = skip
    # decode/save/gallery entirely; the final latent is cached server-side
    # (core.inference.latent_cache) and a latent_id is returned instead of an
    # image, for the next loop step's input_latent_id (no VAE round-trip).
    # SD1.5/SDXL only in this phase (the legacy custom_sampling.py pipeline);
    # other architectures accept the field but always behave as "full".
    "loop_decode": "full",
    # img2img/inpaint: start denoising from a cached latent (see
    # core.inference.latent_cache) instead of an uploaded image -- the
    # loop-generation counterpart of loop_decode="none" on the PRODUCING step.
    # None = normal image upload (default). SD1.5/SDXL img2img only; inpaint
    # accepts the field but rejects a non-null value (mask compositing needs a
    # real source image -- see the design doc).
    "input_latent_id": None,
    # Orthogonal to loop_decode: when true, the generated image is still saved
    # to disk (so the loop can chain to the next step via the file path) but
    # the gallery database record (thumbnail + DB row) is skipped. Used for
    # loop-generation intermediate steps that decode cheaply (e.g. image-space
    # upscale via loop_decode="cheap") but shouldn't clutter the gallery.
    # Meaningless combined with loop_decode="none" (nothing is decoded/saved
    # there either way).
    "skip_gallery": False,
}

# ---------------------------------------------------------------------------
# Per-architecture image defaults overlay (SSOT)
# ---------------------------------------------------------------------------
# `GENERATION_DEFAULTS` above is shared by every image architecture. The
# overlay/resolver pair below is the image-side twin of
# `VIDEO_GEN_ARCH_OVERLAYS`/`video_defaults_for_arch`: an architecture with no
# entry resolves to `GENERATION_DEFAULTS` unchanged (today's behaviour, bit-
# identical), and the four image routes (txt2img/img2img/inpaint/outpaint) fill
# only the fields a request OMITS from it, via `Form(None)` sentinels -- same
# contract as the video routes' `model_fields_set`/`Form(None)` split. Declared
# Form() defaults stay the base values; the schema documents one stable shape.
#
# Scoped to `steps`/`cfg_scale` only: SenseNova's upstream reference operating
# point (`SENSENOVA_GENERATION_DEFAULTS` above) is otherwise unused by any
# route, so every real SenseNova request was served the shared 20/7.0 instead
# of the documented 50/4.0 -- worse there than on a latent-space arch because
# SenseNova decodes no VAE to absorb CFG overshoot.
IMAGE_GEN_ARCH_OVERLAYS: Dict[str, Dict[str, Any]] = {
    "sensenova": {
        "steps": SENSENOVA_GENERATION_DEFAULTS["steps"],
        "cfg_scale": SENSENOVA_GENERATION_DEFAULTS["cfg_scale"],
    },
}


def image_defaults_for_arch(arch: Optional[str],
                            base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`base` (default `GENERATION_DEFAULTS`) with ``arch``'s overlay applied.

    The single resolver behind every image route's omitted-field handling and
    behind the `image_arch_overlays` block of `/schema/generation-defaults`, so
    the frontend and the backend resolve a per-arch image default the same way.

    An unknown or missing arch returns `base` unchanged (a copy), which is both
    every non-SenseNova architecture's behaviour and the safe answer for a
    model whose type has not been resolved yet.
    """
    resolved = dict(base if base is not None else GENERATION_DEFAULTS)
    resolved.update(IMAGE_GEN_ARCH_OVERLAYS.get(arch or "", {}))
    return resolved


# Keys present only in img2img/inpaint (not txt2img)
_IMG2IMG_ONLY = frozenset({
    "denoising_strength", "img2img_fix_steps", "resize_mode", "resampling_method",
    # VAE DC-drift correction requires an input image to measure the round-trip
    # bias, so it exists for img2img + inpaint only (excluded from txt2img).
    "vae_drift_correction",
    # Latent-passthrough loop chaining: only meaningful where an input image
    # would otherwise be required.
    "input_latent_id",
})
# Keys present only in inpaint (not txt2img or img2img)
_INPAINT_ONLY = frozenset({
    "mask_blur", "inpaint_full_res", "inpaint_full_res_padding",
    "inpaint_fill_mode", "inpaint_fill_strength", "inpaint_blur_strength",
    # Regional additional prompt: mask-scoped, so it only makes sense where a
    # generate/repaint mask exists (inpaint) -- inherited by OUTPAINT_DEFAULTS
    # below via the same path as mask_blur.
    "region_prompt", "region_negative_prompt", "region_prompt_strength",
    "region_prompt_method", "region_mask_feather",
    # Seam Structure Continuity: mask-scoped (structures crossing the generate/
    # repaint boundary), so inpaint + outpaint only -- inherited by
    # OUTPAINT_DEFAULTS via the same path as region_* / mask_blur.
    "seam_structure_strength", "seam_structure_depth", "seam_structure_end",
    "seam_structure_saliency", "seam_structure_max_area",
    # Boundary Determinism Relaxation: keep-side seam band, inpaint + outpaint only.
    "boundary_relax_strength", "boundary_relax_width", "boundary_relax_noise",
    "boundary_relax_full_until", "boundary_relax_end", "boundary_relax_paste",
})

TXT2IMG_DEFAULTS: Dict[str, Any] = {
    k: v for k, v in GENERATION_DEFAULTS.items()
    if k not in _IMG2IMG_ONLY and k not in _INPAINT_ONLY
}
IMG2IMG_DEFAULTS: Dict[str, Any] = {
    k: v for k, v in GENERATION_DEFAULTS.items()
    if k not in _INPAINT_ONLY
}
INPAINT_DEFAULTS: Dict[str, Any] = dict(GENERATION_DEFAULTS)

# ---------------------------------------------------------------------------
# Outpaint (POST /generate/outpaint)
# ---------------------------------------------------------------------------
# Image spatial outpaint: place a (optionally cropped/resized) input image
# inside a LARGER canvas and generate everything outside it, preserving the
# placed region byte-exact (see core/inference/outpaint_utils.py +
# PipelineManager.generate_outpaint). Pure orchestration over the existing
# all-architecture generate_inpaint -- shares its ENTIRE parameter set
# (feature parity: LoRA/ControlNet/NAG/quant/block-swap/advanced-CFG/etc),
# built here by deriving from INPAINT_DEFAULTS rather than duplicating
# literal values (single source of truth).
OUTPAINT_DEFAULTS: Dict[str, Any] = {
    **INPAINT_DEFAULTS,
    # Outpaint's default is full-strength generation of the surrounding
    # canvas (the placed region is preserved regardless via the final pixel
    # paste, not via denoising_strength). Inpaint defaults to 0.75 (partial
    # repaint of existing content); outpaint has no existing content in the
    # generated region, so full strength is the natural default. strength<1.0
    # remains available for a more input-guided outpaint.
    "denoising_strength": 1.0,
    # NOTE: "width"/"height" are inherited from INPAINT_DEFAULTS above for
    # schema key-parity, but are NOT accepted as request parameters for
    # outpaint -- canvas_width/canvas_height fully determine the output size
    # (PipelineManager.generate_outpaint overwrites params["width"]/["height"]
    # with the resolved canvas size before delegating to generate_inpaint).
    #
    # --- Placement (new for outpaint) ---
    # Output canvas size. Rounded to the loaded architecture's latent-grid
    # alignment (see outpaint_utils.validate_and_snap_placement, align=8).
    "canvas_width": 1536,
    "canvas_height": 1536,
    # Top-left of the placed input rectangle on the canvas. 0/0 is clamped
    # into bounds server-side; the frontend is expected to compute a sensible
    # (e.g. centered) position rather than relying on this literal default.
    "place_x": 0,
    "place_y": 0,
    # Placed size on the canvas (the resize target -- the resized result IS
    # the preserved content). 0 = use the (cropped) input's native size.
    "place_width": 0,
    "place_height": 0,
    # Trim (crop) applied to the input image BEFORE placement. 0 width/height
    # = crop to the input's edge (i.e. "no trim" when x/y are also 0).
    "input_crop_x": 0,
    "input_crop_y": 0,
    "input_crop_w": 0,
    "input_crop_h": 0,
    # How the canvas outside the placed rect is pre-filled before denoising:
    # "replicate" (edge-extend the placed content outward), "reflect"
    # (mirror it outward), "mean" (solid average color of the placed
    # content), "noise" (uniform random RGB).
    "outpaint_fill_mode": "replicate",
    # mask_blur is already inherited from INPAINT_DEFAULTS (4) -- called out
    # here because outpaint's blur is OUTWARD-ONLY (the softened transition
    # band lies entirely outside the preserved rect; see
    # outpaint_utils.build_outpaint_mask), unlike inpaint's symmetric blur.
    # The numeric default is unchanged.
    #
    # Exposure-seam harmonizer (core.inference.outpaint_utils.match_generated_
    # exposure): corrects a tonal mismatch between the generated surroundings
    # and the preserved rect by comparing strips just inside/outside the rect
    # edges. Applied after generation, before the final unconditional paste,
    # so it never touches the preserved rect. On by default; disable to see
    # the model's raw, uncorrected output.
    "outpaint_seam_fix": True,
    # Harmonic boundary-offset membrane (core.inference.seam_membrane): a
    # post-decode local correction, distinct from the exposure harmonizer
    # above. Solves a smooth per-channel offset field over the generated
    # region (harmonic away from the boundary) whose value AT the seam
    # exactly equals the preserved rect's own pixels there, tapering to 0
    # within seam_membrane_band px. Runs after the exposure harmonizer and
    # before the final unconditional paste, so the preserved rect is never
    # altered (double-guaranteed: the membrane itself never writes rect
    # pixels, and the final paste re-establishes byte-exactness regardless).
    # 0/False = off (byte-identical; the module is not even imported).
    "outpaint_seam_membrane": False,
    # Taper band width (px) over which the membrane's correction fades to 0.
    # 0 = auto (clamp(max(canvas_width, canvas_height) // 8, 64, 256)).
    "outpaint_seam_membrane_band": 0,
    # Cross-seam low-frequency tone membrane (core.inference.seam_membrane.
    # apply_cross_seam_tone, "R2"): a SEPARATE, distinct correction from the
    # harmonic membrane above. Measures the per-channel tone step between the
    # preserved rect's own pixels and the decoded GENERATED pixels
    # immediately across the seam (not the rect-interior reconstruction the
    # harmonic membrane keys on), subtracts the local content gradient
    # estimated from the preserved side so a legitimate ramp is not
    # flattened, low-passes the residual along the seam axis, and writes a
    # decaying offset into the generated side only, within
    # outpaint_seam_tone_band px of the seam. Runs after the harmonic
    # membrane and before the final unconditional paste, so the preserved
    # rect is never altered. 0 = off (byte-identical; the module is not even
    # imported when both this and outpaint_seam_membrane are off).
    "outpaint_seam_tone_strength": 0.0,
    # Decay band width (px) over which the tone membrane's offset fades to 0.
    # 0 = auto (16 px).
    "outpaint_seam_tone_band": 0,
    # Boundary-offset propagation ("G_prop16", core.inference.seam_membrane.
    # apply_seam_offset_propagation): a THIRD, separate mechanism from the two
    # membranes above, strict-preservation-native by construction (writes only
    # generated-side pixels near the seam, never the preserved rect). Measures
    # the same placed-vs-decoded-reconstruction offset the harmonic membrane
    # measures (at the rect-interior boundary, not the cross-seam comparison
    # the tone membrane uses), and propagates it directly into the generated
    # band as a Gaussian low-frequency term plus a short high-frequency
    # residual term (each independently tapered), instead of solving a
    # Poisson field. Runs after the tone membrane and before the final
    # unconditional paste. Scales the applied offset; 0 (default) = off (the
    # module is not imported). Internal constants (low/high-frequency band
    # widths, Gaussian sigma, taper shape, clamp) are fixed -- validated
    # against a real decode in scratchpad/outpaint_seamless_vae_native.md.
    # Default 0.0 (off): on the trained crop_mask CN the real boundary seam
    # decomposes into a hard paste-line (removed by outpaint_paste_feather_px
    # below) and a CN-driven generation frame (a training property, not
    # correctable here); this generated-side offset propagation targets
    # neither cleanly (it clamp-saturated in practice), so it is left off in
    # the shipped recipe. Set > 0 to re-enable for tone-step seams.
    "outpaint_seam_offset_prop": 0.0,
    # Paste-band reconciliation feather ("Option E", core.inference.
    # outpaint_utils.reconcile_and_paste's paste_feather_px; see
    # scratchpad/outpaint_seam_latent_stage.md section 4.1): at the FINAL
    # preserved-rect paste, the last N rows/columns of the preserved rect at
    # its generate-adjacent edges are blended (raised-cosine, via the existing
    # build_paste_alpha/paste_preserved_region alpha-paste path) from the
    # exact input toward the decoded canvas already sitting underneath them,
    # instead of pasted byte-exact -- turning the raw/decoded junction into a
    # gradient. Independent of Boundary Relaxation's own feather paste (BDR
    # Variant B, boundary_relax_strength/boundary_relax_paste above) and takes
    # precedence over it when both are active; applies to every outpaint
    # ControlNet mode including "crop_mask". Only the N-row/column band loses
    # exactness; the rest of the preserved rect is unaffected. Default 24:
    # this is the "tiled-VAE-style" blend that removes the hard raw/decoded
    # paste-line at the seam (verified: left-edge 1px luma gradient 21.5 -> 4.4
    # on the reference case) while keeping the rect's deep interior byte-exact.
    # Set 0 for the strict byte-exact-at-the-edge paste (reintroduces the line).
    "outpaint_paste_feather_px": 24,
    # Preserved-region compositing mode (opt-in; SD1.5/SDXL gets the full
    # mechanism, other architectures get the "vae_reconstruct" behavior only
    # -- see per-mode notes below and core.inference.outpaint_utils.
    # reconcile_and_paste / core.inference.custom_sampling's outpaint keep-
    # paste block). "exact" (default) is the CURRENT byte-exact behavior:
    # the preserved rectangle is pasted pixel-for-pixel from the input,
    # unchanged by this parameter's existence. "vae_reconstruct" outputs a
    # single uniform VAE decode of the whole canvas with NO paste at all --
    # the preserved region becomes a VAE reconstruction of the input, not
    # byte-identical to it, but the boundary is no longer a hard raw/decoded
    # pixel discontinuity. "vae_reconstruct_hf" additionally restores the
    # preserved region's high-frequency detail (raw minus its own VAE
    # roundtrip reconstruction) on top of the uniform decode, tapering that
    # restoration to zero over the last rows/columns approaching the
    # boundary so the boundary itself still matches the uniform decode
    # exactly; higher fidelity to the input than "vae_reconstruct" while
    # keeping the same seamless boundary. Both non-"exact" modes make the
    # preserved region NOT byte-identical to the input and emit a warning.
    "outpaint_preserve_mode": "exact",
    # Honest outpaint preview (display-only; core.inference.custom_sampling's
    # outpaint sampling loop): the loop pins pred_original_sample to the
    # composite (1-M)*K + M*x0_hat before it reaches the preview/TAESD
    # decoder, so mid-sampling previews can show a boundary line that the
    # actual scheduler math (and the final saved image) never has. When true,
    # the UNPINNED model prediction (x0_hat, pre-composite) is sent to the
    # preview decoder INSTEAD, for outpaint generations only. Purely a preview
    # substitution: scheduler.step() still uses the pinned composite
    # unchanged, so the final saved image for a given seed is byte-identical
    # regardless of this flag. False (default) preserves prior preview
    # behavior.
    "outpaint_preview_unpinned_x0": False,
    # B1 continuity fix (SD/SDXL only; core.inference.custom_sampling's
    # outpaint_noise_init-gated x0-space projection injection): a weak
    # low-frequency color/illumination correction applied to the generate
    # region ONLY within a narrow collar near the preserved rect's boundary,
    # active mid/late in the schedule. Nudges coarse chroma/illumination
    # continuity across the seam without constraining composition/content
    # far from it. 0 disables the correction entirely.
    "outpaint_boundary_color_strength": 0.25,
    # B2 continuity fix (SD/SDXL only; core.inference.custom_sampling's
    # outpaint_noise_init-gated RePaint-style band-limited time-travel
    # resampling): after completing a denoise step inside a mid-schedule band,
    # jump back outpaint_jump_length steps by re-noising the WHOLE latent
    # (keep + generate together, so the keep region becomes a correlated
    # sample instead of an independently-pinned one) and re-denoise, repeating
    # outpaint_resample_count times total per band segment. Re-exposes the
    # generate region to the keep constraint while it is still malleable,
    # fixing content divergence (invented/unrelated composition) that B1
    # alone does not address. 1 disables resampling entirely (B1 only).
    # Costs roughly 1.5-2x the requested step count in actual denoise passes.
    # Only takes effect with a resample-compatible sampler (Euler, Euler
    # Ancestral, DDIM, DDPM); other samplers fall back to B1 only.
    "outpaint_resample_count": 1,
    # B2 jump-back length ("u", in step indices) for each resample cycle.
    "outpaint_jump_length": 4,
    # B3 continuity fix (SD/SDXL only; core.inference.custom_sampling's
    # outpaint_noise_init-gated masked self-attention KV injection): reuses the
    # existing StyleAligned/VSP-style reference-KV-injection machinery
    # (core.inference.reference_style) with a noise-matched reference composite
    # built from the preserved rect's own clean latents (not a user-supplied
    # style image), restricted to KNOWN-region tokens via spatial masking, so
    # the generate region's self-attention queries directly attend to the
    # input's own clean features instead of only following the prompt --
    # addressing content/style/palette divergence that persists even with B1+B2.
    # Strength ("gamma") scales the injected reference Key/Value; 0 disables
    # the mechanism entirely (default, until tuned).
    "outpaint_reference_strength": 0.0,
    # Boundary-outward commitment front (SD/SDXL only; EXPERIMENTAL; AUGMENTS
    # B1's x0-space projection -- core.inference.custom_sampling's
    # outpaint_noise_init-gated commit proximal): a distance-graded,
    # low-frequency, EMA-anchored proximal that makes near-boundary generate
    # cells commit their COARSE structure earlier in the schedule (following
    # the preserved rect outward) than far-interior cells, approximating an
    # autoregressive-in-space factorization of the generate region conditioned
    # on the known content. Only the low-frequency band is committed (high-
    # frequency detail keeps refining every step); this is a biased continuity
    # regularizer, not an exact sampler change. 0 disables the proximal
    # entirely (default; byte-identical to B1 alone).
    "outpaint_commit_strength": 0.0,
    # Schedule progress (0-1) fraction at which boundary-touching generate
    # cells (distance 0) start committing.
    "outpaint_commit_near": 0.35,
    # Schedule progress (0-1) fraction at which the farthest generate cells
    # (distance >= outpaint_commit_distance) start committing.
    "outpaint_commit_far": 0.80,
    # Distance (in latent cells) from the preserved rect boundary at which the
    # commit-fraction schedule saturates to outpaint_commit_far. Cells farther
    # than this all commit at outpaint_commit_far.
    "outpaint_commit_distance": 32.0,
    # Outpaint ControlNet -- PART A edge-extrapolation ControlNet (SD/SDXL
    # only; scratchpad/outpaint_controlnet_synthesis.md; core.inference.
    # outpaint_control.build_outpaint_control_image + PipelineManager.
    # generate_outpaint). Detects structures in the preserved region that
    # cross the rect boundary (a held rod, a limb, architectural lines) and
    # extrapolates them a short, confidence-tapered distance into the
    # generate region as a synthetic ControlNet, driving a general structure
    # ControlNet model (e.g. an "anytest"-style checkpoint). The same
    # confidence field also spatially gates the ControlNet's residual
    # contribution in the denoise loop, so its influence tapers to 0 with
    # distance from the boundary instead of applying uniformly. This is an
    # ENFORCEMENT tool over a GUESSED continuation geometry, not a learned
    # one -- off by default. v1: mutually exclusive with a user-supplied
    # ControlNet/LLLite (never overrides the user's own request) and forces
    # boundary_relax_paste="exact" + disables seam_structure_strength while
    # active (see PipelineManager.generate_outpaint). 0/False = off
    # (byte-identical to before this feature existed).
    "outpaint_controlnet_enable": False,
    # Conditioning mode. "edge_extrapolate" (default, PART A): detect edges in
    # the preserved region and extrapolate them a confidence-tapered distance
    # across the boundary -- drives a general "anytest"-style structure
    # ControlNet over a GUESSED geometry. "crop_mask" (PART B): build the
    # trained outpaint-native 4-channel conditioning (crop RGB + binary
    # known-mask, via core.utils.crop_mask_condition) -- requires a ControlNet
    # trained with conditioning_mode="outpaint" (4-ch), which LEARNED the
    # continuation, so no extrapolation/termination heuristic and a flat
    # (untapered) residual gate over the whole generate region.
    "outpaint_controlnet_mode": "crop_mask",
    # Path to the ControlNet model checkpoint driving this feature. Required
    # (non-empty) for the feature to take effect when enabled. For "crop_mask"
    # this must be an outpaint-trained diffusers directory (4-ch conditioning).
    "outpaint_controlnet_model": "",
    # Edge/structure detector run over the preserved region only (never the
    # synthetic canvas fill): "canny" (cv2, no model download) or
    # "lineart"/"lineart_anime" (controlnet_aux, opt-in; falls back to canny
    # if unavailable offline).
    "outpaint_controlnet_detector": "canny",
    # ControlNet conditioning scale for the synthetic control image.
    "outpaint_controlnet_scale": 0.6,
    # ControlNet guidance window start (schedule progress fraction, 0-1).
    "outpaint_controlnet_guidance_start": 0.0,
    # ControlNet guidance window end (schedule progress fraction, 0-1).
    "outpaint_controlnet_guidance_end": 0.55,
    # Maximum extrapolation depth (canvas pixels) a detected boundary-crossing
    # structure is continued into the generate region before its confidence
    # tapers to 0.
    "outpaint_controlnet_depth": 160,
    # Exponent of the cosine-squared distance taper applied to each
    # extrapolated structure's confidence (higher = sharper falloff near the
    # taper depth).
    "outpaint_controlnet_taper": 2.0,
    # "crop_mask" mode only: fixed INWARD edge-feather width (canvas px) passed to
    # build_crop_mask_condition's edge_feather_px at inference (see D3-R1,
    # scratchpad/outpaint_boundary_structure_fix.md). A crop_mask ControlNet
    # RE-TRAINED with TRAINING_DEFAULTS' per-sample randomized
    # outpaint_edge_feather_min_px/max_px range expects a soft perimeter at
    # inference too (the no-skew contract) -- a single FIXED value keeps inference
    # in-distribution without needing to sample per-request; set it to the
    # midpoint of that training range.
    #
    # DEFAULT IS 0.0 (razor-sharp): the current live crop_mask ControlNet was
    # trained BEFORE R1 (edge_feather_px always 0.0 in training), so it expects
    # the razor-sharp cond it was actually trained on and inference must stay
    # byte-faithful to it. Flip this to the training-range midpoint (e.g. 12.0)
    # ONLY once an R1-retrained (soft-edge) crop_mask CN becomes the live model.
    "outpaint_controlnet_edge_feather_px": 0.0,
    # "crop_mask" mode only: opt-in rounded-corner CN conditioning geometry
    # (Feature #3a, secondary lever). Passed as corner_radius_px to
    # build_crop_mask_condition; only takes effect when
    # outpaint_controlnet_edge_feather_px > 0.0 (it rounds the same inward
    # distance field the feather ramp is built from). Softens the 90-degree
    # VERTEX the CN's own conditioning shows it -- a companion to the
    # inference-side outpaint_controlnet_corner_gate_* pair below, which
    # instead attenuates the CN's OUTPUT near a corner. 0.0 (default) =
    # byte-identical to before this feature existed.
    "outpaint_controlnet_corner_radius_px": 0.0,
    # "crop_mask" mode only: per-corner ControlNet residual gate (Feature #2,
    # PRIMARY fix for the corner seam line; see H1 vertex-feature-lock,
    # core.utils.outpaint_corner_gate). Attenuates the CN residual ONLY in a
    # disk of this radius (canvas px) around each of the 4 placed-rect
    # vertices, down to outpaint_controlnet_corner_gate_min at the vertex
    # center; the rect EDGES (away from corners) stay at full CN residual
    # strength for cross-boundary continuation. 0.0 (default, paired with
    # outpaint_controlnet_corner_gate_min's default of 1.0) = the gate field
    # is never built and outpaint_controlnet_gate is left unset (None) for
    # crop_mask mode, exactly as before this feature existed.
    "outpaint_controlnet_corner_gate_radius_px": 0.0,
    # "crop_mask" mode only: the gate VALUE at each vertex center, in [0, 1]
    # (see outpaint_controlnet_corner_gate_radius_px above). 1.0 (default) =
    # no dip = disabled regardless of the radius.
    "outpaint_controlnet_corner_gate_min": 1.0,
    # "crop_mask" mode only: L1 four-corner x0-pin softening
    # (scratchpad/outpaint_seam_diagnosis.md). Softens the PER-STEP x0-pin
    # composite's keep-weight (NOT the CN residual gate above -- a different
    # mechanism) in a disk of this radius (canvas px) around each of the 4
    # placed-rect vertices, down to outpaint_pin_corner_relax_min at the
    # vertex center; the rect EDGES (away from corners) keep the full hard
    # pin. Blunts the re-entrant-corner latent seed that the hard rectangular
    # mask re-stamps every step (image_latents is fixed across all steps),
    # which the CN's structure-completion regime otherwise extends into a
    # seam line. 0.0/1.0 (default) = disabled = byte-identical; the preserved
    # rect stays byte-exact regardless via the final byte-exact paste, so this
    # only relaxes the intermediate latent trajectory near the corners.
    "outpaint_pin_corner_relax_radius_px": 0.0,
    "outpaint_pin_corner_relax_min": 1.0,
}

# ---------------------------------------------------------------------------
# Upscale (POST /generate/upscale)
# ---------------------------------------------------------------------------
# Shared diffusion keys derive from GENERATION_DEFAULTS; the remaining keys
# are owned by the upscale route.

UPSCALE_DEFAULTS: Dict[str, Any] = {
    "upscaler_backend": "spandrel",       # "pil" | "spandrel" | "rtx_vsr"
    "upscaler_model": None,               # filename in models/upscalers/ (required for spandrel)
    "scale_factor": 2.0,                  # 1.0-8.0
    "pil_resample": "lanczos",            # "lanczos" | "bicubic" | "nearest" (pil backend only)
    "tile_size": 512,                     # spandrel tiling; 0 = no tiling
    "tile_overlap": 32,                   # spandrel tile overlap px (feather blend)
    "rtx_vsr_quality": "high",            # "low" | "medium" | "high" | "ultra" (rtx_vsr only)
    "unsharp_enable": False,
    "unsharp_radius": 2.0,
    "unsharp_percent": 100,
    "unsharp_threshold": 3,
    # Diffusion tile upscale (upscaler_backend == "diffusion")
    "prompt": "",
    "negative_prompt": "",
    "diffusion_denoising_strength": 0.3,   # 0.05-0.9
    "steps": GENERATION_DEFAULTS["steps"],
    "cfg_scale": GENERATION_DEFAULTS["cfg_scale"],
    "sampler": GENERATION_DEFAULTS["sampler"],
    "schedule_type": GENERATION_DEFAULTS["schedule_type"],
    "attention_type": GENERATION_DEFAULTS["attention_type"],
    "attention_impl": GENERATION_DEFAULTS["attention_impl"],
    "seed": -1,                            # -1 = random
    "diffusion_pre_upscale_mode": "pil",   # "pil" | "model"
}

# ---------------------------------------------------------------------------
# User-overridable SLIDER BOUNDS registry
# ---------------------------------------------------------------------------
# Single source of truth for which UI slider/number-input UPPER BOUNDS a user
# may raise (or lower) from Settings, and how far. Served to the frontend via
# GET /schema/generation-defaults (key "param_bounds"); the frontend never
# hardcodes this list -- it drives both the Settings page's "Slider Bounds"
# card and every panel control's `resolveBound()` call (see
# frontend/src/utils/paramBounds.ts).
#
# THE RULE (decided; do not relitigate): a bound is user-overridable IFF
# exceeding it can only produce a worse or slower result, never a wrong or
# refused one. That means: UI conveniences this frontend invented for a
# slider/number-input's track (arbitrary "how far should this drag reach"
# choices) qualify. It excludes anything the BACKEND snaps or validates --
# frame grids (17n+5 / 8k+1), `pixel_align`, `min_frames`,
# `min_inference_steps`, `max_pixel_hw`, latent-chunk boundaries, keyframe
# indices -- because a value that violates one of those is not "worse", it is
# either silently snapped to a different value than the one requested or
# rejected outright by a real backend check, and this mechanism must never be
# used to paper over that with a bigger number.
#
# Support for the claim that raising the caps below cannot 400, only OOM: as
# of this registry's introduction there is not a single `Field(ge=/le=)` (or
# equivalent Form()/Query() range constraint) anywhere under `backend/api/`
# for a generation parameter. The only server-side generation-param range
# validators in the whole tree are `validate_video_geometry`,
# `validate_video_steps`, the `% pixel_align == 0` divisibility checks, and
# outpaint's `total_frames` grid check -- none of which this registry touches.
#
# Each entry: {"builtin": <today's literal>, "floor": <lowest a user may set>,
# "ceiling": <highest a user may set>, "family": <settings-card grouping>,
# "label": <settings-card row label>}. `builtin` MUST equal the literal it
# replaces at every wiring site (verified by tests) -- it is what "reset to
# default" (the row's own enable checkbox, unchecked) falls back to, and what
# `resolveBound()` uses when the user has no override on file.
#
# --- Adding a new overridable bound: registry entry + wiring, nothing else --
# 1. Add one entry below (builtin/floor/ceiling/family/label).
# 2. In the frontend, replace the control's bare literal `max` with
#    `resolveBound("your_bound_name", generationDefaults?.param_bounds,
#    sliderBounds, currentValue, archLimit)` (frontend/src/utils/
#    paramBounds.ts). No new Settings-page wiring is needed -- the "Slider
#    Bounds" card in frontend/src/app/settings/page.tsx renders one row per
#    registry entry generically (grouped by `family`), so a new entry appears
#    there automatically.
# That is the entire cost. No DB migration (the override lives in ONE JSON
# column, `UserSettings.slider_bounds`, keyed by bound name -- see
# backend/database/models.py), no new endpoint (POST /settings/generation's
# existing `slider_bounds` handling validates any key present in this
# registry against its own floor/ceiling).
#
# --- Why only ~7 bounds are exposed here, not the ~100 numeric knobs that
# exist across GENERATION_DEFAULTS/OUTPAINT_DEFAULTS/VIDEO_GEN_DEFAULTS ---
# The excluded set (roughly 100 fields: every spectrum_*, fbcache_*,
# seam_structure_*, boundary_relax_*, outpaint_seam_*/outpaint_commit_*/
# outpaint_controlnet_* knob, block-swap tuning, PiD tiling, etc.) is expert
# tuning surface for opt-in quality/acceleration features, not general-purpose
# slider tracks. They fail THE RULE for a different reason than the hard
# backend-validated bounds above: most of them are UNBOUNDED already (no
# `max` literal exists anywhere to override -- a NumberInput with no `max`
# prop, or a value typed rather than dragged), so there is no "raise the
# ceiling" action to register; the handful that do have a `max` prop are
# feature-specific strength/threshold dials (e.g. `seam_structure_max_area`
# capped at a fraction of the region, `pid_tile_native` capped by a decoder
# constant) whose real ceiling is architecture- or algorithm-derived, not a
# frontend convenience -- moving them here would misrepresent them as safe to
# raise arbitrarily. Someone who wants one added should follow the "Adding a
# new overridable bound" steps above; nothing about this registry's shape
# limits it to 7 entries.
PARAM_BOUNDS: Dict[str, Dict[str, Any]] = {
    # --- Canvas (txt2img/img2img width & height sliders) ---
    # Wired: Txt2ImgPanel.tsx, Img2ImgPanel.tsx. NOT YET WIRED: InpaintPanel.tsx
    # (off-limits to this change -- another session owns that file; its two
    # width/height literals at the time of writing still need the same
    # `resolveBound("image_width_max"/"image_height_max", ...)` call).
    "image_width_max": {
        "builtin": 2048, "floor": 512, "ceiling": 8192,
        "family": "canvas", "label": "Image width slider max",
    },
    "image_height_max": {
        "builtin": 2048, "floor": 512, "ceiling": 8192,
        "family": "canvas", "label": "Image height slider max",
    },
    # --- Sampling (txt2img/img2img/upscale-diffusion steps & CFG) ---
    "steps_max": {
        "builtin": 150, "floor": 20, "ceiling": 1000,
        "family": "sampling", "label": "Sampling steps slider max",
    },
    "cfg_scale_max": {
        "builtin": 30, "floor": 5, "ceiling": 100,
        "family": "sampling", "label": "CFG scale slider max",
    },
    # --- Video (txt2vid/img2vid frame-rate slider) ---
    "video_frame_rate_max": {
        "builtin": 60, "floor": 8, "ceiling": 240,
        "family": "video", "label": "Video frame-rate slider max",
    },
    # --- Upscale (spandrel/diffusion tile-size number input) ---
    "upscale_tile_size_max": {
        "builtin": 4096, "floor": 512, "ceiling": 16384,
        "family": "upscale", "label": "Upscale tile size max",
    },
}

# ---------------------------------------------------------------------------
# Video generation (POST /generate/txt2vid — LTX-2.3)
# ---------------------------------------------------------------------------
# Video-only keys, kept out of GENERATION_DEFAULTS (no overlap with the
# image txt2img/img2img/inpaint parameter set). Distilled LTX-2.3 defaults.
# Constraints enforced server-side: width % 32 == 0, height % 32 == 0,
# num_frames % 8 == 1.

VIDEO_GEN_DEFAULTS: Dict[str, Any] = {
    "prompt": "",
    "negative_prompt": "",
    "width": 768,                    # divisible by 32
    "height": 512,                   # divisible by 32
    "num_frames": 121,               # (num_frames - 1) % 8 == 0
    "frame_rate": 24.0,              # output FPS
    "num_inference_steps": 8,        # distilled default
    "guidance_scale": 1.0,           # distilled default (CFG effectively off)
    "seed": -1,                      # -1 = random
    "num_videos_per_prompt": 1,
    "max_sequence_length": 1024,     # Gemma-3 prompt token budget
    "audio_enable": True,            # mux the generated audio track into the mp4
    # AP1: block-swap generation. Number of transformer_blocks kept CPU-resident
    # (weights streamed to GPU during the denoise loop). 0 = disabled (stock
    # enable_model_cpu_offload path, transformer fully GPU-resident) and stays
    # the request default -- opt-in, not forced on -- since a machine that
    # already fits the model should not pay a swap cost it does not need.
    "blocks_to_swap": 0,
    # UI-ONLY (not a request field; not read by any /generate/* route): the
    # value the four video panels' Block Swap checkbox writes when a user
    # turns it ON, so that value lives in one place instead of a literal
    # repeated in Txt2Img/Img2Img/Inpaint/Outpaint panels.
    #
    # Measured on MiniMax-H3 (50 transformer blocks) at 768x1248x345 frames on
    # an RTX 6000 Ada, peak reserved VRAM and step time per `blocks_to_swap`:
    #   0  -> 36.139 GiB, 203.0 s/forward (swap off)
    #   20 -> 32.346 GiB (-10.5%), 203.1 s
    #   40 -> 28.518 GiB (-21%),   203.0 s
    #   49 (max; engages FF chunking) -> 24.510 GiB
    # Step time was flat across all of these (+1.4% end to end, at or inside
    # run-to-run noise): going through the block-loop wrapper is a fixed cost,
    # not a per-block transfer cost, so a small swap count pays the same fixed
    # cost as a large one while saving much less VRAM. 40 gets most of the
    # measured savings (-21%, vs -10.5% at 20) without also engaging the
    # max-value FF-chunking path (49), which is a distinct behaviour change.
    "blocks_to_swap_enabled_default": 40,
    # UI-ONLY (not a request field; not read by any /generate/* route): the
    # value Settings' "Video Frame Count Slider Track Max" checkbox seeds the
    # number box with when a user first checks it (backend UserSettings.
    # video_frame_slider_max, GET/POST /settings/generation). Deliberately
    # equal to VideoFrameCountSlider.tsx's own UNCAPPED_FRAME_SLIDER_CEILING
    # constant (the track's built-in reach on an architecture with neither a
    # hard `max_frames` nor a `trained_max_frames`), so checking the box
    # without changing the seeded value is a no-op against today's behaviour
    # -- the setting only does something once the user actually raises or
    # lowers it.
    "video_frame_slider_max_seed": 241,
    # AP2: First-Block-Cache (dynamic per-step trajectory-redundancy skip). Only
    # available when the transformer runs through Ltx2BlockLoopWrapper (either
    # because blocks_to_swap > 0, or the wrapper is force-attached for FBCache
    # alone). Mutually exclusive with Block Swap (a cache hit skips the block
    # loop, desyncing the per-block swap prefetch rotation) — disabled with a
    # warning if blocks_to_swap > 0.
    "fbcache_enable": False,
    "fbcache_threshold": 0.12,       # relative-L1 first-block-residual threshold
    "fbcache_warmup_steps": 1,       # always-compute the first N steps
    # Spectrum: Adaptive Spectral Feature Forecasting (training-free acceleration),
    # wrapper-hosted (Ltx2BlockLoopWrapper). Forecasts the PRE-CFG per-branch joint
    # (video, audio) transformer output from a Chebyshev fit over actual passes,
    # skipping the whole forward on forecast steps. Mutually exclusive with FBCache
    # (Spectrum takes precedence) and with Block Swap (disabled with a warning if
    # blocks_to_swap > 0). Same knobs as the image GenerationParams schema.
    "spectrum_enable": False,
    "spectrum_w": 0.5,
    "spectrum_w_decay": 0.0,
    "spectrum_delta_cap": 0.0,
    "spectrum_m": 4,
    "spectrum_lam": 0.1,
    "spectrum_warmup_steps": 3,
    "spectrum_window_size": 4,
    "spectrum_flex_window": 0.75,
    "spectrum_tail": 0.12,
    "spectrum_max_cache": 0,
    # AP3: fuses the output-tail Linear heads (`proj_out` / `audio_proj_out`) into the
    # sequence-chunked output-norm loop already used by every MiniMax-H3 forward
    # (`core.models.minimax_h3.adaln_chunking.chunked_norm_out`), so the loop writes
    # straight into the (much smaller) per-modality head outputs instead of
    # materializing the full `(1, S, hidden_size)` intermediate first. MiniMax-H3
    # only -- accepted-and-ignored elsewhere. NOT bit-exact (fusing changes the
    # output-projection GEMM's row count per call, and cuBLAS's tiling/reduction
    # order depends on it): measured against the unfused path on the real
    # w4a8 checkpoint, driving norm_out with hidden_states captured from an actual
    # forward -- max abs 1.46e-4 / 4.6e-6 relative to the output's own magnitude
    # (video head), roughly 30-40x fp32 machine epsilon. A synthetic isolate puts
    # it ~17x lower; the figure here is the one a generation actually sees. See
    # `adaln_chunking.py`'s "Head fusion" note.
    # Worth, measured at 768x1248x345 (97,159 packed rows) with 40 blocks
    # swapped: peak reserved 16.881 -> 14.934 GiB, peak allocated unchanged at
    # 13.173, and 216.4 s vs 219.0 s per forward -- inside the run-to-run spread.
    # Off by default because it is not bit-exact, not because it costs anything.
    # No effect below the chunking row budget, i.e. on short clips.
    "fuse_output_proj": False,
    # Per-generation component overrides (RP2b). Both unsupported on the LTX-2.3
    # video arch (accepted-but-ignored with a warning); kept here so the video
    # request schema carries the same keys as the image routes.
    # NOT the same thing as POST /models/load's text_encoder_file /
    # clip_projection_file, which choose which encoder is BUILT at load time
    # (MiniMax-H3) and therefore have no entry here.
    "vae_path": None,
    "text_encoder_path": None,
    # Transformer quantization. LTX-2.3 is in RUNTIME_INT8_ARCHS, so "int8"
    # converts the video DiT (and only the DiT -- not the Gemma-3 text encoder,
    # not the connectors) to the mixed int8/e4m3 weight-only layout in place,
    # once per model load. Every other value is accepted-but-ignored with a
    # warning (see arch_capabilities.ARCH_SUPPORTED_VALUES["ltx2"]). Same key and
    # same semantics as the image routes' GENERATION_DEFAULTS entry.
    "unet_quantization": None,
    # Which GEMM the already-quantized Linear layers use for THIS generation
    # (null = leave the process flags alone, "w8a8", "dequant"). Meaningful on
    # LTX-2.3 because ltx2 is in QUANTIZED_LINEAR_ARCHS: its loader swaps in
    # Int8Linear/Fp8Linear for a weight-only quantized transformer component, and
    # the in-place int8 conversion above produces the same classes. Same key and
    # same three values as the image routes' GENERATION_DEFAULTS entry.
    "quantized_gemm_mode": None,
    # Which attention BACKEND runs the kernel ("normal"/"native", "flash",
    # "sage", "tq"). Same key, same vocabulary and the same default as the image
    # routes' GENERATION_DEFAULTS entry, which is why it is read from there
    # rather than restated -- the string is normalized by ONE resolver
    # (core.attention.normalize_backend) for every architecture.
    #
    # Honored today by MiniMax-H3, whose vendored transformer routes attention
    # through the unified conduit. LTX-2.3 drives diffusers' own attention
    # dispatch and does not read it (accepted-and-ignored, like every other
    # per-arch parameter on these routes). Browser senders resolve the global
    # selector and transmit it explicitly; a direct API request that omits the
    # field uses this documented "normal" default.
    "attention_type": GENERATION_DEFAULTS["attention_type"],
    # Generation-time LoRA (same shape as every other arch's params["loras"]:
    # list of {"path": str, "strength": float, ...}). Applied today by
    # MiniMax-H3 (core.models.minimax_h3.minimax_h3_lora, hooked into
    # MiniMaxH3Mixin._generate_minimax_h3, which every one of this arch's video
    # entry points -- txt2vid/img2vid/ref2vid/outpaint/inpaint -- routes
    # through). LTX-2.3 has no LoRA loader on its video path at all: the key is
    # accepted and ignored there, with a warning when non-empty (see
    # api.arch_capabilities's "lora" feature).
    "loras": [],
}

# ---------------------------------------------------------------------------
# Studio timeline render defaults (SSOT)
# ---------------------------------------------------------------------------
# These are renderer policy defaults rather than diffusion-model parameters.
# They are still exposed through the generation-defaults schema so the Studio
# client does not need to duplicate limits or output choices.
STUDIO_RENDER_DEFAULTS: Dict[str, Any] = {
    "audio_enabled": True,
    "fit_mode": "cover",
    "video_lossless": False,
    "max_duration_seconds": 600.0,
    "max_assets": 128,
    "max_clips": 512,
    "max_upload_bytes": 2 * 1024 * 1024 * 1024,
    "max_total_input_bytes": 4 * 1024 * 1024 * 1024,
    "max_decode_pixel_frames": 20_000_000_000,
    # Bounds total OUTPUT work (`canvas width * height * timeline duration
    # frames`), independent of `max_decode_pixel_frames`, which only bounds
    # INPUT pixel-frames actually read from source assets. Before this
    # existed, a small still image held for a long duration on a large
    # canvas (e.g. one 320x240 image held for the full 600s duration limit
    # at 120fps on an 8192x8192 canvas) passed the decode budget while the
    # requested OUTPUT was ~120x larger than that budget -- an encode job an
    # order of magnitude past what `max_render_seconds` was sized for.
    # Deliberately the same order of magnitude as `max_decode_pixel_frames`:
    # decode and encode cost the same class of work per pixel-frame, so a
    # single shared ceiling is the simplest symmetric policy and keeps a
    # request that already respects the input budget from being rejected
    # for output reasons in the common case (few, short, native-resolution
    # clips at canvas resolution).
    "max_output_pixel_frames": 20_000_000_000,
    # Internal disk-space safety margin, NOT a claimed/measured encode
    # bitrate: `prepare_render_inputs()` multiplies this by the requested
    # output's pixel-frame count to reserve extra free disk space for the
    # rendered file before staging begins, so a submission that would leave
    # the render with nowhere to write its output fails fast instead of
    # filling the disk mid-encode. Chosen as an order-of-magnitude upper
    # bound for libx264 "medium" output at typical Studio-render bitrates,
    # not a benchmarked value.
    "output_bytes_per_pixel_frame": 0.02,
    "max_render_seconds": 1800.0,
}

# ---------------------------------------------------------------------------
# Per-architecture video defaults (SSOT)
# ---------------------------------------------------------------------------
# `VIDEO_GEN_DEFAULTS` above is LTX-2.3-shaped (768x512, 121 frames, 8 steps).
# A second video architecture needs different geometry, and a route-level
# `if arch == ...` is exactly what this file exists to prevent, so the
# difference is declared here as an OVERLAY and resolved by
# `video_defaults_for_arch`.
#
# The overlay is the whole mechanism: an architecture with no entry resolves to
# `VIDEO_GEN_DEFAULTS` unchanged, so LTX-2.3's behaviour is bit-identical to
# what it was before overlays existed. The video routes apply the resolved
# defaults to the fields a request OMITS (Pydantic's `model_fields_set` /
# `Form(None)` sentinels); the declared Pydantic/Form defaults stay the base
# values so the schema documents one stable shape.
VIDEO_GEN_ARCH_OVERLAYS: Dict[str, Dict[str, Any]] = {
    # MiniMax-H3. Geometry from the agreement between MiniMax's own release
    # (README: 24 fps, 768 short edge, 768x1344 area cap, both axes /32) and the
    # ComfyUI node defaults (1344x768, length 124, shift 12.0 / 3.0).
    "minimax_h3": {
        "width": 1344,
        "height": 768,
        # 17 * 7 + 5. Both the floor of the trained range and the length of
        # every official example request.
        "num_frames": 124,
        # Fixed by the model, not a preference: everything it generates and
        # conditions on lives on a 24 fps clock.
        "frame_rate": 24.0,
        # MiniMax publishes NO step count: their reproducible-768p scripts are
        # HTTP calls to their own server and expose no sampler knobs, and the
        # 50 in the diffusers examples is that library's generic template
        # default rather than a MiniMax figure. 20 is the community baseline and
        # is described as exactly that everywhere it is user-visible.
        "num_inference_steps": 20,
        # Guidance is distilled into the weights: the sampler takes no guidance
        # scale and there is no unconditional branch. Held at the base default
        # so `check_arch_capabilities` warns only on a NON-default value.
        "guidance_scale": 1.0,
    },
}


def video_defaults_for_arch(arch: Optional[str],
                            base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`base` (default `VIDEO_GEN_DEFAULTS`) with ``arch``'s overlay applied.

    The single resolver behind every video route's omitted-field handling and
    behind the `video_arch_overlays` block of `/schema/generation-defaults`, so
    the frontend and the backend resolve a per-arch video default the same way.

    An unknown or missing arch returns `base` unchanged (a copy), which is both
    the LTX-2.3 behaviour and the safe answer for a model whose type has not
    been resolved yet.
    """
    resolved = dict(base if base is not None else VIDEO_GEN_DEFAULTS)
    resolved.update(VIDEO_GEN_ARCH_OVERLAYS.get(arch or "", {}))
    return resolved


TXT2VID_DEFAULTS: Dict[str, Any] = dict(VIDEO_GEN_DEFAULTS)

# Image-to-video (POST /generate/img2vid — LTX-2.3). Same parameter set as
# txt2vid plus a first-frame keyframe supplied as an uploaded image (multipart).
# The LTX2ImageToVideoPipeline resizes the keyframe to (width, height) and pins
# it as frame 0 (conditioning_mask[:, :, 0] = 1); no extra first-frame-strength
# knob is exposed since the pipeline __call__ does not accept one.
IMG2VID_DEFAULTS: Dict[str, Any] = dict(VIDEO_GEN_DEFAULTS)
# OPTIONAL second keyframe, uploaded as a multipart file. MiniMax-H3's `fl2va`
# workflow conditions on the first and/or the last frame (0-2 visual anchors),
# so img2vid needs a way to send the second one; `image` stays the first frame.
# None = first-frame conditioning only, which is the whole of LTX-2.3's img2vid
# behaviour (it declares this key unsupported and warns when it is sent).
#
# Kept out of VIDEO_GEN_DEFAULTS on purpose: txt2vid has no keyframe upload at
# all, so the key exists only on the endpoint that can carry a file. The value
# recorded in `params` (and on the gallery row) is the uploaded FILENAME, not
# the bytes.
IMG2VID_DEFAULTS["last_frame_image"] = None
# WHERE the uploaded `image` sits on the generated clip's timeline, as a PIXEL
# frame index (MiniMax-H3 `fl2va`). 0 is the first frame -- the only placement
# LTX-2.3 has and the one every request made before placement existed -- and
# `-1` names the clip's last frame.
#
# The sentinel is not a convenience: `num_frames` is snapped server-side to the
# arch's own grid (17n+5 on MiniMax-H3), so a client cannot know the last
# frame's index at request time and "the last frame" has to survive the snap.
IMG2VID_DEFAULTS["input_image_frame_index"] = 0
# ADDITIONAL anchors, uploaded as multipart files, and their placements. The two
# lists are positional and must be the same length. None = no extra anchors,
# which is the whole of LTX-2.3's img2vid behaviour and was MiniMax-H3's until
# placement shipped.
#
# Kept out of VIDEO_GEN_DEFAULTS for the same reason `last_frame_image` is: only
# an endpoint that carries file uploads can have them. The value recorded in
# `params` (and on the gallery row) is the uploaded FILENAMES, not the bytes.
IMG2VID_DEFAULTS["keyframe_images"] = None
IMG2VID_DEFAULTS["keyframe_frame_indices"] = None
# An audio track the video is generated AGAINST (MiniMax-H3 `fl2va`), uploaded
# as a multipart file. Its rows are pinned at t = 1.0 for the whole clip -- the
# model's forward process is `x_t = t*x0 + (1-t)*noise`, so t = 1 is exactly
# clean -- and the sampler never writes them. None = the soundtrack is generated
# jointly with the video, which is every request before this existed and the
# whole of LTX-2.3's behaviour (it declares this key unsupported and warns).
#
# Kept out of VIDEO_GEN_DEFAULTS for the same reason the keyframe keys are: only
# an endpoint that carries file uploads can have it. The value recorded in
# `params` (and on the gallery row) is the uploaded FILENAME, not the samples.
IMG2VID_DEFAULTS["input_audio"] = None

# ---------------------------------------------------------------------------
# Omni-reference video (POST /generate/ref2vid — MiniMax-H3 `ref2va` only)
# ---------------------------------------------------------------------------
# Same parameter set as txt2vid plus the reference uploads: up to 9 images, 3
# videos (each optionally carrying its own soundtrack) and 3 audio clips, at
# most 12 files in total. Those limits are the released checkpoint's, not this
# repo's, and are validated server-side.
#
# Unlike img2vid this endpoint serves ONE architecture: LTX-2.3 has no
# omni-reference workflow at all, and MiniMax-H3 serves it from a SECOND
# transformer partition (`transformer_ref`), so a request needs the ref2va
# checkpoint loaded rather than merely a video model.
REF2VID_DEFAULTS: Dict[str, Any] = dict(VIDEO_GEN_DEFAULTS)
# How an IMAGE reference is sized before it is encoded, which is a real
# cost/fidelity choice rather than a preference, because a reference's rows ride
# through every sampling step:
#
#   "max"   - the released recipe (the diffusers `minimax-h3` setup block): each
#             image is put on a short edge of its OWN, 2048 for the released
#             checkpoint, upscaling included and with no area cap. An image
#             reference never binds the generated geometry.
#   "match" - ComfyUI's `ref_image_size="match"`: an aspect-preserving scale,
#             DOWN ONLY, to the generation's pixel area. Fewer reference rows,
#             so a shorter packed sequence.
#
# Video references are not affected: they are always put on the canvas their own
# aspect ratio resolves to, by the same rule the generated video follows.
REF2VID_DEFAULTS["reference_image_size"] = "max"
# The uploaded FILENAMES of the references, recorded in `params` (and on the
# gallery row) in packed order, exactly as img2vid records `last_frame_image` --
# the bytes never go near the database.
REF2VID_DEFAULTS["reference_images"] = None
REF2VID_DEFAULTS["reference_videos"] = None
REF2VID_DEFAULTS["reference_audios"] = None
# C5: keyframe anchors, laid out AFTER the reference blocks
# (h3_pipeline_ops.build_ref2va_packed_layout's keyframe_anchors). Same shape
# and defaults as IMG2VID_DEFAULTS's fields; None = no anchors, the pre-C5
# request. There is no `image`/`last_frame_image` pair here -- ref2vid has no
# single "the" keyframe, so every anchor goes through this positional pair.
REF2VID_DEFAULTS["keyframe_images"] = None
REF2VID_DEFAULTS["keyframe_frame_indices"] = None

# A raw-decoded uint8 RGB clip costs `width * height * 3` bytes PER FRAME
# (see `utils.video_utils.load_video_frames`), with no relationship to the
# compressed upload's size on disk -- a modest H.264 file can decode to tens
# of gigabytes. `/generate/outpaint/video`'s preserved clip(s) (`video`,
# `bridge_video`) are not bounded by any request field on a boundary-
# conditioned architecture (the preserved frames are pasted, not sampled, so
# `total_frames` does not apply to them -- see the route's `_decode_max_frames`
# comment), so nothing else stops an oversized upload from being decoded
# whole. This is a single system-wide RAM ceiling on that decode, in BYTES
# rather than frames, so it scales down automatically for a smaller canvas
# and does not need a second constant per resolution. 8 GiB budgets for one
# in-flight decode alongside the rest of the process's working set (model
# weights, other requests' buffers) without being so tight that an ordinary
# clip at a model's own max canvas (MiniMax-H3: 768x1344) gets refused --
# that works out to roughly 2100 frames (~87s at 24fps) at that canvas, far
# past any architecture's own production length.
MAX_VIDEO_UPLOAD_DECODE_BYTES: int = 8 * 1024 * 1024 * 1024

# A spatial-mask video inpaint request (`spatial_mask_manifest` on
# `/generate/inpaint/video`) allocates float32 buffers ON TOP OF the
# already-decoded uint8 clip that `MAX_VIDEO_UPLOAD_DECODE_BYTES` bounds --
# `core.inference.video_mask_timeline.build_spatial_mask_plan` rasterizes the
# full manifest span and holds it as `full_masks` (one float32 array shaped
# `[clip_frames, height, width]`) alongside a same-shape `range_array`
# produced by `rasterize_mask_timeline` inside it, and `composite_masked_frames`
# (called after) allocates its own output buffer over the same frame count.
# Measured with `tracemalloc` around `build_spatial_mask_plan` at three clip
# lengths (100/362/1000 frames, 768x1344 canvas -- MiniMax-H3's own max): the
# call's own peak was a REPRODUCIBLE 5.00x the raw uint8 clip's byte count
# (`clip_frames * width * height * 3`) at every length tested, not a value
# that drifts with clip length -- this is what `SPATIAL_MASK_GENERATION_PEAK_MULTIPLIER`
# below encodes.  This is a SEPARATE ceiling from `MAX_VIDEO_UPLOAD_DECODE_BYTES`,
# not a replacement for it: both are checked (the raw decode first, this one
# second, once the spatial-mask branch is known to be active), and this one is
# unconditionally tighter at the same canvas because of that 5x multiplier --
# a spatial-mask request is refused at roughly 1/5th the frame count a
# mask-free request of the same resolution is. Kept at the same 8 GiB order of
# magnitude as `MAX_VIDEO_UPLOAD_DECODE_BYTES` (same reasoning: room for the
# rest of the process's working set alongside one in-flight request) rather
# than a larger value, since this buffer is transient CPU RAM that exists
# purely to validate/build a plan before any GPU work starts.
MAX_SPATIAL_MASK_GENERATION_RAM_BYTES: int = 8 * 1024 * 1024 * 1024
SPATIAL_MASK_GENERATION_PEAK_MULTIPLIER: float = 5.0

# ---------------------------------------------------------------------------
# Video temporal outpaint (POST /generate/outpaint/video — LTX-2.3)
# ---------------------------------------------------------------------------
# Places a (optionally trimmed) input clip at a latent-frame offset inside a
# LONGER total timeline and generates the frames before/after, preserving the
# placed input frames byte-exact (see core.pipeline_backends.ltx2
# ._generate_vidoutpaint_ltx2 + core.pipeline.generate_vid_outpaint). Pure
# orchestration over the stock diffusers `LTX2ConditionPipeline` -- no new
# denoise loop. Non-÷32 inputs are preprocessed ONCE (center-crop/resize to a
# ÷32 working resolution) and the PREPROCESSED frames become the exact-
# preserved content (mirrors the image outpaint RESIZE convention).
OUTPAINT_VIDEO_DEFAULTS: Dict[str, Any] = {
    **VIDEO_GEN_DEFAULTS,
    # Total output timeline length (frames). Must satisfy (N-1) % 8 == 0,
    # same constraint as txt2vid/img2vid's num_frames.
    "total_frames": 121,
    # Where the (trimmed) input clip is placed within the output timeline, as
    # a PIXEL frame offset. Snapped server-side to the nearest valid latent
    # frame index's pixel start -- {0, 1, 9, 17, ..., 8m+1} -- since
    # LTX2VideoCondition.index addresses a LATENT frame, not a pixel frame.
    "input_offset_frames": 0,
    # Trim applied to the input clip BEFORE placement, in pixel frames. 0/0 =
    # no trim (use the whole clip, subject to fitting inside total_frames).
    "input_trim_start_frames": 0,
    "input_trim_end_frames": 0,
    # "regenerate" = use the model's own generated audio as-is (the input
    # clip's own audio is NOT preserved). "preserve_input" = after generation,
    # mux the input clip's ORIGINAL audio over its placed span, keeping the
    # model-generated audio outside that span, with a short crossfade confined
    # to the GENERATED side of each boundary so every input audio sample stays
    # exact. Only relevant when audio_enable is true; falls back to
    # "regenerate" (with a warning) if the uploaded clip has no audio stream.
    #
    # The base value is LTX-2.3's, where "regenerate" is a benign default: that
    # pipeline hands back a track spanning the WHOLE output timeline, so the
    # preserved span still carries (model-generated) sound. On an architecture
    # that generates audio only for the frames it generates, "regenerate"
    # leaves the preserved span SILENT -- see OUTPAINT_VIDEO_ARCH_OVERLAYS,
    # which is where that architecture's different default lives.
    "outpaint_video_audio_mode": "regenerate",
    # When true, write the MASTER file as FFV1-in-mkv (`-pix_fmt rgb24`, no
    # forced colorspace conversion) instead of libx264-in-mp4, so the pasted
    # input frames are BIT-EXACT after decode (empirically verified -- see
    # utils.video_utils.save_video_with_metadata's docstring for why plain
    # libx264 "-qp 0" is NOT actually bit-exact). Audio (when present) uses
    # FLAC instead of AAC in this mode. Trade-offs: much larger file size,
    # and the master is not browser-playable (FFV1 has no mainstream browser
    # decoder) -- a separate H.264 mp4 proxy is encoded alongside it for
    # gallery playback (see `preview_filename` on the response); the master
    # itself stays downloadable and untouched for archival/verification.
    "video_lossless": False,
    # OPTIONAL second uploaded clip, preserved at the END of the output
    # timeline, which turns the request into a BRIDGE: `video` is preserved at
    # the head, this one at the tail, and the span between them is generated.
    # Only an architecture whose TemporalSpec lists the `bridge` placement
    # (MiniMax-H3) accepts it; LTX-2.3 refuses it with that reason.
    #
    # Kept here rather than in VIDEO_GEN_DEFAULTS for the same reason
    # IMG2VID_DEFAULTS["last_frame_image"] is: only an endpoint that carries
    # file uploads can have it. The value recorded in `params` (and on the
    # gallery row) is the uploaded FILENAME, not the bytes.
    "bridge_video": None,
    # MiniMax-H3 ref2va only (extend_forward): optional image references on
    # top of the source clip, which is ALWAYS the sole video reference on
    # this endpoint (no reference_videos/reference_audios field here -- see
    # OUTPAINT_VIDEO_ARCH_OVERLAYS / the route's partition gate). Same sizing
    # choice and filename-only recording as REF2VID_DEFAULTS.
    "reference_image_size": "max",
    "reference_images": None,
}

# Per-architecture overlay for the keys that exist ONLY on the video-outpaint
# endpoint. `VIDEO_GEN_ARCH_OVERLAYS` above cannot carry them: it is merged into
# txt2vid/img2vid too, where `total_frames` does not exist.
#
# `total_frames`'s base 121 is a length on LTX-2.3's 8k+1 grid; MiniMax-H3
# cannot produce it. 248 is that architecture's default REQUEST -- roughly a
# doubling of its own 124-frame default clip -- and, like every other
# MiniMax-H3 video length, what actually runs is the value the generated span
# snaps to (the endpoint solves for the generated span, not for the total, and
# reports the effective output length in `warnings[]`).
#
# `outpaint_video_audio_mode`'s base "regenerate" is likewise an LTX-2.3
# semantic. MiniMax-H3 generates audio and video jointly for ONE span, so it
# produces audio only for the frames it generates: under "regenerate" the
# preserved span is left silent, which for the commonest request (extend a clip
# that has sound) means the ORIGINAL audio disappears from the output. That
# outcome is correct for the mode -- the mode means "do not carry the input's
# audio over" -- but it is the wrong thing to do by default, so this
# architecture defaults to "preserve_input" instead. Both modes stay
# selectable; only which one you get for free changes.
OUTPAINT_VIDEO_ARCH_OVERLAYS: Dict[str, Dict[str, Any]] = {
    "minimax_h3": {
        "total_frames": 248,
        "outpaint_video_audio_mode": "preserve_input",
    },
}


def outpaint_video_defaults_for_arch(arch: Optional[str]) -> Dict[str, Any]:
    """`OUTPAINT_VIDEO_DEFAULTS` resolved for ``arch``.

    Two overlays, in order: the shared video one (canvas, frame rate, steps)
    and then the outpaint-only one. Same contract as `video_defaults_for_arch`
    -- an unknown or missing arch returns the base map unchanged.
    """
    resolved = video_defaults_for_arch(arch, OUTPAINT_VIDEO_DEFAULTS)
    resolved.update(OUTPAINT_VIDEO_ARCH_OVERLAYS.get(arch or "", {}))
    return resolved


# ---------------------------------------------------------------------------
# Video temporal inpaint (POST /generate/inpaint/video — MiniMax-H3 fl2va)
# ---------------------------------------------------------------------------
# Regenerates ONE contiguous time range of an uploaded clip and preserves the
# rest. The output is the same length as the trimmed input, which is why there
# is no `num_frames`/`total_frames` key here and `num_frames` is removed from
# the inherited video map: the clip decides the length, and advertising a
# default for it would advertise a field the endpoint does not take.
#
# The length contract is also INVERTED relative to outpaint, which is why this
# is a second endpoint rather than a mode of that one: outpaint's `17n + 5` grid
# binds the GENERATED span, while here every frame of the clip has a row in the
# packed sequence, so it is the TRIMMED INPUT that must be a valid production
# length and the regenerated range may be any latent-group span.
INPAINT_VIDEO_DEFAULTS: Dict[str, Any] = {
    **{key: value for key, value in VIDEO_GEN_DEFAULTS.items() if key != "num_frames"},
    # The range to regenerate, in PIXEL frames of the trimmed clip: start
    # inclusive, end exclusive (Python-slice convention). Both are required --
    # there is no defensible default range -- so the `None`s here are what the
    # route's required Form fields document, not values it falls back to.
    "regenerate_start_frame": None,
    "regenerate_end_frame": None,
    # Trim applied to the uploaded clip BEFORE anything else, in pixel frames.
    # Same semantics as the outpaint endpoint's fields of the same name; the
    # TRIMMED length is what has to be a valid clip length.
    "input_trim_start_frames": 0,
    "input_trim_end_frames": 0,
    # "regenerate" = the audio rows are generated for the whole clip like any
    # other request, so the preserved video span carries generated audio that
    # need not match its visuals. "preserve_input" = the clip's own track is
    # pinned as conditioning across the whole clip (the shipped ia2v mechanism)
    # and muxed back verbatim. "regenerate_range" = the audio latents outside
    # the regenerate range are pinned as conditioning (the same row-permutation
    # mechanism "preserve_input" uses), so the audio inside the range is
    # generated with the surrounding original track as context; the preserved
    # spans are then spliced back from the input clip's own audio after decode,
    # the same way the pixel path pastes preserved frames over a pin that has
    # already been through the VAE. The pinned/free boundary snaps to the audio
    # latent grid, which is finer than the video latent-group grid, so it can
    # differ from the pixel range by up to one audio latent per side. If the
    # range covers the whole clip there is nothing to pin and this falls back
    # to plain "regenerate".
    #
    # The base value is "regenerate" because this map's base is architecture-
    # neutral; the one architecture that implements this endpoint overlays the
    # other value below.
    "inpaint_video_audio_mode": "regenerate",
    # FFV1-in-mkv master + H.264 mp4 proxy, as on the outpaint endpoint. The
    # preserved frames are exact at the frames handoff either way; this is
    # what carries that exactness into the FILE.
    "video_lossless": False,
    # Mirrors /generate/ref2vid's reference keys verbatim (same "max"/"match"
    # meaning, same filename-recording convention). Allowed on `ref2va`,
    # refused on `fl2va`/`hybrid` -- see
    # `resolve_minimax_h3_inpaint_reference_gate`.
    "reference_image_size": "max",
    "reference_images": None,
    "reference_videos": None,
    "reference_audios": None,
}

# Per-architecture overlay for the inpaint-only keys. MiniMax-H3 generates audio
# and video jointly for one span, so under "regenerate" the preserved frames
# come back with a soundtrack that was drawn against a regenerated timeline --
# the wrong thing to hand someone who expressed no preference about audio they
# already have. Both modes stay selectable; only which one you get for free
# changes.
INPAINT_VIDEO_ARCH_OVERLAYS: Dict[str, Dict[str, Any]] = {
    "minimax_h3": {
        "inpaint_video_audio_mode": "preserve_input",
    },
}


def inpaint_video_defaults_for_arch(arch: Optional[str]) -> Dict[str, Any]:
    """`INPAINT_VIDEO_DEFAULTS` resolved for ``arch``.

    Two overlays, in order: the shared video one and then the inpaint-only one.
    ``num_frames`` is dropped again after the shared overlay -- that overlay
    carries a clip length for the endpoints that ask for one, and this endpoint
    takes its length from the uploaded clip.
    """
    resolved = video_defaults_for_arch(arch, INPAINT_VIDEO_DEFAULTS)
    resolved.pop("num_frames", None)
    resolved.update(INPAINT_VIDEO_ARCH_OVERLAYS.get(arch or "", {}))
    return resolved

# ---------------------------------------------------------------------------
# Audio generation (POST /generate/txt2aud — ACE-Step 1.5 turbo)
# ---------------------------------------------------------------------------
# Audio-only keys, kept out of GENERATION_DEFAULTS (no overlap with the image
# txt2img/img2img/inpaint parameter set). Authoritative source:
# `core.pipeline_backends.acestep.AceStepMixin._generate_txt2aud_acestep`.

AUDIO_GEN_DEFAULTS: Dict[str, Any] = {
    "prompt": "",              # caption text (also accepted as "caption")
    "lyrics": "",
    "audio_duration": 30.0,    # seconds
    "seed": -1,                # -1 = random
    "inference_steps": 8,      # turbo distilled default
    "guidance_scale": 1.0,     # turbo is CFG-distilled; overridden to 1.0 if != 1.0
    "shift": 3.0,
    "sampler_mode": "euler",   # accepted for forward-compat; currently a no-op
    "vocal_language": "en",
    # Generation-time LoRA (same shape as every other arch's params["loras"]:
    # list of {"path": str, "strength": float, ...}). See
    # `core.pipeline_backends.acestep.AceStepMixin._load_lora_acestep`.
    "loras": [],
    # aud2aud (audio-to-audio / cover) stub keys -- not yet wired to a route
    # (Phase 4+). Kept here now so the schema key-set is stable once aud2aud
    # ships, avoiding a later SSOT migration.
    "reference_audio_path": None,
    "reference_audio_enable": False,
    "is_cover": False,
    "denoising_strength": GENERATION_DEFAULTS["denoising_strength"],
    # Weight-only quantization, shared with the image/video routes and carrying
    # the same meanings (see GENERATION_DEFAULTS). `acestep` is in
    # RUNTIME_INT8_ARCHS: "int8" converts the audio DiT's 392 Linear layers in
    # place, once per model load. The FP8 values are not implemented for this
    # architecture (arch_capabilities records that, and exempts "int8").
    "unet_quantization": GENERATION_DEFAULTS["unet_quantization"],
    # `acestep` is likewise in QUANTIZED_LINEAR_ARCHS -- its loader swaps in
    # Int8Linear/Fp8Linear for a quantized checkpoint -- so the quantized-GEMM
    # path selection governs real modules here.
    "quantized_gemm_mode": GENERATION_DEFAULTS["quantized_gemm_mode"],
}

TXT2AUD_DEFAULTS: Dict[str, Any] = dict(AUDIO_GEN_DEFAULTS)

# ---------------------------------------------------------------------------
# Per-architecture audio defaults (SSOT)
# ---------------------------------------------------------------------------
# `AUDIO_GEN_DEFAULTS`/`TXT2AUD_DEFAULTS` above are ACE-Step-shaped (30s,
# 8-step turbo sampler, `inference_steps`/`guidance_scale`/`shift`/
# `sampler_mode`). MiniMax Music 3 (design doc "Generation parameter
# contract") needs different values for the keys the two architectures share
# (`audio_duration`) and two keys ACE-Step has no equivalent of at all
# (`num_inference_steps`, a per-CHUNK flow-matching step count, not
# ACE-Step's per-song `inference_steps`; `flow_guidance_scale`, distinct from
# ACE-Step's `guidance_scale`). A route-level `if arch == ...` is exactly
# what this file exists to prevent, so the difference is declared here as an
# OVERLAY and resolved by `audio_defaults_for_arch` -- the audio equivalent
# of `VIDEO_GEN_ARCH_OVERLAYS`/`video_defaults_for_arch`.
#
# The overlay is the whole mechanism: an architecture with no entry (ACE-Step)
# resolves to `AUDIO_GEN_DEFAULTS` unchanged, so its behaviour is bit-identical
# to what it was before this overlay existed. `/generate/txt2aud` applies the
# resolved defaults to the fields a request OMITS (Pydantic's
# `model_fields_set`), mirroring exactly how the video routes apply
# `video_defaults_for_arch`.
AUDIO_GEN_ARCH_OVERLAYS: Dict[str, Dict[str, Any]] = {
    "minimax_music3": {
        # An UPPER BOUND, not a target: the autoregressive stage's language
        # model may emit its end-of-audio token before this is reached
        # (design doc "Generation parameter contract" / "AR early stop").
        "audio_duration": 60.0,
        # Per CHUNK (the flow-matching DiT's 200-frame windows), not per
        # song -- design doc's parameter table.
        "num_inference_steps": 30,
        # Flow-stage CFG. AR CFG (1.5) and top-k (50) are fixed by the
        # reference recipe and are deliberately NOT exposed here (design
        # doc's parameter table).
        "flow_guidance_scale": 1.7,
    },
}


def audio_defaults_for_arch(arch: Optional[str],
                            base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`base` (default `AUDIO_GEN_DEFAULTS`) with ``arch``'s overlay applied.

    The single resolver behind `/generate/txt2aud`'s omitted-field handling and
    behind the `audio_arch_overlays` block of `/schema/generation-defaults`, so
    the frontend and the backend resolve a per-arch audio default the same way.
    Same no-op contract as `video_defaults_for_arch`: an unknown or missing arch
    (including "acestep", which has no overlay entry) returns `base` unchanged
    (a copy).
    """
    resolved = dict(base if base is not None else AUDIO_GEN_DEFAULTS)
    resolved.update(AUDIO_GEN_ARCH_OVERLAYS.get(arch or "", {}))
    return resolved


# ---------------------------------------------------------------------------
# Audio-to-audio COVER (POST /generate/aud2aud — ACE-Step 1.5 turbo, img2img
# analog). Repaint (inpaint analog) is out of scope -- the vendored
# generate_audio has no repaint kwargs (see
# `core.pipeline_backends.acestep.AceStepMixin._generate_aud2aud_acestep`).
# ---------------------------------------------------------------------------
# Note: distinct from the `reference_audio_path` / `is_cover` /
# `denoising_strength` stub keys left in AUDIO_GEN_DEFAULTS above -- those
# predate this implementation and do not match its actual contract
# (multipart file upload; `cover_strength` is a step-count blend, not a
# denoise-strength/start-timestep knob). Authoritative source:
# `core.pipeline_backends.acestep.AceStepMixin._generate_aud2aud_acestep`.

AUD2AUD_DEFAULTS: Dict[str, Any] = {
    "prompt": "",              # caption text (also accepted as "caption") -- the ref is re-rendered under this
    "lyrics": "",
    "seed": -1,                # -1 = random
    "inference_steps": 8,      # turbo distilled default
    "guidance_scale": 1.0,     # turbo is CFG-distilled; overridden to 1.0 if != 1.0
    "shift": 3.0,
    # Fraction of steps that keep the reference's semantic context before
    # switching to a text2music-style (silence) context (step-count blend,
    # NOT an img2img start-timestep/partial-denoise knob). Higher = closer
    # to the reference.
    "cover_strength": 1.0,
    # aud2aud sub-mode: "cover" (re-render the whole reference under a new
    # caption/lyrics) or "repaint" (regenerate only [repaint_start,
    # repaint_end) seconds of the reference, keeping the rest). See
    # `core.pipeline_backends.acestep.AceStepMixin._generate_aud2aud_acestep`.
    "mode": "cover",
    "repaint_start": 0.0,      # seconds, repaint mode only
    "repaint_end": 0.0,        # seconds, repaint mode only
    "vocal_language": "en",
    # Generation-time LoRA (same shape as every other arch's params["loras"]:
    # list of {"path": str, "strength": float, ...}). See
    # `core.pipeline_backends.acestep.AceStepMixin._load_lora_acestep`.
    "loras": [],
    # Weight-only quantization; same two axes and same meanings as in
    # AUDIO_GEN_DEFAULTS above (and inherited from here by
    # OUTPAINT_AUDIO_DEFAULTS).
    "unet_quantization": GENERATION_DEFAULTS["unet_quantization"],
    "quantized_gemm_mode": GENERATION_DEFAULTS["quantized_gemm_mode"],
}

# Per-architecture overlay twin of `AUDIO_GEN_ARCH_OVERLAYS`, for
# `/generate/aud2aud`'s own key set. MiniMax Music 3 (design doc phase plan
# item 8, "repaint") needs three keys ACE-Step has no equivalent of:
# `music3_repaint_mode` (which of the two honest repaint mechanisms to run --
# see `core.pipeline_backends.minimax_music3.MiniMaxMusic3Mixin.
# _generate_aud2aud_minimax_music3`'s docstring), and the same
# `num_inference_steps`/`flow_guidance_scale` flow-stage keys `AUDIO_GEN_ARCH_
# OVERLAYS`/`OUTPAINT_AUDIO_ARCH_OVERLAYS` already overlay for this
# architecture's other two audio routes. `repaint_start`/`repaint_end` are
# already declared in `AUD2AUD_DEFAULTS` above (shared with ACE-Step's own
# repaint sub-mode) but carry a DIFFERENT meaning per MiniMax Music 3 sub-mode
# -- see that same docstring -- so they are not duplicated here.
AUD2AUD_GEN_ARCH_OVERLAYS: Dict[str, Dict[str, Any]] = {
    "minimax_music3": {
        # Which of the two honest repaint mechanisms to run -- "regenerate"
        # (AR-resume with a new tail from a point onward) or "rerender" (keep
        # the codes, redraw the flow stage over a window with a new seed).
        # "cover" (ACE-Step's other aud2aud mode) is never valid for this
        # architecture and has no default here -- `mode` itself (inherited
        # from AUD2AUD_DEFAULTS above, default "cover") must be explicitly
        # "repaint" for a MiniMax Music 3 request to be accepted at all.
        "music3_repaint_mode": "regenerate",
        # Per CHUNK (the flow-matching DiT's 200-frame windows), matching
        # AUDIO_GEN_ARCH_OVERLAYS/OUTPAINT_AUDIO_ARCH_OVERLAYS's identical
        # entries for this architecture's other two audio routes.
        "num_inference_steps": 30,
        # Flow-stage CFG; AR CFG (1.5) and top-k (50) stay fixed by the
        # reference recipe, same as txt2aud/extend.
        "flow_guidance_scale": 1.7,
    },
}


def aud2aud_defaults_for_arch(arch: Optional[str],
                              base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`base` (default `AUD2AUD_DEFAULTS`) with ``arch``'s overlay applied.

    Same no-op contract as `audio_defaults_for_arch`/`video_defaults_for_arch`:
    an unknown or missing arch returns `base` unchanged (a copy). Only
    `minimax_music3` has an overlay today; ACE-Step deliberately has none, so
    its defaults must stay bit-identical to `AUD2AUD_DEFAULTS`.
    """
    resolved = dict(base if base is not None else AUD2AUD_DEFAULTS)
    resolved.update(AUD2AUD_GEN_ARCH_OVERLAYS.get(arch or "", {}))
    return resolved


# ---------------------------------------------------------------------------
# Audio temporal outpaint (POST /generate/outpaint/audio — ACE-Step 1.5)
# ---------------------------------------------------------------------------
# Places a (optionally trimmed) input clip at a time offset inside a LONGER
# total_duration timeline and generates the audio before/and-or after it,
# preserving the placed input sample-exact to the decoded 48kHz/16-bit
# representation (see core.pipeline_backends.acestep
# ._generate_audoutpaint_acestep + core.pipeline.generate_aud_outpaint).
# This is the structural INVERSE of aud2aud's `mode="repaint"`: repaint holds
# everything OUTSIDE a window and generates INSIDE it; outpaint holds the
# window itself (the placed input) and generates OUTSIDE it (before AND/OR
# after). Derived from AUD2AUD_DEFAULTS (single source of truth) rather than
# duplicating literal values -- shares prompt/lyrics/seed/inference_steps/
# guidance_scale/shift/vocal_language/loras verbatim. `mode`/`cover_strength`/
# `repaint_start`/`repaint_end` are inherited for schema key-parity but are
# NOT accepted request parameters on /generate/outpaint/audio (outpaint has
# no cover/repaint sub-mode -- it always holds the placed span and generates
# outside it), mirroring OUTPAINT_DEFAULTS' inherited-but-unused width/height
# note above.
#
# "Exact" for audio (unlike video, which needed a lossless-encode toggle
# because H.264 is not bit-exact): the app's standard FLAC output is already
# lossless, and the placed input's waveform samples (after normalization to
# stereo/48kHz -- see AceStepMixin._acestep_normalize_stereo_48k) are
# re-spliced verbatim over its span after decoding (see
# AceStepMixin._acestep_apply_outpaint_waveform_splice), so no separate
# "lossless" flag is needed here -- the default output path is sample-exact
# to that decoded 48kHz/16-bit representation end to end. An upload that is
# not already 48kHz/16-bit stereo (e.g. 24-bit or 44.1kHz) is faithfully
# resampled/requantized once during normalization, not byte-identical to
# the original file.
OUTPAINT_AUDIO_DEFAULTS: Dict[str, Any] = {
    **AUD2AUD_DEFAULTS,
    # Total output timeline length, in seconds. Capped at 240s server-side
    # (silence-latent tiling supports arbitrary length, but very long extends
    # drift stylistically without lyrics/captions that cover the full span).
    "total_duration": 60.0,
    # Where the (trimmed) input clip is placed within the output timeline, in
    # seconds. Snapped server-side to the nearest 1/25s (the ACE-Step VAE's
    # latent frame rate) and clamped so the placed input fits inside
    # total_duration -- a warning is added if either adjustment occurs.
    "input_offset_sec": 0.0,
    # Trim applied to the input clip BEFORE placement, in seconds. 0/0 = no
    # trim (use the whole clip, subject to fitting inside total_duration).
    "input_trim_start_sec": 0.0,
    "input_trim_end_sec": 0.0,
}

# Per-architecture overlay twin of `AUDIO_GEN_ARCH_OVERLAYS`/
# `AUD2AUD_GEN_ARCH_OVERLAYS`, for `/generate/outpaint/audio`'s own key set
# (`total_duration`/`input_offset_sec`/`input_trim_start_sec`/
# `input_trim_end_sec`). Design doc phase plan item 7 ("Extend"): MiniMax
# Music 3 resumes the autoregressive stage from the frame-code sidecar rather
# than placing a clip on a fixed timeline, so it needs three keys ACE-Step has
# no equivalent of at all -- `extend_duration_sec` (an upper bound on how much
# MORE audio to add, distinct from ACE-Step's `total_duration`, which is the
# whole OUTPUT timeline length), `num_inference_steps` and
# `flow_guidance_scale` (the same flow-stage keys `AUDIO_GEN_ARCH_OVERLAYS`
# overlays for txt2aud). Deliberately NOT declared here: a default for
# `placement` -- `_generate_audoutpaint_minimax_music3`'s docstring
# ("Placement") requires it to be supplied explicitly with no fallback, since
# the only value the causal autoregressive stage can ever honor is
# 'extend_forward' and silently defaulting to it would hide that this is a
# structural limit, not an arbitrary starting choice.
OUTPAINT_AUDIO_ARCH_OVERLAYS: Dict[str, Dict[str, Any]] = {
    "minimax_music3": {
        # An UPPER BOUND, not a target -- same "duration is an upper bound"
        # semantics as `audio_duration`/`AUDIO_GEN_ARCH_OVERLAYS` above (the
        # autoregressive stage may emit its end-of-audio token earlier).
        # Smaller than txt2aud's 60s `audio_duration` default on purpose:
        # extend adds an INCREMENT onto an already-generated song rather than
        # a whole new one, and a smaller default keeps
        # `core.models.minimax_music3.pipeline.check_ar_resume_budget`'s
        # position-budget check comfortably inside the checkpoint's 10,240
        # position budget for a song that has already consumed part of it.
        "extend_duration_sec": 30.0,
        # Per CHUNK (the flow-matching DiT's 200-frame windows), matching
        # `AUDIO_GEN_ARCH_OVERLAYS["minimax_music3"]`'s txt2aud default --
        # extend's flow stage denoises the same window geometry.
        "num_inference_steps": 30,
        # Flow-stage CFG; AR CFG (1.5) and top-k (50) stay fixed by the
        # reference recipe, same as txt2aud.
        "flow_guidance_scale": 1.7,
    },
}


def outpaint_audio_defaults_for_arch(arch: Optional[str]) -> Dict[str, Any]:
    """`OUTPAINT_AUDIO_DEFAULTS` resolved for ``arch``.

    Two overlays, in order -- the shared aud2aud one (prompt/lyrics/seed/
    inference params) and then the outpaint-only one -- mirroring
    `outpaint_video_defaults_for_arch`'s two-overlay composition. Same no-op
    contract for an architecture with no entry (ACE-Step): resolves to
    `OUTPAINT_AUDIO_DEFAULTS` unchanged, with no
    `extend_duration_sec`/`num_inference_steps`/`flow_guidance_scale` keys at
    all (ACE-Step's backend never reads them, mirroring how those same two
    keys stay absent from `audio_defaults_for_arch("acestep")`).
    """
    resolved = aud2aud_defaults_for_arch(arch, OUTPAINT_AUDIO_DEFAULTS)
    resolved.update(OUTPAINT_AUDIO_ARCH_OVERLAYS.get(arch or "", {}))
    return resolved


# ---------------------------------------------------------------------------
# Video chain planning (POST /video-chain/plan, POST /video-chain/validate)
# ---------------------------------------------------------------------------
# Planner knobs for the long-form video Chain Manifest. Deliberately its own
# dict rather than extra keys in VIDEO_GEN_DEFAULTS: those keys are the request
# body of /generate/txt2vid & friends and are mirrored into the video panels'
# DEFAULT_PARAMS, while these describe a PLAN. Same separation as
# PROMPT_ASSIST_DEFAULTS.
#
# Two of them are also the wire defaults of `/generate/outpaint/video`'s
# `continuation_mode` / `continuation_overlap_frames` (see `requested_overlap_frames`
# below), because a continuation executes what the manifest planned: restating
# either value at the route would be a second source of truth for the same
# decision.
#
# Values shared with an actual generation request (fps, negative prompt, root
# seed) are read from VIDEO_GEN_DEFAULTS instead of restated, so a chain plan
# and a single-shot generation cannot drift apart.

VIDEO_CHAIN_DEFAULTS: Dict[str, Any] = {
    # Manifest/plan schema versions. Bumped only on a breaking manifest change;
    # a client holding an older manifest must re-plan rather than guess.
    "manifest_version": 1,
    "plan_schema_version": 1,
    # How the root prompt becomes per-segment prompts:
    #   "timeline"      - parse/normalize into a canonical timeline, then
    #                     compile one prompt per segment (deterministic for an
    #                     H3 structured prompt whose `[Shot N]` headers parse).
    #   "manual"        - the caller supplies the canonical timeline.
    #   "legacy_repeat" - today's behaviour: copy the root prompt verbatim into
    #                     every segment. Explicit opt-in, always warned about.
    # This default applies to a prompt the planner can parse structurally. For
    # free-form prose there is NO defensible default (no measured basis for
    # making the LLM extraction path automatic), so the planner refuses to pick
    # one: it returns a hard error asking the caller to choose a mode. Reading
    # this key does not authorize a silent fallback to it.
    "context_mode": "timeline",
    # How segment boundaries are chosen:
    #   "fixed"        - every segment is `requested_segment_frames` (or the
    #                    architecture's own maximum), the last one is whatever
    #                    is left.
    #   "shot_aligned" - `requested_segment_frames` becomes an UPPER BOUND and
    #                    the planner picks boundaries that split as few shots as
    #                    possible, merging shots shorter than one segment and
    #                    disclosing shots longer than one (design §7.2c).
    # None = the caller did not choose, and the planner resolves it from the
    # canonical timeline: shot-aligned when the clip contains a shot boundary to
    # align to, fixed otherwise (`resolve_segment_length_mode`, which owns that
    # rule). The condition is the TIMELINE, not whether a segment length was
    # requested: on an architecture with no single-inference maximum
    # (MiniMax-H3) a plan without `requested_segment_frames` is a single
    # segment, so a "default only when no length was asked for" rule would never
    # apply there (design §7.2d). An explicit `fixed` still gives uniform
    # segments.
    "segment_length_mode": None,
    # What a manifest that carries no mode is read as: everything planned before
    # shot alignment existed was fixed-length.
    "manifest_segment_length_mode": "fixed",
    # What happens to a shot/event whose frames cross a segment boundary:
    #   "refuse"                    - hard error naming the shot and the frame
    #                                 it is cut at. The shipped behaviour, and
    #                                 the default: a boundary-crossing shot is
    #                                 never resolved for the user silently.
    #   "assign_to_earlier_segment" - the earlier segment owns the whole shot
    #                                 and describes it in full; the later one
    #                                 does not restate it. Every crossing is
    #                                 disclosed as a warning with the shot
    #                                 number and its frame range.
    # Opt-in either way: with `segment_length_mode: fixed` the boundaries fall
    # where the frame arithmetic puts them, which the user cannot predict before
    # planning, so refusing without offering the choice leaves no route at all
    # for a timeline whose shots do not happen to land on them (design §6.2,
    # §10.3). `shot_aligned` already keeps whole the shots it could not fit,
    # since there the planner chose the boundaries itself.
    "boundary_crossing_policy": "refuse",
    # Per-segment noise seeding. "fixed" (every segment reuses the root seed)
    # is today's behaviour and keeps a text-only A/B honest by holding noise
    # constant. "derived" (stable hash of root_seed + plan_hash + segment index)
    # and "explicit" (per-segment values from the editor) ship alongside it as
    # opt-in. Whichever is chosen, concrete seeds are frozen into the manifest
    # at plan time and never re-drawn at run time.
    "seed_policy": "fixed",
    # -1 = draw once while planning and freeze the drawn value in the manifest.
    "root_seed": VIDEO_GEN_DEFAULTS["seed"],
    # What a continuation actually receives from its predecessor. Also the
    # `continuation_mode` default of /generate/outpaint/video. `boundary_frame`
    # (one shared anchor) and `pinned_tail` (the predecessor's decoded tail
    # pinned as the continuation's leading latent frames) are implemented;
    # richer modes are per-arch capabilities. An architecture that does not
    # advertise a mode gets a 400, never a guessed upgrade or a silent
    # downgrade.
    "continuation_mode": "boundary_frame",
    # Allowed |actual - planned| accumulated-frame divergence before the chain
    # pauses for a user decision. Frames, not seconds, so it is fps-independent.
    # Non-zero because re-decoding an mp4 does not always return the frame count
    # that was written; drift only ever shifts event ownership near a boundary,
    # never the local timestamps inside a segment prompt.
    "chain_drift_tolerance_frames": 12,
    # Requested shared frames between neighbouring segments, and the
    # `continuation_overlap_frames` default of /generate/outpaint/video. 0 = the
    # single shared anchor frame of a boundary-frame continuation; a pinning
    # mode requires a length on the video VAE's temporal group grid AND inside
    # the arch's advertised [chain_context_min_frames, chain_context_max_frames]
    # (MiniMax-H3: 5, 9, 13, 17 -- the floor is measured, see
    # arch_capabilities.MINIMAX_H3_PINNED_TAIL_MIN_FRAMES), and refuses anything
    # else rather than snapping it. The authoritative value is the generation's
    # effective_overlap_frames/_samples, never this request value.
    "requested_overlap_frames": 0,
    # How many of the predecessor's frames a `motion_preroll` continuation
    # places on its pre-roll as keyframe anchors, and the
    # `continuation_anchor_count` default of /generate/outpaint/video. 0 = none
    # requested, which is the only valid value for every other continuation
    # mode: sending a count with a mode that places no anchors is a 400, not a
    # silently dropped field. `motion_preroll` requires one inside the arch's
    # advertised [chain_motion_preroll_min_anchors, ..._max_anchors] and no
    # larger than the pre-roll itself. Where the anchors land follows from the
    # count and the pre-roll length (uniform spacing) and is fixed in the
    # manifest.
    "requested_anchor_count": 0,
    # Reference binding for a reference whose segment set is not stated
    # explicitly: apply it to every segment, which is today's carry-to-all
    # behaviour.
    "reference_binding_source": "default_all",
    # Plan request fields that have a meaningful "unset". null segment length
    # means "use the architecture's own maximum span"; null workflow means
    # "the loaded variant's default workflow".
    "requested_segment_frames": None,
    "workflow": None,
    "canonical_timeline": None,
    "references": [],
    "negative_prompt": VIDEO_GEN_DEFAULTS["negative_prompt"],
    "fps": VIDEO_GEN_DEFAULTS["frame_rate"],
}

# ---------------------------------------------------------------------------
# Video chain provenance (design §13) — carried BY a generation request
# ---------------------------------------------------------------------------
# Every value is None: "this generation is not a chain segment" is the only
# defensible default, and a chain segment supplies the whole set from the
# manifest it was planned from. They are stamps a chain runner attaches, never
# knobs a user edits, which is why they are their own dict instead of extra
# keys in VIDEO_GEN_DEFAULTS / TXT2VID_DEFAULTS / OUTPAINT_VIDEO_DEFAULTS:
# those maps are mirrored into the video panels' DEFAULT_PARAMS through
# `/schema/generation-defaults`, and nine permanently-null keys there would
# advertise editable parameters that do not exist. Separate from
# VIDEO_CHAIN_DEFAULTS for the mirror-image reason: those are planner-only
# knobs that never reach a `/generate/*` route, these ONLY reach one.
#
# Design §13 also fixes what is deliberately absent: the full root prompt and
# the canonical timeline live once in the manifest and are never copied onto a
# gallery row — `chain_plan_hash` / `chain_root_prompt_hash` are the references
# that stand in for them.
VIDEO_CHAIN_PROVENANCE_DEFAULTS: Dict[str, Any] = {
    # Which chain this segment belongs to, and the hash of the exact plan it
    # was compiled from (`compute_plan_hash`, the one definition — design §5.1).
    "chain_id": None,
    "chain_plan_hash": None,
    "chain_manifest_version": None,
    # Which segment this is, and how many the plan has. 0 = the first segment
    # (the txt2vid/img2vid/ref2vid request that starts a chain); every later
    # one is an outpaint-video continuation.
    "chain_segment_index": None,
    "chain_segment_count": None,
    # This segment's half-open [start, end) span on the chain's GLOBAL frame
    # timeline (the manifest's `owned_start_frame`/`owned_end_frame`).
    "chain_global_frame_start": None,
    "chain_global_frame_end": None,
    # How the segment prompts were produced ("timeline"/"manual"/
    # "legacy_repeat"), so a row records which planning mode wrote its prompt.
    "chain_context_mode": None,
    # sha256 of the chain's root prompt. The prompt ITSELF stays in the
    # manifest; this is what lets a row be tied back to it.
    "chain_root_prompt_hash": None,
}

# ---------------------------------------------------------------------------
# LoRA / Full-FT Training (TrainingRunCreateRequest)
# ---------------------------------------------------------------------------
# Authoritative source: backend Pydantic model.
# Frontend DEFAULT_PARAMS is aligned to these values.

TRAINING_DEFAULTS: Dict[str, Any] = {
    # Dataset
    "dataset_configs": [],
    "run_name": None,
    "training_method": "lora",
    "base_model_path": "",
    # Training parameters
    "total_steps": 1000,
    "epochs": 10,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "learning_rate": 1e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    # Plateau-then-cosine-floor LR scheduler ("plateau_cosine_floor"). Only
    # consumed when lr_scheduler == "plateau_cosine_floor"; harmless for all
    # other scheduler types.
    "lr_decay_start_ratio": 0.85,
    "lr_floor_ratio": 0.25,
    # Weight EMA (opt-in, default off). When enabled, a fp32 shadow copy of
    # the trainable params is maintained and saved as a separate, fully
    # loadable checkpoint alongside (not instead of) each normal checkpoint,
    # under a run_name with an "_ema" suffix.
    "use_ema": False,
    "ema_decay": 0.9999,
    # Only apply the EMA update every N optimizer steps (default: every
    # step). Decay is raised to the power N for the applied update so the
    # EMA's effective averaging horizon stays ~constant regardless of N.
    "ema_update_every": 1,
    # Where the EMA shadow tensors live: "cpu" (default, no extra VRAM, one
    # GPU->CPU sync per applied update) or "cuda" (no sync, costs ~one extra
    # copy of the trainable params in VRAM).
    "ema_device": "cpu",
    # Optimizer
    # Paging (bitsandbytes CPU offload of the optimizer state) is part of the
    # optimizer NAME -- "paged_adamw" / "paged_adamw8bit" / "paged_lion8bit" --
    # not a separate flag. There used to be an "optimizer_is_paged" boolean
    # here; it reached BaseTrainer and was read by nothing, because
    # OptimizerFactory selects a paged variant from the type string.
    "optimizer": "adamw8bit",
    "optimizer_cautious": False,
    "optimizer_beta1": 0.9,
    "optimizer_beta2": 0.999,
    "optimizer_epsilon": 1e-8,
    "optimizer_weight_decay": 0.01,
    "optimizer_schedule_free": False,
    "optimizer_schedule_free_r": 0.0,
    "optimizer_schedule_free_weight_lr_power": 2.0,
    "optimizer_use_radam": False,
    "optimizer_stochastic_rounding": False,
    # LoRA specific
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dtype": "fp32",
    "network_type": "lora",
    # ReLoRA
    "relora_merge_every": 500,
    "relora_merge_unit": "steps",
    "restart_warmup_steps": 100,
    "optimizer_reset_strategy": "full_reset",
    "optimizer_pruning_ratio": 0.9,
    # Checkpoints & sampling
    "save_every": 100,
    "save_every_unit": "steps",
    "max_step_saves_to_keep": None,
    "sample_every": 100,
    "sample_prompts": [{"positive": "", "negative": ""}],
    "resume_from_checkpoint": "latest",
    "sample_width": 1024,
    "sample_height": 1024,
    "sample_steps": 28,
    "sample_cfg_scale": 7.0,
    "sample_sampler": "euler",
    "sample_schedule_type": "sgm_uniform",  # Fix: frontend had "uniform"
    "sample_seed": -1,
    # Debug
    "debug_latents": False,
    "debug_latents_every": 50,
    # Bucketing
    "enable_bucketing": False,
    "base_resolutions": [1024],
    "bucket_strategy": "resize",
    "multi_resolution_mode": "max",
    # Epoch-dynamic crop augmentation (SDXL only). Per (item, epoch), two independent
    # axes pick how the image is presented: crop (full image vs random crop) and bucket
    # size (largest-fitting vs smaller). Re-bucketed each epoch; forces onthefly_gpu
    # latent encoding (disk/swap caches cannot represent per-epoch crops). Defaults
    # reproduce the previous behavior (disabled). See
    # docs/EPOCH_DYNAMIC_CROP_BUCKETING_DESIGN.md.
    "crop_augment_enable": False,            # master switch
    # Mix proportions (2x2 axes):
    "crop_full_image_prob": 0.7,            # P(full image, minimal crop only)
    "crop_max_bucket_prob": 0.7,            # P(largest-fitting bucket = least downscale)
    # Random-crop controls:
    "crop_min_area_ratio": 0.25,            # crop area >= ratio * original area
    "crop_min_short_side_px": 512,          # crop short side (original px) >= this
    "crop_aspect_mode": "source",           # "source" (keep image aspect) | "free" (any aspect)
    "crop_position_mode": "random",         # "random" (any point) | "corner" (touch a corner)
    # Smaller-bucket controls:
    "crop_smaller_bucket_mode": "base_res",  # "base_res" (use smaller base_resolution) | "scale_range"
    "crop_smaller_scale_range": [0.5, 0.9],  # downscale range when scale_range / single base_res
    # Full-image (minimal crop) position:
    "full_crop_position_mode": "center",    # "center" | "fixed_corner" | "random"
    # Conditioning + seed:
    "crop_microcond_mode": "kohya",         # time_ids semantics: "kohya" = original_size is full image
    "crop_plan_seed": 0,                    # 0 = derive from global training seed
    "cache_latents_to_disk": False,         # Fix: frontend had True
    # Component-specific
    "train_unet": True,
    "train_text_encoder": False,            # Fix: frontend had True
    "unet_lr": 1e-5,
    "text_encoder_lr": 1e-6,
    "text_encoder_1_lr": None,
    "text_encoder_2_lr": None,
    # Frontend-only fields now accepted by backend
    "train_image_encoder": False,
    "image_encoder_lr": None,
    "force_recache": False,
    "reconstruction_loss_weight": 0.0,
    # Precision
    "weight_dtype": "bf16",                 # bf16: works for both LoRA and full-FT
    "training_dtype": "bf16",               # bf16 needs no GradScaler (fp16 full-FT crashes on unscale_)
    "output_dtype": "fp32",
    "vae_dtype": "fp16",                    # Fix: frontend had "fp32"
    # Full-parameter save: also embed the VAE weights into the single-file
    # checkpoint. None = per-arch default (BUNDLE_VAE_DEFAULTS_BY_ARCH below):
    # sd15/sdxl/deus comfy-layout checkpoints are consumed by A1111/ComfyUI which
    # expect first_stage_model.* baked in, so they default True; all other archs
    # default False (their loaders fall back to default VAE resolution when the
    # section is absent). Ignored for LoRA and for pixel-space MiniT2I (no VAE).
    "bundle_vae": None,
    "mixed_precision": True,
    # Attention backend selector for training (single source of truth).
    # Values: "native" (SDPA) | "flash" (FlashAttention). "sage" is refused for
    # training (no backward kernel) and downgraded to native by resolve_backend.
    "attention_backend": "native",
    # DEPRECATED compat mirror of attention_backend: kept so existing YAML/presets
    # that only set the boolean still enable flash. Derived at parse time as
    # (attention_backend != "native"); the string key above is authoritative.
    # Do NOT remove until the deprecation/cleanup phase.
    "use_flash_attention": False,
    # Attention implementation selector for training (which REGISTRY runs the
    # attention kernel; orthogonal to attention_backend, which selects WHICH
    # kernel). "conduit" routes through the unified backend/core/attention
    # dispatch (new default; enables the tq backend in SDXL/SD1.5 training).
    # "diffusers" reproduces the pre-migration set_attention_backend path
    # byte-for-byte. TRAINING-ONLY this pass (FLUX.2/Ideogram4 not yet migrated).
    "attention_impl": "conduit",
    "min_snr_gamma": 5.0,
    # Text / latent encoding
    # text_encoding_mode: "swap_onthefly" | "pre_encoded_cache" | "onthefly_gpu"
    #                   | "cpu_prefetch"
    # cpu_prefetch pins the frozen TE on CPU and runs caption encoding for
    # upcoming batches on a daemon thread, in parallel with GPU train steps.
    # Stalls (queue empty when the trainer pulls the next batch) are logged
    # at epoch end so the user can see whether CPU encode keeps up.
    "text_encoding_mode": "swap_onthefly",
    "text_encoding_swap_interval": 256,
    "text_encoding_prefetch_depth": 4,
    "latent_encoding_mode": "swap_onthefly",
    "latent_encoding_swap_interval": 256,
    # Block swap
    "blocks_to_swap": 0,
    "use_pinned_memory": False,
    "sensenova_mot_phase_eviction": False,
    # SenseNova full fine-tune only: on-disk format of the saved model.
    # SENSENOVA_FULL_FINETUNE_SAVE_FORMATS below carries the measured sizes.
    "sensenova_full_finetune_save_format": "mixed",
    "block_swap_h2d_only": False,   # H2D-only swap (FLUX.2 LoRA training: no D2H of frozen base)
    "block_swap_ring_size": 2,      # GPU weight-buffer ring slots (>=1)
    "num_optimizer_groups": 0,
    # Per-bucket activation offload dispatcher. Predicts training peak per bucket
    # (static + coef * bs * latent_area) before the forward and offloads saved
    # activations to CPU only where it fits the VRAM budget. Proactive (no OOM
    # detection), so it works on Windows WDDM which spills instead of raising.
    "activation_dispatch_enable": False,
    "activation_dispatch_margin_gb": 1.0,
    "activation_dispatch_seed_coef": 24.0e-6,
    "activation_dispatch_residual_frac": 0.85,
    "activation_dispatch_threshold_mb": 4,
    # TREAD token routing (arXiv 2501.04765) — training-only acceleration.
    # Routes a random subset of tokens through a span of transformer blocks;
    # dropped tokens bypass the span (identity transport) and are restored at
    # re-entry. Inference/sampling always runs the full network. Currently wired
    # for the Anima DiT (other archs ignore these keys). Default OFF/neutral.
    "tread_enable": False,
    # Fraction of tokens dropped from the routed span, in (0, 1). Paper default
    # selection rate is 50% (drop_ratio 0.5).
    "tread_drop_ratio": 0.5,
    # Route span [start_block, end_block): tokens are dropped at start_block and
    # reintroduced at end_block. Defaults keep the first/last two of Anima's 28
    # blocks on all tokens and route the middle span (paper routes the bulk of a
    # DiT's depth, e.g. DiT-XL uses a single route r_{0,21}).
    "tread_start_block": 2,
    "tread_end_block": 26,
    # Low-rate stochastic depth (per-batch block dropout) — training-only
    # regularization. Each step, eligible transformer blocks (front/back, outside
    # a protected middle span) are independently dropped with prob block_skip_rate;
    # executed eligible blocks rescale their residual by 1/(1-rate) so the expected
    # contribution matches the full eval network. Sampling always runs every block.
    # Currently wired for the Anima DiT (other archs ignore these keys). Default OFF.
    # 0.0 = off; capped to 0.35 (high dropout on a pretrained DiT degrades quality).
    "block_skip_rate": 0.0,
    # Protected middle span [protect_start, protect_end): NEVER dropped. Block-removal
    # studies of pretrained DiTs show middle blocks are semantically critical while
    # early/late blocks tolerate removal — so only [0, protect_start) U [protect_end,
    # num_blocks) are eligible. Defaults protect Anima's middle 16 of 28 blocks
    # (eligible = first 6 + last 6).
    "block_skip_protect_start": 6,
    "block_skip_protect_end": 22,
    # DiT-BlockSkip (arXiv 2603.20755) — training-only MEMORY-REDUCTION feature
    # for LoRA and full fine-tune training. Skips the FIRST `blockskip_front` and
    # LAST `blockskip_back` transformer blocks; only the unskipped MIDDLE blocks
    # train (LoRA injected there, or full parameters when training_method=
    # full_finetune) — the paper's cross-attention masking study shows the middle
    # blocks are semantically critical. The skipped blocks are frozen (LoRA
    # variant: no adapter; full-FT variant: requires_grad_(False), excluded from
    # the optimizer); per step their contribution is captured once by a no_grad
    # full forward as a residual feature Delta (input->output of the skipped span)
    # and re-added during the gradient forward, so backprop only flows through the
    # middle blocks — eliminating the skipped blocks' backward-activation memory
    # (and, for full-FT, their gradient + optimizer-state memory). Currently wired
    # for the Anima DiT (other archs ignore these keys). Default OFF.
    # Mutually exclusive with TREAD, block_skip_rate (stochastic depth) and
    # blocks_to_swap; unsupported for ReLoRA and ControlNet.
    "blockskip_enable": False,
    # Number of leading / trailing blocks to skip (n + m). Defaults skip 4 + 4 of
    # Anima's 28 blocks (~29% skip ratio; the paper evaluates 30/40/50%). The
    # middle (num_blocks - front - back) blocks must be >= 1.
    "blockskip_front": 4,
    "blockskip_back": 4,
    # Resolution curriculum — training-only, arch-agnostic (data-pipeline feature).
    # Warm up at a lower resolution, then switch to the target (base_resolutions) at an
    # epoch boundary. Lower resolution => fewer latent tokens => much cheaper attention
    # per step (tokens scale with area; attention ~ tokens²), so warmup steps are far
    # faster. Default OFF. To take effect, set res_curriculum_warmup_steps > 0.
    "res_curriculum_enable": False,
    # Number of warmup steps at the scaled resolution. Rounds UP to the end of the epoch
    # that contains it (switch happens at an epoch boundary, so per-epoch batch planning
    # stays intact). 0 = no warmup (feature inert even if enabled).
    "res_curriculum_warmup_steps": 0,
    # Warmup resolution = base_resolutions * this scale, each snapped to the /64 grid the
    # bucket table is defined on (reuses the existing bucket-fit logic). 0.5 => half the
    # linear size => 1/4 the tokens. Must be in (0, 1).
    "res_curriculum_warmup_scale": 0.5,
    # MNT
    "multi_noise_timesteps": 1,
    "multi_noise_mode": "independent",
    "trajectory_blend_alpha": 0.7,
    "timestep_sampling": {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0},
    # Regularization
    "regularization_type": None,
    "snr_regularization_weight": 0.1,       # Fix: frontend had 0.0
    "snr_timestep_adaptive": True,
    "snr_penalty_mode": "relu",
    "energy_regularization_weight": 0.05,   # Fix: frontend had 0.0
    "energy_timestep_adaptive": True,
    "energy_penalty_mode": "abs",           # Fix: frontend had "under"
    "energy_normalize_by_pixels": True,
    # Unified framework
    "noise_process": "auto",
    "prediction_target": "auto",
    "strict_validation": False,
    # SDXL micro-conditioning: derive time_ids (original_size / crop_top_left /
    # target_size) from the real source image + bucketing/crop, instead of the legacy
    # all-equal-to-latent-size, crop=(0,0). Default on; off restores legacy behavior.
    "sdxl_micro_conditioning": True,
    # SDXL high-spec VAE migration: swap the VAE + resize U-Net conv_in/out to the new
    # latent channel count. "none"/"sdxl" = standard 4ch (unchanged). e.g. "flux1" (16ch).
    "sdxl_vae_type": "none",
    # SDXL Text Encoder swap: replace CLIP with an alternative encoder + trainable
    # bridge adapters. "none" = standard CLIP. e.g. "siglip2_text".
    "sdxl_te_type": "none",
    "sdxl_te_hidden_layer": -2,      # which TE hidden-states layer to tap (penultimate)
    "sdxl_te_max_len": 256,          # fixed token length the new TE is padded/truncated to
    "sdxl_te_train_encoder": False,  # False = freeze TE, train adapters only; True = train both
    # Vision encoder
    "use_reference_images": False,
    "vision_encoder_path": None,
    "train_vision_encoder": False,
    "vision_encoder_lr": None,
    "gradient_routing_ve": False,
    # Param tracking
    "param_tracking": False,
    "param_tracking_interval": 100,
    # ControlNet
    "controlnet_type": "standard",
    "controlnet_pretrained_path": None,
    "controlnet_init_from_unet": True,
    "lllite_conditioning_channels": 32,
    "lllite_rank": 64,
    "condition_preprocessors": None,
    "condition_cache_mode": "on_the_fly",
    # Outpaint-native conditioning (PART B)
    "conditioning_mode": "preprocessor",
    "outpaint_crop_min_area": 0.15,
    "outpaint_crop_max_area": 0.8,
    "outpaint_edge_anchor_prob": 0.34,
    "outpaint_corner_anchor_prob": 0.33,
    "outpaint_mask_channel": True,
    "outpaint_known_loss_weight": 0.3,
    "outpaint_seam_loss_boost": 0.0,
    # 1 = current byte-identical 1-cell generate-side ring (default). 2 adds a
    # second ring (one more max_pool2d dilation step outward) weighted at half
    # the boost increment of the first ring. No effect unless
    # outpaint_seam_loss_boost > 0.
    "outpaint_seam_ring_width": 1,
    # 0.0 (default) = off, byte-identical to current loss. >0 adds a cross-seam
    # error-continuity term (native prediction space, computed directly on
    # model_pred - target, no x0 reconstruction) before Min-SNR weighting.
    "outpaint_seam_grad_lambda": 0.0,
    # False = current byte-identical behavior (weighted loss .mean() over all
    # elements, so per-sample loss scale depends on rect area). True divides
    # each sample's weighted loss by that sample's mean weight, decoupling
    # per-sample scale from rect area.
    "outpaint_loss_normalize": False,
    # R1 (scratchpad/outpaint_boundary_structure_fix.md D3-R1): per-sample
    # RANDOMIZED softness of the crop_mask ControlNet conditioning's known/
    # unknown perimeter (build_crop_mask_condition's edge_feather_px, canvas px),
    # drawn per (epoch, image_path) by OutpaintControlPlanner.feather_for. Root
    # cause: position/size/aspect/anchor mode are already randomized per sample,
    # but the boundary is always razor-sharp -- the ControlNet learns to render
    # that hard rect perimeter as scene structure (a "frame"). Randomizing edge
    # softness removes the one invariant a frame-tracing shortcut can rely on.
    # Both 0.0 (default) -> feather_for always returns 0.0 -> the razor-sharp
    # default path in build_crop_mask_condition -> byte-identical to before this
    # feature existed. The doc's recommended non-default training range is
    # 0-24px; see outpaint_controlnet_edge_feather_px below for the matching
    # fixed inference value.
    "outpaint_edge_feather_min_px": 0.0,
    "outpaint_edge_feather_max_px": 0.0,
    # Pre-flight dataset drift check + optional rescan.  4 modes:
    #   "off"   — skip entirely (default)
    #   "path"  — only detect added/missing files
    #   "smart" — path drift + caption sidecar mtime check
    #   "force" — always rescan, no drift detection
    # Also cleans up orphan latent cache files when a rescan happens.
    # See core/training/dataset_drift.py.
    "rescan_before_training": "off",

    # ---- Anima (Cosmos-Predict2 DiT) training ----
    # LoRA targets enumerated by core/models/anima/anima_lora.py.
    # Comma-separated subset of {attention, mlp, mod, llm_adapter}; the
    # AnimaLoRAAdapter dispatch normalises and applies the corresponding
    # scope flags (see core/training/lora_trainer.py:_create_adapter).
    "anima_lora_scope": "attention,mlp,llm_adapter",
    "lens_lora_scope": "img_attn,txt_attn,img_mlp,txt_mlp",
    # Lens full-parameter LR multipliers per stream.
    "lens_img_lr_factor": 1.0,
    "lens_txt_lr_factor": 1.0,
    # Ideogram 4 (flow-matching DiT) LoRA: target scope and options.
    # scope tokens: attn (to_q/to_k/to_v/to_out), mlp (feed_forward), mod (adaln).
    "ideogram4_lora_scope": "attn,mlp",
    # Also LoRA-train the unconditional transformer (asymmetric-CFG branch) with an
    # auxiliary image-only loss. Default False (train the conditional branch only).
    "ideogram4_train_uncond": False,
    "ideogram4_uncond_loss_weight": 1.0,
    # LoRA learning-rate multiplier (applied to unet_lr).
    "ideogram4_lr_factor": 1.0,
    # If False, llm_adapter is dropped from the scope regardless of the
    # csv above. Defaults to True so the LLM Adapter is fine-tuned along
    # with the DiT blocks (Phase C user request).
    "train_llm_adapter": True,

    # ---- MiniT2I (pixel-space MM-JiT, flow matching, x0 prediction) ----
    # LoRA scope tokens: attn (qkv/attn_proj), mlp (w1/w2/w3), txt_embed
    # (txt_embedder/pooled_embedder). Enumerated by core/models/minit2i/minit2i_lora.py.
    "minit2i_lora_scope": "attn,mlp,txt_embed",
    # FLAN-T5 (text encoder) LoRA scope when train_text_encoder is set with LoRA:
    # attn (SelfAttention q/k/v/o), ff (DenseReluDense wi/wi_0/wi_1/wo).
    "minit2i_te_lora_scope": "attn,ff",
    # CFG label-drop rate: per-sample probability of zeroing the text mask so the
    # model sees the mask_token (unconditional) — matches the reference label_drop_rate.
    "minit2i_label_drop_rate": 0.1,
    # LoRA learning-rate multiplier (applied to unet_lr).
    "minit2i_lr_factor": 1.0,
    # Optional override path to FLAN-T5-Large; empty -> resolve next to the model.
    "minit2i_flan_t5_path": "",
    # From-scratch only: inherit compatible weights from an existing MiniT2I model
    # (same variant) instead of pure random init. Body/proj2/embedders copy fully;
    # in/out layers copy overlapping channels when patch is unchanged. Empty = off.
    "minit2i_scratch_init_from": "",
    # From-scratch inheritance: also copy the output head (final_layer.linear) from
    # the source model when shapes match. Default off = relearn the head from scratch.
    "minit2i_inherit_final_layer": False,

    # ---- Krea 2 (single-stream flow-matching MMDiT) training ----
    # LoRA scope tokens: attn (to_q/to_k/to_v/to_gate/to_out), mlp (SwiGLU ff),
    # text_fusion (internal text-fusion sub-transformer + projector, default off),
    # proj (img_in/txt_in/final_layer/time embeds, default off). The Qwen3-VL text
    # encoder is always frozen (no TE LoRA / no TE full-FT). Enumerated by
    # core/models/krea2/krea2_lora.py.
    "krea2_lora_scope": "attn,mlp",
    # LoRA learning-rate multiplier (applied to unet_lr).
    "krea2_lr_factor": 1.0,
    # Discrete flow-matching timestep shift applied to the sampled sigma
    # (sigma' = s*sigma / (1 + (s-1)*sigma)); musubi default 2.5 @1024^2. Set <=1 to disable.
    "krea2_discrete_flow_shift": 2.5,

    # ---- REPA (Representation Alignment) — MiniT2I only ----
    # Aligns a DiT intermediate hidden state with frozen clean-image patch features
    # from a vision encoder (our anime tagger, or off-the-shelf SigLIP2) via a
    # trainable MLP projector + cosine-similarity loss, to speed up convergence
    # (Yu et al., ICLR 2025, arXiv:2410.06940). Training-only; not in the saved model.
    "repa_enable": False,
    # Encoder source: "tagger" (domain-matched anime SigLIP2) or "siglip2".
    "repa_encoder_source": "tagger",
    # Tagger model directory (empty -> auto-pick newest usable under <repo>/tagger_models).
    "repa_tagger_model_dir": "",
    # Off-the-shelf SigLIP2 repo (used when repa_encoder_source == "siglip2").
    "repa_siglip2_repo": "google/siglip2-so400m-patch14-384",
    # DiT block depth to align (0-based). -1 = auto (depth_double // 3).
    "repa_align_depth": -1,
    # Alignment loss weight (lambda) added to the diffusion loss.
    "repa_weight": 0.5,
    # Projector LR multiplier (applied to unet_lr).
    "repa_proj_lr_factor": 1.0,
    # Encoder input square resolution; 0 = follow the encoder's native image_size.
    "repa_encoder_resolution": 0,

    # ---- Anima full-parameter training: per-group LR multipliers ----
    # Applied on top of unet_lr in AnimaFullParameterAdapter. Defaults of
    # 1.0 collapse to a single effective LR; users wanting sd-scripts-style
    # finer control (lower modulation LR etc.) can dial these.
    "anima_attn_mlp_lr_factor": 1.0,
    "anima_mod_lr_factor": 1.0,
    "anima_llm_adapter_lr_factor": 1.0,

    # Gradient checkpointing (activation recompute) on the trainable modules.
    # Default True preserves prior unconditional behavior (lower VRAM, ~slower);
    # set False to trade VRAM for speed. Gated at every enable call site in
    # base_trainer via self.gradient_checkpointing.
    "gradient_checkpointing": True,
    # ---- Anima Phase D: memory optimisations ----
    # Gradient-checkpoint mode for the DiT blocks. Both default to False
    # (i.e. standard GPU-resident checkpointing). When both are True the
    # async variant wins (faster CPU<->GPU overlap).
    "cpu_offload_checkpointing": False,
    "async_cpu_offload_checkpointing": False,
    # Quantise the *frozen* base DiT to FP8 before LoRA wrap. ~50% VRAM
    # reduction on the base. Only meaningful for training_method='lora'
    # (Full FT needs trainable base weights — flag is silently ignored).
    # None | "fp8_e4m3fn" | "fp8_e5m2".
    "fp8_base_dtype": None,
    # ---- MiniMax-H3: weight of the AUDIO half of the joint objective ----
    # H3 is a single stream: its packed sequence carries video AND audio rows and
    # every LoRA-targeted weight is shared by both, so a video-only objective
    # still moves audio behaviour. `loss = video_mean + audio_loss_weight *
    # audio_mean`, each modality's velocity MSE averaged over tokens, channels
    # and samples BEFORE weighting, so the weight's meaning does not depend on
    # the ~20x difference in row counts.
    #
    # 1.0 is the value the design's PRE-REGISTERED three-regime experiment
    # selected (Phase 0T, 200 steps per regime, fixed dataset and fixed
    # evaluation draws): joint real-audio loss reached a held-out VIDEO loss
    # 0.44 % BETTER than the video-only regime (0.295489 vs 0.296808 -- inside
    # the "not worse by 2 %" bar) while its held-out AUDIO loss was 19 % lower
    # (0.346591 vs 0.429390), which is rule 1 of the registered decision rule.
    # 0.0 reproduces a video-only objective; it is exposed because a dataset
    # whose audio is voiceover, music or editing artefacts is a real case, and
    # the experiment measured optimisation behaviour on one small fixed dataset,
    # not the quality of any particular corpus.
    # Ignored by every other architecture.
    "audio_loss_weight": 1.0,
    # ---- torch.compile (opt-in DiT training acceleration) ----
    # Wraps the DiT transformer's forward with torch.compile (Inductor) once,
    # after model/dtype/device + gradient-checkpointing setup, before the loop.
    # Values:
    #   "off"                        (default) — eager, no compile
    #   "default"                    — torch.compile(mode="default")
    #   "reduce-overhead"            — CUDA-graphs mode (higher VRAM)
    #   "max-autotune-no-cudagraphs" — autotuned kernels, no CUDA graphs
    # Gated to DiT archs (transformer is not None) + full-parameter FT;
    # skipped (with a log warning) for LoRA and when block swap is active
    # (blocks_to_swap > 0). Any compile/Inductor failure at runtime falls back
    # to eager. The forward is replaced in-place (module stays the same object),
    # so state_dict keys remain unprefixed and checkpoint saves are unaffected.
    "torch_compile": "off",
    # Dynamic-shape handling for torch.compile. None = auto (Dynamo detects
    # varying shapes and recompiles / promotes to dynamic as needed — best with
    # bucketing). True = force dynamic from the first compile (fewer recompiles,
    # slightly slower kernels). False = assume static (one specialised graph per
    # shape; more recompiles under bucketing).
    "torch_compile_dynamic": None,

    # ---- Online Danbooru augmentation (image-generation training) ----
    # Diffusion-side counterpart of the tagger's Danbooru augmentation. Text
    # conditioning accepts arbitrary tokens, so there is NO vocabulary
    # expansion here — the mechanism only fetches extra training images from
    # Danbooru and injects them as ordinary samples (interrupt-batch).
    # See core/training/danbooru_image_augment.py.
    "danbooru_aug_enable": False,
    # Static path: user-supplied general Danbooru queries (newline-separated;
    # each line is one query, Danbooru tag syntax, e.g. "1girl solo score:>=50").
    "danbooru_aug_queries": "",
    "danbooru_aug_weight_static": 1.0,
    # Deficiency path (auto + manual). Collects images for under-represented
    # tags to rebalance the dataset. No per-tag F1 (diffusion has none) — purely
    # dataset tag-frequency based.
    "danbooru_aug_deficiency_enable": True,      # auto: detect rare tags from dataset captions
    "danbooru_aug_deficiency_min_count": 20,     # tags in fewer than N dataset images are deficient
    "danbooru_aug_deficiency_top_k": 200,        # cap on the number of rarest tags targeted
    "danbooru_aug_deficiency_manual": "",        # manual: extra tags to top up (newline-separated)
    "danbooru_aug_weight_deficiency": 1.0,
    # Injection cadence / sizing.
    "danbooru_aug_injection_interval": 4,        # inject a Danbooru batch every N base batches
    "danbooru_aug_injection_ratio": 1.0,         # injection batch size = ratio x train batch_size
    # Collection / API settings.
    "danbooru_aug_min_score": 0,
    "danbooru_aug_max_posts_per_query": 200,
    "danbooru_aug_api_interval": 1.4,            # Danbooru TOS rate limit (seconds)
    "danbooru_aug_dl_speed_kbps": 500,
    # Download-speed safety: detect sustained throttling (Danbooru throttles before
    # a ban) and pause collection. Robust to transient dips (needs a slow streak
    # AND sustained duration before tripping). Manual resume from the UI.
    "danbooru_speed_check_enable": True,
    "danbooru_speed_degraded_kbps": 250,         # below this = "slow" sample
    "danbooru_speed_min_slow_streak": 8,         # consecutive slow downloads to trip
    "danbooru_speed_min_slow_seconds": 90,       # ...sustained at least this long
    "danbooru_speed_cooldown_seconds": 3600,     # pause duration on trip
    "danbooru_aug_buffer_size": None,            # None -> auto (max(32, 16 x batch_size))
    # Caption construction from the post's per-category tag fields.
    "danbooru_aug_include_rating_tag": False,    # prepend rating word (general/sensitive/...)
    "danbooru_aug_max_caption_tags": 0,          # 0 = keep all tags
    # Score-based quality tag (attach a quality tag derived from the post's
    # Danbooru score). Default thresholds follow the Animagine XL 3.0 convention;
    # override via danbooru_quality_tag_thresholds ("<min_score> <tag>" per line).
    "danbooru_quality_tag_enable": False,
    "danbooru_quality_tag_thresholds": "",       # "" = built-in Animagine XL 3.0 default
    "danbooru_quality_tag_attach_negative": False,  # also attach low/worst-quality tiers
    # Caption tag shuffle / dropout for injected samples. SEPARATE from the
    # per-dataset caption_processing (datasets vary per run by user intent).
    # Applied per-epoch via the same processor the datasets use. All default
    # off = fixed category order, no shuffle/dropout.
    "danbooru_aug_shuffle_tags": False,          # shuffle tags within each category
    "danbooru_aug_shuffle_keep_first_n": 0,      # keep first N tags fixed (no shuffle)
    "danbooru_aug_tag_dropout_rate": 0.0,        # per-tag dropout probability (0-1)
    "danbooru_aug_tag_dropout_keep_first_n": 0,  # keep first N tags fixed (no dropout)
    "danbooru_aug_caption_dropout_rate": 0.0,    # probability to drop the whole caption (0-1)
    "danbooru_aug_keep_tokens": 0,               # first N tokens immune to token dropout
}

# ---------------------------------------------------------------------------
# Per-architecture default timestep_sampling
# ---------------------------------------------------------------------------
# The training timestep distribution is drawn once per step by the shared
# TimestepSampler (core/training/timestep_sampler.py) and is fully user-overridable
# from the UI. Most models default to uniform [0,1]; only MiniT2I defaults to a
# logit-normal that reproduces its reference "lognorm" schedule.
#
# IMPORTANT — convention: the sampler emits t in [0,1]; each model interprets it.
# MiniT2I uses t=1 = data / t=0 = noise (inverted vs the flow models that use
# t=1 = noise). So for MiniT2I, logit_normal(mean=-0.8) concentrates t≈0.31 = the
# NOISE side (matching MiniT2IFlowMatchScheduler's mu=-0.8/sigma=0.8). Do not copy
# this sign to t=1=noise models without flipping it.
#
# Frontend (TrainingConfig.tsx) fetches this map and applies the selected model's
# entry when the base model changes (user edits still win); the backend
# (base_trainer.train) also resolves it when no timestep_sampling is supplied, so
# non-UI/API callers get the right default too. "_default" is the global fallback.
TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH: Dict[str, Any] = {
    "_default": {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0},
    # MiniT2I: t=1=data convention; mean=-0.8 biases toward the noise side.
    "minit2i": {"distribution": "logit_normal", "mean": -0.8, "std": 0.8,
                "min_timestep": 0.0, "max_timestep": 1.0},
    # Krea 2: uniform sigma sampling; the resolution schedule bias comes from the
    # discrete flow shift (krea2_discrete_flow_shift, applied in train_step_krea2).
    "krea2": {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0},
    # MiniMax-H3: uniform, and REGISTERED rather than left to fall through to
    # "_default", because for this architecture the sampler's output is not a
    # sigma -- it is the PRE-SHIFT uniform draw `u`, which train_step then puts
    # through the model's own two schedules (shift 12 for the video rows, shift 3
    # for the audio rows, from the SAME draw, exactly as inference does). Uniform
    # `u` therefore reproduces the sigma distribution the released model is
    # sampled at; a non-uniform distribution here COMPOSES with those shifts
    # instead of replacing them (train_step warns once when it sees one).
    "minimax_h3": {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0},
    # SenseNova uses t=0 noise / t=1 data. This APPROXIMATES, and is not identical
    # to, the released inference schedule: that one is linspace(0,1,steps+1) put
    # through _apply_time_schedule with timestep_shift=3.0
    # (SENSENOVA_GENERATION_DEFAULTS), i.e. sigma = 3*(1-t)/(1+2*(1-t)), whose
    # median is t=0.25; logit_normal(-0.8, 0.8) has median sigmoid(-0.8)=0.31.
    # Also note the training loss: SenseNova's velocity-space MSE equals
    # mse(x0_pred, x0)/(1-t)^2, so mass near t=1 is upweighted (train_runner
    # warns when a config departs from this entry toward t=1).
    "sensenova": {"distribution": "logit_normal", "mean": -0.8, "std": 0.8,
                  "min_timestep": 0.0, "max_timestep": 1.0},
}

# ---------------------------------------------------------------------------
# Per-architecture default bundle_vae (full-parameter save VAE embedding)
# ---------------------------------------------------------------------------
# bundle_vae=None (the TRAINING_DEFAULTS value) resolves per-arch here.
# sd15/sdxl/deus comfy-layout single files are consumed by A1111/ComfyUI, which
# expect first_stage_model.* baked in — a VAE-less save silently produces
# corrupt decodes there, so they default True. All other archs' loaders fall
# back to default VAE resolution when the section is absent, so they default
# False (smaller checkpoints, shared-store VAE reuse).
BUNDLE_VAE_DEFAULTS_BY_ARCH: Dict[str, bool] = {
    "_default": False,
    "sd15": True,
    "sdxl": True,
    "deus": True,
}


# Full fine-tune default for train_text_encoder, by architecture. Everywhere
# else that flag names a separate model that stays frozen unless asked for; on
# SenseNova it names the understanding half of the same MoT decoder that
# denoises, so the shared True would train 16.2 B parameters instead of 8.1 and
# dequantize 15.1 GiB of host RAM at load instead of 7.5
# (SENSENOVA_TRAINING_DESIGN.md 6.2 defaults to the generation half).
FULL_FINETUNE_TRAIN_TEXT_ENCODER_DEFAULTS_BY_ARCH: Dict[str, bool] = {
    "_default": True,
    "sensenova": False,
}


def resolve_full_finetune_train_text_encoder(value, arch: str) -> bool:
    """Resolve a possibly-None full-FT train_text_encoder to the per-arch default.

    An explicit boolean always wins; None looks up
    FULL_FINETUNE_TRAIN_TEXT_ENCODER_DEFAULTS_BY_ARCH.
    """
    if value is not None:
        return bool(value)
    return bool(FULL_FINETUNE_TRAIN_TEXT_ENCODER_DEFAULTS_BY_ARCH.get(
        arch, FULL_FINETUNE_TRAIN_TEXT_ENCODER_DEFAULTS_BY_ARCH["_default"]))


# Architectures whose full fine-tune runs with stochastic rounding regardless of
# optimizer_stochastic_rounding. This is a route requirement, not a default: the
# flag arrives as a materialized bool (routes.py declares it `bool`, and
# training_config only writes the YAML key when it is true), so an explicit False
# and an omitted key are the same value by the time a trainer sees it. Refusing
# on False would refuse every request; honouring it would run the route with
# round-to-nearest, where a measured 84.5% of a bf16 tensor's elements never move
# at any step count (SENSENOVA_TRAINING_DESIGN.md 6.3).
FULL_FINETUNE_FORCED_STOCHASTIC_ROUNDING_BY_ARCH: Dict[str, bool] = {
    "_default": False,
    "sensenova": True,
}


def full_finetune_forces_stochastic_rounding(arch: str) -> bool:
    """Whether ``arch``'s full fine-tune enables stochastic rounding by itself."""
    return bool(FULL_FINETUNE_FORCED_STOCHASTIC_ROUNDING_BY_ARCH.get(
        arch, FULL_FINETUNE_FORCED_STOCHASTIC_ROUNDING_BY_ARCH["_default"]))


# On-disk format of a SenseNova full-fine-tune checkpoint. The trained MoT half
# is dequantized to bf16 at load, so a save has to decide what to do with the
# other half and whether to put the trained one back on the int8 grid
# (SENSENOVA_TRAINING_DESIGN.md 6.4). Measured on the real checkpoint, trained
# half = generation:
#
#   mixed  25.12 GiB on disk, 25.43 GiB peak VRAM at inference
#   bf16   32.66 GiB on disk, 32.99 GiB peak VRAM at inference
#   int8   17.58 GiB on disk (VRAM unmeasured), and LOSSY -- see the adapter.
#
# With both halves trained there is no int8 half left, so "mixed" writes the
# "bf16" file and says so.
SENSENOVA_FULL_FINETUNE_SAVE_FORMATS = ("mixed", "bf16", "int8")


def resolve_bundle_vae(value, arch: str) -> bool:
    """Resolve a possibly-None bundle_vae config value to the per-arch default.

    An explicit user boolean always wins; None looks up BUNDLE_VAE_DEFAULTS_BY_ARCH.
    """
    if value is not None:
        return bool(value)
    return bool(BUNDLE_VAE_DEFAULTS_BY_ARCH.get(
        arch, BUNDLE_VAE_DEFAULTS_BY_ARCH["_default"]))

# ---------------------------------------------------------------------------
# Tagger Training (TaggerTrainingRunCreateRequest)
# ---------------------------------------------------------------------------

TAGGER_TRAINING_DEFAULTS: Dict[str, Any] = {
    "run_name": None,
    "vision_encoder_path": "",
    "training_method": "lora",
    "lora_rank": 32,
    "lora_alpha": 16.0,
    "learning_rate": 3e-4,
    "head_lr_multiplier": 10.0,
    "optimizer": "adamw8bit",
    "warmup_steps": 100,
    "epochs": 10,
    "batch_size": 32,
    "num_workers": 4,
    "num_workers_override": None,
    # Live tag-refresh: pick up tag edits made in the UI during training without
    # restarting. Detection runs on a background thread and workers apply changes
    # via a generation-gated mmap, so iteration speed is unaffected.
    "tag_refresh_enable": False,
    "tag_refresh_interval_seconds": 60,
    "save_every_n_steps": 500,
    "save_every_n_epochs": 0,
    "keep_last_n_checkpoints": 3,
    "checkpoint_save_mode": "lora",
    "mixed_precision": "bf16",
    # FlashAttention-2 for the SigLIP2 encoder self-attention. Requires
    # mixed_precision bf16/fp16 (FA2 cannot run in fp32); falls back to SDPA
    # otherwise. Only affects the encoder (pooling head is unaffected).
    "use_flash_attention": False,
    "gradient_checkpointing": True,
    "weight_decay": 1e-4,
    "loss_function": "asl",
    "loss_gamma_neg": 4.0,
    "loss_gamma_pos": 1.0,
    "loss_clip": 0.05,
    "loss_gamma0": 4.0,
    "loss_m0": 0.2,
    "loss_beta": 2.0,
    "loss_rho": 0.5,
    "loss_label_weight": "fisher",
    "validate_every": 1,
    "val_split": 0.05,
    "val_split_mode": "percent",
    "val_fixed_size": None,
    "save_best_only": False,
    "vocab_min_count": 10,
    # When building the vocabulary, resolve tag categories that are absent from
    # the local Danbooru taglist (taglist/) against the Gelbooru supplement
    # (taglist_gel/). Danbooru always takes precedence; Gelbooru only fills tags
    # Danbooru does not know, so enabling this is non-destructive and simply
    # reduces the number of "Unknown" category tags. Requires the taglist_gel/
    # directory to exist (silently skipped if absent).
    "vocab_use_gelbooru_categories": True,
    "excluded_categories": [],
    "ban_tags": "",
    "use_tag_aliases": False,
    "save_base_model": False,               # Fix: frontend had True
    # Quality-tag loss masking when a quality tag is present on a sample:
    #   "intra_group" (default) — mask sibling tags within the same group
    #                              (best/high/normal/medium share gradients;
    #                              low/bad/worst share gradients).  Trains
    #                              the cross-group good-vs-bad distinction.
    #                              Safer when intra-group prevalence imbalance
    #                              or annotation noise is significant.
    #   "cross_group"           — all non-positive quality tags train as
    #                              negatives.  Correct only when intra-group
    #                              labels are truly mutually exclusive.
    "quality_masking_mode": "intra_group",
    "cls_dim": None,
    "hidden_proj_dim": None,
    "init_head_from": None,
    # LR matrix (conditional inference) — built once at training start
    # when enabled.  See backend/core/tagger/lr_matrix_builder.py.
    "build_lr_matrix_on_start": False,
    "lr_top_anchors":            10000,
    "lr_top_targets":            1000,
    "lr_threshold":              1.0,
    "lr_min_anchor_count":       10,
    # Pre-flight dataset drift check + optional rescan.  4 modes:
    #   "off"   — skip entirely (default)
    #   "path"  — only detect added/missing files (~5 min for 3M items)
    #   "smart" — path drift + caption sidecar mtime check (catches
    #             in-place caption edits — adds a stat per sidecar)
    #   "force" — always rescan, no drift detection (most expensive)
    # See core/training/dataset_drift.py.
    "rescan_before_training": "off",
    # Training-time F1 metrics.
    # N2 (eval_every_n_steps) < N1 (threshold_search_every_n_steps).
    # N1 runs _find_best_threshold() on the rolling buffer (27 calls).
    # N2 runs _compute_f1_macro() once with the current threshold (1 call).
    "train_f1_eval_every_n_steps": 100,
    "train_f1_threshold_search_every_n_steps": 500,
    "train_f1_initial_threshold": 0.35,
    # Number of recent training batches kept in the rolling buffer.
    # Memory ≈ buffer_batches × batch_size × num_tags × 3 bytes (fp16 probs + bool labels).
    "train_f1_buffer_batches": 16,
    # Per-tag threshold metrics saved alongside each checkpoint as
    # ``{name}_tag_metrics.npz``.  Set to False to disable.
    # Memory: ≈ 4 × vocab_size × n_bins × 4 bytes (two epoch histogram slots).
    "save_tag_metrics": True,
    # Probability band considered "hard" for hard_rate computation.
    "hard_rate_lo": 0.25,
    "hard_rate_hi": 0.75,
    # Online Danbooru augmentation
    "enable_danbooru_augmentation": False,
    # ---- Query mode (first-class collection mode) ----
    # The static user-query path is now an independent, toggleable mode on equal
    # footing with vocab-expansion (surveyor) and low-F1. When query_expand is on,
    # the queries' tag tokens / wildcards are resolved via the Danbooru tags API
    # (name_matches) into concrete tags: vocab-absent ones are added to the
    # vocabulary, and ALL resolved tags are collected PER-TAG (round-robin within
    # the Query pool, bounded by danbooru_query_weight_static) so a wildcard that
    # resolves to N tags contributes N collection units — not one. With
    # query_expand off, the path keeps the legacy per-query-string image collection.
    "danbooru_query_enable": True,          # default on for backward compat; now independently toggleable
    "danbooru_query_expand_enable": False,  # resolve query tags -> per-tag collection + vocab expansion
    "danbooru_query_new_tag_min_count": 200,   # post_count floor for adding a resolved tag (main guard)
    "danbooru_query_resolve_top_k": 50,        # per-query cap on resolved tags (0 = unlimited)
    "danbooru_query_max_expanded_tags": 0,     # run-wide cap on NEW tags added via query (0 = unlimited)
    "danbooru_query_expand_categories": [0, 3, 4],  # eligible categories (general/copyright/character)
    "danbooru_query_resolve_interval": 3600,   # seconds between wildcard re-resolutions (API throttle)
    # Per-tag per-epoch collection cap (0 = unlimited). Bounds how many posts a
    # single resolved query tag contributes each epoch so a high-post_count tag
    # does not monopolise the injected batches. Reset each epoch (re-collection
    # next epoch is NOT blocked). Mirrors danbooru_cooc_collect_per_epoch.
    "danbooru_query_collect_per_epoch": 0,
    "danbooru_tags": "",              # newline-separated queries (use !tag or -tag to exclude)
    "danbooru_injection_interval": 4, # interrupt-batch every N base steps
    "danbooru_injection_batch_size_ratio": 1.0,  # 1.0 = full batch, 0.5 = half, etc.
    "danbooru_min_score": 0,
    "danbooru_max_posts_per_query": 200,
    "danbooru_api_interval": 1.4,
    "danbooru_dl_speed_kbps": 500,
    # Download-speed safety (throttle/ban avoidance): pause collection on sustained
    # slowdown; robust to transient dips; manual resume from the UI.
    "danbooru_speed_check_enable": True,
    "danbooru_speed_degraded_kbps": 250,
    "danbooru_speed_min_slow_streak": 8,
    "danbooru_speed_min_slow_seconds": 90,
    "danbooru_speed_cooldown_seconds": 3600,
    "danbooru_buffer_size": None,     # None → auto (2 × batch_size)
    "danbooru_vocab_expand": False,
    "danbooru_new_tag_min_count": 200,
    # Per-tag per-epoch collection cap for surveyor-discovered new tags (0 =
    # unlimited). Same rationale/semantics as danbooru_query_collect_per_epoch.
    "danbooru_new_tag_collect_per_epoch": 0,
    # Per-category post_count threshold overrides for the surveyor (category code
    # → min). Empty = use danbooru_new_tag_min_count for all. Lets copyright use a
    # higher bar than character (copyright post counts dwarf individual chars).
    "danbooru_new_tag_min_count_by_cat": {},
    "danbooru_new_tag_lookback_days": 90,
    "danbooru_new_tag_categories": [0, 3, 4],
    "danbooru_new_tag_survey_interval": 3600,
    # LRU cap on the dynamic new-tag query list (surveyor-discovered tags that
    # keep being collected across epochs/resumes). 0 = unlimited. When >0, the
    # least-recently-collected tag is evicted once the cap is exceeded.
    "danbooru_max_dynamic_tags": 0,
    # Collection-path weights (weighted random selection among available paths)
    "danbooru_query_weight_static": 1.0,   # user static queries
    "danbooru_query_weight_new_tag": 1.0,  # surveyor-discovered new tags (vocab expansion)
    "danbooru_query_weight_low_f1": 1.0,   # low-F1 existing vocab tags (deficiency collection)
    # Low-F1 deficiency collection: existing vocab tags whose per-tag F1 is below
    # threshold are collected. Targets recomputed at the train-F1 threshold-search cadence.
    "danbooru_low_f1_enable": False,
    "danbooru_low_f1_threshold": 0.5,  # tags with valid F1 below this are targeted (NaN excluded)
    "danbooru_low_f1_top_k": 500,      # cap on number of worst-F1 tags targeted
    "danbooru_low_f1_min_posts": 50,   # min Danbooru page-1 posts to include a tag (else skipped)
    # Per-tag per-epoch collection cap for low-F1 tags (0 = unlimited). Same
    # rationale/semantics as danbooru_query_collect_per_epoch.
    "danbooru_low_f1_collect_per_epoch": 0,
    # Train-count deficiency collection: rebalances under-exposed tags using the
    # cumulative per-tag training-exposure count (TagMetricsAccumulator.tag_count).
    # A tag is "under-exposed" when its actual cumulative count is well below what
    # its CURRENT per-epoch rate implies for the elapsed epochs — i.e. it joined
    # late or its image set grew mid-run. Catches the "new character went viral
    # and was added mid-training" case; complementary to low-F1 (performance) and
    # distinct from genuinely-rare-but-stable tags. Requires training-F1 metrics
    # cadence OR this flag for tag_count to keep accumulating.
    "danbooru_train_count_enable": False,
    "danbooru_train_count_top_k": 500,           # cap on number of worst-deficit tags targeted
    "danbooru_train_count_min_deficit_ratio": 0.3,  # target tags >= this fraction under-exposed
    "danbooru_train_count_min_per_epoch": 10,    # ignore tags with fewer exposures/epoch (noise floor)
    "danbooru_train_count_min_posts": 50,        # min Danbooru page-1 posts to include a tag (availability)
    "danbooru_train_count_collect_per_epoch": 0, # per-tag per-epoch collection cap (0 = unlimited)
    "danbooru_query_weight_train_count": 1.0,    # weighted-path weight for the train-count path
    # Score-based quality tag (attach a quality tag derived from the post's
    # Danbooru score to the label set; only trains tiers present in the vocab).
    # Default thresholds follow the Animagine XL 3.0 convention; override via
    # danbooru_quality_tag_thresholds ("<min_score> <tag>" per line).
    "danbooru_quality_tag_enable": False,
    "danbooru_quality_tag_thresholds": "",       # "" = built-in Animagine XL 3.0 default
    "danbooru_quality_tag_attach_negative": False,  # also attach low/worst-quality tiers
    # Co-occurrence vocab discovery: add vocab-absent tags that appear in
    # collected posts >= cooc_min_count times (category from the post fields,
    # filtered to danbooru_new_tag_categories). Created-at independent, so it
    # catches old tags missing from the training vocab. Requires vocab_expand.
    "danbooru_cooc_expand_enable": False,
    "danbooru_cooc_min_count": 50,
    # Categories eligible for co-occurrence discovery — independent of the
    # surveyor's categories, so a character-only surveyor net still lets the
    # accompanying copyright/general tags be added when they co-occur.
    "danbooru_cooc_categories": [0, 3, 4],
    # Co-occurrence ACTIVE collection: once a tag is promoted by cooc, also
    # collect it directly (order:random) so it trains across epochs — but only
    # lightly. A low weight + small per-epoch quota keeps it balanced and avoids
    # over-reinforcing the companion co-occurrence. weight 0 disables it (cooc
    # then only expands the vocab, as before).
    "danbooru_query_weight_cooc": 0.1,       # weighted-path weight (vs new_tag 1.0)
    "danbooru_cooc_collect_per_epoch": 50,   # per-tag per-epoch collection quota
    "danbooru_cooc_order_random": True,      # query with order:random for diversity
}

# ---------------------------------------------------------------------------
# VAE / autoencoder training (network.type == "vae_decoder")
# ---------------------------------------------------------------------------
# Phase 1 = decoder-only fine-tune with the encoder frozen, which is the shape
# of the only published, shipped recipe (stabilityai/sd-vae-ft-mse: decoder
# only, encoder frozen, MSE + 0.1*LPIPS). Consumed by
# core/training/vae/vae_config.resolve_vae_training_config, emitted into YAML by
# TrainingConfigGenerator.generate_vae_config, and served to the frontend by
# GET /schema/vae-training-defaults.
#
# The generic run-shape keys (batch_size ... max_step_saves_to_keep) are written
# into the existing process.train / process.save YAML sections so the existing
# routes, resume plumbing and checkpoint-keep field keep owning them; everything
# from vae_source onwards is written into a dedicated process.vae section.

VAE_TRAINING_DEFAULTS: Dict[str, Any] = {
    # --- run shape (emitted into process.train / process.save) -------------
    "batch_size": 1,
    "total_steps": 2000,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-5,
    "optimizer": "adamw",
    "optimizer_weight_decay": 0.001,
    "max_grad_norm": 0.1,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "seed": 42,
    "num_workers": 2,
    "save_every": 500,
    "max_step_saves_to_keep": 3,

    # --- base VAE selection (emitted into process.vae) ----------------------
    # "model" (default) = the run's own base_model_path, which the create route
    # already validates; "path" = an explicit diffusers directory or bare
    # .safetensors in vae_path; "store" = a shared vae_store key from vae_arch.
    # NOTE the store's "sdxl" entry is madebyollin/sdxl-vae-fp16-fix, whose fp16
    # safety comes from a weight rescaling that fine-tuning does not preserve --
    # the trainer warns when it detects that base, so it is not the default.
    "vae_source": "model",
    "vae_path": "",
    # vae_arch has TWO jobs, and both of them need it to be STATED rather than
    # assumed, which is why its default is the empty string ("not stated"):
    #   1. with vae_source="store" it names the store entry to load (refused
    #      when empty by vae_config._validate);
    #   2. for ANY base VAE that comes from a single file with no config.json,
    #      it is the only thing that can say which family the file belongs to.
    #      SD1.5 and SDXL VAEs are byte-for-byte the same shape, so
    #      from_single_file falls back to 0.18215 for both, and whatever ends up
    #      on vae.config is baked into every export by save_pretrained. A
    #      non-empty default here would mean "silently declare every unlabelled
    #      VAE to be that family" -- for a bare SD1.5 VAE under the old "sdxl"
    #      default that was a 1.40x latent-scale error written to disk. Empty
    #      makes the trainer REFUSE instead (vae_trainer.
    #      repair_single_file_scaling_factor), which is diagnosable.
    "vae_arch": "",

    # --- what to train ------------------------------------------------------
    "train_decoder": True,
    # all | up_blocks | mid_block | conv_out (ai-toolkit's blocks_to_train)
    "decoder_blocks": "all",
    # Encoder training is opt-in behind a DOUBLE GATE: train_encoder AND
    # acknowledge_latent_space_break must BOTH be true, or the run is refused.
    # Training the encoder moves the latent distribution, so every latent cache,
    # every LoRA and every diffusion model trained against this VAE stops
    # matching it. The resulting VAE is a new VAE, not a drop-in replacement:
    # it is exported to a differently-named directory, its provenance sidecar
    # records encoder_trained=true, and the bare-LDM export is refused for it.
    "train_encoder": False,
    "acknowledge_latent_space_break": False,
    # all | down_blocks | mid_block | conv_out — the encoder mirror of
    # decoder_blocks. Only read when train_encoder is true.
    "encoder_blocks": "all",

    # --- optimisation shape -------------------------------------------------
    "resolution": 512,
    # How much an image is resampled BEFORE the square crop is taken. Measured
    # to be the dominant control on what the fine-tune learns
    # (scratchpad/vae_training/results_crop_geometry.md):
    #   "downscale" - the historical behaviour: short side scaled to exactly
    #                 `resolution`, so 95.79% of the corpus is downscaled by a
    #                 median 2.30x. LANCZOS downscaling CONCENTRATES high
    #                 frequency (4.06x the top-octave power of a native crop,
    #                 n=300, t=+21.6), and the measured cost is calibration: the
    #                 fine-tune's accuracy gain is ~30% smaller on native content
    #                 (edge residual -7.7% vs -12.5%, 19/19, t=+7.49).
    #   "native"    - crop out of the full-size pixels; upscale only when the
    #                 short side is genuinely below `resolution` (4.21%).
    #   "mixed"     - draw the factor per sample, log-uniformly over [1, f_max]
    #                 (the study's recommendation). Costs nothing: the native
    #                 crop is 44% CHEAPER in the dataloader (17.4 vs 30.8 ms,
    #                 study §5.2; ratio reproduced in §8.4) and every policy has
    #                 8-30x headroom over the GPU. "mixed" itself is ~15% dearer
    #                 than "downscale" (§8.4), still ~10x the GPU's demand.
    # Default stays "downscale" so no existing run changes what it trains on.
    "crop_scale_policy": "downscale",
    # Upper bound on the per-sample downscale factor under "mixed"; 0 = the
    # image's own short/resolution ratio (no bound). Read ONLY under "mixed" —
    # a non-zero value with any other policy is REFUSED rather than ignored.
    "crop_scale_max_downscale": 0.0,
    # bf16 compute over an fp32 master copy of the weights. fp16 is refused
    # outright (SDXL-family activation overflow; no gradient scaler anywhere in
    # this trainer for the other families).
    "dtype": "bf16",
    "ema_enabled": True,
    "ema_decay": 0.999,

    # --- losses (design.md §5.1 as revised by the Phase-0 outcomes in §9.2) --
    "mse_weight": 1.0,             # ft-MSE's base term
    "l1_weight": 0.0,              # the LDM / ft-EMA reconstruction term
    "lpips_weight": 0.1,           # ft-MSE's weight; 1.0 would work against the goal
    "lpips_net": "vgg",
    "ycbcr_dc_weight": 0.1,        # PiD's colour-drift term (Charbonnier on YCbCr)
    "ycbcr_dc_y_weight": 0.25,
    "ycbcr_dc_chroma_weight": 1.0,
    "ycbcr_dc_eps": 1e-3,
    # Latent-cell grid-phase penalty. Default 0 (opt-in only): Phase 0 measured
    # the 8 px grid artifact at ratio ~1.0 on all four production VAEs under
    # three independent metric definitions, i.e. the defect it targets is not
    # present at a measurable level.
    "pattern_weight": 0.0,
    "pattern_size": 8,
    # Flat-region invented-HF penalty (L_invented). Default 0 (opt-in only).
    # Penalises high-frequency energy in the decode that a least-squares
    # projection onto the source's own high-frequency content cannot explain,
    # inside plane-fit flat/gradient windows. Unlike every other term in this
    # bank it is not an agreement-with-source objective: the projection
    # coefficient is detached, so the only way to reduce it is to emit less
    # unexplained energy, not to correlate more with the source.
    # The window geometry, the highpass basis, the projection epsilon, the
    # alpha clamp and the photometric weight are fixed internal constants of
    # the term (see core/training/vae/vae_losses.InventedHfLoss).
    "l_invented_weight": 0.0,
    "l_invented_y_weight": 1.0,
    "l_invented_chroma_weight": 0.25,
    # Plane-fit residual thresholds that decide which windows count as flat,
    # in 8-bit levels. A plane fit (not a variance test) is what admits smooth
    # gradients as "flat".
    "l_invented_flat_t_y": 2.0,
    "l_invented_flat_t_c": 1.25,
    # Posterior KL weight. This is LDM's 1e-6, and it means the same thing here
    # ONLY because the trainer divides the KL by the per-image element count
    # before weighting it: LDM pairs 1e-6 with a reconstruction term summed over
    # C*H*W, while every reconstruction term in this bank is mean-reduced over
    # B*C*H*W. Applied to a per-element reconstruction the raw 1e-6 would be
    # ~C*H*W (~8e5 at 512px) too strong -- measured at 15x the MSE -- and the
    # balance would shift 4x between 256 and 512px. With the normalisation the
    # knob is resolution-invariant. See vae_losses.VaeLossBank.forward.
    #
    # Only CONSTRUCTED when the encoder is trainable: under a frozen encoder the
    # term is a constant w.r.t. every trainable parameter, so it contributes no
    # gradient and is skipped entirely (the value is then ignored, and the
    # trainer logs that it is).
    "kl_weight": 1e-6,

    # --- export -------------------------------------------------------------
    # Additionally write the fine-tuned VAE as a bare LDM-format .safetensors
    # next to the diffusers directory. Off by default, and REFUSED when the
    # encoder was trained: a bare .safetensors carries no config.json, so the
    # consumer inherits scaling_factor / shift_factor from whatever model it is
    # plugged into — which is exactly what an encoder fine-tune invalidates.
    "export_bare_ldm": False,

    # --- validation (the only signal that a fine-tune is going wrong) -------
    "validation_every": 100,
    "validation_num_images": 8,
    # 1024, not 512. Validation is always a DETERMINISTIC centre crop under the
    # "downscale" policy regardless of crop_scale_policy (see
    # vae_dataset.make_validation_batch), so this is the only axis on which the
    # held-out metric can be made representative. 512 measures the most
    # flattering and least representative regime available (the fine-tune gains
    # +1.15 dB there vs +0.81 dB on native content); at 1024 the median 1131 px
    # source is downscaled only ~1.1x, i.e. nearly native, and native crops from
    # 512 to 1536 showed the PSNR advantage holding at +0.93..+1.20 dB with no
    # significant trend, so the change costs nothing in signal quality.
    # Changing it MID-RUN puts a step in the vae_val_psnr chart.
    "validation_resolution": 1024,
}
