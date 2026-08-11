from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.responses import Response, StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from typing import List, Optional, Dict, Any, Callable, Tuple, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
import contextvars
import os
import re
import sys
import json
import time
import subprocess
from PIL import Image
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor

from database import get_gallery_db, get_datasets_db, get_training_db, get_db  # Legacy
from database.models import GeneratedImage, UserSettings, Dataset, DatasetItem, DatasetCaption, TagDictionary, TrainingRun, TrainingCheckpoint, TrainingSample, TrainingPreset, TaggerTrainingRun, TaggerTrainingMetrics
from core.pipeline import pipeline_manager
from core.utils.taesd import taesd_manager
from core.extensions.lora_manager import lora_manager, LoRAAmbiguousIdentifierError
from core.extensions.controlnet_manager import controlnet_manager
from core.extensions.controlnet_preprocessor import controlnet_preprocessor
from core.extensions.tipo_manager import tipo_manager
from core.extensions.tagger_manager import tagger_manager
from core.training.training_config import TrainingConfigGenerator
from core.training.training_process import training_process_manager
from core.utils.tensorboard_manager import tensorboard_manager
from core.inference.schedulers import (
    get_available_samplers,
    get_sampler_display_names,
    get_available_schedule_types,
    get_schedule_type_display_names
)
from utils import save_image_with_metadata, create_thumbnail, calculate_image_hash, encode_mask_to_base64, extract_lora_names, calculate_file_hash, calculate_bytes_hash
from utils.upload_names import recover_upload_filename
from config.settings import settings
from api.websocket import manager
from auth import create_access_token, verify_credentials, require_auth
from api.param_defaults import (
    GENERATION_DEFAULTS, TXT2IMG_DEFAULTS, IMG2IMG_DEFAULTS, INPAINT_DEFAULTS,
    OUTPAINT_DEFAULTS, OUTPAINT_VIDEO_DEFAULTS, INPAINT_VIDEO_DEFAULTS,
    UPSCALE_DEFAULTS, TXT2VID_DEFAULTS, IMG2VID_DEFAULTS, REF2VID_DEFAULTS,
    TXT2AUD_DEFAULTS, AUD2AUD_DEFAULTS,
    OUTPAINT_AUDIO_DEFAULTS,
    TRAINING_DEFAULTS, TAGGER_TRAINING_DEFAULTS, VAE_TRAINING_DEFAULTS,
    TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH,
    BUNDLE_VAE_DEFAULTS_BY_ARCH,
    VIDEO_GEN_ARCH_OVERLAYS,
    OUTPAINT_VIDEO_ARCH_OVERLAYS,
    INPAINT_VIDEO_ARCH_OVERLAYS,
    PROMPT_ASSIST_DEFAULTS,
    video_defaults_for_arch,
)
from api.generation_utils import (
    process_controlnet_configs,
    create_progress_callback_factory,
    create_db_image_record,
    load_loras_for_generation,
    prepare_params_for_db,
    create_lora_step_callback,
    extract_model_info,
    extract_vision_encoder_info,
    extract_vae_info,
    extract_fp8_gemm_info,
    record_attention_backend,
    record_model_variant,
    sanitize_params_for_logging,
    set_prompt_chunking_settings,
    calculate_generation_metadata,
    apply_generation_timings,
    resolve_video_defaults,
    validate_video_geometry,
    validate_video_steps,
    plan_keyframe_placements,
    MINIMAX_H3_DOCUMENTED_ANCHOR_SCOPE,
)
from api.error_handlers import (
    GenerationError,
    ModelError,
    NotFoundError,
    ValidationError as CustomValidationError
)
from api.media_utils import get_or_create_preview, range_file_response, PREVIEW_DEFAULT_WIDTH
from core.extensions.minimax_h3_prompt_assistant import (
    PromptAssistError,
    PromptAssistOptions,
    build_template as build_h3_prompt_template,
    MiniMaxH3PromptAssistant,
    validate_prompt as validate_h3_prompt,
)

router = APIRouter()

minimax_h3_prompt_assistant = MiniMaxH3PromptAssistant(
    PROMPT_ASSIST_DEFAULTS["cache_max_entries"]
)

# Single source of truth for the API version, also used by main.py when
# constructing the FastAPI() app instance.
APP_VERSION = "0.1.0"

# Thread pool for running blocking operations
executor = ThreadPoolExecutor(max_workers=1)

# Cache for model list (to avoid re-scanning on every API call)
_models_cache: Optional[Dict[str, Any]] = None
_models_cache_timestamp: float = 0

# Cache for TensorBoard EventAccumulators to avoid re-reading event files on every request
# Key: (run_id, event_file_path), Value: (EventAccumulator, last_modified_time)
_event_accumulator_cache: Dict[tuple, tuple] = {}


def _validated_attention_type(value, default: Optional[str] = None) -> Optional[str]:
    """Validate `attention_type` against the attention registry's vocabulary.

    An unrecognized backend used to be absorbed into `native` by
    `normalize_backend`, with the only trace a console line deduped ONCE PER
    PROCESS -- so the generation succeeded, `warnings[]` was empty and the
    gallery row recorded a backend that had never run. That is a client error,
    and it is answered the way every other closed-vocabulary generation
    parameter in this module answers one (`quantized_gemm_mode`, `loop_decode`,
    `vae_tile_mode`): HTTP 400, raised BEFORE `start_generation` so no run is
    opened.

    The vocabulary is DERIVED from `core.attention` (registry keys + aliases +
    passthrough), never restated here: a hand-written backend list is exactly
    the drift this repo has been bitten by before.

    Empty / missing resolves to `default`, so a client that omits the field (or
    sends the empty string a `<select>` can produce) keeps the route's own
    `param_defaults` value instead of being refused.
    """
    from core.attention import validate_backend
    try:
        return validate_backend(value, default=default, param="attention_type")
    except ValueError as exc:
        raise CustomValidationError("Invalid attention_type value", detail=str(exc))


# Pydantic models for requests
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class AuthStatusResponse(BaseModel):
    auth_enabled: bool
    authenticated: bool = False
class LoRAConfig(BaseModel):
    path: str
    strength: float = 1.0
    apply_to_text_encoder: bool = True
    apply_to_unet: bool = True
    unet_layer_weights: Optional[dict] = None  # Per-layer weights
    step_range: Optional[List[int]] = [0, 1000]

class ControlNetConfig(BaseModel):
    model_path: str
    image_base64: Optional[str] = None  # Base64 encoded image
    strength: float = 1.0
    start_step: int = 0      # 0-1000, step number to start applying ControlNet
    end_step: int = 1000     # 0-1000, step number to end applying ControlNet
    layer_weights: Optional[dict] = None  # Per-layer weights like {"IN00": 1.0, ..., "MID": 1.0}
    prompt: Optional[str] = None  # Optional separate prompt
    is_lllite: bool = False
    is_reference_guide: bool = False  # Reference Guide mode: blend latent toward reference image
    preprocessor: Optional[str] = None  # Preprocessor type (auto-detected if None)
    enable_preprocessor: bool = True  # Whether to apply preprocessing
    # Training-free reference-style transfer (StyleAligned/VSP-style KV injection).
    # When true, this entry is NOT a real ControlNet: image_base64 is the style
    # reference image, strength maps to ref_k_strength, start_step/end_step gate
    # the denoise-loop step range. See core.inference.reference_style.
    is_style_transfer: Optional[bool] = None
    style_adain_strength: Optional[float] = None
    style_blocks: Optional[str] = None  # "lo-hi" block range string; None = all blocks
    # Deferred/advanced knobs (carried for parity; no-op at these defaults).
    style_low_scale_end: Optional[float] = None
    style_high_scale: Optional[float] = None
    style_beta: Optional[float] = None
    style_value_mode: Optional[str] = None
    style_value_adain_strength: Optional[float] = None
    style_ref_value_mix: Optional[float] = None
    style_late_release: Optional[float] = None
    style_rope_offset: Optional[int] = None

class AddTagRequest(BaseModel):
    tag: str
    category: str
    count: int = 1

class Txt2VidRequest(BaseModel):
    """Text-to-video generation request (LTX-2.3).

    Constraints (validated server-side): width % 32 == 0, height % 32 == 0,
    num_frames % 8 == 1.
    """
    prompt: str
    negative_prompt: Optional[str] = TXT2VID_DEFAULTS["negative_prompt"]
    width: int = TXT2VID_DEFAULTS["width"]
    height: int = TXT2VID_DEFAULTS["height"]
    num_frames: int = TXT2VID_DEFAULTS["num_frames"]
    frame_rate: float = TXT2VID_DEFAULTS["frame_rate"]
    num_inference_steps: int = TXT2VID_DEFAULTS["num_inference_steps"]
    guidance_scale: float = TXT2VID_DEFAULTS["guidance_scale"]
    seed: int = TXT2VID_DEFAULTS["seed"]
    num_videos_per_prompt: int = TXT2VID_DEFAULTS["num_videos_per_prompt"]
    max_sequence_length: int = TXT2VID_DEFAULTS["max_sequence_length"]
    audio_enable: bool = TXT2VID_DEFAULTS["audio_enable"]
    # AP1: block-swap generation (number of transformer_blocks kept CPU-resident).
    blocks_to_swap: int = TXT2VID_DEFAULTS["blocks_to_swap"]
    # AP3: fuses the output-tail heads into the chunked output-norm loop
    # (MiniMax-H3 only; not bit-exact -- see adaln_chunking.py's "Head fusion" note).
    fuse_output_proj: bool = TXT2VID_DEFAULTS["fuse_output_proj"]
    # AP2: First-Block-Cache (dynamic per-step trajectory-redundancy skip).
    # Mutually exclusive with Block Swap (see fbcache.py / ltx2.py).
    fbcache_enable: bool = TXT2VID_DEFAULTS["fbcache_enable"]
    fbcache_threshold: float = TXT2VID_DEFAULTS["fbcache_threshold"]
    fbcache_warmup_steps: int = TXT2VID_DEFAULTS["fbcache_warmup_steps"]
    # Spectrum (Adaptive Spectral Feature Forecasting). Mutually exclusive with
    # FBCache (Spectrum takes precedence) and with Block Swap (see ltx2.py).
    spectrum_enable: bool = TXT2VID_DEFAULTS["spectrum_enable"]
    spectrum_w: float = TXT2VID_DEFAULTS["spectrum_w"]
    spectrum_w_decay: float = TXT2VID_DEFAULTS["spectrum_w_decay"]
    spectrum_delta_cap: float = TXT2VID_DEFAULTS["spectrum_delta_cap"]
    spectrum_m: int = TXT2VID_DEFAULTS["spectrum_m"]
    spectrum_lam: float = TXT2VID_DEFAULTS["spectrum_lam"]
    spectrum_warmup_steps: int = TXT2VID_DEFAULTS["spectrum_warmup_steps"]
    spectrum_window_size: int = TXT2VID_DEFAULTS["spectrum_window_size"]
    spectrum_flex_window: float = TXT2VID_DEFAULTS["spectrum_flex_window"]
    spectrum_tail: float = TXT2VID_DEFAULTS["spectrum_tail"]
    spectrum_max_cache: int = TXT2VID_DEFAULTS["spectrum_max_cache"]
    # Per-generation component overrides (RP2b). Unsupported on the LTX-2.3 video
    # arch: accepted-but-ignored with a warning (see check_arch_capabilities).
    vae_path: Optional[str] = TXT2VID_DEFAULTS["vae_path"]
    text_encoder_path: Optional[str] = TXT2VID_DEFAULTS["text_encoder_path"]
    # Transformer quantization. Only "int8" is applied on LTX-2.3 (one-time
    # in-place conversion of the video DiT); other values warn and are ignored.
    unet_quantization: Optional[str] = TXT2VID_DEFAULTS["unet_quantization"]
    # Quantized-GEMM path selection for this generation (null / "w8a8" /
    # "dequant"). ltx2 is in QUANTIZED_LINEAR_ARCHS, so this really selects
    # something here; see api/quantized_gemm.py.
    quantized_gemm_mode: Optional[str] = TXT2VID_DEFAULTS["quantized_gemm_mode"]
    # Attention backend for this generation ("normal"/"flash"/"sage"/"tq").
    # Honored by MiniMax-H3 (its vendored transformer runs the unified conduit);
    # LTX-2.3 uses diffusers' own dispatch and ignores it. Omitted -> the process
    # setting.
    attention_type: str = TXT2VID_DEFAULTS["attention_type"]
    # Training-free reference-style transfer (video self-attention KV
    # injection; see core.inference.style_ltx2). An entry with
    # is_style_transfer=true carries the style reference; extracted by
    # process_controlnet_configs() into params["style_transfer"] (not a
    # ControlNet). No image-conditioning ControlNets exist for LTX-2.3 today --
    # this field exists ONLY to carry the style-transfer entry.
    controlnets: Optional[List[ControlNetConfig]] = []
    # Generation-time LoRA. Applied by MiniMax-H3 (see
    # core.models.minimax_h3.minimax_h3_lora / MiniMaxH3Mixin._load_lora_minimax_h3,
    # hooked into every one of this arch's video entry points via
    # _generate_minimax_h3). LTX-2.3 has no LoRA loader on its video path at
    # all -- accepted and ignored, with a warning when non-empty (see
    # api.arch_capabilities's "lora" feature).
    loras: Optional[List[LoRAConfig]] = TXT2VID_DEFAULTS["loras"]


class Txt2AudRequest(BaseModel):
    """Text-to-audio (music) generation request (ACE-Step 1.5 turbo).

    Standalone request model (does not extend GenerationParams -- audio has
    no width/height/steps/cfg_scale/sampler concept). See
    `core.pipeline_backends.acestep.AceStepMixin._generate_txt2aud_acestep`
    for how each field is consumed.
    """
    prompt: str = TXT2AUD_DEFAULTS["prompt"]
    lyrics: Optional[str] = TXT2AUD_DEFAULTS["lyrics"]
    audio_duration: float = TXT2AUD_DEFAULTS["audio_duration"]
    seed: int = TXT2AUD_DEFAULTS["seed"]
    inference_steps: int = TXT2AUD_DEFAULTS["inference_steps"]
    guidance_scale: float = TXT2AUD_DEFAULTS["guidance_scale"]
    shift: float = TXT2AUD_DEFAULTS["shift"]
    sampler_mode: str = TXT2AUD_DEFAULTS["sampler_mode"]
    vocal_language: str = TXT2AUD_DEFAULTS["vocal_language"]
    loras: Optional[List[LoRAConfig]] = TXT2AUD_DEFAULTS["loras"]
    # Weight-only quantization. "int8" converts the ACE-Step DiT in place once
    # per model load (`AceStepMixin._acestep_runtime_int8`, RUNTIME_INT8_ARCHS);
    # the FP8 values are not implemented for this architecture.
    unet_quantization: Optional[str] = TXT2AUD_DEFAULTS["unet_quantization"]
    # Which GEMM an already-quantized Linear runs (`acestep` is in
    # QUANTIZED_LINEAR_ARCHS). None leaves the process flags untouched.
    quantized_gemm_mode: Optional[str] = TXT2AUD_DEFAULTS["quantized_gemm_mode"]


class GenerationParams(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    steps: int = 20
    cfg_scale: float = 7.0
    sampler: str = "euler"
    schedule_type: str = "uniform"
    seed: int = -1
    ancestral_seed: int = -1  # Seed for stochastic samplers (Euler a, DPM2 a, etc.). -1 = use main seed
    width: int = 1024
    height: int = 1024
    model: str = ""
    loras: Optional[List[LoRAConfig]] = []
    controlnets: Optional[List[ControlNetConfig]] = []
    prompt_chunking_mode: str = "a1111"  # Options: a1111, sd_scripts, nobos
    max_prompt_chunks: int = 0  # 0 = unlimited, 1-4 = limit chunks
    developer_mode: bool = False  # Enable CFG metrics visualization
    # SDXL micro-conditioning override (inference): original_size for time_ids.
    original_size_w: int = 0
    original_size_h: int = 0
    original_size_scale: float = 1.0
    # Dynamic CFG scheduling
    cfg_schedule_type: str = "constant"  # constant, linear, quadratic, cosine, snr_based
    cfg_schedule_min: float = 1.0  # Minimum CFG at end of generation
    cfg_schedule_max: Optional[float] = None  # Maximum CFG at start (None = use cfg_scale)
    cfg_schedule_power: float = 2.0  # Power for quadratic schedule
    cfg_rescale_snr_alpha: float = 0.0  # SNR-based adaptive CFG (0.0 = disabled, 0.1-0.5 typical)
    # Dynamic thresholding
    dynamic_threshold_percentile: float = 0.0  # 0.0 = disabled, 99.5 = typical
    dynamic_threshold_mimic_scale: float = 7.0  # Clamp value for static threshold
    # NAG (Normalized Attention Guidance)
    nag_enable: bool = False  # Enable NAG
    nag_scale: float = 5.0  # NAG extrapolation scale (3-7 typical)
    nag_tau: float = 3.5  # NAG normalization threshold (2.5-3.5 typical)
    nag_alpha: float = 0.25  # NAG blending factor (0.25-0.5 typical)
    nag_sigma_end: float = 3.0  # Sigma threshold to disable NAG (0.0 = always enabled)
    nag_negative_prompt: Optional[str] = ""  # Separate negative prompt for NAG (empty = use main negative prompt)
    # Attention processor type
    attention_type: str = "normal"  # "normal", "sage", "flash"
    attention_impl: str = "conduit"  # "conduit" | "diffusers" (FLUX.2 inference kernel impl)
    # U-Net Quantization
    unet_quantization: Optional[str] = None  # None, "int8", "fp8", "int4", "nf4"
    # Text Encoder Quantization (Z-Image only)
    text_encoder_quantization: Optional[str] = None  # None, "fp8_e4m3fn", "fp8_e5m2", "uint8", "uint4"
    # Quantized-GEMM path for checkpoints that already carry weight-only
    # quantized Linear weights (QUANTIZED_LINEAR_ARCHS: anima / flux2 /
    # ideogram4 / krea2). null = leave the
    # process flags alone (env / POST /system/* value stands); "w8a8" = force
    # both W8A8 paths on; "dequant" = force both off. See api/quantized_gemm.py.
    quantized_gemm_mode: Optional[str] = GENERATION_DEFAULTS["quantized_gemm_mode"]
    # CPU text encoding: keep text encoder on CPU during prompt encode (saves VRAM, slower)
    cpu_text_encoding: bool = GENERATION_DEFAULTS["cpu_text_encoding"]
    # torch.compile optimization
    use_torch_compile: bool = False  # Enable torch.compile for U-Net (1.3-2x speedup)
    # Keep-models-hot (queue optimization; SD1.5/SDXL only in this phase)
    keep_models_hot: bool = GENERATION_DEFAULTS["keep_models_hot"]
    vae_tiling: bool = GENERATION_DEFAULTS["vae_tiling"]  # Tiled VAE decode for large images
    vae_tile_threshold: int = GENERATION_DEFAULTS["vae_tile_threshold"]  # px; 0=auto (per-VAE default)
    vae_tile_mode: str = GENERATION_DEFAULTS["vae_tile_mode"]  # "blend" | "context"
    vae_tile_global_norm: bool = GENERATION_DEFAULTS["vae_tile_global_norm"]  # two-pass whole-image GroupNorm stats
    color_flatten_strength: int = GENERATION_DEFAULTS["color_flatten_strength"]  # 0-100 chroma smoothing; 0=off
    flatten_in_loop: bool = GENERATION_DEFAULTS["flatten_in_loop"]  # in-loop hard-flatten of flat bg (SD1.5/SDXL)
    flatten_in_loop_last_steps: int = GENERATION_DEFAULTS["flatten_in_loop_last_steps"]  # inject on last N actual steps
    flatten_in_loop_min_region: float = GENERATION_DEFAULTS["flatten_in_loop_min_region"]  # flat-region area gate
    # Spectrum (Adaptive Spectral Feature Forecasting) acceleration
    spectrum_enable: bool = GENERATION_DEFAULTS["spectrum_enable"]
    fbcache_enable: bool = GENERATION_DEFAULTS["fbcache_enable"]
    fbcache_threshold: float = GENERATION_DEFAULTS["fbcache_threshold"]
    fbcache_warmup_steps: int = GENERATION_DEFAULTS["fbcache_warmup_steps"]
    fbcache_cache_branch: int = GENERATION_DEFAULTS["fbcache_cache_branch"]
    spectrum_w: float = GENERATION_DEFAULTS["spectrum_w"]
    spectrum_w_decay: float = GENERATION_DEFAULTS["spectrum_w_decay"]
    spectrum_delta_cap: float = GENERATION_DEFAULTS["spectrum_delta_cap"]
    spectrum_m: int = GENERATION_DEFAULTS["spectrum_m"]
    spectrum_lam: float = GENERATION_DEFAULTS["spectrum_lam"]
    spectrum_warmup_steps: int = GENERATION_DEFAULTS["spectrum_warmup_steps"]
    spectrum_window_size: int = GENERATION_DEFAULTS["spectrum_window_size"]
    spectrum_flex_window: float = GENERATION_DEFAULTS["spectrum_flex_window"]
    spectrum_tail: float = GENERATION_DEFAULTS["spectrum_tail"]
    spectrum_feature_mode: str = GENERATION_DEFAULTS["spectrum_feature_mode"]
    spectrum_cache_branch: int = GENERATION_DEFAULTS["spectrum_cache_branch"]
    spectrum_max_cache: int = GENERATION_DEFAULTS["spectrum_max_cache"]
    # TIPO (prompt upsampling)
    use_tipo: bool = False  # Enable TIPO prompt upsampling
    tipo_config: Optional[Dict] = None  # TIPO configuration (model, lengths, etc.)
    # Preview mode
    preview_predicted_x0: bool = False  # Show predicted x0 instead of current latent in preview

class Txt2ImgRequest(GenerationParams):
    pass

class Img2ImgRequest(GenerationParams):
    denoising_strength: float = 0.75

# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

@router.get("/health", tags=["system"])
async def health_check():
    """Liveness check for the backend API."""
    return {"status": "ok", "version": APP_VERSION}

# ---------------------------------------------------------------------------
# Schema endpoints — single source of truth for frontend DEFAULT_PARAMS
# ---------------------------------------------------------------------------

# Modality-matched replacement endpoints, keyed by the image route the caller
# actually hit. Suggesting /generate/txt2vid to a caller that just uploaded an
# input image is wrong twice over: it drops the image, and it is not the route
# that accepts one. `None` means the route has no same-shape counterpart for
# that modality, and the caller gets the nearest workable alternative instead.
_VIDEO_ROUTE_FOR_IMAGE_ROUTE = {
    "/generate/txt2img": "/generate/txt2vid",
    "/generate/img2img": "/generate/img2vid",
    # /generate/inpaint/video is the temporal counterpart: it regenerates a time
    # range of a clip and preserves the rest, as inpaint regenerates a region of
    # an image. It is implemented for MiniMax-H3 only, so LTX-2.3 overrides it.
    "/generate/inpaint": "/generate/inpaint/video",
    "/generate/outpaint": "/generate/outpaint/video",
}

# Per-architecture corrections to the map above: a route that exists but does
# not serve THIS architecture must not be suggested to it.
_VIDEO_ROUTE_OVERRIDES_BY_ARCH = {
    "ltx2": {"/generate/inpaint": None},
}

_AUDIO_ROUTE_FOR_IMAGE_ROUTE = {
    "/generate/txt2img": "/generate/txt2aud",
    "/generate/img2img": "/generate/aud2aud",
    # ACE-Step's inpainting equivalent is aud2aud with mode="repaint".
    "/generate/inpaint": "/generate/aud2aud",
    "/generate/outpaint": "/generate/outpaint/audio",
}


def _video_route_hint(endpoint: str, arch: Optional[str] = None) -> str:
    """Return the "use X instead" clause for an image route refused on video."""
    overrides = _VIDEO_ROUTE_OVERRIDES_BY_ARCH.get(arch or "", {})
    replacement = (overrides[endpoint] if endpoint in overrides
                   else _VIDEO_ROUTE_FOR_IMAGE_ROUTE.get(endpoint))
    if replacement:
        return f"use {replacement}"
    return "use /generate/img2vid to animate a still, or /generate/outpaint/video to extend a clip"


def _reject_if_video_model(endpoint: str = "/generate/txt2img"):
    """Reject an image-generation request when a video model is loaded.

    `endpoint` is the image route the caller hit; the refusal names the video
    route with the same request shape (img2img -> img2vid, outpaint ->
    outpaint/video) rather than always pointing at txt2vid.

    Raised before the executor so it surfaces as a 4xx ValidationError instead of
    being re-wrapped as a 500 GenerationError by the route's broad except.
    """
    if getattr(pipeline_manager, "is_ltx2_model", False):
        hint = _video_route_hint(endpoint, "ltx2")
        raise CustomValidationError(
            f"The loaded model is a video model (LTX-2.3); {hint}",
            detail=f"LTX-2.3 produces video, not still images, so {endpoint} cannot serve it. "
                   "Load an image model for txt2img/img2img/inpaint/outpaint.",
        )
    if getattr(pipeline_manager, "is_minimax_h3_model", False):
        hint = _video_route_hint(endpoint, "minimax_h3")
        raise CustomValidationError(
            f"The loaded model is a video model (MiniMax-H3); {hint}",
            detail=f"MiniMax-H3 produces video with a joint audio track, not still images, and its "
                   f"shortest decodable clip is 22 frames, so {endpoint} cannot serve it. "
                   "Load an image model for txt2img/img2img/inpaint/outpaint.",
        )


def _reject_if_video_model_on_audio_route(endpoint: str):
    """Reject an audio-only request when MiniMax-H3 is loaded, with ITS reason.

    MiniMax-H3 does generate audio -- but only jointly with video, in one packed
    sequence, and its audio is not independently addressable output. Without this
    the three ACE-Step routes would refuse it with "No ACE-Step model loaded",
    which is true and useless: it reads as "this model cannot make audio".
    """
    if getattr(pipeline_manager, "is_minimax_h3_model", False):
        raise CustomValidationError(
            "MiniMax-H3 generates audio only jointly with video",
            detail=f"The loaded model is MiniMax-H3. Its audio track is produced together with "
                   f"the video by /generate/txt2vid and is not separately addressable, so "
                   f"{endpoint} cannot serve it. Load an ACE-Step 1.5 model for standalone audio.",
        )


# NOTE: `_reject_if_video_arch_unwired` lived here. It refused a video endpoint
# that dispatched only to LTX-2.3 while a MiniMax-H3 model was loaded, with that
# reason rather than "no LTX-2.3 model loaded". Every video endpoint --
# txt2vid, img2vid and now outpaint/video -- dispatches to both architectures,
# so it had no callers left and is deleted rather than kept as a dead branch.


def _reject_if_audio_model(endpoint: str = "/generate/txt2img"):
    """Reject an image-generation request when an audio model (ACE-Step) is loaded.

    `endpoint` is the image route the caller hit; the refusal names the audio
    route with the same request shape (img2img -> aud2aud, outpaint ->
    outpaint/audio) rather than always pointing at txt2aud.

    Raised before the executor so it surfaces as a 4xx ValidationError instead of
    being re-wrapped as a 500 GenerationError by the route's broad except.
    """
    if getattr(pipeline_manager, "is_acestep_model", False):
        replacement = _AUDIO_ROUTE_FOR_IMAGE_ROUTE.get(endpoint, "/generate/txt2aud")
        raise CustomValidationError(
            f"The loaded model is an audio model (ACE-Step); use {replacement}",
            detail=f"ACE-Step produces audio, not still images, so {endpoint} cannot serve it. "
                   "Load an image model for txt2img/img2img/inpaint/outpaint.",
        )


@router.get("/schema/generation-defaults")
async def get_generation_defaults():
    """Return default parameter values for all generation modes.

    ``video_arch_overlays`` carries the per-architecture video overrides, so a
    client resolves a video default as `base | overlay[arch]` — exactly what the
    video routes do server-side for every field a request omits.
    ``outpaint_video_arch_overlays`` is the same thing for the keys that exist
    only on `/generate/outpaint/video` (chiefly `total_frames`, whose base value
    is on LTX-2.3's grid), applied on top of the video overlay, and
    ``inpaint_video_arch_overlays`` for `/generate/inpaint/video`'s own keys.
    """
    return {
        "video_arch_overlays": VIDEO_GEN_ARCH_OVERLAYS,
        "outpaint_video_arch_overlays": OUTPAINT_VIDEO_ARCH_OVERLAYS,
        "inpaint_video_arch_overlays": INPAINT_VIDEO_ARCH_OVERLAYS,
        "txt2img": TXT2IMG_DEFAULTS,
        "img2img": IMG2IMG_DEFAULTS,
        "inpaint":  INPAINT_DEFAULTS,
        "outpaint": OUTPAINT_DEFAULTS,
        "upscale": UPSCALE_DEFAULTS,
        "txt2vid": TXT2VID_DEFAULTS,
        "img2vid": IMG2VID_DEFAULTS,
        "ref2vid": REF2VID_DEFAULTS,
        "outpaint_vid": OUTPAINT_VIDEO_DEFAULTS,
        "inpaint_vid": INPAINT_VIDEO_DEFAULTS,
        "txt2aud": TXT2AUD_DEFAULTS,
        "aud2aud": AUD2AUD_DEFAULTS,
        "outpaint_aud": OUTPAINT_AUDIO_DEFAULTS,
    }

@router.get("/schema/prompt-assist-defaults")
async def get_prompt_assist_defaults():
    """Return MiniMax-H3 prompt-assistant defaults."""
    return PROMPT_ASSIST_DEFAULTS

@router.get("/schema/training-defaults")
async def get_training_defaults():
    """Return default parameter values for LoRA/Full-FT training."""
    return TRAINING_DEFAULTS

@router.get("/schema/tagger-training-defaults")
async def get_tagger_training_defaults():
    """Return default parameter values for tagger training."""
    return TAGGER_TRAINING_DEFAULTS

@router.get("/schema/vae-training-defaults")
async def get_vae_training_defaults():
    """Return default parameter values for VAE (decoder-only) training.

    Backs `training_method: "vae_decoder"` / `network.type: vae_decoder` runs.
    The first block of keys (batch_size … max_step_saves_to_keep) is written
    into the shared `process.train` / `process.save` YAML sections; everything
    from `vae_source` onwards is written into a dedicated `process.vae` section.
    """
    return VAE_TRAINING_DEFAULTS

@router.get("/schema/timestep-defaults-by-arch")
async def get_timestep_defaults_by_arch():
    """Per-architecture default timestep_sampling configs.

    The frontend applies the selected model's entry when the base model changes
    (user edits still win). Most architectures default to uniform; only MiniT2I
    differs (logit_normal mean=-0.8/std=0.8). "_default" is the global fallback.
    """
    return TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH

@router.get("/schema/bundle-vae-defaults-by-arch")
async def get_bundle_vae_defaults_by_arch():
    """Per-architecture default bundle_vae for full-parameter saves.

    sd15/sdxl/deus default True (comfy-layout checkpoints consumed by
    A1111/ComfyUI require the first_stage_model.* VAE section); other
    architectures default False. "_default" is the global fallback. The frontend
    applies the selected model's entry when the base model changes (user edits win).
    """
    return BUNDLE_VAE_DEFAULTS_BY_ARCH


@router.get("/schema/arch-capabilities")
async def get_arch_capabilities():
    """Per-architecture table of generation features that have NO effect.

    Mirrors `api/arch_capabilities.ARCH_UNSUPPORTED` verbatim: an architecture
    key maps feature name -> a one-line factual reason the feature is ignored
    there. The backend already uses this table to emit an `unsupported_param`
    warning after the fact; exposing it lets the generation panels hide a
    control on an architecture that would ignore it, driven by the same matrix
    instead of a second hardcoded arch list in the frontend.

    `feature_params` maps each feature to the request parameter keys that arm
    it, so a caller can go from a parameter name back to its feature.

    `supported_values` is the exemption list: values of an arming parameter that
    the architecture DOES honor even though the feature is listed unsupported
    (e.g. `unet_quantization="int8"` on Krea 2, which converts an unquantized
    transformer in place, while the FP8 values there remain checkpoint-driven).

    `runtime_int8_archs` is the architectures whose transformer the in-place
    weight-only INT8 converter is wired for, i.e. the ones that honor
    `unet_quantization="int8"`. It mirrors `RUNTIME_INT8_ARCHS` in
    `core/models/common/int8_runtime_quantize.py` (the module that implements
    the conversion), so a client offering the value reads it from here rather
    than keeping its own copy of the list. It cannot be inferred from
    `unsupported`/`supported_values`: an architecture that honors
    `unet_quantization` outright (anima) has no entry in either map.

    `quantized_linear_archs` is the architectures whose LOADERS swap in the
    quantized Linear classes (`Int8Linear` / `Fp8Linear`), i.e. the ones where
    `quantized_gemm_mode` selects anything. It mirrors `QUANTIZED_LINEAR_ARCHS`
    in the same module and is a superset of `runtime_int8_archs` in general (an
    architecture can READ a quantized checkpoint without owning an in-place
    converter). Served so a caller -- and this API's own documentation -- names
    the set from the backend tuple instead of keeping a hand-written copy that
    drifts.

    `training_unsupported` is a DIFFERENT axis: architecture -> training method
    -> why that method is refused there (not ignored -- the trainer raises), so
    the training UI filters its method dropdown from the same table.

    `video_constraints` is the per-video-arch `TemporalSpec` (valid clip
    lengths, production bounds, fixed fps, canvas envelope) that the video
    routes validate against, so a client can build a valid clip-length list
    instead of hardcoding one.
    """
    from api.arch_capabilities import (
        ARCH_SUPPORTED_VALUES, ARCH_UNSUPPORTED, FEATURE_PARAMS, FEATURE_LABELS,
        QUANTIZED_LINEAR_ARCHS, RUNTIME_INT8_ARCHS, TRAINING_UNSUPPORTED,
        video_constraints_payload,
    )
    return {
        "unsupported": ARCH_UNSUPPORTED,
        "supported_values": ARCH_SUPPORTED_VALUES,
        "feature_params": FEATURE_PARAMS,
        "feature_labels": FEATURE_LABELS,
        "training_unsupported": TRAINING_UNSUPPORTED,
        "video_constraints": video_constraints_payload(),
        "runtime_int8_archs": list(RUNTIME_INT8_ARCHS),
        "quantized_linear_archs": list(QUANTIZED_LINEAR_ARCHS),
    }


# ---------------------------------------------------------------------------
# GPU coordinator helpers (shared by all /generate/* endpoints)
# ---------------------------------------------------------------------------

# Conservative per-pixel peak VRAM table.  Used to estimate how much
# headroom we need before starting generation so the GPU coordinator
# can decide whether to offload running tagger training.  Values are
# intentionally on the high side — false positives cost ~5-10s of
# offload time; false negatives cost OOM.
_PEAK_VRAM_GB_BY_KIND = {
    "sd15":   4.0,
    "sdxl":  12.0,
    "zimage": 14.0,
    "flux":  18.0,
    "flux2": 24.0,
    "ideogram4": 26.0,  # two 9.3B fp8 transformers (cond + uncond) resident during denoise
    "minit2i": 8.0,    # small pixel-space DiT (B/L ~0.3-1.8GB) + FLAN-T5 staged
    "krea2": 26.0,     # ~12.9B bf16 MMDiT staged on GPU + Qwen3-VL TE + Qwen-Image VAE
    "ltx2": 40.0,      # ~19B bf16 video MM-DiT + Gemma-3 TE + LTX2 VAEs, cpu-offload staged
    "acestep": 8.0,    # 2B DiT + Oobleck VAE + Qwen3-Embedding-0.6B TE, sequential CPU/GPU staging
    "unknown": 14.0,   # safe default
}


async def _run_generation_in_executor(loop, executor, fn):
    """Run a generation's blocking work in ``executor``, carrying the request's
    context with it.

    ``loop.run_in_executor`` does NOT propagate contextvars into the worker
    thread (unlike ``asyncio.to_thread``), so without this the sampling thread
    would see no ``generation_status`` identity and every warning raised inside
    the denoise — ``attention_kernel_fallback`` above all — would be attributed
    by a global guess instead of by the request that caused it. A fresh
    ``copy_context()`` per call keeps concurrent generations independent (a
    single ``Context`` object cannot be entered twice).
    """
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(executor, lambda: ctx.run(fn))


async def _hash_saved_media(file_path: str) -> str:
    """``calculate_file_hash`` off the event loop, in the shared executor.

    The saved MASTER file this hashes can be a multi-gigabyte FFV1 mkv
    (``video_lossless=True``) or a long WAV/FLAC; reading and SHA-256'ing
    that synchronously inside an ``async def`` route stalls every other
    client's progress WebSocket for however long that read+digest takes.
    Every media-save route in this module calls THIS one wrapper (instead of
    each inlining its own ``run_in_executor`` dispatch) so there is one
    off-loop implementation, not eight in-loop ones.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, calculate_file_hash, file_path)


def _estimate_gen_peak_gb(width: int, height: int, batch_size: int, pipeline_kind: str) -> float:
    """Estimate peak VRAM for an incoming generation request."""
    base = _PEAK_VRAM_GB_BY_KIND.get(pipeline_kind, _PEAK_VRAM_GB_BY_KIND["unknown"])
    pixel_factor = max(1.0, (width * height) / (1024 * 1024))
    return base * pixel_factor * max(1, batch_size)


# Routes
@router.post("/generate/txt2img")
async def generate_txt2img(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(20),
    cfg_scale: float = Form(7.0),
    sampler: str = Form("euler"),
    schedule_type: str = Form("uniform"),
    seed: int = Form(-1),
    ancestral_seed: int = Form(-1),
    width: int = Form(1024),
    height: int = Form(1024),
    batch_size: int = Form(1),
    prompt_chunking_mode: str = Form("a1111"),
    max_prompt_chunks: int = Form(0),
    loras: str = Form("[]"),  # JSON string of LoRA configs
    controlnets: str = Form("[]"),  # JSON string of ControlNet configs
    controlnet_images: List[UploadFile] = File(default=[]),  # Direct ControlNet image upload
    developer_mode: bool = Form(False),
    cfg_schedule_type: str = Form("constant"),
    cfg_schedule_min: float = Form(1.0),
    cfg_schedule_max: Optional[float] = Form(None),
    cfg_schedule_power: float = Form(2.0),
    cfg_rescale_snr_alpha: float = Form(0.0),
    dynamic_threshold_percentile: float = Form(0.0),
    dynamic_threshold_mimic_scale: float = Form(7.0),
    nag_enable: bool = Form(False),
    nag_scale: float = Form(5.0),
    nag_tau: float = Form(3.5),
    nag_alpha: float = Form(0.25),
    nag_sigma_end: float = Form(3.0),
    nag_negative_prompt: str = Form(""),
    attention_type: str = Form(GENERATION_DEFAULTS["attention_type"]),
    attention_impl: str = Form("conduit"),
    unet_quantization: Optional[str] = Form(None),
    text_encoder_quantization: Optional[str] = Form(None),
    quantized_gemm_mode: Optional[str] = Form(GENERATION_DEFAULTS["quantized_gemm_mode"]),
    use_torch_compile: bool = Form(False),
    keep_models_hot: bool = Form(GENERATION_DEFAULTS["keep_models_hot"]),
    vae_tiling: bool = Form(GENERATION_DEFAULTS["vae_tiling"]),
    vae_tile_threshold: int = Form(GENERATION_DEFAULTS["vae_tile_threshold"]),
    vae_tile_mode: str = Form(GENERATION_DEFAULTS["vae_tile_mode"]),
    vae_tile_global_norm: bool = Form(GENERATION_DEFAULTS["vae_tile_global_norm"]),
    color_flatten_strength: int = Form(GENERATION_DEFAULTS["color_flatten_strength"]),
    flatten_in_loop: bool = Form(GENERATION_DEFAULTS["flatten_in_loop"]),
    flatten_in_loop_last_steps: int = Form(GENERATION_DEFAULTS["flatten_in_loop_last_steps"]),
    flatten_in_loop_min_region: float = Form(GENERATION_DEFAULTS["flatten_in_loop_min_region"]),
    spectrum_enable: bool = Form(GENERATION_DEFAULTS["spectrum_enable"]),
    fbcache_enable: bool = Form(GENERATION_DEFAULTS["fbcache_enable"]),
    fbcache_threshold: float = Form(GENERATION_DEFAULTS["fbcache_threshold"]),
    fbcache_warmup_steps: int = Form(GENERATION_DEFAULTS["fbcache_warmup_steps"]),
    fbcache_cache_branch: int = Form(GENERATION_DEFAULTS["fbcache_cache_branch"]),
    spectrum_w: float = Form(GENERATION_DEFAULTS["spectrum_w"]),
    spectrum_w_decay: float = Form(GENERATION_DEFAULTS["spectrum_w_decay"]),
    spectrum_delta_cap: float = Form(GENERATION_DEFAULTS["spectrum_delta_cap"]),
    spectrum_m: int = Form(GENERATION_DEFAULTS["spectrum_m"]),
    spectrum_lam: float = Form(GENERATION_DEFAULTS["spectrum_lam"]),
    spectrum_warmup_steps: int = Form(GENERATION_DEFAULTS["spectrum_warmup_steps"]),
    spectrum_window_size: int = Form(GENERATION_DEFAULTS["spectrum_window_size"]),
    spectrum_flex_window: float = Form(GENERATION_DEFAULTS["spectrum_flex_window"]),
    spectrum_tail: float = Form(GENERATION_DEFAULTS["spectrum_tail"]),
    spectrum_feature_mode: str = Form(GENERATION_DEFAULTS["spectrum_feature_mode"]),
    spectrum_cache_branch: int = Form(GENERATION_DEFAULTS["spectrum_cache_branch"]),
    spectrum_max_cache: int = Form(GENERATION_DEFAULTS["spectrum_max_cache"]),
    use_tipo: bool = Form(False),
    tipo_config: str = Form("{}"),  # JSON string of TIPO config
    preview_predicted_x0: bool = Form(False),  # Show predicted x0 in preview instead of current latent
    preview_decoder: str = Form("matrix"),  # Live-preview decoder for FLUX.2-VAE models: "matrix" | "taef2"
    enable_block_swap: bool = Form(False),
    blocks_to_swap: int = Form(20),
    use_pinned_memory: bool = Form(False),
    block_swap_h2d_only: bool = Form(GENERATION_DEFAULTS["block_swap_h2d_only"]),
    block_swap_ring_size: int = Form(GENERATION_DEFAULTS["block_swap_ring_size"]),
    ref_images: List[UploadFile] = File(default=[]),  # FLUX.2 Image Edit / Vision Encoder reference images
    vision_encoder_path: Optional[str] = Form(None),  # Path to SigLIP2 vision encoder safetensors
    vae_path: Optional[str] = Form(GENERATION_DEFAULTS["vae_path"]),  # Per-generation VAE override (dir or standalone VAE)
    text_encoder_path: Optional[str] = Form(GENERATION_DEFAULTS["text_encoder_path"]),  # Per-generation TE override (SD1.5/SDXL only)
    pid_sr_output: str = Form(GENERATION_DEFAULTS["pid_sr_output"]),  # PiD decoder only: "4x" | "original"
    pid_use_gemma: bool = Form(GENERATION_DEFAULTS["pid_use_gemma"]),  # PiD decoder only: opt-in runtime Gemma captioner
    pid_low_vram: bool = Form(GENERATION_DEFAULTS["pid_low_vram"]),  # PiD decoder only: opt-in row-chunked low-VRAM decode (default off, bit-identical when off)
    pid_tile_native: int = Form(GENERATION_DEFAULTS["pid_tile_native"]),  # PiD decoder only: per-tile native-res ceiling for the F9 tiled large-output decode
    pid_tile_overlap_ratio: float = Form(GENERATION_DEFAULTS["pid_tile_overlap_ratio"]),  # PiD decoder only: F9 tile feather overlap ratio
    pid_fast_large_decode: bool = Form(GENERATION_DEFAULTS["pid_fast_large_decode"]),  # PiD decoder only: opt out of F9 tiling back to the F7 whole-latent cap+bicubic path
    original_size_w: int = Form(0),  # SDXL micro-cond override: original width (0 = auto)
    original_size_h: int = Form(0),  # SDXL micro-cond override: original height (0 = auto)
    original_size_scale: float = Form(1.0),  # SDXL micro-cond: original_size = output * scale
    loop_decode: str = Form(GENERATION_DEFAULTS["loop_decode"]),  # Loop-generation decode mode: "full" | "cheap" | "none"
    skip_gallery: bool = Form(GENERATION_DEFAULTS["skip_gallery"]),  # Save to disk but skip the gallery DB record/thumbnail
    db: Session = Depends(get_gallery_db)
):
    """Generate image from text"""
    _reject_if_video_model("/generate/txt2img")
    _reject_if_audio_model("/generate/txt2img")
    lora_configs = []
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from api.arch_capabilities import check_arch_capabilities
    from api.generation_overrides import plan_overrides, apply_overrides
    from api.error_handlers import ValidationError
    # Quantized-GEMM path (quantized_gemm_mode): normalized BEFORE start_generation
    # so an invalid value is a 400 rather than a 500 from inside the run. None
    # (the default) leaves the process flags completely untouched.
    from api.quantized_gemm import normalize_mode as _normalize_qgm
    try:
        quantized_gemm_mode = _normalize_qgm(quantized_gemm_mode)
    except ValueError as exc:
        raise ValidationError("Invalid quantized_gemm_mode value", detail=str(exc))
    # Attention backend (attention_type): validated against the attention
    # registry's own vocabulary, also before start_generation. An unknown value
    # used to run native and be RECORDED as the unknown string.
    attention_type = _validated_attention_type(
        attention_type, GENERATION_DEFAULTS["attention_type"])
    if loop_decode not in ("full", "cheap", "none"):
        raise ValidationError(
            "Invalid loop_decode value",
            detail=f"loop_decode must be 'full', 'cheap', or 'none', got {loop_decode!r}",
        )
    # Compatibility gate for VAE/TE overrides runs BEFORE start_generation so a
    # HARD mismatch raises ValidationError (HTTP 400) without opening a run.
    _override_plan = plan_overrides(pipeline_manager, vae_path, text_encoder_path)
    _gen_id = start_generation("txt2img")
    try:
        # Reset cancellation flag before starting new generation
        pipeline_manager.reset_cancel_flag()

        # Parse LoRA configs
        import json
        lora_configs = json.loads(loras) if loras else []

        # Parse ControlNet configs
        controlnet_configs = json.loads(controlnets) if controlnets else []

        # Parse TIPO config
        tipo_config_dict = json.loads(tipo_config) if tipo_config else {}

        # TIPO prompt upsampling (if enabled)
        original_prompt = prompt
        if use_tipo:
            print(f"[TIPO] Upsampling prompt with TIPO...")
            try:
                # Load TIPO model if needed
                model_name = tipo_config_dict.get("model_name", "KBlueLeaf/TIPO-500M")
                if not tipo_manager.loaded or tipo_manager.model_name != model_name:
                    tipo_manager.load_model(model_name)

                # Generate upsampled prompt
                upsampled_prompt = tipo_manager.generate_prompt(
                    input_prompt=prompt,
                    tag_length=tipo_config_dict.get("tag_length", "long"),
                    nl_length=tipo_config_dict.get("nl_length", "long"),
                    temperature=tipo_config_dict.get("temperature", 1.0),
                    top_p=tipo_config_dict.get("top_p", 0.95),
                    top_k=tipo_config_dict.get("top_k", 50),
                    max_new_tokens=tipo_config_dict.get("max_new_tokens", 256),
                    category_order=tipo_config_dict.get("category_order", []),
                    enabled_categories=tipo_config_dict.get("enabled_categories", {}),
                    treat_as_nl=tipo_config_dict.get("treat_as_nl", False)
                )

                # If result is dict (tipo-kgen mode), format it to string
                if isinstance(upsampled_prompt, dict):
                    category_order = tipo_config_dict.get("category_order", [])
                    enabled_categories = tipo_config_dict.get("enabled_categories", {})

                    # If no category order specified, use default
                    if not category_order:
                        category_order = ["special", "quality", "rating", "artist", "copyright", "characters", "meta", "general"]

                    # If no enabled categories specified, enable all by default
                    if not enabled_categories:
                        enabled_categories = {cat: True for cat in category_order}
                        enabled_categories["meta"] = False  # Meta disabled by default

                    prompt = tipo_manager.format_kgen_result(
                        upsampled_prompt,
                        category_order,
                        enabled_categories
                    )
                else:
                    prompt = upsampled_prompt

                print(f"[TIPO] Original prompt: {original_prompt[:100]}...")
                print(f"[TIPO] Upsampled prompt: {prompt[:100]}...")

                # Unload TIPO model to free VRAM
                tipo_manager.unload_model()

            except Exception as e:
                print(f"[TIPO] Error during upsampling: {e}")
                print(f"[TIPO] Using original prompt")
                # Continue with original prompt on error

        # Process reference images (FLUX.2 Image Edit / Vision Encoder)
        # NOTE: Image/io are already imported at module scope (top of file) —
        # do NOT re-import them locally here. A local `from PIL import Image`/
        # `import io` anywhere in this function body makes Python treat those
        # names as function-local for the ENTIRE function, so when this branch
        # is skipped (no ref_images, the common case) any later unconditional
        # use of `Image`/`io` in generate_txt2img raises UnboundLocalError.
        ref_image_list = []
        if ref_images:
            for ref_img_file in ref_images:
                img_bytes = await ref_img_file.read()
                ref_image_list.append(Image.open(io.BytesIO(img_bytes)))
            print(f"[RefImages] Loaded {len(ref_image_list)} reference image(s)")

        # Load / reuse Vision Encoder if path provided (SD/SDXL only; FLUX.2 uses its own path)
        is_flux2 = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "flux2"
        if vision_encoder_path and not is_flux2:
            pipeline_manager.load_vision_encoder(vision_encoder_path)
        elif not vision_encoder_path and not is_flux2:
            # No VE path supplied — keep existing VE if already loaded (allows sticky sessions)
            pass

        # Apply (or restore) the planned VAE/TE overrides on the loaded model.
        _override_meta = apply_overrides(
            pipeline_manager, _override_plan,
            pid_sr_output=pid_sr_output, pid_use_gemma=pid_use_gemma, pid_low_vram=pid_low_vram,
            pid_tile_native=pid_tile_native, pid_tile_overlap_ratio=pid_tile_overlap_ratio,
            pid_fast_large_decode=pid_fast_large_decode, prompt=prompt,
        )

        # Generate image
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler": sampler,
            "schedule_type": schedule_type,
            "seed": seed,
            "ancestral_seed": ancestral_seed,
            "width": width,
            "height": height,
            "batch_size": batch_size,
            "loras": lora_configs,  # Add LoRA configs
            "developer_mode": developer_mode,
            "cfg_schedule_type": cfg_schedule_type,
            "cfg_schedule_min": cfg_schedule_min,
            "cfg_schedule_max": cfg_schedule_max,
            "cfg_schedule_power": cfg_schedule_power,
            "cfg_rescale_snr_alpha": cfg_rescale_snr_alpha,
            "dynamic_threshold_percentile": dynamic_threshold_percentile,
            "dynamic_threshold_mimic_scale": dynamic_threshold_mimic_scale,
            "nag_enable": nag_enable,
            "nag_scale": nag_scale,
            "nag_tau": nag_tau,
            "nag_alpha": nag_alpha,
            "nag_sigma_end": nag_sigma_end,
            "nag_negative_prompt": nag_negative_prompt,
            "attention_type": attention_type,
            "attention_impl": attention_impl,
            "unet_quantization": unet_quantization,
            "quantized_gemm_mode": quantized_gemm_mode,
            "original_size_w": original_size_w,
            "original_size_h": original_size_h,
            "original_size_scale": original_size_scale,
            "text_encoder_quantization": text_encoder_quantization,
            "use_torch_compile": use_torch_compile,
            "keep_models_hot": keep_models_hot,
            "vae_tiling": vae_tiling,
            "vae_tile_threshold": vae_tile_threshold,
            "vae_tile_mode": vae_tile_mode,
            "vae_tile_global_norm": vae_tile_global_norm,
            "color_flatten_strength": color_flatten_strength,
            "flatten_in_loop": flatten_in_loop,
            "flatten_in_loop_last_steps": flatten_in_loop_last_steps,
            "flatten_in_loop_min_region": flatten_in_loop_min_region,
            "spectrum_enable": spectrum_enable,
            "fbcache_enable": fbcache_enable,
            "fbcache_threshold": fbcache_threshold,
            "fbcache_warmup_steps": fbcache_warmup_steps,
            "fbcache_cache_branch": fbcache_cache_branch,
            "spectrum_w": spectrum_w,
            "spectrum_w_decay": spectrum_w_decay,
            "spectrum_delta_cap": spectrum_delta_cap,
            "spectrum_m": spectrum_m,
            "spectrum_lam": spectrum_lam,
            "spectrum_warmup_steps": spectrum_warmup_steps,
            "spectrum_window_size": spectrum_window_size,
            "spectrum_flex_window": spectrum_flex_window,
            "spectrum_tail": spectrum_tail,
            "spectrum_feature_mode": spectrum_feature_mode,
            "spectrum_cache_branch": spectrum_cache_branch,
            "spectrum_max_cache": spectrum_max_cache,
            "enable_block_swap": enable_block_swap,
            "blocks_to_swap": blocks_to_swap,
            "use_pinned_memory": use_pinned_memory,
            "block_swap_h2d_only": block_swap_h2d_only,
            "block_swap_ring_size": block_swap_ring_size,
            "preview_decoder": preview_decoder,
            "ref_images": ref_image_list,  # FLUX.2 Image Edit reference images
            "vae_path": vae_path,
            "text_encoder_path": text_encoder_path,
            "pid_sr_output": pid_sr_output,
            "pid_use_gemma": pid_use_gemma,
            "pid_low_vram": pid_low_vram,
            "pid_tile_native": pid_tile_native,
            "pid_tile_overlap_ratio": pid_tile_overlap_ratio,
            "pid_fast_large_decode": pid_fast_large_decode,
            "loop_decode": loop_decode,
            "skip_gallery": skip_gallery,
        }
        params.update(_override_meta)

        # Log params without large base64 data
        print(f"txt2img generation params: {sanitize_params_for_logging(params)}")

        # Set prompt chunking settings
        set_prompt_chunking_settings(
            pipeline_manager,
            prompt_chunking_mode,
            max_prompt_chunks
        )

        # Load LoRAs if specified
        pipeline_manager.txt2img_pipeline, has_step_range_loras = load_loras_for_generation(
            lora_manager,
            pipeline_manager.txt2img_pipeline,
            lora_configs,
            "txt2img"
        )

        # Process ControlNet images
        # Handle direct image uploads (multipart) or base64 (JSON)
        processed_controlnet_images = []
        if controlnet_images and len(controlnet_images) > 0:
            # Direct image upload via multipart (io is already imported at module scope)
            for uploaded_file in controlnet_images:
                image_data = await uploaded_file.read()
                cn_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                processed_controlnet_images.append(cn_image)

        # Also process base64 images from controlnets JSON
        style_transfer = None
        style_transfers: list = []
        style_combine_mode = "stack"
        if controlnet_configs:
            base64_images, style_transfer, style_transfers, style_combine_mode = process_controlnet_configs(
                controlnet_configs,
                generation_type="txt2img"
            )
            processed_controlnet_images.extend(base64_images)

        params["controlnet_images"] = processed_controlnet_images
        params["controlnets"] = controlnet_configs
        params["style_transfer"] = style_transfer
        params["style_transfers"] = style_transfers
        params["style_combine_mode"] = style_combine_mode

        # Detect model type
        is_sdxl = pipeline_manager.txt2img_pipeline is not None and \
                  "XL" in pipeline_manager.txt2img_pipeline.__class__.__name__
        is_zimage = pipeline_manager.current_model_info and \
                    pipeline_manager.current_model_info.get("type") == "zimage"
        is_deus = pipeline_manager.current_model_info and \
                  pipeline_manager.current_model_info.get("type") == "deus"
        is_flux2 = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "flux2"
        # Z-Image with SDXL VAE (4ch) needs TAESD-XL instead of TAEF1
        is_zimage_sdxl_vae = is_zimage and \
                             pipeline_manager.current_model_info.get("vae_type") == "sdxl"
        is_anima = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "anima"
        # Lens shares the same AutoencoderKLFlux2 / 32ch latent format as FLUX.2
        is_lens = pipeline_manager.current_model_info and \
                  pipeline_manager.current_model_info.get("type") == "lens"
        # Ideogram 4 shares AutoencoderKLFlux2's 128-ch packed latent with Lens.
        is_ideogram4 = pipeline_manager.current_model_info and \
                       pipeline_manager.current_model_info.get("type") == "ideogram4"
        is_minit2i = pipeline_manager.current_model_info and \
                     pipeline_manager.current_model_info.get("type") == "minit2i"
        minit2i_vae_type = (pipeline_manager.minit2i_components or {}).get("vae_type", "none") if is_minit2i else "none"
        is_krea2 = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "krea2"

        # Warn about parameters the loaded architecture silently ignores
        _current_arch = pipeline_manager.current_model_info.get("type") if pipeline_manager.current_model_info else None
        check_arch_capabilities(params, _current_arch)

        # Progress callback to send updates via WebSocket
        progress_callback = create_progress_callback_factory(
            taesd_manager,
            manager,
            is_sdxl,
            is_zimage,
            is_deus,
            is_zimage_sdxl_vae,
            is_flux2,
            is_anima,
            is_lens=is_lens,
            is_ideogram4=is_ideogram4,
            is_minit2i=is_minit2i,
            minit2i_vae_type=minit2i_vae_type,
            is_krea2=is_krea2,
            image_width=params.get("width"),
            image_height=params.get("height"),
            # For flow-matching DiTs (Anima / Z-Image / FLUX.2 / Lens), default to
            # the pred_x0 preview: x_t is mostly noise mid-denoising, while
            # pred_x0 = x_t - σ·v shows the model's current clean-image
            # estimate from the very first steps. Any explicit user override
            # via the API still wins.
            preview_predicted_x0=(preview_predicted_x0 or is_anima or is_zimage or is_flux2 or is_lens or is_ideogram4 or is_minit2i or is_krea2),
            preview_enabled=params.get("preview_enabled", True),
            preview_interval=params.get("preview_interval", 4),
            preview_decoder=params.get("preview_decoder", "matrix")
        )

        # Create step callback for LoRA step range if needed
        step_callback = None
        if has_step_range_loras:
            step_callback = create_lora_step_callback(
                lora_manager,
                pipeline_manager.txt2img_pipeline,
                params.get("steps", 20)
            )

        # Run generation in thread pool to avoid blocking event loop.
        # Wrap in gpu_coordinator slot so any active tagger training is
        # paused (and optionally offloaded) at the next batch boundary
        # before we start pushing UNet onto the GPU.
        from core.gpu_coordinator import gpu_coordinator
        from core.inference.generation_timing import generation_timer
        loop = asyncio.get_event_loop()
        _peak_gb = _estimate_gen_peak_gb(width, height, batch_size,
                                         pipeline_manager.current_pipeline_kind)
        generation_timer.reset()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=_peak_gb, timeout=60.0):
            # GEMM flags are process-wide; keep selection and probing in this slot.
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(quantized_gemm_mode)
            image, actual_seed, actual_ancestral_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: pipeline_manager.generate_txt2img(params, progress_callback=progress_callback, step_callback=step_callback)
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        # Record total wall time + any phase breakdown the pipeline populated.
        apply_generation_timings(params, time.perf_counter() - _gen_start)

        # Update params with actual seeds
        params["seed"] = actual_seed
        params["ancestral_seed"] = actual_ancestral_seed

        # loop_decode="none": no VAE/PiD decode happened -- `image` is the raw
        # pre-unscale final latent tensor instead of a PIL.Image. Cache it and
        # return a latent_id for the next loop step's input_latent_id, skipping
        # save/thumbnail/gallery entirely (there is no image to save).
        if not isinstance(image, Image.Image):
            from core.inference.latent_cache import store_latent
            latent_id = store_latent(image, meta={
                "width": params.get("width"),
                "height": params.get("height"),
                "seed": actual_seed,
            })
            # The latent-only path returns before the fp8_gemm block below, so
            # report the resolved quantized-GEMM path here too — a loop step
            # that chains latents must still surface a degraded "w8a8" request.
            from api.quantized_gemm import report_quantized_gemm_outcome
            report_quantized_gemm_outcome(
                quantized_gemm_mode, fp8_gemm, _current_arch
            )
            complete_generation({"latent_id": latent_id, "seed": actual_seed}, generation_id=_gen_id)
            return {"success": True, "latent_id": latent_id, "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

        # Add Vision Encoder info to params for PNG metadata and DB storage.
        # Only record VE info when THIS generation actually used reference images.
        # The VE stays loaded ("sticky") across generations, so extract_vision_encoder_info
        # returns non-empty even for generations that used no reference image.
        if ref_image_list:
            ve_name, ve_hash = extract_vision_encoder_info(pipeline_manager)
            if ve_name:
                params["vision_encoder_name"] = ve_name
            if ve_hash:
                params["vision_encoder_hash"] = ve_hash

        # Add VAE identity to params. The VAE always participates in decode, so this
        # is recorded for every generation where it can be determined.
        vae_name, vae_hash = extract_vae_info(pipeline_manager)
        if vae_name:
            params["vae_name"] = vae_name
        if vae_hash:
            params["vae_hash"] = vae_hash

        # FP8 GEMM path. Recorded only for checkpoints that carry weight-only FP8
        # Linear weights, and it records the RESOLVED path rather than the flag —
        # the W8A8 path can be enabled while the device probe rejects every
        # scaling mode, and the two paths are numerically different.
        if fp8_gemm:
            params["fp8_gemm"] = fp8_gemm
        # Report an explicit "w8a8" request that resolved to the dequant path
        # (probe rejected it, or the checkpoint has no quantized layers).
        from api.quantized_gemm import report_quantized_gemm_outcome
        report_quantized_gemm_outcome(quantized_gemm_mode, fp8_gemm, _current_arch)
        # Attention backend that ACTUALLY ran (observed by the conduit), so the
        # PNG metadata and the gallery row cannot name a backend that did not.
        record_attention_backend(params, _gen_id)

        # Save image with metadata (include model info)
        filename = save_image_with_metadata(
            image,
            params,
            "txt2img",
            model_info=pipeline_manager.current_model_info,
            generation_id=_gen_id
        )

        # skip_gallery: the file is saved (so a generation loop can still chain
        # to the next step via its path) but no thumbnail/DB record is created.
        if skip_gallery:
            complete_generation({"filename": filename, "seed": actual_seed}, generation_id=_gen_id)
            return {
                "success": True,
                "filename": filename,
                "image_path": f"/outputs/{filename}",
                "actual_seed": actual_seed,
                "warnings": get_warnings(_gen_id),
            }

        # Create thumbnail
        image_path = os.path.join(settings.outputs_dir, filename)
        create_thumbnail(image_path)

        # Calculate metadata
        metadata = calculate_generation_metadata(
            image,
            lora_configs,
            extract_lora_names,
            calculate_image_hash
        )

        # Remove image objects from params before saving to DB and calculate ControlNet hashes
        params_for_db = prepare_params_for_db(params, calculate_image_hash)
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        # Extract model name and hash from current_model_info
        model_name, model_hash = extract_model_info(pipeline_manager)

        # Save to database
        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="txt2img",
            image_hash=metadata["image_hash"],
            lora_names=metadata["lora_names"],
            model_name=model_name,
            model_hash=model_hash
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except GenerationError as e:
        # Re-raise custom errors as-is
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except Exception as e:
        # Wrap unexpected errors in GenerationError
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Text-to-image generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )
    finally:
        # Unload LoRAs after generation
        if lora_configs and pipeline_manager.txt2img_pipeline:
            pipeline_manager.txt2img_pipeline = lora_manager.unload_loras(pipeline_manager.txt2img_pipeline)


@router.get("/generation/status")
async def get_generation_status():
    """Poll the current image-generation status.

    This is a polling-friendly complement to the WebSocket
    ``/api/v1/ws/progress`` channel (see ``backend/api/WS_PROTOCOL.md``).
    The WS channel has no dedicated ``complete``/``error`` message type, so a
    client that does not want to hold a WebSocket connection open can instead
    poll this endpoint to observe step progress and the final
    success/failure outcome of the most recent txt2img/img2img/inpaint
    request.
    """
    from api.generation_status import get_snapshot
    return get_snapshot()


# ---------------------------------------------------------------------------
# Training-preview generation (LoRA / Full-FT subprocess)
# ---------------------------------------------------------------------------
#
# When a LoRA / Full-FT training is active, the user can request a preview
# rendered with the in-training model.  The training process runs in a
# subprocess so we can't reach its UNet from here directly — instead we
# use file-based RPC (``core/training/training_preview_rpc``): write a
# request JSON in the run's ``output_dir``; the trainer picks it up at
# the next batch boundary, generates, and writes back PNG + meta JSON.

class TrainingPreviewRequest(GenerationParams):
    """Same as GenerationParams plus an optional run_id to target a
    specific training run when multiple are active, and an opt-in
    flag to persist the result to the regular gallery."""
    run_id: Optional[int] = None
    # When True, save the preview PNG into outputs/ + GeneratedImage row
    # so it shows up in the gallery like a normal generation.  The DB
    # row is tagged with ``model_name = "training-preview:<run>@step<N>"``
    # so it's distinguishable from real-model generations.  Default OFF
    # — the preview blob is transient unless the user explicitly asks
    # for it to be persisted.
    save_to_gallery: bool = False


def _get_active_training_for_preview(run_id_hint: Optional[int]) -> tuple[int, str]:
    """Find which training run to send a preview request to.

    If ``run_id_hint`` is given, use it.  Otherwise pick the single
    active running process; raise 409 if zero or multiple match.
    """
    from core.training.training_process import training_process_manager
    procs = training_process_manager.processes
    if run_id_hint is not None:
        proc = procs.get(int(run_id_hint))
        if proc is None or not proc.is_running:
            raise HTTPException(status_code=404,
                                detail=f"Training run {run_id_hint} is not active")
        return int(run_id_hint), proc.output_dir

    active = [(rid, p) for rid, p in procs.items() if p.is_running]
    if not active:
        raise HTTPException(status_code=409,
                            detail="No active training run to preview against")
    if len(active) > 1:
        ids = ", ".join(str(rid) for rid, _ in active)
        raise HTTPException(status_code=409,
                            detail=f"Multiple active runs ({ids}); pass run_id to disambiguate")
    rid, proc = active[0]
    return int(rid), proc.output_dir


def _broadcast_training_preview_frame(frame: dict) -> None:
    """Decode a training-preview latent frame (written by the trainer subprocess) with the
    API's TAESD model and broadcast it to the WebSocket in the same message shape as normal
    generation previews, so the frontend's existing preview handler shows it live."""
    import io as _io, base64 as _b64
    lat = frame.get("latents")
    if lat is None:
        return
    try:
        preview_pil = taesd_manager.decode_latent(
            lat,
            is_sdxl=bool(frame.get("is_sdxl", False)),
            is_zimage=bool(frame.get("is_zimage", False)),
            is_deus=False,
            is_zimage_sdxl_vae=False,
            is_flux2=False,
            is_anima=bool(frame.get("is_anima", False)),
            is_lens=bool(frame.get("is_lens", False)),
            is_ideogram4=bool(frame.get("is_ideogram4", False)),
            is_minit2i=bool(frame.get("is_minit2i", False)),
            minit2i_vae_type=frame.get("minit2i_vae_type"),
            is_krea2=bool(frame.get("is_krea2", False)),
            image_width=int(frame.get("image_width", 1024)),
            image_height=int(frame.get("image_height", 1024)),
            preview_decoder=str(frame.get("preview_decoder", "matrix")),
        )
        if not preview_pil:
            return
        buf = _io.BytesIO()
        preview_pil.save(buf, format="JPEG", quality=75)
        b64 = _b64.b64encode(buf.getvalue()).decode()
        step = int(frame.get("step", 0))
        total = int(frame.get("total", 1)) or 1
        display_step = 0 if step == -1 else step + 1
        manager.send_progress_sync(
            display_step, total, f"Preview {display_step}/{total}", preview_image=b64,
        )
    except Exception as e:   # noqa: BLE001
        print(f"[Preview] frame decode/broadcast failed: {e}")


async def _await_preview_result(
    output_dir: str, request_id: str, timeout: float = 180.0,
) -> tuple[Optional[bytes], dict]:
    """Poll for a preview result.  Returns (png_bytes, meta).

    The trainer writes the meta file LAST and atomically, so detecting
    the meta file is sufficient to know the PNG is fully written.
    """
    from core.training.training_preview_rpc import (
        result_image_path, result_meta_path, read_preview_frame,
    )
    import json as _json
    deadline = asyncio.get_event_loop().time() + timeout
    meta_path = result_meta_path(output_dir, request_id)
    img_path  = result_image_path(output_dir, request_id)
    last_seq = 0
    while asyncio.get_event_loop().time() < deadline:
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = _json.load(f)
            except Exception:
                meta = {"ok": False, "error": "could not parse meta file"}
            image_bytes: Optional[bytes] = None
            if img_path.exists():
                try:
                    image_bytes = img_path.read_bytes()
                except OSError:
                    image_bytes = None
            for p in (meta_path, img_path):
                try: p.unlink()
                except OSError: pass
            return image_bytes, meta
        # Live preview: the trainer writes a latent frame per preview interval; decode it
        # here (the API has the TAESD model) and broadcast to the WebSocket so the user sees
        # progress during the in-training generation.
        try:
            frame = read_preview_frame(output_dir, request_id)
            if frame and int(frame.get("seq", 0)) > last_seq:
                last_seq = int(frame["seq"])
                _broadcast_training_preview_frame(frame)
        except Exception:
            pass
        await asyncio.sleep(0.25)
    raise HTTPException(
        status_code=504,
        detail=f"Preview request timed out after {timeout:.0f}s "
               f"(training busy or stopped)",
    )


@router.get("/training/active")
async def get_active_training(training_db: Session = Depends(get_training_db)):
    """Return summary info on the currently-active LoRA / Full-FT training,
    or an ``is_running: false`` body when none.  Used by the generate panel to
    enable / disable the "Use training model" toggle and to display the target
    run.

    IDLE IS A 200, NOT A 404.  This endpoint is polled every 10 s for as long as
    a generation panel is mounted, and "nothing is training" is the ordinary
    case, not an error.  Signalling it with a 404 made the browser log a failed
    request on every tick -- the frontend's ``try/catch`` cannot suppress that,
    because the console line is emitted by the browser for any non-2xx XHR
    regardless of what JavaScript does with it.  ``openapi.yaml`` documented the
    200-with-null-fields contract from the start; this is the code catching up
    to it."""
    from core.training.training_process import training_process_manager
    active = [(rid, p) for rid, p in training_process_manager.processes.items()
              if p.is_running]
    if not active:
        return {
            "run_id": None,
            "run_name": None,
            "training_method": None,
            "current_step": None,
            "is_running": False,
        }
    # If multiple, return the one with the highest run_id (typically the
    # most recently started).  Frontend can disambiguate by passing run_id
    # explicitly to the preview endpoint.
    rid, proc = max(active, key=lambda x: x[0])
    run_name: Optional[str] = None
    training_method: Optional[str] = None
    try:
        _row = training_db.query(TrainingRun).filter(TrainingRun.id == int(rid)).first()
        if _row is not None:
            run_name = getattr(_row, "name", None) or getattr(_row, "run_name", None)
            training_method = getattr(_row, "training_method", None)
    except Exception:
        pass
    return {
        "run_id": int(rid),
        "run_name": run_name,
        "training_method": training_method,
        "current_step": getattr(proc, "current_step", 0),
        "is_running": True,
    }


class Img2ImgTrainingPreviewRequest(TrainingPreviewRequest):
    """img2img preview adds an init image (base64-encoded PNG/JPEG)
    and a denoising strength.  Init image is JSON-embedded to keep the
    endpoint shape consistent with the txt2img preview."""
    init_image_base64: str
    denoising_strength: float = 0.75
    inpaint_fill_mode: str = "original"        # forwarded for inpaint variants
    inpaint_fill_strength: float = 1.0
    inpaint_blur_strength: float = 1.0


class InpaintTrainingPreviewRequest(Img2ImgTrainingPreviewRequest):
    """inpaint preview adds a mask image (base64-encoded; alpha or
    grayscale OK — the trainer normalises it)."""
    mask_image_base64: str


async def _run_training_preview(
    request: TrainingPreviewRequest, mode: str, extra_params: Dict[str, Any],
) -> Response:
    """Shared core for the 3 preview endpoints.  Resolves the active
    training, queues the request file, awaits the result, returns a
    PNG response with metadata headers.

    When ``request.save_to_gallery`` is True, the PNG is additionally
    persisted under ``outputs/`` and inserted into the ``GeneratedImage``
    table so it appears in the gallery alongside normal generations.
    The DB row is tagged with ``model_name = "training-preview:<run>@step<N>"``
    so callers can filter previews out if desired.
    """
    from core.training.training_preview_rpc import (
        make_request_id, write_request,
    )
    run_id, output_dir = _get_active_training_for_preview(request.run_id)
    params: Dict[str, Any] = {**request.dict(exclude={"run_id"}), **extra_params}
    params["mode"] = mode
    # Same closed vocabulary as the generation routes: an unknown backend is a
    # 400 here too, rather than a string the trainer would absorb into native.
    params["attention_type"] = _validated_attention_type(
        params.get("attention_type"), GENERATION_DEFAULTS["attention_type"])

    request_id = make_request_id()
    try:
        write_request(output_dir, request_id, params)
    except OSError as e:
        raise HTTPException(status_code=500,
                            detail=f"Could not queue preview request: {e}")

    image_bytes, meta = await _await_preview_result(
        output_dir, request_id, timeout=180.0,
    )
    if not meta.get("ok"):
        raise HTTPException(
            status_code=500,
            detail=f"Preview generation failed: {meta.get('error', 'unknown error')}",
        )
    if image_bytes is None:
        raise HTTPException(status_code=500, detail="Preview returned no image")

    headers: Dict[str, str] = {
        "X-Preview-Run-Id":   str(run_id),
        "X-Preview-Seed":     str(meta.get("seed", "")),
        "X-Preview-Request":  request_id,
        "X-Preview-Mode":     mode,
    }

    # Optional: persist to the regular gallery (outputs/ + GeneratedImage row)
    if request.save_to_gallery:
        try:
            saved_filename = _save_preview_to_gallery(
                image_bytes=image_bytes,
                params=params,
                mode=mode,
                run_id=run_id,
                meta=meta,
            )
            headers["X-Preview-Filename"] = saved_filename
        except Exception as e:   # noqa: BLE001
            # Don't fail the whole request just because gallery save
            # failed; the user still gets the preview blob.
            import traceback
            print(f"[Preview] gallery save failed: {e}\n{traceback.format_exc()}")
            headers["X-Preview-Save-Error"] = str(e)[:200]

    return Response(content=image_bytes, media_type="image/png", headers=headers)


def _save_preview_to_gallery(
    *,
    image_bytes: bytes,
    params: Dict[str, Any],
    mode: str,
    run_id: int,
    meta: Dict[str, Any],
) -> str:
    """Persist a training-preview image to the regular gallery.

    Wraps the existing ``save_image_with_metadata`` + ``create_thumbnail``
    + ``create_db_image_record`` flow used by /generate/txt2img.  The
    model identity is faked as ``training-preview:<run_name|run>@step<N>``
    so it's obviously distinguishable from real-model generations.
    Reproducibility is not preserved — the in-training weights at step N
    are not recoverable from this metadata.  The PNG / DB row is a
    record of "what was used", not "how to reproduce".
    """
    import io as _io
    from PIL import Image as _Image
    from utils import save_image_with_metadata, create_thumbnail, calculate_image_hash
    from api.generation_utils import (
        prepare_params_for_db, create_db_image_record,
    )
    from database import GallerySessionLocal
    from database.models import GeneratedImage

    # Decode PNG bytes back to PIL Image (for metadata embedding +
    # thumbnail generation).
    image = _Image.open(_io.BytesIO(image_bytes)).convert("RGB")

    # Resolve the training context for honest model_name / hash:
    #  - run_name from DB (if available)
    #  - current_step from the process manager
    from core.training.training_process import training_process_manager
    proc = training_process_manager.processes.get(int(run_id))
    current_step = getattr(proc, "current_step", 0) if proc else 0
    run_name: Optional[str] = None
    try:
        with GallerySessionLocal() as _g:  # type: ignore[attr-defined]
            pass
    except Exception:
        pass
    try:
        from database import TrainingSessionLocal
        from database.models import TrainingRun
        with TrainingSessionLocal() as _db:
            _row = _db.query(TrainingRun).filter(TrainingRun.id == int(run_id)).first()
            if _row is not None:
                run_name = getattr(_row, "name", None) or getattr(_row, "run_name", None)
    except Exception:
        pass

    label = run_name or f"run{run_id}"
    fake_model_name = f"training-preview:{label}@step{current_step}"
    fake_model_hash = f"training-preview-step{current_step}"

    # Apply honest seed back into params so PNG / DB carry the real value
    actual_seed = int(meta.get("seed") or params.get("seed") or -1)
    params_for_save: Dict[str, Any] = {**params, "seed": actual_seed}
    # Stash training context inside parameters JSON for future filtering
    params_for_save["training_preview"] = {
        "run_id": int(run_id),
        "run_name": run_name,
        "current_step": current_step,
        "mode": mode,
    }

    model_info = {"source": fake_model_name, "model_hash": fake_model_hash}

    # 1) File save + PNG metadata
    filename = save_image_with_metadata(
        image, params_for_save, generation_type=mode, model_info=model_info,
    )
    image_path = os.path.join(settings.outputs_dir, filename)
    # 2) Thumbnail
    try:
        create_thumbnail(image_path)
    except Exception as _e:
        print(f"[Preview] thumbnail generation failed: {_e}")
    # 3) DB row
    try:
        image_hash = calculate_image_hash(image)
        params_for_db = prepare_params_for_db(params_for_save, calculate_image_hash)
        with GallerySessionLocal() as gdb:  # type: ignore[attr-defined]
            db_image = create_db_image_record(
                db_image_class=GeneratedImage,
                filename=filename,
                params=params_for_db,
                actual_seed=actual_seed,
                generation_type=mode,
                image_hash=image_hash,
                lora_names=None,
                model_name=fake_model_name,
                model_hash=fake_model_hash,
                result_image=image,
            )
            gdb.add(db_image)
            gdb.commit()
    except Exception as _e:
        print(f"[Preview] gallery DB insert failed: {_e}")
        # File is on disk regardless; the gallery will still pick it up
        # on next manual rescan even if this insert failed.
    return filename


@router.post("/generate/img2img/training-preview")
async def generate_img2img_training_preview(request: Img2ImgTrainingPreviewRequest):
    """img2img preview using the in-training model.  Body is JSON with
    ``init_image_base64`` (raw or data-URL) and the usual generation
    params (``denoising_strength``, prompt, steps, cfg, seed, w/h,
    sampler, schedule_type, optional ``loras``/``controlnets``)."""
    return await _run_training_preview(request, mode="img2img", extra_params={})


@router.post("/generate/inpaint/training-preview")
async def generate_inpaint_training_preview(request: InpaintTrainingPreviewRequest):
    """inpaint preview using the in-training model.  Body adds
    ``mask_image_base64`` plus the img2img fields."""
    return await _run_training_preview(request, mode="inpaint", extra_params={})


@router.post("/generate/txt2img/training-preview")
async def generate_txt2img_training_preview(request: TrainingPreviewRequest):
    """Generate an image using the CURRENT state of an active LoRA / Full-FT
    training (Base + LoRA-in-progress).

    The trainer supports stacking additional LoRAs and ControlNets on
    top of the in-training adapter via the request params (peft-based,
    best-effort).  See ``TrainingPreviewGenerator`` for details.
    """
    return await _run_training_preview(request, mode="txt2img", extra_params={})


@router.post("/generate/img2img")
async def generate_img2img(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(20),
    cfg_scale: float = Form(7.0),
    denoising_strength: float = Form(0.75),
    img2img_fix_steps: bool = Form(True),
    sampler: str = Form("euler"),
    schedule_type: str = Form("uniform"),
    seed: int = Form(-1),
    ancestral_seed: int = Form(-1),
    width: int = Form(1024),
    height: int = Form(1024),
    resize_mode: str = Form("image"),
    resampling_method: str = Form("lanczos"),
    prompt_chunking_mode: str = Form("a1111"),
    max_prompt_chunks: int = Form(0),
    loras: str = Form("[]"),  # JSON string of LoRA configs
    controlnets: str = Form("[]"),  # JSON string of ControlNet configs
    developer_mode: bool = Form(False),
    cfg_schedule_type: str = Form("constant"),
    cfg_schedule_min: float = Form(1.0),
    cfg_schedule_max: Optional[float] = Form(None),
    cfg_schedule_power: float = Form(2.0),
    cfg_rescale_snr_alpha: float = Form(0.0),
    dynamic_threshold_percentile: float = Form(0.0),
    dynamic_threshold_mimic_scale: float = Form(7.0),
    nag_enable: bool = Form(False),
    nag_scale: float = Form(5.0),
    nag_tau: float = Form(3.5),
    nag_alpha: float = Form(0.25),
    nag_sigma_end: float = Form(3.0),
    nag_negative_prompt: str = Form(""),
    attention_type: str = Form(GENERATION_DEFAULTS["attention_type"]),
    attention_impl: str = Form("conduit"),
    unet_quantization: Optional[str] = Form(None),
    text_encoder_quantization: Optional[str] = Form(None),
    quantized_gemm_mode: Optional[str] = Form(GENERATION_DEFAULTS["quantized_gemm_mode"]),
    cpu_text_encoding: bool = Form(GENERATION_DEFAULTS["cpu_text_encoding"]),
    use_torch_compile: bool = Form(False),
    keep_models_hot: bool = Form(GENERATION_DEFAULTS["keep_models_hot"]),
    vae_tiling: bool = Form(GENERATION_DEFAULTS["vae_tiling"]),
    vae_tile_threshold: int = Form(GENERATION_DEFAULTS["vae_tile_threshold"]),
    vae_tile_mode: str = Form(GENERATION_DEFAULTS["vae_tile_mode"]),
    vae_tile_global_norm: bool = Form(GENERATION_DEFAULTS["vae_tile_global_norm"]),
    color_flatten_strength: int = Form(GENERATION_DEFAULTS["color_flatten_strength"]),
    flatten_in_loop: bool = Form(GENERATION_DEFAULTS["flatten_in_loop"]),
    flatten_in_loop_last_steps: int = Form(GENERATION_DEFAULTS["flatten_in_loop_last_steps"]),
    flatten_in_loop_min_region: float = Form(GENERATION_DEFAULTS["flatten_in_loop_min_region"]),
    vae_drift_correction: bool = Form(GENERATION_DEFAULTS["vae_drift_correction"]),
    spectrum_enable: bool = Form(GENERATION_DEFAULTS["spectrum_enable"]),
    fbcache_enable: bool = Form(GENERATION_DEFAULTS["fbcache_enable"]),
    fbcache_threshold: float = Form(GENERATION_DEFAULTS["fbcache_threshold"]),
    fbcache_warmup_steps: int = Form(GENERATION_DEFAULTS["fbcache_warmup_steps"]),
    fbcache_cache_branch: int = Form(GENERATION_DEFAULTS["fbcache_cache_branch"]),
    spectrum_w: float = Form(GENERATION_DEFAULTS["spectrum_w"]),
    spectrum_w_decay: float = Form(GENERATION_DEFAULTS["spectrum_w_decay"]),
    spectrum_delta_cap: float = Form(GENERATION_DEFAULTS["spectrum_delta_cap"]),
    spectrum_m: int = Form(GENERATION_DEFAULTS["spectrum_m"]),
    spectrum_lam: float = Form(GENERATION_DEFAULTS["spectrum_lam"]),
    spectrum_warmup_steps: int = Form(GENERATION_DEFAULTS["spectrum_warmup_steps"]),
    spectrum_window_size: int = Form(GENERATION_DEFAULTS["spectrum_window_size"]),
    spectrum_flex_window: float = Form(GENERATION_DEFAULTS["spectrum_flex_window"]),
    spectrum_tail: float = Form(GENERATION_DEFAULTS["spectrum_tail"]),
    spectrum_feature_mode: str = Form(GENERATION_DEFAULTS["spectrum_feature_mode"]),
    spectrum_cache_branch: int = Form(GENERATION_DEFAULTS["spectrum_cache_branch"]),
    spectrum_max_cache: int = Form(GENERATION_DEFAULTS["spectrum_max_cache"]),
    enable_block_swap: bool = Form(False),
    blocks_to_swap: int = Form(GENERATION_DEFAULTS["blocks_to_swap"]),
    use_pinned_memory: bool = Form(False),
    block_swap_h2d_only: bool = Form(GENERATION_DEFAULTS["block_swap_h2d_only"]),
    block_swap_ring_size: int = Form(GENERATION_DEFAULTS["block_swap_ring_size"]),
    use_tipo: bool = Form(False),
    tipo_config: str = Form("{}"),  # JSON string of TIPO config
    preview_predicted_x0: bool = Form(False),  # Show predicted x0 in preview instead of current latent
    preview_decoder: str = Form("matrix"),  # Live-preview decoder for FLUX.2-VAE models: "matrix" | "taef2"
    vision_encoder_path: Optional[str] = Form(None),  # Path to SigLIP2 vision encoder safetensors
    vae_path: Optional[str] = Form(GENERATION_DEFAULTS["vae_path"]),  # Per-generation VAE override (dir or standalone VAE)
    text_encoder_path: Optional[str] = Form(GENERATION_DEFAULTS["text_encoder_path"]),  # Per-generation TE override (SD1.5/SDXL only)
    pid_sr_output: str = Form(GENERATION_DEFAULTS["pid_sr_output"]),  # PiD decoder only: "4x" | "original"
    pid_use_gemma: bool = Form(GENERATION_DEFAULTS["pid_use_gemma"]),  # PiD decoder only: opt-in runtime Gemma captioner
    pid_low_vram: bool = Form(GENERATION_DEFAULTS["pid_low_vram"]),  # PiD decoder only: opt-in row-chunked low-VRAM decode (default off, bit-identical when off)
    pid_tile_native: int = Form(GENERATION_DEFAULTS["pid_tile_native"]),  # PiD decoder only: per-tile native-res ceiling for the F9 tiled large-output decode
    pid_tile_overlap_ratio: float = Form(GENERATION_DEFAULTS["pid_tile_overlap_ratio"]),  # PiD decoder only: F9 tile feather overlap ratio
    pid_fast_large_decode: bool = Form(GENERATION_DEFAULTS["pid_fast_large_decode"]),  # PiD decoder only: opt out of F9 tiling back to the F7 whole-latent cap+bicubic path
    original_size_w: int = Form(0),  # SDXL micro-cond override: original width (0 = auto)
    original_size_h: int = Form(0),  # SDXL micro-cond override: original height (0 = auto)
    original_size_scale: float = Form(1.0),  # SDXL micro-cond: original_size = output * scale
    loop_decode: str = Form(GENERATION_DEFAULTS["loop_decode"]),  # Loop-generation decode mode: "full" | "cheap" | "none"
    input_latent_id: Optional[str] = Form(GENERATION_DEFAULTS["input_latent_id"]),  # Loop latent passthrough: start from a cached latent instead of `image`
    skip_gallery: bool = Form(GENERATION_DEFAULTS["skip_gallery"]),  # Save to disk but skip the gallery DB record/thumbnail
    image: Optional[UploadFile] = File(None),
    ref_images: List[UploadFile] = File(default=[]),  # FLUX.2 Image Edit / Vision Encoder reference images
    db: Session = Depends(get_gallery_db)
):
    """Generate image from image"""
    _reject_if_video_model("/generate/img2img")
    _reject_if_audio_model("/generate/img2img")
    lora_configs = []
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from api.arch_capabilities import check_arch_capabilities
    from api.generation_overrides import plan_overrides, apply_overrides
    from api.error_handlers import ValidationError
    # Quantized-GEMM path (quantized_gemm_mode): normalized BEFORE start_generation
    # so an invalid value is a 400 rather than a 500 from inside the run. None
    # (the default) leaves the process flags completely untouched.
    from api.quantized_gemm import normalize_mode as _normalize_qgm
    try:
        quantized_gemm_mode = _normalize_qgm(quantized_gemm_mode)
    except ValueError as exc:
        raise ValidationError("Invalid quantized_gemm_mode value", detail=str(exc))
    # Attention backend (attention_type): validated against the attention
    # registry's own vocabulary, also before start_generation. An unknown value
    # used to run native and be RECORDED as the unknown string.
    attention_type = _validated_attention_type(
        attention_type, GENERATION_DEFAULTS["attention_type"])
    if loop_decode not in ("full", "cheap", "none"):
        raise ValidationError(
            "Invalid loop_decode value",
            detail=f"loop_decode must be 'full', 'cheap', or 'none', got {loop_decode!r}",
        )
    if image is None and not input_latent_id:
        raise ValidationError("img2img requires either an 'image' upload or 'input_latent_id'")
    if image is not None and input_latent_id:
        raise ValidationError(
            "img2img accepts EXACTLY ONE of 'image' or 'input_latent_id', not both",
            detail="Provide either an uploaded image or a cached latent_id from a previous loop_decode=\"none\" step, not both.",
        )
    _override_plan = plan_overrides(pipeline_manager, vae_path, text_encoder_path)
    _gen_id = start_generation("img2img")
    try:
        # Reset cancellation flag before starting new generation
        pipeline_manager.reset_cancel_flag()

        # Load input image (skipped entirely when input_latent_id is set --
        # pipeline.generate_img2img resolves the cached latent itself and uses
        # a size-only placeholder in place of a real init_image).
        if image is not None:
            image_data = await image.read()
            init_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            init_image = None

        # Parse LoRA configs
        import json
        lora_configs = json.loads(loras) if loras else []

        # Parse ControlNet configs
        controlnet_configs = json.loads(controlnets) if controlnets else []
        controlnet_images, style_transfer, style_transfers, style_combine_mode = process_controlnet_configs(
            controlnet_configs,
            generation_type="img2img"
        )

        # Parse TIPO config
        tipo_config_dict = json.loads(tipo_config) if tipo_config else {}

        # TIPO prompt upsampling (if enabled)
        original_prompt = prompt
        if use_tipo:
            print(f"[TIPO] Upsampling prompt with TIPO...")
            try:
                # Load TIPO model if needed
                model_name = tipo_config_dict.get("model_name", "KBlueLeaf/TIPO-500M")
                if not tipo_manager.loaded or tipo_manager.model_name != model_name:
                    tipo_manager.load_model(model_name)

                # Generate upsampled prompt
                upsampled_prompt = tipo_manager.generate_prompt(
                    input_prompt=prompt,
                    tag_length=tipo_config_dict.get("tag_length", "long"),
                    nl_length=tipo_config_dict.get("nl_length", "long"),
                    temperature=tipo_config_dict.get("temperature", 1.0),
                    top_p=tipo_config_dict.get("top_p", 0.95),
                    top_k=tipo_config_dict.get("top_k", 50),
                    max_new_tokens=tipo_config_dict.get("max_new_tokens", 256),
                    category_order=tipo_config_dict.get("category_order", []),
                    enabled_categories=tipo_config_dict.get("enabled_categories", {}),
                    treat_as_nl=tipo_config_dict.get("treat_as_nl", False)
                )

                # If result is dict (tipo-kgen mode), format it to string
                if isinstance(upsampled_prompt, dict):
                    category_order = tipo_config_dict.get("category_order", [])
                    enabled_categories = tipo_config_dict.get("enabled_categories", {})

                    # If no category order specified, use default
                    if not category_order:
                        category_order = ["special", "quality", "rating", "artist", "copyright", "characters", "meta", "general"]

                    # If no enabled categories specified, enable all by default
                    if not enabled_categories:
                        enabled_categories = {cat: True for cat in category_order}
                        enabled_categories["meta"] = False  # Meta disabled by default

                    prompt = tipo_manager.format_kgen_result(
                        upsampled_prompt,
                        category_order,
                        enabled_categories
                    )
                else:
                    prompt = upsampled_prompt

                print(f"[TIPO] Original prompt: {original_prompt[:100]}...")
                print(f"[TIPO] Upsampled prompt: {prompt[:100]}...")

                # Unload TIPO model to free VRAM
                tipo_manager.unload_model()

            except Exception as e:
                print(f"[TIPO] Error during upsampling: {e}")
                print(f"[TIPO] Using original prompt")
                # Continue with original prompt on error

        # Process reference images (FLUX.2 Image Edit / Vision Encoder)
        ref_image_list = []
        if ref_images:
            for ref_img_file in ref_images:
                img_bytes = await ref_img_file.read()
                ref_image_list.append(Image.open(io.BytesIO(img_bytes)))
            print(f"[FLUX.2 Image Edit] Loaded {len(ref_image_list)} reference image(s)")

        # Load Vision Encoder if requested (non-FLUX.2 only)
        is_flux2 = pipeline_manager.current_model_info and pipeline_manager.current_model_info.get("type") == "flux2"
        if vision_encoder_path and not is_flux2:
            pipeline_manager.load_vision_encoder(vision_encoder_path)

        # Apply (or restore) the planned VAE/TE overrides on the loaded model.
        _override_meta = apply_overrides(
            pipeline_manager, _override_plan,
            pid_sr_output=pid_sr_output, pid_use_gemma=pid_use_gemma, pid_low_vram=pid_low_vram,
            pid_tile_native=pid_tile_native, pid_tile_overlap_ratio=pid_tile_overlap_ratio,
            pid_fast_large_decode=pid_fast_large_decode, prompt=prompt,
        )

        # Generate image
        params = {
            "prompt": prompt,
            "vae_path": vae_path,
            "text_encoder_path": text_encoder_path,
            "pid_sr_output": pid_sr_output,
            "pid_use_gemma": pid_use_gemma,
            "pid_low_vram": pid_low_vram,
            "pid_tile_native": pid_tile_native,
            "pid_tile_overlap_ratio": pid_tile_overlap_ratio,
            "pid_fast_large_decode": pid_fast_large_decode,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_strength": denoising_strength,
            "img2img_fix_steps": img2img_fix_steps,
            "sampler": sampler,
            "schedule_type": schedule_type,
            "seed": seed,
            "ancestral_seed": ancestral_seed,
            "width": width,
            "height": height,
            "resize_mode": resize_mode,
            "resampling_method": resampling_method,
            "loras": lora_configs,  # FLUX.2 needs this in params
            "controlnet_images": controlnet_images,
            "style_transfer": style_transfer,
            "style_transfers": style_transfers,
            "style_combine_mode": style_combine_mode,
            "developer_mode": developer_mode,
            "cfg_schedule_type": cfg_schedule_type,
            "cfg_schedule_min": cfg_schedule_min,
            "cfg_schedule_max": cfg_schedule_max,
            "cfg_schedule_power": cfg_schedule_power,
            "cfg_rescale_snr_alpha": cfg_rescale_snr_alpha,
            "dynamic_threshold_percentile": dynamic_threshold_percentile,
            "dynamic_threshold_mimic_scale": dynamic_threshold_mimic_scale,
            "nag_enable": nag_enable,
            "nag_scale": nag_scale,
            "nag_tau": nag_tau,
            "nag_alpha": nag_alpha,
            "nag_sigma_end": nag_sigma_end,
            "nag_negative_prompt": nag_negative_prompt,
            "attention_type": attention_type,
            "attention_impl": attention_impl,
            "unet_quantization": unet_quantization,
            "quantized_gemm_mode": quantized_gemm_mode,
            "original_size_w": original_size_w,
            "original_size_h": original_size_h,
            "original_size_scale": original_size_scale,
            "text_encoder_quantization": text_encoder_quantization,
            "cpu_text_encoding": cpu_text_encoding,
            "use_torch_compile": use_torch_compile,
            "keep_models_hot": keep_models_hot,
            "vae_tiling": vae_tiling,
            "vae_tile_threshold": vae_tile_threshold,
            "vae_tile_mode": vae_tile_mode,
            "vae_tile_global_norm": vae_tile_global_norm,
            "color_flatten_strength": color_flatten_strength,
            "flatten_in_loop": flatten_in_loop,
            "flatten_in_loop_last_steps": flatten_in_loop_last_steps,
            "flatten_in_loop_min_region": flatten_in_loop_min_region,
            "vae_drift_correction": vae_drift_correction,
            "spectrum_enable": spectrum_enable,
            "fbcache_enable": fbcache_enable,
            "fbcache_threshold": fbcache_threshold,
            "fbcache_warmup_steps": fbcache_warmup_steps,
            "fbcache_cache_branch": fbcache_cache_branch,
            "spectrum_w": spectrum_w,
            "spectrum_w_decay": spectrum_w_decay,
            "spectrum_delta_cap": spectrum_delta_cap,
            "spectrum_m": spectrum_m,
            "spectrum_lam": spectrum_lam,
            "spectrum_warmup_steps": spectrum_warmup_steps,
            "spectrum_window_size": spectrum_window_size,
            "spectrum_flex_window": spectrum_flex_window,
            "spectrum_tail": spectrum_tail,
            "spectrum_feature_mode": spectrum_feature_mode,
            "spectrum_cache_branch": spectrum_cache_branch,
            "spectrum_max_cache": spectrum_max_cache,
            "enable_block_swap": enable_block_swap,
            "blocks_to_swap": blocks_to_swap,
            "use_pinned_memory": use_pinned_memory,
            "block_swap_h2d_only": block_swap_h2d_only,
            "block_swap_ring_size": block_swap_ring_size,
            "preview_decoder": preview_decoder,
            "ref_images": ref_image_list,  # FLUX.2 Image Edit reference images
            "loop_decode": loop_decode,
            "input_latent_id": input_latent_id,
            "skip_gallery": skip_gallery,
        }
        params.update(_override_meta)
        print(f"img2img generation params: {sanitize_params_for_logging(params)}")

        # Set prompt chunking settings
        set_prompt_chunking_settings(
            pipeline_manager,
            prompt_chunking_mode,
            max_prompt_chunks
        )

        # Load LoRAs if specified
        pipeline_manager.img2img_pipeline, has_step_range_loras = load_loras_for_generation(
            lora_manager,
            pipeline_manager.img2img_pipeline,
            lora_configs,
            "img2img"
        )

        # Detect if SDXL
        is_sdxl = pipeline_manager.img2img_pipeline is not None and \
                  "XL" in pipeline_manager.img2img_pipeline.__class__.__name__
        is_zimage = pipeline_manager.current_model_info and \
                    pipeline_manager.current_model_info.get("type") == "zimage"
        is_deus = pipeline_manager.current_model_info and \
                  pipeline_manager.current_model_info.get("type") == "deus"
        is_flux2 = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "flux2"
        # Z-Image with SDXL VAE (4ch) needs TAESD-XL instead of TAEF1
        is_zimage_sdxl_vae = is_zimage and \
                             pipeline_manager.current_model_info.get("vae_type") == "sdxl"
        is_anima = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "anima"
        is_lens = pipeline_manager.current_model_info and \
                  pipeline_manager.current_model_info.get("type") == "lens"
        # Ideogram 4 shares AutoencoderKLFlux2's 128-ch packed latent with Lens.
        is_ideogram4 = pipeline_manager.current_model_info and \
                       pipeline_manager.current_model_info.get("type") == "ideogram4"
        is_minit2i = pipeline_manager.current_model_info and \
                     pipeline_manager.current_model_info.get("type") == "minit2i"
        minit2i_vae_type = (pipeline_manager.minit2i_components or {}).get("vae_type", "none") if is_minit2i else "none"
        is_krea2 = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "krea2"

        # Warn about parameters the loaded architecture silently ignores
        _current_arch = pipeline_manager.current_model_info.get("type") if pipeline_manager.current_model_info else None
        check_arch_capabilities(params, _current_arch)

        # Progress callback to send updates via WebSocket
        progress_callback = create_progress_callback_factory(
            taesd_manager,
            manager,
            is_sdxl,
            is_zimage,
            is_deus,
            is_zimage_sdxl_vae,
            is_flux2,
            is_anima,
            is_lens=is_lens,
            is_ideogram4=is_ideogram4,
            is_minit2i=is_minit2i,
            is_krea2=is_krea2,
            img2img_fix_steps=img2img_fix_steps,
            steps=steps,
            image_width=width,
            image_height=height,
            # For flow-matching DiTs (Anima / Z-Image / FLUX.2 / Lens), default to
            # the pred_x0 preview: x_t is mostly noise mid-denoising, while
            # pred_x0 = x_t - σ·v shows the model's current clean-image
            # estimate from the very first steps. Any explicit user override
            # via the API still wins.
            preview_predicted_x0=(preview_predicted_x0 or is_anima or is_zimage or is_flux2 or is_lens or is_ideogram4 or is_minit2i or is_krea2),
            preview_enabled=params.get("preview_enabled", True),
            preview_interval=params.get("preview_interval", 4),
            preview_decoder=params.get("preview_decoder", "matrix")
        )

        # Create step callback for LoRA step range if needed
        step_callback = None
        if has_step_range_loras:
            # Calculate actual steps based on denoising strength
            actual_steps = int(steps * denoising_strength)
            step_callback = create_lora_step_callback(
                lora_manager,
                pipeline_manager.img2img_pipeline,
                actual_steps
            )

        # Run generation in thread pool to avoid blocking event loop.
        # gpu_coordinator slot pauses any active tagger training first.
        from core.gpu_coordinator import gpu_coordinator
        from core.inference.generation_timing import generation_timer
        loop = asyncio.get_event_loop()
        _peak_gb = _estimate_gen_peak_gb(width, height, 1,
                                         pipeline_manager.current_pipeline_kind)
        generation_timer.reset()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=_peak_gb, timeout=60.0):
            # GEMM flags are process-wide; keep selection and probing in this slot.
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(quantized_gemm_mode)
            result_image, actual_seed, actual_ancestral_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: pipeline_manager.generate_img2img(params, init_image, progress_callback=progress_callback, step_callback=step_callback)
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        # Record total wall time + any phase breakdown the pipeline populated.
        apply_generation_timings(params, time.perf_counter() - _gen_start)

        # Update params with actual seeds
        params["seed"] = actual_seed
        params["ancestral_seed"] = actual_ancestral_seed

        # loop_decode="none": no VAE/PiD decode happened -- `result_image` is
        # the raw pre-unscale final latent tensor instead of a PIL.Image. Cache
        # it and return a latent_id for the next loop step's input_latent_id,
        # skipping save/thumbnail/gallery entirely.
        if not isinstance(result_image, Image.Image):
            from core.inference.latent_cache import store_latent
            latent_id = store_latent(result_image, meta={
                "width": params.get("width"),
                "height": params.get("height"),
                "seed": actual_seed,
            })
            # The latent-only path returns before the fp8_gemm block below, so
            # report the resolved quantized-GEMM path here too — a loop step
            # that chains latents must still surface a degraded "w8a8" request.
            from api.quantized_gemm import report_quantized_gemm_outcome
            report_quantized_gemm_outcome(
                quantized_gemm_mode, fp8_gemm, _current_arch
            )
            complete_generation({"latent_id": latent_id, "seed": actual_seed}, generation_id=_gen_id)
            return {"success": True, "latent_id": latent_id, "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

        # Add Vision Encoder info to params for PNG metadata and DB storage.
        # Only record VE info when THIS generation actually used reference images
        # (the VE stays loaded "sticky" across generations).
        if ref_image_list:
            ve_name, ve_hash = extract_vision_encoder_info(pipeline_manager)
            if ve_name:
                params["vision_encoder_name"] = ve_name
            if ve_hash:
                params["vision_encoder_hash"] = ve_hash

        # Add VAE identity to params. The VAE always participates in decode, so this
        # is recorded for every generation where it can be determined.
        vae_name, vae_hash = extract_vae_info(pipeline_manager)
        if vae_name:
            params["vae_name"] = vae_name
        if vae_hash:
            params["vae_hash"] = vae_hash

        # FP8 GEMM path. Recorded only for checkpoints that carry weight-only FP8
        # Linear weights, and it records the RESOLVED path rather than the flag —
        # the W8A8 path can be enabled while the device probe rejects every
        # scaling mode, and the two paths are numerically different.
        if fp8_gemm:
            params["fp8_gemm"] = fp8_gemm
        # Report an explicit "w8a8" request that resolved to the dequant path
        # (probe rejected it, or the checkpoint has no quantized layers).
        from api.quantized_gemm import report_quantized_gemm_outcome
        report_quantized_gemm_outcome(quantized_gemm_mode, fp8_gemm, _current_arch)
        # Attention backend that ACTUALLY ran (observed by the conduit), so the
        # PNG metadata and the gallery row cannot name a backend that did not.
        record_attention_backend(params, _gen_id)

        # Save image with metadata (include model info)
        filename = save_image_with_metadata(
            result_image,
            params,
            "img2img",
            model_info=pipeline_manager.current_model_info,
            generation_id=_gen_id
        )

        # skip_gallery: the file is saved (so a generation loop can still chain
        # to the next step via its path) but no thumbnail/DB record is created.
        if skip_gallery:
            complete_generation({"filename": filename, "seed": actual_seed}, generation_id=_gen_id)
            return {
                "success": True,
                "filename": filename,
                "image_path": f"/outputs/{filename}",
                "actual_seed": actual_seed,
                "warnings": get_warnings(_gen_id),
            }

        image_path = os.path.join(settings.outputs_dir, filename)
        create_thumbnail(image_path)

        # Calculate metadata (source_image is None for latent-passthrough starts
        # -- init_image is a size-only placeholder there, never the real input --
        # calculate_generation_metadata already skips source_image_hash when falsy)
        metadata = calculate_generation_metadata(
            result_image,
            lora_configs,
            extract_lora_names,
            calculate_image_hash,
            source_image=init_image if not input_latent_id else None
        )

        # Remove image objects from params before saving to DB and calculate ControlNet hashes
        params_for_db = prepare_params_for_db(params, calculate_image_hash)
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        # Extract model name and hash from current_model_info
        model_name, model_hash = extract_model_info(pipeline_manager)

        # Save to database
        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="img2img",
            image_hash=metadata["image_hash"],
            lora_names=metadata["lora_names"],
            model_name=model_name,
            model_hash=model_hash,
            result_image=result_image,
            source_image_hash=metadata.get("source_image_hash")
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except GenerationError as e:
        # Re-raise custom errors as-is
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except Exception as e:
        # Wrap unexpected errors in GenerationError
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Image-to-image generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )
    finally:
        # Unload LoRAs after generation
        if lora_configs and pipeline_manager.img2img_pipeline:
            pipeline_manager.img2img_pipeline = lora_manager.unload_loras(pipeline_manager.img2img_pipeline)

def _resolve_upscaler_model_path(model_name: str, db: Session) -> Optional[str]:
    """Resolve an upscaler_model filename to an absolute path under
    <models_dir>/upscalers/ or an additional model dir's upscalers/ subdir."""
    settings_record = db.query(UserSettings).first()
    additional_model_dirs = settings_record.model_dirs if settings_record else []
    all_dirs = [settings.models_dir] + list(additional_model_dirs)
    for base_dir in all_dirs:
        candidate_dir = os.path.join(base_dir, "upscalers")
        if not os.path.isdir(candidate_dir):
            continue
        candidate = os.path.join(candidate_dir, model_name)
        if os.path.isfile(candidate):
            return candidate
    return None


@router.post("/generate/upscale")
async def generate_upscale(
    upscaler_backend: str = Form(UPSCALE_DEFAULTS["upscaler_backend"]),
    upscaler_model: Optional[str] = Form(UPSCALE_DEFAULTS["upscaler_model"]),
    scale_factor: float = Form(UPSCALE_DEFAULTS["scale_factor"]),
    pil_resample: str = Form(UPSCALE_DEFAULTS["pil_resample"]),
    tile_size: int = Form(UPSCALE_DEFAULTS["tile_size"]),
    tile_overlap: int = Form(UPSCALE_DEFAULTS["tile_overlap"]),
    rtx_vsr_quality: str = Form(UPSCALE_DEFAULTS["rtx_vsr_quality"]),
    unsharp_enable: bool = Form(UPSCALE_DEFAULTS["unsharp_enable"]),
    unsharp_radius: float = Form(UPSCALE_DEFAULTS["unsharp_radius"]),
    unsharp_percent: int = Form(UPSCALE_DEFAULTS["unsharp_percent"]),
    unsharp_threshold: int = Form(UPSCALE_DEFAULTS["unsharp_threshold"]),
    prompt: str = Form(UPSCALE_DEFAULTS["prompt"]),
    negative_prompt: str = Form(UPSCALE_DEFAULTS["negative_prompt"]),
    diffusion_denoising_strength: float = Form(UPSCALE_DEFAULTS["diffusion_denoising_strength"]),
    steps: int = Form(UPSCALE_DEFAULTS["steps"]),
    cfg_scale: float = Form(UPSCALE_DEFAULTS["cfg_scale"]),
    sampler: str = Form(UPSCALE_DEFAULTS["sampler"]),
    schedule_type: str = Form(UPSCALE_DEFAULTS["schedule_type"]),
    attention_type: str = Form(UPSCALE_DEFAULTS["attention_type"]),
    attention_impl: str = Form(UPSCALE_DEFAULTS["attention_impl"]),
    seed: int = Form(UPSCALE_DEFAULTS["seed"]),
    diffusion_pre_upscale_mode: str = Form(UPSCALE_DEFAULTS["diffusion_pre_upscale_mode"]),
    image: UploadFile = File(...),
    db: Session = Depends(get_gallery_db)
):
    """Upscale an image via PIL resample, a spandrel super-resolution model,
    RTX Video Super Resolution (nvvfx), or diffusion tile upscale (img2img
    per tile with the currently loaded model)."""
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from core.upscaler import run_upscale
    _gen_id = start_generation("upscale")
    try:
        # Load input image
        image_data = await image.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        params: Dict[str, Any] = {
            "seed": 0,
            "upscaler_backend": upscaler_backend,
            "upscaler_model": upscaler_model,
            "scale_factor": scale_factor,
            "pil_resample": pil_resample,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "rtx_vsr_quality": rtx_vsr_quality,
            "unsharp_enable": unsharp_enable,
            "unsharp_radius": unsharp_radius,
            "unsharp_percent": unsharp_percent,
            "unsharp_threshold": unsharp_threshold,
        }

        if upscaler_backend == "diffusion":
            params.update({
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "diffusion_denoising_strength": diffusion_denoising_strength,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler": sampler,
                "schedule_type": schedule_type,
                "attention_type": _validated_attention_type(
                    attention_type, UPSCALE_DEFAULTS["attention_type"]),
                "attention_impl": attention_impl,
                "seed": seed,
                "diffusion_pre_upscale_mode": diffusion_pre_upscale_mode,
            })

        print(f"upscale generation params: {sanitize_params_for_logging(params)}")

        if upscaler_backend == "spandrel" or (upscaler_backend == "diffusion" and diffusion_pre_upscale_mode == "model"):
            if not upscaler_model:
                raise CustomValidationError(
                    "upscaler_model is required when upscaler_backend='spandrel' "
                    "(or diffusion_pre_upscale_mode='model')"
                )
            model_path = _resolve_upscaler_model_path(upscaler_model, db)
            if not model_path:
                raise NotFoundError(
                    "Upscaler model not found",
                    detail=f"model: {upscaler_model}"
                )
            params["_upscaler_model_path"] = model_path

        if upscaler_backend == "diffusion" and pipeline_manager.current_model_info is None:
            raise CustomValidationError(
                "No diffusion model loaded",
                detail="Load a model before using upscaler_backend='diffusion'."
            )

        # Progress callback: tiles reported as step/total_steps.
        # send_progress_sync is thread-safe (called from the executor thread).
        def progress_callback(step, total_steps, sub_progress=None):
            from api.generation_status import update_progress
            total = max(total_steps, 1)
            manager.send_progress_sync(
                step, total, f"Upscaling tile {step}/{total}", sub_progress=sub_progress)
            if sub_progress is None:
                update_progress(step, total)

        from core.gpu_coordinator import gpu_coordinator
        loop = asyncio.get_event_loop()
        if upscaler_backend == "diffusion":
            target_w = max(1, round(input_image.width * scale_factor))
            target_h = max(1, round(input_image.height * scale_factor))
            est_w = tile_size if tile_size > 0 else target_w
            est_h = tile_size if tile_size > 0 else target_h
            _peak_gb = _estimate_gen_peak_gb(est_w, est_h, 1, pipeline_manager.current_pipeline_kind)
        else:
            _peak_gb = _estimate_gen_peak_gb(input_image.width, input_image.height, 1, "unknown")
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=_peak_gb, timeout=60.0):
            result_image, upscale_warnings = await _run_generation_in_executor(
                loop, executor,
                lambda: run_upscale(params, input_image, progress_callback=progress_callback, pipeline_manager=pipeline_manager)
            )
        apply_generation_timings(params, time.perf_counter() - _gen_start)
        if upscaler_backend == "diffusion":
            record_attention_backend(params, _gen_id)

        for w in upscale_warnings:
            print(f"[Upscale] Warning: {w}")

        # Record actual output dims so PNG metadata reflects the result, not defaults
        params["width"] = result_image.width
        params["height"] = result_image.height

        # Calculate metadata first so source_image_hash lands in the PNG text chunk
        metadata = calculate_generation_metadata(
            result_image,
            [],
            extract_lora_names,
            calculate_image_hash,
            source_image=input_image
        )
        params["source_image_hash"] = metadata.get("source_image_hash")

        is_diffusion = upscaler_backend == "diffusion"
        diffusion_model_info = pipeline_manager.current_model_info if is_diffusion else None

        # Save image with metadata
        filename = save_image_with_metadata(
            result_image,
            params,
            "upscale",
            model_info=diffusion_model_info,
            generation_id=_gen_id
        )
        image_path = os.path.join(settings.outputs_dir, filename)
        create_thumbnail(image_path)

        # Remove internal-only keys and non-serializable objects before DB save
        params_for_db = {k: v for k, v in params.items() if not k.startswith("_")}
        _effective_warnings = get_warnings(_gen_id) + upscale_warnings
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        if is_diffusion:
            diffusion_model_name, diffusion_model_hash = extract_model_info(pipeline_manager)
            record_model_name = diffusion_model_name or upscaler_backend
            record_model_hash = diffusion_model_hash
            actual_seed = params.get("seed", 0)
        else:
            record_model_name = params.get("upscaler_model") or upscaler_backend
            record_model_hash = params.get("upscaler_model_hash", "")
            actual_seed = 0

        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="upscale",
            image_hash=metadata["image_hash"],
            lora_names=None,
            model_name=record_model_name,
            model_hash=record_model_hash,
            result_image=result_image,
            source_image_hash=metadata.get("source_image_hash")
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {
            "success": True,
            "image": db_image.to_dict(),
            "actual_seed": actual_seed,
            "warnings": get_warnings(_gen_id) + upscale_warnings,
        }

    except (GenerationError, CustomValidationError, NotFoundError) as e:
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Upscale failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )


def _record_media_gemm_outcome(params, fp8_gemm, arch):
    """Record the RESOLVED quantized-GEMM path and warn if `w8a8` degraded.

    The video AND audio counterpart of the identical pair of calls in the image
    routes (named `_record_video_*` while LTX-2.3 was the only non-image arch
    with quantized Linears; ACE-Step's three audio routes now share it).
    `fp8_gemm` is recorded into `params` (and so into the sidecar JSON and the
    gallery row) because the W8A8 and dequantized paths are numerically
    different, so an output is not reproducible without knowing which one ran.
    """
    from api.quantized_gemm import report_quantized_gemm_outcome
    if fp8_gemm:
        params["fp8_gemm"] = fp8_gemm
    report_quantized_gemm_outcome(params.get("quantized_gemm_mode"), fp8_gemm, arch)


def _normalize_media_qgm(mode):
    """`quantized_gemm_mode` for the video and audio routes, as a 400 on junk.

    The image routes each inline this two-liner; the three LTX-2.3 video routes
    and the three ACE-Step audio routes share it rather than repeating the same
    try/except six more times.

    Both archs pass the same test for accepting the parameter at all: `ltx2` and
    `acestep` are in `QUANTIZED_LINEAR_ARCHS`, their loaders swap in
    `Int8Linear`/`Fp8Linear` for a weight-only quantized checkpoint, and
    `_ltx2_runtime_int8` / `_acestep_runtime_int8` produce exactly those classes
    from a bf16 one -- so the flags this parameter sets govern modules these
    routes really can be running, rather than being decorative.
    """
    from api.error_handlers import ValidationError
    from api.quantized_gemm import normalize_mode as _normalize_qgm
    try:
        return _normalize_qgm(mode)
    except ValueError as exc:
        raise ValidationError("Invalid quantized_gemm_mode value", detail=str(exc))


@router.post("/generate/txt2vid")
async def generate_txt2vid(
    request: Txt2VidRequest,
    db: Session = Depends(get_gallery_db)
):
    """Generate a video from a text prompt (LTX-2.3 or MiniMax-H3).

    Produces an H.264 mp4 (with an audio track when audio_enable is true) and a
    gallery row. Requires a video model to be loaded.

    Any field the client omits is filled from the LOADED ARCHITECTURE's video
    defaults (`param_defaults.video_defaults_for_arch`), and the geometry is
    then validated against that architecture's `TemporalSpec` — so neither the
    defaults nor the constraints are written out here.
    """
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from utils.video_utils import save_video_with_metadata

    params = request.dict()
    # Quantized-GEMM path (quantized_gemm_mode): normalized BEFORE
    # start_generation so an invalid value is a 400 rather than a 500 from inside
    # the run. None (the default) leaves the process flags completely untouched.
    params["quantized_gemm_mode"] = _normalize_media_qgm(params.get("quantized_gemm_mode"))
    # Attention backend, validated against the attention registry's vocabulary
    # (same 400-before-start_generation contract as the image routes).
    params["attention_type"] = _validated_attention_type(
        params.get("attention_type"), TXT2VID_DEFAULTS["attention_type"])

    if not (getattr(pipeline_manager, "is_ltx2_model", False)
            or getattr(pipeline_manager, "is_minimax_h3_model", False)):
        raise CustomValidationError(
            "No video model loaded",
            detail="Load an LTX-2.3 or MiniMax-H3 video model before calling /generate/txt2vid.",
        )

    # Per-architecture defaults, THEN spec-driven geometry validation. Order
    # matters: a field the client omitted must be filled from the loaded arch's
    # overlay before it is validated, or an omitted `num_frames` would be checked
    # against LTX-2.3's 121 on a MiniMax-H3 request and snapped for no reason.
    _vid_arch = (pipeline_manager.current_model_info or {}).get("type")
    _vid_defaults = resolve_video_defaults(params, request.model_fields_set, _vid_arch)

    # Training-free reference-style transfer (video). No image-conditioning
    # ControlNets are supported for LTX-2.3 -- `controlnets` exists only to
    # carry an `is_style_transfer` entry (see core.inference.style_ltx2).
    # style_transfers (plural, 0+ entries) + style_combine_mode are threaded
    # through so multi-reference (N>1) style transfer reaches the LTX-2.3
    # backend (pipeline_backends/ltx2.py._ltx2_style_configs); style_transfer
    # (singular) stays for the untouched single-ref path.
    _, style_transfer, style_transfers, style_combine_mode = process_controlnet_configs(
        params.get("controlnets") or [],
        generation_type="txt2vid",
    )
    params["style_transfer"] = style_transfer
    params["style_transfers"] = style_transfers
    params["style_combine_mode"] = style_combine_mode
    params.pop("controlnets", None)

    _gen_id = start_generation("txt2vid")
    try:
        pipeline_manager.reset_cancel_flag()

        # Spec-driven geometry validation (TemporalSpec): a hard 400 for the
        # canvas and, on an arch whose spec says so, a snap-with-warning for the
        # clip length. Inside the try because a snap emits a `warnings[]` entry,
        # which needs the generation context started above; a raise from here is
        # still re-raised as a 4xx by the except clause below.
        validate_video_geometry(params, _vid_arch)
        # Same place, same source (TemporalSpec): a step count the arch's
        # scheduler cannot build a schedule from. It has to be checked HERE and
        # not left to the sampler -- MiniMax-H3 raised for num_inference_steps=1
        # only after a full text encode, turning a bad request into a 500 that
        # cost tens of seconds.
        validate_video_steps(params, _vid_arch)

        # VAE/TE overrides are unsupported on both video archs
        # (accepted-but-ignored). The plan drops them (arch gating) and
        # check_arch_capabilities warns; the apply call clears any stale override
        # from a previous image generation.
        from api.arch_capabilities import check_arch_capabilities
        from api.generation_overrides import plan_overrides, apply_overrides
        _override_plan = plan_overrides(pipeline_manager, params.get("vae_path"), params.get("text_encoder_path"))
        apply_overrides(pipeline_manager, _override_plan)
        # The RESOLVED per-arch defaults, so "the user set this to a non-default
        # value" is judged against the numbers this request was filled from
        # rather than against the image defaults (which carry none of these
        # keys, so every video value would read as user-set).
        check_arch_capabilities(params, _vid_arch, defaults=_vid_defaults)

        print(f"txt2vid generation params: {sanitize_params_for_logging(params)}")

        # Progress via the shared WebSocket step broadcast (mirrors the upscale route).
        def progress_callback(step, total_steps, sub_progress=None):
            from api.generation_status import update_progress
            total = max(total_steps, 1)
            # A mid-step tick names the step still RUNNING; `step` itself keeps
            # counting COMPLETED steps, and the integer-only generation status is
            # left at the last boundary.
            shown = step + 1 if sub_progress is not None else step
            manager.send_progress_sync(
                step, total, f"Generating video: step {shown}/{total}", sub_progress=sub_progress)
            if sub_progress is None:
                update_progress(step, total)

        from core.gpu_coordinator import gpu_coordinator
        from core.inference.generation_timing import generation_timer
        loop = asyncio.get_event_loop()
        # The phase timer ACCUMULATES, so a generation that records phases and
        # never resets reports the sum of every generation since the process
        # started. The four image routes reset; this one did not, which was
        # harmless only for as long as no video backend recorded a phase.
        generation_timer.reset()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=40, timeout=120.0):
            # GEMM flags are process-wide; keep selection and probing in this slot.
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(params.get("quantized_gemm_mode"))
            frames, audio, audio_sample_rate, actual_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: pipeline_manager.generate_txt2vid(params, progress_callback=progress_callback)
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        apply_generation_timings(params, time.perf_counter() - _gen_start)
        _record_media_gemm_outcome(params, fp8_gemm, _vid_arch)
        # Attention backend that ACTUALLY ran, observed by the conduit. Empty
        # (nothing recorded) on an architecture that does not route attention
        # through it -- LTX-2.3 drives diffusers' own dispatch -- which is why
        # nothing is written rather than the request being echoed back.
        record_attention_backend(params, _gen_id)
        # Which MiniMax-H3 checkpoint (fl2va/ref2va) actually ran; a no-op for
        # LTX-2.3. See record_model_variant for why this can't come from the
        # request.
        record_model_variant(params, pipeline_manager)

        # VAE identity, exactly as the four image routes record it. The video
        # routes never called this, so a video gallery row carried no
        # vae_name/vae_hash at all -- even though `extract_vae_info` DERIVES the
        # per-arch components dict and has always been able to name an LTX-2.3
        # or MiniMax-H3 VAE. MiniMax-H3 owns two autoencoders; the one recorded
        # is the VIDEO VAE, which is the one that produced the frames.
        vae_name, vae_hash = extract_vae_info(pipeline_manager)
        if vae_name:
            params["vae_name"] = vae_name
        if vae_hash:
            params["vae_hash"] = vae_hash

        params["seed"] = actual_seed

        # Encode mp4 (+ mux audio), poster PNG, and sidecar JSON.
        filename, _preview_filename = save_video_with_metadata(
            frames,
            audio,
            audio_sample_rate,
            params,
            "txt2vid",
            model_info=pipeline_manager.current_model_info,
        )

        # Thumbnail from the poster PNG (same base name as the mp4).
        base_name = os.path.splitext(filename)[0]
        poster_path = os.path.join(settings.outputs_dir, f"{base_name}.png")
        if os.path.exists(poster_path):
            create_thumbnail(poster_path)

        # Hash the saved MASTER file's bytes (see calculate_bytes_hash's
        # docstring for why video/audio identity is byte-, not pixel-, hashed).
        # This is what lets a later request's source_image_hash resolve back
        # to this row via GET /images/by-hash/{hash}.
        _media_hash = await _hash_saved_media(os.path.join(settings.outputs_dir, filename))

        # Record video-specific fields into parameters JSON for the gallery.
        num_frames_out = int(frames.shape[0])
        fps_out = float(params.get("frame_rate", 24.0))
        params_for_db = {k: v for k, v in params.items() if not k.startswith("_")}
        params_for_db["num_frames"] = num_frames_out
        params_for_db["fps"] = fps_out
        params_for_db["duration"] = (num_frames_out / fps_out) if fps_out else 0.0
        params_for_db["audio_enable"] = bool(params.get("audio_enable", True) and audio is not None)
        params_for_db["is_video"] = True
        # See calculate_bytes_hash's docstring: file-bytes hashing, not the
        # still-image pixel-PNG hash. Recorded so a later audit can tell this
        # row apart from a pre-backfill legacy one without guessing.
        params_for_db["hash_kind"] = "file_bytes"
        # The gallery's `cfg_scale` / `steps` COLUMNS are filled by
        # `create_db_image_record` from the keys the IMAGE routes use. A video
        # request carries neither `cfg_scale` nor `steps`, so every video row so
        # far recorded that helper's literal fallbacks -- 7.0 and 20 -- which no
        # video generation ever used. On a guidance-distilled architecture that
        # is not merely stale: a row claiming CFG 7.0 contradicts the ignored-
        # guidance warning contract stated in the same response. Map the real
        # video keys onto the columns instead.
        params_for_db["cfg_scale"] = float(params.get("guidance_scale", 1.0))
        params_for_db["steps"] = int(params.get("num_inference_steps", 0))
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        model_name, model_hash = extract_model_info(pipeline_manager)

        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="txt2vid",
            image_hash=_media_hash,
            lora_names=extract_lora_names(params.get("loras") or []),
            model_name=model_name,
            model_hash=model_hash,
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except (GenerationError, CustomValidationError, NotFoundError) as e:
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Text-to-video generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )


@router.post("/generate/txt2aud")
async def generate_txt2aud(
    request: Txt2AudRequest,
    db: Session = Depends(get_gallery_db)
):
    """Generate music/audio from a text caption + lyrics using the loaded
    ACE-Step 1.5 model.

    Produces a lossless FLAC file and a gallery row. Requires an ACE-Step
    model to be loaded.
    """
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from utils.audio_utils import save_audio_with_metadata

    params = request.dict()
    # Quantized-GEMM path (quantized_gemm_mode): normalized BEFORE
    # start_generation so an invalid value is a 400 rather than a 500 from inside
    # the run. None (the default) leaves the process flags completely untouched.
    params["quantized_gemm_mode"] = _normalize_media_qgm(params.get("quantized_gemm_mode"))

    # MiniMax-H3 first: it DOES generate audio, but only jointly with video, so
    # the generic "no ACE-Step model" message below would misdescribe it.
    _reject_if_video_model_on_audio_route("/generate/txt2aud")
    if not getattr(pipeline_manager, "is_acestep_model", False):
        raise CustomValidationError(
            "No ACE-Step model loaded",
            detail="Load an ACE-Step 1.5 audio model before calling /generate/txt2aud.",
        )

    _gen_id = start_generation("txt2aud")
    try:
        pipeline_manager.reset_cancel_flag()

        from api.arch_capabilities import check_arch_capabilities
        _acestep_arch = (pipeline_manager.current_model_info or {}).get("type")
        check_arch_capabilities(params, _acestep_arch)

        print(f"txt2aud generation params: {sanitize_params_for_logging(params)}")

        # Progress via the shared WebSocket step broadcast (mirrors txt2vid).
        def progress_callback(step, total_steps, sub_progress=None):
            from api.generation_status import update_progress
            total = max(total_steps, 1)
            manager.send_progress_sync(
                step, total, f"Generating audio: step {step}/{total}", sub_progress=sub_progress)
            if sub_progress is None:
                update_progress(step, total)

        from core.gpu_coordinator import gpu_coordinator
        loop = asyncio.get_event_loop()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=_PEAK_VRAM_GB_BY_KIND["acestep"], timeout=120.0):
            # GEMM flags are process-wide; keep selection and probing in this slot.
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(params.get("quantized_gemm_mode"))
            waveform, sample_rate, actual_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: pipeline_manager.generate_txt2aud(params, progress_callback=progress_callback)
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        apply_generation_timings(params, time.perf_counter() - _gen_start)
        _record_media_gemm_outcome(params, fp8_gemm, _acestep_arch)

        params["seed"] = actual_seed

        # Encode FLAC, waveform PNG (thumbnail seed), and sidecar JSON.
        filename = save_audio_with_metadata(
            waveform,
            sample_rate,
            params,
            "txt2aud",
            model_info=pipeline_manager.current_model_info,
        )

        # Thumbnail from the waveform PNG (same base name as the FLAC).
        base_name = os.path.splitext(filename)[0]
        waveform_png_path = os.path.join(settings.outputs_dir, f"{base_name}.png")
        if os.path.exists(waveform_png_path):
            create_thumbnail(waveform_png_path)

        # Hash the saved FLAC file's bytes (see calculate_bytes_hash's docstring).
        _media_hash = await _hash_saved_media(os.path.join(settings.outputs_dir, filename))

        # Record audio-specific fields into parameters JSON for the gallery.
        num_samples = int(waveform.shape[-1])
        duration_s = (num_samples / sample_rate) if sample_rate else 0.0
        params_for_db = {k: v for k, v in params.items() if not k.startswith("_")}
        # Audio has no visual dimensions; do not let create_db_image_record's
        # width/height fallback (512) fabricate a fake resolution.
        params_for_db["width"] = 0
        params_for_db["height"] = 0
        params_for_db["duration"] = duration_s
        params_for_db["sample_rate"] = sample_rate
        params_for_db["is_audio"] = True
        params_for_db["hash_kind"] = "file_bytes"
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        model_name, model_hash = extract_model_info(pipeline_manager)

        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="txt2aud",
            image_hash=_media_hash,
            lora_names=extract_lora_names(params.get("loras") or []),
            model_name=model_name,
            model_hash=model_hash,
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except (GenerationError, CustomValidationError, NotFoundError) as e:
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Text-to-audio generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )


@router.post("/generate/aud2aud")
async def generate_aud2aud(
    prompt: str = Form(AUD2AUD_DEFAULTS["prompt"]),
    lyrics: Optional[str] = Form(AUD2AUD_DEFAULTS["lyrics"]),
    seed: int = Form(AUD2AUD_DEFAULTS["seed"]),
    inference_steps: int = Form(AUD2AUD_DEFAULTS["inference_steps"]),
    guidance_scale: float = Form(AUD2AUD_DEFAULTS["guidance_scale"]),
    shift: float = Form(AUD2AUD_DEFAULTS["shift"]),
    cover_strength: float = Form(AUD2AUD_DEFAULTS["cover_strength"]),
    mode: str = Form(AUD2AUD_DEFAULTS["mode"]),
    repaint_start: float = Form(AUD2AUD_DEFAULTS["repaint_start"]),
    repaint_end: float = Form(AUD2AUD_DEFAULTS["repaint_end"]),
    vocal_language: str = Form(AUD2AUD_DEFAULTS["vocal_language"]),
    loras: str = Form("[]"),  # JSON string of LoRA configs
    # Weight-only quantization; see the Txt2AudRequest fields for both axes.
    unet_quantization: Optional[str] = Form(AUD2AUD_DEFAULTS["unet_quantization"]),
    quantized_gemm_mode: Optional[str] = Form(AUD2AUD_DEFAULTS["quantized_gemm_mode"]),
    reference_audio: UploadFile = File(...),
    db: Session = Depends(get_gallery_db)
):
    """Generate a cover OR repaint (audio-to-audio) from a reference clip
    using the loaded ACE-Step 1.5 model.

    Multipart form: an uploaded reference audio clip plus the cover/repaint
    parameters. `mode="cover"` (default) re-renders the WHOLE reference
    under a new caption/lyrics (the reference is VAE-encoded and fed back to
    the DiT as the cover context, `is_covers=True`). `mode="repaint"`
    regenerates only `[repaint_start, repaint_end)` seconds of the
    reference, keeping everything outside that window (approximately)
    unchanged -- see
    `core.pipeline_backends.acestep.AceStepMixin._generate_aud2aud_acestep`
    for the full mechanism (latent-domain repaint hold + boundary blend,
    plus a post-decode waveform splice). Duration is always derived from
    the reference's length, not user-supplied. Produces a lossless FLAC
    file and a gallery row. Requires an ACE-Step model to be loaded.
    """
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from utils.audio_utils import save_audio_with_metadata

    # Parse LoRA configs (same JSON-string-of-configs convention as txt2img/img2img/inpaint)
    lora_configs = json.loads(loras) if loras else []

    mode = (mode or "cover").strip().lower()
    if mode not in ("cover", "repaint"):
        raise CustomValidationError(
            "Invalid aud2aud mode",
            detail=f"mode must be 'cover' or 'repaint', got {mode!r}.",
        )
    if mode == "repaint" and repaint_end <= repaint_start:
        raise CustomValidationError(
            "Invalid repaint range",
            detail=f"repaint_end ({repaint_end}) must be greater than repaint_start ({repaint_start}).",
        )

    params = {
        "prompt": prompt,
        "lyrics": lyrics,
        "seed": seed,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "cover_strength": cover_strength,
        "mode": mode,
        "repaint_start": repaint_start,
        "repaint_end": repaint_end,
        "vocal_language": vocal_language,
        "loras": lora_configs,
        "unet_quantization": unet_quantization,
        # Normalized here (not after start_generation) so an invalid value is a
        # 400 rather than a 500 from inside the run.
        "quantized_gemm_mode": _normalize_media_qgm(quantized_gemm_mode),
    }

    # MiniMax-H3 first: it DOES generate audio, but only jointly with video, so
    # the generic "no ACE-Step model" message below would misdescribe it.
    _reject_if_video_model_on_audio_route("/generate/aud2aud")
    if not getattr(pipeline_manager, "is_acestep_model", False):
        raise CustomValidationError(
            "No ACE-Step model loaded",
            detail="Load an ACE-Step 1.5 audio model before calling /generate/aud2aud.",
        )

    # Read the uploaded reference audio clip.
    try:
        reference_audio_bytes = await reference_audio.read()
        if not reference_audio_bytes:
            raise ValueError("uploaded file is empty")
    except Exception as e:
        raise CustomValidationError(
            "Failed to read the uploaded reference audio",
            detail=str(e),
        )

    _gen_id = start_generation("aud2aud")
    try:
        pipeline_manager.reset_cancel_flag()

        from api.arch_capabilities import check_arch_capabilities
        _acestep_arch = (pipeline_manager.current_model_info or {}).get("type")
        check_arch_capabilities(params, _acestep_arch)

        print(f"aud2aud generation params: {sanitize_params_for_logging(params)}")

        # "aud2aud" (cover) or "repaint" -- used for both the saved FLAC's
        # filename prefix and the gallery's generation_type column. The
        # `mode` field in params (and params_for_db below) already carries
        # the same distinction, so this is purely for readability/filtering;
        # nothing downstream (ImageGrid, GeneratedImage type) hardcodes an
        # enum of generation_type values -- it is gated on `is_audio` instead.
        _generation_type = "repaint" if mode == "repaint" else "aud2aud"

        # Progress via the shared WebSocket step broadcast (mirrors txt2aud).
        def progress_callback(step, total_steps, sub_progress=None):
            from api.generation_status import update_progress
            total = max(total_steps, 1)
            label = "repaint" if mode == "repaint" else "cover"
            manager.send_progress_sync(
                step, total, f"Generating {label}: step {step}/{total}", sub_progress=sub_progress)
            if sub_progress is None:
                update_progress(step, total)

        from core.gpu_coordinator import gpu_coordinator
        loop = asyncio.get_event_loop()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=_PEAK_VRAM_GB_BY_KIND["acestep"], timeout=120.0):
            # GEMM flags are process-wide; keep selection and probing in this slot.
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(params.get("quantized_gemm_mode"))
            waveform, sample_rate, actual_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: pipeline_manager.generate_aud2aud(params, reference_audio_bytes, progress_callback=progress_callback)
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        apply_generation_timings(params, time.perf_counter() - _gen_start)
        _record_media_gemm_outcome(params, fp8_gemm, _acestep_arch)

        params["seed"] = actual_seed

        # Hash the reference clip (mirrors img2img/img2vid's source_image_hash).
        params["source_audio_hash"] = calculate_bytes_hash(reference_audio_bytes)

        # Encode FLAC, waveform PNG (thumbnail seed), and sidecar JSON.
        filename = save_audio_with_metadata(
            waveform,
            sample_rate,
            params,
            _generation_type,
            model_info=pipeline_manager.current_model_info,
        )

        # Thumbnail from the waveform PNG (same base name as the FLAC).
        base_name = os.path.splitext(filename)[0]
        waveform_png_path = os.path.join(settings.outputs_dir, f"{base_name}.png")
        if os.path.exists(waveform_png_path):
            create_thumbnail(waveform_png_path)

        # Hash the saved FLAC file's bytes (see calculate_bytes_hash's docstring).
        _media_hash = await _hash_saved_media(os.path.join(settings.outputs_dir, filename))

        # Record audio-specific fields into parameters JSON for the gallery.
        num_samples = int(waveform.shape[-1])
        duration_s = (num_samples / sample_rate) if sample_rate else 0.0
        params_for_db = {k: v for k, v in params.items() if not k.startswith("_")}
        # Audio has no visual dimensions; do not let create_db_image_record's
        # width/height fallback (512) fabricate a fake resolution.
        params_for_db["width"] = 0
        params_for_db["height"] = 0
        params_for_db["duration"] = duration_s
        params_for_db["sample_rate"] = sample_rate
        params_for_db["is_audio"] = True
        params_for_db["hash_kind"] = "file_bytes"
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        model_name, model_hash = extract_model_info(pipeline_manager)

        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type=_generation_type,
            image_hash=_media_hash,
            lora_names=extract_lora_names(params.get("loras") or []),
            model_name=model_name,
            model_hash=model_hash,
            source_image_hash=params["source_audio_hash"],
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except (GenerationError, CustomValidationError, NotFoundError) as e:
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Audio-to-audio generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )


@router.post("/generate/outpaint/audio")
async def generate_outpaint_audio(
    prompt: str = Form(OUTPAINT_AUDIO_DEFAULTS["prompt"]),
    lyrics: Optional[str] = Form(OUTPAINT_AUDIO_DEFAULTS["lyrics"]),
    seed: int = Form(OUTPAINT_AUDIO_DEFAULTS["seed"]),
    inference_steps: int = Form(OUTPAINT_AUDIO_DEFAULTS["inference_steps"]),
    guidance_scale: float = Form(OUTPAINT_AUDIO_DEFAULTS["guidance_scale"]),
    shift: float = Form(OUTPAINT_AUDIO_DEFAULTS["shift"]),
    vocal_language: str = Form(OUTPAINT_AUDIO_DEFAULTS["vocal_language"]),
    # Placement (new for audio outpaint). All in SECONDS.
    total_duration: float = Form(OUTPAINT_AUDIO_DEFAULTS["total_duration"]),
    input_offset_sec: float = Form(OUTPAINT_AUDIO_DEFAULTS["input_offset_sec"]),
    input_trim_start_sec: float = Form(OUTPAINT_AUDIO_DEFAULTS["input_trim_start_sec"]),
    input_trim_end_sec: float = Form(OUTPAINT_AUDIO_DEFAULTS["input_trim_end_sec"]),
    loras: str = Form("[]"),  # JSON string of LoRA configs
    # Weight-only quantization; see the Txt2AudRequest fields for both axes.
    unet_quantization: Optional[str] = Form(OUTPAINT_AUDIO_DEFAULTS["unet_quantization"]),
    quantized_gemm_mode: Optional[str] = Form(OUTPAINT_AUDIO_DEFAULTS["quantized_gemm_mode"]),
    reference_audio: UploadFile = File(...),
    db: Session = Depends(get_gallery_db)
):
    """Audio temporal outpaint (extend, ACE-Step 1.5): place a (optionally
    trimmed) input clip at a time offset inside a LONGER `total_duration`
    output timeline and generate the audio before and/or after it.

    Multipart form: an uploaded `reference_audio` clip (the input to place)
    plus the placement/generation parameters below. Structurally the INVERSE
    of `/generate/aud2aud`'s `mode=repaint`: repaint holds everything
    OUTSIDE a window and generates INSIDE it; outpaint holds the placed
    input window itself and generates OUTSIDE it (before and/or after) --
    see `core.pipeline_backends.acestep.AceStepMixin._generate_audoutpaint_acestep`
    for the full mechanism (inverted latent-domain repaint hold + boundary
    blend, plus a post-decode waveform splice).

    **Strict preservation guarantee:** the placed input span in the returned
    audio is sample-exact to the decoded 48kHz/16-bit representation of the
    (trimmed, stereo/48kHz-normalized) input clip, regardless of
    `total_duration`/`input_offset_sec`. This is enforced by an
    unconditional post-decode waveform splice performed AFTER generation
    (`_acestep_apply_outpaint_waveform_splice`, which splices the FULL
    trimmed input regardless of the latent hold-window length, so no tail
    is ever dropped), with a short (10ms) declick crossfade confined
    entirely to the GENERATED side of each boundary so every preserved
    sample stays untouched. Because the app's standard audio output (FLAC)
    is already lossless, this makes the output sample-exact to that decoded
    representation end to end -- unlike video outpaint, no separate
    lossless toggle is needed. An upload that is not already 48kHz/16-bit
    stereo is faithfully resampled/requantized once during normalization,
    not byte-identical to the original file.

    **Requirements:**
    - An ACE-Step model must be loaded (otherwise 400).
    - `total_duration` in (0, 240] seconds.
    - The (trimmed) input clip must fit inside `total_duration`; if
      `input_offset_sec` would push it out of bounds it is CLAMPED (with a
      warning) rather than rejected, but a trimmed input longer than
      `total_duration` itself is rejected (400).
    """
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from utils.audio_utils import save_audio_with_metadata

    # Parse LoRA configs (same JSON-string-of-configs convention as txt2aud/aud2aud)
    lora_configs = json.loads(loras) if loras else []

    if total_duration <= 0 or total_duration > 240.0:
        raise CustomValidationError(
            "Invalid total_duration",
            detail=f"total_duration must be in (0, 240] seconds, got {total_duration}.",
        )
    if input_offset_sec < 0:
        raise CustomValidationError(
            "Invalid input_offset_sec",
            detail=f"input_offset_sec must be >= 0, got {input_offset_sec}.",
        )
    if input_trim_start_sec < 0 or input_trim_end_sec < 0:
        raise CustomValidationError(
            "Invalid input trim",
            detail=f"input_trim_start_sec/input_trim_end_sec must be >= 0, got "
                   f"{input_trim_start_sec}/{input_trim_end_sec}.",
        )

    # MiniMax-H3 first: it DOES generate audio, but only jointly with video, so
    # the generic "no ACE-Step model" message below would misdescribe it.
    _reject_if_video_model_on_audio_route("/generate/outpaint/audio")
    if not getattr(pipeline_manager, "is_acestep_model", False):
        raise CustomValidationError(
            "No ACE-Step model loaded",
            detail="Load an ACE-Step 1.5 audio model before calling /generate/outpaint/audio.",
        )

    # Read the uploaded input clip.
    try:
        reference_audio_bytes = await reference_audio.read()
        if not reference_audio_bytes:
            raise ValueError("uploaded file is empty")
    except Exception as e:
        raise CustomValidationError(
            "Failed to read the uploaded input audio",
            detail=str(e),
        )

    # Cheap, header-only duration probe (soundfile -- no full decode) so an
    # impossible trim/placement surfaces as a 400 BEFORE a GPU slot is
    # reserved, mirroring /generate/outpaint/video's up-front trimmed_len
    # check. The backend re-validates against the ACTUAL trimmed + VAE-
    # encoded length (authoritative -- see
    # AceStepMixin._generate_audoutpaint_acestep); this is a best-effort
    # early-reject only and is skipped (falls through to the backend) if the
    # header can't be read this way.
    try:
        import soundfile as sf
        import io as _io
        _info = sf.info(_io.BytesIO(reference_audio_bytes))
        _src_duration = (float(_info.frames) / float(_info.samplerate)) if _info.samplerate else None
    except Exception:
        _src_duration = None

    if _src_duration is not None:
        _trimmed_duration = _src_duration - input_trim_start_sec - input_trim_end_sec
        if _trimmed_duration <= 0:
            raise CustomValidationError(
                "Audio outpaint input trim leaves no audio",
                detail=f"Uploaded clip is ~{_src_duration:.3f}s; "
                       f"input_trim_start_sec={input_trim_start_sec}, input_trim_end_sec={input_trim_end_sec}.",
            )
        if _trimmed_duration > total_duration:
            raise CustomValidationError(
                "Input audio (after trim) does not fit inside total_duration",
                detail=f"Trimmed input is ~{_trimmed_duration:.3f}s; total_duration is {total_duration:.3f}s. "
                       f"Either trim the input further or increase total_duration.",
            )

    params = {
        "prompt": prompt,
        "lyrics": lyrics,
        "seed": seed,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "vocal_language": vocal_language,
        "total_duration": total_duration,
        "input_offset_sec": input_offset_sec,
        "input_trim_start_sec": input_trim_start_sec,
        "input_trim_end_sec": input_trim_end_sec,
        "loras": lora_configs,
        "unet_quantization": unet_quantization,
        # Normalized here (not after start_generation) so an invalid value is a
        # 400 rather than a 500 from inside the run.
        "quantized_gemm_mode": _normalize_media_qgm(quantized_gemm_mode),
    }

    _gen_id = start_generation("outpaint_aud")
    try:
        pipeline_manager.reset_cancel_flag()

        from api.arch_capabilities import check_arch_capabilities
        _acestep_arch = (pipeline_manager.current_model_info or {}).get("type")
        check_arch_capabilities(params, _acestep_arch)

        print(f"outpaint_aud generation params: {sanitize_params_for_logging(params)}")

        def progress_callback(step, total_steps, sub_progress=None):
            from api.generation_status import update_progress
            total = max(total_steps, 1)
            manager.send_progress_sync(
                step, total, f"Generating outpaint: step {step}/{total}", sub_progress=sub_progress)
            if sub_progress is None:
                update_progress(step, total)

        from core.gpu_coordinator import gpu_coordinator
        loop = asyncio.get_event_loop()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=_PEAK_VRAM_GB_BY_KIND["acestep"], timeout=120.0):
            # GEMM flags are process-wide; keep selection and probing in this slot.
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(params.get("quantized_gemm_mode"))
            waveform, sample_rate, actual_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: pipeline_manager.generate_aud_outpaint(params, reference_audio_bytes, progress_callback=progress_callback)
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        apply_generation_timings(params, time.perf_counter() - _gen_start)
        _record_media_gemm_outcome(params, fp8_gemm, _acestep_arch)

        params["seed"] = actual_seed

        # Hash the input clip (mirrors aud2aud's source_audio_hash).
        params["source_audio_hash"] = calculate_bytes_hash(reference_audio_bytes)

        # Encode FLAC, waveform PNG (thumbnail seed), and sidecar JSON.
        filename = save_audio_with_metadata(
            waveform,
            sample_rate,
            params,
            "outpaint_aud",
            model_info=pipeline_manager.current_model_info,
        )

        # Thumbnail from the waveform PNG (same base name as the FLAC).
        base_name = os.path.splitext(filename)[0]
        waveform_png_path = os.path.join(settings.outputs_dir, f"{base_name}.png")
        if os.path.exists(waveform_png_path):
            create_thumbnail(waveform_png_path)

        # Hash the saved FLAC file's bytes (see calculate_bytes_hash's docstring).
        _media_hash = await _hash_saved_media(os.path.join(settings.outputs_dir, filename))

        # Record audio-specific fields into parameters JSON for the gallery.
        num_samples = int(waveform.shape[-1])
        duration_s = (num_samples / sample_rate) if sample_rate else 0.0
        params_for_db = {k: v for k, v in params.items() if not k.startswith("_")}
        # Audio has no visual dimensions; do not let create_db_image_record's
        # width/height fallback (512) fabricate a fake resolution.
        params_for_db["width"] = 0
        params_for_db["height"] = 0
        params_for_db["duration"] = duration_s
        params_for_db["sample_rate"] = sample_rate
        params_for_db["is_audio"] = True
        params_for_db["hash_kind"] = "file_bytes"
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        model_name, model_hash = extract_model_info(pipeline_manager)

        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="outpaint_aud",
            image_hash=_media_hash,
            lora_names=extract_lora_names(params.get("loras") or []),
            model_name=model_name,
            model_hash=model_hash,
            source_image_hash=params["source_audio_hash"],
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except (GenerationError, CustomValidationError, NotFoundError) as e:
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Audio outpaint generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )


@router.post("/generate/img2vid")
async def generate_img2vid(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(IMG2VID_DEFAULTS["negative_prompt"]),
    # The six keys a per-arch overlay can change (param_defaults
    # VIDEO_GEN_ARCH_OVERLAYS) are declared as `Form(None)` SENTINELS, not as
    # their base values: a multipart route has no `model_fields_set`, so an
    # omitted field is only distinguishable from a sent one by the sentinel.
    # Whatever comes back None is filled from `video_defaults_for_arch` below --
    # which is what makes an omitted `num_frames` resolve to MiniMax-H3's 124
    # instead of being validated (and snapped) against LTX-2.3's 121. The
    # openapi schema keeps documenting the base values.
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    num_frames: Optional[int] = Form(None),
    frame_rate: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    seed: int = Form(IMG2VID_DEFAULTS["seed"]),
    num_videos_per_prompt: int = Form(IMG2VID_DEFAULTS["num_videos_per_prompt"]),
    max_sequence_length: int = Form(IMG2VID_DEFAULTS["max_sequence_length"]),
    audio_enable: bool = Form(IMG2VID_DEFAULTS["audio_enable"]),
    blocks_to_swap: int = Form(IMG2VID_DEFAULTS["blocks_to_swap"]),
    fuse_output_proj: bool = Form(IMG2VID_DEFAULTS["fuse_output_proj"]),
    fbcache_enable: bool = Form(IMG2VID_DEFAULTS["fbcache_enable"]),
    fbcache_threshold: float = Form(IMG2VID_DEFAULTS["fbcache_threshold"]),
    fbcache_warmup_steps: int = Form(IMG2VID_DEFAULTS["fbcache_warmup_steps"]),
    spectrum_enable: bool = Form(IMG2VID_DEFAULTS["spectrum_enable"]),
    spectrum_w: float = Form(IMG2VID_DEFAULTS["spectrum_w"]),
    spectrum_w_decay: float = Form(IMG2VID_DEFAULTS["spectrum_w_decay"]),
    spectrum_delta_cap: float = Form(IMG2VID_DEFAULTS["spectrum_delta_cap"]),
    spectrum_m: int = Form(IMG2VID_DEFAULTS["spectrum_m"]),
    spectrum_lam: float = Form(IMG2VID_DEFAULTS["spectrum_lam"]),
    spectrum_warmup_steps: int = Form(IMG2VID_DEFAULTS["spectrum_warmup_steps"]),
    spectrum_window_size: int = Form(IMG2VID_DEFAULTS["spectrum_window_size"]),
    spectrum_flex_window: float = Form(IMG2VID_DEFAULTS["spectrum_flex_window"]),
    spectrum_tail: float = Form(IMG2VID_DEFAULTS["spectrum_tail"]),
    spectrum_max_cache: int = Form(IMG2VID_DEFAULTS["spectrum_max_cache"]),
    vae_path: Optional[str] = Form(IMG2VID_DEFAULTS["vae_path"]),
    text_encoder_path: Optional[str] = Form(IMG2VID_DEFAULTS["text_encoder_path"]),
    unet_quantization: Optional[str] = Form(IMG2VID_DEFAULTS["unet_quantization"]),
    quantized_gemm_mode: Optional[str] = Form(IMG2VID_DEFAULTS["quantized_gemm_mode"]),
    # See Txt2VidRequest.attention_type: honored by MiniMax-H3, ignored by LTX-2.3.
    attention_type: str = Form(IMG2VID_DEFAULTS["attention_type"]),
    controlnets: str = Form("[]"),  # JSON string; only is_style_transfer entries are meaningful for LTX-2.3
    # Generation-time LoRA. See Txt2VidRequest.loras.
    loras: str = Form("[]"),
    # WHERE the uploaded `image` sits on the generated clip (MiniMax-H3). 0 is
    # the first frame -- the placement every request made before this existed --
    # and -1 is the clip's last frame AFTER `num_frames` is snapped below, which
    # is the whole reason for the sentinel: the snap happens here, so a client
    # cannot compute the last index itself.
    input_image_frame_index: int = Form(IMG2VID_DEFAULTS["input_image_frame_index"]),
    # OPTIONAL since ia2v: a request that sends `input_audio` may send no image
    # at all, and is then conditioned on the prompt and the track alone. The
    # "at least one conditioning medium" rule is enforced below, per
    # architecture -- LTX-2.3's image-to-video pipeline has no other input.
    image: Optional[UploadFile] = File(None),
    # OPTIONAL second keyframe: the LAST frame. MiniMax-H3's `fl2va` workflow
    # conditions on the two ENDS of the clip (0-2 visual anchors); LTX-2.3
    # conditions on the first frame only and declares this unsupported, so it is
    # accepted-and-warned there rather than refused -- one endpoint, two archs.
    #
    # It stays a live field rather than being deprecated in favour of the
    # keyframe list: it is exactly "a keyframe at -1", it costs nothing to fold
    # in, and LTX-2.3's capability warning for it already exists.
    last_frame_image: Optional[UploadFile] = File(None),
    # ADDITIONAL anchors and their placements, positionally paired. Upload order
    # is NOT semantic (unlike /generate/ref2vid's reference lists): the anchors
    # are packed in ascending frame order, so what a client sends is a set of
    # (image, frame) pairs.
    keyframe_images: List[UploadFile] = File([]),
    keyframe_frame_indices: List[int] = Form([]),
    # An audio track the video is generated AGAINST (MiniMax-H3). Its rows are
    # pinned at t = 1.0 -- exactly clean, since the forward process is
    # `x_t = t*x0 + (1-t)*noise` -- across the WHOLE clip, and the muxed output
    # carries the source samples rather than a decode of rows nothing wrote.
    input_audio: Optional[UploadFile] = File(None),
    db: Session = Depends(get_gallery_db)
):
    """Generate a video from uploaded media (LTX-2.3 or MiniMax-H3).

    Multipart form: the txt2vid parameters plus the media this endpoint
    conditions on -- a keyframe `image`, optionally a `last_frame_image` and
    further `keyframe_images` with their `keyframe_frame_indices`, and
    optionally an `input_audio` track. On LTX-2.3 the keyframe is VAE-encoded
    and pinned as frame 0 (placement and audio conditioning are declared
    unsupported and warned); on MiniMax-H3 each uploaded frame becomes a
    single-frame visual conditioning anchor held at the model's
    `keyframe_noise_aug` level and pinned at ITS OWN frame's rotary position.
    Produces an H.264 mp4 (with an audio track when audio_enable is true) and a
    gallery row.

    WHAT "img2vid" MEANS NOW: the route that carries the media a video is
    conditioned on, images and/or audio, at least one of them. `input_audio`
    conditions the clip on its own (measured), so the image is optional when a
    track is sent -- but the endpoint keeps its name and its shape rather than
    growing a fourth video route, and /generate/txt2vid stays the JSON route
    that uploads nothing.

    Placement is resolved after the clip length is validated and snapped, and
    it is resolved for the whole request at once (`plan_keyframe_placements`):
    `-1` becomes the snapped clip's last index, the two ends become the
    `"first"`/`"last"` anchors that reproduce the pre-placement layout byte for
    byte, and the anchors are packed in ascending frame order. The audio track
    is prepared (resampled, trimmed, length-checked) in the same place and for
    the same reason: its required length is a function of the SNAPPED clip
    length.

    Any field the client omits is filled from the LOADED ARCHITECTURE's video
    defaults, and the geometry is then validated against that architecture's
    `TemporalSpec` -- the same two steps, in the same order, as
    /generate/txt2vid.
    """
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from utils.video_utils import save_video_with_metadata

    # The optional last-frame upload, normalised once: an absent part and a part
    # with no filename both mean "no last frame".
    _last_frame = last_frame_image if (last_frame_image is not None
                                       and last_frame_image.filename) else None
    # Same normalisation for the keyframe list (an empty multipart part counts
    # as absent), then the positional pairing is validated before anything is
    # read: a mismatched request must not first pay for the file reads.
    _keyframes = [f for f in (keyframe_images or []) if f is not None and f.filename]
    _keyframe_indices = list(keyframe_frame_indices or [])
    # `image` is a File(None) now, so it gets the same test as every other
    # upload here: a part with no filename is a part that was not sent.
    _image = image if (image is not None and image.filename) else None
    _input_audio = input_audio if (input_audio is not None and input_audio.filename) else None
    if len(_keyframes) != len(_keyframe_indices):
        raise CustomValidationError(
            "keyframe_images and keyframe_frame_indices must be the same length",
            detail=f"Got {len(_keyframes)} keyframe image(s) and {len(_keyframe_indices)} "
                   f"frame index/indices. The two lists are positional: entry n of "
                   f"keyframe_frame_indices is the placement of entry n of keyframe_images.",
        )

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "num_videos_per_prompt": num_videos_per_prompt,
        "max_sequence_length": max_sequence_length,
        "audio_enable": audio_enable,
        "blocks_to_swap": blocks_to_swap,
        "fuse_output_proj": fuse_output_proj,
        "fbcache_enable": fbcache_enable,
        "fbcache_threshold": fbcache_threshold,
        "fbcache_warmup_steps": fbcache_warmup_steps,
        "spectrum_enable": spectrum_enable,
        "spectrum_w": spectrum_w,
        "spectrum_w_decay": spectrum_w_decay,
        "spectrum_delta_cap": spectrum_delta_cap,
        "spectrum_m": spectrum_m,
        "spectrum_lam": spectrum_lam,
        "spectrum_warmup_steps": spectrum_warmup_steps,
        "spectrum_window_size": spectrum_window_size,
        "spectrum_flex_window": spectrum_flex_window,
        "spectrum_tail": spectrum_tail,
        "spectrum_max_cache": spectrum_max_cache,
        "vae_path": vae_path,
        "text_encoder_path": text_encoder_path,
        "unet_quantization": unet_quantization,
        # Normalized here (before start_generation) so junk is a 400, not a 500.
        "quantized_gemm_mode": _normalize_media_qgm(quantized_gemm_mode),
        # Validated (400 on an unknown backend) rather than silently absorbed
        # into native, which is what recorded a backend that never ran.
        "attention_type": _validated_attention_type(
            attention_type, IMG2VID_DEFAULTS["attention_type"]),
        # The uploaded FILENAME, not the bytes: it is what the gallery row and
        # the capability warning can carry, and `None` is what "no last frame"
        # means for both. The image itself is read below.
        #
        # ONE test decides both this record and the read below (`_last_frame`),
        # so "a last frame was sent" cannot be answered two different ways: a
        # part present but EMPTY (`filename == ""`) is treated as absent, rather
        # than warning about an ignored keyframe on LTX-2.3 while conditioning
        # on nothing, or being read as a keyframe the warning never mentioned.
        "last_frame_image": recover_upload_filename(_last_frame.filename) if _last_frame is not None else None,
        # The placement fields, recorded the same way: the FILENAMES and the
        # requested indices, never the bytes. `params.copy()` carries them to
        # the gallery row; the RESOLVED placements are added after num_frames
        # has been snapped (a requested -1 means nothing without that).
        "input_image_frame_index": input_image_frame_index,
        "keyframe_images": [recover_upload_filename(f.filename) for f in _keyframes] or None,
        "keyframe_frame_indices": list(_keyframe_indices) or None,
        # The ia2v track, recorded the same way: the uploaded FILENAME, never
        # the samples. Non-None is also what makes `check_arch_capabilities`
        # read the field as user-set and warn on an architecture that ignores
        # it (the default in IMG2VID_DEFAULTS is None).
        "input_audio": recover_upload_filename(_input_audio.filename) if _input_audio is not None else None,
    }

    # Generation-time LoRA (JSON string, same convention as the image routes'
    # multipart `loras` field). See Txt2VidRequest.loras.
    import json
    params["loras"] = json.loads(loras) if loras else []

    # Training-free reference-style transfer (video). See generate_txt2vid's
    # identical wiring / core.inference.style_ltx2 for the mechanism.
    # style_transfers (plural, 0+ entries) + style_combine_mode are threaded
    # through so multi-reference (N>1) style transfer reaches the LTX-2.3
    # backend (pipeline_backends/ltx2.py._ltx2_style_configs); style_transfer
    # (singular) stays for the untouched single-ref path.
    _controlnet_configs = json.loads(controlnets) if controlnets else []
    _, style_transfer, style_transfers, style_combine_mode = process_controlnet_configs(
        _controlnet_configs, generation_type="img2vid"
    )
    params["style_transfer"] = style_transfer
    params["style_transfers"] = style_transfers
    params["style_combine_mode"] = style_combine_mode

    if not (getattr(pipeline_manager, "is_ltx2_model", False)
            or getattr(pipeline_manager, "is_minimax_h3_model", False)):
        raise CustomValidationError(
            "No video model loaded",
            detail="Load an LTX-2.3 or MiniMax-H3 video model before calling /generate/img2vid.",
        )
    # ---- Partition gate (the mirror of /generate/ref2vid's, in the other
    # direction). This endpoint's whole request is keyframe conditioning, which
    # is a trained behaviour of MiniMax-H3's `fl2va` partition; the `ref2va` one
    # was trained to read reference blocks instead. Until this phase the
    # direction was unguarded and such a request simply ran. ----
    if getattr(pipeline_manager, "is_minimax_h3_model", False):
        _h3_variant = ((pipeline_manager.current_model_info or {}).get("variant") or "").lower()
        if _h3_variant == "ref2va":
            raise CustomValidationError(
                "The loaded MiniMax-H3 transformer is the ref2va variant, not fl2va",
                detail="Keyframe conditioning through this endpoint is offered on the `fl2va` "
                       "partition, which serves /generate/txt2vid, /generate/img2vid and "
                       "/generate/outpaint/video. Anchors do bind on the ref2va partition when "
                       "they are laid out from its post-reference rotary origin (measured), but "
                       "this endpoint carries no references to lay them out after -- that "
                       "combination is /generate/ref2vid's `keyframe_images`/"
                       "`keyframe_frame_indices` fields, not this endpoint's. Load "
                       "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors, or send "
                       "the request to /generate/ref2vid with the loaded checkpoint.",
            )

    # ---- At least one conditioning medium, and WHICH ones this architecture
    # can read. `image` stopped being a required part when `input_audio`
    # shipped: a pinned track conditions the clip on its own (measured), so an
    # imageless request is a real ia2v request rather than a malformed img2vid
    # one. It is still required on LTX-2.3, whose image-to-video pipeline takes
    # no other conditioning input at all -- there the missing part is a refusal
    # with that reason, not an accepted-and-warned field. ----
    if _image is None and _input_audio is None and not _keyframes:
        raise CustomValidationError(
            "img2vid needs something to condition on",
            detail="Send a keyframe `image` (and optionally `last_frame_image` / "
                   "`keyframe_images`), or an `input_audio` track on MiniMax-H3, or both. A "
                   "request that uploads no media at all is /generate/txt2vid.",
        )
    if _image is None and getattr(pipeline_manager, "is_ltx2_model", False):
        raise CustomValidationError(
            "LTX-2.3 image-to-video needs an input image",
            detail="LTX-2.3's image-to-video pipeline conditions on one uploaded frame and has no "
                   "other conditioning input -- it takes no per-keyframe frame index and no audio "
                   "track -- so there is nothing for this request to generate from. Send `image`, "
                   "or use /generate/txt2vid.",
        )

    # Per-architecture defaults, THEN spec-driven geometry validation -- the same
    # order, and the same two helpers, as /generate/txt2vid. `_provided` is the
    # multipart equivalent of Pydantic's `model_fields_set`: the overlaid keys
    # that came back from their `Form(None)` sentinel with a value.
    _vid_arch = (pipeline_manager.current_model_info or {}).get("type")
    #
    # Every OTHER key counts as provided: a multipart field that is not one of
    # the sentinels already carries its declared `Form()` default, so re-filling
    # it from the resolved map would overwrite REAL values -- `prompt` included,
    # which `IMG2VID_DEFAULTS` carries as "".
    _omitted = {key for key, value in (
        ("width", width), ("height", height), ("num_frames", num_frames),
        ("frame_rate", frame_rate), ("num_inference_steps", num_inference_steps),
        ("guidance_scale", guidance_scale),
    ) if value is None}
    _provided = set(params) - _omitted
    _vid_defaults = resolve_video_defaults(params, _provided, _vid_arch, IMG2VID_DEFAULTS)

    # Read the uploaded keyframe(s), and the ia2v track if there is one. Every
    # decode happens here, before the generation context and the GPU slot, so a
    # malformed upload is a 400 that reserved nothing.
    input_image = None
    if _image is not None:
        try:
            image_data = await _image.read()
            input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise CustomValidationError(
                "Failed to read the uploaded keyframe image",
                detail=str(e),
            )
    input_audio_waveform = None
    input_audio_sample_rate = None
    if _input_audio is not None:
        from core.models.minimax_h3 import h3_references as _h3refs
        try:
            input_audio_waveform, input_audio_sample_rate = _h3refs.decode_audio_bytes(
                await _input_audio.read())
        except Exception as e:
            raise CustomValidationError(
                "Failed to read the uploaded input audio track",
                detail=str(e),
            )
    last_input_image = None
    if _last_frame is not None:
        try:
            last_frame_data = await _last_frame.read()
            last_input_image = Image.open(io.BytesIO(last_frame_data)).convert("RGB")
        except Exception as e:
            raise CustomValidationError(
                "Failed to read the uploaded last-frame keyframe image",
                detail=str(e),
            )
    extra_keyframe_images = []
    for _position, _upload in enumerate(_keyframes):
        try:
            _data = await _upload.read()
            extra_keyframe_images.append(Image.open(io.BytesIO(_data)).convert("RGB"))
        except Exception as e:
            raise CustomValidationError(
                f"Failed to read keyframe_images[{_position}]",
                detail=str(e),
            )

    _gen_id = start_generation("img2vid")
    try:
        pipeline_manager.reset_cancel_flag()

        # Inside the try because a snap emits a `warnings[]` entry, which needs
        # the generation context started above; a raise from here is still
        # re-raised as a 4xx by the except clause below. See the same pair of
        # calls in /generate/txt2vid for what each one owns.
        validate_video_geometry(params, _vid_arch)
        validate_video_steps(params, _vid_arch)

        from api.arch_capabilities import check_arch_capabilities, arch_supports_feature
        from api.generation_status import add_warning
        from api.generation_overrides import plan_overrides, apply_overrides
        _override_plan = plan_overrides(pipeline_manager, params.get("vae_path"), params.get("text_encoder_path"))
        apply_overrides(pipeline_manager, _override_plan)
        # The RESOLVED per-arch VIDEO defaults, not the image ones:
        # `guidance_scale`, `num_frames`, `frame_rate` and friends do not exist
        # in GENERATION_DEFAULTS at all, so without this every video request
        # would read as having set them to a non-default value -- and any arch
        # that declares one of them unsupported (MiniMax-H3 declares both halves
        # of the guidance pair) would warn on EVERY request, including the
        # default one. `_vid_defaults` is the same map this request's omitted
        # fields were filled from.
        check_arch_capabilities(params, _vid_arch, defaults=_vid_defaults)

        # ---- Keyframe placement, resolved AFTER the snap above, because a
        # requested -1 means "the last frame of the clip that will actually be
        # generated" and `validate_video_geometry` is what decides that length.
        # Only for an architecture that honors placement: LTX-2.3 pins the
        # uploaded image as frame 0, has already been warned about by
        # `check_arch_capabilities`, and keeps the pre-placement call shape. ----
        keyframe_plan = None
        # Every shape of THIS request that is outside MiniMax's model card,
        # collected here and warned about ONCE below -- placement and audio
        # conditioning are the same claim ("the released weights honor this
        # mechanism at a position the card does not describe") and a user should
        # read it once, not once per feature.
        undocumented = []
        if arch_supports_feature(_vid_arch, "keyframe_placement"):
            placement_requests = []
            if _image is not None:
                placement_requests.append(("image", input_image_frame_index))
            placement_requests += [(f"keyframe_images[{position}]", index)
                                   for position, index in enumerate(_keyframe_indices)]
            if _last_frame is not None:
                placement_requests.append(("last_frame_image", -1))
            plan = plan_keyframe_placements(placement_requests, params.get("num_frames") or 0)

            sources = {"image": input_image, "last_frame_image": last_input_image}
            for position, pil in enumerate(extra_keyframe_images):
                sources[f"keyframe_images[{position}]"] = pil
            keyframe_plan = [(entry["anchor"], sources[entry["source"]])
                             for entry in plan["anchors"]]
            params["keyframe_resolved_frames"] = [entry["frame"] for entry in plan["anchors"]]
            undocumented += plan["undocumented"]

        # ---- ia2v: prepare the track against the SNAPPED clip length, for the
        # same reason placement is resolved here. How many samples the model
        # needs is `round(T/24*40)` audio latents x 800 samples, and how many
        # the mux needs is `round(T/24*32000)`; both are functions of the length
        # `validate_video_geometry` just decided, so neither can be checked at
        # the client. A track shorter than that is a 400 naming both durations
        # -- padding the remainder with silence would build a half-pinned,
        # half-silent timeline, which is a shape nothing has measured. ----
        input_audio_prepared = None
        if input_audio_waveform is not None and arch_supports_feature(_vid_arch, "audio_conditioning"):
            from core.models.minimax_h3 import h3_references as _h3refs
            # The rates come from the LOADED components, not from this module's
            # idea of them: the number that decides this 400 and the number that
            # decides the encode slice in `_generate_minimax_h3` have to be the
            # same one, or a checkpoint with different audio geometry would make
            # the route promise a length the encoder does not want. (The
            # function's own defaults stay for a caller that has no components.)
            _h3_components = getattr(pipeline_manager, "minimax_h3_components", None) or {}
            try:
                input_audio_prepared = _h3refs.prepare_pinned_audio(
                    input_audio_waveform, int(input_audio_sample_rate),
                    num_frames=int(params.get("num_frames") or 0),
                    fps=float(_h3_components.get("fps") or params.get("frame_rate") or 24.0),
                    target_sample_rate=int(_h3_components.get("audio_sample_rate") or 32000),
                    latent_rate=float(_h3_components.get("audio_latent_rate") or 40.0),
                )
            except ValueError as e:
                raise CustomValidationError(
                    "The uploaded input audio track cannot condition this clip",
                    detail=str(e),
                )
            undocumented.append(
                "an input audio track pinned clean across the whole clip, which the video is "
                "generated against")
            if not params.get("audio_enable", True):
                add_warning(
                    "audio_enable is false and an input_audio track was sent: the track still "
                    "conditions the video (its rows ride the packed sequence at t = 1.0), and "
                    "nothing is muxed into the output file.",
                    code="minimax_h3_input_audio_not_muxed",
                )

        # ONE warning for the whole request, not one per anchor and not one per
        # feature: the shapes below work on the released weights (measured) and
        # are simply outside what MiniMax documents, so the entry states that
        # scope and nothing else. A working feature is not buried in warnings.
        if undocumented:
            add_warning(
                "This request conditions MiniMax-H3 outside the documented shape ("
                + "; ".join(undocumented)
                + f"). {MINIMAX_H3_DOCUMENTED_ANCHOR_SCOPE}; the same pinning mechanism is used "
                  "here at other positions and on the audio rows.",
                code="minimax_h3_undocumented_conditioning",
            )

        print(f"img2vid generation params: {sanitize_params_for_logging(params)}")

        def progress_callback(step, total_steps, sub_progress=None):
            from api.generation_status import update_progress
            total = max(total_steps, 1)
            # A mid-step tick names the step still RUNNING; `step` itself keeps
            # counting COMPLETED steps, and the integer-only generation status is
            # left at the last boundary.
            shown = step + 1 if sub_progress is not None else step
            manager.send_progress_sync(
                step, total, f"Generating video: step {shown}/{total}", sub_progress=sub_progress)
            if sub_progress is None:
                update_progress(step, total)

        from core.gpu_coordinator import gpu_coordinator
        from core.inference.generation_timing import generation_timer
        loop = asyncio.get_event_loop()
        # The phase timer ACCUMULATES across generations. Now that a video
        # backend (MiniMax-H3, via /generate/txt2vid) records phases, an
        # img2vid that never resets would stamp the PREVIOUS generation's
        # per-phase seconds onto its own gallery row.
        generation_timer.reset()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=40, timeout=120.0):
            # GEMM flags are process-wide; keep selection and probing in this slot.
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(params.get("quantized_gemm_mode"))
            frames, audio, audio_sample_rate, actual_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: pipeline_manager.generate_img2vid(
                    params, input_image, progress_callback=progress_callback,
                    last_frame_image=last_input_image, keyframes=keyframe_plan,
                    input_audio=input_audio_prepared)
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        apply_generation_timings(params, time.perf_counter() - _gen_start)
        _record_media_gemm_outcome(params, fp8_gemm, _vid_arch)
        # Attention backend that ACTUALLY ran, observed by the conduit. Empty
        # (nothing recorded) on an architecture that does not route attention
        # through it -- LTX-2.3 drives diffusers' own dispatch -- which is why
        # nothing is written rather than the request being echoed back.
        record_attention_backend(params, _gen_id)
        record_model_variant(params, pipeline_manager)

        # See the note on the same call in /generate/txt2vid.
        vae_name, vae_hash = extract_vae_info(pipeline_manager)
        if vae_name:
            params["vae_name"] = vae_name
        if vae_hash:
            params["vae_hash"] = vae_hash

        params["seed"] = actual_seed

        # Hash the keyframe (reuses the img2img/upscale metadata helper). An
        # imageless ia2v request has no source image to hash, and the column
        # stays null rather than being filled with the generated frame -- that
        # hash means "what this was made from", and audio is not it.
        metadata = calculate_generation_metadata(
            Image.fromarray(frames[0]),
            [],
            extract_lora_names,
            calculate_image_hash,
            source_image=input_image,
        ) if input_image is not None else {}
        params["source_image_hash"] = metadata.get("source_image_hash")

        # Encode mp4 (+ mux audio), poster PNG, and sidecar JSON.
        filename, _preview_filename = save_video_with_metadata(
            frames,
            audio,
            audio_sample_rate,
            params,
            "img2vid",
            model_info=pipeline_manager.current_model_info,
        )

        # Thumbnail from the poster PNG (same base name as the mp4).
        base_name = os.path.splitext(filename)[0]
        poster_path = os.path.join(settings.outputs_dir, f"{base_name}.png")
        if os.path.exists(poster_path):
            create_thumbnail(poster_path)

        # Hash the saved MASTER file's bytes (see calculate_bytes_hash's docstring).
        _media_hash = await _hash_saved_media(os.path.join(settings.outputs_dir, filename))

        # Record video-specific fields into parameters JSON for the gallery.
        num_frames_out = int(frames.shape[0])
        fps_out = float(params.get("frame_rate", 24.0))
        params_for_db = {k: v for k, v in params.items() if not k.startswith("_")}
        params_for_db["num_frames"] = num_frames_out
        params_for_db["fps"] = fps_out
        params_for_db["duration"] = (num_frames_out / fps_out) if fps_out else 0.0
        params_for_db["audio_enable"] = bool(params.get("audio_enable", True) and audio is not None)
        params_for_db["is_video"] = True
        params_for_db["hash_kind"] = "file_bytes"
        # The gallery's `cfg_scale` / `steps` COLUMNS are filled by
        # `create_db_image_record` from the keys the IMAGE routes use. A video
        # request carries neither `cfg_scale` nor `steps`, so every video row so
        # far recorded that helper's literal fallbacks -- 7.0 and 20 -- which no
        # video generation ever used. On a guidance-distilled architecture that
        # is not merely stale: a row claiming CFG 7.0 contradicts the ignored-
        # guidance warning contract stated in the same response. Map the real
        # video keys onto the columns instead.
        params_for_db["cfg_scale"] = float(params.get("guidance_scale", 1.0))
        params_for_db["steps"] = int(params.get("num_inference_steps", 0))
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        model_name, model_hash = extract_model_info(pipeline_manager)

        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="img2vid",
            image_hash=_media_hash,
            lora_names=extract_lora_names(params.get("loras") or []),
            model_name=model_name,
            model_hash=model_hash,
            source_image_hash=metadata.get("source_image_hash"),
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except (GenerationError, CustomValidationError, NotFoundError) as e:
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Image-to-video generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )


@router.post("/generate/ref2vid")
async def generate_ref2vid(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(REF2VID_DEFAULTS["negative_prompt"]),
    # The six overlaid keys are `Form(None)` SENTINELS, exactly as in
    # /generate/img2vid -- see that route for why a multipart endpoint cannot
    # use Pydantic's `model_fields_set`.
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    num_frames: Optional[int] = Form(None),
    frame_rate: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    seed: int = Form(REF2VID_DEFAULTS["seed"]),
    num_videos_per_prompt: int = Form(REF2VID_DEFAULTS["num_videos_per_prompt"]),
    max_sequence_length: int = Form(REF2VID_DEFAULTS["max_sequence_length"]),
    audio_enable: bool = Form(REF2VID_DEFAULTS["audio_enable"]),
    blocks_to_swap: int = Form(REF2VID_DEFAULTS["blocks_to_swap"]),
    fuse_output_proj: bool = Form(REF2VID_DEFAULTS["fuse_output_proj"]),
    fbcache_enable: bool = Form(REF2VID_DEFAULTS["fbcache_enable"]),
    fbcache_threshold: float = Form(REF2VID_DEFAULTS["fbcache_threshold"]),
    fbcache_warmup_steps: int = Form(REF2VID_DEFAULTS["fbcache_warmup_steps"]),
    spectrum_enable: bool = Form(REF2VID_DEFAULTS["spectrum_enable"]),
    spectrum_w: float = Form(REF2VID_DEFAULTS["spectrum_w"]),
    spectrum_w_decay: float = Form(REF2VID_DEFAULTS["spectrum_w_decay"]),
    spectrum_delta_cap: float = Form(REF2VID_DEFAULTS["spectrum_delta_cap"]),
    spectrum_m: int = Form(REF2VID_DEFAULTS["spectrum_m"]),
    spectrum_lam: float = Form(REF2VID_DEFAULTS["spectrum_lam"]),
    spectrum_warmup_steps: int = Form(REF2VID_DEFAULTS["spectrum_warmup_steps"]),
    spectrum_window_size: int = Form(REF2VID_DEFAULTS["spectrum_window_size"]),
    spectrum_flex_window: float = Form(REF2VID_DEFAULTS["spectrum_flex_window"]),
    spectrum_tail: float = Form(REF2VID_DEFAULTS["spectrum_tail"]),
    spectrum_max_cache: int = Form(REF2VID_DEFAULTS["spectrum_max_cache"]),
    vae_path: Optional[str] = Form(REF2VID_DEFAULTS["vae_path"]),
    text_encoder_path: Optional[str] = Form(REF2VID_DEFAULTS["text_encoder_path"]),
    unet_quantization: Optional[str] = Form(REF2VID_DEFAULTS["unet_quantization"]),
    quantized_gemm_mode: Optional[str] = Form(REF2VID_DEFAULTS["quantized_gemm_mode"]),
    attention_type: str = Form(REF2VID_DEFAULTS["attention_type"]),
    reference_image_size: str = Form(REF2VID_DEFAULTS["reference_image_size"]),
    # The references. Every list is read in UPLOAD ORDER, and that order is
    # semantic -- it labels the references in the prompt presentation and lays
    # them out on the packed sequence's rotary clock.
    reference_images: List[UploadFile] = File([]),
    reference_videos: List[UploadFile] = File([]),
    reference_video_audios: List[UploadFile] = File([]),
    reference_audios: List[UploadFile] = File([]),
    # C5: keyframe anchors, laid out AFTER the reference blocks. Positional,
    # like /generate/img2vid's `keyframe_images`/`keyframe_frame_indices` --
    # there is no single "the" keyframe here, so every anchor uses this pair.
    keyframe_images: List[UploadFile] = File([]),
    keyframe_frame_indices: List[int] = Form([]),
    # Generation-time LoRA. See Txt2VidRequest.loras.
    loras: str = Form("[]"),
    db: Session = Depends(get_gallery_db)
):
    """Generate a video from omni-references (MiniMax-H3 `ref2va` only).

    Multipart form: up to 9 image, 3 video and 3 audio reference files (12 in
    total) plus the txt2vid parameters. Each visual reference is VAE-encoded and
    packed ahead of the generated rows, where the sampler never writes it, and
    each reference is additionally shown to the Qwen3-VL conditioner as a
    labelled vision block so the prompt can refer to it by name.

    A DEDICATED endpoint rather than more files on /generate/img2vid: that
    endpoint conditions on 1-2 keyframes pinned to the ends of the generated
    clip and sharing its canvas, and its two architectures both serve it. This
    one takes 12 heterogeneous files of three modalities at their own
    resolutions and rates, in an order that changes the request, and only one
    architecture -- one of its two transformer partitions, in fact -- implements
    it at all.

    OPTIONAL keyframe anchors (C5): `keyframe_images` / `keyframe_frame_indices`
    place additional visual anchors on the SAME generated canvas, after every
    reference block, using `-1` for the resolved last frame exactly as
    /generate/img2vid's do (`plan_keyframe_placements`, resolved after the clip
    length is snapped). References remain content conditioning and anchors
    remain placement conditioning; combining them is measured to bind on this
    partition (`minimax_h3_conditioning_design.md` M-C0a).
    """
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from utils.video_utils import save_video_with_metadata, load_video_frames
    from core.models.minimax_h3 import h3_references as h3refs

    # ---- Architecture gate. Two distinct refusals, because "no MiniMax-H3
    # loaded" and "the wrong MiniMax-H3 partition is loaded" are different
    # problems with different fixes. ----
    if not getattr(pipeline_manager, "is_minimax_h3_model", False):
        raise CustomValidationError(
            "No MiniMax-H3 model loaded",
            detail="Omni-reference conditioning (up to 9 images, 3 videos and 3 audio clips in one "
                   "packed sequence) is a MiniMax-H3 `ref2va` workflow; no other architecture in "
                   "this repo implements it. Load the MiniMax-H3 ref2va transformer before calling "
                   "/generate/ref2vid.",
        )
    _variant = ((pipeline_manager.current_model_info or {}).get("variant") or "").lower()
    if _variant != "ref2va":
        raise CustomValidationError(
            f"The loaded MiniMax-H3 transformer is the {_variant or 'unidentified'} variant, not ref2va",
            detail="MiniMax-H3 ships TWO transformer partitions that share every other component: "
                   "`fl2va` (which serves /generate/txt2vid, /generate/img2vid and "
                   "/generate/outpaint/video) and `ref2va` (this endpoint). Reference conditioning "
                   "is a trained behaviour of the ref2va partition alone, and the two files are "
                   "otherwise indistinguishable -- same config, same size, no distinguishing key -- "
                   "so running this request on the loaded one would silently produce a bad video "
                   "rather than fail. Load "
                   "diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors instead.",
        )

    # ---- The reference files, in upload order. An empty multipart part (a
    # part with no filename) counts as absent, the same normalisation
    # /generate/img2vid applies to `last_frame_image`. ----
    def _present(files) -> List[UploadFile]:
        return [f for f in (files or []) if f is not None and f.filename]

    image_files = _present(reference_images)
    video_files = _present(reference_videos)
    video_audio_files = list(reference_video_audios or [])
    audio_files = _present(reference_audios)
    # C5: the optional keyframe anchors, same positional-pair convention as
    # /generate/img2vid.
    _keyframes = _present(keyframe_images)
    _keyframe_indices = list(keyframe_frame_indices or [])
    if len(_keyframes) != len(_keyframe_indices):
        raise CustomValidationError(
            "keyframe_images and keyframe_frame_indices must be the same length",
            detail=f"Got {len(_keyframes)} keyframe image(s) and {len(_keyframe_indices)} "
                   f"frame index/indices. The two lists are positional: entry n of "
                   f"keyframe_frame_indices is the placement of entry n of keyframe_images.",
        )

    # Counts are validated BEFORE anything is read: these are the model's limits
    # and a request that breaks them must not first pay for 12 file reads.
    total_references = len(image_files) + len(video_files) + len(audio_files)
    for kind, count, limit in (("image", len(image_files), h3refs.MAX_REFERENCE_IMAGES),
                               ("video", len(video_files), h3refs.MAX_REFERENCE_VIDEOS),
                               ("audio", len(audio_files), h3refs.MAX_REFERENCE_AUDIOS)):
        if count > limit:
            raise CustomValidationError(
                f"MiniMax-H3 accepts at most {limit} {kind} reference(s)",
                detail=f"Got {count}. The limit is the released checkpoint's, not SushiUI's.",
            )
    if total_references > h3refs.MAX_REFERENCES:
        raise CustomValidationError(
            f"MiniMax-H3 accepts at most {h3refs.MAX_REFERENCES} references in total",
            detail=f"Got {total_references} ({len(image_files)} image, {len(video_files)} video, "
                   f"{len(audio_files)} audio). The limit is the released checkpoint's, not "
                   f"SushiUI's.",
        )
    if total_references == 0:
        raise CustomValidationError(
            "ref2vid needs at least one reference",
            detail="Send at least one reference_images or reference_videos file, or use "
                   "/generate/txt2vid for a text-only request.",
        )
    if not image_files and not video_files:
        raise CustomValidationError(
            "An audio reference cannot be the only kind sent",
            detail="A standalone soundtrack never reaches the Qwen3-VL conditioner -- it is encoded "
                   "by the audio VAE alone -- so a request built from audio references only "
                   "conditions the vision stream on nothing. Pair it with at least one image or "
                   "video reference.",
        )
    if len(video_audio_files) > len(video_files):
        raise CustomValidationError(
            "More reference_video_audios than reference_videos",
            detail=f"Got {len(video_audio_files)} soundtrack(s) for {len(video_files)} reference "
                   f"video(s). Entry n is the soundtrack of reference video n; send an empty part "
                   f"to skip one.",
        )
    if reference_image_size not in ("max", "match"):
        raise CustomValidationError(
            "Invalid reference_image_size",
            detail=f"Must be 'max' (the released 2048-pixel short edge) or 'match' (scale down to "
                   f"the generation's pixel area), got {reference_image_size!r}.",
        )

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "num_videos_per_prompt": num_videos_per_prompt,
        "max_sequence_length": max_sequence_length,
        "audio_enable": audio_enable,
        "blocks_to_swap": blocks_to_swap,
        "fuse_output_proj": fuse_output_proj,
        "fbcache_enable": fbcache_enable,
        "fbcache_threshold": fbcache_threshold,
        "fbcache_warmup_steps": fbcache_warmup_steps,
        "spectrum_enable": spectrum_enable,
        "spectrum_w": spectrum_w,
        "spectrum_w_decay": spectrum_w_decay,
        "spectrum_delta_cap": spectrum_delta_cap,
        "spectrum_m": spectrum_m,
        "spectrum_lam": spectrum_lam,
        "spectrum_warmup_steps": spectrum_warmup_steps,
        "spectrum_window_size": spectrum_window_size,
        "spectrum_flex_window": spectrum_flex_window,
        "spectrum_tail": spectrum_tail,
        "spectrum_max_cache": spectrum_max_cache,
        "vae_path": vae_path,
        "text_encoder_path": text_encoder_path,
        "unet_quantization": unet_quantization,
        "quantized_gemm_mode": _normalize_media_qgm(quantized_gemm_mode),
        "attention_type": _validated_attention_type(
            attention_type, REF2VID_DEFAULTS["attention_type"]),
        "reference_image_size": reference_image_size,
        # The uploaded FILENAMES, in packed order -- what the gallery row can
        # carry. The bytes never reach the database. Passed through
        # recover_upload_filename: see that module for why a latin-1/UTF-8
        # multipart mis-decode is losslessly reversible here.
        "reference_images": [recover_upload_filename(f.filename) for f in image_files] or None,
        "reference_videos": [recover_upload_filename(f.filename) for f in video_files] or None,
        "reference_audios": [recover_upload_filename(f.filename) for f in audio_files] or None,
        # C5's keyframe fields, recorded the same way img2vid records them: the
        # uploaded FILENAMES and the requested indices. The RESOLVED placements
        # are added after num_frames has been snapped, below.
        "keyframe_images": [recover_upload_filename(f.filename) for f in _keyframes] or None,
        "keyframe_frame_indices": list(_keyframe_indices) or None,
    }

    # Generation-time LoRA (JSON string). See Txt2VidRequest.loras.
    import json
    params["loras"] = json.loads(loras) if loras else []

    _vid_arch = (pipeline_manager.current_model_info or {}).get("type")
    _omitted = {key for key, value in (
        ("width", width), ("height", height), ("num_frames", num_frames),
        ("frame_rate", frame_rate), ("num_inference_steps", num_inference_steps),
        ("guidance_scale", guidance_scale),
    ) if value is None}
    _provided = set(params) - _omitted
    _vid_defaults = resolve_video_defaults(params, _provided, _vid_arch, REF2VID_DEFAULTS)

    # ---- Read and decode the references, in PACKED order: images, then videos
    # (each with its own soundtrack), then standalone audio. Decoding happens
    # before the GPU slot is taken, so a malformed upload is a 400 that reserved
    # nothing. The generated frame count is already resolved above, and it is
    # what a reference video and every soundtrack are truncated to. ----
    _decode_frames_cap = int(params["num_frames"])
    references: List[h3refs.MiniMaxH3Reference] = []
    try:
        for upload in image_files:
            data = await upload.read()
            references.append(h3refs.MiniMaxH3Reference(
                kind="image", image=Image.open(io.BytesIO(data)).convert("RGB"),
                label=upload.filename))
        for index, upload in enumerate(video_files):
            data = await upload.read()
            # A reference video is resampled to 24 fps and truncated to the
            # generated length, so decoding more than the generated length's
            # worth of source frames at the source rate can never be needed
            # beyond the resample's own headroom (a slow-motion 8 fps source
            # needs a third as many). The cap is the generated length scaled by
            # the largest plausible ratio, which `load_video_frames` then trims.
            frames, source_fps = load_video_frames(data, max_frames=max(1, _decode_frames_cap * 4))
            soundtrack = None
            sample_rate = None
            audio_upload = video_audio_files[index] if index < len(video_audio_files) else None
            if audio_upload is not None and audio_upload.filename:
                soundtrack, sample_rate = h3refs.decode_audio_bytes(await audio_upload.read())
            references.append(h3refs.MiniMaxH3Reference(
                kind="video", frames=frames, fps=source_fps, audio=soundtrack,
                sample_rate=sample_rate, label=upload.filename))
        for upload in audio_files:
            waveform, sample_rate = h3refs.decode_audio_bytes(await upload.read())
            references.append(h3refs.MiniMaxH3Reference(
                kind="audio", audio=waveform, sample_rate=sample_rate, label=upload.filename))
    except CustomValidationError:
        raise
    except Exception as e:
        raise CustomValidationError(
            "Failed to read an uploaded reference",
            detail=str(e),
        )

    # C5: the keyframe anchors, read the same way /generate/img2vid reads them
    # -- before the generation context, so a malformed upload is a 400 that
    # reserved nothing.
    extra_keyframe_images = []
    for _position, _upload in enumerate(_keyframes):
        try:
            _data = await _upload.read()
            extra_keyframe_images.append(Image.open(io.BytesIO(_data)).convert("RGB"))
        except Exception as e:
            raise CustomValidationError(
                f"Failed to read keyframe_images[{_position}]",
                detail=str(e),
            )

    _gen_id = start_generation("ref2vid")
    try:
        pipeline_manager.reset_cancel_flag()

        validate_video_geometry(params, _vid_arch)
        validate_video_steps(params, _vid_arch)

        from api.arch_capabilities import check_arch_capabilities, arch_supports_feature
        from api.generation_status import add_warning
        from api.generation_overrides import plan_overrides, apply_overrides
        from api.generation_utils import plan_keyframe_placements, MINIMAX_H3_DOCUMENTED_ANCHOR_SCOPE
        _override_plan = plan_overrides(pipeline_manager, params.get("vae_path"), params.get("text_encoder_path"))
        apply_overrides(pipeline_manager, _override_plan)
        check_arch_capabilities(params, _vid_arch, defaults=_vid_defaults)

        # ---- Keyframe placement, resolved AFTER the snap -- same reasoning as
        # /generate/img2vid's identical block. ----
        keyframe_plan = None
        if _keyframes and arch_supports_feature(_vid_arch, "keyframe_placement"):
            placement_requests = [(f"keyframe_images[{position}]", index)
                                  for position, index in enumerate(_keyframe_indices)]
            plan = plan_keyframe_placements(placement_requests, params.get("num_frames") or 0)
            sources = {f"keyframe_images[{position}]": pil
                      for position, pil in enumerate(extra_keyframe_images)}
            keyframe_plan = [(entry["anchor"], sources[entry["source"]])
                             for entry in plan["anchors"]]
            params["keyframe_resolved_frames"] = [entry["frame"] for entry in plan["anchors"]]
            # ONE warning for the whole request. Unlike /generate/img2vid, EVERY
            # anchor here is beyond the model card on its own terms -- MiniMax
            # never documents anchors combined with reference conditioning at
            # all, not even a first/last placement -- so this fires whenever any
            # anchor is sent, and folds in the placement-specific reasons too.
            _undocumented = list(plan["undocumented"])
            _undocumented.append("keyframe anchors combined with reference conditioning")
            add_warning(
                "This request conditions MiniMax-H3 outside the documented shape ("
                + "; ".join(_undocumented)
                + f"). {MINIMAX_H3_DOCUMENTED_ANCHOR_SCOPE}; combining anchors with reference "
                  "conditioning is measured to bind on this partition but is not itself part of "
                  "MiniMax's model card.",
                code="minimax_h3_undocumented_conditioning",
            )

        print(f"ref2vid generation params: {sanitize_params_for_logging(params)}")

        def progress_callback(step, total_steps, sub_progress=None):
            from api.generation_status import update_progress
            total = max(total_steps, 1)
            # A mid-step tick names the step still RUNNING; `step` itself keeps
            # counting COMPLETED steps, and the integer-only generation status is
            # left at the last boundary.
            shown = step + 1 if sub_progress is not None else step
            manager.send_progress_sync(
                step, total, f"Generating video: step {shown}/{total}", sub_progress=sub_progress)
            if sub_progress is None:
                update_progress(step, total)

        from core.gpu_coordinator import gpu_coordinator
        from core.inference.generation_timing import generation_timer
        loop = asyncio.get_event_loop()
        generation_timer.reset()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=40, timeout=120.0):
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(params.get("quantized_gemm_mode"))
            frames, audio, audio_sample_rate, actual_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: pipeline_manager.generate_ref2vid(
                    params, references, progress_callback=progress_callback,
                    keyframes=keyframe_plan)
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        apply_generation_timings(params, time.perf_counter() - _gen_start)
        _record_media_gemm_outcome(params, fp8_gemm, _vid_arch)
        record_attention_backend(params, _gen_id)
        record_model_variant(params, pipeline_manager)

        vae_name, vae_hash = extract_vae_info(pipeline_manager)
        if vae_name:
            params["vae_name"] = vae_name
        if vae_hash:
            params["vae_hash"] = vae_hash

        params["seed"] = actual_seed

        filename, _preview_filename = save_video_with_metadata(
            frames,
            audio,
            audio_sample_rate,
            params,
            "ref2vid",
            model_info=pipeline_manager.current_model_info,
        )

        base_name = os.path.splitext(filename)[0]
        poster_path = os.path.join(settings.outputs_dir, f"{base_name}.png")
        if os.path.exists(poster_path):
            create_thumbnail(poster_path)

        # Hash the saved MASTER file's bytes (see calculate_bytes_hash's docstring).
        _media_hash = await _hash_saved_media(os.path.join(settings.outputs_dir, filename))

        num_frames_out = int(frames.shape[0])
        fps_out = float(params.get("frame_rate", 24.0))
        params_for_db = {k: v for k, v in params.items() if not k.startswith("_")}
        params_for_db["num_frames"] = num_frames_out
        params_for_db["fps"] = fps_out
        params_for_db["duration"] = (num_frames_out / fps_out) if fps_out else 0.0
        params_for_db["audio_enable"] = bool(params.get("audio_enable", True) and audio is not None)
        params_for_db["is_video"] = True
        params_for_db["hash_kind"] = "file_bytes"
        # See /generate/img2vid: the gallery's cfg_scale / steps COLUMNS are
        # filled from the image routes' key names, which a video request does
        # not carry.
        params_for_db["cfg_scale"] = float(params.get("guidance_scale", 1.0))
        params_for_db["steps"] = int(params.get("num_inference_steps", 0))
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        model_name, model_hash = extract_model_info(pipeline_manager)

        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="ref2vid",
            image_hash=_media_hash,
            lora_names=extract_lora_names(params.get("loras") or []),
            model_name=model_name,
            model_hash=model_hash,
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except (GenerationError, CustomValidationError, NotFoundError) as e:
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except ValueError as e:
        # The reference-normalisation rules (rates, resolutions, the 22-frame
        # floor, the packed-order limits) raise ValueError from the ops layer.
        # They are client errors about the uploaded media, not generation
        # failures, so they must not be re-wrapped as a 500.
        fail_generation(str(e), generation_id=_gen_id)
        raise CustomValidationError("Invalid MiniMax-H3 reference", detail=str(e))
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Reference-to-video generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )


@router.post("/generate/outpaint/video")
async def generate_outpaint_video(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(OUTPAINT_VIDEO_DEFAULTS["negative_prompt"]),
    # The six keys a per-arch overlay can change are `Form(None)` SENTINELS,
    # exactly as in /generate/img2vid: a multipart route has no
    # `model_fields_set`, so an omitted field is only distinguishable from a
    # sent one by the sentinel, and whatever comes back None is filled from the
    # resolved per-arch map below. `total_frames` is in the set because its base
    # 121 is on LTX-2.3's 8k+1 grid and is not a length MiniMax-H3 can produce.
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    total_frames: Optional[int] = Form(None),
    frame_rate: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    seed: int = Form(OUTPAINT_VIDEO_DEFAULTS["seed"]),
    num_videos_per_prompt: int = Form(OUTPAINT_VIDEO_DEFAULTS["num_videos_per_prompt"]),
    max_sequence_length: int = Form(OUTPAINT_VIDEO_DEFAULTS["max_sequence_length"]),
    audio_enable: bool = Form(OUTPAINT_VIDEO_DEFAULTS["audio_enable"]),
    # Placement (new for video outpaint). All in PIXEL frames of the OUTPUT
    # timeline except input_trim_*, which trim the UPLOADED clip itself.
    input_offset_frames: int = Form(OUTPAINT_VIDEO_DEFAULTS["input_offset_frames"]),
    input_trim_start_frames: int = Form(OUTPAINT_VIDEO_DEFAULTS["input_trim_start_frames"]),
    input_trim_end_frames: int = Form(OUTPAINT_VIDEO_DEFAULTS["input_trim_end_frames"]),
    # SENTINEL, for the same reason the six geometry fields above are: the base
    # "regenerate" is an LTX-2.3 semantic (its generated track spans the whole
    # timeline), while MiniMax-H3 generates audio only for the frames it
    # generates and so overlays "preserve_input" -- an omitted field has to be
    # able to reach that overlay instead of freezing the base value here.
    outpaint_video_audio_mode: Optional[str] = Form(None),
    # See Img2VidRequest.attention_type: honored by MiniMax-H3, ignored by
    # LTX-2.3. `OUTPAINT_VIDEO_DEFAULTS` has carried this key since it was
    # derived from VIDEO_GEN_DEFAULTS, but the route had no field for it while
    # this endpoint served LTX-2.3 alone -- which meant the ONE architecture
    # that honors it could not be given it here.
    attention_type: str = Form(OUTPAINT_VIDEO_DEFAULTS["attention_type"]),
    blocks_to_swap: int = Form(OUTPAINT_VIDEO_DEFAULTS["blocks_to_swap"]),
    fuse_output_proj: bool = Form(OUTPAINT_VIDEO_DEFAULTS["fuse_output_proj"]),
    fbcache_enable: bool = Form(OUTPAINT_VIDEO_DEFAULTS["fbcache_enable"]),
    fbcache_threshold: float = Form(OUTPAINT_VIDEO_DEFAULTS["fbcache_threshold"]),
    fbcache_warmup_steps: int = Form(OUTPAINT_VIDEO_DEFAULTS["fbcache_warmup_steps"]),
    spectrum_enable: bool = Form(OUTPAINT_VIDEO_DEFAULTS["spectrum_enable"]),
    spectrum_w: float = Form(OUTPAINT_VIDEO_DEFAULTS["spectrum_w"]),
    spectrum_w_decay: float = Form(OUTPAINT_VIDEO_DEFAULTS["spectrum_w_decay"]),
    spectrum_delta_cap: float = Form(OUTPAINT_VIDEO_DEFAULTS["spectrum_delta_cap"]),
    spectrum_m: int = Form(OUTPAINT_VIDEO_DEFAULTS["spectrum_m"]),
    spectrum_lam: float = Form(OUTPAINT_VIDEO_DEFAULTS["spectrum_lam"]),
    spectrum_warmup_steps: int = Form(OUTPAINT_VIDEO_DEFAULTS["spectrum_warmup_steps"]),
    spectrum_window_size: int = Form(OUTPAINT_VIDEO_DEFAULTS["spectrum_window_size"]),
    spectrum_flex_window: float = Form(OUTPAINT_VIDEO_DEFAULTS["spectrum_flex_window"]),
    spectrum_tail: float = Form(OUTPAINT_VIDEO_DEFAULTS["spectrum_tail"]),
    spectrum_max_cache: int = Form(OUTPAINT_VIDEO_DEFAULTS["spectrum_max_cache"]),
    vae_path: Optional[str] = Form(OUTPAINT_VIDEO_DEFAULTS["vae_path"]),
    text_encoder_path: Optional[str] = Form(OUTPAINT_VIDEO_DEFAULTS["text_encoder_path"]),
    unet_quantization: Optional[str] = Form(OUTPAINT_VIDEO_DEFAULTS["unet_quantization"]),
    quantized_gemm_mode: Optional[str] = Form(OUTPAINT_VIDEO_DEFAULTS["quantized_gemm_mode"]),
    video_lossless: bool = Form(OUTPAINT_VIDEO_DEFAULTS["video_lossless"]),
    # MiniMax-H3 ref2va only (extend_forward, source clip auto-referenced --
    # see the partition/placement gate below). No reference_videos/
    # reference_audios field: the source clip is always the sole video
    # reference on this endpoint.
    reference_image_size: str = Form(OUTPAINT_VIDEO_DEFAULTS["reference_image_size"]),
    reference_images: List[UploadFile] = File([]),
    video: UploadFile = File(...),
    # OPTIONAL second clip, preserved at the END of the timeline: the BRIDGE
    # placement. Only an architecture whose TemporalSpec lists `bridge`
    # (MiniMax-H3, which conditions on boundary frames and can therefore anchor
    # both ends of a generated gap) accepts it; LTX-2.3 refuses it with that
    # reason rather than ignoring it, because ignoring a second uploaded clip
    # would silently produce a one-clip result.
    bridge_video: Optional[UploadFile] = File(None),
    # Generation-time LoRA. See Txt2VidRequest.loras.
    loras: str = Form("[]"),
    db: Session = Depends(get_gallery_db)
):
    """Video temporal outpaint (LTX-2.3 or MiniMax-H3): place a (trimmed) input
    clip inside a LONGER output timeline and generate the frames before/after.
    Multipart form: an uploaded `video` clip, optionally a second
    `bridge_video`, plus the parameters below.

    Non-÷32 clips are preprocessed ONCE (center-crop + resize to
    `width`x`height`, the same resolution the whole output timeline shares)
    and the PREPROCESSED frames become the exact-preserved content (mirrors
    `/generate/outpaint`'s RESIZE convention).

    The two architectures place the clip by different mechanisms and this
    endpoint does not paper over the difference:

    * **LTX-2.3** generates the whole timeline with the clip pinned at an
      arbitrary latent index and pastes the input back frame-exact.
      `input_offset_frames` is snapped server-side to the nearest valid latent
      frame index (a warning is added if snapped), and `total_frames` is what
      must satisfy `(n - 1) % 8 == 0`.
    * **MiniMax-H3** conditions on the FIRST and/or LAST frame of the span it
      generates, so only the missing span is generated and the result is
      concatenated with the untouched input. `input_offset_frames` therefore
      SELECTS the placement -- 0 (extend forward), `total_frames - placed`
      (extend backward), or 0 with a `bridge_video` (bridge) -- and anything
      else is a 400 stating that reason. The `17n + 5` rule binds the
      GENERATED span, which is solved for and rounded up, so `total_frames` is
      a request and the effective output length comes back in `warnings[]`.

    **MiniMax-H3 `ref2va` + `reference_images` (extend_forward only):** with
    the ref2va transformer loaded and an extend-forward placement, the
    preserved clip's own trailing frames become an automatic video reference
    (soundtrack excluded) and `reference_images` add optional image
    references on top -- both compose through the same anchor + placement
    machinery above; nothing about the preservation/concatenation contract
    changes. `extend_backward` and `bridge` are refused on ref2va (unmeasured
    shape), as is a generated span shorter than the reference video floor (22
    frames). See the partition/placement gate below.

    Produces an mp4 (H.264) and a gallery row, unless `video_lossless` is
    true, in which case it produces an FFV1-in-mkv master plus a browser-
    playable H.264 mp4 proxy (`preview_filename` on the row). Requires a
    video model to be loaded.
    """
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from utils.video_utils import save_video_with_metadata, load_video_frames, extract_audio_stream, probe_upload_clip
    from api.generation_utils import plan_video_outpaint_placement
    from api.param_defaults import outpaint_video_defaults_for_arch, MAX_VIDEO_UPLOAD_DECODE_BYTES
    from core.models.components.wiring import temporal_spec_for_arch

    if not (getattr(pipeline_manager, "is_ltx2_model", False)
            or getattr(pipeline_manager, "is_minimax_h3_model", False)):
        raise CustomValidationError(
            "No video model loaded",
            detail="Load an LTX-2.3 or MiniMax-H3 video model before calling "
                   "/generate/outpaint/video.",
        )
    # ---- Partition/placement gate (the decision table in
    # minimax_h3_outpaint_refs_design.md §3). Two parts: what can be told
    # WITHOUT decoding the upload (here), and what needs the placement plan
    # (after the clip is decoded, below). No row of this table ever reroutes
    # to /generate/ref2vid -- a refusal here is a refusal.
    _ref_image_files = [f for f in (reference_images or []) if f is not None and f.filename]
    if _ref_image_files and not getattr(pipeline_manager, "is_minimax_h3_model", False):
        raise CustomValidationError(
            "reference_images on outpaint is a MiniMax-H3 ref2va capability",
            detail="No other architecture in this repo reads reference rows on this endpoint. "
                   "Load a MiniMax-H3 ref2va checkpoint, or omit reference_images.",
        )
    _h3_variant = ""
    if getattr(pipeline_manager, "is_minimax_h3_model", False):
        _h3_variant = ((pipeline_manager.current_model_info or {}).get("variant") or "").lower()
        if _h3_variant == "fl2va" and _ref_image_files:
            raise CustomValidationError(
                "reference_images requires the MiniMax-H3 ref2va transformer, not fl2va",
                detail="fl2va was never trained to read reference rows (mirror of "
                       "/generate/ref2vid's own partition gate). Load "
                       "diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors, or omit "
                       "reference_images to keep using fl2va's exact-preserving extend.",
            )
        from core.models.minimax_h3.h3_references import MAX_REFERENCE_IMAGES as _max_ref_images
        if len(_ref_image_files) > _max_ref_images:
            raise CustomValidationError(
                f"MiniMax-H3 accepts at most {_max_ref_images} image reference(s)",
                detail=f"Got {len(_ref_image_files)}. The limit is the released checkpoint's, not "
                       f"SushiUI's.",
            )
        # ref2va + extend_backward/bridge and ref2va + a too-short generated
        # span are refused below, once the placement plan is known (it needs
        # the decoded clip length). `ref2va` + `extend_forward` with no
        # reference_images at all is still allowed -- the source clip is
        # ALWAYS the sole video reference on this endpoint, so that row is
        # the A-V8 no-image-reference arm, not a no-op.
    if outpaint_video_audio_mode is not None and outpaint_video_audio_mode not in (
            "regenerate", "preserve_input"):
        raise CustomValidationError(
            "Invalid outpaint_video_audio_mode",
            detail=f"Must be 'regenerate' or 'preserve_input', got {outpaint_video_audio_mode!r}.",
        )

    # Per-architecture defaults FIRST: the decode bound and every geometry check
    # below read `total_frames`, which is one of the overlaid keys. Resolving
    # from `outpaint_video_defaults_for_arch` (base -> video overlay -> outpaint
    # overlay) is what makes an omitted `total_frames` resolve to MiniMax-H3's
    # 248 instead of being validated against LTX-2.3's 121.
    _vid_arch = (pipeline_manager.current_model_info or {}).get("type")
    _resolved = outpaint_video_defaults_for_arch(_vid_arch)
    if width is None:
        width = int(_resolved["width"])
    if height is None:
        height = int(_resolved["height"])
    if total_frames is None:
        total_frames = int(_resolved["total_frames"])
    if frame_rate is None:
        frame_rate = float(_resolved["frame_rate"])
    if num_inference_steps is None:
        num_inference_steps = int(_resolved["num_inference_steps"])
    if guidance_scale is None:
        guidance_scale = float(_resolved["guidance_scale"])
    if outpaint_video_audio_mode is None:
        outpaint_video_audio_mode = str(_resolved["outpaint_video_audio_mode"])

    _spec = temporal_spec_for_arch(_vid_arch)
    _placements = tuple(_spec.outpaint_placements) if _spec is not None else ("free",)
    _align = _spec.pixel_align if _spec is not None else 32
    if width < _align or height < _align or width % _align != 0 or height % _align != 0:
        raise CustomValidationError(
            f"width and height must both be >= {_align} and divisible by {_align}",
            detail=f"Got width={width}, height={height}. Round each to a multiple of {_align}, "
                   f"minimum {_align}.",
        )
    if _spec is not None and _spec.max_pixel_hw is not None:
        _short_cap, _long_cap = min(_spec.max_pixel_hw), max(_spec.max_pixel_hw)
        if min(width, height) > _short_cap or max(width, height) > _long_cap:
            raise CustomValidationError(
                f"the canvas exceeds this model's {_short_cap}x{_long_cap} envelope",
                detail=f"Got {width}x{height}. The loaded model generates with a short edge of at "
                       f"most {_short_cap} and a long edge of at most {_long_cap}, in either "
                       f"orientation.",
            )
    if "free" in _placements:
        # LTX-2.3's shipped contract, unchanged: the OUTPUT timeline is what the
        # model generates end to end, so it is what must sit on the 8k+1 grid.
        if total_frames < 9 or total_frames % 8 != 1:
            raise CustomValidationError(
                "total_frames must be >= 9 and satisfy (total_frames - 1) % 8 == 0",
                detail=f"Got total_frames={total_frames}. Use values like 9, 17, ..., 121 (8k + 1, k >= 1).",
            )
    else:
        # A boundary-conditioned arch generates only the MISSING span, so the
        # grid rule binds that span (resolved by plan_video_outpaint_placement
        # once the clip lengths are known) and not this number.
        if total_frames < 2:
            raise CustomValidationError(
                "total_frames must be at least 2",
                detail=f"Got total_frames={total_frames}.",
            )
    if frame_rate <= 0:
        raise CustomValidationError(
            "frame_rate must be > 0",
            detail=f"Got frame_rate={frame_rate}.",
        )
    # Spec-driven step-count floor (TemporalSpec.min_inference_steps). `params`
    # is not built until after the upload is decoded, so the value is passed on
    # its own.
    validate_video_steps({"num_inference_steps": num_inference_steps}, _vid_arch)
    if bridge_video is not None and "bridge" not in _placements:
        raise CustomValidationError(
            "this model has no bridge placement",
            detail="bridge_video adds a SECOND preserved clip at the end of the timeline, which is "
                   "a placement only an architecture that conditions on boundary frames needs. The "
                   "loaded model places one clip at an arbitrary offset instead; upload a single "
                   "`video` and set input_offset_frames.",
        )

    # Decode the uploaded clip + (best-effort) extract its original audio
    # track BEFORE acquiring the GPU slot, so a malformed upload surfaces as
    # a 400/500 without reserving GPU resources.
    #
    # `max_frames` USED TO bound the "free" placement's decode to
    # `input_trim_start_frames + total_frames`, on the reasoning that
    # total_frames IS the output timeline there and nothing past it can ever
    # be placed. That reasoning only holds when the upload is SHORTER than
    # (or equal to) the timeline -- nothing enforces that, and when it is
    # not, capping the DECODE at that bound hid the excess from the one
    # thing that would otherwise report it: `_generate_vidoutpaint_ltx2`
    # (ltx2.py) already warns (`outpaint_video_tail_frames_dropped`)
    # whenever the trimmed clip has more frames than the placement has room
    # for, but it only ever sees the DECODED frame count, so a real excess
    # BEYOND the old cap was truncated before that warning could ever fire
    # -- silently dropping uploaded footage with no error and no warning,
    # exactly the class of bug 94b952d5 fixed for boundary placements,
    # reachable here too under the "free" placement.
    #
    # There is therefore no placement-specific decode bound left: every
    # placement now decodes the whole (trimmed) clip and leaves the
    # trim/fit/drop arithmetic to `plan_video_outpaint_placement` and the
    # backend's own tail-drop warning, both of which already handle "clip
    # longer than what the request can place" correctly once they are
    # actually shown the true frame count -- a "free" arch reports the drop
    # via `outpaint_video_tail_frames_dropped` (same as the 8k+1 grid
    # rounding it already covered) rather than refusing outright, since the
    # placement is well-defined either way (the clip is simply used from
    # `pixel_start` for as many frames as fit). The RAM cost of decoding an
    # arbitrarily long upload is bounded below by `_refuse_if_decode_too_large`
    # (a pre-decode, ffprobe-only check), not by this variable. The bridge
    # clip (below) is a second preserved clip on a boundary-only placement
    # (a "free" arch refuses `bridge_video` above), so it gets the identical
    # `None` treatment for the identical reason.
    _decode_max_frames = None
    # A boundary placement's "no smaller bound exists" above is true of
    # VALIDITY, not of RAM: a raw decoded uint8 RGB clip costs
    # `width * height * 3` bytes PER FRAME, so `max_frames=None` on a
    # multi-minute high-res upload can OOM the whole process rather than just
    # fail the request. ffprobe the clip(s) first (no per-frame decode) and
    # refuse an upload whose decode would exceed a fixed RAM budget before
    # `load_video_frames` is ever called; a clip that passes this is bounded
    # regardless of which `_decode_max_frames` branch above applies to it.
    def _refuse_if_decode_too_large(data: bytes, *, label: str) -> None:
        probed = probe_upload_clip(data)
        if not probed:
            # Unprobeable here; load_video_frames probes again immediately
            # below and raises a descriptive error for the same clip, so
            # nothing is silently skipped.
            return
        w, h = int(probed.get("width") or 0), int(probed.get("height") or 0)
        n = int(probed.get("num_frames") or 0)
        if w <= 0 or h <= 0 or n <= 0:
            return
        per_frame_bytes = w * h * 3
        frame_budget = MAX_VIDEO_UPLOAD_DECODE_BYTES // max(1, per_frame_bytes)
        if n > frame_budget:
            raise CustomValidationError(
                f"the {label} is too long/high-resolution to decode",
                detail=f"Uploaded {label} has {n} frames at {w}x{h}, which would need "
                       f"{n * per_frame_bytes / (1024 ** 3):.1f} GiB of raw RGB frame data to "
                       f"decode; this endpoint's per-clip decode budget is "
                       f"{MAX_VIDEO_UPLOAD_DECODE_BYTES / (1024 ** 3):.0f} GiB, i.e. at most "
                       f"{frame_budget} frames at this resolution. Trim the clip or lower its "
                       f"resolution before uploading.",
            )

    try:
        video_data = await video.read()
        _refuse_if_decode_too_large(video_data, label="input clip")
        video_frames, source_fps = load_video_frames(
            video_data, max_frames=_decode_max_frames, trim_end_frames=input_trim_end_frames
        )
    except CustomValidationError:
        raise
    except Exception as e:
        raise CustomValidationError(
            "Failed to decode the uploaded video clip",
            detail=str(e),
        )

    bridge_frames = None
    bridge_source_fps = None
    bridge_audio = None
    if bridge_video is not None:
        try:
            bridge_data = await bridge_video.read()
            _refuse_if_decode_too_large(bridge_data, label="bridge clip")
            bridge_frames, bridge_source_fps = load_video_frames(
                bridge_data, max_frames=_decode_max_frames
            )
        except CustomValidationError:
            raise
        except Exception as e:
            raise CustomValidationError(
                "Failed to decode the uploaded bridge video clip",
                detail=str(e),
            )

    input_audio = None
    if audio_enable and outpaint_video_audio_mode == "preserve_input":
        input_audio = extract_audio_stream(video_data)
        if bridge_video is not None:
            bridge_audio = extract_audio_stream(bridge_data)

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "num_frames": total_frames,  # metadata parity with txt2vid/img2vid's "num_frames" key
        "frame_rate": frame_rate,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "num_videos_per_prompt": num_videos_per_prompt,
        "max_sequence_length": max_sequence_length,
        "audio_enable": audio_enable,
        "input_offset_frames": input_offset_frames,
        "input_trim_start_frames": input_trim_start_frames,
        "input_trim_end_frames": input_trim_end_frames,
        "outpaint_video_audio_mode": outpaint_video_audio_mode,
        # Validated against the attention registry's vocabulary, same
        # 400-before-start_generation contract as every other route that takes
        # it (`attention_type_validation_test` is the forcing function).
        "attention_type": _validated_attention_type(
            attention_type, OUTPAINT_VIDEO_DEFAULTS["attention_type"]),
        "blocks_to_swap": blocks_to_swap,
        "fuse_output_proj": fuse_output_proj,
        "fbcache_enable": fbcache_enable,
        "fbcache_threshold": fbcache_threshold,
        "fbcache_warmup_steps": fbcache_warmup_steps,
        "spectrum_enable": spectrum_enable,
        "spectrum_w": spectrum_w,
        "spectrum_w_decay": spectrum_w_decay,
        "spectrum_delta_cap": spectrum_delta_cap,
        "spectrum_m": spectrum_m,
        "spectrum_lam": spectrum_lam,
        "spectrum_warmup_steps": spectrum_warmup_steps,
        "spectrum_window_size": spectrum_window_size,
        "spectrum_flex_window": spectrum_flex_window,
        "spectrum_tail": spectrum_tail,
        "spectrum_max_cache": spectrum_max_cache,
        "vae_path": vae_path,
        "text_encoder_path": text_encoder_path,
        "unet_quantization": unet_quantization,
        # Normalized here (before start_generation) so junk is a 400, not a 500.
        "quantized_gemm_mode": _normalize_media_qgm(quantized_gemm_mode),
        "video_lossless": video_lossless,
        # The uploaded FILENAME (never the bytes), so the gallery row records
        # that this was a bridge and which clip closed it.
        "bridge_video": recover_upload_filename(getattr(bridge_video, "filename", None)) if bridge_video is not None else None,
        # MiniMax-H3 ref2va only (extend_forward). Filenames only, same
        # convention as /generate/ref2vid's reference_images.
        "reference_image_size": reference_image_size,
        "reference_images": [recover_upload_filename(f.filename) for f in _ref_image_files] or None,
    }

    # Generation-time LoRA (JSON string). See Txt2VidRequest.loras.
    import json
    params["loras"] = json.loads(loras) if loras else []

    # Reject a clip that cannot fit at all (trim leaves nothing) -- cheap
    # up-front check; the backend re-validates against the SNAPPED offset
    # (which may differ) and the resolution-adjusted frame count.
    trimmed_len = video_frames.shape[0] - max(0, input_trim_start_frames) - max(0, input_trim_end_frames)
    if trimmed_len < 1:
        raise CustomValidationError(
            "Video outpaint input trim leaves no frames",
            detail=f"Uploaded clip has {video_frames.shape[0]} frames; "
                   f"input_trim_start_frames={input_trim_start_frames}, "
                   f"input_trim_end_frames={input_trim_end_frames}.",
        )

    # Placement, against the loaded arch's own conditioning rule. Called HERE
    # (pure, no warnings, no GPU) so a placement this architecture cannot anchor
    # is a fast 400 rather than a failure after a 30-second text encode; the
    # backend calls the same function again and is what warns, so the arithmetic
    # is never duplicated. A "free" arch resolves trivially and keeps today's
    # behaviour.
    _placement_plan = plan_video_outpaint_placement(
        params, _vid_arch,
        head_frames=int(trimmed_len),
        tail_frames=int(bridge_frames.shape[0]) if bridge_frames is not None else None,
    )

    # ---- ref2va placement gate (needs the plan). Shared with the backend's
    # defensive re-check (`resolve_minimax_h3_outpaint_reference_gate`), the
    # decision table of minimax_h3_outpaint_refs_design.md §3: fl2va+refs was
    # already refused above (no placement/decode needed for that row);
    # ref2va serves only extend_forward, whether or not reference_images was
    # sent (the source clip is always auto-referenced there).
    if _h3_variant:
        from api.generation_utils import resolve_minimax_h3_outpaint_reference_gate
        resolve_minimax_h3_outpaint_reference_gate(
            _h3_variant, has_reference_images=bool(_ref_image_files),
            placement=_placement_plan["placement"],
            generated_frames=_placement_plan.get("generated_frames"),
        )

    # ---- Read reference_images, in upload order (packed order). Before the
    # GPU slot, same reasoning as /generate/ref2vid's reference reads: a
    # malformed upload is a 400 that reserved nothing.
    reference_image_pils: List[Image.Image] = []
    try:
        for upload in _ref_image_files:
            data = await upload.read()
            reference_image_pils.append(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception as e:
        raise CustomValidationError(
            "Failed to read an uploaded reference image",
            detail=str(e),
        )

    _gen_id = start_generation("outpaint_vid")
    try:
        pipeline_manager.reset_cancel_flag()

        # Inside the try because a forced frame rate emits a `warnings[]` entry,
        # which needs the generation context started above. `frame_key=None`:
        # the length that has to be on the arch's grid is the GENERATED span's,
        # not this endpoint's OUTPUT timeline (the preserved frames are pasted,
        # never sampled), and the span is resolved by the placement planner.
        # Spatial validation already ran above as a pre-decode 400.
        validate_video_geometry(params, _vid_arch, frame_key=None)
        validate_video_steps(params, _vid_arch)

        from api.arch_capabilities import check_arch_capabilities
        from api.generation_overrides import plan_overrides, apply_overrides
        _override_plan = plan_overrides(pipeline_manager, params.get("vae_path"), params.get("text_encoder_path"))
        apply_overrides(pipeline_manager, _override_plan)
        # See the same call in /generate/img2vid: judged against the VIDEO
        # defaults (here the outpaint ones, INCLUDING the outpaint-only
        # overlay), or every request looks like it set every video-only key by
        # hand -- and on a guidance-distilled arch that means a spurious
        # ignored-guidance warning on every single request.
        check_arch_capabilities(params, _vid_arch,
                                defaults=outpaint_video_defaults_for_arch(_vid_arch))

        print(f"outpaint_vid generation params: {sanitize_params_for_logging(params)}")

        def progress_callback(step, total_steps, sub_progress=None):
            from api.generation_status import update_progress
            total = max(total_steps, 1)
            # A mid-step tick names the step still RUNNING; `step` itself keeps
            # counting COMPLETED steps, and the integer-only generation status is
            # left at the last boundary.
            shown = step + 1 if sub_progress is not None else step
            manager.send_progress_sync(
                step, total, f"Generating video: step {shown}/{total}", sub_progress=sub_progress)
            if sub_progress is None:
                update_progress(step, total)

        from core.gpu_coordinator import gpu_coordinator
        from core.inference.generation_timing import generation_timer
        loop = asyncio.get_event_loop()
        # See the note on the same call in /generate/img2vid: the phase timer
        # accumulates, and a video backend records phases now.
        generation_timer.reset()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=40, timeout=120.0):
            # GEMM flags are process-wide; keep selection and probing in this slot.
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(params.get("quantized_gemm_mode"))
            frames, audio, audio_sample_rate, actual_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: pipeline_manager.generate_vid_outpaint(
                    params, video_frames, source_fps, input_audio,
                    progress_callback=progress_callback,
                    bridge_frames=bridge_frames, bridge_fps=bridge_source_fps,
                    bridge_audio=bridge_audio,
                    reference_images=reference_image_pils,
                )
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        apply_generation_timings(params, time.perf_counter() - _gen_start)
        _record_media_gemm_outcome(params, fp8_gemm, _vid_arch)
        # See the same call in /generate/img2vid: the backend the conduit really
        # used, not the one that was asked for. Nothing is written on an
        # architecture that does not route attention through the conduit.
        record_attention_backend(params, _gen_id)
        record_model_variant(params, pipeline_manager)

        # See the note on the same call in /generate/txt2vid.
        vae_name, vae_hash = extract_vae_info(pipeline_manager)
        if vae_name:
            params["vae_name"] = vae_name
        if vae_hash:
            params["vae_hash"] = vae_hash

        params["seed"] = actual_seed

        # Hash the uploaded clip's raw bytes (see calculate_bytes_hash's
        # docstring: video source/target hashing is defined over FILE BYTES,
        # not decoded pixels -- a frame-0 pixel hash of `video_data` could
        # never match any row's byte-hashed image_hash, since ffmpeg's decode
        # is lossy and non-deterministic across encodes). This SUPERSEDES the
        # previous `calculate_generation_metadata(Image.fromarray(frames[0]), ...)`
        # call, which additionally computed a pixel `image_hash` of the
        # GENERATED frame that was never read (create_db_image_record always
        # received the empty-string sentinel for image_hash below) -- that
        # dead computation is dropped along with it.
        params["source_image_hash"] = calculate_bytes_hash(video_data)

        # Encode mp4 (+ mux audio), poster PNG, and sidecar JSON. When
        # video_lossless, `filename` is an FFV1-in-mkv master (byte-exact,
        # not browser-playable) and preview_filename is an H.264 proxy for
        # the gallery's <video> element (see save_video_with_metadata).
        filename, preview_filename = save_video_with_metadata(
            frames,
            audio,
            audio_sample_rate,
            params,
            "outpaint_vid",
            model_info=pipeline_manager.current_model_info,
            lossless=video_lossless,
        )

        # Thumbnail from the poster PNG (same base name as the master, mkv or mp4).
        base_name = os.path.splitext(filename)[0]
        poster_path = os.path.join(settings.outputs_dir, f"{base_name}.png")
        if os.path.exists(poster_path):
            create_thumbnail(poster_path)

        # Hash the saved MASTER file's bytes (see calculate_bytes_hash's docstring).
        _media_hash = await _hash_saved_media(os.path.join(settings.outputs_dir, filename))

        # Record video-specific fields into parameters JSON for the gallery.
        num_frames_out = int(frames.shape[0])
        fps_out = float(params.get("frame_rate", 24.0))
        params_for_db = {k: v for k, v in params.items() if not k.startswith("_")}
        params_for_db["num_frames"] = num_frames_out
        params_for_db["fps"] = fps_out
        params_for_db["duration"] = (num_frames_out / fps_out) if fps_out else 0.0
        params_for_db["audio_enable"] = bool(params.get("audio_enable", True) and audio is not None)
        params_for_db["is_video"] = True
        params_for_db["hash_kind"] = "file_bytes"
        if preview_filename:
            params_for_db["preview_filename"] = preview_filename
        # The gallery's `cfg_scale` / `steps` COLUMNS are filled by
        # `create_db_image_record` from the keys the IMAGE routes use. A video
        # request carries neither `cfg_scale` nor `steps`, so every video row so
        # far recorded that helper's literal fallbacks -- 7.0 and 20 -- which no
        # video generation ever used. On a guidance-distilled architecture that
        # is not merely stale: a row claiming CFG 7.0 contradicts the ignored-
        # guidance warning contract stated in the same response. Map the real
        # video keys onto the columns instead.
        params_for_db["cfg_scale"] = float(params.get("guidance_scale", 1.0))
        params_for_db["steps"] = int(params.get("num_inference_steps", 0))
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        model_name, model_hash = extract_model_info(pipeline_manager)

        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="outpaint_vid",
            image_hash=_media_hash,
            lora_names=extract_lora_names(params.get("loras") or []),
            model_name=model_name,
            model_hash=model_hash,
            source_image_hash=params["source_image_hash"],
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except (GenerationError, CustomValidationError, NotFoundError) as e:
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except ValueError as e:
        # Same reasoning as /generate/ref2vid's identical clause: the ref2va
        # normalisation rules (aspect ratio, the 22-frame floor) raise
        # ValueError from the ops layer, and they are client errors about the
        # uploaded media, not generation failures.
        fail_generation(str(e), generation_id=_gen_id)
        raise CustomValidationError("Invalid MiniMax-H3 reference", detail=str(e))
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Video outpaint generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )


@router.post("/generate/inpaint/video")
async def generate_inpaint_video(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(INPAINT_VIDEO_DEFAULTS["negative_prompt"]),
    # The five keys a per-arch overlay can change are `Form(None)` SENTINELS,
    # exactly as on /generate/outpaint/video: a multipart route has no
    # `model_fields_set`, so an omitted field is only distinguishable from a
    # sent one by the sentinel. There is no clip-length key in the set (or at
    # all): the output is as long as the trimmed input.
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    frame_rate: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    seed: int = Form(INPAINT_VIDEO_DEFAULTS["seed"]),
    num_videos_per_prompt: int = Form(INPAINT_VIDEO_DEFAULTS["num_videos_per_prompt"]),
    max_sequence_length: int = Form(INPAINT_VIDEO_DEFAULTS["max_sequence_length"]),
    audio_enable: bool = Form(INPAINT_VIDEO_DEFAULTS["audio_enable"]),
    # REQUIRED, and in PIXEL frames of the TRIMMED clip: start inclusive, end
    # exclusive. There is no defensible default range, so neither has one.
    regenerate_start_frame: int = Form(...),
    regenerate_end_frame: int = Form(...),
    input_trim_start_frames: int = Form(INPAINT_VIDEO_DEFAULTS["input_trim_start_frames"]),
    input_trim_end_frames: int = Form(INPAINT_VIDEO_DEFAULTS["input_trim_end_frames"]),
    # SENTINEL, for the reason the geometry fields above are: the base
    # "regenerate" is architecture-neutral and MiniMax-H3 overlays
    # "preserve_input", which an omitted field has to be able to reach.
    inpaint_video_audio_mode: Optional[str] = Form(None),
    spatial_mask_manifest: Optional[str] = Form(None),
    spatial_mask_files: List[UploadFile] = File([]),
    spatial_mask_ids: List[str] = Form([]),
    attention_type: str = Form(INPAINT_VIDEO_DEFAULTS["attention_type"]),
    blocks_to_swap: int = Form(INPAINT_VIDEO_DEFAULTS["blocks_to_swap"]),
    fuse_output_proj: bool = Form(INPAINT_VIDEO_DEFAULTS["fuse_output_proj"]),
    fbcache_enable: bool = Form(INPAINT_VIDEO_DEFAULTS["fbcache_enable"]),
    fbcache_threshold: float = Form(INPAINT_VIDEO_DEFAULTS["fbcache_threshold"]),
    fbcache_warmup_steps: int = Form(INPAINT_VIDEO_DEFAULTS["fbcache_warmup_steps"]),
    spectrum_enable: bool = Form(INPAINT_VIDEO_DEFAULTS["spectrum_enable"]),
    spectrum_w: float = Form(INPAINT_VIDEO_DEFAULTS["spectrum_w"]),
    spectrum_w_decay: float = Form(INPAINT_VIDEO_DEFAULTS["spectrum_w_decay"]),
    spectrum_delta_cap: float = Form(INPAINT_VIDEO_DEFAULTS["spectrum_delta_cap"]),
    spectrum_m: int = Form(INPAINT_VIDEO_DEFAULTS["spectrum_m"]),
    spectrum_lam: float = Form(INPAINT_VIDEO_DEFAULTS["spectrum_lam"]),
    spectrum_warmup_steps: int = Form(INPAINT_VIDEO_DEFAULTS["spectrum_warmup_steps"]),
    spectrum_window_size: int = Form(INPAINT_VIDEO_DEFAULTS["spectrum_window_size"]),
    spectrum_flex_window: float = Form(INPAINT_VIDEO_DEFAULTS["spectrum_flex_window"]),
    spectrum_tail: float = Form(INPAINT_VIDEO_DEFAULTS["spectrum_tail"]),
    spectrum_max_cache: int = Form(INPAINT_VIDEO_DEFAULTS["spectrum_max_cache"]),
    vae_path: Optional[str] = Form(INPAINT_VIDEO_DEFAULTS["vae_path"]),
    text_encoder_path: Optional[str] = Form(INPAINT_VIDEO_DEFAULTS["text_encoder_path"]),
    unet_quantization: Optional[str] = Form(INPAINT_VIDEO_DEFAULTS["unet_quantization"]),
    quantized_gemm_mode: Optional[str] = Form(INPAINT_VIDEO_DEFAULTS["quantized_gemm_mode"]),
    video_lossless: bool = Form(INPAINT_VIDEO_DEFAULTS["video_lossless"]),
    video: UploadFile = File(...),
    # Generation-time LoRA. See Txt2VidRequest.loras.
    loras: str = Form("[]"),
    db: Session = Depends(get_gallery_db)
):
    """Video temporal inpaint (MiniMax-H3 `fl2va`): regenerate the frames in
    `[regenerate_start_frame, regenerate_end_frame)` of an uploaded clip and
    preserve the rest. Multipart form: the `video` clip plus the parameters
    below. The output is the same length as the TRIMMED input.

    **Preserved frames are the input's own pixels, pasted back after decode**;
    the model conditions on their re-encoded latents while generating the
    selected range. The paste is unconditional, so the preserved region is
    exact at the frames handoff (and through the FILE under `video_lossless`,
    which encodes FFV1 instead of lossy H.264). Between the pasted and the
    generated pixels there is a cut at the range edges, the same kind image
    outpaint carries.

    **Granularity.** The video VAE stores frames in groups of up to 4, and a
    group is regenerated or preserved as a whole, so the requested range is
    expanded OUTWARD to group boundaries -- never shrunk -- and the effective
    span comes back in `warnings[]` when the expansion changed the request.

    **Length.** Temporal inpaint samples the whole clip, so it is the TRIMMED
    input that must be a valid clip length (`17n + 5`, 124..362 on MiniMax-H3),
    not a generated span. It is never snapped -- that would delete frames you
    asked to keep -- so an invalid length is a 400 naming the trims that reach
    the nearest valid ones. This is the inverse of `/generate/outpaint/video`,
    where the grid binds the GENERATED span, which is why the two are separate
    endpoints.

    **Audio (`inpaint_video_audio_mode`, only relevant when `audio_enable` is
    true).** `preserve_input` (the MiniMax-H3 default) pins the clip's own track
    as conditioning across the whole clip and muxes it back verbatim, so the
    regenerated span carries the original soundtrack; it falls back to
    `regenerate` with a warning when the clip has no audio stream.
    `regenerate` generates the soundtrack for the whole clip, so the preserved
    video span carries generated audio that need not match its visuals.

    **Spatial mask (optional).** When `spatial_mask_manifest` is supplied,
    `spatial_mask_ids` and `spatial_mask_files` must have the same length and
    provide exactly one PNG for each unique `mask_id` referenced by the
    manifest. IDs, not upload filenames, define the mapping. The manifest and
    PNGs are parsed and decoded before any GPU slot is acquired; the validated
    timeline and decoded arrays are passed to the pipeline as explicit keyword
    arguments. A manifest may contain at most 128 keyframes and 64 unique PNG
    assets.

    **Requirements:** a MiniMax-H3 `fl2va` model must be loaded (400
    otherwise -- the mid-clip pin is measured on that partition and no other
    architecture implements this endpoint). Without a spatial mask, the range
    must leave something preserved; with a spatial mask, a full-clip range is
    allowed only when the mask both generates and preserves at least one video
    token.
    """
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from utils.video_utils import save_video_with_metadata, load_video_frames, extract_audio_stream, probe_upload_clip
    from api.generation_utils import latent_frame_spans, plan_video_inpaint_span
    from api.param_defaults import inpaint_video_defaults_for_arch
    from core.models.components.wiring import temporal_spec_for_arch
    from core.inference.video_mask_timeline import (
        MaskDecodeError,
        ManifestValidationError,
        build_spatial_mask_plan,
        decode_mask_pngs,
        parse_mask_timeline_manifest,
    )

    if not getattr(pipeline_manager, "is_minimax_h3_model", False):
        raise CustomValidationError(
            "No MiniMax-H3 model loaded",
            detail="Temporal inpaint pins the kept frames' own latents inside one packed "
                   "sequence, which is a MiniMax-H3 mechanism; LTX-2.3 has no equivalent. Load a "
                   "MiniMax-H3 fl2va model, or use /generate/outpaint/video to extend a clip.",
        )
    # The partition gate, worded as /generate/img2vid's is: the mid-clip pin was
    # measured on the fl2va weights, and ref2va reads reference blocks instead.
    _h3_variant = ((pipeline_manager.current_model_info or {}).get("variant") or "").lower()
    if _h3_variant == "ref2va":
        raise CustomValidationError(
            "The loaded MiniMax-H3 transformer is the ref2va variant, not fl2va",
            detail="Temporal inpaint is offered on the `fl2va` partition, which serves "
                   "/generate/txt2vid, /generate/img2vid and /generate/outpaint/video: the "
                   "mid-clip pin is measured there and nowhere else, and combining it with "
                   "reference conditioning is not implemented on any endpoint. Load "
                   "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors.",
        )
    if inpaint_video_audio_mode is not None and inpaint_video_audio_mode not in (
            "regenerate", "preserve_input"):
        raise CustomValidationError(
            "Invalid inpaint_video_audio_mode",
            detail=f"Must be 'regenerate' or 'preserve_input', got {inpaint_video_audio_mode!r}.",
        )

    _vid_arch = (pipeline_manager.current_model_info or {}).get("type")
    _resolved = inpaint_video_defaults_for_arch(_vid_arch)
    if width is None:
        width = int(_resolved["width"])
    if height is None:
        height = int(_resolved["height"])
    if frame_rate is None:
        frame_rate = float(_resolved["frame_rate"])
    if num_inference_steps is None:
        num_inference_steps = int(_resolved["num_inference_steps"])
    if guidance_scale is None:
        guidance_scale = float(_resolved["guidance_scale"])
    if inpaint_video_audio_mode is None:
        inpaint_video_audio_mode = str(_resolved["inpaint_video_audio_mode"])

    _spec = temporal_spec_for_arch(_vid_arch)
    _align = _spec.pixel_align if _spec is not None else 32
    if width < _align or height < _align or width % _align != 0 or height % _align != 0:
        raise CustomValidationError(
            f"width and height must both be >= {_align} and divisible by {_align}",
            detail=f"Got width={width}, height={height}. Round each to a multiple of {_align}, "
                   f"minimum {_align}.",
        )
    if _spec is not None and _spec.max_pixel_hw is not None:
        _short_cap, _long_cap = min(_spec.max_pixel_hw), max(_spec.max_pixel_hw)
        if min(width, height) > _short_cap or max(width, height) > _long_cap:
            raise CustomValidationError(
                f"the canvas exceeds this model's {_short_cap}x{_long_cap} envelope",
                detail=f"Got {width}x{height}. The loaded model generates with a short edge of at "
                       f"most {_short_cap} and a long edge of at most {_long_cap}, in either "
                       f"orientation.",
            )

    spatial_mask_timeline = None
    spatial_mask_arrays = None
    if spatial_mask_manifest is not None:
        try:
            spatial_mask_timeline = parse_mask_timeline_manifest(spatial_mask_manifest)
        except ManifestValidationError as exc:
            raise CustomValidationError(
                "Invalid spatial_mask_manifest",
                detail=str(exc),
            ) from exc

        if (spatial_mask_timeline.canvas.width != width or
                spatial_mask_timeline.canvas.height != height):
            raise CustomValidationError(
                "Spatial mask canvas does not match the resolved output canvas",
                detail=(
                    f"Manifest canvas is {spatial_mask_timeline.canvas.width}x"
                    f"{spatial_mask_timeline.canvas.height}; resolved output is "
                    f"{width}x{height}."
                ),
            )
        if any(
            keyframe.frame < regenerate_start_frame
            or keyframe.frame >= regenerate_end_frame
            for keyframe in spatial_mask_timeline.keyframes
        ):
            raise CustomValidationError(
                "Spatial mask keyframes must be inside the regenerate range",
                detail=(
                    f"Expected frames in [{regenerate_start_frame}, {regenerate_end_frame}); "
                    "the range is measured in trimmed input frames."
                ),
            )

        mask_files = spatial_mask_files or []
        mask_ids = spatial_mask_ids or []
        if len(mask_files) != len(mask_ids):
            raise CustomValidationError(
                "spatial_mask_files and spatial_mask_ids must have the same number of entries",
                detail=f"Got {len(mask_files)} files and {len(mask_ids)} IDs.",
            )

        seen_mask_ids = set()
        duplicate_ids = set()
        for mask_id in mask_ids:
            if mask_id in seen_mask_ids:
                duplicate_ids.add(mask_id)
            seen_mask_ids.add(mask_id)
        duplicate_ids = sorted(duplicate_ids)
        if duplicate_ids:
            raise CustomValidationError(
                "spatial_mask_ids must not contain duplicate IDs",
                detail=f"Duplicate IDs: {duplicate_ids}",
            )

        expected_mask_ids = {keyframe.mask_id for keyframe in spatial_mask_timeline.keyframes}
        actual_mask_ids = set(mask_ids)
        missing_mask_ids = sorted(expected_mask_ids - actual_mask_ids)
        unknown_mask_ids = sorted(actual_mask_ids - expected_mask_ids)
        if missing_mask_ids or unknown_mask_ids:
            raise CustomValidationError(
                "spatial_mask_ids must exactly match the manifest keyframe mask IDs",
                detail=(
                    f"Missing IDs: {missing_mask_ids}; unknown IDs: {unknown_mask_ids}."
                ),
            )

        mask_bytes_by_id = {}
        for mask_id, mask_file in zip(mask_ids, mask_files):
            try:
                mask_bytes = await mask_file.read()
            except Exception as exc:
                raise CustomValidationError(
                    "Failed to read a spatial mask upload",
                    detail=f"Mask ID {mask_id!r}: {exc}",
                ) from exc
            if not mask_bytes:
                raise CustomValidationError(
                    "Spatial mask uploads must not be empty",
                    detail=f"Mask ID {mask_id!r} has an empty upload.",
                )
            mask_bytes_by_id[mask_id] = mask_bytes

        try:
            spatial_mask_arrays = decode_mask_pngs(
                mask_bytes_by_id,
                spatial_mask_timeline,
            )
        except MaskDecodeError as exc:
            raise CustomValidationError(
                "Invalid spatial mask PNG upload",
                detail=str(exc),
            ) from exc
    elif spatial_mask_files or spatial_mask_ids:
        raise CustomValidationError(
            "spatial_mask_manifest is required when spatial mask uploads are supplied",
            detail="Send the manifest together with spatial_mask_ids and spatial_mask_files.",
        )

    if frame_rate <= 0:
        raise CustomValidationError(
            "frame_rate must be > 0",
            detail=f"Got frame_rate={frame_rate}.",
        )
    validate_video_steps({"num_inference_steps": num_inference_steps}, _vid_arch)

    # Decode the clip (+ its audio track) BEFORE the GPU slot, so a malformed
    # upload is a 400 that reserved nothing. The bound is the arch's own longest
    # clip: nothing past it can be part of a valid trimmed length. `+ 1`: see
    # the post-decode check below -- this decode bound must never itself equal
    # a valid trimmed length, or a clip that is actually LONGER than the cap
    # decodes down to exactly the cap and looks indistinguishable from a clip
    # that really was that long.
    _arch_max_frames = int(_spec.max_frames if _spec is not None and _spec.max_frames else 362)
    _decode_max_frames = max(0, input_trim_start_frames) + _arch_max_frames + 1

    # Since this endpoint's OUTPUT length equals the TRIMMED INPUT length (no
    # `total_frames` to snap to -- see INPAINT_VIDEO_DEFAULTS's docstring), a
    # clip whose trimmed length exceeds the arch's own longest producible clip
    # cannot be served AT ALL, at any setting. Refuse this with a 400 naming
    # both numbers, using an ffprobe-only (no decode) frame count so the
    # refusal is cheap even for a clip far past the cap. Judged on frames
    # "before trimming" (the probed count) vs. "after trimming" (what
    # plan_video_inpaint_span actually receives): a clip that is only too
    # long BEFORE the user's own input_trim_start_frames/input_trim_end_frames
    # trims it down must still be accepted, so the comparison below applies
    # the requested trim to the probed count first.
    #
    # This is the CHEAP path, not the only one: `dataset_scanner`'s frame-count
    # probe falls back to `round(fps * duration)` on a container/codec that
    # does not report `nb_frames` (e.g. VFR webm/matroska) -- an ESTIMATE that
    # can UNDER-count the true length. An under-counting estimate that still
    # lands at/below `_arch_max_frames` would let a genuinely over-length clip
    # sail past this refusal; the old code then trusted `_decode_max_frames`
    # (== `_arch_max_frames` exactly, no headroom) to silently crop it to the
    # cap, which is always ON the arch's grid and so passed every downstream
    # check with the excess gone and no warning -- exactly the class of bug
    # 94b952d5 fixed elsewhere. The `+ 1` above and the post-decode check below
    # close that: they judge the clip on what ffmpeg ACTUALLY decoded, which is
    # authoritative (not an estimate), so an over-length clip is still refused
    # even when the fast probe above under-counted it.
    try:
        video_data = await video.read()
        _probed = probe_upload_clip(video_data)
        if _probed:
            _probed_n = int(_probed.get("num_frames") or 0)
            if _probed_n > 0:
                _probed_trimmed_len = _probed_n - max(0, input_trim_start_frames) - max(0, input_trim_end_frames)
                if _probed_trimmed_len > _arch_max_frames:
                    raise CustomValidationError(
                        "the trimmed clip is longer than this model can produce",
                        detail=f"This model's longest clip is {_arch_max_frames} frames; the "
                               f"upload has {_probed_n} frames, which after "
                               f"input_trim_start_frames={input_trim_start_frames} and "
                               f"input_trim_end_frames={input_trim_end_frames} is still "
                               f"{_probed_trimmed_len} frames. Since this endpoint's output "
                               f"length equals the trimmed input length, trim the clip down to "
                               f"{_arch_max_frames} frames or fewer before uploading, or trim "
                               f"more of it off with input_trim_start_frames/"
                               f"input_trim_end_frames.",
                    )
        video_frames, source_fps = load_video_frames(
            video_data, max_frames=_decode_max_frames, trim_end_frames=input_trim_end_frames
        )
    except CustomValidationError:
        raise
    except Exception as e:
        raise CustomValidationError(
            "Failed to decode the uploaded video clip",
            detail=str(e),
        )

    input_audio = None
    if inpaint_video_audio_mode == "preserve_input":
        input_audio = extract_audio_stream(video_data)

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "frame_rate": frame_rate,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "num_videos_per_prompt": num_videos_per_prompt,
        "max_sequence_length": max_sequence_length,
        "audio_enable": audio_enable,
        "regenerate_start_frame": regenerate_start_frame,
        "regenerate_end_frame": regenerate_end_frame,
        "input_trim_start_frames": input_trim_start_frames,
        "input_trim_end_frames": input_trim_end_frames,
        "inpaint_video_audio_mode": inpaint_video_audio_mode,
        "attention_type": _validated_attention_type(
            attention_type, INPAINT_VIDEO_DEFAULTS["attention_type"]),
        "blocks_to_swap": blocks_to_swap,
        "fuse_output_proj": fuse_output_proj,
        "fbcache_enable": fbcache_enable,
        "fbcache_threshold": fbcache_threshold,
        "fbcache_warmup_steps": fbcache_warmup_steps,
        "spectrum_enable": spectrum_enable,
        "spectrum_w": spectrum_w,
        "spectrum_w_decay": spectrum_w_decay,
        "spectrum_delta_cap": spectrum_delta_cap,
        "spectrum_m": spectrum_m,
        "spectrum_lam": spectrum_lam,
        "spectrum_warmup_steps": spectrum_warmup_steps,
        "spectrum_window_size": spectrum_window_size,
        "spectrum_flex_window": spectrum_flex_window,
        "spectrum_tail": spectrum_tail,
        "spectrum_max_cache": spectrum_max_cache,
        "vae_path": vae_path,
        "text_encoder_path": text_encoder_path,
        "unet_quantization": unet_quantization,
        "quantized_gemm_mode": _normalize_media_qgm(quantized_gemm_mode),
        "video_lossless": video_lossless,
    }

    # Generation-time LoRA (JSON string). See Txt2VidRequest.loras.
    import json
    params["loras"] = json.loads(loras) if loras else []

    trimmed_len = video_frames.shape[0] - max(0, input_trim_start_frames) - max(0, input_trim_end_frames)
    if trimmed_len < 1:
        raise CustomValidationError(
            "Video inpaint input trim leaves no frames",
            detail=f"Uploaded clip has {video_frames.shape[0]} frames; "
                   f"input_trim_start_frames={input_trim_start_frames}, "
                   f"input_trim_end_frames={input_trim_end_frames}.",
        )

    # Post-decode truth, not just the pre-decode ffprobe estimate (see the
    # comment above `_decode_max_frames`): `plan_video_inpaint_span` below
    # also refuses `clip_frames > spec.max_frames`, but against the OLD
    # (un-bumped) `_decode_max_frames` that check was structurally a no-op --
    # `video_frames.shape[0]` could never itself exceed `_arch_max_frames`
    # (relative to the trims), since the ffmpeg `-frames:v` bound made the
    # violation unobservable by construction. The `+ 1` decode headroom fixes
    # that: a clip that decoded to (or past) the bumped cap really did have
    # at least that many frames on offer, so refuse it here, using the number
    # ffmpeg actually produced rather than the (possibly under-counting, see
    # above) probe estimate.
    if trimmed_len > _arch_max_frames:
        _hit_decode_cap = video_frames.shape[0] >= _decode_max_frames
        raise CustomValidationError(
            "the trimmed clip is longer than this model can produce",
            detail=(
                f"This model's longest clip is {_arch_max_frames} frames; the decoded upload "
                f"has {'at least ' if _hit_decode_cap else ''}{trimmed_len} frames after "
                f"input_trim_start_frames={input_trim_start_frames} and "
                f"input_trim_end_frames={input_trim_end_frames}. Since this endpoint's output "
                f"length equals the trimmed input length, trim the clip down to "
                f"{_arch_max_frames} frames or fewer before uploading, or trim more of it off "
                f"with input_trim_start_frames/input_trim_end_frames."
            ),
        )

    # The span, against the loaded arch's own latent chunking. Called HERE
    # (pure, no warnings, no GPU) so an invalid clip length or range is a fast
    # 400 rather than a failure after a 40-second text encode; the backend calls
    # the same function again and is what warns, so the arithmetic exists once.
    _inpaint_plan = plan_video_inpaint_span(
        params,
        _vid_arch,
        clip_frames=int(trimmed_len),
        allow_full_range=spatial_mask_manifest is not None,
    )
    if spatial_mask_timeline is not None:
        components = getattr(pipeline_manager, "minimax_h3_components", None) or {}
        transformer_config = components.get("transformer_config") or {}
        try:
            spatial_scale = int(components["vae_scale_factor_spatial"])
            patch_size = tuple(transformer_config["patch_size"])
            if spatial_scale <= 0 or len(patch_size) != 3:
                raise ValueError("invalid VAE scale or transformer patch size")
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise CustomValidationError(
                "MiniMax-H3 spatial mask geometry is unavailable",
                detail=str(exc),
            ) from exc
        try:
            spans = latent_frame_spans(
                _spec,
                int(_inpaint_plan["latent_frames"]),
            )
            build_spatial_mask_plan(
                spatial_mask_timeline,
                spatial_mask_arrays,
                clip_frames=int(trimmed_len),
                start_frame=int(_inpaint_plan["start_frame"]),
                end_frame=int(_inpaint_plan["end_frame"]),
                latent_frame_spans=spans,
                spatial_scale=spatial_scale,
                patch_h=int(patch_size[1]),
                patch_w=int(patch_size[2]),
            )
        except Exception as exc:
            raise CustomValidationError(
                "Invalid spatial mask timeline",
                detail=str(exc),
            ) from exc

    _gen_id = start_generation("inpaint_vid")
    try:
        pipeline_manager.reset_cancel_flag()

        # `frame_key=None`: the length that has to be on the arch's grid is the
        # trimmed CLIP's, which the span planner above owns and refuses rather
        # than snaps. This call is the spatial and frame-rate rules.
        validate_video_geometry(params, _vid_arch, frame_key=None)
        validate_video_steps(params, _vid_arch)

        from api.arch_capabilities import check_arch_capabilities
        from api.generation_overrides import plan_overrides, apply_overrides
        _override_plan = plan_overrides(pipeline_manager, params.get("vae_path"), params.get("text_encoder_path"))
        apply_overrides(pipeline_manager, _override_plan)
        check_arch_capabilities(params, _vid_arch,
                                defaults=inpaint_video_defaults_for_arch(_vid_arch))

        print(f"inpaint_vid generation params: {sanitize_params_for_logging(params)}")

        def progress_callback(step, total_steps, sub_progress=None):
            from api.generation_status import update_progress
            total = max(total_steps, 1)
            # A mid-step tick names the step still RUNNING; `step` itself keeps
            # counting COMPLETED steps, and the integer-only generation status is
            # left at the last boundary.
            shown = step + 1 if sub_progress is not None else step
            manager.send_progress_sync(
                step, total, f"Generating video: step {shown}/{total}", sub_progress=sub_progress)
            if sub_progress is None:
                update_progress(step, total)

        from core.gpu_coordinator import gpu_coordinator
        from core.inference.generation_timing import generation_timer
        loop = asyncio.get_event_loop()
        generation_timer.reset()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=40, timeout=120.0):
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(params.get("quantized_gemm_mode"))
            frames, audio, audio_sample_rate, actual_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: (
                    pipeline_manager.generate_vid_inpaint(
                        params, video_frames, source_fps, input_audio,
                        progress_callback=progress_callback,
                        spatial_mask_timeline=spatial_mask_timeline,
                        spatial_mask_arrays=spatial_mask_arrays,
                    )
                    if spatial_mask_timeline is not None
                    else pipeline_manager.generate_vid_inpaint(
                        params, video_frames, source_fps, input_audio,
                        progress_callback=progress_callback,
                    )
                )
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        apply_generation_timings(params, time.perf_counter() - _gen_start)
        _record_media_gemm_outcome(params, fp8_gemm, _vid_arch)
        record_attention_backend(params, _gen_id)
        record_model_variant(params, pipeline_manager)

        vae_name, vae_hash = extract_vae_info(pipeline_manager)
        if vae_name:
            params["vae_name"] = vae_name
        if vae_hash:
            params["vae_hash"] = vae_hash

        params["seed"] = actual_seed

        # Hash the uploaded clip's raw bytes -- see the equivalent block in
        # /generate/outpaint/video for why this supersedes a frame-0 pixel hash
        # of `video_data` and drops the dead generated-frame `image_hash`
        # computation that call used to make alongside it.
        params["source_image_hash"] = calculate_bytes_hash(video_data)

        # See save_video_with_metadata's docstring: when video_lossless,
        # `filename` is an FFV1-in-mkv master (byte-exact, not browser-
        # playable) and preview_filename is an H.264 proxy for the gallery.
        filename, preview_filename = save_video_with_metadata(
            frames,
            audio,
            audio_sample_rate,
            params,
            "inpaint_vid",
            model_info=pipeline_manager.current_model_info,
            lossless=video_lossless,
        )

        base_name = os.path.splitext(filename)[0]
        poster_path = os.path.join(settings.outputs_dir, f"{base_name}.png")
        if os.path.exists(poster_path):
            create_thumbnail(poster_path)

        # Hash the saved MASTER file's bytes (see calculate_bytes_hash's docstring).
        _media_hash = await _hash_saved_media(os.path.join(settings.outputs_dir, filename))

        num_frames_out = int(frames.shape[0])
        fps_out = float(params.get("frame_rate", 24.0))
        params_for_db = {k: v for k, v in params.items() if not k.startswith("_")}
        params_for_db["num_frames"] = num_frames_out
        params_for_db["fps"] = fps_out
        params_for_db["duration"] = (num_frames_out / fps_out) if fps_out else 0.0
        params_for_db["audio_enable"] = bool(params.get("audio_enable", True) and audio is not None)
        params_for_db["is_video"] = True
        params_for_db["hash_kind"] = "file_bytes"
        if preview_filename:
            params_for_db["preview_filename"] = preview_filename
        # See the same two lines on /generate/outpaint/video: the gallery's
        # cfg_scale/steps COLUMNS are filled from the image routes' keys, which
        # a video request does not carry.
        params_for_db["cfg_scale"] = float(params.get("guidance_scale", 1.0))
        params_for_db["steps"] = int(params.get("num_inference_steps", 0))
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        model_name, model_hash = extract_model_info(pipeline_manager)

        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="inpaint_vid",
            image_hash=_media_hash,
            lora_names=extract_lora_names(params.get("loras") or []),
            model_name=model_name,
            model_hash=model_hash,
            source_image_hash=params["source_image_hash"],
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except (GenerationError, CustomValidationError, NotFoundError) as e:
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Video inpaint generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )


@router.get("/models/upscalers")
async def list_upscaler_models(db: Session = Depends(get_gallery_db)):
    """List upscaler model files (.pth/.safetensors) under <models_dir>/upscalers/
    and each additional model dir's upscalers/ subdirectory. Creates the primary
    directory if missing.
    """
    settings_record = db.query(UserSettings).first()
    additional_model_dirs = settings_record.model_dirs if settings_record else []
    all_dirs = [settings.models_dir] + list(additional_model_dirs)

    primary_upscalers_dir = os.path.join(settings.models_dir, "upscalers")
    os.makedirs(primary_upscalers_dir, exist_ok=True)

    upscaler_models = []
    for base_dir in all_dirs:
        candidate_dir = os.path.join(base_dir, "upscalers")
        if not os.path.isdir(candidate_dir):
            continue
        for item in os.listdir(candidate_dir):
            if not (item.endswith(".pth") or item.endswith(".safetensors")):
                continue
            item_path = os.path.join(candidate_dir, item)
            if not os.path.isfile(item_path):
                continue
            size_mb = os.path.getsize(item_path) / (1024 ** 2)
            upscaler_models.append({
                "name": item,
                "path": item_path,
                "size_mb": round(size_mb, 2),
                "source_dir": candidate_dir,
            })
    return {"models": upscaler_models}


@router.post("/generate/inpaint")
async def generate_inpaint(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(20),
    cfg_scale: float = Form(7.0),
    denoising_strength: float = Form(0.75),
    img2img_fix_steps: bool = Form(True),
    sampler: str = Form("euler"),
    schedule_type: str = Form("uniform"),
    seed: int = Form(-1),
    ancestral_seed: int = Form(-1),
    width: int = Form(1024),
    height: int = Form(1024),
    mask_blur: int = Form(4),
    inpaint_full_res: bool = Form(False),
    inpaint_full_res_padding: int = Form(32),
    inpaint_fill_mode: str = Form("original"),
    inpaint_fill_strength: float = Form(1.0),
    inpaint_blur_strength: float = Form(1.0),
    region_prompt: str = Form(GENERATION_DEFAULTS["region_prompt"]),  # Regional additional prompt (SD/SDXL): conditions ONLY the generated/repaint region
    region_negative_prompt: str = Form(GENERATION_DEFAULTS["region_negative_prompt"]),
    region_prompt_strength: float = Form(GENERATION_DEFAULTS["region_prompt_strength"]),  # 0-2; 0 = feature inactive
    region_prompt_method: str = Form(GENERATION_DEFAULTS["region_prompt_method"]),  # "cfg" | "attention" (not yet implemented)
    region_mask_feather: float = Form(GENERATION_DEFAULTS["region_mask_feather"]),  # Gaussian sigma (latent cells); 0 = hard mask
    seam_structure_strength: float = Form(GENERATION_DEFAULTS["seam_structure_strength"]),  # SSC: continue structures across the mask boundary (SD/SDXL); 0 = off
    seam_structure_depth: float = Form(GENERATION_DEFAULTS["seam_structure_depth"]),  # SSC repaint-side collar depth (latent cells)
    seam_structure_end: float = Form(GENERATION_DEFAULTS["seam_structure_end"]),  # SSC schedule progress at which the effect decays to 0
    seam_structure_saliency: float = Form(GENERATION_DEFAULTS["seam_structure_saliency"]),  # SSC saliency-gate midpoint (x boundary-ribbon median)
    seam_structure_max_area: float = Form(GENERATION_DEFAULTS["seam_structure_max_area"]),  # SSC safety cap: max fraction of the repaint region the gate may cover
    boundary_relax_strength: float = Form(GENERATION_DEFAULTS["boundary_relax_strength"]),  # BDR: soft-pin the SSC-salient keep-side seam band (SD/SDXL); 0 = off
    boundary_relax_width: float = Form(GENERATION_DEFAULTS["boundary_relax_width"]),  # BDR keep-side band width (latent cells)
    boundary_relax_noise: float = Form(GENERATION_DEFAULTS["boundary_relax_noise"]),  # BDR band-noise fraction of the x0-posterior std
    boundary_relax_full_until: float = Form(GENERATION_DEFAULTS["boundary_relax_full_until"]),  # BDR: progress fully-soft until
    boundary_relax_end: float = Form(GENERATION_DEFAULTS["boundary_relax_end"]),  # BDR: progress by which the hard pin is restored
    boundary_relax_paste: str = Form(GENERATION_DEFAULTS["boundary_relax_paste"]),  # BDR Q3 paste variant: "feather" | "exact"
    prompt_chunking_mode: str = Form("a1111"),
    max_prompt_chunks: int = Form(0),
    loras: str = Form("[]"),  # JSON string of LoRA configs
    controlnets: str = Form("[]"),  # JSON string of ControlNet configs
    developer_mode: bool = Form(False),
    cfg_schedule_type: str = Form("constant"),
    cfg_schedule_min: float = Form(1.0),
    cfg_schedule_max: Optional[float] = Form(None),
    cfg_schedule_power: float = Form(2.0),
    cfg_rescale_snr_alpha: float = Form(0.0),
    dynamic_threshold_percentile: float = Form(0.0),
    dynamic_threshold_mimic_scale: float = Form(7.0),
    nag_enable: bool = Form(False),
    nag_scale: float = Form(5.0),
    nag_tau: float = Form(3.5),
    nag_alpha: float = Form(0.25),
    nag_sigma_end: float = Form(3.0),
    nag_negative_prompt: str = Form(""),
    attention_type: str = Form(GENERATION_DEFAULTS["attention_type"]),
    attention_impl: str = Form("conduit"),
    unet_quantization: Optional[str] = Form(None),
    text_encoder_quantization: Optional[str] = Form(None),
    quantized_gemm_mode: Optional[str] = Form(GENERATION_DEFAULTS["quantized_gemm_mode"]),
    cpu_text_encoding: bool = Form(GENERATION_DEFAULTS["cpu_text_encoding"]),
    use_torch_compile: bool = Form(False),
    keep_models_hot: bool = Form(GENERATION_DEFAULTS["keep_models_hot"]),
    vae_tiling: bool = Form(GENERATION_DEFAULTS["vae_tiling"]),
    vae_tile_threshold: int = Form(GENERATION_DEFAULTS["vae_tile_threshold"]),
    vae_tile_mode: str = Form(GENERATION_DEFAULTS["vae_tile_mode"]),
    vae_tile_global_norm: bool = Form(GENERATION_DEFAULTS["vae_tile_global_norm"]),
    color_flatten_strength: int = Form(GENERATION_DEFAULTS["color_flatten_strength"]),
    flatten_in_loop: bool = Form(GENERATION_DEFAULTS["flatten_in_loop"]),
    flatten_in_loop_last_steps: int = Form(GENERATION_DEFAULTS["flatten_in_loop_last_steps"]),
    flatten_in_loop_min_region: float = Form(GENERATION_DEFAULTS["flatten_in_loop_min_region"]),
    vae_drift_correction: bool = Form(GENERATION_DEFAULTS["vae_drift_correction"]),
    spectrum_enable: bool = Form(GENERATION_DEFAULTS["spectrum_enable"]),
    fbcache_enable: bool = Form(GENERATION_DEFAULTS["fbcache_enable"]),
    fbcache_threshold: float = Form(GENERATION_DEFAULTS["fbcache_threshold"]),
    fbcache_warmup_steps: int = Form(GENERATION_DEFAULTS["fbcache_warmup_steps"]),
    fbcache_cache_branch: int = Form(GENERATION_DEFAULTS["fbcache_cache_branch"]),
    spectrum_w: float = Form(GENERATION_DEFAULTS["spectrum_w"]),
    spectrum_w_decay: float = Form(GENERATION_DEFAULTS["spectrum_w_decay"]),
    spectrum_delta_cap: float = Form(GENERATION_DEFAULTS["spectrum_delta_cap"]),
    spectrum_m: int = Form(GENERATION_DEFAULTS["spectrum_m"]),
    spectrum_lam: float = Form(GENERATION_DEFAULTS["spectrum_lam"]),
    spectrum_warmup_steps: int = Form(GENERATION_DEFAULTS["spectrum_warmup_steps"]),
    spectrum_window_size: int = Form(GENERATION_DEFAULTS["spectrum_window_size"]),
    spectrum_flex_window: float = Form(GENERATION_DEFAULTS["spectrum_flex_window"]),
    spectrum_tail: float = Form(GENERATION_DEFAULTS["spectrum_tail"]),
    spectrum_feature_mode: str = Form(GENERATION_DEFAULTS["spectrum_feature_mode"]),
    spectrum_cache_branch: int = Form(GENERATION_DEFAULTS["spectrum_cache_branch"]),
    spectrum_max_cache: int = Form(GENERATION_DEFAULTS["spectrum_max_cache"]),
    enable_block_swap: bool = Form(False),
    blocks_to_swap: int = Form(GENERATION_DEFAULTS["blocks_to_swap"]),
    use_pinned_memory: bool = Form(False),
    block_swap_h2d_only: bool = Form(GENERATION_DEFAULTS["block_swap_h2d_only"]),
    block_swap_ring_size: int = Form(GENERATION_DEFAULTS["block_swap_ring_size"]),
    use_tipo: bool = Form(False),
    tipo_config: str = Form("{}"),  # JSON string of TIPO config
    preview_predicted_x0: bool = Form(False),  # Show predicted x0 in preview instead of current latent
    preview_decoder: str = Form("matrix"),  # Live-preview decoder for FLUX.2-VAE models: "matrix" | "taef2"
    vision_encoder_path: Optional[str] = Form(None),  # Path to SigLIP2 vision encoder safetensors
    vae_path: Optional[str] = Form(GENERATION_DEFAULTS["vae_path"]),  # Per-generation VAE override (dir or standalone VAE)
    text_encoder_path: Optional[str] = Form(GENERATION_DEFAULTS["text_encoder_path"]),  # Per-generation TE override (SD1.5/SDXL only)
    pid_sr_output: str = Form(GENERATION_DEFAULTS["pid_sr_output"]),  # PiD decoder only: "4x" | "original"
    pid_use_gemma: bool = Form(GENERATION_DEFAULTS["pid_use_gemma"]),  # PiD decoder only: opt-in runtime Gemma captioner
    pid_low_vram: bool = Form(GENERATION_DEFAULTS["pid_low_vram"]),  # PiD decoder only: opt-in row-chunked low-VRAM decode (default off, bit-identical when off)
    pid_tile_native: int = Form(GENERATION_DEFAULTS["pid_tile_native"]),  # PiD decoder only: per-tile native-res ceiling for the F9 tiled large-output decode
    pid_tile_overlap_ratio: float = Form(GENERATION_DEFAULTS["pid_tile_overlap_ratio"]),  # PiD decoder only: F9 tile feather overlap ratio
    pid_fast_large_decode: bool = Form(GENERATION_DEFAULTS["pid_fast_large_decode"]),  # PiD decoder only: opt out of F9 tiling back to the F7 whole-latent cap+bicubic path
    original_size_w: int = Form(0),  # SDXL micro-cond override: original width (0 = auto)
    original_size_h: int = Form(0),  # SDXL micro-cond override: original height (0 = auto)
    original_size_scale: float = Form(1.0),  # SDXL micro-cond: original_size = output * scale
    loop_decode: str = Form(GENERATION_DEFAULTS["loop_decode"]),  # Loop-generation decode mode: "full" | "cheap" ("none" unsupported for inpaint)
    input_latent_id: Optional[str] = Form(GENERATION_DEFAULTS["input_latent_id"]),  # Reserved for schema parity -- unsupported for inpaint (see validation below)
    skip_gallery: bool = Form(GENERATION_DEFAULTS["skip_gallery"]),  # Save to disk but skip the gallery DB record/thumbnail
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    ref_images: List[UploadFile] = File(default=[]),  # FLUX.2 Image Edit / Vision Encoder reference images
    db: Session = Depends(get_gallery_db)
):
    """Generate inpainted image"""
    _reject_if_video_model("/generate/inpaint")
    _reject_if_audio_model("/generate/inpaint")
    lora_configs = []
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from api.arch_capabilities import check_arch_capabilities
    from api.generation_overrides import plan_overrides, apply_overrides
    from api.error_handlers import ValidationError
    # Quantized-GEMM path (quantized_gemm_mode): normalized BEFORE start_generation
    # so an invalid value is a 400 rather than a 500 from inside the run. None
    # (the default) leaves the process flags completely untouched.
    from api.quantized_gemm import normalize_mode as _normalize_qgm
    try:
        quantized_gemm_mode = _normalize_qgm(quantized_gemm_mode)
    except ValueError as exc:
        raise ValidationError("Invalid quantized_gemm_mode value", detail=str(exc))
    # Attention backend (attention_type): validated against the attention
    # registry's own vocabulary, also before start_generation. An unknown value
    # used to run native and be RECORDED as the unknown string.
    attention_type = _validated_attention_type(
        attention_type, GENERATION_DEFAULTS["attention_type"])
    if loop_decode not in ("full", "cheap", "none"):
        raise ValidationError(
            "Invalid loop_decode value",
            detail=f"loop_decode must be 'full', 'cheap', or 'none', got {loop_decode!r}",
        )
    if loop_decode == "none":
        raise ValidationError(
            "loop_decode='none' is not supported for inpaint",
            detail="Inpaint's mask compositing requires a decoded image. "
                   "Use loop_decode='cheap' for lower-cost intermediate loop steps instead.",
        )
    if input_latent_id:
        raise ValidationError(
            "input_latent_id (loop latent passthrough) is not supported for inpaint",
            detail="Inpaint's mask compositing requires a real source image. "
                   "Use loop_decode='cheap' for lower-cost intermediate loop steps instead.",
        )
    _override_plan = plan_overrides(pipeline_manager, vae_path, text_encoder_path)
    _gen_id = start_generation("inpaint")
    try:
        # Reset cancellation flag before starting new generation
        pipeline_manager.reset_cancel_flag()

        # Load input image and mask
        image_data = await image.read()
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        mask_data = await mask.read()
        mask_image = Image.open(io.BytesIO(mask_data)).convert("L")

        # Debug: Check mask statistics
        import numpy as np
        mask_array = np.array(mask_image)
        print(f"Mask stats - min: {mask_array.min()}, max: {mask_array.max()}, mean: {mask_array.mean():.2f}")
        print(f"Mask shape: {mask_array.shape}, non-zero pixels: {np.count_nonzero(mask_array)}, white pixels (>200): {np.count_nonzero(mask_array > 200)}")

        # Apply mask blur if specified
        if mask_blur > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))

        # Parse LoRA configs
        import json
        lora_configs = json.loads(loras) if loras else []

        # Parse ControlNet configs
        controlnet_configs = json.loads(controlnets) if controlnets else []
        controlnet_images, style_transfer, style_transfers, style_combine_mode = process_controlnet_configs(
            controlnet_configs,
            generation_type="inpaint"
        )

        # Parse TIPO config
        tipo_config_dict = json.loads(tipo_config) if tipo_config else {}

        # TIPO prompt upsampling (if enabled)
        original_prompt = prompt
        if use_tipo:
            print(f"[TIPO] Upsampling prompt with TIPO...")
            try:
                # Load TIPO model if needed
                model_name = tipo_config_dict.get("model_name", "KBlueLeaf/TIPO-500M")
                if not tipo_manager.loaded or tipo_manager.model_name != model_name:
                    tipo_manager.load_model(model_name)

                # Generate upsampled prompt
                upsampled_prompt = tipo_manager.generate_prompt(
                    input_prompt=prompt,
                    tag_length=tipo_config_dict.get("tag_length", "long"),
                    nl_length=tipo_config_dict.get("nl_length", "long"),
                    temperature=tipo_config_dict.get("temperature", 1.0),
                    top_p=tipo_config_dict.get("top_p", 0.95),
                    top_k=tipo_config_dict.get("top_k", 50),
                    max_new_tokens=tipo_config_dict.get("max_new_tokens", 256),
                    category_order=tipo_config_dict.get("category_order", []),
                    enabled_categories=tipo_config_dict.get("enabled_categories", {}),
                    treat_as_nl=tipo_config_dict.get("treat_as_nl", False)
                )

                # If result is dict (tipo-kgen mode), format it to string
                if isinstance(upsampled_prompt, dict):
                    category_order = tipo_config_dict.get("category_order", [])
                    enabled_categories = tipo_config_dict.get("enabled_categories", {})

                    # If no category order specified, use default
                    if not category_order:
                        category_order = ["special", "quality", "rating", "artist", "copyright", "characters", "meta", "general"]

                    # If no enabled categories specified, enable all by default
                    if not enabled_categories:
                        enabled_categories = {cat: True for cat in category_order}
                        enabled_categories["meta"] = False  # Meta disabled by default

                    prompt = tipo_manager.format_kgen_result(
                        upsampled_prompt,
                        category_order,
                        enabled_categories
                    )
                else:
                    prompt = upsampled_prompt

                print(f"[TIPO] Original prompt: {original_prompt[:100]}...")
                print(f"[TIPO] Upsampled prompt: {prompt[:100]}...")

                # Unload TIPO model to free VRAM
                tipo_manager.unload_model()

            except Exception as e:
                print(f"[TIPO] Error during upsampling: {e}")
                print(f"[TIPO] Using original prompt")
                # Continue with original prompt on error

        # Process reference images (FLUX.2 Image Edit / Vision Encoder)
        ref_image_list = []
        if ref_images:
            for ref_img_file in ref_images:
                img_bytes = await ref_img_file.read()
                ref_image_list.append(Image.open(io.BytesIO(img_bytes)))
            print(f"[FLUX.2 Image Edit] Loaded {len(ref_image_list)} reference image(s)")

        # Load Vision Encoder if requested (non-FLUX.2 only)
        is_flux2 = pipeline_manager.current_model_info and pipeline_manager.current_model_info.get("type") == "flux2"
        if vision_encoder_path and not is_flux2:
            pipeline_manager.load_vision_encoder(vision_encoder_path)

        # Apply (or restore) the planned VAE/TE overrides on the loaded model.
        _override_meta = apply_overrides(
            pipeline_manager, _override_plan,
            pid_sr_output=pid_sr_output, pid_use_gemma=pid_use_gemma, pid_low_vram=pid_low_vram,
            pid_tile_native=pid_tile_native, pid_tile_overlap_ratio=pid_tile_overlap_ratio,
            pid_fast_large_decode=pid_fast_large_decode, prompt=prompt,
        )

        # Generate image
        params = {
            "prompt": prompt,
            "vae_path": vae_path,
            "text_encoder_path": text_encoder_path,
            "pid_sr_output": pid_sr_output,
            "pid_use_gemma": pid_use_gemma,
            "pid_low_vram": pid_low_vram,
            "pid_tile_native": pid_tile_native,
            "pid_tile_overlap_ratio": pid_tile_overlap_ratio,
            "pid_fast_large_decode": pid_fast_large_decode,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_strength": denoising_strength,
            "img2img_fix_steps": img2img_fix_steps,
            "sampler": sampler,
            "schedule_type": schedule_type,
            "seed": seed,
            "ancestral_seed": ancestral_seed,
            "width": width,
            "height": height,
            "mask_blur": mask_blur,
            "inpaint_full_res": inpaint_full_res,
            "inpaint_full_res_padding": inpaint_full_res_padding,
            "inpaint_fill_mode": inpaint_fill_mode,
            "inpaint_fill_strength": inpaint_fill_strength,
            "inpaint_blur_strength": inpaint_blur_strength,
            "region_prompt": region_prompt,
            "region_negative_prompt": region_negative_prompt,
            "region_prompt_strength": region_prompt_strength,
            "region_prompt_method": region_prompt_method,
            "region_mask_feather": region_mask_feather,
            "seam_structure_strength": seam_structure_strength,
            "seam_structure_depth": seam_structure_depth,
            "seam_structure_end": seam_structure_end,
            "seam_structure_saliency": seam_structure_saliency,
            "seam_structure_max_area": seam_structure_max_area,
            "boundary_relax_strength": boundary_relax_strength,
            "boundary_relax_width": boundary_relax_width,
            "boundary_relax_noise": boundary_relax_noise,
            "boundary_relax_full_until": boundary_relax_full_until,
            "boundary_relax_end": boundary_relax_end,
            "boundary_relax_paste": boundary_relax_paste,
            "loras": lora_configs,  # FLUX.2 needs this in params
            "controlnet_images": controlnet_images,
            "style_transfer": style_transfer,
            "style_transfers": style_transfers,
            "style_combine_mode": style_combine_mode,
            "developer_mode": developer_mode,
            "cfg_schedule_type": cfg_schedule_type,
            "cfg_schedule_min": cfg_schedule_min,
            "cfg_schedule_max": cfg_schedule_max,
            "cfg_schedule_power": cfg_schedule_power,
            "cfg_rescale_snr_alpha": cfg_rescale_snr_alpha,
            "dynamic_threshold_percentile": dynamic_threshold_percentile,
            "dynamic_threshold_mimic_scale": dynamic_threshold_mimic_scale,
            "nag_enable": nag_enable,
            "nag_scale": nag_scale,
            "nag_tau": nag_tau,
            "nag_alpha": nag_alpha,
            "nag_sigma_end": nag_sigma_end,
            "nag_negative_prompt": nag_negative_prompt,
            "attention_type": attention_type,
            "attention_impl": attention_impl,
            "unet_quantization": unet_quantization,
            "quantized_gemm_mode": quantized_gemm_mode,
            "original_size_w": original_size_w,
            "original_size_h": original_size_h,
            "original_size_scale": original_size_scale,
            "text_encoder_quantization": text_encoder_quantization,
            "cpu_text_encoding": cpu_text_encoding,
            "use_torch_compile": use_torch_compile,
            "keep_models_hot": keep_models_hot,
            "vae_tiling": vae_tiling,
            "vae_tile_threshold": vae_tile_threshold,
            "vae_tile_mode": vae_tile_mode,
            "vae_tile_global_norm": vae_tile_global_norm,
            "color_flatten_strength": color_flatten_strength,
            "flatten_in_loop": flatten_in_loop,
            "flatten_in_loop_last_steps": flatten_in_loop_last_steps,
            "flatten_in_loop_min_region": flatten_in_loop_min_region,
            "vae_drift_correction": vae_drift_correction,
            "spectrum_enable": spectrum_enable,
            "fbcache_enable": fbcache_enable,
            "fbcache_threshold": fbcache_threshold,
            "fbcache_warmup_steps": fbcache_warmup_steps,
            "fbcache_cache_branch": fbcache_cache_branch,
            "spectrum_w": spectrum_w,
            "spectrum_w_decay": spectrum_w_decay,
            "spectrum_delta_cap": spectrum_delta_cap,
            "spectrum_m": spectrum_m,
            "spectrum_lam": spectrum_lam,
            "spectrum_warmup_steps": spectrum_warmup_steps,
            "spectrum_window_size": spectrum_window_size,
            "spectrum_flex_window": spectrum_flex_window,
            "spectrum_tail": spectrum_tail,
            "spectrum_feature_mode": spectrum_feature_mode,
            "spectrum_cache_branch": spectrum_cache_branch,
            "spectrum_max_cache": spectrum_max_cache,
            "enable_block_swap": enable_block_swap,
            "blocks_to_swap": blocks_to_swap,
            "use_pinned_memory": use_pinned_memory,
            "block_swap_h2d_only": block_swap_h2d_only,
            "block_swap_ring_size": block_swap_ring_size,
            "preview_decoder": preview_decoder,
            "ref_images": ref_image_list,  # FLUX.2 Image Edit reference images
            "loop_decode": loop_decode,
            "skip_gallery": skip_gallery,
        }
        params.update(_override_meta)
        print(f"inpaint generation params: {sanitize_params_for_logging(params)}")

        # inpaint_full_res is accepted for API compatibility but not implemented.
        if inpaint_full_res or inpaint_full_res_padding != 32:
            from api.generation_status import add_warning as _add_warning
            _add_warning(
                "inpaint_full_res is accepted but not implemented; it has no effect",
                code="not_implemented",
            )

        # Set prompt chunking settings
        set_prompt_chunking_settings(
            pipeline_manager,
            prompt_chunking_mode,
            max_prompt_chunks
        )

        # Load LoRAs if specified
        pipeline_manager.inpaint_pipeline, has_step_range_loras = load_loras_for_generation(
            lora_manager,
            pipeline_manager.inpaint_pipeline,
            lora_configs,
            "inpaint"
        )

        # Detect if SDXL
        is_sdxl = pipeline_manager.inpaint_pipeline is not None and \
                  "XL" in pipeline_manager.inpaint_pipeline.__class__.__name__
        is_zimage = pipeline_manager.current_model_info and \
                    pipeline_manager.current_model_info.get("type") == "zimage"
        is_deus = pipeline_manager.current_model_info and \
                  pipeline_manager.current_model_info.get("type") == "deus"
        is_flux2 = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "flux2"
        # Z-Image with SDXL VAE (4ch) needs TAESD-XL instead of TAEF1
        is_zimage_sdxl_vae = is_zimage and \
                             pipeline_manager.current_model_info.get("vae_type") == "sdxl"
        is_anima = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "anima"
        is_lens = pipeline_manager.current_model_info and \
                  pipeline_manager.current_model_info.get("type") == "lens"
        # Ideogram 4 shares AutoencoderKLFlux2's 128-ch packed latent with Lens.
        is_ideogram4 = pipeline_manager.current_model_info and \
                       pipeline_manager.current_model_info.get("type") == "ideogram4"
        is_minit2i = pipeline_manager.current_model_info and \
                     pipeline_manager.current_model_info.get("type") == "minit2i"
        minit2i_vae_type = (pipeline_manager.minit2i_components or {}).get("vae_type", "none") if is_minit2i else "none"
        is_krea2 = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "krea2"

        # Warn about parameters the loaded architecture silently ignores
        _current_arch = pipeline_manager.current_model_info.get("type") if pipeline_manager.current_model_info else None
        check_arch_capabilities(params, _current_arch)

        # Progress callback to send updates via WebSocket
        progress_callback = create_progress_callback_factory(
            taesd_manager,
            manager,
            is_sdxl,
            is_zimage,
            is_deus,
            is_zimage_sdxl_vae,
            is_flux2,
            is_anima,
            is_lens=is_lens,
            is_ideogram4=is_ideogram4,
            is_minit2i=is_minit2i,
            is_krea2=is_krea2,
            img2img_fix_steps=img2img_fix_steps,
            steps=steps,
            image_width=width,
            image_height=height,
            # For flow-matching DiTs (Anima / Z-Image / FLUX.2 / Lens), default to
            # the pred_x0 preview: x_t is mostly noise mid-denoising, while
            # pred_x0 = x_t - σ·v shows the model's current clean-image
            # estimate from the very first steps. Any explicit user override
            # via the API still wins.
            preview_predicted_x0=(preview_predicted_x0 or is_anima or is_zimage or is_flux2 or is_lens or is_ideogram4 or is_minit2i or is_krea2),
            preview_enabled=params.get("preview_enabled", True),
            preview_interval=params.get("preview_interval", 4),
            preview_decoder=params.get("preview_decoder", "matrix")
        )

        # Create step callback for LoRA step range if needed
        step_callback = None
        if has_step_range_loras:
            # Calculate actual steps based on denoising strength
            actual_steps = int(steps * denoising_strength)
            step_callback = create_lora_step_callback(
                lora_manager,
                pipeline_manager.inpaint_pipeline,
                actual_steps
            )

        # Run generation in thread pool to avoid blocking event loop.
        # gpu_coordinator slot pauses any active tagger training first.
        from core.gpu_coordinator import gpu_coordinator
        from core.inference.generation_timing import generation_timer
        loop = asyncio.get_event_loop()
        _peak_gb = _estimate_gen_peak_gb(width, height, 1,
                                         pipeline_manager.current_pipeline_kind)
        generation_timer.reset()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=_peak_gb, timeout=60.0):
            # GEMM flags are process-wide; keep selection and probing in this slot.
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(quantized_gemm_mode)
            result_image, actual_seed, actual_ancestral_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: pipeline_manager.generate_inpaint(params, init_image, mask_image, progress_callback=progress_callback, step_callback=step_callback)
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        # Record total wall time + any phase breakdown the pipeline populated.
        apply_generation_timings(params, time.perf_counter() - _gen_start)

        # Update params with actual seeds
        params["seed"] = actual_seed
        params["ancestral_seed"] = actual_ancestral_seed

        # Add Vision Encoder info to params for PNG metadata and DB storage.
        # Only record VE info when THIS generation actually used reference images
        # (the VE stays loaded "sticky" across generations).
        if ref_image_list:
            ve_name, ve_hash = extract_vision_encoder_info(pipeline_manager)
            if ve_name:
                params["vision_encoder_name"] = ve_name
            if ve_hash:
                params["vision_encoder_hash"] = ve_hash

        # Add VAE identity to params. The VAE always participates in decode, so this
        # is recorded for every generation where it can be determined.
        vae_name, vae_hash = extract_vae_info(pipeline_manager)
        if vae_name:
            params["vae_name"] = vae_name
        if vae_hash:
            params["vae_hash"] = vae_hash

        # FP8 GEMM path. Recorded only for checkpoints that carry weight-only FP8
        # Linear weights, and it records the RESOLVED path rather than the flag —
        # the W8A8 path can be enabled while the device probe rejects every
        # scaling mode, and the two paths are numerically different.
        if fp8_gemm:
            params["fp8_gemm"] = fp8_gemm
        # Report an explicit "w8a8" request that resolved to the dequant path
        # (probe rejected it, or the checkpoint has no quantized layers).
        from api.quantized_gemm import report_quantized_gemm_outcome
        report_quantized_gemm_outcome(quantized_gemm_mode, fp8_gemm, _current_arch)
        # Attention backend that ACTUALLY ran (observed by the conduit), so the
        # PNG metadata and the gallery row cannot name a backend that did not.
        record_attention_backend(params, _gen_id)

        # Save image with metadata (include model info)
        filename = save_image_with_metadata(
            result_image,
            params,
            "inpaint",
            model_info=pipeline_manager.current_model_info,
            generation_id=_gen_id
        )

        # skip_gallery: the file is saved (so a generation loop can still chain
        # to the next step via its path) but no thumbnail/DB record is created.
        if skip_gallery:
            complete_generation({"filename": filename, "seed": actual_seed}, generation_id=_gen_id)
            return {
                "success": True,
                "filename": filename,
                "image_path": f"/outputs/{filename}",
                "actual_seed": actual_seed,
                "warnings": get_warnings(_gen_id),
            }

        image_path = os.path.join(settings.outputs_dir, filename)
        create_thumbnail(image_path)

        # Calculate metadata
        metadata = calculate_generation_metadata(
            result_image,
            lora_configs,
            extract_lora_names,
            calculate_image_hash,
            source_image=init_image,
            mask_image=mask_image,
            encode_mask_func=encode_mask_to_base64
        )

        # Remove image objects from params before saving to DB and calculate ControlNet hashes
        params_for_db = prepare_params_for_db(params, calculate_image_hash)
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        # Extract model name and hash from current_model_info
        model_name, model_hash = extract_model_info(pipeline_manager)

        # Save to database
        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="inpaint",
            image_hash=metadata["image_hash"],
            lora_names=metadata["lora_names"],
            model_name=model_name,
            model_hash=model_hash,
            result_image=result_image,
            source_image_hash=metadata.get("source_image_hash"),
            mask_data_base64=metadata.get("mask_data_base64")
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except GenerationError as e:
        # Re-raise custom errors as-is
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except Exception as e:
        # Wrap unexpected errors in GenerationError
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Inpaint generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )
    finally:
        # Unload LoRAs after generation
        if lora_configs and pipeline_manager.inpaint_pipeline:
            pipeline_manager.inpaint_pipeline = lora_manager.unload_loras(pipeline_manager.inpaint_pipeline)


@router.post("/generate/outpaint")
async def generate_outpaint(
    prompt: str = Form(...),
    negative_prompt: str = Form(OUTPAINT_DEFAULTS["negative_prompt"]),
    steps: int = Form(OUTPAINT_DEFAULTS["steps"]),
    cfg_scale: float = Form(OUTPAINT_DEFAULTS["cfg_scale"]),
    denoising_strength: float = Form(OUTPAINT_DEFAULTS["denoising_strength"]),
    img2img_fix_steps: bool = Form(OUTPAINT_DEFAULTS["img2img_fix_steps"]),
    sampler: str = Form(OUTPAINT_DEFAULTS["sampler"]),
    schedule_type: str = Form(OUTPAINT_DEFAULTS["schedule_type"]),
    seed: int = Form(OUTPAINT_DEFAULTS["seed"]),
    ancestral_seed: int = Form(OUTPAINT_DEFAULTS["ancestral_seed"]),
    # Placement (new for outpaint). canvas_width/canvas_height supersede the
    # shared width/height GenerationParams fields -- this route does not
    # accept width/height directly (PipelineManager.generate_outpaint derives
    # them from the resolved canvas size).
    canvas_width: int = Form(OUTPAINT_DEFAULTS["canvas_width"]),
    canvas_height: int = Form(OUTPAINT_DEFAULTS["canvas_height"]),
    place_x: int = Form(OUTPAINT_DEFAULTS["place_x"]),
    place_y: int = Form(OUTPAINT_DEFAULTS["place_y"]),
    place_width: int = Form(OUTPAINT_DEFAULTS["place_width"]),
    place_height: int = Form(OUTPAINT_DEFAULTS["place_height"]),
    input_crop_x: int = Form(OUTPAINT_DEFAULTS["input_crop_x"]),
    input_crop_y: int = Form(OUTPAINT_DEFAULTS["input_crop_y"]),
    input_crop_w: int = Form(OUTPAINT_DEFAULTS["input_crop_w"]),
    input_crop_h: int = Form(OUTPAINT_DEFAULTS["input_crop_h"]),
    outpaint_fill_mode: str = Form(OUTPAINT_DEFAULTS["outpaint_fill_mode"]),
    mask_blur: int = Form(OUTPAINT_DEFAULTS["mask_blur"]),
    inpaint_full_res: bool = Form(OUTPAINT_DEFAULTS["inpaint_full_res"]),
    inpaint_full_res_padding: int = Form(OUTPAINT_DEFAULTS["inpaint_full_res_padding"]),
    inpaint_fill_mode: str = Form(OUTPAINT_DEFAULTS["inpaint_fill_mode"]),
    inpaint_fill_strength: float = Form(OUTPAINT_DEFAULTS["inpaint_fill_strength"]),
    inpaint_blur_strength: float = Form(OUTPAINT_DEFAULTS["inpaint_blur_strength"]),
    region_prompt: str = Form(OUTPAINT_DEFAULTS["region_prompt"]),  # Regional additional prompt (SD/SDXL): conditions ONLY the generated region
    region_negative_prompt: str = Form(OUTPAINT_DEFAULTS["region_negative_prompt"]),
    region_prompt_strength: float = Form(OUTPAINT_DEFAULTS["region_prompt_strength"]),  # 0-2; 0 = feature inactive
    region_prompt_method: str = Form(OUTPAINT_DEFAULTS["region_prompt_method"]),  # "cfg" | "attention" (not yet implemented)
    region_mask_feather: float = Form(OUTPAINT_DEFAULTS["region_mask_feather"]),  # Gaussian sigma (latent cells); 0 = hard mask
    seam_structure_strength: float = Form(OUTPAINT_DEFAULTS["seam_structure_strength"]),  # SSC: continue thin structures across the boundary (SD/SDXL); 0 = off
    seam_structure_depth: float = Form(OUTPAINT_DEFAULTS["seam_structure_depth"]),  # SSC generate-side collar depth (latent cells)
    seam_structure_end: float = Form(OUTPAINT_DEFAULTS["seam_structure_end"]),  # SSC schedule progress at which the effect decays to 0
    seam_structure_saliency: float = Form(OUTPAINT_DEFAULTS["seam_structure_saliency"]),  # SSC saliency-gate midpoint (x boundary-ribbon median)
    seam_structure_max_area: float = Form(OUTPAINT_DEFAULTS["seam_structure_max_area"]),  # SSC safety cap: max fraction of the generate region the gate may cover
    boundary_relax_strength: float = Form(OUTPAINT_DEFAULTS["boundary_relax_strength"]),  # BDR: soft-pin the SSC-salient keep-side seam band (SD/SDXL); 0 = off
    boundary_relax_width: float = Form(OUTPAINT_DEFAULTS["boundary_relax_width"]),  # BDR keep-side band width (latent cells)
    boundary_relax_noise: float = Form(OUTPAINT_DEFAULTS["boundary_relax_noise"]),  # BDR band-noise fraction of the x0-posterior std
    boundary_relax_full_until: float = Form(OUTPAINT_DEFAULTS["boundary_relax_full_until"]),  # BDR: progress fully-soft until
    boundary_relax_end: float = Form(OUTPAINT_DEFAULTS["boundary_relax_end"]),  # BDR: progress by which the hard pin is restored
    boundary_relax_paste: str = Form(OUTPAINT_DEFAULTS["boundary_relax_paste"]),  # BDR Q3 paste variant: "feather" | "exact"
    outpaint_seam_fix: bool = Form(OUTPAINT_DEFAULTS["outpaint_seam_fix"]),  # Post-generation exposure/tone harmonizer between the generated surroundings and the preserved rect
    outpaint_seam_membrane: bool = Form(OUTPAINT_DEFAULTS["outpaint_seam_membrane"]),  # Harmonic boundary-offset blend (post-decode, before the final paste); False = off (byte-identical)
    outpaint_seam_membrane_band: int = Form(OUTPAINT_DEFAULTS["outpaint_seam_membrane_band"]),  # Taper band width (px) for the seam membrane; 0 = auto
    outpaint_seam_tone_strength: float = Form(OUTPAINT_DEFAULTS["outpaint_seam_tone_strength"]),  # R2 cross-seam low-frequency tone membrane (post-decode, before the final paste); 0 = off (byte-identical)
    outpaint_seam_tone_band: int = Form(OUTPAINT_DEFAULTS["outpaint_seam_tone_band"]),  # Decay band width (px) for the cross-seam tone membrane; 0 = auto (16)
    outpaint_seam_offset_prop: float = Form(OUTPAINT_DEFAULTS["outpaint_seam_offset_prop"]),  # G_prop16 boundary-offset propagation (post-decode, before the final paste); generated-side-only, 0 = off (byte-identical)
    outpaint_paste_feather_px: int = Form(OUTPAINT_DEFAULTS["outpaint_paste_feather_px"]),  # Option E paste-band reconciliation feather (px); 0 = off (byte-identical), independent of/takes precedence over BDR's feather paste
    outpaint_preserve_mode: str = Form(OUTPAINT_DEFAULTS["outpaint_preserve_mode"]),  # "exact" (byte-exact paste, default) | "vae_reconstruct" | "vae_reconstruct_hf" (seamless, NOT byte-identical -- see param_defaults.py)
    outpaint_preview_unpinned_x0: bool = Form(OUTPAINT_DEFAULTS["outpaint_preview_unpinned_x0"]),  # Display-only: send the unpinned model x0 prediction to the mid-sampling preview decoder instead of the pinned composite; final saved image unaffected
    outpaint_boundary_color_strength: float = Form(OUTPAINT_DEFAULTS["outpaint_boundary_color_strength"]),  # B1 continuity fix: in-loop low-freq boundary color proximal strength (SD/SDXL only, 0=off)
    outpaint_resample_count: int = Form(OUTPAINT_DEFAULTS["outpaint_resample_count"]),  # B2 continuity fix: RePaint-style time-travel resample count (SD/SDXL only, 1=off)
    outpaint_jump_length: int = Form(OUTPAINT_DEFAULTS["outpaint_jump_length"]),  # B2 continuity fix: time-travel jump-back length in step indices
    outpaint_reference_strength: float = Form(OUTPAINT_DEFAULTS["outpaint_reference_strength"]),  # B3 continuity fix: masked self-attention KV injection strength (SD/SDXL only, 0=off)
    outpaint_commit_strength: float = Form(OUTPAINT_DEFAULTS["outpaint_commit_strength"]),  # Boundary-outward commitment front (SD/SDXL only, EXPERIMENTAL, 0=off): augments B1's x0-projection
    outpaint_commit_near: float = Form(OUTPAINT_DEFAULTS["outpaint_commit_near"]),  # commit-front: schedule progress at which boundary-touching cells commit
    outpaint_commit_far: float = Form(OUTPAINT_DEFAULTS["outpaint_commit_far"]),  # commit-front: schedule progress at which the farthest generate cells commit
    outpaint_commit_distance: float = Form(OUTPAINT_DEFAULTS["outpaint_commit_distance"]),  # commit-front: distance (latent cells) at which the commit schedule saturates
    outpaint_controlnet_enable: bool = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_enable"]),  # PART A edge-extrapolation ControlNet (SD/SDXL only); 0/False = off (byte-identical)
    outpaint_controlnet_mode: str = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_mode"]),  # "edge_extrapolate" (PART A, anytest) | "crop_mask" (PART B, trained 4-ch outpaint-native CN)
    outpaint_controlnet_model: str = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_model"]),  # ControlNet model checkpoint path (required when enabled)
    outpaint_controlnet_detector: str = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_detector"]),  # "canny" | "lineart" | "lineart_anime"
    outpaint_controlnet_scale: float = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_scale"]),  # ControlNet conditioning scale
    outpaint_controlnet_guidance_start: float = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_guidance_start"]),  # guidance window start (0-1)
    outpaint_controlnet_guidance_end: float = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_guidance_end"]),  # guidance window end (0-1)
    outpaint_controlnet_depth: int = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_depth"]),  # max extrapolation depth (canvas px)
    outpaint_controlnet_taper: float = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_taper"]),  # cosine-squared taper exponent
    outpaint_controlnet_edge_feather_px: float = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_edge_feather_px"]),  # crop_mask inward edge-feather (canvas px); 0.0 = hard edge (current live CN)
    outpaint_controlnet_corner_radius_px: float = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_corner_radius_px"]),  # crop_mask rounded-corner CN conditioning (canvas px); 0.0 = disabled
    outpaint_controlnet_corner_gate_radius_px: float = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_corner_gate_radius_px"]),  # crop_mask per-corner residual gate disk radius (canvas px); 0.0 = disabled
    outpaint_controlnet_corner_gate_min: float = Form(OUTPAINT_DEFAULTS["outpaint_controlnet_corner_gate_min"]),  # crop_mask per-corner residual gate min value at vertex; 1.0 = disabled
    outpaint_pin_corner_relax_radius_px: float = Form(OUTPAINT_DEFAULTS["outpaint_pin_corner_relax_radius_px"]),  # crop_mask L1 four-corner x0-pin softening disk radius (canvas px); 0.0 = disabled
    outpaint_pin_corner_relax_min: float = Form(OUTPAINT_DEFAULTS["outpaint_pin_corner_relax_min"]),  # crop_mask L1 four-corner x0-pin softening min keep-weight at vertex; 1.0 = disabled
    prompt_chunking_mode: str = Form(OUTPAINT_DEFAULTS["prompt_chunking_mode"]),
    max_prompt_chunks: int = Form(OUTPAINT_DEFAULTS["max_prompt_chunks"]),
    loras: str = Form("[]"),  # JSON string of LoRA configs
    controlnets: str = Form("[]"),  # JSON string of ControlNet configs
    developer_mode: bool = Form(OUTPAINT_DEFAULTS["developer_mode"]),
    cfg_schedule_type: str = Form(OUTPAINT_DEFAULTS["cfg_schedule_type"]),
    cfg_schedule_min: float = Form(OUTPAINT_DEFAULTS["cfg_schedule_min"]),
    cfg_schedule_max: Optional[float] = Form(OUTPAINT_DEFAULTS["cfg_schedule_max"]),
    cfg_schedule_power: float = Form(OUTPAINT_DEFAULTS["cfg_schedule_power"]),
    cfg_rescale_snr_alpha: float = Form(OUTPAINT_DEFAULTS["cfg_rescale_snr_alpha"]),
    dynamic_threshold_percentile: float = Form(OUTPAINT_DEFAULTS["dynamic_threshold_percentile"]),
    dynamic_threshold_mimic_scale: float = Form(OUTPAINT_DEFAULTS["dynamic_threshold_mimic_scale"]),
    nag_enable: bool = Form(OUTPAINT_DEFAULTS["nag_enable"]),
    nag_scale: float = Form(OUTPAINT_DEFAULTS["nag_scale"]),
    nag_tau: float = Form(OUTPAINT_DEFAULTS["nag_tau"]),
    nag_alpha: float = Form(OUTPAINT_DEFAULTS["nag_alpha"]),
    nag_sigma_end: float = Form(OUTPAINT_DEFAULTS["nag_sigma_end"]),
    nag_negative_prompt: str = Form(OUTPAINT_DEFAULTS["nag_negative_prompt"]),
    attention_type: str = Form(OUTPAINT_DEFAULTS["attention_type"]),
    attention_impl: str = Form(OUTPAINT_DEFAULTS["attention_impl"]),
    unet_quantization: Optional[str] = Form(OUTPAINT_DEFAULTS["unet_quantization"]),
    text_encoder_quantization: Optional[str] = Form(OUTPAINT_DEFAULTS["text_encoder_quantization"]),
    quantized_gemm_mode: Optional[str] = Form(OUTPAINT_DEFAULTS["quantized_gemm_mode"]),
    cpu_text_encoding: bool = Form(OUTPAINT_DEFAULTS["cpu_text_encoding"]),
    use_torch_compile: bool = Form(OUTPAINT_DEFAULTS["use_torch_compile"]),
    keep_models_hot: bool = Form(OUTPAINT_DEFAULTS["keep_models_hot"]),
    vae_tiling: bool = Form(OUTPAINT_DEFAULTS["vae_tiling"]),
    vae_tile_threshold: int = Form(OUTPAINT_DEFAULTS["vae_tile_threshold"]),
    vae_tile_mode: str = Form(OUTPAINT_DEFAULTS["vae_tile_mode"]),
    vae_tile_global_norm: bool = Form(OUTPAINT_DEFAULTS["vae_tile_global_norm"]),
    color_flatten_strength: int = Form(OUTPAINT_DEFAULTS["color_flatten_strength"]),
    flatten_in_loop: bool = Form(OUTPAINT_DEFAULTS["flatten_in_loop"]),
    flatten_in_loop_last_steps: int = Form(OUTPAINT_DEFAULTS["flatten_in_loop_last_steps"]),
    flatten_in_loop_min_region: float = Form(OUTPAINT_DEFAULTS["flatten_in_loop_min_region"]),
    vae_drift_correction: bool = Form(OUTPAINT_DEFAULTS["vae_drift_correction"]),
    spectrum_enable: bool = Form(OUTPAINT_DEFAULTS["spectrum_enable"]),
    fbcache_enable: bool = Form(OUTPAINT_DEFAULTS["fbcache_enable"]),
    fbcache_threshold: float = Form(OUTPAINT_DEFAULTS["fbcache_threshold"]),
    fbcache_warmup_steps: int = Form(OUTPAINT_DEFAULTS["fbcache_warmup_steps"]),
    fbcache_cache_branch: int = Form(OUTPAINT_DEFAULTS["fbcache_cache_branch"]),
    spectrum_w: float = Form(OUTPAINT_DEFAULTS["spectrum_w"]),
    spectrum_w_decay: float = Form(OUTPAINT_DEFAULTS["spectrum_w_decay"]),
    spectrum_delta_cap: float = Form(OUTPAINT_DEFAULTS["spectrum_delta_cap"]),
    spectrum_m: int = Form(OUTPAINT_DEFAULTS["spectrum_m"]),
    spectrum_lam: float = Form(OUTPAINT_DEFAULTS["spectrum_lam"]),
    spectrum_warmup_steps: int = Form(OUTPAINT_DEFAULTS["spectrum_warmup_steps"]),
    spectrum_window_size: int = Form(OUTPAINT_DEFAULTS["spectrum_window_size"]),
    spectrum_flex_window: float = Form(OUTPAINT_DEFAULTS["spectrum_flex_window"]),
    spectrum_tail: float = Form(OUTPAINT_DEFAULTS["spectrum_tail"]),
    spectrum_feature_mode: str = Form(OUTPAINT_DEFAULTS["spectrum_feature_mode"]),
    spectrum_cache_branch: int = Form(OUTPAINT_DEFAULTS["spectrum_cache_branch"]),
    spectrum_max_cache: int = Form(OUTPAINT_DEFAULTS["spectrum_max_cache"]),
    enable_block_swap: bool = Form(OUTPAINT_DEFAULTS["enable_block_swap"]),
    blocks_to_swap: int = Form(OUTPAINT_DEFAULTS["blocks_to_swap"]),
    use_pinned_memory: bool = Form(OUTPAINT_DEFAULTS["use_pinned_memory"]),
    block_swap_h2d_only: bool = Form(OUTPAINT_DEFAULTS["block_swap_h2d_only"]),
    block_swap_ring_size: int = Form(OUTPAINT_DEFAULTS["block_swap_ring_size"]),
    use_tipo: bool = Form(OUTPAINT_DEFAULTS["use_tipo"]),
    tipo_config: str = Form("{}"),  # JSON string of TIPO config
    preview_predicted_x0: bool = Form(OUTPAINT_DEFAULTS["preview_predicted_x0"]),  # Show predicted x0 in preview instead of current latent
    preview_decoder: str = Form("matrix"),  # Live-preview decoder for FLUX.2-VAE models: "matrix" | "taef2"
    vision_encoder_path: Optional[str] = Form(OUTPAINT_DEFAULTS["vision_encoder_path"]),  # Path to SigLIP2 vision encoder safetensors
    vae_path: Optional[str] = Form(OUTPAINT_DEFAULTS["vae_path"]),  # Per-generation VAE override (dir or standalone VAE)
    text_encoder_path: Optional[str] = Form(OUTPAINT_DEFAULTS["text_encoder_path"]),  # Per-generation TE override (SD1.5/SDXL only)
    pid_sr_output: str = Form(OUTPAINT_DEFAULTS["pid_sr_output"]),  # PiD decoder only: "4x" | "original"
    pid_use_gemma: bool = Form(OUTPAINT_DEFAULTS["pid_use_gemma"]),  # PiD decoder only: opt-in runtime Gemma captioner
    pid_low_vram: bool = Form(OUTPAINT_DEFAULTS["pid_low_vram"]),  # PiD decoder only: opt-in row-chunked low-VRAM decode (default off, bit-identical when off)
    pid_tile_native: int = Form(OUTPAINT_DEFAULTS["pid_tile_native"]),  # PiD decoder only: per-tile native-res ceiling for the F9 tiled large-output decode
    pid_tile_overlap_ratio: float = Form(OUTPAINT_DEFAULTS["pid_tile_overlap_ratio"]),  # PiD decoder only: F9 tile feather overlap ratio
    pid_fast_large_decode: bool = Form(OUTPAINT_DEFAULTS["pid_fast_large_decode"]),  # PiD decoder only: opt out of F9 tiling back to the F7 whole-latent cap+bicubic path
    original_size_w: int = Form(OUTPAINT_DEFAULTS["original_size_w"]),  # SDXL micro-cond override: original width (0 = auto)
    original_size_h: int = Form(OUTPAINT_DEFAULTS["original_size_h"]),  # SDXL micro-cond override: original height (0 = auto)
    original_size_scale: float = Form(OUTPAINT_DEFAULTS["original_size_scale"]),  # SDXL micro-cond: original_size = output * scale
    loop_decode: str = Form(OUTPAINT_DEFAULTS["loop_decode"]),  # Loop-generation decode mode: "full" | "cheap" ("none" unsupported for outpaint)
    input_latent_id: Optional[str] = Form(OUTPAINT_DEFAULTS["input_latent_id"]),  # Reserved for schema parity -- unsupported for outpaint (see validation below)
    skip_gallery: bool = Form(OUTPAINT_DEFAULTS["skip_gallery"]),  # Save to disk but skip the gallery DB record/thumbnail
    image: UploadFile = File(...),
    ref_images: List[UploadFile] = File(default=[]),  # FLUX.2 Image Edit / Vision Encoder reference images
    db: Session = Depends(get_gallery_db)
):
    """Generate an outpainted image: place `image` inside a larger canvas and
    generate everything outside it. The placed input region is preserved
    exactly (byte-identical), regardless of the loaded architecture or
    denoising_strength -- see core/inference/outpaint_utils.py and
    PipelineManager.generate_outpaint."""
    _reject_if_video_model("/generate/outpaint")
    _reject_if_audio_model("/generate/outpaint")
    lora_configs = []
    from api.generation_status import start_generation, complete_generation, fail_generation, get_warnings
    from api.arch_capabilities import check_arch_capabilities
    from api.generation_overrides import plan_overrides, apply_overrides
    from api.error_handlers import ValidationError
    # Quantized-GEMM path (quantized_gemm_mode): normalized BEFORE start_generation
    # so an invalid value is a 400 rather than a 500 from inside the run. None
    # (the default) leaves the process flags completely untouched.
    from api.quantized_gemm import normalize_mode as _normalize_qgm
    try:
        quantized_gemm_mode = _normalize_qgm(quantized_gemm_mode)
    except ValueError as exc:
        raise ValidationError("Invalid quantized_gemm_mode value", detail=str(exc))
    # Attention backend (attention_type): validated against the attention
    # registry's own vocabulary, also before start_generation. An unknown value
    # used to run native and be RECORDED as the unknown string.
    attention_type = _validated_attention_type(
        attention_type, GENERATION_DEFAULTS["attention_type"])
    if loop_decode not in ("full", "cheap", "none"):
        raise ValidationError(
            "Invalid loop_decode value",
            detail=f"loop_decode must be 'full', 'cheap', or 'none', got {loop_decode!r}",
        )
    if loop_decode == "none":
        raise ValidationError(
            "loop_decode='none' is not supported for outpaint",
            detail="Outpaint's final pixel-space preservation paste requires a decoded image. "
                   "Use loop_decode='cheap' for lower-cost intermediate loop steps instead.",
        )
    if input_latent_id:
        raise ValidationError(
            "input_latent_id (loop latent passthrough) is not supported for outpaint",
            detail="Outpaint requires a real source image to build the canvas + preserved rect. "
                   "Use loop_decode='cheap' for lower-cost intermediate loop steps instead.",
        )
    _override_plan = plan_overrides(pipeline_manager, vae_path, text_encoder_path)
    _gen_id = start_generation("outpaint")
    try:
        # Reset cancellation flag before starting new generation
        pipeline_manager.reset_cancel_flag()

        # Load input image (outpaint builds its own canvas + mask -- no
        # separate mask upload, unlike /generate/inpaint).
        image_data = await image.read()
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Validate placement geometry UP FRONT (before the GPU slot is
        # acquired below) so degenerate placements (empty crop, rect <8px,
        # rect fully covering the canvas) surface as a 400 ValidationError
        # instead of falling through to the broad except -> 500
        # GenerationError after GPU resources are already reserved.
        # align=16 matches PipelineManager.generate_outpaint (see its
        # docstring); this call is idempotent, so generate_outpaint's own
        # (identical) call re-runs it harmlessly.
        from core.inference.outpaint_utils import validate_and_snap_placement
        try:
            validate_and_snap_placement(
                {
                    "canvas_width": canvas_width,
                    "canvas_height": canvas_height,
                    "place_x": place_x,
                    "place_y": place_y,
                    "place_width": place_width,
                    "place_height": place_height,
                    "input_crop_x": input_crop_x,
                    "input_crop_y": input_crop_y,
                    "input_crop_w": input_crop_w,
                    "input_crop_h": input_crop_h,
                },
                init_image.size,
                align=16,
            )
        except ValueError as e:
            raise ValidationError("Invalid outpaint placement geometry", detail=str(e))

        # Parse LoRA configs
        import json
        lora_configs = json.loads(loras) if loras else []

        # Parse ControlNet configs
        controlnet_configs = json.loads(controlnets) if controlnets else []
        controlnet_images, style_transfer, style_transfers, style_combine_mode = process_controlnet_configs(
            controlnet_configs,
            generation_type="outpaint"
        )

        # Parse TIPO config
        tipo_config_dict = json.loads(tipo_config) if tipo_config else {}

        # TIPO prompt upsampling (if enabled)
        original_prompt = prompt
        if use_tipo:
            print(f"[TIPO] Upsampling prompt with TIPO...")
            try:
                # Load TIPO model if needed
                model_name = tipo_config_dict.get("model_name", "KBlueLeaf/TIPO-500M")
                if not tipo_manager.loaded or tipo_manager.model_name != model_name:
                    tipo_manager.load_model(model_name)

                # Generate upsampled prompt
                upsampled_prompt = tipo_manager.generate_prompt(
                    input_prompt=prompt,
                    tag_length=tipo_config_dict.get("tag_length", "long"),
                    nl_length=tipo_config_dict.get("nl_length", "long"),
                    temperature=tipo_config_dict.get("temperature", 1.0),
                    top_p=tipo_config_dict.get("top_p", 0.95),
                    top_k=tipo_config_dict.get("top_k", 50),
                    max_new_tokens=tipo_config_dict.get("max_new_tokens", 256),
                    category_order=tipo_config_dict.get("category_order", []),
                    enabled_categories=tipo_config_dict.get("enabled_categories", {}),
                    treat_as_nl=tipo_config_dict.get("treat_as_nl", False)
                )

                # If result is dict (tipo-kgen mode), format it to string
                if isinstance(upsampled_prompt, dict):
                    category_order = tipo_config_dict.get("category_order", [])
                    enabled_categories = tipo_config_dict.get("enabled_categories", {})

                    # If no category order specified, use default
                    if not category_order:
                        category_order = ["special", "quality", "rating", "artist", "copyright", "characters", "meta", "general"]

                    # If no enabled categories specified, enable all by default
                    if not enabled_categories:
                        enabled_categories = {cat: True for cat in category_order}
                        enabled_categories["meta"] = False  # Meta disabled by default

                    prompt = tipo_manager.format_kgen_result(
                        upsampled_prompt,
                        category_order,
                        enabled_categories
                    )
                else:
                    prompt = upsampled_prompt

                print(f"[TIPO] Original prompt: {original_prompt[:100]}...")
                print(f"[TIPO] Upsampled prompt: {prompt[:100]}...")

                # Unload TIPO model to free VRAM
                tipo_manager.unload_model()

            except Exception as e:
                print(f"[TIPO] Error during upsampling: {e}")
                print(f"[TIPO] Using original prompt")
                # Continue with original prompt on error

        # Process reference images (FLUX.2 Image Edit / Vision Encoder)
        ref_image_list = []
        if ref_images:
            for ref_img_file in ref_images:
                img_bytes = await ref_img_file.read()
                ref_image_list.append(Image.open(io.BytesIO(img_bytes)))
            print(f"[FLUX.2 Image Edit] Loaded {len(ref_image_list)} reference image(s)")

        # Load Vision Encoder if requested (non-FLUX.2 only)
        is_flux2 = pipeline_manager.current_model_info and pipeline_manager.current_model_info.get("type") == "flux2"
        if vision_encoder_path and not is_flux2:
            pipeline_manager.load_vision_encoder(vision_encoder_path)

        # Apply (or restore) the planned VAE/TE overrides on the loaded model.
        _override_meta = apply_overrides(
            pipeline_manager, _override_plan,
            pid_sr_output=pid_sr_output, pid_use_gemma=pid_use_gemma, pid_low_vram=pid_low_vram,
            pid_tile_native=pid_tile_native, pid_tile_overlap_ratio=pid_tile_overlap_ratio,
            pid_fast_large_decode=pid_fast_large_decode, prompt=prompt,
        )

        # Generate image
        params = {
            "prompt": prompt,
            "vae_path": vae_path,
            "text_encoder_path": text_encoder_path,
            "pid_sr_output": pid_sr_output,
            "pid_use_gemma": pid_use_gemma,
            "pid_low_vram": pid_low_vram,
            "pid_tile_native": pid_tile_native,
            "pid_tile_overlap_ratio": pid_tile_overlap_ratio,
            "pid_fast_large_decode": pid_fast_large_decode,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_strength": denoising_strength,
            "img2img_fix_steps": img2img_fix_steps,
            "sampler": sampler,
            "schedule_type": schedule_type,
            "seed": seed,
            "ancestral_seed": ancestral_seed,
            # Placement (consumed by core.inference.outpaint_utils via
            # PipelineManager.generate_outpaint; "width"/"height" are set
            # internally from the resolved canvas size, not from these).
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "place_x": place_x,
            "place_y": place_y,
            "place_width": place_width,
            "place_height": place_height,
            "input_crop_x": input_crop_x,
            "input_crop_y": input_crop_y,
            "input_crop_w": input_crop_w,
            "input_crop_h": input_crop_h,
            "outpaint_fill_mode": outpaint_fill_mode,
            "mask_blur": mask_blur,
            "inpaint_full_res": inpaint_full_res,
            "inpaint_full_res_padding": inpaint_full_res_padding,
            "inpaint_fill_mode": inpaint_fill_mode,
            "inpaint_fill_strength": inpaint_fill_strength,
            "inpaint_blur_strength": inpaint_blur_strength,
            "region_prompt": region_prompt,
            "region_negative_prompt": region_negative_prompt,
            "region_prompt_strength": region_prompt_strength,
            "region_prompt_method": region_prompt_method,
            "region_mask_feather": region_mask_feather,
            "seam_structure_strength": seam_structure_strength,
            "seam_structure_depth": seam_structure_depth,
            "seam_structure_end": seam_structure_end,
            "seam_structure_saliency": seam_structure_saliency,
            "seam_structure_max_area": seam_structure_max_area,
            "boundary_relax_strength": boundary_relax_strength,
            "boundary_relax_width": boundary_relax_width,
            "boundary_relax_noise": boundary_relax_noise,
            "boundary_relax_full_until": boundary_relax_full_until,
            "boundary_relax_end": boundary_relax_end,
            "boundary_relax_paste": boundary_relax_paste,
            "outpaint_seam_fix": outpaint_seam_fix,
            "outpaint_seam_membrane": outpaint_seam_membrane,
            "outpaint_seam_membrane_band": outpaint_seam_membrane_band,
            "outpaint_seam_tone_strength": outpaint_seam_tone_strength,
            "outpaint_seam_tone_band": outpaint_seam_tone_band,
            "outpaint_seam_offset_prop": outpaint_seam_offset_prop,
            "outpaint_paste_feather_px": outpaint_paste_feather_px,
            "outpaint_preserve_mode": outpaint_preserve_mode,
            "outpaint_preview_unpinned_x0": outpaint_preview_unpinned_x0,
            "outpaint_boundary_color_strength": outpaint_boundary_color_strength,
            "outpaint_resample_count": outpaint_resample_count,
            "outpaint_jump_length": outpaint_jump_length,
            "outpaint_reference_strength": outpaint_reference_strength,
            "outpaint_commit_strength": outpaint_commit_strength,
            "outpaint_commit_near": outpaint_commit_near,
            "outpaint_commit_far": outpaint_commit_far,
            "outpaint_commit_distance": outpaint_commit_distance,
            "outpaint_controlnet_enable": outpaint_controlnet_enable,
            "outpaint_controlnet_mode": outpaint_controlnet_mode,
            "outpaint_controlnet_model": outpaint_controlnet_model,
            "outpaint_controlnet_detector": outpaint_controlnet_detector,
            "outpaint_controlnet_scale": outpaint_controlnet_scale,
            "outpaint_controlnet_guidance_start": outpaint_controlnet_guidance_start,
            "outpaint_controlnet_guidance_end": outpaint_controlnet_guidance_end,
            "outpaint_controlnet_depth": outpaint_controlnet_depth,
            "outpaint_controlnet_taper": outpaint_controlnet_taper,
            "outpaint_controlnet_edge_feather_px": outpaint_controlnet_edge_feather_px,
            "outpaint_controlnet_corner_radius_px": outpaint_controlnet_corner_radius_px,
            "outpaint_controlnet_corner_gate_radius_px": outpaint_controlnet_corner_gate_radius_px,
            "outpaint_controlnet_corner_gate_min": outpaint_controlnet_corner_gate_min,
            "outpaint_pin_corner_relax_radius_px": outpaint_pin_corner_relax_radius_px,
            "outpaint_pin_corner_relax_min": outpaint_pin_corner_relax_min,
            "loras": lora_configs,  # FLUX.2 needs this in params
            "controlnet_images": controlnet_images,
            "style_transfer": style_transfer,
            "style_transfers": style_transfers,
            "style_combine_mode": style_combine_mode,
            "developer_mode": developer_mode,
            "cfg_schedule_type": cfg_schedule_type,
            "cfg_schedule_min": cfg_schedule_min,
            "cfg_schedule_max": cfg_schedule_max,
            "cfg_schedule_power": cfg_schedule_power,
            "cfg_rescale_snr_alpha": cfg_rescale_snr_alpha,
            "dynamic_threshold_percentile": dynamic_threshold_percentile,
            "dynamic_threshold_mimic_scale": dynamic_threshold_mimic_scale,
            "nag_enable": nag_enable,
            "nag_scale": nag_scale,
            "nag_tau": nag_tau,
            "nag_alpha": nag_alpha,
            "nag_sigma_end": nag_sigma_end,
            "nag_negative_prompt": nag_negative_prompt,
            "attention_type": attention_type,
            "attention_impl": attention_impl,
            "unet_quantization": unet_quantization,
            "quantized_gemm_mode": quantized_gemm_mode,
            "original_size_w": original_size_w,
            "original_size_h": original_size_h,
            "original_size_scale": original_size_scale,
            "text_encoder_quantization": text_encoder_quantization,
            "cpu_text_encoding": cpu_text_encoding,
            "use_torch_compile": use_torch_compile,
            "keep_models_hot": keep_models_hot,
            "vae_tiling": vae_tiling,
            "vae_tile_threshold": vae_tile_threshold,
            "vae_tile_mode": vae_tile_mode,
            "vae_tile_global_norm": vae_tile_global_norm,
            "color_flatten_strength": color_flatten_strength,
            "flatten_in_loop": flatten_in_loop,
            "flatten_in_loop_last_steps": flatten_in_loop_last_steps,
            "flatten_in_loop_min_region": flatten_in_loop_min_region,
            "vae_drift_correction": vae_drift_correction,
            "spectrum_enable": spectrum_enable,
            "fbcache_enable": fbcache_enable,
            "fbcache_threshold": fbcache_threshold,
            "fbcache_warmup_steps": fbcache_warmup_steps,
            "fbcache_cache_branch": fbcache_cache_branch,
            "spectrum_w": spectrum_w,
            "spectrum_w_decay": spectrum_w_decay,
            "spectrum_delta_cap": spectrum_delta_cap,
            "spectrum_m": spectrum_m,
            "spectrum_lam": spectrum_lam,
            "spectrum_warmup_steps": spectrum_warmup_steps,
            "spectrum_window_size": spectrum_window_size,
            "spectrum_flex_window": spectrum_flex_window,
            "spectrum_tail": spectrum_tail,
            "spectrum_feature_mode": spectrum_feature_mode,
            "spectrum_cache_branch": spectrum_cache_branch,
            "spectrum_max_cache": spectrum_max_cache,
            "enable_block_swap": enable_block_swap,
            "blocks_to_swap": blocks_to_swap,
            "use_pinned_memory": use_pinned_memory,
            "block_swap_h2d_only": block_swap_h2d_only,
            "block_swap_ring_size": block_swap_ring_size,
            "preview_decoder": preview_decoder,
            "ref_images": ref_image_list,  # FLUX.2 Image Edit reference images
            "loop_decode": loop_decode,
            "skip_gallery": skip_gallery,
        }
        params.update(_override_meta)
        print(f"outpaint generation params: {sanitize_params_for_logging(params)}")

        # inpaint_full_res is accepted for API compatibility but not implemented
        # (shared with /generate/inpaint's underlying generate_inpaint call).
        if inpaint_full_res or inpaint_full_res_padding != 32:
            from api.generation_status import add_warning as _add_warning
            _add_warning(
                "inpaint_full_res is accepted but not implemented; it has no effect",
                code="not_implemented",
            )

        # Set prompt chunking settings
        set_prompt_chunking_settings(
            pipeline_manager,
            prompt_chunking_mode,
            max_prompt_chunks
        )

        # Load LoRAs if specified (outpaint delegates to the shared inpaint
        # pipeline underneath).
        pipeline_manager.inpaint_pipeline, has_step_range_loras = load_loras_for_generation(
            lora_manager,
            pipeline_manager.inpaint_pipeline,
            lora_configs,
            "outpaint"
        )

        # Detect if SDXL
        is_sdxl = pipeline_manager.inpaint_pipeline is not None and \
                  "XL" in pipeline_manager.inpaint_pipeline.__class__.__name__
        is_zimage = pipeline_manager.current_model_info and \
                    pipeline_manager.current_model_info.get("type") == "zimage"
        is_deus = pipeline_manager.current_model_info and \
                  pipeline_manager.current_model_info.get("type") == "deus"
        is_flux2 = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "flux2"
        # Z-Image with SDXL VAE (4ch) needs TAESD-XL instead of TAEF1
        is_zimage_sdxl_vae = is_zimage and \
                             pipeline_manager.current_model_info.get("vae_type") == "sdxl"
        is_anima = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "anima"
        is_lens = pipeline_manager.current_model_info and \
                  pipeline_manager.current_model_info.get("type") == "lens"
        # Ideogram 4 shares AutoencoderKLFlux2's 128-ch packed latent with Lens.
        is_ideogram4 = pipeline_manager.current_model_info and \
                       pipeline_manager.current_model_info.get("type") == "ideogram4"
        is_minit2i = pipeline_manager.current_model_info and \
                     pipeline_manager.current_model_info.get("type") == "minit2i"
        minit2i_vae_type = (pipeline_manager.minit2i_components or {}).get("vae_type", "none") if is_minit2i else "none"
        is_krea2 = pipeline_manager.current_model_info and \
                   pipeline_manager.current_model_info.get("type") == "krea2"

        # Warn about parameters the loaded architecture silently ignores
        _current_arch = pipeline_manager.current_model_info.get("type") if pipeline_manager.current_model_info else None
        check_arch_capabilities(params, _current_arch)

        # Progress callback to send updates via WebSocket. Uses the CANVAS
        # size (the actual output dimensions), not the raw input image size.
        progress_callback = create_progress_callback_factory(
            taesd_manager,
            manager,
            is_sdxl,
            is_zimage,
            is_deus,
            is_zimage_sdxl_vae,
            is_flux2,
            is_anima,
            is_lens=is_lens,
            is_ideogram4=is_ideogram4,
            is_minit2i=is_minit2i,
            is_krea2=is_krea2,
            img2img_fix_steps=img2img_fix_steps,
            steps=steps,
            image_width=canvas_width,
            image_height=canvas_height,
            # For flow-matching DiTs (Anima / Z-Image / FLUX.2 / Lens), default to
            # the pred_x0 preview: x_t is mostly noise mid-denoising, while
            # pred_x0 = x_t - σ·v shows the model's current clean-image
            # estimate from the very first steps. Any explicit user override
            # via the API still wins.
            preview_predicted_x0=(preview_predicted_x0 or is_anima or is_zimage or is_flux2 or is_lens or is_ideogram4 or is_minit2i or is_krea2),
            preview_enabled=params.get("preview_enabled", True),
            preview_interval=params.get("preview_interval", 4),
            preview_decoder=params.get("preview_decoder", "matrix")
        )

        # Create step callback for LoRA step range if needed
        step_callback = None
        if has_step_range_loras:
            # Calculate actual steps based on denoising strength
            actual_steps = int(steps * denoising_strength)
            step_callback = create_lora_step_callback(
                lora_manager,
                pipeline_manager.inpaint_pipeline,
                actual_steps
            )

        # Run generation in thread pool to avoid blocking event loop.
        # gpu_coordinator slot pauses any active tagger training first.
        from core.gpu_coordinator import gpu_coordinator
        from core.inference.generation_timing import generation_timer
        loop = asyncio.get_event_loop()
        _peak_gb = _estimate_gen_peak_gb(canvas_width, canvas_height, 1,
                                         pipeline_manager.current_pipeline_kind)
        generation_timer.reset()
        _gen_start = time.perf_counter()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=_peak_gb, timeout=60.0):
            # GEMM flags are process-wide; keep selection and probing in this slot.
            from api.quantized_gemm import apply_quantized_gemm_mode
            apply_quantized_gemm_mode(quantized_gemm_mode)
            result_image, actual_seed, actual_ancestral_seed = await _run_generation_in_executor(
                loop, executor,
                lambda: pipeline_manager.generate_outpaint(params, init_image, progress_callback=progress_callback, step_callback=step_callback)
            )
            fp8_gemm = extract_fp8_gemm_info(pipeline_manager)
        # Record total wall time + any phase breakdown the pipeline populated.
        apply_generation_timings(params, time.perf_counter() - _gen_start)

        # Update params with actual seeds
        params["seed"] = actual_seed
        params["ancestral_seed"] = actual_ancestral_seed

        # Add Vision Encoder info to params for PNG metadata and DB storage.
        # Only record VE info when THIS generation actually used reference images
        # (the VE stays loaded "sticky" across generations).
        if ref_image_list:
            ve_name, ve_hash = extract_vision_encoder_info(pipeline_manager)
            if ve_name:
                params["vision_encoder_name"] = ve_name
            if ve_hash:
                params["vision_encoder_hash"] = ve_hash

        # Add VAE identity to params. The VAE always participates in decode, so this
        # is recorded for every generation where it can be determined.
        vae_name, vae_hash = extract_vae_info(pipeline_manager)
        if vae_name:
            params["vae_name"] = vae_name
        if vae_hash:
            params["vae_hash"] = vae_hash

        # FP8 GEMM path. Recorded only for checkpoints that carry weight-only FP8
        # Linear weights, and it records the RESOLVED path rather than the flag —
        # the W8A8 path can be enabled while the device probe rejects every
        # scaling mode, and the two paths are numerically different.
        if fp8_gemm:
            params["fp8_gemm"] = fp8_gemm
        # Report an explicit "w8a8" request that resolved to the dequant path
        # (probe rejected it, or the checkpoint has no quantized layers).
        from api.quantized_gemm import report_quantized_gemm_outcome
        report_quantized_gemm_outcome(quantized_gemm_mode, fp8_gemm, _current_arch)
        # Attention backend that ACTUALLY ran (observed by the conduit), so the
        # PNG metadata and the gallery row cannot name a backend that did not.
        record_attention_backend(params, _gen_id)

        # Save image with metadata (include model info). params["width"]/["height"]
        # were overwritten by generate_outpaint to the resolved canvas size.
        filename = save_image_with_metadata(
            result_image,
            params,
            "outpaint",
            model_info=pipeline_manager.current_model_info,
            generation_id=_gen_id
        )

        # skip_gallery: the file is saved (so a generation loop can still chain
        # to the next step via its path) but no thumbnail/DB record is created.
        if skip_gallery:
            complete_generation({"filename": filename, "seed": actual_seed}, generation_id=_gen_id)
            return {
                "success": True,
                "filename": filename,
                "image_path": f"/outputs/{filename}",
                "actual_seed": actual_seed,
                "warnings": get_warnings(_gen_id),
            }

        image_path = os.path.join(settings.outputs_dir, filename)
        create_thumbnail(image_path)

        # Calculate metadata. No mask_image here (outpaint builds its own
        # mask server-side; there is no user-uploaded mask to record).
        metadata = calculate_generation_metadata(
            result_image,
            lora_configs,
            extract_lora_names,
            calculate_image_hash,
            source_image=init_image,
        )

        # Remove image objects from params before saving to DB and calculate ControlNet hashes
        params_for_db = prepare_params_for_db(params, calculate_image_hash)
        _effective_warnings = get_warnings(_gen_id)
        if _effective_warnings:
            params_for_db["effective_warnings"] = _effective_warnings

        # Extract model name and hash from current_model_info
        model_name, model_hash = extract_model_info(pipeline_manager)

        # Save to database
        db_image = create_db_image_record(
            GeneratedImage,
            filename=filename,
            params=params_for_db,
            actual_seed=actual_seed,
            generation_type="outpaint",
            image_hash=metadata["image_hash"],
            lora_names=metadata["lora_names"],
            model_name=model_name,
            model_hash=model_hash,
            result_image=result_image,
            source_image_hash=metadata.get("source_image_hash"),
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        complete_generation({"image_id": db_image.id, "filename": filename, "seed": actual_seed}, generation_id=_gen_id)
        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed, "warnings": get_warnings(_gen_id)}

    except GenerationError as e:
        # Re-raise custom errors as-is
        fail_generation(str(e), generation_id=_gen_id)
        raise
    except Exception as e:
        # Wrap unexpected errors in GenerationError
        import traceback
        error_detail = traceback.format_exc()
        fail_generation(str(e), generation_id=_gen_id)
        raise GenerationError(
            "Outpaint generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )
    finally:
        # Unload LoRAs after generation
        if lora_configs and pipeline_manager.inpaint_pipeline:
            pipeline_manager.inpaint_pipeline = lora_manager.unload_loras(pipeline_manager.inpaint_pipeline)


@router.get("/images")
async def get_images(
    skip: int = 0,
    limit: int = 50,
    search: Optional[str] = None,
    generation_types: Optional[str] = None,  # Comma-separated: txt2img,img2img,inpaint
    date_from: Optional[str] = None,  # ISO format date
    date_to: Optional[str] = None,  # ISO format date
    width_min: Optional[int] = None,
    width_max: Optional[int] = None,
    height_min: Optional[int] = None,
    height_max: Optional[int] = None,
    db: Session = Depends(get_gallery_db)
):
    """Get list of generated images with filtering"""
    query = db.query(GeneratedImage)

    # Search the compact text fields available on the gallery row.
    if search:
        query = query.filter(or_(
            GeneratedImage.prompt.contains(search),
            GeneratedImage.negative_prompt.contains(search),
            GeneratedImage.filename.contains(search),
        ))

    # Filter by generation type
    if generation_types:
        types = [t.strip() for t in generation_types.split(',')]
        query = query.filter(GeneratedImage.generation_type.in_(types))

    # Filter by date range
    if date_from:
        from datetime import datetime
        date_from_dt = datetime.fromisoformat(date_from)
        query = query.filter(GeneratedImage.created_at >= date_from_dt)

    if date_to:
        from datetime import datetime
        date_to_dt = datetime.fromisoformat(date_to)
        query = query.filter(GeneratedImage.created_at <= date_to_dt)

    # Filter by width range
    if width_min is not None:
        query = query.filter(GeneratedImage.width >= width_min)
    if width_max is not None:
        query = query.filter(GeneratedImage.width <= width_max)

    # Filter by height range
    if height_min is not None:
        query = query.filter(GeneratedImage.height >= height_min)
    if height_max is not None:
        query = query.filter(GeneratedImage.height <= height_max)

    # Get total count for pagination
    total_count = query.count()

    # Order by created_at descending and apply pagination
    images = query.order_by(GeneratedImage.created_at.desc()).offset(skip).limit(limit).all()

    # Summary DTO: the grid only renders/filters on a handful of fields (see
    # GeneratedImage.to_summary_dict docstring) -- the full parameters JSON
    # and mask_data are detail-only and can bloat a single row to several MB
    # (embedded ControlNet/style-reference base64). Detail view fetches the
    # full record via GET /images/{id} (below) on open.
    return {
        "images": [img.to_summary_dict() for img in images],
        "total": total_count,
        "skip": skip,
        "limit": limit
    }

@router.get("/images/{image_id}")
async def get_image(image_id: int, db: Session = Depends(get_gallery_db)):
    """Get single image details"""
    image = db.query(GeneratedImage).filter(GeneratedImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image.to_dict()

@router.get("/images/by-hash/{hash}")
async def get_image_by_hash(hash: str, db: Session = Depends(get_gallery_db)):
    """Resolve a gallery row by its `image_hash` (the click target behind a
    `source_image_hash` / `source_audio_hash` / ControlNet reference-hash
    link in the gallery UI).

    `image_hash` is NOT unique -- the same seed can be re-run, or the same
    uploaded file hashed across two different requests -- so this returns
    the OLDEST matching row (`created_at` ascending, the most plausible
    original source) plus `match_count`, the total number of rows that
    matched. Never returns a list.
    """
    # Two narrow queries rather than one `.all()`: `parameters` (and, for an
    # inpaint row, `mask_data`) can be several MB per row (embedded base64
    # references), so fetching every matching row just to report a count and
    # discard the rest does not scale with how many times a hash repeats.
    query = db.query(GeneratedImage).filter(GeneratedImage.image_hash == hash)
    match_count = query.count()
    if match_count == 0:
        raise NotFoundError(
            "No gallery row matches this image_hash",
            detail=f"hash: {hash}",
        )
    oldest = query.order_by(GeneratedImage.created_at.asc()).first()
    result = oldest.to_dict()
    result["match_count"] = match_count
    return result

@router.get("/images/{image_id}/preview")
async def get_image_preview(
    image_id: int,
    request: Request,
    w: int = PREVIEW_DEFAULT_WIDTH,
    db: Session = Depends(get_gallery_db)
):
    """Sized WebP preview of a gallery image for the detail/lightbox view
    (gallery perf redesign Phase 2, Option C). Generated lazily on first
    request and cached on disk (settings.previews_dir); subsequent requests
    for the same (image, width bucket) serve the cached file. Only applies
    to image generations -- video/audio rows use the poster thumbnail +
    range-streamed /outputs original instead (see get_or_create_preview /
    range_file_response in api/media_utils.py).
    """
    image = db.query(GeneratedImage).filter(GeneratedImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    is_video = bool(image.parameters.get("is_video")) if image.parameters else False
    is_audio = bool(image.parameters.get("is_audio")) if image.parameters else False
    if is_video or is_audio or re.search(r"\.(mp4|webm|flac|wav|mp3|ogg)$", image.filename, re.IGNORECASE):
        raise HTTPException(status_code=404, detail="No preview for video/audio generations")

    source_path = os.path.join(settings.outputs_dir, image.filename)
    # Mirror main.py's /outputs traversal guard: image.filename is always a
    # flat, server-generated name today, but this READ path re-serves
    # whatever it opens as WebP, so a crafted `..` filename must not be able
    # to walk it outside outputs_dir.
    abs_src = os.path.realpath(source_path)
    _out_real = os.path.realpath(settings.outputs_dir)
    if not (abs_src == _out_real or abs_src.startswith(_out_real + os.sep)):
        raise HTTPException(status_code=404, detail="Image not found")
    if not os.path.isfile(source_path):
        raise HTTPException(status_code=404, detail="Source file not found")

    cache_path = await get_or_create_preview(source_path, image.filename, w)
    return await range_file_response(
        request,
        cache_path,
        media_type="image/webp",
        cache_control="public, max-age=604800",
    )

def _is_safe_output_name(name: Optional[str]) -> bool:
    """A DB-sourced filename must resolve inside outputs_dir/thumbnails_dir
    once joined -- refuse anything with a separator or `..` component
    (`os.path.basename(name) != name`) rather than let it escape via
    `os.path.join`."""
    return bool(name) and os.path.basename(name) == name


def _generated_image_file_paths(image: "GeneratedImage") -> Dict[str, str]:
    """Every on-disk artefact a gallery row can own, keyed by a short label.

    Thumbnails are keyed by base name + `.png`/`.webp` regardless of the
    original extension (see `create_thumbnail`), not `<filename>` itself.
    `media` is inserted LAST: callers process this dict in order, so a
    failure partway through still leaves the row pointing at a file that
    still exists on a retry.
    """
    filename = image.filename
    if not _is_safe_output_name(filename):
        return {}

    base_name = os.path.splitext(filename)[0]
    media_path = os.path.join(settings.outputs_dir, filename)
    poster_path = os.path.join(settings.outputs_dir, f"{base_name}.png")

    paths = {
        "sidecar": os.path.join(settings.outputs_dir, f"{base_name}.json"),
        "thumbnail_png": os.path.join(settings.thumbnails_dir, f"{base_name}.png"),
        "thumbnail_webp": os.path.join(settings.thumbnails_dir, f"{base_name}.webp"),
    }
    if poster_path != media_path:  # collapses onto media_path for plain images
        paths["poster"] = poster_path

    preview_filename = (image.parameters or {}).get("preview_filename")
    if _is_safe_output_name(preview_filename):
        paths["preview_proxy"] = os.path.join(settings.outputs_dir, preview_filename)

    paths["media"] = media_path
    return paths


@router.delete("/images/{image_id}")
async def delete_image(
    image_id: int,
    delete_files: bool = True,
    db: Session = Depends(get_gallery_db),
):
    """Delete a gallery row.

    `delete_files=true` (default): removes the DB row AND every artefact it
    owns (media, sidecar JSON, poster/waveform PNG, both thumbnail variants,
    and the lossless proxy when present). Files are removed in an order that
    leaves `media` for last, so if a later file fails the row (left intact,
    see the 500 response) still points at a file that still exists on retry.

    `delete_files=false`: removes only the DB row; every file on disk is
    left untouched (they become invisible to the gallery but stay on disk).
    """
    image = db.query(GeneratedImage).filter(GeneratedImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    if not delete_files:
        db.delete(image)
        db.commit()
        return {"success": True, "delete_files": False, "deleted_files": []}

    file_paths = _generated_image_file_paths(image)
    deleted: list = []
    errors: list = []
    for label, path in file_paths.items():
        if not os.path.exists(path):
            continue
        try:
            os.remove(path)
            deleted.append(label)
        except OSError as e:
            errors.append(f"{label} ({os.path.basename(path)}): {e}")

    if errors:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Deleted {len(deleted)} file(s) but failed on {len(errors)}: "
                f"{'; '.join(errors)}. The gallery record was NOT deleted; "
                f"retry once the remaining files are removable."
            ),
        )

    db.delete(image)
    db.commit()

    return {"success": True, "delete_files": True, "deleted_files": deleted}


def _expand_minimax_h3_tree(tree_path: str, name_prefix: str, source_dir: str) -> list:
    """List the selectable DiT partitions (fl2va/ref2va) under an H3 tree.

    Shared by both scan shapes in `get_models`: the tree sitting one level
    under `models_dir` (`item_path`) and `models_dir` pointing directly at the
    tree root, which is how every other arch's own directory is added there
    (see `M:\\model\\{sdxl,krea2,anima}` -- one models_dir per arch root, not
    per parent). Both must resolve to the same two entries.
    """
    from core.models.minimax_h3.loader import (
        detect_minimax_h3_layout, is_minimax_h3_safetensors,
    )
    dit_dir = os.path.join(tree_path, "diffusion_models")
    if not os.path.isdir(dit_dir):
        return []
    dits = sorted(
        os.path.join(dit_dir, f) for f in os.listdir(dit_dir)
        if f.endswith(".safetensors")
        and is_minimax_h3_safetensors(os.path.join(dit_dir, f))
    )
    result = []
    for dit in dits:
        layout = detect_minimax_h3_layout(dit) or {}
        result.append({
            "name": f"{name_prefix}/{os.path.basename(dit)[:-len('.safetensors')]}",
            "path": dit,
            "type": "safetensors",
            "source_type": "safetensors",
            "size_gb": round(os.path.getsize(dit) / (1024 ** 3), 2),
            "source_dir": source_dir,
            "architecture": "minimax_h3",
            # Which workflows this entry can serve, so the frontend does not
            # have to parse a filename.
            "variant": layout.get("variant"),
        })
    return result


@router.get("/models")
async def get_models(db: Session = Depends(get_gallery_db), force_rescan: bool = False):
    """
    Get list of available models from default and user-configured directories.

    Uses cache to avoid expensive scanning on every API call.

    Args:
        force_rescan: Force re-scanning (ignores cache)

    Returns:
        Dictionary with "models" key containing list of model info
    """
    import time
    global _models_cache, _models_cache_timestamp

    # Return cached result if available and not forcing rescan
    if not force_rescan and _models_cache is not None:
        return _models_cache

    print(f"[Models] Scanning model directories...")
    scan_start = time.time()

    from core.model_loader import ModelLoader

    models = []

    # Get user-configured directories
    settings_record = db.query(UserSettings).first()
    additional_model_dirs = settings_record.model_dirs if settings_record else []

    # Combine default directory with user directories
    all_dirs = [settings.models_dir] + additional_model_dirs

    for models_dir in all_dirs:
        if not os.path.exists(models_dir):
            print(f"[Models] Directory does not exist: {models_dir}")
            continue

        print(f"[Models] Scanning directory: {models_dir}")

        # MiniMax-H3: the configured directory can itself BE the H3 tree
        # root (`diffusion_models/` sits directly under it), matching how
        # every other arch is added here -- one models_dir per arch's own
        # root, not per parent. In that shape, os.listdir(models_dir) would
        # otherwise enumerate the tree's own components (official/,
        # text_encoders/, vae/, ...) as if they were separate candidate
        # models, and `official/model_index.json` (config-only, not a
        # loadable checkpoint) wins architecture detection before the DiT
        # files are ever reached -- leaving only "official" selectable.
        # Expand the two DiT partitions here and skip the per-child scan
        # for this directory entirely.
        _h3_dits = _expand_minimax_h3_tree(
            models_dir, os.path.basename(os.path.normpath(models_dir)), models_dir
        )
        if _h3_dits:
            models.extend(_h3_dits)
            continue

        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)

            # Hide individual shards of a sharded save; the sibling
            # <stem>.safetensors.index.json is the single selectable entry.
            if re.search(r"-\d{5}-of-\d{5}\.safetensors$", item):
                continue

            # Detect model architecture (sd15, sdxl, zimage)
            architecture = ModelLoader.detect_model_type(item_path)

            if os.path.isdir(item_path):
                # MiniT2I: a repo root / container holds multiple variant dirs
                # (B/16, L/16) which are separate models — expand each into its own
                # selectable entry instead of listing the container once.
                if architecture == "minit2i":
                    from core.models.minit2i.minit2i_loader import (
                        find_minit2i_variant_dirs, _is_minit2i_variant_dir,
                    )
                    if _is_minit2i_variant_dir(item_path):
                        variant_dirs = [item_path]
                    else:
                        variant_dirs = find_minit2i_variant_dirs(item_path)
                    for vdir in variant_dirs:
                        if vdir == item_path:
                            vname = item
                        else:
                            rel = os.path.relpath(vdir, item_path).replace("\\", "/")
                            vname = f"{item}/{rel}"
                        # Read vae_type from the transformer config so the frontend can
                        # tell pixel ("none") from latent (sdxl/flux1: Full-FT only).
                        vae_type = "none"
                        try:
                            with open(os.path.join(vdir, "transformer", "config.json"), "r", encoding="utf-8") as _cf:
                                vae_type = json.load(_cf).get("vae_type", "none") or "none"
                        except Exception:
                            pass
                        models.append({
                            "name": vname,
                            "path": vdir,
                            "type": "diffusers",
                            "source_type": "diffusers",
                            "source_dir": models_dir,
                            "architecture": "minit2i",
                            "vae_type": vae_type,
                        })
                    continue

                # MiniMax-H3: the tree holds TWO transformer partitions that
                # share every other component -- `fl2va` (txt2vid / img2vid /
                # outpaint) and `ref2va` (/generate/ref2vid) -- and the workflow
                # is selected by WHICH ONE IS LOADED. They are otherwise
                # indistinguishable (same config, same byte size, no
                # distinguishing key), so the DiT file is the variant, and each
                # one is listed as its own selectable model rather than the tree
                # being listed once and silently resolving to the first file.
                # Same shape as the MiniT2I expansion above.
                if architecture == "minimax_h3":
                    _h3_dits = _expand_minimax_h3_tree(item_path, item, models_dir)
                    if _h3_dits:
                        models.extend(_h3_dits)
                        continue

                # Allow Anima split-files layouts even when there's no model_index.json
                is_valid = ModelLoader.is_valid_diffusers_directory(item_path)
                if not is_valid and architecture not in ("anima", "lens", "krea2"):
                    continue
                models.append({
                    "name": item,
                    "path": item_path,
                    "type": "diffusers",
                    "source_type": "diffusers",
                    "source_dir": models_dir,
                    "architecture": architecture
                })
            elif item.endswith('.safetensors.index.json'):
                # Sharded single-file save: one entry, size = sum of its shards.
                if architecture == "vision_encoder":
                    continue
                stem = item[: -len('.safetensors.index.json')]
                total_bytes = 0
                try:
                    with open(item_path, encoding='utf-8') as _idxf:
                        weight_map = json.load(_idxf).get('weight_map', {}) or {}
                    for shard in set(weight_map.values()):
                        shard_path = os.path.join(models_dir, shard)
                        if os.path.exists(shard_path):
                            total_bytes += os.path.getsize(shard_path)
                except Exception:
                    pass
                models.append({
                    "name": stem,
                    "path": item_path,
                    "type": "safetensors",
                    "source_type": "safetensors",
                    "size_gb": round(total_bytes / (1024**3), 2),
                    "source_dir": models_dir,
                    "architecture": architecture
                })
            elif item.endswith('.safetensors'):
                # Exclude vision encoder files from the main model list
                if architecture == "vision_encoder":
                    continue
                # Safetensors file
                file_size = os.path.getsize(item_path) / (1024**3)  # GB
                models.append({
                    "name": item.replace('.safetensors', ''),
                    "path": item_path,
                    "type": "safetensors",
                    "source_type": "safetensors",
                    "size_gb": round(file_size, 2),
                    "source_dir": models_dir,
                    "architecture": architecture
                })

    # Enrich each entry with component-registry data (lazy + persistently cached;
    # header/config-only reads, no weight load). Failures are swallowed per-model
    # so a single bad model never breaks the listing.
    try:
        from core.models.component_registry import get_or_scan as _cr_get_or_scan
    except Exception as _cr_e:
        _cr_get_or_scan = None
        print(f"[Models] Component registry unavailable: {_cr_e}")

    if _cr_get_or_scan is not None:
        for m in models:
            try:
                rec = _cr_get_or_scan(m["path"], m.get("source_type"))
                comps = rec.get("components", {}) or {}
                m["components"] = comps
                m["is_video"] = rec.get("is_video", False)
                m["latent_channels"] = (comps.get("vae", {}) or {}).get("latent_channels")
                m["te_out_dim"] = (comps.get("text_encoder", {}) or {}).get("out_dim")
                # arch already present as "architecture"; expose registry arch too
                m.setdefault("architecture", rec.get("arch"))
            except Exception as _me:
                print(f"[Models] Registry enrich failed for {m.get('path')}: {_me}")

    result = {"models": models}
    scan_duration = time.time() - scan_start

    print(f"[Models] Found {len(models)} models total")
    print(f"[Models] Scan completed in {scan_duration:.2f}s")

    # Cache result
    _models_cache = result
    _models_cache_timestamp = time.time()

    return result

@router.get("/models/vision_encoders")
async def list_vision_encoders(db: Session = Depends(get_gallery_db)):
    """List safetensors files detected as SigLIP2 vision encoders from the model directories.

    Scans directly without cache so new files are always visible immediately.
    """
    from core.model_loader import ModelLoader

    settings_record = db.query(UserSettings).first()
    additional_model_dirs = settings_record.model_dirs if settings_record else []
    all_dirs = [settings.models_dir] + additional_model_dirs

    vision_encoders = []
    for models_dir in all_dirs:
        if not os.path.exists(models_dir):
            continue
        for item in os.listdir(models_dir):
            if not item.endswith('.safetensors'):
                continue
            item_path = os.path.join(models_dir, item)
            architecture = ModelLoader.detect_model_type(item_path)
            if architecture == "vision_encoder":
                file_size = os.path.getsize(item_path) / (1024**3)
                vision_encoders.append({
                    "name": item.replace('.safetensors', ''),
                    "path": item_path,
                    "size_gb": round(file_size, 2),
                    "source_dir": models_dir,
                    "architecture": "vision_encoder",
                })
    return {"vision_encoders": vision_encoders}


def _override_scan_dirs(db: Session) -> List[str]:
    """Configured model dirs plus the shared VAE/text-encoder store dirs, de-duplicated."""
    settings_record = db.query(UserSettings).first()
    additional_model_dirs = settings_record.model_dirs if settings_record else []
    dirs = [settings.models_dir] + list(additional_model_dirs or [])
    vae_store = os.path.join(settings.models_dir, "vae")
    if vae_store not in dirs:
        dirs.append(vae_store)
    # Shared text-encoder store (core.models.common.te_store) — e.g. a locally
    # resolved Gemma-2-2b-it for PiD's opt-in runtime captioner.
    te_store = os.path.join(settings.models_dir, "text_encoders")
    if te_store not in dirs:
        dirs.append(te_store)
    # unique, existing
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def _training_vae_export_dirs() -> List[str]:
    """Diffusers VAE directories exported by a SushiUI VAE fine-tune.

    The training base dir (``get_training_base_dir()``) is NOT one of the
    configured model dirs, so a VAE the user just trained would otherwise be
    unreachable from the VAE-override picker except by typing its path.

    Recursion is bounded to EXACTLY two levels — ``<training_base>/<run>/<dir>``
    — which is where ``VaeTrainer._save_vae`` writes (``<run_name>_vae``,
    ``<run_name>_vae_noema``, ``<run_name>_vae_encoder_trained``[``_noema``]).
    Nothing deeper is listed, so the per-run ``checkpoints/`` (one subdir plus
    several multi-GB files per save), ``.dataset_cache/``, ``samples/`` and
    ``logs/`` trees are never walked: those names are skipped by name before
    any listdir, and a candidate directory is confirmed with two ``isfile``
    probes rather than by enumerating it.
    """
    skip_names = {"checkpoints", "samples", "logs", "sample", "optimizer"}
    out: List[str] = []
    try:
        from core.training.training_utils import get_training_base_dir
        from api.generation_overrides import VAE_TRAINING_SIDECAR
        base = get_training_base_dir()
    except Exception as e:
        print(f"[VAEs] training base dir unavailable: {e}")
        return out
    if not base or not os.path.isdir(base):
        return out
    try:
        run_names = os.listdir(base)
    except OSError:
        return out
    for run_name in run_names:
        if run_name.startswith("."):
            continue
        run_dir = os.path.join(base, run_name)
        if not os.path.isdir(run_dir):
            continue
        try:
            entries = os.listdir(run_dir)
        except OSError:
            continue
        for name in entries:
            if name.startswith(".") or name.lower() in skip_names:
                continue
            cand_dir = os.path.join(run_dir, name)
            # A SushiUI VAE export == diffusers config.json + our provenance
            # sidecar. Requiring the sidecar keeps this scan to VAE fine-tune
            # outputs only (a diffusion run's export dirs are not VAEs).
            if (os.path.isfile(os.path.join(cand_dir, VAE_TRAINING_SIDECAR))
                    and os.path.isfile(os.path.join(cand_dir, "config.json"))):
                out.append(cand_dir)
    return out


@router.get("/models/vaes")
async def list_vaes(db: Session = Depends(get_gallery_db)):
    """List standalone VAE candidates usable as a per-generation VAE override.

    Scans the shared VAE store (models_dir/vae), the configured model dirs, and
    the training base dir (VAE fine-tune exports, see
    ``_training_vae_export_dirs``). A candidate is a diffusers ``vae/`` dir or a
    model whose registry record has a VAE present and no backbone. Dims come
    from the component registry (header/config reads only). Unclassifiable
    entries are skipped (best-effort).
    """
    from api.generation_overrides import classify_vae_candidate

    results: List[Dict[str, Any]] = []
    seen_paths = set()

    def _key(path: str) -> str:
        # Case/separator-insensitive on Windows, so the same directory reached
        # through two different scan roots (e.g. a training dir that also lives
        # under a configured model dir) collapses to one entry.
        return os.path.normcase(os.path.normpath(path))

    def _consider(path: str):
        if not path or _key(path) in seen_paths:
            return
        try:
            cand = classify_vae_candidate(path)
        except Exception as e:
            print(f"[VAEs] classify failed for {path}: {e}")
            return
        if cand is not None:
            seen_paths.add(_key(path))
            # Also dedup on the candidate's RESOLVED path: a PiD checkpoint is
            # reachable both as its containing dir and as its .pth file, but both
            # classify to the same concrete .pth — show it once.
            cand_path = cand.get("path")
            if cand_path and _key(cand_path) != _key(path):
                if _key(cand_path) in seen_paths:
                    return
                seen_paths.add(_key(cand_path))
            results.append(cand)

    for scan_dir in _override_scan_dirs(db):
        try:
            entries = os.listdir(scan_dir)
        except OSError:
            continue
        for name in entries:
            item_path = os.path.join(scan_dir, name)
            if os.path.isdir(item_path):
                # a diffusers model dir OR a standalone VAE dir; classifier decides
                _consider(item_path)
                # one level of nesting (e.g. models_dir/vae/<subdir>)
                try:
                    for sub in os.listdir(item_path):
                        _sub_path = os.path.join(item_path, sub)
                        # Only descend into nested dirs or model/VAE files — never
                        # pass stray non-model files (e.g. sample .png images in a
                        # training run's folder) to the classifier, which would try
                        # a safetensors header read on each and fail noisily.
                        if os.path.isdir(_sub_path) or sub.endswith(".safetensors") or (
                            sub.startswith("PiD_") and sub.endswith(".pth")
                        ):
                            _consider(_sub_path)
                except OSError:
                    pass
            elif name.endswith(".safetensors") or (name.startswith("PiD_") and name.endswith(".pth")):
                # A bare .safetensors VAE, or a PiD (Pixel Diffusion Decoder)
                # checkpoint — classify_vae_candidate recognizes the latter via
                # its "PiD_*.pth" naming and returns kind="pid_decoder".
                _consider(item_path)

    # Training outputs last: a training dir nested inside a configured model dir
    # is already in seen_paths by now and is skipped rather than duplicated.
    # Wrapped so this addition can never take down the endpoint: the guarantee
    # is "degrades to no training entries", unconditionally.
    try:
        for export_dir in _training_vae_export_dirs():
            _consider(export_dir)
    except Exception as e:
        print(f"[VAEs] training-directory scan failed: {e}")

    return {"vaes": results}


@router.get("/models/text_encoders")
async def list_text_encoders(db: Session = Depends(get_gallery_db)):
    """List standalone text-encoder candidates usable as a TE override.

    Same scan surface as ``/models/vaes``; a candidate is a standalone TE dir or
    a model whose registry record has a text encoder present and no backbone.
    Dims (out_dim/te_type) come from the component registry. Best-effort.
    """
    from api.generation_overrides import classify_te_candidate

    results: List[Dict[str, Any]] = []
    seen_paths = set()

    def _consider(path: str):
        if not path or path in seen_paths:
            return
        try:
            cand = classify_te_candidate(path)
        except Exception as e:
            print(f"[TextEncoders] classify failed for {path}: {e}")
            return
        if cand is not None:
            seen_paths.add(path)
            results.append(cand)

    for scan_dir in _override_scan_dirs(db):
        try:
            entries = os.listdir(scan_dir)
        except OSError:
            continue
        for name in entries:
            item_path = os.path.join(scan_dir, name)
            if os.path.isdir(item_path):
                _consider(item_path)
                try:
                    for sub in os.listdir(item_path):
                        _sub_path = os.path.join(item_path, sub)
                        # Only descend into nested dirs or model/VAE files — never
                        # pass stray non-model files (e.g. sample .png images in a
                        # training run's folder) to the classifier, which would try
                        # a safetensors header read on each and fail noisily.
                        if os.path.isdir(_sub_path) or sub.endswith(".safetensors") or (
                            sub.startswith("PiD_") and sub.endswith(".pth")
                        ):
                            _consider(_sub_path)
                except OSError:
                    pass
            elif name.endswith(".safetensors"):
                _consider(item_path)

    return {"text_encoders": results}


class CreateScratchMiniT2IRequest(BaseModel):
    variant: str = "b16"       # "b16" | "l16"
    vae_type: str = "sdxl"     # "sdxl" | "flux1" | "none" (none = pixel-space)
    name: str                  # output directory name
    target_dir: Optional[str] = None  # parent dir; default = first configured models dir


@router.post("/models/minit2i/create-scratch")
async def create_scratch_minit2i_endpoint(
    request: CreateScratchMiniT2IRequest,
    db: Session = Depends(get_gallery_db),
):
    """Create a random-initialized MiniT2I model for from-scratch Full-FT training.

    Writes a diffusers dir (transformer + scheduler) that is then selectable in the
    model list and trained via Full-FT. Latent variants (vae_type sdxl/flux1) load
    their VAE by type at train/inference time (weights not bundled).
    """
    from core.models.minit2i.minit2i_loader import create_scratch_minit2i
    global _models_cache

    name = (request.name or "").strip()
    if not name or any(sep in name for sep in ("/", "\\", "..")):
        raise HTTPException(status_code=400, detail="Invalid model name")
    if request.variant not in ("b16", "l16"):
        raise HTTPException(status_code=400, detail="variant must be b16 or l16")
    if request.vae_type not in ("sdxl", "flux1", "none"):
        raise HTTPException(status_code=400, detail="vae_type must be sdxl, flux1 or none")

    # Resolve target parent dir (must be one of the configured model dirs, or default).
    settings_record = db.query(UserSettings).first()
    additional = settings_record.model_dirs if settings_record else []
    allowed = [settings.models_dir] + list(additional)
    parent = request.target_dir or settings.models_dir
    if parent not in allowed:
        raise HTTPException(status_code=400, detail=f"target_dir must be one of the configured model dirs: {allowed}")

    out_dir = os.path.join(parent, name)
    if os.path.exists(out_dir):
        raise HTTPException(status_code=400, detail=f"Path already exists: {out_dir}")

    try:
        create_scratch_minit2i(request.variant, request.vae_type, out_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create scratch model: {e}")

    _models_cache = None  # invalidate so the new model appears
    return {
        "status": "success",
        "path": out_dir,
        "name": name,
        "variant": request.variant,
        "vae_type": request.vae_type,
    }


@router.post("/models/load")
async def load_model(
    source_type: str = Form(...),
    source: str = Form(...),
    revision: Optional[str] = Form(None),
    force: bool = Form(False)
):
    """Load a model from various sources (fp16 by default).

    ``force`` re-loads even when the requested model is the one already loaded.
    Without it that request is a no-op, which made the documented recovery for
    every per-session component mutation -- above all the one-way in-place
    runtime INT8 conversion -- do nothing at all.
    """
    try:
        kwargs = {}
        if revision:
            kwargs["revision"] = revision

        # Run the (blocking, ~20s) load in the executor so it never blocks the event
        # loop -- important now that load_model serializes on a lock: if the boot
        # auto-load thread holds it, waiting here happens off the event loop.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            lambda: pipeline_manager.load_model(
                source_type=source_type,
                source=source,
                pipeline_type="txt2img",
                force_reload=bool(force),
                **kwargs
            )
        )

        return {
            "success": True,
            "message": "Model loaded successfully",
            "model_info": pipeline_manager.current_model_info
        }
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Error loading model: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))

class QuantizedExportRequest(BaseModel):
    """Body of ``POST /models/export-quantized``."""
    output_path: str
    link_siblings: bool = True
    overwrite: bool = False


@router.get("/models/export-quantized")
async def get_quantized_export_status():
    """Report whether the loaded transformer can be exported, and any job state.

    ``exportable`` is true exactly when the loaded model owns weight-only
    quantized Linear layers — from the checkpoint itself or from an in-place
    runtime conversion, which are indistinguishable by then and equally
    exportable. ``inventory`` is the per-format module census and
    ``suggested_path`` a destination outside the loaded model's own tree.
    """
    try:
        from api.quantized_export_job import export_status
        return export_status(pipeline_manager)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/export-quantized")
async def start_quantized_export(request: QuantizedExportRequest):
    """Write the loaded quantized transformer to a single file (async job).

    Refuses with 409 while an image generation or a training run is in flight
    (the export must describe one settled model state) or while another export
    job is running. The worker holds the model-load lock for the whole write.
    """
    from api.quantized_export_job import (
        ExportBusyError, ExportUnavailableError, job_is_running, start_export,
    )

    if job_is_running():
        raise HTTPException(
            status_code=409,
            detail="A quantized-model export is already running; wait for it to finish.")
    busy = _fp8_scaled_mm_busy_reason()
    if busy:
        raise HTTPException(
            status_code=409,
            detail=(f"Cannot export the quantized model while {busy}. The export reads "
                    f"the live transformer, so it must describe one settled state."))
    try:
        return start_export(
            pipeline_manager,
            request.output_path,
            link_siblings=bool(request.link_siblings),
            overwrite=bool(request.overwrite),
        )
    except ExportBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ExportUnavailableError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a safetensors model file"""
    if not file.filename.endswith('.safetensors'):
        raise HTTPException(status_code=400, detail="Only .safetensors files are supported")

    try:
        os.makedirs(settings.models_dir, exist_ok=True)
        file_path = os.path.join(settings.models_dir, file.filename)

        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {
            "success": True,
            "message": "Model uploaded successfully",
            "filename": file.filename,
            "path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/current")
async def get_current_model():
    """Get currently loaded model info"""
    if pipeline_manager.current_model_info:
        info = dict(pipeline_manager.current_model_info)
        # Enrich with the loaded architecture's latent_channels from the component
        # wiring spec (the single source of truth) so the frontend's compatible-only
        # VAE-override filter can match on latent channels instead of hardcoding an
        # arch->channels map. Best-effort; absent for archs without a wiring entry.
        if "latent_channels" not in info:
            try:
                from core.models.component_registry import _WIRING_BY_ARCH
                _spec = _WIRING_BY_ARCH.get(str(info.get("type") or "").lower())
                if _spec is not None:
                    info["latent_channels"] = _spec.latent_channels
            except Exception:
                pass
        return {
            "loaded": True,
            "model_info": info
        }
    else:
        return {"loaded": False}

@router.get("/samplers")
async def get_samplers():
    """Get available samplers (depends on current model type: SD/SDXL/DEUS vs Flow Matching models)"""
    try:
        # Check if current model is Flow Matching (Z-Image, FLUX.2)
        # Note: DEUS uses SDXL-like architecture with standard diffusion, NOT Flow Matching
        is_flow_matching = (
            pipeline_manager.is_zimage_model or
            pipeline_manager.is_flux2_model or
            pipeline_manager.is_anima_model or
            pipeline_manager.is_lens_model or
            pipeline_manager.is_ideogram4_model or
            pipeline_manager.is_minit2i_model or
            pipeline_manager.is_krea2_model
        )

        if is_flow_matching:
            # Flow Matching samplers (Z-Image, FLUX.2)
            # Only Euler and Heun are truly different; other names map to Euler
            samplers_list = [
                {"id": "euler", "name": "Euler (Flow Match)"},
                {"id": "euler_a", "name": "Euler a (Flow Match + Stochastic)"},
                {"id": "heun", "name": "Heun (Flow Match)"},
            ]
        else:
            # SD/SDXL/DEUS samplers (standard diffusion)
            samplers = get_available_samplers()
            display_names = get_sampler_display_names()
            samplers_list = [
                {"id": sampler_id, "name": display_names.get(sampler_id, sampler_id)}
                for sampler_id in samplers
            ]

        return {
            "samplers": samplers_list,
            "is_flow_matching": is_flow_matching
        }
    except Exception as e:
        print(f"[ERROR] Failed to get samplers: {e}")
        import traceback
        traceback.print_exc()
        # Return hardcoded fallback
        return {
            "samplers": [
                {"id": "euler", "name": "Euler"},
                {"id": "euler_a", "name": "Euler a"},
                {"id": "dpmpp_2m", "name": "DPM++ 2M"},
                {"id": "dpmpp_sde", "name": "DPM++ SDE"},
                {"id": "dpm2", "name": "DPM2"},
                {"id": "dpm2_a", "name": "DPM2 a"},
                {"id": "heun", "name": "Heun"},
                {"id": "ddim", "name": "DDIM"},
                {"id": "lms", "name": "LMS"},
                {"id": "unipc", "name": "UniPC"},
            ]
        }

@router.get("/schedule-types")
async def get_schedule_types():
    """Get available schedule types (static list, doesn't require model)"""
    try:
        schedule_types = get_available_schedule_types()
        display_names = get_schedule_type_display_names()
        return {
            "schedule_types": [
                {"id": schedule_id, "name": display_names.get(schedule_id, schedule_id)}
                for schedule_id in schedule_types
            ]
        }
    except Exception as e:
        print(f"[ERROR] Failed to get schedule types: {e}")
        import traceback
        traceback.print_exc()
        # Return hardcoded fallback
        return {
            "schedule_types": [
                {"id": "uniform", "name": "Uniform"},
                {"id": "karras", "name": "Karras"},
                {"id": "exponential", "name": "Exponential"},
            ]
        }

@router.get("/loras")
async def get_loras():
    """Get available LoRA files"""
    try:
        loras = lora_manager.get_available_loras()
        print(f"[DEBUG] get_loras: Found {len(loras)} LoRA files")
        if len(loras) > 0:
            print(f"[DEBUG] First LoRA: {loras[0]}")
        # get_available_loras() already returns {"path", "name", "arch"} dicts
        result = {"loras": loras}
        print(f"[DEBUG] Returning {len(result['loras'])} LoRAs")
        return result
    except Exception as e:
        print(f"[ERROR] get_loras failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/loras/{lora_name:path}")
async def get_lora_info(lora_name: str):
    """Get information about a specific LoRA"""
    try:
        info = lora_manager.get_lora_info(lora_name)
    except LoRAAmbiguousIdentifierError as e:
        # The identifier resolves to different files across more than one
        # registered directory -- refuse to guess which one the caller means
        # instead of silently applying the wrong LoRA.
        raise HTTPException(status_code=409, detail=str(e))
    if not info:
        raise HTTPException(status_code=404, detail="LoRA not found")
    return info

@router.post("/tokenize")
async def tokenize_prompt(prompt: str = Form(...)):
    """Get token count for a prompt using the loaded model's tokenizer"""
    try:
        if not pipeline_manager.txt2img_pipeline:
            raise HTTPException(status_code=400, detail="No model loaded")

        # Get tokenizer from pipeline
        from diffusers import StableDiffusionXLPipeline
        is_sdxl = isinstance(pipeline_manager.txt2img_pipeline, StableDiffusionXLPipeline)
        tokenizer = pipeline_manager.txt2img_pipeline.tokenizer_2 if is_sdxl else pipeline_manager.txt2img_pipeline.tokenizer

        # Tokenize without special tokens to get actual content token count
        tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        token_count = len(tokens)

        # Add 2 for BOS/EOS tokens
        total_count = token_count + 2

        return {
            "token_count": token_count,
            "total_count": total_count,
            "chunks": (token_count + 74) // 75  # Number of 75-token chunks needed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/controlnets")
async def get_controlnets():
    """Get available ControlNet models"""
    try:
        controlnets = controlnet_manager.get_available_controlnets()
        print(f"[DEBUG] get_controlnets: Found {len(controlnets)} ControlNet models")
        if len(controlnets) > 0:
            print(f"[DEBUG] First ControlNet: {controlnets[0]}")
        result = {
            "controlnets": [
                {"path": cn, "name": os.path.basename(cn)}
                for cn in controlnets
            ]
        }
        print(f"[DEBUG] Returning {len(result['controlnets'])} ControlNets")
        return result
    except Exception as e:
        print(f"[ERROR] get_controlnets failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/controlnets/{controlnet_path:path}/info")
async def get_controlnet_info(controlnet_path: str):
    """Get information about a specific ControlNet model"""
    try:
        is_lllite = controlnet_manager.is_lllite_model(controlnet_path)
        layers = controlnet_manager.get_controlnet_layers(controlnet_path) if not is_lllite else []
        return {
            "name": os.path.basename(controlnet_path),
            "path": controlnet_path,
            "layers": layers,
            "is_lllite": is_lllite,
            "exists": True
        }
    except Exception as e:
        print(f"Error getting ControlNet info: {e}")
        return {
            "name": os.path.basename(controlnet_path),
            "path": controlnet_path,
            "layers": [],
            "is_lllite": False,
            "exists": False,
            "error": str(e)
        }

@router.get("/settings/directories")
async def get_directory_settings(db: Session = Depends(get_gallery_db)):
    """Get user-configured model directories"""
    try:
        # Get or create settings record (we'll only have one record for singleton settings)
        settings_record = db.query(UserSettings).first()
        if not settings_record:
            settings_record = UserSettings(
                model_dirs=[],
                lora_dirs=[],
                controlnet_dirs=[],
                cache_dir=None,
                training_dir=None
            )
            db.add(settings_record)
            db.commit()
            db.refresh(settings_record)

        return settings_record.to_dict()
    except Exception as e:
        print(f"Error getting directory settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/settings/directories")
async def save_directory_settings(
    settings_data: dict,
    db: Session = Depends(get_gallery_db)
):
    """Save user-configured model directories, cache directory, and training directory"""
    # Extract from request body
    model_dirs = settings_data.get("model_dirs", [])
    lora_dirs = settings_data.get("lora_dirs", [])
    controlnet_dirs = settings_data.get("controlnet_dirs", [])
    cache_dir = settings_data.get("cache_dir")
    training_dir = settings_data.get("training_dir")
    try:
        # Get or create settings record
        settings_record = db.query(UserSettings).first()
        if not settings_record:
            settings_record = UserSettings()
            db.add(settings_record)

        # Update directory paths (filter out empty strings)
        settings_record.model_dirs = [d.strip() for d in model_dirs if d.strip()]
        settings_record.lora_dirs = [d.strip() for d in lora_dirs if d.strip()]
        settings_record.controlnet_dirs = [d.strip() for d in controlnet_dirs if d.strip()]
        settings_record.cache_dir = cache_dir.strip() if cache_dir and cache_dir.strip() else None
        settings_record.training_dir = training_dir.strip() if training_dir and training_dir.strip() else None
        settings_record.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(settings_record)

        # Invalidate models cache so new directories take effect immediately
        global _models_cache, _models_cache_timestamp
        _models_cache = None
        _models_cache_timestamp = 0

        print(f"[Settings] Updated directory settings:")
        print(f"  Model dirs: {settings_record.model_dirs}")
        print(f"  LoRA dirs: {settings_record.lora_dirs}")
        print(f"  ControlNet dirs: {settings_record.controlnet_dirs}")
        print(f"  Cache dir: {settings_record.cache_dir}")
        print(f"  Training dir: {settings_record.training_dir}")

        # Update managers with new directories
        lora_manager.set_additional_dirs(settings_record.lora_dirs)
        controlnet_manager.set_additional_dirs(settings_record.controlnet_dirs)

        return {
            "success": True,
            "message": "Directory settings saved successfully",
            "settings": settings_record.to_dict()
        }
    except Exception as e:
        print(f"Error saving directory settings: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/settings/generation")
async def get_generation_settings(db: Session = Depends(get_gallery_db)):
    """Get user-configured generation settings"""
    try:
        settings_record = db.query(UserSettings).first()
        if not settings_record:
            settings_record = UserSettings()
            db.add(settings_record)
            db.commit()
            db.refresh(settings_record)

        return {
            "inpaint_use_dedicated_model": settings_record.inpaint_use_dedicated_model if settings_record.inpaint_use_dedicated_model is not None else False,
        }
    except Exception as e:
        print(f"Error getting generation settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/settings/generation")
async def save_generation_settings(
    settings_data: dict,
    db: Session = Depends(get_gallery_db)
):
    """Save user-configured generation settings"""
    try:
        settings_record = db.query(UserSettings).first()
        if not settings_record:
            settings_record = UserSettings()
            db.add(settings_record)

        # Update generation settings
        if "inpaint_use_dedicated_model" in settings_data:
            settings_record.inpaint_use_dedicated_model = bool(settings_data["inpaint_use_dedicated_model"])

        settings_record.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(settings_record)

        print(f"[Settings] Updated generation settings:")
        print(f"  inpaint_use_dedicated_model: {settings_record.inpaint_use_dedicated_model}")

        return {
            "success": True,
            "message": "Generation settings saved successfully",
            "settings": {
                "inpaint_use_dedicated_model": settings_record.inpaint_use_dedicated_model,
            }
        }
    except Exception as e:
        print(f"Error saving generation settings: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system/restart-backend")
async def restart_backend(background_tasks: BackgroundTasks):
    """Restart the backend server"""
    try:
        python_exe = sys.executable
        backend_dir = os.path.dirname(os.path.dirname(__file__))
        main_path = os.path.join(backend_dir, "main.py")

        if sys.platform == "win32":
            from api.restart_backend_helper import (
                build_helper_command,
                helper_creation_flags,
            )

            helper_path = os.path.join(
                os.path.dirname(__file__), "restart_backend_helper.py"
            )
            helper_command = build_helper_command(
                python_executable=python_exe,
                helper_path=helper_path,
                parent_pid=os.getpid(),
                main_path=main_path,
                backend_dir=backend_dir,
            )

            print(f"Scheduling backend restart: {python_exe} {main_path}")
            print(f"Working directory: {backend_dir}")
            subprocess.Popen(
                helper_command,
                cwd=backend_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=helper_creation_flags(),
                close_fds=True,
            )
            background_tasks.add_task(os._exit, 0)
        else:
            background_tasks.add_task(
                os.execv, python_exe, [python_exe, main_path]
            )

        return {"success": True, "message": "Backend restart scheduled"}
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Restart backend error: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))

class Fp8ScaledMmRequest(BaseModel):
    """Body of ``POST /system/fp8-scaled-mm``."""
    enabled: bool


class Int8MmRequest(BaseModel):
    """Body of ``POST /system/int8-mm``."""
    enabled: bool


def _fp8_scaled_mm_busy_reason() -> Optional[str]:
    """Why the FP8 GEMM path must not be flipped right now, or None.

    Both paths are numerically valid, but they are NOT the same function
    (the W8A8 path additionally quantizes the activation), so a mid-run flip
    would make the metadata recorded for that run describe a path that produced
    only part of it.

    Fails CLOSED: if either introspection probe raises unexpectedly, this
    reports "busy" (refuses the flip) rather than silently allowing it. A
    refused toggle is recoverable (retry once the ambiguity is gone); a flip
    that lands mid-generation is not -- it corrupts the ``fp8_gemm`` metadata
    recorded for that run. The two probes are reported separately so the
    refusal message says which introspection failed instead of a generic
    "internal error".
    """
    try:
        from api.generation_status import get_snapshot
        if get_snapshot().get("status") == "running":
            return "an image generation is running"
    except Exception as exc:
        return f"generation status could not be determined ({type(exc).__name__}: {exc})"
    try:
        from core.training.training_process import training_process_manager
        active = [str(rid) for rid, p in training_process_manager.processes.items()
                  if p.is_running]
        if active:
            return f"training run(s) {', '.join(active)} are active"
    except Exception as exc:
        return f"training status could not be determined ({type(exc).__name__}: {exc})"
    return None


@router.get("/system/fp8-scaled-mm")
async def get_fp8_scaled_mm():
    """Report the FP8 W8A8 ``torch._scaled_mm`` state of this backend process."""
    try:
        from core.models.ideogram4.vendor.fp8_linear import get_scaled_mm_state
        return get_scaled_mm_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/fp8-scaled-mm")
async def set_fp8_scaled_mm(request: Fp8ScaledMmRequest):
    """Select the FP8 GEMM path (W8A8 scaled GEMM or dequantized matmul).

    Per-process and not persisted: a restart returns to the value the
    environment gives (default off).
    """
    busy = _fp8_scaled_mm_busy_reason()
    if busy:
        raise HTTPException(
            status_code=409,
            detail=(f"Cannot change the FP8 GEMM path while {busy}. The two paths "
                    f"are numerically different, so a mid-run change would make the "
                    f"recorded metadata describe a path that produced only part of "
                    f"the result."),
        )
    try:
        from core.models.ideogram4.vendor.fp8_linear import set_scaled_mm_enabled
        return set_scaled_mm_enabled(bool(request.enabled), origin="api")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/int8-mm")
async def get_int8_mm():
    """Report the INT8 W8A8 ``torch._int_mm`` state of this backend process."""
    try:
        from core.models.ideogram4.vendor.int8_linear import get_int8_mm_state
        return get_int8_mm_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/int8-mm")
async def set_int8_mm(request: Int8MmRequest):
    """Select the INT8 GEMM path (W8A8 integer GEMM or dequantized matmul).

    Per-process and not persisted: a restart returns to the value the
    environment gives (default off). Independent of ``/system/fp8-scaled-mm``;
    a mixed int8/e4m3 checkpoint is governed by both, each over its own layers.
    """
    # Same busy check as the FP8 toggle, reused rather than duplicated: the
    # reason to refuse is identical (both paths are numerically valid but
    # DIFFERENT, and a mid-run flip corrupts the ``fp8_gemm`` metadata recorded
    # for that run), and it fails CLOSED for the same reason.
    busy = _fp8_scaled_mm_busy_reason()
    if busy:
        raise HTTPException(
            status_code=409,
            detail=(f"Cannot change the INT8 GEMM path while {busy}. The two paths "
                    f"are numerically different, so a mid-run change would make the "
                    f"recorded metadata describe a path that produced only part of "
                    f"the result."),
        )
    try:
        from core.models.ideogram4.vendor.int8_linear import set_int8_mm_enabled
        return set_int8_mm_enabled(bool(request.enabled), origin="api")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/restart-frontend")
async def restart_frontend():
    """Restart the frontend server (via npm)"""
    try:
        # This will send a signal to restart the frontend
        # The frontend will need to handle this on its side
        return {"success": True, "message": "Frontend restart signal sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Temp image storage endpoints
import base64
import hashlib
import time

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/temp-images/upload")
async def upload_temp_image(image_base64: str = Form(...)):
    """Upload a base64 image to temp storage and return a reference ID"""
    try:
        # Decode base64 image
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_data = base64.b64decode(image_base64)

        # Generate unique filename based on content hash and timestamp
        content_hash = hashlib.sha256(image_data).hexdigest()[:16]
        timestamp = str(int(time.time() * 1000))
        filename = f"{timestamp}_{content_hash}.png"
        filepath = os.path.join(TEMP_DIR, filename)

        # Save image
        image = Image.open(io.BytesIO(image_data))
        image.save(filepath, "PNG")

        return {"success": True, "image_id": filename}
    except Exception as e:
        print(f"Error uploading temp image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/temp-images/{image_id}")
async def get_temp_image(image_id: str):
    """Get a temp image by ID and return as base64"""
    try:
        filepath = os.path.join(TEMP_DIR, image_id)

        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Image not found")

        # Read image and convert to base64
        with open(filepath, "rb") as f:
            image_data = f.read()

        image_base64 = base64.b64encode(image_data).decode("utf-8")

        return {"success": True, "image_base64": f"data:image/png;base64,{image_base64}"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting temp image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/temp-images/{image_id}")
async def delete_temp_image(image_id: str):
    """Delete a temp image by ID"""
    try:
        filepath = os.path.join(TEMP_DIR, image_id)

        if os.path.exists(filepath):
            os.remove(filepath)

        return {"success": True, "message": "Image deleted"}
    except Exception as e:
        print(f"Error deleting temp image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/temp-images/cleanup")
async def cleanup_temp_images(max_age_hours: int = 24):
    """Clean up temp images older than specified hours"""
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0

        for filename in os.listdir(TEMP_DIR):
            filepath = os.path.join(TEMP_DIR, filename)

            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)

                if file_age > max_age_seconds:
                    os.remove(filepath)
                    deleted_count += 1

        return {"success": True, "deleted_count": deleted_count}
    except Exception as e:
        print(f"Error cleaning up temp images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/taglist/timestamps")
async def get_taglist_timestamps():
    """
    Get modification timestamps for all tag files to check if cache is stale.

    MIGRATED: Uses TaglistCache for taglist files (Phase 4).

    Returns Unix timestamps in milliseconds.
    """
    try:
        # Use TaglistCache for taglist timestamps
        timestamps = taglist_cache.get_all_timestamps()

        # Get tag_other_names timestamp (not in taglist, keep manual check)
        tagother_path = os.path.join(settings.root_dir, "tagother", "tag_other_names.json")
        if os.path.exists(tagother_path):
            mtime = os.path.getmtime(tagother_path)
            timestamps["other_names"] = int(mtime * 1000)
        else:
            timestamps["other_names"] = 0

        return timestamps
    except Exception as e:
        print(f"[Taglist API] Error getting tag file timestamps: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tag-category/add")
async def add_tag_to_category(request: AddTagRequest, db: Session = Depends(get_datasets_db)):
    """Add a tag to a category's taglist JSON file and update all datasets' tag statistics

    Args:
        request: AddTagRequest containing tag, category, and count
        db: Database session

    Returns:
        Status message
    """
    import json
    import os

    # Validate category
    valid_categories = ["Artist", "Character", "Copyright", "General", "Meta", "Model", "Quality", "Rating"]
    if request.category not in valid_categories:
        raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}")

    # Taglist file path
    taglist_file = os.path.join(settings.root_dir, "taglist", f"{request.category}.json")

    if not os.path.exists(taglist_file):
        raise HTTPException(status_code=404, detail=f"Taglist file not found: {taglist_file}")

    try:
        # Load existing taglist
        with open(taglist_file, 'r', encoding='utf-8') as f:
            taglist = json.load(f)

        # Check if tag already exists in this category
        tag_already_exists = request.tag in taglist
        json_updated = False

        if not tag_already_exists:
            # Add tag to taglist JSON
            taglist[request.tag] = request.count

            # Sort by count (descending) and write back
            sorted_taglist = dict(sorted(taglist.items(), key=lambda x: int(x[1]), reverse=True))

            with open(taglist_file, 'w', encoding='utf-8') as f:
                json.dump(sorted_taglist, f, ensure_ascii=False, indent=2)

            json_updated = True
            print(f"[TagCategory] Added tag '{request.tag}' to {request.category}.json")

            # Invalidate TaglistCache to ensure cache consistency
            taglist_cache.invalidate_category(request.category)

            # Record user addition in a separate log file (project root, not in taglist/)
            user_additions_file = os.path.join(settings.root_dir, "user_tag_additions.json")
            user_additions = []

            if os.path.exists(user_additions_file):
                try:
                    with open(user_additions_file, 'r', encoding='utf-8') as f:
                        user_additions = json.load(f)
                except:
                    user_additions = []

            # Add new entry with timestamp
            from datetime import datetime
            user_additions.append({
                "tag": request.tag,
                "category": request.category,
                "count": request.count,
                "timestamp": datetime.now().isoformat()
            })

            # Write user additions log (keep last 1000 entries)
            with open(user_additions_file, 'w', encoding='utf-8') as f:
                json.dump(user_additions[-1000:], f, ensure_ascii=False, indent=2)
        else:
            print(f"[TagCategory] Tag '{request.tag}' already exists in {request.category}.json, skipping JSON update")

        # Update tag category in all datasets' tag_statistics
        datasets = db.query(Dataset).all()
        updated_datasets = 0
        for dataset in datasets:
            if dataset.tag_statistics and request.tag in dataset.tag_statistics:
                # Update category for this tag
                dataset.tag_statistics[request.tag]["category"] = request.category
                updated_datasets += 1

        # Commit database changes
        if updated_datasets > 0:
            db.commit()
            print(f"[TagCategory] Updated category for tag '{request.tag}' in {updated_datasets} datasets")

        # Build response message
        if tag_already_exists:
            message = f"Tag '{request.tag}' already exists in {request.category} category. Updated {updated_datasets} dataset(s)."
        else:
            message = f"Tag '{request.tag}' added to {request.category} category. Updated {updated_datasets} dataset(s)."

        return {
            "status": "success",
            "message": message,
            "tag": request.tag,
            "category": request.category,
            "count": request.count,
            "json_updated": json_updated,
            "updated_datasets": updated_datasets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update taglist: {str(e)}")

@router.get("/taglist/{category}")
async def get_taglist(category: str):
    """
    Get tag list for a specific category using TaglistCache.

    MIGRATED: Uses server-side cache instead of direct file read (Phase 4).

    Categories: general, character, artist, copyright, meta, model
    """
    try:
        # Validate category
        valid_categories = ["general", "character", "artist", "copyright", "meta", "model"]
        if category.lower() not in valid_categories:
            raise HTTPException(status_code=404, detail=f"Unknown category: {category}")

        # Use TaglistCache for O(1) lookup with automatic mtime-based invalidation
        tags = taglist_cache.get_category_tags(category.lower())

        if not tags:
            raise HTTPException(status_code=404, detail=f"No tags found for category: {category}")

        return tags
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Taglist API] Error loading taglist for category '{category}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tagother/tag_other_names")
async def get_tag_other_names():
    """
    Get tag other names (multilingual aliases) from tagother directory
    """
    try:
        tagother_path = os.path.join(settings.root_dir, "tagother", "tag_other_names.json")

        if not os.path.exists(tagother_path):
            raise HTTPException(status_code=404, detail=f"Tag other names file not found: {tagother_path}")

        import json
        with open(tagother_path, "r", encoding="utf-8") as f:
            tag_other_names = json.load(f)

        return tag_other_names
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading tag other names: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ResolveCategoriesRequest(BaseModel):
    """Request body for POST /tags/resolve-categories."""
    tags: List[str] = Field(
        ...,
        description="Tags to resolve. Any format is accepted (underscore or space "
                    "separated, escaped or plain parentheses).",
        example=["1girl", "hatsune_miku_(vocaloid)", "masterpiece", "explicit"],
    )


@router.post("/tags/resolve-categories")
async def resolve_tag_categories(request: ResolveCategoriesRequest):
    """
    Resolve the Danbooru-style category for each tag.

    Server-side canonical category resolution for API clients that do not run the
    frontend. Uses the shared backend resolver (TagGroupManager in
    core/training/tag_group_utils.py, backed by the TaglistCache singleton), which
    combines taglist category lookup with the hardcoded Rating/Quality special tags.

    Matching semantics are aligned with frontend/src/utils/tagSuggestions.ts:
    - Normalization removes backslash escapes, converts underscores to spaces, and
      lowercases the tag.
    - Rating/Quality special tags are recognized independently of the taglists.

    Categories returned: General, Character, Artist, Copyright, Meta, Model, Rating,
    Quality, or Unknown when the tag is not found.

    A maximum of 1000 tags is accepted per request.
    """
    if len(request.tags) > 1000:
        raise HTTPException(
            status_code=400,
            detail=f"Too many tags: {len(request.tags)} (maximum 1000 per request)",
        )

    try:
        from core.training.tag_group_utils import (
            get_tag_group_manager,
            normalize_tag_for_matching,
        )

        manager = get_tag_group_manager()

        results = []
        for tag in request.tags:
            category = manager.get_tag_group(tag)
            results.append({
                "tag": tag,
                "category": category if category is not None else "Unknown",
                "normalized": normalize_tag_for_matching(tag),
            })

        return {"results": results}
    except Exception as e:
        print(f"[Tags API] Error resolving tag categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ControlNet Preprocessor Endpoints ====================

@router.get("/controlnet/detect-preprocessor")
async def detect_controlnet_preprocessor(model_path: str):
    """Detect which preprocessor should be used for a ControlNet model"""
    try:
        preprocessor_type = controlnet_preprocessor.detect_preprocessor_from_model_name(model_path)
        return {
            "model_path": model_path,
            "preprocessor": preprocessor_type,
            "requires_preprocessing": preprocessor_type not in ["none", "tile", "blur"]
        }
    except Exception as e:
        print(f"Error detecting preprocessor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/controlnet/preprocess-image")
async def preprocess_controlnet_image(
    image: UploadFile = File(...),
    preprocessor: str = Form(...),
    low_threshold: int = Form(100),
    high_threshold: int = Form(200),
    down_sampling_rate: float = Form(2.0),
    sharpness: float = Form(1.0),
    kernel_size: int = Form(15),
    blur_strength: float = Form(None)
):
    """Preprocess an image for ControlNet

    Args:
        image: Image file to preprocess
        preprocessor: Type of preprocessor to use (canny, depth_midas, openpose, etc.)
        low_threshold: Low threshold for Canny (default: 100)
        high_threshold: High threshold for Canny (default: 200)
        down_sampling_rate: Down sampling rate for tile preprocessors (default: 2.0)
        sharpness: Sharpness for tile_colorfix+sharp (default: 1.0)
        kernel_size: Kernel size for Gaussian blur (default: 15, deprecated)
        blur_strength: Blur strength as percentage of image size (0.0-10.0, recommended)

    Returns:
        Preprocessed image as base64 string
    """
    try:
        # Read uploaded image
        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes))

        # Apply preprocessing
        preprocessed = controlnet_preprocessor.preprocess(
            image_pil,
            preprocessor,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            down_sampling_rate=down_sampling_rate,
            sharpness=sharpness,
            kernel_size=kernel_size,
            blur_strength=blur_strength
        )
        
        # Convert to base64
        buffered = io.BytesIO()
        preprocessed.save(buffered, format="PNG")
        import base64
        preprocessed_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "preprocessed_image": f"data:image/png;base64,{preprocessed_base64}",
            "preprocessor": preprocessor
        }
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/controlnet/preprocessors")
async def get_available_preprocessors():
    """Get list of available preprocessors"""
    return {
        "preprocessors": [
            {"id": "none", "name": "No Preprocessing", "category": "none"},
            # Edge Detection
            {"id": "canny", "name": "Canny Edge Detection", "category": "edge"},
            {"id": "softedge_hed", "name": "Soft Edge (HED)", "category": "edge"},
            {"id": "softedge_pidi", "name": "Soft Edge (PIDI)", "category": "edge"},
            # Scribble (similar to soft edge)
            {"id": "scribble_hed", "name": "Scribble (HED)", "category": "scribble"},
            {"id": "scribble_pidinet", "name": "Scribble (PIDINet)", "category": "scribble"},
            # Depth
            {"id": "depth_midas", "name": "Depth (Midas)", "category": "depth"},
            {"id": "depth_zoe", "name": "Depth (Zoe)", "category": "depth"},
            {"id": "depth_leres", "name": "Depth (Leres)", "category": "depth"},
            # Pose
            {"id": "openpose", "name": "OpenPose (Body)", "category": "pose"},
            {"id": "openpose_hand", "name": "OpenPose (Body + Hand)", "category": "pose"},
            {"id": "openpose_face", "name": "OpenPose (Body + Face)", "category": "pose"},
            {"id": "openpose_full", "name": "OpenPose (Full)", "category": "pose"},
            # Normal Maps
            {"id": "normal_bae", "name": "Normal Map (BAE)", "category": "normal"},
            # Lineart
            {"id": "lineart", "name": "Lineart", "category": "lineart"},
            {"id": "lineart_anime", "name": "Lineart (Anime)", "category": "lineart"},
            # Segmentation
            {"id": "segment_ofade20k", "name": "Segmentation (OFADE20K)", "category": "segment"},
            # Line Detection
            {"id": "mlsd", "name": "MLSD Line Detection", "category": "line"},
            # Tile (for upscaling)
            {"id": "tile", "name": "Tile (No Preprocessing)", "category": "tile"},
            {"id": "tile_resample", "name": "Tile Resample", "category": "tile"},
            {"id": "tile_colorfix", "name": "Tile Color Fix", "category": "tile"},
            {"id": "tile_colorfix+sharp", "name": "Tile Color Fix + Sharp", "category": "tile"},
            # Simple operations
            {"id": "blur", "name": "Gaussian Blur", "category": "simple"},
            {"id": "invert", "name": "Invert (Black/White)", "category": "simple"},
            {"id": "binary", "name": "Binary Threshold", "category": "simple"},
            {"id": "color", "name": "Color Simplification", "category": "simple"},
            {"id": "threshold", "name": "Threshold", "category": "simple"}
        ]
    }



# ============================================================================
# MiniMax-H3 Prompt Assist
# ============================================================================

class PromptAssistReference(BaseModel):
    token: str
    kind: Literal["picture", "video", "audio", "subject"]
    role: str
    description: str = PROMPT_ASSIST_DEFAULTS["reference_description"]


class PromptAssistModelRequest(BaseModel):
    provider: Literal["lm_studio", "ollama"] = PROMPT_ASSIST_DEFAULTS["provider"]
    base_url: str = PROMPT_ASSIST_DEFAULTS["base_url"]
    api_key: str = Field(
        PROMPT_ASSIST_DEFAULTS["api_key"], json_schema_extra={"writeOnly": True}
    )


class PromptAssistTemplateRequest(BaseModel):
    prompt: str
    mode: Literal["t2va", "i2va", "fl2va", "l2va", "ref2va"]
    duration_seconds: float = Field(ge=0.001, le=600)


class PromptAssistTransformRequest(PromptAssistTemplateRequest):
    provider: Literal["lm_studio", "ollama"] = PROMPT_ASSIST_DEFAULTS["provider"]
    base_url: str = PROMPT_ASSIST_DEFAULTS["base_url"]
    model: str = PROMPT_ASSIST_DEFAULTS["model"]
    api_key: str = Field(
        PROMPT_ASSIST_DEFAULTS["api_key"], json_schema_extra={"writeOnly": True}
    )
    references: List[PromptAssistReference] = Field(
        default_factory=lambda: list(PROMPT_ASSIST_DEFAULTS["references"])
    )
    temperature: float = Field(PROMPT_ASSIST_DEFAULTS["temperature"], ge=0, le=1)
    top_p: float = Field(PROMPT_ASSIST_DEFAULTS["top_p"], gt=0, le=1)
    max_output_tokens: int = Field(PROMPT_ASSIST_DEFAULTS["max_output_tokens"], ge=128, le=8192)
    context_length: int = Field(PROMPT_ASSIST_DEFAULTS["context_length"], ge=1024, le=262144)
    timeout_seconds: int = Field(PROMPT_ASSIST_DEFAULTS["timeout_seconds"], ge=10, le=1800)
    force_refresh: bool = PROMPT_ASSIST_DEFAULTS["force_refresh"]


def _prompt_assist_base_url(provider: str, requested: str) -> str:
    if requested.strip():
        return requested
    key = "lm_studio_base_url" if provider == "lm_studio" else "ollama_base_url"
    return str(PROMPT_ASSIST_DEFAULTS[key])


@router.post("/prompt-assist/models", tags=["prompt-assist"])
async def list_prompt_assist_models(request: PromptAssistModelRequest):
    try:
        return {
            "provider": request.provider,
            "models": await asyncio.to_thread(
                minimax_h3_prompt_assistant.list_models,
                request.provider,
                _prompt_assist_base_url(request.provider, request.base_url),
                request.api_key,
            ),
        }
    except PromptAssistError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/prompt-assist/template", tags=["prompt-assist"])
async def create_prompt_assist_template(request: PromptAssistTemplateRequest):
    try:
        prompt = build_h3_prompt_template(
            request.prompt, request.mode, request.duration_seconds
        )
        warnings = validate_h3_prompt(prompt, request.mode, request.duration_seconds)
        return {"prompt": prompt, "warnings": warnings, "valid": not warnings}
    except PromptAssistError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/prompt-assist/transform", tags=["prompt-assist"])
async def transform_h3_prompt(request: PromptAssistTransformRequest):
    options = PromptAssistOptions(
        prompt=request.prompt,
        mode=request.mode.lower(),
        duration_seconds=request.duration_seconds,
        references=[item.model_dump() for item in request.references],
        provider=request.provider,
        base_url=_prompt_assist_base_url(request.provider, request.base_url),
        model=request.model,
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=request.max_output_tokens,
        context_length=request.context_length,
        timeout_seconds=request.timeout_seconds,
        force_refresh=request.force_refresh,
    )
    try:
        return await asyncio.to_thread(
            minimax_h3_prompt_assistant.transform, options, request.api_key
        )
    except PromptAssistError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/prompt-assist/cache/clear", tags=["prompt-assist"])
async def clear_prompt_assist_cache():
    deleted = await asyncio.to_thread(minimax_h3_prompt_assistant.cache.clear)
    return {"status": "success", "deleted": deleted}


# ============================================================================
# TIPO (Prompt Optimization) Endpoints
# ============================================================================

class TIPOGenerateRequest(BaseModel):
    input_prompt: str
    model_name: Optional[str] = "KBlueLeaf/TIPO-500M"  # Model to use (auto-load if needed)
    tag_length: str = "short"  # very_short, short, long, very_long
    nl_length: str = "short"  # very_short, short, long, very_long
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 40
    max_new_tokens: int = 256
    ban_tags: str = ""  # Comma-separated list of tags to exclude from generation
    # Output formatting options
    category_order: Optional[List[str]] = None  # Order of categories in output
    enabled_categories: Optional[Dict[str, bool]] = None  # Which categories to include
    treat_as_nl: bool = False  # Treat input as natural language instead of tags

class TIPOLoadModelRequest(BaseModel):
    model_name: str = "KBlueLeaf/TIPO-500M"

@router.post("/tipo/load-model")
async def load_tipo_model(request: TIPOLoadModelRequest):
    """Load TIPO model for prompt optimization"""
    try:
        tipo_manager.load_model(request.model_name)
        return {
            "status": "success",
            "model_name": request.model_name,
            "loaded": tipo_manager.loaded
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tipo/generate")
async def generate_tipo_prompt(request: TIPOGenerateRequest):
    """Generate enhanced prompt using TIPO

    Args:
        input_prompt: Input prompt (tags or natural language)
        model_name: Model to use (will auto-load if not already loaded)
        tag_length: Length target for tags (very_short/short/long/very_long)
        nl_length: Length target for natural language
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_new_tokens: Maximum tokens to generate
        category_order: Optional order of categories in output
        enabled_categories: Optional dict of which categories to include

    Returns:
        Generated enhanced prompt with parsed structure
    """
    # Track if we auto-loaded the model (to unload it after)
    auto_loaded = False

    try:
        # Auto-load model if not loaded or if different model requested
        if not tipo_manager.loaded or (tipo_manager.model_name != request.model_name):
            print(f"[TIPO] Auto-loading model: {request.model_name}")
            tipo_manager.load_model(request.model_name)
            auto_loaded = True

        # Generate TIPO output
        raw_output = tipo_manager.generate_prompt(
            input_prompt=request.input_prompt,
            tag_length=request.tag_length,
            nl_length=request.nl_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
            ban_tags=request.ban_tags,
            treat_as_nl=request.treat_as_nl
        )

        # Check if using tipo-kgen (returns dict)
        if hasattr(tipo_manager, 'tipo_runner') and isinstance(raw_output, dict):
            # tipo-kgen returns a dict, format according to user preferences
            print("[TIPO] Using tipo-kgen mode: formatting result dict")

            # Merge input tags with TIPO output to preserve user input
            merged_output = tipo_manager.merge_kgen_with_input(request.input_prompt, raw_output)

            if request.category_order and request.enabled_categories:
                formatted_prompt = tipo_manager.format_kgen_result(
                    merged_output,
                    request.category_order,
                    request.enabled_categories
                )
            else:
                # Default order if not specified
                default_order = ['rating', 'quality', 'special', 'copyright', 'characters', 'artist', 'general', 'meta', 'short_nl', 'long_nl']
                default_enabled = {cat: True for cat in default_order}
                formatted_prompt = tipo_manager.format_kgen_result(
                    merged_output,
                    default_order,
                    default_enabled
                )
        else:
            # Transformers-only mode: need to parse, merge, and format
            print("[TIPO] Using transformers mode: parsing and formatting output")

            # Parse input tags to preserve them
            input_parsed = tipo_manager.parse_input_tags(request.input_prompt)

            # Parse TIPO output into structured format
            tipo_parsed = tipo_manager.parse_tipo_output(raw_output)

            # Merge input tags with TIPO generated tags
            merged_parsed = tipo_manager.merge_tags(input_parsed, tipo_parsed)

            # Format according to user preferences
            if request.category_order and request.enabled_categories:
                formatted_prompt = tipo_manager.format_prompt_from_parsed(
                    merged_parsed,
                    request.category_order,
                    request.enabled_categories
                )
            else:
                # Default order if not specified - following TIPO's category structure
                default_order = ['special', 'quality', 'rating', 'artist', 'copyright', 'characters', 'meta', 'general']
                default_enabled = {cat: True for cat in default_order}
                formatted_prompt = tipo_manager.format_prompt_from_parsed(
                    merged_parsed,
                    default_order,
                    default_enabled
                )

        # ALWAYS auto-unload model to free VRAM (TIPO should not occupy VRAM during image generation)
        print(f"[TIPO] Auto-unloading model to free VRAM (auto_loaded={auto_loaded})")
        tipo_manager.unload_model()

        # Build response
        response = {
            "status": "success",
            "original_prompt": request.input_prompt,
            "raw_output": raw_output,
            "generated_prompt": formatted_prompt
        }

        # Add parsed data only if using transformers mode
        if not hasattr(tipo_manager, 'tipo_runner'):
            response["parsed"] = merged_parsed

        return response
    except Exception as e:
        # Make sure to unload if we auto-loaded and hit an error
        if auto_loaded and tipo_manager.loaded:
            print(f"[TIPO] Auto-unloading model after error")
            tipo_manager.unload_model()

        print(f"[API] TIPO generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tipo/status")
async def get_tipo_status():
    """Get TIPO model status"""
    return {
        "loaded": tipo_manager.loaded,
        "model_name": tipo_manager.model_name,
        "device": tipo_manager.device
    }

@router.post("/tipo/unload")
async def unload_tipo_model():
    """Unload TIPO model from memory"""
    try:
        tipo_manager.unload_model()
        return {"status": "success", "loaded": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cancel")
async def cancel_generation():
    """Cancel ongoing generation"""
    try:
        pipeline_manager.cancel_generation()
        return {"status": "success", "message": "Generation cancellation requested"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/port-info")
async def get_port_info():
    """Get backend server port information"""
    import json
    port_info_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".port_info")

    try:
        if os.path.exists(port_info_file):
            with open(port_info_file, 'r') as f:
                port_info = json.load(f)
                return port_info
    except Exception as e:
        print(f"[API] Error reading port info: {e}")

    # Fallback to default
    return {"port": 8000, "host": "localhost"}


# ============================================================================
# Image Tagger Endpoints
# ============================================================================

class TaggerRequest(BaseModel):
    image_base64: str
    gen_threshold: float = 0.45
    char_threshold: float = 0.45
    model_version: str = "cl_tagger_1_02"
    auto_unload: bool = True
    # Individual category thresholds (optional, overrides gen_threshold/char_threshold)
    thresholds: Optional[Dict[str, float]] = None

class TaggerLoadModelRequest(BaseModel):
    model_path: Optional[str] = None
    tag_mapping_path: Optional[str] = None
    use_gpu: bool = True
    use_huggingface: bool = True
    repo_id: str = "cella110n/cl_tagger"
    model_version: str = "cl_tagger_1_02"

@router.post("/tagger/load-model")
async def load_tagger_model(request: TaggerLoadModelRequest):
    """Load image tagger model

    Args:
        model_path: Path to ONNX model file (optional if use_huggingface=True)
        tag_mapping_path: Path to tag mapping JSON file (optional if use_huggingface=True)
        use_gpu: Whether to use GPU acceleration
        use_huggingface: Whether to download from Hugging Face Hub (default: True)
        repo_id: Hugging Face repository ID (default: cella110n/cl_tagger)
        model_version: Model version subdirectory (default: cl_tagger_1_02)
    """
    try:
        tagger_manager.load_model(
            model_path=request.model_path,
            tag_mapping_path=request.tag_mapping_path,
            use_gpu=request.use_gpu,
            use_huggingface=request.use_huggingface,
            repo_id=request.repo_id,
            model_version=request.model_version
        )
        return {
            "status": "success",
            "loaded": tagger_manager.loaded,
            "model_path": tagger_manager.model_path,
            "tag_mapping_path": tagger_manager.tag_mapping_path
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tagger/predict")
async def predict_tags(request: TaggerRequest):
    """Predict tags for an image

    Args:
        image_base64: Base64 encoded image
        gen_threshold: Threshold for general tags (default: 0.45)
        char_threshold: Threshold for character/copyright/artist tags (default: 0.45)
        model_version: Model version to use (default: cl_tagger_1_02)
        auto_unload: Whether to unload model after prediction to free VRAM (default: True)

    Returns:
        Dictionary with categorized tags and confidences
    """
    try:
        # Decode base64 image
        import base64
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))

        # Predict tags (auto-loads model if needed)
        predictions = tagger_manager.predict(
            image,
            gen_threshold=request.gen_threshold,
            char_threshold=request.char_threshold,
            model_version=request.model_version,
            auto_unload=request.auto_unload,
            thresholds=request.thresholds
        )

        return {
            "status": "success",
            "predictions": predictions
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tagger/status")
async def get_tagger_status():
    """Get tagger model status"""
    return {
        "loaded": tagger_manager.loaded,
        "model_path": tagger_manager.model_path,
        "tag_mapping_path": tagger_manager.tag_mapping_path,
        "model_version": tagger_manager.model_version
    }

@router.post("/tagger/unload")
async def unload_tagger_model():
    """Unload tagger model"""
    try:
        tagger_manager.unload_model()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SigLIP2 Tagger Endpoints
# ============================================================================

from core.tagger.siglip2_inference_manager import get_siglip2_inference_manager
from core.tagger.tag_metrics_accumulator import TagMetricsAccumulator


def _siglip2_unload_on_training_resume() -> None:
    """Free the SigLIP2 inference model when training resumes after a generation/
    inference grace period.

    A tagger inference model loaded during training (for vocabulary / calibration
    setup, or as the fallback when use_training_model is off) would otherwise keep
    occupying VRAM after the GPU is handed back to training — the actual forward
    pass uses the live training model, so the loaded inference copy is no longer
    needed once we return to training. Registered with the GPU coordinator so it
    fires exactly at that hand-back point.
    """
    try:
        mgr = get_siglip2_inference_manager()
        if getattr(mgr, "model", None) is not None or getattr(mgr, "onnx_session", None) is not None:
            mgr.unload()  # moves to CPU, drops the model, and empty_caches
            print("[SigLIP2] Inference model unloaded on training resume (VRAM returned to training)")
    except Exception as _e:   # noqa: BLE001 — never block training resume
        print(f"[SigLIP2] resume-unload skipped: {_e}")


try:
    from core.gpu_coordinator import gpu_coordinator as _gpu_coordinator
    _gpu_coordinator.register_resume_callback(_siglip2_unload_on_training_resume)
except Exception as _e:   # noqa: BLE001
    print(f"[SigLIP2] could not register resume-unload callback: {_e}")


class SigLIP2LoadRequest(BaseModel):
    checkpoint_path: str
    vision_encoder_path: str = ""
    vocab_path: str = ""
    lora_rank: int = 32
    lora_alpha: float = 16.0

class SigLIP2PredictRequest(BaseModel):
    image_base64: str
    threshold: float = 0.5
    # Conditional inference parameters
    known_tags_pos: Optional[List[str]] = None
    known_tags_neg: Optional[List[str]] = None
    context_method: str = "none"      # "none" | "head_sim" | "lr_matrix"
    context_lambda: float = 0.5
    # When True, attempt to use the currently-training model instead of the
    # loaded inference model.  Falls back to the inference model automatically
    # if no training is active or the training model is offloaded to CPU.
    use_training_model: bool = False
    # Legacy: apply calibration to both filtering and display.
    use_calibration: bool = False
    # New: filter by per-tag best_thr (raw sigmoid), display probs are still raw.
    use_per_tag_threshold: bool = False
    # New: display calibrated probs in output while filtering uses raw sigmoid + best_thr.
    display_calibration: bool = False
    # Quality filters for per-tag threshold mode.
    # min_best_thr: clamp best_thr to this floor (suppresses noise-level FPs from untrained tags).
    # min_best_f1: skip tags whose best_f1 is below this (exclude effectively-untrained tags).
    min_best_thr: float = 0.30
    min_best_f1: float = 0.05
    # OOD detection: raise threshold for Character/Copyright when image is OOD.
    # Only active when use_per_tag_threshold=True and an OOD reference is loaded.
    use_ood_detection: bool = False

class SigLIP2BuildOodReferenceRequest(BaseModel):
    image_dir: str   # directory to walk for in-distribution images
    max_images: int = 2000

class SigLIP2MergeLoRARequest(BaseModel):
    output_path: str

class SigLIP2ExportONNXRequest(BaseModel):
    output_path: str
    max_num_patches: int = 256
    strip_unknown_tags: bool = False
    # Also emit a WebGPU-loadable split version (sub-models each under ~2GB).
    also_split: bool = False
    split_max_bytes: int = 1_900_000_000
    # Name the output "model.onnx" (+ model_*.json) instead of the checkpoint stem.
    use_model_stem: bool = False


class SigLIP2ExtractEncoderRequest(BaseModel):
    repo_id: str
    output_path: str
    encoder_type: str = "vision"  # "vision" | "text"


@router.post("/tagger/siglip2/load")
async def siglip2_load(request: SigLIP2LoadRequest):
    """Load a SigLIP2 tagger checkpoint (full or LoRA, auto-detected)."""
    try:
        # Pause any active tagger training before loading a second model onto
        # the same GPU (SigLIP2-SO400M bf16 ≈ 1.7 GB + head weights).
        from core.gpu_coordinator import gpu_coordinator
        async with gpu_coordinator.generation_slot(estimated_peak_gb=2.5, timeout=60.0):
            mgr = get_siglip2_inference_manager()
            result = mgr.load_model(
                checkpoint_path=request.checkpoint_path,
                vocab_path=request.vocab_path,
                vision_encoder_path=request.vision_encoder_path,
                lora_rank=request.lora_rank,
                lora_alpha=request.lora_alpha,
            )
        return result
    except Exception as e:
        import traceback
        print(f"[SigLIP2Load] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Tracks whether the "no active training, falling back" notice has already been
# logged for the current no-training stretch (re-armed when a training model
# becomes available), so it isn't printed on every single inference request.
_SIGLIP2_TRAIN_FALLBACK_LOGGED = False


@router.post("/tagger/siglip2/predict")
async def siglip2_predict(request: SigLIP2PredictRequest):
    """Run inference with the loaded SigLIP2 model.

    When use_training_model=True, the currently-training model is used instead
    of the loaded inference model.  Falls back to the inference model if no
    training is active or the training model is temporarily offloaded.
    """
    global _SIGLIP2_TRAIN_FALLBACK_LOGGED
    try:
        import base64
        from core.gpu_coordinator import gpu_coordinator
        image_bytes = base64.b64decode(request.image_base64)

        # Try training-model path when requested
        if request.use_training_model:
            handle = gpu_coordinator.get_active_tagger_handle()
            if handle is not None:
                _SIGLIP2_TRAIN_FALLBACK_LOGGED = False  # available again → re-arm the log
                # Borrow per-tag thresholds / calibration table / OOD reference
                # from the loaded inference checkpoint so the training-model path
                # supports per-tag inference and OOD detection identically to the
                # inference model. Metrics are keyed by tag name, so a training
                # head that has expanded beyond the checkpoint's vocab stays
                # correct (new tags fall back to the global threshold).
                import functools
                _assist = get_siglip2_inference_manager().build_training_assist(
                    want_per_tag=request.use_per_tag_threshold,
                    want_calibration=request.use_calibration,
                    want_ood=request.use_ood_detection,
                )
                _predict = functools.partial(
                    handle.predict, image_bytes, request.threshold,
                    assist=_assist,
                    use_per_tag_threshold=request.use_per_tag_threshold,
                    min_best_thr=request.min_best_thr,
                    min_best_f1=request.min_best_f1,
                    use_ood_detection=request.use_ood_detection,
                    use_calibration=request.use_calibration,
                )

                # Training model is on CUDA — use it directly (it shares the GPU
                # with the coordinator's grace period, no need for generation_slot).
                try:
                    result = await asyncio.get_event_loop().run_in_executor(None, _predict)
                    return result
                except RuntimeError as _e:
                    # Model was offloaded between can_predict() and predict() — fall through
                    print(f"[SigLIP2Predict] Training model unavailable ({_e}); "
                          f"falling back to inference model")
            elif not _SIGLIP2_TRAIN_FALLBACK_LOGGED:
                # Log once per no-active-training stretch; re-armed above when a
                # training model becomes available again. Avoids one line per
                # inference when the UI keeps use_training_model on with no run.
                print("[SigLIP2Predict] use_training_model=True but no active training; "
                      "falling back to inference model (suppressing repeats)")
                _SIGLIP2_TRAIN_FALLBACK_LOGGED = True

        # Standard inference-model path
        mgr = get_siglip2_inference_manager()
        async with gpu_coordinator.generation_slot(estimated_peak_gb=2.5, timeout=60.0):
            result = mgr.predict(
                image_bytes=image_bytes,
                threshold=request.threshold,
                known_tags_pos=request.known_tags_pos,
                known_tags_neg=request.known_tags_neg,
                context_method=request.context_method,
                context_lambda=request.context_lambda,
                use_calibration=request.use_calibration,
                use_per_tag_threshold=request.use_per_tag_threshold,
                min_best_thr=request.min_best_thr,
                min_best_f1=request.min_best_f1,
                display_calibration=request.display_calibration,
                use_ood_detection=request.use_ood_detection,
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tagger/siglip2/build-ood-reference")
async def siglip2_build_ood_reference(request: SigLIP2BuildOodReferenceRequest):
    """Build OOD reference distribution from an in-distribution image directory.

    Walks *image_dir* recursively, extracts CLS embeddings from up to
    *max_images* images, fits a multivariate Gaussian with Ledoit-Wolf
    shrinkage, and saves the result as ``{onnx_base}_ood_ref.npz``.
    """
    mgr = get_siglip2_inference_manager()
    if not mgr.status.get("loaded"):
        raise HTTPException(status_code=400, detail="No model loaded")
    if mgr.model_type != "onnx":
        raise HTTPException(status_code=400, detail="OOD reference requires an ONNX model")

    image_dir = request.image_dir.strip().strip('"').strip("'")
    if not os.path.isdir(image_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {image_dir}")

    # Collect image paths
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))

    if len(paths) == 0:
        raise HTTPException(status_code=400, detail="No images found in directory")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, mgr.build_ood_reference, paths, None, request.max_images
        )
        return result
    except Exception as e:
        import traceback
        print(f"[BuildOodReference] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tagger/siglip2/status")
async def siglip2_status():
    """Return loaded model status."""
    return get_siglip2_inference_manager().status


@router.get("/tagger/siglip2/vocabulary")
async def siglip2_vocabulary():
    """Return the vocabulary of the currently loaded SigLIP2 model."""
    mgr = get_siglip2_inference_manager()
    if not mgr.status.get("loaded"):
        raise HTTPException(status_code=400, detail="No model loaded")
    vocab = mgr.vocabulary
    if not vocab:
        raise HTTPException(status_code=404, detail="Vocabulary not available")
    return vocab


@router.get("/tagger/siglip2/tag-metrics")
async def siglip2_tag_metrics():
    """Return per-tag metrics from the _tag_metrics.npz saved alongside the loaded checkpoint.

    Response is in columnar format: parallel arrays of length n_tags.
    NaN values are serialized as null.
    """
    mgr = get_siglip2_inference_manager()
    if not mgr.status.get("loaded"):
        raise HTTPException(status_code=400, detail="No model loaded")
    path = mgr.get_tag_metrics_path()
    if path is None:
        raise HTTPException(status_code=404, detail="tag_metrics.npz not found for this checkpoint")

    try:
        data = TagMetricsAccumulator.load(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tag_metrics.npz: {e}")

    vocab = mgr.vocabulary or {}
    idx_to_tag = vocab.get("idx_to_tag", {})
    tag_to_category = vocab.get("tag_to_category", {})
    n_pos_arr = data.get("n_pos")
    if n_pos_arr is None:
        raise HTTPException(status_code=500, detail="tag_metrics.npz missing n_pos array")
    V = int(n_pos_arr.shape[0])
    tag_names_arr = data.get("tag_names", None)

    def to_list(key: str):
        arr = data.get(key)
        if arr is None:
            return [None] * V
        return [None if (v != v) else float(v) for v in arr.tolist()]

    tag_names: list = []
    categories: list = []
    for i in range(V):
        if tag_names_arr is not None and i < len(tag_names_arr):
            name = str(tag_names_arr[i])
        else:
            name = idx_to_tag.get(str(i), f"tag_{i}")
        tag_names.append(name)
        categories.append(tag_to_category.get(name, "Unknown"))

    def _scalar(v, default=0.0):
        # npz scalars are saved as 1-element arrays (e.g. np.array([x])); numpy 2.x
        # rejects int()/float() on non-0-d arrays, so unwrap via .item(). Some keys
        # are already unwrapped to Python scalars by TagMetricsAccumulator.load().
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return v
        try:
            return v.item()
        except (AttributeError, ValueError):
            try:
                return v.reshape(-1)[0].item()
            except Exception:
                return default

    return {
        "n_tags":       V,
        "total_images": int(_scalar(data.get("total_images"), 0)),
        "hard_lo":      float(_scalar(data.get("hard_lo"), 0.25)),
        "hard_hi":      float(_scalar(data.get("hard_hi"), 0.75)),
        "tag_names":    tag_names,
        "categories":   categories,
        "n_pos":        to_list("n_pos"),
        "n_neg":        to_list("n_neg"),
        "global_freq":  to_list("global_freq"),
        "hard_rate":    to_list("hard_rate"),
        "fp_rate_50":   to_list("fp_rate_50"),
        "fn_rate_50":   to_list("fn_rate_50"),
        "best_f1":      to_list("best_f1"),
        "best_thr":     to_list("best_thr"),
    }


@router.get("/tagger/siglip2/checkpoint-meta")
async def siglip2_checkpoint_meta(path: str):
    """Read _metadata.json alongside the given safetensors checkpoint path.
    Returns lora_rank, lora_alpha, num_tags, training_method and any other fields present."""
    import json as _json
    meta_path = path.replace(".safetensors", "_metadata.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(status_code=404, detail=f"Metadata file not found: {meta_path}")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tagger/siglip2/unload")
async def siglip2_unload():
    """Unload the SigLIP2 model."""
    get_siglip2_inference_manager().unload()
    return {"status": "ok"}


class SigLIP2CalibrationRequest(BaseModel):
    method: str = "jeffreys"            # "jeffreys" | "beta_bb"
    eps: float = 0.5                    # Jeffreys epsilon (used when method="jeffreys")
    prior_strength: float = 10.0        # Beta-BB prior strength (used when method="beta_bb")


@router.get("/tagger/siglip2/calibration")
async def siglip2_calibration_get():
    """Return current calibration settings for the loaded SigLIP2 model."""
    mgr = get_siglip2_inference_manager()
    if not mgr.status.get("loaded"):
        raise HTTPException(status_code=400, detail="No model loaded")
    return {
        "method":         mgr.calib_method,
        "eps":            mgr.calib_eps,
        "prior_strength": mgr.calib_prior_strength,
        "has_tag_metrics": mgr.tag_metrics is not None,
    }


@router.post("/tagger/siglip2/calibration")
async def siglip2_calibration_set(request: SigLIP2CalibrationRequest):
    """Recompute the in-memory calibration table with the specified method/params."""
    mgr = get_siglip2_inference_manager()
    if not mgr.status.get("loaded"):
        raise HTTPException(status_code=400, detail="No model loaded")
    if mgr.tag_metrics is None:
        raise HTTPException(status_code=404, detail="tag_metrics not available for this checkpoint")
    if request.method not in ("jeffreys", "beta_bb"):
        raise HTTPException(status_code=422, detail="method must be 'jeffreys' or 'beta_bb'")
    ok = mgr.recompute_calibration_table(
        method=request.method,
        eps=request.eps,
        prior_strength=request.prior_strength,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Calibration recomputation failed")
    return {
        "status":         "ok",
        "method":         mgr.calib_method,
        "eps":            mgr.calib_eps,
        "prior_strength": mgr.calib_prior_strength,
    }


@router.post("/tagger/siglip2/merge-lora")
async def siglip2_merge_lora(request: SigLIP2MergeLoRARequest):
    """Merge LoRA weights into the vision encoder and save as a full model."""
    try:
        mgr = get_siglip2_inference_manager()
        saved_path = mgr.merge_lora_and_save(request.output_path)
        return {"saved_path": saved_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tagger/siglip2/export-onnx")
async def siglip2_export_onnx(request: SigLIP2ExportONNXRequest):
    """Export the loaded model to ONNX format."""
    try:
        mgr = get_siglip2_inference_manager()
        onnx_path, vocab_path = mgr.export_onnx(
            output_path=request.output_path,
            max_num_patches=request.max_num_patches,
            strip_unknown_tags=request.strip_unknown_tags,
            also_split=request.also_split,
            split_max_bytes=request.split_max_bytes,
            use_model_stem=request.use_model_stem,
        )
        return {"saved_path": onnx_path, "vocab_path": vocab_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tagger/siglip2/extract-encoder")
async def siglip2_extract_encoder(request: SigLIP2ExtractEncoderRequest):
    """Extract vision or text encoder from a HuggingFace repo and save as safetensors."""
    try:
        from core.tagger.siglip2_extractor import extract_vision_encoder, extract_text_encoder
        if request.encoder_type == "text":
            result = extract_text_encoder(request.repo_id, request.output_path)
        else:
            result = extract_vision_encoder(request.repo_id, request.output_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Tagger Browser Endpoints
# ---------------------------------------------------------------------------
# Security design:
#   - _browser_root is stored in server RAM only (never written to disk / git).
#   - All client-facing responses use rel_path only; absolute paths are never
#     sent to the frontend.
#   - _resolve_browser_path() rejects path-traversal attempts before any I/O.
# ---------------------------------------------------------------------------

_BROWSER_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Active root directory for the browser session.
# Lives in server RAM; cleared on process restart. Never persisted.
_browser_root: Optional[str] = None


def _resolve_browser_path(rel_path: str) -> str:
    """Resolve rel_path under _browser_root and validate it stays within root.

    Raises 400 if no root is set, 403 on path-traversal attempt.
    """
    import os as _os
    if _browser_root is None:
        raise HTTPException(status_code=400, detail="No browser directory set. Call set-directory first.")
    # Normalise: collapse '..' sequences, strip leading separators
    norm = _os.path.normpath(rel_path).lstrip(_os.sep).lstrip("/")
    if norm.startswith(".."):
        raise HTTPException(status_code=403, detail="Path traversal not allowed")
    abs_path = _os.path.join(_browser_root, norm)
    # Final containment check (handles edge cases like symlinks expanding outside)
    real_root = _os.path.realpath(_browser_root)
    real_abs  = _os.path.realpath(abs_path)
    if real_abs != real_root and not real_abs.startswith(real_root + _os.sep):
        raise HTTPException(status_code=403, detail="Path outside allowed directory")
    return abs_path


def _set_browser_root(path: str) -> str:
    """Validate and set _browser_root. Returns the normalised absolute path."""
    import os as _os
    global _browser_root
    abs_dir = _os.path.abspath(path)
    if not _os.path.isdir(abs_dir):
        raise HTTPException(status_code=400, detail="Invalid directory")
    _browser_root = abs_dir
    return abs_dir


class BrowserSetDirectoryRequest(BaseModel):
    dir: str


@router.post("/tagger/browser/set-directory")
async def browser_set_directory(req: BrowserSetDirectoryRequest):
    """Set the active browser root directory (server RAM only; never persisted).

    Returns only the folder display name, not the full path.
    """
    import os as _os
    root = _set_browser_root(req.dir)
    return {"ok": True, "display_name": _os.path.basename(root)}


@router.post("/tagger/browser/pick-directory")
async def browser_pick_directory():
    """Open a native OS folder-picker dialog (tkinter), set as browser root,
    and return only the folder display name.

    Absolute path is stored in server RAM only and never sent to the client.
    Only meaningful when server runs on the same machine as the browser.
    """
    import asyncio
    import os as _os
    from concurrent.futures import ThreadPoolExecutor

    def _pick():
        try:
            import tkinter as _tk
            from tkinter import filedialog as _fd
            root = _tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", True)
            selected = _fd.askdirectory(title="フォルダを選択")
            root.destroy()
            return selected or None
        except Exception as _e:
            raise HTTPException(status_code=500, detail=f"Directory picker failed: {_e}")

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        selected = await loop.run_in_executor(pool, _pick)

    if selected is None:
        return {"ok": False, "display_name": None}

    root = _set_browser_root(selected)
    return {"ok": True, "display_name": _os.path.basename(root)}


@router.get("/tagger/browser/list")
async def browser_list(recursive: bool = False, include_tags: bool = False):
    """List image files under the active browser root.

    Response contains only rel_path (relative to root), has_tags, and mtime.
    Absolute paths are never sent to the client.
    When include_tags=True, each entry also includes a 'tags' list read from the sidecar .txt.
    """
    import os as _os
    if _browser_root is None:
        raise HTTPException(status_code=400, detail="No browser directory set")
    results = []
    if recursive:
        walker = _os.walk(_browser_root)
    else:
        try:
            entries = sorted(_os.listdir(_browser_root))
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
        walker = [(_browser_root, [], entries)]
    for dirpath, _, files in walker:
        for f in sorted(files):
            ext = _os.path.splitext(f)[1].lower()
            if ext not in _BROWSER_IMAGE_EXTS:
                continue
            abs_path = _os.path.join(dirpath, f)
            txt_path = _os.path.splitext(abs_path)[0] + ".txt"
            has_tags = _os.path.isfile(txt_path)
            entry = {
                "rel_path": _os.path.relpath(abs_path, _browser_root),
                "has_tags": has_tags,
                "mtime": _os.path.getmtime(abs_path),
            }
            if include_tags:
                if has_tags:
                    try:
                        with open(txt_path, "r", encoding="utf-8") as fh:
                            raw = fh.read().strip()
                        entry["tags"] = [t.strip() for t in raw.split(",") if t.strip()]
                    except Exception:
                        entry["tags"] = []
                else:
                    entry["tags"] = []
            results.append(entry)
    return {"images": results}


_BROWSER_IMAGE_CACHE_HEADERS = {
    "Cache-Control": "private, max-age=3600",
}

# In-memory LRU cache for resized browser images.
# Key: (abs_path, size, mtime) — mtime invalidates stale entries automatically.
# RAM-only (consistent with security policy: no persistent browser-root data).
from collections import OrderedDict as _OrderedDict

_browser_img_cache: "_OrderedDict[tuple, bytes]" = _OrderedDict()
_BROWSER_IMG_CACHE_MAX = 200  # ~200 * ~300KB ≈ 60MB worst case


def _img_cache_get(key: tuple) -> "bytes | None":
    if key in _browser_img_cache:
        _browser_img_cache.move_to_end(key)
        return _browser_img_cache[key]
    return None


def _img_cache_put(key: tuple, data: bytes) -> None:
    _browser_img_cache[key] = data
    _browser_img_cache.move_to_end(key)
    while len(_browser_img_cache) > _BROWSER_IMG_CACHE_MAX:
        _browser_img_cache.popitem(last=False)


@router.get("/tagger/browser/image")
async def browser_image(rel_path: str, size: int = 0):
    """Serve an image by rel_path (relative to active browser root).

    size=0: original file; size=N: JPEG at NxN max (keep aspect).
    Resized results are kept in an in-memory LRU cache (200 entries) keyed by
    (abs_path, size, mtime) so repeated requests skip PIL encode entirely.
    Cache-Control: private, max-age=3600 for browser-side HTTP caching.
    """
    import os as _os
    abs_path = _resolve_browser_path(rel_path)
    if not _os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not found")
    if size > 0:
        import io
        from PIL import Image as _Image
        mtime = _os.path.getmtime(abs_path)
        cache_key = (abs_path, size, mtime)
        cached = _img_cache_get(cache_key)
        if cached is not None:
            return Response(
                content=cached,
                media_type="image/jpeg",
                headers=_BROWSER_IMAGE_CACHE_HEADERS,
            )
        img = _Image.open(abs_path).convert("RGB")
        img.thumbnail((size, size), _Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
        _img_cache_put(cache_key, data)
        return Response(
            content=data,
            media_type="image/jpeg",
            headers=_BROWSER_IMAGE_CACHE_HEADERS,
        )
    return FileResponse(abs_path, headers=_BROWSER_IMAGE_CACHE_HEADERS)


@router.get("/tagger/browser/tags")
async def browser_get_tags(rel_path: str):
    """Read .txt sidecar file for rel_path. Returns tags list and raw text."""
    import os as _os
    abs_path = _resolve_browser_path(rel_path)
    txt = _os.path.splitext(abs_path)[0] + ".txt"
    if not _os.path.isfile(txt):
        return {"tags": [], "raw": ""}
    with open(txt, "r", encoding="utf-8") as f:
        content = f.read().strip()
    tags = [t.strip() for t in content.split(",") if t.strip()]
    return {"tags": tags, "raw": content}


class BrowserSaveTagsRequest(BaseModel):
    rel_path: str
    tags: List[str]


@router.post("/tagger/browser/tags")
async def browser_save_tags(req: BrowserSaveTagsRequest):
    """Write tags to .txt sidecar file (comma-separated)."""
    import os as _os
    abs_path = _resolve_browser_path(req.rel_path)
    txt = _os.path.splitext(abs_path)[0] + ".txt"
    content = ", ".join(req.tags)
    with open(txt, "w", encoding="utf-8") as f:
        f.write(content)
    return {"saved": True}


class BrowserBatchInferRequest(BaseModel):
    rel_paths: List[str]
    overwrite: bool = False
    use_ood_detection: bool = False


@router.post("/tagger/browser/batch-infer")
async def browser_batch_infer(req: BrowserBatchInferRequest):
    """Batch inference with SSE progress streaming. Writes .txt sidecar files."""
    import json as _json
    import os as _os
    mgr = get_siglip2_inference_manager()
    if not mgr.status.get("loaded"):
        raise HTTPException(status_code=400, detail="No model loaded")

    # Resolve all paths up-front; abort immediately on traversal attempt
    resolved = []
    for rp in req.rel_paths:
        resolved.append((_resolve_browser_path(rp), rp))

    async def generate():
        total = len(resolved)
        for i, (abs_path, rel) in enumerate(resolved):
            txt = _os.path.splitext(abs_path)[0] + ".txt"
            if not req.overwrite and _os.path.isfile(txt):
                yield f"data: {_json.dumps({'type': 'skip', 'i': i, 'total': total, 'rel_path': rel})}\n\n"
                continue
            try:
                from PIL import Image as _Image
                img = _Image.open(abs_path).convert("RGB")
                result = mgr.predict(img, use_ood_detection=req.use_ood_detection)
                tags = result.get("tags", [])
                with open(txt, "w", encoding="utf-8") as f:
                    f.write(", ".join(tags))
                yield f"data: {_json.dumps({'type': 'done', 'i': i, 'total': total, 'rel_path': rel, 'n_tags': len(tags)})}\n\n"
            except Exception as e:
                yield f"data: {_json.dumps({'type': 'error', 'i': i, 'total': total, 'rel_path': rel, 'error': str(e)})}\n\n"
        yield f"data: {_json.dumps({'type': 'complete', 'total': total})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/system/gpu-stats")
async def get_gpu_stats():
    """Get GPU statistics (VRAM, utilization, temperature)"""
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "available": False,
                "error": "CUDA not available"
            }

        stats = []

        # Try nvidia-smi first (most reliable method)
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 8:
                        index = int(parts[0])
                        name = parts[1]
                        temp = int(parts[2]) if parts[2] and parts[2] != '[N/A]' else None
                        gpu_util = int(parts[3]) if parts[3] and parts[3] != '[N/A]' else None
                        mem_util = int(parts[4]) if parts[4] and parts[4] != '[N/A]' else None
                        mem_total = float(parts[5]) / 1024 if parts[5] else 0  # Convert MiB to GiB
                        mem_used = float(parts[6]) / 1024 if parts[6] else 0  # Convert MiB to GiB
                        power = float(parts[7]) if parts[7] and parts[7] != '[N/A]' else None

                        vram_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

                        gpu_stats = {
                            "index": index,
                            "name": name,
                            "vram_used_gb": round(mem_used, 2),
                            "vram_total_gb": round(mem_total, 2),
                            "vram_percent": round(vram_percent, 1),
                            "gpu_utilization": gpu_util,
                            "temperature": temp,
                            "power_watts": round(power, 1) if power else None,
                        }
                        stats.append(gpu_stats)

                # print(f"[GPU Stats] nvidia-smi: {len(stats)} GPU(s) found")
                return {
                    "available": True,
                    "gpus": stats
                }

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            # print(f"[GPU Stats] nvidia-smi failed ({e}), falling back to torch")
            pass

        # Fallback to torch-only stats
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            mem_total = props.total_memory / (1024 ** 3)

            stats.append({
                "index": i,
                "name": props.name,
                "vram_used_gb": round(mem_allocated, 2),
                "vram_total_gb": round(mem_total, 2),
                "vram_percent": round((mem_allocated / mem_total) * 100, 1),
                "gpu_utilization": None,
                "temperature": None,
                "power_watts": None,
            })

        return {
            "available": True,
            "gpus": stats
        }

    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return {
            "available": False,
            "error": str(e)
        }

# Authentication endpoints
@router.get("/auth/status")
async def get_auth_status():
    """Get authentication status"""
    return AuthStatusResponse(auth_enabled=settings.auth_enabled)

@router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint - returns JWT token if credentials are valid"""
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=400,
            detail="Authentication is not enabled"
        )

    if not verify_credentials(request.username, request.password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )

    access_token = create_access_token(request.username)
    return LoginResponse(access_token=access_token)

@router.get("/auth/verify")
async def verify_auth(username: str = Depends(require_auth)):
    """Verify authentication token"""
    return {"authenticated": True, "username": username}

@router.get("/download/{filename}")
async def download_image(filename: str, include_metadata: bool = False):
    """Download image with optional metadata removal

    Args:
        filename: The filename of the image in the outputs directory
        include_metadata: If True, keep metadata; if False, strip metadata (default: False)

    Returns:
        Image file with or without metadata
    """
    try:
        # Validate filename (prevent directory traversal)
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Construct full path
        filepath = os.path.join(settings.outputs_dir, filename)

        # Check if file exists
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Image not found")

        # Video/audio rows have no PIL-decodable metadata to strip -- Image.open
        # below raises on them. Pass those straight through as a byte-exact
        # download instead (also the path for a video_lossless FFV1-in-mkv
        # master; include_metadata is a no-op for non-image files).
        _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff")
        if not filename.lower().endswith(_IMAGE_EXTS):
            return FileResponse(filepath, filename=filename)

        # Read the image
        image = Image.open(filepath)

        # Create BytesIO buffer
        buffer = io.BytesIO()

        if include_metadata:
            # Save with metadata (if it exists)
            if hasattr(image, 'info') and 'pnginfo' in image.info:
                # Preserve existing metadata
                from PIL import PngImagePlugin
                metadata = PngImagePlugin.PngInfo()
                for key, value in image.text.items():
                    metadata.add_text(key, value)
                image.save(buffer, format="PNG", pnginfo=metadata)
            else:
                # No metadata to preserve, just save normally
                image.save(buffer, format="PNG")
        else:
            # Strip metadata by saving without pnginfo
            image.save(buffer, format="PNG")

        # Get bytes and return as response
        buffer.seek(0)
        image_bytes = buffer.getvalue()

        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error downloading image: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")


# ============================================================
# Dataset Management Endpoints
# ============================================================

class DatasetCreateRequest(BaseModel):
    name: str
    path: str
    description: Optional[str] = None
    recursive: bool = True
    read_exif: bool = False

def _dataset_caption_item_counts(db: Session, dataset_id: int) -> Tuple[int, int]:
    """Return ``(items_with_tags, items_with_nl_captions)`` for a dataset.

    Both are DISTINCT image counts, not raw caption-row counts:
      - items_with_tags: images that have a danbooru-tags caption (is_tags_format).
      - items_with_nl_captions: images that have a natural-language caption — a
        TRAINING, non-tags, non-empty caption. Metadata fields (filenames,
        source.*, timestamps, urls) are excluded via field_category, so this is
        the number of images with an actual written caption, not metadata.
    """
    item_ids_subq = db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset_id)
    items_with_tags = db.query(DatasetCaption.item_id).filter(
        DatasetCaption.item_id.in_(item_ids_subq),
        DatasetCaption.is_tags_format == True,
    ).distinct().count()
    items_with_nl = db.query(DatasetCaption.item_id).filter(
        DatasetCaption.item_id.in_(item_ids_subq),
        DatasetCaption.is_tags_format == False,
        DatasetCaption.field_category == "training",
        DatasetCaption.content.isnot(None),
        func.trim(DatasetCaption.content) != "",
    ).distinct().count()
    return items_with_tags, items_with_nl


def update_dataset_statistics(dataset: Dataset, db: Session):
    """Update dataset statistics by counting items and captions.

    ``total_tags`` = images with danbooru tags; ``total_captions`` = images with
    a natural-language caption (metadata fields excluded). See
    _dataset_caption_item_counts.
    """
    total_items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset.id).count()
    total_tags, total_captions = _dataset_caption_item_counts(db, dataset.id)

    # Only update if values changed (avoid unnecessary writes)
    if (dataset.total_items != total_items
            or dataset.total_captions != total_captions
            or dataset.total_tags != total_tags):
        dataset.total_items = total_items
        dataset.total_captions = total_captions
        dataset.total_tags = total_tags
        db.commit()

@router.get("/datasets")
async def list_datasets(db: Session = Depends(get_datasets_db)):
    """List all datasets.

    Two-fold speed-up over the previous implementation:

    1. **Skip the per-dataset COUNT recompute.**  Returns the cached
       ``total_items`` / ``total_captions`` / ``total_tags`` columns
       directly.  The write paths that mutate items/captions
       (scan_dataset etc.) keep these in sync, and re-doing three
       COUNT-with-IN-subquery queries per dataset cost ~6 s for a 3M-item
       corpus.  The list endpoint is hit on every Dataset Manager open
       and shouldn't carry that overhead.  ``POST /datasets/{id}/scan``
       (or the pre-flight rescan modes on training start) refreshes the
       counts if drift is suspected.
    2. **Defer the ``tag_statistics`` column.**  This column can be tens
       of MB per dataset (per-tag count across the full vocabulary) and
       totalled ~68 MB across all 16 datasets in the user's workspace.
       The list UI doesn't render it; the detail endpoint
       (``GET /datasets/{id}``) returns the full payload including stats.
    """
    from sqlalchemy.orm import defer
    try:
        datasets = (
            db.query(Dataset)
            .options(defer(Dataset.tag_statistics))
            .order_by(Dataset.created_at.desc())
            .all()
        )
        return {
            "datasets": [d.to_dict(include_tag_statistics=False) for d in datasets],
            "total": len(datasets),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/datasets", status_code=201)
async def create_dataset(request: DatasetCreateRequest, db: Session = Depends(get_datasets_db)):
    """Create a new dataset"""
    try:
        existing = db.query(Dataset).filter(Dataset.name == request.name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Dataset '{request.name}' already exists")
        
        if not os.path.exists(request.path):
            raise HTTPException(status_code=400, detail=f"Directory not found: {request.path}")
        
        dataset = Dataset(
            name=request.name,
            path=request.path,
            description=request.description,
            recursive=request.recursive,
            read_exif=request.read_exif,
            file_extensions=[".png", ".jpg", ".jpeg", ".webp"],
            total_items=0,
            total_captions=0,
            total_tags=0
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        # Calculate statistics by counting existing items/captions
        # (User may have already added items manually or from previous scan)
        total_items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset.id).count()
        total_tags, total_captions = _dataset_caption_item_counts(db, dataset.id)

        dataset.total_items = total_items
        dataset.total_captions = total_captions
        dataset.total_tags = total_tags
        db.commit()
        db.refresh(dataset)

        return dataset.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: int, db: Session = Depends(get_datasets_db)):
    """Get dataset by ID"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset.to_dict()

class CaptionProcessingUpdateRequest(BaseModel):
    caption_processing: Dict[str, Any]

@router.patch("/datasets/{dataset_id}/caption-processing")
async def update_caption_processing(
    dataset_id: int,
    request: CaptionProcessingUpdateRequest,
    db: Session = Depends(get_datasets_db)
):
    """Update caption processing configuration for a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset.caption_processing = request.caption_processing
    db.commit()
    db.refresh(dataset)

    return dataset.to_dict()


class DatasetSuffixUpdateRequest(BaseModel):
    reference_suffixes: Optional[List[str]] = None
    target_suffixes: Optional[List[str]] = None
    caption_suffixes_for_reference: Optional[List[str]] = None

@router.patch("/datasets/{dataset_id}/suffix-config")
async def update_dataset_suffix_config(
    dataset_id: int,
    request: DatasetSuffixUpdateRequest,
    db: Session = Depends(get_datasets_db)
):
    """Update suffix configuration for dataset structure (reference/target pairing)"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if request.reference_suffixes is not None:
        dataset.reference_suffixes = request.reference_suffixes
    if request.target_suffixes is not None:
        dataset.target_suffixes = request.target_suffixes
    if request.caption_suffixes_for_reference is not None:
        dataset.caption_suffixes_for_reference = request.caption_suffixes_for_reference

    db.commit()
    db.refresh(dataset)

    return dataset.to_dict()


class DatasetExifConfigUpdateRequest(BaseModel):
    read_exif: Optional[bool] = None
    exif_caption_fields: Optional[List[str]] = None


@router.patch("/datasets/{dataset_id}/exif-config")
async def update_dataset_exif_config(
    dataset_id: int,
    request: DatasetExifConfigUpdateRequest,
    db: Session = Depends(get_datasets_db)
):
    """Toggle EXIF-caption reading for a dataset (applied on the next scan).

    When read_exif is on, the scan extracts embedded EXIF caption fields per
    image (namespaced exif.<TagName>). exif_caption_fields optionally restricts
    which EXIF tags are read; an empty list / null uses a default caption-field set.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if request.read_exif is not None:
        dataset.read_exif = request.read_exif
    if request.exif_caption_fields is not None:
        # Empty list → clear (use the default field set on scan).
        dataset.exif_caption_fields = request.exif_caption_fields or None

    db.commit()
    db.refresh(dataset)
    return dataset.to_dict()


# ============================================================
# Caption Processing Presets API
# ============================================================

class CaptionProcessingPresetCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]

class CaptionProcessingPresetUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

@router.get("/caption-processing-presets")
async def list_caption_processing_presets(db: Session = Depends(get_datasets_db)):
    """List all caption processing presets"""
    from database.models import CaptionProcessingPreset
    presets = db.query(CaptionProcessingPreset).order_by(CaptionProcessingPreset.name).all()
    return [preset.to_dict() for preset in presets]

@router.post("/caption-processing-presets")
async def create_caption_processing_preset(
    request: CaptionProcessingPresetCreateRequest,
    db: Session = Depends(get_datasets_db)
):
    """Create a new caption processing preset"""
    from database.models import CaptionProcessingPreset

    # Check if preset with same name already exists
    existing = db.query(CaptionProcessingPreset).filter(CaptionProcessingPreset.name == request.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Preset with name '{request.name}' already exists")

    preset = CaptionProcessingPreset(
        name=request.name,
        description=request.description,
        config=request.config
    )
    db.add(preset)
    db.commit()
    db.refresh(preset)
    return preset.to_dict()

@router.get("/caption-processing-presets/{preset_id}")
async def get_caption_processing_preset(preset_id: int, db: Session = Depends(get_datasets_db)):
    """Get caption processing preset by ID"""
    from database.models import CaptionProcessingPreset
    preset = db.query(CaptionProcessingPreset).filter(CaptionProcessingPreset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return preset.to_dict()

@router.patch("/caption-processing-presets/{preset_id}")
async def update_caption_processing_preset(
    preset_id: int,
    request: CaptionProcessingPresetUpdateRequest,
    db: Session = Depends(get_datasets_db)
):
    """Update caption processing preset"""
    from database.models import CaptionProcessingPreset
    preset = db.query(CaptionProcessingPreset).filter(CaptionProcessingPreset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    if request.name is not None:
        # Check if new name conflicts with existing preset
        existing = db.query(CaptionProcessingPreset).filter(
            CaptionProcessingPreset.name == request.name,
            CaptionProcessingPreset.id != preset_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Preset with name '{request.name}' already exists")
        preset.name = request.name

    if request.description is not None:
        preset.description = request.description

    if request.config is not None:
        preset.config = request.config

    db.commit()
    db.refresh(preset)
    return preset.to_dict()

@router.delete("/caption-processing-presets/{preset_id}", status_code=204)
async def delete_caption_processing_preset(preset_id: int, db: Session = Depends(get_datasets_db)):
    """Delete caption processing preset"""
    from database.models import CaptionProcessingPreset
    preset = db.query(CaptionProcessingPreset).filter(CaptionProcessingPreset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    db.delete(preset)
    db.commit()
    return None


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: int, db: Session = Depends(get_datasets_db)):
    """Delete dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(dataset)
    db.commit()
    return Response(status_code=204)

@router.get("/tag-dictionary")
async def search_tag_dictionary(
    search: Optional[str] = None,
    category: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
    db: Session = Depends(get_datasets_db)
):
    """Search tag dictionary"""
    query = db.query(TagDictionary)
    if search:
        query = query.filter(TagDictionary.tag.like(f"%{search}%"))
    if category:
        query = query.filter(TagDictionary.category == category)
    
    total = query.count()
    offset = (page - 1) * page_size
    tags = query.order_by(TagDictionary.count.desc()).offset(offset).limit(page_size).all()
    
    return {"tags": [t.to_dict() for t in tags], "total": total, "page": page, "page_size": page_size}

@router.get("/tag-dictionary/stats")
async def get_tag_dictionary_stats(db: Session = Depends(get_datasets_db)):
    """Get tag dictionary statistics"""
    from sqlalchemy import func
    total_tags = db.query(func.count(TagDictionary.id)).scalar()
    return {"total_tags": total_tags or 0}

async def compute_tag_statistics(dataset_id: int, db: Session, send_progress: bool = False, total_steps: int = 0, current_step: int = 0) -> dict:
    """
    Compute tag statistics for a dataset with categories.
    Returns: {"tag": {"count": N, "category": "..."}, ...}

    Category resolution priority (highest first):
      1. tag_data with a known category (non-Unknown)
      2. taglist_cache lookup (for captions without tag_data, or tag_data with Unknown)
      3. "Unknown" (fallback when tag is not in taglist)

    "Unknown is lowest priority": any known category overwrites a previously
    stored "Unknown", but known categories never overwrite each other.

    Optimized for large datasets (streaming processing, no full data load).
    """
    import json as _json

    print(f"[Dataset] Computing tag statistics for dataset {dataset_id}...")
    # Ensure category resolution uses the Gelbooru supplement + alias fallback
    # (graceful when taglist_gel/ is absent). The gelbooru load is a one-time
    # latch on the shared cache, so this is cheap on repeat calls.
    taglist_cache.initialize(settings.root_dir, enable_gelbooru=True)

    # Count total items
    total_items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).count()
    if total_items == 0:
        print(f"[Dataset] No items found, returning empty statistics")
        return {}

    # Stream captions in batches to avoid loading all into memory
    tag_counts: dict[str, int] = {}
    tag_categories: dict[str, str] = {}  # tag -> category ("Unknown" = not yet resolved)
    # Collect tags still needing resolution (no tag_data, or tag_data returned Unknown)
    unresolved_tags: set[str] = set()
    batch_size = 1000
    offset = 0
    processed = 0

    def _set_category(tag: str, category: str) -> None:
        """Set category for tag; Unknown is lowest priority and never overwrites a known category."""
        existing = tag_categories.get(tag)
        if existing is None or (existing == "Unknown" and category != "Unknown"):
            tag_categories[tag] = category

    while True:
        # Get batch of captions via JOIN (efficient query)
        batch = db.query(DatasetCaption).join(
            DatasetItem, DatasetCaption.item_id == DatasetItem.id
        ).filter(
            DatasetItem.dataset_id == dataset_id,
            DatasetCaption.caption_type == "tags"
        ).offset(offset).limit(batch_size).all()

        if not batch:
            break

        # Collect tags from captions without tag_data so we can batch-resolve them
        content_tags_batch: list[str] = []

        # Process batch
        for caption in batch:
            if caption.tag_data:
                try:
                    tag_data = _json.loads(caption.tag_data)
                    for item in tag_data:
                        tag = item.get("tag", "").strip()
                        category = item.get("category", "Unknown")
                        if tag:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                            _set_category(tag, category)
                            if tag_categories.get(tag) == "Unknown":
                                unresolved_tags.add(tag)
                except Exception:
                    # Malformed tag_data — fall back to content parse
                    if caption.content:
                        for tag in caption.content.split(","):
                            tag = tag.strip()
                            if tag:
                                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                                content_tags_batch.append(tag)
            else:
                # No tag_data: parse from content, resolve via taglist_cache later
                if caption.content:
                    for tag in caption.content.split(","):
                        tag = tag.strip()
                        if tag:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                            content_tags_batch.append(tag)

        # Batch-resolve tags from content-only captions using taglist_cache
        if content_tags_batch:
            unique_content_tags = list(set(content_tags_batch))
            resolved = taglist_cache.get_categories_batch(unique_content_tags)
            for tag in unique_content_tags:
                category = resolved.get(tag, "Unknown")
                _set_category(tag, category)
            # Remove from unresolved if now known
            for tag in unique_content_tags:
                if tag_categories.get(tag) != "Unknown":
                    unresolved_tags.discard(tag)

        processed += len(batch)
        offset += batch_size

        # Log progress every 10k captions
        if processed % 10000 == 0:
            print(f"[Dataset] Tag statistics: processed {processed} captions, {len(tag_counts)} unique tags so far")
            if send_progress and total_steps > 0:
                estimated_progress = current_step
                manager.send_progress_sync(
                    estimated_progress,
                    total_steps,
                    f"Computing tag statistics: {processed} captions, {len(tag_counts)} unique tags"
                )

    # Final pass: resolve any remaining Unknown tags via taglist_cache
    if unresolved_tags:
        print(f"[Dataset] Resolving {len(unresolved_tags)} remaining Unknown tags via taglist_cache...")
        resolved = taglist_cache.get_categories_batch(list(unresolved_tags))
        for tag in unresolved_tags:
            category = resolved.get(tag, "Unknown")
            if category != "Unknown":
                tag_categories[tag] = category

    print(f"[Dataset] Found {len(tag_counts)} unique tags from {processed} captions")

    # Build final statistics with categories
    statistics = {}
    for tag, count in tag_counts.items():
        statistics[tag] = {
            "count": count,
            "category": tag_categories.get(tag, "Unknown")
        }

    unknown_count = sum(1 for v in statistics.values() if v["category"] == "Unknown")
    print(f"[Dataset] Tag statistics computed: {len(statistics)} tags ({unknown_count} Unknown)")
    return statistics


@router.post("/datasets/{dataset_id}/scan/preview")
async def scan_dataset_preview(dataset_id: int, db: Session = Depends(get_datasets_db)):
    """Preview dataset structure before importing.

    Scans the directory and returns detected file groups, caption suffixes,
    and format classifications without writing to the database.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.isdir(dataset.path):
        raise HTTPException(status_code=400, detail=f"Dataset path not found: {dataset.path}")

    from utils.dataset_scanner import scan_directory_structure, classify_caption_files, build_scan_preview
    from utils.taglist_loader import load_all_tags

    # Load taglist for format detection
    taglist = load_all_tags(settings.root_dir)

    # 2-pass scan
    scan_groups = scan_directory_structure(
        dir_path=dataset.path,
        recursive=dataset.recursive,
        max_depth=dataset.max_depth if dataset.max_depth else None,
        reference_suffixes=dataset.reference_suffixes or [],
        target_suffixes=dataset.target_suffixes or [],
    )

    # Classify caption files (sample up to 500 groups for performance)
    sample_groups = dict(list(scan_groups.items())[:500])
    classify_caption_files(sample_groups, taglist)

    # Build preview
    preview = build_scan_preview(sample_groups)
    preview["dataset_path"] = dataset.path

    print(f"[ScanPreview] {preview['total_groups']} groups, {preview['total_images']} images, "
          f"{preview['total_captions']} captions, suffixes: {list(preview['detected_suffixes'].keys())}")

    return preview


@router.post("/datasets/{dataset_id}/scan")
async def scan_dataset(
    dataset_id: int,
    db: Session = Depends(get_datasets_db),
    *,
    incremental: bool = False,
):
    """Scan dataset directory and register images/captions.

    When *incremental* is True (training pre-flight rescan):
      - If nothing structurally changed (items_found==0, items_purged==0),
        the existing tag_statistics is kept as-is (Case 1).
      - Otherwise, tag_statistics is updated by adding/subtracting only
        the counts for new/purged items (Case 2).
    Both modes avoid the O(total_captions) full recomputation.

    Note: this HTTP route has no way to receive a cancellation callback
    from the client, so cooperative cancellation is always disabled here
    (``should_cancel=None``). The underlying scan helpers still accept a
    real callable when invoked internally (see
    ``core.training.dataset_drift.rescan_dataset_inline``), which is used
    for the training pre-flight rescan path instead of this route.
    """
    import os
    from PIL import Image
    import warnings

    # HTTP callers can never supply a cancellation callback; kept as a
    # local so the body below (which threads it into scan helpers) is
    # unchanged.
    should_cancel: Optional[Callable[[], bool]] = None

    # Suppress PIL warnings for corrupt EXIF data
    warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {dataset.path}")

    # Auto-detect dataset structure before scanning
    from utils.dataset_structure_detector import detect_dataset_structure

    structure_detection_result = None
    if not dataset.reference_suffixes and not dataset.target_suffixes:
        print(f"[Dataset Scan] Auto-detecting dataset structure...")
        import asyncio
        _loop = asyncio.get_event_loop()
        structure_detection_result = await _loop.run_in_executor(
            None,
            lambda: detect_dataset_structure(
                dataset.path,
                recursive=dataset.recursive,
                max_depth=dataset.max_depth if hasattr(dataset, 'max_depth') and dataset.max_depth else None,
            )
        )

        if structure_detection_result["structure_type"] == "paired":
            dataset.reference_suffixes = structure_detection_result["reference_suffixes"]
            dataset.target_suffixes = structure_detection_result["target_suffixes"]
            dataset.caption_suffixes_for_reference = structure_detection_result.get("caption_suffixes_for_reference", [])
            db.commit()
            db.refresh(dataset)
            print(f"[Dataset Scan] Detected paired structure: "
                  f"ref={structure_detection_result['reference_suffixes']}, "
                  f"target={structure_detection_result['target_suffixes']}, "
                  f"caption={structure_detection_result.get('caption_suffixes_for_reference', [])}, "
                  f"confidence={structure_detection_result['confidence']:.3f}")
        else:
            print(f"[Dataset Scan] Normal dataset structure detected")
    else:
        print(f"[Dataset Scan] Using existing suffix configuration: "
              f"ref={dataset.reference_suffixes}, target={dataset.target_suffixes}")

    # Supported image + video + audio extensions
    from utils.dataset_scanner import (
        VIDEO_EXTS as video_exts,
        AUDIO_EXTS as audio_exts,
        probe_video_metadata,
        probe_audio_metadata,
        extract_poster_frame,
    )
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    media_exts = image_exts | video_exts | audio_exts
    caption_exts = {".txt", ".json"}

    # Load taglist for caption format detection (once at start)
    from utils.taglist_loader import load_all_tags
    from utils.caption_detector import classify_field, scan_json_fields, read_exif_captions
    print(f"[Dataset Scan] Loading taglist for format detection...")
    # Gelbooru supplement + alias table improve both format detection (match rate)
    # and category resolution; graceful when taglist_gel/ is absent.
    taglist = load_all_tags(settings.root_dir, include_gelbooru=True)
    taglist_cache.initialize(settings.root_dir, enable_gelbooru=True)
    print(f"[Dataset Scan] Loaded {len(taglist)} tags for format detection")

    # Read-EXIF option: when enabled, embedded EXIF caption fields are extracted
    # per image (namespaced exif.<TagName>) alongside the TXT/JSON sidecars.
    read_exif_enabled = bool(getattr(dataset, "read_exif", False))
    exif_caption_fields = getattr(dataset, "exif_caption_fields", None) or None
    if read_exif_enabled:
        print(f"[Dataset Scan] read_exif enabled (fields={exif_caption_fields or 'default set'})")

    # Per-field-bucket scan counters (added/updated this run), so the progress &
    # result are intuitive instead of conflating images with caption rows. The
    # two training fields are reported on their own; everything else (image.*,
    # source.*, savedAt, exif.*, suffix fields) is aggregated as "other".
    #   tags = the danbooru-tags field;  caption = the natural-language field.
    _fstats = {"tags_add": 0, "tags_upd": 0, "cap_add": 0, "cap_upd": 0,
               "other_add": 0, "other_upd": 0}

    def _fbucket(caption_type: str) -> str:
        if caption_type == "tags":
            return "tags"
        if caption_type in ("natural_language", "caption"):
            return "cap"
        return "other"

    def _fstat_bump(caption_type: str, added: bool) -> None:
        _fstats[f"{_fbucket(caption_type)}_{'add' if added else 'upd'}"] += 1

    def _fstat_msg() -> str:
        """Compact per-field summary (+N new / ~N updated) for progress lines."""
        return (f"tags +{_fstats['tags_add']}/~{_fstats['tags_upd']} | "
                f"caption +{_fstats['cap_add']}/~{_fstats['cap_upd']} | "
                f"other +{_fstats['other_add']}/~{_fstats['other_upd']}")

    def _build_tag_data_json(content: str) -> str:
        """Build tag_data JSON string from comma-separated tag content."""
        import json as _json
        tags = [t.strip() for t in content.split(",") if t.strip()]
        if not tags:
            return "[]"
        cats = taglist_cache.get_categories_batch(tags)
        return _json.dumps(
            [{"tag": t, "category": cats.get(t, "Unknown")} for t in tags],
            ensure_ascii=False,
        )

    def _upsert_caption(item_id_local: int, result: dict) -> bool:
        """Insert or update a caption row keyed by (item_id, caption_type).

        Unifies JSON / EXIF field handling so a non-tags field (image.filename,
        source.*, exif.*, …) is UPDATED in place instead of being re-added on
        every rescan (which previously duplicated those rows). Returns True when a
        new row was added, False when an existing row was updated.
        """
        ctype = result["caption_type"]
        _is_tags = result["is_tags_format"]
        _tag_data = _build_tag_data_json(result["content"]) if _is_tags else None
        existing = db.query(DatasetCaption).filter(
            DatasetCaption.item_id == item_id_local,
            DatasetCaption.caption_type == ctype,
        ).first()
        if existing:
            existing.content = result["content"]
            existing.field_category = result["field_category"]
            existing.is_tags_format = _is_tags
            existing.tag_match_rate = result["tag_match_rate"]
            existing.source = "file"
            existing.source_field = result["source_field"]
            existing.tag_data = _tag_data
            existing.updated_at = datetime.utcnow()
            _fstat_bump(ctype, added=False)
            return False
        db.add(DatasetCaption(
            item_id=item_id_local,
            caption_type=ctype,
            content=result["content"],
            field_category=result["field_category"],
            is_tags_format=_is_tags,
            tag_match_rate=result["tag_match_rate"],
            tag_data=_tag_data,
            source="file",
            source_field=result["source_field"],
        ))
        _fstat_bump(ctype, added=True)
        return True

    # Pre-scan with 2-pass scanner: detect suffix captions + count images in one pass
    from utils.dataset_scanner import scan_directory_structure
    import asyncio
    print(f"[Dataset Scan] Pre-scanning directory structure...")
    loop = asyncio.get_event_loop()
    from core.training.rescan_control import RescanSkipped
    try:
        pre_scan_groups = await loop.run_in_executor(
            None,
            lambda: scan_directory_structure(
                dir_path=dataset.path,
                recursive=dataset.recursive,
                max_depth=dataset.max_depth if dataset.max_depth else None,
                reference_suffixes=dataset.reference_suffixes or [],
                target_suffixes=dataset.target_suffixes or [],
                should_cancel=should_cancel,
            )
        )
    except RescanSkipped:
        # Skipped during the pre-scan walk — nothing written yet.
        print(f"[Dataset Scan] Skipped during pre-scan walk for dataset {dataset_id}")
        return {
            "items_found": 0, "captions_found": 0, "captions_updated": 0,
            "items_purged": 0, "cancelled": True, "dataset": dataset.to_dict(),
        }

    # Build suffix caption lookup and count images from pre-scan results
    suffix_captions_by_stem = {}
    detected_suffixes = set()
    total_images = 0
    for group_name, group_data in pre_scan_groups.items():
        # Count main/target images (not reference)
        total_images += sum(1 for img in group_data["images"] if img["role"] in ("main", "target"))
        # Collect suffix captions
        suffix_caps = [(c["suffix"], c["path"]) for c in group_data["captions"] if c["suffix"]]
        if suffix_caps:
            suffix_captions_by_stem[group_name] = suffix_caps
            for s, _ in suffix_caps:
                detected_suffixes.add(s)
    if detected_suffixes:
        print(f"[Dataset Scan] Detected caption suffixes: {sorted(detected_suffixes)}")
        existing_suffixes = dataset.caption_suffixes or []
        dataset.caption_suffixes = sorted(set(existing_suffixes) | detected_suffixes)

    print(f"[Dataset Scan] Found {total_images} images, {len(suffix_captions_by_stem)} groups with suffix captions")

    # --- Path-based dedup: batch-load existing items (single query) ---
    existing_items_rows = db.query(DatasetItem.id, DatasetItem.image_path).filter(
        DatasetItem.dataset_id == dataset_id
    ).all()
    existing_paths: dict[str, int] = {row.image_path: row.id for row in existing_items_rows}
    # Track which existing paths are still on disk (for purge at the end)
    seen_existing_paths: set[str] = set()
    # mtime threshold: captions updated after this are re-processed
    last_scanned_ts = dataset.last_scanned_at.timestamp() if dataset.last_scanned_at else 0.0
    # Track new item IDs for incremental tag_statistics update
    new_item_ids: list[int] = []
    print(f"[Dataset Scan] Loaded {len(existing_paths)} existing items from DB (path-based dedup)")

    # Scan directory
    items_found = 0
    captions_found = 0
    captions_updated = 0
    files_processed = 0

    # Progress tracking: Phase 1 (File scan): 0-90%, Phase 2 (Tag stats): 90-100%
    # We'll use a unified total_steps = total_images * 1.1 (rounded)
    # This way: file scan uses steps 0 to total_images (90.9%), tag stats uses remaining (9.1%)
    total_steps = int(total_images * 1.1) if total_images > 0 else 100

    # Send initial progress
    print(f"[Dataset Scan] Sending initial progress to frontend...")
    manager.send_progress_sync(0, total_steps, f"Starting scan: 0/{total_images} images to process")
    print(f"[Dataset Scan] Starting directory scan...")

    def scan_directory(dir_path, current_depth=0):
        nonlocal items_found, captions_found, captions_updated, files_processed

        # Cooperative cancellation (training pre-flight skip): checked per
        # directory and per image-group below. Raises RescanSkipped which the
        # registration await catches to commit partial progress.
        if should_cancel is not None and should_cancel():
            raise RescanSkipped()

        try:
            entries = os.listdir(dir_path)
        except PermissionError:
            print(f"[Dataset Scan] Permission denied: {dir_path}")
            return

        print(f"[Dataset Scan] Scanning directory: {dir_path} ({len(entries)} entries)")

        # Get reference image settings from dataset
        reference_suffixes = dataset.reference_suffixes or []
        target_suffixes = dataset.target_suffixes or []
        caption_suffixes_for_ref = dataset.caption_suffixes_for_reference or []

        # Check if reference image mode is enabled
        use_reference_mode = bool(reference_suffixes and target_suffixes)
        if use_reference_mode:
            print(f"[Dataset Scan] Reference mode enabled: ref_suffixes={reference_suffixes}, target_suffixes={target_suffixes}")

        # Helper function to strip suffix and get base group name
        def get_group_name_and_type(filename):
            """
            For reference mode: extract group name and file type from filename.
            Example with suffixes ["_source"] and ["_target"]:
              - "20251026_01k8e370_01k8e370_source.webp" -> ("20251026_01k8e370_01k8e370", "reference")
              - "20251026_01k8e370_01k8e370_target.webp" -> ("20251026_01k8e370_01k8e370", "target")
              - "20251026_01k8e370_01k8e370_instruction.txt" -> ("20251026_01k8e370_01k8e370", "caption")
              - "normal_image.png" -> ("normal_image", "normal")
            """
            base_name, ext = os.path.splitext(filename)

            if use_reference_mode:
                # Check reference suffixes (e.g., "_source")
                for suffix in reference_suffixes:
                    if base_name.endswith(suffix):
                        group_name = base_name[:-len(suffix)]
                        return group_name, "reference"

                # Check target suffixes (e.g., "_target")
                for suffix in target_suffixes:
                    if base_name.endswith(suffix):
                        group_name = base_name[:-len(suffix)]
                        return group_name, "target"

                # Check caption suffixes for reference mode (e.g., "_instruction")
                for suffix in caption_suffixes_for_ref:
                    if base_name.endswith(suffix):
                        group_name = base_name[:-len(suffix)]
                        return group_name, "caption"

            # Normal mode: use base_name as group name
            return base_name, "normal"

        # Group files by group name
        file_groups = {}
        entries_processed = 0
        for entry in entries:
            entries_processed += 1
            # Log progress every 10k entries
            if entries_processed % 10000 == 0:
                print(f"[Dataset Scan] Grouped {entries_processed}/{len(entries)} entries in {dir_path}")
            entry_path = os.path.join(dir_path, entry)

            if os.path.isfile(entry_path):
                base_name, ext = os.path.splitext(entry)
                ext_lower = ext.lower()
                group_name, file_type = get_group_name_and_type(entry)

                if group_name not in file_groups:
                    file_groups[group_name] = {
                        "images": [],       # Normal images (no suffix)
                        "captions": [],     # Normal captions (no suffix)
                        "reference": [],    # Reference images (_source suffix)
                        "target": [],       # Target images (_target suffix)
                        "ref_captions": [], # Reference mode captions (_instruction suffix)
                    }

                if ext_lower in media_exts:
                    if file_type == "reference":
                        file_groups[group_name]["reference"].append(entry_path)
                    elif file_type == "target":
                        file_groups[group_name]["target"].append(entry_path)
                    else:
                        file_groups[group_name]["images"].append(entry_path)
                elif ext_lower in caption_exts:
                    if file_type == "caption":
                        file_groups[group_name]["ref_captions"].append(entry_path)
                    else:
                        file_groups[group_name]["captions"].append(entry_path)

            elif os.path.isdir(entry_path) and dataset.recursive:
                max_depth = dataset.max_depth if dataset.max_depth else float('inf')
                if current_depth < max_depth:
                    scan_directory(entry_path, current_depth + 1)

        print(f"[Dataset Scan] Grouped {len(file_groups)} file groups in {dir_path}, starting processing...")

        # Process file groups
        groups_processed = 0
        for base_name, files in file_groups.items():
            groups_processed += 1
            # Log progress every 1k groups
            if groups_processed % 1000 == 0:
                print(f"[Dataset Scan] Processed {groups_processed}/{len(file_groups)} file groups in {dir_path}")
            # Cooperative cancellation: poll every 200 image-groups so a skip
            # aborts a large flat directory promptly (RescanSkipped propagates).
            if should_cancel is not None and groups_processed % 200 == 0 and should_cancel():
                raise RescanSkipped()

            # Determine which image to use as main and which as reference
            main_images = []
            reference_images = []
            caption_files = []

            if use_reference_mode:
                # Reference mode: use target as main, reference as related
                main_images = files["target"]
                reference_images = files["reference"]
                caption_files = files["ref_captions"] if files["ref_captions"] else files["captions"]
            else:
                # Normal mode: use images as main, no reference
                main_images = files["images"]
                caption_files = files["captions"]

            if not main_images:
                continue

            # Use first image as primary
            image_path = main_images[0]

            _t_item = time.time()
            try:
                # --- Path-based dedup (replaces SHA256 hash + per-item DB query) ---
                # Decide existing-vs-new BEFORE opening the image so unchanged
                # existing items skip the PIL open entirely — their dimensions are
                # already stored. Only NEW images need to be opened for width/height.
                existing_item_id = existing_paths.get(image_path)
                if existing_item_id is not None:
                    # Image already registered — mark as seen (for purge logic)
                    seen_existing_paths.add(image_path)
                    files_processed += 1

                    # Check if any caption files have been updated since last scan
                    any_caption_updated = False
                    for cp in caption_files:
                        try:
                            if os.path.getmtime(cp) > last_scanned_ts:
                                any_caption_updated = True
                                break
                        except OSError:
                            pass
                    # Also check suffix captions
                    if not any_caption_updated and base_name in suffix_captions_by_stem:
                        for _, sp in suffix_captions_by_stem[base_name]:
                            try:
                                if os.path.getmtime(sp) > last_scanned_ts:
                                    any_caption_updated = True
                                    break
                            except OSError:
                                pass
                    # With read_exif on, a newer image file can itself carry updated
                    # embedded captions — treat a newer image mtime as an update.
                    if not any_caption_updated and read_exif_enabled:
                        try:
                            if os.path.getmtime(image_path) > last_scanned_ts:
                                any_caption_updated = True
                        except OSError:
                            pass

                    if not any_caption_updated:
                        # No changes — skip entirely (no Image.open: dimensions are
                        # already stored on the existing item).
                        if files_processed % 10 == 0 or total_images < 100:
                            manager.send_progress_sync(
                                files_processed,
                                total_steps,
                                f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                            )
                        continue

                    # Captions updated — re-process for this existing item. No
                    # Image.open needed (dimensions already in DB).
                    item_id_for_captions = existing_item_id
                    if files_processed % 10 == 0 or total_images < 100:
                        manager.send_progress_sync(
                            files_processed,
                            total_steps,
                            f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                        )
                else:
                    _ext_lower = os.path.splitext(image_path)[1].lower()
                    is_video = _ext_lower in video_exts
                    is_audio = _ext_lower in audio_exts
                    video_meta = None
                    audio_meta = None

                    if is_video:
                        # New video — probe metadata via ffprobe WITHOUT decoding
                        # all frames. A probe failure skips the file (logged).
                        video_meta = probe_video_metadata(image_path)
                        if not video_meta:
                            print(f"[Dataset Scan] Skipping unreadable video {image_path}")
                            files_processed += 1
                            if files_processed % 10 == 0 or total_images < 100:
                                manager.send_progress_sync(
                                    files_processed,
                                    total_steps,
                                    f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                                )
                            continue
                        width = video_meta["width"]
                        height = video_meta["height"]
                    elif is_audio:
                        # New audio clip — probe metadata via soundfile/ffprobe
                        # WITHOUT decoding the whole file. Audio clips have no
                        # spatial dimensions (width/height stored as 0).
                        audio_meta = probe_audio_metadata(image_path)
                        if not audio_meta:
                            print(f"[Dataset Scan] Skipping unreadable audio {image_path}")
                            files_processed += 1
                            if files_processed % 10 == 0 or total_images < 100:
                                manager.send_progress_sync(
                                    files_processed,
                                    total_steps,
                                    f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                                )
                            continue
                        width = 0
                        height = 0
                    else:
                        # New image — open it ONCE for dimensions, then register.
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                with Image.open(image_path) as img:
                                    width, height = img.size
                        except Exception as img_error:
                            # Skip images that can't be opened (corrupt, unsupported, etc.)
                            print(f"[Dataset Scan] Skipping corrupt/unsupported image {image_path}: {img_error}")
                            files_processed += 1
                            if files_processed % 10 == 0 or total_images < 100:
                                manager.send_progress_sync(
                                    files_processed,
                                    total_steps,
                                    f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                                )
                            continue

                    file_size = os.path.getsize(image_path)

                    # Build related_images for reference mode
                    related_images_data = {}
                    if use_reference_mode and reference_images:
                        related_images_data["reference"] = reference_images
                        print(f"[Dataset Scan] Group '{base_name}': {len(reference_images)} reference image(s)")

                    if is_video:
                        item_type = "video"
                    elif is_audio:
                        item_type = "audio"
                    elif use_reference_mode:
                        item_type = "reference"
                    else:
                        item_type = "single"

                    item = DatasetItem(
                        dataset_id=dataset_id,
                        # image_path stores the video/audio file path for
                        # video/audio items (it is just a path string).
                        # Per-clip metadata lives in exif_data (surfaced as
                        # video_meta / audio_meta in to_dict).
                        item_type=item_type,
                        base_name=base_name,
                        image_path=image_path,
                        width=width,
                        height=height,
                        file_size=file_size,
                        image_hash=None,  # SHA256 no longer computed at scan time
                        exif_data=video_meta if is_video else (audio_meta if is_audio else None),
                        related_images=related_images_data if related_images_data else None
                    )
                    db.add(item)
                    db.flush()  # Get item.id
                    item_id_for_captions = item.id
                    items_found += 1
                    files_processed += 1
                    new_item_ids.append(item.id)

                    # Poster thumbnail for videos: extract frame 0 via cv2 and run
                    # it through the shared PNG+WebP thumbnail generator keyed by
                    # base_name, so the dataset UI has a preview to show.
                    if is_video:
                        try:
                            import tempfile
                            poster_tmp = os.path.join(tempfile.gettempdir(), f"_dsposter_{base_name}.png")
                            if extract_poster_frame(image_path, poster_tmp):
                                # create_thumbnail keys the output by the source
                                # basename; rename target so it matches base_name.
                                poster_named = os.path.join(tempfile.gettempdir(), f"{base_name}.png")
                                try:
                                    if poster_tmp != poster_named:
                                        os.replace(poster_tmp, poster_named)
                                    create_thumbnail(poster_named)
                                finally:
                                    for _p in (poster_tmp, poster_named):
                                        try:
                                            os.remove(_p)
                                        except OSError:
                                            pass
                        except Exception as _pe:
                            print(f"[Dataset Scan] poster thumbnail failed for {image_path}: {_pe}")

                    # Waveform thumbnail for audio clips: render a peak-envelope
                    # PNG via soundfile + the shared audio waveform writer, then
                    # run it through the same PNG+WebP thumbnail generator keyed
                    # by base_name, so the dataset UI has a preview to show
                    # (mirrors the video poster-frame path above).
                    if is_audio:
                        try:
                            import tempfile
                            import soundfile as sf
                            from utils.audio_utils import _write_waveform_png
                            wave_named = os.path.join(tempfile.gettempdir(), f"{base_name}.png")
                            try:
                                data, _sr = sf.read(image_path, dtype="float32", always_2d=True)
                                # soundfile returns [samples, channels]; waveform
                                # writer expects [channels, samples].
                                arr = data.T
                                _write_waveform_png(arr, wave_named)
                                create_thumbnail(wave_named)
                            finally:
                                try:
                                    os.remove(wave_named)
                                except OSError:
                                    pass
                        except Exception as _ae:
                            print(f"[Dataset Scan] waveform thumbnail failed for {image_path}: {_ae}")

                    if files_processed % 10 == 0 or total_images < 100:
                        manager.send_progress_sync(
                            files_processed,
                            total_steps,
                            f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                        )

                # Process captions (TXT/JSON files) — for both new and updated items
                # Use item_id_for_captions (set above for both new and existing items)
                _t_caps = time.time()
                _jr = _sjf = _ups = 0.0  # json-read / scan_json_fields / upsert sub-times
                _cf = _btd = _sfx = _exif = 0.0  # classify / build_tag_data / suffix / exif
                _txr = _txq = 0.0  # .txt file read / .txt migration query
                for caption_path in caption_files:
                    try:
                        _, ext = os.path.splitext(caption_path)
                        ext_lower = ext.lower()

                        if ext_lower == '.txt':
                            # TXT file: Read content and detect format
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                _ts = time.time()
                                content = f.read().strip()
                                _txr += time.time() - _ts
                                if content:
                                    # Detect format
                                    _ts = time.time()
                                    field_category, is_tags_format, match_rate = classify_field("tags", content, taglist)
                                    _cf += time.time() - _ts

                                    # Determine caption_type based on detected format
                                    detected_caption_type = "tags" if is_tags_format else "natural_language"

                                    # A .txt sidecar yields exactly ONE caption whose type ('tags' or
                                    # 'natural_language') depends on detection. Find it regardless of its
                                    # CURRENT stored type so a re-detection that flips the type — e.g. a
                                    # fixed detector now recognising a sidecar as tags, or a repaired
                                    # sidecar — MIGRATES the same row instead of leaving a stale
                                    # natural_language row and adding a duplicate tags row.
                                    _ts = time.time()
                                    existing_cap = db.query(DatasetCaption).filter(
                                        DatasetCaption.item_id == item_id_for_captions,
                                        DatasetCaption.source == "file",
                                        DatasetCaption.caption_type.in_(["tags", "natural_language"]),
                                    ).first()
                                    _txq += time.time() - _ts

                                    if existing_cap:
                                        # Update existing (migrating caption_type if it changed)
                                        existing_cap.caption_type = detected_caption_type
                                        existing_cap.content = content
                                        existing_cap.field_category = field_category
                                        existing_cap.is_tags_format = is_tags_format
                                        existing_cap.tag_match_rate = match_rate
                                        existing_cap.source = "file"
                                        existing_cap.source_field = detected_caption_type
                                        _ts = time.time()
                                        existing_cap.tag_data = _build_tag_data_json(content) if is_tags_format else None
                                        _btd += time.time() - _ts
                                        existing_cap.updated_at = datetime.utcnow()
                                        captions_updated += 1
                                        _fstat_bump(detected_caption_type, added=False)
                                    else:
                                        # Create new
                                        _ts = time.time()
                                        _td_new = _build_tag_data_json(content) if is_tags_format else None
                                        _btd += time.time() - _ts
                                        caption = DatasetCaption(
                                            item_id=item_id_for_captions,
                                            caption_type=detected_caption_type,
                                            content=content,
                                            field_category=field_category,
                                            is_tags_format=is_tags_format,
                                            tag_match_rate=match_rate,
                                            tag_data=_td_new,
                                            source="file",
                                            source_field=detected_caption_type
                                        )
                                        db.add(caption)
                                        captions_found += 1
                                        _fstat_bump(detected_caption_type, added=True)

                        elif ext_lower == '.json':
                            # JSON file: Recursively scan all fields
                            import json

                            _ts = time.time()
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            _jr += time.time() - _ts

                            # Scan all fields. Every field (the single tags field
                            # AND the non-tags fields) is upserted by caption_type,
                            # so a rescan UPDATES each row in place instead of
                            # re-adding non-tags fields (which previously duplicated
                            # them on every scan).
                            _ts = time.time()
                            caption_results = scan_json_fields(json_data, taglist)
                            _sjf += time.time() - _ts
                            _ts = time.time()
                            for result in caption_results:
                                if _upsert_caption(item_id_for_captions, result):
                                    captions_found += 1
                                else:
                                    captions_updated += 1
                            _ups += time.time() - _ts

                    except Exception as e:
                        print(f"[Dataset Scan] Failed to read caption {caption_path}: {e}")

                # Process suffix-based caption files detected by 2-pass scanner
                _ts = time.time()
                if base_name in suffix_captions_by_stem:
                    for suffix, suffix_path in suffix_captions_by_stem[base_name]:
                        try:
                            _, sext = os.path.splitext(suffix_path)
                            if sext.lower() == '.txt':
                                with open(suffix_path, 'r', encoding='utf-8') as f:
                                    content = f.read().strip()
                                if content:
                                    field_category, is_tags_format, match_rate = classify_field(
                                        suffix, content, taglist
                                    )
                                    existing_cap = db.query(DatasetCaption).filter(
                                        DatasetCaption.item_id == item_id_for_captions,
                                        DatasetCaption.caption_type == suffix
                                    ).first()
                                    if existing_cap:
                                        existing_cap.content = content
                                        existing_cap.field_category = field_category
                                        existing_cap.is_tags_format = is_tags_format
                                        existing_cap.tag_match_rate = match_rate
                                        existing_cap.source = "file"
                                        existing_cap.source_field = suffix
                                        if is_tags_format:
                                            existing_cap.tag_data = _build_tag_data_json(content)
                                        existing_cap.updated_at = datetime.utcnow()
                                        captions_updated += 1
                                        _fstat_bump(suffix, added=False)
                                    else:
                                        caption = DatasetCaption(
                                            item_id=item_id_for_captions,
                                            caption_type=suffix,
                                            content=content,
                                            field_category=field_category,
                                            is_tags_format=is_tags_format,
                                            tag_match_rate=match_rate,
                                            tag_data=_build_tag_data_json(content) if is_tags_format else None,
                                            source="file",
                                            source_field=suffix
                                        )
                                        db.add(caption)
                                        captions_found += 1
                                        _fstat_bump(suffix, added=True)
                        except Exception as e:
                            print(f"[Dataset Scan] Failed to read suffix caption {suffix_path}: {e}")
                _sfx += time.time() - _ts

                # Process EXIF-embedded captions (when read_exif is enabled). Each
                # field is namespaced exif.<TagName> and upserted by caption_type,
                # so the main tags/natural_language rows are never affected.
                _ts = time.time()
                if read_exif_enabled:
                    try:
                        for result in read_exif_captions(image_path, taglist, exif_caption_fields):
                            if _upsert_caption(item_id_for_captions, result):
                                captions_found += 1
                            else:
                                captions_updated += 1
                    except Exception as e:
                        print(f"[Dataset Scan] Failed to read EXIF captions for {image_path}: {e}")
                _exif += time.time() - _ts

                # Per-item timing probe: surface which items (and which phase) stall
                # in the live backend, since every phase is fast in isolation.
                _caps_ms = (time.time() - _t_caps) * 1000
                _item_ms = (time.time() - _t_item) * 1000
                if _item_ms > 200:
                    _has_json = any(str(c).lower().endswith(".json") for c in caption_files)
                    print(f"[Dataset Scan][SLOW] {_item_ms:.0f}ms (caps {_caps_ms:.0f} = "
                          f"jsonRead {_jr*1000:.0f} + scanFields {_sjf*1000:.0f} + upsert {_ups*1000:.0f} + "
                          f"classify {_cf*1000:.0f} + buildTagData {_btd*1000:.0f} + txtRead {_txr*1000:.0f} + "
                          f"txtQuery {_txq*1000:.0f} + suffix {_sfx*1000:.0f} + "
                          f"exif {_exif*1000:.0f}) json={_has_json} ncaps={len(caption_files)} {os.path.basename(image_path)}")

            except Exception as e:
                print(f"[Dataset Scan] Failed to process image {image_path}: {e}")

            # Periodic commit: flush accumulated changes and let SQLAlchemy release
            # them. With autoflush off and a single end-of-scan commit, every
            # touched caption ORM object stays pinned (dirty objects are strong-refs
            # in the unit of work) for the whole scan — so the session grows to
            # tens of thousands of objects and per-item cost climbs (measured ~4x
            # slower, with the growth ~11x faster for JSON sidecars that yield
            # ~11 caption rows per image). Committing in batches keeps the working
            # set bounded and per-item cost flat. Partial progress is committed,
            # which also matches the cancel/skip-commits-partial behaviour.
            if files_processed > 0 and files_processed % 300 == 0:
                try:
                    db.commit()
                except Exception as _ce:
                    print(f"[Dataset Scan] Periodic commit failed: {_ce}")

    # Run scan in thread pool to avoid blocking event loop (enables WebSocket progress updates)
    # SQLite is configured with check_same_thread=False, so cross-thread access is safe
    import asyncio
    loop = asyncio.get_event_loop()
    _scan_cancelled = False
    try:
        await loop.run_in_executor(None, lambda: scan_directory(dataset.path))
    except RescanSkipped:
        _scan_cancelled = True
        print(f"[Dataset Scan] Rescan skipped mid-walk for dataset {dataset_id}; "
              f"committing {items_found} new items, skipping purge")

    if _scan_cancelled:
        # Skip the purge: we did not finish seeing every on-disk file, so the
        # stale-path diff is incomplete and purging would wrongly delete items
        # we simply hadn't reached. Commit the new items/captions added so far
        # and leave last_scanned_at unchanged so the next pre-flight re-detects
        # drift (already-applied changes stay, per the skip contract).
        db.commit()
        db.refresh(dataset)
        manager.send_progress_sync(
            total_steps, total_steps,
            f"Rescan skipped: committed {items_found} new items (partial)"
        )
        return {
            "items_found": items_found,
            "captions_found": captions_found,
            "captions_updated": captions_updated,
            "items_purged": 0,
            "cancelled": True,
            "dataset": dataset.to_dict(),
        }

    # --- Purge: remove DB records whose files no longer exist on disk ---
    stale_paths = set(existing_paths.keys()) - seen_existing_paths
    items_purged = 0
    # For incremental mode: read purged captions BEFORE deletion so we can
    # subtract their tag counts from the existing tag_statistics.
    purged_tag_counts: dict[str, int] = {}   # tag -> count to subtract
    if stale_paths:
        stale_item_ids = [existing_paths[p] for p in stale_paths]
        if incremental and dataset.tag_statistics:
            import json as _json_purge
            purged_caps = db.query(DatasetCaption).filter(
                DatasetCaption.item_id.in_(stale_item_ids),
                DatasetCaption.caption_type == "tags",
            ).all()
            for cap in purged_caps:
                tags: list[str] = []
                if cap.tag_data:
                    try:
                        tags = [t.get("tag", "").strip() for t in _json_purge.loads(cap.tag_data)]
                    except Exception:
                        pass
                if not tags and cap.content:
                    tags = [t.strip() for t in cap.content.split(",")]
                for tag in tags:
                    if tag:
                        purged_tag_counts[tag] = purged_tag_counts.get(tag, 0) + 1
        # Delete captions first (foreign key), then items
        db.query(DatasetCaption).filter(
            DatasetCaption.item_id.in_(stale_item_ids)
        ).delete(synchronize_session=False)
        db.query(DatasetItem).filter(
            DatasetItem.id.in_(stale_item_ids)
        ).delete(synchronize_session=False)
        items_purged = len(stale_item_ids)
        print(f"[Dataset Scan] Purged {items_purged} items whose files no longer exist on disk")

    # File scan complete - progress is now at ~90%
    manager.send_progress_sync(
        total_images,
        total_steps,
        f"File scan complete: {files_processed} processed, {items_purged} purged | Starting tag statistics..."
    )

    # Compute tag statistics -----------------------------------------------
    # incremental=True (training rescan):
    #   Case 1: no structural change → keep existing stats as-is
    #   Case 2: structural change → differential update (add new, subtract purged)
    # incremental=False (regular UI scan): always full recompute
    if incremental:
        existing_stats: dict = dataset.tag_statistics or {}
        if items_found == 0 and items_purged == 0:
            # Case 1: nothing changed structurally — reuse cached stats
            print(f"[Dataset Scan] No structural change — reusing cached tag statistics ({len(existing_stats)} tags)")
            tag_statistics = existing_stats
        else:
            # Case 2: differential update
            print(f"[Dataset Scan] Incremental tag statistics update: -{items_purged} / +{items_found} items")
            import json as _json_incr
            stats: dict = {tag: dict(v) for tag, v in existing_stats.items()}

            # Subtract counts for purged items (collected before deletion)
            for tag, cnt in purged_tag_counts.items():
                if tag in stats:
                    stats[tag]["count"] -= cnt
                    if stats[tag]["count"] <= 0:
                        del stats[tag]

            # Add counts for new items (their captions are now in DB)
            if new_item_ids:
                new_caps = db.query(DatasetCaption).filter(
                    DatasetCaption.item_id.in_(new_item_ids),
                    DatasetCaption.caption_type == "tags",
                ).all()
                for cap in new_caps:
                    tag_cat_pairs: list[tuple[str, str]] = []
                    if cap.tag_data:
                        try:
                            tag_cat_pairs = [
                                (t.get("tag", "").strip(), t.get("category", "Unknown"))
                                for t in _json_incr.loads(cap.tag_data)
                            ]
                        except Exception:
                            pass
                    if not tag_cat_pairs and cap.content:
                        tag_cat_pairs = [(t.strip(), "Unknown") for t in cap.content.split(",")]
                    for tag, category in tag_cat_pairs:
                        if not tag:
                            continue
                        if tag in stats:
                            stats[tag]["count"] += 1
                            # Upgrade category if currently Unknown
                            if stats[tag]["category"] == "Unknown" and category != "Unknown":
                                stats[tag]["category"] = category
                        else:
                            # Resolve Unknown categories via taglist_cache
                            if category == "Unknown":
                                resolved = taglist_cache.get_categories_batch([tag])
                                category = resolved.get(tag, "Unknown")
                            stats[tag] = {"count": 1, "category": category}

            tag_statistics = stats
            print(f"[Dataset Scan] Incremental update complete: {len(tag_statistics)} unique tags")
    else:
        print(f"[Dataset Scan] Computing tag statistics...")
        tag_statistics = await compute_tag_statistics(dataset_id, db, send_progress=True, total_steps=total_steps, current_step=total_images)

    # Send final completion progress (per-field breakdown; image-with-field totals
    # are returned in the response's field_summary).
    manager.send_progress_sync(
        total_steps,
        total_steps,
        f"Scan complete: {items_found} new images, {items_purged} purged | "
        f"{_fstat_msg()} | {len(tag_statistics)} unique tags"
    )

    # Normalize is_tags_format by majority vote per caption_type
    # Prevents a few misdetected tag-format files from causing issues
    # in predominantly natural-language datasets (and vice versa)
    caption_types_in_dataset = db.query(DatasetCaption.caption_type).filter(
        DatasetCaption.item_id.in_(
            db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset_id)
        )
    ).distinct().all()
    for (ct,) in caption_types_in_dataset:
        captions_of_type = db.query(DatasetCaption).filter(
            DatasetCaption.item_id.in_(
                db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset_id)
            ),
            DatasetCaption.caption_type == ct
        ).all()
        if not captions_of_type:
            continue
        tags_count = sum(1 for c in captions_of_type if c.is_tags_format)
        nl_count = len(captions_of_type) - tags_count
        majority_is_tags = tags_count > nl_count
        minority_count = nl_count if majority_is_tags else tags_count
        if minority_count > 0:
            majority_type_name = "tags" if majority_is_tags else "natural_language"
            print(f"[Dataset Scan] caption_type='{ct}': {tags_count} tags, {nl_count} NL -> "
                  f"normalizing {minority_count} to {majority_type_name}")
            for c in captions_of_type:
                if c.is_tags_format != majority_is_tags:
                    c.is_tags_format = majority_is_tags
                    # Also update caption_type for default txt fields
                    if ct in ("tags", "natural_language"):
                        c.caption_type = majority_type_name

    # Update dataset statistics (count all items in DB, not just newly added).
    # total_tags = images with danbooru tags; total_captions = images with a
    # natural-language caption (metadata excluded). See _dataset_caption_item_counts.
    dataset.total_items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).count()
    dataset.total_tags, dataset.total_captions = _dataset_caption_item_counts(db, dataset_id)
    dataset.tag_statistics = tag_statistics
    dataset.last_scanned_at = datetime.utcnow()

    db.commit()
    db.refresh(dataset)

    # Per-field scan summary. The two training fields (tags / caption) report
    # updated this run, how many images currently HAVE that field, and the total
    # image count; "other" aggregates the metadata fields (image.*, source.*,
    # exif.*, …). total_tags/total_captions = images-with-that-field (see
    # _dataset_caption_item_counts).
    field_summary = {
        "total_images": dataset.total_items,
        "tags":    {"added": _fstats["tags_add"],  "updated": _fstats["tags_upd"],
                    "images_with": dataset.total_tags},
        "caption": {"added": _fstats["cap_add"],   "updated": _fstats["cap_upd"],
                    "images_with": dataset.total_captions},
        "other":   {"added": _fstats["other_add"], "updated": _fstats["other_upd"]},
    }

    response = {
        "items_found": items_found,
        "captions_found": captions_found,
        "captions_updated": captions_updated,
        "items_purged": items_purged,
        "field_summary": field_summary,
        "dataset": dataset.to_dict(),
    }

    # Include auto-detection result if detection was performed
    if structure_detection_result is not None:
        response["structure_detection"] = structure_detection_result

    return response

@router.get("/datasets/{dataset_id}/items")
async def list_dataset_items(
    dataset_id: int,
    page: int = 1,
    page_size: int = 50,
    search: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated tags to filter by
    db: Session = Depends(get_datasets_db)
):
    """List dataset items with pagination and search

    Args:
        dataset_id: Dataset ID
        page: Page number (1-indexed)
        page_size: Items per page
        search: Text search in filename (base_name)
        tags: Comma-separated tags to filter (e.g. "1girl,solo"). Item must contain ALL specified tags.
    """
    query = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id)

    # Filename search
    if search:
        query = query.filter(DatasetItem.base_name.like(f"%{search}%"))

    # Tag filter: Find items that have captions containing ALL specified tags
    if tags:
        tag_list = [t.strip().lower() for t in tags.split(',') if t.strip()]
        if tag_list:
            # Join with DatasetCaption table (caption_type = "tags")
            query = query.join(DatasetCaption, DatasetItem.id == DatasetCaption.item_id)
            query = query.filter(DatasetCaption.caption_type == "tags")

            # Filter by each tag (comma-separated in caption content)
            for tag in tag_list:
                # Match tag as whole word in comma-separated list
                query = query.filter(
                    func.lower(DatasetCaption.content).like(f"%{tag}%")
                )

    total = query.count()
    offset = (page - 1) * page_size
    items = query.order_by(DatasetItem.id).offset(offset).limit(page_size).all()

    return {
        "items": [item.to_dict() for item in items],
        "total": total,
        "page": page,
        "page_size": page_size
    }

@router.get("/datasets/{dataset_id}/items/ids")
async def get_all_dataset_item_ids(
    dataset_id: int,
    search: Optional[str] = None,
    tags: Optional[str] = None,
    db: Session = Depends(get_datasets_db)
):
    """Get all item IDs in dataset (with optional filters)

    Args:
        dataset_id: Dataset ID
        search: Text search in filename (base_name)
        tags: Comma-separated tags to filter by

    Returns:
        List of all matching item IDs
    """
    query = db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset_id)

    # Filename search
    if search:
        query = query.filter(DatasetItem.base_name.like(f"%{search}%"))

    # Tag filter
    if tags:
        tag_list = [t.strip().lower() for t in tags.split(',') if t.strip()]
        if tag_list:
            query = query.join(DatasetCaption, DatasetItem.id == DatasetCaption.item_id)
            query = query.filter(DatasetCaption.caption_type == "tags")
            for tag in tag_list:
                query = query.filter(
                    func.lower(DatasetCaption.content).like(f"%{tag}%")
                )

    # Get all IDs
    item_ids = [row[0] for row in query.order_by(DatasetItem.id).all()]

    return {
        "item_ids": item_ids,
        "total": len(item_ids)
    }

@router.get("/datasets/{dataset_id}/tags")
async def get_dataset_tags(
    dataset_id: int,
    db: Session = Depends(get_datasets_db)
):
    """Get all unique tags in dataset (from 'tags' caption type)

    Returns:
        List of unique tags across all items in the dataset
    """
    # Get all items in dataset
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()

    if not items:
        return {"tags": []}

    # Get all item IDs
    item_ids = [item.id for item in items]

    # Get all tag captions for these items
    tag_captions = db.query(DatasetCaption).filter(
        DatasetCaption.item_id.in_(item_ids),
        DatasetCaption.caption_type == "tags"
    ).all()

    # Extract unique tags
    unique_tags = set()
    for caption in tag_captions:
        if caption.content:
            tags = caption.content.split(",")
            for tag in tags:
                tag = tag.strip()
                if tag:
                    unique_tags.add(tag)

    return {"tags": sorted(list(unique_tags))}

@router.get("/datasets/{dataset_id}/items/{item_id}")
async def get_dataset_item(
    dataset_id: int,
    item_id: int,
    db: Session = Depends(get_datasets_db)
):
    """Get detailed dataset item with captions"""
    item = db.query(DatasetItem).filter(
        DatasetItem.dataset_id == dataset_id,
        DatasetItem.id == item_id
    ).first()

    if not item:
        raise HTTPException(status_code=404, detail="Dataset item not found")

    # Get all captions for this item
    captions = db.query(DatasetCaption).filter(DatasetCaption.item_id == item_id).all()

    result = item.to_dict()
    result["captions"] = [c.to_dict() for c in captions]

    return result

@router.get("/serve-image")
async def serve_image(path: str):
    """Serve image file from filesystem"""
    from fastapi.responses import FileResponse
    import os

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(path)

@router.get("/datasets/{dataset_id}/caption-types")
async def get_dataset_caption_types(
    dataset_id: int,
    db: Session = Depends(get_datasets_db)
):
    """Get available caption types with format detection info"""
    # Check dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Query caption types with aggregated format info
    from sqlalchemy import func

    results = db.query(
        DatasetCaption.caption_type,
        DatasetCaption.field_category,
        DatasetCaption.is_tags_format,
        DatasetCaption.source_field,
        func.count(DatasetCaption.id).label('count'),
        func.avg(DatasetCaption.tag_match_rate).label('avg_match_rate')
    ).join(DatasetItem).filter(
        DatasetItem.dataset_id == dataset_id
    ).group_by(
        DatasetCaption.caption_type,
        DatasetCaption.field_category,
        DatasetCaption.is_tags_format,
        DatasetCaption.source_field
    ).all()

    # Organize results by caption_type
    caption_types_dict = {}
    for caption_type, field_category, is_tags_format, source_field, count, avg_match_rate in results:
        if caption_type not in caption_types_dict:
            caption_types_dict[caption_type] = {
                "caption_type": caption_type,
                "total_count": 0,
                "field_category": field_category or "training",
                "is_tags_format": is_tags_format or False,
                "avg_match_rate": 0.0,
                "source_field": source_field,
                "subtypes": []
            }

        caption_types_dict[caption_type]["total_count"] += count
        # Average of averages (weighted by count would be better, but this is simpler)
        caption_types_dict[caption_type]["avg_match_rate"] = avg_match_rate or 0.0

    # Convert to list and sort: training first, then by count
    caption_types_list = sorted(
        caption_types_dict.values(),
        key=lambda x: (x["field_category"] != "training", -x["total_count"])
    )

    return {
        "caption_types": caption_types_list
    }

@router.get("/datasets/{dataset_id}/random-caption")
async def get_random_caption(
    dataset_id: int,
    caption_types: Optional[str] = None,  # Comma-separated caption types to filter
    db: Session = Depends(get_datasets_db)
):
    """Get a random caption from the dataset, processed exactly as training would.

    Mirrors the training caption pipeline (core/training/train_runner.get_dataset_items):
      1. Resolve the caption types the SAME way: explicit param ->
         dataset.caption_processing.caption_types -> auto (tags > natural_language).
      2. Select ONE caption per item by that priority (so e.g. tweet-body / other
         fields never leak in when "tags" is the training target).
      3. Apply the dataset's caption processing (normalize / shuffle / dropout /
         category order) so the preview matches the string actually fed to training,
         not the raw stored content.
    """
    import random as _random
    from sqlalchemy import func
    from core.training.caption_processor import (
        process_caption, process_caption_with_tag_data, get_default_caption_processing_config,
    )

    # Check dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    caption_config = dataset.caption_processing or get_default_caption_processing_config()

    # Resolve which caption types to use (same priority as training).
    if caption_types:
        selected_types = [t.strip() for t in caption_types.split(",") if t.strip()]
    elif caption_config.get("caption_types"):
        selected_types = list(caption_config.get("caption_types"))
    else:
        selected_types = None  # auto-select per item

    # Pick a random item that actually has a usable caption of the selected types.
    cap_q = db.query(DatasetCaption).join(DatasetItem).filter(DatasetItem.dataset_id == dataset_id)
    if selected_types:
        cap_q = cap_q.filter(DatasetCaption.caption_type.in_(selected_types))
    random_row = cap_q.order_by(func.random()).first()
    if random_row is None:
        raise HTTPException(status_code=404, detail="No captions found in dataset")

    item = db.query(DatasetItem).filter(DatasetItem.id == random_row.item_id).first()

    # Select the primary caption for this item by priority order (mirrors training,
    # which uses the first matching type per item — not a random caption row).
    priority = selected_types if selected_types else ["tags", "natural_language"]
    primary = None
    for ct in priority:
        primary = db.query(DatasetCaption).filter(
            DatasetCaption.item_id == item.id,
            DatasetCaption.caption_type == ct,
        ).first()
        if primary:
            break
    if primary is None and not selected_types:
        primary = db.query(DatasetCaption).filter(DatasetCaption.item_id == item.id).first()
    if primary is None:
        primary = random_row

    raw_caption = primary.content or ""
    is_tags_format = (
        primary.is_tags_format
        if hasattr(primary, "is_tags_format") and primary.is_tags_format is not None
        else True
    )

    # Random epoch so repeated previews reflect the per-epoch shuffle/dropout variety.
    epoch_num = _random.randint(0, 1000)

    if is_tags_format:
        tag_data = None
        if getattr(primary, "tag_data", None):
            import json
            try:
                tag_data = json.loads(primary.tag_data)
            except Exception:
                tag_data = None
        if tag_data:
            processed_caption = process_caption_with_tag_data(
                tag_data=tag_data,
                epoch_num=epoch_num,
                item_path=item.image_path,
                caption_config=caption_config,
            )
        else:
            processed_caption = process_caption(
                caption=raw_caption,
                epoch_num=epoch_num,
                item_path=item.image_path,
                normalize_tags=caption_config.get("normalize_tags", True),
                category_order=caption_config.get("category_order", None),
                caption_dropout_rate=caption_config.get("caption_dropout_rate", 0.0),
                token_dropout_rate=caption_config.get("token_dropout_rate", 0.0),
                keep_tokens=caption_config.get("keep_tokens", 0),
                shuffle_tokens=caption_config.get("shuffle_tokens", False),
                shuffle_per_epoch=caption_config.get("shuffle_per_epoch", False),
                shuffle_keep_first_n=caption_config.get("shuffle_keep_first_n", 0),
                shuffle_tag_groups=caption_config.get("shuffle_tag_groups", None),
                shuffle_groups_together=caption_config.get("shuffle_groups_together", False),
                tag_group_dir=caption_config.get("tag_group_dir", "taglist"),
                exclude_person_count_from_shuffle=caption_config.get("exclude_person_count_from_shuffle", False),
                tag_dropout_rate=caption_config.get("tag_dropout_rate", 0.0),
                tag_dropout_per_epoch=caption_config.get("tag_dropout_per_epoch", False),
                tag_dropout_keep_first_n=caption_config.get("tag_dropout_keep_first_n", 0),
                tag_dropout_category_rates=caption_config.get("tag_dropout_category_rates", {}),
                tag_dropout_exclude_person_count=caption_config.get("tag_dropout_exclude_person_count", False),
            )
    else:
        # Natural language: used as-is by training (no tag processing).
        processed_caption = raw_caption

    # Fetch reference images from the DatasetItem
    reference_images = []
    if item and item.related_images:
        reference_images = item.related_images.get("reference", [])

    return {
        "caption": processed_caption,
        "caption_type": primary.caption_type,
        "caption_subtype": getattr(primary, "caption_subtype", None),
        "item_id": item.id,
        "reference_images": reference_images,
    }

# ============================================================
# Dataset Item Caption Update API
# ============================================================

class CaptionUpdateRequest(BaseModel):
    caption_type: str = "tags"
    content: str
    tag_data: Optional[List[Dict[str, str]]] = None  # [{"tag": "1girl", "category": "General"}, ...]

@router.patch("/datasets/items/{item_id}/captions")
async def update_item_caption(
    item_id: int,
    request: CaptionUpdateRequest,
    db: Session = Depends(get_datasets_db)
):
    """Update caption for a dataset item"""
    # Check item exists
    item = db.query(DatasetItem).filter(DatasetItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Dataset item not found")

    # Get old caption content for tag statistics update
    old_content = None
    caption = db.query(DatasetCaption).filter(
        DatasetCaption.item_id == item_id,
        DatasetCaption.caption_type == request.caption_type
    ).first()

    if caption:
        old_content = caption.content
        # Update existing caption
        caption.content = request.content
        # Update tag_data if provided
        if request.tag_data is not None:
            import json
            caption.tag_data = json.dumps(request.tag_data)
        caption.updated_at = datetime.utcnow()
    else:
        # Create new caption
        tag_data_json = None
        if request.tag_data is not None:
            import json
            tag_data_json = json.dumps(request.tag_data)

        caption = DatasetCaption(
            item_id=item_id,
            caption_type=request.caption_type,
            content=request.content,
            tag_data=tag_data_json,
            source="manual"
        )
        db.add(caption)

    db.commit()
    db.refresh(caption)

    # Update tag statistics if this is a "tags" caption
    if request.caption_type == "tags":
        dataset = db.query(Dataset).filter(Dataset.id == item.dataset_id).first()
        if dataset and dataset.tag_statistics:
            tag_statistics = dataset.tag_statistics.copy()

            # Parse old and new tags
            old_tags = set()
            if old_content:
                old_tags = {tag.strip() for tag in old_content.split(",") if tag.strip()}

            new_tags = set()
            if request.content:
                new_tags = {tag.strip() for tag in request.content.split(",") if tag.strip()}

            # Tags removed
            removed_tags = old_tags - new_tags
            for tag in removed_tags:
                if tag in tag_statistics:
                    tag_statistics[tag]["count"] -= 1
                    if tag_statistics[tag]["count"] <= 0:
                        del tag_statistics[tag]

            # Tags added
            added_tags = new_tags - old_tags
            for tag in added_tags:
                if tag in tag_statistics:
                    tag_statistics[tag]["count"] += 1
                else:
                    # New tag - get category from tag_data if available
                    category = "Unknown"
                    if request.tag_data:
                        for item in request.tag_data:
                            if item.get("tag") == tag:
                                category = item.get("category", "Unknown")
                                break
                    tag_statistics[tag] = {
                        "count": 1,
                        "category": category
                    }

            # Save updated statistics
            dataset.tag_statistics = tag_statistics
            db.commit()

    return {"status": "success", "caption": caption.to_dict()}


# ============================================================
# Dataset Item Reference Images API
# ============================================================

class ReferenceImagesUpdateRequest(BaseModel):
    reference_images: List[str]  # List of file paths to reference images

@router.patch("/datasets/items/{item_id}/reference-images")
async def update_item_reference_images(
    item_id: int,
    request: ReferenceImagesUpdateRequest,
    db: Session = Depends(get_datasets_db)
):
    """Update reference images for a dataset item"""
    import os

    # Get item
    item = db.query(DatasetItem).filter(DatasetItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Dataset item not found")

    # Validate all paths exist
    invalid_paths = []
    for path in request.reference_images:
        if not os.path.exists(path):
            invalid_paths.append(path)

    if invalid_paths:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reference image paths: {', '.join(invalid_paths)}"
        )

    # Update related_images with reference key
    related_images = item.related_images or {}
    related_images["reference"] = request.reference_images

    # Use SQL update to handle JSON properly
    item.related_images = related_images
    item.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    print(f"[Dataset] Updated reference images for item {item_id}: {len(request.reference_images)} images")

    return {
        "status": "success",
        "item_id": item_id,
        "reference_images": request.reference_images
    }

@router.post("/datasets/items/{item_id}/reference-images/add")
async def add_item_reference_image(
    item_id: int,
    image_path: str = Form(...),
    db: Session = Depends(get_datasets_db)
):
    """Add a reference image to a dataset item"""
    import os

    # Get item
    item = db.query(DatasetItem).filter(DatasetItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Dataset item not found")

    # Validate path exists
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail=f"Image file not found: {image_path}")

    # Get current reference images
    related_images = item.related_images or {}
    reference_list = related_images.get("reference", [])

    # Check for duplicates
    if image_path in reference_list:
        return {"status": "already_exists", "item_id": item_id, "reference_images": reference_list}

    # Add new reference image
    reference_list.append(image_path)
    related_images["reference"] = reference_list

    item.related_images = related_images
    item.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    print(f"[Dataset] Added reference image to item {item_id}: {image_path}")

    return {
        "status": "success",
        "item_id": item_id,
        "reference_images": reference_list
    }

@router.delete("/datasets/items/{item_id}/reference-images")
async def remove_item_reference_image(
    item_id: int,
    image_path: str,
    db: Session = Depends(get_datasets_db)
):
    """Remove a reference image from a dataset item"""
    # Get item
    item = db.query(DatasetItem).filter(DatasetItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Dataset item not found")

    # Get current reference images
    related_images = item.related_images or {}
    reference_list = related_images.get("reference", [])

    # Check if image exists in list
    if image_path not in reference_list:
        raise HTTPException(status_code=404, detail=f"Reference image not found: {image_path}")

    # Remove the reference image
    reference_list.remove(image_path)
    related_images["reference"] = reference_list

    item.related_images = related_images
    item.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    print(f"[Dataset] Removed reference image from item {item_id}: {image_path}")

    return {
        "status": "success",
        "item_id": item_id,
        "reference_images": reference_list
    }


@router.post("/datasets/items/{item_id}/save-to-txt")
async def save_item_caption_to_txt(
    item_id: int,
    db: Session = Depends(get_datasets_db)
):
    """Save caption from DB to TXT/JSON file (auto-detect based on existing file)"""
    import os
    import json

    # Get item
    item = db.query(DatasetItem).filter(DatasetItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Dataset item not found")

    # Get tags caption
    caption = db.query(DatasetCaption).filter(
        DatasetCaption.item_id == item_id,
        DatasetCaption.caption_type == "tags"
    ).first()

    if not caption:
        # No caption to save, return success (nothing to do)
        return {"success": True, "message": "No tags caption found, nothing to save"}

    # Determine file paths
    image_path = item.image_path
    base_path = os.path.splitext(image_path)[0]
    txt_path = base_path + ".txt"
    json_path = base_path + ".json"

    saved_files = []

    try:
        # Check if TXT file exists and save to it
        if os.path.exists(txt_path):
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(caption.content)
            saved_files.append(txt_path)
            print(f"[Dataset] Saved caption to TXT: {txt_path}")

        # Check if JSON file exists and save to it
        if os.path.exists(json_path):
            try:
                # Read existing JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # Update caption field (tags)
                json_data['caption'] = caption.content

                # Write back
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

                saved_files.append(json_path)
                print(f"[Dataset] Saved caption to JSON: {json_path}")
            except Exception as json_err:
                print(f"[Dataset] Failed to update JSON file {json_path}: {json_err}")

        # If neither file exists, create a TXT file
        if not saved_files:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(caption.content)
            saved_files.append(txt_path)
            print(f"[Dataset] Created new TXT file: {txt_path}")

        return {
            "success": True,
            "message": f"Saved to {len(saved_files)} file(s): {', '.join(saved_files)}"
        }
    except Exception as e:
        print(f"[Dataset] Failed to save caption: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to write file: {str(e)}")

@router.post("/datasets/{dataset_id}/save-all-to-txt")
async def save_all_captions_to_txt(
    dataset_id: int,
    db: Session = Depends(get_datasets_db)
):
    """Save all captions from DB to TXT files"""
    import os

    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get all items
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()

    saved_count = 0
    failed_count = 0
    failed_items = []

    for item in items:
        # Get tags caption
        caption = db.query(DatasetCaption).filter(
            DatasetCaption.item_id == item.id,
            DatasetCaption.caption_type == "tags"
        ).first()

        if not caption:
            continue

        # Determine TXT file path
        image_path = item.image_path
        txt_path = os.path.splitext(image_path)[0] + ".txt"

        try:
            # Write to TXT file
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(caption.content)
            saved_count += 1
        except Exception as e:
            print(f"[Dataset] Failed to save caption to TXT {txt_path}: {e}")
            failed_count += 1
            failed_items.append({"item_id": item.id, "path": txt_path, "error": str(e)})

    print(f"[Dataset] Saved {saved_count} captions to TXT files, {failed_count} failed")
    return {
        "status": "success",
        "saved_count": saved_count,
        "failed_count": failed_count,
        "failed_items": failed_items
    }

@router.post("/datasets/items/{item_id}/restore-from-txt")
async def restore_item_caption_from_txt(
    item_id: int,
    db: Session = Depends(get_datasets_db)
):
    """Restore caption from TXT file to DB"""
    import os

    # Get item
    item = db.query(DatasetItem).filter(DatasetItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Dataset item not found")

    # Determine TXT file path
    image_path = item.image_path
    txt_path = os.path.splitext(image_path)[0] + ".txt"

    if not os.path.exists(txt_path):
        raise HTTPException(status_code=404, detail=f"TXT file not found: {txt_path}")

    try:
        # Read from TXT file
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Update or create caption
        caption = db.query(DatasetCaption).filter(
            DatasetCaption.item_id == item_id,
            DatasetCaption.caption_type == "tags"
        ).first()

        if caption:
            caption.content = content
            caption.source = "file"
            caption.updated_at = datetime.utcnow()
        else:
            caption = DatasetCaption(
                item_id=item_id,
                caption_type="tags",
                content=content,
                source="file"
            )
            db.add(caption)

        db.commit()
        db.refresh(caption)

        print(f"[Dataset] Restored caption from TXT: {txt_path}")
        return {"status": "success", "caption": caption.to_dict()}
    except Exception as e:
        print(f"[Dataset] Failed to restore caption from TXT: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read TXT file: {str(e)}")

# Tag Dictionary Search API was removed - frontend uses tagSuggestions.ts (JSON files) instead

# ============================================================
# Unified Taglist API (Phase 2: High-performance tag operations)
# ============================================================

from utils.taglist_cache import taglist_cache

# Initialize taglist cache on module load
taglist_cache.initialize(settings.root_dir)

class TagSearchRequest(BaseModel):
    q: str  # Search query (prefix)
    category: Optional[str] = None  # Category filter (general, character, artist, etc.)
    limit: int = 20  # Maximum results

class TagCategorizeRequest(BaseModel):
    tags: List[str]  # List of tags to categorize

@router.get("/tags/search")
async def search_tags(
    q: str,
    category: Optional[str] = None,
    limit: int = 20
):
    """
    High-speed tag autocomplete with prefix search.

    Uses server-side prefix index for O(1) lookup.
    Performance: <10ms for 1.5M tags

    Args:
        q: Search prefix (minimum 2 characters)
        category: Category filter (general, character, artist, copyright, meta, model)
        limit: Maximum results (default: 20)

    Returns:
        List of {tag, count, category} objects, sorted by count descending
    """
    if len(q) < 2:
        return {"results": []}

    results = taglist_cache.search_prefix(q, category=category, limit=limit)

    return {
        "results": [
            {"tag": tag, "count": count, "category": cat}
            for tag, count, cat in results
        ]
    }

@router.post("/tags/categorize")
async def categorize_tags(request: TagCategorizeRequest):
    """
    Batch tag categorization.

    Replaces frontend's 50MB taglist fetch with server-side O(1) lookup.
    Performance: <50ms for 100 tags

    Args:
        tags: List of tag strings

    Returns:
        Dict mapping tag -> category
    """
    categories = taglist_cache.get_categories_batch(request.tags)
    return {"categories": categories}

@router.get("/tags/stats")
async def get_tag_stats():
    """
    Get tag statistics summary.

    Returns:
        Dict of category -> tag count
    """
    stats = taglist_cache.get_stats()
    return {"stats": stats}

# ============================================================
# Training API Endpoints
# ============================================================

from database.models import TrainingRun, TrainingCheckpoint, TrainingSample

class DatasetConfigItem(BaseModel):
    dataset_id: int
    caption_types: List[str] = []  # Empty = use all caption types
    filters: Dict[str, Any] = {}  # {"tag_include": ["1girl"], "tag_exclude": ["photo"], "caption_contains": "smile"}
    ve_reconstruction_mode: Optional[bool] = False

class TrainingRunCreateRequest(BaseModel):
    dataset_id: Optional[int] = None  # Deprecated - use dataset_configs instead
    dataset_configs: Optional[List[DatasetConfigItem]] = None  # Multiple datasets with filters
    run_name: Optional[str] = None  # Optional - will use UUID if not provided
    training_method: str  # 'lora' | 'relora' | 'controlnet' | 'full_finetune' | 'vae_decoder'
    base_model_path: str

    # VAE decoder fine-tune (training_method == "vae_decoder"). A single nested
    # dict mirroring VAE_TRAINING_DEFAULTS (the SSOT) rather than ~30 flat
    # fields, because the VAE modality shares almost none of its knobs with the
    # diffusion trainers. Unknown keys are rejected by generate_vae_config;
    # omitted keys fall back to VAE_TRAINING_DEFAULTS. Ignored for every other
    # training_method.
    vae_config: Optional[Dict[str, Any]] = None

    # Training parameters
    total_steps: Optional[int] = None  # Mutually exclusive with epochs
    epochs: Optional[int] = None  # Mutually exclusive with total_steps
    batch_size: int = 1
    gradient_accumulation_steps: int = TRAINING_DEFAULTS["gradient_accumulation_steps"]
    max_grad_norm: float = TRAINING_DEFAULTS["max_grad_norm"]
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0  # Linear warmup steps before lr_scheduler kicks in
    # Plateau-then-cosine-floor LR scheduler ("plateau_cosine_floor"). Only
    # consumed when lr_scheduler == "plateau_cosine_floor".
    lr_decay_start_ratio: float = TRAINING_DEFAULTS["lr_decay_start_ratio"]
    lr_floor_ratio: float = TRAINING_DEFAULTS["lr_floor_ratio"]
    # Weight EMA (opt-in, default off)
    use_ema: bool = TRAINING_DEFAULTS["use_ema"]
    ema_decay: float = TRAINING_DEFAULTS["ema_decay"]
    ema_update_every: int = TRAINING_DEFAULTS["ema_update_every"]
    ema_device: str = TRAINING_DEFAULTS["ema_device"]
    # Options: adamw, adamw8bit, adamw8bit_ringbuffer, adafactor, lion8bit,
    # lion8bit_ringbuffer, paged_adamw, paged_adamw8bit, paged_lion8bit.
    # Paging is chosen by the NAME; there is deliberately no is_paged boolean
    # (one existed, reached BaseTrainer and was read by nothing). A request
    # that still sends optimizer_is_paged is ignored: this model does not
    # declare the field and Pydantic's default extra="ignore" drops it.
    optimizer: str = "adamw8bit"
    optimizer_cautious: bool = False
    optimizer_beta1: Optional[float] = None
    optimizer_beta2: Optional[float] = None
    optimizer_epsilon: Optional[float] = None
    optimizer_weight_decay: Optional[float] = None
    optimizer_schedule_free: bool = False  # Enable Schedule-Free optimizer (adamw8bit_ringbuffer, lion8bit_ringbuffer)
    optimizer_schedule_free_r: float = 0.0  # Schedule-Free r parameter (default: 0.0)
    optimizer_schedule_free_weight_lr_power: float = 2.0  # Schedule-Free weight lr power (default: 2.0)
    optimizer_use_radam: bool = False  # Use RAdam (Rectified Adam) with Schedule-Free (adamw8bit_ringbuffer, lion8bit_ringbuffer)
    # Stochastic rounding when writing BF16 parameter updates (adamw8bit_ringbuffer, lion8bit_ringbuffer)
    optimizer_stochastic_rounding: bool = TRAINING_DEFAULTS["optimizer_stochastic_rounding"]

    # LoRA specific
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 16
    lora_dtype: Optional[str] = "fp32"  # fp32, fp16, bf16 (LoRA weight dtype, independent of main model)
    network_type: Optional[str] = "lora"

    # Advanced
    save_every: int = 100
    save_every_unit: str = "steps"  # "steps" or "epochs"
    max_step_saves_to_keep: Optional[int] = None  # None = use training method default (LoRA:10, FullFT:3, ControlNet:5)
    sample_every: int = 100
    sample_prompts: List[Dict[str, str]] = []  # List of {positive: str, negative: str, condition_image_path?: str}
    resume_from_checkpoint: Optional[str] = None  # Checkpoint filename to resume from (e.g., "lora_step_100.safetensors")

    # Debug
    debug_latents: bool = False
    debug_latents_every: int = 50

    # Bucketing options
    enable_bucketing: bool = False
    base_resolutions: Optional[List[int]] = None  # e.g., [512, 768, 1024]
    bucket_strategy: str = "resize"  # "resize", "crop", "random_crop"
    multi_resolution_mode: str = "max"  # "max" or "random"
    # Epoch-dynamic crop augmentation (SDXL only). See param_defaults / design doc.
    crop_augment_enable: bool = TRAINING_DEFAULTS["crop_augment_enable"]
    crop_full_image_prob: float = TRAINING_DEFAULTS["crop_full_image_prob"]
    crop_max_bucket_prob: float = TRAINING_DEFAULTS["crop_max_bucket_prob"]
    crop_min_area_ratio: float = TRAINING_DEFAULTS["crop_min_area_ratio"]
    crop_min_short_side_px: int = TRAINING_DEFAULTS["crop_min_short_side_px"]
    crop_aspect_mode: str = TRAINING_DEFAULTS["crop_aspect_mode"]
    crop_position_mode: str = TRAINING_DEFAULTS["crop_position_mode"]
    crop_smaller_bucket_mode: str = TRAINING_DEFAULTS["crop_smaller_bucket_mode"]
    crop_smaller_scale_range: Optional[List[float]] = Field(default_factory=lambda: list(TRAINING_DEFAULTS["crop_smaller_scale_range"]))
    full_crop_position_mode: str = TRAINING_DEFAULTS["full_crop_position_mode"]
    crop_microcond_mode: str = TRAINING_DEFAULTS["crop_microcond_mode"]
    crop_plan_seed: int = TRAINING_DEFAULTS["crop_plan_seed"]
    cache_latents_to_disk: bool = False  # Cache VAE latents and text embeddings to disk (default: False, in-memory cache)
    force_recache: bool = False  # Force regeneration of disk latent cache
    reconstruction_loss_weight: float = 0.0  # Additional reconstruction loss weight (0.0 = disabled)

    # Component-specific training
    train_unet: bool = True
    train_text_encoder: bool = False
    train_image_encoder: bool = False  # Reserved for future image encoder training
    unet_lr: Optional[float] = None  # Defaults to learning_rate if None
    text_encoder_lr: Optional[float] = None  # Defaults to learning_rate if None
    text_encoder_1_lr: Optional[float] = None  # SDXL TE1 LR (defaults to text_encoder_lr if None)
    text_encoder_2_lr: Optional[float] = None  # SDXL TE2 LR (defaults to text_encoder_lr if None)
    image_encoder_lr: Optional[float] = None  # Reserved for future image encoder LR

    # Anima-specific LoRA training knobs (ignored for other architectures).
    # Single Source of Truth: backend/api/param_defaults.py TRAINING_DEFAULTS.
    anima_lora_scope: str = TRAINING_DEFAULTS["anima_lora_scope"]
    lens_lora_scope: str = TRAINING_DEFAULTS["lens_lora_scope"]
    lens_img_lr_factor: float = TRAINING_DEFAULTS["lens_img_lr_factor"]
    lens_txt_lr_factor: float = TRAINING_DEFAULTS["lens_txt_lr_factor"]
    ideogram4_lora_scope: str = TRAINING_DEFAULTS["ideogram4_lora_scope"]
    ideogram4_train_uncond: bool = TRAINING_DEFAULTS["ideogram4_train_uncond"]
    ideogram4_uncond_loss_weight: float = TRAINING_DEFAULTS["ideogram4_uncond_loss_weight"]
    ideogram4_lr_factor: float = TRAINING_DEFAULTS["ideogram4_lr_factor"]
    train_llm_adapter: bool = TRAINING_DEFAULTS["train_llm_adapter"]
    # MiniT2I (pixel-space MM-JiT) training.
    minit2i_lora_scope: str = TRAINING_DEFAULTS["minit2i_lora_scope"]
    minit2i_te_lora_scope: str = TRAINING_DEFAULTS["minit2i_te_lora_scope"]
    minit2i_label_drop_rate: float = TRAINING_DEFAULTS["minit2i_label_drop_rate"]
    minit2i_lr_factor: float = TRAINING_DEFAULTS["minit2i_lr_factor"]
    minit2i_flan_t5_path: str = TRAINING_DEFAULTS["minit2i_flan_t5_path"]
    minit2i_scratch_init_from: str = TRAINING_DEFAULTS["minit2i_scratch_init_from"]
    minit2i_inherit_final_layer: bool = TRAINING_DEFAULTS["minit2i_inherit_final_layer"]
    # Krea 2 (single-stream flow-matching MMDiT) training.
    krea2_lora_scope: str = TRAINING_DEFAULTS["krea2_lora_scope"]
    krea2_lr_factor: float = TRAINING_DEFAULTS["krea2_lr_factor"]
    krea2_discrete_flow_shift: float = TRAINING_DEFAULTS["krea2_discrete_flow_shift"]
    # REPA (Representation Alignment) — MiniT2I only.
    repa_enable: bool = TRAINING_DEFAULTS["repa_enable"]
    repa_encoder_source: str = TRAINING_DEFAULTS["repa_encoder_source"]
    repa_tagger_model_dir: str = TRAINING_DEFAULTS["repa_tagger_model_dir"]
    repa_siglip2_repo: str = TRAINING_DEFAULTS["repa_siglip2_repo"]
    repa_align_depth: int = TRAINING_DEFAULTS["repa_align_depth"]
    repa_weight: float = TRAINING_DEFAULTS["repa_weight"]
    repa_proj_lr_factor: float = TRAINING_DEFAULTS["repa_proj_lr_factor"]
    repa_encoder_resolution: int = TRAINING_DEFAULTS["repa_encoder_resolution"]
    # Anima full-parameter LR multipliers (each applied on top of unet_lr).
    anima_attn_mlp_lr_factor: float = TRAINING_DEFAULTS["anima_attn_mlp_lr_factor"]
    anima_mod_lr_factor: float = TRAINING_DEFAULTS["anima_mod_lr_factor"]
    anima_llm_adapter_lr_factor: float = TRAINING_DEFAULTS["anima_llm_adapter_lr_factor"]
    # TREAD token routing (arXiv 2501.04765) — training-only acceleration (Anima).
    tread_enable: bool = TRAINING_DEFAULTS["tread_enable"]
    tread_drop_ratio: float = TRAINING_DEFAULTS["tread_drop_ratio"]
    tread_start_block: int = TRAINING_DEFAULTS["tread_start_block"]
    tread_end_block: int = TRAINING_DEFAULTS["tread_end_block"]
    # Low-rate stochastic depth (per-batch block dropout) — training-only (Anima).
    block_skip_rate: float = TRAINING_DEFAULTS["block_skip_rate"]
    block_skip_protect_start: int = TRAINING_DEFAULTS["block_skip_protect_start"]
    block_skip_protect_end: int = TRAINING_DEFAULTS["block_skip_protect_end"]
    # DiT-BlockSkip (arXiv 2603.20755) — training-only MEMORY-REDUCTION for Anima LoRA.
    blockskip_enable: bool = TRAINING_DEFAULTS["blockskip_enable"]
    blockskip_front: int = TRAINING_DEFAULTS["blockskip_front"]
    blockskip_back: int = TRAINING_DEFAULTS["blockskip_back"]
    # Resolution curriculum (low-res warmup then switch to target) — training-only, arch-agnostic.
    res_curriculum_enable: bool = TRAINING_DEFAULTS["res_curriculum_enable"]
    res_curriculum_warmup_steps: int = TRAINING_DEFAULTS["res_curriculum_warmup_steps"]
    res_curriculum_warmup_scale: float = TRAINING_DEFAULTS["res_curriculum_warmup_scale"]
    # Full-parameter save: embed the VAE into the single-file checkpoint.
    # None = per-arch default (BUNDLE_VAE_DEFAULTS_BY_ARCH: sd15/sdxl/deus True,
    # others False); an explicit boolean always wins.
    bundle_vae: Optional[bool] = TRAINING_DEFAULTS["bundle_vae"]
    # Gradient checkpointing (activation recompute). Default True = prior behavior.
    gradient_checkpointing: bool = TRAINING_DEFAULTS["gradient_checkpointing"]
    # Anima Phase D memory optimisations.
    cpu_offload_checkpointing: bool = TRAINING_DEFAULTS["cpu_offload_checkpointing"]
    async_cpu_offload_checkpointing: bool = TRAINING_DEFAULTS["async_cpu_offload_checkpointing"]
    fp8_base_dtype: Optional[str] = TRAINING_DEFAULTS["fp8_base_dtype"]
    # MiniMax-H3 only: weight of the audio half of its joint objective. `ge=0`
    # matches openapi's `minimum: 0.0` and is not cosmetic -- a negative weight
    # would INVERT the audio gradient, training the audio head away from its
    # target while the video half trains toward it.
    audio_loss_weight: float = Field(TRAINING_DEFAULTS["audio_loss_weight"], ge=0.0)
    # torch.compile (opt-in DiT training acceleration). "off" (default) |
    # "default" | "reduce-overhead" | "max-autotune-no-cudagraphs". Gated to
    # DiT full-parameter FT; skipped for LoRA / block swap; falls back to eager
    # on Inductor failure. See api/param_defaults.py for semantics.
    torch_compile: str = TRAINING_DEFAULTS["torch_compile"]
    torch_compile_dynamic: Optional[bool] = TRAINING_DEFAULTS["torch_compile_dynamic"]

    # Online Danbooru augmentation (image-generation). SSOT: TRAINING_DEFAULTS.
    # No vocabulary expansion (diffusion text conditioning is open-vocab);
    # interrupt-batch injection of extra Danbooru images only.
    danbooru_aug_enable: bool = TRAINING_DEFAULTS["danbooru_aug_enable"]
    danbooru_aug_queries: str = TRAINING_DEFAULTS["danbooru_aug_queries"]
    danbooru_aug_weight_static: float = TRAINING_DEFAULTS["danbooru_aug_weight_static"]
    danbooru_aug_deficiency_enable: bool = TRAINING_DEFAULTS["danbooru_aug_deficiency_enable"]
    danbooru_aug_deficiency_min_count: int = TRAINING_DEFAULTS["danbooru_aug_deficiency_min_count"]
    danbooru_aug_deficiency_top_k: int = TRAINING_DEFAULTS["danbooru_aug_deficiency_top_k"]
    danbooru_aug_deficiency_manual: str = TRAINING_DEFAULTS["danbooru_aug_deficiency_manual"]
    danbooru_aug_weight_deficiency: float = TRAINING_DEFAULTS["danbooru_aug_weight_deficiency"]
    danbooru_aug_injection_interval: int = TRAINING_DEFAULTS["danbooru_aug_injection_interval"]
    danbooru_aug_injection_ratio: float = TRAINING_DEFAULTS["danbooru_aug_injection_ratio"]
    danbooru_aug_min_score: int = TRAINING_DEFAULTS["danbooru_aug_min_score"]
    danbooru_aug_max_posts_per_query: int = TRAINING_DEFAULTS["danbooru_aug_max_posts_per_query"]
    danbooru_aug_api_interval: float = TRAINING_DEFAULTS["danbooru_aug_api_interval"]
    danbooru_aug_dl_speed_kbps: int = TRAINING_DEFAULTS["danbooru_aug_dl_speed_kbps"]
    danbooru_speed_check_enable: bool = TRAINING_DEFAULTS["danbooru_speed_check_enable"]
    danbooru_speed_degraded_kbps: int = TRAINING_DEFAULTS["danbooru_speed_degraded_kbps"]
    danbooru_speed_min_slow_streak: int = TRAINING_DEFAULTS["danbooru_speed_min_slow_streak"]
    danbooru_speed_min_slow_seconds: int = TRAINING_DEFAULTS["danbooru_speed_min_slow_seconds"]
    danbooru_speed_cooldown_seconds: int = TRAINING_DEFAULTS["danbooru_speed_cooldown_seconds"]
    danbooru_aug_buffer_size: Optional[int] = TRAINING_DEFAULTS["danbooru_aug_buffer_size"]
    danbooru_aug_include_rating_tag: bool = TRAINING_DEFAULTS["danbooru_aug_include_rating_tag"]
    danbooru_aug_max_caption_tags: int = TRAINING_DEFAULTS["danbooru_aug_max_caption_tags"]
    danbooru_quality_tag_enable: bool = TRAINING_DEFAULTS["danbooru_quality_tag_enable"]
    danbooru_quality_tag_thresholds: str = TRAINING_DEFAULTS["danbooru_quality_tag_thresholds"]
    danbooru_quality_tag_attach_negative: bool = TRAINING_DEFAULTS["danbooru_quality_tag_attach_negative"]
    danbooru_aug_shuffle_tags: bool = TRAINING_DEFAULTS["danbooru_aug_shuffle_tags"]
    danbooru_aug_shuffle_keep_first_n: int = TRAINING_DEFAULTS["danbooru_aug_shuffle_keep_first_n"]
    danbooru_aug_tag_dropout_rate: float = TRAINING_DEFAULTS["danbooru_aug_tag_dropout_rate"]
    danbooru_aug_tag_dropout_keep_first_n: int = TRAINING_DEFAULTS["danbooru_aug_tag_dropout_keep_first_n"]
    danbooru_aug_caption_dropout_rate: float = TRAINING_DEFAULTS["danbooru_aug_caption_dropout_rate"]
    danbooru_aug_keep_tokens: int = TRAINING_DEFAULTS["danbooru_aug_keep_tokens"]

    # Precision and dtype settings (VRAM optimization)
    weight_dtype: str = "fp16"  # fp16, fp32, bf16, fp8_e4m3fn, fp8_e5m2
    training_dtype: str = "fp16"  # fp16, bf16, fp8_e4m3fn, fp8_e5m2 (activation dtype during training)
    output_dtype: str = "fp32"  # fp32, fp16, bf16, fp8_e4m3fn, fp8_e5m2 (output latent dtype)
    vae_dtype: str = "fp16"  # VAE-specific dtype (SDXL VAE works fine with fp16)
    mixed_precision: bool = True  # Enable mixed precision training (autocast)
    use_flash_attention: bool = False  # DEPRECATED compat mirror of attention_backend (see below)
    # Attention backend selector for training (single source of truth: param_defaults).
    # "native" (SDPA) | "flash" (FlashAttention). "sage" is inference-only (no backward
    # kernel) and is refused/downgraded to native by resolve_backend at every training hook.
    # When present this wins over the legacy use_flash_attention boolean (training_config.py
    # derives use_flash_attention = attention_backend != "native").
    attention_backend: Optional[str] = TRAINING_DEFAULTS["attention_backend"]
    # Attention implementation selector for training. "conduit" (default) routes
    # through the unified backend/core/attention dispatch; "diffusers" reproduces
    # the pre-migration set_attention_backend path. Orthogonal to attention_backend
    # (which selects WHICH kernel). Persisted into the run config via training_config
    # so resumes reproduce the same registry. TRAINING-ONLY this pass.
    attention_impl: Optional[str] = TRAINING_DEFAULTS["attention_impl"]
    min_snr_gamma: float = 5.0  # Min-SNR gamma for loss weighting (default: 5.0, set to 0 to disable)

    # Text encoding settings
    text_encoding_mode: str = TRAINING_DEFAULTS["text_encoding_mode"]  # swap_onthefly|pre_encoded_cache|onthefly_gpu|cpu_prefetch
    text_encoding_swap_interval: int = TRAINING_DEFAULTS["text_encoding_swap_interval"]
    text_encoding_prefetch_depth: int = TRAINING_DEFAULTS["text_encoding_prefetch_depth"]

    # Latent encoding settings
    latent_encoding_mode: str = "swap_onthefly"  # "swap_onthefly", "pre_encoded_cache", "onthefly_gpu"
    latent_encoding_swap_interval: int = 256  # Swap interval for swap_onthefly mode

    # Block Swap settings (training VRAM optimization)
    blocks_to_swap: int = 0  # Number of transformer blocks to swap (0 to disable)
    use_pinned_memory: bool = False  # Use CUDA pinned memory for faster transfer
    block_swap_h2d_only: bool = TRAINING_DEFAULTS["block_swap_h2d_only"]  # FLUX.2 LoRA: H2D-only swap (no device->host of frozen base)
    block_swap_ring_size: int = TRAINING_DEFAULTS["block_swap_ring_size"]  # GPU weight-buffer ring slots (>=1)
    num_optimizer_groups: int = 0  # Number of optimizer groups for fused optimizer (0 to disable, recommended 4-10)

    # Per-bucket activation offload dispatcher (proactive, OOM-detection-free)
    activation_dispatch_enable: bool = TRAINING_DEFAULTS["activation_dispatch_enable"]
    activation_dispatch_margin_gb: float = TRAINING_DEFAULTS["activation_dispatch_margin_gb"]
    activation_dispatch_seed_coef: float = TRAINING_DEFAULTS["activation_dispatch_seed_coef"]
    activation_dispatch_residual_frac: float = TRAINING_DEFAULTS["activation_dispatch_residual_frac"]
    activation_dispatch_threshold_mb: int = TRAINING_DEFAULTS["activation_dispatch_threshold_mb"]

    # Multi Noise-Timestep (MNT) settings
    multi_noise_timesteps: int = 1  # Number of different timesteps per batch (default: 1, disable MNT)
    multi_noise_mode: str = "independent"  # "independent" or "trajectory_blend"
    trajectory_blend_alpha: float = 0.7  # Blend strength for trajectory_blend mode
    timestep_sampling: Optional[Dict[str, Any]] = None  # Timestep sampling config (distribution, min/max)

    # Regularization settings (prevent overbaking)
    regularization_type: Optional[str] = None  # "snr", "energy", or None
    snr_regularization_weight: float = 0.1
    snr_timestep_adaptive: bool = True
    snr_penalty_mode: str = "relu"
    energy_regularization_weight: float = 0.05
    energy_timestep_adaptive: bool = True
    energy_penalty_mode: str = "abs"
    energy_normalize_by_pixels: bool = True

    # Sample generation parameters
    sample_width: int = 1024
    sample_height: int = 1024
    sample_steps: int = 28
    sample_cfg_scale: float = 7.0
    sample_sampler: str = "euler"
    sample_schedule_type: str = "sgm_uniform"
    sample_seed: int = -1  # -1 for random

    # Unified Training Framework (Phase 2)
    noise_process: str = "auto"  # "auto", "ddpm", "flow"
    prediction_target: str = "auto"  # "auto", "epsilon", "velocity", "sample"
    strict_validation: bool = False  # Abort training if mismatch detected
    sdxl_micro_conditioning: bool = TRAINING_DEFAULTS["sdxl_micro_conditioning"]
    sdxl_vae_type: str = TRAINING_DEFAULTS["sdxl_vae_type"]
    sdxl_te_type: str = TRAINING_DEFAULTS["sdxl_te_type"]
    sdxl_te_hidden_layer: int = TRAINING_DEFAULTS["sdxl_te_hidden_layer"]
    sdxl_te_max_len: int = TRAINING_DEFAULTS["sdxl_te_max_len"]
    sdxl_te_train_encoder: bool = TRAINING_DEFAULTS["sdxl_te_train_encoder"]

    # Reference image conditioning (FLUX.2 only)
    use_reference_images: bool = False  # Enable reference image latent conditioning during training

    # SigLIP2 Vision Encoder
    vision_encoder_path: Optional[str] = None  # Path to SigLIP2 vision encoder safetensors
    train_vision_encoder: bool = False  # Train vision encoder weights
    vision_encoder_lr: Optional[float] = None  # Learning rate for vision encoder (defaults to text_encoder_lr)
    gradient_routing_ve: bool = False  # Block TE gradient when batch has reference images

    # Parameter change tracking
    param_tracking: bool = False  # Track per-component parameter change norms
    param_tracking_interval: int = 100  # Compute tracking every N steps

    # Priority training
    priority_training: Optional[Dict[str, Any]] = None  # Inline priority training config

    # ReLoRA-specific parameters
    relora_merge_every: int = 500  # Steps/epochs between merge-reinit cycles
    relora_merge_unit: str = "steps"  # "steps" or "epochs"
    restart_warmup_steps: int = 100  # Warmup steps after each merge cycle
    optimizer_reset_strategy: str = "full_reset"  # "full_reset", "magnitude_pruning", "random_pruning"
    optimizer_pruning_ratio: float = 0.9  # Pruning ratio for pruning strategies (0.0-1.0)

    # ControlNet-specific parameters
    controlnet_type: str = "standard"  # "standard" (diffusers ControlNetModel) or "lllite" (sd-scripts compatible)
    controlnet_pretrained_path: Optional[str] = None  # Path to existing ControlNet checkpoint for resume
    controlnet_init_from_unet: bool = True  # Initialize ControlNet from base UNet weights (standard only)
    lllite_conditioning_channels: int = 32  # Conditioning channels for LLLite
    lllite_rank: int = 64  # Rank for LLLite linear layers
    condition_preprocessors: Optional[List[str]] = None  # controlnet-aux preprocessor types (e.g., ["canny", "hed"])
    condition_cache_mode: str = "on_the_fly"  # "pre_generate" or "on_the_fly"
    # Outpaint-native conditioning (PART B)
    conditioning_mode: str = "preprocessor"  # "preprocessor" (default) or "outpaint"
    outpaint_crop_min_area: float = 0.15  # Minimum cropped-region area fraction for outpaint conditioning
    outpaint_crop_max_area: float = 0.8  # Maximum cropped-region area fraction for outpaint conditioning
    outpaint_edge_anchor_prob: float = 0.34  # Probability of anchoring the crop to an edge
    outpaint_corner_anchor_prob: float = 0.33  # Probability of anchoring the crop to a corner
    outpaint_mask_channel: bool = True  # Include a known/unknown mask as an extra conditioning channel
    outpaint_known_loss_weight: float = Field(0.3, ge=0.0, lt=0.5)  # Loss weight applied to the known (non-outpainted) region; must stay below the gen-mask threshold (0.5)
    outpaint_seam_loss_boost: float = 0.0  # Additional loss weight applied at the known/unknown seam
    outpaint_seam_ring_width: int = Field(TRAINING_DEFAULTS["outpaint_seam_ring_width"], ge=1, le=2)  # Number of seam rings the boost covers (1 = current 1-cell ring, 2 adds a second ring at half the boost increment)
    outpaint_seam_grad_lambda: float = Field(TRAINING_DEFAULTS["outpaint_seam_grad_lambda"], ge=0.0)  # Weight of the cross-seam error-continuity term (native prediction space); 0.0 disables it
    outpaint_loss_normalize: bool = TRAINING_DEFAULTS["outpaint_loss_normalize"]  # Divide each sample's weighted loss by its mean weight (decouples loss scale from rect area)
    outpaint_edge_feather_min_px: float = Field(TRAINING_DEFAULTS["outpaint_edge_feather_min_px"], ge=0.0)  # Minimum per-sample randomized crop_mask conditioning edge-softness (canvas px); 0/0 range disables (razor-sharp, current behavior)
    outpaint_edge_feather_max_px: float = Field(TRAINING_DEFAULTS["outpaint_edge_feather_max_px"], ge=0.0)  # Maximum per-sample randomized crop_mask conditioning edge-softness (canvas px)
    # sample_condition_image_path is now per-prompt in sample_prompts[].condition_image_path
    # Pre-flight: detect dataset drift + auto-rescan + cleanup orphan
    # latent cache.  Four modes (see core/training/dataset_drift.py):
    #   "off"   — skip entirely (default)
    #   "path"  — only detect added/missing files (cheap path set-diff)
    #   "smart" — path drift + caption sidecar mtime check (full coverage)
    #   "force" — always rescan, no drift detection (most expensive)
    # Legacy bool also accepted: True→"path", False→"off".
    rescan_before_training: Any = TRAINING_DEFAULTS["rescan_before_training"]

@router.post("/training/runs", status_code=201)
async def create_training_run(
    request: TrainingRunCreateRequest,
    training_db: Session = Depends(get_training_db),
    datasets_db: Session = Depends(get_datasets_db)
):
    """Create a new training run"""
    print(f"[Training] Creating training run: {request.run_name}")
    print(f"[Training] Request data: dataset_configs={request.dataset_configs}, method={request.training_method}")
    print(f"[Training] Steps={request.total_steps}, Epochs={request.epochs}, LR={request.learning_rate}")
    try:
        # Validate that either steps or epochs is provided
        if request.total_steps is None and request.epochs is None:
            raise HTTPException(status_code=400, detail="Either total_steps or epochs must be provided")
        if request.total_steps is not None and request.epochs is not None:
            raise HTTPException(status_code=400, detail="Cannot specify both total_steps and epochs")

        # Handle dataset_configs (new format) or fallback to dataset_id (legacy)
        if request.dataset_configs:
            dataset_configs = [config.dict() for config in request.dataset_configs]
            # Validate all datasets exist
            for config in dataset_configs:
                dataset = datasets_db.query(Dataset).filter(Dataset.id == config["dataset_id"]).first()
                if not dataset:
                    raise HTTPException(status_code=404, detail=f"Dataset ID {config['dataset_id']} not found")
            # Use first dataset as primary (for backward compatibility)
            primary_dataset_id = dataset_configs[0]["dataset_id"]
            primary_dataset = datasets_db.query(Dataset).filter(Dataset.id == primary_dataset_id).first()
        elif request.dataset_id:
            # Legacy single dataset mode
            dataset_configs = [{
                "dataset_id": request.dataset_id,
                "caption_types": [],
                "filters": {}
            }]
            primary_dataset_id = request.dataset_id
            primary_dataset = datasets_db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
            if not primary_dataset:
                raise HTTPException(status_code=404, detail="Dataset not found")
        else:
            raise HTTPException(status_code=400, detail="Either dataset_id or dataset_configs must be provided")

        # Build dataset_configs_for_yaml (with path, caption_types, and dataset_id)
        # NOTE: caption_processing is NOT saved to YAML - read from database at training time
        # Dataset-level params (caption_types, ve_reconstruction_mode, etc.) are
        # automatically propagated via extract_dataset_params() from dataset_params.py
        from core.training.dataset_params import extract_dataset_params
        dataset_configs_for_yaml = []
        for config in dataset_configs:
            dataset = datasets_db.query(Dataset).filter(Dataset.id == config["dataset_id"]).first()
            if dataset:
                yaml_config = {
                    "dataset_id": config["dataset_id"],  # Include dataset_id for YAML editing support
                    "path": dataset.path,
                    **extract_dataset_params(config),
                }
                dataset_configs_for_yaml.append(yaml_config)

        # Generate run_id and auto-generate run_name if not provided
        import uuid
        from datetime import datetime
        run_id = str(uuid.uuid4())

        if request.run_name:
            run_name = request.run_name
        else:
            # Auto-generate: YYYYMMDD_HHMMSS_<first 8 chars of UUID>
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            uuid_short = run_id.split('-')[0]  # First segment of UUID (8 chars)
            run_name = f"{timestamp}_{uuid_short}"

        # Check if run name is unique
        existing = training_db.query(TrainingRun).filter(TrainingRun.run_name == run_name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Training run '{run_name}' already exists")

        # Check if base model exists. The from-scratch MiniT2I sentinel
        # ("scratch:minit2i:<variant>:<vae_type>") is not a filesystem path: the
        # trainer builds a random-initialized model in memory, so skip the check.
        if not request.base_model_path.startswith("scratch:minit2i:") \
                and not os.path.exists(request.base_model_path):
            raise HTTPException(status_code=400, detail=f"Base model not found: {request.base_model_path}")

        # Create output directory (use training base dir from user settings or default)
        from core.training.training_utils import get_training_base_dir
        training_base_dir = Path(get_training_base_dir())

        # If relative path, resolve from project root
        if not training_base_dir.is_absolute():
            project_root = Path(__file__).parent.parent.parent  # backend/api/routes.py -> project root
            training_base_dir = project_root / training_base_dir

        output_dir = training_base_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_str = str(output_dir)

        # Get resume setting from request
        resume_from_checkpoint = request.resume_from_checkpoint

        # Resolve temp_img:// references in sample_prompts condition_image_path
        resolved_sample_prompts = []
        for sp in (request.sample_prompts or []):
            prompt_dict = dict(sp) if not isinstance(sp, dict) else sp.copy()
            cip = prompt_dict.get("condition_image_path", "")
            if cip and cip.startswith("temp_img://"):
                image_id = cip[len("temp_img://"):]
                resolved_path = os.path.join(TEMP_DIR, image_id)
                if os.path.exists(resolved_path):
                    prompt_dict["condition_image_path"] = resolved_path
                else:
                    print(f"[Training] WARNING: temp image not found: {resolved_path}")
                    prompt_dict["condition_image_path"] = ""
            resolved_sample_prompts.append(prompt_dict)

        # Calculate total_steps (for the DB column, and for the VAE generator
        # which has no epochs concept of its own). Computed BEFORE the YAML
        # dispatch so `epochs` is not silently dropped for vae_decoder runs;
        # depends only on the request + datasets_db, so the move is inert for
        # every other training method.
        calculated_total_steps = request.total_steps
        if request.epochs is not None:
            # Count items across all configured datasets (with filters applied)
            total_dataset_size = 0
            for config in dataset_configs:
                query = datasets_db.query(DatasetItem).filter(DatasetItem.dataset_id == config["dataset_id"])
                # TODO: Apply filters here when filter logic is implemented
                dataset_size = query.count()
                total_dataset_size += dataset_size

            if total_dataset_size == 0:
                raise HTTPException(status_code=400, detail="No items in configured datasets")
            calculated_total_steps = (total_dataset_size // request.batch_size) * request.epochs
            if calculated_total_steps == 0:
                calculated_total_steps = total_dataset_size * request.epochs  # Fallback if batch_size > dataset_size

        if calculated_total_steps is None or calculated_total_steps <= 0:
            raise HTTPException(status_code=400, detail=f"Invalid total_steps calculation: {calculated_total_steps}")

        print(f"[Training] Calculated total_steps: {calculated_total_steps}")

        # Generate YAML config
        config_generator = TrainingConfigGenerator()

        # Build params dict from Pydantic request, plus resume_from_checkpoint
        # which has special handling (resolved separately above).
        params_dict = request.model_dump()
        params_dict["resume_from_checkpoint"] = resume_from_checkpoint
        # Which fields the caller ACTUALLY sent. model_dump() materialises every
        # Pydantic default as non-None, so a generator cannot otherwise tell
        # "user asked for lr=1e-4" from "the model's default happens to be 1e-4".
        # generate_vae_config gates its flat-field tier on this.
        params_dict["_explicit_fields"] = sorted(request.model_fields_set)

        common_kwargs = {
            "run_name": run_name,
            "base_model_path": request.base_model_path,
            "output_dir": output_dir_str,
            "dataset_path": primary_dataset.path,  # Backward compat
            "dataset_configs": dataset_configs_for_yaml,
            "sample_prompts": resolved_sample_prompts,
            "caption_processing": primary_dataset.caption_processing,
        }

        if request.training_method == "lora":
            config_yaml = config_generator.generate_lora_config(params_dict, **common_kwargs)
        elif request.training_method == "relora":
            config_yaml = config_generator.generate_relora_config(params_dict, **common_kwargs)
        elif request.training_method == "controlnet":
            config_yaml = config_generator.generate_controlnet_config(params_dict, **common_kwargs)
        elif request.training_method == "vae_decoder":
            # The VAE trainer is step-based only, so `epochs` is resolved to the
            # step count here rather than silently falling back to the default.
            config_yaml = config_generator.generate_vae_config(
                {**params_dict, "total_steps": calculated_total_steps,
                 "_explicit_fields": sorted(set(params_dict["_explicit_fields"])
                                            | {"total_steps"})},
                **common_kwargs)
        else:  # full_finetune
            config_yaml = config_generator.generate_full_finetune_config(params_dict, **common_kwargs)

        # Save config file
        config_path = os.path.join(output_dir_str, f"{run_name}_config.yaml")
        config_generator.save_config(config_yaml, config_path)

        # Create training run with specified run_id and run_name
        training_run = TrainingRun(
            dataset_id=primary_dataset_id,  # Keep for backward compatibility
            dataset_configs=dataset_configs,  # New: multiple datasets
            run_id=run_id,
            run_name=run_name,
            training_method=request.training_method,
            base_model_path=request.base_model_path,
            config_yaml=config_yaml,
            total_steps=calculated_total_steps,
            output_dir=output_dir_str,
            status="pending"
        )

        training_db.add(training_run)
        training_db.commit()
        training_db.refresh(training_run)

        return training_run.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Training] ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        training_db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/runs")
async def list_training_runs(db: Session = Depends(get_training_db)):
    """List all training runs.

    Returns the summary projection: skips ``config_yaml`` (multi-KB
    Text column), the YAML-parsed ``unet_lr`` / ``text_encoder_*_lr``
    fields, the ``checkpoints`` relationship (avoids an N+1 once that
    table has rows), and ``dataset_configs``.  The list UI doesn't
    render any of those; the detail endpoint
    ``GET /training/runs/{id}`` returns the full payload.
    """
    from sqlalchemy.orm import defer, raiseload
    try:
        runs = (
            db.query(TrainingRun)
            .options(
                defer(TrainingRun.config_yaml),
                defer(TrainingRun.dataset_configs),
                raiseload(TrainingRun.checkpoints),
            )
            .order_by(TrainingRun.created_at.desc())
            .all()
        )
        return {
            "runs":  [run.to_dict(summary=True) for run in runs],
            "total": len(runs),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/runs/{run_id}")
async def get_training_run(run_id: int, db: Session = Depends(get_training_db)):
    """Get training run details"""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run.to_dict()


@router.get("/training/runs/{run_id}/danbooru-metrics")
async def get_training_danbooru_metrics(run_id: int, db: Session = Depends(get_training_db)):
    """Return online Danbooru augmentation metrics for an image-generation run.

    Reads ``{output_dir}/danbooru_metrics.json`` written periodically by the
    trainer (every 25 base steps).  Returns ``enabled=false`` when the file is
    missing (augmentation disabled or no steps written yet).
    """
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    if not run.output_dir or not os.path.isdir(run.output_dir):
        return {"enabled": False}
    path = os.path.join(run.output_dir, "danbooru_metrics.json")
    if not os.path.isfile(path):
        return {"enabled": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["enabled"] = True
        return data
    except Exception as e:
        return {"enabled": False, "error": str(e)}


def _write_danbooru_resume(output_dir: str) -> None:
    """Write the manual-resume control file the collection worker polls
    ({output_dir}/danbooru_control.json). Atomic (tmp -> replace) so the worker
    never reads a half-written file. Works for both the in-process tagger and the
    image-gen subprocess."""
    import time as _time
    ctl = {"resume_requested_at": _time.time()}
    path = os.path.join(output_dir, "danbooru_control.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ctl, f)
    os.replace(tmp, path)


@router.post("/training/runs/{run_id}/danbooru/resume")
async def resume_training_danbooru(run_id: int, db: Session = Depends(get_training_db)):
    """Manually resume Danbooru collection after a speed-degradation cooldown
    (image-generation run). Clears the active cooldown on the trainer's next poll."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    if not run.output_dir or not os.path.isdir(run.output_dir):
        raise HTTPException(status_code=400, detail="Run output dir unavailable")
    _write_danbooru_resume(run.output_dir)
    return {"success": True}


# YAML field locations for fields that don't live in process_config.train with the same name.
# Format: field_name -> (section_path, [yaml_key])  yaml_key defaults to field_name.
# section_path is dotted: "dtype", "save", "sample", "network", "network.controlnet", "model"
_YAML_FIELD_LOCATIONS: Dict[str, tuple] = {
    # dtype section
    "weight_dtype": ("dtype", "weight"),
    "training_dtype": ("dtype", "training"),
    "vae_dtype": ("dtype", "vae"),
    "output_dtype": ("dtype", "save"),
    # save section
    "save_every": ("save",),
    "save_every_unit": ("save",),
    "max_step_saves_to_keep": ("save",),
    # sample section (some keys are renamed in YAML)
    "sample_every": ("sample",),
    "sample_prompts": ("sample", "prompts"),
    "sample_width": ("sample", "width"),
    "sample_height": ("sample", "height"),
    "sample_steps": ("sample", "sample_steps"),
    "sample_cfg_scale": ("sample", "guidance_scale"),
    "sample_sampler": ("sample", "sampler"),
    "sample_schedule_type": ("sample", "schedule_type"),
    "sample_seed": ("sample", "seed"),
    # network section (LoRA-specific)
    "lora_rank": ("network", "linear"),
    "lora_alpha": ("network", "linear_alpha"),
    "lora_dtype": ("network", "lora_dtype"),
    # network.controlnet section
    "controlnet_type": ("network.controlnet", "type"),
    "controlnet_pretrained_path": ("network.controlnet", "pretrained_path"),
    "controlnet_init_from_unet": ("network.controlnet", "init_from_unet"),
    "lllite_conditioning_channels": ("network.controlnet",),
    "lllite_rank": ("network.controlnet",),
    "condition_preprocessors": ("network.controlnet",),
    "condition_cache_mode": ("network.controlnet",),
    "conditioning_mode": ("network.controlnet",),
    "outpaint_crop_min_area": ("network.controlnet",),
    "outpaint_crop_max_area": ("network.controlnet",),
    "outpaint_edge_anchor_prob": ("network.controlnet",),
    "outpaint_corner_anchor_prob": ("network.controlnet",),
    "outpaint_mask_channel": ("network.controlnet",),
    "outpaint_known_loss_weight": ("network.controlnet",),
    "outpaint_seam_loss_boost": ("network.controlnet",),
    "outpaint_seam_ring_width": ("network.controlnet",),
    "outpaint_seam_grad_lambda": ("network.controlnet",),
    "outpaint_loss_normalize": ("network.controlnet",),
    "outpaint_edge_feather_min_px": ("network.controlnet",),
    "outpaint_edge_feather_max_px": ("network.controlnet",),
    # train section with renamed key
    "total_steps": ("train", "steps"),
    "learning_rate": ("train", "lr"),
    # process_config.model section
    "base_model_path": ("model", "name_or_path"),
}

# Method-specific fields: only meaningful for that method.
_LORA_ONLY_FIELDS = {"lora_rank", "lora_alpha", "lora_dtype"}
_CONTROLNET_ONLY_FIELDS = {
    "controlnet_type", "controlnet_pretrained_path", "controlnet_init_from_unet",
    "lllite_conditioning_channels", "lllite_rank",
    "condition_preprocessors", "condition_cache_mode",
    "conditioning_mode", "outpaint_crop_min_area", "outpaint_crop_max_area",
    "outpaint_edge_anchor_prob", "outpaint_corner_anchor_prob", "outpaint_mask_channel",
    "outpaint_known_loss_weight", "outpaint_seam_loss_boost", "outpaint_seam_ring_width",
    "outpaint_seam_grad_lambda", "outpaint_loss_normalize",
    "outpaint_edge_feather_min_px", "outpaint_edge_feather_max_px",
}

# Fields excluded from auto-extraction (they need special handling outside the schema loop)
_AUTO_EXTRACT_EXCLUDE = {
    "dataset_id", "dataset_configs", "run_name", "training_method",
    "cache_latents_to_disk",  # Read from datasets[0], not train section
    # Whole nested section under process.vae, set explicitly by
    # get_training_run_params. Without this the generic fallback would look for
    # train["vae_config"], find nothing, and DROP the entire VAE config on the
    # /params -> edit -> PUT round-trip (silently rewriting the run).
    "vae_config",
    # Not a request field; injected into the params dict at the write boundary
    # so generators can tell "sent" from "Pydantic default". Listed defensively
    # so it can never round-trip back in through the extractor.
    "_explicit_fields",
}


def _extract_request_params_from_yaml(process_config: dict, job: str) -> Dict[str, Any]:
    """Extract TrainingRunCreateRequest-shaped params from a parsed YAML process_config.

    Uses TrainingRunCreateRequest.model_fields as the schema (single source of truth).
    For each field, looks up the YAML location via _YAML_FIELD_LOCATIONS, falling back
    to process_config.train with the same key name.

    LoRA-only fields are set to None when job != "lora".
    ControlNet-only fields are set to their defaults when job != "controlnet".
    """
    train = process_config.get("train", {})
    dtype_section = process_config.get("dtype", {}) if isinstance(process_config.get("dtype"), dict) else {}
    save_section = process_config.get("save", {})
    sample_section = process_config.get("sample", {})
    network = process_config.get("network", {})
    cn_network = network.get("controlnet", {}) if isinstance(network, dict) else {}
    model_section = process_config.get("model", {})

    sections = {
        "train": train,
        "dtype": dtype_section,
        "save": save_section,
        "sample": sample_section,
        "network": network,
        "network.controlnet": cn_network,
        "model": model_section,
    }

    result: Dict[str, Any] = {}
    for field_name, field_info in TrainingRunCreateRequest.model_fields.items():
        if field_name in _AUTO_EXTRACT_EXCLUDE:
            continue

        # LoRA-specific fields are None for non-LoRA jobs
        if field_name in _LORA_ONLY_FIELDS and job != "lora":
            result[field_name] = None
            continue

        default = field_info.default
        # Pydantic uses PydanticUndefined for required fields
        if default.__class__.__name__ == "PydanticUndefinedType":
            default = None

        if field_name in _YAML_FIELD_LOCATIONS:
            spec = _YAML_FIELD_LOCATIONS[field_name]
            section_path = spec[0]
            yaml_key = spec[1] if len(spec) > 1 else field_name
            section_dict = sections.get(section_path, {})
            result[field_name] = section_dict.get(yaml_key, default)
        else:
            # Default: look up in train section with same name
            result[field_name] = train.get(field_name, default)

    return result


@router.get("/training/runs/{run_id}/params")
async def get_training_run_params(
    run_id: int,
    db: Session = Depends(get_training_db),
    datasets_db: Session = Depends(get_datasets_db)
):
    """Get training run parameters in TrainingRunCreateRequest format for editing"""
    import time
    start_time = time.time()
    print(f"[get_training_run_params] Starting for run_id={run_id}")

    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    print(f"[get_training_run_params] DB query took {time.time() - start_time:.3f}s")

    # Parse YAML config to extract parameters
    import yaml
    yaml_start = time.time()
    try:
        config = yaml.safe_load(run.config_yaml)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse config YAML: {str(e)}")
    print(f"[get_training_run_params] YAML parsing took {time.time() - yaml_start:.3f}s")

    # Extract job config (first job in config)
    job = config.get("config", {}).get("job", config.get("job", "lora"))
    process_config = config.get("config", {}).get("process", [{}])[0] if config.get("config", {}).get("process") else config.get("process", [{}])[0] if config.get("process") else {}

    # Detect training method from network.type (more reliable than job)
    network_config = process_config.get("network", {})
    network_type = network_config.get("type") if isinstance(network_config, dict) else None
    if network_type in ("lora", "controlnet", "relora", "vae_decoder"):
        # `job` in the YAML is the RUN NAME, not the method, so it can never be
        # trusted for methods that are not detected here. relora and vae_decoder
        # used to fall through and be reported as full_finetune, which meant
        # opening the run in the edit form and saving rewrote it as a full
        # fine-tune.
        job = network_type
    elif not network_config:  # No network section means full fine-tune
        job = "full_finetune"
    # Otherwise keep job from config.job
    datasets_config = process_config.get("datasets", [])

    # Build dataset_configs from YAML using the shared dataset_id-first
    # resolver (same semantics as train_runner.py and the write-boundary
    # column sync) so the edit-form, the dataset_configs column, and the
    # trainer never disagree about which dataset a YAML entry refers to.
    dataset_start = time.time()
    from core.training.dataset_params import resolve_dataset_configs_from_yaml
    dataset_configs = resolve_dataset_configs_from_yaml(run.config_yaml, datasets_db) or []
    cache_latents_to_disk = False  # Default
    for ds_config in datasets_config:
        # Extract cache_latents_to_disk from first dataset
        if ds_config.get("cache_latents_to_disk") is not None:
            cache_latents_to_disk = ds_config.get("cache_latents_to_disk", False)
    print(f"[get_training_run_params] Dataset lookup took {time.time() - dataset_start:.3f}s, found {len(dataset_configs)} datasets")

    # Extract training parameters using schema-driven helper
    # (uses TrainingRunCreateRequest.model_fields as single source of truth)
    params = _extract_request_params_from_yaml(process_config, job)

    # Add fields that aren't part of TrainingRunCreateRequest schema
    params["run_id"] = run.id  # Edit mode marker
    params["run_name"] = run.run_name
    params["training_method"] = (
        job if job in ("lora", "controlnet", "relora", "vae_decoder")
        else "full_finetune"
    )
    params["dataset_configs"] = dataset_configs if dataset_configs else None
    params["cache_latents_to_disk"] = cache_latents_to_disk
    # process.vae is a whole nested section, not a train key -- carry it through
    # verbatim so /params -> edit form -> PUT regenerates an identical config.
    vae_section = process_config.get("vae")
    params["vae_config"] = dict(vae_section) if isinstance(vae_section, dict) else None
    if params["vae_config"]:
        # `ema_decay` is the one key that exists BOTH as a flat request field
        # (diffusion full-FT EMA) and inside process.vae. The flat one is not
        # written to a VAE run's YAML, so the schema-driven extractor above fills
        # it with the diffusion default -- which would make /params report two
        # different EMA decays for the same run. vae_config is authoritative;
        # mirror it so the two agree. (Regeneration already prefers vae_config,
        # so this is a reporting fix, not a behaviour change.)
        if "ema_decay" in params["vae_config"] and "ema_decay" in params:
            params["ema_decay"] = params["vae_config"]["ema_decay"]

    # Fallback for base_model_path if YAML doesn't have it
    if not params.get("base_model_path"):
        params["base_model_path"] = run.base_model_path

    # learning_rate must be float (YAML may store as int or string)
    if params.get("learning_rate") is not None:
        params["learning_rate"] = float(params["learning_rate"])

    print(f"[get_training_run_params] Total time: {time.time() - start_time:.3f}s")
    return params

@router.put("/training/runs/{run_id}")
async def update_training_run(
    run_id: int,
    request: TrainingRunCreateRequest,
    db: Session = Depends(get_training_db),
    datasets_db: Session = Depends(get_datasets_db)
):
    """Update training run configuration by regenerating YAML from parameters"""
    print(f"[Training] Updating training run {run_id}")

    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run.status in ["running", "starting"]:
        raise HTTPException(status_code=400, detail="Cannot update config while training is running")

    try:
        # Get dataset configs
        # NOTE: caption_processing is NOT saved to YAML - read from database at training time
        dataset_configs = []
        dataset_configs_for_yaml = []
        if request.dataset_configs:
            for config in request.dataset_configs:
                from core.training.dataset_params import extract_dataset_params
                config_dict = config.model_dump()
                # Store dict format for total_steps calculation
                dataset_configs.append({
                    "dataset_id": config.dataset_id,
                    "filters": {},
                    **extract_dataset_params(config_dict),
                })
                # Build YAML format (with dataset_id for YAML editing support)
                dataset = datasets_db.query(Dataset).filter(Dataset.id == config.dataset_id).first()
                if dataset:
                    yaml_config = {
                        "dataset_id": config.dataset_id,  # Include dataset_id for YAML editing support
                        "path": dataset.path,
                        **extract_dataset_params(config_dict),
                    }
                    dataset_configs_for_yaml.append(yaml_config)

        # Get primary dataset
        primary_dataset_id = request.dataset_configs[0].dataset_id if request.dataset_configs else None
        primary_dataset = datasets_db.query(Dataset).filter(Dataset.id == primary_dataset_id).first() if primary_dataset_id else None

        # Resolve temp_img:// references in sample_prompts condition_image_path
        resolved_sample_prompts = []
        for sp in (request.sample_prompts or []):
            prompt_dict = dict(sp) if not isinstance(sp, dict) else sp.copy()
            cip = prompt_dict.get("condition_image_path", "")
            if cip and cip.startswith("temp_img://"):
                image_id = cip[len("temp_img://"):]
                resolved_path = os.path.join(TEMP_DIR, image_id)
                if os.path.exists(resolved_path):
                    prompt_dict["condition_image_path"] = resolved_path
                else:
                    print(f"[Training] WARNING: temp image not found: {resolved_path}")
                    prompt_dict["condition_image_path"] = ""
            resolved_sample_prompts.append(prompt_dict)

        # Generate YAML config (same as create)
        config_generator = TrainingConfigGenerator()

        params_dict = request.model_dump()
        # See the create route: model_dump() cannot distinguish "sent" from
        # "Pydantic default", and generate_vae_config needs that distinction.
        params_dict["_explicit_fields"] = sorted(request.model_fields_set)
        common_kwargs = {
            "run_name": run.run_name,
            "base_model_path": request.base_model_path,
            "output_dir": run.output_dir,
            "dataset_path": primary_dataset.path if primary_dataset else "",
            "dataset_configs": dataset_configs_for_yaml,
            "sample_prompts": resolved_sample_prompts,
            "caption_processing": primary_dataset.caption_processing if primary_dataset else None,
        }

        if request.training_method == "lora":
            config_yaml = config_generator.generate_lora_config(params_dict, **common_kwargs)
        elif request.training_method == "relora":
            config_yaml = config_generator.generate_relora_config(params_dict, **common_kwargs)
        elif request.training_method == "controlnet":
            config_yaml = config_generator.generate_controlnet_config(params_dict, **common_kwargs)
        elif request.training_method == "vae_decoder":
            config_yaml = config_generator.generate_vae_config(params_dict, **common_kwargs)
        else:  # full_finetune
            config_yaml = config_generator.generate_full_finetune_config(params_dict, **common_kwargs)

        # Update config_yaml and base_model_path in database
        run.config_yaml = config_yaml
        run.base_model_path = request.base_model_path

        # Keep the (denormalized) dataset_configs column in sync with the
        # YAML we just wrote -- config_yaml is the source of truth that the
        # trainer actually reads; the column is a derived cache used by the
        # pre-flight rescan and list views. Never clobber with an empty
        # derivation (resolve_dataset_configs_from_yaml returns None in
        # that case, so we fall back to whatever the column already holds).
        from core.training.dataset_params import resolve_dataset_configs_from_yaml
        run.dataset_configs = resolve_dataset_configs_from_yaml(config_yaml, datasets_db) or run.dataset_configs

        # Calculate total_steps for database (required by NOT NULL constraint)
        if request.total_steps:
            run.total_steps = request.total_steps
            run.epochs = None
        elif request.epochs:
            # Calculate total_steps from epochs (same logic as create)
            total_dataset_size = 0
            for config in dataset_configs:
                query = datasets_db.query(DatasetItem).filter(DatasetItem.dataset_id == config["dataset_id"])
                dataset_size = query.count()
                total_dataset_size += dataset_size

            if total_dataset_size == 0:
                raise HTTPException(status_code=400, detail="No items in configured datasets")
            run.total_steps = (total_dataset_size // request.batch_size) * request.epochs
            run.epochs = request.epochs

        # Save config file
        config_path = os.path.join(run.output_dir, f"{run.run_name}_config.yaml")
        config_generator.save_config(config_yaml, config_path)

        db.commit()

        print(f"[Training] Updated run {run_id}: {run.run_name}")
        return run.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"[Training] Error updating run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/training/runs/{run_id}")
async def delete_training_run(run_id: int, db: Session = Depends(get_training_db)):
    """Delete a training run"""
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Training run not found")

        # Don't delete if running or starting
        if run.status in ["running", "starting"]:
            raise HTTPException(status_code=400, detail=f"Cannot delete {run.status} training run. Please stop it first.")

        db.delete(run)
        db.commit()
        return {"message": "Training run deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/runs/{run_id}/start")
async def start_training_run(run_id: int, db: Session = Depends(get_training_db)):
    """Start a training run"""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run.status == "running":
        raise HTTPException(status_code=400, detail="Training run is already running")

    try:
        print(f"[API] Starting training run {run_id}")

        # Get config path
        config_path = os.path.join(run.output_dir, f"{run.run_name}_config.yaml")
        print(f"[API] Config path: {config_path}")

        if not os.path.exists(config_path):
            raise HTTPException(status_code=500, detail="Config file not found")

        # Update status to "starting" immediately
        print(f"[API] Updating status to 'starting'")
        run.status = "starting"
        run.error_message = None  # Clear stale error from a prior crash on (re)start/resume

        # Set started_at on first start, last_resumed_at and resumed_from_step on resume
        current_time = datetime.utcnow()
        if run.started_at is None:
            run.started_at = current_time
            run.resumed_from_step = None  # Not a resume
            print(f"[API] First start: started_at set")
        else:
            run.last_resumed_at = current_time
            run.resumed_from_step = run.current_step  # Record step at resume
            print(f"[API] Resuming: last_resumed_at set, resumed_from_step={run.current_step}")

        db.commit()
        print(f"[API] Status updated and committed")

        # ----- Pre-flight: dataset drift detection / optional rescan -----
        # Opt-in via config.process[0].train.rescan_before_training in the run's
        # config_yaml (written by TrainingConfigGenerator when the run was
        # created / updated).  TrainingRun has no JSON `config` column -- only
        # config_yaml -- so the YAML is the single per-run config store here.
        # Runs created before this key existed simply resolve to "off".
        # Walks each dataset's root and compares
        # against datasets.db; if drift is found, runs a full rescan
        # via the existing /datasets/{id}/scan handler so the trainer
        # subprocess sees the freshest state.  Also clears orphan
        # latent-cache files so we don't waste training time on stale
        # cache entries that point at missing images.
        _rescan_raw = None
        try:
            import yaml as _yaml
            # safe_load(None) raises; a run with no YAML simply has no key.
            _cfg_doc = _yaml.safe_load(run.config_yaml or "") or {}
            # Accept both the generated shape (config.process[]) and the flat
            # top-level `process[]` shape a hand-supplied YAML can carry --
            # same tolerance as GET /training/runs/{id}/params.
            _root = _cfg_doc.get("config") or _cfg_doc
            _proc = (_root.get("process") or [{}])[0] or {}
            _rescan_raw = (_proc.get("train") or {}).get("rescan_before_training")
        except Exception as _cfg_err:
            print(f"[API] WARNING: could not read rescan_before_training from config_yaml "
                  f"for run {run_id}: {_cfg_err}")
        from core.training.dataset_drift import normalize_rescan_mode
        _rescan_mode = normalize_rescan_mode(_rescan_raw)
        if _rescan_mode != "off":
            try:
                from core.training.dataset_drift import (
                    detect_drift, rescan_dataset_inline,
                    cleanup_orphan_latent_cache,
                )
                from database import DatasetsSessionLocal
                from database.models import Dataset as _Dataset

                ddb = DatasetsSessionLocal()

                # Self-heal: config_yaml is the source of truth for which
                # datasets this run trains on (it's what the trainer
                # subprocess actually reads -- see train_runner.py). The
                # dataset_configs column is a denormalized cache that can
                # drift from it (e.g. old runs from before write-boundary
                # syncing was added, or a since-fixed bug where the PUT
                # endpoint never persisted the column). Re-derive from YAML
                # here so the pre-flight rescan below always targets the
                # same datasets the trainer will actually load.
                try:
                    from core.training.dataset_params import resolve_dataset_configs_from_yaml
                    _derived_cfgs = resolve_dataset_configs_from_yaml(run.config_yaml, ddb)
                    if _derived_cfgs is not None:
                        _current_ids = {int(c["dataset_id"]) for c in (run.dataset_configs or []) if c.get("dataset_id")}
                        _derived_ids = {c["dataset_id"] for c in _derived_cfgs}
                        if _derived_ids != _current_ids:
                            print(f"[API] Self-heal: run {run_id} dataset_configs column ({sorted(_current_ids)}) "
                                  f"diverged from config_yaml ({sorted(_derived_ids)}); syncing column from YAML")
                            run.dataset_configs = _derived_cfgs
                            db.commit()
                except Exception as _heal_err:
                    print(f"[API] WARNING: dataset_configs self-heal failed for run {run_id}: {_heal_err}")

                ds_cfgs = run.dataset_configs or []
                ds_ids = [int(c["dataset_id"]) for c in ds_cfgs if c.get("dataset_id")]
                if not ds_ids and run.dataset_id:
                    ds_ids = [int(run.dataset_id)]

                try:
                    from core.training.rescan_control import rescan_skip_controller, RescanSkipped
                    import asyncio as _asyncio
                    _ev_loop = _asyncio.get_event_loop()
                    for ds_id in ds_ids:
                        # Resolve dataset name once for progress display.
                        try:
                            _ds_row = ddb.query(_Dataset).filter(_Dataset.id == ds_id).first()
                            _ds_name = (_ds_row.name if _ds_row else "") or ""
                        except Exception:
                            _ds_name = ""

                        # Register this dataset as the current rescan target so a
                        # frontend "skip" can flag it; poll the flag via _skip_cb.
                        rescan_skip_controller.begin("training", run_id, ds_id)
                        # Mark the start of the skippable window for this dataset
                        # (begin→end). The UI shows the Skip button only between
                        # scan_start and scan_end, so it can't be pressed when no
                        # rescan is active.
                        try:
                            manager.send_dataset_scan_progress(
                                scope="training", run_id=run_id,
                                dataset_id=int(ds_id), phase="scan_start",
                                dataset_name=_ds_name,
                            )
                        except Exception:
                            pass
                        def _skip_cb(_rid=run_id):
                            return rescan_skip_controller.should_skip("training", _rid)
                        try:
                            # "force" mode skips drift detection entirely and
                            # always triggers a rescan.  "path" and "smart"
                            # walk first; "smart" additionally collects sidecar
                            # mtimes for content-only caption drift.
                            should_rescan: bool
                            report = None
                            if _rescan_mode == "force":
                                should_rescan = True
                            else:
                                # Live walk-progress → WebSocket
                                def _drift_progress(files_walked: int, _ds_id=ds_id, _nm=_ds_name):
                                    try:
                                        manager.send_dataset_scan_progress(
                                            scope="training", run_id=run_id,
                                            dataset_id=int(_ds_id), phase="drift_walk",
                                            files_walked=files_walked, dataset_name=_nm,
                                        )
                                    except Exception:
                                        pass
                                # Run the (blocking) drift walk off the event loop so
                                # the skip endpoint can be received and progress flushes.
                                report = await _ev_loop.run_in_executor(
                                    None,
                                    lambda: detect_drift(
                                        ds_id, ddb,
                                        check_caption_mtime=(_rescan_mode == "smart"),
                                        progress_callback=_drift_progress,
                                        should_cancel=_skip_cb,
                                    ),
                                )
                                print(f"[Training {run_id}] Dataset drift {ds_id} ({_rescan_mode}): {report.to_dict()}")
                                try:
                                    manager.send_dataset_scan_progress(
                                        scope="training", run_id=run_id,
                                        dataset_id=int(ds_id), phase="drift_done",
                                        files_walked=report.files_walked,
                                        items_in_db=report.items_in_db,
                                        items_missing=report.items_missing,
                                        items_new=report.items_new,
                                        dataset_name=_ds_name,
                                    )
                                except Exception:
                                    pass
                                should_rescan = report.has_drift

                            if should_rescan:
                                reason = (
                                    "force mode"
                                    if _rescan_mode == "force"
                                    else (
                                        f"{report.items_missing} missing, {report.items_new} new"
                                        + (f", {report.captions_stale} stale captions" if (report and report.captions_stale) else "")
                                    )
                                )
                                print(f"[Training {run_id}] Rescan triggered for dataset {ds_id} ({reason})")
                                try:
                                    manager.send_dataset_scan_progress(
                                        scope="training", run_id=run_id,
                                        dataset_id=int(ds_id), phase="rescan",
                                        files_walked=(report.files_walked if report else 0),
                                        items_missing=(report.items_missing if report else 0),
                                        items_new=(report.items_new if report else 0),
                                        message=f"Rescanning... ({reason})",
                                        dataset_name=_ds_name,
                                    )
                                except Exception:
                                    pass
                                try:
                                    _res = await rescan_dataset_inline(ds_id, ddb, should_cancel=_skip_cb)
                                    if isinstance(_res, dict) and _res.get("cancelled"):
                                        print(f"[Training {run_id}] Rescan of {ds_id} skipped by user (partial commit kept)")
                                        try:
                                            manager.send_dataset_scan_progress(
                                                scope="training", run_id=run_id,
                                                dataset_id=int(ds_id), phase="skipped",
                                                dataset_name=_ds_name,
                                                message=f"Skipped rescan of {_ds_name or ds_id}",
                                            )
                                        except Exception:
                                            pass
                                except Exception as _re:
                                    print(f"[Training {run_id}] Rescan failed: {_re}")
                                # Cleanup orphan latent cache for this dataset
                                try:
                                    manager.send_dataset_scan_progress(
                                        scope="training", run_id=run_id,
                                        dataset_id=int(ds_id), phase="cleanup",
                                        message="Cleaning orphan latent cache...",
                                        dataset_name=_ds_name,
                                    )
                                except Exception:
                                    pass
                                try:
                                    _ds = ddb.query(_Dataset).filter(_Dataset.id == ds_id).first()
                                    if _ds is not None and getattr(_ds, "unique_id", None):
                                        removed = cleanup_orphan_latent_cache(
                                            dataset_unique_id=_ds.unique_id,
                                            datasets_db=ddb,
                                            dataset_id=ds_id,
                                        )
                                        if removed:
                                            print(f"[Training {run_id}] Cleaned {removed} orphan latent cache files")
                                except Exception as _ce:
                                    print(f"[Training {run_id}] Latent cache cleanup failed: {_ce}")
                        except RescanSkipped:
                            # Raised from the drift walk (executor path) when skipped.
                            print(f"[Training {run_id}] Drift/rescan of {ds_id} skipped by user")
                            try:
                                manager.send_dataset_scan_progress(
                                    scope="training", run_id=run_id,
                                    dataset_id=int(ds_id), phase="skipped",
                                    dataset_name=_ds_name,
                                    message=f"Skipped rescan of {_ds_name or ds_id}",
                                )
                            except Exception:
                                pass
                        finally:
                            rescan_skip_controller.end("training", run_id)
                            # End of the skippable window → UI hides the button.
                            try:
                                manager.send_dataset_scan_progress(
                                    scope="training", run_id=run_id,
                                    dataset_id=int(ds_id), phase="scan_end",
                                    dataset_name=_ds_name,
                                )
                            except Exception:
                                pass
                finally:
                    ddb.close()
            except Exception as _de:
                print(f"[Training {run_id}] Drift check failed (proceeding anyway): {_de}")

        # Re-check status after the (potentially long) drift/rescan pre-flight
        # above: a concurrent /training/runs/{id}/stop request could have set
        # status="stopped" while we were awaiting the drift walk/rescan. Without
        # this guard we would spawn a subprocess for a run the user already
        # cancelled. Uses a fresh query (not the `run` object captured earlier)
        # since another request/session may have committed the change.
        _status_recheck = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if _status_recheck and _status_recheck.status == "stopped":
            print(f"[API] Run {run_id} was stopped during pre-flight; not spawning training process")
            return {"message": "Training run was stopped before it started", "run": _status_recheck.to_dict()}

        # Create training process
        print(f"[API] Creating training process")
        process = training_process_manager.create_process(
            run_id=run.id,
            config_path=config_path,
            output_dir=run.output_dir
        )
        print(f"[API] Training process created")

        # Define progress callback to update database (runs in separate thread)
        def progress_callback_sync(step: int, loss: float, lr: float):
            # Create a new database session for background task
            from database import TrainingSessionLocal
            db_session = TrainingSessionLocal()
            try:
                # Query fresh run object
                current_run = db_session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
                if not current_run:
                    print(f"[Training {run_id}] Run not found in database")
                    return

                # Negative step indicates failure or user stop
                if step == -2:
                    # User requested stop
                    print(f"[Training {run_id}] Process stopped by user, updating status")
                    current_run.status = "stopped"
                    db_session.commit()
                    return
                elif step == -1:
                    # Process failed with error -- unless the child already
                    # recorded itself as "stopped" (train_runner.py's new
                    # except KeyboardInterrupt handler, or the -2 branch above
                    # racing with this one). A stop during initialization
                    # exits non-zero without is_user_stopped necessarily being
                    # observed by the monitor in time (e.g. the flag-triggered
                    # KeyboardInterrupt exits before/around the same time as
                    # process.stop() sets is_user_stopped), which would
                    # otherwise mislabel a user-requested stop as a failure.
                    if current_run.status == "stopped":
                        print(f"[Training {run_id}] Process exited non-zero but run is already 'stopped' (user stop); not overwriting with 'failed'")
                        return
                    print(f"[Training {run_id}] Process failed, updating status")
                    current_run.status = "failed"
                    current_run.error_message = "Training process exited with error"
                    db_session.commit()
                    return

                # Update status to "running" on first progress update
                if current_run.status == "starting":
                    current_run.status = "running"
                    print(f"[Training {run_id}] Status updated: starting -> running")

                current_run.current_step = step
                current_run.loss = loss
                current_run.learning_rate = lr
                current_run.progress = (step / current_run.total_steps) * 100
                db_session.commit()
            except Exception as e:
                print(f"[Training {run_id}] Error updating progress: {e}")
                import traceback
                traceback.print_exc()
            finally:
                db_session.close()

        # Wrap callback to run in thread pool (non-blocking)
        def progress_callback(step: int, loss: float, lr: float):
            executor.submit(progress_callback_sync, step, loss, lr)

        # Define log callback
        def log_callback(log_line: str):
            print(f"[Training {run_id}] {log_line}")

        # Start training process (non-blocking)
        print(f"[API] Starting training process...")
        await process.start(progress_callback=progress_callback, log_callback=log_callback)
        print(f"[API] Training process started")

        print(f"[API] Returning response")
        return {"message": "Training started", "run": run.to_dict()}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.post("/training/runs/{run_id}/stop")
async def stop_training_run(run_id: int, db: Session = Depends(get_training_db)):
    """Stop a training run"""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    # Allow stopping if status is "running" or "starting" (in case of early failure)
    if run.status not in ["running", "starting"]:
        raise HTTPException(status_code=400, detail=f"Cannot stop training with status '{run.status}'")

    try:
        # Get training process
        process = training_process_manager.get_process(run_id)

        if process:
            print(f"[API] Stopping training process for run {run_id}")
            await process.stop()
            await training_process_manager.remove_process(run_id)
        else:
            # Process doesn't exist (likely crashed during startup)
            print(f"[API] No active process found for run {run_id}, updating status only")

        # Update run status
        run.status = "stopped"
        db.commit()

        return {"message": "Training stopped", "run": run.to_dict()}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to stop training: {str(e)}")


class SkipRescanRequest(BaseModel):
    dataset_id: Optional[int] = None  # skip only if it matches the current rescan


@router.post("/training/runs/{run_id}/skip-rescan")
async def skip_training_rescan(run_id: int, request: SkipRescanRequest = SkipRescanRequest()):
    """Skip the dataset currently being rescanned in this run's pre-flight.

    Flags the cooperative-cancel flag the rescan's directory walkers poll, so the
    current dataset's drift-walk / rescan aborts (keeping any already-applied
    changes) and the pre-flight continues with the remaining datasets.
    """
    from core.training.rescan_control import rescan_skip_controller
    flagged = rescan_skip_controller.request_skip("training", run_id, request.dataset_id)
    return {
        "skipped": flagged,
        "current_dataset": rescan_skip_controller.current_dataset("training", run_id),
    }

@router.patch("/training/runs/{run_id}/config")
async def update_training_config(
    run_id: int,
    config_data: dict,
    db: Session = Depends(get_training_db),
    datasets_db: Session = Depends(get_datasets_db),
):
    """Update training configuration (only allowed when not running)"""
    print(f"[Training] Updating config for run_id={run_id}")
    print(f"[Training] config_data keys: {config_data.keys()}")
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        print(f"[Training] ERROR: Run ID {run_id} not found in database")
        raise HTTPException(status_code=404, detail="Training run not found")

    if run.status in ["running", "starting"]:
        raise HTTPException(status_code=400, detail="Cannot update config while training is running")

    try:
        config_yaml = config_data.get("config_yaml")
        if not config_yaml:
            raise HTTPException(status_code=400, detail="config_yaml is required")

        # Update config_yaml in database
        run.config_yaml = config_yaml

        # Keep dataset_configs column in sync with the newly-saved YAML
        # (config_yaml is the source of truth; see resolve_dataset_configs_from_yaml).
        from core.training.dataset_params import resolve_dataset_configs_from_yaml
        run.dataset_configs = resolve_dataset_configs_from_yaml(config_yaml, datasets_db) or run.dataset_configs

        # Update the original config file on disk ({run_name}_config.yaml)
        import yaml
        from pathlib import Path

        config_path = Path(run.output_dir) / f"{run.run_name}_config.yaml"
        if config_path.parent.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_yaml)
            print(f"[Training] Updated config file: {config_path}")

        db.commit()

        return {"message": "Configuration updated successfully", "run": run.to_dict()}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

@router.post("/training/runs/{run_id}/config/reload")
async def reload_training_config(
    run_id: int,
    db: Session = Depends(get_training_db),
    datasets_db: Session = Depends(get_datasets_db),
):
    """Reload training configuration from disk (for external YAML edits)"""
    print(f"[Training] Reloading config from disk for run_id={run_id}")
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        print(f"[Training] ERROR: Run ID {run_id} not found in database")
        raise HTTPException(status_code=404, detail="Training run not found")

    if run.status in ["running", "starting"]:
        raise HTTPException(status_code=400, detail="Cannot reload config while training is running")

    try:
        from pathlib import Path

        # Read config from disk
        config_path = Path(run.output_dir) / f"{run.run_name}_config.yaml"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_yaml = f.read()

        # Update database with disk content
        run.config_yaml = config_yaml

        # Keep dataset_configs column in sync with the reloaded YAML
        # (config_yaml is the source of truth; see resolve_dataset_configs_from_yaml).
        from core.training.dataset_params import resolve_dataset_configs_from_yaml
        run.dataset_configs = resolve_dataset_configs_from_yaml(config_yaml, datasets_db) or run.dataset_configs

        db.commit()

        print(f"[Training] Reloaded config from disk: {config_path}")
        return {"message": "Configuration reloaded from disk", "run": run.to_dict()}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {str(e)}")

@router.get("/training/runs/{run_id}/status")
async def get_training_status(run_id: int, db: Session = Depends(get_training_db)):
    """Get current training status"""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    # Checkpoints are now tracked in DB via TrainingCheckpoint model
    # No need to scan filesystem - checkpoints are loaded via to_dict()

    # Get process status if available
    process = training_process_manager.get_process(run_id)
    process_status = process.get_status() if process else None

    return {
        "status": run.status,
        "progress": run.progress,
        "current_step": run.current_step,
        "total_steps": run.total_steps,
        "loss": run.loss,
        "learning_rate": run.learning_rate,
        "phase": run.phase,
        "phase_progress": run.phase_progress,
        "phase_detail": run.phase_detail,
        "process_status": process_status
    }

@router.post("/training/runs/{run_id}/tensorboard/start")
async def start_tensorboard(run_id: int, db: Session = Depends(get_training_db)):
    """Start TensorBoard server for a training run"""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    # Get tensorboard log directory
    from pathlib import Path
    log_dir = Path(run.output_dir) / "tensorboard"

    if not log_dir.exists():
        raise HTTPException(status_code=404, detail="TensorBoard logs not found")

    try:
        port = tensorboard_manager.start(run_id, str(log_dir))
        url = tensorboard_manager.get_url(run_id)
        return {
            "status": "started",
            "port": port,
            "url": url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start TensorBoard: {str(e)}")

@router.delete("/training/runs/{run_id}/tensorboard/stop")
async def stop_tensorboard(run_id: int):
    """Stop TensorBoard server for a training run"""
    try:
        tensorboard_manager.stop(run_id)
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop TensorBoard: {str(e)}")

@router.get("/training/runs/{run_id}/tensorboard/status")
async def get_tensorboard_status(run_id: int):
    """Get TensorBoard server status"""
    is_running = tensorboard_manager.is_running(run_id)
    url = tensorboard_manager.get_url(run_id) if is_running else None
    port = tensorboard_manager.get_port(run_id) if is_running else None

    return {
        "is_running": is_running,
        "url": url,
        "port": port
    }

@router.get("/training/runs/{run_id}/checkpoints")
async def get_training_checkpoints(run_id: int, db: Session = Depends(get_training_db)):
    """Get list of available checkpoints for a training run"""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    # Get checkpoints from DB (already sorted by step descending)
    checkpoints = []
    for ckpt in sorted(run.checkpoints, key=lambda x: x.step, reverse=True):
        from pathlib import Path
        checkpoints.append({
            "step": ckpt.step,
            "epoch": ckpt.epoch,
            "filename": Path(ckpt.file_path).name,
            "path": ckpt.file_path,
            "file_size": ckpt.file_size,
            "created_at": ckpt.created_at.isoformat() if ckpt.created_at else None,
        })

    return {"checkpoints": checkpoints}

@router.get("/training/runs/{run_id}/checkpoints/{checkpoint_filename}")
async def download_checkpoint(run_id: int, checkpoint_filename: str, db: Session = Depends(get_training_db)):
    """Download a specific checkpoint file"""
    from pathlib import Path
    from fastapi.responses import FileResponse
    import os

    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    output_dir = Path(run.output_dir)
    checkpoint_path = output_dir / checkpoint_filename
    # VAE-decoder checkpoints are DIRECTORIES under <output_dir>/checkpoints/
    # (weights + EMA + optimizer + RNG + train_state.json), so the flat
    # <output_dir>/<name> lookup misses them.
    if not checkpoint_path.exists():
        nested = output_dir / "checkpoints" / checkpoint_filename
        if nested.exists():
            checkpoint_path = nested

    # Security check: ensure the file is within the output directory
    try:
        checkpoint_path = checkpoint_path.resolve()
        output_dir = output_dir.resolve()
        if not str(checkpoint_path).startswith(str(output_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        raise HTTPException(status_code=403, detail="Invalid checkpoint path")

    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint file not found")

    if checkpoint_path.is_dir():
        # Directory checkpoint: serve a zip so the frontend needs no special
        # case (it just downloads whatever the route returns).
        import shutil
        import tempfile
        from starlette.background import BackgroundTask

        tmp_dir = tempfile.mkdtemp(prefix="ckpt_zip_")
        try:
            archive = shutil.make_archive(
                os.path.join(tmp_dir, checkpoint_path.name), "zip",
                root_dir=str(checkpoint_path))
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to archive checkpoint directory: {e}")
        return FileResponse(
            path=archive,
            filename=f"{checkpoint_path.name}.zip",
            media_type="application/zip",
            background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
        )

    if not checkpoint_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    return FileResponse(
        path=str(checkpoint_path),
        filename=checkpoint_filename,
        media_type="application/octet-stream"
    )

@router.get("/training/runs/{run_id}/debug-latents")
async def get_debug_latents(run_id: int, db: Session = Depends(get_training_db)):
    """Get list of debug latent saves for a training run"""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    from pathlib import Path
    import glob

    output_dir = Path(run.output_dir)
    debug_dir = output_dir / "debug"

    if not debug_dir.exists():
        return {"debug_latents": []}

    # Find all step directories
    step_dirs = sorted([d for d in debug_dir.iterdir() if d.is_dir() and d.name.startswith("step_")])

    debug_latents = []
    for step_dir in step_dirs:
        # Extract step number from directory name (step_XXXXXX)
        step_str = step_dir.name.replace("step_", "")
        try:
            step = int(step_str)

            # Find all latent .pt files in this step directory
            latent_files = sorted(step_dir.glob("latents_t*.pt"))

            for latent_file in latent_files:
                # Extract timestep from filename (latents_tXXXX.pt or latents_t0.XXXX.pt)
                timestep_str = latent_file.stem.replace("latents_t", "")
                try:
                    # Try float first (Z-Image), then int (SD/SDXL)
                    timestep = float(timestep_str)
                    debug_latents.append({
                        "step": step,
                        "timestep": timestep,
                        "filename": latent_file.name,
                        "path": str(latent_file)
                    })
                except ValueError:
                    continue
        except ValueError:
            continue

    # Sort by step and timestep
    debug_latents.sort(key=lambda x: (x["step"], x["timestep"]))

    return {"debug_latents": debug_latents}

@router.get("/training/runs/{run_id}/debug-latents/{step}/visualize")
async def visualize_debug_latent(
    run_id: int,
    step: int,
    timestep: Optional[int] = None,
    db: Session = Depends(get_training_db)
):
    """
    Visualize debug latents as images (without VAE decoding).
    Returns base64-encoded images for latents, noisy_latents, and predicted_noise.
    """
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    from pathlib import Path
    import torch
    import numpy as np
    from PIL import Image
    import io
    import base64

    output_dir = Path(run.output_dir)
    debug_dir = output_dir / "debug" / f"step_{step:06d}"

    if not debug_dir.exists():
        raise HTTPException(status_code=404, detail=f"Debug directory for step {step} not found")

    # Find the latent file (use timestep if provided, otherwise use first one)
    if timestep is not None:
        # Try both float format (Z-Image) and int format (SD/SDXL)
        latent_file = debug_dir / f"latents_t{timestep}.pt"
        if not latent_file.exists():
            # Try integer format as fallback
            latent_file = debug_dir / f"latents_t{int(timestep):04d}.pt"
            if not latent_file.exists():
                raise HTTPException(status_code=404, detail=f"Latent file for timestep {timestep} not found")
    else:
        latent_files = sorted(debug_dir.glob("latents_t*.pt"))
        if not latent_files:
            raise HTTPException(status_code=404, detail="No latent files found")
        latent_file = latent_files[0]

    # Load the latent data
    try:
        data = torch.load(latent_file, map_location='cpu')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load latent file: {str(e)}")

    def flux2_unpatchify(latent_tensor):
        """Unpatchify FLUX.2 latents: (C=128, H/2, W/2) -> (C=32, H, W)"""
        # latent_tensor shape: [128, H/2, W/2]
        num_channels, height, width = latent_tensor.shape
        # Reshape: [128, H/2, W/2] -> [32, 2, 2, H/2, W/2]
        latent_tensor = latent_tensor.view(num_channels // 4, 2, 2, height, width)
        # Permute: [32, 2, 2, H/2, W/2] -> [32, H/2, 2, W/2, 2]
        latent_tensor = latent_tensor.permute(0, 3, 1, 4, 2)
        # Reshape: [32, H/2, 2, W/2, 2] -> [32, H, W]
        latent_tensor = latent_tensor.reshape(num_channels // 4, height * 2, width * 2)
        return latent_tensor

    def latent_to_image(latent_tensor, is_flux2=False):
        """Convert latent tensor to PIL Image (without VAE decoding)"""
        # latent_tensor shape: [1, C, H, W] or [C, H, W]
        if latent_tensor.dim() == 4:
            latent_tensor = latent_tensor[0]  # Remove batch dimension

        # latent_tensor shape: [C, H, W]
        # Convert to float32 first (NumPy doesn't support bfloat16)
        if latent_tensor.dtype == torch.bfloat16:
            latent_tensor = latent_tensor.to(torch.float32)

        # FLUX.2: unpatchify 128ch -> 32ch before visualization
        if is_flux2 and latent_tensor.shape[0] == 128:
            latent_tensor = flux2_unpatchify(latent_tensor)

        latent_np = latent_tensor.numpy()  # [C, H, W]

        # Take first 3 channels (or repeat if less than 3)
        if latent_np.shape[0] >= 3:
            # Use first 3 channels as R, G, B
            rgb_channels = latent_np[:3]  # [3, H, W]
        elif latent_np.shape[0] == 1:
            # Single channel, repeat 3 times
            rgb_channels = np.repeat(latent_np, 3, axis=0)  # [3, H, W]
        else:
            # Pad with zeros to 3 channels
            rgb_channels = np.zeros((3,) + latent_np.shape[1:])
            rgb_channels[:latent_np.shape[0]] = latent_np

        # Normalize each channel independently to 0-255
        normalized = np.zeros_like(rgb_channels)
        for i in range(3):
            channel = rgb_channels[i]
            ch_min = channel.min()
            ch_max = channel.max()
            if ch_max - ch_min > 1e-6:
                normalized[i] = (channel - ch_min) / (ch_max - ch_min) * 255
            else:
                normalized[i] = np.zeros_like(channel)

        # Convert to [H, W, 3] and uint8
        rgb_image = normalized.transpose(1, 2, 0).astype(np.uint8)  # [H, W, 3]

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image, mode='RGB')

        return pil_image

    def image_to_base64(pil_image):
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    # Detect model type from saved data
    is_flux2 = data.get("model_type") == "flux2"

    # Convert each latent type to image
    result = {
        "step": step,
        "timestep": data.get("timestep", 0),
        "loss": data.get("loss", 0.0),
        "recon_loss": data.get("recon_loss", 0.0),
        "model_type": data.get("model_type", "unknown"),
    }

    # Add caption if available
    if "caption" in data:
        result["caption"] = data["caption"]

    # SDXL micro-conditioning / crop info (for crop-augmentation verification):
    # original_size, crop_top_left (= crop point), target_size, and the raw time_ids.
    for _k in ("original_size", "crop_top_left", "target_size", "sdxl_time_ids", "sdxl_time_ids_all"):
        if _k in data:
            result[_k] = data[_k]

    # Add reference image thumbnail if available
    if "reference_image_path" in data:
        try:
            from PIL import Image as _PILImage
            ref_path = str(data["reference_image_path"]).replace("temp_img://", "")
            ref_img = _PILImage.open(ref_path).convert("RGB")
            ref_img.thumbnail((256, 256))
            result["reference_image"] = image_to_base64(ref_img)
        except Exception:
            pass

    # MiniT2I latent variant saves true VAE-decoded RGB previews (webp) alongside
    # the .pt. Prefer those for the Target/Predicted comparison — far more
    # meaningful than false-color latent channels. Filenames mirror the .pt:
    #   latents_t<ts>.pt -> decode_t<ts>_target.webp / decode_t<ts>_pred_x0.webp
    try:
        _base = latent_file.name[:-3] if latent_file.name.endswith(".pt") else latent_file.stem  # "latents_t<ts>"
        _ts_part = _base.replace("latents_t", "")
        _target_webp = latent_file.parent / f"decode_t{_ts_part}_target.webp"
        _pred_webp = latent_file.parent / f"decode_t{_ts_part}_pred_x0.webp"
        _noisy_webp = latent_file.parent / f"decode_t{_ts_part}_noisy.webp"
        def _webp_preview(p):
            im = Image.open(p).convert("RGB")
            im.thumbnail((768, 768))  # downscale: debug preview doesn't need full res
            return image_to_base64(im)
        if _target_webp.exists():
            result["latents_image"] = _webp_preview(_target_webp)
        if _pred_webp.exists():
            result["predicted_latent_image"] = _webp_preview(_pred_webp)
        if _noisy_webp.exists():  # older runs have no noisy webp -> falls back below
            result["noisy_latents_image"] = _webp_preview(_noisy_webp)
    except Exception as _webp_e:
        print(f"[debug-latents] webp preview load failed: {_webp_e}")

    if "latents" in data and "latents_image" not in result:
        img = latent_to_image(data["latents"], is_flux2=is_flux2)
        result["latents_image"] = image_to_base64(img)

    if "noisy_latents" in data and "noisy_latents_image" not in result:
        img = latent_to_image(data["noisy_latents"], is_flux2=is_flux2)
        result["noisy_latents_image"] = image_to_base64(img)

    # predicted_noise (SD/SDXL) or predicted_velocity (Z-Image/FLUX.2)
    if "predicted_noise" in data:
        img = latent_to_image(data["predicted_noise"], is_flux2=is_flux2)
        result["predicted_noise_image"] = image_to_base64(img)

    if "predicted_velocity" in data:
        img = latent_to_image(data["predicted_velocity"], is_flux2=is_flux2)
        result["predicted_velocity_image"] = image_to_base64(img)

    if "predicted_latent" in data and "predicted_latent_image" not in result:
        img = latent_to_image(data["predicted_latent"], is_flux2=is_flux2)
        result["predicted_latent_image"] = image_to_base64(img)

    return result

@router.get("/training/runs/{run_id}/metrics")
async def get_training_metrics(
    run_id: int,
    since_step: Optional[int] = None,
    max_points: int = 1000,
    db: Session = Depends(get_training_db)
):
    """
    Get training metrics (loss, learning_rate) from TensorBoard event files.

    Args:
        run_id: Training run ID
        since_step: Only return data after this step (for incremental updates)
        max_points: Maximum number of data points to return (for decimation)
    """
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    from pathlib import Path
    import glob

    output_dir = Path(run.output_dir)
    tensorboard_dir = output_dir / "tensorboard"

    if not tensorboard_dir.exists():
        return {"loss": [], "recon_loss": [], "learning_rate": []}

    try:
        from tensorboard.backend.event_processing import event_accumulator

        # Find all event files in all subdirectories (timestamp-based)
        event_files = []
        for subdir in tensorboard_dir.iterdir():
            if subdir.is_dir():
                event_files.extend(glob.glob(str(subdir / "events.out.tfevents.*")))

        if not event_files:
            return {"loss": [], "recon_loss": [], "learning_rate": []}

        # Optimization: If since_step is provided, only read the most recent event file
        # (since older files won't have data after since_step)
        if since_step is not None and len(event_files) > 1:
            # Sort by modification time, use only the most recent
            event_files_sorted = sorted(event_files, key=lambda f: Path(f).stat().st_mtime)
            event_files = [event_files_sorted[-1]]  # Only most recent

        # Use the most recent event file or merge all
        all_loss = []
        all_recon_loss = []
        all_lr = []

        for event_file in event_files:
            # Check cache first (keyed by run_id and event_file path)
            cache_key = (run_id, event_file)
            event_file_path = Path(event_file)
            current_mtime = event_file_path.stat().st_mtime

            # If cached and file hasn't changed, use cached EventAccumulator
            if cache_key in _event_accumulator_cache:
                cached_ea, cached_mtime = _event_accumulator_cache[cache_key]
                if cached_mtime == current_mtime:
                    ea = cached_ea
                else:
                    # File changed, reload
                    ea = event_accumulator.EventAccumulator(event_file)
                    ea.Reload()
                    _event_accumulator_cache[cache_key] = (ea, current_mtime)
            else:
                # Not cached, create new
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                _event_accumulator_cache[cache_key] = (ea, current_mtime)

            # Get scalar tags
            if 'train/loss' in ea.Tags()['scalars']:
                loss_events = ea.Scalars('train/loss')
                all_loss.extend([
                    {"step": int(e.step), "value": float(e.value), "wall_time": float(e.wall_time)}
                    for e in loss_events
                ])

            if 'train/recon_loss' in ea.Tags()['scalars']:
                recon_loss_events = ea.Scalars('train/recon_loss')
                all_recon_loss.extend([
                    {"step": int(e.step), "value": float(e.value), "wall_time": float(e.wall_time)}
                    for e in recon_loss_events
                ])

            if 'train/learning_rate' in ea.Tags()['scalars']:
                lr_events = ea.Scalars('train/learning_rate')
                all_lr.extend([
                    {"step": int(e.step), "value": float(e.value), "wall_time": float(e.wall_time)}
                    for e in lr_events
                ])

        # Sort by step
        all_loss.sort(key=lambda x: x["step"])
        all_recon_loss.sort(key=lambda x: x["step"])
        all_lr.sort(key=lambda x: x["step"])

        # Deduplicate: If the same step appears multiple times (resume scenario),
        # keep only the one with the latest wall_time (most recent training run)
        def deduplicate_by_latest_wall_time(data):
            if not data:
                return []

            step_to_latest = {}
            for point in data:
                step = point["step"]
                if step not in step_to_latest or point["wall_time"] > step_to_latest[step]["wall_time"]:
                    step_to_latest[step] = point

            # Return sorted by step
            return sorted(step_to_latest.values(), key=lambda x: x["step"])

        all_loss = deduplicate_by_latest_wall_time(all_loss)
        all_recon_loss = deduplicate_by_latest_wall_time(all_recon_loss)
        all_lr = deduplicate_by_latest_wall_time(all_lr)

        # Filter by since_step if provided
        if since_step is not None:
            all_loss = [d for d in all_loss if d["step"] > since_step]
            all_recon_loss = [d for d in all_recon_loss if d["step"] > since_step]
            all_lr = [d for d in all_lr if d["step"] > since_step]

        # Decimate data if too many points (simple nth-point sampling).
        # Always preserve the last point so the chart reflects current state.
        def decimate(data, max_points):
            if len(data) <= max_points:
                return data
            step_size = max(1, len(data) // max_points)
            indices = list(range(0, len(data), step_size))
            if indices[-1] != len(data) - 1:
                indices.append(len(data) - 1)
            return [data[i] for i in indices]

        all_loss = decimate(all_loss, max_points)
        all_recon_loss = decimate(all_recon_loss, max_points)
        all_lr = decimate(all_lr, max_points)

        return {
            "loss": all_loss,
            "recon_loss": all_recon_loss,
            "learning_rate": all_lr
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="TensorBoard library not available")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {str(e)}")


@router.get("/training/runs/{run_id}/metrics_db")
async def get_training_metrics_db(
    run_id: int,
    max_points: int = 1000,
    db: Session = Depends(get_training_db)
):
    """
    Get training metrics from database with uniform step sampling.

    This endpoint reads metrics from SQLAlchemy DB and returns uniformly sampled
    data points to ensure consistent chart density across updates.

    Features:
    - Uniform sampling: Returns evenly distributed steps from 0 to max_step
    - Consistent density: Same points returned on every fetch (no jumping)
    - Indexed queries: Fast filtering by (run_id, step)
    - UPSERT-safe: Metrics with same (run_id, step) are overwritten (training restart support)

    Args:
        run_id: Training run ID
        max_points: Maximum number of data points to return (default: 1000)

    Returns:
        {
            "loss": [{"step": int, "value": float, "timestamp": str}, ...],
            "recon_loss": [{"step": int, "value": float, "timestamp": str}, ...],
            "learning_rate": [{"step": int, "value": float, "timestamp": str}, ...],
            "grad_norm": [{"step": int, "value": float, "timestamp": str}, ...],
            "grad_norm_text_encoder": [{"step": int, "value": float, "timestamp": str}, ...],
            "grad_norm_unet": [{"step": int, "value": float, "timestamp": str}, ...]
        }
    """
    from database.models import TrainingMetrics
    from core.training.metric_registry import EXTRA_METRIC_DEFS

    # Check if run exists
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    try:
        # Get min and max steps for this run
        result = db.query(
            func.min(TrainingMetrics.step),
            func.max(TrainingMetrics.step)
        ).filter(TrainingMetrics.run_id == run_id).first()

        min_step, max_step = result
        if min_step is None or max_step is None:
            # No metrics yet
            return {
                "loss": [],
                "recon_loss": [],
                "learning_rate": [],
                "grad_norm": [],
                "grad_norm_text_encoder": [],
                "grad_norm_unet": [],
                "grad_norm_vision_encoder": [],
                "param_update_norm_unet": [],
                "param_update_norm_te1": [],
                "param_update_norm_te2": [],
                "param_update_norm_ve": [],
                "param_cumulative_drift_unet": [],
                "param_cumulative_drift_te1": [],
                "param_cumulative_drift_te2": [],
                "param_cumulative_drift_ve": [],
                "epoch_boundaries": [],
                "resume_markers": [],
            }

        # Calculate uniform sample steps
        total_steps = max_step - min_step + 1
        if total_steps <= max_points:
            # Fetch all steps
            sample_steps = list(range(min_step, max_step + 1))
        else:
            # Uniform sampling: divide range into max_points intervals
            step_interval = total_steps / max_points
            sample_steps = [
                int(min_step + i * step_interval)
                for i in range(max_points)
            ]
            # Always include the last step
            if max_step not in sample_steps:
                sample_steps.append(max_step)

        # Fetch metrics for sampled steps
        query = db.query(TrainingMetrics).filter(
            TrainingMetrics.run_id == run_id,
            TrainingMetrics.step.in_(sample_steps)
        ).order_by(TrainingMetrics.step.asc())

        metrics = query.all()

        # Convert to response format
        loss_data = []
        recon_loss_data = []
        # Bespoke arch/method-specific metrics (REPA, outpaint gen_loss, …) stored
        # generically in TrainingMetrics.extra_metrics -> {name: MetricPoint[]}.
        extra_series: dict[str, list] = {}
        lr_data = []
        grad_norm_data = []
        grad_norm_te_data = []
        grad_norm_te1_data = []
        grad_norm_te2_data = []
        grad_norm_unet_data = []
        grad_norm_ve_data = []
        param_update_norm_unet_data = []
        param_update_norm_te1_data = []
        param_update_norm_te2_data = []
        param_update_norm_ve_data = []
        param_cumulative_drift_unet_data = []
        param_cumulative_drift_te1_data = []
        param_cumulative_drift_te2_data = []
        param_cumulative_drift_ve_data = []

        import math

        def is_valid_float(v):
            """Check if value is a valid JSON-serializable float (not inf/nan)."""
            if v is None:
                return False
            return not (math.isinf(v) or math.isnan(v))

        for m in metrics:
            point = {
                "step": m.step,
                # resume_seq is carried on every point so the chart can later split
                # curves per resume; epoch markers are precomputed below.
                "resume_seq": getattr(m, "resume_seq", 0) or 0,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None
            }

            if is_valid_float(m.loss):
                loss_data.append({**point, "value": m.loss})

            if is_valid_float(m.recon_loss):
                recon_loss_data.append({**point, "value": m.recon_loss})

            _extra = getattr(m, "extra_metrics", None) or {}
            for _k, _v in _extra.items():
                if is_valid_float(_v):
                    extra_series.setdefault(_k, []).append({**point, "value": _v})

            if is_valid_float(m.learning_rate):
                lr_data.append({**point, "value": m.learning_rate})

            if is_valid_float(m.grad_norm):
                grad_norm_data.append({**point, "value": m.grad_norm})

            if is_valid_float(m.grad_norm_text_encoder):
                grad_norm_te_data.append({**point, "value": m.grad_norm_text_encoder})

            if is_valid_float(getattr(m, 'grad_norm_text_encoder_1', None)):
                grad_norm_te1_data.append({**point, "value": m.grad_norm_text_encoder_1})

            if is_valid_float(getattr(m, 'grad_norm_text_encoder_2', None)):
                grad_norm_te2_data.append({**point, "value": m.grad_norm_text_encoder_2})

            if is_valid_float(m.grad_norm_unet):
                grad_norm_unet_data.append({**point, "value": m.grad_norm_unet})

            if is_valid_float(getattr(m, 'grad_norm_vision_encoder', None)):
                grad_norm_ve_data.append({**point, "value": m.grad_norm_vision_encoder})

            if is_valid_float(getattr(m, 'param_update_norm_unet', None)):
                param_update_norm_unet_data.append({**point, "value": m.param_update_norm_unet})
            if is_valid_float(getattr(m, 'param_update_norm_te1', None)):
                param_update_norm_te1_data.append({**point, "value": m.param_update_norm_te1})
            if is_valid_float(getattr(m, 'param_update_norm_te2', None)):
                param_update_norm_te2_data.append({**point, "value": m.param_update_norm_te2})
            if is_valid_float(getattr(m, 'param_update_norm_ve', None)):
                param_update_norm_ve_data.append({**point, "value": m.param_update_norm_ve})
            if is_valid_float(getattr(m, 'param_cumulative_drift_unet', None)):
                param_cumulative_drift_unet_data.append({**point, "value": m.param_cumulative_drift_unet})
            if is_valid_float(getattr(m, 'param_cumulative_drift_te1', None)):
                param_cumulative_drift_te1_data.append({**point, "value": m.param_cumulative_drift_te1})
            if is_valid_float(getattr(m, 'param_cumulative_drift_te2', None)):
                param_cumulative_drift_te2_data.append({**point, "value": m.param_cumulative_drift_te2})
            if is_valid_float(getattr(m, 'param_cumulative_drift_ve', None)):
                param_cumulative_drift_ve_data.append({**point, "value": m.param_cumulative_drift_ve})

        # Epoch boundaries (last recorded step of each epoch) and resume markers
        # (first step of each resume_seq > 0), computed from ALL rows (not just the
        # sampled subset) so the markers are accurate. The UI draws dotted vertical
        # lines for epochs and a distinct marker for resume boundaries.
        epoch_rows = db.query(
            TrainingMetrics.epoch, func.max(TrainingMetrics.step)
        ).filter(
            TrainingMetrics.run_id == run_id,
            TrainingMetrics.epoch.isnot(None),
        ).group_by(TrainingMetrics.epoch).order_by(TrainingMetrics.epoch.asc()).all()
        epoch_boundaries = [
            {"epoch": int(e), "step": int(s)}
            for e, s in epoch_rows if e is not None and s is not None
        ]

        resume_rows = db.query(
            TrainingMetrics.resume_seq, func.min(TrainingMetrics.step)
        ).filter(
            TrainingMetrics.run_id == run_id,
        ).group_by(TrainingMetrics.resume_seq).order_by(TrainingMetrics.resume_seq.asc()).all()
        resume_markers = [
            {"resume_seq": int(rs), "step": int(s)}
            for rs, s in resume_rows if rs and int(rs) > 0 and s is not None
        ]

        return {
            "loss": loss_data,
            "recon_loss": recon_loss_data,
            "extra_metrics": extra_series,
            "extra_metric_defs": {
                k: EXTRA_METRIC_DEFS.get(k, {"label": k, "dashed": True})
                for k in extra_series
            },
            "learning_rate": lr_data,
            "grad_norm": grad_norm_data,
            "grad_norm_text_encoder": grad_norm_te_data,
            "grad_norm_text_encoder_1": grad_norm_te1_data,
            "grad_norm_text_encoder_2": grad_norm_te2_data,
            "grad_norm_unet": grad_norm_unet_data,
            "grad_norm_vision_encoder": grad_norm_ve_data,
            "param_update_norm_unet": param_update_norm_unet_data,
            "param_update_norm_te1": param_update_norm_te1_data,
            "param_update_norm_te2": param_update_norm_te2_data,
            "param_update_norm_ve": param_update_norm_ve_data,
            "param_cumulative_drift_unet": param_cumulative_drift_unet_data,
            "param_cumulative_drift_te1": param_cumulative_drift_te1_data,
            "param_cumulative_drift_te2": param_cumulative_drift_te2_data,
            "param_cumulative_drift_ve": param_cumulative_drift_ve_data,
            "epoch_boundaries": epoch_boundaries,
            "resume_markers": resume_markers,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to read metrics from DB: {str(e)}")


@router.get("/training/runs/{run_id}/samples")
async def get_training_samples(
    run_id: int,
    db: Session = Depends(get_training_db)
):
    """
    Get list of sample images generated during training.

    Returns:
        List of sample image info with step number and file path
    """
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    from pathlib import Path
    import re

    output_dir = Path(run.output_dir)
    samples_dir = output_dir / "samples"

    if not samples_dir.exists():
        return {"samples": []}

    # Find all sample images: step_{step:06d}_sample_{i}.png
    sample_files = list(samples_dir.glob("step_*_sample_*.png"))

    # Parse step numbers and organize
    samples_by_step = {}
    pattern = re.compile(r"step_(\d+)_sample_(\d+)\.png")

    # Build sample file list using run.output_dir (not UserSettings.training_dir)
    for file in sample_files:
        match = pattern.match(file.name)
        if match:
            step = int(match.group(1))
            sample_idx = int(match.group(2))

            if step not in samples_by_step:
                samples_by_step[step] = []

            # Use API endpoint to serve sample images (not static files)
            # This ensures compatibility even if UserSettings.training_dir changes
            path_url = f"/api/v1/training/runs/{run_id}/samples/{file.name}"

            # Extract generation metadata from PNG (embedded since recent version)
            img_params = None
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(str(file)) as _img:
                    if hasattr(_img, 'text') and _img.text:
                        img_params = dict(_img.text)
            except Exception:
                pass

            samples_by_step[step].append({
                "sample_index": sample_idx,
                "path": path_url,
                "params": img_params,
            })

    # Sort by step and return
    samples = []
    for step in sorted(samples_by_step.keys()):
        samples.append({
            "step": step,
            "images": sorted(samples_by_step[step], key=lambda x: x["sample_index"])
        })

    return {"samples": samples}

@router.get("/training/runs/{run_id}/samples/{filename}")
async def get_training_sample_image(
    run_id: int,
    filename: str,
    db: Session = Depends(get_training_db)
):
    """
    Serve a specific sample image file from run.output_dir
    """
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    from pathlib import Path
    output_dir = Path(run.output_dir)
    samples_dir = output_dir / "samples"
    file_path = samples_dir / filename

    # Security check: ensure file is within samples directory
    try:
        file_path = file_path.resolve()
        samples_dir = samples_dir.resolve()
        if not str(file_path).startswith(str(samples_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid file path")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")

    # Return image file
    from fastapi.responses import FileResponse
    return FileResponse(file_path, media_type="image/png")


# ============================================================
# Training Presets API
# ============================================================

class TrainingPresetCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    training_method: str  # 'lora' or 'full_finetune'
    config: Dict[str, Any]  # Training parameters (excluding dataset and model path)

class TrainingPresetUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

@router.get("/training/presets")
async def list_training_presets(db: Session = Depends(get_training_db)):
    """Get list of all training presets"""
    presets = db.query(TrainingPreset).order_by(TrainingPreset.created_at.desc()).all()
    return {"presets": [preset.to_dict() for preset in presets]}

@router.get("/training/presets/{preset_id}")
async def get_training_preset(preset_id: int, db: Session = Depends(get_training_db)):
    """Get a specific training preset by ID"""
    preset = db.query(TrainingPreset).filter(TrainingPreset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return preset.to_dict()

@router.post("/training/presets", status_code=201)
async def create_training_preset(request: TrainingPresetCreateRequest, db: Session = Depends(get_training_db)):
    """Create a new training preset"""
    # Check if name already exists
    existing = db.query(TrainingPreset).filter(TrainingPreset.name == request.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Preset with name '{request.name}' already exists")

    preset = TrainingPreset(
        name=request.name,
        description=request.description,
        training_method=request.training_method,
        config=request.config
    )
    db.add(preset)
    db.commit()
    db.refresh(preset)
    return preset.to_dict()

@router.patch("/training/presets/{preset_id}")
async def update_training_preset(preset_id: int, request: TrainingPresetUpdateRequest, db: Session = Depends(get_training_db)):
    """Update an existing training preset"""
    preset = db.query(TrainingPreset).filter(TrainingPreset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    if request.name is not None:
        # Check if new name conflicts with another preset
        existing = db.query(TrainingPreset).filter(
            TrainingPreset.name == request.name,
            TrainingPreset.id != preset_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Preset with name '{request.name}' already exists")
        preset.name = request.name

    if request.description is not None:
        preset.description = request.description

    if request.config is not None:
        preset.config = request.config

    db.commit()
    db.refresh(preset)
    return preset.to_dict()

@router.delete("/training/presets/{preset_id}")
async def delete_training_preset(preset_id: int, db: Session = Depends(get_training_db)):
    """Delete a training preset"""
    preset = db.query(TrainingPreset).filter(TrainingPreset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    db.delete(preset)
    db.commit()
    return {"message": "Preset deleted successfully"}


# ============================================================
# Batch Operations for Dataset Items
# ============================================================

from api.batch_operations import (
    BatchTaggerRequest,
    BatchReorderTagsRequest,
    BatchReplaceTagRequest,
    BatchBackfillTagDataRequest,
    BatchOperationResponse,
    batch_tagger_inference,
    batch_reorder_tags,
    batch_replace_tag,
    batch_backfill_tag_data,
    cancel_batch_operation,
)

@router.post("/datasets/{dataset_id}/batch-tagger", response_model=BatchOperationResponse)
async def batch_tagger_endpoint(
    dataset_id: int,
    request: BatchTaggerRequest,
    db: Session = Depends(get_datasets_db)
):
    """
    Run tagger inference on multiple items.
    If item_ids is empty, process all items in the dataset.
    """
    # If no items specified, get all items from dataset
    if not request.item_ids:
        all_items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()
        request.item_ids = [item.id for item in all_items]

    def send_progress(current: int, total: int, message: str):
        manager.send_progress_sync(current, total, message)

    result = await batch_tagger_inference(request, db, send_progress)

    print(f"[BatchTagger] {result.message}")
    print(f"[BatchTagger] Processed: {result.processed_count}, Updated: {result.updated_count}, Skipped: {result.skipped_count}, Failed: {result.failed_count}")

    return result

@router.post("/datasets/{dataset_id}/batch-reorder-tags", response_model=BatchOperationResponse)
async def batch_reorder_tags_endpoint(
    dataset_id: int,
    request: BatchReorderTagsRequest,
    db: Session = Depends(get_datasets_db)
):
    """
    Reorder tags by category for multiple items.
    If item_ids is empty, process all items in the dataset.
    """
    # If no items specified, get all items from dataset
    if not request.item_ids:
        all_items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()
        request.item_ids = [item.id for item in all_items]

    def send_progress(current: int, total: int, message: str):
        manager.send_progress_sync(current, total, message)

    result = await batch_reorder_tags(request, db, send_progress)

    print(f"[BatchReorder] {result.message}")
    print(f"[BatchReorder] Processed: {result.processed_count}, Updated: {result.updated_count}, Skipped: {result.skipped_count}, Failed: {result.failed_count}")

    return result

@router.post("/datasets/{dataset_id}/batch-replace-tag", response_model=BatchOperationResponse)
async def batch_replace_tag_endpoint(
    dataset_id: int,
    request: BatchReplaceTagRequest,
    db: Session = Depends(get_datasets_db)
):
    """
    Replace a specific tag with another tag for multiple items.
    If item_ids is empty, process all items in the dataset.
    """
    # If no items specified, get all items from dataset
    if not request.item_ids:
        all_items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()
        request.item_ids = [item.id for item in all_items]

    def send_progress(current: int, total: int, message: str):
        manager.send_progress_sync(current, total, message)

    result = await batch_replace_tag(request, db, send_progress)

    print(f"[BatchReplace] {result.message}")
    print(f"[BatchReplace] Processed: {result.processed_count}, Updated: {result.updated_count}, Skipped: {result.skipped_count}, Failed: {result.failed_count}")

    return result

@router.post("/datasets/{dataset_id}/backfill-tag-data", response_model=BatchOperationResponse)
async def backfill_tag_data_endpoint(
    dataset_id: int,
    db: Session = Depends(get_datasets_db)
):
    """
    Populate tag_data JSON for all is_tags_format=True captions that currently
    have tag_data=NULL in this dataset.

    This fixes captions created by bulk import/scan that have comma-separated
    tags in 'content' but no category information in 'tag_data'.
    """
    request = BatchBackfillTagDataRequest(dataset_id=dataset_id)

    def send_progress(current: int, total: int, message: str):
        manager.send_progress_sync(current, total, message)

    result = await batch_backfill_tag_data(request, db, send_progress)

    print(f"[BackfillTagData] {result.message}")
    print(f"[BackfillTagData] Processed: {result.processed_count}, Updated: {result.updated_count}, Failed: {result.failed_count}")

    return result


@router.post("/datasets/{dataset_id}/batch-cancel")
async def batch_cancel_endpoint(dataset_id: int):
    """
    Cancel the current batch operation
    """
    cancel_batch_operation()
    return {"message": "Batch operation cancellation requested"}


# ==================== Debug VRAM Inspection ====================

@router.get("/debug/vram")
async def debug_vram_inspection():
    """Inspect CUDA VRAM usage: list all GPU tensors and memory stats.
    Developer mode only debug endpoint."""
    import torch
    import gc

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    # Memory stats from PyTorch
    mem_allocated = torch.cuda.memory_allocated() / 1024**2
    mem_reserved = torch.cuda.memory_reserved() / 1024**2
    mem_max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    mem_max_reserved = torch.cuda.max_memory_reserved() / 1024**2

    # Find all CUDA tensors via gc
    gc.collect()
    cuda_tensors = []
    tensor_summary = {}  # shape+dtype -> {count, total_bytes, referrers}

    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                shape = tuple(obj.shape)
                dtype = str(obj.dtype)
                size_bytes = obj.nelement() * obj.element_size()
                key = f"{shape} {dtype}"

                if key not in tensor_summary:
                    tensor_summary[key] = {
                        "shape": list(shape),
                        "dtype": dtype,
                        "count": 0,
                        "total_mb": 0.0,
                        "referrers": [],
                    }
                tensor_summary[key]["count"] += 1
                tensor_summary[key]["total_mb"] += size_bytes / 1024**2

                # Get referrer info (what holds this tensor)
                if tensor_summary[key]["count"] <= 3:  # Limit referrer inspection
                    try:
                        referrers = gc.get_referrers(obj)
                        for ref in referrers[:3]:
                            ref_type = type(ref).__name__
                            ref_info = ref_type
                            if isinstance(ref, dict):
                                # Find the key that references this tensor
                                for k, v in ref.items():
                                    if v is obj:
                                        ref_info = f"dict['{k}']"
                                        break
                            elif isinstance(ref, (list, tuple)):
                                ref_info = f"{ref_type}[len={len(ref)}]"
                            elif hasattr(ref, '__class__'):
                                ref_info = f"{ref.__class__.__module__}.{ref.__class__.__name__}"
                            if ref_info not in tensor_summary[key]["referrers"]:
                                tensor_summary[key]["referrers"].append(ref_info)
                    except Exception:
                        pass
        except Exception:
            pass

    # Sort by total size descending
    sorted_tensors = sorted(
        tensor_summary.values(),
        key=lambda x: x["total_mb"],
        reverse=True
    )

    # Format for display
    tensor_list = []
    total_tensor_mb = 0.0
    for t in sorted_tensors:
        t["total_mb"] = round(t["total_mb"], 3)
        total_tensor_mb += t["total_mb"]
        tensor_list.append(t)

    # Pipeline component device check
    components = {}
    try:
        from core.pipeline import pipeline_manager
        for name in ['txt2img_pipeline', 'img2img_pipeline', 'inpaint_pipeline']:
            pipe = getattr(pipeline_manager, name, None)
            if pipe is not None:
                for comp_name in ['unet', 'text_encoder', 'text_encoder_2', 'vae']:
                    comp = getattr(pipe, comp_name, None)
                    if comp is not None:
                        device = str(next(comp.parameters()).device)
                        comp_key = f"{name}.{comp_name}"
                        if comp_key not in components:
                            components[comp_key] = device
    except Exception as e:
        components["error"] = str(e)

    # TAESD check
    try:
        from core.utils.taesd import taesd_manager
        for name in ['taesd', 'taesd_xl', 'taef1']:
            model = getattr(taesd_manager, name, None)
            if model is not None:
                device = str(next(model.parameters()).device)
                components[f"taesd.{name}"] = device
    except Exception as e:
        components["taesd_error"] = str(e)

    return {
        "memory": {
            "allocated_mb": round(mem_allocated, 2),
            "reserved_mb": round(mem_reserved, 2),
            "max_allocated_mb": round(mem_max_allocated, 2),
            "max_reserved_mb": round(mem_max_reserved, 2),
            "total_tensor_mb": round(total_tensor_mb, 2),
        },
        "tensor_count": sum(t["count"] for t in tensor_list),
        "unique_shapes": len(tensor_list),
        "tensors": tensor_list[:50],  # Top 50 by size
        "components": components,
    }


@router.post("/debug/vram/release")
async def debug_vram_force_release():
    """Force release all cached CUDA memory back to OS."""
    import torch
    import gc

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    before_reserved = torch.cuda.memory_reserved() / 1024**2
    before_allocated = torch.cuda.memory_allocated() / 1024**2

    gc.collect()
    torch.cuda.empty_cache()

    after_reserved = torch.cuda.memory_reserved() / 1024**2
    after_allocated = torch.cuda.memory_allocated() / 1024**2

    freed = before_reserved - after_reserved
    print(f"[VRAM] Force release: {before_reserved:.1f}MB -> {after_reserved:.1f}MB reserved (freed {freed:.1f}MB)")

    return {
        "before": {"allocated_mb": round(before_allocated, 2), "reserved_mb": round(before_reserved, 2)},
        "after": {"allocated_mb": round(after_allocated, 2), "reserved_mb": round(after_reserved, 2)},
        "freed_mb": round(freed, 2),
    }


# ============================================================
# Tagger Training API
# ============================================================

class TaggerTrainingRunCreateRequest(BaseModel):
    run_name: Optional[str] = None
    vision_encoder_path: str
    dataset_configs: List[Dict[str, Any]]          # [{dataset_id, caption_types}]
    training_method: str = "lora"                  # "lora" | "full"
    lora_rank: int = 32
    lora_alpha: float = 16.0
    learning_rate: float = 3e-4
    head_lr_multiplier: float = 10.0
    optimizer: str = "adamw8bit"
    warmup_steps: int = 100
    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 4
    num_workers_override: Optional[int] = None
    tag_refresh_enable: bool = TAGGER_TRAINING_DEFAULTS["tag_refresh_enable"]
    tag_refresh_interval_seconds: int = TAGGER_TRAINING_DEFAULTS["tag_refresh_interval_seconds"]
    save_every_n_steps: int = 500
    save_every_n_epochs: int = 0
    keep_last_n_checkpoints: int = 3
    checkpoint_save_mode: str = "lora"
    mixed_precision: str = "bf16"
    use_flash_attention: bool = TAGGER_TRAINING_DEFAULTS["use_flash_attention"]
    gradient_checkpointing: bool = True
    weight_decay: float = 1e-4
    loss_function: str = "asl"
    loss_gamma_neg: float = 4.0
    loss_gamma_pos: float = 1.0
    loss_clip: float = 0.05
    loss_gamma0: float = 4.0
    loss_m0: float = 0.2
    loss_beta: float = 2.0
    loss_rho: float = 0.5
    loss_label_weight: str = "fisher"
    validate_every: int = 1
    val_split: float = 0.05
    val_split_mode: str = "percent"
    val_fixed_size: Optional[int] = None
    save_best_only: bool = False
    vocab_min_count: int = 10
    vocab_use_gelbooru_categories: bool = TAGGER_TRAINING_DEFAULTS["vocab_use_gelbooru_categories"]
    output_dir: Optional[str] = None
    excluded_categories: Optional[List[str]] = None
    ban_tags: Optional[str] = None
    use_tag_aliases: bool = False
    save_base_model: bool = False
    # Quality-tag loss masking strategy: "intra_group" or "cross_group".
    # See core/tagger/tagger_dataset.py / param_defaults.py for semantics.
    quality_masking_mode: str = "intra_group"
    cls_dim: Optional[int] = None
    hidden_proj_dim: Optional[int] = None
    init_head_from: Optional[str] = None
    # LR matrix (conditional inference) — built once at training start when enabled.
    build_lr_matrix_on_start: bool = False
    lr_top_anchors:            int   = 10000
    lr_top_targets:            int   = 1000
    lr_threshold:              float = 1.0
    lr_min_anchor_count:       int   = 10
    # Pre-flight dataset drift check + optional rescan.  Four modes (see
    # core/training/dataset_drift.py):
    #   "off"   — skip entirely (default)
    #   "path"  — only detect added/missing files (cheap path set-diff)
    #   "smart" — path drift + caption sidecar mtime check
    #   "force" — always rescan, no drift detection
    # Legacy bool accepted (True→"path", False→"off").
    rescan_before_training:    Any   = TAGGER_TRAINING_DEFAULTS["rescan_before_training"]
    # Training F1: rolling buffer + periodic threshold search.
    # N2 (eval_every) < N1 (search_every).  0 disables the feature.
    train_f1_eval_every_n_steps:             int   = 100
    train_f1_threshold_search_every_n_steps: int   = 500
    train_f1_initial_threshold:              float = 0.35
    train_f1_buffer_batches:                 int   = 16
    # Online Danbooru augmentation
    enable_danbooru_augmentation: bool = False
    # Query mode (first-class collection mode; see param_defaults.py)
    danbooru_query_enable: bool = TAGGER_TRAINING_DEFAULTS["danbooru_query_enable"]
    danbooru_query_expand_enable: bool = TAGGER_TRAINING_DEFAULTS["danbooru_query_expand_enable"]
    danbooru_query_new_tag_min_count: int = TAGGER_TRAINING_DEFAULTS["danbooru_query_new_tag_min_count"]
    danbooru_query_resolve_top_k: int = TAGGER_TRAINING_DEFAULTS["danbooru_query_resolve_top_k"]
    danbooru_query_max_expanded_tags: int = TAGGER_TRAINING_DEFAULTS["danbooru_query_max_expanded_tags"]
    danbooru_query_expand_categories: List[int] = TAGGER_TRAINING_DEFAULTS["danbooru_query_expand_categories"]
    danbooru_query_resolve_interval: int = TAGGER_TRAINING_DEFAULTS["danbooru_query_resolve_interval"]
    danbooru_query_collect_per_epoch: int = TAGGER_TRAINING_DEFAULTS["danbooru_query_collect_per_epoch"]
    danbooru_new_tag_collect_per_epoch: int = TAGGER_TRAINING_DEFAULTS["danbooru_new_tag_collect_per_epoch"]
    danbooru_low_f1_collect_per_epoch: int = TAGGER_TRAINING_DEFAULTS["danbooru_low_f1_collect_per_epoch"]
    # Train-count deficiency collection (exposure balancing)
    danbooru_train_count_enable: bool = TAGGER_TRAINING_DEFAULTS["danbooru_train_count_enable"]
    danbooru_train_count_top_k: int = TAGGER_TRAINING_DEFAULTS["danbooru_train_count_top_k"]
    danbooru_train_count_min_deficit_ratio: float = TAGGER_TRAINING_DEFAULTS["danbooru_train_count_min_deficit_ratio"]
    danbooru_train_count_min_per_epoch: int = TAGGER_TRAINING_DEFAULTS["danbooru_train_count_min_per_epoch"]
    danbooru_train_count_min_posts: int = TAGGER_TRAINING_DEFAULTS["danbooru_train_count_min_posts"]
    danbooru_train_count_collect_per_epoch: int = TAGGER_TRAINING_DEFAULTS["danbooru_train_count_collect_per_epoch"]
    danbooru_query_weight_train_count: float = TAGGER_TRAINING_DEFAULTS["danbooru_query_weight_train_count"]
    danbooru_quality_tag_enable: bool = TAGGER_TRAINING_DEFAULTS["danbooru_quality_tag_enable"]
    danbooru_quality_tag_thresholds: str = TAGGER_TRAINING_DEFAULTS["danbooru_quality_tag_thresholds"]
    danbooru_quality_tag_attach_negative: bool = TAGGER_TRAINING_DEFAULTS["danbooru_quality_tag_attach_negative"]
    danbooru_tags: Optional[str] = None        # newline-separated; use !tag or -tag to exclude
    danbooru_injection_interval: int = 4       # interrupt-batch every N base steps
    danbooru_injection_batch_size_ratio: float = 1.0  # 1.0=B, 0.5=B/2, etc.
    danbooru_min_score: int = 0
    danbooru_max_posts_per_query: int = 200
    danbooru_api_interval: float = 1.4
    danbooru_dl_speed_kbps: int = 500
    danbooru_speed_check_enable: bool = TAGGER_TRAINING_DEFAULTS["danbooru_speed_check_enable"]
    danbooru_speed_degraded_kbps: int = TAGGER_TRAINING_DEFAULTS["danbooru_speed_degraded_kbps"]
    danbooru_speed_min_slow_streak: int = TAGGER_TRAINING_DEFAULTS["danbooru_speed_min_slow_streak"]
    danbooru_speed_min_slow_seconds: int = TAGGER_TRAINING_DEFAULTS["danbooru_speed_min_slow_seconds"]
    danbooru_speed_cooldown_seconds: int = TAGGER_TRAINING_DEFAULTS["danbooru_speed_cooldown_seconds"]
    danbooru_buffer_size: Optional[int] = None  # None → auto (2 × batch_size)
    danbooru_vocab_expand: bool = False
    danbooru_new_tag_min_count: int = 200
    danbooru_new_tag_min_count_by_cat: Dict[str, int] = {}
    danbooru_new_tag_lookback_days: int = 90
    danbooru_new_tag_categories: List[int] = [0, 3, 4]
    danbooru_new_tag_survey_interval: int = 3600
    danbooru_max_dynamic_tags: int = 0
    # Collection-path weights (weighted selection among available paths)
    danbooru_query_weight_static: float = 1.0
    danbooru_query_weight_new_tag: float = 1.0
    danbooru_query_weight_low_f1: float = 1.0
    # Low-F1 deficiency collection (existing vocab tags with low per-tag F1)
    danbooru_low_f1_enable: bool = False
    danbooru_low_f1_threshold: float = 0.5
    danbooru_low_f1_top_k: int = 500
    danbooru_low_f1_min_posts: int = 50
    # Co-occurrence vocab discovery (vocab-absent tags seen in collected posts)
    danbooru_cooc_expand_enable: bool = False
    danbooru_cooc_min_count: int = 50
    danbooru_cooc_categories: List[int] = [0, 3, 4]
    danbooru_query_weight_cooc: float = TAGGER_TRAINING_DEFAULTS["danbooru_query_weight_cooc"]
    danbooru_cooc_collect_per_epoch: int = TAGGER_TRAINING_DEFAULTS["danbooru_cooc_collect_per_epoch"]
    danbooru_cooc_order_random: bool = TAGGER_TRAINING_DEFAULTS["danbooru_cooc_order_random"]
    # save_tag_metrics / hard_rate (passed through to trainer)
    save_tag_metrics: bool = True
    hard_rate_lo: float = 0.25
    hard_rate_hi: float = 0.75


# Active tagger training threads
_tagger_training_threads: Dict[str, Any] = {}


def _make_tagger_progress_callback(run_id: str, training_db_factory):
    """Returns a callback that writes progress to DB.

    Computes ``resume_seq`` once at factory-creation time: ``max(resume_seq)
    + 1`` if any prior metrics exist for this ``run_id`` (= this is a
    resume), else 0 (fresh run).  The value is captured in the closure
    and stamped on every metric row + WS payload, so each resume
    contributes a distinct curve in the loss chart.
    """
    from sqlalchemy import func as _sa_func
    _db = training_db_factory()
    try:
        _max_seq = _db.query(
            _sa_func.coalesce(_sa_func.max(TaggerTrainingMetrics.resume_seq), -1)
        ).filter(TaggerTrainingMetrics.run_id == run_id).scalar()
        resume_seq = int(_max_seq) + 1   # 0 for fresh; existing_max + 1 on resume
    except Exception as e:
        print(f"[TaggerCallback] Could not determine resume_seq for {run_id}: {e}; defaulting to 0")
        resume_seq = 0
    finally:
        _db.close()
    if resume_seq > 0:
        print(f"[TaggerCallback] run_id={run_id}: resume_seq={resume_seq} (subsequent resume)")
    else:
        print(f"[TaggerCallback] run_id={run_id}: resume_seq=0 (initial run)")

    def callback(rid: str, event_type: str, data: dict):
        db = training_db_factory()
        try:
            run = db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == rid).first()
            if not run:
                return
            def _upsert_metric(step: int, **kwargs):
                existing = db.query(TaggerTrainingMetrics).filter(
                    TaggerTrainingMetrics.run_id == rid,
                    TaggerTrainingMetrics.resume_seq == resume_seq,
                    TaggerTrainingMetrics.step == step,
                ).first()
                if existing:
                    for k, v in kwargs.items():
                        if v is not None:
                            setattr(existing, k, v)
                else:
                    db.add(TaggerTrainingMetrics(
                        run_id=rid, resume_seq=resume_seq, step=step, **kwargs
                    ))

            if event_type == "step":
                run.current_step  = data.get("step", run.current_step)
                run.current_epoch = data.get("epoch", run.current_epoch)
                run.latest_loss   = data.get("loss")
                run.latest_lr     = data.get("lr")
                run.progress      = data.get("progress", run.progress)
                run.status_message = None  # clear preparation message once training steps begin
                _upsert_metric(
                    step=data.get("step", 0),
                    epoch=data.get("epoch"),
                    loss=data.get("loss"),
                    learning_rate=data.get("lr"),
                )
                manager.send_tagger_metrics(
                    run_id=rid,
                    event_type="step",
                    step=data.get("step", 0),
                    epoch=data.get("epoch"),
                    loss=data.get("loss"),
                    lr=data.get("lr"),
                    progress=data.get("progress"),
                    resume_seq=resume_seq,
                )
            elif event_type == "epoch":
                run.current_epoch = data.get("epoch", run.current_epoch)
                run.latest_loss   = data.get("loss")
                # Use step from trainer emit (includes global_step), fallback to tracked step
                _epoch_step = data.get("step", run.current_step)
                # Always upsert the epoch row so the step is recorded for epoch-boundary
                # tracking even when validation is skipped (f1 will be None in that case).
                _upsert_metric(
                    step=_epoch_step,
                    epoch=data.get("epoch"),
                    loss=data.get("loss"),
                    f1=data.get("f1"),
                    threshold=data.get("threshold"),
                    precision=data.get("precision"),
                    recall=data.get("recall"),
                )
                manager.send_tagger_metrics(
                    run_id=rid,
                    event_type="epoch",
                    step=_epoch_step,
                    epoch=data.get("epoch"),
                    loss=data.get("loss"),
                    f1=data.get("f1"),
                    threshold=data.get("threshold"),
                    resume_seq=resume_seq,
                    precision=data.get("precision"),
                    recall=data.get("recall"),
                )
            elif event_type == "checkpoint":
                if data.get("name") == "best_f1":
                    run.best_f1 = data.get("f1")
                    run.best_checkpoint_path = os.path.join(
                        run.output_dir or "", "best_f1.safetensors"
                    )
                elif data.get("path"):
                    # Step-based checkpoint: append path to checkpoint_paths list
                    paths = list(run.checkpoint_paths or [])
                    ckpt_path = data["path"]
                    if ckpt_path not in paths:
                        paths.append(ckpt_path)
                    run.checkpoint_paths = paths
            elif event_type == "train_f1":
                _upsert_metric(
                    step=data.get("step", 0),
                    train_f1=data.get("train_f1"),
                    threshold=data.get("threshold") if data.get("threshold_updated") else None,
                    precision=data.get("train_precision"),
                    recall=data.get("train_recall"),
                )
                manager.send_tagger_metrics(
                    run_id=rid,
                    event_type="train_f1",
                    step=data.get("step", 0),
                    train_f1=data.get("train_f1"),
                    threshold=data.get("threshold") if data.get("threshold_updated") else None,
                    resume_seq=resume_seq,
                    precision=data.get("train_precision"),
                    recall=data.get("train_recall"),
                    fp_fn_scatter=data.get("fp_fn_scatter"),
                )
            elif event_type == "vocab":
                run.num_tags = data.get("num_tags")
            elif event_type == "resume":
                run.resumed_from_step = data.get("resumed_from_step")
                run.last_resumed_at   = datetime.now()
            elif event_type == "phase":
                msg = data.get("message") or data.get("phase") or ""
                run.status_message = msg
            elif event_type == "dataset_progress":
                # Live pre-training dataset-loading progress → WS/SSE bar.
                msg = data.get("message") or "Loading dataset..."
                try:
                    manager.send_progress_sync(
                        int(data.get("step", 0)),
                        int(data.get("total", 1)),
                        msg,
                    )
                except Exception:
                    pass
                run.status_message = msg
            elif event_type == "completed":
                run.status         = "completed"
                run.progress       = 1.0
                run.best_f1        = data.get("best_f1")
                run.best_threshold = data.get("optimal_threshold") or data.get("best_threshold")
                run.threshold_f1_curve = data.get("threshold_f1_curve")
                run.total_steps    = data.get("total_steps")
                run.completed_at   = datetime.now()
                run.latest_checkpoint_path = os.path.join(run.output_dir or "", "latest.safetensors")
            db.commit()
        except Exception as e:
            print(f"[TaggerCallback] DB error: {e}")
            db.rollback()
        finally:
            db.close()
    return callback


@router.post("/tagger-training/runs")
def create_tagger_training_run(
    request: TaggerTrainingRunCreateRequest,
    training_db: Session = Depends(get_training_db),
):
    """Create a new tagger training run record."""
    import uuid as _uuid

    run_id = str(_uuid.uuid4())
    run_name = request.run_name or f"tagger-{run_id[:8]}"

    # Determine output directory
    output_dir = request.output_dir or os.path.join(
        settings.root_dir, "tagger_models", run_id
    )

    config = request.dict()
    config["vision_encoder_path"] = request.vision_encoder_path

    run = TaggerTrainingRun(
        run_id=run_id,
        run_name=run_name,
        status="pending",
        training_method=request.training_method,
        vision_encoder_path=request.vision_encoder_path,
        dataset_configs=request.dataset_configs,
        output_dir=output_dir,
        config=config,
        total_epochs=request.epochs,
    )
    training_db.add(run)
    training_db.commit()
    training_db.refresh(run)
    return run.to_dict()


@router.get("/tagger-training/runs")
def list_tagger_training_runs(training_db: Session = Depends(get_training_db)):
    """List all tagger training runs."""
    runs = training_db.query(TaggerTrainingRun).order_by(TaggerTrainingRun.created_at.desc()).all()
    return [r.to_list_dict() for r in runs]


@router.get("/tagger-training/runs/{run_id}")
def get_tagger_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Get a tagger training run by run_id (excludes tag_vocabulary; fetch /vocabulary for that)."""
    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Tagger training run not found")
    return run.to_list_dict()


@router.patch("/tagger-training/runs/{run_id}")
def update_tagger_training_run(
    run_id: str,
    request: TaggerTrainingRunCreateRequest,
    training_db: Session = Depends(get_training_db),
):
    """Update configuration of a pending/stopped/failed tagger training run."""
    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Tagger training run not found")
    if run.status not in ("pending", "stopped", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot edit a run with status '{run.status}'. Only pending/stopped/failed runs can be edited."
        )

    config = request.dict()
    config["vision_encoder_path"] = request.vision_encoder_path

    run.run_name            = request.run_name or run.run_name
    run.training_method     = request.training_method
    run.vision_encoder_path = request.vision_encoder_path
    run.dataset_configs     = request.dataset_configs
    run.config              = config
    run.total_epochs        = request.epochs
    # Reset progress/metrics so re-run starts clean
    run.status              = "pending"
    run.progress            = 0.0
    run.current_epoch       = 0
    run.current_step        = 0
    run.status_message      = None

    training_db.commit()
    training_db.refresh(run)
    return run.to_dict()


@router.get("/tagger-training/runs/{run_id}/vocabulary")
def get_tagger_training_vocabulary(run_id: str, training_db: Session = Depends(get_training_db)):
    """Return the tag vocabulary for a tagger training run.

    Returns the vocabulary stored in the DB (tag_vocabulary column), or reads
    vocabulary.json from the output_dir if the DB field is not yet populated.
    """
    import json as _json
    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Tagger training run not found")

    # Prefer DB-cached vocabulary
    if run.tag_vocabulary:
        return run.tag_vocabulary

    # Fall back to reading vocabulary.json from disk
    if run.output_dir:
        vocab_path = os.path.join(run.output_dir, "vocabulary.json")
        if os.path.isfile(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                return _json.load(f)

    raise HTTPException(status_code=404, detail="Vocabulary not yet available for this run")


@router.post("/tagger-training/runs/{run_id}/start")
async def start_tagger_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Start or resume a tagger training run in a background thread.

    When ``config.rescan_before_training`` is True, runs a pre-flight
    drift check against the dataset(s) and, if drift is detected,
    invokes a full rescan before launching the trainer thread.
    """
    import threading
    from database import TrainingSessionLocal, DatasetsSessionLocal

    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Tagger training run not found")
    if run.status in ("running", "starting"):
        raise HTTPException(status_code=400, detail="Run is already running")

    run.status        = "starting"
    run.started_at    = datetime.now()
    run.error_message = None
    training_db.commit()

    config          = run.config or {}
    dataset_configs = run.dataset_configs or []
    dataset_ids     = [dc["dataset_id"] for dc in dataset_configs]
    output_dir      = run.output_dir

    # Pass output_dir so the trainer auto-detects any resumable checkpoint
    # (latest_state.json or step_XXXXXX_state.json).  When no checkpoint exists,
    # the trainer simply starts from epoch 1.
    resume_from_checkpoint = output_dir if output_dir and os.path.isdir(output_dir) else None

    callback = _make_tagger_progress_callback(run_id, TrainingSessionLocal)

    trainer_holder: list = []

    def _run():
        import asyncio
        from core.tagger.tagger_trainer import run_tagger_training
        from database import DatasetsSessionLocal
        from core.training.dataset_drift import normalize_rescan_mode

        db = TrainingSessionLocal()
        try:
            # ----- Pre-flight: dataset drift detection / optional rescan -----
            # Runs inside the background thread so the HTTP /start response
            # returns immediately (pre-flight can take hours on large datasets).
            _rescan_mode = normalize_rescan_mode(config.get("rescan_before_training"))
            if _rescan_mode != "off" and dataset_ids:
                from core.training.dataset_drift import (
                    detect_drift, rescan_dataset_inline,
                )
                from core.training.rescan_control import rescan_skip_controller, RescanSkipped
                from database.models import Dataset as _Dataset
                ddb = DatasetsSessionLocal()
                try:
                    for ds_id in dataset_ids:
                        # Resolve dataset name once for progress display.
                        try:
                            _ds_row = ddb.query(_Dataset).filter(_Dataset.id == int(ds_id)).first()
                            _ds_name = (_ds_row.name if _ds_row else "") or ""
                        except Exception:
                            _ds_name = ""

                        rescan_skip_controller.begin("tagger", run_id, int(ds_id))
                        # Start of the skippable window (begin→end): the UI shows
                        # the Skip button only between scan_start and scan_end.
                        try:
                            manager.send_dataset_scan_progress(
                                scope="tagger", run_id=run_id,
                                dataset_id=int(ds_id), phase="scan_start",
                                dataset_name=_ds_name,
                            )
                        except Exception:
                            pass
                        def _skip_cb(_rid=run_id):
                            return rescan_skip_controller.should_skip("tagger", _rid)
                        try:
                            should_rescan: bool
                            report = None

                            if _rescan_mode == "force":
                                should_rescan = True
                                callback(run_id, "phase", {
                                    "phase": "dataset_rescan",
                                    "message": f"Force rescan: {_ds_name or ds_id}...",
                                })
                            else:
                                # "path" or "smart" — walk and compare.
                                def _drift_progress(files_walked: int, _ds_id=ds_id, _nm=_ds_name):
                                    try:
                                        manager.send_dataset_scan_progress(
                                            scope="tagger", run_id=run_id,
                                            dataset_id=int(_ds_id), phase="drift_walk",
                                            files_walked=files_walked, dataset_name=_nm,
                                        )
                                    except Exception:
                                        pass
                                callback(run_id, "phase", {
                                    "phase": "dataset_drift",
                                    "message": f"Drift check ({_rescan_mode}): {_ds_name or ds_id}...",
                                })
                                report = detect_drift(
                                    int(ds_id), ddb,
                                    check_caption_mtime=(_rescan_mode == "smart"),
                                    progress_callback=_drift_progress,
                                    should_cancel=_skip_cb,
                                )
                                print(f"[TaggerTraining] Drift {ds_id} ({_rescan_mode}): {report.to_dict()}")
                                callback(run_id, "dataset_drift", report.to_dict())
                                try:
                                    manager.send_dataset_scan_progress(
                                        scope="tagger", run_id=run_id,
                                        dataset_id=int(ds_id), phase="drift_done",
                                        files_walked=report.files_walked,
                                        items_in_db=report.items_in_db,
                                        items_missing=report.items_missing,
                                        items_new=report.items_new,
                                        dataset_name=_ds_name,
                                    )
                                except Exception:
                                    pass
                                should_rescan = report.has_drift

                            if should_rescan:
                                if _rescan_mode == "force":
                                    reason = "force mode"
                                else:
                                    parts = []
                                    if report.items_missing:  parts.append(f"{report.items_missing} missing")
                                    if report.items_new:      parts.append(f"{report.items_new} new")
                                    if report.captions_stale: parts.append(f"{report.captions_stale} stale captions")
                                    reason = ", ".join(parts) or "drift detected"
                                callback(run_id, "phase", {
                                    "phase": "dataset_rescan",
                                    "message": f"Rescanning {_ds_name or ds_id} ({reason})...",
                                })
                                try:
                                    manager.send_dataset_scan_progress(
                                        scope="tagger", run_id=run_id,
                                        dataset_id=int(ds_id), phase="rescan",
                                        files_walked=(report.files_walked if report else 0),
                                        items_missing=(report.items_missing if report else 0),
                                        items_new=(report.items_new if report else 0),
                                        message=f"Rescanning... ({reason})",
                                        dataset_name=_ds_name,
                                    )
                                except Exception:
                                    pass
                                try:
                                    _res = asyncio.run(rescan_dataset_inline(int(ds_id), ddb, should_cancel=_skip_cb))
                                    if isinstance(_res, dict) and _res.get("cancelled"):
                                        print(f"[TaggerTraining] Rescan of {ds_id} skipped by user (partial commit kept)")
                                        try:
                                            manager.send_dataset_scan_progress(
                                                scope="tagger", run_id=run_id,
                                                dataset_id=int(ds_id), phase="skipped",
                                                dataset_name=_ds_name,
                                                message=f"Skipped rescan of {_ds_name or ds_id}",
                                            )
                                        except Exception:
                                            pass
                                except Exception as _rs_e:
                                    print(f"[TaggerTraining] Rescan of {ds_id} failed: {_rs_e}")
                        except RescanSkipped:
                            # Raised from the drift walk when skipped.
                            print(f"[TaggerTraining] Drift/rescan of {ds_id} skipped by user")
                            try:
                                manager.send_dataset_scan_progress(
                                    scope="tagger", run_id=run_id,
                                    dataset_id=int(ds_id), phase="skipped",
                                    dataset_name=_ds_name,
                                    message=f"Skipped rescan of {_ds_name or ds_id}",
                                )
                            except Exception:
                                pass
                        finally:
                            rescan_skip_controller.end("tagger", run_id)
                            # End of the skippable window → UI hides the button.
                            try:
                                manager.send_dataset_scan_progress(
                                    scope="tagger", run_id=run_id,
                                    dataset_id=int(ds_id), phase="scan_end",
                                    dataset_name=_ds_name,
                                )
                            except Exception:
                                pass
                finally:
                    ddb.close()

            # Update status to "running" now that pre-flight is complete
            row = db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
            if row and row.status == "starting":
                row.status = "running"
                db.commit()

            result = run_tagger_training(
                run_id=run_id,
                config=config,
                dataset_ids=dataset_ids,
                output_dir=output_dir,
                progress_callback=callback,
                resume_from_checkpoint=resume_from_checkpoint,
                trainer_holder=trainer_holder,
            )
            # Save vocabulary to DB
            vocab_path = os.path.join(output_dir, "vocabulary.json")
            if os.path.isfile(vocab_path):
                import json as _json
                with open(vocab_path, "r", encoding="utf-8") as f:
                    vocab = _json.load(f)
                row = db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
                if row:
                    row.tag_vocabulary = vocab
                    row.num_tags = vocab.get("num_tags")
                    db.commit()
        except Exception as e:
            row = db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
            if row:
                row.status = "failed"
                row.error_message = str(e)
                db.commit()
            print(f"[TaggerTraining] Run {run_id} failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            db.close()
            _tagger_training_threads.pop(run_id, None)


    thread = threading.Thread(target=_run, daemon=True, name=f"tagger-{run_id[:8]}")
    _tagger_training_threads[run_id] = {"thread": thread, "trainer_holder": trainer_holder}
    thread.start()

    training_db.refresh(run)
    return {"message": "started", "run": run.to_dict()}


@router.post("/tagger-training/runs/{run_id}/stop")
def stop_tagger_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Request stop for a running tagger training run."""
    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Tagger training run not found")

    # Signal the trainer to stop at the next checkpoint boundary
    entry = _tagger_training_threads.get(run_id)
    if entry:
        trainer_holder = entry.get("trainer_holder", [])
        if trainer_holder:
            trainer_holder[0].stop()
            print(f"[TaggerTraining] Stop signal sent to trainer {run_id}")
        else:
            print(f"[TaggerTraining] Stop requested for {run_id} but trainer not yet initialized")

    run.status = "stopped"
    training_db.commit()
    training_db.refresh(run)
    return {"message": "stop_requested", "run": run.to_dict()}


@router.post("/tagger-training/runs/{run_id}/skip-rescan")
def skip_tagger_rescan(run_id: str, request: SkipRescanRequest = SkipRescanRequest()):
    """Skip the dataset currently being rescanned in this tagger run's pre-flight.

    Flags the cooperative-cancel flag the rescan's directory walkers poll, so the
    current dataset's drift-walk / rescan aborts (keeping any already-applied
    changes) and the pre-flight continues with the remaining datasets.
    """
    from core.training.rescan_control import rescan_skip_controller
    flagged = rescan_skip_controller.request_skip("tagger", run_id, request.dataset_id)
    return {
        "skipped": flagged,
        "current_dataset": rescan_skip_controller.current_dataset("tagger", run_id),
    }


@router.delete("/tagger-training/runs/{run_id}")
def delete_tagger_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Delete a tagger training run record."""
    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Tagger training run not found")
    training_db.delete(run)
    training_db.commit()
    return {"deleted": run_id}


@router.get("/tagger-training/runs/{run_id}/danbooru-metrics")
def get_tagger_danbooru_metrics(
    run_id: str,
    training_db: Session = Depends(get_training_db),
):
    """Return Danbooru augmentation metrics for a tagger run.

    Reads ``{output_dir}/danbooru_metrics.json`` written periodically by the
    trainer (every 10 base steps).  Returns ``enabled=false`` when the file
    is missing (augmentation disabled or no steps written yet).
    """
    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if not run.output_dir or not os.path.isdir(run.output_dir):
        return {"enabled": False}
    path = os.path.join(run.output_dir, "danbooru_metrics.json")
    if not os.path.isfile(path):
        return {"enabled": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["enabled"] = True
        return data
    except Exception as e:
        return {"enabled": False, "error": str(e)}


@router.post("/tagger-training/runs/{run_id}/danbooru/resume")
def resume_tagger_danbooru(run_id: str, training_db: Session = Depends(get_training_db)):
    """Manually resume Danbooru collection after a speed-degradation cooldown
    (tagger run). Clears the active cooldown on the worker's next poll."""
    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if not run.output_dir or not os.path.isdir(run.output_dir):
        raise HTTPException(status_code=400, detail="Run output dir unavailable")
    _write_danbooru_resume(run.output_dir)
    return {"success": True}


@router.get("/tagger-training/runs/{run_id}/metrics")
def get_tagger_training_metrics(
    run_id: str,
    since_step: int = 0,
    max_points: int = 2000,
    training_db: Session = Depends(get_training_db),
):
    """Get per-step metrics for a tagger training run.

    Parameters
    ----------
    max_points : Maximum number of data points to return (uniform decimation).
                 0 = no limit (not recommended for long runs).
    """
    rows = (
        training_db.query(TaggerTrainingMetrics)
        .filter(
            TaggerTrainingMetrics.run_id == run_id,
            TaggerTrainingMetrics.step >= since_step,
        )
        .order_by(TaggerTrainingMetrics.resume_seq, TaggerTrainingMetrics.step)
        .all()
    )
    # Group by resume_seq so each curve gets its own decimation budget.
    # Without this, sparse early resumes can get fully decimated away when a
    # later resume has many more points.
    from collections import defaultdict as _defaultdict
    groups: Dict[int, List] = _defaultdict(list)
    for m in rows:
        groups[m.resume_seq].append(m)

    # Validation rows (f1 / threshold non-null) are sparse — typically one per
    # epoch — and easily fall through the cracks of a fixed-step decimation
    # grid.  Split them out, decimate only the dense training-loss rows, and
    # always include the full validation set.  Without this the Validation F1
    # and Optimal Threshold charts render empty for any run with > max_points
    # total metrics.
    def _is_validation_row(m) -> bool:
        return getattr(m, "f1", None) is not None or getattr(m, "threshold", None) is not None

    data: List[dict] = []
    if max_points > 0 and len(rows) > max_points:
        per_group_max = max(50, max_points // max(1, len(groups)))
        for seq in sorted(groups):
            g_all   = groups[seq]
            g_train = [m for m in g_all if not _is_validation_row(m)]
            g_val   = [m for m in g_all if _is_validation_row(m)]
            # Decimate training-loss rows only
            if len(g_train) > per_group_max:
                step_size = max(1, len(g_train) // per_group_max)
                indices = list(range(0, len(g_train), step_size))
                if indices[-1] != len(g_train) - 1:
                    indices.append(len(g_train) - 1)
                g_train = [g_train[i] for i in indices]
            # Re-merge by step so the consumer sees a monotonic series
            merged = sorted(g_train + g_val, key=lambda m: m.step)
            data.extend(m.to_dict() for m in merged)
    else:
        for seq in sorted(groups):
            data.extend(m.to_dict() for m in groups[seq])
    return data


@router.get("/tagger-training/vocabulary-preview")
def preview_tagger_vocabulary(
    dataset_ids: str,
    excluded_categories: Optional[str] = None,
    ban_tags: Optional[str] = None,
    use_gelbooru_categories: bool = TAGGER_TRAINING_DEFAULTS["vocab_use_gelbooru_categories"],
    datasets_db: Session = Depends(get_datasets_db),
):
    """Preview tag vocabulary for given dataset IDs (comma-separated).

    Returns tag count and category breakdown without full vocab.

    Parameters
    ----------
    dataset_ids             : comma-separated dataset IDs
    excluded_categories     : comma-separated category names to exclude
    ban_tags                : newline or comma-separated tag patterns (fnmatch wildcards)
    use_gelbooru_categories : when True, resolve categories absent from the
                              Danbooru taglist against the Gelbooru supplement
                              (matches the training-time vocabulary builder).
    """
    import json as _json
    import fnmatch
    from collections import defaultdict
    from database.models import DatasetItem, DatasetCaption
    from core.tagger.tag_vocabulary import normalize_tag, CATEGORY_ORDER
    from utils.taglist_cache import taglist_cache

    ids = [int(i) for i in dataset_ids.split(",") if i.strip()]
    excl_cats = {c.strip() for c in excluded_categories.split(",") if c.strip()} if excluded_categories else set()
    ban_list  = [t.strip() for t in (ban_tags or "").replace(",", "\n").splitlines() if t.strip()]

    # Mirror the training-time builder: optionally enable the Gelbooru taglist
    # supplement so the previewed "Unknown" count matches what training produces.
    taglist_cache.initialize(settings.root_dir, enable_gelbooru=bool(use_gelbooru_categories))

    # Same comma-split fragment re-merge the vocabulary builder applies, so the
    # preview's tag/category counts match the real vocabulary.
    from core.tagger.comma_tag_resolver import CommaTagResolver
    comma_resolver = CommaTagResolver.build_from_category_map(taglist_cache._category_map)

    # Fetch only the columns we need — avoids loading full ORM objects for large datasets
    rows = (
        datasets_db.query(DatasetCaption.tag_data, DatasetCaption.content)
        .join(DatasetItem, DatasetCaption.item_id == DatasetItem.id)
        .filter(
            DatasetItem.dataset_id.in_(ids),
            DatasetCaption.is_tags_format == True,
        )
        .all()
    )

    tag_counts: dict = defaultdict(int)
    tag_categories: dict = {}

    for tag_data_json, content in rows:
        # Build the per-caption ordered (token, source-category) list, then
        # re-merge comma-split fragments before counting.
        ordered: list = []  # (normalized_token, source_category_or___lookup__)
        if tag_data_json:
            try:
                items = _json.loads(tag_data_json) if isinstance(tag_data_json, str) else tag_data_json
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and "tag" in item:
                            nt = normalize_tag(item["tag"])
                            if nt:
                                ordered.append((nt, item.get("category", "General")))
            except Exception:
                ordered = []
        if not ordered and content:
            for t in content.split(","):
                nt = normalize_tag(t.strip())
                if nt:
                    ordered.append((nt, "__lookup__"))

        if not ordered:
            continue

        src_cat = {nt: cat for nt, cat in ordered}
        canon_tokens = comma_resolver.resolve([nt for nt, _ in ordered])
        for nt in canon_tokens:
            comma_cat = comma_resolver.category_of(nt)
            cat = comma_cat if comma_cat is not None else src_cat.get(nt, "General")
            tag_counts[nt] += 1
            if nt not in tag_categories:
                tag_categories[nt] = cat

    # Batch-resolve any __lookup__ sentinels AND pre-existing "Unknown" tags
    # (tag_data built before a tag entered the taglist). Mirrors the training-time
    # builder so the preview's category breakdown matches the real vocabulary.
    resolve_targets = list({
        t for t, c in tag_categories.items() if c in ("__lookup__", "Unknown")
    })
    if resolve_targets:
        resolved = taglist_cache.get_categories_batch(resolve_targets)
        for norm_tag in resolve_targets:
            original = tag_categories.get(norm_tag)
            found = resolved.get(norm_tag)
            if found:
                tag_categories[norm_tag] = found
            elif original == "__lookup__":
                # Sentinel with no taglist hit → default to General (as before).
                tag_categories[norm_tag] = "General"
            # else: keep "Unknown" if the taglist doesn't know it either.

    # Apply filters
    if excl_cats:
        tag_counts = {t: c for t, c in tag_counts.items()
                      if tag_categories.get(t, "General") not in excl_cats}
    if ban_list:
        tag_counts = {t: c for t, c in tag_counts.items()
                      if not any(fnmatch.fnmatch(t, pat) for pat in ban_list)}

    # Category counts (sorted by CATEGORY_ORDER)
    cat_counts: dict = defaultdict(int)
    for tag in tag_counts:
        cat_counts[tag_categories.get(tag, "General")] += 1

    sorted_cat_counts = dict(
        sorted(cat_counts.items(),
               key=lambda kv: CATEGORY_ORDER.index(kv[0]) if kv[0] in CATEGORY_ORDER else len(CATEGORY_ORDER))
    )

    return {
        "num_tags": len(tag_counts),
        "category_counts": sorted_cat_counts,
    }
