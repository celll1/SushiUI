from PIL import Image, PngImagePlugin
from typing import Dict, Any, Optional
import os
import hashlib
import base64
from io import BytesIO
from datetime import datetime
from config.settings import settings
from utils.path_redaction import (
    display_name_for_path,
    redact_paths,
    redact_params_for_sharing,
)

# ---------------------------------------------------------------------------
# PNG privacy boundary
# ---------------------------------------------------------------------------
# ``save_image_with_metadata`` is the ONLY writer of PNG text chunks, and a PNG
# travels off this machine. The per-key ``add_text`` calls below are an
# allowlist — but the ``sushi_parameters`` blob written at the end of the
# function is NOT: it serializes the whole ``params`` dict, which is why it
# gets ``redact_params_for_sharing`` applied to it. These keys must NEVER
# appear in the chunks in unredacted form:
#
#   vae_path, vae_override_path, vae_override_source, text_encoder_path,
#   and any other value holding a filesystem path.
#
# ``vae_override_source`` in particular can hold a resolved absolute path.
#
# For the identity fields that legitimately go in (``vae_name``,
# ``vision_encoder_name``, ``upscaler_model``) the producer already emits a
# display name (see ``api/generation_utils.describe_vae_override``);
# ``_shareable`` is a second, local defence so that a future call site writing
# a path into one of those params cannot leak filesystem structure into every
# PNG it produces. It REDACTS (reduces a path to its name); it never raises and
# never fails a save.


def _shareable(value):
    """Reduce any absolute path inside a to-be-written text chunk to a name."""
    return redact_paths(value)

def save_image_with_metadata(
    image: Image.Image,
    params: Dict[str, Any],
    generation_type: str = "txt2img",
    model_info: Optional[Dict[str, Any]] = None,
    generation_id: Optional[int] = None
) -> str:
    """Save image with EXIF metadata

    Args:
        image: PIL Image to save
        params: Generation parameters
        generation_type: Type of generation (txt2img, img2img, inpaint)
        model_info: Model information (source, source_type, hash)
        generation_id: id returned by ``generation_status.start_generation()``
            for the generation that produced this image. REQUIRED for the
            ``effective_warnings`` chunk to be written: reading the accumulator
            without an id could stamp a concurrently-running generation's
            warnings into this PNG, and the PNG is the artifact that travels.
            Callers with no generation of their own (e.g. the training-preview
            saver) correctly leave it None and get no warnings chunk.
    """

    # Create outputs directory if not exists
    os.makedirs(settings.outputs_dir, exist_ok=True)
    print(f"Outputs directory: {settings.outputs_dir}")

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = params.get("seed", 0)
    filename = f"{generation_type}_{timestamp}_{seed}.png"
    filepath = os.path.join(settings.outputs_dir, filename)
    print(f"Saving image to: {filepath}")

    # Prepare metadata
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", params.get("prompt", ""))
    metadata.add_text("negative_prompt", params.get("negative_prompt", ""))
    metadata.add_text("steps", str(params.get("steps", settings.default_steps)))
    sampler = params.get("sampler", settings.default_sampler)
    metadata.add_text("sampler", sampler)
    metadata.add_text("cfg_scale", str(params.get("cfg_scale", settings.default_cfg_scale)))
    metadata.add_text("seed", str(seed))

    # Add ancestral_seed only for stochastic samplers (euler_a, dpm2_a, etc.)
    # These samplers add randomness at each step, so ancestral_seed controls that randomness
    stochastic_samplers = ["euler_a", "dpm2_a"]
    ancestral_seed = params.get("ancestral_seed", -1)
    if ancestral_seed != -1 and sampler in stochastic_samplers:
        metadata.add_text("ancestral_seed", str(ancestral_seed))

    metadata.add_text("width", str(params.get("width", settings.default_width)))
    metadata.add_text("height", str(params.get("height", settings.default_height)))
    metadata.add_text("generation_type", generation_type)

    # Post-decode options (only when applied, to keep metadata lean).
    color_flatten_strength = params.get("color_flatten_strength", 0) or 0
    if color_flatten_strength and int(color_flatten_strength) > 0:
        metadata.add_text("color_flatten_strength", str(int(color_flatten_strength)))
    if params.get("vae_drift_correction"):
        metadata.add_text("vae_drift_correction", "true")
    if params.get("flatten_in_loop"):
        metadata.add_text("flatten_in_loop", "true")
        metadata.add_text("flatten_in_loop_last_steps", str(int(params.get("flatten_in_loop_last_steps", 3) or 3)))
        metadata.add_text("flatten_in_loop_min_region", str(params.get("flatten_in_loop_min_region", 0.02)))

    # Add NAG (Normalized Attention Guidance) parameters
    nag_enable = params.get("nag_enable", False)
    if nag_enable:
        metadata.add_text("nag_enable", str(nag_enable))
        metadata.add_text("nag_scale", str(params.get("nag_scale", 5.0)))
        metadata.add_text("nag_tau", str(params.get("nag_tau", 3.5)))
        metadata.add_text("nag_alpha", str(params.get("nag_alpha", 0.25)))
        metadata.add_text("nag_sigma_end", str(params.get("nag_sigma_end", 3.0)))
        nag_negative_prompt = params.get("nag_negative_prompt", "")
        if nag_negative_prompt:
            metadata.add_text("nag_negative_prompt", nag_negative_prompt)

    # Add Advanced CFG parameters (can coexist with NAG)
    # Always save cfg_schedule parameters as they may be used even when type is "constant"
    cfg_schedule_type = params.get("cfg_schedule_type", "constant")
    metadata.add_text("cfg_schedule_type", cfg_schedule_type)

    # Save schedule range parameters
    metadata.add_text("cfg_schedule_min", str(params.get("cfg_schedule_min", 1.0)))
    if params.get("cfg_schedule_max") is not None:
        metadata.add_text("cfg_schedule_max", str(params["cfg_schedule_max"]))

    # Save power parameter for quadratic schedule
    if cfg_schedule_type == "quadratic" or params.get("cfg_schedule_power") is not None:
        metadata.add_text("cfg_schedule_power", str(params.get("cfg_schedule_power", 2.0)))

    # Save SNR-based adaptive CFG
    cfg_rescale_snr_alpha = params.get("cfg_rescale_snr_alpha", 0.0)
    if cfg_rescale_snr_alpha > 0:
        metadata.add_text("cfg_rescale_snr_alpha", str(cfg_rescale_snr_alpha))

    # Save dynamic thresholding parameters
    dynamic_threshold_percentile = params.get("dynamic_threshold_percentile", 0.0)
    if dynamic_threshold_percentile > 0:
        metadata.add_text("dynamic_threshold_percentile", str(dynamic_threshold_percentile))
        metadata.add_text("dynamic_threshold_mimic_scale", str(params.get("dynamic_threshold_mimic_scale", 7.0)))

    # Add generation-type specific parameters
    if generation_type in ("img2img", "inpaint"):
        if "denoising_strength" in params:
            metadata.add_text("denoising_strength", str(params["denoising_strength"]))
        if "img2img_fix_steps" in params:
            metadata.add_text("img2img_fix_steps", str(params["img2img_fix_steps"]))

    if generation_type == "upscale":
        upscaler_backend = params.get("upscaler_backend", "")
        if upscaler_backend:
            metadata.add_text("upscaler_backend", upscaler_backend)
        upscaler_model = params.get("upscaler_model")
        if upscaler_model:
            metadata.add_text("upscaler_model", _shareable(upscaler_model))
        upscaler_model_hash = params.get("upscaler_model_hash")
        if upscaler_model_hash:
            metadata.add_text("upscaler_model_hash", upscaler_model_hash)
        scale_factor = params.get("scale_factor")
        if scale_factor is not None:
            metadata.add_text("scale_factor", str(scale_factor))
        if upscaler_backend == "pil" and params.get("pil_resample"):
            metadata.add_text("pil_resample", params["pil_resample"])
        if upscaler_backend == "spandrel":
            metadata.add_text("tile_size", str(params.get("tile_size", 0)))
            metadata.add_text("tile_overlap", str(params.get("tile_overlap", 0)))
        if upscaler_backend == "rtx_vsr" and params.get("rtx_vsr_quality"):
            metadata.add_text("rtx_vsr_quality", params["rtx_vsr_quality"])
        if upscaler_backend == "diffusion":
            # prompt/negative_prompt/steps/cfg_scale/sampler/seed are already written
            # unconditionally above (they're present at the top level of `params`
            # for diffusion upscale). Only the diffusion-specific fields need adding.
            metadata.add_text("diffusion_denoising_strength", str(params.get("diffusion_denoising_strength", 0.3)))
            metadata.add_text("schedule_type", params.get("schedule_type", "uniform"))
            metadata.add_text("diffusion_pre_upscale_mode", params.get("diffusion_pre_upscale_mode", "pil"))
        if params.get("unsharp_enable"):
            metadata.add_text("unsharp_enable", "true")
            metadata.add_text("unsharp_radius", str(params.get("unsharp_radius", 2.0)))
            metadata.add_text("unsharp_percent", str(params.get("unsharp_percent", 100)))
            metadata.add_text("unsharp_threshold", str(params.get("unsharp_threshold", 3)))
        source_image_hash = params.get("source_image_hash")
        if source_image_hash:
            metadata.add_text("source_image_hash", source_image_hash)
        upscale_time = params.get("upscale_time")
        if upscale_time is not None:
            metadata.add_text("upscale_time", str(upscale_time))

    if generation_type == "inpaint":
        if "mask_blur" in params:
            metadata.add_text("mask_blur", str(params["mask_blur"]))
        # Note: inpaint_full_res and inpaint_full_res_padding are not implemented in backend
        # Commented out to avoid confusion
        # if "inpaint_full_res" in params:
        #     metadata.add_text("inpaint_full_res", str(params["inpaint_full_res"]))
        # if "inpaint_full_res_padding" in params:
        #     metadata.add_text("inpaint_full_res_padding", str(params["inpaint_full_res_padding"]))
        if "inpaint_fill_mode" in params:
            metadata.add_text("inpaint_fill_mode", params["inpaint_fill_mode"])
        if "inpaint_fill_strength" in params:
            metadata.add_text("inpaint_fill_strength", str(params["inpaint_fill_strength"]))
        if "inpaint_blur_strength" in params:
            metadata.add_text("inpaint_blur_strength", str(params["inpaint_blur_strength"]))

    # Add model information
    if model_info:
        # Extract filename from source path
        model_source = model_info.get("source", "")
        if model_source:
            model_filename = os.path.basename(model_source)
            metadata.add_text("model_name", model_filename)

        # Add model hash if available
        model_hash = model_info.get("model_hash", "")
        if model_hash:
            metadata.add_text("model_hash", model_hash)

    # Add U-Net quantization if used
    unet_quantization = params.get("unet_quantization")
    if unet_quantization and unet_quantization != "none":
        metadata.add_text("unet_quantization", unet_quantization)

    # Vision Encoder metadata.
    # Defensive gate: only record VE info when this generation actually used
    # reference images. The VE stays loaded ("sticky") across generations, so
    # stray vision_encoder_* params must not be written when no ref image was used.
    if params.get("ref_images"):
        ve_name = params.get("vision_encoder_name", "")
        ve_hash = params.get("vision_encoder_hash", "")
        if ve_name:
            metadata.add_text("vision_encoder_name", _shareable(ve_name))
        if ve_hash:
            metadata.add_text("vision_encoder_hash", ve_hash)

    # VAE identity. Recorded whenever known — the VAE always affects the decoded
    # output (embedded-in-checkpoint, shared store, env override, or model-own).
    # ``vae_name`` is a display name by construction; the redaction here also
    # covers the non-override branch of ``extract_vae_info``, whose
    # ``vae_source`` note is an absolute directory for some architectures.
    vae_name = params.get("vae_name", "")
    if vae_name:
        metadata.add_text("vae_name", _shareable(vae_name))
    vae_hash = params.get("vae_hash", "")
    if vae_hash:
        metadata.add_text("vae_hash", vae_hash)

    # LoRA weights + hashes (compact JSON). Written only when LoRAs are present.
    lora_meta = _build_lora_metadata(params.get("loras"))
    if lora_meta:
        import json as _json
        metadata.add_text("loras", _json.dumps(lora_meta, ensure_ascii=False))

    # Prompt chunking (non-default only)
    prompt_chunking_mode = params.get("prompt_chunking_mode", "a1111")
    if prompt_chunking_mode and prompt_chunking_mode != "a1111":
        metadata.add_text("prompt_chunking_mode", prompt_chunking_mode)
    max_prompt_chunks = params.get("max_prompt_chunks", 0)
    if max_prompt_chunks and int(max_prompt_chunks) > 0:
        metadata.add_text("max_prompt_chunks", str(max_prompt_chunks))

    # Attention backend. Recorded ALWAYS (including the "normal"/"conduit"
    # defaults): attention selection affects the produced pixels, so reproduction
    # needs the actual value that was requested for this generation, not just the
    # non-default cases. Note: this is the REQUESTED backend; the dispatcher may
    # transparently downgrade to native per-call when a backend is unavailable for
    # a given tensor shape (see core/attention/config.py:resolve_backend).
    attention_type = params.get("attention_type") or "normal"
    metadata.add_text("attention_type", attention_type)
    attention_impl = params.get("attention_impl") or "conduit"
    metadata.add_text("attention_impl", attention_impl)

    # Generation timing (informational, not reproducibility-affecting). Written
    # whenever present — total wall time is always recorded by the endpoint; the
    # phase breakdown is present only for instrumented architectures/paths.
    for _tkey in ("generation_time", "time_text_encode", "time_denoise", "time_vae_decode"):
        if params.get(_tkey) is not None:
            metadata.add_text(_tkey, str(params[_tkey]))

    # Text Encoder quantization (non-default only)
    text_encoder_quantization = params.get("text_encoder_quantization")
    if text_encoder_quantization and text_encoder_quantization != "none":
        metadata.add_text("text_encoder_quantization", text_encoder_quantization)

    # SDXL micro-conditioning original_size override (non-default only)
    original_size_w = params.get("original_size_w", 0)
    original_size_h = params.get("original_size_h", 0)
    original_size_scale = params.get("original_size_scale", 1.0)
    if original_size_w and int(original_size_w) > 0:
        metadata.add_text("original_size_w", str(original_size_w))
    if original_size_h and int(original_size_h) > 0:
        metadata.add_text("original_size_h", str(original_size_h))
    if original_size_scale is not None and float(original_size_scale) != 1.0:
        metadata.add_text("original_size_scale", str(original_size_scale))

    # TIPO prompt upsampling (flag only; effective prompt is already saved as "prompt")
    if params.get("use_tipo", False):
        metadata.add_text("use_tipo", "True")

    # Spectrum forecasting (write full family only when enabled)
    if params.get("spectrum_enable", False):
        metadata.add_text("spectrum_enable", "True")
        for k in (
            "spectrum_w", "spectrum_w_decay", "spectrum_delta_cap", "spectrum_m", "spectrum_lam", "spectrum_warmup_steps",
            "spectrum_window_size", "spectrum_flex_window", "spectrum_tail",
            "spectrum_feature_mode", "spectrum_cache_branch", "spectrum_max_cache",
        ):
            if k in params:
                metadata.add_text(k, str(params[k]))

    # FBCache (write full family only when enabled)
    if params.get("fbcache_enable", False):
        metadata.add_text("fbcache_enable", "True")
        for k in ("fbcache_threshold", "fbcache_warmup_steps", "fbcache_cache_branch"):
            if k in params:
                metadata.add_text(k, str(params[k]))

    # Effective warnings: feature-degradation notices recorded during this
    # generation (explains any divergence between the requested params above and
    # what actually ran). Requested values stay unchanged; this only annotates.
    try:
        import json
        from api.generation_status import get_warnings
        _warnings = get_warnings(generation_id) if generation_id is not None else []
        if _warnings:
            # Warning MESSAGES are backend-generated and can quote the path of
            # whatever failed to load, so they get the same treatment as every
            # other chunk. Redacted structurally (before serialization) rather
            # than on the JSON string, which would corrupt its escaping.
            metadata.add_text(
                "effective_warnings",
                json.dumps(redact_params_for_sharing(_warnings))
            )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Full JSON parameter blob (reproducibility): the individual add_text()
    # calls above are a hand-maintained per-key whitelist kept for
    # human-readable/back-compat metadata (external tools, quick inspection).
    # Any parameter NOT explicitly whitelisted there (e.g. outpaint
    # placement/continuity knobs, region prompts, seam/boundary controls, or
    # any future param) was silently dropped, making the saved image
    # unreproducible from its own metadata. Write the COMPLETE params dict as
    # a single JSON chunk so every param present today -- and any added in
    # the future -- is captured with zero per-param maintenance.
    #
    # Reuses prepare_params_for_db(), the SAME sanitizer already used for the
    # DB `parameters` column, so raw image/mask/reference pixel data
    # (controlnet_images[].image, style_transfer(s)[].image, ref_images) is
    # replaced with stable hashes instead of being embedded as base64 (which
    # would bloat the PNG and risks corrupting the tEXt chunk). This also
    # keeps the PNG blob and the DB record in parity by construction.
    try:
        import json
        from api.generation_utils import prepare_params_for_db
        _full_params = prepare_params_for_db(params, calculate_image_hash)
        # prepare_params_for_db hashes the DECODED image fields
        # (controlnet_images[].image, style_transfer(s)[].image, ref_images) but
        # NOT the raw `controlnets`/`style_transfers` request configs, whose
        # entries still carry a full base64 PNG under `image_base64` (txt2img
        # sends both keys). Drop those base64 strings here so the PNG tEXt chunk
        # can't be bloated by ~0.5-2MB per control/reference image -- the hashed
        # controlnet_images + model_path/strength/mode keep reproducibility.
        for _list_key in ("controlnets", "style_transfers"):
            _lst = _full_params.get(_list_key)
            if isinstance(_lst, list):
                _full_params[_list_key] = [
                    {k: v for k, v in e.items() if k != "image_base64"}
                    if isinstance(e, dict) else e
                    for e in _lst
                ]
        if isinstance(_full_params.get("style_transfer"), dict):
            _full_params["style_transfer"] = {
                k: v for k, v in _full_params["style_transfer"].items() if k != "image_base64"
            }
        # PRIVACY: this blob is NOT an allowlist -- it carries every key of
        # `params`, including the local-only ones the per-key chunks above
        # deliberately omit (`vae_path`, `vae_override_path`,
        # `vae_override_source`, component/LoRA paths). Those are absolute
        # filesystem paths, i.e. personal environment information, and a PNG is
        # a shareable file. Reduce every path to its name here (prompts and
        # other user-typed text are passed through verbatim); the DB row is
        # written from the unmodified `params` and keeps the full paths for
        # local restore, so nothing local is lost.
        _full_params = redact_params_for_sharing(_full_params)
        # default=str is a defensive fallback only (e.g. a stray PIL Image or
        # Enum/Path slipping past the sanitizer serializes to its short repr,
        # not raw pixel data).
        metadata.add_text(
            "sushi_parameters",
            json.dumps(_full_params, default=str, ensure_ascii=False)
        )
    except Exception as e:
        print(f"[Metadata] Failed to write full sushi_parameters JSON blob: {e}")

    # Save image
    try:
        image.save(filepath, pnginfo=metadata)
        print(f"Image saved successfully: {filename}")

        # Verify file exists
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"File exists, size: {file_size} bytes")
        else:
            print(f"ERROR: File was not created at {filepath}")
    except Exception as e:
        print(f"ERROR saving image: {e}")
        raise

    return filename

def create_thumbnail(image_path: str, size: tuple = (256, 256)) -> str:
    """Create thumbnail from image (PNG + WebP versions)

    Creates both PNG (for compatibility) and WebP (for transfer reduction) thumbnails.
    WebP version is ~80% smaller than PNG on average.
    """
    os.makedirs(settings.thumbnails_dir, exist_ok=True)

    image = Image.open(image_path)
    # Convert to RGB if RGBA (WebP quality mode requires RGB)
    if image.mode == 'RGBA':
        # Create white background and paste image on it
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    image.thumbnail(size, Image.Resampling.LANCZOS)

    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]

    # Save PNG version (for compatibility)
    thumb_path_png = os.path.join(settings.thumbnails_dir, f"{base_name}.png")
    image.save(thumb_path_png, format='PNG')

    # Save WebP version (for transfer reduction, ~80% smaller)
    thumb_path_webp = os.path.join(settings.thumbnails_dir, f"{base_name}.webp")
    image.save(thumb_path_webp, format='WEBP', quality=85)

    return thumb_path_png

def extract_metadata_from_image(image_path: str) -> Dict[str, Any]:
    """Extract metadata from PNG image"""
    image = Image.open(image_path)
    metadata = {}

    if hasattr(image, 'text'):
        for key, value in image.text.items():
            metadata[key] = value

    return metadata

def calculate_image_hash(image: Image.Image) -> str:
    """Calculate SHA256 hash of image"""
    # Convert image to bytes
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()

    # Calculate hash
    sha256_hash = hashlib.sha256(image_bytes).hexdigest()
    return sha256_hash

def encode_mask_to_base64(mask_image: Image.Image) -> str:
    """Encode mask image to base64 string"""
    buffer = BytesIO()
    mask_image.save(buffer, format='PNG')
    mask_bytes = buffer.getvalue()
    return base64.b64encode(mask_bytes).decode('utf-8')

def _build_lora_metadata(lora_configs) -> list:
    """Build a compact ``[{name, weight, hash?}]`` list for the PNG ``loras`` chunk.

    ``name`` is the LoRA filename, ``weight`` its applied strength. ``hash`` is added
    only when the file resolves and its (cached) hash is available — hashing reuses
    the shared mtime/size-invalidated hash cache, so it is computed at most once per
    file, never as a new per-generation cost.
    """
    if not lora_configs:
        return []
    result = []
    for lora in lora_configs:
        if not isinstance(lora, dict):
            continue
        path = lora.get("path", "")
        if not path:
            continue
        entry = {
            # Name + content hash, never the path (the same rule as model_name /
            # vae_name). display_name_for_path also disambiguates a LoRA whose
            # filename is a generated one (``checkpoint.safetensors``).
            "name": display_name_for_path(path),
            "weight": lora.get("strength", 1.0),
        }
        try:
            from core.extensions.lora_manager import lora_manager
            from utils.hash_cache import get_cached_file_hash
            resolved = lora_manager._resolve_lora_path(path)
            if resolved is not None:
                h = get_cached_file_hash(str(resolved))
                if h:
                    entry["hash"] = h
        except Exception as e:
            print(f"[LoRA Metadata] Hash resolution skipped for {path}: {e}")
        result.append(entry)
    return result


def extract_lora_names(lora_configs: list) -> str:
    """Extract comma-separated LoRA filenames from configs"""
    if not lora_configs:
        return ""

    lora_names = []
    for lora in lora_configs:
        path = lora.get('path', '')
        if path:
            # Extract filename without extension
            filename = os.path.basename(path)
            lora_names.append(filename)

    return ", ".join(lora_names)

def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Calculate hash of a file

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (sha256, sha1, md5)

    Returns:
        Hexadecimal hash string
    """
    if not os.path.exists(file_path):
        return ""

    hash_obj = hashlib.new(algorithm)

    # Read file in chunks to handle large files
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()
