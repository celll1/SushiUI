from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import Response
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import os
import sys
import subprocess
from PIL import Image
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor

from database import get_gallery_db, get_datasets_db, get_training_db, get_db  # Legacy
from database.models import GeneratedImage, UserSettings, Dataset, DatasetItem, DatasetCaption, TagDictionary, TrainingRun, TrainingCheckpoint, TrainingSample, TrainingPreset, TaggerTrainingRun, TaggerTrainingMetrics
from core.pipeline import pipeline_manager
from core.utils.taesd import taesd_manager
from core.extensions.lora_manager import lora_manager
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
from utils import save_image_with_metadata, create_thumbnail, calculate_image_hash, encode_mask_to_base64, extract_lora_names
from config.settings import settings
from api.websocket import manager
from auth import create_access_token, verify_credentials, require_auth
from api.generation_utils import (
    process_controlnet_configs,
    create_progress_callback_factory,
    create_db_image_record,
    load_loras_for_generation,
    prepare_params_for_db,
    create_lora_step_callback,
    extract_model_info,
    extract_vision_encoder_info,
    sanitize_params_for_logging,
    set_prompt_chunking_settings,
    calculate_generation_metadata
)
from api.error_handlers import (
    GenerationError,
    ModelError,
    NotFoundError,
    ValidationError as CustomValidationError
)

router = APIRouter()

# Thread pool for running blocking operations
executor = ThreadPoolExecutor(max_workers=1)

# Cache for model list (to avoid re-scanning on every API call)
_models_cache: Optional[Dict[str, Any]] = None
_models_cache_timestamp: float = 0

# Cache for TensorBoard EventAccumulators to avoid re-reading event files on every request
# Key: (run_id, event_file_path), Value: (EventAccumulator, last_modified_time)
_event_accumulator_cache: Dict[tuple, tuple] = {}

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

class AddTagRequest(BaseModel):
    tag: str
    category: str
    count: int = 1

class GenerationParams(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    steps: int = 20
    cfg_scale: float = 7.0
    sampler: str = "euler"
    schedule_type: str = "uniform"
    seed: int = -1
    ancestral_seed: int = -1  # Seed for stochastic samplers (Euler a, DPM2 a, etc.). -1 = use main seed
    width: int = 512
    height: int = 512
    model: str = ""
    loras: Optional[List[LoRAConfig]] = []
    controlnets: Optional[List[ControlNetConfig]] = []
    prompt_chunking_mode: str = "a1111"  # Options: a1111, sd_scripts, nobos
    max_prompt_chunks: int = 0  # 0 = unlimited, 1-4 = limit chunks
    developer_mode: bool = False  # Enable CFG metrics visualization
    # Dynamic CFG scheduling
    cfg_schedule_type: str = "constant"  # constant, linear, quadratic, cosine, snr_based
    cfg_schedule_min: float = 1.0  # Minimum CFG at end of generation
    cfg_schedule_max: Optional[float] = None  # Maximum CFG at start (None = use cfg_scale)
    cfg_schedule_power: float = 2.0  # Power for quadratic schedule
    cfg_rescale_snr_alpha: float = 0.0  # SNR-based adaptive CFG (0.0 = disabled, 0.1-0.5 typical)
    # Dynamic thresholding
    dynamic_threshold_percentile: float = 0.0  # 0.0 = disabled, 99.5 = typical
    dynamic_threshold_mimic_scale: float = 1.0  # Clamp value for static threshold
    # NAG (Normalized Attention Guidance)
    nag_enable: bool = False  # Enable NAG
    nag_scale: float = 5.0  # NAG extrapolation scale (3-7 typical)
    nag_tau: float = 3.5  # NAG normalization threshold (2.5-3.5 typical)
    nag_alpha: float = 0.25  # NAG blending factor (0.25-0.5 typical)
    nag_sigma_end: float = 0.0  # Sigma threshold to disable NAG (0.0 = always enabled)
    nag_negative_prompt: Optional[str] = ""  # Separate negative prompt for NAG (empty = use main negative prompt)
    # Attention processor type
    attention_type: str = "normal"  # "normal", "sage", "flash"
    # U-Net Quantization
    unet_quantization: Optional[str] = None  # None, "int8", "fp8", "int4", "nf4"
    # Text Encoder Quantization (Z-Image only)
    text_encoder_quantization: Optional[str] = None  # None, "fp8_e4m3fn", "fp8_e5m2", "uint8", "uint4"
    # torch.compile optimization
    use_torch_compile: bool = False  # Enable torch.compile for U-Net (1.3-2x speedup)
    # TIPO (prompt upsampling)
    use_tipo: bool = False  # Enable TIPO prompt upsampling
    tipo_config: Optional[Dict] = None  # TIPO configuration (model, lengths, etc.)
    # Preview mode
    preview_predicted_x0: bool = False  # Show predicted x0 instead of current latent in preview

class Txt2ImgRequest(GenerationParams):
    pass

class Img2ImgRequest(GenerationParams):
    denoising_strength: float = 0.75

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
    attention_type: str = Form("normal"),
    unet_quantization: Optional[str] = Form(None),
    text_encoder_quantization: Optional[str] = Form(None),
    use_torch_compile: bool = Form(False),
    use_tipo: bool = Form(False),
    tipo_config: str = Form("{}"),  # JSON string of TIPO config
    preview_predicted_x0: bool = Form(False),  # Show predicted x0 in preview instead of current latent
    enable_block_swap: bool = Form(False),
    blocks_to_swap: int = Form(20),
    use_pinned_memory: bool = Form(False),
    ref_images: List[UploadFile] = File(default=[]),  # FLUX.2 Image Edit / Vision Encoder reference images
    vision_encoder_path: Optional[str] = Form(None),  # Path to SigLIP2 vision encoder safetensors
    db: Session = Depends(get_gallery_db)
):
    """Generate image from text"""
    lora_configs = []
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
        ref_image_list = []
        if ref_images:
            from PIL import Image
            import io
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
            "unet_quantization": unet_quantization,
            "text_encoder_quantization": text_encoder_quantization,
            "use_torch_compile": use_torch_compile,
            "enable_block_swap": enable_block_swap,
            "blocks_to_swap": blocks_to_swap,
            "use_pinned_memory": use_pinned_memory,
            "ref_images": ref_image_list,  # FLUX.2 Image Edit reference images
        }

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
            # Direct image upload via multipart
            import io
            for uploaded_file in controlnet_images:
                image_data = await uploaded_file.read()
                cn_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                processed_controlnet_images.append(cn_image)

        # Also process base64 images from controlnets JSON
        if controlnet_configs:
            base64_images = process_controlnet_configs(
                controlnet_configs,
                generation_type="txt2img"
            )
            processed_controlnet_images.extend(base64_images)

        params["controlnet_images"] = processed_controlnet_images
        params["controlnets"] = controlnet_configs

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

        # Progress callback to send updates via WebSocket
        progress_callback = create_progress_callback_factory(
            taesd_manager,
            manager,
            is_sdxl,
            is_zimage,
            is_deus,
            is_zimage_sdxl_vae,
            is_flux2,
            image_width=params.get("width"),
            image_height=params.get("height"),
            preview_predicted_x0=preview_predicted_x0,
            preview_enabled=params.get("preview_enabled", True),
            preview_interval=params.get("preview_interval", 4)
        )

        # Create step callback for LoRA step range if needed
        step_callback = None
        if has_step_range_loras:
            step_callback = create_lora_step_callback(
                lora_manager,
                pipeline_manager.txt2img_pipeline,
                params.get("steps", 20)
            )

        # Run generation in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        image, actual_seed, actual_ancestral_seed = await loop.run_in_executor(
            executor,
            lambda: pipeline_manager.generate_txt2img(params, progress_callback=progress_callback, step_callback=step_callback)
        )

        # Update params with actual seeds
        params["seed"] = actual_seed
        params["ancestral_seed"] = actual_ancestral_seed

        # Add Vision Encoder info to params for PNG metadata and DB storage
        ve_name, ve_hash = extract_vision_encoder_info(pipeline_manager)
        if ve_name:
            params["vision_encoder_name"] = ve_name
        if ve_hash:
            params["vision_encoder_hash"] = ve_hash

        # Save image with metadata (include model info)
        filename = save_image_with_metadata(
            image,
            params,
            "txt2img",
            model_info=pipeline_manager.current_model_info
        )

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

        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed}

    except GenerationError:
        # Re-raise custom errors as-is
        raise
    except Exception as e:
        # Wrap unexpected errors in GenerationError
        import traceback
        error_detail = traceback.format_exc()
        raise GenerationError(
            "Text-to-image generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )
    finally:
        # Unload LoRAs after generation
        if lora_configs and pipeline_manager.txt2img_pipeline:
            pipeline_manager.txt2img_pipeline = lora_manager.unload_loras(pipeline_manager.txt2img_pipeline)

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
    attention_type: str = Form("normal"),
    unet_quantization: Optional[str] = Form(None),
    text_encoder_quantization: Optional[str] = Form(None),
    use_torch_compile: bool = Form(False),
    enable_block_swap: bool = Form(False),
    blocks_to_swap: int = Form(22),
    use_pinned_memory: bool = Form(False),
    use_tipo: bool = Form(False),
    tipo_config: str = Form("{}"),  # JSON string of TIPO config
    preview_predicted_x0: bool = Form(False),  # Show predicted x0 in preview instead of current latent
    vision_encoder_path: Optional[str] = Form(None),  # Path to SigLIP2 vision encoder safetensors
    image: UploadFile = File(...),
    ref_images: List[UploadFile] = File(default=[]),  # FLUX.2 Image Edit / Vision Encoder reference images
    db: Session = Depends(get_gallery_db)
):
    """Generate image from image"""
    lora_configs = []
    try:
        # Reset cancellation flag before starting new generation
        pipeline_manager.reset_cancel_flag()

        # Load input image
        image_data = await image.read()
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Parse LoRA configs
        import json
        lora_configs = json.loads(loras) if loras else []

        # Parse ControlNet configs
        controlnet_configs = json.loads(controlnets) if controlnets else []
        controlnet_images = process_controlnet_configs(
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

        # Generate image
        params = {
            "prompt": prompt,
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
            "unet_quantization": unet_quantization,
            "text_encoder_quantization": text_encoder_quantization,
            "use_torch_compile": use_torch_compile,
            "enable_block_swap": enable_block_swap,
            "blocks_to_swap": blocks_to_swap,
            "use_pinned_memory": use_pinned_memory,
            "ref_images": ref_image_list,  # FLUX.2 Image Edit reference images
        }
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

        # Progress callback to send updates via WebSocket
        progress_callback = create_progress_callback_factory(
            taesd_manager,
            manager,
            is_sdxl,
            is_zimage,
            is_deus,
            is_zimage_sdxl_vae,
            is_flux2,
            img2img_fix_steps=img2img_fix_steps,
            steps=steps,
            image_width=width,
            image_height=height,
            preview_predicted_x0=preview_predicted_x0,
            preview_enabled=params.get("preview_enabled", True),
            preview_interval=params.get("preview_interval", 4)
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

        # Run generation in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result_image, actual_seed, actual_ancestral_seed = await loop.run_in_executor(
            executor,
            lambda: pipeline_manager.generate_img2img(params, init_image, progress_callback=progress_callback, step_callback=step_callback)
        )

        # Update params with actual seeds
        params["seed"] = actual_seed
        params["ancestral_seed"] = actual_ancestral_seed

        # Add Vision Encoder info to params for PNG metadata and DB storage
        ve_name, ve_hash = extract_vision_encoder_info(pipeline_manager)
        if ve_name:
            params["vision_encoder_name"] = ve_name
        if ve_hash:
            params["vision_encoder_hash"] = ve_hash

        # Save image with metadata (include model info)
        filename = save_image_with_metadata(
            result_image,
            params,
            "img2img",
            model_info=pipeline_manager.current_model_info
        )
        image_path = os.path.join(settings.outputs_dir, filename)
        create_thumbnail(image_path)

        # Calculate metadata
        metadata = calculate_generation_metadata(
            result_image,
            lora_configs,
            extract_lora_names,
            calculate_image_hash,
            source_image=init_image
        )

        # Remove image objects from params before saving to DB and calculate ControlNet hashes
        params_for_db = prepare_params_for_db(params, calculate_image_hash)

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

        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed}

    except GenerationError:
        # Re-raise custom errors as-is
        raise
    except Exception as e:
        # Wrap unexpected errors in GenerationError
        import traceback
        error_detail = traceback.format_exc()
        raise GenerationError(
            "Image-to-image generation failed",
            detail=f"{str(e)}\n\n{error_detail}"
        )
    finally:
        # Unload LoRAs after generation
        if lora_configs and pipeline_manager.img2img_pipeline:
            pipeline_manager.img2img_pipeline = lora_manager.unload_loras(pipeline_manager.img2img_pipeline)

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
    attention_type: str = Form("normal"),
    unet_quantization: Optional[str] = Form(None),
    text_encoder_quantization: Optional[str] = Form(None),
    use_torch_compile: bool = Form(False),
    enable_block_swap: bool = Form(False),
    blocks_to_swap: int = Form(22),
    use_pinned_memory: bool = Form(False),
    use_tipo: bool = Form(False),
    tipo_config: str = Form("{}"),  # JSON string of TIPO config
    preview_predicted_x0: bool = Form(False),  # Show predicted x0 in preview instead of current latent
    vision_encoder_path: Optional[str] = Form(None),  # Path to SigLIP2 vision encoder safetensors
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    ref_images: List[UploadFile] = File(default=[]),  # FLUX.2 Image Edit / Vision Encoder reference images
    db: Session = Depends(get_gallery_db)
):
    """Generate inpainted image"""
    lora_configs = []
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
        controlnet_images = process_controlnet_configs(
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

        # Generate image
        params = {
            "prompt": prompt,
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
            "loras": lora_configs,  # FLUX.2 needs this in params
            "controlnet_images": controlnet_images,
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
            "unet_quantization": unet_quantization,
            "text_encoder_quantization": text_encoder_quantization,
            "use_torch_compile": use_torch_compile,
            "enable_block_swap": enable_block_swap,
            "blocks_to_swap": blocks_to_swap,
            "use_pinned_memory": use_pinned_memory,
            "ref_images": ref_image_list,  # FLUX.2 Image Edit reference images
        }
        print(f"inpaint generation params: {sanitize_params_for_logging(params)}")

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

        # Progress callback to send updates via WebSocket
        progress_callback = create_progress_callback_factory(
            taesd_manager,
            manager,
            is_sdxl,
            is_zimage,
            is_deus,
            is_zimage_sdxl_vae,
            is_flux2,
            img2img_fix_steps=img2img_fix_steps,
            steps=steps,
            image_width=width,
            image_height=height,
            preview_predicted_x0=preview_predicted_x0,
            preview_enabled=params.get("preview_enabled", True),
            preview_interval=params.get("preview_interval", 4)
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

        # Run generation in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result_image, actual_seed, actual_ancestral_seed = await loop.run_in_executor(
            executor,
            lambda: pipeline_manager.generate_inpaint(params, init_image, mask_image, progress_callback=progress_callback, step_callback=step_callback)
        )

        # Update params with actual seeds
        params["seed"] = actual_seed
        params["ancestral_seed"] = actual_ancestral_seed

        # Add Vision Encoder info to params for PNG metadata and DB storage
        ve_name, ve_hash = extract_vision_encoder_info(pipeline_manager)
        if ve_name:
            params["vision_encoder_name"] = ve_name
        if ve_hash:
            params["vision_encoder_hash"] = ve_hash

        # Save image with metadata (include model info)
        filename = save_image_with_metadata(
            result_image,
            params,
            "inpaint",
            model_info=pipeline_manager.current_model_info
        )
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

        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed}

    except GenerationError:
        # Re-raise custom errors as-is
        raise
    except Exception as e:
        # Wrap unexpected errors in GenerationError
        import traceback
        error_detail = traceback.format_exc()
        raise GenerationError(
            "Inpaint generation failed",
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

    # Text search in prompt
    if search:
        query = query.filter(GeneratedImage.prompt.contains(search))

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

    return {
        "images": [img.to_dict() for img in images],
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

@router.delete("/images/{image_id}")
async def delete_image(image_id: int, db: Session = Depends(get_gallery_db)):
    """Delete an image"""
    image = db.query(GeneratedImage).filter(GeneratedImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete files
    image_path = os.path.join(settings.outputs_dir, image.filename)
    thumb_path = os.path.join(settings.thumbnails_dir, image.filename)

    if os.path.exists(image_path):
        os.remove(image_path)
    if os.path.exists(thumb_path):
        os.remove(thumb_path)

    db.delete(image)
    db.commit()

    return {"success": True}

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
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)

            # Detect model architecture (sd15, sdxl, zimage)
            architecture = ModelLoader.detect_model_type(item_path)

            if os.path.isdir(item_path):
                # Only include directories that are valid diffusers model directories
                if not ModelLoader.is_valid_diffusers_directory(item_path):
                    continue
                models.append({
                    "name": item,
                    "path": item_path,
                    "type": "diffusers",
                    "source_type": "diffusers",
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

@router.post("/models/load")
async def load_model(
    source_type: str = Form(...),
    source: str = Form(...),
    revision: Optional[str] = Form(None)
):
    """Load a model from various sources (fp16 by default)"""
    try:
        kwargs = {}
        if revision:
            kwargs["revision"] = revision

        pipeline_manager.load_model(
            source_type=source_type,
            source=source,
            pipeline_type="txt2img",
            **kwargs
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
        return {
            "loaded": True,
            "model_info": pipeline_manager.current_model_info
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
            pipeline_manager.is_flux2_model
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
        result = {
            "loras": [
                {"path": lora, "name": os.path.basename(lora)}
                for lora in loras
            ]
        }
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
    info = lora_manager.get_lora_info(lora_name)
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
async def restart_backend():
    """Restart the backend server"""
    try:
        import threading
        import time
        import signal

        def do_restart():
            try:
                time.sleep(1)  # Wait for response to be sent

                # On Windows, we need to use a different approach
                if sys.platform == "win32":
                    # Get the path to Python executable and main.py
                    python_exe = sys.executable
                    backend_dir = os.path.dirname(os.path.dirname(__file__))
                    main_path = os.path.join(backend_dir, "main.py")

                    print(f"Restarting backend: {python_exe} {main_path}")
                    print(f"Working directory: {backend_dir}")

                    # Start a new process
                    subprocess.Popen([python_exe, main_path],
                                   cwd=backend_dir,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)

                    # Exit current process
                    time.sleep(0.5)
                    os._exit(0)
                else:
                    # Unix-like systems
                    python_exe = sys.executable
                    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
                    os.execv(python_exe, [python_exe, main_path])
            except Exception as e:
                import traceback
                print(f"Error in do_restart: {str(e)}")
                print(traceback.format_exc())

        threading.Thread(target=do_restart, daemon=True).start()

        return {"success": True, "message": "Backend restart scheduled"}
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Restart backend error: {error_detail}")
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

class SigLIP2LoadRequest(BaseModel):
    checkpoint_path: str
    vision_encoder_path: str = ""
    vocab_path: str = ""
    lora_rank: int = 32
    lora_alpha: float = 16.0

class SigLIP2PredictRequest(BaseModel):
    image_base64: str
    threshold: float = 0.35

class SigLIP2MergeLoRARequest(BaseModel):
    output_path: str

class SigLIP2ExportONNXRequest(BaseModel):
    output_path: str
    max_num_patches: int = 256


@router.post("/tagger/siglip2/load")
async def siglip2_load(request: SigLIP2LoadRequest):
    """Load a SigLIP2 tagger checkpoint (full or LoRA, auto-detected)."""
    try:
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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tagger/siglip2/predict")
async def siglip2_predict(request: SigLIP2PredictRequest):
    """Run inference with the loaded SigLIP2 model."""
    try:
        import base64
        mgr = get_siglip2_inference_manager()
        image_bytes = base64.b64decode(request.image_base64)
        result = mgr.predict(image_bytes=image_bytes, threshold=request.threshold)
        return result
    except Exception as e:
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
        )
        return {"saved_path": onnx_path, "vocab_path": vocab_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

def update_dataset_statistics(dataset: Dataset, db: Session):
    """Update dataset statistics by counting items and captions"""
    item_ids_subq = db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset.id)
    total_items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset.id).count()
    total_captions = db.query(DatasetCaption).filter(
        DatasetCaption.item_id.in_(item_ids_subq)
    ).count()
    # Count items that have at least one tags-format caption
    total_tags = db.query(DatasetCaption).filter(
        DatasetCaption.item_id.in_(item_ids_subq),
        DatasetCaption.is_tags_format == True
    ).count()

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
    """List all datasets"""
    try:
        datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()

        # Update statistics for each dataset before returning
        for dataset in datasets:
            update_dataset_statistics(dataset, db)

        return {"datasets": [d.to_dict() for d in datasets], "total": len(datasets)}
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
        total_captions = db.query(DatasetCaption).filter(
            DatasetCaption.item_id.in_(
                db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset.id)
            )
        ).count()

        dataset.total_items = total_items
        dataset.total_captions = total_captions
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
    db: Session = Depends(get_datasets_db)
):
    """Scan dataset directory and register images/captions"""
    import os
    from PIL import Image
    import warnings

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

    # Supported image extensions
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    caption_exts = {".txt", ".json"}

    # Load taglist for caption format detection (once at start)
    from utils.taglist_loader import load_all_tags
    from utils.caption_detector import classify_field, scan_json_fields
    print(f"[Dataset Scan] Loading taglist for format detection...")
    taglist = load_all_tags(settings.root_dir)
    taglist_cache.initialize(settings.root_dir)
    print(f"[Dataset Scan] Loaded {len(taglist)} tags for format detection")

    def _build_tag_data_json(content: str) -> str:
        """Build tag_data JSON string from comma-separated tag content."""
        import json as _json
        tags = [t.strip() for t in content.split(",") if t.strip()]
        if not tags:
            return "[]"
        cats = taglist_cache.get_categories_batch(tags)
        return _json.dumps(
            [{"tag": t, "category": cats.get(t, "General")} for t in tags],
            ensure_ascii=False,
        )

    # Pre-scan with 2-pass scanner: detect suffix captions + count images in one pass
    from utils.dataset_scanner import scan_directory_structure
    import asyncio
    print(f"[Dataset Scan] Pre-scanning directory structure...")
    loop = asyncio.get_event_loop()
    pre_scan_groups = await loop.run_in_executor(
        None,
        lambda: scan_directory_structure(
            dir_path=dataset.path,
            recursive=dataset.recursive,
            max_depth=dataset.max_depth if dataset.max_depth else None,
            reference_suffixes=dataset.reference_suffixes or [],
            target_suffixes=dataset.target_suffixes or [],
        )
    )

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

                if ext_lower in image_exts:
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

            try:
                # Read image metadata (with warning suppression for corrupt EXIF)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        with Image.open(image_path) as img:
                            width, height = img.size
                except Exception as img_error:
                    # Skip images that can't be opened (corrupt, unsupported format, etc.)
                    print(f"[Dataset Scan] Skipping corrupt/unsupported image {image_path}: {img_error}")
                    files_processed += 1
                    if files_processed % 10 == 0 or total_images < 100:
                        manager.send_progress_sync(
                            files_processed,
                            total_steps,
                            f"Scanning: {files_processed}/{total_images} images | Found: {items_found} new items, {captions_found} captions"
                        )
                    continue

                file_size = os.path.getsize(image_path)

                # --- Path-based dedup (replaces SHA256 hash + per-item DB query) ---
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

                    if not any_caption_updated:
                        # No changes — skip entirely
                        if files_processed % 10 == 0 or total_images < 100:
                            manager.send_progress_sync(
                                files_processed,
                                total_steps,
                                f"Scanning: {files_processed}/{total_images} images | Found: {items_found} new, {captions_updated} updated"
                            )
                        continue

                    # Caption files updated — re-process captions for this existing item
                    item_id_for_captions = existing_item_id
                    # (fall through to caption processing below)
                    if files_processed % 10 == 0 or total_images < 100:
                        manager.send_progress_sync(
                            files_processed,
                            total_steps,
                            f"Scanning: {files_processed}/{total_images} images | Found: {items_found} new, {captions_updated} updated"
                        )
                else:
                    # New image — register it
                    # Build related_images for reference mode
                    related_images_data = {}
                    if use_reference_mode and reference_images:
                        related_images_data["reference"] = reference_images
                        print(f"[Dataset Scan] Group '{base_name}': {len(reference_images)} reference image(s)")

                    item = DatasetItem(
                        dataset_id=dataset_id,
                        item_type="reference" if use_reference_mode else "single",
                        base_name=base_name,
                        image_path=image_path,
                        width=width,
                        height=height,
                        file_size=file_size,
                        image_hash=None,  # SHA256 no longer computed at scan time
                        related_images=related_images_data if related_images_data else None
                    )
                    db.add(item)
                    db.flush()  # Get item.id
                    item_id_for_captions = item.id
                    items_found += 1
                    files_processed += 1

                    if files_processed % 10 == 0 or total_images < 100:
                        manager.send_progress_sync(
                            files_processed,
                            total_steps,
                            f"Scanning: {files_processed}/{total_images} images | Found: {items_found} new, {captions_updated} updated"
                        )

                # Process captions (TXT/JSON files) — for both new and updated items
                # Use item_id_for_captions (set above for both new and existing items)
                for caption_path in caption_files:
                    try:
                        _, ext = os.path.splitext(caption_path)
                        ext_lower = ext.lower()

                        if ext_lower == '.txt':
                            # TXT file: Read content and detect format
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    # Detect format
                                    field_category, is_tags_format, match_rate = classify_field("tags", content, taglist)

                                    # Determine caption_type based on detected format
                                    detected_caption_type = "tags" if is_tags_format else "natural_language"

                                    # Check if caption of this type already exists
                                    existing_cap = db.query(DatasetCaption).filter(
                                        DatasetCaption.item_id == item_id_for_captions,
                                        DatasetCaption.caption_type == detected_caption_type
                                    ).first()

                                    if existing_cap:
                                        # Update existing
                                        existing_cap.content = content
                                        existing_cap.field_category = field_category
                                        existing_cap.is_tags_format = is_tags_format
                                        existing_cap.tag_match_rate = match_rate
                                        existing_cap.source = "file"
                                        existing_cap.source_field = detected_caption_type
                                        if is_tags_format:
                                            existing_cap.tag_data = _build_tag_data_json(content)
                                        existing_cap.updated_at = datetime.utcnow()
                                        captions_updated += 1
                                    else:
                                        # Create new
                                        caption = DatasetCaption(
                                            item_id=item_id_for_captions,
                                            caption_type=detected_caption_type,
                                            content=content,
                                            field_category=field_category,
                                            is_tags_format=is_tags_format,
                                            tag_match_rate=match_rate,
                                            tag_data=_build_tag_data_json(content) if is_tags_format else None,
                                            source="file",
                                            source_field=detected_caption_type
                                        )
                                        db.add(caption)
                                        captions_found += 1

                        elif ext_lower == '.json':
                            # JSON file: Recursively scan all fields
                            import json

                            with open(caption_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)

                            # Scan all fields
                            caption_results = scan_json_fields(json_data, taglist)

                            for result in caption_results:
                                caption_type = result["caption_type"]

                                # Enforce single tags field per item
                                if caption_type == "tags":
                                    existing_tags = db.query(DatasetCaption).filter(
                                        DatasetCaption.item_id == item_id_for_captions,
                                        DatasetCaption.caption_type == "tags"
                                    ).first()

                                    if existing_tags:
                                        # Update existing tags field
                                        existing_tags.content = result["content"]
                                        existing_tags.field_category = result["field_category"]
                                        existing_tags.is_tags_format = result["is_tags_format"]
                                        existing_tags.tag_match_rate = result["tag_match_rate"]
                                        existing_tags.source = "file"
                                        existing_tags.source_field = result["source_field"]
                                        if result["is_tags_format"]:
                                            existing_tags.tag_data = _build_tag_data_json(result["content"])
                                        existing_tags.updated_at = datetime.utcnow()
                                        continue  # Skip adding new caption

                                # Create new caption (for non-tags fields or first tags field)
                                _is_tags = result["is_tags_format"]
                                caption = DatasetCaption(
                                    item_id=item_id_for_captions,
                                    caption_type=caption_type,
                                    content=result["content"],
                                    field_category=result["field_category"],
                                    is_tags_format=_is_tags,
                                    tag_match_rate=result["tag_match_rate"],
                                    tag_data=_build_tag_data_json(result["content"]) if _is_tags else None,
                                    source="file",
                                    source_field=result["source_field"]
                                )
                                db.add(caption)
                                captions_found += 1

                    except Exception as e:
                        print(f"[Dataset Scan] Failed to read caption {caption_path}: {e}")

                # Process suffix-based caption files detected by 2-pass scanner
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
                        except Exception as e:
                            print(f"[Dataset Scan] Failed to read suffix caption {suffix_path}: {e}")

            except Exception as e:
                print(f"[Dataset Scan] Failed to process image {image_path}: {e}")

    # Run scan in thread pool to avoid blocking event loop (enables WebSocket progress updates)
    # SQLite is configured with check_same_thread=False, so cross-thread access is safe
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: scan_directory(dataset.path))

    # --- Purge: remove DB records whose files no longer exist on disk ---
    stale_paths = set(existing_paths.keys()) - seen_existing_paths
    items_purged = 0
    if stale_paths:
        stale_item_ids = [existing_paths[p] for p in stale_paths]
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

    # Compute tag statistics with progress updates (remaining 10%)
    print(f"[Dataset Scan] Computing tag statistics...")
    tag_statistics = await compute_tag_statistics(dataset_id, db, send_progress=True, total_steps=total_steps, current_step=total_images)

    # Send final completion progress
    manager.send_progress_sync(
        total_steps,
        total_steps,
        f"Scan complete: {items_found} new, {captions_updated} updated, {items_purged} purged, {len(tag_statistics)} unique tags"
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

    # Update dataset statistics (count all items in DB, not just newly added)
    dataset.total_items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).count()
    dataset.total_captions = db.query(DatasetCaption).filter(
        DatasetCaption.item_id.in_(
            db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset_id)
        )
    ).count()
    dataset.tag_statistics = tag_statistics
    dataset.last_scanned_at = datetime.utcnow()

    db.commit()
    db.refresh(dataset)

    response = {
        "items_found": items_found,
        "captions_found": captions_found,
        "captions_updated": captions_updated,
        "items_purged": items_purged,
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
    """Get a random caption from the dataset, optionally filtered by caption type"""
    import random

    # Check dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Build query for captions
    query = db.query(DatasetCaption).join(DatasetItem).filter(DatasetItem.dataset_id == dataset_id)

    # Filter by caption types if provided
    if caption_types:
        types_list = [t.strip() for t in caption_types.split(",")]
        query = query.filter(DatasetCaption.caption_type.in_(types_list))

    # Get all matching captions
    captions = query.all()

    if not captions:
        raise HTTPException(status_code=404, detail="No captions found in dataset")

    # Select random caption
    random_caption = random.choice(captions)

    # Fetch reference images from the DatasetItem
    item = db.query(DatasetItem).filter(DatasetItem.id == random_caption.item_id).first()
    reference_images = []
    if item and item.related_images:
        reference_images = item.related_images.get("reference", [])

    return {
        "caption": random_caption.content,
        "caption_type": random_caption.caption_type,
        "caption_subtype": random_caption.caption_subtype,
        "item_id": random_caption.item_id,
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
    training_method: str  # 'lora' or 'full_finetune'
    base_model_path: str

    # Training parameters
    total_steps: Optional[int] = None  # Mutually exclusive with epochs
    epochs: Optional[int] = None  # Mutually exclusive with total_steps
    batch_size: int = 1
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0  # Linear warmup steps before lr_scheduler kicks in
    optimizer: str = "adamw8bit"  # Options: adamw, adamw8bit, paged_adamw, paged_adamw8bit, adafactor, lion8bit, paged_lion8bit
    optimizer_is_paged: bool = False
    optimizer_cautious: bool = False
    optimizer_beta1: Optional[float] = None
    optimizer_beta2: Optional[float] = None
    optimizer_epsilon: Optional[float] = None
    optimizer_weight_decay: Optional[float] = None
    optimizer_schedule_free: bool = False  # Enable Schedule-Free optimizer (adamw8bit_ringbuffer, lion8bit_ringbuffer)
    optimizer_schedule_free_r: float = 0.0  # Schedule-Free r parameter (default: 0.0)
    optimizer_schedule_free_weight_lr_power: float = 2.0  # Schedule-Free weight lr power (default: 2.0)
    optimizer_use_radam: bool = False  # Use RAdam (Rectified Adam) with Schedule-Free (adamw8bit_ringbuffer, lion8bit_ringbuffer)
    optimizer_stochastic_rounding: bool = False  # Enable stochastic rounding for optimizers

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
    cache_latents_to_disk: bool = False  # Cache VAE latents and text embeddings to disk (default: False, in-memory cache)

    # Component-specific training
    train_unet: bool = True
    train_text_encoder: bool = False
    unet_lr: Optional[float] = None  # Defaults to learning_rate if None
    text_encoder_lr: Optional[float] = None  # Defaults to learning_rate if None
    text_encoder_1_lr: Optional[float] = None  # SDXL TE1 LR (defaults to text_encoder_lr if None)
    text_encoder_2_lr: Optional[float] = None  # SDXL TE2 LR (defaults to text_encoder_lr if None)

    # Precision and dtype settings (VRAM optimization)
    weight_dtype: str = "fp16"  # fp16, fp32, bf16, fp8_e4m3fn, fp8_e5m2
    training_dtype: str = "fp16"  # fp16, bf16, fp8_e4m3fn, fp8_e5m2 (activation dtype during training)
    output_dtype: str = "fp32"  # fp32, fp16, bf16, fp8_e4m3fn, fp8_e5m2 (output latent dtype)
    vae_dtype: str = "fp16"  # VAE-specific dtype (SDXL VAE works fine with fp16)
    mixed_precision: bool = True  # Enable mixed precision training (autocast)
    use_flash_attention: bool = False  # Enable Flash Attention for training (faster, lower memory)
    min_snr_gamma: float = 5.0  # Min-SNR gamma for loss weighting (default: 5.0, set to 0 to disable)

    # Text encoding settings
    text_encoding_mode: str = "swap_onthefly"  # "swap_onthefly", "pre_encoded_cache", "onthefly_gpu"
    text_encoding_swap_interval: int = 256  # Swap interval for swap_onthefly mode

    # Latent encoding settings
    latent_encoding_mode: str = "swap_onthefly"  # "swap_onthefly", "pre_encoded_cache", "onthefly_gpu"
    latent_encoding_swap_interval: int = 256  # Swap interval for swap_onthefly mode

    # Block Swap settings (training VRAM optimization)
    blocks_to_swap: int = 0  # Number of transformer blocks to swap (0 to disable)
    use_pinned_memory: bool = False  # Use CUDA pinned memory for faster transfer
    num_optimizer_groups: int = 0  # Number of optimizer groups for fused optimizer (0 to disable, recommended 4-10)

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
    # sample_condition_image_path is now per-prompt in sample_prompts[].condition_image_path

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

        # Check if base model exists
        if not os.path.exists(request.base_model_path):
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

        # Generate YAML config
        config_generator = TrainingConfigGenerator()

        # Build params dict from Pydantic request, plus resume_from_checkpoint
        # which has special handling (resolved separately above).
        params_dict = request.model_dump()
        params_dict["resume_from_checkpoint"] = resume_from_checkpoint

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
        else:  # full_finetune
            config_yaml = config_generator.generate_full_finetune_config(params_dict, **common_kwargs)

        # Save config file
        config_path = os.path.join(output_dir_str, f"{run_name}_config.yaml")
        config_generator.save_config(config_yaml, config_path)

        # Create training run
        # Calculate total_steps for database if epochs provided
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
    """List all training runs"""
    try:
        runs = db.query(TrainingRun).order_by(TrainingRun.created_at.desc()).all()
        return {"runs": [run.to_dict() for run in runs], "total": len(runs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/runs/{run_id}")
async def get_training_run(run_id: int, db: Session = Depends(get_training_db)):
    """Get training run details"""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run.to_dict()

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
}

# Fields excluded from auto-extraction (they need special handling outside the schema loop)
_AUTO_EXTRACT_EXCLUDE = {
    "dataset_id", "dataset_configs", "run_name", "training_method",
    "cache_latents_to_disk",  # Read from datasets[0], not train section
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
    if network_config.get("type") == "lora":
        job = "lora"
    elif network_config.get("type") == "controlnet":
        job = "controlnet"
    elif not network_config:  # No network section means full fine-tune
        job = "full_finetune"
    # Otherwise keep job from config.job
    datasets_config = process_config.get("datasets", [])

    # Build dataset_configs from YAML
    dataset_start = time.time()
    dataset_configs = []
    cache_latents_to_disk = False  # Default
    for ds_config in datasets_config:
        # Try to find dataset by path
        dataset_path = ds_config.get("folder_path", ds_config.get("path", ""))
        print(f"[get_training_run_params] Looking for dataset with path: {dataset_path}")
        dataset = datasets_db.query(Dataset).filter(Dataset.path == dataset_path).first()
        if dataset:
            print(f"[get_training_run_params] Found dataset: id={dataset.id}, name={dataset.name}")
            from core.training.dataset_params import read_dataset_params
            entry = {
                "dataset_id": dataset.id,
                "filters": {},
                **read_dataset_params(ds_config),
            }
            dataset_configs.append(entry)
        else:
            print(f"[get_training_run_params] Dataset not found in database for path: {dataset_path}")
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
    params["training_method"] = "lora" if job == "lora" else ("controlnet" if job == "controlnet" else "full_finetune")
    params["dataset_configs"] = dataset_configs if dataset_configs else None
    params["cache_latents_to_disk"] = cache_latents_to_disk

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
        else:  # full_finetune
            config_yaml = config_generator.generate_full_finetune_config(params_dict, **common_kwargs)

        # Update config_yaml and base_model_path in database
        run.config_yaml = config_yaml
        run.base_model_path = request.base_model_path

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
                    # Process failed with error
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

@router.patch("/training/runs/{run_id}/config")
async def update_training_config(run_id: int, config_data: dict, db: Session = Depends(get_training_db)):
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
async def reload_training_config(run_id: int, db: Session = Depends(get_training_db)):
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

    if "latents" in data:
        img = latent_to_image(data["latents"], is_flux2=is_flux2)
        result["latents_image"] = image_to_base64(img)

    if "noisy_latents" in data:
        img = latent_to_image(data["noisy_latents"], is_flux2=is_flux2)
        result["noisy_latents_image"] = image_to_base64(img)

    # predicted_noise (SD/SDXL) or predicted_velocity (Z-Image/FLUX.2)
    if "predicted_noise" in data:
        img = latent_to_image(data["predicted_noise"], is_flux2=is_flux2)
        result["predicted_noise_image"] = image_to_base64(img)

    if "predicted_velocity" in data:
        img = latent_to_image(data["predicted_velocity"], is_flux2=is_flux2)
        result["predicted_velocity_image"] = image_to_base64(img)

    if "predicted_latent" in data:
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

        # Decimate data if too many points (simple nth-point sampling)
        def decimate(data, max_points):
            if len(data) <= max_points:
                return data
            step_size = len(data) // max_points
            return [data[i] for i in range(0, len(data), step_size)][:max_points]

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
                "timestamp": m.timestamp.isoformat() if m.timestamp else None
            }

            if is_valid_float(m.loss):
                loss_data.append({**point, "value": m.loss})

            if is_valid_float(m.recon_loss):
                recon_loss_data.append({**point, "value": m.recon_loss})

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

        return {
            "loss": loss_data,
            "recon_loss": recon_loss_data,
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
    save_every_n_steps: int = 500
    save_every_n_epochs: int = 0
    keep_last_n_checkpoints: int = 3
    checkpoint_save_mode: str = "lora"
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    weight_decay: float = 1e-4
    loss_gamma_neg: float = 4.0
    loss_gamma_pos: float = 1.0
    loss_clip: float = 0.05
    validate_every: int = 1
    val_split: float = 0.05
    vocab_min_count: int = 10
    output_dir: Optional[str] = None
    excluded_categories: Optional[List[str]] = None
    ban_tags: Optional[str] = None
    use_tag_aliases: bool = False
    cls_dim: Optional[int] = None
    hidden_proj_dim: Optional[int] = None
    init_head_from: Optional[str] = None


# Active tagger training threads
_tagger_training_threads: Dict[str, Any] = {}


def _make_tagger_progress_callback(run_id: str, training_db_factory):
    """Returns a callback that writes progress to DB."""
    def callback(rid: str, event_type: str, data: dict):
        db = training_db_factory()
        try:
            run = db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == rid).first()
            if not run:
                return
            def _upsert_metric(step: int, **kwargs):
                existing = db.query(TaggerTrainingMetrics).filter(
                    TaggerTrainingMetrics.run_id == rid,
                    TaggerTrainingMetrics.step == step,
                ).first()
                if existing:
                    for k, v in kwargs.items():
                        if v is not None:
                            setattr(existing, k, v)
                else:
                    db.add(TaggerTrainingMetrics(run_id=rid, step=step, **kwargs))

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
            elif event_type == "epoch":
                run.current_epoch = data.get("epoch", run.current_epoch)
                run.latest_loss   = data.get("loss")
                if data.get("f1") is not None:
                    _upsert_metric(
                        step=run.current_step,
                        epoch=data.get("epoch"),
                        loss=data.get("loss"),
                        f1=data.get("f1"),
                        threshold=data.get("threshold"),
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
            elif event_type == "vocab":
                run.num_tags = data.get("num_tags")
            elif event_type == "resume":
                run.resumed_from_step = data.get("resumed_from_step")
                run.last_resumed_at   = datetime.now()
            elif event_type == "phase":
                msg = data.get("message") or data.get("phase") or ""
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
    return [r.to_dict() for r in runs]


@router.get("/tagger-training/runs/{run_id}")
def get_tagger_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Get a tagger training run by run_id."""
    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Tagger training run not found")
    return run.to_dict()


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
def start_tagger_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Start or resume a tagger training run in a background thread."""
    import threading
    from database import TrainingSessionLocal

    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Tagger training run not found")
    if run.status == "running":
        raise HTTPException(status_code=400, detail="Run is already running")

    run.status        = "running"
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

    def _run():
        from core.tagger.tagger_trainer import run_tagger_training
        db = TrainingSessionLocal()
        try:
            result = run_tagger_training(
                run_id=run_id,
                config=config,
                dataset_ids=dataset_ids,
                output_dir=output_dir,
                progress_callback=callback,
                resume_from_checkpoint=resume_from_checkpoint,
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
    _tagger_training_threads[run_id] = thread
    thread.start()

    training_db.refresh(run)
    return {"message": "started", "run": run.to_dict()}


@router.post("/tagger-training/runs/{run_id}/stop")
def stop_tagger_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Request stop for a running tagger training run."""
    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Tagger training run not found")
    run.status = "stopped"
    training_db.commit()
    training_db.refresh(run)
    return {"message": "stop_requested", "run": run.to_dict()}


@router.delete("/tagger-training/runs/{run_id}")
def delete_tagger_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Delete a tagger training run record."""
    run = training_db.query(TaggerTrainingRun).filter(TaggerTrainingRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Tagger training run not found")
    training_db.delete(run)
    training_db.commit()
    return {"deleted": run_id}


@router.get("/tagger-training/runs/{run_id}/metrics")
def get_tagger_training_metrics(
    run_id: str,
    since_step: int = 0,
    training_db: Session = Depends(get_training_db),
):
    """Get per-step metrics for a tagger training run."""
    metrics = (
        training_db.query(TaggerTrainingMetrics)
        .filter(
            TaggerTrainingMetrics.run_id == run_id,
            TaggerTrainingMetrics.step >= since_step,
        )
        .order_by(TaggerTrainingMetrics.step)
        .all()
    )
    return [m.to_dict() for m in metrics]


@router.get("/tagger-training/vocabulary-preview")
def preview_tagger_vocabulary(
    dataset_ids: str,
    excluded_categories: Optional[str] = None,
    ban_tags: Optional[str] = None,
    datasets_db: Session = Depends(get_datasets_db),
):
    """Preview tag vocabulary for given dataset IDs (comma-separated).

    Returns tag count and category breakdown without full vocab.

    Parameters
    ----------
    dataset_ids         : comma-separated dataset IDs
    excluded_categories : comma-separated category names to exclude
    ban_tags            : newline or comma-separated tag patterns (fnmatch wildcards)
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

    taglist_cache.initialize(settings.root_dir)

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
    lookup_needed: list = []

    for tag_data_json, content in rows:
        if tag_data_json:
            try:
                items = _json.loads(tag_data_json) if isinstance(tag_data_json, str) else tag_data_json
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and "tag" in item:
                            norm = normalize_tag(item["tag"])
                            tag_counts[norm] += 1
                            if norm not in tag_categories:
                                tag_categories[norm] = item.get("category", "General")
                    continue
            except Exception:
                pass
        # Fallback: parse content, resolve categories later
        if content:
            for t in content.split(","):
                t = t.strip()
                if t:
                    norm = normalize_tag(t)
                    tag_counts[norm] += 1
                    if norm not in tag_categories:
                        tag_categories[norm] = "__lookup__"
                        lookup_needed.append(norm)

    # Batch-resolve any __lookup__ sentinels
    if lookup_needed:
        resolved = taglist_cache.get_categories_batch(list(set(lookup_needed)))
        for norm_tag in lookup_needed:
            if tag_categories.get(norm_tag) == "__lookup__":
                tag_categories[norm_tag] = resolved.get(norm_tag, "General")

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
