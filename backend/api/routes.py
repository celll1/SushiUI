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
from database.models import GeneratedImage, UserSettings, Dataset, DatasetItem, DatasetCaption, TagDictionary, TrainingRun, TrainingCheckpoint, TrainingSample, TrainingPreset
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
    preprocessor: Optional[str] = None  # Preprocessor type (auto-detected if None)
    enable_preprocessor: bool = True  # Whether to apply preprocessing

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
    # torch.compile optimization
    use_torch_compile: bool = False  # Enable torch.compile for U-Net (1.3-2x speedup)
    # TIPO (prompt upsampling)
    use_tipo: bool = False  # Enable TIPO prompt upsampling
    tipo_config: Optional[Dict] = None  # TIPO configuration (model, lengths, etc.)

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
    unet_quantization: Optional[str] = Form(None),
    use_torch_compile: bool = Form(False),
    use_tipo: bool = Form(False),
    tipo_config: str = Form("{}"),  # JSON string of TIPO config
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
                    enabled_categories=tipo_config_dict.get("enabled_categories", {})
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
            "unet_quantization": unet_quantization,
            "use_torch_compile": use_torch_compile,
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

        # Detect if SDXL
        is_sdxl = pipeline_manager.txt2img_pipeline is not None and \
                  "XL" in pipeline_manager.txt2img_pipeline.__class__.__name__

        # Progress callback to send updates via WebSocket
        progress_callback = create_progress_callback_factory(
            taesd_manager,
            manager,
            is_sdxl
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
        image, actual_seed = await loop.run_in_executor(
            executor,
            lambda: pipeline_manager.generate_txt2img(params, progress_callback=progress_callback, step_callback=step_callback)
        )

        # Update params with actual seed
        params["seed"] = actual_seed

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
    unet_quantization: Optional[str] = Form(None),
    use_torch_compile: bool = Form(False),
    use_tipo: bool = Form(False),
    tipo_config: str = Form("{}"),  # JSON string of TIPO config
    image: UploadFile = File(...),
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
                    enabled_categories=tipo_config_dict.get("enabled_categories", {})
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
            "unet_quantization": unet_quantization,
            "use_torch_compile": use_torch_compile,
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

        # Progress callback to send updates via WebSocket
        progress_callback = create_progress_callback_factory(
            taesd_manager,
            manager,
            is_sdxl,
            img2img_fix_steps,
            steps
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
        result_image, actual_seed = await loop.run_in_executor(
            executor,
            lambda: pipeline_manager.generate_img2img(params, init_image, progress_callback=progress_callback, step_callback=step_callback)
        )

        # Update params with actual seed
        params["seed"] = actual_seed

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
    unet_quantization: Optional[str] = Form(None),
    use_torch_compile: bool = Form(False),
    use_tipo: bool = Form(False),
    tipo_config: str = Form("{}"),  # JSON string of TIPO config
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
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
                    enabled_categories=tipo_config_dict.get("enabled_categories", {})
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
            "unet_quantization": unet_quantization,
            "use_torch_compile": use_torch_compile,
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

        # Progress callback to send updates via WebSocket
        progress_callback = create_progress_callback_factory(
            taesd_manager,
            manager,
            is_sdxl,
            img2img_fix_steps,
            steps
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
        result_image, actual_seed = await loop.run_in_executor(
            executor,
            lambda: pipeline_manager.generate_inpaint(params, init_image, mask_image, progress_callback=progress_callback, step_callback=step_callback)
        )

        # Update params with actual seed
        params["seed"] = actual_seed

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
async def get_models(db: Session = Depends(get_gallery_db)):
    """Get list of available models from default and user-configured directories"""
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
            if os.path.isdir(item_path):
                # Diffusers format directory
                models.append({
                    "name": item,
                    "path": item_path,
                    "type": "diffusers",
                    "source_type": "diffusers",
                    "source_dir": models_dir
                })
            elif item.endswith('.safetensors'):
                # Safetensors file
                file_size = os.path.getsize(item_path) / (1024**3)  # GB
                models.append({
                    "name": item.replace('.safetensors', ''),
                    "path": item_path,
                    "type": "safetensors",
                    "source_type": "safetensors",
                    "size_gb": round(file_size, 2),
                    "source_dir": models_dir
                })

    print(f"[Models] Found {len(models)} models total")
    return {"models": models}

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
    """Get available samplers (static list, doesn't require model)"""
    try:
        samplers = get_available_samplers()
        display_names = get_sampler_display_names()
        return {
            "samplers": [
                {"id": sampler_id, "name": display_names.get(sampler_id, sampler_id)}
                for sampler_id in samplers
            ]
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
                controlnet_dirs=[]
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
    model_dirs: List[str] = [],
    lora_dirs: List[str] = [],
    controlnet_dirs: List[str] = [],
    db: Session = Depends(get_gallery_db)
):
    """Save user-configured model directories"""
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
        settings_record.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(settings_record)

        print(f"[Settings] Updated directory settings:")
        print(f"  Model dirs: {settings_record.model_dirs}")
        print(f"  LoRA dirs: {settings_record.lora_dirs}")
        print(f"  ControlNet dirs: {settings_record.controlnet_dirs}")

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
    Get modification timestamps for all tag files to check if cache is stale
    Returns Unix timestamps in milliseconds
    """
    try:
        category_map = {
            "general": "General",
            "character": "Character",
            "artist": "Artist",
            "copyright": "Copyright",
            "meta": "Meta",
            "model": "Model"
        }

        timestamps = {}

        # Get taglist file timestamps
        for category_key, filename in category_map.items():
            taglist_path = os.path.join(settings.root_dir, "taglist", f"{filename}.json")
            if os.path.exists(taglist_path):
                mtime = os.path.getmtime(taglist_path)
                timestamps[category_key] = int(mtime * 1000)  # Convert to ms
            else:
                timestamps[category_key] = 0  # File doesn't exist

        # Get tag_other_names timestamp
        tagother_path = os.path.join(settings.root_dir, "tagother", "tag_other_names.json")
        if os.path.exists(tagother_path):
            mtime = os.path.getmtime(tagother_path)
            timestamps["other_names"] = int(mtime * 1000)
        else:
            timestamps["other_names"] = 0

        return timestamps
    except Exception as e:
        print(f"Error getting tag file timestamps: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/taglist/{category}")
async def get_taglist(category: str):
    """
    Get tag list for a specific category from taglist directory
    Categories: general, character, artist, copyright, meta, model
    """
    try:
        # Map category names to file names (capitalize first letter)
        category_map = {
            "general": "General",
            "character": "Character",
            "artist": "Artist",
            "copyright": "Copyright",
            "meta": "Meta",
            "model": "Model"
        }

        if category.lower() not in category_map:
            raise HTTPException(status_code=404, detail=f"Unknown category: {category}")

        filename = category_map[category.lower()]
        taglist_path = os.path.join(settings.root_dir, "taglist", f"{filename}.json")

        if not os.path.exists(taglist_path):
            raise HTTPException(status_code=404, detail=f"Taglist file not found: {taglist_path}")

        import json
        with open(taglist_path, "r", encoding="utf-8") as f:
            tags = json.load(f)

        return tags
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading taglist: {e}")
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
            ban_tags=request.ban_tags
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
    total_items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset.id).count()
    total_captions = db.query(DatasetCaption).filter(
        DatasetCaption.item_id.in_(
            db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset.id)
        )
    ).count()

    # Only update if values changed (avoid unnecessary writes)
    if dataset.total_items != total_items or dataset.total_captions != total_captions:
        dataset.total_items = total_items
        dataset.total_captions = total_captions
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

async def compute_tag_statistics(dataset_id: int, db: Session) -> dict:
    """
    Compute tag statistics for a dataset: tag counts only (no categories)
    Returns: {"tag": {"count": N}, ...}

    Note: Categories are determined by frontend (tagSuggestions.ts) to maintain consistency.
    Backend only counts tag occurrences.
    """
    print(f"[Dataset] Computing tag statistics for dataset {dataset_id}...")

    # Get all items in dataset
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()
    if not items:
        print(f"[Dataset] No items found, returning empty statistics")
        return {}

    # Get all tag captions
    item_ids = [item.id for item in items]
    tag_captions = db.query(DatasetCaption).filter(
        DatasetCaption.item_id.in_(item_ids),
        DatasetCaption.caption_type == "tags"
    ).all()

    # Count tag occurrences
    tag_counts: dict[str, int] = {}
    for caption in tag_captions:
        if caption.content:
            tags = caption.content.split(",")
            for tag in tags:
                tag = tag.strip()
                if tag:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"[Dataset] Found {len(tag_counts)} unique tags")

    # Build final statistics (count only, categories added by frontend)
    statistics = {}
    for tag, count in tag_counts.items():
        statistics[tag] = {
            "count": count
        }

    print(f"[Dataset] Tag statistics computed: {len(statistics)} tags")
    return statistics

@router.post("/datasets/{dataset_id}/scan")
async def scan_dataset(
    dataset_id: int,
    db: Session = Depends(get_datasets_db)
):
    """Scan dataset directory and register images/captions"""
    import os
    from PIL import Image
    import hashlib

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {dataset.path}")

    # Supported image extensions
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    caption_exts = {".txt"}

    # Scan directory
    items_found = 0
    captions_found = 0

    def scan_directory(dir_path, current_depth=0):
        nonlocal items_found, captions_found

        try:
            entries = os.listdir(dir_path)
        except PermissionError:
            print(f"[Dataset Scan] Permission denied: {dir_path}")
            return

        # Group files by base name
        file_groups = {}
        for entry in entries:
            entry_path = os.path.join(dir_path, entry)

            if os.path.isfile(entry_path):
                base_name, ext = os.path.splitext(entry)
                ext_lower = ext.lower()

                if ext_lower in image_exts:
                    if base_name not in file_groups:
                        file_groups[base_name] = {"images": [], "captions": []}
                    file_groups[base_name]["images"].append(entry_path)
                elif ext_lower in caption_exts:
                    if base_name not in file_groups:
                        file_groups[base_name] = {"images": [], "captions": []}
                    file_groups[base_name]["captions"].append(entry_path)

            elif os.path.isdir(entry_path) and dataset.recursive:
                max_depth = dataset.max_depth if dataset.max_depth else float('inf')
                if current_depth < max_depth:
                    scan_directory(entry_path, current_depth + 1)

        # Process file groups
        for base_name, files in file_groups.items():
            if not files["images"]:
                continue

            # Use first image as primary
            image_path = files["images"][0]

            try:
                # Read image metadata
                with Image.open(image_path) as img:
                    width, height = img.size
                    file_size = os.path.getsize(image_path)

                    # Calculate image hash
                    with open(image_path, 'rb') as f:
                        image_hash = hashlib.sha256(f.read()).hexdigest()

                    # Check if item already exists
                    existing_item = db.query(DatasetItem).filter(
                        DatasetItem.dataset_id == dataset_id,
                        DatasetItem.image_hash == image_hash
                    ).first()

                    if existing_item:
                        continue  # Skip duplicate

                    # Create dataset item
                    item = DatasetItem(
                        dataset_id=dataset_id,
                        item_type="single",
                        base_name=base_name,
                        image_path=image_path,
                        width=width,
                        height=height,
                        file_size=file_size,
                        image_hash=image_hash
                    )
                    db.add(item)
                    db.flush()  # Get item.id
                    items_found += 1

                    # Process captions
                    for caption_path in files["captions"]:
                        try:
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    caption = DatasetCaption(
                                        item_id=item.id,
                                        caption_type="tags",
                                        content=content,
                                        source="file"
                                    )
                                    db.add(caption)
                                    captions_found += 1
                        except Exception as e:
                            print(f"[Dataset Scan] Failed to read caption {caption_path}: {e}")

            except Exception as e:
                print(f"[Dataset Scan] Failed to process image {image_path}: {e}")

    # Start scanning
    scan_directory(dataset.path)

    # Compute tag statistics
    tag_statistics = await compute_tag_statistics(dataset_id, db)

    # Update dataset statistics
    dataset.total_items = items_found
    dataset.total_captions = captions_found
    dataset.tag_statistics = tag_statistics
    dataset.last_scanned_at = datetime.utcnow()

    db.commit()
    db.refresh(dataset)

    return {
        "items_found": items_found,
        "captions_found": captions_found,
        "dataset": dataset.to_dict()
    }

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
    """Get available caption types in the dataset with counts and subtypes"""
    # Check dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Query caption types with counts
    # Group by caption_type and caption_subtype
    from sqlalchemy import func

    results = db.query(
        DatasetCaption.caption_type,
        DatasetCaption.caption_subtype,
        func.count(DatasetCaption.id).label('count')
    ).join(DatasetItem).filter(
        DatasetItem.dataset_id == dataset_id
    ).group_by(
        DatasetCaption.caption_type,
        DatasetCaption.caption_subtype
    ).all()

    # Organize results by caption_type
    caption_types_dict = {}
    for caption_type, caption_subtype, count in results:
        if caption_type not in caption_types_dict:
            caption_types_dict[caption_type] = {
                "caption_type": caption_type,
                "total_count": 0,
                "subtypes": []
            }

        caption_types_dict[caption_type]["total_count"] += count

        if caption_subtype:
            caption_types_dict[caption_type]["subtypes"].append({
                "subtype": caption_subtype,
                "count": count
            })

    # Convert to list and sort by count
    caption_types_list = sorted(
        caption_types_dict.values(),
        key=lambda x: x["total_count"],
        reverse=True
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

    return {
        "caption": random_caption.content,
        "caption_type": random_caption.caption_type,
        "caption_subtype": random_caption.caption_subtype,
        "item_id": random_caption.item_id,
    }

# ============================================================
# Dataset Item Caption Update API
# ============================================================

class CaptionUpdateRequest(BaseModel):
    caption_type: str = "tags"
    content: str

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
        caption.updated_at = datetime.utcnow()
    else:
        # Create new caption
        caption = DatasetCaption(
            item_id=item_id,
            caption_type=request.caption_type,
            content=request.content,
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
                    # New tag - no category determination (frontend handles it)
                    tag_statistics[tag] = {
                        "count": 1
                    }

            # Save updated statistics
            dataset.tag_statistics = tag_statistics
            db.commit()

    return {"status": "success", "caption": caption.to_dict()}

# Tag Dictionary Search API was removed - frontend uses tagSuggestions.ts (JSON files) instead

# ============================================================
# Training API Endpoints
# ============================================================

from database.models import TrainingRun, TrainingCheckpoint, TrainingSample

class DatasetConfigItem(BaseModel):
    dataset_id: int
    caption_types: List[str] = []  # Empty = use all caption types
    filters: Dict[str, Any] = {}  # {"tag_include": ["1girl"], "tag_exclude": ["photo"], "caption_contains": "smile"}

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
    optimizer: str = "adamw8bit"  # Options: adamw, adamw8bit, paged_adamw, paged_adamw8bit, adafactor, lion8bit, paged_lion8bit

    # LoRA specific
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 16
    network_type: Optional[str] = "lora"

    # Advanced
    save_every: int = 100
    save_every_unit: str = "steps"  # "steps" or "epochs"
    sample_every: int = 100
    sample_prompts: List[Dict[str, str]] = []  # List of {positive: str, negative: str}
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

    # Sample generation parameters
    sample_width: int = 1024
    sample_height: int = 1024
    sample_steps: int = 28
    sample_cfg_scale: float = 7.0
    sample_sampler: str = "euler"
    sample_schedule_type: str = "sgm_uniform"
    sample_seed: int = -1  # -1 for random

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

        # Build dataset_configs_for_yaml (with path and caption_processing)
        dataset_configs_for_yaml = []
        for config in dataset_configs:
            dataset = datasets_db.query(Dataset).filter(Dataset.id == config["dataset_id"]).first()
            if dataset:
                dataset_configs_for_yaml.append({
                    "path": dataset.path,
                    "caption_processing": dataset.caption_processing or {}
                })

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

        # Create output directory (use absolute path from project root)
        project_root = Path(__file__).parent.parent.parent  # backend/api/routes.py -> project root
        output_dir = project_root / "training" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_str = str(output_dir)

        # Generate YAML config
        config_generator = TrainingConfigGenerator()

        if request.training_method == "lora":
            config_yaml = config_generator.generate_lora_config(
                run_name=run_name,
                dataset_path=primary_dataset.path,  # Kept for backward compatibility
                base_model_path=request.base_model_path,
                output_dir=output_dir_str,
                dataset_configs=dataset_configs_for_yaml,  # New: multiple datasets
                total_steps=request.total_steps,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                lr_scheduler=request.lr_scheduler,
                optimizer=request.optimizer,
                lora_rank=request.lora_rank or 16,
                lora_alpha=request.lora_alpha or 16,
                save_every=request.save_every,
                save_every_unit=request.save_every_unit,
                sample_every=request.sample_every,
                sample_prompts=request.sample_prompts or [],
                debug_latents=request.debug_latents,
                debug_latents_every=request.debug_latents_every,
                enable_bucketing=request.enable_bucketing,
                base_resolutions=request.base_resolutions,
                bucket_strategy=request.bucket_strategy,
                multi_resolution_mode=request.multi_resolution_mode,
                train_unet=request.train_unet,
                train_text_encoder=request.train_text_encoder,
                unet_lr=request.unet_lr,
                text_encoder_lr=request.text_encoder_lr,
                text_encoder_1_lr=request.text_encoder_1_lr,
                text_encoder_2_lr=request.text_encoder_2_lr,
                cache_latents_to_disk=request.cache_latents_to_disk,
                weight_dtype=request.weight_dtype,
                training_dtype=request.training_dtype,
                output_dtype=request.output_dtype,
                vae_dtype=request.vae_dtype,
                mixed_precision=request.mixed_precision,
                use_flash_attention=request.use_flash_attention,
                min_snr_gamma=request.min_snr_gamma,
                sample_width=request.sample_width,
                sample_height=request.sample_height,
                sample_steps=request.sample_steps,
                sample_cfg_scale=request.sample_cfg_scale,
                sample_sampler=request.sample_sampler,
                sample_seed=request.sample_seed,
                caption_processing=primary_dataset.caption_processing,  # Pass caption processing config
            )
        else:  # full_finetune
            config_yaml = config_generator.generate_full_finetune_config(
                run_name=run_name,
                dataset_path=primary_dataset.path,  # Kept for backward compatibility
                base_model_path=request.base_model_path,
                output_dir=output_dir_str,
                dataset_configs=dataset_configs_for_yaml,  # New: multiple datasets
                total_steps=request.total_steps,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                lr_scheduler=request.lr_scheduler,
                optimizer=request.optimizer,
                save_every=request.save_every,
                save_every_unit=request.save_every_unit,
                sample_every=request.sample_every,
                sample_prompts=request.sample_prompts or [],
                debug_latents=request.debug_latents,
                debug_latents_every=request.debug_latents_every,
                enable_bucketing=request.enable_bucketing,
                base_resolutions=request.base_resolutions,
                bucket_strategy=request.bucket_strategy,
                multi_resolution_mode=request.multi_resolution_mode,
                train_unet=request.train_unet,
                train_text_encoder=request.train_text_encoder,
                unet_lr=request.unet_lr,
                text_encoder_lr=request.text_encoder_lr,
                text_encoder_1_lr=request.text_encoder_1_lr,
                text_encoder_2_lr=request.text_encoder_2_lr,
                cache_latents_to_disk=request.cache_latents_to_disk,
                weight_dtype=request.weight_dtype,
                training_dtype=request.training_dtype,
                output_dtype=request.output_dtype,
                vae_dtype=request.vae_dtype,
                mixed_precision=request.mixed_precision,
                use_flash_attention=request.use_flash_attention,
                min_snr_gamma=request.min_snr_gamma,
                sample_width=request.sample_width,
                sample_height=request.sample_height,
                sample_steps=request.sample_steps,
                sample_cfg_scale=request.sample_cfg_scale,
                sample_sampler=request.sample_sampler,
                sample_seed=request.sample_seed,
                caption_processing=primary_dataset.caption_processing,  # Pass caption processing config
            )

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
        run.started_at = datetime.utcnow()
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

                # Negative step indicates failure
                if step < 0:
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

    if run.status != "running":
        raise HTTPException(status_code=400, detail="Training run is not running")

    try:
        # Get training process
        process = training_process_manager.get_process(run_id)

        if process:
            await process.stop()
            await training_process_manager.remove_process(run_id)

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

        # Update config_yaml
        run.config_yaml = config_yaml

        # Also update the YAML file on disk if it exists
        import yaml
        from pathlib import Path

        config_path = Path(run.output_dir) / "config.yaml"
        if config_path.parent.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_yaml)

        db.commit()

        return {"message": "Configuration updated successfully", "run": run.to_dict()}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

@router.get("/training/runs/{run_id}/status")
async def get_training_status(run_id: int, db: Session = Depends(get_training_db)):
    """Get current training status"""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    # Update checkpoint_paths by scanning output directory
    from pathlib import Path
    import glob
    output_dir = Path(run.output_dir)
    if output_dir.exists():
        # Find all checkpoint files (*_step_*.safetensors)
        # Pattern matches both old format (lora_step_*.safetensors) and new format ({run_name}_step_*.safetensors)
        checkpoint_files = glob.glob(str(output_dir / "*_step_*.safetensors"))
        # Convert to absolute paths and sort
        checkpoint_paths = sorted([str(Path(p)) for p in checkpoint_files])

        # Update database if changed
        if checkpoint_paths != run.checkpoint_paths:
            run.checkpoint_paths = checkpoint_paths
            db.commit()

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

    from pathlib import Path
    import glob

    output_dir = Path(run.output_dir)
    if not output_dir.exists():
        return {"checkpoints": []}

    # Find all checkpoint files (lora_step_*.safetensors)
    checkpoint_files = glob.glob(str(output_dir / "lora_step_*.safetensors"))

    checkpoints = []
    for ckpt_path in sorted(checkpoint_files):
        ckpt_file = Path(ckpt_path)
        # Extract step number from filename
        filename = ckpt_file.name  # e.g., "lora_step_100.safetensors"
        step_str = filename.replace("lora_step_", "").replace(".safetensors", "")
        try:
            step = int(step_str)
            checkpoints.append({
                "step": step,
                "filename": filename,
                "path": str(ckpt_file)
            })
        except ValueError:
            continue

    # Sort by step number (descending)
    checkpoints.sort(key=lambda x: x["step"], reverse=True)

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
                # Extract timestep from filename (latents_tXXXX.pt)
                timestep_str = latent_file.stem.replace("latents_t", "")
                try:
                    timestep = int(timestep_str)
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
        latent_file = debug_dir / f"latents_t{timestep:04d}.pt"
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

    def latent_to_image(latent_tensor):
        """Convert latent tensor to PIL Image (without VAE decoding)"""
        # latent_tensor shape: [1, C, H, W] or [C, H, W]
        if latent_tensor.dim() == 4:
            latent_tensor = latent_tensor[0]  # Remove batch dimension

        # latent_tensor shape: [C, H, W]
        # For SDXL latents: C=4, we'll map first 3 channels to RGB
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

    # Convert each latent type to image
    result = {
        "step": step,
        "timestep": data.get("timestep", 0),
        "loss": data.get("loss", 0.0),
    }

    # Add caption if available
    if "caption" in data:
        result["caption"] = data["caption"]

    if "latents" in data:
        img = latent_to_image(data["latents"])
        result["latents_image"] = image_to_base64(img)

    if "noisy_latents" in data:
        img = latent_to_image(data["noisy_latents"])
        result["noisy_latents_image"] = image_to_base64(img)

    if "predicted_noise" in data:
        img = latent_to_image(data["predicted_noise"])
        result["predicted_noise_image"] = image_to_base64(img)

    if "predicted_latent" in data:
        img = latent_to_image(data["predicted_latent"])
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
        return {"loss": [], "learning_rate": []}

    try:
        from tensorboard.backend.event_processing import event_accumulator

        # Find all event files in all subdirectories (timestamp-based)
        event_files = []
        for subdir in tensorboard_dir.iterdir():
            if subdir.is_dir():
                event_files.extend(glob.glob(str(subdir / "events.out.tfevents.*")))

        if not event_files:
            return {"loss": [], "learning_rate": []}

        # Use the most recent event file or merge all
        all_loss = []
        all_lr = []

        for event_file in event_files:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            # Get scalar tags
            if 'train/loss' in ea.Tags()['scalars']:
                loss_events = ea.Scalars('train/loss')
                all_loss.extend([
                    {"step": int(e.step), "value": float(e.value), "wall_time": float(e.wall_time)}
                    for e in loss_events
                ])

            if 'train/learning_rate' in ea.Tags()['scalars']:
                lr_events = ea.Scalars('train/learning_rate')
                all_lr.extend([
                    {"step": int(e.step), "value": float(e.value), "wall_time": float(e.wall_time)}
                    for e in lr_events
                ])

        # Sort by step
        all_loss.sort(key=lambda x: x["step"])
        all_lr.sort(key=lambda x: x["step"])

        # Filter by since_step if provided
        if since_step is not None:
            all_loss = [d for d in all_loss if d["step"] > since_step]
            all_lr = [d for d in all_lr if d["step"] > since_step]

        # Decimate data if too many points (simple nth-point sampling)
        def decimate(data, max_points):
            if len(data) <= max_points:
                return data
            step_size = len(data) // max_points
            return [data[i] for i in range(0, len(data), step_size)][:max_points]

        all_loss = decimate(all_loss, max_points)
        all_lr = decimate(all_lr, max_points)

        return {
            "loss": all_loss,
            "learning_rate": all_lr
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="TensorBoard library not available")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {str(e)}")


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

    # Get absolute path of training directory (project root / training)
    import os
    project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    training_base = (project_root / "training").resolve()

    for file in sample_files:
        match = pattern.match(file.name)
        if match:
            step = int(match.group(1))
            sample_idx = int(match.group(2))

            if step not in samples_by_step:
                samples_by_step[step] = []

            # Convert absolute path to relative path from training directory
            # e.g., /path/to/training/20251130_220607_28dee11d/samples/step_000100_sample_0.png
            # -> 20251130_220607_28dee11d/samples/step_000100_sample_0.png
            file_abs = file.resolve()
            try:
                relative_path = file_abs.relative_to(training_base)
                path_url = f"/training/{relative_path.as_posix()}"
            except ValueError:
                # Fallback: if not under training directory, construct path from output_dir
                # This should not happen but handle gracefully
                relative_to_output = file.relative_to(output_dir)
                path_url = f"/training/{output_dir.name}/{relative_to_output.as_posix()}"

            samples_by_step[step].append({
                "sample_index": sample_idx,
                "path": path_url
            })

    # Sort by step and return
    samples = []
    for step in sorted(samples_by_step.keys()):
        samples.append({
            "step": step,
            "images": sorted(samples_by_step[step], key=lambda x: x["sample_index"])
        })

    return {"samples": samples}


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
