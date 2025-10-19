from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import os
import sys
import subprocess
from PIL import Image
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor

from database import get_db
from database.models import GeneratedImage, UserSettings
from core.pipeline import pipeline_manager
from core.taesd import taesd_manager
from core.lora_manager import lora_manager
from core.controlnet_manager import controlnet_manager
from core.schedulers import (
    get_available_samplers,
    get_sampler_display_names,
    get_available_schedule_types,
    get_schedule_type_display_names
)
from utils import save_image_with_metadata, create_thumbnail, calculate_image_hash, encode_mask_to_base64, extract_lora_names
from config.settings import settings
from api.websocket import manager

router = APIRouter()

# Thread pool for running blocking operations
executor = ThreadPoolExecutor(max_workers=1)

# Pydantic models for requests
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
    use_input_image: bool = False  # For img2img/inpaint: use input image as control

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

class Txt2ImgRequest(GenerationParams):
    pass

class Img2ImgRequest(GenerationParams):
    denoising_strength: float = 0.75

# Routes
@router.post("/generate/txt2img")
async def generate_txt2img(request: Txt2ImgRequest, db: Session = Depends(get_db)):
    """Generate image from text"""
    try:
        # Generate image
        params = request.dict()
        # Log params without large base64 data
        params_for_log = params.copy()
        if "controlnets" in params_for_log and params_for_log["controlnets"]:
            params_for_log["controlnets"] = [
                {k: ("<base64_data>" if k == "image_base64" else v) for k, v in cn.items()}
                for cn in params_for_log["controlnets"]
            ]
        print(f"Generation params: {params_for_log}")

        # Set prompt chunking settings
        prompt_chunking_mode = params.get("prompt_chunking_mode", "a1111")
        max_prompt_chunks = params.get("max_prompt_chunks", 0)
        pipeline_manager.prompt_chunking_mode = prompt_chunking_mode
        pipeline_manager.max_prompt_chunks = max_prompt_chunks

        # Load LoRAs if specified
        lora_configs = params.get("loras", [])
        has_step_range_loras = False
        if lora_configs and pipeline_manager.txt2img_pipeline:
            print(f"Loading {len(lora_configs)} LoRA(s)...")
            pipeline_manager.txt2img_pipeline = lora_manager.load_loras(
                pipeline_manager.txt2img_pipeline,
                lora_configs
            )
            # Check if any LoRA has non-default step range
            has_step_range_loras = any(
                lora.get("step_range", [0, 1000]) != [0, 1000]
                for lora in lora_configs
            )

        # Load ControlNets if specified
        controlnet_configs = params.get("controlnets", [])
        controlnet_images = []
        if controlnet_configs:
            print(f"Processing {len(controlnet_configs)} ControlNet(s)...")
            import base64
            from io import BytesIO

            for idx, cn_config in enumerate(controlnet_configs):
                print(f"[ControlNet {idx}] model_path: {cn_config.get('model_path')}, has_image_base64: {bool(cn_config.get('image_base64'))}, use_input_image: {cn_config.get('use_input_image', False)}")

                # For txt2img, we must have image_base64 since there's no input image
                if cn_config.get("image_base64"):
                    try:
                        image_data = base64.b64decode(cn_config["image_base64"])
                        image = Image.open(BytesIO(image_data))
                        print(f"[ControlNet {idx}] Image decoded successfully: {image.size}")
                        controlnet_images.append({
                            "model_path": cn_config["model_path"],
                            "image": image,
                            "strength": cn_config.get("strength", 1.0),
                            "start_step": cn_config.get("start_step", 0.0),
                            "end_step": cn_config.get("end_step", 1.0),
                            "layer_weights": cn_config.get("layer_weights"),
                            "prompt": cn_config.get("prompt"),
                            "is_lllite": cn_config.get("is_lllite", False),
                        })
                    except Exception as e:
                        print(f"[ControlNet {idx}] Error decoding image: {e}")
                else:
                    print(f"[ControlNet {idx}] WARNING: No image_base64 provided for txt2img. ControlNet will be skipped.")
                    print(f"[ControlNet {idx}] For txt2img, you must provide image_base64. use_input_image is only for img2img/inpaint.")

        # Pass ControlNet images to params
        params["controlnet_images"] = controlnet_images
        print(f"[Routes] Total controlnet_images added to params: {len(controlnet_images)}")

        # Detect if SDXL
        is_sdxl = pipeline_manager.txt2img_pipeline is not None and \
                  "XL" in pipeline_manager.txt2img_pipeline.__class__.__name__

        # Progress callback to send updates via WebSocket
        def progress_callback(step, timestep, latents):
            total_steps = params.get("steps", 20)

            # Generate preview image from latent (every 5 steps to reduce overhead)
            preview_image = None
            if step % 5 == 0 or step == total_steps - 1:
                try:
                    preview_pil = taesd_manager.decode_latent(latents, is_sdxl=is_sdxl)
                    if preview_pil:
                        import base64
                        from io import BytesIO
                        buffered = BytesIO()
                        preview_pil.save(buffered, format="JPEG", quality=85)
                        preview_image = base64.b64encode(buffered.getvalue()).decode()
                except Exception as e:
                    print(f"Preview generation error: {e}")

            # Send synchronously from callback thread
            manager.send_progress_sync(step + 1, total_steps, f"Step {step + 1}/{total_steps}", preview_image=preview_image)

        # Create step callback for LoRA step range if needed
        step_callback = None
        if has_step_range_loras:
            total_steps = params.get("steps", 20)
            step_callback = lora_manager.create_step_callback(
                pipeline_manager.txt2img_pipeline,
                total_steps,
                original_callback=None
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
        image_hash = calculate_image_hash(image)
        lora_names = extract_lora_names(lora_configs)

        # Remove image objects from params before saving to DB and calculate ControlNet hashes
        params_for_db = params.copy()
        if "controlnet_images" in params_for_db:
            params_for_db["controlnet_images"] = [
                {
                    k: (calculate_image_hash(v) if k == "image" else v)
                    for k, v in cn.items()
                }
                for cn in params_for_db["controlnet_images"]
            ]

        # Extract model name and hash from current_model_info
        model_name = ""
        model_hash = ""
        if pipeline_manager.current_model_info:
            model_source = pipeline_manager.current_model_info.get("source", "")
            if model_source:
                model_name = os.path.basename(model_source)
            model_hash = pipeline_manager.current_model_info.get("model_hash", "")

        # Save to database
        db_image = GeneratedImage(
            filename=filename,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model_name=model_name,
            sampler=f"{request.sampler} ({request.schedule_type})",
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=actual_seed,
            ancestral_seed=request.ancestral_seed if (request.ancestral_seed != -1 and request.sampler in ["euler_a", "dpm2_a"]) else None,
            width=request.width,
            height=request.height,
            generation_type="txt2img",
            parameters=params_for_db,
            image_hash=image_hash,
            lora_names=lora_names if lora_names else None,
            model_hash=model_hash if model_hash else None,
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed}

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Error generating image: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))
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
    width: int = Form(1024),
    height: int = Form(1024),
    resize_mode: str = Form("image"),
    resampling_method: str = Form("lanczos"),
    loras: str = Form("[]"),  # JSON string of LoRA configs
    controlnets: str = Form("[]"),  # JSON string of ControlNet configs
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Generate image from image"""
    try:
        # Load input image
        image_data = await image.read()
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Parse LoRA configs
        import json
        lora_configs = json.loads(loras) if loras else []

        # Parse ControlNet configs
        controlnet_configs = json.loads(controlnets) if controlnets else []
        controlnet_images = []
        if controlnet_configs:
            print(f"Processing {len(controlnet_configs)} ControlNet(s)...")
            import base64
            from io import BytesIO

            for idx, cn_config in enumerate(controlnet_configs):
                print(f"[ControlNet {idx}] model_path: {cn_config.get('model_path')}, has_image_base64: {bool(cn_config.get('image_base64'))}, use_input_image: {cn_config.get('use_input_image', False)}")

                # Check if using input image
                if cn_config.get("use_input_image"):
                    print(f"[ControlNet {idx}] Using input image as control image")
                    controlnet_images.append({
                        "model_path": cn_config["model_path"],
                        "image": init_image.copy(),  # Use the input image
                        "strength": cn_config.get("strength", 1.0),
                        "start_step": cn_config.get("start_step", 0.0),
                        "end_step": cn_config.get("end_step", 1.0),
                        "layer_weights": cn_config.get("layer_weights"),
                        "prompt": cn_config.get("prompt"),
                        "is_lllite": cn_config.get("is_lllite", False),
                    })
                # Decode base64 image
                elif cn_config.get("image_base64"):
                    try:
                        image_data = base64.b64decode(cn_config["image_base64"])
                        cn_image = Image.open(BytesIO(image_data))
                        print(f"[ControlNet {idx}] Image decoded successfully: {cn_image.size}")
                        controlnet_images.append({
                            "model_path": cn_config["model_path"],
                            "image": cn_image,
                            "strength": cn_config.get("strength", 1.0),
                            "start_step": cn_config.get("start_step", 0.0),
                            "end_step": cn_config.get("end_step", 1.0),
                            "layer_weights": cn_config.get("layer_weights"),
                            "prompt": cn_config.get("prompt"),
                            "is_lllite": cn_config.get("is_lllite", False),
                        })
                    except Exception as e:
                        print(f"[ControlNet {idx}] Error decoding image: {e}")
                else:
                    print(f"[ControlNet {idx}] WARNING: No image_base64 provided for img2img. ControlNet will be skipped.")

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
            "width": width,
            "height": height,
            "resize_mode": resize_mode,
            "resampling_method": resampling_method,
            "controlnet_images": controlnet_images,
        }
        # Log params without large image objects
        params_for_log = params.copy()
        if "controlnet_images" in params_for_log and params_for_log["controlnet_images"]:
            params_for_log["controlnet_images"] = [
                {k: ("<PIL.Image>" if k == "image" else v) for k, v in cn.items()}
                for cn in params_for_log["controlnet_images"]
            ]
        print(f"img2img generation params: {params_for_log}")

        # Load LoRAs if specified
        has_step_range_loras = False
        if lora_configs and pipeline_manager.img2img_pipeline:
            print(f"Loading {len(lora_configs)} LoRA(s)...")
            pipeline_manager.img2img_pipeline = lora_manager.load_loras(
                pipeline_manager.img2img_pipeline,
                lora_configs
            )
            # Check if any LoRA has non-default step range
            has_step_range_loras = any(
                lora.get("step_range", [0, 1000]) != [0, 1000]
                for lora in lora_configs
            )

        # Detect if SDXL
        is_sdxl = pipeline_manager.img2img_pipeline is not None and \
                  "XL" in pipeline_manager.img2img_pipeline.__class__.__name__

        # Progress callback to send updates via WebSocket
        def progress_callback(step, timestep, latents):
            # Calculate actual steps based on img2img_fix_steps setting
            if img2img_fix_steps:
                # When fix_steps is enabled, we show the requested steps
                actual_steps = steps
            else:
                # Standard behavior: steps * strength
                actual_steps = int(steps * denoising_strength)

            # Generate preview image from latent
            preview_image = None
            if step % 5 == 0 or step == actual_steps - 1:
                try:
                    preview_pil = taesd_manager.decode_latent(latents, is_sdxl=is_sdxl)
                    if preview_pil:
                        import base64
                        from io import BytesIO
                        buffered = BytesIO()
                        preview_pil.save(buffered, format="JPEG", quality=85)
                        preview_image = base64.b64encode(buffered.getvalue()).decode()
                except Exception as e:
                    print(f"Preview generation error: {e}")

            manager.send_progress_sync(step + 1, actual_steps, f"Step {step + 1}/{actual_steps}", preview_image=preview_image)

        # Create step callback for LoRA step range if needed
        step_callback = None
        if has_step_range_loras:
            # Calculate actual steps based on denoising strength
            actual_steps = int(steps * denoising_strength)
            step_callback = lora_manager.create_step_callback(
                pipeline_manager.img2img_pipeline,
                actual_steps,
                original_callback=None
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
        image_hash = calculate_image_hash(result_image)
        source_image_hash = calculate_image_hash(init_image)
        lora_names = extract_lora_names(lora_configs)

        # Remove image objects from params before saving to DB and calculate ControlNet hashes
        params_for_db = params.copy()
        if "controlnet_images" in params_for_db:
            params_for_db["controlnet_images"] = [
                {
                    k: (calculate_image_hash(v) if k == "image" else v)
                    for k, v in cn.items()
                }
                for cn in params_for_db["controlnet_images"]
            ]

        # Extract model name and hash from current_model_info
        model_name = ""
        model_hash = ""
        if pipeline_manager.current_model_info:
            model_source = pipeline_manager.current_model_info.get("source", "")
            if model_source:
                model_name = os.path.basename(model_source)
            model_hash = pipeline_manager.current_model_info.get("model_hash", "")

        # Save to database
        ancestral_seed_value = params.get("ancestral_seed", -1)
        db_image = GeneratedImage(
            filename=filename,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_name=model_name,
            sampler=f"{sampler} ({schedule_type})",
            steps=steps,
            cfg_scale=cfg_scale,
            seed=actual_seed,
            ancestral_seed=ancestral_seed_value if (ancestral_seed_value != -1 and sampler in ["euler_a", "dpm2_a"]) else None,
            width=result_image.width,
            height=result_image.height,
            generation_type="img2img",
            parameters=params_for_db,
            image_hash=image_hash,
            source_image_hash=source_image_hash,
            lora_names=lora_names if lora_names else None,
            model_hash=model_hash if model_hash else None,
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed}

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Error generating img2img: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))
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
    width: int = Form(1024),
    height: int = Form(1024),
    mask_blur: int = Form(4),
    inpaint_full_res: bool = Form(False),
    inpaint_full_res_padding: int = Form(32),
    loras: str = Form("[]"),  # JSON string of LoRA configs
    controlnets: str = Form("[]"),  # JSON string of ControlNet configs
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Generate inpainted image"""
    try:
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
        controlnet_images = []
        if controlnet_configs:
            print(f"Processing {len(controlnet_configs)} ControlNet(s)...")
            import base64
            from io import BytesIO

            for idx, cn_config in enumerate(controlnet_configs):
                print(f"[ControlNet {idx}] model_path: {cn_config.get('model_path')}, has_image_base64: {bool(cn_config.get('image_base64'))}, use_input_image: {cn_config.get('use_input_image', False)}")

                # Check if using input image
                if cn_config.get("use_input_image"):
                    print(f"[ControlNet {idx}] Using input image as control image")
                    controlnet_images.append({
                        "model_path": cn_config["model_path"],
                        "image": init_image.copy(),  # Use the input image
                        "strength": cn_config.get("strength", 1.0),
                        "start_step": cn_config.get("start_step", 0.0),
                        "end_step": cn_config.get("end_step", 1.0),
                        "layer_weights": cn_config.get("layer_weights"),
                        "prompt": cn_config.get("prompt"),
                        "is_lllite": cn_config.get("is_lllite", False),
                    })
                # Decode base64 image
                elif cn_config.get("image_base64"):
                    try:
                        image_data = base64.b64decode(cn_config["image_base64"])
                        cn_image = Image.open(BytesIO(image_data))
                        print(f"[ControlNet {idx}] Image decoded successfully: {cn_image.size}")
                        controlnet_images.append({
                            "model_path": cn_config["model_path"],
                            "image": cn_image,
                            "strength": cn_config.get("strength", 1.0),
                            "start_step": cn_config.get("start_step", 0.0),
                            "end_step": cn_config.get("end_step", 1.0),
                            "layer_weights": cn_config.get("layer_weights"),
                            "prompt": cn_config.get("prompt"),
                            "is_lllite": cn_config.get("is_lllite", False),
                        })
                    except Exception as e:
                        print(f"[ControlNet {idx}] Error decoding image: {e}")
                else:
                    print(f"[ControlNet {idx}] WARNING: No image_base64 provided for inpaint. ControlNet will be skipped.")

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
            "width": width,
            "height": height,
            "mask_blur": mask_blur,
            "inpaint_full_res": inpaint_full_res,
            "inpaint_full_res_padding": inpaint_full_res_padding,
            "controlnet_images": controlnet_images,
        }
        # Log params without large image objects
        params_for_log = params.copy()
        if "controlnet_images" in params_for_log and params_for_log["controlnet_images"]:
            params_for_log["controlnet_images"] = [
                {k: ("<PIL.Image>" if k == "image" else v) for k, v in cn.items()}
                for cn in params_for_log["controlnet_images"]
            ]
        print(f"inpaint generation params: {params_for_log}")

        # Load LoRAs if specified
        has_step_range_loras = False
        if lora_configs and pipeline_manager.inpaint_pipeline:
            print(f"Loading {len(lora_configs)} LoRA(s)...")
            pipeline_manager.inpaint_pipeline = lora_manager.load_loras(
                pipeline_manager.inpaint_pipeline,
                lora_configs
            )
            # Check if any LoRA has non-default step range
            has_step_range_loras = any(
                lora.get("step_range", [0, 1000]) != [0, 1000]
                for lora in lora_configs
            )

        # Detect if SDXL
        is_sdxl = pipeline_manager.inpaint_pipeline is not None and \
                  "XL" in pipeline_manager.inpaint_pipeline.__class__.__name__

        # Progress callback to send updates via WebSocket
        def progress_callback(step, timestep, latents):
            # Calculate actual steps based on img2img_fix_steps setting
            if img2img_fix_steps:
                # When fix_steps is enabled, we show the requested steps
                actual_steps = steps
            else:
                # Standard behavior: steps * strength
                actual_steps = int(steps * denoising_strength)

            # Generate preview image from latent (every 5 steps to reduce overhead)
            preview_image = None
            if step % 5 == 0 or step == actual_steps - 1:
                try:
                    preview_pil = taesd_manager.decode_latent(latents, is_sdxl=is_sdxl)
                    if preview_pil:
                        import base64
                        from io import BytesIO
                        buffered = BytesIO()
                        preview_pil.save(buffered, format="JPEG", quality=85)
                        preview_image = base64.b64encode(buffered.getvalue()).decode()
                except Exception as e:
                    print(f"Preview generation error: {e}")

            manager.send_progress_sync(step + 1, actual_steps, f"Step {step + 1}/{actual_steps}", preview_image=preview_image)

        # Create step callback for LoRA step range if needed
        step_callback = None
        if has_step_range_loras:
            # Calculate actual steps based on denoising strength
            actual_steps = int(steps * denoising_strength)
            step_callback = lora_manager.create_step_callback(
                pipeline_manager.inpaint_pipeline,
                actual_steps,
                original_callback=None
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
        image_hash = calculate_image_hash(result_image)
        source_image_hash = calculate_image_hash(init_image)
        mask_data_base64 = encode_mask_to_base64(mask_image)
        lora_names = extract_lora_names(lora_configs)

        # Remove image objects from params before saving to DB and calculate ControlNet hashes
        params_for_db = params.copy()
        if "controlnet_images" in params_for_db:
            params_for_db["controlnet_images"] = [
                {
                    k: (calculate_image_hash(v) if k == "image" else v)
                    for k, v in cn.items()
                }
                for cn in params_for_db["controlnet_images"]
            ]

        # Extract model name and hash from current_model_info
        model_name = ""
        model_hash = ""
        if pipeline_manager.current_model_info:
            model_source = pipeline_manager.current_model_info.get("source", "")
            if model_source:
                model_name = os.path.basename(model_source)
            model_hash = pipeline_manager.current_model_info.get("model_hash", "")

        # Save to database
        ancestral_seed_value = params.get("ancestral_seed", -1)
        db_image = GeneratedImage(
            filename=filename,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_name=model_name,
            sampler=f"{sampler} ({schedule_type})",
            steps=steps,
            cfg_scale=cfg_scale,
            seed=actual_seed,
            ancestral_seed=ancestral_seed_value if (ancestral_seed_value != -1 and sampler in ["euler_a", "dpm2_a"]) else None,
            width=result_image.width,
            height=result_image.height,
            generation_type="inpaint",
            parameters=params_for_db,
            image_hash=image_hash,
            source_image_hash=source_image_hash,
            mask_data=mask_data_base64,
            lora_names=lora_names if lora_names else None,
            model_hash=model_hash if model_hash else None,
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        return {"success": True, "image": db_image.to_dict(), "actual_seed": actual_seed}

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Error generating inpaint: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))
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
    db: Session = Depends(get_db)
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
async def get_image(image_id: int, db: Session = Depends(get_db)):
    """Get single image details"""
    image = db.query(GeneratedImage).filter(GeneratedImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image.to_dict()

@router.delete("/images/{image_id}")
async def delete_image(image_id: int, db: Session = Depends(get_db)):
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
async def get_models(db: Session = Depends(get_db)):
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
    """Get available samplers"""
    samplers = get_available_samplers()
    display_names = get_sampler_display_names()
    return {
        "samplers": [
            {"id": sampler_id, "name": display_names.get(sampler_id, sampler_id)}
            for sampler_id in samplers
        ]
    }

@router.get("/schedule-types")
async def get_schedule_types():
    """Get available schedule types"""
    schedule_types = get_available_schedule_types()
    display_names = get_schedule_type_display_names()
    return {
        "schedule_types": [
            {"id": schedule_id, "name": display_names.get(schedule_id, schedule_id)}
            for schedule_id in schedule_types
        ]
    }

@router.get("/loras")
async def get_loras():
    """Get available LoRA files"""
    loras = lora_manager.get_available_loras()
    return {
        "loras": [
            {"path": lora, "name": os.path.basename(lora)}
            for lora in loras
        ]
    }

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
    controlnets = controlnet_manager.get_available_controlnets()
    return {
        "controlnets": [
            {"path": cn, "name": os.path.basename(cn)}
            for cn in controlnets
        ]
    }

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
async def get_directory_settings(db: Session = Depends(get_db)):
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
    db: Session = Depends(get_db)
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
