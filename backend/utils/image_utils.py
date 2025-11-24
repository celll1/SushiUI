from PIL import Image, PngImagePlugin
from typing import Dict, Any, Optional
import os
import hashlib
import base64
from io import BytesIO
from datetime import datetime
from config.settings import settings

def save_image_with_metadata(
    image: Image.Image,
    params: Dict[str, Any],
    generation_type: str = "txt2img",
    model_info: Optional[Dict[str, Any]] = None
) -> str:
    """Save image with EXIF metadata

    Args:
        image: PIL Image to save
        params: Generation parameters
        generation_type: Type of generation (txt2img, img2img, inpaint)
        model_info: Model information (source, source_type, hash)
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

    # Add NAG (Normalized Attention Guidance) parameters
    nag_enable = params.get("nag_enable", False)
    if nag_enable:
        metadata.add_text("nag_enable", str(nag_enable))
        metadata.add_text("nag_scale", str(params.get("nag_scale", 5.0)))
        metadata.add_text("nag_tau", str(params.get("nag_tau", 3.5)))
        metadata.add_text("nag_alpha", str(params.get("nag_alpha", 0.25)))
        metadata.add_text("nag_sigma_end", str(params.get("nag_sigma_end", 3.0)))

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
    """Create thumbnail from image"""
    os.makedirs(settings.thumbnails_dir, exist_ok=True)

    image = Image.open(image_path)
    image.thumbnail(size, Image.Resampling.LANCZOS)

    filename = os.path.basename(image_path)
    thumb_path = os.path.join(settings.thumbnails_dir, filename)
    image.save(thumb_path)

    return thumb_path

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
