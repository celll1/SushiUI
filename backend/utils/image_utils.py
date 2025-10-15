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
    generation_type: str = "txt2img"
) -> str:
    """Save image with EXIF metadata"""

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
    metadata.add_text("sampler", params.get("sampler", settings.default_sampler))
    metadata.add_text("cfg_scale", str(params.get("cfg_scale", settings.default_cfg_scale)))
    metadata.add_text("seed", str(seed))
    metadata.add_text("width", str(params.get("width", settings.default_width)))
    metadata.add_text("height", str(params.get("height", settings.default_height)))
    metadata.add_text("model", params.get("model", ""))
    metadata.add_text("generation_type", generation_type)

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
