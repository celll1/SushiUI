"""
Image utilities for VAE decoding and caching.
"""

import torch
from pathlib import Path
from PIL import Image
from typing import Optional
import numpy as np


def decode_latent_to_image(
    vae,
    latent: torch.Tensor,
    device: str = "cuda",
) -> Image.Image:
    """
    Decode latent to PIL Image using VAE.

    Args:
        vae: AutoencoderKL instance
        latent: [1, 16, H, W] or [16, H, W] latent tensor
        device: Device to use

    Returns:
        PIL Image
    """
    # Ensure 4D tensor
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)  # [1, 16, H, W]

    latent = latent.to(device)

    # Decode
    with torch.no_grad():
        vae.to(device)
        image = vae.decode(latent, return_dict=False)[0]  # [1, 3, H*8, W*8]

    # Post-process
    image = (image / 2 + 0.5).clamp(0, 1)  # [-1, 1] → [0, 1]
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]  # [H, W, 3]
    image = (image * 255).astype(np.uint8)

    return Image.fromarray(image)


def decode_and_save_latent_pair(
    vae,
    true_latent: torch.Tensor,
    predicted_latent: torch.Tensor,
    output_dir: Path,
    record_id: int,
    device: str = "cuda",
) -> tuple[str, str]:
    """
    Decode true and predicted latents and save as images.

    Args:
        vae: AutoencoderKL instance
        true_latent: [1, 16, H, W] Ground truth latent
        predicted_latent: [1, 16, H, W] Predicted latent
        output_dir: Output directory for images
        record_id: LatentRecord ID (for filename)
        device: Device to use

    Returns:
        (true_image_path, predicted_image_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decode true latent
    true_image = decode_latent_to_image(vae, true_latent, device)
    true_image_path = output_dir / f"latent_{record_id:06d}_true.png"
    true_image.save(true_image_path)

    # Decode predicted latent
    predicted_image = decode_latent_to_image(vae, predicted_latent, device)
    predicted_image_path = output_dir / f"latent_{record_id:06d}_predicted.png"
    predicted_image.save(predicted_image_path)

    return str(true_image_path), str(predicted_image_path)


def batch_decode_latents(
    vae,
    record_ids: list[int],
    latent_files: list[Path],
    output_dir: Path,
    device: str = "cuda",
) -> dict[int, tuple[str, str]]:
    """
    Batch decode latents for multiple records.

    Args:
        vae: AutoencoderKL instance
        record_ids: List of LatentRecord IDs
        latent_files: List of .pt file paths
        output_dir: Output directory for images
        device: Device to use

    Returns:
        Dict mapping record_id → (true_image_path, predicted_image_path)
    """
    from tqdm import tqdm

    results = {}

    vae.to(device)

    for record_id, latent_file in tqdm(
        zip(record_ids, latent_files),
        desc="Decoding latents",
        total=len(record_ids),
    ):
        try:
            # Load latent data
            data = torch.load(latent_file, map_location="cpu")
            true_latent = data["latents"]
            predicted_latent = data["predicted_latent"]

            # Decode and save
            true_path, pred_path = decode_and_save_latent_pair(
                vae,
                true_latent,
                predicted_latent,
                output_dir,
                record_id,
                device,
            )

            results[record_id] = (true_path, pred_path)

        except Exception as e:
            print(f"[ImageUtils] Error decoding record {record_id}: {e}")
            continue

    return results
