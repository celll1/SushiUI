"""
DEUS Architecture Inference Pipeline

Complete pipeline for text-to-image generation with the DEUS architecture.
(Dual-Embeddings U-Net Structure)

Components:
- Text Encoder: SigLIP-2
- Image Encoder: SigLIP-2 (optional)
- U-Net: DEUS architecture with RoPE and sparse skip connections
- VAE: SDXL VAE (4-channel latents)
- Scheduler: Standard diffusion schedulers (Euler, DPM++, etc.)

Supports:
- T2I: Text-only generation
- I2I: Image-to-image with optional text
- TI2I: Text + Image instruction-based generation
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Callable, Any
from PIL import Image
import numpy as np
from tqdm import tqdm

from ..models.siglip2_wrapper import SigLIP2MultiModalEncoder
from ..models.sdxl_vae_wrapper import SDXLVAEWrapper
from ..models.unet_deus import DeusUNet, UNetConfig


class DeusPipeline(nn.Module):
    """
    Complete inference pipeline for DEUS architecture.
    (Dual-Embeddings U-Net Structure)
    """

    def __init__(
        self,
        unet: DeusUNet,
        vae: SDXLVAEWrapper,
        encoder: SigLIP2MultiModalEncoder,
        scheduler: Optional[Any] = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        super().__init__()

        self.unet = unet
        self.vae = vae
        self.encoder = encoder
        self.scheduler = scheduler
        self.dtype = dtype
        self.device_name = device

        # Keep components on CPU by default (pipeline.py will manage device placement)
        self.unet = self.unet.to("cpu")
        self.vae.to("cpu")
        # encoder already on CPU (handled by SigLIP2MultiModalEncoder)

        print(f"[Pipeline] DEUS pipeline initialized (components on CPU):")
        print(f"  Device: {device}")
        print(f"  Dtype: {dtype}")
        print(f"  U-Net variant: {unet.config.variant}")
        print(f"  VAE latent channels: {vae.latent_channels}")
        print(f"  Encoder hidden size: {encoder.hidden_size}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        clip_skip: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Image.Image]:
        """
        Generate images from text and optional images.

        Args:
            prompt: Text prompt(s)
            images: Optional input images (for I2I/TI2I modes)
            height: Output height
            width: Output width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt(s)
            num_images_per_prompt: Number of images to generate per prompt
            seed: Random seed
            clip_skip: Number of layers to skip from the end (default: 1=penultimate/layer 26)
            progress_callback: Callback for progress updates

        Returns:
            List of generated PIL Images
        """
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Convert single prompt to list
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt) * num_images_per_prompt

        # Encode prompt (text + optional images)
        print(f"[Pipeline] Encoding prompt (clip_skip={clip_skip})...")

        # For CFG, encode both prompts with new token format
        if guidance_scale > 1.0:
            if negative_prompt is None:
                negative_prompt = [""] * len(prompt)
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]

            batch_size_single = len(prompt)

            # Encode with new format: <text> [IMG0] <image> [END] for TI2I, <text> [END] for T2I
            if images is None:
                # T2I: Use [END] token only (no images)
                encoder_hidden_states = self.encoder.encode(
                    prompts=prompt,
                    images=None,
                    use_end_token=True,
                    clip_skip=clip_skip
                )
                negative_encoder_hidden_states = self.encoder.encode(
                    prompts=negative_prompt,
                    images=None,
                    use_end_token=True,
                    clip_skip=clip_skip
                )
            else:
                # TI2I: Use [IMG0] token + image + [END] token
                if not isinstance(images, list):
                    images = [images]
                encoder_hidden_states = self.encoder.encode(
                    prompts=prompt,
                    images=images,
                    use_end_token=True,
                    clip_skip=clip_skip
                )
                # Negative: Use [END] token only (no images for negative)
                negative_encoder_hidden_states = self.encoder.encode(
                    prompts=negative_prompt,
                    images=None,
                    use_end_token=True,
                    clip_skip=clip_skip
                )

            # Pad negative embeddings to match positive if needed (TI2I has longer positive)
            if encoder_hidden_states.shape[1] != negative_encoder_hidden_states.shape[1]:
                seq_len_diff = encoder_hidden_states.shape[1] - negative_encoder_hidden_states.shape[1]
                if seq_len_diff > 0:
                    padding = torch.zeros(
                        (negative_encoder_hidden_states.shape[0], seq_len_diff, negative_encoder_hidden_states.shape[2]),
                        dtype=negative_encoder_hidden_states.dtype,
                        device=negative_encoder_hidden_states.device
                    )
                    negative_encoder_hidden_states = torch.cat([negative_encoder_hidden_states, padding], dim=1)

            # Repeat for num_images_per_prompt
            if num_images_per_prompt > 1:
                negative_encoder_hidden_states = negative_encoder_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)

            # Concatenate for CFG
            encoder_hidden_states = torch.cat([negative_encoder_hidden_states, encoder_hidden_states])
        else:
            # No CFG, just encode positive prompt with [END] token
            encoder_hidden_states = self.encoder.encode(
                prompts=prompt,
                images=images,
                use_end_token=True,
                clip_skip=clip_skip
            )

            # Repeat for num_images_per_prompt
            if num_images_per_prompt > 1:
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)

            negative_encoder_hidden_states = None

        # Prepare latents
        latent_height = height // 8
        latent_width = width // 8
        latent_channels = self.vae.latent_channels  # 16

        latents = torch.randn(
            batch_size,
            latent_channels,
            latent_height,
            latent_width,
            dtype=self.dtype,
            device=self.device_name
        )

        # Get scheduler (use simple DDPM for now)
        if self.scheduler is None:
            # Simple linear schedule
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        else:
            timesteps = self.scheduler.timesteps

        # Denoising loop
        print(f"[Pipeline] Denoising ({num_inference_steps} steps)...")

        # Debug: Initial VRAM
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[VRAM] Before denoising: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            # Expand latents for CFG
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            # Prepare timestep
            if isinstance(t, float):
                t_tensor = torch.tensor([t], dtype=self.dtype, device=self.device_name)
            else:
                t_tensor = t.to(self.device_name)

            # Predict noise
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t_tensor,
                encoder_hidden_states=encoder_hidden_states
            )

            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Simple Euler step (for now)
            if i < len(timesteps) - 1:
                dt = timesteps[i] - timesteps[i + 1]
            else:
                dt = timesteps[i]

            latents = latents - noise_pred * dt

            # Debug: VRAM per step
            if torch.cuda.is_available() and i % 5 == 0:
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[VRAM] Step {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            # Progress callback
            if progress_callback is not None:
                progress_callback(i + 1, num_inference_steps)

        # Decode latents
        print(f"[Pipeline] Decoding latents...")
        images = self.vae.decode(latents)

        # Convert to PIL
        images = self.tensor_to_pil(images)

        return images

    @staticmethod
    def tensor_to_pil(images: torch.Tensor) -> List[Image.Image]:
        """
        Convert tensor to PIL Images.

        Args:
            images: Tensor [batch, 3, height, width] in range [-1, 1]

        Returns:
            List of PIL Images
        """
        # Denormalize: [-1, 1] -> [0, 1]
        images = (images + 1.0) / 2.0

        # Clamp
        images = torch.clamp(images, 0, 1)

        # To numpy: [batch, 3, H, W] -> [batch, H, W, 3]
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        # To uint8
        images = (images * 255).round().astype(np.uint8)

        # Convert to PIL
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images


def create_deus_pipeline(
    unet_variant: str = "medium",
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> DeusPipeline:
    """
    Create a new DEUS pipeline with randomly initialized weights.

    This is for testing the architecture. The U-Net is randomly initialized,
    so it will not produce meaningful images yet.

    Args:
        unet_variant: "small", "medium", or "large"
        dtype: Model dtype
        device: Device to load on

    Returns:
        DeusPipeline instance
    """
    print(f"[Pipeline] Creating DEUS pipeline ({unet_variant})...")

    # Create encoder
    encoder = SigLIP2MultiModalEncoder(
        dtype=dtype,
        device=device
    )

    # Create VAE
    vae = SDXLVAEWrapper(
        dtype=dtype,
        device=device
    )

    # Create U-Net
    config = UNetConfig.from_variant(unet_variant)
    unet = DeusUNet(config)
    unet = unet.to(dtype).to(device)

    # Create scheduler (using Euler Ancestral Discrete Scheduler as default)
    from diffusers import EulerAncestralDiscreteScheduler
    scheduler = EulerAncestralDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon"
    )

    # Create pipeline
    pipeline = DeusPipeline(
        unet=unet,
        vae=vae,
        encoder=encoder,
        scheduler=scheduler,
        dtype=dtype,
        device=device
    )

    print(f"[Pipeline] Pipeline created successfully!")
    print(f"[Pipeline] NOTE: U-Net is randomly initialized and will produce random noise.")

    return pipeline


def load_deus_pipeline_from_checkpoint(
    checkpoint_path: str,
    unet_variant: str = "medium",
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> DeusPipeline:
    """
    Load DEUS pipeline from unified checkpoint.

    Args:
        checkpoint_path: Path to unified safetensors checkpoint
        unet_variant: U-Net variant (for structure creation)
        dtype: Model dtype
        device: Device to load on

    Returns:
        DeusPipeline instance
    """
    from ..models.checkpoint_utils import load_unified_checkpoint

    print(f"[Pipeline] Loading DEUS pipeline from checkpoint...")

    # Load checkpoint
    components = load_unified_checkpoint(
        checkpoint_path=checkpoint_path,
        unet_variant=unet_variant,
        device=device,
        dtype=dtype,
        load_text_encoder=True,
        load_image_encoder=True,
        load_vae=True
    )

    # Get components
    unet = components["unet"]
    text_encoder = components["text_encoder"]
    image_encoder = components["image_encoder"]
    vae = components["vae"]

    # Create multi-modal encoder wrapper
    from ..models.siglip2_wrapper import SigLIP2MultiModalEncoder

    # Create encoder (will use checkpoint encoders if available, create missing ones)
    if text_encoder is not None or image_encoder is not None:
        print(f"[Pipeline] Using available encoders from checkpoint...")
        encoder = SigLIP2MultiModalEncoder(
            dtype=dtype,
            device=device,
            text_encoder=text_encoder,
            image_encoder=image_encoder
        )
    else:
        # Both encoders are missing, create new ones
        print(f"[Pipeline] Creating new encoders (not in checkpoint)...")
        encoder = SigLIP2MultiModalEncoder(dtype=dtype, device=device)

    # Create VAE if not in checkpoint
    if vae is None:
        print(f"[Pipeline] Creating new VAE (not in checkpoint)...")
        vae = SDXLVAEWrapper(dtype=dtype, device=device)

    # Create scheduler (using Euler Ancestral Discrete Scheduler as default)
    from diffusers import EulerAncestralDiscreteScheduler
    scheduler = EulerAncestralDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon"
    )

    # Create pipeline
    pipeline = DeusPipeline(
        unet=unet,
        vae=vae,
        encoder=encoder,
        scheduler=scheduler,
        dtype=dtype,
        device=device
    )

    print(f"[Pipeline] Pipeline loaded successfully!")

    return pipeline


if __name__ == "__main__":
    # Test pipeline creation
    pipeline = create_deus_pipeline(unet_variant="small")

    # Test inference
    images = pipeline(
        prompt="a beautiful landscape",
        height=512,
        width=512,
        num_inference_steps=10,
        guidance_scale=1.0  # No CFG for faster testing
    )

    print(f"Generated {len(images)} images")
    images[0].save("test_original_pipeline.png")
    print(f"Saved test image to test_original_pipeline.png")
