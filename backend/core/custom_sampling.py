"""Custom sampling loop for advanced prompt control

This module provides a custom sampling loop that allows:
- Prompt editing (changing prompts mid-generation)
- Fine-grained control over each denoising step
- Access to intermediate latents

Based on diffusers' pipeline implementation but with added flexibility.
"""

import torch
from typing import Optional, Callable, Dict, Any, Union, List
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image
import numpy as np
import math
from math import pi, cos


def calculate_cfg_metrics(noise_pred_uncond: torch.Tensor, noise_pred_text: torch.Tensor, guidance_scale: float, developer_mode: bool = False) -> Optional[Dict]:
    """Calculate CFG metrics for developer mode visualization

    Key metrics:
    - cosine_similarity: Direction similarity between yp and yn (-1 to 1, closer to 1 = more similar)
    - relative_diff: ||yp - yn|| / ||yn|| (relative strength of CFG direction)
    - snr: Signal-to-noise ratio = ||yp - yn||² / ||yn||² (squared relative strength)
    """
    if not developer_mode:
        return None

    # Calculate L2 norms (magnitude of vectors)
    uncond_norm = torch.norm(noise_pred_uncond).item()
    text_norm = torch.norm(noise_pred_text).item()
    diff = noise_pred_text - noise_pred_uncond
    diff_norm = torch.norm(diff).item()

    # Relative difference: how much CFG will change the prediction
    # This is more meaningful than absolute norms
    relative_diff = diff_norm / uncond_norm if uncond_norm > 1e-8 else 0.0

    # SNR (Signal-to-Noise Ratio): squared relative difference
    snr = (diff_norm ** 2) / (uncond_norm ** 2) if uncond_norm > 1e-8 else 0.0

    # Per-channel statistics to see variation patterns
    uncond_mean = noise_pred_uncond.mean().item()
    text_mean = noise_pred_text.mean().item()
    diff_mean = diff.mean().item()
    uncond_std = noise_pred_uncond.std().item()
    text_std = noise_pred_text.std().item()
    diff_std = diff.std().item()

    return {
        # Primary metrics (most important for understanding CFG)
        'relative_diff': round(relative_diff, 6),
        'snr': round(snr, 6),

        # L2 norms (for reference)
        'uncond_norm': round(uncond_norm, 4),
        'text_norm': round(text_norm, 4),
        'diff_norm': round(diff_norm, 4),

        # Statistics
        'uncond_mean': round(uncond_mean, 6),
        'text_mean': round(text_mean, 6),
        'diff_mean': round(diff_mean, 6),
        'uncond_std': round(uncond_std, 6),
        'text_std': round(text_std, 6),
        'diff_std': round(diff_std, 6),

        'guidance_scale': guidance_scale,
    }


def calculate_dynamic_cfg(
    sigma: float,
    sigma_max: float,
    cfg_base: float,
    cfg_schedule_type: str = "constant",
    cfg_schedule_min: float = 1.0,
    cfg_schedule_max: Optional[float] = None,
    cfg_schedule_power: float = 2.0,
    snr: Optional[float] = None,
    cfg_rescale_snr_alpha: float = 0.0,
) -> float:
    """Calculate dynamic CFG scale based on sigma (noise level) and optionally SNR

    Args:
        sigma: Current noise level
        sigma_max: Maximum sigma value (from scheduler)
        cfg_base: Base CFG scale (used when schedule_type is "constant")
        cfg_schedule_type: Type of schedule ("constant", "linear", "quadratic", "cosine", "snr_based")
        cfg_schedule_min: Minimum CFG scale (at sigma=0, end of generation)
        cfg_schedule_max: Maximum CFG scale (at sigma=sigma_max, start of generation)
                          If None, uses cfg_base
        cfg_schedule_power: Power for quadratic schedule (default: 2.0)
        snr: Signal-to-Noise Ratio from CFG metrics (optional, for SNR-based scheduling)
        cfg_rescale_snr_alpha: Alpha parameter for SNR rescaling (0.0 = disabled)

    Returns:
        CFG scale for current step
    """
    if cfg_schedule_type == "constant":
        return cfg_base

    # Use cfg_base as max if not specified
    if cfg_schedule_max is None:
        cfg_schedule_max = cfg_base

    # Normalize sigma to [0, 1] range
    sigma_norm = min(sigma / sigma_max, 1.0) if sigma_max > 0 else 0.0

    # Calculate CFG based on schedule type
    if cfg_schedule_type == "linear":
        # Linear interpolation: high CFG at start (high sigma), low at end
        cfg = cfg_schedule_min + (cfg_schedule_max - cfg_schedule_min) * sigma_norm
    elif cfg_schedule_type == "quadratic":
        # Quadratic: more gradual at start, steeper drop at end
        cfg = cfg_schedule_min + (cfg_schedule_max - cfg_schedule_min) * (sigma_norm ** cfg_schedule_power)
    elif cfg_schedule_type == "cosine":
        # Cosine: smooth transition
        cfg = cfg_schedule_min + (cfg_schedule_max - cfg_schedule_min) * cos((1 - sigma_norm) * pi / 2)
    elif cfg_schedule_type == "snr_based" and snr is not None:
        # SNR-based adaptive CFG: reduce CFG when SNR is high
        # cfg = cfg_base / (1 + alpha * sqrt(SNR))
        import math
        snr_sqrt = math.sqrt(max(snr, 0))
        cfg = cfg_base / (1.0 + cfg_rescale_snr_alpha * snr_sqrt)
        # Clamp to min/max range
        cfg = max(cfg_schedule_min, min(cfg_schedule_max if cfg_schedule_max else cfg_base, cfg))
    else:
        # Fallback to constant
        cfg = cfg_base

    return cfg


def rescale_noise_cfg(noise_cfg: torch.Tensor, noise_pred_text: torch.Tensor, guidance_rescale: float = 0.0) -> torch.Tensor:
    """
    Rescale noise predictions to fix overexposure and improve image quality.

    Based on Section 3.4 from "Common Diffusion Noise Schedules and Sample Steps are Flawed"
    https://arxiv.org/abs/2305.08891

    This is particularly important for v-prediction models to avoid washed out or blurry images.

    Args:
        noise_cfg: The predicted noise tensor after CFG (classifier-free guidance)
        noise_pred_text: The predicted noise tensor from text conditioning only (before CFG)
        guidance_rescale: Rescale factor (0.0 = no rescaling, 0.7 = recommended for v-pred)

    Returns:
        Rescaled noise prediction tensor
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # Rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # Mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def dynamic_thresholding(
    noise_pred: torch.Tensor,
    percentile: float = 99.5,
    clamp_value: float = 1.0
) -> torch.Tensor:
    """
    Apply dynamic thresholding to prevent CFG from causing extreme values.

    Based on Imagen paper (https://arxiv.org/abs/2205.11487):
    "We use a dynamic thresholding mechanism where we set s to a certain percentile
    absolute pixel value in x_t for each sample. We then threshold x_t to the range
    [-s, s] and then divide by s."

    Args:
        noise_pred: Noise prediction tensor after CFG
        percentile: Percentile to use for dynamic threshold (default: 99.5)
        clamp_value: Minimum threshold value (prevents over-clamping, default: 1.0)

    Returns:
        Thresholded noise prediction tensor
    """
    batch_size = noise_pred.shape[0]
    original_dtype = noise_pred.dtype

    # Flatten all dimensions except batch for per-sample thresholding
    noise_flat = noise_pred.reshape(batch_size, -1)

    # Calculate dynamic threshold as percentile of absolute values
    # Convert to float32 for quantile (doesn't support float16)
    abs_noise = torch.abs(noise_flat).float()
    s = torch.quantile(abs_noise, percentile / 100.0, dim=1, keepdim=True)
    s = s.to(original_dtype)

    # Apply static threshold: s = max(s, clamp_value)
    # This ensures s is at least clamp_value (typically 1.0)
    s = torch.maximum(s, torch.tensor(clamp_value, device=noise_pred.device, dtype=original_dtype))

    # Reshape for broadcasting
    s = s.reshape(batch_size, *([1] * (noise_pred.ndim - 1)))

    # Imagen dynamic thresholding: simply clamp to [-s, s]
    # This prevents extreme values while preserving most of the signal
    noise_pred = torch.clamp(noise_pred, -s, s)

    return noise_pred


def custom_sampling_loop(
    pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    guidance_rescale: float = 0.0,
    width: int = 512,
    height: int = 512,
    generator: Optional[torch.Generator] = None,
    ancestral_generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds_callback: Optional[Callable[[int], tuple]] = None,
    progress_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    step_callback: Optional[Callable[[Any, int, int, Dict], Dict]] = None,
    developer_mode: bool = False,
    controlnet_images: Optional[List[Image.Image]] = None,
    controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
    control_guidance_start: Optional[Union[float, List[float]]] = None,
    control_guidance_end: Optional[Union[float, List[float]]] = None,
    cfg_schedule_type: str = "constant",
    cfg_schedule_min: float = 1.0,
    cfg_schedule_max: Optional[float] = None,
    cfg_schedule_power: float = 2.0,
    cfg_rescale_snr_alpha: float = 0.0,  # SNR-based adaptive CFG (0.0 = disabled)
    dynamic_threshold_percentile: float = 0.0,  # 0.0 = disabled, 99.5 = typical value
    dynamic_threshold_mimic_scale: float = 1.0,  # Clamp value for static threshold
    nag_enable: bool = False,  # Enable NAG (Normalized Attention Guidance)
    nag_scale: float = 5.0,  # NAG extrapolation scale (similar to CFG scale, typical: 3-7)
    nag_tau: float = 3.5,  # NAG normalization threshold (typical: 2.5-3.5)
    nag_alpha: float = 0.25,  # NAG blending factor (typical: 0.25-0.5)
    nag_sigma_end: float = 0.0,  # Sigma threshold to disable NAG (0.0 = always enabled)
    nag_negative_prompt_embeds: Optional[torch.Tensor] = None,  # Separate negative embeds for NAG
    nag_negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # Separate pooled embeds for NAG (SDXL)
    attention_type: str = "normal",  # Attention backend - "normal", "sage", or "flash"
) -> Image.Image:
    """Custom sampling loop with prompt editing and ControlNet support

    Args:
        pipeline: The diffusers pipeline (SD or SDXL)
        prompt_embeds: Initial prompt embeddings [batch, seq_len, hidden_size]
        negative_prompt_embeds: Initial negative prompt embeddings
        pooled_prompt_embeds: Pooled prompt embeds (SDXL only)
        negative_pooled_prompt_embeds: Negative pooled embeds (SDXL only)
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        width: Output width
        height: Output height
        generator: Random generator for initial latent generation
        ancestral_generator: Separate generator for stochastic samplers (Euler a, etc.). If None, uses generator.
        latents: Initial latents (optional)
        prompt_embeds_callback: Callback to get new embeddings at each step
            Called with (step_index) -> (prompt_embeds, negative_prompt_embeds, pooled, neg_pooled)
        progress_callback: Callback for progress updates (step, total, latents)
        step_callback: Callback after each step for custom processing
        controlnet_images: List of control images for ControlNet
        controlnet_conditioning_scale: Strength of ControlNet conditioning (float or list)
        control_guidance_start: When to start ControlNet guidance (0.0-1.0, float or list)
        control_guidance_end: When to end ControlNet guidance (0.0-1.0, float or list)

    Returns:
        Generated PIL Image
    """
    device = pipeline.device
    # Use unet's dtype for consistency with diffusers
    dtype = pipeline.unet.dtype

    # Check if SDXL by checking if text_encoder_2 exists (more reliable than isinstance for ControlNet pipelines)
    is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None
    print(f"[CustomSampling] Pipeline type: {type(pipeline).__name__}, is_sdxl: {is_sdxl}")

    # Use ancestral_generator for stochastic samplers, fallback to generator if not provided
    step_generator = ancestral_generator if ancestral_generator is not None else generator
    if ancestral_generator is not None:
        print(f"[CustomSampling] Using separate ancestral generator for stochastic sampler")

    # Get components
    unet = pipeline.unet
    vae = pipeline.vae
    scheduler = pipeline.scheduler

    # Check if ControlNet is present
    controlnet = getattr(pipeline, 'controlnet', None)
    has_controlnet = controlnet is not None and controlnet_images is not None

    if has_controlnet:
        print(f"[CustomSampling] ControlNet detected, preparing control images")
        # Prepare control images
        if not isinstance(controlnet_images, list):
            controlnet_images = [controlnet_images]

        # Convert PIL images to tensors
        control_image_tensors = []
        for img in controlnet_images:
            if isinstance(img, Image.Image):
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                img = torch.from_numpy(np.array(img)).float() / 255.0
                if img.ndim == 2:  # Grayscale
                    img = img.unsqueeze(-1).repeat(1, 1, 3)
                img = img.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            control_image_tensors.append(img.to(device=device, dtype=dtype))

        # Normalize conditioning scales
        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = 1.0
        if not isinstance(controlnet_conditioning_scale, list):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(control_image_tensors)

        # Normalize guidance ranges
        if control_guidance_start is None:
            control_guidance_start = 0.0
        if not isinstance(control_guidance_start, list):
            control_guidance_start = [control_guidance_start] * len(control_image_tensors)

        if control_guidance_end is None:
            control_guidance_end = 1.0
        if not isinstance(control_guidance_end, list):
            control_guidance_end = [control_guidance_end] * len(control_image_tensors)

    # Setup NAG if enabled
    nag_active = nag_enable and nag_negative_prompt_embeds is not None
    original_processors = None

    if nag_active:
        from core.nag_processor import set_nag_processors
        print(f"[CustomSampling] NAG enabled: scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}, sigma_end={nag_sigma_end}, attention={attention_type}")

        # Set NAG processors on cross-attention layers
        original_processors = set_nag_processors(
            unet,
            nag_scale=nag_scale,
            nag_tau=nag_tau,
            nag_alpha=nag_alpha,
            attention_type=attention_type,
        )

        # Ensure NAG embeddings on correct device/dtype
        nag_negative_prompt_embeds = nag_negative_prompt_embeds.to(device=device, dtype=dtype)
        if is_sdxl and nag_negative_pooled_prompt_embeds is not None:
            nag_negative_pooled_prompt_embeds = nag_negative_pooled_prompt_embeds.to(device=device, dtype=dtype)

    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Prepare latents
    if latents is None:
        latent_channels = unet.config.in_channels
        latent_height = height // 8
        latent_width = width // 8

        # Ensure generator is on the correct device
        if generator.device.type != device:
            current_seed = generator.initial_seed()
            generator = torch.Generator(device=device).manual_seed(current_seed)

        latents = torch.randn(
            (1, latent_channels, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype
        )
        latents = latents * scheduler.init_noise_sigma

    # Current prompt embeds (will be updated by callback)
    current_prompt_embeds = prompt_embeds
    current_negative_prompt_embeds = negative_prompt_embeds
    current_pooled_prompt_embeds = pooled_prompt_embeds
    current_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

    print(f"[CustomSampling] Starting sampling loop with {num_inference_steps} steps")
    print(f"[CustomSampling] Actual timesteps: {len(timesteps)} (some schedulers like DPM2 use 2x steps)")
    print(f"[CustomSampling] Latents shape: {latents.shape}, dtype: {latents.dtype}")
    print(f"[CustomSampling] Prompt embeds shape: {prompt_embeds.shape}")

    # Get sigma_max for dynamic CFG scheduling
    sigma_max = 0.0
    if hasattr(scheduler, 'sigmas') and len(scheduler.sigmas) > 0:
        sigma_max = float(scheduler.sigmas[0].item())
    print(f"[CustomSampling] Sigma max: {sigma_max}, CFG schedule: {cfg_schedule_type}")

    # Track previous SNR for SNR-based adaptive CFG
    previous_snr = None

    # Denoising loop
    for i, t in enumerate(timesteps):
        # Check for cancellation
        from core.pipeline import pipeline_manager
        if pipeline_manager.cancel_requested:
            print("[CustomSampling] Generation cancelled by user")
            raise RuntimeError("Generation cancelled by user")

        # Check if NAG should be deactivated based on sigma threshold
        if nag_active and nag_sigma_end > 0.0:
            if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
                current_sigma = float(scheduler.sigmas[i].item())
                if current_sigma < nag_sigma_end:
                    print(f"[CustomSampling] Deactivating NAG at step {i} (sigma={current_sigma:.4f} < {nag_sigma_end})")
                    from core.nag_processor import restore_original_processors
                    restore_original_processors(unet, original_processors)
                    nag_active = False
                    # IMPORTANT: Clear NAG negative embeddings so they won't be concatenated in future steps
                    # Following official implementation: prompt_embeds = prompt_embeds[:len(latent_model_input)]
                    # After NAG ends, we only use [cfg_negative, cfg_positive] without nag_negative
                    nag_negative_prompt_embeds = None
                    print(f"[CustomSampling] NAG negative embeddings cleared for subsequent steps")

        # Check if prompt should be updated at this step
        if prompt_embeds_callback is not None:
            new_embeds = prompt_embeds_callback(i)
            if new_embeds is not None:
                current_prompt_embeds, current_negative_prompt_embeds, current_pooled_prompt_embeds, current_negative_pooled_prompt_embeds = new_embeds
                print(f"[CustomSampling] Step {i}: Updated prompt embeddings")

        # Calculate current sigma and guidance scale first to determine if we need CFG
        current_sigma = 0.0
        if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
            current_sigma = float(scheduler.sigmas[i].item())

        current_guidance_scale = calculate_dynamic_cfg(
            sigma=current_sigma,
            sigma_max=sigma_max,
            cfg_base=guidance_scale,
            cfg_schedule_type=cfg_schedule_type,
            cfg_schedule_min=cfg_schedule_min,
            cfg_schedule_max=cfg_schedule_max,
            cfg_schedule_power=cfg_schedule_power,
            snr=previous_snr,
            cfg_rescale_snr_alpha=cfg_rescale_snr_alpha
        )

        # Optimize: skip unconditional pass if guidance_scale ~= 1.0 and NAG is not active
        do_classifier_free_guidance = (abs(current_guidance_scale - 1.0) > 1e-5) or nag_active

        # Prepare latent input: single batch if no CFG needed, double batch otherwise
        if do_classifier_free_guidance:
            # Both NAG and CFG use double batch structure: [negative, positive]
            # NAG processors will apply guidance in attention space on positive batch
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Prepare prompt embeddings based on CFG and NAG configuration
            # Official NAG implementation concatenates: [cfg_negative, cfg_positive] + [nag_negative]
            if nag_active:
                # NAG mode (following official implementation):
                # prompt_embeds = [cfg_negative, cfg_positive, nag_negative] (batch=3)
                # Pad NAG negative embeddings to match the longest sequence length
                max_seq_len = max(
                    current_negative_prompt_embeds.shape[1],
                    current_prompt_embeds.shape[1],
                    nag_negative_prompt_embeds.shape[1]
                )

                # Pad each embedding to max_seq_len with zeros
                def pad_embeds(embeds, target_len):
                    if embeds.shape[1] < target_len:
                        pad_len = target_len - embeds.shape[1]
                        padding = torch.zeros(
                            embeds.shape[0], pad_len, embeds.shape[2],
                            dtype=embeds.dtype, device=embeds.device
                        )
                        return torch.cat([embeds, padding], dim=1)
                    return embeds

                current_negative_prompt_embeds_padded = pad_embeds(current_negative_prompt_embeds, max_seq_len)
                current_prompt_embeds_padded = pad_embeds(current_prompt_embeds, max_seq_len)
                nag_negative_prompt_embeds_padded = pad_embeds(nag_negative_prompt_embeds, max_seq_len)

                prompt_embeds_input = torch.cat([
                    current_negative_prompt_embeds_padded,
                    current_prompt_embeds_padded,
                    nag_negative_prompt_embeds_padded
                ], dim=0)
            else:
                # Standard CFG: [negative, positive] (batch=2)
                prompt_embeds_input = torch.cat([current_negative_prompt_embeds, current_prompt_embeds])
        else:
            # CFG = 1.0: only use conditional (positive) pass
            latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            prompt_embeds_input = current_prompt_embeds

        # Prepare added conditions for SDXL
        added_cond_kwargs = {}
        if is_sdxl:
            # SDXL requires time_ids
            original_size = (height, width)
            crops_coords_top_left = (0, 0)
            target_size = (height, width)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)

            if do_classifier_free_guidance:
                # IMPORTANT: add_time_ids and add_text_embeds must match latent batch size (2)
                # even when NAG is active, because they're used for timestep embedding
                # Only prompt_embeds (encoder_hidden_states) can be batch=3 for NAG
                add_time_ids = torch.cat([add_time_ids] * 2, dim=0)

                if current_pooled_prompt_embeds is not None:
                    # Standard CFG structure for SDXL augmentation embeddings: [negative, positive] (batch=2)
                    if current_negative_pooled_prompt_embeds is not None:
                        add_text_embeds = torch.cat([current_negative_pooled_prompt_embeds, current_pooled_prompt_embeds], dim=0)
                    else:
                        add_text_embeds = None
                else:
                    add_text_embeds = None
            else:
                # CFG = 1.0: only use positive embeddings
                add_text_embeds = current_pooled_prompt_embeds

            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids
            }

        # Get ControlNet residuals if present
        down_block_res_samples = None
        mid_block_res_sample = None

        if has_controlnet:
            # Check if this step is within the guidance range
            current_fraction = i / num_inference_steps

            # Calculate active ControlNet scales for this step
            active_scales = []
            for idx, (start, end, scale) in enumerate(zip(control_guidance_start, control_guidance_end, controlnet_conditioning_scale)):
                if start <= current_fraction <= end:
                    active_scales.append(scale)
                else:
                    active_scales.append(0.0)  # Disable ControlNet outside guidance range

            # Only run ControlNet if at least one is active
            if any(s > 0 for s in active_scales):
                with torch.no_grad():
                    # Determine batch size for ControlNet conditioning
                    batch_multiplier = 2 if do_classifier_free_guidance else 1

                    # Get ControlNet conditioning
                    if isinstance(controlnet, list):
                        # Multiple ControlNets
                        down_block_res_samples_list = []
                        mid_block_res_sample_list = []
                        for cn, ctrl_img, scale in zip(controlnet, control_image_tensors, active_scales):
                            if scale > 0:
                                controlnet_kwargs = {
                                    "encoder_hidden_states": prompt_embeds_input,
                                    "controlnet_cond": ctrl_img.repeat(batch_multiplier, 1, 1, 1),
                                    "conditioning_scale": scale,
                                    "return_dict": False,
                                }
                                # Add SDXL-specific conditioning to ControlNet
                                if is_sdxl and added_cond_kwargs:
                                    controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                                ctrl_result = cn(
                                    latent_model_input,
                                    t,
                                    **controlnet_kwargs
                                )
                                down_samples, mid_sample = ctrl_result
                                down_block_res_samples_list.append(down_samples)
                                mid_block_res_sample_list.append(mid_sample)

                        # Sum all ControlNet outputs
                        if down_block_res_samples_list:
                            down_block_res_samples = [
                                sum(samples) for samples in zip(*down_block_res_samples_list)
                            ]
                            mid_block_res_sample = sum(mid_block_res_sample_list)
                    else:
                        # Single ControlNet
                        if active_scales[0] > 0:
                            controlnet_kwargs = {
                                "encoder_hidden_states": prompt_embeds_input,
                                "controlnet_cond": control_image_tensors[0].repeat(batch_multiplier, 1, 1, 1),
                                "conditioning_scale": active_scales[0],
                                "return_dict": False,
                            }
                            # Add SDXL-specific conditioning to ControlNet
                            if is_sdxl and added_cond_kwargs:
                                controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                            down_block_res_samples, mid_block_res_sample = controlnet(
                                latent_model_input,
                                t,
                                **controlnet_kwargs
                            )

        # Predict noise residual
        with torch.no_grad():
            unet_kwargs = {
                "encoder_hidden_states": prompt_embeds_input,
            }
            if down_block_res_samples is not None:
                unet_kwargs["down_block_additional_residuals"] = down_block_res_samples
            if mid_block_res_sample is not None:
                unet_kwargs["mid_block_additional_residual"] = mid_block_res_sample

            # Add SDXL-specific conditioning as a nested dict
            if is_sdxl and added_cond_kwargs:
                unet_kwargs["added_cond_kwargs"] = added_cond_kwargs

            noise_pred = unet(
                latent_model_input,
                t,
                **unet_kwargs
            ).sample

        # Perform guidance with CFG
        if do_classifier_free_guidance:
            # NAG mode: noise_pred still has [negative, positive] batches
            # NAG guidance was applied in attention space, but CFG is still applied here
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            # Calculate preliminary CFG metrics to get SNR (if SNR-based adaptive CFG is enabled)
            current_snr = None
            if cfg_rescale_snr_alpha > 0.0 or developer_mode:
                # Calculate SNR from CFG components
                uncond_norm = torch.norm(noise_pred_uncond).item()
                diff = noise_pred_text - noise_pred_uncond
                diff_norm = torch.norm(diff).item()
                if uncond_norm > 1e-8:
                    current_snr = (diff_norm ** 2) / (uncond_norm ** 2)

            # Store current SNR for next step
            if current_snr is not None:
                previous_snr = current_snr

            # Apply CFG
            noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Apply dynamic thresholding if enabled (prevents CFG saturation)
            if dynamic_threshold_percentile > 0.0:
                noise_pred = dynamic_thresholding(
                    noise_pred,
                    percentile=dynamic_threshold_percentile,
                    clamp_value=dynamic_threshold_mimic_scale
                )

            # Apply guidance rescale if specified (important for v-prediction models)
            if guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
        else:
            # CFG = 1.0: use the prediction directly (no guidance needed)
            noise_pred_text = noise_pred
            noise_pred_uncond = None

        # Compute previous noisy sample
        # Pass step_generator to ensure reproducibility with stochastic samplers (e.g., Euler a)
        latents = scheduler.step(noise_pred, t, latents, generator=step_generator).prev_sample

        # Progress callback
        # Note: Some schedulers (DPM2, DPM2a) create more timesteps than num_inference_steps
        # so we pass len(timesteps) as the total to avoid showing progress > 100%
        if progress_callback is not None:
            # Calculate CFG metrics for developer mode
            cfg_metrics = None
            if do_classifier_free_guidance:
                cfg_metrics = calculate_cfg_metrics(
                    noise_pred_uncond,
                    noise_pred_text,
                    current_guidance_scale,
                    developer_mode=developer_mode
                )
            # Add timestep/sigma info to metrics
            if cfg_metrics is not None:
                cfg_metrics['timestep'] = int(t.item())
                cfg_metrics['step'] = i
                # Get sigma from scheduler if available
                if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
                    cfg_metrics['sigma'] = float(scheduler.sigmas[i].item())

            progress_callback(i, len(timesteps), latents, cfg_metrics=cfg_metrics)

        # Step callback
        if step_callback is not None:
            callback_kwargs = {"latents": latents}
            callback_kwargs = step_callback(pipeline, i, t, callback_kwargs)
            latents = callback_kwargs.get("latents", latents)

    print(f"[CustomSampling] Sampling complete, decoding latents")

    # Restore original processors if NAG was active
    if nag_active and original_processors is not None:
        from core.nag_processor import restore_original_processors
        restore_original_processors(unet, original_processors)

    # Decode latents to image
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents).sample

    # Offload VAE to CPU after decoding to free VRAM
    print("[CustomSampling] Offloading VAE to CPU after decode...")
    vae.to('cpu')
    torch.cuda.empty_cache()
    print("[CustomSampling] VAE offloaded to CPU")

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image[0])

    return image


def custom_img2img_sampling_loop(
    pipeline: Union[StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline],
    init_image: Image.Image,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    guidance_rescale: float = 0.0,
    generator: Optional[torch.Generator] = None,
    ancestral_generator: Optional[torch.Generator] = None,
    t_start_override: Optional[int] = None,
    prompt_embeds_callback: Optional[Callable[[int], tuple]] = None,
    progress_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    step_callback: Optional[Callable[[Any, int, int, Dict], Dict]] = None,
    developer_mode: bool = False,
    controlnet_images: Optional[List[Image.Image]] = None,
    controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
    control_guidance_start: Optional[Union[float, List[float]]] = None,
    control_guidance_end: Optional[Union[float, List[float]]] = None,
    cfg_schedule_type: str = "constant",
    cfg_schedule_min: float = 1.0,
    cfg_schedule_max: Optional[float] = None,
    cfg_schedule_power: float = 2.0,
    cfg_rescale_snr_alpha: float = 0.0,  # SNR-based adaptive CFG (0.0 = disabled)
    dynamic_threshold_percentile: float = 0.0,  # 0.0 = disabled, 99.5 = typical value
    dynamic_threshold_mimic_scale: float = 1.0,  # Clamp value for static threshold
    nag_enable: bool = False,  # Enable NAG (Normalized Attention Guidance)
    nag_scale: float = 5.0,  # NAG extrapolation scale
    nag_tau: float = 3.5,  # NAG normalization threshold
    nag_alpha: float = 0.25,  # NAG blending factor
    nag_sigma_end: float = 0.0,  # Sigma threshold to disable NAG
    nag_negative_prompt_embeds: Optional[torch.Tensor] = None,  # Separate negative embeds for NAG
    nag_negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # Separate pooled embeds for NAG (SDXL)
    attention_type: str = "normal",  # Attention backend - "normal", "sage", or "flash"
) -> Image.Image:
    """Custom img2img sampling loop with prompt editing and ControlNet support

    Args:
        pipeline: The diffusers img2img pipeline
        init_image: Initial PIL image
        prompt_embeds: Initial prompt embeddings
        negative_prompt_embeds: Initial negative prompt embeddings
        pooled_prompt_embeds: Pooled prompt embeds (SDXL only)
        negative_pooled_prompt_embeds: Negative pooled embeds (SDXL only)
        num_inference_steps: Number of denoising steps
        strength: Denoising strength (0.0 to 1.0)
        guidance_scale: CFG scale
        generator: Random generator for reproducibility
        prompt_embeds_callback: Callback to get new embeddings at each step
        progress_callback: Callback for progress updates
        step_callback: Callback after each step
        controlnet_images: List of control images for ControlNet
        controlnet_conditioning_scale: Strength of ControlNet conditioning
        control_guidance_start: When to start ControlNet guidance (0.0-1.0)
        control_guidance_end: When to end ControlNet guidance (0.0-1.0)

    Returns:
        Generated PIL Image
    """
    device = pipeline.device
    dtype = pipeline.unet.dtype

    # Check if SDXL by checking if text_encoder_2 exists
    is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

    # Use ancestral_generator for stochastic samplers, fallback to generator if not provided
    step_generator = ancestral_generator if ancestral_generator is not None else generator
    if ancestral_generator is not None:
        print(f"[CustomSampling] Using separate ancestral generator for stochastic sampler")

    # Get components
    unet = pipeline.unet
    vae = pipeline.vae
    scheduler = pipeline.scheduler

    # Get image dimensions (save before converting to tensor)
    original_width, original_height = init_image.size

    # Check if ControlNet is present
    controlnet = getattr(pipeline, 'controlnet', None)
    has_controlnet = controlnet is not None and controlnet_images is not None

    if has_controlnet:
        print(f"[CustomSampling] ControlNet detected in img2img, preparing control images")
        # Prepare control images
        if not isinstance(controlnet_images, list):
            controlnet_images = [controlnet_images]

        # Convert PIL images to tensors
        control_image_tensors = []
        for img in controlnet_images:
            if isinstance(img, Image.Image):
                img = img.resize((original_width, original_height), Image.Resampling.LANCZOS)
                img = torch.from_numpy(np.array(img)).float() / 255.0
                if img.ndim == 2:  # Grayscale
                    img = img.unsqueeze(-1).repeat(1, 1, 3)
                img = img.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            control_image_tensors.append(img.to(device=device, dtype=dtype))

        # Normalize conditioning scales
        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = 1.0
        if not isinstance(controlnet_conditioning_scale, list):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(control_image_tensors)

        # Normalize guidance ranges
        if control_guidance_start is None:
            control_guidance_start = 0.0
        if not isinstance(control_guidance_start, list):
            control_guidance_start = [control_guidance_start] * len(control_image_tensors)

        if control_guidance_end is None:
            control_guidance_end = 1.0
        if not isinstance(control_guidance_end, list):
            control_guidance_end = [control_guidance_end] * len(control_image_tensors)

    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Calculate timestep to start from
    if t_start_override is not None:
        # Use explicit t_start (for "Do full steps" mode)
        t_start = t_start_override
        print(f"[CustomSampling] Using explicit t_start={t_start} (Do full steps mode)")
    else:
        # Calculate from strength (standard img2img)
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

    timesteps = timesteps[t_start:]

    # Encode initial image to latents
    # Convert PIL image to tensor if needed
    if isinstance(init_image, Image.Image):
        init_image = torch.from_numpy(np.array(init_image)).float() / 255.0
        init_image = init_image.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        init_image = init_image * 2.0 - 1.0  # Normalize to [-1, 1]

    with torch.no_grad():
        init_latents = vae.encode(
            init_image.to(device=device, dtype=dtype)
        ).latent_dist.sample(generator)
        init_latents = init_latents * vae.config.scaling_factor

    # Add noise to latents based on timestep
    # Ensure generator is on the correct device
    if generator.device.type != device:
        current_seed = generator.initial_seed()
        generator = torch.Generator(device=device).manual_seed(current_seed)
    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
    latents = scheduler.add_noise(init_latents, noise, timesteps[0:1])

    # Current prompt embeds
    current_prompt_embeds = prompt_embeds
    current_negative_prompt_embeds = negative_prompt_embeds
    current_pooled_prompt_embeds = pooled_prompt_embeds
    current_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

    # Setup NAG if enabled
    nag_active = nag_enable and nag_negative_prompt_embeds is not None
    original_processors = None

    if nag_active:
        from core.nag_processor import set_nag_processors
        print(f"[CustomSampling] NAG enabled: scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}, sigma_end={nag_sigma_end}")

        original_processors = set_nag_processors(unet, nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha, attention_type=attention_type)

        nag_negative_prompt_embeds = nag_negative_prompt_embeds.to(device=device, dtype=dtype)
        if is_sdxl and nag_negative_pooled_prompt_embeds is not None:
            nag_negative_pooled_prompt_embeds = nag_negative_pooled_prompt_embeds.to(device=device, dtype=dtype)

    print(f"[CustomSampling] Starting img2img loop with {len(timesteps)} steps (strength={strength})")
    print(f"[CustomSampling] Latents shape: {latents.shape}, dtype: {latents.dtype}")

    # Get sigma_max for dynamic CFG scheduling
    sigma_max = 0.0
    if hasattr(scheduler, 'sigmas') and len(scheduler.sigmas) > 0:
        sigma_max = float(scheduler.sigmas[0].item())
    print(f"[CustomSampling] Sigma max: {sigma_max}, CFG schedule: {cfg_schedule_type}")

    # Track previous SNR for SNR-based adaptive CFG
    previous_snr = None

    # Denoising loop
    for i, t in enumerate(timesteps):
        # Check for cancellation
        from core.pipeline import pipeline_manager
        if pipeline_manager.cancel_requested:
            print("[CustomSampling] Generation cancelled by user")
            raise RuntimeError("Generation cancelled by user")

        # Check if NAG should be deactivated based on sigma threshold
        if nag_active and nag_sigma_end > 0.0:
            if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
                current_sigma = float(scheduler.sigmas[i].item())
                if current_sigma < nag_sigma_end:
                    print(f"[CustomSampling] Deactivating NAG at step {i} (sigma={current_sigma:.4f} < {nag_sigma_end})")
                    from core.nag_processor import restore_original_processors
                    restore_original_processors(unet, original_processors)
                    nag_active = False
                    # IMPORTANT: Clear NAG negative embeddings so they won't be concatenated in future steps
                    # Following official implementation: prompt_embeds = prompt_embeds[:len(latent_model_input)]
                    # After NAG ends, we only use [cfg_negative, cfg_positive] without nag_negative
                    nag_negative_prompt_embeds = None
                    print(f"[CustomSampling] NAG negative embeddings cleared for subsequent steps")

        # Check if prompt should be updated
        if prompt_embeds_callback is not None:
            new_embeds = prompt_embeds_callback(t_start + i)
            if new_embeds is not None:
                current_prompt_embeds, current_negative_prompt_embeds, current_pooled_prompt_embeds, current_negative_pooled_prompt_embeds = new_embeds
                print(f"[CustomSampling] Step {t_start + i}: Updated prompt embeddings")

        # Calculate current sigma and guidance scale first to determine if we need CFG
        current_sigma = 0.0
        if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
            current_sigma = float(scheduler.sigmas[i].item())

        current_guidance_scale = calculate_dynamic_cfg(
            sigma=current_sigma,
            sigma_max=sigma_max,
            cfg_base=guidance_scale,
            cfg_schedule_type=cfg_schedule_type,
            cfg_schedule_min=cfg_schedule_min,
            cfg_schedule_max=cfg_schedule_max,
            cfg_schedule_power=cfg_schedule_power,
            snr=previous_snr,
            cfg_rescale_snr_alpha=cfg_rescale_snr_alpha
        )

        # Optimize: skip unconditional pass if guidance_scale ~= 1.0 and NAG is not active
        do_classifier_free_guidance = (abs(current_guidance_scale - 1.0) > 1e-5) or nag_active

        # Prepare latent input: single batch if no CFG needed, double batch otherwise
        if do_classifier_free_guidance:
            # Both NAG and CFG use double batch structure: [negative, positive]
            # NAG processors will apply guidance in attention space on positive batch
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Prepare prompt embeddings based on CFG and NAG configuration
            # Official NAG implementation concatenates: [cfg_negative, cfg_positive] + [nag_negative]
            if nag_active:
                # NAG mode (following official implementation):
                # prompt_embeds = [cfg_negative, cfg_positive, nag_negative] (batch=3)
                # Pad NAG negative embeddings to match the longest sequence length
                max_seq_len = max(
                    current_negative_prompt_embeds.shape[1],
                    current_prompt_embeds.shape[1],
                    nag_negative_prompt_embeds.shape[1]
                )

                # Pad each embedding to max_seq_len with zeros
                def pad_embeds(embeds, target_len):
                    if embeds.shape[1] < target_len:
                        pad_len = target_len - embeds.shape[1]
                        padding = torch.zeros(
                            embeds.shape[0], pad_len, embeds.shape[2],
                            dtype=embeds.dtype, device=embeds.device
                        )
                        return torch.cat([embeds, padding], dim=1)
                    return embeds

                current_negative_prompt_embeds_padded = pad_embeds(current_negative_prompt_embeds, max_seq_len)
                current_prompt_embeds_padded = pad_embeds(current_prompt_embeds, max_seq_len)
                nag_negative_prompt_embeds_padded = pad_embeds(nag_negative_prompt_embeds, max_seq_len)

                prompt_embeds_input = torch.cat([
                    current_negative_prompt_embeds_padded,
                    current_prompt_embeds_padded,
                    nag_negative_prompt_embeds_padded
                ], dim=0)
            else:
                # Standard CFG: [negative, positive] (batch=2)
                prompt_embeds_input = torch.cat([current_negative_prompt_embeds, current_prompt_embeds])
        else:
            # CFG = 1.0: only use conditional (positive) pass
            latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            prompt_embeds_input = current_prompt_embeds

        # Prepare added conditions for SDXL
        added_cond_kwargs = {}
        if is_sdxl:
            # SDXL requires time_ids
            original_size = (original_height, original_width)
            crops_coords_top_left = (0, 0)
            target_size = (original_height, original_width)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)

            if do_classifier_free_guidance:
                # IMPORTANT: add_time_ids and add_text_embeds must match latent batch size (2)
                # even when NAG is active, because they're used for timestep embedding
                # Only prompt_embeds (encoder_hidden_states) can be batch=3 for NAG
                add_time_ids = torch.cat([add_time_ids] * 2, dim=0)

                if current_pooled_prompt_embeds is not None:
                    # Standard CFG structure for SDXL augmentation embeddings: [negative, positive] (batch=2)
                    if current_negative_pooled_prompt_embeds is not None:
                        add_text_embeds = torch.cat([current_negative_pooled_prompt_embeds, current_pooled_prompt_embeds], dim=0)
                    else:
                        add_text_embeds = None
                else:
                    add_text_embeds = None
            else:
                # CFG = 1.0: only use positive embeddings
                add_text_embeds = current_pooled_prompt_embeds

            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids
            }

        # Get ControlNet residuals if present
        down_block_res_samples = None
        mid_block_res_sample = None

        if has_controlnet:
            # Check if this step is within the guidance range
            current_fraction = (t_start + i) / num_inference_steps

            # Calculate active ControlNet scales for this step
            active_scales = []
            for idx, (start, end, scale) in enumerate(zip(control_guidance_start, control_guidance_end, controlnet_conditioning_scale)):
                if start <= current_fraction <= end:
                    active_scales.append(scale)
                else:
                    active_scales.append(0.0)

            # Only run ControlNet if at least one is active
            if any(s > 0 for s in active_scales):
                with torch.no_grad():
                    # Determine batch size for ControlNet conditioning
                    batch_multiplier = 2 if do_classifier_free_guidance else 1

                    # Get ControlNet conditioning
                    if isinstance(controlnet, list):
                        # Multiple ControlNets
                        down_block_res_samples_list = []
                        mid_block_res_sample_list = []
                        for cn, ctrl_img, scale in zip(controlnet, control_image_tensors, active_scales):
                            if scale > 0:
                                controlnet_kwargs = {
                                    "encoder_hidden_states": prompt_embeds_input,
                                    "controlnet_cond": ctrl_img.repeat(batch_multiplier, 1, 1, 1),
                                    "conditioning_scale": scale,
                                    "return_dict": False,
                                }
                                if is_sdxl and added_cond_kwargs:
                                    controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                                ctrl_result = cn(
                                    latent_model_input,
                                    t,
                                    **controlnet_kwargs
                                )
                                down_samples, mid_sample = ctrl_result
                                down_block_res_samples_list.append(down_samples)
                                mid_block_res_sample_list.append(mid_sample)

                        # Sum all ControlNet outputs
                        if down_block_res_samples_list:
                            down_block_res_samples = [
                                sum(samples) for samples in zip(*down_block_res_samples_list)
                            ]
                            mid_block_res_sample = sum(mid_block_res_sample_list)
                    else:
                        # Single ControlNet
                        if active_scales[0] > 0:
                            controlnet_kwargs = {
                                "encoder_hidden_states": prompt_embeds_input,
                                "controlnet_cond": control_image_tensors[0].repeat(batch_multiplier, 1, 1, 1),
                                "conditioning_scale": active_scales[0],
                                "return_dict": False,
                            }
                            if is_sdxl and added_cond_kwargs:
                                controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                            down_block_res_samples, mid_block_res_sample = controlnet(
                                latent_model_input,
                                t,
                                **controlnet_kwargs
                            )

        # Predict noise residual
        with torch.no_grad():
            unet_kwargs = {
                "encoder_hidden_states": prompt_embeds_input,
            }
            if down_block_res_samples is not None:
                unet_kwargs["down_block_additional_residuals"] = down_block_res_samples
            if mid_block_res_sample is not None:
                unet_kwargs["mid_block_additional_residual"] = mid_block_res_sample

            # Add SDXL-specific conditioning as a nested dict
            if is_sdxl and added_cond_kwargs:
                unet_kwargs["added_cond_kwargs"] = added_cond_kwargs

            noise_pred = unet(
                latent_model_input,
                t,
                **unet_kwargs
            ).sample

        # Perform guidance with CFG
        if do_classifier_free_guidance:
            # NAG mode: noise_pred still has [negative, positive] batches
            # NAG guidance was applied in attention space, but CFG is still applied here
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            # Calculate preliminary CFG metrics to get SNR (if SNR-based adaptive CFG is enabled)
            current_snr = None
            if cfg_rescale_snr_alpha > 0.0 or developer_mode:
                # Calculate SNR from CFG components
                uncond_norm = torch.norm(noise_pred_uncond).item()
                diff = noise_pred_text - noise_pred_uncond
                diff_norm = torch.norm(diff).item()
                if uncond_norm > 1e-8:
                    current_snr = (diff_norm ** 2) / (uncond_norm ** 2)

            # Store current SNR for next step
            if current_snr is not None:
                previous_snr = current_snr

            # Apply CFG
            noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Apply dynamic thresholding if enabled (prevents CFG saturation)
            if dynamic_threshold_percentile > 0.0:
                noise_pred = dynamic_thresholding(
                    noise_pred,
                    percentile=dynamic_threshold_percentile,
                    clamp_value=dynamic_threshold_mimic_scale
                )

            # Apply guidance rescale if specified (important for v-prediction models)
            if guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
        else:
            # CFG = 1.0: use the prediction directly (no guidance needed)
            noise_pred_text = noise_pred
            noise_pred_uncond = None

        # Compute previous noisy sample
        # Pass step_generator to ensure reproducibility with stochastic samplers (e.g., Euler a)
        latents = scheduler.step(noise_pred, t, latents, generator=step_generator).prev_sample

        # Progress callback
        if progress_callback is not None:
            # Calculate CFG metrics for developer mode
            cfg_metrics = None
            if do_classifier_free_guidance:
                cfg_metrics = calculate_cfg_metrics(
                    noise_pred_uncond,
                    noise_pred_text,
                    current_guidance_scale,
                    developer_mode=developer_mode
                )
            # Add timestep/sigma info to metrics
            if cfg_metrics is not None:
                cfg_metrics['timestep'] = int(t.item())
                cfg_metrics['step'] = i
                # Get sigma from scheduler if available
                if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
                    cfg_metrics['sigma'] = float(scheduler.sigmas[i].item())

            progress_callback(i, len(timesteps), latents, cfg_metrics=cfg_metrics)

        # Step callback
        if step_callback is not None:
            callback_kwargs = {"latents": latents}
            callback_kwargs = step_callback(pipeline, t_start + i, t, callback_kwargs)
            latents = callback_kwargs.get("latents", latents)

    print(f"[CustomSampling] Sampling complete, decoding latents")

    # Restore original processors if NAG was active
    if nag_active and original_processors is not None:
        from core.nag_processor import restore_original_processors
        restore_original_processors(unet, original_processors)

    # Decode latents to image
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents).sample

    # Offload VAE to CPU after decoding to free VRAM
    print("[CustomSampling] Offloading VAE to CPU after decode...")
    vae.to('cpu')
    torch.cuda.empty_cache()
    print("[CustomSampling] VAE offloaded to CPU")

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image[0])

    return image


def custom_inpaint_sampling_loop(
    pipeline: Union[StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline],
    init_image: Image.Image,
    mask_image: Image.Image,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    guidance_rescale: float = 0.0,
    generator: Optional[torch.Generator] = None,
    ancestral_generator: Optional[torch.Generator] = None,
    t_start_override: Optional[int] = None,
    prompt_embeds_callback: Optional[Callable[[int], tuple]] = None,
    progress_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    step_callback: Optional[Callable[[Any, int, int, Dict], Dict]] = None,
    developer_mode: bool = False,
    controlnet_images: Optional[List[Image.Image]] = None,
    controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
    control_guidance_start: Optional[Union[float, List[float]]] = None,
    control_guidance_end: Optional[Union[float, List[float]]] = None,
    inpaint_fill_mode: str = "original",
    inpaint_fill_strength: float = 1.0,
    inpaint_blur_strength: float = 1.0,
    cfg_schedule_type: str = "constant",
    cfg_schedule_min: float = 1.0,
    cfg_schedule_max: Optional[float] = None,
    cfg_schedule_power: float = 2.0,
    cfg_rescale_snr_alpha: float = 0.0,  # SNR-based adaptive CFG (0.0 = disabled)
    dynamic_threshold_percentile: float = 0.0,  # 0.0 = disabled, 99.5 = typical value
    dynamic_threshold_mimic_scale: float = 1.0,  # Clamp value for static threshold
    nag_enable: bool = False,  # Enable NAG (Normalized Attention Guidance)
    nag_scale: float = 5.0,  # NAG extrapolation scale
    nag_tau: float = 3.5,  # NAG normalization threshold
    nag_alpha: float = 0.25,  # NAG blending factor
    nag_sigma_end: float = 0.0,  # Sigma threshold to disable NAG
    nag_negative_prompt_embeds: Optional[torch.Tensor] = None,  # Separate negative embeds for NAG
    nag_negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,  # Separate pooled embeds for NAG (SDXL)
    attention_type: str = "normal",  # Attention backend - "normal", "sage", or "flash"
) -> Image.Image:
    """Custom inpaint sampling loop with prompt editing and ControlNet support"""
    device = pipeline.device
    dtype = pipeline.unet.dtype

    # Check if SDXL by checking if text_encoder_2 exists
    is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

    # Use ancestral_generator for stochastic samplers, fallback to generator if not provided
    step_generator = ancestral_generator if ancestral_generator is not None else generator
    if ancestral_generator is not None:
        print(f"[CustomSampling] Using separate ancestral generator for stochastic sampler")

    unet = pipeline.unet
    vae = pipeline.vae
    scheduler = pipeline.scheduler

    # Check if this is an inpaint-specific UNet (9 channels) or regular UNet (4 channels)
    # Regular UNets cannot accept concatenated mask+image, so we'll use img2img-style masking
    is_inpaint_unet = unet.config.in_channels == 9
    print(f"[CustomSampling] UNet in_channels: {unet.config.in_channels}, is_inpaint_unet: {is_inpaint_unet}")

    # Get image dimensions (save before converting to tensor)
    original_width, original_height = init_image.size

    # Check if ControlNet is present
    controlnet = getattr(pipeline, 'controlnet', None)
    has_controlnet = controlnet is not None and controlnet_images is not None

    if has_controlnet:
        print(f"[CustomSampling] ControlNet detected in inpaint, preparing control images")
        if not isinstance(controlnet_images, list):
            controlnet_images = [controlnet_images]

        control_image_tensors = []
        for img in controlnet_images:
            if isinstance(img, Image.Image):
                img = img.resize((original_width, original_height), Image.Resampling.LANCZOS)
                img = torch.from_numpy(np.array(img)).float() / 255.0
                if img.ndim == 2:
                    img = img.unsqueeze(-1).repeat(1, 1, 3)
                img = img.permute(2, 0, 1).unsqueeze(0)
            control_image_tensors.append(img.to(device=device, dtype=dtype))

        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = 1.0
        if not isinstance(controlnet_conditioning_scale, list):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(control_image_tensors)

        if control_guidance_start is None:
            control_guidance_start = 0.0
        if not isinstance(control_guidance_start, list):
            control_guidance_start = [control_guidance_start] * len(control_image_tensors)

        if control_guidance_end is None:
            control_guidance_end = 1.0
        if not isinstance(control_guidance_end, list):
            control_guidance_end = [control_guidance_end] * len(control_image_tensors)

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Calculate timestep to start from
    if t_start_override is not None:
        # Use explicit t_start (for "Do full steps" mode)
        t_start = t_start_override
        print(f"[CustomSampling] Using explicit t_start={t_start} (Do full steps mode)")
    else:
        # Calculate from strength (standard inpaint)
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

    timesteps = timesteps[t_start:]

    # Prepare images
    if isinstance(init_image, Image.Image):
        init_image_tensor = torch.from_numpy(np.array(init_image)).float() / 255.0
        init_image_tensor = init_image_tensor.permute(2, 0, 1).unsqueeze(0)
        init_image_tensor = init_image_tensor * 2.0 - 1.0
    else:
        init_image_tensor = init_image

    if isinstance(mask_image, Image.Image):
        mask_tensor = torch.from_numpy(np.array(mask_image.convert("L"))).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    else:
        mask_tensor = mask_image

    with torch.no_grad():
        init_latents = vae.encode(
            init_image_tensor.to(device=device, dtype=dtype)
        ).latent_dist.sample(generator)
        init_latents = init_latents * vae.config.scaling_factor

    mask_latent = torch.nn.functional.interpolate(
        mask_tensor.to(device=device, dtype=dtype),
        size=(init_latents.shape[-2], init_latents.shape[-1]),
        mode="nearest"
    )

    # Store original image latents for mask blending (before adding noise)
    image_latents = init_latents.clone()

    # Apply inpaint fill mode to init_latents (before adding scheduler noise)
    # mask_latent: 1.0 = inpaint area (white), 0.0 = keep original (black)
    if inpaint_fill_mode != "original" and inpaint_fill_strength > 0:
        print(f"[CustomSampling] Applying inpaint fill mode: {inpaint_fill_mode} (strength: {inpaint_fill_strength})")

        if inpaint_fill_mode == "blur":
            # Apply gaussian blur to the original image
            import torch.nn.functional as F
            # Blur with kernel size proportional to image size and blur strength
            # inpaint_blur_strength: 0.1 = very weak blur, 1.0 = default, 2.0+ = very strong blur
            base_kernel_size = max(3, int(original_width / 10) | 1)  # Ensure odd number
            kernel_size = max(3, int(base_kernel_size * inpaint_blur_strength) | 1)
            sigma = kernel_size / 3.0

            # Create gaussian kernel
            x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype, device=device)
            gauss = torch.exp(-x**2 / (2 * sigma**2))
            gauss = gauss / gauss.sum()
            kernel_1d = gauss.unsqueeze(0)

            # Apply separable 2D gaussian blur
            # Number of iterations based on blur strength (1-5 iterations)
            blur_iterations = max(1, min(5, int(3 * inpaint_blur_strength)))
            blurred = init_image_tensor.to(device=device, dtype=dtype)
            for _ in range(blur_iterations):
                blurred = F.conv2d(blurred, kernel_1d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1), padding=(0, kernel_size // 2), groups=3)
                blurred = F.conv2d(blurred, kernel_1d.t().unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1), padding=(kernel_size // 2, 0), groups=3)

            print(f"[CustomSampling] Blur applied: kernel_size={kernel_size}, iterations={blur_iterations}, strength={inpaint_blur_strength}")

            with torch.no_grad():
                blurred_latents = vae.encode(blurred).latent_dist.sample(generator)
                blurred_latents = blurred_latents * vae.config.scaling_factor

            # Mix blurred latents into masked region (mask=1 is inpaint area)
            # Formula: original * (1-mask) + fill * mask * strength + original * mask * (1-strength)
            init_latents = init_latents * (1 - mask_latent) + blurred_latents * mask_latent * inpaint_fill_strength + init_latents * mask_latent * (1 - inpaint_fill_strength)

        elif inpaint_fill_mode == "noise":
            # Fill masked region with random latent noise (mask=1 is inpaint area)
            # Ensure generator is on the correct device
            if generator.device.type != device:
                current_seed = generator.initial_seed()
                generator = torch.Generator(device=device).manual_seed(current_seed)
            random_latents = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
            init_latents = init_latents * (1 - mask_latent) + random_latents * mask_latent * inpaint_fill_strength + init_latents * mask_latent * (1 - inpaint_fill_strength)

        elif inpaint_fill_mode == "erase":
            # Fill masked region with zeros/latent nothing (mask=1 is inpaint area)
            # Keep original where mask=0, zero out where mask=1 (scaled by strength)
            init_latents = init_latents * (1 - mask_latent * inpaint_fill_strength)

    # Ensure generator is on the correct device
    if generator.device.type != device:
        current_seed = generator.initial_seed()
        generator = torch.Generator(device=device).manual_seed(current_seed)
    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
    latents = scheduler.add_noise(init_latents, noise, timesteps[0:1])

    current_prompt_embeds = prompt_embeds
    current_negative_prompt_embeds = negative_prompt_embeds
    current_pooled_prompt_embeds = pooled_prompt_embeds
    current_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

    # Setup NAG if enabled
    nag_active = nag_enable and nag_negative_prompt_embeds is not None
    original_processors = None

    if nag_active:
        from core.nag_processor import set_nag_processors
        print(f"[CustomSampling] NAG enabled: scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}, sigma_end={nag_sigma_end}")

        original_processors = set_nag_processors(unet, nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha, attention_type=attention_type)

        nag_negative_prompt_embeds = nag_negative_prompt_embeds.to(device=device, dtype=dtype)
        if is_sdxl and nag_negative_pooled_prompt_embeds is not None:
            nag_negative_pooled_prompt_embeds = nag_negative_pooled_prompt_embeds.to(device=device, dtype=dtype)

    print(f"[CustomSampling] Starting inpaint loop with {len(timesteps)} steps")

    # Get sigma_max for dynamic CFG scheduling
    sigma_max = 0.0
    if hasattr(scheduler, 'sigmas') and len(scheduler.sigmas) > 0:
        sigma_max = float(scheduler.sigmas[0].item())
    print(f"[CustomSampling] Sigma max: {sigma_max}, CFG schedule: {cfg_schedule_type}")

    # Track previous SNR for SNR-based adaptive CFG
    previous_snr = None

    for i, t in enumerate(timesteps):
        # Check for cancellation
        from core.pipeline import pipeline_manager
        if pipeline_manager.cancel_requested:
            print("[CustomSampling] Generation cancelled by user")
            raise RuntimeError("Generation cancelled by user")

        # Check if NAG should be deactivated based on sigma threshold
        if nag_active and nag_sigma_end > 0.0:
            if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
                current_sigma = float(scheduler.sigmas[i].item())
                if current_sigma < nag_sigma_end:
                    print(f"[CustomSampling] Deactivating NAG at step {i} (sigma={current_sigma:.4f} < {nag_sigma_end})")
                    from core.nag_processor import restore_original_processors
                    restore_original_processors(unet, original_processors)
                    nag_active = False
                    # IMPORTANT: Clear NAG negative embeddings so they won't be concatenated in future steps
                    # Following official implementation: prompt_embeds = prompt_embeds[:len(latent_model_input)]
                    # After NAG ends, we only use [cfg_negative, cfg_positive] without nag_negative
                    nag_negative_prompt_embeds = None
                    print(f"[CustomSampling] NAG negative embeddings cleared for subsequent steps")

        if prompt_embeds_callback is not None:
            new_embeds = prompt_embeds_callback(t_start + i)
            if new_embeds is not None:
                current_prompt_embeds, current_negative_prompt_embeds, current_pooled_prompt_embeds, current_negative_pooled_prompt_embeds = new_embeds

        # Calculate current sigma and guidance scale first to determine if we need CFG
        current_sigma = 0.0
        if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
            current_sigma = float(scheduler.sigmas[i].item())

        current_guidance_scale = calculate_dynamic_cfg(
            sigma=current_sigma,
            sigma_max=sigma_max,
            cfg_base=guidance_scale,
            cfg_schedule_type=cfg_schedule_type,
            cfg_schedule_min=cfg_schedule_min,
            cfg_schedule_max=cfg_schedule_max,
            cfg_schedule_power=cfg_schedule_power,
            snr=previous_snr,
            cfg_rescale_snr_alpha=cfg_rescale_snr_alpha
        )

        # Optimize: skip unconditional pass if guidance_scale ~= 1.0 and NAG is not active
        do_classifier_free_guidance = (abs(current_guidance_scale - 1.0) > 1e-5) or nag_active

        # Prepare latent input: single batch if no CFG needed, double batch otherwise
        if do_classifier_free_guidance:
            # Both NAG and CFG use double batch structure: [negative, positive]
            # NAG processors will apply guidance in attention space on positive batch
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Only concatenate mask and masked image for inpaint-specific UNets
            # Regular UNets use post-processing masking instead (see after scheduler.step)
            if is_inpaint_unet:
                # Use original clean image latents, masked to show only non-inpaint regions
                masked_image_latents = image_latents * (1 - mask_latent)
                latent_model_input = torch.cat([latent_model_input, mask_latent.repeat(2, 1, 1, 1), masked_image_latents.repeat(2, 1, 1, 1)], dim=1)

            # Prepare prompt embeddings: [negative, positive]
            # NAG mode: use NAG negative embeddings for cross-attention guidance
            if nag_active:
                prompt_embeds_input = torch.cat([nag_negative_prompt_embeds, current_prompt_embeds])
            else:
                prompt_embeds_input = torch.cat([current_negative_prompt_embeds, current_prompt_embeds])
        else:
            # CFG = 1.0: only use conditional (positive) pass
            latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Only concatenate mask and masked image for inpaint-specific UNets
            if is_inpaint_unet:
                # Use original clean image latents, masked to show only non-inpaint regions
                masked_image_latents = image_latents * (1 - mask_latent)
                latent_model_input = torch.cat([latent_model_input, mask_latent, masked_image_latents], dim=1)

            prompt_embeds_input = current_prompt_embeds

        # Prepare added conditions for SDXL
        added_cond_kwargs = {}
        if is_sdxl:
            # SDXL requires time_ids
            original_size = (original_height, original_width)
            crops_coords_top_left = (0, 0)
            target_size = (original_height, original_width)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)

            if do_classifier_free_guidance:
                # IMPORTANT: add_time_ids and add_text_embeds must match latent batch size (2)
                # even when NAG is active, because they're used for timestep embedding
                # Only prompt_embeds (encoder_hidden_states) can be batch=3 for NAG
                add_time_ids = torch.cat([add_time_ids] * 2, dim=0)

                if current_pooled_prompt_embeds is not None:
                    # Standard CFG structure for SDXL augmentation embeddings: [negative, positive] (batch=2)
                    if current_negative_pooled_prompt_embeds is not None:
                        add_text_embeds = torch.cat([current_negative_pooled_prompt_embeds, current_pooled_prompt_embeds], dim=0)
                    else:
                        add_text_embeds = None
                else:
                    add_text_embeds = None
            else:
                # CFG = 1.0: only use positive embeddings
                add_text_embeds = current_pooled_prompt_embeds

            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids
            }

        # Get ControlNet residuals if present
        down_block_res_samples = None
        mid_block_res_sample = None

        if has_controlnet:
            current_fraction = (t_start + i) / num_inference_steps
            active_scales = []
            for idx, (start, end, scale) in enumerate(zip(control_guidance_start, control_guidance_end, controlnet_conditioning_scale)):
                if start <= current_fraction <= end:
                    active_scales.append(scale)
                else:
                    active_scales.append(0.0)

            if any(s > 0 for s in active_scales):
                with torch.no_grad():
                    # Determine batch size for ControlNet conditioning
                    batch_multiplier = 2 if do_classifier_free_guidance else 1

                    # Get ControlNet conditioning
                    if isinstance(controlnet, list):
                        down_block_res_samples_list = []
                        mid_block_res_sample_list = []
                        for cn, ctrl_img, scale in zip(controlnet, control_image_tensors, active_scales):
                            if scale > 0:
                                controlnet_kwargs = {
                                    "encoder_hidden_states": prompt_embeds_input,
                                    "controlnet_cond": ctrl_img.repeat(batch_multiplier, 1, 1, 1),
                                    "conditioning_scale": scale,
                                    "return_dict": False,
                                }
                                if is_sdxl and added_cond_kwargs:
                                    controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                                ctrl_result = cn(
                                    latent_model_input,
                                    t,
                                    **controlnet_kwargs
                                )
                                down_samples, mid_sample = ctrl_result
                                down_block_res_samples_list.append(down_samples)
                                mid_block_res_sample_list.append(mid_sample)

                        if down_block_res_samples_list:
                            down_block_res_samples = [
                                sum(samples) for samples in zip(*down_block_res_samples_list)
                            ]
                            mid_block_res_sample = sum(mid_block_res_sample_list)
                    else:
                        if active_scales[0] > 0:
                            controlnet_kwargs = {
                                "encoder_hidden_states": prompt_embeds_input,
                                "controlnet_cond": control_image_tensors[0].repeat(batch_multiplier, 1, 1, 1),
                                "conditioning_scale": active_scales[0],
                                "return_dict": False,
                            }
                            if is_sdxl and added_cond_kwargs:
                                controlnet_kwargs["added_cond_kwargs"] = added_cond_kwargs

                            down_block_res_samples, mid_block_res_sample = controlnet(
                                latent_model_input,
                                t,
                                **controlnet_kwargs
                            )

        with torch.no_grad():
            unet_kwargs = {
                "encoder_hidden_states": prompt_embeds_input,
            }
            if down_block_res_samples is not None:
                unet_kwargs["down_block_additional_residuals"] = down_block_res_samples
            if mid_block_res_sample is not None:
                unet_kwargs["mid_block_additional_residual"] = mid_block_res_sample

            # Add SDXL-specific conditioning as a nested dict
            if is_sdxl and added_cond_kwargs:
                unet_kwargs["added_cond_kwargs"] = added_cond_kwargs

            noise_pred = unet(
                latent_model_input,
                t,
                **unet_kwargs
            ).sample

        # Perform guidance with CFG
        if do_classifier_free_guidance:
            # NAG mode: noise_pred still has [negative, positive] batches
            # NAG guidance was applied in attention space, but CFG is still applied here
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            # Calculate preliminary CFG metrics to get SNR (if SNR-based adaptive CFG is enabled)
            current_snr = None
            if cfg_rescale_snr_alpha > 0.0 or developer_mode:
                # Calculate SNR from CFG components
                uncond_norm = torch.norm(noise_pred_uncond).item()
                diff = noise_pred_text - noise_pred_uncond
                diff_norm = torch.norm(diff).item()
                if uncond_norm > 1e-8:
                    current_snr = (diff_norm ** 2) / (uncond_norm ** 2)

            # Store current SNR for next step
            if current_snr is not None:
                previous_snr = current_snr

            # Apply CFG
            noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Apply dynamic thresholding if enabled (prevents CFG saturation)
            if dynamic_threshold_percentile > 0.0:
                noise_pred = dynamic_thresholding(
                    noise_pred,
                    percentile=dynamic_threshold_percentile,
                    clamp_value=dynamic_threshold_mimic_scale
                )

            # Apply guidance rescale if specified (important for v-prediction models)
            if guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
        else:
            # CFG = 1.0: use the prediction directly (no guidance needed)
            noise_pred_text = noise_pred
            noise_pred_uncond = None

        # Pass step_generator to ensure reproducibility with stochastic samplers (e.g., Euler a)
        latents = scheduler.step(noise_pred, t, latents, generator=step_generator).prev_sample

        # Apply mask blending ONLY for 4-channel UNets (regular models)
        # 9-channel inpaint UNets handle masking internally via concatenation
        if not is_inpaint_unet:
            init_latents_proper = image_latents  # Use clean original image latents

            # Re-noise original to match the noise level of denoised latents
            # Use NEXT timestep (where denoised latents are), not current timestep
            # Skip re-noising on the last step
            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                init_latents_proper = scheduler.add_noise(
                    init_latents_proper,
                    noise,
                    noise_timestep.unsqueeze(0) if noise_timestep.dim() == 0 else noise_timestep
                )

            # Blend: preserve original outside mask (mask=0), use generated inside mask (mask=1)
            latents = (1 - mask_latent) * init_latents_proper + mask_latent * latents

        if progress_callback is not None:
            # Calculate CFG metrics for developer mode
            cfg_metrics = None
            if do_classifier_free_guidance:
                cfg_metrics = calculate_cfg_metrics(
                    noise_pred_uncond,
                    noise_pred_text,
                    current_guidance_scale,
                    developer_mode=developer_mode
                )
            # Add timestep/sigma info to metrics
            if cfg_metrics is not None:
                cfg_metrics['timestep'] = int(t.item())
                cfg_metrics['step'] = i
                # Get sigma from scheduler if available
                if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
                    cfg_metrics['sigma'] = float(scheduler.sigmas[i].item())

            progress_callback(i, len(timesteps), latents, cfg_metrics=cfg_metrics)

        if step_callback is not None:
            callback_kwargs = step_callback(pipeline, t_start + i, t, {"latents": latents})
            latents = callback_kwargs.get("latents", latents)

    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)

    # Apply pixel-space mask blending for non-inpaint UNets
    # This preserves the original image exactly in non-masked regions
    if not is_inpaint_unet and t_start_override == 0:
        print("[CustomSampling] Applying pixel-space mask blending for exact preservation")

        # Convert original init_image to tensor in same format as decoded image
        if isinstance(init_image, Image.Image):
            original_tensor = torch.from_numpy(np.array(init_image)).float() / 255.0
            original_tensor = original_tensor.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
            # Normalize to [-1, 1] then back to [0, 1]
            original_tensor = original_tensor * 2.0 - 1.0
            original_tensor = (original_tensor / 2 + 0.5).clamp(0, 1)
        else:
            original_tensor = init_image

        # Resize mask to image dimensions if needed
        mask_pixel = torch.nn.functional.interpolate(
            mask_tensor.to(device=device, dtype=dtype),
            size=(image.shape[2], image.shape[3]),
            mode="nearest"
        )

        # Blend: keep original where mask=0, use generated where mask=1
        image = (1 - mask_pixel) * original_tensor + mask_pixel * image
        print("[CustomSampling] Pixel-space blending completed")

    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")

    # Clean up ControlNet after generation
    from .controlnet_manager import controlnet_manager
    controlnet_manager.remove_lllite_patches()
    controlnet_manager.offload_controlnets_to_cpu()

    return Image.fromarray(image[0])
