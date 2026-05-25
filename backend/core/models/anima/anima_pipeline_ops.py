"""High-level inference operations for Anima:
  - text encoding (Qwen3 + T5 tokenization for LLM Adapter)
  - txt2img / img2img / inpaint sampling loops (Rectified Flow Euler)
  - VAE encode / decode helpers

These are intentionally kept self-contained so pipeline.py can call them
without having to learn Anima internals.
"""

from typing import Optional, Tuple, Dict, Any, Callable, List

import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .anima_scheduler import AnimaFlowMatchScheduler, calculate_shift_anima


def _to_device(model, device):
    if model is None:
        return None
    return model.to(device) if model.device != torch.device(device) else model


# --------- Tokenization & encoding ---------

def tokenize_for_anima(qwen3_tokenizer, t5_tokenizer, prompt: str,
                       qwen3_max_length: int = 512,
                       t5_max_length: int = 512) -> Dict[str, torch.Tensor]:
    """Produce Qwen3 + T5 token tensors for a single prompt."""
    qwen3_enc = qwen3_tokenizer(
        [prompt], return_tensors="pt", truncation=True,
        padding="max_length", max_length=qwen3_max_length,
    )
    t5_enc = t5_tokenizer(
        [prompt], return_tensors="pt", truncation=True,
        padding="max_length", max_length=t5_max_length,
    )
    return {
        "qwen3_input_ids": qwen3_enc["input_ids"],
        "qwen3_attn_mask": qwen3_enc["attention_mask"],
        "t5_input_ids": t5_enc["input_ids"],
        "t5_attn_mask": t5_enc["attention_mask"],
    }


@torch.no_grad()
def encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer, prompt: str,
                  device: str = "cuda",
                  dtype: torch.dtype = torch.bfloat16,
                  qwen3_max_length: int = 512,
                  t5_max_length: int = 512) -> Dict[str, torch.Tensor]:
    """Run the Qwen3 text encoder and prepare the inputs the Anima DiT expects.

    Returns a dict with:
      - prompt_embeds:  Qwen3 hidden states [1, L_qwen, 1024], zero-masked
      - source_mask:    Qwen3 attention mask [1, L_qwen]
      - t5_input_ids:   T5 token ids [1, L_t5]
      - t5_attn_mask:   T5 attention mask [1, L_t5]
    """
    toks = tokenize_for_anima(qwen3_tokenizer, t5_tokenizer, prompt or "",
                              qwen3_max_length, t5_max_length)
    qwen3_input_ids = toks["qwen3_input_ids"].to(device)
    qwen3_attn_mask = toks["qwen3_attn_mask"].to(device)
    t5_input_ids = toks["t5_input_ids"].to(device)
    t5_attn_mask = toks["t5_attn_mask"].to(device)

    outputs = text_encoder(input_ids=qwen3_input_ids, attention_mask=qwen3_attn_mask)
    prompt_embeds = outputs.last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype)
    prompt_embeds[~qwen3_attn_mask.bool()] = 0

    return {
        "prompt_embeds": prompt_embeds,
        "source_mask": qwen3_attn_mask,
        "t5_input_ids": t5_input_ids,
        "t5_attn_mask": t5_attn_mask,
    }


# --------- VAE helpers ---------

def _get_qwen_vae_normalization(vae, device, dtype):
    """Return (mean, std) tensors shaped (1, z_dim, 1, 1, 1) for Qwen-Image VAE."""
    z_dim = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, dtype=dtype, device=device).view(1, z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=dtype, device=device).view(1, z_dim, 1, 1, 1)
    return mean, std


@torch.no_grad()
def vae_encode_image(vae, image: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Encode a PIL image to normalized Anima latents (B=1, C=16, T=1, H/8, W/8).

    Applies the Qwen-Image latents_mean / latents_std normalization
    (matches diffusers QwenImagePipeline conventions).
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
    t = t.unsqueeze(2)  # [1, 3, 1, H, W]
    posterior = vae.encode(t).latent_dist
    latents = posterior.sample()

    mean, std = _get_qwen_vae_normalization(vae, latents.device, latents.dtype)
    latents = (latents - mean) / std
    return latents


@torch.no_grad()
def vae_decode_latents(vae, latents: torch.Tensor) -> List[Image.Image]:
    """Decode normalized (B, 16, 1, H/8, W/8) latents to PIL images.

    Reverses the latents_mean / latents_std normalization before calling the
    VAE decoder.
    """
    if latents.dim() == 4:
        latents = latents.unsqueeze(2)

    mean, std = _get_qwen_vae_normalization(vae, latents.device, latents.dtype)
    raw_latents = latents * std + mean

    out = vae.decode(raw_latents)
    sample = out.sample if hasattr(out, "sample") else out
    if sample.dim() == 5 and sample.shape[2] == 1:
        sample = sample.squeeze(2)
    sample = (sample.float().clamp(-1, 1) + 1) / 2.0
    sample = (sample * 255.0).round().clamp(0, 255).to(torch.uint8)
    images = []
    for i in range(sample.shape[0]):
        arr = sample[i].permute(1, 2, 0).cpu().numpy()
        images.append(Image.fromarray(arr))
    return images


# --------- Sampling ---------

def _prepare_padding_mask(batch: int, height: int, width: int,
                          device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Padding mask in [B, 1, H, W] (0 = valid).

    Matches sd-scripts anima_train_utils.sample(); the DiT internally resizes
    this to the latent resolution and concatenates as an extra input channel.
    """
    return torch.zeros(batch, 1, height, width, device=device, dtype=dtype)


@torch.no_grad()
def sample_txt2img(
    transformer,
    scheduler: AnimaFlowMatchScheduler,
    cond_embeds: Dict[str, torch.Tensor],
    uncond_embeds: Optional[Dict[str, torch.Tensor]],
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    device: str,
    dtype: torch.dtype,
    step_callback: Optional[Callable] = None,
) -> torch.Tensor:
    """Run the Rectified-Flow Euler denoising loop and return latents
    of shape [1, 16, 1, H/8, W/8].
    """
    do_cfg = guidance_scale is not None and guidance_scale > 1.0 and uncond_embeds is not None
    latent_h = height // 8
    latent_w = width // 8

    # Resolution-dependent shift (FLUX/Z-Image-style)
    seq_len = latent_h * latent_w
    shift = calculate_shift_anima(seq_len)
    scheduler.set_timesteps(num_inference_steps, device=torch.device(device), shift=shift)

    # Initial noise [B, 16, 1, latent_h, latent_w]
    latents = torch.randn(
        (1, 16, 1, latent_h, latent_w), generator=generator,
        device=device, dtype=dtype,
    )

    padding_mask = _prepare_padding_mask(1, latent_h, latent_w, torch.device(device), dtype)

    for i in range(num_inference_steps):
        timestep = scheduler.get_timestep(i, device=torch.device(device), dtype=dtype)
        timestep_batch = timestep.expand(latents.shape[0])

        # Conditional pass
        v_cond = transformer(
            x=latents,
            timesteps=timestep_batch,
            context=cond_embeds["prompt_embeds"],
            padding_mask=padding_mask,
            target_input_ids=cond_embeds["t5_input_ids"],
            target_attention_mask=cond_embeds["t5_attn_mask"],
            source_attention_mask=cond_embeds["source_mask"],
        )

        if do_cfg:
            v_uncond = transformer(
                x=latents,
                timesteps=timestep_batch,
                context=uncond_embeds["prompt_embeds"],
                padding_mask=padding_mask,
                target_input_ids=uncond_embeds["t5_input_ids"],
                target_attention_mask=uncond_embeds["t5_attn_mask"],
                source_attention_mask=uncond_embeds["source_mask"],
            )
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        latents = scheduler.step(v, i, latents)

        if step_callback is not None:
            try:
                step_callback(i + 1, num_inference_steps, latents)
            except Exception:
                pass

    return latents


@torch.no_grad()
def sample_img2img(
    transformer,
    scheduler: AnimaFlowMatchScheduler,
    init_latents: torch.Tensor,
    cond_embeds: Dict[str, torch.Tensor],
    uncond_embeds: Optional[Dict[str, torch.Tensor]],
    num_inference_steps: int,
    denoising_strength: float,
    guidance_scale: float,
    generator: torch.Generator,
    device: str,
    dtype: torch.dtype,
    step_callback: Optional[Callable] = None,
) -> torch.Tensor:
    """img2img: start from `init_latents` partially noised. Returns final latents."""
    do_cfg = guidance_scale is not None and guidance_scale > 1.0 and uncond_embeds is not None
    if init_latents.dim() == 4:
        init_latents = init_latents.unsqueeze(2)  # [1, 16, 1, H, W]

    latent_h = init_latents.shape[-2]
    latent_w = init_latents.shape[-1]
    seq_len = latent_h * latent_w
    shift = calculate_shift_anima(seq_len)
    scheduler.set_timesteps(num_inference_steps, device=torch.device(device), shift=shift)

    # Pick starting step from denoising_strength
    start_step = int(num_inference_steps * (1.0 - denoising_strength))
    start_step = max(0, min(start_step, num_inference_steps - 1))

    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
    latents = scheduler.scale_noise(init_latents.to(device, dtype), start_step, noise)

    padding_mask = _prepare_padding_mask(1, latent_h, latent_w, torch.device(device), dtype)

    for i in range(start_step, num_inference_steps):
        timestep = scheduler.get_timestep(i, device=torch.device(device), dtype=dtype)
        timestep_batch = timestep.expand(latents.shape[0])

        v_cond = transformer(
            x=latents, timesteps=timestep_batch, context=cond_embeds["prompt_embeds"],
            padding_mask=padding_mask,
            target_input_ids=cond_embeds["t5_input_ids"],
            target_attention_mask=cond_embeds["t5_attn_mask"],
            source_attention_mask=cond_embeds["source_mask"],
        )
        if do_cfg:
            v_uncond = transformer(
                x=latents, timesteps=timestep_batch, context=uncond_embeds["prompt_embeds"],
                padding_mask=padding_mask,
                target_input_ids=uncond_embeds["t5_input_ids"],
                target_attention_mask=uncond_embeds["t5_attn_mask"],
                source_attention_mask=uncond_embeds["source_mask"],
            )
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        latents = scheduler.step(v, i, latents)

        if step_callback is not None:
            try:
                step_callback(i + 1 - start_step, num_inference_steps - start_step, latents)
            except Exception:
                pass

    return latents


@torch.no_grad()
def sample_inpaint(
    transformer,
    scheduler: AnimaFlowMatchScheduler,
    init_latents: torch.Tensor,
    mask_latents: torch.Tensor,
    cond_embeds: Dict[str, torch.Tensor],
    uncond_embeds: Optional[Dict[str, torch.Tensor]],
    num_inference_steps: int,
    denoising_strength: float,
    guidance_scale: float,
    generator: torch.Generator,
    device: str,
    dtype: torch.dtype,
    step_callback: Optional[Callable] = None,
) -> torch.Tensor:
    """Latent-space inpainting via per-step blending.

    Each step we re-blend the masked region with a freshly-noised reference latent
    so the unmasked region stays close to the original.
    """
    do_cfg = guidance_scale is not None and guidance_scale > 1.0 and uncond_embeds is not None
    if init_latents.dim() == 4:
        init_latents = init_latents.unsqueeze(2)
    if mask_latents.dim() == 3:
        mask_latents = mask_latents.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif mask_latents.dim() == 4:
        mask_latents = mask_latents.unsqueeze(2)
    # Expand mask to match latent channels for broadcasting
    mask_latents = mask_latents.to(device, dtype)

    latent_h = init_latents.shape[-2]
    latent_w = init_latents.shape[-1]
    seq_len = latent_h * latent_w
    shift = calculate_shift_anima(seq_len)
    scheduler.set_timesteps(num_inference_steps, device=torch.device(device), shift=shift)

    start_step = int(num_inference_steps * (1.0 - denoising_strength))
    start_step = max(0, min(start_step, num_inference_steps - 1))

    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
    init_latents = init_latents.to(device, dtype)
    latents = scheduler.scale_noise(init_latents, start_step, noise)

    padding_mask = _prepare_padding_mask(1, latent_h, latent_w, torch.device(device), dtype)

    for i in range(start_step, num_inference_steps):
        timestep = scheduler.get_timestep(i, device=torch.device(device), dtype=dtype)
        timestep_batch = timestep.expand(latents.shape[0])

        v_cond = transformer(
            x=latents, timesteps=timestep_batch, context=cond_embeds["prompt_embeds"],
            padding_mask=padding_mask,
            target_input_ids=cond_embeds["t5_input_ids"],
            target_attention_mask=cond_embeds["t5_attn_mask"],
            source_attention_mask=cond_embeds["source_mask"],
        )
        if do_cfg:
            v_uncond = transformer(
                x=latents, timesteps=timestep_batch, context=uncond_embeds["prompt_embeds"],
                padding_mask=padding_mask,
                target_input_ids=uncond_embeds["t5_input_ids"],
                target_attention_mask=uncond_embeds["t5_attn_mask"],
                source_attention_mask=uncond_embeds["source_mask"],
            )
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        latents = scheduler.step(v, i, latents)

        # Re-blend unmasked region with a noised version of the original
        if i + 1 < num_inference_steps:
            sigma_next = scheduler.sigmas[i + 1].to(latents.dtype).to(latents.device)
            reference_noisy = (1 - sigma_next) * init_latents + sigma_next * noise
            latents = mask_latents * latents + (1 - mask_latents) * reference_noisy
        else:
            latents = mask_latents * latents + (1 - mask_latents) * init_latents

        if step_callback is not None:
            try:
                step_callback(i + 1 - start_step, num_inference_steps - start_step, latents)
            except Exception:
                pass

    return latents


def make_mask_latents(mask_image: Image.Image, latent_h: int, latent_w: int,
                      device: str, dtype: torch.dtype) -> torch.Tensor:
    """Convert a PIL mask image to a [1, 1, 1, latent_h, latent_w] tensor in [0, 1].

    White (255) = paint, Black (0) = preserve.
    """
    mask = mask_image.convert("L").resize((latent_w, latent_h), Image.NEAREST)
    arr = np.array(mask, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device, dtype)
    return t  # [1, 1, 1, H, W]
