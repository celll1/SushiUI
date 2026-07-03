"""Standalone generation operations for Krea 2.

All functions operate on bare PyTorch tensors / PIL Images (no DiffusionPipeline),
so they can be driven from the component-based staging loop in
pipeline_backends/krea2.py.

Krea 2 is a single-stream flow-matching MMDiT. Text conditioning is a stack of
hidden states tapped from 12 Qwen3-VL layers (shape (B, seq, 12, 2560)); the DiT
fuses the layer axis internally. CFG follows the Krea convention
``v = cond + guidance * (cond - uncond)``; this maps to standard UI semantics via
``guidance = cfg_scale - 1`` (so ``cfg_scale=1`` disables CFG). The base (raw)
checkpoint uses a resolution-aware timestep shift (mu from image sequence length);
the distilled (turbo) checkpoint uses a fixed ``mu=1.15``.

Logic ported from the Apache-2.0 reference (huggingface/diffusers
``pipeline_krea2.py``, krea-ai/krea-2 ``sampling.py``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

# Krea 2 fixed prompt template (Qwen-Image chat layout; pad-in-the-middle).
PROMPT_TEMPLATE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PROMPT_TEMPLATE_START_IDX = 34
PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS = 5

# Resolution-aware shift endpoints (scheduler config defaults).
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 6400
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15


# ---------------------------------------------------------------------------
# Timestep shift
# ---------------------------------------------------------------------------

def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = MAX_IMAGE_SEQ_LEN,
    base_shift: float = BASE_SHIFT,
    max_shift: float = MAX_SHIFT,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def compute_mu(image_seq_len: int, is_distilled: bool) -> float:
    """Fixed mu=1.15 for the distilled/turbo checkpoint; resolution-aware otherwise."""
    if is_distilled:
        return 1.15
    return calculate_shift(image_seq_len)


# ---------------------------------------------------------------------------
# Text encoding (Qwen3-VL, 12-layer hidden-state stack)
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_text_hidden_states(
    text_encoder,
    tokenizer,
    prompt,
    select_layers: List[int],
    max_sequence_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize into the fixed-length Krea 2 layout and tap the selected encoder
    hidden states. Returns (hidden_states (B, seq, num_layers, dim), mask (B, seq) bool)."""
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    prefix_idx = PROMPT_TEMPLATE_START_IDX

    text = [PROMPT_TEMPLATE_PREFIX + e for e in prompts]
    text_tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length + prefix_idx - PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS,
        return_tensors="pt",
    ).to(device)
    suffix_tokens = tokenizer(
        [PROMPT_TEMPLATE_SUFFIX] * len(text), return_tensors="pt"
    ).to(device)

    input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
    attention_mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()

    # Cumulative-valid-token positions (padding does not consume a position),
    # broadcast across the 3 mRoPE axes (equal for text).
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    )
    hidden_states = torch.stack([outputs.hidden_states[i] for i in select_layers], dim=2)

    hidden_states = hidden_states[:, prefix_idx:]
    attention_mask = attention_mask[:, prefix_idx:]
    return hidden_states, attention_mask


@torch.no_grad()
def encode_prompt(
    text_encoder,
    tokenizer,
    prompt,
    select_layers: List[int],
    max_sequence_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (prompt_embeds (B, seq, num_layers, dim), prompt_embeds_mask (B, seq))."""
    return get_text_hidden_states(
        text_encoder, tokenizer, prompt, select_layers, max_sequence_length, device
    )


# ---------------------------------------------------------------------------
# Latent pack / unpack + position ids
# ---------------------------------------------------------------------------

def pack_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """(B, C, H, W) -> (B, (H/p)*(W/p), C*p*p)."""
    b, c, h, w = latents.shape
    p = patch_size
    latents = latents.view(b, c, h // p, p, w // p, p)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // p) * (w // p), c * p * p)


def unpack_latents(latents: torch.Tensor, grid_h: int, grid_w: int, patch_size: int = 2) -> torch.Tensor:
    """(B, grid_h*grid_w, C*p*p) -> (B, C, 1, grid_h*p, grid_w*p)."""
    b, _, channels = latents.shape
    p = patch_size
    c = channels // (p * p)
    latents = latents.view(b, grid_h, grid_w, c, p, p)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, c, 1, grid_h * p, grid_w * p)


def prepare_position_ids(text_seq_len: int, grid_height: int, grid_width: int, device: torch.device) -> torch.Tensor:
    """(text_seq_len + grid_h*grid_w, 3): text tokens at origin, image tokens at (0, h, w)."""
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
    image_ids = image_ids.reshape(grid_height * grid_width, 3)
    return torch.cat([text_ids, image_ids], dim=0)


def prepare_latents_txt2img(
    num_channels_latents: int,
    grid_h: int,
    grid_w: int,
    patch_size: int,
    dtype: torch.dtype,
    device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Packed Gaussian noise (1, grid_h*grid_w, C*p*p)."""
    latent_h = grid_h * patch_size
    latent_w = grid_w * patch_size
    shape = (1, num_channels_latents, latent_h, latent_w)
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return pack_latents(latents, patch_size)


# ---------------------------------------------------------------------------
# VAE encode / decode (Qwen-Image VAE, per-channel latents_mean/std)
# ---------------------------------------------------------------------------

def _vae_norm_stats(vae, device, dtype):
    z_dim = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device=device, dtype=dtype)
    std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(device=device, dtype=dtype)
    return mean, std


@torch.no_grad()
def vae_encode(vae, image: Image.Image, height: int, width: int, patch_size: int, device, dtype) -> torch.Tensor:
    """Encode a PIL image -> packed normalized latent (1, grid_h*grid_w, C*p*p)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)
    img_np = np.array(image).astype(np.float32) / 127.5 - 1.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1,3,1,H,W)
    img_t = img_t.to(device=device, dtype=vae.dtype)

    latent = vae.encode(img_t).latent_dist.mode()  # (1, z_dim, 1, h, w)
    mean, std = _vae_norm_stats(vae, latent.device, latent.dtype)
    latent = (latent - mean) / std
    latent = latent[:, :, 0]  # (1, z_dim, h, w)
    return pack_latents(latent, patch_size).to(dtype)


@torch.no_grad()
def vae_decode(vae, latents: torch.Tensor, grid_h: int, grid_w: int, patch_size: int) -> Image.Image:
    """Decode packed latents (1, grid_h*grid_w, C*p*p) -> PIL Image."""
    z = unpack_latents(latents, grid_h, grid_w, patch_size).to(vae.dtype)
    mean, std = _vae_norm_stats(vae, z.device, z.dtype)
    z = z * std + mean
    image = vae.decode(z, return_dict=False)[0][:, :, 0]  # (1, 3, H, W)
    image = image.clamp(-1.0, 1.0)
    image = (image + 1.0) * (255.0 / 2.0)
    arr = image.permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(arr[0])


def prepare_mask_latent(mask_image: Image.Image, grid_h: int, grid_w: int, device, dtype) -> torch.Tensor:
    """PIL mask (white=inpaint) -> packed (1, grid_h*grid_w, 1) at token resolution (1=inpaint)."""
    import torch.nn.functional as F

    mask_np = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    mask_small = F.interpolate(mask_t, size=(grid_h, grid_w), mode="nearest")
    return mask_small.view(1, grid_h * grid_w, 1).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# CFG blend
# ---------------------------------------------------------------------------

def _blend_guidance(
    v_cond: torch.Tensor,
    v_uncond: Optional[torch.Tensor],
    guidance: float,
    sigma_now: float,
    advanced_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Any]:
    """CFG blend. ``guidance`` is the Krea convention scale (== cfg_scale - 1).

    When uncond is None (guidance <= 0), returns v_cond unchanged. Otherwise blends
    ``v_uncond + cfg_now * (v_cond - v_uncond)`` where ``cfg_now = 1 + guidance``,
    matching the Krea velocity ``cond + guidance*(cond - uncond)`` while exposing a
    standard CFG scale to the shared Advanced-CFG schedule/threshold helpers.
    """
    if v_uncond is None or guidance <= 0.0:
        return v_cond, None

    cfg = advanced_cfg or {}
    schedule_type = cfg.get("cfg_schedule_type", "constant") or "constant"
    schedule_min = float(cfg.get("cfg_schedule_min", 1.0) or 1.0)
    schedule_max = cfg.get("cfg_schedule_max")
    schedule_power = float(cfg.get("cfg_schedule_power", 2.0) or 2.0)
    snr_alpha = float(cfg.get("cfg_rescale_snr_alpha", 0.0) or 0.0)
    dyn_percentile = float(cfg.get("dynamic_threshold_percentile", 0.0) or 0.0)
    dyn_mimic = float(cfg.get("dynamic_threshold_mimic_scale", 1.0) or 1.0)
    developer_mode = bool(cfg.get("developer_mode", False))

    from core.inference.custom_sampling import (
        calculate_cfg_metrics,
        calculate_dynamic_cfg,
        dynamic_thresholding,
    )

    cfg_base = 1.0 + guidance  # standard CFG scale

    current_snr = None
    if snr_alpha > 0.0 or developer_mode:
        uncond_norm = torch.norm(v_uncond).item()
        if uncond_norm > 1e-8:
            current_snr = (torch.norm(v_cond - v_uncond).item() ** 2) / (uncond_norm ** 2)

    cfg_now = calculate_dynamic_cfg(
        sigma=sigma_now, sigma_max=1.0, cfg_base=cfg_base,
        cfg_schedule_type=schedule_type, cfg_schedule_min=schedule_min,
        cfg_schedule_max=schedule_max, cfg_schedule_power=schedule_power,
        snr=current_snr, cfg_rescale_snr_alpha=snr_alpha,
    )

    v = v_uncond + cfg_now * (v_cond - v_uncond)
    if dyn_percentile > 0.0:
        v = dynamic_thresholding(v, percentile=dyn_percentile, clamp_value=dyn_mimic)

    cfg_metrics = (
        calculate_cfg_metrics(v_uncond, v_cond, cfg_now, developer_mode) if developer_mode else None
    )
    return v, cfg_metrics


# ---------------------------------------------------------------------------
# Denoising loop
# ---------------------------------------------------------------------------

def _set_scheduler_timesteps(scheduler, num_inference_steps: int, image_seq_len: int,
                             is_distilled: bool, device):
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    mu = compute_mu(image_seq_len, is_distilled)
    scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
    scheduler.set_begin_index(0)
    return scheduler.timesteps


@torch.no_grad()
def _run_loop(
    transformer,
    scheduler,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    neg_prompt_embeds: Optional[torch.Tensor],
    neg_prompt_embeds_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    neg_position_ids: Optional[torch.Tensor],
    timesteps,
    guidance: float,
    grid_h: int,
    grid_w: int,
    patch_size: int,
    progress_callback=None,
    advanced_cfg: Optional[Dict[str, Any]] = None,
    init_latents: Optional[torch.Tensor] = None,
    init_noise: Optional[torch.Tensor] = None,
    mask_latent: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Shared flow-matching Euler loop (txt2img / img2img / inpaint)."""
    from core.inference.cancellation import raise_if_cancelled

    num_train = scheduler.config.num_train_timesteps
    total_steps = len(timesteps)
    do_cfg = neg_prompt_embeds is not None and guidance > 0.0
    t_dtype = transformer.dtype

    for i, t in enumerate(timesteps):
        raise_if_cancelled()
        # FlowMatchEuler timesteps == sigmas * num_train_timesteps, so sigma = t/num_train
        # (robust to the trimmed img2img/inpaint schedule, unlike indexing sigmas[i]).
        sigma_now = float(t.item()) / num_train
        timestep = (t / num_train).expand(latents.shape[0]).to(t_dtype)

        v_cond = transformer(
            hidden_states=latents.to(t_dtype),
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            position_ids=position_ids,
            encoder_attention_mask=prompt_embeds_mask,
            return_dict=False,
        )[0].to(torch.float32)

        v_uncond = None
        if do_cfg:
            v_uncond = transformer(
                hidden_states=latents.to(t_dtype),
                encoder_hidden_states=neg_prompt_embeds,
                timestep=timestep,
                position_ids=neg_position_ids if neg_position_ids is not None else position_ids,
                encoder_attention_mask=neg_prompt_embeds_mask,
                return_dict=False,
            )[0].to(torch.float32)

        v, cfg_metrics = _blend_guidance(v_cond, v_uncond, guidance, sigma_now, advanced_cfg)

        # x0 estimate for preview: x_t = (1-sigma)x0 + sigma*noise, v = noise - x0.
        pred_x0 = latents - sigma_now * v

        latents = scheduler.step(v, t, latents, return_dict=False)[0]

        if mask_latent is not None and init_latents is not None and init_noise is not None:
            sigma_next = float(timesteps[i + 1].item()) / num_train if (i + 1) < total_steps else 0.0
            noised_init = (1.0 - sigma_next) * init_latents + sigma_next * init_noise
            latents = mask_latent * latents + (1.0 - mask_latent) * noised_init
            if progress_callback is not None:
                preview_x0 = mask_latent * pred_x0 + (1.0 - mask_latent) * init_latents
                progress_callback(i, total_steps, latents.detach(), cfg_metrics, preview_x0.detach())
        elif progress_callback is not None:
            progress_callback(i, total_steps, latents.detach(), cfg_metrics, pred_x0.detach())

    return latents


@torch.no_grad()
def denoise_loop(
    transformer, scheduler, latents, prompt_embeds, prompt_embeds_mask,
    neg_prompt_embeds, neg_prompt_embeds_mask, guidance, num_inference_steps,
    grid_h, grid_w, patch_size, is_distilled, device,
    progress_callback=None, advanced_cfg=None,
) -> torch.Tensor:
    """txt2img flow-matching loop."""
    timesteps = _set_scheduler_timesteps(scheduler, num_inference_steps, latents.shape[1], is_distilled, device)
    position_ids = prepare_position_ids(prompt_embeds.shape[1], grid_h, grid_w, device)
    neg_position_ids = (
        prepare_position_ids(neg_prompt_embeds.shape[1], grid_h, grid_w, device)
        if neg_prompt_embeds is not None else None
    )
    return _run_loop(
        transformer, scheduler, latents, prompt_embeds, prompt_embeds_mask,
        neg_prompt_embeds, neg_prompt_embeds_mask, position_ids, neg_position_ids,
        timesteps, guidance, grid_h, grid_w, patch_size,
        progress_callback=progress_callback, advanced_cfg=advanced_cfg,
    )


@torch.no_grad()
def denoise_loop_img2img(
    transformer, scheduler, init_latents, denoising_strength,
    prompt_embeds, prompt_embeds_mask, neg_prompt_embeds, neg_prompt_embeds_mask,
    guidance, num_inference_steps, grid_h, grid_w, patch_size, is_distilled, device,
    seed=None, progress_callback=None, advanced_cfg=None,
) -> torch.Tensor:
    """SDEdit-style img2img on the flow-matching schedule."""
    all_timesteps = _set_scheduler_timesteps(scheduler, num_inference_steps, init_latents.shape[1], is_distilled, device)
    num_train = scheduler.config.num_train_timesteps
    start_step = max(int(len(all_timesteps) * (1.0 - denoising_strength)), 1)
    timesteps = all_timesteps[start_step:]

    sigma_start = float(timesteps[0].item()) / num_train
    generator = torch.Generator(device=device).manual_seed(seed) if (seed is not None and seed >= 0) else None
    noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
    latents = (1.0 - sigma_start) * init_latents + sigma_start * noise

    # Re-align the scheduler's internal step index to the trimmed schedule.
    scheduler.set_begin_index(start_step)

    position_ids = prepare_position_ids(prompt_embeds.shape[1], grid_h, grid_w, device)
    neg_position_ids = (
        prepare_position_ids(neg_prompt_embeds.shape[1], grid_h, grid_w, device)
        if neg_prompt_embeds is not None else None
    )
    return _run_loop(
        transformer, scheduler, latents, prompt_embeds, prompt_embeds_mask,
        neg_prompt_embeds, neg_prompt_embeds_mask, position_ids, neg_position_ids,
        timesteps, guidance, grid_h, grid_w, patch_size,
        progress_callback=progress_callback, advanced_cfg=advanced_cfg,
    )


@torch.no_grad()
def denoise_loop_inpaint(
    transformer, scheduler, init_latents, mask_latent, denoising_strength,
    prompt_embeds, prompt_embeds_mask, neg_prompt_embeds, neg_prompt_embeds_mask,
    guidance, num_inference_steps, grid_h, grid_w, patch_size, is_distilled, device,
    seed=None, progress_callback=None, advanced_cfg=None,
) -> torch.Tensor:
    """Repaint-style inpaint. mask_latent: (1, grid_h*grid_w, 1), 1=inpaint."""
    all_timesteps = _set_scheduler_timesteps(scheduler, num_inference_steps, init_latents.shape[1], is_distilled, device)
    num_train = scheduler.config.num_train_timesteps
    start_step = max(int(len(all_timesteps) * (1.0 - denoising_strength)), 1)
    timesteps = all_timesteps[start_step:]

    sigma_start = float(timesteps[0].item()) / num_train
    generator = torch.Generator(device=device).manual_seed(seed) if (seed is not None and seed >= 0) else None
    init_noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
    latents = (1.0 - sigma_start) * init_latents + sigma_start * init_noise
    mask_latent = mask_latent.to(device=device, dtype=init_latents.dtype)

    scheduler.set_begin_index(start_step)

    position_ids = prepare_position_ids(prompt_embeds.shape[1], grid_h, grid_w, device)
    neg_position_ids = (
        prepare_position_ids(neg_prompt_embeds.shape[1], grid_h, grid_w, device)
        if neg_prompt_embeds is not None else None
    )
    return _run_loop(
        transformer, scheduler, latents, prompt_embeds, prompt_embeds_mask,
        neg_prompt_embeds, neg_prompt_embeds_mask, position_ids, neg_position_ids,
        timesteps, guidance, grid_h, grid_w, patch_size,
        progress_callback=progress_callback, advanced_cfg=advanced_cfg,
        init_latents=init_latents, init_noise=init_noise, mask_latent=mask_latent,
    )
