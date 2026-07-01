"""Standalone generation operations for Ideogram 4.

All functions operate on bare PyTorch tensors and PIL Images — no DiffusionPipeline
dependency — so they can be called from the component-based staging loop in
pipeline.py without instantiating the reference Ideogram4Pipeline.

Ideogram 4 is a single-stream flow-matching DiT with asymmetric classifier-free
guidance: a conditional `transformer` consumes a packed [text][image] sequence,
and a separate `unconditional_transformer` denoises the image-only tokens with
zeroed text features. The two velocities are blended per step as
``v = gw * v_cond + (1 - gw) * v_uncond`` — algebraically identical to standard
CFG ``v_uncond + gw * (v_cond - v_uncond)`` with ``cfg = gw``.

Logic ported from the Apache-2.0 reference (huggingface/diffusers
``pipeline_ideogram4.py`` and ideogram-oss/ideogram4).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from core.inference.spectrum_forecaster import build_output_forecaster

from .vendor.transformer import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
)

# Hidden states of these Qwen3-VL decoder layers are concatenated to form the
# per-token text conditioning consumed by the Ideogram4 transformer.
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

# Latent geometry: VAE 8x downscale + 2x2 patchify => 16px grid, 128 packed channels.
VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2
GRID_ALIGN = VAE_SCALE_FACTOR * PATCH_SIZE  # 16
LATENT_DIM = 128


# ---------------------------------------------------------------------------
# Flow-matching schedule (logit-normal, resolution-aware)
# ---------------------------------------------------------------------------

def logit_normal_sigmas(
    num_inference_steps: int,
    mu: float,
    std: float = 1.0,
    logsnr_min: float = -15.0,
    logsnr_max: float = 18.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Length-`num_inference_steps` decreasing sigma schedule (1=noise, 0=data)."""
    intervals = torch.linspace(0.0, 1.0, num_inference_steps + 1, dtype=torch.float64)
    z = torch.special.ndtri(intervals)
    y = mu + std * z
    t = 1.0 - torch.special.expit(y)
    t_min = 1.0 / (1.0 + math.exp(0.5 * logsnr_max))
    t_max = 1.0 / (1.0 + math.exp(0.5 * logsnr_min))
    t = t.clamp(t_min, t_max)
    sigmas = (1.0 - t).flip(0)
    sigmas = sigmas[:-1].to(dtype=torch.float32, device=device)
    return sigmas


def resolution_aware_mu(
    height: int,
    width: int,
    base_mu: float,
    base_resolution: Tuple[int, int] = (512, 512),
) -> float:
    """Shift the schedule mean as a function of image resolution."""
    num_pixels = height * width
    base_pixels = base_resolution[0] * base_resolution[1]
    return base_mu + 0.5 * math.log(num_pixels / base_pixels)


def setup_schedule(
    scheduler,
    num_inference_steps: int,
    height: int,
    width: int,
    mu: float,
    std: float,
    device,
):
    """Configure the scheduler with the resolution-aware logit-normal schedule."""
    schedule_mu = resolution_aware_mu(height=height, width=width, base_mu=mu)
    sigmas = logit_normal_sigmas(num_inference_steps, schedule_mu, std=std, device=device)
    scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
    return scheduler.timesteps


def resolve_guidance_schedule(
    num_inference_steps: int,
    guidance_scale: Optional[float],
    guidance_schedule: Optional[List[float]],
) -> List[float]:
    """Return a per-step guidance weight list of length `num_inference_steps`."""
    if guidance_schedule is not None:
        if len(guidance_schedule) != num_inference_steps:
            raise ValueError(
                f"guidance_schedule length {len(guidance_schedule)} != num_inference_steps {num_inference_steps}"
            )
        return [float(g) for g in guidance_schedule]
    if guidance_scale is None:
        raise ValueError("One of guidance_scale / guidance_schedule must be set.")
    return [float(guidance_scale)] * num_inference_steps


# ---------------------------------------------------------------------------
# Packed-sequence layout
# ---------------------------------------------------------------------------

def _prepare_ids(
    text_lengths: List[int],
    grid_h: int,
    grid_w: int,
    max_text_tokens: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the packed `[left-pad][text][image]` layout.

    Returns `position_ids` (3-axis MRoPE), `segment_ids` (block-diagonal mask) and
    `indicator` (per-token text/image/pad role).
    """
    batch_size = len(text_lengths)
    num_image_tokens = grid_h * grid_w
    total_seq_len = max_text_tokens + num_image_tokens

    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
    segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
    indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

    for b, num_text in enumerate(text_lengths):
        offset = max_text_tokens - num_text

        text_pos = torch.arange(num_text)
        text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
        position_ids[b, offset : offset + num_text] = text_pos_3d
        position_ids[b, offset + num_text :] = image_pos

        indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
        indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

        segment_ids[b, offset : offset + num_text + num_image_tokens] = 1

    return position_ids.to(device), segment_ids.to(device), indicator.to(device)


@torch.no_grad()
def _get_text_encoder_hidden_states(
    text_encoder,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_2d: torch.Tensor,
) -> List[torch.Tensor]:
    """Run the Qwen3-VL decoder layers, returning hidden states at each activation layer."""
    import inspect

    from transformers.masking_utils import create_causal_mask

    language_model = text_encoder.language_model

    inputs_embeds = language_model.embed_tokens(token_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    # create_causal_mask's signature differs across transformers versions
    # (`input_embeds` vs `inputs_embeds`, and a required `cache_position` in 5.x).
    mask_params = inspect.signature(create_causal_mask).parameters
    emb_key = "input_embeds" if "input_embeds" in mask_params else "inputs_embeds"
    mask_kwargs = {
        "config": language_model.config,
        emb_key: inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": text_position_ids,
    }
    if "cache_position" in mask_params:
        mask_kwargs["cache_position"] = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        )
    causal_mask = create_causal_mask(**mask_kwargs)
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: Dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        layer_out = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        if layer_idx in tap_set:
            captured[layer_idx] = hidden_states

    return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]


@torch.no_grad()
def encode_text_layers(text_encoder, tokenizer, prompt: str, max_sequence_length: int = 512):
    """Encode one caption to the 13-layer Qwen3-VL hidden states for training cache.

    Returns (stacked [13, n_text, 4096] float32 CPU, mask [n_text] bool CPU). The
    13 layers are concatenated to the 53248-dim conditioning in train_step (matching
    the inference `encode_prompt` ordering). Text encoder must be on its device.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    toks = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n = int(toks.shape[0])
    if n > max_sequence_length:
        toks = toks[-max_sequence_length:]
        n = max_sequence_length
    te_device = text_encoder.device
    token_ids = toks.unsqueeze(0).to(te_device)
    attention_mask = torch.ones(1, n, dtype=torch.long, device=te_device)
    pos = torch.arange(n, device=te_device).unsqueeze(0)
    selected = _get_text_encoder_hidden_states(text_encoder, token_ids, attention_mask, pos)
    stacked = torch.stack([s[0] for s in selected], dim=0)  # [13, n, 4096]
    return stacked.float().cpu(), torch.ones(n, dtype=torch.bool)


def concat_layer_features(encoder_features: torch.Tensor) -> torch.Tensor:
    """Concatenate batched 13-layer features [B, 13, L, 4096] -> [B, L, 53248].

    Matches the inference `encode_prompt` ordering (stack then permute/reshape so
    the 13 layer values are interleaved per channel).
    """
    B, num_layers, L, D = encoder_features.shape
    return encoder_features.permute(0, 2, 3, 1).reshape(B, L, D * num_layers)


def build_training_conditioning(
    text_features: torch.Tensor,
    text_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> Dict[str, Any]:
    """Build the packed conditional inputs for a training batch.

    Mirrors `encode_prompt` packing but for batched, pre-encoded text features
    (concatenated 13-layer Qwen3-VL hidden states). Real text tokens (text_mask==1)
    get sequential positions and the LLM-token indicator; padding is masked out via
    indicator (0) and segment id (-1). Image tokens are appended after the text
    region with the image-grid positions.

    Args:
        text_features: (B, L, llm_features_dim) concatenated text hidden states.
        text_mask:     (B, L) 1 for real text tokens, 0 for padding.
        grid_h, grid_w: latent grid (height//16, width//16).

    Returns dict consumable by the transformer denoise forward (cond branch):
        llm_features, position_ids, segment_ids, indicator, max_text_tokens.
    """
    device = text_features.device
    B, L, _ = text_features.shape
    num_image = grid_h * grid_w
    total = L + num_image
    mask_long = text_mask.to(device=device, dtype=torch.long)

    # Image position ids (t=0, h, w) offset to stay disjoint from text positions.
    h_idx = torch.arange(grid_h, device=device).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w, device=device).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET  # (num_image, 3)

    position_ids = torch.zeros(B, total, 3, dtype=torch.long, device=device)
    segment_ids = torch.full((B, total), SEQUENCE_PADDING_INDICATOR, dtype=torch.long, device=device)
    indicator = torch.zeros(B, total, dtype=torch.long, device=device)

    # Text region: sequential position per real token (cumsum-1 over the mask).
    text_pos = (mask_long.cumsum(dim=1) - 1).clamp(min=0)  # (B, L)
    position_ids[:, :L, 0] = text_pos
    position_ids[:, :L, 1] = text_pos
    position_ids[:, :L, 2] = text_pos
    real_text = mask_long.bool()
    indicator[:, :L][real_text] = LLM_TOKEN_INDICATOR
    segment_ids[:, :L][real_text] = 1

    # Image region.
    position_ids[:, L:] = image_pos.unsqueeze(0).expand(B, -1, -1)
    indicator[:, L:] = OUTPUT_IMAGE_INDICATOR
    segment_ids[:, L:] = 1

    image_feature_padding = torch.zeros(
        B, num_image, text_features.shape[-1], dtype=text_features.dtype, device=device
    )
    llm_features = torch.cat([text_features, image_feature_padding], dim=1)

    return {
        "llm_features": llm_features,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "indicator": indicator,
        "max_text_tokens": L,
    }


@torch.no_grad()
def encode_prompt(
    text_encoder,
    tokenizer,
    prompt,
    grid_h: int,
    grid_w: int,
    max_sequence_length: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Encode the prompt(s) into the packed conditioning for the dual-branch loop.

    The text encoder must already be on `device`. Returns a dict with the
    conditional packed inputs (`llm_features`, `position_ids`, `segment_ids`,
    `indicator`) and the image-only unconditional inputs (`neg_*`).
    """
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    batch_size = len(prompts)
    num_image_tokens = grid_h * grid_w

    token_ids = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
    text_position_ids = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
    text_lengths: List[int] = []
    for b, text_prompt in enumerate(prompts):
        messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        toks = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        n = int(toks.shape[0])
        if n > max_sequence_length:
            # Keep the most recent tokens (chat suffix carries the generation prompt).
            toks = toks[-max_sequence_length:]
            n = max_sequence_length
        text_lengths.append(n)
        offset = max_sequence_length - n
        token_ids[b, offset:] = toks
        attention_mask[b, offset:] = 1
        text_position_ids[b, offset:] = torch.arange(n)

    te_device = text_encoder.device
    token_ids = token_ids.to(te_device)
    attention_mask = attention_mask.to(te_device)
    text_position_ids = text_position_ids.to(te_device)

    selected = _get_text_encoder_hidden_states(
        text_encoder, token_ids, attention_mask, text_position_ids
    )
    text_features = torch.stack(selected, dim=0).permute(1, 2, 3, 0).reshape(batch_size, max_sequence_length, -1)
    text_features = (text_features * attention_mask.to(text_features.dtype).unsqueeze(-1)).to(torch.float32)
    text_features = text_features.to(device)

    position_ids, segment_ids, indicator = _prepare_ids(
        text_lengths, grid_h, grid_w, max_sequence_length, device
    )

    image_feature_padding = torch.zeros(
        batch_size, num_image_tokens, text_features.shape[-1], dtype=text_features.dtype, device=device
    )
    llm_features = torch.cat([text_features, image_feature_padding], dim=1)

    # Unconditional (image-only) branch: zeroed text features and the image-region slices.
    neg_llm_features = torch.zeros(
        batch_size, num_image_tokens, llm_features.shape[-1], dtype=llm_features.dtype, device=device
    )
    neg_position_ids = position_ids[:, max_sequence_length:]
    neg_segment_ids = segment_ids[:, max_sequence_length:]
    neg_indicator = indicator[:, max_sequence_length:]

    return {
        "llm_features": llm_features,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "indicator": indicator,
        "neg_llm_features": neg_llm_features,
        "neg_position_ids": neg_position_ids,
        "neg_segment_ids": neg_segment_ids,
        "neg_indicator": neg_indicator,
        "max_text_tokens": max_sequence_length,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Latent helpers
# ---------------------------------------------------------------------------

def prepare_latents(
    grid_h: int,
    grid_w: int,
    dtype: torch.dtype,
    device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Random noise latents for txt2img. Shape: (1, grid_h * grid_w, 128)."""
    num_image_tokens = grid_h * grid_w
    shape = (1, num_image_tokens, LATENT_DIM)
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)
    return randn_tensor(shape, generator=generator, device=device, dtype=dtype)


def _bn_stats(vae, device, dtype):
    bn = vae.bn
    mean = bn.running_mean.view(1, 1, -1).to(device=device, dtype=dtype)
    std = torch.sqrt(bn.running_var + vae.config.batch_norm_eps).view(1, 1, -1).to(device=device, dtype=dtype)
    return mean, std


@torch.no_grad()
def vae_encode(vae, image: Image.Image, height: int, width: int, device, dtype) -> torch.Tensor:
    """Encode a PIL image -> packed normalized latent (1, grid_h*grid_w, 128)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)
    img_np = np.array(image).astype(np.float32) / 127.5 - 1.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=vae.dtype)

    grid_h = height // GRID_ALIGN
    grid_w = width // GRID_ALIGN
    patch = PATCH_SIZE

    # (1, 32, grid_h*2, grid_w*2)
    raw = vae.encode(img_tensor).latent_dist.mode()
    ae_channels = raw.shape[1]

    # Patchify (inverse of the decode unpatchify): -> (1, grid_h*grid_w, 128)
    z = raw.view(1, ae_channels, grid_h, patch, grid_w, patch)
    z = z.permute(0, 2, 4, 3, 5, 1).contiguous()
    z = z.view(1, grid_h * grid_w, ae_channels * patch * patch)

    mean, std = _bn_stats(vae, z.device, z.dtype)
    z = (z - mean) / std
    return z.to(dtype)


@torch.no_grad()
def vae_decode(vae, latents: torch.Tensor, grid_h: int, grid_w: int) -> Image.Image:
    """Decode packed latents (1, grid_h*grid_w, 128) -> PIL Image."""
    z = latents.to(vae.dtype)
    mean, std = _bn_stats(vae, z.device, z.dtype)
    z = z * std + mean

    patch = PATCH_SIZE
    ae_channels = z.shape[-1] // (patch * patch)
    z = z.view(1, grid_h, grid_w, patch, patch, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    z = z.view(1, ae_channels, grid_h * patch, grid_w * patch)

    decoded = vae.decode(z).sample
    decoded = decoded.clamp(-1.0, 1.0)
    decoded = (decoded + 1.0) * (255.0 / 2.0)
    decoded = decoded.permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(decoded[0])


def prepare_mask_latent(
    mask_image: Image.Image, grid_h: int, grid_w: int, device, dtype
) -> torch.Tensor:
    """Convert a PIL mask (white=inpaint) to a (1, grid_h*grid_w, 1) tensor (1=inpaint)."""
    import torch.nn.functional as F

    mask_gray = mask_image.convert("L")
    mask_np = np.array(mask_gray).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    mask_small = F.interpolate(mask_tensor, size=(grid_h, grid_w), mode="nearest")
    return mask_small.view(1, grid_h * grid_w, 1).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Guidance / CFG
# ---------------------------------------------------------------------------

def _blend_guidance(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    gw: float,
    sigma_now: float,
    advanced_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Any]:
    """Asymmetric CFG blend (== standard CFG with cfg=gw) plus optional schedule/threshold.

    Returns (velocity, cfg_metrics).
    """
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

    current_snr = None
    if snr_alpha > 0.0 or developer_mode:
        uncond_norm = torch.norm(v_uncond).item()
        if uncond_norm > 1e-8:
            current_snr = (torch.norm(v_cond - v_uncond).item() ** 2) / (uncond_norm ** 2)

    cfg_now = calculate_dynamic_cfg(
        sigma=sigma_now, sigma_max=1.0, cfg_base=gw,
        cfg_schedule_type=schedule_type,
        cfg_schedule_min=schedule_min,
        cfg_schedule_max=schedule_max,
        cfg_schedule_power=schedule_power,
        snr=current_snr,
        cfg_rescale_snr_alpha=snr_alpha,
    )

    v = v_uncond + cfg_now * (v_cond - v_uncond)

    if dyn_percentile > 0.0:
        v = dynamic_thresholding(v, percentile=dyn_percentile, clamp_value=dyn_mimic)

    cfg_metrics = (
        calculate_cfg_metrics(v_uncond, v_cond, cfg_now, developer_mode) if developer_mode else None
    )
    return v, cfg_metrics


def _dual_branch_velocity(
    transformer,
    unconditional_transformer,
    latents: torch.Tensor,
    cond: Dict[str, Any],
    t_model: torch.Tensor,
    gw_i: float,
    sigma_t: float,
    advanced_cfg: Optional[Dict[str, Any]],
) -> Tuple[torch.Tensor, Any]:
    """One dual-branch forward pass returning the guided velocity (float32)."""
    max_text = cond["max_text_tokens"]
    t_dtype = transformer.dtype

    # NAG: hand the doubled-batch nag-negative text features to the wrapped conditional
    # transformer for this step (no-op when NAG is off / transformer is not the wrapper).
    if cond.get("nag_llm_features") is not None and hasattr(transformer, "_procs"):
        transformer._nag_llm_features = cond["nag_llm_features"]

    # Conditional pass on the full packed [text-pad-latent][image-latent] sequence.
    text_z_padding = torch.zeros(
        latents.shape[0], max_text, latents.shape[-1], dtype=latents.dtype, device=latents.device
    )
    pos_z = torch.cat([text_z_padding, latents], dim=1).to(t_dtype)
    pos_out = transformer(
        hidden_states=pos_z,
        timestep=t_model,
        encoder_hidden_states=cond["llm_features"],
        position_ids=cond["position_ids"],
        segment_ids=cond["segment_ids"],
        indicator=cond["indicator"],
        return_dict=False,
    )[0]
    pos_v = pos_out[:, max_text:].to(torch.float32)

    # Unconditional pass on the image-only positions with zeroed text features.
    neg_out = unconditional_transformer(
        hidden_states=latents.to(unconditional_transformer.dtype),
        timestep=t_model,
        encoder_hidden_states=cond["neg_llm_features"],
        position_ids=cond["neg_position_ids"],
        segment_ids=cond["neg_segment_ids"],
        indicator=cond["neg_indicator"],
        return_dict=False,
    )[0]
    neg_v = neg_out.to(torch.float32)

    return _blend_guidance(pos_v, neg_v, gw_i, sigma_t, advanced_cfg)


# ---------------------------------------------------------------------------
# First Block Cache (FBCache) helpers
# ---------------------------------------------------------------------------

def _unwrap_ideogram4_transformer(driver):
    """Return the raw Ideogram4 transformer (whose ``forward`` runs the ``self.layers`` block
    loop) from a possibly NAG-wrapped driver. ``Ideogram4NAGWrapper`` holds ``.transformer`` and
    delegates forward to it, so ``_fbcache`` / ``_fbcache_step`` must be set on the real object.
    (NegPip installs attention processors in place and does not wrap the object.)"""
    # NOTE: peel by wrapper CLASS NAME, not `not hasattr(real, "layers")`:
    # Ideogram4NAGWrapper.__getattr__ delegates missing attrs to self.transformer, so
    # hasattr(wrapper, "layers") is True and a hasattr-based loop would stop on the wrapper
    # and set _fbcache where the real forward never reads it.
    real = driver
    while hasattr(real, "transformer") and "Wrapper" in real.__class__.__name__:
        real = real.transformer
    return real


def _build_ideogram4_fbcache(spectrum_params, spectrum, do_cfg):
    """Build FBCache instance(s) for the Ideogram4 denoise loop, or (None, None).

    Ideogram4 runs CFG as TWO SEPARATE transformer objects per step -- the conditional
    ``transformer`` and the ``unconditional_transformer`` -- each with its own denoising
    trajectory, so a hit on one must not reuse the other's residual -> two independent
    FirstBlockCache instances (uncond is None when do_cfg is off).

    FBCache is mutually exclusive with:
      (a) Spectrum -- both target the same trajectory redundancy; combining compounds error.
      (b) Block Swap -- a cache hit skips layers[1:], desyncing the block-swap rotation
          (the offloader expects every block to run each step).
    It runs only when BOTH are off. Returns (fbcache_cond, fbcache_uncond)."""
    from core.inference.fbcache import build_fbcache, fbcache_active
    if spectrum_params is None or not fbcache_active(spectrum_params):
        return None, None
    block_swap_on = bool(spectrum_params.get("enable_block_swap", False)) and \
        int(spectrum_params.get("blocks_to_swap", 0)) > 0
    if spectrum is not None:
        print("[FBCache] Ideogram4 disabled: Spectrum is enabled (same redundancy target)")
        return None, None
    if block_swap_on:
        print("[FBCache] Ideogram4 disabled: Block Swap is enabled (block skip desyncs rotation)")
        return None, None
    fbcache_cond = build_fbcache(spectrum_params, label="Ideogram4 (cond)")
    fbcache_uncond = build_fbcache(spectrum_params, label="Ideogram4 (uncond)") if do_cfg else None
    return fbcache_cond, fbcache_uncond


def _cleanup_ideogram4_fbcache(real_cond, real_uncond, fbcache_cond, fbcache_uncond):
    """Detach FBCache state from both real transformers so it never leaks into a later
    forward (VAE-adjacent, or a subsequent generation reusing these transformer objects)."""
    if fbcache_cond is not None:
        print(f"[FBCache] Ideogram4 cond summary: {fbcache_cond.n_hits} hit(s), {fbcache_cond.n_miss} miss(es)")
    if fbcache_uncond is not None:
        print(f"[FBCache] Ideogram4 uncond summary: {fbcache_uncond.n_hits} hit(s), {fbcache_uncond.n_miss} miss(es)")
    for real in (real_cond, real_uncond):
        if real is None:
            continue
        if hasattr(real, "_fbcache"):
            real._fbcache = None
        if hasattr(real, "_fbcache_step"):
            real._fbcache_step = None


# ---------------------------------------------------------------------------
# Denoising loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_loop(
    transformer,
    unconditional_transformer,
    scheduler,
    latents: torch.Tensor,
    cond: Dict[str, Any],
    timesteps,
    guidance: List[float],
    num_train_timesteps: int,
    progress_callback=None,
    advanced_cfg: Optional[Dict[str, Any]] = None,
    init_latents: Optional[torch.Tensor] = None,
    init_noise: Optional[torch.Tensor] = None,
    mask_latent: Optional[torch.Tensor] = None,
    spectrum_params=None,
) -> torch.Tensor:
    """Shared dual-branch flow-matching loop (txt2img / img2img / inpaint)."""
    from core.inference.cancellation import raise_if_cancelled
    total_steps = len(timesteps)
    spectrum = build_output_forecaster(spectrum_params, total_steps, "Ideogram4")
    batch = latents.shape[0]

    # FBCache: two instances (cond/uncond) for the two-transformer-object CFG. None when
    # inactive/guarded (Spectrum or block swap). Attached to the REAL transformers (past any
    # NAG wrapper) so the vendored forward's block loop reads _fbcache; the step index is
    # refreshed each step (mirrors how _block_offloader is attached to both transformers).
    fbcache_cond, fbcache_uncond = _build_ideogram4_fbcache(spectrum_params, spectrum, True)
    real_cond = _unwrap_ideogram4_transformer(transformer)
    real_uncond = _unwrap_ideogram4_transformer(unconditional_transformer)
    if fbcache_cond is not None:
        real_cond._fbcache = fbcache_cond
    if fbcache_uncond is not None:
        real_uncond._fbcache = fbcache_uncond

    for i, t in enumerate(timesteps):
        raise_if_cancelled()
        sigma_t = t.item() / num_train_timesteps
        t_model = (1.0 - (t.float() / num_train_timesteps)).expand(batch).to(transformer.dtype)

        # Spectrum: forecast the dual-branch velocity on skip steps
        spectrum_skip = spectrum is not None and not spectrum.is_anchor(i)
        if spectrum_skip:
            v = spectrum.forecast(i)
            cfg_metrics = None
        else:
            # FBCache: set the current step index on each transformer's cache (no-op when
            # the cache is None -> the vendored forward keeps its default block loop).
            if fbcache_cond is not None:
                real_cond._fbcache_step = i
            if fbcache_uncond is not None:
                real_uncond._fbcache_step = i
            v, cfg_metrics = _dual_branch_velocity(
                transformer, unconditional_transformer, latents, cond, t_model,
                guidance[i], sigma_t, advanced_cfg,
            )
            if spectrum is not None:
                spectrum.record(i, v)

        # x0 estimate for preview (diffusers passes -v to step; x0 = z + sigma * v).
        pred_x0 = latents + sigma_t * v

        # Scheduler integrates the ODE with the negated velocity (diffusers convention).
        latents = scheduler.step(-v, t, latents, return_dict=False)[0]

        if mask_latent is not None and init_latents is not None and init_noise is not None:
            # Repaint: keep the non-masked region pinned to the noised reference.
            noised_init = (1.0 - sigma_t) * init_latents + sigma_t * init_noise
            latents = mask_latent * latents + (1.0 - mask_latent) * noised_init
            if progress_callback is not None:
                preview_x0 = mask_latent * pred_x0 + (1.0 - mask_latent) * init_latents
                progress_callback(i, total_steps, latents.detach(), cfg_metrics, preview_x0.detach())
        elif progress_callback is not None:
            progress_callback(i, total_steps, latents.detach(), cfg_metrics, pred_x0.detach())

    _cleanup_ideogram4_fbcache(real_cond, real_uncond, fbcache_cond, fbcache_uncond)
    return latents


@torch.no_grad()
def denoise_loop(
    transformer,
    unconditional_transformer,
    scheduler,
    latents: torch.Tensor,
    cond: Dict[str, Any],
    guidance_scale: float,
    num_inference_steps: int,
    grid_h: int,
    grid_w: int,
    height: int,
    width: int,
    mu: float = 0.0,
    std: float = 1.5,
    guidance_schedule: Optional[List[float]] = None,
    progress_callback=None,
    advanced_cfg: Optional[Dict[str, Any]] = None,
    spectrum_params=None,
) -> torch.Tensor:
    """Flow-matching denoising loop for txt2img (dual-branch asymmetric CFG)."""
    device = latents.device
    timesteps = setup_schedule(scheduler, num_inference_steps, height, width, mu, std, device)
    guidance = resolve_guidance_schedule(num_inference_steps, guidance_scale, guidance_schedule)
    num_train_timesteps = scheduler.config.num_train_timesteps
    return _run_loop(
        transformer, unconditional_transformer, scheduler, latents, cond,
        timesteps, guidance, num_train_timesteps,
        progress_callback=progress_callback, advanced_cfg=advanced_cfg,
        spectrum_params=spectrum_params,
    )


@torch.no_grad()
def denoise_loop_img2img(
    transformer,
    unconditional_transformer,
    scheduler,
    init_latents: torch.Tensor,
    denoising_strength: float,
    cond: Dict[str, Any],
    guidance_scale: float,
    num_inference_steps: int,
    grid_h: int,
    grid_w: int,
    height: int,
    width: int,
    mu: float = 0.0,
    std: float = 1.5,
    seed: Optional[int] = None,
    guidance_schedule: Optional[List[float]] = None,
    progress_callback=None,
    advanced_cfg: Optional[Dict[str, Any]] = None,
    spectrum_params=None,
) -> torch.Tensor:
    """SDEdit-style img2img on the flow-matching schedule."""
    device = init_latents.device
    all_timesteps = setup_schedule(scheduler, num_inference_steps, height, width, mu, std, device)
    full_guidance = resolve_guidance_schedule(num_inference_steps, guidance_scale, guidance_schedule)
    num_train_timesteps = scheduler.config.num_train_timesteps

    start_step = max(int(len(all_timesteps) * (1.0 - denoising_strength)), 1)
    timesteps = all_timesteps[start_step:]
    guidance = full_guidance[start_step:]

    # Re-noise the encoded latent to the starting sigma (flow-matching interpolation).
    sigma_start = timesteps[0].item() / num_train_timesteps
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)
    noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
    latents = (1.0 - sigma_start) * init_latents + sigma_start * noise

    return _run_loop(
        transformer, unconditional_transformer, scheduler, latents, cond,
        timesteps, guidance, num_train_timesteps,
        progress_callback=progress_callback, advanced_cfg=advanced_cfg,
        spectrum_params=spectrum_params,
    )


@torch.no_grad()
def denoise_loop_inpaint(
    transformer,
    unconditional_transformer,
    scheduler,
    init_latents: torch.Tensor,
    mask_latent: torch.Tensor,
    denoising_strength: float,
    cond: Dict[str, Any],
    guidance_scale: float,
    num_inference_steps: int,
    grid_h: int,
    grid_w: int,
    height: int,
    width: int,
    mu: float = 0.0,
    std: float = 1.5,
    seed: Optional[int] = None,
    guidance_schedule: Optional[List[float]] = None,
    progress_callback=None,
    advanced_cfg: Optional[Dict[str, Any]] = None,
    spectrum_params=None,
) -> torch.Tensor:
    """Repaint-style inpaint on the flow-matching schedule.

    mask_latent: (1, grid_h*grid_w, 1) — 1.0 = inpaint, 0.0 = keep.
    """
    device = init_latents.device
    all_timesteps = setup_schedule(scheduler, num_inference_steps, height, width, mu, std, device)
    full_guidance = resolve_guidance_schedule(num_inference_steps, guidance_scale, guidance_schedule)
    num_train_timesteps = scheduler.config.num_train_timesteps

    start_step = max(int(len(all_timesteps) * (1.0 - denoising_strength)), 1)
    timesteps = all_timesteps[start_step:]
    guidance = full_guidance[start_step:]

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)
    init_noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)

    sigma_start = timesteps[0].item() / num_train_timesteps
    latents = (1.0 - sigma_start) * init_latents + sigma_start * init_noise
    mask_latent = mask_latent.to(device=device, dtype=init_latents.dtype)

    return _run_loop(
        transformer, unconditional_transformer, scheduler, latents, cond,
        timesteps, guidance, num_train_timesteps,
        progress_callback=progress_callback, advanced_cfg=advanced_cfg,
        spectrum_params=spectrum_params,
        init_latents=init_latents, init_noise=init_noise, mask_latent=mask_latent,
    )
