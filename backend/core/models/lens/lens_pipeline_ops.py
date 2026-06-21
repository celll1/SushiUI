"""Standalone generation operations for Microsoft/Lens.

All functions operate on bare PyTorch tensors and PIL Images — no DiffusionPipeline
dependency — so they can be called from the component-based staging loop in
pipeline.py without instantiating LensPipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

from core.inference.cancellation import raise_if_cancelled


# ---------------------------------------------------------------------------
# Constants (mirrored from vendor/pipeline.py)
# ---------------------------------------------------------------------------

_CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
_CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
DEFAULT_TXT_OFFSET = 97


# ---------------------------------------------------------------------------
# Scheduler shift calibration
# ---------------------------------------------------------------------------

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Empirical mu for FlowMatchEulerDiscreteScheduler dynamic shift (Lens calibration)."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


# ---------------------------------------------------------------------------
# Text encoding helpers
# ---------------------------------------------------------------------------

def _build_chat_inputs(tokenizer, prompts: List[str], max_sequence_length: int, device):
    rendered = []
    for prompt in prompts:
        conversation = [
            {"role": "system", "content": _CHAT_SYSTEM, "thinking": None},
            {"role": "user", "content": prompt, "thinking": None},
            {"role": "assistant", "thinking": _CHAT_ASSISTANT_THINKING, "content": ""},
        ]
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        text = text.split("<|return|>")[0]
        rendered.append(text)

    encoded = tokenizer(
        rendered,
        padding=True,
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


@torch.no_grad()
def _get_text_embeddings(
    text_encoder, tokenizer, prompts: List[str],
    max_sequence_length: int, device,
    txt_offset: int = DEFAULT_TXT_OFFSET,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    input_ids, attn_mask = _build_chat_inputs(tokenizer, prompts, max_sequence_length, device)
    layer_outputs = text_encoder.encode_layers(input_ids, attn_mask)

    if input_ids.shape[1] > txt_offset:
        features = [feat[:, txt_offset:, :].contiguous() for feat in layer_outputs]
        mask = attn_mask[:, txt_offset:].bool()
    else:
        zero_shape = (input_ids.shape[0], 0, layer_outputs[0].shape[-1])
        features = [layer_outputs[0].new_zeros(zero_shape) for _ in layer_outputs]
        mask = torch.zeros((input_ids.shape[0], 0), dtype=torch.bool, device=device)
    return features, mask


def _align_text_features(
    pos_features: List[torch.Tensor], pos_mask: torch.Tensor,
    neg_features: List[torch.Tensor], neg_mask: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Pad positive and negative encodings to the same sequence length."""
    seq_pos = pos_features[0].shape[1]
    seq_neg = neg_features[0].shape[1]
    target = max(seq_pos, seq_neg)

    def pad_feats(features, cur):
        if cur == target:
            return features
        pad_len = target - cur
        return [
            torch.cat([f, f.new_zeros((f.shape[0], pad_len, f.shape[-1]))], dim=1)
            for f in features
        ]

    def pad_mask(mask, cur):
        if cur == target:
            return mask
        return torch.cat(
            [mask, torch.zeros((mask.shape[0], target - cur), dtype=torch.bool, device=mask.device)],
            dim=1,
        )

    pos_m = pos_mask.bool()
    neg_m = neg_mask.bool()
    return pad_feats(pos_features, seq_pos), pad_mask(pos_m, seq_pos), pad_feats(neg_features, seq_neg), pad_mask(neg_m, seq_neg)


def _build_emphasis_lens(prompt: str, tokenizer, max_length: int):
    """Strip A1111-style emphasis syntax from *prompt* and return
    (clean_prompt, per_token_weights).

    Works exactly like Anima's _build_emphasis but uses the GPT-OSS tokenizer.
    Returns (original_prompt, []) when no emphasis syntax is present.
    """
    from core.prompts.prompt_parser import parse_prompt_attention

    if not prompt or ("(" not in prompt and "[" not in prompt):
        return prompt or "", []

    parsed = parse_prompt_attention(prompt)
    parsed = [(t, w) for (t, w) in parsed if t != "BREAK"]
    has_emphasis = any(abs(w - 1.0) > 1e-4 for (_, w) in parsed)
    if not has_emphasis:
        return "".join(t for (t, _) in parsed), []

    clean_parts: List[str] = []
    weights: List[float] = []
    for text, weight in parsed:
        if not text:
            continue
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            ids = []
        clean_parts.append(text)
        weights.extend([float(weight)] * len(ids))

    clean_text = "".join(clean_parts)
    if len(weights) > max_length:
        weights = weights[:max_length]
    return clean_text, weights


def _apply_emphasis_lens(
    features: List[torch.Tensor],
    token_weights: List[float],
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Apply per-token emphasis weights multiplicatively to each layer of features.

    *features* is a list of [1, S_txt, hidden_dim] tensors already trimmed to
    start at DEFAULT_TXT_OFFSET, so position 0 corresponds to the first user-
    prompt token.  We apply weights to positions 0..len(token_weights)-1.
    """
    if not token_weights:
        return features
    seq_len = features[0].shape[1]
    n = min(len(token_weights), seq_len)
    if n <= 0:
        return features
    full_w = torch.ones(seq_len, device=features[0].device, dtype=dtype)
    full_w[:n] = torch.tensor(token_weights[:n], device=features[0].device, dtype=dtype)
    w = full_w.unsqueeze(0).unsqueeze(-1)  # [1, S_txt, 1]
    return [feat * w for feat in features]


@torch.no_grad()
def encode_prompt(
    text_encoder, tokenizer, prompt, negative_prompt,
    device, dtype, max_length: int = 512,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Encode prompts and build CFG-batched encoder_features and encoder_mask.

    Returns:
        encoder_features: list of Tensors, each [2, S_txt, hidden_dim]  (cond first)
        encoder_mask:     BoolTensor [2, S_txt]
    """
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    negatives = [negative_prompt] if isinstance(negative_prompt, str) else list(negative_prompt)
    if len(negatives) == 1 and len(prompts) > 1:
        negatives = negatives * len(prompts)

    # Parse emphasis syntax before tokenising so clean_prompt goes to encoder
    clean_prompts, prompt_weights = [], []
    for p in prompts:
        clean, weights = _build_emphasis_lens(p or "", tokenizer, max_length)
        clean_prompts.append(clean)
        prompt_weights.append(weights)

    pos_features, pos_mask = _get_text_embeddings(text_encoder, tokenizer, clean_prompts, max_length, device)

    # Apply per-token emphasis to each sample in the batch
    for bi, weights in enumerate(prompt_weights):
        if weights:
            try:
                emphasised = _apply_emphasis_lens(
                    [f[bi:bi+1] for f in pos_features], weights, dtype
                )
                for li, feat in enumerate(emphasised):
                    pos_features[li][bi:bi+1] = feat
            except Exception as e:
                print(f"[Lens] emphasis application failed (ignored): {e}")

    if all(not neg.strip() for neg in negatives):
        # Empty negative → zero tensors of same shape as positive
        neg_features = [f.new_zeros(f.shape) for f in pos_features]
        neg_mask = torch.zeros_like(pos_mask, dtype=torch.bool)
    else:
        neg_features, neg_mask = _get_text_embeddings(text_encoder, tokenizer, negatives, max_length, device)

    pos_features, pos_mask, neg_features, neg_mask = _align_text_features(
        pos_features, pos_mask, neg_features, neg_mask
    )

    # Stack: cond first, uncond second (matches latents.repeat(2,1,1) ordering in loop)
    encoder_features = [
        torch.cat([pf, nf], dim=0).to(dtype=dtype)
        for pf, nf in zip(pos_features, neg_features)
    ]
    encoder_mask = torch.cat([pos_mask, neg_mask], dim=0)

    return encoder_features, encoder_mask


# ---------------------------------------------------------------------------
# Latent helpers
# ---------------------------------------------------------------------------

def _patchify(latents: torch.Tensor) -> torch.Tensor:
    """(b, c, h, w) → (b, c*4, h//2, w//2)  — group 2×2 spatial into channels."""
    b, c, h, w = latents.shape
    x = latents.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * 4, h // 2, w // 2)


def _unpatchify(latents: torch.Tensor) -> torch.Tensor:
    """(b, c*4, h, w) → (b, c, h*2, w*2)."""
    b, c4, h, w = latents.shape
    x = latents.reshape(b, c4 // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c4 // 4, h * 2, w * 2)


def _bn_normalize(x: torch.Tensor, vae) -> torch.Tensor:
    """Normalize patchified VAE latents using VAE BatchNorm running statistics."""
    bn = vae.bn
    mean = bn.running_mean.view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    var = bn.running_var.view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    std = torch.sqrt(var + vae.config.batch_norm_eps)
    return (x - mean) / std


def _bn_denormalize(x: torch.Tensor, vae) -> torch.Tensor:
    """Denormalize patchified latents (inverse of _bn_normalize)."""
    bn = vae.bn
    mean = bn.running_mean.view(1, -1, 1, 1)
    var = bn.running_var.view(1, -1, 1, 1)
    std = torch.sqrt(var + vae.config.batch_norm_eps)
    shift = (-mean).to(device=x.device, dtype=x.dtype)
    scale = (1.0 / std).to(device=x.device, dtype=x.dtype)
    return x / scale - shift  # = x * std + mean


def prepare_latents(
    height: int, width: int, dtype: torch.dtype, device, seed: Optional[int] = None
) -> torch.Tensor:
    """Random noise latents for txt2img.  Shape: (1, latent_h * latent_w, 128)."""
    latent_h = height // 16
    latent_w = width // 16
    shape = (1, latent_h * latent_w, 128)
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)
    return randn_tensor(shape, generator=generator, device=device, dtype=dtype)


@torch.no_grad()
def vae_encode(vae, image: Image.Image, height: int, width: int, device, dtype) -> torch.Tensor:
    """Encode PIL image → Lens flat-sequence latent  (1, latent_h * latent_w, 128).

    The normalization mirrors the inverse of _decode() in vendor/pipeline.py:
      raw = vae.encode(img)             # (1, 32, H//8, W//8)
      patchified = _patchify(raw)       # (1, 128, H//16, W//16)
      normalized = (p - mean) / std    # BN normalize
      unpatch = _unpatchify(normalized) # (1, 32, H//8, W//8)
      flat = rearrange to (1, latent_h*latent_w, 128)
    """
    image = image.resize((width, height), Image.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_np = np.array(image).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    latent_h = height // 16
    latent_w = width // 16

    # Encode: (1, 32, H//8, W//8)
    raw = vae.encode(img_tensor).latent_dist.mode()

    # Patchify → (1, 128, latent_h, latent_w)
    x = _patchify(raw)

    # BN normalize
    x = _bn_normalize(x, vae)

    # Unpatchify → (1, 32, latent_h*2, latent_w*2)
    x = _unpatchify(x)

    # Rearrange to transformer flat-sequence format
    x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2, h=latent_h, w=latent_w)

    return x


@torch.no_grad()
def vae_decode(vae, latents: torch.Tensor, latent_h: int, latent_w: int) -> Image.Image:
    """Decode Lens flat-sequence latents → PIL Image.

    Implements vendor/pipeline.py _decode() as a standalone function.
    """
    # (1, latent_h*latent_w, 128) → (1, 32, latent_h*2, latent_w*2)
    x = rearrange(
        latents,
        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
        p1=2, p2=2, h=latent_h, w=latent_w,
    )
    x = x.to(vae.dtype)

    # Patchify → (1, 128, latent_h, latent_w), denormalize, unpatchify
    x = _patchify(x)
    x = _bn_denormalize(x, vae)
    x = _unpatchify(x)

    # VAE decode → (1, 3, H, W)
    decoded = vae.decode(x).sample

    # Convert to PIL
    decoded = decoded.clamp(-1.0, 1.0)
    decoded = (decoded + 1.0) * (255.0 / 2.0)
    decoded = decoded.permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(decoded[0])


# ---------------------------------------------------------------------------
# Advanced CFG helper
# ---------------------------------------------------------------------------

def _apply_advanced_cfg_lens(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    guidance_scale: float,
    sigma_now: float,
    sigma_max: float = 1.0,
    advanced_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, float, Any]:
    """CFG + Lens-specific norm-scaling + optional schedule/SNR-rescale/threshold.

    Returns (noise_pred, cfg_now, cfg_metrics).
    Lens applies norm-scaled CFG: noise_pred = comb * (||v_cond|| / ||comb||).
    """
    from core.inference.custom_sampling import (
        calculate_dynamic_cfg, dynamic_thresholding, calculate_cfg_metrics,
    )

    cfg = advanced_cfg or {}
    schedule_type = cfg.get("cfg_schedule_type", "constant") or "constant"
    schedule_min = float(cfg.get("cfg_schedule_min", 1.0) or 1.0)
    schedule_max = cfg.get("cfg_schedule_max")
    schedule_power = float(cfg.get("cfg_schedule_power", 2.0) or 2.0)
    snr_alpha = float(cfg.get("cfg_rescale_snr_alpha", 0.0) or 0.0)
    dyn_percentile = float(cfg.get("dynamic_threshold_percentile", 0.0) or 0.0)
    dyn_mimic = float(cfg.get("dynamic_threshold_mimic_scale", 1.0) or 1.0)
    developer_mode = bool(cfg.get("developer_mode", False))

    current_snr = None
    if snr_alpha > 0.0 or developer_mode:
        uncond_norm = torch.norm(v_uncond).item()
        if uncond_norm > 1e-8:
            current_snr = (torch.norm(v_cond - v_uncond).item() ** 2) / (uncond_norm ** 2)

    cfg_now = calculate_dynamic_cfg(
        sigma=sigma_now, sigma_max=sigma_max, cfg_base=guidance_scale,
        cfg_schedule_type=schedule_type,
        cfg_schedule_min=schedule_min,
        cfg_schedule_max=schedule_max,
        cfg_schedule_power=schedule_power,
        snr=current_snr,
        cfg_rescale_snr_alpha=snr_alpha,
    )

    comb = v_uncond + cfg_now * (v_cond - v_uncond)

    # Lens-specific norm-scaled CFG: re-scale combined vector to cond magnitude
    cond_norm = torch.norm(v_cond, dim=-1, keepdim=True)
    comb_norm = torch.norm(comb, dim=-1, keepdim=True)
    scale = torch.where(
        comb_norm > 0,
        cond_norm / comb_norm.clamp_min(1e-12),
        torch.ones_like(comb_norm),
    )
    noise_pred = comb * scale

    if dyn_percentile > 0.0:
        noise_pred = dynamic_thresholding(noise_pred, percentile=dyn_percentile, clamp_value=dyn_mimic)

    cfg_metrics = calculate_cfg_metrics(v_uncond, v_cond, cfg_now, developer_mode) \
        if developer_mode else None
    return noise_pred, cfg_now, cfg_metrics


# ---------------------------------------------------------------------------
# Denoising loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoise_loop(
    transformer, scheduler, latents: torch.Tensor,
    encoder_features: List[torch.Tensor], encoder_mask: torch.Tensor,
    guidance_scale: float, num_inference_steps: int,
    latent_h: int, latent_w: int,
    progress_callback=None,
    advanced_cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Flow-matching denoising loop for txt2img."""
    seq_len = latent_h * latent_w
    mu = compute_empirical_mu(seq_len, num_inference_steps)
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=latents.device, mu=mu)

    img_shapes = [(1, latent_h, latent_w)]

    for i, t in enumerate(scheduler.timesteps):
        raise_if_cancelled()
        timestep = t.expand(2).to(latents.dtype)           # CFG: 2 × batch=1
        hidden_states = latents.repeat(2, 1, 1)            # [cond, uncond]

        noise_out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_features,
            encoder_hidden_states_mask=encoder_mask,
            timestep=timestep / 1000,
            img_shapes=img_shapes,
        )

        cond, uncond = noise_out.chunk(2)
        sigma_t = t.item() / 1000.0
        noise_pred, _cfg_now, cfg_metrics = _apply_advanced_cfg_lens(
            cond, uncond, guidance_scale, sigma_t, 1.0, advanced_cfg,
        )

        # pred_x0 = x_t - σ·v  (Flow Matching clean-image estimate)
        pred_x0 = latents - sigma_t * noise_pred

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if progress_callback is not None:
            progress_callback(i, num_inference_steps, latents.detach(), cfg_metrics, pred_x0.detach())

    return latents


@torch.no_grad()
def denoise_loop_img2img(
    transformer, scheduler,
    init_latents: torch.Tensor,
    denoising_strength: float,
    encoder_features: List[torch.Tensor], encoder_mask: torch.Tensor,
    guidance_scale: float, num_inference_steps: int,
    latent_h: int, latent_w: int,
    seed: Optional[int] = None,
    progress_callback=None,
    advanced_cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """SDEdit-style img2img on flow-matching schedule."""
    seq_len = latent_h * latent_w
    mu = compute_empirical_mu(seq_len, num_inference_steps)
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=init_latents.device, mu=mu)

    timesteps = scheduler.timesteps
    start_step = max(int(len(timesteps) * (1.0 - denoising_strength)), 1)
    timesteps_to_use = timesteps[start_step:]

    # Add noise at the start timestep level (flow-matching linear interpolation)
    t_start_value = timesteps_to_use[0].item() / 1000.0
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=init_latents.device).manual_seed(seed)
    noise = randn_tensor(init_latents.shape, generator=generator,
                         device=init_latents.device, dtype=init_latents.dtype)
    latents = (1.0 - t_start_value) * init_latents + t_start_value * noise

    img_shapes = [(1, latent_h, latent_w)]
    total_steps = len(timesteps_to_use)

    for i, t in enumerate(timesteps_to_use):
        raise_if_cancelled()
        timestep = t.expand(2).to(latents.dtype)
        hidden_states = latents.repeat(2, 1, 1)

        noise_out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_features,
            encoder_hidden_states_mask=encoder_mask,
            timestep=timestep / 1000,
            img_shapes=img_shapes,
        )

        cond, uncond = noise_out.chunk(2)
        sigma_t = t.item() / 1000.0
        noise_pred, _cfg_now, cfg_metrics = _apply_advanced_cfg_lens(
            cond, uncond, guidance_scale, sigma_t, 1.0, advanced_cfg,
        )

        pred_x0 = latents - sigma_t * noise_pred

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if progress_callback is not None:
            progress_callback(i, total_steps, latents.detach(), cfg_metrics, pred_x0.detach())

    return latents


@torch.no_grad()
def denoise_loop_inpaint(
    transformer, scheduler,
    init_latents: torch.Tensor,
    mask_latent: torch.Tensor,
    denoising_strength: float,
    encoder_features: List[torch.Tensor], encoder_mask: torch.Tensor,
    guidance_scale: float, num_inference_steps: int,
    latent_h: int, latent_w: int,
    seed: Optional[int] = None,
    progress_callback=None,
    advanced_cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Repaint-style inpaint on flow-matching schedule.

    mask_latent: float tensor (1, latent_h * latent_w, 1)  — 1.0 = inpaint, 0.0 = keep.
    """
    seq_len = latent_h * latent_w
    mu = compute_empirical_mu(seq_len, num_inference_steps)
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=init_latents.device, mu=mu)

    timesteps = scheduler.timesteps
    start_step = max(int(len(timesteps) * (1.0 - denoising_strength)), 1)
    timesteps_to_use = timesteps[start_step:]

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=init_latents.device).manual_seed(seed)

    # Fixed noise used for re-noising init_latents at each step (repaint trick)
    init_noise = randn_tensor(init_latents.shape, generator=generator,
                              device=init_latents.device, dtype=init_latents.dtype)

    # Starting latent: noise init_latents at the start noise level
    t_start_value = timesteps_to_use[0].item() / 1000.0
    latents = (1.0 - t_start_value) * init_latents + t_start_value * init_noise

    img_shapes = [(1, latent_h, latent_w)]
    total_steps = len(timesteps_to_use)
    mask_latent = mask_latent.to(device=init_latents.device, dtype=init_latents.dtype)

    for i, t in enumerate(timesteps_to_use):
        raise_if_cancelled()
        timestep = t.expand(2).to(latents.dtype)
        hidden_states = latents.repeat(2, 1, 1)

        noise_out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_features,
            encoder_hidden_states_mask=encoder_mask,
            timestep=timestep / 1000,
            img_shapes=img_shapes,
        )

        cond, uncond = noise_out.chunk(2)
        sigma_t = t.item() / 1000.0
        noise_pred, _cfg_now, cfg_metrics = _apply_advanced_cfg_lens(
            cond, uncond, guidance_scale, sigma_t, 1.0, advanced_cfg,
        )

        pred_x0 = latents - sigma_t * noise_pred

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Repaint: replace non-masked region with noised init at current t level
        noised_init = (1.0 - sigma_t) * init_latents + sigma_t * init_noise
        latents = mask_latent * latents + (1.0 - mask_latent) * noised_init

        if progress_callback is not None:
            # Blend pred_x0 with known region for a geometry-aware preview
            preview_x0 = mask_latent * pred_x0 + (1.0 - mask_latent) * init_latents
            progress_callback(i, total_steps, latents.detach(), cfg_metrics, preview_x0.detach())

    return latents


def prepare_mask_latent(
    mask_image: Image.Image, latent_h: int, latent_w: int,
    device, dtype,
) -> torch.Tensor:
    """Convert PIL mask (white=inpaint) to flat-sequence mask tensor.

    Returns:
        Float tensor (1, latent_h * latent_w, 1)  — 1.0=inpaint, 0.0=keep.
    """
    mask_gray = mask_image.convert("L")
    mask_np = np.array(mask_gray).astype(np.float32) / 255.0  # (H, W)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    # Resize to transformer spatial resolution (latent_h × latent_w)
    mask_small = F.interpolate(mask_tensor, size=(latent_h, latent_w), mode="nearest")
    # Flatten to sequence: (1, latent_h * latent_w, 1)
    return mask_small.view(1, latent_h * latent_w, 1).to(device=device, dtype=dtype)
