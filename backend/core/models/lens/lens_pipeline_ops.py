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
from core.inference.generation_timing import time_phase
from core.inference.spectrum_forecaster import build_output_forecaster


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
@time_phase("text_encode")
def encode_prompt(
    text_encoder, tokenizer, prompt, negative_prompt,
    device, dtype, max_length: int = 512,
    skip_emphasis: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Encode prompts and build CFG-batched encoder_features and encoder_mask.

    Returns:
        encoder_features: list of Tensors, each [2, S_txt, hidden_dim]  (cond first)
        encoder_mask:     BoolTensor [2, S_txt]

    When ``skip_emphasis`` is True the emphasis SYNTAX is still stripped (so the clean
    text goes to the encoder) but the per-token weights are NOT multiplied into the
    embeddings — used by NegPip, which instead carries the (possibly negative) signed
    weights on the attention value V. Default False keeps the standard path unchanged.
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

    # Apply per-token emphasis to each sample in the batch (skipped for NegPip,
    # which carries the signed weights on the attention value V instead).
    for bi, weights in enumerate(prompt_weights):
        if weights and not skip_emphasis:
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


@torch.no_grad()
def encode_nag_negative(
    text_encoder, tokenizer, nag_negative_prompt: str,
    device, dtype, max_length: int = 512,
    skip_emphasis: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Encode the NAG-negative prompt via the SAME encoder path as encode_prompt.

    Returns (nag_features, nag_mask): list of [1, S, H] tensors and a bool mask [1, S].
    Emphasis syntax is stripped/applied exactly as in encode_prompt.
    """
    prompt = nag_negative_prompt or ""
    clean, weights = _build_emphasis_lens(prompt, tokenizer, max_length)
    features, mask = _get_text_embeddings(text_encoder, tokenizer, [clean], max_length, device)
    if weights and not skip_emphasis:
        try:
            features = _apply_emphasis_lens(features, weights, dtype)
        except Exception as e:
            print(f"[Lens] NAG emphasis application failed (ignored): {e}")
    features = [f.to(dtype=dtype) for f in features]
    return features, mask.bool()


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

    ``normalize`` applies the VAE's BatchNorm on its own 2x2-packed domain and
    hands back a raw 32ch latent (design §8.4); the rearrange below is the
    backbone's separate packing.
    """
    image = image.resize((width, height), Image.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_np = np.array(image).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    latent_h = height // 16
    latent_w = width // 16

    # Encode: (1, 32, H//8, W//8)
    from core.models.components.vae_registry import normalize

    raw = vae.encode(img_tensor).latent_dist.mode()
    x = normalize(raw, vae)

    # Rearrange to transformer flat-sequence format
    x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2, h=latent_h, w=latent_w)

    return x


@torch.no_grad()
@time_phase("vae_decode")
def vae_decode(vae, latents: torch.Tensor, latent_h: int, latent_w: int, color_flatten_strength: int = 0) -> Image.Image:
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

    from core.models.components.vae_registry import denormalize

    x = denormalize(x, vae)

    # VAE decode → (1, 3, H, W)
    decoded = vae.decode(x).sample

    # Convert to PIL
    decoded = decoded.clamp(-1.0, 1.0)
    decoded01 = (decoded + 1.0) / 2.0
    if color_flatten_strength and color_flatten_strength > 0:
        from core.inference.color_flatten import flatten_chroma
        decoded01 = flatten_chroma(decoded01, color_flatten_strength)
    decoded = (decoded01 * 255.0)
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
# NAG (Normalized Attention Guidance) setup
# ---------------------------------------------------------------------------

def _maybe_setup_nag(transformer, encoder_features, encoder_mask, nag_params):
    """If NAG is active, wrap the transformer and build the batch-3 text encoding.

    nag_params (or None) is a dict with keys: nag_features (list of [1,S,H]), nag_mask
    ([1,S]), nag_scale, nag_tau, nag_alpha. Returns
    (transformer, encoder_features, encoder_mask, nag_wrapper_or_None). When NAG is off,
    everything is returned unchanged and nag_wrapper is None (default path byte-identical).
    """
    if not nag_params:
        return transformer, encoder_features, encoder_mask, None

    nag_features = nag_params.get("nag_features")
    nag_mask = nag_params.get("nag_mask")
    scale = float(nag_params.get("nag_scale", 5.0))
    if nag_features is None or nag_mask is None or scale <= 1.0:
        return transformer, encoder_features, encoder_mask, None

    from core.inference.nag_lens import LensNAGWrapper, build_nag_text_batch

    features3, mask3 = build_nag_text_batch(
        encoder_features, encoder_mask, nag_features, nag_mask,
    )
    wrapper = LensNAGWrapper(
        transformer,
        nag_scale=scale,
        nag_tau=float(nag_params.get("nag_tau", 2.5)),
        nag_alpha=float(nag_params.get("nag_alpha", 0.25)),
    )
    print(f"[Lens] NAG enabled: scale={scale}, tau={nag_params.get('nag_tau', 2.5)}, "
          f"alpha={nag_params.get('nag_alpha', 0.25)}")
    return wrapper, features3, mask3, wrapper


# ---------------------------------------------------------------------------
# NegPip (negative-emphasis prompting) setup
# ---------------------------------------------------------------------------

def _maybe_setup_negpip(transformer, encoder_features, encoder_mask, tokenizer,
                        negpip_params, nag_active: bool):
    """If NegPip is active, install the signed-V hook on every attention module.

    negpip_params (or None) is a dict with keys: prompt, negative_prompt,
    nag_negative_prompt, max_length. Returns the list of touched modules (for
    restoration), or None when NegPip is inactive (default path byte-identical).

    The signed weight batch is aligned to the CURRENT text batch order and the CURRENT
    (post-NAG-padding) sequence length:
        plain CFG  -> [positive, negative]
        NAG active -> [positive, negative, nag_neg]   (nag_neg appended by build_nag_text_batch)
    """
    if not negpip_params:
        return None

    from core.inference.negpip_lens import build_lens_signed_weight_batch, install_negpip

    seq_txt = encoder_features[0].shape[1]
    device = encoder_features[0].device
    dtype = encoder_features[0].dtype
    max_length = int(negpip_params.get("max_length", 512))

    prompts = [
        negpip_params.get("prompt", ""),
        negpip_params.get("negative_prompt", ""),
    ]
    if nag_active:
        prompts.append(negpip_params.get("nag_negative_prompt", "") or negpip_params.get("negative_prompt", ""))

    weights = build_lens_signed_weight_batch(
        prompts, tokenizer, seq_txt, device, dtype, max_length,
    )
    modules = install_negpip(transformer, weights)
    print(f"[Lens] NegPip enabled: signed V on text tokens, batch={weights.shape[0]}, seq_txt={seq_txt}")
    return modules


# ---------------------------------------------------------------------------
# First Block Cache (FBCache)
# ---------------------------------------------------------------------------

def _unwrap_transformer(driver):
    """Return the raw LensTransformer2DModel (whose transformer_blocks loop runs) from a
    possibly-nested NAG wrapper. LensNAGWrapper holds ``.transformer`` and delegates forward
    to it, so ``_fbcache`` / ``_fbcache_step`` must be set on this real object."""
    real = driver
    while hasattr(real, "transformer") and not hasattr(real, "transformer_blocks"):
        real = real.transformer
    return real


def _build_lens_fbcache(spectrum_params, spectrum, style_active: bool = False):
    """Build a single FBCache instance for the Lens denoise loop, or None.

    Lens runs ONE BATCHED transformer forward per step (cond/uncond — and the NAG negative —
    concatenated in the batch dim, then chunked), so a single FirstBlockCache is correct.

    FBCache is mutually exclusive with:
      (a) Spectrum -- both target the same trajectory redundancy; combining compounds error.
      (b) Block Swap -- a cache hit skips transformer_blocks[1:], desyncing the block-swap
          rotation (the offloader expects every block to run each step).
      (c) Reference-style transfer -- a cache hit skips transformer_blocks[1:] on the
          capture AND/OR inject forward, leaving the per-block K/V store only partially
          populated (or the inject forward silently reusing a stale cached residual
          instead of reading the just-captured reference), desyncing exactly like (b).
    It runs only when ALL three are off."""
    from core.inference.fbcache import build_fbcache, fbcache_active
    if spectrum_params is None or not fbcache_active(spectrum_params):
        return None
    block_swap_on = bool(spectrum_params.get("enable_block_swap", False)) and \
        int(spectrum_params.get("blocks_to_swap", 0)) > 0
    if spectrum is not None:
        print("[FBCache] Lens disabled: Spectrum is enabled (same redundancy target)")
        return None
    if block_swap_on:
        print("[FBCache] Lens disabled: Block Swap is enabled (block skip desyncs rotation)")
        return None
    if style_active:
        print("[FBCache] Lens disabled: Style transfer is active (block skip desyncs the per-block K/V store)")
        return None
    return build_fbcache(spectrum_params, label="Lens")


def _cleanup_lens_fbcache(real_transformer, fbcache):
    """Detach FBCache state so it never leaks into a later forward (VAE-adjacent or a
    subsequent generation reusing this transformer instance)."""
    if fbcache is not None:
        print(f"[FBCache] Lens summary: {fbcache.n_hits} hit(s), {fbcache.n_miss} miss(es)")
    if hasattr(real_transformer, "_fbcache"):
        real_transformer._fbcache = None
    if hasattr(real_transformer, "_fbcache_step"):
        real_transformer._fbcache_step = None


# ---------------------------------------------------------------------------
# Training-free reference-style transfer (StyleAligned/VSP-style KV injection)
# ---------------------------------------------------------------------------

def _lens_style_step(
    real_transformer,
    style_cfg,
    style_ref_x0: torch.Tensor,
    style_eps_ref: torch.Tensor,
    step_idx: int,
    total_steps: int,
    t,
    latents: torch.Tensor,
    encoder_features: List[torch.Tensor],
    encoder_mask: torch.Tensor,
    guidance_scale: float,
    img_shapes,
    advanced_cfg: Optional[Dict[str, Any]],
) -> Tuple[torch.Tensor, Any]:
    """One style-active denoise step for Lens: a REF capture forward (the style
    reference re-noised to this step's CURRENT sigma, using the TARGET's own
    POSITIVE-prompt conditioning so the image-token layout lines up exactly)
    stashes post-RoPE image-token Q/K/V per block; the COND forward then reads/
    injects them via ``inject_kv``. The UNCOND forward is always run with the
    style context disarmed (untouched), matching the Krea2/FLUX.2 wiring.

    Bypasses Lens's normal batched-CFG single-forward fast path for this step:
    capture + cond + uncond become THREE single-batch (bsz=1) forwards instead
    of one batch-2 forward, since the style hook only stashes/reads ONE
    reference K/V set and mixing it into a [cond, uncond] batched forward would
    inject the reference into the uncond branch too (undefined for StyleAligned).
    NAG/NegPip/FBCache are already disabled for the WHOLE generation whenever
    style transfer is active (see the ``style_active`` gate in each
    ``denoise_loop*`` below), so this never has to reconcile with those wrappers.

    ``encoder_features``/``encoder_mask`` are the CFG-batched (cond-first,
    uncond-second) tensors built by ``encode_prompt``; slicing ``[0:1]``/``[1:2]``
    recovers the single-batch positive/negative conditioning.

    Noising convention (verified against this loop's own scheduler stepping):
    flow-matching ``x_t = (1 - sigma) * x0 + sigma * eps``, ``sigma = t / 1000``
    -- identical to Krea2's/FLUX.2's reference-noising convention and to this
    module's own img2img/inpaint SDEdit re-noising.
    """
    from core.inference.reference_style import StyleContext

    sigma_now = float(t.item()) / 1000.0
    ref_t = (1.0 - sigma_now) * style_ref_x0 + sigma_now * style_eps_ref
    progress = style_cfg.step_progress(step_idx, total_steps)

    cond_features = [f[0:1] for f in encoder_features]
    cond_mask = encoder_mask[0:1]
    uncond_features = [f[1:2] for f in encoder_features]
    uncond_mask = encoder_mask[1:2]
    timestep1 = t.expand(1).to(latents.dtype)

    try:
        capture_ctx = StyleContext(mode="capture", config=style_cfg, progress=progress)
        real_transformer._style_ctx = capture_ctx
        real_transformer(
            hidden_states=ref_t.to(latents.dtype),
            encoder_hidden_states=cond_features,
            encoder_hidden_states_mask=cond_mask,
            timestep=timestep1 / 1000,
            img_shapes=img_shapes,
        )

        inject_ctx = StyleContext(mode="inject", config=style_cfg, store=capture_ctx.store, progress=progress)
        real_transformer._style_ctx = inject_ctx
        noise_pred_cond = real_transformer(
            hidden_states=latents.to(latents.dtype),
            encoder_hidden_states=cond_features,
            encoder_hidden_states_mask=cond_mask,
            timestep=timestep1 / 1000,
            img_shapes=img_shapes,
        )
    finally:
        real_transformer._style_ctx = None

    noise_pred_uncond = real_transformer(
        hidden_states=latents.to(latents.dtype),
        encoder_hidden_states=uncond_features,
        encoder_hidden_states_mask=uncond_mask,
        timestep=timestep1 / 1000,
        img_shapes=img_shapes,
    )

    noise_pred, _cfg_now, cfg_metrics = _apply_advanced_cfg_lens(
        noise_pred_cond, noise_pred_uncond, guidance_scale, sigma_now, 1.0, advanced_cfg,
    )

    # --- CFG-decoupled style guidance (Lens) ---
    # Disabled by default (style_guidance_scale is None/<=0): this block is
    # skipped entirely and `noise_pred`/`cfg_metrics` stay exactly the combine
    # above -- byte-identical to before this feature (zero extra forwards).
    # Enabled (>0): this function is ONLY ever called on a style-active step
    # (the caller's own ``is_step_active`` gate selects ``_lens_style_step``
    # instead of the plain branch), so no extra gating is needed here -- run a
    # 4th forward -- the SAME cond forward as ``noise_pred_cond`` above
    # (identical ``latents``/``timestep1``/``cond_features``/``cond_mask``/
    # ``img_shapes``) but with ``_style_ctx`` disarmed (already ``None`` from
    # the ``finally`` above) -- to get the cond prediction WITHOUT style
    # (cond_ns).
    #
    # FBCache note: ``_build_lens_fbcache`` unconditionally returns ``None``
    # whenever style transfer is active (see its ``style_active`` guard), so
    # ``real_transformer._fbcache`` is already ``None`` for the ENTIRE
    # style-active generation -- unlike Anima, Lens never needs a defensive
    # disarm here.
    #
    # Lens's OWN combine (``_apply_advanced_cfg_lens``) is:
    #   comb = v_uncond + cfg_now * (v_cond - v_uncond)
    #   noise_pred = comb * (||v_cond|| / ||comb||)   [+ optional thresholding]
    # Rewriting the cond term to cond' = cond_ns + (lambda/cfg_now)*(cond_s -
    # cond_ns) makes the LINEAR part of that combine reproduce the
    # style-guidance target:
    #   uncond + cfg_now*(cond' - uncond)
    # = uncond + cfg_now*(cond_ns-uncond) + cfg_now*(lambda/cfg_now)*(cond_s-cond_ns)
    # = uncond + cfg_now*(cond_ns - uncond) + lambda*(cond_s - cond_ns)
    # -- prompt guidance stays at cfg_now, style strength is lambda, decoupled
    # from cfg_now, exactly like the SDXL/Anima prototypes. The norm-rescale
    # and dynamic-thresholding steps inside ``_apply_advanced_cfg_lens`` are
    # left completely untouched -- they simply run against ``cond'`` instead
    # of the plain styled cond, exactly as Anima re-runs its OWN combine
    # helper against ``cond_rewritten`` rather than hand-rolling the
    # norm-scale/threshold logic here.
    #
    # ``_cfg_now`` above is the SAME per-step value ``_apply_advanced_cfg_lens``
    # already derived from the TRUE styled (cond_s, noise_pred_uncond) pair
    # (relevant for the SNR-rescale schedule, which reads norms of the actual
    # cond/uncond passed in) -- so any CFG schedule sees the real styled
    # output, unaffected by this rewrite. The second call below is forced to
    # ``cfg_schedule_type="constant"`` with the SAME ``_cfg_now`` (no
    # re-derivation from the rewritten cond) so it reproduces the identical
    # cfg_now while still re-applying the norm-rescale/dynamic-thresholding
    # against the corrected pred. Guarded on ``_cfg_now > 1e-6`` (else
    # ``noise_pred``/``cfg_metrics`` above stay untouched, i.e. the plain
    # styled-cond combine).
    if style_cfg.style_guidance_scale is not None and style_cfg.style_guidance_scale > 0:
        cond_s = noise_pred_cond
        cond_ns = real_transformer(
            hidden_states=latents.to(latents.dtype),
            encoder_hidden_states=cond_features,
            encoder_hidden_states_mask=cond_mask,
            timestep=timestep1 / 1000,
            img_shapes=img_shapes,
        )
        lam = style_cfg.style_guidance_scale
        if _cfg_now > 1e-6:
            cond_rewritten = cond_ns + (lam / _cfg_now) * (cond_s - cond_ns)
            forced_advanced_cfg = dict(advanced_cfg or {})
            forced_advanced_cfg["cfg_schedule_type"] = "constant"
            noise_pred, _, cfg_metrics = _apply_advanced_cfg_lens(
                cond_rewritten, noise_pred_uncond, _cfg_now, sigma_now, 1.0, forced_advanced_cfg,
            )

    return noise_pred, cfg_metrics


def _lens_style_step_multi(
    real_transformer,
    style_refs: List[Tuple[Any, torch.Tensor, torch.Tensor]],
    style_combine_mode: str,
    step_idx: int,
    total_steps: int,
    t,
    latents: torch.Tensor,
    encoder_features: List[torch.Tensor],
    encoder_mask: torch.Tensor,
    guidance_scale: float,
    img_shapes,
    advanced_cfg: Optional[Dict[str, Any]],
) -> Tuple[torch.Tensor, Any]:
    """Multi-reference (N>1) generalization of ``_lens_style_step``: ONE REF
    capture forward PER reference (each with its OWN ``StyleTransferConfig`` --
    block_range, strengths, freq curve, step gating -- all independent, and
    skipped if not step-active this step, mirroring ``_lens_style_step``'s
    single-ref ``is_step_active`` gate applied per-ref instead of globally),
    then a single COND forward reading a ``StyleContext`` holding the full
    ``refs`` list (``collect_block_refs``/``inject_kv_multi`` do the
    combining). The UNCOND forward always runs with the style context
    disarmed, exactly like ``_lens_style_step``.

    Bypasses Lens's normal batched-CFG single-forward fast path for this step
    (same reasoning as ``_lens_style_step``): N capture forwards + 1 cond +
    1 uncond, all single-batch (bsz=1), since the style hook only stashes/
    reads reference K/V for one image-token layout at a time and mixing it
    into a [cond, uncond] batched forward would inject the reference into the
    uncond branch too (undefined for StyleAligned). NAG/NegPip/FBCache are
    already disabled for the WHOLE generation whenever multi-reference style
    transfer is active (see the ``style_multi_active`` gate in each
    ``denoise_loop*`` below).

    Only ever called when ``len(style_refs) > 1`` (callers route a single
    reference through ``_lens_style_step`` instead so that exact pre-multi-ref
    code path executes byte-identically)."""
    from core.inference.reference_style import StyleContext

    sigma_now = float(t.item()) / 1000.0

    cond_features = [f[0:1] for f in encoder_features]
    cond_mask = encoder_mask[0:1]
    uncond_features = [f[1:2] for f in encoder_features]
    uncond_mask = encoder_mask[1:2]
    timestep1 = t.expand(1).to(latents.dtype)

    try:
        active_refs = []
        for cfg_i, x0_i, eps_i in style_refs:
            if not cfg_i.is_step_active(step_idx, total_steps):
                continue
            ref_t_i = (1.0 - sigma_now) * x0_i + sigma_now * eps_i
            progress_i = cfg_i.step_progress(step_idx, total_steps)
            capture_ctx_i = StyleContext(mode="capture", config=cfg_i, progress=progress_i)
            real_transformer._style_ctx = capture_ctx_i
            real_transformer(
                hidden_states=ref_t_i.to(latents.dtype),
                encoder_hidden_states=cond_features,
                encoder_hidden_states_mask=cond_mask,
                timestep=timestep1 / 1000,
                img_shapes=img_shapes,
            )
            active_refs.append((capture_ctx_i.store, cfg_i))

        if active_refs:
            overall_progress = active_refs[0][1].step_progress(step_idx, total_steps)
            real_transformer._style_ctx = StyleContext(
                mode="inject", config=active_refs[0][1], refs=active_refs,
                combine_mode=style_combine_mode, progress=overall_progress,
            )
        else:
            real_transformer._style_ctx = None

        noise_pred_cond = real_transformer(
            hidden_states=latents.to(latents.dtype),
            encoder_hidden_states=cond_features,
            encoder_hidden_states_mask=cond_mask,
            timestep=timestep1 / 1000,
            img_shapes=img_shapes,
        )
    finally:
        real_transformer._style_ctx = None

    noise_pred_uncond = real_transformer(
        hidden_states=latents.to(latents.dtype),
        encoder_hidden_states=uncond_features,
        encoder_hidden_states_mask=uncond_mask,
        timestep=timestep1 / 1000,
        img_shapes=img_shapes,
    )

    noise_pred, _cfg_now, cfg_metrics = _apply_advanced_cfg_lens(
        noise_pred_cond, noise_pred_uncond, guidance_scale, sigma_now, 1.0, advanced_cfg,
    )
    return noise_pred, cfg_metrics


# ---------------------------------------------------------------------------
# Denoising loops
# ---------------------------------------------------------------------------

@torch.no_grad()
@time_phase("denoise")
def denoise_loop(
    transformer, scheduler, latents: torch.Tensor,
    encoder_features: List[torch.Tensor], encoder_mask: torch.Tensor,
    guidance_scale: float, num_inference_steps: int,
    latent_h: int, latent_w: int,
    progress_callback=None,
    advanced_cfg: Optional[Dict[str, Any]] = None,
    spectrum_params=None,
    nag_params: Optional[Dict[str, Any]] = None,
    negpip_params: Optional[Dict[str, Any]] = None,
    tokenizer=None,
    style_cfg=None,
    style_ref_x0: Optional[torch.Tensor] = None,
    style_eps_ref: Optional[torch.Tensor] = None,
    style_refs: Optional[List[Tuple[Any, torch.Tensor, torch.Tensor]]] = None,
    style_combine_mode: str = "stack",
) -> torch.Tensor:
    """Flow-matching denoising loop for txt2img.

    ``style_refs``/``style_combine_mode`` (optional, multi-reference): a list
    of ``(StyleTransferConfig, ref_x0, ref_eps)`` triples, one per reference
    image, each keeping its OWN config (block_range, strengths, freq curve,
    step gating). Only consulted when it has 2+ entries -- ``len(style_refs)
    <= 1`` is intentionally NOT specially handled here (callers route that
    case through the ``style_cfg``/``style_ref_x0``/``style_eps_ref`` single-
    ref path instead so the exact pre-multi-ref code executes
    byte-identically). ``style_combine_mode`` selects how the N refs combine
    ("stack" or "common_concept", see
    ``core.inference.reference_style.inject_kv_multi``)."""
    seq_len = latent_h * latent_w
    mu = compute_empirical_mu(seq_len, num_inference_steps)
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=latents.device, mu=mu)

    img_shapes = [(1, latent_h, latent_w)]
    total_steps = len(scheduler.timesteps)

    # Training-free reference-style transfer (see core.inference.reference_style):
    # active only when a style reference image is attached. Mutually exclusive
    # with NAG/NegPip/FBCache for the WHOLE generation (not just the style-active
    # steps) -- all three rewrite the attention-time token layout or cache
    # attention outputs, which a per-block reference K/V store cannot coexist with.
    style_active = style_cfg is not None and style_ref_x0 is not None and style_eps_ref is not None
    # Multi-reference (N>1): populated (with style_cfg/style_ref_x0/style_eps_ref
    # left None) ONLY when the caller resolved 2+ references -- a single
    # reference always goes through the style_active path above instead, so
    # that code path stays byte-identical.
    style_multi_active = style_refs is not None and len(style_refs) > 1
    if (style_active or style_multi_active) and (nag_params is not None or negpip_params is not None):
        print("[Lens] Style transfer active: disabling NAG/NegPip for this generation "
              "(both rewrite the attention-time token layout, same conflict as FBCache)")
        nag_params = None
        negpip_params = None

    transformer, encoder_features, encoder_mask, _nag_wrapper = _maybe_setup_nag(
        transformer, encoder_features, encoder_mask, nag_params,
    )
    _negpip_modules = _maybe_setup_negpip(
        transformer, encoder_features, encoder_mask, tokenizer,
        negpip_params, nag_active=_nag_wrapper is not None,
    )

    spectrum = build_output_forecaster(spectrum_params, len(scheduler.timesteps), "Lens")
    # FBCache: one instance for the batched-CFG forward. None when inactive/guarded.
    fbcache = _build_lens_fbcache(spectrum_params, spectrum, style_active=(style_active or style_multi_active))
    real_transformer = _unwrap_transformer(transformer)
    if hasattr(real_transformer, "_fbcache"):
        real_transformer._fbcache = None
    try:
        for i, t in enumerate(scheduler.timesteps):
            raise_if_cancelled()
            timestep = t.expand(2).to(latents.dtype)           # CFG: 2 × batch=1
            hidden_states = latents.repeat(2, 1, 1)            # [cond, uncond]

            # Spectrum: forecast the model output on skip steps
            spectrum_skip = spectrum is not None and not spectrum.is_anchor(i)
            style_active_step = style_active and style_cfg.is_step_active(i, total_steps)
            style_multi_active_step = style_multi_active and any(
                cfg_i.is_step_active(i, total_steps) for cfg_i, _x0_i, _eps_i in style_refs
            )
            if spectrum_skip:
                noise_pred = spectrum.forecast(i)
                cfg_metrics = None
                sigma_t = t.item() / 1000.0
            elif style_active_step:
                sigma_t = t.item() / 1000.0
                noise_pred, cfg_metrics = _lens_style_step(
                    real_transformer, style_cfg, style_ref_x0, style_eps_ref,
                    i, total_steps, t, latents, encoder_features, encoder_mask,
                    guidance_scale, img_shapes, advanced_cfg,
                )
                if spectrum is not None:
                    spectrum.record(i, noise_pred)
            elif style_multi_active_step:
                sigma_t = t.item() / 1000.0
                noise_pred, cfg_metrics = _lens_style_step_multi(
                    real_transformer, style_refs, style_combine_mode,
                    i, total_steps, t, latents, encoder_features, encoder_mask,
                    guidance_scale, img_shapes, advanced_cfg,
                )
                if spectrum is not None:
                    spectrum.record(i, noise_pred)
            else:
                # FBCache: attach this step's cache to the real transformer (None -> forward unchanged).
                if fbcache is not None:
                    real_transformer._fbcache = fbcache
                    real_transformer._fbcache_step = i
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
                if spectrum is not None:
                    spectrum.record(i, noise_pred)

            # pred_x0 = x_t - σ·v  (Flow Matching clean-image estimate)
            pred_x0 = latents - sigma_t * noise_pred

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if progress_callback is not None:
                progress_callback(i, total_steps, latents.detach(), cfg_metrics, pred_x0.detach())
    finally:
        # Defensive clear: never let a stale StyleContext leak into a later forward
        # (e.g. a subsequent non-style generation reusing this transformer instance,
        # or an exception/cancellation mid-loop) -- mirrors the Z-Image style commit's
        # defensive clear.
        if hasattr(real_transformer, "_style_ctx"):
            real_transformer._style_ctx = None

    _cleanup_lens_fbcache(real_transformer, fbcache)
    if _negpip_modules is not None:
        from core.inference.negpip_lens import restore_negpip
        restore_negpip(_negpip_modules)
    if _nag_wrapper is not None:
        _nag_wrapper.restore()
    return latents


@torch.no_grad()
@time_phase("denoise")
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
    spectrum_params=None,
    nag_params: Optional[Dict[str, Any]] = None,
    negpip_params: Optional[Dict[str, Any]] = None,
    tokenizer=None,
    style_cfg=None,
    style_ref_x0: Optional[torch.Tensor] = None,
    style_eps_ref: Optional[torch.Tensor] = None,
    style_refs: Optional[List[Tuple[Any, torch.Tensor, torch.Tensor]]] = None,
    style_combine_mode: str = "stack",
) -> torch.Tensor:
    """SDEdit-style img2img on flow-matching schedule.

    ``style_refs``/``style_combine_mode`` (optional, multi-reference): see
    ``denoise_loop``'s docstring; only consulted when ``style_refs`` has 2+
    entries."""
    seq_len = latent_h * latent_w
    mu = compute_empirical_mu(seq_len, num_inference_steps)
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=init_latents.device, mu=mu)

    style_active = style_cfg is not None and style_ref_x0 is not None and style_eps_ref is not None
    style_multi_active = style_refs is not None and len(style_refs) > 1
    if (style_active or style_multi_active) and (nag_params is not None or negpip_params is not None):
        print("[Lens] Style transfer active: disabling NAG/NegPip for this generation "
              "(both rewrite the attention-time token layout, same conflict as FBCache)")
        nag_params = None
        negpip_params = None

    transformer, encoder_features, encoder_mask, _nag_wrapper = _maybe_setup_nag(
        transformer, encoder_features, encoder_mask, nag_params,
    )
    _negpip_modules = _maybe_setup_negpip(
        transformer, encoder_features, encoder_mask, tokenizer,
        negpip_params, nag_active=_nag_wrapper is not None,
    )

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

    spectrum = build_output_forecaster(spectrum_params, len(timesteps_to_use), "Lens")
    # FBCache: one instance for the batched-CFG forward. None when inactive/guarded.
    fbcache = _build_lens_fbcache(spectrum_params, spectrum, style_active=(style_active or style_multi_active))
    real_transformer = _unwrap_transformer(transformer)
    if hasattr(real_transformer, "_fbcache"):
        real_transformer._fbcache = None
    try:
        for i, t in enumerate(timesteps_to_use):
            raise_if_cancelled()
            timestep = t.expand(2).to(latents.dtype)
            hidden_states = latents.repeat(2, 1, 1)

            # Spectrum: forecast the model output on skip steps
            spectrum_skip = spectrum is not None and not spectrum.is_anchor(i)
            style_active_step = style_active and style_cfg.is_step_active(i, total_steps)
            style_multi_active_step = style_multi_active and any(
                cfg_i.is_step_active(i, total_steps) for cfg_i, _x0_i, _eps_i in style_refs
            )
            if spectrum_skip:
                noise_pred = spectrum.forecast(i)
                cfg_metrics = None
                sigma_t = t.item() / 1000.0
            elif style_active_step:
                sigma_t = t.item() / 1000.0
                noise_pred, cfg_metrics = _lens_style_step(
                    real_transformer, style_cfg, style_ref_x0, style_eps_ref,
                    i, total_steps, t, latents, encoder_features, encoder_mask,
                    guidance_scale, img_shapes, advanced_cfg,
                )
                if spectrum is not None:
                    spectrum.record(i, noise_pred)
            elif style_multi_active_step:
                sigma_t = t.item() / 1000.0
                noise_pred, cfg_metrics = _lens_style_step_multi(
                    real_transformer, style_refs, style_combine_mode,
                    i, total_steps, t, latents, encoder_features, encoder_mask,
                    guidance_scale, img_shapes, advanced_cfg,
                )
                if spectrum is not None:
                    spectrum.record(i, noise_pred)
            else:
                # FBCache: attach this step's cache to the real transformer (None -> forward unchanged).
                if fbcache is not None:
                    real_transformer._fbcache = fbcache
                    real_transformer._fbcache_step = i
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
                if spectrum is not None:
                    spectrum.record(i, noise_pred)

            pred_x0 = latents - sigma_t * noise_pred

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if progress_callback is not None:
                progress_callback(i, total_steps, latents.detach(), cfg_metrics, pred_x0.detach())
    finally:
        if hasattr(real_transformer, "_style_ctx"):
            real_transformer._style_ctx = None

    _cleanup_lens_fbcache(real_transformer, fbcache)
    if _negpip_modules is not None:
        from core.inference.negpip_lens import restore_negpip
        restore_negpip(_negpip_modules)
    if _nag_wrapper is not None:
        _nag_wrapper.restore()
    return latents


@torch.no_grad()
@time_phase("denoise")
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
    spectrum_params=None,
    nag_params: Optional[Dict[str, Any]] = None,
    negpip_params: Optional[Dict[str, Any]] = None,
    tokenizer=None,
    style_cfg=None,
    style_ref_x0: Optional[torch.Tensor] = None,
    style_eps_ref: Optional[torch.Tensor] = None,
    style_refs: Optional[List[Tuple[Any, torch.Tensor, torch.Tensor]]] = None,
    style_combine_mode: str = "stack",
) -> torch.Tensor:
    """Repaint-style inpaint on flow-matching schedule.

    mask_latent: float tensor (1, latent_h * latent_w, 1)  — 1.0 = inpaint, 0.0 = keep.

    ``style_refs``/``style_combine_mode`` (optional, multi-reference): see
    ``denoise_loop``'s docstring; only consulted when ``style_refs`` has 2+
    entries.
    """
    seq_len = latent_h * latent_w
    mu = compute_empirical_mu(seq_len, num_inference_steps)
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=init_latents.device, mu=mu)

    style_active = style_cfg is not None and style_ref_x0 is not None and style_eps_ref is not None
    style_multi_active = style_refs is not None and len(style_refs) > 1
    if (style_active or style_multi_active) and (nag_params is not None or negpip_params is not None):
        print("[Lens] Style transfer active: disabling NAG/NegPip for this generation "
              "(both rewrite the attention-time token layout, same conflict as FBCache)")
        nag_params = None
        negpip_params = None

    transformer, encoder_features, encoder_mask, _nag_wrapper = _maybe_setup_nag(
        transformer, encoder_features, encoder_mask, nag_params,
    )
    _negpip_modules = _maybe_setup_negpip(
        transformer, encoder_features, encoder_mask, tokenizer,
        negpip_params, nag_active=_nag_wrapper is not None,
    )

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

    spectrum = build_output_forecaster(spectrum_params, len(timesteps_to_use), "Lens")
    # FBCache: one instance for the batched-CFG forward. None when inactive/guarded.
    fbcache = _build_lens_fbcache(spectrum_params, spectrum, style_active=(style_active or style_multi_active))
    real_transformer = _unwrap_transformer(transformer)
    if hasattr(real_transformer, "_fbcache"):
        real_transformer._fbcache = None
    try:
        for i, t in enumerate(timesteps_to_use):
            raise_if_cancelled()
            timestep = t.expand(2).to(latents.dtype)
            hidden_states = latents.repeat(2, 1, 1)

            # Spectrum: forecast the model output on skip steps
            spectrum_skip = spectrum is not None and not spectrum.is_anchor(i)
            style_active_step = style_active and style_cfg.is_step_active(i, total_steps)
            style_multi_active_step = style_multi_active and any(
                cfg_i.is_step_active(i, total_steps) for cfg_i, _x0_i, _eps_i in style_refs
            )
            if spectrum_skip:
                noise_pred = spectrum.forecast(i)
                cfg_metrics = None
                sigma_t = t.item() / 1000.0
            elif style_active_step:
                sigma_t = t.item() / 1000.0
                noise_pred, cfg_metrics = _lens_style_step(
                    real_transformer, style_cfg, style_ref_x0, style_eps_ref,
                    i, total_steps, t, latents, encoder_features, encoder_mask,
                    guidance_scale, img_shapes, advanced_cfg,
                )
                if spectrum is not None:
                    spectrum.record(i, noise_pred)
            elif style_multi_active_step:
                sigma_t = t.item() / 1000.0
                noise_pred, cfg_metrics = _lens_style_step_multi(
                    real_transformer, style_refs, style_combine_mode,
                    i, total_steps, t, latents, encoder_features, encoder_mask,
                    guidance_scale, img_shapes, advanced_cfg,
                )
                if spectrum is not None:
                    spectrum.record(i, noise_pred)
            else:
                # FBCache: attach this step's cache to the real transformer (None -> forward unchanged).
                if fbcache is not None:
                    real_transformer._fbcache = fbcache
                    real_transformer._fbcache_step = i
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
                if spectrum is not None:
                    spectrum.record(i, noise_pred)

            pred_x0 = latents - sigma_t * noise_pred

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Repaint: replace non-masked region with noised init at current t level
            noised_init = (1.0 - sigma_t) * init_latents + sigma_t * init_noise
            latents = mask_latent * latents + (1.0 - mask_latent) * noised_init

            if progress_callback is not None:
                # Blend pred_x0 with known region for a geometry-aware preview
                preview_x0 = mask_latent * pred_x0 + (1.0 - mask_latent) * init_latents
                progress_callback(i, total_steps, latents.detach(), cfg_metrics, preview_x0.detach())
    finally:
        if hasattr(real_transformer, "_style_ctx"):
            real_transformer._style_ctx = None

    _cleanup_lens_fbcache(real_transformer, fbcache)
    if _negpip_modules is not None:
        from core.inference.negpip_lens import restore_negpip
        restore_negpip(_negpip_modules)
    if _nag_wrapper is not None:
        _nag_wrapper.restore()
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
