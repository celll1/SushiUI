"""Standalone generation ops for MiniT2I (pixel-space, no VAE).

Flow matching with x0 prediction: t in [0,1] (t=1 data, t=0 noise),
x_t = image*t + noise*(1-t), noise = randn*noise_scale (2.0). The model predicts
x0; the Euler velocity is v=(x0-x)/clamp(1-t,0.05); x += v*dt. CFG supports a
negative prompt (uncond = neg-prompt branch) or mask-zeroed pure-uncond fallback.
Arbitrary resolution (multiples of 16) is supported by the generalized MMJiT.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image

NOISE_SCALE = 2.0
GRID_ALIGN = 16  # patch_size


def align_to_grid(value: int, align: int = GRID_ALIGN) -> int:
    if value <= 0:
        return align
    return max(align, round(value / align) * align)


def normalize_resolution(width: int, height: int) -> tuple[int, int]:
    return align_to_grid(width), align_to_grid(height)


@torch.no_grad()
def encode_prompt(text_encoder, tokenizer, prompt, prompt_length: int, device):
    """FLAN-T5 encode -> (last_hidden_state [B,L,1024], attention_mask [B,L])."""
    if isinstance(prompt, str):
        prompt = [prompt]
    toks = tokenizer(
        prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=prompt_length,
    )
    input_ids = toks.input_ids.to(device)
    attn = toks.attention_mask.to(device)
    out = text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
    return out, attn


def image_to_tensor(image: Image.Image, height: int, width: int, device, dtype) -> torch.Tensor:
    """PIL -> normalized [-1,1] RGB tensor [1,3,H,W]."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)
    arr = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def tensor_to_image(x: torch.Tensor) -> Image.Image:
    """[-1,1] RGB tensor [1,3,H,W] -> PIL."""
    x = x.clamp(-1, 1)
    arr = (x[0] * 127.5 + 128.0).clamp(0, 255).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(arr)


def prepare_noise(height, width, device, dtype, seed: Optional[int] = None) -> torch.Tensor:
    gen = None
    if seed is not None and seed >= 0:
        gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1, 3, height, width, generator=gen, device=device, dtype=dtype) * NOISE_SCALE


@torch.no_grad()
def _predict_x0_cfg(transformer, x, t, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval):
    """CFG x0 prediction. neg_* may be None -> mask-zeroed pure uncond."""
    t_val = float(t.reshape(-1)[0].item())
    use_cfg = (cfg_scale != 1.0) and (cfg_interval[0] <= t_val <= cfg_interval[1])
    if not use_cfg:
        return transformer(x, t, text, mask)

    if neg_text is not None:
        u_text, u_mask = neg_text, neg_mask
    else:
        u_text, u_mask = text, torch.zeros_like(mask)
    # Batched cond+uncond pass.
    xx = torch.cat([x, x], dim=0)
    tt = torch.cat([t, t], dim=0)
    yy = torch.cat([text, u_text], dim=0)
    mm = torch.cat([mask, u_mask], dim=0)
    out = transformer(xx, tt, yy, mm)
    cond, uncond = out[:1], out[1:]
    return uncond + (cond - uncond) * cfg_scale


@torch.no_grad()
def _euler_run(transformer, x, ts, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval,
               start_idx=0, progress_callback=None, mask_latent=None, init_image=None, fixed_noise=None):
    """Shared Euler loop from ts[start_idx] -> 1. Returns final [-1,1] RGB.

    If mask_latent/init_image/fixed_noise are given (inpaint), the kept region is
    pinned to the noised init each step.
    """
    from core.inference.cancellation import raise_if_cancelled
    n = len(ts) - 1
    total = n - start_idx
    for j, i in enumerate(range(start_idx, n)):
        raise_if_cancelled()
        t0 = ts[i]
        t1 = ts[i + 1]
        t = t0.expand(1).to(x.dtype)
        pred_x0 = _predict_x0_cfg(transformer, x, t, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval)
        v = (pred_x0 - x) / (1.0 - t0).clamp_min(0.05)
        x = x + v * (t1 - t0)
        if mask_latent is not None:
            known = init_image * t1 + fixed_noise * (1.0 - t1)
            x = mask_latent * x + (1.0 - mask_latent) * known
        if progress_callback is not None:
            progress_callback(j, total, x.detach(), None, pred_x0.detach())
    return x.clamp(-1, 1)


@torch.no_grad()
def denoise_loop(transformer, text, mask, height, width, num_inference_steps, cfg_scale,
                 cfg_interval, device, dtype, seed=None, neg_text=None, neg_mask=None,
                 progress_callback=None):
    """txt2img: start from pure noise, integrate t:0->1."""
    x = prepare_noise(height, width, device, dtype, seed)
    ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=dtype)
    return _euler_run(transformer, x, ts, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval,
                      progress_callback=progress_callback)


@torch.no_grad()
def denoise_loop_img2img(transformer, init_image, denoising_strength, text, mask, num_inference_steps,
                         cfg_scale, cfg_interval, device, dtype, seed=None, neg_text=None, neg_mask=None,
                         progress_callback=None):
    """img2img (SDEdit): start at t_start = 1 - strength with the noised init."""
    ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=dtype)
    t_start = max(0.0, min(1.0, 1.0 - float(denoising_strength)))
    start_idx = int((ts <= t_start).sum().item()) - 1
    start_idx = max(0, min(start_idx, num_inference_steps - 1))
    noise = prepare_noise(init_image.shape[-2], init_image.shape[-1], device, dtype, seed)
    ti = ts[start_idx]
    x = init_image.to(dtype) * ti + noise * (1.0 - ti)
    return _euler_run(transformer, x, ts, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval,
                      start_idx=start_idx, progress_callback=progress_callback)


def prepare_mask(mask_image: Image.Image, height, width, device, dtype) -> torch.Tensor:
    """PIL mask (white=inpaint) -> [1,1,H,W] in {0,1}-ish, broadcast over channels."""
    m = mask_image.convert("L").resize((width, height), Image.NEAREST)
    arr = np.asarray(m).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, None].to(device=device, dtype=dtype)


@torch.no_grad()
def denoise_loop_inpaint(transformer, init_image, mask_latent, denoising_strength, text, mask,
                         num_inference_steps, cfg_scale, cfg_interval, device, dtype, seed=None,
                         neg_text=None, neg_mask=None, progress_callback=None):
    """inpaint (repaint): keep non-masked pixels pinned to the noised init each step."""
    ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=dtype)
    t_start = max(0.0, min(1.0, 1.0 - float(denoising_strength)))
    start_idx = int((ts <= t_start).sum().item()) - 1
    start_idx = max(0, min(start_idx, num_inference_steps - 1))
    fixed_noise = prepare_noise(init_image.shape[-2], init_image.shape[-1], device, dtype, seed)
    ti = ts[start_idx]
    init_image = init_image.to(dtype)
    x = init_image * ti + fixed_noise * (1.0 - ti)
    return _euler_run(transformer, x, ts, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval,
                      start_idx=start_idx, progress_callback=progress_callback,
                      mask_latent=mask_latent, init_image=init_image, fixed_noise=fixed_noise)
