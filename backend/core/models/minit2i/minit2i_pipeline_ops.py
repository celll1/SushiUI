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
from core.inference.generation_timing import time_phase
from core.inference.spectrum_forecaster import build_output_forecaster
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
@time_phase("text_encode")
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


def tensor_to_image(x: torch.Tensor, color_flatten_strength: int = 0) -> Image.Image:
    """[-1,1] RGB tensor [1,3,H,W] -> PIL. Optional post-decode chroma smoothing."""
    x = x.clamp(-1, 1)
    if color_flatten_strength and color_flatten_strength > 0:
        from core.inference.color_flatten import flatten_chroma
        x = flatten_chroma((x + 1.0) / 2.0, color_flatten_strength) * 2.0 - 1.0
    arr = (x[0] * 127.5 + 128.0).clamp(0, 255).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(arr)


def prepare_noise(height, width, device, dtype, seed: Optional[int] = None,
                  channels: int = 3, noise_scale: float = NOISE_SCALE) -> torch.Tensor:
    """Random start tensor [1, channels, height, width] * noise_scale.

    Pixel space: channels=3, noise_scale=2 (default), height/width = image dims.
    Latent space: channels=VAE latent channels, noise_scale=1, height/width = latent
    dims (image // vae_scale_factor).
    """
    gen = None
    if seed is not None and seed >= 0:
        gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1, channels, height, width, generator=gen, device=device, dtype=dtype) * noise_scale


@torch.no_grad()
def vae_encode_image(vae, image: Image.Image, height: int, width: int, device, dtype) -> torch.Tensor:
    """PIL -> normalized VAE latent [1, C, H/8, W/8] (for latent-space MiniT2I)."""
    from .minit2i_vae import normalize_latent
    px = image_to_tensor(image, height, width, device, dtype)  # [1,3,H,W] in [-1,1]
    sample = vae.encode(px).latent_dist.sample()
    return normalize_latent(sample, vae)


@torch.no_grad()
@time_phase("vae_decode")
def vae_decode_latent(vae, latent: torch.Tensor, color_flatten_strength: int = 0) -> Image.Image:
    """Normalized VAE latent [1, C, h, w] -> PIL image."""
    from .minit2i_vae import denormalize_latent
    sample = denormalize_latent(latent.to(vae.dtype), vae)
    img = vae.decode(sample).sample  # [1,3,H,W] in ~[-1,1]
    return tensor_to_image(img.float(), color_flatten_strength=color_flatten_strength)


@torch.no_grad()
def prepare_style_reference(vae, style_image: Image.Image, height: int, width: int, device, dtype,
                            is_latent: bool, channels: int, noise_scale: float, seed: Optional[int] = None):
    """Build the (fixed) reference x0 + reference noise pair for training-free style
    transfer, encoded the SAME way MiniT2I consumes images: pixel-space checkpoints
    (default, ``is_latent=False``) get the raw normalized-[-1,1] RGB pixel tensor
    (``image_to_tensor``); the optional latent-VAE variant (``is_latent=True``) gets a
    normalized VAE latent (``vae_encode_image``). Both are returned at the SAME spatial
    resolution as the target (``height``/``width`` already resolved to pixel or latent
    dims by the caller). The reference noise is drawn ONCE per generation (not per step)
    with a seed offset decorrelated from -- but reproducible alongside -- the main
    generation seed (mirrors Krea2/Z-Image's ``prepare_style_reference``); re-drawing
    fresh noise every step would make the reference K/V flicker step to step."""
    if is_latent:
        ref_x0 = vae_encode_image(vae, style_image, height, width, device, dtype)
    else:
        ref_x0 = image_to_tensor(style_image, height, width, device, dtype)
    ref_seed = None if seed is None or seed < 0 else (int(seed) + 991) % (2**32)
    eps_ref = prepare_noise(ref_x0.shape[-2], ref_x0.shape[-1], device, dtype, ref_seed,
                            channels=channels, noise_scale=noise_scale)
    return ref_x0, eps_ref


@torch.no_grad()
def _predict_x0_style_step(net, x, t, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval,
                           style_cfg, style_ref_x0, style_eps_ref, step_idx, num_steps):
    """One style-active x0-prediction step: bypasses ``_predict_x0_cfg``'s BATCHED
    ``[cond, uncond]`` forward for this step entirely -- capture forward (style
    reference re-noised to the CURRENT ``t``, using the SAME ``x_t = image*t +
    noise*(1-t)`` convention this module's own img2img/inpaint noising uses) stashes
    post-RoPE image-token Q/K/V per joint block; the COND forward then reads/injects
    them; the UNCOND forward (when CFG is active) is ALWAYS run with the style
    context disarmed (untouched) -- mirrors Krea2/FLUX.2/Z-Image's two-pass wiring.
    ``net`` must be the RAW MMJiT (NOT a NAG/NegPip wrapper): style transfer is
    mutually exclusive with NAG/NegPip for the generation (see
    ``pipeline_backends/minit2i.py``'s style gating), so bypassing the wrapper here is
    intentional, not a shortcut. The final CFG blend (``uncond + (cond-uncond)*cfg_scale``)
    is the SAME x0-prediction convention as ``_predict_x0_cfg`` -- re-noising the
    (fixed) reference to the step's noise level only changes what the model SEES as
    input, it does not change what the model predicts (x0) or how the blend combines
    cond/uncond x0 estimates."""
    from core.inference.reference_style import StyleContext
    from core.inference.style_minit2i import set_minit2i_style_context

    t_val = float(t.reshape(-1)[0].item())
    use_cfg = (cfg_scale != 1.0) and (cfg_interval[0] <= t_val <= cfg_interval[1])
    progress = style_cfg.step_progress(step_idx, num_steps)

    ref_input = (style_ref_x0 * t_val + style_eps_ref * (1.0 - t_val)).to(x.dtype)

    try:
        capture_ctx = StyleContext(mode="capture", config=style_cfg, progress=progress)
        set_minit2i_style_context(net, capture_ctx)
        net(ref_input, t, text, mask)

        inject_ctx = StyleContext(mode="inject", config=style_cfg, store=capture_ctx.store, progress=progress)
        set_minit2i_style_context(net, inject_ctx)
        cond = net(x, t, text, mask)
    finally:
        set_minit2i_style_context(net, None)

    if not use_cfg:
        return cond

    if neg_text is not None:
        u_text, u_mask = neg_text, neg_mask
    else:
        u_text, u_mask = text, torch.zeros_like(mask)
    uncond = net(x, t, u_text, u_mask)
    return uncond + (cond - uncond) * cfg_scale


@torch.no_grad()
def _unwrap_minit2i_net(transformer):
    """Return the raw MMJiT net (whose forward reads _fbcache) from the call target.

    The call target is either the raw MiniT2IMMJiTModel (``.model.net`` == MMJiT) or a
    NAG/NegPip wrapper (``.net`` == the same MMJiT). This is the same object block swap's
    _block_offloader is attached to, so _fbcache / _fbcache_step must live here too."""
    net = getattr(transformer, "net", None)
    if net is not None and hasattr(net, "double_blocks"):
        return net
    model = getattr(transformer, "model", None)
    if model is not None:
        net = getattr(model, "net", None)
        if net is not None and hasattr(net, "double_blocks"):
            return net
    return None


def _build_minit2i_fbcache(net, spectrum, spectrum_params, style_active: bool = False):
    """Build a FirstBlockCache for the MiniT2I denoise loop, or None.

    MiniT2I runs ONE BATCHED transformer forward per step (cond+uncond concatenated in
    _predict_x0_cfg; NAG expands the batch inside a single forward too), so a SINGLE
    FirstBlockCache instance is correct. Mutually exclusive with:
      (a) Spectrum -- both target the same trajectory redundancy; combining compounds error.
      (b) Block Swap -- a cache hit skips double_blocks[1:], desyncing the swap rotation
          (the offloader expects every block to run each step).
      (c) Style transfer -- a cache hit ALSO skips double_blocks[1:], which would desync
          the per-block style capture/inject store across steps (mirrors Z-Image/FLUX.2).
    Runs only when ALL are off. Ensures no stale cache leaks in either way."""
    from core.inference.fbcache import build_fbcache, fbcache_active
    if net is not None and hasattr(net, "_fbcache"):
        net._fbcache = None
    if spectrum_params is None or not fbcache_active(spectrum_params):
        return None
    if net is None:
        print("[FBCache] MiniT2I disabled: could not locate MMJiT net on call target")
        return None
    block_swap_on = bool(spectrum_params.get("enable_block_swap", False)) and \
        int(spectrum_params.get("blocks_to_swap", 0)) > 0
    if spectrum is not None:
        print("[FBCache] MiniT2I disabled: Spectrum is enabled (same redundancy target)")
        return None
    if style_active:
        print("[FBCache] MiniT2I disabled: Style transfer is enabled (block skip desyncs the per-block style store)")
        return None
    if block_swap_on or getattr(net, "_block_offloader", None) is not None:
        print("[FBCache] MiniT2I disabled: Block Swap is enabled (block skip desyncs rotation)")
        return None
    return build_fbcache(spectrum_params, label="MiniT2I")


def _cleanup_minit2i_fbcache(net, fbcache):
    """Detach FBCache state so it never leaks into a later forward / generation."""
    if fbcache is not None:
        print(f"[FBCache] MiniT2I summary: {fbcache.n_hits} hit(s), {fbcache.n_miss} miss(es)")
    if net is not None:
        if hasattr(net, "_fbcache"):
            net._fbcache = None
        if hasattr(net, "_fbcache_step"):
            net._fbcache_step = None


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
               start_idx=0, progress_callback=None, mask_latent=None, init_image=None, fixed_noise=None,
               clamp_output=True, spectrum_params=None,
               style_cfg=None, style_ref_x0=None, style_eps_ref=None):
    """Shared Euler loop from ts[start_idx] -> 1.

    Returns the final tensor. clamp_output=True clamps to [-1,1] (pixel RGB);
    latent-space callers pass clamp_output=False (latents are not bounded to [-1,1]).
    If mask_latent/init_image/fixed_noise are given (inpaint), the kept region is
    pinned to the noised init each step.

    Training-free reference-style transfer (``style_cfg`` is a
    ``core.inference.reference_style.StyleTransferConfig``, non-None only when a style
    reference is attached, built by ``pipeline_backends/minit2i.py``'s
    ``_minit2i_style_config``): the style-patched double blocks (see
    ``core.inference.style_minit2i``) are installed ONCE for the whole loop (not
    per-step -- installing/restoring per step would be wasteful and the patch is a
    no-op via ``_style_ctx=None`` when a step isn't style-active), and at each
    ``style_cfg.is_step_active`` step the BATCHED ``_predict_x0_cfg`` fast path is
    bypassed in favor of ``_predict_x0_style_step``'s separate capture/cond/uncond
    forwards. FBCache is disabled for the whole loop whenever style is active (see
    ``_build_minit2i_fbcache``). Uninstalled in a ``finally`` so a raised exception
    mid-loop never leaves the style patch (or a stale ``_style_ctx``) installed on the
    net for a later, non-style generation.
    """
    from core.inference.cancellation import raise_if_cancelled
    n = len(ts) - 1
    total = n - start_idx
    spectrum = build_output_forecaster(spectrum_params, total, "MiniT2I")
    style_active = style_cfg is not None and style_ref_x0 is not None and style_eps_ref is not None
    # FBCache: single instance (batched CFG). None when inactive/guarded (Spectrum/Block Swap/Style).
    _fb_net = _unwrap_minit2i_net(transformer)
    fbcache = _build_minit2i_fbcache(_fb_net, spectrum, spectrum_params, style_active=style_active)
    if fbcache is not None and _fb_net is not None:
        _fb_net._fbcache = fbcache

    _style_saved = None
    if style_active:
        if _fb_net is None:
            print("[Style] MiniT2I disabled: could not locate MMJiT net on call target")
            style_active = False
        else:
            from core.inference.style_minit2i import install_minit2i_style_blocks
            _style_saved = install_minit2i_style_blocks(_fb_net)

    try:
        for j, i in enumerate(range(start_idx, n)):
            raise_if_cancelled()
            t0 = ts[i]
            t1 = ts[i + 1]
            t = t0.expand(1).to(x.dtype)
            spectrum_skip = spectrum is not None and not spectrum.is_anchor(j)
            if spectrum_skip:
                pred_x0 = spectrum.forecast(j)
            elif style_active and style_cfg.is_step_active(j, total):
                pred_x0 = _predict_x0_style_step(
                    _fb_net, x, t, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval,
                    style_cfg, style_ref_x0, style_eps_ref, j, total,
                )
                if spectrum is not None:
                    spectrum.record(j, pred_x0)
            else:
                # FBCache: hand the net the current step index (mirrors _block_offloader attach).
                if fbcache is not None and _fb_net is not None:
                    _fb_net._fbcache_step = j
                pred_x0 = _predict_x0_cfg(transformer, x, t, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval)
                if spectrum is not None:
                    spectrum.record(j, pred_x0)
            v = (pred_x0 - x) / (1.0 - t0).clamp_min(0.05)
            x = x + v * (t1 - t0)
            if mask_latent is not None:
                known = init_image * t1 + fixed_noise * (1.0 - t1)
                x = mask_latent * x + (1.0 - mask_latent) * known
            if progress_callback is not None:
                progress_callback(j, total, x.detach(), None, pred_x0.detach())
    finally:
        if _style_saved is not None:
            from core.inference.style_minit2i import restore_minit2i_style_blocks
            restore_minit2i_style_blocks(_fb_net, _style_saved)
    _cleanup_minit2i_fbcache(_fb_net, fbcache)
    return x.clamp(-1, 1) if clamp_output else x


@torch.no_grad()
@time_phase("denoise")
def denoise_loop(transformer, text, mask, height, width, num_inference_steps, cfg_scale,
                 cfg_interval, device, dtype, seed=None, neg_text=None, neg_mask=None,
                 progress_callback=None, channels: int = 3, noise_scale: float = NOISE_SCALE,
                 clamp_output: bool = True, spectrum_params=None,
                 style_cfg=None, style_ref_x0=None, style_eps_ref=None):
    """txt2img: start from pure noise, integrate t:0->1.

    Pixel: channels=3, noise_scale=2, height/width = image dims, clamp_output=True.
    Latent: channels=C, noise_scale=1, height/width = latent dims, clamp_output=False.
    """
    x = prepare_noise(height, width, device, dtype, seed, channels=channels, noise_scale=noise_scale)
    ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=dtype)
    return _euler_run(transformer, x, ts, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval,
                      progress_callback=progress_callback, clamp_output=clamp_output, spectrum_params=spectrum_params,
                      style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref)


@torch.no_grad()
@time_phase("denoise")
def denoise_loop_img2img(transformer, init_image, denoising_strength, text, mask, num_inference_steps,
                         cfg_scale, cfg_interval, device, dtype, seed=None, neg_text=None, neg_mask=None,
                         progress_callback=None, noise_scale: float = NOISE_SCALE, clamp_output: bool = True, spectrum_params=None,
                         style_cfg=None, style_ref_x0=None, style_eps_ref=None):
    """img2img (SDEdit): start at t_start = 1 - strength with the noised init.

    init_image is the working tensor: pixel RGB [1,3,H,W] or (latent) a normalized
    VAE latent [1,C,h,w]. channels/noise scale follow init_image / noise_scale.
    """
    ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=dtype)
    t_start = max(0.0, min(1.0, 1.0 - float(denoising_strength)))
    start_idx = int((ts <= t_start).sum().item()) - 1
    start_idx = max(0, min(start_idx, num_inference_steps - 1))
    noise = prepare_noise(init_image.shape[-2], init_image.shape[-1], device, dtype, seed,
                          channels=init_image.shape[1], noise_scale=noise_scale)
    ti = ts[start_idx]
    x = init_image.to(dtype) * ti + noise * (1.0 - ti)
    return _euler_run(transformer, x, ts, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval,
                      start_idx=start_idx, progress_callback=progress_callback, clamp_output=clamp_output, spectrum_params=spectrum_params,
                      style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref)


def prepare_mask(mask_image: Image.Image, height, width, device, dtype) -> torch.Tensor:
    """PIL mask (white=inpaint) -> [1,1,H,W] in {0,1}-ish, broadcast over channels."""
    m = mask_image.convert("L").resize((width, height), Image.NEAREST)
    arr = np.asarray(m).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, None].to(device=device, dtype=dtype)


@torch.no_grad()
@time_phase("denoise")
def denoise_loop_inpaint(transformer, init_image, mask_latent, denoising_strength, text, mask,
                         num_inference_steps, cfg_scale, cfg_interval, device, dtype, seed=None,
                         neg_text=None, neg_mask=None, progress_callback=None,
                         noise_scale: float = NOISE_SCALE, clamp_output: bool = True, spectrum_params=None,
                         style_cfg=None, style_ref_x0=None, style_eps_ref=None):
    """inpaint (repaint): keep non-masked pixels pinned to the noised init each step.

    init_image is the working tensor (pixel RGB or normalized VAE latent); mask_latent
    must be at the same spatial resolution (1 = regenerate, 0 = keep).
    """
    ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=dtype)
    t_start = max(0.0, min(1.0, 1.0 - float(denoising_strength)))
    start_idx = int((ts <= t_start).sum().item()) - 1
    start_idx = max(0, min(start_idx, num_inference_steps - 1))
    fixed_noise = prepare_noise(init_image.shape[-2], init_image.shape[-1], device, dtype, seed,
                                channels=init_image.shape[1], noise_scale=noise_scale)
    ti = ts[start_idx]
    init_image = init_image.to(dtype)
    x = init_image * ti + fixed_noise * (1.0 - ti)
    return _euler_run(transformer, x, ts, text, mask, neg_text, neg_mask, cfg_scale, cfg_interval,
                      start_idx=start_idx, progress_callback=progress_callback,
                      mask_latent=mask_latent, init_image=init_image, fixed_noise=fixed_noise,
                      clamp_output=clamp_output, spectrum_params=spectrum_params,
                      style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref)
