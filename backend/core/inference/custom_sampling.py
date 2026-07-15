"""Custom sampling loop for advanced prompt control

This module provides a custom sampling loop that allows:
- Prompt editing (changing prompts mid-generation)
- Fine-grained control over each denoising step
- Access to intermediate latents

Based on diffusers' pipeline implementation but with added flexibility.
"""

import torch


def _add_generation_warning(message: str, code: str = None) -> None:
    """Best-effort: record a feature-degradation warning for the current generation.

    Lazily imported so this inference module never hard-depends on the api
    package at import time. Never raises.
    """
    try:
        from api.generation_status import add_warning
        add_warning(message, code=code)
    except Exception:
        pass


def _make_pid_decode_progress(progress_callback):
    """Build the decoupled `(cur, total, label)` decode-progress adapter PiD's
    `pid_final_decode` expects, from the denoise-phase `progress_callback`
    (see `create_progress_callback_factory` in api/generation_utils.py).

    Returns None when `progress_callback` is None, or when it doesn't declare
    support for the `phase_label` kwarg (via the `_supports_phase_label`
    marker) — no exception-based probing, so a genuine TypeError raised
    inside the callback is never silently swallowed.
    """
    if progress_callback is None or not getattr(progress_callback, "_supports_phase_label", False):
        return None

    def _decode_cb(cur, total, label):
        progress_callback(cur, total, None, phase_label=label)

    return _decode_cb


def get_inpaint_use_dedicated_model_setting() -> bool:
    """Get the inpaint_use_dedicated_model setting from database.

    Returns:
        bool: True if dedicated 9ch inpaint model should be used (legacy SD/SDXL method),
              False if mask blending should be used (default, same as Z-Image/FLUX.2).
    """
    try:
        from database.database import get_gallery_db_sync
        from database.models import UserSettings

        db = get_gallery_db_sync()
        try:
            settings = db.query(UserSettings).first()
            if settings and settings.inpaint_use_dedicated_model is not None:
                return settings.inpaint_use_dedicated_model
            return False  # Default: mask blending
        finally:
            db.close()
    except Exception as e:
        print(f"[CustomSampling] Warning: Could not read inpaint_use_dedicated_model setting: {e}")
        return False  # Default: mask blending
from typing import Optional, Callable, Dict, Any, Union, List, Tuple
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


def prepare_reference_guide_latents(
    ref_guide_configs,
    pipeline,
    width,
    height,
    device,
    dtype,
    generator,
):
    """Prepare reference guide clean latents by VAE-encoding the reference images.

    Args:
        ref_guide_configs: List of dicts with "image" (PIL), "strength", "start_step", "end_step"
        pipeline: Pipeline with VAE
        width, height: Target resolution
        device, dtype: Torch device and dtype
        generator: Torch generator for noise

    Returns:
        List of dicts with "clean_latent", "noise", "strength", "start_step", "end_step"
        or empty list if no ref guides
    """
    if not ref_guide_configs:
        return []

    vae_dtype = next(pipeline.vae.parameters()).dtype
    ref_guides = []

    for idx, cfg in enumerate(ref_guide_configs):
        image = cfg["image"]
        # Ensure RGB (drop alpha channel if RGBA)
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Resize to target resolution
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        # PIL -> tensor -> VAE encode
        img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        img_tensor = img_tensor * 2.0 - 1.0  # [-1, 1]

        with torch.no_grad():
            clean_latent = pipeline.vae.encode(
                img_tensor.to(device=device, dtype=vae_dtype)
            ).latent_dist.sample(generator)
            clean_latent = (clean_latent - (getattr(pipeline.vae.config, "shift_factor", None) or 0.0)) * pipeline.vae.config.scaling_factor
            clean_latent = clean_latent.to(dtype=dtype)

        # Generate noise for re-noising at each step
        noise = torch.randn(clean_latent.shape, generator=generator, device=device, dtype=dtype)

        # Normalize start/end from 0-1000 to 0.0-1.0
        start_frac = cfg.get("start_step", 0) / 1000.0
        end_frac = cfg.get("end_step", 1000) / 1000.0

        ref_guides.append({
            "clean_latent": clean_latent,
            "noise": noise,
            "strength": cfg.get("strength", 0.4),
            "start_frac": start_frac,
            "end_frac": end_frac,
        })
        print(f"[RefGuide {idx}] Prepared: strength={cfg.get('strength', 0.4)}, "
              f"range={start_frac:.2f}-{end_frac:.2f}, latent shape={clean_latent.shape}")

    return ref_guides


def apply_reference_guide_blend(
    latents,
    pred_original_sample,
    ref_guides,
    current_fraction,
    step_index,
    timesteps,
    scheduler,
):
    """Apply reference guide blending after scheduler.step.

    Args:
        latents: Current denoised latents
        pred_original_sample: x0 prediction (or None)
        ref_guides: List from prepare_reference_guide_latents()
        current_fraction: Current step as 0.0-1.0 fraction
        step_index: Current step index
        timesteps: Full timestep tensor
        scheduler: Diffusion scheduler

    Returns:
        (latents, pred_original_sample) - blended
    """
    if not ref_guides:
        return latents, pred_original_sample

    for rg in ref_guides:
        if rg["start_frac"] <= current_fraction <= rg["end_frac"] and rg["strength"] > 0:
            weight = rg["strength"]
            if step_index < len(timesteps) - 1:
                next_t = timesteps[step_index + 1]
                ref_at_t = scheduler.add_noise(rg["clean_latent"], rg["noise"], next_t.unsqueeze(0))
            else:
                ref_at_t = rg["clean_latent"]

            latents = (1 - weight) * latents + weight * ref_at_t

            if pred_original_sample is not None:
                pred_original_sample = (1 - weight) * pred_original_sample + weight * rg["clean_latent"]

    return latents, pred_original_sample


# ---------------------------------------------------------------------------
# Training-free reference-style transfer (SD1.5/SDXL U-Net wiring)
# ---------------------------------------------------------------------------

def prepare_style_reference_latent(image, pipeline, width, height, device, dtype, seed, ref_index: int = 0):
    """VAE-encode the style reference image to the SAME latent shape/scaling as
    the target latents (so it can be re-noised to any step's sigma and its
    self-attention K/V injected into the target's own self-attention), and draw
    the ONE fixed reference noise tensor used for every active step's
    re-noising (drawing fresh noise per step would make the injected reference
    K/V flicker step to step). Mirrors `prepare_reference_guide_latents`'
    VAE-encode convention (shift_factor + scaling_factor) and Krea2's
    `prepare_style_reference` fixed-seed-offset convention.

    Resizing the reference image to the exact target (width, height) is what
    keeps every U-Net block's spatial resolution -- and therefore its
    self-attention sequence length -- identical between the reference capture
    forward and the target's own forward, which is required for the per-block
    K/V injection to line up (see `attention_processors.UnifiedAttnProcessor`).

    ``ref_index`` (multi-reference only): decorrelates the fixed re-noising
    noise tensor across simultaneous references -- without it every reference
    would draw the EXACT same noise from the ``seed+991`` offset below, since
    that offset does not depend on which reference is being prepared.
    ``ref_index=0`` (the default, used by the single-ref ``build_style_transfer``
    caller) reproduces the pre-multi-ref ``seed+991`` offset exactly.
    """
    vae = pipeline.vae
    vae_dtype = next(vae.parameters()).dtype
    img = image.convert("RGB") if image.mode != "RGB" else image
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)

    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
    img_tensor = img_tensor * 2.0 - 1.0  # [-1, 1]

    # The txt2img path does not stage the VAE to GPU before this point (only U-Net
    # is on GPU), so the VAE is typically still on CPU here. Encoding a cuda input
    # against a cpu VAE crashes ("input is cuda, weight is cpu"), and a cpu fp16
    # conv is unsupported/very slow -- so temporarily stage the VAE to the target
    # device for the (one-image) reference encode, then restore its original
    # placement so the normal offload flow (decode-time VAE staging) is unaffected.
    _orig_vae_device = next(vae.parameters()).device
    _target_device = torch.device(device)
    _staged = _orig_vae_device != _target_device
    try:
        if _staged:
            vae.to(_target_device)
        with torch.no_grad():
            ref_x0 = vae.encode(img_tensor.to(device=_target_device, dtype=vae_dtype)).latent_dist.mode()
            ref_x0 = (ref_x0 - (getattr(vae.config, "shift_factor", None) or 0.0)) * vae.config.scaling_factor
            ref_x0 = ref_x0.to(device=_target_device, dtype=dtype)
    finally:
        if _staged:
            vae.to(_orig_vae_device)

    ref_seed = None if seed is None or seed < 0 else (int(seed) + ref_index + 991) % (2**32)
    generator = torch.Generator(device=device).manual_seed(ref_seed) if ref_seed is not None else None
    eps_ref = torch.randn(ref_x0.shape, generator=generator, device=device, dtype=dtype)
    return ref_x0, eps_ref


def build_style_transfer(params, pipeline, width, height, device, dtype, seed=-1):
    """Build a (StyleTransferConfig, ref_x0, eps_ref) triple from
    ``params["style_transfer"]`` (assembled by
    ``generation_utils.process_controlnet_configs`` from an ``is_style_transfer``
    ControlNet-shaped entry), or ``(None, None, None)`` when no style reference
    is attached.

    SD1.5/SDXL-specific note: this U-Net has NO RoPE, so (unlike Krea2/Flux2)
    ``cfg.axes_dims`` is intentionally left ``None`` and
    ``cfg.get_freq_scale_vector`` is never called for this arch -- the
    per-block hook (`attention_processors.UnifiedAttnProcessor`) substitutes a
    constant ``torch.ones(head_dim)`` frequency-scale vector directly, relying
    on block selection (``style_blocks`` / ``block_range``) + AdaIN +
    ``ref_k_strength`` for content/style control instead (StyleAligned's
    original recipe, since there is no RoPE-frequency axis to suppress).
    """
    style_dict = params.get("style_transfer")
    if not style_dict or not style_dict.get("image"):
        return None, None, None

    from core.inference.reference_style import style_config_from_dict

    cfg = style_config_from_dict(style_dict)
    ref_x0, eps_ref = prepare_style_reference_latent(
        style_dict["image"], pipeline, width, height, device, dtype, seed,
    )
    return cfg, ref_x0, eps_ref


def build_style_transfer_all(params, pipeline, width, height, device, dtype, seed=-1):
    """Build the FULL style-transfer configuration for SDXL/SD1.5 generation,
    covering both the single-reference path (legacy ``(style_cfg, style_ref_x0,
    style_eps_ref)`` triple, exactly as ``build_style_transfer`` would return)
    and the multi-reference path (``style_refs``, a list of per-ref
    ``(StyleTransferConfig, ref_x0, eps_ref)`` triples, populated ONLY when
    ``params["style_transfers"]`` has more than one valid entry). A single-entry
    ``style_transfers`` list (or the legacy singular ``style_transfer`` key) is
    intentionally routed through the single-ref triple instead (``style_refs``
    stays ``None``), so the pre-multi-ref code path executes byte-identically
    end to end.

    Returns ``(style_cfg, style_ref_x0, style_eps_ref, style_refs,
    style_combine_mode)``.
    """
    style_list = params.get("style_transfers")
    if style_list and len(style_list) > 1:
        from core.inference.reference_style import style_config_from_dict

        combine_mode = str(params.get("style_combine_mode", "stack") or "stack")
        refs = []
        for idx, style_dict in enumerate(style_list):
            if not style_dict or not style_dict.get("image"):
                continue
            # U-Net-family multi-ref strength damping: the U-Net attention path has
            # NO RoPE frequency curve (collect_block_refs uses an all-ones freq
            # vector), so it lacks the implicit high-frequency reference-Key damping
            # that RoPE DiT archs get for free. At the single-ref-tuned default
            # ref_k_strength=0.75, stacking 2+ refs collapses the target content into
            # a blob on U-Net (GPU-validated), whereas the same 0.75 stays legible on
            # DiT archs. So for the U-Net multi-ref case, when the caller did NOT set
            # an explicit strength, default it to 0.35 (GPU sweep winner: content
            # preserved, strictly >= the 0.75 result) instead of 0.75. An explicit
            # per-entry strength always wins. This is the U-Net analog of the
            # arch-agnostic multi-ref AdaIN=0 default applied at intake; it lives here
            # (not intake) because it must NOT lower strength for the DiT archs.
            if style_dict.get("ref_k_strength") is None:
                style_dict = {**style_dict, "ref_k_strength": 0.35}
            cfg = style_config_from_dict(style_dict)
            ref_x0, eps_ref = prepare_style_reference_latent(
                style_dict["image"], pipeline, width, height, device, dtype, seed, ref_index=idx,
            )
            refs.append((cfg, ref_x0, eps_ref))

        if len(refs) > 1:
            return None, None, None, refs, combine_mode
        if len(refs) == 1:
            cfg, ref_x0, eps_ref = refs[0]
            return cfg, ref_x0, eps_ref, None, combine_mode
        return None, None, None, None, combine_mode

    style_cfg, style_ref_x0, style_eps_ref = build_style_transfer(params, pipeline, width, height, device, dtype, seed)
    return style_cfg, style_ref_x0, style_eps_ref, None, "stack"


def vae_output_to_pil(
    image: torch.Tensor,
    color_flatten_strength: int = 0,
    dc_bias: Optional[torch.Tensor] = None,
) -> Image.Image:
    """Convert VAE decoder output tensor to PIL Image with robust nan/inf handling.

    Optional post-decode passes (both zero-cost when disabled):
      - ``color_flatten_strength`` (0-100): RGB-guided chroma smoothing applied to
        the [0,1] image (see core.inference.color_flatten). <=0 = no-op.
      - ``dc_bias`` [1,C,1,1]: per-channel VAE DC-drift bias subtracted from the
        image before the final clamp (img2img/inpaint drift correction).

    Args:
        image: VAE decoder output tensor [B, C, H, W] in range [-1, 1]

    Returns:
        PIL Image

    Note:
        - nan values are replaced with gray (0.5)
        - positive inf values are replaced with white (1.0)
        - negative inf values are replaced with black (0.0)
        - Valid pixels are preserved even if some pixels are invalid
    """
    # Scale from [-1, 1] to [0, 1]
    image = (image / 2 + 0.5)

    # Replace nan/inf with fallback values before clamping
    if torch.isnan(image).any() or torch.isinf(image).any():
        nan_count = torch.isnan(image).sum().item()
        inf_count = torch.isinf(image).sum().item()
        total_pixels = image.numel()
        print(f"[VAE Decode] Warning: {nan_count} nan, {inf_count} inf out of {total_pixels} pixels ({(nan_count + inf_count) / total_pixels * 100:.2f}%)")

        # Replace nan with gray (0.5), positive inf with white (1.0), negative inf with black (0.0)
        image = torch.where(torch.isnan(image), torch.tensor(0.5, device=image.device, dtype=image.dtype), image)
        image = torch.where(torch.isposinf(image), torch.tensor(1.0, device=image.device, dtype=image.dtype), image)
        image = torch.where(torch.isneginf(image), torch.tensor(0.0, device=image.device, dtype=image.dtype), image)

    image = image.clamp(0, 1)
    if color_flatten_strength and color_flatten_strength > 0:
        from core.inference.color_flatten import flatten_chroma
        image = flatten_chroma(image, color_flatten_strength)
    if dc_bias is not None:
        # Strength-independent VAE DC-drift correction: subtract the per-channel
        # round-trip bias, then re-clamp to the valid range.
        image = image - dc_bias.to(image.device, image.dtype)
    image = image.clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image[0])


def compute_vae_dc_bias(pipeline, ref_latents: torch.Tensor, input_mean: torch.Tensor, vae_shift: float) -> Optional[torch.Tensor]:
    """Per-channel VAE DC-drift bias = mean(decode(encode(input))) - mean(input), in [0,1].

    ``ref_latents`` are the SCALED init latents (== encode(input), as used for
    denoising); this reverses the scaling and runs ONE extra reference decode with
    the VAE still on GPU. Returns a [1,C,1,1] bias to subtract from the final
    decode, or None on failure. Strength-independent (corrects a VAE property).
    """
    if ref_latents is None or input_mean is None:
        return None
    try:
        lat = ref_latents / pipeline.vae.config.scaling_factor + vae_shift
        lat = lat.to(dtype=pipeline.vae.dtype)
        with torch.no_grad():
            ref = pipeline.vae.decode(lat, return_dict=True).sample
        ref_mean = (ref.float() / 2 + 0.5).clamp(0, 1).mean(dim=(0, 2, 3), keepdim=True)
        return (ref_mean - input_mean.to(ref_mean.device)).float()
    except Exception as e:
        print(f"[VAE Drift] Reference decode failed, skipping correction: {e}")
        return None


def compute_flatten_inject_steps(num_timesteps: int, last_steps: int) -> set:
    """Step indices on which the in-loop hard-flatten fires: the last ``last_steps``
    ACTUAL denoise steps the loop executes (``num_timesteps`` is ``len(timesteps)``,
    which some schedulers, e.g. DPM2, double). Relative to the real step sequence,
    NOT a fixed fraction, so it is stable across step counts / accelerators."""
    n = int(last_steps)
    if n <= 0 or num_timesteps <= 0:
        return set()
    n = min(n, num_timesteps)
    return set(range(num_timesteps - n, num_timesteps))


def _setup_inloop_flatten(pipeline, timesteps, spectrum, fbcache_ctrl,
                          flatten_in_loop, last_steps, min_region):
    """Compute the injection step set and, when accelerators are active, force
    genuine U-Net forwards on those steps. Shared by the three sampling loops.

    Accelerator interplay: Spectrum/FBCache skip or forecast U-Net forwards on
    some steps, so x0 on such a step would be synthetic. Making the injection
    steps ANCHORS (spectrum) / forced misses (fbcache) is the minimal correct
    guard - it keeps each accelerator's own fit/cache consistent (an anchor is
    recorded, a miss captures the cache) while guaranteeing the hard-flatten sees
    a real x0. Note spectrum_tail (default 0.12) already forces the tail to real
    passes, so with default spectrum the last ~3-4 of 28 steps are real anyway;
    this guard covers larger N, lower spectrum_tail, and FBCache (no tail).

    Returns ``(inject_steps:set, vae_shift:float)``.
    """
    if not flatten_in_loop:
        return set(), 0.0
    inject = compute_flatten_inject_steps(len(timesteps), last_steps)
    vae_shift = getattr(pipeline.vae.config, "shift_factor", None) or 0.0
    if spectrum is not None:
        spectrum.anchors = set(spectrum.anchors) | inject
    if fbcache_ctrl is not None:
        fbcache_ctrl.force_real_steps = set(fbcache_ctrl.force_real_steps) | inject
    print(f"[InLoopFlatten] enabled: inject on steps {sorted(inject)} of "
          f"{len(timesteps)} (min_region={min_region})")
    return inject, vae_shift


def inloop_hard_flatten_step(
    pipeline,
    latents: torch.Tensor,
    pred_original_sample: torch.Tensor,
    min_region_frac: float,
    vae_shift: float,
) -> Tuple[torch.Tensor, bool]:
    """In-loop hard-flatten latent injection (SD1.5/SDXL, validated in proto2).

    Decode the current x0 prediction (``pred_original_sample``, a SCALED latent) to
    pixels, detect + hard-replace the flat background region with its dominant
    colour (feathered - see ``core.inference.inloop_flatten.hard_flatten``), encode
    the corrected image back, and inject the x0-space delta into the running
    latents: ``latents += (x0_corrected - x0)`` (EXACTLY the prototype's Euler
    injection - the x0-space delta maps 1:1 into ``prev_sample``; no scheduler
    scaling). When no confident flat region is found the step is a complete no-op.

    The VAE is staged to GPU only for this decode/encode and returned to CPU
    afterwards, matching the loop's existing VRAM discipline (VAE on CPU during
    U-Net work). Returns ``(latents, applied)``; on any failure the latents are
    returned unchanged so a bad step can never corrupt the run.
    """
    if pred_original_sample is None:
        return latents, False
    from core.vram_optimization import move_vae_to_gpu, move_vae_to_cpu
    from core.inference.inloop_flatten import hard_flatten
    scaling = pipeline.vae.config.scaling_factor
    try:
        move_vae_to_gpu(pipeline)
        lat = (pred_original_sample / scaling + vae_shift).to(dtype=pipeline.vae.dtype)
        with torch.no_grad():
            img = pipeline.vae.decode(lat, return_dict=True).sample
        img01 = (img.float() / 2 + 0.5).clamp(0, 1)
        arr = img01[0].permute(1, 2, 0).cpu().numpy()
        out_arr, applied = hard_flatten(arr, min_region_frac=min_region_frac)
        if not applied:
            return latents, False
        t = torch.from_numpy(out_arr).permute(2, 0, 1).unsqueeze(0).to(
            device=pipeline.vae.device, dtype=pipeline.vae.dtype)
        x = t * 2.0 - 1.0
        with torch.no_grad():
            enc = pipeline.vae.encode(x).latent_dist.mode()
        x0c = ((enc - vae_shift) * scaling)
        delta = (x0c - pred_original_sample.to(x0c.device, x0c.dtype))
        latents = latents + delta.to(latents.device, latents.dtype)
        return latents, True
    except Exception as e:
        print(f"[InLoopFlatten] step skipped (decode/encode failed): {e}")
        return latents, False
    finally:
        move_vae_to_cpu(pipeline)


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


def _resolve_sdxl_original_size(default_h: int, default_w: int,
                               original_size_w: int = 0, original_size_h: int = 0,
                               original_size_scale: float = 1.0):
    """Resolve the SDXL original_size (h, w) for time_ids at inference.

    Explicit width+height override take precedence; otherwise the default (output
    size, or the input image size for img2img/inpaint) is scaled by original_size_scale.
    crop_coords_top_left stays (0,0); target_size remains the output size.
    """
    if original_size_w and original_size_h and original_size_w > 0 and original_size_h > 0:
        return int(original_size_h), int(original_size_w)
    s = float(original_size_scale) if original_size_scale else 1.0
    return int(round(default_h * s)), int(round(default_w * s))


def _prepare_negpip_weights(negpip_weights, nag_active):
    """Prepare NegPip per-context signed weight rows from the pipeline-supplied dict.

    negpip_weights: {"pos","neg"[, "nag_neg"]} 1-D signed weight vectors, or None.
    Returns (negpip_active, nag_token_weights, negpip_token_weights) where:
      - nag_token_weights: [3, seq] rows [cfg_neg, cfg_pos, nag_neg] when NAG is active,
        folded into the NAG processor; else None.
      - negpip_token_weights: [2, seq] rows [neg, pos] for the standalone NegPip
        processor when NAG is not active; else None.
    The batch order matches the U-Net's [negative, positive(, nag_negative)] context.
    """
    if negpip_weights is None:
        return False, None, None
    pos_w = negpip_weights.get("pos")
    neg_w = negpip_weights.get("neg")
    if neg_w is None and pos_w is not None:
        neg_w = torch.ones_like(pos_w)
    if pos_w is None and neg_w is not None:
        pos_w = torch.ones_like(neg_w)
    if pos_w is None and neg_w is None:
        return False, None, None

    def _pad_stack(rows):
        # Rows can differ in length: the main prompt may be chunked (e.g. 231) while
        # the NAG negative is a single chunk (77). The sampling loop pads the embeds to
        # the longest sequence at the END, so pad the weight rows the same way (1.0 =
        # identity) before stacking.
        max_len = max(r.shape[-1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[-1] < max_len:
                pad = torch.ones(max_len - r.shape[-1], device=r.device, dtype=r.dtype)
                r = torch.cat([r, pad], dim=-1)
            padded.append(r)
        return torch.stack(padded, dim=0)

    if nag_active:
        nag_neg_w = negpip_weights.get("nag_neg")
        if nag_neg_w is None:
            nag_neg_w = neg_w
        return True, _pad_stack([neg_w, pos_w, nag_neg_w]), None
    return True, None, _pad_stack([neg_w, pos_w])


# =============================================================================
# OUTPAINT B1: trajectory-consistent x0-space projection injection + boundary
# color proximal (custom_inpaint_sampling_loop, SD/SDXL only). See
# scratchpad/outpaint_continuity_design.md section "B1". Gated entirely on
# the outpaint_noise_init kwarg -- normal inpaint is untouched.
# =============================================================================

# Scheduler classes whose running `sample` (this loop's `latents`) is the
# plain k-diffusion/EDM sigma-scale representation x_sigma = x0 + sigma*eps
# (their OWN `add_noise()` has no alpha term at all, e.g.
# EulerDiscreteScheduler.add_noise: `noisy_samples = original_samples +
# noise * sigma`). Verified by reading each class's `add_noise`/`step` in
# diffusers 0.38.0 (scheduling_euler_discrete.py, scheduling_euler_ancestral_
# discrete.py, scheduling_k_dpm_2_discrete.py, scheduling_k_dpm_2_ancestral_
# discrete.py, scheduling_heun_discrete.py, scheduling_lms_discrete.py) --
# every one of them computes `pred_original_sample` (with s_churn=0, the
# default this loop uses, so sigma_hat == sigma) as:
#   epsilon:      pred_original_sample = sample - sigma * model_output
#   v_prediction: pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5)
#                                         + (sample / (sigma**2 + 1))
# All OTHER schedulers in SAMPLER_MAP keep `sample` in the VP scale
# x_t = alpha_t*x0 + sigma_t*eps. DDIM/DDPM/PNDM read
# `self.alphas_cumprod[timestep]` directly (integer timesteps) -- exact.
# The DPM-solver family + UniPC instead derive (alpha_t, sigma_t) from
# `self.sigmas[i]` via `_sigma_to_alpha_sigma_t` (alpha_t=1/sqrt(sigma**2+1)).
# With the DEFAULT sigma schedule that equals alphas_cumprod[timestep], but
# under KARRAS sigmas (schedule_type="karras", user-selectable) it does NOT --
# so for those classes we MUST read the actual `self.sigmas[i]`, not
# alphas_cumprod[timestep] (~6% divergence otherwise).
_OUTPAINT_SIGMA_SCALE_SCHEDULERS = (
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
    "KDPM2DiscreteScheduler",
    "KDPM2AncestralDiscreteScheduler",
    "HeunDiscreteScheduler",
    "LMSDiscreteScheduler",
)

# VP-scale schedulers whose (alpha_t, sigma_t) come from `self.sigmas[i]` (via
# _sigma_to_alpha_sigma_t) -- must use the actual sigma (Karras-safe), NOT
# alphas_cumprod[timestep]. DDIM/DDPM/PNDM are NOT here (they read alphas_cumprod).
_OUTPAINT_VP_SIGMA_SCHEDULERS = (
    "DPMSolverMultistepScheduler",
    "DPMSolverSinglestepScheduler",
    "UniPCMultistepScheduler",
)


def _outpaint_x0_transform(scheduler, sample: torch.Tensor, t, i: int):
    """Return (predict_x0, to_model_output) closures for the OUTPAINT B1
    x0-space projection.

    ``predict_x0(model_output) -> x0_hat`` and ``to_model_output(x0_new) ->
    model_output'`` are exact inverses of each other and match the SAME
    convention diffusers' own ``scheduler.step()``/``convert_model_output()``
    use for ``scheduler``'s class at this exact step index ``i`` / timestep
    ``t`` -- see ``_OUTPAINT_SIGMA_SCALE_SCHEDULERS`` above for the two
    families and their formulas. This lets the projection run BEFORE
    ``scheduler.step()`` (instead of only previewing pred_original_sample
    after it, which is too late to influence the step).
    """
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    cls_name = type(scheduler).__name__

    if cls_name in _OUTPAINT_SIGMA_SCALE_SCHEDULERS and hasattr(scheduler, "sigmas"):
        # KDPM2(-Ancestral) are 2nd-order methods that visit each nominal step
        # TWICE (a "first order" pass, then an interpolated 2nd-order
        # correction pass, toggled via the scheduler's own
        # `state_in_first_order` property/`self.sample` cache -- see
        # scheduling_k_dpm_2_discrete.py / scheduling_k_dpm_2_ancestral_
        # discrete.py `step()`). On the 2nd-order pass they compute
        # pred_original_sample from `sigmas_interpol` (a log-interpolated
        # intermediate sigma), NOT plain `self.sigmas[step_index]` -- the two
        # classes even index `sigmas_interpol` slightly differently
        # (`[step_index]` vs `[step_index - 1]`). Reading
        # `scheduler.state_in_first_order` here (BEFORE `scheduler.step()`
        # runs for this same iteration) is safe: it/​`self.sample` are only
        # mutated INSIDE `step()`, so it still reflects the state `step()` is
        # about to use. Heun is also 2nd-order but its own 2nd-order sigma
        # (`sigma_next`) reduces to plain `self.sigmas[step_index]` at this
        # point, so it needs no special case (verified empirically).
        if cls_name == "KDPM2DiscreteScheduler" and not scheduler.state_in_first_order:
            sigma = scheduler.sigmas_interpol[i]
        elif cls_name == "KDPM2AncestralDiscreteScheduler" and not scheduler.state_in_first_order:
            sigma = scheduler.sigmas_interpol[i - 1]
        else:
            sigma = scheduler.sigmas[i]
        sigma = sigma.to(device=sample.device, dtype=sample.dtype)

        if prediction_type == "v_prediction":
            def predict_x0(model_output):
                return model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + (sample / (sigma ** 2 + 1))

            def to_model_output(x0_new):
                return (sample / (sigma ** 2 + 1) - x0_new) * ((sigma ** 2 + 1) ** 0.5) / sigma
        elif prediction_type in ("sample", "original_sample"):
            def predict_x0(model_output):
                return model_output

            def to_model_output(x0_new):
                return x0_new
        else:  # epsilon (default)
            def predict_x0(model_output):
                return sample - sigma * model_output

            def to_model_output(x0_new):
                return (sample - x0_new) / sigma
    elif cls_name in _OUTPAINT_VP_SIGMA_SCHEDULERS and hasattr(scheduler, "sigmas") and i < len(scheduler.sigmas):
        # DPM++/UniPC: derive (alpha_t, sigma_t) from the ACTUAL sigma the
        # scheduler uses at step i (Karras-safe), matching diffusers'
        # _sigma_to_alpha_sigma_t. alphas_cumprod[t] would diverge ~6% under
        # Karras sigmas. These plug into the SAME VP predict_x0/to_model_output
        # formulas below (sqrt_alpha == alpha_t, sqrt_beta == sigma_t).
        sigma = scheduler.sigmas[i].to(device=sample.device, dtype=sample.dtype)
        sqrt_alpha = 1.0 / (sigma ** 2 + 1) ** 0.5   # alpha_t
        sqrt_beta = sigma * sqrt_alpha                # sigma_t = sigma * alpha_t
    else:
        # DDIM/DDPM/PNDM: integer timesteps, exact via alphas_cumprod[t].
        t_index = int(t.item()) if torch.is_tensor(t) else int(t)
        alpha_prod_t = scheduler.alphas_cumprod[t_index].to(device=sample.device, dtype=sample.dtype)
        beta_prod_t = 1 - alpha_prod_t
        sqrt_alpha = alpha_prod_t ** 0.5
        sqrt_beta = beta_prod_t ** 0.5

        if prediction_type == "v_prediction":
            def predict_x0(model_output):
                return sqrt_alpha * sample - sqrt_beta * model_output

            def to_model_output(x0_new):
                return (sqrt_alpha * sample - x0_new) / sqrt_beta
        elif prediction_type in ("sample", "original_sample"):
            def predict_x0(model_output):
                return model_output

            def to_model_output(x0_new):
                return x0_new
        else:  # epsilon (default)
            def predict_x0(model_output):
                return (sample - sqrt_beta * model_output) / sqrt_alpha

            def to_model_output(x0_new):
                return (sample - sqrt_alpha * x0_new) / sqrt_beta

    return predict_x0, to_model_output


def _outpaint_collar_weight(mask_latent: torch.Tensor, collar_cells: float = 6.0) -> Optional[torch.Tensor]:
    """Static per-cell weight W_b(d) in [0,1] for the B1 low-frequency
    boundary color proximal: 1 at generate-side cells touching the rect
    boundary, decaying LINEARLY to 0 by ``collar_cells`` latent cells into
    the generate region, and exactly 0 in the keep region (mask_latent==0)
    and far from the boundary. ``d`` = per-cell Euclidean distance (in latent
    cells) from each generate cell to the nearest keep cell, via scipy's
    distance transform (scipy is already a hard dependency of this inference
    stack -- see core/inference/inloop_flatten.py). Computed ONCE before the
    denoise loop since mask_latent never changes across steps.

    Returns None when there is no generate region at all (degenerate; no-op).
    """
    from scipy import ndimage
    mask_np = mask_latent.detach().to(dtype=torch.float32, device="cpu").numpy()
    gen_np = (mask_np[0, 0] > 0.5).astype(np.float64)
    if gen_np.max() == 0:
        return None
    dist = ndimage.distance_transform_edt(gen_np)
    w = np.clip(1.0 - dist / max(1e-6, collar_cells), 0.0, 1.0) * gen_np
    w_t = torch.from_numpy(w).to(device=mask_latent.device, dtype=mask_latent.dtype)
    return w_t.view(1, 1, *w_t.shape).expand(mask_latent.shape[0], mask_latent.shape[1], -1, -1).contiguous()


def _outpaint_gaussian_lowpass(x: torch.Tensor, kappa: float = 3.0) -> torch.Tensor:
    """Depthwise Gaussian low-pass (G_kappa) in latent space, replicate-padded
    to avoid edge artifacts at the canvas boundary. ``kappa`` = blur sigma in
    latent cells ("a few latent px" per the design doc). Used by the B1
    low-frequency boundary color proximal.
    """
    import torch.nn.functional as F
    radius = max(1, int(round(kappa * 3)))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32, device=x.device) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2 * kappa ** 2))
    kernel_1d = (kernel_1d / kernel_1d.sum()).to(dtype=x.dtype)
    channels = x.shape[1]
    kernel_h = kernel_1d.view(1, 1, 1, size).expand(channels, 1, 1, size).contiguous()
    kernel_v = kernel_1d.view(1, 1, size, 1).expand(channels, 1, size, 1).contiguous()
    padded = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    blurred = F.conv2d(padded, kernel_h, groups=channels)
    padded = F.pad(blurred, (0, 0, radius, radius), mode="replicate")
    blurred = F.conv2d(padded, kernel_v, groups=channels)
    return blurred


def _outpaint_apply_boundary_color(
    x0: torch.Tensor,
    target_lowfreq: torch.Tensor,
    collar_weight: torch.Tensor,
    strength: float,
    kappa: float = 3.0,
) -> torch.Tensor:
    """B1 low-frequency boundary color proximal (design doc section 2):
    ``x0 += strength * W_b(d) * (E(G_kappa(z0_keep)) - G_kappa(x0))``,
    restricted to the generate-side collar near the rect boundary via
    ``collar_weight`` (zero elsewhere, so this is an exact no-op outside the
    collar). ``target_lowfreq`` is the precomputed ``G_kappa(image_latents)``
    (the simple/robust "low-pass the whole canvas latents, read the collar
    value" approximation the design doc endorses -- with the default
    outpaint_fill_mode="replicate" the fill IS an edge-extension, so this
    already approximates "extend the keep-side low-freq value outward").
    """
    current_lowfreq = _outpaint_gaussian_lowpass(x0, kappa=kappa)
    return x0 + strength * collar_weight * (target_lowfreq - current_lowfreq)


# ============================================================================
# OUTPAINT B2 -- RePaint-style band-limited time-travel resampling
# (scratchpad/outpaint_continuity_design.md section "B2"). Builds on B1 above:
# every re-denoise pass triggered by a jump still runs through the B1
# x0-projection block in the main loop. Gated on `outpaint_noise_init` (same
# flag as B1) AND `outpaint_resample_count > 1`; normal inpaint and
# outpaint_resample_count<=1 are byte-identical (the schedule below degenerates
# to the plain 0..T-1 walk -- see `_build_outpaint_resample_schedule`'s r<=1
# early return).
# ============================================================================

# Compatible schedulers hold no cross-step solver history/state that a
# backward jump would invalidate (unlike DPM++/UniPC's multistep buffers or
# KDPM2/Heun's 2nd-order interpolated-sigma bookkeeping) -- see the design
# doc's "SCHEDULER GATING". Two families, exactly like `_outpaint_x0_transform`
# above, but narrower: only the schedulers with NO history are eligible at all
# for B2 (DPM++/UniPC/PNDM/LMS/KDPM2/Heun are excluded outright, not merely
# re-derived differently).
_OUTPAINT_RESAMPLE_SIGMA_SCHEDULERS = (
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
)
_OUTPAINT_RESAMPLE_VP_SCHEDULERS = (
    "DDIMScheduler",
    "DDPMScheduler",
)

# Band-limited: skip jumps in the first `_OUTPAINT_RESAMPLE_BAND_LO` and last
# `1 - _OUTPAINT_RESAMPLE_BAND_HI` fraction of the schedule (design doc
# defaults). Not user-exposed params yet (B2 checklist item 5).
_OUTPAINT_RESAMPLE_BAND_LO = 0.15
_OUTPAINT_RESAMPLE_BAND_HI = 0.70


def _build_outpaint_resample_schedule(
    num_timesteps: int, r: int, u: int, band_lo: float, band_hi: float
) -> List[Tuple[int, bool]]:
    """Build the ordered VISIT schedule for OUTPAINT B2 time-travel resampling.

    Returns a list of ``(step_index, is_forward_jump)`` tuples. ``step_index``
    is the LOGICAL diffusion index (0..num_timesteps-1) the denoise-step body
    must run at for this visit -- callers index ``timesteps[step_index]`` /
    ``scheduler.sigmas[step_index]`` / any other per-step schedule (cfg
    schedule, prompt-edit callback, spectrum, controlnet fraction, ...) with
    THIS value, NEVER a running visit counter, so a revisit reproduces exactly
    the same conditioning as the original forward visit to that index.

    ``is_forward_jump`` is True exactly on the first visit of a re-denoise
    cycle (the visit immediately following a backward jump): before running
    the denoise-step body for that visit, the caller must re-noise the WHOLE
    latent from the level it is actually at (the level reached after
    finishing the immediately PRECEDING visit in this list, which is always
    ``step_index + u`` -- see below) up to the level at ``step_index``. It is
    False on every other visit (including ``step_index == 0``), where the
    latent is already at the correct level from the immediately preceding
    visit and no re-noise is needed.

    Degenerate case: ``r <= 1`` or ``u <= 0`` or ``num_timesteps <= 0``
    disables resampling entirely -- the returned schedule is the plain
    ``[(0, False), (1, False), ..., (num_timesteps - 1, False)]`` walk,
    ITERATION-ORDER-IDENTICAL to ``enumerate(timesteps)``.

    Anchors (landing positions, i.e. the index reached right after a normal
    forward step): ``j = band_start, band_start + u, band_start + 2u, ...``
    where ``band_start = ceil(band_lo * num_timesteps)``, while ``j <=
    band_end`` (``band_end = floor(band_hi * num_timesteps)``) and ``j - u >=
    0`` (a full u-step segment must exist behind the anchor to jump into).
    At each anchor, once the plain forward walk reaches ``step_index == j -
    1`` (the last step of the ``[j-u, j-1]`` segment, landing the latent at
    level ``j``), ``r - 1`` EXTRA full passes through that segment are
    inserted, each starting with an ``is_forward_jump=True`` visit at
    ``step_index = j - u``. The walk then continues forward from ``j``
    unmodified -- exactly ``r`` total traversals of the segment (1 original +
    r-1 resampled), matching the design doc's "r cycles".

    NFE (total visit count) == ``num_timesteps + (r - 1) * u * len(anchors)``.
    """
    if r <= 1 or u <= 0 or num_timesteps <= 0:
        return [(i, False) for i in range(num_timesteps)]

    band_start = int(math.ceil(band_lo * num_timesteps))
    band_end = int(math.floor(band_hi * num_timesteps))

    anchors = []
    j = band_start
    while j <= band_end and (j - u) >= 0 and j <= num_timesteps:
        anchors.append(j)
        j += u
    anchors_set = set(anchors)

    schedule: List[Tuple[int, bool]] = []
    for i in range(num_timesteps):
        schedule.append((i, False))
        landing = i + 1
        if landing in anchors_set:
            for _cycle in range(r - 1):
                for k in range(u):
                    schedule.append((landing - u + k, k == 0))
    return schedule


def _outpaint_resample_jump(
    scheduler,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    hi_index: int,
    lo_index: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """OUTPAINT B2 time-travel jump: re-noise the WHOLE latent (keep AND
    generate regions TOGETHER -- no mask special-casing here, unlike B1) from
    its CURRENT level (index ``lo_index``, the level the running ``latents``
    are actually at, having just landed there after the previous visit) UP to
    the higher-noise level at index ``hi_index`` (``hi_index < lo_index``: a
    SMALLER diffusion index is noisier). Because the whole composite is
    re-noised together, the keep region becomes a CORRELATED
    ``q(z_hi | z0)`` sample whose noise realization matches the generate
    region's -- RePaint's key property (design doc "B2"). Fresh noise every
    call, drawn from ``generator`` (the sampler's ``step_generator``, for
    run-to-run determinism given a fixed seed).

    ``timesteps`` is the CALLER's local (t_start-sliced) timesteps array --
    used instead of ``scheduler.timesteps`` directly so this helper never has
    to assume ``t_start == 0`` itself (the caller's B1 guard already enforces
    that for the outpaint path, but keeping the dependency explicit avoids a
    second, easy-to-miss implicit assumption inside this helper).

    Only called when the scheduler class is one of
    ``_OUTPAINT_RESAMPLE_SIGMA_SCHEDULERS`` / ``_OUTPAINT_RESAMPLE_VP_SCHEDULERS``
    (see the gating in ``custom_inpaint_sampling_loop``) -- the two branches
    below mirror ``_outpaint_x0_transform``'s per-family sigma/alpha
    derivation, restricted to the two families with no cross-step solver
    state.
    """
    cls_name = type(scheduler).__name__
    noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)

    if cls_name in _OUTPAINT_RESAMPLE_SIGMA_SCHEDULERS:
        sigma_hi = scheduler.sigmas[hi_index].to(device=latents.device, dtype=latents.dtype)
        sigma_lo = scheduler.sigmas[lo_index].to(device=latents.device, dtype=latents.dtype)
        delta_sq = (sigma_hi ** 2 - sigma_lo ** 2).clamp_min(0.0)
        return latents + delta_sq.sqrt() * noise
    else:  # DDIM / DDPM: VP / alphas_cumprod convention
        t_hi = int(timesteps[hi_index].item())
        t_lo = int(timesteps[lo_index].item())
        abar_hi = scheduler.alphas_cumprod[t_hi].to(device=latents.device, dtype=latents.dtype)
        abar_lo = scheduler.alphas_cumprod[t_lo].to(device=latents.device, dtype=latents.dtype)
        ratio = (abar_hi / abar_lo).clamp(max=1.0)
        return ratio.sqrt() * latents + (1.0 - ratio).clamp_min(0.0).sqrt() * noise


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
    is_deus: bool = False,  # DEUS model flag - uses 2-Pass CFG instead of batch concatenation
    ref_guide_configs: Optional[List[Dict]] = None,  # Reference Guide configs for latent blending
    vision_encoder=None,  # SigLIP2 VisionEncoderWrapper for VRAM status logging
    original_size_w: int = 0,  # SDXL micro-cond override: explicit original width (0 = auto)
    original_size_h: int = 0,  # SDXL micro-cond override: explicit original height (0 = auto)
    original_size_scale: float = 1.0,  # SDXL micro-cond: original_size = output size * scale (when not explicit)
    negpip_weights: Optional[Dict[str, torch.Tensor]] = None,  # NegPip signed per-token weights {"pos","neg","nag_neg"}; auto-set when prompt has negative weights
    loop_decode: str = "full",  # Loop-generation decode mode: "full" (decode as usual) | "cheap"
                                # (if a PidVaeWrapper is active, use its embedded real VAE instead of
                                # the PiD student net; no-op otherwise) | "none" (skip decode entirely,
                                # return the pre-unscale latent for the caller to cache -- see the
                                # Stage-3 VAE DECODE section below).
    spectrum_enable: bool = False,  # Spectrum (Adaptive Spectral Feature Forecasting) acceleration
    spectrum_w: float = 0.5,  # Spectral/linear mix (1.0 = spectral only; lower = more linear/stable)
    spectrum_w_decay: float = 0.0,  # OPT-IN per-step decay exponent for spectrum_w (0 = off, default)
    spectrum_delta_cap: float = 0.0,  # OPT-IN trajectory speed limiter multiplier K (0 = off, default)
    spectrum_m: int = 4,  # Number of Chebyshev basis
    spectrum_lam: float = 0.1,  # Ridge regularization
    spectrum_warmup_steps: int = 3,  # Leading full-eval steps
    spectrum_window_size: int = 4,  # Initial skip interval
    spectrum_flex_window: float = 0.75,  # Skip damping (0 = max skip)
    spectrum_tail: float = 0.12,  # Fraction of final steps forced to actual passes (detail)
    spectrum_feature_mode: str = "output",  # "output" (black-box) or "block" (deep-feature, paper-faithful)
    spectrum_cache_branch: int = 1,  # block mode: down_blocks[cache_branch:] + mid are forecast
    spectrum_max_cache: int = 0,  # forecaster sliding-window size (0 = unlimited; block mode defaults to 6)
    fbcache_enable: bool = False,  # FBCache (First Block Cache) dynamic U-Net block caching
    fbcache_threshold: float = 0.12,  # relative-L1 indicator threshold (higher = more skips/faster)
    fbcache_warmup_steps: int = 1,  # always compute the first N steps
    fbcache_cache_branch: int = 1,  # indicator = down[branch]; reused region = down[branch+1:]+mid
    color_flatten_strength: int = 0,  # 0-100 post-decode chroma smoothing; 0 = off
    flatten_in_loop: bool = False,  # in-loop hard-flatten of the flat background (SD1.5/SDXL)
    flatten_in_loop_last_steps: int = 3,  # inject on the last N ACTUAL denoise steps
    flatten_in_loop_min_region: float = 0.02,  # flat-region area gate (fraction of frame)
    style_cfg=None,  # core.inference.reference_style.StyleTransferConfig, or None (default off)
    style_ref_x0: Optional[torch.Tensor] = None,  # VAE-encoded style reference latent (build_style_transfer)
    style_eps_ref: Optional[torch.Tensor] = None,  # fixed reference noise (build_style_transfer)
    style_refs: Optional[List[Tuple[Any, torch.Tensor, torch.Tensor]]] = None,  # multi-reference (N>1): list of (StyleTransferConfig, ref_x0, eps_ref) triples, one per reference image; only consulted when len>1 (build_style_transfer_multi)
    style_combine_mode: str = "stack",  # "stack" | "common_concept" -- multi-reference combine mode (core.inference.reference_style.inject_kv_multi)
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
    # CRITICAL FIX: Use U-Net's device instead of pipeline.device
    # pipeline.device returns cpu after text encoders are offloaded
    if hasattr(pipeline, 'unet'):
        # Get device from first parameter (nn.Module doesn't have .device attribute)
        device = next(pipeline.unet.parameters()).device
    else:
        device = pipeline.device

    # Get U-Net dtype, but use float16 for latents if U-Net is FP8 or UINT quantized
    # (torch.randn doesn't support FP8, and UINT quantization uses FP16 activations)
    # nn.Module doesn't have .dtype, get from first parameter
    unet_dtype = next(pipeline.unet.parameters()).dtype
    is_uint_quantized = hasattr(pipeline.unet, '_is_uint_quantized') and pipeline.unet._is_uint_quantized

    if unet_dtype == torch.float8_e4m3fn or unet_dtype == torch.float8_e5m2 or is_uint_quantized:
        dtype = torch.float16  # Use float16 for latents
        if is_uint_quantized:
            print(f"[CustomSampling] U-Net is UINT quantized, using float16 for latents and activations")
        else:
            print(f"[CustomSampling] U-Net is {unet_dtype}, using float16 for latents")
    else:
        dtype = unet_dtype

    # Check if SDXL by checking if text_encoder_2 exists (more reliable than isinstance for ControlNet pipelines)
    is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

    # DEUS uses 2-Pass CFG (separate negative/positive passes) instead of batch concatenation
    # This is required because DEUS has variable sequence length embeddings
    if is_deus:
        print(f"[CustomSampling] DEUS mode: Using 2-Pass CFG (separate negative/positive passes)")

    print(f"[CustomSampling] Pipeline type: {type(pipeline).__name__}, is_sdxl: {is_sdxl}, is_deus: {is_deus}")

    # Use ancestral_generator for stochastic samplers (always provided by pipeline)
    step_generator = ancestral_generator
    if ancestral_generator is not None:
        print(f"[CustomSampling] Using ancestral generator for stochastic sampler")

    # Get components
    unet = pipeline.unet
    scheduler = pipeline.scheduler

    # Training-free reference-style transfer (StyleAligned/VSP-style KV injection).
    # No style config => style_active is False and nothing below this ever runs
    # (byte-identical to the pre-style-transfer code path).
    style_active = style_cfg is not None and style_ref_x0 is not None and style_eps_ref is not None
    if style_active:
        from core.inference.attention_processors import ensure_style_block_indices
        num_style_blocks = ensure_style_block_indices(unet)
        style_cfg.resolve_default_block_range(num_style_blocks)
        print(f"[CustomSampling] Style transfer active: {num_style_blocks} self-attention layers "
              f"eligible, block_range={style_cfg.block_range} (None = all)")

    # Multi-reference (N>1) style transfer: style_refs is populated (and style_cfg/
    # style_ref_x0/style_eps_ref left None) ONLY when the caller collected 2+
    # references -- a single reference always resolves through style_active above,
    # so this is mutually exclusive with it and never affects single-ref behavior.
    style_refs_active = style_refs is not None and len(style_refs) > 1
    if style_refs_active:
        from core.inference.attention_processors import ensure_style_block_indices
        num_style_blocks = ensure_style_block_indices(unet)
        for _style_cfg_i, _style_x0_i, _style_eps_i in style_refs:
            _style_cfg_i.resolve_default_block_range(num_style_blocks)
        print(f"[CustomSampling] Multi-ref style transfer active: {len(style_refs)} references, "
              f"{num_style_blocks} self-attention layers eligible")

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

    # NegPip: auto-activated by the pipeline when the prompt(s) contain negative
    # emphasis weights. Build the per-context signed weight rows aligned with the
    # batch order the U-Net receives: [negative, positive] for CFG, plus nag_negative
    # for NAG. When NAG is active the weights are folded into the NAG processor
    # (single forward); otherwise a dedicated NegPip processor is installed.
    negpip_active, nag_token_weights, negpip_token_weights = _prepare_negpip_weights(negpip_weights, nag_active)
    if negpip_active:
        if nag_active:
            print(f"[CustomSampling] NegPip + NAG: signed V weighting folded into NAG processor (seq={nag_token_weights.shape[-1]})")
        else:
            print(f"[CustomSampling] NegPip active: signed V weighting on cross-attention (seq={negpip_token_weights.shape[-1]})")

    if nag_active:
        from core.inference.nag_processor import set_nag_processors
        print(f"[CustomSampling] NAG enabled: scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}, sigma_end={nag_sigma_end}, attention={attention_type}")

        # Set NAG processors on cross-attention layers (with optional NegPip weights)
        original_processors = set_nag_processors(
            unet,
            nag_scale=nag_scale,
            nag_tau=nag_tau,
            nag_alpha=nag_alpha,
            attention_type=attention_type,
            token_weights=nag_token_weights,
        )

        # Ensure NAG embeddings on correct device/dtype
        nag_negative_prompt_embeds = nag_negative_prompt_embeds.to(device=device, dtype=dtype)
        if is_sdxl and nag_negative_pooled_prompt_embeds is not None:
            nag_negative_pooled_prompt_embeds = nag_negative_pooled_prompt_embeds.to(device=device, dtype=dtype)
    elif negpip_active:
        from core.inference.negpip_processor import set_negpip_processors
        original_processors = set_negpip_processors(unet, negpip_token_weights, attention_type=attention_type)

    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Spectrum (Adaptive Spectral Feature Forecasting): skip U-Net forwards on selected
    # steps by forecasting the raw output from a Chebyshev fit over actual passes.
    # Auto-disabled when the per-step conditioning is not stable (prompt editing,
    # ControlNet), for DEUS (2-pass output), or when there are too few steps to warm up.
    spectrum = None
    spectrum_block_ctrl = None
    if spectrum_enable:
        _n_steps = len(timesteps)
        _spectrum_blocked = (
            is_deus or has_controlnet or (prompt_embeds_callback is not None)
        )
        if _spectrum_blocked:
            print("[Spectrum] requested but disabled (prompt-editing / ControlNet / DEUS "
                  "change the output per step; needs stable conditioning)")
            _add_generation_warning(
                "Spectrum was requested but disabled: prompt-editing / ControlNet / DEUS "
                "change the output per step and need stable conditioning",
                code="feature_auto_disabled",
            )
        elif _n_steps < spectrum_warmup_steps + 3:
            print(f"[Spectrum] requested but disabled ({_n_steps} steps < warmup+3; "
                  f"little benefit at low step counts)")
            _add_generation_warning(
                f"Spectrum was requested but disabled: {_n_steps} steps is below "
                f"warmup+3 (little benefit at low step counts)",
                code="feature_auto_disabled",
            )
        else:
            from core.inference.spectrum_forecaster import SpectrumForecaster
            _block = spectrum_feature_mode == "block"
            # A small local window localizes the Chebyshev fit (with per-window tau
            # renormalization in the forecaster) so extrapolation past the last anchor
            # stays well-conditioned -- the key fix for output-mode graininess.
            _max_cache = spectrum_max_cache if spectrum_max_cache > 0 else (6 if _block else 5)
            spectrum = SpectrumForecaster(
                _n_steps, num_basis=spectrum_m, lam=spectrum_lam, w=spectrum_w,
                w_decay=spectrum_w_decay,
                delta_cap=spectrum_delta_cap,
                warmup_steps=spectrum_warmup_steps, window_size=spectrum_window_size,
                flex_window=spectrum_flex_window, tail_fraction=spectrum_tail,
                max_cache=_max_cache,
            )
            _n_anchor = len(spectrum.anchors)
            if _block:
                # Paper-faithful: forecast the smooth DEEP features (down_blocks[branch:]
                # + mid) and recompute the shallow/up path every step. SDXL/SD U-Net only.
                from core.inference.spectrum_unet import SpectrumBlockController
                spectrum_block_ctrl = SpectrumBlockController(unet, spectrum, cache_branch=spectrum_cache_branch)
                print(f"[Spectrum] enabled (block mode): {_n_anchor}/{_n_steps} deep-feature "
                      f"passes, cache_branch={spectrum_block_ctrl.branch}/{spectrum_block_ctrl.n_down}, "
                      f"m={spectrum_m}, lam={spectrum_lam}, w={spectrum_w}, max_cache={_max_cache}, "
                      f"warmup={spectrum_warmup_steps}, window={spectrum_window_size}, "
                      f"flex={spectrum_flex_window}, tail={spectrum_tail}")
            else:
                print(f"[Spectrum] enabled (output mode): {_n_anchor}/{_n_steps} actual passes "
                      f"(m={spectrum_m}, lam={spectrum_lam}, w={spectrum_w}, "
                      f"warmup={spectrum_warmup_steps}, window={spectrum_window_size}, "
                      f"flex={spectrum_flex_window}, tail={spectrum_tail})")

    # Style transfer runs its own separate capture/cond/uncond forwards, which are
    # incompatible with NAG's batch-3 layout, ControlNet's batch-2 residuals, and
    # Spectrum's per-step record/forecast (style never calls spectrum.record()).
    # Yield to the established feature (disable style) with a warning rather than
    # crash / silently misbehave.
    # TODO: ControlNet(structure)+style(appearance) is a desirable combo; supporting
    # it needs per-pass batch-1 residual recompute -- future enhancement.
    if style_active and (nag_active or has_controlnet or spectrum is not None):
        print("[CustomSampling] Style transfer disabled: not compatible with NAG / "
              "ControlNet / Spectrum in this version")
        _add_generation_warning(
            "Style transfer disabled: not compatible with NAG / ControlNet / Spectrum in this version.",
            code="style_incompatible",
        )
        style_active = False

    # Multi-reference style transfer has the exact same batch-layout incompatibility
    # (separate per-ref capture forwards + a 2-Pass CFG cond/uncond split).
    if style_refs_active and (nag_active or has_controlnet or spectrum is not None):
        print("[CustomSampling] Multi-ref style transfer disabled: not compatible with NAG / "
              "ControlNet / Spectrum in this version")
        _add_generation_warning(
            "Style transfer disabled: not compatible with NAG / ControlNet / Spectrum in this version.",
            code="style_incompatible",
        )
        style_refs_active = False

    # FBCache (First Block Cache): dynamic per-step deep-block caching via the same
    # per-block interception as Spectrum block mode. Mutually exclusive with Spectrum
    # (same monkey-patch), and auto-disabled for the same unstable-conditioning cases
    # (prompt editing / ControlNet / DEUS) that make per-step block outputs non-reusable.
    # Also disabled when style transfer is active: the style branch's per-step capture
    # forward (on the style ref latent) would run through the FBCache block wrappers
    # and pollute the cache with the style ref's residuals (begin_step/end_step reset
    # is only invoked on the standard path, never in the style branch).
    fbcache_ctrl = None
    if fbcache_enable:
        if spectrum_block_ctrl is not None or spectrum is not None:
            print("[FBCache] requested but disabled (Spectrum is active; they share the "
                  "same block interception and are mutually exclusive)")
            _add_generation_warning(
                "FBCache was requested but disabled: Spectrum is active and they share the "
                "same block interception (mutually exclusive)",
                code="feature_auto_disabled",
            )
        elif is_deus or has_controlnet or (prompt_embeds_callback is not None) or style_active or style_refs_active:
            print("[FBCache] requested but disabled (prompt-editing / ControlNet / DEUS / "
                  "style transfer change the block outputs per step; needs stable conditioning)")
            _add_generation_warning(
                "FBCache was requested but disabled: prompt-editing / ControlNet / DEUS / "
                "style transfer change the block outputs per step and need stable conditioning",
                code="feature_auto_disabled",
            )
        else:
            from core.inference.fbcache_unet import build_unet_fbcache_controller
            fbcache_ctrl = build_unet_fbcache_controller(
                unet,
                {
                    "fbcache_enable": fbcache_enable,
                    "fbcache_threshold": fbcache_threshold,
                    "fbcache_warmup_steps": fbcache_warmup_steps,
                    "fbcache_cache_branch": fbcache_cache_branch,
                },
                label="txt2img",
            )

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

    # Prepare Reference Guide latents (VAE encode reference images)
    ref_guides = []
    if ref_guide_configs:
        from core.vram_optimization import move_vae_to_gpu, move_vae_to_cpu
        print(f"[RefGuide] Preparing {len(ref_guide_configs)} reference guide(s) for txt2img")
        move_vae_to_gpu(pipeline)
        ref_guides = prepare_reference_guide_latents(
            ref_guide_configs, pipeline, width, height, device, dtype, generator
        )
        move_vae_to_cpu(pipeline)

    # Current prompt embeds (will be updated by callback)
    current_prompt_embeds = prompt_embeds
    current_negative_prompt_embeds = negative_prompt_embeds
    current_pooled_prompt_embeds = pooled_prompt_embeds
    current_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

    # ============================================================
    # DEBUG: Scheduler initialization (for comparison with training)
    # ============================================================
    print(f"\n[CustomSampling] [Debug] ========== SCHEDULER INITIALIZATION ==========")
    print(f"[CustomSampling] [Debug] Scheduler timesteps (first 5): {scheduler.timesteps[:5].tolist()}")
    print(f"[CustomSampling] [Debug] Scheduler timesteps (last 5): {scheduler.timesteps[-5:].tolist()}")
    print(f"[CustomSampling] [Debug] init_noise_sigma: {scheduler.init_noise_sigma}")
    print(f"[CustomSampling] [Debug] Latents shape: {latents.shape}, dtype: {latents.dtype}")
    print(f"[CustomSampling] [Debug] Latents AFTER init_noise_sigma scaling:")
    print(f"[CustomSampling] [Debug]   - min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}")

    print(f"[CustomSampling] Starting sampling loop with {num_inference_steps} steps")
    print(f"[CustomSampling] Actual timesteps: {len(timesteps)} (some schedulers like DPM2 use 2x steps)")
    print(f"[CustomSampling] Latents shape: {latents.shape}, dtype: {latents.dtype}")
    print(f"[CustomSampling] Prompt embeds shape: {prompt_embeds.shape}")

    # Send initial noise preview (step 0) before denoising loop starts
    if progress_callback is not None:
        print(f"[CustomSampling] Sending initial noise preview (step 0)")
        progress_callback(-1, len(timesteps), latents, cfg_metrics=None)

    # Get sigma_max for dynamic CFG scheduling
    sigma_max = 0.0
    if hasattr(scheduler, 'sigmas') and len(scheduler.sigmas) > 0:
        sigma_max = float(scheduler.sigmas[0].item())
    print(f"[CustomSampling] Sigma max: {sigma_max}, CFG schedule: {cfg_schedule_type}")

    # Track previous SNR for SNR-based adaptive CFG
    previous_snr = None
    first_iteration_debug = True

    # ---- In-loop hard-flatten setup (SD1.5/SDXL, opt-in) -----------------------
    _flatten_inject_steps, _flatten_vae_shift = _setup_inloop_flatten(
        pipeline, timesteps, spectrum, fbcache_ctrl,
        flatten_in_loop, flatten_in_loop_last_steps, flatten_in_loop_min_region)

    # Denoising loop
    for i, t in enumerate(timesteps):
        # Check for cancellation (only in inference context, not training)
        try:
            from core.pipeline import pipeline_manager
            if pipeline_manager.cancel_requested:
                print("[CustomSampling] Generation cancelled by user")
                raise RuntimeError("Generation cancelled by user")
        except (ImportError, AttributeError):
            # pipeline_manager not available (e.g., in training subprocess)
            pass

        # Check if NAG should be deactivated based on sigma threshold
        if nag_active and nag_sigma_end > 0.0:
            if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
                current_sigma = float(scheduler.sigmas[i].item())
                if current_sigma < nag_sigma_end:
                    print(f"[CustomSampling] Deactivating NAG at step {i} (sigma={current_sigma:.4f} < {nag_sigma_end})")
                    from core.inference.nag_processor import restore_original_processors
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

        # Optimize: skip unconditional pass if guidance_scale ~= 1.0 and neither NAG
        # nor NegPip is active. NegPip needs the [negative, positive] batch so its
        # per-context V weights align (and negative-prompt double-negation works).
        do_classifier_free_guidance = (abs(current_guidance_scale - 1.0) > 1e-5) or nag_active or negpip_active

        # Prepare latent input based on CFG mode
        if nag_active:
            # NAG mode: Use batch approach (legacy, backward compatible)
            # Both NAG and CFG use double batch structure: [negative, positive]
            # NAG processors will apply guidance in attention space on positive batch
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Prepare prompt embeddings for NAG
            # Official NAG implementation concatenates: [cfg_negative, cfg_positive] + [nag_negative]
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
        elif do_classifier_free_guidance:
            if is_deus or style_active or style_refs_active:
                # DEUS (variable seq-len embeds) or active style transfer (single- or
                # multi-reference): prepare a single (batch=1) latent -- the U-Net is
                # called twice below with different embeds/context instead of a
                # batch-2 concatenation, so style's reference-K/V injection can be
                # isolated to ONLY the conditional pass (mirrors the Krea2 wiring's
                # split forward).
                latent_model_input = scheduler.scale_model_input(latents, t)
                # prompt_embeds_input is not used for this path (separate negative/positive passes below)
                prompt_embeds_input = None
            else:
                # Standard CFG (SDXL/SD1.5): Use batch approach [negative, positive] (batch=2)
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
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
            original_size = _resolve_sdxl_original_size(height, width, original_size_w, original_size_h, original_size_scale)
            crops_coords_top_left = (0, 0)
            target_size = (height, width)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)

            if nag_active or do_classifier_free_guidance:
                # NAG mode or standard CFG (SDXL/SD1.5): Use batch approach
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

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids
                }
            else:
                # No CFG: Use single-batch
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
            # Use autocast for FP8 or UINT quantized U-Net (required for FP16 activations)
            is_uint_quantized = hasattr(unet, '_is_uint_quantized') and unet._is_uint_quantized
            use_autocast = unet_dtype == torch.float8_e4m3fn or unet_dtype == torch.float8_e5m2 or is_uint_quantized

            if is_deus and do_classifier_free_guidance:
                # DEUS: 2-Pass CFG - separate U-Net calls for negative and positive embeddings
                # This is required because DEUS has variable sequence length embeddings
                # that cannot be batch concatenated

                # ============================================================
                # DEBUG: First iteration details (DEUS 2-Pass CFG)
                # ============================================================
                if first_iteration_debug:
                    print(f"\n[CustomSampling] [Debug] ========== FIRST DENOISING ITERATION (DEUS 2-Pass CFG) ==========")
                    print(f"[CustomSampling] [Debug] timestep (t): {t.item()}")
                    print(f"[CustomSampling] [Debug] latent_model_input shape: {latent_model_input.shape}, dtype: {latent_model_input.dtype}")
                    print(f"[CustomSampling] [Debug] latent_model_input min: {latent_model_input.min().item():.4f}, max: {latent_model_input.max().item():.4f}, mean: {latent_model_input.mean().item():.4f}")
                    print(f"[CustomSampling] [Debug] negative_prompt_embeds shape: {current_negative_prompt_embeds.shape}, dtype: {current_negative_prompt_embeds.dtype}")
                    print(f"[CustomSampling] [Debug] positive_prompt_embeds shape: {current_prompt_embeds.shape}, dtype: {current_prompt_embeds.dtype}")

                # Pass 1: Unconditional (negative) prediction
                unet_kwargs_uncond = {
                    "encoder_hidden_states": current_negative_prompt_embeds,
                }
                if down_block_res_samples is not None:
                    unet_kwargs_uncond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_uncond["mid_block_additional_residual"] = mid_block_res_sample

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample
                else:
                    noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample

                # Pass 2: Conditional (positive) prediction
                unet_kwargs_cond = {
                    "encoder_hidden_states": current_prompt_embeds,
                }
                if down_block_res_samples is not None:
                    unet_kwargs_cond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_cond["mid_block_additional_residual"] = mid_block_res_sample

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample
                else:
                    noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample

                # noise_pred_uncond and noise_pred_text are already separate (no chunk needed)
            elif style_active and do_classifier_free_guidance:
                # Active style transfer: 2-Pass CFG (separate uncond/cond U-Net calls),
                # so the reference-style KV injection can be isolated to ONLY the
                # conditional (positive) pass -- the unconditional pass is always run
                # with no style context (untouched), exactly like the Krea2 wiring.
                from core.inference.reference_style import StyleContext
                from core.inference.attention_processors import set_style_context

                def _slice_added_cond_kwargs(row: int):
                    if not (is_sdxl and added_cond_kwargs):
                        return None
                    text_embeds = added_cond_kwargs.get("text_embeds")
                    return {
                        "text_embeds": text_embeds[row:row + 1] if text_embeds is not None else None,
                        "time_ids": added_cond_kwargs["time_ids"][row:row + 1],
                    }

                # Pass 1: Unconditional (negative) prediction -- no style context.
                set_style_context(unet, None)
                unet_kwargs_uncond = {"encoder_hidden_states": current_negative_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_uncond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_uncond["mid_block_additional_residual"] = mid_block_res_sample
                uncond_added_cond_kwargs = _slice_added_cond_kwargs(0)
                if uncond_added_cond_kwargs is not None:
                    unet_kwargs_uncond["added_cond_kwargs"] = uncond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample
                else:
                    noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample

                # Pass 2: Conditional (positive) prediction -- style capture + inject,
                # only when this step falls within the style config's active range.
                cond_added_cond_kwargs = _slice_added_cond_kwargs(1)
                if style_cfg.is_step_active(i, num_inference_steps):
                    ref_t = scheduler.add_noise(style_ref_x0, style_eps_ref, t.unsqueeze(0))
                    ref_t_scaled = scheduler.scale_model_input(ref_t, t)
                    progress = style_cfg.step_progress(i, num_inference_steps)

                    ref_unet_kwargs = {"encoder_hidden_states": current_prompt_embeds}
                    if cond_added_cond_kwargs is not None:
                        ref_unet_kwargs["added_cond_kwargs"] = cond_added_cond_kwargs

                    capture_ctx = StyleContext(mode="capture", config=style_cfg, progress=progress)
                    set_style_context(unet, capture_ctx)
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)
                    else:
                        unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)

                    inject_ctx = StyleContext(mode="inject", config=style_cfg, store=capture_ctx.store, progress=progress)
                    set_style_context(unet, inject_ctx)

                unet_kwargs_cond = {"encoder_hidden_states": current_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_cond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_cond["mid_block_additional_residual"] = mid_block_res_sample
                if cond_added_cond_kwargs is not None:
                    unet_kwargs_cond["added_cond_kwargs"] = cond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample
                else:
                    noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample

                # --- CFG-decoupled style guidance (SDXL/SD1.5 prototype) ---
                # Disabled by default (style_guidance_scale is None/<=0): this block
                # is skipped entirely and noise_pred_text stays exactly the styled
                # cond pred above (cond_s) -- byte-identical to before this feature.
                # Enabled (>0) AND this step actually injected style (is_step_active
                # above, same gate as the capture/inject pass): run a 3rd forward --
                # SAME unet_kwargs_cond (same encoder_hidden_states/residuals/
                # added_cond_kwargs as the styled pass) but with style context
                # cleared -- to get the cond prediction WITHOUT style (cond_ns), then
                # rewrite noise_pred_text so the UNCHANGED shared CFG combine
                # (noise_pred = uncond + cfg*(text - uncond)) reproduces the
                # style-guidance target:
                #   uncond + cfg*(cond_ns - uncond) + lambda*(cond_s - cond_ns)
                # Algebra: let text' = cond_ns + (lambda/cfg)*(cond_s - cond_ns).
                # Substituting into the shared combine:
                #   uncond + cfg*(text' - uncond)
                # = uncond + cfg*(cond_ns - uncond) + cfg*(lambda/cfg)*(cond_s-cond_ns)
                # = uncond + cfg*(cond_ns - uncond) + lambda*(cond_s - cond_ns)
                # which is exactly the target above -- so assigning
                # noise_pred_text = text' lets the untouched shared combine line
                # produce style guidance decoupled from cfg. cfg is guarded (>1e-6)
                # even though do_classifier_free_guidance guarantees cfg>1 here; if
                # it were ever ~0 we skip the rewrite and keep noise_pred_text=cond_s.
                if (
                    style_cfg.style_guidance_scale is not None
                    and style_cfg.style_guidance_scale > 0
                    and style_cfg.is_step_active(i, num_inference_steps)
                ):
                    cond_s = noise_pred_text
                    set_style_context(unet, None)
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            cond_ns = unet(latent_model_input, t, **unet_kwargs_cond).sample
                    else:
                        cond_ns = unet(latent_model_input, t, **unet_kwargs_cond).sample
                    cfg = current_guidance_scale
                    lam = style_cfg.style_guidance_scale
                    if cfg > 1e-6:
                        noise_pred_text = cond_ns + (lam / cfg) * (cond_s - cond_ns)

                set_style_context(unet, None)

                # noise_pred_uncond and noise_pred_text are already separate (no chunk needed)
            elif style_refs_active and do_classifier_free_guidance:
                # Multi-reference (N>1) style transfer: 2-Pass CFG identical to the
                # single-ref branch above, but the conditional pass runs ONE capture
                # forward PER reference (each with its OWN StyleTransferConfig --
                # block_range, strengths, freq curve, step gating -- fully
                # independent) into its own store, then a single multi-ref inject via
                # inject_kv_multi (see attention_processors.UnifiedAttnProcessor).
                # style_refs_active requires 2+ entries (see its definition above),
                # so this branch never fires for a single reference -- that case is
                # always routed through style_active above, unchanged.
                from core.inference.reference_style import StyleContext
                from core.inference.attention_processors import set_style_context

                def _slice_added_cond_kwargs(row: int):
                    if not (is_sdxl and added_cond_kwargs):
                        return None
                    text_embeds = added_cond_kwargs.get("text_embeds")
                    return {
                        "text_embeds": text_embeds[row:row + 1] if text_embeds is not None else None,
                        "time_ids": added_cond_kwargs["time_ids"][row:row + 1],
                    }

                # Pass 1: Unconditional (negative) prediction -- no style context.
                set_style_context(unet, None)
                unet_kwargs_uncond = {"encoder_hidden_states": current_negative_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_uncond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_uncond["mid_block_additional_residual"] = mid_block_res_sample
                uncond_added_cond_kwargs = _slice_added_cond_kwargs(0)
                if uncond_added_cond_kwargs is not None:
                    unet_kwargs_uncond["added_cond_kwargs"] = uncond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample
                else:
                    noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample

                # Pass 2: Conditional (positive) prediction -- one capture forward PER
                # active reference (skipping refs not step-active this step, mirroring
                # the single-ref "not is_step_active -> no injection" case), then a
                # single multi-ref inject.
                cond_added_cond_kwargs = _slice_added_cond_kwargs(1)
                active_style_refs = []
                for _sref_cfg, _sref_x0, _sref_eps in style_refs:
                    if not _sref_cfg.is_step_active(i, num_inference_steps):
                        continue
                    ref_t = scheduler.add_noise(_sref_x0, _sref_eps, t.unsqueeze(0))
                    ref_t_scaled = scheduler.scale_model_input(ref_t, t)
                    ref_progress = _sref_cfg.step_progress(i, num_inference_steps)

                    ref_unet_kwargs = {"encoder_hidden_states": current_prompt_embeds}
                    if cond_added_cond_kwargs is not None:
                        ref_unet_kwargs["added_cond_kwargs"] = cond_added_cond_kwargs

                    ref_capture_ctx = StyleContext(mode="capture", config=_sref_cfg, progress=ref_progress)
                    set_style_context(unet, ref_capture_ctx)
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)
                    else:
                        unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)

                    active_style_refs.append((ref_capture_ctx.store, _sref_cfg))

                if active_style_refs:
                    overall_progress = active_style_refs[0][1].step_progress(i, num_inference_steps)
                    inject_ctx = StyleContext(
                        mode="inject", config=active_style_refs[0][1], refs=active_style_refs,
                        combine_mode=style_combine_mode, progress=overall_progress,
                    )
                    set_style_context(unet, inject_ctx)
                # else: no reference active this step -- context stays None (set by
                # Pass 1 above), matching the single-ref "not step-active" case.

                unet_kwargs_cond = {"encoder_hidden_states": current_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_cond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_cond["mid_block_additional_residual"] = mid_block_res_sample
                if cond_added_cond_kwargs is not None:
                    unet_kwargs_cond["added_cond_kwargs"] = cond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample
                else:
                    noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample

                set_style_context(unet, None)

                # noise_pred_uncond and noise_pred_text are already separate (no chunk needed)
            elif spectrum is not None and spectrum_block_ctrl is None and not spectrum.is_anchor(i):
                # Spectrum output (black-box) skip step: forecast the raw U-Net output
                # (Eq.14) instead of running the forward. NAG/NegPip effects are baked
                # into the recorded anchor outputs, so they carry through the forecast.
                noise_pred = spectrum.forecast(i)
            else:
                # Standard batch approach: NAG mode, Standard CFG (SDXL/SD1.5), or No CFG
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

                # ============================================================
                # DEBUG: First iteration details (for comparison with training)
                # ============================================================
                if first_iteration_debug:
                    print(f"\n[CustomSampling] [Debug] ========== FIRST DENOISING ITERATION ==========")
                    print(f"[CustomSampling] [Debug] timestep (t): {t.item()}")
                    print(f"[CustomSampling] [Debug] latent_model_input shape: {latent_model_input.shape}, dtype: {latent_model_input.dtype}")
                    print(f"[CustomSampling] [Debug] latent_model_input min: {latent_model_input.min().item():.4f}, max: {latent_model_input.max().item():.4f}, mean: {latent_model_input.mean().item():.4f}")
                    print(f"[CustomSampling] [Debug] prompt_embeds_input shape: {prompt_embeds_input.shape}, dtype: {prompt_embeds_input.dtype}")

                # Spectrum block mode: deep blocks are captured (anchor) or forecast
                # (skip) inside the U-Net via wrappers installed for this single call.
                # FBCache block mode: deep blocks are reused (hit) or captured (miss)
                # dynamically per step via wrappers installed for this single call.
                if spectrum_block_ctrl is not None:
                    spectrum_block_ctrl.begin_step(i)
                if fbcache_ctrl is not None:
                    fbcache_ctrl.begin_step(i)
                try:
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            noise_pred = unet(
                                latent_model_input,
                                t,
                                **unet_kwargs
                            ).sample
                    else:
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            **unet_kwargs
                        ).sample
                finally:
                    if spectrum_block_ctrl is not None:
                        spectrum_block_ctrl.end_step()
                    if fbcache_ctrl is not None:
                        fbcache_ctrl.end_step()

                # Spectrum output mode: record this actual-pass output and refit.
                if spectrum is not None and spectrum_block_ctrl is None:
                    spectrum.record(i, noise_pred)

        # Perform guidance with CFG
        if do_classifier_free_guidance:
            if is_deus or style_active or style_refs_active:
                # DEUS / active style transfer (single- or multi-reference): noise_pred_uncond
                # and noise_pred_text are already separate (from the 2-Pass CFG block above).
                pass  # Variables already set in the 2-Pass CFG block
            else:
                # NAG mode or Standard CFG: noise_pred has [negative, positive] batches
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

            # ============================================================
            # DEBUG: Noise prediction AFTER CFG (for comparison with training)
            # ============================================================
            if first_iteration_debug:
                print(f"[CustomSampling] [Debug] noise_pred AFTER CFG shape: {noise_pred.shape}, dtype: {noise_pred.dtype}")
                print(f"[CustomSampling] [Debug] noise_pred AFTER CFG min: {noise_pred.min().item():.4f}, max: {noise_pred.max().item():.4f}, mean: {noise_pred.mean().item():.4f}")

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
        step_output = scheduler.step(noise_pred, t, latents, generator=step_generator)
        latents = step_output.prev_sample

        # Get predicted x0 (original sample) if available from scheduler
        # This is the model's prediction of what the final denoised image should look like
        # Use .detach().clone() to disconnect from computation graph and ensure contiguous memory
        # This prevents GPU sync delays during TAESD preview decoding
        pred_original_sample = getattr(step_output, 'pred_original_sample', None)
        if pred_original_sample is not None:
            pred_original_sample = pred_original_sample.detach().clone()

        # Reference Guide blending (txt2img)
        if ref_guides:
            ref_frac = i / num_inference_steps
            latents, pred_original_sample = apply_reference_guide_blend(
                latents, pred_original_sample, ref_guides, ref_frac, i, timesteps, scheduler
            )

        # In-loop hard-flatten of the flat background (SD1.5/SDXL, opt-in).
        if flatten_in_loop and i in _flatten_inject_steps:
            latents, _ = inloop_hard_flatten_step(
                pipeline, latents, pred_original_sample,
                flatten_in_loop_min_region, _flatten_vae_shift)

        # ============================================================
        # DEBUG: Latents AFTER scheduler.step() (for comparison with training)
        # ============================================================
        if first_iteration_debug:
            print(f"[CustomSampling] [Debug] latents AFTER scheduler.step() shape: {latents.shape}, dtype: {latents.dtype}")
            print(f"[CustomSampling] [Debug] latents AFTER scheduler.step() min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}")
            if pred_original_sample is not None:
                print(f"[CustomSampling] [Debug] pred_original_sample available: shape={pred_original_sample.shape}")
            print(f"[CustomSampling] [Debug] ========== END FIRST ITERATION ==========\n")
            first_iteration_debug = False

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

            progress_callback(i, len(timesteps), latents, cfg_metrics=cfg_metrics, pred_original_sample=pred_original_sample)

        # Step callback
        if step_callback is not None:
            callback_kwargs = {"latents": latents}
            callback_kwargs = step_callback(pipeline, i, t, callback_kwargs)
            latents = callback_kwargs.get("latents", latents)

    print(f"[CustomSampling] Sampling complete, decoding latents")

    # Clean up Reference Guide GPU tensors
    if ref_guides:
        for rg in ref_guides:
            del rg["clean_latent"], rg["noise"]
        ref_guides.clear()

    # Restore original processors if NAG or NegPip was active
    if original_processors is not None and (nag_active or negpip_active):
        from core.inference.nag_processor import restore_original_processors
        restore_original_processors(unet, original_processors)

    # ===== STAGE 3: VAE DECODE =====
    from core.vram_optimization import log_device_status, move_unet_to_cpu, move_vae_to_gpu, move_vae_to_cpu

    # Offload U-Net to CPU to free VRAM for VAE
    move_unet_to_cpu(pipeline)

    # loop_decode="none": latent passthrough for loop generation. Skip VAE/PiD
    # entirely and hand back the clean scaled latent (pre-unscale -- the SAME
    # frame img2img's init_latents encode path produces:
    # (vae.encode(img) - shift_factor) * scaling_factor), so a later img2img/
    # inpaint step can feed it directly as init_latents_override with no VAE
    # round-trip. pipeline.py distinguishes this from an Image.Image return.
    if loop_decode == "none":
        print("[CustomSampling] loop_decode='none': skipping VAE decode (latent passthrough)")
        return latents

    from core.models.pid.pid_vae_wrapper import PidVaeWrapper
    _pid_active = isinstance(pipeline.vae, PidVaeWrapper)
    # loop_decode="cheap": when a PiD override is active, decode with its
    # EMBEDDED real SDXL VAE instead of running the PiD student net (cheaper
    # intermediate-step decode for a loop). No-op when PiD isn't active
    # ("cheap" == "full" == the normal VAE either way).
    _use_real_vae_only = loop_decode == "cheap" and _pid_active

    # PiD stages its own ~2.7GB net for the final decode and does NOT use the held
    # real VAE, so don't stage that VAE to GPU when PiD is active (saves VRAM and
    # avoids leaving it resident if the PiD decode raises) -- UNLESS this decode
    # is routed to the real VAE instead (loop_decode="cheap").
    if not _pid_active or _use_real_vae_only:
        move_vae_to_gpu(pipeline)
    log_device_status("Ready for VAE decode", pipeline, vision_encoder=vision_encoder)

    # Decode latents to image
    _vae_shift = getattr(pipeline.vae.config, "shift_factor", None) or 0.0
    latents = latents / pipeline.vae.config.scaling_factor + _vae_shift
    if not _pid_active or _use_real_vae_only:
        # Convert latents to VAE dtype (important for fp16 VAE with fp32 latents).
        # PiD re-normalizes in fp32 internally, so keep full precision for it.
        latents = latents.to(dtype=pipeline.vae.dtype)
    with torch.no_grad():
        if _pid_active and not _use_real_vae_only:
            # PiD (Pixel Diffusion Decoder) override: run the SDXL 4-step
            # distilled decoder instead of a plain VAE decode. `latents` here is
            # the SAME already-unscaled tensor a plain decode would receive —
            # the wrapper re-normalizes it back into PiD's training frame
            # internally (F1, see pid_vae_wrapper.py's module docstring).
            _pid_seed = generator.initial_seed() if generator is not None else 0
            _decode_cb = _make_pid_decode_progress(progress_callback)
            image = pipeline.vae.pid_final_decode(latents, seed=_pid_seed, progress_callback=_decode_cb).sample
        else:
            image = pipeline.vae.decode(latents, return_dict=True).sample

    # Free GPU latents before VAE offload
    del latents

    # Offload VAE to CPU after decoding (skipped for PiD — its held VAE was never
    # staged; the PiD net offloads itself in pid_final_decode's finally).
    if not _pid_active or _use_real_vae_only:
        move_vae_to_cpu(pipeline)

    # Convert to PIL with robust nan/inf handling (moves image tensor to CPU internally)
    image = vae_output_to_pil(image, color_flatten_strength=color_flatten_strength)

    return image


def custom_img2img_sampling_loop(
    pipeline: Union[StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline],
    init_image: Optional[Image.Image],
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
    width: Optional[int] = None,  # Target width (resizes init_image if specified)
    height: Optional[int] = None,  # Target height (resizes init_image if specified)
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
    is_deus: bool = False,  # DEUS model flag - uses 2-Pass CFG instead of batch concatenation
    ref_guide_configs: Optional[List[Dict]] = None,  # Reference Guide configs for latent blending
    vision_encoder=None,  # SigLIP2 VisionEncoderWrapper for VRAM status logging
    original_size_w: int = 0,  # SDXL micro-cond override: explicit original width (0 = auto)
    original_size_h: int = 0,  # SDXL micro-cond override: explicit original height (0 = auto)
    original_size_scale: float = 1.0,  # SDXL micro-cond: original_size = output size * scale (when not explicit)
    negpip_weights: Optional[Dict[str, torch.Tensor]] = None,  # NegPip signed per-token weights {"pos","neg","nag_neg"}; auto-set when prompt has negative weights
    loop_decode: str = "full",  # Loop-generation decode mode: "full" (decode as usual) | "cheap"
                                # (if a PidVaeWrapper is active, use its embedded real VAE instead of
                                # the PiD student net; no-op otherwise) | "none" (skip decode entirely,
                                # return the pre-unscale latent for the caller to cache -- see the
                                # Stage-3 VAE DECODE section below).
    spectrum_enable: bool = False,  # Spectrum (Adaptive Spectral Feature Forecasting) acceleration
    spectrum_w: float = 0.5,  # Spectral/linear mix (1.0 = spectral only; lower = more linear/stable)
    spectrum_w_decay: float = 0.0,  # OPT-IN per-step decay exponent for spectrum_w (0 = off, default)
    spectrum_delta_cap: float = 0.0,  # OPT-IN trajectory speed limiter multiplier K (0 = off, default)
    spectrum_m: int = 4,  # Number of Chebyshev basis
    spectrum_lam: float = 0.1,  # Ridge regularization
    spectrum_warmup_steps: int = 3,  # Leading full-eval steps
    spectrum_window_size: int = 4,  # Initial skip interval
    spectrum_flex_window: float = 0.75,  # Skip damping (0 = max skip)
    spectrum_tail: float = 0.12,  # Fraction of final steps forced to actual passes (detail)
    spectrum_feature_mode: str = "output",  # "output" (black-box) or "block" (deep-feature)
    spectrum_cache_branch: int = 1,  # block mode: down_blocks[cache_branch:] + mid are forecast
    spectrum_max_cache: int = 0,  # forecaster sliding-window size (0 = unlimited)
    fbcache_enable: bool = False,  # FBCache (First Block Cache) dynamic U-Net block caching
    fbcache_threshold: float = 0.12,  # relative-L1 indicator threshold (higher = more skips/faster)
    fbcache_warmup_steps: int = 1,  # always compute the first N steps
    fbcache_cache_branch: int = 1,  # indicator = down[branch]; reused region = down[branch+1:]+mid
    color_flatten_strength: int = 0,  # 0-100 post-decode chroma smoothing; 0 = off
    vae_drift_correction: bool = False,  # subtract VAE round-trip DC bias (strength-independent)
    flatten_in_loop: bool = False,  # in-loop hard-flatten of the flat background (SD1.5/SDXL)
    flatten_in_loop_last_steps: int = 3,  # inject on the last N ACTUAL denoise steps
    flatten_in_loop_min_region: float = 0.02,  # flat-region area gate (fraction of frame)
    style_cfg=None,  # core.inference.reference_style.StyleTransferConfig, or None (default off)
    style_ref_x0: Optional[torch.Tensor] = None,  # VAE-encoded style reference latent (build_style_transfer)
    style_eps_ref: Optional[torch.Tensor] = None,  # fixed reference noise (build_style_transfer)
    style_refs: Optional[List[Tuple[Any, torch.Tensor, torch.Tensor]]] = None,  # multi-reference (N>1): list of (StyleTransferConfig, ref_x0, eps_ref) triples, one per reference image; only consulted when len>1 (build_style_transfer_multi)
    style_combine_mode: str = "stack",  # "stack" | "common_concept" -- multi-reference combine mode (core.inference.reference_style.inject_kv_multi)
    init_latents_override: Optional[torch.Tensor] = None,  # Loop-generation latent passthrough: when
                                # set, SKIPS the init_image VAE-encode entirely and uses this tensor
                                # directly as init_latents (already in the (encode(img) - shift) *
                                # scaling_factor frame -- the SAME frame this function's own encode
                                # block produces). init_image is then only a size placeholder (its
                                # pixels are never read/encoded) -- see pipeline.py's generate_img2img.
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
    # CRITICAL FIX: Use U-Net's device instead of pipeline.device
    # pipeline.device returns cpu after text encoders are offloaded
    if hasattr(pipeline, 'unet'):
        # Get device from first parameter (nn.Module doesn't have .device attribute)
        device = next(pipeline.unet.parameters()).device
    else:
        device = pipeline.device

    # Get U-Net dtype, but use float16 for latents if U-Net is FP8 or UINT quantized
    # (torch.randn doesn't support FP8, and UINT quantization uses FP16 activations)
    # nn.Module doesn't have .dtype, get from first parameter
    unet_dtype = next(pipeline.unet.parameters()).dtype
    is_uint_quantized = hasattr(pipeline.unet, '_is_uint_quantized') and pipeline.unet._is_uint_quantized

    if unet_dtype == torch.float8_e4m3fn or unet_dtype == torch.float8_e5m2 or is_uint_quantized:
        dtype = torch.float16  # Use float16 for latents
        if is_uint_quantized:
            print(f"[CustomSampling] U-Net is UINT quantized, using float16 for latents and activations")
        else:
            print(f"[CustomSampling] U-Net is {unet_dtype}, using float16 for latents")
    else:
        dtype = unet_dtype

    # Check if SDXL by checking if text_encoder_2 exists
    is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

    # DEUS uses 2-Pass CFG (separate negative/positive passes) instead of batch concatenation
    if is_deus:
        print(f"[CustomSampling] [img2img] DEUS mode: Using 2-Pass CFG (separate negative/positive passes)")

    print(f"[CustomSampling] [img2img] Pipeline type: {type(pipeline).__name__}, is_sdxl: {is_sdxl}, is_deus: {is_deus}")

    # Use ancestral_generator for stochastic samplers (always provided by pipeline)
    step_generator = ancestral_generator
    if ancestral_generator is not None:
        print(f"[CustomSampling] Using ancestral generator for stochastic sampler")

    # Get components
    unet = pipeline.unet
    scheduler = pipeline.scheduler

    # Training-free reference-style transfer (StyleAligned/VSP-style KV injection).
    # No style config => style_active is False and nothing below this ever runs
    # (byte-identical to the pre-style-transfer code path).
    style_active = style_cfg is not None and style_ref_x0 is not None and style_eps_ref is not None
    if style_active:
        from core.inference.attention_processors import ensure_style_block_indices
        num_style_blocks = ensure_style_block_indices(unet)
        style_cfg.resolve_default_block_range(num_style_blocks)
        print(f"[CustomSampling] [img2img] Style transfer active: {num_style_blocks} self-attention layers "
              f"eligible, block_range={style_cfg.block_range} (None = all)")

    # Multi-reference (N>1) style transfer -- see the txt2img loop for the full
    # rationale; mutually exclusive with style_active (single-ref) above.
    style_refs_active = style_refs is not None and len(style_refs) > 1
    if style_refs_active:
        from core.inference.attention_processors import ensure_style_block_indices
        num_style_blocks = ensure_style_block_indices(unet)
        for _style_cfg_i, _style_x0_i, _style_eps_i in style_refs:
            _style_cfg_i.resolve_default_block_range(num_style_blocks)
        print(f"[CustomSampling] [img2img] Multi-ref style transfer active: {len(style_refs)} references, "
              f"{num_style_blocks} self-attention layers eligible")

    # Resize init_image if width/height are specified
    if width is not None and height is not None:
        if init_image.size != (width, height):
            print(f"[CustomSampling] Resizing init_image from {init_image.size} to ({width}, {height})")
            init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)

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

    from core.vram_optimization import move_vae_to_gpu, move_vae_to_cpu
    _drift_input_mean = None
    _drift_ref_latents = None

    if init_latents_override is not None:
        # Loop-generation latent passthrough (loop_decode="none" chaining): the
        # cached latent is ALREADY in the (encode(img) - shift_factor) *
        # scaling_factor frame -- the SAME frame this block's own encode
        # produces below -- so it is used directly, with no re-scaling and no
        # VAE encode/staging at all. init_image (a size-only placeholder in
        # this path -- see pipeline.py's generate_img2img) is never read.
        print(f"[CustomSampling] Using cached init latents (latent passthrough), shape: {init_latents_override.shape}")
        init_latents = init_latents_override.to(device=device, dtype=dtype)
        if vae_drift_correction:
            print("[CustomSampling] vae_drift_correction requested but no source image is available "
                  "for latent passthrough -- skipping (needs the real input pixels)")
    else:
        # Ensure VAE is on GPU for initial encoding
        vae_device = next(pipeline.vae.parameters()).device
        if vae_device.type != device:
            print(f"[CustomSampling] Moving VAE from {vae_device} to {device} for initial encoding")
            move_vae_to_gpu(pipeline)

        # Encode initial image to latents
        # Convert PIL image to tensor if needed
        if isinstance(init_image, Image.Image):
            init_image = torch.from_numpy(np.array(init_image)).float() / 255.0
            init_image = init_image.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            init_image = init_image * 2.0 - 1.0  # Normalize to [-1, 1]

        # Use VAE's dtype for encoding (VAE may be FP32 even if U-Net is FP16)
        vae_dtype = next(pipeline.vae.parameters()).dtype
        with torch.no_grad():
            init_latents = pipeline.vae.encode(
                init_image.to(device=device, dtype=vae_dtype)
            ).latent_dist.sample(generator)
            init_latents = (init_latents - (getattr(pipeline.vae.config, "shift_factor", None) or 0.0)) * pipeline.vae.config.scaling_factor
            # Convert latents back to U-Net dtype for denoising
            init_latents = init_latents.to(dtype=dtype)

            # VAE DC-drift correction: capture the input image's per-channel mean and a
            # reference latent (== encode(input)) for a round-trip decode near the final
            # decode. This corrects a fixed VAE property, so it is measured once here and
            # is strength-independent. Only when the option is enabled (zero cost otherwise).
            if vae_drift_correction:
                _drift_input_mean = (
                    init_image.to(device=device, dtype=torch.float32) / 2 + 0.5
                ).clamp(0, 1).mean(dim=(0, 2, 3), keepdim=True)
                _drift_ref_latents = init_latents.detach().clone()

    # Prepare Reference Guide latents while VAE is still on GPU (stage it now if
    # the encode above was skipped -- init_latents_override never staged it).
    ref_guides = []
    if ref_guide_configs:
        if init_latents_override is not None:
            move_vae_to_gpu(pipeline)
        # Use actual image dimensions (width/height may be None in img2img)
        ref_w = width if width is not None else original_width
        ref_h = height if height is not None else original_height
        print(f"[RefGuide] Preparing {len(ref_guide_configs)} reference guide(s) for img2img ({ref_w}x{ref_h})")
        ref_guides = prepare_reference_guide_latents(
            ref_guide_configs, pipeline, ref_w, ref_h, device, dtype, generator
        )

    # Move VAE back to CPU after initial encoding (harmless no-op if it was
    # never staged -- e.g. latent passthrough with no reference guides).
    print(f"[CustomSampling] Moving VAE to CPU after initial encoding")
    move_vae_to_cpu(pipeline)

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

    # NegPip: auto-activated by the pipeline when the prompt(s) contain negative
    # emphasis weights (see custom_sampling_loop for details). Folded into the NAG
    # processor when NAG is active; otherwise a dedicated NegPip processor is used.
    negpip_active, nag_token_weights, negpip_token_weights = _prepare_negpip_weights(negpip_weights, nag_active)
    if negpip_active:
        seq = (nag_token_weights if nag_active else negpip_token_weights).shape[-1]
        print(f"[CustomSampling] NegPip active (img2img): signed V weighting on cross-attention (seq={seq})")

    if nag_active:
        from core.inference.nag_processor import set_nag_processors
        print(f"[CustomSampling] NAG enabled: scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}, sigma_end={nag_sigma_end}")

        original_processors = set_nag_processors(unet, nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha, attention_type=attention_type, token_weights=nag_token_weights)

        nag_negative_prompt_embeds = nag_negative_prompt_embeds.to(device=device, dtype=dtype)
        if is_sdxl and nag_negative_pooled_prompt_embeds is not None:
            nag_negative_pooled_prompt_embeds = nag_negative_pooled_prompt_embeds.to(device=device, dtype=dtype)
    elif negpip_active:
        from core.inference.negpip_processor import set_negpip_processors
        original_processors = set_negpip_processors(unet, negpip_token_weights, attention_type=attention_type)

    # Spectrum (Adaptive Spectral Feature Forecasting) acceleration -- see the txt2img
    # loop for details. Auto-disabled for unstable per-step conditioning / DEUS / very
    # few steps.
    spectrum = None
    spectrum_block_ctrl = None
    if spectrum_enable:
        _n_steps = len(timesteps)
        if is_deus or has_controlnet or (prompt_embeds_callback is not None):
            print("[Spectrum] requested but disabled (prompt-editing / ControlNet / DEUS; "
                  "needs stable conditioning)")
        elif _n_steps < spectrum_warmup_steps + 3:
            print(f"[Spectrum] requested but disabled ({_n_steps} steps < warmup+3)")
        else:
            from core.inference.spectrum_forecaster import SpectrumForecaster
            _block = spectrum_feature_mode == "block"
            _max_cache = spectrum_max_cache if spectrum_max_cache > 0 else (6 if _block else 5)
            spectrum = SpectrumForecaster(
                _n_steps, num_basis=spectrum_m, lam=spectrum_lam, w=spectrum_w,
                w_decay=spectrum_w_decay,
                delta_cap=spectrum_delta_cap,
                warmup_steps=spectrum_warmup_steps, window_size=spectrum_window_size,
                flex_window=spectrum_flex_window, tail_fraction=spectrum_tail,
                max_cache=_max_cache,
            )
            if _block:
                from core.inference.spectrum_unet import SpectrumBlockController
                spectrum_block_ctrl = SpectrumBlockController(unet, spectrum, cache_branch=spectrum_cache_branch)
                print(f"[Spectrum] enabled (img2img, block mode): {len(spectrum.anchors)}/{_n_steps} "
                      f"deep-feature passes, cache_branch={spectrum_block_ctrl.branch}/{spectrum_block_ctrl.n_down}")
            else:
                print(f"[Spectrum] enabled (img2img, output mode): {len(spectrum.anchors)}/{_n_steps} actual passes")

    # Style transfer yields to NAG / ControlNet / Spectrum -- see the txt2img loop for
    # the full rationale (incompatible batch layouts / stale spectrum state).
    # TODO: ControlNet(structure)+style(appearance) is a desirable combo; supporting
    # it needs per-pass batch-1 residual recompute -- future enhancement.
    if style_active and (nag_active or has_controlnet or spectrum is not None):
        print("[CustomSampling] Style transfer disabled: not compatible with NAG / "
              "ControlNet / Spectrum in this version")
        _add_generation_warning(
            "Style transfer disabled: not compatible with NAG / ControlNet / Spectrum in this version.",
            code="style_incompatible",
        )
        style_active = False

    # Multi-reference style transfer has the exact same batch-layout incompatibility
    # (separate per-ref capture forwards + a 2-Pass CFG cond/uncond split).
    if style_refs_active and (nag_active or has_controlnet or spectrum is not None):
        print("[CustomSampling] Multi-ref style transfer disabled: not compatible with NAG / "
              "ControlNet / Spectrum in this version")
        _add_generation_warning(
            "Style transfer disabled: not compatible with NAG / ControlNet / Spectrum in this version.",
            code="style_incompatible",
        )
        style_refs_active = False

    # FBCache: dynamic per-step deep-block caching, mutually exclusive with Spectrum
    # and auto-disabled for unstable conditioning (prompt editing / ControlNet / DEUS),
    # and also for style transfer (its capture forward would pollute the cache; see
    # the txt2img loop for details).
    fbcache_ctrl = None
    if fbcache_enable:
        if spectrum_block_ctrl is not None or spectrum is not None:
            print("[FBCache] requested but disabled (Spectrum is active; mutually exclusive)")
        elif is_deus or has_controlnet or (prompt_embeds_callback is not None) or style_active or style_refs_active:
            print("[FBCache] requested but disabled (prompt-editing / ControlNet / DEUS / "
                  "style transfer; needs stable conditioning)")
        else:
            from core.inference.fbcache_unet import build_unet_fbcache_controller
            fbcache_ctrl = build_unet_fbcache_controller(
                unet,
                {
                    "fbcache_enable": fbcache_enable,
                    "fbcache_threshold": fbcache_threshold,
                    "fbcache_warmup_steps": fbcache_warmup_steps,
                    "fbcache_cache_branch": fbcache_cache_branch,
                },
                label="img2img",
            )
    print(f"[CustomSampling] Starting img2img loop with {len(timesteps)} steps (strength={strength})")
    print(f"[CustomSampling] Latents shape: {latents.shape}, dtype: {latents.dtype}")

    # Get sigma_max for dynamic CFG scheduling
    sigma_max = 0.0
    if hasattr(scheduler, 'sigmas') and len(scheduler.sigmas) > 0:
        sigma_max = float(scheduler.sigmas[0].item())
    print(f"[CustomSampling] Sigma max: {sigma_max}, CFG schedule: {cfg_schedule_type}")

    # Track previous SNR for SNR-based adaptive CFG
    previous_snr = None
    first_iteration_debug = True

    # Send initial noise preview (step 0) before denoising loop starts
    if progress_callback is not None:
        print(f"[CustomSampling] Sending initial noise preview (step 0)")
        progress_callback(-1, len(timesteps), latents, cfg_metrics=None)

    # ---- In-loop hard-flatten setup (SD1.5/SDXL, opt-in) -----------------------
    _flatten_inject_steps, _flatten_vae_shift = _setup_inloop_flatten(
        pipeline, timesteps, spectrum, fbcache_ctrl,
        flatten_in_loop, flatten_in_loop_last_steps, flatten_in_loop_min_region)

    # Denoising loop
    for i, t in enumerate(timesteps):
        # Check for cancellation (only in inference context, not training)
        try:
            from core.pipeline import pipeline_manager
            if pipeline_manager.cancel_requested:
                print("[CustomSampling] Generation cancelled by user")
                raise RuntimeError("Generation cancelled by user")
        except (ImportError, AttributeError):
            # pipeline_manager not available (e.g., in training subprocess)
            pass

        # Check if NAG should be deactivated based on sigma threshold
        if nag_active and nag_sigma_end > 0.0:
            if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
                current_sigma = float(scheduler.sigmas[i].item())
                if current_sigma < nag_sigma_end:
                    print(f"[CustomSampling] Deactivating NAG at step {i} (sigma={current_sigma:.4f} < {nag_sigma_end})")
                    from core.inference.nag_processor import restore_original_processors
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

        # Optimize: skip unconditional pass if guidance_scale ~= 1.0 and neither NAG
        # nor NegPip is active. NegPip needs the [negative, positive] batch so its
        # per-context V weights align (and negative-prompt double-negation works).
        do_classifier_free_guidance = (abs(current_guidance_scale - 1.0) > 1e-5) or nag_active or negpip_active

        # Prepare latent input based on CFG mode
        if nag_active:
            # NAG mode: Use batch approach (legacy, backward compatible)
            # Both NAG and CFG use double batch structure: [negative, positive]
            # NAG processors will apply guidance in attention space on positive batch
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Prepare prompt embeddings based on CFG and NAG configuration
            # Official NAG implementation concatenates: [cfg_negative, cfg_positive] + [nag_negative]
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

        elif do_classifier_free_guidance:
            if is_deus or style_active or style_refs_active:
                # DEUS (variable seq-len embeds) or active multi-reference style
                # transfer: prepare a single (batch=1) latent -- the U-Net is called
                # twice below with different embeds/context instead of a batch-2
                # concatenation (see the txt2img loop's style branch for the
                # rationale). Both single-ref (style_active) and multi-ref
                # (style_refs_active) need this batch=1 latent because the style
                # branch below runs two separate forwards; style_active was
                # previously MISSING from this img2img/inpaint gate (a pre-existing
                # single-ref crash -- txt2img always had it).
                latent_model_input = scheduler.scale_model_input(latents, t)
                prompt_embeds_input = None
            else:
                # Standard CFG (SDXL/SD1.5): Use batch approach [negative, positive] (batch=2)
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
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
            original_size = _resolve_sdxl_original_size(original_height, original_width, original_size_w, original_size_h, original_size_scale)
            crops_coords_top_left = (0, 0)
            target_size = (original_height, original_width)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)

            if nag_active or do_classifier_free_guidance:
                # NAG mode or standard CFG (SDXL/SD1.5): Use batch approach
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

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids
                }

            else:
                # No CFG: Use single-batch
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
            # Use autocast for FP8 or UINT quantized U-Net (required for FP16 activations)
            is_uint_quantized = hasattr(unet, '_is_uint_quantized') and unet._is_uint_quantized
            use_autocast = unet_dtype == torch.float8_e4m3fn or unet_dtype == torch.float8_e5m2 or is_uint_quantized

            if is_deus and do_classifier_free_guidance:
                # DEUS: 2-Pass CFG - separate U-Net calls for negative and positive embeddings

                # ============================================================
                # DEBUG: First iteration details (DEUS 2-Pass CFG)
                # ============================================================
                if first_iteration_debug:
                    print(f"\n[CustomSampling] [Debug] ========== FIRST DENOISING ITERATION (DEUS 2-Pass CFG) ==========")
                    print(f"[CustomSampling] [Debug] timestep (t): {t.item()}")
                    print(f"[CustomSampling] [Debug] latent_model_input shape: {latent_model_input.shape}, dtype: {latent_model_input.dtype}")
                    print(f"[CustomSampling] [Debug] latent_model_input min: {latent_model_input.min().item():.4f}, max: {latent_model_input.max().item():.4f}, mean: {latent_model_input.mean().item():.4f}")
                    print(f"[CustomSampling] [Debug] negative_prompt_embeds shape: {current_negative_prompt_embeds.shape}, dtype: {current_negative_prompt_embeds.dtype}")
                    print(f"[CustomSampling] [Debug] positive_prompt_embeds shape: {current_prompt_embeds.shape}, dtype: {current_prompt_embeds.dtype}")

                # Pass 1: Unconditional (negative) prediction
                unet_kwargs_uncond = {
                    "encoder_hidden_states": current_negative_prompt_embeds,
                }
                if down_block_res_samples is not None:
                    unet_kwargs_uncond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_uncond["mid_block_additional_residual"] = mid_block_res_sample

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample
                else:
                    noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample

                # Pass 2: Conditional (positive) prediction
                unet_kwargs_cond = {
                    "encoder_hidden_states": current_prompt_embeds,
                }
                if down_block_res_samples is not None:
                    unet_kwargs_cond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_cond["mid_block_additional_residual"] = mid_block_res_sample

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample
                else:
                    noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample
            elif style_active and do_classifier_free_guidance:
                # Active style transfer: 2-Pass CFG (separate uncond/cond U-Net calls),
                # so the reference-style KV injection can be isolated to ONLY the
                # conditional (positive) pass -- the unconditional pass is always run
                # with no style context (untouched), exactly like the txt2img wiring.
                from core.inference.reference_style import StyleContext
                from core.inference.attention_processors import set_style_context

                def _slice_added_cond_kwargs(row: int):
                    if not (is_sdxl and added_cond_kwargs):
                        return None
                    text_embeds = added_cond_kwargs.get("text_embeds")
                    return {
                        "text_embeds": text_embeds[row:row + 1] if text_embeds is not None else None,
                        "time_ids": added_cond_kwargs["time_ids"][row:row + 1],
                    }

                # Pass 1: Unconditional (negative) prediction -- no style context.
                set_style_context(unet, None)
                unet_kwargs_uncond = {"encoder_hidden_states": current_negative_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_uncond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_uncond["mid_block_additional_residual"] = mid_block_res_sample
                uncond_added_cond_kwargs = _slice_added_cond_kwargs(0)
                if uncond_added_cond_kwargs is not None:
                    unet_kwargs_uncond["added_cond_kwargs"] = uncond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample
                else:
                    noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample

                # Pass 2: Conditional (positive) prediction -- style capture + inject,
                # only when this step falls within the style config's active range.
                cond_added_cond_kwargs = _slice_added_cond_kwargs(1)
                if style_cfg.is_step_active(i, num_inference_steps):
                    ref_t = scheduler.add_noise(style_ref_x0, style_eps_ref, t.unsqueeze(0))
                    ref_t_scaled = scheduler.scale_model_input(ref_t, t)
                    progress = style_cfg.step_progress(i, num_inference_steps)

                    ref_unet_kwargs = {"encoder_hidden_states": current_prompt_embeds}
                    if cond_added_cond_kwargs is not None:
                        ref_unet_kwargs["added_cond_kwargs"] = cond_added_cond_kwargs

                    capture_ctx = StyleContext(mode="capture", config=style_cfg, progress=progress)
                    set_style_context(unet, capture_ctx)
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)
                    else:
                        unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)

                    inject_ctx = StyleContext(mode="inject", config=style_cfg, store=capture_ctx.store, progress=progress)
                    set_style_context(unet, inject_ctx)

                unet_kwargs_cond = {"encoder_hidden_states": current_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_cond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_cond["mid_block_additional_residual"] = mid_block_res_sample
                if cond_added_cond_kwargs is not None:
                    unet_kwargs_cond["added_cond_kwargs"] = cond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample
                else:
                    noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample

                # --- CFG-decoupled style guidance (SDXL/SD1.5 prototype) ---
                # Disabled by default (style_guidance_scale is None/<=0): this block
                # is skipped entirely and noise_pred_text stays exactly the styled
                # cond pred above (cond_s) -- byte-identical to before this feature.
                # Enabled (>0) AND this step actually injected style (is_step_active
                # above, same gate as the capture/inject pass): run a 3rd forward --
                # SAME unet_kwargs_cond (same encoder_hidden_states/residuals/
                # added_cond_kwargs as the styled pass) but with style context
                # cleared -- to get the cond prediction WITHOUT style (cond_ns), then
                # rewrite noise_pred_text so the UNCHANGED shared CFG combine
                # (noise_pred = uncond + cfg*(text - uncond)) reproduces the
                # style-guidance target:
                #   uncond + cfg*(cond_ns - uncond) + lambda*(cond_s - cond_ns)
                # Algebra: let text' = cond_ns + (lambda/cfg)*(cond_s - cond_ns).
                # Substituting into the shared combine:
                #   uncond + cfg*(text' - uncond)
                # = uncond + cfg*(cond_ns - uncond) + cfg*(lambda/cfg)*(cond_s-cond_ns)
                # = uncond + cfg*(cond_ns - uncond) + lambda*(cond_s - cond_ns)
                # which is exactly the target above -- so assigning
                # noise_pred_text = text' lets the untouched shared combine line
                # produce style guidance decoupled from cfg. cfg is guarded (>1e-6)
                # even though do_classifier_free_guidance guarantees cfg>1 here; if
                # it were ever ~0 we skip the rewrite and keep noise_pred_text=cond_s.
                if (
                    style_cfg.style_guidance_scale is not None
                    and style_cfg.style_guidance_scale > 0
                    and style_cfg.is_step_active(i, num_inference_steps)
                ):
                    cond_s = noise_pred_text
                    set_style_context(unet, None)
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            cond_ns = unet(latent_model_input, t, **unet_kwargs_cond).sample
                    else:
                        cond_ns = unet(latent_model_input, t, **unet_kwargs_cond).sample
                    cfg = current_guidance_scale
                    lam = style_cfg.style_guidance_scale
                    if cfg > 1e-6:
                        noise_pred_text = cond_ns + (lam / cfg) * (cond_s - cond_ns)

                set_style_context(unet, None)

                # noise_pred_uncond and noise_pred_text are already separate (no chunk needed)
            elif style_refs_active and do_classifier_free_guidance:
                # Multi-reference (N>1) style transfer: 2-Pass CFG identical to the
                # single-ref branch above, but the conditional pass runs ONE capture
                # forward PER reference (each with its OWN StyleTransferConfig --
                # block_range, strengths, freq curve, step gating -- fully
                # independent) into its own store, then a single multi-ref inject via
                # inject_kv_multi (see attention_processors.UnifiedAttnProcessor).
                # style_refs_active requires 2+ entries (see its definition above),
                # so this branch never fires for a single reference -- that case is
                # always routed through style_active above, unchanged.
                from core.inference.reference_style import StyleContext
                from core.inference.attention_processors import set_style_context

                def _slice_added_cond_kwargs(row: int):
                    if not (is_sdxl and added_cond_kwargs):
                        return None
                    text_embeds = added_cond_kwargs.get("text_embeds")
                    return {
                        "text_embeds": text_embeds[row:row + 1] if text_embeds is not None else None,
                        "time_ids": added_cond_kwargs["time_ids"][row:row + 1],
                    }

                # Pass 1: Unconditional (negative) prediction -- no style context.
                set_style_context(unet, None)
                unet_kwargs_uncond = {"encoder_hidden_states": current_negative_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_uncond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_uncond["mid_block_additional_residual"] = mid_block_res_sample
                uncond_added_cond_kwargs = _slice_added_cond_kwargs(0)
                if uncond_added_cond_kwargs is not None:
                    unet_kwargs_uncond["added_cond_kwargs"] = uncond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample
                else:
                    noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample

                # Pass 2: Conditional (positive) prediction -- one capture forward PER
                # active reference (skipping refs not step-active this step, mirroring
                # the single-ref "not is_step_active -> no injection" case), then a
                # single multi-ref inject.
                cond_added_cond_kwargs = _slice_added_cond_kwargs(1)
                active_style_refs = []
                for _sref_cfg, _sref_x0, _sref_eps in style_refs:
                    if not _sref_cfg.is_step_active(i, num_inference_steps):
                        continue
                    ref_t = scheduler.add_noise(_sref_x0, _sref_eps, t.unsqueeze(0))
                    ref_t_scaled = scheduler.scale_model_input(ref_t, t)
                    ref_progress = _sref_cfg.step_progress(i, num_inference_steps)

                    ref_unet_kwargs = {"encoder_hidden_states": current_prompt_embeds}
                    if cond_added_cond_kwargs is not None:
                        ref_unet_kwargs["added_cond_kwargs"] = cond_added_cond_kwargs

                    ref_capture_ctx = StyleContext(mode="capture", config=_sref_cfg, progress=ref_progress)
                    set_style_context(unet, ref_capture_ctx)
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)
                    else:
                        unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)

                    active_style_refs.append((ref_capture_ctx.store, _sref_cfg))

                if active_style_refs:
                    overall_progress = active_style_refs[0][1].step_progress(i, num_inference_steps)
                    inject_ctx = StyleContext(
                        mode="inject", config=active_style_refs[0][1], refs=active_style_refs,
                        combine_mode=style_combine_mode, progress=overall_progress,
                    )
                    set_style_context(unet, inject_ctx)
                # else: no reference active this step -- context stays None (set by
                # Pass 1 above), matching the single-ref "not step-active" case.

                unet_kwargs_cond = {"encoder_hidden_states": current_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_cond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_cond["mid_block_additional_residual"] = mid_block_res_sample
                if cond_added_cond_kwargs is not None:
                    unet_kwargs_cond["added_cond_kwargs"] = cond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample
                else:
                    noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample

                set_style_context(unet, None)

                # noise_pred_uncond and noise_pred_text are already separate (no chunk needed)
            elif spectrum is not None and spectrum_block_ctrl is None and not spectrum.is_anchor(i):
                # Spectrum output (black-box) skip step: forecast the raw U-Net output
                # (Eq.14) instead of running the forward. NAG/NegPip effects are baked
                # into the recorded anchor outputs, so they carry through the forecast.
                noise_pred = spectrum.forecast(i)
            else:
                # Standard batch approach: NAG mode, Standard CFG (SDXL/SD1.5), or No CFG
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

                # ============================================================
                # DEBUG: First iteration details (for comparison with training)
                # ============================================================
                if first_iteration_debug:
                    print(f"\n[CustomSampling] [Debug] ========== FIRST DENOISING ITERATION ==========")
                    print(f"[CustomSampling] [Debug] timestep (t): {t.item()}")
                    print(f"[CustomSampling] [Debug] latent_model_input shape: {latent_model_input.shape}, dtype: {latent_model_input.dtype}")
                    print(f"[CustomSampling] [Debug] latent_model_input min: {latent_model_input.min().item():.4f}, max: {latent_model_input.max().item():.4f}, mean: {latent_model_input.mean().item():.4f}")
                    print(f"[CustomSampling] [Debug] prompt_embeds_input shape: {prompt_embeds_input.shape}, dtype: {prompt_embeds_input.dtype}")

                # Spectrum block mode: deep blocks are captured (anchor) or forecast
                # (skip) inside the U-Net via wrappers installed for this single call.
                # FBCache block mode: deep blocks are reused (hit) or captured (miss)
                # dynamically per step via wrappers installed for this single call.
                if spectrum_block_ctrl is not None:
                    spectrum_block_ctrl.begin_step(i)
                if fbcache_ctrl is not None:
                    fbcache_ctrl.begin_step(i)
                try:
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            noise_pred = unet(
                                latent_model_input,
                                t,
                                **unet_kwargs
                            ).sample
                    else:
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            **unet_kwargs
                        ).sample
                finally:
                    if spectrum_block_ctrl is not None:
                        spectrum_block_ctrl.end_step()
                    if fbcache_ctrl is not None:
                        fbcache_ctrl.end_step()

                # Spectrum output mode: record this actual-pass output and refit.
                if spectrum is not None and spectrum_block_ctrl is None:
                    spectrum.record(i, noise_pred)

        # Perform guidance with CFG
        if do_classifier_free_guidance:
            if is_deus or style_active or style_refs_active:
                # DEUS / active multi-reference style transfer: noise_pred_uncond and
                # noise_pred_text are already separate (from the 2-Pass CFG block
                # above), for both single-ref (style_active) and multi-ref
                # (style_refs_active). style_active was previously MISSING from this
                # img2img/inpaint gate (a pre-existing single-ref crash: the else
                # branch chunks a batch-2 noise_pred that the 2-pass block never set).
                pass  # Variables already set in the 2-Pass CFG block
            else:
                # NAG mode or Standard CFG: noise_pred has [negative, positive] batches
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

            # ============================================================
            # DEBUG: Noise prediction AFTER CFG (for comparison with training)
            # ============================================================
            if first_iteration_debug:
                print(f"[CustomSampling] [Debug] noise_pred AFTER CFG shape: {noise_pred.shape}, dtype: {noise_pred.dtype}")
                print(f"[CustomSampling] [Debug] noise_pred AFTER CFG min: {noise_pred.min().item():.4f}, max: {noise_pred.max().item():.4f}, mean: {noise_pred.mean().item():.4f}")

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
        step_output = scheduler.step(noise_pred, t, latents, generator=step_generator)
        latents = step_output.prev_sample

        # Get predicted x0 (original sample) if available from scheduler
        # Use .detach().clone() to disconnect from computation graph and ensure contiguous memory
        pred_original_sample = getattr(step_output, 'pred_original_sample', None)
        if pred_original_sample is not None:
            pred_original_sample = pred_original_sample.detach().clone()

        # Reference Guide blending (img2img)
        if ref_guides:
            ref_frac = (t_start + i) / num_inference_steps
            latents, pred_original_sample = apply_reference_guide_blend(
                latents, pred_original_sample, ref_guides, ref_frac, i, timesteps, scheduler
            )

        # In-loop hard-flatten of the flat background (SD1.5/SDXL, opt-in).
        if flatten_in_loop and i in _flatten_inject_steps:
            latents, _ = inloop_hard_flatten_step(
                pipeline, latents, pred_original_sample,
                flatten_in_loop_min_region, _flatten_vae_shift)

        # ============================================================
        # DEBUG: Latents AFTER scheduler.step() (for comparison with training)
        # ============================================================
        if first_iteration_debug:
            print(f"[CustomSampling] [Debug] latents AFTER scheduler.step() shape: {latents.shape}, dtype: {latents.dtype}")
            print(f"[CustomSampling] [Debug] latents AFTER scheduler.step() min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}")
            if pred_original_sample is not None:
                print(f"[CustomSampling] [Debug] pred_original_sample available: shape={pred_original_sample.shape}")
            print(f"[CustomSampling] [Debug] ========== END FIRST ITERATION ==========\n")
            first_iteration_debug = False

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

            progress_callback(i, len(timesteps), latents, cfg_metrics=cfg_metrics, pred_original_sample=pred_original_sample)

        # Step callback
        if step_callback is not None:
            callback_kwargs = {"latents": latents}
            callback_kwargs = step_callback(pipeline, t_start + i, t, callback_kwargs)
            latents = callback_kwargs.get("latents", latents)

    print(f"[CustomSampling] Sampling complete, decoding latents")

    # Restore original processors if NAG or NegPip was active
    if original_processors is not None and (nag_active or negpip_active):
        from core.inference.nag_processor import restore_original_processors
        restore_original_processors(unet, original_processors)

    # ===== STAGE 3: VAE DECODE =====
    from core.vram_optimization import log_device_status, move_unet_to_cpu, move_vae_to_gpu, move_vae_to_cpu

    # Offload U-Net to CPU to free VRAM for VAE
    move_unet_to_cpu(pipeline)

    # loop_decode="none": latent passthrough for loop generation -- see
    # custom_sampling_loop's Stage-3 site for the full rationale. `latents`
    # here is still the pre-unscale scaled x0, the SAME frame this function's
    # own init_latents encode block produces, so a later img2img step can feed
    # it back as init_latents_override with no re-scaling.
    if loop_decode == "none":
        print("[CustomSampling] loop_decode='none': skipping VAE decode (latent passthrough)")
        return latents

    from core.models.pid.pid_vae_wrapper import PidVaeWrapper
    _pid_active = isinstance(pipeline.vae, PidVaeWrapper)
    # loop_decode="cheap": see custom_sampling_loop's Stage-3 site.
    _use_real_vae_only = loop_decode == "cheap" and _pid_active
    # PiD stages its own net and does not use the held real VAE for the final
    # decode — don't stage that VAE to GPU when PiD is active, unless this
    # decode is routed to the real VAE instead (loop_decode="cheap").
    if not _pid_active or _use_real_vae_only:
        move_vae_to_gpu(pipeline)
    log_device_status("Ready for VAE decode", pipeline, vision_encoder=vision_encoder)

    # Decode latents to image
    _vae_shift = getattr(pipeline.vae.config, "shift_factor", None) or 0.0
    latents = latents / pipeline.vae.config.scaling_factor + _vae_shift
    if not _pid_active or _use_real_vae_only:
        # Convert latents to VAE dtype (fp16 VAE + fp32 latents); PiD re-normalizes
        # in fp32 internally so keep full precision for it.
        latents = latents.to(dtype=pipeline.vae.dtype)
    with torch.no_grad():
        if _pid_active and not _use_real_vae_only:
            # PiD override: see custom_sampling_loop's Stage-3 site for the
            # F1/F2 rationale (`latents` is already the pre-unscaled tensor
            # the wrapper re-normalizes internally).
            _pid_seed = generator.initial_seed() if generator is not None else 0
            _decode_cb = _make_pid_decode_progress(progress_callback)
            image = pipeline.vae.pid_final_decode(latents, seed=_pid_seed, progress_callback=_decode_cb).sample
        else:
            image = pipeline.vae.decode(latents, return_dict=True).sample

    # Free GPU latents before VAE offload
    del latents

    # VAE DC-drift correction (one extra reference decode, VAE still on GPU).
    # PiD has no encoder-based drift-correction path (accepted but not applied,
    # same "accepted but not applied" pattern as the DiT archs — see
    # arch_capabilities.py's vae_drift_correction entries). Also inert when
    # _drift_ref_latents is None (init_latents_override / latent-passthrough
    # path -- no source image to measure the round-trip bias against).
    _dc_bias = None
    if vae_drift_correction and not _pid_active and _drift_ref_latents is not None:
        _dc_bias = compute_vae_dc_bias(pipeline, _drift_ref_latents, _drift_input_mean, _vae_shift)

    # Offload VAE to CPU after decoding (skipped for PiD — its held VAE was never staged).
    if not _pid_active or _use_real_vae_only:
        move_vae_to_cpu(pipeline)

    # Convert to PIL with robust nan/inf handling
    image = vae_output_to_pil(image, color_flatten_strength=color_flatten_strength, dc_bias=_dc_bias)

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
    width: Optional[int] = None,  # Target width (resizes init_image and mask if specified)
    height: Optional[int] = None,  # Target height (resizes init_image and mask if specified)
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
    is_deus: bool = False,  # DEUS model flag - uses 2-Pass CFG instead of batch concatenation
    ref_guide_configs: Optional[List[Dict]] = None,  # Reference Guide configs for latent blending
    vision_encoder=None,  # SigLIP2 VisionEncoderWrapper for VRAM status logging
    original_size_w: int = 0,  # SDXL micro-cond override: explicit original width (0 = auto)
    original_size_h: int = 0,  # SDXL micro-cond override: explicit original height (0 = auto)
    original_size_scale: float = 1.0,  # SDXL micro-cond: original_size = output size * scale (when not explicit)
    negpip_weights: Optional[Dict[str, torch.Tensor]] = None,  # NegPip signed per-token weights {"pos","neg","nag_neg"}; auto-set when prompt has negative weights
    loop_decode: str = "full",  # Loop-generation decode mode: "full" (decode as usual) | "cheap"
                                # (if a PidVaeWrapper is active, use its embedded real VAE instead of
                                # the PiD student net; no-op otherwise) | "none" (skip decode entirely,
                                # return the pre-unscale latent for the caller to cache -- see the
                                # Stage-3 VAE DECODE section below).
    spectrum_enable: bool = False,  # Spectrum (Adaptive Spectral Feature Forecasting) acceleration
    spectrum_w: float = 0.5,  # Spectral/linear mix (1.0 = spectral only; lower = more linear/stable)
    spectrum_w_decay: float = 0.0,  # OPT-IN per-step decay exponent for spectrum_w (0 = off, default)
    spectrum_delta_cap: float = 0.0,  # OPT-IN trajectory speed limiter multiplier K (0 = off, default)
    spectrum_m: int = 4,  # Number of Chebyshev basis
    spectrum_lam: float = 0.1,  # Ridge regularization
    spectrum_warmup_steps: int = 3,  # Leading full-eval steps
    spectrum_window_size: int = 4,  # Initial skip interval
    spectrum_flex_window: float = 0.75,  # Skip damping (0 = max skip)
    spectrum_tail: float = 0.12,  # Fraction of final steps forced to actual passes (detail)
    spectrum_feature_mode: str = "output",  # "output" (black-box) or "block" (deep-feature)
    spectrum_cache_branch: int = 1,  # block mode: down_blocks[cache_branch:] + mid are forecast
    spectrum_max_cache: int = 0,  # forecaster sliding-window size (0 = unlimited)
    fbcache_enable: bool = False,  # FBCache (First Block Cache) dynamic U-Net block caching
    fbcache_threshold: float = 0.12,  # relative-L1 indicator threshold (higher = more skips/faster)
    fbcache_warmup_steps: int = 1,  # always compute the first N steps
    fbcache_cache_branch: int = 1,  # indicator = down[branch]; reused region = down[branch+1:]+mid
    color_flatten_strength: int = 0,  # 0-100 post-decode chroma smoothing; 0 = off
    vae_drift_correction: bool = False,  # subtract VAE round-trip DC bias (strength-independent)
    flatten_in_loop: bool = False,  # in-loop hard-flatten of the flat background (SD1.5/SDXL)
    flatten_in_loop_last_steps: int = 3,  # inject on the last N ACTUAL denoise steps
    flatten_in_loop_min_region: float = 0.02,  # flat-region area gate (fraction of frame)
    style_cfg=None,  # core.inference.reference_style.StyleTransferConfig, or None (default off)
    style_ref_x0: Optional[torch.Tensor] = None,  # VAE-encoded style reference latent (build_style_transfer)
    style_eps_ref: Optional[torch.Tensor] = None,  # fixed reference noise (build_style_transfer)
    style_refs: Optional[List[Tuple[Any, torch.Tensor, torch.Tensor]]] = None,  # multi-reference (N>1): list of (StyleTransferConfig, ref_x0, eps_ref) triples, one per reference image; only consulted when len>1 (build_style_transfer_multi)
    style_combine_mode: str = "stack",  # "stack" | "common_concept" -- multi-reference combine mode (core.inference.reference_style.inject_kv_multi)
    outpaint_noise_init: bool = False,  # Outpaint noise-init (core.pipeline.generate_outpaint's `_outpaint_noise_init`):
                                        # the GENERATE region (mask_latent==1) starts from pure
                                        # architecture-native noise instead of noised encode(canvas
                                        # fill), independent of the fill content -- see
                                        # core.inference.outpaint_utils.compose_outpaint_start.
                                        # ALSO gates the B1 x0-space projection injection + boundary
                                        # color proximal below (this same flag; normal inpaint is
                                        # byte-identical when it is False).
    outpaint_boundary_color_strength: float = 0.25,  # B1 low-frequency boundary color proximal strength
                                                      # (0 = off). Only active when outpaint_noise_init is
                                                      # True; see _outpaint_apply_boundary_color.
    outpaint_resample_count: int = 2,  # B2 RePaint-style time-travel resampling: number of denoise
                                        # traversals ("r") through each resampled band segment (1 = off).
                                        # Only active when outpaint_noise_init is True, not a dedicated
                                        # 9ch inpaint UNet, and the scheduler is resample-compatible
                                        # (Euler/EulerAncestral/DDIM/DDPM) -- see
                                        # _build_outpaint_resample_schedule / _outpaint_resample_jump.
    outpaint_jump_length: int = 4,  # B2 jump-back length ("u", in step indices) for each resample cycle.
) -> Image.Image:
    """Custom inpaint sampling loop with prompt editing and ControlNet support"""
    # CRITICAL FIX: Use U-Net's device instead of pipeline.device
    # pipeline.device returns cpu after text encoders are offloaded
    if hasattr(pipeline, 'unet'):
        # Get device from first parameter (nn.Module doesn't have .device attribute)
        device = next(pipeline.unet.parameters()).device
    else:
        device = pipeline.device

    # Get U-Net dtype, but use float16 for latents if U-Net is FP8 or UINT quantized
    # (torch.randn doesn't support FP8, and UINT quantization uses FP16 activations)
    # nn.Module doesn't have .dtype, get from first parameter
    unet_dtype = next(pipeline.unet.parameters()).dtype
    is_uint_quantized = hasattr(pipeline.unet, '_is_uint_quantized') and pipeline.unet._is_uint_quantized

    if unet_dtype == torch.float8_e4m3fn or unet_dtype == torch.float8_e5m2 or is_uint_quantized:
        dtype = torch.float16  # Use float16 for latents
        if is_uint_quantized:
            print(f"[CustomSampling] U-Net is UINT quantized, using float16 for latents and activations")
        else:
            print(f"[CustomSampling] U-Net is {unet_dtype}, using float16 for latents")
    else:
        dtype = unet_dtype

    # Check if SDXL by checking if text_encoder_2 exists
    is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

    # DEUS uses 2-Pass CFG (separate negative/positive passes) instead of batch concatenation
    if is_deus:
        print(f"[CustomSampling] [inpaint] DEUS mode: Using 2-Pass CFG (separate negative/positive passes)")

    print(f"[CustomSampling] [inpaint] Pipeline type: {type(pipeline).__name__}, is_sdxl: {is_sdxl}, is_deus: {is_deus}")

    # Use ancestral_generator for stochastic samplers (always provided by pipeline)
    step_generator = ancestral_generator
    if ancestral_generator is not None:
        print(f"[CustomSampling] Using ancestral generator for stochastic sampler")

    unet = pipeline.unet
    vae = pipeline.vae
    scheduler = pipeline.scheduler

    # Training-free reference-style transfer (StyleAligned/VSP-style KV injection).
    # No style config => style_active is False and nothing below this ever runs
    # (byte-identical to the pre-style-transfer code path).
    style_active = style_cfg is not None and style_ref_x0 is not None and style_eps_ref is not None
    if style_active:
        from core.inference.attention_processors import ensure_style_block_indices
        num_style_blocks = ensure_style_block_indices(unet)
        style_cfg.resolve_default_block_range(num_style_blocks)
        print(f"[CustomSampling] [inpaint] Style transfer active: {num_style_blocks} self-attention layers "
              f"eligible, block_range={style_cfg.block_range} (None = all)")

    # Multi-reference (N>1) style transfer -- see the txt2img loop for the full
    # rationale; mutually exclusive with style_active (single-ref) above.
    style_refs_active = style_refs is not None and len(style_refs) > 1
    if style_refs_active:
        from core.inference.attention_processors import ensure_style_block_indices
        num_style_blocks = ensure_style_block_indices(unet)
        for _style_cfg_i, _style_x0_i, _style_eps_i in style_refs:
            _style_cfg_i.resolve_default_block_range(num_style_blocks)
        print(f"[CustomSampling] [inpaint] Multi-ref style transfer active: {len(style_refs)} references, "
              f"{num_style_blocks} self-attention layers eligible")

    # Resize init_image and mask_image if width/height are specified
    if width is not None and height is not None:
        if init_image.size != (width, height):
            print(f"[CustomSampling] Resizing init_image from {init_image.size} to ({width}, {height})")
            init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)
        if mask_image.size != (width, height):
            print(f"[CustomSampling] Resizing mask_image from {mask_image.size} to ({width}, {height})")
            mask_image = mask_image.resize((width, height), Image.Resampling.LANCZOS)

    # Check if this is an inpaint-specific UNet (9 channels) or regular UNet (4 channels)
    # Regular UNets cannot accept concatenated mask+image, so we'll use img2img-style masking
    #
    # NEW: User setting controls whether to use dedicated 9ch inpaint model or mask blending
    # - inpaint_use_dedicated_model=False (default): Always use mask blending (like Z-Image/FLUX.2)
    # - inpaint_use_dedicated_model=True: Use 9ch inpaint model if available (legacy SD/SDXL method)
    unet_supports_9ch = unet.config.in_channels == 9
    use_dedicated_model_setting = get_inpaint_use_dedicated_model_setting()

    # Only use 9ch inpaint mode if BOTH: setting is enabled AND UNet supports it
    is_inpaint_unet = unet_supports_9ch and use_dedicated_model_setting

    print(f"[CustomSampling] UNet in_channels: {unet.config.in_channels}, "
          f"use_dedicated_model_setting: {use_dedicated_model_setting}, "
          f"is_inpaint_unet: {is_inpaint_unet}")

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

    # OUTPAINT B1 guard: the noise-init GENERATE-region start
    # (`noise * scheduler.init_noise_sigma`, below) assumes `timesteps[0]` IS
    # the schedule's max-noise timestep. generate_outpaint forces
    # denoising_strength=1.0 whenever outpaint_noise_init is active, which
    # normally yields t_start=0 via either branch above -- EXCEPT
    # `_setup_img2img_steps` with `img2img_fix_steps=False` computes
    # `actual_steps = int(min(strength, 0.999) * requested_steps)`, capping
    # strength at 0.999 and off-by-one'ing t_start to 1 even when the caller
    # requested strength=1.0. Force t_start=0 defensively so the two stay
    # consistent regardless of that setting.
    if outpaint_noise_init and t_start != 0:
        print(f"[CustomSampling] [outpaint] t_start={t_start} != 0 while outpaint_noise_init is active "
              f"(likely img2img_fix_steps=False rounding denoising_strength=1.0 down to 0.999) -- forcing t_start=0")
        t_start = 0

    timesteps = timesteps[t_start:]

    # ============================================================
    # OUTPAINT B2: RePaint-style time-travel resample schedule (design doc
    # section "B2"). Gated on outpaint_noise_init + a resample-compatible
    # scheduler + not a dedicated 9ch inpaint UNet (mirrors the B1 gate --
    # 9ch inpaint models never go through the x0-projection this resampling
    # relies on). `_outpaint_resample_active=False` (the off-path, including
    # ALL normal non-outpaint inpaint calls) makes
    # `_build_outpaint_resample_schedule` return the plain
    # `[(0, False), (1, False), ..., (T-1, False)]` walk -- iteration-order-
    # identical to `enumerate(timesteps)`, so the main loop below can
    # unconditionally iterate the schedule without any behavior change here.
    # ============================================================
    _outpaint_resample_active = False
    if outpaint_noise_init and not is_inpaint_unet and outpaint_resample_count > 1 and outpaint_jump_length > 0:
        _outpaint_scheduler_cls = type(scheduler).__name__
        if (_outpaint_scheduler_cls in _OUTPAINT_RESAMPLE_SIGMA_SCHEDULERS
                or _outpaint_scheduler_cls in _OUTPAINT_RESAMPLE_VP_SCHEDULERS):
            _outpaint_resample_active = True
        else:
            print(f"[Outpaint][B2] resample requested (outpaint_resample_count={outpaint_resample_count}) but "
                  f"scheduler {_outpaint_scheduler_cls} is unsupported (Euler/Euler-a/DDIM/DDPM only, no "
                  f"cross-step solver state) -- disabling resampling, running B1 only")
            _add_generation_warning(
                f"Outpaint time-travel resampling is not supported with the {_outpaint_scheduler_cls} sampler "
                "(only Euler/Euler-a/DDIM/DDPM hold no cross-step solver state) -- running without resampling.",
                code="outpaint_resample_unsupported_sampler",
            )

    _outpaint_visit_schedule = _build_outpaint_resample_schedule(
        len(timesteps),
        outpaint_resample_count if _outpaint_resample_active else 1,
        outpaint_jump_length,
        _OUTPAINT_RESAMPLE_BAND_LO,
        _OUTPAINT_RESAMPLE_BAND_HI,
    )
    if _outpaint_resample_active:
        _outpaint_extra_visits = len(_outpaint_visit_schedule) - len(timesteps)
        print(f"[Outpaint][B2] time-travel resampling active: {len(timesteps)} nominal steps -> "
              f"{len(_outpaint_visit_schedule)} actual NFE ({_outpaint_extra_visits} extra denoise passes, "
              f"~{len(_outpaint_visit_schedule) / max(1, len(timesteps)):.2f}x)")
        _add_generation_warning(
            f"Outpaint time-travel resampling active: ~{len(_outpaint_visit_schedule) / max(1, len(timesteps)):.2f}x "
            f"the requested step count ({len(_outpaint_visit_schedule)} actual denoise passes instead of "
            f"{len(timesteps)}).",
            code="outpaint_resample_nfe",
        )

    # Ensure VAE is on GPU for initial encoding
    from core.vram_optimization import move_vae_to_gpu, move_vae_to_cpu
    vae_device = next(pipeline.vae.parameters()).device
    if vae_device.type != device:
        print(f"[CustomSampling] Moving VAE from {vae_device} to {device} for initial encoding")
        move_vae_to_gpu(pipeline)

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

    # Use VAE's dtype for encoding (VAE may be FP32 even if U-Net is FP16)
    vae_dtype = next(pipeline.vae.parameters()).dtype
    with torch.no_grad():
        init_latents = pipeline.vae.encode(
            init_image_tensor.to(device=device, dtype=vae_dtype)
        ).latent_dist.sample(generator)
        init_latents = (init_latents - (getattr(pipeline.vae.config, "shift_factor", None) or 0.0)) * pipeline.vae.config.scaling_factor
        # Convert latents back to U-Net dtype for denoising
        init_latents = init_latents.to(dtype=dtype)

        # VAE DC-drift correction: capture input mean + reference latent (see the
        # img2img loop). Strength-independent; only when enabled.
        _drift_input_mean = None
        _drift_ref_latents = None
        if vae_drift_correction:
            _drift_input_mean = (
                init_image_tensor.to(device=device, dtype=torch.float32) / 2 + 0.5
            ).clamp(0, 1).mean(dim=(0, 2, 3), keepdim=True)
            _drift_ref_latents = init_latents.detach().clone()

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
            blurred = init_image_tensor.to(device=device, dtype=vae_dtype)
            for _ in range(blur_iterations):
                blurred = F.conv2d(blurred, kernel_1d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1), padding=(0, kernel_size // 2), groups=3)
                blurred = F.conv2d(blurred, kernel_1d.t().unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1), padding=(kernel_size // 2, 0), groups=3)

            print(f"[CustomSampling] Blur applied: kernel_size={kernel_size}, iterations={blur_iterations}, strength={inpaint_blur_strength}")

            with torch.no_grad():
                blurred_latents = pipeline.vae.encode(blurred).latent_dist.sample(generator)
                blurred_latents = (blurred_latents - (getattr(pipeline.vae.config, "shift_factor", None) or 0.0)) * pipeline.vae.config.scaling_factor
                blurred_latents = blurred_latents.to(dtype=dtype)

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
    if outpaint_noise_init:
        # Outpaint noise-init: the GENERATE region (mask_latent==1) starts from
        # pure architecture-native noise (`noise * init_noise_sigma` -- the
        # SAME native start custom_sampling_loop's txt2img path uses), NOT
        # from a noised encode of the canvas fill -- independent of the fill
        # content. The KEEP region (mask_latent==0) is unaffected: it still
        # gets the normal noised init below. SAME `noise` tensor is reused for
        # both terms (continuity with the per-step keep re-injection further
        # down, which also reuses `noise`).
        from core.inference.outpaint_utils import compose_outpaint_start
        latents = compose_outpaint_start(
            scheduler.add_noise(init_latents, noise, timesteps[0:1]),
            noise * scheduler.init_noise_sigma,
            mask_latent,
        )
    else:
        latents = scheduler.add_noise(init_latents, noise, timesteps[0:1])

    # OUTPAINT B1: precompute the low-frequency boundary color proximal's
    # STATIC inputs once (mask_latent/image_latents never change across
    # steps) -- collar_weight (W_b(d)) and the extended keep-side low-freq
    # target G_kappa(image_latents). None/no-op when the flag or strength is
    # off, or there is no generate region at all.
    _outpaint_collar_weight_map = None
    _outpaint_target_lowfreq = None
    if outpaint_noise_init and outpaint_boundary_color_strength > 0.0 and not is_inpaint_unet:
        _outpaint_collar_weight_map = _outpaint_collar_weight(mask_latent)
        if _outpaint_collar_weight_map is not None:
            _outpaint_target_lowfreq = _outpaint_gaussian_lowpass(image_latents)

    # Prepare Reference Guide latents while VAE is still on GPU
    ref_guides = []
    if ref_guide_configs:
        ref_w = width if width is not None else original_width
        ref_h = height if height is not None else original_height
        print(f"[RefGuide] Preparing {len(ref_guide_configs)} reference guide(s) for inpaint ({ref_w}x{ref_h})")
        ref_guides = prepare_reference_guide_latents(
            ref_guide_configs, pipeline, ref_w, ref_h, device, dtype, generator
        )

    # Move VAE back to CPU after initial encoding
    print(f"[CustomSampling] Moving VAE to CPU after initial encoding")
    move_vae_to_cpu(pipeline)

    current_prompt_embeds = prompt_embeds
    current_negative_prompt_embeds = negative_prompt_embeds
    current_pooled_prompt_embeds = pooled_prompt_embeds
    current_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

    # Setup NAG if enabled
    nag_active = nag_enable and nag_negative_prompt_embeds is not None
    original_processors = None

    # NegPip: auto-activated by the pipeline when the prompt(s) contain negative
    # emphasis weights. Folded into the NAG processor when NAG is active; otherwise
    # a dedicated NegPip processor is used.
    negpip_active, nag_token_weights, negpip_token_weights = _prepare_negpip_weights(negpip_weights, nag_active)
    if negpip_active:
        seq = (nag_token_weights if nag_active else negpip_token_weights).shape[-1]
        print(f"[CustomSampling] NegPip active (inpaint): signed V weighting on cross-attention (seq={seq})")

    if nag_active:
        from core.inference.nag_processor import set_nag_processors
        print(f"[CustomSampling] NAG enabled: scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}, sigma_end={nag_sigma_end}")

        original_processors = set_nag_processors(unet, nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha, attention_type=attention_type, token_weights=nag_token_weights)

        nag_negative_prompt_embeds = nag_negative_prompt_embeds.to(device=device, dtype=dtype)
        if is_sdxl and nag_negative_pooled_prompt_embeds is not None:
            nag_negative_pooled_prompt_embeds = nag_negative_pooled_prompt_embeds.to(device=device, dtype=dtype)
    elif negpip_active:
        from core.inference.negpip_processor import set_negpip_processors
        original_processors = set_negpip_processors(unet, negpip_token_weights, attention_type=attention_type)

    # Spectrum (Adaptive Spectral Feature Forecasting) acceleration -- see the txt2img
    # loop for details. Auto-disabled for unstable per-step conditioning / DEUS / very
    # few steps.
    spectrum = None
    spectrum_block_ctrl = None
    if spectrum_enable:
        _n_steps = len(timesteps)
        if _outpaint_resample_active:
            # OUTPAINT B2: revisited timesteps under time-travel resampling break
            # Spectrum's monotonic anchor/forecast assumptions (it assumes each
            # step index is visited at most once, in increasing order). Prefer
            # keeping resampling (the requested quality fix) and disabling
            # Spectrum instead, per the design doc's B2 SCHEDULER GATING.
            print("[Spectrum] requested but disabled (outpaint time-travel resampling is active; "
                  "revisited timesteps break Spectrum's monotonic anchor/forecast assumptions)")
            _add_generation_warning(
                "Spectrum acceleration disabled: incompatible with outpaint time-travel resampling "
                "(revisited timesteps break its monotonic anchor/forecast assumptions).",
                code="outpaint_resample_spectrum_disabled",
            )
        elif is_deus or has_controlnet or (prompt_embeds_callback is not None):
            print("[Spectrum] requested but disabled (prompt-editing / ControlNet / DEUS; "
                  "needs stable conditioning)")
        elif _n_steps < spectrum_warmup_steps + 3:
            print(f"[Spectrum] requested but disabled ({_n_steps} steps < warmup+3)")
        else:
            from core.inference.spectrum_forecaster import SpectrumForecaster
            _block = spectrum_feature_mode == "block"
            _max_cache = spectrum_max_cache if spectrum_max_cache > 0 else (6 if _block else 5)
            spectrum = SpectrumForecaster(
                _n_steps, num_basis=spectrum_m, lam=spectrum_lam, w=spectrum_w,
                w_decay=spectrum_w_decay,
                delta_cap=spectrum_delta_cap,
                warmup_steps=spectrum_warmup_steps, window_size=spectrum_window_size,
                flex_window=spectrum_flex_window, tail_fraction=spectrum_tail,
                max_cache=_max_cache,
            )
            if _block:
                from core.inference.spectrum_unet import SpectrumBlockController
                spectrum_block_ctrl = SpectrumBlockController(unet, spectrum, cache_branch=spectrum_cache_branch)
                print(f"[Spectrum] enabled (inpaint, block mode): {len(spectrum.anchors)}/{_n_steps} "
                      f"deep-feature passes, cache_branch={spectrum_block_ctrl.branch}/{spectrum_block_ctrl.n_down}")
            else:
                print(f"[Spectrum] enabled (inpaint, output mode): {len(spectrum.anchors)}/{_n_steps} actual passes")

    # Style transfer yields to NAG / ControlNet / Spectrum -- see the txt2img loop for
    # the full rationale (incompatible batch layouts / stale spectrum state).
    # TODO: ControlNet(structure)+style(appearance) is a desirable combo; supporting
    # it needs per-pass batch-1 residual recompute -- future enhancement.
    if style_active and (nag_active or has_controlnet or spectrum is not None):
        print("[CustomSampling] Style transfer disabled: not compatible with NAG / "
              "ControlNet / Spectrum in this version")
        _add_generation_warning(
            "Style transfer disabled: not compatible with NAG / ControlNet / Spectrum in this version.",
            code="style_incompatible",
        )
        style_active = False

    # Multi-reference style transfer has the exact same batch-layout incompatibility
    # (separate per-ref capture forwards + a 2-Pass CFG cond/uncond split).
    if style_refs_active and (nag_active or has_controlnet or spectrum is not None):
        print("[CustomSampling] Multi-ref style transfer disabled: not compatible with NAG / "
              "ControlNet / Spectrum in this version")
        _add_generation_warning(
            "Style transfer disabled: not compatible with NAG / ControlNet / Spectrum in this version.",
            code="style_incompatible",
        )
        style_refs_active = False

    # FBCache: dynamic per-step deep-block caching, mutually exclusive with Spectrum
    # and auto-disabled for unstable conditioning (prompt editing / ControlNet / DEUS),
    # and also for style transfer (its capture forward would pollute the cache; see
    # the txt2img loop for details).
    fbcache_ctrl = None
    if fbcache_enable:
        if _outpaint_resample_active:
            # OUTPAINT B2: same rationale as the Spectrum gate above -- revisited
            # timesteps break FBCache's dynamic cache-hit/reuse assumptions
            # (it also assumes each step index is visited at most once).
            print("[FBCache] requested but disabled (outpaint time-travel resampling is active; "
                  "revisited timesteps break FBCache's cache-hit/reuse assumptions)")
            _add_generation_warning(
                "FBCache disabled: incompatible with outpaint time-travel resampling "
                "(revisited timesteps break its cache-hit/reuse assumptions).",
                code="outpaint_resample_fbcache_disabled",
            )
        elif spectrum_block_ctrl is not None or spectrum is not None:
            print("[FBCache] requested but disabled (Spectrum is active; mutually exclusive)")
        elif is_deus or has_controlnet or (prompt_embeds_callback is not None) or style_active or style_refs_active:
            print("[FBCache] requested but disabled (prompt-editing / ControlNet / DEUS / "
                  "style transfer; needs stable conditioning)")
        else:
            from core.inference.fbcache_unet import build_unet_fbcache_controller
            fbcache_ctrl = build_unet_fbcache_controller(
                unet,
                {
                    "fbcache_enable": fbcache_enable,
                    "fbcache_threshold": fbcache_threshold,
                    "fbcache_warmup_steps": fbcache_warmup_steps,
                    "fbcache_cache_branch": fbcache_cache_branch,
                },
                label="inpaint",
            )
    print(f"[CustomSampling] Starting inpaint loop with {len(timesteps)} steps")

    # Get sigma_max for dynamic CFG scheduling
    sigma_max = 0.0
    if hasattr(scheduler, 'sigmas') and len(scheduler.sigmas) > 0:
        sigma_max = float(scheduler.sigmas[0].item())
    print(f"[CustomSampling] Sigma max: {sigma_max}, CFG schedule: {cfg_schedule_type}")

    # Track previous SNR for SNR-based adaptive CFG
    previous_snr = None

    # Debug flag for first iteration logging (used throughout the loop)
    first_iteration_debug = True

    # Send initial noise preview (step 0) before denoising loop starts.
    # Total is len(_outpaint_visit_schedule), not len(timesteps): identical to
    # len(timesteps) off the OUTPAINT B2 resample path (see
    # _build_outpaint_resample_schedule's r<=1 early return), but reflects the
    # true (higher) NFE when resampling is active -- so the progress bar's
    # total always matches the number of denoise-step visits actually run.
    if progress_callback is not None:
        print(f"[CustomSampling] Sending initial noise preview (step 0)")
        progress_callback(-1, len(_outpaint_visit_schedule), latents, cfg_metrics=None)

    # ---- In-loop hard-flatten setup (SD1.5/SDXL, opt-in) -----------------------
    _flatten_inject_steps, _flatten_vae_shift = _setup_inloop_flatten(
        pipeline, timesteps, spectrum, fbcache_ctrl,
        flatten_in_loop, flatten_in_loop_last_steps, flatten_in_loop_min_region)

    # OUTPAINT B2: iterate the precomputed VISIT schedule (see
    # _build_outpaint_resample_schedule) instead of a plain enumerate(timesteps).
    # `visit_idx` is a MONOTONIC running counter (used only for progress-total
    # reporting below); `i` is the LOGICAL diffusion index (used for every
    # per-step schedule lookup -- sigma, cfg schedule, prompt-edit callback,
    # controlnet fraction, spectrum/fbcache, style progress, etc. -- exactly as
    # the plain `for i, t in enumerate(timesteps)` loop it replaces). Off the
    # resample path (outpaint_resample_count<=1, non-outpaint calls, or an
    # incompatible scheduler) `_outpaint_visit_schedule == [(0, False), (1,
    # False), ..., (T-1, False)]`, so `visit_idx == i` always and this loop is
    # ITERATION-ORDER-IDENTICAL to the original `enumerate(timesteps)`.
    for visit_idx, (i, is_forward_jump) in enumerate(_outpaint_visit_schedule):
        t = timesteps[i]

        # OUTPAINT B2: time-travel jump. `is_forward_jump` marks the first
        # visit of a re-denoise cycle -- re-noise the WHOLE latent (keep +
        # generate together, no mask special-casing -- see
        # _outpaint_resample_jump) from its current level (index i +
        # outpaint_jump_length, where the previous visit just landed) back UP
        # to the level at index i. Never true off the resample path.
        if is_forward_jump:
            latents = _outpaint_resample_jump(
                scheduler, latents, timesteps, hi_index=i, lo_index=i + outpaint_jump_length,
                generator=step_generator if step_generator is not None else generator,
            )

        # OUTPAINT B2: EulerDiscreteScheduler/EulerAncestralDiscreteScheduler
        # cache an internal `_step_index` that `scale_model_input`/`step` read
        # `self.sigmas[self.step_index]` from -- it is set from the passed
        # `timestep` ONLY on the very first ever call (`_init_step_index`,
        # gated on `self.step_index is None`); every call after that just
        # blindly increments it by 1, IGNORING the `timestep`/`t` argument.
        # A backward jump therefore leaves it silently stale (still counting
        # up past the jump) unless forced back in sync with the LOGICAL index
        # `i` here -- a true no-op on every visit that isn't preceded by a
        # jump (matches whatever the auto-increment already holds). DDIM/DDPM
        # (the other _outpaint_resample_active-eligible family) derive
        # everything from `t` directly every call and have no such counter,
        # so this is intentionally scoped to only the sigma-scale family, and
        # only while resampling is genuinely active (never touches a B1-only
        # run with e.g. DPM++/UniPC, which are excluded from resampling
        # entirely by the scheduler gate above but still use their OWN
        # `_step_index` semantics for B1 alone -- untouched here).
        if _outpaint_resample_active and type(scheduler).__name__ in _OUTPAINT_RESAMPLE_SIGMA_SCHEDULERS:
            scheduler._step_index = i

        # Check for cancellation (only in inference context, not training)
        try:
            from core.pipeline import pipeline_manager
            if pipeline_manager.cancel_requested:
                print("[CustomSampling] Generation cancelled by user")
                raise RuntimeError("Generation cancelled by user")
        except (ImportError, AttributeError):
            # pipeline_manager not available (e.g., in training subprocess)
            pass

        # Check if NAG should be deactivated based on sigma threshold
        if nag_active and nag_sigma_end > 0.0:
            if hasattr(scheduler, 'sigmas') and i < len(scheduler.sigmas):
                current_sigma = float(scheduler.sigmas[i].item())
                if current_sigma < nag_sigma_end:
                    print(f"[CustomSampling] Deactivating NAG at step {i} (sigma={current_sigma:.4f} < {nag_sigma_end})")
                    from core.inference.nag_processor import restore_original_processors
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

        # Optimize: skip unconditional pass if guidance_scale ~= 1.0 and neither NAG
        # nor NegPip is active. NegPip needs the [negative, positive] batch so its
        # per-context V weights align (and negative-prompt double-negation works).
        do_classifier_free_guidance = (abs(current_guidance_scale - 1.0) > 1e-5) or nag_active or negpip_active

        # Prepare latent input based on CFG mode
        if nag_active:
            # NAG mode: Use batch approach (legacy, backward compatible)
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
            prompt_embeds_input = torch.cat([nag_negative_prompt_embeds, current_prompt_embeds])

        elif do_classifier_free_guidance:
            if is_deus or style_active or style_refs_active:
                # DEUS (variable seq-len embeds) or active multi-reference style
                # transfer: prepare a single (batch=1) latent (see the txt2img loop's
                # style branch for the rationale; style_refs_active requires 2+
                # references, so this never affects the single-ref style_active path).
                latent_model_input = scheduler.scale_model_input(latents, t)

                # Only concatenate mask and masked image for inpaint-specific UNets
                if is_inpaint_unet:
                    masked_image_latents = image_latents * (1 - mask_latent)
                    latent_model_input = torch.cat([latent_model_input, mask_latent, masked_image_latents], dim=1)

                prompt_embeds_input = None
            else:
                # Standard CFG (SDXL/SD1.5): Use batch approach [negative, positive] (batch=2)
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # Only concatenate mask and masked image for inpaint-specific UNets
                if is_inpaint_unet:
                    # Use original clean image latents, masked to show only non-inpaint regions
                    masked_image_latents = image_latents * (1 - mask_latent)
                    latent_model_input = torch.cat([latent_model_input, mask_latent.repeat(2, 1, 1, 1), masked_image_latents.repeat(2, 1, 1, 1)], dim=1)

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
            original_size = _resolve_sdxl_original_size(original_height, original_width, original_size_w, original_size_h, original_size_scale)
            crops_coords_top_left = (0, 0)
            target_size = (original_height, original_width)

            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)

            if nag_active or do_classifier_free_guidance:
                # NAG mode or standard CFG (SDXL/SD1.5): Use batch approach
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

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids
                }

            else:
                # No CFG: Use single-batch
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
            # Use autocast for FP8 or UINT quantized U-Net (required for FP16 activations)
            is_uint_quantized = hasattr(unet, '_is_uint_quantized') and unet._is_uint_quantized
            use_autocast = unet_dtype == torch.float8_e4m3fn or unet_dtype == torch.float8_e5m2 or is_uint_quantized

            if is_deus and do_classifier_free_guidance:
                # DEUS: 2-Pass CFG - separate U-Net calls for negative and positive embeddings

                # ============================================================
                # DEBUG: First iteration details (DEUS 2-Pass CFG)
                # ============================================================
                if first_iteration_debug:
                    print(f"\n[CustomSampling] [Debug] ========== FIRST DENOISING ITERATION (DEUS 2-Pass CFG) ==========")
                    print(f"[CustomSampling] [Debug] timestep (t): {t.item()}")
                    print(f"[CustomSampling] [Debug] latent_model_input shape: {latent_model_input.shape}, dtype: {latent_model_input.dtype}")
                    print(f"[CustomSampling] [Debug] latent_model_input min: {latent_model_input.min().item():.4f}, max: {latent_model_input.max().item():.4f}, mean: {latent_model_input.mean().item():.4f}")
                    print(f"[CustomSampling] [Debug] negative_prompt_embeds shape: {current_negative_prompt_embeds.shape}, dtype: {current_negative_prompt_embeds.dtype}")
                    print(f"[CustomSampling] [Debug] positive_prompt_embeds shape: {current_prompt_embeds.shape}, dtype: {current_prompt_embeds.dtype}")

                # Pass 1: Unconditional (negative) prediction
                unet_kwargs_uncond = {
                    "encoder_hidden_states": current_negative_prompt_embeds,
                }
                if down_block_res_samples is not None:
                    unet_kwargs_uncond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_uncond["mid_block_additional_residual"] = mid_block_res_sample

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample
                else:
                    noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample

                # Pass 2: Conditional (positive) prediction
                unet_kwargs_cond = {
                    "encoder_hidden_states": current_prompt_embeds,
                }
                if down_block_res_samples is not None:
                    unet_kwargs_cond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_cond["mid_block_additional_residual"] = mid_block_res_sample

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample
                else:
                    noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample
            elif style_active and do_classifier_free_guidance:
                # Active style transfer: 2-Pass CFG (separate uncond/cond U-Net calls),
                # so the reference-style KV injection can be isolated to ONLY the
                # conditional (positive) pass -- the unconditional pass is always run
                # with no style context (untouched), exactly like the txt2img wiring.
                from core.inference.reference_style import StyleContext
                from core.inference.attention_processors import set_style_context

                def _slice_added_cond_kwargs(row: int):
                    if not (is_sdxl and added_cond_kwargs):
                        return None
                    text_embeds = added_cond_kwargs.get("text_embeds")
                    return {
                        "text_embeds": text_embeds[row:row + 1] if text_embeds is not None else None,
                        "time_ids": added_cond_kwargs["time_ids"][row:row + 1],
                    }

                # Pass 1: Unconditional (negative) prediction -- no style context.
                set_style_context(unet, None)
                unet_kwargs_uncond = {"encoder_hidden_states": current_negative_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_uncond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_uncond["mid_block_additional_residual"] = mid_block_res_sample
                uncond_added_cond_kwargs = _slice_added_cond_kwargs(0)
                if uncond_added_cond_kwargs is not None:
                    unet_kwargs_uncond["added_cond_kwargs"] = uncond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample
                else:
                    noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample

                # Pass 2: Conditional (positive) prediction -- style capture + inject,
                # only when this step falls within the style config's active range.
                cond_added_cond_kwargs = _slice_added_cond_kwargs(1)
                if style_cfg.is_step_active(i, num_inference_steps):
                    ref_t = scheduler.add_noise(style_ref_x0, style_eps_ref, t.unsqueeze(0))
                    ref_t_scaled = scheduler.scale_model_input(ref_t, t)
                    progress = style_cfg.step_progress(i, num_inference_steps)

                    ref_unet_kwargs = {"encoder_hidden_states": current_prompt_embeds}
                    if cond_added_cond_kwargs is not None:
                        ref_unet_kwargs["added_cond_kwargs"] = cond_added_cond_kwargs

                    capture_ctx = StyleContext(mode="capture", config=style_cfg, progress=progress)
                    set_style_context(unet, capture_ctx)
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)
                    else:
                        unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)

                    inject_ctx = StyleContext(mode="inject", config=style_cfg, store=capture_ctx.store, progress=progress)
                    set_style_context(unet, inject_ctx)

                unet_kwargs_cond = {"encoder_hidden_states": current_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_cond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_cond["mid_block_additional_residual"] = mid_block_res_sample
                if cond_added_cond_kwargs is not None:
                    unet_kwargs_cond["added_cond_kwargs"] = cond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample
                else:
                    noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample

                # --- CFG-decoupled style guidance (SDXL/SD1.5 prototype) ---
                # Disabled by default (style_guidance_scale is None/<=0): this block
                # is skipped entirely and noise_pred_text stays exactly the styled
                # cond pred above (cond_s) -- byte-identical to before this feature.
                # Enabled (>0) AND this step actually injected style (is_step_active
                # above, same gate as the capture/inject pass): run a 3rd forward --
                # SAME unet_kwargs_cond (same encoder_hidden_states/residuals/
                # added_cond_kwargs as the styled pass) but with style context
                # cleared -- to get the cond prediction WITHOUT style (cond_ns), then
                # rewrite noise_pred_text so the UNCHANGED shared CFG combine
                # (noise_pred = uncond + cfg*(text - uncond)) reproduces the
                # style-guidance target:
                #   uncond + cfg*(cond_ns - uncond) + lambda*(cond_s - cond_ns)
                # Algebra: let text' = cond_ns + (lambda/cfg)*(cond_s - cond_ns).
                # Substituting into the shared combine:
                #   uncond + cfg*(text' - uncond)
                # = uncond + cfg*(cond_ns - uncond) + cfg*(lambda/cfg)*(cond_s-cond_ns)
                # = uncond + cfg*(cond_ns - uncond) + lambda*(cond_s - cond_ns)
                # which is exactly the target above -- so assigning
                # noise_pred_text = text' lets the untouched shared combine line
                # produce style guidance decoupled from cfg. cfg is guarded (>1e-6)
                # even though do_classifier_free_guidance guarantees cfg>1 here; if
                # it were ever ~0 we skip the rewrite and keep noise_pred_text=cond_s.
                if (
                    style_cfg.style_guidance_scale is not None
                    and style_cfg.style_guidance_scale > 0
                    and style_cfg.is_step_active(i, num_inference_steps)
                ):
                    cond_s = noise_pred_text
                    set_style_context(unet, None)
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            cond_ns = unet(latent_model_input, t, **unet_kwargs_cond).sample
                    else:
                        cond_ns = unet(latent_model_input, t, **unet_kwargs_cond).sample
                    cfg = current_guidance_scale
                    lam = style_cfg.style_guidance_scale
                    if cfg > 1e-6:
                        noise_pred_text = cond_ns + (lam / cfg) * (cond_s - cond_ns)

                set_style_context(unet, None)

                # noise_pred_uncond and noise_pred_text are already separate (no chunk needed)
            elif style_refs_active and do_classifier_free_guidance:
                # Multi-reference (N>1) style transfer: 2-Pass CFG identical to the
                # single-ref branch above, but the conditional pass runs ONE capture
                # forward PER reference (each with its OWN StyleTransferConfig --
                # block_range, strengths, freq curve, step gating -- fully
                # independent) into its own store, then a single multi-ref inject via
                # inject_kv_multi (see attention_processors.UnifiedAttnProcessor).
                # style_refs_active requires 2+ entries (see its definition above),
                # so this branch never fires for a single reference -- that case is
                # always routed through style_active above, unchanged.
                from core.inference.reference_style import StyleContext
                from core.inference.attention_processors import set_style_context

                def _slice_added_cond_kwargs(row: int):
                    if not (is_sdxl and added_cond_kwargs):
                        return None
                    text_embeds = added_cond_kwargs.get("text_embeds")
                    return {
                        "text_embeds": text_embeds[row:row + 1] if text_embeds is not None else None,
                        "time_ids": added_cond_kwargs["time_ids"][row:row + 1],
                    }

                # Pass 1: Unconditional (negative) prediction -- no style context.
                set_style_context(unet, None)
                unet_kwargs_uncond = {"encoder_hidden_states": current_negative_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_uncond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_uncond["mid_block_additional_residual"] = mid_block_res_sample
                uncond_added_cond_kwargs = _slice_added_cond_kwargs(0)
                if uncond_added_cond_kwargs is not None:
                    unet_kwargs_uncond["added_cond_kwargs"] = uncond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample
                else:
                    noise_pred_uncond = unet(latent_model_input, t, **unet_kwargs_uncond).sample

                # Pass 2: Conditional (positive) prediction -- one capture forward PER
                # active reference (skipping refs not step-active this step, mirroring
                # the single-ref "not is_step_active -> no injection" case), then a
                # single multi-ref inject.
                cond_added_cond_kwargs = _slice_added_cond_kwargs(1)
                active_style_refs = []
                for _sref_cfg, _sref_x0, _sref_eps in style_refs:
                    if not _sref_cfg.is_step_active(i, num_inference_steps):
                        continue
                    ref_t = scheduler.add_noise(_sref_x0, _sref_eps, t.unsqueeze(0))
                    ref_t_scaled = scheduler.scale_model_input(ref_t, t)
                    ref_progress = _sref_cfg.step_progress(i, num_inference_steps)

                    ref_unet_kwargs = {"encoder_hidden_states": current_prompt_embeds}
                    if cond_added_cond_kwargs is not None:
                        ref_unet_kwargs["added_cond_kwargs"] = cond_added_cond_kwargs

                    ref_capture_ctx = StyleContext(mode="capture", config=_sref_cfg, progress=ref_progress)
                    set_style_context(unet, ref_capture_ctx)
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)
                    else:
                        unet(ref_t_scaled.to(dtype), t, **ref_unet_kwargs)

                    active_style_refs.append((ref_capture_ctx.store, _sref_cfg))

                if active_style_refs:
                    overall_progress = active_style_refs[0][1].step_progress(i, num_inference_steps)
                    inject_ctx = StyleContext(
                        mode="inject", config=active_style_refs[0][1], refs=active_style_refs,
                        combine_mode=style_combine_mode, progress=overall_progress,
                    )
                    set_style_context(unet, inject_ctx)
                # else: no reference active this step -- context stays None (set by
                # Pass 1 above), matching the single-ref "not step-active" case.

                unet_kwargs_cond = {"encoder_hidden_states": current_prompt_embeds}
                if down_block_res_samples is not None:
                    unet_kwargs_cond["down_block_additional_residuals"] = down_block_res_samples
                if mid_block_res_sample is not None:
                    unet_kwargs_cond["mid_block_additional_residual"] = mid_block_res_sample
                if cond_added_cond_kwargs is not None:
                    unet_kwargs_cond["added_cond_kwargs"] = cond_added_cond_kwargs

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample
                else:
                    noise_pred_text = unet(latent_model_input, t, **unet_kwargs_cond).sample

                set_style_context(unet, None)

                # noise_pred_uncond and noise_pred_text are already separate (no chunk needed)
            elif spectrum is not None and spectrum_block_ctrl is None and not spectrum.is_anchor(i):
                # Spectrum output (black-box) skip step: forecast the raw U-Net output
                # (Eq.14) instead of running the forward. NAG/NegPip effects are baked
                # into the recorded anchor outputs, so they carry through the forecast.
                noise_pred = spectrum.forecast(i)
            else:
                # Standard batch approach: NAG mode, Standard CFG (SDXL/SD1.5), or No CFG
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

                # ============================================================
                # DEBUG: First iteration details (for comparison with training)
                # ============================================================
                if first_iteration_debug:
                    print(f"\n[CustomSampling] [Debug] ========== FIRST DENOISING ITERATION ==========")
                    print(f"[CustomSampling] [Debug] timestep (t): {t.item()}")
                    print(f"[CustomSampling] [Debug] latent_model_input shape: {latent_model_input.shape}, dtype: {latent_model_input.dtype}")
                    print(f"[CustomSampling] [Debug] latent_model_input min: {latent_model_input.min().item():.4f}, max: {latent_model_input.max().item():.4f}, mean: {latent_model_input.mean().item():.4f}")
                    print(f"[CustomSampling] [Debug] prompt_embeds_input shape: {prompt_embeds_input.shape}, dtype: {prompt_embeds_input.dtype}")

                # Spectrum block mode: deep blocks are captured (anchor) or forecast
                # (skip) inside the U-Net via wrappers installed for this single call.
                # FBCache block mode: deep blocks are reused (hit) or captured (miss)
                # dynamically per step via wrappers installed for this single call.
                if spectrum_block_ctrl is not None:
                    spectrum_block_ctrl.begin_step(i)
                if fbcache_ctrl is not None:
                    fbcache_ctrl.begin_step(i)
                try:
                    if use_autocast:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            noise_pred = unet(
                                latent_model_input,
                                t,
                                **unet_kwargs
                            ).sample
                    else:
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            **unet_kwargs
                        ).sample
                finally:
                    if spectrum_block_ctrl is not None:
                        spectrum_block_ctrl.end_step()
                    if fbcache_ctrl is not None:
                        fbcache_ctrl.end_step()

                # Spectrum output mode: record this actual-pass output and refit.
                if spectrum is not None and spectrum_block_ctrl is None:
                    spectrum.record(i, noise_pred)

        # Perform guidance with CFG
        if do_classifier_free_guidance:
            if is_deus or style_active or style_refs_active:
                # DEUS / active multi-reference style transfer: noise_pred_uncond and
                # noise_pred_text are already separate (from the 2-Pass CFG block
                # above), for both single-ref (style_active) and multi-ref
                # (style_refs_active). style_active was previously MISSING from this
                # img2img/inpaint gate (a pre-existing single-ref crash: the else
                # branch chunks a batch-2 noise_pred that the 2-pass block never set).
                pass  # Variables already set in the 2-Pass CFG block
            else:
                # NAG mode or Standard CFG: noise_pred has [negative, positive] batches
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

            # ============================================================
            # DEBUG: Noise prediction AFTER CFG (for comparison with training)
            # ============================================================
            if first_iteration_debug:
                print(f"[CustomSampling] [Debug] noise_pred AFTER CFG shape: {noise_pred.shape}, dtype: {noise_pred.dtype}")
                print(f"[CustomSampling] [Debug] noise_pred AFTER CFG min: {noise_pred.min().item():.4f}, max: {noise_pred.max().item():.4f}, mean: {noise_pred.mean().item():.4f}")

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

        # ============================================================
        # OUTPAINT B1: trajectory-consistent x0-space projection injection
        # (scratchpad/outpaint_continuity_design.md section "B1"). REPLACES
        # the post-step keep re-injection further below with a PRE-step
        # projection of the CFG-adjusted model output, so the keep-region
        # constraint rides the SAME sampler transition (ancestral noise draw,
        # multistep solver history, etc.) as the generate region instead of
        # being pinned to an independent forward marginal
        # (scheduler.add_noise(image_latents, noise, t_{i+1})) that is
        # off-trajectory relative to the gen region -- the seam-line source.
        # A no-op (nothing in this block runs) when outpaint_noise_init is
        # False, so normal inpaint is byte-identical.
        # ============================================================
        if outpaint_noise_init and not is_inpaint_unet:
            predict_x0, to_model_output = _outpaint_x0_transform(scheduler, latents, t, i)
            x0_hat = predict_x0(noise_pred)
            # Project the known constraint: keep region = clean image latents,
            # generate region = the model's own x0 estimate.
            x0_proj = (1 - mask_latent) * image_latents + mask_latent * x0_hat

            if _outpaint_collar_weight_map is not None:
                # Low-frequency boundary color proximal (design doc section
                # 2): active mid/late schedule only (progress >= 20%) -- at
                # high noise the x0 estimate is too unreliable for a
                # meaningful low-freq correction.
                _outpaint_progress = (t_start + i) / num_inference_steps
                if _outpaint_progress >= 0.2:
                    x0_proj = _outpaint_apply_boundary_color(
                        x0_proj, _outpaint_target_lowfreq, _outpaint_collar_weight_map,
                        outpaint_boundary_color_strength,
                    )

            # Convert back to the model-output space scheduler.step expects
            # (exact inverse of predict_x0 for this scheduler/prediction_type).
            noise_pred = to_model_output(x0_proj)

        # Pass step_generator to ensure reproducibility with stochastic samplers (e.g., Euler a)
        step_output = scheduler.step(noise_pred, t, latents, generator=step_generator)
        latents = step_output.prev_sample

        # Get predicted x0 (original sample) if available from scheduler
        # Use .detach().clone() to disconnect from computation graph and ensure contiguous memory
        pred_original_sample = getattr(step_output, 'pred_original_sample', None)
        if pred_original_sample is not None:
            pred_original_sample = pred_original_sample.detach().clone()

        # Reference Guide blending (inpaint) - applied before mask blending
        if ref_guides:
            ref_frac = (t_start + i) / num_inference_steps
            latents, pred_original_sample = apply_reference_guide_blend(
                latents, pred_original_sample, ref_guides, ref_frac, i, timesteps, scheduler
            )

        # In-loop hard-flatten of the generated background (SD1.5/SDXL, opt-in).
        # Applied to the running latents before mask blending, so the pristine
        # original (unmasked) regions are never altered.
        if flatten_in_loop and i in _flatten_inject_steps:
            latents, _ = inloop_hard_flatten_step(
                pipeline, latents, pred_original_sample,
                flatten_in_loop_min_region, _flatten_vae_shift)

        # Apply mask blending ONLY for 4-channel UNets (regular models)
        # 9-channel inpaint UNets handle masking internally via concatenation
        if not is_inpaint_unet:
            if outpaint_noise_init:
                # OUTPAINT B1: the keep-region constraint was already applied
                # PRE-step above (x0-space projection injection), so the old
                # post-step add_noise-and-overwrite is gated OFF here -- it is
                # exactly the off-trajectory replacement B1 removes. Only a
                # single hard overwrite remains, on the FINAL VISIT (the
                # OUTPAINT B2 schedule's last entry, not necessarily
                # `i == len(timesteps) - 1` under resampling -- though in
                # practice it always is, since the band is capped below
                # `len(timesteps)`; using the visit-schedule position is the
                # robust check regardless of band placement), as a latent
                # anchor (belt-and-suspenders; the pixel-exact guarantee is
                # the unconditional final paste in outpaint_utils regardless
                # of this).
                if visit_idx == len(_outpaint_visit_schedule) - 1:
                    latents = (1 - mask_latent) * image_latents + mask_latent * latents
                if pred_original_sample is not None:
                    pred_original_sample = (1 - mask_latent) * image_latents + mask_latent * pred_original_sample
            else:
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

                # Apply same mask blending to pred_original_sample for consistent x0 preview
                # Without this, the preview shows unblended generation (incorrect outside mask area)
                if pred_original_sample is not None:
                    pred_original_sample = (1 - mask_latent) * image_latents + mask_latent * pred_original_sample

        # Reset debug flag after first iteration
        if first_iteration_debug:
            first_iteration_debug = False

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

            # visit_idx/len(_outpaint_visit_schedule): a MONOTONIC progress
            # total (see the initial-preview call above) -- identical to
            # i/len(timesteps) off the OUTPAINT B2 resample path.
            progress_callback(visit_idx, len(_outpaint_visit_schedule), latents, cfg_metrics=cfg_metrics, pred_original_sample=pred_original_sample)

        if step_callback is not None:
            callback_kwargs = step_callback(pipeline, t_start + i, t, {"latents": latents})
            latents = callback_kwargs.get("latents", latents)

    # Clean up Reference Guide GPU tensors
    if ref_guides:
        for rg in ref_guides:
            del rg["clean_latent"], rg["noise"]
        ref_guides.clear()

    # Restore original processors if NAG or NegPip was active
    if original_processors is not None and (nag_active or negpip_active):
        from core.inference.nag_processor import restore_original_processors
        restore_original_processors(unet, original_processors)

    # ===== STAGE 3: VAE DECODE =====
    from core.vram_optimization import log_device_status, move_unet_to_cpu, move_vae_to_gpu, move_vae_to_cpu

    # Offload U-Net to CPU to free VRAM for VAE
    move_unet_to_cpu(pipeline)

    # loop_decode="none": latent passthrough -- see custom_sampling_loop's
    # Stage-3 site for the full rationale. NOTE: inpaint's pixel-space mask
    # compositing below (blending the generated region back into the original
    # image) never runs in this branch, since it needs a decoded image -- the
    # routes.py inpaint endpoint rejects loop_decode="none" up front so this
    # is defensive/unreachable via the API, kept only for symmetry with the
    # other two Stage-3 sites.
    if loop_decode == "none":
        print("[CustomSampling] loop_decode='none': skipping VAE decode (latent passthrough)")
        return latents

    from core.models.pid.pid_vae_wrapper import PidVaeWrapper
    _pid_active = isinstance(pipeline.vae, PidVaeWrapper)
    # loop_decode="cheap": see custom_sampling_loop's Stage-3 site.
    _use_real_vae_only = loop_decode == "cheap" and _pid_active
    # PiD stages its own net and does not use the held real VAE for the final
    # decode — don't stage that VAE to GPU when PiD is active, unless this
    # decode is routed to the real VAE instead (loop_decode="cheap").
    if not _pid_active or _use_real_vae_only:
        move_vae_to_gpu(pipeline)
    log_device_status("Ready for VAE decode (inpaint)", pipeline, vision_encoder=vision_encoder)

    # Decode latents to image
    _vae_shift = getattr(pipeline.vae.config, "shift_factor", None) or 0.0
    latents = latents / pipeline.vae.config.scaling_factor + _vae_shift
    if not _pid_active or _use_real_vae_only:
        # Convert latents to VAE dtype (fp16 VAE + fp32 latents); PiD re-normalizes
        # in fp32 internally so keep full precision for it.
        latents = latents.to(dtype=pipeline.vae.dtype)
    with torch.no_grad():
        if _pid_active and not _use_real_vae_only:
            # PiD override: see custom_sampling_loop's Stage-3 site for the
            # F1/F2 rationale (`latents` is already the pre-unscaled tensor
            # the wrapper re-normalizes internally).
            _pid_seed = generator.initial_seed() if generator is not None else 0
            _decode_cb = _make_pid_decode_progress(progress_callback)
            image = pipeline.vae.pid_final_decode(latents, seed=_pid_seed, progress_callback=_decode_cb).sample
        else:
            image = pipeline.vae.decode(latents, return_dict=True).sample

    # Free GPU latents before VAE offload
    del latents

    # VAE DC-drift correction (one extra reference decode, VAE still on GPU).
    # PiD has no encoder-based drift-correction path (accepted but not applied).
    _dc_bias = None
    if vae_drift_correction and not _pid_active and _drift_ref_latents is not None:
        _dc_bias = compute_vae_dc_bias(pipeline, _drift_ref_latents, _drift_input_mean, _vae_shift)

    # Offload VAE to CPU after decoding (skipped for PiD — its held VAE was never staged).
    if not _pid_active or _use_real_vae_only:
        move_vae_to_cpu(pipeline)

    # Scale from [-1, 1] to [0, 1] with robust nan/inf handling
    image = (image / 2 + 0.5)

    # Replace nan/inf with fallback values before clamping
    if torch.isnan(image).any() or torch.isinf(image).any():
        nan_count = torch.isnan(image).sum().item()
        inf_count = torch.isinf(image).sum().item()
        total_pixels = image.numel()
        print(f"[VAE Decode] Warning: {nan_count} nan, {inf_count} inf out of {total_pixels} pixels ({(nan_count + inf_count) / total_pixels * 100:.2f}%)")

        # Replace nan with gray (0.5), positive inf with white (1.0), negative inf with black (0.0)
        image = torch.where(torch.isnan(image), torch.tensor(0.5, device=image.device, dtype=image.dtype), image)
        image = torch.where(torch.isposinf(image), torch.tensor(1.0, device=image.device, dtype=image.dtype), image)
        image = torch.where(torch.isneginf(image), torch.tensor(0.0, device=image.device, dtype=image.dtype), image)

    image = image.clamp(0, 1)

    # Post-decode passes on the GENERATED content (before mask blending, so the
    # pristine original regions are never altered). Both zero-cost when disabled.
    if color_flatten_strength and color_flatten_strength > 0:
        from core.inference.color_flatten import flatten_chroma
        image = flatten_chroma(image, color_flatten_strength)
    if _dc_bias is not None:
        image = (image - _dc_bias.to(image.device, image.dtype)).clamp(0, 1)

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
    from core.extensions.controlnet_manager import controlnet_manager
    controlnet_manager.remove_lllite_patches()
    controlnet_manager.offload_controlnets_to_cpu()

    return Image.fromarray(image[0])
