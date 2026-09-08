"""Crop Decode Auxiliary Loss (Phase 3).

Integrates autograd-enabled context crop decoding with VaeLossBank
to provide an opt-in auxiliary pixel-space reconstruction loss during diffusion training.

Flow:
    model_pred -> predict_x0 -> denormalise -> crop rect -> decode_crop_with_context
                                                                │
    clean GT latents -> denormalise -> crop rect -> decode_crop_with_context (detached)
                                                                ▼
                                                    VaeLossBank (LPIPS/L1/MSE/YCbCr-DC)
                                                                ▼
                                                    aux_loss (scaled by weight)

Measures and logs:
    - crop_decode_loss: raw auxiliary loss value
    - crop_decode_grad_norm_ratio: ||grad_aux|| / ||grad_main|| for coefficient balancing (Phase 3-3)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.inference.context_tiled_decode import spatial_compression_of
from core.models.components.vae_registry import denormalize
from core.training.ops.crop_decode import decode_crop_with_context, make_crop_rect
from core.training.ops.x0_recovery import predict_x0, snr_band_mask
from core.training.vae.vae_losses import VaeLossBank


def parse_snr_range(snr_str: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse comma-separated SNR range 'min,max' or return (None, None) if empty."""
    if not snr_str or not snr_str.strip():
        return None, None
    parts = [p.strip() for p in snr_str.split(",") if p.strip()]
    if len(parts) == 1:
        return float(parts[0]), None
    if len(parts) >= 2:
        snr_min = float(parts[0]) if parts[0].lower() not in ("none", "") else None
        snr_max = float(parts[1]) if parts[1].lower() not in ("none", "") else None
        return snr_min, snr_max
    return None, None


class CropDecodeLossModule(nn.Module):
    """Auxiliary crop-decode loss module backed by VaeLossBank."""

    def __init__(
        self,
        vae: nn.Module,
        metric: str = "lpips",
        margin_cells: int = 16,
        out_cells: int = 32,
        snr_range: str = "",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.vae = vae
        self.metric = metric.lower().strip()
        self.margin_cells = max(0, int(margin_cells))
        self.out_cells = max(4, int(out_cells))
        self.snr_min, self.snr_max = parse_snr_range(snr_range)

        target_device = device if device is not None else next(vae.parameters()).device

        # Configure VaeLossBank for single metric evaluation
        bank_cfg: Dict[str, Any] = {
            "mse_weight": 1.0 if self.metric == "mse" else 0.0,
            "l1_weight": 1.0 if self.metric == "l1" else 0.0,
            "lpips_weight": 1.0 if self.metric == "lpips" else 0.0,
            "lpips_net": "vgg",
            "ycbcr_dc_weight": 1.0 if self.metric == "ycbcr_dc" else 0.0,
            "ycbcr_dc_y_weight": 1.0,
            "ycbcr_dc_chroma_weight": 0.5,
            "ycbcr_dc_eps": 1e-4,
            "pattern_weight": 0.0,
            "pattern_size": 8,
            "l_invented_weight": 0.0,
            "kl_weight": 0.0,
        }

        # If chosen metric is not recognized, fallback to lpips
        if all(bank_cfg[k] == 0.0 for k in ("mse_weight", "l1_weight", "lpips_weight", "ycbcr_dc_weight")):
            bank_cfg["lpips_weight"] = 1.0

        self.loss_bank = VaeLossBank(bank_cfg, target_device, kl_enabled=False)

        # Freeze VAE and loss bank parameters
        for p in self.loss_bank.parameters():
            p.requires_grad = False
        for p in self.vae.parameters():
            p.requires_grad = False


def compute_crop_decode_loss(
    trainer: Any,
    model_pred: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    clean_latents: torch.Tensor,
    noise_process: str,
    prediction_target: str,
    noise_scheduler: Any,
    velocity_sign: Optional[str] = None,
    alphas_cumprod_cached: Optional[torch.Tensor] = None,
    predicted_latent: Optional[torch.Tensor] = None,
    main_loss: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], float]:
    """Compute crop decode auxiliary loss and log gradient norm diagnostics (Phase 3-2 & 3-3).

    Returns:
        (scaled_aux_loss_tensor, raw_aux_loss_float)
    """
    enable = getattr(trainer, "crop_decode_loss_enable", False)
    weight = getattr(trainer, "crop_decode_loss_weight", 0.0)
    if not enable or weight <= 0.0:
        return None, 0.0

    vae = getattr(trainer, "vae", None)
    if vae is None:
        return None, 0.0

    # Lazy initialization of module on trainer (or re-init if VAE instance changed via VAE swap)
    loss_module: Optional[CropDecodeLossModule] = getattr(trainer, "_crop_decode_loss_module", None)
    if loss_module is None or getattr(loss_module, "vae", None) is not vae:
        loss_module = CropDecodeLossModule(
            vae=vae,
            metric=getattr(trainer, "crop_decode_loss_metric", "lpips"),
            margin_cells=getattr(trainer, "crop_decode_loss_margin_cells", 16),
            out_cells=getattr(trainer, "crop_decode_loss_out_cells", 32),
            snr_range=getattr(trainer, "crop_decode_loss_snr_range", ""),
            device=trainer.device,
        )
        trainer._crop_decode_loss_module = loss_module

    # Ensure VAE parameters are on the trainer's execution device (handles swap_onthefly CPU offload)
    if hasattr(loss_module.vae, "parameters"):
        vae_p = next(loss_module.vae.parameters(), None)
        if vae_p is not None and vae_p.device != trainer.device:
            loss_module.vae.to(device=trainer.device, dtype=getattr(trainer, "vae_dtype", None))

    # SNR Band Filtering
    if loss_module.snr_min is not None or loss_module.snr_max is not None:
        mask = snr_band_mask(
            noise_process=noise_process,
            timesteps=timesteps,
            noise_scheduler=noise_scheduler,
            snr_min=loss_module.snr_min,
            snr_max=loss_module.snr_max,
            alphas_cumprod_cached=alphas_cumprod_cached,
        )
        if not mask.any():
            return None, 0.0
        # Subset batch to samples in band
        model_pred_sub = model_pred[mask]
        noisy_latents_sub = noisy_latents[mask]
        timesteps_sub = timesteps[mask]
        clean_latents_sub = clean_latents[mask]
        pred_x0_sub = predicted_latent[mask] if predicted_latent is not None else None
    else:
        model_pred_sub = model_pred
        noisy_latents_sub = noisy_latents
        timesteps_sub = timesteps
        clean_latents_sub = clean_latents
        pred_x0_sub = predicted_latent

    # Predict x0 if not supplied
    if pred_x0_sub is None:
        try:
            pred_x0_sub = predict_x0(
                noise_process=noise_process,
                prediction_target=prediction_target,
                noisy_latents=noisy_latents_sub,
                model_pred=model_pred_sub,
                timesteps=timesteps_sub,
                noise_scheduler=noise_scheduler,
                velocity_sign=velocity_sign,
            )
        except Exception as e:
            print(f"{trainer.log_prefix} [crop_decode_loss] predict_x0 failed: {e}")
            return None, 0.0

    # Ensure 4-D [B, C, H, W]
    if pred_x0_sub.ndim == 5 and pred_x0_sub.shape[2] == 1:
        pred_x0_sub = pred_x0_sub.squeeze(2)
    if clean_latents_sub.ndim == 5 and clean_latents_sub.shape[2] == 1:
        clean_latents_sub = clean_latents_sub.squeeze(2)

    # Every caller's latents are NORMALISED, so the decoder's own domain is one
    # denormalisation away -- cast first, then denormalise, as latent_space.decode
    # does. Whole-tensor, before the crop: a 2x2-packed domain (FLUX.2, Lens) is
    # not slice-invariant.
    spec = getattr(trainer, "wiring", None)
    vae_param = next(loss_module.vae.parameters(), None)
    cast = ({"device": vae_param.device, "dtype": vae_param.dtype}
            if vae_param is not None else {})
    pred_x0_sub = denormalize(pred_x0_sub.to(**cast), loss_module.vae, spec)
    with torch.no_grad():
        clean_latents_sub = denormalize(
            clean_latents_sub.detach().to(**cast), loss_module.vae, spec)

    lat_h, lat_w = pred_x0_sub.shape[-2], pred_x0_sub.shape[-1]
    out_c = loss_module.out_cells
    margin = loss_module.margin_cells

    c_h = min(out_c, lat_h)
    c_w = min(out_c, lat_w)

    # Random crop location (consistent across batch elements for efficient decode)
    max_y = max(0, lat_h - c_h)
    max_x = max(0, lat_w - c_w)
    y0 = int(torch.randint(0, max_y + 1, (1,)).item()) if max_y > 0 else 0
    x0 = int(torch.randint(0, max_x + 1, (1,)).item()) if max_x > 0 else 0
    y1 = y0 + c_h
    x1 = x0 + c_w
    rect = make_crop_rect(lat_h, lat_w, y0, y1, x0, x1, margin_cells=margin)
    scale = spatial_compression_of(loss_module.vae)

    # Decode predicted crop (differentiable into pred_x0_sub and model_pred_sub)
    pred_crop_rgb = decode_crop_with_context(loss_module.vae, pred_x0_sub, rect, scale=scale)

    # Decode target crop (detached GT)
    with torch.no_grad():
        gt_crop_rgb = decode_crop_with_context(loss_module.vae, clean_latents_sub, rect, scale=scale)

    # VAE Loss Bank evaluation
    bank_out = loss_module.loss_bank(pred_crop_rgb, gt_crop_rgb)
    if isinstance(bank_out, tuple):
        raw_aux_loss = bank_out[0]
    elif isinstance(bank_out, dict):
        raw_aux_loss = bank_out.get("total", bank_out.get(loss_module.metric, None))
    else:
        raw_aux_loss = bank_out

    if raw_aux_loss is None:
        raw_aux_loss = F.l1_loss(pred_crop_rgb, gt_crop_rgb)

    raw_val = float(raw_aux_loss.detach().item())
    if trainer is not None and hasattr(trainer, "log_extra_metric"):
        trainer.log_extra_metric("crop_decode_loss", raw_val)

    # Phase 3-3: Gradient Norm Ratio Diagnostic (||grad_aux|| / ||grad_main||)
    if main_loss is not None and hasattr(trainer, "log_extra_metric"):
        try:
            # Measure gradient norms with respect to model_pred_sub (retains backward graph)
            g_aux = torch.autograd.grad(
                raw_aux_loss, model_pred_sub,
                retain_graph=True, create_graph=False,
                allow_unused=True
            )[0]
            g_main = torch.autograd.grad(
                main_loss, model_pred_sub,
                retain_graph=True, create_graph=False,
                allow_unused=True
            )[0]

            if g_aux is not None and g_main is not None:
                norm_aux = float(torch.linalg.norm(g_aux.flatten()).item())
                norm_main = float(torch.linalg.norm(g_main.flatten()).item())
                grad_ratio = norm_aux / (norm_main + 1e-12)
                trainer.log_extra_metric("crop_decode_grad_norm_ratio", grad_ratio)
        except Exception:
            # Gradient probe is diagnostic only; non-fatal if autograd graph cannot bifurcate
            pass

    scaled_loss = raw_aux_loss * weight
    return scaled_loss, raw_val
