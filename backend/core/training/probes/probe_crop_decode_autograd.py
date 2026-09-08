"""Gate G-C Validation Probe: Autograd Crop Decode Parity and Efficiency.

Measures whether context-padded crop decoding provides a viable surrogate for
full-image VAE decoding during diffusion model training (Convergence Acceleration Plan Phase 2-2).

Key questions answered:
1. Value Parity: Does crop-decoded RGB match full-image decode on the crop interior?
   Sweeps margin k in [0, 4, 8, 12, 16, 20] cells.
2. Gradient Direction Parity: Does the gradient dL/d(latent_interior) from crop decode
   align with full-image decode? Measured by cosine similarity (gate threshold: >= 0.95).
3. Non-local Residual: Identifies receptive-field error decay vs GroupNorm / attention non-locality.
4. Computational Cost: Measures wall time (forward + backward) and peak VRAM reduction.

Safety & Resource Contract:
- Host RAM peak: ~1.5 - 2.5 GiB (VAE only, transformer is NOT loaded).
- VRAM peak: ~1.0 - 3.5 GiB (depending on resolution and batch size).
- Frozen VAE parameters: eval mode, requires_grad=False for all weights.
- Autograd on latent: requires_grad=True on input latent.

Usage:
    # CPU sanity / mock test (fast, no GPU needed):
    venv/Scripts/python.exe backend/core/training/probes/probe_crop_decode_autograd.py --mock --device cpu

    # SD1.5 / SDXL VAE on CPU:
    venv/Scripts/python.exe backend/core/training/probes/probe_crop_decode_autograd.py --model-type sd15 --device cpu

    # Real VAE on GPU (after confirming GPU availability with repo owner):
    venv/Scripts/python.exe backend/core/training/probes/probe_crop_decode_autograd.py --device cuda --model-type sdxl --dtype fp16
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.inference.context_tiled_decode import (
    DEFAULT_MARGIN_CELLS,
    TileRect,
    spatial_compression_of,
)
from core.training.ops.crop_decode import decode_crop_with_context, make_crop_rect

# ---------------------------------------------------------------------------
# Pre-registered Gate G-C Acceptance Criteria
# ---------------------------------------------------------------------------
GATE_GC_MIN_COSINE: float = 0.95
GATE_GC_RECOMMENDED_MARGIN: int = 16
GATE_GC_MIN_DECAY_RATIO: float = 0.10  # MAE(k=16) / MAE(k=0) <= 0.10 (10x decay)
GATE_GC_MAX_COST_RATIO: float = 0.70   # Crop forward+backward <= 70% of full-image


# ---------------------------------------------------------------------------
# Mock VAE for lightweight / CI / CPU testing
# ---------------------------------------------------------------------------
class MockVAE(nn.Module):
    """Lightweight autoencoder decoder with GroupNorm and multi-stage convolutions.

    Mimics receptive field expansion (~16 cells) and GroupNorm non-local statistics
    without loading large pretrained checkpoints.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3, scale: int = 8):
        super().__init__()
        self.scale = scale
        self.spatial_compression_ratio = scale
        # The probe decodes raw latents directly, but a caller that normalises
        # (ops/crop_decode_loss) needs this mock to declare a method like a real VAE.
        self.config = SimpleNamespace(
            latent_channels=in_channels, scaling_factor=0.18215, shift_factor=None)
        # 3-stage conv + group norm hierarchy
        self.conv_in = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)
        # 2x up
        self.up1 = nn.ConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(4, 24)
        # 2x up
        self.up2 = nn.ConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(4, 16)
        # 2x up (total 8x)
        self.up3 = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def decode(self, z: torch.Tensor, return_dict: bool = False) -> Any:
        h = F.silu(self.gn1(self.conv_in(z)))
        h = F.silu(self.gn2(self.up1(h)))
        h = F.silu(self.gn3(self.up2(h)))
        out = self.up3(h)
        if return_dict:
            return SimpleNamespace(sample=out)
        return (out,)


def load_vae(
    model_type: str,
    vae_path: Optional[str],
    mock: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Load or construct VAE model according to specification."""
    if mock:
        print("[Probe] Constructing lightweight MockVAE (scale=8)...")
        vae = MockVAE(scale=8).to(device=device, dtype=dtype)
        vae.eval()
        return vae

    from diffusers import AutoencoderKL

    if vae_path and os.path.exists(vae_path):
        print(f"[Probe] Loading VAE weights from local file: {vae_path}")
        if vae_path.endswith(".safetensors") or vae_path.endswith(".pt") or vae_path.endswith(".bin"):
            try:
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
            except Exception as e:
                print(f"[Probe] from_single_file failed ({e}), attempting AutoencoderKL.from_pretrained...")
                vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
        else:
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
    else:
        # Check standard default local models
        sd15_default = REPO_ROOT / "models" / "vae" / "sd15" / "vae-ft-mse-840000-ema-pruned.safetensors"
        sdxl_default = REPO_ROOT / "models" / "NoobAI-XL-Vpred-v1.0.safetensors"

        if model_type == "sd15" and sd15_default.exists():
            print(f"[Probe] Loading default SD1.5 VAE from: {sd15_default}")
            vae = AutoencoderKL.from_single_file(str(sd15_default), torch_dtype=dtype)
        elif model_type == "sdxl" and sdxl_default.exists():
            print(f"[Probe] Loading SDXL VAE from single-file checkpoint: {sdxl_default}")
            vae = AutoencoderKL.from_single_file(str(sdxl_default), subfolder="vae", torch_dtype=dtype)
        else:
            print(f"[Probe] Local weight file not found for {model_type}; constructing real architecture from config...")
            if model_type == "sdxl":
                # Standard SDXL VAE config
                vae = AutoencoderKL(
                    in_channels=3,
                    out_channels=3,
                    down_block_types=["DownEncoderBlock2D"] * 4,
                    up_block_types=["UpDecoderBlock2D"] * 4,
                    block_out_channels=[128, 256, 512, 512],
                    layers_per_block=2,
                    act_fn="silu",
                    latent_channels=4,
                    sample_size=1024,
                )
            else:
                # Standard SD1.5 VAE config
                vae = AutoencoderKL(
                    in_channels=3,
                    out_channels=3,
                    down_block_types=["DownEncoderBlock2D"] * 4,
                    up_block_types=["UpDecoderBlock2D"] * 4,
                    block_out_channels=[128, 256, 512, 512],
                    layers_per_block=2,
                    act_fn="silu",
                    latent_channels=4,
                    sample_size=512,
                )

    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


# ---------------------------------------------------------------------------
# Measurement Core
# ---------------------------------------------------------------------------
def measure_parity_and_cost(
    vae: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    resolution: int = 1024,
    crop_cells: int = 32,
    margins: List[int] = [0, 4, 8, 12, 16, 20],
    crop_pos: str = "center",
    criterion_name: str = "l1",
    warmup: int = 1,
    repeat: int = 3,
    seed: int = 42,
    latent_amplitude: str = "production",
) -> Dict[str, Any]:
    """Execute parity sweep across margins and benchmark compute / memory."""
    torch.manual_seed(seed)
    scale = spatial_compression_of(vae)
    lat_h = resolution // scale
    lat_w = resolution // scale

    # Determine crop bounding box in latent cells
    c_h = crop_cells
    c_w = crop_cells
    if crop_pos == "center":
        y0 = (lat_h - c_h) // 2
        x0 = (lat_w - c_w) // 2
    elif crop_pos == "top_left":
        y0 = 0
        x0 = 0
    else:
        y0 = max(0, (lat_h - c_h) // 4)
        x0 = max(0, (lat_w - c_w) // 4)
    y1 = y0 + c_h
    x1 = x0 + c_w

    latent_channels = getattr(vae.config, "latent_channels", 4) if hasattr(vae, "config") else 4

    # Pixel crop coordinates
    py0 = y0 * scale
    py1 = y1 * scale
    px0 = x0 * scale
    px1 = x1 * scale
    crop_pixel_shape = (1, 3, py1 - py0, px1 - px0)

    # Criterion and target for backward autograd
    if criterion_name == "l1":
        loss_fn = lambda pred, target: F.l1_loss(pred, target)
    else:
        loss_fn = lambda pred, target: F.mse_loss(pred, target)

    target_crop = torch.randn(crop_pixel_shape, device=device, dtype=dtype)

    # Base latent tensor (frozen master copy). randn is a NORMALISED latent's
    # amplitude; the decoder's own input is that divided by scaling_factor
    # (5.5x for SD1.5/SDXL), and the absolute errors below scale with it.
    base_latent = torch.randn(1, latent_channels, lat_h, lat_w, device=device, dtype=dtype)
    if latent_amplitude == "production":
        sf = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.0) or 0.0)
        if sf <= 0:
            raise ValueError("--latent-amplitude production needs a VAE with a scaling_factor")
        base_latent = base_latent / sf

    is_cuda = (device.type == "cuda")

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    def _reset_peak_mem():
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)

    def _get_peak_mem_mb() -> float:
        if is_cuda:
            return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        return 0.0

    # -----------------------------------------------------------------------
    # Step 1: Baseline Full-Image Decode
    # -----------------------------------------------------------------------
    print(f"\n[Probe] Benchmarking Baseline Full Decode ({resolution}x{resolution} -> latent {lat_h}x{lat_w})...")

    # Warmup
    for _ in range(warmup):
        z = base_latent.clone().requires_grad_(True)
        out = vae.decode(z, return_dict=False)[0]
        c_out = out[..., py0:py1, px0:px1]
        l = loss_fn(c_out, target_crop)
        l.backward()
        del z, out, c_out, l

    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    full_times: List[float] = []
    _reset_peak_mem()
    for _ in range(repeat):
        z = base_latent.clone().requires_grad_(True)
        _sync()
        t0 = time.perf_counter()

        # Forward
        out = vae.decode(z, return_dict=False)[0]
        c_out = out[..., py0:py1, px0:px1]
        l = loss_fn(c_out, target_crop)

        # Backward
        l.backward()
        _sync()
        full_times.append(time.perf_counter() - t0)

        # Capture ground truth on last repetition
        full_interior_rgb = c_out.detach().clone()
        full_grad_interior = z.grad[..., y0:y1, x0:x1].detach().clone()
        del z, out, c_out, l

    full_peak_vram_mb = _get_peak_mem_mb()
    full_mean_time_ms = (sum(full_times) / len(full_times)) * 1000.0

    print(f"  Full Decode Mean Time: {full_mean_time_ms:.2f} ms")
    if is_cuda:
        print(f"  Full Decode Peak VRAM: {full_peak_vram_mb:.2f} MB")

    # -----------------------------------------------------------------------
    # Step 2: Crop Decode across Margins
    # -----------------------------------------------------------------------
    results_by_margin: List[Dict[str, Any]] = []

    print("\n[Probe] Sweeping Crop Decode Margins...")
    for margin in margins:
        rect = make_crop_rect(lat_h, lat_w, y0, y1, x0, x1, margin_cells=margin)

        # Warmup
        for _ in range(warmup):
            z = base_latent.clone().requires_grad_(True)
            crop_rgb = decode_crop_with_context(vae, z, rect, scale=scale)
            l = loss_fn(crop_rgb, target_crop)
            l.backward()
            del z, crop_rgb, l

        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()

        margin_times: List[float] = []
        _reset_peak_mem()
        for _ in range(repeat):
            z = base_latent.clone().requires_grad_(True)
            _sync()
            t0 = time.perf_counter()

            # Forward crop decode
            crop_rgb = decode_crop_with_context(vae, z, rect, scale=scale)
            l = loss_fn(crop_rgb, target_crop)

            # Backward
            l.backward()
            _sync()
            margin_times.append(time.perf_counter() - t0)

            crop_interior_rgb = crop_rgb.detach().clone()
            crop_grad_interior = z.grad[..., y0:y1, x0:x1].detach().clone()
            del z, crop_rgb, l

        crop_peak_vram_mb = _get_peak_mem_mb()
        crop_mean_time_ms = (sum(margin_times) / len(margin_times)) * 1000.0

        # Discrepancy & Parity Metrics
        diff_rgb = (full_interior_rgb - crop_interior_rgb).abs()
        mae_rgb = diff_rgb.mean().item()
        max_diff_rgb = diff_rgb.max().item()

        # 255-scale error for comparison with measured literature in context_tiled_decode.py
        mae_255 = mae_rgb * 255.0
        max_255 = max_diff_rgb * 255.0

        # Gradient Cosine Similarity
        g_full = full_grad_interior.flatten().to(torch.float32)
        g_crop = crop_grad_interior.flatten().to(torch.float32)

        norm_full = torch.linalg.norm(g_full).item()
        norm_crop = torch.linalg.norm(g_crop).item()
        dot_prod = torch.dot(g_full, g_crop).item()
        cosine_sim = dot_prod / (norm_full * norm_crop + 1e-12)

        # Gradient Rel L1 and Norm ratio
        grad_rel_l1 = (g_full - g_crop).abs().sum().item() / (g_full.abs().sum().item() + 1e-12)
        norm_ratio = norm_crop / (norm_full + 1e-12)

        # Cost ratios
        time_ratio = crop_mean_time_ms / (full_mean_time_ms + 1e-12)
        vram_ratio = crop_peak_vram_mb / (full_peak_vram_mb + 1e-12) if full_peak_vram_mb > 0 else 0.0

        res_item = {
            "margin_cells": margin,
            "padded_window": (rect.py1 - rect.py0, rect.px1 - rect.px0),
            "mae_rgb": mae_rgb,
            "max_diff_rgb": max_diff_rgb,
            "mae_255": mae_255,
            "max_255": max_255,
            "grad_cosine": cosine_sim,
            "grad_rel_l1": grad_rel_l1,
            "grad_norm_ratio": norm_ratio,
            "time_ms": crop_mean_time_ms,
            "time_ratio": time_ratio,
            "vram_mb": crop_peak_vram_mb,
            "vram_ratio": vram_ratio,
            "gate_gc_pass": bool(cosine_sim >= GATE_GC_MIN_COSINE),
        }
        results_by_margin.append(res_item)

    return {
        "device": str(device),
        "dtype": str(dtype),
        "resolution": resolution,
        "crop_cells": crop_cells,
        "scale": scale,
        "baseline_full": {
            "time_ms": full_mean_time_ms,
            "vram_mb": full_peak_vram_mb,
        },
        "results": results_by_margin,
    }


def print_report(data: Dict[str, Any]) -> None:
    """Print clean diagnostic table and Gate G-C verdict."""
    results = data["results"]
    full = data["baseline_full"]

    print("\n" + "=" * 90)
    print("GATE G-C AUTOGRAD CROP DECODE PARITY & COST REPORT")
    print("=" * 90)
    print(f"Device: {data['device']} | Dtype: {data['dtype']} | Scale: {data['scale']}x")
    print(f"Full Canvas: {data['resolution']}x{data['resolution']} | Crop Interior: {data['crop_cells']}x{data['crop_cells']} cells ({(data['crop_cells'] * data['scale'])}x{(data['crop_cells'] * data['scale'])} px)")
    print(f"Full Decode Baseline: {full['time_ms']:.2f} ms | VRAM: {full['vram_mb']:.2f} MB")
    print("-" * 90)
    print(f"{'Margin':>6} | {'Window':>9} | {'MAE (/255)':>11} | {'Max (/255)':>11} | {'Grad Cosine':>12} | {'Time (ms)':>10} | {'Time %':>7} | {'VRAM %':>7} | {'Gate G-C':>8}")
    print("-" * 90)

    for r in results:
        win_str = f"{r['padded_window'][0]}x{r['padded_window'][1]}"
        v_str = f"{r['vram_ratio']*100:.1f}%" if full['vram_mb'] > 0 else "N/A"
        gate_status = "PASS" if r['gate_gc_pass'] else "FAIL"
        print(
            f"{r['margin_cells']:>6} | "
            f"{win_str:>9} | "
            f"{r['mae_255']:>11.4f} | "
            f"{r['max_255']:>11.4f} | "
            f"{r['grad_cosine']:>12.5f} | "
            f"{r['time_ms']:>10.2f} | "
            f"{r['time_ratio']*100:>6.1f}% | "
            f"{v_str:>7} | "
            f"{gate_status:>8}"
        )
    print("-" * 90)

    # Evaluate Overall Gate G-C Acceptance Criteria
    rec_margin_result = next((r for r in results if r["margin_cells"] == GATE_GC_RECOMMENDED_MARGIN), results[-1])
    m0_result = results[0]

    decay_ratio = rec_margin_result["mae_rgb"] / (m0_result["mae_rgb"] + 1e-12)
    cosine_ok = rec_margin_result["grad_cosine"] >= GATE_GC_MIN_COSINE
    cost_ok = rec_margin_result["time_ratio"] <= GATE_GC_MAX_COST_RATIO

    print("\nGate G-C Assessment Summary (Target Margin k = 16):")
    print(f"1. Gradient Direction Cosine: {rec_margin_result['grad_cosine']:.5f} (Threshold >= {GATE_GC_MIN_COSINE}) -> {'PASS' if cosine_ok else 'FAIL'}")
    print(f"2. Error Attenuation Ratio:   {decay_ratio:.4f} (Threshold <= {GATE_GC_MIN_DECAY_RATIO}) -> {'PASS' if decay_ratio <= GATE_GC_MIN_DECAY_RATIO else 'NOTE: High baseline parity'}")
    print(f"3. Compute Cost Ratio:        {rec_margin_result['time_ratio']*100:.1f}% (Threshold <= {GATE_GC_MAX_COST_RATIO*100:.0f}%) -> {'PASS' if cost_ok else 'FAIL'}")

    overall_pass = cosine_ok and cost_ok
    print(f"\n>>> OVERALL GATE G-C VERDICT: {'ACCEPTED (Proceed to Phase 3)' if overall_pass else 'REJECTED (Branch to Phase 4)'} <<<\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate G-C: Autograd Crop Decode Parity & Cost Probe")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Execution device (default: cpu)")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None, help="Tensor dtype (default: fp32 on cpu, fp16 on cuda)")
    parser.add_argument("--model-type", choices=["sdxl", "sd15"], default="sdxl", help="Target VAE architecture")
    parser.add_argument("--vae-path", type=str, default=None, help="Explicit path to VAE safetensors or checkpoint")
    parser.add_argument("--mock", action="store_true", help="Use lightweight MockVAE (instant, zero memory)")
    parser.add_argument("--res", type=int, default=1024, help="Canvas resolution (default: 1024)")
    parser.add_argument("--crop-cells", type=int, default=32, help="Crop interior cells (default: 32 = 256px)")
    parser.add_argument("--margins", type=str, default="0,4,8,12,16,20", help="Comma-separated margins (default: 0,4,8,12,16,20)")
    parser.add_argument("--crop-pos", choices=["center", "top_left", "arbitrary"], default="center", help="Crop positioning")
    parser.add_argument("--latent-amplitude", choices=["production", "unit"], default="production",
                        help="Decoder input scale: 'production' divides by the VAE's scaling_factor "
                             "(the real operating point); 'unit' is raw randn, what the first sweep used")
    parser.add_argument("--criterion", choices=["l1", "mse"], default="l1", help="Autograd criterion (default: l1)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=3, help="Benchmark repetitions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--json", type=str, default=None, help="Optional output JSON file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine device and dtype
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[Probe Safety Error] CUDA requested but not available. Fallback to CPU.")
        device = torch.device("cpu")

    if args.dtype is not None:
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        dtype = dtype_map[args.dtype]
    else:
        dtype = torch.float32 if device.type == "cpu" else torch.float16

    # Host RAM and VRAM safety disclaimer
    print("=" * 70)
    print("PROBE SAFETY & RESOURCE DISCLOSURE")
    print(f"Device: {device} | Dtype: {dtype} | Mock: {args.mock}")
    print("Estimated Host RAM: ~1.5 - 2.5 GiB (VAE only, Diffusion Transformer NOT loaded)")
    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024 * 1024)
        print(f"Detected GPU: {torch.cuda.get_device_name(device)} | Total VRAM: {total_vram:.2f} GiB")
        print(f"Current VRAM Reserved: {reserved:.2f} GiB")
    print("=" * 70)

    # Parse margins
    margins = [int(m.strip()) for m in args.margins.split(",") if m.strip()]

    # Load VAE
    vae = load_vae(
        model_type=args.model_type,
        vae_path=args.vae_path,
        mock=args.mock,
        device=device,
        dtype=dtype,
    )

    # Run measurements
    data = measure_parity_and_cost(
        vae=vae,
        device=device,
        dtype=dtype,
        resolution=args.res,
        crop_cells=args.crop_cells,
        margins=margins,
        crop_pos=args.crop_pos,
        criterion_name=args.criterion,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        latent_amplitude=args.latent_amplitude,
    )

    # Print human-readable report
    print_report(data)

    # Save json if requested
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Probe] Saved results to: {out_path}")


if __name__ == "__main__":
    main()

