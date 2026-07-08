"""Upscale backend: PIL resize, spandrel super-resolution models, RTX Video Super
Resolution (nvvfx). Backend-agnostic entry point: run_upscale().

Design notes:
- spandrel and nvvfx are optional dependencies. Both are imported lazily inside
  the functions that need them so `import core.upscaler` never fails even when
  neither package is installed.
- The spandrel model cache is a single module-level slot (last-loaded model,
  keyed by absolute path) — mirrors the project's existing "keep one model
  hot" VRAM discipline. The model lives on GPU only for the duration of the
  job; it is moved back to CPU afterward.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import math
import os
import time

from PIL import Image, ImageFilter

from api.error_handlers import ValidationError, GenerationError


# ---------------------------------------------------------------------------
# spandrel module-level single-entry model cache
# ---------------------------------------------------------------------------
_spandrel_cache: Dict[str, Any] = {"path": None, "model": None}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_target_dims(input_w: int, input_h: int, scale_factor: float) -> Tuple[int, int]:
    target_w = max(1, round(input_w * scale_factor))
    target_h = max(1, round(input_h * scale_factor))
    return target_w, target_h


def _maybe_unsharp(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    if not params.get("unsharp_enable"):
        return image
    radius = float(params.get("unsharp_radius", 2.0))
    percent = int(params.get("unsharp_percent", 100))
    threshold = int(params.get("unsharp_threshold", 3))
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


# ---------------------------------------------------------------------------
# PIL backend
# ---------------------------------------------------------------------------

_PIL_RESAMPLE_MAP = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "nearest": Image.Resampling.NEAREST,
}


def _run_pil(image: Image.Image, params: Dict[str, Any]) -> Tuple[Image.Image, List[str]]:
    warnings: List[str] = []
    resample_name = params.get("pil_resample", "lanczos")
    resample = _PIL_RESAMPLE_MAP.get(resample_name)
    if resample is None:
        warnings.append(f"Unknown pil_resample '{resample_name}', falling back to lanczos.")
        resample = Image.Resampling.LANCZOS
        resample_name = "lanczos"
    params["pil_resample"] = resample_name

    target_w, target_h = _resolve_target_dims(image.width, image.height, params.get("scale_factor", 2.0))
    result = image.resize((target_w, target_h), resample=resample)
    return result, warnings


# ---------------------------------------------------------------------------
# spandrel backend
# ---------------------------------------------------------------------------

def _load_spandrel_model(model_path: str):
    """Load (or reuse the cached) spandrel model from a .pth/.safetensors file."""
    try:
        from spandrel import ModelLoader
    except ImportError as e:
        raise ValidationError(
            "spandrel is not installed",
            detail="Install it with: pip install spandrel>=0.4.2"
        ) from e

    if _spandrel_cache["path"] == model_path and _spandrel_cache["model"] is not None:
        return _spandrel_cache["model"]

    if not os.path.isfile(model_path):
        raise ValidationError(
            "Upscaler model file not found",
            detail=f"Path: {model_path}"
        )

    loader = ModelLoader()
    loaded = loader.load_from_file(model_path)
    loaded = loaded.eval()

    _spandrel_cache["path"] = model_path
    _spandrel_cache["model"] = loaded
    return loaded


def _tensor_from_image(image: Image.Image, device):
    import numpy as np
    import torch

    arr = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor


def _image_from_tensor(tensor) -> Image.Image:
    import numpy as np

    arr = tensor.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _run_spandrel_tiled(
    model,
    image: Image.Image,
    tile_size: int,
    tile_overlap: int,
    progress_callback: Optional[Callable] = None,
) -> Image.Image:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    try:
        with torch.no_grad():
            if tile_size <= 0:
                # No tiling: run the whole image through the model in one shot.
                if progress_callback:
                    progress_callback(0, 1)
                in_tensor = _tensor_from_image(image, device)
                out_tensor = model(in_tensor)
                if progress_callback:
                    progress_callback(1, 1)
                return _image_from_tensor(out_tensor)

            in_w, in_h = image.width, image.height
            # Determine model's native scale by probing a small patch.
            probe_size = min(tile_size, in_w, in_h, 64)
            probe = image.crop((0, 0, probe_size, probe_size))
            probe_tensor = _tensor_from_image(probe, device)
            probe_out = model(probe_tensor)
            native_scale = probe_out.shape[-1] / probe_tensor.shape[-1]

            out_w = round(in_w * native_scale)
            out_h = round(in_h * native_scale)
            out_tensor = torch.zeros((1, 3, out_h, out_w), dtype=torch.float32, device=device)
            weight_tensor = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device=device)

            step = max(1, tile_size - tile_overlap)
            xs = list(range(0, in_w, step))
            ys = list(range(0, in_h, step))
            if xs[-1] + tile_size < in_w:
                xs.append(in_w - tile_size)
            if ys[-1] + tile_size < in_h:
                ys.append(in_h - tile_size)
            xs = sorted(set(max(0, min(x, in_w - min(tile_size, in_w))) for x in xs))
            ys = sorted(set(max(0, min(y, in_h - min(tile_size, in_h))) for y in ys))

            tiles = [(x, y) for y in ys for x in xs]
            total_tiles = len(tiles)

            for idx, (x, y) in enumerate(tiles):
                if progress_callback:
                    progress_callback(idx, total_tiles)

                tx1, ty1 = x, y
                tx2, ty2 = min(x + tile_size, in_w), min(y + tile_size, in_h)
                tile = image.crop((tx1, ty1, tx2, ty2))
                tile_tensor = _tensor_from_image(tile, device)
                tile_out = model(tile_tensor)

                tw = tile_out.shape[-1]
                th = tile_out.shape[-2]
                ox1 = round(tx1 * native_scale)
                oy1 = round(ty1 * native_scale)
                ox2 = ox1 + tw
                oy2 = oy1 + th
                ox2 = min(ox2, out_w)
                oy2 = min(oy2, out_h)
                tw_c = ox2 - ox1
                th_c = oy2 - oy1
                if tw_c <= 0 or th_c <= 0:
                    continue

                # Feather blend weight: linear ramp toward zero at the tile edges,
                # proportional to overlap so seams blend smoothly.
                feather_px = round(tile_overlap * native_scale)
                w_mask = torch.ones((th_c, tw_c), dtype=torch.float32, device=device)
                if feather_px > 0:
                    ramp_y = torch.ones(th_c, dtype=torch.float32, device=device)
                    ramp_x = torch.ones(tw_c, dtype=torch.float32, device=device)
                    n = min(feather_px, th_c // 2)
                    if n > 0:
                        ramp = torch.linspace(0.0, 1.0, n, device=device)
                        ramp_y[:n] = ramp
                        ramp_y[-n:] = ramp.flip(0)
                    n = min(feather_px, tw_c // 2)
                    if n > 0:
                        ramp = torch.linspace(0.0, 1.0, n, device=device)
                        ramp_x[:n] = ramp
                        ramp_x[-n:] = ramp.flip(0)
                    w_mask = ramp_y.unsqueeze(1) * ramp_x.unsqueeze(0)

                out_tensor[:, :, oy1:oy2, ox1:ox2] += tile_out[:, :, :th_c, :tw_c] * w_mask
                weight_tensor[:, :, oy1:oy2, ox1:ox2] += w_mask

            if progress_callback:
                progress_callback(total_tiles, total_tiles)

            weight_tensor = weight_tensor.clamp(min=1e-6)
            out_tensor = out_tensor / weight_tensor
            return _image_from_tensor(out_tensor)
    finally:
        model.to("cpu")
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()


def _run_spandrel(
    image: Image.Image,
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Tuple[Image.Image, List[str]]:
    warnings: List[str] = []
    model_name = params.get("upscaler_model")
    if not model_name:
        raise ValidationError(
            "upscaler_model is required when upscaler_backend='spandrel'",
        )

    model_path = params.get("_upscaler_model_path")
    if not model_path or not os.path.isfile(model_path):
        raise ValidationError(
            "Upscaler model file not found",
            detail=f"model: {model_name}"
        )

    model = _load_spandrel_model(model_path)

    tile_size = int(params.get("tile_size", 512) or 0)
    tile_overlap = int(params.get("tile_overlap", 32) or 0)

    result = _run_spandrel_tiled(model, image, tile_size, tile_overlap, progress_callback)

    # If the model's native scale doesn't match the requested scale_factor,
    # Lanczos-resize to the exact target.
    target_w, target_h = _resolve_target_dims(image.width, image.height, params.get("scale_factor", 2.0))
    if result.width != target_w or result.height != target_h:
        result = result.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

    # Compute + record model hash for metadata/DB.
    try:
        params["upscaler_model_hash"] = _sha256_file(model_path)
    except OSError as e:
        warnings.append(f"Could not hash upscaler model file: {e}")

    return result, warnings


# ---------------------------------------------------------------------------
# RTX Video Super Resolution backend (nvvfx)
# ---------------------------------------------------------------------------

_NVVFX_INSTALL_HINT = (
    "pip install -U --no-build-isolation nvidia-vfx --index-url https://pypi.nvidia.com"
)


def _run_rtx_vsr(
    image: Image.Image,
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Tuple[Image.Image, List[str]]:
    warnings: List[str] = []
    try:
        import nvvfx
    except ImportError as e:
        raise ValidationError(
            "RTX Video Super Resolution (nvvfx) is not installed",
            detail=f"Install with: {_NVVFX_INSTALL_HINT}"
        ) from e

    import torch

    if not torch.cuda.is_available():
        raise ValidationError(
            "RTX Video Super Resolution requires a CUDA device",
        )

    quality = params.get("rtx_vsr_quality", "high")
    target_w, target_h = _resolve_target_dims(image.width, image.height, params.get("scale_factor", 2.0))

    # nvvfx requires output dims to be multiples of 8, clamped to [64, 8192].
    def _snap(v: int) -> int:
        v = max(64, min(8192, v))
        return int(round(v / 8) * 8)

    output_width = _snap(target_w)
    output_height = _snap(target_h)
    if output_width != target_w or output_height != target_h:
        warnings.append(
            f"RTX VSR output snapped to multiple-of-8: requested {target_w}x{target_h}, "
            f"used {output_width}x{output_height}."
        )

    if progress_callback:
        progress_callback(0, 1)

    device = torch.device("cuda")
    arr_tensor = _tensor_from_image(image, device).squeeze(0).contiguous()  # CHW float32 CUDA RGB

    quality_levels = nvvfx.VideoSuperRes.QualityLevel
    quality_map = {
        "low": quality_levels.LOW,
        "medium": quality_levels.MEDIUM,
        "high": quality_levels.HIGH,
        "ultra": quality_levels.ULTRA,
    }
    if quality not in quality_map:
        raise ValidationError(
            "Invalid rtx_vsr_quality parameter",
            detail=f"rtx_vsr_quality must be one of {sorted(quality_map)}, got {quality}"
        )

    with nvvfx.VideoSuperRes(quality=quality_map[quality]) as effect:
        effect.output_width = output_width
        effect.output_height = output_height
        effect.load()
        out = effect.run(arr_tensor)
        out_tensor = torch.from_dlpack(out.image).clone()

    if progress_callback:
        progress_callback(1, 1)

    if out_tensor.dim() == 3:
        out_tensor = out_tensor.unsqueeze(0)
    result = _image_from_tensor(out_tensor)

    params["rtx_vsr_quality"] = quality
    return result, warnings


# ---------------------------------------------------------------------------
# diffusion tile upscale backend
# ---------------------------------------------------------------------------

def _compute_tile_boxes(width: int, height: int, tile_size: int, tile_overlap: int) -> List[Tuple[int, int, int, int]]:
    """Compute tile crop boxes covering (width, height) in output-pixel space.

    Tile dims are snapped to a multiple of 8. Edge tiles are pulled inward
    (rather than padded) so every tile is a real crop of the image; edge
    tiles may overlap their neighbor more than `tile_overlap`.
    """
    if tile_size <= 0:
        return [(0, 0, width, height)]

    tile_size = max(8, int(round(tile_size / 8)) * 8)
    tile_w = min(tile_size, (width // 8) * 8 or width)
    tile_h = min(tile_size, (height // 8) * 8 or height)
    tile_w = max(8, tile_w)
    tile_h = max(8, tile_h)

    step_x = max(1, tile_w - tile_overlap)
    step_y = max(1, tile_h - tile_overlap)

    xs = list(range(0, max(1, width - tile_w) + 1, step_x))
    if not xs or xs[-1] != width - tile_w:
        xs.append(max(0, width - tile_w))
    ys = list(range(0, max(1, height - tile_h) + 1, step_y))
    if not ys or ys[-1] != height - tile_h:
        ys.append(max(0, height - tile_h))
    xs = sorted(set(max(0, min(x, width - tile_w)) for x in xs))
    ys = sorted(set(max(0, min(y, height - tile_h)) for y in ys))

    boxes = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, x + tile_w, y + tile_h))
    return boxes


def _feather_blend_tiles(
    width: int,
    height: int,
    boxes: List[Tuple[int, int, int, int]],
    tile_images: List[Image.Image],
    tile_overlap: int,
) -> Image.Image:
    """Blend tile_images (aligned with boxes) back into a (width, height) canvas
    using a linear feather ramp proportional to tile_overlap."""
    import numpy as np

    canvas = np.zeros((height, width, 3), dtype="float32")
    weight = np.zeros((height, width, 1), dtype="float32")

    for (x1, y1, x2, y2), tile_img in zip(boxes, tile_images):
        tw, th = x2 - x1, y2 - y1
        arr = np.asarray(tile_img.convert("RGB"), dtype="float32")
        if arr.shape[0] != th or arr.shape[1] != tw:
            tile_img = tile_img.resize((tw, th), resample=Image.Resampling.LANCZOS)
            arr = np.asarray(tile_img.convert("RGB"), dtype="float32")

        feather_px = max(0, int(tile_overlap))
        ramp_y = np.ones(th, dtype="float32")
        ramp_x = np.ones(tw, dtype="float32")
        n = min(feather_px, th // 2)
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype="float32")
            ramp_y[:n] = ramp
            ramp_y[-n:] = ramp[::-1]
        n = min(feather_px, tw // 2)
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype="float32")
            ramp_x[:n] = ramp
            ramp_x[-n:] = ramp[::-1]
        w_mask = (ramp_y[:, None] * ramp_x[None, :])[:, :, None]

        canvas[y1:y2, x1:x2, :] += arr * w_mask
        weight[y1:y2, x1:x2, :] += w_mask

    weight = np.clip(weight, 1e-6, None)
    result = canvas / weight
    result = (result + 0.5).clip(0, 255).astype("uint8")
    return Image.fromarray(result, mode="RGB")


def run_diffusion_upscale(
    params: Dict[str, Any],
    input_image: Image.Image,
    pipeline_manager,
    progress_callback: Optional[Callable] = None,
) -> Tuple[Image.Image, List[str]]:
    """Diffusion tile upscale: pre-upscale to target dims, split into
    overlapping tiles, run img2img on each tile via pipeline_manager, and
    feather-blend the results back together.

    Mutates `params` with resolved values (seed, pre-upscale mode, etc.).
    Returns (result_image, warnings).
    """
    import random

    warnings: List[str] = []

    if pipeline_manager.current_model_info is None:
        raise ValidationError(
            "No diffusion model loaded",
            detail="Load a model before using upscaler_backend='diffusion'."
        )

    denoising_strength = float(params.get("diffusion_denoising_strength", 0.3))
    if denoising_strength < 0.05 or denoising_strength > 0.9:
        raise ValidationError(
            "Invalid diffusion_denoising_strength parameter",
            detail=f"diffusion_denoising_strength must be between 0.05 and 0.9, got {denoising_strength}"
        )
    params["diffusion_denoising_strength"] = denoising_strength

    pre_upscale_mode = params.get("diffusion_pre_upscale_mode", "pil")
    if pre_upscale_mode not in ("pil", "model"):
        warnings.append(f"Unknown diffusion_pre_upscale_mode '{pre_upscale_mode}', falling back to pil.")
        pre_upscale_mode = "pil"
    params["diffusion_pre_upscale_mode"] = pre_upscale_mode

    target_w, target_h = _resolve_target_dims(input_image.width, input_image.height, params.get("scale_factor", 2.0))

    # --- Pre-upscale to target dims ---
    if pre_upscale_mode == "model":
        spandrel_params = dict(params)
        pre_image, spandrel_warnings = _run_spandrel(input_image, spandrel_params, progress_callback=None)
        warnings.extend(spandrel_warnings)
        # _run_spandrel already resizes to the target dims when native scale differs.
        params["upscaler_model_hash"] = spandrel_params.get("upscaler_model_hash")
    else:
        resample_name = params.get("pil_resample", "lanczos")
        resample = _PIL_RESAMPLE_MAP.get(resample_name)
        if resample is None:
            warnings.append(f"Unknown pil_resample '{resample_name}', falling back to lanczos.")
            resample = Image.Resampling.LANCZOS
            resample_name = "lanczos"
        params["pil_resample"] = resample_name
        pre_image = input_image.resize((target_w, target_h), resample=resample)

    if pre_image.width != target_w or pre_image.height != target_h:
        pre_image = pre_image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

    # --- Resolve seed (same base seed for all tiles, offset by tile index) ---
    seed = int(params.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    params["seed"] = seed

    # --- Tile split + per-tile img2img ---
    tile_size = int(params.get("tile_size", 512) or 0)
    tile_overlap = int(params.get("tile_overlap", 32) or 0)
    boxes = _compute_tile_boxes(pre_image.width, pre_image.height, tile_size, tile_overlap)
    total_tiles = len(boxes)

    tile_images: List[Image.Image] = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        if progress_callback:
            progress_callback(idx, total_tiles)

        tile_crop = pre_image.crop((x1, y1, x2, y2))
        tile_params: Dict[str, Any] = {
            "prompt": params.get("prompt", ""),
            "negative_prompt": params.get("negative_prompt", ""),
            "steps": params.get("steps"),
            "cfg_scale": params.get("cfg_scale"),
            "sampler": params.get("sampler"),
            "schedule_type": params.get("schedule_type"),
            "seed": seed + idx,
            "denoising_strength": denoising_strength,
            "width": tile_crop.width,
            "height": tile_crop.height,
        }
        tile_result, _tile_seed, _tile_ancestral_seed = pipeline_manager.generate_img2img(
            tile_params, tile_crop, progress_callback=None, step_callback=None
        )
        tile_images.append(tile_result)

    if progress_callback:
        progress_callback(total_tiles, total_tiles)

    result = _feather_blend_tiles(pre_image.width, pre_image.height, boxes, tile_images, tile_overlap)
    return result, warnings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_upscale(
    params: Dict[str, Any],
    input_image: Image.Image,
    progress_callback: Optional[Callable] = None,
    pipeline_manager=None,
) -> Tuple[Image.Image, List[str]]:
    """Run the configured upscale backend on input_image.

    Mutates `params` in place with metadata-relevant resolved values
    (upscaler_model_hash, pil_resample, rtx_vsr_quality, etc.) so callers can
    pass the same dict on to save_image_with_metadata / create_db_image_record.

    Returns (result_image, warnings).
    """
    backend = params.get("upscaler_backend", "spandrel")
    scale_factor = float(params.get("scale_factor", 2.0))
    if scale_factor < 1.0 or scale_factor > 8.0:
        raise ValidationError(
            "Invalid scale_factor parameter",
            detail=f"scale_factor must be between 1.0 and 8.0, got {scale_factor}"
        )
    params["scale_factor"] = scale_factor

    start = time.perf_counter()
    if backend == "pil":
        result, warnings = _run_pil(input_image, params)
    elif backend == "spandrel":
        result, warnings = _run_spandrel(input_image, params, progress_callback)
    elif backend == "rtx_vsr":
        result, warnings = _run_rtx_vsr(input_image, params, progress_callback)
    elif backend == "diffusion":
        if pipeline_manager is None:
            raise ValidationError(
                "Diffusion upscale requires a pipeline_manager",
            )
        result, warnings = run_diffusion_upscale(params, input_image, pipeline_manager, progress_callback)
    else:
        raise ValidationError(
            "Invalid upscaler_backend parameter",
            detail=f"upscaler_backend must be 'pil', 'spandrel', 'rtx_vsr' or 'diffusion', got '{backend}'"
        )

    if params.get("unsharp_enable"):
        result = _maybe_unsharp(result, params)

    params["upscale_time"] = round(time.perf_counter() - start, 3)
    params["width"] = result.width
    params["height"] = result.height

    return result, warnings
