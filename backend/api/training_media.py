"""Bandwidth-bounded media helpers for the training monitor."""

from __future__ import annotations

import hashlib
import os
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4


PREVIEW_SIZES = (256, 512, 768)


class PreviewSize(IntEnum):
    SMALL = 256
    MEDIUM = 512
    LARGE = 768


_DEBUG_TENSOR_KEYS = {
    "target": "latents",
    "noisy": "noisy_latents",
    "predicted_latent": "predicted_latent",
    "predicted_noise": "predicted_noise",
    "predicted_velocity": "predicted_velocity",
    "audio_target": "audio_latents",
    "audio_noisy": "audio_noisy_latents",
    "audio_predicted_velocity": "audio_predicted_velocity",
    "audio_actual_velocity": "audio_actual_velocity",
    "audio_predicted_latent": "audio_predicted_latent",
}

_DEBUG_RESPONSE_KEYS = {
    "target": "latents_image",
    "noisy": "noisy_latents_image",
    "predicted_latent": "predicted_latent_image",
    "predicted_noise": "predicted_noise_image",
    "predicted_velocity": "predicted_velocity_image",
    "reference": "reference_image",
    "audio_target": "audio_latents_image",
    "audio_noisy": "audio_noisy_latents_image",
    "audio_predicted_velocity": "audio_predicted_velocity_image",
    "audio_actual_velocity": "audio_actual_velocity_image",
    "audio_predicted_latent": "audio_predicted_latent_image",
}


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns:x}-{stat.st_size:x}"


def publish_sample_png(image: Any, path: Path, pnginfo: Any) -> None:
    """Expose a training sample only after its PNG encoder has finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        image.save(pending, format="PNG", pnginfo=pnginfo)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def preview_etag(*parts: object) -> str:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).hexdigest()
    return f'"{digest[:32]}"'


def immutable_headers(etag: str) -> dict[str, str]:
    return {
        "Cache-Control": "private, max-age=31536000, immutable",
        "ETag": etag,
        # The body is already compressed image data. Do not let the app-wide
        # gzip middleware spend CPU or disturb Content-Length.
        "Content-Encoding": "identity",
    }


@lru_cache(maxsize=2048)
def _png_text_for_version(path: str, version: str) -> dict[str, str] | None:
    del version
    from PIL import Image

    with Image.open(path) as image:
        return dict(image.text) if getattr(image, "text", None) else None


def cached_png_text(path: Path) -> dict[str, str] | None:
    return _png_text_for_version(str(path), file_fingerprint(path))


def find_debug_latent_file(debug_dir: Path, timestep: float | None) -> Path:
    if timestep is None:
        files = sorted(debug_dir.glob("latents_t*.pt"))
        if not files:
            raise FileNotFoundError("No latent files found")
        return files[0]

    candidates = [
        debug_dir / f"latents_t{timestep}.pt",
        debug_dir / f"latents_t{int(timestep):04d}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Float formatting is not stable across JSON/filename round trips. Match
    # numerically as the final bounded fallback.
    for candidate in debug_dir.glob("latents_t*.pt"):
        try:
            saved = float(candidate.stem.replace("latents_t", ""))
        except ValueError:
            continue
        if abs(saved - float(timestep)) <= 1e-6:
            return candidate
    raise FileNotFoundError(f"Latent file for timestep {timestep} not found")


def _decoded_webp_path(latent_file: Path, kind: str) -> Path | None:
    suffix = {
        "target": "target",
        "predicted_latent": "pred_x0",
        "noisy": "noisy",
    }.get(kind)
    if suffix is None:
        return None
    ts = latent_file.stem.replace("latents_t", "")
    candidate = latent_file.parent / f"decode_t{ts}_{suffix}.webp"
    return candidate if candidate.exists() else None


def debug_image_sources(data: dict[str, Any], latent_file: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for kind, tensor_key in _DEBUG_TENSOR_KEYS.items():
        decoded = _decoded_webp_path(latent_file, kind)
        if decoded is not None:
            sources[kind] = decoded
        elif tensor_key in data:
            sources[kind] = latent_file
    reference = data.get("reference_image_path")
    if reference and not str(reference).startswith("temp_img://"):
        path = Path(str(reference))
        if path.is_file():
            sources["reference"] = path
    return sources


def debug_image_urls(
    run_id: int,
    step: int,
    data: dict[str, Any],
    latent_file: Path,
) -> dict[str, str]:
    urls: dict[str, str] = {}
    # The tensor may retain more precision than the filename used to locate it.
    # Publish the filename token so the image request resolves the same artifact.
    timestep = latent_file.stem.replace("latents_t", "", 1)
    for kind, source in debug_image_sources(data, latent_file).items():
        version = file_fingerprint(source)
        query = urlencode({"timestep": timestep, "v": version})
        urls[_DEBUG_RESPONSE_KEYS[kind]] = (
            f"/api/v1/training/runs/{run_id}/debug-latents/{step}/images/{kind}?{query}"
        )
    return urls


def _flux2_unpatchify(tensor):
    channels, height, width = tensor.shape
    tensor = tensor.view(channels // 4, 2, 2, height, width)
    return tensor.permute(0, 3, 1, 4, 2).reshape(channels // 4, height * 2, width * 2)


def _tensor_image(tensor, *, is_flux2: bool):
    import numpy as np
    import torch
    from PIL import Image

    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    if is_flux2 and tensor.shape[0] == 128:
        tensor = _flux2_unpatchify(tensor)
    values = tensor.numpy()
    if values.shape[0] >= 3:
        rgb = values[:3]
    elif values.shape[0] == 1:
        rgb = np.repeat(values, 3, axis=0)
    else:
        rgb = np.zeros((3,) + values.shape[1:], dtype=values.dtype)
        rgb[: values.shape[0]] = values
    normalized = np.zeros_like(rgb)
    for channel_index in range(3):
        channel = rgb[channel_index]
        lo, hi = channel.min(), channel.max()
        if hi - lo > 1e-6:
            normalized[channel_index] = (channel - lo) / (hi - lo) * 255.0
    pixels = normalized.transpose(1, 2, 0).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _atomic_webp(image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        image.save(temporary, format="WEBP", quality=82, method=4)
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def cached_image_preview(source: Path, cache_dir: Path, size: int) -> tuple[Path, str]:
    from PIL import Image

    if size not in PREVIEW_SIZES:
        raise ValueError(f"unsupported preview size: {size}")
    version = file_fingerprint(source)
    etag = preview_etag(source.resolve(), version, size)
    target = cache_dir / f"{source.stem}.{version}.{size}.webp"
    if not target.exists():
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            _atomic_webp(image, target)
    return target, etag


def cached_debug_preview(
    latent_file: Path,
    data: dict[str, Any],
    kind: str,
    size: int,
) -> tuple[Path, str]:
    from PIL import Image

    sources = debug_image_sources(data, latent_file)
    source = sources.get(kind)
    if source is None:
        raise KeyError(kind)
    if source != latent_file:
        return cached_image_preview(source, latent_file.parent / ".previews", size)

    tensor_key = _DEBUG_TENSOR_KEYS[kind]
    version = file_fingerprint(latent_file)
    etag = preview_etag(latent_file.resolve(), version, kind, size)
    target = latent_file.parent / ".previews" / f"{latent_file.stem}.{version}.{kind}.{size}.webp"
    if not target.exists():
        image = _tensor_image(
            data[tensor_key], is_flux2=data.get("model_type") == "flux2"
        )
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        _atomic_webp(image, target)
    return target, etag
