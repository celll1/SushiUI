"""Distill an SDXL VAE latent stem for the frozen REPA SigLIP trunk.

This deliberately lives outside training runs.  It writes one identity-bound
safetensors artifact plus a JSON report containing validation and gate-3 costs.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.models.components.vae_registry import normalize
from core.training.image_preprocessing import flatten_to_rgb
from core.training.repa import encode_repa_targets, load_repa_encoder
from core.training.repa_latent_stem import (
    LatentRepaStem, economic_gate, encode_latent_targets, save_latent_stem,
    teacher_content_identity, vae_encoder_identity,
)

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_vae(args, dtype):
    if args.vae_source:
        from core.models.common.vae_source import resolve_vae_source
        return resolve_vae_source(args.vae_source, arch="sdxl").load_module(dtype)
    path = Path(args.base_model)
    if path.is_dir():
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(str(path), subfolder="vae", torch_dtype=dtype)
    from diffusers import StableDiffusionXLPipeline
    pipeline = StableDiffusionXLPipeline.from_single_file(
        str(path), torch_dtype=dtype, use_safetensors=True)
    vae = pipeline.vae
    pipeline.vae = None
    del pipeline
    return vae


def _fit(image: Image.Image, width: int, height: int, strategy: str) -> Image.Image:
    image = flatten_to_rgb(image)
    if strategy == "resize":
        return image.resize((width, height), Image.LANCZOS)
    scale = max(width / image.width, height / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((array - 0.5) * 2.0).permute(2, 0, 1)


def _images(roots: list[Path], seed: int, max_items: int) -> list[Path]:
    paths = sorted(
        path for root in roots for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _EXTENSIONS)
    random.Random(seed).shuffle(paths)
    return paths[:max_items] if max_items > 0 else paths


def _paired_batch(paths: list[Path], args, teacher_size: int | None = None):
    images, teachers = [], []
    for path in paths:
        with Image.open(path) as image:
            fitted = _fit(image, args.width, args.height, args.bucket_strategy)
            images.append(_tensor(fitted))
            if teacher_size is not None:
                teachers.append(_tensor(fitted.resize(
                    (teacher_size, teacher_size), Image.BICUBIC)))
    image_batch = torch.stack(images)
    return (image_batch, torch.stack(teachers)) if teacher_size is not None else image_batch


def _batch(paths: list[Path], args) -> torch.Tensor:
    return _paired_batch(paths, args)


def _latent(vae, pixels: torch.Tensor) -> torch.Tensor:
    pixels = pixels.to(dtype=next(vae.parameters()).dtype)
    posterior = vae.encode(pixels).latent_dist
    return normalize(posterior.sample(), vae, None)


def _cosine(student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(student.float(), target.float(), dim=-1).mean()


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("distill_repa_latent_stem requires CUDA")
    if not args.vae_source and not args.base_model:
        raise ValueError("provide --base-model or --vae-source")
    if args.steps <= 0 or args.batch_size <= 0 or args.online_batch_size <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be between zero and one")
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    vae_dtype = {
        "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16,
    }[args.vae_dtype]
    torch.manual_seed(args.seed)

    paths = _images(args.image_dir, args.seed, args.max_items)
    if len(paths) < 3:
        raise ValueError("distillation needs at least three readable images")
    split = min(len(paths) - 2, max(1, round(len(paths) * (1.0 - args.val_fraction))))
    train_paths, val_paths = paths[:split], paths[split:]

    vae = _load_vae(args, vae_dtype).to(
        device=device, dtype=vae_dtype).eval().requires_grad_(False)
    vae_identity, vae_norm = vae_encoder_identity(vae)
    from core.training.repa import _resolve_tagger_checkpoint
    teacher_checkpoint, teacher_repo = _resolve_tagger_checkpoint(args.tagger_model)
    teacher, enc_dim, native_size = load_repa_encoder(
        "tagger", tagger_model_dir=args.tagger_model, dtype=dtype,
        device=device, attn_implementation=args.attention)
    teacher_identity = teacher_content_identity(teacher)
    teacher_size = int(native_size or 384)
    if hasattr(teacher, "gradient_checkpointing_enable") and args.gradient_checkpointing:
        teacher.gradient_checkpointing_enable()

    in_channels = int(getattr(vae.config, "latent_channels", 4))
    stem = LatentRepaStem(in_channels, enc_dim, args.width_channels).to(device=device)
    optimizer = torch.optim.AdamW(stem.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")
    order = list(train_paths)
    step = 0
    processed_items = 0
    item_times = []
    stem.train()
    while step < args.steps:
        random.Random(args.seed + step // max(1, len(order))).shuffle(order)
        for offset in range(0, len(order), args.batch_size):
            selected = order[offset:offset + args.batch_size]
            if not selected:
                continue
            cpu, teacher_cpu = _paired_batch(selected, args, teacher_size)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            pixels = cpu.to(device=device, dtype=dtype)
            teacher_pixels = teacher_cpu.to(device=device, dtype=dtype)
            with torch.no_grad():
                latents = _latent(vae, pixels)
                targets = encode_repa_targets(
                    teacher, teacher_pixels,
                    27, 27, teacher_size)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                predicted = encode_latent_targets(teacher, stem, latents, 27, 27)
                loss = 1.0 - _cosine(predicted, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            item_times.append((time.perf_counter() - started) * 1000.0 / len(selected))
            processed_items += len(selected)
            step += 1
            if step == 1 or step % args.log_every == 0 or step >= args.steps:
                print(f"[REPA stem] step={step}/{args.steps} loss={loss.item():.6f}")
            if step >= args.steps:
                break

    stem.eval()
    cosines, shuffled = [], []
    with torch.no_grad():
        for offset in range(0, len(val_paths), args.batch_size):
            cpu, teacher_cpu = _paired_batch(
                val_paths[offset:offset + args.batch_size], args, teacher_size)
            pixels = cpu.to(device=device, dtype=dtype)
            teacher_pixels = teacher_cpu.to(device=device, dtype=dtype)
            latents = _latent(vae, pixels)
            targets = encode_repa_targets(teacher, teacher_pixels, 27, 27, teacher_size)
            with torch.autocast("cuda", dtype=dtype):
                predicted = encode_latent_targets(teacher, stem, latents, 27, 27)
            cosines.append(float(_cosine(predicted, targets).item()))
            shuffled.append(float(_cosine(predicted, targets.flip(1)).item()))

        pair_cpu, pair_teacher_cpu = _paired_batch(val_paths[:2], args, teacher_size)
        pair_pixels = pair_cpu.to(device=device, dtype=dtype)
        pair_teacher_pixels = pair_teacher_cpu.to(device=device, dtype=dtype)
        pair_latents = _latent(vae, pair_pixels)
        pair_targets = encode_repa_targets(
            teacher, pair_teacher_pixels, 27, 27, teacher_size)
        with torch.autocast("cuda", dtype=dtype):
            pair_predicted = encode_latent_targets(
                teacher, stem, pair_latents, 27, 27)
        different_image_cosine = float(
            _cosine(pair_predicted, pair_targets.roll(1, dims=0)).item())

    probe_latent = _latent(vae, _batch([val_paths[0]], args).to(device=device, dtype=dtype))
    def _stem_forward():
        with torch.autocast("cuda", dtype=dtype):
            return stem(probe_latent)

    for _ in range(5):
        _stem_forward()
    samples = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _stem_forward()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    stem_ms = sorted(samples)[len(samples) // 2]
    distill_ms = sum(item_times) / len(item_times)
    gate = economic_gate(
        redistill_items=processed_items, distill_ms_per_item=distill_ms,
        online_steps=args.expected_online_steps, batch_size=args.online_batch_size,
        replaced_ms_per_item=args.replaced_ms_per_item, stem_ms_per_item=stem_ms)
    metadata = {
        "vae_encoder_identity": vae_identity,
        "vae_normalization": vae_norm,
        "teacher_identity": teacher_identity,
        "teacher_checkpoint": str(Path(teacher_checkpoint).resolve()),
        "teacher_repo": teacher_repo,
        "teacher_size": teacher_size,
        "teacher_dtype": args.dtype,
        "vae_dtype": args.vae_dtype,
        "training_resolution": [args.height, args.width],
        "bucket_strategy": args.bucket_strategy,
        "distill_items": processed_items,
        "distill_steps": args.steps,
    }
    save_latent_stem(args.output, stem, metadata)
    report = {
        **metadata,
        "artifact": str(args.output.resolve()),
        "validation_items": len(val_paths),
        "validation_patch_cosine": sum(cosines) / len(cosines),
        "validation_different_image_cosine": different_image_cosine,
        "validation_spatially_reversed_cosine": sum(shuffled) / len(shuffled),
        "distill_ms_per_item": distill_ms,
        "stem_median_ms_per_item": stem_ms,
        "economic_gate": gate,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, action="append", required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--vae-source", default="")
    parser.add_argument(
        "--tagger-model", "--tagger-dir", dest="tagger_model", required=True,
        help="Exact .onnx/.safetensors file, or a legacy tagger model directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--bucket-strategy", choices=("resize", "crop"), default="resize")
    parser.add_argument("--width-channels", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-items", type=int, default=4096)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--vae-dtype", choices=("fp32", "bf16", "fp16"), default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--expected-online-steps", type=int, required=True)
    parser.add_argument("--online-batch-size", type=int, default=4)
    parser.add_argument("--replaced-ms-per-item", type=float, default=6.02)
    return parser


if __name__ == "__main__":
    result = run(_parser().parse_args())
    print(json.dumps(result, indent=2))
