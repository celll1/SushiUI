"""Real-checkpoint MiniMax-H3 activation-offload A/B probe.

The probe loads the production H3 transformer, injects the production training
LoRA, and executes ``minimax_h3_ops.train_step`` before any optimizer update.
It deliberately skips the VAE and text encoder: inputs model the cached latent,
audio-latent, and text-embedding records consumed by the training step.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.memory_management import offload_activations
from core.models.minimax_h3.h3_pipeline_ops import AUDIO_CHANNELS, audio_latent_frames
from core.models.minimax_h3.loader import (
    MINIMAX_H3_AUDIO_LATENT_RATE,
    MINIMAX_H3_FPS,
    _build_transformer,
    detect_minimax_h3_layout,
)
from core.training.adapters.minimax_h3_adapter import MiniMaxH3LoRAAdapter
from core.training.ops import minimax_h3_ops

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


GIB = 1024 ** 3
MEMORY_FRACTION = 0.80
MIN_FREE_GIB = 34.0
MIN_HOST_AVAILABLE_GIB = 24.0
THRESHOLD_BYTES = 4 * 1024 * 1024
CLIPS = {
    "short": {"pixel_frames": 22, "latent_frames": 7},
    "long": {"pixel_frames": 124, "latent_frames": 37},
}


def _rss_gib() -> float | None:
    return None if psutil is None else psutil.Process().memory_info().rss / GIB


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _trainable(layers: dict):
    seen = set()
    for layer_name, layer in layers.items():
        for param_name, param in layer.named_parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                yield f"{layer_name}.{param_name}", param


def _initialise_lora(layers: dict) -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260910)
    with torch.no_grad():
        for _name, param in _trainable(layers):
            value = torch.randn(param.shape, generator=generator, dtype=torch.float32)
            param.copy_(value.to(device=param.device, dtype=param.dtype).mul_(0.01))


def _capture_gradients(layers: dict) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().cpu().clone()
        for name, param in _trainable(layers)
        if param.grad is not None
    }


def _compare_gradients(off: dict, on: dict) -> dict:
    missing = sorted(set(off) ^ set(on))
    exact_mismatches = []
    tolerance_mismatches = []
    max_abs_error = 0.0
    for name in sorted(set(off) & set(on)):
        left, right = off[name], on[name]
        if not torch.equal(left, right):
            exact_mismatches.append(name)
        if not torch.allclose(left, right, atol=1.0e-3, rtol=1.0e-2):
            tolerance_mismatches.append(name)
        max_abs_error = max(
            max_abs_error,
            float((left.float() - right.float()).abs().max()),
        )
    return {
        "missing": missing,
        "exact": not missing and not exact_mismatches,
        "exact_mismatch_count": len(exact_mismatches),
        "exact_mismatch_examples": exact_mismatches[:10],
        "within_bf16_tolerance": not missing and not tolerance_mismatches,
        "tolerance_mismatch_count": len(tolerance_mismatches),
        "tolerance_mismatch_examples": tolerance_mismatches[:10],
        "max_abs_error": max_abs_error,
        "atol": 1.0e-3,
        "rtol": 1.0e-2,
    }


def _optimizer_smoke(layers: dict, gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    named = list(_trainable(layers))
    initial = {name: param.detach().cpu().clone() for name, param in named}
    optimizer = torch.optim.AdamW((param for _name, param in named), lr=1.0e-4)
    try:
        for name, param in named:
            param.grad = gradients[name].to(device=param.device, dtype=param.dtype)
        optimizer.step()
        return {name: param.detach().cpu().clone() for name, param in named}
    finally:
        with torch.no_grad():
            for name, param in named:
                param.copy_(initial[name].to(device=param.device, dtype=param.dtype))
                param.grad = None


def _zero_grad(layers: dict) -> None:
    for _name, param in _trainable(layers):
        param.grad = None


def _execute(trainer, layers, inputs, enabled: bool, seed: int, capture: bool) -> dict:
    _zero_grad(layers)
    trainer._probe_gradients = {}
    trainer._pending_extra_metrics = {}
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    stats = {"bytes": 0}
    rss_before = _rss_gib()
    started = time.perf_counter()
    with offload_activations(
        enabled, threshold_bytes=THRESHOLD_BYTES, stats=stats
    ):
        loss, _prediction_loss, _reconstruction_loss = minimax_h3_ops.train_step(
            trainer,
            latents=inputs["latents"],
            prompt_embeds=inputs["prompt_embeds"],
            h3_aux=inputs["h3_aux"],
            timesteps=inputs["timesteps"],
        )
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if trainer.layer_offload_conductor is not None:
        gradients = dict(trainer._probe_gradients) if capture else None
        grad_values = trainer._probe_gradients.values()
    else:
        gradients = _capture_gradients(layers) if capture else None
        grad_values = (param.grad for _name, param in _trainable(layers)
                       if param.grad is not None)
    finite = bool(torch.isfinite(loss).item()) and all(
        torch.isfinite(grad).all().item() for grad in grad_values
    )
    result = {
        "loss": float(loss.detach()),
        "finite": finite,
        "gradient_tensor_count": (
            len(trainer._probe_gradients)
            if trainer.layer_offload_conductor is not None
            else sum(param.grad is not None for _name, param in _trainable(layers))
        ),
        "offloaded_bytes": int(stats["bytes"]),
        "resident_before_gib": allocated_before / GIB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "host_rss_before_gib": rss_before,
        "host_rss_after_gib": _rss_gib(),
        "elapsed_seconds": elapsed,
        "gradients": gradients,
    }
    del loss
    return result


def _without_gradients(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "gradients"}


def _build_inputs(clip: str) -> dict:
    geometry = CLIPS[clip]
    generator = torch.Generator(device="cpu").manual_seed(127)
    latent_t = geometry["latent_frames"]
    latents = torch.randn((1, 24, latent_t, 24, 40), generator=generator)
    prompt = torch.randn((1, 32, 5120), generator=generator).to(torch.bfloat16)
    audio_t = audio_latent_frames(
        geometry["pixel_frames"], MINIMAX_H3_FPS, MINIMAX_H3_AUDIO_LATENT_RATE
    )
    audio_rows = AUDIO_CHANNELS * audio_t
    audio = torch.randn((1, audio_rows, 32), generator=generator)
    return {
        "latents": latents,
        "prompt_embeds": prompt,
        "h3_aux": {
            "num_text_tokens": torch.tensor([prompt.shape[1]], dtype=torch.long),
            "audio_latents": audio,
            "audio_present": torch.tensor([True]),
            "audio_valid_rows": torch.ones((1, audio_rows), dtype=torch.bool),
        },
        "timesteps": torch.tensor([0.5]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--clip", choices=tuple(CLIPS), required=True)
    parser.add_argument("--arm", choices=("both", "off", "on"), default="both")
    parser.add_argument("--blocks-to-swap", type=int, default=0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.blocks_to_swap < 0 or args.blocks_to_swap >= 50:
        parser.error("--blocks-to-swap must be between 0 and 49")
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2

    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    host_available = None if psutil is None else psutil.virtual_memory().available / GIB
    if free / GIB < MIN_FREE_GIB:
        print(f"Refusing probe: only {free / GIB:.2f} GiB VRAM free", file=sys.stderr)
        return 3
    if host_available is not None and host_available < MIN_HOST_AVAILABLE_GIB:
        print(f"Refusing probe: only {host_available:.2f} GiB host RAM available", file=sys.stderr)
        return 4
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)

    layout = detect_minimax_h3_layout(args.model)
    if layout is None:
        raise ValueError(f"MiniMax-H3 layout not found at {args.model!r}")
    transformer, config = _build_transformer(
        layout["dit"], torch.bfloat16, layout["official"]
    )
    if not args.no_gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.to("cuda")
    transformer.train()

    trainer = SimpleNamespace(
        transformer=transformer,
        device=torch.device("cuda"),
        training_dtype=torch.bfloat16,
        config={},
        adapter_algorithm="lora",
        weight_decompose=False,
        adapter_config={},
        arch=None,
        blocks_to_swap=args.blocks_to_swap,
        block_swap_ring_size=2,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_pinned_memory=False,
        layer_offload_conductor=None,
        is_minimax_h3=True,
        learning_rate=1.0e-4,
        unet_lr=1.0e-4,
        log_prefix="[H3 ActDispatch Probe]",
        minimax_h3_fps=MINIMAX_H3_FPS,
        minimax_h3_audio_latent_rate=MINIMAX_H3_AUDIO_LATENT_RATE,
        minimax_h3_components={"audio_latent_channels": 32},
        audio_loss_weight=1.0,
        timestep_sampler=None,
        reconstruction_loss_weight=0.0,
        _pending_extra_metrics={},
        defer_extra_metric=lambda _name, _value: None,
        log_extra_metric=lambda _name, _value: None,
    )
    layers = {}
    adapter = MiniMaxH3LoRAAdapter(
        trainer, lora_rank=1, lora_alpha=1, lora_dtype=torch.float32
    )
    target_count = adapter.apply_lora_to_unet(layers)
    _initialise_lora(layers)
    if args.blocks_to_swap:
        minimax_h3_ops.setup_block_swap(trainer)
        trainer._probe_gradients = {}
        trainer._probe_update_handles = []
        for name, parameter in _trainable(layers):
            def capture_and_clear(tensor, key=name):
                trainer._probe_gradients[key] = tensor.grad.detach().cpu().clone()
                tensor.grad = None
            trainer._probe_update_handles.append(
                parameter.register_post_accumulate_grad_hook(capture_and_clear)
            )
        trainer.layer_offload_conductor.register_optimizer_hooks()
    inputs = _build_inputs(args.clip)

    captured = {}
    replicated = {}
    timings = {}
    enabled_arms = {
        "both": (False, True),
        "off": (False,),
        "on": (True,),
    }[args.arm]
    try:
        for enabled in enabled_arms:
            captured[enabled] = _execute(
                trainer, layers, inputs, enabled, seed=8181, capture=True
            )
            if trainer.layer_offload_conductor is not None:
                trainer.layer_offload_conductor.abort_step()
            samples = [captured[enabled]["elapsed_seconds"]]
            for repeat in range(1, args.repeats):
                result = _execute(
                    trainer, layers, inputs, enabled, seed=8181, capture=repeat == 1
                )
                if trainer.layer_offload_conductor is not None:
                    trainer.layer_offload_conductor.abort_step()
                samples.append(result["elapsed_seconds"])
                if repeat == 1:
                    replicated[enabled] = result
            timings[enabled] = {
                "median_seconds": statistics.median(samples),
                "p95_seconds": _p95(samples),
                "samples_seconds": samples,
            }
    except torch.OutOfMemoryError as exc:
        failure = {
            "probe": "minimax_h3_activation_dispatch",
            "status": "blocked_oom",
            "checkpoint": layout["dit"],
            "variant": layout["variant"],
            "clip": args.clip,
            "geometry": CLIPS[args.clip],
            "arm": args.arm,
            "failed_activation_dispatch": bool(enabled),
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "blocks_to_swap": args.blocks_to_swap,
            "device": torch.cuda.get_device_name(0),
            "entry_free_gib": free / GIB,
            "total_gib": total / GIB,
            "allocator_fraction": MEMORY_FRACTION,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
            "error": str(exc),
        }
        encoded = json.dumps(failure, indent=2)
        print(encoded)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
        return 5

    comparison = None
    optimizer_comparison = None
    if args.arm == "both":
        comparison = _compare_gradients(
            captured[False]["gradients"], captured[True]["gradients"]
        )
        optimizer_comparison = _compare_gradients(
            _optimizer_smoke(layers, captured[False]["gradients"]),
            _optimizer_smoke(layers, captured[True]["gradients"]),
        )
    repeatability = {}
    for enabled, label in ((False, "off"), (True, "on")):
        if enabled not in replicated:
            repeatability[label] = None
            continue
        gradient_repeat = _compare_gradients(
            captured[enabled]["gradients"], replicated[enabled]["gradients"]
        )
        optimizer_repeat = _compare_gradients(
            _optimizer_smoke(layers, captured[enabled]["gradients"]),
            _optimizer_smoke(layers, replicated[enabled]["gradients"]),
        )
        repeatability[label] = {
            "gradient_comparison": gradient_repeat,
            "optimizer_comparison": optimizer_repeat,
        }
    payload = {
        "probe": "minimax_h3_activation_dispatch",
        "scope": "real_transformer_production_lora_and_train_step_cached_inputs",
        "checkpoint": layout["dit"],
        "variant": layout["variant"],
        "clip": args.clip,
        "geometry": CLIPS[args.clip],
        "canvas_pixels": [384, 640],
        "prompt_tokens": 32,
        "audio_present": True,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "blocks_to_swap": args.blocks_to_swap,
        "lora_rank": 1,
        "lora_targets": target_count,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "entry_free_gib": free / GIB,
        "total_gib": total / GIB,
        "entry_host_available_gib": host_available,
        "allocator_fraction": MEMORY_FRACTION,
        "threshold_bytes": THRESHOLD_BYTES,
        "transformer_config": {
            "num_layers": config["num_layers"],
            "hidden_size": config["hidden_size"],
            "ffn_dim": config["ffn_dim"],
        },
        "arm": args.arm,
        "off": _without_gradients(captured[False]) if False in captured else None,
        "on": _without_gradients(captured[True]) if True in captured else None,
        "comparison": comparison,
        "optimizer_smoke": {
            "optimizer": "AdamW",
            "learning_rate": 1.0e-4,
            "comparison": optimizer_comparison,
        },
        "same_mode_repeatability": repeatability,
        "timing_off": timings.get(False),
        "timing_on": timings.get(True),
    }
    payload["loss_exact"] = (
        payload["off"]["loss"] == payload["on"]["loss"]
        if args.arm == "both"
        else None
    )
    arm_valid = all(result["finite"] for result in captured.values())
    if True in captured:
        arm_valid = arm_valid and captured[True]["offloaded_bytes"] > 0
    payload["passed"] = (
        target_count == 300
        and arm_valid
        and (args.arm != "both" or payload["loss_exact"])
        and (comparison is None or comparison["within_bf16_tolerance"])
        and (
            optimizer_comparison is None
            or optimizer_comparison["within_bf16_tolerance"]
        )
        and all(
            item is None
            or (
                item["gradient_comparison"]["within_bf16_tolerance"]
                and item["optimizer_comparison"]["within_bf16_tolerance"]
            )
            for item in repeatability.values()
        )
    )
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded + "\n", encoding="utf-8")
    for handle in getattr(trainer, "_probe_update_handles", ()):
        handle.remove()
    if trainer.layer_offload_conductor is not None:
        trainer.layer_offload_conductor.cleanup()
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
