"""Compare Qwen-Image 2.1 full and complete-coverage partitioned LoRA steps.

The probe loads only the transformer, uses synthetic cached latents/text, and
reports predictions, functional LoRA-update agreement, CUDA time, and peak activation delta.
It deliberately excludes VAE/text-encoder/dataloader/optimizer time.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from core.models.qwen_image_21.artifact import load_manifest, load_transformer
from core.models.common.quantized_frozen_training import (
    enable_frozen_training_cached_backward,
    enable_frozen_training_fused,
)
from core.training.adapters.qwen_image_21_adapter import QwenImage21LoRAAdapter
from core.training.ops import qwen_image_21_ops


GB = 1024 ** 3


class _ProbeTrainer(SimpleNamespace):
    def log_extra_metric(self, key: str, value: float) -> None:
        self.metrics[key] = float(value)

    @staticmethod
    def _resolve_training_backend(backend: str) -> str:
        return backend


def _trainable_parameters(layers: dict[str, torch.nn.Module]):
    for layer in layers.values():
        yield from layer.trainable_parameters()


def _snapshot_gradients(layers: dict[str, torch.nn.Module]):
    snapshot = {}
    for name, layer in layers.items():
        if not hasattr(layer, "lora_down"):
            continue
        snapshot[name] = (
            layer.lora_down.weight.grad.detach().to("cpu", torch.bfloat16),
            layer.lora_up.weight.grad.detach().to("cpu", torch.bfloat16),
        )
    return snapshot


def _functional_update_agreement(
    reference,
    layers: dict[str, torch.nn.Module],
    *,
    seed: int,
    sketch_rows: int,
) -> dict[str, object]:
    """Compare the first-order effective LoRA weight update on Gaussian inputs.

    Raw factor-gradient cosine is gauge dependent. Acting with
    ``dB @ A + B @ dA`` on the same validation inputs measures the update in
    the wrapped Linear's output space without materialising full 4096² weights.
    """
    total_dot = total_ref_sq = total_candidate_sq = total_difference_sq = 0.0
    layer_cosines = []
    layer_relative_l2 = []
    for index, (name, layer) in enumerate(layers.items()):
        if name not in reference:
            continue
        ref_down_cpu, ref_up_cpu = reference[name]
        device = layer.lora_down.weight.device
        generator = torch.Generator(device=device).manual_seed(seed + 1009 * index)
        x = torch.randn(
            sketch_rows,
            layer.lora_down.in_features,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        down = layer.lora_down.weight.detach().float()
        up = layer.lora_up.weight.detach().float()
        ref_down = ref_down_cpu.to(device=device, dtype=torch.float32)
        ref_up = ref_up_cpu.to(device=device, dtype=torch.float32)
        candidate_down = layer.lora_down.weight.grad.detach().float()
        candidate_up = layer.lora_up.weight.grad.detach().float()
        scale = float(layer.scale)
        reference_action = (
            (x @ down.T) @ ref_up.T + (x @ ref_down.T) @ up.T
        ) * scale
        candidate_action = (
            (x @ down.T) @ candidate_up.T + (x @ candidate_down.T) @ up.T
        ) * scale
        dot = float(torch.sum(reference_action * candidate_action))
        ref_sq = float(torch.sum(reference_action * reference_action))
        candidate_sq = float(torch.sum(candidate_action * candidate_action))
        difference_sq = float(torch.sum((reference_action - candidate_action) ** 2))
        denominator = math.sqrt(ref_sq * candidate_sq)
        if denominator:
            layer_cosines.append(dot / denominator)
        if ref_sq:
            layer_relative_l2.append(math.sqrt(difference_sq / ref_sq))
        total_dot += dot
        total_ref_sq += ref_sq
        total_candidate_sq += candidate_sq
        total_difference_sq += difference_sq
        del x, ref_down, ref_up, reference_action, candidate_action
    denominator = math.sqrt(total_ref_sq * total_candidate_sq)
    return {
        "output_action_cosine": total_dot / denominator if denominator else float("nan"),
        "output_action_relative_l2": (
            math.sqrt(total_difference_sq / total_ref_sq)
            if total_ref_sq else float("nan")
        ),
        "reference_action_l2": math.sqrt(total_ref_sq),
        "candidate_action_l2": math.sqrt(total_candidate_sq),
        "layer_cosine": _summarize(layer_cosines),
        "layer_relative_l2": _summarize(layer_relative_l2),
        "sketch_rows_per_layer": sketch_rows,
    }


def _tensor_agreement(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    left = reference.float().reshape(-1)
    right = candidate.float().reshape(-1)
    difference = left - right
    ref_sq = float(torch.sum(left * left))
    candidate_sq = float(torch.sum(right * right))
    denominator = math.sqrt(ref_sq * candidate_sq)
    return {
        "cosine": float(torch.sum(left * right)) / denominator if denominator else float("nan"),
        "relative_l2": math.sqrt(float(torch.sum(difference * difference)) / ref_sq),
        "mean_absolute_error": float(torch.mean(torch.abs(difference))),
        "root_mean_square_error": math.sqrt(float(torch.mean(difference * difference))),
    }


@torch.no_grad()
def _prediction_agreement(
    trainer: _ProbeTrainer,
    *,
    mode: str,
    latents: torch.Tensor,
    encoder_features: torch.Tensor,
    encoder_mask: torch.Tensor,
    timesteps: torch.Tensor,
    latent_h: int,
    latent_w: int,
    full_prediction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float] | None]:
    from core.training.qwen_partition import flatten_region, full_canvas_position_ids

    noise = trainer._active_mnt_noise.to(latents)
    sigma = timesteps.to(device=latents.device, dtype=latents.dtype)
    noisy = (1 - sigma[:, None, None]) * latents + sigma[:, None, None] * noise
    batch, tokens, channels = noisy.shape
    prefix_mask = torch.zeros(
        encoder_features.shape[:2], dtype=torch.bool, device=latents.device
    )

    def forward(tile, height, width, positions=None):
        image_mask = torch.cat(
            [
                prefix_mask,
                torch.ones(batch, tile.shape[1] // 4, dtype=torch.bool, device=latents.device),
            ],
            dim=1,
        )
        with torch.autocast(device_type=latents.device.type, dtype=latents.dtype):
            residual = None
            global_adapter = getattr(
                trainer.transformer, "qwen_partition_global_adapter", None
            )
            if global_adapter is not None and positions is not None:
                residual = global_adapter(noisy_grid, active_region.input)
            return trainer.transformer(
                hidden_states=tile,
                timestep=sigma,
                encoder_hidden_states=encoder_features,
                encoder_hidden_states_mask=encoder_mask,
                img_shapes=[[(1, height, width)]] * batch,
                img_mask=image_mask,
                target_spatial_position_ids=positions,
                target_input_residual=residual,
                return_dict=False,
            )[0][:, -tile.shape[1]:]

    if mode == "full":
        prediction = forward(noisy, latent_h, latent_w)
        return prediction.detach(), None

    trainer.config["qwen_partition_fixed_count"] = int(mode)
    plan = qwen_image_21_ops._partition_plan(trainer, latent_h, latent_w)
    noisy_grid = noisy.reshape(batch, latent_h, latent_w, channels)
    stitched = torch.empty_like(noisy_grid)
    for region in plan.regions:
        active_region = region
        tile = flatten_region(noisy_grid, region.input)
        tile_prediction = forward(
            tile,
            region.input.height,
            region.input.width,
            full_canvas_position_ids(latent_h, latent_w, region.input),
        ).reshape(batch, region.input.height, region.input.width, channels)
        local = region.core_in_input
        stitched[
            :, region.core.top : region.core.bottom, region.core.left : region.core.right
        ] = tile_prediction[
            :, local.top : local.bottom, local.left : local.right
        ]
    prediction = stitched.reshape(batch, tokens, channels)
    return prediction.detach(), _tensor_agreement(full_prediction, prediction)


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def _zero_grad(parameters: list[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.grad = None


def _distill_global_adapter(
    trainer: _ProbeTrainer,
    layers: dict[str, torch.nn.Module],
    *,
    steps: int,
    learning_rate: float,
    latents: torch.Tensor,
    encoder_features: torch.Tensor,
    encoder_mask: torch.Tensor,
    timesteps: torch.Tensor,
    latent_h: int,
    latent_w: int,
) -> dict[str, object] | None:
    if steps <= 0:
        return None
    from core.training.qwen_partition import flatten_region, full_canvas_position_ids

    adapter = layers["qwen_partition_global_adapter"]
    ordinary = [layer for name, layer in layers.items() if name != "qwen_partition_global_adapter"]
    for layer in ordinary:
        layer.requires_grad_(False)
    noise = trainer._active_mnt_noise.to(latents)
    sigma = timesteps.to(latents)
    noisy = (1 - sigma[:, None, None]) * latents + sigma[:, None, None] * noise
    batch, tokens, channels = noisy.shape
    grid = noisy.reshape(batch, latent_h, latent_w, channels)
    prefix_mask = torch.zeros(
        encoder_features.shape[:2], dtype=torch.bool, device=latents.device
    )

    def run_full():
        image_mask = torch.cat(
            [prefix_mask, torch.ones(batch, tokens // 4, dtype=torch.bool, device=latents.device)],
            dim=1,
        )
        return trainer.transformer(
            hidden_states=noisy, timestep=sigma, encoder_hidden_states=encoder_features,
            encoder_hidden_states_mask=encoder_mask,
            img_shapes=[[(1, latent_h, latent_w)]] * batch, img_mask=image_mask,
            return_dict=False,
        )[0][:, -tokens:]

    with torch.no_grad(), torch.autocast(device_type=latents.device.type, dtype=latents.dtype):
        teacher = run_full().detach()

    plan = qwen_image_21_ops._partition_plan(trainer, latent_h, latent_w)

    def run_partitioned():
        stitched = torch.empty_like(grid)
        for region in plan.regions:
            tile = flatten_region(grid, region.input)
            image_mask = torch.cat(
                [prefix_mask, torch.ones(batch, tile.shape[1] // 4, dtype=torch.bool, device=latents.device)],
                dim=1,
            )
            prediction = trainer.transformer(
                hidden_states=tile, timestep=sigma, encoder_hidden_states=encoder_features,
                encoder_hidden_states_mask=encoder_mask,
                img_shapes=[[(1, region.input.height, region.input.width)]] * batch,
                img_mask=image_mask,
                target_spatial_position_ids=full_canvas_position_ids(
                    latent_h, latent_w, region.input, device=latents.device
                ),
                target_input_residual=adapter(grid, region.input),
                return_dict=False,
            )[0][:, -tile.shape[1]:].reshape(
                batch, region.input.height, region.input.width, channels
            )
            local = region.core_in_input
            stitched[:, region.core.top:region.core.bottom, region.core.left:region.core.right] = (
                prediction[:, local.top:local.bottom, local.left:local.right]
            )
        return stitched.reshape(batch, tokens, channels)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=0)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=latents.device.type, dtype=latents.dtype):
            prediction = run_partitioned()
            loss = torch.nn.functional.mse_loss(prediction.float(), teacher.float())
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    for layer in ordinary:
        layer.requires_grad_(True)
    return {
        "steps": steps,
        "learning_rate": learning_rate,
        "initial_mse": losses[0],
        "final_mse": losses[-1],
        "mse_reduction": 1.0 - losses[-1] / losses[0] if losses[0] else 0.0,
    }


def _run_step(
    trainer: _ProbeTrainer,
    parameters: list[torch.nn.Parameter],
    *,
    mode: str,
    latents: torch.Tensor,
    encoder_features: torch.Tensor,
    encoder_mask: torch.Tensor,
    timesteps: torch.Tensor,
    latent_h: int,
    latent_w: int,
) -> dict[str, float]:
    _zero_grad(parameters)
    trainer.metrics = {}
    trainer.config["qwen_partition_training_enabled"] = mode != "full"
    if mode != "full":
        trainer.config["qwen_partition_fixed_count"] = int(mode)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(trainer.device)
    baseline = torch.cuda.memory_allocated(trainer.device)
    start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(trainer.device))
    if mode == "full":
        loss, _, _ = qwen_image_21_ops.train_step(
            trainer,
            latents,
            encoder_features,
            encoder_mask,
            timesteps=timesteps,
            latent_h=latent_h,
            latent_w=latent_w,
        )
        forward_end.record(torch.cuda.current_stream(trainer.device))
        loss.backward()
        end.record(torch.cuda.current_stream(trainer.device))
        end.synchronize()
        loss_value = float(loss.detach())
        forward_ms = float(start.elapsed_time(forward_end))
        backward_ms = float(forward_end.elapsed_time(end))
    else:
        loss_value, _, _ = qwen_image_21_ops.train_step_partitioned_backward(
            trainer,
            latents=latents,
            encoder_features=encoder_features,
            encoder_mask=encoder_mask,
            timesteps=timesteps,
            latent_h=latent_h,
            latent_w=latent_w,
            backward_scale=1.0,
        )
        end.record(torch.cuda.current_stream(trainer.device))
        end.synchronize()
        forward_ms = trainer.metrics.get("qwen_partition_forward_ms", float("nan"))
        backward_ms = trainer.metrics.get("qwen_partition_backward_ms", float("nan"))
    peak = torch.cuda.max_memory_allocated(trainer.device)
    return {
        "loss": loss_value,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "step_ms": float(start.elapsed_time(end)),
        "baseline_allocated_gb": baseline / GB,
        "peak_allocated_gb": peak / GB,
        "peak_step_delta_gb": (peak - baseline) / GB,
        **trainer.metrics,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("qwen_partition_training probe requires CUDA")
    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    manifest = load_manifest(args.model)
    transformer, variant = load_transformer(
        manifest.transformer, manifest.transformer_config, dtype
    )
    transformer.requires_grad_(False).train().to(device)
    transformer.enable_gradient_checkpointing()
    transformer._training_gradient_checkpointing_blocks = args.checkpoint_blocks
    if variant != "int8_convrot":
        fused_layers, backward_cache_bytes = 0, 0
        convrot_policy = "dense_bf16"
    elif args.convrot_backward == "cached":
        fused_layers, backward_cache_bytes = enable_frozen_training_cached_backward(
            transformer, dtype=dtype, label="Qwen partition probe"
        )
        convrot_policy = "cached"
    else:
        fused_layers = enable_frozen_training_fused(
            transformer, label="Qwen partition probe"
        )
        backward_cache_bytes = 0
        convrot_policy = "transient"

    trainer = _ProbeTrainer(
        transformer=transformer,
        device=device,
        training_dtype=dtype,
        mixed_precision=True,
        use_grad_scaler=False,
        config={
            "qwen_partition_training_enabled": args.global_adapter,
            "qwen_partition_mode": "fixed",
            "qwen_partition_fixed_count": 2,
            "qwen_partition_halo_tokens": args.halo,
            "qwen_partition_split_ratio_min": args.split_ratio_min,
            "qwen_partition_split_ratio_max": args.split_ratio_max,
            "qwen_partition_seed": args.seed,
            "qwen_partition_gradient_checkpointing_blocks": args.partition_checkpoint_blocks,
            "qwen_partition_profile": True,
            "qwen_full_kv_query_chunk_tokens": args.query_chunk_tokens,
            "qwen_partition_global_adapter_enabled": args.global_adapter,
            "qwen_partition_global_rank": args.global_rank,
            "qwen_partition_global_tokens": args.global_tokens,
        },
        metrics={},
        log_prefix="[QwenPartitionProbe]",
        _current_epoch=args.epoch,
        _current_batch_position=args.occurrence,
        _active_mnt_noise=None,
        adapter_algorithm="lora",
        weight_decompose=False,
        adapter_config={},
        learning_rate=1e-4,
        unet_lr=None,
    )
    qwen_image_21_ops.setup_attention_backend(trainer, args.attention)
    layers: dict[str, torch.nn.Module] = {}
    adapter = QwenImage21LoRAAdapter(
        trainer, lora_rank=args.rank, lora_alpha=args.alpha, lora_dtype=dtype
    )
    adapter.apply_lora_to_unet(layers)
    parameters = list(_trainable_parameters(layers))
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if args.lora_init_std > 0:
        with torch.no_grad():
            for parameter in parameters:
                parameter.normal_(mean=0.0, std=args.lora_init_std, generator=generator)

    tokens = args.latent_height * args.latent_width
    latents = torch.randn(
        args.batch, tokens, manifest.transformer_config["in_channels"],
        device=device, dtype=dtype, generator=generator,
    )
    encoder_features = torch.randn(
        args.batch, args.text_tokens, manifest.transformer_config["context_in_dim"],
        device=device, dtype=dtype, generator=generator,
    )
    encoder_mask = torch.ones(
        args.batch, args.text_tokens, device=device, dtype=torch.long
    )
    timesteps = torch.full(
        (args.batch,), args.sigma, device=device, dtype=dtype
    )
    trainer._active_mnt_noise = torch.randn(
        latents.shape, device=device, dtype=dtype, generator=generator
    )

    distillation = _distill_global_adapter(
        trainer,
        layers,
        steps=args.global_distill_steps if args.global_adapter else 0,
        learning_rate=args.global_distill_lr,
        latents=latents,
        encoder_features=encoder_features,
        encoder_mask=encoder_mask,
        timesteps=timesteps,
        latent_h=args.latent_height,
        latent_w=args.latent_width,
    )

    modes = ["full", *[str(value) for value in args.partitions]]
    full_prediction, _ = _prediction_agreement(
        trainer,
        mode="full",
        latents=latents,
        encoder_features=encoder_features,
        encoder_mask=encoder_mask,
        timesteps=timesteps,
        latent_h=args.latent_height,
        latent_w=args.latent_width,
    )
    prediction_agreements = {}
    for mode in modes[1:]:
        _, prediction_agreements[mode] = _prediction_agreement(
            trainer,
            mode=mode,
            latents=latents,
            encoder_features=encoder_features,
            encoder_mask=encoder_mask,
            timesteps=timesteps,
            latent_h=args.latent_height,
            latent_w=args.latent_width,
            full_prediction=full_prediction,
        )
    del full_prediction

    raw: dict[str, list[dict[str, float]]] = {mode: [] for mode in modes}
    reference_gradients = None
    agreements: dict[str, dict[str, float]] = {}
    for mode in modes:
        for iteration in range(args.warmup + args.iterations):
            result = _run_step(
                trainer,
                parameters,
                mode=mode,
                latents=latents,
                encoder_features=encoder_features,
                encoder_mask=encoder_mask,
                timesteps=timesteps,
                latent_h=args.latent_height,
                latent_w=args.latent_width,
            )
            if iteration >= args.warmup:
                raw[mode].append(result)
        if mode == "full":
            reference_gradients = _snapshot_gradients(layers)
        else:
            agreements[mode] = _functional_update_agreement(
                reference_gradients,
                layers,
                seed=args.seed,
                sketch_rows=args.update_sketch_rows,
            )

    summary = {}
    full_loss = raw["full"][-1]["loss"]
    full_step = statistics.median(item["step_ms"] for item in raw["full"])
    full_peak = statistics.median(item["peak_step_delta_gb"] for item in raw["full"])
    for mode, samples in raw.items():
        step = _summarize([item["step_ms"] for item in samples])
        peak = _summarize([item["peak_step_delta_gb"] for item in samples])
        entry: dict[str, object] = {
            "loss": samples[-1]["loss"],
            "loss_absolute_delta_vs_full": abs(samples[-1]["loss"] - full_loss),
            "loss_relative_delta_vs_full": abs(samples[-1]["loss"] - full_loss) / abs(full_loss),
            "step_ms": step,
            "forward_ms": _summarize([item["forward_ms"] for item in samples]),
            "backward_ms": _summarize([item["backward_ms"] for item in samples]),
            "peak_step_delta_gb": peak,
            "peak_allocated_gb": _summarize(
                [item["peak_allocated_gb"] for item in samples]
            ),
            "step_speedup_vs_full": full_step / step["median"],
            "activation_delta_reduction_vs_full": (
                1.0 - peak["median"] / full_peak if full_peak else float("nan")
            ),
        }
        if mode != "full":
            entry["prediction_vs_full"] = prediction_agreements[mode]
            entry["functional_update_vs_full"] = agreements[mode]
            entry["partition_count"] = samples[-1]["qwen_partition_count"]
            entry["largest_input_tokens"] = samples[-1]["qwen_partition_largest_tokens"]
            entry["total_input_tokens"] = samples[-1]["qwen_partition_total_input_tokens"]
            entry["checkpoint_blocks"] = samples[-1]["qwen_partition_checkpoint_blocks"]
        summary[mode] = entry

    return {
        "device": torch.cuda.get_device_name(device),
        "model": str(args.model),
        "variant": variant,
        "attention": args.attention,
        "query_chunk_tokens": args.query_chunk_tokens,
        "shape": {
            "batch": args.batch,
            "latent_height": args.latent_height,
            "latent_width": args.latent_width,
            "image_tokens": tokens,
            "text_tokens": args.text_tokens,
        },
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "target_layers": len(layers),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "dtype": str(parameters[0].dtype),
        },
        "global_adapter": {
            "enabled": args.global_adapter,
            "rank": args.global_rank,
            "summary_tokens": args.global_tokens,
            "trainable_parameters": (
                sum(
                    parameter.numel()
                    for parameter in layers.get(
                        "qwen_partition_global_adapter", []
                    ).parameters()
                )
                if args.global_adapter else 0
            ),
            "distillation": distillation,
        },
        "convrot": {
            "backward_weight_policy": convrot_policy,
            "enabled_layers": fused_layers,
            "backward_cache_gb": backward_cache_bytes / GB,
        },
        "checkpoint_blocks": {
            "full": args.checkpoint_blocks,
            "partition": (
                "auto"
                if args.partition_checkpoint_blocks is None
                else args.partition_checkpoint_blocks
            ),
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": summary,
        "limits": [
            "Synthetic cached latents/text; excludes VAE, text encoder, data loading, optimizer, and logging.",
            "Peak step delta subtracts retained model/LoRA/backward-cache allocation from the CUDA peak.",
            "Partitioned gradients are intentionally non-equivalent because target-target attention across cores is removed.",
            "Agreement is measured in prediction/output-update space; raw LoRA-factor gradient cosine is deliberately not reported.",
            "Detailed partition forward/backward timings synchronize each region and are diagnostic-only.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(r"M:\model\qwen21\int8_convrot"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=("native", "flash"), default="flash")
    parser.add_argument(
        "--convrot-backward", choices=("cached", "transient"), default="cached"
    )
    parser.add_argument("--query-chunk-tokens", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=64)
    parser.add_argument("--latent-width", type=int, default=64)
    parser.add_argument("--text-tokens", type=int, default=128)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument(
        "--lora-init-std", type=float, default=0.0,
        help="Override production LoRA initialization only when explicitly nonzero",
    )
    parser.add_argument("--update-sketch-rows", type=int, default=16)
    parser.add_argument("--checkpoint-blocks", type=int, default=16)
    parser.add_argument("--partition-checkpoint-blocks", type=int)
    parser.add_argument("--partitions", type=int, nargs="+", choices=(2, 4), default=(2, 4))
    parser.add_argument("--halo", type=int, default=0)
    parser.add_argument("--global-adapter", action="store_true")
    parser.add_argument("--global-rank", type=int, default=64)
    parser.add_argument("--global-tokens", type=int, default=16)
    parser.add_argument("--global-distill-steps", type=int, default=0)
    parser.add_argument("--global-distill-lr", type=float, default=1e-2)
    parser.add_argument("--split-ratio-min", type=float, default=0.35)
    parser.add_argument("--split-ratio-max", type=float, default=0.65)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--occurrence", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    parsed = _parser().parse_args()
    result = run(parsed)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if parsed.output is not None:
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        parsed.output.write_text(rendered + "\n", encoding="utf-8")
