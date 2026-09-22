"""Measure empty-prompt prediction drift on fixed Qwen training latents.

Run after a cooperative checkpoint save; this probe does not modify the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch
from safetensors import safe_open

from core.models.qwen_image_21.artifact import (
    load_manifest,
    load_text_encoder,
    load_transformer,
)
from core.models.qwen_image_21.vendor.pipeline import QwenImage21Pipeline
from core.training.adapters.qwen_image_21_adapter import QwenImage21LoRAAdapter
from core.training.arch.qwen_image_21 import QwenImage21ArchHandler
from core.training.ops import qwen_image_21_ops
from core.training.qwen_partition import flatten_region, full_canvas_position_ids


def _blank_encoding(manifest, device):
    from transformers import Qwen3VLProcessor

    encoder, _ = load_text_encoder(
        manifest.text_encoder, manifest.text_encoder_config, torch.bfloat16
    )
    encoder = encoder.eval().to(device)
    processor = Qwen3VLProcessor.from_pretrained(manifest.processor)
    pipeline = QwenImage21Pipeline(
        scheduler=None, vae=None, text_encoder=encoder,
        processor=processor, transformer=None,
    )
    with torch.inference_mode():
        features, mask, _ = pipeline.encode_prompt(prompt="", device=device)
    if mask is None:
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
    features, mask = features.detach().cpu(), mask.detach().cpu()
    del pipeline, encoder, processor
    torch.cuda.empty_cache()
    return features, mask


def _load_adapter_state(path, layers):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for stem, layer in layers.items():
            if stem == "qwen_partition_global_adapter":
                prefix = f"{stem}."
                state = {
                    key[len(prefix):]: handle.get_tensor(key)
                    for key in handle.keys()
                    if key.startswith(prefix) and key != f"{stem}.alpha"
                }
                layer.load_state_dict(state, strict=True)
            else:
                layer.load_tensors({
                    name: handle.get_tensor(f"{stem}.{name}")
                    for name in layer.tensor_names()
                })


def _predict(transformer, trainer, clean, noise, sigma, features, mask, *, global_enabled):
    _, channels, height, width = clean.shape
    clean = clean.to(device=trainer.device, dtype=torch.bfloat16)
    noise = noise.to(device=trainer.device, dtype=torch.bfloat16)
    noisy = ((1 - sigma) * clean + sigma * noise).permute(0, 2, 3, 1).reshape(1, -1, channels)
    grid = noisy.reshape(1, height, width, channels)
    stitched = torch.empty_like(grid)
    plan = qwen_image_21_ops._partition_plan(trainer, height, width)
    timestep = torch.tensor([sigma], device=trainer.device, dtype=torch.bfloat16)
    features = features.to(device=trainer.device, dtype=torch.bfloat16)
    mask = mask.to(device=trainer.device)
    for region in plan.regions:
        tile = flatten_region(grid, region.input)
        image_mask = torch.cat((
            torch.zeros(features.shape[:2], dtype=torch.bool, device=trainer.device),
            torch.ones(1, tile.shape[1] // 4, dtype=torch.bool, device=trainer.device),
        ), dim=1)
        residual = (
            transformer.qwen_partition_global_adapter(grid, region.input)
            if global_enabled else None
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = transformer(
                hidden_states=tile, timestep=timestep,
                encoder_hidden_states=features,
                encoder_hidden_states_mask=mask,
                img_shapes=[[(1, region.input.height, region.input.width)]],
                img_mask=image_mask,
                target_spatial_position_ids=full_canvas_position_ids(
                    height, width, region.input, device=trainer.device
                ),
                target_input_residual=residual,
                return_dict=False,
            )[0][:, -tile.shape[1]:].reshape(
                1, region.input.height, region.input.width, channels
            )
        local = region.core_in_input
        stitched[:, region.core.top:region.core.bottom, region.core.left:region.core.right] = (
            prediction[:, local.top:local.bottom, local.left:local.right]
        )
    return stitched.permute(0, 3, 1, 2).float().cpu()


def _rms(tensor):
    return float(tensor.float().square().mean().sqrt())


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    manifest = load_manifest(args.model)
    print("Encoding the exact empty prompt used by training", flush=True)
    features, mask = _blank_encoding(manifest, device)
    print("Loading transformer", flush=True)
    transformer, variant = load_transformer(
        manifest.transformer, manifest.transformer_config, torch.bfloat16
    )
    if variant != "int8_convrot":
        raise ValueError(f"Expected int8_convrot, got {variant}")
    transformer = transformer.eval().to(device)
    trainer = SimpleNamespace(
        arch=QwenImage21ArchHandler(), transformer=transformer, device=device,
        config={
            "dit_partition_mode": "fixed",
            "dit_partition_fixed_count": 2,
            "dit_partition_halo_tokens": 0,
            "dit_partition_split_ratio_min": 0.35,
            "dit_partition_split_ratio_max": 0.65,
            "dit_partition_seed": 0,
            "dit_partition_global_adapter_enabled": True,
            "dit_partition_training_enabled": True,
            "dit_partition_global_rank": 64,
            "dit_partition_global_tokens": 16,
        },
        adapter_algorithm="lora", weight_decompose=False, adapter_config={},
        _current_epoch=0, _current_batch_position=0,
    )
    layers = {}
    adapter = QwenImage21LoRAAdapter(
        trainer, lora_rank=128, lora_alpha=128, lora_dtype=torch.float32
    )
    adapter.apply_lora_to_unet(layers)
    transformer.eval()
    samples = []
    for index, path in enumerate(args.debug_latents):
        data = torch.load(path, map_location="cpu", weights_only=True)
        clean = data["latents"].float()
        generator = torch.Generator(device="cpu").manual_seed(args.seed + index)
        noise = torch.randn(clean.shape, generator=generator)
        samples.append((str(path), clean, noise))
    outputs = {}
    for label, checkpoint, global_enabled in (
        ("base", None, False),
        ("step0", args.step0, True),
        ("trained_no_global", args.trained, False),
        ("trained", args.trained, True),
    ):
        print(f"Evaluating {label}", flush=True)
        if checkpoint:
            _load_adapter_state(checkpoint, layers)
        for stem, layer in layers.items():
            if stem != "qwen_partition_global_adapter":
                layer.set_adapter_strength(0.0 if checkpoint is None else 1.0)
        for path, clean, noise in samples:
            for sigma in args.sigmas:
                key = (path, sigma)
                with torch.inference_mode():
                    prediction = _predict(
                        transformer, trainer, clean, noise, sigma,
                        features, mask, global_enabled=global_enabled,
                    )
                outputs.setdefault(key, {})[label] = prediction
                print(f"  {Path(path).parent.name} sigma={sigma}: rms={_rms(prediction):.5f}", flush=True)
    records = []
    for (path, sigma), values in outputs.items():
        clean = next(item[1] for item in samples if item[0] == path)
        noise = next(item[2] for item in samples if item[0] == path)
        target = noise - clean
        base, step0, trained = (values[name] for name in ("base", "step0", "trained"))
        trained_no_global = values["trained_no_global"]
        change = trained - step0
        records.append({
            "sample": str(Path(path).parent.name), "sigma": sigma,
            "target_rms": _rms(target),
            "base_to_step0_rms": _rms(step0 - base),
            "base_to_trained_rms": _rms(trained - base),
            "training_drift_rms": _rms(change),
            "lora_only_drift_rms": _rms(trained_no_global - step0),
            "global_effect_rms": _rms(trained - trained_no_global),
            "training_drift_relative_to_target": _rms(change) / _rms(target),
            "hybrid_target_drift_rms": 0.5 * sigma * _rms(change),
            "base_mse": float((base - target).square().mean()),
            "step0_mse": float((step0 - target).square().mean()),
            "trained_mse": float((trained - target).square().mean()),
            "step0_target_gap_rms": _rms(target - step0),
            "trained_target_gap_rms": _rms(target - trained),
        })
    summary = []
    for sigma in args.sigmas:
        selected = [record for record in records if record["sigma"] == sigma]
        summary.append({
            "sigma": sigma,
            "samples": len(selected),
            **{
                key: mean(record[key] for record in selected)
                for key in (
                    "target_rms", "training_drift_rms", "lora_only_drift_rms",
                    "global_effect_rms", "hybrid_target_drift_rms",
                    "base_mse", "trained_mse", "step0_target_gap_rms",
                    "trained_target_gap_rms",
                )
            },
        })
    print(json.dumps({"summary": summary, "records": records}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--step0", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--debug-latents", type=Path, nargs="+", required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.2, 0.5, 0.8, 0.95])
    parser.add_argument("--seed", type=int, default=154)
    main(parser.parse_args())
