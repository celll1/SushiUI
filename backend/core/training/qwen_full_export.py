"""Export a floating Qwen full-parameter checkpoint as a split ConvRot artifact."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

from core.models.common.convrot_export import convrot_marker_fields, rotate_and_quantize
from core.models.common.quantized_checkpoint_guard import encode_comfy_quant_marker
from core.models.common.quantized_export import DEFAULT_EXPORT_SHARD_BYTES, ShardWriter
from core.models.qwen_image_21.artifact import artifact_metadata, load_manifest


def _metadata(path: Path) -> dict[str, str]:
    if path.name.endswith(".safetensors.index.json"):
        with path.open(encoding="utf-8") as handle:
            return dict(json.load(handle).get("metadata") or {})
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _iter_tensors(path: Path):
    if path.name.endswith(".safetensors.index.json"):
        with path.open(encoding="utf-8") as handle:
            members = sorted(set(json.load(handle)["weight_map"].values()))
        files = [path.parent / member for member in members]
    else:
        files = [path]
    for member in files:
        with safe_open(str(member), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


def _linear_names(component: str, config: dict) -> set[str]:
    from accelerate import init_empty_weights
    from torch import nn
    from core.models.qwen_image_21.vendor import QwenImage21Transformer2DModel

    with init_empty_weights():
        if component == "transformer":
            model = QwenImage21Transformer2DModel(**config)
        else:
            from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

            model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**config))
    return {name for name, layer in model.named_modules() if isinstance(layer, nn.Linear)}


def _quantize_component(
    source: Path, destination: Path, component: str, config: dict,
    *, key_prefix: str = "", device: str = "cpu",
) -> str:
    linears = _linear_names(component, config)
    marker = encode_comfy_quant_marker(convrot_marker_fields())
    writer = ShardWriter(
        str(destination), artifact_metadata(component, config, variant="int8_convrot"),
        DEFAULT_EXPORT_SHARD_BYTES,
    )
    count = 0
    try:
        for source_key, tensor in _iter_tensors(source):
            if key_prefix:
                if not source_key.startswith(key_prefix):
                    continue
                key = source_key[len(key_prefix):]
            else:
                key = source_key
            stem = key[: -len(".weight")] if key.endswith(".weight") else ""
            if stem in linears and tensor.ndim == 2 and tensor.shape[1] % 256 == 0:
                packed, scale = rotate_and_quantize(tensor.to(device))
                writer.add(key, packed.cpu().contiguous())
                writer.add(f"{stem}.weight_scale", scale.cpu().contiguous())
                writer.add(f"{stem}.comfy_quant", marker.clone())
                count += 1
                del packed, scale
            else:
                writer.add(key, tensor.contiguous())
        if not count:
            raise ValueError(f"{component}: no eligible ConvRot Linears were found")
        result = Path(writer.close())
    except BaseException:
        writer.abort()
        raise
    return result.name


def _copy_component(source: Path, output: Path) -> str:
    if source.name.endswith(".safetensors.index.json"):
        with source.open(encoding="utf-8") as handle:
            members = set(json.load(handle)["weight_map"].values())
        for member in members:
            shutil.copy2(source.parent / member, output / member)
    shutil.copy2(source, output / source.name)
    return source.name


def export_qwen_full_checkpoint(
    checkpoint: str | Path, run_output_dir: str | Path, *, overwrite: bool = False,
    device: str = "cpu",
) -> Path:
    checkpoint = Path(checkpoint).resolve()
    run_output_dir = Path(run_output_dir).resolve()
    if checkpoint.parent != run_output_dir or not checkpoint.is_file():
        raise ValueError("Export source must be a checkpoint file inside this run")
    metadata = _metadata(checkpoint)
    if metadata.get("model_type") != "qwen_image_21" or metadata.get("component") not in {
        "transformer", "training_bundle"
    }:
        raise ValueError("Export source is not a Qwen full-parameter checkpoint")
    if not metadata.get("companion_path"):
        raise ValueError("Training checkpoint has no companion model path")
    companion = Path(metadata["companion_path"])
    manifest = load_manifest(str(companion))
    with (Path(manifest.root) / "manifest.json").open(encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    export_parent = run_output_dir / "exports"
    export_parent.mkdir(exist_ok=True)
    published = export_parent / "int8_convrot"
    if published.exists() and not overwrite:
        raise FileExistsError("INT8 ConvRot export already exists; choose overwrite")
    backup = export_parent / ".int8_convrot_previous"
    if backup.exists():
        raise FileExistsError(f"Previous export replacement is incomplete: {backup}")
    staging = Path(tempfile.mkdtemp(prefix=".int8_convrot_", dir=export_parent))
    try:
        transformer_config = json.loads(metadata.get("config") or "{}")
        transformer_name = _quantize_component(
            checkpoint, staging / "qwen_image_2.1_int8_convrot.safetensors",
            "transformer", transformer_config,
            key_prefix="transformer." if metadata["component"] == "training_bundle" else "",
            device=device,
        )
        with Path(manifest.text_encoder_config).open(encoding="utf-8") as handle:
            te_config = json.load(handle)
        if metadata["component"] == "training_bundle":
            te_name = _quantize_component(
                checkpoint, staging / "qwen3vl_8b_int8_convrot.safetensors",
                "text_encoder", te_config, key_prefix="text_encoder.", device=device,
            )
        elif _metadata(Path(manifest.text_encoder)).get("variant") == "int8_convrot":
            te_name = _copy_component(Path(manifest.text_encoder), staging)
        else:
            te_name = _quantize_component(
                Path(manifest.text_encoder),
                staging / "qwen3vl_8b_int8_convrot.safetensors",
                "text_encoder", te_config, device=device,
            )
        vae_name = _copy_component(Path(manifest.vae), staging)
        shutil.copy2(manifest.text_encoder_config, staging / "text_encoder_config.json")
        shutil.copytree(manifest.processor, staging / "processor")
        if manifest.scheduler and Path(manifest.scheduler).is_dir():
            shutil.copytree(manifest.scheduler, staging / "scheduler")
        manifest_data["variant"] = "int8_convrot"
        manifest_data["components"].update(
            transformer=transformer_name, text_encoder=te_name,
            vae=vae_name, text_encoder_config="text_encoder_config.json",
            processor="processor", scheduler="scheduler" if manifest.scheduler else None,
        )
        manifest_data["export_source_step"] = metadata.get("step")
        with (staging / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest_data, handle, indent=2)
        load_manifest(str(staging))
        if published.exists():
            os.replace(published, backup)
        try:
            os.replace(staging, published)
        except BaseException:
            if backup.exists():
                os.replace(backup, published)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return published
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
