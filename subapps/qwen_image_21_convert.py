"""Build SushiUI Qwen-Image 2.1 original and INT8 ConvRot artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.models.common.convrot_export import convrot_marker_fields, rotate_and_quantize
from core.models.common.quantized_checkpoint_guard import encode_comfy_quant_marker
from core.models.common.quantized_export import ShardWriter
from core.models.qwen_image_21.artifact import artifact_metadata
from core.models.qwen_image_21.vendor import QwenImage21Transformer2DModel


def _config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return {key: value for key, value in json.load(handle).items() if not key.startswith("_")}


def _source_files(component_dir: Path, basename: str) -> list[Path]:
    index_path = component_dir / f"{basename}.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        return [component_dir / name for name in sorted(set(weight_map.values()))]
    single = component_dir / f"{basename}.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"No {basename} safetensors under {component_dir}")


def _iter_tensors(files: list[Path]):
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


def _linear_names(component: str, config: dict) -> set[str]:
    from accelerate import init_empty_weights

    with init_empty_weights():
        if component == "transformer":
            model = QwenImage21Transformer2DModel(**config)
        else:
            from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

            model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**config))
    return {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}


def _write_component(
    files: list[Path],
    output: Path,
    component: str,
    config: dict,
    *,
    convrot: bool,
    device: str,
    max_shard_bytes: int,
) -> str:
    variant = "int8_convrot" if convrot else "bf16"
    writer = ShardWriter(str(output), artifact_metadata(component, config, variant=variant), max_shard_bytes)
    linears = _linear_names(component, config) if convrot else set()
    marker = encode_comfy_quant_marker(convrot_marker_fields())
    quantized = 0
    excluded = 0
    try:
        for key, tensor in _iter_tensors(files):
            stem = key[: -len(".weight")] if key.endswith(".weight") else ""
            if stem in linears and tensor.ndim == 2:
                if tensor.shape[1] % 256:
                    excluded += 1
                    writer.add(key, tensor.contiguous())
                    continue
                qweight, scale = rotate_and_quantize(tensor.to(device))
                writer.add(key, qweight.cpu().contiguous())
                writer.add(f"{stem}.weight_scale", scale.cpu().contiguous())
                writer.add(f"{stem}.comfy_quant", marker.clone())
                quantized += 1
            else:
                writer.add(key, tensor.contiguous())
        result = writer.close()
    except BaseException:
        writer.abort()
        raise
    print(f"[{component}] wrote {result}; ConvRot={quantized}, excluded={excluded}")
    return os.path.relpath(result, output.parent).replace("\\", "/")


def convert(source: str, output_root: str, *, convrot: bool, device: str, max_shard_gb: float) -> Path:
    source_path = Path(source).resolve()
    variant = "int8_convrot" if convrot else "original"
    output = Path(output_root).resolve() / variant
    output.mkdir(parents=True, exist_ok=True)
    transformer_config = _config(source_path / "transformer" / "config.json")
    text_encoder_config = _config(source_path / "text_encoder" / "config.json")
    vae_config = _config(source_path / "vae" / "config.json")
    max_shard_bytes = int(max_shard_gb * 1024**3)

    transformer = _write_component(
        _source_files(source_path / "transformer", "diffusion_pytorch_model"),
        output / f"qwen_image_2.1_{variant}.safetensors",
        "transformer",
        transformer_config,
        convrot=convrot,
        device=device,
        max_shard_bytes=max_shard_bytes,
    )
    text_encoder = _write_component(
        _source_files(source_path / "text_encoder", "model"),
        output / f"qwen3vl_8b_{variant}.safetensors",
        "text_encoder",
        text_encoder_config,
        convrot=convrot,
        device=device,
        max_shard_bytes=max_shard_bytes,
    )
    vae = _write_component(
        _source_files(source_path / "vae", "diffusion_pytorch_model"),
        output / "qwen_image_2.1_vae_bf16.safetensors",
        "vae",
        vae_config,
        convrot=False,
        device=device,
        max_shard_bytes=max_shard_bytes,
    )

    shutil.copytree(source_path / "processor", output / "processor", dirs_exist_ok=True)
    shutil.copytree(source_path / "scheduler", output / "scheduler", dirs_exist_ok=True)
    shutil.copy2(source_path / "text_encoder" / "config.json", output / "text_encoder_config.json")
    manifest = {
        "model_type": "qwen_image_21",
        "format_version": "1",
        "variant": variant,
        "components": {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "vae": vae,
            "processor": "processor",
            "scheduler": "scheduler",
            "text_encoder_config": "text_encoder_config.json",
        },
        "transformer_config": transformer_config,
        "vae_config": vae_config,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", choices=("original", "int8_convrot"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-shard-gb",
        type=float,
        default=100.0,
        help="Safety ceiling per output file; the default keeps each Qwen component in one safetensors file.",
    )
    args = parser.parse_args()
    result = convert(
        args.source,
        args.output_root,
        convrot=args.variant == "int8_convrot",
        device=args.device,
        max_shard_gb=args.max_shard_gb,
    )
    print(result)


if __name__ == "__main__":
    main()
