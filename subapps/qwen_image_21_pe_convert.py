"""Convert a Qwen-Image 2.1 PE checkpoint to one ConvRot INT8 artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch.nn as nn
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.models.common.convrot_export import convrot_marker_fields, rotate_and_quantize
from core.models.common.quantized_checkpoint_guard import encode_comfy_quant_marker
from core.models.common.quantized_export import ShardWriter


def _source_files(source: Path) -> list[Path]:
    index = source / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [source / name for name in sorted(set(weight_map.values()))]
    single = source / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"No model safetensors under {source}")


def convert(source: str, output: str, mode: str, device: str) -> Path:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(str(source_path), local_files_only=True)
    with init_empty_weights():
        model = Qwen3_5ForConditionalGeneration(config)
    linears = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    marker = encode_comfy_quant_marker(convrot_marker_fields())
    metadata = {
        "model_type": "qwen_image_21_prompt_enhancer",
        "format_version": "1",
        "mode": mode,
        "variant": "int8_convrot",
        "format": "pt",
    }
    writer = ShardWriter(
        str(output_path / f"qwen_image_2.1_pe_{mode}_int8_convrot.safetensors"),
        metadata,
        100 * 1024**3,
    )
    quantized = excluded = 0
    try:
        for file in _source_files(source_path):
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    stem = key.removesuffix(".weight") if key.endswith(".weight") else ""
                    if stem in linears and tensor.ndim == 2 and tensor.shape[1] % 256 == 0:
                        qweight, scale = rotate_and_quantize(tensor.to(device))
                        writer.add(key, qweight.cpu().contiguous())
                        writer.add(f"{stem}.weight_scale", scale.cpu().contiguous())
                        writer.add(f"{stem}.comfy_quant", marker.clone())
                        quantized += 1
                    else:
                        if stem in linears and tensor.ndim == 2:
                            excluded += 1
                        writer.add(key, tensor.contiguous())
        weight_result = Path(writer.close())
    except BaseException:
        writer.abort()
        raise

    for name in (
        "config.json", "generation_config.json", "chat_template.jinja",
        "processor_config.json", "tokenizer.json", "tokenizer_config.json",
        "system_prompt.txt",
    ):
        source_file = source_path / name
        if source_file.is_file():
            shutil.copy2(source_file, output_path / name)
    manifest = {
        "model_type": "qwen_image_21_prompt_enhancer",
        "format_version": "1",
        "mode": mode,
        "variant": "int8_convrot",
        "components": {
            "weights": weight_result.name,
            "config": ".",
            "processor": ".",
            "system_prompt": "system_prompt.txt",
        },
        "quantized_linears": quantized,
        "unquantized_linears": excluded,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{mode}] ConvRot={quantized}, excluded={excluded}, output={output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=("t2i", "i2i"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    convert(args.source, args.output, args.mode, args.device)


if __name__ == "__main__":
    main()
