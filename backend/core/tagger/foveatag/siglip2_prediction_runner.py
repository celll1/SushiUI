"""Run current NaFlex checkpoint predictions over a FoveaTag benchmark ladder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image

from ..siglip2_inference_manager import SigLIP2InferenceManager
from .benchmark import PREDICTION_FORMAT_VERSION, BenchmarkItem, load_benchmark_manifest


_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def resize_benchmark_image(
    image: Image.Image,
    item: BenchmarkItem,
    resolution: str,
    width: int,
    height: int,
) -> Image.Image:
    if resolution == "native":
        return image.copy()
    return image.resize((width, height), _BICUBIC)


def run_siglip2_predictions(
    manifest_path: Path | str,
    checkpoint: Path | str,
    vocabulary: Path | str,
    output_dir: Path | str,
    *,
    patch_budgets: Sequence[int],
) -> list[Path]:
    manifest = load_benchmark_manifest(manifest_path)
    if not patch_budgets or any(budget <= 0 for budget in patch_budgets):
        raise ValueError("patch_budgets must contain positive values")
    manager = SigLIP2InferenceManager()
    manager.load_model(str(checkpoint), str(vocabulary))
    target_tags = sorted({tag for item in manifest.items for tag in item.positive_tags})
    tag_indices = {
        tag: manager.vocabulary["tag_to_idx"].get(tag)
        for tag in target_tags
    }
    present_tags = [tag for tag, index in tag_indices.items() if index is not None]
    present_indices = torch.tensor(
        [tag_indices[tag] for tag in present_tags], device=manager.device, dtype=torch.long
    )
    run_keys: Sequence[int | str] = patch_budgets if manager.is_naflex else ("fixed",)
    rows = {key: [] for key in run_keys}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(manager.device)
    try:
        for item in manifest.items:
            decode_started = time.perf_counter()
            with Image.open(item.image_path) as source:
                source_image = source.convert("RGB")
                source_image.load()
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            for resolution, width, height in manifest.resolution_ladder(item):
                resize_started = time.perf_counter()
                image = resize_benchmark_image(
                    source_image, item, resolution, width, height
                )
                resize_ms = (time.perf_counter() - resize_started) * 1000.0
                for key in run_keys:
                    preprocess_started = time.perf_counter()
                    processor_kwargs = {"images": image, "return_tensors": "pt"}
                    if manager.is_naflex:
                        processor_kwargs["max_num_patches"] = key
                    inputs = manager.processor(**processor_kwargs)
                    preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
                    pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
                    attention_mask = inputs.get("pixel_attention_mask")
                    spatial_shapes = inputs.get("spatial_shapes")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device, non_blocking=True)
                    if spatial_shapes is not None:
                        spatial_shapes = spatial_shapes.to(device, non_blocking=True)
                    torch.cuda.synchronize(device)
                    forward_started = time.perf_counter()
                    with torch.no_grad(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        logits = manager.model(
                            pixel_values, attention_mask, spatial_shapes
                        )[0]
                    probabilities = logits.sigmoid()
                    torch.cuda.synchronize(device)
                    forward_ms = (time.perf_counter() - forward_started) * 1000.0
                    selected = probabilities.index_select(0, present_indices).float().cpu().tolist()
                    scores = dict(zip(present_tags, selected))
                    for tag, index in tag_indices.items():
                        if index is None:
                            scores[tag] = 0.0
                    rows[key].append({
                        "sample_id": item.sample_id,
                        "resolution": resolution,
                        "scores": scores,
                        "stage_ms": {
                            "decode": decode_ms,
                            "resize": resize_ms,
                            "preprocess": preprocess_ms,
                            "forward": forward_ms,
                        },
                    })
        written = []
        for key in run_keys:
            is_fixed = key == "fixed"
            payload = {
                "format_version": PREDICTION_FORMAT_VERSION,
                "model_name": "siglip2-fixed" if is_fixed else f"siglip2-naflex-p{key}",
                "checkpoint": str(Path(checkpoint).resolve()),
                "patch_budget": None if is_fixed else key,
                "missing_tags": [tag for tag, index in tag_indices.items() if index is None],
                "predictions": rows[key],
            }
            filename = "siglip2_fixed_predictions.json" if is_fixed else f"siglip2_naflex_p{key}_predictions.json"
            path = output_dir / filename
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(path)
        return written
    finally:
        manager.unload()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-budgets", type=int, nargs="+", default=(256, 512, 1024))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    paths = run_siglip2_predictions(
        args.manifest,
        args.checkpoint,
        args.vocabulary,
        args.output_dir,
        patch_budgets=args.patch_budgets,
    )
    print(json.dumps({"outputs": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
