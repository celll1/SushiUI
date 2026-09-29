"""Write registered resolution-ladder predictions for a joint FoveaTag artifact."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

import torch
from PIL import Image

from .artifact import FoveaArtifactConfig, load_inference_artifact
from .benchmark import PREDICTION_FORMAT_VERSION, load_benchmark_manifest
from .inference import predict_image
from .model import FoveaTagger
from .router import EvidenceRouter
from .siglip2_prediction_runner import resize_benchmark_image
from .supervised_prediction_runner import _artifact_vocabulary, _file_sha256


def run_routed_predictions(
    manifest_path: Path | str,
    model_path: Path | str,
    output_path: Path | str,
    *,
    inference_mode: str = "balanced",
    resolutions: Optional[Sequence[str]] = None,
    device: torch.device | str = "cuda",
) -> Path:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for routed FoveaTag benchmark prediction")
    manifest = load_benchmark_manifest(manifest_path)
    model_path = Path(model_path)
    metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = FoveaArtifactConfig.from_dict(metadata["config"])
    provenance = validate_joint_artifact(metadata)
    vocabulary = _artifact_vocabulary(metadata, config.num_tags)
    tag_to_idx = vocabulary["tag_to_idx"]
    if inference_mode not in config.inference_modes:
        raise ValueError(f"unknown FoveaTag inference mode: {inference_mode}")

    model = FoveaTagger(
        config.num_tags,
        encoder_config=config.encoder,
        feature_dim=config.feature_dim,
        evidence_slots_per_view=config.evidence_slots_per_view,
        pool_heads=config.pool_heads,
        temperature=config.temperature,
    )
    router = EvidenceRouter(
        feature_dim=config.encoder.widths[2],
        routing_dim=config.routing_dim,
        routing_slots=config.routing_slots,
    )
    load_inference_artifact(model_path, model, router=router)
    model.to(device).eval()
    router.to(device).eval()
    amp_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()

    target_tags = sorted({tag for item in manifest.items for tag in item.positive_tags})
    tag_indices = {
        tag: int(tag_to_idx[tag]) if tag in tag_to_idx else None
        for tag in target_tags
    }
    present_tags = [tag for tag, index in tag_indices.items() if index is not None]
    present_indices = torch.tensor(
        [tag_indices[tag] for tag in present_tags], dtype=torch.long
    )
    selected_resolutions = set(resolutions or ())
    known_resolutions = {
        resolution
        for item in manifest.items
        for resolution, _width, _height in manifest.resolution_ladder(item)
    }
    unknown = sorted(selected_resolutions - known_resolutions)
    if unknown:
        raise ValueError(f"requested resolutions are absent from the manifest: {unknown}")

    rows = []
    try:
        for item_index, item in enumerate(manifest.items):
            decode_started = time.perf_counter()
            with Image.open(item.image_path) as source:
                source_image = source.convert("RGB")
                source_image.load()
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            for resolution, width, height in manifest.resolution_ladder(item):
                if selected_resolutions and resolution not in selected_resolutions:
                    continue
                resize_started = time.perf_counter()
                image = resize_benchmark_image(source_image, item, resolution, width, height)
                resize_ms = (time.perf_counter() - resize_started) * 1000.0
                result = predict_image(
                    image,
                    model,
                    router,
                    config,
                    mode=inference_mode,
                    mixed_precision=amp_dtype,
                )
                selected = result.probabilities.index_select(0, present_indices).tolist()
                scores = dict(zip(present_tags, selected))
                scores.update((tag, 0.0) for tag, index in tag_indices.items() if index is None)
                rows.append({
                    "sample_id": item.sample_id,
                    "resolution": resolution,
                    "scores": scores,
                    "stage_ms": {
                        "decode": decode_ms,
                        "resize": resize_ms,
                        **result.stage_milliseconds,
                    },
                    "selected_boxes": [
                        {
                            "x": box.x, "y": box.y,
                            "width": box.width, "height": box.height,
                        }
                        for box in result.selected_boxes
                    ],
                })
            if (item_index + 1) % 100 == 0 or item_index + 1 == len(manifest.items):
                print(
                    f"[FoveaTagRouted] predicted {item_index + 1}/{len(manifest.items)} images",
                    flush=True,
                )
        payload = {
            "format_version": PREDICTION_FORMAT_VERSION,
            "model_name": f"foveatag-routed-{inference_mode}",
            "checkpoint": str(model_path.resolve()),
            "checkpoint_sha256": _file_sha256(model_path),
            "checkpoint_metadata_sha256": _file_sha256(metadata_path),
            "artifact_provenance": provenance,
            "inference_mode": inference_mode,
            "missing_tags": [tag for tag, index in tag_indices.items() if index is None],
            "runtime": {
                "device": str(device),
                "dtype": (
                    str(amp_dtype).removeprefix("torch.")
                    if amp_dtype is not None else "float32"
                ),
                "prediction_rows": len(rows),
                "wall_seconds": time.perf_counter() - run_started,
                "peak_allocated_mib": (
                    torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda" else None
                ),
                "peak_reserved_mib": (
                    torch.cuda.max_memory_reserved(device) / 2**20
                    if device.type == "cuda" else None
                ),
            },
            "predictions": rows,
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
        return output_path
    finally:
        del router, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def validate_joint_artifact(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("joint FoveaTag artifact has no provenance")
    if provenance.get("initialization") != "scratch" or provenance.get("external_weights") != []:
        raise ValueError("joint FoveaTag artifact does not have pure-scratch lineage")
    if provenance.get("stage") != "joint_routed_finetune":
        raise ValueError("artifact is not a joint routed FoveaTag checkpoint")
    utility = provenance.get("router_validation")
    if not isinstance(utility, Mapping):
        raise ValueError("joint artifact has no router validation")
    learned = utility.get("mean_utility_recovery")
    random_baseline = utility.get("random_mean_utility_recovery")
    if not isinstance(learned, (int, float)) or not isinstance(random_baseline, (int, float)):
        raise ValueError("joint artifact lacks router utility recovery")
    if learned <= random_baseline:
        raise ValueError("joint artifact router did not beat random views")
    threshold = provenance.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("joint artifact has no valid threshold")
    return provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inference-mode", default="balanced")
    parser.add_argument("--resolutions", nargs="+")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output = run_routed_predictions(
        args.manifest,
        args.model,
        args.output,
        inference_mode=args.inference_mode,
        resolutions=args.resolutions,
    )
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
