"""Write resolution-ladder predictions for Phase-3 FoveaTag A/B/C branches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

import torch
from PIL import Image

from .artifact import FoveaArtifactConfig, load_inference_artifact
from .batch import collate_view_batches
from .benchmark import PREDICTION_FORMAT_VERSION, load_benchmark_manifest
from .model import FoveaTagger
from .siglip2_prediction_runner import resize_benchmark_image
from .supervised_data import FoveaInputMode, build_supervised_views


def run_supervised_predictions(
    manifest_path: Path | str,
    model_path: Path | str,
    output_path: Path | str,
    *,
    resolutions: Optional[Sequence[str]] = None,
    device: torch.device | str = "cuda",
) -> Path:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FoveaTag benchmark prediction")
    manifest = load_benchmark_manifest(manifest_path)
    model_path = Path(model_path)
    metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
    artifact_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = FoveaArtifactConfig.from_dict(artifact_metadata["config"])
    input_mode = _artifact_input_mode(artifact_metadata)
    vocabulary = _artifact_vocabulary(artifact_metadata, config.num_tags)
    tag_to_idx = vocabulary["tag_to_idx"]

    model = FoveaTagger(
        config.num_tags,
        encoder_config=config.encoder,
        feature_dim=config.feature_dim,
        evidence_slots_per_view=config.evidence_slots_per_view,
        pool_heads=config.pool_heads,
        temperature=config.temperature,
    )
    load_inference_artifact(model_path, model)
    model.to(device).eval()
    amp_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ) if device.type == "cuda" else torch.float32

    target_tags = sorted({tag for item in manifest.items for tag in item.positive_tags})
    tag_indices = {
        tag: int(tag_to_idx[tag]) if tag in tag_to_idx else None
        for tag in target_tags
    }
    present_tags = [tag for tag, index in tag_indices.items() if index is not None]
    present_indices = torch.tensor(
        [tag_indices[tag] for tag in present_tags], device=device, dtype=torch.long
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
                render_started = time.perf_counter()
                views = build_supervised_views(
                    image,
                    config.views,
                    seed=_view_seed(item.sample_id, resolution),
                    training=False,
                    input_mode=input_mode,
                )
                render_ms = (time.perf_counter() - render_started) * 1000.0
                transfer_started = time.perf_counter()
                batch = collate_view_batches([views]).to(
                    device, dtype=amp_dtype, non_blocking=True
                )
                _synchronize(device)
                transfer_ms = (time.perf_counter() - transfer_started) * 1000.0
                forward_started = time.perf_counter()
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    logits = model(
                        batch.overview_images,
                        batch.overview_geometry,
                        batch.local_images,
                        batch.local_geometry,
                        local_view_mask=batch.local_view_mask,
                    )[0]
                probabilities = logits.sigmoid()
                _synchronize(device)
                forward_ms = (time.perf_counter() - forward_started) * 1000.0
                selected = probabilities.index_select(0, present_indices).float().cpu().tolist()
                scores = dict(zip(present_tags, selected))
                scores.update(
                    (tag, 0.0) for tag, index in tag_indices.items() if index is None
                )
                rows.append(
                    {
                        "sample_id": item.sample_id,
                        "resolution": resolution,
                        "scores": scores,
                        "stage_ms": {
                            "decode": decode_ms,
                            "resize": resize_ms,
                            "view_render": render_ms,
                            "transfer": transfer_ms,
                            "forward": forward_ms,
                        },
                    }
                )
            if (item_index + 1) % 100 == 0 or item_index + 1 == len(manifest.items):
                print(
                    f"[FoveaTagSupervised] predicted {item_index + 1}/{len(manifest.items)} images",
                    flush=True,
                )
        payload = {
            "format_version": PREDICTION_FORMAT_VERSION,
            "model_name": f"foveatag-{input_mode}",
            "checkpoint": str(model_path.resolve()),
            "artifact_provenance": artifact_metadata.get("provenance", {}),
            "missing_tags": [tag for tag, index in tag_indices.items() if index is None],
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
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _artifact_input_mode(metadata: Mapping[str, Any]) -> FoveaInputMode:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("FoveaTag artifact has no provenance")
    input_mode = provenance.get("input_mode")
    if input_mode not in {"overview_only", "overview_placebo", "source_native"}:
        raise ValueError("FoveaTag artifact has no valid Phase-3 input mode")
    return input_mode


def _artifact_vocabulary(
    metadata: Mapping[str, Any], num_tags: int
) -> Mapping[str, Any]:
    vocabulary = metadata.get("vocabulary")
    if not isinstance(vocabulary, Mapping):
        raise ValueError("FoveaTag artifact has no vocabulary")
    tag_to_idx = vocabulary.get("tag_to_idx")
    idx_to_tag = vocabulary.get("idx_to_tag")
    if not isinstance(tag_to_idx, Mapping) or not isinstance(idx_to_tag, Mapping):
        raise ValueError("FoveaTag artifact vocabulary is incomplete")
    if len(tag_to_idx) != num_tags or len(idx_to_tag) != num_tags:
        raise ValueError("FoveaTag artifact vocabulary size does not match the model")
    return vocabulary


def _view_seed(sample_id: str, resolution: str) -> int:
    digest = hashlib.blake2b(
        f"{sample_id}\0{resolution}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", nargs="+")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output = run_supervised_predictions(
        args.manifest,
        args.model,
        args.output,
        resolutions=args.resolutions,
    )
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
