"""Evaluate trained Phase-1 control weights on a FoveaTag benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image

from ..siglip2_inference_manager import SigLIP2InferenceManager
from .benchmark import PREDICTION_FORMAT_VERSION, load_benchmark_manifest
from .control_checkpoint import load_control_weights
from .control_run import _checkpoint_vocabulary, _file_sha256
from .siglip2_control import SigLIP2ControlTagger
from .siglip2_control_data import collate_siglip2_control_samples
from .siglip2_prediction_runner import resize_benchmark_image
from .supervised_data import FoveaSupervisedSample
from .views import ViewConfig, build_views, preserve_source_overview


def run_control_predictions(
    manifest_path: Path | str,
    encoder_checkpoint: Path | str,
    vocabulary_path: Path | str,
    control_weights: Path | str,
    output_path: Path | str,
    *,
    local_views: int = 4,
    resolutions: Optional[Sequence[str]] = None,
) -> Path:
    manifest = load_benchmark_manifest(manifest_path)
    encoder_checkpoint = Path(encoder_checkpoint)
    control_weights = Path(control_weights).resolve()
    artifact_hashes = _control_artifact_hashes(control_weights)
    vocabulary_path = _checkpoint_vocabulary(encoder_checkpoint, Path(vocabulary_path))
    manager = SigLIP2InferenceManager()
    manager.load_model(str(encoder_checkpoint), str(vocabulary_path))
    processor = manager.processor
    is_naflex = manager.is_naflex
    vocabulary = manager.vocabulary
    original = manager.model
    encoder = original.vision_encoder
    baseline_head = original.head
    hidden_size = int(encoder.config.hidden_size)
    encoder_path = Path(manager.vision_encoder_path) if manager.vision_encoder_path else None
    manager.model = None
    del original
    gc.collect()
    torch.cuda.empty_cache()

    provenance = {
        "checkpoint": str(encoder_checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(encoder_checkpoint),
        "vocabulary": str(vocabulary_path.resolve()),
        "vocabulary_sha256": _file_sha256(vocabulary_path),
        "vision_encoder": str(encoder_path.resolve()) if encoder_path is not None else None,
        "vision_encoder_sha256": (
            _file_sha256(encoder_path) if encoder_path is not None and encoder_path.is_file() else None
        ),
        "is_naflex": is_naflex,
        "control_head": "baseline_zero_residual_v1",
    }
    model = SigLIP2ControlTagger(
        encoder,
        hidden_size=hidden_size,
        feature_dim=512,
        num_tags=len(vocabulary["idx_to_tag"]),
        evidence_slots_per_view=2,
        is_naflex=is_naflex,
        freeze_encoder=True,
        baseline_head=baseline_head,
    ).to("cuda").eval()
    metadata = load_control_weights(
        control_weights,
        model,
        expected_encoder_provenance=provenance,
    )
    target_tags = sorted({tag for item in manifest.items for tag in item.positive_tags})
    tag_indices = {tag: vocabulary["tag_to_idx"].get(tag) for tag in target_tags}
    present_tags = [tag for tag, index in tag_indices.items() if index is not None]
    present_indices = torch.tensor(
        [tag_indices[tag] for tag in present_tags], device="cuda", dtype=torch.long
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
    view_config = ViewConfig(overview_size=448, local_size=256, local_views=local_views)
    rows = []
    device = torch.device("cuda")
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
                views = build_views(
                    image,
                    view_config,
                    seed=_view_seed(item.sample_id, resolution),
                    training=False,
                )
                views = preserve_source_overview(image, views)
                render_ms = (time.perf_counter() - render_started) * 1000.0
                preprocess_started = time.perf_counter()
                sample = FoveaSupervisedSample(
                    views=views,
                    label=torch.zeros(len(vocabulary["idx_to_tag"])),
                    loss_mask=torch.zeros(len(vocabulary["idx_to_tag"])),
                )
                batch = collate_siglip2_control_samples(
                    [sample],
                    processor=processor,
                    is_naflex=is_naflex,
                )
                assert batch is not None
                batch = batch.to(device, non_blocking=True)
                preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
                torch.cuda.synchronize(device)
                forward_started = time.perf_counter()
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    logits = model(
                        batch.overview,
                        batch.local,
                        local_view_mask=batch.local_view_mask,
                    )[0]
                probabilities = logits.sigmoid()
                torch.cuda.synchronize(device)
                forward_ms = (time.perf_counter() - forward_started) * 1000.0
                selected = probabilities.index_select(0, present_indices).float().cpu().tolist()
                scores = dict(zip(present_tags, selected))
                for tag, index in tag_indices.items():
                    if index is None:
                        scores[tag] = 0.0
                rows.append(
                    {
                        "sample_id": item.sample_id,
                        "resolution": resolution,
                        "scores": scores,
                        "stage_ms": {
                            "decode": decode_ms,
                            "resize": resize_ms,
                            "view_render": render_ms,
                            "preprocess": preprocess_ms,
                            "forward": forward_ms,
                        },
                    }
                )
            if (item_index + 1) % 100 == 0 or item_index + 1 == len(manifest.items):
                print(
                    f"[FoveaTagControl] predicted {item_index + 1}/{len(manifest.items)} images",
                    flush=True,
                )
        if _control_artifact_hashes(control_weights) != artifact_hashes:
            raise RuntimeError("control artifact changed during prediction")
        payload = {
            "format_version": PREDICTION_FORMAT_VERSION,
            "model_name": f"foveatag-control-random-{local_views}",
            "checkpoint": str(control_weights),
            **artifact_hashes,
            "control_metadata": metadata,
            "missing_tags": [tag for tag, index in tag_indices.items() if index is None],
            "predictions": rows,
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output_path
    finally:
        del model, encoder
        manager.unload()
        gc.collect()
        torch.cuda.empty_cache()


def _view_seed(sample_id: str, resolution: str) -> int:
    digest = hashlib.blake2b(
        f"{sample_id}\0{resolution}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def _control_artifact_hashes(path: Path) -> dict[str, str]:
    metadata = path.with_name(f"{path.stem}_metadata.json")
    return {
        "checkpoint_sha256": _file_sha256(path),
        "checkpoint_metadata_sha256": _file_sha256(metadata),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--control-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-views", type=int, default=4)
    parser.add_argument("--resolutions", nargs="+")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output = run_control_predictions(
        args.manifest,
        args.encoder_checkpoint,
        args.vocabulary,
        args.control_weights,
        args.output,
        local_views=args.local_views,
        resolutions=args.resolutions,
    )
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
