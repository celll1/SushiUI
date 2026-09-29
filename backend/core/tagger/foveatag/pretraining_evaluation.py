"""Evaluate a Phase-2 checkpoint or encoder without resuming training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

import torch

from .encoder import FoveaEncoder, FoveaEncoderConfig
from .feature_health import FeatureHealthAccumulator
from .objectives import FoveaSelfSupervisedLoss, overlap_correspondence_loss
from .pretraining import FoveaPretrainer
from .pretraining_checkpoint import (
    PRETRAINING_CHECKPOINT_FORMAT_VERSION,
    load_pretrained_encoder,
)
from .pretraining_manifest import ManifestPretrainingDataset, open_domain_manifests
from .pretraining_run import _loader, _normalized_domain_weights, _validate
from .reconstruction import mask_image_patches, random_patch_mask
from .views import ViewConfig


def evaluate_pretraining_checkpoint(
    checkpoint: str | Path,
    output_path: str | Path,
    *,
    batch_size: int | None = None,
    workers: int = 8,
    include_scratch_baseline: bool = True,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FoveaTag Phase-2 evaluation")
    checkpoint = Path(checkpoint)
    checkpoint_sha256 = _file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if _file_sha256(checkpoint) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed while it was being loaded")
    model, contract = _model_from_payload(payload)
    progress = {
        "global_step": int(payload["global_step"]),
        "epoch": int(payload["epoch"]),
        "sample_offset": int(payload["sample_offset"]),
    }
    del payload
    resolved_batch_size = int(batch_size or contract["batch_size"])
    if resolved_batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers must be non-negative")
    manifest_dir = checkpoint.parent / "manifests"
    manifest_metadata = {"domains": contract["manifest_domains"]}
    _verify_validation_manifests(manifest_dir, manifest_metadata)
    manifests = open_domain_manifests(manifest_dir, manifest_metadata, "validation")
    domain_ids = tuple(sorted(int(value) for value in contract["manifest_domains"]))
    domain_weights = _normalized_domain_weights(
        domain_ids,
        {int(key): float(value) for key, value in contract["domain_weights"].items()},
    )
    device = torch.device("cuda")
    model.to(device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    validation = _validate(
        model,
        manifests,
        domain_weights=domain_weights,
        view_config=ViewConfig(**contract["views"]),
        seed=int(contract["seed"]),
        batch_size=resolved_batch_size,
        workers=workers,
        max_source_pixels=contract["max_source_pixels"],
        pyramid_cache_root=_pyramid_cache_root(contract),
        amp_dtype=amp_dtype,
        device=device,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    scratch_validation = None
    if include_scratch_baseline:
        seed = int(contract["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        scratch_model = _new_model(contract).to(device)
        scratch_validation = _validate(
            scratch_model,
            manifests,
            domain_weights=domain_weights,
            view_config=ViewConfig(**contract["views"]),
            seed=seed,
            batch_size=resolved_batch_size,
            workers=workers,
            max_source_pixels=contract["max_source_pixels"],
            pyramid_cache_root=_pyramid_cache_root(contract),
            amp_dtype=amp_dtype,
            device=device,
        )
        del scratch_model
        gc.collect()
        torch.cuda.empty_cache()
    if _file_sha256(checkpoint) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during evaluation")
    result = {
        "format_version": 1,
        "kind": "foveatag_phase2_checkpoint_evaluation",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        **progress,
        "run_contract_sha256": _json_sha256(contract),
        "dtype": str(amp_dtype).removeprefix("torch."),
        "validation": validation,
        "scratch_initialization_validation": scratch_validation,
    }
    _write_json_atomic(Path(output_path), result)
    return result


def evaluate_pretrained_encoder(
    encoder_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int | None = None,
    workers: int = 8,
    include_scratch_baseline: bool = True,
) -> dict[str, Any]:
    """Evaluate the exact encoder artifact proposed for Phase 3."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FoveaTag Phase-2 evaluation")
    encoder_path = Path(encoder_path)
    metadata_path = encoder_path.with_name(f"{encoder_path.stem}_metadata.json")
    artifact_sha256 = _file_sha256(encoder_path)
    metadata_sha256 = _file_sha256(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = dict(metadata["run_contract"])
    run_summary_path = encoder_path.parent / "summary.json"
    run_summary_sha256 = _file_sha256(run_summary_path)
    run_summary = _completed_run_summary(run_summary_path, contract, metadata)
    resolved_batch_size = int(batch_size or contract["batch_size"])
    if resolved_batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers must be non-negative")
    manifest_dir = encoder_path.parent / "manifests"
    manifest_metadata = {"domains": contract["manifest_domains"]}
    _verify_validation_manifests(manifest_dir, manifest_metadata)
    manifests = open_domain_manifests(manifest_dir, manifest_metadata, "validation")
    domain_ids = tuple(sorted(int(value) for value in contract["manifest_domains"]))
    domain_weights = _normalized_domain_weights(
        domain_ids,
        {int(key): float(value) for key, value in contract["domain_weights"].items()},
    )
    config = FoveaEncoderConfig(**contract["encoder"])
    encoder = FoveaEncoder(config)
    load_pretrained_encoder(encoder_path, encoder, expected_run_contract=contract)
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    validation_kwargs = {
        "domain_weights": domain_weights,
        "view_config": ViewConfig(**contract["views"]),
        "seed": int(contract["seed"]),
        "batch_size": resolved_batch_size,
        "workers": workers,
        "max_source_pixels": contract["max_source_pixels"],
        "pyramid_cache_root": _pyramid_cache_root(contract),
        "mask_ratio": float(contract["mask_ratio"]),
        "max_coordinate_distance": float(contract["objective"]["max_coordinate_distance"]),
        "amp_dtype": amp_dtype,
        "device": device,
    }
    trained = _validate_encoder(encoder.to(device), manifests, **validation_kwargs)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    scratch = None
    if include_scratch_baseline:
        seed = int(contract["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        scratch_encoder = FoveaEncoder(config).to(device)
        scratch = _validate_encoder(scratch_encoder, manifests, **validation_kwargs)
        del scratch_encoder
        gc.collect()
        torch.cuda.empty_cache()
    if _file_sha256(encoder_path) != artifact_sha256:
        raise RuntimeError("encoder artifact changed during evaluation")
    if _file_sha256(metadata_path) != metadata_sha256:
        raise RuntimeError("encoder metadata changed during evaluation")
    if _file_sha256(run_summary_path) != run_summary_sha256:
        raise RuntimeError("pretraining summary changed during evaluation")
    result = {
        "format_version": 1,
        "kind": "foveatag_phase2_encoder_evaluation",
        "encoder": str(encoder_path.resolve()),
        "encoder_sha256": artifact_sha256,
        "encoder_metadata": str(metadata_path.resolve()),
        "encoder_metadata_sha256": metadata_sha256,
        "run_summary": str(run_summary_path.resolve()),
        "run_summary_sha256": run_summary_sha256,
        "epoch": int(metadata["epoch"]),
        "global_step": int(metadata["global_step"]),
        "run_contract_sha256": _json_sha256(contract),
        "dtype": str(amp_dtype).removeprefix("torch."),
        "training_validation": metadata["validation"],
        "encoder_validation": trained,
        "scratch_initialization_validation": scratch,
        "validation_manifests": _validation_manifest_evidence(
            manifest_dir, manifest_metadata
        ),
    }
    _write_json_atomic(Path(output_path), result)
    return result


def _completed_run_summary(
    path: Path,
    contract: Mapping[str, Any],
    encoder_metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError("pretraining summary must contain an object")
    if summary.get("kind") != "foveatag_teacher_free_pretraining_run":
        raise ValueError("artifact is not a Phase-2 pretraining summary")
    if summary.get("status") != "complete":
        raise ValueError("Phase-2 pretraining run is not complete")
    if summary.get("run_contract") != dict(contract):
        raise ValueError("pretraining summary run contract does not match encoder")
    if int(summary.get("epoch", -1)) != int(contract["epochs"]):
        raise ValueError("pretraining summary did not complete every registered epoch")
    encoder_loss = encoder_metadata.get("validation", {}).get("weighted", {}).get("loss")
    best_loss = summary.get("best_validation_loss")
    if not isinstance(encoder_loss, (int, float)) or not isinstance(best_loss, (int, float)):
        raise ValueError("pretraining summary lacks the best validation loss")
    if not math.isclose(float(encoder_loss), float(best_loss), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("encoder artifact is not the completed run's best checkpoint")
    return summary


@torch.no_grad()
def _validate_encoder(
    encoder: FoveaEncoder,
    manifests,
    *,
    domain_weights: Mapping[int, float],
    view_config: ViewConfig,
    seed: int,
    batch_size: int,
    workers: int,
    max_source_pixels: int | None,
    pyramid_cache_root: str | None,
    mask_ratio: float,
    max_coordinate_distance: float,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    encoder.eval()
    domains = {}
    for domain, manifest in sorted(manifests.items()):
        dataset = ManifestPretrainingDataset(
            manifest,
            view_config=view_config,
            seed=seed + domain * 1009,
            max_source_pixels=max_source_pixels,
            pyramid_cache_root=pyramid_cache_root,
        )
        loader = _loader(dataset, batch_size=batch_size, workers=workers, shuffle=False)
        generator = torch.Generator(device=device).manual_seed(seed + domain * 65537)
        local_health = FeatureHealthAccumulator(encoder.config.widths[2])
        overview_health = FeatureHealthAccumulator(encoder.config.widths[2])
        correspondence_sum = 0.0
        samples = skipped = consumed = 0
        for batch in loader:
            requested = min(batch_size, len(dataset) - consumed)
            consumed += requested
            if batch is None:
                skipped += requested
                continue
            batch = batch.to(device, non_blocking=True)
            height, width = batch.first_images.shape[-2:]
            patch_mask = random_patch_mask(
                batch.first_images.shape[0],
                height // 16,
                width // 16,
                mask_ratio,
                device=device,
                generator=generator,
            )
            masked = mask_image_patches(batch.first_images, patch_mask, patch_size=16)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                local = encoder(masked, batch.first_geometry)
                overview = encoder(batch.second_images, batch.second_geometry)
                correspondence = overlap_correspondence_loss(
                    local.local_map,
                    local.local_coordinates,
                    overview.local_map,
                    overview.local_coordinates,
                    max_coordinate_distance=max_coordinate_distance,
                )
            actual = batch.first_images.shape[0]
            correspondence_sum += float(correspondence) * actual
            local_health.update(local.local_map)
            overview_health.update(overview.local_map)
            samples += actual
            skipped += requested - actual
        if not samples:
            raise ValueError(f"validation domain {domain} produced no readable samples")
        domains[str(domain)] = {
            "samples": samples,
            "skipped": skipped,
            "correspondence": correspondence_sum / samples,
            "feature_health": {
                "masked_local": local_health.finalize(),
                "overview": overview_health.finalize(),
            },
        }
    weighted = sum(
        domain_weights[int(domain)] * values["correspondence"]
        for domain, values in domains.items()
    )
    return {"weighted_correspondence": weighted, "domains": domains}


def _model_from_payload(payload: Mapping[str, Any]) -> tuple[FoveaPretrainer, dict[str, Any]]:
    if payload.get("format_version") != PRETRAINING_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag pretraining checkpoint")
    if payload.get("kind") != "foveatag_scratch_pretraining":
        raise ValueError("checkpoint is not FoveaTag scratch pretraining")
    contract = dict(payload["run_contract"])
    if contract.get("initialization") != "scratch" or contract.get("external_weights") != []:
        raise ValueError("checkpoint does not have pure-scratch lineage")
    model = _new_model(contract)
    model.load_state_dict(payload["model"], strict=True)
    return model, contract


def _new_model(contract: Mapping[str, Any]) -> FoveaPretrainer:
    return FoveaPretrainer(
        FoveaEncoderConfig(**contract["encoder"]),
        mask_ratio=float(contract["mask_ratio"]),
        objective=FoveaSelfSupervisedLoss(**contract["objective"]),
    )


def _pyramid_cache_root(contract: Mapping[str, Any]) -> str | None:
    request = contract.get("manifest_request")
    if not isinstance(request, Mapping):
        return None
    cache = request.get("pyramid_cache")
    return str(cache["root"]) if isinstance(cache, Mapping) else None


def _verify_validation_manifests(
    manifest_dir: Path,
    metadata: Mapping[str, Any],
) -> None:
    for details in metadata["domains"].values():
        validation = details["validation"]
        path = manifest_dir / validation["path"]
        index_path = manifest_dir / validation["index_path"]
        if not path.is_file() or not index_path.is_file():
            raise FileNotFoundError("checkpoint validation manifest is missing")
        if _file_sha256(path) != validation["sha256"]:
            raise ValueError(f"validation manifest hash changed: {path}")
        if index_path.stat().st_size != int(validation["count"]) * 8:
            raise ValueError(f"validation manifest index changed: {index_path}")


def _validation_manifest_evidence(
    manifest_dir: Path,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for domain, details in sorted(metadata["domains"].items(), key=lambda item: int(item[0])):
        validation = details["validation"]
        for role, name in (
            ("paths", validation["path"]),
            ("index", validation["index_path"]),
        ):
            path = (manifest_dir / name).resolve()
            rows.append({
                "domain": int(domain),
                "role": role,
                "path": str(path),
                "sha256": _file_sha256(path),
            })
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--encoder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-scratch-baseline", action="store_true")
    args = parser.parse_args()
    function = (
        evaluate_pretraining_checkpoint
        if args.checkpoint is not None
        else evaluate_pretrained_encoder
    )
    result = function(
        args.checkpoint or args.encoder,
        args.output,
        batch_size=args.batch_size,
        workers=args.workers,
        include_scratch_baseline=not args.no_scratch_baseline,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
