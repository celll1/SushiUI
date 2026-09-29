"""Evaluate a Phase-2 checkpoint without resuming its optimizer run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .encoder import FoveaEncoderConfig
from .objectives import FoveaSelfSupervisedLoss
from .pretraining import FoveaPretrainer
from .pretraining_checkpoint import PRETRAINING_CHECKPOINT_FORMAT_VERSION
from .pretraining_manifest import open_domain_manifests
from .pretraining_run import _normalized_domain_weights, _validate
from .views import ViewConfig


def evaluate_pretraining_checkpoint(
    checkpoint: str | Path,
    output_path: str | Path,
    *,
    batch_size: int | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FoveaTag Phase-2 evaluation")
    checkpoint = Path(checkpoint)
    checkpoint_sha256 = _file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if _file_sha256(checkpoint) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed while it was being loaded")
    model, contract = _model_from_payload(payload)
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
        amp_dtype=amp_dtype,
        device=device,
    )
    if _file_sha256(checkpoint) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during evaluation")
    result = {
        "format_version": 1,
        "kind": "foveatag_phase2_checkpoint_evaluation",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "global_step": int(payload["global_step"]),
        "epoch": int(payload["epoch"]),
        "sample_offset": int(payload["sample_offset"]),
        "run_contract_sha256": _json_sha256(contract),
        "dtype": str(amp_dtype).removeprefix("torch."),
        "validation": validation,
    }
    _write_json_atomic(Path(output_path), result)
    return result


def _model_from_payload(payload: Mapping[str, Any]) -> tuple[FoveaPretrainer, dict[str, Any]]:
    if payload.get("format_version") != PRETRAINING_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag pretraining checkpoint")
    if payload.get("kind") != "foveatag_scratch_pretraining":
        raise ValueError("checkpoint is not FoveaTag scratch pretraining")
    contract = dict(payload["run_contract"])
    if contract.get("initialization") != "scratch" or contract.get("external_weights") != []:
        raise ValueError("checkpoint does not have pure-scratch lineage")
    model = FoveaPretrainer(
        FoveaEncoderConfig(**contract["encoder"]),
        mask_ratio=float(contract["mask_ratio"]),
        objective=FoveaSelfSupervisedLoss(**contract["objective"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model, contract


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = evaluate_pretraining_checkpoint(
        args.checkpoint,
        args.output,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
