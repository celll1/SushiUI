"""Catalog/lifecycle bridge for contract-bound experimental comparisons."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

MODES = ("overview_only", "overview_placebo", "source_native")
PATH_ARGUMENTS = ("phase2_gate", "pretrained_encoder", "vocabulary_path", "datasets_db", "comparison_dir")


def import_experiment(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Validate saved lineage and recover launch settings without changing it."""
    import torch
    from .phase2_gate import load_passed_phase2_gate, require_phase2_encoder

    directory = Path(arguments["comparison_dir"]).resolve()
    require_no_external_worker(directory)
    comparison = _json(directory / "comparison_summary.json")
    if comparison.get("kind") != "foveatag_phase3_supervised_comparison":
        raise ValueError("not a Phase 3 comparison directory")
    mode = comparison.get("active_mode")
    if mode not in MODES:
        raise ValueError("comparison must have an active paused branch")
    checkpoint = directory / mode / "latest.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("kind") != "foveatag_supervised_training":
        raise ValueError("not a supervised resume checkpoint")
    if not payload.get("optimizer", {}).get("state") or payload.get("scheduler") is None:
        raise ValueError("resume optimizer/scheduler state is missing")
    for name in ("python_rng_state", "torch_rng_state", "cuda_rng_state"):
        if payload.get(name) is None:
            raise ValueError(f"resume {name} is missing")
    contract = payload["run_contract"]
    if contract["input_mode"] != mode:
        raise ValueError("resume branch mismatch")
    for role, name in (("shared_data_contract", "shared_data_contract.json"), ("shared_training_contract", "shared_training_contract.json"), ("shared_initialization", "shared_initialization.safetensors")):
        if _sha(directory / name) != contract[f"{role}_sha256"]:
            raise ValueError(f"{role} changed since checkpoint")
    gate = load_passed_phase2_gate(arguments["phase2_gate"])
    require_phase2_encoder(gate, Path(arguments["pretrained_encoder"]))
    launch = {key: contract[key] for key in (
        "datasets_db", "train_counts", "validation_counts", "domain_weights", "holdout_manifests",
        "photo_dataset_id", "photo_loss_weight", "epochs", "epoch_samples", "batch_size", "workers",
        "learning_rate", "weight_decay", "warmup_steps", "seed", "max_source_pixels",
    )}
    launch.update({key: str(Path(arguments[key]).resolve()) for key in ("phase2_gate", "pretrained_encoder", "vocabulary_path")})
    launch.update(comparison_dir=str(directory), local_views=contract["views"]["local_views"],
                  checkpoint_every_steps=int(arguments["checkpoint_every_steps"]), log_every=int(arguments["log_every"]))
    for key in ("checkpoint_every_steps", "log_every"):
        if launch[key] <= 0:
            raise ValueError(f"{key} must be positive")
    vocab = _json(Path(launch["vocabulary_path"]))
    init_metadata = _json(directory / "shared_initialization_metadata.json")
    initialization = init_metadata["initialization_contract"]
    # Bind the caller's vocabulary/encoder to the exact common initialization.
    if initialization.get("vocabulary_sha256") != _sha(Path(launch["vocabulary_path"])):
        raise ValueError("vocabulary differs from registered initialization")
    if initialization.get("pretrained_encoder_sha256") != _sha(Path(launch["pretrained_encoder"])):
        raise ValueError("encoder differs from registered initialization")
    if initialization.get("phase2_gate_sha256") != _sha(Path(launch["phase2_gate"])):
        raise ValueError("Phase 2 gate differs from registered initialization")
    snapshot = {
        "mode": mode, "epoch": int(payload["epoch"]), "step": int(payload["global_step"]),
        "sample_offset": int(payload["sample_offset"]), "checkpoint": str(checkpoint),
        "best_f1": float(payload["best_validation_f1"]), "best_threshold": float(payload["best_threshold"]),
        "history": payload["history"], "vocabulary": vocab,
    }
    return {"arguments": launch, "snapshot": snapshot}


def run_catalog_comparison(config: Mapping[str, Any], emit: Callable, control) -> dict:
    from .supervised_run import run_supervised_comparison
    arguments = dict(config["foveatag"]["arguments"])
    for key in PATH_ARGUMENTS:
        arguments[key] = Path(arguments[key])
    for key in ("train_counts", "validation_counts", "domain_weights"):
        arguments[key] = {int(k): v for k, v in arguments[key].items()}
    arguments["holdout_manifests"] = [Path(p) for p in arguments["holdout_manifests"]]
    return run_supervised_comparison(**arguments, progress_callback=emit, run_control=control)


def project_progress(event: str, data: Mapping[str, Any], arguments: Mapping[str, Any]) -> list[tuple[str, dict]]:
    """Map branch-local counters to the shared training UI's global counters."""
    mode = data.get("input_mode")
    if mode not in MODES:
        raise ValueError("progress requires a known branch")
    branch = MODES.index(mode)
    epochs = int(arguments["epochs"])
    epoch_samples = int(arguments["epoch_samples"])
    nominal_steps = math.ceil(epoch_samples / int(arguments["batch_size"])) * epochs
    step = branch * nominal_steps + int(data.get("global_step", 0))
    epoch = int(data.get("epoch", 0))
    display_epoch = branch * epochs + epoch + 1
    if event == "branch_start":
        return [("phase", {"message": f"FoveaTag {mode}: preparing/resuming registered branch"})]
    if event == "resumed":
        return [("resume", {"resumed_from_step": step})]
    if event == "train":
        fraction = min(1.0, (branch * epochs + epoch + int(data["sample_offset"]) / epoch_samples) / (3 * epochs))
        return [("step", {"step": step, "epoch": display_epoch, "loss": data.get("loss"),
                          "lr": data.get("learning_rate"), "progress": fraction})]
    if event == "validation":
        return [("epoch", {"step": step, "epoch": display_epoch, "loss": data["mean_loss"],
                           "f1": data["macro_f1"], "precision": data["macro_precision"],
                           "recall": data["macro_recall"], "threshold": data["threshold"]})]
    return []


def require_no_external_worker(directory: Path) -> None:
    import psutil
    for process in psutil.process_iter(["cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if "core.tagger.foveatag.supervised_run" not in command or "--comparison-dir" not in command:
                continue
            position = command.index("--comparison-dir") + 1
            if position < len(command) and Path(command[position]).resolve() == directory.resolve():
                raise ValueError("comparison still has an external CLI worker; stop it before importing/resuming")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
