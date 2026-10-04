"""Owner-reviewed joint training; no A/B/C or learned-versus-random gate."""
from __future__ import annotations

from dataclasses import asdict, replace
import gc
import json
import math
from pathlib import Path
import random

import torch
from torch.utils.data import Subset

from ..tag_vocabulary import TagVocabulary
from ..tagger_loss import AsymmetricLossOptimized
from .artifact import FoveaArtifactConfig, load_inference_artifact, save_inference_artifact
from .control_run import select_domain_splits
from .joint_checkpoint import EDITABLE_JOINT_FIELDS, load_joint_checkpoint, save_joint_checkpoint
from .joint_experiment import _train_epoch, _model, _write_json_atomic
from .joint_training import FoveaJointTrainer
from .router import EvidenceRouter, ROUTER_OBJECTIVE_VERSION
from .router_data import FoveaRouterDataset
from .router_run import _loader
from .supervised_data import DomainBalancedSupervisedDataset, SupervisionPolicy
from .supervised_run import _cosine_with_warmup, _file_sha256, _holdout_contract, _load_raw_dataset


def validate_plan(plan):
    if plan.get("kind") != "foveatag_direct_joint_review_v1" or plan.get("acceptance") != "owner_visual_review":
        raise ValueError("A direct-joint review plan is required")
    for key in ("bootstrap", "bootstrap_metadata", "vocabulary_path", "datasets_db"):
        if _file_sha256(Path(plan[key])) != plan["hashes"][key]:
            raise ValueError(f"Direct-joint input changed: {key}")
    for path, digest in plan["holdout_hashes"].items():
        if _file_sha256(Path(path)) != digest:
            raise ValueError("Direct-joint holdout changed")
    cfg = FoveaArtifactConfig.from_dict(json.loads(Path(plan["bootstrap_metadata"]).read_text())["config"])
    if json.loads(json.dumps(cfg.to_dict())) != plan["artifact_config"]:
        raise ValueError("Review artifact configuration changed")
    if cfg.views.overview_resize_mode != "stretch" or cfg.views.local_border_mode != "shift_inside":
        raise ValueError("Direct-joint review requires fixed unpadded source views")
    return cfg


def run_direct_joint(plan, *, progress_callback, run_control, resume=None, segment_steps=None,
        diagnostic_due=None, diagnostic_callback=None):
    cfg = validate_plan(plan)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for direct joint training")
    root = Path(plan["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    vocab = TagVocabulary.from_dict(json.loads(Path(plan["vocabulary_path"]).read_text()))
    counts = {int(k): v for k, v in plan["train_counts"].items()}
    val_counts = {int(k): v for k, v in plan["validation_counts"].items()}
    weights = {int(k): v for k, v in plan["domain_weights"].items()}
    excluded, fine_tags = _holdout_contract([Path(p) for p in plan["holdout_manifests"]])
    raw = _load_raw_dataset(Path(plan["datasets_db"]), dataset_ids=sorted(counts), vocabulary=vocab,
        max_source_pixels=plan["max_source_pixels"])
    train_ids, val_ids, _ = select_domain_splits(raw, train_counts=counts, validation_counts=val_counts,
        excluded_paths=excluded, seed=plan["seed"])
    domains = {key: tuple(i for i in train_ids if raw.dataset_id_for_index(i) == key) for key in counts}
    fine = tuple(sorted(vocab.tag_to_idx[tag] for tag in fine_tags if tag in vocab.tag_to_idx))
    policies = {plan["photo_dataset_id"]: SupervisionPolicy(loss_weight=plan["photo_loss_weight"], positive_only_indices=fine)}
    candidates = replace(cfg.views, local_views=plan["candidate_count"])
    train_source = FoveaRouterDataset(raw, view_config=candidates, candidate_count=plan["candidate_count"],
        seed=plan["seed"], training=True, supervision_policies=policies)
    train_data = DomainBalancedSupervisedDataset(train_source, domains, weights,
        epoch_samples=plan["epoch_samples"], seed=plan["seed"])
    val_data = FoveaRouterDataset(raw, view_config=candidates, candidate_count=plan["candidate_count"],
        seed=plan["seed"], training=False, supervision_policies=policies)
    random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    torch.cuda.manual_seed_all(plan["seed"])
    model = _model(cfg)
    load_inference_artifact(plan["bootstrap"], model)
    router = EvidenceRouter(cfg.encoder.widths[2], routing_dim=cfg.routing_dim, routing_slots=cfg.routing_slots)
    device = torch.device("cuda")
    model.to(device)
    router.to(device)
    model.encoder.set_sdpa(True, external_flash=plan["external_flash"])
    optimizer = torch.optim.AdamW([*model.parameters(), *router.parameters()], lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
    total = plan["epochs"] * math.ceil(plan["epoch_samples"] / plan["batch_size"])
    scheduler = _cosine_with_warmup(optimizer, total_steps=total,
        warmup_steps=plan["warmup_steps"], learning_rate=plan["learning_rate"])
    trainer = FoveaJointTrainer(model, router, AsymmetricLossOptimized(), optimizer, device=device,
        selection_budget=plan["selection_budget"], random_count=plan["random_count"],
        oracle_interval=plan["oracle_interval"], oracle_chunk_size=plan["oracle_chunk_size"],
        router_weight=plan["router_weight"], mixed_precision=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        scheduler=scheduler)
    trainer.diagnostic_due = diagnostic_due
    trainer.diagnostic_callback = diagnostic_callback
    contract = json.loads(json.dumps(plan))
    contract["router_objective"] = ROUTER_OBJECTIVE_VERSION
    epoch = offset = step = 0
    best, threshold = -1.0, .35
    history, accumulator = [], None
    if resume:
        saved = load_joint_checkpoint(resume, model, router, optimizer, scheduler=scheduler,
            expected_run_contract=contract, allowed_contract_changes=(*EDITABLE_JOINT_FIELDS, "router_objective"),
            scheduler_contract_fields=("epochs", "batch_size", "learning_rate", "warmup_steps"))
        epoch, offset, step = saved["epoch"], saved["sample_offset"], saved["global_step"]
        best, threshold = saved["best_validation_f1"], saved["best_threshold"]
        history, accumulator = saved["history"], saved["epoch_accumulator"]
        prior_objective = saved["run_contract"].get("router_objective", "positive_margin_v1")
        if prior_objective not in ("positive_margin_v1", ROUTER_OBJECTIVE_VERSION):
            raise ValueError("Unsupported prior router objective")
        if prior_objective != ROUTER_OBJECTIVE_VERSION:
            revisions_path = root / "objective_revisions.json"
            revisions = json.loads(revisions_path.read_text()) if revisions_path.exists() else []
            revision = {"saved_step": step, "from": prior_objective, "to": ROUTER_OBJECTIVE_VERSION,
                "optimizer_state": "preserved", "epoch_loss_accumulator": "includes_prior_objective"}
            if revision not in revisions:
                _write_json_atomic(revisions_path, [*revisions, revision])
        changed = {key for key in EDITABLE_JOINT_FIELDS if saved["run_contract"].get(key) != plan.get(key)}
        for group in optimizer.param_groups:
            group["weight_decay"] = plan["weight_decay"]
        if changed & {"epochs", "batch_size", "learning_rate", "warmup_steps"}:
            total = step + math.ceil(max(0, plan["epoch_samples"] - offset) / plan["batch_size"]) + max(0, plan["epochs"] - epoch - 1) * math.ceil(plan["epoch_samples"] / plan["batch_size"])
            scheduler = _rebase_schedule(optimizer, step, plan, total)
            trainer.scheduler = scheduler
    cfg = replace(cfg, inference_modes={"balanced": replace(cfg.inference_modes["balanced"], local_views=plan["selection_budget"])})
    start_step = step
    progress_callback("resumed", {"epoch": epoch, "global_step": step, "sample_offset": offset})

    def checkpoint():
        progress_callback("checkpoint_saved", {"path": str(root / "latest.pt")})

    run_control.attach(model, optimizer, trainer.criterion)
    trainer.batch_boundary = run_control.batch_boundary
    status, train_summary = "complete", {}
    try:
        while epoch < plan["epochs"]:
            trainer.diagnostic_threshold = threshold
            trainer.diagnostic_epoch = epoch
            train_data.set_epoch(epoch)
            train_summary, offset, step, _, stopped, accumulator = _train_epoch(trainer, train_data, epoch=epoch,
                sample_offset=offset, batch_size=plan["batch_size"], workers=plan["workers"],
                checkpoint_every_steps=plan["checkpoint_every_steps"], log_every=plan["log_every"],
                global_step=step, output_dir=root, best_f1=best, best_threshold=threshold,
                run_contract=contract, history=history, initial_accumulator=accumulator,
                remaining_step_budget=(None if segment_steps is None else segment_steps - (step - start_step)),
                seed=plan["seed"], progress_callback=progress_callback, run_control=run_control,
                checkpoint_callback=checkpoint)
            if stopped:
                status = stopped
                break
            result = trainer.validate_epoch(_loader(Subset(val_data, val_ids), batch_size=plan["batch_size"], workers=plan["workers"]))
            metrics = {"epoch": epoch, "global_step": step, **asdict(result)}
            history.append(metrics)
            progress_callback("validation", metrics)
            if result.macro_f1 > best:
                best, threshold = result.macro_f1, result.threshold
                save_inference_artifact(root / "best_joint.safetensors", model, cfg, router=router,
                    vocabulary=vocab.to_dict(), provenance={"initialization": "scratch", "external_weights": [],
                        "stage": "direct_joint_owner_review", "acceptance": "awaiting_owner_visual_review",
                        "global_step": step, "epoch": epoch, "threshold": threshold,
                        "router_objective": ROUTER_OBJECTIVE_VERSION})
            epoch, offset, accumulator = epoch + 1, 0, None
            save_joint_checkpoint(root / "latest.pt", model, router, optimizer, scheduler=scheduler,
                epoch=epoch, sample_offset=offset, global_step=step, best_validation_f1=best,
                best_threshold=threshold, run_contract=contract, history=history)
            checkpoint()
            if segment_steps is not None and step - start_step >= segment_steps:
                status = "stopped_for_owner_review"
                break
        save_inference_artifact(root / "review.safetensors", model, cfg, router=router,
            vocabulary=vocab.to_dict(), provenance={"initialization": "scratch", "external_weights": [],
                "stage": "direct_joint_owner_review", "acceptance": "awaiting_owner_visual_review",
                "global_step": step, "epoch": epoch, "threshold": threshold,
                "router_objective": ROUTER_OBJECTIVE_VERSION})
        summary = {"status": status, "global_step": step, "epoch": epoch, "sample_offset": offset, "train": train_summary,
            "acceptance": "awaiting_owner_visual_review", "history": history,
            "router_objective": ROUTER_OBJECTIVE_VERSION}
        _write_json_atomic(root / "summary.json", summary)
        return summary
    finally:
        run_control.detach()
        del trainer, optimizer, scheduler, model, router
        gc.collect()
        torch.cuda.empty_cache()


def _rebase_schedule(optimizer, step, plan, total):
    warmup = min(plan["warmup_steps"], max(0, total - 1))
    base = plan["learning_rate"]
    for group in optimizer.param_groups:
        group["initial_lr"], group["lr"] = base, base
    scheduler = _cosine_with_warmup(optimizer, total_steps=total, warmup_steps=plan["warmup_steps"], learning_rate=base)
    factor = .01 + .99 * step / warmup if warmup and step < warmup else (
        .01 + .99 * (1 + math.cos(math.pi * min(max(0, step - warmup), max(1, total - warmup)) / max(1, total - warmup))) / 2)
    lr = base * factor
    state = scheduler.state_dict()
    state["last_epoch"], state["_last_lr"] = step, [lr for _ in optimizer.param_groups]
    if "_schedulers" in state:
        linear, cosine = state["_schedulers"]
        linear["last_epoch"] = min(step, warmup)
        linear["_last_lr"] = [base * (.01 + .99 * min(step, warmup) / warmup) for _ in optimizer.param_groups]
        cosine["last_epoch"] = max(0, step - warmup)
        cosine["_last_lr"] = state["_last_lr"]
    scheduler.load_state_dict(state)
    for group in optimizer.param_groups:
        group["initial_lr"], group["lr"] = base, lr
    return scheduler
