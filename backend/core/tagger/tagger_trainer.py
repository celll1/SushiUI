"""
SigLIP2 Tagger Training Loop.

Supports:
  - Full parameter training and LoRA training
  - Mixed precision (bf16 / fp16 / fp32)
  - Gradient checkpointing
  - Cosine LR schedule with linear warmup
  - Validation: F1 macro, threshold optimization
  - Checkpoint saving (best F1 + latest + step-based)
  - Resume from checkpoint (epoch-boundary or mid-epoch with RNG state)
  - Progress callback for WebSocket updates
"""

from __future__ import annotations

import base64
import json
import os
import random
import re as _re
import threading
import time
from queue import Queue as _Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from safetensors.torch import load_file as _load_safetensors_file

from .siglip2_tagger_model import (
    SIGLIP2_DEFAULT_REPO_ID,
    SigLIP2TaggerLoRAModel,
    SigLIP2TaggerModel,
    _inherit_head,
    build_tagger_model,
)
from .tag_vocabulary import TagVocabulary
from .tagger_dataset import TaggerDataset, tagger_collate_fn
from .tagger_loss import AsymmetricLossOptimized

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False


# ------------------------------------------------------------------
# Optimizer factory
# ------------------------------------------------------------------

def _build_optimizer(
    params,
    optimizer_name: str,
    lr: float,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-6,
):
    name = optimizer_name.lower()
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif name == "adamw8bit":
        if not _BNB_AVAILABLE:
            raise RuntimeError("bitsandbytes not installed. Run: pip install bitsandbytes")
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif name == "lion8bit":
        if not _BNB_AVAILABLE:
            raise RuntimeError("bitsandbytes not installed. Run: pip install bitsandbytes")
        return bnb.optim.Lion8bit(params, lr=lr, weight_decay=weight_decay, betas=(betas[0], betas[1]))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}. Use 'adamw', 'adamw8bit', or 'lion8bit'.")


# ------------------------------------------------------------------
# Validation utilities
# ------------------------------------------------------------------

def _compute_f1_macro(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    threshold: float = 0.35,
) -> float:
    """Compute macro F1 score across all tags."""
    preds_bin = (all_preds >= threshold).float()
    tp = (preds_bin * all_labels).sum(dim=0)
    fp = (preds_bin * (1 - all_labels)).sum(dim=0)
    fn = ((1 - preds_bin) * all_labels).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    # Only include tags that appear at least once in labels
    active = all_labels.sum(dim=0) > 0
    if active.sum() == 0:
        return 0.0
    return f1[active].mean().item()


def _find_best_threshold(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """Find threshold maximizing F1. Returns (best_threshold, best_f1)."""
    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(2, 18)]  # 0.10 to 0.85

    best_f1 = 0.0
    best_thr = 0.35
    for thr in thresholds:
        f1 = _compute_f1_macro(all_preds, all_labels, threshold=thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    # Select minimum threshold where F1 >= 95% of max (maximize recall)
    floor_f1 = best_f1 * 0.95
    for thr in thresholds:
        f1 = _compute_f1_macro(all_preds, all_labels, threshold=thr)
        if f1 >= floor_f1:
            return thr, best_f1

    return best_thr, best_f1


# ------------------------------------------------------------------
# Prefetch helper
# ------------------------------------------------------------------

def _prefetch_loader(loader, maxsize: int = 2):
    """Iterate *loader* in a background thread, yielding batches via a bounded queue.

    On Windows (num_workers=0) PIL decode and NaFlex processing release the GIL,
    so the background thread runs concurrently with GPU training in the main thread.
    maxsize=2 bounds memory to ~2 batches worth of tensors.
    """
    _END = object()
    q = _Queue(maxsize=maxsize)

    def _worker():
        try:
            for batch in loader:
                q.put(batch)
        finally:
            q.put(_END)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        item = q.get()
        if item is _END:
            break
        yield item


# ------------------------------------------------------------------
# Checkpoint state helpers
# ------------------------------------------------------------------

def _capture_rng() -> Dict[str, Any]:
    """Capture current Python + PyTorch CPU RNG state as a serialisable dict."""
    rs = random.getstate()
    return {
        "random_state": {
            "version": rs[0],
            "state": list(rs[1]),
            "gauss_next": rs[2],
        },
        "torch_rng_state": base64.b64encode(
            torch.get_rng_state().numpy().tobytes()
        ).decode("utf-8"),
    }


def _restore_rng(rng_snapshot: Dict[str, Any]) -> None:
    """Restore Python + PyTorch CPU RNG state from a snapshot produced by _capture_rng."""
    rs = rng_snapshot.get("random_state")
    if rs:
        random.setstate((rs["version"], tuple(rs["state"]), rs["gauss_next"]))
    torch_rng = rng_snapshot.get("torch_rng_state")
    if torch_rng:
        rng_bytes = base64.b64decode(torch_rng)
        torch.set_rng_state(torch.frombuffer(bytearray(rng_bytes), dtype=torch.uint8))


def _save_training_state(
    output_dir: str,
    name: str,
    epoch: int,
    global_step: int,
    batch_idx: int,
    best_f1: float,
    best_threshold: float,
    epoch_start_rng: Optional[Dict[str, Any]] = None,
) -> None:
    """Save training state JSON for resume.

    epoch_start_rng must be the RNG snapshot captured *before* iterating the
    DataLoader for `epoch`.  On resume, restoring this snapshot and re-iterating
    the DataLoader from batch 0 (while skipping ≤ batch_idx) reproduces the
    exact same shuffle permutation and therefore the exact same batch sequence.
    """
    state: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "batch_idx": batch_idx,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
    }
    if epoch_start_rng is not None:
        state["epoch_start_rng"] = epoch_start_rng
    path = os.path.join(output_dir, f"{name}_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f)


def _load_training_state(output_dir: str, name: str) -> Optional[Dict[str, Any]]:
    """Load training state JSON. Returns None if not found."""
    path = os.path.join(output_dir, f"{name}_state.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_optimizer_state(optimizer: Any, output_dir: str, name: str) -> None:
    """Save optimizer state dict to <name>_optimizer.pt."""
    path = os.path.join(output_dir, f"{name}_optimizer.pt")
    torch.save(optimizer.state_dict(), path)


def _load_optimizer_state(optimizer: Any, output_dir: str, name: str) -> bool:
    """Load optimizer state dict from <name>_optimizer.pt. Returns True if loaded."""
    path = os.path.join(output_dir, f"{name}_optimizer.pt")
    if not os.path.isfile(path):
        return False
    state = torch.load(path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state)
    return True


def _prune_step_checkpoints(output_dir: str, keep_last_n: int) -> None:
    """Delete oldest step-based checkpoints, keeping the most recent *keep_last_n*.

    Each checkpoint consists of three files:
      step_XXXXXX.safetensors
      step_XXXXXX_state.json
      step_XXXXXX_optimizer.pt

    Only step_* checkpoints are pruned; 'latest' and 'best_f1' are never touched.
    """
    if keep_last_n <= 0:
        return

    step_names: List[Tuple[int, str]] = []
    for fn in os.listdir(output_dir):
        m = _re.match(r"^step_(\d+)\.safetensors$", fn)
        if m:
            step_names.append((int(m.group(1)), f"step_{int(m.group(1)):06d}"))

    step_names.sort(key=lambda x: x[0])          # ascending → oldest first
    to_delete = step_names[:-keep_last_n] if len(step_names) > keep_last_n else []

    for _, name in to_delete:
        for suffix in (".safetensors", "_state.json", "_optimizer.pt"):
            path = os.path.join(output_dir, f"{name}{suffix}")
            if os.path.isfile(path):
                os.remove(path)
        print(f"[TaggerTrainer] Pruned old checkpoint: {name}")


def _save_model_checkpoint(
    model,
    output_dir: str,
    name: str,
    metadata: Optional[Dict],
    save_mode: str,
) -> str:
    """Dispatch to the appropriate save method based on *save_mode*.

    ``save_mode="lora"``   – compact LoRA+head only (SigLIP2TaggerLoRAModel)
                             or full weights (SigLIP2TaggerModel).
    ``save_mode="merged"`` – merge LoRA into vision encoder and save full model
                             (SigLIP2TaggerLoRAModel only; falls back to normal
                              save for non-LoRA models).
    The metadata is augmented with a ``checkpoint_save_mode`` key.
    """
    from core.tagger.siglip2_tagger_model import SigLIP2TaggerLoRAModel

    if metadata is None:
        metadata = {}
    metadata["checkpoint_save_mode"] = save_mode

    if save_mode == "merged" and isinstance(model, SigLIP2TaggerLoRAModel):
        return model.save_merged_checkpoint(output_dir, name, metadata)
    return model.save_checkpoint(output_dir, name, metadata)


def _find_resume_checkpoint(output_dir: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Find the checkpoint with the highest global_step to resume from.

    Considers both epoch-boundary checkpoints (latest_state.json) and
    step-based checkpoints (step_XXXXXX_state.json).  The one with the
    largest global_step wins, so mid-epoch stops are always preferred over
    an earlier epoch-boundary save.

    Returns (checkpoint_name, state_dict) or None if nothing resumable exists.
    """
    if not os.path.isdir(output_dir):
        return None

    candidates: List[Tuple[int, str, Dict[str, Any]]] = []

    # Epoch-boundary checkpoint ("latest")
    state = _load_training_state(output_dir, "latest")
    if state is not None and os.path.isfile(os.path.join(output_dir, "latest.safetensors")):
        candidates.append((state.get("global_step", 0), "latest", state))

    # Step-based checkpoints
    for fn in os.listdir(output_dir):
        m = _re.match(r"^step_(\d+)_state\.json$", fn)
        if m:
            ckpt_name = f"step_{int(m.group(1)):06d}"
            if os.path.isfile(os.path.join(output_dir, f"{ckpt_name}.safetensors")):
                s = _load_training_state(output_dir, ckpt_name)
                if s is not None:
                    candidates.append((s.get("global_step", 0), ckpt_name, s))

    if not candidates:
        return None

    # Pick the checkpoint that advanced furthest in training
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_name, best_state = candidates[0]
    return best_name, best_state


# ------------------------------------------------------------------
# Main trainer
# ------------------------------------------------------------------

class TaggerTrainer:
    """Training loop for SigLIP2 tagger.

    Parameters
    ----------
    run_id          : unique identifier (used in progress callbacks)
    config          : training hyperparameters dict
    vocabulary      : TagVocabulary
    output_dir      : directory to save checkpoints
    progress_callback : optional callable(run_id, event_type, data)
    """

    def __init__(
        self,
        run_id: str,
        config: Dict[str, Any],
        vocabulary: TagVocabulary,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
        old_vocabulary: Optional[TagVocabulary] = None,
    ) -> None:
        self.run_id = run_id
        self.config = config
        self.vocabulary = vocabulary
        self.output_dir = output_dir
        self.callback = progress_callback
        self.old_vocabulary = old_vocabulary
        self._stop_requested = False

        os.makedirs(output_dir, exist_ok=True)

        # Save vocabulary snapshot
        vocab_path = os.path.join(output_dir, "vocabulary.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocabulary.to_dict(), f, ensure_ascii=False, indent=2)

    def stop(self) -> None:
        self._stop_requested = True

    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        processor: AutoProcessor,
        resume_state: Optional[Dict[str, Any]] = None,
        resume_ckpt_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run training loop. Returns summary metrics dict."""
        cfg = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        print(f"[TaggerTrainer] === Phase: model build ===")
        print(f"[TaggerTrainer] Building model (method={cfg.get('training_method', 'lora')}, "
              f"num_tags={self.vocabulary.num_tags}, device={device})...")
        self._emit("phase", {"phase": "initializing", "message": "Building model..."})
        model = build_tagger_model(
            training_method=cfg.get("training_method", "lora"),
            num_tags=self.vocabulary.num_tags,
            vision_encoder_path=cfg["vision_encoder_path"],
            lora_rank=cfg.get("lora_rank", 32),
            lora_alpha=float(cfg.get("lora_alpha", 16.0)),
            cls_dim=cfg.get("cls_dim") or None,
            hidden_proj_dim=cfg.get("hidden_proj_dim") or None,
            init_head_from=cfg.get("init_head_from") or None,
            new_vocab=self.vocabulary.tag_to_idx if cfg.get("init_head_from") else None,
            repo_id=cfg.get("vision_encoder_repo", SIGLIP2_DEFAULT_REPO_ID),
        )
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count     = sum(p.numel() for p in model.parameters())
        print(f"[TaggerTrainer] Model built: {trainable_count:,} trainable / {total_count:,} total parameters")
        model = model.to(device)
        print(f"[TaggerTrainer] Model moved to {device}")

        # Load checkpoint weights if resuming (before any training)
        if resume_state is not None and resume_ckpt_name is not None:
            ckpt_path = os.path.join(self.output_dir, f"{resume_ckpt_name}.safetensors")
            if os.path.isfile(ckpt_path):
                _saved = _load_safetensors_file(ckpt_path)
                _ckpt_num_tags = _saved["head.weight"].shape[0] if "head.weight" in _saved else None
                _vocab_mismatch = _ckpt_num_tags is not None and _ckpt_num_tags != self.vocabulary.num_tags

                if _vocab_mismatch:
                    print(f"[TaggerTrainer] Vocabulary mismatch: checkpoint={_ckpt_num_tags}, "
                          f"current={self.vocabulary.num_tags}. Aligning by tag name...")
                    old_tag_to_idx = self.old_vocabulary.tag_to_idx if self.old_vocabulary else None
                    if old_tag_to_idx is None:
                        print("[TaggerTrainer] WARNING: old_vocabulary unavailable, falling back to positional copy")
                    _inherit_head(
                        model=model,
                        checkpoint_path=ckpt_path,
                        new_num_tags=self.vocabulary.num_tags,
                        new_vocab=self.vocabulary.tag_to_idx,
                        old_tag_to_idx=old_tag_to_idx,
                    )
                    # LoRA weights are vocab-independent — copy them separately
                    if isinstance(model, SigLIP2TaggerLoRAModel):
                        for module_name, lora_module in model._lora_modules.items():
                            prefix = f"lora.{module_name}"
                            if f"{prefix}.lora_A" in _saved:
                                lora_module.lora_A.data.copy_(_saved[f"{prefix}.lora_A"])
                            if f"{prefix}.lora_B" in _saved:
                                lora_module.lora_B.data.copy_(_saved[f"{prefix}.lora_B"])
                        print(f"[TaggerTrainer] LoRA weights restored from {resume_ckpt_name}")
                else:
                    model.load_weights_inplace(ckpt_path)
                    print(f"[TaggerTrainer] Loaded checkpoint weights from {resume_ckpt_name}")

        # Gradient checkpointing
        if cfg.get("gradient_checkpointing", True):
            if hasattr(model, "vision_encoder") and hasattr(model.vision_encoder, "encoder"):
                model.vision_encoder.encoder.gradient_checkpointing = True
                print("[Trainer] Gradient checkpointing enabled")

        # Mixed precision
        mp = cfg.get("mixed_precision", "bf16").lower()
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mp)
        use_amp = amp_dtype is not None and device.type == "cuda"
        scaler = GradScaler() if (use_amp and amp_dtype == torch.float16) else None

        # Optimizer with separate LR for head
        head_lr_mult = float(cfg.get("head_lr_multiplier", 10.0))
        base_lr      = float(cfg.get("learning_rate", 3e-4))

        trainable = model.trainable_parameters() if hasattr(model, "trainable_parameters") else [
            p for p in model.parameters() if p.requires_grad
        ]
        head_params    = list(model.head.parameters())
        head_ids       = {id(p) for p in head_params}
        encoder_params = [p for p in trainable if id(p) not in head_ids]

        param_groups = [
            {"params": encoder_params, "lr": base_lr},
            {"params": head_params,    "lr": base_lr * head_lr_mult},
        ]
        optimizer = _build_optimizer(
            param_groups,
            optimizer_name=cfg.get("optimizer", "adamw8bit"),
            lr=base_lr,
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )
        print(f"[TaggerTrainer] Optimizer: {cfg.get('optimizer', 'adamw8bit')}, "
              f"base_lr={base_lr}, head_lr={base_lr * head_lr_mult} (x{head_lr_mult})")

        # LR schedule: linear warmup → cosine decay
        epochs       = int(cfg.get("epochs", 10))
        warmup_steps = int(cfg.get("warmup_steps", 100))
        total_steps  = epochs * len(train_loader)
        print(f"[TaggerTrainer] Schedule: {epochs} epochs, {total_steps} total steps, "
              f"{warmup_steps} warmup steps")

        warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-7)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                  milestones=[warmup_steps])

        # Loss function
        criterion = AsymmetricLossOptimized(
            gamma_neg=float(cfg.get("loss_gamma_neg", 4.0)),
            gamma_pos=float(cfg.get("loss_gamma_pos", 1.0)),
            clip=float(cfg.get("loss_clip", 0.05)),
        ).to(device)

        # Step-based checkpoint interval (0 = disabled)
        save_every_n_steps    = int(cfg.get("save_every_n_steps", 500))
        # Epoch-based checkpoint interval (0 = disabled)
        save_every_n_epochs   = int(cfg.get("save_every_n_epochs", 0))
        # How many step checkpoints to keep (0 = keep all)
        keep_last_n_checkpoints = int(cfg.get("keep_last_n_checkpoints", 3))
        # "lora" = save LoRA+head only (compact); "merged" = merge LoRA into encoder and save full model
        checkpoint_save_mode = cfg.get("checkpoint_save_mode", "lora")

        # Training state
        best_f1         = 0.0
        best_threshold  = 0.35
        global_step     = 0
        resume_epoch    = 1   # first epoch to process (1-indexed)
        resume_batch_idx = -1  # last already-processed batch in resume_epoch (-1 = none)
        metrics_history: List[Dict] = []

        # ------------------------------------------------------------------
        # Resume from checkpoint
        # ------------------------------------------------------------------
        # epoch_start_rng_for_resume: the RNG snapshot saved at the start of
        # resume_epoch.  Restoring it before iterating the DataLoader reproduces
        # the exact shuffle permutation and therefore the exact batch order.
        epoch_start_rng_for_resume: Optional[Dict[str, Any]] = None

        if resume_state is not None:
            resume_epoch     = resume_state["epoch"]         # next epoch to train
            resume_batch_idx = resume_state["batch_idx"]     # last completed batch (-1 = full epoch done)
            global_step      = resume_state["global_step"]
            best_f1          = resume_state.get("best_f1", 0.0)
            best_threshold   = resume_state.get("best_threshold", 0.35)
            epoch_start_rng_for_resume = resume_state.get("epoch_start_rng")

            # Restore optimizer state
            if resume_ckpt_name:
                loaded = _load_optimizer_state(optimizer, self.output_dir, resume_ckpt_name)
                if loaded:
                    print(f"[TaggerTrainer] Optimizer state restored from {resume_ckpt_name}")

            # Fast-forward LR scheduler to match resumed global_step
            for _ in range(global_step):
                scheduler.step()

            print(
                f"[TaggerTrainer] Resuming from step {global_step} "
                f"(epoch {resume_epoch}, batch {resume_batch_idx})"
            )
            self._emit("phase", {
                "phase": "resuming",
                "message": f"Resuming from step {global_step} (epoch {resume_epoch})",
            })

        print(f"[TaggerTrainer] === Phase: training ===")
        print(f"[TaggerTrainer] Training started: {epochs} epochs, "
              f"{len(train_loader)} steps/epoch, amp={'on ('+mp+')' if use_amp else 'off'}")
        self._emit("phase", {"phase": "training", "message": "Training started"})

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        for epoch in range(1, epochs + 1):
            if self._stop_requested:
                break

            # Skip epochs that were fully completed before the resume point
            if epoch < resume_epoch:
                continue

            # -------------------------------------------------------
            # Capture / restore epoch-start RNG state
            #
            # PyTorch DataLoader (shuffle=True) calls torch.randperm at the
            # very start of each iteration, consuming the RNG to generate the
            # epoch permutation.  To reproduce the exact batch order on resume
            # we must restore the RNG to where it was *before* that randperm
            # call, then let the DataLoader re-generate the same permutation,
            # and simply skip the already-processed batches.
            # -------------------------------------------------------
            if epoch == resume_epoch and epoch_start_rng_for_resume is not None:
                # Restore the saved epoch-start RNG so the DataLoader produces
                # the same shuffle permutation as the interrupted run.
                _restore_rng(epoch_start_rng_for_resume)
                epoch_start_rng_for_resume = None  # only needed once

            # Snapshot the RNG right before the DataLoader begins iterating
            # (i.e. before torch.randperm is called).  This will be stored in
            # every step-checkpoint so that future resumes can replay it.
            epoch_start_rng = _capture_rng()

            print(f"[TaggerTrainer] --- Epoch {epoch}/{epochs} ---")
            model.train()
            epoch_loss       = 0.0
            batches_processed = 0

            # When num_workers=0 (single-process), use a background thread to
            # prefetch the next batch while the GPU processes the current one.
            # PIL decode and NaFlex transforms release the GIL, so true CPU/GPU
            # parallelism is achievable.  With num_workers>0, the DataLoader
            # already handles prefetching via worker processes.
            if epoch == resume_epoch and resume_batch_idx >= 0:
                # Efficiently skip already-processed batches by reconstructing the
                # same shuffle permutation (RNG was restored to epoch-start state above)
                # and slicing off the first (resume_batch_idx + 1) batches.
                # This avoids loading/decoding thousands of images just to discard them.
                _bs = train_loader.batch_size or 1
                _ds = train_loader.dataset
                _full_perm = torch.randperm(len(_ds)).tolist()
                _skip_items = (resume_batch_idx + 1) * _bs
                _resume_subset = torch.utils.data.Subset(_ds, _full_perm[_skip_items:])
                _resume_loader = DataLoader(
                    _resume_subset, batch_size=_bs, shuffle=False,
                    num_workers=train_loader.num_workers,
                    collate_fn=tagger_collate_fn, pin_memory=False,
                )
                print(f"[TaggerTrainer] Skipped {resume_batch_idx + 1} batches efficiently "
                      f"(resume from batch {resume_batch_idx + 1})")
                _loader_for_epoch = _resume_loader
                _batch_idx_offset = resume_batch_idx + 1
            else:
                _loader_for_epoch = train_loader
                _batch_idx_offset = 0

            loader_iter = (
                _prefetch_loader(_loader_for_epoch) if _loader_for_epoch.num_workers == 0
                else iter(_loader_for_epoch)
            )
            for _loop_idx, batch in enumerate(loader_iter):
                batch_idx = _loop_idx + _batch_idx_offset

                if batch is None:
                    continue  # entire batch was corrupt images

                pv, pam, ss, labels, loss_masks = batch

                if self._stop_requested:
                    break

                pv         = pv.to(device)
                pam        = pam.to(device)
                ss         = ss.to(device)
                labels     = labels.to(device)
                loss_masks = loss_masks.to(device)

                optimizer.zero_grad()

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(pv, pam, ss)
                        loss   = criterion(logits, labels, loss_masks)
                else:
                    logits = model(pv, pam, ss)
                    loss   = criterion(logits, labels, loss_masks)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                global_step      += 1
                epoch_loss       += loss.item()
                batches_processed += 1

                # Progress callback every 10 steps
                if global_step % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else base_lr
                    self._emit("step", {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss.item(),
                        "lr": current_lr,
                        "progress": global_step / total_steps,
                    })

                # Step-based checkpoint (model + state + optimizer)
                if save_every_n_steps > 0 and global_step % save_every_n_steps == 0:
                    ckpt_name = f"step_{global_step:06d}"
                    metadata  = self._make_metadata(epoch, global_step, best_f1, best_threshold)
                    ckpt_path = _save_model_checkpoint(model, self.output_dir, ckpt_name, metadata, checkpoint_save_mode)
                    _save_training_state(
                        self.output_dir, ckpt_name,
                        epoch, global_step, batch_idx,
                        best_f1, best_threshold,
                        epoch_start_rng=epoch_start_rng,  # epoch-start RNG for exact replay
                    )
                    _save_optimizer_state(optimizer, self.output_dir, ckpt_name)
                    if keep_last_n_checkpoints > 0:
                        _prune_step_checkpoints(self.output_dir, keep_last_n_checkpoints)
                    self._emit("checkpoint", {
                        "name": ckpt_name,
                        "step": global_step,
                        "epoch": epoch,
                        "path": ckpt_path,
                    })

            # --- Stop checkpoint (mid-epoch or epoch-boundary) ---
            if self._stop_requested:
                print(f"[TaggerTrainer] Stop requested. Saving checkpoint at step {global_step}...")
                metadata = self._make_metadata(epoch, global_step, best_f1, best_threshold)
                ckpt_name = f"step_{global_step:06d}"
                ckpt_path = _save_model_checkpoint(model, self.output_dir, ckpt_name, metadata, checkpoint_save_mode)
                _save_training_state(
                    self.output_dir, ckpt_name,
                    epoch, global_step, batch_idx,
                    best_f1, best_threshold,
                    epoch_start_rng=epoch_start_rng,
                )
                _save_optimizer_state(optimizer, self.output_dir, ckpt_name)
                # Also update "latest" to the stop position
                _save_model_checkpoint(model, self.output_dir, "latest", metadata, checkpoint_save_mode)
                _save_training_state(
                    self.output_dir, "latest",
                    epoch, global_step, batch_idx,
                    best_f1, best_threshold,
                    epoch_start_rng=epoch_start_rng,
                )
                _save_optimizer_state(optimizer, self.output_dir, "latest")
                self._emit("checkpoint", {
                    "name": ckpt_name,
                    "step": global_step,
                    "epoch": epoch,
                    "path": ckpt_path,
                })
                print(f"[TaggerTrainer] Stopped. Checkpoint: {ckpt_name} (epoch {epoch}, step {global_step})")
                break  # exit epoch loop → skips validation, epoch-end save, _final_threshold_search

            avg_loss = epoch_loss / max(batches_processed, 1)

            # Validation
            # val_max_batches caps memory usage: num_tags=84k×batch×float16 can be huge.
            # Default 256 batches ≈ 4096 samples ≈ 689 MB for 84k-tag vocab.
            val_max_batches = int(cfg.get("val_max_batches", 256)) or None
            val_metrics: Dict[str, Any] = {}
            if val_loader and epoch % int(cfg.get("validate_every", 1)) == 0:
                val_metrics = self._validate(model, val_loader, device, amp_dtype if use_amp else None,
                                             max_batches=val_max_batches)
                epoch_f1  = val_metrics.get("f1", 0.0)
                epoch_thr = val_metrics.get("threshold", 0.35)

                if epoch_f1 > best_f1:
                    best_f1        = epoch_f1
                    best_threshold = epoch_thr
                    metadata = self._make_metadata(epoch, global_step, best_f1, best_threshold)
                    _save_model_checkpoint(model, self.output_dir, "best_f1", metadata, checkpoint_save_mode)
                    self._emit("checkpoint", {"name": "best_f1", "f1": best_f1, "epoch": epoch})

            # Save latest checkpoint at epoch boundary.
            # epoch+1 / batch_idx=-1 means "start of next epoch, no batches to skip".
            # epoch_start_rng is not needed for epoch-boundary resumes (batch_idx=-1
            # means we start the next epoch fresh and capture a new epoch_start_rng
            # at that time), so we omit it to keep the file compact.
            metadata = self._make_metadata(epoch, global_step, best_f1, best_threshold)
            _save_model_checkpoint(model, self.output_dir, "latest", metadata, checkpoint_save_mode)

            # Epoch-based checkpoint (model only; training state = same as latest)
            if save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0:
                ckpt_name = f"epoch_{epoch:04d}"
                _save_model_checkpoint(model, self.output_dir, ckpt_name, metadata, checkpoint_save_mode)
                self._emit("checkpoint", {"name": ckpt_name, "epoch": epoch, "step": global_step})
            _save_training_state(
                self.output_dir, "latest",
                epoch + 1, global_step, -1,
                best_f1, best_threshold,
            )
            _save_optimizer_state(optimizer, self.output_dir, "latest")

            epoch_summary = {
                "epoch": epoch,
                "loss": avg_loss,
                "step": global_step,
                **val_metrics,
            }
            metrics_history.append(epoch_summary)

            self._emit("epoch", {
                "epoch": epoch,
                "total_epochs": epochs,
                "loss": avg_loss,
                **val_metrics,
            })

        if self._stop_requested:
            # Training was stopped mid-run; skip threshold search and "completed" event
            return {
                "best_f1": best_f1,
                "best_threshold": best_threshold,
                "total_steps": global_step,
                "metrics_history": metrics_history,
            }

        # Final threshold grid search on validation set
        final_search = self._final_threshold_search(
            model, val_loader, device, amp_dtype if use_amp else None,
            max_batches=val_max_batches,
        )
        if final_search:
            best_threshold = final_search["optimal_threshold"]

        completed_data: Dict[str, Any] = {
            "best_f1": best_f1,
            "best_threshold": best_threshold,
            "total_steps": global_step,
        }
        if final_search:
            completed_data["threshold_f1_curve"] = final_search["threshold_f1_curve"]
            completed_data["optimal_threshold"]  = final_search["optimal_threshold"]

        self._emit("completed", completed_data)

        return {
            "best_f1": best_f1,
            "best_threshold": best_threshold,
            "total_steps": global_step,
            "metrics_history": metrics_history,
            **(final_search or {}),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        amp_dtype: Optional[torch.dtype],
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                if batch is None:
                    continue
                pv, pam, ss, labels, _ = batch
                pv    = pv.to(device)
                pam   = pam.to(device)
                ss    = ss.to(device)

                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(pv, pam, ss)
                else:
                    logits = model(pv, pam, ss)

                # float16 to halve memory usage (84k tags × many samples)
                probs = torch.sigmoid(logits).to(torch.float16).cpu()
                all_preds.append(probs)
                all_labels.append(labels.to(torch.float16))

        all_preds  = torch.cat(all_preds,  dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        threshold, f1 = _find_best_threshold(all_preds, all_labels)
        return {"f1": f1, "threshold": threshold}

    def _collect_val_preds(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        amp_dtype: Optional[torch.dtype],
        max_batches: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect validation predictions and labels as float16 tensors."""
        model.eval()
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                if batch is None:
                    continue
                pv, pam, ss, labels, _ = batch
                pv  = pv.to(device)
                pam = pam.to(device)
                ss  = ss.to(device)
                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(pv, pam, ss)
                else:
                    logits = model(pv, pam, ss)
                all_preds.append(torch.sigmoid(logits).to(torch.float16).cpu())
                all_labels.append(labels.to(torch.float16))
        return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)

    def _final_threshold_search(
        self,
        model: nn.Module,
        val_loader: Optional[DataLoader],
        device: torch.device,
        amp_dtype: Optional[torch.dtype],
        max_batches: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run full threshold grid search (0.05–0.95) on val set.

        Returns dict with 'threshold_f1_curve' and 'optimal_threshold',
        or None if val_loader is unavailable.
        """
        if val_loader is None:
            return None

        print("[TaggerTrainer] Running final threshold grid search...")
        self._emit("phase", {"phase": "threshold_search", "message": "Running threshold grid search..."})

        all_preds, all_labels = self._collect_val_preds(model, val_loader, device, amp_dtype,
                                                         max_batches=max_batches)

        thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 to 0.95
        curve: Dict[str, float] = {}
        for thr in thresholds:
            f1 = _compute_f1_macro(all_preds, all_labels, threshold=thr)
            curve[f"{thr:.2f}"] = round(f1, 6)

        best_thr_str = max(curve, key=lambda k: (curve[k], -float(k)))
        optimal_threshold = float(best_thr_str)
        print(f"[TaggerTrainer] Threshold grid done. Optimal: {optimal_threshold:.2f} F1={curve[best_thr_str]:.4f}")
        return {
            "threshold_f1_curve": curve,
            "optimal_threshold": optimal_threshold,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_metadata(
        self, epoch: int, step: int, best_f1: float, best_threshold: float
    ) -> Dict[str, Any]:
        from core.tagger.siglip2_tagger_model import SIGLIP2_DEFAULT_REPO_ID, _is_hf_repo_or_url
        return {
            "run_id": self.run_id,
            "num_tags": self.vocabulary.num_tags,
            "epoch": epoch,
            "step": step,
            "best_f1": best_f1,
            "best_threshold": best_threshold,
            "lora_rank": self.config.get("lora_rank"),
            "lora_alpha": self.config.get("lora_alpha"),
            "training_method": self.config.get("training_method"),
            "use_tag_aliases": bool(self.config.get("use_tag_aliases", False)),
            "category_counts": self.vocabulary.category_counts(),
            # Record the HuggingFace repo used for the vision encoder architecture.
            # Merged (full) checkpoints use this to reconstruct the model without
            # requiring vision_encoder_path at load time.
            "vision_encoder_repo": self.config.get("vision_encoder_repo", SIGLIP2_DEFAULT_REPO_ID),
            # For LoRA checkpoints trained on a locally fine-tuned base: relative
            # filename set by run_tagger_training when it copies the base into output_dir.
            **({
                "base_model_path": self.config["base_model_path"],
            } if self.config.get("base_model_path") else {}),
        }

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        if self.callback:
            try:
                self.callback(self.run_id, event_type, data)
            except Exception as e:
                print(f"[TaggerTrainer] Callback error: {e}")


# ------------------------------------------------------------------
# Entry point (called from API in a background thread/process)
# ------------------------------------------------------------------

def run_tagger_training(
    run_id: str,
    config: Dict[str, Any],
    dataset_ids: List[int],
    output_dir: str,
    progress_callback: Optional[Callable] = None,
    resume_from_checkpoint: Optional[str] = None,
    trainer_holder: Optional[List] = None,
) -> Dict[str, Any]:
    """Top-level function to build everything and start training.

    Called from the API route handler in a background thread.

    Parameters
    ----------
    resume_from_checkpoint : directory path to scan for a resumable checkpoint.
        If a valid checkpoint is found (latest_state.json or step_XXXXXX_state.json),
        training resumes from that point; otherwise starts from epoch 1.
    """
    from database import DatasetsSessionLocal

    datasets_db = DatasetsSessionLocal()
    try:
        # Build vocabulary
        print(f"[TaggerTraining] === Phase: vocabulary ===")
        print(f"[TaggerTraining] Building tag vocabulary from {len(dataset_ids)} dataset(s)...")
        progress_callback and progress_callback(run_id, "phase", {
            "phase": "vocabulary", "message": "Building tag vocabulary..."
        })
        excl_cats = config.get("excluded_categories") or None
        ban_tags  = config.get("ban_tags") or None
        if isinstance(ban_tags, str):
            ban_tags = [t.strip() for t in ban_tags.splitlines() if t.strip()] or None

        # Build tag alias resolver if requested
        use_tag_aliases = bool(config.get("use_tag_aliases", False))
        alias_resolver = None
        if use_tag_aliases:
            from .tag_alias_resolver import TagAliasResolver
            try:
                from config import settings as _settings
                alias_path = os.path.join(_settings.root_dir, "tagother", "tag_aliases.json")
            except Exception:
                alias_path = os.path.join(output_dir, "..", "..", "tagother", "tag_aliases.json")
            if os.path.isfile(alias_path):
                alias_resolver = TagAliasResolver.load(alias_path)
                print(f"[TaggerTraining] Loaded {len(alias_resolver)} tag aliases from {alias_path}")
            else:
                print(f"[TaggerTraining] WARNING: use_tag_aliases=True but tag_aliases.json not found at {alias_path}")

        vocabulary = TagVocabulary.build_from_dataset_ids(
            dataset_ids=dataset_ids,
            datasets_db=datasets_db,
            min_count=config.get("vocab_min_count", 1),
            excluded_categories=excl_cats,
            ban_tags=ban_tags,
            alias_resolver=alias_resolver,
        )
        print(f"[TaggerTraining] Vocabulary: {vocabulary.num_tags} tags")
        if excl_cats:
            print(f"[TaggerTraining] Excluded categories: {excl_cats}")
        if ban_tags:
            print(f"[TaggerTraining] Banned tags: {len(ban_tags)} entries")
        progress_callback and progress_callback(run_id, "vocab", {"num_tags": vocabulary.num_tags})

        # Build processor
        print(f"[TaggerTraining] === Phase: processor ===")
        print(f"[TaggerTraining] Loading processor...")
        REPO_ID = "google/siglip2-so400m-patch16-naflex"
        processor = AutoProcessor.from_pretrained(REPO_ID)
        print(f"[TaggerTraining] Processor loaded (repo: {REPO_ID})")

        # Build datasets
        print(f"[TaggerTraining] === Phase: dataset ===")
        print(f"[TaggerTraining] Building TaggerDataset ({vocabulary.num_tags} tags)...")
        progress_callback and progress_callback(run_id, "phase", {
            "phase": "dataset", "message": f"Loading dataset ({vocabulary.num_tags} tags)..."
        })
        val_split = float(config.get("val_split", 0.05))
        full_ds = TaggerDataset(
            dataset_ids=dataset_ids,
            vocabulary=vocabulary,
            datasets_db=datasets_db,
            processor=processor,
            alias_resolver=alias_resolver,
        )
        print(f"[TaggerTraining] Dataset: {len(full_ds)} samples total")

        val_size   = max(1, int(len(full_ds) * val_split))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"[TaggerTraining] Split: {train_size} train / {val_size} val (val_split={val_split})")

        batch_size  = int(config.get("batch_size", 32))
        num_workers = int(config.get("num_workers", 4))
        # num_workers_override: explicit override (int >= 0 forces that value; None = auto).
        # On Windows, spawn mode re-imports main.py per worker and pickles the full sample
        # list — this can cause MemoryError with large datasets.  The default now allows
        # the configured num_workers even on Windows; set num_workers_override=0 in config
        # to force single-process loading if MemoryErrors occur.
        import sys as _sys
        num_workers_override = config.get("num_workers_override")
        if num_workers_override is not None and int(num_workers_override) >= 0:
            effective_workers = int(num_workers_override)
        else:
            effective_workers = num_workers
        print(f"[TaggerTraining] DataLoader: batch_size={batch_size}, num_workers={effective_workers}")
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=effective_workers, collate_fn=tagger_collate_fn,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=effective_workers, collate_fn=tagger_collate_fn,
            pin_memory=False,
        )
        steps_per_epoch = len(train_loader)
        print(f"[TaggerTraining] Steps per epoch: {steps_per_epoch}")

        # ------------------------------------------------------------------
        # Detect resume checkpoint
        # ------------------------------------------------------------------
        resume_state: Optional[Dict[str, Any]] = None
        resume_ckpt_name: Optional[str] = None

        if resume_from_checkpoint:
            result = _find_resume_checkpoint(resume_from_checkpoint)
            if result is not None:
                resume_ckpt_name, resume_state = result
                print(
                    f"[TaggerTraining] Found resume checkpoint: {resume_ckpt_name} "
                    f"(step {resume_state['global_step']}, epoch {resume_state['epoch']})"
                )
            else:
                print("[TaggerTraining] No resumable checkpoint found; starting from scratch")

        # Resolve vision_encoder_repo for metadata recording and architecture loading.
        # Priority (highest to lowest):
        #   1. Explicit HF repo ID / URL in vision_encoder_path
        #   2. vision_encoder_repo already set in config (caller pre-filled)
        #   3. _metadata.json alongside a local safetensors vision_encoder_path
        #   4. Default (so400m)
        _ve_path = config.get("vision_encoder_path", "")
        from core.tagger.siglip2_tagger_model import _is_hf_repo_or_url, SIGLIP2_DEFAULT_REPO_ID as _DEFAULT_REPO
        _is_hf_ve, _resolved_ve = _is_hf_repo_or_url(_ve_path)
        if _is_hf_ve or "vision_encoder_repo" not in config:
            config = dict(config)  # shallow copy — do not mutate caller's dict
            if _is_hf_ve:
                config["vision_encoder_repo"] = _resolved_ve
                print(f"[TaggerTraining] HF repo detected for vision encoder: {_resolved_ve}")
            else:
                # Try to read vision_encoder_repo from the safetensors metadata
                # (covers merged tagger checkpoints used as vision_encoder_path)
                import json as _json
                _ve_meta_path = _ve_path.strip().strip('"').strip("'")
                _ve_meta_path = _ve_meta_path.replace(".safetensors", "_metadata.json")
                _repo_from_meta = None
                if os.path.isfile(_ve_meta_path):
                    try:
                        with open(_ve_meta_path, "r", encoding="utf-8") as _fh:
                            _ve_meta = _json.load(_fh)
                        _repo_from_meta = _ve_meta.get("vision_encoder_repo")
                    except Exception:
                        pass
                config["vision_encoder_repo"] = _repo_from_meta or _DEFAULT_REPO
                if _repo_from_meta:
                    print(f"[TaggerTraining] vision_encoder_repo read from checkpoint metadata: {_repo_from_meta}")

        # Save base model into training directory for self-contained checkpoints.
        # Controlled by save_base_model (default True).
        # - Local safetensors: copy file (+ _metadata.json) into output_dir
        # - HF repo: extract vision encoder weights via siglip2_extractor
        # Stores relative filename "base_model.safetensors" in config so _make_metadata
        # can write it without leaking absolute paths.
        if config.get("save_base_model", True):
            import shutil as _shutil
            _base_dst = os.path.join(output_dir, "base_model.safetensors")
            _ve_path_for_copy = config.get("vision_encoder_path", "")
            _is_hf_ve, _resolved_ve_id = _is_hf_repo_or_url(_ve_path_for_copy)

            if os.path.isfile(_base_dst):
                print(f"[TaggerTraining] Base model already present in training directory: {_base_dst}")
                config["base_model_path"] = "base_model.safetensors"
            elif _ve_path_for_copy and not _is_hf_ve and _ve_path_for_copy.endswith(".safetensors") and os.path.isfile(_ve_path_for_copy):
                # Local safetensors base — copy into training directory
                print(f"[TaggerTraining] Copying local base model to training directory: {_base_dst}")
                _shutil.copy2(_ve_path_for_copy, _base_dst)
                _ve_meta_src = _ve_path_for_copy.replace(".safetensors", "_metadata.json")
                if os.path.isfile(_ve_meta_src):
                    _shutil.copy2(_ve_meta_src, _base_dst.replace(".safetensors", "_metadata.json"))
                config["base_model_path"] = "base_model.safetensors"
            elif _is_hf_ve:
                # HF repo base — extract vision encoder weights
                _hf_repo = _resolved_ve_id or config.get("vision_encoder_repo", _DEFAULT_REPO)
                print(f"[TaggerTraining] Extracting vision encoder from HF repo '{_hf_repo}' → {_base_dst}")
                try:
                    from core.tagger.siglip2_extractor import extract_vision_encoder as _extract_ve
                    _extract_ve(_hf_repo, _base_dst)
                    # Write _metadata.json with vision_encoder_repo for later loading
                    _base_meta_dst = _base_dst.replace(".safetensors", "_metadata.json")
                    json.dump({"vision_encoder_repo": _hf_repo}, open(_base_meta_dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                    config["base_model_path"] = "base_model.safetensors"
                    print(f"[TaggerTraining] Vision encoder extracted: {_base_dst}")
                except Exception as _e:
                    print(f"[TaggerTraining] WARNING: could not extract base model from HF repo: {_e}")

        # Read the old vocabulary.json BEFORE TaggerTrainer overwrites it
        old_vocabulary: Optional[TagVocabulary] = None
        if resume_ckpt_name is not None:
            old_vocab_path = os.path.join(output_dir, "vocabulary.json")
            if os.path.isfile(old_vocab_path):
                try:
                    with open(old_vocab_path, "r", encoding="utf-8") as _f:
                        old_vocabulary = TagVocabulary.from_dict(json.load(_f))
                    print(f"[TaggerTraining] Loaded old vocabulary ({old_vocabulary.num_tags} tags) for head alignment")
                except Exception as _e:
                    print(f"[TaggerTraining] WARNING: could not load old vocabulary: {_e}")

        # Run trainer
        trainer = TaggerTrainer(
            run_id=run_id,
            config=config,
            vocabulary=vocabulary,
            output_dir=output_dir,
            progress_callback=progress_callback,
            old_vocabulary=old_vocabulary,
        )
        # Expose trainer reference so the API can call trainer.stop()
        if trainer_holder is not None:
            trainer_holder.append(trainer)

        # Load checkpoint weights into the model before training starts
        # (trainer.train() builds the model, so we pass resume info and let it handle loading)
        if resume_state is not None and resume_ckpt_name is not None:
            # Notify the API about the resume step for DB recording
            progress_callback and progress_callback(run_id, "resume", {
                "resumed_from_step": resume_state["global_step"],
                "resume_ckpt_name": resume_ckpt_name,
            })

        return trainer.train(
            train_loader, val_loader, processor,
            resume_state=resume_state,
            resume_ckpt_name=resume_ckpt_name,
        )

    finally:
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()
        print("[TaggerTrainer] GPU memory freed")
        datasets_db.close()
