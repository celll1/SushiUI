"""
SigLIP2 Tagger Training Loop.

Supports:
  - Full parameter training and LoRA training
  - Mixed precision (bf16 / fp16 / fp32)
  - Gradient checkpointing
  - Cosine LR schedule with linear warmup
  - Validation: F1 macro, threshold optimization
  - Checkpoint saving (best F1 + latest)
  - Progress callback for WebSocket updates
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from .siglip2_tagger_model import (
    SigLIP2TaggerLoRAModel,
    SigLIP2TaggerModel,
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
    ) -> None:
        self.run_id = run_id
        self.config = config
        self.vocabulary = vocabulary
        self.output_dir = output_dir
        self.callback = progress_callback
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
    ) -> Dict[str, Any]:
        """Run training loop. Returns summary metrics dict."""
        cfg = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self._emit("phase", {"phase": "initializing", "message": "Building model..."})
        model = build_tagger_model(
            training_method=cfg.get("training_method", "lora"),
            num_tags=self.vocabulary.num_tags,
            vision_encoder_path=cfg["vision_encoder_path"],
            lora_rank=cfg.get("lora_rank", 32),
            lora_alpha=float(cfg.get("lora_alpha", 16.0)),
            cls_dim=cfg.get("cls_dim") or None,
            hidden_proj_dim=cfg.get("hidden_proj_dim") or None,
        )
        model = model.to(device)

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
        # Build param groups: head gets higher LR
        head_params = list(model.head.parameters())
        head_ids     = {id(p) for p in head_params}
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

        # LR schedule: linear warmup → cosine decay
        epochs      = int(cfg.get("epochs", 10))
        warmup_steps = int(cfg.get("warmup_steps", 100))
        total_steps  = epochs * len(train_loader)

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

        # Training state
        best_f1         = 0.0
        best_threshold  = 0.35
        global_step     = 0
        metrics_history: List[Dict] = []

        self._emit("phase", {"phase": "training", "message": "Training started"})

        for epoch in range(1, epochs + 1):
            if self._stop_requested:
                break

            model.train()
            epoch_loss = 0.0

            for batch_idx, (pv, pam, ss, labels, loss_masks) in enumerate(train_loader):
                if self._stop_requested:
                    break

                pv        = pv.to(device)
                pam       = pam.to(device)
                ss        = ss.to(device)
                labels    = labels.to(device)
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
                global_step += 1
                epoch_loss  += loss.item()

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

            avg_loss = epoch_loss / max(len(train_loader), 1)

            # Validation
            val_metrics: Dict[str, Any] = {}
            if val_loader and epoch % int(cfg.get("validate_every", 1)) == 0:
                val_metrics = self._validate(model, val_loader, device, amp_dtype if use_amp else None)
                epoch_f1  = val_metrics.get("f1", 0.0)
                epoch_thr = val_metrics.get("threshold", 0.35)

                if epoch_f1 > best_f1:
                    best_f1        = epoch_f1
                    best_threshold = epoch_thr
                    metadata = self._make_metadata(epoch, global_step, best_f1, best_threshold)
                    model.save_checkpoint(self.output_dir, "best_f1", metadata)
                    self._emit("checkpoint", {"name": "best_f1", "f1": best_f1, "epoch": epoch})

            # Save latest checkpoint
            metadata = self._make_metadata(epoch, global_step, best_f1, best_threshold)
            model.save_checkpoint(self.output_dir, "latest", metadata)

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

        # Final threshold grid search on validation set
        final_search = self._final_threshold_search(
            model, val_loader, device, amp_dtype if use_amp else None
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
    ) -> Dict[str, Any]:
        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for pv, pam, ss, labels, _ in loader:
                pv    = pv.to(device)
                pam   = pam.to(device)
                ss    = ss.to(device)

                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(pv, pam, ss)
                else:
                    logits = model(pv, pam, ss)

                probs = torch.sigmoid(logits).cpu()
                all_preds.append(probs)
                all_labels.append(labels)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect all validation predictions and labels as tensors."""
        model.eval()
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for pv, pam, ss, labels, _ in loader:
                pv  = pv.to(device)
                pam = pam.to(device)
                ss  = ss.to(device)
                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(pv, pam, ss)
                else:
                    logits = model(pv, pam, ss)
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels)
        return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)

    def _final_threshold_search(
        self,
        model: nn.Module,
        val_loader: Optional[DataLoader],
        device: torch.device,
        amp_dtype: Optional[torch.dtype],
    ) -> Optional[Dict[str, Any]]:
        """Run full threshold grid search (0.05–0.95) on val set.

        Returns dict with 'threshold_f1_curve' and 'optimal_threshold',
        or None if val_loader is unavailable.
        """
        if val_loader is None:
            return None

        print("[TaggerTrainer] Running final threshold grid search...")
        self._emit("phase", {"phase": "threshold_search", "message": "Running threshold grid search..."})

        all_preds, all_labels = self._collect_val_preds(model, val_loader, device, amp_dtype)

        thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 to 0.95
        curve: Dict[str, float] = {}
        for thr in thresholds:
            f1 = _compute_f1_macro(all_preds, all_labels, threshold=thr)
            curve[f"{thr:.2f}"] = round(f1, 6)

        # Best threshold: max F1, tie-break by lowest threshold (prefer recall)
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
            "category_counts": self.vocabulary.category_counts(),
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
) -> Dict[str, Any]:
    """Top-level function to build everything and start training.

    Called from the API route handler in a background thread.
    """
    from database import DatasetsSessionLocal

    datasets_db = DatasetsSessionLocal()
    try:
        # Build vocabulary
        progress_callback and progress_callback(run_id, "phase", {
            "phase": "vocabulary", "message": "Building tag vocabulary..."
        })
        excl_cats = config.get("excluded_categories") or None
        ban_tags  = config.get("ban_tags") or None
        # ban_tags may be a newline-separated string from the UI
        if isinstance(ban_tags, str):
            ban_tags = [t.strip() for t in ban_tags.splitlines() if t.strip()] or None
        vocabulary = TagVocabulary.build_from_dataset_ids(
            dataset_ids=dataset_ids,
            datasets_db=datasets_db,
            min_count=config.get("vocab_min_count", 1),
            excluded_categories=excl_cats,
            ban_tags=ban_tags,
        )
        print(f"[TaggerTraining] Vocabulary: {vocabulary.num_tags} tags")

        # Build processor
        REPO_ID = "google/siglip2-so400m-patch16-naflex"
        processor = AutoProcessor.from_pretrained(REPO_ID)

        # Build datasets
        progress_callback and progress_callback(run_id, "phase", {
            "phase": "dataset", "message": f"Loading dataset ({vocabulary.num_tags} tags)..."
        })
        val_split = float(config.get("val_split", 0.05))
        full_ds = TaggerDataset(
            dataset_ids=dataset_ids,
            vocabulary=vocabulary,
            datasets_db=datasets_db,
            processor=processor,
        )

        val_size   = max(1, int(len(full_ds) * val_split))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        batch_size   = int(config.get("batch_size", 32))
        num_workers  = int(config.get("num_workers", 4))
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=tagger_collate_fn, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=tagger_collate_fn, pin_memory=True,
        )

        # Run trainer
        trainer = TaggerTrainer(
            run_id=run_id,
            config=config,
            vocabulary=vocabulary,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )
        return trainer.train(train_loader, val_loader, processor)

    finally:
        datasets_db.close()
