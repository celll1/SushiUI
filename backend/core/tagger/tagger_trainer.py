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
from collections import deque
from queue import Queue as _Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from PIL import Image as _PILImage
from safetensors.torch import load_file as _load_safetensors_file

from .siglip2_tagger_model import (
    SIGLIP2_DEFAULT_REPO_ID,
    SigLIP2TaggerLoRAModel,
    SigLIP2TaggerModel,
    _inherit_head,
    build_tagger_model,
)
from .tag_vocabulary import TagVocabulary, normalize_tag
from .tagger_dataset import TaggerDataset, tagger_collate_fn
from .tagger_loss import AsymmetricLossOptimized, CSASL, HCSASL, LASASL, FWBBCE

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

def _compute_all_metrics(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute macro F1, precision, and recall in a single pass.

    Only active tags (at least one positive sample in labels) are included.
    Returns a dict with keys 'f1', 'precision', 'recall'.
    """
    preds_bin = (all_preds >= threshold).float()
    tp = (preds_bin * all_labels).sum(dim=0)
    fp = (preds_bin * (1 - all_labels)).sum(dim=0)
    fn = ((1 - preds_bin) * all_labels).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    active = all_labels.sum(dim=0) > 0
    if active.sum() == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "f1":        f1[active].mean().item(),
        "precision": precision[active].mean().item(),
        "recall":    recall[active].mean().item(),
    }


def _compute_f1_macro(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Thin wrapper around _compute_all_metrics that returns only the F1 scalar."""
    return _compute_all_metrics(all_preds, all_labels, threshold)["f1"]


def _compute_pr_metrics(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Thin wrapper around _compute_all_metrics that returns precision and recall."""
    m = _compute_all_metrics(all_preds, all_labels, threshold)
    return {"precision": m["precision"], "recall": m["recall"]}


def _find_best_threshold(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """Find the threshold maximising macro F1.

    Two-stage search:
      1. Coarse grid 0.05–0.95 step 0.05 (19 points)
      2. Refinement around the best at 0.01 step (≤8 new points)

    Total ≤27 _compute_all_metrics calls.  Returns ``(best_threshold, best_f1)``
    where both values correspond to the same threshold (consistent).
    """
    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05..0.95

    f1_at_thr: Dict[float, float] = {}
    for thr in thresholds:
        f1_at_thr[thr] = _compute_all_metrics(all_preds, all_labels, threshold=thr)["f1"]

    best_thr = max(f1_at_thr, key=f1_at_thr.get)

    # Refinement: ±0.04 around the best at 0.01 step
    refine_candidates = [round(best_thr + d * 0.01, 2) for d in range(-4, 5)]
    refine = [t for t in refine_candidates
              if 0.01 <= t <= 0.99 and t not in f1_at_thr]
    for thr in refine:
        f1_at_thr[thr] = _compute_all_metrics(all_preds, all_labels, threshold=thr)["f1"]

    best_thr = max(f1_at_thr, key=f1_at_thr.get)
    return best_thr, f1_at_thr[best_thr]


# ------------------------------------------------------------------
# Prefetch helper
# ------------------------------------------------------------------

def _prefetch_loader(loader, stop_event: threading.Event, maxsize: int = 2):
    """Iterate *loader* in a background thread, yielding batches via a bounded queue.

    On Windows (num_workers=0) PIL decode and NaFlex processing release the GIL,
    so the background thread runs concurrently with GPU training in the main thread.
    maxsize=2 bounds memory to ~2 batches worth of tensors.

    stop_event: when set, the worker thread exits its loop so the DataLoader
    reference is released and workers/resources can be cleaned up promptly.
    """
    _END = object()
    q = _Queue(maxsize=maxsize)

    def _worker():
        try:
            for batch in loader:
                if stop_event.is_set():
                    break
                # Use a timeout-based put so we can check stop_event even when the
                # consumer has stopped reading (queue full after early break).
                while True:
                    if stop_event.is_set():
                        return
                    try:
                        q.put(batch, timeout=0.1)
                        break
                    except Exception:
                        continue
        finally:
            q.put(_END)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        item = q.get()
        if item is _END:
            break
        yield item


# ------------------------------------------------------------------
# Label statistics for π-aware loss functions
# ------------------------------------------------------------------

def _grow_criterion_buffers(criterion, new_size: int) -> int:
    """Pad any 1-D per-tag buffer on *criterion* (pi, gammas, label_weight, …)
    up to *new_size* when the vocabulary expands mid-training.

    New-tag entries get the buffer's mean (a neutral default) — approximate but
    keeps the loss's per-tag tensors aligned with the grown logits/labels.
    Scalar-gamma losses (simple ASL) register no such buffers, so this is a
    no-op for them. Returns the number of buffers grown.
    """
    grown = 0
    bufs = getattr(criterion, "_buffers", None)
    if not bufs:
        return 0
    for name, buf in list(bufs.items()):
        if buf is None or buf.dim() != 1 or buf.shape[0] >= new_size:
            continue
        pad = new_size - buf.shape[0]
        fill = float(buf.float().mean().item()) if buf.numel() > 0 else 0.0
        bufs[name] = torch.cat([buf, buf.new_full((pad,), fill)])
        grown += 1
    return grown


def _compute_label_stats(
    dataset,
    num_tags: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-label positive rate and counts from a TaggerDataset.

    Iterates label vectors only (no image I/O) for speed.

    Returns
    -------
    pi    : Tensor [num_tags]  positive rate, clipped to [1e-4, 1-1e-4]
    N_pos : Tensor [num_tags]  positive sample count (float)
    N_neg : Tensor [num_tags]  negative sample count (float)
    """
    # Support DataLoader wrappers that expose .dataset (e.g. Subset)
    ds = dataset
    while not hasattr(ds, "_samples") and hasattr(ds, "dataset"):
        ds = ds.dataset

    pos_counts = torch.zeros(num_tags, dtype=torch.float32)
    total = len(ds._samples)
    from tqdm import tqdm
    for _path, tags in tqdm(ds._samples, total=total, desc="[TaggerTraining] Computing label stats", unit="samples"):
        labels, _ = ds._build_label_and_mask(tags)
        pos_counts += labels

    N_pos = pos_counts
    N_neg = float(total) - pos_counts
    pi = (pos_counts / max(total, 1)).clamp(1e-4, 1.0 - 1e-4)
    return pi, N_pos, N_neg


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


def _compute_dataset_fingerprint(dataset: Any) -> Dict[str, Any]:
    """Fingerprint of the dataset structure for change detection on resume.

    Mirrors ``BaseTrainer._compute_dataset_fingerprint`` (LoRA/Full-FT)
    but works on the flat ``TaggerDataset`` shape (a single sequence of
    items rather than multiple ``Dataset`` objects).

    Only image-identifying info is hashed — caption changes do NOT
    invalidate the resume shuffle state (captions don't affect batch
    order, just labels).  When in-place caption edits trigger a
    pre-flight rescan that adds/removes items, ``items_in_db`` changes
    and the fingerprint mismatches → trainer restarts the epoch.
    """
    import hashlib

    paths: List[str] = []
    if hasattr(dataset, "items"):
        for it in dataset.items:
            try:
                p = it.get("image_path") if isinstance(it, dict) else getattr(it, "image_path", "")
            except Exception:
                p = ""
            paths.append(p or "")

    sorted_paths = sorted(paths)
    paths_hash = hashlib.md5("\n".join(sorted_paths).encode("utf-8")).hexdigest()

    return {
        "total_item_count": len(paths),
        "image_paths_hash": paths_hash,
    }


def _dataset_fingerprint_changed(
    saved: Optional[Dict[str, Any]],
    current: Dict[str, Any],
) -> bool:
    """Return True iff the dataset shape changed between save and now.

    A missing ``saved`` (old state without a fingerprint) is treated as
    "unchanged" for backwards compatibility — the user gets the old
    behavior on legacy checkpoints rather than a forced epoch restart.
    """
    if not saved:
        return False
    if saved.get("total_item_count") != current.get("total_item_count"):
        return True
    if saved.get("image_paths_hash") != current.get("image_paths_hash"):
        return True
    return False


def _save_training_state(
    output_dir: str,
    name: str,
    epoch: int,
    global_step: int,
    batch_idx: int,
    best_f1: float,
    best_threshold: float,
    epoch_start_rng: Optional[Dict[str, Any]] = None,
    dataset_fingerprint: Optional[Dict[str, Any]] = None,
) -> None:
    """Save training state JSON for resume.

    epoch_start_rng must be the RNG snapshot captured *before* iterating the
    DataLoader for `epoch`.  On resume, restoring this snapshot and re-iterating
    the DataLoader from batch 0 (while skipping ≤ batch_idx) reproduces the
    exact same shuffle permutation and therefore the exact same batch sequence.

    dataset_fingerprint (optional) records the dataset shape at save time
    so resume can detect mid-run dataset changes (added/removed items) and
    restart the current epoch from scratch rather than skipping arbitrary
    samples from a re-shuffled new order.
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
    if dataset_fingerprint is not None:
        state["dataset_fingerprint"] = dataset_fingerprint
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


def _save_vocabulary_snapshot(vocabulary: Any, output_dir: str, name: str) -> None:
    """Save a per-checkpoint vocabulary snapshot to ``<name>_vocabulary.json``.

    This pinpoints the exact tag→idx mapping that was active when this
    checkpoint was saved, so that resume / inference can re-align the head
    (and optimizer state) without depending on the singleton ``vocabulary.json``
    in *output_dir* — which is overwritten on every new run start.
    """
    try:
        path = os.path.join(output_dir, f"{name}_vocabulary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocabulary.to_dict(), f, ensure_ascii=False, indent=2)
    except Exception as e:   # noqa: BLE001
        print(f"[TaggerTrainer] WARNING: could not save vocabulary snapshot for {name}: {e}")


def _save_tag_metrics(
    accumulator: Any,
    output_dir: str,
    name: str,
    vocabulary: Any,
    epoch_boundary: bool,
    save_enabled: bool = True,
    hard_lo: float = 0.25,
    hard_hi: float = 0.75,
    calib_method: str = "jeffreys",
    calib_eps: float = 0.5,
    calib_prior_strength: float = 10.0,
) -> None:
    """Save per-tag threshold metrics alongside a checkpoint.

    Skipped silently when ``save_enabled=False`` or when the accumulator has
    not yet processed any data (e.g. step-0 checkpoint before the first batch).
    """
    if not save_enabled:
        return
    if not accumulator.has_data:
        return
    try:
        path = os.path.join(output_dir, f"{name}_tag_metrics.npz")
        tag_names = [vocabulary.idx_to_tag[i] for i in range(vocabulary.num_tags)]
        accumulator.save(
            path,
            epoch_boundary=epoch_boundary,
            tag_names=tag_names,
            hard_lo=hard_lo,
            hard_hi=hard_hi,
            calib_method=calib_method,
            calib_eps=calib_eps,
            calib_prior_strength=calib_prior_strength,
        )
    except Exception as e:   # noqa: BLE001
        print(f"[TaggerTrainer] WARNING: could not save tag_metrics for {name}: {e}")


def _save_ood_reference(
    accumulator: Any,
    output_dir: str,
    name: str,
    save_enabled: bool = True,
) -> None:
    """Fit and save the OOD reference distribution alongside a checkpoint.

    Skipped when *save_enabled* is False or fewer than 10 embeddings have been
    collected.  Saves ``{name}_ood_ref.npz`` in *output_dir*.
    Also keeps a ``latest_ood_reservoir.npz`` so that training can be resumed
    without re-collecting embeddings from scratch.
    """
    if not save_enabled:
        return
    if accumulator.n_seen < 10:
        return
    try:
        path = os.path.join(output_dir, f"{name}_ood_ref.npz")
        accumulator.finalize(path)
        # Persist raw reservoir for resume
        reservoir_path = os.path.join(output_dir, "latest_ood_reservoir.npz")
        accumulator.save_reservoir(reservoir_path)
    except Exception as _e:
        print(f"[TaggerTrainer] WARNING: could not save ood_reference for {name}: {_e}")


def _resolve_checkpoint_vocab_path(output_dir: str, ckpt_name: str) -> Optional[str]:
    """Resolve the vocabulary file for a specific checkpoint.

    Priority:
      1. Per-checkpoint snapshot ``<name>_vocabulary.json`` (preferred — frozen
         at save time, immune to later overwrites).
      2. Common ``vocabulary.json`` in ``output_dir`` (fallback — may have been
         overwritten by a subsequent run; emits a WARNING).

    Returns the resolved absolute path or ``None`` if neither file exists.
    """
    per_ckpt = os.path.join(output_dir, f"{ckpt_name}_vocabulary.json")
    if os.path.isfile(per_ckpt):
        return per_ckpt
    common = os.path.join(output_dir, "vocabulary.json")
    if os.path.isfile(common):
        print(f"[TaggerTrainer] WARNING: per-checkpoint vocabulary "
              f"{ckpt_name}_vocabulary.json not found; falling back to "
              f"vocabulary.json. The fallback file may have been overwritten "
              f"by a later run — tag→idx alignment for {ckpt_name} cannot be "
              f"verified.")
        return common
    return None


def _load_optimizer_state(
    optimizer: Any,
    output_dir: str,
    name: str,
    *,
    model: Optional[Any] = None,
    old_tag_to_idx: Optional[Dict[str, int]] = None,
    new_tag_to_idx: Optional[Dict[str, int]] = None,
    lineage: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """Load optimizer state dict from ``<name>_optimizer.pt``.

    When all three of ``model``, ``old_tag_to_idx``, ``new_tag_to_idx``
    are supplied AND the vocabulary mapping differs (size OR content),
    the head's per-parameter state tensors (``exp_avg``, ``exp_avg_sq``,
    ``state1``, ``state2``) are tag-name aligned to the new shape before
    loading.  Block-wise quantisation metadata (``absmax*``, ``new_max*``)
    is reset and recomputed by bnb on the next ``step()``.

    Note: a content mismatch with identical size occurs when alias edits
    cause tag merges to be balanced by additions, or whenever vocabulary
    construction is non-deterministic between runs.  Without migration,
    PyTorch's ``load_state_dict`` would byte-copy old momentum into the
    new param, scrambling per-tag history across unrelated tags — so the
    check must be by content, not just shape.

    Returns True if loaded.
    """
    path = os.path.join(output_dir, f"{name}_optimizer.pt")
    if not os.path.isfile(path):
        return False
    state = torch.load(path, map_location="cpu", weights_only=True)

    # Migrate head's per-parameter state when vocabulary changed.
    # Failures here are non-fatal — fall through to the raw load (which
    # may produce a temporary loss spike but not break training).  The
    # head WEIGHTS themselves are already migrated upstream by
    # ``_inherit_head``; this only concerns the optimizer momentum.
    _vocab_changed = (
        old_tag_to_idx is not None
        and new_tag_to_idx is not None
        and old_tag_to_idx != new_tag_to_idx   # dict equality: keys + values
    )
    if (
        model is not None
        and _vocab_changed
        and getattr(model, "head", None) is not None
    ):
        try:
            from .optimizer_state_migrate import migrate_head_optimizer_state
            head_bias = getattr(model.head, "bias", None)
            summary = migrate_head_optimizer_state(
                saved_state=state,
                optimizer=optimizer,
                head_weight=model.head.weight,
                head_bias=head_bias,
                old_tag_to_idx=old_tag_to_idx,
                new_tag_to_idx=new_tag_to_idx,
                lineage=lineage,
            )
            _mode = summary.get("weight", {}).get("mode", "?")
            if _mode == "reset_8bit":
                print(
                    f"[TaggerTrainer] Optimizer head state RESET for 8-bit "
                    f"({summary['head_weight_shape_old']} -> {summary['head_weight_shape_new']}). "
                    f"bnb will re-initialise head momentum on the next step "
                    f"(expect a small 1-step loss perturbation). "
                    f"Vision encoder / LoRA / bias states are unaffected."
                )
            else:
                print(
                    f"[TaggerTrainer] Optimizer head state migrated: "
                    f"old shape {summary['head_weight_shape_old']} -> "
                    f"new shape {summary['head_weight_shape_new']}, "
                    f"weight stats={summary['weight']}, bias stats={summary['bias']}"
                )
        except Exception as e:   # noqa: BLE001
            print(
                f"[TaggerTrainer] WARNING: head optimizer state migration failed: {e}; "
                f"loading raw state (may produce loss spike). "
                f"Head weights themselves are still re-aligned via _inherit_head."
            )

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
        for suffix in (".safetensors", "_state.json", "_optimizer.pt", "_vocabulary.json"):
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
        vocab_lineage: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.run_id = run_id
        self.config = config
        self.vocabulary = vocabulary
        self.output_dir = output_dir
        self.callback = progress_callback
        self.old_vocabulary = old_vocabulary
        # new_tag -> [old_predecessor, ...] for head / optimizer-state inheritance
        # across vocab renames (alias) and merges (comma re-join). See vocab_lineage.py.
        self.vocab_lineage = vocab_lineage or {}
        self._stop_requested = False
        self._stop_event = threading.Event()
        # GPU coordinator pause / resume signalling.  Created here so the
        # handle exists before train() runs and the trainer thread doesn't
        # have to allocate Events while holding the model.
        self._pause_event    = threading.Event()
        self._resumed_event  = threading.Event()
        self._restored_event = threading.Event()
        from core.tagger.tagger_offload import TaggerTrainerHandle
        self._coordinator_handle = TaggerTrainerHandle(
            run_id=run_id,
            output_dir=output_dir,
            pause_event=self._pause_event,
            resumed_event=self._resumed_event,
            restored_event=self._restored_event,
        )

        os.makedirs(output_dir, exist_ok=True)

        # Save vocabulary snapshot
        vocab_path = os.path.join(output_dir, "vocabulary.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocabulary.to_dict(), f, ensure_ascii=False, indent=2)

    def stop(self) -> None:
        self._stop_requested = True
        self._stop_event.set()

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
        # FlashAttention-2 for the encoder self-attention (opt-in per run).
        # Three guards before we commit to FA2, each falling back to SDPA:
        #   1. dtype  — FA2 only engages under fp16/bf16 autocast (not fp32).
        #   2. avail. — flash-attn must be installed/usable in this environment.
        #   3. build  — if the FA2 model build still fails, retry with SDPA.
        _use_fa2 = bool(cfg.get("use_flash_attention", False))
        _mp_str = str(cfg.get("mixed_precision", "bf16")).lower()
        _attn_impl = "sdpa"
        if _use_fa2:
            if _mp_str not in ("bf16", "fp16"):
                print(f"[TaggerTrainer] FlashAttention-2 requested but mixed_precision={_mp_str} "
                      f"(FA2 needs bf16/fp16) — falling back to SDPA")
            else:
                try:
                    from transformers.utils import is_flash_attn_2_available
                    _fa2_ok = bool(is_flash_attn_2_available())
                except Exception:
                    _fa2_ok = False
                if _fa2_ok:
                    _attn_impl = "flash_attention_2"
                    print(f"[TaggerTrainer] Using FlashAttention-2 for encoder self-attention "
                          f"(mixed_precision={_mp_str})")
                else:
                    print("[TaggerTrainer] FlashAttention-2 requested but not available "
                          "(flash-attn not installed or unsupported here) — falling back to SDPA")

        def _build_model(_attn: str):
            return build_tagger_model(
                training_method=cfg.get("training_method", "lora"),
                num_tags=self.vocabulary.num_tags,
                vision_encoder_path=cfg["vision_encoder_path"],
                lora_rank=cfg.get("lora_rank", 32),
                lora_alpha=float(cfg.get("lora_alpha", 16.0)),
                cls_dim=cfg.get("cls_dim") or None,
                hidden_proj_dim=cfg.get("hidden_proj_dim") or None,
                init_head_from=cfg.get("init_head_from") or None,
                new_vocab=self.vocabulary.tag_to_idx,
                lineage=self.vocab_lineage,
                repo_id=cfg.get("vision_encoder_repo", SIGLIP2_DEFAULT_REPO_ID),
                is_naflex=cfg.get("is_naflex", True),
                attn_implementation=_attn,
            )

        try:
            model = _build_model(_attn_impl)
        except Exception as _be:
            if _attn_impl == "flash_attention_2":
                print(f"[TaggerTrainer] FlashAttention-2 model build failed "
                      f"({type(_be).__name__}: {str(_be)[:140]}) — retrying with SDPA")
                _attn_impl = "sdpa"
                model = _build_model(_attn_impl)
            else:
                raise
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count     = sum(p.numel() for p in model.parameters())
        print(f"[TaggerTrainer] Model built: {trainable_count:,} trainable / {total_count:,} total parameters")
        model = model.to(device)
        print(f"[TaggerTrainer] Model moved to {device}")
        # Expose the live model so the Danbooru vocab-expansion callback (defined
        # in run_tagger_training, outside train()'s scope) can grow the head.
        self.model = model

        # Load checkpoint weights if resuming (before any training)
        if resume_state is not None and resume_ckpt_name is not None:
            ckpt_path = os.path.join(self.output_dir, f"{resume_ckpt_name}.safetensors")
            if os.path.isfile(ckpt_path):
                _saved = _load_safetensors_file(ckpt_path)
                _ckpt_num_tags = _saved["head.weight"].shape[0] if "head.weight" in _saved else None
                _new_tag_to_idx = self.vocabulary.tag_to_idx
                _old_tag_to_idx = self.old_vocabulary.tag_to_idx if self.old_vocabulary else None

                # Detect mismatch: size differs OR same size but tag→idx mapping differs.
                # The latter happens when alias additions caused some tags to merge while
                # others were added (net zero count change), or whenever the deterministic
                # ordering of vocabulary construction shifted between runs.  A pure
                # positional copy in that case scrambles head rows across unrelated tags,
                # which is far worse than a clean reset.
                _size_mismatch    = _ckpt_num_tags is not None and _ckpt_num_tags != self.vocabulary.num_tags
                _mapping_mismatch = (
                    _old_tag_to_idx is not None
                    and _old_tag_to_idx != _new_tag_to_idx
                )
                _vocab_mismatch = _size_mismatch or _mapping_mismatch

                if _vocab_mismatch:
                    if _size_mismatch:
                        print(f"[TaggerTrainer] Vocabulary size mismatch: checkpoint={_ckpt_num_tags}, "
                              f"current={self.vocabulary.num_tags}. Aligning by tag name...")
                    else:
                        print(f"[TaggerTrainer] Vocabulary mapping mismatch (same size {self.vocabulary.num_tags} "
                              f"but tag→idx differs). Aligning by tag name...")
                    if _old_tag_to_idx is None:
                        print("[TaggerTrainer] WARNING: old_vocabulary unavailable, "
                              "_inherit_head will fall back to vocabulary.json next to checkpoint")
                    _inherit_head(
                        model=model,
                        checkpoint_path=ckpt_path,
                        new_num_tags=self.vocabulary.num_tags,
                        new_vocab=_new_tag_to_idx,
                        old_tag_to_idx=_old_tag_to_idx,
                        lineage=self.vocab_lineage,
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
                    # Non-head, non-LoRA weights (vision encoder, pooler, etc.) are vocab-
                    # independent — copy them by name from the saved state.
                    _vocab_independent: Dict[str, torch.Tensor] = {
                        k: v for k, v in _saved.items()
                        if not (k.startswith("head.") or k.startswith("lora."))
                    }
                    if _vocab_independent:
                        _missing, _unexpected = model.load_state_dict(_vocab_independent, strict=False)
                        # Remove head/LoRA from missing (they are loaded separately)
                        _missing = [m for m in _missing if not (m.startswith("head.") or m.startswith("lora."))]
                        if _missing:
                            print(f"[TaggerTrainer] WARNING: missing keys during vocab-independent load: {_missing[:5]}{'...' if len(_missing) > 5 else ''}")
                        if _unexpected:
                            print(f"[TaggerTrainer] WARNING: unexpected keys during vocab-independent load: {_unexpected[:5]}{'...' if len(_unexpected) > 5 else ''}")
                else:
                    model.load_weights_inplace(ckpt_path)
                    print(f"[TaggerTrainer] Loaded checkpoint weights from {resume_ckpt_name}")

        # Gradient checkpointing
        # Must be enabled via gradient_checkpointing_enable() so that the flag is
        # set on each Siglip2EncoderLayer (GradientCheckpointingLayer subclass).
        # Setting it on Siglip2Encoder directly has no effect.
        if cfg.get("gradient_checkpointing", True):
            if hasattr(model, "vision_encoder") and hasattr(model.vision_encoder, "gradient_checkpointing_enable"):
                model.vision_encoder.gradient_checkpointing_enable()
                print("[Trainer] Gradient checkpointing enabled (via gradient_checkpointing_enable)")
            elif hasattr(model, "vision_encoder") and hasattr(model.vision_encoder, "encoder"):
                # Fallback: set flag on each encoder layer directly
                for layer in model.vision_encoder.encoder.layers:
                    if hasattr(layer, "gradient_checkpointing"):
                        layer.gradient_checkpointing = True
                print("[Trainer] Gradient checkpointing enabled (per-layer fallback)")

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
        # Expose the live optimizer for the vocab-expansion callback (see above).
        self.optimizer = optimizer

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

        # ----------------------------------------------------------------
        # Optional: build LR matrix for conditional inference
        # ----------------------------------------------------------------
        # This precomputes co-occurrence statistics so that the inference
        # manager can apply context corrections.  We share the dataset's
        # already-loaded sample list (no DB rescan).  Resume runs always
        # regenerate so the matrix matches the current dataset configuration.
        if bool(cfg.get("build_lr_matrix_on_start", False)):
            try:
                from core.tagger.lr_matrix_builder import build_lr_matrix
                # Unwrap Subset wrappers (matches _compute_label_stats logic)
                _ds = train_loader.dataset
                while not hasattr(_ds, "_samples") and hasattr(_ds, "dataset"):
                    _ds = _ds.dataset
                _samples = [tags for _path, tags in _ds._samples]
                lr_path = os.path.join(self.output_dir, "lr_matrix.npz")
                print(f"[TaggerTrainer] === Phase: LR matrix ===")
                self._emit("phase", {"phase": "lr_matrix", "message": "Building LR matrix..."})
                build_lr_matrix(
                    output_path=lr_path,
                    samples=_samples,
                    tag_to_idx=self.vocabulary.tag_to_idx,
                    n_tags=self.vocabulary.num_tags,
                    top_anchors=int(cfg.get("lr_top_anchors", 10000)),
                    top_targets=int(cfg.get("lr_top_targets", 1000)),
                    lr_threshold=float(cfg.get("lr_threshold", 1.0)),
                    min_anchor_count=int(cfg.get("lr_min_anchor_count", 10)),
                )
                print(f"[TaggerTrainer] LR matrix saved -> {lr_path}")
            except Exception as e:
                # Don't fail training if LR matrix build fails; just log.
                print(f"[TaggerTrainer] WARNING: LR matrix build failed: {e}")

        # Loss function
        loss_fn_name = cfg.get("loss_function", "asl")
        if loss_fn_name == "asl":
            criterion = AsymmetricLossOptimized(
                gamma_neg=float(cfg.get("loss_gamma_neg", 4.0)),
                gamma_pos=float(cfg.get("loss_gamma_pos", 1.0)),
                clip=float(cfg.get("loss_clip", 0.05)),
            ).to(device)
        elif loss_fn_name in ("cs_asl", "h_cs_asl", "la_s_asl", "fw_bbce"):
            _pi, _N_pos, _N_neg = _compute_label_stats(train_loader.dataset, self.vocabulary.num_tags)
            _pi   = _pi.to(device)
            _N_pos = _N_pos.to(device)
            _N_neg = _N_neg.to(device)
            print(f"[TaggerTrainer] π_n stats: mean={_pi.mean():.4f} "
                  f"min={_pi.min():.4f} max={_pi.max():.4f}")
            # Save label stats for inference-time CS-ASL logit bias correction
            if loss_fn_name in ("cs_asl", "h_cs_asl"):
                import numpy as np
                _stats_path = os.path.join(self.output_dir, "label_stats.npz")
                np.savez(
                    _stats_path,
                    pi=_pi.cpu().numpy().astype(np.float32),
                    loss_fn=np.array([loss_fn_name]),
                    rho=np.array([float(cfg.get("loss_rho", 0.5))], dtype=np.float32),
                )
                print(f"[TaggerTrainer] Saved label stats -> {_stats_path}")
            _kw = dict(
                pi=_pi,
                gamma0=float(cfg.get("loss_gamma0", 4.0)),
                m0=float(cfg.get("loss_m0", 0.2)),
                beta=float(cfg.get("loss_beta", 2.0)),
                clip=float(cfg.get("loss_clip", 0.0)),
                eps=1e-4,
            )
            if loss_fn_name == "cs_asl":
                criterion = CSASL(**_kw, rho=float(cfg.get("loss_rho", 0.5))).to(device)
            elif loss_fn_name == "h_cs_asl":
                criterion = HCSASL(
                    **_kw,
                    rho=float(cfg.get("loss_rho", 0.5)),
                    N_pos=_N_pos, N_neg=_N_neg,
                    label_weight=cfg.get("loss_label_weight", "fisher"),
                ).to(device)
            elif loss_fn_name == "la_s_asl":
                criterion = LASASL(**_kw).to(device)
            elif loss_fn_name == "fw_bbce":
                criterion = FWBBCE(pi=_pi, N_pos=_N_pos, N_neg=_N_neg, eps=1e-4).to(device)
        else:
            raise ValueError(f"Unknown loss_function: {loss_fn_name!r}")

        # Expose for the Danbooru vocab-expansion callback (defined in
        # run_tagger_training) so it can grow per-tag structures when the
        # vocabulary expands mid-training. See _expansion_callback.
        self.criterion = criterion

        # Register with the GPU coordinator so image-generation requests can
        # pause us at the next batch boundary.  Cleanup happens at the end of
        # train() and defensively in run_tagger_training()'s finally.
        from core.gpu_coordinator import gpu_coordinator
        self._coordinator_handle.attach(model, optimizer, criterion,
                                        processor=processor,
                                        vocabulary=self.vocabulary)
        gpu_coordinator.register_trainer(self._coordinator_handle)

        # Step-based checkpoint interval (0 = disabled)
        save_every_n_steps    = int(cfg.get("save_every_n_steps", 500))
        # Epoch-based checkpoint interval (0 = disabled)
        save_every_n_epochs   = int(cfg.get("save_every_n_epochs", 0))
        # How many step checkpoints to keep (0 = keep all)
        keep_last_n_checkpoints = int(cfg.get("keep_last_n_checkpoints", 3))
        # "lora" = save LoRA+head only (compact); "merged" = merge LoRA into encoder and save full model
        checkpoint_save_mode = cfg.get("checkpoint_save_mode", "lora")

        # Training F1 rolling buffer
        _n2_eval      = int(cfg.get("train_f1_eval_every_n_steps", 100))
        _n1_search    = int(cfg.get("train_f1_threshold_search_every_n_steps", 500))
        # Train-count deficiency augmentation: needs tag_count to keep
        # accumulating even when training-F1 metrics are disabled.
        _train_count_on = bool(cfg.get("danbooru_train_count_enable", False))
        _buf_size     = max(int(cfg.get("train_f1_buffer_batches", 16)), 1)
        _f1_threshold = float(cfg.get("train_f1_initial_threshold", 0.35))
        _train_f1_buffer: deque = deque(maxlen=_buf_size)
        self._train_f1_buffer = _train_f1_buffer  # exposed for vocab-expansion reset

        # Per-tag threshold metrics accumulator
        from core.tagger.tag_metrics_accumulator import TagMetricsAccumulator
        _save_tag_metrics_enabled = bool(cfg.get("save_tag_metrics", True))
        _hard_lo = float(cfg.get("hard_rate_lo", 0.25))
        _hard_hi = float(cfg.get("hard_rate_hi", 0.75))
        _calib_method         = str(cfg.get("calib_method", "jeffreys"))
        _calib_eps            = float(cfg.get("calib_eps", 0.5))
        _calib_prior_strength = float(cfg.get("calib_prior_strength", 10.0))
        _tag_metrics_acc = TagMetricsAccumulator(vocab_size=self.vocabulary.num_tags)
        self._tag_metrics_acc = _tag_metrics_acc  # exposed for vocab-expansion grow

        # OOD embedding accumulator (reservoir sampling of CLS embeddings)
        from core.tagger.ood_embedding_accumulator import OodEmbeddingAccumulator
        _save_ood_ref_enabled = bool(cfg.get("save_ood_reference", True))
        _ood_emb_acc = OodEmbeddingAccumulator(max_samples=4000)

        # Training state
        best_f1         = 0.0
        best_threshold  = 0.5
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

        # Compute the *current* dataset fingerprint up-front so we can both
        # (a) compare it to the saved one to detect mid-run dataset changes
        #     (e.g. caused by a pre-flight rescan adding/removing items), and
        # (b) include it in every state file we save below.
        current_fingerprint = _compute_dataset_fingerprint(train_loader.dataset)
        print(
            f"[TaggerTrainer] Dataset fingerprint: {current_fingerprint['total_item_count']} items, "
            f"hash={current_fingerprint['image_paths_hash'][:8]}..."
        )

        if resume_state is not None:
            resume_epoch     = resume_state["epoch"]         # next epoch to train
            resume_batch_idx = resume_state["batch_idx"]     # last completed batch (-1 = full epoch done)
            global_step      = resume_state["global_step"]
            best_f1          = resume_state.get("best_f1", 0.0)
            best_threshold   = resume_state.get("best_threshold", 0.5)
            epoch_start_rng_for_resume = resume_state.get("epoch_start_rng")

            # Dataset-change detection: if the on-disk dataset differs from
            # the one this checkpoint was saved against, the saved RNG +
            # batch_idx no longer point at the same samples.  Restart the
            # current epoch from scratch with a fresh shuffle — global_step
            # and optimizer state are preserved.
            saved_fingerprint = resume_state.get("dataset_fingerprint")
            if _dataset_fingerprint_changed(saved_fingerprint, current_fingerprint):
                print(
                    f"[TaggerTrainer] WARNING: dataset changed since checkpoint "
                    f"(saved={saved_fingerprint}, current={current_fingerprint})"
                )
                print(
                    f"[TaggerTrainer] Restarting epoch {resume_epoch} from batch 0 "
                    f"(global_step={global_step} preserved)"
                )
                resume_batch_idx           = -1
                epoch_start_rng_for_resume = None

            # Restore optimizer state.  Pass the old/new vocab maps so that
            # the head's per-parameter state can be tag-name aligned when the
            # vocabulary size changed between the saved checkpoint and the
            # current run (see _load_optimizer_state docstring).
            if resume_ckpt_name:
                loaded = _load_optimizer_state(
                    optimizer,
                    self.output_dir,
                    resume_ckpt_name,
                    model=model,
                    old_tag_to_idx=(
                        self.old_vocabulary.tag_to_idx
                        if self.old_vocabulary is not None else None
                    ),
                    new_tag_to_idx=self.vocabulary.tag_to_idx,
                    lineage=self.vocab_lineage,
                )
                if loaded:
                    print(f"[TaggerTrainer] Optimizer state restored from {resume_ckpt_name}")

            # Restore tag metrics accumulator from latest_tag_metrics.npz so
            # histogram data is not discarded across resume boundaries.
            if _save_tag_metrics_enabled:
                _metrics_npz = os.path.join(self.output_dir, "latest_tag_metrics.npz")
                restored = _tag_metrics_acc.restore_from_npz(_metrics_npz)
                if restored:
                    print(
                        f"[TaggerTrainer] Tag metrics accumulator restored from "
                        f"latest_tag_metrics.npz "
                        f"(prev={_tag_metrics_acc.total_images_prev:,}, "
                        f"pp={_tag_metrics_acc.total_images_pp:,}, "
                        f"all={_tag_metrics_acc.total_images_all:,})"
                    )
                else:
                    print(
                        f"[TaggerTrainer] WARNING: latest_tag_metrics.npz not found or "
                        f"incompatible — accumulator starts fresh"
                    )

            # Restore OOD embedding reservoir for resume continuity
            if _save_ood_ref_enabled:
                _ood_reservoir_npz = os.path.join(self.output_dir, "latest_ood_reservoir.npz")
                _ood_emb_acc.restore_from_reservoir(_ood_reservoir_npz)

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
        # Forward hook to capture CLS embeddings for OOD reference building.
        # The hook fires on model.head (the final Linear layer), capturing its
        # input (= pooled CLS embedding).  A one-shot flag prevents double
        # capture during gradient-checkpointing recomputation.
        # ------------------------------------------------------------------
        _ood_cap_buf: list = []  # receives (B, D) float32 CPU tensors
        _ood_cap_once = [False]  # one-shot flag: reset before forward, cleared by hook

        def _ood_forward_pre_hook(module, args):
            if _ood_cap_once[0]:
                _ood_cap_once[0] = False
                inp = args[0]  # (B, D) on GPU, possibly bf16/fp16
                _ood_cap_buf.append(inp.detach().float().cpu())

        _ood_hook_handle = model.head.register_forward_pre_hook(_ood_forward_pre_hook)

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
                # If Danbooru augmentation is active (train_loader is a
                # MixedDataLoader), re-wrap the rebuilt base loader so the
                # interrupt-batch injection continues for the resumed epoch.
                # Without this, resuming mid-epoch silently disables Danbooru
                # injection until the next epoch boundary.
                if hasattr(train_loader, "rewrap"):
                    _resume_loader = train_loader.rewrap(_resume_loader)
                print(f"[TaggerTrainer] Skipped {resume_batch_idx + 1} batches efficiently "
                      f"(resume from batch {resume_batch_idx + 1})")
                _loader_for_epoch = _resume_loader
                _batch_idx_offset = resume_batch_idx + 1
            else:
                _loader_for_epoch = train_loader
                _batch_idx_offset = 0

            # Tell the mixed loader which epoch this is so the Danbooru buffer can
            # restore a matching per-epoch collection progress on a mid-epoch
            # resume (continue collection instead of restarting from the top).
            if hasattr(_loader_for_epoch, "set_epoch"):
                _loader_for_epoch.set_epoch(epoch)

            loader_iter = (
                _prefetch_loader(_loader_for_epoch, self._stop_event) if _loader_for_epoch.num_workers == 0
                else iter(_loader_for_epoch)
            )
            for _loop_idx, _yielded in enumerate(loader_iter):
                batch_idx = _loop_idx + _batch_idx_offset

                # MixedDataLoader yields (payload, is_injection); plain loaders
                # yield payload directly. Normalize here so the rest of the loop
                # is uniform.
                if (isinstance(_yielded, tuple) and len(_yielded) == 2
                        and isinstance(_yielded[1], bool)):
                    batch, is_injection_batch = _yielded
                else:
                    batch, is_injection_batch = _yielded, False

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

                # Arm the OOD hook (one-shot flag) before the forward pass
                if _save_ood_ref_enabled:
                    _ood_cap_once[0] = True

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(pv, pam, ss)
                        loss   = criterion(logits, labels, loss_masks)
                else:
                    logits = model(pv, pam, ss)
                    loss   = criterion(logits, labels, loss_masks)

                # Feed any captured CLS embeddings to the OOD accumulator
                if _save_ood_ref_enabled and _ood_cap_buf:
                    _ood_emb_acc.update(_ood_cap_buf.pop().numpy())

                # Skip batch only when loss itself is NaN/Inf (backward is meaningless)
                loss_val = loss.item()
                if loss_val != loss_val or loss_val == float("inf"):
                    print(f"[TaggerTrainer] WARNING: NaN/Inf loss at step {global_step}, skipping batch")
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.update()
                    continue

                # Capture sigmoid probs for training F1 rolling buffer (fp16, CPU).
                # Also needed when train-count augmentation is on (tag_count update).
                if _n2_eval > 0 or _train_count_on:
                    _step_probs  = torch.sigmoid(logits.detach()).to(torch.float16).cpu()
                    _step_labels = labels.detach().bool().cpu()

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    # Clip first — large-but-finite gradients are clipped and learned from.
                    # clip_grad_norm_ returns the pre-clip total norm; if it is non-finite
                    # (fp16 overflow produced Inf gradients), skip to avoid 0×Inf = NaN.
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if not torch.isfinite(grad_norm):
                        print(f"[TaggerTrainer] WARNING: non-finite grad norm ({grad_norm:.3g}) "
                              f"at step {global_step}, skipping optimizer step")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if not torch.isfinite(grad_norm):
                        print(f"[TaggerTrainer] WARNING: non-finite grad norm ({grad_norm:.3g}) "
                              f"at step {global_step}, skipping optimizer step")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    optimizer.step()

                # Injection (Danbooru pure-batch) updates: skip LR scheduler
                # and step counter so resume reproducibility is preserved.
                if not is_injection_batch:
                    scheduler.step()
                    global_step      += 1
                epoch_loss       += loss_val
                batches_processed += 1

                # Append to training F1 rolling buffer (F1 only); accumulate
                # tag_count whenever F1 metrics OR train-count augmentation is on.
                if _n2_eval > 0:
                    _train_f1_buffer.append((_step_probs, _step_labels))
                if _n2_eval > 0 or _train_count_on:
                    _tag_metrics_acc.update(_step_probs.float(), _step_labels.float())

                # ----- GPU-coordinator pause check (batch boundary) ----------
                # Generation requests set ``pause_event`` and stash an offload
                # ``OffloadDecision`` in ``pending_decision``.  We perform the
                # state movement here (in the trainer thread) so cross-thread
                # ``.to()`` calls never happen.  Stop wins over pause.
                if self._pause_event.is_set():
                    decision = self._coordinator_handle.pending_decision
                    if decision is not None:
                        self._coordinator_handle.offload(decision)
                    self._resumed_event.set()
                    # Block until either coordinator clears pause_event or a
                    # stop is requested.  Re-check at 0.5s cadence so a fast
                    # stop is responsive.
                    while self._pause_event.is_set() and not self._stop_requested:
                        self._stop_event.wait(timeout=0.5)
                    # Restore state regardless of why we exited the wait —
                    # either we'll continue training (need GPU state) or we'll
                    # hit the stop checkpoint below (need state to save it).
                    if decision is not None:
                        self._coordinator_handle.restore()
                    self._restored_event.set()
                    # Reset events for the next pause cycle
                    self._resumed_event.clear()
                    self._restored_event.clear()
                    self._coordinator_handle.pending_decision = None

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

                    # Snapshot Danbooru metrics to JSON (for the frontend panel)
                    _db_buf = getattr(train_loader, "_buffer", None)
                    if _db_buf is not None and hasattr(_db_buf, "get_metrics"):
                        try:
                            _m = _db_buf.get_metrics()
                            _mp = os.path.join(self.output_dir, "danbooru_metrics.json")
                            _tmp = _mp + ".tmp"
                            with open(_tmp, "w", encoding="utf-8") as _mf:
                                json.dump(_m, _mf, ensure_ascii=False)
                            os.replace(_tmp, _mp)
                        except Exception:
                            pass
                        # Persist the dynamic new-tag query list so it survives a
                        # resume (the surveyor can't rediscover already-in-vocab
                        # tags — see initial_dynamic_tags).
                        if hasattr(_db_buf, "snapshot_dynamic_tags"):
                            try:
                                _dyn = _db_buf.snapshot_dynamic_tags()
                                _dp = os.path.join(self.output_dir, "danbooru_dynamic_tags.json")
                                _dtmp = _dp + ".tmp"
                                with open(_dtmp, "w", encoding="utf-8") as _df:
                                    json.dump(_dyn, _df, ensure_ascii=False)
                                os.replace(_dtmp, _dp)
                            except Exception:
                                pass
                        # Persist the cooc active-collection list (same rationale).
                        if hasattr(_db_buf, "snapshot_cooc_active_tags"):
                            try:
                                _ct = _db_buf.snapshot_cooc_active_tags()
                                _cp = os.path.join(self.output_dir, "danbooru_cooc_active_tags.json")
                                _ctmp = _cp + ".tmp"
                                with open(_ctmp, "w", encoding="utf-8") as _cf:
                                    json.dump(_ct, _cf, ensure_ascii=False)
                                os.replace(_ctmp, _cp)
                            except Exception:
                                pass
                        # Persist the resolved Query collection pool (per-tag
                        # collection of query-resolved tags continues across resume).
                        if hasattr(_db_buf, "snapshot_query_tags"):
                            try:
                                _qt = _db_buf.snapshot_query_tags()
                                _qp = os.path.join(self.output_dir, "danbooru_query_tags.json")
                                _qtmp = _qp + ".tmp"
                                with open(_qtmp, "w", encoding="utf-8") as _qf:
                                    json.dump(_qt, _qf, ensure_ascii=False)
                                os.replace(_qtmp, _qp)
                            except Exception:
                                pass
                        # Persist per-epoch collection progress (collect_count +
                        # exhausted_tags) tagged with the current epoch, so a
                        # mid-epoch resume INTO THE SAME epoch continues collection
                        # rather than re-collecting already-collected tags.
                        if hasattr(_db_buf, "snapshot_epoch_progress"):
                            try:
                                _ep = _db_buf.snapshot_epoch_progress()
                                _ep["epoch"] = epoch
                                _epp = os.path.join(self.output_dir, "danbooru_epoch_progress.json")
                                _eptmp = _epp + ".tmp"
                                with open(_eptmp, "w", encoding="utf-8") as _epf:
                                    json.dump(_ep, _epf, ensure_ascii=False)
                                os.replace(_eptmp, _epp)
                            except Exception:
                                pass

                # Step-based checkpoint (model + state + optimizer + vocab)
                if save_every_n_steps > 0 and global_step % save_every_n_steps == 0:
                    ckpt_name = f"step_{global_step:06d}"
                    metadata  = self._make_metadata(epoch, global_step, best_f1, best_threshold)
                    ckpt_path = _save_model_checkpoint(model, self.output_dir, ckpt_name, metadata, checkpoint_save_mode)
                    _save_training_state(
                        self.output_dir, ckpt_name,
                        epoch, global_step, batch_idx,
                        best_f1, best_threshold,
                        epoch_start_rng=epoch_start_rng,  # epoch-start RNG for exact replay
                        dataset_fingerprint=current_fingerprint,
                    )
                    _save_optimizer_state(optimizer, self.output_dir, ckpt_name)
                    _save_vocabulary_snapshot(self.vocabulary, self.output_dir, ckpt_name)
                    _save_tag_metrics(_tag_metrics_acc, self.output_dir, ckpt_name,
                                      self.vocabulary, epoch_boundary=False,
                                      save_enabled=_save_tag_metrics_enabled,
                                      hard_lo=_hard_lo, hard_hi=_hard_hi,
                                      calib_method=_calib_method,
                                      calib_eps=_calib_eps,
                                      calib_prior_strength=_calib_prior_strength)
                    _save_ood_reference(_ood_emb_acc, self.output_dir, ckpt_name,
                                        save_enabled=_save_ood_ref_enabled)
                    if keep_last_n_checkpoints > 0:
                        _prune_step_checkpoints(self.output_dir, keep_last_n_checkpoints)
                    self._emit("checkpoint", {
                        "name": ckpt_name,
                        "step": global_step,
                        "epoch": epoch,
                        "path": ckpt_path,
                    })

                # Training F1 periodic evaluation (N2 steps) and threshold search (N1 steps)
                if _n2_eval > 0 and global_step > 0 and global_step % _n2_eval == 0 \
                        and len(_train_f1_buffer) > 0:
                    _buf_p = torch.cat([b[0] for b in _train_f1_buffer]).float()
                    _buf_l = torch.cat([b[1] for b in _train_f1_buffer]).float()
                    _threshold_updated = False
                    if _n1_search > 0 and global_step % _n1_search == 0:
                        _f1_threshold, _train_f1_val = _find_best_threshold(_buf_p, _buf_l)
                        _threshold_updated = True
                    # Single-pass computation of F1 + precision + recall at the current threshold
                    _buf_metrics = _compute_all_metrics(_buf_p, _buf_l, threshold=_f1_threshold)
                    if not _threshold_updated:
                        _train_f1_val = _buf_metrics["f1"]
                    # Scatter data: compute at threshold-search intervals only
                    _scatter_data: Optional[Dict] = None
                    if _n1_search > 0 and global_step % _n1_search == 0:
                        _scatter_data = _tag_metrics_acc.compute_scatter_for_vis(min_npos=20)

                        # Refresh low-F1 deficiency targets for Danbooru
                        # augmentation (synced to the threshold-search cadence).
                        # Existing vocab tags with a valid F1 below the threshold
                        # are pushed to the sampler's deficiency provider so it
                        # collects extra samples for them.
                        _db_buf2 = getattr(train_loader, "_buffer", None)
                        _provider = getattr(_db_buf2, "_deficiency_provider", None)
                        if _provider is not None:
                            try:
                                _idxs = _tag_metrics_acc.deficient_tag_indices(
                                    f1_threshold=float(cfg.get("danbooru_low_f1_threshold", 0.5)),
                                    top_k=int(cfg.get("danbooru_low_f1_top_k", 500)),
                                )
                                _names = [
                                    self.vocabulary.idx_to_tag.get(i, "") for i in _idxs
                                ]
                                _provider.set_targets([n for n in _names if n])
                            except Exception as _de:
                                print(f"[TaggerTrainer] Deficiency target refresh error: {_de}")
                    self._emit("train_f1", {
                        "step": global_step,
                        "train_f1": _train_f1_val,
                        "train_precision": _buf_metrics["precision"],
                        "train_recall": _buf_metrics["recall"],
                        "threshold": _f1_threshold,
                        "threshold_updated": _threshold_updated,
                        "fp_fn_scatter": _scatter_data,
                    })
                    del _buf_p, _buf_l

            # --- Stop checkpoint (mid-epoch or epoch-boundary) ---
            if self._stop_requested:
                # Release the DataLoader iterator early so worker processes
                # (num_workers > 0) are terminated before the checkpoint is saved.
                try:
                    loader_iter.close()  # generator close
                except Exception:
                    pass
                try:
                    del loader_iter
                except Exception:
                    pass
                print(f"[TaggerTrainer] Stop requested. Saving checkpoint at step {global_step}...")
                metadata = self._make_metadata(epoch, global_step, best_f1, best_threshold)
                ckpt_name = f"step_{global_step:06d}"
                ckpt_path = _save_model_checkpoint(model, self.output_dir, ckpt_name, metadata, checkpoint_save_mode)
                _save_training_state(
                    self.output_dir, ckpt_name,
                    epoch, global_step, batch_idx,
                    best_f1, best_threshold,
                    epoch_start_rng=epoch_start_rng,
                    dataset_fingerprint=current_fingerprint,
                )
                _save_optimizer_state(optimizer, self.output_dir, ckpt_name)
                _save_vocabulary_snapshot(self.vocabulary, self.output_dir, ckpt_name)
                _save_tag_metrics(_tag_metrics_acc, self.output_dir, ckpt_name,
                                  self.vocabulary, epoch_boundary=False,
                                  save_enabled=_save_tag_metrics_enabled,
                                  hard_lo=_hard_lo, hard_hi=_hard_hi,
                                  calib_method=_calib_method,
                                  calib_eps=_calib_eps,
                                  calib_prior_strength=_calib_prior_strength)
                _save_ood_reference(_ood_emb_acc, self.output_dir, ckpt_name,
                                    save_enabled=_save_ood_ref_enabled)
                # Also update "latest" to the stop position
                _save_model_checkpoint(model, self.output_dir, "latest", metadata, checkpoint_save_mode)
                _save_training_state(
                    self.output_dir, "latest",
                    epoch, global_step, batch_idx,
                    best_f1, best_threshold,
                    epoch_start_rng=epoch_start_rng,
                    dataset_fingerprint=current_fingerprint,
                )
                _save_optimizer_state(optimizer, self.output_dir, "latest")
                _save_vocabulary_snapshot(self.vocabulary, self.output_dir, "latest")
                _save_tag_metrics(_tag_metrics_acc, self.output_dir, "latest",
                                  self.vocabulary, epoch_boundary=False,
                                  save_enabled=_save_tag_metrics_enabled,
                                  hard_lo=_hard_lo, hard_hi=_hard_hi,
                                  calib_method=_calib_method,
                                  calib_eps=_calib_eps,
                                  calib_prior_strength=_calib_prior_strength)
                _save_ood_reference(_ood_emb_acc, self.output_dir, "latest",
                                    save_enabled=_save_ood_ref_enabled)
                self._emit("checkpoint", {
                    "name": ckpt_name,
                    "step": global_step,
                    "epoch": epoch,
                    "path": ckpt_path,
                })
                print(f"[TaggerTrainer] Stopped. Checkpoint: {ckpt_name} (epoch {epoch}, step {global_step})")
                break  # exit epoch loop → skips validation, epoch-end save, _final_threshold_search

            # Release the prefetch iterator and last-batch GPU tensors before
            # validation so we don't hold two epochs' worth of VRAM simultaneously.
            try:
                loader_iter.close()
            except Exception:
                pass
            try:
                del loader_iter
            except Exception:
                pass
            try:
                del pv, pam, ss, labels, loss_masks
            except Exception:
                pass
            try:
                del logits, loss
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            avg_loss = epoch_loss / max(batches_processed, 1)

            # Validation
            # val_max_batches caps memory usage: num_tags=84k×batch×float16 can be huge.
            # Default 256 batches ≈ 4096 samples ≈ 689 MB for 84k-tag vocab.
            val_max_batches = int(cfg.get("val_max_batches", 256)) or None
            val_metrics: Dict[str, Any] = {}
            # max(1, ...): 0 would divide by zero here, and "never validate" is
            # already spelled val_split=0 (no val_loader), so 0 has no meaning.
            validate_every = max(1, int(cfg.get("validate_every", 1) or 1))
            if val_loader and epoch % validate_every == 0:
                val_metrics = self._validate(model, val_loader, device, amp_dtype if use_amp else None,
                                             max_batches=val_max_batches)
                epoch_f1  = val_metrics.get("f1", 0.0)
                epoch_thr = val_metrics.get("threshold", 0.5)
                # Sync training-buffer threshold with the validation-optimal value so
                # subsequent buffer F1 evaluations use a calibrated threshold rather
                # than converging to a training-distribution-biased value (~0.42).
                _f1_threshold = epoch_thr

                if epoch_f1 > best_f1:
                    best_f1        = epoch_f1
                    best_threshold = epoch_thr
                    metadata = self._make_metadata(epoch, global_step, best_f1, best_threshold)
                    _save_model_checkpoint(model, self.output_dir, "best_f1", metadata, checkpoint_save_mode)
                    _save_vocabulary_snapshot(self.vocabulary, self.output_dir, "best_f1")
                    _save_tag_metrics(_tag_metrics_acc, self.output_dir, "best_f1",
                                      self.vocabulary, epoch_boundary=True,
                                      save_enabled=_save_tag_metrics_enabled,
                                      hard_lo=_hard_lo, hard_hi=_hard_hi,
                                      calib_method=_calib_method,
                                      calib_eps=_calib_eps,
                                      calib_prior_strength=_calib_prior_strength)
                    _save_ood_reference(_ood_emb_acc, self.output_dir, "best_f1",
                                        save_enabled=_save_ood_ref_enabled)
                    self._emit("checkpoint", {"name": "best_f1", "f1": best_f1, "epoch": epoch})

            # Save latest checkpoint at epoch boundary.
            # epoch+1 / batch_idx=-1 means "start of next epoch, no batches to skip".
            # epoch_start_rng is not needed for epoch-boundary resumes (batch_idx=-1
            # means we start the next epoch fresh and capture a new epoch_start_rng
            # at that time), so we omit it to keep the file compact.
            metadata = self._make_metadata(epoch, global_step, best_f1, best_threshold)
            _save_model_checkpoint(model, self.output_dir, "latest", metadata, checkpoint_save_mode)
            _save_vocabulary_snapshot(self.vocabulary, self.output_dir, "latest")
            _save_tag_metrics(_tag_metrics_acc, self.output_dir, "latest",
                              self.vocabulary, epoch_boundary=True,
                              save_enabled=_save_tag_metrics_enabled,
                              hard_lo=_hard_lo, hard_hi=_hard_hi,
                              calib_method=_calib_method,
                              calib_eps=_calib_eps,
                              calib_prior_strength=_calib_prior_strength)
            _save_ood_reference(_ood_emb_acc, self.output_dir, "latest",
                                save_enabled=_save_ood_ref_enabled)

            # Epoch-based checkpoint (model only; training state = same as latest)
            if save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0:
                ckpt_name = f"epoch_{epoch:04d}"
                _save_model_checkpoint(model, self.output_dir, ckpt_name, metadata, checkpoint_save_mode)
                _save_vocabulary_snapshot(self.vocabulary, self.output_dir, ckpt_name)
                _save_tag_metrics(_tag_metrics_acc, self.output_dir, ckpt_name,
                                  self.vocabulary, epoch_boundary=True,
                                  save_enabled=_save_tag_metrics_enabled,
                                  hard_lo=_hard_lo, hard_hi=_hard_hi,
                                  calib_method=_calib_method,
                                  calib_eps=_calib_eps,
                                  calib_prior_strength=_calib_prior_strength)
                _save_ood_reference(_ood_emb_acc, self.output_dir, ckpt_name,
                                    save_enabled=_save_ood_ref_enabled)
                self._emit("checkpoint", {"name": ckpt_name, "epoch": epoch, "step": global_step})
            _save_training_state(
                self.output_dir, "latest",
                epoch + 1, global_step, -1,
                best_f1, best_threshold,
                dataset_fingerprint=current_fingerprint,
            )
            _save_optimizer_state(optimizer, self.output_dir, "latest")

            # Rotate epoch histograms: prev ← cur, cur ← zero. This also finalizes
            # the per-epoch exposure delta used by train-count deficiency.
            _tag_metrics_acc.rotate_epoch()

            # Refresh train-count deficiency targets (under-exposed tags) for the
            # Danbooru sampler. Deficit is epoch-granular, so refresh once per
            # epoch after rotation. Empty until >= 2 epochs have completed.
            if _train_count_on:
                _tc_buf = getattr(train_loader, "_buffer", None)
                _tc_provider = getattr(_tc_buf, "_train_count_provider", None)
                if _tc_provider is not None:
                    try:
                        _tc_idxs = _tag_metrics_acc.deficient_train_count_indices(
                            top_k=int(cfg.get("danbooru_train_count_top_k", 500)),
                            min_deficit_ratio=float(cfg.get("danbooru_train_count_min_deficit_ratio", 0.3)),
                            min_per_epoch=int(cfg.get("danbooru_train_count_min_per_epoch", 10)),
                        )
                        _tc_names = [self.vocabulary.idx_to_tag.get(i, "") for i in _tc_idxs]
                        _tc_provider.set_targets([n for n in _tc_names if n])
                        if _tc_names:
                            print(f"[TaggerTrainer] Train-count deficiency: {len(_tc_names)} "
                                  f"under-exposed tag(s) targeted (epoch {epoch + 1})")
                    except Exception as _tce:
                        print(f"[TaggerTrainer] Train-count target refresh error: {_tce}")

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
                "step": global_step,
                "loss": avg_loss,
                **val_metrics,
            })

        # Remove OOD hook early (validation does not need it)
        try:
            _ood_hook_handle.remove()
        except Exception:
            pass

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

        # Unregister from GPU coordinator and release handle references.
        # (Defensive duplicate cleanup also runs in run_tagger_training().)
        try:
            from core.gpu_coordinator import gpu_coordinator
            gpu_coordinator.unregister_trainer(self._coordinator_handle)
            self._coordinator_handle.detach()
        except Exception as _e:
            print(f"[TaggerTrainer] coordinator cleanup at end of train(): {_e}")

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

                # Vocab may have expanded mid-training: the val loader's workers
                # hold a stale vocabulary snapshot and emit old-width labels, so
                # pad them to the current logit width before comparing.
                if labels.shape[1] < logits.shape[1]:
                    labels = torch.nn.functional.pad(
                        labels, (0, logits.shape[1] - labels.shape[1]), value=0.0
                    )

                # float16 to halve memory usage (84k tags × many samples)
                probs = torch.sigmoid(logits).to(torch.float16).cpu()
                all_preds.append(probs)
                all_labels.append(labels.to(torch.float16))

        all_preds  = torch.cat(all_preds,  dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        threshold, f1 = _find_best_threshold(all_preds, all_labels)
        m = _compute_all_metrics(all_preds, all_labels, threshold=threshold)
        return {"f1": f1, "threshold": threshold,
                "precision": m["precision"], "recall": m["recall"]}

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
                # Pad stale-width val labels after a mid-training vocab expansion.
                if labels.shape[1] < logits.shape[1]:
                    labels = torch.nn.functional.pad(
                        labels, (0, logits.shape[1] - labels.shape[1]), value=0.0
                    )
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
            "is_naflex": self.config.get("is_naflex", True),
            # Custom pooling dimensions — required for load_checkpoint to reconstruct
            # the correct model architecture.  Omitted when None (default pooling).
            **({
                "cls_dim": self.config["cls_dim"],
            } if self.config.get("cls_dim") is not None else {}),
            **({
                "hidden_proj_dim": self.config["hidden_proj_dim"],
            } if self.config.get("hidden_proj_dim") is not None else {}),
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

        def _vocab_progress(done: int, total: int, message: str) -> None:
            # Forward to the route callback → WS/SSE progress bar (send_progress_sync).
            if progress_callback:
                progress_callback(run_id, "dataset_progress", {
                    "step": done, "total": total, "message": message,
                })

        # Build the comma-tag resolver once and share it between the vocabulary
        # builder and the dataset so labels match vocab indices. Comma-containing
        # tags (mostly Gelbooru titles) get re-merged into a single comma-free
        # canonical tag instead of breaking into Unknown fragments.
        use_gel = bool(config.get("vocab_use_gelbooru_categories", True))
        from core.tagger.comma_tag_resolver import CommaTagResolver
        try:
            from config import settings as _settings
            _root_dir = _settings.root_dir
        except Exception:
            _root_dir = os.path.join(output_dir, "..", "..")
        comma_resolver = CommaTagResolver.build_from_taglist_cache(_root_dir, use_gelbooru=use_gel)
        print(f"[TaggerTraining] Comma-tag resolver: {len(comma_resolver)} comma-containing tags")

        vocabulary = TagVocabulary.build_from_dataset_ids(
            dataset_ids=dataset_ids,
            datasets_db=datasets_db,
            min_count=config.get("vocab_min_count", 1),
            excluded_categories=excl_cats,
            ban_tags=ban_tags,
            alias_resolver=alias_resolver,
            use_gelbooru_categories=use_gel,
            comma_resolver=comma_resolver,
            progress_callback=_vocab_progress,
        )
        print(f"[TaggerTraining] Vocabulary: {vocabulary.num_tags} tags")
        if excl_cats:
            print(f"[TaggerTraining] Excluded categories: {excl_cats}")
        if ban_tags:
            print(f"[TaggerTraining] Banned tags: {len(ban_tags)} entries")
        progress_callback and progress_callback(run_id, "vocab", {"num_tags": vocabulary.num_tags})

        from core.tagger.siglip2_tagger_model import _is_hf_repo_or_url, SIGLIP2_DEFAULT_REPO_ID as _DEFAULT_REPO

        # Resume fallback: if we're trying to resume but the original local
        # vision_encoder_path is gone (file moved / deleted across runs), use
        # the self-contained ``base_model.safetensors`` snapshot in
        # ``output_dir``.  Only triggers when:
        #   - resume is requested
        #   - output_dir/base_model.safetensors exists
        #   - the configured vision_encoder_path is a local file that is
        #     currently missing (HF repos are not handled here — they have
        #     their own HF cache fallback)
        if resume_from_checkpoint and output_dir:
            _local_base = os.path.join(output_dir, "base_model.safetensors")
            _orig_ve   = config.get("vision_encoder_path", "")
            _orig_ve_s = _orig_ve.strip().strip('"').strip("'") if _orig_ve else ""
            _orig_is_hf, _ = _is_hf_repo_or_url(_orig_ve_s)
            if (
                not _orig_is_hf
                and os.path.isfile(_local_base)
                and (not _orig_ve_s or not os.path.isfile(_orig_ve_s))
            ):
                print(
                    f"[TaggerTraining] WARNING: original vision_encoder_path "
                    f"'{_orig_ve_s or '<empty>'}' not accessible; falling back "
                    f"to local '{_local_base}' for self-contained resume"
                )
                config = dict(config)
                config["vision_encoder_path"] = _local_base

        # Resolve vision_encoder_repo FIRST — processor must match the vision encoder.
        # Priority (highest to lowest):
        #   1. Explicit HF repo ID / URL in vision_encoder_path
        #   2. vision_encoder_repo already set in config (caller pre-filled)
        #   3. _metadata.json alongside a local safetensors vision_encoder_path
        #   4. Default (patch16-naflex)
        _ve_path = config.get("vision_encoder_path", "")
        _is_hf_ve, _resolved_ve = _is_hf_repo_or_url(_ve_path)
        if _is_hf_ve or "vision_encoder_repo" not in config:
            config = dict(config)  # shallow copy — do not mutate caller's dict
            if _is_hf_ve:
                config["vision_encoder_repo"] = _resolved_ve
                print(f"[TaggerTraining] HF repo detected for vision encoder: {_resolved_ve}")
            else:
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

        # Build processor — must match the vision encoder architecture.
        # Only NaFlex-compatible models (those whose processor returns
        # pixel_attention_mask and spatial_shapes) are supported.
        processor_repo = config["vision_encoder_repo"]
        print(f"[TaggerTraining] === Phase: processor ===")
        print(f"[TaggerTraining] Loading processor from {processor_repo}...")
        try:
            processor = AutoProcessor.from_pretrained(processor_repo, local_files_only=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(processor_repo)
        # Probe processor to detect architecture (NaFlex vs standard fixed-resolution).
        _probe = processor(images=[_PILImage.new("RGB", (64, 64))], return_tensors="pt")
        _is_naflex = "pixel_attention_mask" in _probe and "spatial_shapes" in _probe
        _mode_str = "NaFlex (variable resolution)" if _is_naflex else "standard (fixed resolution)"
        print(f"[TaggerTraining] Processor loaded (repo: {processor_repo}, mode: {_mode_str})")
        # Store in config so build_tagger_model() and _make_metadata() can access it.
        # Ensure config is a mutable dict copy (the if-branch above may not have copied it).
        if not isinstance(config, dict):
            config = dict(config)
        config["is_naflex"] = _is_naflex

        # Build datasets
        print(f"[TaggerTraining] === Phase: dataset ===")
        print(f"[TaggerTraining] Building TaggerDataset ({vocabulary.num_tags} tags)...")
        progress_callback and progress_callback(run_id, "phase", {
            "phase": "dataset", "message": f"Loading dataset ({vocabulary.num_tags} tags)..."
        })

        def _ds_progress(done: int, total: int, message: str) -> None:
            # Forwarded to the route callback as a "dataset_progress" event,
            # which pushes it to the WS/SSE progress bar (send_progress_sync).
            if progress_callback:
                progress_callback(run_id, "dataset_progress", {
                    "step": done, "total": total, "message": message,
                })

        full_ds = TaggerDataset(
            dataset_ids=dataset_ids,
            vocabulary=vocabulary,
            datasets_db=datasets_db,
            processor=processor,
            alias_resolver=alias_resolver,
            comma_resolver=comma_resolver,
            quality_masking_mode=config.get("quality_masking_mode", "intra_group"),
            progress_callback=_ds_progress,
        )
        total_samples = len(full_ds)
        print(f"[TaggerTraining] Dataset: {total_samples} samples total")

        val_split_mode = config.get("val_split_mode", "percent")
        if val_split_mode == "fixed":
            val_size = max(1, int(config.get("val_fixed_size", 500)))
            val_size = min(val_size, total_samples - 1)  # keep at least 1 train sample
        else:
            val_split = float(config.get("val_split", 0.05))
            val_size = max(1, int(total_samples * val_split))
        train_size = total_samples - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"[TaggerTraining] Split: {train_size} train / {val_size} val "
              f"(mode={val_split_mode})")

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

        # ------------------------------------------------------------------
        # Online Danbooru augmentation (optional)
        # ------------------------------------------------------------------
        _danbooru_buffer = None
        _tag_refresh_detector = None
        _expander = None
        _surveyor = None
        _deficiency_provider = None
        _train_count_provider = None
        if config.get("enable_danbooru_augmentation", False):
            _tag_queries = [
                t.strip()
                for t in (config.get("danbooru_tags") or "").splitlines()
                if t.strip()
            ]
            # Query is now a first-class, independently toggleable mode: static
            # queries only run when danbooru_query_enable is on (default True for
            # backward compat). When danbooru_query_expand_enable is also on, the
            # queries' tag tokens / wildcards are resolved to concrete tags and
            # collected per-tag (+ vocab expansion).
            _query_on = bool(config.get("danbooru_query_enable", True))
            _query_expand_on = bool(config.get("danbooru_query_expand_enable", False))
            _active_queries = _tag_queries if _query_on else []
            # Augmentation runs when there are active queries OR vocab expansion
            # OR low-F1 deficiency collection is enabled (the surveyor / trainer
            # then drive dynamic queries on their own, so static queries are
            # optional in those modes).
            _vocab_expand_on = config.get("danbooru_vocab_expand", False)
            _low_f1_on = config.get("danbooru_low_f1_enable", False)
            _train_count_on = bool(config.get("danbooru_train_count_enable", False))
            if _active_queries or _vocab_expand_on or _low_f1_on or _train_count_on:
                from .danbooru_sampler import DanbooruSampleBuffer, MixedDataLoader as _MixedDL
                from .danbooru_vocab_expander import VocabExpander, expand_vocab_and_head

                _expander = VocabExpander()

                if _train_count_on:
                    from .danbooru_deficiency_provider import DanbooruDeficiencyProvider
                    _train_count_provider = DanbooruDeficiencyProvider()

                if _low_f1_on:
                    from .danbooru_deficiency_provider import DanbooruDeficiencyProvider
                    _deficiency_provider = DanbooruDeficiencyProvider()
                    # Low-F1 targets are recomputed inside the train-F1
                    # threshold-search block; if that cadence is disabled the
                    # provider is never fed and the low-F1 path stays idle.
                    _lf1_n1 = int(config.get("train_f1_threshold_search_every_n_steps", 0) or 0)
                    _lf1_n2 = int(config.get("train_f1_eval_every_n_steps", 0) or 0)
                    if _lf1_n1 <= 0 or _lf1_n2 <= 0:
                        print(
                            "[TaggerTraining] WARNING: danbooru_low_f1_enable=True but training-F1 "
                            "metrics are disabled (train_f1_eval_every_n_steps and "
                            "train_f1_threshold_search_every_n_steps must both be > 0). "
                            "Low-F1 deficiency targets will never be computed — the low-F1 "
                            "collection path will stay idle."
                        )

                if config.get("danbooru_vocab_expand", False):
                    from .danbooru_tag_surveyor import DanbooruTagSurveyor
                    _surveyor = DanbooruTagSurveyor(
                        vocabulary=vocabulary,
                        categories=config.get("danbooru_new_tag_categories", [0, 3, 4]),
                        min_count=config.get("danbooru_new_tag_min_count", 200),
                        min_count_by_cat=config.get("danbooru_new_tag_min_count_by_cat", {}),
                        lookback_days=config.get("danbooru_new_tag_lookback_days", 90),
                        survey_interval=float(config.get("danbooru_new_tag_survey_interval", 3600)),
                        api_interval=config.get("danbooru_api_interval", 1.4),
                        dl_speed_kbps=config.get("danbooru_dl_speed_kbps", 500),
                    )
                    _surveyor.start()

                # Cumulative set of augmentation-expanded tags (surveyor + cooc),
                # persisted so they can be re-added to the vocab on resume — the
                # vocab is rebuilt from the dataset on resume, so without this
                # their learned head rows would be dropped every time. Loaded
                # here so the set accumulates across resumes.
                _expanded_tags_path = os.path.join(output_dir, "danbooru_expanded_tags.json")
                _expanded_accum: set = set()
                try:
                    if os.path.isfile(_expanded_tags_path):
                        with open(_expanded_tags_path, "r", encoding="utf-8") as _ef:
                            _el = json.load(_ef)
                        _expanded_accum = set(_el if isinstance(_el, list) else _el.keys())
                except Exception:
                    _expanded_accum = set()

                def _save_expanded_tags() -> None:
                    try:
                        _tmp = _expanded_tags_path + ".tmp"
                        with open(_tmp, "w", encoding="utf-8") as _wf:
                            json.dump(sorted(_expanded_accum), _wf, ensure_ascii=False)
                        os.replace(_tmp, _expanded_tags_path)
                    except Exception as _se:
                        print(f"[TaggerTraining] expanded-tags save error: {_se}")

                def _expansion_callback(new_tags: List[str]) -> None:
                    # model/optimizer live in trainer.train()'s scope; reach them
                    # via the trainer instance (assigned below, before train()
                    # runs, so the closure resolves them at call time).
                    # Record which tags are genuinely NEW (not already present)
                    # so the cross-resume preservation set stays precise — tags
                    # that were already in the vocab (e.g. dataset tags re-proposed
                    # by co-occurrence) must NOT enter it, or deprecated dataset
                    # tags would wrongly survive resume.
                    _norms = [normalize_tag(t) for t in new_tags]
                    _were_present = {t for t in _norms if t in vocabulary.tag_to_idx}
                    n = expand_vocab_and_head(
                        new_tags, vocabulary, trainer.model, trainer.optimizer
                    )
                    if n > 0:
                        _newly = [
                            t for t in _norms
                            if t not in _were_present and t in vocabulary.tag_to_idx
                        ]
                        if _newly:
                            _expanded_accum.update(_newly)
                            _save_expanded_tags()
                        _new_size = vocabulary.num_tags
                        # Grow the other in-train per-tag structures so they stay
                        # aligned with the expanded head/labels (otherwise the
                        # next metrics update / loss hits a shape mismatch).
                        try:
                            _acc = getattr(trainer, "_tag_metrics_acc", None)
                            if _acc is not None:
                                _acc.grow(_new_size)
                        except Exception as _ge:
                            print(f"[TaggerTraining] metrics accumulator grow failed: {_ge}")
                        try:
                            _ng = _grow_criterion_buffers(trainer.criterion, _new_size)
                            if _ng:
                                print(f"[TaggerTraining] Grew {_ng} per-tag loss buffer(s) for new tags")
                        except Exception as _ce:
                            print(f"[TaggerTraining] criterion buffer grow failed: {_ce}")
                        # The train-F1 rolling buffer holds old-width probs/labels
                        # which can no longer be concatenated with new-width ones.
                        try:
                            _tb = getattr(trainer, "_train_f1_buffer", None)
                            if _tb is not None:
                                _tb.clear()
                        except Exception:
                            pass

                        if _surveyor is not None:
                            _surveyor.mark_added(new_tags)
                        # Save vocabulary snapshot so training can resume with expanded vocab
                        try:
                            vocab_path = os.path.join(output_dir, "vocabulary_latest.json")
                            with open(vocab_path, "w", encoding="utf-8") as _vf:
                                _vf.write(vocabulary.to_json())
                        except Exception as _ve:
                            print(f"[TaggerTraining] Vocab snapshot save error: {_ve}")
                        print(
                            f"[TaggerTraining] Vocab expanded: +{n} tag(s), "
                            f"total={vocabulary.num_tags}"
                        )

                # Resolve buffer_size: None → 2 * batch_size
                _base_B = int(config.get("batch_size", 8) or 8)
                _buf_cfg = config.get("danbooru_buffer_size")
                _buffer_size = int(_buf_cfg) if _buf_cfg else 2 * _base_B

                # Resolve injection batch size from ratio (default 1.0 × B)
                _inj_ratio = float(config.get("danbooru_injection_batch_size_ratio", 1.0) or 1.0)
                _inj_batch_size = max(1, int(round(_base_B * _inj_ratio)))
                _inj_interval = int(config.get("danbooru_injection_interval", 4) or 4)

                # Restore the dynamic (new-tag) query list so previously expanded
                # tags keep being collected after a resume — the surveyor won't
                # re-discover them (they are already in the vocab now).
                # {tag: last_used} (current) or [tag, ...] (legacy) — the buffer
                # accepts both. Preserves LRU recency across resume.
                _initial_dynamic_tags: Any = None
                try:
                    _dt_path = os.path.join(output_dir, "danbooru_dynamic_tags.json")
                    if os.path.isfile(_dt_path):
                        with open(_dt_path, "r", encoding="utf-8") as _df:
                            _loaded = json.load(_df)
                        if isinstance(_loaded, (dict, list)) and len(_loaded) > 0:
                            _initial_dynamic_tags = _loaded
                            print(f"[TaggerTraining] Restored {len(_loaded)} dynamic new-tag "
                                  f"queries for continued collection across resume")
                except Exception as _dte:
                    print(f"[TaggerTraining] Could not restore dynamic tag list: {_dte}")

                # Restore the cooc active-collection list so co-occurrence-promoted
                # tags keep being actively collected (order:random, quota-bounded)
                # across resume.
                _initial_cooc_active: Any = None
                try:
                    _ct_path = os.path.join(output_dir, "danbooru_cooc_active_tags.json")
                    if os.path.isfile(_ct_path):
                        with open(_ct_path, "r", encoding="utf-8") as _cf:
                            _cl = json.load(_cf)
                        # {tag: last_used} (current) or [tag, ...] (legacy) — the
                        # buffer accepts both; the dict form preserves LRU recency.
                        if isinstance(_cl, (dict, list)) and len(_cl) > 0:
                            _initial_cooc_active = _cl
                            print(f"[TaggerTraining] Restored {len(_cl)} cooc active-collection "
                                  f"queries for continued collection across resume")
                except Exception as _cte:
                    print(f"[TaggerTraining] Could not restore cooc active tag list: {_cte}")

                # Restore the resolved Query collection pool so per-tag collection
                # of query-resolved tags continues across resume without re-hitting
                # the tags API. {tag: last_used} (current) or [tag, ...] (legacy).
                _initial_query_tags: Any = None
                try:
                    _qt_path = os.path.join(output_dir, "danbooru_query_tags.json")
                    if os.path.isfile(_qt_path):
                        with open(_qt_path, "r", encoding="utf-8") as _qf:
                            _ql = json.load(_qf)
                        if isinstance(_ql, (dict, list)) and len(_ql) > 0:
                            _initial_query_tags = _ql
                            print(f"[TaggerTraining] Restored {len(_ql)} resolved query tag(s) "
                                  f"for continued per-tag collection across resume")
                except Exception as _qte:
                    print(f"[TaggerTraining] Could not restore query tag list: {_qte}")

                # Restore per-epoch collection progress for a mid-epoch resume.
                # Only loaded when actually resuming (avoids applying a stale file
                # on a fresh run that reuses an output_dir). The buffer applies it
                # only if the resumed epoch matches the persisted epoch, so a
                # cross-epoch resume still starts the new epoch's collection clean.
                _initial_epoch_progress: Any = None
                if resume_from_checkpoint:
                    try:
                        _epp_path = os.path.join(output_dir, "danbooru_epoch_progress.json")
                        if os.path.isfile(_epp_path):
                            with open(_epp_path, "r", encoding="utf-8") as _epf:
                                _epl = json.load(_epf)
                            if isinstance(_epl, dict) and _epl.get("epoch") is not None:
                                _initial_epoch_progress = _epl
                                print(f"[TaggerTraining] Loaded epoch {_epl.get('epoch')} collection "
                                      f"progress ({len(_epl.get('exhausted_tags') or [])} exhausted "
                                      f"tag(s)) for potential mid-epoch resume")
                    except Exception as _eppe:
                        print(f"[TaggerTraining] Could not restore epoch progress: {_eppe}")

                _danbooru_buffer = DanbooruSampleBuffer(
                    tag_queries=_active_queries,
                    vocabulary=vocabulary,
                    processor=processor,
                    is_naflex=_is_naflex,
                    quality_masking_mode=config.get("quality_masking_mode", "intra_group"),
                    alias_resolver=alias_resolver,
                    max_posts_per_query=config.get("danbooru_max_posts_per_query", 200),
                    min_score=config.get("danbooru_min_score", 0),
                    buffer_size=_buffer_size,
                    api_interval=config.get("danbooru_api_interval", 1.4),
                    dl_speed_kbps=config.get("danbooru_dl_speed_kbps", 500),
                    expander=_expander,
                    surveyor=_surveyor,
                    deficiency_provider=_deficiency_provider,
                    weight_static=config.get("danbooru_query_weight_static", 1.0),
                    weight_new_tag=config.get("danbooru_query_weight_new_tag", 1.0),
                    weight_low_f1=config.get("danbooru_query_weight_low_f1", 1.0),
                    low_f1_min_posts=config.get("danbooru_low_f1_min_posts", 50),
                    # Co-occurrence discovery only makes sense when vocab expansion
                    # is on (it feeds the same expander/head-growth path).
                    cooc_expand_enable=bool(config.get("danbooru_cooc_expand_enable", False)) and _vocab_expand_on,
                    cooc_min_count=config.get("danbooru_cooc_min_count", 50),
                    cooc_categories=config.get("danbooru_cooc_categories", [0, 3, 4]),
                    initial_dynamic_tags=_initial_dynamic_tags,
                    max_dynamic_tags=config.get("danbooru_max_dynamic_tags", 0),
                    # Cooc ACTIVE collection (only meaningful with cooc expansion on).
                    weight_cooc=(config.get("danbooru_query_weight_cooc", 0.1) if
                                 (bool(config.get("danbooru_cooc_expand_enable", False)) and _vocab_expand_on) else 0.0),
                    cooc_collect_per_epoch=config.get("danbooru_cooc_collect_per_epoch", 50),
                    cooc_order_random=bool(config.get("danbooru_cooc_order_random", True)),
                    initial_cooc_active_tags=_initial_cooc_active,
                    # Query mode (per-tag collection + resolution-based expansion).
                    query_expand=_query_expand_on,
                    query_min_count=config.get("danbooru_query_new_tag_min_count", 200),
                    query_categories=config.get("danbooru_query_expand_categories", [0, 3, 4]),
                    query_top_k=config.get("danbooru_query_resolve_top_k", 50),
                    query_max_expanded=config.get("danbooru_query_max_expanded_tags", 0),
                    query_resolve_interval=float(config.get("danbooru_query_resolve_interval", 3600)),
                    initial_query_tags=_initial_query_tags,
                    # Per-tag per-epoch collection caps (0 = unlimited).
                    query_collect_per_epoch=config.get("danbooru_query_collect_per_epoch", 0),
                    new_tag_collect_per_epoch=config.get("danbooru_new_tag_collect_per_epoch", 0),
                    low_f1_collect_per_epoch=config.get("danbooru_low_f1_collect_per_epoch", 0),
                    # Train-count deficiency path (exposure balancing).
                    train_count_provider=_train_count_provider,
                    weight_train_count=(config.get("danbooru_query_weight_train_count", 1.0)
                                        if _train_count_on else 0.0),
                    train_count_min_posts=config.get("danbooru_train_count_min_posts", 50),
                    train_count_collect_per_epoch=config.get("danbooru_train_count_collect_per_epoch", 0),
                    # Score-based quality tag (label derived from post score).
                    quality_tag_enable=bool(config.get("danbooru_quality_tag_enable", False)),
                    quality_tag_thresholds=str(config.get("danbooru_quality_tag_thresholds", "") or ""),
                    quality_tag_attach_negative=bool(config.get("danbooru_quality_tag_attach_negative", False)),
                    # Mid-epoch resume: continue collection from where it stopped.
                    initial_epoch_progress=_initial_epoch_progress,
                    # Manual-resume control channel (danbooru_control.json here).
                    control_dir=output_dir,
                )
                # Configure the download-speed safety monitor (throttle/ban guard).
                from .download_speed_monitor import get_speed_monitor
                get_speed_monitor().configure(
                    enabled=bool(config.get("danbooru_speed_check_enable", True)),
                    degraded_kbps=int(config.get("danbooru_speed_degraded_kbps", 250)),
                    min_slow_streak=int(config.get("danbooru_speed_min_slow_streak", 8)),
                    min_slow_seconds=float(config.get("danbooru_speed_min_slow_seconds", 90)),
                    cooldown_seconds=float(config.get("danbooru_speed_cooldown_seconds", 3600)),
                )
                _danbooru_buffer.start()
                train_loader = _MixedDL(
                    train_loader,
                    buffer=_danbooru_buffer,
                    injection_interval=_inj_interval,
                    injection_batch_size=_inj_batch_size,
                    expander=_expander,
                    # Head-growth path is needed by surveyor vocab-expansion AND
                    # query-mode resolution-based expansion.
                    expansion_callback=_expansion_callback if (_vocab_expand_on or _query_expand_on) else None,
                    vocabulary=vocabulary,
                    quality_masking_mode=config.get("quality_masking_mode", "intra_group"),
                    alias_resolver=alias_resolver,
                )
                print(
                    f"[TaggerTraining] Danbooru augmentation: {len(_active_queries)} quer"
                    f"{'y' if len(_active_queries) == 1 else 'ies'} "
                    f"(query_mode={'on' if _query_on else 'off'}), "
                    f"interrupt-batch every {_inj_interval} steps "
                    f"(size={_inj_batch_size}), buffer={_buffer_size}, "
                    f"weights(query={config.get('danbooru_query_weight_static', 1.0)}, "
                    f"new_tag={config.get('danbooru_query_weight_new_tag', 1.0)}, "
                    f"low_f1={config.get('danbooru_query_weight_low_f1', 1.0)}, "
                    f"train_count={config.get('danbooru_query_weight_train_count', 1.0)})"
                    + (f", query_expand=on (min_count={config.get('danbooru_query_new_tag_min_count', 200)}, "
                       f"top_k={config.get('danbooru_query_resolve_top_k', 50)}, "
                       f"max={config.get('danbooru_query_max_expanded_tags', 0)}, "
                       f"categories={config.get('danbooru_query_expand_categories', [0, 3, 4])})"
                       if _query_expand_on else "")
                    + (f", vocab_expand=on (min_count={config.get('danbooru_new_tag_min_count', 200)})"
                       if _vocab_expand_on else "")
                    + (f", low_f1=on (threshold={config.get('danbooru_low_f1_threshold', 0.5)}, "
                       f"top_k={config.get('danbooru_low_f1_top_k', 500)}, "
                       f"min_posts={config.get('danbooru_low_f1_min_posts', 50)})"
                       if _low_f1_on else "")
                    + (f", cooc_expand=on (min_count={config.get('danbooru_cooc_min_count', 50)}, "
                       f"categories={config.get('danbooru_new_tag_categories', [0, 3, 4])})"
                       if (config.get('danbooru_cooc_expand_enable', False) and _vocab_expand_on) else "")
                    + (f", train_count=on (deficit_ratio>={config.get('danbooru_train_count_min_deficit_ratio', 0.3)}, "
                       f"top_k={config.get('danbooru_train_count_top_k', 500)}, "
                       f"min_per_epoch={config.get('danbooru_train_count_min_per_epoch', 10)})"
                       if _train_count_on else "")
                )
                if config.get("danbooru_cooc_expand_enable", False) and not _vocab_expand_on:
                    print("[TaggerTraining] WARNING: danbooru_cooc_expand_enable=True but "
                          "danbooru_vocab_expand=False — co-occurrence discovery needs vocab "
                          "expansion (the head-growth path) to be on; it will stay idle.")
            else:
                print("[TaggerTraining] enable_danbooru_augmentation=True but no tag queries "
                      "and vocab_expand is off — nothing to fetch, skipping")

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

        # Resolve the OLD vocabulary belonging to *resume_ckpt_name*.
        # Priority: per-checkpoint snapshot ``<name>_vocabulary.json`` > common
        # ``vocabulary.json`` (latter is overwritten on every run start, so it
        # may not match the checkpoint's tag→idx mapping).
        old_vocabulary: Optional[TagVocabulary] = None
        if resume_ckpt_name is not None:
            old_vocab_path = _resolve_checkpoint_vocab_path(output_dir, resume_ckpt_name)
            if old_vocab_path is not None:
                try:
                    with open(old_vocab_path, "r", encoding="utf-8") as _f:
                        old_vocabulary = TagVocabulary.from_dict(json.load(_f))
                    _src = (
                        "per-checkpoint snapshot"
                        if old_vocab_path.endswith(f"{resume_ckpt_name}_vocabulary.json")
                        else "common vocabulary.json (FALLBACK)"
                    )
                    print(f"[TaggerTraining] Loaded old vocabulary ({old_vocabulary.num_tags} tags) "
                          f"from {_src} for head alignment")
                except Exception as _e:
                    print(f"[TaggerTraining] WARNING: could not load old vocabulary: {_e}")
            else:
                print(f"[TaggerTraining] WARNING: no vocabulary file found for {resume_ckpt_name}; "
                      f"head alignment will be unable to detect mapping shifts (positional copy only)")

        # Preserve augmentation-expanded tags across resume.
        # build_from_dataset_ids() above rebuilds the vocab from the dataset
        # captions only, so Danbooru-augmentation-expanded tags (which are not in
        # any dataset caption) would be dropped — and their learned head rows lost
        # — on every resume. Re-add ONLY those tags, tracked in
        # danbooru_expanded_tags.json. Tags genuinely removed from the dataset
        # (deprecated / aliased away) are NOT in that set, so they are still
        # dropped, preserving the intended head-cleanup behaviour. _inherit_head
        # then copies the surviving head rows for the re-added tags by tag name.
        if resume_ckpt_name is not None and old_vocabulary is not None and output_dir:
            _exp_path = os.path.join(output_dir, "danbooru_expanded_tags.json")
            if os.path.isfile(_exp_path):
                try:
                    with open(_exp_path, "r", encoding="utf-8") as _ef:
                        _exp_loaded = json.load(_ef)
                    _exp_set = set(_exp_loaded if isinstance(_exp_loaded, list) else _exp_loaded.keys())
                    # Only re-add tags the checkpoint actually had (so a head row
                    # exists to inherit) AND that the rebuilt dataset vocab lacks.
                    _to_add = sorted(
                        t for t in _exp_set
                        if t in old_vocabulary.tag_to_idx and t not in vocabulary.tag_to_idx
                    )
                    if _to_add:
                        _by_cat: Dict[str, List[str]] = {}
                        for _t in _to_add:
                            _c = old_vocabulary.tag_to_category.get(_t, "General")
                            _by_cat.setdefault(_c, []).append(_t)
                        _added_total = 0
                        for _c in sorted(_by_cat):
                            _added_total += len(vocabulary.add_tags(_by_cat[_c], category=_c))
                        print(f"[TaggerTraining] Preserved {_added_total} augmentation-expanded "
                              f"tag(s) across resume (vocab now {vocabulary.num_tags}); "
                              f"deprecated dataset tags still dropped")
                    else:
                        print(f"[TaggerTraining] Expanded-tags file present ({len(_exp_set)} tags) "
                              f"but none needed re-adding (already in dataset vocab)")
                except Exception as _xe:
                    print(f"[TaggerTraining] WARNING: could not restore expanded tags: {_xe}")

        # Build the vocab lineage so renamed (alias) / merged (comma re-join) tags
        # inherit their predecessor's head row + optimizer momentum across a vocab
        # change, instead of being zero-initialized. Only meaningful when an old
        # vocabulary exists (resume / init_head_from). Built AFTER the expanded-tag
        # re-add above so new_tag_to_idx is final.
        vocab_lineage: Dict[str, List[str]] = {}
        if old_vocabulary is not None:
            from core.tagger.vocab_lineage import build_vocab_lineage
            vocab_lineage = build_vocab_lineage(
                old_tag_to_idx=old_vocabulary.tag_to_idx,
                new_tag_to_idx=vocabulary.tag_to_idx,
                alias_resolver=alias_resolver,
                comma_resolver=comma_resolver,
            )
            if vocab_lineage:
                print(f"[TaggerTraining] Vocab lineage: {len(vocab_lineage)} renamed/merged "
                      f"tag(s) will inherit head weights from predecessors")

        # Run trainer
        trainer = TaggerTrainer(
            run_id=run_id,
            config=config,
            vocabulary=vocabulary,
            output_dir=output_dir,
            progress_callback=progress_callback,
            old_vocabulary=old_vocabulary,
            vocab_lineage=vocab_lineage,
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

        # ------------------------------------------------------------------
        # Live tag-refresh (optional): pick up tag edits made in the UI during
        # training without slowing the iteration. Detection runs on a background
        # thread; workers apply overrides via a generation-gated mmap (see
        # core/tagger/tag_refresh.py). Must be wired before train() spawns the
        # first DataLoader workers so the file paths travel into the worker pickle.
        # ------------------------------------------------------------------
        if config.get("tag_refresh_enable", False):
            try:
                from core.tagger.tag_refresh import TagRefreshStore, TagRefreshDetector
                from database import datasets_db_path as _ds_db_path
                _tr_store = TagRefreshStore(output_dir)
                full_ds._refresh_gen_path     = _tr_store.gen_path
                full_ds._refresh_payload_path = _tr_store.payload_path
                full_ds._refresh_enabled      = True
                _tag_refresh_detector = TagRefreshDetector(
                    db_path=_ds_db_path,
                    dataset_ids=dataset_ids,
                    item_ids=full_ds._item_ids,
                    caption_types=None,  # matches the dataset build (all tags-format)
                    comma_resolver=comma_resolver,
                    alias_resolver=alias_resolver,
                    store=_tr_store,
                    interval=float(config.get("tag_refresh_interval_seconds", 60)),
                )
                _tag_refresh_detector.start()
            except Exception as _e:
                print(f"[TagRefresh] disabled (setup failed): {_e}")
                _tag_refresh_detector = None

        try:
            return trainer.train(
                train_loader, val_loader, processor,
                resume_state=resume_state,
                resume_ckpt_name=resume_ckpt_name,
            )
        finally:
            # Defensive: ensure the trainer is unregistered from the GPU
            # coordinator even if train() raised mid-loop.  Calls are
            # idempotent — duplicate with the cleanup inside train()
            # is harmless.
            try:
                from core.gpu_coordinator import gpu_coordinator
                gpu_coordinator.unregister_trainer(trainer._coordinator_handle)
                trainer._coordinator_handle.detach()
            except Exception as _e:
                print(f"[TaggerTraining] coordinator cleanup: {_e}")

    finally:
        import gc as _gc
        # Stop the live tag-refresh detector thread if it was started.
        try:
            if _tag_refresh_detector is not None:
                _tag_refresh_detector.stop()
        except NameError:
            pass
        except Exception as _e:
            print(f"[TaggerTraining] tag-refresh detector stop error: {_e}")
        # Stop the Danbooru background fetch thread if it was started.
        try:
            if _danbooru_buffer is not None:
                _danbooru_buffer.stop()
        except NameError:
            pass
        except Exception as _e:
            print(f"[TaggerTraining] Danbooru buffer stop error: {_e}")
        try:
            if _surveyor is not None:
                _surveyor.stop()
        except NameError:
            pass
        except Exception as _e:
            print(f"[TaggerTraining] Danbooru surveyor stop error: {_e}")
        # Explicitly delete DataLoaders to terminate worker processes before GC.
        # Without this, worker processes (num_workers > 0) outlive the finally block
        # because train_loader/val_loader locals still reference the iterators.
        try:
            del train_loader
        except NameError:
            pass
        try:
            del val_loader
        except NameError:
            pass
        _gc.collect()
        # Synchronize CUDA streams so all pending GPU ops finish before we claim
        # the GPU is free.  empty_cache() only releases cached (unused) memory;
        # without synchronize() the GPU may still be executing queued kernels.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        torch.cuda.empty_cache()
        print("[TaggerTrainer] GPU memory freed")
        datasets_db.close()
