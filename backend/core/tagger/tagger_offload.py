"""Pause / offload helpers for the tagger trainer.

The trainer's main loop calls ``handle.offload(decision)`` / ``handle.restore()``
at batch boundaries when the GPU coordinator requests a pause.  All work
runs on the trainer thread — no cross-thread ``.to()`` calls.

Three things have to move (or be cleared) to free GPU memory:

  1. ``model`` parameters + buffers — moved via in-place ``p.data = p.data.to(cpu)``
     so Parameter identity is preserved (the optimizer holds references
     by ``id()``).
  2. ``optimizer`` per-parameter state — moved via the
     ``state_dict() / load_state_dict()`` round-trip.  This is the
     officially-supported path for bnb 8-bit optimisers; direct
     ``.to(cpu)`` on the uint8 state buffers is not guaranteed safe.
  3. ``criterion`` (loss function) module buffers — for CS-ASL et al.
     this includes pi / a_pos / gamma_pos / m_pos / ... pre-computed
     tensors.  Standard ``module.to(cpu)`` suffices.

Gradients are *not* preserved — we zero ``p.grad`` and let the next
backward pass regenerate them.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

import torch

from core.gpu_coordinator import OffloadDecision, TrainerHandle  # type: ignore


def _move_state_dict_inplace(state: Dict[str, Any], device: torch.device | str) -> None:
    """Recursively move every torch.Tensor inside an optimizer state dict
    to *device*.  Leaves non-tensor entries (step counts, qmap globals
    that are already on the target device, etc.) untouched."""
    # state["state"] is { param_index: { key: tensor_or_scalar } }
    inner = state.get("state", {})
    for _idx, per_param in inner.items():
        if not isinstance(per_param, dict):
            continue
        for k, v in list(per_param.items()):
            if torch.is_tensor(v):
                per_param[k] = v.to(device, non_blocking=True)


class TaggerTrainerHandle:
    """TrainerHandle implementation for SigLIP2 tagger training.

    Constructed by the trainer in __init__ with just the pause events;
    model / optimizer / criterion references are attached at train() start
    via :meth:`attach` and detached on exit via :meth:`detach`.
    """

    def __init__(self, run_id: str, output_dir: str,
                 pause_event: threading.Event,
                 resumed_event: threading.Event,
                 restored_event: threading.Event,
                 device_index: Optional[int] = None):
        self.run_id = run_id
        self.output_dir = output_dir
        # None = the default device, which is also where generation runs, so the
        # coordinator pauses this trainer. A pinned index other than the
        # generation device tells it not to bother.
        self.device_index = device_index
        self.pause_event = pause_event
        self.resumed_event = resumed_event
        self.restored_event = restored_event
        self.pending_decision: Optional[OffloadDecision] = None

        # Set by attach()
        self._model: Optional[torch.nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._criterion: Optional[torch.nn.Module] = None
        self._processor = None  # AutoProcessor — for training-model inference
        self._vocabulary = None  # TagVocabulary snapshot — for training-model inference
        self._owner_tid: Optional[int] = None

        # Set by offload(), consumed by restore()
        self._param_origin_devices: Dict[int, torch.device] = {}
        self._buffer_origin_devices: Dict[int, torch.device] = {}
        self._cached_opt_state: Optional[Dict[str, Any]] = None
        self._swap_path: Optional[str] = None
        self._criterion_origin_device: Optional[torch.device] = None
        self._active_decision: Optional[OffloadDecision] = None

    # -- TrainerHandle protocol ------------------------------------------

    def trainer_label(self) -> str:
        return f"tagger:{self.run_id[:8]}"

    def estimate_state_bytes(self) -> int:
        if self._model is None:
            return 0
        m = sum(p.numel() * p.element_size() for p in self._model.parameters())
        b = sum(t.numel() * t.element_size() for t in self._model.buffers())
        # 8-bit optimiser state ≈ 2× params (state1 + state2 in AdamW8bit, uint8).
        # FP32 AdamW would be 8× params (exp_avg + exp_avg_sq in fp32) but in
        # practice we use 8-bit, so this is a safe ceiling.
        opt = m  # uint8 state1 (~param numel bytes) + state2 (~param numel)
        return m + b + 2 * opt

    # -- public API used from the trainer thread -------------------------

    def attach(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[torch.nn.Module],
        processor=None,
        vocabulary=None,
    ) -> None:
        """Called by the trainer once model / optimizer / criterion exist
        (start of train()).  Must be invoked from the trainer thread."""
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._processor = processor
        self._vocabulary = vocabulary
        self._owner_tid = threading.get_ident()

    def detach(self) -> None:
        """Called at train() exit (success, error, or stop)."""
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._processor = None
        self._vocabulary = None
        self._owner_tid = None
        # Cleanup any leftover swap file
        if self._swap_path and os.path.isfile(self._swap_path):
            try:
                os.remove(self._swap_path)
            except OSError:
                pass
        self._swap_path = None
        self._cached_opt_state = None

    def offload(self, decision: OffloadDecision) -> None:
        """Move trainer state off GPU according to *decision*.  Must be
        invoked from the trainer thread."""
        assert self._owner_tid is None or threading.get_ident() == self._owner_tid, \
            "TaggerTrainerHandle.offload() must run on the trainer thread"
        if decision.mode == "none":
            self._active_decision = decision
            return
        if self._model is None or self._optimizer is None:
            print(f"[TaggerHandle:{self.run_id[:8]}] offload called but trainer "
                  f"not yet attached; skipping")
            return

        torch.cuda.synchronize()

        # 1. Model params (in-place data swap so optimizer refs stay valid)
        self._param_origin_devices.clear()
        for p in self._model.parameters():
            self._param_origin_devices[id(p)] = p.device
            p.data = p.data.to("cpu", non_blocking=True)
            if p.grad is not None:
                p.grad = None
        self._buffer_origin_devices.clear()
        for b in self._model.buffers():
            self._buffer_origin_devices[id(b)] = b.device
            b.data = b.data.to("cpu", non_blocking=True)

        # 2. Optimizer state — state_dict round-trip (bnb-safe path)
        opt_state = self._optimizer.state_dict()
        _move_state_dict_inplace(opt_state, "cpu")
        if decision.mode in ("disk", "split"):
            assert decision.swap_dir, "disk/split mode requires swap_dir"
            self._swap_path = os.path.join(decision.swap_dir, "tagger_optim_state.pt")
            torch.save(opt_state, self._swap_path)
            opt_state = None   # drop in-memory copy
        else:
            self._cached_opt_state = opt_state
        # Clear the optimizer's per-parameter state to release the CUDA buffers
        self._optimizer.state.clear()

        # 3. Loss criterion buffers (CS-ASL pre-computed per-tag tensors)
        if self._criterion is not None:
            try:
                first_buf = next(self._criterion.buffers(), None)
                self._criterion_origin_device = first_buf.device if first_buf is not None else None
                self._criterion.to("cpu")
            except StopIteration:
                self._criterion_origin_device = None

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self._active_decision = decision
        print(f"[TaggerHandle:{self.run_id[:8]}] offload complete (mode={decision.mode})")

    def restore(self) -> None:
        """Move trainer state back to GPU.  Must be invoked from the
        trainer thread."""
        assert self._owner_tid is None or threading.get_ident() == self._owner_tid, \
            "TaggerTrainerHandle.restore() must run on the trainer thread"
        if self._active_decision is None or self._active_decision.mode == "none":
            self._active_decision = None
            return
        if self._model is None or self._optimizer is None:
            return

        device = torch.device("cuda")

        # 1. Model params/buffers
        for p in self._model.parameters():
            orig = self._param_origin_devices.get(id(p), device)
            p.data = p.data.to(orig, non_blocking=True)
        for b in self._model.buffers():
            orig = self._buffer_origin_devices.get(id(b), device)
            b.data = b.data.to(orig, non_blocking=True)
        self._param_origin_devices.clear()
        self._buffer_origin_devices.clear()

        # 2. Optimizer state
        if self._swap_path is not None and os.path.isfile(self._swap_path):
            opt_state = torch.load(self._swap_path, map_location="cpu", weights_only=False)
            _move_state_dict_inplace(opt_state, device)
            self._optimizer.load_state_dict(opt_state)
            try:
                os.remove(self._swap_path)
            except OSError:
                pass
            self._swap_path = None
        elif self._cached_opt_state is not None:
            _move_state_dict_inplace(self._cached_opt_state, device)
            self._optimizer.load_state_dict(self._cached_opt_state)
            self._cached_opt_state = None

        # 3. Criterion
        if self._criterion is not None and self._criterion_origin_device is not None:
            self._criterion.to(self._criterion_origin_device)
        self._criterion_origin_device = None

        torch.cuda.synchronize()
        mode_was = self._active_decision.mode
        self._active_decision = None
        print(f"[TaggerHandle:{self.run_id[:8]}] restore complete (mode was {mode_was})")

    # -- inference from training model -----------------------------------

    def can_predict(self) -> bool:
        """True when the training model is attached and currently on CUDA."""
        if self._model is None or self._processor is None or self._vocabulary is None:
            return False
        p = next(iter(self._model.parameters()), None)
        return p is not None and p.device.type == "cuda"

    def predict(
        self,
        image_bytes: bytes,
        threshold: float,
        *,
        assist: "Optional[dict]" = None,
        use_per_tag_threshold: bool = False,
        min_best_thr: float = 0.30,
        min_best_f1: float = 0.05,
        min_samples_for_per_tag: int = 5,
        use_ood_detection: bool = False,
        use_calibration: bool = False,
    ) -> dict:
        """Run inference with the training model (eval mode, no_grad).

        Returns the same dict shape as ``SigLIP2InferenceManager.predict()`` —
        including ``used_best_thr``, ``ood_distance``, ``has_calibration`` and
        per-tag ``raw_prob`` / ``cal_prob`` — so the "Use training model" toggle
        supports per-tag thresholds and OOD detection identically to the loaded
        inference model.

        *assist* is the vocab-agnostic metrics bundle produced by
        ``SigLIP2InferenceManager.build_training_assist()`` (name-keyed per-tag
        metrics, name-keyed calibration table, n_bins, OOD reference). All
        lookups are by tag NAME so they apply correctly even when the live
        training head has expanded beyond the inference checkpoint's vocabulary.

        Note: the OOD reference embedding space is the inference checkpoint's;
        the live training model's pooled embeddings drift from it as training
        progresses, so the training-model OOD distance is approximate.

        Raises RuntimeError if the model is not on CUDA (offloaded).
        """
        if not self.can_predict():
            raise RuntimeError("Training model not available for inference "
                               "(not attached or currently offloaded to CPU/disk)")

        import io
        from PIL import Image as PILImage
        from core.tagger.tag_selection import (
            select_tags, ood_threshold_scale, mahalanobis,
            apply_calibration_by_name,
        )

        vocab    = self._vocabulary
        _assist  = assist or {}
        _ood_ref = _assist.get("ood_ref") if use_ood_detection else None

        pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(images=[pil_img], return_tensors="pt")
        device = next(iter(self._model.parameters())).device
        inputs = {k: v.to(device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        # OOD: capture the CLS embedding (classification-head input) via a
        # transient forward_pre_hook, mirroring the inference manager's hook.
        _cap: dict = {}
        _hook = None
        if _ood_ref is not None and hasattr(self._model, "head"):
            def _capture_cls_emb(module, args):
                e = args[0].detach().float().cpu()
                if e.dim() > 1:
                    e = e[0]
                _cap["emb"] = e.numpy()
            _hook = self._model.head.register_forward_pre_hook(_capture_cls_emb)

        self._model.eval()
        try:
            with torch.no_grad():
                logits = self._model(**inputs)  # [1, V]
                probs = torch.sigmoid(logits[0]).cpu().float().numpy()  # [V]
        finally:
            if _hook is not None:
                _hook.remove()

        idx_to_tag = vocab.idx_to_tag
        tag_to_cat = vocab.tag_to_category

        # Calibration (name-keyed so a vocab-expanded head stays aligned).
        cal_probs = apply_calibration_by_name(
            probs, idx_to_tag, _assist.get("name_calibration"),
            int(_assist.get("n_bins", 100)),
        )
        raw_probs = probs
        _calibrated = False
        if cal_probs is not None and use_calibration:
            raw_probs = cal_probs
            _calibrated = True

        # Build the full item list (Quality/Rating handled separately below).
        all_items = []
        for i in range(len(raw_probs)):
            tag = idx_to_tag.get(i)
            if tag is None:
                continue
            category = tag_to_cat.get(tag, "General")
            item = {
                "tag":      tag,
                "prob":     float(raw_probs[i]),
                "raw_prob": float(raw_probs[i]),
                "category": category,
            }
            if cal_probs is not None:
                item["cal_prob"] = float(cal_probs[i])
            all_items.append(item)

        quality_items = [it for it in all_items if it["category"] == "Quality"]
        rating_items  = [it for it in all_items if it["category"] == "Rating"]
        quality_top = max(quality_items, key=lambda x: x["prob"]) if quality_items else None
        rating_top  = max(rating_items,  key=lambda x: x["prob"]) if rating_items  else None

        # OOD distance + dynamic threshold scale.
        ood_distance = None
        if _ood_ref is not None and "emb" in _cap:
            ood_distance = mahalanobis(_cap["emb"], _ood_ref["mu"], _ood_ref["cov_inv"])
        ood_t = (
            ood_threshold_scale(ood_distance, float(_ood_ref["p50"]), float(_ood_ref["p95"]))
            if (ood_distance is not None and _ood_ref is not None) else 0.0
        )

        _name_metrics = _assist.get("name_metrics")
        _use_ptt = bool(use_per_tag_threshold and _name_metrics)
        _get_metrics = (lambda tag: _name_metrics.get(tag)) if _use_ptt else None

        filtered, used_best_thr = select_tags(
            all_items,
            threshold=threshold,
            use_per_tag_threshold=_use_ptt,
            get_metrics=_get_metrics,
            min_best_thr=min_best_thr,
            min_best_f1=min_best_f1,
            min_samples_for_per_tag=min_samples_for_per_tag,
            ood_t=ood_t,
        )

        return {
            "tags":            filtered,
            "quality_top":     quality_top,
            "rating_top":      rating_top,
            "num_predicted":   len(filtered),
            "source":          "training_model",
            "run_id":          self.run_id,
            "calibrated":      _calibrated,
            "has_calibration": cal_probs is not None,
            "used_best_thr":   used_best_thr,
            "ood_distance":    ood_distance,
        }
