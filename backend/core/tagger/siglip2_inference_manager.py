"""
SigLIP2 Tagger Inference Manager

Singleton manager for loading and running SigLIP2-based tagger models
(both full-parameter and LoRA variants).
"""

from __future__ import annotations

import gc
import io
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_manager: Optional["SigLIP2InferenceManager"] = None


def get_siglip2_inference_manager() -> "SigLIP2InferenceManager":
    global _manager
    if _manager is None:
        _manager = SigLIP2InferenceManager()
    return _manager


# ---------------------------------------------------------------------------
# Helper: peek at safetensors keys without loading weights
# ---------------------------------------------------------------------------

def _detect_model_type(checkpoint_path: str) -> str:
    """Return 'lora' if checkpoint contains LoRA keys, else 'full'."""
    from safetensors import safe_open
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    if any(k.startswith("lora.") for k in keys):
        return "lora"
    return "full"


def _read_metadata(checkpoint_path: str) -> dict:
    meta_path = checkpoint_path.replace(".safetensors", "_metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------

# ONNX intermediate node name for CLS embedding (input to final classification head).
# Used for OOD detection via Mahalanobis distance on the 1152-dim feature vector.
_OOD_EMB_NODE = "/vision_encoder/head/Gather_1_output_0"


class SigLIP2InferenceManager:
    """Manages a single loaded SigLIP2 tagger model for inference."""

    def __init__(self) -> None:
        self.model: Optional[nn.Module] = None
        self.onnx_session = None  # onnxruntime.InferenceSession when model_type == "onnx"
        self.processor = None
        self.is_naflex: bool = True   # NaFlex (variable-res) vs standard (fixed-res)
        self.vocabulary: Optional[Dict[str, Any]] = None   # from vocabulary.json
        self.checkpoint_path: str = ""
        self.vision_encoder_path: str = ""
        self.vocab_path: str = ""
        self.model_type: str = ""   # "full" | "lora" | "onnx"
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.logit_bias: Optional["np.ndarray"] = None  # [num_tags] float32, CS-ASL calibration
        # Sparse Likelihood-Ratio matrix for context-conditional inference (Phase 1).
        # Loaded from <model_dir>/lr_matrix.npz when present; None otherwise.
        # Format: {anchor_idx, offsets, target_idx, lr_values, anchor_lookup}.
        self.lr_matrix: Optional[Dict[str, Any]] = None
        # Current calibration settings (used by recompute_calibration_table)
        self.calib_method: str = "jeffreys"
        self.calib_eps: float = 0.5
        self.calib_prior_strength: float = 10.0
        # OOD detection state (Mahalanobis distance on CLS embedding).
        # ood_ref: dict with mu, cov_inv, p50 loaded from {onnx_base}_ood_ref.npz
        # ood_emb_session: ORT session for modified ONNX with embedding output
        self.ood_ref: Optional[Dict[str, Any]] = None
        self.ood_emb_session = None
        # PyTorch path: forward_pre_hook on model.head captures the CLS embedding.
        self._last_cls_emb: Optional["np.ndarray"] = None
        self._cls_emb_hook_handle = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(
        self,
        checkpoint_path: str,
        vocab_path: str,
        vision_encoder_path: str = "",
        lora_rank: int = 32,
        lora_alpha: float = 16.0,
    ) -> Dict[str, Any]:
        """Load a SigLIP2 tagger checkpoint.

        Auto-detects model type (full / lora) from the checkpoint keys.
        If *vision_encoder_path* is empty and model_type is 'lora', raises ValueError.
        """
        self.unload()

        checkpoint_path = checkpoint_path.strip().strip('"').strip("'")
        vocab_path = vocab_path.strip().strip('"').strip("'")
        vision_encoder_path = vision_encoder_path.strip().strip('"').strip("'")

        # Early file-existence checks — fail fast before any network/IO work.
        if not checkpoint_path:
            raise ValueError("checkpoint_path is empty")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Resolve vocabulary path with per-checkpoint priority.
        # The per-checkpoint snapshot ``<ckpt_basename>_vocabulary.json`` is
        # frozen at save time and is the authoritative source for tag→idx
        # mapping.  The common ``vocabulary.json`` in the run directory is
        # overwritten on every new run start, so it may not match the
        # checkpoint we are loading — fall back only with a warning.
        if not checkpoint_path.endswith(".onnx"):
            _ckpt_dir  = os.path.dirname(os.path.abspath(checkpoint_path))
            _ckpt_base = os.path.splitext(os.path.basename(checkpoint_path))[0]
            _per_ckpt_vocab = os.path.join(_ckpt_dir, f"{_ckpt_base}_vocabulary.json")
            if os.path.isfile(_per_ckpt_vocab):
                if vocab_path and os.path.normpath(vocab_path) != os.path.normpath(_per_ckpt_vocab):
                    print(f"[SigLIP2InferenceManager] Using per-checkpoint vocabulary "
                          f"'{os.path.basename(_per_ckpt_vocab)}' (caller passed "
                          f"'{os.path.basename(vocab_path)}')")
                vocab_path = _per_ckpt_vocab
            elif vocab_path and os.path.isfile(vocab_path):
                print(f"[SigLIP2InferenceManager] WARNING: per-checkpoint vocabulary "
                      f"'{_ckpt_base}_vocabulary.json' not found; falling back to "
                      f"'{os.path.basename(vocab_path)}'. Tag→idx alignment with "
                      f"the checkpoint head cannot be verified — predictions may "
                      f"be wrong if vocabulary changed since the checkpoint was saved.")

        # 1. Load vocabulary (needed for all model types)
        with open(vocab_path, "r", encoding="utf-8") as fh:
            vocab = json.load(fh)
        idx_to_tag       = {int(k): v for k, v in vocab["idx_to_tag"].items()}
        tag_to_category  = vocab.get("tag_to_category", {})
        num_tags         = len(idx_to_tag)

        # 2. Read checkpoint metadata (before processor — needed to resolve processor repo)
        from core.tagger.siglip2_tagger_model import SIGLIP2_DEFAULT_REPO_ID
        if checkpoint_path.endswith(".onnx"):
            _meta_path = checkpoint_path.replace(".onnx", "_metadata.json")
            meta = json.load(open(_meta_path, encoding="utf-8")) if os.path.isfile(_meta_path) else {}
        else:
            meta = _read_metadata(checkpoint_path)

        # 3. Load processor — repo derived from metadata so it always matches the vision encoder
        from transformers import AutoProcessor
        processor_repo = meta.get("vision_encoder_repo", SIGLIP2_DEFAULT_REPO_ID)
        print(f"[SigLIP2Manager] Loading processor from {processor_repo}...")
        try:
            processor = AutoProcessor.from_pretrained(processor_repo, local_files_only=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(processor_repo)
        # Detect NaFlex vs standard by probing the processor output.
        # Prefer metadata field (written by trainer/exporter); fall back to probe.
        if "is_naflex" in meta:
            is_naflex = bool(meta["is_naflex"])
        else:
            _probe = processor(images=[Image.new("RGB", (64, 64))], return_tensors="pt")
            is_naflex = "pixel_attention_mask" in _probe and "spatial_shapes" in _probe
        _mode_str = "NaFlex" if is_naflex else "standard"
        print(f"[SigLIP2Manager] Processor mode: {_mode_str}")

        # 4. Load model (branched by type)
        if checkpoint_path.endswith(".onnx"):
            # --- ONNX model ---
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 2
            self.onnx_session = ort.InferenceSession(
                checkpoint_path,
                sess_options=opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            model_type = "onnx"
            _provider = self.onnx_session.get_providers()[0]
            print(f"[SigLIP2Manager] ONNX session created | provider={_provider}")
        else:
            # --- safetensors model (full / lora) ---
            model_type = _detect_model_type(checkpoint_path)
            # meta already read above

            lora_rank  = int(meta.get("lora_rank",  lora_rank))
            lora_alpha = float(meta.get("lora_alpha", lora_alpha))

            from core.tagger.siglip2_tagger_model import (
                SigLIP2TaggerModel,
                SigLIP2TaggerLoRAModel,
            )

            if model_type == "lora":
                if not vision_encoder_path:
                    from core.tagger.siglip2_tagger_model import SIGLIP2_DEFAULT_REPO_ID
                    # Prefer base_model_path (locally fine-tuned base) over vision_encoder_repo
                    # (HF architecture repo).  base_model_path is written by the trainer when the
                    # LoRA was trained on a local safetensors checkpoint rather than an HF repo,
                    # so that merge/inference uses the correct base weights.
                    _base_path = meta.get("base_model_path", "")
                    if _base_path:
                        # Resolve relative path (filename only) against the checkpoint directory
                        if not os.path.isabs(_base_path):
                            _base_path = os.path.join(os.path.dirname(checkpoint_path), _base_path)
                    if _base_path and os.path.isfile(_base_path):
                        vision_encoder_path = _base_path
                        print(f"[SigLIP2Manager] Using locally fine-tuned base from metadata: {vision_encoder_path}")
                    else:
                        if _base_path:
                            print(f"[SigLIP2Manager] WARNING: base_model_path in metadata not found ({_base_path}); falling back to HF repo")
                        vision_encoder_path = meta.get("vision_encoder_repo", SIGLIP2_DEFAULT_REPO_ID)
                        print(f"[SigLIP2Manager] vision_encoder_path not provided; using HF repo from metadata: {vision_encoder_path}")
                model = SigLIP2TaggerLoRAModel.load_checkpoint(
                    checkpoint_path=checkpoint_path,
                    vision_encoder_path=vision_encoder_path,
                    num_tags=num_tags,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                )
            else:
                model = SigLIP2TaggerModel.load_checkpoint(
                    checkpoint_path=checkpoint_path,
                    vision_encoder_path=vision_encoder_path,
                    num_tags=num_tags,
                )

            model.eval()
            model.to(self.device)
            self.model = model

        # 5. Store state
        self.processor           = processor
        self.is_naflex           = is_naflex
        # Build reverse map for tag→idx lookup (needed for conditional inference)
        tag_to_idx = {tag: idx for idx, tag in idx_to_tag.items()}
        self.vocabulary          = {
            "idx_to_tag":      idx_to_tag,
            "tag_to_category": tag_to_category,
            "tag_to_idx":      tag_to_idx,
        }
        self.checkpoint_path     = checkpoint_path
        self.vision_encoder_path = vision_encoder_path
        self.vocab_path          = vocab_path
        self.model_type          = model_type

        # 6. Logit bias correction (label_stats.npz) is intentionally NOT applied.
        # The analytical CS-ASL equilibrium p*(pi) = pi^(1-rho)/(pi^(1-rho)+(1-pi)^(1-rho))
        # assumes ideal gradient convergence.  Empirically (verified on cs_asl ckpt at
        # step 438798), the trained model's raw output mean is ~10× higher than the
        # average p*_n, so subtracting bias_n (largely negative for rare labels) over-
        # shoots and pushes nearly every label above threshold.  Use raw probabilities
        # with a manually chosen threshold (0.5–0.9) instead.
        self.logit_bias = None

        # 7. LR matrix (optional, for context-based conditional inference).
        # File path: <model_dir>/lr_matrix.npz, generated by
        # `python -m core.tagger.lr_matrix_builder ...`.
        self.lr_matrix = None
        _lr_path = os.path.join(os.path.dirname(vocab_path), "lr_matrix.npz")
        if os.path.isfile(_lr_path):
            try:
                import numpy as _np
                _data = _np.load(_lr_path)
                self.lr_matrix = {
                    "anchor_idx":    _data["anchor_tag_indices"].astype(_np.int32),
                    "offsets":       _data["anchor_to_offset"].astype(_np.int32),
                    "target_idx":    _data["target_tag_indices"].astype(_np.int32),
                    "lr_values":     _data["lr_values"].astype(_np.float32),
                }
                self.lr_matrix["anchor_lookup"] = {
                    int(idx): pos
                    for pos, idx in enumerate(self.lr_matrix["anchor_idx"])
                }
                print(
                    f"[SigLIP2Manager] LR matrix loaded "
                    f"({len(self.lr_matrix['anchor_idx'])} anchors, "
                    f"{len(self.lr_matrix['lr_values'])} entries)"
                )
            except Exception as _e:
                print(f"[SigLIP2Manager] WARNING: LR matrix load failed: {_e}")
                self.lr_matrix = None

        # 8. Per-tag threshold metrics (optional, saved alongside checkpoint by tagger_trainer).
        self.tag_metrics: Optional[Dict] = None
        if not checkpoint_path.endswith(".onnx"):
            _metrics_path = os.path.join(_ckpt_dir, f"{_ckpt_base}_tag_metrics.npz")
            if os.path.isfile(_metrics_path):
                try:
                    from core.tagger.tag_metrics_accumulator import TagMetricsAccumulator
                    self.tag_metrics = TagMetricsAccumulator.load(_metrics_path)
                    # Restore calibration settings from NPZ (if saved by newer trainer)
                    _cm = self.tag_metrics.get("calib_method")
                    if _cm is not None:
                        _cm_val = _cm.item() if hasattr(_cm, "item") else str(_cm)
                        self.calib_method = str(_cm_val)
                    _ce = self.tag_metrics.get("calib_eps")
                    if _ce is not None:
                        self.calib_eps = float(_ce.item() if hasattr(_ce, "item") else _ce)
                    print(
                        f"[SigLIP2Manager] Per-tag metrics loaded "
                        f"({int(self.tag_metrics.get('n_bins', 100))} bins, "
                        f"calib={self.calib_method})"
                    )
                except Exception as _e:
                    print(f"[SigLIP2Manager] WARNING: tag_metrics load failed: {_e}")
                    self.tag_metrics = None

        # 9. OOD reference (optional, built by build_ood_reference()).
        # File: {checkpoint_base}_ood_ref.npz alongside the checkpoint.
        self.ood_ref = None
        self.ood_emb_session = None
        self._last_cls_emb = None
        self._cls_emb_hook_handle = None
        _ood_ref_path = os.path.join(_ckpt_dir, f"{_ckpt_base}_ood_ref.npz")
        if os.path.isfile(_ood_ref_path):
            try:
                self._load_ood_reference(_ood_ref_path)
                print(f"[SigLIP2Manager] OOD reference loaded from {_ood_ref_path}")
            except Exception as _e:
                print(f"[SigLIP2Manager] WARNING: OOD reference load failed: {_e}")
        if not checkpoint_path.endswith(".onnx") and self.ood_ref is not None and self.model is not None:
            # Register a persistent forward_pre_hook on the classification head to
            # capture the CLS embedding (head input) for Mahalanobis distance computation.
            def _capture_cls_emb(module, args):
                import numpy as np
                # args[0]: [1, pool_dim] or [pool_dim] tensor
                emb = args[0].detach().float().cpu()
                if emb.dim() > 1:
                    emb = emb[0]
                self._last_cls_emb = emb.numpy()
            self._cls_emb_hook_handle = self.model.head.register_forward_pre_hook(_capture_cls_emb)

        print(
            f"[SigLIP2Manager] Loaded {model_type} model | "
            f"{num_tags} tags | {self.device}"
        )
        return {
            "status":     "ok",
            "model_type": model_type,
            "num_tags":   num_tags,
        }

    # ------------------------------------------------------------------

    def predict(
        self,
        image_bytes: bytes,
        threshold: float = 0.5,
        max_num_patches: int = 256,
        known_tags_pos: Optional[List[str]] = None,
        known_tags_neg: Optional[List[str]] = None,
        context_method: str = "none",
        context_lambda: float = 0.5,
        use_per_tag_threshold: bool = False,
        min_samples_for_per_tag: int = 5,
        min_best_thr: float = 0.30,
        min_best_f1: float = 0.05,
        use_calibration: bool = False,
        display_calibration: bool = False,
        use_ood_detection: bool = False,
    ) -> Dict[str, Any]:
        """Run inference on raw image bytes.

        Conditional inference parameters
        --------------------------------
        known_tags_pos : list of tags asserted to be PRESENT in the image
        known_tags_neg : list of tags asserted to be ABSENT from the image
        context_method : "none" | "head_sim" | "lr_matrix"
            How to derive the per-tag correction vector from known tags.
        context_lambda : strength multiplier (0 = off, 1 = full strength).

        When known tags are provided and *context_method* != "none", a
        correction vector is added to the raw logits before sigmoid:

            adjusted_logit_n = raw_logit_n
                             + Σ_{c in known_pos} weight(n, c)
                             - Σ_{c in known_neg} weight(n, c)

        For "head_sim" the weight is λ × cos_sim(head.W[n], head.W[c]).
        For "lr_matrix" the weight is λ × LR(n, c) from the precomputed
        sparse co-occurrence matrix (loaded by load_model() if available).
        Quality / Rating tags are excluded from the correction so that
        their per-image top-1 selection stays content-driven.  Tags listed
        in known_tags_pos are forced to prob=1.0 in the output.

        Returns:
            {
              "tags": [{"tag": str, "prob": float, "category": str}, ...],
              "quality_top": {"tag": str, "prob": float, "category": "Quality"} | None,
              "rating_top":  {"tag": str, "prob": float, "category": "Rating"}  | None,
              "num_predicted": int,
            }
        Tags are filtered by *threshold* (except Quality / Rating which always
        return the top-scoring item regardless of threshold).
        """
        if self.model is None and self.onnx_session is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        import numpy as np

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        proc_kwargs = {"images": pil_image, "return_tensors": "pt"}
        if self.is_naflex:
            proc_kwargs["max_num_patches"] = max_num_patches
        inputs = self.processor(**proc_kwargs)

        # Pre-compute context correction (numpy [num_tags]) once; it is shared
        # between the ONNX and torch paths.
        context_correction = self._compute_context_correction(
            known_tags_pos, known_tags_neg, context_method, context_lambda,
        )

        ood_distance: Optional[float] = None
        if self.model_type == "onnx":
            pv_np = inputs["pixel_values"].float().numpy()
            # Use OOD embedding session when requested and available
            _use_ood = (
                use_ood_detection
                and self.ood_ref is not None
                and self.ood_emb_session is not None
            )
            if _use_ood:
                # OOD session outputs: [logits, embedding]
                ood_out = self.ood_emb_session.run(
                    ["logits", _OOD_EMB_NODE],
                    {"pixel_values": pv_np},
                )
                logits_np = ood_out[0][0]  # [num_tags]
                emb_np    = ood_out[1][0]  # [cls_dim]
                ood_distance = float(self._compute_mahalanobis(emb_np))
            elif self.is_naflex:
                pam_np = inputs["pixel_attention_mask"].float().numpy()
                ss_np  = inputs["spatial_shapes"].numpy().astype(np.int64)
                outputs = self.onnx_session.run(
                    ["logits"],
                    {"pixel_values": pv_np, "pixel_attention_mask": pam_np, "spatial_shapes": ss_np},
                )
                logits_np = outputs[0][0]
            else:
                outputs = self.onnx_session.run(["logits"], {"pixel_values": pv_np})
                logits_np = outputs[0][0]  # [num_tags]
            if self.logit_bias is not None:
                logits_np = logits_np - self.logit_bias
            if context_correction is not None:
                logits_np = logits_np + context_correction.astype(logits_np.dtype)
            probs = 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64)))
        else:
            pixel_values = inputs["pixel_values"].to(self.device)
            if self.is_naflex:
                pixel_attn_mask = inputs["pixel_attention_mask"].to(self.device)
                spatial_shapes  = inputs["spatial_shapes"].to(self.device)
            else:
                pixel_attn_mask = torch.zeros(0, dtype=torch.int32, device=self.device)
                spatial_shapes  = torch.zeros(0, dtype=torch.int64, device=self.device)
            with torch.no_grad():
                logits = self.model(pixel_values, pixel_attn_mask, spatial_shapes)
            if (
                use_ood_detection
                and self.ood_ref is not None
                and self._last_cls_emb is not None
            ):
                ood_distance = float(self._compute_mahalanobis(self._last_cls_emb))
            _logits = logits[0]
            if self.logit_bias is not None:
                _logits = _logits - torch.from_numpy(self.logit_bias).to(_logits.device)
            if context_correction is not None:
                _logits = _logits + torch.from_numpy(context_correction).to(_logits.device)
            probs = torch.sigmoid(_logits).cpu().numpy()  # [num_tags]

        # Keep raw sigmoid probs for best_thr filtering; calibrated only for display
        raw_probs = probs.copy()

        # use_calibration (legacy): replaces probs used for both filtering AND display
        _calibrated = False
        if use_calibration and self.tag_metrics is not None:
            _calib = self.tag_metrics.get("calibration_table")
            if _calib is not None:
                _nb = self.tag_metrics.get("n_bins", 100)
                _n_bins = int(_nb) if np.ndim(_nb) == 0 else int(_nb[0])
                _bin_idx = np.clip(
                    (probs * _n_bins).astype(np.int32), 0, _n_bins - 1
                )
                probs = _calib[np.arange(len(probs)), _bin_idx].astype(np.float32)
                raw_probs = probs  # in legacy mode, display == filtered probs
                _calibrated = True

        # display_calibration: keep raw probs for filtering, use calibrated for display only
        display_probs = raw_probs
        _display_calibrated = False
        if display_calibration and not use_calibration and self.tag_metrics is not None:
            _calib = self.tag_metrics.get("calibration_table")
            if _calib is not None:
                _nb = self.tag_metrics.get("n_bins", 100)
                _n_bins = int(_nb) if np.ndim(_nb) == 0 else int(_nb[0])
                _bin_idx = np.clip(
                    (raw_probs * _n_bins).astype(np.int32), 0, _n_bins - 1
                )
                _cal = _calib[np.arange(len(raw_probs)), _bin_idx].astype(np.float32)
                _nan = np.isnan(_cal)
                if _nan.any():
                    _cal[_nan] = raw_probs[_nan]
                display_probs = _cal
                _display_calibrated = True

        idx_to_tag      = self.vocabulary["idx_to_tag"]
        tag_to_category = self.vocabulary["tag_to_category"]
        tag_to_idx      = self.vocabulary.get("tag_to_idx", {})

        # Force known-positive tags to prob=1.0 in the output (after sigmoid).
        if known_tags_pos:
            for _tag in known_tags_pos:
                _idx = tag_to_idx.get(_tag)
                if _idx is not None:
                    raw_probs[_idx] = 1.0
                    display_probs[_idx] = 1.0

        # Build full list — display_probs for shown values, raw_probs for filtering
        all_items: List[Dict] = []
        for i in range(len(raw_probs)):
            tag      = idx_to_tag.get(i, f"__unk_{i}__")
            category = tag_to_category.get(tag, "Unknown")
            all_items.append({
                "tag": tag,
                "prob": float(display_probs[i]),
                "raw_prob": float(raw_probs[i]),
                "category": category,
            })

        # Quality / Rating: pick the max regardless of threshold
        quality_top: Optional[Dict] = None
        rating_top:  Optional[Dict] = None

        quality_items = [it for it in all_items if it["category"] == "Quality"]
        rating_items  = [it for it in all_items if it["category"] == "Rating"]

        if quality_items:
            quality_top = max(quality_items, key=lambda x: x["prob"])
        if rating_items:
            rating_top  = max(rating_items,  key=lambda x: x["prob"])

        # Threshold-filtered tags (exclude Quality / Rating from the main list)
        # Filtering always uses raw_prob; display prob is in the "prob" field.
        # OOD dynamic threshold scale factor (0 = in-dist, 1 = fully OOD).
        # Only computed when OOD detection is active and distance is available.
        _ood_t: float = 0.0
        if ood_distance is not None and self.ood_ref is not None:
            _p50 = float(self.ood_ref["p50"])
            # Linear ramp: 0 at in-dist p50, 1 at p50*40 (well below OOD median ~1143)
            _ood_t = max(0.0, min(1.0, (ood_distance - _p50) / (_p50 * 39.0)))

        _used_best_thr = False
        if use_per_tag_threshold and self.tag_metrics is not None:
            import math
            _bthr = self.tag_metrics.get("best_thr")
            _bf1  = self.tag_metrics.get("best_f1")
            _npos = self.tag_metrics.get("n_pos")
            filtered = []
            for it in all_items:
                if it["category"] in ("Quality", "Rating"):
                    continue
                _idx = tag_to_idx.get(it["tag"])
                thr_t = threshold  # fallback
                if (
                    _idx is not None
                    and _bthr is not None
                    and _npos is not None
                    and int(_npos[_idx]) >= min_samples_for_per_tag
                    and not math.isnan(float(_bthr[_idx]))
                ):
                    raw_thr = float(_bthr[_idx])
                    # Skip tag entirely if best_f1 is below minimum (unreliable detector)
                    if _bf1 is not None and not math.isnan(float(_bf1[_idx])):
                        if float(_bf1[_idx]) < min_best_f1:
                            continue
                    # Clamp best_thr to minimum to suppress noise-level FPs
                    thr_t = max(raw_thr, min_best_thr)
                # OOD dynamic threshold: raise threshold for Character/Copyright
                # proportionally to how far the image is from the training distribution.
                if _ood_t > 0.0 and it["category"] in ("Character", "Copyright"):
                    thr_t = thr_t + _ood_t * (0.85 - thr_t)
                if it["raw_prob"] >= thr_t:
                    filtered.append(it)
            _used_best_thr = True
        else:
            filtered = [
                it for it in all_items
                if it["raw_prob"] >= threshold
                and it["category"] not in ("Quality", "Rating")
            ]
        filtered.sort(key=lambda x: x["prob"], reverse=True)

        return {
            "tags":               filtered,
            "quality_top":        quality_top,
            "rating_top":         rating_top,
            "num_predicted":      len(filtered),
            "calibrated":         _calibrated or _display_calibrated,
            "display_calibrated": _display_calibrated,
            "used_best_thr":      _used_best_thr,
            "ood_distance":       ood_distance,
        }

    # ------------------------------------------------------------------
    # Context-conditional inference helpers
    # ------------------------------------------------------------------

    # Categories whose logits are intentionally NOT modified by context
    # correction (they have their own top-1 selection per image).
    _CONTEXT_EXCLUDED_CATEGORIES = ("Quality", "Rating")

    def _compute_context_correction(
        self,
        pos_tags: Optional[List[str]],
        neg_tags: Optional[List[str]],
        method: str,
        lam: float,
    ) -> Optional["np.ndarray"]:
        """Compute per-tag logit correction from known context tags.

        Returns a numpy array [num_tags] (float32) or None when no
        correction should be applied.
        """
        import numpy as np
        if method in (None, "", "none") or lam == 0.0:
            return None
        if not pos_tags and not neg_tags:
            return None
        if self.vocabulary is None:
            return None

        tag_to_idx = self.vocabulary.get("tag_to_idx") or {}
        tag_to_cat = self.vocabulary.get("tag_to_category") or {}
        n_tags = len(self.vocabulary["idx_to_tag"])

        pos_idx = [tag_to_idx[t] for t in (pos_tags or []) if t in tag_to_idx]
        neg_idx = [tag_to_idx[t] for t in (neg_tags or []) if t in tag_to_idx]
        if not pos_idx and not neg_idx:
            return None

        correction = np.zeros(n_tags, dtype=np.float32)

        if method == "head_sim":
            if self.model is None or not hasattr(self.model, "head"):
                return None
            with torch.no_grad():
                W = self.model.head.weight.detach().float()           # [N, D]
                Wn = F.normalize(W, dim=-1)
                for c in pos_idx:
                    sim = (Wn @ Wn[c]).cpu().numpy()                  # [N]
                    correction += float(lam) * sim
                for c in neg_idx:
                    sim = (Wn @ Wn[c]).cpu().numpy()
                    correction -= float(lam) * sim

        elif method == "lr_matrix":
            if self.lr_matrix is None:
                # Caller asked for LR but it's not loaded — silently fall back to
                # head_sim so that the inference still benefits from context.
                return self._compute_context_correction(
                    pos_tags, neg_tags, "head_sim", lam,
                )
            for c in pos_idx:
                self._apply_lr_correction(correction, c, +1, lam)
            for c in neg_idx:
                self._apply_lr_correction(correction, c, -1, lam)

        else:
            # Unknown method → no correction
            return None

        # Zero out excluded categories (Quality / Rating) so they are
        # decided purely by image content.
        for i in range(n_tags):
            cat = tag_to_cat.get(self.vocabulary["idx_to_tag"].get(i), "")
            if cat in self._CONTEXT_EXCLUDED_CATEGORIES:
                correction[i] = 0.0

        # Clip to avoid extreme overrides
        np.clip(correction, -5.0, 5.0, out=correction)
        return correction

    def _apply_lr_correction(
        self,
        correction: "np.ndarray",
        anchor_idx: int,
        sign: int,
        lam: float,
    ) -> None:
        """Add λ × sign × LR(:, anchor_idx) to the correction vector in place."""
        if self.lr_matrix is None:
            return
        pos = self.lr_matrix["anchor_lookup"].get(int(anchor_idx))
        if pos is None:
            return
        offsets = self.lr_matrix["offsets"]
        targets = self.lr_matrix["target_idx"][offsets[pos]:offsets[pos + 1]]
        values  = self.lr_matrix["lr_values"][offsets[pos]:offsets[pos + 1]]
        correction[targets] += float(sign) * float(lam) * values

    # ------------------------------------------------------------------

    def merge_lora_and_save(self, output_path: str) -> str:
        """Merge LoRA weights into the vision encoder and save as a full model.

        Only valid when model_type == 'lora'.
        """
        if self.model is None and self.onnx_session is None:
            raise RuntimeError("No model loaded.")
        if self.model_type == "onnx":
            raise ValueError("Cannot merge an ONNX model. Load a LoRA safetensors checkpoint instead.")
        from core.tagger.siglip2_tagger_model import SigLIP2TaggerLoRAModel
        if not isinstance(self.model, SigLIP2TaggerLoRAModel):
            raise ValueError("merge_lora_and_save is only valid for LoRA models.")

        # Treat output_path as a directory; always save as model.safetensors.
        # If empty, fall back to a "merged" subdirectory alongside the checkpoint.
        output_dir = output_path.strip().strip('"').strip("'")
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(self.checkpoint_path), "merged")
            print(f"[SigLIP2Manager] output_path not specified; saving to {output_dir}")
        output_name = "model"
        os.makedirs(output_dir, exist_ok=True)

        # F5: build clean metadata — keep architecture/training info, strip local paths.
        _raw_meta = _read_metadata(self.checkpoint_path)
        from core.tagger.siglip2_tagger_model import SIGLIP2_DEFAULT_REPO_ID
        _KEEP_KEYS = {
            "num_tags", "lora_rank", "lora_alpha", "training_method",
            "use_tag_aliases", "category_counts", "best_f1", "best_threshold",
            "vision_encoder_repo", "cls_dim", "hidden_proj_dim",
        }
        meta = {k: v for k, v in _raw_meta.items() if k in _KEEP_KEYS}
        meta["training_method"] = "full"   # merged checkpoint IS a full model
        if "vision_encoder_repo" not in meta:
            meta["vision_encoder_repo"] = SIGLIP2_DEFAULT_REPO_ID

        saved = self.model.save_merged_checkpoint(output_dir, output_name, meta)

        # Copy the vocabulary alongside the merged checkpoint so future runs
        # that load this file as their base model can do tag-name alignment
        # in _inherit_head.  Two filenames are written:
        #   - {output_name}_vocabulary.json (per-checkpoint snapshot — preferred
        #     by inference manager and _inherit_head)
        #   - vocabulary.json (legacy/general — also recognized)
        if self.vocab_path and os.path.isfile(self.vocab_path):
            try:
                import shutil as _shutil
                per_ckpt = os.path.join(output_dir, f"{output_name}_vocabulary.json")
                general  = os.path.join(output_dir, "vocabulary.json")
                _shutil.copyfile(self.vocab_path, per_ckpt)
                if not os.path.exists(general):
                    _shutil.copyfile(self.vocab_path, general)
                print(f"[SigLIP2Manager] Vocabulary copied: {os.path.basename(per_ckpt)} (+ vocabulary.json)")
            except Exception as e:
                print(f"[SigLIP2Manager] WARNING: failed to copy vocabulary alongside merged checkpoint: {e}")
        else:
            print(f"[SigLIP2Manager] WARNING: no vocab_path on inference manager — merged checkpoint "
                  f"has no vocabulary.json; downstream tag-name alignment will fall back to positional copy.")

        print(f"[SigLIP2Manager] Merged LoRA checkpoint saved → {saved}")
        return saved

    # ------------------------------------------------------------------

    def export_onnx(
        self,
        output_path: str,
        max_num_patches: int = 256,
        strip_unknown_tags: bool = False,
    ) -> Tuple[str, str]:
        """Export the model to ONNX format.

        If the model is a LoRA model, merges weights first using a temporary
        SigLIP2TaggerModel.

        strip_unknown_tags: if True, remove head rows for Unknown-category tags
        before export and write a filtered vocabulary alongside the ONNX file.

        Returns (onnx_path, vocab_path).
        """
        if self.model is None and self.onnx_session is None:
            raise RuntimeError("No model loaded.")
        if self.model_type == "onnx":
            raise ValueError("Cannot export an ONNX model to ONNX. Load a safetensors checkpoint instead.")

        from core.tagger.siglip2_tagger_model import (
            LoRALinear,
            SigLIP2TaggerLoRAModel,
            SigLIP2TaggerModel,
        )

        # If empty, fall back to an "onnx" subdirectory alongside the checkpoint.
        output_path = output_path.strip().strip('"').strip("'")
        if not output_path:
            ckpt_stem = os.path.splitext(os.path.basename(self.checkpoint_path))[0]
            output_path = os.path.join(
                os.path.dirname(self.checkpoint_path), "onnx", f"{ckpt_stem}.onnx"
            )
            print(f"[SigLIP2Manager] output_path not specified; saving ONNX to {output_path}")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # If LoRA: build an export model from the already-loaded base + LoRA.
        # No filesystem re-read, no HuggingFace round-trip: deep-copy the live
        # model, merge each LoRA delta into its base Linear's weight, then
        # replace the LoRALinear wrappers with their inner Linear so the
        # module hierarchy matches a non-LoRA SigLIP2TaggerModel.  This
        # preserves whatever architecture / fine-tuned base the inference
        # manager actually loaded (e.g. a patch14-384 base, or a chain of
        # previously-merged tagger runs), without depending on the
        # vision_encoder_repo metadata being accurate.
        if isinstance(self.model, SigLIP2TaggerLoRAModel):
            import copy as _copy
            print(f"[SigLIP2Manager] ONNX export: merging LoRA into the live base "
                  f"(no re-load; is_naflex={self.is_naflex})")
            export_lora = _copy.deepcopy(self.model)
            for _lm in export_lora._lora_modules.values():
                _lm.merge_into_base()

            def _strip_lora_inplace(mod: nn.Module) -> None:
                """Replace every LoRALinear in *mod* (recursively) with its inner .base."""
                for _name, _child in list(mod.named_children()):
                    if isinstance(_child, LoRALinear):
                        setattr(mod, _name, _child.base)
                    else:
                        _strip_lora_inplace(_child)

            _strip_lora_inplace(export_lora.vision_encoder)

            num_tags = export_lora.head.out_features
            # LoRA mode ignores cls_dim/hidden_proj_dim by design, so the
            # wrapper has no custom_pooler — build a default-pooled
            # SigLIP2TaggerModel and copy across the trained head.
            export_model = SigLIP2TaggerModel(
                num_tags=num_tags,
                vision_encoder=export_lora.vision_encoder,
                is_naflex=self.is_naflex,
            )
            export_model.head.load_state_dict(export_lora.head.state_dict())
        else:
            export_model = self.model
            num_tags     = self.model.head.out_features

        export_model.eval().to("cpu")

        # Optionally strip Unknown-category tag heads before export.
        # Builds a new head with only non-Unknown rows and writes a filtered
        # vocabulary alongside the ONNX file.
        export_vocab_data: Optional[dict] = None  # None → use self.vocab_path as-is
        if strip_unknown_tags and self.vocabulary is not None:
            tag_to_idx: dict = self.vocabulary.get("tag_to_idx", {})
            tag_to_cat: dict = self.vocabulary.get("tag_to_category", {})
            keep_indices = sorted(
                idx for tag, idx in tag_to_idx.items()
                if tag_to_cat.get(tag, "General") != "Unknown"
            )
            removed = num_tags - len(keep_indices)
            if removed > 0:
                import torch as _torch
                _keep = _torch.tensor(keep_indices, dtype=_torch.long)
                new_head = nn.Linear(export_model.head.in_features, len(keep_indices), bias=True)
                new_head.weight.data = export_model.head.weight.data[_keep]
                new_head.bias.data   = export_model.head.bias.data[_keep]
                export_model.head = new_head
                num_tags = len(keep_indices)
                print(f"[SigLIP2Manager] strip_unknown_tags: removed {removed} Unknown heads → {num_tags} tags remain")
                # Build filtered vocabulary dict for output
                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
                new_t2i = {tag: old_to_new[idx] for tag, idx in tag_to_idx.items() if idx in old_to_new}
                new_i2t = {str(v): k for k, v in new_t2i.items()}
                new_t2c = {tag: tag_to_cat.get(tag, "General") for tag in new_t2i}
                new_cats: dict = {}
                for tag, cat in new_t2c.items():
                    new_cats.setdefault(cat, []).append(tag)
                export_vocab_data = {
                    "tag_to_idx": new_t2i,
                    "idx_to_tag": new_i2t,
                    "tag_to_category": new_t2c,
                    "num_tags": num_tags,
                    "categories": new_cats,
                }
            else:
                print("[SigLIP2Manager] strip_unknown_tags: no Unknown tags found, skipping")

        # Build a dummy input using the processor on a tiny image
        dummy_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        _proc_kw = {"images": dummy_img, "return_tensors": "pt"}
        if self.is_naflex:
            _proc_kw["max_num_patches"] = max_num_patches
        inputs = self.processor(**_proc_kw)

        # Export to a temp subdirectory so the scattered files don't pollute the
        # final output directory.  After consolidation the temp dir is removed.
        _out_dir  = os.path.dirname(output_path) or "."
        _out_name = os.path.basename(output_path)
        _tmp_dir  = tempfile.mkdtemp(dir=_out_dir, prefix=".onnx_tmp_")
        _tmp_path = os.path.join(_tmp_dir, _out_name)
        try:
            with torch.no_grad():
                if self.is_naflex:
                    dummy_pv  = inputs["pixel_values"].float()
                    dummy_pam = inputs["pixel_attention_mask"].float()
                    dummy_ss  = inputs["spatial_shapes"]
                    torch.onnx.export(
                        export_model,
                        (dummy_pv, dummy_pam, dummy_ss),
                        _tmp_path,
                        input_names=["pixel_values", "pixel_attention_mask", "spatial_shapes"],
                        output_names=["logits"],
                        dynamic_axes={
                            "pixel_values":         {0: "batch_size", 1: "num_patches"},
                            "pixel_attention_mask": {0: "batch_size", 1: "num_patches"},
                            "spatial_shapes":       {0: "batch_size"},
                        },
                        opset_version=18,
                        do_constant_folding=True,
                        dynamo=False,
                    )
                else:
                    dummy_pv = inputs["pixel_values"].float()
                    torch.onnx.export(
                        export_model,
                        (dummy_pv,),
                        _tmp_path,
                        input_names=["pixel_values"],
                        output_names=["logits"],
                        dynamic_axes={"pixel_values": {0: "batch_size"}},
                        opset_version=18,
                        do_constant_folding=True,
                        dynamo=False,
                    )

            # Consolidate into a single .onnx + .onnx.data pair in the final location.
            import onnx as _onnx_lib
            import onnx.shape_inference as _onnx_si
            _data_file = _out_name + ".data"
            _loaded = _onnx_lib.load(_tmp_path, load_external_data=True)
            _onnx_lib.save_model(
                _loaded,
                output_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=_data_file,
                convert_attribute=True,
            )
            print(f"[SigLIP2Manager] External data consolidated → {_data_file}")
            # Propagate shape info through If-nodes so TensorRT can infer tensor ranks.
            # Uses infer_shapes_path to operate on files directly, avoiding the
            # in-memory 2GB protobuf limit that causes save_model to produce empty output.
            _onnx_si.infer_shapes_path(output_path, output_path,
                                        check_type=True, strict_mode=False)
            print(f"[SigLIP2Manager] Shape inference applied (TensorRT If-node fix)")
        finally:
            shutil.rmtree(_tmp_dir, ignore_errors=True)

        # Restore model device
        export_model.to(self.device)

        # Save vocabulary alongside the ONNX file
        vocab_out = os.path.splitext(output_path)[0] + "_vocabulary.json"
        if export_vocab_data is not None:
            with open(vocab_out, "w", encoding="utf-8") as fh:
                json.dump(export_vocab_data, fh, ensure_ascii=False, indent=2)
        else:
            with open(self.vocab_path, "r", encoding="utf-8") as fh:
                raw_vocab = fh.read()
            with open(vocab_out, "w", encoding="utf-8") as fh:
                fh.write(raw_vocab)

        # Write _metadata.json alongside the ONNX so inference (and spaces app) can
        # load the correct processor without hardcoding the repo ID.
        from core.tagger.siglip2_tagger_model import SIGLIP2_DEFAULT_REPO_ID as _DEFAULT_REPO_ID
        _src_meta = _read_metadata(self.checkpoint_path) if not self.checkpoint_path.endswith(".onnx") else {}
        _onnx_meta = {
            "vision_encoder_repo": _src_meta.get("vision_encoder_repo", _DEFAULT_REPO_ID),
            "is_naflex": self.is_naflex,
            "num_tags": num_tags,
        }
        _onnx_meta_path = os.path.splitext(output_path)[0] + "_metadata.json"
        with open(_onnx_meta_path, "w", encoding="utf-8") as _fh:
            json.dump(_onnx_meta, _fh, ensure_ascii=False, indent=2)

        print(f"[SigLIP2Manager] ONNX exported → {output_path}")
        print(f"[SigLIP2Manager] Vocabulary  → {vocab_out}")
        print(f"[SigLIP2Manager] Metadata    → {_onnx_meta_path}")
        return output_path, vocab_out

    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Unload the current model and free VRAM."""
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self.onnx_session is not None:
            del self.onnx_session
            self.onnx_session = None
        self.processor           = None
        self.vocabulary          = None
        self.checkpoint_path     = ""
        self.vision_encoder_path = ""
        self.vocab_path          = ""
        self.model_type          = ""
        self.logit_bias          = None
        self.lr_matrix           = None
        if self.ood_emb_session is not None:
            del self.ood_emb_session
            self.ood_emb_session = None
        self.ood_ref = None
        if self._cls_emb_hook_handle is not None:
            self._cls_emb_hook_handle.remove()
            self._cls_emb_hook_handle = None
        self._last_cls_emb = None
        print("[SigLIP2Manager] Model unloaded.")

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # OOD detection helpers
    # ------------------------------------------------------------------

    def _ensure_ood_session(self) -> None:
        """Create (or reuse) an ORT session that outputs logits + CLS embedding.

        Modifies a copy of the loaded ONNX model to expose the intermediate
        CLS-embedding node as a graph output, then creates an ORT session from
        the modified model bytes (no file written).
        """
        if self.ood_emb_session is not None:
            return
        if not (self.checkpoint_path and self.checkpoint_path.endswith(".onnx")):
            raise RuntimeError("OOD session requires an ONNX model.")

        import onnx
        import onnxruntime as ort

        # Load ONNX without external data (weights referenced via .data file)
        model_proto = onnx.load(self.checkpoint_path, load_external_data=False)

        # Add the CLS embedding node as a graph output
        emb_type = onnx.helper.make_tensor_value_info(
            _OOD_EMB_NODE, onnx.TensorProto.FLOAT, None
        )
        model_proto.graph.output.append(emb_type)

        # Serialize to bytes (external data stays referenced via relative path)
        model_bytes = model_proto.SerializeToString()

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.ood_emb_session = ort.InferenceSession(
            model_bytes,
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def _load_ood_reference(self, path: str) -> None:
        """Load OOD reference distribution from npz file."""
        import numpy as np
        data = np.load(path)
        self.ood_ref = {
            "mu":      data["mu"].astype(np.float64),
            "cov_inv": data["cov_inv"].astype(np.float64),
            "p50":     float(data["p50"]),
            "p95":     float(data["p95"]) if "p95" in data else 0.0,
        }

    def _compute_mahalanobis(self, emb: "np.ndarray") -> float:
        """Compute Mahalanobis distance between emb and the in-dist reference."""
        import numpy as np
        diff = emb.astype(np.float64) - self.ood_ref["mu"]
        return float(np.sqrt(max(0.0, diff @ self.ood_ref["cov_inv"] @ diff)))

    def build_ood_reference(
        self,
        image_paths: List[str],
        save_path: Optional[str] = None,
        max_images: int = 2000,
    ) -> Dict[str, Any]:
        """Build OOD reference distribution from in-distribution images.

        Collects CLS embeddings, fits a multivariate Gaussian (full covariance
        with shrinkage regularisation), and saves to *save_path* (defaults to
        ``{onnx_base}_ood_ref.npz``).

        Returns summary statistics dict.
        """
        import numpy as np

        if not (self.checkpoint_path and self.checkpoint_path.endswith(".onnx")):
            raise RuntimeError("build_ood_reference requires an ONNX model.")

        self._ensure_ood_session()

        if save_path is None:
            save_path = os.path.splitext(self.checkpoint_path)[0] + "_ood_ref.npz"

        # Sample image paths
        import random
        rng = random.Random(42)
        paths = list(image_paths)
        rng.shuffle(paths)
        paths = paths[:max_images]

        print(f"[SigLIP2Manager] Building OOD reference from {len(paths)} images...")

        IMG_SIZE = 384
        MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        embeddings = []
        errors = 0
        for i, p in enumerate(paths):
            if i % 200 == 0:
                print(f"  [{i}/{len(paths)}]")
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = (arr - MEAN) / STD
                pv_np = arr.transpose(2, 0, 1)[None]  # (1,3,H,W)
                out = self.ood_emb_session.run(
                    ["logits", _OOD_EMB_NODE],
                    {"pixel_values": pv_np},
                )
                embeddings.append(out[1][0].astype(np.float64))
            except Exception as _e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR {p}: {_e}")

        if len(embeddings) < 10:
            raise RuntimeError(f"Too few successful embeddings ({len(embeddings)}); cannot fit distribution.")

        E = np.stack(embeddings, axis=0)  # (N, D)
        mu = E.mean(axis=0)

        # Full covariance with Ledoit-Wolf shrinkage
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(assume_centered=False)
            lw.fit(E)
            cov_inv = np.linalg.inv(lw.covariance_)
        except Exception:
            # Fallback: diagonal regularisation
            cov = np.cov(E, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(cov)

        # Compute distances on the training set to get percentiles
        dists = np.array([
            float(np.sqrt(max(0.0, (e - mu) @ cov_inv @ (e - mu))))
            for e in embeddings
        ])
        p50 = float(np.percentile(dists, 50))
        p95 = float(np.percentile(dists, 95))

        np.savez_compressed(
            save_path,
            mu=mu.astype(np.float32),
            cov_inv=cov_inv.astype(np.float32),
            p50=np.float32(p50),
            p95=np.float32(p95),
        )
        self._load_ood_reference(save_path)

        print(
            f"[SigLIP2Manager] OOD reference saved → {save_path} "
            f"| n={len(embeddings)} | p50={p50:.2f} | p95={p95:.2f}"
        )
        return {
            "n_images":  len(embeddings),
            "n_errors":  errors,
            "p50":       p50,
            "p95":       p95,
            "save_path": save_path,
        }

    def get_tag_metrics_path(self) -> Optional[str]:
        """Return _tag_metrics.npz path for the loaded checkpoint, or None if absent."""
        if not self.checkpoint_path:
            return None
        base = os.path.splitext(self.checkpoint_path)[0]
        p = base + "_tag_metrics.npz"
        return p if os.path.isfile(p) else None

    def recompute_calibration_table(
        self,
        method: str = "jeffreys",
        eps: float = 0.5,
        prior_strength: float = 10.0,
    ) -> bool:
        """Recompute and replace the in-memory calibration_table with new settings.

        Reads ``pos_hist`` and ``total_hist`` from the loaded tag_metrics dict and
        applies the requested formula.  Returns True on success, False if tag_metrics
        or required arrays are not loaded.
        """
        if self.tag_metrics is None:
            return False
        pos_h   = self.tag_metrics.get("pos_hist")
        total_h = self.tag_metrics.get("total_hist")
        if pos_h is None or total_h is None:
            return False

        pos_f   = pos_h.astype(np.float32)
        total_f = total_h.astype(np.float32)

        n_pos_tag   = pos_f.sum(axis=1, keepdims=True)
        n_total_tag = total_f.sum(axis=1, keepdims=True)
        pi = np.where(n_total_tag > 0, n_pos_tag / n_total_tag, 0.0)

        if method == "beta_bb":
            alpha = pi * prior_strength
            beta  = (1.0 - pi) * prior_strength
            calib = (pos_f + alpha) / (total_f + alpha + beta)
        else:  # "jeffreys"
            calib = (pos_f + eps) / (total_f + 2.0 * eps)
            calib = np.where(total_f > 0, calib, pi)

        self.tag_metrics["calibration_table"] = calib.astype(np.float16)
        self.calib_method      = method
        self.calib_eps         = eps
        self.calib_prior_strength = prior_strength
        return True

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "loaded":            self.model is not None or self.onnx_session is not None,
            "checkpoint_path":   self.checkpoint_path,
            "vocab_path":        self.vocab_path,
            "model_type":        self.model_type,
            "num_tags":          len(self.vocabulary["idx_to_tag"]) if self.vocabulary else 0,
            "lr_matrix_loaded":  self.lr_matrix is not None,
            "has_tag_metrics":   self.get_tag_metrics_path() is not None,
            "has_ood_reference": self.ood_ref is not None,
            "calib_method":      self.calib_method,
            "calib_eps":         self.calib_eps,
            "calib_prior_strength": self.calib_prior_strength,
        }
