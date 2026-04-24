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
        self.vocabulary          = {
            "idx_to_tag":      idx_to_tag,
            "tag_to_category": tag_to_category,
        }
        self.checkpoint_path     = checkpoint_path
        self.vision_encoder_path = vision_encoder_path
        self.vocab_path          = vocab_path
        self.model_type          = model_type

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
        threshold: float = 0.35,
        max_num_patches: int = 256,
    ) -> Dict[str, Any]:
        """Run inference on raw image bytes.

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

        if self.model_type == "onnx":
            pv_np = inputs["pixel_values"].float().numpy()
            if self.is_naflex:
                pam_np = inputs["pixel_attention_mask"].float().numpy()
                ss_np  = inputs["spatial_shapes"].numpy().astype(np.int64)
                outputs = self.onnx_session.run(
                    ["logits"],
                    {"pixel_values": pv_np, "pixel_attention_mask": pam_np, "spatial_shapes": ss_np},
                )
            else:
                outputs = self.onnx_session.run(["logits"], {"pixel_values": pv_np})
            logits_np = outputs[0][0]  # [num_tags]
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
            probs = torch.sigmoid(logits[0]).cpu().numpy()  # [num_tags]

        idx_to_tag      = self.vocabulary["idx_to_tag"]
        tag_to_category = self.vocabulary["tag_to_category"]

        # Build full list
        all_items: List[Dict] = []
        for i, prob in enumerate(probs):
            tag      = idx_to_tag.get(i, f"__unk_{i}__")
            category = tag_to_category.get(tag, "Unknown")
            all_items.append({"tag": tag, "prob": float(prob), "category": category})

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
        filtered = [
            it for it in all_items
            if it["prob"] >= threshold
            and it["category"] not in ("Quality", "Rating")
        ]
        filtered.sort(key=lambda x: x["prob"], reverse=True)

        return {
            "tags":          filtered,
            "quality_top":   quality_top,
            "rating_top":    rating_top,
            "num_predicted": len(filtered),
        }

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
        print(f"[SigLIP2Manager] Merged LoRA checkpoint saved → {saved}")
        return saved

    # ------------------------------------------------------------------

    def export_onnx(
        self,
        output_path: str,
        max_num_patches: int = 256,
    ) -> Tuple[str, str]:
        """Export the model to ONNX format.

        If the model is a LoRA model, merges weights first using a temporary
        SigLIP2TaggerModel.

        Returns (onnx_path, vocab_path).
        """
        if self.model is None and self.onnx_session is None:
            raise RuntimeError("No model loaded.")
        if self.model_type == "onnx":
            raise ValueError("Cannot export an ONNX model to ONNX. Load a safetensors checkpoint instead.")

        from core.tagger.siglip2_tagger_model import (
            SigLIP2TaggerLoRAModel,
            SigLIP2TaggerModel,
            _load_vision_encoder,
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

        # If LoRA: merge into a temporary full model for export
        if isinstance(self.model, SigLIP2TaggerLoRAModel):
            tmp_dir  = tempfile.mkdtemp(prefix="siglip2_onnx_")
            tmp_name = "merged_for_export"
            try:
                merged_path = self.model.save_merged_checkpoint(tmp_dir, tmp_name)
                vision_enc  = _load_vision_encoder(self.vision_encoder_path)
                num_tags    = self.model.head.out_features
                export_model = SigLIP2TaggerModel(
                    num_tags=num_tags,
                    vision_encoder=vision_enc,
                )
                from safetensors.torch import load_file as _load_file
                export_model.load_state_dict(_load_file(merged_path), strict=False)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            export_model = self.model
            num_tags     = self.model.head.out_features

        export_model.eval().to("cpu")

        # Build a dummy input using the processor on a tiny image
        dummy_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        _proc_kw = {"images": dummy_img, "return_tensors": "pt"}
        if self.is_naflex:
            _proc_kw["max_num_patches"] = max_num_patches
        inputs = self.processor(**_proc_kw)

        with torch.no_grad():
            if self.is_naflex:
                dummy_pv  = inputs["pixel_values"].float()
                dummy_pam = inputs["pixel_attention_mask"].float()
                dummy_ss  = inputs["spatial_shapes"]
                torch.onnx.export(
                    export_model,
                    (dummy_pv, dummy_pam, dummy_ss),
                    output_path,
                    input_names=["pixel_values", "pixel_attention_mask", "spatial_shapes"],
                    output_names=["logits"],
                    dynamic_axes={
                        "pixel_values":         {0: "batch_size", 1: "num_patches"},
                        "pixel_attention_mask": {0: "batch_size", 1: "num_patches"},
                        "spatial_shapes":       {0: "batch_size"},
                    },
                    opset_version=18,
                    do_constant_folding=True,
                )
            else:
                dummy_pv = inputs["pixel_values"].float()
                torch.onnx.export(
                    export_model,
                    (dummy_pv,),
                    output_path,
                    input_names=["pixel_values"],
                    output_names=["logits"],
                    dynamic_axes={"pixel_values": {0: "batch_size"}},
                    opset_version=18,
                    do_constant_folding=True,
                )

        # Restore model device
        export_model.to(self.device)

        # Save vocabulary alongside the ONNX file
        vocab_out = os.path.splitext(output_path)[0] + "_vocabulary.json"
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
        print("[SigLIP2Manager] Model unloaded.")

    # ------------------------------------------------------------------

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "loaded":           self.model is not None or self.onnx_session is not None,
            "checkpoint_path":  self.checkpoint_path,
            "vocab_path":       self.vocab_path,
            "model_type":       self.model_type,
            "num_tags":         len(self.vocabulary["idx_to_tag"]) if self.vocabulary else 0,
        }
