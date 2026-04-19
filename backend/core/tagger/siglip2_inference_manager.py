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
        self.processor = None
        self.vocabulary: Optional[Dict[str, Any]] = None   # from vocabulary.json
        self.checkpoint_path: str = ""
        self.vision_encoder_path: str = ""
        self.vocab_path: str = ""
        self.model_type: str = ""   # "full" | "lora"
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

        # 1. Detect model type
        model_type = _detect_model_type(checkpoint_path)
        meta = _read_metadata(checkpoint_path)

        # Override lora_rank / lora_alpha from metadata if present
        lora_rank  = int(meta.get("lora_rank",  lora_rank))
        lora_alpha = float(meta.get("lora_alpha", lora_alpha))

        # 2. Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as fh:
            vocab = json.load(fh)
        idx_to_tag       = {int(k): v for k, v in vocab["idx_to_tag"].items()}
        tag_to_category  = vocab.get("tag_to_category", {})
        num_tags         = len(idx_to_tag)

        # 3. Load model
        from core.tagger.siglip2_tagger_model import (
            SigLIP2TaggerModel,
            SigLIP2TaggerLoRAModel,
        )

        if model_type == "lora":
            if not vision_encoder_path:
                raise ValueError(
                    "vision_encoder_path is required for LoRA checkpoints."
                )
            model = SigLIP2TaggerLoRAModel.load_checkpoint(
                checkpoint_path=checkpoint_path,
                vision_encoder_path=vision_encoder_path,
                num_tags=num_tags,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
        else:
            # vision_encoder_path is optional for merged (full) checkpoints —
            # the checkpoint already contains all vision encoder weights.
            model = SigLIP2TaggerModel.load_checkpoint(
                checkpoint_path=checkpoint_path,
                vision_encoder_path=vision_encoder_path,
                num_tags=num_tags,
            )

        model.eval()
        model.to(self.device)

        # 4. Load processor
        from transformers import AutoProcessor
        REPO_ID = "google/siglip2-so400m-patch16-naflex"
        try:
            processor = AutoProcessor.from_pretrained(REPO_ID, local_files_only=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(REPO_ID)

        # 5. Store state
        self.model               = model
        self.processor           = processor
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
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(
            images=pil_image,
            return_tensors="pt",
            max_num_patches=max_num_patches,
        )
        pixel_values    = inputs["pixel_values"].to(self.device)
        pixel_attn_mask = inputs["pixel_attention_mask"].to(self.device)
        spatial_shapes  = inputs["spatial_shapes"].to(self.device)

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
        if self.model is None:
            raise RuntimeError("No model loaded.")
        from core.tagger.siglip2_tagger_model import SigLIP2TaggerLoRAModel
        if not isinstance(self.model, SigLIP2TaggerLoRAModel):
            raise ValueError("merge_lora_and_save is only valid for LoRA models.")

        # F4: treat output_path as a directory; always save as model.safetensors
        output_dir  = output_path.strip().strip('"').strip("'")
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
        if self.model is None:
            raise RuntimeError("No model loaded.")

        from core.tagger.siglip2_tagger_model import (
            SigLIP2TaggerLoRAModel,
            SigLIP2TaggerModel,
            _load_vision_encoder,
        )

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
        inputs    = self.processor(
            images=dummy_img,
            return_tensors="pt",
            max_num_patches=max_num_patches,
        )
        dummy_pv  = inputs["pixel_values"].float()
        dummy_pam = inputs["pixel_attention_mask"].float()
        dummy_ss  = inputs["spatial_shapes"]

        with torch.no_grad():
            torch.onnx.export(
                export_model,
                (dummy_pv, dummy_pam, dummy_ss),
                output_path,
                input_names=["pixel_values", "pixel_attention_mask", "spatial_shapes"],
                output_names=["logits"],
                dynamic_axes={
                    "pixel_values":         {1: "num_patches"},
                    "pixel_attention_mask": {1: "num_patches"},
                    "spatial_shapes":       {0: "num_sequences"},
                },
                opset_version=17,
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

        print(f"[SigLIP2Manager] ONNX exported → {output_path}")
        print(f"[SigLIP2Manager] Vocabulary  → {vocab_out}")
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
            "loaded":           self.model is not None,
            "checkpoint_path":  self.checkpoint_path,
            "vocab_path":       self.vocab_path,
            "model_type":       self.model_type,
            "num_tags":         len(self.vocabulary["idx_to_tag"]) if self.vocabulary else 0,
        }
