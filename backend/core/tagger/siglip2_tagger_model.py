"""
SigLIP2 Tagger Model.

Architecture:
    SigLIP2 Vision Encoder (frozen or LoRA) → pooler_output [B, 1152] → Linear(1152, num_tags)

Two variants:
    SigLIP2TaggerModel     : full-parameter training (vision encoder fully trainable or frozen)
    SigLIP2TaggerLoRAModel : LoRA adapters on attention layers, head always trainable
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


# ------------------------------------------------------------------
# LoRA primitives
# ------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA adapters.

    Forward: W·x + (B·A·x) * scale
    where scale = alpha / rank
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 32,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank

        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(in_f, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze base weights
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scale
        return base_out + lora_out

    def merge_into_base(self) -> None:
        """Merge LoRA weights into base weight (for export)."""
        with torch.no_grad():
            delta = (self.lora_A @ self.lora_B).T * self.scale
            self.base.weight.data += delta


# ------------------------------------------------------------------
# Vision encoder loader
# ------------------------------------------------------------------

def _load_vision_encoder(safetensors_path: str) -> nn.Module:
    """Load SigLIP2 so400m vision encoder from safetensors file."""
    from transformers import AutoModel
    import tempfile, shutil

    # We need the full model just to get architecture, then replace state_dict
    # Use HF cached model if available, else load from hub temporarily
    REPO_ID = "google/siglip2-so400m-patch16-naflex"
    full_model = AutoModel.from_pretrained(REPO_ID, dtype=torch.float32)
    vision_encoder = full_model.vision_model

    # Load our fine-tuned / custom weights
    state_dict = load_file(safetensors_path)
    vision_encoder.load_state_dict(state_dict, strict=True)

    return vision_encoder


# ------------------------------------------------------------------
# Full-parameter model
# ------------------------------------------------------------------

class SigLIP2TaggerModel(nn.Module):
    """SigLIP2 vision encoder + classification head.

    Parameters
    ----------
    num_tags          : number of output classes
    vision_encoder    : pre-loaded vision encoder nn.Module
    freeze_encoder    : if True, gradients do not flow through vision encoder
    hidden_size       : vision encoder output dimension (1152 for so400m)
    """

    HIDDEN_SIZE = 1152  # so400m

    def __init__(
        self,
        num_tags: int,
        vision_encoder: nn.Module,
        freeze_encoder: bool = False,
        hidden_size: int = HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.head = nn.Linear(hidden_size, num_tags)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        if freeze_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits [B, num_tags]."""
        out = self.vision_encoder(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        pooled = out.pooler_output  # [B, 1152]
        return self.head(pooled)    # [B, num_tags]

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, output_dir: str, name: str, metadata: Optional[dict] = None) -> str:
        """Save model weights and metadata JSON.

        Returns path to saved safetensors file.
        """
        os.makedirs(output_dir, exist_ok=True)
        path_st  = os.path.join(output_dir, f"{name}.safetensors")
        path_meta = os.path.join(output_dir, f"{name}_metadata.json")

        sd = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(sd, path_st)

        if metadata:
            with open(path_meta, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        return path_st

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        vision_encoder_path: str,
        num_tags: Optional[int] = None,
    ) -> "SigLIP2TaggerModel":
        """Load model from checkpoint safetensors."""
        meta_path = checkpoint_path.replace(".safetensors", "_metadata.json")
        metadata: dict = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        if num_tags is None:
            num_tags = metadata.get("num_tags")
            if num_tags is None:
                raise ValueError("num_tags must be provided or present in metadata")

        vision_encoder = _load_vision_encoder(vision_encoder_path)
        model = cls(num_tags=num_tags, vision_encoder=vision_encoder)
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)
        return model

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ------------------------------------------------------------------
# LoRA model
# ------------------------------------------------------------------

class SigLIP2TaggerLoRAModel(nn.Module):
    """SigLIP2 vision encoder with LoRA adapters + classification head.

    LoRA is applied to attention projection layers in the vision encoder.
    The head is always fully trainable.
    The base encoder weights (non-LoRA) are frozen.
    """

    # Regex patterns for target modules in SigLIP2 vision encoder
    LORA_TARGET_PATTERNS: List[str] = [
        r"encoder\.layers\.\d+\.self_attn\.q_proj$",
        r"encoder\.layers\.\d+\.self_attn\.k_proj$",
        r"encoder\.layers\.\d+\.self_attn\.v_proj$",
        r"encoder\.layers\.\d+\.self_attn\.out_proj$",
    ]

    HIDDEN_SIZE = 1152

    def __init__(
        self,
        num_tags: int,
        vision_encoder: nn.Module,
        lora_rank: int = 32,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        hidden_size: int = HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.head = nn.Linear(hidden_size, num_tags)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Freeze all encoder parameters first
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # Replace target Linear layers with LoRALinear
        self._lora_modules: Dict[str, LoRALinear] = {}
        self._inject_lora(lora_rank, lora_alpha, lora_dropout)

    def _inject_lora(self, rank: int, alpha: float, dropout: float) -> None:
        patterns = [re.compile(p) for p in self.LORA_TARGET_PATTERNS]

        for name, module in list(self.vision_encoder.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(p.match(name) for p in patterns):
                continue

            # Navigate to parent module and replace child
            parts = name.split(".")
            parent = self.vision_encoder
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]

            lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, lora_linear)
            self._lora_modules[name] = lora_linear

        print(f"[TaggerLoRA] Injected LoRA into {len(self._lora_modules)} modules")

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        out = self.vision_encoder(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        pooled = out.pooler_output  # [B, 1152]
        return self.head(pooled)    # [B, num_tags]

    # ------------------------------------------------------------------
    # Save / load (saves only LoRA + head, not full encoder)
    # ------------------------------------------------------------------

    def save_checkpoint(self, output_dir: str, name: str, metadata: Optional[dict] = None) -> str:
        """Save LoRA weights + head weights only (compact checkpoint)."""
        os.makedirs(output_dir, exist_ok=True)
        path_st   = os.path.join(output_dir, f"{name}.safetensors")
        path_meta = os.path.join(output_dir, f"{name}_metadata.json")

        sd: Dict[str, torch.Tensor] = {}
        # LoRA parameters (prefixed with "lora.")
        for module_name, lora_module in self._lora_modules.items():
            prefix = f"lora.{module_name}"
            sd[f"{prefix}.lora_A"] = lora_module.lora_A.detach().contiguous()
            sd[f"{prefix}.lora_B"] = lora_module.lora_B.detach().contiguous()
        # Head parameters
        sd["head.weight"] = self.head.weight.detach().contiguous()
        sd["head.bias"]   = self.head.bias.detach().contiguous()

        if metadata is None:
            metadata = {}
        metadata.update({
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "num_lora_modules": len(self._lora_modules),
        })

        save_file(sd, path_st)
        with open(path_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return path_st

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        vision_encoder_path: str,
        num_tags: Optional[int] = None,
        lora_rank: int = 32,
        lora_alpha: float = 16.0,
    ) -> "SigLIP2TaggerLoRAModel":
        meta_path = checkpoint_path.replace(".safetensors", "_metadata.json")
        metadata: dict = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        if num_tags is None:
            num_tags = metadata.get("num_tags")
            if num_tags is None:
                raise ValueError("num_tags must be provided or present in metadata")

        lora_rank  = metadata.get("lora_rank", lora_rank)
        lora_alpha = metadata.get("lora_alpha", lora_alpha)

        vision_encoder = _load_vision_encoder(vision_encoder_path)
        model = cls(
            num_tags=num_tags,
            vision_encoder=vision_encoder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        saved = load_file(checkpoint_path)

        # Restore head
        model.head.weight.data.copy_(saved["head.weight"])
        model.head.bias.data.copy_(saved["head.bias"])

        # Restore LoRA weights
        for module_name, lora_module in model._lora_modules.items():
            prefix = f"lora.{module_name}"
            key_A, key_B = f"{prefix}.lora_A", f"{prefix}.lora_B"
            if key_A in saved:
                lora_module.lora_A.data.copy_(saved[key_A])
            if key_B in saved:
                lora_module.lora_B.data.copy_(saved[key_B])

        return model

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_count(self) -> Dict[str, int]:
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_only = sum(
            lm.lora_A.numel() + lm.lora_B.numel()
            for lm in self._lora_modules.values()
        )
        return {"total": total, "trainable": trainable, "lora": lora_only}


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def build_tagger_model(
    training_method: str,
    num_tags: int,
    vision_encoder_path: str,
    lora_rank: int = 32,
    lora_alpha: float = 16.0,
    freeze_encoder: bool = False,
) -> nn.Module:
    """Build the appropriate tagger model.

    Parameters
    ----------
    training_method : "full" | "lora"
    num_tags        : number of output tag classes
    vision_encoder_path : path to siglip2_so400m_vision_encoder.safetensors
    lora_rank       : LoRA rank (only used when training_method="lora")
    lora_alpha      : LoRA alpha
    freeze_encoder  : freeze encoder entirely (only used when training_method="full")
    """
    print(f"[TaggerModel] Loading vision encoder from: {vision_encoder_path}")
    vision_encoder = _load_vision_encoder(vision_encoder_path)

    if training_method == "lora":
        return SigLIP2TaggerLoRAModel(
            num_tags=num_tags,
            vision_encoder=vision_encoder,
            lora_rank=lora_rank,
            lora_alpha=float(lora_alpha),
        )
    elif training_method == "full":
        return SigLIP2TaggerModel(
            num_tags=num_tags,
            vision_encoder=vision_encoder,
            freeze_encoder=freeze_encoder,
        )
    else:
        raise ValueError(f"Unknown training_method: {training_method!r}. Use 'full' or 'lora'.")
