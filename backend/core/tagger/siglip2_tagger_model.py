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

    # Strip surrounding quotes that may come from user input
    safetensors_path = safetensors_path.strip().strip('"').strip("'")

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
# Custom Attention Pooling (for Full FT with configurable output dim)
# ------------------------------------------------------------------

class CustomAttentionPooling(nn.Module):
    """Replace SigLIP2's built-in pooler with a learnable attention pooling.

    Takes last_hidden_state [B, N, in_dim] and produces [B, out_dim].
    A single learnable query attends over all patch tokens.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        # Adjust num_heads so out_dim is divisible
        while out_dim % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.query  = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.normal_(self.query, std=0.02)
        self.proj_k = nn.Linear(in_dim, out_dim)
        self.proj_v = nn.Linear(in_dim, out_dim)
        self.attn   = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, in_dim] → [B, out_dim]"""
        k = self.proj_k(x)                           # [B, N, out_dim]
        v = self.proj_v(x)
        q = self.query.expand(x.size(0), -1, -1)     # [B, 1, out_dim]
        out, _ = self.attn(q, k, v)
        return out.squeeze(1)                         # [B, out_dim]


def _build_head(pool_dim: int, num_tags: int, head_hidden_dim: Optional[int]) -> nn.Module:
    """Build classification head: 1-layer Linear or 2-layer MLP."""
    if head_hidden_dim:
        head = nn.Sequential(
            nn.Linear(pool_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, num_tags),
        )
        nn.init.zeros_(head[0].weight)
        nn.init.zeros_(head[0].bias)
        nn.init.zeros_(head[2].weight)
        nn.init.zeros_(head[2].bias)
    else:
        head = nn.Linear(pool_dim, num_tags)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    return head


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
    cls_dim           : if set, use CustomAttentionPooling(1152 → cls_dim)
                        instead of built-in pooler_output; Full FT only
    head_hidden_dim   : if set, insert a hidden layer in the classification
                        head: Linear(pool_dim → head_hidden_dim) → GELU
                        → Linear(head_hidden_dim → num_tags)
    """

    HIDDEN_SIZE = 1152  # so400m

    def __init__(
        self,
        num_tags: int,
        vision_encoder: nn.Module,
        freeze_encoder: bool = False,
        hidden_size: int = HIDDEN_SIZE,
        cls_dim: Optional[int] = None,
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.cls_dim        = cls_dim
        self.head_hidden_dim = head_hidden_dim

        # Custom attention pooler (replaces built-in pooler_output)
        if cls_dim:
            self.custom_pooler: Optional[CustomAttentionPooling] = CustomAttentionPooling(
                in_dim=hidden_size, out_dim=cls_dim
            )
            pool_dim = cls_dim
        else:
            self.custom_pooler = None
            pool_dim = hidden_size

        self.head = _build_head(pool_dim, num_tags, head_hidden_dim)

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
        if self.custom_pooler is not None:
            pooled = self.custom_pooler(out.last_hidden_state)  # [B, cls_dim]
        else:
            pooled = out.pooler_output  # [B, 1152]
        return self.head(pooled)        # [B, num_tags]

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

        cls_dim        = metadata.get("cls_dim")
        head_hidden_dim = metadata.get("head_hidden_dim")
        vision_encoder = _load_vision_encoder(vision_encoder_path)
        model = cls(
            num_tags=num_tags,
            vision_encoder=vision_encoder,
            cls_dim=cls_dim,
            head_hidden_dim=head_hidden_dim,
        )
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
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.head_hidden_dim = head_hidden_dim
        self.head = _build_head(hidden_size, num_tags, head_hidden_dim)
        # Zero-init all head weights for stable start
        for p in self.head.parameters():
            nn.init.zeros_(p)

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
        # Head parameters (generic: works for Linear or MLP Sequential)
        for key, tensor in self.head.state_dict().items():
            sd[f"head.{key}"] = tensor.detach().contiguous()

        if metadata is None:
            metadata = {}
        metadata.update({
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "num_lora_modules": len(self._lora_modules),
            "head_hidden_dim": self.head_hidden_dim,
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

        lora_rank       = metadata.get("lora_rank", lora_rank)
        lora_alpha      = metadata.get("lora_alpha", lora_alpha)
        head_hidden_dim = metadata.get("head_hidden_dim", None)

        vision_encoder = _load_vision_encoder(vision_encoder_path)
        model = cls(
            num_tags=num_tags,
            vision_encoder=vision_encoder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            head_hidden_dim=head_hidden_dim,
        )

        saved = load_file(checkpoint_path)

        # Restore head (generic state_dict restore)
        head_sd = {k[len("head."):]: v for k, v in saved.items() if k.startswith("head.")}
        model.head.load_state_dict(head_sd)

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
    cls_dim: Optional[int] = None,
    head_hidden_dim: Optional[int] = None,
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
    cls_dim         : custom attention pooling output dim (Full FT only)
    head_hidden_dim : hidden layer dim in classification MLP head (both modes)
    """
    print(f"[TaggerModel] Loading vision encoder from: {vision_encoder_path}")
    vision_encoder = _load_vision_encoder(vision_encoder_path)

    if training_method == "lora":
        if cls_dim:
            print("[TaggerModel] Warning: cls_dim is ignored for LoRA training (no custom pooler)")
        return SigLIP2TaggerLoRAModel(
            num_tags=num_tags,
            vision_encoder=vision_encoder,
            lora_rank=lora_rank,
            lora_alpha=float(lora_alpha),
            head_hidden_dim=head_hidden_dim,
        )
    elif training_method == "full":
        return SigLIP2TaggerModel(
            num_tags=num_tags,
            vision_encoder=vision_encoder,
            freeze_encoder=freeze_encoder,
            cls_dim=cls_dim,
            head_hidden_dim=head_hidden_dim,
        )
    else:
        raise ValueError(f"Unknown training_method: {training_method!r}. Use 'full' or 'lora'.")
