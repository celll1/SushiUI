"""
SigLIP2 Tagger Model.

Architecture options (Full FT):

  Default (cls_dim=None):
    pooler_output [B, 1152] -> Linear(1152, num_tags)

  Custom Attention Pooling (cls_dim set, hidden_proj_dim=None):
    last_hidden_state [B, N, 1152]
        -> proj_k/proj_v: Linear(1152 -> cls_dim)
        -> MHA(Q=[B,1,cls_dim], K/V=[B,N,cls_dim])
        -> [B, cls_dim]
        -> Linear(cls_dim, num_tags)

  Custom Attention Pooling + hidden projection (cls_dim and hidden_proj_dim set):
    last_hidden_state [B, N, 1152]
        -> proj_k/proj_v: Linear(1152 -> hidden_proj_dim)
        -> [B, N, hidden_proj_dim]  <- extractable via get_token_features() for SDXL
        -> MHA(Q=[B,1,cls_dim], K/V=[B,N,hidden_proj_dim]; kdim=vdim=hidden_proj_dim)
        -> [B, cls_dim]
        -> Linear(cls_dim, num_tags)

Two variants:
    SigLIP2TaggerModel     : full-parameter training (vision encoder fully trainable or frozen)
    SigLIP2TaggerLoRAModel : LoRA adapters on attention layers, head always trainable
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file as _save_file_raw


def _save_file_safe(tensors: dict, path: str) -> None:
    """Save safetensors via a temp file to avoid Windows ERROR_USER_MAPPED_FILE (os error 1224).

    On Windows, if the target path is currently memory-mapped (e.g. loaded for inference),
    writing directly to it fails.  Writing to a sibling .tmp then os.replace() sidesteps
    the restriction because rename updates only the directory entry.
    """
    tmp = path + ".tmp"
    try:
        _save_file_raw(tensors, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# Default HuggingFace repo ID for the SigLIP2 vision encoder.
# Stored in checkpoint metadata so merged checkpoints can be loaded
# without requiring the user to specify vision_encoder_path.
SIGLIP2_DEFAULT_REPO_ID = "google/siglip2-so400m-patch16-naflex"


def _is_hf_repo_or_url(path: str) -> "tuple[bool, str]":
    """Detect whether *path* is a HuggingFace repo ID or URL.

    Returns ``(True, resolved_repo_id)`` for:
    - ``"https://huggingface.co/google/siglip2-so400m-patch16-naflex"``
    - ``"google/siglip2-so400m-patch16-naflex"``

    Returns ``(False, "")`` for local file paths.
    """
    p = path.strip().strip('"').strip("'")
    if p.startswith("https://huggingface.co/"):
        return True, p[len("https://huggingface.co/"):]
    # org/model-name format: contains "/", not an absolute path, not a local file
    if "/" in p and not os.path.isabs(p) and not os.path.isfile(p):
        return True, p
    return False, ""


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

def _load_vision_encoder(safetensors_path: str, repo_id: str = SIGLIP2_DEFAULT_REPO_ID) -> nn.Module:
    """Load SigLIP2 vision encoder from a safetensors file, a HF repo ID, or URL.

    *safetensors_path* may be:
    - A local .safetensors path (pure vision encoder, LoRA checkpoint, or merged tagger checkpoint)
    - A HuggingFace repo ID  (e.g. ``"google/siglip2-so400m-patch16-naflex"``)
    - A HuggingFace URL      (e.g. ``"https://huggingface.co/google/siglip2-..."`` )

    *repo_id* controls which HuggingFace model is used for the module structure when
    loading a local safetensors file.  Defaults to ``SIGLIP2_DEFAULT_REPO_ID``.
    """
    from transformers import AutoModel

    # Strip surrounding quotes that may come from user input
    safetensors_path = safetensors_path.strip().strip('"').strip("'")

    # --- HuggingFace repo / URL shortcut ---
    is_hf, resolved_repo = _is_hf_repo_or_url(safetensors_path)
    if is_hf:
        print(f"[VisionEncoder] Loading directly from HuggingFace repo: {resolved_repo}")
        try:
            full_model = AutoModel.from_pretrained(resolved_repo, dtype=torch.float32, local_files_only=True)
        except Exception:
            full_model = AutoModel.from_pretrained(resolved_repo, dtype=torch.float32)
        return full_model.vision_model

    # Try local cache first to avoid network access and reduce peak memory.
    try:
        full_model = AutoModel.from_pretrained(
            repo_id, dtype=torch.float32, local_files_only=True
        )
    except Exception:
        full_model = AutoModel.from_pretrained(repo_id, dtype=torch.float32)
    vision_encoder = full_model.vision_model

    # Load our fine-tuned / custom weights
    state_dict = load_file(safetensors_path)

    # --- Merged tagger checkpoint detection (F2) ---
    # Merged tagger checkpoints have BOTH:
    #   - "head.weight" / "head.bias"  (the classification Linear layer)
    #   - "vision_encoder.*"           (the vision encoder sub-module)
    # NOTE: pure vision encoder safetensors also contain a "head.*" sub-module
    # (e.g. "head.probe", "head.attention.*") that belongs to the HF vision model
    # itself.  Using *only* "head.*" as the trigger causes false positives for
    # pure encoder files.  The combined condition below is unambiguous.
    _is_merged_tagger = (
        "head.weight" in state_dict
        and any(k.startswith("vision_encoder.") for k in state_dict)
    )
    if _is_merged_tagger:
        print(f"[VisionEncoder] Detected merged tagger checkpoint; extracting vision encoder sub-keys.")
        state_dict = {
            k[len("vision_encoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("vision_encoder.")
        }

    lora_keys = [k for k in state_dict if k.startswith("lora.")]
    if lora_keys:
        # This is a tagger LoRA checkpoint — merge LoRA deltas into the base weights
        # before returning the vision encoder.
        # Key format: "lora.encoder.layers.N.self_attn.{proj}.lora_A/B"
        # → maps to vision_encoder: "encoder.layers.N.self_attn.{proj}.weight"
        # Merge formula: W += (lora_A @ lora_B).T * (lora_alpha / lora_rank)
        # lora_rank is inferred from lora_A shape[:, 1]; lora_alpha from metadata if present.
        meta_path = safetensors_path.replace(".safetensors", "_metadata.json")
        lora_alpha: float = 1.0
        lora_rank: int = 1
        if os.path.isfile(meta_path):
            import json as _json
            with open(meta_path, "r", encoding="utf-8") as _f:
                _meta = _json.load(_f)
            lora_alpha = float(_meta.get("lora_alpha", 1.0))
            lora_rank  = int(_meta.get("lora_rank", 1))
        else:
            # Infer rank from first lora_A tensor (shape: [in_features, rank])
            _first_A = next(v for k, v in state_dict.items() if k.endswith(".lora_A"))
            lora_rank = _first_A.shape[1]
            lora_alpha = float(lora_rank)  # default scale=1 when alpha==rank

        scale = lora_alpha / lora_rank

        # Collect pairs: module_path -> (lora_A, lora_B)
        _lora_pairs: dict = {}
        for k, v in state_dict.items():
            if not k.startswith("lora."):
                continue
            # strip "lora." prefix and ".lora_A" / ".lora_B" suffix
            if k.endswith(".lora_A"):
                mod_path = k[len("lora."):-len(".lora_A")]
                _lora_pairs.setdefault(mod_path, {})["A"] = v
            elif k.endswith(".lora_B"):
                mod_path = k[len("lora."):-len(".lora_B")]
                _lora_pairs.setdefault(mod_path, {})["B"] = v

        vs_dict = vision_encoder.state_dict()
        merged = 0
        for mod_path, ab in _lora_pairs.items():
            if "A" not in ab or "B" not in ab:
                continue
            weight_key = f"{mod_path}.weight"
            if weight_key not in vs_dict:
                continue
            lora_A = ab["A"].float()
            lora_B = ab["B"].float()
            delta = (lora_A @ lora_B).T * scale
            vs_dict[weight_key] = vs_dict[weight_key].float() + delta
            merged += 1

        vision_encoder.load_state_dict(vs_dict, strict=True)
        print(f"[VisionEncoder] Merged {merged} LoRA modules from {os.path.basename(safetensors_path)} (alpha={lora_alpha}, rank={lora_rank})")
    else:
        # Pure vision encoder weights
        vision_encoder.load_state_dict(state_dict, strict=True)

    return vision_encoder


# ------------------------------------------------------------------
# Custom Attention Pooling
# ------------------------------------------------------------------

class CustomAttentionPooling(nn.Module):
    """Learnable single-query attention pooling over patch tokens.

    Parameters
    ----------
    in_dim          : input token dimension (1152 for SigLIP2 so400m)
    cls_dim         : output dimension (query space / pooled vector size)
    hidden_proj_dim : if set, proj_k/proj_v expand tokens to this dimension
                      (larger than in_dim for richer representations).
                      MHA uses kdim=vdim=hidden_proj_dim with embed_dim=cls_dim.
                      If None, proj_k/proj_v map directly to cls_dim (no overhead).
    num_heads       : MHA heads; auto-halved until cls_dim % num_heads == 0
    """

    def __init__(
        self,
        in_dim: int,
        cls_dim: int,
        hidden_proj_dim: Optional[int] = None,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_proj_dim = hidden_proj_dim
        kv_dim = hidden_proj_dim if hidden_proj_dim else cls_dim

        # Adjust num_heads so cls_dim is divisible
        while cls_dim % num_heads != 0 and num_heads > 1:
            num_heads //= 2

        self.query  = nn.Parameter(torch.zeros(1, 1, cls_dim))
        nn.init.normal_(self.query, std=0.02)
        self.proj_k = nn.Linear(in_dim, kv_dim)
        self.proj_v = nn.Linear(in_dim, kv_dim)

        if hidden_proj_dim:
            # K/V live in hidden_proj_dim space; Q lives in cls_dim space
            self.attn = nn.MultiheadAttention(
                embed_dim=cls_dim,
                num_heads=num_heads,
                kdim=hidden_proj_dim,
                vdim=hidden_proj_dim,
                batch_first=True,
            )
        else:
            # All in cls_dim space — no extra overhead
            self.attn = nn.MultiheadAttention(cls_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, in_dim] -> [B, cls_dim]"""
        k = self.proj_k(x)                        # [B, N, kv_dim]
        v = self.proj_v(x)                        # [B, N, kv_dim]
        q = self.query.expand(x.size(0), -1, -1)  # [B, 1, cls_dim]
        out, _ = self.attn(q, k, v)
        return out.squeeze(1)                      # [B, cls_dim]

    def get_token_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B, N, hidden_proj_dim] token features for external use (e.g. SDXL).

        Reuses proj_v weights; only available when hidden_proj_dim is set.
        Raises RuntimeError otherwise.
        """
        if self.hidden_proj_dim is None:
            raise RuntimeError(
                "get_token_features() requires hidden_proj_dim to be set. "
                "Use hidden_proj_dim > 0 in the model config."
            )
        return self.proj_v(x)  # [B, N, hidden_proj_dim]


# ------------------------------------------------------------------
# Full-parameter model
# ------------------------------------------------------------------

class SigLIP2TaggerModel(nn.Module):
    """SigLIP2 vision encoder + classification head (full-parameter training).

    Parameters
    ----------
    num_tags         : number of output classes
    vision_encoder   : pre-loaded vision encoder nn.Module
    freeze_encoder   : if True, gradients do not flow through vision encoder
    hidden_size      : vision encoder hidden dimension (1152 for so400m)
    cls_dim          : if set, use CustomAttentionPooling instead of pooler_output.
                       Sets the query/output dimension of attention pooling.
    hidden_proj_dim  : if set (requires cls_dim), proj_k/proj_v expand tokens to
                       hidden_proj_dim before attention pooling. The expanded token
                       features [B, N, hidden_proj_dim] are accessible via
                       custom_pooler.get_token_features() for SDXL conditioning.
    """

    HIDDEN_SIZE = 1152  # so400m

    def __init__(
        self,
        num_tags: int,
        vision_encoder: nn.Module,
        freeze_encoder: bool = False,
        hidden_size: int = HIDDEN_SIZE,
        cls_dim: Optional[int] = None,
        hidden_proj_dim: Optional[int] = None,
        is_naflex: bool = True,
    ) -> None:
        super().__init__()
        self.vision_encoder  = vision_encoder
        self.cls_dim         = cls_dim
        self.hidden_proj_dim = hidden_proj_dim
        self.is_naflex       = is_naflex

        if cls_dim:
            self.custom_pooler: Optional[CustomAttentionPooling] = CustomAttentionPooling(
                in_dim=hidden_size,
                cls_dim=cls_dim,
                hidden_proj_dim=hidden_proj_dim,
            )
            pool_dim = cls_dim
        else:
            self.custom_pooler = None
            pool_dim = hidden_size

        self.head = nn.Linear(pool_dim, num_tags)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        if freeze_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return logits [B, num_tags]."""
        if self.is_naflex:
            out = self.vision_encoder(
                pixel_values=pixel_values,
                attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
        else:
            out = self.vision_encoder(pixel_values=pixel_values)
        if self.custom_pooler is not None:
            pooled = self.custom_pooler(out.last_hidden_state)  # [B, cls_dim]
        else:
            pooled = out.pooler_output                          # [B, hidden_size]
        return self.head(pooled)                                # [B, num_tags]

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, output_dir: str, name: str, metadata: Optional[dict] = None) -> str:
        """Save model weights and metadata JSON. Returns path to safetensors file."""
        os.makedirs(output_dir, exist_ok=True)
        path_st   = os.path.join(output_dir, f"{name}.safetensors")
        path_meta = os.path.join(output_dir, f"{name}_metadata.json")

        sd = {k: v.contiguous() for k, v in self.state_dict().items()}
        _save_file_safe(sd, path_st)

        if metadata:
            with open(path_meta, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        return path_st

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        vision_encoder_path: str = "",
        num_tags: Optional[int] = None,
    ) -> "SigLIP2TaggerModel":
        """Load model from checkpoint safetensors.

        For merged (full) checkpoints *vision_encoder_path* may be omitted — the
        checkpoint already contains all vision encoder weights.  In that case the
        HuggingFace base model is used only to supply the module structure, and
        all weights are then overwritten by the checkpoint.
        """
        meta_path = checkpoint_path.replace(".safetensors", "_metadata.json")
        metadata: dict = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        if num_tags is None:
            num_tags = metadata.get("num_tags")
            if num_tags is None:
                raise ValueError("num_tags must be provided or present in metadata")

        cls_dim         = metadata.get("cls_dim")
        hidden_proj_dim = metadata.get("hidden_proj_dim")
        repo_id         = metadata.get("vision_encoder_repo", SIGLIP2_DEFAULT_REPO_ID)
        is_naflex       = metadata.get("is_naflex", True)

        # Detect merged checkpoint: contains vision_encoder.* keys so all weights
        # are embedded — vision_encoder_path is only needed for delta (LoRA) ckpts.
        # Even if the caller passes vision_encoder_path, skip _load_vision_encoder
        # for merged checkpoints to avoid architecture mismatches (the weights get
        # overwritten by model.load_state_dict below regardless).
        from safetensors import safe_open as _safe_open
        with _safe_open(checkpoint_path, framework="pt", device="cpu") as _f:
            _ckpt_has_encoder = any(k.startswith("vision_encoder.") for k in _f.keys())

        if vision_encoder_path and not _ckpt_has_encoder:
            vision_encoder = _load_vision_encoder(vision_encoder_path, repo_id=repo_id)
        else:
            if vision_encoder_path and _ckpt_has_encoder:
                print(f"[SigLIP2Load] Merged checkpoint detected - ignoring vision_encoder_path ({os.path.basename(vision_encoder_path)}); encoder weights come from checkpoint.")
            # Merged or no-path: load HuggingFace base model for structure only;
            # all weights will be overwritten by the checkpoint state dict below.
            from transformers import AutoModel
            try:
                full_model = AutoModel.from_pretrained(repo_id, dtype=torch.float32, local_files_only=True)
            except Exception:
                full_model = AutoModel.from_pretrained(repo_id, dtype=torch.float32)
            vision_encoder = full_model.vision_model

        model = cls(
            num_tags=num_tags,
            vision_encoder=vision_encoder,
            cls_dim=cls_dim,
            hidden_proj_dim=hidden_proj_dim,
            is_naflex=is_naflex,
        )
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)
        return model

    def load_weights_inplace(self, ckpt_path: str) -> None:
        """Load full model weights from checkpoint into this instance (for resume)."""
        state_dict = load_file(ckpt_path)
        self.load_state_dict(state_dict, strict=True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_count(self) -> Dict[str, int]:
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ------------------------------------------------------------------
# LoRA model
# ------------------------------------------------------------------

class SigLIP2TaggerLoRAModel(nn.Module):
    """SigLIP2 vision encoder with LoRA adapters + classification head.

    LoRA is applied to attention projection layers in the vision encoder.
    The head is always fully trainable. The base encoder weights are frozen.
    Uses pooler_output [B, 1152] -> Linear head (no custom pooling for LoRA).
    """

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
        is_naflex: bool = True,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.lora_rank  = lora_rank
        self.lora_alpha = lora_alpha
        self.is_naflex  = is_naflex

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

            parts  = name.split(".")
            parent = self.vision_encoder
            for part in parts[:-1]:
                parent = getattr(parent, part)

            lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, parts[-1], lora_linear)
            self._lora_modules[name] = lora_linear

        print(f"[TaggerLoRA] Injected LoRA into {len(self._lora_modules)} modules")

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_naflex:
            out = self.vision_encoder(
                pixel_values=pixel_values,
                attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
        else:
            out = self.vision_encoder(pixel_values=pixel_values)
        return self.head(out.pooler_output)  # [B, num_tags]

    # ------------------------------------------------------------------
    # Save / load (saves only LoRA + head, not full encoder)
    # ------------------------------------------------------------------

    def save_checkpoint(self, output_dir: str, name: str, metadata: Optional[dict] = None) -> str:
        """Save LoRA weights + head weights only (compact checkpoint)."""
        os.makedirs(output_dir, exist_ok=True)
        path_st   = os.path.join(output_dir, f"{name}.safetensors")
        path_meta = os.path.join(output_dir, f"{name}_metadata.json")

        sd: Dict[str, torch.Tensor] = {}
        for module_name, lora_module in self._lora_modules.items():
            prefix = f"lora.{module_name}"
            sd[f"{prefix}.lora_A"] = lora_module.lora_A.detach().contiguous()
            sd[f"{prefix}.lora_B"] = lora_module.lora_B.detach().contiguous()
        sd["head.weight"] = self.head.weight.detach().contiguous()
        sd["head.bias"]   = self.head.bias.detach().contiguous()

        if metadata is None:
            metadata = {}
        metadata.update({
            "lora_rank":        self.lora_rank,
            "lora_alpha":       self.lora_alpha,
            "num_lora_modules": len(self._lora_modules),
        })

        _save_file_safe(sd, path_st)
        with open(path_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return path_st

    def save_merged_checkpoint(
        self,
        output_dir: str,
        name: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Merge LoRA weights into the vision encoder and save as a full model checkpoint.

        The resulting file is compatible with ``SigLIP2TaggerModel.load_checkpoint``
        (no LoRA keys, full vision encoder weights included).
        """
        import copy as _copy

        os.makedirs(output_dir, exist_ok=True)
        path_st   = os.path.join(output_dir, f"{name}.safetensors")
        path_meta = os.path.join(output_dir, f"{name}_metadata.json")

        # Build merged state dict by deep-copying the full model state dict.
        # LoRALinear modules store the original weight as ``base.weight``
        # plus ``lora_A`` / ``lora_B``; we materialise the merged weight and
        # produce a state dict that looks like a plain SigLIP2TaggerModel.
        merged_sd: Dict[str, torch.Tensor] = {}
        scale = self.lora_alpha / self.lora_rank

        # Collect merged weights for all LoRA-replaced Linear layers first.
        merged_weights: Dict[str, torch.Tensor] = {}
        for module_path, lora_module in self._lora_modules.items():
            # lora_A: [in_features, rank], lora_B: [rank, out_features]
            # merged delta: (lora_A @ lora_B).T  →  [out_features, in_features]
            A = lora_module.lora_A.float()  # [in, rank]
            B = lora_module.lora_B.float()  # [rank, out]
            delta = (A @ B).T * scale       # [out, in]
            w     = lora_module.base.weight.float() + delta
            merged_weights[module_path + ".weight"] = w
            if lora_module.base.bias is not None:
                merged_weights[module_path + ".bias"] = lora_module.base.bias.float()

        # Build full state dict with the same key format as SigLIP2TaggerModel.
        # Keys from vision_encoder are stored under "vision_encoder.*".
        for k, v in self.state_dict().items():
            # Translate LoRALinear keys: "vision_encoder.*.lora_A" etc. → skip or replace
            # LoRALinear keys look like "vision_encoder.<path>.lora_A" / "lora_B" / "base.weight"
            matched = False
            for module_path in self._lora_modules:
                ve_prefix = f"vision_encoder.{module_path}"
                if k.startswith(ve_prefix + "."):
                    # This key belongs to a LoRA module – skip (handled below)
                    matched = True
                    break
            if not matched:
                # Regular non-LoRA key: copy as-is but cast to float16 for compactness
                merged_sd[k] = v.detach().to(torch.float16).contiguous()

        # Insert merged Linear weights
        for relative_path, tensor in merged_weights.items():
            full_key = f"vision_encoder.{relative_path}"
            merged_sd[full_key] = tensor.to(torch.float16).contiguous()

        # Head
        merged_sd["head.weight"] = self.head.weight.detach().to(torch.float16).contiguous()
        merged_sd["head.bias"]   = self.head.bias.detach().to(torch.float16).contiguous()

        _save_file_safe(merged_sd, path_st)

        if metadata is None:
            metadata = {}
        metadata.update({
            "checkpoint_type": "merged",
            "num_lora_modules_merged": len(self._lora_modules),
        })
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

        lora_rank  = metadata.get("lora_rank",  lora_rank)
        lora_alpha = metadata.get("lora_alpha", lora_alpha)
        repo_id    = metadata.get("vision_encoder_repo", SIGLIP2_DEFAULT_REPO_ID)
        is_naflex  = metadata.get("is_naflex", True)

        vision_encoder = _load_vision_encoder(vision_encoder_path, repo_id=repo_id)
        model = cls(
            num_tags=num_tags,
            vision_encoder=vision_encoder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            is_naflex=is_naflex,
        )

        saved = load_file(checkpoint_path)
        model.head.weight.data.copy_(saved["head.weight"])
        model.head.bias.data.copy_(saved["head.bias"])

        for module_name, lora_module in model._lora_modules.items():
            prefix = f"lora.{module_name}"
            if f"{prefix}.lora_A" in saved:
                lora_module.lora_A.data.copy_(saved[f"{prefix}.lora_A"])
            if f"{prefix}.lora_B" in saved:
                lora_module.lora_B.data.copy_(saved[f"{prefix}.lora_B"])

        return model

    def load_weights_inplace(self, ckpt_path: str) -> None:
        """Load LoRA + head weights from checkpoint into this instance (for resume)."""
        saved = load_file(ckpt_path)
        self.head.weight.data.copy_(saved["head.weight"])
        self.head.bias.data.copy_(saved["head.bias"])
        for module_name, lora_module in self._lora_modules.items():
            prefix = f"lora.{module_name}"
            if f"{prefix}.lora_A" in saved:
                lora_module.lora_A.data.copy_(saved[f"{prefix}.lora_A"])
            if f"{prefix}.lora_B" in saved:
                lora_module.lora_B.data.copy_(saved[f"{prefix}.lora_B"])

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

def _inherit_head(
    model: nn.Module,
    checkpoint_path: str,
    new_num_tags: int,
    new_vocab: Optional[Dict[str, int]] = None,
    old_tag_to_idx: Optional[Dict[str, int]] = None,
) -> None:
    """Copy head weights from *checkpoint_path* into *model* using tag-name alignment.

    Tag-name alignment
    ------------------
    The checkpoint directory is expected to contain a ``vocabulary.json`` file
    (produced by TagVocabulary.to_dict()) that maps tag names to their old indices.
    The new vocabulary (*new_vocab*: tag → new_idx) is used to look up each old tag
    so weights are placed at the correct new index regardless of ordering changes.

    Tags present in the old vocabulary but absent from the new one are simply
    dropped — the new head rows for those positions remain zero-initialized.
    Tags present in the new vocabulary but absent from the old one are
    zero-initialized (new tags to learn from scratch).

    If no vocabulary.json is found next to the checkpoint, falls back to
    positional copy (old row i → new row i, up to min(old, new) rows) with a
    warning.  This handles the exact-same-vocabulary case safely.
    """
    import json as _json

    saved = load_file(checkpoint_path)
    if "head.weight" not in saved:
        print(f"[TaggerModel] _inherit_head: no head.weight in {checkpoint_path}, skipping")
        return

    src_w = saved["head.weight"].float()  # [old_num_tags, hidden]
    src_b = saved["head.bias"].float()    # [old_num_tags]
    hidden = src_w.shape[1]

    # Build new head (zero-initialized) on CPU
    new_head = nn.Linear(hidden, new_num_tags)
    nn.init.zeros_(new_head.weight)
    nn.init.zeros_(new_head.bias)

    # Try to load old vocabulary for tag-name-based alignment
    ckpt_dir   = os.path.dirname(os.path.abspath(checkpoint_path))
    vocab_path = os.path.join(ckpt_dir, "vocabulary.json")

    copied = skipped = 0

    # Resolve old_tag_to_idx: caller may supply it directly (resume path);
    # otherwise fall back to reading vocabulary.json next to the checkpoint.
    _old_tag_to_idx: Optional[Dict[str, int]] = old_tag_to_idx
    if _old_tag_to_idx is None and os.path.isfile(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            old_vocab_data = _json.load(f)
        _old_tag_to_idx = {
            k: int(v) for k, v in old_vocab_data["tag_to_idx"].items()
        }

    if _old_tag_to_idx is not None and new_vocab is not None:
        # For each tag in the new vocabulary, copy the row from the old head if it existed
        for tag, new_idx in new_vocab.items():
            old_idx = _old_tag_to_idx.get(tag)
            if old_idx is None:
                skipped += 1
                continue  # new tag — stays zero-initialized
            if old_idx >= src_w.shape[0]:
                skipped += 1
                continue  # shouldn't happen, but guard anyway
            new_head.weight.data[new_idx] = src_w[old_idx]
            new_head.bias.data[new_idx]   = src_b[old_idx]
            copied += 1
        old_size = len(_old_tag_to_idx)
        print(f"[TaggerModel] Head inherited via tag-name alignment: "
              f"{copied} tags copied, {skipped} new/missing tags zero-initialized "
              f"(checkpoint head: {src_w.shape[0]}, old vocab: {old_size}, new vocab: {new_num_tags})")
    else:
        # Fallback: positional copy — safe only when vocab order is unchanged
        if not os.path.isfile(vocab_path) and old_tag_to_idx is None:
            print(f"[TaggerModel] Warning: vocabulary.json not found in {ckpt_dir}, "
                  f"falling back to positional head copy (assumes identical tag order)")
        copy_rows = min(src_w.shape[0], new_num_tags)
        new_head.weight.data[:copy_rows] = src_w[:copy_rows]
        new_head.bias.data[:copy_rows]   = src_b[:copy_rows]
        copied = copy_rows
        print(f"[TaggerModel] Head inherited (positional): {copy_rows} rows copied, "
              f"{new_num_tags - copy_rows} new rows zero-initialized")

    # Move to same device as existing head
    device = next(model.head.parameters()).device
    model.head = new_head.to(device)


def build_tagger_model(
    training_method: str,
    num_tags: int,
    vision_encoder_path: str,
    lora_rank: int = 32,
    lora_alpha: float = 16.0,
    freeze_encoder: bool = False,
    cls_dim: Optional[int] = None,
    hidden_proj_dim: Optional[int] = None,
    init_head_from: Optional[str] = None,
    new_vocab: Optional[Dict[str, int]] = None,
    repo_id: str = SIGLIP2_DEFAULT_REPO_ID,
    is_naflex: bool = True,
) -> nn.Module:
    """Build the appropriate tagger model.

    Parameters
    ----------
    training_method     : "full" | "lora"
    num_tags            : number of output tag classes
    vision_encoder_path : path to siglip2_so400m_vision_encoder.safetensors,
                          OR a tagger LoRA checkpoint — LoRA weights will be merged
                          into the base encoder automatically.
    lora_rank           : LoRA rank (lora mode only)
    lora_alpha          : LoRA alpha (lora mode only)
    freeze_encoder      : freeze encoder entirely (full mode only)
    cls_dim             : CustomAttentionPooling output dim (full mode only)
    hidden_proj_dim     : proj_k/proj_v expansion dim; requires cls_dim (full mode only)
    init_head_from      : optional path to a tagger checkpoint whose head.weight/bias
                          should be inherited.  Tag-name alignment is used when
                          vocabulary.json is present in the same directory: each tag in
                          the new vocabulary is looked up in the old vocabulary and its
                          weight row is placed at the correct new index.  Tags absent
                          from the old vocabulary are zero-initialized.  Old tags not
                          present in the new vocabulary are simply dropped (not an error).
    new_vocab           : new tag→idx mapping (TagVocabulary.tag_to_idx) required for
                          tag-name alignment; falls back to positional copy if None.
    """
    print(f"[TaggerModel] Loading vision encoder from: {vision_encoder_path}")
    vision_encoder = _load_vision_encoder(vision_encoder_path, repo_id=repo_id)

    if training_method == "lora":
        if cls_dim:
            print("[TaggerModel] Warning: cls_dim / hidden_proj_dim are ignored for LoRA training")
        model = SigLIP2TaggerLoRAModel(
            num_tags=num_tags,
            vision_encoder=vision_encoder,
            lora_rank=lora_rank,
            lora_alpha=float(lora_alpha),
            is_naflex=is_naflex,
        )
    elif training_method == "full":
        if hidden_proj_dim and not cls_dim:
            raise ValueError("hidden_proj_dim requires cls_dim to be set")
        model = SigLIP2TaggerModel(
            num_tags=num_tags,
            vision_encoder=vision_encoder,
            freeze_encoder=freeze_encoder,
            cls_dim=cls_dim,
            hidden_proj_dim=hidden_proj_dim,
            is_naflex=is_naflex,
        )
    else:
        raise ValueError(f"Unknown training_method: {training_method!r}. Use 'full' or 'lora'.")

    # If init_head_from is not set but vision_encoder_path is a merged tagger
    # checkpoint (contains head.weight), auto-inherit the head from it so that
    # weights for existing tags are preserved across vocabulary changes.
    _effective_head_src = init_head_from
    if not _effective_head_src and os.path.isfile(vision_encoder_path):
        from safetensors import safe_open as _safe_open
        with _safe_open(vision_encoder_path, framework="pt", device="cpu") as _f:
            _keys = set(_f.keys())
        if "head.weight" in _keys and any(k.startswith("vision_encoder.") for k in _keys):
            _effective_head_src = vision_encoder_path
            print(f"[TaggerModel] Auto-inheriting head from merged checkpoint: {vision_encoder_path}")

    if _effective_head_src:
        _inherit_head(model, _effective_head_src, num_tags, new_vocab=new_vocab)

    return model
