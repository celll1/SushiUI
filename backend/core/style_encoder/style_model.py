"""Style encoder: SigLIP2 NaFlex trunk + masked attention pooling + projector (design §4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_bidirectional_mask
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings, Siglip2VisionModel

from core.tagger.siglip2_tagger_model import CustomAttentionPooling, _load_vision_encoder, inject_lora

MODES = ("full", "lora")
_LORA_PROJECTIONS = "q_proj|k_proj|v_proj|out_proj"
_ATTN_SUPPORT_FLAGS = {"eager": None, "sdpa": "_supports_sdpa", "flash_attention_2": "_supports_flash_attn",
                       "flex_attention": "_supports_flex_attn"}


@dataclass(frozen=True)
class LoRASpec:
    rank: int
    alpha: float
    dropout: float


@dataclass(frozen=True)
class StyleOutput:
    embedding: torch.Tensor   # [B, embed_dim], L2-normalised, fp32
    features: torch.Tensor    # [B, N, token_dim], exactly 0 where valid_mask is False
    valid_mask: torch.Tensor  # [B, N] bool


def supported_attn_implementations(model_cls: type) -> tuple[str, ...]:
    return tuple(name for name, flag in _ATTN_SUPPORT_FLAGS.items()
                 if flag is None or getattr(model_cls, flag, False))


def check_attn_implementation(name: str, model_cls: type) -> None:
    supported = supported_attn_implementations(model_cls)
    if name not in supported:
        raise ValueError(f"attn_implementation {name!r} is not supported by the installed {model_cls.__name__}; "
                         f"supported: {list(supported)}")


def resolve_feature_layers(feature_layers: Sequence[int], num_layers: int) -> tuple[int, ...]:
    """Indices into the hidden-state list [embeddings, layer 1, ..., layer L]; negative counts from the end."""
    if not feature_layers:
        raise ValueError("style_feature_layers must not be empty")
    resolved = []
    for layer in feature_layers:
        if isinstance(layer, bool) or not isinstance(layer, int) or not -(num_layers + 1) <= layer <= num_layers:
            raise ValueError(f"style_feature_layers entry {layer!r} out of range for {num_layers} encoder layers")
        resolved.append(layer % (num_layers + 1))
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"style_feature_layers {list(feature_layers)} name the same layer twice")
    return tuple(resolved)


class StyleEncoder(nn.Module):
    def __init__(self, vision_encoder: nn.Module, feature_layers: Sequence[int], token_dim: int, pool_heads: int,
                 projector_hidden_dim: int, embed_dim: int, mode: str, lora: Optional[LoRASpec]) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if (mode == "lora") != (lora is not None):
            raise ValueError("a LoRASpec is required for mode='lora' and not allowed for mode='full'")
        if not isinstance(getattr(vision_encoder, "embeddings", None), Siglip2VisionEmbeddings):
            raise ValueError("the style encoder needs a SigLIP2 NaFlex vision model (Siglip2VisionEmbeddings)")
        if token_dim % pool_heads != 0:
            raise ValueError(f"token_dim {token_dim} is not divisible by pool_heads {pool_heads}")
        check_attn_implementation(vision_encoder.config._attn_implementation, type(vision_encoder))
        self.vision_encoder = vision_encoder
        self.mode = mode
        config = vision_encoder.config
        hidden = config.hidden_size
        self.feature_layers = resolve_feature_layers(feature_layers, len(vision_encoder.encoder.layers))
        self.depth = max(self.feature_layers)
        self.layer_norms = nn.ModuleList(nn.LayerNorm(hidden, eps=config.layer_norm_eps) for _ in self.feature_layers)
        self.token_proj = nn.Linear(hidden * len(self.feature_layers), token_dim)
        self.pooling = CustomAttentionPooling(in_dim=token_dim, cls_dim=token_dim, hidden_proj_dim=None,
                                              num_heads=pool_heads)
        self.projector = nn.Sequential(nn.Linear(token_dim, projector_hidden_dim), nn.GELU(),
                                       nn.Linear(projector_hidden_dim, embed_dim))

        # Layers past the deepest feature layer, post_layernorm and the HF pooling head never run.
        for p in vision_encoder.parameters():
            p.requires_grad = mode == "full"
        for module in (*vision_encoder.encoder.layers[self.depth:], vision_encoder.post_layernorm,
                       getattr(vision_encoder, "head", None)):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False
        self.lora_modules = {}
        if lora is not None:
            used = "|".join(str(i) for i in range(self.depth))
            patterns = [rf"encoder\.layers\.(?:{used})\.self_attn\.(?:{_LORA_PROJECTIONS})$"] if used else []
            self.lora_modules = inject_lora(vision_encoder, patterns, lora.rank, lora.alpha, lora.dropout)
            if not self.lora_modules:
                raise ValueError(f"mode='lora' with style_feature_layers {list(feature_layers)} runs no encoder "
                                 f"layer to adapt; the encoder would stay frozen")

    def forward(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor,
                spatial_shapes: torch.Tensor) -> StyleOutput:
        valid = pixel_attention_mask != 0
        encoder = self.vision_encoder
        # Zeroed so non-finite pixels at invalid positions cannot reach valid tokens (0 * NaN in P.V).
        pixel_values = torch.where(valid.unsqueeze(-1), pixel_values, torch.zeros((), dtype=pixel_values.dtype,
                                                                                  device=pixel_values.device))
        hidden = encoder.embeddings(pixel_values, spatial_shapes)
        # Keys only: a masked token's own hidden state still carries its pixels (I-1), hence the zeroing below.
        mask = create_bidirectional_mask(config=encoder.config, inputs_embeds=hidden, attention_mask=valid)
        states = [hidden]
        for layer in encoder.encoder.layers[:self.depth]:
            hidden = layer(hidden, mask)
            states.append(hidden)
        features = torch.cat([norm(states[i]) for norm, i in zip(self.layer_norms, self.feature_layers)], dim=-1)
        features = self.token_proj(features)
        features = torch.where(valid.unsqueeze(-1), features, torch.zeros((), dtype=features.dtype,
                                                                          device=features.device))
        pooled = self.pooling(features, valid)
        embedding = F.normalize(self.projector(pooled).float(), dim=-1)
        return StyleOutput(embedding=embedding, features=features, valid_mask=valid)

    def embed(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor,
              spatial_shapes: torch.Tensor) -> torch.Tensor:
        return self(pixel_values, pixel_attention_mask, spatial_shapes).embedding

    def token_features(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor,
                       spatial_shapes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self(pixel_values, pixel_attention_mask, spatial_shapes)
        return out.features, out.valid_mask

    def trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(name, p) for name, p in self.named_parameters() if p.requires_grad]


def build_style_encoder(source: str, repo_id: str, attn_implementation: str, feature_layers: Sequence[int],
                        token_dim: int, pool_heads: int, projector_hidden_dim: int, embed_dim: int, mode: str,
                        lora: Optional[LoRASpec]) -> StyleEncoder:
    check_attn_implementation(attn_implementation, Siglip2VisionModel)
    vision_encoder = _load_vision_encoder(source, repo_id=repo_id, attn_implementation=attn_implementation)
    return StyleEncoder(vision_encoder, feature_layers, token_dim, pool_heads, projector_hidden_dim, embed_dim,
                        mode, lora)
