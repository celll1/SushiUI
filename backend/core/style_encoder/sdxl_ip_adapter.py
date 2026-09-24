"""Trainable SDXL cross-attention bridge for 512-D style embeddings.

The style encoder and U-Net can remain frozen: only this module's token
projection and per-layer image K/V projections require gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import tempfile

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file


@dataclass
class StyleIPContext:
    adapter: "SDXLStyleIPAdapter"
    tokens: torch.Tensor
    scale: float
    cfg_batch: bool


class SDXLStyleIPAdapter(nn.Module):
    def __init__(self, unet, *, embedding_dim: int = 512, num_tokens: int = 4):
        super().__init__()
        if embedding_dim != 512 or num_tokens < 1:
            raise ValueError("Expected a 512-D style embedding and at least one image token")
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        specs = {}
        for name in unet.attn_processors:
            if not name.endswith("attn2.processor"):
                continue
            attn = unet.get_submodule(name.removesuffix(".processor"))
            specs[name] = (attn.to_k.in_features, attn.to_k.out_features)
        if not specs:
            raise ValueError("U-Net has no cross-attention processors")
        cross_dims = {cross_dim for cross_dim, _ in specs.values()}
        if len(cross_dims) != 1:
            raise ValueError("SDXL style bridge requires one text cross-attention width")
        self.cross_attention_dim = cross_dims.pop()
        self.layer_names = tuple(specs)
        self.projection = nn.Linear(embedding_dim, num_tokens * self.cross_attention_dim)
        self.norm = nn.LayerNorm(self.cross_attention_dim)
        self.to_k_ip = nn.ModuleList(
            nn.Linear(self.cross_attention_dim, width, bias=False) for _, width in specs.values()
        )
        self.to_v_ip = nn.ModuleList(
            nn.Linear(self.cross_attention_dim, width, bias=False) for _, width in specs.values()
        )
        self._layer_index = {name: index for index, name in enumerate(self.layer_names)}

    def encode(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != self.embedding_dim:
            raise ValueError(f"style embedding must have shape [batch, {self.embedding_dim}]")
        embedding = embedding.to(device=self.projection.weight.device, dtype=self.projection.weight.dtype)
        return self.norm(self.projection(embedding).reshape(-1, self.num_tokens, self.cross_attention_dim))

    def context(self, embedding: torch.Tensor, scale: float = 1.0, *, cfg_batch: bool = False) -> StyleIPContext:
        if not torch.isfinite(torch.tensor(scale)) or scale < 0:
            raise ValueError("style strength must be finite and nonnegative")
        return StyleIPContext(self, self.encode(embedding), float(scale), bool(cfg_batch))

    def project_kv(self, layer_name: str | None, tokens: torch.Tensor, heads: int,
                   head_dim: int, *, device: torch.device, dtype: torch.dtype):
        if layer_name not in self._layer_index:
            raise ValueError(f"Unknown style IP-Adapter layer: {layer_name}")
        index = self._layer_index[layer_name]
        weight = self.to_k_ip[index].weight
        tokens = tokens.to(device=weight.device, dtype=weight.dtype)
        key = self.to_k_ip[index](tokens)
        value = self.to_v_ip[index](tokens)
        if key.shape[-1] != heads * head_dim:
            raise ValueError(f"U-Net attention width changed for {layer_name}")
        return (
            key.reshape(key.shape[0], -1, heads, head_dim).transpose(1, 2).to(device=device, dtype=dtype),
            value.reshape(value.shape[0], -1, heads, head_dim).transpose(1, 2).to(device=device, dtype=dtype),
        )

    def architecture(self) -> dict:
        return {
            "type": "sdxl_style_ip_adapter_v1",
            "embedding_dim": self.embedding_dim,
            "num_tokens": self.num_tokens,
            "cross_attention_dim": self.cross_attention_dim,
            "layers": list(self.layer_names),
            "layer_widths": [layer.out_features for layer in self.to_k_ip],
        }

    def check_architecture(self, architecture: dict) -> None:
        if architecture != self.architecture():
            raise ValueError("Style IP-Adapter artifact does not match this SDXL U-Net")


def save_sdxl_style_ip_adapter(adapter: SDXLStyleIPAdapter, path: str,
                               *, style_encoder_sha256: str, sdxl_model_sha256: str) -> None:
    """Save only trainable bridge weights with source weight identities."""
    for name, digest in (("style encoder", style_encoder_sha256), ("SDXL model", sdxl_model_sha256)):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{name} SHA-256 must be lowercase hexadecimal")
    metadata = {
        "architecture": json.dumps(adapter.architecture(), separators=(",", ":")),
        "style_encoder_sha256": style_encoder_sha256,
        "sdxl_model_sha256": sdxl_model_sha256,
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, staged = tempfile.mkstemp(prefix=".style-ip-", suffix=".safetensors", dir=directory)
    os.close(fd)
    try:
        state = {key: value.detach().cpu().contiguous() for key, value in adapter.state_dict().items()}
        save_file(state, staged, metadata=metadata)
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def load_sdxl_style_ip_adapter(unet, path: str, *, style_encoder_sha256: str,
                               sdxl_model_sha256: str) -> SDXLStyleIPAdapter:
    with safe_open(path, framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata() or {}
        if metadata.get("style_encoder_sha256") != style_encoder_sha256:
            raise ValueError("Style IP-Adapter was trained with a different style encoder")
        if metadata.get("sdxl_model_sha256") != sdxl_model_sha256:
            raise ValueError("Style IP-Adapter was trained with a different SDXL model")
        architecture = json.loads(metadata["architecture"])
        adapter = SDXLStyleIPAdapter(
            unet, embedding_dim=architecture["embedding_dim"], num_tokens=architecture["num_tokens"]
        )
        adapter.check_architecture(architecture)
        adapter.load_state_dict({key: artifact.get_tensor(key) for key in artifact.keys()}, strict=True)
    return adapter
