"""Shared full-canvas summary adapter and checkpoint contract for Qwen-Image 2.1."""

from __future__ import annotations

from typing import Mapping, TYPE_CHECKING

import torch
import torch.nn as nn
from core.models.components.global_summary_attention import global_summary_attention

if TYPE_CHECKING:
    from core.training.partition import PartitionBox


class QwenPartitionGlobalAdapter(nn.Module):
    """Small trainable full-canvas communication path for partitioned training."""

    ADAPTER_ALGORITHM = "qwen_partition_global"
    WEIGHT_DECOMPOSE = False

    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int,
        *,
        rank: int = 64,
        summary_tokens: int = 16,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if rank <= 0 or summary_tokens <= 0:
            raise ValueError("Qwen partition global rank and summary tokens must be positive")
        self.rank = int(rank)
        self.summary_tokens = int(summary_tokens)
        input_dim = int(latent_channels) + 2
        self.key = nn.Linear(input_dim, self.rank, bias=False, dtype=dtype)
        self.value = nn.Linear(input_dim, self.rank, bias=False, dtype=dtype)
        self.query = nn.Linear(input_dim, self.rank, bias=False, dtype=dtype)
        self.summary_queries = nn.Parameter(
            torch.randn(self.summary_tokens, self.rank, dtype=dtype) / self.rank**0.5
        )
        self.out = nn.Linear(self.rank, int(hidden_dim), bias=False, dtype=dtype)
        nn.init.zeros_(self.out.weight)

    @staticmethod
    def _features(grid: torch.Tensor, box: PartitionBox | None = None) -> torch.Tensor:
        batch, height, width, _ = grid.shape
        y = torch.linspace(-1, 1, height, device=grid.device, dtype=grid.dtype)
        x = torch.linspace(-1, 1, width, device=grid.device, dtype=grid.dtype)
        yy = y[:, None].expand(height, width)
        xx = x[None, :].expand(height, width)
        coords = torch.stack((yy, xx), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        features = torch.cat((grid, coords), dim=-1)
        if box is not None:
            features = features[:, box.top : box.bottom, box.left : box.right]
        return features.reshape(batch, -1, features.shape[-1])

    def forward(self, full_grid: torch.Tensor, box: PartitionBox) -> torch.Tensor:
        full = self._features(full_grid)
        local = self._features(full_grid, box)
        if not torch.is_autocast_enabled(full.device.type):
            parameter_dtype = self.key.weight.dtype
            full = full.to(parameter_dtype)
            local = local.to(parameter_dtype)
        keys = self.key(full)
        values = self.value(full)
        queries = self.query(local)
        context = global_summary_attention(self.summary_queries, keys, values, queries)
        return self.out(context)

    def branch_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "key.weight": self.key.weight,
            "value.weight": self.value.weight,
            "query.weight": self.query.weight,
            "summary_queries": self.summary_queries,
            "out.weight": self.out.weight,
        }

    def trainable_parameters(self):
        return iter(self.parameters())

    def export_tensors(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.branch_tensors().items()}

    def load_tensors(self, tensors: Mapping[str, torch.Tensor]) -> None:
        for name, parameter in self.branch_tensors().items():
            value = tensors.get(name)
            if value is not None:
                parameter.data.copy_(value)

    def spec_constants(self) -> tuple[str, ...]:
        return ()


GLOBAL_ADAPTER_NAME = "qwen_partition_global_adapter"
GLOBAL_ADAPTER_VERSION = "latent_summary_v1"


def global_adapter_metadata(adapter: QwenPartitionGlobalAdapter | None, training_config=None) -> dict[str, str]:
    if adapter is None:
        return {}
    from api.param_defaults import QWEN_IMAGE_21_GENERATION_DEFAULTS, TRAINING_DEFAULTS
    metadata = {
        GLOBAL_ADAPTER_NAME: GLOBAL_ADAPTER_VERSION,
        "qwen_partition_global_rank": str(adapter.rank),
        "qwen_partition_global_tokens": str(adapter.summary_tokens),
    }
    if training_config is not None:
        metadata["qwen_partition_global_generation_halo_tokens"] = str(
            QWEN_IMAGE_21_GENERATION_DEFAULTS["qwen_image_21_partition"]["halo_tokens"])
        halo = training_config.get("dit_partition_halo_tokens")
        if halo is None:
            halo = training_config.get("qwen_partition_halo_tokens")
        if halo is None:
            halo = TRAINING_DEFAULTS["dit_partition_halo_tokens"]
        metadata["qwen_partition_global_training_halo_tokens"] = str(halo)
    return metadata


def read_partition_global_payload(model, state, metadata, *, allow_alpha=False):
    """Validate the payload without mutating the model or checkpoint."""
    prefix = f"{GLOBAL_ADAPTER_NAME}."
    tensors = {key[len(prefix):]: value for key, value in state.items()
               if key.startswith(prefix)}
    version = metadata.get(GLOBAL_ADAPTER_NAME)
    if not tensors and not version:
        return None
    if allow_alpha:
        tensors.pop("alpha", None)
    if version != GLOBAL_ADAPTER_VERSION:
        raise ValueError(f"Unsupported Qwen partition global adapter version: {version!r}")
    try:
        rank = int(metadata["qwen_partition_global_rank"])
        tokens = int(metadata["qwen_partition_global_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Qwen partition global adapter metadata is incomplete") from exc
    with torch.device("meta"):
        spec = QwenPartitionGlobalAdapter(
            int(model.config.in_channels), int(model.inner_dim),
            rank=rank, summary_tokens=tokens,
        )
    targets = spec.branch_tensors()
    if set(tensors) != set(targets):
        raise ValueError("Qwen partition global adapter checkpoint tensors are incomplete or unexpected")
    for key, value in tensors.items():
        if value.shape != targets[key].shape or not value.is_floating_point():
            raise ValueError(f"Invalid Qwen partition global adapter tensor {key!r}")
    return {
        "state": tensors, "metadata": {**metadata, **global_adapter_metadata(spec)},
    }


def stash_partition_global_payload(model, state, metadata) -> None:
    """Keep validated weights on CPU outside the ordinary transformer module tree."""
    payload = read_partition_global_payload(model, state, metadata)
    if payload is not None:
        model._qwen_partition_global_payload = payload
        for key in payload["state"]:
            del state[f"{GLOBAL_ADAPTER_NAME}.{key}"]


def prepare_training_global_adapter(trainer, *, dtype: torch.dtype):
    config = getattr(trainer, "config", {}) or {}
    enabled = bool(config.get(
        "dit_partition_global_adapter_enabled",
        config.get("qwen_partition_global_adapter_enabled", False),
    ))
    model = trainer.transformer
    payload = getattr(model, "_qwen_partition_global_payload", None)
    existing = getattr(model, GLOBAL_ADAPTER_NAME, None)
    if not enabled:
        if payload is not None or existing is not None:
            raise ValueError("Qwen checkpoint contains a global adapter, but training has it disabled")
        return None
    if not getattr(trainer, "train_unet", True):
        raise ValueError("Qwen partition global adapter requires train_unet=true")
    if not config.get(
        "dit_partition_training_enabled", config.get("qwen_partition_training_enabled", False)
    ):
        raise ValueError("Qwen partition global adapter requires dit_partition_training_enabled")
    if payload is not None and not getattr(trainer, "trains_base_weights", False):
        raise ValueError(
            "LoRA training on a full-FT global-adapter base needs an additive global "
            "adapter contract; use full FT or a base without global weights"
        )
    from api.param_defaults import TRAINING_DEFAULTS
    rank = int(config.get("dit_partition_global_rank", config.get(
        "qwen_partition_global_rank", TRAINING_DEFAULTS["dit_partition_global_rank"])))
    tokens = int(config.get("dit_partition_global_tokens", config.get(
        "qwen_partition_global_tokens", TRAINING_DEFAULTS["dit_partition_global_tokens"])))
    if existing is not None:
        if (existing.rank, existing.summary_tokens) != (rank, tokens):
            raise ValueError("Qwen global adapter dimensions changed from the existing module")
        return existing.requires_grad_(True)
    if payload is not None:
        expected = {"qwen_partition_global_rank": str(rank), "qwen_partition_global_tokens": str(tokens)}
        if any(payload["metadata"].get(key) != value for key, value in expected.items()):
            raise ValueError("Qwen global adapter dimensions changed from the saved checkpoint")
    elif (getattr(trainer, "trains_base_weights", False)
          and getattr(trainer, "_loaded_checkpoint_path", None)):
        raise ValueError(
            "Cannot resume Qwen full FT with a global adapter absent from its checkpoint; "
            "start a new run to add one"
        )
    adapter = QwenPartitionGlobalAdapter(
        int(model.config.in_channels), int(model.inner_dim),
        rank=rank, summary_tokens=tokens, dtype=dtype,
    ).to(next(model.parameters()).device)
    if payload is not None:
        adapter.load_tensors(payload["state"])
        del model._qwen_partition_global_payload
    model.qwen_partition_global_adapter = adapter
    return adapter
