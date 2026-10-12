"""Full-image summaries injected into Anima's native patch embeddings."""
import torch
from torch import nn

from core.models.components.global_summary_attention import global_summary_attention
from core.training.partition import PartitionBox

NAME = "anima_partition_global_adapter"
METADATA_KEY = "sushi.anima.partition_global"


class AnimaPartitionGlobalAdapter(nn.Module):
    def __init__(self, latent_channels, hidden_dim, *, rank, summary_tokens, dtype=torch.float32):
        super().__init__()
        if any(type(value) is not int or value < 1
               for value in (latent_channels, hidden_dim, rank, summary_tokens)):
            raise ValueError("Anima global adapter dimensions must be positive integers")
        if rank > 512 or summary_tokens > 256:
            raise ValueError("Anima global adapter exceeds rank/token bounds")
        self.latent_channels, self.hidden_dim = latent_channels, hidden_dim
        self.rank, self.summary_tokens = rank, summary_tokens
        features = 4 * latent_channels + 2
        self.key = nn.Linear(features, rank, bias=False, dtype=dtype)
        self.value = nn.Linear(features, rank, bias=False, dtype=dtype)
        self.query = nn.Linear(features, rank, bias=False, dtype=dtype)
        self.summary_queries = nn.Parameter(torch.randn(summary_tokens, rank, dtype=dtype) / rank**.5)
        self.out = nn.Linear(rank, hidden_dim, bias=False, dtype=dtype)
        nn.init.zeros_(self.out.weight)

    def _features(self, latents):
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels or latents.shape[2] != 1:
            raise ValueError("Anima global summaries require native [B,C,1,H,W] image latents")
        batch, channels, _, height, width = latents.shape
        if min(batch, height, width) < 1 or height % 2 or width % 2:
            raise ValueError("Anima global summaries require nonempty patch-aligned images")
        h, w = height // 2, width // 2
        # Use the same channel/patch ordering as Anima.PatchEmbed.
        grid = latents.reshape(batch, channels, 1, h, 2, w, 2)
        grid = grid.permute(0, 2, 3, 5, 1, 4, 6).reshape(batch, h, w, 4 * channels)
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=latents.device, dtype=latents.dtype),
                             torch.linspace(-1, 1, w, device=latents.device, dtype=latents.dtype), indexing="ij")
        coordinates = torch.stack((y, x), dim=-1).expand(batch, -1, -1, -1)
        return torch.cat((grid, coordinates), dim=-1)

    def forward(self, latents, box=None):
        grid = self._features(latents)
        batch, height, width, _ = grid.shape
        box = PartitionBox(0, 0, height, width) if box is None else box
        if not isinstance(box, PartitionBox) or not (
            0 <= box.top < box.bottom <= height and 0 <= box.left < box.right <= width
        ):
            raise ValueError("Anima global adapter region lies outside the full patch canvas")
        full = grid.flatten(1, 2)
        local = grid[:, box.top:box.bottom, box.left:box.right].flatten(1, 2)
        if not torch.is_autocast_enabled(full.device.type):
            full, local = full.to(self.key.weight.dtype), local.to(self.key.weight.dtype)
        keys, values = self.key(full), self.value(full)
        queries = self.query(local)
        result = self.out(global_summary_attention(self.summary_queries, keys, values, queries))
        return result.reshape(batch, 1, box.height, box.width, self.hidden_dim)

    def metadata(self):
        return dict(schema_version=1, algorithm="anima_partition_global", latent_channels=self.latent_channels,
                    hidden_dim=self.hidden_dim, rank=self.rank, summary_tokens=self.summary_tokens,
                    target_patch_size=2, coordinate_axes=["height", "width"])


def install(model, declaration, *, dtype=None):
    if declaration is None:
        return None
    if not isinstance(declaration, dict):
        raise ValueError("Anima global adapter declaration must be an object")
    adapter = AnimaPartitionGlobalAdapter(model.in_channels, model.model_channels,
        rank=declaration.get("rank"), summary_tokens=declaration.get("summary_tokens"),
        dtype=dtype or next(model.parameters()).dtype)
    if declaration != adapter.metadata():
        raise ValueError("Anima global adapter declaration conflicts with native model geometry")
    if model.patch_spatial != 2 or model.patch_temporal != 1:
        raise ValueError("Anima global adapter requires native image patch geometry")
    model.add_module(NAME, adapter)
    return adapter


def metadata(model):
    import json
    adapter = getattr(model, NAME, None)
    return {} if adapter is None else {METADATA_KEY: json.dumps(adapter.metadata(), sort_keys=True)}


def validate_loaded(model, missing, unexpected):
    prefix = NAME + "."
    if any(key.startswith(prefix) for key in (*missing, *unexpected)):
        raise ValueError("Anima global adapter declared weights are missing or unexpected")
    adapter = getattr(model, NAME, None)
    if adapter is not None and any(not torch.isfinite(value).all() for value in adapter.parameters()):
        raise ValueError("Anima global adapter weights must be finite")


def configure_training(model, config):
    from api.param_defaults import TRAINING_DEFAULTS
    existing = getattr(model, NAME, None)
    if not config.get("dit_partition_global_adapter_enabled", TRAINING_DEFAULTS["dit_partition_global_adapter_enabled"]):
        return existing
    rank = config.get("dit_partition_global_rank", TRAINING_DEFAULTS["dit_partition_global_rank"])
    tokens = config.get("dit_partition_global_tokens", TRAINING_DEFAULTS["dit_partition_global_tokens"])
    if existing is not None:
        if (existing.rank, existing.summary_tokens) != (rank, tokens):
            raise ValueError("Changing a restored Anima global adapter requires a fresh base")
        return existing
    declaration = dict(schema_version=1, algorithm="anima_partition_global", latent_channels=model.in_channels,
        hidden_dim=model.model_channels, rank=rank, summary_tokens=tokens,
        target_patch_size=2, coordinate_axes=["height", "width"])
    return install(model, declaration)
