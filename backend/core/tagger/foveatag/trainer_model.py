"""Native model protocol for the shared tagger trainer."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

from torch import nn
from safetensors.torch import load_file

from ..siglip2_tagger_model import LoRALinear, _save_file_safe
from .batch import FoveaViewTensorBatch
from .encoder import FoveaEncoderConfig, SpatialAttention
from .model import FoveaTagger


class FoveaTrainerModel(nn.Module):
    def __init__(
        self, num_tags: int, *, encoder_config: FoveaEncoderConfig,
        feature_dim: int, evidence_slots: int, pool_heads: int, temperature: float,
        training_method: str, lora_rank: int, lora_alpha: float,
    ) -> None:
        super().__init__()
        if training_method not in ("full", "lora"):
            raise ValueError("FoveaTag supports full or lora training")
        if training_method == "lora" and (lora_rank <= 0 or lora_alpha <= 0):
            raise ValueError("LoRA rank and alpha must be positive")
        self.core = FoveaTagger(
            num_tags, encoder_config=encoder_config, feature_dim=feature_dim,
            evidence_slots_per_view=evidence_slots, pool_heads=pool_heads,
            temperature=temperature,
        )
        self.training_method = training_method
        self.feature_dim = feature_dim
        self.pool_heads = pool_heads
        self.temperature = temperature
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self._lora_modules = {}
        if training_method == "lora":
            for parameter in self.core.encoder.parameters():
                parameter.requires_grad = False
            for name, module in self.core.encoder.named_modules():
                if isinstance(module, SpatialAttention):
                    for target in ("qkv", "output"):
                        adapter = LoRALinear(getattr(module, target), lora_rank, lora_alpha)
                        setattr(module, target, adapter)
                        self._lora_modules[f"{name}.{target}"] = adapter

    @property
    def head(self):
        return self.core.head

    @property
    def vision_encoder(self):
        return self.core.encoder

    def forward(self, views: FoveaViewTensorBatch, pixel_attention_mask=None, spatial_shapes=None):
        if not isinstance(views, FoveaViewTensorBatch):
            raise TypeError("FoveaTag requires native coordinate-aware view tensors")
        return self.core(
            views.overview_images, views.overview_geometry,
            views.local_images, views.local_geometry,
            local_view_mask=views.local_view_mask,
        )

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def expand_head(self, new_num_tags):
        return self.core.expand_head(new_num_tags)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = {
            (f"core.evidence_head.{key}" if key.startswith("head.") else key): value
            for key, value in state_dict.items()
        }
        return super().load_state_dict(mapped, strict=strict, assign=assign)

    def load_weights_inplace(self, path):
        self.load_state_dict(load_file(str(path), device="cpu"), strict=True)

    def save_checkpoint(self, output_dir, name, metadata=None):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.safetensors"
        state = {
            (key.removeprefix("core.evidence_head.") if key.startswith("core.evidence_head.head.") else key):
            value.detach().cpu().contiguous()
            for key, value in self.state_dict().items()
        }
        _save_file_safe(state, str(path))
        document = dict(metadata or {})
        document.update({
            "architecture": "foveatag", "kind": "foveatag_shared_trainer_weights",
            "training_method": self.training_method,
            "encoder_config": asdict(self.core.encoder.config),
            "feature_dim": self.feature_dim,
            "pool_heads": self.pool_heads, "temperature": self.temperature,
            "evidence_slots_per_view": self.core.evidence_slots_per_view,
            "lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha,
            "contains_frozen_base": True,
        })
        metadata_path = directory / f"{name}_metadata.json"
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
        os.replace(temporary, metadata_path)
        return str(path)
