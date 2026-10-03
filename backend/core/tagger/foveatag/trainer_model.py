"""Native model protocol for the shared tagger trainer."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file

from ..siglip2_tagger_model import LoRALinear, _save_file_safe
from .batch import FoveaViewTensorBatch
from .encoder import FoveaEncoderConfig, SpatialAttention
from .model import FoveaTagger


class FoveaTrainerModel(nn.Module):
    architecture = "foveatag"

    def __init__(
        self, num_tags: int, *, encoder_config: FoveaEncoderConfig,
        feature_dim: int, evidence_slots: int, pool_heads: int, temperature: float,
        training_method: str, lora_rank: int, lora_alpha: float,
        encoder_path: str | None = None,
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
        self._ood_tap = nn.Identity()
        if encoder_path:
            from .pretraining_checkpoint import load_pretrained_encoder
            load_pretrained_encoder(encoder_path, self.core.encoder)
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
        capture = bool(self._ood_tap._forward_pre_hooks)
        result = self.core(
            views.overview_images, views.overview_geometry,
            views.local_images, views.local_geometry,
            local_view_mask=views.local_view_mask,
            return_details=capture,
        )
        if not capture:
            return result
        evidence = result.head_output.mixed_evidence
        mask = views.local_view_mask.repeat_interleave(self.core.evidence_slots_per_view, dim=1)
        local_mean = (evidence * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        self._ood_tap(torch.cat((result.global_features, local_mean), dim=1).detach())
        return result.logits

    def register_ood_feature_hook(self, hook):
        return self._ood_tap.register_forward_pre_hook(hook)

    def replace_head(self, head):
        if head.in_features != self.head.in_features:
            raise ValueError("FoveaTag head input dimension must not change during inheritance")
        self.core.evidence_head.head = head
        self.core.evidence_head.num_tags = head.out_features

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
        mode = (metadata or {}).get("checkpoint_save_mode", "full")
        if mode not in ("full", "lora", "merged"):
            raise ValueError("unsupported native checkpoint save mode")
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
        document.pop("inference_export", None)
        document.update({
            "format_version": 1,
            "architecture": "foveatag", "kind": "foveatag_shared_trainer_weights",
            "training_method": self.training_method,
            "encoder_config": asdict(self.core.encoder.config),
            "feature_dim": self.feature_dim,
            "pool_heads": self.pool_heads, "temperature": self.temperature,
            "evidence_slots_per_view": self.core.evidence_slots_per_view,
            "lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha,
            "contains_frozen_base": True,
            "ood_feature_definition": "overview_plus_masked_mean_mixed_evidence_v1",
            "resume_weight_mode": "full_unmerged",
        })
        if self.training_method == "lora" and mode != "full":
            export_path = directory / f"{name}_export.safetensors"
            export_state = self.export_state(merged=mode == "merged")
            _save_file_safe(export_state, str(export_path))
            export_metadata = dict(document)
            export_metadata.pop("resume_weight_mode", None)
            export_metadata.update({
                "kind": "foveatag_inference_export", "export_mode": mode,
                "export_training_method": "full" if mode == "merged" else "lora",
                "contains_frozen_base": mode == "merged",
                "vocabulary_file": f"{name}_vocabulary.json",
            })
            if mode == "lora":
                export_metadata["frozen_encoder_tensor_sha256"] = self.frozen_encoder_fingerprint()
            _write_metadata(directory / f"{name}_export_metadata.json", export_metadata)
            document["inference_export"] = export_path.name
        metadata_path = directory / f"{name}_metadata.json"
        _write_metadata(metadata_path, document)
        return str(path)

    def frozen_encoder_state(self):
        state = dict(self.vision_encoder.state_dict())
        for name, adapter in self._lora_modules.items():
            state.pop(f"{name}.lora_A")
            state.pop(f"{name}.lora_B")
            state[f"{name}.weight"] = state.pop(f"{name}.base.weight")
            if adapter.base.bias is not None:
                state[f"{name}.bias"] = state.pop(f"{name}.base.bias")
        return state

    def frozen_encoder_fingerprint(self):
        digest = hashlib.sha256()
        for key, value in sorted(self.frozen_encoder_state().items()):
            value = value.detach().cpu().contiguous()
            digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode("utf-8"))
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def export_state(self, *, merged):
        trainable = {key for key, parameter in self.named_parameters() if parameter.requires_grad}
        state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in self.state_dict().items()
            if merged or key in trainable
        }
        if merged:
            for name, adapter in self._lora_modules.items():
                prefix = f"core.encoder.{name}"
                delta = (state.pop(f"{prefix}.lora_A") @ state.pop(f"{prefix}.lora_B")).T * adapter.scale
                state[f"{prefix}.weight"] = state.pop(f"{prefix}.base.weight") + delta
                if adapter.base.bias is not None:
                    state[f"{prefix}.bias"] = state.pop(f"{prefix}.base.bias")
        return {
            (key.removeprefix("core.evidence_head.") if key.startswith("core.evidence_head.head.") else key): value
            for key, value in state.items()
        }


def load_native_checkpoint(path, *, base_encoder_path=None):
    """Load model-only artifacts without requiring optimizer/resume files."""
    path = Path(path)
    metadata = json.loads(path.with_name(f"{path.stem}_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") == "foveatag_inference_export":
        return load_inference_export(path, base_encoder_path=base_encoder_path)
    if (metadata.get("format_version") != 1 or metadata.get("architecture") != "foveatag"
            or metadata.get("kind") != "foveatag_shared_trainer_weights"
            or metadata.get("resume_weight_mode") != "full_unmerged"
            or metadata.get("contains_frozen_base") is not True):
        raise ValueError("not a supported self-contained native FoveaTag checkpoint")
    config = dict(metadata["encoder_config"])
    for key in ("widths", "depths", "attention_heads"):
        config[key] = tuple(config[key])
    state = load_file(str(path), device="cpu")
    instance = FoveaTrainerModel(
        state["head.weight"].shape[0], encoder_config=FoveaEncoderConfig(**config),
        feature_dim=metadata["feature_dim"], evidence_slots=metadata["evidence_slots_per_view"],
        pool_heads=metadata["pool_heads"], temperature=metadata["temperature"],
        training_method=metadata["training_method"], lora_rank=metadata["lora_rank"],
        lora_alpha=metadata["lora_alpha"],
    )
    instance.load_state_dict(state, strict=True)
    return instance.eval(), metadata


def load_inference_export(path, *, base_encoder_path=None):
    path = Path(path)
    metadata = json.loads(path.with_name(f"{path.stem}_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("kind") != "foveatag_inference_export":
        raise ValueError("not a supported native FoveaTag inference export")
    mode = metadata.get("export_mode")
    if mode not in ("lora", "merged"):
        raise ValueError("unsupported native export mode")
    if mode == "lora" and not base_encoder_path:
        raise ValueError("compact native export requires its scratch backbone")
    config = dict(metadata["encoder_config"])
    for key in ("widths", "depths", "attention_heads"):
        config[key] = tuple(config[key])
    state = load_file(str(path), device="cpu")
    instance = FoveaTrainerModel(
        state["head.weight"].shape[0], encoder_config=FoveaEncoderConfig(**config),
        feature_dim=metadata["feature_dim"], evidence_slots=metadata["evidence_slots_per_view"],
        pool_heads=metadata["pool_heads"], temperature=metadata["temperature"],
        training_method="lora" if mode == "lora" else "full",
        lora_rank=metadata["lora_rank"], lora_alpha=metadata["lora_alpha"],
        encoder_path=str(base_encoder_path) if mode == "lora" else None,
    )
    if mode == "lora":
        if instance.frozen_encoder_fingerprint() != metadata.get("frozen_encoder_tensor_sha256"):
            raise ValueError("compact native export scratch backbone fingerprint differs")
        if set(state) != set(instance.export_state(merged=False)):
            raise ValueError("compact native export trainable keys differ")
    instance.load_state_dict(state, strict=mode == "merged")
    return instance.eval(), metadata


def _write_metadata(path, document):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(temporary, path)
