"""Checked native construction, separate from the frozen A/B/C runner."""

from __future__ import annotations

import json
from pathlib import Path

from .artifact import FoveaArtifactConfig
from .encoder import FoveaEncoderConfig
from .processor import FoveaProcessor
from .trainer_model import FoveaTrainerModel
from .views import ViewConfig


def view_config(config):
    return ViewConfig(
        overview_size=config["fovea_overview_size"],
        local_size=config["fovea_local_size"], local_views=config["fovea_local_views"],
    )


def build_processor(config, *, training):
    return FoveaProcessor(view_config(config), seed=config["fovea_seed"], training=training)


def validate_config(config, *, resuming=False):
    if config.get("cls_dim") is not None or config.get("hidden_proj_dim") is not None:
        raise ValueError("SigLIP2 pooling dimensions do not apply to FoveaTag; use fovea_feature_dim")
    if config["training_method"] not in ("full", "lora"):
        raise ValueError("FoveaTag supports full or lora training")
    if config["checkpoint_save_mode"] not in ("full", "lora", "merged"):
        raise ValueError("unsupported native checkpoint save mode")
    if config["mixed_precision"].lower() not in ("no", "fp32", "fp16", "bf16"):
        raise ValueError("unsupported native mixed precision")
    base_path = str(config.get("vision_encoder_path") or "").strip().strip('"').strip("'")
    if config["training_method"] == "lora" and not base_path and not resuming:
        raise ValueError("native LoRA requires a scratch-pretrained backbone; a random frozen encoder is refused")
    if base_path:
        from ..siglip2_tagger_model import _is_hf_repo_or_url
        is_repo, _ = _is_hf_repo_or_url(base_path)
        if is_repo:
            raise ValueError("FoveaTag requires a local scratch artifact, not a SigLIP2 repository or URL")
    view_config(config)


def build_model(config, num_tags, *, vocabulary, lineage, resume_path=None):
    validate_config(config, resuming=resume_path is not None)
    base_path = str(config.get("vision_encoder_path") or "").strip().strip('"').strip("'")
    method = config["training_method"]
    encoder_config = FoveaEncoderConfig()
    if resume_path is not None:
        metadata = json.loads(Path(resume_path).with_name(
            f"{Path(resume_path).stem}_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("kind") != "foveatag_shared_trainer_weights":
            raise ValueError("resume weights are not a native FoveaTag checkpoint")
        for key, expected in (("training_method", method), ("feature_dim", config["fovea_feature_dim"])):
            if metadata.get(key) != expected:
                raise ValueError(f"native resume configuration differs: {key}")
        if method == "lora" and any(
            metadata.get(key) != config[key] for key in ("lora_rank", "lora_alpha")
        ):
            raise ValueError("native resume LoRA configuration differs")
        value = metadata["encoder_config"]
        base_path = ""
    elif base_path:
        metadata = json.loads(Path(base_path).with_name(
            f"{Path(base_path).stem}_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("kind") != "foveatag_scratch_encoder":
            raise ValueError("FoveaTag initialization requires a scratch encoder artifact, not SigLIP2")
        value = metadata["run_contract"]["encoder"]
    else:
        if method == "lora":
            raise ValueError("native LoRA requires a scratch-pretrained backbone; a random frozen encoder is refused")
        value = None
    if value is not None:
        value = dict(value)
        for key in ("widths", "depths", "attention_heads"):
            value[key] = tuple(value[key])
        encoder_config = FoveaEncoderConfig(**value)
    architecture = FoveaArtifactConfig(num_tags=num_tags)
    model = FoveaTrainerModel(
        num_tags, encoder_config=encoder_config, feature_dim=config["fovea_feature_dim"],
        evidence_slots=architecture.evidence_slots_per_view,
        pool_heads=architecture.pool_heads, temperature=architecture.temperature,
        training_method=method, lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"],
        encoder_path=base_path or None,
    )
    flash = config["use_flash_attention"]
    if flash:
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except ImportError as exc:
            raise ValueError("FoveaTag FlashAttention requires an importable flash-attn installation") from exc
    model.vision_encoder.set_sdpa(True, allow_flash=flash, external_flash=flash)
    source = config.get("init_head_from")
    if source and resume_path is None:
        source_metadata = json.loads(Path(source).with_name(
            f"{Path(source).stem}_metadata.json").read_text(encoding="utf-8"))
        if (source_metadata.get("kind") != "foveatag_shared_trainer_weights"
                or source_metadata.get("feature_dim") != model.feature_dim):
            raise ValueError("head inheritance requires a compatible native FoveaTag checkpoint")
        from ..siglip2_tagger_model import _inherit_head
        _inherit_head(model, source, num_tags, new_vocab=vocabulary, lineage=lineage)
    return model
