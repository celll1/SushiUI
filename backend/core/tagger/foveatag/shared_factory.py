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
    learned = config.get("fovea_learned_routing", False)
    return ViewConfig(
        overview_size=config["fovea_overview_size"],
        local_size=config["fovea_local_size"], local_views=(config["fovea_candidate_count"] if learned else config["fovea_local_views"]),
        overview_resize_mode="stretch" if learned else "letterbox",
        local_border_mode="shift_inside" if learned else "pad",
    )


def build_processor(config, *, training):
    return FoveaProcessor(view_config(config), seed=config["fovea_seed"], training=training,
                          max_source_pixels=config["fovea_max_source_pixels"])


def save_initial_base(model, config, output_dir):
    """Keep the actual scratch initialization when no input encoder file exists."""
    from dataclasses import asdict
    from .pretraining_checkpoint import save_pretrained_encoder
    path = Path(output_dir) / "base_model.safetensors"
    if not config["save_base_model"]:
        return
    if not path.is_file():
        if model.training_method != "full":
            raise ValueError("native LoRA base snapshot must be copied from its scratch encoder")
        save_pretrained_encoder(path, model.vision_encoder, run_contract={
            "encoder": asdict(model.vision_encoder.config), "source_stage": "tagger_initialization",
            "external_weights": []}, epoch=0, global_step=0, validation={})
    config["base_model_path"] = path.name


def validate_config(config, *, resuming=False):
    if config.get("cls_dim") is not None or config.get("hidden_proj_dim") is not None:
        raise ValueError("SigLIP2 pooling dimensions do not apply to FoveaTag; use fovea_feature_dim")
    if config["training_method"] not in ("full", "lora"):
        raise ValueError("FoveaTag supports full or lora training")
    if config["checkpoint_save_mode"] not in ("full", "lora", "merged"):
        raise ValueError("unsupported native checkpoint save mode")
    if config["mixed_precision"].lower() not in ("no", "fp32", "fp16", "bf16"):
        raise ValueError("unsupported native mixed precision")
    if not 0 <= config["train_f1_initial_threshold"] <= 1:
        raise ValueError("native F1 threshold must be a probability in [0, 1]")
    base_path = str(config.get("vision_encoder_path") or "").strip().strip('"').strip("'")
    if config["training_method"] == "lora" and not base_path and not resuming:
        raise ValueError("native LoRA requires a scratch-pretrained backbone; a random frozen encoder is refused")
    if base_path:
        from ..siglip2_tagger_model import _is_hf_repo_or_url
        is_repo, _ = _is_hf_repo_or_url(base_path)
        if is_repo:
            raise ValueError("FoveaTag requires a local scratch artifact, not a SigLIP2 repository or URL")
    view_config(config)
    if config.get("fovea_learned_routing") and not (
            0 <= config["fovea_random_count"] < config["fovea_local_views"] <= config["fovea_candidate_count"]):
        raise ValueError("router requires 0 <= random < local views <= candidates")


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
    routing = None
    if config.get("fovea_learned_routing"):
        routing = {"candidate_count": config["fovea_candidate_count"], "selection_budget": config["fovea_local_views"],
            "random_count": config["fovea_random_count"], "oracle_interval": config["fovea_oracle_interval"],
            "oracle_chunk_size": config["fovea_oracle_chunk_size"], "router_weight": config["fovea_router_weight"],
            "routing_dim": architecture.routing_dim, "routing_slots": architecture.routing_slots}
    if resume_path is not None and metadata.get("routing") != routing:
        raise ValueError("native routing settings differ from resume checkpoint")
    model = FoveaTrainerModel(
        num_tags, encoder_config=encoder_config, feature_dim=config["fovea_feature_dim"],
        evidence_slots=architecture.evidence_slots_per_view,
        pool_heads=architecture.pool_heads, temperature=architecture.temperature,
        training_method=method, lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"],
        encoder_path=base_path or None,
        routing=routing,
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
