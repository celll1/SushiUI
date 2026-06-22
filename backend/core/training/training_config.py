"""
Training configuration generator for ai-toolkit.

Generates YAML configuration files based on training parameters.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml

from core.training.dataset_params import extract_dataset_params


# Legacy kwarg names that are renamed in the new dict-based API.
# Old name → new name (matches Pydantic field name).
_LEGACY_KEY_MAP = {
    "timestep_sampling_config": "timestep_sampling",
}


def _normalize_params(p: Dict[str, Any]) -> Dict[str, Any]:
    """Apply legacy key renames so old kwargs work with the new dict API."""
    p = dict(p)  # Don't mutate caller's dict
    for old_key, new_key in _LEGACY_KEY_MAP.items():
        if old_key in p:
            p.setdefault(new_key, p.pop(old_key))
    return p


def _build_train_section(
    p: Dict[str, Any],
    *,
    total_steps: Optional[int],
    epochs: Optional[int],
    train_unet: bool,
    train_text_encoder: bool,
    train_image_encoder: bool = False,
    learning_rate: Optional[float] = None,
    component_lr_always_emit: bool = False,
    bucketing_always_emit: bool = False,
    include_block_swap: bool = True,
    include_vision_encoder: bool = True,
    include_param_tracking: bool = True,
    include_reference_images: bool = True,
    include_priority_training: bool = True,
    include_image_encoder_lr: bool = True,
    include_te_split_lrs: bool = True,
) -> Dict[str, Any]:
    """Build the 'train' section dict shared across all generate_*_config functions.

    p: dict of training parameters (TrainingRunCreateRequest model_dump or kwargs).
    learning_rate: explicit learning rate (defaults to p["learning_rate"]).
    component_lr_always_emit: if True, emit unet_lr/text_encoder_lr always with fallback to lr.
    bucketing_always_emit: if True, emit base_resolutions/bucket_strategy/multi_resolution_mode always.

    The flags `include_*` allow each training method to opt in/out of optional sections.
    """
    lr = learning_rate if learning_rate is not None else p.get("learning_rate", 1e-4)
    train: Dict[str, Any] = {
        "batch_size": p.get("batch_size", 1),
        **({"steps": total_steps} if total_steps else {"epochs": epochs}),
        "gradient_accumulation_steps": p.get("gradient_accumulation_steps", 1),
        "max_grad_norm": p.get("max_grad_norm", 1.0),
        "train_unet": train_unet,
        "train_text_encoder": train_text_encoder,
        "train_image_encoder": train_image_encoder,
        # Unified training framework
        "noise_process": p.get("noise_process", "auto"),
        "prediction_target": p.get("prediction_target", "auto"),
        "strict_validation": p.get("strict_validation", False),
        # Optimizer
        "optimizer": p.get("optimizer", "adamw8bit"),
        "lr": lr,
        "lr_scheduler": p.get("lr_scheduler", "constant"),
    }

    # Conditional optimizer fields
    if p.get("lr_warmup_steps", 0) > 0:
        train["lr_warmup_steps"] = p["lr_warmup_steps"]
    if p.get("optimizer_is_paged"):
        train["optimizer_is_paged"] = p["optimizer_is_paged"]
    if p.get("optimizer_cautious"):
        train["optimizer_cautious"] = p["optimizer_cautious"]
    if p.get("optimizer_beta1") is not None:
        train["optimizer_beta1"] = p["optimizer_beta1"]
    if p.get("optimizer_beta2") is not None:
        train["optimizer_beta2"] = p["optimizer_beta2"]
    if p.get("optimizer_epsilon") is not None:
        train["optimizer_epsilon"] = p["optimizer_epsilon"]
    if p.get("optimizer_weight_decay") is not None:
        train["optimizer_weight_decay"] = p["optimizer_weight_decay"]
    if p.get("optimizer_schedule_free"):
        train["optimizer_schedule_free"] = p["optimizer_schedule_free"]
        if p.get("optimizer_schedule_free_r", 0.0) != 0.0:
            train["optimizer_schedule_free_r"] = p["optimizer_schedule_free_r"]
        if p.get("optimizer_schedule_free_weight_lr_power", 2.0) != 2.0:
            train["optimizer_schedule_free_weight_lr_power"] = p["optimizer_schedule_free_weight_lr_power"]
        if p.get("optimizer_use_radam"):
            train["optimizer_use_radam"] = p["optimizer_use_radam"]
    if p.get("optimizer_stochastic_rounding"):
        train["optimizer_stochastic_rounding"] = p["optimizer_stochastic_rounding"]

    # Component learning rates
    if component_lr_always_emit:
        # LoRA-style: always emit with fallback to learning_rate
        train["unet_lr"] = p.get("unet_lr") if p.get("unet_lr") is not None else lr
        train["text_encoder_lr"] = p.get("text_encoder_lr") if p.get("text_encoder_lr") is not None else lr
        if include_te_split_lrs:
            te_lr = p.get("text_encoder_lr") if p.get("text_encoder_lr") is not None else lr
            train["text_encoder_1_lr"] = p.get("text_encoder_1_lr") if p.get("text_encoder_1_lr") is not None else te_lr
            train["text_encoder_2_lr"] = p.get("text_encoder_2_lr") if p.get("text_encoder_2_lr") is not None else te_lr
        if include_image_encoder_lr:
            train["image_encoder_lr"] = p.get("image_encoder_lr") if p.get("image_encoder_lr") is not None else lr
    else:
        # Full FT / ControlNet style: only emit if not None
        if p.get("unet_lr") is not None:
            train["unet_lr"] = p["unet_lr"]
        if p.get("text_encoder_lr") is not None:
            train["text_encoder_lr"] = p["text_encoder_lr"]
        if include_te_split_lrs:
            if p.get("text_encoder_1_lr") is not None:
                train["text_encoder_1_lr"] = p["text_encoder_1_lr"]
            if p.get("text_encoder_2_lr") is not None:
                train["text_encoder_2_lr"] = p["text_encoder_2_lr"]
        if include_image_encoder_lr and p.get("image_encoder_lr") is not None:
            train["image_encoder_lr"] = p["image_encoder_lr"]

    # Mixed precision and dtype
    train["mixed_precision"] = p.get("mixed_precision", True)

    # Bucketing
    if bucketing_always_emit:
        train["enable_bucketing"] = p.get("enable_bucketing", False)
        train["base_resolutions"] = p.get("base_resolutions") or [1024]
        train["bucket_strategy"] = p.get("bucket_strategy", "resize")
        train["multi_resolution_mode"] = p.get("multi_resolution_mode", "max")
    else:
        if p.get("enable_bucketing"):
            train["enable_bucketing"] = True
            train["base_resolutions"] = p.get("base_resolutions") or [1024]
            train["bucket_strategy"] = p.get("bucket_strategy", "resize")
            train["multi_resolution_mode"] = p.get("multi_resolution_mode", "max")

    # Common training fields
    train["use_flash_attention"] = p.get("use_flash_attention", False)
    train["min_snr_gamma"] = p.get("min_snr_gamma", 5.0)
    train["reconstruction_loss_weight"] = p.get("reconstruction_loss_weight", 0.0)

    # Block Swap (training VRAM optimization) - LoRA/Full FT only
    if include_block_swap:
        train["blocks_to_swap"] = p.get("blocks_to_swap", 0)
        train["use_pinned_memory"] = p.get("use_pinned_memory", False)
        train["num_optimizer_groups"] = p.get("num_optimizer_groups", 0)

    # Text/Latent encoding
    train["text_encoding_mode"] = p.get("text_encoding_mode", "swap_onthefly")
    train["text_encoding_swap_interval"] = p.get("text_encoding_swap_interval", 256)
    # cpu_prefetch mode: how many batches ahead the CPU worker may pre-encode
    train["text_encoding_prefetch_depth"] = p.get("text_encoding_prefetch_depth", 4)

    # ---- Anima (Cosmos-Predict2 DiT) training knobs ----
    # Read unconditionally; non-Anima trainers ignore them via config.get().
    # SSoT: api/param_defaults.TRAINING_DEFAULTS.
    train["anima_lora_scope"] = p.get("anima_lora_scope", "attention,mlp,llm_adapter")
    train["train_llm_adapter"] = p.get("train_llm_adapter", True)
    train["anima_attn_mlp_lr_factor"] = p.get("anima_attn_mlp_lr_factor", 1.0)
    train["anima_mod_lr_factor"] = p.get("anima_mod_lr_factor", 1.0)
    train["anima_llm_adapter_lr_factor"] = p.get("anima_llm_adapter_lr_factor", 1.0)
    # Phase D memory optimisations (Anima only — other archs ignore).
    train["cpu_offload_checkpointing"] = p.get("cpu_offload_checkpointing", False)
    train["async_cpu_offload_checkpointing"] = p.get("async_cpu_offload_checkpointing", False)
    train["fp8_base_dtype"] = p.get("fp8_base_dtype", None)

    # ---- Ideogram 4 LoRA (flow-matching DiT) — other archs ignore. ----
    train["ideogram4_lora_scope"] = p.get("ideogram4_lora_scope", "attn,mlp")
    train["ideogram4_train_uncond"] = p.get("ideogram4_train_uncond", False)
    train["ideogram4_uncond_loss_weight"] = p.get("ideogram4_uncond_loss_weight", 1.0)
    train["ideogram4_lr_factor"] = p.get("ideogram4_lr_factor", 1.0)

    # ---- MiniT2I (pixel-space MM-JiT) — other archs ignore. ----
    train["minit2i_lora_scope"] = p.get("minit2i_lora_scope", "attn,mlp,txt_embed")
    train["minit2i_te_lora_scope"] = p.get("minit2i_te_lora_scope", "attn,ff")
    train["minit2i_label_drop_rate"] = p.get("minit2i_label_drop_rate", 0.1)
    train["minit2i_lr_factor"] = p.get("minit2i_lr_factor", 1.0)
    train["minit2i_flan_t5_path"] = p.get("minit2i_flan_t5_path", "")

    # ---- Online Danbooru augmentation (image-generation) ----
    # Read unconditionally; ignored when danbooru_aug_enable is False.
    # SSoT: api/param_defaults.TRAINING_DEFAULTS.
    train["danbooru_aug_enable"] = p.get("danbooru_aug_enable", False)
    train["danbooru_aug_queries"] = p.get("danbooru_aug_queries", "")
    train["danbooru_aug_weight_static"] = p.get("danbooru_aug_weight_static", 1.0)
    train["danbooru_aug_deficiency_enable"] = p.get("danbooru_aug_deficiency_enable", True)
    train["danbooru_aug_deficiency_min_count"] = p.get("danbooru_aug_deficiency_min_count", 20)
    train["danbooru_aug_deficiency_top_k"] = p.get("danbooru_aug_deficiency_top_k", 200)
    train["danbooru_aug_deficiency_manual"] = p.get("danbooru_aug_deficiency_manual", "")
    train["danbooru_aug_weight_deficiency"] = p.get("danbooru_aug_weight_deficiency", 1.0)
    train["danbooru_aug_injection_interval"] = p.get("danbooru_aug_injection_interval", 4)
    train["danbooru_aug_injection_ratio"] = p.get("danbooru_aug_injection_ratio", 1.0)
    train["danbooru_aug_min_score"] = p.get("danbooru_aug_min_score", 0)
    train["danbooru_aug_max_posts_per_query"] = p.get("danbooru_aug_max_posts_per_query", 200)
    train["danbooru_aug_api_interval"] = p.get("danbooru_aug_api_interval", 1.4)
    train["danbooru_aug_dl_speed_kbps"] = p.get("danbooru_aug_dl_speed_kbps", 500)
    train["danbooru_speed_check_enable"] = p.get("danbooru_speed_check_enable", True)
    train["danbooru_speed_degraded_kbps"] = p.get("danbooru_speed_degraded_kbps", 250)
    train["danbooru_speed_min_slow_streak"] = p.get("danbooru_speed_min_slow_streak", 8)
    train["danbooru_speed_min_slow_seconds"] = p.get("danbooru_speed_min_slow_seconds", 90)
    train["danbooru_speed_cooldown_seconds"] = p.get("danbooru_speed_cooldown_seconds", 3600)
    train["danbooru_aug_buffer_size"] = p.get("danbooru_aug_buffer_size", None)
    train["danbooru_aug_include_rating_tag"] = p.get("danbooru_aug_include_rating_tag", False)
    train["danbooru_aug_max_caption_tags"] = p.get("danbooru_aug_max_caption_tags", 0)
    train["danbooru_quality_tag_enable"] = p.get("danbooru_quality_tag_enable", False)
    train["danbooru_quality_tag_thresholds"] = p.get("danbooru_quality_tag_thresholds", "")
    train["danbooru_quality_tag_attach_negative"] = p.get("danbooru_quality_tag_attach_negative", False)
    train["danbooru_aug_shuffle_tags"] = p.get("danbooru_aug_shuffle_tags", False)
    train["danbooru_aug_shuffle_keep_first_n"] = p.get("danbooru_aug_shuffle_keep_first_n", 0)
    train["danbooru_aug_tag_dropout_rate"] = p.get("danbooru_aug_tag_dropout_rate", 0.0)
    train["danbooru_aug_tag_dropout_keep_first_n"] = p.get("danbooru_aug_tag_dropout_keep_first_n", 0)
    train["danbooru_aug_caption_dropout_rate"] = p.get("danbooru_aug_caption_dropout_rate", 0.0)
    train["danbooru_aug_keep_tokens"] = p.get("danbooru_aug_keep_tokens", 0)

    # Reference images / Vision encoder
    if include_reference_images:
        train["use_reference_images"] = p.get("use_reference_images", False)
    if include_vision_encoder:
        if p.get("vision_encoder_path"):
            train["vision_encoder_path"] = p["vision_encoder_path"]
        train["train_vision_encoder"] = p.get("train_vision_encoder", False)
        if p.get("vision_encoder_lr") is not None:
            train["vision_encoder_lr"] = p["vision_encoder_lr"]
        train["gradient_routing_ve"] = p.get("gradient_routing_ve", False)

    # Param tracking
    if include_param_tracking:
        train["param_tracking"] = p.get("param_tracking", False)
        train["param_tracking_interval"] = p.get("param_tracking_interval", 100)

    # Priority training
    if include_priority_training and p.get("priority_training"):
        train["priority_training"] = p["priority_training"]

    train["latent_encoding_mode"] = p.get("latent_encoding_mode", "swap_onthefly")
    train["latent_encoding_swap_interval"] = p.get("latent_encoding_swap_interval", 256)

    train["debug_latents"] = p.get("debug_latents", False)
    train["debug_latents_every"] = p.get("debug_latents_every", 50)

    # Multi Noise-Timestep
    train["multi_noise_timesteps"] = p.get("multi_noise_timesteps", 1)
    train["multi_noise_mode"] = p.get("multi_noise_mode", "independent")
    train["trajectory_blend_alpha"] = p.get("trajectory_blend_alpha", 0.7)
    if p.get("timestep_sampling"):
        train["timestep_sampling"] = p["timestep_sampling"]

    # Resume
    train["resume_from_checkpoint"] = p.get("resume_from_checkpoint")

    # Regularization
    if p.get("regularization_type"):
        train["regularization_type"] = p["regularization_type"]
    train["snr_regularization_weight"] = p.get("snr_regularization_weight", 0.1)
    train["snr_timestep_adaptive"] = p.get("snr_timestep_adaptive", True)
    train["snr_penalty_mode"] = p.get("snr_penalty_mode", "relu")
    train["energy_regularization_weight"] = p.get("energy_regularization_weight", 0.05)
    train["energy_timestep_adaptive"] = p.get("energy_timestep_adaptive", True)
    train["energy_penalty_mode"] = p.get("energy_penalty_mode", "abs")
    train["energy_normalize_by_pixels"] = p.get("energy_normalize_by_pixels", True)

    return train


class TrainingConfigGenerator:
    """Generate ai-toolkit YAML config from training parameters."""

    @staticmethod
    def generate_lora_config(
        p: Optional[Dict[str, Any]] = None,
        *,
        run_name: str,
        base_model_path: str,
        output_dir: str,
        dataset_path: str = "",  # Deprecated - kept for backward compatibility
        dataset_configs: Optional[List[Dict[str, Any]]] = None,
        sample_prompts: Optional[list] = None,
        caption_processing: Optional[Dict[str, Any]] = None,
        **legacy_kwargs: Any,
    ) -> str:
        """Generate LoRA training configuration YAML.

        Args:
            p: Training parameters dict (typically from TrainingRunCreateRequest.model_dump()).
               If None, kwargs are used (backward compatibility).
            run_name: Training run identifier
            base_model_path: Path to base model
            output_dir: Output directory for checkpoints
            dataset_path: Deprecated, use dataset_configs
            dataset_configs: List of dataset configurations
            sample_prompts: List of sample prompts
            caption_processing: Caption processing config (from database)
            **legacy_kwargs: Backward compatibility for old kwargs-style calls

        Returns:
            YAML configuration string
        """
        # Backward compat: if no p dict provided, build from legacy_kwargs
        if p is None:
            p = legacy_kwargs
        elif legacy_kwargs:
            # Both provided: legacy_kwargs override p (caller-supplied wins)
            p = {**p, **legacy_kwargs}
        p = _normalize_params(p)

        # Validate that either steps or epochs is provided
        total_steps = p.get("total_steps")
        epochs = p.get("epochs")
        if total_steps is None and epochs is None:
            raise ValueError("Either total_steps or epochs must be provided")
        if total_steps is not None and epochs is not None:
            raise ValueError("Cannot specify both total_steps and epochs")

        # Build datasets array
        # NOTE: caption_processing settings are NOT saved to YAML
        # They are read from the database (Dataset.caption_processing) at training time
        cache_latents_to_disk = p.get("cache_latents_to_disk", False)
        base_resolutions = p.get("base_resolutions")
        datasets_array = []
        if dataset_configs:
            for ds_config in dataset_configs:
                ds_path = ds_config.get("path", "")
                ds_dataset_id = ds_config.get("dataset_id")
                dataset_entry = {
                    "folder_path": ds_path,
                    "caption_ext": "txt",
                    **({"dataset_id": ds_dataset_id} if ds_dataset_id else {}),
                    "cache_latents_to_disk": cache_latents_to_disk,
                    "resolution": base_resolutions or [512, 768, 1024],
                }
                dataset_entry.update(extract_dataset_params(ds_config))
                datasets_array.append(dataset_entry)
        else:
            datasets_array.append({
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "cache_latents_to_disk": cache_latents_to_disk,
                "resolution": base_resolutions or [512, 768, 1024],
            })

        config = {
            "job": run_name,
            "config": {
                "name": run_name,
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": output_dir,
                        "device": "cuda:0",
                        "network": {
                            "type": "lora",
                            "linear": p.get("lora_rank") or 16,
                            "linear_alpha": p.get("lora_alpha") or 16,
                            "lora_dtype": p.get("lora_dtype") or "fp32",
                        },
                        "dtype": {
                            "weight": p.get("weight_dtype", "fp16"),
                            "training": p.get("training_dtype", "fp16"),
                            "vae": p.get("vae_dtype", "fp16"),
                            "save": p.get("output_dtype", "fp32"),
                        },
                        "save": {
                            "save_every": p.get("save_every", 100),
                            "save_every_unit": p.get("save_every_unit", "steps"),
                            "max_step_saves_to_keep": p.get("max_step_saves_to_keep") if p.get("max_step_saves_to_keep") is not None else 10,
                        },
                        "datasets": datasets_array,
                        "train": _build_train_section(
                            p,
                            total_steps=total_steps,
                            epochs=epochs,
                            train_unet=p.get("train_unet", True),
                            train_text_encoder=p.get("train_text_encoder", False),
                            train_image_encoder=p.get("train_image_encoder", False),
                            component_lr_always_emit=True,
                            bucketing_always_emit=True,
                        ),
                        "model": {
                            "name_or_path": base_model_path,
                        },
                        "sample": {
                            "sampler": p.get("sample_sampler", "euler"),
                            "schedule_type": p.get("sample_schedule_type", "sgm_uniform"),
                            "sample_every": p.get("sample_every", 100),
                            "width": p.get("sample_width", 1024),
                            "height": p.get("sample_height", 1024),
                            "prompts": sample_prompts or [],
                            "neg": "",
                            "seed": p.get("sample_seed", 42),
                            "guidance_scale": p.get("sample_cfg_scale", 7.0),
                            "sample_steps": p.get("sample_steps", 28),
                        },
                        "prompt_chunking_mode": p.get("prompt_chunking_mode", "a1111"),
                        "max_prompt_chunks": p.get("max_prompt_chunks", 0),
                    }
                ],
            },
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @staticmethod
    def generate_relora_config(
        p: Optional[Dict[str, Any]] = None,
        *,
        run_name: str,
        base_model_path: str,
        output_dir: str,
        dataset_path: str = "",
        dataset_configs: Optional[List[Dict[str, Any]]] = None,
        sample_prompts: Optional[list] = None,
        caption_processing: Optional[Dict[str, Any]] = None,
        **legacy_kwargs: Any,
    ) -> str:
        """Generate ReLoRA training configuration YAML.

        Generates LoRA config, then replaces network type with 'relora' and adds
        ReLoRA-specific settings (merge_every, restart_warmup_steps, etc.).
        """
        if p is None:
            p = legacy_kwargs
        elif legacy_kwargs:
            p = {**p, **legacy_kwargs}
        p = _normalize_params(p)

        # Generate base LoRA config YAML using the same params dict
        lora_yaml = TrainingConfigGenerator.generate_lora_config(
            p,
            run_name=run_name,
            base_model_path=base_model_path,
            output_dir=output_dir,
            dataset_path=dataset_path,
            dataset_configs=dataset_configs,
            sample_prompts=sample_prompts,
            caption_processing=caption_processing,
        )

        # Parse the LoRA YAML and modify for ReLoRA
        config = yaml.safe_load(lora_yaml)
        process = config["config"]["process"][0]
        process["network"]["type"] = "relora"
        process["network"]["relora"] = {
            "merge_every": p.get("relora_merge_every", 500),
            "merge_unit": p.get("relora_merge_unit", "steps"),
            "restart_warmup_steps": p.get("restart_warmup_steps", 100),
            "optimizer_reset_strategy": p.get("optimizer_reset_strategy", "full_reset"),
            "optimizer_pruning_ratio": p.get("optimizer_pruning_ratio", 0.9),
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @staticmethod
    def generate_full_finetune_config(
        p: Optional[Dict[str, Any]] = None,
        *,
        run_name: str,
        base_model_path: str,
        output_dir: str,
        dataset_path: str = "",
        dataset_configs: Optional[List[Dict[str, Any]]] = None,
        sample_prompts: Optional[list] = None,
        caption_processing: Optional[Dict[str, Any]] = None,
        **legacy_kwargs: Any,
    ) -> str:
        """Generate full fine-tuning configuration YAML.

        See generate_lora_config for argument descriptions. Differences:
        - learning_rate default 1e-6 (vs 1e-4 for LoRA)
        - train_text_encoder default True
        - max_step_saves_to_keep default 3
        - noise_process default "add_noise", strict_validation default True
        - Component LRs only emitted if not None (vs always-emit for LoRA)
        - Bucketing only emitted if enable_bucketing
        """
        if p is None:
            p = legacy_kwargs
        elif legacy_kwargs:
            p = {**p, **legacy_kwargs}
        p = _normalize_params(p)

        total_steps = p.get("total_steps")
        epochs = p.get("epochs")
        if total_steps is None and epochs is None:
            raise ValueError("Either total_steps or epochs must be provided")
        if total_steps is not None and epochs is not None:
            raise ValueError("Cannot specify both total_steps and epochs")

        cache_latents_to_disk = p.get("cache_latents_to_disk", False)
        datasets_array = []
        if dataset_configs:
            for ds_config in dataset_configs:
                ds_path = ds_config.get("path", "")
                ds_dataset_id = ds_config.get("dataset_id")
                dataset_entry = {
                    "folder_path": ds_path,
                    "caption_ext": "txt",
                    "cache_latents_to_disk": cache_latents_to_disk,
                    **({"dataset_id": ds_dataset_id} if ds_dataset_id else {}),
                }
                dataset_entry.update(extract_dataset_params(ds_config))
                datasets_array.append(dataset_entry)
        else:
            datasets_array.append({
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "cache_latents_to_disk": cache_latents_to_disk,
            })

        # Full FT defaults differ from LoRA defaults
        full_ft_defaults = {
            "learning_rate": 1e-6,
            "train_text_encoder": True,
            "noise_process": "add_noise",
            "strict_validation": True,
        }
        for k, v in full_ft_defaults.items():
            p.setdefault(k, v)

        config = {
            "job": run_name,
            "config": {
                "name": run_name,
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": output_dir,
                        "device": "cuda:0",
                        "network": {
                            "type": "full_finetune",
                        },
                        "dtype": {
                            "weight": p.get("weight_dtype", "fp16"),
                            "training": p.get("training_dtype", "fp16"),
                            "vae": p.get("vae_dtype", "fp16"),
                            "save": p.get("output_dtype", "fp32"),
                        },
                        "save": {
                            "save_every": p.get("save_every", 100),
                            "save_every_unit": p.get("save_every_unit", "steps"),
                            "max_step_saves_to_keep": p.get("max_step_saves_to_keep") if p.get("max_step_saves_to_keep") is not None else 3,
                        },
                        "datasets": datasets_array,
                        "train": _build_train_section(
                            p,
                            total_steps=total_steps,
                            epochs=epochs,
                            train_unet=p.get("train_unet", True),
                            train_text_encoder=p.get("train_text_encoder", True),
                            train_image_encoder=p.get("train_image_encoder", False),
                            component_lr_always_emit=False,
                            bucketing_always_emit=False,
                        ),
                        "model": {
                            "name_or_path": base_model_path,
                        },
                        "sample": {
                            "sampler": p.get("sample_sampler", "euler"),
                            "schedule_type": p.get("sample_schedule_type", "sgm_uniform"),
                            "sample_every": p.get("sample_every", 100),
                            "width": p.get("sample_width", 1024),
                            "height": p.get("sample_height", 1024),
                            "prompts": sample_prompts or [],
                            "neg": "",
                            "seed": p.get("sample_seed", -1),
                            "guidance_scale": p.get("sample_cfg_scale", 7.0),
                            "sample_steps": p.get("sample_steps", 28),
                        },
                        "prompt_chunking_mode": p.get("prompt_chunking_mode", "a1111"),
                        "max_prompt_chunks": p.get("max_prompt_chunks", 0),
                    }
                ],
            },
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @staticmethod
    def generate_controlnet_config(
        p: Optional[Dict[str, Any]] = None,
        *,
        run_name: str,
        base_model_path: str,
        output_dir: str,
        dataset_path: str = "",
        dataset_configs: Optional[List[Dict[str, Any]]] = None,
        sample_prompts: Optional[list] = None,
        caption_processing: Optional[Dict[str, Any]] = None,
        **legacy_kwargs: Any,
    ) -> str:
        """Generate ControlNet training configuration YAML.

        ControlNet does NOT train UNet/TE (hardcoded). It uses unet_lr as the
        ControlNet LR. No block swap, vision encoder, param tracking, reference
        images, priority training, image_encoder_lr, or split TE LRs.

        ControlNet-specific params (in p):
        - controlnet_type: "standard" or "lllite"
        - controlnet_pretrained_path: Path to existing ControlNet checkpoint
        - controlnet_init_from_unet: Init from base UNet weights
        - lllite_conditioning_channels, lllite_rank
        - condition_preprocessors, condition_cache_mode
        """
        if p is None:
            p = legacy_kwargs
        elif legacy_kwargs:
            p = {**p, **legacy_kwargs}
        p = _normalize_params(p)

        total_steps = p.get("total_steps")
        epochs = p.get("epochs")
        if total_steps is None and epochs is None:
            raise ValueError("Either total_steps or epochs must be provided")
        if total_steps is not None and epochs is not None:
            raise ValueError("Cannot specify both total_steps and epochs")

        cache_latents_to_disk = p.get("cache_latents_to_disk", False)
        datasets_array = []
        if dataset_configs:
            for ds_config in dataset_configs:
                ds_path = ds_config.get("path", "")
                ds_dataset_id = ds_config.get("dataset_id")
                dataset_entry = {
                    "folder_path": ds_path,
                    "caption_ext": "txt",
                    "cache_latents_to_disk": cache_latents_to_disk,
                    **({"dataset_id": ds_dataset_id} if ds_dataset_id else {}),
                }
                dataset_entry.update(extract_dataset_params(ds_config))
                datasets_array.append(dataset_entry)
        else:
            datasets_array.append({
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "cache_latents_to_disk": cache_latents_to_disk,
            })

        # ControlNet-specific network config
        controlnet_network_config = {
            "type": p.get("controlnet_type", "standard"),
            "init_from_unet": p.get("controlnet_init_from_unet", True),
        }
        if p.get("controlnet_pretrained_path"):
            controlnet_network_config["pretrained_path"] = p["controlnet_pretrained_path"]
        if p.get("controlnet_type") == "lllite":
            controlnet_network_config["lllite_conditioning_channels"] = p.get("lllite_conditioning_channels", 32)
            controlnet_network_config["lllite_rank"] = p.get("lllite_rank", 64)
        if p.get("condition_preprocessors"):
            controlnet_network_config["condition_preprocessors"] = p["condition_preprocessors"]
            controlnet_network_config["condition_cache_mode"] = p.get("condition_cache_mode", "on_the_fly")

        config = {
            "job": run_name,
            "config": {
                "name": run_name,
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": output_dir,
                        "device": "cuda:0",
                        "network": {
                            "type": "controlnet",
                            "controlnet": controlnet_network_config,
                        },
                        "dtype": {
                            "weight": p.get("weight_dtype", "fp16"),
                            "training": p.get("training_dtype", "fp16"),
                            "vae": p.get("vae_dtype", "fp16"),
                            "save": p.get("output_dtype", "fp32"),
                        },
                        "save": {
                            "save_every": p.get("save_every", 500),
                            "save_every_unit": p.get("save_every_unit", "steps"),
                            "max_step_saves_to_keep": p.get("max_step_saves_to_keep") if p.get("max_step_saves_to_keep") is not None else 5,
                        },
                        "datasets": datasets_array,
                        "train": _build_train_section(
                            p,
                            total_steps=total_steps,
                            epochs=epochs,
                            train_unet=False,
                            train_text_encoder=False,
                            train_image_encoder=False,
                            component_lr_always_emit=False,
                            bucketing_always_emit=False,
                            include_block_swap=False,
                            include_vision_encoder=False,
                            include_param_tracking=False,
                            include_reference_images=False,
                            include_priority_training=False,
                            include_image_encoder_lr=False,
                            include_te_split_lrs=False,
                        ),
                        "model": {
                            "name_or_path": base_model_path,
                        },
                        "sample": {
                            "sampler": p.get("sample_sampler", "euler"),
                            "schedule_type": p.get("sample_schedule_type", "normal"),
                            "sample_every": p.get("sample_every", 500),
                            "width": p.get("sample_width", 512),
                            "height": p.get("sample_height", 512),
                            "prompts": sample_prompts or [],
                            "neg": "",
                            "seed": p.get("sample_seed", 42),
                            "guidance_scale": p.get("sample_cfg_scale", 7.0),
                            "sample_steps": p.get("sample_steps", 20),
                        },
                        "prompt_chunking_mode": p.get("prompt_chunking_mode", "a1111"),
                        "max_prompt_chunks": p.get("max_prompt_chunks", 0),
                    }
                ],
            },
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @staticmethod
    def save_config(config_yaml: str, output_path: str) -> None:
        """
        Save YAML configuration to file.

        Args:
            config_yaml: YAML configuration string
            output_path: Path to save the config file
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "w", encoding="utf-8") as f:
            f.write(config_yaml)
