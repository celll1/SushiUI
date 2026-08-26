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


def _detect_arch(base_model_path: str) -> str:
    """The architecture a config is being generated for, or ``"unknown"``.

    Two channels, the pair ``_apply_sensenova_training_contract`` uses: the
    detector, then the path name if it raises.
    """
    try:
        from core.model_loader import ModelLoader
        return ModelLoader.detect_model_type(base_model_path)
    except Exception:
        lowered = (base_model_path or "").lower()
        if "sensenova" in lowered or "sense-nova" in lowered:
            return "sensenova"
        return "unknown"


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
    from api.param_defaults import TRAINING_DEFAULTS as _TD

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
        "sdxl_micro_conditioning": p.get("sdxl_micro_conditioning", True),
        "sdxl_vae_type": p.get("sdxl_vae_type", "none"),
        "sdxl_te_type": p.get("sdxl_te_type", "none"),
        "sdxl_te_hidden_layer": p.get("sdxl_te_hidden_layer", -2),
        "sdxl_te_max_len": p.get("sdxl_te_max_len", 256),
        "sdxl_te_train_encoder": p.get("sdxl_te_train_encoder", False),
        # Optimizer
        "optimizer": p.get("optimizer", "adamw8bit"),
        "lr": lr,
        "lr_scheduler": p.get("lr_scheduler", "constant"),
    }

    # Conditional optimizer fields
    if p.get("lr_warmup_steps", 0) > 0:
        train["lr_warmup_steps"] = p["lr_warmup_steps"]
        # FIX (2026-07): the trainer only ever reads `optimizer_warmup_steps`
        # (base_trainer.py __init__ / train_runner.py `train_config.get(
        # 'optimizer_warmup_steps', 0)`), never `lr_warmup_steps`. Without this,
        # the UI's warmup value was a silent no-op. Write both keys — keep
        # `lr_warmup_steps` for back-compat with anything that reads it.
        train["optimizer_warmup_steps"] = p["lr_warmup_steps"]
    if str(p.get("lr_scheduler", "constant")) == "plateau_cosine_floor":
        # Plateau-then-cosine-floor scheduler knobs. Only meaningful for this
        # scheduler type; only emitted when selected to keep YAML output for
        # all other schedulers unchanged.
        train["lr_decay_start_ratio"] = p.get("lr_decay_start_ratio", 0.85)
        train["lr_floor_ratio"] = p.get("lr_floor_ratio", 0.25)
    if p.get("use_ema"):
        train["use_ema"] = p["use_ema"]
        train["ema_decay"] = p.get("ema_decay", 0.9999)
        train["ema_update_every"] = p.get("ema_update_every", 1)
        train["ema_device"] = p.get("ema_device", "cpu")
    # No optimizer_is_paged here: paging is part of the optimizer name
    # (paged_adamw / paged_adamw8bit / paged_lion8bit). The key an older run's
    # YAML may still carry is simply not read by anything.
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
        # base_resolutions is meaningful even without bucketing: the no-bucketing
        # path fits oversized items into the base-resolution area (bounds VAE-encode
        # and training memory). Emit it always so the trainer does not silently fall
        # back to its [1024] default when bucketing is off.
        train["base_resolutions"] = p.get("base_resolutions") or [1024]
        if p.get("enable_bucketing"):
            train["enable_bucketing"] = True
            train["bucket_strategy"] = p.get("bucket_strategy", "resize")
            train["multi_resolution_mode"] = p.get("multi_resolution_mode", "max")

    # Epoch-dynamic crop augmentation (SDXL only). Emit the switch always; emit details
    # only when enabled (the trainer falls back to TRAINING_DEFAULTS for absent keys).
    train["crop_augment_enable"] = p.get("crop_augment_enable", False)
    if train["crop_augment_enable"]:
        train["crop_full_image_prob"] = p.get("crop_full_image_prob", 0.7)
        train["crop_max_bucket_prob"] = p.get("crop_max_bucket_prob", 0.7)
        train["crop_min_area_ratio"] = p.get("crop_min_area_ratio", 0.25)
        train["crop_min_short_side_px"] = p.get("crop_min_short_side_px", 512)
        train["crop_aspect_mode"] = p.get("crop_aspect_mode", "source")
        train["crop_position_mode"] = p.get("crop_position_mode", "random")
        train["crop_smaller_bucket_mode"] = p.get("crop_smaller_bucket_mode", "base_res")
        train["crop_smaller_scale_range"] = p.get("crop_smaller_scale_range") or [0.5, 0.9]
        train["full_crop_position_mode"] = p.get("full_crop_position_mode", "center")
        train["crop_microcond_mode"] = p.get("crop_microcond_mode", "kohya")
        train["crop_plan_seed"] = p.get("crop_plan_seed", 0)

    # Common training fields
    # Attention backend (string selector). Back-compat: an old preset that only
    # set the boolean use_flash_attention still resolves to "flash". The string
    # key is authoritative when present.
    train["attention_backend"] = p.get("attention_backend") or (
        "flash" if p.get("use_flash_attention") else "native"
    )
    train["use_flash_attention"] = p.get("use_flash_attention", False)
    # Attention implementation registry ("conduit" | "diffusers"). Persisted into
    # the run config so resumes reproduce the same registry. Fresh runs default to
    # "conduit"; base_trainer applies the resume backward-compat (missing key on a
    # resume -> "diffusers"). Orthogonal to attention_backend.
    train["attention_impl"] = p.get("attention_impl", "conduit")
    train["min_snr_gamma"] = p.get("min_snr_gamma", 5.0)
    train["reconstruction_loss_weight"] = p.get("reconstruction_loss_weight", 0.0)

    # Block Swap (training VRAM optimization) - LoRA/Full FT only
    if include_block_swap:
        train["blocks_to_swap"] = p.get("blocks_to_swap", 0)
        train["use_pinned_memory"] = p.get("use_pinned_memory", False)
        train["sensenova_mot_phase_eviction"] = p.get(
            "sensenova_mot_phase_eviction", _TD["sensenova_mot_phase_eviction"]
        )
        train["sensenova_four_phase_eviction"] = p.get(
            "sensenova_four_phase_eviction", _TD["sensenova_four_phase_eviction"]
        )
        train["block_swap_h2d_only"] = p.get("block_swap_h2d_only", False)
        train["block_swap_ring_size"] = p.get("block_swap_ring_size", 2)
        train["num_optimizer_groups"] = p.get("num_optimizer_groups", 0)
        # Per-bucket activation offload dispatcher
        train["activation_dispatch_enable"] = p.get("activation_dispatch_enable", False)
        train["activation_dispatch_margin_gb"] = p.get("activation_dispatch_margin_gb", 1.0)
        train["activation_dispatch_seed_coef"] = p.get("activation_dispatch_seed_coef", 24.0e-6)
        train["activation_dispatch_residual_frac"] = p.get("activation_dispatch_residual_frac", 0.85)
        train["activation_dispatch_threshold_mb"] = p.get("activation_dispatch_threshold_mb", 4)

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
    # Full-parameter save: embed the VAE into the single-file checkpoint.
    # None passes through = per-arch default resolved by the adapters
    # (BUNDLE_VAE_DEFAULTS_BY_ARCH in api/param_defaults.py; sd15/sdxl/deus True,
    # others False); an explicit boolean always wins.
    # Read unconditionally; LoRA / pixel-space archs ignore it.
    train["bundle_vae"] = p.get("bundle_vae", None)
    # SenseNova full-fine-tune save format. Written unconditionally (it is not a
    # block-swap option and full FT does not go through that branch); every other
    # architecture reads it and ignores it.
    train["sensenova_full_finetune_save_format"] = p.get(
        "sensenova_full_finetune_save_format",
        _TD["sensenova_full_finetune_save_format"],
    )
    train["gradient_checkpointing"] = p.get("gradient_checkpointing", True)
    train["cpu_offload_checkpointing"] = p.get("cpu_offload_checkpointing", False)
    train["async_cpu_offload_checkpointing"] = p.get("async_cpu_offload_checkpointing", False)
    train["fp8_base_dtype"] = p.get("fp8_base_dtype", None)
    # MiniMax-H3 joint video+audio objective weight. Read unconditionally; every
    # other architecture ignores it. SSoT: api/param_defaults.TRAINING_DEFAULTS.
    train["audio_loss_weight"] = p.get("audio_loss_weight", _TD["audio_loss_weight"])
    # torch.compile (opt-in DiT training acceleration). Persisted so resumes
    # reproduce the same compile mode. Non-DiT / LoRA / block-swap runs read it
    # but the trainer no-ops with a warning.
    train["torch_compile"] = p.get("torch_compile", "off")
    train["torch_compile_dynamic"] = p.get("torch_compile_dynamic", None)
    # TREAD token routing (arXiv 2501.04765) — training-only. Read unconditionally;
    # non-Anima trainers ignore it (base_trainer only builds self.tread_config when
    # tread_enable is True). SSoT: api/param_defaults.TRAINING_DEFAULTS.
    train["tread_enable"] = p.get("tread_enable", False)
    train["tread_drop_ratio"] = p.get("tread_drop_ratio", 0.5)
    train["tread_start_block"] = p.get("tread_start_block", 2)
    train["tread_end_block"] = p.get("tread_end_block", 26)
    # Low-rate stochastic depth (per-batch block dropout) — training-only. Read
    # unconditionally; non-Anima trainers ignore it (base_trainer only builds
    # self.block_skip_config when block_skip_rate > 0). SSoT: TRAINING_DEFAULTS.
    train["block_skip_rate"] = p.get("block_skip_rate", 0.0)
    train["block_skip_protect_start"] = p.get("block_skip_protect_start", 6)
    train["block_skip_protect_end"] = p.get("block_skip_protect_end", 22)
    # DiT-BlockSkip (arXiv 2603.20755) — training-only memory-reduction for Anima
    # LoRA. Read unconditionally; base_trainer only builds self.blockskip_config
    # when blockskip_enable is True. SSoT: api/param_defaults.TRAINING_DEFAULTS.
    train["blockskip_enable"] = p.get("blockskip_enable", False)
    train["blockskip_front"] = p.get("blockskip_front", 4)
    train["blockskip_back"] = p.get("blockskip_back", 4)
    # Resolution curriculum (low-res warmup -> target) — training-only, arch-agnostic.
    # Read unconditionally; inert unless res_curriculum_enable and warmup_steps>0.
    # SSoT: TRAINING_DEFAULTS.
    train["res_curriculum_enable"] = p.get("res_curriculum_enable", False)
    train["res_curriculum_warmup_steps"] = p.get("res_curriculum_warmup_steps", 0)
    train["res_curriculum_warmup_scale"] = p.get("res_curriculum_warmup_scale", 0.5)

    # ---- Lens (bf16-native dual-stream DiT) — other archs ignore. ----
    # These were reachable from param_defaults, routes, openapi and the training
    # panel but were never written into the YAML, so lora_trainer's
    # `self.config.get("lens_lora_scope")` and lens_adapter's
    # `trainer.config.get("lens_*_lr_factor")` always saw their fallbacks and the
    # user's choice was dropped. SSoT: api/param_defaults.TRAINING_DEFAULTS.
    train["lens_lora_scope"] = p.get("lens_lora_scope", "img_attn,txt_attn,img_mlp,txt_mlp")
    train["lens_img_lr_factor"] = p.get("lens_img_lr_factor", 1.0)
    train["lens_txt_lr_factor"] = p.get("lens_txt_lr_factor", 1.0)

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
    train["minit2i_scratch_init_from"] = p.get("minit2i_scratch_init_from", "")
    train["minit2i_inherit_final_layer"] = p.get("minit2i_inherit_final_layer", False)

    # ---- Krea 2 (single-stream flow-matching MMDiT) — other archs ignore. ----
    train["krea2_lora_scope"] = p.get("krea2_lora_scope", "attn,mlp")
    train["krea2_lr_factor"] = p.get("krea2_lr_factor", 1.0)
    train["krea2_discrete_flow_shift"] = p.get("krea2_discrete_flow_shift", 2.5)

    # ---- ACE-Step 1.5 (turbo audio DiT) — other archs ignore. ----
    train["acestep_lora_scope"] = p.get("acestep_lora_scope", "attention")

    # ---- REPA (Representation Alignment) — MiniT2I only. SSoT: param_defaults. ----
    train["repa_enable"] = p.get("repa_enable", False)
    train["repa_encoder_source"] = p.get("repa_encoder_source", "tagger")
    train["repa_tagger_model_dir"] = p.get("repa_tagger_model_dir", "")
    train["repa_siglip2_repo"] = p.get("repa_siglip2_repo", "google/siglip2-so400m-patch14-384")
    train["repa_align_depth"] = p.get("repa_align_depth", -1)
    train["repa_weight"] = p.get("repa_weight", 0.5)
    train["repa_proj_lr_factor"] = p.get("repa_proj_lr_factor", 1.0)
    train["repa_encoder_resolution"] = p.get("repa_encoder_resolution", 0)

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

    # Pre-flight dataset rescan mode. Not read by the trainer subprocess; it is
    # persisted here because config_yaml is the only per-run config store for
    # model training (TrainingRun has no JSON config column, unlike
    # TaggerTrainingRun) and the /training/runs/{id}/start handler reads it back
    # from the YAML before spawning the trainer.
    # No literal fallback here: param_defaults.py owns the default (the API
    # request model supplies it), and a missing key normalizes to "off" in
    # normalize_rescan_mode() anyway. Importing api.param_defaults from this
    # module would be a circular import (api/__init__ -> api.routes -> here).
    train["rescan_before_training"] = p.get("rescan_before_training")

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
        force_recache = p.get("force_recache", False)
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
                    "force_recache": force_recache,
                    "resolution": base_resolutions or [512, 768, 1024],
                }
                dataset_entry.update(extract_dataset_params(ds_config))
                datasets_array.append(dataset_entry)
        else:
            datasets_array.append({
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "cache_latents_to_disk": cache_latents_to_disk,
                "force_recache": force_recache,
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
        - train_text_encoder default from
          param_defaults.resolve_full_finetune_train_text_encoder (True, except
          where the flag names half the denoiser)
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
        force_recache = p.get("force_recache", False)
        datasets_array = []
        if dataset_configs:
            for ds_config in dataset_configs:
                ds_path = ds_config.get("path", "")
                ds_dataset_id = ds_config.get("dataset_id")
                dataset_entry = {
                    "folder_path": ds_path,
                    "caption_ext": "txt",
                    "cache_latents_to_disk": cache_latents_to_disk,
                    "force_recache": force_recache,
                    **({"dataset_id": ds_dataset_id} if ds_dataset_id else {}),
                }
                dataset_entry.update(extract_dataset_params(ds_config))
                datasets_array.append(dataset_entry)
        else:
            datasets_array.append({
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "cache_latents_to_disk": cache_latents_to_disk,
                "force_recache": force_recache,
            })

        # Full FT defaults differ from LoRA defaults
        from api.param_defaults import resolve_full_finetune_train_text_encoder

        full_ft_defaults = {
            "learning_rate": 1e-6,
            "train_text_encoder": resolve_full_finetune_train_text_encoder(
                None, _detect_arch(base_model_path)
            ),
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
                            train_text_encoder=p["train_text_encoder"],
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
        force_recache = p.get("force_recache", False)
        datasets_array = []
        if dataset_configs:
            for ds_config in dataset_configs:
                ds_path = ds_config.get("path", "")
                ds_dataset_id = ds_config.get("dataset_id")
                dataset_entry = {
                    "folder_path": ds_path,
                    "caption_ext": "txt",
                    "cache_latents_to_disk": cache_latents_to_disk,
                    "force_recache": force_recache,
                    **({"dataset_id": ds_dataset_id} if ds_dataset_id else {}),
                }
                dataset_entry.update(extract_dataset_params(ds_config))
                datasets_array.append(dataset_entry)
        else:
            datasets_array.append({
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "cache_latents_to_disk": cache_latents_to_disk,
                "force_recache": force_recache,
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
        # Outpaint-native conditioning (PART B)
        controlnet_network_config["conditioning_mode"] = p.get("conditioning_mode", "preprocessor")
        controlnet_network_config["outpaint_crop_min_area"] = p.get("outpaint_crop_min_area", 0.15)
        controlnet_network_config["outpaint_crop_max_area"] = p.get("outpaint_crop_max_area", 0.8)
        controlnet_network_config["outpaint_edge_anchor_prob"] = p.get("outpaint_edge_anchor_prob", 0.34)
        controlnet_network_config["outpaint_corner_anchor_prob"] = p.get("outpaint_corner_anchor_prob", 0.33)
        controlnet_network_config["outpaint_mask_channel"] = p.get("outpaint_mask_channel", True)
        controlnet_network_config["outpaint_known_loss_weight"] = p.get("outpaint_known_loss_weight", 0.3)
        controlnet_network_config["outpaint_seam_loss_boost"] = p.get("outpaint_seam_loss_boost", 0.0)
        controlnet_network_config["outpaint_seam_ring_width"] = p.get("outpaint_seam_ring_width", 1)
        controlnet_network_config["outpaint_seam_grad_lambda"] = p.get("outpaint_seam_grad_lambda", 0.0)
        controlnet_network_config["outpaint_loss_normalize"] = p.get("outpaint_loss_normalize", False)
        # R1 (scratchpad/outpaint_boundary_structure_fix.md D3-R1): per-sample
        # randomized crop_mask conditioning edge-softness range (canvas px). 0/0
        # (default) -> byte-identical (razor-sharp) behavior.
        controlnet_network_config["outpaint_edge_feather_min_px"] = p.get("outpaint_edge_feather_min_px", 0.0)
        controlnet_network_config["outpaint_edge_feather_max_px"] = p.get("outpaint_edge_feather_max_px", 0.0)

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
    def generate_vae_config(
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
        """Generate a decoder-only VAE fine-tune configuration YAML.

        Shape (consumed by train_runner's ``network.type == "vae_decoder"``
        branch and ``core/training/vae/vae_config.resolve_vae_training_config``):

        - ``network.type: vae_decoder`` is the discriminator.
        - the generic run-shape knobs go into ``process.train`` / ``process.save``
          so the existing routes, resume plumbing and checkpoint-keep field keep
          owning them unchanged;
        - every VAE-specific knob goes into a dedicated ``process.vae`` section,
          written explicitly from ``VAE_TRAINING_DEFAULTS`` so the YAML is
          self-describing and hand-editable.

        There is no ``sample`` section: a VAE fine-tune has no denoiser to sample
        from — its qualitative signal is the held-out validation PSNR/blockiness
        chart, which the trainer emits into ``TrainingMetrics.extra_metrics``.

        ``sample_prompts`` / ``caption_processing`` are accepted and ignored so
        this signature stays interchangeable with the other generators at the
        route's dispatch site.

        Precedence per key: ``p["vae_config"][key]`` > ``p[key]`` **only when the
        caller explicitly set that flat field** > ``VAE_TRAINING_DEFAULTS[key]``.

        The middle tier is gated on ``p["_explicit_fields"]`` (the route passes
        ``request.model_fields_set``) because ``request.model_dump()``
        materialises EVERY Pydantic default as a non-None value. Without the
        gate, tier 2 is not "what the caller sent" but "the diffusion trainer's
        defaults, unconditionally", which silently overrode five VAE defaults
        (learning_rate 1e-5 -> 1e-4, max_grad_norm 0.1 -> 1.0, optimizer adamw ->
        adamw8bit, save_every 500 -> 100, ema_decay 0.999 -> 0.9999 — the last of
        which is not even a run-shape key and so mis-reported itself in the YAML,
        the sidecar and /params). A caller with no ``_explicit_fields`` key at
        all (direct/legacy kwargs, tests) keeps the permissive behaviour.
        """
        # Lazy + function-local: backend/api/__init__.py imports the whole API
        # surface, so a module-level api.* import from core/training/ would be a
        # cycle. Same pattern as core/training/adapters/*.py.
        from api.param_defaults import VAE_TRAINING_DEFAULTS
        from api.error_handlers import ValidationError

        if p is None:
            p = legacy_kwargs
        elif legacy_kwargs:
            p = {**p, **legacy_kwargs}
        p = _normalize_params(p)

        # The caller may pass the VAE options either flat (p["resolution"]) or
        # nested under p["vae_config"]; nested wins.
        nested = p.get("vae_config") or {}
        if not isinstance(nested, dict):
            raise ValidationError(
                "vae_config must be a mapping",
                detail=f"got {type(nested).__name__}",
            )
        unknown = sorted(set(nested) - set(VAE_TRAINING_DEFAULTS))
        if unknown:
            raise ValidationError(
                f"Unknown vae_config key(s): {unknown}",
                detail=f"Valid keys: {sorted(VAE_TRAINING_DEFAULTS)}",
            )

        explicit = p.get("_explicit_fields")
        explicit_set = set(explicit) if explicit is not None else None

        def value_of(key: str) -> Any:
            if key in nested and nested[key] is not None:
                return nested[key]
            if p.get(key) is not None and (explicit_set is None or key in explicit_set):
                return p[key]
            return VAE_TRAINING_DEFAULTS[key]

        # Keys that live in the shared train/save sections rather than
        # process.vae. Every one of these is ALSO a TrainingRunCreateRequest
        # field, which is what makes them survive the
        # create -> GET /params -> PUT regenerate cycle: the extractor rebuilds
        # them from the request schema.
        #
        # `seed` and `num_workers` deliberately do NOT belong here even though
        # they are run-shape-ish: they are not request fields, so a train-section
        # placement made them unreadable by _extract_request_params_from_yaml and
        # they were silently reset to their defaults on every edit-form save.
        # They live in process.vae, which /params carries through verbatim.
        run_shape_keys = {
            "batch_size", "total_steps", "gradient_accumulation_steps",
            "learning_rate", "optimizer", "optimizer_weight_decay",
            "max_grad_norm", "lr_scheduler", "lr_warmup_steps",
            "save_every", "max_step_saves_to_keep",
        }
        vae_section = {k: value_of(k) for k in VAE_TRAINING_DEFAULTS
                       if k not in run_shape_keys}

        cache_latents_to_disk = False  # VAE training is raw-pixel by definition
        datasets_array = []
        if dataset_configs:
            for ds_config in dataset_configs:
                dataset_entry = {
                    "folder_path": ds_config.get("path", ""),
                    "caption_ext": "txt",
                    "cache_latents_to_disk": cache_latents_to_disk,
                    **({"dataset_id": ds_config["dataset_id"]}
                       if ds_config.get("dataset_id") else {}),
                }
                dataset_entry.update(extract_dataset_params(ds_config))
                datasets_array.append(dataset_entry)
        else:
            datasets_array.append({
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "cache_latents_to_disk": cache_latents_to_disk,
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
                        "network": {"type": "vae_decoder"},
                        "model": {"name_or_path": base_model_path},
                        "datasets": datasets_array,
                        "save": {
                            "save_every": value_of("save_every"),
                            "save_every_unit": "steps",
                            "max_step_saves_to_keep": value_of("max_step_saves_to_keep"),
                        },
                        "train": {
                            "batch_size": value_of("batch_size"),
                            "steps": value_of("total_steps"),
                            "gradient_accumulation_steps": value_of("gradient_accumulation_steps"),
                            "lr": value_of("learning_rate"),
                            "optimizer": value_of("optimizer"),
                            "optimizer_weight_decay": value_of("optimizer_weight_decay"),
                            "max_grad_norm": value_of("max_grad_norm"),
                            "lr_scheduler": value_of("lr_scheduler"),
                            "lr_warmup_steps": value_of("lr_warmup_steps"),
                            # seed / num_workers are emitted inside process.vae
                            # (see run_shape_keys) so they survive the /params
                            # readback.
                            "resume_from_checkpoint": p.get("resume_from_checkpoint"),
                        },
                        "vae": vae_section,
                    }
                ],
            },
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False,
                         allow_unicode=True)

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
