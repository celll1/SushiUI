"""
Single source of truth for all API parameter default values.

Backend Pydantic models and Form() defaults reference this module.
Frontend fetches these via /schema/* endpoints, eliminating manual sync.
"""

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Generation (txt2img / img2img / inpaint)
# ---------------------------------------------------------------------------
# Authoritative source: Form() defaults in generate_* route handlers.
# Pydantic GenerationParams defaults are aligned to these values.

GENERATION_DEFAULTS: Dict[str, Any] = {
    # Core
    "negative_prompt": "",
    "steps": 20,
    "cfg_scale": 7.0,
    "sampler": "euler",
    "schedule_type": "uniform",
    "seed": -1,
    "ancestral_seed": -1,
    "width": 1024,
    "height": 1024,
    "batch_size": 1,
    "prompt_chunking_mode": "a1111",
    "max_prompt_chunks": 0,
    "developer_mode": False,
    # CFG scheduling
    "cfg_schedule_type": "constant",
    "cfg_schedule_min": 1.0,
    "cfg_schedule_max": None,
    "cfg_schedule_power": 2.0,
    "cfg_rescale_snr_alpha": 0.0,
    # Dynamic thresholding
    "dynamic_threshold_percentile": 0.0,
    "dynamic_threshold_mimic_scale": 7.0,   # Form() default; Pydantic had 1.0 (bug)
    # NAG
    "nag_enable": False,
    "nag_scale": 5.0,
    "nag_tau": 3.5,
    "nag_alpha": 0.25,
    "nag_sigma_end": 3.0,                   # Form() default; Pydantic had 0.0 (bug)
    "nag_negative_prompt": "",
    # Attention / quantization
    "attention_type": "normal",
    "unet_quantization": None,
    "text_encoder_quantization": None,
    "cpu_text_encoding": False,
    "use_torch_compile": False,
    "use_tipo": False,
    "preview_predicted_x0": False,
    # Block swap (Form-only in original code)
    "enable_block_swap": False,
    "blocks_to_swap": 20,
    "use_pinned_memory": False,
    # Vision encoder
    "vision_encoder_path": None,
    # img2img / inpaint shared
    "denoising_strength": 0.75,
    "img2img_fix_steps": True,
    "resize_mode": "image",
    "resampling_method": "lanczos",
    # inpaint only
    "mask_blur": 4,
    "inpaint_full_res": False,
    "inpaint_full_res_padding": 32,
    "inpaint_fill_mode": "original",
    "inpaint_fill_strength": 1.0,
    "inpaint_blur_strength": 1.0,
}

# Keys present only in img2img/inpaint (not txt2img)
_IMG2IMG_ONLY = frozenset({
    "denoising_strength", "img2img_fix_steps", "resize_mode", "resampling_method",
})
# Keys present only in inpaint (not txt2img or img2img)
_INPAINT_ONLY = frozenset({
    "mask_blur", "inpaint_full_res", "inpaint_full_res_padding",
    "inpaint_fill_mode", "inpaint_fill_strength", "inpaint_blur_strength",
})

TXT2IMG_DEFAULTS: Dict[str, Any] = {
    k: v for k, v in GENERATION_DEFAULTS.items()
    if k not in _IMG2IMG_ONLY and k not in _INPAINT_ONLY
}
IMG2IMG_DEFAULTS: Dict[str, Any] = {
    k: v for k, v in GENERATION_DEFAULTS.items()
    if k not in _INPAINT_ONLY
}
INPAINT_DEFAULTS: Dict[str, Any] = dict(GENERATION_DEFAULTS)

# ---------------------------------------------------------------------------
# LoRA / Full-FT Training (TrainingRunCreateRequest)
# ---------------------------------------------------------------------------
# Authoritative source: backend Pydantic model.
# Frontend DEFAULT_PARAMS is aligned to these values.

TRAINING_DEFAULTS: Dict[str, Any] = {
    # Dataset
    "dataset_configs": [],
    "run_name": None,
    "training_method": "lora",
    "base_model_path": "",
    # Training parameters
    "total_steps": 1000,
    "epochs": 10,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    # Optimizer
    "optimizer": "adamw8bit",
    "optimizer_is_paged": False,
    "optimizer_cautious": False,
    "optimizer_beta1": 0.9,
    "optimizer_beta2": 0.999,
    "optimizer_epsilon": 1e-8,
    "optimizer_weight_decay": 0.01,
    "optimizer_schedule_free": False,
    "optimizer_schedule_free_r": 0.0,
    "optimizer_schedule_free_weight_lr_power": 2.0,
    "optimizer_use_radam": False,
    "optimizer_stochastic_rounding": False,
    # LoRA specific
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dtype": "fp32",
    "network_type": "lora",
    # ReLoRA
    "relora_merge_every": 500,
    "relora_merge_unit": "steps",
    "restart_warmup_steps": 100,
    "optimizer_reset_strategy": "full_reset",
    "optimizer_pruning_ratio": 0.9,
    # Checkpoints & sampling
    "save_every": 100,
    "save_every_unit": "steps",
    "max_step_saves_to_keep": None,
    "sample_every": 100,
    "sample_prompts": [{"positive": "", "negative": ""}],
    "resume_from_checkpoint": "latest",
    "sample_width": 1024,
    "sample_height": 1024,
    "sample_steps": 28,
    "sample_cfg_scale": 7.0,
    "sample_sampler": "euler",
    "sample_schedule_type": "sgm_uniform",  # Fix: frontend had "uniform"
    "sample_seed": -1,
    # Debug
    "debug_latents": False,
    "debug_latents_every": 50,
    # Bucketing
    "enable_bucketing": False,
    "base_resolutions": [1024],
    "bucket_strategy": "resize",
    "multi_resolution_mode": "max",
    "cache_latents_to_disk": False,         # Fix: frontend had True
    # Component-specific
    "train_unet": True,
    "train_text_encoder": False,            # Fix: frontend had True
    "unet_lr": 1e-5,
    "text_encoder_lr": 1e-6,
    "text_encoder_1_lr": None,
    "text_encoder_2_lr": None,
    # Frontend-only fields now accepted by backend
    "train_image_encoder": False,
    "image_encoder_lr": None,
    "force_recache": False,
    "reconstruction_loss_weight": 0.0,
    # Precision
    "weight_dtype": "fp16",
    "training_dtype": "fp16",
    "output_dtype": "fp32",
    "vae_dtype": "fp16",                    # Fix: frontend had "fp32"
    "mixed_precision": True,
    "use_flash_attention": False,
    "min_snr_gamma": 5.0,
    # Text / latent encoding
    # text_encoding_mode: "swap_onthefly" | "pre_encoded_cache" | "onthefly_gpu"
    #                   | "cpu_prefetch"
    # cpu_prefetch pins the frozen TE on CPU and runs caption encoding for
    # upcoming batches on a daemon thread, in parallel with GPU train steps.
    # Stalls (queue empty when the trainer pulls the next batch) are logged
    # at epoch end so the user can see whether CPU encode keeps up.
    "text_encoding_mode": "swap_onthefly",
    "text_encoding_swap_interval": 256,
    "text_encoding_prefetch_depth": 4,
    "latent_encoding_mode": "swap_onthefly",
    "latent_encoding_swap_interval": 256,
    # Block swap
    "blocks_to_swap": 0,
    "use_pinned_memory": False,
    "num_optimizer_groups": 0,
    # MNT
    "multi_noise_timesteps": 1,
    "multi_noise_mode": "independent",
    "trajectory_blend_alpha": 0.7,
    "timestep_sampling": {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0},
    # Regularization
    "regularization_type": None,
    "snr_regularization_weight": 0.1,       # Fix: frontend had 0.0
    "snr_timestep_adaptive": True,
    "snr_penalty_mode": "relu",
    "energy_regularization_weight": 0.05,   # Fix: frontend had 0.0
    "energy_timestep_adaptive": True,
    "energy_penalty_mode": "abs",           # Fix: frontend had "under"
    "energy_normalize_by_pixels": True,
    # Unified framework
    "noise_process": "auto",
    "prediction_target": "auto",
    "strict_validation": False,
    # Vision encoder
    "use_reference_images": False,
    "vision_encoder_path": None,
    "train_vision_encoder": False,
    "vision_encoder_lr": None,
    "gradient_routing_ve": False,
    # Param tracking
    "param_tracking": False,
    "param_tracking_interval": 100,
    # ControlNet
    "controlnet_type": "standard",
    "controlnet_pretrained_path": None,
    "controlnet_init_from_unet": True,
    "lllite_conditioning_channels": 32,
    "lllite_rank": 64,
    "condition_preprocessors": None,
    "condition_cache_mode": "on_the_fly",
    # Pre-flight dataset drift check + optional rescan.  4 modes:
    #   "off"   — skip entirely (default)
    #   "path"  — only detect added/missing files
    #   "smart" — path drift + caption sidecar mtime check
    #   "force" — always rescan, no drift detection
    # Also cleans up orphan latent cache files when a rescan happens.
    # See core/training/dataset_drift.py.
    "rescan_before_training": "off",

    # ---- Anima (Cosmos-Predict2 DiT) training ----
    # LoRA targets enumerated by core/models/anima/anima_lora.py.
    # Comma-separated subset of {attention, mlp, mod, llm_adapter}; the
    # AnimaLoRAAdapter dispatch normalises and applies the corresponding
    # scope flags (see core/training/lora_trainer.py:_create_adapter).
    "anima_lora_scope": "attention,mlp,llm_adapter",
    "lens_lora_scope": "img_attn,txt_attn,img_mlp,txt_mlp",
    # Lens full-parameter LR multipliers per stream.
    "lens_img_lr_factor": 1.0,
    "lens_txt_lr_factor": 1.0,
    # If False, llm_adapter is dropped from the scope regardless of the
    # csv above. Defaults to True so the LLM Adapter is fine-tuned along
    # with the DiT blocks (Phase C user request).
    "train_llm_adapter": True,

    # ---- Anima full-parameter training: per-group LR multipliers ----
    # Applied on top of unet_lr in AnimaFullParameterAdapter. Defaults of
    # 1.0 collapse to a single effective LR; users wanting sd-scripts-style
    # finer control (lower modulation LR etc.) can dial these.
    "anima_attn_mlp_lr_factor": 1.0,
    "anima_mod_lr_factor": 1.0,
    "anima_llm_adapter_lr_factor": 1.0,

    # ---- Anima Phase D: memory optimisations ----
    # Gradient-checkpoint mode for the DiT blocks. Both default to False
    # (i.e. standard GPU-resident checkpointing). When both are True the
    # async variant wins (faster CPU<->GPU overlap).
    "cpu_offload_checkpointing": False,
    "async_cpu_offload_checkpointing": False,
    # Quantise the *frozen* base DiT to FP8 before LoRA wrap. ~50% VRAM
    # reduction on the base. Only meaningful for training_method='lora'
    # (Full FT needs trainable base weights — flag is silently ignored).
    # None | "fp8_e4m3fn" | "fp8_e5m2".
    "fp8_base_dtype": None,

    # ---- Online Danbooru augmentation (image-generation training) ----
    # Diffusion-side counterpart of the tagger's Danbooru augmentation. Text
    # conditioning accepts arbitrary tokens, so there is NO vocabulary
    # expansion here — the mechanism only fetches extra training images from
    # Danbooru and injects them as ordinary samples (interrupt-batch).
    # See core/training/danbooru_image_augment.py.
    "danbooru_aug_enable": False,
    # Static path: user-supplied general Danbooru queries (newline-separated;
    # each line is one query, Danbooru tag syntax, e.g. "1girl solo score:>=50").
    "danbooru_aug_queries": "",
    "danbooru_aug_weight_static": 1.0,
    # Deficiency path (auto + manual). Collects images for under-represented
    # tags to rebalance the dataset. No per-tag F1 (diffusion has none) — purely
    # dataset tag-frequency based.
    "danbooru_aug_deficiency_enable": True,      # auto: detect rare tags from dataset captions
    "danbooru_aug_deficiency_min_count": 20,     # tags in fewer than N dataset images are deficient
    "danbooru_aug_deficiency_top_k": 200,        # cap on the number of rarest tags targeted
    "danbooru_aug_deficiency_manual": "",        # manual: extra tags to top up (newline-separated)
    "danbooru_aug_weight_deficiency": 1.0,
    # Injection cadence / sizing.
    "danbooru_aug_injection_interval": 4,        # inject a Danbooru batch every N base batches
    "danbooru_aug_injection_ratio": 1.0,         # injection batch size = ratio x train batch_size
    # Collection / API settings.
    "danbooru_aug_min_score": 0,
    "danbooru_aug_max_posts_per_query": 200,
    "danbooru_aug_api_interval": 1.4,            # Danbooru TOS rate limit (seconds)
    "danbooru_aug_dl_speed_kbps": 500,
    "danbooru_aug_buffer_size": None,            # None -> auto (max(32, 16 x batch_size))
    # Caption construction from the post's per-category tag fields.
    "danbooru_aug_include_rating_tag": False,    # prepend rating word (general/sensitive/...)
    "danbooru_aug_max_caption_tags": 0,          # 0 = keep all tags
    # Caption tag shuffle / dropout for injected samples. SEPARATE from the
    # per-dataset caption_processing (datasets vary per run by user intent).
    # Applied per-epoch via the same processor the datasets use. All default
    # off = fixed category order, no shuffle/dropout.
    "danbooru_aug_shuffle_tags": False,          # shuffle tags within each category
    "danbooru_aug_shuffle_keep_first_n": 0,      # keep first N tags fixed (no shuffle)
    "danbooru_aug_tag_dropout_rate": 0.0,        # per-tag dropout probability (0-1)
    "danbooru_aug_tag_dropout_keep_first_n": 0,  # keep first N tags fixed (no dropout)
    "danbooru_aug_caption_dropout_rate": 0.0,    # probability to drop the whole caption (0-1)
    "danbooru_aug_keep_tokens": 0,               # first N tokens immune to token dropout
}

# ---------------------------------------------------------------------------
# Tagger Training (TaggerTrainingRunCreateRequest)
# ---------------------------------------------------------------------------

TAGGER_TRAINING_DEFAULTS: Dict[str, Any] = {
    "run_name": None,
    "vision_encoder_path": "",
    "training_method": "lora",
    "lora_rank": 32,
    "lora_alpha": 16.0,
    "learning_rate": 3e-4,
    "head_lr_multiplier": 10.0,
    "optimizer": "adamw8bit",
    "warmup_steps": 100,
    "epochs": 10,
    "batch_size": 32,
    "num_workers": 4,
    "num_workers_override": None,
    "save_every_n_steps": 500,
    "save_every_n_epochs": 0,
    "keep_last_n_checkpoints": 3,
    "checkpoint_save_mode": "lora",
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "weight_decay": 1e-4,
    "loss_function": "asl",
    "loss_gamma_neg": 4.0,
    "loss_gamma_pos": 1.0,
    "loss_clip": 0.05,
    "loss_gamma0": 4.0,
    "loss_m0": 0.2,
    "loss_beta": 2.0,
    "loss_rho": 0.5,
    "loss_label_weight": "fisher",
    "validate_every": 1,
    "val_split": 0.05,
    "val_split_mode": "percent",
    "val_fixed_size": None,
    "save_best_only": False,
    "vocab_min_count": 10,
    "excluded_categories": [],
    "ban_tags": "",
    "use_tag_aliases": False,
    "save_base_model": False,               # Fix: frontend had True
    # Quality-tag loss masking when a quality tag is present on a sample:
    #   "intra_group" (default) — mask sibling tags within the same group
    #                              (best/high/normal/medium share gradients;
    #                              low/bad/worst share gradients).  Trains
    #                              the cross-group good-vs-bad distinction.
    #                              Safer when intra-group prevalence imbalance
    #                              or annotation noise is significant.
    #   "cross_group"           — all non-positive quality tags train as
    #                              negatives.  Correct only when intra-group
    #                              labels are truly mutually exclusive.
    "quality_masking_mode": "intra_group",
    "cls_dim": None,
    "hidden_proj_dim": None,
    "init_head_from": None,
    # LR matrix (conditional inference) — built once at training start
    # when enabled.  See backend/core/tagger/lr_matrix_builder.py.
    "build_lr_matrix_on_start": False,
    "lr_top_anchors":            10000,
    "lr_top_targets":            1000,
    "lr_threshold":              1.0,
    "lr_min_anchor_count":       10,
    # Pre-flight dataset drift check + optional rescan.  4 modes:
    #   "off"   — skip entirely (default)
    #   "path"  — only detect added/missing files (~5 min for 3M items)
    #   "smart" — path drift + caption sidecar mtime check (catches
    #             in-place caption edits — adds a stat per sidecar)
    #   "force" — always rescan, no drift detection (most expensive)
    # See core/training/dataset_drift.py.
    "rescan_before_training": "off",
    # Training-time F1 metrics.
    # N2 (eval_every_n_steps) < N1 (threshold_search_every_n_steps).
    # N1 runs _find_best_threshold() on the rolling buffer (27 calls).
    # N2 runs _compute_f1_macro() once with the current threshold (1 call).
    "train_f1_eval_every_n_steps": 100,
    "train_f1_threshold_search_every_n_steps": 500,
    "train_f1_initial_threshold": 0.35,
    # Number of recent training batches kept in the rolling buffer.
    # Memory ≈ buffer_batches × batch_size × num_tags × 3 bytes (fp16 probs + bool labels).
    "train_f1_buffer_batches": 16,
    # Per-tag threshold metrics saved alongside each checkpoint as
    # ``{name}_tag_metrics.npz``.  Set to False to disable.
    # Memory: ≈ 4 × vocab_size × n_bins × 4 bytes (two epoch histogram slots).
    "save_tag_metrics": True,
    # Probability band considered "hard" for hard_rate computation.
    "hard_rate_lo": 0.25,
    "hard_rate_hi": 0.75,
    # Online Danbooru augmentation
    "enable_danbooru_augmentation": False,
    "danbooru_tags": "",              # newline-separated queries (use !tag or -tag to exclude)
    "danbooru_injection_interval": 4, # interrupt-batch every N base steps
    "danbooru_injection_batch_size_ratio": 1.0,  # 1.0 = full batch, 0.5 = half, etc.
    "danbooru_min_score": 0,
    "danbooru_max_posts_per_query": 200,
    "danbooru_api_interval": 1.4,
    "danbooru_dl_speed_kbps": 500,
    "danbooru_buffer_size": None,     # None → auto (2 × batch_size)
    "danbooru_vocab_expand": False,
    "danbooru_new_tag_min_count": 200,
    # Per-category post_count threshold overrides for the surveyor (category code
    # → min). Empty = use danbooru_new_tag_min_count for all. Lets copyright use a
    # higher bar than character (copyright post counts dwarf individual chars).
    "danbooru_new_tag_min_count_by_cat": {},
    "danbooru_new_tag_lookback_days": 90,
    "danbooru_new_tag_categories": [0, 3, 4],
    "danbooru_new_tag_survey_interval": 3600,
    # LRU cap on the dynamic new-tag query list (surveyor-discovered tags that
    # keep being collected across epochs/resumes). 0 = unlimited. When >0, the
    # least-recently-collected tag is evicted once the cap is exceeded.
    "danbooru_max_dynamic_tags": 0,
    # Collection-path weights (weighted random selection among available paths)
    "danbooru_query_weight_static": 1.0,   # user static queries
    "danbooru_query_weight_new_tag": 1.0,  # surveyor-discovered new tags (vocab expansion)
    "danbooru_query_weight_low_f1": 1.0,   # low-F1 existing vocab tags (deficiency collection)
    # Low-F1 deficiency collection: existing vocab tags whose per-tag F1 is below
    # threshold are collected. Targets recomputed at the train-F1 threshold-search cadence.
    "danbooru_low_f1_enable": False,
    "danbooru_low_f1_threshold": 0.5,  # tags with valid F1 below this are targeted (NaN excluded)
    "danbooru_low_f1_top_k": 500,      # cap on number of worst-F1 tags targeted
    "danbooru_low_f1_min_posts": 50,   # min Danbooru page-1 posts to include a tag (else skipped)
    # Co-occurrence vocab discovery: add vocab-absent tags that appear in
    # collected posts >= cooc_min_count times (category from the post fields,
    # filtered to danbooru_new_tag_categories). Created-at independent, so it
    # catches old tags missing from the training vocab. Requires vocab_expand.
    "danbooru_cooc_expand_enable": False,
    "danbooru_cooc_min_count": 50,
    # Categories eligible for co-occurrence discovery — independent of the
    # surveyor's categories, so a character-only surveyor net still lets the
    # accompanying copyright/general tags be added when they co-occur.
    "danbooru_cooc_categories": [0, 3, 4],
}
