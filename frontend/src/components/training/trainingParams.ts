// The training form's parameter surface: which request fields it restores,
// which a preset carries, and how a stored preset's older spellings map back.
//
// A module rather than a block inside TrainingConfig.tsx, because these lists
// only mean anything relative to each other -- PRESET_RESTORABLE_KEYS is built
// from PARAM_KEYS, the camel table is built from those, and presetConfigToParams
// reads all of them. Kept in the component they were four hundred lines adrift
// of the code that consumes them, and the only place their agreement could be
// checked was by parsing the component's source.
//
// Public surface is PARAM_KEYS, PRESET_EXCLUDED_KEYS,
// PRESET_CLEARABLE_NUMERIC_KEYS and presetConfigToParams; the rest is internal.

import { TrainingRunCreateRequest } from "@/utils/api";

// ============================================================
// Preset payload plumbing
// ------------------------------------------------------------
// A preset is the outgoing request minus PRESET_EXCLUDED_KEYS, so a new
// parameter survives a save/load without anyone remembering to list it.
// backend/tests/training_preset_payload_test.py fails if it stops holding.
// ============================================================

/**
 * Every TrainingRunCreateRequest field this form restores. getRequestData()
 * writes them; applyParamsToState() forwards each one that is present.
 */
export const PARAM_KEYS: (keyof TrainingRunCreateRequest)[] = [
  "lora_rank", "lora_alpha", "lora_dtype", "adapter_algorithm",
  "weight_decompose", "adapter_config",
  "total_steps", "epochs",
  "batch_size", "gradient_accumulation_steps", "max_grad_norm", "learning_rate", "lr_scheduler", "lr_warmup_steps",
  "lr_decay_start_ratio", "lr_floor_ratio", "rewarmup_on_optimizer_reset",
  "use_ema", "ema_decay", "ema_update_every", "ema_device", "gpu_index", "optimizer",
  "optimizer_beta1", "optimizer_beta2", "optimizer_epsilon", "optimizer_weight_decay",
  "optimizer_cautious", "optimizer_schedule_free",
  "optimizer_schedule_free_r", "optimizer_schedule_free_weight_lr_power",
  "optimizer_use_radam", "optimizer_stochastic_rounding",
  "optimizer_state_host_resident",
  "save_every", "save_every_unit", "max_step_saves_to_keep",
  "max_optimizer_saves_to_keep", "resume_from_checkpoint",
  "train_unet", "train_text_encoder", "train_image_encoder",
  "unet_lr", "text_encoder_lr", "text_encoder_1_lr", "text_encoder_2_lr", "image_encoder_lr",
  "weight_dtype", "training_dtype", "output_dtype", "vae_dtype",
  "mixed_precision", "attention_backend", "attention_impl", "use_flash_attention", "min_snr_gamma", "reconstruction_loss_weight",
  "audio_loss_weight",
  "text_encoding_mode", "text_encoding_swap_interval",
  "text_encoding_prefetch_depth",
  "latent_encoding_mode", "latent_encoding_swap_interval",
  "gradient_checkpointing", "torch_compile", "torch_compile_dynamic",
  "cpu_offload_checkpointing", "async_cpu_offload_checkpointing", "fp8_base_dtype",
  "res_curriculum_enable", "res_curriculum_warmup_steps", "res_curriculum_warmup_scale",
  "cfg_uncond_drop_rate", "cfg_uncond_drop_per_mnt",
  "minit2i_label_drop_rate", "minit2i_lr_factor", "minit2i_flan_t5_path", "minit2i_scratch_init_from",
  "minit2i_inherit_final_layer", "minit2i_lora_scope", "minit2i_te_lora_scope",
  "anima_lora_scope", "train_llm_adapter", "anima_attn_mlp_lr_factor",
  "anima_mod_lr_factor", "anima_llm_adapter_lr_factor",
  "lens_lora_scope", "lens_img_lr_factor", "lens_txt_lr_factor",
  "ideogram4_lora_scope", "ideogram4_train_uncond",
  "ideogram4_uncond_loss_weight", "ideogram4_lr_factor",
  "krea2_lora_scope", "krea2_lr_factor", "krea2_discrete_flow_shift",
  "repa_enable", "repa_encoder_source", "repa_tagger_model_dir", "repa_siglip2_repo",
  "repa_align_depth", "repa_weight", "repa_proj_lr_factor", "repa_encoder_resolution",
  "danbooru_aug_enable", "danbooru_aug_queries", "danbooru_aug_weight_static",
  "danbooru_aug_deficiency_enable", "danbooru_aug_deficiency_min_count",
  "danbooru_aug_deficiency_top_k", "danbooru_aug_deficiency_manual",
  "danbooru_aug_weight_deficiency", "danbooru_aug_injection_interval",
  "danbooru_aug_injection_ratio", "danbooru_aug_min_score",
  "danbooru_aug_max_posts_per_query", "danbooru_aug_api_interval",
  "danbooru_aug_dl_speed_kbps",
  "danbooru_speed_check_enable", "danbooru_speed_degraded_kbps",
  "danbooru_speed_min_slow_streak", "danbooru_speed_min_slow_seconds",
  "danbooru_speed_cooldown_seconds",
  "danbooru_aug_buffer_size",
  "danbooru_aug_include_rating_tag", "danbooru_aug_max_caption_tags",
  "danbooru_quality_tag_enable", "danbooru_quality_tag_thresholds",
  "danbooru_quality_tag_attach_negative",
  "danbooru_aug_shuffle_tags", "danbooru_aug_shuffle_keep_first_n",
  "danbooru_aug_tag_dropout_rate", "danbooru_aug_tag_dropout_keep_first_n",
  "danbooru_aug_caption_dropout_rate", "danbooru_aug_keep_tokens",
  "blocks_to_swap", "use_pinned_memory", "sensenova_mot_phase_eviction",
  "sensenova_four_phase_eviction", "sensenova_four_phase_shared_prefix",
  "sensenova_four_phase_grad_reduction", "sensenova_full_finetune_save_format",
  "sensenova_sample_kv_cache_streaming", "sensenova_mot_pageable_staging",
  "sensenova_mot_overlap_transfer", "sensenova_train_fm_modules",
  "block_swap_h2d_only", "block_swap_ring_size", "num_optimizer_groups",
  "bundle_vae",
  "activation_dispatch_enable", "activation_dispatch_margin_gb",
  "activation_dispatch_seed_coef", "activation_dispatch_residual_frac",
  "activation_dispatch_threshold_mb",
  "multi_noise_timesteps", "multi_noise_mode", "stratified_timesteps",
  "grad_timestep_cosine_probe", "grad_timestep_cosine_sketch_dim", "trajectory_blend_alpha",
  "snr_regularization_weight", "snr_timestep_adaptive", "snr_penalty_mode",
  "energy_regularization_weight", "energy_timestep_adaptive", "energy_penalty_mode",
  "energy_normalize_by_pixels",
  "noise_process", "prediction_target", "strict_validation", "sdxl_vae_type",
  "sdxl_te_type", "sdxl_te_hidden_layer", "sdxl_te_max_len", "sdxl_te_train_encoder",
  "controlnet_type", "controlnet_init_from_unet",
  "lllite_conditioning_channels", "lllite_rank",
  "condition_cache_mode",
  "conditioning_mode", "outpaint_crop_min_area", "outpaint_crop_max_area",
  "outpaint_edge_anchor_prob", "outpaint_corner_anchor_prob", "outpaint_mask_channel",
  "outpaint_known_loss_weight", "outpaint_seam_loss_boost", "outpaint_seam_ring_width",
  "outpaint_seam_grad_lambda", "outpaint_loss_normalize",
  "sample_every", "sample_prompts", "sample_width", "sample_height",
  "sample_steps", "sample_cfg_scale", "sample_sampler", "sample_schedule_type", "sample_seed",
  "sample_cfg_schedule_type", "sample_cfg_schedule_min", "sample_cfg_schedule_max", "sample_cfg_schedule_power",
  "sample_cfg_rescale_snr_alpha", "sample_dynamic_threshold_percentile", "sample_dynamic_threshold_mimic_scale",
  "sample_nag_enable", "sample_nag_scale", "sample_nag_tau", "sample_nag_alpha", "sample_nag_sigma_end", "sample_nag_negative_prompt",
  "sensenova_sample_timestep_shift", "sensenova_sample_img_cfg_scale", "sensenova_sample_cfg_norm",
  "debug_latents", "debug_latents_every",
  "enable_bucketing", "bucket_strategy", "multi_resolution_mode",
  "crop_augment_enable", "crop_full_image_prob", "crop_max_bucket_prob",
  "crop_min_area_ratio", "crop_min_short_side_px", "crop_aspect_mode",
  "crop_position_mode", "crop_smaller_bucket_mode", "crop_smaller_scale_range",
  "full_crop_position_mode", "crop_microcond_mode", "crop_plan_seed",
  "cache_latents_to_disk", "force_recache",
  // Restored from the run being edited; excluded from PRESETS only
  // (a preset carries no dataset for a "force" to rescan).
  "rescan_before_training",
  "use_reference_images", "train_vision_encoder", "gradient_routing_ve",
  "vision_encoder_lr", "param_tracking", "param_tracking_interval",
  "relora_merge_every", "relora_merge_unit", "restart_warmup_steps",
  "optimizer_reset_strategy", "optimizer_pruning_ratio",
];

/** Request fields applyParamsToState() restores through its own coercion or
 *  nested-object branches instead of the PARAM_KEYS loop. */
// Request keys getRequestData() builds itself -- from a local text state, an
// either-or pair, a method gate or a derived flag -- rather than copying from
// `params`. Everything else in PARAM_KEYS is copied verbatim, so the request
// enumerates nothing: see `passThroughParams`.
const COMPUTED_REQUEST_KEYS = new Set<string>([
  "lora_rank", "lora_alpha", "lora_dtype",
  "adapter_algorithm", "weight_decompose", "adapter_config",
  "total_steps", "epochs", "learning_rate",
  "optimizer_beta1", "optimizer_beta2", "optimizer_epsilon",
  "optimizer_weight_decay", "optimizer_schedule_free_r", "optimizer_schedule_free_weight_lr_power",
  "resume_from_checkpoint", "unet_lr", "text_encoder_lr",
  "text_encoder_1_lr", "text_encoder_2_lr", "image_encoder_lr",
  "cfg_uncond_drop_rate", "minit2i_label_drop_rate", "controlnet_type",
  "controlnet_init_from_unet", "lllite_conditioning_channels", "lllite_rank",
  "condition_cache_mode", "conditioning_mode", "outpaint_crop_min_area",
  "outpaint_crop_max_area", "outpaint_edge_anchor_prob", "outpaint_corner_anchor_prob",
  "outpaint_mask_channel", "outpaint_known_loss_weight", "outpaint_seam_loss_boost",
  "outpaint_seam_ring_width", "outpaint_seam_grad_lambda", "outpaint_loss_normalize",
  "bucket_strategy", "multi_resolution_mode", "crop_augment_enable",
  "crop_smaller_scale_range", "cache_latents_to_disk", "force_recache",
  "force", "rescan_before_training", "use_reference_images",
  "train_vision_encoder", "gradient_routing_ve", "vision_encoder_lr",
  "relora_merge_every", "relora_merge_unit", "restart_warmup_steps",
  "optimizer_reset_strategy", "optimizer_pruning_ratio",
]);

/** The PARAM_KEYS the request copies straight from `params`. */
export function passThroughParams(
  params: Record<string, any>
): Record<string, any> {
  const out: Record<string, any> = {};
  for (const key of PARAM_KEYS) {
    if (!COMPUTED_REQUEST_KEYS.has(key)) out[key] = params[key];
  }
  return out;
}

const PARAM_EXTRA_RESTORE_KEYS: string[] = [
  "base_resolutions", "regularization_type", "vision_encoder_path",
  "controlnet_pretrained_path", "condition_preprocessors",
  "timestep_sampling", "priority_training",
];

/**
 * What a preset deliberately does NOT carry. Excluded from SAVING only -- most
 * are still in PARAM_KEYS because edit mode must restore them from a run's own
 * YAML, and this list is not the restore list: a key here and NOWHERE else is
 * silently lost on edit (that was rescan_before_training). Everything else in
 * the request is saved; adding an entry is a visible choice, pinned by
 * training_preset_payload_test.py and training_edit_restore_coverage_test.py.
 *
 * The auxiliary model paths (vision_encoder_path, repa_tagger_model_dir,
 * repa_siglip2_repo, minit2i_flan_t5_path, minit2i_scratch_init_from) are
 * deliberately KEPT: which encoder to train against is a preference, and the
 * use_reference_images flip a restored vision_encoder_path causes is the point
 * of setting one.
 */
export const PRESET_EXCLUDED_KEYS: string[] = [
  "dataset_configs",
  "base_model_path",
  "run_name",
  "training_method",          // stored as the preset's own column
  "gpu_index",                // a device on this machine, not a recipe
  "resume_from_checkpoint",   // another run's checkpoint, silently continued
  // Scans at run start exactly as configured -- but the preset excludes
  // dataset_configs, so a "force" would rescan a dataset nobody chose.
  "rescan_before_training",
];

const PRESET_RESTORABLE_KEYS: string[] = [...PARAM_KEYS, ...PARAM_EXTRA_RESTORE_KEYS];

const snakeToCamel = (key: string): string =>
  key.replace(/_([a-z0-9])/g, (_m, c: string) => c.toUpperCase());

/**
 * Presets written before the payload was derived stored camelCase. Mechanical
 * in this direction only -- camel -> snake is ambiguous (optimizerBeta1 comes
 * back as optimizer_beta_1) -- so the table is built from the snake keys.
 */
const PRESET_CAMEL_TO_SNAKE: Record<string, string | undefined> = (() => {
  const map: Record<string, string | undefined> = {};
  for (const key of PRESET_RESTORABLE_KEYS) map[snakeToCamel(key)] = key;
  return map;
})();

/** Flat camel keys predating the nested timestep_sampling object. */
const PRESET_LEGACY_TIMESTEP_KEYS: Record<string, string | undefined> = {
  timestepDistribution: "distribution",
  timestepMin: "min_timestep",
  timestepMax: "max_timestep",
  timestepMean: "mean",
  timestepStd: "std",
  timestepAlpha: "alpha",
  timestepBeta: "beta",
};

/** Params whose control keeps the raw text; old presets stored that text.
 *  A superset of PRESET_CLEARABLE_NUMERIC_KEYS below. */
const PRESET_NUMERIC_TEXT_KEYS = new Set<string>([
  "learning_rate", "optimizer_beta1", "optimizer_beta2", "optimizer_epsilon",
  "optimizer_weight_decay", "optimizer_schedule_free_r",
  "optimizer_schedule_free_weight_lr_power",
  "unet_lr", "text_encoder_lr", "text_encoder_1_lr", "text_encoder_2_lr",
  "image_encoder_lr",
]);

/**
 * Numeric-text controls that can legitimately be EMPTY. getRequestData() spells
 * empty as null (the LR overrides) or as an omitted key (the optimizer
 * hyperparameters); a preset stores null for both, so a load CLEARS the box
 * instead of leaving the previous preset's value in it -- which submit would
 * then use, because getRequestData() reads the text state, not params.
 * learning_rate and the two schedule-free fields are absent on purpose: they
 * are never unset, and their restore branches are not null-safe.
 */
export const PRESET_CLEARABLE_NUMERIC_KEYS: string[] = [
  "unet_lr", "text_encoder_lr", "text_encoder_1_lr", "text_encoder_2_lr",
  "image_encoder_lr",
  "optimizer_beta1", "optimizer_beta2", "optimizer_epsilon",
  "optimizer_weight_decay",
];

/**
 * Normalize a stored preset into the snake_case dict applyParamsToState()
 * reads. Both spellings are accepted; a key belonging to neither (e.g.
 * optimizerIsPaged, removed) is dropped rather than thrown on.
 */
export function presetConfigToParams(config: Record<string, any>): Record<string, any> {
  const restorable = new Set(PRESET_RESTORABLE_KEYS);
  const excluded = new Set(PRESET_EXCLUDED_KEYS);
  const incoming: Record<string, any> = {};
  const timestep: Record<string, any> = {};

  const clearable = new Set(PRESET_CLEARABLE_NUMERIC_KEYS);
  const put = (key: string, value: any) => {
    if (excluded.has(key) || value === undefined) return;
    if (typeof value === "string" && PRESET_NUMERIC_TEXT_KEYS.has(key)) {
      const parsed = parseFloat(value);
      // An old preset stored "" for an empty box; that is "unset", not garbage.
      if (Number.isNaN(parsed)) {
        if (clearable.has(key)) incoming[key] = null;
        return;
      }
      incoming[key] = parsed;
      return;
    }
    incoming[key] = value;
  };

  const entries = Object.entries(config || {});
  // Current spelling first, so it wins if a blob carries both.
  for (const [key, value] of entries) {
    if (restorable.has(key)) put(key, value);
  }
  for (const [key, value] of entries) {
    if (restorable.has(key)) continue;
    const nested = PRESET_LEGACY_TIMESTEP_KEYS[key];
    if (nested !== undefined) {
      if (value !== undefined) timestep[nested] = value;
      continue;
    }
    const snake = PRESET_CAMEL_TO_SNAKE[key];
    if (snake !== undefined && incoming[snake] === undefined) put(snake, value);
  }

  if (Object.keys(timestep).length > 0) {
    incoming.timestep_sampling = { ...timestep, ...(incoming.timestep_sampling || {}) };
  }
  // R6 compat: attention_backend is authoritative; map the legacy bool when it
  // is the only spelling present, then keep the mirror consistent with it.
  if (incoming.attention_backend === undefined && incoming.use_flash_attention !== undefined) {
    incoming.attention_backend = incoming.use_flash_attention ? "flash" : "native";
  }
  if (incoming.attention_backend !== undefined) {
    incoming.use_flash_attention = incoming.attention_backend === "flash";
  }
  return incoming;
}
