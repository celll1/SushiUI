"use client";

import type {
  SenseNovaTaskView,
  SenseNovaTrainScope,
  TrainingRequiredValue,
  TrainingRunCreateRequest,
} from "@/utils/api";

// Architectures whose weight/training dtype the backend forces to BF16 whatever
// the dtype dropdowns say (train_runner.py, "forcing training_dtype=bf16").
// SD1.5, SDXL and Flux 2 are deliberately absent: they keep the configured dtype,
// which defaults to FP16. Used to decide whether a run will actually have BF16
// parameters, which is the only thing stochastic rounding can act on.
export const FORCED_BF16_ARCHITECTURES = new Set([
  "zimage", "anima", "ideogram4", "minit2i", "krea2", "qwen_image_21", "lens", "ltx2", "acestep",
  // MiniMax-H3: bf16 is not merely its native precision, it is the dtype its
  // weight-only FP8 Linears DEQUANTIZE INTO inside every forward. Left at the
  // non-bf16 default (fp32) the whole 50-block stack runs fp32 and the per-block
  // dequantized-weight transient roughly doubles -- silently.
  "minimax_h3",
]);

// Optimizer configuration: defines available options and defaults for each optimizer
export const OPTIMIZER_CONFIGS: Record<string, {
  label: string;
  supportsCautious?: boolean;
  defaults: {
    beta1?: string;
    beta2?: string;
    epsilon?: string;
    weight_decay?: string;
  };
}> = {
  "adamw": {
    label: "AdamW",
    defaults: { beta1: "0.9", beta2: "0.999", epsilon: "1e-8", weight_decay: "0.01" }
  },
  "adamw8bit": {
    label: "AdamW 8-bit",
    defaults: { beta1: "0.9", beta2: "0.999", epsilon: "1e-8", weight_decay: "0.01" }
  },
  "adamw8bit_ringbuffer": {
    label: "AdamW 8-bit Ring Buffer",
    supportsCautious: true,
    defaults: { beta1: "0.9", beta2: "0.999", epsilon: "1e-8", weight_decay: "0.01" }
  },
  "lion8bit": {
    label: "Lion 8-bit",
    defaults: { beta1: "0.9", beta2: "0.99", weight_decay: "0.01" }  // Lion uses different beta2
  },
  "lion8bit_ringbuffer": {
    label: "Lion 8-bit Ring Buffer",
    supportsCautious: true,
    defaults: { beta1: "0.9", beta2: "0.99", weight_decay: "0.01" }
  },
  "adafactor": {
    label: "Adafactor",
    defaults: { weight_decay: "0.01" }  // Adafactor has adaptive beta1/beta2
  }
};

// A conditional requirement's lift, as a phrase, used wherever a pin is
// rendered so a conditional entry never reads as an absolute one.
export const describeRequirementLift = (
  unless: Record<string, string | number | boolean>
): string =>
  Object.entries(unless)
    .map(([param, value]) => (typeof value === "boolean"
      ? `${param} is ${value ? "on" : "off"}`
      : `${param} = ${String(value)}`))
    .join(" and ");

// A control whose value the backend's capability matrix FIXES for the selected
// architecture and training method (`training_required_values`). Rendered under
// the pinned control so the value and the backend's reason for it are visible
// before submit rather than in a run-failed message afterwards. An entry
// carrying `unless` reached here only because the lift does NOT hold, so the
// note says what would release the pin instead of claiming the value is fixed.
export const RequiredValueNote = ({ entry }: { entry?: TrainingRequiredValue }) =>
  entry ? (
    <p className="text-xs text-amber-400 mt-1">
      {entry.unless
        ? `Fixed at ${String(entry.value)} unless ${describeRequirementLift(entry.unless)}: ${entry.reason}`
        : `Fixed at ${String(entry.value)} for this architecture and training method: ${entry.reason}`}
    </p>
  ) : null;

// Adapter algebras, in the spelling the API uses. Descriptions state the
// tensor form only: what each costs or is worth is not measured here.
export const ADAPTER_ALGORITHM_LABELS: Record<string, string> = {
  lora: "LoRA (low-rank)",
  loha: "LoHa (Hadamard product)",
  lokr: "LoKr (Kronecker product)",
};

export const SENSENOVA_SCOPE_OPTIONS: { value: SenseNovaTrainScope; label: string }[] = [
  { value: "understanding_vision", label: "Understanding vision + projector" },
  { value: "understanding_decoder", label: "Understanding decoder" },
  { value: "understanding_norms", label: "Understanding decoder norms" },
  { value: "shared", label: "Shared embeddings + LM head" },
  { value: "generation_decoder", label: "Generation decoder" },
  { value: "generation_norms", label: "Generation decoder norms" },
  { value: "generation_flow", label: "Generation vision + flow modules" },
];

export const newSenseNovaTaskView = (): SenseNovaTaskView => ({
  task: "i2t_caption",
  target_caption_types: [],
  hint_caption_types: [],
  weight: 1,
  loss_weight: 1,
  hint_dropout: 0.25,
  prompt_template_version: 1,
});

export const captionTypeList = (value: string): string[] =>
  value.split(",").map((part) => part.trim()).filter(Boolean);

// What each schedule does, in the vocabulary the backend registry defines.
// constant_with_warmup is accepted but not offered: it is the same curve as
// constant, which now applies the warmup too.
// Exported so the runtime retarget form offers the same vocabulary; this list
// is the one mirror of LR_SCHEDULER_NAMES and lr_schedule_vocabulary_test.py
// pins it against the registry.
export const LR_SCHEDULER_OPTIONS: { value: string; label: string; note: string }[] = [
  { value: "constant", label: "Constant",
    note: "Warmup, then holds the base LR for the rest of training." },
  { value: "linear", label: "Linear",
    note: "Decays in a straight line from the end of warmup to the floor at the end of the run." },
  { value: "cosine", label: "Cosine",
    note: "Cosine decay from the end of warmup to the floor at the end of the run, then holds the floor." },
  { value: "cosine_with_restarts", label: "Cosine with Restarts",
    note: "A cosine that restarts every cycle. Each restart is a step change back up to that cycle's peak." },
  { value: "polynomial", label: "Polynomial",
    note: "Linear decay to the floor (the exponent is 1)." },
  { value: "plateau_cosine_floor", label: "Plateau then Cosine Floor",
    note: "Warmup, holds the base LR flat, then cosine-decays to the floor and holds it." },
  { value: "wsd", label: "WSD (warmup / stable / decay)",
    note: "Holds the base LR until the decay start step, then decays in the chosen shape. With start step 0 nothing decays until a \"Decay now\" command is sent to the running run." },
  { value: "rex", label: "REX",
    note: "Decays from the end of warmup with the REX shape (1-q)/(1-q/2), which leaves the base LR at a finite slope and drops steeply at the end." },
];

export const ADAPTER_ALGORITHM_NOTES: Record<string, string> = {
  lora: "Two low-rank factors per target (lora_down, lora_up).",
  loha: "Element-wise product of two low-rank factorizations.",
  lokr: "Kronecker product of a full and a low-rank factor.",
};

export const DEFAULT_PARAMS: TrainingRunCreateRequest = {
  training_method: "lora",
  base_model_path: "",
  gpu_index: null,
  dataset_configs: [],
  total_steps: 1000,
  // Initialized (not undefined) so users can toggle the "Epochs" radio
  // and submit without having to touch the input — matches legacy behaviour
  // where useState(10) guaranteed the value was always present.
  // getRequestData() strips one of them based on `useEpochs`.
  epochs: 10,
  // Mirrors TRAINING_DEFAULTS["batch_size"]; overwritten by /schema/training-defaults
  // on startup, so this literal is the no-backend fallback.
  batch_size: 1,
  gradient_accumulation_steps: 1,
  max_grad_norm: 1.0,
  fused_grad_clip_factor: 0,
  fused_grad_clip_warmup_steps: 200,
  grad_spike_log_factor: 8.0,
  learning_rate: 1e-5,
  lr_scheduler: "constant",
  lr_warmup_steps: 0,
  lr_decay_start_ratio: 0.85,
  lr_floor_ratio: 0.25,
  lr_decay_start_step: 0,
  lr_decay_steps: 0,
  lr_decay_shape: "cosine",
  lr_cycle_steps: 0,
  lr_cycle_peak_decay: 1.0,
  lr_group_schedules: null,
  lr_layer_decay: 1.0,
  rewarmup_on_optimizer_reset: true,
  use_ema: false,
  ema_decay: 0.9999,
  ema_update_every: 1,
  ema_device: "cpu",
  optimizer: "adamw8bit",
  optimizer_cautious: false,
  optimizer_beta1: 0.9,
  optimizer_beta2: 0.999,
  optimizer_epsilon: 1e-8,
  optimizer_weight_decay: 0.01,
  optimizer_schedule_free: false,
  optimizer_schedule_free_r: 0.0,
  optimizer_schedule_free_weight_lr_power: 2.0,
  optimizer_use_radam: false,
  // Tri-state: null = "not specified", let the architecture decide.
  optimizer_stochastic_rounding: null,
  optimizer_state_host_resident: false,
  lora_rank: 16,
  lora_alpha: 16,
  lora_dtype: "fp32",
  adapter_algorithm: "lora",
  // No UI: accepted, refused (DoRA is Phase 3). Present so an edit-form PUT
  // round-trips the value the run was created with instead of dropping it.
  weight_decompose: false,
  // API-only (LoKr's factor/decompose_both). Dropping it on a PUT reset the
  // factorization to -1, changing every tensor shape and orphaning the run's
  // own checkpoints.
  adapter_config: null,
  relora_merge_every: 500,
  relora_merge_unit: "steps",
  restart_warmup_steps: 100,
  optimizer_reset_strategy: "full_reset",
  optimizer_pruning_ratio: 0.9,
  save_every: 100,
  save_every_unit: "steps",
  max_step_saves_to_keep: null,
  max_optimizer_saves_to_keep: 1,
  sample_every: 100,
  sample_prompts: [{ positive: "", negative: "" }],
  resume_from_checkpoint: "latest",
  sample_width: 1024,
  sample_height: 1024,
  sample_steps: 28,
  sample_cfg_scale: 7.0,
  sample_sampler: "euler",
  sample_schedule_type: "sgm_uniform",
  sample_seed: -1,
  debug_latents: false,
  debug_latents_every: 50,
  convergence_diagnostics_enable: false,
  convergence_diagnostics_interval: 100,
  crop_decode_loss_enable: false,
  crop_decode_loss_weight: 0.0,
  crop_decode_loss_margin_cells: 16,
  crop_decode_loss_out_cells: 32,
  crop_decode_loss_metric: "lpips",
  crop_decode_loss_snr_range: "",
  enable_bucketing: false,
  base_resolutions: [1024],
  bucket_strategy: "resize",
  multi_resolution_mode: "max",
  res_curriculum_enable: false,
  res_curriculum_warmup_steps: 0,
  res_curriculum_warmup_scale: 0.5,
  // Epoch-dynamic crop augmentation (SDXL only)
  crop_augment_enable: false,
  crop_full_image_prob: 0.7,
  crop_max_bucket_prob: 0.7,
  crop_min_area_ratio: 0.25,
  crop_min_short_side_px: 512,
  crop_aspect_mode: "source",
  crop_position_mode: "random",
  crop_smaller_bucket_mode: "base_res",
  crop_smaller_scale_range: [0.5, 0.9],
  full_crop_position_mode: "center",
  crop_microcond_mode: "kohya",
  crop_plan_seed: 0,
  train_unet: true,
  train_text_encoder: false,
  train_image_encoder: false,
  unet_lr: null,
  text_encoder_lr: null,
  text_encoder_1_lr: null,
  text_encoder_2_lr: null,
  image_encoder_lr: null,
  weight_dtype: "fp32",
  training_dtype: "fp16",
  output_dtype: "fp32",
  vae_dtype: "fp16",
  mixed_precision: true,
  gradient_checkpointing: true,
  torch_compile: "off",
  torch_compile_dynamic: null,
  // Attention backend for training: "native" | "flash" | "tq" (sage is inference-only).
  // Overwritten by trainingDefaults on startup; literal here is the no-backend fallback.
  attention_backend: "native",
  // DEPRECATED compat mirror of attention_backend (true ONLY for flash; native/tq -> false).
  // Kept synchronized on every UI change; attention_backend is authoritative.
  use_flash_attention: false,
  // Attention implementation registry: "conduit" | "diffusers". Selects WHICH registry
  // runs the kernel (orthogonal to attention_backend). Overwritten by trainingDefaults
  // on startup; literal here is the no-backend fallback. Affects SDXL/SD1.5 training.
  attention_impl: "conduit",
  tq_backward_mode: "triton",
  min_snr_gamma: 5.0,
  reconstruction_loss_weight: 0.0,
  // Deliberately unset, not 0: "not supplied" resolves the per-architecture
  // default, while 0 explicitly disables the mechanism. getRequestData omits
  // the key while it is null/undefined so the backend sees the difference.
  cfg_uncond_drop_rate: undefined,
  cfg_uncond_drop_per_mnt: true,
  // MiniMax-H3 only: weight of the audio half of its joint objective.
  // Overwritten by trainingDefaults on startup; literal here is the
  // no-backend fallback (and matches TRAINING_DEFAULTS).
  audio_loss_weight: 1.0,
  text_encoding_mode: "swap_onthefly",
  text_encoding_swap_interval: 256,
  latent_encoding_mode: "swap_onthefly",
  latent_encoding_swap_interval: 256,
  // Online Danbooru augmentation (image-generation). Overwritten by
  // trainingDefaults on startup; literals here are the no-backend fallback.
  danbooru_aug_enable: false,
  danbooru_aug_queries: "",
  danbooru_aug_weight_static: 1.0,
  danbooru_aug_deficiency_enable: true,
  danbooru_aug_deficiency_min_count: 20,
  danbooru_aug_deficiency_top_k: 200,
  danbooru_aug_deficiency_manual: "",
  danbooru_aug_weight_deficiency: 1.0,
  danbooru_aug_injection_interval: 4,
  danbooru_aug_injection_ratio: 1.0,
  danbooru_aug_min_score: 0,
  danbooru_aug_max_posts_per_query: 200,
  danbooru_aug_api_interval: 1.4,
  danbooru_aug_dl_speed_kbps: 500,
  danbooru_speed_check_enable: true,
  danbooru_speed_degraded_kbps: 250,
  danbooru_speed_min_slow_streak: 8,
  danbooru_speed_min_slow_seconds: 90,
  danbooru_speed_cooldown_seconds: 3600,
  danbooru_aug_buffer_size: null,
  danbooru_aug_include_rating_tag: false,
  danbooru_aug_max_caption_tags: 0,
  danbooru_quality_tag_enable: false,
  danbooru_quality_tag_thresholds: "",
  danbooru_quality_tag_attach_negative: false,
  danbooru_aug_shuffle_tags: false,
  danbooru_aug_shuffle_keep_first_n: 0,
  danbooru_aug_tag_dropout_rate: 0.0,
  danbooru_aug_tag_dropout_keep_first_n: 0,
  danbooru_aug_caption_dropout_rate: 0.0,
  danbooru_aug_keep_tokens: 0,
  blocks_to_swap: 0,
  use_pinned_memory: false,
  sensenova_mot_phase_eviction: false,
  sensenova_four_phase_eviction: false,
  sensenova_four_phase_shared_prefix: false,
  sensenova_four_phase_grad_reduction: "sum",
  sensenova_full_finetune_save_format: "mixed",
  sensenova_sample_kv_cache_streaming: false,
  sensenova_mot_pageable_staging: false,
  sensenova_mot_overlap_transfer: false,
  sensenova_train_fm_modules: true,
  sensenova_train_generation_norms: true,
  sensenova_train_scopes: [],
  chimera_training_stage: "unet",
  chimera_flow_version: "auto",
  chimera_v2_parameterization: "auto",
  chimera_v2_latent_mean: null,
  chimera_v2_latent_centered_second_moment: null,
  chimera_v2_calibration: null,
  chimera_bridge_align_steps: 0,
  chimera_allow_unaligned_scratch: false,
  chimera_conditioning_cache: true,
  chimera_prefix_prefetch: true,
  chimera_prefix_prefetch_device: "auto",
  chimera_prefix_prefetch_depth: 1,
  chimera_bridge_lr: null,
  chimera_context_dropout: 0.1,
  chimera_clip_hidden_weight: null,
  chimera_clip_pooled_weight: null,
  block_swap_h2d_only: false,
  block_swap_ring_size: 2,
  num_optimizer_groups: 0,
  bundle_vae: false,
  vae_swap_source: "",
  sensenova_gen_patch: 0,
  sensenova_latent_refiner: "inherit",
  sensenova_refiner_width: 0,
  sensenova_refiner_depth: 0,
  sensenova_refiner_training_mode: "joint",
  sensenova_refiner_lr_factor: 1,
  sensenova_refiner_detach_mode: "anneal",
  sensenova_refiner_detach_steps: 1000,
  sensenova_noise_scale_gain: 0,
  sensenova_noise_scale_auto: false,
  activation_dispatch_enable: false,
  activation_dispatch_margin_gb: 1.0,
  activation_dispatch_seed_coef: 0.000024,
  activation_dispatch_residual_frac: 0.85,
  activation_dispatch_threshold_mb: 4,
  qwen_partition_training_enabled: false,
  qwen_partition_mode: "fixed",
  qwen_partition_fixed_count: 2,
  qwen_partition_halo_tokens: 0,
  qwen_partition_split_ratio_min: 0.35,
  qwen_partition_split_ratio_max: 0.65,
  qwen_partition_seed: 0,
  qwen_partition_gradient_checkpointing_blocks: null,
  qwen_partition_profile: false,
  multi_noise_timesteps: 1,
  multi_noise_mode: "independent",
  stratified_timesteps: true,
  grad_timestep_cosine_probe: false,
  grad_timestep_cosine_sketch_dim: 8,
  trajectory_blend_alpha: 0.7,
  timestep_sampling: {
    distribution: "uniform",
    min_timestep: 0.0,
    max_timestep: 1.0,
    // Off by default: a morph is never implied by picking a model or a
    // distribution, only by asking for one before a resume.
    morph: {
      enabled: false,
      steps: 2000,
      curve: "cosine",
      interpolation: "quantile",
      from: null,
    },
    adaptive: {
      mode: "off",
      warmup_updates: 2000,
      control_interval: 500,
      bins: 8,
      log_snr_min: -10,
      log_snr_max: 10,
      coverage_floor: 0.2,
      max_density_ratio: 2,
      controller_gain: 0.15,
      morph_updates: 1000,
      cooldown_updates: 500,
      min_observations: 128,
      auto_observe_controls: 3,
      auto_min_bin_observations: 8,
      auto_min_bin_probability: 0.01,
    },
  },
  regularization_type: null,
  snr_regularization_weight: 0.1,
  snr_timestep_adaptive: true,
  snr_penalty_mode: "relu",
  energy_regularization_weight: 0.05,
  energy_timestep_adaptive: true,
  energy_penalty_mode: "abs",
  energy_normalize_by_pixels: true,
  noise_process: "auto",
  prediction_target: "auto",
  strict_validation: false,
  use_reference_images: false,
  vision_encoder_path: null,
  train_vision_encoder: false,
  vision_encoder_lr: null,
  gradient_routing_ve: false,
  param_tracking: false,
  param_tracking_interval: 100,
  controlnet_type: "standard",
  controlnet_pretrained_path: null,
  controlnet_init_from_unet: true,
  lllite_conditioning_channels: 32,
  lllite_rank: 64,
  condition_preprocessors: null,
  condition_cache_mode: "on_the_fly",
  conditioning_mode: "preprocessor",
  outpaint_crop_min_area: 0.15,
  outpaint_crop_max_area: 0.8,
  outpaint_edge_anchor_prob: 0.34,
  outpaint_corner_anchor_prob: 0.33,
  outpaint_mask_channel: true,
  outpaint_known_loss_weight: 0.3,
  outpaint_seam_loss_boost: 0.0,
  outpaint_seam_ring_width: 1,
  outpaint_seam_grad_lambda: 0.0,
  outpaint_loss_normalize: false,
  rescan_before_training: "off",
};
/**
 * Compact labelled numeric input. Integer vs float is inferred from `step`.
 * Float fields render `step="any"`: a numeric step would restrict the field to
 * the `min + n*step` grid and let the spinner/wheel rewrite typed values.
 */
export function NumField({
  label,
  value,
  onChange,
  step = 1,
}: {
  label: string;
  value: number | undefined;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1">{label}</label>
      <input
        type="number"
        step={step % 1 === 0 ? step : "any"}
        value={value ?? ""}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") return;
          const v = step % 1 === 0 ? parseInt(raw, 10) : parseFloat(raw);
          if (!Number.isNaN(v)) onChange(v);
        }}
        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
      />
    </div>
  );
}
