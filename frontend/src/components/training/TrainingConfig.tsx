"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { X, Save, FolderOpen, Trash2 } from "lucide-react";
import { createTrainingRun, updateTrainingRun, listDatasets, Dataset, TrainingRun, getModels, DatasetConfigItem, getRandomCaption, getSamplers, getScheduleTypes, listTrainingPresets, createTrainingPreset, deleteTrainingPreset, TrainingPreset, getTrainingRunParams, updateTrainingConfig, getControlNets, SamplePrompt, TrainingRunCreateRequest, listTrainingRuns, trainingMethodUnsupportedReason, trainingFeatureUnsupportedReason, trainingRequiredValues, TrainingRequiredValue, trainingFeatureAdvisory, TrainingFeatureAdvisory, archDisplayName, cfgUncondDropDefault, trainingSampleParameterSupported, trainingSampleNote, trainableAdapterAlgorithms, adapterTrainingRefusalReason, weightDecomposeTrainable, decomposedAdapterFamily } from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";
import TextareaWithTagSuggestions from "../common/TextareaWithTagSuggestions";
import NumberInput from "../common/NumberInput";
import VisionEncoderSelector from "../common/VisionEncoderSelector";
import TimestepDistributionGraph from "./TimestepDistributionGraph";
import GpuSelect from "./GpuSelect";
import {
  PARAM_KEYS,
  PRESET_EXCLUDED_KEYS,
  PRESET_CLEARABLE_NUMERIC_KEYS,
  passThroughParams,
  presetConfigToParams,
} from "./trainingParams";

interface TrainingConfigProps {
  onClose: () => void;
  onRunCreated: (run: TrainingRun) => void;
  editRunId?: number | null;
  onRunUpdated?: (run: TrainingRun) => void;
}

interface DatasetConfig {
  dataset_id: number;
  caption_types: string[];
  filters: Record<string, any>;
  ve_reconstruction_mode?: boolean;
}

interface ModelInfo {
  name: string;
  path: string;
  type: string;
  architecture: string;
  size_gb?: number;
  source_dir: string;
}

// Architectures whose weight/training dtype the backend forces to BF16 whatever
// the dtype dropdowns say (train_runner.py, "forcing training_dtype=bf16").
// SD1.5, SDXL and Flux 2 are deliberately absent: they keep the configured dtype,
// which defaults to FP16. Used to decide whether a run will actually have BF16
// parameters, which is the only thing stochastic rounding can act on.
const FORCED_BF16_ARCHITECTURES = new Set([
  "zimage", "anima", "ideogram4", "minit2i", "krea2", "lens", "ltx2", "acestep",
  // MiniMax-H3: bf16 is not merely its native precision, it is the dtype its
  // weight-only FP8 Linears DEQUANTIZE INTO inside every forward. Left at the
  // non-bf16 default (fp32) the whole 50-block stack runs fp32 and the per-block
  // dequantized-weight transient roughly doubles -- silently.
  "minimax_h3",
]);

// Optimizer configuration: defines available options and defaults for each optimizer
const OPTIMIZER_CONFIGS: Record<string, {
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
const describeRequirementLift = (
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
const RequiredValueNote = ({ entry }: { entry?: TrainingRequiredValue }) =>
  entry ? (
    <p className="text-xs text-amber-400 mt-1">
      {entry.unless
        ? `Fixed at ${String(entry.value)} unless ${describeRequirementLift(entry.unless)}: ${entry.reason}`
        : `Fixed at ${String(entry.value)} for this architecture and training method: ${entry.reason}`}
    </p>
  ) : null;

// Adapter algebras, in the spelling the API uses. Descriptions state the
// tensor form only: what each costs or is worth is not measured here.
const ADAPTER_ALGORITHM_LABELS: Record<string, string> = {
  lora: "LoRA (low-rank)",
  loha: "LoHa (Hadamard product)",
  lokr: "LoKr (Kronecker product)",
};

const ADAPTER_ALGORITHM_NOTES: Record<string, string> = {
  lora: "Two low-rank factors per target (lora_down, lora_up).",
  loha: "Element-wise product of two low-rank factorizations.",
  lokr: "Kronecker product of a full and a low-rank factor.",
};

// ============================================================
// Single-state migration (Phase 3a foundation)
// ============================================================
// All training parameters will be progressively migrated into this single
// `params` object so request construction and config restoration share one state.
const DEFAULT_PARAMS: TrainingRunCreateRequest = {
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
  learning_rate: 1e-5,
  lr_scheduler: "constant",
  lr_warmup_steps: 0,
  lr_decay_start_ratio: 0.85,
  lr_floor_ratio: 0.25,
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
  cache_latents_to_disk: false,
  force_recache: false,
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
  sensenova_train_fm_modules: false,
  block_swap_h2d_only: false,
  block_swap_ring_size: 2,
  num_optimizer_groups: 0,
  bundle_vae: false,
  activation_dispatch_enable: false,
  activation_dispatch_margin_gb: 1.0,
  activation_dispatch_seed_coef: 0.000024,
  activation_dispatch_residual_frac: 0.85,
  activation_dispatch_threshold_mb: 4,
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


export default function TrainingConfig({ onClose, onRunCreated, editRunId, onRunUpdated }: TrainingConfigProps) {
  console.log(`[TrainingConfig] Component mounted/re-rendered, editRunId=${editRunId}`);

  // ============================================================
  // Single-state form (Phase 3a–3m complete)
  // ============================================================
  // All top-level TrainingRunCreateRequest fields live in `params`.
  // UI inputs read via const aliases (e.g. `const batchSize = params.batch_size ?? 1`)
  // and write via `updateParam("batch_size", v)`.
  //
  // Remaining useState declarations are strictly for:
  //   - API-loaded data (datasets, presets, samplers, ...)
  //   - UI-only toggles (showPresetDialog, loading, error, ...)
  //   - Local numeric-input text buffers (localLrText, localBeta1Text, ...)
  //     which preserve in-progress scientific-notation input and sync to
  //     params.* on blur
  //   - Derived UI states that feed into nested objects on submit
  //     (timestep_sampling, priority_training)
  //   - `useEpochs` radio state (total_steps vs epochs exclusive)
  //
  // Adding a new top-level param:
  //   1. Add field to TrainingRunCreateRequest in frontend/src/utils/api.ts
  //   2. Add backend Pydantic field
  //   3. Add default to DEFAULT_PARAMS above
  //   4. Add UI input: read `params.x`, write via `updateParam("x", v)`
  //   No changes to getRequestData/applyParamsToState required.
  const [params, setParams] = useState<TrainingRunCreateRequest>(DEFAULT_PARAMS);
  const { trainingDefaults, trainingSampleDefaultsByArch, timestepDefaultsByArch, bundleVaeDefaultsByArch, archCapabilities } = useStartup();

  // Apply backend-fetched defaults when they arrive (only for new runs, not edit mode)
  useEffect(() => {
    if (!trainingDefaults || editRunId) return;
    setParams(prev => ({ ...DEFAULT_PARAMS, ...(trainingDefaults as Partial<TrainingRunCreateRequest>) }));
  }, [trainingDefaults, editRunId]);

  // Edit mode gets no such replacement -- it would overwrite the run's own
  // values with defaults -- so a control added AFTER a run was created had
  // nothing to render: its key is in neither the run's YAML nor DEFAULT_PARAMS,
  // and the input came up empty. That is not "unset"; the backend resolves
  // TRAINING_DEFAULTS for an absent key, so the empty box stated something
  // untrue about the run. Run 121's config predates the three SenseNova preview
  // controls and showed exactly that.
  //
  // Back-fill only what the restore left undefined, so a value the YAML does
  // carry always wins, and only preview keys: cfg_uncond_drop_rate and its
  // deprecated twin distinguish "the caller said nothing" from an explicit 0.0,
  // and filling one in would convert one into the other. No sample key carries
  // that distinction. Order-independent -- the restore below merges by key, so
  // it does not matter whether the defaults or the run's params land first.
  useEffect(() => {
    if (!editRunId || !trainingDefaults) return;
    setParams(prev => {
      const next = { ...prev } as Record<string, unknown>;
      let changed = false;
      for (const [key, value] of Object.entries(trainingDefaults as Record<string, unknown>)) {
        if (key === "sample_prompts") continue;
        if (!key.startsWith("sample_") && !key.startsWith("sensenova_sample_")) continue;
        if (next[key] === undefined && value !== undefined) {
          next[key] = value;
          changed = true;
        }
      }
      return changed ? (next as typeof prev) : prev;
    });
  }, [trainingDefaults, editRunId]);

  const updateParam = useCallback(
    <K extends keyof TrainingRunCreateRequest>(
      key: K,
      value: TrainingRunCreateRequest[K]
    ) => {
      setParams((prev) => ({ ...prev, [key]: value }));
    },
    []
  );
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [runName, setRunName] = useState("");
  // "Copy settings from existing run" (new-run mode only).
  const [copySourceRuns, setCopySourceRuns] = useState<TrainingRun[]>([]);
  const [copyingFromRun, setCopyingFromRun] = useState(false);

  // Model architecture filter: the architectures the user has UNCHECKED. Kept
  // as an exclusion set (not one boolean per arch) so the offered checkboxes can
  // be derived from the model list itself — an architecture nobody added a flag
  // for is shown, instead of disappearing from the dropdown.
  const [hiddenArchs, setHiddenArchs] = useState<string[]>([]);

  // Flag to track if dtype settings have been explicitly set (from YAML or user)
  // When true, baseModelPath changes will NOT override dtype settings
  const dtypeExplicitlySetRef = useRef(false);

  // Flag to track if we are in the middle of restoring from YAML
  // When true, optimizer useEffect will NOT reset hyperparameters to defaults
  const restoringFromYAMLRef = useRef(false);

  // A preset writes the optimizer and its hyperparameters together, and the
  // optimizer effect would replace the restored betas with the new optimizer's
  // defaults. Only that half is skipped: the option clearing below it must run,
  // or the preset parks a ticked box the optimizer ignores.
  const skipOptimizerHyperparamResetRef = useRef(false);

  // Tracks the baseModelPath for which the per-arch default timestep_sampling has
  // already been applied, so model changes apply the model's default exactly once
  // while user edits (which don't change baseModelPath) are never clobbered.
  const lastTimestepModelRef = useRef<string | null>(null);
  // Same pattern for the per-arch default bundle_vae (sd15/sdxl/deus -> true).
  const lastBundleVaeModelRef = useRef<string | null>(null);
  const lastSampleDefaultsModelRef = useRef<string | null>(null);
  const sampleDefaultsExplicitlySetRef = useRef(false);

  // Tracks which editRunId has already been restored from YAML.
  // Prevents React StrictMode's double-invoked mount effect from calling
  // loadTrainingRunParams twice — the second async fetch would otherwise
  // overwrite user edits made between the two restorations.
  const loadedEditRunIdRef = useRef<number | null>(null);

  // Multiple datasets support
  const [datasetConfigs, setDatasetConfigs] = useState<DatasetConfig[]>([
    { dataset_id: 0, caption_types: [], filters: {} }
  ]);

  // Available caption types for each dataset
  // Caption types selection moved to Dataset Management > Caption Processing page

  const [trainingMethod, setTrainingMethod] = useState<"lora" | "relora" | "full_finetune" | "controlnet">("lora");
  const [baseModelPath, setBaseModelPath] = useState("");

  // From-scratch MiniT2I (in-memory; no init model written to disk). When enabled,
  // baseModelPath is set to the sentinel "scratch:minit2i:<variant>:<vae_type>" and
  // the trainer builds a random-initialized model for Full Fine-tune.
  const [fromScratchMiniT2I, setFromScratchMiniT2I] = useState(false);
  const [scratchVariant, setScratchVariant] = useState("b16");
  const [scratchVaeType, setScratchVaeType] = useState("sdxl");

  // ControlNet parameters
  // ControlNet parameters (Phase 3l: migrated to params)
  const controlnetType = (params.controlnet_type ?? "standard") as "standard" | "lllite";
  const controlnetPretrainedPath = params.controlnet_pretrained_path ?? "";
  const controlnetInitFromUnet = params.controlnet_init_from_unet ?? true;
  const [availableControlNets, setAvailableControlNets] = useState<{path: string; name: string}[]>([]);
  const llliteConditioningChannels = params.lllite_conditioning_channels ?? 32;
  const llliteRank = params.lllite_rank ?? 64;
  const conditionPreprocessors = params.condition_preprocessors ?? [];
  const conditionCacheMode = (params.condition_cache_mode ?? "on_the_fly") as "on_the_fly" | "pre_generate";
  const conditioningMode = (params.conditioning_mode ?? "preprocessor") as "preprocessor" | "outpaint";
  const outpaintCropMinArea = params.outpaint_crop_min_area ?? 0.15;
  const outpaintCropMaxArea = params.outpaint_crop_max_area ?? 0.8;
  const outpaintEdgeAnchorProb = params.outpaint_edge_anchor_prob ?? 0.34;
  const outpaintCornerAnchorProb = params.outpaint_corner_anchor_prob ?? 0.33;
  const outpaintMaskChannel = params.outpaint_mask_channel ?? true;
  const outpaintKnownLossWeight = params.outpaint_known_loss_weight ?? 0.3;
  const outpaintSeamLossBoost = params.outpaint_seam_loss_boost ?? 0.0;
  const outpaintSeamRingWidth = params.outpaint_seam_ring_width ?? 1;
  const outpaintSeamGradLambda = params.outpaint_seam_grad_lambda ?? 0.0;
  const outpaintLossNormalize = params.outpaint_loss_normalize ?? false;

  // ReLoRA parameters (Phase 3d: migrated to params)
  const reloraMergeEvery = params.relora_merge_every ?? 500;
  const reloraMergeUnit = params.relora_merge_unit ?? "steps";
  const restartWarmupSteps = params.restart_warmup_steps ?? 100;
  const optimizerResetStrategy = params.optimizer_reset_strategy ?? "full_reset";
  const optimizerPruningRatio = params.optimizer_pruning_ratio ?? 0.9;

  // Training parameters (Phase 3b: migrated to params)
  const [useEpochs, setUseEpochs] = useState(false);
  // Local text state for learning rate (preserves in-progress scientific notation input)
  const [localLrText, setLocalLrText] = useState<string>(String(DEFAULT_PARAMS.learning_rate ?? "1e-5"));
  // Convenience read-only aliases into params (used by existing UI code)
  const totalSteps = params.total_steps ?? 1000;
  const epochs = params.epochs ?? 10;
  const batchSize = params.batch_size ?? 1;
  const learningRate = localLrText;
  const lrScheduler = params.lr_scheduler ?? "constant";
  const lrWarmupSteps = params.lr_warmup_steps ?? 0;
  const lrDecayStartRatio = params.lr_decay_start_ratio ?? 0.85;
  const lrFloorRatio = params.lr_floor_ratio ?? 0.25;
  const useEma = params.use_ema ?? false;
  const emaDecay = params.ema_decay ?? 0.9999;
  const emaUpdateEvery = params.ema_update_every ?? 1;
  const emaDevice = params.ema_device ?? "cpu";
  const optimizer = params.optimizer ?? "adamw8bit";

  // Optimizer-specific options (Phase 3c: migrated to params)
  // Local text states preserve in-progress numeric input (e.g. "1e-")
  const [localBeta1Text, setLocalBeta1Text] = useState<string>("0.9");
  const [localBeta2Text, setLocalBeta2Text] = useState<string>("0.999");
  const [localEpsilonText, setLocalEpsilonText] = useState<string>("1e-8");
  const [localWeightDecayText, setLocalWeightDecayText] = useState<string>("0.01");
  const [localScheduleFreeRText, setLocalScheduleFreeRText] = useState<string>("0.0");
  const [localScheduleFreeWeightLrPowerText, setLocalScheduleFreeWeightLrPowerText] = useState<string>("2.0");
  // Convenience read-only aliases into params
  const optimizerCautious = params.optimizer_cautious ?? false;
  const optimizerBeta1 = localBeta1Text;
  const optimizerBeta2 = localBeta2Text;
  const optimizerEpsilon = localEpsilonText;
  const optimizerWeightDecay = localWeightDecayText;
  const optimizerScheduleFree = params.optimizer_schedule_free ?? false;
  const optimizerScheduleFreeR = localScheduleFreeRText;
  const optimizerScheduleFreeWeightLrPower = localScheduleFreeWeightLrPowerText;
  const optimizerUseRadam = params.optimizer_use_radam ?? false;
  // Tri-state: "auto" (null/undefined, architecture decides), "on", "off".
  const optimizerStochasticRounding: "auto" | "on" | "off" =
    params.optimizer_stochastic_rounding === true ? "on"
    : params.optimizer_stochastic_rounding === false ? "off"
    : "auto";

  // LoRA parameters (Phase 3d: migrated to params)
  const loraRank = params.lora_rank ?? 16;
  const loraAlpha = params.lora_alpha ?? 16;
  const loraDtype = params.lora_dtype ?? "fp32";
  const adapterAlgorithm = params.adapter_algorithm ?? "lora";
  const weightDecompose = params.weight_decompose ?? false;

  // Advanced (Phase 3e: migrated to params)
  const [availableCheckpoints, setAvailableCheckpoints] = useState<Array<{step: number, filename: string}>>([]);
  const saveEvery = params.save_every ?? 100;
  const saveEveryUnit = (params.save_every_unit ?? "steps") as "steps" | "epochs";
  const maxStepSavesToKeep = params.max_step_saves_to_keep ?? null;
  const maxOptimizerSavesToKeep = params.max_optimizer_saves_to_keep ?? DEFAULT_PARAMS.max_optimizer_saves_to_keep;
  const sampleEvery = params.sample_every ?? DEFAULT_PARAMS.sample_every!;
  const resumeFromCheckpoint = params.resume_from_checkpoint ?? null;

  // Sample generation (Phase 3e: migrated to params)
  const samplePrompts = params.sample_prompts ?? DEFAULT_PARAMS.sample_prompts ?? [];
  const setSamplePrompts = useCallback((next: SamplePrompt[] | ((prev: SamplePrompt[]) => SamplePrompt[])) => {
    setParams(prev => ({
      ...prev,
      sample_prompts: typeof next === "function"
        ? (next as (p: SamplePrompt[]) => SamplePrompt[])(prev.sample_prompts ?? [])
        : next,
    }));
  }, []);
  const sampleWidth = params.sample_width ?? DEFAULT_PARAMS.sample_width!;
  const sampleHeight = params.sample_height ?? DEFAULT_PARAMS.sample_height!;
  const selectedSampleDefaults = trainingSampleDefaultsByArch
    ? (trainingSampleDefaultsByArch[getModelArchitecture(baseModelPath)]
      || trainingSampleDefaultsByArch["_default"])
    : undefined;
  const sampleStepsDefault = (selectedSampleDefaults?.sample_steps as number | undefined)
    ?? DEFAULT_PARAMS.sample_steps!;
  const sampleCfgScaleDefault = (selectedSampleDefaults?.sample_cfg_scale as number | undefined)
    ?? DEFAULT_PARAMS.sample_cfg_scale!;
  const sampleSteps = params.sample_steps ?? sampleStepsDefault;
  const sampleCfgScale = params.sample_cfg_scale ?? sampleCfgScaleDefault;
  const sampleSampler = params.sample_sampler ?? DEFAULT_PARAMS.sample_sampler!;
  const sampleScheduleType = params.sample_schedule_type ?? DEFAULT_PARAMS.sample_schedule_type!;
  const sampleCfgScheduleType = params.sample_cfg_schedule_type ?? "";
  const sampleSeed = params.sample_seed ?? DEFAULT_PARAMS.sample_seed!;
  const [conditionImagePreviews, setConditionImagePreviews] = useState<Record<number, string>>({});
  const conditionImageInputRefs = useRef<Record<number, HTMLInputElement | null>>({});
  const [referenceImagePreviews, setReferenceImagePreviews] = useState<Record<number, string>>({});
  const referenceImageInputRefs = useRef<Record<number, HTMLInputElement | null>>({});

  // Debug options (Phase 3f: migrated to params)
  const debugLatents = params.debug_latents ?? false;
  const debugLatentsEvery = params.debug_latents_every ?? 50;

  // Reference image conditioning — Phase 3k: migrated to params
  const useReferenceImages = params.use_reference_images ?? false;

  // SigLIP2 Vision Encoder — Phase 3k: migrated to params
  const visionEncoderPath = params.vision_encoder_path ?? "";
  const trainVisionEncoder = params.train_vision_encoder ?? false;
  const gradientRoutingVE = params.gradient_routing_ve ?? false;
  const [localVisionEncoderLrText, setLocalVisionEncoderLrText] = useState<string>("");
  const visionEncoderLr = localVisionEncoderLrText;

  // Parameter change tracking — Phase 3k: migrated to params
  const paramTracking = params.param_tracking ?? false;
  const paramTrackingInterval = params.param_tracking_interval ?? 100;

  // Priority training (one entry per line in textarea)
  const [priorityEnabled, setPriorityEnabled] = useState(false);
  const [priorityText, setPriorityText] = useState("");  // newline-separated entries
  const [priorityMultiplier, setPriorityMultiplier] = useState(1);
  const [priorityExpanded, setPriorityExpanded] = useState(false);  // expand textarea modal

  // Bucketing options (Phase 3f: migrated to params)
  const enableBucketing = params.enable_bucketing ?? false;
  const baseResolutions = params.base_resolutions ?? [1024];
  const bucketStrategy = (params.bucket_strategy ?? "resize") as "resize" | "crop" | "random_crop";
  const multiResolutionMode = (params.multi_resolution_mode ?? "max") as "max" | "random";
  const forceRecache = params.force_recache ?? false;

  // Outside bucketing the trainer uses only max(base_resolutions) as an area
  // ceiling, so retain one value rather than presenting inert extra choices.
  useEffect(() => {
    if (!enableBucketing && baseResolutions.length > 1) {
      updateParam("base_resolutions", [Math.max(...baseResolutions)]);
    }
  }, [enableBucketing, baseResolutions, updateParam]);

  // Component-specific training (Phase 3g: migrated to params)
  const trainUnet = params.train_unet ?? true;
  const trainTextEncoder = params.train_text_encoder ?? true;
  const trainImageEncoder = params.train_image_encoder ?? false;
  // Local text states preserve in-progress numeric input (scientific notation)
  const [localUnetLrText, setLocalUnetLrText] = useState<string>("");
  const [localTextEncoderLrText, setLocalTextEncoderLrText] = useState<string>("");
  const [localTextEncoder1LrText, setLocalTextEncoder1LrText] = useState<string>("");
  const [localTextEncoder2LrText, setLocalTextEncoder2LrText] = useState<string>("");
  const [localImageEncoderLrText, setLocalImageEncoderLrText] = useState<string>("");
  const unetLr = localUnetLrText;
  const textEncoderLr = localTextEncoderLrText;
  const textEncoder1Lr = localTextEncoder1LrText;
  const textEncoder2Lr = localTextEncoder2LrText;
  const imageEncoderLr = localImageEncoderLrText;

  // Precision and dtype settings (Phase 3h: migrated to params)
  const weightDtype = params.weight_dtype ?? "fp32";
  const trainingDtype = params.training_dtype ?? "fp16";
  const outputDtype = params.output_dtype ?? "fp32";
  const vaeDtype = params.vae_dtype ?? "fp16";
  const mixedPrecision = params.mixed_precision ?? true;
  // attention_backend is authoritative; use_flash_attention is a derived compat mirror.
  const attentionBackend = params.attention_backend ?? "native";
  // Attention implementation registry (conduit|diffusers). See DEFAULT_CONFIG note.
  const attentionImpl = params.attention_impl ?? "conduit";
  const minSnrGamma = params.min_snr_gamma ?? 5.0;
  const reconstructionLossWeight = params.reconstruction_loss_weight ?? 0.0;
  const audioLossWeight = params.audio_loss_weight ?? 1.0;

  // Text encoding mode (Phase 3i: migrated to params)
  const textEncodingMode = params.text_encoding_mode ?? "swap_onthefly";
  const textEncodingSwapInterval = params.text_encoding_swap_interval ?? 256;

  // Latent encoding mode
  const latentEncodingMode = params.latent_encoding_mode ?? "swap_onthefly";
  const latentEncodingSwapInterval = params.latent_encoding_swap_interval ?? 256;
  const usesLatentDiskCache = latentEncodingMode === "pre_encoded_cache";

  // Block Swap settings (training VRAM optimization)
  const blocksToSwap = params.blocks_to_swap ?? 0;
  const usePinnedMemory = params.use_pinned_memory ?? false;
  const numOptimizerGroups = params.num_optimizer_groups ?? 0;

  // Per-bucket activation offload dispatcher
  const activationDispatchEnable = params.activation_dispatch_enable ?? false;
  const activationDispatchMarginGb = params.activation_dispatch_margin_gb ?? 1.0;

  // Multi Noise-Timestep (MNT) settings
  const multiNoiseTimesteps = params.multi_noise_timesteps ?? 1;
  const multiNoiseMode = params.multi_noise_mode ?? "independent";
  const trajectoryBlendAlpha = params.trajectory_blend_alpha ?? 0.7;
  const [timestepDistribution, setTimestepDistribution] = useState<string>("uniform");
  const [timestepMin, setTimestepMin] = useState<number>(0.0);
  const [timestepMax, setTimestepMax] = useState<number>(1.0);
  // Distribution-specific parameters
  const [timestepMean, setTimestepMean] = useState<number>(0.0);  // For logit_normal/normal
  const [timestepStd, setTimestepStd] = useState<number>(1.0);    // For logit_normal/normal
  const [timestepAlpha, setTimestepAlpha] = useState<number>(2.0); // For beta
  const [timestepBeta, setTimestepBeta] = useState<number>(2.0);   // For beta

  // Regularization settings (prevent overbaking)
  // Regularization (Phase 3j: migrated to params)
  const regularizationType = params.regularization_type ?? "none";
  const snrRegularizationWeight = params.snr_regularization_weight ?? 0.0;
  const snrTimestepAdaptive = params.snr_timestep_adaptive ?? true;
  const snrPenaltyMode = params.snr_penalty_mode ?? "relu";
  const energyRegularizationWeight = params.energy_regularization_weight ?? 0.0;
  const energyTimestepAdaptive = params.energy_timestep_adaptive ?? true;
  const energyPenaltyMode = params.energy_penalty_mode ?? "under";
  const energyNormalizeByPixels = params.energy_normalize_by_pixels ?? true;

  // Unified Training Framework settings
  // Unified Training Framework (Phase 3j: migrated to params)
  const noiseProcess = params.noise_process ?? "auto";
  const predictionTarget = params.prediction_target ?? "auto";
  const strictValidation = params.strict_validation ?? false;

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // State for samplers and schedule types from API
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);

  // Presets
  const [presets, setPresets] = useState<TrainingPreset[]>([]);
  const [showPresetDialog, setShowPresetDialog] = useState(false);
  const [presetName, setPresetName] = useState("");
  const [presetDescription, setPresetDescription] = useState("");
  const [showLoadPresetDialog, setShowLoadPresetDialog] = useState(false);

  // Helper: Detect model architecture. These exist for arch-SPECIFIC config
  // blocks (an option only that architecture has). A capability gate must not
  // be written with them — use unsupportedTrainingFeature/Method below.
  // DEUS support removed
  // const isDEUSModel = (modelPath: string): boolean => {
  //   const model = availableModels.find(m => m.path === modelPath);
  //   return model?.architecture === "deus";
  // };

  const isFlux2Model = (modelPath: string): boolean => {
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture === "flux2";
  };

  const isAnimaModel = (modelPath: string): boolean => {
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture === "anima";
  };

  const isLensModel = (modelPath: string): boolean => {
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture === "lens";
  };

  const isIdeogram4Model = (modelPath: string): boolean => {
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture === "ideogram4";
  };

  const isMiniT2IModel = (modelPath: string): boolean => {
    if (modelPath.startsWith("scratch:minit2i:")) return true;
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture === "minit2i";
  };


  const isKrea2Model = (modelPath: string): boolean => {
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture === "krea2";
  };

  const isLtx2Model = (modelPath: string): boolean => {
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture === "ltx2";
  };

  const isSenseNovaModel = (modelPath: string): boolean => {
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture === "sensenova";
  };

  const getModelArchitecture = (modelPath: string): string | undefined => {
    if (modelPath.startsWith("scratch:minit2i:")) return "minit2i";
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture;
  };

  // Which end of the timestep_sampling [0,1] range is the CLEAN (noise-free)
  // latent for the selected architecture. Mirrors the backend's single source
  // of truth: backend/core/training/arch/*.py ArchHandler.timestep_convention
  // / resolve_timestep_convention(). SD15/SDXL flip between the two
  // conventions depending on noise_process (ops/sd_sdxl_ops.py ddpm/flow
  // branches), so "auto" cannot be resolved client-side.
  const getTimestepConvention = (modelPath: string, noiseProcessValue: string): "t0" | "t1" | "auto" => {
    const arch = getModelArchitecture(modelPath);
    if (arch === "sensenova" || arch === "minit2i") return "t1";
    if (arch === "sd15" || arch === "sdxl") {
      if (noiseProcessValue === "flow") return "t0";
      if (noiseProcessValue === "ddpm") return "t1";
      return "auto";
    }
    return "t0";
  };

  // The backend's own reason a training method is REFUSED for this base model,
  // or undefined when it is offered. Read from GET /schema/arch-capabilities
  // (`training_unsupported`), so the disabled control, its tooltip and the
  // trainer's exception all come from ONE table instead of the UI carrying a
  // second copy of the policy. Ideogram 4's full-FT block below predates this
  // table and keeps its hardcoded check; new entries need no UI change at all.
  const unsupportedTrainingMethod = (method: string): string | undefined =>
    trainingMethodUnsupportedReason(
      archCapabilities, getModelArchitecture(baseModelPath), method
    );

  // Same contract for a training-config FEATURE (block swap, reference images,
  // ...): the backend declares which mechanisms it does not have for this base
  // model, and the sections below hide/disable themselves from that. Nothing in
  // this file may re-derive it from an architecture name — that is how the
  // architecture filter above went stale for eight architectures.
  const unsupportedTrainingFeature = (feature: string): string | undefined =>
    trainingFeatureUnsupportedReason(
      archCapabilities, getModelArchitecture(baseModelPath), feature, trainingMethod
    );
  const blockSwapUnsupported = unsupportedTrainingFeature("block_swap");
  const fusedGroupsUnsupported = unsupportedTrainingFeature("fused_optimizer_groups");
  const referenceImagesUnsupported = unsupportedTrainingFeature("reference_images");
  const referenceConditioningEnabled = (
    (isSDOrSDXLModel(baseModelPath) && !!visionEncoderPath) ||
    ((isFlux2Model(baseModelPath) || isSenseNovaModel(baseModelPath)) && useReferenceImages)
  );
  const textEncoderTrainingUnsupported = unsupportedTrainingFeature("text_encoder_training");
  const trainingSamplesUnsupported = unsupportedTrainingFeature("training_samples");
  const trainingSampleArch = getModelArchitecture(baseModelPath);
  const sampleSamplerSupported = trainingSampleParameterSupported(
    archCapabilities, trainingSampleArch, "sample_sampler");
  const sampleScheduleSupported = trainingSampleParameterSupported(
    archCapabilities, trainingSampleArch, "sample_schedule_type");
  const sampleAdvancedCfgSupported = trainingSampleParameterSupported(
    archCapabilities, trainingSampleArch, "sample_cfg_schedule_type");
  const sensenovaTimestepShiftSupported = trainingSampleParameterSupported(
    archCapabilities, trainingSampleArch, "sensenova_sample_timestep_shift");
  const sensenovaImgCfgSupported = trainingSampleParameterSupported(
    archCapabilities, trainingSampleArch, "sensenova_sample_img_cfg_scale");
  const sensenovaCfgNormSupported = trainingSampleParameterSupported(
    archCapabilities, trainingSampleArch, "sensenova_sample_cfg_norm");
  const selectedTrainingSampleNote = trainingSampleNote(archCapabilities, trainingSampleArch);
  const vaeUnsupported = unsupportedTrainingFeature("vae");
  // Aligned CFG null-condition training. A string here means the backend cannot
  // build this architecture's inference uncond condition, and answers 400 for
  // any explicit rate -- 0 included -- so the control must not be offered.
  const cfgUncondDropUnsupported = unsupportedTrainingFeature("cfg_uncond_drop");
  const cfgUncondDropDefaultRate = cfgUncondDropDefault(
    archCapabilities, getModelArchitecture(baseModelPath));
  const minit2iLabelDropDefault = cfgUncondDropDefault(archCapabilities, "minit2i");
  const selectedModel = availableModels.find((model) => model.path === baseModelPath);
  const pixelSpaceMiniT2I = (
    (selectedModel?.architecture === "minit2i" && selectedModel.vae_type === "none")
    || (fromScratchMiniT2I && scratchVaeType === "none")
  );
  const latentEncodingAvailable = !vaeUnsupported && !pixelSpaceMiniT2I;

  // FIFTH capability axis, and the opposite claim from the one above: a feature
  // the backend DOES implement and DOES accept, with what it costs. Shown next
  // to the control, never used to hide or disable it — a control that is
  // disabled here while the API runs the parameter is the three-way
  // contradiction this axis replaced.
  const featureAdvisory = (feature: string): TrainingFeatureAdvisory | undefined =>
    trainingFeatureAdvisory(
      archCapabilities, getModelArchitecture(baseModelPath), feature, trainingMethod
    );
  const textEncoderTrainingAdvisory = featureAdvisory("text_encoder_training");
  const motEvictionAdvisory = featureAdvisory("sensenova_mot_eviction");
  const motEvictionUnsupported = unsupportedTrainingFeature("sensenova_mot_eviction");
  // fm_modules is generation-side, so the flag does nothing on an
  // understanding-only branch; the backend warns and proceeds rather than
  // refusing, so this is a note next to the control, not a disable.
  const fmModulesUnsupported = unsupportedTrainingFeature("sensenova_train_fm_modules");
  const fmModulesInertReason: string | undefined =
    !trainUnet
      ? "Train U-Net is off, so this run trains the understanding half only. fm_modules is generation-side: the backend warns and trains the decoder Linears alone."
      : undefined;
  // The preconditions train_runner checks before the checkpoint loads: three on
  // the split itself, plus the both-halves branch that its own required flag
  // (MoT Phase Eviction under a full fine-tune) demands. undefined = the split
  // is selectable; a string = why it is not, in the backend's own terms.
  const fourPhaseBlockedReason: string | undefined =
    trainingMethod !== "full_finetune"
      ? "The backward split is implemented for full fine-tuning only: it leaves the generation half on CPU at the step boundary, which is safe only on the fused backward route."
      : !trainTextEncoder
      ? "Needs Train Text Encoder: the split exists so a TRAINED understanding half can still be evicted. With that half frozen, MoT Phase Eviction alone already does this."
      : !params.sensenova_mot_phase_eviction
      ? "Needs MoT Phase Eviction: on its own the split only adds a second backward and a recomputed understanding forward, with both halves resident."
      : !trainUnet
      ? "Needs Train U-Net as well: MoT Phase Eviction under a full fine-tune requires both halves (the evictor moves whole halves and needs them to hold the same kind of weight, and a single-branch full fine-tune leaves the other one INT8), so an understanding-only run is refused before the model loads."
      : undefined;
  // The same refusal where it is actually raised. Shown, not disabling: the
  // eviction checkbox is legitimate on every other branch and method.
  const motEvictionBranchRefusal: string | undefined =
    trainingMethod === "full_finetune" && params.sensenova_mot_phase_eviction
      && !(trainUnet && trainTextEncoder)
      ? "MoT Phase Eviction under a full fine-tune is refused before the model loads unless BOTH Train U-Net and Train Text Encoder are on: a single-branch full fine-tune materializes only the half it trains and leaves the other INT8, and the evictor requires the two halves to hold the same kind of weight."
      : undefined;
  // The refusal in the OTHER direction (train_runner: train_text_encoder cannot
  // be combined with MoT Phase Eviction). Under full fine-tuning the split lifts
  // it; under LoRA nothing does — the split is full-fine-tune only — so the
  // remedy this names differs by method.
  const evictionPairRefusal: string | undefined =
    trainTextEncoder && params.sensenova_mot_phase_eviction
      && !params.sensenova_four_phase_eviction
      ? trainingMethod === "full_finetune"
        ? "Train Text Encoder with MoT Phase Eviction is refused before the model loads unless the Four-Phase Backward Split is on: the three-phase evictor moves the understanding half to CPU before its own backward. Turn the split on, or turn one of the other two off."
        : "Train Text Encoder with MoT Phase Eviction is refused before the model loads. The Four-Phase Backward Split that would lift it is implemented for full fine-tuning only, so here the remedy is to turn one of the two off."
      : undefined;

  // Global-norm clipping cannot happen on the fused route: every parameter is
  // updated from its own post-accumulate-grad hook, so no global norm ever
  // exists. Mirrors the three conditions in base_trainer.setup_optimizer, and
  // says the same thing base_trainer._warn_grad_clipping_ignored_under_fused
  // says as a trainer notice once the run has started.
  const gradClippingIgnoredReason: string | undefined = (() => {
    if ((params.max_grad_norm ?? 1.0) <= 0) return undefined;
    const fusedOptimizer = ["adafactor", "adamw8bit", "adamw8bit_ringbuffer",
      "lion8bit_ringbuffer"].includes(optimizer);
    const route =
      blocksToSwap > 0 && numOptimizerGroups > 0
        ? "Block Swap with fused optimizer groups"
        : blocksToSwap > 0 && fusedOptimizer
        ? `Block Swap with ${optimizer}`
        : isSenseNovaModel(baseModelPath) && trainingMethod === "full_finetune"
          && numOptimizerGroups === 0 && fusedOptimizer
        ? "a SenseNova full fine-tune"
        : undefined;
    if (!route) return undefined;
    // The remedy is route-dependent. A SenseNova full fine-tune ALREADY has
    // Block Swap off and 0 optimizer groups (its contract pins both), and the
    // trainer refuses the unfused shape, so "take the non-fused route" would be
    // advice the user has already followed and that cannot be followed further.
    const remedy =
      route === "a SenseNova full fine-tune"
        ? "There is no non-fused route to take here: this architecture's full-fine-tune contract already pins Block Swap off and 0 optimizer groups, and it accepts only the fused per-parameter optimizer, so nothing in this form can turn global-norm clipping back on. Use LoRA if you need it."
        : "Set it to 0 to silence the notice, or clip by taking the non-fused route (Block Swap off and 0 optimizer groups).";
    return `NOT APPLIED on this route: ${route} applies each parameter's update as soon as that parameter's gradient exists, so the global norm this threshold needs never exists. No clipping of any kind happens, and the run reports this once as a trainer notice. ${remedy}${
      optimizer === "adafactor"
        ? " Adafactor still applies its own clip_threshold, which bounds the RMS of each parameter's update independently — a different mechanism, unaffected by this field."
        : ""}`;
  })();

  // FOURTH capability axis: config values this base model REQUIRES under the
  // selected method (SenseNova's full-fine-tune contract fixes the optimizer
  // and accumulation). The backend refuses a run that violates
  // one, before the model loads, so the controls below are pinned to the value
  // rather than offering a default the run would be rejected for. Derived from
  // arch + method, so switching either unpins whatever is no longer required,
  // and from the run's own lift params, so a CONDITIONAL entry (`unless`) is
  // resolved rather than presented as an absolute pin.
  const baseModelArch = getModelArchitecture(baseModelPath) ?? "";
  // Which params can LIFT a conditional requirement here, read off the served
  // table: the backend owns that list (SenseNova's batch_size names
  // enable_bucketing), and a copy kept here would go stale silently.
  const liftParams = useMemo(() => {
    const entries: Record<string, TrainingRequiredValue> =
      archCapabilities?.training_required_values?.[baseModelArch] ?? {};
    return Array.from(new Set(
      Object.values(entries).flatMap((entry) => Object.keys(entry.unless ?? {}))
    )).sort();
  }, [archCapabilities, baseModelArch]);
  // Keyed on the lift name=value pairs, never on `params`: the effect below
  // compares by identity, so a fresh object per keystroke would wipe the
  // "(changed from X)" record -- while a lift actually moving is a new
  // contract and should.
  const liftSignature = liftParams
    .map((param) => `${param}=${JSON.stringify((params as any)[param])}`)
    .join("&");
  const requiredValueConfig = useMemo<Record<string, any>>(
    () => Object.fromEntries(
      liftParams.map((param) => [param, (params as any)[param]])),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [liftSignature]
  );
  const requiredValues = useMemo(
    () => trainingRequiredValues(archCapabilities, baseModelArch, trainingMethod,
                                 requiredValueConfig),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [archCapabilities, baseModelPath, trainingMethod, availableModels,
     requiredValueConfig]
  );
  const requiredValue = (param: string): TrainingRequiredValue | undefined =>
    requiredValues[param];

  // Which adapter algebras this base model can actually TRAIN, from the
  // backend's own table. ReLoRA takes the ordinary branch only (its merge and
  // optimizer reset are not defined for the others), so the choice collapses
  // there rather than offering a run the backend refuses. Block swap does the
  // same: no offloader moves a LoHa/LoKr factor.
  const adapterAlgorithmChoices = useMemo<Array<"lora" | "loha" | "lokr">>(
    () => (
      trainingMethod !== "lora" || blocksToSwap > 0
        ? ["lora"]
        : trainableAdapterAlgorithms(archCapabilities, baseModelArch)
    ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [archCapabilities, baseModelArch, trainingMethod, blocksToSwap]
  );
  // Why the choice collapsed, when it did. The two settings-driven causes are
  // the non-obvious ones -- without this the control simply vanishes when block
  // swap goes on or the method changes.
  const adapterAlgorithmCollapsedNote: string | undefined = (
    adapterAlgorithmChoices.length > 1 ? undefined
    : trainingMethod !== "lora"
      ? "ReLoRA trains the ordinary low-rank branch only: its merge, reinitialize and optimizer reset are not defined for LoHa or LoKr."
    : blocksToSwap > 0
      ? "Block swap is on, so only ordinary LoRA is offered: a LoHa or LoKr branch owns bare parameters, which no block offloader moves."
    : adapterTrainingRefusalReason(archCapabilities, baseModelArch, "loha")
  );
  // Weight decomposition (DoRA/DoHa/DoKr) is the SECOND axis, not a fourth
  // algorithm, so it is offered per (algorithm, arch) pair. The two settings
  // that contradict it are refused from the run's config before the model
  // loads, so the control is hidden rather than left to earn a 400: no block
  // offloader moves a bare dora_scale, and fp8_base_dtype quantizes the very
  // base weight the magnitude epilogue divides by.
  const weightDecomposeAvailable = (
    trainingMethod === "lora"
    && blocksToSwap === 0
    && !params.fp8_base_dtype
    && weightDecomposeTrainable(archCapabilities, baseModelArch, adapterAlgorithm)
  );
  const weightDecomposeUnavailableNote: string | undefined = (
    weightDecomposeAvailable ? undefined
    : trainingMethod !== "lora"
      ? "ReLoRA trains the ordinary low-rank branch only."
    : blocksToSwap > 0
      ? "Block swap is on: no block offloader moves a bare dora_scale, so a decomposed branch would stay on the host."
    : params.fp8_base_dtype
      ? "An FP8 base is selected: the magnitude epilogue reads the base weight's direction and norm every forward."
    // Silent where the architecture simply has no LyCORIS row: the algorithm
    // select is hidden there too, and a decomposition refusal under a plain
    // LoRA panel names a feature the user was never offered.
    : adapterAlgorithmChoices.length > 1
      ? adapterTrainingRefusalReason(archCapabilities, baseModelArch,
                                     decomposedAdapterFamily(adapterAlgorithm))
    : undefined
  );

  // param -> the value it held before this component pinned it, so the banner
  // names what was replaced instead of just saying something was.
  const [contractAdjusted, setContractAdjusted] = useState<Record<string, string>>({});
  // Identity of the requirement set the record above belongs to; a new arch or
  // method starts a new record rather than accumulating across contracts.
  const pinnedForRef = useRef<Record<string, TrainingRequiredValue> | null>(null);

  // MiniMax-H3 is the only architecture that reads audio_loss_weight (the only
  // one whose packed training sequence carries audio rows), so its control is
  // shown only for it rather than as a knob that silently does nothing.
  const isMiniMaxH3Model = getModelArchitecture(baseModelPath) === "minimax_h3";

  function isSDOrSDXLModel(modelPath: string): boolean {
    const arch = getModelArchitecture(modelPath);
    return arch === "sd15" || arch === "sdxl";
  }
  const isSDXLModel = (modelPath: string): boolean =>
    getModelArchitecture(modelPath) === "sdxl";

  // The architectures actually present in the model list, labelled from the
  // backend's ARCH_DISPLAY_NAMES (GET /schema/arch-capabilities). Both the list
  // and the labels therefore come from the backend; adding an architecture needs
  // no edit here.
  const archFilterOptions = Array.from(
    new Set(availableModels.map((m) => m.architecture).filter(Boolean))
  )
    .map((arch) => ({ arch, label: archDisplayName(archCapabilities, arch) }))
    .sort((a, b) => a.label.localeCompare(b.label));

  // Filter models by architecture (unchecked = hidden; unknown arch = shown).
  const filteredModels = availableModels.filter(
    (model) => !hiddenArchs.includes(model.architecture)
  );

  // ============================================================
  // Centralized state <-> params dict conversion
  // ============================================================
  // These two functions are the SINGLE SOURCE OF TRUTH for which
  // training parameters flow through the form. handleSubmit() and
  // loadTrainingRunParams() both delegate to them, so adding a new
  // field requires updating only:
  //   1. The useState declaration above
  //   2. getRequestData() (build outgoing dict)
  //   3. applyParamsToState() (restore from incoming dict)
  // and the UI <input> element. Preset save/load derives from the same two,
  // so it needs no edit.
  // ============================================================

  /**
   * Build the outgoing requestData dict from current useState values.
   * Used by handleSubmit() and Loop generation stepParams.
   */
  const getRequestData = useCallback((): any => {
    return {
      ...passThroughParams(params),
      dataset_configs: datasetConfigs.filter(c => c.dataset_id !== 0),
      run_name: runName.trim() || undefined,
      training_method: trainingMethod,
      base_model_path: baseModelPath.trim(),
      // MiniT2I config (sent so UI values reach the backend, not just defaults).
      // The two CFG null-drop keys are OMITTED when unset rather than sent as
      // null: the backend distinguishes "not supplied" (resolve the
      // per-architecture default) from an explicit value, and a key sent as
      // null on every submit would make every run look explicit.
      // Sending both is a 400 (they set the same rate), and an edit form loads
      // the deprecated key back from an older run's config, so the new key wins
      // here rather than colliding with it.
      ...(params.cfg_uncond_drop_rate != null
        ? { cfg_uncond_drop_rate: params.cfg_uncond_drop_rate }
        : params.minit2i_label_drop_rate != null
        ? { minit2i_label_drop_rate: params.minit2i_label_drop_rate }
        : {}),
      total_steps: useEpochs ? undefined : params.total_steps,
      epochs: useEpochs ? params.epochs : undefined,
      learning_rate: parseFloat(localLrText),
      optimizer_beta1: localBeta1Text ? parseFloat(localBeta1Text) : undefined,
      optimizer_beta2: localBeta2Text ? parseFloat(localBeta2Text) : undefined,
      optimizer_epsilon: localEpsilonText ? parseFloat(localEpsilonText) : undefined,
      optimizer_weight_decay: localWeightDecayText ? parseFloat(localWeightDecayText) : undefined,
      optimizer_schedule_free_r: localScheduleFreeRText ? parseFloat(localScheduleFreeRText) : 0.0,
      optimizer_schedule_free_weight_lr_power: localScheduleFreeWeightLrPowerText ? parseFloat(localScheduleFreeWeightLrPowerText) : 2.0,
      lora_rank: (trainingMethod === "lora" || trainingMethod === "relora") ? params.lora_rank : undefined,
      lora_alpha: (trainingMethod === "lora" || trainingMethod === "relora") ? params.lora_alpha : undefined,
      lora_dtype: (trainingMethod === "lora" || trainingMethod === "relora") ? params.lora_dtype : undefined,
      // ReLoRA is deliberately excluded: its merge/reinitialize and optimizer
      // reset are defined for the ordinary low-rank branch alone, and the
      // backend refuses the combination.
      adapter_algorithm: trainingMethod === "lora" ? params.adapter_algorithm : undefined,
      // Both methods write these into the YAML (the ReLoRA generator derives
      // from the LoRA one), so both must send them back or the PUT resets them.
      weight_decompose: (trainingMethod === "lora" || trainingMethod === "relora")
        ? params.weight_decompose : undefined,
      adapter_config: (trainingMethod === "lora" || trainingMethod === "relora")
        ? params.adapter_config : undefined,
      ...(trainingMethod === "relora" ? {
        relora_merge_every: params.relora_merge_every,
        relora_merge_unit: params.relora_merge_unit,
        restart_warmup_steps: params.restart_warmup_steps,
        optimizer_reset_strategy: params.optimizer_reset_strategy,
        optimizer_pruning_ratio: params.optimizer_pruning_ratio,
      } : {}),
      resume_from_checkpoint: params.resume_from_checkpoint || undefined,
      // Without bucketing only one area ceiling is meaningful. Normalize old
      // presets here too, even before the UI normalization effect has rendered.
      base_resolutions: !params.enable_bucketing && (params.base_resolutions?.length ?? 0) > 1
        ? [Math.max(...params.base_resolutions!)]
        : params.base_resolutions,
      bucket_strategy: params.enable_bucketing ? params.bucket_strategy : undefined,
      multi_resolution_mode: params.enable_bucketing ? params.multi_resolution_mode : undefined,
      // Epoch-dynamic crop augmentation (SDXL only; requires bucketing)
      crop_augment_enable: params.enable_bucketing ? params.crop_augment_enable : false,
      crop_smaller_scale_range: params.crop_smaller_scale_range ?? [0.5, 0.9],
      // Legacy dataset mirror. latent_encoding_mode is authoritative.
      cache_latents_to_disk: params.latent_encoding_mode === "pre_encoded_cache",
      force_recache: params.latent_encoding_mode === "pre_encoded_cache"
        ? params.force_recache
        : false,
      unet_lr: localUnetLrText ? parseFloat(localUnetLrText) : null,
      text_encoder_lr: localTextEncoderLrText ? parseFloat(localTextEncoderLrText) : null,
      text_encoder_1_lr: localTextEncoder1LrText ? parseFloat(localTextEncoder1LrText) : null,
      text_encoder_2_lr: localTextEncoder2LrText ? parseFloat(localTextEncoder2LrText) : null,
      image_encoder_lr: localImageEncoderLrText ? parseFloat(localImageEncoderLrText) : null,
      use_reference_images: isSDOrSDXLModel(baseModelPath)
        ? !!params.vision_encoder_path
        : params.use_reference_images,
      vision_encoder_path: isSDOrSDXLModel(baseModelPath)
        ? params.vision_encoder_path || null
        : null,
      train_vision_encoder: isSDOrSDXLModel(baseModelPath)
        ? params.train_vision_encoder
        : false,
      vision_encoder_lr: localVisionEncoderLrText ? parseFloat(localVisionEncoderLrText) : null,
      gradient_routing_ve: isSDOrSDXLModel(baseModelPath)
        ? params.gradient_routing_ve
        : false,
      timestep_sampling: {
        distribution: timestepDistribution,
        min_timestep: timestepMin,
        max_timestep: timestepMax,
        ...(timestepDistribution === "logit_normal" || timestepDistribution === "lognormal" || timestepDistribution === "normal" ? {
          mean: timestepMean,
          std: timestepStd,
        } : {}),
        ...(timestepDistribution === "beta" ? {
          alpha: timestepAlpha,
          beta: timestepBeta,
        } : {}),
      },
      regularization_type: regularizationType !== "none" ? regularizationType : null,
      controlnet_type: trainingMethod === "controlnet" ? params.controlnet_type : undefined,
      controlnet_pretrained_path: trainingMethod === "controlnet" && params.controlnet_pretrained_path ? params.controlnet_pretrained_path : undefined,
      controlnet_init_from_unet: trainingMethod === "controlnet" ? params.controlnet_init_from_unet : undefined,
      lllite_conditioning_channels: trainingMethod === "controlnet" && params.controlnet_type === "lllite" ? params.lllite_conditioning_channels : undefined,
      lllite_rank: trainingMethod === "controlnet" && params.controlnet_type === "lllite" ? params.lllite_rank : undefined,
      condition_preprocessors: trainingMethod === "controlnet" && (params.condition_preprocessors?.length ?? 0) > 0 ? params.condition_preprocessors : undefined,
      condition_cache_mode: trainingMethod === "controlnet" && (params.condition_preprocessors?.length ?? 0) > 0 ? params.condition_cache_mode : undefined,
      conditioning_mode: trainingMethod === "controlnet" ? params.conditioning_mode : undefined,
      outpaint_crop_min_area: trainingMethod === "controlnet" && params.conditioning_mode === "outpaint" ? params.outpaint_crop_min_area : undefined,
      outpaint_crop_max_area: trainingMethod === "controlnet" && params.conditioning_mode === "outpaint" ? params.outpaint_crop_max_area : undefined,
      outpaint_edge_anchor_prob: trainingMethod === "controlnet" && params.conditioning_mode === "outpaint" ? params.outpaint_edge_anchor_prob : undefined,
      outpaint_corner_anchor_prob: trainingMethod === "controlnet" && params.conditioning_mode === "outpaint" ? params.outpaint_corner_anchor_prob : undefined,
      outpaint_mask_channel: trainingMethod === "controlnet" && params.conditioning_mode === "outpaint" ? params.outpaint_mask_channel : undefined,
      outpaint_known_loss_weight: trainingMethod === "controlnet" && params.conditioning_mode === "outpaint" ? params.outpaint_known_loss_weight : undefined,
      outpaint_seam_loss_boost: trainingMethod === "controlnet" && params.conditioning_mode === "outpaint" ? params.outpaint_seam_loss_boost : undefined,
      outpaint_seam_ring_width: trainingMethod === "controlnet" && params.conditioning_mode === "outpaint" ? params.outpaint_seam_ring_width : undefined,
      outpaint_seam_grad_lambda: trainingMethod === "controlnet" && params.conditioning_mode === "outpaint" ? params.outpaint_seam_grad_lambda : undefined,
      outpaint_loss_normalize: trainingMethod === "controlnet" && params.conditioning_mode === "outpaint" ? params.outpaint_loss_normalize : undefined,
      rescan_before_training: params.rescan_before_training ?? "off",
      priority_training: priorityEnabled && priorityText.trim() ? {
        entries: priorityText.trim().split("\n").map(line => line.trim()).filter(Boolean),
        multiplier: priorityMultiplier,
      } : undefined,
    };
  }, [
    // Core: entire params object (single source of truth for ~90 fields)
    params,
    // Non-params form state
    datasetConfigs, runName, trainingMethod, baseModelPath, useEpochs,
    // Local text states (numeric-input helpers; written on blur but read here)
    localLrText,
    localBeta1Text, localBeta2Text, localEpsilonText, localWeightDecayText,
    localScheduleFreeRText, localScheduleFreeWeightLrPowerText,
    localUnetLrText, localTextEncoderLrText, localTextEncoder1LrText,
    localTextEncoder2LrText, localImageEncoderLrText, localVisionEncoderLrText,
    // Derived-UI states that feed back into timestep_sampling / priority_training
    timestepDistribution, timestepMin, timestepMax, timestepMean,
    timestepStd, timestepAlpha, timestepBeta,
    priorityEnabled, priorityText, priorityMultiplier,
  ]);

  /**
   * Apply an incoming params dict (from get_training_run_params API) to all state.
   * Used by loadTrainingRunParams() for Edit Config restoration.
   *
   * Strategy:
   *   1. Merge known top-level TrainingRunCreateRequest fields into `params`
   *      via a single setParams call (one render).
   *   2. Sync local text states for numeric inputs (scientific notation).
   *   3. Restore UI-only states (runName, baseModelPath, trainingMethod,
   *      datasetConfigs, useEpochs, timestep_sampling breakdown,
   *      priority_training breakdown).
   */
  const applyParamsToState = useCallback((incoming: any) => {
    if (incoming.sample_steps !== undefined || incoming.sample_cfg_scale !== undefined) {
      sampleDefaultsExplicitlySetRef.current = true;
    }
    // --- UI-only / non-params states ---
    if (incoming.run_name) setRunName(incoming.run_name);
    if (incoming.base_model_path !== undefined) {
      const bmp = incoming.base_model_path || "";
      setBaseModelPath(bmp);
      // Reflect a from-scratch MiniT2I sentinel back into the checkbox/selectors.
      if (bmp.startsWith("scratch:minit2i:")) {
        const parts = bmp.slice("scratch:minit2i:".length).split(":");
        if (parts[0]) setScratchVariant(parts[0]);
        setScratchVaeType(parts[1] || "none");
        setFromScratchMiniT2I(true);
      } else {
        setFromScratchMiniT2I(false);
      }
    }
    // "vae_decoder" runs are edited by VaeTrainingConfig, not by this form, so
    // never adopt that method here (it would be submitted as a diffusion run).
    if (incoming.training_method && incoming.training_method !== "vae_decoder") {
      setTrainingMethod(incoming.training_method);
    }
    if (incoming.dataset_configs) setDatasetConfigs(incoming.dataset_configs);

    // Exclusive steps/epochs radio state
    if (incoming.total_steps !== undefined && incoming.total_steps !== null) setUseEpochs(false);
    if (incoming.epochs !== undefined && incoming.epochs !== null) setUseEpochs(true);

    // --- Fields that require local text sync (numeric-input helpers) ---
    // null means the box was empty and must be CLEARED: submit reads the text
    // state, so leaving the previous value there would silently train with it.
    if (incoming.learning_rate !== undefined && incoming.learning_rate !== null) {
      setLocalLrText(incoming.learning_rate.toString());
    }
    if (incoming.optimizer_beta1 !== undefined) {
      setLocalBeta1Text(incoming.optimizer_beta1 != null ? String(incoming.optimizer_beta1) : "");
    }
    if (incoming.optimizer_beta2 !== undefined) {
      setLocalBeta2Text(incoming.optimizer_beta2 != null ? String(incoming.optimizer_beta2) : "");
    }
    if (incoming.optimizer_epsilon !== undefined) {
      setLocalEpsilonText(incoming.optimizer_epsilon != null ? String(incoming.optimizer_epsilon) : "");
    }
    if (incoming.optimizer_weight_decay !== undefined) {
      setLocalWeightDecayText(incoming.optimizer_weight_decay != null ? String(incoming.optimizer_weight_decay) : "");
    }
    if (incoming.optimizer_schedule_free_r !== undefined) {
      setLocalScheduleFreeRText(incoming.optimizer_schedule_free_r.toString());
    }
    if (incoming.optimizer_schedule_free_weight_lr_power !== undefined) {
      setLocalScheduleFreeWeightLrPowerText(incoming.optimizer_schedule_free_weight_lr_power.toString());
    }
    if (incoming.unet_lr !== undefined) {
      setLocalUnetLrText(incoming.unet_lr != null ? String(incoming.unet_lr) : "");
    }
    if (incoming.text_encoder_lr !== undefined) {
      setLocalTextEncoderLrText(incoming.text_encoder_lr != null ? String(incoming.text_encoder_lr) : "");
    }
    if (incoming.text_encoder_1_lr !== undefined) {
      setLocalTextEncoder1LrText(incoming.text_encoder_1_lr != null ? String(incoming.text_encoder_1_lr) : "");
    }
    if (incoming.text_encoder_2_lr !== undefined) {
      setLocalTextEncoder2LrText(incoming.text_encoder_2_lr != null ? String(incoming.text_encoder_2_lr) : "");
    }
    if (incoming.image_encoder_lr !== undefined) {
      setLocalImageEncoderLrText(incoming.image_encoder_lr != null ? String(incoming.image_encoder_lr) : "");
    }
    if (incoming.vision_encoder_lr !== undefined) {
      setLocalVisionEncoderLrText(incoming.vision_encoder_lr != null ? String(incoming.vision_encoder_lr) : "");
    }

    // --- timestep_sampling (nested object expands to several UI states) ---
    if (incoming.timestep_sampling) {
      const ts = incoming.timestep_sampling;
      if (ts.distribution !== undefined) setTimestepDistribution(ts.distribution);
      if (ts.min_timestep !== undefined) setTimestepMin(ts.min_timestep);
      if (ts.max_timestep !== undefined) setTimestepMax(ts.max_timestep);
      if (ts.mean !== undefined) setTimestepMean(ts.mean);
      if (ts.std !== undefined) setTimestepStd(ts.std);
      if (ts.alpha !== undefined) setTimestepAlpha(ts.alpha);
      if (ts.beta !== undefined) setTimestepBeta(ts.beta);
    }

    // --- priority_training (object expands to 3 UI states) ---
    if (incoming.priority_training) {
      setPriorityEnabled(true);
      const entries = incoming.priority_training.entries || [];
      setPriorityText(entries.map((e: any) => typeof e === "string" ? e : JSON.stringify(e)).join("\n"));
      setPriorityMultiplier(incoming.priority_training.multiplier || 1);
    }

    // --- Single batched params update: merge every known top-level field ---
    // Fields that need special defaulting/coercion are handled explicitly;
    // all others are forwarded when present (undefined means "don't touch").
    const patch: Partial<TrainingRunCreateRequest> = {};
    for (const key of PARAM_KEYS) {
      if (incoming[key] !== undefined) {
        (patch as any)[key] = incoming[key];
      }
    }
    // Fields with custom coercion rules
    if (incoming.regularization_type !== undefined) {
      patch.regularization_type = incoming.regularization_type || "none";
    }
    if (incoming.vision_encoder_path !== undefined) {
      patch.vision_encoder_path = incoming.vision_encoder_path || "";
    }
    if (incoming.controlnet_pretrained_path !== undefined && incoming.controlnet_pretrained_path !== null) {
      patch.controlnet_pretrained_path = incoming.controlnet_pretrained_path;
    }
    if (incoming.condition_preprocessors !== undefined && incoming.condition_preprocessors !== null) {
      patch.condition_preprocessors = incoming.condition_preprocessors;
    }
    if (incoming.base_resolutions !== undefined) {
      patch.base_resolutions = incoming.base_resolutions === null ? [1024] : incoming.base_resolutions;
    }
    // Migrate configs written before latent_encoding_mode became authoritative.
    if (incoming.latent_encoding_mode === undefined && incoming.cache_latents_to_disk === true) {
      patch.latent_encoding_mode = "pre_encoded_cache";
    }
    // sample_prompts: only overwrite when non-empty (preserve default)
    if (incoming.sample_prompts && incoming.sample_prompts.length > 0) {
      patch.sample_prompts = incoming.sample_prompts;
    } else {
      delete patch.sample_prompts;
    }

    setParams(prev => ({ ...prev, ...patch }));
  }, []);

  // Load training run parameters for edit mode
  // Copy settings from an existing run into THIS new run (does not set editRunId,
  // so submit still creates a new run). Reuses the edit-mode restore path; the run
  // name is cleared so the user names the new run (avoids a duplicate-name error).
  const handleCopyFromRun = useCallback(async (runId: number) => {
    if (!runId) return;
    setCopyingFromRun(true);
    try {
      const params = await getTrainingRunParams(runId);
      dtypeExplicitlySetRef.current = true;
      restoringFromYAMLRef.current = true;
      applyParamsToState(params);
      setRunName("");  // new run needs its own name
      setTimeout(() => { restoringFromYAMLRef.current = false; }, 0);
    } catch (err: any) {
      console.error("[TrainingConfig] Failed to copy settings from run:", err);
      setError(err?.response?.data?.detail || "Failed to copy settings from run");
    } finally {
      setCopyingFromRun(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [applyParamsToState]);

  const loadTrainingRunParams = useCallback(async (runId: number) => {
    const startTime = performance.now();
    console.log(`[TrainingConfig] Loading parameters for training run ${runId}...`);

    // Mark dtype as explicitly set before loading params
    // This prevents baseModelPath useEffect from overriding YAML values
    dtypeExplicitlySetRef.current = true;
    // Mark as restoring from YAML to prevent optimizer useEffect from overwriting
    // restored optimizer hyperparameters with optimizer defaults
    restoringFromYAMLRef.current = true;

    try {
      const apiStartTime = performance.now();
      const params = await getTrainingRunParams(runId);
      console.log(`[TrainingConfig] API call took ${performance.now() - apiStartTime}ms`);
      console.log(`[TrainingConfig] Received parameters:`, params);

      // Populate all form fields from loaded parameters
      setRunName(params.run_name || "");
      // Apply all params via centralized helper (single source of truth for restoration)
      applyParamsToState(params);

      console.log(`[TrainingConfig] Successfully loaded all parameters for training run ${runId}`);
      console.log(`[TrainingConfig] Sample prompts restored:`, params.sample_prompts);
      console.log(`[TrainingConfig] MNT mode restored:`, params.multi_noise_mode);
      console.log(`[TrainingConfig] sample_every restored:`, params.sample_every);
      console.log(`[TrainingConfig] Total loadTrainingRunParams time: ${performance.now() - startTime}ms`);

      // Reset restoringFromYAMLRef after effects have fired
      // (setTimeout defers to after React's useEffect flush)
      setTimeout(() => { restoringFromYAMLRef.current = false; }, 0);
    } catch (err: any) {
      console.error("[TrainingConfig] Failed to load training run parameters:", err);
      console.error("[TrainingConfig] Error details:", err.response?.data);
      console.error("[TrainingConfig] Error message:", err.message);
      setError(`Failed to load training run parameters: ${err.response?.data?.detail || err.message}`);
      restoringFromYAMLRef.current = false;
    } finally {
      // dtypeExplicitlySetRef stays true - we don't reset it
      // This ensures dtype settings are never overwritten by baseModelPath changes
    }
  }, [applyParamsToState]);

  useEffect(() => {
    // If in edit mode, load YAML parameters first (fast)
    // Guard: only restore once per editRunId (prevents StrictMode double-invoke
    // from triggering a second async fetch that would overwrite user edits).
    if (editRunId && loadedEditRunIdRef.current !== editRunId) {
      loadedEditRunIdRef.current = editRunId;
      loadTrainingRunParams(editRunId);
    }

    // Then load datasets/models/etc (slow)
    loadDatasets();
    loadModels();
    loadSamplers();
    loadScheduleTypes();
    loadPresets();
    loadControlNets();
    // New-run mode: load the run list for the "copy settings from" selector.
    if (!editRunId) {
      listTrainingRuns()
        // VAE decoder runs have a different config shape (process.vae) that this
        // form cannot represent, so they are not offered as copy sources.
        .then((res) => setCopySourceRuns((res.runs || []).filter((r) => r.training_method !== "vae_decoder")))
        .catch((err) => console.error("[TrainingConfig] Failed to list runs for copy:", err));
    }
  }, [editRunId, loadTrainingRunParams]);

  // Auto-configure precision settings when model changes (only if not explicitly set)
  useEffect(() => {
    if (!baseModelPath) return;

    // Skip if dtype was explicitly set (from YAML load or user change)
    if (dtypeExplicitlySetRef.current) return;

    const arch = getModelArchitecture(baseModelPath);

    // SenseNova: the flag selects the second 294-Linear MoT half, so a `true`
    // carried over from another architecture turns the run into the both-half
    // configuration (32.66 GiB VRAM, up to 61.67 GiB host) without anyone
    // choosing it. Starting value only — checking the box is offered, and the
    // advisory next to it says what it costs. Same one line as Z-Image below,
    // and the same value as TRAINING_DEFAULTS["train_text_encoder"].
    if (arch === "sensenova") {
      updateParam("train_text_encoder", false);
    }

    // Dtype presets based on architecture:
    // - SD1.5/SDXL/DEUS: VAE=fp16, weight=fp32, training=fp16, save=fp16
    // - Z-Image/FLUX.2: VAE=fp32, weight=bf16, training=bf16, save=bf16
    if (arch === "zimage" || arch === "flux2") {
      // Z-Image/FLUX.2: bf16 for weights/training/output, fp32 for VAE
      updateParam("weight_dtype", "bf16");
      updateParam("training_dtype", "bf16");
      updateParam("output_dtype", "bf16");
      updateParam("vae_dtype", "fp32");
      // Z-Image LoRA injects no TE layers, so default it off there. FLUX.2 and
      // Z-Image Full FT both implement TE training; the capability table gates
      // the control, this only picks a starting value.
      if (arch === "zimage") {
        updateParam("train_text_encoder", false);
      }
    } else if (
      arch === "anima" || arch === "lens" || arch === "ideogram4" ||
      arch === "minit2i" || arch === "krea2" || arch === "ltx2" || arch === "acestep" ||
      arch === "minimax_h3"
    ) {
      // Other bf16-native DiT archs: same bf16 dtype preset as Z-Image/FLUX.2.
      // These models' weights are bfloat16, so bf16 training is the correct default
      // AND is REQUIRED for Full fine-tune -- fp16 Full-FT trains the fp16 base and
      // torch's GradScaler.unscale_ then rejects it (needs fp32 master params);
      // bf16 needs no GradScaler. (Only the dtype preset is set here; each arch keeps
      // its own text-encoder / vision-encoder trainability defaults untouched.)
      updateParam("weight_dtype", "bf16");
      updateParam("training_dtype", "bf16");
      updateParam("output_dtype", "bf16");
      updateParam("vae_dtype", "fp32");
    } else {
      // SD1.5/SDXL/DEUS: fp32 for weights, fp16 for training/output/VAE
      updateParam("weight_dtype", "fp32");
      updateParam("training_dtype", "fp16");
      updateParam("output_dtype", "fp16");
      updateParam("vae_dtype", "fp16");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseModelPath]);

  // Apply the per-architecture default timestep_sampling when the base model changes
  // (e.g. MiniT2I -> logit_normal(-0.8,0.8), most others -> uniform). The per-arch map
  // is fetched once from the backend (param_defaults SSOT). Applied exactly once per
  // model so user edits afterward persist; skipped during YAML/edit restore so a loaded
  // run's own timestep config is preserved (the restored model is recorded so a later
  // defaults-map load does not overwrite it).
  useEffect(() => {
    if (!baseModelPath) return;
    if (restoringFromYAMLRef.current) { lastTimestepModelRef.current = baseModelPath; return; }
    if (!timestepDefaultsByArch) return;
    if (lastTimestepModelRef.current === baseModelPath) return;  // already applied for this model
    const arch = getModelArchitecture(baseModelPath);
    const ts = (arch && timestepDefaultsByArch[arch]) || timestepDefaultsByArch["_default"];
    lastTimestepModelRef.current = baseModelPath;
    if (!ts) return;
    setTimestepDistribution((ts.distribution as string) ?? "uniform");
    setTimestepMin((ts.min_timestep as number) ?? 0.0);
    setTimestepMax((ts.max_timestep as number) ?? 1.0);
    setTimestepMean((ts.mean as number) ?? 0.0);
    setTimestepStd((ts.std as number) ?? 1.0);
    setTimestepAlpha((ts.alpha as number) ?? 2.0);
    setTimestepBeta((ts.beta as number) ?? 2.0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseModelPath, timestepDefaultsByArch]);

  // Apply the per-architecture default bundle_vae when the base model changes
  // (sd15/sdxl/deus -> true: their comfy-layout checkpoints are consumed by
  // A1111/ComfyUI which require the first_stage_model.* VAE section; others ->
  // false). Fetched from the backend (param_defaults SSOT); applied once per model
  // so user edits persist; skipped during YAML/edit restore.
  useEffect(() => {
    if (!baseModelPath) return;
    if (restoringFromYAMLRef.current) { lastBundleVaeModelRef.current = baseModelPath; return; }
    if (!bundleVaeDefaultsByArch) return;
    if (lastBundleVaeModelRef.current === baseModelPath) return;  // already applied for this model
    const arch = getModelArchitecture(baseModelPath);
    const def = (arch && bundleVaeDefaultsByArch[arch] !== undefined)
      ? bundleVaeDefaultsByArch[arch]
      : bundleVaeDefaultsByArch["_default"];
    lastBundleVaeModelRef.current = baseModelPath;
    if (def === undefined) return;
    updateParam("bundle_vae", !!def);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseModelPath, bundleVaeDefaultsByArch]);

  // Apply architecture-specific training-preview defaults once for a newly
  // selected model. Values restored/imported or edited by the user remain
  // explicit and are never replaced by a later architecture change.
  useEffect(() => {
    if (!baseModelPath) return;
    if (restoringFromYAMLRef.current) {
      lastSampleDefaultsModelRef.current = baseModelPath;
      return;
    }
    if (!trainingSampleDefaultsByArch) return;
    if (lastSampleDefaultsModelRef.current === baseModelPath) return;
    lastSampleDefaultsModelRef.current = baseModelPath;
    if (sampleDefaultsExplicitlySetRef.current) return;
    const arch = getModelArchitecture(baseModelPath);
    const overlay = (arch && trainingSampleDefaultsByArch[arch])
      || trainingSampleDefaultsByArch["_default"];
    if (!overlay) return;
    setParams(prev => ({
      ...prev,
      sample_steps: overlay.sample_steps as number,
      sample_cfg_scale: overlay.sample_cfg_scale as number,
    }));
  }, [baseModelPath, trainingSampleDefaultsByArch]);

  // Fall back to LoRA when the backend's TRAINING_UNSUPPORTED table says the
  // selected method is not offered for the selected base model (the run would
  // otherwise be rejected at submit time). `archCapabilities` is in the deps
  // because it arrives asynchronously — a model chosen before it loads must
  // still be re-checked once it does.
  useEffect(() => {
    if (unsupportedTrainingMethod(trainingMethod)) {
      setTrainingMethod("lora");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseModelPath, trainingMethod, archCapabilities, availableModels]);

  // Same fallback for the algebra: a "loha" carried over from another base
  // model, or left behind by switching to ReLoRA or turning block swap on,
  // would be refused before the model loads.
  //
  // Both guards are load-bearing, not defensive. `adapterAlgorithmChoices`
  // narrows to ["lora"] while `availableModels` is still empty, because
  // `trainableAdapterAlgorithms` fails closed on an unresolved architecture --
  // and edit mode restores the run's YAML BEFORE that list arrives (see the
  // load order below), so without them this effect silently rewrote a restored
  // LoKr run to ordinary LoRA and the next resume died on a missing
  // `lora_down.weight`.
  const capabilitiesReady = !!archCapabilities && !!baseModelArch;
  useEffect(() => {
    if (restoringFromYAMLRef.current || !capabilitiesReady) return;
    if (!adapterAlgorithmChoices.includes(adapterAlgorithm)) {
      updateParam("adapter_algorithm", "lora");
    }
    if (weightDecompose && !weightDecomposeAvailable) {
      updateParam("weight_decompose", false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adapterAlgorithmChoices, adapterAlgorithm, capabilitiesReady,
      weightDecompose, weightDecomposeAvailable]);

  // Clear the arming values of a feature this base model has no mechanism for,
  // so a value carried over from a previous model is not submitted and refused.
  // Skipped while restoring a run from YAML (that run's own config wins).
  useEffect(() => {
    if (restoringFromYAMLRef.current) return;
    if (blockSwapUnsupported) {
      if (params.blocks_to_swap) updateParam("blocks_to_swap", 0);
      if (params.use_pinned_memory) updateParam("use_pinned_memory", false);
      if (params.block_swap_h2d_only) updateParam("block_swap_h2d_only", false);
    }
    if (fusedGroupsUnsupported && params.num_optimizer_groups) {
      updateParam("num_optimizer_groups", 0);
    }
    if (referenceImagesUnsupported && params.use_reference_images) {
      updateParam("use_reference_images", false);
    }
    if (textEncoderTrainingUnsupported && params.train_text_encoder) {
      updateParam("train_text_encoder", false);
    }
    if (trainingSamplesUnsupported && params.sample_every) {
      updateParam("sample_every", 0);
    }
    if (motEvictionUnsupported) {
      if (params.sensenova_mot_phase_eviction) updateParam("sensenova_mot_phase_eviction", false);
      if (params.sensenova_four_phase_eviction) updateParam("sensenova_four_phase_eviction", false);
      if (params.sensenova_mot_pageable_staging) updateParam("sensenova_mot_pageable_staging", false);
      if (params.sensenova_mot_overlap_transfer) updateParam("sensenova_mot_overlap_transfer", false);
    }
    // Refused before the model loads without MoT Phase Eviction, same reason
    // as the shared window below.
    if (params.sensenova_mot_pageable_staging && !params.sensenova_mot_phase_eviction) {
      updateParam("sensenova_mot_pageable_staging", false);
    }
    if (params.sensenova_mot_overlap_transfer && !params.sensenova_mot_phase_eviction) {
      updateParam("sensenova_mot_overlap_transfer", false);
    }
    // Also refused as a PAIR: an async copy against pageable host memory is
    // bounce-buffered and effectively host-synchronous. Pageable wins here
    // because it is the one that answers a hard host-RAM limit.
    if (params.sensenova_mot_overlap_transfer && params.sensenova_mot_pageable_staging) {
      updateParam("sensenova_mot_overlap_transfer", false);
    }
    // Not a preference: the split is refused before the model loads once any of
    // its three preconditions stops holding, so leaving it set submits a run the
    // backend rejects.
    if (fourPhaseBlockedReason && params.sensenova_four_phase_eviction) {
      updateParam("sensenova_four_phase_eviction", false);
    }
    // The shared window is refused without the split it shares, same reason.
    if (params.sensenova_four_phase_shared_prefix
        && (motEvictionUnsupported || fourPhaseBlockedReason
            || !params.sensenova_four_phase_eviction)) {
      updateParam("sensenova_four_phase_shared_prefix", false);
    }
    // The two eviction params and the two branch flags are dependencies as
    // well as reads: a preset or a
    // copy-from-run writes them without touching arch or method, and
    // `fourPhaseBlockedReason` does not move when only the split itself is
    // written — so an identity-keyed list would park it true inside a control
    // that is not even rendered. Same trap the value-keyed effect below names.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseModelPath, trainingMethod, archCapabilities, availableModels,
      blockSwapUnsupported, fusedGroupsUnsupported, referenceImagesUnsupported,
      textEncoderTrainingUnsupported, trainingSamplesUnsupported,
      motEvictionUnsupported, fourPhaseBlockedReason,
      params.sensenova_four_phase_eviction, params.sensenova_mot_phase_eviction,
      params.sensenova_four_phase_shared_prefix, params.sensenova_mot_pageable_staging,
      params.sensenova_mot_overlap_transfer,
      params.train_unet, params.train_text_encoder]);

  // SigLIP2 selection is SD/SDXL's reference-conditioning opt-in. Clear it
  // when moving to an architecture whose reference path is unrelated.
  useEffect(() => {
    if (restoringFromYAMLRef.current || !baseModelPath) return;
    if (isSDOrSDXLModel(baseModelPath)) {
      if (!!visionEncoderPath !== useReferenceImages) {
        updateParam("use_reference_images", !!visionEncoderPath);
      }
      return;
    }
    if (visionEncoderPath) {
      updateParam("vision_encoder_path", "");
      updateParam("train_vision_encoder", false);
      updateParam("gradient_routing_ve", false);
      updateParam("use_reference_images", false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseModelPath, visionEncoderPath, useReferenceImages]);

  // Keep the pinned parameters at their required values, and record what was
  // replaced. Recorded rather than applied silently: the backend refuses (or
  // overwrites) either way, so the choice is between an explained adjustment
  // here and a rejection after submit — not between adjusting and honouring the
  // user's value.
  //
  // Converges on VALUE drift, not on arch/method identity. A preset, a
  // copy-from-run, or the startup `trainingDefaults` replacement writes these
  // params without touching arch or method, and an identity-keyed effect would
  // leave the violating value parked inside a control this form has disabled —
  // unreachable except by toggling the method radio. No loop: `contractAdjusted`
  // is not a dependency, and once the values match the body changes nothing.
  //
  // Skipped while restoring a run from YAML: that run's own config wins, and a
  // config that violates the contract must be seen as-is.
  useEffect(() => {
    if (restoringFromYAMLRef.current) return;
    const startsNewContract = pinnedForRef.current !== requiredValues;
    pinnedForRef.current = requiredValues;
    const changed = Object.entries(requiredValues)
      // `entry.values` (the full admitted set) means the current value is only
      // drift if it is outside that set; `entry.value` is then just the member
      // to fall back to.
      .filter(([param, entry]) => (entry.values
        ? !entry.values.includes((params as any)[param])
        : (params as any)[param] !== entry.value));
    for (const [param, entry] of changed) {
      updateParam(param as keyof TrainingRunCreateRequest, entry.value as any);
    }
    setContractAdjusted((prev) => {
      const next = startsNewContract ? {} : { ...prev };
      for (const [param] of changed) {
        next[param] = String((params as any)[param]);
      }
      const keys = Object.keys(next);
      const unchanged = keys.length === Object.keys(prev).length
        && keys.every((key) => prev[key] === next[key]);
      return unchanged ? prev : next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requiredValues, params]);


  // Reset optimizer hyperparameters when optimizer changes
  useEffect(() => {
    // Skip during YAML restoration — params are already being restored correctly
    if (restoringFromYAMLRef.current) return;

    const config = OPTIMIZER_CONFIGS[optimizer];
    if (!config) return;

    if (!skipOptimizerHyperparamResetRef.current) {
      const { beta1, beta2, epsilon, weight_decay } = config.defaults;
      if (beta1 !== undefined) {
        setLocalBeta1Text(beta1);
        updateParam("optimizer_beta1", parseFloat(beta1));
      }
      if (beta2 !== undefined) {
        setLocalBeta2Text(beta2);
        updateParam("optimizer_beta2", parseFloat(beta2));
      }
      if (epsilon !== undefined) {
        setLocalEpsilonText(epsilon);
        updateParam("optimizer_epsilon", parseFloat(epsilon));
      }
      if (weight_decay !== undefined) {
        setLocalWeightDecayText(weight_decay);
        updateParam("optimizer_weight_decay", parseFloat(weight_decay));
      }
    }

    // Reset options that are not supported by the new optimizer
    if (!config.supportsCautious) updateParam("optimizer_cautious", false);
    // Host-resident state is a ring-buffer-only allocator choice; every other
    // optimizer accepts and ignores it, so leaving it set would show a ticked
    // box doing nothing.
    if (!optimizer.endsWith("_ringbuffer")) {
      updateParam("optimizer_state_host_resident", false);
    }
  }, [params.optimizer, updateParam]);

  const loadDatasets = async () => {
    const startTime = performance.now();
    console.log("[TrainingConfig] loadDatasets starting...");
    try {
      const response = await listDatasets();
      console.log(`[TrainingConfig] loadDatasets API took ${performance.now() - startTime}ms`);
      setDatasets(response.datasets);

      // Only auto-select first dataset if NOT in edit mode
      // (edit mode will have already loaded datasetConfigs from YAML)
      if (!editRunId && response.datasets.length > 0) {
        const firstDatasetId = response.datasets[0].id;
        console.log(`[TrainingConfig] New run mode: auto-selecting first dataset ${firstDatasetId}`);
        setDatasetConfigs([{ dataset_id: firstDatasetId, caption_types: [], filters: {} }]);
      } else if (editRunId) {
        console.log(`[TrainingConfig] Edit mode: keeping existing datasetConfigs from YAML`);
      }
    } catch (err) {
      console.error("Failed to load datasets:", err);
    }
  };

  // Keep the base-model sentinel in sync with the from-scratch selectors. The
  // trainer parses "scratch:minit2i:<variant>:<vae_type>" and builds a random model
  // in memory (no init model on disk). From-scratch is Full Fine-tune only.
  useEffect(() => {
    if (fromScratchMiniT2I) {
      setBaseModelPath(`scratch:minit2i:${scratchVariant}:${scratchVaeType}`);
      setTrainingMethod("full_finetune");
    } else if (baseModelPath.startsWith("scratch:minit2i:")) {
      setBaseModelPath("");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fromScratchMiniT2I, scratchVariant, scratchVaeType]);

  const loadModels = async () => {
    const startTime = performance.now();
    console.log("[TrainingConfig] loadModels starting...");
    try {
      const response = await getModels();
      console.log(`[TrainingConfig] loadModels API took ${performance.now() - startTime}ms`);
      const models = response.models || [];
      setAvailableModels(models);

      // Only auto-select first model if NOT in edit mode
      // (edit mode will have already loaded baseModelPath from YAML)
      if (!editRunId && models.length > 0) {
        console.log(`[TrainingConfig] New run mode: auto-selecting first model ${models[0].path}`);
        setBaseModelPath(models[0].path);
      } else if (editRunId) {
        console.log(`[TrainingConfig] Edit mode: keeping existing baseModelPath from YAML`);
      }
    } catch (err) {
      console.error("Failed to load models:", err);
    }
  };

  // Helper function: Load samplers from API
  const loadSamplers = async () => {
    try {
      const data = await getSamplers();
      setSamplers(data.samplers);
    } catch (error) {
      console.error("Failed to load samplers:", error);
    }
  };

  // Helper function: Load schedule types from API
  const loadScheduleTypes = async () => {
    try {
      const data = await getScheduleTypes();
      setScheduleTypes(data.schedule_types);
    } catch (error) {
      console.error("Failed to load schedule types:", error);
    }
  };

  // Helper function: Load presets from API
  const loadPresets = async () => {
    try {
      const response = await listTrainingPresets();
      setPresets(response.presets);
    } catch (error) {
      console.error("Failed to load presets:", error);
    }
  };

  // Helper function: Load available ControlNet models from API
  const loadControlNets = async () => {
    try {
      const response = await getControlNets();
      setAvailableControlNets(response.controlnets || []);
    } catch (error) {
      console.error("Failed to load ControlNet models:", error);
    }
  };

  // Helper function: Get random caption from selected datasets
  const handleRandomPrompt = async (promptIndex: number) => {
    const selectedDatasets = datasetConfigs.filter(c => c.dataset_id !== 0);
    if (selectedDatasets.length === 0) {
      setError("Please select at least one dataset first");
      return;
    }

    try {
      // Pick a random dataset from selected ones
      const randomDataset = selectedDatasets[Math.floor(Math.random() * selectedDatasets.length)];
      const response = await getRandomCaption(randomDataset.dataset_id, randomDataset.caption_types);

      // Set the positive prompt
      const updated = [...samplePrompts];
      updated[promptIndex] = { ...updated[promptIndex], positive: response.caption };

      // Auto-populate reference/condition image if the item has reference images
      if (response.reference_images && response.reference_images.length > 0) {
        const refPath = response.reference_images[0];
        const showRefUI = trainingMethod !== "controlnet" && referenceConditioningEnabled;
        if (trainingMethod === "controlnet") {
          updated[promptIndex].condition_image_path = refPath;
          setConditionImagePreviews(prev => ({
            ...prev,
            [promptIndex]: `/api/serve-image?path=${encodeURIComponent(refPath)}`
          }));
        } else if (showRefUI) {
          updated[promptIndex].reference_image_path = refPath;
          setReferenceImagePreviews(prev => ({
            ...prev,
            [promptIndex]: `/api/serve-image?path=${encodeURIComponent(refPath)}`
          }));
        }
      }

      setSamplePrompts(updated);
    } catch (err) {
      console.error("Failed to get random caption:", err);
      setError("Failed to get random caption from dataset");
    }
  };

  // Condition image handlers for per-prompt ControlNet sample generation
  const handleConditionImageUpload = async (promptIndex: number, file: File) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      const base64 = e.target?.result as string;
      if (!base64) return;
      try {
        const reference = await saveTempImage(base64);
        const updated = [...samplePrompts];
        updated[promptIndex] = { ...updated[promptIndex], condition_image_path: reference };
        setSamplePrompts(updated);
        setConditionImagePreviews(prev => ({ ...prev, [promptIndex]: base64 }));
      } catch (err) {
        console.error("Failed to save condition image:", err);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleConditionImageRemove = async (promptIndex: number) => {
    const currentPath = samplePrompts[promptIndex]?.condition_image_path;
    if (currentPath) {
      await deleteTempImageRef(currentPath);
    }
    const updated = [...samplePrompts];
    updated[promptIndex] = { ...updated[promptIndex], condition_image_path: "" };
    setSamplePrompts(updated);
    setConditionImagePreviews(prev => {
      const next = { ...prev };
      delete next[promptIndex];
      return next;
    });
  };

  const handleConditionImageDrop = (promptIndex: number, e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const files = e.dataTransfer.files;
    if (files && files.length > 0 && files[0].type.startsWith("image/")) {
      handleConditionImageUpload(promptIndex, files[0]);
    }
  };

  // Reference image handlers for VE/FLUX.2 sample generation
  const handleReferenceImageUpload = async (promptIndex: number, file: File) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      const base64 = e.target?.result as string;
      try {
        const reference = await saveTempImage(base64);
        const updated = [...samplePrompts];
        updated[promptIndex] = { ...updated[promptIndex], reference_image_path: reference };
        setSamplePrompts(updated);
        setReferenceImagePreviews(prev => ({ ...prev, [promptIndex]: base64 }));
      } catch (err) {
        console.error("Failed to save reference image:", err);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleReferenceImageRemove = async (promptIndex: number) => {
    const currentPath = samplePrompts[promptIndex]?.reference_image_path;
    if (currentPath) {
      await deleteTempImageRef(currentPath);
    }
    const updated = [...samplePrompts];
    updated[promptIndex] = { ...updated[promptIndex], reference_image_path: "" };
    setSamplePrompts(updated);
    setReferenceImagePreviews(prev => {
      const next = { ...prev };
      delete next[promptIndex];
      return next;
    });
  };

  // Apply reference image dimensions (floor to multiple of 8) to sample width/height
  const applyRefImageSize = () => {
    // Find the first prompt that has a preview loaded
    const firstIndex = samplePrompts.findIndex((_, i) => referenceImagePreviews[i]);
    if (firstIndex === -1) return;
    const url = referenceImagePreviews[firstIndex];
    const img = new Image();
    img.onload = () => {
      updateParam("sample_width", Math.floor(img.naturalWidth / 8) * 8);
      updateParam("sample_height", Math.floor(img.naturalHeight / 8) * 8);
    };
    img.src = url;
  };

  const handleReferenceImageDrop = (promptIndex: number, e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const files = e.dataTransfer.files;
    if (files && files.length > 0 && files[0].type.startsWith("image/")) {
      handleReferenceImageUpload(promptIndex, files[0]);
    }
  };

  // Load condition/reference image previews when samplePrompts paths are set
  useEffect(() => {
    samplePrompts.forEach(async (prompt, index) => {
      if (prompt.condition_image_path && !conditionImagePreviews[index]) {
        try {
          if (prompt.condition_image_path.startsWith("temp_img://")) {
            const dataUrl = await loadTempImage(prompt.condition_image_path);
            if (dataUrl) {
              setConditionImagePreviews(prev => ({ ...prev, [index]: dataUrl }));
            }
          } else {
            setConditionImagePreviews(prev => ({
              ...prev,
              [index]: `/api/serve-image?path=${encodeURIComponent(prompt.condition_image_path!)}`
            }));
          }
        } catch {
          // Ignore load errors for previews
        }
      }
      if (prompt.reference_image_path && !referenceImagePreviews[index]) {
        try {
          if (prompt.reference_image_path.startsWith("temp_img://")) {
            const dataUrl = await loadTempImage(prompt.reference_image_path);
            if (dataUrl) {
              setReferenceImagePreviews(prev => ({ ...prev, [index]: dataUrl }));
            }
          } else {
            setReferenceImagePreviews(prev => ({
              ...prev,
              [index]: `/api/serve-image?path=${encodeURIComponent(prompt.reference_image_path!)}`
            }));
          }
        } catch {
          // Ignore load errors for previews
        }
      }
    });
  }, [samplePrompts]);

  // Helper function: Import params from txt2img panel
  const handleImportFromGeneration = () => {
    // Try to read from localStorage where generation panels store their params
    try {
      const txt2imgParams = localStorage.getItem("txt2img_params");
      if (txt2imgParams) {
        const params = JSON.parse(txt2imgParams);
        // Update sample generation parameters
        if (params.prompt) {
          const updated = [...samplePrompts];
          updated[0].positive = params.prompt;
          updated[0].negative = params.negative_prompt || "";
          setSamplePrompts(updated);
        }
        if (params.width) updateParam("sample_width", params.width);
        if (params.height) updateParam("sample_height", params.height);
        if (params.steps) {
          sampleDefaultsExplicitlySetRef.current = true;
          updateParam("sample_steps", params.steps);
        }
        if (params.cfg_scale) {
          sampleDefaultsExplicitlySetRef.current = true;
          updateParam("sample_cfg_scale", params.cfg_scale);
        }
        if (params.sampler) updateParam("sample_sampler", params.sampler);
        if (params.schedule_type) updateParam("sample_schedule_type", params.schedule_type);
        if (params.cfg_schedule_type !== undefined) updateParam("sample_cfg_schedule_type", params.cfg_schedule_type);
        if (params.cfg_schedule_min !== undefined) updateParam("sample_cfg_schedule_min", params.cfg_schedule_min);
        if (params.cfg_schedule_max !== undefined) updateParam("sample_cfg_schedule_max", params.cfg_schedule_max);
        if (params.cfg_schedule_power !== undefined) updateParam("sample_cfg_schedule_power", params.cfg_schedule_power);
        if (params.cfg_rescale_snr_alpha !== undefined) updateParam("sample_cfg_rescale_snr_alpha", params.cfg_rescale_snr_alpha);
        if (params.dynamic_threshold_percentile !== undefined) updateParam("sample_dynamic_threshold_percentile", params.dynamic_threshold_percentile);
        if (params.dynamic_threshold_mimic_scale !== undefined) updateParam("sample_dynamic_threshold_mimic_scale", params.dynamic_threshold_mimic_scale);
        if (params.nag_enable !== undefined) updateParam("sample_nag_enable", params.nag_enable);
        if (params.nag_scale !== undefined) updateParam("sample_nag_scale", params.nag_scale);
        if (params.nag_tau !== undefined) updateParam("sample_nag_tau", params.nag_tau);
        if (params.nag_alpha !== undefined) updateParam("sample_nag_alpha", params.nag_alpha);
        if (params.nag_sigma_end !== undefined) updateParam("sample_nag_sigma_end", params.nag_sigma_end);
        if (params.nag_negative_prompt !== undefined) updateParam("sample_nag_negative_prompt", params.nag_negative_prompt);
        if (params.seed !== undefined) updateParam("sample_seed", params.seed);
      }
    } catch (err) {
      console.error("Failed to import from generation panel:", err);
    }
  };

  // Derived rather than listed: the hand-written mapping this replaced drifted
  // every time a parameter was added, and 132 of them had silently stopped
  // round-tripping. See the module header for the rule.
  //
  // Inherits getRequestData()'s method gates, so a preset saved under one
  // training_method carries none of another method's fields (no lora_*/
  // adapter_* under full_finetune, no controlnet_*/outpaint_* under lora).
  // Inert, because a preset restores its own training_method.
  const getCurrentConfig = (): Record<string, any> => {
    const config: Record<string, any> = {};
    const excluded = new Set(PRESET_EXCLUDED_KEYS);
    for (const [key, value] of Object.entries(getRequestData())) {
      if (excluded.has(key) || value === undefined) continue;
      // An empty learning-rate box parses to NaN, which JSON stores as null.
      if (typeof value === "number" && Number.isNaN(value)) continue;
      config[key] = value;
    }
    // A missing key means "leave it alone" on restore, so an empty optional
    // box has to be saved as an explicit null or it cannot be cleared.
    for (const key of PRESET_CLEARABLE_NUMERIC_KEYS) {
      if (config[key] === undefined) config[key] = null;
    }
    // getRequestData() drops the inactive one of the steps/epochs pair; a preset
    // keeps both plus the radio state, so flipping it after load shows what was
    // saved (applyParamsToState otherwise infers the radio from presence).
    config.useEpochs = useEpochs;
    if (params.total_steps !== undefined) config.total_steps = params.total_steps;
    if (params.epochs !== undefined) config.epochs = params.epochs;
    return config;
  };

  // Save current config as preset
  const handleSavePreset = async () => {
    if (!presetName.trim()) {
      alert("Please enter a preset name");
      return;
    }

    try {
      await createTrainingPreset({
        name: presetName,
        description: presetDescription || undefined,
        training_method: trainingMethod,
        config: getCurrentConfig(),
      });
      await loadPresets();
      setShowPresetDialog(false);
      setPresetName("");
      setPresetDescription("");
      alert("Preset saved successfully");
    } catch (error: any) {
      console.error("Failed to save preset:", error);
      alert(error.response?.data?.detail || "Failed to save preset");
    }
  };

  // Load preset into form
  const handleLoadPreset = (preset: TrainingPreset) => {
    const config = preset.config || {};
    // The preset carries its own optimizer hyperparameters; without this the
    // optimizer-change effect replaces them with the new optimizer's defaults.
    skipOptimizerHyperparamResetRef.current = true;
    applyParamsToState(presetConfigToParams(config));
    // applyParamsToState() infers the steps/epochs radio from which key is
    // present; the preset carries both, so its own flag wins.
    if (config.useEpochs !== undefined) setUseEpochs(config.useEpochs);
    if (preset.training_method) setTrainingMethod(preset.training_method);
    setShowLoadPresetDialog(false);
    // Cleared after this render's effects, same pattern as handleCopyFromRun:
    // the effect may not fire at all (identical optimizer), so it cannot own the
    // reset without poisoning the next genuine optimizer change.
    setTimeout(() => { skipOptimizerHyperparamResetRef.current = false; }, 0);
  };

  // Delete preset
  const handleDeletePreset = async (presetId: number) => {
    if (!confirm("Are you sure you want to delete this preset?")) return;

    try {
      await deleteTrainingPreset(presetId);
      await loadPresets();
    } catch (error) {
      console.error("Failed to delete preset:", error);
      alert("Failed to delete preset");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    console.log("[TrainingConfig] Form submitted");
    console.log("[TrainingConfig] Run name:", runName);
    console.log("[TrainingConfig] Dataset configs:", datasetConfigs);
    console.log("[TrainingConfig] Base model path:", baseModelPath);

    // Validate at least one dataset is selected
    if (datasetConfigs.length === 0 || datasetConfigs.every(c => c.dataset_id === 0)) {
      setError("Please select at least one dataset");
      return;
    }

    if (!baseModelPath.trim()) {
      setError("Base model path is required");
      return;
    }

    // Validate that at least one component is being trained (not applicable for ControlNet)
    if (trainingMethod !== "controlnet" && !trainUnet && !trainTextEncoder) {
      setError("At least one component (U-Net or Text Encoder) must be trained");
      return;
    }

    // Learning-rate bounds: submitted values are read from the local text
    // buffers below, not from `params`, so this is the seam that actually
    // decides what reaches the backend (backend: learning_rate gt=0,
    // component rates ge=0, empty component rate = inherit base LR).
    const baseLr = parseFloat(localLrText);
    if (isNaN(baseLr) || baseLr <= 0) {
      setError("Learning Rate must be a positive number greater than 0.");
      return;
    }
    const componentRateFields: Array<[string, string]> = [
      ["U-Net LR", localUnetLrText],
      ["Text Encoder LR", localTextEncoderLrText],
      ["TE1 LR", localTextEncoder1LrText],
      ["TE2 LR", localTextEncoder2LrText],
      ["Image Encoder LR", localImageEncoderLrText],
      ["Vision Encoder LR", localVisionEncoderLrText],
    ];
    for (const [label, text] of componentRateFields) {
      if (text === "") continue; // empty means "use base LR", legal
      const v = parseFloat(text);
      if (isNaN(v) || v < 0) {
        setError(`${label} must be zero or a positive number (leave empty to use the base learning rate).`);
        return;
      }
    }

    setLoading(true);
    setError(null);

    // Build requestData via centralized helper (single source of truth)
    const requestData = getRequestData();

    console.log("[TrainingConfig] Request data:", requestData);
    console.log("[TrainingConfig] Learning rates:", {
      base_lr: requestData.learning_rate,
      unet_lr: requestData.unet_lr,
      text_encoder_lr: requestData.text_encoder_lr,
      text_encoder_1_lr: requestData.text_encoder_1_lr,
      text_encoder_2_lr: requestData.text_encoder_2_lr,
    });

    try {
      if (editRunId) {
        // Update existing run
        const updatedRun = await updateTrainingRun(editRunId, requestData);
        console.log("[TrainingConfig] Training run updated:", updatedRun);
        if (onRunUpdated) {
          onRunUpdated(updatedRun);
        }
      } else {
        // Create new run
        const newRun = await createTrainingRun(requestData);
        console.log("[TrainingConfig] Training run created:", newRun);
        onRunCreated(newRun);
      }
    } catch (err: any) {
      console.error("[TrainingConfig] Error details:", err);
      console.error("[TrainingConfig] Error response:", err.response);
      console.error("[TrainingConfig] Error data:", err.response?.data);
      const errorMessage = err.response?.data?.detail || err.response?.data?.message || err.message || (editRunId ? "Failed to update training run" : "Failed to create training run");
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-3 sm:p-4 border-b border-gray-700 flex items-center justify-between bg-gray-800/50 sticky top-0 z-10">
        <h2 className="text-base sm:text-lg font-semibold truncate mr-2">{editRunId ? "Edit Training Run" : "New Training Run"}</h2>
        <div className="flex items-center gap-1.5 sm:gap-2 flex-shrink-0">
          <button
            type="button"
            onClick={() => setShowLoadPresetDialog(true)}
            className="hidden sm:flex items-center gap-2 px-2 sm:px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-xs sm:text-sm transition-colors"
          >
            <FolderOpen className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
            Load Preset
          </button>
          <button
            type="button"
            onClick={() => setShowLoadPresetDialog(true)}
            className="sm:hidden p-1.5 bg-blue-600 hover:bg-blue-500 rounded transition-colors"
            title="Load Preset"
          >
            <FolderOpen className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => setShowPresetDialog(true)}
            className="hidden sm:flex items-center gap-2 px-2 sm:px-3 py-1.5 bg-green-600 hover:bg-green-500 rounded text-xs sm:text-sm transition-colors"
          >
            <Save className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
            Save Preset
          </button>
          <button
            type="button"
            onClick={() => setShowPresetDialog(true)}
            className="sm:hidden p-1.5 bg-green-600 hover:bg-green-500 rounded transition-colors"
            title="Save Preset"
          >
            <Save className="h-4 w-4" />
          </button>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-gray-700 rounded transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="p-3 sm:p-4">
        {error && (
          <div className="bg-red-900/20 border border-red-500 text-red-400 rounded p-2.5 sm:p-3 text-xs sm:text-sm mb-3 sm:mb-4">
            {error}
          </div>
        )}

        <div className="columns-1 lg:columns-2 gap-3 sm:gap-4 space-y-3 sm:space-y-4">
        {/* Copy settings from an existing run (new-run mode only) */}
        {!editRunId && copySourceRuns.length > 0 && (
          <div className="break-inside-avoid">
            <label className="block text-xs sm:text-sm font-medium mb-1.5 sm:mb-2">
              Copy settings from existing run{" "}
              <span className="text-gray-500 text-xxs sm:text-xs font-normal">(optional — fills the form from a previous run)</span>
            </label>
            <select
              defaultValue=""
              disabled={copyingFromRun}
              onChange={(e) => { const id = Number(e.target.value); if (id) handleCopyFromRun(id); e.target.value = ""; }}
              className="w-full px-2.5 sm:px-3 py-1.5 sm:py-2 bg-gray-800 border border-gray-700 rounded text-xs sm:text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
            >
              <option value="">{copyingFromRun ? "Copying…" : "Select a run to copy from…"}</option>
              {copySourceRuns.map((r) => (
                <option key={r.id} value={r.id}>
                  #{r.id} {r.run_name} ({r.training_method || "?"}{r.status ? `, ${r.status}` : ""})
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">Run name and dataset selection should be reviewed after copying.</p>
          </div>
        )}

        {/* Run Name */}
        <div className="break-inside-avoid">
          <label className="block text-xs sm:text-sm font-medium mb-1.5 sm:mb-2">
            Run Name {editRunId ? (
              <span className="text-gray-500 text-xxs sm:text-xs font-normal">(cannot be changed after creation)</span>
            ) : (
              <span className="text-gray-500 text-xxs sm:text-xs font-normal">(optional, auto-generated if empty)</span>
            )}
          </label>
          <input
            type="text"
            value={runName}
            onChange={(e) => setRunName(e.target.value)}
            placeholder="Leave empty for auto-generated name (e.g., 20251130_174523_a1b2c3d4)"
            className={`w-full px-2.5 sm:px-3 py-1.5 sm:py-2 bg-gray-800 border border-gray-700 rounded text-xs sm:text-sm focus:outline-none focus:border-blue-500 ${editRunId ? 'opacity-50 cursor-not-allowed' : ''}`}
            disabled={!!editRunId}
          />
        </div>

        {/* GPU Selection */}
        <div className="break-inside-avoid">
          <GpuSelect
            value={params.gpu_index ?? null}
            onChange={(v) => updateParam("gpu_index", v)}
            label="GPU"
          />
        </div>

        {/* Datasets */}
        <div className="break-inside-avoid border border-gray-700 rounded p-4 space-y-3">
          <div className="flex justify-between items-center">
            <label className="block text-sm font-medium">
              Datasets <span className="text-red-400">*</span>
            </label>
            <button
              type="button"
              onClick={() => setDatasetConfigs([...datasetConfigs, { dataset_id: datasets[0]?.id || 0, caption_types: [], filters: {} }])}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-xs transition-colors"
              disabled={datasets.length === 0}
            >
              + Add Dataset
            </button>
          </div>

          {datasetConfigs.map((config, index) => (
            <div key={index} className="border border-gray-600 rounded p-3 space-y-2">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs text-gray-400">Dataset {index + 1}</span>
                {datasetConfigs.length > 1 && (
                  <button
                    type="button"
                    onClick={() => setDatasetConfigs(datasetConfigs.filter((_, i) => i !== index))}
                    className="text-red-400 hover:text-red-300 text-xs"
                  >
                    Remove
                  </button>
                )}
              </div>

              <select
                value={config.dataset_id}
                onChange={(e) => {
                  const newDatasetId = parseInt(e.target.value);
                  const updated = [...datasetConfigs];
                  updated[index].dataset_id = newDatasetId;
                  updated[index].caption_types = []; // Reset caption types when dataset changes
                  setDatasetConfigs(updated);
                }}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value={0}>Select dataset...</option>
                {datasets.map((ds) => (
                  <option key={ds.id} value={ds.id}>
                    {ds.name} ({ds.total_items} items)
                  </option>
                ))}
              </select>

              {/* Caption Types: Moved to Dataset Management > Caption Processing */}
              {/* Configure caption types in Dataset Management page for each dataset */}

              {/* VE Reconstruction Mode - only shown when VE is configured */}
              {visionEncoderPath && (
                <div className="flex items-center space-x-2 mt-1.5">
                  <input
                    type="checkbox"
                    id={`ve-recon-mode-${index}`}
                    checked={config.ve_reconstruction_mode || false}
                    onChange={(e) => {
                      const updated = [...datasetConfigs];
                      updated[index] = { ...updated[index], ve_reconstruction_mode: e.target.checked };
                      setDatasetConfigs(updated);
                    }}
                    className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  />
                  <label htmlFor={`ve-recon-mode-${index}`} className="text-xs text-gray-400 cursor-pointer">
                    VE Reconstruction Mode
                  </label>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Training Method */}
        <div className="break-inside-avoid">
          <label className="block text-sm font-medium mb-2">Training Method</label>
          <div className="flex space-x-4">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="training_method"
                value="lora"
                checked={trainingMethod === "lora"}
                onChange={() => setTrainingMethod("lora")}
                disabled={fromScratchMiniT2I}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className={`text-sm ${fromScratchMiniT2I ? 'text-gray-500' : ''}`}>LoRA (Recommended)</span>
            </label>
            {(() => {
              const fullFtReason = unsupportedTrainingMethod("full_finetune");
              const fullFtBlocked = !!fullFtReason;
              const title = fullFtReason;
              return (
            <label
              className={`flex items-center space-x-2 ${fullFtBlocked ? 'cursor-not-allowed' : 'cursor-pointer'}`}
              title={title}
            >
              <input
                type="radio"
                name="training_method"
                value="full_finetune"
                checked={trainingMethod === "full_finetune"}
                onChange={() => setTrainingMethod("full_finetune")}
                disabled={fullFtBlocked}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className={`text-sm ${fullFtBlocked ? 'text-gray-500' : ''}`}>
                Full Fine-tune{fullFtReason ? ' (not supported for this model)' : ''}
              </span>
            </label>
              );
            })()}
            {(() => {
              const controlnetReason = unsupportedTrainingMethod("controlnet");
              const controlnetBlocked = fromScratchMiniT2I || !!controlnetReason;
              return (
            <label
              className={`flex items-center space-x-2 ${controlnetBlocked ? 'cursor-not-allowed' : 'cursor-pointer'}`}
              title={controlnetReason}
            >
              <input
                type="radio"
                name="training_method"
                value="controlnet"
                checked={trainingMethod === "controlnet"}
                onChange={() => setTrainingMethod("controlnet")}
                disabled={controlnetBlocked}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className={`text-sm ${controlnetBlocked ? 'text-gray-500' : ''}`}>
                ControlNet (SD1.5/SDXL){controlnetReason ? ' (not supported for this model)' : ''}
              </span>
            </label>
              );
            })()}
            {(() => {
              const reloraReason = unsupportedTrainingMethod("relora");
              const reloraBlocked = fromScratchMiniT2I || !!reloraReason;
              const title = reloraReason;
              return (
            <label
              className={`flex items-center space-x-2 ${reloraBlocked ? 'cursor-not-allowed' : 'cursor-pointer'}`}
              title={title}
            >
              <input
                type="radio"
                name="training_method"
                value="relora"
                checked={trainingMethod === "relora"}
                onChange={() => setTrainingMethod("relora")}
                disabled={reloraBlocked}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className={`text-sm ${reloraBlocked ? 'text-gray-500' : ''}`}>
                ReLoRA (Periodic Merge + Reinit){reloraReason ? ' (not supported for this model)' : ''}
              </span>
            </label>
              );
            })()}
          </div>

          {/* The contract this architecture + method runs under, and what this
              form had to change to meet it. Shown here because the method radio
              above is what selects the contract. */}
          {Object.keys(requiredValues).length > 0 && (
            <div className="mt-2 p-2 border border-amber-700/60 bg-amber-950/30 rounded text-xs space-y-1">
              <p className="text-amber-300">
                {archDisplayName(archCapabilities, baseModelArch)} requires
                these settings for {trainingMethod === "full_finetune" ? "full fine-tuning" : trainingMethod}:
              </p>
              <ul className="list-disc list-inside text-gray-300">
                {Object.entries(requiredValues).map(([param, entry]) => (
                  <li key={param}>
                    <span className="text-gray-200">{param} = {String(entry.value)}</span>
                    {entry.unless && (
                      <span className="text-amber-300"> unless {describeRequirementLift(entry.unless)}</span>
                    )}
                    {contractAdjusted[param] !== undefined && (
                      <span className="text-amber-400"> (changed from {contractAdjusted[param]})</span>
                    )}
                    <span className="text-gray-500"> — {entry.reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Base Model */}
        <div className="break-inside-avoid">
          <label className="block text-sm font-medium mb-2">
            Base Model <span className="text-red-400">*</span>
          </label>

          {/* Model Architecture Filter (one checkbox per architecture present) */}
          <div className="flex items-center flex-wrap gap-x-4 gap-y-1 mb-2 text-xs">
            <span className="text-gray-400">Filter:</span>
            {archFilterOptions.map(({ arch, label }) => (
              <label key={arch} className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={!hiddenArchs.includes(arch)}
                  onChange={(e) =>
                    setHiddenArchs((prev) =>
                      e.target.checked
                        ? prev.filter((a) => a !== arch)
                        : [...prev, arch]
                    )
                  }
                  className="w-3.5 h-3.5"
                />
                <span className="text-gray-300">{label}</span>
              </label>
            ))}
          </div>

          <select
            value={baseModelPath}
            onChange={(e) => setBaseModelPath(e.target.value)}
            disabled={fromScratchMiniT2I}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
            required={!fromScratchMiniT2I}
          >
            <option value="">Select a model...</option>
            {filteredModels.map((model) => {
              // For MiniT2I show the latent format (vae_type) since it is fixed per
              // model and determines training/inference (pixel vs sdxl/flux1 latent).
              const label = (model.architecture === "minit2i" && model.vae_type)
                ? `${model.architecture.toUpperCase()} · ${model.vae_type === "none" ? "pixel" : model.vae_type + " latent"}`
                : model.architecture.toUpperCase();
              return (
                <option key={model.path} value={model.path}>
                  {model.name} ({label})
                </option>
              );
            })}
          </select>
          {availableModels.length === 0 && (
            <p className="text-xs text-gray-500 mt-1">No models available. Please add models to the models directory.</p>
          )}
          {filteredModels.length === 0 && availableModels.length > 0 && (
            <p className="text-xs text-gray-500 mt-1">No models match the selected filters.</p>
          )}

          {/* Train a MiniT2I from scratch (in-memory random init; no init model on disk).
              The trainer builds the model from variant + latent VAE; Full Fine-tune only. */}
          {(!baseModelPath || isMiniT2IModel(baseModelPath) || fromScratchMiniT2I) && (
          <div className="mt-2 p-3 bg-gray-800/60 border border-gray-700 rounded space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={fromScratchMiniT2I}
                onChange={(e) => setFromScratchMiniT2I(e.target.checked)}
                className="w-3.5 h-3.5"
              />
              <span className="text-sm text-gray-300">Train MiniT2I from scratch</span>
            </label>
            {fromScratchMiniT2I && (
              <>
                <div className="flex gap-2">
                  <div className="flex-1">
                    <label className="block text-xs text-gray-400 mb-1">Variant</label>
                    <select value={scratchVariant} onChange={(e) => setScratchVariant(e.target.value)}
                            className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs">
                      <option value="b16">B/16 (~0.26B)</option>
                      <option value="l16">L/16 (~1.8B)</option>
                    </select>
                  </div>
                  <div className="flex-1">
                    <label className="block text-xs text-gray-400 mb-1">Latent VAE</label>
                    <select value={scratchVaeType} onChange={(e) => setScratchVaeType(e.target.value)}
                            className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs">
                      <option value="sdxl">SDXL VAE (4ch)</option>
                      <option value="flux1">FLUX.1 VAE (16ch)</option>
                      <option value="none">None (pixel-space)</option>
                    </select>
                  </div>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Inherit weights from (optional)</label>
                  <select
                    value={params.minit2i_scratch_init_from || ""}
                    onChange={(e) => updateParam("minit2i_scratch_init_from", e.target.value)}
                    className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs"
                  >
                    <option value="">None (random init)</option>
                    {availableModels.filter((m) => m.architecture === "minit2i").map((m) => (
                      <option key={m.path} value={m.path}>{m.name}</option>
                    ))}
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    Same variant only. Transformer body + proj2 + embedders copy fully; the
                    in/out layers copy overlapping channels when the patch is unchanged
                    (latent↔latent), else they are re-initialized (pixel→latent).
                  </p>
                  {params.minit2i_scratch_init_from ? (
                    <label className="flex items-center gap-2 cursor-pointer mt-2">
                      <input
                        type="checkbox"
                        checked={!!params.minit2i_inherit_final_layer}
                        onChange={(e) => updateParam("minit2i_inherit_final_layer", e.target.checked)}
                        className="w-3.5 h-3.5"
                      />
                      <span className="text-xs text-gray-300">
                        Inherit output head (final_layer) when shape matches
                      </span>
                    </label>
                  ) : null}
                </div>
                <p className="text-xs text-gray-500">
                  Random-initialized in memory and trained with Full Fine-tune. Latent variants
                  (SDXL/FLUX.1) load the VAE by type at train/inference time. Base model:
                  <span className="font-mono text-gray-400"> scratch:minit2i:{scratchVariant}:{scratchVaeType}</span>
                </p>
              </>
            )}
          </div>
          )}

          {/* REPA (Representation Alignment) — MiniT2I only. Aligns a DiT hidden state
              with frozen clean-image features to accelerate convergence (arXiv:2410.06940). */}
          {(isMiniT2IModel(baseModelPath) || fromScratchMiniT2I) && (
            <div className="mt-2 p-3 bg-gray-800/60 border border-gray-700 rounded space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={!!params.repa_enable}
                  onChange={(e) => updateParam("repa_enable", e.target.checked)}
                  className="w-3.5 h-3.5"
                />
                <span className="text-sm text-gray-300">REPA (Representation Alignment)</span>
              </label>
              {params.repa_enable && (
                <>
                  <div className="flex gap-2">
                    <div className="flex-1">
                      <label className="block text-xs text-gray-400 mb-1">Encoder source</label>
                      <select
                        value={params.repa_encoder_source || "tagger"}
                        onChange={(e) => updateParam("repa_encoder_source", e.target.value)}
                        className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs"
                      >
                        <option value="tagger">Anime tagger (SigLIP2, domain-matched)</option>
                        <option value="siglip2">google/siglip2 (off-the-shelf)</option>
                      </select>
                    </div>
                    <div className="flex-1">
                      <label className="block text-xs text-gray-400 mb-1">Align depth (-1 = auto)</label>
                      <input
                        type="number"
                        value={params.repa_align_depth ?? -1}
                        onChange={(e) => updateParam("repa_align_depth", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("repa_align_depth", -1); }}
                        className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs"
                      />
                    </div>
                  </div>
                  {(params.repa_encoder_source || "tagger") === "tagger" ? (
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Tagger model dir (empty = auto-pick newest)</label>
                      <input
                        type="text"
                        value={params.repa_tagger_model_dir || ""}
                        onChange={(e) => updateParam("repa_tagger_model_dir", e.target.value)}
                        placeholder="tagger_models/<uuid>"
                        className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs font-mono"
                      />
                    </div>
                  ) : (
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">SigLIP2 repo</label>
                      <input
                        type="text"
                        value={params.repa_siglip2_repo || ""}
                        onChange={(e) => updateParam("repa_siglip2_repo", e.target.value)}
                        placeholder="google/siglip2-so400m-patch14-384"
                        className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs font-mono"
                      />
                    </div>
                  )}
                  <div className="flex gap-2">
                    <div className="flex-1">
                      <label className="block text-xs text-gray-400 mb-1">Weight (λ)</label>
                      <input
                        type="number"
                        step="any"
                        value={params.repa_weight ?? 0.5}
                        onChange={(e) => updateParam("repa_weight", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("repa_weight", 0.5); }}
                        className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs"
                      />
                    </div>
                    <div className="flex-1">
                      <label className="block text-xs text-gray-400 mb-1">Projector LR factor</label>
                      <input
                        type="number"
                        step="any"
                        value={params.repa_proj_lr_factor ?? 1.0}
                        onChange={(e) => updateParam("repa_proj_lr_factor", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("repa_proj_lr_factor", 1.0); }}
                        min={0}
                        className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs"
                      />
                    </div>
                    <div className="flex-1">
                      <label className="block text-xs text-gray-400 mb-1">Enc. res (0 = native)</label>
                      <input
                        type="number"
                        value={params.repa_encoder_resolution ?? 0}
                        onChange={(e) => updateParam("repa_encoder_resolution", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("repa_encoder_resolution", 0); }}
                        className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs"
                      />
                    </div>
                  </div>
                  <p className="text-xs text-gray-500">
                    Aligns a DiT hidden state (at the align depth) with frozen clean-image
                    patch features from the encoder, accelerating convergence (arXiv:2410.06940).
                    Training-only; the projector is not saved into the model.
                  </p>
                </>
              )}
            </div>
          )}
        </div>

        {/* Vision Encoder selector — SD/SDXL only, shown below Base Model */}
        {isSDOrSDXLModel(baseModelPath) && (
          <div className="break-inside-avoid bg-gray-800/50 rounded-lg p-3 space-y-2">
            <label className="block text-xs text-gray-400 font-medium">
              Vision Encoder (SigLIP2)
              <span className="ml-1 text-gray-500 font-normal">— optional, SD/SDXL only</span>
            </label>
            <VisionEncoderSelector
              value={visionEncoderPath || null}
              onChange={(path) => {
                updateParam("vision_encoder_path", path || "");
                updateParam("use_reference_images", !!path);
                if (!path) {
                  updateParam("train_vision_encoder", false);
                  updateParam("gradient_routing_ve", false);
                }
              }}
              label=""
            />
            <p className="text-xs text-gray-500">
              Selecting a VE enables reference conditioning. SigLIP2 tokens are appended to the text context for referenced items; items without a reference remain text-conditioned normally.
            </p>
          </div>
        )}

        {/* LoRA Settings */}
        {(trainingMethod === "lora" || trainingMethod === "relora") && (
          <div className="break-inside-avoid bg-gray-800/50 rounded-lg p-3 space-y-3">
            <h3 className="text-sm font-semibold">LoRA Settings</h3>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Rank</label>
                <input
                  type="number"
                  value={loraRank}
                  onChange={(e) => updateParam("lora_rank", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("lora_rank", 16); }}
                  min="1"
                  max="256"
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-xs text-gray-400 mb-1">Alpha</label>
                <input
                  type="number"
                  value={loraAlpha}
                  onChange={(e) => updateParam("lora_alpha", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("lora_alpha", 16); }}
                  min="1"
                  max="256"
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">LoRA Weight Dtype</label>
              <select
                value={loraDtype}
                onChange={(e) => updateParam("lora_dtype", e.target.value as "fp32" | "fp16" | "bf16")}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="fp32">FP32 (Full Precision)</option>
                <option value="bf16">BF16 (Brain Float 16)</option>
                <option value="fp16">FP16 (Half Precision)</option>
              </select>
            </div>

            {adapterAlgorithmChoices.length > 1 && (
              <div>
                <label className="block text-xs text-gray-400 mb-1">Adapter Algorithm</label>
                <select
                  value={adapterAlgorithm}
                  onChange={(e) => updateParam("adapter_algorithm", e.target.value as "lora" | "loha" | "lokr")}
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                >
                  {adapterAlgorithmChoices.map((name) => (
                    <option key={name} value={name}>
                      {ADAPTER_ALGORITHM_LABELS[name]}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  {ADAPTER_ALGORITHM_NOTES[adapterAlgorithm]}
                </p>
                {adapterAlgorithm !== "lora" && (
                  <p className="text-xs text-gray-500 mt-1">
                    Block swap is not available with this algorithm.
                  </p>
                )}
              </div>
            )}
            {adapterAlgorithmCollapsedNote && (
              <p className="text-xs text-gray-500">{adapterAlgorithmCollapsedNote}</p>
            )}

            {weightDecomposeAvailable && (
              <div>
                <label className="flex items-center gap-2 text-xs text-gray-300">
                  <input
                    type="checkbox"
                    checked={weightDecompose}
                    onChange={(e) => updateParam("weight_decompose", e.target.checked)}
                    className="rounded"
                  />
                  Weight decomposition ({decomposedAdapterFamily(adapterAlgorithm).toUpperCase()})
                </label>
                <p className="text-xs text-gray-500 mt-1">
                  One magnitude vector per target (dora_scale) on top of the
                  algebra&apos;s factors. Not available with block swap or an FP8 base.
                </p>
              </div>
            )}
            {!weightDecomposeAvailable && weightDecomposeUnavailableNote && (
              <p className="text-xs text-gray-500">{weightDecomposeUnavailableNote}</p>
            )}
          </div>
        )}

        {/* ReLoRA Settings */}
        {trainingMethod === "relora" && (
          <div className="break-inside-avoid bg-gray-800/50 rounded-lg p-3 space-y-3">
            <h3 className="text-sm font-semibold">ReLoRA Settings</h3>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Merge Interval</label>
                <input
                  type="number"
                  value={reloraMergeEvery}
                  onChange={(e) => updateParam("relora_merge_every", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))}
                  onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("relora_merge_every", 500); }}
                  min="1"
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-xs text-gray-400 mb-1">Merge Unit</label>
                <select
                  value={reloraMergeUnit}
                  onChange={(e) => updateParam("relora_merge_unit", e.target.value as "steps" | "epochs")}
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                >
                  <option value="steps">Steps</option>
                  <option value="epochs">Epochs</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Restart Warmup Steps</label>
              <input
                type="number"
                value={restartWarmupSteps}
                onChange={(e) => updateParam("restart_warmup_steps", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))}
                onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("restart_warmup_steps", 100); }}
                min="0"
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">LR warmup steps after each merge-reinit cycle</p>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Optimizer Reset Strategy</label>
              <select
                value={optimizerResetStrategy}
                onChange={(e) => updateParam("optimizer_reset_strategy", e.target.value as "full_reset" | "magnitude_pruning" | "random_pruning")}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="full_reset">Full Reset (Clear all optimizer state)</option>
                <option value="magnitude_pruning">Magnitude Pruning (Keep top-N% by magnitude)</option>
                <option value="random_pruning">Random Pruning (Randomly keep N%)</option>
              </select>
            </div>

            {optimizerResetStrategy !== "full_reset" && (
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Pruning Ratio: {optimizerPruningRatio.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={optimizerPruningRatio}
                  onChange={(e) => updateParam("optimizer_pruning_ratio", parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
                <p className="text-xs text-gray-500 mt-1">Fraction of optimizer state entries to prune (set to 0)</p>
              </div>
            )}
          </div>
        )}

        {/* ControlNet Settings */}
        {trainingMethod === "controlnet" && (
          <div className="break-inside-avoid bg-gray-800/50 rounded-lg p-3 space-y-3">
            <h3 className="text-sm font-semibold">ControlNet Settings</h3>

            <div>
              <label className="block text-xs text-gray-400 mb-1">ControlNet Type</label>
              <select
                value={controlnetType}
                onChange={(e) => updateParam("controlnet_type", e.target.value as "standard" | "lllite")}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="standard">Standard (diffusers ControlNetModel)</option>
                <option value="lllite" disabled={conditioningMode === "outpaint"}>LLLite (kohya-ss sd-scripts compatible){conditioningMode === "outpaint" ? " - unavailable in Outpaint-native mode" : ""}</option>
              </select>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Conditioning Source</label>
              <select
                value={conditioningMode}
                onChange={(e) => {
                  const mode = e.target.value as "preprocessor" | "outpaint";
                  updateParam("conditioning_mode", mode);
                  // Outpaint conditioning is structurally incompatible with LLLite
                  // (4-ch crop+mask cond vs. LLLite's hardcoded 3-ch encoder).
                  if (mode === "outpaint" && controlnetType === "lllite") {
                    updateParam("controlnet_type", "standard");
                  }
                }}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="preprocessor">Preprocessor (paired / aux)</option>
                <option value="outpaint">Outpaint-native (crop→full)</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">Outpaint-native builds a self-supervised crop-to-full conditioning pair from each dataset item's own image (no paired condition images needed).</p>
            </div>

            {conditioningMode === "outpaint" && (
              <div className={`text-xs rounded p-2 border ${bucketStrategy === "resize" && !params.crop_augment_enable ? "border-gray-700 bg-gray-900/50 text-gray-400" : "border-yellow-700 bg-yellow-900/20 text-yellow-300"}`}>
                Outpaint-native conditioning requires <strong>ControlNet Type = Standard</strong>, <strong>Bucketing Strategy = Resize</strong>, and <strong>Crop Augmentation = disabled</strong> (Dataset / Bucketing section). Mismatched settings will fail training at startup.
                {(bucketStrategy !== "resize" || params.crop_augment_enable) && (
                  <> Current: bucket_strategy=&quot;{bucketStrategy}&quot;, crop_augment_enable={String(!!params.crop_augment_enable)}.</>
                )}
              </div>
            )}

            {conditioningMode === "outpaint" && (
              <div className="space-y-3 border-t border-gray-700 pt-3">
                <h4 className="text-xs font-semibold text-gray-300">Outpaint Conditioning Settings</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Crop Min Area</label>
                    <input
                      type="number"
                      value={outpaintCropMinArea}
                      onChange={(e) => updateParam("outpaint_crop_min_area", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                      onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("outpaint_crop_min_area", 0.15); }}
                      min="0.0"
                      max="1.0"
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Crop Max Area</label>
                    <input
                      type="number"
                      value={outpaintCropMaxArea}
                      onChange={(e) => updateParam("outpaint_crop_max_area", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                      onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("outpaint_crop_max_area", 0.8); }}
                      min="0.0"
                      max="1.0"
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Edge Anchor Probability</label>
                    <input
                      type="number"
                      value={outpaintEdgeAnchorProb}
                      onChange={(e) => updateParam("outpaint_edge_anchor_prob", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                      onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("outpaint_edge_anchor_prob", 0.34); }}
                      min="0.0"
                      max="1.0"
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Corner Anchor Probability</label>
                    <input
                      type="number"
                      value={outpaintCornerAnchorProb}
                      onChange={(e) => updateParam("outpaint_corner_anchor_prob", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                      onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("outpaint_corner_anchor_prob", 0.33); }}
                      min="0.0"
                      max="1.0"
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Known-region Loss Weight</label>
                    <input
                      type="number"
                      value={outpaintKnownLossWeight}
                      onChange={(e) => updateParam("outpaint_known_loss_weight", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                      onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("outpaint_known_loss_weight", 0.3); }}
                      min="0.0"
                      max="0.499"
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">Must stay below 0.5 (backend clamps to [0, 0.5)).</p>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Seam Loss Boost</label>
                    <input
                      type="number"
                      value={outpaintSeamLossBoost}
                      onChange={(e) => updateParam("outpaint_seam_loss_boost", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                      onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("outpaint_seam_loss_boost", 0.0); }}
                      min="0.0"
                      max="1.0"
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Seam Ring Width (latent cells)</label>
                    <select
                      value={outpaintSeamRingWidth}
                      onChange={(e) => updateParam("outpaint_seam_ring_width", parseInt(e.target.value, 10))}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    >
                      <option value={1}>1 (single ring)</option>
                      <option value={2}>2 (adds a second ring at half the boost increment)</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">Number of seam rings Seam Loss Boost covers. No effect when Seam Loss Boost is 0.</p>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Seam Continuity Lambda</label>
                    <input
                      type="number"
                      value={outpaintSeamGradLambda}
                      onChange={(e) => updateParam("outpaint_seam_grad_lambda", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                      onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("outpaint_seam_grad_lambda", 0.0); }}
                      min="0.0"
                      max="1.0"
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">Weight of the cross-seam prediction-error continuity term. 0 (default) disables it.</p>
                  </div>
                </div>

                <div>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={outpaintMaskChannel}
                      onChange={(e) => updateParam("outpaint_mask_channel", e.target.checked)}
                      className="w-4 h-4"
                    />
                    <span className="text-sm">Add known-region mask channel (4-ch conditioning)</span>
                  </label>
                  <p className="text-xs text-gray-500 mt-1">Adds a binary known/unknown mask as a 4th conditioning channel alongside crop RGB.</p>
                </div>

                <div>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={outpaintLossNormalize}
                      onChange={(e) => updateParam("outpaint_loss_normalize", e.target.checked)}
                      className="w-4 h-4"
                    />
                    <span className="text-sm">Normalize loss by weight-sum (opt-in)</span>
                  </label>
                  <p className="text-xs text-gray-500 mt-1">Decouples per-sample loss scale from the known/generate rect area. Default off preserves existing behavior.</p>
                </div>
              </div>
            )}

            {controlnetType === "standard" && (
              <div>
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={controlnetInitFromUnet}
                    onChange={(e) => updateParam("controlnet_init_from_unet", e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="text-sm">Initialize from UNet weights</span>
                </label>
                <p className="text-xs text-gray-500 mt-1">Copy UNet encoder weights to ControlNet for faster convergence</p>
              </div>
            )}

            <div>
              <label className="block text-xs text-gray-400 mb-1">Pretrained ControlNet (optional)</label>
              <select
                value={controlnetPretrainedPath}
                onChange={(e) => updateParam("controlnet_pretrained_path", e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="">None (initialize from scratch)</option>
                {availableControlNets.map((cn) => (
                  <option key={cn.path} value={cn.path}>{cn.name}</option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-1">Select an existing ControlNet checkpoint to resume training from</p>
            </div>

            {controlnetType === "lllite" && (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Conditioning Channels</label>
                  <input
                    type="number"
                    value={llliteConditioningChannels}
                    onChange={(e) => updateParam("lllite_conditioning_channels", parseInt(e.target.value) || 32)}
                    min="8"
                    max="128"
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Rank</label>
                  <input
                    type="number"
                    value={llliteRank}
                    onChange={(e) => updateParam("lllite_rank", parseInt(e.target.value) || 64)}
                    min="4"
                    max="256"
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>
            )}

            {conditioningMode === "preprocessor" && (
              <div>
                <label className="block text-xs text-gray-400 mb-1">Condition Image Preprocessors</label>
                <div className="flex flex-wrap gap-2 mt-1">
                  {["canny", "hed", "lineart", "lineart_anime", "depth_midas", "depth_zoe", "normal_bae", "openpose", "pidi", "shuffle", "teed", "anyline"].map((pp) => (
                    <label key={pp} className="flex items-center space-x-1 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={conditionPreprocessors.includes(pp)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            updateParam("condition_preprocessors", [...conditionPreprocessors, pp]);
                          } else {
                            updateParam("condition_preprocessors", conditionPreprocessors.filter(p => p !== pp));
                          }
                        }}
                        className="w-3.5 h-3.5"
                      />
                      <span className="text-xs text-gray-300">{pp}</span>
                    </label>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-1">Auto-generate condition images when reference images are not provided. Multiple selections = random per image.</p>
              </div>
            )}

            {conditioningMode === "preprocessor" && conditionPreprocessors.length > 0 && (
              <div>
                <label className="block text-xs text-gray-400 mb-1">Cache Mode</label>
                <select
                  value={conditionCacheMode}
                  onChange={(e) => updateParam("condition_cache_mode", e.target.value as "on_the_fly" | "pre_generate")}
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                >
                  <option value="on_the_fly">On-the-fly (generate during training)</option>
                  <option value="pre_generate">Pre-generate (cache before training)</option>
                </select>
              </div>
            )}

            {/* Condition image for sample generation is now configured per-prompt in the Sample Generation section */}

            <p className="text-xs text-gray-500">ControlNet training freezes UNet/VAE/Text Encoder. Only the ControlNet module is trained.</p>
          </div>
        )}

        {/* Training Parameters */}
        <div className="break-inside-avoid bg-gray-800/50 rounded-lg p-3 space-y-3">
          <h3 className="text-sm font-semibold">Training Parameters</h3>

          {/* Steps/Epochs Toggle */}
          <div className="flex items-center space-x-4 mb-2">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                checked={!useEpochs}
                onChange={() => setUseEpochs(false)}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm">Steps</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                checked={useEpochs}
                onChange={() => setUseEpochs(true)}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm">Epochs</span>
            </label>
          </div>

          <div className="grid grid-cols-2 gap-3">
            {!useEpochs ? (
              <div>
                <label className="block text-xs text-gray-400 mb-1">Steps</label>
                <input
                  type="number"
                  value={totalSteps}
                  onChange={(e) => updateParam("total_steps", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("total_steps", 1000); }}
                  min="1"
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
              </div>
            ) : (
              <div>
                <label className="block text-xs text-gray-400 mb-1">Epochs</label>
                <input
                  type="number"
                  value={epochs}
                  onChange={(e) => updateParam("epochs", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("epochs", 10); }}
                  min="1"
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
              </div>
            )}

            <div>
              <label className="block text-xs text-gray-400 mb-1">Batch Size</label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => updateParam("batch_size", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("batch_size", requiredValue("batch_size") ? Number(requiredValue("batch_size")!.value) : 1); }}
                min="1"
                disabled={!!requiredValue("batch_size")}
                title={requiredValue("batch_size")?.reason}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-60"
              />
              <RequiredValueNote entry={requiredValue("batch_size")} />
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Gradient Accumulation Steps</label>
              <input
                type="number"
                value={params.gradient_accumulation_steps ?? 1}
                onChange={(e) => updateParam("gradient_accumulation_steps", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value)) || parseInt(e.target.value) < 1) updateParam("gradient_accumulation_steps", 1); }}
                min="1"
                disabled={!!requiredValue("gradient_accumulation_steps")}
                title={requiredValue("gradient_accumulation_steps")?.reason}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-60"
              />
              <RequiredValueNote entry={requiredValue("gradient_accumulation_steps")} />
              <p className="text-xs text-gray-500 mt-1">Effective batch = Batch Size × this. Reduces gradient noise without extra VRAM.</p>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Max Grad Norm</label>
              <input
                type="number"
                step="any"
                value={params.max_grad_norm ?? 1.0}
                onChange={(e) => updateParam("max_grad_norm", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("max_grad_norm", 1.0); }}
                min="0"
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">Gradient clipping threshold. 0 disables clipping.</p>
              {gradClippingIgnoredReason && (
                <p className="text-xs text-amber-400 mt-1">{gradClippingIgnoredReason}</p>
              )}
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Multi Noise-Timesteps (MNT)
              </label>
              <input
                type="number"
                value={multiNoiseTimesteps}
                onChange={(e) => updateParam("multi_noise_timesteps", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("multi_noise_timesteps", 1); }}
                min="1"
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                Process each batch with multiple different timesteps (default: 1)
              </p>
            </div>

            {multiNoiseTimesteps > 1 && (
              <div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={params.stratified_timesteps ?? true}
                    onChange={(e) => updateParam("stratified_timesteps", e.target.checked)}
                    className="w-3.5 h-3.5"
                  />
                  <span className="text-xs text-gray-400">Stratified timesteps</span>
                </label>
                <p className="text-xs text-gray-500 mt-1">
                  Draw one timestep per equal-probability stratum across the {multiNoiseTimesteps}
                  {" "}MNT iterations instead of drawing each independently. Each draw keeps the same
                  marginal distribution, so the configured timestep density is unchanged.
                  Not available for the beta distribution.
                </p>
              </div>
            )}

            {multiNoiseTimesteps > 1 && (
              <div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={params.grad_timestep_cosine_probe ?? false}
                    onChange={(e) => updateParam("grad_timestep_cosine_probe", e.target.checked)}
                    className="w-3.5 h-3.5"
                  />
                  <span className="text-xs text-gray-400">Gradient cosine probe (noisy vs clean timesteps)</span>
                </label>
                <p className="text-xs text-gray-500 mt-1">
                  Diagnostic. Splits each MNT window's gradients at the sampler's median
                  timestep and charts the cosine between the two halves per branch
                  (grad_cos_t_*). Near 0 means distant timesteps are uncorrelated;
                  negative means they conflict. Needs the fused backward path.
                </p>
                {params.grad_timestep_cosine_probe && (
                  <div className="mt-2">
                    <label className="block text-xs text-gray-400 mb-1">Sketch dimension</label>
                    <NumberInput
                      value={params.grad_timestep_cosine_sketch_dim ?? 8}
                      onCommit={(v) => updateParam("grad_timestep_cosine_sketch_dim", v)}
                      defaultValue={8}
                      min={1}
                      max={64}
                      step={1}
                      parse="int"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Higher is a less noisy cosine estimate at more compute
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* MNT Mode */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                MNT Mode
              </label>
              <select
                value={multiNoiseMode}
                onChange={(e) => updateParam("multi_noise_mode", e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="independent">Independent (Different noise)</option>
                <option value="shared">Shared (Same noise)</option>
                <option value="trajectory">Trajectory (Sequential learning)</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {multiNoiseMode === "independent" && "Each MNT iteration uses different noise (default)"}
                {multiNoiseMode === "shared" && "All MNT iterations use same noise (trajectory consistency)"}
                {multiNoiseMode === "trajectory" && "Sequential trajectory learning with blending"}
              </p>
            </div>

            {/* Trajectory Blend Alpha (only for trajectory mode) */}
            {multiNoiseMode === "trajectory" && (
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Trajectory Blend Alpha
                </label>
                <input
                  type="number"
                  value={trajectoryBlendAlpha}
                  onChange={(e) => updateParam("trajectory_blend_alpha", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("trajectory_blend_alpha", 0.7); }}
                  min="0.0"
                  max="1.0"
                  step="any"
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Blending coefficient: 0.0=ideal only, 1.0=stepped only (default: 0.7)
                </p>
              </div>
            )}

            {/* Timestep Sampling */}
            {(() => {
              const timestepConvention = getTimestepConvention(baseModelPath, noiseProcess);
              const cleanEndLabel = timestepConvention === "t1"
                ? "1.0 = clean, 0.0 = fully noised"
                : timestepConvention === "auto"
                  ? "0.0 = clean, 1.0 = fully noised for noise_process=flow; the reverse for noise_process=ddpm"
                  : "0.0 = clean, 1.0 = fully noised";
              const meanBiasNote = timestepConvention === "t1"
                ? "Mean: positive = high timesteps (clean), negative = low timesteps (noisy). Std: spread"
                : timestepConvention === "auto"
                  ? "Mean sign meaning depends on noise_process (flow: positive=noisy/negative=clean; ddpm: reversed). Std: spread"
                  : "Mean: positive = high timesteps (noisy), negative = low timesteps (clean). Std: spread";
              const graphAxisLabel = timestepConvention === "t1"
                ? "X-axis: Timestep (0=noisy, 1=clean) | Y-axis: Sampling probability"
                : timestepConvention === "auto"
                  ? "X-axis: Timestep (0=clean/1=noisy for flow; reversed for ddpm) | Y-axis: Sampling probability"
                  : "X-axis: Timestep (0=clean, 1=noisy) | Y-axis: Sampling probability";
              return (
            <div className="col-span-2 border-t border-gray-700 pt-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">Timestep Sampling</h3>
              <div className="grid grid-cols-1 gap-4">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Distribution</label>
                  <select
                    value={timestepDistribution}
                    onChange={(e) => setTimestepDistribution(e.target.value)}
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  >
                    <option value="uniform">Uniform (Default)</option>
                    <option value="logit_normal">Logit-Normal (FLUX/SD3)</option>
                    <option value="normal">Normal (Gaussian)</option>
                    <option value="beta">Beta Distribution</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    Probability distribution for sampling timesteps during training
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Min Timestep</label>
                    <input
                      type="number"
                      value={timestepMin}
                      onChange={(e) => setTimestepMin(e.target.value === ''  ? '' as any : parseFloat(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) setTimestepMin(0.0); }}
                      min="0.0"
                      max="1.0"
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Max Timestep</label>
                    <input
                      type="number"
                      value={timestepMax}
                      onChange={(e) => setTimestepMax(e.target.value === ''  ? '' as any : parseFloat(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) setTimestepMax(1.0); }}
                      min="0.0"
                      max="1.0"
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                </div>
                <p className="text-xs text-gray-500">
                  Timestep range for sampling ({cleanEndLabel})
                </p>

                {/* Distribution-specific parameters: Mean/Std for logit_normal and normal */}
                {(timestepDistribution === "logit_normal" || timestepDistribution === "lognormal" || timestepDistribution === "normal") && (
                  <div className="grid grid-cols-2 gap-4 mt-2">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">
                        Mean {timestepDistribution === "normal" ? "(center)" : "(bias)"}
                      </label>
                      <input
                        type="number"
                        value={timestepMean}
                        onChange={(e) => setTimestepMean(e.target.value === '' ? '' as any : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) setTimestepMean(timestepDistribution === "normal" ? 0.5 : 0.0); }}
                        step="any"
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Std (spread)</label>
                      <input
                        type="number"
                        value={timestepStd}
                        onChange={(e) => setTimestepStd(e.target.value === '' ? '' as any : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) setTimestepStd(1.0); }}
                        min="0.01"
                        step="any"
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <p className="col-span-2 text-xs text-gray-500">
                      {timestepDistribution === "normal" ? (
                        <>Mean: center of distribution (0.0-1.0). Std: spread (smaller = more concentrated)</>
                      ) : (
                        <>{meanBiasNote}</>
                      )}
                    </p>
                  </div>
                )}

                {/* Distribution-specific parameters: Alpha/Beta for beta distribution */}
                {timestepDistribution === "beta" && (
                  <div className="grid grid-cols-2 gap-4 mt-2">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Alpha</label>
                      <input
                        type="number"
                        value={timestepAlpha}
                        onChange={(e) => setTimestepAlpha(e.target.value === '' ? '' as any : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) setTimestepAlpha(2.0); }}
                        min="0.1"
                        step="any"
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Beta</label>
                      <input
                        type="number"
                        value={timestepBeta}
                        onChange={(e) => setTimestepBeta(e.target.value === '' ? '' as any : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) setTimestepBeta(2.0); }}
                        min="0.1"
                        step="any"
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <p className="col-span-2 text-xs text-gray-500">
                      α=β=1: uniform, α=β=2: bell-shaped, α&gt;β: bias to high timesteps, α&lt;β: bias to low timesteps
                    </p>
                  </div>
                )}

                {/* Distribution Preview Graph */}
                <div className="mt-4">
                  <label className="block text-xs text-gray-400 mb-2">Distribution Preview</label>
                  <TimestepDistributionGraph
                    distribution={timestepDistribution}
                    minTimestep={timestepMin}
                    maxTimestep={timestepMax}
                    mean={timestepMean}
                    std={timestepStd}
                    alpha={timestepAlpha}
                    beta={timestepBeta}
                  />
                  <p className="text-[10px] text-gray-500 mt-1 text-center">
                    {graphAxisLabel}
                  </p>
                </div>
              </div>
            </div>
              );
            })()}

            {/* Regularization Settings */}
            <div className="space-y-4 p-3 bg-gray-900/50 rounded border border-gray-700/50">
              <div>
                <label className="block text-xs text-gray-400 mb-1 font-semibold">
                  Regularization (Prevent Overbaking)
                </label>
                <p className="text-xs text-gray-500 mt-1">
                  Both SNR and Energy regularization can be enabled simultaneously for comprehensive overbaking prevention.
                </p>
              </div>

              {/* SNR Regularization */}
              <div className="space-y-3 p-2 bg-gray-800/30 rounded border border-gray-700/30">
                <div className="flex items-center justify-between">
                  <label className="block text-xs text-gray-300 font-semibold">
                    SNR Regularization (Frequency Domain)
                  </label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="enable-snr-reg"
                      checked={snrRegularizationWeight > 0}
                      onChange={(e) => updateParam("snr_regularization_weight", e.target.checked ? 0.1 : 0.0)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="enable-snr-reg" className="text-xs text-gray-400 cursor-pointer">
                      Enable
                    </label>
                  </div>
                </div>
                <p className="text-xs text-gray-500">
                  Penalizes high SNR in predicted latents (prevents over-denoising in frequency domain)
                </p>

              {snrRegularizationWeight > 0 && (
                <>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      SNR Weight
                    </label>
                    <input
                      type="number"
                      value={snrRegularizationWeight}
                      onChange={(e) => updateParam("snr_regularization_weight", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("snr_regularization_weight", 0.0); }}
                      min="0.0"
                      max="1.0"
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">Recommended: 0.1</p>
                  </div>

                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="snr-timestep-adaptive"
                      checked={snrTimestepAdaptive}
                      onChange={(e) => updateParam("snr_timestep_adaptive", e.target.checked)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="snr-timestep-adaptive" className="text-xs text-gray-300 cursor-pointer">
                      Timestep Adaptive (stronger penalty at low timesteps)
                    </label>
                  </div>

                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Penalty Mode
                    </label>
                    <select
                      value={snrPenaltyMode}
                      onChange={(e) => updateParam("snr_penalty_mode", e.target.value)}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    >
                      <option value="relu">ReLU (one-sided, penalize only over-denoising)</option>
                      <option value="abs">Absolute (two-sided, penalize any deviation)</option>
                    </select>
                  </div>
                </>
              )}
              </div>

              {/* Energy Regularization */}
              <div className="space-y-3 p-2 bg-gray-800/30 rounded border border-gray-700/30">
                <div className="flex items-center justify-between">
                  <label className="block text-xs text-gray-300 font-semibold">
                    Energy Regularization (Spatial Domain)
                  </label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="enable-energy-reg"
                      checked={energyRegularizationWeight > 0}
                      onChange={(e) => updateParam("energy_regularization_weight", e.target.checked ? 0.1 : 0.0)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="enable-energy-reg" className="text-xs text-gray-400 cursor-pointer">
                      Enable
                    </label>
                  </div>
                </div>
                <p className="text-xs text-gray-500">
                  Penalizes energy deviation in predicted latents (prevents detail loss in spatial domain)
                </p>

              {energyRegularizationWeight > 0 && (
                <>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Energy Weight
                    </label>
                    <input
                      type="number"
                      value={energyRegularizationWeight}
                      onChange={(e) => updateParam("energy_regularization_weight", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("energy_regularization_weight", 0.0); }}
                      min="0.0"
                      max="1.0"
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">Recommended: 0.1</p>
                  </div>

                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="energy-timestep-adaptive"
                      checked={energyTimestepAdaptive}
                      onChange={(e) => updateParam("energy_timestep_adaptive", e.target.checked)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="energy-timestep-adaptive" className="text-xs text-gray-300 cursor-pointer">
                      Timestep Adaptive (stronger penalty at low timesteps)
                    </label>
                  </div>

                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Penalty Mode
                    </label>
                    <select
                      value={energyPenaltyMode}
                      onChange={(e) => updateParam("energy_penalty_mode", e.target.value)}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    >
                      <option value="under">Under (one-sided, penalize only energy loss - recommended)</option>
                      <option value="abs">Absolute (two-sided, penalize any deviation)</option>
                    </select>
                  </div>

                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="energy-normalize-by-pixels"
                      checked={energyNormalizeByPixels}
                      onChange={(e) => updateParam("energy_normalize_by_pixels", e.target.checked)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="energy-normalize-by-pixels" className="text-xs text-gray-300 cursor-pointer">
                      Normalize by Pixels (resolution-independent)
                    </label>
                  </div>
                </>
              )}
              </div>
            </div>

            {/* Unified Training Framework Settings */}
            <div className="bg-gray-800 p-3 rounded space-y-3">
              <div>
                <label className="block text-xs text-gray-300 font-semibold mb-2">
                  Unified Training Framework
                </label>
                <p className="text-xs text-gray-500 mb-3">
                  Configure noise process and prediction target for training
                </p>

                <div className="space-y-3">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Noise Process
                    </label>
                    <select
                      value={noiseProcess}
                      onChange={(e) => updateParam("noise_process", e.target.value)}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    >
                      <option value="auto">Auto (detect from model)</option>
                      <option value="ddpm">DDPM (Scheduled, for SDXL/SD1.5)</option>
                      <option value="flow">Flow Matching (Linear, for Z-Image)</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      How noise is added during training. Auto-detect uses model&apos;s original configuration.
                    </p>
                  </div>

                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Prediction Target
                    </label>
                    <select
                      value={predictionTarget}
                      onChange={(e) => updateParam("prediction_target", e.target.value)}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    >
                      <option value="auto">Auto (detect from model)</option>
                      <option value="epsilon">Epsilon (predict noise)</option>
                      <option value="velocity">Velocity (predict direction)</option>
                      <option value="sample">Sample (predict x₀)</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      What the model predicts during training. Auto-detect uses model&apos;s original configuration.
                    </p>
                  </div>

                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      SDXL VAE Migration (SDXL only)
                    </label>
                    <select
                      value={params.sdxl_vae_type ?? "none"}
                      onChange={(e) => updateParam("sdxl_vae_type", e.target.value)}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    >
                      <option value="none">None (standard SDXL VAE, 4ch)</option>
                      <option value="flux1">FLUX.1 VAE (16ch)</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Swaps the VAE and resizes the U-Net conv_in/out to the new latent
                      channel count (body kept; in/out re-adapt during training). Produces a
                      non-standard &quot;sdxl-custom&quot; checkpoint.
                    </p>
                  </div>

                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      SDXL Text Encoder swap (SDXL only)
                    </label>
                    <select
                      value={params.sdxl_te_type ?? "none"}
                      onChange={(e) => updateParam("sdxl_te_type", e.target.value)}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    >
                      <option value="none">None (standard CLIP)</option>
                      <option value="siglip2_text">SigLIP2 text tower</option>
                      <option value="flan_t5">FLAN-T5</option>
                      <option value="qwen3">Qwen3</option>
                    </select>
                    {params.sdxl_te_type && params.sdxl_te_type !== "none" ? (
                      <div className="mt-2 grid grid-cols-2 gap-2">
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Hidden layer (-2=penultimate)</label>
                          <input type="number" value={params.sdxl_te_hidden_layer ?? -2}
                            onChange={(e) => updateParam("sdxl_te_hidden_layer", parseInt(e.target.value))}
                            className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs" />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Max token length</label>
                          <input type="number" min={16} value={params.sdxl_te_max_len ?? 256}
                            onChange={(e) => updateParam("sdxl_te_max_len", parseInt(e.target.value) || 256)}
                            className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs" />
                        </div>
                        <label className="col-span-2 flex items-center gap-2 cursor-pointer">
                          <input type="checkbox" checked={!!params.sdxl_te_train_encoder}
                            onChange={(e) => updateParam("sdxl_te_train_encoder", e.target.checked)}
                            className="w-3.5 h-3.5" />
                          <span className="text-xs text-gray-300">
                            Train encoder body too (off = freeze TE, train bridge adapters only)
                          </span>
                        </label>
                      </div>
                    ) : null}
                    <p className="text-xs text-gray-500 mt-1">
                      Replaces CLIP with the selected encoder + trainable adapters bridging to
                      the U-Net (2048 / 1280). Produces a non-standard &quot;sdxl-custom&quot; checkpoint.
                    </p>
                  </div>

                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="strict-validation"
                      checked={strictValidation}
                      onChange={(e) => updateParam("strict_validation", e.target.checked)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="strict-validation" className="text-xs text-gray-300 cursor-pointer">
                      Strict Validation (abort training if mismatch detected)
                    </label>
                  </div>
                  <p className="text-xs text-gray-500">
                    When enabled, training aborts if noise_process/prediction_target doesn&apos;t match model&apos;s config. When disabled, shows warning and continues.
                  </p>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Learning Rate</label>
              <input
                type="text"
                value={localLrText}
                onChange={(e) => setLocalLrText(e.target.value)}
                onBlur={(e) => {
                  const v = parseFloat(e.target.value);
                  // Must be > 0: this is the fallback every component rate
                  // resolves to when unset, so 0 trains nothing and a
                  // negative value ascends the loss (backend refuses it too).
                  if (!isNaN(v) && v > 0) updateParam("learning_rate", v);
                }}
                placeholder="e.g., 1e-4"
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">LR Scheduler</label>
              <select
                value={lrScheduler}
                onChange={(e) => updateParam("lr_scheduler", e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="constant">Constant</option>
                <option value="cosine">Cosine</option>
                <option value="linear">Linear</option>
                <option value="plateau_cosine_floor">Plateau then Cosine Floor</option>
              </select>
              {lrScheduler === "plateau_cosine_floor" && (
                <p className="text-xs text-gray-500 mt-1">
                  Warmup, then holds the base LR flat, then cosine-decays down to a floor
                  (fraction of base LR) and holds that floor for the rest of training.
                </p>
              )}
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">LR Warmup Steps</label>
              <NumberInput
                value={lrWarmupSteps}
                onCommit={(v) => updateParam("lr_warmup_steps", v)}
                defaultValue={0}
                min={0}
                step={1}
                parse="int"
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>

            {lrScheduler === "plateau_cosine_floor" && (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Decay Start Ratio</label>
                  <NumberInput
                    value={lrDecayStartRatio}
                    onCommit={(v) => updateParam("lr_decay_start_ratio", v)}
                    defaultValue={0.85}
                    min={0}
                    max={1}
                    step="any"
                    parse="float"
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">Fraction of total steps where the plateau ends</p>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">LR Floor Ratio</label>
                  <NumberInput
                    value={lrFloorRatio}
                    onCommit={(v) => updateParam("lr_floor_ratio", v)}
                    defaultValue={0.25}
                    min={0}
                    max={1}
                    step="any"
                    parse="float"
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">Floor as a fraction of base LR (held after decay)</p>
                </div>
              </div>
            )}

            {lrWarmupSteps > 0 && (
              <div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={params.rewarmup_on_optimizer_reset ?? true}
                    onChange={(e) => updateParam("rewarmup_on_optimizer_reset", e.target.checked)}
                    className="w-3.5 h-3.5"
                  />
                  <span className="text-xs text-gray-400">Re-apply warmup if a resume resets the optimizer</span>
                </label>
                <p className="text-xs text-gray-500 mt-1">
                  When a resume finds no usable optimizer state, the schedule has already
                  advanced past its warmup. This re-applies the {lrWarmupSteps}-step ramp
                  from the resumed step without moving the schedule.
                </p>
              </div>
            )}

            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="use-ema"
                  checked={useEma}
                  onChange={(e) => updateParam("use_ema", e.target.checked)}
                  disabled={!!requiredValue("use_ema")}
                  title={requiredValue("use_ema")?.reason}
                  className="w-4 h-4 disabled:opacity-60"
                />
                <label htmlFor="use-ema" className="text-xs text-gray-300 cursor-pointer">
                  Weight EMA
                </label>
              </div>
              <RequiredValueNote entry={requiredValue("use_ema")} />
              <p className="text-xs text-gray-500">
                Maintains an exponential moving average of the trained weights and saves it
                as a separate, loadable checkpoint alongside each normal checkpoint
                (run name suffix &quot;_ema&quot;).
              </p>
              {useEma && (
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">EMA Decay</label>
                    <NumberInput
                      value={emaDecay}
                      onCommit={(v) => updateParam("ema_decay", v)}
                      defaultValue={0.9999}
                      min={0}
                      max={1}
                      step="any"
                      parse="float"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Update Every N Steps</label>
                    <NumberInput
                      value={emaUpdateEvery}
                      onCommit={(v) => updateParam("ema_update_every", Math.max(1, Math.round(v)))}
                      defaultValue={1}
                      min={1}
                      step={1}
                      parse="int"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">Decay is raised to the power N to keep the averaging horizon constant.</p>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Shadow Device</label>
                    <select
                      value={emaDevice}
                      onChange={(e) => updateParam("ema_device", e.target.value)}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    >
                      <option value="cpu">CPU (no extra VRAM)</option>
                      <option value="cuda">CUDA (no sync, uses VRAM)</option>
                    </select>
                  </div>
                </div>
              )}
            </div>

            {/* Optimizer Selection */}
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Optimizer</label>
                <select
                  value={optimizer}
                  onChange={(e) => updateParam("optimizer", e.target.value)}
                  disabled={!!requiredValue("optimizer")
                            && !requiredValue("optimizer")!.values}
                  title={requiredValue("optimizer")?.reason}
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-60"
                >
                  {/* Narrowed to the admitted set when the backend constrains
                      it, so the list never offers a value the run is refused
                      for. One admitted value also disables the control. */}
                  {requiredValue("optimizer") ? (
                    (requiredValue("optimizer")!.values
                     ?? [requiredValue("optimizer")!.value]).map((v) => (
                      <option key={String(v)} value={String(v)}>
                        {OPTIMIZER_CONFIGS[String(v)]?.label ?? String(v)}
                      </option>
                    ))
                  ) : (
                    <>
                  <option value="adamw">AdamW</option>
                  <option value="adamw8bit">AdamW 8-bit</option>
                  <option value="adamw8bit_ringbuffer">AdamW 8-bit Ring Buffer</option>
                  <option value="lion8bit">Lion 8-bit</option>
                  <option value="lion8bit_ringbuffer">Lion 8-bit Ring Buffer</option>
                  <option value="adafactor">Adafactor</option>
                    </>
                  )}
                </select>
                <RequiredValueNote entry={requiredValue("optimizer")} />
                <p className="text-xs text-gray-500 mt-1">
                  {optimizer === "adafactor" && "Adaptive learning rate"}
                  {optimizer === "lion8bit" && "Sign-based momentum, 8-bit quantization"}
                  {optimizer === "lion8bit_ringbuffer" && "Sign-based momentum, 8-bit quantization, CPU state allocation"}
                  {optimizer === "adamw8bit_ringbuffer" && "8-bit quantization, CPU state allocation"}
                  {optimizer === "adamw8bit" && "8-bit quantization"}
                  {optimizer === "adamw" && "Full precision"}
                </p>
              </div>

              {/* Optimizer Options */}
              <div className="grid grid-cols-2 gap-3">
                {/* No "Paged (CPU offload)" checkbox: paging is selected by the
                    optimizer name (paged_adamw / paged_adamw8bit /
                    paged_lion8bit), which this dropdown does not offer. The
                    checkbox set a flag no trainer read. */}

                {/* cautious option (Ring Buffer optimizers only) */}
                {OPTIMIZER_CONFIGS[optimizer]?.supportsCautious && (
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="optimizer-cautious"
                      checked={optimizerCautious}
                      onChange={(e) => updateParam("optimizer_cautious", e.target.checked)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="optimizer-cautious" className="text-xs text-gray-300 cursor-pointer">
                      Cautious (sign mask)
                    </label>
                  </div>
                )}

                {/* schedule-free option (Ring Buffer optimizers only) */}
                {OPTIMIZER_CONFIGS[optimizer]?.supportsCautious && (
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="optimizer-schedule-free"
                        checked={optimizerScheduleFree}
                        onChange={(e) => updateParam("optimizer_schedule_free", e.target.checked)}
                        className="w-4 h-4"
                      />
                      <label htmlFor="optimizer-schedule-free" className="text-xs text-gray-300 cursor-pointer">
                        Schedule-Free (learning rate scheduling)
                      </label>
                    </div>

                    {optimizerScheduleFree && (
                      <div className="ml-6 space-y-2 border-l-2 border-gray-600 pl-3">
                        {/* RAdam toggle */}
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            id="optimizer-use-radam"
                            checked={optimizerUseRadam}
                            onChange={(e) => updateParam("optimizer_use_radam", e.target.checked)}
                            className="w-4 h-4"
                          />
                          <label htmlFor="optimizer-use-radam" className="text-xs text-gray-300 cursor-pointer">
                            Use RAdam (Rectified Adam)
                          </label>
                        </div>

                        {/* Schedule-Free r (hidden when RAdam is enabled) */}
                        {!optimizerUseRadam && (
                          <div>
                            <label className="block text-xs text-gray-400 mb-1">r (warmup parameter)</label>
                            <input
                              type="text"
                              value={optimizerScheduleFreeR}
                              onChange={(e) => setLocalScheduleFreeRText(e.target.value)}
                              onBlur={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) updateParam("optimizer_schedule_free_r", v); }}
                              className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs focus:outline-none focus:border-blue-500"
                            />
                            <p className="text-xs text-gray-500 mt-1">Default: 0.0 (no warmup)</p>
                          </div>
                        )}

                        {/* Schedule-Free weight_lr_power */}
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Weight LR Power</label>
                          <input
                            type="text"
                            value={optimizerScheduleFreeWeightLrPower}
                            onChange={(e) => setLocalScheduleFreeWeightLrPowerText(e.target.value)}
                            onBlur={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) updateParam("optimizer_schedule_free_weight_lr_power", v); }}
                            className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs focus:outline-none focus:border-blue-500"
                          />
                          <p className="text-xs text-gray-500 mt-1">Default: 2.0</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Stochastic Rounding.
                    Applies to BF16 parameters only (should_use_stochastic_rounding
                    returns false for FP16/FP32), so the control is disabled when
                    the run will not have BF16 parameters. That is NOT simply the
                    dtype dropdown: train_runner forces weight_dtype=bf16 for the
                    architectures in FORCED_BF16_ARCHITECTURES below whatever the
                    dropdown says, and it leaves SD1.5 / SDXL / Flux 2 on the
                    configured dtype (fp16 by default).
                    It used to be nested inside the Schedule-Free block and
                    limited to the two Ring Buffer optimizers, which made it
                    unreachable for a default configuration. Every optimizer here
                    supports it except AdamW, which updates all of its parameters
                    in one call with no per-parameter seam to apply it at. */}
                {optimizer !== "adamw" && (() => {
                  const forcedBf16 = FORCED_BF16_ARCHITECTURES.has(
                    getModelArchitecture(baseModelPath) ?? ""
                  );
                  const bf16Params = forcedBf16 || weightDtype === "bf16" || trainingDtype === "bf16";
                  return (
                    <div className="col-span-2">
                      <label className={`block text-xs mb-1 ${bf16Params ? "text-gray-300" : "text-gray-500"}`}>
                        Stochastic Rounding (BF16 parameters)
                      </label>
                      <select
                        disabled={!bf16Params}
                        value={optimizerStochasticRounding}
                        onChange={(e) => {
                          const v = e.target.value;
                          updateParam(
                            "optimizer_stochastic_rounding",
                            v === "on" ? true : v === "off" ? false : null
                          );
                        }}
                        className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                      >
                        <option value="auto">Not specified (architecture decides)</option>
                        <option value="on">On</option>
                        <option value="off">Off</option>
                      </select>
                      <p className="text-xs text-gray-500 mt-1">
                        {bf16Params
                          ? "Rounds each BF16 parameter update up or down with probability equal to its fractional part. Without it, an update smaller than half a BF16 step is rounded away every step and the weight never changes. Some full fine-tune routes force this on when not specified and refuse an explicit Off."
                          : `Unavailable: this run's parameters are ${weightDtype.toUpperCase()}. Stochastic rounding applies to BF16 parameters only. Set the weight or training dtype to BF16 to enable it.`}
                      </p>
                    </div>
                  );
                })()}

                {/* Ring-buffer-only allocator choice. SenseNova full
                    fine-tuning REFUSES those two optimizers without it. */}
                {optimizer.endsWith("_ringbuffer") && (
                  <div className="col-span-2">
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="optimizer-state-host-resident"
                        checked={params.optimizer_state_host_resident ?? false}
                        onChange={(e) => updateParam("optimizer_state_host_resident", e.target.checked)}
                        className="w-4 h-4"
                      />
                      <label htmlFor="optimizer-state-host-resident" className="text-xs text-gray-300 cursor-pointer">
                        Host-resident optimizer state (pinned CPU memory)
                      </label>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      Allocates the 8-bit optimizer state as pinned host memory
                      instead of GPU memory; absmax stays on the GPU. Measured:
                      GPU state falls from 2.031250 to 0.031250 bytes per
                      parameter (AdamW) or 1.015625 to 0.015625 (Lion), against
                      2.0 / 1.0 bytes per parameter pinned on the host. The host
                      allocation cannot be paged out and is held for the whole
                      run. SenseNova full fine-tuning requires this for these two
                      optimizers and refuses the run without it.
                    </p>
                  </div>
                )}
              </div>

              {/* Optimizer Hyperparameters */}
              <div className="bg-gray-900 border border-gray-700 rounded p-3 space-y-2">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium text-gray-400">Hyperparameters</span>
                  <button
                    type="button"
                    onClick={() => {
                      const config = OPTIMIZER_CONFIGS[optimizer];
                      if (!config) return;
                      const { beta1, beta2, epsilon, weight_decay } = config.defaults;
                      if (beta1 !== undefined) { setLocalBeta1Text(beta1); updateParam("optimizer_beta1", parseFloat(beta1)); }
                      if (beta2 !== undefined) { setLocalBeta2Text(beta2); updateParam("optimizer_beta2", parseFloat(beta2)); }
                      if (epsilon !== undefined) { setLocalEpsilonText(epsilon); updateParam("optimizer_epsilon", parseFloat(epsilon)); }
                      if (weight_decay !== undefined) { setLocalWeightDecayText(weight_decay); updateParam("optimizer_weight_decay", parseFloat(weight_decay)); }
                    }}
                    className="text-xs text-blue-400 hover:text-blue-300"
                  >
                    Reset to Defaults
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  {/* Beta1 (not for Adafactor) */}
                  {OPTIMIZER_CONFIGS[optimizer]?.defaults.beta1 !== undefined && (
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Beta1</label>
                      <input
                        type="text"
                        value={optimizerBeta1}
                        onChange={(e) => setLocalBeta1Text(e.target.value)}
                        onBlur={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) updateParam("optimizer_beta1", v); }}
                        className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  )}

                  {/* Beta2 (not for Adafactor) */}
                  {OPTIMIZER_CONFIGS[optimizer]?.defaults.beta2 !== undefined && (
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Beta2</label>
                      <input
                        type="text"
                        value={optimizerBeta2}
                        onChange={(e) => setLocalBeta2Text(e.target.value)}
                        onBlur={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) updateParam("optimizer_beta2", v); }}
                        className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  )}

                  {/* Epsilon (not for Lion) */}
                  {OPTIMIZER_CONFIGS[optimizer]?.defaults.epsilon !== undefined && (
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Epsilon</label>
                      <input
                        type="text"
                        value={optimizerEpsilon}
                        onChange={(e) => setLocalEpsilonText(e.target.value)}
                        onBlur={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) updateParam("optimizer_epsilon", v); }}
                        className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  )}

                  {/* Weight Decay (all optimizers) */}
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Weight Decay</label>
                    <input
                      type="text"
                      value={optimizerWeightDecay}
                      onChange={(e) => setLocalWeightDecayText(e.target.value)}
                      onBlur={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) updateParam("optimizer_weight_decay", v); }}
                      className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs focus:outline-none focus:border-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Component-Specific Settings */}
          <div className="pt-3 mt-3 border-t border-gray-700">
            <h4 className="text-xs font-medium text-gray-400 mb-2">Component-Specific Learning Rates</h4>

            {/* Train toggles in 2 columns */}
            <div className="grid grid-cols-2 gap-3 mb-2">
              {/* Train U-Net */}
              <div className="flex items-center space-x-2" title={requiredValue("train_unet")?.reason}>
                <input
                  type="checkbox"
                  id="train-unet"
                  checked={trainUnet}
                  onChange={(e) => updateParam("train_unet", e.target.checked)}
                  disabled={!!requiredValue("train_unet")}
                  className="w-4 h-4 disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <label htmlFor="train-unet" className="text-xs text-gray-300 cursor-pointer">
                  Train U-Net
                </label>
              </div>

              {/* Train Text Encoder */}
              <div>
                <div className="flex items-center space-x-2" title={textEncoderTrainingUnsupported ?? textEncoderTrainingAdvisory?.reason}>
                  <input
                    type="checkbox"
                    id="train-text-encoder"
                    checked={trainTextEncoder && !textEncoderTrainingUnsupported}
                    onChange={(e) => updateParam("train_text_encoder", e.target.checked)}
                    disabled={!!textEncoderTrainingUnsupported}
                    className="w-4 h-4 disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <label htmlFor="train-text-encoder" className={`text-xs cursor-pointer ${textEncoderTrainingUnsupported ? 'text-gray-500' : 'text-gray-300'}`}>
                    Train Text Encoder {textEncoderTrainingUnsupported && '(not supported for this model)'}
                    {!textEncoderTrainingUnsupported && textEncoderTrainingAdvisory?.level === "high_memory" && ' (high memory)'}
                    {isMiniT2IModel(baseModelPath) && '(FLAN-T5)'}
                  </label>
                </div>
                {/* Advisory, not a refusal: the run is accepted either way. */}
                {!textEncoderTrainingUnsupported && textEncoderTrainingAdvisory && (
                  <p className="text-xs text-amber-400 mt-1">{textEncoderTrainingAdvisory.reason}</p>
                )}
              </div>

              {/* Train Image Encoder - DEUS support removed */}
            </div>

            {/* U-Net Learning Rate */}
            {trainUnet && (
              <div className="mb-3">
                <label className="block text-xs text-gray-400 mb-1">
                  U-Net LR <span className="text-xs text-gray-500">(empty = use base LR)</span>
                </label>
                <input
                  type="text"
                  value={unetLr}
                  onChange={(e) => setLocalUnetLrText(e.target.value)}
                  onBlur={(e) => { const v = parseFloat(e.target.value); updateParam("unet_lr", (isNaN(v) || v < 0) ? null : v); }}
                  placeholder={`Default: ${learningRate} (e.g., 1e-4)`}
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
              </div>
            )}

            {/* Text Encoder Learning Rates */}
            {trainTextEncoder && (
              <div className="space-y-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    Text Encoder LR <span className="text-xs text-gray-500">(base, empty = use base LR)</span>
                  </label>
                  <input
                    type="text"
                    value={textEncoderLr}
                    onChange={(e) => setLocalTextEncoderLrText(e.target.value)}
                    onBlur={(e) => { const v = parseFloat(e.target.value); updateParam("text_encoder_lr", (isNaN(v) || v < 0) ? null : v); }}
                    placeholder={`Default: ${learningRate} (e.g., 1e-5)`}
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  />
                </div>

                {/* SDXL-specific TE1/TE2 in 2 columns */}
                <div className="pl-3 space-y-2 border-l-2 border-gray-700">
                  <p className="text-xs text-gray-500">SDXL: Individual TEs (optional)</p>

                  <div className="grid grid-cols-2 gap-3">
                    {/* TE1 LR (CLIP-L) */}
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">
                        TE1 LR <span className="text-xs text-gray-500">(CLIP-L)</span>
                      </label>
                      <input
                        type="text"
                        value={textEncoder1Lr}
                        onChange={(e) => setLocalTextEncoder1LrText(e.target.value)}
                        onBlur={(e) => { const v = parseFloat(e.target.value); updateParam("text_encoder_1_lr", (isNaN(v) || v < 0) ? null : v); }}
                        placeholder={`Default: ${textEncoderLr || learningRate}`}
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                      />
                    </div>

                    {/* TE2 LR (CLIP-G) */}
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">
                        TE2 LR <span className="text-xs text-gray-500">(CLIP-G)</span>
                      </label>
                      <input
                        type="text"
                        value={textEncoder2Lr}
                        onChange={(e) => setLocalTextEncoder2LrText(e.target.value)}
                        onBlur={(e) => { const v = parseFloat(e.target.value); updateParam("text_encoder_2_lr", (isNaN(v) || v < 0) ? null : v); }}
                        placeholder={`Default: ${textEncoderLr || learningRate}`}
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Vision Encoder LR — shown only when VE is selected on SD/SDXL */}
            {visionEncoderPath && isSDOrSDXLModel(baseModelPath) && (
              <div className="mt-3 pt-3 border-t border-gray-700 space-y-2">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="train-vision-encoder"
                    checked={trainVisionEncoder}
                    onChange={(e) => updateParam("train_vision_encoder", e.target.checked)}
                    className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  />
                  <label htmlFor="train-vision-encoder" className="text-xs text-gray-300 cursor-pointer">
                    Train Vision Encoder
                  </label>
                </div>
                {trainVisionEncoder && (
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      VE LR <span className="text-xs text-gray-500">(empty = use text encoder LR)</span>
                    </label>
                    <input
                      type="text"
                      value={visionEncoderLr}
                      onChange={(e) => setLocalVisionEncoderLrText(e.target.value)}
                      onBlur={(e) => { const v = parseFloat(e.target.value); updateParam("vision_encoder_lr", (isNaN(v) || v < 0) ? null : v); }}
                      placeholder={`Default: ${textEncoderLr || learningRate}`}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                )}
                {trainTextEncoder && (
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="gradient-routing-ve"
                      checked={gradientRoutingVE}
                      onChange={(e) => updateParam("gradient_routing_ve", e.target.checked)}
                      className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                    />
                    <label htmlFor="gradient-routing-ve" className="text-xs text-gray-300 cursor-pointer">
                      Block text-encoder gradients on reference batches
                    </label>
                  </div>
                )}
              </div>
            )}

            {/* Image Encoder Learning Rate - DEUS support removed */}
          </div>
        </div>

        {/* Precision Settings (VRAM Optimization) */}
        <div className="break-inside-avoid border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Precision Settings (VRAM Optimization)</h3>

          <div className="grid grid-cols-2 gap-3">
            {/* Weight dtype */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">Weight dtype</label>
              <select
                value={weightDtype}
                onChange={(e) => updateParam("weight_dtype", e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="fp32">FP32 (推奨)</option>
                <option value="bf16">BF16</option>
                <option value="fp16">FP16 (非推奨)</option>
                <option value="fp8_e4m3fn">FP8 E4M3FN (非推奨)</option>
                <option value="fp8_e5m2">FP8 E5M2 (非推奨)</option>
              </select>
            </div>

            {/* Training/Activation dtype */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">Training dtype</label>
              <select
                value={trainingDtype}
                onChange={(e) => updateParam("training_dtype", e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="fp32">FP32</option>
                <option value="fp16">FP16</option>
                <option value="bf16">BF16</option>
                <option value="fp8_e4m3fn">FP8 E4M3FN (動作保証対象外)</option>
                <option value="fp8_e5m2">FP8 E5M2 (動作保証対象外)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            {/* Output dtype */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">Output dtype</label>
              <select
                value={outputDtype}
                onChange={(e) => updateParam("output_dtype", e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="fp32">FP32</option>
                <option value="fp16">FP16</option>
                <option value="bf16">BF16</option>
              </select>
            </div>

            {/* VAE dtype */}
            {!vaeUnsupported && (
            <div>
              <label className="block text-xs text-gray-400 mb-1">VAE dtype</label>
              <select
                value={vaeDtype}
                onChange={(e) => updateParam("vae_dtype", e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="fp32">FP32 (推奨)</option>
                <option value="fp16">FP16 (SDXL madebyollin VAEのみ許容)</option>
                <option value="bf16">BF16 (非推奨)</option>
                <option value="fp8_e4m3fn">FP8 E4M3FN (動作保証対象外)</option>
                <option value="fp8_e5m2">FP8 E5M2 (動作保証対象外)</option>
              </select>
            </div>
            )}
          </div>

          <div className="space-y-2">
            {/* Mixed Precision */}
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="mixed-precision"
                checked={mixedPrecision}
                onChange={(e) => updateParam("mixed_precision", e.target.checked)}
                className="w-4 h-4"
              />
              <label htmlFor="mixed-precision" className="text-xs text-gray-300 cursor-pointer">
                Mixed Precision (Autocast)
              </label>
            </div>

            {/* Bundle VAE (full-parameter save only) */}
            {!vaeUnsupported && (
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="bundle-vae"
                checked={!!params.bundle_vae}
                onChange={(e) => updateParam("bundle_vae", e.target.checked)}
                className="w-4 h-4"
              />
              <label htmlFor="bundle-vae" className="text-xs text-gray-300 cursor-pointer">
                Bundle VAE weights into the checkpoint
              </label>
            </div>
            )}

            {/* Attention Backend */}
            <div className="space-y-1">
              <label htmlFor="attention-backend" className="block text-xs text-gray-300">
                Attention Backend
              </label>
              <select
                id="attention-backend"
                value={attentionBackend}
                onChange={(e) => {
                  const backend = e.target.value;
                  updateParam("attention_backend", backend);
                  // R6: keep the deprecated compat mirror synchronized.
                  // use_flash_attention is true ONLY for the flash backend; tq and native map to false.
                  updateParam("use_flash_attention", backend === "flash");
                }}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
              >
                <option value="native">Native (PyTorch SDPA)</option>
                <option value="flash">Flash Attention</option>
                <option value="tq">TQ (Triton-Quantized)</option>
                <option value="sage" disabled title="Sage Attention is inference only (no backward pass)">
                  Sage (inference only)
                </option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                TQ (Triton-Quantized) applies to Z-Image, Lens, MiniT2I, and Anima training. Other architectures fall back to native.
              </p>
            </div>

            {/* Attention Impl */}
            <div className="space-y-1">
              <label htmlFor="attention-impl" className="block text-xs text-gray-300">
                Attention Impl
              </label>
              <select
                id="attention-impl"
                value={attentionImpl}
                onChange={(e) => updateParam("attention_impl", e.target.value)}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
              >
                <option value="conduit">Conduit (unified dispatch)</option>
                <option value="diffusers">Diffusers (legacy set_attention_backend)</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                Selects which registry runs the attention kernel (orthogonal to the backend above).
                "diffusers" reproduces the legacy set_attention_backend path. Affects SDXL/SD1.5 training;
                FLUX.2 is not yet migrated and ignores this setting.
              </p>
            </div>

            {/* Min-SNR Gamma */}
            <div className="space-y-1">
              <label htmlFor="min-snr-gamma" className="block text-xs text-gray-300">
                Min-SNR Gamma (loss weighting)
              </label>
              <input
                type="number"
                id="min-snr-gamma"
                value={minSnrGamma}
                onChange={(e) => updateParam("min_snr_gamma", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("min_snr_gamma", 5.0); }}
                step="any"
                min={0}
                max={20}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
              />
              <p className="text-xs text-gray-500">
                Default: 5.0. Set to 0 to disable. Prevents overfitting to high-noise timesteps.
              </p>
            </div>

            {/* CFG unconditional drop rate. Hidden entirely on an architecture
                the backend declares has no aligned null condition: there an
                explicit value -- 0 included -- is answered 400. */}
            {!cfgUncondDropUnsupported && (
              <div className="space-y-1">
                <label htmlFor="cfg-uncond-drop-rate" className="block text-xs text-gray-300">
                  CFG unconditional drop rate
                </label>
                <input
                  type="number"
                  id="cfg-uncond-drop-rate"
                  value={params.cfg_uncond_drop_rate ?? ""}
                  onChange={(e) => updateParam("cfg_uncond_drop_rate", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                  step="any"
                  min={0}
                  max={1}
                  placeholder={cfgUncondDropDefaultRate !== undefined ? String(cfgUncondDropDefaultRate) : ""}
                  className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                />
                <p className="text-xs text-gray-500">
                  Per-sample probability of training the item against the same
                  null condition this architecture&apos;s inference CFG uncond
                  branch builds. Leave empty for the architecture default
                  ({cfgUncondDropDefaultRate ?? "none"}); 0 disables it.
                  Different from the dataset&apos;s caption dropout, which
                  encodes an empty caption; a run that sets both is refused.
                  A nonzero rate is also refused together with reference
                  images: the null trained here is the text-only one, while a
                  reference-conditioned generation blends against the
                  reference-conditioned branch at the default img_cfg_scale 1.
                </p>
                <label className="flex items-center space-x-2 cursor-pointer mt-2">
                  <input
                    type="checkbox"
                    checked={params.cfg_uncond_drop_per_mnt ?? true}
                    onChange={(e) => updateParam("cfg_uncond_drop_per_mnt", e.target.checked)}
                    className="w-3.5 h-3.5"
                  />
                  <span className="text-xs text-gray-300">
                    Draw the CFG-null label independently for each multi-noise timestep
                  </span>
                </label>
                <p className="text-xs text-gray-500">
                  Only applies when multi_noise_timesteps &gt; 1. With this off, one draw
                  covers the whole multi-noise-timestep window for an item instead of each
                  iteration drawing its own.
                </p>
              </div>
            )}

            {/* Reconstruction Loss Weight */}
            <div>
              <label htmlFor="reconstruction-loss-weight" className="block text-xs font-medium text-gray-400 mb-1">
                Reconstruction Loss Weight
              </label>
              <input
                type="number"
                id="reconstruction-loss-weight"
                value={reconstructionLossWeight}
                onChange={(e) => updateParam("reconstruction_loss_weight", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("reconstruction_loss_weight", 0.0); }}
                step="any"
                min={0}
                max={1.0}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
              />
              <p className="text-xs text-gray-500">
                Default: 0.0 (prediction loss only). Dual loss: loss = (1-β)*pred_loss + β*recon_loss. Try 0.1 for faster learning in noisy timesteps.
              </p>
            </div>

            {/* Audio Loss Weight (MiniMax-H3 only) */}
            {isMiniMaxH3Model && (
              <div>
                <label htmlFor="audio-loss-weight" className="block text-xs font-medium text-gray-400 mb-1">
                  Audio Loss Weight
                </label>
                <input
                  type="number"
                  id="audio-loss-weight"
                  value={audioLossWeight}
                  onChange={(e) => updateParam("audio_loss_weight", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                  onBlur={(e) => {
                    const v = parseFloat(e.target.value);
                    if (e.target.value === '' || isNaN(v) || v < 0) updateParam("audio_loss_weight", 1.0);
                  }}
                  step="any"
                  min={0}
                  className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                />
                <p className="text-xs text-gray-500">
                  Default: 1.0. loss = video_mean + weight * audio_mean (each modality&apos;s
                  velocity MSE averaged over tokens, channels and samples before weighting).
                  0 trains on the video half only; the audio rows still ride along in the
                  packed sequence because they are part of its structure.
                </p>
              </div>
            )}
          </div>

          <p className="text-xs text-gray-500">
            Lower precision dtypes reduce VRAM usage. FP8 can save ~50% VRAM. Use FP32 output for best loss calculation accuracy. Flash Attention improves training speed and reduces memory usage. Min-SNR gamma reweights loss to balance learning across all timesteps. Reconstruction loss weight enables dual loss training (direct image quality optimization).
          </p>
        </div>

        {/* One interlocked setting, not two independent toggles — the backend
            refuses eviction + train_text_encoder without the split, and the
            split without either of them. Gated on the capability tables, so a
            method that stops advertising it loses the section. The arch check is
            belt-and-braces for the window before /schema/arch-capabilities
            resolves, when the table answers "supported" for everything. */}
        {isSenseNovaModel(baseModelPath) && !motEvictionUnsupported && (trainingMethod === "lora" || trainingMethod === "full_finetune") && (
          <div className="break-inside-avoid border border-gray-700 rounded p-4 space-y-2">
            <h3 className="text-sm font-medium text-gray-300">SenseNova Training Memory</h3>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="sensenova-mot-phase-eviction"
                checked={params.sensenova_mot_phase_eviction ?? false}
                onChange={(e) => {
                  updateParam("sensenova_mot_phase_eviction", e.target.checked);
                  if (!e.target.checked) updateParam("sensenova_four_phase_eviction", false);
                  if (!e.target.checked) updateParam("sensenova_mot_pageable_staging", false);
                  if (!e.target.checked) updateParam("sensenova_mot_overlap_transfer", false);
                }}
                className="w-4 h-4"
              />
              <label htmlFor="sensenova-mot-phase-eviction" className="text-xs text-gray-300 cursor-pointer">
                MoT Phase Eviction
              </label>
            </div>
            <p className="text-xs text-gray-500">
              Keeps only the active understanding or generation weight half on GPU. Opt-in. This architecture has no block swap, so the Block Swap controls are not offered for it.
            </p>

            {params.sensenova_mot_phase_eviction && (
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="sensenova-mot-pageable-staging"
                  checked={params.sensenova_mot_pageable_staging ?? false}
                  onChange={(e) => {
                    updateParam("sensenova_mot_pageable_staging", e.target.checked);
                    // Refused as a pair before the model loads, so this is not a
                    // preference: async copies need pinned host memory.
                    if (e.target.checked) updateParam("sensenova_mot_overlap_transfer", false);
                  }}
                  className="w-4 h-4"
                />
                <label htmlFor="sensenova-mot-pageable-staging" className="text-xs text-gray-300 cursor-pointer">
                  Pageable Host Staging
                </label>
              </div>
            )}
            {params.sensenova_mot_phase_eviction && (
              <p className="text-xs text-gray-500">
                Stages the evicted half to ordinary host memory instead of pinned. Trades the pinned pool&apos;s high-water, which stays allocated for the rest of the run, for host RAM the OS can reclaim, at an unmeasured transfer-time cost.
              </p>
            )}

            {params.sensenova_mot_phase_eviction && (
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="sensenova-mot-overlap-transfer"
                  checked={params.sensenova_mot_overlap_transfer ?? false}
                  onChange={(e) => {
                    updateParam("sensenova_mot_overlap_transfer", e.target.checked);
                    if (e.target.checked) updateParam("sensenova_mot_pageable_staging", false);
                  }}
                  className="w-4 h-4"
                />
                <label htmlFor="sensenova-mot-overlap-transfer" className="text-xs text-gray-300 cursor-pointer">
                  Overlapped Half Swap
                </label>
              </div>
            )}
            {params.sensenova_mot_phase_eviction && (
              <p className="text-xs text-gray-500">
                A half swap moves one module out and its twin in, pair by pair. This issues each pair&apos;s two legs on separate CUDA streams instead of back to back, so the two directions can use their own copy engines. The transfer term&apos;s arithmetic ceiling drops from the sum of the two directions to the larger of them; what a real run reaches is unmeasured. Holds at most four extra modules on GPU while a swap is in flight, and cannot be combined with Pageable Host Staging.
              </p>
            )}

            {trainingMethod === "full_finetune" && (
              <>
                <div className="flex items-center space-x-2" title={fourPhaseBlockedReason}>
                  <input
                    type="checkbox"
                    id="sensenova-four-phase-eviction"
                    checked={params.sensenova_four_phase_eviction ?? false}
                    onChange={(e) => updateParam("sensenova_four_phase_eviction", e.target.checked)}
                    disabled={!!fourPhaseBlockedReason}
                    className="w-4 h-4 disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <label htmlFor="sensenova-four-phase-eviction" className={`text-xs cursor-pointer ${fourPhaseBlockedReason ? 'text-gray-500' : 'text-gray-300'}`}>
                    Four-Phase Backward Split
                  </label>
                </div>
                {fourPhaseBlockedReason && (
                  <p className="text-xs text-gray-500">{fourPhaseBlockedReason}</p>
                )}

                {/* Only rendered on top of the split, which is what the backend
                    requires; the effect above clears it when the split goes. */}
                {params.sensenova_four_phase_eviction && !fourPhaseBlockedReason && (
                  <div className="pl-6 space-y-2 border-l border-gray-700">
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="sensenova-four-phase-shared-prefix"
                        checked={params.sensenova_four_phase_shared_prefix ?? false}
                        onChange={(e) => updateParam("sensenova_four_phase_shared_prefix", e.target.checked)}
                        className="w-4 h-4"
                      />
                      <label htmlFor="sensenova-four-phase-shared-prefix" className="text-xs text-gray-300 cursor-pointer">
                        Share One Prefix Across the Multi-Noise Window
                      </label>
                    </div>
                    <p className="text-xs text-gray-500">
                      Cuts the prefix once per multi-noise-timestep window instead of once per iteration, so the understanding half stays on CPU for the whole window and its backward runs once. Only has an effect at Multi-Noise Timesteps above 1.
                    </p>
                    {params.sensenova_four_phase_shared_prefix && (
                      <>
                        <p className="text-xs text-amber-400">
                          This changes what is trained. With N multi-noise timesteps the understanding half takes ONE update per window, computed at the weights the window started with, while the generation half takes N. Adafactor&apos;s step counter for that half advances once per window, so its beta2 schedule moves N times more slowly, and the single update uses the learning rate reached after all N iterations. The TE1 gradient-norm series becomes N-1 zeros and one spike per window, because the understanding half only receives a gradient on the step that closes the window. A batch skipped mid-window ends that batch and discards the window&apos;s accumulated understanding gradient while the generation updates it already applied stand; the count is reported on the training log and charted.
                        </p>
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Window Gradient Reduction</label>
                          <select
                            value={params.sensenova_four_phase_grad_reduction ?? "sum"}
                            onChange={(e) => updateParam("sensenova_four_phase_grad_reduction", e.target.value)}
                            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200"
                          >
                            <option value="sum">Sum (gradient of the window&apos;s summed loss)</option>
                            <option value="mean">Mean (gradient of the window&apos;s averaged loss)</option>
                          </select>
                          <p className="text-xs text-gray-500 mt-1">
                            Sum is what the accumulation itself produces. Mean divides by the number of iterations in the window, giving an understanding-side update N times smaller. Neither matches the generation half, which takes N separate updates from N separate gradients.
                          </p>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </>
            )}
            {/* Outside the full-fine-tune block: the pair is refused under LoRA
                too, and there the split is not the way out. */}
            {evictionPairRefusal && (
              <p className="text-xs text-red-400">{evictionPairRefusal}</p>
            )}
            {motEvictionBranchRefusal && (
              <p className="text-xs text-red-400">{motEvictionBranchRefusal}</p>
            )}
            {motEvictionAdvisory && (
              <p className="text-xs text-amber-400">{motEvictionAdvisory.reason}</p>
            )}
          </div>
        )}

        {isSenseNovaModel(baseModelPath) && trainingMethod === "full_finetune" && (
          <div className="break-inside-avoid border border-gray-700 rounded p-4 space-y-2">
            <h3 className="text-sm font-medium text-gray-300">SenseNova Checkpoint Format</h3>
            <select
              id="sensenova-full-finetune-save-format"
              value={params.sensenova_full_finetune_save_format ?? "mixed"}
              onChange={(e) => updateParam("sensenova_full_finetune_save_format", e.target.value)}
              className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
            >
              <option value="mixed">Mixed (trained half BF16, untrained half INT8)</option>
              <option value="bf16">BF16 (both halves)</option>
              <option value="int8">INT8 (trained half requantized)</option>
            </select>
            <p className="text-xs text-gray-500">
              Measured on the distributed checkpoint with the generation half trained:
              mixed 25.12 GiB on disk / 25.43 GiB peak VRAM at inference; BF16 32.66 GiB / 32.99 GiB.
              On an untrained model both reproduce the INT8 base bitwise.
              INT8 is 17.58 GiB (inference VRAM not measured) and matches the distributed layout,
              so it is the only one of the three that can be selected again as a training base.
            </p>
            <p className="text-xs text-gray-500">
              INT8 discards weight updates below half a grid step. Measured over all 294 generation
              Linears (8,103,395,328 elements, element-weighted mean grid step 2.70e-3) with a
              synthetic isotropic Gaussian update applied in FP32, not a real optimizer trajectory:
              at update std 1e-4, 99.94% of elements do not move and 0.55% of the update direction
              survives; at 3.16e-4, 95.57% do not move; at 1e-3, 52.09%; at 1e-2, 1.81% and 99.96%
              of the direction survives. In a run the trained half is BF16, whose ULP at unit scale
              is the same order as half the mean grid step, so an unmeasured share of the
              immobility at the smallest update sizes belongs to BF16 storage rather than to this
              requantization. Requantizing an untrained half gives RMS relative error 6.58e-4,
              max 7.81e-3. Training both halves leaves no INT8 half, so mixed writes the BF16 file.
            </p>
          </div>
        )}

        {isSenseNovaModel(baseModelPath) && !fmModulesUnsupported && trainingMethod === "full_finetune" && (
          <div className="break-inside-avoid border border-gray-700 rounded p-4 space-y-2">
            <h3 className="text-sm font-medium text-gray-300">SenseNova Trained Scope</h3>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="sensenova-train-fm-modules"
                checked={params.sensenova_train_fm_modules ?? false}
                onChange={(e) => updateParam("sensenova_train_fm_modules", e.target.checked)}
                className="w-4 h-4"
              />
              <label htmlFor="sensenova-train-fm-modules" className="text-xs text-gray-300 cursor-pointer">
                Train Flow-Matching Modules (fm_modules)
              </label>
            </div>
            <p className="text-xs text-gray-500">
              A full fine-tune trains the 294 decoder Linears per MoT half, which is the set the
              INT8 load dequantizes. fm_modules is not quantized, so it is not in that set: the
              generation ViT&apos;s patch and dense embeddings, the timestep and noise-scale
              embedders and the two fm_head convolutions — 16 tensors, 63,117,504 parameters
              (120.4 MiB in BF16) — stay frozen. Measured across two run checkpoints 4,960 steps
              apart, every one of them is byte-identical while the generation decoder moved
              3.09e-3 relative.
              Enabling this adds them to the generation parameter group at the U-Net learning rate.
              Every save format already writes them, so an update is kept.
              Changing this setting on a resume changes the generation group&apos;s parameter count,
              so the saved optimizer state cannot be reloaded and momentum/variance restart from
              zero for every trained parameter, not just the new ones.
              Cost is not measured. Enabling this makes the training step build an autograd graph
              over the generation ViT and the timestep/noise-scale embedders, which it did not
              build before — with these frozen that stage runs under no_grad and builds no graph
              at all — so expect extra activation memory when the option is on.
            </p>
            {fmModulesInertReason && (
              <p className="text-xs text-amber-400">{fmModulesInertReason}</p>
            )}
          </div>
        )}

        {/* Block Swap Settings (VRAM Optimization) */}
        <div className="break-inside-avoid border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Block Swap (Training VRAM Optimization)</h3>

          {blockSwapUnsupported && (
            <p className="text-xs text-yellow-500">
              Block Swap is not available for this base model: {blockSwapUnsupported}
            </p>
          )}

          <div className="space-y-3">
            {/* Blocks to Swap */}
            {!blockSwapUnsupported && (
            <div>
              <label htmlFor="blocks-to-swap" className="block text-xs text-gray-300 mb-1">
                Blocks to Swap (0 to disable)
              </label>
              <input
                type="number"
                id="blocks-to-swap"
                value={blocksToSwap}
                onChange={(e) => updateParam("blocks_to_swap", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("blocks_to_swap", 0); }}
                min={0}
                step={1}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
              />
              <p className="text-xs text-gray-500 mt-1">
                Number of transformer blocks to swap between GPU and CPU during training. Higher values reduce VRAM usage but may slow training. Default: 0 (disabled). Recommended: 10-20 for large models.
              </p>
              {blocksToSwap > 0 && (
                <p className="text-xs text-blue-400 mt-1">
                  Estimated VRAM saving: ~{Math.round((blocksToSwap / 30) * 100)}% of transformer parameters
                </p>
              )}
            </div>
            )}

            {/* Use Pinned Memory */}
            {!blockSwapUnsupported && blocksToSwap > 0 && (
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="use-pinned-memory"
                  checked={usePinnedMemory}
                  onChange={(e) => updateParam("use_pinned_memory", e.target.checked)}
                  className="w-4 h-4"
                />
                <label htmlFor="use-pinned-memory" className="text-xs text-gray-300 cursor-pointer">
                  Use Pinned Memory (faster CPU-GPU transfer)
                </label>
              </div>
            )}

            {/* H2D-only block swap (FLUX.2 LoRA training) */}
            {!blockSwapUnsupported && blocksToSwap > 0 && (
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="block-swap-h2d-only"
                  checked={params.block_swap_h2d_only ?? false}
                  onChange={(e) => updateParam("block_swap_h2d_only", e.target.checked)}
                  className="w-4 h-4"
                />
                <label htmlFor="block-swap-h2d-only" className="text-xs text-gray-300 cursor-pointer">
                  H2D-only (FLUX.2 LoRA: no device-to-host of frozen base; requires gradient checkpointing)
                </label>
              </div>
            )}

            {/* H2D-only ring size */}
            {!blockSwapUnsupported && blocksToSwap > 0 && (params.block_swap_h2d_only ?? false) && (
              <div>
                <label className="block text-xs text-gray-400 mb-1">Ring Size (GPU weight buffer slots)</label>
                <input
                  type="number"
                  min={1}
                  max={4}
                  value={params.block_swap_ring_size ?? 2}
                  onChange={(e) => updateParam("block_swap_ring_size", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))}
                  onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("block_swap_ring_size", 2); }}
                  className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200"
                />
              </div>
            )}

            {/* Per-bucket activation offload dispatcher */}
            <div className="pt-2 border-t border-gray-700">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="activation-dispatch-enable"
                  checked={activationDispatchEnable}
                  onChange={(e) => updateParam("activation_dispatch_enable", e.target.checked)}
                  className="w-4 h-4"
                />
                <label htmlFor="activation-dispatch-enable" className="text-xs text-gray-300 cursor-pointer">
                  Per-Bucket Activation Offload
                </label>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Predicts the training peak per resolution bucket before the forward pass and offloads
                saved activations to CPU only on buckets that would exceed the VRAM budget. Proactive
                (no OOM detection). Off by default.
              </p>
              {activationDispatchEnable && (
                <div className="mt-2">
                  <label htmlFor="activation-dispatch-margin" className="block text-xs text-gray-300 mb-1">
                    VRAM Safety Margin (GB)
                  </label>
                  <input
                    type="number"
                    id="activation-dispatch-margin"
                    value={activationDispatchMarginGb}
                    onChange={(e) => updateParam("activation_dispatch_margin_gb", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                    onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("activation_dispatch_margin_gb", 1.0); }}
                    min={0}
                    step="any"
                    className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Headroom kept free when deciding offload, to avoid driver spill.
                  </p>
                </div>
              )}
            </div>

            {/* Anima-only Phase D memory-optimisation toggles */}
            {isAnimaModel(baseModelPath) && (
              <>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="cpu-offload-checkpointing"
                    checked={!!params.cpu_offload_checkpointing}
                    onChange={(e) => updateParam("cpu_offload_checkpointing", e.target.checked)}
                    className="w-4 h-4"
                  />
                  <label htmlFor="cpu-offload-checkpointing" className="text-xs text-gray-300 cursor-pointer">
                    CPU-offload checkpointing (blocking)
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="async-cpu-offload-checkpointing"
                    checked={!!params.async_cpu_offload_checkpointing}
                    onChange={(e) => updateParam("async_cpu_offload_checkpointing", e.target.checked)}
                    className="w-4 h-4"
                  />
                  <label htmlFor="async-cpu-offload-checkpointing" className="text-xs text-gray-300 cursor-pointer">
                    Async CPU-offload checkpointing (non-blocking, faster)
                  </label>
                </div>
                {trainingMethod === "lora" && (
                  <div>
                    <label htmlFor="fp8-base-dtype" className="block text-xs text-gray-300 mb-1">
                      FP8 base weights (LoRA only)
                    </label>
                    <select
                      id="fp8-base-dtype"
                      value={params.fp8_base_dtype ?? ""}
                      onChange={(e) =>
                        updateParam("fp8_base_dtype", e.target.value === "" ? null : e.target.value)
                      }
                      className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                    >
                      <option value="">None (BF16 base)</option>
                      <option value="fp8_e4m3fn">FP8 E4M3</option>
                      <option value="fp8_e5m2">FP8 E5M2</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Quantise the frozen Anima DiT base to FP8 before LoRA wrap.
                    </p>
                  </div>
                )}

                {/* LoRA scope — which DiT module families get LoRA wraps. */}
                {trainingMethod === "lora" && (() => {
                  const scopeCsv: string = (params.anima_lora_scope ?? "attention,mlp,llm_adapter");
                  const scopeSet = new Set(scopeCsv.split(",").map((s: string) => s.trim()).filter(Boolean));
                  const toggle = (tok: string) => {
                    const next = new Set(scopeSet);
                    if (next.has(tok)) next.delete(tok); else next.add(tok);
                    // Keep a deterministic order so YAML diffs stay stable.
                    const ordered = ["attention", "mlp", "mod", "llm_adapter"].filter((t) => next.has(t));
                    updateParam("anima_lora_scope", ordered.join(","));
                    // Mirror the llm_adapter scope into train_llm_adapter so
                    // the two stay coherent (the trainer prefers the explicit
                    // flag when present).
                    if (tok === "llm_adapter") {
                      updateParam("train_llm_adapter", next.has("llm_adapter"));
                    }
                  };
                  return (
                    <div>
                      <label className="block text-xs text-gray-300 mb-1">
                        LoRA Scope (which module families get wrapped)
                      </label>
                      <div className="grid grid-cols-2 gap-1.5">
                        {[
                          ["attention", "Attention (Q/K/V/Out)"],
                          ["mlp", "MLP / FFN"],
                          ["mod", "AdaLN modulation"],
                          ["llm_adapter", "LLM Adapter"],
                        ].map(([tok, label]) => (
                          <label key={tok} className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={scopeSet.has(tok)}
                              onChange={() => toggle(tok)}
                              className="w-3.5 h-3.5"
                            />
                            <span>{label}</span>
                          </label>
                        ))}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        Default: attention + mlp + llm_adapter. AdaLN modulation is off
                        by default (typically small and easy to overfit).
                      </p>
                    </div>
                  );
                })()}

                {/* Train LLM Adapter — for Full FT only (LoRA covers this
                    via the scope multi-select above). */}
                {trainingMethod !== "lora" && (
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="train-llm-adapter"
                      checked={params.train_llm_adapter ?? true}
                      onChange={(e) => updateParam("train_llm_adapter", e.target.checked)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="train-llm-adapter" className="text-xs text-gray-300 cursor-pointer">
                      Train LLM Adapter (Qwen3→T5 projection)
                    </label>
                  </div>
                )}

                {/* Per-group LR multipliers — Full FT only. */}
                {trainingMethod !== "lora" && (
                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <label htmlFor="anima-attn-mlp-lr-factor" className="block text-xs text-gray-300 mb-1">
                        Attn+MLP LR ×
                      </label>
                      <input
                        type="number"
                        id="anima-attn-mlp-lr-factor"
                        value={params.anima_attn_mlp_lr_factor ?? 1.0}
                        onChange={(e) => updateParam("anima_attn_mlp_lr_factor", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("anima_attn_mlp_lr_factor", 1.0); }}
                        min={0}
                        step="any"
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                      />
                    </div>
                    <div>
                      <label htmlFor="anima-mod-lr-factor" className="block text-xs text-gray-300 mb-1">
                        AdaLN-mod LR ×
                      </label>
                      <input
                        type="number"
                        id="anima-mod-lr-factor"
                        value={params.anima_mod_lr_factor ?? 1.0}
                        onChange={(e) => updateParam("anima_mod_lr_factor", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("anima_mod_lr_factor", 1.0); }}
                        min={0}
                        step="any"
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                      />
                    </div>
                    <div>
                      <label htmlFor="anima-llm-adapter-lr-factor" className="block text-xs text-gray-300 mb-1">
                        LLM-Adapter LR ×
                      </label>
                      <input
                        type="number"
                        id="anima-llm-adapter-lr-factor"
                        value={params.anima_llm_adapter_lr_factor ?? 1.0}
                        onChange={(e) => updateParam("anima_llm_adapter_lr_factor", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("anima_llm_adapter_lr_factor", 1.0); }}
                        min={0}
                        step="any"
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                      />
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Lens-only options */}
            {isLensModel(baseModelPath) && (
              <>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="lens-cpu-offload-checkpointing"
                    checked={!!params.cpu_offload_checkpointing}
                    onChange={(e) => updateParam("cpu_offload_checkpointing", e.target.checked)}
                    className="w-4 h-4"
                  />
                  <label htmlFor="lens-cpu-offload-checkpointing" className="text-xs text-gray-300 cursor-pointer">
                    CPU-offload checkpointing (blocking)
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="lens-async-cpu-offload-checkpointing"
                    checked={!!params.async_cpu_offload_checkpointing}
                    onChange={(e) => updateParam("async_cpu_offload_checkpointing", e.target.checked)}
                    className="w-4 h-4"
                  />
                  <label htmlFor="lens-async-cpu-offload-checkpointing" className="text-xs text-gray-300 cursor-pointer">
                    Async CPU-offload checkpointing (non-blocking, faster)
                  </label>
                </div>
                {trainingMethod === "lora" && (
                  <div>
                    <label htmlFor="lens-fp8-base-dtype" className="block text-xs text-gray-300 mb-1">
                      FP8 base weights (LoRA only)
                    </label>
                    <select
                      id="lens-fp8-base-dtype"
                      value={params.fp8_base_dtype ?? ""}
                      onChange={(e) =>
                        updateParam("fp8_base_dtype", e.target.value === "" ? null : e.target.value)
                      }
                      className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                    >
                      <option value="">None (BF16 base)</option>
                      <option value="fp8_e4m3fn">FP8 E4M3</option>
                      <option value="fp8_e5m2">FP8 E5M2</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Quantise the frozen Lens DiT base to FP8 before LoRA wrap.
                    </p>
                  </div>
                )}

                {/* LoRA scope — which Lens DiT module families get LoRA wraps. */}
                {trainingMethod === "lora" && (() => {
                  const scopeCsv: string = (params.lens_lora_scope ?? "img_attn,txt_attn,img_mlp,txt_mlp");
                  const scopeSet = new Set(scopeCsv.split(",").map((s: string) => s.trim()).filter(Boolean));
                  const toggle = (tok: string) => {
                    const next = new Set(scopeSet);
                    if (next.has(tok)) next.delete(tok); else next.add(tok);
                    const ordered = ["img_attn", "txt_attn", "img_mlp", "txt_mlp", "mod"].filter((t) => next.has(t));
                    updateParam("lens_lora_scope", ordered.join(","));
                  };
                  return (
                    <div>
                      <label className="block text-xs text-gray-300 mb-1">
                        LoRA Scope (Lens DiT module families)
                      </label>
                      <div className="grid grid-cols-2 gap-1.5">
                        {[
                          ["img_attn", "Image Attention (QKV/Out)"],
                          ["txt_attn", "Text Attention (QKV/Out)"],
                          ["img_mlp", "Image MLP (GateMLP)"],
                          ["txt_mlp", "Text MLP (GateMLP)"],
                          ["mod", "AdaLN modulation"],
                        ].map(([tok, label]) => (
                          <label key={tok} className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={scopeSet.has(tok)}
                              onChange={() => toggle(tok)}
                              className="w-3.5 h-3.5"
                            />
                            <span>{label}</span>
                          </label>
                        ))}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        Default: img_attn + txt_attn + img_mlp + txt_mlp. AdaLN modulation is off by default.
                      </p>
                    </div>
                  );
                })()}

                {/* Per-stream LR multipliers — Full FT only. */}
                {trainingMethod !== "lora" && (
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label htmlFor="lens-img-lr-factor" className="block text-xs text-gray-300 mb-1">
                        Image stream LR ×
                      </label>
                      <input
                        type="number"
                        id="lens-img-lr-factor"
                        value={params.lens_img_lr_factor ?? 1.0}
                        onChange={(e) => updateParam("lens_img_lr_factor", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("lens_img_lr_factor", 1.0); }}
                        min={0}
                        step="any"
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                      />
                    </div>
                    <div>
                      <label htmlFor="lens-txt-lr-factor" className="block text-xs text-gray-300 mb-1">
                        Text stream LR ×
                      </label>
                      <input
                        type="number"
                        id="lens-txt-lr-factor"
                        value={params.lens_txt_lr_factor ?? 1.0}
                        onChange={(e) => updateParam("lens_txt_lr_factor", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("lens_txt_lr_factor", 1.0); }}
                        min={0}
                        step="any"
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                      />
                    </div>
                  </div>
                )}
              </>
            )}

            {/* LTX-2.3-only options */}
            {isLtx2Model(baseModelPath) && trainingMethod === "lora" && (
              <div>
                <label htmlFor="ltx2-fp8-base-dtype" className="block text-xs text-gray-300 mb-1">
                  FP8 base weights (LoRA only)
                </label>
                <select
                  id="ltx2-fp8-base-dtype"
                  value={params.fp8_base_dtype ?? ""}
                  onChange={(e) =>
                    updateParam("fp8_base_dtype", e.target.value === "" ? null : e.target.value)
                  }
                  className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                >
                  <option value="">None (BF16 base)</option>
                  <option value="fp8_e4m3fn">FP8 E4M3</option>
                  <option value="fp8_e5m2">FP8 E5M2</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  Quantise the frozen LTX-2.3 DiT base to FP8 before LoRA wrap.
                </p>
              </div>
            )}

            {/* Ideogram 4 (flow-matching DiT) LoRA options */}
            {isIdeogram4Model(baseModelPath) && trainingMethod === "lora" && (
              <>
                {(() => {
                  const scopeCsv: string = (params.ideogram4_lora_scope ?? "attn,mlp");
                  const scopeSet = new Set(scopeCsv.split(",").map((s: string) => s.trim()).filter(Boolean));
                  const toggle = (tok: string) => {
                    const next = new Set(scopeSet);
                    if (next.has(tok)) next.delete(tok); else next.add(tok);
                    const ordered = ["attn", "mlp", "mod"].filter((t) => next.has(t));
                    updateParam("ideogram4_lora_scope", ordered.join(","));
                  };
                  return (
                    <div>
                      <label className="block text-xs text-gray-300 mb-1">
                        LoRA Scope (Ideogram 4 DiT module families)
                      </label>
                      <div className="grid grid-cols-2 gap-1.5">
                        {[
                          ["attn", "Attention (Q/K/V/Out)"],
                          ["mlp", "MLP (SwiGLU)"],
                          ["mod", "AdaLN modulation"],
                        ].map(([tok, label]) => (
                          <label key={tok} className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={scopeSet.has(tok)}
                              onChange={() => toggle(tok)}
                              className="w-3.5 h-3.5"
                            />
                            <span>{label}</span>
                          </label>
                        ))}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        Default: attn + mlp. AdaLN modulation is off by default.
                      </p>
                    </div>
                  );
                })()}

                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="ideogram4-train-uncond"
                    checked={!!params.ideogram4_train_uncond}
                    onChange={(e) => updateParam("ideogram4_train_uncond", e.target.checked)}
                    className="w-4 h-4"
                  />
                  <label htmlFor="ideogram4-train-uncond" className="text-xs text-gray-300 cursor-pointer">
                    Also train unconditional transformer (asymmetric-CFG branch)
                  </label>
                </div>

                {params.ideogram4_train_uncond && (
                  <div>
                    <label htmlFor="ideogram4-uncond-loss-weight" className="block text-xs text-gray-300 mb-1">
                      Unconditional loss weight
                    </label>
                    <input
                      type="number"
                      id="ideogram4-uncond-loss-weight"
                      value={params.ideogram4_uncond_loss_weight ?? 1.0}
                      onChange={(e) => updateParam("ideogram4_uncond_loss_weight", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                      onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("ideogram4_uncond_loss_weight", 1.0); }}
                      min={0}
                      step="any"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                    />
                  </div>
                )}

                <div>
                  <label htmlFor="ideogram4-lr-factor" className="block text-xs text-gray-300 mb-1">
                    LoRA LR ×
                  </label>
                  <input
                    type="number"
                    id="ideogram4-lr-factor"
                    value={params.ideogram4_lr_factor ?? 1.0}
                    onChange={(e) => updateParam("ideogram4_lr_factor", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                    onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("ideogram4_lr_factor", 1.0); }}
                    min={0}
                    step="any"
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                  />
                </div>
              </>
            )}

            {/* Krea 2 (single-stream flow-matching MMDiT) LoRA options */}
            {isKrea2Model(baseModelPath) && trainingMethod === "lora" && (
              <>
                {(() => {
                  const scopeCsv: string = (params.krea2_lora_scope ?? "attn,mlp");
                  const scopeSet = new Set(scopeCsv.split(",").map((s: string) => s.trim()).filter(Boolean));
                  const toggle = (tok: string) => {
                    const next = new Set(scopeSet);
                    if (next.has(tok)) next.delete(tok); else next.add(tok);
                    const ordered = ["attn", "mlp", "text_fusion", "proj"].filter((t) => next.has(t));
                    updateParam("krea2_lora_scope", ordered.join(","));
                  };
                  return (
                    <div>
                      <label className="block text-xs text-gray-300 mb-1">
                        LoRA Scope (Krea 2 DiT module families)
                      </label>
                      <div className="grid grid-cols-2 gap-1.5">
                        {[
                          ["attn", "Attention (Q/K/V/Gate/Out)"],
                          ["mlp", "MLP (SwiGLU)"],
                          ["text_fusion", "Text fusion + projector"],
                          ["proj", "Input/output projections"],
                        ].map(([tok, label]) => (
                          <label key={tok} className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={scopeSet.has(tok)}
                              onChange={() => toggle(tok)}
                              className="w-3.5 h-3.5"
                            />
                            <span>{label}</span>
                          </label>
                        ))}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        Default: attn + mlp (28 main blocks). Qwen3-VL text encoder is frozen.
                      </p>
                    </div>
                  );
                })()}

                <div>
                  <label htmlFor="krea2-lr-factor" className="block text-xs text-gray-300 mb-1">
                    LoRA LR ×
                  </label>
                  <input
                    type="number"
                    id="krea2-lr-factor"
                    value={params.krea2_lr_factor ?? 1.0}
                    onChange={(e) => updateParam("krea2_lr_factor", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                    onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("krea2_lr_factor", 1.0); }}
                    min={0}
                    step="any"
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                  />
                </div>

                <div>
                  <label htmlFor="krea2-flow-shift" className="block text-xs text-gray-300 mb-1">
                    Discrete flow shift
                  </label>
                  <input
                    type="number"
                    id="krea2-flow-shift"
                    value={params.krea2_discrete_flow_shift ?? 2.5}
                    onChange={(e) => updateParam("krea2_discrete_flow_shift", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                    onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("krea2_discrete_flow_shift", 2.5); }}
                    min={1}
                    step="any"
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Timestep shift applied to sampled sigma. Set to 1 to disable.
                  </p>
                </div>
              </>
            )}

            {/* MiniT2I (pixel-space MM-JiT) LoRA options */}
            {isMiniT2IModel(baseModelPath) && trainingMethod === "lora" && (
              <>
                {(() => {
                  const scopeCsv: string = (params.minit2i_lora_scope ?? "attn,mlp,txt_embed");
                  const scopeSet = new Set(scopeCsv.split(",").map((s: string) => s.trim()).filter(Boolean));
                  const toggle = (tok: string) => {
                    const next = new Set(scopeSet);
                    if (next.has(tok)) next.delete(tok); else next.add(tok);
                    const ordered = ["attn", "mlp", "txt_embed"].filter((t) => next.has(t));
                    updateParam("minit2i_lora_scope", ordered.join(","));
                  };
                  return (
                    <div>
                      <label className="block text-xs text-gray-300 mb-1">
                        LoRA Scope (MiniT2I MM-JiT module families)
                      </label>
                      <div className="grid grid-cols-2 gap-1.5">
                        {[
                          ["attn", "Attention (QKV/Proj)"],
                          ["mlp", "MLP (SwiGLU w1/w2/w3)"],
                          ["txt_embed", "Text embedders"],
                        ].map(([tok, label]) => (
                          <label key={tok} className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={scopeSet.has(tok)}
                              onChange={() => toggle(tok)}
                              className="w-3.5 h-3.5"
                            />
                            <span>{label}</span>
                          </label>
                        ))}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        Default: attn + mlp + txt_embed.
                      </p>
                    </div>
                  );
                })()}

                <div>
                  <label htmlFor="minit2i-label-drop-rate" className="block text-xs text-gray-300 mb-1">
                    CFG label-drop rate
                  </label>
                  <input
                    type="number"
                    id="minit2i-label-drop-rate"
                    value={params.minit2i_label_drop_rate ?? ""}
                    onChange={(e) => updateParam("minit2i_label_drop_rate", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                    onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("minit2i_label_drop_rate", undefined as any); }}
                    min={0}
                    max={1}
                    step="any"
                    placeholder={minit2iLabelDropDefault !== undefined ? String(minit2iLabelDropDefault) : ""}
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Deprecated spelling of cfg_uncond_drop_rate. When both carry
                    a value only cfg_uncond_drop_rate is sent; sending both is
                    answered 400. Cleared = not supplied, which resolves to
                    {" "}{minit2iLabelDropDefault ?? "the backend default"}.
                  </p>
                </div>

                <div>
                  <label htmlFor="minit2i-lr-factor" className="block text-xs text-gray-300 mb-1">
                    LoRA LR ×
                  </label>
                  <input
                    type="number"
                    id="minit2i-lr-factor"
                    value={params.minit2i_lr_factor ?? 1.0}
                    onChange={(e) => updateParam("minit2i_lr_factor", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                    onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("minit2i_lr_factor", 1.0); }}
                    min={0}
                    step="any"
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                  />
                </div>

                {/* TE (FLAN-T5) LoRA scope — shown when Train Text Encoder is on */}
                {trainTextEncoder && (() => {
                  const teScopeCsv: string = (params.minit2i_te_lora_scope ?? "attn,ff");
                  const teScopeSet = new Set(teScopeCsv.split(",").map((s: string) => s.trim()).filter(Boolean));
                  const toggleTe = (tok: string) => {
                    const next = new Set(teScopeSet);
                    if (next.has(tok)) next.delete(tok); else next.add(tok);
                    const ordered = ["attn", "ff"].filter((t) => next.has(t));
                    updateParam("minit2i_te_lora_scope", ordered.join(","));
                  };
                  return (
                    <div>
                      <label className="block text-xs text-gray-300 mb-1">
                        TE LoRA Scope (FLAN-T5)
                      </label>
                      <div className="grid grid-cols-2 gap-1.5">
                        {[
                          ["attn", "Attention (q/k/v/o)"],
                          ["ff", "FeedForward (wi/wo)"],
                        ].map(([tok, label]) => (
                          <label key={tok} className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={teScopeSet.has(tok)}
                              onChange={() => toggleTe(tok)}
                              className="w-3.5 h-3.5"
                            />
                            <span>{label}</span>
                          </label>
                        ))}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        Applied to FLAN-T5 when Train Text Encoder is enabled. Default: attn + ff.
                      </p>
                    </div>
                  );
                })()}
              </>
            )}

            {/* Fused Optimizer Groups */}
            {!fusedGroupsUnsupported && blocksToSwap > 0 && (
              <div>
                <label htmlFor="num-optimizer-groups" className="block text-xs text-gray-300 mb-1">
                  Fused Optimizer Groups (0 to disable, recommended 4-10)
                </label>
                <input
                  type="number"
                  id="num-optimizer-groups"
                  value={numOptimizerGroups}
                  onChange={(e) => updateParam("num_optimizer_groups", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("num_optimizer_groups", 0); }}
                  min={0}
                  max={20}
                  step={1}
                  className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Divides parameters into N groups with separate optimizers. Updates each group immediately after gradients are computed. Works with ANY optimizer (AdamW, AdamW8bit, Lion8bit, etc.). Set to 0 to use Fused Backward Pass (Adafactor only).
                </p>
              </div>
            )}
          </div>

          <div className="text-xs text-gray-500 space-y-1">
            <p><strong>Block Swap:</strong> Offloads transformer blocks to CPU during training, reducing VRAM usage. Only active during forward and backward passes.</p>
            <p><strong>Pinned Memory:</strong> Uses CUDA pinned memory for faster transfer between CPU and GPU. Recommended if you have sufficient system RAM.</p>
            <p><strong>Fused Optimizer Groups:</strong> Enables ANY optimizer to work with Block Swap by dividing parameters into groups. Recommended: 4-10 groups for large models.</p>
            <p className="text-blue-500"><strong>Optimizer Compatibility:</strong> If "Fused Optimizer Groups" is 0, only Adafactor works with Block Swap (uses per-parameter updates). If Fused Optimizer Groups &gt; 0, ANY optimizer works (AdamW, AdamW8bit, Lion8bit, etc.).</p>
            <p className="text-yellow-500"><strong>Note:</strong> Only supported for Full Fine-tuning (not LoRA). Training speed may decrease with higher block swap counts. Requires PyTorch 2.1+ for fused backward/optimizer.</p>
          </div>
        </div>

        {/* Text Encoding Mode */}
        <div className="border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Text Encoding Mode</h3>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Encoding Mode</label>
            <select
              value={textEncodingMode}
              onChange={(e) => updateParam("text_encoding_mode", e.target.value)}
              disabled={!!requiredValue("text_encoding_mode")}
              title={requiredValue("text_encoding_mode")?.reason}
              className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-60"
            >
              <option value="swap_onthefly">Swap On-the-Fly (Recommended)</option>
              <option value="pre_encoded_cache">Pre-Encoded Cache (Disk)</option>
              <option value="onthefly_gpu">On-the-Fly GPU Encoding</option>
              <option value="cpu_prefetch">CPU Prefetch (background thread; TE pinned to CPU)</option>
            </select>
            <RequiredValueNote entry={requiredValue("text_encoding_mode")} />
          </div>

          {textEncodingMode === "cpu_prefetch" && (
            <div>
              <label htmlFor="text-encoding-prefetch-depth" className="block text-xs text-gray-400 mb-1">
                Prefetch Depth (batches ahead)
              </label>
              <input
                type="number"
                id="text-encoding-prefetch-depth"
                value={params.text_encoding_prefetch_depth ?? 4}
                onChange={(e) => updateParam("text_encoding_prefetch_depth", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))}
                onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("text_encoding_prefetch_depth", 4); }}
                min={1}
                max={32}
                step={1}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                How many batches ahead the worker encodes. Stall ratio is logged at epoch end.
              </p>
            </div>
          )}

          {textEncodingMode === "swap_onthefly" && (
            <div>
              <label htmlFor="text-encoding-swap-interval" className="block text-xs text-gray-400 mb-1">
                Swap Interval (steps)
              </label>
              <input
                type="number"
                id="text-encoding-swap-interval"
                value={textEncodingSwapInterval}
                onChange={(e) => updateParam("text_encoding_swap_interval", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("text_encoding_swap_interval", 256); }}
                min={1}
                max={1024}
                step={1}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
              />
              <p className="text-xs text-gray-500 mt-1">
                Memory usage: ~{Math.ceil(textEncodingSwapInterval * 2 / 1024)}MB DRAM (swap_interval × 2MB)
              </p>
            </div>
          )}

          <div className="text-xs text-gray-500 space-y-1">
            <p><strong>Swap On-the-Fly:</strong> Text Encoder swaps with main model (U-Net or Transformer) every N steps. Uses DRAM buffer. Recommended for large datasets.</p>
            <p><strong>Pre-Encoded Cache:</strong> Pre-encode all captions to disk cache. Not recommended if cache size exceeds disk capacity.</p>
            <p><strong>On-the-Fly GPU:</strong> Encode captions on GPU without cache. Slower, uses more VRAM.</p>
          </div>
        </div>

        {/* Reference Image Conditioning (hidden where the trainer has no such path) */}
        {!referenceImagesUnsupported &&
          (isFlux2Model(baseModelPath) || isSenseNovaModel(baseModelPath)) && (
        <div className="border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Reference Image Conditioning</h3>
          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              id="use-reference-images"
              checked={useReferenceImages}
              onChange={(e) => updateParam("use_reference_images", e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
            />
            <label htmlFor="use-reference-images" className="text-sm text-gray-400">
              Enable reference image conditioning ({isSenseNovaModel(baseModelPath) ? "SenseNova" : "FLUX.2"})
            </label>
          </div>
          <div className="text-xs text-gray-500 space-y-1">
            <p>
              {isSenseNovaModel(baseModelPath)
                ? "Reference images are encoded by the understanding vision tower and inserted as a prompt prefix."
                : "Reference images are encoded to latents and concatenated to the FLUX.2 image sequence."}
            </p>
            <p>Items without a reference image remain normally conditioned.</p>
            <p>Dataset items must have reference images configured (e.g., <code className="bg-gray-800 px-1 rounded">image_ref.png</code> suffix).</p>
          </div>
        </div>
        )}

        {/* SigLIP2 Vision Encoder — info only; selector is near Base Model, train/LR are in Component-Specific LR */}
        {isSDOrSDXLModel(baseModelPath) && (
        <div className="border border-gray-700 rounded p-4 space-y-2">
          <h3 className="text-sm font-medium text-gray-300 mb-2">SigLIP2 Vision Encoder</h3>
          <div className="text-xs text-gray-500 space-y-1">
            <p>参照画像を持つデータセットアイテムにのみ VE 条件付けが適用されます。参照画像なしのアイテムは通常のトレーニングが行われます。</p>
            <p>VE チェックポイントは <code className="bg-gray-800 px-1 rounded">*_vision_encoder_step_*.safetensors</code> として保存されます。</p>
            <p className="text-yellow-500/80">⚠️ SD 1.5 / SDXL モデルのみ対応。VE 選択はモデル選択欄の下にあります。</p>
          </div>
        </div>
        )}

        {/* Priority Training */}
        <div className="border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Priority Training</h3>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={priorityEnabled}
              onChange={(e) => setPriorityEnabled(e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded"
            />
            <span className="text-sm text-gray-300">Enable Priority Training</span>
          </label>

          {priorityEnabled && (
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400">Multiplier</label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={priorityMultiplier}
                  onChange={(e) => setPriorityMultiplier(Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-16 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                />
                <span className="text-xs text-gray-500">x per epoch</span>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">
                    Priority entries ({priorityText.split("\n").filter(l => l.trim()).length} items)
                  </label>
                  <button
                    type="button"
                    onClick={() => setPriorityExpanded(true)}
                    className="text-xs text-blue-400 hover:text-blue-300"
                  >
                    Expand
                  </button>
                </div>
                <TextareaWithTagSuggestions
                  value={priorityText}
                  onChange={(e) => setPriorityText(e.target.value)}
                  rows={Math.min(15, Math.max(5, priorityText.split("\n").length + 1))}
                  placeholder={"hatsune_miku\nkagamine_rin\nblue_hair, twintails\ncaption:dragon"}
                  tagSeparator="newline"
                />
              </div>

              <div className="text-xs text-gray-500 space-y-0.5">
                <p>One entry per line. Format:</p>
                <p><code className="bg-gray-800 px-1 rounded">tag_name</code> — single tag match</p>
                <p><code className="bg-gray-800 px-1 rounded">tag1, tag2</code> — AND condition (both required)</p>
                <p><code className="bg-gray-800 px-1 rounded">caption:text</code> — caption substring match</p>
              </div>
            </div>
          )}

          {!priorityEnabled && (
            <p className="text-xs text-gray-500">Focus training on specific tags/concepts at the beginning of each epoch.</p>
          )}
        </div>

        {/* Priority Training Expand Modal */}
        {priorityExpanded && (
          <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
            <div className="bg-gray-800 rounded-lg w-full max-w-3xl h-[80vh] flex flex-col">
              <div className="flex items-center justify-between p-3 border-b border-gray-700">
                <span className="text-sm font-medium text-gray-300">
                  Priority Entries ({priorityText.split("\n").filter(l => l.trim()).length} items)
                </span>
                <button
                  type="button"
                  onClick={() => setPriorityExpanded(false)}
                  className="text-gray-400 hover:text-white text-lg px-2"
                >
                  &times;
                </button>
              </div>
              <div className="flex-1 m-3 min-h-0">
                <TextareaWithTagSuggestions
                  value={priorityText}
                  onChange={(e) => setPriorityText(e.target.value)}
                  rows={30}
                  placeholder={"hatsune_miku\nkagamine_rin\nblue_hair, twintails\ncaption:dragon"}
                  tagSeparator="newline"
                />
              </div>
              <div className="p-3 border-t border-gray-700 flex justify-end">
                <button
                  type="button"
                  onClick={() => setPriorityExpanded(false)}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm text-white"
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Online Danbooru Augmentation */}
        <div className="border border-gray-700 rounded p-4 space-y-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={!!params.danbooru_aug_enable}
              onChange={(e) => updateParam("danbooru_aug_enable", e.target.checked)}
            />
            <h3 className="text-sm font-medium text-gray-300">Online Danbooru Augmentation</h3>
          </label>
          <p className="text-xs text-gray-500">
            Fetch extra training images from Danbooru during training and inject them as samples
            (interrupt-batch). No vocabulary expansion. Requires Latent Encoding Mode
            swap_onthefly or onthefly_gpu.
          </p>

          {params.danbooru_aug_enable && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Static queries (one per line)</label>
                <textarea
                  value={params.danbooru_aug_queries ?? ""}
                  onChange={(e) => updateParam("danbooru_aug_queries", e.target.value)}
                  rows={3}
                  placeholder={"1girl solo score:>=50\nhatsune_miku"}
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm font-mono focus:outline-none focus:border-blue-500"
                />
              </div>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={!!params.danbooru_aug_deficiency_enable}
                  onChange={(e) => updateParam("danbooru_aug_deficiency_enable", e.target.checked)}
                />
                <span className="text-xs text-gray-300">
                  Auto-collect under-represented tags (dataset frequency based)
                </span>
              </label>

              {params.danbooru_aug_deficiency_enable && (
                <div className="grid grid-cols-2 gap-3">
                  <NumField label="Deficiency min count" value={params.danbooru_aug_deficiency_min_count}
                    onChange={(v) => updateParam("danbooru_aug_deficiency_min_count", v)} step={1} />
                  <NumField label="Deficiency top-K" value={params.danbooru_aug_deficiency_top_k}
                    onChange={(v) => updateParam("danbooru_aug_deficiency_top_k", v)} step={1} />
                </div>
              )}

              <div>
                <label className="block text-xs text-gray-400 mb-1">Manual deficiency tags (comma or newline)</label>
                <textarea
                  value={params.danbooru_aug_deficiency_manual ?? ""}
                  onChange={(e) => updateParam("danbooru_aug_deficiency_manual", e.target.value)}
                  rows={2}
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm font-mono focus:outline-none focus:border-blue-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <NumField label="Weight: static" value={params.danbooru_aug_weight_static}
                  onChange={(v) => updateParam("danbooru_aug_weight_static", v)} step={0.1} />
                <NumField label="Weight: deficiency" value={params.danbooru_aug_weight_deficiency}
                  onChange={(v) => updateParam("danbooru_aug_weight_deficiency", v)} step={0.1} />
                <NumField label="Injection interval (batches)" value={params.danbooru_aug_injection_interval}
                  onChange={(v) => updateParam("danbooru_aug_injection_interval", v)} step={1} />
                <NumField label="Injection ratio (x batch)" value={params.danbooru_aug_injection_ratio}
                  onChange={(v) => updateParam("danbooru_aug_injection_ratio", v)} step={0.1} />
                <NumField label="Min score" value={params.danbooru_aug_min_score}
                  onChange={(v) => updateParam("danbooru_aug_min_score", v)} step={1} />
                <NumField label="Max posts / query" value={params.danbooru_aug_max_posts_per_query}
                  onChange={(v) => updateParam("danbooru_aug_max_posts_per_query", v)} step={1} />
                <NumField label="API interval (s)" value={params.danbooru_aug_api_interval}
                  onChange={(v) => updateParam("danbooru_aug_api_interval", v)} step={0.1} />
                <NumField label="DL speed (KB/s)" value={params.danbooru_aug_dl_speed_kbps}
                  onChange={(v) => updateParam("danbooru_aug_dl_speed_kbps", v)} step={1} />
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Buffer size (blank=auto)</label>
                  <input
                    type="number"
                    step={1}
                    value={params.danbooru_aug_buffer_size ?? ""}
                    onChange={(e) =>
                      updateParam("danbooru_aug_buffer_size", e.target.value === "" ? null : parseInt(e.target.value, 10))
                    }
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  />
                </div>
                <NumField label="Max caption tags (0=all)" value={params.danbooru_aug_max_caption_tags}
                  onChange={(v) => updateParam("danbooru_aug_max_caption_tags", v)} step={1} />
              </div>

              {/* Download-speed safety (throttle/ban avoidance) */}
              <div className="border-t border-gray-700 pt-3 mt-1 space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={!!params.danbooru_speed_check_enable}
                    onChange={(e) => updateParam("danbooru_speed_check_enable", e.target.checked)}
                  />
                  <span className="text-sm text-gray-300">Download-speed safety (throttle/ban avoidance)</span>
                </label>
                <p className="text-xs text-gray-500">
                  Pause collection when download speed stays degraded (Danbooru often throttles before a
                  hard ban). Robust to transient dips — a sustained slow streak is required. Live speed and
                  manual resume are in the metrics panel.
                </p>
                {params.danbooru_speed_check_enable && (
                  <div className="grid grid-cols-2 gap-2">
                    <NumField label="Degraded below (KB/s)" value={params.danbooru_speed_degraded_kbps}
                      onChange={(v) => updateParam("danbooru_speed_degraded_kbps", v)} step={1} />
                    <NumField label="Slow streak to trip" value={params.danbooru_speed_min_slow_streak}
                      onChange={(v) => updateParam("danbooru_speed_min_slow_streak", v)} step={1} />
                    <NumField label="Sustained at least (s)" value={params.danbooru_speed_min_slow_seconds}
                      onChange={(v) => updateParam("danbooru_speed_min_slow_seconds", v)} step={1} />
                    <NumField label="Cooldown (s)" value={params.danbooru_speed_cooldown_seconds}
                      onChange={(v) => updateParam("danbooru_speed_cooldown_seconds", v)} step={1} />
                  </div>
                )}
              </div>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={!!params.danbooru_aug_include_rating_tag}
                  onChange={(e) => updateParam("danbooru_aug_include_rating_tag", e.target.checked)}
                />
                <span className="text-xs text-gray-300">Include rating word in caption (general/sensitive/…)</span>
              </label>

              {/* Score-based quality tag */}
              <div className="border-t border-gray-700 pt-3 mt-1 space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={!!params.danbooru_quality_tag_enable}
                    onChange={(e) => updateParam("danbooru_quality_tag_enable", e.target.checked)}
                  />
                  <span className="text-xs text-gray-300">Add quality tag from Danbooru score</span>
                </label>
                {params.danbooru_quality_tag_enable && (
                  <div className="space-y-2 pl-6">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={!!params.danbooru_quality_tag_attach_negative}
                        onChange={(e) => updateParam("danbooru_quality_tag_attach_negative", e.target.checked)}
                      />
                      <span className="text-xs text-gray-300">Also attach low/worst-quality tiers</span>
                    </label>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">
                        Thresholds — one <code>&lt;min_score&gt; &lt;tag&gt;</code> per line (empty = Animagine XL 3.0 default)
                      </label>
                      <textarea
                        value={params.danbooru_quality_tag_thresholds ?? ""}
                        onChange={(e) => updateParam("danbooru_quality_tag_thresholds", e.target.value)}
                        rows={4}
                        placeholder={"151 masterpiece\n100 best quality\n75 high quality\n25 medium quality\n0 normal quality\n-5 low quality\n-1000000 worst quality"}
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-xs font-mono focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Caption tag shuffle / dropout (dedicated — independent of the
                  per-dataset caption processing). */}
              <div className="border-t border-gray-700 pt-3 mt-1 space-y-3">
                <p className="text-xs text-gray-400">
                  Caption tag shuffle / dropout for injected samples (separate from per-dataset caption processing)
                </p>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={!!params.danbooru_aug_shuffle_tags}
                    onChange={(e) => updateParam("danbooru_aug_shuffle_tags", e.target.checked)}
                  />
                  <span className="text-xs text-gray-300">Shuffle tags within each category (per-epoch)</span>
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <NumField label="Shuffle keep first N" value={params.danbooru_aug_shuffle_keep_first_n}
                    onChange={(v) => updateParam("danbooru_aug_shuffle_keep_first_n", v)} step={1} />
                  <NumField label="Keep tokens (token dropout)" value={params.danbooru_aug_keep_tokens}
                    onChange={(v) => updateParam("danbooru_aug_keep_tokens", v)} step={1} />
                  <NumField label="Tag dropout rate (0-1)" value={params.danbooru_aug_tag_dropout_rate}
                    onChange={(v) => updateParam("danbooru_aug_tag_dropout_rate", v)} step={0.05} />
                  <NumField label="Tag dropout keep first N" value={params.danbooru_aug_tag_dropout_keep_first_n}
                    onChange={(v) => updateParam("danbooru_aug_tag_dropout_keep_first_n", v)} step={1} />
                  <NumField label="Caption dropout rate (0-1)" value={params.danbooru_aug_caption_dropout_rate}
                    onChange={(v) => updateParam("danbooru_aug_caption_dropout_rate", v)} step={0.05} />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Latent Encoding Mode */}
        {latentEncodingAvailable && (
        <div className="border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Latent Encoding Mode (VAE)</h3>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Encoding Mode</label>
            <select
              value={latentEncodingMode}
              onChange={(e) => {
                const mode = e.target.value;
                updateParam("latent_encoding_mode", mode);
                if (mode !== "pre_encoded_cache") updateParam("force_recache", false);
              }}
              disabled={!!requiredValue("latent_encoding_mode")}
              title={requiredValue("latent_encoding_mode")?.reason}
              className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-60"
            >
              <option value="swap_onthefly">Swap On-the-Fly (Recommended)</option>
              <option value="pre_encoded_cache">Pre-Encoded Cache (Disk)</option>
              <option value="onthefly_gpu">On-the-Fly GPU Encoding</option>
            </select>
            <RequiredValueNote entry={requiredValue("latent_encoding_mode")} />
          </div>

          {latentEncodingMode === "swap_onthefly" && (
            <div>
              <label htmlFor="latent-encoding-swap-interval" className="block text-xs text-gray-400 mb-1">
                Swap Interval (steps)
              </label>
              <input
                type="number"
                id="latent-encoding-swap-interval"
                value={latentEncodingSwapInterval}
                onChange={(e) => updateParam("latent_encoding_swap_interval", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("latent_encoding_swap_interval", 256); }}
                min={1}
                max={1024}
                step={1}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
              />
              <p className="text-xs text-gray-500 mt-1">
                Memory usage: ~{Math.ceil(latentEncodingSwapInterval * 0.25)}MB DRAM (swap_interval × 256KB)
              </p>
            </div>
          )}

          {usesLatentDiskCache && (
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                id="force-recache"
                checked={forceRecache}
                onChange={(e) => updateParam("force_recache", e.target.checked)}
                className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
              />
              <span className="text-sm text-gray-400">Force regenerate latent cache</span>
            </label>
          )}

          <div className="text-xs text-gray-500 space-y-1">
            <p><strong>Swap On-the-Fly:</strong> VAE swaps with main model (U-Net or Transformer) every N steps. Uses DRAM buffer (~64MB for 256 steps). Recommended for VRAM efficiency.</p>
            <p><strong>Pre-Encoded Cache:</strong> Pre-encode all images to latents and cache to disk. Uses more disk space but no VRAM for VAE during training.</p>
            <p className="text-gray-400"><strong>Video datasets:</strong> a cached clip is addressed by its WINDOW, so this mode encodes and reuses ONE fixed (centred) window per video for the whole run — no temporal augmentation. The other two modes sample a fresh random window every time the clip is encoded.</p>
            <p><strong>On-the-Fly GPU:</strong> Encode images on GPU without cache. VAE stays on GPU, uses more VRAM.</p>
          </div>
        </div>
        )}

        {/* Advanced Settings */}
        <div className="break-inside-avoid border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Advanced Settings</h3>

          <div className="space-y-3 pb-3 border-b border-gray-700">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={params.gradient_checkpointing ?? true}
                onChange={(e) => updateParam("gradient_checkpointing", e.target.checked)}
                className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded"
              />
              <span className="text-sm text-gray-300">Gradient checkpointing</span>
            </label>
            <p className="text-xs text-gray-500">
              Recomputes activations during backward to reduce VRAM. Disabling it can be
              faster, but substantially increases peak memory.
            </p>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">torch.compile</label>
                <select
                  value={params.torch_compile ?? "off"}
                  onChange={(e) => updateParam("torch_compile", e.target.value)}
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                >
                  <option value="off">Off</option>
                  <option value="default">Default</option>
                  <option value="reduce-overhead">Reduce overhead</option>
                  <option value="max-autotune">Max autotune</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Dynamic shapes</label>
                <select
                  value={params.torch_compile_dynamic == null ? "auto" : String(params.torch_compile_dynamic)}
                  onChange={(e) => updateParam(
                    "torch_compile_dynamic",
                    e.target.value === "auto" ? null : e.target.value === "true"
                  )}
                  disabled={(params.torch_compile ?? "off") === "off"}
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm disabled:opacity-50"
                >
                  <option value="auto">Auto</option>
                  <option value="true">Enabled</option>
                  <option value="false">Disabled</option>
                </select>
              </div>
            </div>
            <p className="text-xs text-gray-500">
              Compilation is opt-in and may spend extra time compiling each resolution shape.
            </p>
          </div>

          {/* Save Checkpoint Every */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Save Checkpoint Every</label>
            <div className="flex items-center space-x-4 mb-2">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  checked={saveEveryUnit === "steps"}
                  onChange={() => updateParam("save_every_unit", "steps")}
                  className="text-blue-500 focus:ring-blue-500"
                />
                <span className="text-sm">Steps</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  checked={saveEveryUnit === "epochs"}
                  onChange={() => updateParam("save_every_unit", "epochs")}
                  className="text-blue-500 focus:ring-blue-500"
                />
                <span className="text-sm">Epochs</span>
              </label>
            </div>
            <input
              type="number"
              min="0"
              value={saveEvery}
              onChange={(e) => updateParam("save_every", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("save_every", 100); }}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
              placeholder={saveEveryUnit === "steps" ? "e.g., 100" : "e.g., 1"}
            />
            {saveEvery === 0 && (
              <p className="text-xs text-yellow-400 mt-1">
                0 = never save periodically. A checkpoint is still attempted if you
                stop the run or it fails, but that attempt can produce nothing —
                a failure that leaves the CUDA context dead, or one detected
                before any training happened, writes no checkpoint. With 0 there
                is no earlier checkpoint to fall back on.
              </p>
            )}
          </div>

          {/* Max Checkpoints to Keep */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Max Checkpoints to Keep</label>
            <input
              type="number"
              min="1"
              value={maxStepSavesToKeep ?? ""}
              onChange={(e) => updateParam("max_step_saves_to_keep", e.target.value === '' ? null : parseInt(e.target.value))}
              onBlur={(e) => { if (e.target.value !== '' && isNaN(parseInt(e.target.value))) updateParam("max_step_saves_to_keep", null); }}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
              placeholder="Method default (LoRA: 10, Full-FT: 3, ControlNet: 5)"
            />
            <p className="text-xs text-gray-500 mt-1">
              Number of most recent checkpoints to keep on disk (older ones are pruned). Leave empty to use the training method default.
              If the output volume cannot hold the next save, this is temporarily reduced (never below one complete checkpoint) and you are warned.
            </p>
          </div>

          {/* Max Optimizer States to Keep */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Max Optimizer States to Keep</label>
            <input
              type="number"
              min="0"
              value={maxOptimizerSavesToKeep}
              onChange={(e) => updateParam("max_optimizer_saves_to_keep", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))}
              onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("max_optimizer_saves_to_keep", DEFAULT_PARAMS.max_optimizer_saves_to_keep); }}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
              placeholder={String(DEFAULT_PARAMS.max_optimizer_saves_to_keep)}
            />
            <p className="text-xs text-gray-500 mt-1">
              Optimizer state files ({".pt"}) to keep, pruned independently of the checkpoints above. 0 = keep all.
              An optimizer state is about the size of the weights and only the newest one is used when resuming;
              with 1, falling back to an older checkpoint resumes without optimizer state.
            </p>
          </div>

          {/* Resume from Checkpoint */}
          <div>
            <label className="block text-sm text-gray-400 mb-1.5">Resume from Checkpoint</label>
            <select
              value={resumeFromCheckpoint || ""}
              onChange={(e) => updateParam("resume_from_checkpoint", e.target.value || null)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
            >
              <option value="">Start from Beginning</option>
              <option value="latest">Resume from Latest (Auto-detect)</option>
              {availableCheckpoints.map((ckpt) => (
                <option key={ckpt.filename} value={ckpt.filename}>
                  Step {ckpt.step} - {ckpt.filename}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Latest checkpoint will be auto-detected from the output directory
            </p>
          </div>
        </div>

        {/* Sample Generation */}
        {!trainingSamplesUnsupported && (
        <div className="break-inside-avoid border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Sample Generation (Optional)</h3>
          {selectedTrainingSampleNote && (
            <p className="text-xs text-amber-400">{selectedTrainingSampleNote}</p>
          )}

          {/* Sample Every */}
          <div>
            <label className="block text-sm text-gray-400 mb-1.5">Generate Sample Every (steps)</label>
            <input
              type="number"
              min="0"
              value={sampleEvery}
              onChange={(e) => updateParam("sample_every", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("sample_every", DEFAULT_PARAMS.sample_every); }}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
              placeholder="e.g., 100 (0 to disable)"
            />
            <p className="text-xs text-gray-500 mt-1">
              Set to 0 to disable sample generation during training
            </p>
          </div>

          {/* Sample Prompts */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="block text-sm text-gray-400">Sample Prompts</label>
              <button
                type="button"
                onClick={handleImportFromGeneration}
                className="px-2 py-1 bg-green-600 hover:bg-green-500 rounded text-xs transition-colors"
                title="Import prompt and settings from Txt2Img panel"
              >
                Import from Txt2Img
              </button>
            </div>

            {samplePrompts.map((prompt, index) => (
              <div key={index} className="mb-3 p-3 bg-gray-800 rounded border border-gray-700">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs text-gray-400">Sample {index + 1}</span>
                  <div className="flex space-x-2">
                    <button
                      type="button"
                      onClick={() => handleRandomPrompt(index)}
                      className="px-2 py-0.5 bg-purple-600 hover:bg-purple-500 rounded text-xs transition-colors"
                      title="Get random prompt from selected datasets"
                    >
                      Random
                    </button>
                    {samplePrompts.length > 1 && (
                      <button
                        type="button"
                        onClick={() => setSamplePrompts(samplePrompts.filter((_, i) => i !== index))}
                        className="text-red-400 hover:text-red-300 text-xs"
                      >
                        Remove
                      </button>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Positive Prompt</label>
                    <textarea
                      value={prompt.positive}
                      onChange={(e) => {
                        const updated = [...samplePrompts];
                        updated[index].positive = e.target.value;
                        setSamplePrompts(updated);
                      }}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                      rows={2}
                      placeholder="Enter positive prompt..."
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Negative Prompt</label>
                    <textarea
                      value={prompt.negative}
                      onChange={(e) => {
                        const updated = [...samplePrompts];
                        updated[index].negative = e.target.value;
                        setSamplePrompts(updated);
                      }}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                      rows={2}
                      placeholder="Enter negative prompt..."
                    />
                  </div>
                  {trainingMethod === "controlnet" && (
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">Condition Image</label>
                      <div
                        className={`relative border border-dashed rounded p-2 text-center transition-colors ${
                          conditionImagePreviews[index]
                            ? "border-green-600 bg-green-900/10"
                            : "border-gray-600 hover:border-blue-500 bg-gray-900/50"
                        }`}
                        onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                        onDrop={(e) => handleConditionImageDrop(index, e)}
                      >
                        <input
                          type="file"
                          accept="image/*"
                          className="hidden"
                          ref={(el) => { conditionImageInputRefs.current[index] = el; }}
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) handleConditionImageUpload(index, file);
                            e.target.value = "";
                          }}
                        />
                        {conditionImagePreviews[index] ? (
                          <div className="flex items-center space-x-2">
                            <img
                              src={conditionImagePreviews[index]}
                              alt="Condition"
                              className="h-16 w-16 object-cover rounded border border-gray-700"
                            />
                            <div className="flex-1 text-left">
                              <p className="text-xs text-green-400">Condition image set</p>
                              <p className="text-xs text-gray-500 truncate max-w-[200px]">
                                {prompt.condition_image_path?.replace("temp_img://", "") || ""}
                              </p>
                            </div>
                            <button
                              type="button"
                              onClick={() => handleConditionImageRemove(index)}
                              className="text-red-400 hover:text-red-300 text-xs px-2 py-1"
                            >
                              Remove
                            </button>
                          </div>
                        ) : (
                          <button
                            type="button"
                            onClick={() => conditionImageInputRefs.current[index]?.click()}
                            className="w-full py-2 text-xs text-gray-400 hover:text-gray-300"
                          >
                            Click to select or drag & drop condition image
                          </button>
                        )}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        If empty, the first reference image from the dataset is used.
                      </p>
                    </div>
                  )}
                  {trainingMethod !== "controlnet" &&
                    referenceConditioningEnabled && (
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">
                        Reference Image
                        <span className="text-gray-600 ml-1">
                          ({isFlux2Model(baseModelPath)
                            ? "Latent concat"
                            : isSenseNovaModel(baseModelPath)
                              ? "Understanding prompt prefix"
                              : "Vision Encoder"})
                        </span>
                      </label>
                      <div
                        className={`relative border border-dashed rounded p-2 text-center transition-colors ${
                          referenceImagePreviews[index]
                            ? "border-green-600 bg-green-900/10"
                            : "border-gray-600 hover:border-blue-500 bg-gray-900/50"
                        }`}
                        onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                        onDrop={(e) => handleReferenceImageDrop(index, e)}
                      >
                        <input
                          type="file"
                          accept="image/*"
                          className="hidden"
                          ref={(el) => { referenceImageInputRefs.current[index] = el; }}
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) handleReferenceImageUpload(index, file);
                            e.target.value = "";
                          }}
                        />
                        {referenceImagePreviews[index] ? (
                          <div className="flex items-center space-x-2">
                            <img
                              src={referenceImagePreviews[index]}
                              alt="Reference"
                              className="h-16 w-16 object-cover rounded border border-gray-700"
                            />
                            <div className="flex-1 text-left">
                              <p className="text-xs text-green-400">Reference image set</p>
                              <p className="text-xs text-gray-500 truncate max-w-[200px]">
                                {prompt.reference_image_path?.replace("temp_img://", "") || ""}
                              </p>
                            </div>
                            <button
                              type="button"
                              onClick={() => handleReferenceImageRemove(index)}
                              className="text-red-400 hover:text-red-300 text-xs px-2 py-1"
                            >
                              Remove
                            </button>
                          </div>
                        ) : (
                          <button
                            type="button"
                            onClick={() => referenceImageInputRefs.current[index]?.click()}
                            className="w-full py-2 text-xs text-gray-400 hover:text-gray-300"
                          >
                            Click to select or drag & drop reference image
                          </button>
                        )}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        If set, this image is used for reference conditioning during sample generation.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={() => setSamplePrompts([...samplePrompts, { positive: "", negative: "" }])}
              className="w-full px-3 py-2 bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded text-sm transition-colors"
            >
              + Add Sample Prompt
            </button>
          </div>

          {/* Sample Parameters */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-gray-400">Width</label>
                {Object.keys(referenceImagePreviews).length > 0 && (
                  <button
                    type="button"
                    onClick={applyRefImageSize}
                    className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                    title="Set width and height from reference image (floor to multiple of 8)"
                  >
                    From ref image
                  </button>
                )}
              </div>
              <input
                type="number"
                min="512"
                max="2048"
                step="8"
                value={sampleWidth}
                onChange={(e) => updateParam("sample_width", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("sample_width", DEFAULT_PARAMS.sample_width); }}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Height</label>
              <input
                type="number"
                min="512"
                max="2048"
                step="8"
                value={sampleHeight}
                onChange={(e) => updateParam("sample_height", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("sample_height", DEFAULT_PARAMS.sample_height); }}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Steps</label>
              <input
                type="number"
                min="1"
                max="150"
                value={sampleSteps}
                onChange={(e) => { sampleDefaultsExplicitlySetRef.current = true; updateParam("sample_steps", e.target.value === '' ? (undefined as any) : parseInt(e.target.value)); }} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("sample_steps", sampleStepsDefault); }}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">CFG Scale</label>
              <input
                type="number"
                min="1"
                max="30"
                step="any"
                value={sampleCfgScale}
                onChange={(e) => { sampleDefaultsExplicitlySetRef.current = true; updateParam("sample_cfg_scale", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value)); }} onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("sample_cfg_scale", sampleCfgScaleDefault); }}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
            {sampleSamplerSupported && <div>
              <label className="block text-xs text-gray-400 mb-1">Sampler</label>
              <select
                value={sampleSampler}
                onChange={(e) => updateParam("sample_sampler", e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                {samplers.map((sampler) => (
                  <option key={sampler.id} value={sampler.id}>
                    {sampler.name}
                  </option>
                ))}
              </select>
            </div>}
            {sampleScheduleSupported && <div>
              <label className="block text-xs text-gray-400 mb-1">Schedule Type</label>
              <select
                value={sampleScheduleType}
                onChange={(e) => updateParam("sample_schedule_type", e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                {scheduleTypes.map((scheduleType) => (
                  <option key={scheduleType.id} value={scheduleType.id}>
                    {scheduleType.name}
                  </option>
                ))}
              </select>
            </div>}
          </div>

          {/* SenseNova's preview controls belong to the preview, not to the
              memory options they used to sit among -- eleven cards away from
              the section they configure, which is where they were looked for
              and not found. */}
          {(
            sensenovaTimestepShiftSupported || sensenovaImgCfgSupported || sensenovaCfgNormSupported
          ) && (
            <details className="border border-gray-700 rounded p-3">
              <summary className="text-sm text-gray-300 cursor-pointer">SenseNova Preview Options</summary>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mt-3">
                {sensenovaTimestepShiftSupported && <div>
                  <label className="block text-xs text-gray-400 mb-1">Timestep Shift</label>
                  <input
                    type="number"
                    min="0"
                    step="any"
                    value={params.sensenova_sample_timestep_shift ?? ""}
                    onChange={(e) => updateParam("sensenova_sample_timestep_shift", e.target.value === "" ? undefined : parseFloat(e.target.value))}
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  />
                </div>}
                {sensenovaImgCfgSupported && <div>
                  <label className="block text-xs text-gray-400 mb-1">Image CFG Scale</label>
                  <input
                    type="number"
                    min="0"
                    step="any"
                    value={params.sensenova_sample_img_cfg_scale ?? ""}
                    onChange={(e) => updateParam("sensenova_sample_img_cfg_scale", e.target.value === "" ? undefined : parseFloat(e.target.value))}
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  />
                </div>}
                {sensenovaCfgNormSupported && <div>
                  <label className="block text-xs text-gray-400 mb-1">CFG Norm</label>
                  <select
                    value={params.sensenova_sample_cfg_norm ?? ""}
                    onChange={(e) => updateParam("sensenova_sample_cfg_norm", e.target.value as "none" | "global")}
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  >
                    <option value="global">Global</option>
                    <option value="none">None</option>
                  </select>
                </div>}
              </div>
              <p className="text-xs text-gray-500 mt-2">
                These settings affect only SenseNova training previews. Image CFG is used when the sample prompt includes a reference image.
              </p>
              <div className="flex items-center space-x-2 mt-2">
                <input
                  type="checkbox"
                  id="sensenova-sample-kv-cache-streaming"
                  checked={params.sensenova_sample_kv_cache_streaming ?? false}
                  onChange={(e) => updateParam("sensenova_sample_kv_cache_streaming", e.target.checked)}
                  className="w-4 h-4"
                />
                <label htmlFor="sensenova-sample-kv-cache-streaming" className="text-xs text-gray-300 cursor-pointer">
                  Stream Sample KV Cache
                </label>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Applies only to the in-training sample image, not to training steps. Streams each layer&apos;s prefix KV cache from pinned host memory through a 2-slot GPU ring instead of holding the full per-layer, per-branch KV cache resident during the sample&apos;s denoise loop. Independent of MoT Phase Eviction in the SenseNova Training Memory section. If the install fails, the sample runs with the full resident cache and a warning is logged.
              </p>
            </details>
          )}

          {sampleAdvancedCfgSupported && (
            <details className="border border-gray-700 rounded p-3">
              <summary className="text-sm text-gray-300 cursor-pointer">Advanced CFG, Dynamic Threshold &amp; NAG</summary>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                <label className="text-xs text-gray-400">CFG schedule
                  <select value={sampleCfgScheduleType} onChange={(e) => updateParam("sample_cfg_schedule_type", e.target.value)} className="mt-1 w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm">
                    {['constant', 'linear', 'cosine', 'exponential', 'quadratic'].map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                </label>
                {([
                  ['sample_cfg_schedule_min', 'CFG min'],
                  ['sample_cfg_schedule_max', 'CFG max (blank = scale)'],
                  ['sample_cfg_schedule_power', 'CFG power'],
                  ['sample_cfg_rescale_snr_alpha', 'SNR rescale alpha'],
                  ['sample_dynamic_threshold_percentile', 'Dynamic threshold %'],
                  ['sample_dynamic_threshold_mimic_scale', 'Mimic scale'],
                ] as const).map(([key, label]) => (
                  <label key={key} className="text-xs text-gray-400">{label}
                    <input type="number" step="any" value={params[key] ?? ''}
                      onChange={(e) => updateParam(key, e.target.value === '' ? (key === 'sample_cfg_schedule_max' ? null : undefined as any) : parseFloat(e.target.value))}
                      className="mt-1 w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm" />
                  </label>
                ))}
              </div>
              <label className="flex items-center gap-2 mt-4 text-sm text-gray-300">
                <input type="checkbox" checked={Boolean(params.sample_nag_enable)} onChange={(e) => updateParam("sample_nag_enable", e.target.checked)} />
                Enable NAG
              </label>
              {params.sample_nag_enable && <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-3">
                {([
                  ['sample_nag_scale', 'NAG scale'], ['sample_nag_tau', 'NAG tau'],
                  ['sample_nag_alpha', 'NAG alpha'], ['sample_nag_sigma_end', 'NAG sigma end'],
                ] as const).map(([key, label]) => (
                  <label key={key} className="text-xs text-gray-400">{label}
                    <input type="number" step="any" value={params[key] ?? ''} onChange={(e) => updateParam(key, e.target.value === '' ? undefined as any : parseFloat(e.target.value))}
                      className="mt-1 w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm" />
                  </label>
                ))}
                <label className="text-xs text-gray-400 md:col-span-5">NAG negative prompt (blank uses sample negative prompt)
                  <textarea value={params.sample_nag_negative_prompt ?? ''} onChange={(e) => updateParam("sample_nag_negative_prompt", e.target.value)} rows={2}
                    className="mt-1 w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm" />
                </label>
              </div>}
            </details>
          )}

          {/* Sample Seed */}
          <div>
            <label className="block text-sm text-gray-400 mb-1.5">Seed</label>
            <input
              type="number"
              value={sampleSeed}
              onChange={(e) => updateParam("sample_seed", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("sample_seed", DEFAULT_PARAMS.sample_seed); }}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
              placeholder="-1 for random"
            />
            <p className="text-xs text-gray-500 mt-1">
              Use -1 for random seed (different each time)
            </p>
          </div>
        </div>
        )}

        {/* Debug Options */}
        <div className="border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Debug Options</h3>

          {/* Rescan datasets before training */}
          <div className="space-y-1">
            <label htmlFor="rescan-before-training" className="text-sm text-gray-300">
              Rescan datasets before training
            </label>
            <select
              id="rescan-before-training"
              value={
                typeof params.rescan_before_training === "boolean"
                  ? (params.rescan_before_training ? "path" : "off")
                  : (params.rescan_before_training ?? "off")
              }
              onChange={(e) => updateParam("rescan_before_training", e.target.value as "off" | "path" | "smart" | "force")}
              className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="off">Off — skip pre-flight check</option>
              <option value="path">Path drift only — detect added / missing files</option>
              <option value="smart">Smart — path drift + caption mtime (catches in-place edits)</option>
              <option value="force">Force — always rescan, no drift detection</option>
            </select>
            <p className="text-xs text-gray-500">
              When the chosen mode detects drift (or in &quot;force&quot;), runs a full
              rescan and cleans up orphan latent cache.  Pre-flight walk adds
              ~1 directory-walk worth of time per dataset.
            </p>
          </div>

          {/* Debug Latents Toggle */}
          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              id="debug-latents"
              checked={debugLatents}
              onChange={(e) => updateParam("debug_latents", e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
            />
            <label htmlFor="debug-latents" className="text-sm text-gray-400">
              Save debug latents during training
            </label>
          </div>

          {/* Debug Latents Every (only shown if enabled) */}
          {debugLatents && (
            <div>
              <label className="block text-sm text-gray-400 mb-1.5">Save Debug Latents Every (steps)</label>
              <input
                type="number"
                min="0"
                value={debugLatentsEvery}
                onChange={(e) => updateParam("debug_latents_every", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("debug_latents_every", 50); }}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
                placeholder="e.g., 50"
              />
              <p className="text-xs text-gray-500 mt-1">
                Saves noisy latents, predicted latents, and timestep info to debug/ folder for debugging training issues
              </p>
            </div>
          )}
        </div>

        {/* Parameter Change Tracking */}
        <div className="border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Parameter Change Tracking</h3>

          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              id="param-tracking"
              checked={paramTracking}
              onChange={(e) => updateParam("param_tracking", e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
            />
            <label htmlFor="param-tracking" className="text-sm text-gray-400">
              Track per-component parameter change norms (Update Norm / Cumulative Drift)
            </label>
          </div>

          {paramTracking && (
            <div>
              <label className="block text-sm text-gray-400 mb-1.5">Tracking Interval (steps)</label>
              <input
                type="number"
                min="1"
                value={paramTrackingInterval}
                onChange={(e) => updateParam("param_tracking_interval", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))}
                onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("param_tracking_interval", 100); }}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
                placeholder="e.g., 100"
              />
              <p className="text-xs text-gray-500 mt-1">
                {"Computes ||θ_t - θ_{t-K}||_F (update norm) and ||θ_t - θ_0||_F / ||θ_0||_F (cumulative drift) per component on CPU every N steps"}
              </p>
            </div>
          )}
        </div>

        {/* Bucketing Options */}
        <div className="border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Training Resolution &amp; Aspect Ratio Bucketing</h3>

          {/* Enable Bucketing Toggle */}
          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              id="enable-bucketing"
              checked={enableBucketing}
              onChange={(e) => {
                const enabled = e.target.checked;
                updateParam("enable_bucketing", enabled);
                if (!enabled && baseResolutions.length > 1) {
                  updateParam("base_resolutions", [Math.max(...baseResolutions)]);
                }
              }}
              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
            />
            <label htmlFor="enable-bucketing" className="text-sm text-gray-400">
              Enable aspect ratio bucketing
            </label>
          </div>
          <p className="text-xs text-gray-500">
            Allows training on images with different aspect ratios by bucketing them into similar sizes
          </p>

          {/* Base resolutions also bound oversized images when bucketing is off. */}
          <>
              {/* Base Resolutions */}
              <div>
                <label className="block text-sm text-gray-400 mb-1.5">
                  {enableBucketing ? "Base Resolutions" : "Base Resolution"}
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    [256, 512, 768, 1024],
                    [1280, 1536, 1792, 2048],
                    [2304, 2560, 3072, 4096],
                  ].map((resGroup, groupIdx) => (
                    <div key={groupIdx} className="space-y-2">
                      {resGroup.map(res => (
                        <div key={res} className="flex items-center space-x-2">
                          <input
                            type={enableBucketing ? "checkbox" : "radio"}
                            name={enableBucketing ? undefined : "base-resolution"}
                            id={`res-${res}`}
                            checked={baseResolutions.includes(res)}
                            onChange={(e) => {
                              if (!enableBucketing) {
                                updateParam("base_resolutions", [res]);
                              } else if (e.target.checked) {
                                updateParam("base_resolutions", [...baseResolutions, res].sort((a, b) => a - b));
                              } else {
                                // Prevent unchecking the last resolution
                                if (baseResolutions.length > 1) {
                                  updateParam("base_resolutions", baseResolutions.filter(r => r !== res));
                                }
                              }
                            }}
                            disabled={enableBucketing && baseResolutions.length === 1 && baseResolutions.includes(res)}
                            className="w-4 h-4"
                          />
                          <label htmlFor={`res-${res}`} className="text-sm text-gray-300 cursor-pointer">
                            {res}
                          </label>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Selected: {baseResolutions.length > 0 ? baseResolutions.join(", ") : "None"}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {enableBucketing
                    ? "Each image is assigned to a resolution bucket according to the mode below."
                    : "Oversized images are fitted into this resolution area while smaller images keep their source size."}
                </p>
              </div>

              <div className="border-t border-gray-700 pt-3 space-y-3">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={params.res_curriculum_enable ?? false}
                    onChange={(e) => updateParam("res_curriculum_enable", e.target.checked)}
                    className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded"
                  />
                  <span className="text-sm text-gray-300">Lower-resolution warmup</span>
                </label>
                <p className="text-xs text-gray-500">
                  Starts below the selected base resolutions, then switches to them at an
                  epoch boundary. This reduces early-step attention cost.
                </p>
                {params.res_curriculum_enable && (
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Warmup steps</label>
                      <NumberInput
                        min={0} step={1}
                        value={params.res_curriculum_warmup_steps ?? 0}
                        defaultValue={0}
                        placeholder="e.g. 500"
                        onCommit={(v) => updateParam("res_curriculum_warmup_steps", v)}
                        className="w-full px-3 py-2 text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Linear scale</label>
                      <NumberInput
                        min={0.1} max={0.99} step="any" parse="float"
                        value={params.res_curriculum_warmup_scale ?? 0.5}
                        defaultValue={0.5}
                        placeholder="0.5"
                        onCommit={(v) => updateParam("res_curriculum_warmup_scale", v)}
                        className="w-full px-3 py-2 text-sm"
                      />
                    </div>
                  </div>
                )}
              </div>

            {/* Bucketing-specific settings */}
            {enableBucketing && (
              <>

              {/* Multi-Resolution Mode (only show if multiple resolutions) */}
              {baseResolutions.length > 1 && (
                <div>
                  <label className="block text-sm text-gray-400 mb-1.5">Multi-Resolution Mode</label>
                  <select
                    value={multiResolutionMode}
                    onChange={(e) => updateParam("multi_resolution_mode", e.target.value as "max" | "random")}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
                  >
                    <option value="max">Max (use largest resolution that fits)</option>
                    <option value="random">Random (randomly select resolution)</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    How to assign images to resolutions when multiple are specified
                  </p>
                </div>
              )}

              {/* Bucket Strategy */}
              <div>
                <label className="block text-sm text-gray-400 mb-1.5">Bucket Strategy</label>
                <select
                  value={bucketStrategy}
                  onChange={(e) => updateParam("bucket_strategy", e.target.value as "resize" | "crop" | "random_crop")}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
                >
                  <option value="resize">Resize (Lanczos)</option>
                  <option value="crop">Center Crop</option>
                  <option value="random_crop">Random Crop</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  How to handle images that don't fit bucket exactly
                </p>
              </div>

              {/* Epoch-dynamic crop augmentation (SDXL only) */}
              {isSDXLModel(baseModelPath) && (
              <div className="border-t border-gray-700 pt-3">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={params.crop_augment_enable ?? false}
                    onChange={(e) => updateParam("crop_augment_enable", e.target.checked)}
                    className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded"
                  />
                  <span className="text-sm text-gray-300">Epoch-dynamic crop augmentation (SDXL)</span>
                </label>
                <p className="text-xs text-gray-500 mt-1">
                  Per (image, epoch), two independent axes decide presentation: crop
                  (full image vs random crop) and bucket size (largest-fitting vs smaller).
                  Re-bucketed each epoch. Forces on-the-fly latent encoding (no latent
                  cache). SDXL only.
                </p>

                {params.crop_augment_enable && (
                  <div className="mt-3 space-y-3 pl-2 border-l-2 border-gray-700">
                    {/* Mix proportions (2x2 axes) */}
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">
                        Full-image probability: {(params.crop_full_image_prob ?? 0.7).toFixed(2)}
                      </label>
                      <input
                        type="range" min={0} max={1} step={0.05}
                        value={params.crop_full_image_prob ?? 0.7}
                        onChange={(e) => updateParam("crop_full_image_prob", parseFloat(e.target.value))}
                        className="w-full"
                      />
                      <p className="text-xs text-gray-500">P(full image, minimal crop only); rest are random crops.</p>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">
                        Max-bucket probability: {(params.crop_max_bucket_prob ?? 0.7).toFixed(2)}
                      </label>
                      <input
                        type="range" min={0} max={1} step={0.05}
                        value={params.crop_max_bucket_prob ?? 0.7}
                        onChange={(e) => updateParam("crop_max_bucket_prob", parseFloat(e.target.value))}
                        className="w-full"
                      />
                      <p className="text-xs text-gray-500">P(largest-fitting bucket = least downscale); rest use a smaller bucket.</p>
                    </div>

                    {/* Random-crop controls */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-sm text-gray-400 mb-1">Min area ratio</label>
                        <NumberInput
                          min={0.01} max={1} step="any" parse="float"
                          value={params.crop_min_area_ratio ?? 0.25}
                          defaultValue={0.25}
                          placeholder="0.25"
                          onCommit={(v) => updateParam("crop_min_area_ratio", v)}
                          className="w-full px-3 py-2 text-sm"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-gray-400 mb-1">Min short side (px)</label>
                        <NumberInput
                          min={64} step={64}
                          value={params.crop_min_short_side_px ?? 512}
                          defaultValue={512}
                          placeholder="512"
                          onCommit={(v) => updateParam("crop_min_short_side_px", v)}
                          className="w-full px-3 py-2 text-sm"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-sm text-gray-400 mb-1">Crop aspect</label>
                        <select
                          value={params.crop_aspect_mode ?? "source"}
                          onChange={(e) => updateParam("crop_aspect_mode", e.target.value as "source" | "free")}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm"
                        >
                          <option value="source">Source (keep image aspect)</option>
                          <option value="free">Free (any aspect)</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm text-gray-400 mb-1">Crop position</label>
                        <select
                          value={params.crop_position_mode ?? "random"}
                          onChange={(e) => updateParam("crop_position_mode", e.target.value as "random" | "corner")}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm"
                        >
                          <option value="random">Random point</option>
                          <option value="corner">Include a corner</option>
                        </select>
                      </div>
                    </div>

                    {/* Smaller-bucket controls */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-sm text-gray-400 mb-1">Smaller-bucket mode</label>
                        <select
                          value={params.crop_smaller_bucket_mode ?? "base_res"}
                          onChange={(e) => updateParam("crop_smaller_bucket_mode", e.target.value as "base_res" | "scale_range")}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm"
                        >
                          <option value="base_res">Smaller base resolution</option>
                          <option value="scale_range">Continuous scale range</option>
                        </select>
                        <p className="text-xs text-gray-500 mt-0.5">base_res needs multiple base_resolutions.</p>
                      </div>
                      <div>
                        <label className="block text-sm text-gray-400 mb-1">Full-image crop position</label>
                        <select
                          value={params.full_crop_position_mode ?? "center"}
                          onChange={(e) => updateParam("full_crop_position_mode", e.target.value as "center" | "fixed_corner" | "random")}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm"
                        >
                          <option value="center">Center</option>
                          <option value="fixed_corner">Fixed corner (top-left)</option>
                          <option value="random">Random per epoch</option>
                        </select>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-sm text-gray-400 mb-1">Smaller scale min</label>
                        <NumberInput
                          min={0.1} max={1} step="any" parse="float"
                          value={(params.crop_smaller_scale_range ?? [0.5, 0.9])[0]}
                          defaultValue={0.5}
                          placeholder="0.5"
                          onCommit={(v) => updateParam("crop_smaller_scale_range", [v, (params.crop_smaller_scale_range ?? [0.5, 0.9])[1]])}
                          className="w-full px-3 py-2 text-sm"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-gray-400 mb-1">Smaller scale max (≤ 1.0)</label>
                        <NumberInput
                          min={0.1} max={1} step="any" parse="float"
                          value={(params.crop_smaller_scale_range ?? [0.5, 0.9])[1]}
                          defaultValue={0.9}
                          placeholder="0.9"
                          onCommit={(v) => updateParam("crop_smaller_scale_range", [(params.crop_smaller_scale_range ?? [0.5, 0.9])[0], v])}
                          className="w-full px-3 py-2 text-sm"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm text-gray-400 mb-1">Plan seed (0 = train seed)</label>
                      <NumberInput
                        min={0} step={1}
                        value={params.crop_plan_seed ?? 0}
                        defaultValue={0}
                        placeholder="0"
                        onCommit={(v) => updateParam("crop_plan_seed", v)}
                        className="w-full px-3 py-2 text-sm"
                      />
                    </div>
                    <p className="text-xs text-gray-500">
                      Micro-conditioning: kohya (original_size = full image). Crops are
                      deterministic per (seed, epoch, image) for reproducible resume.
                    </p>
                  </div>
                )}
              </div>
              )}
            </>
            )}
          </>
        </div>
        </div>

        {/* Buttons - Outside grid */}
        <div className="flex flex-col sm:flex-row justify-end gap-2 sm:gap-3 pt-3 sm:pt-4 mt-3 sm:mt-4">
          <button
            type="button"
            onClick={onClose}
            className="px-3 sm:px-4 py-1.5 sm:py-2 bg-gray-700 hover:bg-gray-600 rounded text-xs sm:text-sm transition-colors"
            disabled={loading}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="px-3 sm:px-4 py-1.5 sm:py-2 bg-blue-600 hover:bg-blue-500 rounded text-xs sm:text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={loading}
          >
            {loading ? (editRunId ? "Updating..." : "Creating...") : (editRunId ? "Update Training Run" : "Create Training Run")}
          </button>
        </div>
      </form>

      {/* Save Preset Dialog */}
      {showPresetDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-2 sm:p-4">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 sm:p-6 w-full max-w-md">
            <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">Save Training Preset</h3>
            <div className="space-y-3 sm:space-y-4">
              <div>
                <label className="block text-xs sm:text-sm text-gray-300 mb-1">Preset Name *</label>
                <input
                  type="text"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                  className="w-full px-2.5 sm:px-3 py-1.5 sm:py-2 bg-gray-700 border border-gray-600 rounded text-xs sm:text-sm focus:outline-none focus:border-blue-500"
                  placeholder="e.g., SDXL LoRA Quick"
                />
              </div>
              <div>
                <label className="block text-xs sm:text-sm text-gray-300 mb-1">Description (Optional)</label>
                <textarea
                  value={presetDescription}
                  onChange={(e) => setPresetDescription(e.target.value)}
                  className="w-full px-2.5 sm:px-3 py-1.5 sm:py-2 bg-gray-700 border border-gray-600 rounded text-xs sm:text-sm focus:outline-none focus:border-blue-500"
                  rows={3}
                  placeholder="Describe this preset..."
                />
              </div>
              <div className="flex flex-col sm:flex-row gap-2 justify-end">
                <button
                  type="button"
                  onClick={() => setShowPresetDialog(false)}
                  className="px-3 sm:px-4 py-1.5 sm:py-2 bg-gray-700 hover:bg-gray-600 rounded text-xs sm:text-sm transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSavePreset}
                  className="px-3 sm:px-4 py-1.5 sm:py-2 bg-green-600 hover:bg-green-500 rounded text-xs sm:text-sm transition-colors"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Load Preset Dialog */}
      {showLoadPresetDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-2 sm:p-4">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 sm:p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
            <h3 className="text-lg font-semibold mb-4">Load Training Preset</h3>
            {presets.length === 0 ? (
              <p className="text-gray-400 text-sm">No presets saved yet</p>
            ) : (
              <div className="space-y-2">
                {presets.map((preset) => (
                  <div
                    key={preset.id}
                    className="bg-gray-700/50 border border-gray-600 rounded p-3 hover:bg-gray-700 transition-colors"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="font-medium">{preset.name}</h4>
                          <span className="text-xs px-2 py-0.5 bg-blue-600 rounded">
                            {preset.training_method === "lora" ? "LoRA" : preset.training_method === "relora" ? "ReLoRA" : preset.training_method === "controlnet" ? "ControlNet" : "Full Finetune"}
                          </span>
                        </div>
                        {preset.description && (
                          <p className="text-sm text-gray-400 mb-2">{preset.description}</p>
                        )}
                        <p className="text-xs text-gray-500">
                          Created: {new Date(preset.created_at).toLocaleString()}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => handleLoadPreset(preset)}
                          className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-sm transition-colors"
                        >
                          Load
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeletePreset(preset.id)}
                          className="p-1.5 bg-red-600 hover:bg-red-500 rounded transition-colors"
                          title="Delete preset"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
            <div className="mt-4 flex justify-end">
              <button
                type="button"
                onClick={() => setShowLoadPresetDialog(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Compact labelled numeric input. Integer vs float is inferred from `step`.
 * Float fields render `step="any"`: a numeric step would restrict the field to
 * the `min + n*step` grid and let the spinner/wheel rewrite typed values.
 */
function NumField({
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
