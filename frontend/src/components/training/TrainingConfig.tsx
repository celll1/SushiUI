"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { X, Save, FolderOpen, Trash2 } from "lucide-react";
import { createTrainingRun, updateTrainingRun, listDatasets, Dataset, TrainingRun, getModels, DatasetConfigItem, getRandomCaption, getSamplers, getScheduleTypes, listTrainingPresets, createTrainingPreset, deleteTrainingPreset, TrainingPreset, getTrainingRunParams, updateTrainingConfig, getControlNets, SamplePrompt, TrainingRunCreateRequest, listTrainingRuns } from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";
import TextareaWithTagSuggestions from "../common/TextareaWithTagSuggestions";
import NumberInput from "../common/NumberInput";
import VisionEncoderSelector from "../common/VisionEncoderSelector";
import TimestepDistributionGraph from "./TimestepDistributionGraph";

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

// Optimizer configuration: defines available options and defaults for each optimizer
const OPTIMIZER_CONFIGS: Record<string, {
  label: string;
  supportsPaged?: boolean;
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
    supportsPaged: true,
    defaults: { beta1: "0.9", beta2: "0.999", epsilon: "1e-8", weight_decay: "0.01" }
  },
  "adamw8bit": {
    label: "AdamW 8-bit",
    supportsPaged: true,
    defaults: { beta1: "0.9", beta2: "0.999", epsilon: "1e-8", weight_decay: "0.01" }
  },
  "adamw8bit_ringbuffer": {
    label: "AdamW 8-bit Ring Buffer",
    supportsCautious: true,
    defaults: { beta1: "0.9", beta2: "0.999", epsilon: "1e-8", weight_decay: "0.01" }
  },
  "lion8bit": {
    label: "Lion 8-bit",
    supportsPaged: true,
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

// ============================================================
// Single-state migration (Phase 3a foundation)
// ============================================================
// All training parameters will be progressively migrated into this single
// `params` object. See SINGLE_STATE_MIGRATION_PLAN.md for details.
const DEFAULT_PARAMS: TrainingRunCreateRequest = {
  training_method: "lora",
  base_model_path: "",
  dataset_configs: [],
  total_steps: 1000,
  // Initialized (not undefined) so users can toggle the "Epochs" radio
  // and submit without having to touch the input — matches legacy behaviour
  // where useState(10) guaranteed the value was always present.
  // getRequestData() strips one of them based on `useEpochs`.
  epochs: 10,
  batch_size: 4,
  gradient_accumulation_steps: 1,
  max_grad_norm: 1.0,
  learning_rate: 1e-5,
  lr_scheduler: "constant",
  lr_warmup_steps: 0,
  optimizer: "adamw8bit",
  optimizer_is_paged: false,
  optimizer_cautious: false,
  optimizer_beta1: 0.9,
  optimizer_beta2: 0.999,
  optimizer_epsilon: 1e-8,
  optimizer_weight_decay: 0.01,
  optimizer_schedule_free: false,
  optimizer_schedule_free_r: 0.0,
  optimizer_schedule_free_weight_lr_power: 2.0,
  optimizer_use_radam: false,
  optimizer_stochastic_rounding: false,
  lora_rank: 16,
  lora_alpha: 16,
  lora_dtype: "fp32",
  relora_merge_every: 500,
  relora_merge_unit: "steps",
  restart_warmup_steps: 100,
  optimizer_reset_strategy: "full_reset",
  optimizer_pruning_ratio: 0.9,
  save_every: 100,
  save_every_unit: "steps",
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
  unet_lr: 1e-5,
  text_encoder_lr: 1e-6,
  text_encoder_1_lr: null,
  text_encoder_2_lr: null,
  image_encoder_lr: null,
  weight_dtype: "fp32",
  training_dtype: "fp16",
  output_dtype: "fp32",
  vae_dtype: "fp16",
  mixed_precision: true,
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
  rescan_before_training: "off",
};

export default function TrainingConfig({ onClose, onRunCreated, editRunId, onRunUpdated }: TrainingConfigProps) {
  console.log(`[TrainingConfig] Component mounted/re-rendered, editRunId=${editRunId}`);

  // ============================================================
  // Single-state form (Phase 3a–3m complete)
  // ============================================================
  // All top-level TrainingRunCreateRequest fields live in `params`.
  // UI inputs read via const aliases (e.g. `const batchSize = params.batch_size ?? 4`)
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
  const { trainingDefaults, timestepDefaultsByArch, bundleVaeDefaultsByArch } = useStartup();

  // Apply backend-fetched defaults when they arrive (only for new runs, not edit mode)
  useEffect(() => {
    if (!trainingDefaults || editRunId) return;
    setParams(prev => ({ ...DEFAULT_PARAMS, ...(trainingDefaults as Partial<TrainingRunCreateRequest>) }));
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

  // Model architecture filters
  const [showSD15, setShowSD15] = useState(true);
  const [showSDXL, setShowSDXL] = useState(true);
  const [showZImage, setShowZImage] = useState(true);
  // DEUS support removed: const [showDEUS, setShowDEUS] = useState(true);
  const [showFlux2, setShowFlux2] = useState(true);
  const [showAnima, setShowAnima] = useState(true);

  // Flag to track if dtype settings have been explicitly set (from YAML or user)
  // When true, baseModelPath changes will NOT override dtype settings
  const dtypeExplicitlySetRef = useRef(false);

  // Flag to track if we are in the middle of restoring from YAML
  // When true, optimizer useEffect will NOT reset hyperparameters to defaults
  const restoringFromYAMLRef = useRef(false);

  // Tracks the baseModelPath for which the per-arch default timestep_sampling has
  // already been applied, so model changes apply the model's default exactly once
  // while user edits (which don't change baseModelPath) are never clobbered.
  const lastTimestepModelRef = useRef<string | null>(null);
  // Same pattern for the per-arch default bundle_vae (sd15/sdxl/deus -> true).
  const lastBundleVaeModelRef = useRef<string | null>(null);

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
  const batchSize = params.batch_size ?? 4;
  const learningRate = localLrText;
  const lrScheduler = params.lr_scheduler ?? "constant";
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
  const optimizerIsPaged = params.optimizer_is_paged ?? false;
  const optimizerCautious = params.optimizer_cautious ?? false;
  const optimizerBeta1 = localBeta1Text;
  const optimizerBeta2 = localBeta2Text;
  const optimizerEpsilon = localEpsilonText;
  const optimizerWeightDecay = localWeightDecayText;
  const optimizerScheduleFree = params.optimizer_schedule_free ?? false;
  const optimizerScheduleFreeR = localScheduleFreeRText;
  const optimizerScheduleFreeWeightLrPower = localScheduleFreeWeightLrPowerText;
  const optimizerUseRadam = params.optimizer_use_radam ?? false;
  const optimizerStochasticRounding = params.optimizer_stochastic_rounding ?? false;

  // LoRA parameters (Phase 3d: migrated to params)
  const loraRank = params.lora_rank ?? 16;
  const loraAlpha = params.lora_alpha ?? 16;
  const loraDtype = params.lora_dtype ?? "fp32";

  // Advanced (Phase 3e: migrated to params)
  const [availableCheckpoints, setAvailableCheckpoints] = useState<Array<{step: number, filename: string}>>([]);
  const saveEvery = params.save_every ?? 100;
  const saveEveryUnit = (params.save_every_unit ?? "steps") as "steps" | "epochs";
  const sampleEvery = params.sample_every ?? 100;
  const resumeFromCheckpoint = params.resume_from_checkpoint ?? null;

  // Sample generation (Phase 3e: migrated to params)
  const samplePrompts = params.sample_prompts ?? [];
  const setSamplePrompts = useCallback((next: SamplePrompt[] | ((prev: SamplePrompt[]) => SamplePrompt[])) => {
    setParams(prev => ({
      ...prev,
      sample_prompts: typeof next === "function"
        ? (next as (p: SamplePrompt[]) => SamplePrompt[])(prev.sample_prompts ?? [])
        : next,
    }));
  }, []);
  const sampleWidth = params.sample_width ?? 1024;
  const sampleHeight = params.sample_height ?? 1024;
  const sampleSteps = params.sample_steps ?? 28;
  const sampleCfgScale = params.sample_cfg_scale ?? 7.0;
  const sampleSampler = params.sample_sampler ?? "euler";
  const sampleScheduleType = params.sample_schedule_type ?? "uniform";
  const sampleSeed = params.sample_seed ?? -1;
  const [conditionImagePreviews, setConditionImagePreviews] = useState<Record<number, string>>({});
  const conditionImageInputRefs = useRef<Record<number, HTMLInputElement | null>>({});
  const [referenceImagePreviews, setReferenceImagePreviews] = useState<Record<number, string>>({});
  const referenceImageInputRefs = useRef<Record<number, HTMLInputElement | null>>({});

  // Debug options (Phase 3f: migrated to params)
  const debugLatents = params.debug_latents ?? false;
  const debugLatentsEvery = params.debug_latents_every ?? 50;

  // Reference image conditioning (FLUX.2 only) — Phase 3k: migrated to params
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
  const cacheLatentsToDisk = params.cache_latents_to_disk ?? true;
  const forceRecache = params.force_recache ?? false;

  // Component-specific training (Phase 3g: migrated to params)
  const trainUnet = params.train_unet ?? true;
  const trainTextEncoder = params.train_text_encoder ?? true;
  const trainImageEncoder = params.train_image_encoder ?? false;
  // Local text states preserve in-progress numeric input (scientific notation)
  const [localUnetLrText, setLocalUnetLrText] = useState<string>("1e-5");
  const [localTextEncoderLrText, setLocalTextEncoderLrText] = useState<string>("1e-6");
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

  // Text encoding mode (Phase 3i: migrated to params)
  const textEncodingMode = params.text_encoding_mode ?? "swap_onthefly";
  const textEncodingSwapInterval = params.text_encoding_swap_interval ?? 256;

  // Latent encoding mode
  const latentEncodingMode = params.latent_encoding_mode ?? "swap_onthefly";
  const latentEncodingSwapInterval = params.latent_encoding_swap_interval ?? 256;

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

  // Helper: Detect model architecture
  const isZImageModel = (modelPath: string): boolean => {
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture === "zimage";
  };

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

  const getModelArchitecture = (modelPath: string): string | undefined => {
    if (modelPath.startsWith("scratch:minit2i:")) return "minit2i";
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture;
  };

  const isSDOrSDXLModel = (modelPath: string): boolean => {
    const arch = getModelArchitecture(modelPath);
    return arch === "sd15" || arch === "sdxl";
  };

  // Filter models by architecture
  const filteredModels = availableModels.filter((model) => {
    if (model.architecture === "sd15" && !showSD15) return false;
    if (model.architecture === "sdxl" && !showSDXL) return false;
    if (model.architecture === "zimage" && !showZImage) return false;
    // DEUS support removed: if (model.architecture === "deus" && !showDEUS) return false;
    if (model.architecture === "flux2" && !showFlux2) return false;
    if (model.architecture === "anima" && !showAnima) return false;
    return true;
  });

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
  // and the UI <input> element.
  // ============================================================

  /**
   * Build the outgoing requestData dict from current useState values.
   * Used by handleSubmit() and Loop generation stepParams.
   */
  const getRequestData = useCallback((): any => {
    return {
      dataset_configs: datasetConfigs.filter(c => c.dataset_id !== 0),
      run_name: runName.trim() || undefined,
      training_method: trainingMethod,
      base_model_path: baseModelPath.trim(),
      // MiniT2I config (sent so UI values reach the backend, not just defaults).
      minit2i_label_drop_rate: params.minit2i_label_drop_rate,
      minit2i_lr_factor: params.minit2i_lr_factor,
      minit2i_flan_t5_path: params.minit2i_flan_t5_path,
      minit2i_scratch_init_from: params.minit2i_scratch_init_from,
      minit2i_inherit_final_layer: params.minit2i_inherit_final_layer,
      // Krea 2 config (sent so UI values reach the backend, not just defaults).
      krea2_lora_scope: params.krea2_lora_scope,
      krea2_lr_factor: params.krea2_lr_factor,
      krea2_discrete_flow_shift: params.krea2_discrete_flow_shift,
      // REPA (Representation Alignment) — MiniT2I only.
      repa_enable: params.repa_enable,
      repa_encoder_source: params.repa_encoder_source,
      repa_tagger_model_dir: params.repa_tagger_model_dir,
      repa_siglip2_repo: params.repa_siglip2_repo,
      repa_align_depth: params.repa_align_depth,
      repa_weight: params.repa_weight,
      repa_proj_lr_factor: params.repa_proj_lr_factor,
      repa_encoder_resolution: params.repa_encoder_resolution,
      total_steps: useEpochs ? undefined : params.total_steps,
      epochs: useEpochs ? params.epochs : undefined,
      batch_size: params.batch_size,
      gradient_accumulation_steps: params.gradient_accumulation_steps,
      max_grad_norm: params.max_grad_norm,
      learning_rate: parseFloat(localLrText),
      lr_scheduler: params.lr_scheduler,
      lr_warmup_steps: params.lr_warmup_steps,
      optimizer: params.optimizer,
      optimizer_is_paged: params.optimizer_is_paged,
      optimizer_cautious: params.optimizer_cautious,
      optimizer_beta1: localBeta1Text ? parseFloat(localBeta1Text) : undefined,
      optimizer_beta2: localBeta2Text ? parseFloat(localBeta2Text) : undefined,
      optimizer_epsilon: localEpsilonText ? parseFloat(localEpsilonText) : undefined,
      optimizer_weight_decay: localWeightDecayText ? parseFloat(localWeightDecayText) : undefined,
      optimizer_schedule_free: params.optimizer_schedule_free,
      optimizer_schedule_free_r: localScheduleFreeRText ? parseFloat(localScheduleFreeRText) : 0.0,
      optimizer_schedule_free_weight_lr_power: localScheduleFreeWeightLrPowerText ? parseFloat(localScheduleFreeWeightLrPowerText) : 2.0,
      optimizer_use_radam: params.optimizer_use_radam,
      optimizer_stochastic_rounding: params.optimizer_stochastic_rounding,
      lora_rank: (trainingMethod === "lora" || trainingMethod === "relora") ? params.lora_rank : undefined,
      lora_alpha: (trainingMethod === "lora" || trainingMethod === "relora") ? params.lora_alpha : undefined,
      lora_dtype: (trainingMethod === "lora" || trainingMethod === "relora") ? params.lora_dtype : undefined,
      ...(trainingMethod === "relora" ? {
        relora_merge_every: params.relora_merge_every,
        relora_merge_unit: params.relora_merge_unit,
        restart_warmup_steps: params.restart_warmup_steps,
        optimizer_reset_strategy: params.optimizer_reset_strategy,
        optimizer_pruning_ratio: params.optimizer_pruning_ratio,
      } : {}),
      save_every: params.save_every,
      save_every_unit: params.save_every_unit,
      sample_every: params.sample_every,
      sample_prompts: params.sample_prompts,
      sample_width: params.sample_width,
      sample_height: params.sample_height,
      sample_steps: params.sample_steps,
      sample_cfg_scale: params.sample_cfg_scale,
      sample_sampler: params.sample_sampler,
      sample_schedule_type: params.sample_schedule_type,
      sample_seed: params.sample_seed,
      resume_from_checkpoint: params.resume_from_checkpoint || undefined,
      debug_latents: params.debug_latents,
      debug_latents_every: params.debug_latents_every,
      enable_bucketing: params.enable_bucketing,
      base_resolutions: params.enable_bucketing ? params.base_resolutions : undefined,
      bucket_strategy: params.enable_bucketing ? params.bucket_strategy : undefined,
      multi_resolution_mode: params.enable_bucketing ? params.multi_resolution_mode : undefined,
      // Epoch-dynamic crop augmentation (SDXL only; requires bucketing)
      crop_augment_enable: params.enable_bucketing ? params.crop_augment_enable : false,
      crop_full_image_prob: params.crop_full_image_prob,
      crop_max_bucket_prob: params.crop_max_bucket_prob,
      crop_min_area_ratio: params.crop_min_area_ratio,
      crop_min_short_side_px: params.crop_min_short_side_px,
      crop_aspect_mode: params.crop_aspect_mode,
      crop_position_mode: params.crop_position_mode,
      crop_smaller_bucket_mode: params.crop_smaller_bucket_mode,
      crop_smaller_scale_range: params.crop_smaller_scale_range ?? [0.5, 0.9],
      full_crop_position_mode: params.full_crop_position_mode,
      crop_microcond_mode: params.crop_microcond_mode,
      crop_plan_seed: params.crop_plan_seed,
      cache_latents_to_disk: params.cache_latents_to_disk,
      force_recache: params.force_recache,
      train_unet: params.train_unet,
      train_text_encoder: params.train_text_encoder,
      train_image_encoder: params.train_image_encoder,
      unet_lr: localUnetLrText ? parseFloat(localUnetLrText) : null,
      text_encoder_lr: localTextEncoderLrText ? parseFloat(localTextEncoderLrText) : null,
      text_encoder_1_lr: localTextEncoder1LrText ? parseFloat(localTextEncoder1LrText) : null,
      text_encoder_2_lr: localTextEncoder2LrText ? parseFloat(localTextEncoder2LrText) : null,
      image_encoder_lr: localImageEncoderLrText ? parseFloat(localImageEncoderLrText) : null,
      weight_dtype: params.weight_dtype,
      training_dtype: params.training_dtype,
      output_dtype: params.output_dtype,
      vae_dtype: params.vae_dtype,
      mixed_precision: params.mixed_precision,
      attention_backend: params.attention_backend,
      attention_impl: params.attention_impl,
      use_flash_attention: params.use_flash_attention,
      min_snr_gamma: params.min_snr_gamma,
      reconstruction_loss_weight: params.reconstruction_loss_weight,
      text_encoding_mode: params.text_encoding_mode,
      text_encoding_swap_interval: params.text_encoding_swap_interval,
      use_reference_images: params.use_reference_images,
      vision_encoder_path: params.vision_encoder_path || null,
      train_vision_encoder: params.train_vision_encoder,
      vision_encoder_lr: localVisionEncoderLrText ? parseFloat(localVisionEncoderLrText) : null,
      gradient_routing_ve: params.gradient_routing_ve,
      param_tracking: params.param_tracking,
      param_tracking_interval: params.param_tracking_interval,
      latent_encoding_mode: params.latent_encoding_mode,
      latent_encoding_swap_interval: params.latent_encoding_swap_interval,
      blocks_to_swap: params.blocks_to_swap,
      use_pinned_memory: params.use_pinned_memory,
      block_swap_h2d_only: params.block_swap_h2d_only,
      block_swap_ring_size: params.block_swap_ring_size,
      num_optimizer_groups: params.num_optimizer_groups,
      bundle_vae: params.bundle_vae,
      activation_dispatch_enable: params.activation_dispatch_enable,
      activation_dispatch_margin_gb: params.activation_dispatch_margin_gb,
      activation_dispatch_seed_coef: params.activation_dispatch_seed_coef,
      activation_dispatch_residual_frac: params.activation_dispatch_residual_frac,
      activation_dispatch_threshold_mb: params.activation_dispatch_threshold_mb,
      multi_noise_timesteps: params.multi_noise_timesteps,
      multi_noise_mode: params.multi_noise_mode,
      trajectory_blend_alpha: params.trajectory_blend_alpha,
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
      snr_regularization_weight: params.snr_regularization_weight,
      snr_timestep_adaptive: params.snr_timestep_adaptive,
      snr_penalty_mode: params.snr_penalty_mode,
      energy_regularization_weight: params.energy_regularization_weight,
      energy_timestep_adaptive: params.energy_timestep_adaptive,
      energy_penalty_mode: params.energy_penalty_mode,
      energy_normalize_by_pixels: params.energy_normalize_by_pixels,
      noise_process: params.noise_process,
      prediction_target: params.prediction_target,
      sdxl_vae_type: params.sdxl_vae_type,
      sdxl_te_type: params.sdxl_te_type,
      sdxl_te_hidden_layer: params.sdxl_te_hidden_layer,
      sdxl_te_max_len: params.sdxl_te_max_len,
      sdxl_te_train_encoder: params.sdxl_te_train_encoder,
      strict_validation: params.strict_validation,
      controlnet_type: trainingMethod === "controlnet" ? params.controlnet_type : undefined,
      controlnet_pretrained_path: trainingMethod === "controlnet" && params.controlnet_pretrained_path ? params.controlnet_pretrained_path : undefined,
      controlnet_init_from_unet: trainingMethod === "controlnet" ? params.controlnet_init_from_unet : undefined,
      lllite_conditioning_channels: trainingMethod === "controlnet" && params.controlnet_type === "lllite" ? params.lllite_conditioning_channels : undefined,
      lllite_rank: trainingMethod === "controlnet" && params.controlnet_type === "lllite" ? params.lllite_rank : undefined,
      condition_preprocessors: trainingMethod === "controlnet" && (params.condition_preprocessors?.length ?? 0) > 0 ? params.condition_preprocessors : undefined,
      condition_cache_mode: trainingMethod === "controlnet" && (params.condition_preprocessors?.length ?? 0) > 0 ? params.condition_cache_mode : undefined,
      rescan_before_training: params.rescan_before_training ?? "off",
      danbooru_aug_enable: params.danbooru_aug_enable,
      danbooru_aug_queries: params.danbooru_aug_queries,
      danbooru_aug_weight_static: params.danbooru_aug_weight_static,
      danbooru_aug_deficiency_enable: params.danbooru_aug_deficiency_enable,
      danbooru_aug_deficiency_min_count: params.danbooru_aug_deficiency_min_count,
      danbooru_aug_deficiency_top_k: params.danbooru_aug_deficiency_top_k,
      danbooru_aug_deficiency_manual: params.danbooru_aug_deficiency_manual,
      danbooru_aug_weight_deficiency: params.danbooru_aug_weight_deficiency,
      danbooru_aug_injection_interval: params.danbooru_aug_injection_interval,
      danbooru_aug_injection_ratio: params.danbooru_aug_injection_ratio,
      danbooru_aug_min_score: params.danbooru_aug_min_score,
      danbooru_aug_max_posts_per_query: params.danbooru_aug_max_posts_per_query,
      danbooru_aug_api_interval: params.danbooru_aug_api_interval,
      danbooru_aug_dl_speed_kbps: params.danbooru_aug_dl_speed_kbps,
      danbooru_speed_check_enable: params.danbooru_speed_check_enable,
      danbooru_speed_degraded_kbps: params.danbooru_speed_degraded_kbps,
      danbooru_speed_min_slow_streak: params.danbooru_speed_min_slow_streak,
      danbooru_speed_min_slow_seconds: params.danbooru_speed_min_slow_seconds,
      danbooru_speed_cooldown_seconds: params.danbooru_speed_cooldown_seconds,
      danbooru_aug_buffer_size: params.danbooru_aug_buffer_size,
      danbooru_aug_include_rating_tag: params.danbooru_aug_include_rating_tag,
      danbooru_aug_max_caption_tags: params.danbooru_aug_max_caption_tags,
      danbooru_quality_tag_enable: params.danbooru_quality_tag_enable,
      danbooru_quality_tag_thresholds: params.danbooru_quality_tag_thresholds,
      danbooru_quality_tag_attach_negative: params.danbooru_quality_tag_attach_negative,
      danbooru_aug_shuffle_tags: params.danbooru_aug_shuffle_tags,
      danbooru_aug_shuffle_keep_first_n: params.danbooru_aug_shuffle_keep_first_n,
      danbooru_aug_tag_dropout_rate: params.danbooru_aug_tag_dropout_rate,
      danbooru_aug_tag_dropout_keep_first_n: params.danbooru_aug_tag_dropout_keep_first_n,
      danbooru_aug_caption_dropout_rate: params.danbooru_aug_caption_dropout_rate,
      danbooru_aug_keep_tokens: params.danbooru_aug_keep_tokens,
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
    if (incoming.training_method) setTrainingMethod(incoming.training_method);
    if (incoming.dataset_configs) setDatasetConfigs(incoming.dataset_configs);

    // Exclusive steps/epochs radio state
    if (incoming.total_steps !== undefined && incoming.total_steps !== null) setUseEpochs(false);
    if (incoming.epochs !== undefined && incoming.epochs !== null) setUseEpochs(true);

    // --- Fields that require local text sync (numeric-input helpers) ---
    if (incoming.learning_rate !== undefined && incoming.learning_rate !== null) {
      setLocalLrText(incoming.learning_rate.toString());
    }
    if (incoming.optimizer_beta1 !== undefined && incoming.optimizer_beta1 !== null) {
      setLocalBeta1Text(incoming.optimizer_beta1.toString());
    }
    if (incoming.optimizer_beta2 !== undefined && incoming.optimizer_beta2 !== null) {
      setLocalBeta2Text(incoming.optimizer_beta2.toString());
    }
    if (incoming.optimizer_epsilon !== undefined && incoming.optimizer_epsilon !== null) {
      setLocalEpsilonText(incoming.optimizer_epsilon.toString());
    }
    if (incoming.optimizer_weight_decay !== undefined && incoming.optimizer_weight_decay !== null) {
      setLocalWeightDecayText(incoming.optimizer_weight_decay.toString());
    }
    if (incoming.optimizer_schedule_free_r !== undefined) {
      setLocalScheduleFreeRText(incoming.optimizer_schedule_free_r.toString());
    }
    if (incoming.optimizer_schedule_free_weight_lr_power !== undefined) {
      setLocalScheduleFreeWeightLrPowerText(incoming.optimizer_schedule_free_weight_lr_power.toString());
    }
    if (incoming.unet_lr !== undefined && incoming.unet_lr !== null) {
      setLocalUnetLrText(incoming.unet_lr.toString());
    }
    if (incoming.text_encoder_lr !== undefined && incoming.text_encoder_lr !== null) {
      setLocalTextEncoderLrText(incoming.text_encoder_lr.toString());
    }
    if (incoming.text_encoder_1_lr !== undefined && incoming.text_encoder_1_lr !== null) {
      setLocalTextEncoder1LrText(incoming.text_encoder_1_lr.toString());
    }
    if (incoming.text_encoder_2_lr !== undefined && incoming.text_encoder_2_lr !== null) {
      setLocalTextEncoder2LrText(incoming.text_encoder_2_lr.toString());
    }
    if (incoming.image_encoder_lr !== undefined && incoming.image_encoder_lr !== null) {
      setLocalImageEncoderLrText(incoming.image_encoder_lr.toString());
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
    const PARAM_KEYS: (keyof TrainingRunCreateRequest)[] = [
      "lora_rank", "lora_alpha", "lora_dtype",
      "total_steps", "epochs",
      "batch_size", "gradient_accumulation_steps", "max_grad_norm", "learning_rate", "lr_scheduler", "lr_warmup_steps", "optimizer",
      "optimizer_beta1", "optimizer_beta2", "optimizer_epsilon", "optimizer_weight_decay",
      "optimizer_is_paged", "optimizer_cautious", "optimizer_schedule_free",
      "optimizer_schedule_free_r", "optimizer_schedule_free_weight_lr_power",
      "optimizer_use_radam", "optimizer_stochastic_rounding",
      "save_every", "save_every_unit", "resume_from_checkpoint",
      "train_unet", "train_text_encoder", "train_image_encoder",
      "unet_lr", "text_encoder_lr", "text_encoder_1_lr", "text_encoder_2_lr", "image_encoder_lr",
      "weight_dtype", "training_dtype", "output_dtype", "vae_dtype",
      "mixed_precision", "attention_backend", "attention_impl", "use_flash_attention", "min_snr_gamma", "reconstruction_loss_weight",
      "text_encoding_mode", "text_encoding_swap_interval",
      "latent_encoding_mode", "latent_encoding_swap_interval",
      "minit2i_label_drop_rate", "minit2i_lr_factor", "minit2i_flan_t5_path", "minit2i_scratch_init_from",
      "minit2i_inherit_final_layer",
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
      "blocks_to_swap", "use_pinned_memory", "block_swap_h2d_only", "block_swap_ring_size", "num_optimizer_groups",
      "bundle_vae",
      "activation_dispatch_enable", "activation_dispatch_margin_gb",
      "activation_dispatch_seed_coef", "activation_dispatch_residual_frac",
      "activation_dispatch_threshold_mb",
      "multi_noise_timesteps", "multi_noise_mode", "trajectory_blend_alpha",
      "snr_regularization_weight", "snr_timestep_adaptive", "snr_penalty_mode",
      "energy_regularization_weight", "energy_timestep_adaptive", "energy_penalty_mode",
      "energy_normalize_by_pixels",
      "noise_process", "prediction_target", "strict_validation", "sdxl_vae_type",
      "sdxl_te_type", "sdxl_te_hidden_layer", "sdxl_te_max_len", "sdxl_te_train_encoder",
      "controlnet_type", "controlnet_init_from_unet",
      "lllite_conditioning_channels", "lllite_rank",
      "condition_cache_mode",
      "sample_every", "sample_prompts", "sample_width", "sample_height",
      "sample_steps", "sample_cfg_scale", "sample_sampler", "sample_schedule_type", "sample_seed",
      "debug_latents", "debug_latents_every",
      "enable_bucketing", "bucket_strategy", "multi_resolution_mode",
      "crop_augment_enable", "crop_full_image_prob", "crop_max_bucket_prob",
      "crop_min_area_ratio", "crop_min_short_side_px", "crop_aspect_mode",
      "crop_position_mode", "crop_smaller_bucket_mode", "crop_smaller_scale_range",
      "full_crop_position_mode", "crop_microcond_mode", "crop_plan_seed",
      "cache_latents_to_disk", "force_recache",
      "use_reference_images", "train_vision_encoder", "gradient_routing_ve",
      "vision_encoder_lr", "param_tracking", "param_tracking_interval",
      "relora_merge_every", "relora_merge_unit", "restart_warmup_steps",
      "optimizer_reset_strategy", "optimizer_pruning_ratio",
    ];
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
        .then((res) => setCopySourceRuns(res.runs || []))
        .catch((err) => console.error("[TrainingConfig] Failed to list runs for copy:", err));
    }
  }, [editRunId, loadTrainingRunParams]);

  // Auto-configure precision settings when model changes (only if not explicitly set)
  useEffect(() => {
    if (!baseModelPath) return;

    // Skip if dtype was explicitly set (from YAML load or user change)
    if (dtypeExplicitlySetRef.current) return;

    const arch = getModelArchitecture(baseModelPath);

    // Dtype presets based on architecture:
    // - SD1.5/SDXL/DEUS: VAE=fp16, weight=fp32, training=fp16, save=fp16
    // - Z-Image/FLUX.2: VAE=fp32, weight=bf16, training=bf16, save=bf16
    if (arch === "zimage" || arch === "flux2") {
      // Z-Image/FLUX.2: bf16 for weights/training/output, fp32 for VAE
      updateParam("weight_dtype", "bf16");
      updateParam("training_dtype", "bf16");
      updateParam("output_dtype", "bf16");
      updateParam("vae_dtype", "fp32");
      // Z-Image/FLUX.2: Cannot train text encoder (frozen)
      updateParam("train_text_encoder", false);
      // Z-Image/FLUX.2: VE not supported — clear selection
      updateParam("vision_encoder_path", "");
      updateParam("train_vision_encoder", false);
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

  // Ideogram 4 does not support Full Fine-tune (fp8 base; VRAM-impractical) —
  // fall back to LoRA if a full-FT method was carried over from another model/preset.
  useEffect(() => {
    if (isIdeogram4Model(baseModelPath) && (trainingMethod === "full_finetune" || trainingMethod === "relora")) {
      setTrainingMethod("lora");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseModelPath, trainingMethod]);


  // Reset optimizer hyperparameters when optimizer changes
  useEffect(() => {
    // Skip during YAML restoration — params are already being restored correctly
    if (restoringFromYAMLRef.current) return;

    const config = OPTIMIZER_CONFIGS[optimizer];
    if (!config) return;

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

    // Reset options that are not supported by the new optimizer
    if (!config.supportsPaged) updateParam("optimizer_is_paged", false);
    if (!config.supportsCautious) updateParam("optimizer_cautious", false);
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
        const showRefUI = trainingMethod !== "controlnet" &&
          ((isSDOrSDXLModel(baseModelPath) && !!visionEncoderPath) || isFlux2Model(baseModelPath));
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
        if (params.steps) updateParam("sample_steps", params.steps);
        if (params.cfg_scale) updateParam("sample_cfg_scale", params.cfg_scale);
        if (params.sampler) updateParam("sample_sampler", params.sampler);
        if (params.schedule_type) updateParam("sample_schedule_type", params.schedule_type);
        if (params.seed) updateParam("sample_seed", params.seed);
      }
    } catch (err) {
      console.error("Failed to import from generation panel:", err);
    }
  };

  // Get current config (excluding dataset and model path)
  const getCurrentConfig = () => {
    return {
      useEpochs,
      totalSteps: params.total_steps,
      epochs: params.epochs,
      batchSize: params.batch_size,
      gradientAccumulationSteps: params.gradient_accumulation_steps,
      maxGradNorm: params.max_grad_norm,
      learningRate: localLrText,
      lrScheduler: params.lr_scheduler,
      optimizer: params.optimizer,
      optimizerIsPaged: params.optimizer_is_paged,
      optimizerCautious: params.optimizer_cautious,
      optimizerBeta1: localBeta1Text,
      optimizerBeta2: localBeta2Text,
      optimizerEpsilon: localEpsilonText,
      optimizerWeightDecay: localWeightDecayText,
      optimizerScheduleFree: params.optimizer_schedule_free,
      optimizerScheduleFreeR: localScheduleFreeRText,
      optimizerScheduleFreeWeightLrPower: localScheduleFreeWeightLrPowerText,
      loraRank: params.lora_rank,
      loraAlpha: params.lora_alpha,
      loraDtype: params.lora_dtype,
      saveEvery,
      saveEveryUnit,
      sampleEvery,
      resumeFromCheckpoint,
      samplePrompts,
      sampleWidth,
      sampleHeight,
      sampleSteps,
      sampleCfgScale,
      sampleSampler,
      sampleScheduleType,
      sampleSeed,
      debugLatents,
      debugLatentsEvery,
      enableBucketing,
      baseResolutions,
      bucketStrategy,
      multiResolutionMode,
      cropAugmentEnable: params.crop_augment_enable,
      cropFullImageProb: params.crop_full_image_prob,
      cropMaxBucketProb: params.crop_max_bucket_prob,
      cropMinAreaRatio: params.crop_min_area_ratio,
      cropMinShortSidePx: params.crop_min_short_side_px,
      cropAspectMode: params.crop_aspect_mode,
      cropPositionMode: params.crop_position_mode,
      cropSmallerBucketMode: params.crop_smaller_bucket_mode,
      cropSmallerScaleRange: params.crop_smaller_scale_range,
      fullCropPositionMode: params.full_crop_position_mode,
      cropMicrocondMode: params.crop_microcond_mode,
      cropPlanSeed: params.crop_plan_seed,
      cacheLatentsToDisk,
      forceRecache,
      trainUnet,
      trainTextEncoder,
      unetLr,
      textEncoderLr,
      textEncoder1Lr,
      textEncoder2Lr,
      weightDtype,
      trainingDtype,
      outputDtype,
      vaeDtype,
      mixedPrecision,
      attentionBackend,
      attentionImpl,
      // Legacy compat mirror so old importers still read a flash flag.
      // Only 'flash' maps to the bool; tq/sage/native -> false (matches onChange + restore).
      useFlashAttention: attentionBackend === "flash",
      minSnrGamma,
      reconstructionLossWeight,
      textEncodingMode,
      textEncodingSwapInterval,
      latentEncodingMode,
      latentEncodingSwapInterval,
      blocksToSwap,
      usePinnedMemory,
      numOptimizerGroups,
      multiNoiseTimesteps,
      timestepDistribution,
      timestepMin,
      timestepMax,
      // ControlNet parameters
      controlnetType,
      controlnetPretrainedPath,
      controlnetInitFromUnet,
      llliteConditioningChannels,
      llliteRank,
      conditionPreprocessors,
      conditionCacheMode,
      // ReLoRA parameters
      reloraMergeEvery: params.relora_merge_every,
      reloraMergeUnit: params.relora_merge_unit,
      restartWarmupSteps: params.restart_warmup_steps,
      optimizerResetStrategy: params.optimizer_reset_strategy,
      optimizerPruningRatio: params.optimizer_pruning_ratio,
    };
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
    const config = preset.config;

    // Apply config (excluding dataset and model path)
    if (config.useEpochs !== undefined) setUseEpochs(config.useEpochs);
    if (config.totalSteps !== undefined) updateParam("total_steps", config.totalSteps);
    if (config.epochs !== undefined) updateParam("epochs", config.epochs);
    if (config.batchSize !== undefined) updateParam("batch_size", config.batchSize);
    if (config.gradientAccumulationSteps !== undefined) updateParam("gradient_accumulation_steps", config.gradientAccumulationSteps);
    if (config.maxGradNorm !== undefined) updateParam("max_grad_norm", config.maxGradNorm);
    if (config.learningRate !== undefined) {
      const v = parseFloat(config.learningRate);
      if (!isNaN(v)) updateParam("learning_rate", v);
      setLocalLrText(config.learningRate);
    }
    if (config.lrScheduler !== undefined) updateParam("lr_scheduler", config.lrScheduler);
    if (config.optimizer !== undefined) updateParam("optimizer", config.optimizer);
    if (config.optimizerIsPaged !== undefined) updateParam("optimizer_is_paged", config.optimizerIsPaged);
    if (config.optimizerCautious !== undefined) updateParam("optimizer_cautious", config.optimizerCautious);
    if (config.optimizerBeta1 !== undefined) {
      const v = parseFloat(config.optimizerBeta1);
      if (!isNaN(v)) updateParam("optimizer_beta1", v);
      setLocalBeta1Text(config.optimizerBeta1);
    }
    if (config.optimizerBeta2 !== undefined) {
      const v = parseFloat(config.optimizerBeta2);
      if (!isNaN(v)) updateParam("optimizer_beta2", v);
      setLocalBeta2Text(config.optimizerBeta2);
    }
    if (config.optimizerEpsilon !== undefined) {
      const v = parseFloat(config.optimizerEpsilon);
      if (!isNaN(v)) updateParam("optimizer_epsilon", v);
      setLocalEpsilonText(config.optimizerEpsilon);
    }
    if (config.optimizerWeightDecay !== undefined) {
      const v = parseFloat(config.optimizerWeightDecay);
      if (!isNaN(v)) updateParam("optimizer_weight_decay", v);
      setLocalWeightDecayText(config.optimizerWeightDecay);
    }
    if (config.optimizerScheduleFree !== undefined) updateParam("optimizer_schedule_free", config.optimizerScheduleFree);
    if (config.optimizerScheduleFreeR !== undefined) {
      const v = parseFloat(config.optimizerScheduleFreeR);
      if (!isNaN(v)) updateParam("optimizer_schedule_free_r", v);
      setLocalScheduleFreeRText(config.optimizerScheduleFreeR);
    }
    if (config.optimizerScheduleFreeWeightLrPower !== undefined) {
      const v = parseFloat(config.optimizerScheduleFreeWeightLrPower);
      if (!isNaN(v)) updateParam("optimizer_schedule_free_weight_lr_power", v);
      setLocalScheduleFreeWeightLrPowerText(config.optimizerScheduleFreeWeightLrPower);
    }
    if (config.loraRank !== undefined) updateParam("lora_rank", config.loraRank);
    if (config.loraAlpha !== undefined) updateParam("lora_alpha", config.loraAlpha);
    if (config.loraDtype !== undefined) updateParam("lora_dtype", config.loraDtype);
    if (config.saveEvery !== undefined) updateParam("save_every", config.saveEvery);
    if (config.saveEveryUnit !== undefined) updateParam("save_every_unit", config.saveEveryUnit);
    if (config.sampleEvery !== undefined) updateParam("sample_every", config.sampleEvery);
    if (config.resumeFromCheckpoint !== undefined) updateParam("resume_from_checkpoint", config.resumeFromCheckpoint);
    if (config.samplePrompts !== undefined) updateParam("sample_prompts", config.samplePrompts);
    if (config.sampleWidth !== undefined) updateParam("sample_width", config.sampleWidth);
    if (config.sampleHeight !== undefined) updateParam("sample_height", config.sampleHeight);
    if (config.sampleSteps !== undefined) updateParam("sample_steps", config.sampleSteps);
    if (config.sampleCfgScale !== undefined) updateParam("sample_cfg_scale", config.sampleCfgScale);
    if (config.sampleSampler !== undefined) updateParam("sample_sampler", config.sampleSampler);
    if (config.sampleScheduleType !== undefined) updateParam("sample_schedule_type", config.sampleScheduleType);
    if (config.sampleSeed !== undefined) updateParam("sample_seed", config.sampleSeed);
    if (config.debugLatents !== undefined) updateParam("debug_latents", config.debugLatents);
    if (config.debugLatentsEvery !== undefined) updateParam("debug_latents_every", config.debugLatentsEvery);
    if (config.useReferenceImages !== undefined) updateParam("use_reference_images", config.useReferenceImages);
    if (config.priority_training) {
      setPriorityEnabled(true);
      const entries = config.priority_training.entries || [];
      setPriorityText(entries.map((e: any) => typeof e === "string" ? e : JSON.stringify(e)).join("\n"));
      setPriorityMultiplier(config.priority_training.multiplier || 1);
    }
    if (config.enableBucketing !== undefined) updateParam("enable_bucketing", config.enableBucketing);
    if (config.baseResolutions !== undefined) updateParam("base_resolutions", config.baseResolutions);
    if (config.bucketStrategy !== undefined) updateParam("bucket_strategy", config.bucketStrategy);
    if (config.multiResolutionMode !== undefined) updateParam("multi_resolution_mode", config.multiResolutionMode);
    if (config.cropAugmentEnable !== undefined) updateParam("crop_augment_enable", config.cropAugmentEnable);
    if (config.cropFullImageProb !== undefined) updateParam("crop_full_image_prob", config.cropFullImageProb);
    if (config.cropMaxBucketProb !== undefined) updateParam("crop_max_bucket_prob", config.cropMaxBucketProb);
    if (config.cropMinAreaRatio !== undefined) updateParam("crop_min_area_ratio", config.cropMinAreaRatio);
    if (config.cropMinShortSidePx !== undefined) updateParam("crop_min_short_side_px", config.cropMinShortSidePx);
    if (config.cropAspectMode !== undefined) updateParam("crop_aspect_mode", config.cropAspectMode);
    if (config.cropPositionMode !== undefined) updateParam("crop_position_mode", config.cropPositionMode);
    if (config.cropSmallerBucketMode !== undefined) updateParam("crop_smaller_bucket_mode", config.cropSmallerBucketMode);
    if (config.cropSmallerScaleRange !== undefined) updateParam("crop_smaller_scale_range", config.cropSmallerScaleRange);
    if (config.fullCropPositionMode !== undefined) updateParam("full_crop_position_mode", config.fullCropPositionMode);
    if (config.cropMicrocondMode !== undefined) updateParam("crop_microcond_mode", config.cropMicrocondMode);
    if (config.cropPlanSeed !== undefined) updateParam("crop_plan_seed", config.cropPlanSeed);
    if (config.cacheLatentsToDisk !== undefined) updateParam("cache_latents_to_disk", config.cacheLatentsToDisk);
    if (config.forceRecache !== undefined) updateParam("force_recache", config.forceRecache);
    if (config.trainUnet !== undefined) updateParam("train_unet", config.trainUnet);
    if (config.trainTextEncoder !== undefined) updateParam("train_text_encoder", config.trainTextEncoder);
    if (config.unetLr !== undefined) {
      const v = parseFloat(config.unetLr);
      if (!isNaN(v)) updateParam("unet_lr", v);
      setLocalUnetLrText(config.unetLr);
    }
    if (config.textEncoderLr !== undefined) {
      const v = parseFloat(config.textEncoderLr);
      if (!isNaN(v)) updateParam("text_encoder_lr", v);
      setLocalTextEncoderLrText(config.textEncoderLr);
    }
    if (config.textEncoder1Lr !== undefined) {
      const v = parseFloat(config.textEncoder1Lr);
      if (!isNaN(v)) updateParam("text_encoder_1_lr", v);
      setLocalTextEncoder1LrText(config.textEncoder1Lr);
    }
    if (config.textEncoder2Lr !== undefined) {
      const v = parseFloat(config.textEncoder2Lr);
      if (!isNaN(v)) updateParam("text_encoder_2_lr", v);
      setLocalTextEncoder2LrText(config.textEncoder2Lr);
    }
    if (config.weightDtype !== undefined) updateParam("weight_dtype", config.weightDtype);
    if (config.trainingDtype !== undefined) updateParam("training_dtype", config.trainingDtype);
    if (config.outputDtype !== undefined) updateParam("output_dtype", config.outputDtype);
    if (config.vaeDtype !== undefined) updateParam("vae_dtype", config.vaeDtype);
    if (config.mixedPrecision !== undefined) updateParam("mixed_precision", config.mixedPrecision);
    // R6 compat: attention_backend is authoritative if present; otherwise map the
    // legacy useFlashAttention bool (true->flash) so old presets don't silently drop
    // to native. use_flash_attention is kept synchronized as the derived mirror.
    if (config.attentionBackend !== undefined) {
      updateParam("attention_backend", config.attentionBackend);
      // use_flash_attention is true ONLY for the flash backend; tq and native map to false.
      updateParam("use_flash_attention", config.attentionBackend === "flash");
    } else if (config.useFlashAttention !== undefined) {
      updateParam("attention_backend", config.useFlashAttention ? "flash" : "native");
      updateParam("use_flash_attention", config.useFlashAttention);
    }
    // Attention implementation registry (conduit|diffusers); orthogonal to backend.
    if (config.attentionImpl !== undefined) {
      updateParam("attention_impl", config.attentionImpl);
    }
    if (config.minSnrGamma !== undefined) updateParam("min_snr_gamma", config.minSnrGamma);
    if (config.reconstructionLossWeight !== undefined) updateParam("reconstruction_loss_weight", config.reconstructionLossWeight);
    if (config.textEncodingMode !== undefined) updateParam("text_encoding_mode", config.textEncodingMode);
    if (config.textEncodingSwapInterval !== undefined) updateParam("text_encoding_swap_interval", config.textEncodingSwapInterval);
    if (config.latentEncodingMode !== undefined) updateParam("latent_encoding_mode", config.latentEncodingMode);
    if (config.latentEncodingSwapInterval !== undefined) updateParam("latent_encoding_swap_interval", config.latentEncodingSwapInterval);
    if (config.blocksToSwap !== undefined) updateParam("blocks_to_swap", config.blocksToSwap);
    if (config.usePinnedMemory !== undefined) updateParam("use_pinned_memory", config.usePinnedMemory);
    if (config.numOptimizerGroups !== undefined) updateParam("num_optimizer_groups", config.numOptimizerGroups);
    if (config.multiNoiseTimesteps !== undefined) updateParam("multi_noise_timesteps", config.multiNoiseTimesteps);
    if (config.timestepDistribution !== undefined) setTimestepDistribution(config.timestepDistribution);
    if (config.timestepMin !== undefined) setTimestepMin(config.timestepMin);
    if (config.timestepMax !== undefined) setTimestepMax(config.timestepMax);
    if (config.timestepMean !== undefined) setTimestepMean(config.timestepMean);
    if (config.timestepStd !== undefined) setTimestepStd(config.timestepStd);
    if (config.timestepAlpha !== undefined) setTimestepAlpha(config.timestepAlpha);
    if (config.timestepBeta !== undefined) setTimestepBeta(config.timestepBeta);

    // ControlNet parameters
    if (config.controlnetType !== undefined) updateParam("controlnet_type", config.controlnetType);
    if (config.controlnetPretrainedPath !== undefined) updateParam("controlnet_pretrained_path", config.controlnetPretrainedPath);
    if (config.controlnetInitFromUnet !== undefined) updateParam("controlnet_init_from_unet", config.controlnetInitFromUnet);
    if (config.llliteConditioningChannels !== undefined) updateParam("lllite_conditioning_channels", config.llliteConditioningChannels);
    if (config.llliteRank !== undefined) updateParam("lllite_rank", config.llliteRank);
    if (config.conditionPreprocessors !== undefined) updateParam("condition_preprocessors", config.conditionPreprocessors);
    if (config.conditionCacheMode !== undefined) updateParam("condition_cache_mode", config.conditionCacheMode);
    // Legacy: sampleConditionImagePath is now per-prompt (ignored here)

    // ReLoRA parameters
    if (config.reloraMergeEvery !== undefined) updateParam("relora_merge_every", config.reloraMergeEvery);
    if (config.reloraMergeUnit !== undefined) updateParam("relora_merge_unit", config.reloraMergeUnit);
    if (config.restartWarmupSteps !== undefined) updateParam("restart_warmup_steps", config.restartWarmupSteps);
    if (config.optimizerResetStrategy !== undefined) updateParam("optimizer_reset_strategy", config.optimizerResetStrategy);
    if (config.optimizerPruningRatio !== undefined) updateParam("optimizer_pruning_ratio", config.optimizerPruningRatio);

    // Also switch to the preset's training method
    if (preset.training_method) setTrainingMethod(preset.training_method);

    setShowLoadPresetDialog(false);
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
            <label
              className={`flex items-center space-x-2 ${isIdeogram4Model(baseModelPath) ? 'cursor-not-allowed' : 'cursor-pointer'}`}
              title={isIdeogram4Model(baseModelPath) ? 'Ideogram 4 Full Fine-tune is not supported (fp8 base; VRAM-impractical for individuals). Use LoRA.' : undefined}
            >
              <input
                type="radio"
                name="training_method"
                value="full_finetune"
                checked={trainingMethod === "full_finetune"}
                onChange={() => setTrainingMethod("full_finetune")}
                disabled={isIdeogram4Model(baseModelPath)}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className={`text-sm ${isIdeogram4Model(baseModelPath) ? 'text-gray-500' : ''}`}>
                Full Fine-tune{isIdeogram4Model(baseModelPath) ? ' (N/A for Ideogram 4)' : ''}
              </span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="training_method"
                value="controlnet"
                checked={trainingMethod === "controlnet"}
                onChange={() => setTrainingMethod("controlnet")}
                disabled={fromScratchMiniT2I}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className={`text-sm ${fromScratchMiniT2I ? 'text-gray-500' : ''}`}>ControlNet (SD1.5/SDXL)</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="training_method"
                value="relora"
                checked={trainingMethod === "relora"}
                onChange={() => setTrainingMethod("relora")}
                disabled={fromScratchMiniT2I}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className={`text-sm ${fromScratchMiniT2I ? 'text-gray-500' : ''}`}>ReLoRA (Periodic Merge + Reinit)</span>
            </label>
          </div>
        </div>

        {/* Base Model */}
        <div className="break-inside-avoid">
          <label className="block text-sm font-medium mb-2">
            Base Model <span className="text-red-400">*</span>
          </label>

          {/* Model Architecture Filter */}
          <div className="flex items-center gap-4 mb-2 text-xs">
            <span className="text-gray-400">Filter:</span>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={showSD15}
                onChange={(e) => setShowSD15(e.target.checked)}
                className="w-3.5 h-3.5"
              />
              <span className="text-gray-300">SD 1.5</span>
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={showSDXL}
                onChange={(e) => setShowSDXL(e.target.checked)}
                className="w-3.5 h-3.5"
              />
              <span className="text-gray-300">SDXL</span>
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={showZImage}
                onChange={(e) => setShowZImage(e.target.checked)}
                className="w-3.5 h-3.5"
              />
              <span className="text-gray-300">Z-Image</span>
            </label>
            {/* DEUS support removed
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={showDEUS}
                onChange={(e) => setShowDEUS(e.target.checked)}
                className="w-3.5 h-3.5"
              />
              <span className="text-gray-300">DEUS</span>
            </label>
            */}
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={showFlux2}
                onChange={(e) => setShowFlux2(e.target.checked)}
                className="w-3.5 h-3.5"
              />
              <span className="text-gray-300">FLUX.2</span>
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={showAnima}
                onChange={(e) => setShowAnima(e.target.checked)}
                className="w-3.5 h-3.5"
              />
              <span className="text-gray-300">Anima</span>
            </label>
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
                        step="0.05"
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
                        step="0.1"
                        value={params.repa_proj_lr_factor ?? 1.0}
                        onChange={(e) => updateParam("repa_proj_lr_factor", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                        onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("repa_proj_lr_factor", 1.0); }}
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
              onChange={(path) => updateParam("vision_encoder_path", path || "")}
              label=""
            />
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
                <option value="lllite">LLLite (kohya-ss sd-scripts compatible)</option>
              </select>
            </div>

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

            {conditionPreprocessors.length > 0 && (
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
                onChange={(e) => updateParam("batch_size", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("batch_size", 4); }}
                min="1"
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Gradient Accumulation Steps</label>
              <input
                type="number"
                value={params.gradient_accumulation_steps ?? 1}
                onChange={(e) => updateParam("gradient_accumulation_steps", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value)) || parseInt(e.target.value) < 1) updateParam("gradient_accumulation_steps", 1); }}
                min="1"
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">Effective batch = Batch Size × this. Reduces gradient noise without extra VRAM.</p>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Max Grad Norm</label>
              <input
                type="number"
                step="0.1"
                value={params.max_grad_norm ?? 1.0}
                onChange={(e) => updateParam("max_grad_norm", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("max_grad_norm", 1.0); }}
                min="0"
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">Gradient clipping threshold. 0 disables clipping.</p>
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
                  step="0.1"
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Blending coefficient: 0.0=ideal only, 1.0=stepped only (default: 0.7)
                </p>
              </div>
            )}

            {/* Timestep Sampling */}
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
                      step="0.05"
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
                      step="0.05"
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                </div>
                <p className="text-xs text-gray-500">
                  Timestep range for sampling (0.0 = clean, 1.0 = fully noised)
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
                        step="0.1"
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
                        step="0.1"
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <p className="col-span-2 text-xs text-gray-500">
                      {timestepDistribution === "normal" ? (
                        <>Mean: center of distribution (0.0-1.0). Std: spread (smaller = more concentrated)</>
                      ) : (
                        <>Mean: positive = high timesteps (noisy), negative = low timesteps (clean). Std: spread</>
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
                        step="0.5"
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
                        step="0.5"
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
                    X-axis: Timestep (0=clean, 1=noisy) | Y-axis: Sampling probability
                  </p>
                </div>
              </div>
            </div>

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
                      step="0.01"
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
                      step="0.01"
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
                  if (!isNaN(v)) updateParam("learning_rate", v);
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
              </select>
            </div>

            {/* Optimizer Selection */}
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Optimizer</label>
                <select
                  value={optimizer}
                  onChange={(e) => updateParam("optimizer", e.target.value)}
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                >
                  <option value="adamw">AdamW</option>
                  <option value="adamw8bit">AdamW 8-bit</option>
                  <option value="adamw8bit_ringbuffer">AdamW 8-bit Ring Buffer</option>
                  <option value="lion8bit">Lion 8-bit</option>
                  <option value="lion8bit_ringbuffer">Lion 8-bit Ring Buffer</option>
                  <option value="adafactor">Adafactor</option>
                </select>
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
                {/* is_paged option (AdamW, AdamW8bit, Lion8bit) */}
                {OPTIMIZER_CONFIGS[optimizer]?.supportsPaged && (
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="optimizer-is-paged"
                      checked={optimizerIsPaged}
                      onChange={(e) => updateParam("optimizer_is_paged", e.target.checked)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="optimizer-is-paged" className="text-xs text-gray-300 cursor-pointer">
                      Paged (CPU offload)
                    </label>
                  </div>
                )}

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

                        {/* Stochastic Rounding (BF16 only, AdamW8bit/Lion8bit only) */}
                        {trainingDtype === "bf16" && (optimizer === "adamw8bit_ringbuffer" || optimizer === "lion8bit_ringbuffer") && (
                          <div>
                            <label className="flex items-center text-xs text-gray-300">
                              <input
                                type="checkbox"
                                checked={optimizerStochasticRounding}
                                onChange={(e) => updateParam("optimizer_stochastic_rounding", e.target.checked)}
                                className="mr-2"
                              />
                              Stochastic Rounding (BF16)
                            </label>
                            <p className="text-xs text-gray-500 mt-1">
                              Reduces quantization bias for BF16 training. Only affects AdamW8bit/Lion8bit with BF16.
                            </p>
                          </div>
                        )}

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
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="train-unet"
                  checked={trainUnet}
                  onChange={(e) => updateParam("train_unet", e.target.checked)}
                  className="w-4 h-4"
                />
                <label htmlFor="train-unet" className="text-xs text-gray-300 cursor-pointer">
                  Train U-Net
                </label>
              </div>

              {/* Train Text Encoder */}
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="train-text-encoder"
                  checked={trainTextEncoder}
                  onChange={(e) => updateParam("train_text_encoder", e.target.checked)}
                  disabled={isZImageModel(baseModelPath) || isAnimaModel(baseModelPath) || isLensModel(baseModelPath) || isIdeogram4Model(baseModelPath)}
                  className="w-4 h-4 disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <label htmlFor="train-text-encoder" className={`text-xs cursor-pointer ${isZImageModel(baseModelPath) || isAnimaModel(baseModelPath) || isLensModel(baseModelPath) || isIdeogram4Model(baseModelPath) ? 'text-gray-500' : 'text-gray-300'}`}>
                  Train Text Encoder {isZImageModel(baseModelPath) && '(Not supported for Z-Image)'}
                  {isAnimaModel(baseModelPath) && '(Not supported for Anima)'}
                  {isLensModel(baseModelPath) && '(Not supported for Lens)'}
                  {isMiniT2IModel(baseModelPath) && '(FLAN-T5)'}
                </label>
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
                  onBlur={(e) => { const v = parseFloat(e.target.value); updateParam("unet_lr", isNaN(v) ? null : v); }}
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
                    onBlur={(e) => { const v = parseFloat(e.target.value); updateParam("text_encoder_lr", isNaN(v) ? null : v); }}
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
                        onBlur={(e) => { const v = parseFloat(e.target.value); updateParam("text_encoder_1_lr", isNaN(v) ? null : v); }}
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
                        onBlur={(e) => { const v = parseFloat(e.target.value); updateParam("text_encoder_2_lr", isNaN(v) ? null : v); }}
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
                      onBlur={(e) => { const v = parseFloat(e.target.value); updateParam("vision_encoder_lr", isNaN(v) ? null : v); }}
                      placeholder={`Default: ${textEncoderLr || learningRate}`}
                      className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                )}
                {trainVisionEncoder && (
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="gradient-routing-ve"
                      checked={gradientRoutingVE}
                      onChange={(e) => updateParam("gradient_routing_ve", e.target.checked)}
                      className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                    />
                    <label htmlFor="gradient-routing-ve" className="text-xs text-gray-300 cursor-pointer">
                      Gradient Routing
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
                step={0.5}
                min={0}
                max={20}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
              />
              <p className="text-xs text-gray-500">
                Default: 5.0. Set to 0 to disable. Prevents overfitting to high-noise timesteps.
              </p>
            </div>

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
                step={0.05}
                min={0}
                max={1.0}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
              />
              <p className="text-xs text-gray-500">
                Default: 0.0 (prediction loss only). Dual loss: loss = (1-β)*pred_loss + β*recon_loss. Try 0.1 for faster learning in noisy timesteps.
              </p>
            </div>
          </div>

          <p className="text-xs text-gray-500">
            Lower precision dtypes reduce VRAM usage. FP8 can save ~50% VRAM. Use FP32 output for best loss calculation accuracy. Flash Attention improves training speed and reduces memory usage. Min-SNR gamma reweights loss to balance learning across all timesteps. Reconstruction loss weight enables dual loss training (direct image quality optimization).
          </p>
        </div>

        {/* Block Swap Settings (VRAM Optimization) */}
        <div className="break-inside-avoid border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Block Swap (Training VRAM Optimization)</h3>

          <div className="space-y-3">
            {/* Blocks to Swap */}
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

            {/* Use Pinned Memory */}
            {blocksToSwap > 0 && (
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
            {blocksToSwap > 0 && (
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
            {blocksToSwap > 0 && (params.block_swap_h2d_only ?? false) && (
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
                    step={0.5}
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
                      <option value="fp8_e4m3fn">FP8 E4M3 (recommended)</option>
                      <option value="fp8_e5m2">FP8 E5M2</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Quantise the frozen Anima DiT base to FP8 before LoRA wrap (~50% VRAM saving).
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
                        step={0.1}
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
                        step={0.1}
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
                        step={0.1}
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
                      <option value="fp8_e4m3fn">FP8 E4M3 (recommended)</option>
                      <option value="fp8_e5m2">FP8 E5M2</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Quantise the frozen Lens DiT base to FP8 before LoRA wrap (~50% VRAM saving).
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
                        step={0.1}
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
                        step={0.1}
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                      />
                    </div>
                  </div>
                )}
              </>
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
                      step={0.1}
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
                    step={0.1}
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
                    step={0.1}
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
                    step={0.1}
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
                    value={params.minit2i_label_drop_rate ?? 0.1}
                    onChange={(e) => updateParam("minit2i_label_drop_rate", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))}
                    onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("minit2i_label_drop_rate", 0.1); }}
                    min={0}
                    max={1}
                    step={0.05}
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm"
                  />
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
                    step={0.1}
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
            {blocksToSwap > 0 && (
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
              className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
            >
              <option value="swap_onthefly">Swap On-the-Fly (Recommended)</option>
              <option value="pre_encoded_cache">Pre-Encoded Cache (Disk)</option>
              <option value="onthefly_gpu">On-the-Fly GPU Encoding</option>
              <option value="cpu_prefetch">CPU Prefetch (background thread; TE pinned to CPU)</option>
            </select>
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

        {/* Reference Image Conditioning (FLUX.2 only) */}
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
              Enable reference image conditioning (FLUX.2 only)
            </label>
          </div>
          <div className="text-xs text-gray-500 space-y-1">
            <p>Uses reference images from dataset to condition the model during training via latent concatenation.</p>
            <p>Dataset items must have reference images configured (e.g., <code className="bg-gray-800 px-1 rounded">image_ref.png</code> suffix).</p>
            <p className="text-yellow-500/80">⚠️ Only supported for FLUX.2 models. Will be ignored for other architectures.</p>
          </div>
        </div>

        {/* SigLIP2 Vision Encoder — info only; selector is near Base Model, train/LR are in Component-Specific LR */}
        <div className="border border-gray-700 rounded p-4 space-y-2">
          <h3 className="text-sm font-medium text-gray-300 mb-2">SigLIP2 Vision Encoder</h3>
          <div className="text-xs text-gray-500 space-y-1">
            <p>参照画像を持つデータセットアイテムにのみ VE 条件付けが適用されます。参照画像なしのアイテムは通常のトレーニングが行われます。</p>
            <p>VE チェックポイントは <code className="bg-gray-800 px-1 rounded">*_vision_encoder_step_*.safetensors</code> として保存されます。</p>
            <p className="text-yellow-500/80">⚠️ SD 1.5 / SDXL モデルのみ対応。VE 選択はモデル選択欄の下にあります。</p>
          </div>
        </div>

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
        <div className="border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Latent Encoding Mode (VAE)</h3>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Encoding Mode</label>
            <select
              value={latentEncodingMode}
              onChange={(e) => updateParam("latent_encoding_mode", e.target.value)}
              className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
            >
              <option value="swap_onthefly">Swap On-the-Fly (Recommended)</option>
              <option value="pre_encoded_cache">Pre-Encoded Cache (Disk)</option>
              <option value="onthefly_gpu">On-the-Fly GPU Encoding</option>
            </select>
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

          <div className="text-xs text-gray-500 space-y-1">
            <p><strong>Swap On-the-Fly:</strong> VAE swaps with main model (U-Net or Transformer) every N steps. Uses DRAM buffer (~64MB for 256 steps). Recommended for VRAM efficiency.</p>
            <p><strong>Pre-Encoded Cache:</strong> Pre-encode all images to latents and cache to disk. Uses more disk space but no VRAM for VAE during training.</p>
            <p><strong>On-the-Fly GPU:</strong> Encode images on GPU without cache. VAE stays on GPU, uses more VRAM.</p>
          </div>
        </div>

        {/* Advanced Settings */}
        <div className="break-inside-avoid border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Advanced Settings</h3>

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
              min="1"
              value={saveEvery}
              onChange={(e) => updateParam("save_every", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("save_every", 100); }}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
              placeholder={saveEveryUnit === "steps" ? "e.g., 100" : "e.g., 1"}
            />
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
        <div className="break-inside-avoid border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Sample Generation (Optional)</h3>

          {/* Sample Every */}
          <div>
            <label className="block text-sm text-gray-400 mb-1.5">Generate Sample Every (steps)</label>
            <input
              type="number"
              min="0"
              value={sampleEvery}
              onChange={(e) => updateParam("sample_every", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("sample_every", 100); }}
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
                    ((isSDOrSDXLModel(baseModelPath) && !!visionEncoderPath) || isFlux2Model(baseModelPath)) && (
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">
                        Reference Image
                        <span className="text-gray-600 ml-1">
                          ({isFlux2Model(baseModelPath) ? "Latent concat" : "Vision Encoder"})
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
                onChange={(e) => updateParam("sample_width", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("sample_width", 1024); }}
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
                onChange={(e) => updateParam("sample_height", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("sample_height", 1024); }}
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
                onChange={(e) => updateParam("sample_steps", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("sample_steps", 28); }}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">CFG Scale</label>
              <input
                type="number"
                min="1"
                max="30"
                step="0.5"
                value={sampleCfgScale}
                onChange={(e) => updateParam("sample_cfg_scale", e.target.value === '' ? (undefined as any) : parseFloat(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseFloat(e.target.value))) updateParam("sample_cfg_scale", 7.0); }}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
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
            </div>
            <div>
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
            </div>
          </div>

          {/* Sample Seed */}
          <div>
            <label className="block text-sm text-gray-400 mb-1.5">Seed</label>
            <input
              type="number"
              value={sampleSeed}
              onChange={(e) => updateParam("sample_seed", e.target.value === '' ? (undefined as any) : parseInt(e.target.value))} onBlur={(e) => { if (e.target.value === '' || isNaN(parseInt(e.target.value))) updateParam("sample_seed", 42); }}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
              placeholder="-1 for random"
            />
            <p className="text-xs text-gray-500 mt-1">
              Use -1 for random seed (different each time)
            </p>
          </div>
        </div>

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
                min="1"
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
          <h3 className="text-sm font-medium text-gray-300 mb-3">Aspect Ratio Bucketing</h3>

          {/* Enable Bucketing Toggle */}
          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              id="enable-bucketing"
              checked={enableBucketing}
              onChange={(e) => updateParam("enable_bucketing", e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
            />
            <label htmlFor="enable-bucketing" className="text-sm text-gray-400">
              Enable aspect ratio bucketing
            </label>
          </div>
          <p className="text-xs text-gray-500">
            Allows training on images with different aspect ratios by bucketing them into similar sizes
          </p>

          {/* Bucketing Settings (only shown if enabled) */}
          {enableBucketing && (
            <>
              {/* Base Resolutions */}
              <div>
                <label className="block text-sm text-gray-400 mb-1.5">Base Resolutions</label>
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
                            type="checkbox"
                            id={`res-${res}`}
                            checked={baseResolutions.includes(res)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                updateParam("base_resolutions", [...baseResolutions, res].sort((a, b) => a - b));
                              } else {
                                // Prevent unchecking the last resolution
                                if (baseResolutions.length > 1) {
                                  updateParam("base_resolutions", baseResolutions.filter(r => r !== res));
                                }
                              }
                            }}
                            disabled={baseResolutions.length === 1 && baseResolutions.includes(res)}
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
              </div>

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
                          min={0.01} max={1} step={0.01} parse="float"
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
                          min={0.1} max={1} step={0.05} parse="float"
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
                          min={0.1} max={1} step={0.05} parse="float"
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
            </>
          )}

          {/* Cache Latents (always shown, works with or without bucketing) */}
          <div className="flex items-center space-x-3 pt-2 border-t border-gray-700">
            <input
              type="checkbox"
              id="cache-latents"
              checked={cacheLatentsToDisk}
              onChange={(e) => updateParam("cache_latents_to_disk", e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
            />
            <label htmlFor="cache-latents" className="text-sm text-gray-400">
              Cache latents to disk (reduces VRAM usage)
            </label>
          </div>
          <p className="text-xs text-gray-500">
            Pre-encode images to latents and cache to disk. Significantly reduces VRAM during training (VAE stays on CPU). Text encoding cache is configured separately via "Text Encoding Mode".
          </p>
        </div>

        {/* Force Recache */}
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="force-recache"
              checked={forceRecache}
              onChange={(e) => updateParam("force_recache", e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
            />
            <label htmlFor="force-recache" className="text-sm text-gray-400">
              Force regenerate latent cache
            </label>
          </div>
          <p className="text-xs text-gray-500">
            Force regenerate latent cache even if valid cache exists. Use this if you switched to a different VAE or if cache validation fails.
          </p>
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

/** Compact labelled numeric input. Integer vs float is inferred from `step`. */
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
        step={step}
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
