"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { X, Save, FolderOpen, Trash2 } from "lucide-react";
import { createTrainingRun, updateTrainingRun, listDatasets, Dataset, TrainingRun, getModels, DatasetConfigItem, getRandomCaption, getSamplers, getScheduleTypes, listTrainingPresets, createTrainingPreset, deleteTrainingPreset, TrainingPreset, getTrainingRunParams, updateTrainingConfig, getControlNets, SamplePrompt, TrainingRunCreateRequest } from "@/utils/api";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";
import TextareaWithTagSuggestions from "../common/TextareaWithTagSuggestions";
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
  sample_schedule_type: "uniform",
  sample_seed: -1,
  debug_latents: false,
  debug_latents_every: 50,
  enable_bucketing: false,
  base_resolutions: [1024],
  bucket_strategy: "resize",
  multi_resolution_mode: "max",
  cache_latents_to_disk: true,
  force_recache: false,
  train_unet: true,
  train_text_encoder: true,
  train_image_encoder: false,
  unet_lr: 1e-5,
  text_encoder_lr: 1e-6,
  text_encoder_1_lr: null,
  text_encoder_2_lr: null,
  image_encoder_lr: null,
  weight_dtype: "fp32",
  training_dtype: "fp16",
  output_dtype: "fp32",
  vae_dtype: "fp32",
  mixed_precision: true,
  use_flash_attention: false,
  min_snr_gamma: 5.0,
  reconstruction_loss_weight: 0.0,
  text_encoding_mode: "swap_onthefly",
  text_encoding_swap_interval: 256,
  latent_encoding_mode: "swap_onthefly",
  latent_encoding_swap_interval: 256,
  blocks_to_swap: 0,
  use_pinned_memory: false,
  num_optimizer_groups: 0,
  multi_noise_timesteps: 1,
  multi_noise_mode: "independent",
  trajectory_blend_alpha: 0.7,
  timestep_sampling: {
    distribution: "uniform",
    min_timestep: 0.0,
    max_timestep: 1.0,
  },
  regularization_type: null,
  snr_regularization_weight: 0.0,
  snr_timestep_adaptive: true,
  snr_penalty_mode: "relu",
  energy_regularization_weight: 0.0,
  energy_timestep_adaptive: true,
  energy_penalty_mode: "under",
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
};

export default function TrainingConfig({ onClose, onRunCreated, editRunId, onRunUpdated }: TrainingConfigProps) {
  console.log(`[TrainingConfig] Component mounted/re-rendered, editRunId=${editRunId}`);

  // ============================================================
  // Single-state migration (Phase 3a)
  // ============================================================
  // `params` will progressively absorb all individual useState fields.
  // For now it lives alongside the existing useStates; subsequent phases
  // will migrate UI inputs onto `params.x` / `updateParam("x", v)` and
  // remove the corresponding useState declarations.
  const [params, setParams] = useState<TrainingRunCreateRequest>(DEFAULT_PARAMS);

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

  // Model architecture filters
  const [showSD15, setShowSD15] = useState(true);
  const [showSDXL, setShowSDXL] = useState(true);
  const [showZImage, setShowZImage] = useState(true);
  // DEUS support removed: const [showDEUS, setShowDEUS] = useState(true);
  const [showFlux2, setShowFlux2] = useState(true);

  // Flag to track if dtype settings have been explicitly set (from YAML or user)
  // When true, baseModelPath changes will NOT override dtype settings
  const dtypeExplicitlySetRef = useRef(false);

  // Flag to track if we are in the middle of restoring from YAML
  // When true, optimizer useEffect will NOT reset hyperparameters to defaults
  const restoringFromYAMLRef = useRef(false);

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

  // ControlNet parameters
  const [controlnetType, setControlnetType] = useState<"standard" | "lllite">("standard");
  const [controlnetPretrainedPath, setControlnetPretrainedPath] = useState("");
  const [controlnetInitFromUnet, setControlnetInitFromUnet] = useState(true);
  const [availableControlNets, setAvailableControlNets] = useState<{path: string; name: string}[]>([]);
  const [llliteConditioningChannels, setLlliteConditioningChannels] = useState(32);
  const [llliteRank, setLlliteRank] = useState(64);
  const [conditionPreprocessors, setConditionPreprocessors] = useState<string[]>([]);
  const [conditionCacheMode, setConditionCacheMode] = useState<"on_the_fly" | "pre_generate">("on_the_fly");

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
  const vaeDtype = params.vae_dtype ?? "fp32";
  const mixedPrecision = params.mixed_precision ?? true;
  const useFlashAttention = params.use_flash_attention ?? false;
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

  const getModelArchitecture = (modelPath: string): string | undefined => {
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
      total_steps: useEpochs ? undefined : params.total_steps,
      epochs: useEpochs ? params.epochs : undefined,
      batch_size: params.batch_size,
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
      num_optimizer_groups: params.num_optimizer_groups,
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
      strict_validation: params.strict_validation,
      controlnet_type: trainingMethod === "controlnet" ? controlnetType : undefined,
      controlnet_pretrained_path: trainingMethod === "controlnet" && controlnetPretrainedPath ? controlnetPretrainedPath : undefined,
      controlnet_init_from_unet: trainingMethod === "controlnet" ? controlnetInitFromUnet : undefined,
      lllite_conditioning_channels: trainingMethod === "controlnet" && controlnetType === "lllite" ? llliteConditioningChannels : undefined,
      lllite_rank: trainingMethod === "controlnet" && controlnetType === "lllite" ? llliteRank : undefined,
      condition_preprocessors: trainingMethod === "controlnet" && conditionPreprocessors.length > 0 ? conditionPreprocessors : undefined,
      condition_cache_mode: trainingMethod === "controlnet" && conditionPreprocessors.length > 0 ? conditionCacheMode : undefined,
      priority_training: priorityEnabled && priorityText.trim() ? {
        entries: priorityText.trim().split("\n").map(line => line.trim()).filter(Boolean),
        multiplier: priorityMultiplier,
      } : undefined,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    datasetConfigs, runName, trainingMethod, baseModelPath, useEpochs, params, localLrText,
    localBeta1Text, localBeta2Text, localEpsilonText, localWeightDecayText,
    localScheduleFreeRText, localScheduleFreeWeightLrPowerText,
    localUnetLrText, localTextEncoderLrText, localTextEncoder1LrText, localTextEncoder2LrText, localImageEncoderLrText,
    localVisionEncoderLrText,
    timestepDistribution, timestepMin, timestepMax, timestepMean,
    timestepStd, timestepAlpha, timestepBeta, controlnetType,
    controlnetPretrainedPath, controlnetInitFromUnet, llliteConditioningChannels, llliteRank,
    conditionPreprocessors, conditionCacheMode, priorityEnabled, priorityText, priorityMultiplier,
  ]);

  /**
   * Apply an incoming params dict (from get_training_run_params API) to all useStates.
   * Used by loadTrainingRunParams() for Edit Config restoration.
   */
  const applyParamsToState = useCallback((params: any) => {
    if (params.run_name) setRunName(params.run_name);
    if (params.base_model_path !== undefined) setBaseModelPath(params.base_model_path || "");
    if (params.training_method) setTrainingMethod(params.training_method);
    if (params.dataset_configs) setDatasetConfigs(params.dataset_configs);

    // LoRA
    if (params.lora_rank !== undefined) setParams(prev => ({ ...prev, lora_rank: params.lora_rank }));
    if (params.lora_alpha !== undefined) setParams(prev => ({ ...prev, lora_alpha: params.lora_alpha }));
    if (params.lora_dtype !== undefined) setParams(prev => ({ ...prev, lora_dtype: params.lora_dtype as "fp32" | "fp16" | "bf16" }));

    // Steps/epochs
    if (params.total_steps !== undefined && params.total_steps !== null) {
      setParams(prev => ({ ...prev, total_steps: params.total_steps }));
      setUseEpochs(false);
    }
    if (params.epochs !== undefined && params.epochs !== null) {
      setParams(prev => ({ ...prev, epochs: params.epochs }));
      setUseEpochs(true);
    }

    // Core training
    if (params.batch_size !== undefined) setParams(prev => ({ ...prev, batch_size: params.batch_size }));
    if (params.learning_rate !== undefined && params.learning_rate !== null) {
      setParams(prev => ({ ...prev, learning_rate: params.learning_rate }));
      setLocalLrText(params.learning_rate.toString());
    }
    if (params.lr_scheduler !== undefined) setParams(prev => ({ ...prev, lr_scheduler: params.lr_scheduler }));
    if (params.lr_warmup_steps !== undefined) setParams(prev => ({ ...prev, lr_warmup_steps: params.lr_warmup_steps }));
    if (params.optimizer !== undefined) setParams(prev => ({ ...prev, optimizer: params.optimizer }));

    // Optimizer hyperparameters
    if (params.optimizer_beta1 !== undefined && params.optimizer_beta1 !== null) {
      setParams(prev => ({ ...prev, optimizer_beta1: params.optimizer_beta1 }));
      setLocalBeta1Text(params.optimizer_beta1.toString());
    }
    if (params.optimizer_beta2 !== undefined && params.optimizer_beta2 !== null) {
      setParams(prev => ({ ...prev, optimizer_beta2: params.optimizer_beta2 }));
      setLocalBeta2Text(params.optimizer_beta2.toString());
    }
    if (params.optimizer_epsilon !== undefined && params.optimizer_epsilon !== null) {
      setParams(prev => ({ ...prev, optimizer_epsilon: params.optimizer_epsilon }));
      setLocalEpsilonText(params.optimizer_epsilon.toString());
    }
    if (params.optimizer_weight_decay !== undefined && params.optimizer_weight_decay !== null) {
      setParams(prev => ({ ...prev, optimizer_weight_decay: params.optimizer_weight_decay }));
      setLocalWeightDecayText(params.optimizer_weight_decay.toString());
    }
    if (params.optimizer_is_paged !== undefined) setParams(prev => ({ ...prev, optimizer_is_paged: params.optimizer_is_paged }));
    if (params.optimizer_cautious !== undefined) setParams(prev => ({ ...prev, optimizer_cautious: params.optimizer_cautious }));
    if (params.optimizer_schedule_free !== undefined) setParams(prev => ({ ...prev, optimizer_schedule_free: params.optimizer_schedule_free }));
    if (params.optimizer_schedule_free_r !== undefined) {
      setParams(prev => ({ ...prev, optimizer_schedule_free_r: params.optimizer_schedule_free_r }));
      setLocalScheduleFreeRText(params.optimizer_schedule_free_r.toString());
    }
    if (params.optimizer_schedule_free_weight_lr_power !== undefined) {
      setParams(prev => ({ ...prev, optimizer_schedule_free_weight_lr_power: params.optimizer_schedule_free_weight_lr_power }));
      setLocalScheduleFreeWeightLrPowerText(params.optimizer_schedule_free_weight_lr_power.toString());
    }
    if (params.optimizer_use_radam !== undefined) setParams(prev => ({ ...prev, optimizer_use_radam: params.optimizer_use_radam }));
    if (params.optimizer_stochastic_rounding !== undefined) setParams(prev => ({ ...prev, optimizer_stochastic_rounding: params.optimizer_stochastic_rounding }));

    // Save/Resume
    if (params.save_every !== undefined) setParams(prev => ({ ...prev, save_every: params.save_every }));
    if (params.save_every_unit !== undefined) setParams(prev => ({ ...prev, save_every_unit: params.save_every_unit }));
    if (params.resume_from_checkpoint !== undefined) setParams(prev => ({ ...prev, resume_from_checkpoint: params.resume_from_checkpoint }));

    // Component training
    if (params.train_unet !== undefined) setParams(prev => ({ ...prev, train_unet: params.train_unet }));
    if (params.train_text_encoder !== undefined) setParams(prev => ({ ...prev, train_text_encoder: params.train_text_encoder }));
    if (params.train_image_encoder !== undefined) setParams(prev => ({ ...prev, train_image_encoder: params.train_image_encoder }));
    if (params.unet_lr !== undefined && params.unet_lr !== null) {
      setParams(prev => ({ ...prev, unet_lr: params.unet_lr }));
      setLocalUnetLrText(params.unet_lr.toString());
    }
    if (params.text_encoder_lr !== undefined && params.text_encoder_lr !== null) {
      setParams(prev => ({ ...prev, text_encoder_lr: params.text_encoder_lr }));
      setLocalTextEncoderLrText(params.text_encoder_lr.toString());
    }
    if (params.text_encoder_1_lr !== undefined && params.text_encoder_1_lr !== null) {
      setParams(prev => ({ ...prev, text_encoder_1_lr: params.text_encoder_1_lr }));
      setLocalTextEncoder1LrText(params.text_encoder_1_lr.toString());
    }
    if (params.text_encoder_2_lr !== undefined && params.text_encoder_2_lr !== null) {
      setParams(prev => ({ ...prev, text_encoder_2_lr: params.text_encoder_2_lr }));
      setLocalTextEncoder2LrText(params.text_encoder_2_lr.toString());
    }
    if (params.image_encoder_lr !== undefined && params.image_encoder_lr !== null) {
      setParams(prev => ({ ...prev, image_encoder_lr: params.image_encoder_lr }));
      setLocalImageEncoderLrText(params.image_encoder_lr.toString());
    }

    // Precision
    if (params.weight_dtype !== undefined) setParams(prev => ({ ...prev, weight_dtype: params.weight_dtype }));
    if (params.training_dtype !== undefined) setParams(prev => ({ ...prev, training_dtype: params.training_dtype }));
    if (params.output_dtype !== undefined) setParams(prev => ({ ...prev, output_dtype: params.output_dtype }));
    if (params.vae_dtype !== undefined) setParams(prev => ({ ...prev, vae_dtype: params.vae_dtype }));
    if (params.mixed_precision !== undefined) setParams(prev => ({ ...prev, mixed_precision: params.mixed_precision }));
    if (params.use_flash_attention !== undefined) setParams(prev => ({ ...prev, use_flash_attention: params.use_flash_attention }));
    if (params.min_snr_gamma !== undefined) setParams(prev => ({ ...prev, min_snr_gamma: params.min_snr_gamma }));
    if (params.reconstruction_loss_weight !== undefined) setParams(prev => ({ ...prev, reconstruction_loss_weight: params.reconstruction_loss_weight }));

    // Memory optimization
    if (params.text_encoding_mode !== undefined) setParams(prev => ({ ...prev, text_encoding_mode: params.text_encoding_mode }));
    if (params.text_encoding_swap_interval !== undefined) setParams(prev => ({ ...prev, text_encoding_swap_interval: params.text_encoding_swap_interval }));
    if (params.latent_encoding_mode !== undefined) setParams(prev => ({ ...prev, latent_encoding_mode: params.latent_encoding_mode }));
    if (params.latent_encoding_swap_interval !== undefined) setParams(prev => ({ ...prev, latent_encoding_swap_interval: params.latent_encoding_swap_interval }));
    if (params.blocks_to_swap !== undefined) setParams(prev => ({ ...prev, blocks_to_swap: params.blocks_to_swap }));
    if (params.use_pinned_memory !== undefined) setParams(prev => ({ ...prev, use_pinned_memory: params.use_pinned_memory }));
    if (params.num_optimizer_groups !== undefined) setParams(prev => ({ ...prev, num_optimizer_groups: params.num_optimizer_groups }));

    // MNT
    if (params.multi_noise_timesteps !== undefined) setParams(prev => ({ ...prev, multi_noise_timesteps: params.multi_noise_timesteps }));
    if (params.multi_noise_mode !== undefined) setParams(prev => ({ ...prev, multi_noise_mode: params.multi_noise_mode }));
    if (params.trajectory_blend_alpha !== undefined) setParams(prev => ({ ...prev, trajectory_blend_alpha: params.trajectory_blend_alpha }));
    if (params.timestep_sampling) {
      if (params.timestep_sampling.distribution !== undefined) setTimestepDistribution(params.timestep_sampling.distribution);
      if (params.timestep_sampling.min_timestep !== undefined) setTimestepMin(params.timestep_sampling.min_timestep);
      if (params.timestep_sampling.max_timestep !== undefined) setTimestepMax(params.timestep_sampling.max_timestep);
      if (params.timestep_sampling.mean !== undefined) setTimestepMean(params.timestep_sampling.mean);
      if (params.timestep_sampling.std !== undefined) setTimestepStd(params.timestep_sampling.std);
      if (params.timestep_sampling.alpha !== undefined) setTimestepAlpha(params.timestep_sampling.alpha);
      if (params.timestep_sampling.beta !== undefined) setTimestepBeta(params.timestep_sampling.beta);
    }

    // Regularization
    if (params.regularization_type !== undefined) setParams(prev => ({ ...prev, regularization_type: params.regularization_type || "none" }));
    if (params.snr_regularization_weight !== undefined) setParams(prev => ({ ...prev, snr_regularization_weight: params.snr_regularization_weight }));
    if (params.snr_timestep_adaptive !== undefined) setParams(prev => ({ ...prev, snr_timestep_adaptive: params.snr_timestep_adaptive }));
    if (params.snr_penalty_mode !== undefined) setParams(prev => ({ ...prev, snr_penalty_mode: params.snr_penalty_mode }));
    if (params.energy_regularization_weight !== undefined) setParams(prev => ({ ...prev, energy_regularization_weight: params.energy_regularization_weight }));
    if (params.energy_timestep_adaptive !== undefined) setParams(prev => ({ ...prev, energy_timestep_adaptive: params.energy_timestep_adaptive }));
    if (params.energy_penalty_mode !== undefined) setParams(prev => ({ ...prev, energy_penalty_mode: params.energy_penalty_mode }));
    if (params.energy_normalize_by_pixels !== undefined) setParams(prev => ({ ...prev, energy_normalize_by_pixels: params.energy_normalize_by_pixels }));

    // Unified Training Framework
    if (params.noise_process !== undefined) setParams(prev => ({ ...prev, noise_process: params.noise_process }));
    if (params.prediction_target !== undefined) setParams(prev => ({ ...prev, prediction_target: params.prediction_target }));
    if (params.strict_validation !== undefined) setParams(prev => ({ ...prev, strict_validation: params.strict_validation }));

    // ControlNet
    if (params.controlnet_type !== undefined) setControlnetType(params.controlnet_type as "standard" | "lllite");
    if (params.controlnet_pretrained_path !== undefined && params.controlnet_pretrained_path !== null) setControlnetPretrainedPath(params.controlnet_pretrained_path);
    if (params.controlnet_init_from_unet !== undefined) setControlnetInitFromUnet(params.controlnet_init_from_unet);
    if (params.lllite_conditioning_channels !== undefined) setLlliteConditioningChannels(params.lllite_conditioning_channels);
    if (params.lllite_rank !== undefined) setLlliteRank(params.lllite_rank);
    if (params.condition_preprocessors !== undefined && params.condition_preprocessors !== null) setConditionPreprocessors(params.condition_preprocessors);
    if (params.condition_cache_mode !== undefined) setConditionCacheMode(params.condition_cache_mode as "on_the_fly" | "pre_generate");

    // Sample
    if (params.sample_every !== undefined) setParams(prev => ({ ...prev, sample_every: params.sample_every }));
    if (params.sample_prompts && params.sample_prompts.length > 0) setParams(prev => ({ ...prev, sample_prompts: params.sample_prompts }));
    if (params.sample_width !== undefined) setParams(prev => ({ ...prev, sample_width: params.sample_width }));
    if (params.sample_height !== undefined) setParams(prev => ({ ...prev, sample_height: params.sample_height }));
    if (params.sample_steps !== undefined) setParams(prev => ({ ...prev, sample_steps: params.sample_steps }));
    if (params.sample_cfg_scale !== undefined) setParams(prev => ({ ...prev, sample_cfg_scale: params.sample_cfg_scale }));
    if (params.sample_sampler !== undefined) setParams(prev => ({ ...prev, sample_sampler: params.sample_sampler }));
    if (params.sample_schedule_type !== undefined) setParams(prev => ({ ...prev, sample_schedule_type: params.sample_schedule_type }));
    if (params.sample_seed !== undefined) setParams(prev => ({ ...prev, sample_seed: params.sample_seed }));

    // Debug
    if (params.debug_latents !== undefined) setParams(prev => ({ ...prev, debug_latents: params.debug_latents }));
    if (params.debug_latents_every !== undefined) setParams(prev => ({ ...prev, debug_latents_every: params.debug_latents_every }));

    // Bucketing
    if (params.enable_bucketing !== undefined) setParams(prev => ({ ...prev, enable_bucketing: params.enable_bucketing }));
    if (params.base_resolutions !== undefined && params.base_resolutions !== null) {
      setParams(prev => ({ ...prev, base_resolutions: params.base_resolutions }));
    } else if (params.base_resolutions === null) {
      setParams(prev => ({ ...prev, base_resolutions: [1024] }));
    }
    if (params.bucket_strategy !== undefined) setParams(prev => ({ ...prev, bucket_strategy: params.bucket_strategy }));
    if (params.multi_resolution_mode !== undefined) setParams(prev => ({ ...prev, multi_resolution_mode: params.multi_resolution_mode }));

    // Cache
    if (params.cache_latents_to_disk !== undefined) setParams(prev => ({ ...prev, cache_latents_to_disk: params.cache_latents_to_disk }));
    if (params.force_recache !== undefined) setParams(prev => ({ ...prev, force_recache: params.force_recache }));

    // Reference images / Vision encoder
    if (params.use_reference_images !== undefined) setParams(prev => ({ ...prev, use_reference_images: params.use_reference_images }));
    if (params.vision_encoder_path !== undefined) setParams(prev => ({ ...prev, vision_encoder_path: params.vision_encoder_path || "" }));
    if (params.train_vision_encoder !== undefined) setParams(prev => ({ ...prev, train_vision_encoder: params.train_vision_encoder }));
    if (params.gradient_routing_ve !== undefined) setParams(prev => ({ ...prev, gradient_routing_ve: params.gradient_routing_ve }));
    if (params.vision_encoder_lr !== undefined) {
      setParams(prev => ({ ...prev, vision_encoder_lr: params.vision_encoder_lr }));
      setLocalVisionEncoderLrText(params.vision_encoder_lr != null ? String(params.vision_encoder_lr) : "");
    }
    if (params.param_tracking !== undefined) setParams(prev => ({ ...prev, param_tracking: params.param_tracking }));
    if (params.param_tracking_interval !== undefined) setParams(prev => ({ ...prev, param_tracking_interval: params.param_tracking_interval }));

    // Priority training
    if (params.priority_training) {
      setPriorityEnabled(true);
      const entries = params.priority_training.entries || [];
      setPriorityText(entries.map((e: any) => typeof e === "string" ? e : JSON.stringify(e)).join("\n"));
      setPriorityMultiplier(params.priority_training.multiplier || 1);
    }

    // ReLoRA
    if (params.relora_merge_every !== undefined) setParams(prev => ({ ...prev, relora_merge_every: params.relora_merge_every }));
    if (params.relora_merge_unit !== undefined) setParams(prev => ({ ...prev, relora_merge_unit: params.relora_merge_unit }));
    if (params.restart_warmup_steps !== undefined) setParams(prev => ({ ...prev, restart_warmup_steps: params.restart_warmup_steps }));
    if (params.optimizer_reset_strategy !== undefined) setParams(prev => ({ ...prev, optimizer_reset_strategy: params.optimizer_reset_strategy }));
    if (params.optimizer_pruning_ratio !== undefined) setParams(prev => ({ ...prev, optimizer_pruning_ratio: params.optimizer_pruning_ratio }));
  }, []);

  // Load training run parameters for edit mode
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
      useFlashAttention,
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
    if (config.useFlashAttention !== undefined) updateParam("use_flash_attention", config.useFlashAttention);
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
    if (config.controlnetType !== undefined) setControlnetType(config.controlnetType);
    if (config.controlnetPretrainedPath !== undefined) setControlnetPretrainedPath(config.controlnetPretrainedPath);
    if (config.controlnetInitFromUnet !== undefined) setControlnetInitFromUnet(config.controlnetInitFromUnet);
    if (config.llliteConditioningChannels !== undefined) setLlliteConditioningChannels(config.llliteConditioningChannels);
    if (config.llliteRank !== undefined) setLlliteRank(config.llliteRank);
    if (config.conditionPreprocessors !== undefined) setConditionPreprocessors(config.conditionPreprocessors);
    if (config.conditionCacheMode !== undefined) setConditionCacheMode(config.conditionCacheMode);
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
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm">LoRA (Recommended)</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="training_method"
                value="full_finetune"
                checked={trainingMethod === "full_finetune"}
                onChange={() => setTrainingMethod("full_finetune")}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm">Full Fine-tune</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="training_method"
                value="controlnet"
                checked={trainingMethod === "controlnet"}
                onChange={() => setTrainingMethod("controlnet")}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm">ControlNet (SD1.5/SDXL)</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="training_method"
                value="relora"
                checked={trainingMethod === "relora"}
                onChange={() => setTrainingMethod("relora")}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm">ReLoRA (Periodic Merge + Reinit)</span>
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
          </div>

          <select
            value={baseModelPath}
            onChange={(e) => setBaseModelPath(e.target.value)}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
            required
          >
            <option value="">Select a model...</option>
            {filteredModels.map((model) => (
              <option key={model.path} value={model.path}>
                {model.name} ({model.architecture.toUpperCase()})
              </option>
            ))}
          </select>
          {availableModels.length === 0 && (
            <p className="text-xs text-gray-500 mt-1">No models available. Please add models to the models directory.</p>
          )}
          {filteredModels.length === 0 && availableModels.length > 0 && (
            <p className="text-xs text-gray-500 mt-1">No models match the selected filters.</p>
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
                onChange={(e) => setControlnetType(e.target.value as "standard" | "lllite")}
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
                    onChange={(e) => setControlnetInitFromUnet(e.target.checked)}
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
                onChange={(e) => setControlnetPretrainedPath(e.target.value)}
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
                    onChange={(e) => setLlliteConditioningChannels(parseInt(e.target.value) || 32)}
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
                    onChange={(e) => setLlliteRank(parseInt(e.target.value) || 64)}
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
                          setConditionPreprocessors([...conditionPreprocessors, pp]);
                        } else {
                          setConditionPreprocessors(conditionPreprocessors.filter(p => p !== pp));
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
                  onChange={(e) => setConditionCacheMode(e.target.value as "on_the_fly" | "pre_generate")}
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
                  disabled={isZImageModel(baseModelPath)}
                  className="w-4 h-4 disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <label htmlFor="train-text-encoder" className={`text-xs cursor-pointer ${isZImageModel(baseModelPath) ? 'text-gray-500' : 'text-gray-300'}`}>
                  Train Text Encoder {isZImageModel(baseModelPath) && '(Not supported for Z-Image)'}
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

            {/* Flash Attention */}
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="use-flash-attention"
                checked={useFlashAttention}
                onChange={(e) => updateParam("use_flash_attention", e.target.checked)}
                className="w-4 h-4"
              />
              <label htmlFor="use-flash-attention" className="text-xs text-gray-300 cursor-pointer">
                Flash Attention (faster training, lower memory)
              </label>
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
            </select>
          </div>

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
                <div className="grid grid-cols-2 gap-2">
                  {[
                    [256, 512, 768, 1024],
                    [1280, 1536, 1792, 2048],
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
