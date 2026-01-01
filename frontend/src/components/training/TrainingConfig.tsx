"use client";

import { useState, useEffect } from "react";
import { X, Save, FolderOpen, Trash2 } from "lucide-react";
import { createTrainingRun, updateTrainingRun, listDatasets, Dataset, TrainingRun, getModels, DatasetConfigItem, getRandomCaption, getSamplers, getScheduleTypes, listTrainingPresets, createTrainingPreset, deleteTrainingPreset, TrainingPreset, getTrainingRunParams, updateTrainingConfig } from "@/utils/api";

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

export default function TrainingConfig({ onClose, onRunCreated, editRunId, onRunUpdated }: TrainingConfigProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [runName, setRunName] = useState("");

  // Model architecture filters
  const [showSD15, setShowSD15] = useState(true);
  const [showSDXL, setShowSDXL] = useState(true);
  const [showZImage, setShowZImage] = useState(true);

  // Multiple datasets support
  const [datasetConfigs, setDatasetConfigs] = useState<DatasetConfig[]>([
    { dataset_id: 0, caption_types: [], filters: {} }
  ]);

  // Available caption types for each dataset
  // Caption types selection moved to Dataset Management > Caption Processing page

  const [trainingMethod, setTrainingMethod] = useState<"lora" | "full_finetune">("lora");
  const [baseModelPath, setBaseModelPath] = useState("");

  // Training parameters
  const [useEpochs, setUseEpochs] = useState(false);
  const [totalSteps, setTotalSteps] = useState(1000);
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(4);
  const [learningRate, setLearningRate] = useState<string>("1e-5");
  const [lrScheduler, setLrScheduler] = useState("constant");
  const [optimizer, setOptimizer] = useState("adamw8bit");

  // Optimizer-specific options
  const [optimizerIsPaged, setOptimizerIsPaged] = useState(false);
  const [optimizerCautious, setOptimizerCautious] = useState(false);
  const [optimizerBeta1, setOptimizerBeta1] = useState<string>("0.9");
  const [optimizerBeta2, setOptimizerBeta2] = useState<string>("0.999");
  const [optimizerEpsilon, setOptimizerEpsilon] = useState<string>("1e-8");
  const [optimizerWeightDecay, setOptimizerWeightDecay] = useState<string>("0.01");
  const [optimizerScheduleFree, setOptimizerScheduleFree] = useState(false);
  const [optimizerScheduleFreeR, setOptimizerScheduleFreeR] = useState<string>("0.0");
  const [optimizerScheduleFreeWeightLrPower, setOptimizerScheduleFreeWeightLrPower] = useState<string>("2.0");

  // LoRA parameters
  const [loraRank, setLoraRank] = useState(16);
  const [loraAlpha, setLoraAlpha] = useState(16);

  // Advanced
  const [saveEvery, setSaveEvery] = useState(100);
  const [saveEveryUnit, setSaveEveryUnit] = useState<"steps" | "epochs">("steps");
  const [sampleEvery, setSampleEvery] = useState(100);
  const [resumeFromCheckpoint, setResumeFromCheckpoint] = useState<string | null>(null);
  const [availableCheckpoints, setAvailableCheckpoints] = useState<Array<{step: number, filename: string}>>([]);

  // Sample generation
  const [samplePrompts, setSamplePrompts] = useState<Array<{positive: string, negative: string}>>([
    { positive: "", negative: "" }
  ]);
  const [sampleWidth, setSampleWidth] = useState(1024);
  const [sampleHeight, setSampleHeight] = useState(1024);
  const [sampleSteps, setSampleSteps] = useState(28);
  const [sampleCfgScale, setSampleCfgScale] = useState(7.0);
  const [sampleSampler, setSampleSampler] = useState("euler");
  const [sampleScheduleType, setSampleScheduleType] = useState("uniform");
  const [sampleSeed, setSampleSeed] = useState(-1);

  // Debug options
  const [debugLatents, setDebugLatents] = useState(false);
  const [debugLatentsEvery, setDebugLatentsEvery] = useState(50);

  // Bucketing options
  const [enableBucketing, setEnableBucketing] = useState(false);
  const [baseResolutions, setBaseResolutions] = useState<number[]>([1024]);
  const [bucketStrategy, setBucketStrategy] = useState<"resize" | "crop" | "random_crop">("resize");
  const [multiResolutionMode, setMultiResolutionMode] = useState<"max" | "random">("max");
  const [cacheLatentsToDisk, setCacheLatentsToDisk] = useState(true);
  const [forceRecache, setForceRecache] = useState(false);

  // Component-specific training
  const [trainUnet, setTrainUnet] = useState(true);
  const [trainTextEncoder, setTrainTextEncoder] = useState(true);
  const [unetLr, setUnetLr] = useState<string>("1e-5");
  const [textEncoderLr, setTextEncoderLr] = useState<string>("1e-6");
  const [textEncoder1Lr, setTextEncoder1Lr] = useState<string>("");
  const [textEncoder2Lr, setTextEncoder2Lr] = useState<string>("");

  // Precision and dtype settings (VRAM optimization)
  const [weightDtype, setWeightDtype] = useState<string>("fp16");
  const [trainingDtype, setTrainingDtype] = useState<string>("fp16");
  const [outputDtype, setOutputDtype] = useState<string>("fp32");
  const [vaeDtype, setVaeDtype] = useState<string>("fp16");
  const [mixedPrecision, setMixedPrecision] = useState(true);
  const [useFlashAttention, setUseFlashAttention] = useState(false);
  const [minSnrGamma, setMinSnrGamma] = useState<number>(5.0);

  // Text encoding mode
  const [textEncodingMode, setTextEncodingMode] = useState<string>("swap_onthefly");
  const [textEncodingSwapInterval, setTextEncodingSwapInterval] = useState<number>(256);

  // Latent encoding mode
  const [latentEncodingMode, setLatentEncodingMode] = useState<string>("swap_onthefly");
  const [latentEncodingSwapInterval, setLatentEncodingSwapInterval] = useState<number>(256);

  // Block Swap settings (training VRAM optimization)
  const [blocksToSwap, setBlocksToSwap] = useState<number>(0);
  const [usePinnedMemory, setUsePinnedMemory] = useState<boolean>(false);
  const [numOptimizerGroups, setNumOptimizerGroups] = useState<number>(0);

  // Multi Noise-Timestep (MNT) settings
  const [multiNoiseTimesteps, setMultiNoiseTimesteps] = useState<number>(1);
  const [multiNoiseMode, setMultiNoiseMode] = useState<string>("independent");
  const [trajectoryBlendAlpha, setTrajectoryBlendAlpha] = useState<number>(0.7);
  const [timestepDistribution, setTimestepDistribution] = useState<string>("uniform");
  const [timestepMin, setTimestepMin] = useState<number>(0.0);
  const [timestepMax, setTimestepMax] = useState<number>(1.0);

  // Regularization settings (prevent overbaking)
  const [regularizationType, setRegularizationType] = useState<string>("none");  // Deprecated, kept for API compatibility
  const [snrRegularizationWeight, setSnrRegularizationWeight] = useState<number>(0.0);  // 0.0 = disabled
  const [snrTimestepAdaptive, setSnrTimestepAdaptive] = useState<boolean>(true);
  const [snrPenaltyMode, setSnrPenaltyMode] = useState<string>("relu");
  const [energyRegularizationWeight, setEnergyRegularizationWeight] = useState<number>(0.0);  // 0.0 = disabled
  const [energyTimestepAdaptive, setEnergyTimestepAdaptive] = useState<boolean>(true);
  const [energyPenaltyMode, setEnergyPenaltyMode] = useState<string>("under");  // Changed from "abs" to "under" (recommended)
  const [energyNormalizeByPixels, setEnergyNormalizeByPixels] = useState<boolean>(true);

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

  // Helper: Detect if model is Z-Image (architecture-based)
  const isZImageModel = (modelPath: string): boolean => {
    const model = availableModels.find(m => m.path === modelPath);
    return model?.architecture === "zimage";
  };

  // Filter models by architecture
  const filteredModels = availableModels.filter((model) => {
    if (model.architecture === "sd15" && !showSD15) return false;
    if (model.architecture === "sdxl" && !showSDXL) return false;
    if (model.architecture === "zimage" && !showZImage) return false;
    return true;
  });

  useEffect(() => {
    loadDatasets();
    loadModels();
    loadSamplers();
    loadScheduleTypes();
    loadPresets();
  }, []);

  // Load training run parameters when in edit mode
  useEffect(() => {
    if (editRunId) {
      loadTrainingRunParams(editRunId);
    }
  }, [editRunId]);

  // Auto-configure precision settings when model changes
  useEffect(() => {
    if (!baseModelPath) return;

    const isZImage = isZImageModel(baseModelPath);

    if (isZImage) {
      // Z-Image defaults: bf16 for weights/training/output, fp32 for VAE
      setWeightDtype("bf16");
      setTrainingDtype("bf16");
      setOutputDtype("bf16");
      setVaeDtype("fp32");
      // Z-Image: Cannot train text encoder (frozen)
      setTrainTextEncoder(false);
    } else {
      // SD/SDXL defaults: fp16 for all
      setWeightDtype("fp16");
      setTrainingDtype("fp16");
      setOutputDtype("fp16");
      setVaeDtype("fp16");
    }
  }, [baseModelPath]);

  // Reset optimizer hyperparameters when optimizer changes
  useEffect(() => {
    const config = OPTIMIZER_CONFIGS[optimizer];
    if (!config) return;

    const { beta1, beta2, epsilon, weight_decay } = config.defaults;
    if (beta1 !== undefined) setOptimizerBeta1(beta1);
    if (beta2 !== undefined) setOptimizerBeta2(beta2);
    if (epsilon !== undefined) setOptimizerEpsilon(epsilon);
    if (weight_decay !== undefined) setOptimizerWeightDecay(weight_decay);

    // Reset options that are not supported by the new optimizer
    if (!config.supportsPaged) setOptimizerIsPaged(false);
    if (!config.supportsCautious) setOptimizerCautious(false);
  }, [optimizer]);

  const loadDatasets = async () => {
    try {
      const response = await listDatasets();
      setDatasets(response.datasets);
      if (response.datasets.length > 0) {
        const firstDatasetId = response.datasets[0].id;
        // Initialize first dataset config with first available dataset
        setDatasetConfigs([{ dataset_id: firstDatasetId, caption_types: [], filters: {} }]);
      }
    } catch (err) {
      console.error("Failed to load datasets:", err);
    }
  };

  const loadModels = async () => {
    try {
      const response = await getModels();
      const models = response.models || [];
      setAvailableModels(models);
      if (models.length > 0) {
        setBaseModelPath(models[0].path);
      }
    } catch (err) {
      console.error("Failed to load models:", err);
    }
  };

  // Load training run parameters for edit mode
  const loadTrainingRunParams = async (runId: number) => {
    console.log(`[TrainingConfig] Loading parameters for training run ${runId}...`);
    try {
      const params = await getTrainingRunParams(runId);
      console.log(`[TrainingConfig] Received parameters:`, params);

      // Populate all form fields from loaded parameters
      setRunName(params.run_name || "");
      setBaseModelPath(params.base_model_path || "");
      setTrainingMethod(params.training_method || "lora");

      // Dataset configs
      if (params.dataset_configs) {
        setDatasetConfigs(params.dataset_configs);
      }

      // LoRA rank
      if (params.lora_rank !== undefined) setLoraRank(params.lora_rank);

      // Training parameters
      if (params.total_steps !== undefined) setTotalSteps(params.total_steps);
      if (params.learning_rate !== undefined) setLearningRate(params.learning_rate.toString());
      if (params.lr_scheduler !== undefined) setLrScheduler(params.lr_scheduler);
      if (params.lr_warmup_steps !== undefined) setLrWarmupSteps(params.lr_warmup_steps);
      if (params.optimizer !== undefined) setOptimizer(params.optimizer);

      // Optimizer parameters
      if (params.optimizer_beta1 !== undefined) setOptimizerBeta1(params.optimizer_beta1.toString());
      if (params.optimizer_beta2 !== undefined) setOptimizerBeta2(params.optimizer_beta2.toString());
      if (params.optimizer_epsilon !== undefined) setOptimizerEpsilon(params.optimizer_epsilon.toString());
      if (params.optimizer_weight_decay !== undefined) setOptimizerWeightDecay(params.optimizer_weight_decay.toString());
      if (params.optimizer_is_paged !== undefined) setOptimizerIsPaged(params.optimizer_is_paged);
      if (params.optimizer_cautious !== undefined) setOptimizerCautious(params.optimizer_cautious);
      if (params.optimizer_schedule_free !== undefined) setOptimizerScheduleFree(params.optimizer_schedule_free);
      if (params.optimizer_schedule_free_r !== undefined) setOptimizerScheduleFreeR(params.optimizer_schedule_free_r);
      if (params.optimizer_schedule_free_weight_lr_power !== undefined) setOptimizerScheduleFreeWeightLrPower(params.optimizer_schedule_free_weight_lr_power);

      // Precision settings
      if (params.weight_dtype !== undefined) setWeightDtype(params.weight_dtype);
      if (params.training_dtype !== undefined) setTrainingDtype(params.training_dtype);
      if (params.output_dtype !== undefined) setOutputDtype(params.output_dtype);
      if (params.vae_dtype !== undefined) setVaeDtype(params.vae_dtype);
      if (params.train_text_encoder !== undefined) setTrainTextEncoder(params.train_text_encoder);

      // Memory optimization
      if (params.text_encoding_mode !== undefined) setTextEncodingMode(params.text_encoding_mode);
      if (params.text_encoding_swap_interval !== undefined) setTextEncodingSwapInterval(params.text_encoding_swap_interval);
      if (params.latent_encoding_mode !== undefined) setLatentEncodingMode(params.latent_encoding_mode);
      if (params.latent_encoding_swap_interval !== undefined) setLatentEncodingSwapInterval(params.latent_encoding_swap_interval);
      if (params.blocks_to_swap !== undefined) setBlocksToSwap(params.blocks_to_swap);
      if (params.use_pinned_memory !== undefined) setUsePinnedMemory(params.use_pinned_memory);
      if (params.num_optimizer_groups !== undefined) setNumOptimizerGroups(params.num_optimizer_groups);

      // MNT settings
      if (params.multi_noise_timesteps !== undefined) setMultiNoiseTimesteps(params.multi_noise_timesteps);
      if (params.multi_noise_mode !== undefined) setMultiNoiseMode(params.multi_noise_mode);
      if (params.trajectory_blend_alpha !== undefined) setTrajectoryBlendAlpha(params.trajectory_blend_alpha);
      if (params.timestep_sampling) {
        if (params.timestep_sampling.distribution !== undefined) setTimestepDistribution(params.timestep_sampling.distribution);
        if (params.timestep_sampling.min_timestep !== undefined) setTimestepMin(params.timestep_sampling.min_timestep);
        if (params.timestep_sampling.max_timestep !== undefined) setTimestepMax(params.timestep_sampling.max_timestep);
      }

      // Regularization
      if (params.snr_regularization_weight !== undefined) setSnrRegularizationWeight(params.snr_regularization_weight);
      if (params.snr_timestep_adaptive !== undefined) setSnrTimestepAdaptive(params.snr_timestep_adaptive);
      if (params.snr_penalty_mode !== undefined) setSnrPenaltyMode(params.snr_penalty_mode);
      if (params.energy_regularization_weight !== undefined) setEnergyRegularizationWeight(params.energy_regularization_weight);
      if (params.energy_timestep_adaptive !== undefined) setEnergyTimestepAdaptive(params.energy_timestep_adaptive);
      if (params.energy_penalty_mode !== undefined) setEnergyPenaltyMode(params.energy_penalty_mode);

      // Sample Generation
      if (params.sample_every !== undefined) setSampleEvery(params.sample_every);
      if (params.sample_prompts && params.sample_prompts.length > 0) {
        setSamplePrompts(params.sample_prompts);
      }
      if (params.sample_width !== undefined) setSampleWidth(params.sample_width);
      if (params.sample_height !== undefined) setSampleHeight(params.sample_height);
      if (params.sample_steps !== undefined) setSampleSteps(params.sample_steps);
      if (params.sample_cfg_scale !== undefined) setSampleCfgScale(params.sample_cfg_scale);
      if (params.sample_sampler !== undefined) setSampleSampler(params.sample_sampler);
      if (params.sample_schedule_type !== undefined) setSampleScheduleType(params.sample_schedule_type);
      if (params.sample_seed !== undefined) setSampleSeed(params.sample_seed);

      // Debug Latents
      if (params.debug_latents !== undefined) setDebugLatents(params.debug_latents);
      if (params.debug_latents_every !== undefined) setDebugLatentsEvery(params.debug_latents_every);

      // Bucketing
      if (params.enable_bucketing !== undefined) setEnableBucketing(params.enable_bucketing);
      if (params.base_resolutions !== undefined) setBaseResolutions(params.base_resolutions);
      if (params.bucket_strategy !== undefined) setBucketStrategy(params.bucket_strategy);
      if (params.multi_resolution_mode !== undefined) setMultiResolutionMode(params.multi_resolution_mode);

      // Cache
      if (params.cache_latents_to_disk !== undefined) setCacheLatentsToDisk(params.cache_latents_to_disk);

      console.log(`[TrainingConfig] Successfully loaded all parameters for training run ${runId}`);
      console.log(`[TrainingConfig] Sample prompts restored:`, params.sample_prompts);
      console.log(`[TrainingConfig] MNT mode restored:`, params.multi_noise_mode);
    } catch (err) {
      console.error("[TrainingConfig] Failed to load training run parameters:", err);
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
      updated[promptIndex].positive = response.caption;
      setSamplePrompts(updated);
    } catch (err) {
      console.error("Failed to get random caption:", err);
      setError("Failed to get random caption from dataset");
    }
  };

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
        if (params.width) setSampleWidth(params.width);
        if (params.height) setSampleHeight(params.height);
        if (params.steps) setSampleSteps(params.steps);
        if (params.cfg_scale) setSampleCfgScale(params.cfg_scale);
        if (params.sampler) setSampleSampler(params.sampler);
        if (params.schedule_type) setSampleScheduleType(params.schedule_type);
        if (params.seed) setSampleSeed(params.seed);
      }
    } catch (err) {
      console.error("Failed to import from generation panel:", err);
    }
  };

  // Get current config (excluding dataset and model path)
  const getCurrentConfig = () => {
    return {
      useEpochs,
      totalSteps,
      epochs,
      batchSize,
      learningRate,
      lrScheduler,
      optimizer,
      optimizerIsPaged,
      optimizerCautious,
      optimizerBeta1,
      optimizerBeta2,
      optimizerEpsilon,
      optimizerWeightDecay,
      optimizerScheduleFree,
      optimizerScheduleFreeR,
      optimizerScheduleFreeWeightLrPower,
      loraRank,
      loraAlpha,
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
    if (config.totalSteps !== undefined) setTotalSteps(config.totalSteps);
    if (config.epochs !== undefined) setEpochs(config.epochs);
    if (config.batchSize !== undefined) setBatchSize(config.batchSize);
    if (config.learningRate !== undefined) setLearningRate(config.learningRate);
    if (config.lrScheduler !== undefined) setLrScheduler(config.lrScheduler);
    if (config.optimizer !== undefined) setOptimizer(config.optimizer);
    if (config.optimizerIsPaged !== undefined) setOptimizerIsPaged(config.optimizerIsPaged);
    if (config.optimizerCautious !== undefined) setOptimizerCautious(config.optimizerCautious);
    if (config.optimizerBeta1 !== undefined) setOptimizerBeta1(config.optimizerBeta1);
    if (config.optimizerBeta2 !== undefined) setOptimizerBeta2(config.optimizerBeta2);
    if (config.optimizerEpsilon !== undefined) setOptimizerEpsilon(config.optimizerEpsilon);
    if (config.optimizerWeightDecay !== undefined) setOptimizerWeightDecay(config.optimizerWeightDecay);
    if (config.optimizerScheduleFree !== undefined) setOptimizerScheduleFree(config.optimizerScheduleFree);
    if (config.optimizerScheduleFreeR !== undefined) setOptimizerScheduleFreeR(config.optimizerScheduleFreeR);
    if (config.optimizerScheduleFreeWeightLrPower !== undefined) setOptimizerScheduleFreeWeightLrPower(config.optimizerScheduleFreeWeightLrPower);
    if (config.loraRank !== undefined) setLoraRank(config.loraRank);
    if (config.loraAlpha !== undefined) setLoraAlpha(config.loraAlpha);
    if (config.saveEvery !== undefined) setSaveEvery(config.saveEvery);
    if (config.saveEveryUnit !== undefined) setSaveEveryUnit(config.saveEveryUnit);
    if (config.sampleEvery !== undefined) setSampleEvery(config.sampleEvery);
    if (config.resumeFromCheckpoint !== undefined) setResumeFromCheckpoint(config.resumeFromCheckpoint);
    if (config.samplePrompts !== undefined) setSamplePrompts(config.samplePrompts);
    if (config.sampleWidth !== undefined) setSampleWidth(config.sampleWidth);
    if (config.sampleHeight !== undefined) setSampleHeight(config.sampleHeight);
    if (config.sampleSteps !== undefined) setSampleSteps(config.sampleSteps);
    if (config.sampleCfgScale !== undefined) setSampleCfgScale(config.sampleCfgScale);
    if (config.sampleSampler !== undefined) setSampleSampler(config.sampleSampler);
    if (config.sampleScheduleType !== undefined) setSampleScheduleType(config.sampleScheduleType);
    if (config.sampleSeed !== undefined) setSampleSeed(config.sampleSeed);
    if (config.debugLatents !== undefined) setDebugLatents(config.debugLatents);
    if (config.debugLatentsEvery !== undefined) setDebugLatentsEvery(config.debugLatentsEvery);
    if (config.enableBucketing !== undefined) setEnableBucketing(config.enableBucketing);
    if (config.baseResolutions !== undefined) setBaseResolutions(config.baseResolutions);
    if (config.bucketStrategy !== undefined) setBucketStrategy(config.bucketStrategy);
    if (config.multiResolutionMode !== undefined) setMultiResolutionMode(config.multiResolutionMode);
    if (config.cacheLatentsToDisk !== undefined) setCacheLatentsToDisk(config.cacheLatentsToDisk);
    if (config.forceRecache !== undefined) setForceRecache(config.forceRecache);
    if (config.trainUnet !== undefined) setTrainUnet(config.trainUnet);
    if (config.trainTextEncoder !== undefined) setTrainTextEncoder(config.trainTextEncoder);
    if (config.unetLr !== undefined) setUnetLr(config.unetLr);
    if (config.textEncoderLr !== undefined) setTextEncoderLr(config.textEncoderLr);
    if (config.textEncoder1Lr !== undefined) setTextEncoder1Lr(config.textEncoder1Lr);
    if (config.textEncoder2Lr !== undefined) setTextEncoder2Lr(config.textEncoder2Lr);
    if (config.weightDtype !== undefined) setWeightDtype(config.weightDtype);
    if (config.trainingDtype !== undefined) setTrainingDtype(config.trainingDtype);
    if (config.outputDtype !== undefined) setOutputDtype(config.outputDtype);
    if (config.vaeDtype !== undefined) setVaeDtype(config.vaeDtype);
    if (config.mixedPrecision !== undefined) setMixedPrecision(config.mixedPrecision);
    if (config.useFlashAttention !== undefined) setUseFlashAttention(config.useFlashAttention);
    if (config.minSnrGamma !== undefined) setMinSnrGamma(config.minSnrGamma);
    if (config.textEncodingMode !== undefined) setTextEncodingMode(config.textEncodingMode);
    if (config.textEncodingSwapInterval !== undefined) setTextEncodingSwapInterval(config.textEncodingSwapInterval);
    if (config.latentEncodingMode !== undefined) setLatentEncodingMode(config.latentEncodingMode);
    if (config.latentEncodingSwapInterval !== undefined) setLatentEncodingSwapInterval(config.latentEncodingSwapInterval);
    if (config.blocksToSwap !== undefined) setBlocksToSwap(config.blocksToSwap);
    if (config.usePinnedMemory !== undefined) setUsePinnedMemory(config.usePinnedMemory);
    if (config.numOptimizerGroups !== undefined) setNumOptimizerGroups(config.numOptimizerGroups);
    if (config.multiNoiseTimesteps !== undefined) setMultiNoiseTimesteps(config.multiNoiseTimesteps);
    if (config.timestepDistribution !== undefined) setTimestepDistribution(config.timestepDistribution);
    if (config.timestepMin !== undefined) setTimestepMin(config.timestepMin);
    if (config.timestepMax !== undefined) setTimestepMax(config.timestepMax);

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

    // Validate that at least one component is being trained
    if (!trainUnet && !trainTextEncoder) {
      setError("At least one component (U-Net or Text Encoder) must be trained");
      return;
    }

    setLoading(true);
    setError(null);

    const requestData = {
      dataset_configs: datasetConfigs.filter(c => c.dataset_id !== 0),
      run_name: runName.trim() || undefined,  // Send undefined if empty (backend will auto-generate)
      training_method: trainingMethod,
      base_model_path: baseModelPath.trim(),
      total_steps: useEpochs ? undefined : totalSteps,
      epochs: useEpochs ? epochs : undefined,
      batch_size: batchSize,
      learning_rate: parseFloat(learningRate),
      lr_scheduler: lrScheduler,
      lr_warmup_steps: lrWarmupSteps,
      optimizer: optimizer,
      optimizer_is_paged: optimizerIsPaged,
      optimizer_cautious: optimizerCautious,
      optimizer_beta1: optimizerBeta1 ? parseFloat(optimizerBeta1) : undefined,
      optimizer_beta2: optimizerBeta2 ? parseFloat(optimizerBeta2) : undefined,
      optimizer_epsilon: optimizerEpsilon ? parseFloat(optimizerEpsilon) : undefined,
      optimizer_weight_decay: optimizerWeightDecay ? parseFloat(optimizerWeightDecay) : undefined,
      optimizer_schedule_free: optimizerScheduleFree,
      optimizer_schedule_free_r: optimizerScheduleFreeR ? parseFloat(optimizerScheduleFreeR) : 0.0,
      optimizer_schedule_free_weight_lr_power: optimizerScheduleFreeWeightLrPower ? parseFloat(optimizerScheduleFreeWeightLrPower) : 2.0,
      lora_rank: trainingMethod === "lora" ? loraRank : undefined,
      lora_alpha: trainingMethod === "lora" ? loraAlpha : undefined,
      save_every: saveEvery,
      save_every_unit: saveEveryUnit,
      sample_every: sampleEvery,
      sample_prompts: samplePrompts,  // Allow empty prompts (SD/SDXL/Z-Image can generate with empty prompts)
      sample_width: sampleWidth,
      sample_height: sampleHeight,
      sample_steps: sampleSteps,
      sample_cfg_scale: sampleCfgScale,
      sample_sampler: sampleSampler,
      sample_schedule_type: sampleScheduleType,
      sample_seed: sampleSeed,
      resume_from_checkpoint: resumeFromCheckpoint || undefined,
      debug_latents: debugLatents,
      debug_latents_every: debugLatentsEvery,
      enable_bucketing: enableBucketing,
      base_resolutions: enableBucketing ? baseResolutions : undefined,
      bucket_strategy: enableBucketing ? bucketStrategy : undefined,
      multi_resolution_mode: enableBucketing ? multiResolutionMode : undefined,
      cache_latents_to_disk: cacheLatentsToDisk,
      force_recache: forceRecache,
      train_unet: trainUnet,
      train_text_encoder: trainTextEncoder,
      unet_lr: unetLr ? parseFloat(unetLr) : null,
      text_encoder_lr: textEncoderLr ? parseFloat(textEncoderLr) : null,
      text_encoder_1_lr: textEncoder1Lr ? parseFloat(textEncoder1Lr) : null,
      text_encoder_2_lr: textEncoder2Lr ? parseFloat(textEncoder2Lr) : null,
      weight_dtype: weightDtype,
      training_dtype: trainingDtype,
      output_dtype: outputDtype,
      vae_dtype: vaeDtype,
      mixed_precision: mixedPrecision,
      use_flash_attention: useFlashAttention,
      min_snr_gamma: minSnrGamma,
      text_encoding_mode: textEncodingMode,
      text_encoding_swap_interval: textEncodingSwapInterval,
      latent_encoding_mode: latentEncodingMode,
      latent_encoding_swap_interval: latentEncodingSwapInterval,
      blocks_to_swap: blocksToSwap,
      use_pinned_memory: usePinnedMemory,
      num_optimizer_groups: numOptimizerGroups,
      multi_noise_timesteps: multiNoiseTimesteps,
      multi_noise_mode: multiNoiseMode,
      trajectory_blend_alpha: trajectoryBlendAlpha,
      timestep_sampling: {
        distribution: timestepDistribution,
        min_timestep: timestepMin,
        max_timestep: timestepMax,
      },
      // Regularization settings
      regularization_type: regularizationType !== "none" ? regularizationType : null,
      snr_regularization_weight: snrRegularizationWeight,
      snr_timestep_adaptive: snrTimestepAdaptive,
      snr_penalty_mode: snrPenaltyMode,
      energy_regularization_weight: energyRegularizationWeight,
      energy_timestep_adaptive: energyTimestepAdaptive,
      energy_penalty_mode: energyPenaltyMode,
      energy_normalize_by_pixels: energyNormalizeByPixels,
    };

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
      <div className="p-4 border-b border-gray-700 flex items-center justify-between bg-gray-800/50 sticky top-0 z-10">
        <h2 className="text-lg font-semibold">{editRunId ? "Edit Training Run" : "New Training Run"}</h2>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowLoadPresetDialog(true)}
            className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-sm transition-colors"
          >
            <FolderOpen className="h-4 w-4" />
            Load Preset
          </button>
          <button
            type="button"
            onClick={() => setShowPresetDialog(true)}
            className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-500 rounded text-sm transition-colors"
          >
            <Save className="h-4 w-4" />
            Save Preset
          </button>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-gray-700 rounded transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="p-4">
        {error && (
          <div className="bg-red-900/20 border border-red-500 text-red-400 rounded p-3 text-sm mb-4">
            {error}
          </div>
        )}

        <div className="columns-1 lg:columns-2 gap-4 space-y-4">
        {/* Run Name */}
        <div className="break-inside-avoid">
          <label className="block text-sm font-medium mb-2">
            Run Name <span className="text-gray-500 text-xs font-normal">(optional, auto-generated if empty)</span>
          </label>
          <input
            type="text"
            value={runName}
            onChange={(e) => setRunName(e.target.value)}
            placeholder="Leave empty for auto-generated name (e.g., 20251130_174523_a1b2c3d4)"
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
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

        {/* LoRA Settings */}
        {trainingMethod === "lora" && (
          <div className="break-inside-avoid bg-gray-800/50 rounded-lg p-3 space-y-3">
            <h3 className="text-sm font-semibold">LoRA Settings</h3>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Rank</label>
                <input
                  type="number"
                  value={loraRank}
                  onChange={(e) => setLoraRank(parseInt(e.target.value))}
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
                  onChange={(e) => setLoraAlpha(parseInt(e.target.value))}
                  min="1"
                  max="256"
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
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
                  onChange={(e) => setTotalSteps(parseInt(e.target.value))}
                  min="1"
                  max="50000"
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
              </div>
            ) : (
              <div>
                <label className="block text-xs text-gray-400 mb-1">Epochs</label>
                <input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  min="1"
                  max="1000"
                  className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
              </div>
            )}

            <div>
              <label className="block text-xs text-gray-400 mb-1">Batch Size</label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
                min="1"
                max="16"
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
                onChange={(e) => setMultiNoiseTimesteps(parseInt(e.target.value))}
                min="1"
                max="10"
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
                onChange={(e) => setMultiNoiseMode(e.target.value)}
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
                  onChange={(e) => setTrajectoryBlendAlpha(parseFloat(e.target.value))}
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
                    <option value="normal">Normal (Gaussian)</option>
                    <option value="lognormal">Log-Normal</option>
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
                      onChange={(e) => setTimestepMin(parseFloat(e.target.value))}
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
                      onChange={(e) => setTimestepMax(parseFloat(e.target.value))}
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
                      onChange={(e) => setSnrRegularizationWeight(e.target.checked ? 0.1 : 0.0)}
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
                      onChange={(e) => setSnrRegularizationWeight(parseFloat(e.target.value))}
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
                      onChange={(e) => setSnrTimestepAdaptive(e.target.checked)}
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
                      onChange={(e) => setSnrPenaltyMode(e.target.value)}
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
                      onChange={(e) => setEnergyRegularizationWeight(e.target.checked ? 0.1 : 0.0)}
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
                      onChange={(e) => setEnergyRegularizationWeight(parseFloat(e.target.value))}
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
                      onChange={(e) => setEnergyTimestepAdaptive(e.target.checked)}
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
                      onChange={(e) => setEnergyPenaltyMode(e.target.value)}
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
                      onChange={(e) => setEnergyNormalizeByPixels(e.target.checked)}
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

            <div>
              <label className="block text-xs text-gray-400 mb-1">Learning Rate</label>
              <input
                type="text"
                value={learningRate}
                onChange={(e) => setLearningRate(e.target.value)}
                placeholder="e.g., 1e-4"
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">LR Scheduler</label>
              <select
                value={lrScheduler}
                onChange={(e) => setLrScheduler(e.target.value)}
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
                  onChange={(e) => setOptimizer(e.target.value)}
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
                      onChange={(e) => setOptimizerIsPaged(e.target.checked)}
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
                      onChange={(e) => setOptimizerCautious(e.target.checked)}
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
                        onChange={(e) => setOptimizerScheduleFree(e.target.checked)}
                        className="w-4 h-4"
                      />
                      <label htmlFor="optimizer-schedule-free" className="text-xs text-gray-300 cursor-pointer">
                        Schedule-Free (learning rate scheduling)
                      </label>
                    </div>

                    {optimizerScheduleFree && (
                      <div className="ml-6 space-y-2 border-l-2 border-gray-600 pl-3">
                        {/* Schedule-Free r */}
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">r (warmup parameter)</label>
                          <input
                            type="text"
                            value={optimizerScheduleFreeR}
                            onChange={(e) => setOptimizerScheduleFreeR(e.target.value)}
                            className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs focus:outline-none focus:border-blue-500"
                          />
                          <p className="text-xs text-gray-500 mt-1">Default: 0.0 (no warmup)</p>
                        </div>

                        {/* Schedule-Free weight_lr_power */}
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Weight LR Power</label>
                          <input
                            type="text"
                            value={optimizerScheduleFreeWeightLrPower}
                            onChange={(e) => setOptimizerScheduleFreeWeightLrPower(e.target.value)}
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
                      if (beta1 !== undefined) setOptimizerBeta1(beta1);
                      if (beta2 !== undefined) setOptimizerBeta2(beta2);
                      if (epsilon !== undefined) setOptimizerEpsilon(epsilon);
                      if (weight_decay !== undefined) setOptimizerWeightDecay(weight_decay);
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
                        onChange={(e) => setOptimizerBeta1(e.target.value)}
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
                        onChange={(e) => setOptimizerBeta2(e.target.value)}
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
                        onChange={(e) => setOptimizerEpsilon(e.target.value)}
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
                      onChange={(e) => setOptimizerWeightDecay(e.target.value)}
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
                  onChange={(e) => setTrainUnet(e.target.checked)}
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
                  onChange={(e) => setTrainTextEncoder(e.target.checked)}
                  disabled={isZImageModel(baseModelPath)}
                  className="w-4 h-4 disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <label htmlFor="train-text-encoder" className={`text-xs cursor-pointer ${isZImageModel(baseModelPath) ? 'text-gray-500' : 'text-gray-300'}`}>
                  Train Text Encoder {isZImageModel(baseModelPath) && '(Not supported for Z-Image)'}
                </label>
              </div>
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
                  onChange={(e) => setUnetLr(e.target.value)}
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
                    onChange={(e) => setTextEncoderLr(e.target.value)}
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
                        onChange={(e) => setTextEncoder1Lr(e.target.value)}
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
                        onChange={(e) => setTextEncoder2Lr(e.target.value)}
                        placeholder={`Default: ${textEncoderLr || learningRate}`}
                        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}
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
                onChange={(e) => setWeightDtype(e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="fp16">FP16 (Default)</option>
                <option value="fp32">FP32 (Higher precision)</option>
                <option value="bf16">BF16 (Balanced)</option>
                <option value="fp8_e4m3fn">FP8 E4M3FN (~50% VRAM)</option>
                <option value="fp8_e5m2">FP8 E5M2 (~50% VRAM)</option>
              </select>
            </div>

            {/* Training/Activation dtype */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">Training dtype</label>
              <select
                value={trainingDtype}
                onChange={(e) => setTrainingDtype(e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="fp16">FP16 (Default)</option>
                <option value="bf16">BF16</option>
                <option value="fp8_e4m3fn">FP8 E4M3FN</option>
                <option value="fp8_e5m2">FP8 E5M2</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            {/* Output dtype */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">Output dtype</label>
              <select
                value={outputDtype}
                onChange={(e) => setOutputDtype(e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="fp32">FP32 (Default, highest precision)</option>
                <option value="fp16">FP16</option>
                <option value="bf16">BF16</option>
                <option value="fp8_e4m3fn">FP8 E4M3FN</option>
                <option value="fp8_e5m2">FP8 E5M2</option>
              </select>
            </div>

            {/* VAE dtype */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">VAE dtype</label>
              <select
                value={vaeDtype}
                onChange={(e) => setVaeDtype(e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="fp16">FP16 (Default, SDXL VAE works fine)</option>
                <option value="fp32">FP32 (Higher precision)</option>
                <option value="bf16">BF16 (Balanced)</option>
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
                onChange={(e) => setMixedPrecision(e.target.checked)}
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
                onChange={(e) => setUseFlashAttention(e.target.checked)}
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
                onChange={(e) => setMinSnrGamma(parseFloat(e.target.value) || 0)}
                step={0.5}
                min={0}
                max={20}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
              />
              <p className="text-xs text-gray-500">
                Default: 5.0. Set to 0 to disable. Prevents overfitting to high-noise timesteps.
              </p>
            </div>
          </div>

          <p className="text-xs text-gray-500">
            Lower precision dtypes reduce VRAM usage. FP8 can save ~50% VRAM. Use FP32 output for best loss calculation accuracy. Flash Attention improves training speed and reduces memory usage. Min-SNR gamma reweights loss to balance learning across all timesteps.
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
                onChange={(e) => setBlocksToSwap(parseInt(e.target.value) || 0)}
                min={0}
                max={29}
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
                  onChange={(e) => setUsePinnedMemory(e.target.checked)}
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
                  onChange={(e) => setNumOptimizerGroups(parseInt(e.target.value) || 0)}
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
              onChange={(e) => setTextEncodingMode(e.target.value)}
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
                onChange={(e) => setTextEncodingSwapInterval(parseInt(e.target.value) || 256)}
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

        {/* Latent Encoding Mode */}
        <div className="border border-gray-700 rounded p-4 space-y-3">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Latent Encoding Mode (VAE)</h3>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Encoding Mode</label>
            <select
              value={latentEncodingMode}
              onChange={(e) => setLatentEncodingMode(e.target.value)}
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
                onChange={(e) => setLatentEncodingSwapInterval(parseInt(e.target.value) || 256)}
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
                  onChange={() => setSaveEveryUnit("steps")}
                  className="text-blue-500 focus:ring-blue-500"
                />
                <span className="text-sm">Steps</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  checked={saveEveryUnit === "epochs"}
                  onChange={() => setSaveEveryUnit("epochs")}
                  className="text-blue-500 focus:ring-blue-500"
                />
                <span className="text-sm">Epochs</span>
              </label>
            </div>
            <input
              type="number"
              min="1"
              value={saveEvery}
              onChange={(e) => setSaveEvery(parseInt(e.target.value))}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
              placeholder={saveEveryUnit === "steps" ? "e.g., 100" : "e.g., 1"}
            />
          </div>

          {/* Resume from Checkpoint */}
          <div>
            <label className="block text-sm text-gray-400 mb-1.5">Resume from Checkpoint</label>
            <select
              value={resumeFromCheckpoint || ""}
              onChange={(e) => setResumeFromCheckpoint(e.target.value || null)}
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
              onChange={(e) => setSampleEvery(parseInt(e.target.value))}
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
              <label className="block text-xs text-gray-400 mb-1">Width</label>
              <input
                type="number"
                min="512"
                max="2048"
                step="8"
                value={sampleWidth}
                onChange={(e) => setSampleWidth(parseInt(e.target.value))}
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
                onChange={(e) => setSampleHeight(parseInt(e.target.value))}
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
                onChange={(e) => setSampleSteps(parseInt(e.target.value))}
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
                onChange={(e) => setSampleCfgScale(parseFloat(e.target.value))}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Sampler</label>
              <select
                value={sampleSampler}
                onChange={(e) => setSampleSampler(e.target.value)}
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
                onChange={(e) => setSampleScheduleType(e.target.value)}
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
              onChange={(e) => setSampleSeed(parseInt(e.target.value))}
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
              onChange={(e) => setDebugLatents(e.target.checked)}
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
                onChange={(e) => setDebugLatentsEvery(parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
                placeholder="e.g., 50"
              />
              <p className="text-xs text-gray-500 mt-1">
                Saves noisy latents, predicted latents, and timestep info to debug/ folder for debugging training issues
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
              onChange={(e) => setEnableBucketing(e.target.checked)}
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
                    [256, 512, 768],
                    [1024, 1280, 1536],
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
                                setBaseResolutions([...baseResolutions, res].sort((a, b) => a - b));
                              } else {
                                // Prevent unchecking the last resolution
                                if (baseResolutions.length > 1) {
                                  setBaseResolutions(baseResolutions.filter(r => r !== res));
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
                    onChange={(e) => setMultiResolutionMode(e.target.value as "max" | "random")}
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
                  onChange={(e) => setBucketStrategy(e.target.value as "resize" | "crop" | "random_crop")}
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
              onChange={(e) => setCacheLatentsToDisk(e.target.checked)}
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
              onChange={(e) => setForceRecache(e.target.checked)}
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
        <div className="flex justify-end space-x-3 pt-4 mt-4">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
            disabled={loading}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={loading}
          >
            {loading ? (editRunId ? "Updating..." : "Creating...") : (editRunId ? "Update Training Run" : "Create Training Run")}
          </button>
        </div>
      </form>

      {/* Save Preset Dialog */}
      {showPresetDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Save Training Preset</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-300 mb-1">Preset Name *</label>
                <input
                  type="text"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500"
                  placeholder="e.g., SDXL LoRA Quick"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">Description (Optional)</label>
                <textarea
                  value={presetDescription}
                  onChange={(e) => setPresetDescription(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500"
                  rows={3}
                  placeholder="Describe this preset..."
                />
              </div>
              <div className="flex gap-2 justify-end">
                <button
                  type="button"
                  onClick={() => setShowPresetDialog(false)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSavePreset}
                  className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded text-sm transition-colors"
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
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
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
                            {preset.training_method === "lora" ? "LoRA" : "Full Finetune"}
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
