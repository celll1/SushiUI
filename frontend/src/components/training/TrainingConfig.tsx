"use client";

import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { createTrainingRun, listDatasets, Dataset, TrainingRun, getModels, DatasetConfigItem, getRandomCaption, getDatasetCaptionTypes, CaptionTypeInfo, getSamplers, getScheduleTypes } from "@/utils/api";

interface TrainingConfigProps {
  onClose: () => void;
  onRunCreated: (run: TrainingRun) => void;
}

interface DatasetConfig {
  dataset_id: number;
  caption_types: string[];
  filters: Record<string, any>;
}

export default function TrainingConfig({ onClose, onRunCreated }: TrainingConfigProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [runName, setRunName] = useState("");

  // Multiple datasets support
  const [datasetConfigs, setDatasetConfigs] = useState<DatasetConfig[]>([
    { dataset_id: 0, caption_types: [], filters: {} }
  ]);

  // Available caption types for each dataset
  const [datasetCaptionTypes, setDatasetCaptionTypes] = useState<Record<number, CaptionTypeInfo[]>>({});

  const [trainingMethod, setTrainingMethod] = useState<"lora" | "full_finetune">("lora");
  const [baseModelPath, setBaseModelPath] = useState("");

  // Training parameters
  const [useEpochs, setUseEpochs] = useState(false);
  const [totalSteps, setTotalSteps] = useState(1000);
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(1);
  const [learningRate, setLearningRate] = useState<string>("0.0001");
  const [lrScheduler, setLrScheduler] = useState("constant");
  const [optimizer, setOptimizer] = useState("adamw8bit");

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

  // Component-specific training
  const [trainUnet, setTrainUnet] = useState(true);
  const [trainTextEncoder, setTrainTextEncoder] = useState(false);
  const [unetLr, setUnetLr] = useState<string>("");
  const [textEncoderLr, setTextEncoderLr] = useState<string>("");
  const [textEncoder1Lr, setTextEncoder1Lr] = useState<string>("");
  const [textEncoder2Lr, setTextEncoder2Lr] = useState<string>("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // State for samplers and schedule types from API
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);

  useEffect(() => {
    loadDatasets();
    loadModels();
    loadSamplers();
    loadScheduleTypes();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await listDatasets();
      setDatasets(response.datasets);
      if (response.datasets.length > 0) {
        const firstDatasetId = response.datasets[0].id;
        // Initialize first dataset config with first available dataset
        setDatasetConfigs([{ dataset_id: firstDatasetId, caption_types: [], filters: {} }]);
        // Load caption types for the first dataset
        loadCaptionTypes(firstDatasetId);
      }
    } catch (err) {
      console.error("Failed to load datasets:", err);
    }
  };

  const loadModels = async () => {
    try {
      const response = await getModels();
      const models = response.models || [];
      // Extract paths from model objects
      const modelPaths = models.map((m: any) => m.path);
      setAvailableModels(modelPaths);
      if (modelPaths.length > 0) {
        setBaseModelPath(modelPaths[0]);
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

  // Helper function: Load caption types for a dataset
  const loadCaptionTypes = async (datasetId: number) => {
    if (datasetId === 0) return;

    try {
      const response = await getDatasetCaptionTypes(datasetId);
      setDatasetCaptionTypes(prev => ({
        ...prev,
        [datasetId]: response.caption_types
      }));
    } catch (err) {
      console.error("Failed to load caption types:", err);
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
      optimizer: optimizer,
      lora_rank: trainingMethod === "lora" ? loraRank : undefined,
      lora_alpha: trainingMethod === "lora" ? loraAlpha : undefined,
      save_every: saveEvery,
      save_every_unit: saveEveryUnit,
      sample_every: sampleEvery,
      sample_prompts: samplePrompts.filter(p => p.positive.trim() !== ""),
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
      train_unet: trainUnet,
      train_text_encoder: trainTextEncoder,
      unet_lr: unetLr ? parseFloat(unetLr) : null,
      text_encoder_lr: textEncoderLr ? parseFloat(textEncoderLr) : null,
      text_encoder_1_lr: textEncoder1Lr ? parseFloat(textEncoder1Lr) : null,
      text_encoder_2_lr: textEncoder2Lr ? parseFloat(textEncoder2Lr) : null,
    };

    console.log("[TrainingConfig] Request data:", requestData);

    try {
      const newRun = await createTrainingRun(requestData);
      console.log("[TrainingConfig] Training run created:", newRun);
      onRunCreated(newRun);
    } catch (err: any) {
      console.error("[TrainingConfig] Error details:", err);
      console.error("[TrainingConfig] Error response:", err.response);
      console.error("[TrainingConfig] Error data:", err.response?.data);
      const errorMessage = err.response?.data?.detail || err.response?.data?.message || err.message || "Failed to create training run";
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-4 border-b border-gray-700 flex items-center justify-between bg-gray-800/50 sticky top-0 z-10">
        <h2 className="text-lg font-semibold">New Training Run</h2>
        <button
          onClick={onClose}
          className="p-1.5 hover:bg-gray-700 rounded transition-colors"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      <form onSubmit={handleSubmit} className="p-4 space-y-4 max-w-2xl">
        {error && (
          <div className="bg-red-900/20 border border-red-500 text-red-400 rounded p-3 text-sm">
            {error}
          </div>
        )}

        {/* Run Name */}
        <div>
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
        <div className="border border-gray-700 rounded p-4 space-y-3">
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
                  // Load caption types for the new dataset
                  loadCaptionTypes(newDatasetId);
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

              {/* Caption Types */}
              {config.dataset_id !== 0 && datasetCaptionTypes[config.dataset_id] && (
                <div>
                  <label className="block text-xs text-gray-400 mb-1.5">
                    Caption Types (select which to use for training)
                  </label>
                  <div className="bg-gray-900 border border-gray-700 rounded p-2 space-y-1 max-h-40 overflow-y-auto">
                    {datasetCaptionTypes[config.dataset_id].length === 0 ? (
                      <p className="text-xs text-gray-500">No captions found in this dataset</p>
                    ) : (
                      datasetCaptionTypes[config.dataset_id].map((captionType) => (
                        <label key={captionType.caption_type} className="flex items-center space-x-2 cursor-pointer hover:bg-gray-800 p-1 rounded">
                          <input
                            type="checkbox"
                            checked={config.caption_types.includes(captionType.caption_type)}
                            onChange={(e) => {
                              const updated = [...datasetConfigs];
                              if (e.target.checked) {
                                updated[index].caption_types = [...updated[index].caption_types, captionType.caption_type];
                              } else {
                                updated[index].caption_types = updated[index].caption_types.filter(t => t !== captionType.caption_type);
                              }
                              setDatasetConfigs(updated);
                            }}
                            className="rounded"
                          />
                          <span className="text-xs flex-1">
                            {captionType.caption_type}
                            <span className="text-gray-500 ml-1">({captionType.total_count} items)</span>
                          </span>
                        </label>
                      ))
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Leave unchecked to use all caption types
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Training Method */}
        <div>
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
        <div>
          <label className="block text-sm font-medium mb-2">
            Base Model <span className="text-red-400">*</span>
          </label>
          <select
            value={baseModelPath}
            onChange={(e) => setBaseModelPath(e.target.value)}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
            required
          >
            <option value="">Select a model...</option>
            {availableModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
          {availableModels.length === 0 && (
            <p className="text-xs text-gray-500 mt-1">No models available. Please add models to the models directory.</p>
          )}
        </div>

        {/* LoRA Settings */}
        {trainingMethod === "lora" && (
          <div className="bg-gray-800/50 rounded-lg p-3 space-y-3">
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
        <div className="bg-gray-800/50 rounded-lg p-3 space-y-3">
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
                  min="100"
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

            <div>
              <label className="block text-xs text-gray-400 mb-1">Optimizer</label>
              <select
                value={optimizer}
                onChange={(e) => setOptimizer(e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="adamw8bit">AdamW 8-bit</option>
                <option value="adamw">AdamW</option>
                <option value="lion">Lion</option>
              </select>
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
                  className="w-4 h-4"
                />
                <label htmlFor="train-text-encoder" className="text-xs text-gray-300 cursor-pointer">
                  Train Text Encoder
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

        {/* Advanced Settings */}
        <div className="border border-gray-700 rounded p-4 space-y-3">
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
            <label className="block text-sm text-gray-400 mb-1.5">Resume from Checkpoint (Optional)</label>
            <select
              value={resumeFromCheckpoint || ""}
              onChange={(e) => setResumeFromCheckpoint(e.target.value || null)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
            >
              <option value="">Latest (Auto-detect)</option>
              {availableCheckpoints.map((ckpt) => (
                <option key={ckpt.filename} value={ckpt.filename}>
                  Step {ckpt.step} - {ckpt.filename}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Note: Checkpoints will be available after first training session
            </p>
          </div>
        </div>

        {/* Sample Generation */}
        <div className="border border-gray-700 rounded p-4 space-y-3">
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
                step="64"
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
                step="64"
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
        </div>

        {/* Buttons */}
        <div className="flex justify-end space-x-3 pt-4">
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
            {loading ? "Creating..." : "Create Training Run"}
          </button>
        </div>
      </form>
    </div>
  );
}
