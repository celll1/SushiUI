"use client";

import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { createTrainingRun, listDatasets, Dataset, TrainingRun, getModels, DatasetConfigItem, getRandomCaption } from "@/utils/api";

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

  const [trainingMethod, setTrainingMethod] = useState<"lora" | "full_finetune">("lora");
  const [baseModelPath, setBaseModelPath] = useState("");

  // Training parameters
  const [useEpochs, setUseEpochs] = useState(false);
  const [totalSteps, setTotalSteps] = useState(1000);
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(1);
  const [learningRate, setLearningRate] = useState(0.0001);
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
  const [sampleScheduleType, setSampleScheduleType] = useState("sgm_uniform");
  const [sampleSeed, setSampleSeed] = useState(-1);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDatasets();
    loadModels();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await listDatasets();
      setDatasets(response.datasets);
      if (response.datasets.length > 0) {
        // Initialize first dataset config with first available dataset
        setDatasetConfigs([{ dataset_id: response.datasets[0].id, caption_types: [], filters: {} }]);
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
      learning_rate: learningRate,
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
                  const updated = [...datasetConfigs];
                  updated[index].dataset_id = parseInt(e.target.value);
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

              {/* Caption Types - TODO: Implement caption type selector */}
              <div>
                <label className="block text-xs text-gray-400 mb-1">Caption Types (leave empty for all)</label>
                <input
                  type="text"
                  placeholder="e.g., caption, instruction"
                  value={config.caption_types.join(", ")}
                  onChange={(e) => {
                    const updated = [...datasetConfigs];
                    updated[index].caption_types = e.target.value.split(",").map(t => t.trim()).filter(t => t !== "");
                    setDatasetConfigs(updated);
                  }}
                  className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs focus:outline-none focus:border-blue-500"
                />
              </div>
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
                type="number"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                step="any"
                min="0.000001"
                max="0.01"
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
                <option value="euler">Euler</option>
                <option value="euler_a">Euler A</option>
                <option value="dpmpp_2m">DPM++ 2M</option>
                <option value="dpmpp_2m_sde">DPM++ 2M SDE</option>
                <option value="dpmpp_3m_sde">DPM++ 3M SDE</option>
                <option value="heun">Heun</option>
                <option value="lms">LMS</option>
                <option value="ddim">DDIM</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Schedule Type</label>
              <select
                value={sampleScheduleType}
                onChange={(e) => setSampleScheduleType(e.target.value)}
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="sgm_uniform">SGM Uniform</option>
                <option value="karras">Karras</option>
                <option value="exponential">Exponential</option>
                <option value="simple">Simple</option>
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
