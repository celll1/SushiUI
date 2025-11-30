"use client";

import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { createTrainingRun, listDatasets, Dataset, TrainingRun, getModels } from "@/utils/api";

interface TrainingConfigProps {
  onClose: () => void;
  onRunCreated: (run: TrainingRun) => void;
}

export default function TrainingConfig({ onClose, onRunCreated }: TrainingConfigProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [runName, setRunName] = useState("");
  const [datasetId, setDatasetId] = useState<number | null>(null);
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
        setDatasetId(response.datasets[0].id);
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    console.log("[TrainingConfig] Form submitted");
    console.log("[TrainingConfig] Run name:", runName);
    console.log("[TrainingConfig] Dataset ID:", datasetId);
    console.log("[TrainingConfig] Base model path:", baseModelPath);

    if (!datasetId) {
      setError("Please select a dataset");
      return;
    }

    if (!baseModelPath.trim()) {
      setError("Base model path is required");
      return;
    }

    setLoading(true);
    setError(null);

    const requestData = {
      dataset_id: datasetId,
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
      sample_prompts: [],
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

        {/* Dataset */}
        <div>
          <label className="block text-sm font-medium mb-2">
            Dataset <span className="text-red-400">*</span>
          </label>
          <select
            value={datasetId || ""}
            onChange={(e) => setDatasetId(parseInt(e.target.value))}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
            required
          >
            <option value="">Select dataset...</option>
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>
                {ds.name} ({ds.total_items} items)
              </option>
            ))}
          </select>
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
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm text-gray-400 mb-1.5">Save Checkpoint Every</label>
              <input
                type="number"
                min="1"
                value={saveEvery}
                onChange={(e) => setSaveEvery(parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1.5">Unit</label>
              <select
                value={saveEveryUnit}
                onChange={(e) => setSaveEveryUnit(e.target.value as "steps" | "epochs")}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="steps">Steps</option>
                <option value="epochs">Epochs</option>
              </select>
            </div>
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
