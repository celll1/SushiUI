"use client";

import { useState, useEffect, useCallback } from "react";
import {
  createTaggerTrainingRun,
  getTaggerVocabularyPreview,
  listDatasets,
  Dataset,
  TaggerTrainingRun,
  TaggerTrainingRunCreateRequest,
} from "@/utils/api";

interface TaggerTrainingConfigProps {
  onClose: () => void;
  onRunCreated: (run: TaggerTrainingRun) => void;
}

const DEFAULT_CONFIG: TaggerTrainingRunCreateRequest = {
  run_name: "",
  training_method: "lora",
  vision_encoder_path: "",
  dataset_configs: [],
  lora_rank: 32,
  lora_alpha: 16,
  learning_rate: 3e-4,
  head_lr_multiplier: 10.0,
  optimizer: "adamw8bit",
  warmup_steps: 100,
  epochs: 10,
  batch_size: 32,
  mixed_precision: "bf16",
  gradient_checkpointing: true,
  loss_gamma_neg: 4,
  loss_gamma_pos: 1,
  validate_every: 1,
  save_best_only: true,
};

export default function TaggerTrainingConfig({
  onClose,
  onRunCreated,
}: TaggerTrainingConfigProps) {
  const [config, setConfig] = useState<TaggerTrainingRunCreateRequest>(DEFAULT_CONFIG);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [vocabPreview, setVocabPreview] = useState<{ total_tags: number; sample_tags: string[] } | null>(null);
  const [vocabLoading, setVocabLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Only show datasets that have tags-format captions
  const tagDatasets = datasets.filter((d) => d.has_tags_captions);

  useEffect(() => {
    listDatasets()
      .then((res) => setDatasets(res.datasets || []))
      .catch(console.error);
  }, []);

  const loadVocabPreview = useCallback(async () => {
    if (config.dataset_configs.length === 0) {
      setVocabPreview(null);
      return;
    }
    setVocabLoading(true);
    try {
      const preview = await getTaggerVocabularyPreview(config.dataset_configs);
      setVocabPreview(preview);
    } catch (err) {
      console.error("[TaggerTrainingConfig] Vocab preview error:", err);
      setVocabPreview(null);
    } finally {
      setVocabLoading(false);
    }
  }, [config.dataset_configs]);

  useEffect(() => {
    loadVocabPreview();
  }, [loadVocabPreview]);

  const handleDatasetToggle = (datasetId: string) => {
    const current = config.dataset_configs;
    const updated = current.includes(datasetId)
      ? current.filter((id) => id !== datasetId)
      : [...current, datasetId];
    setConfig({ ...config, dataset_configs: updated });
  };

  const handleCreate = async () => {
    if (!config.run_name.trim()) {
      setError("Run name is required.");
      return;
    }
    if (config.dataset_configs.length === 0) {
      setError("At least one dataset must be selected.");
      return;
    }
    setCreating(true);
    setError(null);
    try {
      const run = await createTaggerTrainingRun(config);
      onRunCreated(run);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setCreating(false);
    }
  };

  const setField = <K extends keyof TaggerTrainingRunCreateRequest>(
    key: K,
    value: TaggerTrainingRunCreateRequest[K]
  ) => setConfig((prev) => ({ ...prev, [key]: value }));

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700 flex-shrink-0">
        <h2 className="text-lg font-semibold">New Tagger Training Run</h2>
        <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">

        {/* Run name */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-1">Run Name</label>
          <input
            type="text"
            value={config.run_name}
            onChange={(e) => setField("run_name", e.target.value)}
            placeholder="e.g. siglip2_tagger_v1_20260409"
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
        </section>

        {/* Training method */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-2">Training Method</label>
          <div className="flex gap-3">
            {(["lora", "full"] as const).map((method) => (
              <button
                key={method}
                onClick={() => setField("training_method", method)}
                className={`px-4 py-2 rounded text-sm border transition-colors ${
                  config.training_method === method
                    ? "border-blue-500 bg-blue-600 text-white"
                    : "border-gray-600 bg-gray-800 text-gray-300 hover:bg-gray-700"
                }`}
              >
                {method === "lora" ? "LoRA" : "Full Parameter"}
              </button>
            ))}
          </div>
        </section>

        {/* Vision encoder path */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Vision Encoder Path
          </label>
          <input
            type="text"
            value={config.vision_encoder_path}
            onChange={(e) => setField("vision_encoder_path", e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Path to siglip2_so400m_vision_encoder.safetensors (relative to project root or absolute)
          </p>
        </section>

        {/* Dataset selection */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Datasets <span className="text-gray-500">(tags format only)</span>
          </label>
          {datasets.length === 0 ? (
            <p className="text-sm text-gray-500">Loading datasets...</p>
          ) : tagDatasets.length === 0 ? (
            <p className="text-sm text-gray-500">
              No datasets with tags-format captions found. Scan a dataset that has tag captions first.
            </p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {tagDatasets.map((dataset) => {
                const id = dataset.unique_id || String(dataset.id);
                const selected = config.dataset_configs.includes(id);
                return (
                  <label
                    key={dataset.id}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded border cursor-pointer transition-colors text-sm ${
                      selected
                        ? "border-blue-500 bg-blue-900/30 text-white"
                        : "border-gray-600 bg-gray-800 text-gray-300 hover:bg-gray-700"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => handleDatasetToggle(id)}
                      className="accent-blue-500"
                    />
                    <span className="font-medium">{dataset.name}</span>
                    <span className="text-xs text-gray-400">
                      {dataset.total_items.toLocaleString()} imgs · {dataset.total_tags.toLocaleString()} tagged
                    </span>
                  </label>
                );
              })}
            </div>
          )}

          {/* Vocab preview */}
          {config.dataset_configs.length > 0 && (
            <div className="mt-3 p-3 bg-gray-800 rounded border border-gray-700 text-xs">
              {vocabLoading ? (
                <span className="text-gray-400">Building vocabulary preview...</span>
              ) : vocabPreview ? (
                <>
                  <div className="text-green-400 font-medium mb-1">
                    Vocabulary: {vocabPreview.total_tags.toLocaleString()} tags
                  </div>
                  {vocabPreview.sample_tags.length > 0 && (
                    <div className="text-gray-400">
                      Sample: {vocabPreview.sample_tags.slice(0, 10).join(", ")}
                      {vocabPreview.sample_tags.length > 10 && "..."}
                    </div>
                  )}
                </>
              ) : (
                <span className="text-gray-500">Vocabulary preview unavailable</span>
              )}
            </div>
          )}
        </section>

        {/* LoRA parameters */}
        {config.training_method === "lora" && (
          <section>
            <label className="block text-sm font-medium text-gray-300 mb-3">LoRA Parameters</label>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Rank</label>
                <input
                  type="number"
                  min={1}
                  max={256}
                  value={config.lora_rank}
                  onChange={(e) => setField("lora_rank", parseInt(e.target.value) || 32)}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Alpha</label>
                <input
                  type="number"
                  min={1}
                  max={256}
                  value={config.lora_alpha}
                  onChange={(e) => setField("lora_alpha", parseInt(e.target.value) || 16)}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
          </section>
        )}

        {/* Training hyperparameters */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-3">Training Parameters</label>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Learning Rate</label>
              <input
                type="number"
                step="1e-5"
                min="1e-6"
                max="1"
                value={config.learning_rate}
                onChange={(e) => setField("learning_rate", parseFloat(e.target.value) || 3e-4)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Head LR Multiplier</label>
              <input
                type="number"
                step="0.5"
                min="0.1"
                max="100"
                value={config.head_lr_multiplier}
                onChange={(e) => setField("head_lr_multiplier", parseFloat(e.target.value) || 10)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Epochs</label>
              <input
                type="number"
                min={1}
                max={1000}
                value={config.epochs}
                onChange={(e) => setField("epochs", parseInt(e.target.value) || 10)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Batch Size</label>
              <input
                type="number"
                min={1}
                max={512}
                value={config.batch_size}
                onChange={(e) => setField("batch_size", parseInt(e.target.value) || 32)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Warmup Steps</label>
              <input
                type="number"
                min={0}
                max={10000}
                value={config.warmup_steps}
                onChange={(e) => setField("warmup_steps", parseInt(e.target.value) || 0)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Validate Every (epochs)</label>
              <input
                type="number"
                min={1}
                max={100}
                value={config.validate_every}
                onChange={(e) => setField("validate_every", parseInt(e.target.value) || 1)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>
        </section>

        {/* Optimizer */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-2">Optimizer</label>
          <select
            value={config.optimizer}
            onChange={(e) => setField("optimizer", e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
          >
            <option value="adamw">AdamW</option>
            <option value="adamw8bit">AdamW 8-bit</option>
            <option value="lion8bit">Lion 8-bit</option>
          </select>
        </section>

        {/* Mixed precision */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-2">Mixed Precision</label>
          <select
            value={config.mixed_precision}
            onChange={(e) => setField("mixed_precision", e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
          >
            <option value="bf16">bf16</option>
            <option value="fp16">fp16</option>
            <option value="fp32">fp32 (no AMP)</option>
          </select>
        </section>

        {/* Loss parameters */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-3">
            Asymmetric Loss Parameters
          </label>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">gamma_neg</label>
              <input
                type="number"
                step="0.5"
                min="0"
                max="10"
                value={config.loss_gamma_neg}
                onChange={(e) => setField("loss_gamma_neg", parseFloat(e.target.value) || 4)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">gamma_pos</label>
              <input
                type="number"
                step="0.5"
                min="0"
                max="10"
                value={config.loss_gamma_pos}
                onChange={(e) => setField("loss_gamma_pos", parseFloat(e.target.value) || 1)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>
        </section>

        {/* Boolean options */}
        <section className="space-y-3">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={config.gradient_checkpointing}
              onChange={(e) => setField("gradient_checkpointing", e.target.checked)}
              className="accent-blue-500"
            />
            <span className="text-sm text-gray-300">Gradient Checkpointing</span>
          </label>
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={config.save_best_only}
              onChange={(e) => setField("save_best_only", e.target.checked)}
              className="accent-blue-500"
            />
            <span className="text-sm text-gray-300">Save Best Only (by F1)</span>
          </label>
        </section>

        {/* Error */}
        {error && (
          <div className="p-3 bg-red-900/30 border border-red-700 rounded text-sm text-red-400">
            {error}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 p-4 border-t border-gray-700 flex justify-end gap-3">
        <button
          onClick={onClose}
          className="px-4 py-2 text-sm text-gray-300 hover:text-white transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleCreate}
          disabled={creating}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:text-gray-400 rounded text-sm transition-colors"
        >
          {creating ? "Creating..." : "Create Run"}
        </button>
      </div>
    </div>
  );
}
