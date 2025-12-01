"use client";

import React, { useState, useEffect } from "react";
import {
  CaptionProcessingConfig,
  CaptionProcessingPreset,
  listCaptionProcessingPresets,
  createCaptionProcessingPreset,
  deleteCaptionProcessingPreset,
} from "@/utils/api";

interface CaptionProcessingSettingsProps {
  config: CaptionProcessingConfig;
  onChange: (config: CaptionProcessingConfig) => void;
  readOnly?: boolean;
}

export default function CaptionProcessingSettings({
  config,
  onChange,
  readOnly = false,
}: CaptionProcessingSettingsProps) {
  const [localConfig, setLocalConfig] = useState<CaptionProcessingConfig>(config);
  const [presets, setPresets] = useState<CaptionProcessingPreset[]>([]);
  const [showSavePresetDialog, setShowSavePresetDialog] = useState(false);
  const [newPresetName, setNewPresetName] = useState("");
  const [newPresetDescription, setNewPresetDescription] = useState("");

  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  useEffect(() => {
    loadPresets();
  }, []);

  const loadPresets = async () => {
    try {
      const loadedPresets = await listCaptionProcessingPresets();
      setPresets(loadedPresets);
    } catch (error) {
      console.error("Failed to load presets:", error);
    }
  };

  const handleChange = (key: keyof CaptionProcessingConfig, value: any) => {
    const newConfig = { ...localConfig, [key]: value };
    setLocalConfig(newConfig);
    onChange(newConfig);
  };

  const handleLoadPreset = (preset: CaptionProcessingPreset) => {
    setLocalConfig(preset.config);
    onChange(preset.config);
  };

  const handleSavePreset = async () => {
    if (!newPresetName.trim()) {
      alert("Please enter a preset name");
      return;
    }

    try {
      await createCaptionProcessingPreset(
        newPresetName.trim(),
        newPresetDescription.trim() || null,
        localConfig
      );
      await loadPresets();
      setShowSavePresetDialog(false);
      setNewPresetName("");
      setNewPresetDescription("");
      alert("Preset saved successfully!");
    } catch (error: any) {
      alert(`Failed to save preset: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleDeletePreset = async (presetId: number, presetName: string) => {
    if (!confirm(`Delete preset "${presetName}"?`)) {
      return;
    }

    try {
      await deleteCaptionProcessingPreset(presetId);
      await loadPresets();
    } catch (error: any) {
      alert(`Failed to delete preset: ${error.response?.data?.detail || error.message}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Preset Management */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-200 border-b border-gray-700 pb-2">
          Presets
        </h3>
        <div className="space-y-2">
          <div className="flex gap-2">
            <select
              className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm"
              onChange={(e) => {
                const presetId = parseInt(e.target.value);
                const preset = presets.find((p) => p.id === presetId);
                if (preset) {
                  handleLoadPreset(preset);
                }
              }}
              disabled={readOnly}
              defaultValue=""
            >
              <option value="" disabled>
                Load preset...
              </option>
              {presets.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => setShowSavePresetDialog(true)}
              disabled={readOnly}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm disabled:opacity-50"
            >
              Save as Preset
            </button>
          </div>

          {/* Preset List */}
          {presets.length > 0 && (
            <div className="mt-3 space-y-1">
              {presets.map((preset) => (
                <div
                  key={preset.id}
                  className="flex items-center justify-between p-2 bg-gray-800 rounded text-sm"
                >
                  <div className="flex-1">
                    <div className="font-medium text-gray-200">{preset.name}</div>
                    {preset.description && (
                      <div className="text-xs text-gray-400">{preset.description}</div>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => handleLoadPreset(preset)}
                      disabled={readOnly}
                      className="px-2 py-1 text-xs bg-green-600 hover:bg-green-700 rounded disabled:opacity-50"
                    >
                      Load
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDeletePreset(preset.id, preset.name)}
                      disabled={readOnly}
                      className="px-2 py-1 text-xs bg-red-600 hover:bg-red-700 rounded disabled:opacity-50"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Save Preset Dialog */}
      {showSavePresetDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl max-w-md w-full">
            <h3 className="text-lg font-semibold text-gray-200 mb-4">Save Preset</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-300 mb-1">Preset Name *</label>
                <input
                  type="text"
                  value={newPresetName}
                  onChange={(e) => setNewPresetName(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm"
                  placeholder="e.g., My Favorite Settings"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">Description (optional)</label>
                <textarea
                  value={newPresetDescription}
                  onChange={(e) => setNewPresetDescription(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm"
                  placeholder="e.g., Aggressive dropout for overfitting prevention"
                  rows={3}
                />
              </div>
              <div className="flex gap-2 justify-end">
                <button
                  type="button"
                  onClick={() => {
                    setShowSavePresetDialog(false);
                    setNewPresetName("");
                    setNewPresetDescription("");
                  }}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded text-sm"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSavePreset}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tag Normalization */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-200 border-b border-gray-700 pb-2">
          Tag Normalization
        </h3>
        <div className="space-y-2">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={localConfig.normalize_tags !== false}  // Default true
              onChange={(e) => handleChange("normalize_tags", e.target.checked)}
              disabled={readOnly}
              className="rounded"
            />
            <span className="text-sm text-gray-300">Normalize Tags to Standard Format</span>
          </label>
          <p className="text-xs text-gray-500">
            Standardize tag format to <code className="text-gray-400 bg-gray-800 px-1 rounded">tag_name \(qualifier\)</code> for consistency.
            Handles various patterns: underscore format (<code className="text-gray-400 bg-gray-800 px-1 rounded">tag_(qualifier)</code>),
            space format (<code className="text-gray-400 bg-gray-800 px-1 rounded">tag (qualifier)</code>),
            escaped/over-escaped parentheses.
          </p>
        </div>
      </div>

      {/* Category Order */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-200 border-b border-gray-700 pb-2">
          Category Order
        </h3>
        <div className="space-y-2">
          <p className="text-xs text-gray-500">
            Reorder tags by category (processed FIRST, before dropout/shuffle)
          </p>

          {/* Enable Category Ordering */}
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={(localConfig.category_order || []).length > 0}
              onChange={(e) => {
                if (e.target.checked) {
                  // Enable with default order
                  handleChange("category_order", ["Rating", "Quality", "Character", "General", "Copyright", "Artist", "Meta", "Model"]);
                } else {
                  // Disable
                  handleChange("category_order", []);
                }
              }}
              disabled={readOnly}
              className="rounded"
            />
            <span className="text-sm text-gray-300">Enable Category Ordering</span>
          </label>

          {/* Category Order List (drag and drop) */}
          {(localConfig.category_order || []).length > 0 && (
            <div className="pl-6 space-y-1 mt-2">
              {(localConfig.category_order || []).map((category, index) => (
                <div
                  key={category}
                  className="flex items-center gap-2 px-3 py-2 bg-gray-700 rounded hover:bg-gray-650"
                >
                  <span className="text-gray-400 text-sm cursor-move">⋮⋮</span>
                  <span className="text-sm text-gray-200 flex-1">{category}</span>
                  <button
                    type="button"
                    onClick={() => {
                      const current = localConfig.category_order || [];
                      if (index > 0) {
                        const updated = [...current];
                        [updated[index - 1], updated[index]] = [updated[index], updated[index - 1]];
                        handleChange("category_order", updated);
                      }
                    }}
                    disabled={readOnly || index === 0}
                    className="text-xs px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded disabled:opacity-30"
                  >
                    ↑
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const current = localConfig.category_order || [];
                      if (index < current.length - 1) {
                        const updated = [...current];
                        [updated[index], updated[index + 1]] = [updated[index + 1], updated[index]];
                        handleChange("category_order", updated);
                      }
                    }}
                    disabled={readOnly || index === (localConfig.category_order || []).length - 1}
                    className="text-xs px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded disabled:opacity-30"
                  >
                    ↓
                  </button>
                </div>
              ))}
              <p className="text-xs text-gray-500 mt-2">
                Use ↑/↓ buttons to reorder categories. Shuffle/dropout's "keep first N tokens" and "within-category shuffle" will be based on this order.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Caption Dropout */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-200 border-b border-gray-700 pb-2">
          Caption Dropout
        </h3>
        <div className="space-y-2">
          <label className="flex items-center justify-between">
            <span className="text-sm text-gray-300">Caption Dropout Rate</span>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={localConfig.caption_dropout_rate || 0}
                onChange={(e) => handleChange("caption_dropout_rate", parseFloat(e.target.value))}
                disabled={readOnly}
                className="w-32"
              />
              <span className="text-sm text-gray-400 w-12 text-right">
                {((localConfig.caption_dropout_rate || 0) * 100).toFixed(0)}%
              </span>
            </div>
          </label>
          <p className="text-xs text-gray-500">
            Probability of dropping entire caption (0.0 = never, 1.0 = always)
          </p>
        </div>
      </div>

      {/* Token Dropout */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-200 border-b border-gray-700 pb-2">
          Token Dropout
        </h3>
        <div className="space-y-2">
          <label className="flex items-center justify-between">
            <span className="text-sm text-gray-300">Token Dropout Rate</span>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={localConfig.token_dropout_rate || 0}
                onChange={(e) => handleChange("token_dropout_rate", parseFloat(e.target.value))}
                disabled={readOnly}
                className="w-32"
              />
              <span className="text-sm text-gray-400 w-12 text-right">
                {((localConfig.token_dropout_rate || 0) * 100).toFixed(0)}%
              </span>
            </div>
          </label>
          <p className="text-xs text-gray-500">
            Probability of dropping each token individually
          </p>

          <label className="flex items-center justify-between">
            <span className="text-sm text-gray-300">Keep First N Tokens</span>
            <input
              type="number"
              min="0"
              max="20"
              value={localConfig.keep_tokens || 0}
              onChange={(e) => handleChange("keep_tokens", parseInt(e.target.value))}
              disabled={readOnly}
              className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm"
            />
          </label>
          <p className="text-xs text-gray-500">
            Number of first tokens to always keep (immune to token dropout)
          </p>
        </div>
      </div>

      {/* Token Shuffle */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-200 border-b border-gray-700 pb-2">
          Token Shuffle
        </h3>
        <div className="space-y-2">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={localConfig.shuffle_tokens || false}
              onChange={(e) => handleChange("shuffle_tokens", e.target.checked)}
              disabled={readOnly}
              className="rounded"
            />
            <span className="text-sm text-gray-300">Enable Token Shuffle</span>
          </label>

          {localConfig.shuffle_tokens && (
            <>
              <label className="flex items-center gap-2 pl-6">
                <input
                  type="checkbox"
                  checked={localConfig.shuffle_per_epoch || false}
                  onChange={(e) => handleChange("shuffle_per_epoch", e.target.checked)}
                  disabled={readOnly}
                  className="rounded"
                />
                <span className="text-sm text-gray-300">Shuffle Per Epoch (Reproducible)</span>
              </label>
              <p className="text-xs text-gray-500 pl-6">
                Different shuffle for each epoch, but consistent across runs
              </p>

              <label className="flex items-center justify-between pl-6">
                <span className="text-sm text-gray-300">Keep First N Tokens Unshuffled</span>
                <input
                  type="number"
                  min="0"
                  max="20"
                  value={localConfig.shuffle_keep_first_n || 0}
                  onChange={(e) => handleChange("shuffle_keep_first_n", parseInt(e.target.value))}
                  disabled={readOnly}
                  className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm"
                />
              </label>
            </>
          )}
        </div>
      </div>

      {/* Tag Group Shuffle */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-200 border-b border-gray-700 pb-2">
          Tag Group Shuffle
        </h3>
        <div className="space-y-2">
          <p className="text-xs text-gray-500">
            Shuffle tags by category (requires tag group JSON files in backend/taggroup/)
          </p>

          <div className="space-y-1">
            <label className="text-sm text-gray-300">Select Tag Groups to Shuffle</label>
            <div className="grid grid-cols-2 gap-2">
              {["Character", "General", "Copyright", "Artist", "Meta", "Model", "Rating", "Quality"].map((group) => (
                <label key={group} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={(localConfig.shuffle_tag_groups || []).includes(group)}
                    onChange={(e) => {
                      const current = localConfig.shuffle_tag_groups || [];
                      const updated = e.target.checked
                        ? [...current, group]
                        : current.filter((g) => g !== group);
                      handleChange("shuffle_tag_groups", updated);
                    }}
                    disabled={readOnly}
                    className="rounded"
                  />
                  <span className="text-sm text-gray-300">{group}</span>
                </label>
              ))}
            </div>
          </div>

          {(localConfig.shuffle_tag_groups || []).length > 0 && (
            <>
              <label className="flex items-center gap-2 pl-6">
                <input
                  type="checkbox"
                  checked={localConfig.shuffle_groups_together || false}
                  onChange={(e) => handleChange("shuffle_groups_together", e.target.checked)}
                  disabled={readOnly}
                  className="rounded"
                />
                <span className="text-sm text-gray-300">Shuffle All Groups Together</span>
              </label>
              <p className="text-xs text-gray-500 pl-6">
                If checked: all selected groups shuffle together. If unchecked: each group shuffles independently.
              </p>

              <label className="flex items-center gap-2 pl-6">
                <input
                  type="checkbox"
                  checked={localConfig.exclude_person_count_from_shuffle || false}
                  onChange={(e) => handleChange("exclude_person_count_from_shuffle", e.target.checked)}
                  disabled={readOnly}
                  className="rounded"
                />
                <span className="text-sm text-gray-300">Exclude Person Count Tags (1girl, 2boys, solo, etc.)</span>
              </label>
              <p className="text-xs text-gray-500 pl-6">
                Prevents person count tags from being shuffled (keeps them in original position)
              </p>
            </>
          )}
        </div>
      </div>

      {/* Tag Dropout */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-gray-200 border-b border-gray-700 pb-2">
          Tag-Level Dropout
        </h3>
        <div className="space-y-2">
          <label className="flex items-center justify-between">
            <span className="text-sm text-gray-300">Tag Dropout Rate (Base)</span>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={localConfig.tag_dropout_rate || 0}
                onChange={(e) => handleChange("tag_dropout_rate", parseFloat(e.target.value))}
                disabled={readOnly}
                className="w-32"
              />
              <span className="text-sm text-gray-400 w-12 text-right">
                {((localConfig.tag_dropout_rate || 0) * 100).toFixed(0)}%
              </span>
            </div>
          </label>
          <p className="text-xs text-gray-500">
            Base probability of dropping each tag (can be overridden by category-specific rates)
          </p>

          {(localConfig.tag_dropout_rate || 0) > 0 && (
            <>
              <label className="flex items-center gap-2 pl-6">
                <input
                  type="checkbox"
                  checked={localConfig.tag_dropout_per_epoch || false}
                  onChange={(e) => handleChange("tag_dropout_per_epoch", e.target.checked)}
                  disabled={readOnly}
                  className="rounded"
                />
                <span className="text-sm text-gray-300">Dropout Per Epoch (Reproducible)</span>
              </label>

              <label className="flex items-center justify-between pl-6">
                <span className="text-sm text-gray-300">Keep First N Tags</span>
                <input
                  type="number"
                  min="0"
                  max="20"
                  value={localConfig.tag_dropout_keep_first_n || 0}
                  onChange={(e) => handleChange("tag_dropout_keep_first_n", parseInt(e.target.value))}
                  disabled={readOnly}
                  className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm"
                />
              </label>

              <label className="flex items-center gap-2 pl-6">
                <input
                  type="checkbox"
                  checked={localConfig.tag_dropout_exclude_person_count || false}
                  onChange={(e) => handleChange("tag_dropout_exclude_person_count", e.target.checked)}
                  disabled={readOnly}
                  className="rounded"
                />
                <span className="text-sm text-gray-300">Exclude Person Count Tags (1girl, 2boys, etc.)</span>
              </label>
            </>
          )}

          {/* Category-Specific Dropout Rates */}
          {(localConfig.tag_dropout_rate || 0) > 0 && (
            <div className="pl-6 space-y-2 mt-3 pt-3 border-t border-gray-700">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-300 font-medium">Category-Specific Dropout Rates</span>
                <button
                  type="button"
                  onClick={() => {
                    const current = localConfig.tag_dropout_category_rates || {};
                    if (Object.keys(current).length === 0) {
                      // Initialize with default categories
                      handleChange("tag_dropout_category_rates", {
                        Character: 0.1,
                        General: 0.3,
                      });
                    } else {
                      // Clear all
                      handleChange("tag_dropout_category_rates", {});
                    }
                  }}
                  disabled={readOnly}
                  className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded disabled:opacity-50"
                >
                  {Object.keys(localConfig.tag_dropout_category_rates || {}).length === 0 ? "Add Categories" : "Clear All"}
                </button>
              </div>
              <p className="text-xs text-gray-500">
                Override base dropout rate for specific tag categories
              </p>

              {Object.keys(localConfig.tag_dropout_category_rates || {}).length > 0 && (
                <div className="space-y-2">
                  {Object.entries(localConfig.tag_dropout_category_rates || {}).map(([category, rate]) => (
                    <div key={category} className="flex items-center gap-2">
                      <select
                        value={category}
                        onChange={(e) => {
                          const current = localConfig.tag_dropout_category_rates || {};
                          const updated = { ...current };
                          delete updated[category];
                          updated[e.target.value] = rate;
                          handleChange("tag_dropout_category_rates", updated);
                        }}
                        disabled={readOnly}
                        className="flex-1 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm"
                      >
                        {["Character", "General", "Copyright", "Artist", "Meta", "Model", "Rating", "Quality"].map((g) => (
                          <option key={g} value={g}>{g}</option>
                        ))}
                      </select>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={rate}
                        onChange={(e) => {
                          const current = localConfig.tag_dropout_category_rates || {};
                          handleChange("tag_dropout_category_rates", {
                            ...current,
                            [category]: parseFloat(e.target.value),
                          });
                        }}
                        disabled={readOnly}
                        className="w-24"
                      />
                      <span className="text-sm text-gray-400 w-12 text-right">
                        {(rate * 100).toFixed(0)}%
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          const current = localConfig.tag_dropout_category_rates || {};
                          const updated = { ...current };
                          delete updated[category];
                          handleChange("tag_dropout_category_rates", updated);
                        }}
                        disabled={readOnly}
                        className="text-red-400 hover:text-red-300 px-2 py-1 text-xs"
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={() => {
                      const current = localConfig.tag_dropout_category_rates || {};
                      const usedCategories = Object.keys(current);
                      const availableCategories = ["Character", "General", "Copyright", "Artist", "Meta", "Model", "Rating", "Quality"]
                        .filter(g => !usedCategories.includes(g));
                      if (availableCategories.length > 0) {
                        handleChange("tag_dropout_category_rates", {
                          ...current,
                          [availableCategories[0]]: 0.1,
                        });
                      }
                    }}
                    disabled={readOnly}
                    className="text-xs px-2 py-1 bg-green-600 hover:bg-green-700 rounded disabled:opacity-50"
                  >
                    Add Another Category
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Info */}
      <div className="mt-4 p-3 bg-blue-900/20 border border-blue-800/50 rounded text-xs text-gray-400">
        <p className="font-semibold text-blue-400 mb-1">ℹ️ Caption Processing</p>
        <p>
          These settings control how captions are processed during training. Dropout and shuffle
          help prevent overfitting and improve generalization.
        </p>
      </div>
    </div>
  );
}
