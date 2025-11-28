"use client";

import { useState, useEffect } from "react";
import { getTIPOStatus, loadTIPOModel, unloadTIPOModel } from "@/utils/api";
import { getCategoryOrder } from "./CategoryOrderPanel";

interface TIPODialogProps {
  isOpen: boolean;
  onClose: () => void;
  settings: TIPOSettings;
  onSettingsChange: (settings: TIPOSettings) => void;
}

export interface TIPOCategory {
  id: string;
  label: string;
  enabled: boolean;
}

export interface TIPOSettings {
  model_name: string;
  tag_length: string;
  nl_length: string;
  temperature: number;
  top_p: number;
  top_k: number;
  max_new_tokens: number;
  categories: TIPOCategory[];
}

export default function TIPODialog({ isOpen, onClose, settings, onSettingsChange }: TIPODialogProps) {
  const [localSettings, setLocalSettings] = useState<TIPOSettings>(settings);
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean; model_name: string | null }>({
    loaded: false,
    model_name: null
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      // Load category order from CategoryOrderPanel settings
      const categoryOrder = getCategoryOrder();
      setLocalSettings({
        ...settings,
        categories: categoryOrder.map(cat => ({
          id: cat.id,
          label: cat.label,
          enabled: cat.enabled,
        })),
      });
      loadStatus();
    }
  }, [isOpen, settings]);

  const loadStatus = async () => {
    try {
      const status = await getTIPOStatus();
      setModelStatus({ loaded: status.loaded, model_name: status.model_name });
    } catch (error) {
      console.error("[TIPO] Failed to load status:", error);
    }
  };

  const handleLoadModel = async () => {
    setLoading(true);
    try {
      await loadTIPOModel(localSettings.model_name);
      await loadStatus();
      alert("TIPO model loaded successfully!");
    } catch (error) {
      console.error("[TIPO] Failed to load model:", error);
      alert("Failed to load TIPO model. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  const handleUnloadModel = async () => {
    setLoading(true);
    try {
      await unloadTIPOModel();
      await loadStatus();
      alert("TIPO model unloaded!");
    } catch (error) {
      console.error("[TIPO] Failed to unload model:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = () => {
    // Save TIPO settings to localStorage
    onSettingsChange(localSettings);

    // Also save category order back to CategoryOrderPanel settings
    const categoryOrderKey = "category_order";
    const categoryOrderSettings = localSettings.categories.map(cat => ({
      id: cat.id,
      label: cat.label,
      enabled: cat.enabled,
    }));
    localStorage.setItem(categoryOrderKey, JSON.stringify(categoryOrderSettings));

    onClose();
  };

  const handleToggleCategory = (index: number) => {
    const newCategories = [...localSettings.categories];
    newCategories[index].enabled = !newCategories[index].enabled;
    setLocalSettings({ ...localSettings, categories: newCategories });
  };

  const handleMoveCategory = (index: number, direction: 'up' | 'down') => {
    const newCategories = [...localSettings.categories];
    const targetIndex = direction === 'up' ? index - 1 : index + 1;

    if (targetIndex < 0 || targetIndex >= newCategories.length) {
      return;
    }

    // Swap categories
    [newCategories[index], newCategories[targetIndex]] = [newCategories[targetIndex], newCategories[index]];
    setLocalSettings({ ...localSettings, categories: newCategories });
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <h2 className="text-xl font-bold text-white mb-4">TIPO Settings</h2>

        {/* Model Status */}
        <div className="mb-4 p-3 bg-gray-700 rounded">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-300">Model Status:</p>
              <p className="text-sm font-medium text-white">
                {modelStatus.loaded ? (
                  <span className="text-green-400">✓ Loaded: {modelStatus.model_name}</span>
                ) : (
                  <span className="text-yellow-400">⚠ Not Loaded</span>
                )}
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleLoadModel}
                disabled={loading || modelStatus.loaded}
                className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Loading..." : "Load Model"}
              </button>
              <button
                onClick={handleUnloadModel}
                disabled={loading || !modelStatus.loaded}
                className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Unload
              </button>
            </div>
          </div>
        </div>

        {/* Model Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Model
          </label>
          <select
            value={localSettings.model_name}
            onChange={(e) => setLocalSettings({ ...localSettings, model_name: e.target.value })}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
          >
            <option value="KBlueLeaf/TIPO-500M">TIPO-500M (Recommended)</option>
            <option value="KBlueLeaf/TIPO-500M-ft">TIPO-500M-ft</option>
            <option value="KBlueLeaf/TIPO-200M">TIPO-200M</option>
            <option value="KBlueLeaf/TIPO-200M-ft2">TIPO-200M-ft2</option>
            <option value="KBlueLeaf/TIPO-200M-ft">TIPO-200M-ft</option>
          </select>
        </div>

        {/* Tag Length */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Tag Length
          </label>
          <select
            value={localSettings.tag_length}
            onChange={(e) => setLocalSettings({ ...localSettings, tag_length: e.target.value })}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
          >
            <option value="very_short">Very Short (6-17 tags)</option>
            <option value="short">Short (18-35 tags) - Recommended</option>
            <option value="long">Long (36-53 tags)</option>
            <option value="very_long">Very Long (54-72 tags)</option>
          </select>
        </div>

        {/* Natural Language Length */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Natural Language Length
          </label>
          <select
            value={localSettings.nl_length}
            onChange={(e) => setLocalSettings({ ...localSettings, nl_length: e.target.value })}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
          >
            <option value="very_short">Very Short (1-2 sentences)</option>
            <option value="short">Short (2-4 sentences) - Recommended</option>
            <option value="long">Long (4-6 sentences)</option>
            <option value="very_long">Very Long (6-8 sentences)</option>
          </select>
        </div>

        {/* Category Order */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Output Category Order
          </label>
          <div className="bg-gray-700 rounded p-3 space-y-2">
            {localSettings.categories.map((category, index) => (
              <div key={category.id} className="flex items-center gap-2 bg-gray-800 p-2 rounded">
                <input
                  type="checkbox"
                  checked={category.enabled}
                  onChange={() => handleToggleCategory(index)}
                  className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-2 focus:ring-blue-500 cursor-pointer"
                />
                <span className="flex-1 text-sm text-gray-300">{category.label}</span>
                <div className="flex gap-1">
                  <button
                    onClick={() => handleMoveCategory(index, 'up')}
                    disabled={index === 0}
                    className="px-2 py-1 bg-gray-600 text-white rounded text-xs hover:bg-gray-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    ↑
                  </button>
                  <button
                    onClick={() => handleMoveCategory(index, 'down')}
                    disabled={index === localSettings.categories.length - 1}
                    className="px-2 py-1 bg-gray-600 text-white rounded text-xs hover:bg-gray-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    ↓
                  </button>
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Drag categories using ↑/↓ buttons to reorder. Toggle checkboxes to enable/disable categories.
            Settings are shared with Prompt Editor.
          </p>
        </div>

        {/* Advanced Settings */}
        <details className="mb-4">
          <summary className="cursor-pointer text-sm font-medium text-gray-300 mb-2">
            Advanced Settings
          </summary>
          <div className="mt-2 space-y-3">
            {/* Temperature */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Temperature: {localSettings.temperature.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={localSettings.temperature}
                onChange={(e) => setLocalSettings({ ...localSettings, temperature: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* Top P */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Top P: {localSettings.top_p.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={localSettings.top_p}
                onChange={(e) => setLocalSettings({ ...localSettings, top_p: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* Top K */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Top K: {localSettings.top_k}
              </label>
              <input
                type="range"
                min="1"
                max="100"
                step="1"
                value={localSettings.top_k}
                onChange={(e) => setLocalSettings({ ...localSettings, top_k: parseInt(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* Max New Tokens */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Max New Tokens: {localSettings.max_new_tokens}
              </label>
              <input
                type="range"
                min="50"
                max="512"
                step="10"
                value={localSettings.max_new_tokens}
                onChange={(e) => setLocalSettings({ ...localSettings, max_new_tokens: parseInt(e.target.value) })}
                className="w-full"
              />
            </div>
          </div>
        </details>

        {/* Buttons */}
        <div className="flex gap-2 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
}
