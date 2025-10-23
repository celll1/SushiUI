"use client";

import { useState, useEffect } from "react";
import { getTIPOStatus, loadTIPOModel, unloadTIPOModel } from "@/utils/api";

interface TIPODialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (settings: TIPOSettings) => void;
  currentSettings: TIPOSettings;
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

export default function TIPODialog({ isOpen, onClose, onSave, currentSettings }: TIPODialogProps) {
  const [settings, setSettings] = useState<TIPOSettings>(currentSettings);
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean; model_name: string | null }>({
    loaded: false,
    model_name: null
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadStatus();
    }
  }, [isOpen]);

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
      await loadTIPOModel(settings.model_name);
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
    onSave(settings);
    onClose();
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
            value={settings.model_name}
            onChange={(e) => setSettings({ ...settings, model_name: e.target.value })}
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
            value={settings.tag_length}
            onChange={(e) => setSettings({ ...settings, tag_length: e.target.value })}
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
            value={settings.nl_length}
            onChange={(e) => setSettings({ ...settings, nl_length: e.target.value })}
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
            {settings.categories.map((category, index) => (
              <div key={category.id} className="flex items-center gap-2 bg-gray-800 p-2 rounded">
                <input
                  type="checkbox"
                  checked={category.enabled}
                  onChange={() => {
                    const newCategories = [...settings.categories];
                    newCategories[index].enabled = !newCategories[index].enabled;
                    setSettings({ ...settings, categories: newCategories });
                  }}
                  className="w-4 h-4"
                />
                <span className="flex-1 text-sm text-gray-300">{category.label}</span>
                <div className="flex gap-1">
                  <button
                    onClick={() => {
                      if (index === 0) return;
                      const newCategories = [...settings.categories];
                      [newCategories[index - 1], newCategories[index]] = [newCategories[index], newCategories[index - 1]];
                      setSettings({ ...settings, categories: newCategories });
                    }}
                    disabled={index === 0}
                    className="px-2 py-1 bg-gray-600 text-white rounded text-xs hover:bg-gray-500 disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    ↑
                  </button>
                  <button
                    onClick={() => {
                      if (index === settings.categories.length - 1) return;
                      const newCategories = [...settings.categories];
                      [newCategories[index], newCategories[index + 1]] = [newCategories[index + 1], newCategories[index]];
                      setSettings({ ...settings, categories: newCategories });
                    }}
                    disabled={index === settings.categories.length - 1}
                    className="px-2 py-1 bg-gray-600 text-white rounded text-xs hover:bg-gray-500 disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    ↓
                  </button>
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Check to enable, use arrows to reorder. Tags will be output in this order.
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
                Temperature: {settings.temperature.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={settings.temperature}
                onChange={(e) => setSettings({ ...settings, temperature: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* Top P */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Top P: {settings.top_p.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={settings.top_p}
                onChange={(e) => setSettings({ ...settings, top_p: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* Top K */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Top K: {settings.top_k}
              </label>
              <input
                type="range"
                min="1"
                max="100"
                step="1"
                value={settings.top_k}
                onChange={(e) => setSettings({ ...settings, top_k: parseInt(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* Max New Tokens */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Max New Tokens: {settings.max_new_tokens}
              </label>
              <input
                type="range"
                min="50"
                max="512"
                step="10"
                value={settings.max_new_tokens}
                onChange={(e) => setSettings({ ...settings, max_new_tokens: parseInt(e.target.value) })}
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
