"use client";

import { useState, useEffect, useRef } from "react";
import { X } from "lucide-react";

export interface TaggerSettings {
  categoryThresholds: CategoryThreshold[];
  modelVersion: string;
}

export interface CategoryThreshold {
  id: string;
  label: string;
  removeThreshold: number; // Below this: remove existing tags
  addThreshold: number;    // Above this: add predicted tags
  enabled: boolean;
}

const DEFAULT_THRESHOLDS: CategoryThreshold[] = [
  { id: "rating", label: "Rating", removeThreshold: 0.0, addThreshold: 0.45, enabled: true },
  { id: "quality", label: "Quality", removeThreshold: 0.0, addThreshold: 0.45, enabled: true },
  { id: "character", label: "Character", removeThreshold: 0.0, addThreshold: 0.45, enabled: true },
  { id: "copyright", label: "Copyright", removeThreshold: 0.0, addThreshold: 0.45, enabled: true },
  { id: "artist", label: "Artist", removeThreshold: 0.0, addThreshold: 0.45, enabled: true },
  { id: "general", label: "General", removeThreshold: 0.0, addThreshold: 0.45, enabled: true },
  { id: "meta", label: "Meta", removeThreshold: 0.0, addThreshold: 0.45, enabled: true },
  { id: "model", label: "Model", removeThreshold: 0.0, addThreshold: 0.45, enabled: true },
];

const MODEL_VERSIONS = [
  { value: "cl_tagger_1_00", label: "v1.00" },
  { value: "cl_tagger_1_01", label: "v1.01" },
  { value: "cl_tagger_1_02", label: "v1.02 (Latest)" },
];

const STORAGE_KEY = "dataset_tagger_settings";

interface TaggerSettingsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (settings: TaggerSettings) => void;
}

export default function TaggerSettingsDialog({ isOpen, onClose, onSave }: TaggerSettingsDialogProps) {
  const [settings, setSettings] = useState<TaggerSettings>({
    categoryThresholds: DEFAULT_THRESHOLDS,
    modelVersion: "cl_tagger_1_02",
  });

  // Load settings from localStorage
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        setSettings(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to load tagger settings:", e);
      }
    }
  }, []);

  const handleSave = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    onSave(settings);
    onClose();
  };

  const updateCategoryThresholds = (index: number, removeThreshold: number, addThreshold: number) => {
    const newThresholds = [...settings.categoryThresholds];
    newThresholds[index].removeThreshold = removeThreshold;
    newThresholds[index].addThreshold = addThreshold;
    setSettings({ ...settings, categoryThresholds: newThresholds });
  };

  const toggleCategory = (index: number) => {
    const newThresholds = [...settings.categoryThresholds];
    newThresholds[index].enabled = !newThresholds[index].enabled;
    setSettings({ ...settings, categoryThresholds: newThresholds });
  };

  const resetToDefaults = () => {
    if (confirm("Reset all settings to defaults?")) {
      setSettings({
        categoryThresholds: DEFAULT_THRESHOLDS,
        modelVersion: "cl_tagger_1_02",
      });
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg border border-gray-700 w-full max-w-md max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b border-gray-700">
          <h2 className="text-sm font-semibold">Tagger Settings</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-3 space-y-3">
          {/* Model Version */}
          <div>
            <label className="block text-xs font-medium text-gray-300 mb-1">
              Model Version
            </label>
            <select
              value={settings.modelVersion}
              onChange={(e) => setSettings({ ...settings, modelVersion: e.target.value })}
              className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs"
            >
              {MODEL_VERSIONS.map(v => (
                <option key={v.value} value={v.value}>{v.label}</option>
              ))}
            </select>
          </div>

          {/* Category Thresholds */}
          <div>
            <label className="block text-xs font-medium text-gray-300 mb-1">
              Category Threshold Ranges
            </label>
            <div className="bg-gray-900 rounded p-2 space-y-1 max-h-56 overflow-y-auto">
              {settings.categoryThresholds.map((cat, index) => (
                <DualThresholdSlider
                  key={cat.id}
                  label={cat.label}
                  enabled={cat.enabled}
                  removeThreshold={cat.removeThreshold}
                  addThreshold={cat.addThreshold}
                  onToggle={() => toggleCategory(index)}
                  onChange={(removeThreshold, addThreshold) =>
                    updateCategoryThresholds(index, removeThreshold, addThreshold)
                  }
                />
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-3 border-t border-gray-700">
          <button
            onClick={resetToDefaults}
            className="px-2 py-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
          >
            Reset to Defaults
          </button>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-xs transition-colors"
            >
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Dual-threshold slider component (2 handles on single track)
function DualThresholdSlider({
  label,
  enabled,
  removeThreshold,
  addThreshold,
  onToggle,
  onChange,
}: {
  label: string;
  enabled: boolean;
  removeThreshold: number;
  addThreshold: number;
  onToggle: () => void;
  onChange: (removeThreshold: number, addThreshold: number) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [isDraggingRemove, setIsDraggingRemove] = useState(false);
  const [isDraggingAdd, setIsDraggingAdd] = useState(false);

  const handleMouseDown = (type: 'remove' | 'add') => (e: React.MouseEvent) => {
    e.preventDefault();
    if (type === 'remove') {
      setIsDraggingRemove(true);
    } else {
      setIsDraggingAdd(true);
    }
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!trackRef.current || (!isDraggingRemove && !isDraggingAdd)) return;

      const rect = trackRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      const percent = x / rect.width;
      const value = Math.round(percent * 20) / 20; // Snap to 0.05 increments

      if (isDraggingRemove) {
        // Ensure removeThreshold < addThreshold
        const newRemove = Math.min(value, addThreshold - 0.05);
        onChange(Math.max(0, newRemove), addThreshold);
      } else if (isDraggingAdd) {
        // Ensure addThreshold > removeThreshold
        const newAdd = Math.max(value, removeThreshold + 0.05);
        onChange(removeThreshold, Math.min(1, newAdd));
      }
    };

    const handleMouseUp = () => {
      setIsDraggingRemove(false);
      setIsDraggingAdd(false);
    };

    if (isDraggingRemove || isDraggingAdd) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDraggingRemove, isDraggingAdd, removeThreshold, addThreshold, onChange]);

  return (
    <div className="flex items-center gap-2 py-0.5">
      {/* Checkbox */}
      <input
        type="checkbox"
        checked={enabled}
        onChange={onToggle}
        className="cursor-pointer w-3 h-3 flex-shrink-0"
      />

      {/* Label */}
      <span className={`text-[10px] font-medium w-16 flex-shrink-0 ${enabled ? 'text-gray-200' : 'text-gray-500'}`}>
        {label}
      </span>

      {/* Dual-handle slider (compact) */}
      <div
        ref={trackRef}
        className={`relative h-4 flex-1 ${!enabled ? 'opacity-40 pointer-events-none' : ''}`}
      >
        {/* Track background with 3 zones */}
        <div className="absolute top-1.5 left-0 right-0 h-1 rounded overflow-hidden bg-gray-700">
          {/* Red zone: 0 to removeThreshold */}
          <div
            className="absolute top-0 left-0 h-full bg-red-500"
            style={{ width: `${removeThreshold * 100}%` }}
          />
          {/* Gray zone: removeThreshold to addThreshold */}
          <div
            className="absolute top-0 h-full bg-gray-600"
            style={{
              left: `${removeThreshold * 100}%`,
              width: `${(addThreshold - removeThreshold) * 100}%`,
            }}
          />
          {/* Blue zone: addThreshold to 1.0 */}
          <div
            className="absolute top-0 right-0 h-full bg-blue-500"
            style={{ width: `${(1 - addThreshold) * 100}%` }}
          />
        </div>

        {/* Remove threshold handle (red) */}
        <div
          className="absolute top-0 w-3 h-4 cursor-pointer"
          style={{ left: `calc(${removeThreshold * 100}% - 6px)` }}
          onMouseDown={handleMouseDown('remove')}
        >
          <div className="w-3 h-4 bg-red-500 border border-red-300 rounded shadow hover:scale-125 transition-transform" />
        </div>

        {/* Add threshold handle (blue) */}
        <div
          className="absolute top-0 w-3 h-4 cursor-pointer"
          style={{ left: `calc(${addThreshold * 100}% - 6px)` }}
          onMouseDown={handleMouseDown('add')}
        >
          <div className="w-3 h-4 bg-blue-500 border border-blue-300 rounded shadow hover:scale-125 transition-transform" />
        </div>
      </div>

      {/* Values */}
      <div className="flex gap-1 text-[9px] flex-shrink-0 w-14 justify-end">
        <span className="text-red-400" title="Remove threshold">
          {removeThreshold.toFixed(2)}
        </span>
        <span className="text-gray-600">-</span>
        <span className="text-blue-400" title="Add threshold">
          {addThreshold.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
