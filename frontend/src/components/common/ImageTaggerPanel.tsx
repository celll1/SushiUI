"use client";

import { useState, useRef, useEffect } from "react";
import Button from "./Button";
import { predictTags, loadTaggerModel, getTaggerStatus, TaggerPredictionsResponse } from "@/utils/api";

interface ImageTaggerPanelProps {
  onInsert: (content: string) => void;
  onOverwrite: (content: string) => void;
  currentPrompt: string;
}

interface CategoryThreshold {
  id: string;
  label: string;
  threshold: number;
  enabled: boolean;
}

const DEFAULT_THRESHOLDS: CategoryThreshold[] = [
  { id: "rating", label: "Rating", threshold: 0.45, enabled: true },
  { id: "quality", label: "Quality", threshold: 0.45, enabled: true },
  { id: "character", label: "Character", threshold: 0.45, enabled: true },
  { id: "copyright", label: "Copyright", threshold: 0.45, enabled: true },
  { id: "artist", label: "Artist", threshold: 0.45, enabled: true },
  { id: "general", label: "General", threshold: 0.45, enabled: true },
  { id: "meta", label: "Meta", threshold: 0.45, enabled: true },
  { id: "model", label: "Model", threshold: 0.45, enabled: true },
];

const STORAGE_KEY_THRESHOLDS = "tagger_thresholds";
const STORAGE_KEY_CATEGORY_ORDER = "tagger_category_order";

export default function ImageTaggerPanel({ onInsert, onOverwrite, currentPrompt }: ImageTaggerPanelProps) {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [isTagging, setIsTagging] = useState(false);
  const [predictions, setPredictions] = useState<TaggerPredictionsResponse["predictions"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const taggerContainerRef = useRef<HTMLDivElement>(null);

  // Category thresholds and order
  const [categoryThresholds, setCategoryThresholds] = useState<CategoryThreshold[]>(DEFAULT_THRESHOLDS);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

  // Load saved settings
  useEffect(() => {
    checkTaggerStatus();

    // Load thresholds
    const savedThresholds = localStorage.getItem(STORAGE_KEY_THRESHOLDS);
    if (savedThresholds) {
      try {
        setCategoryThresholds(JSON.parse(savedThresholds));
      } catch (error) {
        console.error("Failed to load thresholds:", error);
      }
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Enter: Insert
      if (e.ctrlKey && e.key === "Enter" && !e.shiftKey) {
        if (selectedTags.size > 0 && predictions) {
          e.preventDefault();
          handleInsertTags();
        }
      }
      // Ctrl+Shift+Enter: Overwrite
      if (e.ctrlKey && e.shiftKey && e.key === "Enter") {
        if (selectedTags.size > 0 && predictions) {
          e.preventDefault();
          handleOverwriteTags();
        }
      }
    };

    const container = taggerContainerRef.current;
    if (container) {
      container.addEventListener('keydown', handleKeyDown);
      return () => container.removeEventListener('keydown', handleKeyDown);
    }
  }, [selectedTags, predictions]);

  // Save thresholds when changed
  const saveThresholds = (newThresholds: CategoryThreshold[]) => {
    setCategoryThresholds(newThresholds);
    localStorage.setItem(STORAGE_KEY_THRESHOLDS, JSON.stringify(newThresholds));
  };

  const checkTaggerStatus = async () => {
    try {
      const status = await getTaggerStatus();
      setModelLoaded(status.loaded);
    } catch (err) {
      console.error("[Tagger] Failed to check status:", err);
    }
  };

  const handleLoadModel = async () => {
    try {
      setError(null);
      await loadTaggerModel(
        undefined,
        undefined,
        true,
        true,
        "cella110n/cl_tagger",
        "cl_tagger_1_02"
      );
      setModelLoaded(true);
    } catch (err: any) {
      console.error("[Tagger] Failed to load model:", err);
      setError(err.response?.data?.detail || err.message || "Failed to load tagger model");
    }
  };

  const processImageFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const dataUrl = event.target?.result as string;
      setImagePreview(dataUrl);
      const base64 = dataUrl.split(',')[1];
      setImageBase64(base64);
    };
    reader.readAsDataURL(file);

    setPredictions(null);
    setSelectedTags(new Set());
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processImageFile(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file) processImageFile(file);
  };

  const handlePredictTags = async () => {
    if (!imageBase64) return;
    if (!modelLoaded) {
      setError("Please load the model first");
      return;
    }

    setIsTagging(true);
    setError(null);

    try {
      // Use general threshold for general tags, character threshold for others
      const genThreshold = categoryThresholds.find(c => c.id === "general")?.threshold || 0.45;
      const charThreshold = categoryThresholds.find(c => c.id === "character")?.threshold || 0.45;

      const response = await predictTags(imageBase64, genThreshold, charThreshold);
      setPredictions(response.predictions);

      // Auto-select all tags by default
      const allTags = new Set<string>();
      Object.values(response.predictions).forEach(categoryTags => {
        categoryTags.forEach(([tag, _]) => allTags.add(tag));
      });
      setSelectedTags(allTags);
    } catch (err: any) {
      console.error("[Tagger] Prediction failed:", err);
      setError(err.response?.data?.detail || err.message || "Failed to predict tags");
    } finally {
      setIsTagging(false);
    }
  };

  const handleTagClick = (tag: string) => {
    const newSelected = new Set(selectedTags);
    if (newSelected.has(tag)) {
      newSelected.delete(tag);
    } else {
      newSelected.add(tag);
    }
    setSelectedTags(newSelected);
  };

  const getSortedTags = (): string => {
    if (!predictions) return "";

    const tagsByCategory: { [key: string]: string[] } = {};

    // Group selected tags by category
    Object.entries(predictions).forEach(([category, tags]) => {
      tagsByCategory[category] = tags
        .filter(([tag, _]) => selectedTags.has(tag))
        .map(([tag, _]) => tag);
    });

    // Sort by category order
    const orderedTags: string[] = [];
    categoryThresholds.forEach(catThreshold => {
      if (catThreshold.enabled && tagsByCategory[catThreshold.id]) {
        orderedTags.push(...tagsByCategory[catThreshold.id]);
      }
    });

    return orderedTags.join(", ");
  };

  const handleInsertTags = () => {
    const tagsString = getSortedTags();
    if (tagsString) onInsert(tagsString);
  };

  const handleOverwriteTags = () => {
    const tagsString = getSortedTags();
    if (tagsString) onOverwrite(tagsString);
  };

  const handleSelectAll = () => {
    if (!predictions) return;
    const allTags = new Set<string>();
    Object.values(predictions).forEach(categoryTags => {
      categoryTags.forEach(([tag, _]) => allTags.add(tag));
    });
    setSelectedTags(allTags);
  };

  const handleDeselectAll = () => {
    setSelectedTags(new Set());
  };

  const updateThreshold = (index: number, threshold: number) => {
    const newThresholds = [...categoryThresholds];
    newThresholds[index].threshold = threshold;
    saveThresholds(newThresholds);
  };

  const toggleCategory = (index: number) => {
    const newThresholds = [...categoryThresholds];
    newThresholds[index].enabled = !newThresholds[index].enabled;
    saveThresholds(newThresholds);
  };

  const resetThresholds = () => {
    if (confirm("Reset all thresholds to default (0.45)?")) {
      saveThresholds(DEFAULT_THRESHOLDS);
    }
  };

  // Drag and Drop handlers for category order
  const handleCategoryDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleCategoryDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    setDragOverIndex(index);
  };

  const handleCategoryDragLeave = () => {
    setDragOverIndex(null);
  };

  const handleCategoryDrop = (e: React.DragEvent, dropIndex: number) => {
    e.preventDefault();

    if (draggedIndex === null || draggedIndex === dropIndex) {
      setDraggedIndex(null);
      setDragOverIndex(null);
      return;
    }

    const newThresholds = [...categoryThresholds];
    const [draggedItem] = newThresholds.splice(draggedIndex, 1);
    newThresholds.splice(dropIndex, 0, draggedItem);

    saveThresholds(newThresholds);
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  const handleCategoryDragEnd = () => {
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  const renderTagCategory = (categoryName: string, tags: [string, number][]) => {
    if (tags.length === 0) return null;

    return (
      <div key={categoryName} className="mb-3">
        <h5 className="text-xs font-semibold text-gray-400 mb-1.5 capitalize">
          {categoryName} ({tags.length})
        </h5>
        <div className="flex flex-wrap gap-1.5">
          {tags.map(([tag, confidence]) => (
            <button
              key={tag}
              onClick={() => handleTagClick(tag)}
              className={`px-2 py-0.5 rounded text-xs transition-colors ${
                selectedTags.has(tag)
                  ? "bg-blue-600 text-white"
                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"
              }`}
              title={`Confidence: ${(confidence * 100).toFixed(1)}%`}
            >
              {tag}
              <span className="ml-1 text-gray-400 text-[10px]">
                {(confidence * 100).toFixed(0)}%
              </span>
            </button>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div ref={taggerContainerRef} className="grid grid-cols-3 gap-4 h-full" tabIndex={-1}>
      {/* Left Column: Image Upload */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-200">Image</h3>
          {!modelLoaded && (
            <Button onClick={handleLoadModel} variant="secondary" className="text-xs px-2 py-1">
              Load Model
            </Button>
          )}
          {modelLoaded && (
            <span className="text-xs text-green-400">✓ Loaded</span>
          )}
        </div>

        {/* Drag & Drop Area */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
            isDragging
              ? 'border-blue-500 bg-gray-700'
              : 'border-gray-600 bg-gray-800'
          } ${imagePreview ? 'p-2' : 'p-8'}`}
        >
          {imagePreview ? (
            <img
              src={imagePreview}
              alt="Preview"
              className="w-full h-auto object-contain rounded"
            />
          ) : (
            <div className="text-center text-gray-400 text-sm">
              <p className="mb-1">Drop image here</p>
              <p className="text-xs">or click to browse</p>
            </div>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="hidden"
        />

        <Button
          onClick={handlePredictTags}
          variant="primary"
          disabled={!imageBase64 || isTagging || !modelLoaded}
          className="w-full"
        >
          {isTagging ? "Tagging..." : "Predict Tags"}
        </Button>

        {error && (
          <div className="bg-red-900 border border-red-700 rounded p-2 text-xs text-red-200">
            {error}
          </div>
        )}
      </div>

      {/* Middle Column: Category Thresholds & Order */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-200">Category Settings</h3>
          <Button onClick={resetThresholds} variant="secondary" className="text-xs px-2 py-1">
            Reset
          </Button>
        </div>

        <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 space-y-2 max-h-[calc(100vh-300px)] overflow-y-auto">
          {categoryThresholds.map((cat, index) => (
            <div
              key={cat.id}
              draggable
              onDragStart={() => handleCategoryDragStart(index)}
              onDragOver={(e) => handleCategoryDragOver(e, index)}
              onDragLeave={handleCategoryDragLeave}
              onDrop={(e) => handleCategoryDrop(e, index)}
              onDragEnd={handleCategoryDragEnd}
              className={`bg-gray-700 rounded p-2 cursor-move transition-all ${
                draggedIndex === index ? 'opacity-50' : ''
              } ${
                dragOverIndex === index ? 'border-2 border-blue-500' : 'border border-gray-600'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 text-xs">⋮⋮</span>
                  <input
                    type="checkbox"
                    checked={cat.enabled}
                    onChange={() => toggleCategory(index)}
                    className="cursor-pointer"
                  />
                  <span className={`text-xs font-medium ${cat.enabled ? 'text-gray-200' : 'text-gray-500'}`}>
                    {cat.label}
                  </span>
                </div>
                <span className="text-xs text-gray-400">{cat.threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={cat.threshold}
                onChange={(e) => updateThreshold(index, parseFloat(e.target.value))}
                disabled={!cat.enabled}
                className="w-full h-1"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Right Column: Results & Actions */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-200">
            Results {predictions ? `(${selectedTags.size} selected)` : ""}
          </h3>
          {predictions && (
            <div className="flex gap-1">
              <Button onClick={handleSelectAll} variant="secondary" className="text-xs px-2 py-1">
                All
              </Button>
              <Button onClick={handleDeselectAll} variant="secondary" className="text-xs px-2 py-1">
                None
              </Button>
            </div>
          )}
        </div>

        {predictions ? (
          <>
            {/* Sorted Results Display */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 max-h-[calc(100vh-400px)] overflow-y-auto">
              {categoryThresholds.map(catThreshold => {
                if (!catThreshold.enabled) return null;
                const tags = predictions[catThreshold.id as keyof typeof predictions];
                if (!tags || tags.length === 0) return null;
                return renderTagCategory(catThreshold.id, tags);
              })}
            </div>

            {/* Action Buttons */}
            <div className="space-y-2">
              <Button
                onClick={handleInsertTags}
                variant="primary"
                disabled={selectedTags.size === 0}
                className="w-full"
              >
                Insert ({selectedTags.size})
              </Button>
              <Button
                onClick={handleOverwriteTags}
                variant="secondary"
                disabled={selectedTags.size === 0}
                className="w-full"
              >
                Overwrite ({selectedTags.size})
              </Button>
              <div className="text-xs text-gray-500 text-center">
                Ctrl+Enter: Insert | Ctrl+Shift+Enter: Overwrite
              </div>
            </div>
          </>
        ) : (
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-8 text-center text-gray-500 text-sm">
            Upload and predict an image to see tags
          </div>
        )}
      </div>
    </div>
  );
}
