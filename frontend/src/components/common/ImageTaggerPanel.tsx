"use client";

import { useState, useRef, useEffect } from "react";
import Button from "./Button";
import { predictTags, loadTaggerModel, getTaggerStatus, TaggerPredictionsResponse } from "@/utils/api";

interface ImageTaggerPanelProps {
  onInsert: (content: string) => void;
}

export default function ImageTaggerPanel({ onInsert }: ImageTaggerPanelProps) {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [isTagging, setIsTagging] = useState(false);
  const [predictions, setPredictions] = useState<TaggerPredictionsResponse["predictions"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [genThreshold, setGenThreshold] = useState(0.45);
  const [charThreshold, setCharThreshold] = useState(0.45);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check tagger status on mount
  useEffect(() => {
    checkTaggerStatus();
  }, []);

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
      // Default model path - user should configure this
      const modelPath = "D:/celll1/tagutl/cl_tagger_1_02/model.onnx";
      const tagMappingPath = "D:/celll1/tagutl/cl_tagger_1_02/selected_tags.json";

      await loadTaggerModel(modelPath, tagMappingPath, true);
      setModelLoaded(true);
    } catch (err: any) {
      console.error("[Tagger] Failed to load model:", err);
      setError(err.response?.data?.detail || err.message || "Failed to load tagger model");
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Create preview
    const reader = new FileReader();
    reader.onload = (event) => {
      const dataUrl = event.target?.result as string;
      setImagePreview(dataUrl);

      // Extract base64 (remove data:image/...;base64, prefix)
      const base64 = dataUrl.split(',')[1];
      setImageBase64(base64);
    };
    reader.readAsDataURL(file);

    // Clear previous predictions
    setPredictions(null);
    setSelectedTags(new Set());
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

  const handleInsertTags = () => {
    if (selectedTags.size === 0) return;

    // Convert selected tags to comma-separated string
    const tagsString = Array.from(selectedTags).join(', ');
    onInsert(tagsString);
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

  const renderTagCategory = (categoryName: string, tags: [string, number][]) => {
    if (tags.length === 0) return null;

    return (
      <div key={categoryName} className="mb-4">
        <h4 className="text-sm font-semibold text-gray-300 mb-2 capitalize">
          {categoryName} ({tags.length})
        </h4>
        <div className="flex flex-wrap gap-2">
          {tags.map(([tag, confidence]) => (
            <button
              key={tag}
              onClick={() => handleTagClick(tag)}
              className={`px-2 py-1 rounded text-xs transition-colors ${
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
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-200">Image Tagger</h3>
        {!modelLoaded && (
          <Button onClick={handleLoadModel} variant="secondary" className="text-xs">
            Load Model
          </Button>
        )}
        {modelLoaded && (
          <span className="text-xs text-green-400">Model Loaded</span>
        )}
      </div>

      {/* Image Upload */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-300">
          Upload Image
        </label>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer"
        />
      </div>

      {/* Image Preview */}
      {imagePreview && (
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-300">
            Preview
          </label>
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-2">
            <img
              src={imagePreview}
              alt="Preview"
              className="max-w-full max-h-96 mx-auto object-contain"
            />
          </div>
        </div>
      )}

      {/* Threshold Settings */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
        <h4 className="text-sm font-semibold text-gray-300">Threshold Settings</h4>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">
              General Threshold: {genThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={genThreshold}
              onChange={(e) => setGenThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">
              Character Threshold: {charThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={charThreshold}
              onChange={(e) => setCharThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Predict Button */}
      <div className="flex gap-2">
        <Button
          onClick={handlePredictTags}
          variant="primary"
          disabled={!imageBase64 || isTagging || !modelLoaded}
          className="flex-1"
        >
          {isTagging ? "Tagging..." : "Predict Tags"}
        </Button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900 border border-red-700 rounded-lg p-3">
          <p className="text-sm text-red-200">{error}</p>
        </div>
      )}

      {/* Predictions Display */}
      {predictions && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-gray-300">
              Predictions ({selectedTags.size} selected)
            </h4>
            <div className="flex gap-2">
              <Button onClick={handleSelectAll} variant="secondary" className="text-xs">
                Select All
              </Button>
              <Button onClick={handleDeselectAll} variant="secondary" className="text-xs">
                Deselect All
              </Button>
            </div>
          </div>

          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 max-h-96 overflow-y-auto">
            {renderTagCategory("rating", predictions.rating)}
            {renderTagCategory("quality", predictions.quality)}
            {renderTagCategory("character", predictions.character)}
            {renderTagCategory("copyright", predictions.copyright)}
            {renderTagCategory("artist", predictions.artist)}
            {renderTagCategory("general", predictions.general)}
            {renderTagCategory("meta", predictions.meta)}
            {renderTagCategory("model", predictions.model)}
          </div>

          <Button
            onClick={handleInsertTags}
            variant="primary"
            disabled={selectedTags.size === 0}
          >
            Insert Selected Tags ({selectedTags.size})
          </Button>
        </div>
      )}

      {/* Usage Info */}
      <div className="mt-4 p-3 bg-gray-900 rounded text-xs text-gray-400">
        <p className="mb-1">
          <strong>Usage:</strong>
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>Upload an image to analyze</li>
          <li>Adjust thresholds to filter tags by confidence</li>
          <li>Click tags to select/deselect them</li>
          <li>Insert selected tags into your prompt</li>
          <li>Model: cl_tagger (https://huggingface.co/cella110n/cl_tagger)</li>
        </ul>
      </div>
    </div>
  );
}
