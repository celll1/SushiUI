"use client";

import { useState, useRef } from "react";
import Button from "./Button";
import TextareaWithTagSuggestions from "./TextareaWithTagSuggestions";
import { generateTIPOPrompt, TIPOGenerateResponse } from "@/utils/api";
import { getCategoryOrder } from "./CategoryOrderPanel";
import { reorderPromptByCategory } from "@/utils/tagCategorization";

interface TIPOPanelProps {
  onInsert: (content: string) => void;
  tipoSettings: {
    model_name: string;
    tag_length: string;
    nl_length: string;
    temperature: number;
    top_p: number;
    top_k: number;
    max_new_tokens: number;
  };
}

export default function TIPOPanel({ onInsert, tipoSettings: initialSettings }: TIPOPanelProps) {
  const [inputPrompt, setInputPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<TIPOGenerateResponse | null>(null);
  const [reorderedOutput, setReorderedOutput] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Local settings state (editable in this panel)
  const [localSettings, setLocalSettings] = useState({
    model_name: initialSettings.model_name,
    tag_length: initialSettings.tag_length,
    nl_length: initialSettings.nl_length,
    temperature: initialSettings.temperature,
    top_p: initialSettings.top_p,
    top_k: initialSettings.top_k,
    max_new_tokens: initialSettings.max_new_tokens,
  });

  const handleGenerate = async () => {
    if (!inputPrompt.trim()) {
      setError("Please enter an input prompt");
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      // Get category order settings
      const categoryOrder = getCategoryOrder();
      const categoryOrderIds = categoryOrder.map(cat => cat.id);
      const enabledCategories = categoryOrder.reduce((acc, cat) => {
        acc[cat.id] = cat.enabled;
        return acc;
      }, {} as Record<string, boolean>);

      // Call TIPO API with local settings
      const response = await generateTIPOPrompt({
        input_prompt: inputPrompt.trim(),
        model_name: localSettings.model_name,
        tag_length: localSettings.tag_length,
        nl_length: localSettings.nl_length,
        temperature: localSettings.temperature,
        top_p: localSettings.top_p,
        top_k: localSettings.top_k,
        max_new_tokens: localSettings.max_new_tokens,
        category_order: categoryOrderIds,
        enabled_categories: enabledCategories,
      });

      setResult(response);

      // Reorder the generated prompt using category order
      const reordered = await reorderPromptByCategory(response.generated_prompt, categoryOrder);
      setReorderedOutput(reordered);

    } catch (err: any) {
      console.error("[TIPO] Generation failed:", err);
      setError(err.response?.data?.detail || err.message || "Failed to generate TIPO prompt");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleInsert = () => {
    if (reorderedOutput) {
      onInsert(reorderedOutput);
    }
  };

  const handleClear = () => {
    setInputPrompt("");
    setResult(null);
    setReorderedOutput("");
    setError(null);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-200">TIPO - Tag Inference & Prompt Optimization</h3>
      </div>

      {/* Settings Section */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
        <h4 className="text-sm font-semibold text-gray-300">Generation Settings</h4>

        <div className="grid grid-cols-2 gap-3">
          {/* Model Selection */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Model</label>
            <select
              value={localSettings.model_name}
              onChange={(e) => setLocalSettings({ ...localSettings, model_name: e.target.value })}
              className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-xs"
            >
              <option value="KBlueLeaf/TIPO-500M">TIPO-500M</option>
              <option value="KBlueLeaf/TIPO-500M-ft">TIPO-500M-ft</option>
              <option value="KBlueLeaf/TIPO-200M">TIPO-200M</option>
              <option value="KBlueLeaf/TIPO-200M-ft2">TIPO-200M-ft2</option>
              <option value="KBlueLeaf/TIPO-200M-ft">TIPO-200M-ft</option>
            </select>
          </div>

          {/* Tag Length */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Tag Length</label>
            <select
              value={localSettings.tag_length}
              onChange={(e) => setLocalSettings({ ...localSettings, tag_length: e.target.value })}
              className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-xs"
            >
              <option value="very_short">Very Short</option>
              <option value="short">Short</option>
              <option value="long">Long</option>
              <option value="very_long">Very Long</option>
            </select>
          </div>

          {/* NL Length */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">NL Length</label>
            <select
              value={localSettings.nl_length}
              onChange={(e) => setLocalSettings({ ...localSettings, nl_length: e.target.value })}
              className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-xs"
            >
              <option value="very_short">Very Short</option>
              <option value="short">Short</option>
              <option value="long">Long</option>
              <option value="very_long">Very Long</option>
            </select>
          </div>

          {/* Max Tokens */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">
              Max Tokens: {localSettings.max_new_tokens}
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

        {/* Advanced Settings */}
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-gray-400 hover:text-gray-300">
            Advanced Settings
          </summary>
          <div className="mt-2 space-y-2">
            {/* Temperature */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">
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
              <label className="block text-xs text-gray-400 mb-1">
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
              <label className="block text-xs text-gray-400 mb-1">
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
          </div>
        </details>
      </div>

      {/* Input Section */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-300">
          Input Prompt
        </label>
        <TextareaWithTagSuggestions
          ref={textareaRef}
          value={inputPrompt}
          onChange={(e) => setInputPrompt(e.target.value)}
          placeholder="Enter tags or natural language description... (e.g., 1girl, red hair, standing in garden)"
          rows={3}
          enableWeightControl={false}
        />
        <div className="flex gap-2">
          <Button
            onClick={handleGenerate}
            variant="primary"
            disabled={isGenerating || !inputPrompt.trim()}
          >
            {isGenerating ? "Generating..." : "Generate"}
          </Button>
          <Button
            onClick={handleClear}
            variant="secondary"
            disabled={isGenerating}
          >
            Clear
          </Button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900 border border-red-700 rounded-lg p-3">
          <p className="text-sm text-red-200">{error}</p>
        </div>
      )}

      {/* Results Section */}
      {result && (
        <div className="space-y-4">
          {/* Raw Output */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-300">
              Raw TIPO Output
            </label>
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 max-h-40 overflow-y-auto">
              <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words">
                {result.raw_output}
              </pre>
            </div>
          </div>

          {/* Parsed Categories */}
          {result.parsed && typeof result.parsed === 'object' && (
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-300">
                Parsed by Category
              </label>
              <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 space-y-2">
                {Object.entries(result.parsed)
                  .map(([category, content]) => {
                    // Skip null/undefined/empty content
                    if (content === null || content === undefined || content === '') {
                      return null;
                    }

                    // Format content based on type
                    let displayContent: string;
                    if (Array.isArray(content)) {
                      if (content.length === 0) return null;
                      displayContent = content.join(", ");
                    } else if (typeof content === 'object') {
                      // Handle nested objects
                      displayContent = JSON.stringify(content, null, 2);
                      if (displayContent === '{}' || displayContent === '[]') {
                        return null;
                      }
                    } else {
                      displayContent = String(content);
                      if (!displayContent.trim()) return null;
                    }

                    return (
                      <div key={category} className="flex gap-2">
                        <span className="text-xs font-semibold text-blue-400 min-w-[80px]">
                          {category}:
                        </span>
                        <span className="text-xs text-gray-300 break-words whitespace-pre-wrap">
                          {displayContent}
                        </span>
                      </div>
                    );
                  })
                  .filter(Boolean)}
              </div>
            </div>
          )}

          {/* Reordered Output */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-300">
              Reordered Output (Ready to Insert)
            </label>
            <div className="bg-gray-900 border border-gray-600 rounded-lg p-3 max-h-60 overflow-y-auto">
              <p className="text-sm text-gray-100 whitespace-pre-wrap break-words">
                {reorderedOutput}
              </p>
            </div>
            <Button
              onClick={handleInsert}
              variant="primary"
            >
              Insert into Prompt
            </Button>
          </div>
        </div>
      )}

      {/* Usage Info */}
      <div className="mt-4 p-3 bg-gray-900 rounded text-xs text-gray-400">
        <p className="mb-1">
          <strong>Usage:</strong>
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>Enter a short prompt (tags or natural language)</li>
          <li>TIPO will expand it into a detailed, well-structured prompt</li>
          <li>Output is automatically reordered by your category settings</li>
          <li>Click "Insert" to add the result to your main prompt at cursor position</li>
          <li>Configure TIPO settings in the Settings panel</li>
        </ul>
      </div>
    </div>
  );
}
