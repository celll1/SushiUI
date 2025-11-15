"use client";

import { useState, useRef } from "react";
import Button from "./Button";
import TextareaWithTagSuggestions from "./TextareaWithTagSuggestions";
import { generateTIPOPrompt, TIPOGenerateResponse } from "@/utils/api";
import { getCategoryOrder } from "./CategoryOrderPanel";
import { reorderTagsByCategory } from "@/utils/tagCategorization";

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

export default function TIPOPanel({ onInsert, tipoSettings }: TIPOPanelProps) {
  const [inputPrompt, setInputPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<TIPOGenerateResponse | null>(null);
  const [reorderedOutput, setReorderedOutput] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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

      // Call TIPO API
      const response = await generateTIPOPrompt({
        input_prompt: inputPrompt.trim(),
        model_name: tipoSettings.model_name,
        tag_length: tipoSettings.tag_length,
        nl_length: tipoSettings.nl_length,
        temperature: tipoSettings.temperature,
        top_p: tipoSettings.top_p,
        top_k: tipoSettings.top_k,
        max_new_tokens: tipoSettings.max_new_tokens,
        category_order: categoryOrderIds,
        enabled_categories: enabledCategories,
      });

      setResult(response);

      // Reorder the generated prompt using category order
      const reordered = reorderTagsByCategory(response.generated_prompt, categoryOrder);
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
          {result.parsed && Object.keys(result.parsed).length > 0 && (
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-300">
                Parsed by Category
              </label>
              <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 space-y-2">
                {Object.entries(result.parsed).map(([category, content]) => (
                  <div key={category} className="flex gap-2">
                    <span className="text-xs font-semibold text-blue-400 min-w-[80px]">
                      {category}:
                    </span>
                    <span className="text-xs text-gray-300 break-words">
                      {Array.isArray(content) ? content.join(", ") : content}
                    </span>
                  </div>
                ))}
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
