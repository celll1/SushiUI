"use client";

import { useState, useRef, forwardRef, useImperativeHandle, useCallback } from "react";
import Button from "./Button";
import TextareaWithTagSuggestions from "./TextareaWithTagSuggestions";
import { generateTIPOPrompt, TIPOGenerateResponse } from "@/utils/api";
import { getCategoryOrder, TagCategory } from "./CategoryOrderPanel";
import { reorderPromptByCategory } from "@/utils/tagCategorization";

// TIPO-specific category mapping
const TIPO_CATEGORY_MAP: Record<string, string> = {
  "rating": "rating",
  "quality": "quality",
  "count": "special", // TIPO returns count tags in "special"
  "character": "characters",
  "copyright": "copyrights",
  "artist": "artist",
  "general": "general",
  "meta": "meta",
};

interface TIPOCategoryOrder {
  id: string;
  label: string;
  enabled: boolean;
  randomize?: boolean;
}

interface TIPOPanelProps {
  onInsert: (content: string) => void;
  onOverwrite: (content: string) => void;
  currentPrompt: string;
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

export interface TIPOPanelRef {
  hasResult: () => boolean;
  insertResult: () => void;
  overwriteResult: () => void;
  copyFromMain: () => void;
}

const TIPOPanel = forwardRef<TIPOPanelRef, TIPOPanelProps>(({ onInsert, onOverwrite, currentPrompt, tipoSettings: initialSettings }, ref) => {
  const [inputPrompt, setInputPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<TIPOGenerateResponse | null>(null);
  const [reorderedOutput, setReorderedOutput] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

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

  // BAN_TAGS state (comma-separated tags to exclude)
  const [banTags, setBanTags] = useState("monochrome, grayscale");

  // Category order settings (from CategoryOrderPanel)
  const [categoryOrder, setCategoryOrder] = useState<TIPOCategoryOrder[]>(() => {
    const savedOrder = getCategoryOrder();
    return savedOrder.map(cat => ({
      id: cat.id,
      label: cat.label,
      enabled: cat.enabled,
      randomize: cat.randomize || false,
    }));
  });

  // Reorder TIPO output based on category order
  const reorderTIPOOutput = (rawOutput: any): string => {
    if (!rawOutput || typeof rawOutput !== 'object') return '';

    const tagsByCategory: Record<string, string[]> = {};

    // Extract tags from TIPO output structure
    for (const [tipoKey, content] of Object.entries(rawOutput)) {
      // Skip non-array fields and special fields
      if (!Array.isArray(content) || tipoKey === 'target' || tipoKey === 'tag' || tipoKey === 'extended') {
        continue;
      }

      // Map TIPO category to our category system
      let categoryId = tipoKey;
      for (const [ourCat, tipoCat] of Object.entries(TIPO_CATEGORY_MAP)) {
        if (tipoCat === tipoKey) {
          categoryId = ourCat;
          break;
        }
      }

      if (content.length > 0) {
        tagsByCategory[categoryId] = content as string[];
      }
    }

    // Reorder based on category order
    const orderedTags: string[] = [];
    for (const { id, enabled, randomize } of categoryOrder) {
      if (enabled && tagsByCategory[id]) {
        let tags = [...tagsByCategory[id]];

        // Apply randomization if enabled
        if (randomize) {
          tags = tags.sort(() => Math.random() - 0.5);
        }

        orderedTags.push(...tags);
      }
    }

    return orderedTags.join(', ');
  };

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
        ban_tags: banTags.trim(),
        category_order: categoryOrderIds,
        enabled_categories: enabledCategories,
      });

      console.log('[TIPO Panel] Response:', response);
      console.log('[TIPO Panel] Parsed type:', typeof response.parsed);
      console.log('[TIPO Panel] Parsed value:', response.parsed);
      console.log('[TIPO Panel] Raw output:', response.raw_output);

      setResult(response);

      // Reorder the TIPO output using our category order
      const reordered = reorderTIPOOutput(response.raw_output);
      setReorderedOutput(reordered);

      // Blur input prompt (remove focus)
      if (textareaRef.current) {
        textareaRef.current.blur();
      }

      // Auto-scroll to result after generation and focus it
      setTimeout(() => {
        if (resultRef.current) {
          resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          resultRef.current.focus();
        }
      }, 100);

    } catch (err: any) {
      console.error("[TIPO] Generation failed:", err);
      setError(err.response?.data?.detail || err.message || "Failed to generate TIPO prompt");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleInsert = useCallback(() => {
    if (reorderedOutput) {
      onInsert(reorderedOutput);
    }
  }, [reorderedOutput, onInsert]);

  const handleOverwrite = useCallback(() => {
    if (reorderedOutput) {
      onOverwrite(reorderedOutput);
    }
  }, [reorderedOutput, onOverwrite]);

  const handleCopyFromMain = useCallback(() => {
    setInputPrompt(currentPrompt);
    // Focus on input prompt
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }, 0);
  }, [currentPrompt]);

  const handleClear = () => {
    setInputPrompt("");
    setResult(null);
    setReorderedOutput("");
    setError(null);
  };

  // Expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    hasResult: () => !!reorderedOutput,
    insertResult: handleInsert,
    overwriteResult: handleOverwrite,
    copyFromMain: handleCopyFromMain,
  }), [reorderedOutput, handleInsert, handleOverwrite, handleCopyFromMain]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-200">TIPO - Tag Inference & Prompt Optimization</h3>
      </div>

      {/* Settings Section */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
        <h4 className="text-sm font-semibold text-gray-300">Generation Settings</h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
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

            {/* BAN_TAGS */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                BAN_TAGS (comma-separated tags to exclude)
              </label>
              <TextareaWithTagSuggestions
                value={banTags}
                onChange={(e) => setBanTags(e.target.value)}
                placeholder="e.g., monochrome, grayscale"
                rows={2}
                enableWeightControl={false}
              />
            </div>
          </div>
        </details>

        {/* Category Order */}
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-gray-400 hover:text-gray-300">
            Output Category Order
          </summary>
          <div className="mt-2">
            <div className="flex flex-wrap gap-2">
              {categoryOrder.map((category, index) => (
                <div
                  key={category.id}
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.effectAllowed = 'move';
                    e.dataTransfer.setData('text/plain', index.toString());
                  }}
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    const dragIndex = parseInt(e.dataTransfer.getData('text/plain'));
                    if (dragIndex === index) return;

                    const newOrder = [...categoryOrder];
                    const [removed] = newOrder.splice(dragIndex, 1);
                    newOrder.splice(index, 0, removed);
                    setCategoryOrder(newOrder);
                  }}
                  className="flex flex-col items-center gap-2 px-4 py-3 rounded-lg transition-all cursor-move min-w-[100px] bg-gray-700 hover:bg-gray-650"
                >
                  {/* Drag Handle Icon */}
                  <div className="text-gray-400">
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 8h16M4 16h16"
                      />
                    </svg>
                  </div>

                  {/* Category Label */}
                  <div className="text-center">
                    <span
                      className={`text-sm font-medium ${
                        category.enabled ? "text-gray-200" : "text-gray-500"
                      }`}
                    >
                      {category.label}
                    </span>
                  </div>

                  {/* Control Buttons */}
                  <div className="flex flex-col gap-1">
                    {/* Enable/Disable Checkbox */}
                    <label className="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={category.enabled}
                        onChange={(e) => {
                          const newOrder = [...categoryOrder];
                          newOrder[index].enabled = e.target.checked;
                          setCategoryOrder(newOrder);
                        }}
                        className="w-3 h-3 rounded border-gray-500 text-blue-600 focus:ring-blue-500 focus:ring-offset-gray-800"
                      />
                      <span>Enable</span>
                    </label>

                    {/* Randomize Checkbox */}
                    <label className="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={category.randomize || false}
                        onChange={(e) => {
                          const newOrder = [...categoryOrder];
                          newOrder[index].randomize = e.target.checked;
                          setCategoryOrder(newOrder);
                        }}
                        disabled={!category.enabled}
                        className="w-3 h-3 rounded border-gray-500 text-green-600 focus:ring-green-500 focus:ring-offset-gray-800 disabled:opacity-30"
                      />
                      <span className={!category.enabled ? "opacity-30" : ""}>Random</span>
                    </label>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Drag and drop cards to reorder • Enable: Include in output • Random: Randomize order within category
            </p>
          </div>
        </details>
      </div>

      {/* Input Section */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="block text-sm font-medium text-gray-300">
            Input Prompt
          </label>
          <Button
            onClick={handleCopyFromMain}
            variant="secondary"
            className="text-xs py-1 px-2"
          >
            Copy from Main (Ctrl+M)
          </Button>
        </div>
        <TextareaWithTagSuggestions
          ref={textareaRef}
          value={inputPrompt}
          onChange={(e) => setInputPrompt(e.target.value)}
          onKeyDown={(e) => {
            // Ctrl+Enter: Start generation
            if (e.ctrlKey && e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (inputPrompt.trim() && !isGenerating) {
                handleGenerate();
              }
            }
            // Ctrl+M: Copy from main
            if (e.ctrlKey && e.key === "m") {
              e.preventDefault();
              handleCopyFromMain();
            }
          }}
          placeholder="Enter tags or natural language description... (Ctrl+Enter to generate)"
          rows={3}
          enableWeightControl={false}
        />
        <div className="flex gap-2">
          <Button
            onClick={handleGenerate}
            variant="primary"
            disabled={isGenerating || !inputPrompt.trim()}
          >
            {isGenerating ? "Generating..." : "Generate (Ctrl+Enter)"}
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
        <div
          className="space-y-4"
          ref={resultRef}
          tabIndex={-1}
          onKeyDown={(e) => {
            // Ctrl+Enter: Insert
            if (e.ctrlKey && e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleInsert();
            }
            // Ctrl+Shift+Enter: Overwrite
            if (e.ctrlKey && e.shiftKey && e.key === "Enter") {
              e.preventDefault();
              handleOverwrite();
            }
          }}
        >
          {/* Raw Output */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-300">
              Raw TIPO Output
            </label>
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 max-h-40 overflow-y-auto">
              <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words">
                {typeof result.raw_output === 'string'
                  ? result.raw_output
                  : JSON.stringify(result.raw_output, null, 2)}
              </pre>
            </div>
          </div>

          {/* Parsed Categories */}
          {result.parsed && typeof result.parsed === 'object' && (
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-300">
                Parsed by Category
              </label>
              <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words">
                  {JSON.stringify(result.parsed, null, 2)}
                </pre>
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
            <div className="flex gap-2">
              <Button
                onClick={handleInsert}
                variant="primary"
              >
                Insert (Ctrl+Enter)
              </Button>
              <Button
                onClick={handleOverwrite}
                variant="secondary"
              >
                Overwrite (Ctrl+Shift+Enter)
              </Button>
            </div>
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
});

TIPOPanel.displayName = "TIPOPanel";

export default TIPOPanel;
