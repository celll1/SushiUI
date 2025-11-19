"use client";

import { useState, useEffect } from "react";
import { reorderPromptByCategory } from "@/utils/tagCategorization";
import Button from "./Button";

export interface TagCategory {
  id: string;
  label: string;
  enabled: boolean;
  randomize?: boolean;
}

const DEFAULT_CATEGORIES: TagCategory[] = [
  { id: "rating", label: "Rating", enabled: true, randomize: false },
  { id: "quality", label: "Quality", enabled: true, randomize: false },
  { id: "count", label: "Count", enabled: true, randomize: false },
  { id: "general", label: "General", enabled: true, randomize: false },
  { id: "character", label: "Character", enabled: true, randomize: false },
  { id: "copyright", label: "Copyright", enabled: true, randomize: false },
  { id: "artist", label: "Artist", enabled: true, randomize: false },
  { id: "meta", label: "Meta", enabled: true, randomize: false },
  { id: "model", label: "Model", enabled: true, randomize: false },
  { id: "unknown", label: "Unknown", enabled: true, randomize: false },
];

const STORAGE_KEY = "tag_category_order";

interface CategoryOrderPanelProps {
  currentPrompt: string;
  onApplyOrder?: (reorderedPrompt: string) => void;
}

export default function CategoryOrderPanel({ currentPrompt, onApplyOrder }: CategoryOrderPanelProps) {
  const [categories, setCategories] = useState<TagCategory[]>(DEFAULT_CATEGORIES);
  const [isReordering, setIsReordering] = useState(false);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

  // Load saved order from localStorage with migration
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed: TagCategory[] = JSON.parse(saved);

        // Migrate: add "unknown" category if it doesn't exist
        const hasUnknown = parsed.some(cat => cat.id === "unknown");
        if (!hasUnknown) {
          parsed.push({ id: "unknown", label: "Unknown", enabled: true });
        }

        // Migrate: add "count" category if it doesn't exist (insert after quality)
        const hasCount = parsed.some(cat => cat.id === "count");
        if (!hasCount) {
          const qualityIndex = parsed.findIndex(cat => cat.id === "quality");
          const insertIndex = qualityIndex >= 0 ? qualityIndex + 1 : 2;
          parsed.splice(insertIndex, 0, { id: "count", label: "Count", enabled: true });
        }

        setCategories(parsed);
      }
    } catch (error) {
      console.error("Failed to load category order:", error);
    }
  }, []);

  // Save order to localStorage whenever it changes
  const saveOrder = (newCategories: TagCategory[]) => {
    setCategories(newCategories);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(newCategories));
    } catch (error) {
      console.error("Failed to save category order:", error);
    }
  };

  const toggleCategory = (index: number) => {
    const newCategories = [...categories];
    newCategories[index].enabled = !newCategories[index].enabled;
    saveOrder(newCategories);
  };

  const toggleRandomize = (index: number) => {
    const newCategories = [...categories];
    newCategories[index].randomize = !newCategories[index].randomize;
    saveOrder(newCategories);
  };

  const resetToDefault = () => {
    if (confirm("Reset category order to default?")) {
      saveOrder(DEFAULT_CATEGORIES);
    }
  };

  const handleApplyOrder = async () => {
    if (!currentPrompt.trim()) {
      alert("Current prompt is empty");
      return;
    }

    setIsReordering(true);
    try {
      const reordered = await reorderPromptByCategory(currentPrompt, categories);
      if (onApplyOrder) {
        onApplyOrder(reordered);
      }
    } catch (error) {
      console.error("Failed to reorder prompt:", error);
      alert("Failed to reorder prompt. Check console for details.");
    } finally {
      setIsReordering(false);
    }
  };

  // Drag and Drop handlers
  const handleDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    setDragOverIndex(index);
  };

  const handleDragLeave = () => {
    setDragOverIndex(null);
  };

  const handleDrop = (e: React.DragEvent, dropIndex: number) => {
    e.preventDefault();

    if (draggedIndex === null || draggedIndex === dropIndex) {
      setDraggedIndex(null);
      setDragOverIndex(null);
      return;
    }

    const newCategories = [...categories];
    const [draggedItem] = newCategories.splice(draggedIndex, 1);
    newCategories.splice(dropIndex, 0, draggedItem);

    saveOrder(newCategories);
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  const handleDragEnd = () => {
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-200">Tag Category Order</h3>
          <p className="text-sm text-gray-400 mt-1">
            Reorder tags in the current prompt by category
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={handleApplyOrder}
            variant="primary"
            size="sm"
            disabled={isReordering || !currentPrompt.trim()}
          >
            {isReordering ? "Reordering..." : "Apply to Current Prompt"}
          </Button>
          <button
            onClick={resetToDefault}
            className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700"
          >
            Reset to Default
          </button>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-2 sm:p-4">
        <div className="space-y-0.5 sm:space-y-1">
          {categories.map((category, index) => (
            <div
              key={category.id}
              draggable
              onDragStart={() => handleDragStart(index)}
              onDragOver={(e) => handleDragOver(e, index)}
              onDragLeave={handleDragLeave}
              onDrop={(e) => handleDrop(e, index)}
              onDragEnd={handleDragEnd}
              className={`flex items-center gap-2 sm:gap-3 px-2 sm:px-3 py-1.5 sm:py-2 rounded transition-all cursor-move ${
                draggedIndex === index
                  ? "opacity-50 bg-gray-600"
                  : dragOverIndex === index
                  ? "bg-blue-700"
                  : "bg-gray-700 hover:bg-gray-650"
              }`}
            >
              {/* Drag Handle */}
              <span className="text-gray-400 text-xs sm:text-sm">⋮⋮</span>

              {/* Category Label */}
              <span
                className={`text-xs sm:text-sm font-medium flex-1 ${
                  category.enabled ? "text-gray-200" : "text-gray-500"
                }`}
              >
                {category.label}
              </span>

              {/* Enable Checkbox */}
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={category.enabled}
                  onChange={() => toggleCategory(index)}
                  className="w-3 h-3 sm:w-4 sm:h-4 rounded border-gray-500 text-blue-600 focus:ring-blue-500 focus:ring-offset-gray-800"
                />
                <span className="text-[10px] sm:text-xs text-gray-400">Enable</span>
              </label>

              {/* Randomize Checkbox */}
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={category.randomize || false}
                  onChange={() => toggleRandomize(index)}
                  disabled={!category.enabled}
                  className="w-3 h-3 sm:w-4 sm:h-4 rounded border-gray-500 text-green-600 focus:ring-green-500 focus:ring-offset-gray-800 disabled:opacity-30"
                />
                <span className={`text-[10px] sm:text-xs text-gray-400 ${!category.enabled ? "opacity-30" : ""}`}>
                  Random
                </span>
              </label>
            </div>
          ))}
        </div>

        <div className="mt-4 p-3 bg-gray-900 rounded text-xs text-gray-400">
          <p className="mb-1">
            <strong>Usage:</strong>
          </p>
          <ul className="list-disc list-inside space-y-1">
            <li>Drag and drop rows to reorder categories</li>
            <li>Enable: Include category in reordering</li>
            <li>Random: Randomize tag order within category</li>
            <li>Click "Apply to Current Prompt" to reorder tags in the prompt above</li>
            <li>Category order also affects TIPO tag generation output</li>
            <li>Changes are saved automatically to localStorage</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

/**
 * Get current category order from localStorage
 */
export function getCategoryOrder(): TagCategory[] {
  if (typeof window === "undefined") return DEFAULT_CATEGORIES;

  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed: TagCategory[] = JSON.parse(saved);

      // Migrate: add "unknown" category if it doesn't exist
      const hasUnknown = parsed.some(cat => cat.id === "unknown");
      if (!hasUnknown) {
        parsed.push({ id: "unknown", label: "Unknown", enabled: true });
      }

      // Migrate: add "count" category if it doesn't exist (insert after quality)
      const hasCount = parsed.some(cat => cat.id === "count");
      if (!hasCount) {
        const qualityIndex = parsed.findIndex(cat => cat.id === "quality");
        const insertIndex = qualityIndex >= 0 ? qualityIndex + 1 : 2;
        parsed.splice(insertIndex, 0, { id: "count", label: "Count", enabled: true });
      }

      return parsed;
    }
  } catch (error) {
    console.error("Failed to load category order:", error);
  }

  return DEFAULT_CATEGORIES;
}
