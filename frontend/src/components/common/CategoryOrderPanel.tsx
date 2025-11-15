"use client";

import { useState, useEffect } from "react";
import { reorderPromptByCategory } from "@/utils/tagCategorization";
import Button from "./Button";

export interface TagCategory {
  id: string;
  label: string;
  enabled: boolean;
}

const DEFAULT_CATEGORIES: TagCategory[] = [
  { id: "rating", label: "Rating", enabled: true },
  { id: "quality", label: "Quality", enabled: true },
  { id: "general", label: "General", enabled: true },
  { id: "character", label: "Character", enabled: true },
  { id: "copyright", label: "Copyright", enabled: true },
  { id: "artist", label: "Artist", enabled: true },
  { id: "meta", label: "Meta", enabled: true },
  { id: "model", label: "Model", enabled: true },
];

const STORAGE_KEY = "tag_category_order";

interface CategoryOrderPanelProps {
  currentPrompt: string;
  onApplyOrder?: (reorderedPrompt: string) => void;
}

export default function CategoryOrderPanel({ currentPrompt, onApplyOrder }: CategoryOrderPanelProps) {
  const [categories, setCategories] = useState<TagCategory[]>(DEFAULT_CATEGORIES);
  const [isReordering, setIsReordering] = useState(false);

  // Load saved order from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
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

  const moveUp = (index: number) => {
    if (index === 0) return;
    const newCategories = [...categories];
    [newCategories[index - 1], newCategories[index]] = [
      newCategories[index],
      newCategories[index - 1],
    ];
    saveOrder(newCategories);
  };

  const moveDown = (index: number) => {
    if (index === categories.length - 1) return;
    const newCategories = [...categories];
    [newCategories[index], newCategories[index + 1]] = [
      newCategories[index + 1],
      newCategories[index],
    ];
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

      <div className="bg-gray-800 rounded-lg p-4">
        <div className="space-y-2">
          {categories.map((category, index) => (
            <div
              key={category.id}
              className="flex items-center gap-3 bg-gray-700 p-3 rounded-lg hover:bg-gray-650 transition-colors"
            >
              {/* Enable/Disable Checkbox */}
              <input
                type="checkbox"
                checked={category.enabled}
                onChange={() => toggleCategory(index)}
                className="w-5 h-5 rounded border-gray-500 text-blue-600 focus:ring-blue-500 focus:ring-offset-gray-800"
                title={category.enabled ? "Disable category" : "Enable category"}
              />

              {/* Category Label */}
              <div className="flex-1">
                <span
                  className={`text-sm font-medium ${
                    category.enabled ? "text-gray-200" : "text-gray-500"
                  }`}
                >
                  {category.label}
                </span>
                {!category.enabled && (
                  <span className="ml-2 text-xs text-gray-500">(Disabled)</span>
                )}
              </div>

              {/* Move Buttons */}
              <div className="flex gap-1">
                <button
                  onClick={() => moveUp(index)}
                  disabled={index === 0}
                  className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  title="Move up"
                >
                  ↑
                </button>
                <button
                  onClick={() => moveDown(index)}
                  disabled={index === categories.length - 1}
                  className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  title="Move down"
                >
                  ↓
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 p-3 bg-gray-900 rounded text-xs text-gray-400">
          <p className="mb-1">
            <strong>Usage:</strong>
          </p>
          <ul className="list-disc list-inside space-y-1">
            <li>Use ↑/↓ buttons to reorder categories</li>
            <li>Uncheck categories to exclude them from reordering</li>
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
      return JSON.parse(saved);
    }
  } catch (error) {
    console.error("Failed to load category order:", error);
  }

  return DEFAULT_CATEGORIES;
}
