"use client";

import { useState } from "react";
import { addTagToCategory } from "@/utils/api";
import { X } from "lucide-react";

interface TagStatistic {
  category: string;
  count: number;
}

interface ActionsColumnProps {
  datasetId: number;
  tagStatistics?: Record<string, TagStatistic>;
  onRefresh: () => void;
}

// Category colors (same as ItemDetailColumn)
const getCategoryColor = (category: string): string => {
  const normalized = category.toLowerCase().replace(/\s+/g, '');
  const colors: Record<string, string> = {
    character: "bg-blue-600 dark:bg-blue-700",
    artist: "bg-purple-600 dark:bg-purple-700",
    copyright: "bg-pink-600 dark:bg-pink-700",
    general: "bg-green-600 dark:bg-green-700",
    meta: "bg-gray-600 dark:bg-gray-700",
    quality: "bg-yellow-600 dark:bg-yellow-700",
    qualitytag: "bg-yellow-600 dark:bg-yellow-700",
    rating: "bg-red-600 dark:bg-red-700",
    ratingtag: "bg-red-600 dark:bg-red-700",
    model: "bg-indigo-600 dark:bg-indigo-700",
    unknown: "bg-orange-600 dark:bg-orange-700",
  };
  return colors[normalized] || "bg-orange-600 dark:bg-orange-700";
};

export default function ActionsColumn({
  datasetId,
  tagStatistics,
  onRefresh,
}: ActionsColumnProps) {
  // Category visibility state (all visible by default)
  const [visibleCategories, setVisibleCategories] = useState<Set<string>>(
    new Set(["character", "artist", "copyright", "general", "meta", "quality", "rating", "model", "unknown"])
  );

  // Categorization dialog state
  const [selectedTag, setSelectedTag] = useState<{ tag: string; count: number } | null>(null);
  const [isCategorizing, setIsCategorizing] = useState(false);

  // Get unique categories from tag statistics
  const allCategories = tagStatistics
    ? Array.from(new Set(Object.values(tagStatistics).map(s => s.category)))
    : [];

  // Sort tags by count (most common first) and filter by visible categories
  const sortedTags = tagStatistics
    ? Object.entries(tagStatistics)
        .filter(([_, stats]) => visibleCategories.has(stats.category.toLowerCase()))
        .sort((a, b) => b[1].count - a[1].count)
        .slice(0, 100) // Show top 100
    : [];

  const toggleCategory = (category: string) => {
    const newVisible = new Set(visibleCategories);
    const normalized = category.toLowerCase();
    if (newVisible.has(normalized)) {
      newVisible.delete(normalized);
    } else {
      newVisible.add(normalized);
    }
    setVisibleCategories(newVisible);
  };

  const handleTagClick = (tag: string, stats: TagStatistic) => {
    // Only allow categorization for Unknown tags
    if (stats.category.toLowerCase() === "unknown") {
      setSelectedTag({ tag, count: stats.count });
    }
  };

  const handleCategorize = async (targetCategory: string) => {
    if (!selectedTag) return;

    setIsCategorizing(true);
    try {
      await addTagToCategory(selectedTag.tag, targetCategory, selectedTag.count);
      alert(`Tag "${selectedTag.tag}" added to ${targetCategory} category`);
      setSelectedTag(null);
      onRefresh(); // Refresh dataset to update tag categories
    } catch (error) {
      console.error("Failed to categorize tag:", error);
      alert("Failed to categorize tag. See console for details.");
    } finally {
      setIsCategorizing(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-3 border-b border-gray-700">
        <h3 className="text-sm font-semibold">Tag Statistics</h3>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* Category Filter */}
        {allCategories.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-2">
            <div className="text-[10px] font-semibold text-gray-400 mb-1">Filter by Category</div>
            <div className="flex flex-wrap gap-1">
              {allCategories.map(category => {
                const normalized = category.toLowerCase();
                const isVisible = visibleCategories.has(normalized);
                const colorClass = getCategoryColor(category);
                return (
                  <button
                    key={category}
                    onClick={() => toggleCategory(category)}
                    className={`px-1.5 py-0.5 rounded text-[9px] transition-opacity ${colorClass} ${
                      isVisible ? 'opacity-100' : 'opacity-30'
                    }`}
                    title={isVisible ? `Hide ${category}` : `Show ${category}`}
                  >
                    {category}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Tag List */}
        <div className="bg-gray-800 rounded-lg p-3">
          {sortedTags.length > 0 ? (
            <div className="space-y-1 max-h-96 overflow-y-auto">
              {sortedTags.map(([tag, stats]) => {
                const colorClass = getCategoryColor(stats.category);
                const isUnknown = stats.category.toLowerCase() === "unknown";
                return (
                  <div
                    key={tag}
                    className={`flex items-center justify-between text-xs group hover:bg-gray-700 rounded px-1.5 py-0.5 transition-colors ${
                      isUnknown ? 'cursor-pointer' : ''
                    }`}
                    onClick={() => handleTagClick(tag, stats)}
                    title={isUnknown ? "Click to categorize" : ""}
                  >
                    <div className="flex items-center space-x-1.5 flex-1 min-w-0">
                      <span className={`px-1.5 py-0.5 ${colorClass} rounded text-[10px] flex-shrink-0`}>
                        {stats.category}
                      </span>
                      <span className="text-gray-200 truncate">{tag}</span>
                    </div>
                    <span className="text-gray-400 text-[10px] font-mono ml-2 flex-shrink-0">
                      {stats.count}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-[10px] text-gray-500 text-center py-4">
              No tag statistics available. Scan dataset to generate.
            </div>
          )}
        </div>
      </div>

      {/* Categorization Dialog */}
      {selectedTag && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg border border-gray-700 w-full max-w-sm p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold">Categorize Tag</h3>
              <button
                onClick={() => setSelectedTag(null)}
                className="p-1 hover:bg-gray-700 rounded transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mb-3">
              <div className="text-xs text-gray-300 mb-1">Tag:</div>
              <div className="bg-gray-900 rounded px-2 py-1 text-xs text-gray-200">
                {selectedTag.tag}
              </div>
            </div>

            <div className="mb-3">
              <div className="text-xs text-gray-300 mb-2">Select Category:</div>
              <div className="grid grid-cols-2 gap-2">
                {["Artist", "Character", "Copyright", "General", "Meta", "Model", "Quality", "Rating"].map(category => (
                  <button
                    key={category}
                    onClick={() => handleCategorize(category)}
                    disabled={isCategorizing}
                    className={`px-3 py-2 rounded text-xs transition-colors ${getCategoryColor(category)} hover:opacity-80 disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {category}
                  </button>
                ))}
              </div>
            </div>

            <div className="text-[10px] text-gray-500">
              Count: {selectedTag.count} (will be saved to taglist JSON)
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
