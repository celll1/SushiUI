"use client";

import { useState } from "react";
import { CheckSquare, Square, Tag, Trash2, Download, Save, RotateCcw } from "lucide-react";
import { saveAllCaptionsToTxt } from "@/utils/api";

interface TagStatistic {
  category: string;
  count: number;
}

interface ActionsColumnProps {
  datasetId: number;
  selectedItems: Set<number>;
  totalItems: number;
  tagStatistics?: Record<string, TagStatistic>;
  onSelectAll: () => void;
  onDeselectAll: () => void;
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
  selectedItems,
  totalItems,
  tagStatistics,
  onSelectAll,
  onDeselectAll,
  onRefresh,
}: ActionsColumnProps) {
  const [isSavingToTxt, setIsSavingToTxt] = useState(false);

  // Sort tags by count (most common first)
  const sortedTags = tagStatistics
    ? Object.entries(tagStatistics)
        .sort((a, b) => b[1].count - a[1].count)
        .slice(0, 50) // Show top 50
    : [];
  const handleBatchTag = () => {
    console.log("Batch tagging", selectedItems.size, "items");
    // TODO: Implement batch tagger
  };

  const handleClearTags = () => {
    if (confirm(`Clear tags from ${selectedItems.size} selected items?`)) {
      console.log("Clearing tags from", selectedItems.size, "items");
      // TODO: Implement clear tags
    }
  };

  const handleExport = () => {
    console.log("Exporting dataset");
    // TODO: Implement export
  };

  const handleSaveAllToTxt = async () => {
    if (!confirm("Save all captions to TXT files? This will overwrite existing TXT files.")) {
      return;
    }

    setIsSavingToTxt(true);
    try {
      const result = await saveAllCaptionsToTxt(datasetId);
      console.log("[ActionsColumn] Bulk save result:", result);
      alert(`Saved ${result.saved} captions to TXT files\n` +
            `Skipped: ${result.skipped} (no caption or no image file)\n` +
            `Errors: ${result.errors}`);
      onRefresh(); // Refresh dataset view
    } catch (err) {
      console.error("[ActionsColumn] Failed to save all to TXT:", err);
      alert("Failed to save captions to TXT files. Please try again.");
    } finally {
      setIsSavingToTxt(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-3 border-b border-gray-700">
        <h3 className="text-sm font-semibold">Actions</h3>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Selection Info */}
        <div className="bg-gray-800 rounded-lg p-3">
          <h4 className="text-xs font-semibold mb-2">Selection</h4>
          <div className="text-xs text-gray-300 mb-3">
            <div>{selectedItems.size} selected</div>
            <div className="text-gray-500">{totalItems} total items</div>
          </div>
          <div className="space-y-1.5">
            <button
              onClick={onSelectAll}
              className="w-full flex items-center justify-center space-x-1.5 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
            >
              <CheckSquare className="h-3.5 w-3.5" />
              <span>Select All</span>
            </button>
            <button
              onClick={onDeselectAll}
              disabled={selectedItems.size === 0}
              className="w-full flex items-center justify-center space-x-1.5 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Square className="h-3.5 w-3.5" />
              <span>Deselect All</span>
            </button>
          </div>
        </div>

        {/* Batch Operations */}
        <div className="bg-gray-800 rounded-lg p-3">
          <h4 className="text-xs font-semibold mb-2">Batch Operations</h4>
          <div className="text-[10px] text-gray-400 mb-2">
            Apply to: {selectedItems.size > 0 ? `${selectedItems.size} selected` : "all items"}
          </div>
          <div className="space-y-1.5">
            <button
              onClick={handleBatchTag}
              disabled={selectedItems.size === 0}
              className="w-full flex items-center justify-center space-x-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Tag className="h-3.5 w-3.5" />
              <span>Tag Selected</span>
            </button>
            <button
              onClick={handleClearTags}
              disabled={selectedItems.size === 0}
              className="w-full flex items-center justify-center space-x-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-500 rounded text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Clear Tags</span>
            </button>
            <button
              onClick={handleExport}
              className="w-full flex items-center justify-center space-x-1.5 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Export Dataset</span>
            </button>
          </div>
        </div>

        {/* TXT File Synchronization */}
        <div className="bg-gray-800 rounded-lg p-3">
          <h4 className="text-xs font-semibold mb-2">TXT File Sync</h4>
          <div className="space-y-1.5">
            <button
              onClick={handleSaveAllToTxt}
              disabled={isSavingToTxt}
              className="w-full flex items-center justify-center space-x-1.5 px-3 py-1.5 bg-green-600 hover:bg-green-500 rounded text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Save all DB captions to TXT files"
            >
              <Save className="h-3.5 w-3.5" />
              <span>{isSavingToTxt ? "Saving..." : "Save All to TXT"}</span>
            </button>
          </div>
        </div>

        {/* Tag Statistics */}
        <div className="bg-gray-800 rounded-lg p-3">
          <h4 className="text-xs font-semibold mb-2">Tag Statistics</h4>
          {sortedTags.length > 0 ? (
            <div className="space-y-1 max-h-96 overflow-y-auto">
              {sortedTags.map(([tag, stats]) => {
                const colorClass = getCategoryColor(stats.category);
                return (
                  <div
                    key={tag}
                    className="flex items-center justify-between text-xs group hover:bg-gray-700 rounded px-1.5 py-0.5 transition-colors"
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

        {/* Auto-Tagger (Placeholder) */}
        <div className="bg-gray-800 rounded-lg p-3">
          <h4 className="text-xs font-semibold mb-2">Auto-Tagger</h4>
          <div className="text-[10px] text-gray-500 text-center py-4">
            Tagger integration coming soon
          </div>
        </div>
      </div>
    </div>
  );
}
