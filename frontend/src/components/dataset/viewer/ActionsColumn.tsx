"use client";

import { CheckSquare, Square, Tag, Trash2, Download } from "lucide-react";

interface ActionsColumnProps {
  datasetId: number;
  selectedItems: Set<number>;
  totalItems: number;
  onSelectAll: () => void;
  onDeselectAll: () => void;
  onRefresh: () => void;
}

export default function ActionsColumn({
  datasetId,
  selectedItems,
  totalItems,
  onSelectAll,
  onDeselectAll,
  onRefresh,
}: ActionsColumnProps) {
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
