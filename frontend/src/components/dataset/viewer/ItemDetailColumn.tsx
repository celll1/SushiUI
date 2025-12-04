"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Undo2, Redo2, Copy, Clipboard } from "lucide-react";
import {
  getDatasetItem,
  DatasetItem,
  updateItemCaption,
} from "@/utils/api";
import InputWithTagSuggestions from "@/components/common/InputWithTagSuggestions";
import { normalizeTagForMatching } from "@/utils/tagSuggestions";
import { useTagSuggestions } from "@/contexts/TagSuggestionsContext";

interface ItemDetailColumnProps {
  item: DatasetItem | null;
  datasetId: number;
}

interface EditHistory {
  past: string[][];
  present: string[];
  future: string[][];
}


// Category colors mapping (for tag chips)
const getCategoryColor = (category: string): string => {
  const normalized = category.toLowerCase().replace(/\s+/g, '');
  const colors: Record<string, string> = {
    character: "bg-blue-600 dark:bg-blue-700 hover:bg-blue-500",
    artist: "bg-purple-600 dark:bg-purple-700 hover:bg-purple-500",
    copyright: "bg-pink-600 dark:bg-pink-700 hover:bg-pink-500",
    general: "bg-green-600 dark:bg-green-700 hover:bg-green-500",
    meta: "bg-gray-600 dark:bg-gray-700 hover:bg-gray-500",
    quality: "bg-yellow-600 dark:bg-yellow-700 hover:bg-yellow-500",
    qualitytag: "bg-yellow-600 dark:bg-yellow-700 hover:bg-yellow-500", // "Quality Tag"
    rating: "bg-red-600 dark:bg-red-700 hover:bg-red-500",
    ratingtag: "bg-red-600 dark:bg-red-700 hover:bg-red-500", // "Rating Tag"
    model: "bg-indigo-600 dark:bg-indigo-700 hover:bg-indigo-500",
  };
  return colors[normalized] || "bg-green-600 dark:bg-green-700 hover:bg-green-500";
};

export default function ItemDetailColumn({ item, datasetId }: ItemDetailColumnProps) {
  const tagSuggestionsContext = useTagSuggestions();
  const [detailedItem, setDetailedItem] = useState<DatasetItem | null>(null);
  const [tags, setTags] = useState<string[]>([]);
  const [tagCategories, setTagCategories] = useState<Record<string, string>>({});
  const [newTag, setNewTag] = useState("");
  const [history, setHistory] = useState<EditHistory>({
    past: [],
    present: [],
    future: [],
  });
  const [hasChanges, setHasChanges] = useState(false);

  const loadItemDetails = useCallback(async () => {
    if (!item) return;

    try {
      const details = await getDatasetItem(datasetId, item.id);
      setDetailedItem(details);

      // Extract tags from captions
      const tagCaption = details.captions?.find(c => c.caption_type === "tags");
      if (tagCaption) {
        const tagList = tagCaption.content.split(",").map(t => t.trim()).filter(Boolean);
        setTags(tagList);
        setHistory({
          past: [],
          present: tagList,
          future: [],
        });
        setHasChanges(false);

        // Fetch category information for existing tags using Context (pre-loaded tags)
        if (tagList.length > 0 && tagSuggestionsContext.isLoaded) {
          try {
            const categoryMap: Record<string, string> = {};
            const notFoundTags: string[] = [];

            // Search each tag to get its category from pre-loaded JSON files
            // Use normalized matching to handle various tag formats
            for (const tag of tagList) {
              const results = await tagSuggestionsContext.searchTags(tag, 1, 'all');
              if (results.length > 0) {
                // Compare normalized forms to handle format differences
                const normalizedUserTag = normalizeTagForMatching(tag);
                const normalizedResultTag = normalizeTagForMatching(results[0].tag);
                if (normalizedUserTag === normalizedResultTag) {
                  // Store category as-is (will be normalized in getCategoryColor)
                  categoryMap[tag] = results[0].category;
                } else {
                  console.warn(`[ItemDetail] Tag normalization mismatch: "${tag}" -> "${normalizedUserTag}" vs result "${normalizedResultTag}"`);
                  notFoundTags.push(tag);
                }
              } else {
                console.warn(`[ItemDetail] Tag not found in suggestions: "${tag}"`);
                notFoundTags.push(tag);
              }
            }

            // Merge with existing categories instead of replacing
            setTagCategories(prev => ({ ...prev, ...categoryMap }));
            console.log("[ItemDetail] Loaded tag categories:", categoryMap);
            if (notFoundTags.length > 0) {
              console.warn("[ItemDetail] Tags without category:", notFoundTags);
            }
          } catch (err) {
            console.error("Failed to load tag categories:", err);
          }
        }
      } else {
        setTags([]);
        setHistory({ past: [], present: [], future: [] });
      }
    } catch (err) {
      console.error("Failed to load item details:", err);
    }
  }, [item, datasetId, tagSuggestionsContext]);

  useEffect(() => {
    if (item) {
      loadItemDetails();
    }
  }, [item, loadItemDetails]);

  const pushHistory = (newTags: string[]) => {
    setHistory({
      past: [...history.past, history.present],
      present: newTags,
      future: [],
    });
    setTags(newTags);
    setHasChanges(true);
  };

  const handleUndo = () => {
    if (history.past.length === 0) return;

    const previous = history.past[history.past.length - 1];
    const newPast = history.past.slice(0, -1);

    setHistory({
      past: newPast,
      present: previous,
      future: [history.present, ...history.future],
    });
    setTags(previous);
    setHasChanges(newPast.length > 0 || history.future.length > 0);
  };

  const handleRedo = () => {
    if (history.future.length === 0) return;

    const next = history.future[0];
    const newFuture = history.future.slice(1);

    setHistory({
      past: [...history.past, history.present],
      present: next,
      future: newFuture,
    });
    setTags(next);
    setHasChanges(true);
  };


  const handleRemoveTag = (index: number) => {
    const newTags = tags.filter((_, i) => i !== index);
    pushHistory(newTags);
  };

  const handleCopyTags = () => {
    navigator.clipboard.writeText(tags.join(", "));
  };

  const handlePasteTags = async () => {
    try {
      const text = await navigator.clipboard.readText();
      const pastedTags = text.split(",").map(t => t.trim()).filter(Boolean);
      pushHistory([...tags, ...pastedTags]);
    } catch (err) {
      console.error("Failed to paste tags:", err);
    }
  };

  const handleSave = async () => {
    if (!item) return;

    try {
      const content = tags.join(", ");
      await updateItemCaption(item.id, {
        caption_type: "tags",
        content,
      });
      setHasChanges(false);
      console.log("Tags saved successfully");
    } catch (err) {
      console.error("Failed to save tags:", err);
      alert("Failed to save tags. Please try again.");
    }
  };

  const handleRevert = () => {
    loadItemDetails();
  };

  const handleTagAdd = (tag: string, category: string) => {
    if (tags.includes(tag)) {
      return; // Don't add duplicates
    }

    pushHistory([...tags, tag]);

    // Store category for this tag
    setTagCategories(prev => ({
      ...prev,
      [tag]: category
    }));
  };

  if (!item) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        Select an item to view details
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header - Compact */}
      <div className="flex-shrink-0 p-2 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold">Item Details</h3>
          <div className="text-[10px] text-gray-400">
            {item.width}×{item.height} • {(item.file_size / 1024).toFixed(1)}KB
          </div>
        </div>
      </div>

      {/* Content - Optimized Layout */}
      <div className="flex-1 flex flex-col min-h-0 overflow-y-auto p-2 space-y-2">
        {/* Image + File Info - Horizontal Layout */}
        <div className="flex-shrink-0 flex gap-2">
          {/* Image Preview - Small Thumbnail */}
          <div className="w-32 h-32 bg-gray-800 rounded overflow-hidden flex-shrink-0">
            <img
              src={`/api/serve-image?path=${encodeURIComponent(item.image_path)}`}
              alt={item.base_name}
              className="w-full h-full object-contain bg-gray-900"
            />
          </div>

          {/* File Info */}
          <div className="flex-1 bg-gray-800 rounded p-2 min-w-0">
            <div className="text-xs font-medium text-gray-200 truncate mb-1" title={item.base_name}>
              {item.base_name}
            </div>
            <div className="text-[10px] text-gray-400 truncate" title={item.image_path}>
              {item.image_path}
            </div>
          </div>
        </div>

        {/* Tags Section - Compact */}
        <div className="flex-1 bg-gray-800 rounded-lg p-2 flex flex-col min-h-0">
          <div className="flex-shrink-0 flex items-center justify-between mb-2">
            <h4 className="text-xs font-semibold">Tags ({tags.length})</h4>
            <div className="flex items-center space-x-0.5">
              <button
                onClick={handleUndo}
                disabled={history.past.length === 0}
                className="p-0.5 hover:bg-gray-700 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                title="Undo (Ctrl+Z)"
              >
                <Undo2 className="h-3 w-3" />
              </button>
              <button
                onClick={handleRedo}
                disabled={history.future.length === 0}
                className="p-0.5 hover:bg-gray-700 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                title="Redo (Ctrl+Y)"
              >
                <Redo2 className="h-3 w-3" />
              </button>
              <button
                onClick={handleCopyTags}
                className="p-0.5 hover:bg-gray-700 rounded transition-colors"
                title="Copy Tags (Ctrl+C)"
              >
                <Copy className="h-3 w-3" />
              </button>
              <button
                onClick={handlePasteTags}
                className="p-0.5 hover:bg-gray-700 rounded transition-colors"
                title="Paste Tags (Ctrl+V)"
              >
                <Clipboard className="h-3 w-3" />
              </button>
            </div>
          </div>

          {/* Tag List - Scrollable */}
          <div className="flex-1 flex flex-wrap gap-1 content-start bg-gray-900 rounded p-2 overflow-y-auto min-h-0">
            {tags.length === 0 ? (
              <div className="text-xs text-gray-500 w-full text-center py-2">No tags</div>
            ) : (
              tags.map((tag, index) => {
                const category = tagCategories[tag] || "general";
                const colorClass = getCategoryColor(category);
                return (
                  <div
                    key={index}
                    className={`flex items-center space-x-1 px-2 py-1 ${colorClass} rounded text-xs transition-colors group h-fit cursor-pointer`}
                    title={`Category: ${category}`}
                    onClick={() => handleRemoveTag(index)}
                  >
                    <span>{tag}</span>
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity text-[10px]">
                      ✕
                    </span>
                  </div>
                );
              })
            )}
          </div>

          {/* Add Tag - with Autocomplete */}
          <div className="flex-shrink-0 mt-2">
            <InputWithTagSuggestions
              value={newTag}
              onChange={setNewTag}
              onTagAdd={handleTagAdd}
              placeholder="Type to search tags..."
              className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs focus:outline-none focus:border-blue-500"
              showSuggestionsAbove={true}
            />
          </div>
        </div>

        {/* Other Caption Types - Collapsible */}
        {detailedItem?.captions?.filter(c => c.caption_type !== "tags").length > 0 && (
          <div className="flex-shrink-0 bg-gray-800 rounded-lg p-2">
            <details className="group">
              <summary className="text-xs font-semibold cursor-pointer list-none flex items-center justify-between">
                <span>Other Captions ({detailedItem?.captions?.filter(c => c.caption_type !== "tags").length})</span>
                <span className="group-open:rotate-180 transition-transform">▼</span>
              </summary>
              <div className="mt-2 space-y-2">
                {detailedItem?.captions?.filter(c => c.caption_type !== "tags").map(caption => (
                  <div key={caption.id} className="bg-gray-900 rounded p-2">
                    <h5 className="text-[10px] font-semibold text-gray-400 mb-1 capitalize">
                      {caption.caption_type.replace(/_/g, " ")} ({caption.source})
                    </h5>
                    <p className="text-xs text-gray-300">{caption.content}</p>
                  </div>
                ))}
              </div>
            </details>
          </div>
        )}
      </div>

      {/* Footer Actions - Sticky */}
      {hasChanges && (
        <div className="flex-shrink-0 p-2 border-t border-gray-700 flex space-x-2">
          <button
            onClick={handleSave}
            className="flex-1 px-3 py-1.5 bg-green-600 hover:bg-green-500 rounded text-xs transition-colors"
          >
            Save
          </button>
          <button
            onClick={handleRevert}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
          >
            Revert
          </button>
        </div>
      )}
    </div>
  );
}
