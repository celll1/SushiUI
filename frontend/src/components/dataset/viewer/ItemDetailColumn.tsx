"use client";

import { useState, useEffect, useRef } from "react";
import { Undo2, Redo2, Copy, Clipboard } from "lucide-react";
import {
  getDatasetItem,
  DatasetItem,
  updateItemCaption,
} from "@/utils/api";
import TagSuggestions from "@/components/common/TagSuggestions";
import { TagFilterMode } from "@/utils/tagSuggestions";
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

interface TagSuggestion {
  tag: string;
  count: number;
  category: string;
}

// Category colors mapping (for tag chips)
const getCategoryColor = (category: string): string => {
  const colors: Record<string, string> = {
    character: "bg-blue-600 dark:bg-blue-700 hover:bg-blue-500",
    artist: "bg-purple-600 dark:bg-purple-700 hover:bg-purple-500",
    copyright: "bg-pink-600 dark:bg-pink-700 hover:bg-pink-500",
    general: "bg-green-600 dark:bg-green-700 hover:bg-green-500",
    meta: "bg-gray-600 dark:bg-gray-700 hover:bg-gray-500",
    quality: "bg-yellow-600 dark:bg-yellow-700 hover:bg-yellow-500",
    rating: "bg-red-600 dark:bg-red-700 hover:bg-red-500",
    model: "bg-indigo-600 dark:bg-indigo-700 hover:bg-indigo-500",
  };
  return colors[category.toLowerCase()] || "bg-green-600 dark:bg-green-700 hover:bg-green-500";
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
  const [tagSuggestions, setTagSuggestions] = useState<TagSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(0);
  const [filterMode, setFilterMode] = useState<TagFilterMode>("all");
  const [suggestionPosition, setSuggestionPosition] = useState({ top: 0, left: 0 });
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (item) {
      loadItemDetails();
    }
  }, [item]);

  const loadItemDetails = async () => {
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
            // Search each tag to get its category from pre-loaded JSON files
            for (const tag of tagList) {
              const results = await tagSuggestionsContext.searchTags(tag, 1, 'all');
              if (results.length > 0 && results[0].tag === tag) {
                categoryMap[tag] = results[0].category.toLowerCase();
              }
            }
            setTagCategories(categoryMap);
            console.log("[ItemDetail] Loaded tag categories:", categoryMap);
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
  };

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

  const handleAddTag = (tag?: string) => {
    const tagToAdd = tag || newTag.trim().toLowerCase().replace(/\s+/g, "_");
    if (!tagToAdd) return;

    if (tags.includes(tagToAdd)) {
      setNewTag("");
      setShowSuggestions(false);
      return;
    }

    pushHistory([...tags, tagToAdd]);
    setNewTag("");
    setShowSuggestions(false);

    // If tag was selected from suggestions, category is already in tagCategories
    // If manually typed, try to look it up
    if (tag && !tagCategories[tag] && tagSuggestionsContext.isLoaded) {
      tagSuggestionsContext.searchTags(tag, 1, 'all').then(results => {
        if (results.length > 0 && results[0].tag === tag) {
          setTagCategories(prev => ({
            ...prev,
            [tag]: results[0].category.toLowerCase()
          }));
        }
      }).catch(err => console.error("Failed to fetch tag category:", err));
    }
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

  // Tag suggestions logic using tagSuggestions.ts (same as Generate screen)
  useEffect(() => {
    const handleSearch = async () => {
      if (newTag.trim().length < 2) {
        setTagSuggestions([]);
        setShowSuggestions(false);
        return;
      }

      console.log("[Autocomplete] Searching for:", newTag.trim());

      try {
        // Use tagSuggestions Context (pre-loaded JSON files)
        const results = await tagSuggestionsContext.searchTags(newTag.trim(), 20, filterMode);
        console.log("[Autocomplete] Response:", results);

        // Results already match TagSuggestion format
        const suggestions: TagSuggestion[] = results.map(r => ({
          tag: r.tag,
          count: r.count,
          category: r.category,
        }));

        setTagSuggestions(suggestions);
        setShowSuggestions(suggestions.length > 0);
        setSelectedSuggestionIndex(0);

        // Update category map with discovered categories
        if (suggestions.length > 0) {
          const newCategories: Record<string, string> = { ...tagCategories };
          suggestions.forEach(s => {
            if (!newCategories[s.tag]) {
              newCategories[s.tag] = s.category.toLowerCase();
            }
          });
          setTagCategories(newCategories);
        }

        // Update position
        if (inputRef.current) {
          const rect = inputRef.current.getBoundingClientRect();
          setSuggestionPosition({
            top: rect.bottom + window.scrollY + 4,
            left: rect.left + window.scrollX,
          });
        }
      } catch (err) {
        console.error("[Autocomplete] Failed to search tags:", err);
      }
    };

    const debounceTimer = setTimeout(handleSearch, 300);
    return () => clearTimeout(debounceTimer);
  }, [newTag, filterMode]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showSuggestions || tagSuggestions.length === 0) {
      if (e.key === "Enter") {
        e.preventDefault();
        handleAddTag();
      }
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedSuggestionIndex((prev) =>
        Math.min(prev + 1, tagSuggestions.length - 1)
      );
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedSuggestionIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      handleAddTag(tagSuggestions[selectedSuggestionIndex].tag);
    } else if (e.key === "Escape") {
      setShowSuggestions(false);
    }
  };

  const handleFilterChange = (direction: 'next' | 'prev') => {
    const newMode = direction === 'next'
      ? tagSuggestionsContext.getNextFilterMode(filterMode)
      : tagSuggestionsContext.getPreviousFilterMode(filterMode);
    setFilterMode(newMode);
    console.log("[Filter] Changed to:", newMode);
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
                    className={`flex items-center space-x-1 px-1.5 py-0.5 ${colorClass} rounded text-[10px] transition-colors group h-fit cursor-pointer`}
                    title={`Category: ${category}`}
                    onClick={() => handleRemoveTag(index)}
                  >
                    <span>{tag}</span>
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity text-[8px]">
                      ✕
                    </span>
                  </div>
                );
              })
            )}
          </div>

          {/* Add Tag - Compact with Autocomplete */}
          <div className="flex-shrink-0 mt-2 relative">
            <input
              ref={inputRef}
              type="text"
              value={newTag}
              onChange={(e) => setNewTag(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type to search tags..."
              className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs focus:outline-none focus:border-blue-500"
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

      {/* Tag Suggestions Dropdown */}
      {showSuggestions && tagSuggestions.length > 0 && (
        <TagSuggestions
          suggestions={tagSuggestions}
          selectedIndex={selectedSuggestionIndex}
          onSelect={handleAddTag}
          position={suggestionPosition}
          filterMode={filterMode}
          onFilterChange={handleFilterChange}
        />
      )}
    </div>
  );
}
