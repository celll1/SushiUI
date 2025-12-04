"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Search, ChevronLeft, ChevronRight, X } from "lucide-react";
import { DatasetItem } from "@/utils/api";
import TagSuggestions from "@/components/common/TagSuggestions";
import { TagFilterMode, normalizeTagForMatching } from "@/utils/tagSuggestions";
import { useTagSuggestions } from "@/contexts/TagSuggestionsContext";

interface ItemGridColumnProps {
  items: DatasetItem[];
  selectedItems: Set<number>;
  currentItem: DatasetItem | null;
  search: string;
  tagFilter: string;
  page: number;
  total: number;
  pageSize: number;
  loading: boolean;
  onSelectItem: (item: DatasetItem) => void;
  onToggleSelection: (itemId: number) => void;
  onSearchChange: (search: string) => void;
  onTagFilterChange: (tagFilter: string) => void;
  onPageChange: (page: number) => void;
}

interface TagSuggestion {
  tag: string;
  count: number;
  category: string;
}

// Category colors (same as ItemDetailColumn)
const getCategoryColor = (category: string): string => {
  const normalized = category.toLowerCase().replace(/\s+/g, '');
  const colors: Record<string, string> = {
    character: "bg-blue-600 dark:bg-blue-700 hover:bg-blue-500",
    artist: "bg-purple-600 dark:bg-purple-700 hover:bg-purple-500",
    copyright: "bg-pink-600 dark:bg-pink-700 hover:bg-pink-500",
    general: "bg-green-600 dark:bg-green-700 hover:bg-green-500",
    meta: "bg-gray-600 dark:bg-gray-700 hover:bg-gray-500",
    quality: "bg-yellow-600 dark:bg-yellow-700 hover:bg-yellow-500",
    qualitytag: "bg-yellow-600 dark:bg-yellow-700 hover:bg-yellow-500",
    rating: "bg-red-600 dark:bg-red-700 hover:bg-red-500",
    ratingtag: "bg-red-600 dark:bg-red-700 hover:bg-red-500",
    model: "bg-indigo-600 dark:bg-indigo-700 hover:bg-indigo-500",
  };
  return colors[normalized] || "bg-green-600 dark:bg-green-700 hover:bg-green-500";
};

export default function ItemGridColumn({
  items,
  selectedItems,
  currentItem,
  search,
  tagFilter,
  page,
  total,
  pageSize,
  loading,
  onSelectItem,
  onToggleSelection,
  onSearchChange,
  onTagFilterChange,
  onPageChange,
}: ItemGridColumnProps) {
  const totalPages = Math.ceil(total / pageSize);
  const tagSuggestionsContext = useTagSuggestions();

  // Tag filter internal state
  const [tagFilterTags, setTagFilterTags] = useState<string[]>([]); // Array of tags
  const [tagCategories, setTagCategories] = useState<Record<string, string>>({});
  const [newFilterTag, setNewFilterTag] = useState(""); // Input for new tag
  const [tagSuggestions, setTagSuggestions] = useState<TagSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(0);
  const [filterMode, setFilterMode] = useState<TagFilterMode>("all");
  const [suggestionPosition, setSuggestionPosition] = useState({ top: 0, left: 0 });
  const [suppressSuggestions, setSuppressSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Sync tagFilterTags array with parent's comma-separated string
  useEffect(() => {
    if (tagFilter) {
      const tags = tagFilter.split(",").map(t => t.trim()).filter(Boolean);
      setTagFilterTags(tags);
    } else {
      setTagFilterTags([]);
    }
  }, [tagFilter]);

  // Update parent when internal array changes
  const updateParentFilter = useCallback((tags: string[]) => {
    onTagFilterChange(tags.join(","));
  }, [onTagFilterChange]);

  // Tag suggestions search
  useEffect(() => {
    const handleSearch = async () => {
      if (newFilterTag.trim().length < 2) {
        setTagSuggestions([]);
        setShowSuggestions(false);
        return;
      }

      if (suppressSuggestions) {
        return;
      }

      try {
        const results = await tagSuggestionsContext.searchTags(newFilterTag.trim(), 20, filterMode);
        const suggestions: TagSuggestion[] = results.map(r => ({
          tag: r.tag,
          count: r.count,
          category: r.category,
        }));

        setTagSuggestions(suggestions);
        setShowSuggestions(suggestions.length > 0);
        setSelectedSuggestionIndex(0);

        // Update category map
        if (suggestions.length > 0) {
          const newCategories: Record<string, string> = { ...tagCategories };
          suggestions.forEach(s => {
            if (!newCategories[s.tag]) {
              newCategories[s.tag] = s.category;
            }
          });
          setTagCategories(newCategories);
        }

        // Update position - show above input
        if (inputRef.current) {
          const rect = inputRef.current.getBoundingClientRect();
          const suggestionsHeight = 256;
          setSuggestionPosition({
            top: rect.top + window.scrollY - suggestionsHeight - 8,
            left: rect.left + window.scrollX,
          });
        }
      } catch (err) {
        console.error("[TagFilter] Failed to search tags:", err);
      }
    };

    const debounceTimer = setTimeout(handleSearch, 300);
    return () => clearTimeout(debounceTimer);
  }, [newFilterTag, filterMode, tagSuggestionsContext, tagCategories, suppressSuggestions]);

  const handleAddFilterTag = (tagToAdd?: string) => {
    const tag = tagToAdd || newFilterTag.trim();
    if (!tag) return;

    // Don't add duplicates
    if (tagFilterTags.includes(tag)) {
      setNewFilterTag("");
      setShowSuggestions(false);
      return;
    }

    const newTags = [...tagFilterTags, tag];
    setTagFilterTags(newTags);
    updateParentFilter(newTags);
    setNewFilterTag("");
    setShowSuggestions(false);
    setSuppressSuggestions(false);
  };

  const handleRemoveFilterTag = (index: number) => {
    const newTags = tagFilterTags.filter((_, i) => i !== index);
    setTagFilterTags(newTags);
    updateParentFilter(newTags);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showSuggestions || tagSuggestions.length === 0) {
      if (e.key === "Enter") {
        e.preventDefault();
        handleAddFilterTag();
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
      handleAddFilterTag(tagSuggestions[selectedSuggestionIndex].tag);
    } else if (e.key === "Escape") {
      setShowSuggestions(false);
      setSuppressSuggestions(true);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNewFilterTag(e.target.value);
    setSuppressSuggestions(false);
  };

  const handleFilterChange = (direction: 'next' | 'prev') => {
    const newMode = direction === 'next'
      ? tagSuggestionsContext.getNextFilterMode(filterMode)
      : tagSuggestionsContext.getPreviousFilterMode(filterMode);
    setFilterMode(newMode);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-3 border-b border-gray-700">
        <h3 className="text-sm font-semibold mb-3">Items ({total})</h3>

        {/* Filename Search */}
        <div className="relative mb-2">
          <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search filename..."
            className="w-full pl-8 pr-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-xs focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Tag Filter - Cards Display */}
        {tagFilterTags.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {tagFilterTags.map((tag, index) => {
              const category = tagCategories[tag] || "general";
              const colorClass = getCategoryColor(category);
              return (
                <div
                  key={index}
                  className={`flex items-center space-x-1 px-2 py-1 ${colorClass} rounded text-xs transition-colors group h-fit cursor-pointer`}
                  title={`Category: ${category}`}
                  onClick={() => handleRemoveFilterTag(index)}
                >
                  <span>{tag}</span>
                  <X className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              );
            })}
          </div>
        )}

        {/* Tag Filter Input with Autocomplete */}
        <div className="relative">
          <input
            ref={inputRef}
            type="text"
            value={newFilterTag}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Add tag filter..."
            className="w-full px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-xs focus:outline-none focus:border-blue-500"
          />
        </div>
      </div>

      {/* Tag Suggestions Dropdown */}
      {showSuggestions && tagSuggestions.length > 0 && (
        <TagSuggestions
          suggestions={tagSuggestions}
          selectedIndex={selectedSuggestionIndex}
          onSelect={handleAddFilterTag}
          position={suggestionPosition}
          filterMode={filterMode}
          onFilterChange={handleFilterChange}
        />
      )}

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-2">
        {loading && items.length === 0 ? (
          <div className="text-center text-gray-400 text-xs py-8">Loading...</div>
        ) : items.length === 0 ? (
          <div className="text-center text-gray-400 text-xs py-8">
            {search || tagFilter ? "No items found" : "No items"}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            {items.map((item) => (
              <div
                key={item.id}
                onClick={() => onSelectItem(item)}
                className={`relative bg-gray-800 rounded border cursor-pointer transition-all group ${
                  currentItem?.id === item.id
                    ? "border-blue-500 ring-1 ring-blue-500"
                    : "border-gray-700 hover:border-gray-600"
                }`}
              >
                {/* Checkbox */}
                <div
                  className="absolute top-1 left-1 z-10"
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleSelection(item.id);
                  }}
                >
                  <input
                    type="checkbox"
                    checked={selectedItems.has(item.id)}
                    onChange={() => {}}
                    className="w-4 h-4 cursor-pointer"
                  />
                </div>

                {/* Image */}
                <div className="aspect-square bg-gray-900">
                  <img
                    src={`/api/serve-image?path=${encodeURIComponent(item.image_path)}`}
                    alt={item.base_name}
                    className="w-full h-full object-cover rounded-t"
                    loading="lazy"
                  />
                </div>

                {/* Info */}
                <div className="p-1.5">
                  <p className="text-xs text-gray-300 truncate" title={item.base_name}>
                    {item.base_name}
                  </p>
                  <div className="flex items-center justify-between text-[10px] text-gray-500 mt-0.5">
                    <span>{item.width}×{item.height}</span>
                    <span>{(item.file_size / 1024).toFixed(0)}KB</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="p-2 border-t border-gray-700">
          <div className="flex items-center justify-between text-xs">
            <button
              onClick={() => onPageChange(Math.max(1, page - 1))}
              disabled={page === 1}
              className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>

            <span className="text-gray-400">
              Page {page} / {totalPages}
            </span>

            <button
              onClick={() => onPageChange(Math.min(totalPages, page + 1))}
              disabled={page === totalPages}
              className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
