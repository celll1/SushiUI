"use client";

import { useState, useRef, useEffect, KeyboardEvent, ChangeEvent } from "react";
import TagSuggestions from "./TagSuggestions";
import { TagFilterMode } from "@/utils/tagSuggestions";
import { useTagSuggestions } from "@/contexts/TagSuggestionsContext";

interface TagSuggestion {
  tag: string;
  count: number;
  category: string;
}

interface InputWithTagSuggestionsProps {
  value: string;
  onChange: (value: string) => void;
  onTagAdd: (tag: string, category: string) => void;
  placeholder?: string;
  className?: string;
  showSuggestionsAbove?: boolean; // Show suggestions above input instead of below
}

/**
 * InputWithTagSuggestions - Input field with tag autocompletion
 *
 * Features:
 * - Tag suggestions while typing (min 2 characters)
 * - Arrow keys to navigate suggestions
 * - Enter to accept selected suggestion
 * - Esc to close suggestions (won't reshow until next input)
 * - Automatic category detection from tag suggestions
 *
 * Usage:
 * <InputWithTagSuggestions
 *   value={inputValue}
 *   onChange={setInputValue}
 *   onTagAdd={(tag, category) => {
 *     // Add tag with category
 *     setTags([...tags, tag]);
 *     setCategories({ ...categories, [tag]: category });
 *   }}
 *   placeholder="Type to search tags..."
 *   showSuggestionsAbove={true}
 * />
 */
export default function InputWithTagSuggestions({
  value,
  onChange,
  onTagAdd,
  placeholder = "Type to search tags...",
  className = "",
  showSuggestionsAbove = false,
}: InputWithTagSuggestionsProps) {
  const tagSuggestionsContext = useTagSuggestions();
  const [tagSuggestions, setTagSuggestions] = useState<TagSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(0);
  const [filterMode, setFilterMode] = useState<TagFilterMode>("all");
  const [suggestionPosition, setSuggestionPosition] = useState({ top: 0, left: 0 });
  const [suppressSuggestions, setSuppressSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Tag suggestions search
  useEffect(() => {
    const handleSearch = async () => {
      if (value.trim().length < 2) {
        setTagSuggestions([]);
        setShowSuggestions(false);
        return;
      }

      // Don't show suggestions if user pressed Esc
      if (suppressSuggestions) {
        return;
      }

      try {
        const results = await tagSuggestionsContext.searchTags(value.trim(), 20, filterMode);
        const suggestions: TagSuggestion[] = results.map(r => ({
          tag: r.tag,
          count: r.count,
          category: r.category,
        }));

        setTagSuggestions(suggestions);
        setShowSuggestions(suggestions.length > 0);
        setSelectedSuggestionIndex(0);

        // Update position
        if (inputRef.current) {
          const rect = inputRef.current.getBoundingClientRect();
          const suggestionsHeight = 256;

          if (showSuggestionsAbove) {
            setSuggestionPosition({
              top: rect.top + window.scrollY - suggestionsHeight - 8,
              left: rect.left + window.scrollX,
            });
          } else {
            setSuggestionPosition({
              top: rect.bottom + window.scrollY + 4,
              left: rect.left + window.scrollX,
            });
          }
        }
      } catch (err) {
        console.error("[InputWithTagSuggestions] Failed to search tags:", err);
      }
    };

    const debounceTimer = setTimeout(handleSearch, 300);
    return () => clearTimeout(debounceTimer);
  }, [value, filterMode, tagSuggestionsContext, suppressSuggestions, showSuggestionsAbove]);

  const handleAddTag = (tagToAdd?: string) => {
    const tag = tagToAdd || value.trim();
    if (!tag) return;

    // Find category from suggestions
    const suggestion = tagSuggestions.find(s => s.tag === tag);
    const category = suggestion?.category || "general";

    onTagAdd(tag, category);
    onChange("");
    setShowSuggestions(false);
    setSuppressSuggestions(false);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
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
      setSuppressSuggestions(true);
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value);
    setSuppressSuggestions(false);
  };

  const handleFilterChange = (direction: 'next' | 'prev') => {
    const newMode = direction === 'next'
      ? tagSuggestionsContext.getNextFilterMode(filterMode)
      : tagSuggestionsContext.getPreviousFilterMode(filterMode);
    setFilterMode(newMode);
  };

  return (
    <div className="relative">
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className={className}
      />

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
