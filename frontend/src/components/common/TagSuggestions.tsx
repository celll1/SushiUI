"use client";

import { useEffect, useState, useRef } from "react";

interface TagSuggestion {
  tag: string;
  count: number;
  category: string;
}

interface TagSuggestionsProps {
  suggestions: TagSuggestion[];
  selectedIndex: number;
  onSelect: (tag: string) => void;
  position: { top: number; left: number };
}

/**
 * TagSuggestions - Dropdown component for tag autocompletion
 * Displays tag suggestions with category and count information
 */
export default function TagSuggestions({
  suggestions,
  selectedIndex,
  onSelect,
  position,
}: TagSuggestionsProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Scroll selected item into view
  useEffect(() => {
    if (containerRef.current && selectedIndex >= 0) {
      const selectedElement = containerRef.current.children[selectedIndex] as HTMLElement;
      if (selectedElement) {
        selectedElement.scrollIntoView({ block: "nearest" });
      }
    }
  }, [selectedIndex]);

  if (suggestions.length === 0) {
    return null;
  }

  return (
    <div
      ref={containerRef}
      className="absolute z-50 bg-gray-800 border border-gray-600 rounded-lg shadow-lg max-h-64 overflow-y-auto"
      style={{
        top: position.top,
        left: position.left,
        minWidth: "300px",
        maxWidth: "500px",
      }}
    >
      {suggestions.map((suggestion, index) => (
        <div
          key={`${suggestion.category}-${suggestion.tag}`}
          className={`px-3 py-2 cursor-pointer transition-colors ${
            index === selectedIndex
              ? "bg-blue-600 text-white"
              : "hover:bg-gray-700 text-gray-100"
          }`}
          onClick={() => onSelect(suggestion.tag)}
          onMouseEnter={(e) => {
            // Prevent mouse hover from changing selection while using keyboard
            if (e.movementX !== 0 || e.movementY !== 0) {
              // Only change selection if mouse actually moved
            }
          }}
        >
          <div className="flex items-baseline justify-between gap-2">
            <span className="font-medium truncate">{suggestion.tag.replace(/_/g, " ")}</span>
            <span className="text-xs text-gray-400 flex-shrink-0">
              {suggestion.count.toLocaleString()}
            </span>
          </div>
          <div className="text-xs text-gray-400 mt-0.5">{suggestion.category}</div>
        </div>
      ))}
    </div>
  );
}
