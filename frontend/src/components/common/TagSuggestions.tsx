"use client";

import { useEffect, useState, useRef } from "react";

interface TagSuggestion {
  tag: string;
  count: number;
  category: string;
  alias?: string;
}

interface TagSuggestionsProps {
  suggestions: TagSuggestion[];
  selectedIndex: number;
  onSelect: (tag: string) => void;
  position: { top: number; left: number };
}

// Category colors
const CATEGORY_COLORS: Record<string, string> = {
  "General": "text-blue-400",
  "Character": "text-green-400",
  "Artist": "text-purple-400",
  "Copyright": "text-yellow-400",
  "Meta": "text-pink-400",
  "Model": "text-cyan-400",
  "Rating Tag": "text-red-400",
  "Quality Tag": "text-orange-400",
};

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
  const [isMobile, setIsMobile] = useState(false);
  const [adjustedPosition, setAdjustedPosition] = useState(position);

  // Detect mobile
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Adjust position to prevent overflow
  useEffect(() => {
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const suggestionsWidth = isMobile ? Math.min(viewportWidth - 32, 300) : 500;
    const suggestionsHeight = 256; // max-h-64 = 16rem = 256px

    let newLeft = position.left;
    let newTop = position.top;

    // Prevent horizontal overflow
    if (newLeft + suggestionsWidth > viewportWidth - 16) {
      newLeft = Math.max(16, viewportWidth - suggestionsWidth - 16);
    }

    // Prevent vertical overflow
    if (newTop + suggestionsHeight > viewportHeight - 16) {
      // Show above cursor instead
      newTop = Math.max(16, position.top - suggestionsHeight - 10);
    }

    setAdjustedPosition({ top: newTop, left: newLeft });
  }, [position, isMobile]);

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
      className={`fixed z-50 bg-gray-800 border border-gray-600 rounded-lg shadow-lg max-h-64 overflow-y-auto ${
        isMobile ? 'text-sm' : ''
      }`}
      style={{
        top: adjustedPosition.top + 'px',
        left: adjustedPosition.left + 'px',
        minWidth: isMobile ? "auto" : "300px",
        maxWidth: isMobile ? `${Math.min(window.innerWidth - 32, 300)}px` : "500px",
        width: isMobile ? `${Math.min(window.innerWidth - 32, 300)}px` : "auto",
      }}
    >
      {suggestions.map((suggestion, index) => {
        const categoryColor = CATEGORY_COLORS[suggestion.category] || "text-gray-400";
        const isSpecialTag = suggestion.count === -1;

        // For special tags, show shortened category name (e.g., "Quality" instead of "Quality Tag")
        const displayCategory = isSpecialTag
          ? suggestion.category.replace(' Tag', '')
          : suggestion.category;

        return (
          <div
            key={`${suggestion.category}-${suggestion.tag}-${suggestion.alias || 'main'}-${index}`}
            className={`cursor-pointer transition-colors ${
              isMobile ? 'px-2 py-1.5' : 'px-3 py-2'
            } ${
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
              <span
                className={`font-medium truncate ${
                  index === selectedIndex ? '' : categoryColor
                }`}
              >
                {suggestion.alias ? `${suggestion.tag.replace(/_/g, " ")} < ${suggestion.alias}` : suggestion.tag.replace(/_/g, " ")}
              </span>
              <span className={`text-xs flex-shrink-0 ${index === selectedIndex ? 'text-blue-200' : 'text-gray-400'}`}>
                {isSpecialTag ? suggestion.category : suggestion.count.toLocaleString()}
              </span>
            </div>
            {!isMobile && (
              <div className={`text-xs mt-0.5 font-semibold ${index === selectedIndex ? 'text-blue-200' : categoryColor}`}>
                {displayCategory}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
