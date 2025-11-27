"use client";

import { useEffect, useState, useRef } from "react";
import { TagFilterMode, getFilterDisplayName } from "@/utils/tagSuggestions";

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
  filterMode: TagFilterMode;
  onFilterChange: (direction: 'next' | 'prev') => void;
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
  filterMode,
  onFilterChange,
}: TagSuggestionsProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isMobile, setIsMobile] = useState(false);
  const [adjustedPosition, setAdjustedPosition] = useState(position);

  // Swipe gesture state
  const touchStartX = useRef<number>(0);
  const touchEndX = useRef<number>(0);

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
      const listElement = containerRef.current.querySelector('.suggestions-list');
      if (listElement) {
        const selectedElement = listElement.children[selectedIndex] as HTMLElement;
        if (selectedElement) {
          selectedElement.scrollIntoView({ block: "nearest" });
        }
      }
    }
  }, [selectedIndex]);

  // Handle touch gestures for mobile swipe
  const handleTouchStart = (e: React.TouchEvent) => {
    touchStartX.current = e.touches[0].clientX;
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    touchEndX.current = e.touches[0].clientX;
  };

  const handleTouchEnd = () => {
    const swipeThreshold = 50; // Minimum swipe distance in pixels
    const diff = touchStartX.current - touchEndX.current;

    if (Math.abs(diff) > swipeThreshold) {
      if (diff > 0) {
        // Swipe left -> next filter
        onFilterChange('next');
      } else {
        // Swipe right -> previous filter
        onFilterChange('prev');
      }
    }

    // Reset
    touchStartX.current = 0;
    touchEndX.current = 0;
  };

  if (suggestions.length === 0) {
    return null;
  }

  return (
    <div
      ref={containerRef}
      className={`fixed z-50 bg-gray-800 border border-gray-600 rounded-lg shadow-lg ${
        isMobile ? 'text-sm' : ''
      }`}
      style={{
        top: adjustedPosition.top + 'px',
        left: adjustedPosition.left + 'px',
        minWidth: isMobile ? "auto" : "300px",
        maxWidth: isMobile ? `${Math.min(window.innerWidth - 32, 300)}px` : "500px",
        width: isMobile ? `${Math.min(window.innerWidth - 32, 300)}px` : "auto",
      }}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      {/* Filter mode indicator */}
      <div className="px-3 py-2 bg-gray-700 border-b border-gray-600 flex items-center justify-between">
        <span className="text-xs text-gray-300">
          Filter: <span className="font-semibold text-blue-400">{getFilterDisplayName(filterMode)}</span>
        </span>
        <span className="text-xs text-gray-400">
          {isMobile ? '← Swipe →' : 'Alt+←/→'}
        </span>
      </div>

      {/* Suggestions list */}
      <div className="suggestions-list max-h-56 overflow-y-auto">
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
    </div>
  );
}
