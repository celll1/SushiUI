"use client";

/**
 * GalleryFilter - Memoized filter panel for gallery
 *
 * This component is wrapped with React.memo to prevent re-renders
 * when only the image list changes (e.g., pagination).
 * All props must be stable references (memoized with useCallback/useMemo).
 */

import React, { memo } from "react";
import Card from "../common/Card";
import Button from "../common/Button";
import RangeSlider from "../common/RangeSlider";
import Slider from "../common/Slider";

interface GalleryFilterProps {
  // Filter states
  filterTxt2Img: boolean;
  setFilterTxt2Img: (value: boolean) => void;
  filterImg2Img: boolean;
  setFilterImg2Img: (value: boolean) => void;
  filterInpaint: boolean;
  setFilterInpaint: (value: boolean) => void;
  dateFrom: string;
  setDateFrom: (value: string) => void;
  dateTo: string;
  setDateTo: (value: string) => void;
  widthRange: [number, number];
  setWidthRange: (value: [number, number]) => void;
  heightRange: [number, number];
  setHeightRange: (value: [number, number]) => void;
  setCommittedWidthRange: (value: [number, number]) => void;
  setCommittedHeightRange: (value: [number, number]) => void;
  tagSearchInput: string;
  setTagSearchInput: (value: string) => void;
  tagSearchCommitted: string[];
  setTagSearchCommitted: (value: string[]) => void;
  searchInNegative: boolean;
  setSearchInNegative: (value: boolean) => void;
  showSuggestions: boolean;
  setShowSuggestions: (value: boolean) => void;
  selectedSuggestionIndex: number;
  setSelectedSuggestionIndex: (value: number) => void;
  excludeRareTags: boolean;
  setExcludeRareTags: (value: boolean) => void;
  tagSuggestions: string[];
  handleTagSearchSubmit: () => void;
  handleTagSearchKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  handleSuggestionClick: (suggestion: string) => void;
  removeTag: (tag: string) => void;
  clearAllTags: () => void;

  // UI states
  gridColumns: number;
  setGridColumns: (value: number) => void;

  // Pagination
  currentPage: number;
  setCurrentPage: (value: number) => void;
  totalImages: number;
  imagesPerPage: number;
  loading: boolean;
}

const GalleryFilter: React.FC<GalleryFilterProps> = memo(({
  filterTxt2Img,
  setFilterTxt2Img,
  filterImg2Img,
  setFilterImg2Img,
  filterInpaint,
  setFilterInpaint,
  dateFrom,
  setDateFrom,
  dateTo,
  setDateTo,
  widthRange,
  setWidthRange,
  heightRange,
  setHeightRange,
  setCommittedWidthRange,
  setCommittedHeightRange,
  tagSearchInput,
  setTagSearchInput,
  tagSearchCommitted,
  setTagSearchCommitted,
  searchInNegative,
  setSearchInNegative,
  showSuggestions,
  setShowSuggestions,
  selectedSuggestionIndex,
  setSelectedSuggestionIndex,
  excludeRareTags,
  setExcludeRareTags,
  tagSuggestions,
  handleTagSearchSubmit,
  handleTagSearchKeyDown,
  handleSuggestionClick,
  removeTag,
  clearAllTags,
  gridColumns,
  setGridColumns,
  currentPage,
  setCurrentPage,
  totalImages,
  imagesPerPage,
  loading,
}) => {
  return (
    <div className="w-full lg:w-80 flex-shrink-0">
      <Card title="Filters" defaultCollapsed={false} storageKey="gallery_filters_collapsed">
        <div className="space-y-3">
          {/* Generation Type Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Generation Type</label>
            <div className="space-y-1">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={filterTxt2Img}
                  onChange={(e) => setFilterTxt2Img(e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm text-gray-300">txt2img</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={filterImg2Img}
                  onChange={(e) => setFilterImg2Img(e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm text-gray-300">img2img</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={filterInpaint}
                  onChange={(e) => setFilterInpaint(e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm text-gray-300">inpaint</span>
              </label>
            </div>
          </div>

          {/* Tag Search Filter */}
          <div>
            <div className="mb-2">
              <label className="block text-sm font-medium text-gray-300 mb-1">Tag Search</label>
              <div className="flex flex-col gap-1">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={searchInNegative}
                    onChange={(e) => setSearchInNegative(e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-xs text-gray-400">Search in negative prompt</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={excludeRareTags}
                    onChange={(e) => setExcludeRareTags(e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-xs text-gray-400">Exclude tags appearing only once</span>
                </label>
              </div>
            </div>
            <div className="relative">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={tagSearchInput}
                  onChange={(e) => {
                    setTagSearchInput(e.target.value);
                    setShowSuggestions(e.target.value.length >= 2);
                    setSelectedSuggestionIndex(-1);
                  }}
                  onKeyDown={handleTagSearchKeyDown}
                  onFocus={() => setShowSuggestions(tagSearchInput.length >= 2)}
                  onBlur={() => setTimeout(() => {
                    setShowSuggestions(false);
                    setSelectedSuggestionIndex(-1);
                  }, 200)}
                  placeholder="Enter tag (press Enter to search)"
                  className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
                <Button
                  onClick={handleTagSearchSubmit}
                  variant="primary"
                  size="sm"
                >
                  Search
                </Button>
              </div>

              {/* Autocomplete suggestions */}
              {showSuggestions && tagSuggestions.length > 0 && (
                <div className="absolute z-10 w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                  {tagSuggestions.map((suggestion, index) => (
                    <button
                      key={index}
                      onClick={() => handleSuggestionClick(suggestion)}
                      className={`w-full px-3 py-2 text-left text-sm text-gray-100 hover:bg-gray-700 focus:outline-none ${
                        index === selectedSuggestionIndex ? 'bg-gray-700' : ''
                      }`}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </div>
            {tagSearchCommitted.length > 0 && (
              <div className="mt-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-gray-400">Searching for (AND):</span>
                  <button
                    onClick={clearAllTags}
                    className="text-xs text-red-400 hover:text-red-300"
                  >
                    Clear All
                  </button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {tagSearchCommitted.map((tag, index) => (
                    <div
                      key={index}
                      className="inline-flex items-center gap-1 px-2 py-1 bg-blue-600 text-white text-xs rounded border border-blue-500"
                    >
                      <span>{tag}</span>
                      <button
                        onClick={() => removeTag(tag)}
                        className="hover:text-red-300 focus:outline-none"
                        title="Remove tag"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Date Range Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Date Range</label>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">From</label>
                <input
                  type="date"
                  value={dateFrom}
                  onChange={(e) => setDateFrom(e.target.value)}
                  className="w-full px-2 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 text-xs cursor-pointer hover:border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 [&::-webkit-calendar-picker-indicator]:cursor-pointer [&::-webkit-calendar-picker-indicator]:opacity-60 [&::-webkit-calendar-picker-indicator]:hover:opacity-100"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">To</label>
                <input
                  type="date"
                  value={dateTo}
                  onChange={(e) => setDateTo(e.target.value)}
                  className="w-full px-2 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 text-xs cursor-pointer hover:border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 [&::-webkit-calendar-picker-indicator]:cursor-pointer [&::-webkit-calendar-picker-indicator]:opacity-60 [&::-webkit-calendar-picker-indicator]:hover:opacity-100"
                />
              </div>
            </div>
          </div>

          {/* Size Range Filters */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <RangeSlider
                label="Width Range"
                min={0}
                max={2048}
                step={64}
                value={widthRange}
                onChange={setWidthRange}
                onCommit={setCommittedWidthRange}
              />
            </div>

            <div>
              <RangeSlider
                label="Height Range"
                min={0}
                max={2048}
                step={64}
                value={heightRange}
                onChange={setHeightRange}
                onCommit={setCommittedHeightRange}
              />
            </div>
          </div>

          {/* Grid Layout Control */}
          <div>
            <Slider
              label="Images per Row"
              min={2}
              max={8}
              step={1}
              value={gridColumns}
              onChange={(e) => setGridColumns(Number(e.target.value))}
            />
          </div>

          {/* Pagination */}
          <div className="border-t border-gray-700 pt-3 mt-3">
            <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
              <span>
                {totalImages > 0 ? (
                  <>
                    {(currentPage - 1) * imagesPerPage + 1}-
                    {Math.min(currentPage * imagesPerPage, totalImages)} of {totalImages}
                  </>
                ) : (
                  "No images"
                )}
              </span>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1 || loading}
                variant="secondary"
                size="sm"
                className="flex-1"
              >
                Previous
              </Button>
              <Button
                onClick={() => setCurrentPage(currentPage + 1)}
                disabled={currentPage * imagesPerPage >= totalImages || loading}
                variant="secondary"
                size="sm"
                className="flex-1"
              >
                Next
              </Button>
            </div>
            <div className="text-center mt-2 text-xs text-gray-500">
              Page {currentPage} of {Math.max(1, Math.ceil(totalImages / imagesPerPage))}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
});

GalleryFilter.displayName = "GalleryFilter";

export default GalleryFilter;
