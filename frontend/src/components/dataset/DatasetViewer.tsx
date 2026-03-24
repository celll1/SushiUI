"use client";

import { useState, useEffect, useCallback } from "react";
import { listDatasetItems, DatasetItem, Dataset, getDataset, getAllDatasetItemIds } from "@/utils/api";
import { normalizeTagForMatching } from "@/utils/tagSuggestions";
import { useTagSuggestions } from "@/contexts/TagSuggestionsContext";
import ItemGridColumn from "./viewer/ItemGridColumn";
import ItemDetailColumn from "./viewer/ItemDetailColumn";
import ActionsColumn from "./viewer/ActionsColumn";

interface DatasetViewerProps {
  datasetId: number;
}

// Tagger settings interface (from ItemDetailColumn)
interface TaggerSettings {
  categoryThresholds: Array<{
    id: string;
    label: string;
    addThreshold: number;
    removeThreshold: number;
    enabled: boolean;
  }>;
}

export default function DatasetViewer({ datasetId }: DatasetViewerProps) {
  const tagSuggestionsContext = useTagSuggestions();
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [items, setItems] = useState<DatasetItem[]>([]);
  const [selectedItems, setSelectedItems] = useState<Set<number>>(new Set());
  const [currentItem, setCurrentItem] = useState<DatasetItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [tagFilter, setTagFilter] = useState(""); // Comma-separated tags
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const pageSize = 50;

  // Tag category cache (tag -> category)
  const [tagCategoryCache, setTagCategoryCache] = useState<Record<string, string>>({});
  // Tag statistics with categories (tag -> {category, count})
  const [tagStatistics, setTagStatistics] = useState<Record<string, { category: string; count: number }> | undefined>(undefined);

  // Tagger settings (shared across batch operations)
  const [taggerSettings, setTaggerSettings] = useState<TaggerSettings | null>(null);

  // Load dataset and compute categories using tagSuggestions
  useEffect(() => {
    const loadDataset = async () => {
      if (!tagSuggestionsContext.isLoaded) {
        return; // Wait for tag suggestions to load
      }

      try {
        const data = await getDataset(datasetId);
        setDataset(data);

        // Build tag category cache using tagSuggestions (batch operation)
        if (data.tag_statistics) {
          const tags = Object.keys(data.tag_statistics);

          // Batch categorize all tags at once (much faster than individual searches)
          const categoryMap = await tagSuggestionsContext.getCategoriesForTags(tags);

          // Convert Map to Record for state
          const categoryRecord: Record<string, string> = {};
          const statsWithCategories: Record<string, { category: string; count: number }> = {};

          for (const [tag, stats] of Object.entries(data.tag_statistics)) {
            const category = categoryMap.get(tag) || "Unknown";
            categoryRecord[tag] = category;
            statsWithCategories[tag] = {
              category,
              count: stats.count
            };
          }

          setTagCategoryCache(categoryRecord);
          setTagStatistics(statsWithCategories);
          console.log(`[DatasetViewer] Loaded ${Object.keys(categoryRecord).length} tag categories using batch categorization`);
        }
      } catch (err) {
        console.error("[DatasetViewer] Failed to load dataset:", err);
      }
    };

    loadDataset();
  }, [datasetId, tagSuggestionsContext.isLoaded]);

  useEffect(() => {
    loadItems();
  }, [datasetId, page, search, tagFilter]);

  const loadItems = async () => {
    setLoading(true);
    try {
      const response = await listDatasetItems(
        datasetId,
        page,
        pageSize,
        search || undefined,
        tagFilter || undefined
      );
      setItems(response.items);
      setTotal(response.total);

      // Auto-select first item if none selected
      if (!currentItem && response.items.length > 0) {
        setCurrentItem(response.items[0]);
      }
    } catch (err) {
      console.error("Failed to load dataset items:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectItem = (item: DatasetItem) => {
    setCurrentItem(item);
  };

  const handleToggleSelection = (itemId: number) => {
    const newSelected = new Set(selectedItems);
    if (newSelected.has(itemId)) {
      newSelected.delete(itemId);
    } else {
      newSelected.add(itemId);
    }
    setSelectedItems(newSelected);
  };

  const handleSelectAll = async () => {
    try {
      // Fetch all item IDs (respecting current search/tag filters)
      const response = await getAllDatasetItemIds(
        datasetId,
        search || undefined,
        tagFilter || undefined
      );
      setSelectedItems(new Set(response.item_ids));
    } catch (err) {
      console.error("Failed to select all items:", err);
    }
  };

  const handleDeselectAll = () => {
    setSelectedItems(new Set());
  };

  const handleSearchChange = (value: string) => {
    setSearch(value);
    setPage(1);
  };

  const handleTagFilterChange = (value: string) => {
    setTagFilter(value);
    setPage(1);
  };

  const [mobileDetailOpen, setMobileDetailOpen] = useState(false);
  const [mobileActionsOpen, setMobileActionsOpen] = useState(false);

  // Open detail panel on mobile when item is selected
  const handleSelectItemMobile = useCallback((item: DatasetItem) => {
    handleSelectItem(item);
    setMobileDetailOpen(true);
  }, [handleSelectItem]);

  return (
    <div className="flex flex-col lg:flex-row h-full gap-0 lg:gap-3 relative">
      {/* Left Column: Item Grid - always visible on mobile, fixed width on desktop */}
      <div className={`
        w-full lg:w-80 lg:flex-shrink-0 flex flex-col bg-gray-900/50 rounded-lg
        min-h-0 overflow-y-auto lg:overflow-visible
        ${mobileDetailOpen ? 'hidden lg:flex' : 'flex'}
      `}>
        <ItemGridColumn
          items={items}
          selectedItems={selectedItems}
          currentItem={currentItem}
          search={search}
          tagFilter={tagFilter}
          page={page}
          total={total}
          pageSize={pageSize}
          loading={loading}
          onSelectItem={handleSelectItemMobile}
          onToggleSelection={handleToggleSelection}
          onSearchChange={handleSearchChange}
          onTagFilterChange={handleTagFilterChange}
          onPageChange={setPage}
          onSelectAll={handleSelectAll}
          onDeselectAll={handleDeselectAll}
        />
      </div>

      {/* Center Column: Detail View - fullscreen overlay on mobile */}
      <div className={`
        fixed inset-0 z-30 lg:relative lg:z-auto
        lg:flex-1 flex flex-col bg-gray-900 lg:bg-gray-900/50 rounded-lg min-w-0
        ${mobileDetailOpen ? 'flex' : 'hidden lg:flex'}
      `}>
        {/* Mobile back button */}
        <button
          onClick={() => setMobileDetailOpen(false)}
          className="lg:hidden flex items-center gap-2 px-3 py-2 text-sm text-gray-300 bg-gray-800 border-b border-gray-700"
        >
          <span>&#8592;</span> Back to Grid
        </button>
        <ItemDetailColumn
          item={currentItem}
          datasetId={datasetId}
          tagCategoryCache={tagCategoryCache}
          onTaggerSettingsChange={setTaggerSettings}
        />
      </div>

      {/* Right Column: Actions - slide-over on mobile */}
      <div className={`
        fixed top-0 right-0 h-full w-80 max-w-[calc(100vw-3rem)] z-40 lg:relative lg:z-auto
        lg:flex-shrink-0 flex flex-col bg-gray-900 lg:bg-gray-900/50 rounded-lg
        transform transition-transform duration-200 ease-in-out
        ${mobileActionsOpen ? 'translate-x-0' : 'translate-x-full lg:translate-x-0'}
        overflow-y-auto lg:overflow-visible
        shadow-2xl lg:shadow-none
      `}>
        {/* Mobile close button */}
        <button
          onClick={() => setMobileActionsOpen(false)}
          className="lg:hidden flex items-center justify-between px-3 py-2 text-sm text-gray-300 bg-gray-800 border-b border-gray-700"
        >
          <span>Actions & Statistics</span>
          <span>&times;</span>
        </button>
        <ActionsColumn
          datasetId={datasetId}
          tagStatistics={tagStatistics}
          onRefresh={loadItems}
          selectedItemIds={Array.from(selectedItems)}
          totalItems={total}
          captionProcessingConfig={dataset?.caption_processing}
          taggerSettings={taggerSettings}
        />
      </div>

      {/* Mobile backdrop for actions panel */}
      {mobileActionsOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => setMobileActionsOpen(false)}
        />
      )}

      {/* Mobile FAB: Actions toggle */}
      <button
        onClick={() => setMobileActionsOpen(!mobileActionsOpen)}
        className="fixed bottom-4 right-4 z-50 p-3 rounded-full bg-blue-600 text-white shadow-lg lg:hidden"
        title="Actions & Statistics"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" />
          <line x1="12" y1="21" x2="12" y2="12" /><line x1="12" y1="8" x2="12" y2="3" />
          <line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" />
          <line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" />
          <line x1="17" y1="16" x2="23" y2="16" />
        </svg>
      </button>
    </div>
  );
}
