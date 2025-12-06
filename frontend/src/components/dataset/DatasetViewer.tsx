"use client";

import { useState, useEffect, useCallback } from "react";
import { listDatasetItems, DatasetItem, Dataset, getDataset } from "@/utils/api";
import { normalizeTagForMatching } from "@/utils/tagSuggestions";
import { useTagSuggestions } from "@/contexts/TagSuggestionsContext";
import ItemGridColumn from "./viewer/ItemGridColumn";
import ItemDetailColumn from "./viewer/ItemDetailColumn";
import ActionsColumn from "./viewer/ActionsColumn";

interface DatasetViewerProps {
  datasetId: number;
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

  const handleSelectAll = () => {
    const newSelected = new Set(items.map(item => item.id));
    setSelectedItems(newSelected);
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

  return (
    <div className="flex h-full gap-3">
      {/* Left Column: Item Grid */}
      <div className="w-80 flex-shrink-0 flex flex-col bg-gray-900/50 rounded-lg">
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
          onSelectItem={handleSelectItem}
          onToggleSelection={handleToggleSelection}
          onSearchChange={handleSearchChange}
          onTagFilterChange={handleTagFilterChange}
          onPageChange={setPage}
        />
      </div>

      {/* Center Column: Detail View */}
      <div className="flex-1 flex flex-col bg-gray-900/50 rounded-lg min-w-0">
        <ItemDetailColumn
          item={currentItem}
          datasetId={datasetId}
          tagCategoryCache={tagCategoryCache}
        />
      </div>

      {/* Right Column: Actions */}
      <div className="w-80 flex-shrink-0 flex flex-col bg-gray-900/50 rounded-lg">
        <ActionsColumn
          datasetId={datasetId}
          selectedItems={selectedItems}
          totalItems={total}
          tagStatistics={tagStatistics}
          onSelectAll={handleSelectAll}
          onDeselectAll={handleDeselectAll}
          onRefresh={loadItems}
        />
      </div>
    </div>
  );
}
