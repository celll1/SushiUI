"use client";

import { useState, useEffect, useCallback } from "react";
import { listDatasetItems, DatasetItem, getDatasetTags } from "@/utils/api";
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
  const [items, setItems] = useState<DatasetItem[]>([]);
  const [selectedItems, setSelectedItems] = useState<Set<number>>(new Set());
  const [currentItem, setCurrentItem] = useState<DatasetItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [tagFilter, setTagFilter] = useState(""); // Comma-separated tags
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const pageSize = 50;

  // Tag category cache for entire dataset (shared across all items)
  const [tagCategoryCache, setTagCategoryCache] = useState<Record<string, string>>({});
  const [cacheLoading, setCacheLoading] = useState(false);

  // Load tag categories for entire dataset (runs once on mount)
  useEffect(() => {
    const loadTagCategories = async () => {
      if (!tagSuggestionsContext.isLoaded) {
        return; // Wait for tag suggestions to load
      }

      setCacheLoading(true);
      try {
        console.log("[DatasetViewer] Loading all tags for dataset...");
        const allTags = await getDatasetTags(datasetId);
        console.log(`[DatasetViewer] Found ${allTags.length} unique tags`);

        // Fetch categories for all tags (batched search)
        const categoryMap: Record<string, string> = {};
        let foundCount = 0;
        let notFoundCount = 0;

        for (const tag of allTags) {
          const results = await tagSuggestionsContext.searchTags(tag, 1, 'all');
          if (results.length > 0) {
            const normalizedUserTag = normalizeTagForMatching(tag);
            const normalizedResultTag = normalizeTagForMatching(results[0].tag);
            if (normalizedUserTag === normalizedResultTag) {
              categoryMap[tag] = results[0].category;
              foundCount++;
            } else {
              notFoundCount++;
            }
          } else {
            notFoundCount++;
          }
        }

        setTagCategoryCache(categoryMap);
        console.log(`[DatasetViewer] Tag category cache ready: ${foundCount} found, ${notFoundCount} not found`);
      } catch (err) {
        console.error("[DatasetViewer] Failed to load tag categories:", err);
      } finally {
        setCacheLoading(false);
      }
    };

    loadTagCategories();
  }, [datasetId, tagSuggestionsContext]);

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
          onSelectAll={handleSelectAll}
          onDeselectAll={handleDeselectAll}
          onRefresh={loadItems}
        />
      </div>
    </div>
  );
}
