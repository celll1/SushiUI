"use client";

import { useState, useEffect } from "react";
import { listDatasetItems, DatasetItem } from "@/utils/api";
import ItemGridColumn from "./viewer/ItemGridColumn";
import ItemDetailColumn from "./viewer/ItemDetailColumn";
import ActionsColumn from "./viewer/ActionsColumn";

interface DatasetViewerProps {
  datasetId: number;
}

export default function DatasetViewer({ datasetId }: DatasetViewerProps) {
  const [items, setItems] = useState<DatasetItem[]>([]);
  const [selectedItems, setSelectedItems] = useState<Set<number>>(new Set());
  const [currentItem, setCurrentItem] = useState<DatasetItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const pageSize = 50;

  useEffect(() => {
    loadItems();
  }, [datasetId, page, search]);

  const loadItems = async () => {
    setLoading(true);
    try {
      const response = await listDatasetItems(datasetId, page, pageSize, search || undefined);
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

  return (
    <div className="flex h-full gap-3">
      {/* Left Column: Item Grid */}
      <div className="w-80 flex-shrink-0 flex flex-col bg-gray-900/50 rounded-lg">
        <ItemGridColumn
          items={items}
          selectedItems={selectedItems}
          currentItem={currentItem}
          search={search}
          page={page}
          total={total}
          pageSize={pageSize}
          loading={loading}
          onSelectItem={handleSelectItem}
          onToggleSelection={handleToggleSelection}
          onSearchChange={handleSearchChange}
          onPageChange={setPage}
        />
      </div>

      {/* Center Column: Detail View */}
      <div className="flex-1 flex flex-col bg-gray-900/50 rounded-lg min-w-0">
        <ItemDetailColumn
          item={currentItem}
          datasetId={datasetId}
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
