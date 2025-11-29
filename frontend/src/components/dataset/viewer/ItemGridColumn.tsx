"use client";

import { Search, ChevronLeft, ChevronRight, Filter } from "lucide-react";
import { DatasetItem } from "@/utils/api";

interface ItemGridColumnProps {
  items: DatasetItem[];
  selectedItems: Set<number>;
  currentItem: DatasetItem | null;
  search: string;
  page: number;
  total: number;
  pageSize: number;
  loading: boolean;
  onSelectItem: (item: DatasetItem) => void;
  onToggleSelection: (itemId: number) => void;
  onSearchChange: (search: string) => void;
  onPageChange: (page: number) => void;
}

export default function ItemGridColumn({
  items,
  selectedItems,
  currentItem,
  search,
  page,
  total,
  pageSize,
  loading,
  onSelectItem,
  onToggleSelection,
  onSearchChange,
  onPageChange,
}: ItemGridColumnProps) {
  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-3 border-b border-gray-700">
        <h3 className="text-sm font-semibold mb-3">Items ({total})</h3>

        {/* Search */}
        <div className="relative mb-2">
          <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search..."
            className="w-full pl-8 pr-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-xs focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Filters button */}
        <button className="w-full flex items-center justify-center space-x-1 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-xs transition-colors">
          <Filter className="h-3.5 w-3.5" />
          <span>Filters</span>
        </button>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-2">
        {loading && items.length === 0 ? (
          <div className="text-center text-gray-400 text-xs py-8">Loading...</div>
        ) : items.length === 0 ? (
          <div className="text-center text-gray-400 text-xs py-8">
            {search ? "No items found" : "No items"}
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
