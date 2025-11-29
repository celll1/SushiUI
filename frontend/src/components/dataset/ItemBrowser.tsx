"use client";

import { useState, useEffect } from "react";
import { Search, ChevronLeft, ChevronRight, Image as ImageIcon } from "lucide-react";
import { listDatasetItems, DatasetItem } from "@/utils/api";

interface ItemBrowserProps {
  datasetId: number;
  onSelectItem?: (item: DatasetItem) => void;
}

export default function ItemBrowser({ datasetId, onSelectItem }: ItemBrowserProps) {
  const [items, setItems] = useState<DatasetItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [pageSize] = useState(50);

  useEffect(() => {
    loadItems();
  }, [datasetId, page, search]);

  const loadItems = async () => {
    setLoading(true);
    try {
      const response = await listDatasetItems(datasetId, page, pageSize, search || undefined);
      setItems(response.items);
      setTotal(response.total);
    } catch (err) {
      console.error("Failed to load dataset items:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchChange = (value: string) => {
    setSearch(value);
    setPage(1); // Reset to first page on search
  };

  const totalPages = Math.ceil(total / pageSize);

  if (loading && items.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        <div className="text-center">
          <ImageIcon className="h-12 w-12 mx-auto mb-2 opacity-50" />
          <p>Loading items...</p>
        </div>
      </div>
    );
  }

  if (items.length === 0 && !loading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        <div className="text-center">
          <ImageIcon className="h-12 w-12 mx-auto mb-2 opacity-50" />
          <p>{search ? "No items found" : "No items in dataset"}</p>
          {!search && <p className="text-sm mt-2">Click &quot;Scan&quot; to find images</p>}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div className="flex items-center space-x-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
            placeholder="Search by filename..."
            className="w-full pl-10 pr-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
          />
        </div>
        <div className="text-sm text-gray-400">
          {total} item{total !== 1 ? "s" : ""}
        </div>
      </div>

      {/* Image Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
        {items.map((item) => (
          <div
            key={item.id}
            onClick={() => onSelectItem?.(item)}
            className="bg-gray-900 rounded border border-gray-700 hover:border-blue-500 cursor-pointer transition-colors overflow-hidden group"
          >
            {/* Image Thumbnail */}
            <div className="aspect-square bg-gray-800 relative">
              <img
                src={`/api/serve-image?path=${encodeURIComponent(item.image_path)}`}
                alt={item.base_name}
                className="w-full h-full object-cover"
                loading="lazy"
              />
              <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-opacity flex items-center justify-center">
                <ImageIcon className="h-8 w-8 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </div>

            {/* Item Info */}
            <div className="p-2">
              <p className="text-xs text-gray-300 truncate" title={item.base_name}>
                {item.base_name}
              </p>
              <div className="flex items-center justify-between mt-1">
                <span className="text-xs text-gray-500">
                  {item.width}×{item.height}
                </span>
                <span className="text-xs text-gray-500">
                  {(item.file_size / 1024).toFixed(0)}KB
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center space-x-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>

          <div className="flex items-center space-x-1">
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              let pageNum: number;
              if (totalPages <= 5) {
                pageNum = i + 1;
              } else if (page <= 3) {
                pageNum = i + 1;
              } else if (page >= totalPages - 2) {
                pageNum = totalPages - 4 + i;
              } else {
                pageNum = page - 2 + i;
              }

              return (
                <button
                  key={pageNum}
                  onClick={() => setPage(pageNum)}
                  className={`px-3 py-1 rounded text-sm transition-colors ${
                    page === pageNum
                      ? "bg-blue-600 text-white"
                      : "bg-gray-700 hover:bg-gray-600"
                  }`}
                >
                  {pageNum}
                </button>
              );
            })}
          </div>

          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
}
