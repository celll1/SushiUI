"use client";

import { useState, useEffect } from "react";
import { Search, Plus, Download, Upload, RefreshCw } from "lucide-react";

interface TagDictionaryEntry {
  id: number;
  tag: string;
  category: string;
  count: number;
  is_official: boolean;
}

export default function TagDictionaryManager() {
  const [tags, setTags] = useState<TagDictionaryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");

  const categories = ["all", "character", "artist", "copyright", "general", "meta", "model"];

  const loadTags = async () => {
    setLoading(true);
    try {
      // TODO: Implement API call
      // const response = await api.get("/tag-dictionary", { params: { search: searchQuery, category: categoryFilter } });
      // setTags(response.data);
      setTags([]); // Placeholder
    } catch (err) {
      console.error("Failed to load tags:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTags();
  }, [searchQuery, categoryFilter]);

  const handleImport = () => {
    console.log("Import tags");
    // TODO: Open file picker and import
  };

  const handleExport = () => {
    console.log("Export tags");
    // TODO: Export to JSON/CSV
  };

  const handleAddTag = () => {
    console.log("Add new tag");
    // TODO: Open add tag modal
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Tag Dictionary</h2>
        <div className="flex space-x-2">
          <button
            onClick={handleImport}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm flex items-center space-x-1 transition-colors"
          >
            <Upload className="h-4 w-4" />
            <span>Import</span>
          </button>
          <button
            onClick={handleExport}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm flex items-center space-x-1 transition-colors"
          >
            <Download className="h-4 w-4" />
            <span>Export</span>
          </button>
          <button
            onClick={handleAddTag}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-sm flex items-center space-x-1 transition-colors"
          >
            <Plus className="h-4 w-4" />
            <span>Add Tag</span>
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex space-x-2 mb-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search tags..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
          />
        </div>
        <select
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value)}
          className="px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
        >
          {categories.map((cat) => (
            <option key={cat} value={cat}>
              {cat.charAt(0).toUpperCase() + cat.slice(1)}
            </option>
          ))}
        </select>
        <button
          onClick={loadTags}
          className="p-2 rounded bg-gray-700 hover:bg-gray-600 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="h-4 w-4" />
        </button>
      </div>

      {/* Tags Table */}
      <div className="bg-gray-900/50 rounded overflow-hidden">
        {loading ? (
          <div className="text-center text-gray-400 py-8">Loading tags...</div>
        ) : tags.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            <p>No tags found</p>
            <p className="text-sm mt-1">
              {searchQuery || categoryFilter !== "all"
                ? "Try different filters"
                : "Import tags from taglist/*.json"}
            </p>
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-gray-800 text-gray-300">
              <tr>
                <th className="text-left p-2">Tag</th>
                <th className="text-left p-2">Category</th>
                <th className="text-right p-2">Count</th>
                <th className="text-center p-2">Source</th>
              </tr>
            </thead>
            <tbody>
              {tags.map((tag) => (
                <tr
                  key={tag.id}
                  className="border-t border-gray-800 hover:bg-gray-800/50 cursor-pointer"
                >
                  <td className="p-2">{tag.tag}</td>
                  <td className="p-2">
                    <span className="px-2 py-0.5 bg-blue-900/50 text-blue-300 rounded text-xs">
                      {tag.category}
                    </span>
                  </td>
                  <td className="p-2 text-right text-gray-400">{tag.count.toLocaleString()}</td>
                  <td className="p-2 text-center">
                    {tag.is_official ? (
                      <span className="text-green-400 text-xs">Official</span>
                    ) : (
                      <span className="text-yellow-400 text-xs">Custom</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
