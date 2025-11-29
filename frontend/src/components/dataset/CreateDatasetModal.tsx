"use client";

import { useState } from "react";
import { X, Folder, AlertCircle } from "lucide-react";
import { createDataset, scanDataset, Dataset } from "@/utils/api";

interface CreateDatasetModalProps {
  initialPath: string | null;
  onClose: () => void;
  onCreate: (dataset: Dataset) => void;
}

export default function CreateDatasetModal({ initialPath, onClose, onCreate }: CreateDatasetModalProps) {
  const [name, setName] = useState("");
  const [path, setPath] = useState(initialPath || "");
  const [description, setDescription] = useState("");
  const [recursive, setRecursive] = useState(true);
  const [readExif, setReadExif] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validation
    if (!path.trim()) {
      setError("Dataset path is required");
      return;
    }

    // Auto-generate name from folder if not provided
    let datasetName = name.trim();
    if (!datasetName) {
      const pathParts = path.trim().replace(/\\/g, "/").split("/");
      datasetName = pathParts[pathParts.length - 1] || "unnamed_dataset";
    }

    setLoading(true);
    setError(null);

    try {
      const newDataset = await createDataset({
        name: datasetName,
        path: path.trim(),
        description: description.trim() || undefined,
        recursive,
        read_exif: readExif,
      });

      // Automatically scan the dataset after creation
      try {
        const scanResult = await scanDataset(newDataset.id);
        onCreate(scanResult.dataset); // Return scanned dataset with updated counts
      } catch (scanErr) {
        console.error("Failed to scan dataset:", scanErr);
        onCreate(newDataset); // Return dataset even if scan fails
      }
    } catch (err: any) {
      setError(err.response?.data?.error || "Failed to create dataset");
      console.error("Failed to create dataset:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold">Create New Dataset</h2>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-gray-700 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {error && (
            <div className="bg-red-900/20 border border-red-500 text-red-400 rounded p-3 flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* Dataset Name */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Dataset Name (optional)
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., character_training_dataset (auto-filled from folder name if empty)"
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
            />
            <p className="text-xs text-gray-400 mt-1">Unique identifier for this dataset (uses folder name if left empty)</p>
          </div>

          {/* Dataset Path */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Dataset Directory Path <span className="text-red-400">*</span>
            </label>
            <div className="relative">
              <Folder className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="e.g., /path/to/your/dataset"
                className="w-full pl-10 pr-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500 font-mono"
                required
              />
            </div>
            <p className="text-xs text-gray-400 mt-1">
              Full path to the directory containing your training images
            </p>
            <p className="text-xs text-blue-400 mt-1">
              Windows: <code className="bg-gray-900 px-1 rounded">D:\training\anime_dataset</code> |
              Linux/Mac: <code className="bg-gray-900 px-1 rounded">/mnt/data/training/dataset</code>
            </p>
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium mb-2">Description (optional)</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief description of this dataset..."
              rows={3}
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500 resize-none"
            />
          </div>

          {/* Options */}
          <div className="space-y-3">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={recursive}
                onChange={(e) => setRecursive(e.target.checked)}
                className="rounded bg-gray-900 border-gray-700 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
              />
              <span className="text-sm">Scan subdirectories recursively</span>
            </label>

            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={readExif}
                onChange={(e) => setReadExif(e.target.checked)}
                className="rounded bg-gray-900 border-gray-700 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
              />
              <span className="text-sm">Read EXIF metadata from images</span>
            </label>
          </div>

          {/* Info Box */}
          <div className="bg-blue-900/20 border border-blue-700 rounded p-3">
            <p className="text-sm text-blue-300 mb-2">
              <strong>Supported formats:</strong> PNG, JPG, JPEG, WebP
            </p>
            <p className="text-sm text-blue-300 mb-2">
              <strong>Caption files:</strong> .txt files with the same base name as images
            </p>
            <p className="text-sm text-blue-300">
              <strong>Image pairs:</strong> Use suffixes like <code className="bg-blue-900/50 px-1 rounded">_source</code>, <code className="bg-blue-900/50 px-1 rounded">_target</code>, <code className="bg-blue-900/50 px-1 rounded">_cref</code>
            </p>
          </div>

          {/* Buttons */}
          <div className="flex justify-end space-x-3 pt-4 border-t border-gray-700">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={loading}
            >
              {loading ? "Creating & Scanning..." : "Create Dataset"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
