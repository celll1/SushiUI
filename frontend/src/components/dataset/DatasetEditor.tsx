"use client";

import { useState, useEffect } from "react";
import { X, Scan, Save, FileText } from "lucide-react";

interface DatasetEditorProps {
  datasetId: number;
  onClose: () => void;
}

export default function DatasetEditor({ datasetId, onClose }: DatasetEditorProps) {
  const [loading, setLoading] = useState(true);
  const [dataset, setDataset] = useState<any>(null);

  useEffect(() => {
    loadDataset();
  }, [datasetId]);

  const loadDataset = async () => {
    setLoading(true);
    try {
      // TODO: Implement API call
      // const response = await api.get(`/datasets/${datasetId}`);
      // setDataset(response.data);
      setDataset({
        id: datasetId,
        name: "Example Dataset",
        path: "/path/to/dataset",
        total_items: 0,
      }); // Placeholder
    } catch (err) {
      console.error("Failed to load dataset:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleScan = async () => {
    console.log("Scanning dataset:", datasetId);
    // TODO: Implement scan
  };

  const handleSave = async () => {
    console.log("Saving dataset:", dataset);
    // TODO: Implement save
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="text-center text-gray-400">Loading dataset...</div>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="text-center text-red-400">Dataset not found</div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold">{dataset.name}</h2>
        <div className="flex space-x-2">
          <button
            onClick={handleScan}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-sm flex items-center space-x-1 transition-colors"
          >
            <Scan className="h-4 w-4" />
            <span>Scan</span>
          </button>
          <button
            onClick={handleSave}
            className="px-3 py-1.5 bg-green-600 hover:bg-green-500 rounded text-sm flex items-center space-x-1 transition-colors"
          >
            <Save className="h-4 w-4" />
            <span>Save</span>
          </button>
          <button
            onClick={onClose}
            className="p-1.5 rounded hover:bg-gray-700 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Dataset Info */}
        <div className="bg-gray-900/50 rounded p-4">
          <h3 className="text-sm font-semibold mb-2">Dataset Information</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="text-gray-400">Path:</div>
            <div className="text-gray-200">{dataset.path}</div>
            <div className="text-gray-400">Items:</div>
            <div className="text-gray-200">{dataset.total_items}</div>
          </div>
        </div>

        {/* Items Browser (Placeholder) */}
        <div className="bg-gray-900/50 rounded p-4">
          <h3 className="text-sm font-semibold mb-2">Dataset Items</h3>
          <div className="text-center text-gray-400 py-8">
            <FileText className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>Item browser coming soon</p>
            <p className="text-xs mt-1">Scan the dataset to populate items</p>
          </div>
        </div>
      </div>
    </div>
  );
}
