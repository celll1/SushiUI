"use client";

import { useState, useEffect, useRef } from "react";
import { Plus, Folder, RefreshCw, FolderPlus } from "lucide-react";
import CreateDatasetModal from "./CreateDatasetModal";
import { listDatasets, Dataset } from "@/utils/api";

interface DatasetListProps {
  selectedDatasetId: number | null;
  onSelectDataset: (id: number) => void;
}

export default function DatasetList({ selectedDatasetId, onSelectDataset }: DatasetListProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [initialFolderPath, setInitialFolderPath] = useState<string | null>(null);
  const loadDatasets = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listDatasets();
      setDatasets(response.datasets);
    } catch (err) {
      setError("Failed to load datasets");
      console.error("Failed to load datasets:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  const handleCreateDataset = () => {
    setShowCreateModal(true);
  };

  const handleDatasetCreated = (newDataset: Dataset) => {
    setDatasets([...datasets, newDataset]);
    setShowCreateModal(false);
    setInitialFolderPath(null);
    onSelectDataset(newDataset.id);
  };

  return (
    <>
      <div className="bg-gray-800 rounded-lg p-3">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold">Datasets</h2>
          <div className="flex space-x-1.5">
            <button
              onClick={loadDatasets}
              className="p-1.5 rounded bg-gray-700 hover:bg-gray-600 transition-colors"
              title="Refresh"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => handleCreateDataset()}
              className="p-1.5 rounded bg-blue-600 hover:bg-blue-500 transition-colors"
              title="Create Dataset"
            >
              <Plus className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-900/20 border border-red-500 text-red-400 rounded p-2 mb-2 text-xs">
            {error}
          </div>
        )}

        {loading && (
          <div className="text-center text-gray-400 py-4 text-xs">Loading datasets...</div>
        )}

        {!loading && datasets.length === 0 && (
          <div className="text-center text-gray-400 py-4">
            <FolderPlus className="h-8 w-8 mx-auto mb-1 opacity-50" />
            <p className="text-xs font-medium mb-0.5">No datasets yet</p>
            <p className="text-[10px]">Click + to create</p>
          </div>
        )}

        {!loading && datasets.length > 0 && (
          <div className="space-y-1.5">
            {datasets.map((dataset) => (
              <button
                key={dataset.id}
                onClick={() => onSelectDataset(dataset.id)}
                className={`w-full text-left p-2 rounded transition-colors ${
                  selectedDatasetId === dataset.id
                    ? "bg-blue-600 text-white"
                    : "bg-gray-700 hover:bg-gray-600 text-gray-100"
                }`}
              >
                <div className="flex items-center space-x-1.5 mb-0.5">
                  <Folder className="h-3.5 w-3.5 flex-shrink-0" />
                  <span className="text-xs font-medium truncate">{dataset.name}</span>
                </div>
                <div className="text-[10px] text-gray-300 space-y-0.5 ml-5">
                  <p className="truncate">{dataset.path}</p>
                  <p>
                    {dataset.total_items} items • {dataset.total_captions} captions
                  </p>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {showCreateModal && (
        <CreateDatasetModal
          initialPath={initialFolderPath}
          onClose={() => {
            setShowCreateModal(false);
            setInitialFolderPath(null);
          }}
          onCreate={handleDatasetCreated}
        />
      )}
    </>
  );
}
