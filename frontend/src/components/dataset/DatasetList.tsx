"use client";

import { useState, useEffect, useRef } from "react";
import { Plus, Folder, RefreshCw, FolderPlus } from "lucide-react";
import CreateDatasetModal from "./CreateDatasetModal";

interface Dataset {
  id: number;
  name: string;
  path: string;
  total_items: number;
  total_captions: number;
  total_tags: number;
  created_at: string;
  last_scanned_at: string | null;
}

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
      // TODO: Implement API call
      // const response = await api.get("/datasets");
      // setDatasets(response.data.datasets);
      setDatasets([]); // Placeholder
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
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Datasets</h2>
          <div className="flex space-x-2">
            <button
              onClick={loadDatasets}
              className="p-2 rounded bg-gray-700 hover:bg-gray-600 transition-colors"
              title="Refresh"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
            <button
              onClick={() => handleCreateDataset()}
              className="p-2 rounded bg-blue-600 hover:bg-blue-500 transition-colors"
              title="Create Dataset"
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-900/20 border border-red-500 text-red-400 rounded p-3 mb-4">
            {error}
          </div>
        )}

        {loading && (
          <div className="text-center text-gray-400 py-8">Loading datasets...</div>
        )}

        {!loading && datasets.length === 0 && (
          <div className="text-center text-gray-400 py-8">
            <FolderPlus className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p className="font-medium mb-1">No datasets yet</p>
            <p className="text-sm">Click the + button to create a dataset</p>
          </div>
        )}

        {!loading && datasets.length > 0 && (
          <div className="space-y-2">
            {datasets.map((dataset) => (
              <button
                key={dataset.id}
                onClick={() => onSelectDataset(dataset.id)}
                className={`w-full text-left p-3 rounded transition-colors ${
                  selectedDatasetId === dataset.id
                    ? "bg-blue-600 text-white"
                    : "bg-gray-700 hover:bg-gray-600 text-gray-100"
                }`}
              >
                <div className="flex items-center space-x-2 mb-1">
                  <Folder className="h-4 w-4" />
                  <span className="font-medium">{dataset.name}</span>
                </div>
                <div className="text-xs text-gray-300 space-y-0.5">
                  <p className="truncate">{dataset.path}</p>
                  <p>
                    {dataset.total_items} items • {dataset.total_captions} captions •{" "}
                    {dataset.total_tags} tags
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
