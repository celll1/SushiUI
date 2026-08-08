"use client";

import { useState, useEffect } from "react";
import { Plus, Folder, RefreshCw, FolderPlus, Trash2 } from "lucide-react";
import CreateDatasetModal from "./CreateDatasetModal";
import { listDatasets, Dataset, deleteDataset } from "@/utils/api";

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
  const totalItems = datasets.reduce((sum, dataset) => sum + dataset.total_items, 0);
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

  const handleDeleteDataset = async (datasetId: number, datasetName: string) => {
    if (!confirm(`Are you sure you want to delete dataset "${datasetName}"?\n\nThis will remove all dataset items and captions from the database.`)) {
      return;
    }

    try {
      await deleteDataset(datasetId);
      setDatasets(datasets.filter(d => d.id !== datasetId));
      if (selectedDatasetId === datasetId) {
        onSelectDataset(datasets.length > 1 ? datasets[0].id : 0);
      }
    } catch (err) {
      console.error("Failed to delete dataset:", err);
      setError("Failed to delete dataset");
    }
  };

  return (
    <>
      <div className="rounded-md border border-gray-800 bg-gray-900 p-3">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold">Datasets</h2>
            {!loading && datasets.length > 0 && (
              <p className="mt-0.5 text-[10px] text-gray-500">
                {datasets.length} collections · {totalItems.toLocaleString()} items
              </p>
            )}
          </div>
          <div className="flex space-x-1.5">
            <button
              onClick={loadDatasets}
              className="rounded-md border border-gray-700 bg-gray-800 p-1.5 transition-colors hover:bg-gray-700"
              title="Refresh"
              aria-label="Refresh datasets"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => handleCreateDataset()}
              className="rounded-md border border-violet-400/30 bg-violet-600 p-1.5 transition-colors hover:bg-violet-500"
              title="Create Dataset"
              aria-label="Create dataset"
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
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {datasets.map((dataset) => {
              const taggedPercent = dataset.total_items > 0
                ? Math.round((dataset.total_tags / dataset.total_items) * 100)
                : 0;
              const captionedPercent = dataset.total_items > 0
                ? Math.round((dataset.total_captions / dataset.total_items) * 100)
                : 0;

              return (
              <div
                key={dataset.id}
                className={`group relative min-w-0 rounded-md border transition-colors ${
                  selectedDatasetId === dataset.id
                    ? "border-violet-400/60 bg-violet-500/15"
                    : "border-gray-700 bg-gray-800/80 hover:border-gray-600 hover:bg-gray-800"
                }`}
              >
                <button
                  onClick={() => onSelectDataset(dataset.id)}
                  className="w-full p-2.5 text-left text-gray-100"
                >
                  <div className="mb-2 flex items-center space-x-1.5">
                    <span className="grid h-6 w-6 flex-shrink-0 place-items-center rounded bg-gray-900 text-violet-300">
                      <Folder className="h-3.5 w-3.5" />
                    </span>
                    <span className="truncate pr-8 text-xs font-semibold">{dataset.name}</span>
                  </div>
                  <p className="mb-2 truncate font-mono text-[9px] text-gray-500" title={dataset.path}>{dataset.path}</p>
                  <div className="grid grid-cols-3 gap-1.5">
                    <div className="rounded bg-gray-900/80 px-2 py-1.5">
                      <p className="text-[9px] uppercase tracking-wide text-gray-500">Items</p>
                      <p className="mt-0.5 text-xs font-medium text-gray-200">{dataset.total_items.toLocaleString()}</p>
                    </div>
                    <div className="rounded bg-gray-900/80 px-2 py-1.5">
                      <p className="text-[9px] uppercase tracking-wide text-gray-500">Tagged</p>
                      <p className="mt-0.5 text-xs font-medium text-gray-200">{dataset.total_tags.toLocaleString()}</p>
                      <p className="text-[9px] text-gray-500">{taggedPercent}%</p>
                    </div>
                    <div className="rounded bg-gray-900/80 px-2 py-1.5">
                      <p className="text-[9px] uppercase tracking-wide text-gray-500">Captioned</p>
                      <p className="mt-0.5 text-xs font-medium text-gray-200">{dataset.total_captions.toLocaleString()}</p>
                      <p className="text-[9px] text-gray-500">{captionedPercent}%</p>
                    </div>
                  </div>
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteDataset(dataset.id, dataset.name);
                  }}
                  className="absolute right-2 top-2 rounded bg-red-600/80 p-1 opacity-0 transition-opacity hover:bg-red-500 focus:opacity-100 group-hover:opacity-100"
                  title="Delete dataset"
                  aria-label={`Delete ${dataset.name}`}
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
              );
            })}
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
