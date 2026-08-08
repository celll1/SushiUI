"use client";

import { useState, useEffect } from "react";
import Sidebar from "@/components/common/Sidebar";
import ProtectedRoute from "@/components/common/ProtectedRoute";
import DatasetList from "@/components/dataset/DatasetList";
import DatasetEditor from "@/components/dataset/DatasetEditor";
import TagDictionaryManager from "@/components/dataset/TagDictionaryManager";
import { listDatasets, Dataset } from "@/utils/api";
import { ChevronDown } from "lucide-react";

export default function DatasetPage() {
  return (
    <ProtectedRoute>
      <DatasetPageContent />
    </ProtectedRoute>
  );
}

function DatasetPageContent() {
  const [activeTab, setActiveTab] = useState<"datasets" | "tags">("datasets");
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [showDatasetSelector, setShowDatasetSelector] = useState(false);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await listDatasets();
      setDatasets(response.datasets);
    } catch (err) {
      console.error("Failed to load datasets:", err);
    }
  };

  const handleSelectDataset = (id: number) => {
    setSelectedDatasetId(id);
    setShowDatasetSelector(false);
  };

  const handleCloseDataset = () => {
    setSelectedDatasetId(null);
  };

  const selectedDataset = datasets.find(d => d.id === selectedDatasetId);

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="app-main compact-workspace flex flex-col overflow-hidden">
        {/* Header */}
        <div className="app-topbar relative flex-shrink-0 flex-col items-stretch gap-1 py-1 sm:flex-row sm:items-center sm:py-0">
          <div className="flex items-center justify-between">
            <div className="shrink-0">
              <p className="app-kicker">Assets</p>
              <h1 className="app-title">Dataset</h1>
            </div>

            {/* Dataset Tabs (when dataset is selected) */}
            {selectedDatasetId && selectedDataset && (
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setShowDatasetSelector(!showDatasetSelector)}
                  className="flex h-8 items-center space-x-2 rounded-md border border-gray-700 bg-gray-800 px-2.5 text-xs transition-colors hover:bg-gray-700"
                >
                  <span className="font-medium">{selectedDataset.name}</span>
                  <ChevronDown className="h-4 w-4" />
                </button>
                <button
                  onClick={handleCloseDataset}
                  className="h-8 rounded-md border border-gray-700 bg-gray-800 px-2.5 text-xs transition-colors hover:bg-gray-700"
                >
                  Close
                </button>
              </div>
            )}
          </div>

          {/* Main Tabs */}
          {!selectedDatasetId && (
            <div className="app-tabs border-b-0 sm:ml-5">
              <button
                onClick={() => setActiveTab("datasets")}
                className={`app-tab ${
                  activeTab === "datasets"
                    ? "app-tab-active"
                    : ""
                }`}
              >
                Datasets
              </button>
              <button
                onClick={() => setActiveTab("tags")}
                className={`app-tab ${
                  activeTab === "tags"
                    ? "app-tab-active"
                    : ""
                }`}
              >
                Tag Dictionary
              </button>
            </div>
          )}

          {/* Dataset Selector Dropdown */}
          {showDatasetSelector && (
            <div className="absolute right-4 top-11 z-50 max-h-96 w-72 overflow-y-auto rounded-md border border-gray-700 bg-gray-800 shadow-xl">
              <div className="p-2">
                {datasets.map((dataset) => (
                  <button
                    key={dataset.id}
                    onClick={() => handleSelectDataset(dataset.id)}
                    className={`w-full text-left p-2 rounded transition-colors mb-1 ${
                      dataset.id === selectedDatasetId
                        ? "bg-violet-600 text-white"
                        : "hover:bg-gray-700"
                    }`}
                  >
                    <div className="text-sm font-medium">{dataset.name}</div>
                    <div className="text-xs text-gray-400">{dataset.total_items} items</div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {selectedDatasetId ? (
            <DatasetEditor
              datasetId={selectedDatasetId}
              onClose={handleCloseDataset}
            />
          ) : activeTab === "datasets" ? (
            <div className="h-full overflow-auto p-2.5 sm:p-3">
              <DatasetList
                selectedDatasetId={selectedDatasetId}
                onSelectDataset={handleSelectDataset}
              />
            </div>
          ) : (
            <div className="h-full overflow-auto">
              <TagDictionaryManager />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
