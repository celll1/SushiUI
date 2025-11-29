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
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden pt-16 lg:pt-0">
        {/* Header */}
        <div className="flex-shrink-0 p-3 sm:p-4 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <h1 className="text-lg sm:text-xl font-bold">Dataset Management</h1>

            {/* Dataset Tabs (when dataset is selected) */}
            {selectedDatasetId && selectedDataset && (
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setShowDatasetSelector(!showDatasetSelector)}
                  className="flex items-center space-x-2 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-sm transition-colors"
                >
                  <span className="font-medium">{selectedDataset.name}</span>
                  <ChevronDown className="h-4 w-4" />
                </button>
                <button
                  onClick={handleCloseDataset}
                  className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
                >
                  Close
                </button>
              </div>
            )}
          </div>

          {/* Main Tabs */}
          {!selectedDatasetId && (
            <div className="flex space-x-1 sm:space-x-2 border-b border-gray-700 mt-3 overflow-x-auto">
              <button
                onClick={() => setActiveTab("datasets")}
                className={`px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium transition-colors whitespace-nowrap ${
                  activeTab === "datasets"
                    ? "border-b-2 border-blue-500 text-white"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                Datasets
              </button>
              <button
                onClick={() => setActiveTab("tags")}
                className={`px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium transition-colors whitespace-nowrap ${
                  activeTab === "tags"
                    ? "border-b-2 border-blue-500 text-white"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                Tag Dictionary
              </button>
            </div>
          )}

          {/* Dataset Selector Dropdown */}
          {showDatasetSelector && (
            <div className="absolute right-4 mt-2 w-80 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
              <div className="p-2">
                {datasets.map((dataset) => (
                  <button
                    key={dataset.id}
                    onClick={() => handleSelectDataset(dataset.id)}
                    className={`w-full text-left p-2 rounded transition-colors mb-1 ${
                      dataset.id === selectedDatasetId
                        ? "bg-blue-600 text-white"
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
            <div className="h-full p-3 sm:p-4 overflow-auto">
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
