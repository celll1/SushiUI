"use client";

import { useState, useEffect } from "react";
import Sidebar from "@/components/common/Sidebar";
import ProtectedRoute from "@/components/common/ProtectedRoute";
import DatasetList from "@/components/dataset/DatasetList";
import DatasetEditor from "@/components/dataset/DatasetEditor";
import TagDictionaryManager from "@/components/dataset/TagDictionaryManager";

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

  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto p-3 sm:p-6 pt-16 lg:pt-6">
        <h1 className="text-xl sm:text-2xl font-bold mb-4 sm:mb-6">Dataset Management</h1>

        {/* Tabs */}
        <div className="flex space-x-1 sm:space-x-2 border-b border-gray-700 mb-4 sm:mb-6 overflow-x-auto">
          <button
            onClick={() => {
              setActiveTab("datasets");
              setSelectedDatasetId(null);
            }}
            className={`px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium transition-colors whitespace-nowrap ${
              activeTab === "datasets"
                ? "border-b-2 border-blue-500 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            Datasets
          </button>
          <button
            onClick={() => {
              setActiveTab("tags");
              setSelectedDatasetId(null);
            }}
            className={`px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium transition-colors whitespace-nowrap ${
              activeTab === "tags"
                ? "border-b-2 border-blue-500 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            Tag Dictionary
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === "datasets" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <div className="lg:col-span-4">
              <DatasetList
                selectedDatasetId={selectedDatasetId}
                onSelectDataset={setSelectedDatasetId}
              />
            </div>
            <div className="lg:col-span-8">
              {selectedDatasetId ? (
                <DatasetEditor
                  datasetId={selectedDatasetId}
                  onClose={() => setSelectedDatasetId(null)}
                />
              ) : (
                <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-400">
                  Select a dataset from the list or create a new one
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "tags" && <TagDictionaryManager />}
      </main>
    </div>
  );
}
