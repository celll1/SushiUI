"use client";

import { useState, useEffect } from "react";
import { X, Scan, Save } from "lucide-react";
import { getDataset, scanDataset, Dataset, updateCaptionProcessing, CaptionProcessingConfig } from "@/utils/api";
import DatasetViewer from "./DatasetViewer";
import CaptionProcessingSettings from "../datasets/CaptionProcessingSettings";

interface DatasetEditorProps {
  datasetId: number;
  onClose: () => void;
}

export default function DatasetEditor({ datasetId, onClose }: DatasetEditorProps) {
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [dataset, setDataset] = useState<any>(null);
  const [scanMessage, setScanMessage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"viewer" | "caption-processing">("viewer");
  const [captionConfig, setCaptionConfig] = useState<CaptionProcessingConfig>({});
  const [savingConfig, setSavingConfig] = useState(false);

  useEffect(() => {
    loadDataset();
  }, [datasetId]);

  const loadDataset = async () => {
    setLoading(true);
    try {
      const data = await getDataset(datasetId);
      setDataset(data);
      setCaptionConfig(data.caption_processing || {});
    } catch (err) {
      console.error("Failed to load dataset:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveCaptionConfig = async () => {
    setSavingConfig(true);
    try {
      const updatedDataset = await updateCaptionProcessing(datasetId, captionConfig);
      setDataset(updatedDataset);
      setScanMessage("Caption processing settings saved successfully");
      setTimeout(() => setScanMessage(null), 3000);
    } catch (err) {
      console.error("Failed to save caption processing config:", err);
      setScanMessage("Failed to save settings");
    } finally {
      setSavingConfig(false);
    }
  };

  const handleScan = async () => {
    setScanning(true);
    setScanMessage(null);
    try {
      const result = await scanDataset(datasetId);
      setDataset(result.dataset);
      setScanMessage(`Scan complete: ${result.items_found} items, ${result.captions_found} captions found`);
      setTimeout(() => setScanMessage(null), 5000);
    } catch (err) {
      console.error("Failed to scan dataset:", err);
      setScanMessage("Scan failed. Please check console for details.");
    } finally {
      setScanning(false);
    }
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
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex-shrink-0 px-4 py-3 border-b border-gray-700 bg-gray-800/50">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            <h2 className="text-base font-semibold">{dataset.name}</h2>
            <span className="text-xs text-gray-400">{dataset.total_items} items</span>
          </div>
          <div className="flex space-x-2">
            {activeTab === "viewer" && (
              <button
                onClick={handleScan}
                disabled={scanning}
                className="px-2.5 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-xs flex items-center space-x-1 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Scan className="h-3.5 w-3.5" />
                <span>{scanning ? "Scanning..." : "Scan"}</span>
              </button>
            )}
            {activeTab === "caption-processing" && (
              <button
                onClick={handleSaveCaptionConfig}
                disabled={savingConfig}
                className="px-2.5 py-1.5 bg-green-600 hover:bg-green-500 rounded text-xs flex items-center space-x-1 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Save className="h-3.5 w-3.5" />
                <span>{savingConfig ? "Saving..." : "Save Settings"}</span>
              </button>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 border-b border-gray-700">
          <button
            onClick={() => setActiveTab("viewer")}
            className={`px-3 py-1.5 text-xs font-medium transition-colors ${
              activeTab === "viewer"
                ? "text-blue-400 border-b-2 border-blue-400"
                : "text-gray-400 hover:text-gray-300"
            }`}
          >
            Viewer
          </button>
          <button
            onClick={() => setActiveTab("caption-processing")}
            className={`px-3 py-1.5 text-xs font-medium transition-colors ${
              activeTab === "caption-processing"
                ? "text-blue-400 border-b-2 border-blue-400"
                : "text-gray-400 hover:text-gray-300"
            }`}
          >
            Caption Processing
          </button>
        </div>
      </div>

      {/* Scan Message */}
      {scanMessage && (
        <div className="mx-4 mt-3 bg-green-900/20 border border-green-500 text-green-400 rounded p-2 text-xs">
          {scanMessage}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 px-4 py-3 overflow-hidden">
        {activeTab === "viewer" && (
          <DatasetViewer datasetId={datasetId} />
        )}
        {activeTab === "caption-processing" && (
          <div className="h-full overflow-y-auto">
            <div className="max-w-2xl mx-auto">
              <CaptionProcessingSettings
                config={captionConfig}
                onChange={setCaptionConfig}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
