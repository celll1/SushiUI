"use client";

import { useState, useEffect } from "react";
import { X, Scan, Save } from "lucide-react";
import { getDataset, scanDataset, updateCaptionProcessing, updateDatasetExifConfig, CaptionProcessingConfig } from "@/utils/api";
import DatasetViewer from "./DatasetViewer";
import CaptionProcessingSettings from "../datasets/CaptionProcessingSettings";
import { wsClient } from "@/utils/websocket";

interface DatasetEditorProps {
  datasetId: number;
  onClose: () => void;
}

export default function DatasetEditor({ datasetId, onClose }: DatasetEditorProps) {
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState<number>(0);
  const [dataset, setDataset] = useState<any>(null);
  const [scanMessage, setScanMessage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"viewer" | "caption-processing">("viewer");
  const [captionConfig, setCaptionConfig] = useState<CaptionProcessingConfig>({});
  const [savingConfig, setSavingConfig] = useState(false);

  useEffect(() => {
    loadDataset();
  }, [datasetId]);

  // WebSocket progress handler for scanning
  useEffect(() => {
    const handleProgress = (step: number, totalSteps: number, message: string) => {
      if (scanning) {
        const progress = totalSteps > 0 ? (step / totalSteps) * 100 : 0;
        setScanProgress(progress);
        setScanMessage(message || "Scanning...");
      }
    };

    wsClient.subscribe(handleProgress);
    return () => wsClient.unsubscribe(handleProgress);
  }, [scanning]);

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
    setScanProgress(0);
    setScanMessage("Starting scan...");

    // Ensure WebSocket is connected
    wsClient.connect();

    try {
      const result = await scanDataset(datasetId);
      setDataset(result.dataset);
      setScanProgress(100);
      setScanMessage(`Scan complete: ${result.items_found} items, ${result.captions_found} captions found`);
      setTimeout(() => {
        setScanMessage(null);
        setScanProgress(0);
      }, 5000);
    } catch (err) {
      console.error("Failed to scan dataset:", err);
      setScanMessage("Scan failed. Please check console for details.");
      setScanProgress(0);
    } finally {
      setScanning(false);
    }
  };

  const handleSave = async () => {
    console.log("Saving dataset:", dataset);
    // TODO: Implement save
  };

  const [savingExif, setSavingExif] = useState(false);
  const handleToggleReadExif = async () => {
    if (!dataset) return;
    setSavingExif(true);
    try {
      const updated = await updateDatasetExifConfig(datasetId, { read_exif: !dataset.read_exif });
      setDataset(updated);
    } catch (err) {
      console.error("Failed to update read_exif:", err);
    } finally {
      setSavingExif(false);
    }
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
          <div className="flex items-center space-x-2">
            {activeTab === "viewer" && (
              <label
                className="flex items-center space-x-1.5 text-xs text-gray-300 cursor-pointer select-none"
                title="Read caption fields embedded in image EXIF metadata on the next scan"
              >
                <input
                  type="checkbox"
                  checked={!!dataset.read_exif}
                  disabled={savingExif || scanning}
                  onChange={handleToggleReadExif}
                  className="cursor-pointer disabled:opacity-50"
                />
                <span>Read EXIF</span>
              </label>
            )}
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

      {/* Scan Progress */}
      {scanning && (
        <div className="mx-4 mt-3 bg-gray-900 rounded-lg p-3 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Scanning Dataset...</span>
            <span className="text-sm text-gray-400">{Math.round(scanProgress)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${scanProgress}%` }}
            />
          </div>
          <p className="text-xs text-gray-400">
            {scanMessage || "Scanning..."}
          </p>
        </div>
      )}

      {/* Scan Message (completion/error) */}
      {!scanning && scanMessage && (
        <div className={`mx-4 mt-3 rounded p-2 text-xs ${
          scanMessage.includes("complete") || scanMessage.includes("success")
            ? "bg-green-900/20 border border-green-500 text-green-400"
            : "bg-red-900/20 border border-red-500 text-red-400"
        }`}>
          {scanMessage}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 px-2 py-2 lg:px-4 lg:py-3 overflow-auto lg:overflow-hidden">
        {activeTab === "viewer" && (
          <DatasetViewer datasetId={datasetId} />
        )}
        {activeTab === "caption-processing" && (
          <div className="h-full overflow-y-auto">
            <div className="max-w-2xl mx-auto">
              <CaptionProcessingSettings
                config={captionConfig}
                onChange={setCaptionConfig}
                datasetId={datasetId}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
