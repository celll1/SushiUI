"use client";

import { useState, useEffect } from "react";
import { X, Scan, Save } from "lucide-react";
import { getDataset, scanDataset, Dataset } from "@/utils/api";
import DatasetViewer from "./DatasetViewer";

interface DatasetEditorProps {
  datasetId: number;
  onClose: () => void;
}

export default function DatasetEditor({ datasetId, onClose }: DatasetEditorProps) {
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [dataset, setDataset] = useState<any>(null);
  const [scanMessage, setScanMessage] = useState<string | null>(null);

  useEffect(() => {
    loadDataset();
  }, [datasetId]);

  const loadDataset = async () => {
    setLoading(true);
    try {
      const data = await getDataset(datasetId);
      setDataset(data);
    } catch (err) {
      console.error("Failed to load dataset:", err);
    } finally {
      setLoading(false);
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
      <div className="flex-shrink-0 flex items-center justify-between px-4 py-3 border-b border-gray-700 bg-gray-800/50">
        <div className="flex items-center space-x-3">
          <h2 className="text-base font-semibold">{dataset.name}</h2>
          <span className="text-xs text-gray-400">{dataset.total_items} items</span>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={handleScan}
            disabled={scanning}
            className="px-2.5 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-xs flex items-center space-x-1 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Scan className="h-3.5 w-3.5" />
            <span>{scanning ? "Scanning..." : "Scan"}</span>
          </button>
        </div>
      </div>

      {/* Scan Message */}
      {scanMessage && (
        <div className="mx-4 mt-3 bg-green-900/20 border border-green-500 text-green-400 rounded p-2 text-xs">
          {scanMessage}
        </div>
      )}

      {/* Content - 3 Column Viewer */}
      <div className="flex-1 px-4 py-3 overflow-hidden">
        <DatasetViewer datasetId={datasetId} />
      </div>
    </div>
  );
}
