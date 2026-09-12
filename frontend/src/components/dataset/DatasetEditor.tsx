"use client";

import { useState, useEffect } from "react";
import { Copy, ExternalLink, FolderOpen, Scan, Save } from "lucide-react";
import { getDataset, getDatasetHealth, launchDatasetEditor, openDatasetFolder, openDatasetWorkspace, scanDataset, updateCaptionProcessing, updateDatasetExifConfig, BrowserWorkspaceResponse, CaptionProcessingConfig, DatasetHealth, ScanFieldSummary } from "@/utils/api";
import DatasetViewer from "./DatasetViewer";
import DatasetBrowserPanel from "../tagger/DatasetBrowserPanel";
import LatentCacheRow from "./LatentCacheRow";
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
  const [scanSummary, setScanSummary] = useState<ScanFieldSummary | null>(null);
  const [activeTab, setActiveTab] = useState<"viewer" | "workspace" | "caption-processing">("viewer");
  const [captionConfig, setCaptionConfig] = useState<CaptionProcessingConfig>({});
  const [savingConfig, setSavingConfig] = useState(false);
  const [health, setHealth] = useState<DatasetHealth | null>(null);
  const [checkingHealth, setCheckingHealth] = useState(false);
  const [workspace, setWorkspace] = useState<BrowserWorkspaceResponse | null>(null);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);

  useEffect(() => {
    loadDataset();
    setWorkspace(null);
    setWorkspaceError(null);
  }, [datasetId]);

  useEffect(() => {
    if (activeTab !== "workspace" || workspace) return;
    openDatasetWorkspace(datasetId)
      .then(setWorkspace)
      .catch((error) => setWorkspaceError(String(error)));
  }, [activeTab, datasetId, workspace]);

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

  const handleScan = async (incremental: boolean) => {
    setScanning(true);
    setScanProgress(0);
    setScanMessage(incremental ? "Reconciling external changes..." : "Starting scan...");

    // Ensure WebSocket is connected
    wsClient.connect();

    try {
      const result = await scanDataset(datasetId, incremental);
      setDataset(result.dataset);
      setScanProgress(100);
      setScanSummary(result.field_summary || null);
      setScanMessage(
        `${incremental ? "Reconcile" : "Scan"} complete: ${result.items_found} new image(s), ${result.captions_updated ?? 0} updated caption(s)`
      );
      setTimeout(() => {
        setScanMessage(null);
        setScanProgress(0);
      }, 8000);
    } catch (err) {
      console.error("Failed to scan dataset:", err);
      setScanMessage("Scan failed. Please check console for details.");
      setScanProgress(0);
    } finally {
      setScanning(false);
    }
  };

  const handleHealthCheck = async () => {
    setCheckingHealth(true);
    try {
      setHealth(await getDatasetHealth(datasetId));
    } catch (err) {
      console.error("Failed to inspect dataset health:", err);
      setScanMessage("Dataset health check failed");
    } finally {
      setCheckingHealth(false);
    }
  };

  const handleOpenFolder = async () => {
    try {
      await openDatasetFolder(datasetId);
    } catch (err) {
      console.error("Failed to open dataset folder:", err);
      setScanMessage("Could not open the dataset folder");
    }
  };

  const handleLaunchEditor = async () => {
    try {
      await launchDatasetEditor(datasetId);
    } catch (err) {
      console.error("Failed to launch dataset editor:", err);
      setScanMessage("Configure a valid dataset editor in Settings first");
    }
  };

  const handleCopyPath = async () => {
    try {
      await navigator.clipboard.writeText(dataset.path);
      setScanMessage("Dataset path copied");
    } catch (err) {
      console.error("Failed to copy dataset path:", err);
      setScanMessage("Could not copy the dataset path");
    }
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
            <button
              onClick={() => void handleCopyPath()}
              className="rounded bg-gray-700 p-1.5 hover:bg-gray-600"
              title="Copy dataset path"
            >
              <Copy className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => void handleOpenFolder()}
              className="rounded bg-gray-700 p-1.5 hover:bg-gray-600"
              title="Open dataset folder"
            >
              <FolderOpen className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => void handleLaunchEditor()}
              className="rounded bg-gray-700 p-1.5 hover:bg-gray-600"
              title="Open in configured dataset editor"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </button>
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
                onClick={handleHealthCheck}
                disabled={checkingHealth || scanning}
                className="px-2.5 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors disabled:opacity-50"
              >
                {checkingHealth ? "Checking..." : "Health"}
              </button>
            )}
            {activeTab === "viewer" && (
              <button
                onClick={() => void handleScan(true)}
                disabled={scanning}
                className="px-2.5 py-1.5 bg-cyan-700 hover:bg-cyan-600 rounded text-xs transition-colors disabled:opacity-50"
                title="Import sidecar changes made by an external editor"
              >
                Reconcile
              </button>
            )}
            {activeTab === "viewer" && (
              <button
                onClick={() => void handleScan(false)}
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
            onClick={() => setActiveTab("workspace")}
            className={`px-3 py-1.5 text-xs font-medium transition-colors ${
              activeTab === "workspace"
                ? "text-blue-400 border-b-2 border-blue-400"
                : "text-gray-400 hover:text-gray-300"
            }`}
          >
            Tag Workspace
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

      {/* Per-field scan summary */}
      {!scanning && scanSummary && (
        <div className="mx-4 mt-2 bg-gray-900 rounded p-3 border border-gray-700 text-xs space-y-1">
          <div className="text-gray-300 font-medium">
            Scan summary · {scanSummary.total_images} images
          </div>
          {(["tags", "caption"] as const).map((k) => (
            <div key={k} className="flex flex-wrap gap-x-2 text-gray-400">
              <span className="text-gray-200 capitalize w-16 shrink-0">{k}</span>
              <span>{scanSummary[k].updated} updated</span>
              <span>· {scanSummary[k].added} new</span>
              <span>
                ·{" "}
                <span className="text-gray-200">{scanSummary[k].images_with ?? 0}</span>
                /{scanSummary.total_images} images have {k}
              </span>
            </div>
          ))}
          <div className="text-gray-500">
            Other fields: {scanSummary.other.updated} updated · {scanSummary.other.added} new
          </div>
        </div>
      )}

      {!scanning && health && (
        <div className={`mx-4 mt-2 rounded border p-3 text-xs ${health.healthy ? "border-green-700 bg-green-950/20" : "border-yellow-700 bg-yellow-950/20"}`}>
          <div className="mb-1 font-medium">
            {health.healthy ? "Dataset is consistent" : "Dataset needs attention"}
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-gray-400">
            {Object.entries(health.counts)
              .filter(([key, value]) => key !== "total_items" && value > 0)
              .map(([key, value]) => (
                <span key={key} title={(health.samples[key] ?? []).join("\n")}>
                  {key.replaceAll("_", " ")}: {value.toLocaleString()}
                </span>
              ))}
            <span>items: {(health.counts.total_items ?? 0).toLocaleString()}</span>
          </div>
        </div>
      )}

      <LatentCacheRow datasetId={datasetId} />

      {/* Content */}
      <div className="flex-1 px-2 py-2 lg:px-4 lg:py-3 overflow-auto lg:overflow-hidden">
        {activeTab === "viewer" && (
          <DatasetViewer datasetId={datasetId} />
        )}
        {activeTab === "workspace" && (
          workspace ? (
            <DatasetBrowserPanel
              modelLoaded={false}
              initialWorkspace={workspace}
              initialRecursive={!!dataset?.recursive}
              lockedWorkspace
            />
          ) : (
            <div className="p-4 text-sm text-gray-400">
              {workspaceError || "Opening dataset workspace..."}
            </div>
          )
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
