"use client";

import { useState, useEffect } from "react";
import { X, Folder, AlertCircle } from "lucide-react";
import { createDataset, scanDataset, scanDatasetPreview, Dataset, StructureDetectionResult, ScanPreviewResult } from "@/utils/api";
import { wsClient } from "@/utils/websocket";

interface CreateDatasetModalProps {
  initialPath: string | null;
  onClose: () => void;
  onCreate: (dataset: Dataset) => void;
}

export default function CreateDatasetModal({ initialPath, onClose, onCreate }: CreateDatasetModalProps) {
  const [name, setName] = useState("");
  const [path, setPath] = useState(initialPath || "");
  const [description, setDescription] = useState("");
  const [recursive, setRecursive] = useState(true);
  const [readExif, setReadExif] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scanningProgress, setScanningProgress] = useState<number | null>(null);
  const [scanningMessage, setScanningMessage] = useState<string>("");
  const [detectionResult, setDetectionResult] = useState<StructureDetectionResult | null>(null);
  const [scanPreview, setScanPreview] = useState<ScanPreviewResult | null>(null);

  // WebSocket progress handler
  useEffect(() => {
    const handleProgress = (step: number, totalSteps: number, message: string) => {
      if (scanningProgress !== null) {
        // Only update if we're currently scanning
        const progress = totalSteps > 0 ? (step / totalSteps) * 100 : 0;
        setScanningProgress(progress);
        setScanningMessage(message || "Scanning dataset...");
      }
    };

    wsClient.subscribe(handleProgress);
    return () => wsClient.unsubscribe(handleProgress);
  }, [scanningProgress]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validation
    if (!path.trim()) {
      setError("Dataset path is required");
      return;
    }

    // Auto-generate name from folder if not provided
    let datasetName = name.trim();
    if (!datasetName) {
      const pathParts = path.trim().replace(/\\/g, "/").split("/");
      datasetName = pathParts[pathParts.length - 1] || "unnamed_dataset";
    }

    setLoading(true);
    setError(null);

    try {
      const newDataset = await createDataset({
        name: datasetName,
        path: path.trim(),
        description: description.trim() || undefined,
        recursive,
        read_exif: readExif,
      });

      // Automatically scan the dataset after creation
      try {
        setScanningProgress(0);
        setScanningMessage("Starting scan...");

        // Ensure WebSocket is connected
        wsClient.connect();

        // Start scanning (this will send progress via WebSocket)
        const scanResult = await scanDataset(newDataset.id);

        // Scan complete
        setScanningProgress(100);
        setScanningMessage("Scan complete!");

        // Fetch scan preview to show detected structure
        try {
          const preview = await scanDatasetPreview(newDataset.id);
          setScanPreview(preview);
        } catch (previewErr) {
          console.error("Failed to get scan preview:", previewErr);
        }

        // Show detection result if paired structure was found
        if (scanResult.structure_detection?.structure_type === "paired") {
          setDetectionResult(scanResult.structure_detection);
          // Longer delay to let user see the detection result
          await new Promise(resolve => setTimeout(resolve, 3000));
        } else {
          // Small delay to show completion
          await new Promise(resolve => setTimeout(resolve, 1500));
        }

        onCreate(scanResult.dataset); // Return scanned dataset with updated counts
      } catch (scanErr) {
        console.error("Failed to scan dataset:", scanErr);
        setScanningProgress(null);
        setScanningMessage("");
        onCreate(newDataset); // Return dataset even if scan fails
      }
    } catch (err: any) {
      setError(err.response?.data?.error || "Failed to create dataset");
      console.error("Failed to create dataset:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold">Create New Dataset</h2>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-gray-700 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {error && (
            <div className="bg-red-900/20 border border-red-500 text-red-400 rounded p-3 flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* Dataset Name */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Dataset Name (optional)
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., character_training_dataset (auto-filled from folder name if empty)"
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
            />
            <p className="text-xs text-gray-400 mt-1">Unique identifier for this dataset (uses folder name if left empty)</p>
          </div>

          {/* Dataset Path */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Dataset Directory Path <span className="text-red-400">*</span>
            </label>
            <div className="relative">
              <Folder className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="e.g., /path/to/your/dataset"
                className="w-full pl-10 pr-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500 font-mono"
                required
              />
            </div>
            <p className="text-xs text-gray-400 mt-1">
              Full path to the directory containing your training images
            </p>
            <p className="text-xs text-blue-400 mt-1">
              Windows: <code className="bg-gray-900 px-1 rounded">D:\training\anime_dataset</code> |
              Linux/Mac: <code className="bg-gray-900 px-1 rounded">/mnt/data/training/dataset</code>
            </p>
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium mb-2">Description (optional)</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief description of this dataset..."
              rows={3}
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500 resize-none"
            />
          </div>

          {/* Options */}
          <div className="space-y-3">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={recursive}
                onChange={(e) => setRecursive(e.target.checked)}
                className="rounded bg-gray-900 border-gray-700 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
              />
              <span className="text-sm">Scan subdirectories recursively</span>
            </label>

            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={readExif}
                onChange={(e) => setReadExif(e.target.checked)}
                className="rounded bg-gray-900 border-gray-700 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
              />
              <span className="text-sm">Read EXIF metadata from images</span>
            </label>
          </div>

          {/* Info Box */}
          <div className="bg-blue-900/20 border border-blue-700 rounded p-3">
            <p className="text-sm text-blue-300 mb-2">
              <strong>Supported formats:</strong> PNG, JPG, JPEG, WebP
            </p>
            <p className="text-sm text-blue-300 mb-2">
              <strong>Caption files:</strong> .txt files with the same base name as images
            </p>
            <p className="text-sm text-blue-300">
              <strong>Image pairs:</strong> Use suffixes like <code className="bg-blue-900/50 px-1 rounded">_source</code>, <code className="bg-blue-900/50 px-1 rounded">_target</code>, <code className="bg-blue-900/50 px-1 rounded">_cref</code>
            </p>
          </div>

          {/* Scanning Progress */}
          {scanningProgress !== null && (
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Scanning Dataset...</span>
                <span className="text-sm text-gray-400">{Math.round(scanningProgress)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${scanningProgress}%` }}
                />
              </div>
              <p className="text-xs text-gray-400 mt-2">
                {scanningMessage || "Scanning images and captions in the dataset directory..."}
              </p>
            </div>
          )}

          {/* Structure Detection Result */}
          {detectionResult && detectionResult.structure_type === "paired" && (
            <div className="bg-green-900/20 border border-green-700 rounded p-3">
              <p className="text-sm text-green-300 font-medium mb-2">
                Paired dataset structure detected (confidence: {Math.round(detectionResult.confidence * 100)}%)
              </p>
              <div className="text-xs text-green-400 space-y-1">
                <p>
                  <span className="text-gray-400">Target suffixes:</span>{" "}
                  {detectionResult.target_suffixes.map(s => (
                    <code key={s} className="bg-green-900/50 px-1 rounded mr-1">{s}</code>
                  ))}
                </p>
                <p>
                  <span className="text-gray-400">Reference suffixes:</span>{" "}
                  {detectionResult.reference_suffixes.map(s => (
                    <code key={s} className="bg-green-900/50 px-1 rounded mr-1">{s}</code>
                  ))}
                </p>
                {detectionResult.caption_suffixes_for_reference.length > 0 && (
                  <p>
                    <span className="text-gray-400">Caption suffixes:</span>{" "}
                    {detectionResult.caption_suffixes_for_reference.map(s => (
                      <code key={s} className="bg-green-900/50 px-1 rounded mr-1">{s}</code>
                    ))}
                  </p>
                )}
                <p className="text-gray-500 mt-1">
                  {detectionResult.stats.paired_groups} paired groups found from {detectionResult.stats.total_files_sampled} files
                </p>
              </div>
            </div>
          )}

          {/* Scan Preview - Caption Suffixes */}
          {scanPreview && Object.keys(scanPreview.detected_suffixes).length > 0 && (
            <div className="bg-blue-900/20 border border-blue-700 rounded p-3">
              <p className="text-sm text-blue-300 font-medium mb-2">
                Detected Caption Types ({scanPreview.total_images} images, {scanPreview.total_captions} captions)
              </p>
              <div className="space-y-1">
                {Object.entries(scanPreview.detected_suffixes).map(([suffix, info]) => (
                  <div key={suffix} className="flex items-center gap-2 text-xs">
                    <code className="bg-blue-900/50 px-1.5 py-0.5 rounded text-blue-300">
                      {suffix === "(default)" ? "(default .txt)" : `_${suffix}.txt`}
                    </code>
                    <span className="text-gray-400">{info.count} files</span>
                    <span className={`px-1.5 py-0.5 rounded ${
                      info.sample_types.includes("tags") ? "bg-green-900/50 text-green-300" :
                      info.sample_types.includes("natural_language") ? "bg-purple-900/50 text-purple-300" :
                      "bg-gray-700 text-gray-300"
                    }`}>
                      {info.sample_types.join(", ")}
                    </span>
                  </div>
                ))}
              </div>
              {scanPreview.sample_groups.length > 0 && (
                <div className="mt-3 border-t border-blue-800 pt-2">
                  <p className="text-xs text-gray-400 mb-1">Sample group: {scanPreview.sample_groups[0].group_name}</p>
                  <div className="text-xs text-gray-500 space-y-0.5">
                    {scanPreview.sample_groups[0].images.map((img, i) => (
                      <div key={i}>
                        <span className="text-gray-400">{img.role}:</span>{" "}
                        {img.path.split(/[/\\]/).pop()}
                      </div>
                    ))}
                    {scanPreview.sample_groups[0].captions.map((cap, i) => (
                      <div key={i}>
                        <span className={cap.detected_type === "tags" ? "text-green-400" : "text-purple-400"}>
                          {cap.suffix || "default"}:
                        </span>{" "}
                        {cap.path.split(/[/\\]/).pop()}
                        {cap.content_preview && (
                          <span className="text-gray-600 ml-1 truncate inline-block max-w-[200px] align-bottom">
                            {cap.content_preview.substring(0, 60)}...
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Buttons */}
          <div className="flex justify-end space-x-3 pt-4 border-t border-gray-700">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={loading}
            >
              {loading ? (scanningProgress !== null ? "Scanning..." : "Creating...") : "Create Dataset"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
