"use client";

import { useState, useEffect } from "react";
import { X, Play, Square, Trash2 } from "lucide-react";
import { TrainingRun, getTrainingStatus, startTrainingRun, stopTrainingRun, deleteTrainingRun, getTrainingSamples, TrainingSampleStep } from "@/utils/api";
import LossChart from "./LossChart";

interface TrainingMonitorProps {
  run: TrainingRun;
  onClose: () => void;
  onStatusChange: (updatedRun: TrainingRun) => void;
  onDelete?: () => void;
}

export default function TrainingMonitor({ run, onClose, onStatusChange, onDelete }: TrainingMonitorProps) {
  const [currentRun, setCurrentRun] = useState<TrainingRun>(run);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [samples, setSamples] = useState<TrainingSampleStep[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  // Poll training status
  useEffect(() => {
    if (currentRun.status !== "starting" && currentRun.status !== "running") {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const status = await getTrainingStatus(currentRun.id);
        setCurrentRun((prev) => ({
          ...prev,
          progress: status.progress,
          current_step: status.current_step,
          loss: status.loss,
          learning_rate: status.learning_rate,
          status: status.status,
        }));
        onStatusChange({
          ...currentRun,
          progress: status.progress,
          current_step: status.current_step,
          loss: status.loss,
          learning_rate: status.learning_rate,
          status: status.status,
        });
      } catch (err) {
        console.error("[TrainingMonitor] Failed to fetch training status:", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [currentRun.status, currentRun.id]);

  // Load sample images
  useEffect(() => {
    loadSamples();

    // Reload samples every 5 seconds when running
    if (currentRun.status === "running") {
      const interval = setInterval(loadSamples, 5000);
      return () => clearInterval(interval);
    }
  }, [currentRun.id, currentRun.status]);

  const loadSamples = async () => {
    try {
      const data = await getTrainingSamples(currentRun.id);
      setSamples(data.samples);
    } catch (err) {
      console.error("Failed to load sample images:", err);
    }
  };

  const handleStart = async () => {
    setIsStarting(true);
    try {
      const response = await startTrainingRun(currentRun.id);
      setCurrentRun(response.run);
      onStatusChange(response.run);
    } catch (err: any) {
      console.error("[TrainingMonitor] Failed to start training:", err);
      alert(err.response?.data?.detail || "Failed to start training");
    } finally {
      setIsStarting(false);
    }
  };

  const handleStop = async () => {
    setIsStopping(true);
    try {
      const response = await stopTrainingRun(currentRun.id);
      setCurrentRun(response.run);
      onStatusChange(response.run);
    } catch (err: any) {
      console.error("Failed to stop training:", err);
      alert(err.response?.data?.detail || "Failed to stop training");
    } finally {
      setIsStopping(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm(`Are you sure you want to delete training run "${currentRun.run_name}"?`)) {
      return;
    }

    setIsDeleting(true);
    try {
      await deleteTrainingRun(currentRun.id);
      if (onDelete) {
        onDelete();
      }
      onClose();
    } catch (err: any) {
      console.error("Failed to delete training run:", err);
      alert(err.response?.data?.detail || "Failed to delete training run");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-700 flex items-center justify-between bg-gray-800/50 shrink-0">
        <h2 className="text-lg font-semibold">Training Monitor: {currentRun.run_name}</h2>
        <button
          onClick={onClose}
          className="p-1.5 hover:bg-gray-700 rounded transition-colors"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Main Content - 2 Column Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Training Info */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Status */}
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium">Status</span>
              <span
                className={`px-2 py-1 rounded text-xs font-medium ${
                  currentRun.status === "running"
                    ? "bg-green-900/50 text-green-400"
                    : currentRun.status === "completed"
                    ? "bg-blue-900/50 text-blue-400"
                    : currentRun.status === "failed"
                    ? "bg-red-900/50 text-red-400"
                    : "bg-gray-700 text-gray-300"
                }`}
              >
                {currentRun.status.toUpperCase()}
              </span>
            </div>

            {/* Progress Bar */}
            <div className="mb-2">
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>
                  Step {currentRun.current_step} / {currentRun.total_steps}
                </span>
                <span>{currentRun.progress.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: `${currentRun.progress}%` }}
                />
              </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-400">Loss:</span>{" "}
                <span className="font-mono">{currentRun.loss?.toFixed(6) || "N/A"}</span>
              </div>
              <div>
                <span className="text-gray-400">LR:</span>{" "}
                <span className="font-mono">{currentRun.learning_rate?.toExponential(2) || "N/A"}</span>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="flex space-x-3">
            {currentRun.status === "pending" || currentRun.status === "stopped" || currentRun.status === "failed" ? (
              <>
                <button
                  onClick={handleStart}
                  disabled={isStarting}
                  className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-500 rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  <Play className="h-4 w-4" />
                  <span>
                    {isStarting ? "Starting..." :
                     currentRun.status === "pending" ? "Start Training" : "Resume Training"}
                  </span>
                </button>
                <button
                  onClick={handleDelete}
                  disabled={isDeleting}
                  className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  <Trash2 className="h-4 w-4" />
                  <span>{isDeleting ? "Deleting..." : "Delete"}</span>
                </button>
              </>
            ) : currentRun.status === "running" || currentRun.status === "starting" ? (
              <button
                onClick={handleStop}
                disabled={isStopping}
                className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-500 rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                <Square className="h-4 w-4" />
                <span>{isStopping ? "Stopping..." : "Stop Training"}</span>
              </button>
            ) : currentRun.status === "completed" ? (
              <button
                onClick={handleDelete}
                disabled={isDeleting}
                className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-500 rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                <Trash2 className="h-4 w-4" />
                <span>{isDeleting ? "Deleting..." : "Delete"}</span>
              </button>
            ) : null}
          </div>

          {/* Configuration Info */}
          <div className="bg-gray-800 rounded-lg p-3 space-y-2 text-sm">
            <h3 className="font-semibold mb-2">Configuration</h3>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <span className="text-gray-400">Method:</span>{" "}
                <span className="capitalize">{currentRun.training_method}</span>
              </div>
              <div>
                <span className="text-gray-400">Total Steps:</span> {currentRun.total_steps}
              </div>
              <div className="col-span-2">
                <span className="text-gray-400">Model:</span>{" "}
                <span className="font-mono text-xs break-all">{currentRun.base_model_path}</span>
              </div>
              <div className="col-span-2">
                <span className="text-gray-400">Output:</span>{" "}
                <span className="font-mono text-xs break-all">{currentRun.output_dir}</span>
              </div>
            </div>
          </div>

          {/* Loss Chart */}
          {(currentRun.status === "running" || currentRun.status === "completed") && (
            <div className="bg-gray-800 rounded-lg p-3">
              <h3 className="font-semibold mb-2 text-sm">Loss</h3>
              <LossChart runId={currentRun.id} isRunning={currentRun.status === "running"} />
            </div>
          )}
        </div>

        {/* Right Panel - Sample Images */}
        <div className="w-80 border-l border-gray-700 overflow-y-auto p-4 space-y-3">
          <h3 className="font-semibold text-sm sticky top-0 bg-gray-900 pb-2">Sample Images</h3>
          {samples.length === 0 ? (
            <div className="text-gray-500 text-sm text-center py-8">
              No samples generated yet
            </div>
          ) : (
            samples.map((stepData) => (
              <div key={stepData.step} className="space-y-2">
                <div className="text-xs text-gray-400 font-medium">Step {stepData.step}</div>
                {stepData.images.map((img) => (
                  <div
                    key={img.path}
                    className="relative cursor-pointer group"
                    onDoubleClick={() => setSelectedImage(img.path)}
                  >
                    <img
                      src={img.path}
                      alt={`Step ${stepData.step} Sample ${img.sample_index}`}
                      className="w-full rounded border border-gray-700 hover:border-blue-500 transition-colors"
                    />
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded">
                      <span className="text-white text-xs">Double-click to view</span>
                    </div>
                  </div>
                ))}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Fullscreen Image Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-w-full max-h-full">
            <img
              src={selectedImage}
              alt="Sample"
              className="max-w-full max-h-full object-contain"
            />
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute top-2 right-2 p-2 bg-gray-900/80 hover:bg-gray-800 rounded-full transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
