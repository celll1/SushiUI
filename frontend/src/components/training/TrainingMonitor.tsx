"use client";

import { useState, useEffect } from "react";
import { X, Play, Square, BarChart3, Trash2 } from "lucide-react";
import { TrainingRun, getTrainingStatus, startTrainingRun, stopTrainingRun, deleteTrainingRun, startTensorBoard, stopTensorBoard, getTensorBoardStatus } from "@/utils/api";
import LossChart from "./LossChart";

interface TrainingMonitorProps {
  run: TrainingRun;
  onClose: () => void;
  onStatusChange: (updatedRun: TrainingRun) => void;
  onDelete?: () => void;
}

export default function TrainingMonitor({ run, onClose, onStatusChange, onDelete }: TrainingMonitorProps) {
  const [currentRun, setCurrentRun] = useState<TrainingRun>(run);
  const [logs, setLogs] = useState<string[]>([]);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [tensorboardUrl, setTensorboardUrl] = useState<string | null>(null);
  const [tensorboardRunning, setTensorboardRunning] = useState(false);
  const [tensorboardLoading, setTensorboardLoading] = useState(false);

  // Poll training status
  useEffect(() => {
    // Poll when status is "starting" or "running"
    console.log(`[TrainingMonitor] Poll effect: status="${currentRun.status}", id=${currentRun.id}`);

    if (currentRun.status !== "starting" && currentRun.status !== "running") {
      console.log(`[TrainingMonitor] Not polling (status: ${currentRun.status})`);
      return;
    }

    console.log(`[TrainingMonitor] Starting polling for run ${currentRun.id}`);

    const interval = setInterval(async () => {
      try {
        console.log(`[TrainingMonitor] Polling status for run ${currentRun.id}...`);
        const status = await getTrainingStatus(currentRun.id);
        console.log(`[TrainingMonitor] Received status:`, status);

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
    }, 2000); // Poll every 2 seconds

    return () => {
      console.log(`[TrainingMonitor] Stopping polling for run ${currentRun.id}`);
      clearInterval(interval);
    };
  }, [currentRun.status, currentRun.id]);

  // Check TensorBoard status on mount
  useEffect(() => {
    checkTensorBoardStatus();
  }, [currentRun.id]);

  const checkTensorBoardStatus = async () => {
    try {
      const status = await getTensorBoardStatus(currentRun.id);
      setTensorboardRunning(status.is_running);
      setTensorboardUrl(status.url || null);
    } catch (err) {
      console.error("Failed to check TensorBoard status:", err);
    }
  };

  const handleStartTensorBoard = async () => {
    setTensorboardLoading(true);
    try {
      const response = await startTensorBoard(currentRun.id);
      setTensorboardRunning(true);
      setTensorboardUrl(response.url);
    } catch (err: any) {
      console.error("Failed to start TensorBoard:", err);
      alert(err.response?.data?.detail || "Failed to start TensorBoard");
    } finally {
      setTensorboardLoading(false);
    }
  };

  const handleStopTensorBoard = async () => {
    setTensorboardLoading(true);
    try {
      await stopTensorBoard(currentRun.id);
      setTensorboardRunning(false);
      setTensorboardUrl(null);
    } catch (err: any) {
      console.error("Failed to stop TensorBoard:", err);
      alert(err.response?.data?.detail || "Failed to stop TensorBoard");
    } finally {
      setTensorboardLoading(false);
    }
  };

  const handleStart = async () => {
    console.log(`[TrainingMonitor] Starting training run ${currentRun.id}...`);
    setIsStarting(true);
    try {
      console.log(`[TrainingMonitor] Calling API startTrainingRun(${currentRun.id})...`);
      const response = await startTrainingRun(currentRun.id);
      console.log(`[TrainingMonitor] API response received:`, response);
      console.log(`[TrainingMonitor] Response run status: ${response.run.status}`);
      console.log(`[TrainingMonitor] Setting currentRun state...`);
      setCurrentRun(response.run);
      console.log(`[TrainingMonitor] Calling onStatusChange...`);
      onStatusChange(response.run);
      console.log(`[TrainingMonitor] handleStart completed successfully`);
    } catch (err: any) {
      console.error("[TrainingMonitor] Failed to start training:", err);
      alert(err.response?.data?.detail || "Failed to start training");
    } finally {
      console.log(`[TrainingMonitor] Setting isStarting=false`);
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
    <div className="h-full overflow-y-auto">
      <div className="p-4 border-b border-gray-700 flex items-center justify-between bg-gray-800/50 sticky top-0 z-10">
        <h2 className="text-lg font-semibold">Training Monitor: {currentRun.run_name}</h2>
        <button
          onClick={onClose}
          className="p-1.5 hover:bg-gray-700 rounded transition-colors"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      <div className="p-4 space-y-4 max-w-2xl">
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

        {/* TensorBoard */}
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-sm flex items-center space-x-2">
              <BarChart3 className="h-4 w-4" />
              <span>TensorBoard</span>
            </h3>
            {!tensorboardRunning ? (
              <button
                onClick={handleStartTensorBoard}
                disabled={tensorboardLoading}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {tensorboardLoading ? "Starting..." : "Start TensorBoard"}
              </button>
            ) : (
              <button
                onClick={handleStopTensorBoard}
                disabled={tensorboardLoading}
                className="px-3 py-1 bg-red-600 hover:bg-red-500 rounded text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {tensorboardLoading ? "Stopping..." : "Stop TensorBoard"}
              </button>
            )}
          </div>

          {/* Loss Chart - show directly without TensorBoard server */}
          {(currentRun.status === "running" || currentRun.status === "completed") && (
            <div className="mt-4">
              <LossChart runId={currentRun.id} isRunning={currentRun.status === "running"} />
            </div>
          )}

          {tensorboardRunning && tensorboardUrl ? (
            <div className="mt-4 bg-gray-900 rounded overflow-hidden" style={{ height: "600px" }}>
              <iframe
                src={tensorboardUrl}
                className="w-full h-full border-0"
                title="TensorBoard"
              />
            </div>
          ) : (
            <div className="mt-4 bg-gray-900 rounded p-8 text-center text-gray-500 text-sm">
              <p className="mb-2">Click "Start TensorBoard" to view all metrics</p>
              <p className="text-xs">Full TensorBoard interface with all training metrics and visualizations</p>
            </div>
          )}
        </div>

        {/* Logs (placeholder) */}
        <div className="bg-gray-800 rounded-lg p-3">
          <h3 className="font-semibold mb-2 text-sm">Training Logs</h3>
          <div className="bg-gray-900 rounded p-2 font-mono text-xs h-64 overflow-y-auto">
            {logs.length > 0 ? (
              logs.map((log, i) => <div key={i}>{log}</div>)
            ) : (
              <div className="text-gray-500">Logs will appear here when training starts...</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
