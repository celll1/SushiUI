"use client";

import { useState, useEffect } from "react";
import { X, Play, Square } from "lucide-react";
import { TrainingRun, getTrainingStatus, startTrainingRun, stopTrainingRun } from "@/utils/api";

interface TrainingMonitorProps {
  run: TrainingRun;
  onClose: () => void;
  onStatusChange: (updatedRun: TrainingRun) => void;
}

export default function TrainingMonitor({ run, onClose, onStatusChange }: TrainingMonitorProps) {
  const [currentRun, setCurrentRun] = useState<TrainingRun>(run);
  const [logs, setLogs] = useState<string[]>([]);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);

  // Poll training status
  useEffect(() => {
    // Poll when status is "starting" or "running"
    if (currentRun.status !== "starting" && currentRun.status !== "running") return;

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
        console.error("Failed to fetch training status:", err);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [currentRun.status, currentRun.id]);

  const handleStart = async () => {
    setIsStarting(true);
    try {
      const response = await startTrainingRun(currentRun.id);
      setCurrentRun(response.run);
      onStatusChange(response.run);
    } catch (err: any) {
      console.error("Failed to start training:", err);
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
          {currentRun.status === "pending" || currentRun.status === "stopped" ? (
            <button
              onClick={handleStart}
              disabled={isStarting}
              className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-500 rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              <Play className="h-4 w-4" />
              <span>{isStarting ? "Starting..." : "Start Training"}</span>
            </button>
          ) : currentRun.status === "running" ? (
            <button
              onClick={handleStop}
              disabled={isStopping}
              className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-500 rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              <Square className="h-4 w-4" />
              <span>{isStopping ? "Stopping..." : "Stop Training"}</span>
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
