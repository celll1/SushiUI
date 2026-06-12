"use client";

import { useState, useEffect } from "react";
import { X, Play, Square, Trash2 } from "lucide-react";
import { TrainingRun, getTrainingStatus, startTrainingRun, stopTrainingRun, deleteTrainingRun, updateTrainingConfig, reloadTrainingConfig, getTrainingSamples, TrainingSampleStep, getDebugLatents, DebugLatent, visualizeDebugLatent, DebugLatentVisualization, skipTrainingRescan } from "@/utils/api";
import { wsClient, DatasetScanProgress } from "@/utils/websocket";
import LossChart from "./LossChart";
import GradNormChart from "./GradNormChart";
import ParamChangeChart from "./ParamChangeChart";
import CheckpointList from "./CheckpointList";

interface TrainingMonitorProps {
  run: TrainingRun;
  onClose: () => void;
  onStatusChange: (updatedRun: TrainingRun) => void;
  onDelete?: () => void;
  onEditConfig?: () => void;
}

export default function TrainingMonitor({ run, onClose, onStatusChange, onDelete, onEditConfig }: TrainingMonitorProps) {
  const [currentRun, setCurrentRun] = useState<TrainingRun>(run);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [samples, setSamples] = useState<TrainingSampleStep[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedStepIndex, setSelectedStepIndex] = useState<number>(0); // For step slider

  // Configuration viewing and editing
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [editedConfig, setEditedConfig] = useState<string>("");
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const [isReloadingConfig, setIsReloadingConfig] = useState(false);

  // Debug latents
  const [viewMode, setViewMode] = useState<"samples" | "debug">("samples");
  const [debugLatents, setDebugLatents] = useState<DebugLatent[]>([]);
  const [selectedDebugStep, setSelectedDebugStep] = useState<number | null>(null);
  const [debugVisualization, setDebugVisualization] = useState<DebugLatentVisualization | null>(null);
  const [comparisonSlider, setComparisonSlider] = useState<number>(50); // 0-100
  const [, setTimeTick] = useState(0); // Force re-render for time update

  // Dataset scan progress (drift detection / rescan) — shown until training proper starts
  const [scanMessage, setScanMessage] = useState<string | null>(null);
  // Dataset currently being rescanned (drift_walk / rescan phase) — drives the
  // "Skip rescan" button. null when no skippable scan is in progress.
  const [scanDatasetId, setScanDatasetId] = useState<number | null>(null);
  const [scanSkipping, setScanSkipping] = useState(false);

  // Poll training status
  useEffect(() => {
    if (currentRun.status !== "starting" && currentRun.status !== "running") {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const status = await getTrainingStatus(currentRun.id);
        const updatedRun = {
          ...currentRun,
          progress: status.progress,
          current_step: status.current_step,
          total_steps: status.total_steps,  // Update total_steps (may change on MNT change)
          loss: status.loss,
          learning_rate: status.learning_rate,
          status: status.status,
          phase: status.phase,
          phase_progress: status.phase_progress,
          phase_detail: status.phase_detail,
        };
        setCurrentRun(updatedRun);
        onStatusChange(updatedRun);
      } catch (err) {
        console.error("[TrainingMonitor] Failed to fetch training status:", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [currentRun.status, currentRun.id]);

  // Update time display every second
  useEffect(() => {
    if (currentRun.status !== "running") {
      return;
    }

    const interval = setInterval(() => {
      setTimeTick(prev => prev + 1); // Force re-render
    }, 1000);

    return () => clearInterval(interval);
  }, [currentRun.status]);

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

  // Subscribe to dataset scan progress (drift check / rescan / cleanup) over WS
  useEffect(() => {
    const handler = (ev: DatasetScanProgress) => {
      if (ev.scope !== "training") return;
      if (String(ev.run_id) !== String(currentRun.id)) return;
      // "MyDataset (#25)" when a name is known, else "dataset 25".
      const dsLabel = ev.dataset_name ? `${ev.dataset_name} (#${ev.dataset_id})` : `dataset ${ev.dataset_id}`;
      let msg = "";
      if (ev.phase === "drift_walk") {
        msg = `Drift check: ${dsLabel} — walked ${(ev.files_walked ?? 0).toLocaleString()} files`;
        setScanDatasetId(ev.dataset_id);
      } else if (ev.phase === "drift_done") {
        if ((ev.items_missing ?? 0) === 0 && (ev.items_new ?? 0) === 0) {
          msg = `Drift check: ${dsLabel} — no drift (${(ev.files_walked ?? 0).toLocaleString()} files)`;
        } else {
          msg = `Drift check: ${dsLabel} — ${ev.items_missing ?? 0} missing, ${ev.items_new ?? 0} new`;
        }
      } else if (ev.phase === "rescan") {
        msg = `Rescanning ${dsLabel}${ev.message ? ` — ${ev.message}` : "..."}`;
        setScanDatasetId(ev.dataset_id);
      } else if (ev.phase === "cleanup") {
        msg = `Cleaning orphan latent cache for ${dsLabel}...`;
        setScanDatasetId(null);
      } else if (ev.phase === "skipped") {
        msg = `Skipped rescan of ${dsLabel}`;
        setScanDatasetId(null);
        setScanSkipping(false);
      }
      if (msg) setScanMessage(msg);
    };
    wsClient.subscribeToDatasetScanProgress(handler);
    return () => wsClient.unsubscribeFromDatasetScanProgress(handler);
  }, [currentRun.id]);

  // Clear scan message once training proper starts
  useEffect(() => {
    if (currentRun.phase === "training" || currentRun.phase === "latent_cache" || currentRun.phase === "text_encoder_cache") {
      setScanMessage(null);
      setScanDatasetId(null);
      setScanSkipping(false);
    }
  }, [currentRun.phase]);

  // Load debug latents
  useEffect(() => {
    loadDebugLatents();

    // Reload debug latents every 5 seconds when running
    if (currentRun.status === "running") {
      const interval = setInterval(loadDebugLatents, 5000);
      return () => clearInterval(interval);
    }
  }, [currentRun.id, currentRun.status]);

  const loadDebugLatents = async () => {
    try {
      const data = await getDebugLatents(currentRun.id);
      setDebugLatents(data.debug_latents);
    } catch (err) {
      console.error("Failed to load debug latents:", err);
    }
  };

  // Load debug visualization when step is selected
  useEffect(() => {
    if (selectedDebugStep !== null && viewMode === "debug") {
      loadDebugVisualization(selectedDebugStep);
    }
  }, [selectedDebugStep, viewMode]);

  const loadDebugVisualization = async (step: number) => {
    try {
      const data = await visualizeDebugLatent(currentRun.id, step);
      setDebugVisualization(data);
    } catch (err) {
      console.error("Failed to load debug visualization:", err);
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

  // Calculate elapsed time and ETA
  const calculateTimeInfo = () => {
    if (!currentRun.started_at || currentRun.status === "pending") {
      return { elapsed: "N/A", eta: "N/A" };
    }

    // Use last_resumed_at if available (resume case), otherwise use started_at (first start)
    const referenceTime = currentRun.last_resumed_at || currentRun.started_at;
    const startTime = new Date(referenceTime).getTime();
    const now = Date.now();
    const elapsedMs = now - startTime;

    // Format elapsed time
    const elapsedSeconds = Math.floor(elapsedMs / 1000);
    const elapsedHours = Math.floor(elapsedSeconds / 3600);
    const elapsedMinutes = Math.floor((elapsedSeconds % 3600) / 60);
    const elapsedSecs = elapsedSeconds % 60;
    const elapsed = `${elapsedHours}h ${elapsedMinutes}m ${elapsedSecs}s`;

    // Calculate ETA (only if training is running and progress > 0)
    if (currentRun.status !== "running" || currentRun.progress <= 0 || currentRun.current_step === 0) {
      return { elapsed, eta: "N/A" };
    }

    // Calculate progress based on steps completed since resume (or from start)
    const startStep = currentRun.resumed_from_step ?? 0;  // Step at resume (or 0 if first start)
    const stepsCompleted = currentRun.current_step - startStep;
    const remainingSteps = currentRun.total_steps - currentRun.current_step;

    if (stepsCompleted <= 0) {
      return { elapsed, eta: "Calculating..." };
    }

    if (remainingSteps <= 0) {
      return { elapsed, eta: "Completed" };
    }

    // ETA = (elapsed time / steps completed) * remaining steps
    const msPerStep = elapsedMs / stepsCompleted;
    const remainingMs = msPerStep * remainingSteps;

    // Format ETA
    const remainingSeconds = Math.floor(remainingMs / 1000);
    const etaHours = Math.floor(remainingSeconds / 3600);
    const etaMinutes = Math.floor((remainingSeconds % 3600) / 60);
    const etaSecs = remainingSeconds % 60;
    const eta = `${etaHours}h ${etaMinutes}m ${etaSecs}s`;

    return { elapsed, eta };
  };

  const timeInfo = calculateTimeInfo();

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-3 sm:p-4 border-b border-gray-700 flex items-center justify-between bg-gray-800/50 shrink-0">
        <h2 className="text-base sm:text-lg font-semibold truncate mr-2">Training Monitor: {currentRun.run_name}</h2>
        <button
          onClick={onClose}
          className="p-1.5 hover:bg-gray-700 rounded transition-colors flex-shrink-0"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Main Content - Responsive Layout */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Left Panel - Training Info */}
        <div className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-3 sm:space-y-4">
          {/* Status */}
          <div className="bg-gray-800 rounded-lg p-2.5 sm:p-3">
            <div className="flex items-center justify-between mb-2 sm:mb-3">
              <span className="text-xs sm:text-sm font-medium">Status</span>
              <span
                className={`px-1.5 sm:px-2 py-0.5 sm:py-1 rounded text-xxs sm:text-xs font-medium ${
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
              <div className="flex justify-between text-xxs sm:text-xs text-gray-400 mb-1">
                <span>
                  {/* Phase-based display */}
                  {currentRun.phase === "latent_cache" && "Latent Cache"}
                  {currentRun.phase === "text_encoder_cache" && "Text Encoder Cache"}
                  {currentRun.phase === "training" && `Step ${currentRun.current_step} / ${currentRun.total_steps}`}
                  {currentRun.phase === "initializing" && "Initializing..."}
                  {!currentRun.phase && `Step ${currentRun.current_step} / ${currentRun.total_steps}`}
                </span>
                <span>
                  {currentRun.phase === "training" || !currentRun.phase
                    ? `${currentRun.progress.toFixed(1)}%`
                    : `${(currentRun.phase_progress || 0).toFixed(1)}%`
                  }
                </span>
              </div>

              {/* Detail message — scan progress takes priority over phase_detail */}
              {scanMessage ? (
                <div className="flex items-center gap-2 mb-1">
                  <div className="text-xs text-blue-300 flex-1">{scanMessage}</div>
                  {scanDatasetId !== null && (
                    <button
                      onClick={async () => {
                        setScanSkipping(true);
                        try {
                          await skipTrainingRescan(currentRun.id, scanDatasetId);
                        } catch (err) {
                          console.error("Failed to skip rescan:", err);
                          setScanSkipping(false);
                        }
                      }}
                      disabled={scanSkipping}
                      className="text-xs px-2 py-0.5 rounded border border-yellow-600 text-yellow-400 hover:bg-yellow-900/30 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
                      title="Skip rescanning this dataset and continue (already-applied changes are kept)"
                    >
                      {scanSkipping ? "Skipping…" : "Skip rescan"}
                    </button>
                  )}
                </div>
              ) : currentRun.phase_detail && currentRun.phase !== "training" ? (
                <div className="text-xs text-gray-500 mb-1">
                  {currentRun.phase_detail}
                </div>
              ) : null}

              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{
                    width: currentRun.phase === "training" || !currentRun.phase
                      ? `${currentRun.progress}%`
                      : `${currentRun.phase_progress || 0}%`
                  }}
                />
              </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-1.5 sm:gap-2 text-xs sm:text-sm mb-2">
              <div>
                <span className="text-gray-400">Loss:</span>{" "}
                <span className="font-mono text-xs sm:text-sm">{currentRun.loss?.toFixed(6) || "N/A"}</span>
              </div>
              <div>
                <span className="text-gray-400">LR:</span>{" "}
                <span className="font-mono text-xs sm:text-sm">{currentRun.learning_rate?.toExponential(2) || "N/A"}</span>
              </div>
            </div>

            {/* Time Info */}
            <div className="grid grid-cols-2 gap-1.5 sm:gap-2 text-xs sm:text-sm">
              <div>
                <span className="text-gray-400">Elapsed:</span>{" "}
                <span className="font-mono text-blue-400 text-xs sm:text-sm">{timeInfo.elapsed}</span>
              </div>
              <div>
                <span className="text-gray-400">ETA:</span>{" "}
                <span className="font-mono text-green-400 text-xs sm:text-sm">{timeInfo.eta}</span>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="flex space-x-2 sm:space-x-3">
            {currentRun.status === "pending" || currentRun.status === "stopped" || currentRun.status === "failed" || currentRun.status === "completed" ? (
              <>
                <button
                  onClick={handleStart}
                  disabled={isStarting}
                  className="flex-1 px-3 sm:px-4 py-1.5 sm:py-2 bg-green-600 hover:bg-green-500 rounded text-xs sm:text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-1.5 sm:space-x-2"
                >
                  <Play className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                  <span>
                    {isStarting ? "Starting..." :
                     currentRun.status === "pending" ? "Start Training" : "Resume Training"}
                  </span>
                </button>
                <button
                  onClick={handleDelete}
                  disabled={isDeleting}
                  className="px-3 sm:px-4 py-1.5 sm:py-2 bg-red-600 hover:bg-red-500 rounded text-xs sm:text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-1.5 sm:space-x-2"
                >
                  <Trash2 className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                  <span>{isDeleting ? "Deleting..." : "Delete"}</span>
                </button>
              </>
            ) : currentRun.status === "running" || currentRun.status === "starting" ? (
              <button
                onClick={handleStop}
                disabled={isStopping}
                className="flex-1 px-3 sm:px-4 py-1.5 sm:py-2 bg-red-600 hover:bg-red-500 rounded text-xs sm:text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-1.5 sm:space-x-2"
              >
                <Square className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                <span>{isStopping ? "Stopping..." : "Stop Training"}</span>
              </button>
            ) : null}
          </div>

          {/* Configuration Info */}
          <div className="bg-gray-800 rounded-lg p-2.5 sm:p-3 space-y-2 text-xs sm:text-sm">
            <div className="flex items-start justify-between mb-2 gap-2">
              <h3 className="font-semibold text-sm flex-shrink-0">Configuration</h3>
              <div className="flex flex-wrap gap-1.5 sm:gap-2 justify-end">
                {onEditConfig && (
                  <button
                    onClick={onEditConfig}
                    disabled={currentRun.status === "running" || currentRun.status === "starting"}
                    className="text-xxs sm:text-xs px-1.5 sm:px-2 py-0.5 sm:py-1 bg-blue-700 hover:bg-blue-600 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
                  >
                    Edit Config
                  </button>
                )}
                <button
                  onClick={async () => {
                    console.log("[TrainingMonitor] Reloading config from disk for run ID:", currentRun.id);
                    setIsReloadingConfig(true);
                    try {
                      const response = await reloadTrainingConfig(currentRun.id);
                      console.log("[TrainingMonitor] Config reload response:", response);
                      setCurrentRun(response.run);
                      onStatusChange(response.run);
                      alert("Configuration reloaded from disk successfully!");
                    } catch (err: any) {
                      console.error("[TrainingMonitor] Failed to reload config:", err);
                      console.error("[TrainingMonitor] Error response:", err.response);
                      alert(err.response?.data?.detail || err.message || "Failed to reload configuration");
                    } finally {
                      setIsReloadingConfig(false);
                    }
                  }}
                  disabled={isReloadingConfig || currentRun.status === "running" || currentRun.status === "starting"}
                  className="text-xxs sm:text-xs px-1.5 sm:px-2 py-0.5 sm:py-1 bg-green-700 hover:bg-green-600 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
                >
                  {isReloadingConfig ? "Reloading..." : "Reload"}
                </button>
                <button
                  onClick={() => {
                    setEditedConfig(currentRun.config_yaml || "");
                    setShowConfigModal(true);
                  }}
                  className="text-xxs sm:text-xs px-1.5 sm:px-2 py-0.5 sm:py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors whitespace-nowrap"
                >
                  View Full
                </button>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 sm:gap-2">
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
          {(currentRun.status === "running" || currentRun.status === "completed" || currentRun.status === "failed" || currentRun.status === "stopped") && (
            <>
              <div className="bg-gray-800 rounded-lg p-2.5 sm:p-3">
                <h3 className="font-semibold mb-2 text-xs sm:text-sm">Loss</h3>
                <LossChart runId={currentRun.id} isRunning={currentRun.status === "running"} />
              </div>
              <div className="bg-gray-800 rounded-lg p-2.5 sm:p-3">
                <h3 className="font-semibold mb-2 text-xs sm:text-sm">Gradient Norm</h3>
                <GradNormChart runId={currentRun.id} isRunning={currentRun.status === "running"} />
              </div>
              <ParamChangeChart runId={currentRun.id} isRunning={currentRun.status === "running"} />
            </>
          )}

          {/* Checkpoint List */}
          {(currentRun.status === "running" || currentRun.status === "completed" || currentRun.status === "failed" || currentRun.status === "stopped") && (
            <CheckpointList checkpoints={currentRun.checkpoint_paths} runId={currentRun.id} />
          )}
        </div>

        {/* Right Panel - Sample Images / Debug Latents - Stacked on mobile, side-by-side on desktop */}
        <div className="w-full lg:w-80 border-t lg:border-t-0 lg:border-l border-gray-700 flex flex-col">
          {/* Tab Header */}
          <div className="flex border-b border-gray-700 bg-gray-900 sticky top-0 z-10">
            <button
              onClick={() => setViewMode("samples")}
              className={`flex-1 px-3 sm:px-4 py-1.5 sm:py-2 text-xs sm:text-sm font-medium transition-colors ${
                viewMode === "samples"
                  ? "text-blue-400 border-b-2 border-blue-400"
                  : "text-gray-400 hover:text-gray-300"
              }`}
            >
              Samples
            </button>
            <button
              onClick={() => setViewMode("debug")}
              className={`flex-1 px-3 sm:px-4 py-1.5 sm:py-2 text-xs sm:text-sm font-medium transition-colors ${
                viewMode === "debug"
                  ? "text-blue-400 border-b-2 border-blue-400"
                  : "text-gray-400 hover:text-gray-300"
              }`}
            >
              Debug Latents
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-2.5 sm:space-y-3">
            {viewMode === "samples" ? (
              <>
                {samples.length === 0 ? (
                  <div className="text-gray-500 text-xs sm:text-sm text-center py-8">
                    No samples generated yet
                  </div>
                ) : (
                  <div className="space-y-2.5 sm:space-y-3">
                    {/* Step Selector */}
                    <div>
                      <label className="block text-xxs sm:text-xs text-gray-400 mb-1.5">Training Step</label>
                      <input
                        type="range"
                        min="0"
                        max={Math.max(0, samples.length - 1)}
                        value={selectedStepIndex}
                        onChange={(e) => setSelectedStepIndex(Number(e.target.value))}
                        className="w-full mb-1"
                      />
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-300 font-medium">
                          Step {samples[selectedStepIndex]?.step || 0}
                        </span>
                        <span className="text-gray-500">
                          {selectedStepIndex + 1} / {samples.length}
                        </span>
                      </div>
                    </div>

                    {/* Generation Settings */}
                    {samples[selectedStepIndex]?.images[0]?.params && (() => {
                      const p = samples[selectedStepIndex].images[0].params!;
                      return (
                        <div className="text-xs space-y-1.5 bg-gray-800 rounded p-2">
                          <div className="font-semibold text-gray-300 mb-1">Generation Settings</div>
                          {p.prompt && (
                            <div className="text-gray-300 italic leading-relaxed border-b border-gray-700 pb-1.5 mb-1">
                              {p.prompt.length > 120 ? `${p.prompt.substring(0, 120)}…` : p.prompt}
                            </div>
                          )}
                          <div className="text-gray-400">
                            {p.steps} steps / CFG {p.cfg_scale} / seed {p.seed}
                          </div>
                          <div className="text-gray-400">{p.width} × {p.height}</div>
                          {p.schedule_type && p.schedule_type !== "uniform" && (
                            <div>
                              <span className="text-gray-500">Schedule:</span>{" "}
                              <span className="text-gray-300">{p.schedule_type}</span>
                            </div>
                          )}
                          {p.reference_image_path && (
                            <div className="pt-1 border-t border-gray-700">
                              <div className="text-gray-500 mb-1">Reference Image</div>
                              <img
                                src={`/api/serve-image?path=${encodeURIComponent(
                                  p.reference_image_path.replace("temp_img://", "")
                                )}`}
                                className="h-20 w-20 object-cover rounded border border-gray-700"
                                alt="Reference"
                              />
                            </div>
                          )}
                          {p.condition_image_path && (
                            <div className="pt-1 border-t border-gray-700">
                              <div className="text-gray-500 mb-1">Condition Image</div>
                              <img
                                src={`/api/serve-image?path=${encodeURIComponent(
                                  p.condition_image_path.replace("temp_img://", "")
                                )}`}
                                className="h-20 w-20 object-cover rounded border border-gray-700"
                                alt="Condition"
                              />
                            </div>
                          )}
                        </div>
                      );
                    })()}

                    {/* Sample Images */}
                    <div className="space-y-2">
                      {samples[selectedStepIndex]?.images.map((img) => (
                        <div
                          key={img.path}
                          className="relative cursor-pointer group"
                          onDoubleClick={() => setSelectedImage(img.path)}
                        >
                          <img
                            src={img.path}
                            alt={`Step ${samples[selectedStepIndex].step} Sample ${img.sample_index}`}
                            className="w-full rounded border border-gray-700 hover:border-blue-500 transition-colors"
                          />
                          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded">
                            <span className="text-white text-xs">Double-click to view</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <>
                {/* Debug Latents View */}
                {debugLatents.length === 0 ? (
                  <div className="text-gray-500 text-sm text-center py-8">
                    No debug latents saved yet
                    <div className="text-xs mt-2">Enable debug mode in training config</div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {/* Step Selector */}
                    <div>
                      <label className="block text-xs text-gray-400 mb-1.5">Select Step</label>
                      <select
                        value={selectedDebugStep ?? ""}
                        onChange={(e) => setSelectedDebugStep(Number(e.target.value))}
                        className="w-full px-2 py-1.5 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500"
                      >
                        <option value="">-- Select Step --</option>
                        {Array.from(new Set(debugLatents.map(d => d.step)))
                          .sort((a, b) => a - b)
                          .map(step => (
                            <option key={step} value={step}>Step {step}</option>
                          ))
                        }
                      </select>
                    </div>

                    {/* Latent Visualization */}
                    {debugVisualization && (
                      <div className="space-y-3">
                        <div className="text-xs space-y-1 bg-gray-800 rounded p-2">
                          <div><span className="text-gray-400">Step:</span> {debugVisualization.step}</div>
                          <div><span className="text-gray-400">Timestep:</span> {debugVisualization.timestep}</div>
                          <div><span className="text-gray-400">Prediction Loss:</span> {debugVisualization.loss.toFixed(6)}</div>
                          {debugVisualization.recon_loss !== undefined && (
                            <div><span className="text-gray-400">Recon Loss:</span> {debugVisualization.recon_loss.toFixed(6)}</div>
                          )}
                          {debugVisualization.caption && (
                            <div className="pt-1 border-t border-gray-700">
                              <div className="text-gray-400 mb-0.5">Caption (processed):</div>
                              <div
                                className="text-gray-300 text-xs leading-relaxed max-h-20 overflow-y-auto cursor-help"
                                title={debugVisualization.caption}
                              >
                                {debugVisualization.caption.length > 150
                                  ? `${debugVisualization.caption.substring(0, 150)}...`
                                  : debugVisualization.caption}
                              </div>
                              <div className="text-gray-500 text-xxs mt-0.5">
                                (Hover to see full caption)
                              </div>
                            </div>
                          )}
                          {debugVisualization.reference_image && (
                            <div className="pt-1 border-t border-gray-700">
                              <div className="text-gray-400 mb-0.5">Reference Image (training batch):</div>
                              <img
                                src={`data:image/png;base64,${debugVisualization.reference_image}`}
                                className="h-20 w-20 object-cover rounded border border-gray-700"
                                alt="Reference"
                              />
                            </div>
                          )}
                        </div>

                        {/* Image Comparison with Slider */}
                        <div className="space-y-2">
                          <div className="text-xs text-gray-400">Latent Comparison (Goal: Minimize Difference)</div>

                          {/* Comparison Container */}
                          <div className="relative aspect-square bg-gray-800 rounded overflow-hidden">
                            {/* Base Layer: Latents (original/target) */}
                            {debugVisualization.latents_image && (
                              <div className="absolute inset-0">
                                <img
                                  src={`data:image/png;base64,${debugVisualization.latents_image}`}
                                  alt="Latents (Target)"
                                  className="w-full h-full object-contain"
                                />
                                <div className="absolute top-1 left-1 bg-green-700/80 text-white text-xs px-1.5 py-0.5 rounded">
                                  Target
                                </div>
                              </div>
                            )}

                            {/* Top Layer: Predicted Latents (clipped by slider) */}
                            {debugVisualization.predicted_latent_image && (
                              <div
                                className="absolute inset-0"
                                style={{ clipPath: `inset(0 ${100 - comparisonSlider}% 0 0)` }}
                              >
                                <img
                                  src={`data:image/png;base64,${debugVisualization.predicted_latent_image}`}
                                  alt="Predicted Latent"
                                  className="w-full h-full object-contain"
                                />
                                <div className="absolute top-1 right-1 bg-blue-700/80 text-white text-xs px-1.5 py-0.5 rounded">
                                  Predicted
                                </div>
                              </div>
                            )}

                            {/* Slider Line */}
                            <div
                              className="absolute top-0 bottom-0 w-0.5 bg-yellow-500 pointer-events-none"
                              style={{ left: `${comparisonSlider}%` }}
                            />
                          </div>

                          {/* Slider Control */}
                          <input
                            type="range"
                            min="0"
                            max="100"
                            value={comparisonSlider}
                            onChange={(e) => setComparisonSlider(Number(e.target.value))}
                            className="w-full"
                          />
                          <div className="flex justify-between text-xs text-gray-500">
                            <span>Target (Original)</span>
                            <span>Predicted (t=0)</span>
                          </div>

                          {/* Additional Debug Images */}
                          <div className="grid grid-cols-2 gap-2 mt-3">
                            {/* Noisy Latents */}
                            {debugVisualization.noisy_latents_image && (
                              <div>
                                <div className="text-xs text-gray-400 mb-1">Noisy Latents (t={debugVisualization.timestep})</div>
                                <div className="relative aspect-square bg-gray-800 rounded overflow-hidden">
                                  <img
                                    src={`data:image/png;base64,${debugVisualization.noisy_latents_image}`}
                                    alt="Noisy Latents"
                                    className="w-full h-full object-contain"
                                  />
                                </div>
                              </div>
                            )}

                            {/* Predicted Noise */}
                            {debugVisualization.predicted_noise_image && (
                              <div>
                                <div className="text-xs text-gray-400 mb-1">Predicted Noise</div>
                                <div className="relative aspect-square bg-gray-800 rounded overflow-hidden">
                                  <img
                                    src={`data:image/png;base64,${debugVisualization.predicted_noise_image}`}
                                    alt="Predicted Noise"
                                    className="w-full h-full object-contain"
                                  />
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Fullscreen Image Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-2 sm:p-4"
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
              className="absolute top-1 right-1 sm:top-2 sm:right-2 p-1.5 sm:p-2 bg-gray-900/80 hover:bg-gray-800 rounded-full transition-colors"
            >
              <X className="h-4 w-4 sm:h-5 sm:w-5" />
            </button>
          </div>
        </div>
      )}

      {/* Config Modal */}
      {showConfigModal && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-2 sm:p-4">
          <div className="bg-gray-900 rounded-lg w-full max-w-4xl max-h-[90vh] flex flex-col">
            {/* Header */}
            <div className="p-3 sm:p-4 border-b border-gray-700 flex items-center justify-between">
              <h3 className="text-base sm:text-lg font-semibold">Training Configuration</h3>
              <button
                onClick={() => setShowConfigModal(false)}
                className="p-1.5 hover:bg-gray-700 rounded transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-3 sm:p-4">
              <div className="space-y-3">
                {/* Read-only view for running training */}
                {(currentRun.status === "running" || currentRun.status === "starting") ? (
                  <div>
                    <div className="text-xs sm:text-sm text-gray-400 mb-2">
                      Configuration is read-only while training is running.
                    </div>
                    <pre className="bg-gray-800 p-3 sm:p-4 rounded text-xxs sm:text-xs font-mono overflow-x-auto">
                      {currentRun.config_yaml || "No configuration available"}
                    </pre>
                  </div>
                ) : (
                  // Editable view for stopped/failed training
                  <div>
                    <div className="text-xs sm:text-sm text-gray-400 mb-2">
                      {currentRun.status === "pending"
                        ? "Edit configuration before starting training:"
                        : "Edit configuration and resume training:"}
                    </div>
                    <textarea
                      value={editedConfig}
                      onChange={(e) => setEditedConfig(e.target.value)}
                      className="w-full h-64 sm:h-96 bg-gray-800 p-3 sm:p-4 rounded text-xxs sm:text-xs font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="YAML configuration..."
                    />
                    <div className="mt-3 flex flex-col sm:flex-row justify-end gap-2 sm:gap-3">
                      <button
                        onClick={() => setShowConfigModal(false)}
                        className="px-3 sm:px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-xs sm:text-sm transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={async () => {
                          console.log("[TrainingMonitor] Saving config for run ID:", currentRun.id);
                          console.log("[TrainingMonitor] Edited config length:", editedConfig.length);
                          setIsSavingConfig(true);
                          try {
                            const response = await updateTrainingConfig(currentRun.id, editedConfig);
                            console.log("[TrainingMonitor] Config update response:", response);
                            setCurrentRun(response.run);
                            onStatusChange(response.run);
                            alert("Configuration updated successfully! You can now start training.");
                            setShowConfigModal(false);
                          } catch (err: any) {
                            console.error("[TrainingMonitor] Failed to update config:", err);
                            console.error("[TrainingMonitor] Error response:", err.response);
                            alert(err.response?.data?.detail || err.message || "Failed to update configuration");
                          } finally {
                            setIsSavingConfig(false);
                          }
                        }}
                        disabled={isSavingConfig}
                        className="px-3 sm:px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-xs sm:text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {isSavingConfig ? "Saving..." : "Save Configuration"}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
