"use client";

import { useState, useEffect } from "react";
import { X, Play, Square, Trash2 } from "lucide-react";
import { TrainingRun, getTrainingStatus, startTrainingRun, stopTrainingRun, deleteTrainingRun, updateTrainingConfig, getTrainingSamples, TrainingSampleStep, getDebugLatents, DebugLatent, visualizeDebugLatent, DebugLatentVisualization } from "@/utils/api";
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
  const [selectedStepIndex, setSelectedStepIndex] = useState<number>(0); // For step slider

  // Configuration viewing and editing
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [editedConfig, setEditedConfig] = useState<string>("");
  const [isSavingConfig, setIsSavingConfig] = useState(false);

  // Debug latents
  const [viewMode, setViewMode] = useState<"samples" | "debug">("samples");
  const [debugLatents, setDebugLatents] = useState<DebugLatent[]>([]);
  const [selectedDebugStep, setSelectedDebugStep] = useState<number | null>(null);
  const [debugVisualization, setDebugVisualization] = useState<DebugLatentVisualization | null>(null);
  const [comparisonSlider, setComparisonSlider] = useState<number>(50); // 0-100

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
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold">Configuration</h3>
              <button
                onClick={() => {
                  setEditedConfig(currentRun.config_yaml || "");
                  setShowConfigModal(true);
                }}
                className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
              >
                View Full Config
              </button>
            </div>
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

        {/* Right Panel - Sample Images / Debug Latents */}
        <div className="w-80 border-l border-gray-700 flex flex-col">
          {/* Tab Header */}
          <div className="flex border-b border-gray-700 bg-gray-900 sticky top-0 z-10">
            <button
              onClick={() => setViewMode("samples")}
              className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                viewMode === "samples"
                  ? "text-blue-400 border-b-2 border-blue-400"
                  : "text-gray-400 hover:text-gray-300"
              }`}
            >
              Samples
            </button>
            <button
              onClick={() => setViewMode("debug")}
              className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                viewMode === "debug"
                  ? "text-blue-400 border-b-2 border-blue-400"
                  : "text-gray-400 hover:text-gray-300"
              }`}
            >
              Debug Latents
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {viewMode === "samples" ? (
              <>
                {samples.length === 0 ? (
                  <div className="text-gray-500 text-sm text-center py-8">
                    No samples generated yet
                  </div>
                ) : (
                  <div className="space-y-3">
                    {/* Step Selector */}
                    <div>
                      <label className="block text-xs text-gray-400 mb-1.5">Training Step</label>
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
                    {samples[selectedStepIndex]?.images[0]?.params && (
                      <div className="text-xs space-y-1 bg-gray-800 rounded p-2">
                        <div className="font-semibold text-gray-300 mb-1">Generation Settings</div>
                        <div>
                          <span className="text-gray-400">Steps:</span>{" "}
                          {samples[selectedStepIndex].images[0].params.steps || 20}
                        </div>
                        <div>
                          <span className="text-gray-400">CFG Scale:</span>{" "}
                          {samples[selectedStepIndex].images[0].params.cfg_scale || 7.0}
                        </div>
                        <div>
                          <span className="text-gray-400">Sampler:</span>{" "}
                          {samples[selectedStepIndex].images[0].params.sampler || "N/A"}
                        </div>
                        <div>
                          <span className="text-gray-400">Size:</span>{" "}
                          {samples[selectedStepIndex].images[0].params.width || 1024} × {samples[selectedStepIndex].images[0].params.height || 1024}
                        </div>
                      </div>
                    )}

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
                          <div><span className="text-gray-400">Loss:</span> {debugVisualization.loss.toFixed(6)}</div>
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

      {/* Config Modal */}
      {showConfigModal && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 rounded-lg w-full max-w-4xl max-h-[90vh] flex flex-col">
            {/* Header */}
            <div className="p-4 border-b border-gray-700 flex items-center justify-between">
              <h3 className="text-lg font-semibold">Training Configuration</h3>
              <button
                onClick={() => setShowConfigModal(false)}
                className="p-1.5 hover:bg-gray-700 rounded transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4">
              <div className="space-y-3">
                {/* Read-only view for running training */}
                {(currentRun.status === "running" || currentRun.status === "starting") ? (
                  <div>
                    <div className="text-sm text-gray-400 mb-2">
                      Configuration is read-only while training is running.
                    </div>
                    <pre className="bg-gray-800 p-4 rounded text-xs font-mono overflow-x-auto">
                      {currentRun.config_yaml || "No configuration available"}
                    </pre>
                  </div>
                ) : (
                  // Editable view for stopped/failed training
                  <div>
                    <div className="text-sm text-gray-400 mb-2">
                      {currentRun.status === "pending"
                        ? "Edit configuration before starting training:"
                        : "Edit configuration and resume training:"}
                    </div>
                    <textarea
                      value={editedConfig}
                      onChange={(e) => setEditedConfig(e.target.value)}
                      className="w-full h-96 bg-gray-800 p-4 rounded text-xs font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="YAML configuration..."
                    />
                    <div className="mt-3 flex justify-end space-x-3">
                      <button
                        onClick={() => setShowConfigModal(false)}
                        className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={async () => {
                          setIsSavingConfig(true);
                          try {
                            const response = await updateTrainingConfig(currentRun.id, editedConfig);
                            setCurrentRun(response.run);
                            onStatusChange(response.run);
                            alert("Configuration updated successfully! You can now start training.");
                            setShowConfigModal(false);
                          } catch (err: any) {
                            console.error("Failed to update config:", err);
                            alert(err.response?.data?.detail || "Failed to update configuration");
                          } finally {
                            setIsSavingConfig(false);
                          }
                        }}
                        disabled={isSavingConfig}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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
