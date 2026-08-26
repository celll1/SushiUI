"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { X } from "lucide-react";
import Sidebar from "@/components/common/Sidebar";
import ProtectedRoute from "@/components/common/ProtectedRoute";
import TrainingList from "@/components/training/TrainingList";
import TrainingConfig from "@/components/training/TrainingConfig";
import TrainingMonitor from "@/components/training/TrainingMonitor";
import TaggerTrainingConfig from "@/components/training/tagger/TaggerTrainingConfig";
import TaggerTrainingMonitor from "@/components/training/tagger/TaggerTrainingMonitor";
import VaeTrainingConfig from "@/components/training/vae/VaeTrainingConfig";
import { listTrainingRuns, listTaggerTrainingRuns, TrainingRun, TaggerTrainingRun } from "@/utils/api";

export default function TrainingPage() {
  return (
    <ProtectedRoute>
      <TrainingPageContent />
    </ProtectedRoute>
  );
}

function TrainingPageContent() {
  const [trainingMode, setTrainingMode] = useState<"model" | "tagger" | "vae">("model");

  // Model training state
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [showConfig, setShowConfig] = useState(false);
  const [editRunId, setEditRunId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [preservedConfigKeys, setPreservedConfigKeys] = useState<string[]>([]);

  // Tagger training state
  const [taggerRuns, setTaggerRuns] = useState<TaggerTrainingRun[]>([]);
  const [selectedTaggerRunId, setSelectedTaggerRunId] = useState<string | null>(null);
  const [showTaggerConfig, setShowTaggerConfig] = useState(false);
  const [editingTaggerRun, setEditingTaggerRun] = useState<TaggerTrainingRun | null>(null);
  const [taggerLoading, setTaggerLoading] = useState(true);

  // VAE training state. VAE runs are ordinary TrainingRun rows in the same
  // /training/runs list, so they share `runs` / loadRuns / the polling effect
  // with model training and are split by training_method below.
  const [selectedVaeRunId, setSelectedVaeRunId] = useState<number | null>(null);
  const [showVaeConfig, setShowVaeConfig] = useState(false);
  const [editVaeRunId, setEditVaeRunId] = useState<number | null>(null);

  const [isMobile, setIsMobile] = useState(false);
  const [showMobileDetail, setShowMobileDetail] = useState(false);

  // Detect mobile screen size
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024); // lg breakpoint
    };
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  const loadRuns = useCallback(async () => {
    try {
      const response = await listTrainingRuns();
      console.log(`[TrainingPage] Loaded ${response.runs.length} runs:`, response.runs.map(r => `ID:${r.id} status:${r.status} progress:${r.progress}%`));
      setRuns(response.runs);
    } catch (err) {
      console.error("[TrainingPage] Failed to load training runs:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadTaggerRuns = useCallback(async () => {
    try {
      const data = await listTaggerTrainingRuns();
      setTaggerRuns(data);
    } catch (err) {
      console.error("[TrainingPage] Failed to load tagger runs:", err);
    } finally {
      setTaggerLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  useEffect(() => {
    if (trainingMode === "tagger") {
      loadTaggerRuns();
    }
  }, [trainingMode, loadTaggerRuns]);

  // Poll running model trainings to update list
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const hasRunningTraining = runs.some(r => r.status === "running" || r.status === "starting");

    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }

    if (!hasRunningTraining) return;

    pollingIntervalRef.current = setInterval(() => {
      loadRuns();
    }, 3000);

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [runs, loadRuns]);

  // Poll running tagger trainings to update list
  const taggerPollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const hasRunningTagger = taggerRuns.some(
      r => r.status === "running" || r.status === "starting"
    );

    if (taggerPollingIntervalRef.current) {
      clearInterval(taggerPollingIntervalRef.current);
      taggerPollingIntervalRef.current = null;
    }

    if (!hasRunningTagger) return;

    taggerPollingIntervalRef.current = setInterval(() => {
      loadTaggerRuns();
    }, 3000);

    return () => {
      if (taggerPollingIntervalRef.current) {
        clearInterval(taggerPollingIntervalRef.current);
        taggerPollingIntervalRef.current = null;
      }
    };
  }, [taggerRuns, loadTaggerRuns]);

  // Model training handlers
  const handleCreateRun = () => {
    setSelectedRunId(null);
    setEditRunId(null);
    setShowConfig(true);
    setPreservedConfigKeys([]);
  };

  const handleEditRun = (runId: number) => {
    setEditRunId(runId);
    setShowConfig(true);
    setPreservedConfigKeys([]);
  };

  const handleRunCreated = (newRun: TrainingRun) => {
    setRuns([newRun, ...runs]);
    setShowConfig(false);
    setEditRunId(null);
    setSelectedRunId(newRun.id);
  };

  const handleRunUpdated = (updatedRun: TrainingRun) => {
    setRuns((prevRuns) =>
      prevRuns.map((r) => (r.id === updatedRun.id ? updatedRun : r))
    );
    setShowConfig(false);
    setEditRunId(null);
    setSelectedRunId(updatedRun.id);
    // TrainingConfig unmounts in this same commit (showConfig -> false), so
    // its own preserved-keys notice can never render. Surface it here instead.
    setPreservedConfigKeys(updatedRun.preserved_config_keys ?? []);
  };

  const handleSelectRun = (id: number) => {
    setSelectedRunId(id);
    setShowConfig(false);
    setPreservedConfigKeys([]);
    if (isMobile) setShowMobileDetail(true);
  };

  const handleStatusChange = (updatedRun: TrainingRun) => {
    setRuns((prevRuns) =>
      prevRuns.map((r) => (r.id === updatedRun.id ? updatedRun : r))
    );
  };

  const handleDelete = (deletedRunId: number) => {
    setRuns((prevRuns) => prevRuns.filter((r) => r.id !== deletedRunId));
    setSelectedRunId(null);
  };

  // Tagger training handlers
  const handleCreateTaggerRun = () => {
    setSelectedTaggerRunId(null);
    setShowTaggerConfig(true);
    if (isMobile) setShowMobileDetail(true);
  };

  const handleSelectTaggerRun = (runId: string) => {
    setSelectedTaggerRunId(runId);
    setShowTaggerConfig(false);
    if (isMobile) setShowMobileDetail(true);
  };

  const handleTaggerRunCreated = (newRun: TaggerTrainingRun) => {
    setTaggerRuns((prev) => {
      const exists = prev.some((r) => r.run_id === newRun.run_id);
      return exists ? prev.map((r) => (r.run_id === newRun.run_id ? newRun : r)) : [newRun, ...prev];
    });
    setShowTaggerConfig(false);
    setEditingTaggerRun(null);
    setSelectedTaggerRunId(newRun.run_id);
  };

  const handleTaggerEditConfig = (run: TaggerTrainingRun) => {
    setEditingTaggerRun(run);
    setShowTaggerConfig(true);
  };

  const handleTaggerStatusChange = (updatedRun: TaggerTrainingRun) => {
    setTaggerRuns((prev) =>
      prev.map((r) => (r.run_id === updatedRun.run_id ? updatedRun : r))
    );
  };

  // VAE training handlers
  const handleCreateVaeRun = () => {
    setSelectedVaeRunId(null);
    setEditVaeRunId(null);
    setShowVaeConfig(true);
    if (isMobile) setShowMobileDetail(true);
  };

  const handleEditVaeRun = (runId: number) => {
    setEditVaeRunId(runId);
    setShowVaeConfig(true);
  };

  const handleSelectVaeRun = (id: number) => {
    setSelectedVaeRunId(id);
    setShowVaeConfig(false);
    if (isMobile) setShowMobileDetail(true);
  };

  const handleVaeRunCreated = (newRun: TrainingRun) => {
    setRuns([newRun, ...runs]);
    setShowVaeConfig(false);
    setEditVaeRunId(null);
    setSelectedVaeRunId(newRun.id);
  };

  const handleVaeRunUpdated = (updatedRun: TrainingRun) => {
    setRuns((prevRuns) =>
      prevRuns.map((r) => (r.id === updatedRun.id ? updatedRun : r))
    );
    setShowVaeConfig(false);
    setEditVaeRunId(null);
    setSelectedVaeRunId(updatedRun.id);
  };

  const handleVaeDelete = (deletedRunId: number) => {
    setRuns((prevRuns) => prevRuns.filter((r) => r.id !== deletedRunId));
    setSelectedVaeRunId(null);
    if (isMobile) setShowMobileDetail(false);
  };

  const handleTaggerDelete = (deletedRunId: string) => {
    setTaggerRuns((prev) => prev.filter((r) => r.run_id !== deletedRunId));
    setSelectedTaggerRunId(null);
    if (isMobile) setShowMobileDetail(false);
  };

  // VAE runs are excluded from the model list and vice versa: they are the same
  // row type but have no denoiser, no samples and a different config form.
  const modelRuns = runs.filter(r => r.training_method !== "vae_decoder");
  const vaeRuns = runs.filter(r => r.training_method === "vae_decoder");

  const selectedRun = modelRuns.find(r => r.id === selectedRunId);
  const selectedTaggerRun = taggerRuns.find(r => r.run_id === selectedTaggerRunId);
  const selectedVaeRun = vaeRuns.find(r => r.id === selectedVaeRunId);

  const handleTabChange = (mode: "model" | "tagger" | "vae") => {
    setTrainingMode(mode);
    setShowMobileDetail(false);
    setShowConfig(false);
    setShowTaggerConfig(false);
    setShowVaeConfig(false);
  };

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="app-main compact-workspace flex flex-col overflow-hidden">
        {/* Header */}
        <div className="app-topbar flex-shrink-0">
          <div className="flex items-center justify-between gap-2">
            {/* Mobile: Back button when showing detail */}
            {isMobile && showMobileDetail && (
              <button
                onClick={() => setShowMobileDetail(false)}
                className="mr-1 text-gray-400 hover:text-white transition-colors flex-shrink-0"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
            )}

            {/* Tab buttons */}
            <div className="flex flex-shrink-0 overflow-hidden rounded-md border border-gray-700 bg-gray-900 p-0.5">
              <button
                onClick={() => handleTabChange("model")}
                className={`px-3 py-1.5 text-xs sm:text-sm transition-colors ${
                  trainingMode === "model"
                    ? "bg-violet-600 text-white"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
                }`}
              >
                {isMobile ? "Model" : "Model Training"}
              </button>
              <button
                onClick={() => handleTabChange("tagger")}
                className={`px-3 py-1.5 text-xs sm:text-sm transition-colors border-l border-gray-600 ${
                  trainingMode === "tagger"
                    ? "bg-violet-600 text-white"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
                }`}
              >
                {isMobile ? "Tagger" : "Tagger Training"}
              </button>
              <button
                onClick={() => handleTabChange("vae")}
                className={`px-3 py-1.5 text-xs sm:text-sm transition-colors border-l border-gray-600 ${
                  trainingMode === "vae"
                    ? "bg-violet-600 text-white"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
                }`}
              >
                {isMobile ? "VAE" : "VAE Training"}
              </button>
            </div>

            <div className="flex-1" />

            {/* New run button */}
            {trainingMode === "model" ? (
              <button
                onClick={handleCreateRun}
                className="whitespace-nowrap rounded-md border border-violet-400/30 bg-violet-600 px-2 py-1.5 text-xs transition-colors hover:bg-violet-500 sm:px-3"
              >
                {isMobile ? "New" : "New Training Run"}
              </button>
            ) : trainingMode === "tagger" ? (
              <button
                onClick={handleCreateTaggerRun}
                className="whitespace-nowrap rounded-md border border-violet-400/30 bg-violet-600 px-2 py-1.5 text-xs transition-colors hover:bg-violet-500 sm:px-3"
              >
                {isMobile ? "New" : "New Tagger Run"}
              </button>
            ) : (
              <button
                onClick={handleCreateVaeRun}
                className="whitespace-nowrap rounded-md border border-violet-400/30 bg-violet-600 px-2 py-1.5 text-xs transition-colors hover:bg-violet-500 sm:px-3"
              >
                {isMobile ? "New" : "New VAE Run"}
              </button>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex">
          {trainingMode === "model" ? (
            <>
              {/* Model Training Runs List */}
              <div className={`${isMobile && showMobileDetail ? 'hidden' : 'flex'} ${isMobile ? 'w-full' : 'w-60 lg:w-72'} flex-shrink-0 ${!isMobile && 'border-r border-gray-800'} overflow-y-auto`}>
                <TrainingList
                  runs={modelRuns}
                  selectedRunId={selectedRunId}
                  onSelectRun={handleSelectRun}
                  onRefresh={loadRuns}
                  loading={loading}
                />
              </div>

              {/* Model Training Config or Monitor */}
              <div className={`${isMobile && !showMobileDetail ? 'hidden' : 'flex-1'} overflow-y-auto`}>
                {!showConfig && preservedConfigKeys.length > 0 && (
                  <div className="m-3 sm:m-4 flex items-start justify-between gap-2 rounded border border-blue-500 bg-blue-900/20 p-2.5 sm:p-3 text-xs sm:text-sm text-blue-300">
                    <span>Kept config-only keys this form cannot edit: {preservedConfigKeys.join(", ")}</span>
                    <button
                      type="button"
                      onClick={() => setPreservedConfigKeys([])}
                      className="flex-shrink-0 text-blue-300 hover:text-white"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                )}
                {showConfig ? (
                  <TrainingConfig
                    onClose={() => {
                      setShowConfig(false);
                      setEditRunId(null);
                      if (isMobile) setShowMobileDetail(false);
                    }}
                    onRunCreated={handleRunCreated}
                    editRunId={editRunId}
                    onRunUpdated={handleRunUpdated}
                  />
                ) : selectedRun ? (
                  <TrainingMonitor
                    key={selectedRun.id}
                    run={selectedRun}
                    onClose={() => {
                      setSelectedRunId(null);
                      if (isMobile) setShowMobileDetail(false);
                    }}
                    onStatusChange={handleStatusChange}
                    onDelete={() => handleDelete(selectedRun.id)}
                    onEditConfig={() => handleEditRun(selectedRun.id)}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    <div className="text-center p-4">
                      <p className="text-base sm:text-lg font-medium">No training run selected</p>
                      <p className="text-xs sm:text-sm mt-2">Select a run from the list or create a new one</p>
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : trainingMode === "vae" ? (
            <>
              {/* VAE Training Runs List (same TrainingRun rows, filtered) */}
              <div className={`${isMobile && showMobileDetail ? 'hidden' : 'flex'} ${isMobile ? 'w-full' : 'w-60 lg:w-72'} flex-shrink-0 ${!isMobile && 'border-r border-gray-800'} overflow-y-auto`}>
                <TrainingList
                  runs={vaeRuns}
                  selectedRunId={selectedVaeRunId}
                  onSelectRun={handleSelectVaeRun}
                  onRefresh={loadRuns}
                  loading={loading}
                />
              </div>

              {/* VAE Training Config or Monitor */}
              <div className={`${isMobile && !showMobileDetail ? 'hidden' : 'flex-1'} overflow-y-auto`}>
                {showVaeConfig ? (
                  <VaeTrainingConfig
                    key={editVaeRunId ?? "new"}
                    onClose={() => {
                      setShowVaeConfig(false);
                      setEditVaeRunId(null);
                      if (isMobile) setShowMobileDetail(false);
                    }}
                    onRunCreated={handleVaeRunCreated}
                    onRunUpdated={handleVaeRunUpdated}
                    editRunId={editVaeRunId}
                  />
                ) : selectedVaeRun ? (
                  <TrainingMonitor
                    key={selectedVaeRun.id}
                    run={selectedVaeRun}
                    onClose={() => {
                      setSelectedVaeRunId(null);
                      if (isMobile) setShowMobileDetail(false);
                    }}
                    onStatusChange={handleStatusChange}
                    onDelete={() => handleVaeDelete(selectedVaeRun.id)}
                    onEditConfig={() => handleEditVaeRun(selectedVaeRun.id)}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    <div className="text-center p-4">
                      <p className="text-base sm:text-lg font-medium">No VAE training run selected</p>
                      <p className="text-xs sm:text-sm mt-2">Select a run from the list or create a new one</p>
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <>
              {/* Tagger Training Runs List */}
              <div className={`${isMobile && showMobileDetail ? 'hidden' : 'flex'} ${isMobile ? 'w-full' : 'w-60 lg:w-72'} flex-shrink-0 ${!isMobile && 'border-r border-gray-800'} overflow-y-auto flex-col`}>
                {taggerLoading && taggerRuns.length === 0 ? (
                  <div className="flex items-center justify-center p-8 text-gray-400 text-sm">
                    Loading...
                  </div>
                ) : taggerRuns.length === 0 ? (
                  <div className="flex items-center justify-center p-8 text-gray-400 text-sm text-center">
                    No tagger runs yet.<br />Create a new one to get started.
                  </div>
                ) : (
                  <div className="flex-1">
                    {taggerRuns.map((run) => (
                      <button
                        key={run.run_id}
                        onClick={() => handleSelectTaggerRun(run.run_id)}
                        className={`w-full text-left px-4 py-3 border-b border-gray-700 hover:bg-gray-700/50 transition-colors ${
                          selectedTaggerRunId === run.run_id ? "bg-gray-700" : ""
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-sm font-medium truncate">{run.run_name}</span>
                          <span className={`text-xs px-1.5 py-0.5 rounded flex-shrink-0 ${
                            run.status === "running" ? "bg-blue-600 text-white" :
                            run.status === "completed" ? "bg-green-700 text-white" :
                            run.status === "failed" ? "bg-red-700 text-white" :
                            run.status === "stopped" ? "bg-yellow-700 text-white" :
                            "bg-gray-600 text-gray-300"
                          }`}>
                            {run.status}
                          </span>
                        </div>
                        <div className="text-xs text-gray-400 mt-1">
                          {run.training_method.toUpperCase()} · {run.num_tags} tags
                          {run.best_f1 !== null && ` · F1: ${run.best_f1.toFixed(3)}`}
                        </div>
                        {(run.status === "running" || run.status === "starting") && (
                          <div className="mt-1.5 w-full bg-gray-600 rounded-full h-1">
                            <div
                              className="bg-blue-500 h-1 rounded-full transition-all"
                              style={{ width: `${run.progress * 100}%` }}
                            />
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Tagger Training Config or Monitor */}
              <div className={`${isMobile && !showMobileDetail ? 'hidden' : 'flex-1'} overflow-y-auto`}>
                {showTaggerConfig ? (
                  <TaggerTrainingConfig
                    key={editingTaggerRun?.run_id ?? "new"}
                    onClose={() => {
                      setShowTaggerConfig(false);
                      setEditingTaggerRun(null);
                      if (isMobile) setShowMobileDetail(false);
                    }}
                    onRunCreated={handleTaggerRunCreated}
                    editRun={editingTaggerRun ?? undefined}
                  />
                ) : selectedTaggerRun ? (
                  <TaggerTrainingMonitor
                    key={selectedTaggerRun.run_id}
                    run={selectedTaggerRun}
                    onClose={() => {
                      setSelectedTaggerRunId(null);
                      if (isMobile) setShowMobileDetail(false);
                    }}
                    onStatusChange={handleTaggerStatusChange}
                    onDelete={() => handleTaggerDelete(selectedTaggerRun.run_id)}
                    onEditConfig={() => handleTaggerEditConfig(selectedTaggerRun)}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    <div className="text-center p-4">
                      <p className="text-base sm:text-lg font-medium">No tagger run selected</p>
                      <p className="text-xs sm:text-sm mt-2">Select a run from the list or create a new one</p>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
