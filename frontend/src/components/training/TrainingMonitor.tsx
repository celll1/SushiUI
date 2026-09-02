"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { X, Play, Square, Trash2, AlertTriangle } from "lucide-react";
import { TrainingRun, TrainingLogEvent, getTrainingRun, getTrainingStatus, startTrainingRun, stopTrainingRun, deleteTrainingRun, updateTrainingConfig, reloadTrainingConfig, getTrainingSamples, TrainingSampleStep, getDebugLatents, DebugLatent, visualizeDebugLatent, DebugLatentVisualization, skipTrainingRescan, queueTrainingSample, getTrainingSampleQueue, TrainingSampleQueueResponse, trainingFeatureUnsupportedReason } from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";
import { wsClient, DatasetScanProgress, TrainingLogMessage } from "@/utils/websocket";
import { TrainingMetricsProvider } from "./TrainingMetricsContext";
import TrainingMetricsChart from "./TrainingMetricsChart";
import ResizableChartRow, { ChartPaneCount, useChartLayout } from "./ResizableChartRow";
import DanbooruImageMetricsPanel from "./DanbooruImageMetricsPanel";
import CheckpointList from "./CheckpointList";
import ImageViewer from "../common/ImageViewer";

interface TrainingMonitorProps {
  run: TrainingRun;
  onClose: () => void;
  onStatusChange: (updatedRun: TrainingRun) => void;
  onDelete?: () => void;
  onEditConfig?: () => void;
}

interface TrainingCadence {
  batchSize: number;
  mnt: number;
  optimizerEvery: number;
  fused: boolean;
  eviction: boolean;
}

const yamlScalar = (yaml: string | undefined, key: string): string | undefined => {
  if (!yaml) return undefined;
  return yaml.match(new RegExp(`^\\s*${key}:\\s*([^#\\r\\n]+)`, "m"))?.[1]?.trim();
};

const yamlInt = (yaml: string | undefined, key: string, fallback: number): number => {
  const value = Number.parseInt(yamlScalar(yaml, key) ?? "", 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
};

const yamlBool = (yaml: string | undefined, key: string): boolean =>
  ["true", "1", "yes", "on"].includes((yamlScalar(yaml, key) ?? "").toLowerCase());

const formatIterationRate = (seconds: number | null): string => {
  if (seconds === null || !Number.isFinite(seconds) || seconds <= 0) return "Calculating...";
  if (seconds < 1) return `${(1 / seconds).toFixed(2)} iter/s`;
  return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s/iter`;
};

const formatSeconds = (seconds: number): string =>
  `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`;

// Per-pane persistence slot and the preset each pane opens on. Pane 3 defaults
// to the param-change view, which is why ParamChangeChart no longer exists as a
// component -- its two tabs are two presets now.
const PANE_SLOTS = ["a", "b", "c"] as const;
const PANE_DEFAULT_PRESETS = ["loss-overview", "gradient-norms", "param-update"];

export default function TrainingMonitor({ run, onClose, onStatusChange, onDelete, onEditConfig }: TrainingMonitorProps) {
  const { layout: chartLayout, setLayout: setChartLayout, setPanes: setChartPanes } = useChartLayout();
  const { archCapabilities } = useStartup();
  const [currentRun, setCurrentRun] = useState<TrainingRun>(run);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [samples, setSamples] = useState<TrainingSampleStep[]>([]);
  const [sampleQueue, setSampleQueue] = useState<TrainingSampleQueueResponse | null>(null);
  const [isQueueingSample, setIsQueueingSample] = useState(false);
  const [sampleQueueError, setSampleQueueError] = useState<string | null>(null);
  // Index into the flattened sample list (see sampleImages) rather than a URL,
  // so the enlarged view can walk across step boundaries and drag the slider
  // with it.
  const [viewerIndex, setViewerIndex] = useState<number | null>(null);
  const [selectedStepIndex, setSelectedStepIndex] = useState<number>(0); // For step slider
  // Epoch lives only on the status response (there is no epoch column on the
  // run row), so it is held here rather than folded into currentRun.
  const [epochInfo, setEpochInfo] = useState<{ current: number | null; total: number | null }>({ current: null, total: null });

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
  const [recentSecondsPerIteration, setRecentSecondsPerIteration] = useState<number | null>(null);
  const speedWindowRef = useRef<Array<{ step: number; at: number }>>([]);

  // Dataset scan progress (drift detection / rescan) — shown until training proper starts
  const [scanMessage, setScanMessage] = useState<string | null>(null);
  // Dataset currently being rescanned (drift_walk / rescan phase) — drives the
  // "Skip rescan" button. null when no skippable scan is in progress.
  const [scanDatasetId, setScanDatasetId] = useState<number | null>(null);
  const [scanSkipping, setScanSkipping] = useState(false);

  // Structured notices from the trainer (settings overridden or ignored). The
  // WebSocket replays nothing on connect, so the backlog comes from the status
  // response and live events are merged into it by (level, code, message).
  const [notices, setNotices] = useState<TrainingLogEvent[]>(run.warnings ?? []);
  const [noticesOpen, setNoticesOpen] = useState(true);

  // JSON rather than a delimiter: no separator can collide with message text.
  const noticeKey = (n: TrainingLogEvent) =>
    JSON.stringify([n.level, n.code ?? null, n.message]);

  const mergeNotices = (incoming: TrainingLogEvent[]) => {
    setNotices((prev) => {
      const seen = new Set(prev.map(noticeKey));
      const added = incoming.filter((n) => !seen.has(noticeKey(n)));
      return added.length ? [...prev, ...added] : prev;
    });
  };

  // Epoch is not on the run row, so fetch it once on open (covers finished runs
  // and the gap before the first poll tick). Same call carries the notice backlog.
  useEffect(() => {
    let cancelled = false;
    getTrainingRun(currentRun.id)
      .then((detail) => {
        if (!cancelled && detail.config_yaml) {
          setCurrentRun((prev) => ({ ...prev, config_yaml: detail.config_yaml }));
        }
      })
      .catch(() => {});
    getTrainingStatus(currentRun.id)
      .then((status) => {
        if (!cancelled) {
          setEpochInfo({ current: status.current_epoch ?? null, total: status.total_epochs ?? null });
          if (status.warnings?.length) mergeNotices(status.warnings);
        }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [currentRun.id]);

  useEffect(() => {
    speedWindowRef.current = [];
    setRecentSecondsPerIteration(null);
  }, [currentRun.id, currentRun.last_resumed_at]);

  // Live notices for this run.
  useEffect(() => {
    const handler = (ev: TrainingLogMessage) => {
      if (Number(ev.run_id) !== Number(currentRun.id)) return;
      mergeNotices([{ level: ev.level, code: ev.code ?? null, message: ev.message }]);
    };
    wsClient.subscribeToTrainingLog(handler);
    return () => wsClient.unsubscribeFromTrainingLog(handler);
  }, [currentRun.id]);

  // A (re)start clears the run's notices server-side: they describe the attempt
  // that emitted them.
  useEffect(() => {
    if (currentRun.status === "starting") setNotices([]);
  }, [currentRun.status]);

  // Poll training status
  useEffect(() => {
    if (currentRun.status !== "starting" && currentRun.status !== "running") {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const status = await getTrainingStatus(currentRun.id);
        setEpochInfo({ current: status.current_epoch ?? null, total: status.total_epochs ?? null });
        // Also the recovery path when the SSE connection dropped and reconnected.
        if (status.warnings?.length) mergeNotices(status.warnings);
        const now = performance.now();
        if (status.status === "running" && status.phase === "training") {
          const window = speedWindowRef.current;
          if (window.length && status.current_step < window[window.length - 1].step) {
            window.length = 0;
          }
          window.push({ step: status.current_step, at: now });
          while (window.length > 2 && now - window[0].at > 20_000) window.shift();
          const first = window[0];
          const advanced = status.current_step - first.step;
          const elapsed = now - first.at;
          setRecentSecondsPerIteration(
            advanced > 0 && elapsed > 0 ? elapsed / 1000 / advanced : null
          );
        } else if (status.status === "running" && status.phase === "sampling") {
          // A sample doesn't advance training steps — leave the window and the
          // displayed speed as they were rather than feeding it a non-step
          // sample or blanking the readout for the duration of the render.
        } else {
          speedWindowRef.current = [];
          setRecentSecondsPerIteration(null);
        }
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
  }, [currentRun.status, currentRun.id, currentRun.config_yaml]);

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

  // A VAE fine-tune has no sample-image concept at all: it has no denoiser and
  // no prompt to generate from, and its quality signal is the validation
  // PSNR / blockiness chart. Nothing ever lands in samples/, so neither the
  // initial fetch nor the 5s poll has anything to find.
  const hasSampleImages = currentRun.training_method !== "vae_decoder";

  // Every sample image in listing order, carrying the step it belongs to. The
  // backend returns steps ascending and only ever appends, so an index stays
  // pointing at the same image across the 5s reload.
  const sampleImages = useMemo(
    () => samples.flatMap((s, stepIndex) => s.images.map((img) => ({ path: img.path, stepIndex }))),
    [samples]
  );
  const viewerImage = viewerIndex === null ? undefined : sampleImages[viewerIndex];

  const navigateSample = useCallback((direction: "prev" | "next") => {
    setViewerIndex((prev) => {
      if (prev === null) return prev;
      const next = prev + (direction === "next" ? 1 : -1);
      return next >= 0 && next < sampleImages.length ? next : prev;
    });
  }, [sampleImages.length]);

  // Keep the step slider on whatever the enlarged view is showing.
  useEffect(() => {
    if (viewerImage) setSelectedStepIndex(viewerImage.stepIndex);
  }, [viewerImage]);

  // Load sample images
  useEffect(() => {
    if (!hasSampleImages) {
      setSamples([]);
      return;
    }
    loadSamples();

    // Reload samples every 5 seconds when running
    if (currentRun.status === "running") {
      const interval = setInterval(loadSamples, 5000);
      return () => clearInterval(interval);
    }
  }, [currentRun.id, currentRun.status, hasSampleImages]);

  // One tick for both: the image itself is observed through the samples
  // listing, the queue call only reports what is pending / what failed.
  const loadSamples = async () => {
    try {
      const data = await getTrainingSamples(currentRun.id);
      setSamples(data.samples);
    } catch (err) {
      console.error("Failed to load sample images:", err);
    }
    try {
      setSampleQueue(await getTrainingSampleQueue(currentRun.id));
    } catch (err) {
      console.error("Failed to load sample queue:", err);
    }
  };

  const samplesUnsupportedReason =
    trainingFeatureUnsupportedReason(
      archCapabilities, sampleQueue?.architecture, "training_samples",
      currentRun.training_method
    ) ?? sampleQueue?.unsupported_reason ?? undefined;

  // sampleQueue === null until the first fetch lands, and the arch gate reads
  // its `architecture` — enabling the button before then would offer it on an
  // architecture that cannot sample.
  const canQueueSample =
    hasSampleImages && sampleQueue !== null && !samplesUnsupportedReason &&
    currentRun.status === "running";

  const handleQueueSample = async () => {
    setIsQueueingSample(true);
    setSampleQueueError(null);
    try {
      await queueTrainingSample(currentRun.id);
      setSampleQueue(await getTrainingSampleQueue(currentRun.id));
    } catch (err: any) {
      setSampleQueueError(
        err?.response?.data?.detail || err?.message || "Failed to queue sample"
      );
    } finally {
      setIsQueueingSample(false);
    }
  };

  // Subscribe to dataset scan progress (drift check / rescan / cleanup) over WS
  useEffect(() => {
    const handler = (ev: DatasetScanProgress) => {
      if (ev.scope !== "training") return;
      if (String(ev.run_id) !== String(currentRun.id)) return;
      // "MyDataset (#25)" when a name is known, else "dataset 25".
      const dsLabel = ev.dataset_name ? `${ev.dataset_name} (#${ev.dataset_id})` : `dataset ${ev.dataset_id}`;
      // scan_start/scan_end bracket the skippable window — the Skip button is
      // shown only while scanDatasetId is set (between these two events).
      if (ev.phase === "scan_start") {
        setScanDatasetId(ev.dataset_id);
        setScanSkipping(false);
        setScanMessage(`Checking ${dsLabel}...`);
        return;
      }
      if (ev.phase === "scan_end") {
        setScanDatasetId((cur) => (cur === ev.dataset_id ? null : cur));
        return;
      }
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
    if (currentRun.phase === "training" || currentRun.phase === "bucketing" || currentRun.phase === "crop_precompute" || currentRun.phase === "latent_cache" || currentRun.phase === "text_encoder_cache") {
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
      return { elapsed: "N/A", eta: "N/A", averageSecondsPerIteration: null };
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
      return { elapsed, eta: "N/A", averageSecondsPerIteration: null };
    }

    // Calculate progress based on steps completed since resume (or from start)
    const startStep = currentRun.resumed_from_step ?? 0;  // Step at resume (or 0 if first start)
    const stepsCompleted = currentRun.current_step - startStep;
    const remainingSteps = currentRun.total_steps - currentRun.current_step;

    if (stepsCompleted <= 0) {
      return { elapsed, eta: "Calculating...", averageSecondsPerIteration: null };
    }

    if (remainingSteps <= 0) {
      return { elapsed, eta: "Completed", averageSecondsPerIteration: elapsedMs / stepsCompleted / 1000 };
    }

    // ETA = (elapsed time / steps completed) * remaining steps
    const averageSecondsPerIteration = elapsedMs / stepsCompleted / 1000;
    const etaSecondsPerIteration = recentSecondsPerIteration ?? averageSecondsPerIteration;
    const remainingMs = etaSecondsPerIteration * 1000 * remainingSteps;

    // Format ETA
    const remainingSeconds = Math.floor(remainingMs / 1000);
    const etaHours = Math.floor(remainingSeconds / 3600);
    const etaMinutes = Math.floor((remainingSeconds % 3600) / 60);
    const etaSecs = remainingSeconds % 60;
    const eta = `${etaHours}h ${etaMinutes}m ${etaSecs}s`;

    return { elapsed, eta, averageSecondsPerIteration };
  };

  const timeInfo = calculateTimeInfo();
  const cadence = useMemo<TrainingCadence>(() => {
    const yaml = currentRun.config_yaml;
    const optimizer = (yamlScalar(yaml, "optimizer") ?? "").toLowerCase();
    const blocksToSwap = yamlInt(yaml, "blocks_to_swap", 0);
    const optimizerGroups = yamlInt(yaml, "num_optimizer_groups", 0);
    const isSenseNovaFull = currentRun.training_method === "full_finetune" &&
      currentRun.base_model_path.toLowerCase().includes("sensenova");
    const fusedOptimizers = new Set([
      "adafactor", "adamw8bit", "paged_adamw8bit", "lion8bit", "paged_lion8bit",
      "adamw8bit_ringbuffer", "lion8bit_ringbuffer",
    ]);
    const fused = optimizerGroups > 0 ||
      (blocksToSwap > 0 && fusedOptimizers.has(optimizer)) ||
      (isSenseNovaFull && optimizer === "adafactor");
    const accumulation = yamlInt(yaml, "gradient_accumulation_steps", 1);
    return {
      batchSize: yamlInt(yaml, "batch_size", 1),
      mnt: yamlInt(yaml, "multi_noise_timesteps", 1),
      optimizerEvery: fused ? 1 : accumulation,
      fused,
      eviction: yamlBool(yaml, "sensenova_mot_phase_eviction") ||
        yamlBool(yaml, "sensenova_four_phase_eviction"),
    };
  }, [currentRun.config_yaml, currentRun.training_method, currentRun.base_model_path]);
  const recentSpeedLabel = recentSecondsPerIteration !== null
    ? formatIterationRate(recentSecondsPerIteration)
    : speedWindowRef.current.length > 1
      ? "No iteration in window"
      : "Calculating...";

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-gray-800 bg-gray-900/50 px-3">
        <div className="min-w-0">
          <p className="app-kicker">Training monitor</p>
          <h2 className="truncate text-sm font-semibold">
            <span className="mr-1.5 font-mono text-xs text-gray-500">#{currentRun.id}</span>
            {currentRun.run_name}
          </h2>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 hover:bg-gray-700 rounded transition-colors flex-shrink-0"
          aria-label="Close training monitor"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Main Content - Responsive Layout */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
        {/* Left Panel - Training Info (internal scroll on desktop; flows on mobile) */}
        <div className="flex-1 space-y-3 p-3 lg:overflow-y-auto">
          <div className="grid items-start gap-3 2xl:grid-cols-[minmax(0,1.1fr)_minmax(340px,0.9fr)]">
            <div className="space-y-2">
          {/* Status */}
          <div className="rounded-md border border-gray-700 bg-gray-800/80 p-3">
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
                  {currentRun.phase === "bucketing" && "Assigning buckets"}
                  {currentRun.phase === "crop_precompute" && "Planning crop schedule"}
                  {currentRun.phase === "latent_cache" && "Latent Cache"}
                  {currentRun.phase === "text_encoder_cache" && "Text Encoder Cache"}
                  {currentRun.phase === "training" && `Iteration ${currentRun.current_step} / ${currentRun.total_steps}`}
                  {currentRun.phase === "sampling" && "Rendering sample"}
                  {currentRun.phase === "initializing" && "Initializing..."}
                  {!currentRun.phase && `Iteration ${currentRun.current_step} / ${currentRun.total_steps}`}
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
                  className={`h-2 rounded-full transition-all ${
                    currentRun.phase === "sampling" ? "bg-yellow-500" : "bg-blue-600"
                  }`}
                  style={{
                    width: currentRun.phase === "training" || !currentRun.phase
                      ? `${currentRun.progress}%`
                      : `${currentRun.phase_progress || 0}%`
                  }}
                />
              </div>
            </div>

            {/* Metrics */}
            <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-xs [&>div]:whitespace-nowrap">
              {epochInfo.current !== null && (
                <div>
                  <span className="text-gray-400">Epoch:</span>{" "}
                  <span className="font-mono">
                    {epochInfo.total ? `${epochInfo.current} / ${epochInfo.total}` : epochInfo.current}
                  </span>
                </div>
              )}
              <div>
                <span className="text-gray-400">Loss:</span>{" "}
                <span className="font-mono">{currentRun.loss?.toFixed(6) || "N/A"}</span>
              </div>
              <div>
                <span className="text-gray-400">LR:</span>{" "}
                <span className="font-mono">{currentRun.learning_rate?.toExponential(2) || "N/A"}</span>
              </div>
              <div title="Rolling wall-clock throughput over the latest 20 seconds. One iteration is one forward/backward at one MNT timestep.">
                <span className="text-gray-400">Speed:</span>{" "}
                <span className="font-mono text-cyan-400">{recentSpeedLabel}</span>
                {timeInfo.averageSecondsPerIteration !== null && (
                  <span className="ml-1 text-gray-500">(avg {formatIterationRate(timeInfo.averageSecondsPerIteration)})</span>
                )}
              </div>
              <div>
                <span className="text-gray-400">Elapsed:</span>{" "}
                <span className="font-mono text-blue-400">{timeInfo.elapsed}</span>
              </div>
              <div>
                <span className="text-gray-400">ETA:</span>{" "}
                <span className="font-mono text-green-400">{timeInfo.eta}</span>
              </div>
            </div>
            {currentRun.phase === "training" && recentSecondsPerIteration !== null && (
              <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 border-t border-gray-700/70 pt-2 text-xxs text-gray-400 [&>span]:whitespace-nowrap">
                {cadence.mnt > 1 && cadence.optimizerEvery === cadence.mnt ? (
                  <span title="The MNT window and gradient-accumulation window end at the same iteration.">
                    Input batch/update wall: <span className="font-mono text-gray-300">≈{formatSeconds(recentSecondsPerIteration * cadence.mnt)}</span>
                    {` (MNT×${cadence.mnt})`}
                  </span>
                ) : (
                  <>
                    {cadence.mnt > 1 && (
                      <span title="One dataset batch is reused for each configured MNT timestep.">
                        Input-batch wall: <span className="font-mono text-gray-300">≈{formatSeconds(recentSecondsPerIteration * cadence.mnt)}</span>
                        {` (MNT×${cadence.mnt})`}
                      </span>
                    )}
                    {cadence.optimizerEvery > 1 && (
                      <span title="Derived wall throughput per update cadence, not isolated optimizer kernel time.">
                        Wall/update: <span className="font-mono text-gray-300">≈{formatSeconds(recentSecondsPerIteration * cadence.optimizerEvery)}</span>
                        {` / ${cadence.optimizerEvery} iters`}
                      </span>
                    )}
                  </>
                )}
                {cadence.fused && (
                  <span title="Fused backward applies parameter updates during every backward; configured gradient accumulation is not effective.">
                    Update cadence: every iter (fused)
                  </span>
                )}
                <span title="Sample presentations contributing to one optimizer update. With MNT, the same input batch is presented at multiple timesteps.">
                  Sample passes/update: <span className="font-mono text-gray-300">{cadence.batchSize * cadence.optimizerEvery}</span>
                </span>
                {cadence.eviction && <span title="Eviction transfers and split phases are already included in wall-clock iteration time.">Eviction included</span>}
              </div>
            )}
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

          {/* Trainer notices — settings this run overrode or ignored. Backlog
              from the status response, live ones over the WebSocket. */}
          {notices.length > 0 && (
            <div className="rounded-md border border-yellow-700/60 bg-yellow-950/20 p-3">
              <button
                onClick={() => setNoticesOpen((v) => !v)}
                className="flex w-full items-center justify-between gap-2 text-left"
              >
                <span className="flex items-center gap-1.5 text-xs sm:text-sm font-medium text-yellow-300">
                  <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0" />
                  Trainer notices ({notices.length})
                </span>
                <span className="text-xxs text-gray-400">{noticesOpen ? "Hide" : "Show"}</span>
              </button>
              {noticesOpen && (
                <ul className="mt-2 space-y-2">
                  {notices.map((n, i) => (
                    <li key={`${n.code ?? "-"}-${i}`} className="text-xxs sm:text-xs">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span
                          className={`rounded px-1 py-0.5 font-medium ${
                            n.level === "error"
                              ? "bg-red-900/60 text-red-300"
                              : n.level === "warning"
                              ? "bg-yellow-900/60 text-yellow-300"
                              : "bg-gray-700 text-gray-300"
                          }`}
                        >
                          {n.level.toUpperCase()}
                        </span>
                        {n.code && <span className="font-mono text-gray-500">{n.code}</span>}
                      </div>
                      <p className="mt-1 whitespace-pre-wrap text-gray-300">{n.message}</p>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
            </div>

          {/* Configuration Info */}
          <div className="space-y-2 rounded-md border border-gray-700 bg-gray-800/80 p-3 text-xs">
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
                {currentRun.training_method === "vae_decoder" ? (
                  <span>VAE Decoder</span>
                ) : (
                  <span className="capitalize">{currentRun.training_method}</span>
                )}
              </div>
              <div>
                <span className="text-gray-400">Total Steps:</span> {currentRun.total_steps}
              </div>
              <div className="col-span-2 min-w-0">
                <span className="text-gray-400">Model:</span>{" "}
                <span className="block truncate font-mono text-[10px]" title={currentRun.base_model_path}>{currentRun.base_model_path}</span>
              </div>
              <div className="col-span-2 min-w-0">
                <span className="text-gray-400">Output:</span>{" "}
                <span className="block truncate font-mono text-[10px]" title={currentRun.output_dir}>{currentRun.output_dir}</span>
              </div>
            </div>
          </div>
          </div>

          {/* Loss Chart */}
          {(currentRun.status === "running" || currentRun.status === "completed" || currentRun.status === "failed" || currentRun.status === "stopped") && (
            <TrainingMetricsProvider key={currentRun.id} runId={currentRun.id} isRunning={currentRun.status === "running"}>
              {/* Two panes, not one: a third scale group is refused rather
                  than crammed onto a shared axis, and the second pane is the
                  escape hatch that makes that refusal workable. The pair is
                  resizable as a unit -- panes that could differ in height stop
                  being comparable at a glance, which is why there are two. */}
              <div className="flex items-center justify-end mb-1">
                <ChartPaneCount panes={chartLayout.panes} onChange={setChartPanes} />
              </div>
              <ResizableChartRow
                layout={chartLayout}
                onLayoutChange={setChartLayout}
                renderPane={(i, h) => (
                  <TrainingMetricsChart
                    slot={PANE_SLOTS[i]}
                    defaultPreset={PANE_DEFAULT_PRESETS[i]}
                    height={h}
                  />
                )}
              />
              <div className="grid grid-cols-1 gap-3 min-[1800px]:grid-cols-2 [&>*]:min-w-0">
                <DanbooruImageMetricsPanel runId={currentRun.id} active={currentRun.status === "running"} />
              </div>
            </TrainingMetricsProvider>
          )}

          {/* Checkpoint List */}
          {(currentRun.status === "running" || currentRun.status === "completed" || currentRun.status === "failed" || currentRun.status === "stopped") && (
            <CheckpointList checkpoints={currentRun.checkpoint_paths} runId={currentRun.id} />
          )}
        </div>

        {/* Right Panel - Sample Images / Debug Latents - Stacked on mobile, side-by-side on desktop */}
        <div className="flex w-full flex-col border-t border-gray-700 lg:w-72 lg:border-l lg:border-t-0 2xl:w-80">
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

          {/* Tab Content — internal scroll on desktop; on mobile the whole content area
              (parent) scrolls so nothing below (e.g. the comparison slider) gets clipped. */}
          <div className="flex-1 lg:overflow-y-auto p-3 sm:p-4 space-y-2.5 sm:space-y-3">
            {viewMode === "samples" ? (
              <>
                {hasSampleImages && (
                  <div className="space-y-1.5 rounded border border-gray-700 bg-gray-800/60 p-2">
                    <button
                      onClick={handleQueueSample}
                      disabled={!canQueueSample || isQueueingSample}
                      title={
                        samplesUnsupportedReason
                          ? samplesUnsupportedReason
                          : currentRun.status !== "running"
                          ? "Only a running training run can render a sample."
                          : sampleQueue === null
                          ? "Checking whether this run can render samples..."
                          : undefined
                      }
                      className="w-full px-2 py-1.5 bg-blue-700 hover:bg-blue-600 rounded text-xxs sm:text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isQueueingSample ? "Queueing..." : "Sample now"}
                    </button>
                    <p className="text-xxs leading-relaxed text-gray-400">
                      Renders the run&apos;s configured sample prompts without waiting for
                      the next scheduled step. The request runs at the end of the batch
                      the trainer is currently in, so it appears whenever that batch
                      finishes — for a large model or batch that can be minutes.
                    </p>
                    {samplesUnsupportedReason && (
                      <p className="text-xxs leading-relaxed text-yellow-400">
                        {samplesUnsupportedReason}
                      </p>
                    )}
                    {sampleQueueError && (
                      <p className="text-xxs leading-relaxed text-red-400">{sampleQueueError}</p>
                    )}
                    {currentRun.phase === "sampling" && (
                      <p className="text-xxs text-yellow-400">
                        Rendering now{currentRun.phase_detail ? `: ${currentRun.phase_detail}` : "..."}
                      </p>
                    )}
                    {!!sampleQueue?.pending?.length && (
                      <p className="text-xxs text-gray-300">
                        {sampleQueue.pending.length} queued (max {sampleQueue.max_pending})
                      </p>
                    )}
                    {sampleQueue?.results
                      ?.filter((r) => !r.ok)
                      .slice(0, 3)
                      .map((r) => (
                        <p key={r.request_id} className="text-xxs leading-relaxed text-red-400">
                          Sample at step {r.step} failed: {r.error ?? "unknown reason"}
                        </p>
                      ))}
                    {sampleQueue?.results?.[0]?.notes?.map((note, i) => (
                      <p key={i} className="text-xxs leading-relaxed text-gray-500">
                        {note}
                      </p>
                    ))}
                  </div>
                )}
                {samples.length === 0 ? (
                  <div className="text-gray-500 text-xs sm:text-sm text-center py-8">
                    {hasSampleImages ? (
                      "No samples generated yet"
                    ) : (
                      <>
                        This training method does not generate sample images.
                        <br />
                        Reconstruction quality is tracked by the validation
                        PSNR / blockiness chart.
                      </>
                    )}
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
                          {p.negative_prompt && (
                            <div className="text-gray-400 leading-relaxed border-b border-gray-700 pb-1.5 mb-1">
                              <span className="text-gray-500">Negative Prompt:</span>{" "}
                              <span className="italic">
                                {p.negative_prompt.length > 120
                                  ? `${p.negative_prompt.substring(0, 120)}…`
                                  : p.negative_prompt}
                              </span>
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
                          onDoubleClick={() =>
                            setViewerIndex(sampleImages.findIndex((s) => s.path === img.path))
                          }
                        >
                          <img
                            src={img.path}
                            alt={`Step ${samples[selectedStepIndex].step} Sample ${img.sample_index}`}
                            className="w-full rounded border border-gray-700 hover:border-blue-500 transition-colors"
                          />
                          {img.on_demand && (
                            <span className="absolute left-1 top-1 rounded bg-blue-900/80 px-1 py-0.5 text-[10px] text-blue-200">
                              On demand
                            </span>
                          )}
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
                          {/* SDXL crop / micro-conditioning (crop-augmentation verification) */}
                          {(debugVisualization.original_size || debugVisualization.crop_top_left || debugVisualization.target_size) && (
                            <div className="pt-1 border-t border-gray-700">
                              <div className="text-gray-400 mb-0.5">SDXL micro-conditioning (item 0):</div>
                              {debugVisualization.original_size && (
                                <div><span className="text-gray-400">Original size (WxH):</span> {debugVisualization.original_size[0]}×{debugVisualization.original_size[1]}</div>
                              )}
                              {debugVisualization.crop_top_left && (
                                <div>
                                  <span className="text-gray-400">Crop point (left,top):</span> {debugVisualization.crop_top_left[0]},{debugVisualization.crop_top_left[1]}
                                  {(debugVisualization.crop_top_left[0] !== 0 || debugVisualization.crop_top_left[1] !== 0) && (
                                    <span className="text-green-400 ml-1">(cropped)</span>
                                  )}
                                </div>
                              )}
                              {debugVisualization.target_size && (
                                <div><span className="text-gray-400">Target/bucket (WxH):</span> {debugVisualization.target_size[0]}×{debugVisualization.target_size[1]}</div>
                              )}
                              {debugVisualization.sdxl_time_ids_all && debugVisualization.sdxl_time_ids_all.length > 1 && (
                                <div className="text-gray-500 text-xxs mt-0.5">
                                  batch: {debugVisualization.sdxl_time_ids_all.length} items (time_ids = [oh,ow,ct,cl,th,tw])
                                </div>
                              )}
                            </div>
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

                          {/* Comparison Container — staged left→right wipe so the yellow
                              line ALWAYS sits exactly on the image transition.
                              3-way (latent runs): drag right reveals noisy→predicted→target.
                                stage 1 [0..50]: base=noisy,     top=predicted (clip [0,wipe])
                                stage 2 [50..100]: base=predicted, top=target    (clip [0,wipe])
                              wipe = position of the transition (= the line), 0→100 per stage.
                              2-way fallback (no noisy): base=target, top=predicted clipped to s. */}
                          {(() => {
                            const v = debugVisualization;
                            const has3 = !!v.noisy_latents_image && !!v.predicted_latent_image && !!v.latents_image;
                            const s = comparisonSlider;
                            const px = (b64?: string) => b64 ? `data:image/png;base64,${b64}` : undefined;
                            const GREEN = "bg-green-700/80", BLUE = "bg-blue-700/80", PURPLE = "bg-purple-700/80";
                            let baseSrc: string | undefined, baseLabel = "", baseColor = GREEN;
                            let topSrc: string | undefined, topLabel = "", topColor = BLUE;
                            let wipe: number;  // transition position (%) = yellow line
                            if (has3) {
                              const stage2 = s > 50;
                              wipe = stage2 ? (s - 50) * 2 : s * 2;  // 0→100 within each stage
                              if (stage2) {
                                baseSrc = px(v.predicted_latent_image); baseLabel = "Predicted"; baseColor = BLUE;
                                topSrc = px(v.latents_image);           topLabel = "Target";     topColor = GREEN;
                              } else {
                                baseSrc = px(v.noisy_latents_image);    baseLabel = "Noisy";     baseColor = PURPLE;
                                topSrc = px(v.predicted_latent_image);  topLabel = "Predicted";  topColor = BLUE;
                              }
                            } else {
                              wipe = s;
                              baseSrc = px(v.latents_image);            baseLabel = "Target";    baseColor = GREEN;
                              topSrc = px(v.predicted_latent_image);    topLabel = "Predicted";  topColor = BLUE;
                            }
                            return (
                          <div className="relative aspect-square bg-gray-800 rounded overflow-hidden">
                            {/* Base layer (right side of the wipe) */}
                            {baseSrc && (
                              <div className="absolute inset-0">
                                <img src={baseSrc} alt={baseLabel} className="w-full h-full object-contain" />
                                <div className={`absolute top-1 right-1 ${baseColor} text-white text-xs px-1.5 py-0.5 rounded`}>{baseLabel}</div>
                              </div>
                            )}
                            {/* Top layer (left side of the wipe), clipped to [0, wipe] */}
                            {topSrc && (
                              <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - wipe}% 0 0)` }}>
                                <img src={topSrc} alt={topLabel} className="w-full h-full object-contain" />
                                <div className={`absolute top-1 left-1 ${topColor} text-white text-xs px-1.5 py-0.5 rounded`}>{topLabel}</div>
                              </div>
                            )}
                            {/* Yellow line exactly on the transition */}
                            <div
                              className="absolute top-0 bottom-0 w-0.5 bg-yellow-500 pointer-events-none"
                              style={{ left: `${wipe}%` }}
                            />
                          </div>
                            );
                          })()}

                          {/* Slider Control */}
                          <input
                            type="range"
                            min="0"
                            max="100"
                            step="any"
                            value={comparisonSlider}
                            onChange={(e) => setComparisonSlider(Number(e.target.value))}
                            className="w-full"
                          />
                          <div className="flex justify-between text-xs text-gray-500">
                            {debugVisualization.noisy_latents_image ? (
                              <>
                                <span>Noisy (t={debugVisualization.timestep})</span>
                                <span>Predicted (t=0)</span>
                                <span>Target</span>
                              </>
                            ) : (
                              <>
                                <span>Target (Original)</span>
                                <span>Predicted (t=0)</span>
                              </>
                            )}
                          </div>

                          {/* Additional Debug Images */}
                          <div className="grid grid-cols-2 gap-2 mt-3">
                            {/* Noisy Latents — only as a separate thumbnail when NOT
                                shown in the 3-way comparison above (backward compat). */}
                            {debugVisualization.noisy_latents_image && !(debugVisualization.predicted_latent_image && debugVisualization.latents_image) && (
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
      {viewerImage && viewerIndex !== null && (
        <ImageViewer
          imageUrl={viewerImage.path}
          onClose={() => setViewerIndex(null)}
          onNavigate={navigateSample}
          hasPrev={viewerIndex > 0}
          hasNext={viewerIndex < sampleImages.length - 1}
          showDownload={false}
        />
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
