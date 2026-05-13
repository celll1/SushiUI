"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  getTaggerTrainingRun,
  getTaggerTrainingMetrics,
  startTaggerTrainingRun,
  stopTaggerTrainingRun,
  deleteTaggerTrainingRun,
  TaggerTrainingRun,
  TaggerTrainingMetric,
} from "@/utils/api";
import { wsClient, TaggerMetrics } from "@/utils/websocket";
import VocabularyBrowser from "@/components/tagger/VocabularyBrowser";
import TaggerMetricChart from "./TaggerMetricChart";

interface TaggerTrainingMonitorProps {
  run: TaggerTrainingRun;
  onClose: () => void;
  onStatusChange: (run: TaggerTrainingRun) => void;
  onDelete: () => void;
  onEditConfig?: () => void;
}

export default function TaggerTrainingMonitor({
  run: initialRun,
  onClose,
  onStatusChange,
  onDelete,
  onEditConfig,
}: TaggerTrainingMonitorProps) {
  const [run, setRun] = useState<TaggerTrainingRun>(initialRun);
  const [metrics, setMetrics] = useState<TaggerTrainingMetric[]>([]);
  const [actionLoading, setActionLoading] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [iterPerSec, setIterPerSec] = useState<number | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  // Buffer incoming WS step events; flush to state at most once per second
  const wsBufferRef = useRef<TaggerTrainingMetric[]>([]);
  const wsFlushRef = useRef<NodeJS.Timeout | null>(null);
  // Latest run-status fields from WS (progress / current_step / ...).
  // Overwritten on every incoming message so the 1s flush applies only the
  // most recent values without burning re-renders per step.  No DB load —
  // the trainer's send_tagger_metrics already carries these.
  const wsRunUpdateRef = useRef<{
    progress?: number;
    step?: number;
    epoch?: number;
    loss?: number;
  } | null>(null);
  // Rolling (step, client_recv_time) samples for iter/s calculation.
  // We use client recv time rather than server timestamp because the WS
  // payload does not carry one and clock skew would just shift the rate
  // by a constant.  Reset when resume_seq changes (step counter restarts).
  const iterRateSamplesRef = useRef<Array<{ step: number; t: number; seq: number }>>([]);

  const updateRun = useCallback((updated: TaggerTrainingRun) => {
    setRun(updated);
    onStatusChange(updated);
  }, [onStatusChange]);

  const fetchStatus = useCallback(async () => {
    try {
      const updated = await getTaggerTrainingRun(run.run_id);
      updateRun(updated);
    } catch (err) {
      console.error("[TaggerMonitor] Failed to fetch status:", err);
    }
  }, [run.run_id, updateRun]);

  const fetchMetrics = useCallback(async () => {
    try {
      const data = await getTaggerTrainingMetrics(run.run_id);
      if (data.length > 0) setMetrics(data);
    } catch (err) {
      console.error("[TaggerMonitor] Failed to fetch metrics:", err);
    }
  }, [run.run_id]);

  // WebSocket: receive live tagger metrics during training
  useEffect(() => {
    const handler = (m: TaggerMetrics) => {
      if (m.run_id !== run.run_id) return;

      // Capture latest run-status fields for the 1s flush.  Each new WS
      // message overwrites previous (we only want the most recent values
      // — older progress / step / loss are stale by then).
      wsRunUpdateRef.current = {
        progress: m.progress,
        step:     m.step,
        epoch:    m.epoch,
        loss:     m.loss,
      };

      // Record sample for iter/s.  Only "step" events advance the step
      // counter; "epoch" events fire on the same step and would inflate
      // the rate if counted.
      if (m.event === "step") {
        const seq = m.resume_seq ?? 0;
        const samples = iterRateSamplesRef.current;
        // Drop the window when we cross a resume boundary (step
        // counter is monotonic within a resume but can jump backwards
        // across resumes).
        if (samples.length > 0 && samples[samples.length - 1].seq !== seq) {
          iterRateSamplesRef.current = [];
        }
        iterRateSamplesRef.current.push({ step: m.step, t: performance.now(), seq });
        // Keep at most 20 samples; oldest evicted FIFO.
        if (iterRateSamplesRef.current.length > 20) {
          iterRateSamplesRef.current.shift();
        }
      }

      const item: TaggerTrainingMetric = {
        step: m.step,
        resume_seq: m.resume_seq ?? 0,
        epoch: m.epoch ?? null,
        loss: m.loss ?? null,
        f1: m.f1 ?? null,
        threshold: m.threshold ?? null,
        learning_rate: m.lr ?? null,
        timestamp: new Date().toISOString(),
      };

      // Buffer and flush at most 1x/sec to avoid per-step re-renders
      wsBufferRef.current.push(item);
      if (!wsFlushRef.current) {
        wsFlushRef.current = setTimeout(() => {
          wsFlushRef.current = null;

          // Apply the latest run-status fields to local run state.
          // Bypasses onStatusChange (parent stays on 10s poll cadence to
          // avoid re-rendering the full run list every second).
          const upd = wsRunUpdateRef.current;
          if (upd) {
            wsRunUpdateRef.current = null;
            setRun(prev => ({
              ...prev,
              progress:      upd.progress ?? prev.progress,
              current_step:  upd.step ?? prev.current_step,
              current_epoch: upd.epoch ?? prev.current_epoch,
              latest_loss:   upd.loss ?? prev.latest_loss,
            }));
          }

          // Compute iter/s from the rolling window (oldest → newest).
          // Need ≥ 2 samples spanning > 0.2s to produce a stable value.
          const samples = iterRateSamplesRef.current;
          if (samples.length >= 2) {
            const first = samples[0];
            const last  = samples[samples.length - 1];
            const dStep = last.step - first.step;
            const dT    = (last.t - first.t) / 1000;
            if (dStep > 0 && dT >= 0.2) {
              setIterPerSec(dStep / dT);
            }
          }

          const incoming = wsBufferRef.current.splice(0);
          if (incoming.length === 0) return;
          setMetrics(prev => {
            const MAX_POINTS = 2000;
            // Compound key (resume_seq:step) so different resumes coexist
            const keyOf = (r: TaggerTrainingMetric) => `${r.resume_seq ?? 0}:${r.step}`;
            const map = new Map<string, TaggerTrainingMetric>(prev.map(r => [keyOf(r), r]));
            for (const r of incoming) {
              const k = keyOf(r);
              const existing = map.get(k);
              // merge: epoch events carry f1/threshold, step events carry loss/lr
              map.set(k, existing ? { ...existing, ...Object.fromEntries(
                Object.entries(r).filter(([, v]) => v !== null && v !== undefined)
              ) } : r);
            }
            let sorted = Array.from(map.values()).sort(
              (a, b) => (a.resume_seq ?? 0) - (b.resume_seq ?? 0) || a.step - b.step
            );
            // Per-group decimation so sparse early resumes survive
            if (sorted.length > MAX_POINTS) {
              const groups = new Map<number, TaggerTrainingMetric[]>();
              for (const r of sorted) {
                const seq = r.resume_seq ?? 0;
                if (!groups.has(seq)) groups.set(seq, []);
                groups.get(seq)!.push(r);
              }
              const perGroup = Math.max(50, Math.floor(MAX_POINTS / groups.size));
              const out: TaggerTrainingMetric[] = [];
              for (const seq of [...groups.keys()].sort((a, b) => a - b)) {
                const g = groups.get(seq)!;
                if (g.length > perGroup) {
                  const stride = Math.ceil(g.length / perGroup);
                  const decimated = g.filter((_, i) => i % stride === 0);
                  if (decimated[decimated.length - 1] !== g[g.length - 1]) {
                    decimated.push(g[g.length - 1]);
                  }
                  out.push(...decimated);
                } else {
                  out.push(...g);
                }
              }
              sorted = out;
            }
            return sorted;
          });
        }, 1000);
      }
    };

    wsClient.subscribeToTaggerMetrics(handler);
    return () => {
      wsClient.unsubscribeFromTaggerMetrics(handler);
      if (wsFlushRef.current) {
        clearTimeout(wsFlushRef.current);
        wsFlushRef.current = null;
      }
    };
  }, [run.run_id]);

  // Poll status only (not metrics — metrics come via WebSocket).
  //
  // Streaming fields (progress / current_step / current_epoch / latest_loss)
  // are now driven by WebSocket, so polling only needs to catch terminal
  // status transitions (completed / failed / stopped) and fields not in
  // the WS payload (best_f1, error_message, ...).  Relaxed to 15s.
  //
  // ``fetchStatus`` is stashed in a ref so the effect's deps only include
  // run.status — otherwise a re-created fetchStatus (cascading through
  // updateRun → onStatusChange identity changes) would clear and re-arm
  // the interval on every parent render, which can starve the timer if
  // the parent renders faster than the polling period.
  const fetchStatusRef = useRef(fetchStatus);
  useEffect(() => { fetchStatusRef.current = fetchStatus; }, [fetchStatus]);

  useEffect(() => {
    const isActive = run.status === "running" || run.status === "starting";

    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }

    if (!isActive) {
      // Run stopped — drop the rate display so it doesn't look frozen.
      setIterPerSec(null);
      iterRateSamplesRef.current = [];
      return;
    }

    pollingRef.current = setInterval(() => fetchStatusRef.current(), 15000);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [run.status]);

  // Decay the rate display when no new step events arrive (e.g. while a
  // validation pass runs between epochs).  After 10s of silence, clear.
  useEffect(() => {
    if (run.status !== "running") return;
    const timer = setInterval(() => {
      const samples = iterRateSamplesRef.current;
      if (samples.length === 0) return;
      const last = samples[samples.length - 1];
      if (performance.now() - last.t > 10_000) {
        setIterPerSec(null);
        iterRateSamplesRef.current = [];
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [run.status]);

  // Load full metrics history on mount (for resumed/completed runs)
  useEffect(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  const handleStart = async () => {
    setActionLoading(true);
    setError(null);
    try {
      const result = await startTaggerTrainingRun(run.run_id);
      updateRun(result.run);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setActionLoading(false);
    }
  };

  const handleStop = async () => {
    setActionLoading(true);
    setError(null);
    try {
      const result = await stopTaggerTrainingRun(run.run_id);
      updateRun(result.run);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setActionLoading(false);
    }
  };

  const handleDelete = async () => {
    setActionLoading(true);
    try {
      await deleteTaggerTrainingRun(run.run_id);
      onDelete();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setActionLoading(false);
    }
  };

  const isActive = run.status === "running" || run.status === "starting";
  const canStart = !isActive && run.status !== "completed";
  const canStop = isActive;

  const latestThr = [...metrics].reverse().find((m) => m.threshold !== null)?.threshold ?? null;

  const statusColor =
    run.status === "running" ? "text-blue-400" :
    run.status === "completed" ? "text-green-400" :
    run.status === "failed" ? "text-red-400" :
    run.status === "stopped" ? "text-yellow-400" :
    "text-gray-400";

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700 flex-shrink-0">
        <div>
          <h2 className="text-lg font-semibold">{run.run_name}</h2>
          <div className="flex items-center gap-3 mt-0.5">
            <span className={`text-sm font-medium ${statusColor}`}>{run.status}</span>
            <span className="text-xs text-gray-500">
              {run.training_method.toUpperCase()} · {run.num_tags} tags
            </span>
          </div>
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">

        {/* Progress */}
        {(isActive || run.progress > 0) && (
          <section>
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-gray-400">
                {run.status_message
                  ? run.status_message
                  : `Epoch ${run.current_epoch} · Step ${run.current_step}`}
                {iterPerSec !== null && (
                  <span className="ml-2 text-gray-500 font-mono">
                    · {iterPerSec >= 1
                        ? `${iterPerSec.toFixed(2)} it/s`
                        : `${(1 / iterPerSec).toFixed(2)} s/it`}
                  </span>
                )}
              </span>
              <span className="text-gray-300">{(run.progress * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${run.progress * 100}%` }}
              />
            </div>
          </section>
        )}

        {/* Stats */}
        <section className="grid grid-cols-4 gap-3">
          <div className="bg-gray-800 rounded p-3">
            <div className="text-xs text-gray-400 mb-1">Best F1</div>
            <div className="text-lg font-mono text-green-400">
              {run.best_f1 !== null ? run.best_f1.toFixed(4) : "—"}
            </div>
          </div>
          <div className="bg-gray-800 rounded p-3">
            <div className="text-xs text-gray-400 mb-1">Best Threshold</div>
            <div className="text-lg font-mono text-blue-400">
              {run.best_threshold !== null ? run.best_threshold.toFixed(3) : "—"}
            </div>
          </div>
          <div className="bg-gray-800 rounded p-3">
            <div className="text-xs text-gray-400 mb-1">Latest Threshold</div>
            <div className="text-lg font-mono text-cyan-400">
              {latestThr !== null ? latestThr.toFixed(3) : "—"}
            </div>
          </div>
          <div className="bg-gray-800 rounded p-3">
            <div className="text-xs text-gray-400 mb-1">Latest Loss</div>
            <div className="text-lg font-mono text-orange-400">
              {run.latest_loss !== null ? run.latest_loss.toFixed(4) : "—"}
            </div>
          </div>
        </section>

        {/* Charts (col-span-2) + Side panel (col-span-1) */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Charts column */}
          <div className="lg:col-span-2 space-y-3">
            <TaggerMetricChart
              data={metrics}
              valueKey="loss"
              color="#f97316"
              title="Training Loss"
              height={200}
              smoothable={true}
              defaultSmoothing={0.9}
              yMinFloor={0}
            />
            <TaggerMetricChart
              data={metrics}
              valueKey="f1"
              color="#22c55e"
              title="Validation F1"
              height={140}
              yMinFloor={0}
            />
            <TaggerMetricChart
              data={metrics}
              valueKey="threshold"
              color="#06b6d4"
              title="Optimal Threshold"
              height={100}
              yMinFloor={0}
            />

            {/* Error message */}
            {run.error_message && (
              <div>
                <div className="text-sm font-medium text-red-400 mb-1">Error</div>
                <div className="bg-red-900/20 border border-red-700 rounded p-3 text-xs text-red-300 font-mono whitespace-pre-wrap">
                  {run.error_message}
                </div>
              </div>
            )}
          </div>

          {/* Side column */}
          <div className="space-y-4 min-w-0">
            {/* Configuration */}
            <div>
              <div className="text-sm font-medium text-gray-300 mb-2">Configuration</div>
              <div className="text-xs text-gray-400 bg-gray-800 rounded p-3 space-y-1">
                <div className="flex gap-1 min-w-0">
                  <span className="text-gray-500 shrink-0">Vision encoder:</span>
                  <span className="text-gray-300 font-mono truncate" title={run.vision_encoder_path}>{run.vision_encoder_path}</span>
                </div>
                <div><span className="text-gray-500">Datasets:</span> <span className="text-gray-300">{run.dataset_configs.length}</span></div>
              </div>
              {run.config && typeof run.config === "object" && (() => {
                const CONFIG_LABELS: Record<string, string> = {
                  learning_rate: "LR",
                  head_lr_multiplier: "Head LR ×",
                  epochs: "Epochs",
                  batch_size: "Batch size",
                  optimizer: "Optimizer",
                  mixed_precision: "Precision",
                  loss_function: "Loss fn",
                  lora_rank: "LoRA rank",
                  lora_alpha: "LoRA alpha",
                  warmup_steps: "Warmup steps",
                  save_every_n_steps: "Save / N steps",
                  save_every_n_epochs: "Save / N epochs",
                  keep_last_n_checkpoints: "Keep last N",
                  checkpoint_save_mode: "Save mode",
                  loss_gamma_neg: "γ- (ASL)",
                  loss_gamma_pos: "γ+ (ASL)",
                  loss_gamma0: "γ₀",
                  loss_m0: "m₀",
                  loss_rho: "ρ",
                  loss_beta: "β",
                  loss_label_weight: "Label weight",
                  gradient_checkpointing: "Grad ckpt",
                  validate_every: "Validate / epochs",
                  vocab_min_count: "Min tag count",
                  val_split_mode: "Val split mode",
                  val_split: "Val split (%)",
                  val_fixed_size: "Val size (fixed)",
                  excluded_categories: "Excl. cats",
                  use_tag_aliases: "Tag aliases",
                  ban_tags: "Ban tags",
                  init_head_from: "Init head from",
                  cls_dim: "CLS dim",
                  hidden_proj_dim: "Hidden proj dim",
                  num_workers: "Workers",
                  num_workers_override: "Workers (override)",
                  weight_decay: "Weight decay",
                  loss_clip: "Loss clip",
                  build_lr_matrix_on_start: "Build LR matrix",
                  lr_top_anchors: "LR top anchors",
                  lr_top_targets: "LR top targets",
                  lr_threshold: "LR threshold",
                  lr_min_anchor_count: "LR min anchor count",
                };
                const cfg = run.config as Record<string, unknown>;
                const lossFn = String(cfg.loss_function ?? "asl");
                const isLora = run.training_method === "lora";
                const LORA_ONLY_KEYS = new Set(["lora_rank", "lora_alpha"]);
                const ASL_ONLY_KEYS  = new Set(["loss_gamma_neg", "loss_gamma_pos"]);
                const CS_ASL_KEYS    = new Set(["loss_gamma0", "loss_m0", "loss_beta", "loss_rho"]);
                const H_CS_ASL_KEYS  = new Set(["loss_label_weight"]);
                // LR sub-parameters only meaningful when build_lr_matrix_on_start is true
                const LR_SUB_KEYS    = new Set([
                  "lr_top_anchors", "lr_top_targets", "lr_threshold", "lr_min_anchor_count",
                ]);
                const buildLR = Boolean(cfg.build_lr_matrix_on_start);
                const entries = Object.entries(CONFIG_LABELS)
                  .map(([key, label]) => ({
                    key,
                    label,
                    value: key === "loss_function" ? (cfg[key] ?? "asl") : cfg[key],
                  }))
                  .filter(({ key, value }) => {
                    if (value === undefined || value === null || value === "") return false;
                    if (LORA_ONLY_KEYS.has(key) && !isLora) return false;
                    if (ASL_ONLY_KEYS.has(key) && lossFn !== "asl") return false;
                    if (CS_ASL_KEYS.has(key) && !["cs_asl", "h_cs_asl", "la_s_asl"].includes(lossFn)) return false;
                    if (H_CS_ASL_KEYS.has(key) && lossFn !== "h_cs_asl") return false;
                    if (LR_SUB_KEYS.has(key) && !buildLR) return false;
                    return true;
                  });
                return (
                  <div className="mt-2 bg-gray-800 rounded p-3 space-y-1 text-xs">
                    {entries.map(({ key, label, value }) => {
                      const display = Array.isArray(value)
                        ? value.join(", ") || "—"
                        : String(value);
                      return (
                        <div key={key} className="flex gap-1 min-w-0">
                          <span className="text-gray-500 shrink-0">{label}:</span>
                          <span className="text-gray-300 truncate" title={display}>{display}</span>
                        </div>
                      );
                    })}
                  </div>
                );
              })()}
            </div>

            {/* Threshold F1 Curve */}
            {run.threshold_f1_curve && Object.keys(run.threshold_f1_curve).length > 0 && (() => {
              const curve = run.threshold_f1_curve!;
              const bestThr = Object.keys(curve).reduce((a, b) => curve[a] >= curve[b] ? a : b);
              return (
                <div>
                  <div className="text-sm font-medium text-gray-300 mb-2">Threshold Grid Search</div>
                  <div className="bg-gray-800 rounded p-2 border border-gray-700 overflow-x-auto">
                    <table className="text-xs w-full">
                      <thead>
                        <tr className="text-gray-400 border-b border-gray-700">
                          <th className="text-left pb-1 pr-2">Thr</th>
                          <th className="text-left pb-1 pr-2">F1</th>
                          <th className="text-left pb-1">Bar</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(curve).map(([thr, f1]) => {
                          const isBest = thr === bestThr;
                          return (
                            <tr key={thr} className={isBest ? "text-green-400 font-bold" : "text-gray-300"}>
                              <td className="pr-2 py-0.5">{thr}</td>
                              <td className="pr-2 py-0.5 font-mono">{(f1 as number).toFixed(4)}</td>
                              <td className="py-0.5">
                                <div className="bg-gray-700 rounded-full h-1.5 w-full">
                                  <div
                                    className={`h-1.5 rounded-full ${isBest ? "bg-green-400" : "bg-blue-500"}`}
                                    style={{ width: `${Math.min((f1 as number) * 100, 100)}%` }}
                                  />
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div className="text-xs text-green-400 mt-1">
                    Optimal: {bestThr} (F1={curve[bestThr].toFixed(4)})
                  </div>
                </div>
              );
            })()}

            {/* Checkpoint paths */}
            {(run.head_checkpoint_path || run.lora_checkpoint_path) && (
              <div>
                <div className="text-sm font-medium text-gray-300 mb-2">Checkpoints</div>
                <div className="space-y-1">
                  {run.head_checkpoint_path && (
                    <div className="text-xs text-gray-400 bg-gray-800 rounded p-2 font-mono truncate" title={run.head_checkpoint_path}>
                      Head: {run.head_checkpoint_path}
                    </div>
                  )}
                  {run.lora_checkpoint_path && (
                    <div className="text-xs text-gray-400 bg-gray-800 rounded p-2 font-mono truncate" title={run.lora_checkpoint_path}>
                      LoRA: {run.lora_checkpoint_path}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Vocabulary browser */}
        <section>
          <VocabularyBrowser runId={run.run_id} />
        </section>

        {/* Action error */}
        {error && (
          <div className="p-3 bg-red-900/30 border border-red-700 rounded text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Delete confirmation */}
        {confirmDelete && (
          <div className="p-4 bg-gray-800 border border-red-700 rounded">
            <p className="text-sm text-gray-300 mb-3">
              Delete this tagger run? This cannot be undone.
            </p>
            <div className="flex gap-3">
              <button
                onClick={handleDelete}
                disabled={actionLoading}
                className="px-3 py-1.5 bg-red-700 hover:bg-red-600 disabled:bg-gray-600 rounded text-sm transition-colors"
              >
                {actionLoading ? "Deleting..." : "Delete"}
              </button>
              <button
                onClick={() => setConfirmDelete(false)}
                className="px-3 py-1.5 text-sm text-gray-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Footer actions */}
      <div className="flex-shrink-0 p-4 border-t border-gray-700 flex justify-between items-center">
        <button
          onClick={() => setConfirmDelete(true)}
          disabled={actionLoading || isActive}
          className="px-3 py-1.5 text-sm text-red-400 hover:text-red-300 disabled:text-gray-600 transition-colors"
        >
          Delete
        </button>
        <div className="flex gap-2">
          {onEditConfig && ["pending", "stopped", "failed"].includes(run.status) && (
            <button
              onClick={onEditConfig}
              disabled={actionLoading}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-600 disabled:text-gray-400 rounded text-sm transition-colors"
            >
              Edit Config
            </button>
          )}
          {canStart && (
            <button
              onClick={handleStart}
              disabled={actionLoading}
              className="px-4 py-2 bg-green-700 hover:bg-green-600 disabled:bg-gray-600 disabled:text-gray-400 rounded text-sm transition-colors"
            >
              {actionLoading ? "Starting..." : "Start Training"}
            </button>
          )}
          {canStop && (
            <button
              onClick={handleStop}
              disabled={actionLoading}
              className="px-4 py-2 bg-yellow-700 hover:bg-yellow-600 disabled:bg-gray-600 disabled:text-gray-400 rounded text-sm transition-colors"
            >
              {actionLoading ? "Stopping..." : "Stop"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
