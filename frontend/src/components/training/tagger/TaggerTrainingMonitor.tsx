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
import VocabularyBrowser from "@/components/tagger/VocabularyBrowser";

interface TaggerTrainingMonitorProps {
  run: TaggerTrainingRun;
  onClose: () => void;
  onStatusChange: (run: TaggerTrainingRun) => void;
  onDelete: () => void;
  onEditConfig?: () => void;
}

function MiniChart({
  data,
  valueKey,
  color,
  height = 80,
}: {
  data: TaggerTrainingMetric[];
  valueKey: "loss" | "f1";
  color: string;
  height?: number;
}) {
  const points = data
    .map((d) => ({ step: d.step, value: d[valueKey] }))
    .filter((d): d is { step: number; value: number } => d.value !== null);

  if (points.length < 2) {
    return (
      <div
        className="flex items-center justify-center text-gray-500 text-xs"
        style={{ height }}
      >
        Not enough data
      </div>
    );
  }

  const minV = Math.min(...points.map((p) => p.value));
  const maxV = Math.max(...points.map((p) => p.value));
  const range = maxV - minV || 1;
  const minStep = points[0].step;
  const maxStep = points[points.length - 1].step;
  const stepRange = maxStep - minStep || 1;

  const w = 400;
  const h = height;
  const pad = 4;

  const toX = (step: number) =>
    pad + ((step - minStep) / stepRange) * (w - 2 * pad);
  const toY = (v: number) =>
    pad + ((maxV - v) / range) * (h - 2 * pad);

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.step).toFixed(1)} ${toY(p.value).toFixed(1)}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full"
      style={{ height }}
      preserveAspectRatio="none"
    >
      <path d={pathD} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
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
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

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
      setMetrics(data);
    } catch (err) {
      console.error("[TaggerMonitor] Failed to fetch metrics:", err);
    }
  }, [run.run_id]);

  // Poll when running
  useEffect(() => {
    const isActive = run.status === "running" || run.status === "starting";

    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }

    if (!isActive) return;

    pollingRef.current = setInterval(async () => {
      await fetchStatus();
      await fetchMetrics();
    }, 3000);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [run.status, fetchStatus, fetchMetrics]);

  // Load metrics on mount
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

  const lossData = metrics.filter((m) => m.loss !== null);
  const f1Data = metrics.filter((m) => m.f1 !== null);

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
        <section className="grid grid-cols-3 gap-3">
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
            <div className="text-xs text-gray-400 mb-1">Latest Loss</div>
            <div className="text-lg font-mono text-orange-400">
              {run.latest_loss !== null ? run.latest_loss.toFixed(4) : "—"}
            </div>
          </div>
        </section>

        {/* Loss chart */}
        {lossData.length >= 2 && (
          <section>
            <div className="text-sm font-medium text-gray-300 mb-2">Training Loss</div>
            <div className="bg-gray-800 rounded p-2 border border-gray-700">
              <MiniChart data={lossData} valueKey="loss" color="#f97316" height={80} />
            </div>
          </section>
        )}

        {/* F1 chart */}
        {f1Data.length >= 2 && (
          <section>
            <div className="text-sm font-medium text-gray-300 mb-2">Validation F1</div>
            <div className="bg-gray-800 rounded p-2 border border-gray-700">
              <MiniChart data={f1Data} valueKey="f1" color="#22c55e" height={80} />
            </div>
          </section>
        )}

        {/* Error message */}
        {run.error_message && (
          <section>
            <div className="text-sm font-medium text-red-400 mb-1">Error</div>
            <div className="bg-red-900/20 border border-red-700 rounded p-3 text-xs text-red-300 font-mono whitespace-pre-wrap">
              {run.error_message}
            </div>
          </section>
        )}

        {/* Checkpoint paths */}
        {(run.head_checkpoint_path || run.lora_checkpoint_path) && (
          <section>
            <div className="text-sm font-medium text-gray-300 mb-2">Checkpoints</div>
            <div className="space-y-1">
              {run.head_checkpoint_path && (
                <div className="text-xs text-gray-400 bg-gray-800 rounded p-2 font-mono truncate">
                  Head: {run.head_checkpoint_path}
                </div>
              )}
              {run.lora_checkpoint_path && (
                <div className="text-xs text-gray-400 bg-gray-800 rounded p-2 font-mono truncate">
                  LoRA: {run.lora_checkpoint_path}
                </div>
              )}
            </div>
          </section>
        )}

        {/* Threshold F1 Curve */}
        {run.threshold_f1_curve && Object.keys(run.threshold_f1_curve).length > 0 && (() => {
          const curve = run.threshold_f1_curve!;
          const bestThr = Object.keys(curve).reduce((a, b) => curve[a] >= curve[b] ? a : b);
          return (
            <section>
              <div className="text-sm font-medium text-gray-300 mb-2">Threshold Grid Search</div>
              <div className="bg-gray-800 rounded p-2 border border-gray-700 overflow-x-auto">
                <table className="text-xs w-full">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-700">
                      <th className="text-left pb-1 pr-4">Threshold</th>
                      <th className="text-left pb-1 pr-4">F1</th>
                      <th className="text-left pb-1">Bar</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(curve).map(([thr, f1]) => {
                      const isBest = thr === bestThr;
                      return (
                        <tr key={thr} className={isBest ? "text-green-400 font-bold" : "text-gray-300"}>
                          <td className="pr-4 py-0.5">{thr}</td>
                          <td className="pr-4 py-0.5 font-mono">{(f1 as number).toFixed(4)}</td>
                          <td className="py-0.5 w-32">
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
                Optimal threshold: {bestThr} (F1={curve[bestThr].toFixed(4)})
              </div>
            </section>
          );
        })()}

        {/* Config info */}
        <section>
          <div className="text-sm font-medium text-gray-300 mb-2">Configuration</div>
          <div className="text-xs text-gray-400 bg-gray-800 rounded p-3 space-y-1">
            <div className="flex gap-1">
              <span className="shrink-0">Vision encoder:</span>
              <span className="text-gray-300 font-mono truncate" title={run.vision_encoder_path}>{run.vision_encoder_path}</span>
            </div>
            <div>Datasets: <span className="text-gray-300">{run.dataset_configs.length}</span></div>
          </div>
          {run.config && typeof run.config === "object" && (() => {
            const CONFIG_LABELS: Record<string, string> = {
              learning_rate: "LR",
              head_lr_multiplier: "Head LR ×",
              epochs: "Epochs",
              batch_size: "Batch size",
              optimizer: "Optimizer",
              mixed_precision: "Precision",
              lora_rank: "LoRA rank",
              lora_alpha: "LoRA alpha",
              warmup_steps: "Warmup steps",
              save_every_n_steps: "Save / N steps",
              save_every_n_epochs: "Save / N epochs",
              keep_last_n_checkpoints: "Keep last N",
              checkpoint_save_mode: "Save mode",
              loss_function: "Loss fn",
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
              ban_tags: "Ban tags",
              init_head_from: "Init head from",
              cls_dim: "CLS dim",
              hidden_proj_dim: "Hidden proj dim",
              num_workers: "Workers",
              num_workers_override: "Workers (override)",
              weight_decay: "Weight decay",
              loss_clip: "Loss clip",
            };
            const entries = Object.entries(CONFIG_LABELS)
              .map(([key, label]) => ({ key, label, value: (run.config as Record<string, unknown>)[key] }))
              .filter(({ value }) => value !== undefined && value !== null && value !== "");
            return (
              <div className="mt-2 bg-gray-800 rounded p-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
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
