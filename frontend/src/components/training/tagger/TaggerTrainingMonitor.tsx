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
import { wsClient, TaggerMetrics, FpFnScatterData, DatasetScanProgress } from "@/utils/websocket";
import VocabularyBrowser from "@/components/tagger/VocabularyBrowser";
import TaggerMetricChart, { EpochBoundary } from "./TaggerMetricChart";
import DanbooruMetricsPanel from "./DanbooruMetricsPanel";

interface TaggerTrainingMonitorProps {
  run: TaggerTrainingRun;
  onClose: () => void;
  onStatusChange: (run: TaggerTrainingRun) => void;
  onDelete: () => void;
  onEditConfig?: () => void;
}

/** Format a duration in seconds as a compact human-readable string.
 *  Examples: 23 → "23s", 90 → "1m 30s", 5400 → "1h 30m", 100000 → "1d 3h" */
function formatDuration(sec: number | null): string {
  if (sec === null || !Number.isFinite(sec) || sec < 0) return "—";
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) {
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return s === 0 ? `${m}m` : `${m}m ${s}s`;
  }
  if (sec < 86400) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return m === 0 ? `${h}h` : `${h}h ${m}m`;
  }
  const d = Math.floor(sec / 86400);
  const h = Math.floor((sec % 86400) / 3600);
  return h === 0 ? `${d}d` : `${d}d ${h}h`;
}

/** Format an ETA's wall-clock arrival time.  "today HH:MM", "tomorrow
 *  HH:MM", or "M/D HH:MM" for ≥ 2 days out. */
function formatArrivalTime(etaSec: number | null, nowMs: number): string {
  if (etaSec === null || !Number.isFinite(etaSec) || etaSec < 0) return "";
  const arrival = new Date(nowMs + etaSec * 1000);
  const now = new Date(nowMs);
  const hhmm = `${String(arrival.getHours()).padStart(2, "0")}:${String(arrival.getMinutes()).padStart(2, "0")}`;
  const sameDay = arrival.getFullYear() === now.getFullYear()
    && arrival.getMonth() === now.getMonth()
    && arrival.getDate() === now.getDate();
  if (sameDay) return `today ${hhmm}`;
  const tomorrow = new Date(now);
  tomorrow.setDate(now.getDate() + 1);
  const isTomorrow = arrival.getFullYear() === tomorrow.getFullYear()
    && arrival.getMonth() === tomorrow.getMonth()
    && arrival.getDate() === tomorrow.getDate();
  if (isTomorrow) return `tomorrow ${hhmm}`;
  return `${arrival.getMonth() + 1}/${arrival.getDate()} ${hhmm}`;
}

// ---------------------------------------------------------------------------
// FP/FN Scatter Plot component
// ---------------------------------------------------------------------------
function FpFnScatterPlot({ data }: { data: FpFnScatterData }) {
  const W = 280, H = 280;
  const margin = { top: 28, right: 16, bottom: 36, left: 40 };
  const pw = W - margin.left - margin.right;
  const ph = H - margin.top - margin.bottom;

  // n_pos range for opacity mapping
  const maxNpos = Math.max(...data.n_pos, 1);

  // Map data coordinates → SVG pixel coordinates
  const px = (fp: number) => margin.left + fp * pw;
  const py = (fn: number) => margin.top + (1 - fn) * ph;

  // Y-axis tick labels (0, 0.25, 0.5, 0.75, 1.0)
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <div className="bg-gray-900 rounded-lg p-3">
      <div className="text-xs font-medium text-gray-300 mb-1">
        FP/FN Rate Distribution @ thr=0.5
      </div>
      <svg width={W} height={H} className="block">
        {/* Grid lines */}
        {ticks.map((t) => (
          <line
            key={`gy-${t}`}
            x1={margin.left} y1={py(t)}
            x2={margin.left + pw} y2={py(t)}
            stroke="#374151" strokeWidth={0.5}
          />
        ))}
        {ticks.map((t) => (
          <line
            key={`gx-${t}`}
            x1={px(t)} y1={margin.top}
            x2={px(t)} y2={margin.top + ph}
            stroke="#374151" strokeWidth={0.5}
          />
        ))}

        {/* Reference lines: FP=0.5 and FN=0.5 */}
        <line x1={px(0.5)} y1={margin.top} x2={px(0.5)} y2={margin.top + ph}
          stroke="#6b7280" strokeWidth={1} strokeDasharray="4 3" />
        <line x1={margin.left} y1={py(0.5)} x2={margin.left + pw} y2={py(0.5)}
          stroke="#6b7280" strokeWidth={1} strokeDasharray="4 3" />

        {/* Diagonal reference line FP=FN */}
        <line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)}
          stroke="#4b5563" strokeWidth={0.8} />

        {/* Data points */}
        {data.fp.map((fp, i) => {
          const fn = data.fn[i];
          if (isNaN(fp) || isNaN(fn)) return null;
          const opacity = 0.35 + 0.65 * Math.sqrt(data.n_pos[i] / maxNpos);
          return (
            <circle
              key={i}
              cx={px(fp)} cy={py(fn)}
              r={2.5}
              fill="#60a5fa"
              fillOpacity={opacity}
            />
          );
        })}

        {/* Y-axis labels */}
        {ticks.map((t) => (
          <text key={`ty-${t}`} x={margin.left - 4} y={py(t) + 4}
            textAnchor="end" fontSize={9} fill="#9ca3af">
            {t.toFixed(2)}
          </text>
        ))}

        {/* X-axis labels */}
        {ticks.map((t) => (
          <text key={`tx-${t}`} x={px(t)} y={margin.top + ph + 14}
            textAnchor="middle" fontSize={9} fill="#9ca3af">
            {t.toFixed(2)}
          </text>
        ))}

        {/* Axis labels */}
        <text x={margin.left + pw / 2} y={H - 2}
          textAnchor="middle" fontSize={10} fill="#6b7280">
          FP rate
        </text>
        <text
          x={10} y={margin.top + ph / 2}
          textAnchor="middle" fontSize={10} fill="#6b7280"
          transform={`rotate(-90, 10, ${margin.top + ph / 2})`}
        >
          FN rate
        </text>

        {/* Border */}
        <rect x={margin.left} y={margin.top} width={pw} height={ph}
          fill="none" stroke="#374151" strokeWidth={1} />
      </svg>
      <div className="text-xs text-gray-500 mt-1">
        {data.n_tags} tags (n_pos ≥ 20, {data.total_images.toLocaleString()} images)
      </div>
    </div>
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
  const [epochBoundaries, setEpochBoundaries] = useState<EpochBoundary[]>([]);
  const [scatterData, setScatterData] = useState<FpFnScatterData | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Short-window rate (last ~20 samples, ~30s of WS step events).  Used
  // as the primary iter/s display and the basis for the ETA estimate.
  const [iterPerSec, setIterPerSec] = useState<number | null>(null);
  // Long-window rate (whole sample buffer, up to ~5 min of step events).
  // Used as a consistency reference for ETA — large divergence vs. the
  // short-window rate flags a recent slowdown / speedup the user should
  // be aware of (e.g. GPU contention, dataloader stall, batch-size
  // change).  null when not enough samples accumulated yet.
  const [iterPerSecLong, setIterPerSecLong] = useState<number | null>(null);
  // Scan progress during dataset rescan (step/total/pct 0–1). null when not scanning.
  const [scanProgress, setScanProgress] = useState<{ step: number; total: number; pct: number } | null>(null);
  // 1Hz wall-clock tick to recompute elapsed time without re-rendering
  // on every WS event.  Stored as ms-since-epoch.
  const [nowMs, setNowMs] = useState<number>(() => Date.now());
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const isScanningRef = useRef(false);
  // Full raw metrics accumulator — never decimated, keyed by "resume_seq:step".
  // The decimated display array (metrics state) is always recomputed from this
  // ref on every WS flush, so historical points are never progressively lost.
  const rawMetricsRef = useRef<Map<string, TaggerTrainingMetric>>(new Map());
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

  // Extract epoch boundaries from a raw metric map.
  // Only rows with epoch != null and step > 0 are considered.
  // If multiple rows share the same epoch, the one with the largest step wins.
  const extractEpochBoundaries = useCallback(
    (rawMap: Map<string, TaggerTrainingMetric>): EpochBoundary[] => {
      const byEpoch = new Map<number, number>();
      for (const m of rawMap.values()) {
        if (m.epoch !== null && m.epoch !== undefined && m.step > 0) {
          const existing = byEpoch.get(m.epoch);
          if (existing === undefined || m.step > existing) byEpoch.set(m.epoch, m.step);
        }
      }
      return [...byEpoch.entries()]
        .sort(([a], [b]) => a - b)
        .map(([epoch, step]) => ({ epoch, step }));
    },
    []
  );

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
      if (data.length > 0) {
        // Seed the raw accumulator with the full history from the API.
        const keyOf = (r: TaggerTrainingMetric) => `${r.resume_seq ?? 0}:${r.step}`;
        const rawMap = new Map<string, TaggerTrainingMetric>();
        for (const r of data) rawMap.set(keyOf(r), r);
        rawMetricsRef.current = rawMap;

        // Extract epoch boundaries from the full undecimented data
        setEpochBoundaries(extractEpochBoundaries(rawMap));

        // Apply the same global-stride decimation as the WS flush path so the
        // initial render is consistent with live updates.
        const MAX_POINTS = 2000;
        let display = data;
        if (data.length > MAX_POINTS) {
          const groups = new Map<number, TaggerTrainingMetric[]>();
          for (const r of data) {
            const seq = r.resume_seq ?? 0;
            if (!groups.has(seq)) groups.set(seq, []);
            groups.get(seq)!.push(r);
          }
          const globalStride = Math.ceil(data.length / MAX_POINTS);
          const out: TaggerTrainingMetric[] = [];
          for (const [, g] of [...groups.entries()].sort(([a], [b]) => a - b)) {
            const dec = g.filter((_, i) => i % globalStride === 0);
            if (dec.length === 0 || dec[dec.length - 1] !== g[g.length - 1]) dec.push(g[g.length - 1]);
            out.push(...dec);
          }
          display = out;
        }
        setMetrics(display);
      }
    } catch (err) {
      console.error("[TaggerMonitor] Failed to fetch metrics:", err);
    }
  }, [run.run_id, extractEpochBoundaries]);

  // WebSocket: receive live tagger metrics during training
  useEffect(() => {
    wsClient.connect();

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
        // Keep at most 200 samples (~3-5 min of step events at typical
        // training rates).  The short window slices the tail; the long
        // window uses the whole buffer.  Older evicted FIFO.
        if (iterRateSamplesRef.current.length > 200) {
          iterRateSamplesRef.current.shift();
        }
      }

      const item: TaggerTrainingMetric = {
        step: m.step,
        resume_seq: m.resume_seq ?? 0,
        epoch: m.epoch ?? null,
        loss: m.loss ?? null,
        f1: m.f1 ?? null,
        train_f1: m.train_f1 ?? null,
        threshold: m.threshold ?? null,
        learning_rate: m.lr ?? null,
        precision: m.precision ?? null,
        recall: m.recall ?? null,
        timestamp: new Date().toISOString(),
      };

      // Update scatter data immediately (arrives infrequently — every 500 steps)
      if (m.fp_fn_scatter && m.fp_fn_scatter.n_tags > 0) {
        setScatterData(m.fp_fn_scatter);
      }

      // Training is live — clear any in-progress scan overlay
      if (isScanningRef.current) {
        isScanningRef.current = false;
        setScanProgress(null);
      }

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

          // Compute iter/s from two windows:
          //   short: last 20 samples  (≈ 20–30 s of step events)
          //   long:  whole buffer     (≈ up to 3–5 min)
          // Both Δstep > 0 and Δt above a minimum to avoid noise.
          const samples = iterRateSamplesRef.current;
          if (samples.length >= 2) {
            const rateOf = (slice: typeof samples, minDt: number): number | null => {
              if (slice.length < 2) return null;
              const f = slice[0];
              const l = slice[slice.length - 1];
              const dStep = l.step - f.step;
              const dT    = (l.t - f.t) / 1000;
              if (dStep <= 0 || dT < minDt) return null;
              return dStep / dT;
            };
            const shortSlice = samples.slice(-20);
            setIterPerSec(rateOf(shortSlice, 0.2));
            // Long window: require ≥ 30 samples AND ≥ 5 s span before
            // displaying so it's actually a "long-term" reference.
            if (samples.length >= 30) {
              setIterPerSecLong(rateOf(samples, 5));
            }
          }

          const incoming = wsBufferRef.current.splice(0);
          if (incoming.length === 0) return;

          // 1. Merge incoming WS events into the raw accumulator (never decimated).
          //    epoch events carry f1/threshold; step events carry loss/lr — merge both.
          const keyOf = (r: TaggerTrainingMetric) => `${r.resume_seq ?? 0}:${r.step}`;
          const rawMap = rawMetricsRef.current;
          for (const r of incoming) {
            const k = keyOf(r);
            const existing = rawMap.get(k);
            rawMap.set(k, existing ? { ...existing, ...Object.fromEntries(
              Object.entries(r).filter(([, v]) => v !== null && v !== undefined)
            ) } : r);
          }

          // 2. Recompute the decimated display array from the *full* raw map
          //    every flush.  Because we start from rawMap each time, historical
          //    points are never progressively lost across flushes.
          //
          //    Use a single global stride across ALL groups so that all resumes
          //    appear at the same visual density.  Per-group strides caused sudden
          //    density jumps when one group's length crossed its individual quota.
          const MAX_POINTS = 2000;
          let sorted = Array.from(rawMap.values()).sort(
            (a, b) => (a.resume_seq ?? 0) - (b.resume_seq ?? 0) || a.step - b.step
          );
          if (sorted.length > MAX_POINTS) {
            const groups = new Map<number, TaggerTrainingMetric[]>();
            for (const r of sorted) {
              const seq = r.resume_seq ?? 0;
              if (!groups.has(seq)) groups.set(seq, []);
              groups.get(seq)!.push(r);
            }
            const seqs = [...groups.keys()].sort((a, b) => a - b);
            const totalRaw = sorted.length;
            // Single stride applied uniformly to every group.
            const globalStride = Math.ceil(totalRaw / MAX_POINTS);
            const out: TaggerTrainingMetric[] = [];
            for (const seq of seqs) {
              const g = groups.get(seq)!;
              const decimated = g.filter((_, i) => i % globalStride === 0);
              // Always keep the last point of each group so lines reach the edge.
              if (decimated.length === 0 || decimated[decimated.length - 1] !== g[g.length - 1]) {
                decimated.push(g[g.length - 1]);
              }
              out.push(...decimated);
            }
            sorted = out;
          }
          setMetrics(sorted);
          // Update epoch boundaries from the full raw map (never decimated)
          setEpochBoundaries(extractEpochBoundaries(rawMap));
        }, 1000);
      }
    };

    wsClient.subscribeToTaggerMetrics(handler);

    // Pre-flight dataset drift/rescan progress — fold into status_message
    // so the existing progress-section UI shows real-time scan progress
    // (e.g. "Drift check: dataset 25 — walked 850,000 files (142s)").
    const scanHandler = (ev: DatasetScanProgress) => {
      if (ev.scope !== "tagger") return;
      if (String(ev.run_id) !== String(run.run_id)) return;
      let msg = "";
      if (ev.phase === "drift_walk") {
        msg = `Drift check: dataset ${ev.dataset_id} — walked ${(ev.files_walked ?? 0).toLocaleString()} files`;
      } else if (ev.phase === "drift_done") {
        if ((ev.items_missing ?? 0) === 0 && (ev.items_new ?? 0) === 0) {
          msg = `Drift check: dataset ${ev.dataset_id} — no drift (${(ev.files_walked ?? 0).toLocaleString()} files)`;
        } else {
          msg = `Drift check: dataset ${ev.dataset_id} — ${ev.items_missing ?? 0} missing, ${ev.items_new ?? 0} new`;
        }
      } else if (ev.phase === "rescan") {
        msg = ev.message ?? `Rescanning dataset ${ev.dataset_id}...`;
        isScanningRef.current = true;
      } else if (ev.phase === "cleanup") {
        isScanningRef.current = false;
        setScanProgress(null);
        msg = ev.message ?? `Cleaning orphan cache for dataset ${ev.dataset_id}...`;
      }
      if (msg) {
        setRun(prev => ({ ...prev, status_message: msg }));
      }
    };
    wsClient.subscribeToDatasetScanProgress(scanHandler);

    // Generic type:"progress" events from scan_dataset (no run_id).
    // The "rescan" dataset_scan_progress event and the first progress event
    // may arrive out of order (both queued on the same backend thread, so
    // progress events can arrive before the rescan flag is set).
    // Detect scan events by their message pattern so the bar updates even
    // when isScanningRef hasn't been set yet.
    const progressHandler = (step: number, totalSteps: number, message: string) => {
      const isScanning = isScanningRef.current || (message ?? "").startsWith("Scanning:");
      if (!isScanning) return;
      if (!isScanningRef.current) isScanningRef.current = true;
      const pct = totalSteps > 0 ? step / totalSteps : 0;
      setScanProgress({ step, total: totalSteps, pct });
    };
    wsClient.subscribe(progressHandler);

    return () => {
      wsClient.unsubscribeFromTaggerMetrics(handler);
      wsClient.unsubscribeFromDatasetScanProgress(scanHandler);
      wsClient.unsubscribe(progressHandler);
      if (wsFlushRef.current) {
        clearTimeout(wsFlushRef.current);
        wsFlushRef.current = null;
      }
    };
  }, [run.run_id, extractEpochBoundaries]);

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
      setIterPerSecLong(null);
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
        setIterPerSecLong(null);
        iterRateSamplesRef.current = [];
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [run.status]);

  // 1Hz wall-clock tick for elapsed-time display.  Cheap (single
  // setState per second) and isolated from the WS-driven flush.
  useEffect(() => {
    const isActive = run.status === "running" || run.status === "starting";
    if (!isActive) return;
    const tick = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(tick);
  }, [run.status]);

  // Load full metrics history on mount (for resumed/completed runs)
  useEffect(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  // ────────────────────────────────────────────────────────────────────
  // Derived values for the progress header (elapsed / ETA / consistency)
  // ────────────────────────────────────────────────────────────────────

  // Session start: prefer last_resumed_at (resume case) over started_at
  // (initial run case).  If neither is set, we can't show elapsed time.
  const sessionStartMs = run.last_resumed_at
    ? new Date(run.last_resumed_at).getTime()
    : (run.started_at ? new Date(run.started_at).getTime() : null);
  const elapsedSec = sessionStartMs !== null
    ? Math.max(0, (nowMs - sessionStartMs) / 1000)
    : null;

  // Total steps: prefer the backend's reported value; fall back to
  // deriving from progress (current_step / progress) once progress > 0.
  const totalStepsEst: number | null = (() => {
    if (typeof run.total_steps === "number" && run.total_steps > 0) {
      return run.total_steps;
    }
    if (run.progress > 0 && run.current_step > 0) {
      return Math.round(run.current_step / run.progress);
    }
    return null;
  })();
  const remainingSteps = totalStepsEst !== null
    ? Math.max(0, totalStepsEst - run.current_step)
    : null;

  // ETA: short-window primary, long-window for consistency check.
  const etaShortSec = remainingSteps !== null && iterPerSec && iterPerSec > 0
    ? remainingSteps / iterPerSec
    : null;
  const etaLongSec = remainingSteps !== null && iterPerSecLong && iterPerSecLong > 0
    ? remainingSteps / iterPerSecLong
    : null;

  // Divergence: |short - long| / long.  null when either window is
  // unavailable.
  const etaDivergence = etaShortSec !== null && etaLongSec !== null && etaLongSec > 0
    ? Math.abs(etaShortSec - etaLongSec) / etaLongSec
    : null;
  const consistencyLevel: "stable" | "variable" | "unstable" | null =
    etaDivergence === null
      ? null
      : etaDivergence < 0.15 ? "stable"
      : etaDivergence < 0.40 ? "variable"
      : "unstable";

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
              <span className="text-gray-300">
                {scanProgress
                  ? `${(scanProgress.pct * 100).toFixed(1)}%`
                  : `${(run.progress * 100).toFixed(1)}%`}
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-500 ${scanProgress ? "bg-yellow-500" : "bg-blue-500"}`}
                style={{ width: `${(scanProgress ? scanProgress.pct : run.progress) * 100}%` }}
              />
            </div>
            {/* Elapsed (session) / ETA / consistency indicator */}
            {(elapsedSec !== null || etaShortSec !== null) && (
              <div className="flex items-center justify-between text-xs mt-1.5 font-mono">
                {elapsedSec !== null ? (
                  <span className="text-gray-500" title={
                    run.last_resumed_at
                      ? `Session resumed at ${new Date(run.last_resumed_at).toLocaleString()}`
                      : run.started_at
                        ? `Started at ${new Date(run.started_at).toLocaleString()}`
                        : ""
                  }>
                    elapsed {formatDuration(elapsedSec)}
                    {run.last_resumed_at && (
                      <span className="text-gray-600 ml-1">(this session)</span>
                    )}
                  </span>
                ) : <span />}
                {etaShortSec !== null && (
                  <span className="text-gray-400">
                    {consistencyLevel === "unstable" && etaLongSec !== null ? (
                      // Show range when divergence > 40%
                      <span title={`Recent rate (${iterPerSec?.toFixed(2)} it/s) differs >40% from session average (${iterPerSecLong?.toFixed(2)} it/s). Range shown.`}>
                        ETA{" "}
                        {formatDuration(Math.min(etaShortSec, etaLongSec))}
                        {"–"}
                        {formatDuration(Math.max(etaShortSec, etaLongSec))}
                        <span className="ml-1 text-red-400" aria-label="unstable">●</span>
                      </span>
                    ) : (
                      <span title={
                        etaLongSec !== null
                          ? `Recent rate ${iterPerSec?.toFixed(2)} it/s; session avg ${iterPerSecLong?.toFixed(2)} it/s`
                          : ""
                      }>
                        ETA {formatDuration(etaShortSec)}
                        <span className="text-gray-600 ml-1">
                          ({formatArrivalTime(etaShortSec, nowMs)})
                        </span>
                        {consistencyLevel === "variable" && (
                          <span className="ml-1 text-yellow-400" aria-label="variable">●</span>
                        )}
                        {consistencyLevel === "stable" && (
                          <span className="ml-1 text-green-500/60" aria-label="stable">●</span>
                        )}
                      </span>
                    )}
                  </span>
                )}
              </div>
            )}
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
              epochBoundaries={epochBoundaries}
            />
            <TaggerMetricChart
              data={metrics}
              valueKey="train_f1"
              secondaryValueKey="f1"
              secondaryColor="#22c55e"
              secondaryLabel="Val F1"
              color="#f97316"
              title="F1 Score"
              height={160}
              smoothable={true}
              defaultSmoothing={0.7}
              yMinFloor={0}
              epochBoundaries={epochBoundaries}
            />
            <TaggerMetricChart
              data={metrics}
              valueKey="precision"
              secondaryValueKey="recall"
              secondaryColor="#a78bfa"
              secondaryLabel="Recall"
              color="#38bdf8"
              title="Precision / Recall"
              height={120}
              smoothable={true}
              defaultSmoothing={0.7}
              yMinFloor={0}
              epochBoundaries={epochBoundaries}
            />
            <TaggerMetricChart
              data={metrics}
              valueKey="threshold"
              color="#06b6d4"
              title="Optimal Threshold"
              height={100}
              yMinFloor={0}
              epochBoundaries={epochBoundaries}
            />

            {/* FP/FN scatter plot — shown once first scatter data arrives */}
            {scatterData && scatterData.n_tags > 0 && (
              <FpFnScatterPlot data={scatterData} />
            )}

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
            {/* Danbooru augmentation metrics (only when enabled) */}
            <DanbooruMetricsPanel
              runId={run.run_id}
              active={run.status === "running" || run.status === "starting"}
            />

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
                  // Online Danbooru augmentation
                  enable_danbooru_augmentation: "Danbooru aug",
                  danbooru_tags: "Danbooru tags",
                  danbooru_injection_interval: "Injection interval",
                  danbooru_injection_batch_size_ratio: "Injection batch ratio",
                  danbooru_min_score: "Danbooru min score",
                  danbooru_max_posts_per_query: "Max posts / query",
                  danbooru_api_interval: "API interval (s)",
                  danbooru_dl_speed_kbps: "DL speed (kbps)",
                  danbooru_buffer_size: "Buffer size",
                  danbooru_vocab_expand: "Vocab expansion",
                  danbooru_new_tag_min_count: "New-tag min count",
                  danbooru_new_tag_lookback_days: "New-tag lookback (days)",
                  danbooru_new_tag_categories: "New-tag categories",
                  danbooru_new_tag_survey_interval: "Survey interval (s)",
                  danbooru_query_weight_static: "Weight: static",
                  danbooru_query_weight_new_tag: "Weight: new-tag",
                  danbooru_query_weight_low_f1: "Weight: low-F1",
                  danbooru_low_f1_enable: "Low-F1 collection",
                  danbooru_low_f1_threshold: "Low-F1 threshold",
                  danbooru_low_f1_top_k: "Low-F1 top-K",
                  danbooru_low_f1_min_posts: "Low-F1 min posts",
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
                // Danbooru augmentation: detail keys only meaningful when enabled;
                // new-tag (vocab-expansion) sub-keys only when vocab_expand is on.
                const danbooruOn = Boolean(cfg.enable_danbooru_augmentation);
                const vocabExpandOn = Boolean(cfg.danbooru_vocab_expand);
                const lowF1On = Boolean(cfg.danbooru_low_f1_enable);
                const DANBOORU_DETAIL_KEYS = new Set([
                  "danbooru_tags", "danbooru_injection_interval", "danbooru_injection_batch_size_ratio",
                  "danbooru_min_score", "danbooru_max_posts_per_query", "danbooru_api_interval",
                  "danbooru_dl_speed_kbps", "danbooru_buffer_size", "danbooru_vocab_expand",
                  "danbooru_new_tag_min_count", "danbooru_new_tag_lookback_days", "danbooru_new_tag_categories",
                  "danbooru_new_tag_survey_interval",
                  "danbooru_query_weight_static", "danbooru_query_weight_new_tag", "danbooru_query_weight_low_f1",
                  "danbooru_low_f1_enable", "danbooru_low_f1_threshold", "danbooru_low_f1_top_k",
                  "danbooru_low_f1_min_posts",
                ]);
                const DANBOORU_VOCAB_KEYS = new Set([
                  "danbooru_new_tag_min_count", "danbooru_new_tag_lookback_days", "danbooru_new_tag_categories",
                  "danbooru_new_tag_survey_interval",
                ]);
                // Low-F1 sub-parameters only meaningful when low-F1 collection is on.
                const DANBOORU_LOW_F1_KEYS = new Set([
                  "danbooru_low_f1_threshold", "danbooru_low_f1_top_k", "danbooru_low_f1_min_posts",
                ]);
                const entries = Object.entries(CONFIG_LABELS)
                  .map(([key, label]) => ({
                    key,
                    label,
                    value:
                      key === "loss_function" ? (cfg[key] ?? "asl")
                      : (key === "enable_danbooru_augmentation" || key === "danbooru_low_f1_enable") ? Boolean(cfg[key])
                      : cfg[key],
                  }))
                  .filter(({ key, value }) => {
                    if (value === undefined || value === null || value === "") return false;
                    if (LORA_ONLY_KEYS.has(key) && !isLora) return false;
                    if (ASL_ONLY_KEYS.has(key) && lossFn !== "asl") return false;
                    if (CS_ASL_KEYS.has(key) && !["cs_asl", "h_cs_asl", "la_s_asl"].includes(lossFn)) return false;
                    if (H_CS_ASL_KEYS.has(key) && lossFn !== "h_cs_asl") return false;
                    if (LR_SUB_KEYS.has(key) && !buildLR) return false;
                    if (DANBOORU_DETAIL_KEYS.has(key) && !danbooruOn) return false;
                    if (DANBOORU_VOCAB_KEYS.has(key) && !vocabExpandOn) return false;
                    if (DANBOORU_LOW_F1_KEYS.has(key) && !lowF1On) return false;
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
