"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  getTrainingMetrics,
  type MetricPoint,
  type EpochBoundary,
  type ResumeMarker,
  type MetricSeriesDef,
  type TrainingMetrics as MetricsPayload,
} from "@/utils/api";
import { wsClient, type TrainingMetrics as LiveMetrics } from "@/utils/websocket";

/**
 * Single source of per-run training metrics for every chart in the monitor.
 *
 * Replaces three independent copies of the same fetch + 7s poll + SSE
 * subscription + epoch/resume derivation (the former loss / grad-norm charts
 * and ParamChangeChart), which issued three requests for one payload and each
 * re-derived identical epoch boundaries and resume markers.
 */

/** DB columns exposed under a stable series key. `recon_loss` is keyed `recon`
 *  to match the id the loss chart has always used. `learning_rate` is absent on
 *  purpose — see the `lr` fold in seriesByKey below. */
const BUILTIN_COLUMNS: { key: string; field: keyof MetricsPayload }[] = [
  { key: "loss", field: "loss" },
  { key: "recon", field: "recon_loss" },
  { key: "grad_norm", field: "grad_norm" },
  { key: "grad_norm_unet", field: "grad_norm_unet" },
  { key: "grad_norm_text_encoder", field: "grad_norm_text_encoder" },
  { key: "grad_norm_text_encoder_1", field: "grad_norm_text_encoder_1" },
  { key: "grad_norm_text_encoder_2", field: "grad_norm_text_encoder_2" },
  { key: "grad_norm_vision_encoder", field: "grad_norm_vision_encoder" },
  { key: "param_update_norm_unet", field: "param_update_norm_unet" },
  { key: "param_update_norm_te1", field: "param_update_norm_te1" },
  { key: "param_update_norm_te2", field: "param_update_norm_te2" },
  { key: "param_update_norm_ve", field: "param_update_norm_ve" },
  { key: "param_cumulative_drift_unet", field: "param_cumulative_drift_unet" },
  { key: "param_cumulative_drift_te1", field: "param_cumulative_drift_te1" },
  { key: "param_cumulative_drift_te2", field: "param_cumulative_drift_te2" },
  { key: "param_cumulative_drift_ve", field: "param_cumulative_drift_ve" },
];

/** The subset of built-ins that also arrives per-step over SSE. Param norms are
 *  not pushed, so polling remains their only live source. */
const LIVE_COLUMNS: { key: string; field: keyof LiveMetrics }[] = [
  { key: "loss", field: "loss" },
  { key: "recon", field: "recon_loss" },
  { key: "grad_norm", field: "grad_norm" },
  { key: "grad_norm_unet", field: "grad_norm_unet" },
  { key: "grad_norm_text_encoder", field: "grad_norm_text_encoder" },
  { key: "grad_norm_text_encoder_1", field: "grad_norm_text_encoder_1" },
  { key: "grad_norm_text_encoder_2", field: "grad_norm_text_encoder_2" },
  { key: "grad_norm_vision_encoder", field: "grad_norm_vision_encoder" },
  { key: "learning_rate", field: "learning_rate" },
];

export interface TrainingMetricsValue {
  /** Every series the run has, built-in columns and bespoke extras alike. */
  seriesByKey: Record<string, MetricPoint[]>;
  /** The bespoke extra_metrics channel only (the set the registry describes). */
  extraSeries: Record<string, MetricPoint[]>;
  defs: Record<string, MetricSeriesDef>;
  epochBoundaries: EpochBoundary[];
  resumeMarkers: ResumeMarker[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

const Ctx = createContext<TrainingMetricsValue | null>(null);

export function useTrainingMetrics(): TrainingMetricsValue {
  const v = useContext(Ctx);
  if (!v) throw new Error("useTrainingMetrics must be used inside <TrainingMetricsProvider>");
  return v;
}

function upsert(prev: MetricPoint[], step: number, value: number, resumeSeq: number): MetricPoint[] {
  const pt: MetricPoint = { step, value, wall_time: Date.now() / 1000, resume_seq: resumeSeq };
  const i = prev.findIndex((p) => p.step === step);
  if (i >= 0) { const next = [...prev]; next[i] = pt; return next; }
  return [...prev, pt];
}

export function TrainingMetricsProvider({
  runId, isRunning, children,
}: { runId: number; isRunning: boolean; children: React.ReactNode }) {
  const [builtins, setBuiltins] = useState<Record<string, MetricPoint[]>>({});
  const [extras, setExtras] = useState<Record<string, MetricPoint[]>>({});
  const [defs, setDefs] = useState<Record<string, MetricSeriesDef>>({});
  const [fetchedBoundaries, setFetchedBoundaries] = useState<EpochBoundary[]>([]);
  const [fetchedMarkers, setFetchedMarkers] = useState<ResumeMarker[]>([]);
  const [liveBoundaries, setLiveBoundaries] = useState<EpochBoundary[]>([]);
  const [liveMarkers, setLiveMarkers] = useState<ResumeMarker[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Live epoch/resume tracking (refs survive re-renders without re-subscribing).
  const liveEpochRef = useRef<{ epoch: number; maxStep: number } | null>(null);
  const seenResumesRef = useRef<Set<number>>(new Set());

  const fetchMetrics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getTrainingMetrics(runId);
      const cols = data as unknown as Record<string, MetricPoint[] | undefined>;
      const next: Record<string, MetricPoint[]> = {};
      for (const c of BUILTIN_COLUMNS) next[c.key] = cols[c.field] || [];
      next.learning_rate = data.learning_rate || [];
      setBuiltins(next);
      setExtras(data.extra_metrics || {});
      setDefs(data.extra_metric_defs || {});
      setFetchedBoundaries(data.epoch_boundaries || []);
      setFetchedMarkers(data.resume_markers || []);
    } catch (err: any) {
      console.error("[TrainingMetrics] Error fetching metrics:", err);
      setError(err.message || "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => { fetchMetrics(); }, [runId, fetchMetrics]);

  // Auto-refresh while running: periodically re-fetch the (server-decimated)
  // view so the charts update without pressing Refresh. Async — never blocks
  // the training process; the backend uniform-samples to max_points.
  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(fetchMetrics, 7000);
    return () => clearInterval(id);
  }, [isRunning, fetchMetrics]);

  // Reset the live-derived state when the viewed run changes, so a previous
  // run's boundaries/resumes do not bleed into the new one before its fetch
  // lands.
  useEffect(() => {
    liveEpochRef.current = null;
    seenResumesRef.current = new Set();
    setLiveBoundaries([]);
    setLiveMarkers([]);
  }, [runId]);

  useEffect(() => {
    if (!isRunning) return;
    wsClient.connect();
    const handle = (m: LiveMetrics) => {
      if (m.run_id !== runId) return;
      const rs = m.resume_seq ?? 0;
      const live = m as unknown as Record<string, number | undefined | null>;
      setBuiltins((prev) => {
        let next: Record<string, MetricPoint[]> | null = null;
        for (const c of LIVE_COLUMNS) {
          const v = live[c.field];
          if (v === undefined || v === null) continue;
          if (!next) next = { ...prev };
          next[c.key] = upsert(prev[c.key] || [], m.step, v, rs);
        }
        return next ?? prev;
      });
      if (m.extra_metrics) {
        const em = m.extra_metrics;
        setExtras((prev) => {
          const next = { ...prev };
          for (const [k, v] of Object.entries(em)) {
            if (v === undefined || v === null) continue;
            next[k] = upsert(next[k] || [], m.step, v as number, rs);
          }
          return next;
        });
      }

      // Live epoch boundary: when epoch increments, the previous epoch ended at its max step.
      if (m.epoch !== undefined && m.epoch !== null) {
        const cur = liveEpochRef.current;
        if (cur && m.epoch > cur.epoch) {
          const ended = { epoch: cur.epoch, step: cur.maxStep };
          setLiveBoundaries((prev) => prev.some((b) => b.epoch === ended.epoch) ? prev : [...prev, ended]);
          liveEpochRef.current = { epoch: m.epoch, maxStep: m.step };
        } else {
          liveEpochRef.current = { epoch: m.epoch, maxStep: Math.max(cur?.maxStep ?? 0, m.step) };
        }
      }
      // Live resume marker
      if (rs > 0 && !seenResumesRef.current.has(rs)) {
        seenResumesRef.current.add(rs);
        setLiveMarkers((prev) => prev.some((x) => x.resume_seq === rs) ? prev : [...prev, { resume_seq: rs, step: m.step }]);
      }
    };
    wsClient.subscribeToTrainingMetrics(handle);
    return () => wsClient.unsubscribeFromTrainingMetrics(handle);
  }, [isRunning, runId]);

  // Merge fetched (authoritative) + live boundaries/markers.
  const epochBoundaries = useMemo<EpochBoundary[]>(() => {
    const map = new Map<number, number>();
    for (const b of fetchedBoundaries) map.set(b.epoch, b.step);
    for (const b of liveBoundaries) if (!map.has(b.epoch)) map.set(b.epoch, b.step);
    return [...map.entries()].sort((a, b) => a[0] - b[0]).map(([epoch, step]) => ({ epoch, step }));
  }, [fetchedBoundaries, liveBoundaries]);

  const resumeMarkers = useMemo<ResumeMarker[]>(() => {
    const map = new Map<number, number>();
    for (const m of fetchedMarkers) map.set(m.resume_seq, m.step);
    for (const m of liveMarkers) if (!map.has(m.resume_seq)) map.set(m.resume_seq, m.step);
    return [...map.entries()].sort((a, b) => a[0] - b[0]).map(([resume_seq, step]) => ({ resume_seq, step }));
  }, [fetchedMarkers, liveMarkers]);

  const seriesByKey = useMemo<Record<string, MetricPoint[]>>(() => {
    const out: Record<string, MetricPoint[]> = {};
    for (const c of BUILTIN_COLUMNS) if (builtins[c.key]?.length) out[c.key] = builtins[c.key];
    for (const [k, pts] of Object.entries(extras)) if (pts.length) out[k] = pts;
    // `lr` COLLISION, resolved explicitly rather than by key order: the
    // extra_metrics "lr" series is the ACTUALLY-APPLIED per-step learning rate
    // read off optimizer.param_groups, while the learning_rate DB column is the
    // configured value. The applied one wins; the column is only a fallback for
    // a run that predates the extra metric.
    const lr = extras.lr?.length ? extras.lr : builtins.learning_rate;
    if (lr?.length) out.lr = lr;
    return out;
  }, [builtins, extras]);

  const value = useMemo<TrainingMetricsValue>(() => ({
    seriesByKey, extraSeries: extras, defs,
    epochBoundaries, resumeMarkers, loading, error, refresh: fetchMetrics,
  }), [seriesByKey, extras, defs, epochBoundaries, resumeMarkers, loading, error, fetchMetrics]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
