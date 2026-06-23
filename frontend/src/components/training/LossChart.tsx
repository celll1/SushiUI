"use client";

import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { RefreshCw } from "lucide-react";
import { getTrainingMetrics, type MetricPoint, type EpochBoundary, type ResumeMarker } from "@/utils/api";
import { wsClient, type TrainingMetrics } from "@/utils/websocket";
import SharedMetricChart, { type ChartSeries } from "./SharedMetricChart";

interface LossChartProps {
  runId: number;
  isRunning: boolean;
}

/**
 * Loss chart — thin wrapper over SharedMetricChart. Fetches metrics_db (+ live
 * SSE updates) and renders loss (solid) + recon_loss (dashed) with epoch
 * boundary lines and resume markers.
 */
export default function LossChart({ runId, isRunning }: LossChartProps) {
  const [lossData, setLossData] = useState<MetricPoint[]>([]);
  const [reconLossData, setReconLossData] = useState<MetricPoint[]>([]);
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
      setLossData(data.loss);
      setReconLossData(data.recon_loss || []);
      setFetchedBoundaries(data.epoch_boundaries || []);
      setFetchedMarkers(data.resume_markers || []);
    } catch (err: any) {
      console.error("[LossChart] Error fetching metrics:", err);
      setError(err.message || "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => { fetchMetrics(); }, [runId, fetchMetrics]);

  // Auto-refresh while running: periodically re-fetch the (server-decimated) view
  // so the chart updates without pressing Refresh. Async fetch — never blocks the
  // training process (separate); backend uniform-samples to max_points.
  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(fetchMetrics, 7000);
    return () => clearInterval(id);
  }, [isRunning, fetchMetrics]);

  // Live SSE updates
  useEffect(() => {
    if (!isRunning) return;
    wsClient.connect();
    const handle = (m: TrainingMetrics) => {
      if (m.run_id !== runId) return;
      const rs = m.resume_seq ?? 0;
      const upsert = (prev: MetricPoint[], value: number): MetricPoint[] => {
        const pt: MetricPoint = { step: m.step, value, wall_time: Date.now() / 1000, resume_seq: rs };
        const i = prev.findIndex((p) => p.step === m.step);
        if (i >= 0) { const next = [...prev]; next[i] = pt; return next; }
        return [...prev, pt];
      };
      if (m.loss !== undefined && m.loss !== null) setLossData((p) => upsert(p, m.loss));
      if (m.recon_loss !== undefined && m.recon_loss !== null) setReconLossData((p) => upsert(p, m.recon_loss as number));

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

  const series = useMemo<ChartSeries[]>(() => [
    { id: "loss", label: "Loss", color: "#60a5fa", points: lossData },
    { id: "recon", label: "Recon", color: "#34d399", points: reconLossData, dashed: true },
  ], [lossData, reconLossData]);

  return (
    <div>
      <div className="flex items-center justify-end mb-1">
        <button
          onClick={fetchMetrics}
          disabled={loading}
          className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 inline-flex items-center gap-1 disabled:opacity-50"
          title="Refresh"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} /> Refresh
        </button>
      </div>
      {error && <div className="text-xs text-red-400 mb-1">{error}</div>}
      <SharedMetricChart
        title="Loss"
        series={series}
        yMinFloor={0}
        epochBoundaries={epochBoundaries}
        resumeMarkers={resumeMarkers}
      />
    </div>
  );
}
