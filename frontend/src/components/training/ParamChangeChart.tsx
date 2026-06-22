"use client";

import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { RefreshCw } from "lucide-react";
import { getTrainingMetrics, type MetricPoint, type EpochBoundary, type ResumeMarker } from "@/utils/api";
import { wsClient, type TrainingMetrics } from "@/utils/websocket";
import SharedMetricChart, { type ChartSeries } from "./SharedMetricChart";

interface ParamChangeChartProps {
  runId: number;
  isRunning: boolean;
}

type TabType = "update_norm" | "cumulative_drift";

const COMPONENTS = [
  { key: "unet", label: "U-Net/DiT", color: "#60a5fa" },
  { key: "te1", label: "TE1", color: "#34d399" },
  { key: "te2", label: "TE2", color: "#f59e0b" },
  { key: "ve", label: "VE", color: "#f87171" },
] as const;

/**
 * Parameter-change chart — thin wrapper over SharedMetricChart. Tabs between
 * step-wise update norm and cumulative drift (each: unet/te1/te2/ve). Param
 * values arrive via fetch/poll (not SSE); SSE only advances epoch/resume markers.
 */
export default function ParamChangeChart({ runId, isRunning }: ParamChangeChartProps) {
  const [tab, setTab] = useState<TabType>("update_norm");
  const [update, setUpdate] = useState<Record<string, MetricPoint[]>>({});
  const [drift, setDrift] = useState<Record<string, MetricPoint[]>>({});
  const [fetchedBoundaries, setFetchedBoundaries] = useState<EpochBoundary[]>([]);
  const [fetchedMarkers, setFetchedMarkers] = useState<ResumeMarker[]>([]);
  const [liveBoundaries, setLiveBoundaries] = useState<EpochBoundary[]>([]);
  const [liveMarkers, setLiveMarkers] = useState<ResumeMarker[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const liveEpochRef = useRef<{ epoch: number; maxStep: number } | null>(null);
  const seenResumesRef = useRef<Set<number>>(new Set());

  const fetchMetrics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const d = await getTrainingMetrics(runId);
      setUpdate({
        unet: d.param_update_norm_unet || [],
        te1: d.param_update_norm_te1 || [],
        te2: d.param_update_norm_te2 || [],
        ve: d.param_update_norm_ve || [],
      });
      setDrift({
        unet: d.param_cumulative_drift_unet || [],
        te1: d.param_cumulative_drift_te1 || [],
        te2: d.param_cumulative_drift_te2 || [],
        ve: d.param_cumulative_drift_ve || [],
      });
      setFetchedBoundaries(d.epoch_boundaries || []);
      setFetchedMarkers(d.resume_markers || []);
    } catch (err: any) {
      console.error("[ParamChangeChart] Error fetching metrics:", err);
      setError(err.message || "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => { fetchMetrics(); }, [runId, fetchMetrics]);

  // Poll while running (param metrics are not pushed over SSE).
  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(fetchMetrics, 15000);
    return () => clearInterval(id);
  }, [isRunning, fetchMetrics]);

  // SSE only advances epoch/resume markers live.
  useEffect(() => {
    if (!isRunning) return;
    wsClient.connect();
    const handle = (m: TrainingMetrics) => {
      if (m.run_id !== runId) return;
      const rs = m.resume_seq ?? 0;
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
      if (rs > 0 && !seenResumesRef.current.has(rs)) {
        seenResumesRef.current.add(rs);
        setLiveMarkers((prev) => prev.some((x) => x.resume_seq === rs) ? prev : [...prev, { resume_seq: rs, step: m.step }]);
      }
    };
    wsClient.subscribeToTrainingMetrics(handle);
    return () => wsClient.unsubscribeFromTrainingMetrics(handle);
  }, [isRunning, runId]);

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

  const active = tab === "update_norm" ? update : drift;
  const series = useMemo<ChartSeries[]>(() =>
    COMPONENTS
      .map((c) => ({ id: c.key, label: c.label, color: c.color, points: active[c.key] || [] }))
      .filter((s) => s.points.length > 0),
  [active]);

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <div className="flex gap-1">
          {(["update_norm", "cumulative_drift"] as TabType[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`text-[10px] px-1.5 py-0.5 rounded transition-colors ${tab === t ? "bg-blue-700 text-blue-100" : "bg-gray-700 hover:bg-gray-600 text-gray-300"}`}
            >
              {t === "update_norm" ? "Update norm" : "Cumulative drift"}
            </button>
          ))}
        </div>
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
        title={tab === "update_norm" ? "Param Update Norm" : "Param Cumulative Drift"}
        series={series}
        yMinFloor={0}
        defaultSmoothing={0.6}
        epochBoundaries={epochBoundaries}
        resumeMarkers={resumeMarkers}
      />
    </div>
  );
}
