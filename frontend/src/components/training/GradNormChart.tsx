"use client";

import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { RefreshCw } from "lucide-react";
import { getTrainingMetrics, type MetricPoint, type EpochBoundary, type ResumeMarker } from "@/utils/api";
import { wsClient, type TrainingMetrics } from "@/utils/websocket";
import SharedMetricChart, { type ChartSeries } from "./SharedMetricChart";

interface GradNormChartProps {
  runId: number;
  isRunning: boolean;
}

/** Gradient-norm chart — thin wrapper over SharedMetricChart. */
export default function GradNormChart({ runId, isRunning }: GradNormChartProps) {
  const [total, setTotal] = useState<MetricPoint[]>([]);
  const [te, setTe] = useState<MetricPoint[]>([]);
  const [te1, setTe1] = useState<MetricPoint[]>([]);
  const [te2, setTe2] = useState<MetricPoint[]>([]);
  const [unet, setUnet] = useState<MetricPoint[]>([]);
  const [ve, setVe] = useState<MetricPoint[]>([]);
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
      const data = await getTrainingMetrics(runId);
      setTotal(data.grad_norm || []);
      setTe(data.grad_norm_text_encoder || []);
      setTe1(data.grad_norm_text_encoder_1 || []);
      setTe2(data.grad_norm_text_encoder_2 || []);
      setUnet(data.grad_norm_unet || []);
      setVe(data.grad_norm_vision_encoder || []);
      setFetchedBoundaries(data.epoch_boundaries || []);
      setFetchedMarkers(data.resume_markers || []);
    } catch (err: any) {
      console.error("[GradNormChart] Error fetching metrics:", err);
      setError(err.message || "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => { fetchMetrics(); }, [runId, fetchMetrics]);

  useEffect(() => {
    if (!isRunning) return;
    wsClient.connect();
    const handle = (m: TrainingMetrics) => {
      if (m.run_id !== runId) return;
      const rs = m.resume_seq ?? 0;
      const upsert = (setter: React.Dispatch<React.SetStateAction<MetricPoint[]>>, value: number | undefined | null) => {
        if (value === undefined || value === null) return;
        setter((prev) => {
          const pt: MetricPoint = { step: m.step, value, wall_time: Date.now() / 1000, resume_seq: rs };
          const i = prev.findIndex((p) => p.step === m.step);
          if (i >= 0) { const next = [...prev]; next[i] = pt; return next; }
          return [...prev, pt];
        });
      };
      upsert(setTotal, m.grad_norm);
      upsert(setTe, m.grad_norm_text_encoder);
      upsert(setTe1, m.grad_norm_text_encoder_1);
      upsert(setTe2, m.grad_norm_text_encoder_2);
      upsert(setUnet, m.grad_norm_unet);
      upsert(setVe, m.grad_norm_vision_encoder);
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

  const series = useMemo<ChartSeries[]>(() => ([
    { id: "total", label: "Total", color: "#60a5fa", points: total },
    { id: "unet", label: "U-Net/DiT", color: "#34d399", points: unet },
    { id: "te", label: "TE", color: "#f472b6", points: te },
    { id: "te1", label: "TE1", color: "#a78bfa", points: te1 },
    { id: "te2", label: "TE2", color: "#facc15", points: te2 },
    { id: "ve", label: "VE", color: "#22d3ee", points: ve },
  ] as ChartSeries[]).filter((s) => s.points.length > 0), [total, unet, te, te1, te2, ve]);

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
        title="Gradient Norm"
        series={series}
        yMinFloor={0}
        allowLogScale
        epochBoundaries={epochBoundaries}
        resumeMarkers={resumeMarkers}
      />
    </div>
  );
}
