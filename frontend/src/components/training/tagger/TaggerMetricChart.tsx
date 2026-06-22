"use client";

import { useMemo } from "react";
import type { TaggerTrainingMetric } from "@/utils/api";
import SharedMetricChart, { type ChartSeries } from "../SharedMetricChart";

// Re-exported for TaggerTrainingMonitor (kept stable across the shared-core refactor).
export interface EpochBoundary {
  epoch: number;
  step: number;
}

type ValueKey = "loss" | "f1" | "threshold" | "train_f1" | "precision" | "recall";

interface TaggerMetricChartProps {
  data: TaggerTrainingMetric[];
  valueKey: ValueKey;
  /** Color for the initial (resume_seq 0) curve; other resumes cycle the palette. */
  color: string;
  title: string;
  height?: number;
  smoothable?: boolean;
  defaultSmoothing?: number;
  yMinFloor?: number;
  /** Optional secondary series (e.g. val F1 overlaid on train F1) — dashed. */
  secondaryValueKey?: ValueKey;
  secondaryColor?: string;
  secondaryLabel?: string;
  epochBoundaries?: EpochBoundary[];
}

// Dark-background palette for per-resume curves (resume_seq 0 uses caller's color).
const RESUME_PALETTE = ["#60a5fa", "#f97316", "#34d399", "#f472b6", "#a78bfa", "#facc15", "#22d3ee"];
const colorForResume = (seq: number, fallback: string) =>
  seq === 0 ? fallback : RESUME_PALETTE[seq % RESUME_PALETTE.length];
const labelForResume = (seq: number) => (seq === 0 ? "Initial" : `Resume #${seq}`);

/**
 * Tagger metric chart — thin wrapper over SharedMetricChart. Splits the chosen
 * metric into one curve per resume_seq (resume "split" mode), and overlays the
 * optional secondary metric as a single dashed series.
 */
export default function TaggerMetricChart({
  data,
  valueKey,
  color,
  title,
  height = 160,
  smoothable = false,
  defaultSmoothing = 0,
  yMinFloor = -Infinity,
  secondaryValueKey,
  secondaryColor = "#f59e0b",
  secondaryLabel = "secondary",
  epochBoundaries,
}: TaggerMetricChartProps) {
  const series = useMemo<ChartSeries[]>(() => {
    // Group primary by resume_seq → one curve per resume.
    const groups = new Map<number, { step: number; value: number; resume_seq: number }[]>();
    for (const r of data) {
      const v = (r as any)[valueKey];
      if (v === undefined || v === null || !Number.isFinite(v)) continue;
      const seq = r.resume_seq ?? 0;
      if (!groups.has(seq)) groups.set(seq, []);
      groups.get(seq)!.push({ step: r.step, value: v, resume_seq: seq });
    }
    const multi = groups.size > 1;
    const out: ChartSeries[] = [];
    for (const seq of [...groups.keys()].sort((a, b) => a - b)) {
      out.push({
        id: `${valueKey}:r${seq}`,
        label: multi ? `${title} (${labelForResume(seq)})` : title,
        color: colorForResume(seq, color),
        points: groups.get(seq)!.sort((a, b) => a.step - b.step),
      });
    }
    // Secondary: merge all resumes into one dashed series.
    if (secondaryValueKey) {
      const secPts = data
        .map((r) => ({ step: r.step, value: (r as any)[secondaryValueKey], resume_seq: r.resume_seq ?? 0 }))
        .filter((p) => p.value !== undefined && p.value !== null && Number.isFinite(p.value))
        .sort((a, b) => a.step - b.step) as { step: number; value: number; resume_seq: number }[];
      if (secPts.length > 0) {
        out.push({ id: `sec:${secondaryValueKey}`, label: secondaryLabel, color: secondaryColor, points: secPts, dashed: true, rawRange: true });
      }
    }
    return out;
  }, [data, valueKey, color, title, secondaryValueKey, secondaryColor, secondaryLabel]);

  return (
    <SharedMetricChart
      title={title}
      series={series}
      height={height}
      smoothable={smoothable}
      defaultSmoothing={defaultSmoothing}
      yMinFloor={yMinFloor}
      bounded={valueKey !== "loss"}
      epochBoundaries={epochBoundaries}
    />
  );
}
