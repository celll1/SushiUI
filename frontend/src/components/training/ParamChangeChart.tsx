"use client";

import { useState, useMemo } from "react";
import { RefreshCw } from "lucide-react";
import { type MetricPoint } from "@/utils/api";
import SharedMetricChart, { type ChartSeries } from "./SharedMetricChart";
import { useTrainingMetrics } from "./TrainingMetricsContext";

const EMPTY: MetricPoint[] = [];

type TabType = "update_norm" | "cumulative_drift";

const COMPONENTS = [
  { key: "unet", label: "U-Net/DiT", color: "#60a5fa" },
  { key: "te1", label: "TE1", color: "#34d399" },
  { key: "te2", label: "TE2", color: "#f59e0b" },
  { key: "ve", label: "VE", color: "#f87171" },
] as const;

/**
 * Parameter-change chart — thin wrapper over SharedMetricChart, reading the
 * shared TrainingMetricsContext. Tabs between step-wise update norm and
 * cumulative drift (each: unet/te1/te2/ve).
 */
export default function ParamChangeChart() {
  const [tab, setTab] = useState<TabType>("update_norm");
  const { seriesByKey, epochBoundaries, resumeMarkers, loading, error, refresh } = useTrainingMetrics();

  const prefix = tab === "update_norm" ? "param_update_norm_" : "param_cumulative_drift_";
  const series = useMemo<ChartSeries[]>(() =>
    COMPONENTS
      .map((c) => ({ id: c.key, label: c.label, color: c.color, points: seriesByKey[prefix + c.key] ?? EMPTY }))
      .filter((s) => s.points.length > 0),
  [seriesByKey, prefix]);

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
          onClick={refresh}
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
        defaultSmoothing={0.9}
        epochBoundaries={epochBoundaries}
        resumeMarkers={resumeMarkers}
      />
    </div>
  );
}
