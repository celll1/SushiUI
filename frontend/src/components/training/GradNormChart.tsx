"use client";

import { useEffect, useState, useMemo } from "react";
import { RefreshCw } from "lucide-react";
import { type MetricPoint } from "@/utils/api";
import SharedMetricChart, { type ChartSeries } from "./SharedMetricChart";
import CfgConditionFilter, { type CfgCondFilter } from "./CfgConditionFilter";
import { useTrainingMetrics } from "./TrainingMetricsContext";

const EMPTY: MetricPoint[] = [];

// The optimizer step's total grad norm, labelled by how its items were drawn.
// Unlike the loss this is not a split: a norm is one number for the whole
// accumulated batch, so the backend emits it only when EVERY item behind it was
// drawn the same way. That is why the filter offers no per-component breakdown
// here — U-Net/TE norms are not attributable either.
const SPLIT_KEY: Record<Exclude<CfgCondFilter, "all">, string> = {
  null: "gnorm_null",
  cond: "gnorm_cond",
};

/** Gradient-norm chart — thin wrapper over SharedMetricChart, reading the
 *  shared TrainingMetricsContext. */
export default function GradNormChart() {
  const {
    seriesByKey, extraSeries, epochBoundaries, resumeMarkers, loading, error, refresh,
  } = useTrainingMetrics();
  const total = seriesByKey.grad_norm ?? EMPTY;
  const te = seriesByKey.grad_norm_text_encoder ?? EMPTY;
  const te1 = seriesByKey.grad_norm_text_encoder_1 ?? EMPTY;
  const te2 = seriesByKey.grad_norm_text_encoder_2 ?? EMPTY;
  const unet = seriesByKey.grad_norm_unet ?? EMPTY;
  const ve = seriesByKey.grad_norm_vision_encoder ?? EMPTY;
  // Only the CFG-condition-labelled norms are read out of extra_metrics here;
  // the rest of that channel is loss-scale and belongs on the loss chart.
  const gnormNull = extraSeries[SPLIT_KEY.null] ?? EMPTY;
  const gnormCond = extraSeries[SPLIT_KEY.cond] ?? EMPTY;

  const [cfgFilter, setCfgFilter] = useState<CfgCondFilter>("all");
  const splitAvailable = gnormNull.length > 0 || gnormCond.length > 0;
  const emptySides = useMemo<CfgCondFilter[]>(() => {
    const out: CfgCondFilter[] = [];
    if (!gnormNull.length) out.push("null");
    if (!gnormCond.length) out.push("cond");
    return out;
  }, [gnormNull, gnormCond]);
  useEffect(() => {
    if (cfgFilter !== "all" && emptySides.includes(cfgFilter)) setCfgFilter("all");
  }, [cfgFilter, emptySides]);

  const series = useMemo<ChartSeries[]>(() => {
    // Under a filter only the labelled total survives: the per-component norms
    // carry no label, and the unlabelled total is the whole batch either way.
    if (cfgFilter !== "all") {
      const points = cfgFilter === "null" ? gnormNull : gnormCond;
      return ([{
        id: `gnorm_${cfgFilter}`,
        label: cfgFilter === "null" ? "Total (null)" : "Total (cond)",
        color: cfgFilter === "null" ? "#fb7185" : "#4ade80",
        points,
      }] as ChartSeries[]).filter((s) => s.points.length > 0);
    }
    return ([
      { id: "total", label: "Total", color: "#60a5fa", points: total },
      { id: "unet", label: "U-Net/DiT", color: "#34d399", points: unet },
      { id: "te", label: "TE", color: "#f472b6", points: te },
      { id: "te1", label: "TE1", color: "#a78bfa", points: te1 },
      { id: "te2", label: "TE2", color: "#facc15", points: te2 },
      { id: "ve", label: "VE", color: "#22d3ee", points: ve },
    ] as ChartSeries[]).filter((s) => s.points.length > 0);
  }, [total, unet, te, te1, te2, ve, cfgFilter, gnormNull, gnormCond]);

  return (
    <div>
      <div className="flex items-center justify-end mb-1">
        <CfgConditionFilter
          value={cfgFilter}
          onChange={setCfgFilter}
          available={splitAvailable}
          emptyValues={emptySides}
        />
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
        title={cfgFilter === "all" ? "Gradient Norm"
          : cfgFilter === "null" ? "Gradient Norm — CFG null steps"
          : "Gradient Norm — conditional steps"}
        series={series}
        yMinFloor={0}
        allowLogScale
        epochBoundaries={epochBoundaries}
        resumeMarkers={resumeMarkers}
      />
    </div>
  );
}
