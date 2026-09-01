"use client";

import { useEffect, useState, useMemo } from "react";
import { RefreshCw } from "lucide-react";
import { type MetricPoint } from "@/utils/api";
import SharedMetricChart, { type ChartSeries } from "./SharedMetricChart";
import CfgConditionFilter, { type CfgCondFilter } from "./CfgConditionFilter";
import { useTrainingMetrics } from "./TrainingMetricsContext";
import { fallbackColor } from "./metricCatalog";

const EMPTY: MetricPoint[] = [];

// The aligned-CFG-null split of the main loss, emitted by BaseTrainer only for
// a run with a nonzero cfg_uncond_drop_rate. Filtering to one of them replaces
// the pooled "Loss" series rather than adding to it: the pooled series IS the
// blend of these two, so showing all three at once is what the filter exists
// to escape.
const SPLIT_KEY: Record<Exclude<CfgCondFilter, "all">, string> = {
  null: "loss_null",
  cond: "loss_cond",
};

/**
 * Loss chart — thin wrapper over SharedMetricChart. Reads the shared
 * TrainingMetricsContext (one fetch/poll/SSE source for the whole monitor) and
 * renders loss (solid) + recon_loss (dashed) with epoch boundary lines and
 * resume markers.
 */
export default function LossChart() {
  const {
    seriesByKey, extraSeries: extraData, defs: extraDefs,
    epochBoundaries, resumeMarkers, loading, error, refresh,
  } = useTrainingMetrics();
  const lossData = seriesByKey.loss ?? EMPTY;
  const reconLossData = seriesByKey.recon ?? EMPTY;

  // CFG-condition filter. Available only once the run has actually recorded a
  // split (it starts emitting on its next start), and each side is offered only
  // when that side has steps — at a low drop rate the null side can be empty in
  // a short window, and a filter that silently blanks the chart is worse than
  // one that says there is nothing there.
  const [cfgFilter, setCfgFilter] = useState<CfgCondFilter>("all");
  const splitAvailable = (extraData[SPLIT_KEY.null]?.length ?? 0) > 0
    || (extraData[SPLIT_KEY.cond]?.length ?? 0) > 0;
  const emptySides = useMemo<CfgCondFilter[]>(() => (
    (["null", "cond"] as const).filter((s) => !(extraData[SPLIT_KEY[s]]?.length))
  ), [extraData]);
  // Never leave the chart on a side the current run has no points for (e.g.
  // after switching runs).
  useEffect(() => {
    if (cfgFilter !== "all" && emptySides.includes(cfgFilter)) setCfgFilter("all");
  }, [cfgFilter, emptySides]);

  const series = useMemo<ChartSeries[]>(() => {
    const filtered = cfgFilter !== "all";
    // Under a filter the pooled loss and its recon counterpart are dropped:
    // both are computed over every item in the step regardless of how it was
    // drawn, so neither belongs to the selected side.
    const base: ChartSeries[] = filtered ? [] : [
      { id: "loss", label: "Loss", color: "#60a5fa", points: lossData },
      { id: "recon", label: "Recon", color: "#34d399", points: reconLossData, dashed: true },
    ];
    const dropped = filtered
      ? Object.values(SPLIT_KEY).filter((k) => k !== SPLIT_KEY[cfgFilter as Exclude<CfgCondFilter, "all">])
      : [];
    // Dynamic bespoke metrics: one series per key, styled from the backend
    // registry def (falling back to the raw key + a hashed color when unknown).
    const extra: ChartSeries[] = Object.entries(extraData)
      .filter(([key]) => !dropped.includes(key))
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, points]) => {
        const def = extraDefs[key] || {};
        // Any lr/lr_* series (e.g. per-component actual LRs logged by
        // base_trainer for multi-group runs) belongs on the secondary axis
        // even if it isn't in the registry yet -- e.g. the g{i}/lr_controlnet
        // fallback keys base_trainer emits for components not covered by
        // metric_registry.py's curated entries.
        const isLrSeries = /^lr(_|$)/.test(key);
        return {
          id: `extra:${key}`,
          label: def.label || key,
          color: def.color || fallbackColor(key),
          points,
          dashed: def.dashed ?? true,
          secondaryAxis: def.axis === "right" || isLrSeries,
        } as ChartSeries;
      });
    return [...base, ...extra].filter((s) => s.points.length > 0);
  }, [lossData, reconLossData, extraData, extraDefs, cfgFilter]);

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
        title={cfgFilter === "all" ? "Loss"
          : cfgFilter === "null" ? "Loss — CFG null items"
          : "Loss — conditional items"}
        series={series}
        yMinFloor={0}
        allowLogScale
        epochBoundaries={epochBoundaries}
        resumeMarkers={resumeMarkers}
      />
    </div>
  );
}
