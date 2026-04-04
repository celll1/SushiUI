"use client";

import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { RefreshCw } from "lucide-react";
import { getTrainingMetrics } from "@/utils/api";

interface MetricPoint {
  step: number;
  value: number;
  wall_time?: number;
}

const calculateSmoothing = (data: MetricPoint[], smoothingFactor: number): MetricPoint[] => {
  if (data.length === 0 || smoothingFactor === 0) return data;
  const smoothed: MetricPoint[] = [];
  let lastSmoothed = data[0].value;
  for (const point of data) {
    lastSmoothed = lastSmoothed * smoothingFactor + point.value * (1 - smoothingFactor);
    smoothed.push({ step: point.step, value: lastSmoothed, wall_time: point.wall_time });
  }
  return smoothed;
};

const calculateRobustYRange = (values: number[]): { min: number; max: number } => {
  if (values.length === 0) return { min: 0, max: 1 };
  const validValues = values.filter(v => isFinite(v) && !isNaN(v));
  if (validValues.length === 0) return { min: 0, max: 1 };
  const sorted = [...validValues].sort((a, b) => a - b);
  const lowerIndex = Math.floor(sorted.length * 0.01);
  const upperIndex = Math.ceil(sorted.length * 0.99) - 1;
  const pMin = sorted[Math.max(0, lowerIndex)];
  const pMax = sorted[Math.min(sorted.length - 1, upperIndex)];
  const range = pMax - pMin;
  const padding = range * 0.05;
  return { min: Math.max(0, pMin - padding), max: pMax + padding };
};

interface ParamChangeChartProps {
  runId: number;
  isRunning: boolean;
}

type TabType = "update_norm" | "cumulative_drift";

const SERIES = [
  { key: "unet", label: "U-Net", color: "#60a5fa" },
  { key: "te1", label: "TE1", color: "#34d399" },
  { key: "te2", label: "TE2", color: "#f59e0b" },
  { key: "ve", label: "VE", color: "#f87171" },
] as const;

export default function ParamChangeChart({ runId, isRunning }: ParamChangeChartProps) {
  const [tab, setTab] = useState<TabType>("update_norm");

  // Update norm data
  const [updateNormUNet, setUpdateNormUNet] = useState<MetricPoint[]>([]);
  const [updateNormTE1, setUpdateNormTE1] = useState<MetricPoint[]>([]);
  const [updateNormTE2, setUpdateNormTE2] = useState<MetricPoint[]>([]);
  const [updateNormVE, setUpdateNormVE] = useState<MetricPoint[]>([]);

  // Cumulative drift data
  const [driftUNet, setDriftUNet] = useState<MetricPoint[]>([]);
  const [driftTE1, setDriftTE1] = useState<MetricPoint[]>([]);
  const [driftTE2, setDriftTE2] = useState<MetricPoint[]>([]);
  const [driftVE, setDriftVE] = useState<MetricPoint[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [smoothingFactor, setSmoothingFactor] = useState(0.6);
  const [showUNet, setShowUNet] = useState(true);
  const [showTE1, setShowTE1] = useState(true);
  const [showTE2, setShowTE2] = useState(true);
  const [showVE, setShowVE] = useState(true);
  const [pollingInterval, setPollingInterval] = useState<number>(0);

  const [tooltip, setTooltip] = useState<{
    x: number; y: number; step: number;
    values: Record<string, number | undefined>;
    smoothValues: Record<string, number | undefined>;
  } | null>(null);

  const svgRef = useRef<SVGSVGElement>(null);
  const [svgWidth, setSvgWidth] = useState<number>(550);

  // Active data sets based on tab
  const activeData = useMemo(() => {
    if (tab === "update_norm") {
      return { unet: updateNormUNet, te1: updateNormTE1, te2: updateNormTE2, ve: updateNormVE };
    } else {
      return { unet: driftUNet, te1: driftTE1, te2: driftTE2, ve: driftVE };
    }
  }, [tab, updateNormUNet, updateNormTE1, updateNormTE2, updateNormVE, driftUNet, driftTE1, driftTE2, driftVE]);

  const smoothData = useMemo(() => ({
    unet: calculateSmoothing(activeData.unet, smoothingFactor),
    te1: calculateSmoothing(activeData.te1, smoothingFactor),
    te2: calculateSmoothing(activeData.te2, smoothingFactor),
    ve: calculateSmoothing(activeData.ve, smoothingFactor),
  }), [activeData, smoothingFactor]);

  const fetchMetrics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getTrainingMetrics(runId);
      setUpdateNormUNet(data.param_update_norm_unet ?? []);
      setUpdateNormTE1(data.param_update_norm_te1 ?? []);
      setUpdateNormTE2(data.param_update_norm_te2 ?? []);
      setUpdateNormVE(data.param_update_norm_ve ?? []);
      setDriftUNet(data.param_cumulative_drift_unet ?? []);
      setDriftTE1(data.param_cumulative_drift_te1 ?? []);
      setDriftTE2(data.param_cumulative_drift_te2 ?? []);
      setDriftVE(data.param_cumulative_drift_ve ?? []);
    } catch (err: any) {
      setError(err.message || "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => { fetchMetrics(); }, [runId, fetchMetrics]);

  useEffect(() => {
    if (pollingInterval > 0) {
      const interval = setInterval(() => fetchMetrics(), pollingInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [pollingInterval, fetchMetrics]);

  // Real-time update when training is running
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => fetchMetrics(), 30000);
    return () => clearInterval(interval);
  }, [isRunning, fetchMetrics]);

  // Responsive width
  useEffect(() => {
    if (!svgRef.current) return;
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        setSvgWidth(entry.contentRect.width);
      }
    });
    observer.observe(svgRef.current);
    return () => observer.disconnect();
  }, []);

  const hasAnyData = (
    (showUNet && activeData.unet.length > 0) ||
    (showTE1 && activeData.te1.length > 0) ||
    (showTE2 && activeData.te2.length > 0) ||
    (showVE && activeData.ve.length > 0)
  );

  const totalDataPoints = (
    activeData.unet.length + activeData.te1.length +
    activeData.te2.length + activeData.ve.length
  );

  // SVG chart constants
  const margin = { top: 20, right: 20, bottom: 40, left: 60 };
  const height = 200;
  const chartWidth = Math.max(100, svgWidth - margin.left - margin.right);
  const chartHeight = height - margin.top - margin.bottom;

  // Compute axis ranges
  const allActiveValues: number[] = [];
  if (showUNet) smoothData.unet.forEach(p => allActiveValues.push(p.value));
  if (showTE1) smoothData.te1.forEach(p => allActiveValues.push(p.value));
  if (showTE2) smoothData.te2.forEach(p => allActiveValues.push(p.value));
  if (showVE) smoothData.ve.forEach(p => allActiveValues.push(p.value));

  const allSteps: number[] = [];
  if (showUNet) activeData.unet.forEach(p => allSteps.push(p.step));
  if (showTE1) activeData.te1.forEach(p => allSteps.push(p.step));
  if (showTE2) activeData.te2.forEach(p => allSteps.push(p.step));
  if (showVE) activeData.ve.forEach(p => allSteps.push(p.step));

  const yRange = calculateRobustYRange(allActiveValues);
  const xMin = allSteps.length > 0 ? Math.min(...allSteps) : 0;
  const xMax = allSteps.length > 0 ? Math.max(...allSteps) : 1;

  const toX = (step: number) =>
    ((step - xMin) / Math.max(xMax - xMin, 1)) * chartWidth;
  const toY = (val: number) =>
    chartHeight - ((val - yRange.min) / Math.max(yRange.max - yRange.min, 1e-10)) * chartHeight;

  const buildPath = (data: MetricPoint[]): string => {
    if (data.length === 0) return "";
    return data
      .map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.step).toFixed(1)},${toY(p.value).toFixed(1)}`)
      .join(" ");
  };

  const formatValue = (v: number | undefined) => {
    if (v === undefined) return "—";
    if (v >= 1000) return v.toFixed(0);
    if (v >= 1) return v.toFixed(2);
    if (v >= 0.001) return v.toFixed(4);
    return v.toExponential(2);
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!hasAnyData) return;
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mouseX = e.clientX - rect.left - margin.left;
    const step = Math.round((mouseX / chartWidth) * (xMax - xMin) + xMin);

    const nearest = (data: MetricPoint[]) => {
      if (data.length === 0) return undefined;
      const idx = data.reduce((best, p, i) =>
        Math.abs(p.step - step) < Math.abs(data[best].step - step) ? i : best, 0);
      return data[idx].value;
    };

    const nearestSmooth = (data: MetricPoint[]) => {
      if (data.length === 0) return undefined;
      const idx = data.reduce((best, p, i) =>
        Math.abs(p.step - step) < Math.abs(data[best].step - step) ? i : best, 0);
      return data[idx].value;
    };

    setTooltip({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      step,
      values: {
        unet: showUNet ? nearest(activeData.unet) : undefined,
        te1: showTE1 ? nearest(activeData.te1) : undefined,
        te2: showTE2 ? nearest(activeData.te2) : undefined,
        ve: showVE ? nearest(activeData.ve) : undefined,
      },
      smoothValues: {
        unet: showUNet ? nearestSmooth(smoothData.unet) : undefined,
        te1: showTE1 ? nearestSmooth(smoothData.te1) : undefined,
        te2: showTE2 ? nearestSmooth(smoothData.te2) : undefined,
        ve: showVE ? nearestSmooth(smoothData.ve) : undefined,
      },
    });
  };

  // Y-axis ticks
  const yTicks = useMemo(() => {
    const n = 4;
    return Array.from({ length: n + 1 }, (_, i) => {
      const v = yRange.min + (i / n) * (yRange.max - yRange.min);
      return { v, y: toY(v) };
    });
  }, [yRange, chartHeight]); // eslint-disable-line react-hooks/exhaustive-deps

  // X-axis ticks
  const xTicks = useMemo(() => {
    const n = 5;
    return Array.from({ length: n + 1 }, (_, i) => {
      const step = Math.round(xMin + (i / n) * (xMax - xMin));
      return { step, x: toX(step) };
    });
  }, [xMin, xMax, chartWidth]); // eslint-disable-line react-hooks/exhaustive-deps

  const tabLabel = tab === "update_norm"
    ? "Update Norm (||θ_t - θ_{t-K}||_F)"
    : "Cumulative Drift (||θ_t - θ_0||_F / ||θ_0||_F)";

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300">Parameter Change</h3>
        <div className="flex items-center gap-2">
          <select
            value={pollingInterval}
            onChange={(e) => setPollingInterval(Number(e.target.value))}
            className="text-xs bg-gray-700 border border-gray-600 rounded px-1.5 py-0.5"
          >
            <option value={0}>Manual</option>
            <option value={30}>30s</option>
            <option value={60}>1m</option>
            <option value={120}>2m</option>
          </select>
          <button
            onClick={fetchMetrics}
            disabled={loading}
            className="text-gray-400 hover:text-gray-200 disabled:opacity-50"
            title="Refresh"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-900 rounded p-0.5">
        {(["update_norm", "cumulative_drift"] as TabType[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 text-xs py-1 rounded transition-colors ${
              tab === t ? "bg-gray-600 text-white" : "text-gray-400 hover:text-gray-200"
            }`}
          >
            {t === "update_norm" ? "Update Norm" : "Cumulative Drift"}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-400">Smooth:</span>
          <input
            type="range" min="0" max="0.99" step="0.01"
            value={smoothingFactor}
            onChange={(e) => setSmoothingFactor(parseFloat(e.target.value))}
            className="w-16 h-1 accent-blue-500"
          />
          <span className="text-xs text-gray-400 w-8">{smoothingFactor.toFixed(2)}</span>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {SERIES.map(s => {
            const show = s.key === "unet" ? showUNet : s.key === "te1" ? showTE1 : s.key === "te2" ? showTE2 : showVE;
            const setShow = s.key === "unet" ? setShowUNet : s.key === "te1" ? setShowTE1 : s.key === "te2" ? setShowTE2 : setShowVE;
            const count = activeData[s.key].length;
            return (
              <button
                key={s.key}
                onClick={() => setShow(!show)}
                className={`flex items-center gap-1 text-xs px-1.5 py-0.5 rounded transition-opacity ${show ? "" : "opacity-40"}`}
              >
                <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: s.color }} />
                <span style={{ color: s.color }}>{s.label}</span>
                {count > 0 && <span className="text-gray-500">({count})</span>}
              </button>
            );
          })}
        </div>
      </div>

      {/* Chart */}
      <div className="relative">
        {error ? (
          <div className="text-xs text-red-400 py-4 text-center">{error}</div>
        ) : !hasAnyData ? (
          <div className="text-xs text-gray-500 py-8 text-center">
            {totalDataPoints === 0
              ? "No parameter tracking data. Enable param_tracking in training config."
              : "No data for selected series."}
          </div>
        ) : (
          <svg
            ref={svgRef}
            width="100%"
            height={height}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setTooltip(null)}
            className="overflow-visible"
          >
            <g transform={`translate(${margin.left},${margin.top})`}>
              {/* Grid lines */}
              {yTicks.map(({ v, y }) => (
                <line key={v} x1={0} x2={chartWidth} y1={y} y2={y} stroke="#374151" strokeWidth={0.5} />
              ))}

              {/* Y-axis ticks */}
              {yTicks.map(({ v, y }) => (
                <text key={v} x={-4} y={y} textAnchor="end" dominantBaseline="middle" fill="#9ca3af" fontSize={9}>
                  {formatValue(v)}
                </text>
              ))}

              {/* X-axis ticks */}
              {xTicks.map(({ step, x }) => (
                <text key={step} x={x} y={chartHeight + 14} textAnchor="middle" fill="#9ca3af" fontSize={9}>
                  {step}
                </text>
              ))}

              {/* X-axis label */}
              <text x={chartWidth / 2} y={chartHeight + 30} textAnchor="middle" fill="#6b7280" fontSize={9}>
                Step
              </text>

              {/* Data lines */}
              {showUNet && smoothData.unet.length > 0 && (
                <path d={buildPath(smoothData.unet)} fill="none" stroke="#60a5fa" strokeWidth={1.5} />
              )}
              {showTE1 && smoothData.te1.length > 0 && (
                <path d={buildPath(smoothData.te1)} fill="none" stroke="#34d399" strokeWidth={1.5} />
              )}
              {showTE2 && smoothData.te2.length > 0 && (
                <path d={buildPath(smoothData.te2)} fill="none" stroke="#f59e0b" strokeWidth={1.5} />
              )}
              {showVE && smoothData.ve.length > 0 && (
                <path d={buildPath(smoothData.ve)} fill="none" stroke="#f87171" strokeWidth={1.5} />
              )}

              {/* Border */}
              <rect x={0} y={0} width={chartWidth} height={chartHeight} fill="none" stroke="#374151" strokeWidth={0.5} />
            </g>
          </svg>
        )}

        {/* Tooltip */}
        {tooltip && hasAnyData && (
          <div
            className="absolute pointer-events-none bg-gray-900 border border-gray-600 rounded px-2 py-1.5 text-xs space-y-0.5 z-10"
            style={{
              left: tooltip.x + 12,
              top: tooltip.y - 10,
              transform: tooltip.x > svgWidth * 0.6 ? "translateX(-110%)" : undefined,
            }}
          >
            <div className="text-gray-400 font-medium">Step {tooltip.step}</div>
            {SERIES.map(s => {
              const show = s.key === "unet" ? showUNet : s.key === "te1" ? showTE1 : s.key === "te2" ? showTE2 : showVE;
              if (!show) return null;
              const sv = tooltip.smoothValues[s.key];
              if (sv === undefined) return null;
              return (
                <div key={s.key} className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: s.color }} />
                  <span style={{ color: s.color }}>{s.label}:</span>
                  <span className="text-white">{formatValue(sv)}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Tab description */}
      <p className="text-xs text-gray-500">{tabLabel}</p>
    </div>
  );
}
