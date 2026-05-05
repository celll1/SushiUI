"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import type { TaggerTrainingMetric } from "@/utils/api";

interface TaggerMetricChartProps {
  data: TaggerTrainingMetric[];
  valueKey: "loss" | "f1" | "threshold";
  color: string;
  title: string;
  height?: number;
  smoothable?: boolean;
  defaultSmoothing?: number;
  yMinFloor?: number;
}

interface Point {
  step: number;
  value: number;
}

// EMA smoothing
function applySmoothing(points: Point[], factor: number): Point[] {
  if (factor <= 0 || points.length === 0) return points;
  const out: Point[] = [];
  let s = points[0].value;
  for (const p of points) {
    s = s * factor + p.value * (1 - factor);
    out.push({ step: p.step, value: s });
  }
  return out;
}

// Robust Y-range: 5–95th percentiles + 5% padding
function robustYRange(values: number[], yMinFloor: number): { min: number; max: number } {
  const valid = values.filter((v) => Number.isFinite(v));
  if (valid.length === 0) return { min: yMinFloor, max: yMinFloor + 1 };
  if (valid.length === 1) {
    const v = valid[0];
    const pad = Math.max(Math.abs(v) * 0.1, 1e-6);
    return { min: Math.max(yMinFloor, v - pad), max: v + pad };
  }
  const sorted = [...valid].sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.05)];
  const hi = sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)];
  const range = hi - lo || Math.max(Math.abs(hi) * 0.1, 1e-6);
  const pad = range * 0.05;
  return { min: Math.max(yMinFloor, lo - pad), max: hi + pad };
}

export default function TaggerMetricChart({
  data,
  valueKey,
  color,
  title,
  height = 160,
  smoothable = false,
  defaultSmoothing = 0.9,
  yMinFloor = 0,
}: TaggerMetricChartProps) {
  // Extract valid (step, value) points
  const points = useMemo<Point[]>(() => {
    return data
      .map((d) => ({ step: d.step, value: d[valueKey] as number | null }))
      .filter((p): p is Point => p.value !== null && Number.isFinite(p.value));
  }, [data, valueKey]);

  const minStepAll = points.length > 0 ? points[0].step : 0;
  const maxStepAll = points.length > 0 ? points[points.length - 1].step : 0;

  // x range (null = full)
  const [xRange, setXRange] = useState<{ min: number; max: number } | null>(null);
  // smoothing
  const [smoothing, setSmoothing] = useState(smoothable ? defaultSmoothing : 0);
  // brush (drag) state in CHART pixel coords
  const [brush, setBrush] = useState<{ startX: number; curX: number } | null>(null);
  // hover tooltip (in chart coords)
  const [tooltip, setTooltip] = useState<{
    px: number;
    py: number;
    step: number;
    value: number;
    smoothValue: number | null;
  } | null>(null);

  // Container width (responsive).
  // Use a callback ref so the ResizeObserver re-attaches when the element
  // appears (e.g. after the empty-state early return is replaced by the
  // full chart on data arrival).
  const [width, setWidth] = useState(0);
  const roRef = useRef<ResizeObserver | null>(null);
  const containerRef = useCallback((el: HTMLDivElement | null) => {
    if (roRef.current) {
      roRef.current.disconnect();
      roRef.current = null;
    }
    if (el) {
      const ro = new ResizeObserver((entries) => {
        const w = entries[0]?.contentRect.width ?? 0;
        if (w > 0) setWidth(w);
      });
      ro.observe(el);
      roRef.current = ro;
      // Capture immediate width synchronously so first paint is correct
      const rect = el.getBoundingClientRect();
      if (rect.width > 0) setWidth(rect.width);
    }
  }, []);
  useEffect(() => () => { roRef.current?.disconnect(); }, []);

  // ESC key resets zoom
  useEffect(() => {
    if (!xRange) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setXRange(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [xRange]);

  // Visible points after x-range filter
  const visiblePoints = useMemo(() => {
    if (!xRange) return points;
    return points.filter((p) => p.step >= xRange.min && p.step <= xRange.max);
  }, [points, xRange]);

  // Smoothed series (full and visible)
  const smoothedAll = useMemo(() => applySmoothing(points, smoothing), [points, smoothing]);
  const smoothedVisible = useMemo(() => {
    if (!xRange) return smoothedAll;
    return smoothedAll.filter((p) => p.step >= xRange.min && p.step <= xRange.max);
  }, [smoothedAll, xRange]);

  // Layout
  const padding = { top: 6, right: 8, bottom: 18, left: 44 };
  const chartW = Math.max(50, width - padding.left - padding.right);
  const chartH = Math.max(20, height - padding.top - padding.bottom);
  const hasEnoughData = points.length >= 2 && width > 0;

  // X scale
  const xMin = xRange ? xRange.min : minStepAll;
  const xMax = xRange ? xRange.max : maxStepAll;
  const xSpan = xMax - xMin || 1;
  const toX = (step: number) => padding.left + ((step - xMin) / xSpan) * chartW;

  // Y scale: from visible points (use smoothed values for range when smoothing > 0)
  const sourceForRange =
    smoothing > 0 && smoothedVisible.length > 0
      ? smoothedVisible.map((p) => p.value)
      : visiblePoints.map((p) => p.value);
  const { min: yMin, max: yMax } = robustYRange(sourceForRange, yMinFloor);
  const ySpan = yMax - yMin || 1;
  const toY = (v: number) => padding.top + ((yMax - v) / ySpan) * chartH;

  // Path builders
  const buildPath = (pts: Point[]) =>
    pts
      .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.step).toFixed(1)} ${toY(p.value).toFixed(1)}`)
      .join(" ");

  const rawPath = buildPath(visiblePoints);
  const smoothPath = smoothing > 0 ? buildPath(smoothedVisible) : "";

  // Y-axis tick formatting
  const formatY = (v: number) => {
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (Math.abs(v) >= 1) return v.toFixed(2);
    if (Math.abs(v) >= 0.01) return v.toFixed(3);
    if (v === 0) return "0";
    return v.toExponential(1);
  };
  const yTickValues = [yMax, yMin + ySpan * 0.5, yMin];

  // X-axis tick formatting
  const formatX = (step: number) => {
    if (step >= 1_000_000) return `${(step / 1_000_000).toFixed(1)}M`;
    if (step >= 1000) return `${(step / 1000).toFixed(1)}k`;
    return String(step);
  };
  const xTickValues = [xMin, xMin + xSpan * 0.5, xMax].map((s) => Math.round(s));

  // ── Pointer interactions ────────────────────────────────────────────

  const pxToStep = (pxInChart: number) => xMin + (pxInChart / chartW) * xSpan;

  const onMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - padding.left;
    if (x < 0 || x > chartW) return;
    setBrush({ startX: x, curX: x });
    setTooltip(null);
  };

  const onMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - padding.left;
    const y = e.clientY - rect.top - padding.top;

    if (brush) {
      const clamped = Math.max(0, Math.min(chartW, x));
      setBrush({ ...brush, curX: clamped });
      return;
    }
    if (x < 0 || x > chartW || y < -padding.top || y > chartH + padding.bottom) {
      setTooltip(null);
      return;
    }
    // Nearest-point search (linear; visible point counts are small after API decimation)
    const targetStep = pxToStep(x);
    let best = visiblePoints[0];
    let bestSmooth = smoothedVisible[0] ?? null;
    let bestDist = Math.abs(best.step - targetStep);
    for (let i = 1; i < visiblePoints.length; i++) {
      const d = Math.abs(visiblePoints[i].step - targetStep);
      if (d < bestDist) {
        bestDist = d;
        best = visiblePoints[i];
        bestSmooth = smoothedVisible[i] ?? null;
      }
    }
    setTooltip({
      px: toX(best.step),
      py: toY((bestSmooth ?? best).value),
      step: best.step,
      value: best.value,
      smoothValue: bestSmooth ? bestSmooth.value : null,
    });
  };

  const onMouseUp = () => {
    if (!brush) return;
    const a = Math.min(brush.startX, brush.curX);
    const b = Math.max(brush.startX, brush.curX);
    const dragPx = b - a;
    setBrush(null);
    if (dragPx < 4) return; // treated as click — no zoom
    const newMin = Math.round(pxToStep(a));
    const newMax = Math.round(pxToStep(b));
    if (newMax > newMin) setXRange({ min: newMin, max: newMax });
  };

  const onMouseLeave = () => {
    setTooltip(null);
    if (brush) setBrush(null);
  };

  const onDoubleClick = () => setXRange(null);

  // Brush rect coords
  const brushRect = brush
    ? (() => {
        const a = Math.min(brush.startX, brush.curX);
        const b = Math.max(brush.startX, brush.curX);
        return { x: padding.left + a, w: b - a };
      })()
    : null;

  // ── Render ──────────────────────────────────────────────────────────

  return (
    <div className="bg-gray-800 rounded p-2 border border-gray-700">
      <div className="flex items-center justify-between mb-1">
        <div className="text-sm font-medium text-gray-300">{title}</div>
        <div className="flex items-center gap-2">
          {smoothable && (
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-gray-500">Smooth</span>
              <input
                type="range"
                min={0}
                max={0.99}
                step={0.01}
                value={smoothing}
                onChange={(e) => setSmoothing(parseFloat(e.target.value))}
                className="w-20 h-1 cursor-pointer"
                title="EMA smoothing"
              />
              <span className="text-[10px] text-gray-400 font-mono w-7 text-right">
                {(smoothing * 100).toFixed(0)}%
              </span>
            </div>
          )}
          {xRange && (
            <button
              onClick={() => setXRange(null)}
              className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
              title="Reset zoom (or double-click chart, or ESC)"
            >
              ↺ Reset
            </button>
          )}
        </div>
      </div>

      <div ref={containerRef} className="w-full select-none" style={{ position: "relative", minHeight: height }}>
        {!hasEnoughData && (
          <div
            className="flex items-center justify-center text-gray-500 text-xs"
            style={{ height }}
          >
            {points.length < 2 ? "Not enough data" : ""}
          </div>
        )}
        {hasEnoughData && (
        <svg
          width={width}
          height={height}
          className="block cursor-crosshair"
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseLeave}
          onDoubleClick={onDoubleClick}
        >
          {/* Y grid lines + labels */}
          {yTickValues.map((v, i) => (
            <g key={`y-${i}`}>
              <line
                x1={padding.left}
                x2={padding.left + chartW}
                y1={toY(v)}
                y2={toY(v)}
                stroke="#374151"
                strokeWidth={0.5}
                strokeDasharray={i === 1 ? "2 2" : undefined}
              />
              <text
                x={padding.left - 4}
                y={toY(v)}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize={9}
                fill="#9ca3af"
              >
                {formatY(v)}
              </text>
            </g>
          ))}

          {/* X tick labels */}
          {xTickValues.map((s, i) => (
            <text
              key={`x-${i}`}
              x={toX(s)}
              y={height - 4}
              textAnchor={i === 0 ? "start" : i === xTickValues.length - 1 ? "end" : "middle"}
              fontSize={9}
              fill="#9ca3af"
            >
              {formatX(s)}
            </text>
          ))}

          {/* Raw line (faded if smoothing on) */}
          {rawPath && (
            <path
              d={rawPath}
              fill="none"
              stroke={color}
              strokeWidth={1.2}
              opacity={smoothing > 0 ? 0.3 : 1}
            />
          )}
          {/* Smoothed line */}
          {smoothPath && (
            <path d={smoothPath} fill="none" stroke={color} strokeWidth={1.6} />
          )}

          {/* Brush preview */}
          {brushRect && brushRect.w > 0 && (
            <rect
              x={brushRect.x}
              y={padding.top}
              width={brushRect.w}
              height={chartH}
              fill={color}
              fillOpacity={0.15}
              stroke={color}
              strokeWidth={0.8}
              strokeDasharray="3 2"
            />
          )}

          {/* Tooltip marker */}
          {tooltip && !brush && (
            <>
              <line
                x1={tooltip.px}
                x2={tooltip.px}
                y1={padding.top}
                y2={padding.top + chartH}
                stroke="#9ca3af"
                strokeWidth={0.5}
                strokeDasharray="2 2"
              />
              <circle cx={tooltip.px} cy={tooltip.py} r={3} fill={color} />
            </>
          )}
        </svg>
        )}

        {/* Tooltip box (HTML overlay) */}
        {hasEnoughData && tooltip && !brush && (
          <div
            className="pointer-events-none absolute bg-gray-900 border border-gray-600 rounded px-2 py-1 text-[10px] font-mono text-gray-200 shadow-lg whitespace-nowrap"
            style={{
              left: Math.min(width - 140, tooltip.px + 8),
              top: Math.max(0, tooltip.py - 8),
              zIndex: 10,
            }}
          >
            <div className="text-gray-400">step {tooltip.step.toLocaleString()}</div>
            <div>
              <span className="text-gray-500">{valueKey}: </span>
              {formatY(tooltip.value)}
            </div>
            {tooltip.smoothValue !== null && smoothing > 0 && (
              <div>
                <span className="text-gray-500">smooth: </span>
                {formatY(tooltip.smoothValue)}
              </div>
            )}
          </div>
        )}
      </div>

      {xRange && (
        <div className="text-[10px] text-gray-500 mt-1 font-mono">
          x: [{xRange.min.toLocaleString()}, {xRange.max.toLocaleString()}]
          {visiblePoints.length > 0 && ` · ${visiblePoints.length} pts`}
        </div>
      )}
    </div>
  );
}
