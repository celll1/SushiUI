"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import type { TaggerTrainingMetric } from "@/utils/api";

interface TaggerMetricChartProps {
  data: TaggerTrainingMetric[];
  valueKey: "loss" | "f1" | "threshold";
  /** Color used when only one resume_seq is present (initial/legacy single-curve render).
   *  Multi-series rendering cycles through the built-in palette below. */
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

// Dark-background-friendly palette for per-resume curves.  Cycles when
// resume_seq exceeds the palette length.
const RESUME_PALETTE = [
  "#60a5fa", // blue-400  — initial run
  "#f97316", // orange-500
  "#34d399", // emerald-400
  "#f472b6", // pink-400
  "#a78bfa", // violet-400
  "#facc15", // yellow-400
  "#22d3ee", // cyan-400
];
const colorForResume = (seq: number, fallback: string): string => {
  if (seq === 0) return fallback;          // honour caller's color for the initial curve
  return RESUME_PALETTE[seq % RESUME_PALETTE.length];
};
const labelForResume = (seq: number): string => (seq === 0 ? "Initial" : `Resume #${seq}`);

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

// Chart padding (used everywhere; constant)
const PAD = { top: 6, right: 8, bottom: 18, left: 44 };

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
  // Group points by resume_seq so each resume becomes its own curve.
  const groups = useMemo<Map<number, Point[]>>(() => {
    const m = new Map<number, Point[]>();
    for (const d of data) {
      const v = d[valueKey] as number | null;
      if (v === null || !Number.isFinite(v)) continue;
      const seq = d.resume_seq ?? 0;
      if (!m.has(seq)) m.set(seq, []);
      m.get(seq)!.push({ step: d.step, value: v });
    }
    // Sort each group by step (data may arrive out of order across resumes)
    for (const arr of m.values()) arr.sort((a, b) => a.step - b.step);
    return m;
  }, [data, valueKey]);

  // Resume seqs in render order (ascending so newer overlays older)
  const groupKeys = useMemo(() => [...groups.keys()].sort((a, b) => a - b), [groups]);

  // Latest resume (highest seq) — used as the source for tooltip nearest-point search
  const latestSeq = groupKeys.length > 0 ? groupKeys[groupKeys.length - 1] : 0;

  // Pooled range across all groups for x-axis bounds
  const allPoints = useMemo(
    () => groupKeys.flatMap((seq) => groups.get(seq) ?? []),
    [groups, groupKeys]
  );
  const minStepAll = allPoints.length > 0
    ? Math.min(...allPoints.map((p) => p.step))
    : 0;
  const maxStepAll = allPoints.length > 0
    ? Math.max(...allPoints.map((p) => p.step))
    : 0;
  const totalPoints = allPoints.length;

  // x range (null = full)
  const [xRange, setXRange] = useState<{ min: number; max: number } | null>(null);
  // smoothing
  const [smoothing, setSmoothing] = useState(smoothable ? defaultSmoothing : 0);
  // brush (drag) state in STEP coords so it survives xRange auto-extension
  const [brush, setBrush] = useState<{ startStep: number; endStep: number } | null>(null);
  // hover tooltip (in chart coords)
  const [tooltip, setTooltip] = useState<{
    px: number;
    py: number;
    step: number;
    value: number;
    smoothValue: number | null;
  } | null>(null);

  // Pointer x (in chart pixel coords) — read by the auto-extend rAF loop
  const pointerXRef = useRef<number | null>(null);

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

  // Refs for values read inside the auto-extend rAF loop (avoid restarting it)
  const chartWRef = useRef(0);
  const xRangeRef = useRef(xRange);
  const minStepAllRef = useRef(minStepAll);
  const maxStepAllRef = useRef(maxStepAll);
  useEffect(() => { xRangeRef.current = xRange; }, [xRange]);
  useEffect(() => { minStepAllRef.current = minStepAll; }, [minStepAll]);
  useEffect(() => { maxStepAllRef.current = maxStepAll; }, [maxStepAll]);
  // chartW is a derived value (depends on width); sync after render
  useEffect(() => {
    chartWRef.current = Math.max(50, width - PAD.left - PAD.right);
  }, [width]);

  // Auto-extend xRange when the pointer is past the chart edge during a brush.
  // Only triggers when already zoomed (xRange !== null) — otherwise nothing to extend.
  const isBrushing = brush !== null;
  useEffect(() => {
    if (!isBrushing) return;
    let raf = 0;
    let last = performance.now();

    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;

      const px = pointerXRef.current;
      const cw = chartWRef.current;
      const xr = xRangeRef.current;

      if (xr && px !== null && cw > 0 && (px < 0 || px > cw)) {
        const overflow = px < 0 ? px : px - cw;       // signed
        // Speed: ~75% of current span per second when 200px past the edge
        const fractionPerSec = Math.min(2, Math.abs(overflow) / 200);
        const span = xr.max - xr.min;
        const dStep = span * fractionPerSec * dt * Math.sign(overflow);

        let newMin = xr.min;
        let newMax = xr.max;
        if (overflow < 0) {
          newMin = Math.max(minStepAllRef.current, xr.min + dStep); // dStep < 0
        } else {
          newMax = Math.min(maxStepAllRef.current, xr.max + dStep);
        }

        if (newMin !== xr.min || newMax !== xr.max) {
          setXRange({ min: newMin, max: newMax });
          setBrush((b) => (b ? { ...b, endStep: overflow < 0 ? newMin : newMax } : b));
        }
      }

      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isBrushing]);

  // Per-group visible points after x-range filter
  const visibleGroups = useMemo<Map<number, Point[]>>(() => {
    const out = new Map<number, Point[]>();
    for (const [seq, pts] of groups) {
      const filtered = xRange
        ? pts.filter((p) => p.step >= xRange.min && p.step <= xRange.max)
        : pts;
      out.set(seq, filtered);
    }
    return out;
  }, [groups, xRange]);

  // Per-group smoothed series (smoothing must be applied within each
  // group; mixing across resumes would create jumps at the boundary).
  const smoothedAllGroups = useMemo<Map<number, Point[]>>(() => {
    const out = new Map<number, Point[]>();
    for (const [seq, pts] of groups) out.set(seq, applySmoothing(pts, smoothing));
    return out;
  }, [groups, smoothing]);

  const smoothedVisibleGroups = useMemo<Map<number, Point[]>>(() => {
    if (!xRange) return smoothedAllGroups;
    const out = new Map<number, Point[]>();
    for (const [seq, pts] of smoothedAllGroups) {
      out.set(seq, pts.filter((p) => p.step >= xRange.min && p.step <= xRange.max));
    }
    return out;
  }, [smoothedAllGroups, xRange]);

  // For Y-range pooling and tooltip search, flatten the visible points
  const visiblePoints = useMemo(
    () => [...visibleGroups.values()].flat(),
    [visibleGroups]
  );
  const smoothedVisibleAllPoints = useMemo(
    () => [...smoothedVisibleGroups.values()].flat(),
    [smoothedVisibleGroups]
  );

  // Layout
  const chartW = Math.max(50, width - PAD.left - PAD.right);
  const chartH = Math.max(20, height - PAD.top - PAD.bottom);
  const hasEnoughData = totalPoints >= 2 && width > 0;

  // X scale
  const xMin = xRange ? xRange.min : minStepAll;
  const xMax = xRange ? xRange.max : maxStepAll;
  const xSpan = xMax - xMin || 1;
  const toX = (step: number) => PAD.left + ((step - xMin) / xSpan) * chartW;

  // Y scale: from visible points (use smoothed values for range when smoothing > 0).
  // Pool across all groups so all curves share the same scale.
  const sourceForRange =
    smoothing > 0 && smoothedVisibleAllPoints.length > 0
      ? smoothedVisibleAllPoints.map((p) => p.value)
      : visiblePoints.map((p) => p.value);
  const { min: yMin, max: yMax } = robustYRange(sourceForRange, yMinFloor);
  const ySpan = yMax - yMin || 1;
  const toY = (v: number) => PAD.top + ((yMax - v) / ySpan) * chartH;

  // Path builder
  const buildPath = (pts: Point[]) =>
    pts
      .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.step).toFixed(1)} ${toY(p.value).toFixed(1)}`)
      .join(" ");

  // Y-axis tick formatting
  const formatY = (v: number) => {
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (Math.abs(v) >= 1) return v.toFixed(2);
    if (Math.abs(v) >= 0.01) return v.toFixed(3);
    if (v === 0) return "0";
    return v.toExponential(1);
  };

  // Tooltip value formatting (2 decimal places; scientific with 2-place
  // mantissa for tiny values so loss like 1.54e-4 stays informative)
  const formatTooltip = (v: number) => {
    if (v === 0) return "0.00";
    if (Math.abs(v) >= 0.01) return v.toFixed(2);
    return v.toExponential(2);
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

  // Pointer Events with setPointerCapture so the brush survives mouseup
  // outside the SVG.  When the cursor leaves the chart while brushing, an
  // auto-extend rAF loop (see useEffect below) gradually expands xRange in
  // that direction.
  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - PAD.left;
    if (x < 0 || x > chartW) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    pointerXRef.current = x;
    const step = pxToStep(x);
    setBrush({ startStep: step, endStep: step });
    setTooltip(null);
  };

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - PAD.left;
    const y = e.clientY - rect.top - PAD.top;

    if (brush) {
      pointerXRef.current = x;
      // While inside chart, set endStep from cursor.  When outside, the
      // rAF loop drives xRange and endStep.
      if (x >= 0 && x <= chartW) {
        setBrush({ ...brush, endStep: pxToStep(x) });
      }
      return;
    }
    if (x < 0 || x > chartW || y < -PAD.top || y > chartH + PAD.bottom) {
      setTooltip(null);
      return;
    }
    // Nearest-point search confined to the latest resume's curve.
    // Overlapping resumes share the X axis; the user almost always wants
    // to read the current run's value, so we don't bounce between curves.
    const targetStep = pxToStep(x);
    const latestPts   = visibleGroups.get(latestSeq) ?? [];
    const latestSmPts = smoothedVisibleGroups.get(latestSeq) ?? [];
    if (latestPts.length === 0) {
      setTooltip(null);
      return;
    }
    let bestIdx = 0;
    let bestDist = Math.abs(latestPts[0].step - targetStep);
    for (let i = 1; i < latestPts.length; i++) {
      const d = Math.abs(latestPts[i].step - targetStep);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    const best = latestPts[bestIdx];
    const bestSmooth = latestSmPts[bestIdx] ?? null;
    setTooltip({
      px: toX(best.step),
      py: toY((bestSmooth ?? best).value),
      step: best.step,
      value: best.value,
      smoothValue: bestSmooth ? bestSmooth.value : null,
    });
  };

  const onPointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!brush) return;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    pointerXRef.current = null;
    const a = Math.min(brush.startStep, brush.endStep);
    const b = Math.max(brush.startStep, brush.endStep);
    setBrush(null);
    // Treat as click if span is < 4 px equivalent in the current view
    const minSpan = (4 / Math.max(1, chartW)) * xSpan;
    if (b - a < minSpan) return;
    const newMin = Math.round(a);
    const newMax = Math.round(b);
    if (newMax > newMin) setXRange({ min: newMin, max: newMax });
  };

  const onPointerLeave = () => {
    // Hide tooltip when the pointer leaves the SVG, but DO NOT cancel the
    // brush — pointer capture keeps move/up events flowing here.
    if (!brush) setTooltip(null);
  };

  const onDoubleClick = () => setXRange(null);

  // Brush rect coords (translate step coords → pixels via current xRange)
  const brushRect = brush
    ? (() => {
        const aStep = Math.min(brush.startStep, brush.endStep);
        const bStep = Math.max(brush.startStep, brush.endStep);
        // Clamp to current visible range so the rect doesn't overflow the chart
        const aClamped = Math.max(xMin, Math.min(xMax, aStep));
        const bClamped = Math.max(xMin, Math.min(xMax, bStep));
        const x = toX(aClamped);
        const w = toX(bClamped) - x;
        return { x, w };
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
            {totalPoints < 2 ? "Not enough data" : ""}
          </div>
        )}
        {/* Per-resume legend (only when multiple curves are present) */}
        {hasEnoughData && groups.size > 1 && (
          <div className="absolute top-1 right-1 flex flex-wrap gap-1.5 text-[10px] bg-gray-900/80 px-1.5 py-0.5 rounded border border-gray-700 z-20 pointer-events-none">
            {groupKeys.map((seq) => (
              <div key={`lg-${seq}`} className="flex items-center gap-1">
                <span
                  className="inline-block w-2 h-2 rounded-sm"
                  style={{ background: colorForResume(seq, color) }}
                />
                <span className="text-gray-300">{labelForResume(seq)}</span>
              </div>
            ))}
          </div>
        )}
        {hasEnoughData && (
        <svg
          width={width}
          height={height}
          className="block cursor-crosshair"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onPointerLeave={onPointerLeave}
          onDoubleClick={onDoubleClick}
        >
          {/* Y grid lines + labels */}
          {yTickValues.map((v, i) => (
            <g key={`y-${i}`}>
              <line
                x1={PAD.left}
                x2={PAD.left + chartW}
                y1={toY(v)}
                y2={toY(v)}
                stroke="#374151"
                strokeWidth={0.5}
                strokeDasharray={i === 1 ? "2 2" : undefined}
              />
              <text
                x={PAD.left - 4}
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

          {/* Per-resume raw + smoothed lines (older first, newer on top) */}
          {groupKeys.map((seq) => {
            const seqColor = colorForResume(seq, color);
            const visPts   = visibleGroups.get(seq) ?? [];
            const visSm    = smoothedVisibleGroups.get(seq) ?? [];
            const dRaw     = visPts.length > 0 ? buildPath(visPts) : "";
            const dSm      = smoothing > 0 && visSm.length > 0 ? buildPath(visSm) : "";
            return (
              <g key={`series-${seq}`}>
                {dRaw && (
                  <path
                    d={dRaw}
                    fill="none"
                    stroke={seqColor}
                    strokeWidth={1.2}
                    opacity={smoothing > 0 ? 0.3 : 1}
                  />
                )}
                {dSm && (
                  <path d={dSm} fill="none" stroke={seqColor} strokeWidth={1.6} />
                )}
              </g>
            );
          })}

          {/* Brush preview (uses latest-resume color for consistency) */}
          {brushRect && brushRect.w > 0 && (
            <rect
              x={brushRect.x}
              y={PAD.top}
              width={brushRect.w}
              height={chartH}
              fill={colorForResume(latestSeq, color)}
              fillOpacity={0.15}
              stroke={colorForResume(latestSeq, color)}
              strokeWidth={0.8}
              strokeDasharray="3 2"
            />
          )}

          {/* Tooltip marker — pinned to the latest resume's curve */}
          {tooltip && !brush && (
            <>
              <line
                x1={tooltip.px}
                x2={tooltip.px}
                y1={PAD.top}
                y2={PAD.top + chartH}
                stroke="#9ca3af"
                strokeWidth={0.5}
                strokeDasharray="2 2"
              />
              <circle cx={tooltip.px} cy={tooltip.py} r={3} fill={colorForResume(latestSeq, color)} />
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
            <div className="text-gray-400">
              step {tooltip.step.toLocaleString()}
              {groups.size > 1 && (
                <span className="ml-1" style={{ color: colorForResume(latestSeq, color) }}>
                  · {labelForResume(latestSeq)}
                </span>
              )}
            </div>
            <div>
              <span className="text-gray-500">{valueKey}: </span>
              {formatTooltip(tooltip.value)}
            </div>
            {tooltip.smoothValue !== null && smoothing > 0 && (
              <div>
                <span className="text-gray-500">smooth: </span>
                {formatTooltip(tooltip.smoothValue)}
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
