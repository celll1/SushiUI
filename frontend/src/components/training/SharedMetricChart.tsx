"use client";

import { useState, useEffect, useMemo, useRef } from "react";

/**
 * Generic multi-series metric chart shared by the image-gen and tagger training
 * monitors. Features: per-series EMA smoothing, epoch-boundary dotted lines,
 * resume markers, drag-to-zoom (+ ESC / double-click reset), a multi-series
 * crosshair tooltip, an optional log Y scale, and a collapsible legend.
 *
 * Resume handling is a rendering choice made by the CALLER:
 *  - "marker" mode: pass merged series + `resumeMarkers` (vertical markers).
 *  - "split" mode (future): pass one series per (metric, resume) — no markers.
 * The core only knows about generic series, so both modes need no core changes.
 */

export interface ChartSeriesPoint {
  step: number;
  value: number;
  resume_seq?: number;
}

export interface ChartSeries {
  id: string;
  label: string;
  color: string;
  points: ChartSeriesPoint[];
  /** Render as a dashed line (e.g. secondary/overlay series). */
  dashed?: boolean;
  /** Exclude from the pooled Y-range percentile calc (raw min/max still kept visible). */
  rawRange?: boolean;
}

export interface EpochBoundary {
  epoch: number;
  step: number;
}

export interface ResumeMarker {
  resume_seq: number;
  step: number;
}

interface SharedMetricChartProps {
  series: ChartSeries[];
  title: string;
  height?: number;
  yMinFloor?: number;
  /** Bounded metrics (F1 etc., in [0,1]) use full min/max; unbounded (loss) clip to 5–95%. */
  bounded?: boolean;
  smoothable?: boolean;
  defaultSmoothing?: number;
  /** Optional log Y scale toggle (off by default; useful for grad norm). */
  allowLogScale?: boolean;
  epochBoundaries?: EpochBoundary[];
  resumeMarkers?: ResumeMarker[];
  /** Header right-side extra controls (rendered before the smoothing slider). */
  headerExtra?: React.ReactNode;
}

interface Pt { step: number; value: number; }

const PAD = { top: 6, right: 8, bottom: 18, left: 44 };

function applySmoothing(points: Pt[], factor: number): Pt[] {
  if (factor <= 0 || points.length === 0) return points;
  const out: Pt[] = [];
  let s = points[0].value;
  for (const p of points) {
    s = s * factor + p.value * (1 - factor);
    out.push({ step: p.step, value: s });
  }
  return out;
}

function robustYRange(
  values: number[],
  yMinFloor: number,
  mustInclude: number[] = [],
  bounded = false,
): { min: number; max: number } {
  const valid = values.filter((v) => Number.isFinite(v));
  let lo: number, hi: number;
  if (valid.length === 0) {
    lo = yMinFloor; hi = yMinFloor + 1;
  } else if (valid.length === 1) {
    const v = valid[0];
    const pad = Math.max(Math.abs(v) * 0.1, 1e-6);
    lo = Math.max(yMinFloor, v - pad); hi = v + pad;
  } else if (bounded) {
    const sorted = [...valid].sort((a, b) => a - b);
    lo = sorted[0];
    hi = sorted[sorted.length - 1];
  } else {
    const sorted = [...valid].sort((a, b) => a - b);
    lo = sorted[Math.floor(sorted.length * 0.05)];
    hi = sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)];
  }
  for (const v of mustInclude) {
    if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  const range = hi - lo || Math.max(Math.abs(hi) * 0.1, 1e-6);
  const pad = range * 0.05;
  return { min: Math.max(yMinFloor, lo - pad), max: hi + pad };
}

/** Compute epoch boundaries from per-point resume-aware data when the caller
 *  does not provide them. Returns the last step seen for each epoch. */
export function computeEpochBoundariesFromPoints(
  points: { step: number; epoch?: number | null }[],
): EpochBoundary[] {
  const lastStep = new Map<number, number>();
  for (const p of points) {
    if (p.epoch === undefined || p.epoch === null) continue;
    const prev = lastStep.get(p.epoch);
    if (prev === undefined || p.step > prev) lastStep.set(p.epoch, p.step);
  }
  return [...lastStep.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([epoch, step]) => ({ epoch, step }));
}

export default function SharedMetricChart({
  series,
  title,
  height = 160,
  yMinFloor = -Infinity,
  bounded = false,
  smoothable = true,
  defaultSmoothing = 0.9,
  allowLogScale = false,
  epochBoundaries,
  resumeMarkers,
  headerExtra,
}: SharedMetricChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [smoothing, setSmoothing] = useState(smoothable ? defaultSmoothing : 0);
  const [logScale, setLogScale] = useState(false);
  const [legendOpen, setLegendOpen] = useState(false);
  // Series hidden via legend clicks. Hidden series are excluded from rendering
  // AND from the Y-range pooling, so the remaining series auto-rescale to fill
  // the view (lets you isolate one metric's variation instead of all forced on).
  const [hiddenIds, setHiddenIds] = useState<Set<string>>(new Set());
  const [xRange, setXRange] = useState<{ min: number; max: number } | null>(null);
  const [brush, setBrush] = useState<{ startStep: number; endStep: number } | null>(null);
  const pointerXRef = useRef<number | null>(null);
  const [tooltip, setTooltip] = useState<{
    px: number; py: number; step: number;
    values: { id: string; label: string; color: string; value: number; smoothValue: number | null }[];
  } | null>(null);

  // Width tracking
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setWidth(e.contentRect.width);
    });
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  // ESC resets zoom
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setXRange(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const seriesNonEmpty = useMemo(
    () => series.filter((s) => s.points && s.points.length > 0),
    [series],
  );
  // Series actually drawn/scaled: the non-empty set minus any hidden via legend.
  // The legend itself still lists every non-empty series (hidden ones dimmed) so
  // they can be toggled back on.
  const visibleSeries = useMemo(
    () => seriesNonEmpty.filter((s) => !hiddenIds.has(s.id)),
    [seriesNonEmpty, hiddenIds],
  );
  const allPoints = useMemo(() => visibleSeries.flatMap((s) => s.points), [visibleSeries]);
  const totalPoints = allPoints.length;

  const minStepAll = useMemo(
    () => (allPoints.length ? Math.min(...allPoints.map((p) => p.step)) : 0),
    [allPoints],
  );
  const maxStepAll = useMemo(
    () => (allPoints.length ? Math.max(...allPoints.map((p) => p.step)) : 1),
    [allPoints],
  );

  // Per-series smoothed points
  const smoothedSeries = useMemo<Map<string, Pt[]>>(() => {
    const out = new Map<string, Pt[]>();
    for (const s of visibleSeries) {
      out.set(s.id, applySmoothing(s.points.map((p) => ({ step: p.step, value: p.value })), smoothing));
    }
    return out;
  }, [visibleSeries, smoothing]);

  // Layout
  const chartW = Math.max(50, width - PAD.left - PAD.right);
  const chartH = Math.max(20, height - PAD.top - PAD.bottom);
  const hasEnoughData = totalPoints >= 2 && width > 0;

  // X scale
  const xMin = xRange ? xRange.min : minStepAll;
  const xMax = xRange ? xRange.max : maxStepAll;
  const xSpan = xMax - xMin || 1;
  const toX = (step: number) => PAD.left + ((step - xMin) / xSpan) * chartW;
  const pxToStep = (pxInChart: number) => xMin + (pxInChart / chartW) * xSpan;

  const inX = (step: number) => step >= xMin && step <= xMax;

  // Visible (clipped to xRange) points for Y range + tooltip
  const visibleSmoothed = useMemo<Map<string, Pt[]>>(() => {
    if (!xRange) return smoothedSeries;
    const out = new Map<string, Pt[]>();
    for (const [id, pts] of smoothedSeries) out.set(id, pts.filter((p) => inX(p.step)));
    return out;
  }, [smoothedSeries, xRange, xMin, xMax]);
  const visibleRaw = useMemo<Map<string, Pt[]>>(() => {
    const out = new Map<string, Pt[]>();
    for (const s of visibleSeries) {
      const pts = s.points.map((p) => ({ step: p.step, value: p.value }));
      out.set(s.id, xRange ? pts.filter((p) => inX(p.step)) : pts);
    }
    return out;
  }, [visibleSeries, xRange, xMin, xMax]);

  // Y scale (pool primary series; rawRange/dashed series contribute raw min/max only)
  const { primaryVals, mustInclude } = useMemo(() => {
    const prim: number[] = [];
    const must: number[] = [];
    for (const s of visibleSeries) {
      const sm = visibleSmoothed.get(s.id) ?? [];
      const rw = visibleRaw.get(s.id) ?? [];
      if (s.dashed || s.rawRange) {
        for (const p of rw) must.push(p.value);
      } else if (smoothing > 0 && sm.length > 0) {
        for (const p of sm) prim.push(p.value);
      } else {
        for (const p of rw) prim.push(p.value);
      }
    }
    return { primaryVals: prim, mustInclude: must };
  }, [visibleSeries, visibleSmoothed, visibleRaw, smoothing]);

  const rawYRange = robustYRange(primaryVals, yMinFloor, mustInclude, bounded);
  const canLog = allowLogScale && logScale && rawYRange.min > 0;
  const yMin = canLog ? Math.log10(Math.max(rawYRange.min, 1e-9)) : rawYRange.min;
  const yMax = canLog ? Math.log10(Math.max(rawYRange.max, 1e-9)) : rawYRange.max;
  const ySpan = yMax - yMin || 1;
  const yTransform = (v: number) => (canLog ? Math.log10(Math.max(v, 1e-9)) : v);
  const toY = (v: number) => PAD.top + ((yMax - yTransform(v)) / ySpan) * chartH;

  const buildPath = (pts: Pt[]) =>
    pts
      .filter((p) => inX(p.step))
      .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.step).toFixed(1)} ${toY(p.value).toFixed(1)}`)
      .join(" ");

  const formatY = (v: number) => {
    const real = canLog ? Math.pow(10, v) : v;
    if (Math.abs(real) >= 100) return real.toFixed(0);
    if (Math.abs(real) >= 1) return real.toFixed(2);
    if (Math.abs(real) >= 0.01) return real.toFixed(3);
    if (real === 0) return "0";
    return real.toExponential(1);
  };
  const formatTooltip = (v: number) => {
    if (v === 0) return "0.00";
    if (Math.abs(v) >= 0.01) return v.toFixed(2);
    return v.toExponential(2);
  };
  const formatX = (step: number) => {
    if (step >= 1_000_000) return `${(step / 1_000_000).toFixed(1)}M`;
    if (step >= 1000) return `${(step / 1000).toFixed(1)}k`;
    return String(step);
  };
  const yTickValues = [yMax, yMin + ySpan * 0.5, yMin];
  const xTickValues = [xMin, xMin + xSpan * 0.5, xMax].map((s) => Math.round(s));

  // Auto-extend xRange while brushing past the chart edge
  const isBrushing = brush !== null;
  useEffect(() => {
    if (!isBrushing) return;
    let raf = 0;
    const tick = () => {
      const px = pointerXRef.current;
      if (px !== null && xRange) {
        const span = xRange.max - xRange.min;
        if (px < 0) {
          const d = Math.max(1, Math.round((-px / Math.max(1, chartW)) * span));
          const newMin = Math.max(minStepAll, xRange.min - d);
          if (newMin !== xRange.min) {
            setXRange({ min: newMin, max: xRange.max });
            setBrush((b) => (b ? { ...b, endStep: newMin } : b));
          }
        } else if (px > chartW) {
          const d = Math.max(1, Math.round(((px - chartW) / Math.max(1, chartW)) * span));
          const newMax = Math.min(maxStepAll, xRange.max + d);
          if (newMax !== xRange.max) {
            setXRange({ min: xRange.min, max: newMax });
            setBrush((b) => (b ? { ...b, endStep: newMax } : b));
          }
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isBrushing, xRange, chartW, minStepAll, maxStepAll]);

  // Pointer handlers
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
      if (x >= 0 && x <= chartW) setBrush({ ...brush, endStep: pxToStep(x) });
      return;
    }
    if (x < 0 || x > chartW || y < -PAD.top || y > chartH + PAD.bottom) { setTooltip(null); return; }
    const targetStep = pxToStep(x);
    const stepTolerance = (16 / Math.max(1, chartW)) * xSpan;
    const values: { id: string; label: string; color: string; value: number; smoothValue: number | null }[] = [];
    let anchorY: number | null = null;
    for (const s of visibleSeries) {
      const pts = visibleRaw.get(s.id) ?? [];
      const sm = visibleSmoothed.get(s.id) ?? [];
      if (pts.length === 0) continue;
      if (targetStep < pts[0].step - stepTolerance || targetStep > pts[pts.length - 1].step + stepTolerance) continue;
      let idx = 0, dist = Math.abs(pts[0].step - targetStep);
      for (let i = 1; i < pts.length; i++) {
        const d = Math.abs(pts[i].step - targetStep);
        if (d < dist) { dist = d; idx = i; }
      }
      const smv = sm[idx]?.value ?? null;
      values.push({ id: s.id, label: s.label, color: s.color, value: pts[idx].value, smoothValue: smv });
      anchorY = (smoothing > 0 && smv !== null) ? smv : pts[idx].value;
    }
    if (values.length === 0 || anchorY === null) { setTooltip(null); return; }
    setTooltip({ px: toX(targetStep), py: toY(anchorY), step: Math.round(targetStep), values });
  };

  const onPointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!brush) return;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    pointerXRef.current = null;
    const a = Math.min(brush.startStep, brush.endStep);
    const b = Math.max(brush.startStep, brush.endStep);
    setBrush(null);
    const minSpan = (4 / Math.max(1, chartW)) * xSpan;
    if (b - a < minSpan) return;
    const newMin = Math.round(a), newMax = Math.round(b);
    if (newMax > newMin) setXRange({ min: newMin, max: newMax });
  };

  const onPointerLeave = () => { if (!brush) setTooltip(null); };
  const onDoubleClick = () => setXRange(null);

  const brushRect = brush ? (() => {
    const aStep = Math.min(brush.startStep, brush.endStep);
    const bStep = Math.max(brush.startStep, brush.endStep);
    const aClamped = Math.max(xMin, Math.min(xMax, aStep));
    const bClamped = Math.max(xMin, Math.min(xMax, bStep));
    const x = toX(aClamped);
    return { x, w: toX(bClamped) - x };
  })() : null;

  // Legend visibility is based on the FULL non-empty set (not the visible subset)
  // and only needs a measured width — so toggling a series off (even all of them)
  // never hides the legend that toggles it back on.
  const showLegend = width > 0 && seriesNonEmpty.length > 1;

  return (
    <div className="bg-gray-800 rounded p-2 border border-gray-700">
      <div className="flex items-center justify-between mb-1">
        <div className="text-sm font-medium text-gray-300">{title}</div>
        <div className="flex items-center gap-2">
          {headerExtra}
          {smoothable && (
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-gray-500">Smooth</span>
              <input
                type="range" min={0} max={0.99} step={0.01} value={smoothing}
                onChange={(e) => setSmoothing(parseFloat(e.target.value))}
                className="w-20 h-1 cursor-pointer" title="EMA smoothing"
              />
              <span className="text-[10px] text-gray-400 font-mono w-7 text-right">
                {(smoothing * 100).toFixed(0)}%
              </span>
            </div>
          )}
          {allowLogScale && (
            <button
              onClick={() => setLogScale((v) => !v)}
              className={`text-[10px] px-1.5 py-0.5 rounded transition-colors ${logScale ? "bg-blue-700 text-blue-100" : "bg-gray-700 hover:bg-gray-600 text-gray-300"}`}
              title="Toggle log Y scale"
            >log</button>
          )}
          {epochBoundaries && epochBoundaries.length >= 1 && (() => {
            const steps = epochBoundaries.map((b) => b.step);
            const eMin = Math.min(...steps), eMax = Math.max(...steps);
            const isEpochView = !!xRange && xRange.min === eMin && xRange.max === eMax;
            return (
              <button
                onClick={() => { if (isEpochView) setXRange(null); else setXRange({ min: eMin, max: eMax }); }}
                className={`text-[10px] px-1.5 py-0.5 rounded transition-colors ${isEpochView ? "bg-blue-700 text-blue-100" : "bg-gray-700 hover:bg-gray-600 text-gray-300"}`}
                title={isEpochView ? "Reset zoom" : "Zoom to epoch range"}
              >Epochs</button>
            );
          })()}
          {xRange && (
            <button
              onClick={() => setXRange(null)}
              className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
              title="Reset zoom (double-click chart, or ESC)"
            >↺ Reset</button>
          )}
        </div>
      </div>

      {showLegend && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 mb-1">
          {(legendOpen || seriesNonEmpty.length <= 4 ? seriesNonEmpty : seriesNonEmpty.slice(0, 4)).map((s) => {
            const hidden = hiddenIds.has(s.id);
            return (
              <button
                key={s.id}
                onClick={() => setHiddenIds((prev) => {
                  const next = new Set(prev);
                  if (next.has(s.id)) next.delete(s.id); else next.add(s.id);
                  return next;
                })}
                className={`inline-flex items-center gap-1 text-[10px] transition-opacity hover:text-gray-200 ${hidden ? "text-gray-600 opacity-50 line-through" : "text-gray-400"}`}
                title={hidden ? `Show ${s.label}` : `Hide ${s.label}`}
              >
                <span style={{ background: s.color, width: 10, height: 2, display: "inline-block", borderTop: s.dashed ? `2px dashed ${s.color}` : undefined }} />
                {s.label}
              </button>
            );
          })}
          {seriesNonEmpty.length > 4 && (
            <button onClick={() => setLegendOpen((v) => !v)} className="text-[10px] text-blue-400 hover:text-blue-300">
              {legendOpen ? "−" : `+${seriesNonEmpty.length - 4}`}
            </button>
          )}
        </div>
      )}

      <div ref={containerRef} className="w-full select-none" style={{ position: "relative", minHeight: height }}>
        {!hasEnoughData && (
          <div className="flex items-center justify-center text-gray-500 text-xs" style={{ height }}>
            {totalPoints < 2 ? "Not enough data" : ""}
          </div>
        )}
        {hasEnoughData && (
          <svg
            width={width} height={height}
            onPointerDown={onPointerDown} onPointerMove={onPointerMove}
            onPointerUp={onPointerUp} onPointerLeave={onPointerLeave}
            onDoubleClick={onDoubleClick}
            style={{ touchAction: "none", cursor: brush ? "ew-resize" : "crosshair" }}
          >
            {/* Y grid + ticks */}
            {yTickValues.map((v, i) => (
              <g key={`y${i}`}>
                <line x1={PAD.left} y1={toY(canLog ? Math.pow(10, v) : v)} x2={width - PAD.right} y2={toY(canLog ? Math.pow(10, v) : v)} stroke="#374151" strokeWidth={1} />
                <text x={PAD.left - 4} y={toY(canLog ? Math.pow(10, v) : v) + 3} textAnchor="end" fontSize={9} fill="#6b7280">{formatY(v)}</text>
              </g>
            ))}
            {/* X ticks */}
            {xTickValues.map((s, i) => (
              <text key={`x${i}`} x={toX(s)} y={height - 4} textAnchor="middle" fontSize={9} fill="#6b7280">{formatX(s)}</text>
            ))}

            {/* Epoch boundaries (dotted) */}
            {epochBoundaries?.filter((b) => inX(b.step)).map((b, i) => (
              <g key={`ep${i}`}>
                <line x1={toX(b.step)} y1={PAD.top} x2={toX(b.step)} y2={PAD.top + chartH} stroke="#6b7280" strokeWidth={1} strokeDasharray="3 3" opacity={0.7} />
                <text x={toX(b.step) + 2} y={PAD.top + 8} fontSize={8} fill="#9ca3af">E{b.epoch}</text>
              </g>
            ))}

            {/* Resume markers (solid, distinct color) */}
            {resumeMarkers?.filter((m) => inX(m.step)).map((m, i) => (
              <g key={`rs${i}`}>
                <line x1={toX(m.step)} y1={PAD.top} x2={toX(m.step)} y2={PAD.top + chartH} stroke="#f59e0b" strokeWidth={1.5} opacity={0.8} />
                <text x={toX(m.step) + 2} y={PAD.top + chartH - 2} fontSize={8} fill="#fbbf24">R{m.resume_seq}</text>
              </g>
            ))}

            {/* Series lines. When smoothing is on, the raw values are drawn faintly
                behind the smoothed line (so the actual noise is still visible). */}
            {smoothing > 0 && visibleSeries.map((s) => (
              <path key={`${s.id}-raw`} d={buildPath(s.points.map((p) => ({ step: p.step, value: p.value })))}
                fill="none" stroke={s.color} strokeWidth={1}
                strokeDasharray={s.dashed ? "4 3" : undefined} opacity={0.22} />
            ))}
            {visibleSeries.map((s) => {
              const pts = (smoothing > 0 ? (smoothedSeries.get(s.id) ?? []) : s.points.map((p) => ({ step: p.step, value: p.value })));
              return (
                <path key={s.id} d={buildPath(pts)} fill="none" stroke={s.color} strokeWidth={1.5}
                  strokeDasharray={s.dashed ? "4 3" : undefined} opacity={0.95} />
              );
            })}

            {/* Brush rect */}
            {brushRect && brushRect.w > 0 && (
              <rect x={brushRect.x} y={PAD.top} width={brushRect.w} height={chartH} fill="#3b82f6" opacity={0.15} />
            )}

            {/* Crosshair */}
            {tooltip && (
              <g>
                <line x1={tooltip.px} y1={PAD.top} x2={tooltip.px} y2={PAD.top + chartH} stroke="#9ca3af" strokeWidth={1} opacity={0.5} />
                <circle cx={tooltip.px} cy={tooltip.py} r={3} fill="#fff" />
              </g>
            )}
          </svg>
        )}

        {/* Tooltip box */}
        {tooltip && (
          <div
            className="absolute pointer-events-none bg-gray-900/95 border border-gray-700 rounded px-2 py-1 text-[10px] text-gray-200 shadow-lg"
            style={{
              left: Math.min(Math.max(tooltip.px + 8, 0), Math.max(0, width - 130)),
              top: Math.max(0, tooltip.py - 8),
              zIndex: 10,
            }}
          >
            <div className="text-gray-400 mb-0.5">step {tooltip.step}</div>
            {tooltip.values.map((v) => (
              <div key={v.id} className="flex items-center gap-1">
                <span style={{ background: v.color, width: 8, height: 8, borderRadius: 2, display: "inline-block" }} />
                <span className="text-gray-400">{v.label}</span>
                <span className="font-mono ml-auto">
                  {formatTooltip(smoothing > 0 && v.smoothValue !== null ? v.smoothValue : v.value)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
