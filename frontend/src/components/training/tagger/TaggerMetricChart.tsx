"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import type { TaggerTrainingMetric } from "@/utils/api";

export interface EpochBoundary {
  epoch: number;
  /** Global training step at which this epoch ended.  Only boundaries actually
   *  recorded during training are included; epochs before recording started are
   *  represented by the pre-tracking label ("Epoch 1-N"). */
  step: number;
}

interface TaggerMetricChartProps {
  data: TaggerTrainingMetric[];
  valueKey: "loss" | "f1" | "threshold" | "train_f1" | "precision" | "recall";
  /** Color used when only one resume_seq is present (initial/legacy single-curve render).
   *  Multi-series rendering cycles through the built-in palette below. */
  color: string;
  title: string;
  height?: number;
  smoothable?: boolean;
  defaultSmoothing?: number;
  yMinFloor?: number;
  /** Optional secondary series (e.g. val F1 overlaid on train F1).
   *  All resume_seq values are merged into a single dashed line. */
  secondaryValueKey?: "loss" | "f1" | "threshold" | "train_f1" | "precision" | "recall";
  secondaryColor?: string;
  secondaryLabel?: string;
  /** Epoch boundary markers to render as vertical lines.  Missing or empty ⇒ no markers. */
  epochBoundaries?: EpochBoundary[];
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
function applySmoothing(points: Point[], factor: number, initialState?: number): Point[] {
  if (factor <= 0 || points.length === 0) return points;
  const out: Point[] = [];
  let s = initialState !== undefined ? initialState : points[0].value;
  for (const p of points) {
    s = s * factor + p.value * (1 - factor);
    out.push({ step: p.step, value: s });
  }
  return out;
}

// Chart padding (used everywhere; constant)
const PAD = { top: 6, right: 8, bottom: 18, left: 44 };

// Robust Y-range: 5–95th percentiles on primary + 5% padding.
// mustInclude values (e.g. secondary series actual min/max) are always
// extended into the range so they are never clipped by the percentile cut.
function robustYRange(
  values: number[],
  yMinFloor: number,
  mustInclude: number[] = [],
): { min: number; max: number } {
  const valid = values.filter((v) => Number.isFinite(v));
  let lo: number, hi: number;
  if (valid.length === 0) {
    lo = yMinFloor; hi = yMinFloor + 1;
  } else if (valid.length === 1) {
    const v = valid[0];
    const pad = Math.max(Math.abs(v) * 0.1, 1e-6);
    lo = Math.max(yMinFloor, v - pad); hi = v + pad;
  } else {
    const sorted = [...valid].sort((a, b) => a - b);
    lo = sorted[Math.floor(sorted.length * 0.05)];
    hi = sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)];
  }
  // Extend to include secondary series min/max before applying padding
  for (const v of mustInclude) {
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
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
  secondaryValueKey,
  secondaryColor = "#22c55e",
  secondaryLabel,
  epochBoundaries,
}: TaggerMetricChartProps) {
  // Group primary points by resume_seq so each resume becomes its own curve.
  const groups = useMemo<Map<number, Point[]>>(() => {
    const m = new Map<number, Point[]>();
    for (const d of data) {
      const v = d[valueKey] as number | null | undefined;
      if (v === null || v === undefined || !Number.isFinite(v)) continue;
      const seq = d.resume_seq ?? 0;
      if (!m.has(seq)) m.set(seq, []);
      m.get(seq)!.push({ step: d.step, value: v });
    }
    // Sort each group by step (data may arrive out of order across resumes)
    for (const arr of m.values()) arr.sort((a, b) => a.step - b.step);
    return m;
  }, [data, valueKey]);

  // Secondary series: merge ALL resume_seq into one sorted list of points.
  const secondaryPoints = useMemo<Point[]>(() => {
    if (!secondaryValueKey) return [];
    const pts: Point[] = [];
    for (const d of data) {
      const v = d[secondaryValueKey] as number | null | undefined;
      if (v === null || v === undefined || !Number.isFinite(v)) continue;
      pts.push({ step: d.step, value: v });
    }
    pts.sort((a, b) => a.step - b.step);
    return pts;
  }, [data, secondaryValueKey]);

  // Resume seqs in render order (ascending so newer overlays older)
  const groupKeys = useMemo(() => [...groups.keys()].sort((a, b) => a - b), [groups]);

  // Latest resume (highest seq) — used as the source for tooltip nearest-point search
  const latestSeq = groupKeys.length > 0 ? groupKeys[groupKeys.length - 1] : 0;

  // Pooled range across all primary groups for x-axis bounds
  const allPoints = useMemo(
    () => groupKeys.flatMap((seq) => groups.get(seq) ?? []),
    [groups, groupKeys]
  );
  const minStepAll = useMemo(() => {
    const candidates = [
      ...(allPoints.length > 0 ? [Math.min(...allPoints.map((p) => p.step))] : []),
      ...(secondaryPoints.length > 0 ? [Math.min(...secondaryPoints.map((p) => p.step))] : []),
    ];
    return candidates.length > 0 ? Math.min(...candidates) : 0;
  }, [allPoints, secondaryPoints]);
  const maxStepAll = useMemo(() => {
    const candidates = [
      ...(allPoints.length > 0 ? [Math.max(...allPoints.map((p) => p.step))] : []),
      ...(secondaryPoints.length > 0 ? [Math.max(...secondaryPoints.map((p) => p.step))] : []),
    ];
    return candidates.length > 0 ? Math.max(...candidates) : 0;
  }, [allPoints, secondaryPoints]);
  const totalPoints = allPoints.length + secondaryPoints.length;

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
    // Per-resume values at the hovered step (keyed by resume_seq)
    seriesValues: Map<number, { value: number; smoothValue: number | null }>;
    // Secondary series value at the hovered step
    secondaryValue?: { value: number; smoothValue: number | null };
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

  // Visible secondary points
  const visibleSecondary = useMemo<Point[]>(
    () => xRange
      ? secondaryPoints.filter((p) => p.step >= xRange.min && p.step <= xRange.max)
      : secondaryPoints,
    [secondaryPoints, xRange]
  );

  // Per-group smoothed series (smoothing must be applied within each
  // group; mixing across resumes would create jumps at the boundary).
  const smoothedAllGroups = useMemo<Map<number, Point[]>>(() => {
    const out = new Map<number, Point[]>();
    // Process groups in ascending resume_seq order so the EMA state
    // at the end of seq N can seed the start of seq N+1, preventing
    // a cold-start discontinuity at resume boundaries.
    let carryState: number | undefined;
    for (const seq of groupKeys) {
      const pts = groups.get(seq) ?? [];
      const smoothed = applySmoothing(pts, smoothing, carryState);
      out.set(seq, smoothed);
      if (smoothed.length > 0) carryState = smoothed[smoothed.length - 1].value;
    }
    return out;
  }, [groups, groupKeys, smoothing]);

  // Smoothed secondary (treat as a single group — no resume-boundary carry needed)
  const smoothedSecondaryAll = useMemo<Point[]>(
    () => applySmoothing(secondaryPoints, smoothing),
    [secondaryPoints, smoothing]
  );

  const smoothedVisibleGroups = useMemo<Map<number, Point[]>>(() => {
    if (!xRange) return smoothedAllGroups;
    const out = new Map<number, Point[]>();
    for (const [seq, pts] of smoothedAllGroups) {
      out.set(seq, pts.filter((p) => p.step >= xRange.min && p.step <= xRange.max));
    }
    return out;
  }, [smoothedAllGroups, xRange]);

  const smoothedVisibleSecondary = useMemo<Point[]>(
    () => xRange
      ? smoothedSecondaryAll.filter((p) => p.step >= xRange.min && p.step <= xRange.max)
      : smoothedSecondaryAll,
    [smoothedSecondaryAll, xRange]
  );

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

  // Y scale: primary uses percentile-based robust range (handles loss spikes).
  // Secondary actual min/max are passed as mustInclude so they are never clipped.
  const { primaryValsForRange, secondaryValsForRange } = useMemo(() => {
    const primaryVals = smoothing > 0 && smoothedVisibleAllPoints.length > 0
      ? smoothedVisibleAllPoints.map((p) => p.value)
      : visiblePoints.map((p) => p.value);
    const secondaryVals = secondaryValueKey
      ? smoothing > 0 && smoothedVisibleSecondary.length > 0
        ? smoothedVisibleSecondary.map((p) => p.value)
        : visibleSecondary.map((p) => p.value)
      : [];
    return { primaryValsForRange: primaryVals, secondaryValsForRange: secondaryVals };
  }, [
    smoothing, smoothedVisibleAllPoints, visiblePoints,
    secondaryValueKey, smoothedVisibleSecondary, visibleSecondary,
  ]);
  const { min: yMin, max: yMax } = robustYRange(
    primaryValsForRange,
    yMinFloor,
    secondaryValsForRange,  // always include secondary actual min/max
  );
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
    // Crosshair follows the cursor in step-space.  Each resume independently
    // finds its nearest point to targetStep and shows its own value.
    // A resume is only included when the cursor falls within its actual
    // step range (±16px tolerance at the endpoints).
    const targetStep = pxToStep(x);

    // Tolerance: 16px in step-space — allows targeting the endpoint of a resume
    const stepTolerance = (16 / Math.max(1, chartW)) * xSpan;

    const seriesValues = new Map<number, { value: number; smoothValue: number | null }>();
    for (const seq of groupKeys) {
      const pts   = visibleGroups.get(seq) ?? [];
      const smPts = smoothedVisibleGroups.get(seq) ?? [];
      if (pts.length === 0) continue;
      // Only include this resume when the cursor is within its actual step range
      const resumeMin = pts[0].step;
      const resumeMax = pts[pts.length - 1].step;
      if (targetStep < resumeMin - stepTolerance || targetStep > resumeMax + stepTolerance) continue;
      let idx = 0;
      let dist = Math.abs(pts[0].step - targetStep);
      for (let i = 1; i < pts.length; i++) {
        const d = Math.abs(pts[i].step - targetStep);
        if (d < dist) { dist = d; idx = i; }
      }
      seriesValues.set(seq, {
        value: pts[idx].value,
        smoothValue: smPts[idx]?.value ?? null,
      });
    }

    // Secondary series nearest point
    let secondaryValue: { value: number; smoothValue: number | null } | undefined;
    if (visibleSecondary.length > 0) {
      const secMin = visibleSecondary[0].step;
      const secMax = visibleSecondary[visibleSecondary.length - 1].step;
      if (targetStep >= secMin - stepTolerance && targetStep <= secMax + stepTolerance) {
        let idx = 0;
        let dist = Math.abs(visibleSecondary[0].step - targetStep);
        for (let i = 1; i < visibleSecondary.length; i++) {
          const d = Math.abs(visibleSecondary[i].step - targetStep);
          if (d < dist) { dist = d; idx = i; }
        }
        secondaryValue = {
          value: visibleSecondary[idx].value,
          smoothValue: smoothedVisibleSecondary[idx]?.value ?? null,
        };
      }
    }

    if (seriesValues.size === 0 && !secondaryValue) {
      setTooltip(null);
      return;
    }

    // Pin the crosshair dot on the latest visible primary resume (or secondary).
    const presentSeqs = [...seriesValues.keys()].sort((a, b) => a - b);
    let anchorY: number;
    if (presentSeqs.length > 0) {
      const anchorSeq = presentSeqs[presentSeqs.length - 1];
      const anchorEntry = seriesValues.get(anchorSeq)!;
      anchorY = anchorEntry.smoothValue !== null && smoothing > 0
        ? anchorEntry.smoothValue
        : anchorEntry.value;
    } else {
      // Only secondary present
      anchorY = secondaryValue!.smoothValue !== null && smoothing > 0
        ? secondaryValue!.smoothValue!
        : secondaryValue!.value;
    }

    setTooltip({
      px: toX(targetStep),
      py: toY(anchorY),
      step: Math.round(targetStep),
      seriesValues,
      secondaryValue,
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

  // Whether the legend needs to be shown (multi-resume OR secondary present)
  const showLegend = hasEnoughData && (groups.size > 1 || (secondaryValueKey && secondaryPoints.length > 0));
  // Whether to show resume labels in legend
  const showResumeLabels = groups.size > 1;

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
          {epochBoundaries && epochBoundaries.length >= 1 && (() => {
            const steps = epochBoundaries.map((b) => b.step);
            const eMin = Math.min(...steps);
            const eMax = Math.max(...steps);
            const isEpochView = xRange && xRange.min === eMin && xRange.max === eMax;
            return (
              <button
                onClick={() => {
                  if (isEpochView) setXRange(null);
                  else setXRange({ min: eMin, max: eMax });
                }}
                className={`text-[10px] px-1.5 py-0.5 rounded transition-colors ${
                  isEpochView
                    ? "bg-blue-700 text-blue-100"
                    : "bg-gray-700 hover:bg-gray-600 text-gray-300"
                }`}
                title={isEpochView ? "Reset zoom" : "Zoom to epoch boundary range"}
              >
                Epochs
              </button>
            );
          })()}
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
        {/* Legend (multi-resume OR secondary series present) */}
        {showLegend && (
          <div className="absolute top-1 right-1 flex flex-wrap gap-1.5 text-[10px] bg-gray-900/80 px-1.5 py-0.5 rounded border border-gray-700 z-20 pointer-events-none">
            {showResumeLabels && groupKeys.map((seq) => (
              <div key={`lg-${seq}`} className="flex items-center gap-1">
                <span
                  className="inline-block w-2 h-2 rounded-sm"
                  style={{ background: colorForResume(seq, color) }}
                />
                <span className="text-gray-300">{labelForResume(seq)}</span>
              </div>
            ))}
            {secondaryValueKey && secondaryPoints.length > 0 && (
              <div className="flex items-center gap-1">
                {/* Dashed line swatch for secondary */}
                <svg width="16" height="8" className="inline-block">
                  <line x1="0" y1="4" x2="16" y2="4" stroke={secondaryColor} strokeWidth="1.5" strokeDasharray="4 2" />
                </svg>
                <span style={{ color: secondaryColor }}>{secondaryLabel ?? secondaryValueKey}</span>
              </div>
            )}
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

          {/* Epoch boundary vertical lines and labels */}
          {epochBoundaries && epochBoundaries.length > 0 && (() => {
            const firstKnown = epochBoundaries[0];
            const visibleBdrs = epochBoundaries.filter(
              (b) => b.step >= xMin - (xSpan * 0.02) && b.step <= xMax + (xSpan * 0.02)
            );
            // Pre-tracking region: epochs 1...(firstKnown.epoch-1) happened before step tracking
            const showPreLabel = firstKnown.epoch > 1 && xMin < firstKnown.step;
            return (
              <g>
                {showPreLabel && (() => {
                  const regionEnd = Math.min(firstKnown.step, xMax);
                  const regionStart = xMin;
                  if (regionEnd <= regionStart) return null;
                  const midX = toX((regionStart + regionEnd) / 2);
                  return (
                    <text
                      x={Math.max(PAD.left + 4, Math.min(PAD.left + chartW - 4, midX))}
                      y={PAD.top + 10}
                      textAnchor="middle"
                      fontSize={8}
                      fill="#6b7280"
                    >
                      Epoch 1-{firstKnown.epoch - 1}
                    </text>
                  );
                })()}
                {visibleBdrs.map((b) => {
                  const bx = toX(b.step);
                  const labelX = Math.max(PAD.left + 2, Math.min(PAD.left + chartW - 2, bx));
                  return (
                    <g key={`eb-${b.epoch}`}>
                      <line
                        x1={bx} x2={bx}
                        y1={PAD.top} y2={PAD.top + chartH}
                        stroke="#374151"
                        strokeWidth={1}
                        strokeDasharray="4 3"
                      />
                      <text
                        x={labelX}
                        y={PAD.top + 9}
                        textAnchor="middle"
                        fontSize={8}
                        fill="#9ca3af"
                      >
                        Ep {b.epoch}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })()}

          {/* Secondary series (dashed, below primary so primary renders on top) */}
          {secondaryValueKey && (() => {
            const rawPts = visibleSecondary;
            const smPts  = smoothedVisibleSecondary;
            const dRaw = rawPts.length >= 2 ? buildPath(rawPts) : "";
            const dSm  = smoothing > 0 && smPts.length >= 2 ? buildPath(smPts) : "";
            const lonePt = (!dRaw && rawPts.length === 1) ? rawPts[0] : null;
            return (
              <g key="secondary-series">
                {dRaw && (
                  <path
                    d={dRaw}
                    fill="none"
                    stroke={secondaryColor}
                    strokeWidth={1.2}
                    strokeDasharray="4 3"
                    opacity={smoothing > 0 ? 0.3 : 1}
                  />
                )}
                {dSm && (
                  <path
                    d={dSm}
                    fill="none"
                    stroke={secondaryColor}
                    strokeWidth={1.6}
                    strokeDasharray="4 3"
                  />
                )}
                {lonePt && (
                  <circle
                    cx={toX(lonePt.step)}
                    cy={toY(lonePt.value)}
                    r={2.5}
                    fill={secondaryColor}
                  />
                )}
              </g>
            );
          })()}

          {/* Per-resume raw + smoothed lines (older first, newer on top).
              Prepend the previous resume's last point so each segment connects
              to the prior one — without this, a resume with only 1 point
              (e.g. sparse F1 logged once per epoch) renders an invisible
              single moveto and the legend entry has nothing to point at. */}
          {groupKeys.map((seq, gi) => {
            const seqColor = colorForResume(seq, color);
            const visPts   = visibleGroups.get(seq) ?? [];
            const visSm    = smoothedVisibleGroups.get(seq) ?? [];

            // Carry-in from the chronologically previous resume (use the
            // full unfiltered group so the connector survives zooming —
            // SVG clips off-canvas coordinates naturally).
            const prevSeq = gi > 0 ? groupKeys[gi - 1] : null;
            const prevRawAll = prevSeq !== null ? (groups.get(prevSeq) ?? []) : [];
            const prevSmAll  = prevSeq !== null ? (smoothedAllGroups.get(prevSeq) ?? []) : [];
            const carryRaw = prevRawAll.length > 0 ? [prevRawAll[prevRawAll.length - 1]] : [];
            const carrySm  = prevSmAll.length  > 0 ? [prevSmAll[prevSmAll.length  - 1]] : [];

            const rawPath = [...carryRaw, ...visPts];
            const smPath  = [...carrySm,  ...visSm];

            const dRaw = rawPath.length >= 2 ? buildPath(rawPath) : "";
            const dSm  = smoothing > 0 && smPath.length >= 2 ? buildPath(smPath) : "";

            // For series with no carry-in AND only 1 point (i.e. seq 0
            // with a single validation row), still mark the point so it's
            // visible — a tiny dot is better than nothing.
            const lonePoint = (!dRaw && visPts.length === 1) ? visPts[0] : null;

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
                {lonePoint && (
                  <circle
                    cx={toX(lonePoint.step)}
                    cy={toY(lonePoint.value)}
                    r={2.5}
                    fill={seqColor}
                  />
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

          {/* Tooltip markers — one circle per resume at their respective y */}
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
              {groupKeys.map((seq) => {
                const entry = tooltip.seriesValues.get(seq);
                if (!entry) return null;
                const displayV = (smoothing > 0 && entry.smoothValue !== null) ? entry.smoothValue : entry.value;
                return (
                  <circle
                    key={`dot-${seq}`}
                    cx={tooltip.px}
                    cy={toY(displayV)}
                    r={3}
                    fill={colorForResume(seq, color)}
                  />
                );
              })}
              {tooltip.secondaryValue && (
                <circle
                  cx={tooltip.px}
                  cy={toY(
                    smoothing > 0 && tooltip.secondaryValue.smoothValue !== null
                      ? tooltip.secondaryValue.smoothValue
                      : tooltip.secondaryValue.value
                  )}
                  r={3}
                  fill={secondaryColor}
                  stroke={secondaryColor}
                  strokeDasharray="2 1"
                />
              )}
            </>
          )}
        </svg>
        )}

        {/* Tooltip box (HTML overlay) */}
        {hasEnoughData && tooltip && !brush && (
          <div
            className="pointer-events-none absolute bg-gray-900 border border-gray-600 rounded px-2 py-1 text-[10px] font-mono text-gray-200 shadow-lg whitespace-nowrap"
            style={{
              left: Math.min(width - 160, tooltip.px + 8),
              top: Math.max(0, tooltip.py - 8),
              zIndex: 10,
            }}
          >
            <div className="text-gray-400 mb-0.5">step {tooltip.step.toLocaleString()}</div>
            {groupKeys.map((seq) => {
              const entry = tooltip.seriesValues.get(seq);
              if (!entry) return null;
              const seqColor = colorForResume(seq, color);
              const displayV = smoothing > 0 && entry.smoothValue !== null ? entry.smoothValue : entry.value;
              return (
                <div key={`tv-${seq}`} className="flex items-center gap-1">
                  <span className="inline-block w-1.5 h-1.5 rounded-sm flex-shrink-0" style={{ background: seqColor }} />
                  <span style={{ color: seqColor }}>{showResumeLabels ? labelForResume(seq) : (title)}:</span>
                  <span>{formatTooltip(displayV)}</span>
                  {smoothing > 0 && entry.smoothValue !== null && (
                    <span className="text-gray-500">(raw {formatTooltip(entry.value)})</span>
                  )}
                </div>
              );
            })}
            {tooltip.secondaryValue && (
              <div className="flex items-center gap-1">
                <svg width="10" height="8" className="flex-shrink-0">
                  <line x1="0" y1="4" x2="10" y2="4" stroke={secondaryColor} strokeWidth="1.5" strokeDasharray="3 2" />
                </svg>
                <span style={{ color: secondaryColor }}>{secondaryLabel ?? secondaryValueKey}:</span>
                <span>{formatTooltip(
                  smoothing > 0 && tooltip.secondaryValue.smoothValue !== null
                    ? tooltip.secondaryValue.smoothValue
                    : tooltip.secondaryValue.value
                )}</span>
                {smoothing > 0 && tooltip.secondaryValue.smoothValue !== null && (
                  <span className="text-gray-500">(raw {formatTooltip(tooltip.secondaryValue.value)})</span>
                )}
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
