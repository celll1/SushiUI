"use client";

import { useState, useEffect, useMemo, useRef, useId } from "react";

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
  /** Render against a separate, independently-scaled right-hand Y-axis instead
   *  of pooling into the primary Y-range. For series living on a wildly
   *  different scale than the primary series (e.g. learning rate ~1e-4 vs
   *  loss ~0.03), which would otherwise render as an invisible flat line. */
  secondaryAxis?: boolean;
  /** Explicit axis assignment. When absent the legacy `secondaryAxis` hint
   *  decides (right when set, left otherwise). */
  axis?: AxisSide;
  /** How the series is drawn.
   *  - `line` (default): polyline, optionally EMA-smoothed.
   *  - `markers`: a dot per sample plus a faint joining line, for a series whose
   *    samples are too few or too widely spaced to read as a curve.
   *  - `band`: a 0/1 state strip in a reserved row below the plot. A 0/1 series
   *    drawn as a polyline at ~1px per 20 steps is a wall of vertical
   *    connectors; as a strip it reads as "on here, off there". A band belongs
   *    to NO Y-axis and is excluded from every axis's pooling. */
  renderMode?: "line" | "markers" | "band";
  /** Exempt from EMA smoothing (0/1 indicators and sparse periodic probes read
   *  as their own values, not as a lagged average of them). */
  noSmooth?: boolean;
}

export type AxisSide = "left" | "right";

/** How an axis derives its domain. `auto` runs the robust percentile clip with
 *  an optional hard floor; `fixed` pins the domain and bypasses the clip. */
export type MetricRangePolicy =
  | { kind: "auto"; floor?: number }
  | { kind: "fixed"; min: number; max: number };

export interface AxisConfig {
  scale: "linear" | "log";
  range: MetricRangePolicy;
  /** Draw a gridline at 0 when the domain spans it. Defaults on for `fixed`. */
  zeroLine?: boolean;
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
  /** Header controls rendered AFTER the smoothing slider, where the legacy log
   *  button sits — for a caller that owns its own per-axis log toggles. */
  headerTrailing?: React.ReactNode;
  /** Declared per-axis scale/domain. When absent the entire legacy path runs:
   *  left = robustYRange(yMinFloor, bounded) + the `allowLogScale` toggle,
   *  right = independent linear auto-range. */
  axes?: { left: AxisConfig; right?: AxisConfig };
  /** Controlled legend visibility; falls back to local state when absent. */
  hiddenIds?: Set<string>;
  onHiddenIdsChange?: (next: Set<string>) => void;
  /** Controlled smoothing; falls back to local state when absent. */
  smoothing?: number;
  onSmoothingChange?: (next: number) => void;
}

interface Pt { step: number; value: number; }

interface TooltipValue {
  id: string;
  label: string;
  color: string;
  value: number;
  smoothValue: number | null;
  /** The matched sample's own step, when it is far enough from the hovered step
   *  that reporting it under the crosshair's step would be a lie (a sparse
   *  series matched via its widened tolerance). Null when they agree. */
  realStep: number | null;
}

const BASE_PAD = { top: 6, right: 8, bottom: 18, left: 44 };
// One band row: a 6px strip plus 2px of separation, reserved below the plot and
// above the x-tick text.
const BAND_ROW_H = 8;
// Clear of the bottom Y-axis tick label, whose glyphs reach ~5px below the plot.
const BAND_TOP_GAP = 5;
const BAND_H = 6;
// Extra right margin reserved for the secondary axis's tick labels when at
// least one visible series uses it.
const SECONDARY_AXIS_RIGHT_PAD = 34;
// Domain the right axis falls back to when no series is assigned to it (module
// constant so the memo below keeps a stable identity).
const INACTIVE_RIGHT_RANGE = { min: 0, max: 1 };

const axisOf = (s: ChartSeries): AxisSide => s.axis ?? (s.secondaryAxis ? "right" : "left");
const isBand = (s: ChartSeries) => s.renderMode === "band";

/** Median step spacing of a series, or null when it has fewer than two
 *  distinct steps. Used to widen the tooltip's nearest-point tolerance for a
 *  sparse series (kept local so this shared chart stays independent of the
 *  training-side metric catalog). */
function medianStepGap(pts: { step: number }[]): number | null {
  if (pts.length < 2) return null;
  const gaps: number[] = [];
  for (let i = 1; i < pts.length; i++) {
    const g = pts[i].step - pts[i - 1].step;
    if (g > 0) gaps.push(g);
  }
  if (gaps.length === 0) return null;
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

/**
 * Bias-corrected EMA (the Adam correction, same algebra).
 *
 * Seeding `s = points[0].value` anchored the whole head of the curve to ONE
 * sample: at factor 0.99 that sample still carries 99% of the weight at the
 * second point and ~37% a hundred points later, so a run whose first step
 * happened to draw a high-loss timestep drew a curve pulled visibly upward for
 * its first few hundred points. Two runs then could not be compared at the
 * start, and neither could two series inside one run.
 *
 * Accumulating from zero and dividing by `1 - factor^n` fixes it exactly: the
 * result is the weighted mean of the points seen SO FAR, so point 1 is its own
 * value, point 2 is a proper weighted pair, and the tail is the same EMA as
 * before. No luck of the draw, no new parameter.
 */
function applySmoothing(points: Pt[], factor: number): Pt[] {
  if (factor <= 0 || points.length === 0) return points;
  const out: Pt[] = [];
  let s = 0;
  let bias = 1;
  for (const p of points) {
    s = s * factor + p.value * (1 - factor);
    bias *= factor;
    out.push({ step: p.step, value: s / (1 - bias) });
  }
  return out;
}

/**
 * The slider travels in the EMA's effective WINDOW, log-spaced, not in its
 * factor.
 *
 * The factor and the window are related by N = 1/(1-f), which is violently
 * non-linear: a linear 0..0.99 slider spent 90 of its 100 positions on windows
 * of 10 points or fewer, and offered exactly ONE position at 50 points or more.
 * On a 25,000-point run that made every useful setting land in the last
 * fraction of the travel, and its maximum -- a 100-point window, 0.4% of the
 * run -- was still the weakest smoothing anyone wanted.
 *
 * Log-spacing N over 1..MAX_SMOOTH_WINDOW gives even travel in the thing the
 * eye actually responds to, and the readout is a point count rather than a
 * percentage that means nothing on its own.
 *
 * Raising the ceiling is only safe because the EMA is bias-corrected: with the
 * old seeded form a 2000-point window would have pinned the head of the curve
 * to its first sample for the whole visible range.
 */
const MAX_SMOOTH_WINDOW = 2000;

export function smoothingToWindow(factor: number): number {
  return factor <= 0 ? 1 : Math.min(MAX_SMOOTH_WINDOW, 1 / (1 - factor));
}

/** Slider position (0..1) -> EMA factor. */
function positionToSmoothing(pos: number): number {
  const n = Math.exp(pos * Math.log(MAX_SMOOTH_WINDOW));
  return n <= 1 ? 0 : 1 - 1 / n;
}

/** EMA factor -> slider position (0..1). */
function smoothingToPosition(factor: number): number {
  return Math.log(smoothingToWindow(factor)) / Math.log(MAX_SMOOTH_WINDOW);
}

/** p5-p95 of ONE series, or its full extent when `bounded`. */
function seriesExtent(values: number[], bounded: boolean): { lo: number; hi: number } | null {
  const valid = values.filter((v) => Number.isFinite(v));
  if (valid.length === 0) return null;
  if (valid.length === 1) return { lo: valid[0], hi: valid[0] };
  const sorted = [...valid].sort((a, b) => a - b);
  if (bounded) return { lo: sorted[0], hi: sorted[sorted.length - 1] };
  return {
    lo: sorted[Math.floor(sorted.length * 0.05)],
    hi: sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)],
  };
}

function robustYRange(
  perSeries: number[][],
  yMinFloor: number,
  mustInclude: number[] = [],
  bounded = false,
): { min: number; max: number } {
  // Percentile-clip each series SEPARATELY and take the union, rather than
  // clipping one pooled array. Pooling let a dense series decide the ceiling for
  // a sparse one: loss_null is 3.2% of the pooled points and sits in the highest
  // band, so the pooled p95 cut all of it and it drew as a flat line pinned to
  // the top of the frame. Per-series, each keeps its own middle 90% and the axis
  // covers all of them, while within-series outliers are still clipped -- which
  // was the point of the percentile in the first place.
  const extents = perSeries.map((v) => seriesExtent(v, bounded)).filter(Boolean) as { lo: number; hi: number }[];
  const valid = extents.length ? [Math.min(...extents.map((e) => e.lo)), Math.max(...extents.map((e) => e.hi))] : [];
  let lo: number, hi: number;
  if (valid.length === 0) {
    // yMinFloor is -Infinity for a floorless axis, which would make the whole
    // domain NaN. An axis with nothing on it (every selected series is a band,
    // say) just needs a sane empty frame.
    lo = Number.isFinite(yMinFloor) ? yMinFloor : 0;
    hi = lo + 1;
  } else if (valid[0] === valid[1]) {
    const v = valid[0];
    const pad = Math.max(Math.abs(v) * 0.1, 1e-6);
    lo = Math.max(yMinFloor, v - pad); hi = v + pad;
  } else {
    [lo, hi] = valid;
  }
  for (const v of mustInclude) {
    if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  const range = hi - lo || Math.max(Math.abs(hi) * 0.1, 1e-6);
  const pad = range * 0.05;
  return { min: Math.max(yMinFloor, lo - pad), max: hi + pad };
}

/** Log-domain counterpart of robustYRange: same p5-p95 percentile-clip idea,
 *  restricted to strictly-positive values (log10 is undefined at/below 0) and
 *  padded in LOG space (not linear) so the padding stays proportionate across
 *  a range spanning orders of magnitude. Returns null when no positive value
 *  exists — the caller uses that to disable log mode rather than silently
 *  showing an empty/degenerate axis.
 *
 *  Using the single absolute minimum positive value as the log floor (an
 *  earlier version of this) let one low outlier — e.g. a transient near-zero
 *  point during a resume's warm-up — pin the whole axis low and compress the
 *  steady-state data into a thin top band. Percentile-clipping like the
 *  linear axis avoids that. */
function robustPositiveLogRange(values: number[]): { min: number; max: number } | null {
  const valid = values.filter((v) => Number.isFinite(v) && v > 0);
  if (valid.length === 0) return null;
  let lo: number, hi: number;
  if (valid.length === 1) {
    lo = valid[0]; hi = valid[0];
  } else {
    const sorted = [...valid].sort((a, b) => a - b);
    lo = sorted[Math.floor(sorted.length * 0.05)];
    hi = sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)];
  }
  const logLo = Math.log10(lo);
  const logHi = Math.log10(hi);
  const logPad = (logHi - logLo) * 0.05 || 0.15;
  return { min: Math.pow(10, logLo - logPad), max: Math.pow(10, logHi + logPad) };
}

interface AxisScale {
  min: number;
  max: number;
  span: number;
  isLog: boolean;
  /** Whether a log domain exists at all (drives the log button's disabled state). */
  logAvailable: boolean;
  toY: (realValue: number) => number;
  /** Domain-space tick values (log10'd when isLog), top -> bottom. */
  tickValues: number[];
  /** Domain value -> real value (undoes the log10 when isLog). */
  fromDomain: (domainValue: number) => number;
  /** Y pixel of the zero line, or null when it should not be drawn. */
  zeroY: number | null;
}

/** One axis's pooling + domain + pixel mapping. Both the left and the right
 *  axis go through this; the two previously-duplicated blocks differed only in
 *  which series they pooled and in whether a floor/bounded/log applied. */
function useAxisScale(o: {
  /** Series assigned to this axis (memoized by the caller). */
  seriesForAxis: ChartSeries[];
  visibleSmoothed: Map<string, Pt[]>;
  visibleRaw: Map<string, Pt[]>;
  smoothing: number;
  /** Hard lower clamp for an auto range (legacy `yMinFloor`; -Infinity = none). */
  yMinFloor: number;
  bounded: boolean;
  allowLog: boolean;
  logOn: boolean;
  config?: AxisConfig;
  /** Domain used when the axis carries no series (legacy right axis: 0..1). */
  inactiveRange: { min: number; max: number } | null;
  padTop: number;
  chartH: number;
}): AxisScale {
  const { perSeries, values, mustInclude } = useMemo(() => {
    const per: number[][] = [];
    const flat: number[] = [];
    const must: number[] = [];
    for (const s of o.seriesForAxis) {
      const sm = o.visibleSmoothed.get(s.id) ?? [];
      const rw = o.visibleRaw.get(s.id) ?? [];
      if (s.rawRange) {
        for (const p of rw) must.push(p.value);
        continue;
      }
      const src = (o.smoothing > 0 && sm.length > 0) ? sm : rw;
      const one = src.map((p) => p.value);
      if (one.length) { per.push(one); flat.push(...one); }
    }
    // `values` stays flat for the log range, which needs every positive sample.
    return { perSeries: per, values: flat, mustInclude: must };
  }, [o.seriesForAxis, o.visibleSmoothed, o.visibleRaw, o.smoothing]);

  const policy = o.config?.range;
  const fixed = policy && policy.kind === "fixed" ? policy : null;
  const floor = !policy ? o.yMinFloor : policy.kind === "auto" ? (policy.floor ?? -Infinity) : -Infinity;
  const inactive = o.inactiveRange;
  const empty = o.seriesForAxis.length === 0;

  const linearRange = useMemo(() => {
    if (fixed) return { min: fixed.min, max: fixed.max };
    if (empty && inactive) return inactive;
    return robustYRange(perSeries, floor, mustInclude, o.bounded);
  }, [fixed?.min, fixed?.max, empty, inactive?.min, inactive?.max, values, floor, mustInclude, o.bounded]);

  const logRange = useMemo(() => {
    if (!o.allowLog) return null;
    // A fixed domain declares its own bounds; only a positive one is log-able.
    if (fixed) return fixed.min > 0 ? { min: fixed.min, max: fixed.max } : null;
    return robustPositiveLogRange([...values, ...mustInclude]);
  }, [o.allowLog, fixed?.min, fixed?.max, values, mustInclude]);

  const isLog = o.allowLog && o.logOn && logRange !== null;
  const min = isLog && logRange ? Math.log10(logRange.min) : linearRange.min;
  const max = isLog && logRange ? Math.log10(logRange.max) : linearRange.max;
  const span = max - min || 1;
  const toY = (v: number) =>
    o.padTop + ((max - (isLog ? Math.log10(Math.max(v, 1e-9)) : v)) / span) * o.chartH;
  const wantZeroLine = policy ? (o.config?.zeroLine ?? policy.kind === "fixed") : false;
  return {
    min, max, span, isLog,
    logAvailable: logRange !== null,
    toY,
    tickValues: [max, min + span * 0.5, min],
    fromDomain: (d: number) => (isLog ? Math.pow(10, d) : d),
    zeroY: wantZeroLine && !isLog && min < 0 && max > 0 ? toY(0) : null,
  };
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
  headerTrailing,
  axes,
  hiddenIds: hiddenIdsProp,
  onHiddenIdsChange,
  smoothing: smoothingProp,
  onSmoothingChange,
}: SharedMetricChartProps) {
  // Unique per-instance id for the plot-area clipPath (multiple charts render
  // in the same DOM — e.g. Loss + GradNorm + ParamChange stacked in one run's
  // metrics tab — so a shared/static id would collide and one chart's clip
  // rect would silently clip a DIFFERENT chart's series). `:` from useId()'s
  // default format isn't a valid bare SVG id char in older UAs, so strip it.
  const clipIdRaw = useId();
  const clipId = `metric-chart-plot-clip-${clipIdRaw.replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [localSmoothing, setLocalSmoothing] = useState(smoothable ? defaultSmoothing : 0);
  const smoothing = smoothingProp ?? localSmoothing;
  // Separate "display" value so the slider thumb / % label track the pointer
  // instantly (cheap state, no recompute below), while the value that actually
  // drives the EMA + Y-range recompute is committed at most once per animation
  // frame. A native <input type=range> fires an 'input' event on every pixel of
  // pointer movement during a drag — far more often than the browser can paint —
  // and each commit re-runs the EMA over every series plus the Y-range percentile
  // sort. Without this split, either the slider janks (if we committed on every
  // event) or the thumb would visually snap back on unrelated re-renders (if we
  // rAF-throttled the single controlled `value` directly).
  const [smoothingDisplay, setSmoothingDisplay] = useState(smoothing);
  const smoothingRafRef = useRef<number | null>(null);
  const pendingSmoothingRef = useRef<number | null>(null);
  const commitSmoothingRef = useRef<(v: number) => void>(() => {});
  commitSmoothingRef.current = (v: number) => {
    if (onSmoothingChange) onSmoothingChange(v); else setLocalSmoothing(v);
  };
  useEffect(() => () => { if (smoothingRafRef.current !== null) cancelAnimationFrame(smoothingRafRef.current); }, []);
  // A controlled value that did NOT come from this slider (parent reset, slot
  // restore) must move the thumb; one echoed back from our own pending commit
  // must not, or it would drag the thumb back a frame behind the pointer.
  useEffect(() => {
    if (smoothingProp !== undefined && smoothingProp !== pendingSmoothingRef.current) {
      setSmoothingDisplay(smoothingProp);
    }
  }, [smoothingProp]);
  const scheduleSmoothing = (v: number) => {
    setSmoothingDisplay(v);
    pendingSmoothingRef.current = v;
    if (smoothingRafRef.current !== null) return;
    smoothingRafRef.current = requestAnimationFrame(() => {
      smoothingRafRef.current = null;
      if (pendingSmoothingRef.current !== null) commitSmoothingRef.current(pendingSmoothingRef.current);
    });
  };
  const [logScale, setLogScale] = useState(false);
  const [legendOpen, setLegendOpen] = useState(false);
  // Series hidden via legend clicks. Hidden series are excluded from rendering
  // AND from the Y-range pooling, so the remaining series auto-rescale to fill
  // the view (lets you isolate one metric's variation instead of all forced on).
  const [localHiddenIds, setLocalHiddenIds] = useState<Set<string>>(new Set());
  const hiddenIds = hiddenIdsProp ?? localHiddenIds;
  const setHiddenIds = (next: Set<string>) => {
    if (onHiddenIdsChange) onHiddenIdsChange(next); else setLocalHiddenIds(next);
  };
  const [xRange, setXRange] = useState<{ min: number; max: number } | null>(null);
  const [brush, setBrush] = useState<{ startStep: number; endStep: number } | null>(null);
  const pointerXRef = useRef<number | null>(null);
  const [tooltip, setTooltip] = useState<{
    px: number; py: number; step: number;
    values: TooltipValue[];
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
      const pts = s.points.map((p) => ({ step: p.step, value: p.value }));
      out.set(s.id, s.noSmooth ? pts : applySmoothing(pts, smoothing));
    }
    return out;
  }, [visibleSeries, smoothing]);

  // Whether any currently-visible series wants a secondary (right-hand) axis,
  // and which color to label its ticks with (first such series' own color, so
  // the axis reads as "belonging to" that series rather than a fixed hue).
  // Band series belong to no axis: they never pool into a Y-range, never make
  // the secondary axis appear, and get their own reserved rows below the plot.
  const bandSeries = useMemo(() => visibleSeries.filter(isBand), [visibleSeries]);
  const curveSeries = useMemo(() => visibleSeries.filter((s) => !isBand(s)), [visibleSeries]);
  const hasSecondary = useMemo(() => curveSeries.some((s) => axisOf(s) === "right"), [curveSeries]);
  const secondaryColor = useMemo(() => curveSeries.find((s) => axisOf(s) === "right")?.color ?? "#38bdf8", [curveSeries]);
  const leftSeries = useMemo(() => curveSeries.filter((s) => axisOf(s) === "left"), [curveSeries]);
  const rightSeries = useMemo(() => curveSeries.filter((s) => axisOf(s) === "right"), [curveSeries]);

  // Layout. Reserve extra right margin for the secondary axis's tick labels
  // when active, and one row of bottom margin per band.
  const PAD = useMemo(() => ({
    ...BASE_PAD,
    right: hasSecondary ? SECONDARY_AXIS_RIGHT_PAD : BASE_PAD.right,
    bottom: BASE_PAD.bottom + (bandSeries.length ? BAND_TOP_GAP : 0) + BAND_ROW_H * bandSeries.length,
  }), [hasSecondary, bandSeries.length]);
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

  // Precomputed once per (series, zoom) rather than per pointermove: the
  // tooltip needs each series' sample spacing on every mouse move, and the
  // median is a sort.
  const stepGaps = useMemo(() => {
    const out = new Map<string, number | null>();
    for (const [id, pts] of visibleRaw) out.set(id, medianStepGap(pts));
    return out;
  }, [visibleRaw]);

  // Y scale (pool all visible series' SMOOTHED values through the percentile
  // calc; only rawRange series are forced fully visible via raw min/max).
  //
  // Design history / trade-off (do not re-litigate without re-reading this):
  //  - Pre-82e76105: dashed aux-loss series (Recon/Gen region/Known region/Seam
  //    on the Loss chart) were forced into `mustInclude` using their RAW spike
  //    values — that pinned the axis top to the largest raw spike and crushed
  //    the primary Loss trend into the bottom ~15% of the frame.
  //  - A same-day follow-up tried excluding dashed series from the pool
  //    entirely (axis = primary series only). That over-corrected: with the
  //    axis tight around Loss alone, the aux series (living on a visibly
  //    higher scale, e.g. ~0.1-0.19 vs Loss's ~0.02-0.07) clip almost
  //    entirely out of frame, and their un-smoothed noise (see the raw ghost
  //    line below) repeatedly pokes across the top edge — a dense noisy band
  //    that buries the Loss curve it was meant to protect.
  //  - Current: pool every visible series' SMOOTHED values together (dashed
  //    included, exactly like solid series) — matches what 82e76105 shipped.
  //    This keeps the axis wide enough that aux series mostly stay in frame
  //    (so they don't fragment into a noisy clipped band), while still
  //    tracking the smoothing slider (unlike raw pooling) since every input
  //    to the percentile calc is smoothed once smoothing > 0. It won't track
  //    the slider as tightly as a single-series chart (Tagger) — that's the
  //    accepted trade-off for a 5-series chart with heterogeneous scales.
  //
  // Log-domain range (primary only in the legacy path): a robust (p5-p95,
  // log-space padded) range over the POSITIVE subset of the pooled data — see
  // robustPositiveLogRange()'s history note. Two prior guards were both wrong
  // in different ways:
  //  - `rawYRange.min > 0`: a single zero/unrecorded point anywhere pinned the
  //    LINEAR range's min to 0 (via yMinFloor clamping), so log mode never
  //    activated at all — the button looked pressable but did nothing.
  //  - absolute min positive value as the log floor: activated log mode, but
  //    one low transient (e.g. nearzero warm-up point right after a resume)
  //    pinned the axis low and squeezed the steady-state data into a thin
  //    band at the top.
  const leftAxis = useAxisScale({
    seriesForAxis: leftSeries,
    visibleSmoothed, visibleRaw, smoothing,
    yMinFloor, bounded,
    allowLog: axes ? axes.left.scale === "log" : allowLogScale,
    logOn: axes ? true : logScale,
    config: axes?.left,
    inactiveRange: null,
    padTop: PAD.top, chartH,
  });
  // Legacy right axis: independent linear auto-range over its own series only,
  // no floor / no bounded / no log.
  const rightAxis = useAxisScale({
    seriesForAxis: rightSeries,
    visibleSmoothed, visibleRaw, smoothing,
    yMinFloor: -Infinity, bounded: false,
    allowLog: axes?.right ? axes.right.scale === "log" : false,
    logOn: true,
    config: axes?.right,
    inactiveRange: INACTIVE_RIGHT_RANGE,
    padTop: PAD.top, chartH,
  });

  const toY = leftAxis.toY;
  const toY2 = rightAxis.toY;

  const buildPath = (pts: Pt[], toYFn: (v: number) => number = toY) =>
    pts
      .filter((p) => inX(p.step))
      .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.step).toFixed(1)} ${toYFn(p.value).toFixed(1)}`)
      .join(" ");

  // Shared numeric formatting; each axis unwraps its own domain (log10 or
  // identity) via fromDomain before formatting.
  const formatReal = (real: number) => {
    if (Math.abs(real) >= 100) return real.toFixed(0);
    if (Math.abs(real) >= 1) return real.toFixed(2);
    if (Math.abs(real) >= 0.01) return real.toFixed(3);
    if (real === 0) return "0";
    return real.toExponential(1);
  };
  const formatY = (v: number) => formatReal(leftAxis.fromDomain(v));
  const formatY2 = (v: number) => formatReal(rightAxis.fromDomain(v));
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
  const yTickValues = leftAxis.tickValues;
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
    const values: TooltipValue[] = [];
    // Tracked in PIXEL space (not raw value) so it stays correct regardless of
    // which axis (primary/log or secondary/linear) the last-matched series
    // belongs to -- toY and toY2 have independent domains.
    let anchorPy: number | null = null;
    for (const s of visibleSeries) {
      const pts = visibleRaw.get(s.id) ?? [];
      const sm = visibleSmoothed.get(s.id) ?? [];
      if (pts.length === 0) continue;
      // A series sampled every 500 steps has NO point within the 16px global
      // tolerance at most hover positions, so it would silently never appear in
      // the tooltip. Half its own sample spacing is the widest tolerance that
      // still maps each hover to exactly one of its samples.
      const gap = stepGaps.get(s.id) ?? null;
      const tol = Math.max(stepTolerance, gap !== null ? gap / 2 : 0);
      if (targetStep < pts[0].step - tol || targetStep > pts[pts.length - 1].step + tol) continue;
      let idx = 0, dist = Math.abs(pts[0].step - targetStep);
      for (let i = 1; i < pts.length; i++) {
        const d = Math.abs(pts[i].step - targetStep);
        if (d < dist) { dist = d; idx = i; }
      }
      if (dist > tol) continue;
      // A noSmooth series' "smoothed" points ARE its raw points, so surfacing
      // one would read as "the EMA equals the sample" for exactly the series
      // where an EMA was never computed.
      const smv = s.noSmooth ? null : (sm[idx]?.value ?? null);
      values.push({
        id: s.id, label: s.label, color: s.color,
        value: pts[idx].value, smoothValue: smv,
        realStep: dist > stepTolerance ? pts[idx].step : null,
      });
      // A band has no Y-axis, so its value cannot place the crosshair dot.
      if (isBand(s)) continue;
      const av = (smoothing > 0 && smv !== null) ? smv : pts[idx].value;
      anchorPy = axisOf(s) === "right" ? toY2(av) : toY(av);
    }
    if (values.length === 0) { setTooltip(null); return; }
    // Bands-only selection: no curve to anchor the dot to, so pin it to the top
    // of the plot rather than dropping the tooltip entirely.
    setTooltip({ px: toX(targetStep), py: anchorPy ?? PAD.top, step: Math.round(targetStep), values });
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
      {/* h-6 and truncate together: a wrapping title made this row two lines
          tall, so two panes of the same `height` drew their plots at different
          y and stopped lining up. */}
      <div className="flex items-center justify-between mb-1 h-6">
        <div className="text-sm font-medium text-gray-300 truncate shrink min-w-0 mr-2" title={title}>{title}</div>
        <div className="flex items-center gap-2 shrink-0">
          {headerExtra}
          {smoothable && (
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-gray-500">Smooth</span>
              <input
                type="range" min={0} max={1} step={0.005}
                value={smoothingToPosition(smoothingDisplay)}
                onChange={(e) => scheduleSmoothing(positionToSmoothing(parseFloat(e.target.value)))}
                className="w-20 h-1 cursor-pointer"
                title="EMA window, log-spaced in points"
              />
              <span
                className="text-[10px] text-gray-400 font-mono w-10 text-right"
                title={`EMA factor ${smoothingDisplay.toFixed(4)}`}
              >
                {smoothingDisplay <= 0 ? "off" : `${Math.round(smoothingToWindow(smoothingDisplay))}pt`}
              </span>
            </div>
          )}
          {headerTrailing}
          {/* F8: `axes` callers own their per-axis log state and drive it through
              axes.{left,right}.scale, which is why useAxisScale passes logOn:true
              in that path. The legacy single-axis button would toggle a state
              that path ignores, so it is suppressed rather than left dead. */}
          {allowLogScale && !axes && (
            <button
              onClick={() => setLogScale((v) => !v)}
              disabled={!leftAxis.logAvailable}
              className={`text-[10px] px-1.5 py-0.5 rounded transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${logScale ? "bg-blue-700 text-blue-100" : "bg-gray-700 hover:bg-gray-600 text-gray-300"}`}
              title={!leftAxis.logAvailable ? "No positive values to log-scale" : "Toggle log Y scale"}
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
                onClick={() => {
                  const next = new Set(hiddenIds);
                  if (next.has(s.id)) next.delete(s.id); else next.add(s.id);
                  setHiddenIds(next);
                }}
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
            {/* Plot-area clip rect: series lines are wrapped in a <g clipPath=...>
                using this so a value beyond yMin/yMax stops exactly at the plot
                boundary on BOTH edges. Without it, only the SVG's own edges clip
                (the root <svg>'s default `overflow: hidden`) — a value below
                yMin renders into the PAD.bottom strip (X-axis label margin,
                still inside the SVG) and leaks visibly past the plot frame,
                while a value above yMax has much less PAD.top margin to leak
                into before hitting the SVG's own top edge, so the two edges
                looked inconsistently clipped. */}
            <defs>
              <clipPath id={clipId}>
                <rect x={PAD.left} y={PAD.top} width={chartW} height={chartH} />
              </clipPath>
            </defs>
            {/* Y grid + ticks (primary axis) */}
            {yTickValues.map((v, i) => (
              <g key={`y${i}`}>
                <line x1={PAD.left} y1={toY(leftAxis.fromDomain(v))} x2={width - PAD.right} y2={toY(leftAxis.fromDomain(v))} stroke="#374151" strokeWidth={1} />
                <text x={PAD.left - 4} y={toY(leftAxis.fromDomain(v)) + 3} textAnchor="end" fontSize={9} fill="#6b7280">{formatY(v)}</text>
              </g>
            ))}
            {/* Zero gridline for an axis whose declared domain spans zero (a
                signed-correlation axis, say) — the percentile-clipped auto
                axes never guarantee 0 sits on a tick row. */}
            {leftAxis.zeroY !== null && (
              <line x1={PAD.left} y1={leftAxis.zeroY} x2={width - PAD.right} y2={leftAxis.zeroY} stroke="#4b5563" strokeWidth={1} strokeDasharray="2 2" />
            )}
            {hasSecondary && rightAxis.zeroY !== null && (
              <line x1={PAD.left} y1={rightAxis.zeroY} x2={width - PAD.right} y2={rightAxis.zeroY} stroke="#4b5563" strokeWidth={1} strokeDasharray="2 2" />
            )}
            {/* Secondary axis tick labels (right side). No separate gridlines --
                toY and toY2 both map their own [min,max] linearly onto the same
                [PAD.top, PAD.top+chartH] pixel span, so the canonical top/mid/
                bottom rows already line up with the primary axis's gridlines;
                only the displayed VALUE differs per axis. */}
            {hasSecondary && rightAxis.tickValues.map((v, i) => (
              <text key={`y2-${i}`} x={width - PAD.right + 4} y={PAD.top + (i * chartH) / 2 + 3} textAnchor="start" fontSize={9} fill={secondaryColor}>
                {formatY2(v)}
              </text>
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

            {/* Series lines. When smoothing is on, the raw values of PRIMARY
                (non-dashed) series are drawn faintly behind the smoothed line
                so the actual noise is still visible. Dashed aux-loss overlays
                (Recon/Gen region/Known region/Seam on the Loss chart) skip the
                raw ghost: they typically live on a different scale than the
                primary series, so their unsmoothed noise repeatedly pokes
                across the Y-range boundary and reads as a dense noisy band
                rather than a legible line — the smoothed dashed line alone is
                enough context for an overlay metric. */}
            <g clipPath={`url(#${clipId})`}>
              {smoothing > 0 && curveSeries.filter((s) => !s.dashed && !s.noSmooth && s.renderMode !== "markers" && axisOf(s) !== "right").map((s) => (
                <path key={`${s.id}-raw`} d={buildPath(s.points.map((p) => ({ step: p.step, value: p.value })))}
                  fill="none" stroke={s.color} strokeWidth={1}
                  opacity={0.22} />
              ))}
              {curveSeries.map((s) => {
                const pts = (smoothing > 0 ? (smoothedSeries.get(s.id) ?? []) : s.points.map((p) => ({ step: p.step, value: p.value })));
                const yFn = axisOf(s) === "right" ? toY2 : toY;
                if (s.renderMode === "markers") {
                  // A dot per sample: a handful of widely-spaced probes drawn as
                  // a polyline reads as a straight line through the dense noise
                  // around it, hiding where the samples actually are.
                  const vis = pts.filter((p) => inX(p.step));
                  return (
                    <g key={s.id}>
                      {vis.length > 1 && (
                        <path d={buildPath(pts, yFn)} fill="none" stroke={s.color} strokeWidth={1}
                          strokeDasharray={s.dashed ? "4 3" : undefined} opacity={0.55} />
                      )}
                      {/* One sample has no line to draw, and a lone 2.5px dot is
                          easy to miss — give it a tick to sit on. */}
                      {vis.length === 1 && (
                        <line x1={toX(vis[0].step) - 6} y1={yFn(vis[0].value)} x2={toX(vis[0].step) + 6} y2={yFn(vis[0].value)}
                          stroke={s.color} strokeWidth={1} opacity={0.55} />
                      )}
                      {vis.map((p, i) => (
                        <circle key={i} cx={toX(p.step)} cy={yFn(p.value)} r={2.5} fill={s.color} opacity={0.95} />
                      ))}
                    </g>
                  );
                }
                return (
                  <path key={s.id} d={buildPath(pts, yFn)} fill="none" stroke={s.color} strokeWidth={1.5}
                    strokeDasharray={s.dashed ? "4 3" : undefined} opacity={0.95} />
                );
              })}
            </g>

            {/* State bands. Outside the plot clip (their reserved rows are in
                PAD.bottom), so the run rects are clamped to the plot width by
                hand instead. */}
            {bandSeries.map((s, bi) => {
              const yTop = PAD.top + chartH + BAND_TOP_GAP + bi * BAND_ROW_H;
              const xLo = PAD.left, xHi = PAD.left + chartW;
              const rects: { x: number; w: number }[] = [];
              const push = (a: number, b: number) => {
                const xa = toX(a), xb = toX(b);
                if (xb < xLo || xa > xHi) return;
                const ca = Math.max(xLo, xa), cb = Math.min(xHi, xb);
                rects.push({ x: ca, w: Math.max(1.5, cb - ca) });
              };
              // A run of consecutive "on" samples coalesces into ONE rect; a
              // sample holds until the next one, so a run ends at the step of
              // the first "off" sample after it. Fractional values (a partial
              // batch) read as on above 0.5 — the band answers "was this step
              // mostly in this state", which is what the 0/1 curve failed to.
              let start: number | null = null;
              for (let i = 0; i < s.points.length; i++) {
                const on = s.points[i].value >= 0.5;
                if (on && start === null) start = s.points[i].step;
                else if (!on && start !== null) { push(start, s.points[i].step); start = null; }
              }
              if (start !== null) {
                // Hold the trailing run for one more sample interval: closing it
                // at its own last step made the CURRENT state a 1.5px sliver,
                // while every interior run got a full hold width.
                const last = s.points[s.points.length - 1].step;
                push(start, last + (stepGaps.get(s.id) ?? 0));
              }
              return (
                <g key={`band-${s.id}`}>
                  <rect x={xLo} y={yTop} width={chartW} height={BAND_H} fill="#1f2937" />
                  {rects.map((r, i) => (
                    <rect key={i} x={r.x} y={yTop} width={r.w} height={BAND_H} fill={s.color} opacity={0.8} />
                  ))}
                  <text x={PAD.left - 4} y={yTop + BAND_H - 1} textAnchor="end" fontSize={8} fill="#6b7280">
                    {s.label.length > 9 ? `${s.label.slice(0, 8)}…` : s.label}
                  </text>
                </g>
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
                {v.realStep !== null && (
                  <span className="font-mono text-gray-500">@{v.realStep}</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
