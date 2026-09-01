"use client";

import { RefreshCw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MetricPoint } from "@/utils/api";
import SharedMetricChart, {
  type AxisConfig, type ChartSeries, type MetricRangePolicy,
} from "./SharedMetricChart";
import { useTrainingMetrics } from "./TrainingMetricsContext";
import MetricSeriesPicker from "./MetricSeriesPicker";
import {
  PRESETS, assignAxes, describeSeries, isSparse, mergedFixedDomain, resolvePreset,
  type MetricDescriptor, type MetricPreset,
} from "./metricCatalog";

/**
 * One training-metrics pane: a preset (or a hand-picked set) of this run's
 * series on at most two semantically-compatible Y-axes.
 *
 * Two panes are rendered side by side. That is deliberate: a third scale group
 * is REFUSED rather than crammed onto a shared axis, and the second pane is the
 * escape hatch that makes the refusal workable.
 */

const EMPTY: MetricPoint[] = [];
const AXIS_LOG_DISABLED = "Log needs an auto-ranged axis whose values are all > 0";

interface PersistedState {
  preset?: string;
  /** Written only in custom mode; a preset re-resolves against the run. */
  selected?: string[];
  hidden?: string[];
  log?: { left?: boolean; right?: boolean };
  smoothing?: number;
  collapsed?: string[];
}

const isBandFamily = (d: MetricDescriptor) => d.family === "binary_indicator";

/** The domain policy for one axis. A merged (`a+b`) group takes the union of
 *  its members' declared domains; a single group takes its members' declared
 *  range, falling back to an auto range with the loosest floor they agree on. */
export function axisPolicy(
  axisDescs: MetricDescriptor[], group: string | null, members: MetricDescriptor[],
): MetricRangePolicy {
  if (group && group.includes("+")) {
    const d = mergedFixedDomain(axisDescs, group);
    if (d) return { kind: "fixed", min: d.min, max: d.max };
  }
  if (members.length && members.every((m) => m.range.kind === "fixed")) {
    let min = Infinity, max = -Infinity;
    for (const m of members) {
      if (m.range.kind !== "fixed") continue;
      min = Math.min(min, m.range.min);
      max = Math.max(max, m.range.max);
    }
    return { kind: "fixed", min, max };
  }
  let floor: number | undefined;
  for (const m of members) {
    const f = m.range.kind === "auto" ? m.range.floor : undefined;
    if (f === undefined) return { kind: "auto" };
    floor = floor === undefined ? f : Math.min(floor, f);
  }
  return floor === undefined ? { kind: "auto" } : { kind: "auto", floor };
}

export default function TrainingMetricsChart({
  slot, defaultPreset, height = 160,
}: { slot: "a" | "b"; defaultPreset: string; height?: number }) {
  const { seriesByKey, defs, epochBoundaries, resumeMarkers, error, refresh, loading} = useTrainingMetrics();

  const [preset, setPreset] = useState(defaultPreset);
  const [custom, setCustom] = useState<string[] | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [log, setLog] = useState<{ left: boolean; right: boolean }>({ left: false, right: false });
  const [smoothing, setSmoothing] = useState(0.9);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [swapped, setSwapped] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const pickerAnchor = useRef<HTMLDivElement>(null);

  const storageKey = `sushi.trainingChart.v1.${slot}`;

  // Restore after mount, not in the state initializer: this renders on the
  // server too, and reading localStorage during the first render would make the
  // client's markup disagree with it.
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw) {
        const s: PersistedState = JSON.parse(raw);
        if (s.preset) setPreset(s.preset);
        if (s.selected) setCustom(s.selected);
        if (s.hidden) setHidden(new Set(s.hidden));
        if (s.log) setLog({ left: !!s.log.left, right: !!s.log.right });
        if (typeof s.smoothing === "number") setSmoothing(s.smoothing);
        if (s.collapsed) setCollapsed(new Set(s.collapsed));
      }
    } catch { /* corrupt or unavailable storage: keep the defaults */ }
    setLoaded(true);
  }, [storageKey]);

  // `loaded` is state, not a ref, so this effect cannot run with the restored
  // values still pending and write the defaults back over them.
  useEffect(() => {
    if (!loaded) return;
    const s: PersistedState = {
      preset,
      selected: custom ?? undefined,
      hidden: [...hidden],
      log,
      smoothing,
      collapsed: [...collapsed],
    };
    try { window.localStorage.setItem(storageKey, JSON.stringify(s)); } catch { /* ignore */ }
  }, [loaded, storageKey, preset, custom, hidden, log, smoothing, collapsed]);

  // Every series this run actually has, described.
  const inventory = useMemo<MetricDescriptor[]>(() => (
    Object.entries(seriesByKey)
      .filter(([, pts]) => pts.length > 0)
      .map(([key]) => describeSeries(key, defs[key]))
      .sort((a, b) => a.label.localeCompare(b.label))
  ), [seriesByKey, defs]);
  const byKey = useMemo(() => new Map(inventory.map((d) => [d.key, d])), [inventory]);

  const presetMembers = useMemo(() => {
    const m = new Map<string, MetricDescriptor[]>();
    for (const p of PRESETS) m.set(p.id, resolvePreset(p, inventory));
    return m;
  }, [inventory]);

  // Anchors are context, not content: a preset that resolved to nothing but its
  // anchors (timestep-gradcos on a run with no grad_cos_t_* yet) has no answer
  // to give for this run and is offered disabled, like one resolving to nothing.
  const presetUsable = useCallback((p: MetricPreset) => {
    const members = presetMembers.get(p.id) ?? [];
    if (!members.some((d) => !p.anchors.includes(d.key))) return false;
    // A preset named for a distinction the run does not draw is worse than
    // absent: cfg-loss-split resolves to exactly loss-overview's membership on
    // a run with cfg_uncond_drop_rate == 0. The deleted CfgConditionFilter hid
    // itself in that case; this is the same rule.
    if (p.requires) return p.requires.some((k) => byKey.has(k));
    return true;
  }, [presetMembers, byKey]);

  const activePreset = useMemo(() => PRESETS.find((p) => p.id === preset), [preset]);

  // A pane restored onto a preset this run cannot answer falls back rather than
  // rendering permanently empty.
  useEffect(() => {
    if (!loaded || custom || inventory.length === 0) return;
    if (activePreset && presetUsable(activePreset)) return;
    const fb = PRESETS.find((p) => p.id === defaultPreset && presetUsable(p)) ?? PRESETS.find(presetUsable);
    if (fb && fb.id !== preset) setPreset(fb.id);
  }, [loaded, custom, inventory.length, activePreset, presetUsable, defaultPreset, preset]);

  const selectedDescs = useMemo<MetricDescriptor[]>(() => {
    // Intersect with the run's inventory either way: a persisted custom set may
    // name a series this run never emitted.
    if (custom) return custom.map((k) => byKey.get(k)).filter((d): d is MetricDescriptor => !!d);
    return activePreset ? (presetMembers.get(activePreset.id) ?? []) : [];
  }, [custom, byKey, activePreset, presetMembers]);
  const selectedSet = useMemo(() => new Set(selectedDescs.map((d) => d.key)), [selectedDescs]);

  // Bands own no axis, so they are excluded from the assignment entirely — a
  // 0/1 indicator must never cost the chart one of its two scale slots.
  const axisDescs = useMemo(() => selectedDescs.filter((d) => !isBandFamily(d)), [selectedDescs]);
  const assignment = useMemo(
    () => assignAxes(axisDescs, activePreset, swapped),
    [axisDescs, activePreset, swapped],
  );

  const leftMembers = useMemo(() => axisDescs.filter((d) => assignment.byKey[d.key] === "left"), [axisDescs, assignment]);
  const rightMembers = useMemo(() => axisDescs.filter((d) => assignment.byKey[d.key] === "right"), [axisDescs, assignment]);

  const leftPolicy = useMemo(() => axisPolicy(axisDescs, assignment.left, leftMembers), [axisDescs, assignment.left, leftMembers]);
  const rightPolicy = useMemo(() => axisPolicy(axisDescs, assignment.right, rightMembers), [axisDescs, assignment.right, rightMembers]);

  // Log is only meaningful for an auto-ranged axis every one of whose values is
  // strictly positive. A declared [-1,1] axis is never log-able, and one zero
  // sample would otherwise produce a silently-clamped floor.
  const logEligible = useCallback((policy: MetricRangePolicy, members: MetricDescriptor[]) => {
    if (policy.kind !== "auto" || members.length === 0) return false;
    for (const m of members) {
      for (const p of seriesByKey[m.key] ?? EMPTY) if (!(p.value > 0)) return false;
    }
    return true;
  }, [seriesByKey]);
  const leftLogOk = useMemo(() => logEligible(leftPolicy, leftMembers), [logEligible, leftPolicy, leftMembers]);
  const rightLogOk = useMemo(() => logEligible(rightPolicy, rightMembers), [logEligible, rightPolicy, rightMembers]);

  const axes = useMemo<{ left: AxisConfig; right?: AxisConfig }>(() => ({
    left: { scale: log.left && leftLogOk ? "log" : "linear", range: leftPolicy },
    right: assignment.right
      ? { scale: log.right && rightLogOk ? "log" : "linear", range: rightPolicy }
      : undefined,
  }), [log.left, log.right, leftLogOk, rightLogOk, leftPolicy, rightPolicy, assignment.right]);

  const chartSeries = useMemo<ChartSeries[]>(() => {
    const withPoints = selectedDescs.map((d) => ({ d, points: seriesByKey[d.key] ?? EMPTY }));
    return withPoints
      .filter(({ d }) => !assignment.refusedKeys.includes(d.key))
      .map(({ d, points }): ChartSeries => {
        const band = isBandFamily(d);
        const sparse = !band && isSparse({ points, sampling: d.sampling }, withPoints);
        return {
          id: d.key,
          label: d.label,
          color: d.color,
          points,
          dashed: d.dashed,
          axis: band ? undefined : assignment.byKey[d.key],
          renderMode: band ? "band" : sparse ? "markers" : "line",
          // A 0/1 state and a periodic probe both read as their own values; an
          // EMA of them is a lagged average of something that was never a curve.
          noSmooth: band || sparse,
        };
      });
  }, [selectedDescs, seriesByKey, assignment]);

  // Which unselected series cannot be added without pushing a scale group off
  // the chart. Computed by trial assignment, so the reason shown is the real one.
  const blocked = useMemo(() => {
    const m = new Map<string, string>();
    const base = assignment.refusedKeys.length;
    for (const d of inventory) {
      if (selectedSet.has(d.key) || isBandFamily(d)) continue;
      const trial = assignAxes([...axisDescs, d], activePreset, swapped);
      if (trial.refusedKeys.length > base) {
        m.set(d.key, trial.refusalMessage ?? "This chart already uses both of its axes.");
      }
    }
    return m;
  }, [inventory, selectedSet, axisDescs, activePreset, swapped, assignment.refusedKeys.length]);

  // Bulk selection has to be checked incrementally: `blocked` above trial-adds
  // ONE series, so with a single group in use nothing is individually blocked
  // and a whole-family or All click could select a set that then refuses 23 of
  // its own members -- ticked in the picker, absent from the chart, unexplained.
  const addable = useCallback((current: Set<string>, candidates: MetricDescriptor[]) => {
    const next = new Set(current);
    let descs = inventory.filter((d) => next.has(d.key) && !isBandFamily(d));
    for (const d of candidates) {
      if (next.has(d.key)) continue;
      if (isBandFamily(d)) { next.add(d.key); continue; }   // bands take no axis
      const trial = assignAxes([...descs, d], activePreset, swapped);
      if (trial.refusedKeys.length) continue;
      next.add(d.key);
      descs = [...descs, d];
    }
    return next;
  }, [inventory, activePreset, swapped]);

  const axisTag = useCallback((key: string): "L" | "R" | "▭" | null => {
    const d = byKey.get(key);
    if (!d) return null;
    if (isBandFamily(d)) return "▭";
    const side = assignment.byKey[key];
    return side === "left" ? "L" : side === "right" ? "R" : null;
  }, [byKey, assignment]);

  const setSelected = useCallback((next: Set<string>) => {
    // Any hand edit leaves preset mode; the preset stays selectable to return to.
    setCustom(inventory.filter((d) => next.has(d.key)).map((d) => d.key));
  }, [inventory]);

  // Close the picker on an outside press.
  useEffect(() => {
    if (!pickerOpen) return;
    const onDown = (e: PointerEvent) => {
      if (!pickerAnchor.current?.contains(e.target as Node)) setPickerOpen(false);
    };
    window.addEventListener("pointerdown", onDown);
    return () => window.removeEventListener("pointerdown", onDown);
  }, [pickerOpen]);

  const title = custom ? "Custom" : (activePreset?.name ?? "Metrics");

  const header = (
    <>
      <select
        value={custom ? "__custom" : preset}
        onChange={(e) => {
          const v = e.target.value;
          if (v === "__custom") return;
          setCustom(null);
          setPreset(v);
          setSwapped(false);
        }}
        title={activePreset?.question}
        className="text-[10px] px-1 py-0.5 rounded bg-gray-700 text-gray-300 border border-gray-600 max-w-[9rem]"
      >
        {custom && <option value="__custom">Custom</option>}
        {PRESETS.map((p) => {
          const usable = presetUsable(p);
          return (
            <option key={p.id} value={p.id} disabled={!usable}
              title={usable ? p.question : "No matching metrics in this run."}>
              {p.name}{usable ? "" : " —"}
            </option>
          );
        })}
      </select>
      <div className="relative" ref={pickerAnchor}>
        <button
          onClick={() => setPickerOpen((v) => !v)}
          className={`text-[10px] px-1.5 py-0.5 rounded transition-colors ${pickerOpen ? "bg-blue-700 text-blue-100" : "bg-gray-700 hover:bg-gray-600 text-gray-300"}`}
          title="Choose which series are on this chart"
        >
          Series {selectedDescs.length}/{inventory.length} ▾
        </button>
        {pickerOpen && (
          <MetricSeriesPicker
            inventory={inventory}
            selected={selectedSet}
            onChange={setSelected}
            addable={addable}
            blocked={blocked}
            axisTag={axisTag}
            collapsed={collapsed}
            onCollapsedChange={setCollapsed}
            canSwap={assignment.canSwap}
            onSwap={() => setSwapped((v) => !v)}
          />
        )}
      </div>
      {/* Both deleted charts had one; the shared provider makes it one fetch for
          every pane, so either pane's button refreshes all of them. */}
      <button
        onClick={refresh}
        disabled={loading}
        title="Refresh"
        className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 disabled:opacity-50"
      >
        <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
      </button>
    </>
  );

  const logButton = (side: "left" | "right", ok: boolean, active: boolean) => (
    <button
      key={side}
      onClick={() => setLog((v) => (side === "left" ? { ...v, left: !v.left } : { ...v, right: !v.right }))}
      disabled={!ok}
      className={`text-[10px] px-1.5 py-0.5 rounded transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
        active && ok ? "bg-blue-700 text-blue-100" : "bg-gray-700 hover:bg-gray-600 text-gray-300"
      }`}
      title={ok ? `Toggle log scale on the ${side} axis` : AXIS_LOG_DISABLED}
    >log {side === "left" ? "L" : "R"}</button>
  );

  return (
    <div className="min-w-0">
      {error && <div className="text-xs text-red-400 mb-1">{error}</div>}
      {assignment.refusalMessage && (
        <div className="text-[10px] text-amber-500/90 mb-1 leading-tight">{assignment.refusalMessage}</div>
      )}
      <SharedMetricChart
        title={title}
        series={chartSeries}
        height={height}
        axes={axes}
        headerExtra={header}
        headerTrailing={
          <>
            {logButton("left", leftLogOk, log.left)}
            {assignment.right && logButton("right", rightLogOk, log.right)}
          </>
        }
        hiddenIds={hidden}
        onHiddenIdsChange={setHidden}
        smoothing={smoothing}
        onSmoothingChange={setSmoothing}
        epochBoundaries={epochBoundaries}
        resumeMarkers={resumeMarkers}
      />
    </div>
  );
}
