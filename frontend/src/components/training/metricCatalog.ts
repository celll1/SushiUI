/**
 * Metric semantics for the training charts: what each series IS, which series
 * may share a Y-axis, and which preset answers which question.
 *
 * The backend registry (core/training/metric_registry.py) is the source of
 * truth; everything here is either a description of a hardcoded DB column or a
 * fallback for a metric whose author has not annotated it yet. Pure module — no
 * React, no I/O.
 */

import type { MetricFamily, MetricRange, MetricSeriesDef } from "@/utils/api";

export type MetricSampling = "dense" | "periodic" | "event";
export type AxisSide = "left" | "right";

export interface MetricDescriptor {
  key: string;
  label: string;
  color: string;
  dashed: boolean;
  family: MetricFamily;
  /** Two series may share an axis iff their scale groups are equal. */
  scaleGroup: string;
  range: MetricRange;
  sampling: MetricSampling;
}

const AUTO: MetricRange = { kind: "auto" };
const AUTO_0: MetricRange = { kind: "auto", floor: 0 };
const UNIT: MetricRange = { kind: "fixed", min: 0, max: 1 };
const SIGNED_UNIT: MetricRange = { kind: "fixed", min: -1, max: 1 };

// Deterministic color for a bespoke metric with no registry def, so an unknown
// series still renders with a stable (per-name) hue instead of colliding.
const FALLBACK_PALETTE = ["#f59e0b", "#a78bfa", "#f472b6", "#22d3ee", "#a3e635", "#fb923c", "#e879f9"];
export function fallbackColor(key: string): string {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  return FALLBACK_PALETTE[h % FALLBACK_PALETTE.length];
}

/** The dedicated DB columns, which are not registry entries. These key names
 *  are already hardcoded in backend/database/models.py, so naming them here
 *  adds no new staleness. Colours match what the pre-redesign charts used. */
export const BUILTIN_DESCRIPTORS: Record<string, Omit<MetricDescriptor, "key">> = {
  loss: { label: "Loss", color: "#60a5fa", dashed: false, family: "loss", scaleGroup: "loss_scale", range: AUTO_0, sampling: "dense" },
  recon: { label: "Recon", color: "#34d399", dashed: true, family: "loss", scaleGroup: "loss_scale", range: AUTO_0, sampling: "dense" },
  grad_norm: { label: "Grad norm (total)", color: "#60a5fa", dashed: false, family: "gradient_norm", scaleGroup: "gradient_norm", range: AUTO_0, sampling: "dense" },
  grad_norm_unet: { label: "Grad norm (U-Net/DiT)", color: "#34d399", dashed: false, family: "gradient_norm", scaleGroup: "gradient_norm", range: AUTO_0, sampling: "dense" },
  grad_norm_text_encoder: { label: "Grad norm (TE)", color: "#f472b6", dashed: false, family: "gradient_norm", scaleGroup: "gradient_norm", range: AUTO_0, sampling: "dense" },
  grad_norm_text_encoder_1: { label: "Grad norm (TE1)", color: "#a78bfa", dashed: false, family: "gradient_norm", scaleGroup: "gradient_norm", range: AUTO_0, sampling: "dense" },
  grad_norm_text_encoder_2: { label: "Grad norm (TE2)", color: "#facc15", dashed: false, family: "gradient_norm", scaleGroup: "gradient_norm", range: AUTO_0, sampling: "dense" },
  grad_norm_vision_encoder: { label: "Grad norm (VE)", color: "#22d3ee", dashed: false, family: "gradient_norm", scaleGroup: "gradient_norm", range: AUTO_0, sampling: "dense" },
};

/** Key-shape fallbacks for an unannotated metric, in priority order. `_cos$`
 *  deliberately precedes `^cfg_guidance` so cfg_guidance_cos reads as a signed
 *  correlation rather than a 0..1 diagnostic. */
const HEURISTICS: { test: RegExp; family: MetricFamily; scaleGroup: string; range: MetricRange }[] = [
  { test: /^lr(_|$)/, family: "learning_rate", scaleGroup: "learning_rate", range: AUTO_0 },
  { test: /^gnorm|^grad_norm/, family: "gradient_norm", scaleGroup: "gradient_norm", range: AUTO_0 },
  { test: /^grad_cos|_cos$/, family: "signed_correlation", scaleGroup: "signed_unit", range: SIGNED_UNIT },
  { test: /_frac$|_present$|_overlap$/, family: "binary_indicator", scaleGroup: "unit_interval", range: UNIT },
  { test: /_cov$/, family: "bounded_diagnostic", scaleGroup: "unit_interval", range: UNIT },
  { test: /^cfg_guidance/, family: "bounded_diagnostic", scaleGroup: "guidance_relative", range: AUTO_0 },
  { test: /_gib$/, family: "data_volume", scaleGroup: "gibibytes", range: AUTO_0 },
  { test: /_s$/, family: "duration", scaleGroup: "seconds", range: AUTO_0 },
  { test: /skipped$|dropped$/, family: "count", scaleGroup: "counts", range: AUTO_0 },
  { test: /loss$/, family: "loss", scaleGroup: "loss_scale", range: AUTO_0 },
];

function heuristic(key: string) {
  for (const h of HEURISTICS) if (h.test.test(key)) return h;
  // Its OWN scale group, so an unknown series can never silently pool with a
  // known one.
  return { family: "other" as MetricFamily, scaleGroup: `other:${key}`, range: AUTO };
}

/**
 * Resolve one series key to a full descriptor. Order:
 *  1. BUILTIN_DESCRIPTORS
 *  2. the registry's family / scale_group / range / sampling when present
 *  3. key-shape heuristics
 *  4. a legacy `axis: "right"` with nothing else -> own group `legacy_right`,
 *     so the pre-annotation window cannot regress into the loss axis
 *  5. label / color / dashed from the registry always win for presentation
 */
export function describeSeries(key: string, def?: MetricSeriesDef): MetricDescriptor {
  const builtin = BUILTIN_DESCRIPTORS[key];
  let family: MetricFamily;
  let scaleGroup: string;
  let range: MetricRange;
  let sampling: MetricSampling;
  let label: string;
  let color: string;
  let dashed: boolean;

  if (builtin) {
    ({ family, scaleGroup, range, sampling, label, color, dashed } = builtin);
  } else {
    const h = heuristic(key);
    family = def?.family ?? h.family;
    scaleGroup = def?.scale_group ?? h.scaleGroup;
    range = def?.range ?? h.range;
    sampling = def?.sampling ?? "dense";
    if (!def?.family && !def?.scale_group && h.family === "other" && def?.axis === "right") {
      scaleGroup = "legacy_right";
    }
    label = key;
    color = fallbackColor(key);
    dashed = true;
  }

  return {
    key,
    label: def?.label ?? label,
    color: def?.color ?? color,
    dashed: def?.dashed ?? dashed,
    family, scaleGroup, range, sampling,
  };
}

/** Human name for a scale group, for the axis tags and the refusal message. */
export const SCALE_GROUP_LABELS: Record<string, string> = {
  loss_scale: "loss",
  gradient_norm: "gradient norm",
  learning_rate: "learning rate",
  unit_interval: "0..1 diagnostics",
  signed_unit: "correlations (-1..1)",
  seconds: "seconds",
  gibibytes: "GiB",
  counts: "counts",
  decibels: "dB",
  blockiness: "blockiness",
  legacy_right: "unannotated (right)",
  guidance_relative: "guidance strength",
  gibibytes_peak: "GiB (run peak)",
};

export function scaleGroupLabel(group: string): string {
  // A merged fixed-range group is named by joining its members with "+".
  if (group.includes("+")) return group.split("+").map(scaleGroupLabel).join(" / ");
  return SCALE_GROUP_LABELS[group] ?? group.replace(/^other:/, "");
}

export interface MetricPreset {
  id: string;
  name: string;
  /** The question the preset answers, shown as its tooltip. */
  question: string;
  families: MetricFamily[];
  /** Narrows membership WITHIN the named family; families absent here are taken whole. */
  restrict?: Partial<Record<MetricFamily, RegExp>>;
  /** Extra built-in keys included for context, regardless of family. */
  anchors: string[];
  /** Scale groups in the order they should claim the left then right axis.
   *  Anything past the second active group is refused (see assignAxes). */
  preferredAxes: string[];
}

export const PRESETS: MetricPreset[] = [
  {
    id: "loss-overview",
    name: "Loss overview",
    question: "Is the loss going down?",
    families: ["loss"],
    anchors: [],
    preferredAxes: ["loss_scale"],
  },
  {
    id: "cfg-loss-split",
    name: "CFG loss split",
    question: "Do the caption-free items behave differently from the conditional ones?",
    families: ["loss", "binary_indicator"],
    restrict: { binary_indicator: /^cfg_null_frac$/ },
    anchors: [],
    preferredAxes: ["loss_scale", "unit_interval"],
  },
  {
    id: "gradient-norms",
    name: "Gradient norms",
    question: "Is any component's gradient exploding or dead?",
    families: ["gradient_norm"],
    anchors: [],
    preferredAxes: ["gradient_norm"],
  },
  {
    id: "learning-rates",
    name: "Learning rates",
    question: "Is the schedule doing what it was configured to do?",
    families: ["learning_rate"],
    anchors: ["loss"],
    preferredAxes: ["learning_rate", "loss_scale"],
  },
  {
    id: "cfg-guidance",
    name: "CFG guidance",
    question: "Is the caption still changing what the model predicts?",
    families: ["bounded_diagnostic", "signed_correlation"],
    restrict: { bounded_diagnostic: /^cfg_guidance/, signed_correlation: /^cfg_guidance/ },
    // No `loss` anchor: the guidance ratio is auto-ranged and the cosine is
    // fixed [-1,1], so they are already two axes. The question -- is the model
    // still using the caption -- is answered by these two alone.
    anchors: [],
    preferredAxes: ["guidance_relative", "signed_unit"],
  },
  {
    id: "timestep-gradcos",
    name: "Timestep grad cos",
    question: "Do distant timesteps pull the gradient in opposite directions?",
    families: ["signed_correlation"],
    restrict: { signed_correlation: /^grad_cos_t_/ },
    anchors: ["loss"],
    preferredAxes: ["signed_unit", "loss_scale"],
  },
  {
    id: "runtime-memory",
    name: "Runtime & memory",
    question: "Where is the step time going, and what is it costing?",
    // `count` is deliberately NOT here: seconds + gibibytes + counts is three
    // auto-ranged groups and the third would be refused. The two that answer
    // "what is a swap costing" are time and volume; the monotone counters
    // (batches_skipped, sn_und_grad_dropped) stay reachable from the picker.
    families: ["duration", "data_volume"],
    // Per-step transfer volume only. The run-cumulative peaks are their own
    // scale group precisely because they would flatten these onto the floor.
    restrict: { data_volume: /^sn_(d2h|h2d)_gib$/ },
    anchors: [],
    preferredAxes: ["seconds", "gibibytes"],
  },
  {
    id: "validation",
    name: "Validation",
    question: "Is the held-out quality still improving?",
    families: ["validation"],
    anchors: [],
    preferredAxes: ["decibels", "blockiness"],
  },
];

/** The preset's membership among the series this run actually has. Empty means
 *  the preset is not applicable and should render disabled. */
export function resolvePreset(preset: MetricPreset, available: MetricDescriptor[]): MetricDescriptor[] {
  const out: MetricDescriptor[] = [];
  for (const d of available) {
    if (preset.anchors.includes(d.key)) { out.push(d); continue; }
    if (!preset.families.includes(d.family)) continue;
    const r = preset.restrict?.[d.family];
    if (r && !r.test(d.key)) continue;
    out.push(d);
  }
  return out;
}

export interface AxisAssignment {
  /** Scale group on each axis, or null when unused. */
  left: string | null;
  right: string | null;
  byKey: Record<string, AxisSide>;
  /** Groups beyond the second: REFUSED, not squeezed onto a shared axis and not
   *  ghost-drawn normalized. A third scale on one frame is what made the old
   *  chart unreadable; the escape hatch is the other pane. */
  refusedGroups: string[];
  refusedKeys: string[];
  refusalMessage: string | null;
  /** A `Swap axes` control is only meaningful with two groups active. */
  canSwap: boolean;
}

export function assignAxes(
  selected: MetricDescriptor[],
  preset?: MetricPreset,
  swapped = false,
): AxisAssignment {
  // Two groups whose ranges are BOTH declared `fixed` share one axis over the
  // union of their domains: [0,1] and [-1,1] coexist on [-1,1] with no rescaling
  // and nothing inferred from the data, so this does not reintroduce the
  // magnitude-clustering that was rejected -- it is as declarative as the groups
  // themselves. Without it `cfg-guidance` needs three axes for two bounded
  // diagnostics plus its loss anchor, and one of them would be refused.
  const merged = mergeBoundedGroups(selected);
  const groupOf = (d: MetricDescriptor) => merged.get(d.scaleGroup) ?? d.scaleGroup;

  const groups: string[] = [];
  for (const d of selected) { const g = groupOf(d); if (!groups.includes(g)) groups.push(g); }
  const pref = (preset?.preferredAxes ?? []).map((g) => merged.get(g) ?? g);
  // Stable sort: preferred groups first in their declared order, everything
  // else in first-appearance order.
  const rank = (g: string) => { const i = pref.indexOf(g); return i < 0 ? pref.length : i; };
  groups.sort((a, b) => rank(a) - rank(b));

  let left: string | null = groups[0] ?? null;
  let right: string | null = groups[1] ?? null;
  if (swapped && left && right) [left, right] = [right, left];

  const byKey: Record<string, AxisSide> = {};
  const refusedKeys: string[] = [];
  for (const d of selected) {
    const g = groupOf(d);
    if (g === left) byKey[d.key] = "left";
    else if (g === right) byKey[d.key] = "right";
    else refusedKeys.push(d.key);
  }
  const refusedGroups = groups.slice(2);
  return {
    left, right, byKey, refusedGroups, refusedKeys,
    refusalMessage: refusedGroups.length
      ? `Both axes are in use: ${scaleGroupLabel(left!)} (left) and ${scaleGroupLabel(right!)} (right). `
        + `Mute one of them to make room for ${refusedGroups.map(scaleGroupLabel).join(", ")}, or use the other pane.`
      : null,
    canSwap: left !== null && right !== null,
  };
}

/**
 * `scaleGroup -> merged group id` for the fixed-range groups present in
 * `selected`. All of them collapse onto one synthetic group whose axis domain is
 * the union of theirs; auto-ranged groups are never merged, since their extent
 * is a property of the data rather than a declaration.
 */
function mergeBoundedGroups(selected: MetricDescriptor[]): Map<string, string> {
  const fixed = new Map<string, { min: number; max: number }>();
  for (const d of selected) {
    if (d.range.kind !== "fixed" || fixed.has(d.scaleGroup)) continue;
    fixed.set(d.scaleGroup, { min: d.range.min, max: d.range.max });
  }
  if (fixed.size < 2) return new Map();
  // Only merge domains of comparable extent. [0,1] with [-1,1] is fine; [0,1]
  // with [0,100] would flatten the first onto the baseline, which is the very
  // readability failure the single-axis-per-scale-group rule exists to prevent.
  const spans = [...fixed.values()].map((d) => d.max - d.min).filter((x) => x > 0);
  if (spans.length && Math.max(...spans) / Math.min(...spans) > 4) return new Map();
  const id = [...fixed.keys()].sort().join("+");
  const out = new Map<string, string>();
  for (const g of fixed.keys()) out.set(g, id);
  return out;
}

/** Axis domain for a (possibly merged) fixed-range group. */
export function mergedFixedDomain(
  selected: MetricDescriptor[], group: string,
): { min: number; max: number } | null {
  let min = Infinity, max = -Infinity;
  const merged = mergeBoundedGroups(selected);
  for (const d of selected) {
    if ((merged.get(d.scaleGroup) ?? d.scaleGroup) !== group) continue;
    if (d.range.kind !== "fixed") return null;
    min = Math.min(min, d.range.min);
    max = Math.max(max, d.range.max);
  }
  return min <= max ? { min, max } : null;
}

function medianGap(points: { step: number }[]): number | null {
  if (points.length < 2) return null;
  const gaps: number[] = [];
  for (let i = 1; i < points.length; i++) {
    const g = points[i].step - points[i - 1].step;
    if (g > 0) gaps.push(g);
  }
  if (gaps.length === 0) return null;
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

/**
 * Whether a series should render as markers + a thin joining line rather than a
 * curve: too few points to read as a trend, or a step spacing far coarser than
 * the densest series on the same chart (a periodic probe drawn as a polyline
 * against a per-step loss reads as a straight line through the noise).
 *
 * Takes the series (for its declared `sampling`) rather than a bare point array
 * so the declared answer can short-circuit the measurement.
 */
export function isSparse(
  series: { points: { step: number }[]; sampling?: MetricSampling },
  allSelected: { points: { step: number }[] }[],
): boolean {
  if (series.sampling && series.sampling !== "dense") return true;
  if (series.points.length < 40) return true;
  const median = medianGap(series.points);
  if (median === null) return false;
  let densest = Infinity;
  for (const s of allSelected) {
    const g = medianGap(s.points);
    if (g !== null && g < densest) densest = g;
  }
  if (!Number.isFinite(densest) || densest <= 0) return false;
  return median > 20 * densest;
}
