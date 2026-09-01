"use client";

import type { MetricFamily } from "@/utils/api";
import type { MetricDescriptor } from "./metricCatalog";

/**
 * Series picker popover for a TrainingMetricsChart pane: every series the run
 * actually emitted, grouped by what it IS (family), each group collapsible with
 * a tri-state checkbox.
 *
 * Replaces the old implicit All/Null/Cond filter, whose semantics differed per
 * panel (the loss chart dropped the pooled loss and recon; the grad-norm chart
 * dropped every per-component norm). Here nothing is dropped behind the
 * operator's back — a series is on the chart iff its box is ticked.
 *
 * A row whose scale group would be a THIRD axis is disabled rather than
 * silently squeezed onto a shared axis or ghost-drawn normalized: a third scale
 * on one frame is what made the previous chart unreadable. The escape hatch is
 * the other pane.
 */

const FAMILY_ORDER: MetricFamily[] = [
  "loss", "gradient_norm", "learning_rate", "bounded_diagnostic",
  "signed_correlation", "binary_indicator", "duration", "data_volume",
  "count", "validation", "other",
];

const FAMILY_LABELS: Record<MetricFamily, string> = {
  loss: "Loss",
  gradient_norm: "Gradient norm",
  learning_rate: "Learning rate",
  bounded_diagnostic: "Bounded diagnostic",
  signed_correlation: "Correlation",
  binary_indicator: "State (0/1)",
  count: "Count",
  duration: "Duration",
  data_volume: "Data volume",
  validation: "Validation",
  param_change: "Param change",
  other: "Other",
};

export interface MetricSeriesPickerProps {
  /** Every non-empty series in this run. */
  inventory: MetricDescriptor[];
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
  /** Refusal-aware bulk add: returns the subset of `candidates` that can join
   *  `current` without pushing a scale group off the chart. */
  addable: (current: Set<string>, candidates: MetricDescriptor[]) => Set<string>;
  /** key -> why it cannot be added (a third scale group). Selected keys are
   *  never blocked; unticking is always allowed. */
  blocked: Map<string, string>;
  /** `L` / `R` for an axis-bound series, `▭` for one drawn as a state band. */
  axisTag: (key: string) => "L" | "R" | "▭" | null;
  collapsed: Set<string>;
  onCollapsedChange: (next: Set<string>) => void;
  canSwap: boolean;
  onSwap: () => void;
}

export default function MetricSeriesPicker({
  inventory, selected, onChange, blocked, axisTag, collapsed, onCollapsedChange, canSwap, onSwap,
}: MetricSeriesPickerProps) {
  const groups = FAMILY_ORDER
    .map((f) => ({ family: f, items: inventory.filter((d) => d.family === f) }))
    .filter((g) => g.items.length > 0);

  const toggle = (key: string) => {
    const next = new Set(selected);
    if (next.has(key)) next.delete(key); else next.add(key);
    onChange(next);
  };

  const setGroup = (items: MetricDescriptor[], on: boolean) => {
    if (on) { onChange(addable(selected, items)); return; }
    const next = new Set(selected);
    for (const d of items) next.delete(d.key);
    onChange(next);
  };

  return (
    <div
      className="absolute right-0 top-full mt-1 z-20 w-64 max-h-72 overflow-y-auto bg-gray-900 border border-gray-700 rounded shadow-lg p-1.5 text-[10px]"
      onPointerDown={(e) => e.stopPropagation()}
    >
      <div className="flex items-center gap-1 pb-1 mb-1 border-b border-gray-700">
        <button
          onClick={() => onChange(addable(new Set(), inventory))}
          className="px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300"
        >All</button>
        <button
          onClick={() => onChange(new Set())}
          className="px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300"
        >None</button>
        {canSwap && (
          <button
            onClick={onSwap}
            className="ml-auto px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300"
            title="Swap which scale group owns the left and right axis"
          >⇄ Swap axes</button>
        )}
      </div>

      {groups.map((g) => {
        const on = g.items.filter((d) => selected.has(d.key)).length;
        const all = on === g.items.length;
        const some = on > 0 && !all;
        const isCollapsed = collapsed.has(g.family);
        return (
          <div key={g.family} className="mb-1">
            <div className="flex items-center gap-1 px-1 py-0.5 text-gray-400">
              <input
                type="checkbox"
                checked={all}
                ref={(el) => { if (el) el.indeterminate = some; }}
                onChange={() => setGroup(g.items, !all)}
                className="w-3 h-3 accent-blue-500 cursor-pointer"
                title={all ? `Deselect all ${FAMILY_LABELS[g.family]}` : `Select all ${FAMILY_LABELS[g.family]}`}
              />
              <button
                onClick={() => {
                  const next = new Set(collapsed);
                  if (next.has(g.family)) next.delete(g.family); else next.add(g.family);
                  onCollapsedChange(next);
                }}
                className="flex-1 text-left hover:text-gray-200"
              >
                {isCollapsed ? "▸" : "▾"} {FAMILY_LABELS[g.family]}
                <span className="text-gray-600"> {on}/{g.items.length}</span>
              </button>
            </div>
            {!isCollapsed && g.items.map((d) => {
              const isSelected = selected.has(d.key);
              const block = isSelected ? undefined : blocked.get(d.key);
              const tag = isSelected ? axisTag(d.key) : null;
              return (
                <button
                  key={d.key}
                  disabled={!!block}
                  // Alt+click solos: with 20+ series in a run, ticking one and
                  // unticking the rest is the most common action there is.
                  onClick={(e) => { if (e.altKey) onChange(new Set([d.key])); else toggle(d.key); }}
                  title={block ?? `${d.key} — Alt+click to show only this series`}
                  className={`w-full flex items-center gap-1.5 pl-5 pr-1 py-0.5 rounded text-left ${
                    block ? "opacity-40 cursor-not-allowed" : "hover:bg-gray-800"
                  } ${isSelected ? "text-gray-200" : "text-gray-500"}`}
                >
                  <span
                    className="shrink-0"
                    style={{
                      background: isSelected ? d.color : "transparent",
                      border: `1px solid ${d.color}`,
                      width: 8, height: 8, borderRadius: 2, display: "inline-block",
                    }}
                  />
                  <span className="truncate">{d.label}</span>
                  {tag && <span className="ml-auto shrink-0 text-gray-500 font-mono">{tag}</span>}
                </button>
              );
            })}
            {!isCollapsed && (() => {
              // One message per group is enough; naming both axes in every row
              // would fill the popover with the same sentence.
              const msg = g.items.find((d) => !selected.has(d.key) && blocked.has(d.key));
              return msg ? (
                <div className="pl-5 pr-1 py-0.5 text-[9px] text-amber-500/80 leading-tight">
                  {blocked.get(msg.key)}
                </div>
              ) : null;
            })()}
          </div>
        );
      })}
    </div>
  );
}
