"use client";

/**
 * Null / conditional filter for the loss and grad-norm monitors.
 *
 * A run with a nonzero `cfg_uncond_drop_rate` trains a fraction of its items
 * against the architecture's inference CFG null instead of their caption.
 * Those items optimize the caption-free marginal, so their loss sits in its own
 * band and the charted aggregate is a blend of two populations mixed at the
 * drop rate — a change in either one moves it. This picks which population the
 * chart shows.
 *
 * Rendered only when the backend actually emitted the split series
 * (`loss_null`/`loss_cond`, `gnorm_null`/`gnorm_cond`), which a run picks up on
 * its next start; older rows carry no split and the control stays hidden.
 */

export type CfgCondFilter = "all" | "null" | "cond";

const OPTIONS: { value: CfgCondFilter; label: string; title: string }[] = [
  { value: "all", label: "All", title: "Every item, null and conditional pooled — the aggregate series" },
  { value: "null", label: "Null", title: "Only the items trained against the CFG null condition" },
  { value: "cond", label: "Cond", title: "Only the items trained against their own caption" },
];

interface CfgConditionFilterProps {
  value: CfgCondFilter;
  onChange: (value: CfgCondFilter) => void;
  /** Hide entirely when the run emitted no split series. */
  available: boolean;
  /** Filter values with no points in this run — shown disabled, not removed. */
  emptyValues?: CfgCondFilter[];
}

export default function CfgConditionFilter({
  value, onChange, available, emptyValues = [],
}: CfgConditionFilterProps) {
  if (!available) return null;
  return (
    <div className="inline-flex items-center gap-1 mr-2" title="CFG condition">
      <span className="text-[10px] text-gray-500">CFG</span>
      <div className="inline-flex rounded overflow-hidden border border-gray-700">
        {OPTIONS.map((opt) => {
          const empty = emptyValues.includes(opt.value);
          const active = value === opt.value;
          return (
            <button
              key={opt.value}
              onClick={() => onChange(opt.value)}
              disabled={empty}
              title={empty ? `${opt.title} — no steps recorded` : opt.title}
              className={`text-[10px] px-1.5 py-0.5 ${
                active ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              } disabled:opacity-40 disabled:hover:bg-gray-800`}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
