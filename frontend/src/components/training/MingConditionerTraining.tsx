"use client";

import type { MingConditionerGroup } from "@/utils/api";

const GROUPS: MingConditionerGroup[] = ["language", "vision", "connector", "queries", "projections"];

export default function MingConditionerTraining({ groups, rates, trainTextEncoder, onGroups, onRates }: {
  groups?: MingConditionerGroup[] | null;
  rates?: Partial<Record<MingConditionerGroup, number>> | null;
  trainTextEncoder: boolean;
  onGroups: (value: MingConditionerGroup[] | null) => void;
  onRates: (value: Partial<Record<MingConditionerGroup, number>>) => void;
}) {
  const selected = groups ?? (trainTextEncoder ? GROUPS : []);
  return <fieldset className="mb-3 space-y-2 rounded border border-sky-800 p-3 text-xs">
    <legend className="px-1 text-sky-300">Ming conditioner training</legend>
    <button type="button" className="text-sky-400 underline" onClick={() => onGroups(null)}>
      Inherit text encoder training ({trainTextEncoder ? "all groups" : "frozen"})
    </button>
    {GROUPS.map(name => <label key={name} className="flex items-center gap-3">
      <input type="checkbox" checked={selected.includes(name)} onChange={event =>
        onGroups(GROUPS.filter(item => item === name ? event.target.checked : selected.includes(item)))} />
      <span className="w-24">{name}</span>
      <input type="number" min={0} step="any" value={rates?.[name] ?? ""}
        disabled={!selected.includes(name)} aria-label={`${name} learning rate`}
        placeholder="TE / base LR" className="w-36 rounded bg-gray-900 px-2 py-1 disabled:opacity-40"
        onChange={event => {
          const next = { ...rates };
          if (event.target.value === "") delete next[name];
          else {
            const value = Number(event.target.value);
            if (!Number.isFinite(value) || value < 0) return;
            next[name] = value;
          }
          onRates(next);
        }} />
    </label>)}
    <p className="text-gray-400">Explicit groups override the global text encoder switch. Empty LR inherits TE then base LR; zero holds weights still. Selected groups retain their live conditioning graph.</p>
  </fieldset>;
}
