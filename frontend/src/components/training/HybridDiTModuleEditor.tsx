"use client";

import { useEffect, useRef, useState } from "react";
import { buildHybridDiTRewiring, getDiTRewiringSchema } from "@/utils/api";
import type { HybridDiTRewiringSpec } from "@/utils/api";

interface Props {
  spec: HybridDiTRewiringSpec;
  onApply: (spec: HybridDiTRewiringSpec) => void;
  onEditingChange: (editing: boolean) => void;
}
type Defaults = Awaited<ReturnType<typeof getDiTRewiringSchema>>["hybrid_defaults"];
const inputClass = "ml-2 w-20 rounded bg-gray-800 p-1";
const buttonClass = "rounded bg-gray-700 px-2 py-1 disabled:opacity-40";

function NumberField({ label, value, min, max, onChange }: {
  label: string; value: number; min: number; max: number; onChange: (value: number) => void;
}) {
  return <label>{label}<input aria-label={label} type="number" step={1} min={min} max={max}
    value={Number.isFinite(value) ? value : ""} className={inputClass}
    onChange={event => onChange(event.target.value === "" ? NaN : Number(event.target.value))} /></label>;
}

export default function HybridDiTModuleEditor({ spec, onApply, onEditingChange }: Props) {
  const [draft, setDraft] = useState(spec);
  const [dirty, setDirty] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [defaults, setDefaults] = useState<Defaults | null>(null);
  const revision = useRef(0);
  const source = useRef(spec);
  source.current = spec;
  useEffect(() => {
    revision.current++;
    setDraft(spec); setDirty(false); setPending(false); setError(null);
    onEditingChange(false);
  }, [spec, onEditingChange]);
  useEffect(() => () => { onEditingChange(false); }, [onEditingChange]);
  useEffect(() => {
    let cancelled = false;
    getDiTRewiringSchema().then(value => { if (!cancelled) setDefaults(value.hybrid_defaults); })
      .catch(() => { if (!cancelled) setError("Could not load hybrid defaults. Existing modules can still be edited."); });
    return () => { cancelled = true; revision.current++; };
  }, []);
  const change = (next: HybridDiTRewiringSpec) => {
    revision.current++;
    setDraft(next); setDirty(true); setError(null); onEditingChange(true);
  };
  const freshId = (prefix: string) => {
    const ids = new Set([...draft.boundaries, ...draft.refiners, ...draft.skips, ...draft.regions].map(item => item.id));
    let index = 0; while (ids.has(`${prefix}_${index}`)) index++;
    return `${prefix}_${index}`;
  };
  const count = (stream: string) => draft.streams.find(item => item.name === stream)?.block_count ?? 0;
  const apply = async () => {
    const stamp = ++revision.current;
    const original = spec;
    setPending(true); setError(null); onEditingChange(true);
    try {
      const value = await buildHybridDiTRewiring(draft);
      if (stamp !== revision.current || source.current !== original) return;
      onApply(value.spec);
    } catch (failure) {
      if (stamp !== revision.current || source.current !== original) return;
      const detail = (failure as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Hybrid validation failed. Check integer fields and same-grid skip endpoints.");
    } finally {
      if (stamp === revision.current && source.current === original) setPending(false);
    }
  };
  const discard = () => {
    revision.current++; setDraft(spec); setDirty(false); setPending(false); setError(null); onEditingChange(false);
  };
  const kernel = (label: string, value: 3 | 5, update: (value: 3 | 5) => void) => <label>{label}
    <select aria-label={label} value={value} className={inputClass}
      onChange={event => update(Number(event.target.value) as 3 | 5)}>
      <option value={3}>3×3</option><option value={5}>5×5</option>
    </select></label>;
  const streamSelect = (label: string, value: string, update: (value: string) => void) => <label>{label}
    <select aria-label={label} value={value} className={inputClass} onChange={event => update(event.target.value)}>
      {draft.streams.map(stream => <option key={stream.name} value={stream.name}>{stream.name}</option>)}
    </select></label>;
  const addRegion = (parentId: string | null) => {
    if (!defaults || draft.regions.length >= 64) return;
    const parent = draft.regions.find(region => region.id === parentId);
    const stream = parent?.stream ?? draft.streams[0].name;
    const siblings = draft.regions.filter(region => region.stream === stream && region.parent === parentId)
      .sort((a, b) => a.start_block - b.start_block);
    let start = parent ? parent.start_block + 1 : 1;
    const limit = parent ? parent.end_block - 1 : count(stream) - 1;
    let end = limit;
    for (const sibling of siblings) {
      if (start < sibling.start_block) { end = Math.min(limit, sibling.start_block); break; }
      start = Math.max(start, sibling.end_block);
    }
    if (start >= end) { setError("No free interval here. Adjust the existing region boundaries first."); return; }
    const id = freshId("region");
    const template = draft.regions.find(region => region.stream === stream);
    if (!template) { setError("No spatial region template for this stream."); return; }
    change({ ...draft, regions: [...draft.regions, { ...template, id, parent: parentId, stream,
      start_block: start, end_block: end, down_factor: [...template.down_factor] }], boundaries: [...draft.boundaries,
      { id: freshId("boundary"), region: id, rank: Math.min(defaults.rank, draft.stream_widths[stream]), kernel: defaults.kernel }] });
  };
  const regionRemoval = (id: string) => {
    const removed = new Set([id]);
    for (let index = 0; index < draft.regions.length; index++)
      draft.regions.forEach(region => { if (region.parent && removed.has(region.parent)) removed.add(region.id); });
    return removed;
  };
  const removeRegion = (id: string) => {
    const removed = regionRemoval(id);
    if (removed.size === draft.regions.length) return;
    change({ ...draft, regions: draft.regions.filter(region => !removed.has(region.id)),
      boundaries: draft.boundaries.filter(boundary => !removed.has(boundary.region)) });
  };
  return <div className="space-y-3 rounded border border-gray-700 p-3">
    <p className="text-gray-400">Edit the draft, then validate and apply. The diagram shows the applied configuration. Invalid endpoints are retained for correction; they cannot be used to start training.</p>
    <fieldset disabled={pending} className="space-y-3">
      <h5 className="font-medium">Resolution regions — 1/2 per axis at each learned boundary</h5>
      {draft.regions.map(region => <div key={region.id} className="flex flex-wrap items-center gap-3">
        <span>{region.id} / {region.stream} / parent {region.parent ?? "full grid"}</span>
        {(["start_block", "end_block"] as const).map(side => <NumberField key={side}
          label={`${region.id} ${side === "start_block" ? "start" : "end"}`} value={region[side]} min={1} max={count(region.stream) - 1}
          onChange={value => change({ ...draft, regions: draft.regions.map(item => item.id === region.id ? { ...item, [side]: value } : item) })} />)}
        <button type="button" disabled={!defaults || draft.regions.length >= 64} className={buttonClass} onClick={() => addRegion(region.id)}>Add nested region</button>
        <button type="button" disabled={regionRemoval(region.id).size === draft.regions.length} className={buttonClass} onClick={() => removeRegion(region.id)}>Remove region and children</button>
      </div>)}
      <button type="button" disabled={!defaults || draft.regions.length >= 64} className={buttonClass} onClick={() => addRegion(null)}>Add independent region</button>
      <h5 className="font-medium">Learned down / up</h5>
      {draft.boundaries.map(item => {
        const region = draft.regions.find(region => region.id === item.region);
        const update = (patch: Partial<typeof item>) => change({ ...draft, boundaries: draft.boundaries.map(value => value.id === item.id ? { ...value, ...patch } : value) });
        return <div key={item.id} className="flex flex-wrap items-center gap-3"><span>{item.id} / {item.region}</span>
          <NumberField label={`${item.id} rank`} value={item.rank} min={1} max={draft.stream_widths[region?.stream ?? ""] ?? 1} onChange={rank => update({ rank })} />
          {kernel(`${item.id} kernel`, item.kernel, kernel => update({ kernel }))}
        </div>;
      })}
      <h5 className="font-medium">Local spatial refiners</h5>
      {draft.refiners.map(item => {
        const update = (patch: Partial<typeof item>) => change({ ...draft, refiners: draft.refiners.map(value => value.id === item.id ? { ...value, ...patch } : value) });
        return <div key={item.id} className="flex flex-wrap items-center gap-3"><span>{item.id}</span>
          {streamSelect(`${item.id} stream`, item.stream, stream => update({ stream }))}
          <NumberField label={`${item.id} after block`} value={item.after_block} min={0} max={count(item.stream) - 1} onChange={after_block => update({ after_block })} />
          <NumberField label={`${item.id} rank`} value={item.rank} min={1} max={draft.stream_widths[item.stream]} onChange={rank => update({ rank })} />
          {kernel(`${item.id} kernel`, item.kernel, kernel => update({ kernel }))}
          <button type="button" className={buttonClass} onClick={() => change({ ...draft, refiners: draft.refiners.filter(value => value.id !== item.id) })}>Remove</button>
        </div>;
      })}
      <button type="button" disabled={!defaults || draft.refiners.length >= 128} className={buttonClass} onClick={() => {
        if (!defaults) return;
        const stream = draft.streams[0].name;
        change({ ...draft, refiners: [...draft.refiners, { id: freshId("local"), stream, after_block: 0,
          rank: Math.min(defaults.rank, draft.stream_widths[stream]), kernel: defaults.kernel }] });
      }}>Add local refiner</button>
      <h5 className="font-medium">Learned compressed skips — source after block → destination before block</h5>
      {draft.skips.map(item => {
        const update = (patch: Partial<typeof item>) => change({ ...draft, skips: draft.skips.map(value => value.id === item.id ? { ...value, ...patch } : value) });
        return <div key={item.id} className="flex flex-wrap items-center gap-3"><span>{item.id}</span>
          {streamSelect(`${item.id} stream`, item.stream, stream => update({ stream }))}
          {(["source_after", "destination_before"] as const).map(side => <NumberField key={side} label={`${item.id} ${side}`}
            value={item[side]} min={side === "source_after" ? 0 : 1} max={count(item.stream) - 1} onChange={value => update({ [side]: value })} />)}
          <NumberField label={`${item.id} rank`} value={item.rank} min={1} max={draft.stream_widths[item.stream]} onChange={rank => update({ rank })} />
          <button type="button" className={buttonClass} onClick={() => change({ ...draft, skips: draft.skips.filter(value => value.id !== item.id) })}>Remove</button>
        </div>;
      })}
      <button type="button" disabled={!defaults || draft.skips.length >= 128} className={buttonClass} onClick={() => {
        if (!defaults) return;
        const stream = draft.streams[0].name;
        change({ ...draft, skips: [...draft.skips, { id: freshId("skip"), stream, source_after: 0,
          destination_before: count(stream) - 1, rank: Math.min(defaults.rank, draft.stream_widths[stream]) }] });
      }}>Add compressed skip</button>
    </fieldset>
    <div className="flex items-center gap-3">
      <button type="button" disabled={!dirty || pending} className={buttonClass} onClick={apply}>Validate and apply draft</button>
      <button type="button" disabled={!dirty && !pending} className={buttonClass} onClick={discard}>Discard draft</button>
      <span>{pending ? "Validating draft…" : dirty ? "Unapplied draft — training configuration is invalidated" : "Applied configuration"}</span>
    </div>
    {error && <p role="alert" className="text-red-400">{error}</p>}
  </div>;
}
