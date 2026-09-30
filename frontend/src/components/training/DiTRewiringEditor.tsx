"use client";

import { useEffect, useState } from "react";
import { inspectDiTRewiring, previewDiTRewiring } from "@/utils/api";
import type { DiTRewiringModelInfo, DiTRewiringPlan, DiTRewiringSpec } from "@/utils/api";
import DiTRewiringGraph from "../common/DiTRewiringGraph";

interface Props {
  modelPath: string; resolution: number;
  mode: "inherit" | "configure" | undefined; spec: DiTRewiringSpec | null | undefined;
  onChange: (mode: "inherit" | "configure", spec: DiTRewiringSpec | null) => void;
}
const message = (error: unknown): string => {
  const detail = (error as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  return typeof detail === "string" ? detail : error instanceof Error ? error.message : "Topology validation failed";
};

export default function DiTRewiringEditor({ modelPath, resolution, mode, spec, onChange }: Props) {
  const [info, setInfo] = useState<DiTRewiringModelInfo | null>(null);
  const [plan, setPlan] = useState<DiTRewiringPlan | null>(null);
  const [validatedSpec, setValidatedSpec] = useState<DiTRewiringSpec | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string>("");
  const [selectedStream, setSelectedStream] = useState<string>("");
  const [pixels, setPixels] = useState({ height: resolution, width: resolution });
  const [pending, setPending] = useState(false);
  useEffect(() => { setPixels({ height: resolution, width: resolution }); }, [resolution]);
  useEffect(() => {
    let cancelled = false;
    setInfo(null); setPlan(null); setValidatedSpec(null); setSelected(""); setSelectedStream(""); setError(null); setPending(true);
    inspectDiTRewiring(modelPath).then(value => {
      if (!cancelled) setInfo(value);
    }).catch(err => { if (!cancelled) setError(message(err)); })
      .finally(() => { if (!cancelled) setPending(false); });
    return () => { cancelled = true; };
  }, [modelPath]);
  const configuring = mode === "configure";
  const current = configuring ? (spec ?? info?.default_spec ?? null) : (spec ?? info?.topology ?? null);
  useEffect(() => {
    let cancelled = false;
    if (!info) return;
    setPending(false);
    if (!current) { setPlan(null); setError(null); return; }
    if (current.base_config_sha256 !== info.base_config_sha256 || current.block_layout_sha256 !== info.block_layout_sha256) {
      setError("This topology belongs to a different model configuration. Select the model preset to start a new topology.");
      setPlan(null); return;
    }
    if (!configuring && spec && !info.topology) {
      setError("The inherited topology assertion differs from the checkpoint."); setPlan(null); return;
    }
    if (pixels.height % info.pixel_alignment || pixels.width % info.pixel_alignment) {
      setError(`Image dimensions must be multiples of ${info.pixel_alignment}px.`); setPlan(null); return;
    }
    setError(null); setPending(true);
    const timer = setTimeout(() => {
      previewDiTRewiring(current, pixels.height / info.latent_scale_factor, pixels.width / info.latent_scale_factor)
         .then(value => {
          if (cancelled) return;
          if (!configuring && info.topology_sha256 !== value.spec_sha256) {
            setError("The inherited topology assertion differs from the checkpoint."); setPlan(null); return;
          }
          setValidatedSpec(current); setPlan(value.plan); setError(null);
        })
        .catch(err => { if (!cancelled) { setError(message(err)); setPlan(null); } })
        .finally(() => { if (!cancelled) setPending(false); });
    }, 120);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [current, info, configuring, spec, pixels]);
  const streams = current?.streams ?? info?.streams ?? [];
  const stream = streams.find(s => s.name === selectedStream) ?? streams[0];
  const regions = current?.regions.filter(r => r.stream === stream?.name) ?? [];
  const region = regions.find(r => r.id === selected) ?? regions[0];
  const selectRegion = (id: string) => {
    const chosen = current?.regions.find(r => r.id === id);
    if (!chosen) return;
    setSelectedStream(chosen.stream); setSelected(id);
  };
  const customLayout = (): DiTRewiringSpec | null => {
    if (!info || !stream) return null;
    const template = info.initial_spec.regions[0];
    return { ...info.initial_spec, regions: [{ ...template, stream: stream.name, parent: null,
      start_block: template.start_block, end_block: stream.block_count - template.start_block }] };
  };
  const gridSummary = new Map<string, { stream: string; height: number; width: number; tokens: number; count: number }>();
  for (const block of plan?.blocks ?? []) {
    const key = `${block.stream}/${block.height}/${block.width}`;
    const value = gridSummary.get(key);
    if (value) value.count++;
    else gridSummary.set(key, { stream: block.stream, height: block.height, width: block.width, tokens: block.tokens, count: 1 });
  }
  const editRegion = (id: string, patch: Partial<NonNullable<typeof region>>) => {
    if (!current || !configuring) return;
    onChange("configure", { ...current, regions: current.regions.map(r => r.id === id ? { ...r, ...patch } : r) });
  };
  const freeInterval = (parent: string | null): [number, number] | null => {
    if (!current || !info || current.regions.length >= 64) return null;
    const template = info.initial_spec.regions[0];
    const owner = parent ? current.regions.find(r => r.id === parent) : null;
    const streamName = owner?.stream ?? stream?.name;
    const count = current.streams.find(s => s.name === streamName)?.block_count;
    if (count === undefined) return null;
    let cursor = owner ? owner.start_block + 1 : template.start_block;
    const end = owner ? owner.end_block - 1 : count - template.start_block;
    const siblings = current.regions.filter(r => r.parent === parent && r.stream === streamName)
      .sort((a, b) => a.start_block - b.start_block);
    for (const sibling of siblings) {
      if (cursor < Math.min(sibling.start_block, end)) return [cursor, Math.min(sibling.start_block, end)];
      cursor = Math.max(cursor, sibling.end_block);
    }
    return cursor < end ? [cursor, end] : null;
  };
  const addRegion = (parent: string | null) => {
    if (!info || !stream || !configuring) return;
    if (!current) {
      const layout = customLayout();
      if (layout) { onChange("configure", layout); setSelected(layout.regions[0].id); }
      return;
    }
    const interval = freeInterval(parent);
    if (!interval) return;
    let index = 0; while (current.regions.some(r => r.id === `r${index}`)) index++;
    const added = { ...info.initial_spec.regions[0], id: `r${index}`, parent,
      stream: stream.name,
      start_block: interval[0], end_block: interval[1] };
    onChange("configure", { ...current, regions: [...current.regions, added] });
    setSelected(added.id);
  };
  const removedIds = new Set(region ? [region.id] : []);
  if (current) for (let i = 0; i < current.regions.length; i++)
    current.regions.forEach(r => { if (r.parent && removedIds.has(r.parent)) removedIds.add(r.id); });
  const canRemove = !!current && current.regions.some(r => !removedIds.has(r.id));
  const removeRegion = () => {
    if (!current || !region || !canRemove) return;
    onChange("configure", { ...current, regions: current.regions.filter(r => !removedIds.has(r.id)) });
    setSelected("");
  };
  return <section className="space-y-3 rounded border border-gray-700 p-3 text-xs">
    <h4 className="font-medium text-gray-200">Residual U-Net topology</h4>
    <div className="flex flex-wrap items-center gap-3">
      <select aria-label="Topology source" value={mode ?? ""} disabled={!info}
        className="rounded bg-gray-800 p-2" onChange={e => onChange(e.target.value as "inherit" | "configure",
          e.target.value === "configure" ? (spec ?? info?.default_spec ?? null) : null)}>
        {mode === undefined && <option value="">Loading training defaults…</option>}
        <option value="inherit">Inherit checkpoint</option>
        <option value="configure">Configure a new U-Net</option>
      </select>
      {configuring && <button type="button" disabled={!info?.default_spec} className="rounded bg-gray-700 px-3 py-2"
        onClick={() => onChange("configure", info?.default_spec ?? null)}>Model preset</button>}
      {info && <span>{info.architecture}</span>}
      {streams.length > 0 && <label>Stream
        <select aria-label="DiT stream" value={stream?.name ?? ""} className="ml-2 rounded bg-gray-800 p-1"
          onChange={e => { setSelectedStream(e.target.value); setSelected(""); }}>
          {streams.map(s => <option key={s.name} value={s.name}>{s.name} / {s.block_count} blocks</option>)}
        </select>
      </label>}
      {configuring && info && <button type="button" className="rounded bg-gray-700 px-3 py-2"
        disabled={!!current && !freeInterval(null)} onClick={() => addRegion(null)}>
        {current ? "Add independent region" : "Create first region"}</button>}
      {configuring && info && current && <button type="button" className="rounded bg-gray-700 px-3 py-2"
        title="Replace the topology with one editable region in the selected stream" onClick={() => {
          const layout = customLayout();
          if (layout) { onChange("configure", layout); setSelected(layout.regions[0].id); }
        }}>Start custom layout</button>}
    </div>
    {current && <>
      <div className="flex flex-wrap items-center gap-3">
        {(["height", "width"] as const).map(axis => <label key={axis}>{axis} (px)
          <input type="number" aria-label={`Preview ${axis}`} min={info?.pixel_alignment} step={info?.pixel_alignment}
            value={pixels[axis]} className="ml-2 w-24 rounded bg-gray-800 p-1"
            onChange={e => setPixels(p => ({ ...p, [axis]: Number(e.target.value) }))} />
        </label>)}
        <span>latent grid: {info ? `${pixels.height / info.latent_scale_factor} × ${pixels.width / info.latent_scale_factor}` : "…"}</span>
      </div>
      {plan && <DiTRewiringGraph spec={validatedSpec ?? current} plan={plan} selected={region?.id} onSelect={selectRegion}
        onBoundaryChange={configuring ? (id, side, value) => editRegion(id, { [side]: value }) : undefined} />}
      {plan && <div className="flex flex-wrap gap-3">{Array.from(gridSummary.entries()).map(([key, grid]) =>
        <span key={key}>{grid.stream} / {grid.height}×{grid.width}: {grid.tokens} tokens × {grid.count} blocks</span>)}</div>}
      {region && configuring && <div className="flex flex-wrap items-center gap-3">
        <select aria-label="Region" className="rounded bg-gray-800 p-1" value={region.id} onChange={e => setSelected(e.target.value)}>
          {regions.map(r => <option key={r.id} value={r.id}>{r.id}</option>)}
        </select>
        {(["start_block", "end_block"] as const).map(side => <label key={side}>{side === "start_block" ? "Start" : "End boundary"}
          <input type="number" value={region[side]} min={1} max={(stream?.block_count ?? 1) - 1}
            className="ml-2 w-16 rounded bg-gray-800 p-1" onChange={e => editRegion(region.id, { [side]: Number(e.target.value) })} />
        </label>)}
        {([0, 1] as const).map(axis => <label key={axis}>{axis === 0 ? "Height" : "Width"} downsample
          <select aria-label={`${axis === 0 ? "Height" : "Width"} downsample`} value={region.down_factor[axis]}
            className="ml-2 rounded bg-gray-800 p-1" onChange={e => {
              const factor: [number, number] = [...region.down_factor]; factor[axis] = Number(e.target.value);
              editRegion(region.id, { down_factor: factor });
            }}>{[1,2,4,8].map(f => <option key={f} value={f}>{f === 1 ? "Keep parent size" : `1/${f} of parent`}</option>)}</select>
        </label>)}
        <label>Parent
          <select aria-label="Parent region" value={region.parent ?? ""} className="ml-2 rounded bg-gray-800 p-1"
            onChange={e => editRegion(region.id, { parent: e.target.value || null })}>
            <option value="">Full grid</option>
            {regions.filter(r => !removedIds.has(r.id)).map(r => <option key={r.id} value={r.id}>{r.id}</option>)}
          </select>
        </label>
        <span>{region.stream} skip: {region.start_block} → {region.end_block}; parent: {region.parent ?? "full grid"}</span>
        <button type="button" className="rounded bg-gray-700 px-2 py-1" disabled={!freeInterval(region.id)} onClick={() => addRegion(region.id)}>Add nested region</button>
        <button type="button" className="rounded bg-gray-700 px-2 py-1" disabled={!canRemove} onClick={removeRegion}>Remove region and children</button>
      </div>}
      <p className="text-gray-400">Block indices start at 0; end boundaries are exclusive. Drag skip endpoints or edit the interval and per-axis shrink factors. Fixed average pooling and residual update addition preserve parameter count. Uncalibrated quality may degrade.</p>
    </>}
    {info && !current && <p className="text-gray-400">{configuring ? "No model preset is available. Create a region and edit its interval and shrink factors." : "The checkpoint uses the native DiT sequence."}</p>}
    {pending && <p className="text-gray-400">Validating topology…{plan && " The graph shows the last validated configuration."}</p>}
    {error && <p role="alert" className="text-red-400">{error}</p>}
  </section>;
}
