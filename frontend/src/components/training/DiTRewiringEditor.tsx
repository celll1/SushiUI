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
  const [pixels, setPixels] = useState({ height: resolution, width: resolution });
  const [pending, setPending] = useState(false);
  useEffect(() => { setPixels({ height: resolution, width: resolution }); }, [resolution]);
  useEffect(() => {
    let cancelled = false;
    setInfo(null); setPlan(null); setError(null); setPending(true);
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
  const region = current?.regions.find(r => r.id === selected) ?? current?.regions[0];
  const editRegion = (id: string, patch: Partial<NonNullable<typeof region>>) => {
    if (!current || !configuring) return;
    onChange("configure", { ...current, regions: current.regions.map(r => r.id === id ? { ...r, ...patch } : r) });
  };
  const addNested = () => {
    if (!current || !region || region.end_block - region.start_block < 3) return;
    let index = 0; while (current.regions.some(r => r.id === `r${index}`)) index++;
    const child = { ...region, id: `r${index}`, parent: region.id,
      start_block: region.start_block + 1, end_block: region.end_block - 1 };
    onChange("configure", { ...current, regions: [...current.regions, child] });
    setSelected(child.id);
  };
  const removeRegion = () => {
    if (!current || !region) return;
    const removed = new Set([region.id]);
    for (let i = 0; i < current.regions.length; i++)
      current.regions.forEach(r => { if (r.parent && removed.has(r.parent)) removed.add(r.id); });
    onChange("configure", { ...current, regions: current.regions.filter(r => !removed.has(r.id)) });
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
      {info && <span>{info.architecture} / {info.streams[0]?.block_count} blocks</span>}
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
      {plan && <DiTRewiringGraph spec={validatedSpec ?? current} plan={plan} selected={region?.id} onSelect={setSelected}
        onBoundaryChange={configuring ? (id, side, value) => editRegion(id, { [side]: value }) : undefined} />}
      {plan && <div className="flex flex-wrap gap-3">{Array.from(new Set(plan.blocks.map(b => b.tokens))).map(tokens =>
        <span key={tokens}>{plan.blocks.find(b => b.tokens === tokens)?.height}×{plan.blocks.find(b => b.tokens === tokens)?.width}: {tokens} tokens × {plan.blocks.filter(b => b.tokens === tokens).length} blocks</span>)}</div>}
      {region && configuring && <div className="flex flex-wrap items-center gap-3">
        <select aria-label="Region" className="rounded bg-gray-800 p-1" value={region.id} onChange={e => setSelected(e.target.value)}>
          {current.regions.map(r => <option key={r.id} value={r.id}>{r.id}</option>)}
        </select>
        {(["start_block", "end_block"] as const).map(side => <label key={side}>{side === "start_block" ? "Start" : "End boundary"}
          <input type="number" value={region[side]} min={1} max={current.streams[0].block_count - 1}
            className="ml-2 w-16 rounded bg-gray-800 p-1" onChange={e => editRegion(region.id, { [side]: Number(e.target.value) })} />
        </label>)}
        {([0, 1] as const).map(axis => <label key={axis}>{axis === 0 ? "Height" : "Width"} downsample
          <select aria-label={`${axis === 0 ? "Height" : "Width"} downsample`} value={region.down_factor[axis]}
            className="ml-2 rounded bg-gray-800 p-1" onChange={e => {
              const factor: [number, number] = [...region.down_factor]; factor[axis] = Number(e.target.value);
              editRegion(region.id, { down_factor: factor });
            }}>{[2,4,8].map(f => <option key={f} value={f}>1/{f} of parent</option>)}</select>
        </label>)}
        <label>Parent
          <select aria-label="Parent region" value={region.parent ?? ""} className="ml-2 rounded bg-gray-800 p-1"
            onChange={e => editRegion(region.id, { parent: e.target.value || null })}>
            <option value="">Full grid</option>
            {current.regions.filter(r => r.id !== region.id).map(r => <option key={r.id} value={r.id}>{r.id}</option>)}
          </select>
        </label>
        <span>Skip: {region.start_block} → {region.end_block}; parent: {region.parent ?? "full grid"}</span>
        <button type="button" className="rounded bg-gray-700 px-2 py-1" onClick={addNested}>Add nested region</button>
        <button type="button" className="rounded bg-gray-700 px-2 py-1" onClick={removeRegion}>Remove region and children</button>
      </div>}
      <p className="text-gray-400">Block indices start at 0; end boundaries are exclusive. Drag skip endpoints or edit the interval and per-axis shrink factors. Fixed average pooling and residual update addition preserve parameter count. Uncalibrated quality may degrade.</p>
    </>}
    {info && !current && <p className="text-gray-400">{configuring ? "No preset is available for this layout. Supply a topology in the run configuration." : "The checkpoint uses the native DiT sequence."}</p>}
    {pending && <p className="text-gray-400">Validating topology…{plan && " The graph shows the last validated configuration."}</p>}
    {error && <p role="alert" className="text-red-400">{error}</p>}
  </section>;
}
