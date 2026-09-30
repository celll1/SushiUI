"use client";

import { useEffect, useState } from "react";
import { inspectDiTRewiring, previewDiTRewiring, temporalRewiringGrid } from "@/utils/api";
import type { DiTRewiringModelInfo, DiTRewiringPlan, DiTRewiringSpec } from "@/utils/api";
import DiTRewiringGraph from "../common/DiTRewiringGraph";

interface Props {
  modelPath: string; resolution: number;
  mode: "inherit" | "configure" | undefined; spec: DiTRewiringSpec | null | undefined;
  onChange: (mode: "inherit" | "configure", spec: DiTRewiringSpec | null) => void;
  onValidated?: (spec: DiTRewiringSpec | null) => void;
}
const message = (error: unknown): string => {
  const detail = (error as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  return typeof detail === "string" ? detail : error instanceof Error ? error.message : "Topology validation failed";
};

export default function DiTRewiringEditor({ modelPath, resolution, mode, spec, onChange, onValidated }: Props) {
  const [info, setInfo] = useState<DiTRewiringModelInfo | null>(null);
  const [plan, setPlan] = useState<DiTRewiringPlan | null>(null);
  const [validatedSpec, setValidatedSpec] = useState<DiTRewiringSpec | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string>("");
  const [selectedStream, setSelectedStream] = useState<string>("");
  const [pixels, setPixels] = useState({ height: resolution, width: resolution });
  const [frames, setFrames] = useState<number | null>(null);
  const [pending, setPending] = useState(false);
  useEffect(() => { setPixels({ height: resolution, width: resolution }); }, [resolution]);
  useEffect(() => {
    let cancelled = false;
    onValidated?.(null);
    setInfo(null); setPlan(null); setValidatedSpec(null); setSelected(""); setSelectedStream(""); setFrames(null); setError(null); setPending(true);
    inspectDiTRewiring(modelPath).then(value => {
      if (!cancelled) { setInfo(value); setFrames(value.temporal_geometry?.preview_frames ?? null); }
    }).catch(err => { if (!cancelled) setError(message(err)); })
      .finally(() => { if (!cancelled) setPending(false); });
    return () => { cancelled = true; };
  }, [modelPath, onValidated]);
  const configuring = mode === "configure";
  const current = configuring ? (spec ?? info?.default_spec ?? null) : (spec ?? info?.topology ?? null);
  useEffect(() => {
    let cancelled = false;
    onValidated?.(null);
    if (!info) return;
    setPending(false);
    if (!current) { setPlan(null); setError(null); return; }
    if ((current.base_config_sha256 !== info.base_config_sha256 && current.base_config_sha256 !== info.topology?.base_config_sha256) || current.block_layout_sha256 !== info.block_layout_sha256) {
      setError("This topology belongs to a different model configuration. Select the model preset to start a new topology.");
      setPlan(null); return;
    }
    if (!configuring && spec && !info.topology) {
      setError("The inherited topology assertion differs from the checkpoint."); setPlan(null); return;
    }
    const heightScale = info.video_geometry?.height_scale_factor ?? info.latent_scale_factor;
    const widthScale = info.video_geometry?.width_scale_factor ?? info.latent_scale_factor;
    if (!info.temporal_geometry && (pixels.height % (info.video_geometry ? heightScale : info.pixel_alignment) || pixels.width % (info.video_geometry ? widthScale : info.pixel_alignment))) {
      setError(`Spatial dimensions must align to the native grid (${heightScale}×${widthScale}px per token).`); setPlan(null); return;
    }
    let tokenHeight = pixels.height / heightScale;
    let tokenWidth = pixels.width / widthScale;
    if (info.temporal_geometry) {
      try {
        tokenHeight = temporalRewiringGrid(current, frames ?? info.temporal_geometry.preview_frames,
          info.temporal_geometry, info.base_config_sha256).padded;
        tokenWidth = 1;
      } catch (err) { setError(message(err)); setPlan(null); return; }
    }
    setError(null); setPending(true);
    const timer = setTimeout(() => {
      previewDiTRewiring(current, tokenHeight, tokenWidth)
         .then(value => {
          if (cancelled) return;
          if (!configuring && info.topology_sha256 !== value.spec_sha256) {
            setError("The inherited topology assertion differs from the checkpoint."); setPlan(null); return;
          }
          setValidatedSpec(current); setPlan(value.plan); setError(null);
          onValidated?.(value.spec);
        })
        .catch(err => { if (!cancelled) { setError(message(err)); setPlan(null); } })
        .finally(() => { if (!cancelled) setPending(false); });
    }, 120);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [current, info, configuring, spec, pixels, frames, onValidated]);
  let timeGrid: { native: number; padded: number } | null = null;
  if (current && info?.temporal_geometry) {
    try { timeGrid = temporalRewiringGrid(current, frames ?? info.temporal_geometry.preview_frames,
      info.temporal_geometry, info.base_config_sha256); } catch { /* Validation effect reports the error. */ }
  }
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
      {info?.video_geometry && <p className="my-2 text-gray-400">Spatial grids and skip connections apply independently to each video frame. Frame count and audio tokens stay at native resolution.</p>}
      <div className="flex flex-wrap items-center gap-3">
        {info?.temporal_geometry ? <label>Clip frames ({info.temporal_geometry.frame_rate}/s)
          <input type="number" aria-label="Preview audio frames" min={1} step={1}
            value={frames ?? info.temporal_geometry.preview_frames} className="ml-2 w-24 rounded bg-gray-800 p-1"
            onChange={e => setFrames(Number(e.target.value))} />
        </label> : (["height", "width"] as const).map(axis => <label key={axis}>{axis} (px)
          <input type="number" aria-label={`Preview ${axis}`} min={info?.video_geometry ? (axis === "height" ? info.video_geometry.height_scale_factor : info.video_geometry.width_scale_factor) : info?.pixel_alignment} step={info?.video_geometry ? (axis === "height" ? info.video_geometry.height_scale_factor : info.video_geometry.width_scale_factor) : info?.pixel_alignment}
            value={pixels[axis]} className="ml-2 w-24 rounded bg-gray-800 p-1"
            onChange={e => setPixels(p => ({ ...p, [axis]: Number(e.target.value) }))} />
        </label>)}
        {info?.temporal_geometry ? <span>{timeGrid ? `${timeGrid.native} native tokens → ${timeGrid.padded} transport tokens (${timeGrid.padded - timeGrid.native} tail tokens)` : "…"}; final audio length is preserved.</span>
          : <span>Base token grid: {info ? `${pixels.height / (info.video_geometry?.height_scale_factor ?? info.latent_scale_factor)} × ${pixels.width / (info.video_geometry?.width_scale_factor ?? info.latent_scale_factor)}` : "…"}</span>}
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
        {([0, 1] as const).filter(axis => !info?.temporal_geometry || axis === 0).map(axis => <label key={axis}>{info?.temporal_geometry ? "Time" : axis === 0 ? "Height" : "Width"} downsample
          <select aria-label={`${info?.temporal_geometry ? "Time" : axis === 0 ? "Height" : "Width"} downsample`} value={region.down_factor[axis]}
            className="ml-2 rounded bg-gray-800 p-1" onChange={e => {
              const factor: [number, number] = [...region.down_factor]; factor[axis] = Number(e.target.value);
              editRegion(region.id, { down_factor: factor });
            }}>{[1,2,4,8].filter(f => !info?.temporal_geometry || f > 1).map(f => <option key={f} value={f}>{f === 1 ? "Keep parent size" : `1/${f} of parent`}</option>)}</select>
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
