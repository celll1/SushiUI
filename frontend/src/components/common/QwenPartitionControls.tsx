"use client";

import { useRef, useState } from "react";
import { useStartup } from "@/contexts/StartupContext";
import type { QwenPartitionConfig, QwenPartitionSplit } from "@/utils/api";
import { area, partitionPreview, replaceNode, strips } from "@/utils/qwenPartition";

const colors = ["#60a5fa", "#fbbf24", "#34d399", "#f472b6", "#a78bfa", "#fb923c"];
type Props = { value?: QwenPartitionConfig | null; onChange: (v: QwenPartitionConfig) => void;
  width?: number; height?: number; background?: string | null;
  architecture?: "qwen_image_21" | "ming_image_01"; targetFrames?: number };

export default function QwenPartitionControls({ value, onChange, width, height, background, architecture = "qwen_image_21", targetFrames = 1 }: Props) {
  const { generationDefaults } = useStartup();
  const defaults = generationDefaults?.txt2img?.[architecture === "ming_image_01" ? "ming_image_01_partition" : "qwen_image_21_partition"];
  const config = value ?? defaults;
  const [selected, setSelected] = useState("");
  const [drag, setDrag] = useState<string | null>(null);
  const [history, setHistory] = useState<Array<QwenPartitionSplit | null>>([]);
  const svg = useRef<SVGSVGElement>(null);
  if (!config) return null;
  const w = Math.max(2, Math.floor((width ?? 32) / 32) * 2);
  const h = Math.max(2, Math.floor((height ?? 32) / 32) * 2);
  const change = (patch: Partial<QwenPartitionConfig>) => onChange({ ...config, ...patch });
  let preview: ReturnType<typeof partitionPreview> | null = null;
  let error = "";
  try { preview = partitionPreview({ ...config, max_target_tokens: Math.floor(config.max_target_tokens / targetFrames) }, h, w); } catch (e) { error = (e as Error).message; }
  const remember = () => setHistory(previous => [...previous.slice(-19), config.layout]);
  const layout = (node: QwenPartitionSplit | null) => { remember(); change({ mode: "manual", layout: node }); setSelected(""); };
  const move = (event: React.PointerEvent<SVGSVGElement>) => {
    if (drag === null || !preview || !svg.current) return;
    const divider = preview.dividers.find(d => d.path === drag);
    if (!divider) return;
    const point = svg.current.createSVGPoint();
    point.x = event.clientX; point.y = event.clientY;
    const matrix = svg.current.getScreenCTM();
    if (!matrix) return;
    const cursor = point.matrixTransform(matrix.inverse());
    const b = divider.box;
    const start = divider.axis === "x" ? b.left : b.top;
    const end = divider.axis === "x" ? b.right : b.bottom;
    const cut = Math.max(start + 2, Math.min(end - 2, Math.round((divider.axis === "x" ? cursor.x : cursor.y) / 2) * 2));
    change({ layout: replaceNode(config.layout, drag, node => node && ({ ...node, ratio: (cut - start) / (end - start) })) });
  };
  return <details className="rounded border border-gray-700 p-3 my-2">
    <summary className="cursor-pointer text-sm">{architecture === "ming_image_01" ? "Ming" : "Qwen"} partition generation · {config.mode}</summary>
    <div className="flex flex-wrap gap-3 my-3 text-sm">
      <label>Mode <select value={config.mode} onChange={e => change({ mode: e.target.value as QwenPartitionConfig["mode"] })}
        className="bg-gray-900 border rounded p-1 ml-2"><option value="off">Off</option><option value="auto">Auto</option><option value="manual">Manual</option></select></label>
      {config.mode === "auto" && <label>Max target tokens <input type="number" min={1} step={1}
        value={config.max_target_tokens} onChange={e => { if (e.target.value) change({ max_target_tokens: Number(e.target.value) }); }}
        className="bg-gray-900 border rounded p-1 ml-2 w-24" /></label>}
      {config.mode !== "off" && <label>Halo ({architecture === "ming_image_01" ? "spatial patch tokens" : "latent pixels"}) <input type="number" min={0} step={2}
        value={config.halo_tokens} onChange={e => { if (e.target.value) change({ halo_tokens: Number(e.target.value) }); }}
        className="bg-gray-900 border rounded p-1 ml-2 w-20" /></label>}
    </div>
    {config.mode === "off" ? <p className="text-xs text-gray-400">Full-frame path. Global adapters are unused.</p> : <>
      <p className="text-xs text-gray-400">{w * 16} × {h * 16} px · {w} × {h} {architecture === "ming_image_01" ? "patch grid" : "latent"} · {preview?.regions.length ?? "?"} regions
        {preview && ` · largest input ${Math.max(...preview.regions.map(r => area(r.input) * targetFrames))} tokens`}. Reference Guide and Style Transfer are unavailable {architecture === "ming_image_01" ? "while partition mode is enabled" : "with multiple regions"}.</p>
      {architecture === "ming_image_01" && <p className="text-xs text-gray-400">All {targetFrames} target frames, including the Layer auxiliary canvas, count toward the ceiling. References and captions stay complete; their attention is recomputed per region. Global summaries are used when saved in the base or LoRA.</p>}
      {config.mode === "manual" && <div className="flex flex-wrap gap-2 my-2 text-xs">
        {[2, 3, 6].map(n => <button key={n} type="button" onClick={() => layout(strips(n, "x"))}> {n} columns </button>)}
        <button type="button" onClick={() => layout({ axis: "y", ratio: 0.5, first: strips(3, "x"), second: strips(3, "x") })}>2 × 3</button>
        <button type="button" onClick={() => layout({ axis: "y", ratio: 0.5, first: strips(2, "x"), second: strips(2, "x") })}>2 × 2</button>
        {["x", "y"].map(axis => <button key={axis} type="button" onClick={() => {
          remember();
          change({ layout: replaceNode(config.layout, selected, () => ({ axis: axis as "x" | "y", ratio: 0.5, first: null, second: null })) });
          setSelected(selected + "0");
        }}>Split selected {axis}</button>)}
        <button type="button" onClick={() => layout(null)}>Reset</button>
        <button type="button" disabled={!history.length} onClick={() => {
          change({ layout: history[history.length - 1] }); setHistory(previous => previous.slice(0, -1)); setSelected("");
        }}>Undo</button>
      </div>}
      {error && <p role="alert" className="text-xs text-red-400">{error}</p>}
      {preview && <svg ref={svg} viewBox={`0 0 ${w} ${h}`} className="w-full max-h-64 my-2 bg-gray-950"
        style={{ touchAction: "none" }} onPointerMove={move} onPointerUp={() => setDrag(null)} onPointerCancel={() => setDrag(null)}>
        {background && <image href={background} width={w} height={h} opacity={0.4} preserveAspectRatio="xMidYMid meet" />}
        {preview.regions.map((r, i) => <g key={r.path}>
          <rect x={r.input.left} y={r.input.top} width={r.input.right - r.input.left} height={r.input.bottom - r.input.top}
            fill={colors[i % colors.length]} opacity={0.06} pointerEvents="none" />
          <rect x={r.core.left} y={r.core.top} width={r.core.right - r.core.left} height={r.core.bottom - r.core.top}
            fill={colors[i % colors.length]} fillOpacity={selected === r.path ? 0.2 : 0.08}
            stroke={colors[i % colors.length]} strokeWidth={0.4} strokeDasharray="2 1"
            onClick={() => setSelected(r.path)}><title>Region {i + 1}: ({r.core.left * 16}, {r.core.top * 16})–({r.core.right * 16}, {r.core.bottom * 16}) px; core {area(r.core) * targetFrames}, input {area(r.input) * targetFrames} target tokens</title></rect>
          <text x={(r.core.left + r.core.right) / 2} y={(r.core.top + r.core.bottom) / 2} fontSize={Math.min(w, h) / 18}
            fill={colors[i % colors.length]} textAnchor="middle" pointerEvents="none">{i + 1}</text>
        </g>)}
        {preview.dividers.map(d => <line key={d.path} x1={d.axis === "x" ? d.cut : d.box.left} y1={d.axis === "x" ? d.box.top : d.cut}
          x2={d.axis === "x" ? d.cut : d.box.right} y2={d.axis === "x" ? d.box.bottom : d.cut}
          stroke="transparent" strokeWidth={Math.min(w, h) / 15} style={{ cursor: d.axis === "x" ? "ew-resize" : "ns-resize" }}
          onPointerDown={e => { remember(); setDrag(d.path); svg.current?.setPointerCapture(e.pointerId); e.preventDefault(); }} />)}
      </svg>}
      <p className="text-xs text-gray-400">Select a region to split; drag its boundaries. Snaps to 32 px. Halo supplies shared border context; zero halo can split subjects even with a global adapter. Global adapters follow their LoRA strength and step range. Without an adapter, regions share no learned global summary.</p>
    </>}
  </details>;
}
