"use client";

import { useRef } from "react";
import type { DiTRewiringPlan, DiTRewiringSpec } from "@/utils/api";

interface Props {
  spec: DiTRewiringSpec; plan: DiTRewiringPlan; selected?: string;
  onSelect?: (id: string) => void;
  onBoundaryChange?: (id: string, side: "start_block" | "end_block", value: number) => void;
}

export default function DiTRewiringGraph(props: Props) {
  return <div className="space-y-3">
    {props.spec.method === "hybrid_unet_ssdit" && <p className="text-gray-400">
      Solid curves: residual bypass. Yellow dashed curves: learned compressed skips. Pink squares: learned spatial convolutions.
    </p>}
    {props.spec.streams.map(stream => <div key={stream.name}>
    <p className="mb-1 text-gray-300">{stream.name} / {stream.block_count} blocks</p>
    {props.plan.hybrid_convolution_halo?.[stream.name] && <p className="mb-1 text-gray-400">
      Convolution context: {props.plan.hybrid_convolution_halo?.[stream.name]?.join(" × ")} base token cells (H × W)
    </p>}
    <StreamGraph {...props} stream={stream} />
    {props.plan.hybrid_activation_layout && <ActivationSummary layout={props.plan.hybrid_activation_layout} stream={stream.name} />}
  </div>)}</div>;
}

function ActivationSummary({ layout, stream }: {
  layout: NonNullable<DiTRewiringPlan["hybrid_activation_layout"]>; stream: string;
}) {
  const peak = layout.streams.find(item => item.stream === stream);
  const entries = layout.entries.filter(item => item.stream === stream);
  const mib = (elements: number, bytes: number) => (elements * bytes / 2 ** 20).toFixed(2);
  const labels = { bypass: "Region bypass", low_input: "Region low input", compressed_skip: "Compressed skip" };
  return <details className="rounded border border-gray-700 p-2">
    <summary className="cursor-pointer text-gray-300">Retained forward payload: {peak ? `${mib(peak.peak_elements, 2)} MiB BF16 / ${mib(peak.peak_elements, 4)} MiB FP32` : "unavailable"}</summary>
    <p className="my-2 text-gray-400">Per image grid and batch item; multiply by batch size and ordered layer grids.
      Logical payload excludes autograd, shared storage, prefix tokens, temporary tensors, weights and optimizer; this is not peak VRAM.
      {peak && <> Peak ownership {peak.peak_phase} block {peak.peak_block}.</>}</p>
    <div className="overflow-x-auto"><table className="w-full text-left text-gray-300">
      <thead><tr><th>Tensor</th><th>Training storage</th><th>Lifetime</th><th>BF16 MiB</th><th>FP32 MiB</th></tr></thead>
      <tbody>{entries.map(item => <tr key={`${item.kind}/${item.id}`}>
        <td>{item.id} / {labels[item.kind]}</td>
        <td>{item.kind === "compressed_skip" ? (item.training_storage?.toUpperCase() ?? "Schema default") : "Execution device"}</td>
        <td>{item.capture_phase} {item.capture_block} → before {item.release_before_block}</td>
        <td>{mib(item.elements, 2)}</td><td>{mib(item.elements, 4)}</td>
      </tr>)}</tbody>
    </table></div>
  </details>;
}

function StreamGraph({ spec, plan, selected, onSelect, onBoundaryChange, stream }:
  Props & { stream: DiTRewiringSpec["streams"][number] }) {
  const drag = useRef<{ id: string; side: "start_block" | "end_block" } | null>(null);
  const blocks = plan.blocks.filter(b => b.stream === stream.name);
  const depth = Math.max(0, ...blocks.map(b => b.depth));
  const height = 100 + depth * 74;
  const x = (index: number) => 85 + index / stream.block_count * 770;
  const y = (level: number) => 50 + level * 74;
  const gap = Math.min(1, 770 / stream.block_count * 0.15);
  return <svg viewBox={`0 0 900 ${height}`} role="img" aria-label={`${stream.name}: DiT resolution levels and residual skip connections`}
    className="w-full rounded bg-gray-950 touch-none"
    onPointerMove={e => {
      if (!drag.current || !onBoundaryChange) return;
      const bounds = e.currentTarget.getBoundingClientRect();
      const px = (e.clientX - bounds.left) / bounds.width * 900;
      const boundary = Math.max(1, Math.min(stream.block_count - 1, Math.round((px - 85) / 770 * stream.block_count)));
      onBoundaryChange(drag.current.id, drag.current.side, boundary);
    }}
    onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
    {Array.from({ length: depth + 1 }, (_, level) => {
      const tokens = Array.from(new Set(blocks.filter(b => b.depth === level).map(b => b.tokens)));
      return <g key={level}>
        <line x1={85} x2={855} y1={y(level)} y2={y(level)} stroke="#374151" strokeDasharray="3 5" />
        <text x={8} y={y(level) + 4} fill="#9ca3af" fontSize={12}>{tokens.length === 1 ? `${tokens[0]} tokens` : `Depth ${level}`}
          <title>{tokens.map(value => `${value} tokens`).join(", ")}</title></text>
      </g>;
    })}
    <polyline points={blocks.map(b => `${x(b.index + 0.5)},${y(b.depth)}`).join(" ")}
      fill="none" stroke="#60a5fa" strokeWidth={2} />
    {blocks.map(block => <g key={block.index}>
      <rect x={x(block.index) + gap} y={y(block.depth) - 6} width={770 / stream.block_count - gap * 2}
        height={12} fill={["#38bdf8", "#a78bfa", "#fbbf24"][Math.min(2, block.depth)]}>
        <title>{`${stream.name} block ${block.index}: ${block.height}×${block.width}, ${block.tokens} tokens`}</title>
      </rect>
      {(block.index % 4 === 0 || block.index === stream.block_count - 1) &&
        <text x={x(block.index)} y={height - 12} fill="#9ca3af" fontSize={11}>{block.index}</text>}
    </g>)}
    {spec.regions.filter(r => r.stream === stream.name).map(region => {
      const level = plan.levels.find(l => l.id === region.id);
      if (!level) return null;
      const base = y(level.depth - 1) - 12;
      const color = selected === region.id ? "#f9fafb" : "#6ee7b7";
      return <g key={region.id} onClick={() => onSelect?.(region.id)}>
        <path d={`M ${x(region.start_block)} ${base} Q ${(x(region.start_block) + x(region.end_block)) / 2} ${base - 35} ${x(region.end_block)} ${base}`}
          fill="none" stroke={color} strokeWidth={2}>
          <title>{`${region.id}: residual skip ${region.start_block} → ${region.end_block}; ${region.down === "rms_pool" ? "RMS pooling" : "average pooling"}`}</title>
        </path>
        {(["start_block", "end_block"] as const).map(side => <circle key={side}
          cx={x(region[side])} cy={base} r={6} fill={color}
          className={onBoundaryChange ? "cursor-ew-resize" : ""}
          onPointerDown={e => {
            onSelect?.(region.id);
            if (!onBoundaryChange) return;
            e.currentTarget.ownerSVGElement?.setPointerCapture(e.pointerId);
            drag.current = { id: region.id, side };
          }} />)}
      </g>;
    })}
    {spec.method === "hybrid_unet_ssdit" && spec.skips.filter(edge => edge.stream === stream.name).map(edge => {
      const source = blocks.find(block => block.index === edge.source_after);
      const target = blocks.find(block => block.index === edge.destination_before);
      if (!source || !target) return null;
      const retained = plan.hybrid_activation_layout?.entries.find(item => item.kind === "compressed_skip" && item.id === edge.id);
      const sx = x(source.index + 0.5), tx = x(target.index + 0.5), base = y(source.depth) - 20;
      return <path key={`learned-${edge.id}`} d={`M ${sx} ${base} Q ${(sx + tx) / 2} ${base - 30} ${tx} ${base}`}
        fill="none" stroke="#fbbf24" strokeWidth={2} strokeDasharray="5 3">
        <title>{`${edge.id}: learned skip after ${edge.source_after} → before ${edge.destination_before}; rank ${edge.rank}${retained ? `; ${retained.elements.toLocaleString()} retained elements per image grid/batch item; training storage ${retained.training_storage ?? "schema default"}` : ""}`}</title>
      </path>;
    })}
    {spec.method === "hybrid_unet_ssdit" && spec.refiners.filter(item => item.stream === stream.name).map(item => {
      const block = blocks.find(value => value.index === item.after_block);
      if (!block) return null;
      return <rect key={`local-${item.id}`} x={x(block.index + 0.5) - 4} y={y(block.depth) + 10}
        width={8} height={8} fill="#fb7185">
        <title>{`${item.id}: local ${item.kernel}×${item.kernel} convolution after block ${item.after_block}; rank ${item.rank}`}</title>
      </rect>;
    })}
    {spec.method === "hybrid_unet_ssdit" && spec.boundaries.map(item => {
      const region = spec.regions.find(value => value.id === item.region && value.stream === stream.name);
      const level = plan.levels.find(value => value.id === item.region);
      if (!region || !level) return null;
      return <g key={`conv-${item.id}`}>{([region.start_block, region.end_block]).map((index, side) =>
        <rect key={side} x={x(index) - 4} y={y(level.depth - 0.5) - 4} width={8} height={8} fill="#fb7185">
          <title>{`${item.id}: learned ${side === 0 ? "down" : "up"} ${item.kernel}×${item.kernel} convolution; rank ${item.rank}`}</title>
        </rect>)}</g>;
    })}
  </svg>;
}
