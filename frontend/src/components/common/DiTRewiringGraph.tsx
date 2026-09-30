"use client";

import { useRef } from "react";
import type { DiTRewiringPlan, DiTRewiringSpec } from "@/utils/api";

interface Props {
  spec: DiTRewiringSpec; plan: DiTRewiringPlan; selected?: string;
  onSelect?: (id: string) => void;
  onBoundaryChange?: (id: string, side: "start_block" | "end_block", value: number) => void;
}

export default function DiTRewiringGraph(props: Props) {
  return <div className="space-y-3">{props.spec.streams.map(stream => <div key={stream.name}>
    <p className="mb-1 text-gray-300">{stream.name} / {stream.block_count} blocks</p>
    <StreamGraph {...props} stream={stream} />
  </div>)}</div>;
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
          <title>{`${region.id}: residual skip ${region.start_block} → ${region.end_block}`}</title>
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
  </svg>;
}
