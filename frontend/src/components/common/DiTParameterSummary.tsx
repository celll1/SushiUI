"use client";

import type { DiTRewiringParameterCounts, DiTRewiringPlan } from "@/utils/api";

export default function DiTParameterSummary({ counts, plan }: {
  counts?: DiTRewiringParameterCounts | null; plan?: DiTRewiringPlan | null;
}) {
  if (!counts) return <p className="text-gray-400">Native parameter count unavailable for this architecture. Fixed U-Net transport adds 0 parameters.</p>;
  const groups = new Map<string, number>([["Shared projections / conditioning", counts.shared]]);
  const levels = new Map<string, DiTRewiringPlan["blocks"][number]>(plan?.blocks.map(b => [`${b.stream}/${b.index}`, b]) ?? []);
  for (const block of counts.blocks) {
    const level = levels.get(`${block.stream}/${block.index}`);
    const label = level ? `${block.stream}: ${level.height}×${level.width} / depth ${level.depth}` : `${block.stream}: native blocks`;
    groups.set(label, (groups.get(label) ?? 0) + block.count);
  }
  return <div className="space-y-2 rounded border border-gray-700 p-2">
    <p>Native DiT: {counts.total.toLocaleString()} parameters / U-Net: {counts.total.toLocaleString()} / added: 0</p>
    <div role="img" aria-label="Native DiT parameter distribution by token resolution">
      {Array.from(groups, ([label, count]) => <div key={label} className="my-1">
        <div className="flex flex-wrap justify-between gap-2"><span>{label}</span><span>{count.toLocaleString()}</span></div>
        <div className="h-2 rounded bg-gray-800"><div className="h-2 rounded bg-sky-500"
          style={{ width: `${counts.total ? count / counts.total * 100 : 0}%` }} /></div>
      </div>)}
    </div>
    <p className="text-gray-400">Logical model parameters; excludes text encoders and VAE. Token reduction changes computation, while these weights are retained. Quantized file size is a separate measure.</p>
  </div>;
}
