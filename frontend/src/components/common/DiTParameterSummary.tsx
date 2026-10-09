"use client";

import type { DiTRewiringParameterCounts, DiTRewiringPlan, DiTRewiringSpec } from "@/utils/api";

export default function DiTParameterSummary({ counts, plan, spec }: {
  counts?: DiTRewiringParameterCounts | null; plan?: DiTRewiringPlan | null;
  spec?: DiTRewiringSpec | null;
}) {
  const added = spec?.method === "hybrid_unet_ssdit" ? plan?.hybrid_parameter_counts?.added ?? null : 0;
  if (!counts) return <p className="text-gray-400">Native parameter count unavailable for this architecture.
    {added !== null && <> Added U-Net parameters: {added.toLocaleString()}.</>}</p>;
  const total = counts.total + (added ?? 0);
  const groups = new Map<string, number>([["Shared projections / conditioning", counts.shared]]);
  if (added) groups.set("Learned U-Net modules", added);
  const levels = new Map<string, DiTRewiringPlan["blocks"][number]>(plan?.blocks.map(b => [`${b.stream}/${b.index}`, b]) ?? []);
  for (const block of counts.blocks) {
    const level = levels.get(`${block.stream}/${block.index}`);
    const label = level ? `${block.stream}: ${level.height}×${level.width} / depth ${level.depth}` : `${block.stream}: native blocks`;
    groups.set(label, (groups.get(label) ?? 0) + block.count);
  }
  return <div className="space-y-2 rounded border border-gray-700 p-2">
    <p>Native DiT: {counts.total.toLocaleString()} parameters / U-Net: {added === null ? "pending preview" : total.toLocaleString()} / added: {added === null ? "pending preview" : added.toLocaleString()}</p>
    <div role="img" aria-label="DiT and learned U-Net parameter distribution by token resolution">
      {Array.from(groups, ([label, count]) => <div key={label} className="my-1">
        <div className="flex flex-wrap justify-between gap-2"><span>{label}</span><span>{count.toLocaleString()}</span></div>
        <div className="h-2 rounded bg-gray-800"><div className="h-2 rounded bg-sky-500"
          style={{ width: `${total ? count / total * 100 : 0}%` }} /></div>
      </div>)}
    </div>
    <p className="text-gray-400">Logical model parameters; excludes text encoders and VAE. Token reduction changes computation, while these weights are retained. Quantized file size is a separate measure.</p>
  </div>;
}
