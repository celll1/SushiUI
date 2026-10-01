import type { MingGlobalSummaryInfo } from "@/utils/api";

export default function MingGlobalSummary({ info }: { info?: MingGlobalSummaryInfo | null }) {
  if (!info) return null;
  const spec = info.metadata;
  return <div className="my-2 rounded border border-violet-800 p-2 text-xs">
    <div role="img" aria-label="Full-canvas summaries connect to each target tile" className="flex flex-wrap items-center gap-2 text-violet-300">
      <span className="rounded bg-gray-800 p-1">All target frames</span><span>→</span>
      <span className="rounded bg-gray-800 p-1">{spec.summary_tokens} summaries · rank {spec.rank}</span><span>→</span>
      <span className="rounded bg-gray-800 p-1">Target tile · {spec.hidden_dim} features</span>
    </div>
    <p className="mt-2 text-gray-400">Independent global branch: {info.parameters.toLocaleString()} parameters / {(info.storage_bytes / 1024 ** 2).toFixed(3)} MiB floating tensors. {info.loaded_in_model ? "Included in loaded parameter counts." : "Stored separately; excluded from loaded parameter counts until partition generation or training uses it."} Full-frame generation ignores summaries.</p>
  </div>;
}
