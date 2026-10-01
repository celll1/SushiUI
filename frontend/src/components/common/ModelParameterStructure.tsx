"use client";

import type { ModelInfo } from "@/utils/api";

const labels: Record<string, string> = {
  dit: "DiT", language: "MLLM language", vision: "Vision + projector",
  connector: "Bidirectional connector", projections: "Conditioning projections",
  queries: "Learned queries", vae: "RGBA VAE",
};
const bytes = (value: number) => `${(value / 1024 ** 3).toFixed(3)} GiB`;

export default function ModelParameterStructure({ info }: { info: ModelInfo }) {
  const counts = info.component_parameter_counts;
  if (!counts) return null;
  const rows = Object.entries(counts);
  const total = rows.reduce((sum, [, row]) => sum + row.parameters, 0);
  const stored = Object.values(info.serialized_storage ?? {}).reduce((sum, row) => sum + row.storage_bytes, 0);
  return <details className="mt-2 w-full rounded border border-gray-700 p-2 text-xs">
    <summary className="cursor-pointer text-sky-300">Loaded component structure / {total.toLocaleString()} parameters</summary>
    <p className="my-2 text-gray-400">{info.profile} / {info.main_blocks} main blocks + {info.noise_refiner_blocks} image and {info.context_refiner_blocks} context refiners. Green: trainable; blue: frozen.</p>
    <div role="img" aria-label="Loaded model parameter distribution and training ownership">
      {rows.map(([name, row]) => <div key={name} className="my-2">
        <div className="flex flex-wrap justify-between gap-2"><span>{labels[name] ?? name}</span><span>{row.parameters.toLocaleString()} / trainable {row.trainable_parameters.toLocaleString()}</span></div>
        <div className="h-2 rounded bg-gray-800"><div className="h-2 overflow-hidden rounded bg-sky-600" style={{ width: `${total ? row.parameters / total * 100 : 0}%` }}>
          <div className="h-2 bg-emerald-500" style={{ width: `${row.parameters ? row.trainable_parameters / row.parameters * 100 : 0}%` }} />
        </div></div>
        <p className="mt-1 text-gray-500">Logical parameters: {bytes(row.logical_parameter_bytes)} / {Object.keys(row.bytes_by_dtype).join(", ")} / {Object.keys(row.logical_bytes_by_device).join(", ")}</p>
      </div>)}
    </div>
    <p className="text-gray-400">Serialized tensors: {bytes(stored)}. Logical parameter bytes exclude buffers and quantization payloads; these values are separate from measured VRAM. Fixed U-Net transport adds 0 parameters.</p>
    {info.generation_available === false && <p className="mt-2 text-amber-300">Native model loaded. Platform generation is not yet enabled.</p>}
  </details>;
}
