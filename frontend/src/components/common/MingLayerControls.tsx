"use client";

import { useStartup } from "@/contexts/StartupContext";

export default function MingLayerControls({ value, onChange }: { value?: number; onChange: (value: number) => void }) {
  const { modelInfo, generationDefaults } = useStartup();
  if (modelInfo?.type !== "ming_image_01" || modelInfo.profile !== "layer_decompose") return null;
  const layers = value ?? generationDefaults?.txt2img.ming_image_01_layers;
  if (typeof layers !== "number") return <p className="text-xs text-gray-400">Loading layer defaults…</p>;
  return <fieldset className="rounded border border-sky-800 p-3 text-sm">
    <legend className="px-1 text-sky-300">Ming Layer decomposition</legend>
    <label className="flex items-center gap-3">Ordered RGBA layers
      <input type="number" min={1} step={1} value={layers} className="w-20 rounded bg-gray-800 px-2 py-1"
        onChange={event => { const next = Number(event.target.value); if (Number.isInteger(next) && next >= 1) onChange(next); }} />
    </label>
    <p className="mt-2 text-xs text-gray-400">Reference image required. Output: one native auxiliary canvas followed by {layers} ordered RGBA layers. The canvas role is retained separately.</p>
  </fieldset>;
}
