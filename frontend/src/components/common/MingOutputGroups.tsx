"use client";

import { useEffect, useState } from "react";
import { GenerationPanelId, useGenerationQueue } from "@/contexts/GenerationQueueContext";
import ImageViewer from "./ImageViewer";

export default function MingOutputGroups({ panel }: { panel: GenerationPanelId }) {
  const { completedResults } = useGenerationQueue();
  const snapshot = completedResults[panel];
  const groups = snapshot?.kind === "image" ? snapshot.imageGroups : undefined;
  const [selected, setSelected] = useState<string | null>(null);
  useEffect(() => { setSelected(null); }, [snapshot?.revision]);
  if (!groups?.length) return null;
  return <section className="rounded border border-sky-800 p-3 text-sm">
    <h3 className="text-sky-200">Ming ordered RGBA outputs</h3>
    {groups.map(group => <div key={group.group_id} className="mt-2">
      <p className="text-xs text-gray-400">{group.profile} · group {group.batch_index + 1} · seed {group.seed}</p>
      <div className="mt-2 flex flex-wrap gap-2">
        {group.filenames.map((filename, index) => {
          const role = group.roles[index];
          const label = role === "auxiliary_canvas" ? "Auxiliary canvas" : role?.replace("layer_", "Layer ") || "Image";
          return <div key={filename} className="w-28">
            <button onClick={() => setSelected(`/outputs/${filename}`)} className="w-full rounded bg-gray-900 p-1" title={label}>
              <img src={`/outputs/${filename}`} alt={label} loading="lazy" className="h-24 w-full object-contain" />
              <span className="text-xs text-sky-200">{label}</span>
            </button>
            <a href={`/outputs/${filename}`} download={filename} className="block text-center text-xs text-gray-400">RGBA PNG</a>
          </div>;
        })}
      </div>
    </div>)}
    {selected && <ImageViewer imageUrl={selected} kind="image" onClose={() => setSelected(null)} />}
  </section>;
}
