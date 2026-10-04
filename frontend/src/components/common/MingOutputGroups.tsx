"use client";

import { GenerationPanelId, useGenerationQueue } from "@/contexts/GenerationQueueContext";
import { stripCacheBuster } from "@/utils/previewStorage";

interface MingOutputGroupsProps {
  panel: GenerationPanelId;
  selectedUrl: string | null;
  disabled: boolean;
  onSelect: (url: string, seed: number) => void;
}

export default function MingOutputGroups({ panel, selectedUrl, disabled, onSelect }: MingOutputGroupsProps) {
  const { completedResults } = useGenerationQueue();
  const snapshot = completedResults[panel];
  const groups = snapshot?.kind === "image" ? snapshot.imageGroups : undefined;
  const selected = selectedUrl ? stripCacheBuster(selectedUrl) : null;
  if (!groups?.length || !selected || !groups.some(group =>
    group.filenames.some(filename => `/outputs/${filename}` === selected))) return null;
  return <section aria-label="Ming output selection" className="max-h-48 shrink-0 overflow-y-auto rounded border border-sky-800 p-2 text-sm">
    <h3 className="text-sky-200">Ming outputs</h3>
    {groups.map(group => <div key={group.group_id} className="mt-2">
      <p className="text-xs text-gray-400">Group {group.batch_index + 1} · seed {group.seed}</p>
      <div className="mt-2 flex flex-wrap gap-2">
        {group.filenames.map((filename, index) => {
          const role = group.roles[index];
          const label = role === "auxiliary_canvas" ? "Auxiliary canvas" : role?.replace("layer_", "Layer ") || "Image";
          const url = `/outputs/${filename}`;
          const active = selected === url;
          return <div key={filename} className="w-24">
            <button type="button" disabled={disabled} aria-pressed={active}
              aria-label={`Show ${label} in preview`}
              onClick={() => onSelect(url, group.seed)}
              className={`w-full rounded border p-1 disabled:opacity-50 ${active ? "border-sky-400 bg-sky-950" : "border-gray-700 bg-gray-900"}`}>
              <img src={url} alt={label} loading="lazy" className="h-16 w-full object-contain" />
              <span className="text-xs text-sky-200">{label}</span>
            </button>
            <a href={url} download={filename} className="block text-center text-xs text-gray-400">RGBA PNG</a>
          </div>;
        })}
      </div>
    </div>)}
  </section>;
}
