import { useEffect, useState } from "react";
import { ConceptExposureResponse, getTrainingConceptExposure } from "@/utils/api";

interface Props {
  runId: number;
  active: boolean;
}

export default function ConceptExposurePanel({ runId, active }: Props) {
  const [order, setOrder] = useState<"latest" | "top">("latest");
  const [data, setData] = useState<ConceptExposureResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetchOnce = async () => {
      try {
        const response = await getTrainingConceptExposure(runId, order);
        if (!cancelled) setData(response);
      } catch {
        // Keep the last published snapshot during a transient read failure.
      }
    };
    fetchOnce();
    if (!active) return () => { cancelled = true; };
    const timer = setInterval(fetchOnce, 5000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [runId, order, active]);

  if (!data?.enabled) return null;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-gray-200">Concept exposure</h3>
        <select
          value={order}
          onChange={(event) => setOrder(event.target.value as "latest" | "top")}
          aria-label="Concept exposure order"
          className="bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-gray-200"
        >
          <option value="latest">Latest</option>
          <option value="top">Most passes</option>
        </select>
      </div>
      <p className="text-xs text-gray-500">
        {data.mode === "priority" ? "Priority entries" : "Concept tags"} · {data.total_groups ?? 0} trained
      </p>
      <div className="max-h-56 overflow-y-auto space-y-1 text-xs">
        {(data.rows ?? []).map((row) => (
          <div key={row.key} className="flex justify-between gap-2 border-b border-gray-800 pb-1">
            <span className="min-w-0 truncate text-gray-200" title={row.name}>{row.name}</span>
            <span className="shrink-0 text-gray-400" title={`${row.target_images.toLocaleString()} target images; last step ${row.last_step.toLocaleString()}`}>
              {row.sample_passes.toLocaleString()} passes · {row.mean_passes_per_image.toFixed(2)}×/image
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
