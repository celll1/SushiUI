import { useEffect, useState } from "react";
import { ConceptExposureResponse, ConceptExposureRow, getTrainingConceptExposure } from "@/utils/api";

interface Props {
  runId: number;
  active: boolean;
}

function TimestepHeatmap({ row, edges, maximum }: {
  row: ConceptExposureRow; edges?: number[]; maximum: number;
}) {
  const bins = row.timestep_counts;
  const recorded = row.timestep_recorded_passes ?? 0;
  const unknown = row.timestep_unknown_passes ?? row.sample_passes;
  if (!edges || !bins || bins.length !== edges.length - 1 || !recorded) {
    return <div className="h-4 rounded bg-gray-800 text-center text-[10px] text-gray-500"
      title={`${unknown.toLocaleString()} passes have no recorded timestep`}>No timestep data</div>;
  }
  const target = row.target_images || 1;
  const density = bins.map((count, index) => count / target / (edges[index + 1] - edges[index]));
  const color = (value: number) => {
    if (!value) return "rgb(31,41,55)";
    const level = Math.log1p(value) / Math.log1p(maximum || 1);
    return `hsl(${215 - 170 * level} 85% ${25 + 35 * level}%)`;
  };
  const stops = density.map((value, index) => `${color(value)} ${(edges[index] + edges[index + 1]) * 50}%`);
  const high = bins.reduce((sum, count, index) => sum + (edges[index] >= 0.95 ? count : 0), 0);
  const summary = `${recorded.toLocaleString()} recorded; ${unknown.toLocaleString()} unknown; t >= 0.95: ${high.toLocaleString()} passes (${row.target_images ? (high / target).toFixed(2) : "n/a"} per image). Brightness: passes/image/bin width, scaled across visible rows.`;
  return (
    <div className="relative flex h-4 overflow-hidden rounded" role="img" aria-label={summary}
      title={summary} tabIndex={0} style={{ background: `linear-gradient(to right, ${stops.join(",")})` }}>
      {bins.map((count, index) => (
        <span key={index} style={{ width: `${(edges[index + 1] - edges[index]) * 100}%` }}
          title={`t ${edges[index].toFixed(2)}–${edges[index + 1].toFixed(2)}: ${count.toLocaleString()} passes; ${row.target_images ? (count / target).toFixed(2) : "n/a"} per image. ${unknown.toLocaleString()} historical/unknown passes excluded.`} />
      ))}
      {[0.8, 0.95].map((value) => <span key={value} className="pointer-events-none absolute inset-y-0 border-l border-white/40"
        style={{ left: `${value * 100}%` }} />)}
    </div>
  );
}

export default function ConceptExposurePanel({ runId, active }: Props) {
  const [order, setOrder] = useState<"latest" | "top">("latest");
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const [data, setData] = useState<ConceptExposureResponse | null>(null);

  useEffect(() => { setData(null); }, [runId]);
  useEffect(() => {
    const timer = setTimeout(() => setQuery(search.trim()), 350);
    return () => clearTimeout(timer);
  }, [search]);

  useEffect(() => {
    let cancelled = false;
    let pending = false;
    const controller = new AbortController();
    const fetchOnce = async () => {
      if (pending || cancelled) return;
      pending = true;
      try {
        const response = await getTrainingConceptExposure(runId, order, query, controller.signal);
        if (!cancelled) setData(response);
      } catch {
        // Keep the last published snapshot during a transient read failure.
      } finally { pending = false; }
    };
    fetchOnce();
    if (!active) return () => { cancelled = true; controller.abort(); };
    const timer = setInterval(fetchOnce, 5000);
    return () => { cancelled = true; controller.abort(); clearInterval(timer); };
  }, [runId, order, active, query]);

  if (!data?.enabled) return null;
  const edges = data.timestep_bin_edges;
  const maximum = Math.max(1, ...data.rows.flatMap((row) =>
    edges && row.timestep_counts?.length === edges.length - 1
      ? row.timestep_counts.map((count, index) => count / (row.target_images || 1) / (edges[index + 1] - edges[index]))
      : []));

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
      <input type="search" value={search} onChange={(event) => setSearch(event.target.value)}
        maxLength={256} placeholder="Search concept tags or priority entries…" aria-label="Search concept exposure"
        className="w-full rounded border border-gray-600 bg-gray-800 px-2 py-1 text-xs text-gray-200"
      />
      {(search.trim() !== query || (data.query !== undefined && data.query !== query)) &&
        <p className="text-[10px] text-gray-500">Updating search…</p>}
      <p className="text-xs text-gray-500">
        {data.mode === "priority" ? "Priority entries" : "Concept tags"} · {data.total_groups ?? 0} trained{query && ` · ${data.matched_groups ?? data.rows.length} matches`}
      </p>
      <p className="text-[10px] text-gray-500" title="Brightness is normalized exposure density across visible rows. Marks: 0.8 and 0.95. Hover each interval for counts; earlier passes remain unknown.">
        Timestep heatmap: clean 0 → noisy 1 · marks at 0.8 / 0.95 · hover for counts
      </p>
      <div className="max-h-56 overflow-auto space-y-1 text-xs">
        {data.rows.length === 0 && <p className="text-gray-500">No matching concepts.</p>}
        <div className="grid min-w-[600px] grid-cols-[minmax(0,1fr)_180px_max-content] gap-x-3">
          {(data.rows ?? []).map((row) => (
            <div key={row.key} className="col-span-3 grid grid-cols-subgrid items-center gap-x-3 border-b border-gray-800 py-1">
              <span className="min-w-0 truncate text-gray-200" title={row.name}>{row.name}</span>
              <TimestepHeatmap row={row} edges={edges} maximum={maximum} />
              <span className="whitespace-nowrap text-right tabular-nums text-gray-400" title={`${row.target_images.toLocaleString()} target images; last step ${row.last_step.toLocaleString()}`}>
                {row.sample_passes.toLocaleString()} passes · {row.mean_passes_per_image.toFixed(2)}×/image
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
