import { useEffect, useState } from "react";
import { ConceptExposureResponse, ConceptExposureRow, getTrainingConceptExposure } from "@/utils/api";

interface Props {
  runId: number;
  active: boolean;
}

function TimestepHistogram({ row, edges, maximum, lossRange, maximumStddev }: {
  row: ConceptExposureRow; edges?: number[]; maximum: number; lossRange: [number, number] | null; maximumStddev: number;
}) {
  const bins = row.timestep_counts;
  const recorded = row.timestep_recorded_passes ?? 0;
  const unknown = row.timestep_unknown_passes ?? row.sample_passes;
  if (!edges || !bins || bins.length !== edges.length - 1 || !recorded) {
    return <div className="h-6 rounded bg-gray-800 text-center text-[10px] leading-6 text-gray-500"
      title={`${unknown.toLocaleString()} passes have no recorded timestep`}>No timestep data</div>;
  }
  const target = row.target_images || 1;
  const density = bins.map((count, index) => count / (edges[index + 1] - edges[index]));
  const color = (index: number) => {
    const loss = row.timestep_loss_means?.[index];
    if (loss == null || !lossRange) return "rgb(107,114,128)";
    const [low, high] = lossRange;
    const level = high > low ? (loss - low) / (high - low) : 0.5;
    const stddev = row.timestep_loss_stddevs?.[index];
    const saturation = stddev != null && maximumStddev > 0 ? 1 - 0.75 * stddev / maximumStddev : 1;
    const channel = (value: number) => 205 + (value - 205) * saturation;
    return `rgb(${channel(59 + 190 * level)}, ${channel(130 - 15 * level)}, ${channel(246 - 224 * level)})`;
  };
  const high = bins.reduce((sum, count, index) => sum + (edges[index] >= 0.95 ? count : 0), 0);
  const summary = `${recorded.toLocaleString()} timesteps recorded; ${unknown.toLocaleString()} unknown; ${(row.timestep_loss_recorded_passes ?? 0).toLocaleString()} losses recorded; t >= 0.95: ${high.toLocaleString()} passes (${row.target_images ? (high / target).toFixed(2) : "n/a"} per image). Height: passes/bin width; color: mean loss; paler: greater standard deviation, shared scales across visible rows. Stripes: standard deviation unavailable. Gray: loss unrecorded.`;
  return (
    <div className="relative flex h-6 items-end overflow-hidden rounded bg-gray-800" role="img" aria-label={summary}
      title={summary} tabIndex={0}>
      {bins.map((count, index) => (
        <span key={index} className="relative h-full shrink-0"
          style={{ width: `${(edges[index + 1] - edges[index]) * 100}%` }}
          title={`t ${edges[index].toFixed(2)}–${edges[index + 1].toFixed(2)}: ${count.toLocaleString()} passes; ${row.target_images ? (count / target).toFixed(2) : "n/a"} per image. Mean loss: ${row.timestep_loss_means?.[index]?.toPrecision(4) ?? "unrecorded"} (${row.timestep_loss_counts?.[index] ?? 0}/${count} passes). Stddev: ${row.timestep_loss_stddevs?.[index]?.toPrecision(4) ?? "unavailable (needs 2 observations)"}; variance cohort: ${row.timestep_loss_variance_counts?.[index] ?? 0} passes, mean ${row.timestep_loss_variance_means?.[index]?.toPrecision(4) ?? "unrecorded"}. Historical mean-only losses excluded from variance. ${unknown.toLocaleString()} historical/unknown timesteps excluded.`}>
          <span className="pointer-events-none absolute inset-x-0 bottom-0"
            style={{ height: `${density[index] / maximum * 100}%`, backgroundColor: "rgb(107,114,128)" }}>
            <span className="absolute inset-x-0 top-0"
              style={{ height: `${count ? (row.timestep_loss_counts?.[index] ?? 0) / count * 100 : 0}%`, backgroundColor: color(index),
                backgroundImage: row.timestep_loss_stddevs?.[index] == null
                  ? "repeating-linear-gradient(45deg, transparent 0px, transparent 3px, rgba(255,255,255,0.35) 3px, rgba(255,255,255,0.35) 4px)" : undefined }} />
          </span>
        </span>
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
      ? row.timestep_counts.map((count, index) => count / (edges[index + 1] - edges[index]))
      : []));
  const losses = data.rows.flatMap((row) => row.timestep_loss_means ?? [])
    .filter((value): value is number => value != null && Number.isFinite(value));
  const lossRange: [number, number] | null = losses.length ? [Math.min(...losses), Math.max(...losses)] : null;
  const stddevs = data.rows.flatMap((row) => row.timestep_loss_stddevs ?? [])
    .filter((value): value is number => value != null && Number.isFinite(value));
  const maximumStddev = Math.max(0, ...stddevs);

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
      <p className="text-[10px] text-gray-500" title="Height is passes per timestep unit; bar area is proportional to recorded passes. Hue is mean per-image prediction/reconstruction loss, including guidance weighting but excluding auxiliary losses. Greater sample standard deviation reduces saturation. Both scales are shared across visible rows; variance excludes historical mean-only losses.">
        Timestep: clean 0 → noisy 1 · area = passes · marks at 0.8 / 0.95 · gray = no loss
      </p>
      <p className="text-[10px] text-gray-500">
        Paler = higher stddev{stddevs.length > 0 && ` (0–${maximumStddev.toPrecision(3)}, visible rows)`} · stripes = stddev unavailable
      </p>
      <p className="flex items-center gap-2 text-[10px] text-gray-500">
        Mean loss {lossRange ? <><span className="text-blue-400">{lossRange[0].toPrecision(3)}</span>
          <span className="h-2 w-20 rounded" style={{ background: "linear-gradient(to right, rgb(59,130,246), rgb(249,115,22))" }} />
          <span className="text-orange-400">{lossRange[1].toPrecision(3)}</span> · visible rows</> : "unrecorded"}
      </p>
      <div className="max-h-56 overflow-auto space-y-1 text-xs">
        {data.rows.length === 0 && <p className="text-gray-500">No matching concepts.</p>}
        <div className="grid min-w-[680px] grid-cols-[minmax(0,1fr)_180px_max-content_max-content] gap-x-3">
          {(data.rows ?? []).map((row) => (
            <div key={row.key} className="col-span-4 grid grid-cols-subgrid items-center gap-x-3 border-b border-gray-800 py-1">
              <span className="min-w-0 truncate text-gray-200" title={row.name}>{row.name}</span>
              <TimestepHistogram row={row} edges={edges} maximum={maximum} lossRange={lossRange} maximumStddev={maximumStddev} />
              <span className="whitespace-nowrap text-right tabular-nums text-gray-400"
                title={`${(row.timestep_unknown_passes ?? row.sample_passes).toLocaleString()} passes have no recorded timestep; loss recorded for ${(row.timestep_loss_recorded_passes ?? 0).toLocaleString()}/${row.sample_passes.toLocaleString()} passes`}>
                t recorded: {(row.timestep_recorded_passes ?? 0).toLocaleString()}
              </span>
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
