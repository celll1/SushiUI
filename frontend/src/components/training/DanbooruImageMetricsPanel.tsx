"use client";

import { useEffect, useState } from "react";
import { getTrainingDanbooruMetrics, DanbooruImageAugMetrics } from "@/utils/api";

interface Props {
  runId: number;
  active: boolean;  // poll only while training is active
}

/** Compact panel that polls /training/runs/{id}/danbooru-metrics every 3s and
 *  renders online Danbooru augmentation stats for image-generation training.
 *  Returns null when augmentation is disabled (no metrics file present). */
export default function DanbooruImageMetricsPanel({ runId, active }: Props) {
  const [data, setData] = useState<DanbooruImageAugMetrics | null>(null);
  const [tab, setTab] = useState<"top" | "recent">("top");

  useEffect(() => {
    let cancelled = false;
    const fetchOnce = async () => {
      try {
        const m = await getTrainingDanbooruMetrics(runId);
        if (!cancelled) setData(m);
      } catch {
        /* network glitch — keep stale */
      }
    };
    fetchOnce();
    if (!active) return () => { cancelled = true; };
    const timer = setInterval(fetchOnce, 3000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [runId, active]);

  if (!data || !data.enabled) return null;

  const bufPct =
    data.buffer_capacity && data.buffer_capacity > 0
      ? Math.min(100, ((data.buffer_current ?? 0) / data.buffer_capacity) * 100)
      : 0;

  const topTags = data.top_tags ?? [];
  const topMax = topTags.reduce((m, t) => Math.max(m, t.count), 0) || 1;
  const buckets = data.bucket_distribution ?? {};
  const bucketKeys = Object.keys(buckets).sort((a, b) => (buckets[b] ?? 0) - (buckets[a] ?? 0));

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Danbooru Augmentation</h3>
        <span className="text-xs text-gray-500">refreshed every 3s</span>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
        <Stat label="Collected" value={(data.total_collected ?? 0).toLocaleString()} />
        <Stat label="Injection batches" value={(data.total_injected_batches ?? 0).toLocaleString()} />
        <Stat label="Unique tags" value={(data.unique_tags_seen ?? 0).toLocaleString()} />
        <Stat label="Starvations" value={(data.buffer_starvation_count ?? 0).toLocaleString()} />
      </div>

      {/* Path breakdown */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <Stat label="Static collected" value={(data.static_collected ?? 0).toLocaleString()} />
        <Stat label="Deficiency collected" value={(data.deficiency_collected ?? 0).toLocaleString()} />
        <Stat label="Deficiency queries" value={(data.deficiency_query_count ?? 0).toLocaleString()} />
      </div>

      {/* Buffer fill bar */}
      <div>
        <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
          <span>Buffer</span>
          <span>{data.buffer_current ?? 0} / {data.buffer_capacity ?? "?"}</span>
        </div>
        <div className="h-2 bg-gray-800 rounded overflow-hidden">
          <div
            className={`h-full ${
              bufPct < 25 ? "bg-yellow-500" : bufPct < 75 ? "bg-blue-500" : "bg-green-500"
            }`}
            style={{ width: `${bufPct}%` }}
          />
        </div>
      </div>

      {/* Bucket distribution chips */}
      {bucketKeys.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {bucketKeys.map((k) => (
            <span
              key={k}
              className="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700 text-gray-300 text-xs"
              title={`${buckets[k]} buffered at ${k}`}
            >
              {k}: {buckets[k]}
            </span>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-700 text-xs">
        <TabBtn label="Top tags" active={tab === "top"} onClick={() => setTab("top")} />
        <TabBtn label="Recent posts" active={tab === "recent"} onClick={() => setTab("recent")} />
      </div>

      {/* Top tags */}
      {tab === "top" && (
        <div className="max-h-60 overflow-y-auto space-y-0.5 pr-1">
          {topTags.slice(0, 50).map((t) => (
            <div key={t.tag} className="flex items-center gap-2 text-xs">
              <span className="w-40 truncate text-gray-200" title={t.tag}>{t.tag}</span>
              <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                <div className="h-full bg-blue-500" style={{ width: `${(t.count / topMax) * 100}%` }} />
              </div>
              <span className="w-10 text-right text-gray-500 font-mono">{t.count}</span>
            </div>
          ))}
          {topTags.length === 0 && (
            <p className="text-xs text-gray-500 italic">No tags collected yet.</p>
          )}
        </div>
      )}

      {/* Recent posts */}
      {tab === "recent" && (
        <div className="max-h-60 overflow-y-auto space-y-1 pr-1">
          {[...(data.recent_posts ?? [])].reverse().slice(0, 30).map((p) => (
            <div key={p.post_id} className="text-xs border-b border-gray-800 pb-1">
              <div className="flex items-center justify-between text-gray-400">
                <a
                  href={`https://danbooru.donmai.us/posts/${p.post_id}`}
                  target="_blank" rel="noopener noreferrer"
                  className="text-blue-400 hover:underline font-mono"
                >
                  #{p.post_id}
                </a>
                <span className="text-gray-600">
                  {p.tag_count} tags{p.path ? ` · ${p.path}` : ""}
                </span>
              </div>
              <div className="text-gray-300 mt-0.5">
                {p.tags.slice(0, 12).join(", ")}{p.tags.length > 12 ? "…" : ""}
              </div>
            </div>
          ))}
          {(data.recent_posts ?? []).length === 0 && (
            <p className="text-xs text-gray-500 italic">No posts collected yet.</p>
          )}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-800 rounded px-2 py-1.5">
      <div className="text-gray-500 text-[10px] uppercase tracking-wide">{label}</div>
      <div className="text-gray-100 font-semibold">{value}</div>
    </div>
  );
}

function TabBtn({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 transition-colors ${
        active ? "text-blue-400 border-b-2 border-blue-400 -mb-px" : "text-gray-400 hover:text-gray-200"
      }`}
    >
      {label}
    </button>
  );
}
