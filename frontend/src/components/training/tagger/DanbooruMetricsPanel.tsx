"use client";

import { useEffect, useState } from "react";
import { getTaggerDanbooruMetrics, DanbooruAugmentationMetrics } from "@/utils/api";

interface Props {
  runId: string;
  active: boolean;  // poll only while training is active
}

/** Compact panel that polls /tagger-training/runs/{id}/danbooru-metrics every 3s
 *  and renders collection stats + top tags + recent posts. Returns null when
 *  augmentation is disabled (no metrics file present). */
export default function DanbooruMetricsPanel({ runId, active }: Props) {
  const [data, setData] = useState<DanbooruAugmentationMetrics | null>(null);
  const [tab, setTab] = useState<"top" | "queries" | "new" | "lowf1" | "traincount" | "cooc" | "recent">("top");

  useEffect(() => {
    let cancelled = false;
    const fetchOnce = async () => {
      try {
        const m = await getTaggerDanbooruMetrics(runId);
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

  const topMax = (data.top_tags ?? []).reduce((m, t) => Math.max(m, t.count), 0) || 1;
  const newTags = data.top_dynamic_tags ?? [];
  const newMax = newTags.reduce((m, t) => Math.max(m, t.count), 0) || 1;
  const hasNewTags = (data.dynamic_tags_count ?? 0) > 0;

  const lowF1Tags = data.top_low_f1_tags ?? [];
  const lowF1Max = lowF1Tags.reduce((m, t) => Math.max(m, t.count), 0) || 1;
  const hasLowF1 =
    (data.low_f1_tags_count ?? 0) > 0 ||
    (data.total_low_f1_collected ?? 0) > 0 ||
    lowF1Tags.length > 0;

  const trainCountTags = data.top_train_count_tags ?? [];
  const trainCountMax = trainCountTags.reduce((m, t) => Math.max(m, t.count), 0) || 1;
  const hasTrainCount =
    (data.train_count_tags_count ?? 0) > 0 ||
    (data.total_train_count_collected ?? 0) > 0 ||
    trainCountTags.length > 0;

  const coocTags = data.cooc_proposed_tags ?? [];
  const hasCooc =
    (data.total_cooc_proposed ?? 0) > 0 ||
    (data.cooc_pending_count ?? 0) > 0 ||
    coocTags.length > 0;

  // Query mode: per-tag resolved collection (expand) and/or per-string static.
  const queryTags = data.top_query_tags ?? [];
  const staticQueries = data.top_static_queries ?? [];
  const queryMax = queryTags.reduce((m, t) => Math.max(m, t.count), 0) || 1;
  const staticMax = staticQueries.reduce((m, t) => Math.max(m, t.count), 0) || 1;
  const hasQuery =
    (data.query_tags_count ?? 0) > 0 ||
    (data.total_query_collected ?? 0) > 0 ||
    (data.total_static_collected ?? 0) > 0 ||
    queryTags.length > 0 ||
    staticQueries.length > 0;

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

      {/* New-tag (dynamic query) stats — only when vocab expansion is active */}
      {hasNewTags && (
        <div className="grid grid-cols-3 gap-2 text-xs">
          <Stat label="New tags targeted" value={(data.dynamic_tags_count ?? 0).toLocaleString()} />
          <Stat label="New tags collected" value={(data.dynamic_unique_tags_collected ?? 0).toLocaleString()} />
          <Stat label="New-tag posts" value={(data.total_dynamic_collected ?? 0).toLocaleString()} />
        </div>
      )}

      {/* Query mode stats — per-tag resolved collection + vocab expansion */}
      {hasQuery && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <Stat label="Query tags (pool)" value={(data.query_tags_count ?? 0).toLocaleString()} />
          <Stat label="Expanded via query" value={(data.query_expanded_count ?? 0).toLocaleString()} />
          <Stat label="Query posts (per-tag)" value={(data.total_query_collected ?? 0).toLocaleString()} />
          <Stat label="Static posts (per-query)" value={(data.total_static_collected ?? 0).toLocaleString()} />
        </div>
      )}

      {/* Low-F1 deficiency stats — only when low-F1 collection is active */}
      {hasLowF1 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <Stat label="Low-F1 targeted" value={(data.low_f1_tags_count ?? 0).toLocaleString()} />
          <Stat label="Low-F1 collected" value={(data.low_f1_unique_tags_collected ?? 0).toLocaleString()} />
          <Stat label="Low-F1 posts" value={(data.total_low_f1_collected ?? 0).toLocaleString()} />
          <Stat label="Unavailable" value={(data.low_f1_unavailable_count ?? 0).toLocaleString()} />
        </div>
      )}

      {/* Train-count deficiency (exposure balancing) stats — only when active */}
      {hasTrainCount && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <Stat label="Train-count targeted" value={(data.train_count_tags_count ?? 0).toLocaleString()} />
          <Stat label="Train-count collected" value={(data.train_count_unique_tags_collected ?? 0).toLocaleString()} />
          <Stat label="Train-count posts" value={(data.total_train_count_collected ?? 0).toLocaleString()} />
          <Stat label="Unavailable" value={(data.train_count_unavailable_count ?? 0).toLocaleString()} />
        </div>
      )}

      {/* Co-occurrence vocab-discovery + active-collection stats — only when active */}
      {((data.total_cooc_proposed ?? 0) > 0 || (data.cooc_pending_count ?? 0) > 0 || (data.cooc_active_count ?? 0) > 0) && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <Stat label="Co-occur added" value={(data.total_cooc_proposed ?? 0).toLocaleString()} />
          <Stat label="Co-occur pending" value={(data.cooc_pending_count ?? 0).toLocaleString()} />
          <Stat label="Co-occur active" value={(data.cooc_active_count ?? 0).toLocaleString()} />
          <Stat label="Co-occur posts" value={(data.total_cooc_collected ?? 0).toLocaleString()} />
        </div>
      )}

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

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-700 text-xs">
        <TabBtn label="Top tags" active={tab === "top"} onClick={() => setTab("top")} />
        {hasQuery && (
          <TabBtn label="Queries" active={tab === "queries"} onClick={() => setTab("queries")} />
        )}
        {hasNewTags && (
          <TabBtn label="New tags" active={tab === "new"} onClick={() => setTab("new")} />
        )}
        {hasLowF1 && (
          <TabBtn label="Low-F1 tags" active={tab === "lowf1"} onClick={() => setTab("lowf1")} />
        )}
        {hasTrainCount && (
          <TabBtn label="Train-count tags" active={tab === "traincount"} onClick={() => setTab("traincount")} />
        )}
        {hasCooc && (
          <TabBtn label="Co-occur tags" active={tab === "cooc"} onClick={() => setTab("cooc")} />
        )}
        <TabBtn label="Recent posts" active={tab === "recent"} onClick={() => setTab("recent")} />
      </div>

      {/* Top tags */}
      {tab === "top" && (
        <div className="max-h-60 overflow-y-auto space-y-0.5 pr-1">
          {(data.top_tags ?? []).slice(0, 50).map((t) => (
            <div key={t.tag} className="flex items-center gap-2 text-xs">
              <span className="w-40 truncate text-gray-200" title={t.tag}>{t.tag}</span>
              <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                <div className="h-full bg-blue-500" style={{ width: `${(t.count / topMax) * 100}%` }} />
              </div>
              <span className="w-10 text-right text-gray-500 font-mono">{t.count}</span>
            </div>
          ))}
          {(data.top_tags ?? []).length === 0 && (
            <p className="text-xs text-gray-500 italic">No tags collected yet.</p>
          )}
        </div>
      )}

      {/* Queries — per-resolved-tag collection (expand mode) and/or per-query-string
          collection (legacy static). Shows which query-focused tags/queries
          augmentation is actually gathering, like New-tag / Low-F1. */}
      {tab === "queries" && (
        <div className="max-h-60 overflow-y-auto space-y-2 pr-1">
          {queryTags.length > 0 && (
            <div className="space-y-0.5">
              <p className="text-xs text-gray-400 font-semibold">Resolved tags (per-tag)</p>
              {queryTags.slice(0, 50).map((t) => (
                <div key={`q-${t.tag}`} className="flex items-center gap-2 text-xs">
                  <span className="w-40 truncate text-gray-200" title={t.tag}>{t.tag}</span>
                  <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                    <div className="h-full bg-purple-500" style={{ width: `${(t.count / queryMax) * 100}%` }} />
                  </div>
                  <span className="w-10 text-right text-gray-500 font-mono">{t.count}</span>
                </div>
              ))}
            </div>
          )}
          {staticQueries.length > 0 && (
            <div className="space-y-0.5">
              <p className="text-xs text-gray-400 font-semibold">Query strings (per-query)</p>
              {staticQueries.slice(0, 50).map((t) => (
                <div key={`s-${t.tag}`} className="flex items-center gap-2 text-xs">
                  <span className="w-40 truncate text-gray-200" title={t.tag}>{t.tag}</span>
                  <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                    <div className="h-full bg-indigo-500" style={{ width: `${(t.count / staticMax) * 100}%` }} />
                  </div>
                  <span className="w-10 text-right text-gray-500 font-mono">{t.count}</span>
                </div>
              ))}
            </div>
          )}
          {queryTags.length === 0 && staticQueries.length === 0 && (
            <p className="text-xs text-gray-500 italic">No query samples collected yet.</p>
          )}
        </div>
      )}

      {/* New tags — per-targeted-new-tag collected sample counts. Surfaces which
          surveyor-approved new/deficient tags augmentation is actively gathering,
          independent of the ever-dominant common tags in the Top-tags view. */}
      {tab === "new" && (
        <div className="max-h-60 overflow-y-auto space-y-0.5 pr-1">
          {newTags.slice(0, 50).map((t) => (
            <div key={t.tag} className="flex items-center gap-2 text-xs">
              <span className="w-40 truncate text-gray-200" title={t.tag}>{t.tag}</span>
              <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                <div className="h-full bg-emerald-500" style={{ width: `${(t.count / newMax) * 100}%` }} />
              </div>
              <span className="w-10 text-right text-gray-500 font-mono">{t.count}</span>
            </div>
          ))}
          {newTags.length === 0 && (
            <p className="text-xs text-gray-500 italic">No new-tag samples collected yet.</p>
          )}
        </div>
      )}

      {/* Low-F1 tags — per-targeted-low-F1-tag collected sample counts. Surfaces
          which deficient existing vocab tags augmentation is gathering extra
          samples for (driven by the trainer's per-tag F1). */}
      {tab === "lowf1" && (
        <div className="max-h-60 overflow-y-auto space-y-0.5 pr-1">
          {lowF1Tags.slice(0, 50).map((t) => (
            <div key={t.tag} className="flex items-center gap-2 text-xs">
              <span className="w-40 truncate text-gray-200" title={t.tag}>{t.tag}</span>
              <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                <div className="h-full bg-amber-500" style={{ width: `${(t.count / lowF1Max) * 100}%` }} />
              </div>
              <span className="w-10 text-right text-gray-500 font-mono">{t.count}</span>
            </div>
          ))}
          {lowF1Tags.length === 0 && (
            <p className="text-xs text-gray-500 italic">No low-F1 samples collected yet.</p>
          )}
        </div>
      )}

      {/* Train-count tags — under-exposed tags (low cumulative training count
          vs current per-epoch rate) being rebalanced. Per-tag collected counts. */}
      {tab === "traincount" && (
        <div className="max-h-60 overflow-y-auto space-y-0.5 pr-1">
          {trainCountTags.slice(0, 50).map((t) => (
            <div key={t.tag} className="flex items-center gap-2 text-xs">
              <span className="w-40 truncate text-gray-200" title={t.tag}>{t.tag}</span>
              <div className="flex-1 h-2 bg-gray-800 rounded overflow-hidden">
                <div className="h-full bg-teal-500" style={{ width: `${(t.count / trainCountMax) * 100}%` }} />
              </div>
              <span className="w-10 text-right text-gray-500 font-mono">{t.count}</span>
            </div>
          ))}
          {trainCountTags.length === 0 && (
            <p className="text-xs text-gray-500 italic">No train-count samples collected yet (needs ≥2 epochs).</p>
          )}
        </div>
      )}

      {/* Co-occurrence tags — vocab-absent tags that co-occurred frequently
          enough across collected posts to be promoted into the vocab head.
          Unlike new/low-F1 tags these are not actively queried (no per-tag
          collection count), so we list names only, most-recently promoted
          first (bounded to the latest 200). */}
      {tab === "cooc" && (
        <div className="max-h-60 overflow-y-auto pr-1">
          {coocTags.length > 0 ? (
            <div className="flex flex-wrap gap-1">
              {coocTags.map((t) => (
                <span
                  key={t}
                  className="px-1.5 py-0.5 rounded bg-purple-900/50 border border-purple-700 text-purple-200 text-xs truncate max-w-[12rem]"
                  title={t}
                >
                  {t}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-xs text-gray-500 italic">No co-occurrence tags promoted yet.</p>
          )}
        </div>
      )}

      {/* Recent posts */}
      {tab === "recent" && (
        <div className="max-h-60 overflow-y-auto space-y-1 pr-1">
          {[...(data.recent_posts ?? [])].reverse().slice(0, 30).map((p) => (
            <div key={p.post_id + "_" + p.timestamp} className="text-xs border-b border-gray-800 pb-1">
              <div className="flex items-center justify-between text-gray-400">
                <a
                  href={`https://danbooru.donmai.us/posts/${p.post_id}`}
                  target="_blank" rel="noopener noreferrer"
                  className="text-blue-400 hover:underline font-mono"
                >
                  #{p.post_id}
                </a>
                <span className="text-gray-600">
                  {p.tag_count} tags · {Math.max(0, Math.round((Date.now() / 1000) - p.timestamp))}s ago
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
