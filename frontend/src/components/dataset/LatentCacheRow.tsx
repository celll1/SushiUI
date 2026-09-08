"use client";

import { useCallback, useEffect, useState } from "react";
import { Database, Trash2 } from "lucide-react";
import {
  getDatasetLatentCache,
  deleteDatasetLatentCache,
  LatentCacheStatus,
  LatentCacheNamespace,
  LatentCacheDeleteResponse,
} from "@/utils/api";

export function formatCacheBytes(bytes: number): string {
  if (bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function nsLabel(ns: LatentCacheNamespace): string {
  if (ns.namespace && ns.vae_namespace) return `${ns.namespace} / ${ns.vae_namespace}`;
  return ns.path;
}

export default function LatentCacheRow({ datasetId }: { datasetId: number }) {
  const [status, setStatus] = useState<LatentCacheStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [pending, setPending] = useState<
    { preview: LatentCacheDeleteResponse; target?: LatentCacheNamespace } | null
  >(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setStatus(await getDatasetLatentCache(datasetId));
    } catch (err) {
      console.error("Failed to load latent cache status:", err);
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => {
    load();
  }, [load]);

  const preview = async (target?: LatentCacheNamespace) => {
    setMessage(null);
    try {
      const result = await deleteDatasetLatentCache(datasetId, {
        namespace: target?.namespace,
        vae_namespace: target?.vae_namespace,
        dry_run: true,
      });
      setPending({ preview: result, target });
    } catch (err: any) {
      setMessage(err?.response?.data?.detail || "Failed to inspect the cache");
    }
  };

  const confirm = async () => {
    if (!pending) return;
    setBusy(true);
    try {
      const result = await deleteDatasetLatentCache(datasetId, {
        namespace: pending.target?.namespace,
        vae_namespace: pending.target?.vae_namespace,
      });
      setMessage(
        `Deleted ${result.targets.length} namespace(s), ${result.total_entries} entries ` +
          `(${formatCacheBytes(result.total_bytes)})`
      );
      setPending(null);
      await load();
    } catch (err: any) {
      setMessage(err?.response?.data?.detail || "Delete failed");
    } finally {
      setBusy(false);
    }
  };

  const namespaces = status?.namespaces ?? [];

  return (
    <div className="mx-4 mt-2 bg-gray-900 rounded border border-gray-700 text-xs">
      <div className="flex items-center gap-2 px-3 py-2">
        <Database className="h-3.5 w-3.5 text-gray-400 shrink-0" />
        <span className="text-gray-300">Latent cache</span>
        <span className="text-gray-500">
          {loading
            ? "loading…"
            : namespaces.length === 0
            ? "none on disk — a run with Pre-Encoded Cache creates it at start"
            : `${namespaces.length} VAE namespace(s) · ${status?.total_entries} entries · ` +
              `${formatCacheBytes(status?.total_bytes ?? 0)} · ${status?.item_count} dataset items`}
        </span>
        <div className="ml-auto flex items-center gap-2">
          {namespaces.length > 0 && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-gray-300"
            >
              {expanded ? "Hide" : "Details"}
            </button>
          )}
          {namespaces.length > 0 && (
            <button
              onClick={() => preview()}
              className="px-2 py-1 bg-red-900/40 hover:bg-red-900/70 border border-red-800 rounded text-red-300 flex items-center gap-1"
            >
              <Trash2 className="h-3 w-3" />
              Delete all
            </button>
          )}
          <button
            onClick={load}
            className="px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-gray-400"
          >
            Refresh
          </button>
        </div>
      </div>

      {expanded && namespaces.length > 0 && (
        <div className="px-3 pb-2 space-y-1">
          <p className="text-gray-500">
            One directory per VAE that has encoded this dataset. Deleting one is how it is
            rebuilt: the next run using that VAE with Pre-Encoded Cache re-encodes it. Text
            embeddings live outside these directories and are never deleted here.
          </p>
          {namespaces.map((ns) => (
            <div
              key={ns.path}
              className="flex items-center gap-2 py-1 border-t border-gray-800"
            >
              <div className="min-w-0">
                <div className="text-gray-300 truncate" title={ns.path}>
                  {nsLabel(ns)}
                </div>
                <div className="text-gray-500">
                  {ns.entries} entries · {formatCacheBytes(ns.bytes)}
                  {ns.vae_family ? ` · VAE ${ns.vae_family}` : ""}
                  {ns.model_path ? ` · written by ${ns.model_path}` : ""}
                </div>
              </div>
              {ns.deletable === false ? (
                // Deleting this one would take the text embeddings beside it, so
                // the endpoint refuses. A Delete button here would drop both
                // query parameters and target every other namespace instead.
                <span className="ml-auto text-xs text-gray-500 shrink-0">
                  remove by hand
                </span>
              ) : (
                <button
                  onClick={() => preview(ns)}
                  className="ml-auto px-2 py-1 bg-red-900/40 hover:bg-red-900/70 border border-red-800 rounded text-red-300 shrink-0"
                >
                  Delete
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {pending && (
        <div className="px-3 pb-3 space-y-2">
          <div className="bg-red-900/20 border border-red-700 rounded p-2 space-y-1">
            <div className="text-red-300">
              Delete {pending.preview.targets.length} namespace(s) —{" "}
              {pending.preview.total_entries} entries,{" "}
              {formatCacheBytes(pending.preview.total_bytes)}. This cannot be undone.
            </div>
            {pending.preview.targets.map((t) => (
              <div key={t.path} className="text-gray-400 truncate" title={t.path}>
                {t.path}
              </div>
            ))}
            {pending.preview.active_runs.length > 0 && (
              <div className="text-yellow-400">
                Refused while these runs use this dataset:{" "}
                {pending.preview.active_runs.join(", ")}
              </div>
            )}
            <div className="flex gap-2 pt-1">
              <button
                onClick={confirm}
                disabled={busy || pending.preview.active_runs.length > 0}
                className="px-2 py-1 bg-red-700 hover:bg-red-600 rounded text-white disabled:opacity-50"
              >
                {busy ? "Deleting…" : "Delete"}
              </button>
              <button
                onClick={() => setPending(null)}
                className="px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-gray-300"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {message && <div className="px-3 pb-2 text-gray-400">{message}</div>}
    </div>
  );
}
