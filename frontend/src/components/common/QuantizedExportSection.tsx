"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Button from "./Button";
import Card from "./Card";
import Input from "./Input";
import {
  getQuantizedExportStatus,
  startQuantizedExport,
  type QuantizedExportStatus,
} from "@/utils/api";

interface QuantizedExportSectionProps {
  /** Loaded model identity inputs; used only to re-read status on a model change. */
  arch?: string | null;
  modelInfoVersion?: number;
  modelLoadRevision?: number;
  storageKeyPrefix?: string;
  embedded?: boolean;
  onAvailabilityChange?: (available: boolean, resolved: boolean) => void;
}

/**
 * Write the loaded, weight-only quantized transformer to a single file.
 *
 * Renders NOTHING unless the backend reports the loaded transformer owns
 * quantized Linear modules (`GET /models/export-quantized` -> `exportable`).
 * That covers both provenances, which are indistinguishable by then: a
 * checkpoint that shipped quantized, and a bf16 checkpoint converted in place
 * at generation time (which is one-way until the model is reloaded — exporting
 * is how that result survives a restart).
 *
 * The destination is pre-filled by the backend with a path outside the loaded
 * model's own directory tree, in the subdirectory the architecture's loader
 * needs; it is editable and the backend re-validates it.
 */
export default function QuantizedExportSection({
  arch,
  modelInfoVersion = 0,
  modelLoadRevision = 0,
  storageKeyPrefix = "model_load",
  embedded = false,
  onAvailabilityChange,
}: QuantizedExportSectionProps) {
  const [status, setStatus] = useState<QuantizedExportStatus | null>(null);
  const [path, setPath] = useState("");
  const [pathTouched, setPathTouched] = useState(false);
  const [linkSiblings, setLinkSiblings] = useState(true);
  const [overwrite, setOverwrite] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [availabilityResolved, setAvailabilityResolved] = useState(false);
  const pathTouchedRef = useRef(pathTouched);
  const modelIdentity = `${arch ?? ""}:${modelInfoVersion}:${modelLoadRevision}`;
  const modelIdentityRef = useRef(modelIdentity);
  pathTouchedRef.current = pathTouched;
  modelIdentityRef.current = modelIdentity;

  const refresh = useCallback(async () => {
    const requestedModel = modelIdentityRef.current;
    try {
      const next = await getQuantizedExportStatus();
      if (requestedModel !== modelIdentityRef.current) return;
      setStatus(next);
      if (!pathTouchedRef.current && next.suggested_path) {
        setPath(next.suggested_path);
      }
    } catch {
      if (requestedModel !== modelIdentityRef.current) return;
      setStatus(null);
    } finally {
      if (requestedModel === modelIdentityRef.current) {
        setAvailabilityResolved(true);
      }
    }
  }, []);

  // Re-read on mount and whenever the loaded model changes (a reload undoes an
  // in-place conversion, so exportability can flip either way).
  useEffect(() => {
    pathTouchedRef.current = false;
    setPathTouched(false);
    setPath("");
    setStatus(null);
    setError(null);
    setAvailabilityResolved(false);
    refresh();
  }, [modelIdentity, refresh]);

  // While a job runs, poll for progress. Otherwise poll slowly: a generation
  // can convert the transformer in place (the request that flips this section
  // on), and that happens without any model-load event to react to.
  const running = status?.job?.state === "running";
  useEffect(() => {
    const id = setInterval(refresh, running ? 1500 : 10000);
    return () => clearInterval(id);
  }, [running, refresh]);

  const handleExport = async () => {
    setError(null);
    setStarting(true);
    try {
      await startQuantizedExport({
        output_path: path,
        link_siblings: linkSiblings,
        overwrite,
      });
      await refresh();
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? "Export failed to start");
    } finally {
      setStarting(false);
    }
  };

  const job = status?.job;
  const hasJob = !!job && job.state !== "idle";
  const available = !!status && (status.exportable || hasJob);
  useEffect(() => {
    onAvailabilityChange?.(available, availabilityResolved);
  }, [available, availabilityResolved, onAvailabilityChange]);

  if (!available || !status) return null;

  const inv = status.inventory;
  const quantized = inv.int8 + inv.e4m3;
  const progress =
    job?.total && job.total > 0
      ? Math.round(((job.processed ?? 0) / job.total) * 100)
      : 0;

  const content = (
    <div className="space-y-3">
      <p className="rounded bg-gray-800 px-2 py-1 text-xs text-gray-400">
        {quantized} quantized Linear layer(s) ({inv.int8} INT8, {inv.e4m3} FP8 E4M3, {inv.plain} unquantized)
      </p>
      <p className="text-xs text-gray-400">
        Writes the loaded transformer&apos;s quantized weights to a single
        file that this backend can load directly. Transformer only — no text
        encoder and no VAE are embedded.
        {status.has_runtime_audit
          ? " The per-layer audit for this session's in-place conversion is written next to it."
          : " No per-layer audit exists for this model (its quantized layers came from the checkpoint), and the file's metadata records that."}
      </p>

      <Input
        label="Destination"
        value={path}
        onChange={(e) => {
          setPathTouched(true);
          setPath(e.target.value);
        }}
        placeholder="D:/.../model_int8.safetensors"
        disabled={running}
      />

      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={linkSiblings}
          onChange={(e) => setLinkSiblings(e.target.checked)}
          disabled={running}
          className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-2 focus:ring-blue-500"
        />
        <span className="text-sm text-gray-300">
          Link the loaded model&apos;s component directories next to the file
        </span>
      </label>
      <p className="text-xs text-gray-500">
        Creates directory junctions so the loader resolves the same text
        encoder and VAE the current model uses.
      </p>

      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={overwrite}
          onChange={(e) => setOverwrite(e.target.checked)}
          disabled={running}
          className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-2 focus:ring-blue-500"
        />
        <span className="text-sm text-gray-300">
          Overwrite an existing file at the destination
        </span>
      </label>

      <Button
        onClick={handleExport}
        disabled={running || starting || !path.trim() || !status.exportable}
        size="sm"
      >
        {running ? "Exporting…" : starting ? "Starting…" : "Export"}
      </Button>

      {error && <p className="text-xs text-red-400 break-words">{error}</p>}

      {hasJob && (
        <div className="space-y-1 border-t border-gray-700 pt-2">
          {running && (
            <>
              <div className="h-1.5 w-full rounded bg-gray-700">
                <div
                  className="h-1.5 rounded bg-blue-500 transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-xs text-gray-400">
                {job?.processed ?? 0} / {job?.total ?? 0} tensors — {job?.message}
              </p>
            </>
          )}
          {job?.state === "completed" && (
            <p className="text-xs text-green-400 break-words">
              Wrote {job.written_path}
            </p>
          )}
          {job?.state === "failed" && (
            <p className="text-xs text-red-400 break-words">{job.error}</p>
          )}
        </div>
      )}
    </div>
  );

  if (embedded) return content;

  return (
    <Card
      title="Export quantized model"
      collapsible={true}
      defaultCollapsed={true}
      collapsedPreview={
        <p className="text-xs text-gray-400">
          {quantized} quantized Linear layer(s) ({inv.int8} INT8, {inv.e4m3} FP8 E4M3, {inv.plain} unquantized)
        </p>
      }
      storageKey={`${storageKeyPrefix}_quant_export_collapsed`}
    >
      {content}
    </Card>
  );
}
