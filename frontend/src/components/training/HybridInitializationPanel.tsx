"use client";

import { useEffect, useRef, useState } from "react";
import { cancelHybridInitialization, getDiTRewiringSchema, getHybridInitializationStatus,
  listDatasets, startHybridInitialization } from "@/utils/api";
import type { Dataset, HybridDiTRewiringSpec, HybridInitializationJob,
  HybridInitializationRequest } from "@/utils/api";

const active = (job: HybridInitializationJob | null) =>
  !!job && ["queued", "running", "cancelling"].includes(job.state);
const samePath = (a?: string, b?: string) => !!a && !!b &&
  a.replace(/\\/g, "/").toLowerCase() === b.replace(/\\/g, "/").toLowerCase();
const errorMessage = (error: unknown) => {
  const detail = (error as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  return typeof detail === "string" ? detail : Array.isArray(detail) ? JSON.stringify(detail)
    : error instanceof Error ? error.message : "Initialization request failed";
};

export default function HybridInitializationPanel({ modelPath, architecture, spec, onUse }: {
  modelPath: string; spec: HybridDiTRewiringSpec | null; onUse: (path: string) => void | Promise<void>;
  architecture: string;
}) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [name, setName] = useState("");
  const [settings, setSettings] = useState("");
  const [supported, setSupported] = useState(false);
  const [job, setJob] = useState<HybridInitializationJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const alive = useRef(true);
  const actionPending = useRef(false);
  const actionVersion = useRef(0);
  useEffect(() => {
    let cancelled = false;
    alive.current = true;
    Promise.all([getDiTRewiringSchema(), listDatasets()]).then(([schema, data]) => {
      if (cancelled) return;
      const { spec: _spec, item_ids: _ids, ...defaults } = schema.initialization_defaults;
      setSettings(JSON.stringify(defaults, null, 2));
      setDatasets(data.datasets);
      setSupported(schema.initializer_architectures.includes(architecture));
    }).catch(err => { if (!cancelled) setError(errorMessage(err)); });
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      const version = actionVersion.current;
      try {
        const value = await getHybridInitializationStatus();
        if (!cancelled && !actionPending.current && version === actionVersion.current) { setJob(value); setPollError(null); }
      } catch (err) { if (!cancelled) setPollError(errorMessage(err)); }
      finally { if (!cancelled) timer = setTimeout(poll, 2000); }
    };
    void poll();
    return () => { cancelled = true; alive.current = false; clearTimeout(timer); };
  }, [architecture]);
  const start = async () => {
    if (actionPending.current || pending || active(job) || !spec) return;
    setPending(true); actionPending.current = true; setError(null);
    actionVersion.current++;
    try {
      const parsed: unknown = JSON.parse(settings);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Settings must be a JSON object");
      const request = { ...parsed, model_path: modelPath, output_name: name.trim(),
        dataset_id: Number(datasetId), spec } as HybridInitializationRequest;
      const result = await startHybridInitialization(request);
      if (alive.current) setJob(result);
    } catch (err) { if (alive.current) setError(errorMessage(err)); }
    finally { actionPending.current = false; if (alive.current) setPending(false); }
  };
  const cancel = async () => {
    if (!job?.job_id || actionPending.current || pending) return;
    setPending(true); actionPending.current = true; setError(null);
    actionVersion.current++;
    try {
      const result = await cancelHybridInitialization(job.job_id);
      if (alive.current) setJob(result);
    } catch (err) { if (alive.current) setError(errorMessage(err)); }
    finally { actionPending.current = false; if (alive.current) setPending(false); }
  };
  const useResult = async () => {
    if (actionPending.current || pending || !job?.result || job.state !== "completed" || !samePath(job.model_path, modelPath)) return;
    setPending(true); actionPending.current = true; actionVersion.current++; setError(null);
    try { await onUse(job.result.model_path); }
    catch (err) { if (alive.current) setError(errorMessage(err)); }
    finally { actionPending.current = false; if (alive.current) setPending(false); }
  };
  const busy = pending || active(job);
  const ownJob = samePath(job?.model_path, modelPath);
  const percent = job?.total ? Math.min(100, 100 * (job.completed ?? 0) / job.total) : 0;
  const packageOutput = architecture === "qwen_image_21" || architecture === "ming_image_01";
  if (!supported && settings) return null;
  return <details className="rounded border border-gray-700 p-3 text-xs">
    <summary className="cursor-pointer text-sky-300">Initialize hybrid modules with the native teacher</summary>
    <p className="my-2 text-gray-400">Load this native model in Generate first, with adapters and VAE swap disabled.
      Initialization freezes its existing weights and writes a new checkpoint beside the source.
      {architecture === "qwen_image_21" && " Qwen writes a package directory with this name; original TE/VAE assets must remain available."}
      {architecture === "ming_image_01" && " Ming copies native assets into a new package. Layer initialization uses a native-generated canvas and one layer with the calibration image as reference."}
      The final holdout images are excluded from fitting. Closing this panel leaves the job running.</p>
    <fieldset disabled={busy} className="space-y-2">
      <label className="block">Calibration dataset <select className="ml-2 rounded bg-gray-800 p-2"
        value={datasetId} onChange={e => setDatasetId(e.target.value)}>
        <option value="">Select dataset</option>{datasets.map(ds =>
          <option key={ds.id} value={ds.id}>{ds.name} (#{ds.id}, {ds.total_items} items)</option>)}
      </select></label>
      <label className="block">{packageOutput ? "New package directory name" : "New checkpoint filename"}
        <input className="ml-2 rounded bg-gray-800 p-2" value={name}
          placeholder={packageOutput ? `${architecture}-hybrid` : "anima-hybrid.safetensors"}
          onChange={e => setName(e.target.value)} /></label>
      <details><summary>Fitting settings and ordered phases</summary>
        <p className="my-2 text-gray-400">Defaults come from the API. Optional item_ids selects explicit dataset items in order;
          otherwise the first sample_count IDs are used. Each phase can set skip_learning_rate_multiplier
          to reduce compressed-skip learning rates while keeping other branches at learning_rate.
          This affects initialization only.</p>
        <textarea aria-label="Hybrid initialization settings" rows={18} value={settings}
          className="w-full rounded bg-gray-900 p-2 font-mono" onChange={e => setSettings(e.target.value)} />
      </details>
      <button type="button" className="rounded bg-sky-800 px-3 py-2 disabled:opacity-50"
        disabled={!supported || !spec || !datasetId || !name.trim() || !settings || !!pollError}
        onClick={() => void start()}>Initialize and save new checkpoint</button>
    </fieldset>
    {job && job.state !== "idle" && <div role="status" className="mt-3 space-y-2">
      <p>{job.state}: {job.stage} · {job.completed ?? 0}/{job.total ?? 0} {job.units}
        {job.loss != null && ` · loss ${job.loss.toPrecision(4)}`}</p>
      {!ownJob && <p>Job source: {job.model_path}</p>}
      {active(job) && <><progress className="w-full" max={100} value={percent} />
        <button type="button" disabled={pending || job.state === "cancelling"} onClick={() => void cancel()}
          className="rounded bg-gray-700 px-3 py-2">{job.state === "cancelling" ? "Waiting for cleanup…" : "Cancel initialization"}</button></>}
      {job.error && <p role="alert" className="text-red-400">{job.error}</p>}
      {job.state === "completed" && job.result && <><p className="break-all">{job.result.model_path}</p>
        {ownJob && <button type="button" disabled={pending} className="rounded bg-sky-800 px-3 py-2" onClick={() => void useResult()}>
          Use initialized checkpoint in this training config</button>}</>}
    </div>}
    {error && <p role="alert" className="mt-2 text-red-400">{error}</p>}
    {pollError && <p role="alert" className="mt-2 text-red-400">Progress unavailable; job state is unknown: {pollError}</p>}
  </details>;
}
