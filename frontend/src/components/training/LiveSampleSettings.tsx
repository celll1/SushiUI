import { useEffect, useRef, useState } from "react";
import {
  getTrainingLiveSampleConfig, updateTrainingLiveSampleConfig,
  TrainingLiveSampleConfig, TrainingLiveSampleConfigStatus,
} from "@/utils/api";

interface Props {
  runId: number;
  isRunning: boolean;
  disabledReason?: string;
}

export default function LiveSampleSettings({ runId, isRunning, disabledReason }: Props) {
  const [status, setStatus] = useState<TrainingLiveSampleConfigStatus | null>(null);
  const [draft, setDraft] = useState<TrainingLiveSampleConfig | null>(null);
  const [dirty, setDirty] = useState(false);
  const dirtyRef = useRef(false);
  const editRevisionRef = useRef(0);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    dirtyRef.current = false;
    setDirty(false);
    setDraft(null);
    setStatus(null);
  }, [runId]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const next = await getTrainingLiveSampleConfig(runId);
        if (cancelled) return;
        setStatus(next);
        if (!dirtyRef.current) {
          setDraft(next.config);
          editRevisionRef.current = next.desired_revision;
        }
        if (!dirtyRef.current) setError(null);
      } catch (err: any) {
        if (!cancelled) setError(err?.response?.status === 404
          ? "Available after the next backend restart. The current run is unchanged."
          : err?.response?.data?.detail || "Could not read sample settings");
      }
    };
    void load();
    const timer = setInterval(load, 5000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [runId]);

  const edit = (next: TrainingLiveSampleConfig) => {
    if (!dirtyRef.current) editRevisionRef.current = status?.desired_revision ?? 0;
    dirtyRef.current = true;
    setDirty(true);
    setDraft(next);
  };

  const save = async () => {
    if (!draft || !status) return;
    setSaving(true);
    setError(null);
    try {
      const next = await updateTrainingLiveSampleConfig(runId, draft, editRevisionRef.current);
      setStatus(next);
      setDraft(next.config);
      dirtyRef.current = false;
      setDirty(false);
      editRevisionRef.current = next.desired_revision;
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Could not save sample settings");
    } finally {
      setSaving(false);
    }
  };

  return (
    <details className="rounded border border-gray-700 bg-gray-800/60 p-2 text-xxs sm:text-xs">
      <summary className="cursor-pointer text-gray-200">Sample settings</summary>
      <div className="mt-3 space-y-2.5">
        <p className="text-gray-400">Changes apply to future samples; an image already rendering keeps its original settings.</p>
        {status?.pending && <p className="text-yellow-400">Saved; waiting for the trainer to apply revision {status.desired_revision}.</p>}
        {status && !status.pending && status.applied_revision !== null && status.applied_revision === status.desired_revision && status.applied_revision > 0 && (
          <p className="text-green-400">Revision {status.applied_revision} applied at step {status.applied_step ?? "?"}.</p>
        )}
        {error && <p className="text-red-400">{error}</p>}
        {draft && <>
          {draft.prompts.map((prompt, index) => (
            <div key={index} className="space-y-1 rounded border border-gray-700 p-2">
              <div className="flex justify-between"><span>Prompt {index + 1}</span>
                {draft.prompts.length > 1 && <button type="button" className="text-red-400" onClick={() => edit({ ...draft, prompts: draft.prompts.filter((_, i) => i !== index) })}>Remove</button>}
              </div>
              <textarea aria-label={`Sample positive prompt ${index + 1}`} className="w-full rounded bg-gray-900 p-1.5" rows={4} value={prompt.positive}
                onChange={(e) => edit({ ...draft, prompts: draft.prompts.map((p, i) => i === index ? { ...p, positive: e.target.value } : p) })} />
              <textarea aria-label={`Sample negative prompt ${index + 1}`} className="w-full rounded bg-gray-900 p-1.5" rows={2} placeholder="Negative prompt" value={prompt.negative ?? ""}
                onChange={(e) => edit({ ...draft, prompts: draft.prompts.map((p, i) => i === index ? { ...p, negative: e.target.value } : p) })} />
              {(prompt.condition_image_path || prompt.reference_image_path) && <p className="text-gray-500">Existing condition/reference image paths are preserved.</p>}
            </div>
          ))}
          {draft.prompts.length < 20 && <button type="button" className="text-blue-400" onClick={() => edit({ ...draft, prompts: [...draft.prompts, { positive: "", negative: "" }] })}>+ Add prompt</button>}
          <div className="grid grid-cols-2 gap-2">
            {([
              ["sample_every", "Every N steps"], ["sample_steps", "Inference steps"],
              ["width", "Width"], ["height", "Height"],
              ["guidance_scale", "CFG scale"], ["seed", "Seed (-1 = random)"],
            ] as const).map(([key, label]) => <label key={key} className="space-y-1 text-gray-400">{label}
              <input type="number" className="w-full rounded bg-gray-900 p-1.5 text-gray-100"
                min={key === "seed" ? -1 : key === "sample_every" || key === "guidance_scale" ? 0 : 1}
                step={key === "guidance_scale" ? "0.1" : "1"}
                value={draft[key]} onChange={(e) => edit({ ...draft, [key]: Number(e.target.value) })} />
            </label>)}
          </div>
          <button type="button" onClick={save} disabled={!dirty || saving || !isRunning || !!disabledReason}
            className="w-full rounded bg-blue-700 px-2 py-1.5 hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-50">
            {saving ? "Saving..." : "Apply to future samples"}
          </button>
          {dirty && status && <button type="button" className="w-full text-gray-400 hover:text-gray-200" onClick={() => {
            setDraft(status.config);
            editRevisionRef.current = status.desired_revision;
            dirtyRef.current = false;
            setDirty(false);
            setError(null);
          }}>Discard edits and reload</button>}
          {!isRunning && <p className="text-gray-500">Live editing is available while training is running.</p>}
          {disabledReason && <p className="text-yellow-400">{disabledReason}</p>}
        </>}
      </div>
    </details>
  );
}
