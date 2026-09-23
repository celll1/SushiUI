"use client";

import { useCallback, useEffect, useState } from "react";
import {
  createStyleTrainingRun, deleteStyleTrainingRun, fetchStyleTrainingDefaults,
  getStyleTrainingMetrics, getStyleTrainingRun, listDatasets, listStyleTrainingRuns,
  startStyleTrainingRun, stopStyleTrainingRun, updateStyleTrainingRun,
  type Dataset, type StyleDatasetConfig, type StyleTrainingConfig, type StyleTrainingMetric,
  type StyleTrainingRun, type StyleTrainingRunDetail,
} from "@/utils/api";

type Editor = { runId: string | null; name: string; datasetIds: number[]; datasetConfigs: StyleDatasetConfig[]; config: StyleTrainingConfig };
const editableKeys = [
  "training_method", "epochs", "learning_rate", "style_max_num_patches",
  "style_artists_per_batch", "style_images_per_artist", "num_workers",
  "style_hsic_tagger_path", "vision_encoder_path",
] as const;

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

export default function StyleTrainingPanel() {
  const [runs, setRuns] = useState<StyleTrainingRun[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<StyleTrainingRunDetail | null>(null);
  const [metrics, setMetrics] = useState<StyleTrainingMetric[]>([]);
  const [editor, setEditor] = useState<Editor | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const next = await listStyleTrainingRuns();
      setRuns(next);
      if (selectedId) {
        const [selected, points] = await Promise.all([
          getStyleTrainingRun(selectedId), getStyleTrainingMetrics(selectedId),
        ]);
        setDetail(selected);
        setMetrics(points);
      }
      setError("");
    } catch (err) {
      setError(errorText(err));
    }
  }, [selectedId]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    const timer = setInterval(() => {
      if (runs.some(run => run.active || run.status === "starting" || run.status === "stopping")) void refresh();
    }, 3000);
    return () => clearInterval(timer);
  }, [runs, refresh]);

  const select = (runId: string) => {
    setSelectedId(runId);
    setEditor(null);
    setDetail(null);
    setMetrics([]);
  };

  const openEditor = async (runId: string | null) => {
    setBusy(true);
    setError("");
    try {
      const [defaults, datasetList, existing] = await Promise.all([
        fetchStyleTrainingDefaults(), listDatasets(),
        runId ? getStyleTrainingRun(runId) : Promise.resolve(null),
      ]);
      setDatasets(datasetList.datasets);
      setEditor({
        runId,
        name: existing?.run_name ?? "",
        datasetIds: existing?.dataset_configs.map(item => item.dataset_id) ?? [],
        datasetConfigs: existing?.dataset_configs ?? [],
        config: existing?.config ?? defaults,
      });
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const change = <K extends keyof StyleTrainingConfig>(key: K, value: StyleTrainingConfig[K]) => {
    setEditor(current => current && ({ ...current, config: { ...current.config, [key]: value } }));
  };

  const save = async () => {
    if (!editor) return;
    if (editor.datasetIds.length === 0) { setError("Select at least one dataset."); return; }
    setBusy(true);
    setError("");
    try {
      const data = {
        run_name: editor.name.trim() || undefined,
        dataset_configs: editor.datasetIds.map(dataset_id => ({
          dataset_id,
          caption_types: editor.datasetConfigs.find(item => item.dataset_id === dataset_id)?.caption_types,
        })),
        ...Object.fromEntries(editableKeys.map(key => [key, editor.config[key]])),
      };
      const saved = editor.runId
        ? await updateStyleTrainingRun(editor.runId, data)
        : await createStyleTrainingRun(data);
      setEditor(null);
      setSelectedId(saved.run_id);
      await refresh();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const act = async (action: "start" | "stop" | "delete") => {
    if (!selectedId) return;
    if (action === "delete" && !window.confirm("Delete this style training run?")) return;
    setBusy(true);
    setError("");
    try {
      if (action === "start") await startStyleTrainingRun(selectedId);
      if (action === "stop") await stopStyleTrainingRun(selectedId);
      if (action === "delete") {
        await deleteStyleTrainingRun(selectedId);
        setRuns(current => current.filter(run => run.run_id !== selectedId));
        setSelectedId(null);
        setDetail(null);
        setMetrics([]);
        return;
      }
      await refresh();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const selected = runs.find(run => run.run_id === selectedId);
  const latest = metrics.at(-1);
  const button = "rounded border border-gray-600 px-3 py-1.5 text-sm hover:bg-gray-700 disabled:opacity-40";
  const field = "w-full rounded border border-gray-600 bg-gray-900 px-2 py-1.5 text-sm";
  return (
    <div className="flex h-full w-full flex-col overflow-hidden text-gray-200">
      <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
        <h2 className="font-medium">Style encoder training</h2>
        <div className="flex gap-2">
          <button className={button} disabled={busy} onClick={() => void refresh()}>Refresh</button>
          <button className={button} disabled={busy} onClick={() => void openEditor(null)}>New run</button>
        </div>
      </div>
      {error && <div role="alert" className="m-3 rounded border border-red-700 bg-red-950/30 p-2 text-sm text-red-300">{error}</div>}
      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <div className="max-h-48 overflow-y-auto border-b border-gray-800 md:max-h-none md:w-64 md:shrink-0 md:border-b-0 md:border-r">
          {runs.map(run => (
            <button key={run.run_id} onClick={() => select(run.run_id)}
              className={`w-full border-b border-gray-800 px-4 py-3 text-left text-sm hover:bg-gray-800 ${selectedId === run.run_id ? "bg-gray-800" : ""}`}>
              <div className="truncate font-medium">{run.run_name}</div>
              <div className="mt-1 text-xs text-gray-400">{run.status} · step {run.current_step}{run.total_steps != null ? ` / ${run.total_steps}` : ""}</div>
            </button>
          ))}
          {runs.length === 0 && <p className="p-4 text-sm text-gray-400">No style runs yet.</p>}
        </div>
        <div className="min-w-0 flex-1 overflow-y-auto p-4">
          {editor ? (
            <div className="max-w-2xl space-y-4">
              <h3 className="font-medium">{editor.runId ? "Edit run" : "New run"}</h3>
              <label className="block text-sm">Run name<input className={field} value={editor.name} onChange={event => setEditor({ ...editor, name: event.target.value })} /></label>
              <fieldset className="space-y-1 text-sm"><legend>Datasets with tag captions</legend>
                {datasets.filter(dataset => dataset.has_tags_captions).map(dataset => (
                  <label key={dataset.id} className="flex gap-2"><input type="checkbox" checked={editor.datasetIds.includes(dataset.id)}
                    onChange={event => setEditor({ ...editor, datasetIds: event.target.checked
                      ? [...editor.datasetIds, dataset.id] : editor.datasetIds.filter(id => id !== dataset.id) })} />{dataset.name}</label>
                ))}
                {!datasets.some(dataset => dataset.has_tags_captions) && <p className="text-gray-400">No scanned dataset with tag captions.</p>}
              </fieldset>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="text-sm">Training method<select className={field} value={editor.config.training_method} onChange={event => change("training_method", event.target.value as "full" | "lora")}><option value="lora">LoRA</option><option value="full">Full</option></select></label>
                {(["epochs", "learning_rate", "style_max_num_patches", "style_artists_per_batch", "style_images_per_artist", "num_workers"] as const).map(key => (
                  <label key={key} className="text-sm">{key.replaceAll("_", " ")}<input className={field} type="number" min={key === "num_workers" ? 0 : 1} step={key === "learning_rate" ? "any" : 1}
                    value={editor.config[key]} onChange={event => change(key, Number(event.target.value))} /></label>
                ))}
                <label className="text-sm sm:col-span-2">Vision encoder path or repo ID<input className={field} value={editor.config.vision_encoder_path} onChange={event => change("vision_encoder_path", event.target.value)} /></label>
                <label className="text-sm sm:col-span-2">HSIC tagger path (blank disables HSIC)<input className={field} value={editor.config.style_hsic_tagger_path} onChange={event => change("style_hsic_tagger_path", event.target.value)} /></label>
              </div>
              <div className="flex gap-2"><button className={button} disabled={busy} onClick={() => void save()}>Save</button><button className={button} onClick={() => setEditor(null)}>Cancel</button></div>
            </div>
          ) : selected ? (
            <div className="max-w-2xl space-y-4 text-sm">
              <div><h3 className="text-lg font-medium">{selected.run_name}</h3><p className="text-gray-400">{selected.status} · epoch {selected.current_epoch} / {selected.total_epochs} · step {selected.current_step}{selected.total_steps != null ? ` / ${selected.total_steps}` : ""}</p></div>
              {selected.status_message && <p>{selected.status_message}</p>}
              {detail?.detail_error && <p className="text-amber-300">{detail.detail_error}</p>}
              {selected.error_message && <p className="text-red-300">{selected.error_message}</p>}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div>Loss<br /><strong>{selected.latest_loss?.toFixed(4) ?? "—"}</strong></div>
                <div>HSIC<br /><strong>{latest?.hsic_loss?.toFixed(4) ?? "—"}</strong></div>
                <div>Best mAP@R<br /><strong>{selected.best_map_at_r?.toFixed(4) ?? "—"}</strong></div>
                <div>Learning rate<br /><strong>{selected.latest_lr?.toExponential(2) ?? "—"}</strong></div>
              </div>
              {latest?.hsic_components && <p>HSIC: {Object.entries(latest.hsic_components).map(([name, value]) => `${name} ${value.toFixed(4)}`).join(" · ")}</p>}
              {detail?.warnings.map((warning, index) => <p className="text-amber-300" key={index}>{warning.message}</p>)}
              <div className="flex flex-wrap gap-2">
                <button className={button} disabled={busy || selected.active || selected.status === "starting" || selected.status === "stopping"} onClick={() => void act("start")}>Start / resume</button>
                <button className={button} disabled={busy || !selected.active} onClick={() => void act("stop")}>Stop</button>
                <button className={button} disabled={busy || selected.active} onClick={() => void openEditor(selected.run_id)}>Edit</button>
                <button className={button} disabled={busy || selected.active} onClick={() => void act("delete")}>Delete</button>
              </div>
            </div>
          ) : <p className="text-gray-400">Select a run or create a new one.</p>}
        </div>
      </div>
    </div>
  );
}
