"use client";

import { useCallback, useEffect, useState } from "react";
import { ChevronDown, ChevronUp, Sparkles } from "lucide-react";
import Textarea from "./Textarea";
import {
  getPromptAssistDefaults,
  listPromptAssistModels,
  transformQwen21Prompt,
} from "@/utils/api";
import type {
  PromptAssistModel,
  Qwen21PromptUpsampleEngine,
  Qwen21PromptUpsampleSettings,
} from "@/utils/api";
import {
  imageSourceToDataUrl,
  resolveQwen21PromptUpsampleSettings,
  saveQwen21PromptUpsampleSettings,
} from "@/utils/qwen21PromptUpsample";

interface Props {
  prompt: string;
  mode: "t2i" | "i2i";
  images?: Array<File | string>;
  onApply: (prompt: string) => void;
}

const message = (error: any) => error?.response?.data?.detail || error?.message || "Prompt upsampling failed";

export default function Qwen21PromptUpsample({ prompt, mode, images = [], onApply }: Props) {
  const [open, setOpen] = useState(false);
  const [settings, setSettings] = useState<Qwen21PromptUpsampleSettings | null>(null);
  const [models, setModels] = useState<PromptAssistModel[]>([]);
  const [result, setResult] = useState("");
  const [ratio, setRatio] = useState("");
  const [busy, setBusy] = useState(false);
  const [modelsBusy, setModelsBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    resolveQwen21PromptUpsampleSettings().then(setSettings).catch((reason) => setError(message(reason)));
    const update = (event: Event) => setSettings((event as CustomEvent<Qwen21PromptUpsampleSettings>).detail);
    window.addEventListener("qwen21-prompt-upsample-settings", update);
    return () => window.removeEventListener("qwen21-prompt-upsample-settings", update);
  }, []);

  const persist = useCallback((next: Qwen21PromptUpsampleSettings) => {
    setSettings(next);
    saveQwen21PromptUpsampleSettings(next);
  }, []);

  const refreshModels = useCallback(async () => {
    if (!settings || settings.engine === "official") return;
    setModelsBusy(true);
    setError("");
    try {
      const found = await listPromptAssistModels(settings.engine, settings.base_url);
      setModels(found);
      if (!settings.model && found.length === 1) persist({ ...settings, model: found[0].id });
    } catch (reason) {
      setError(message(reason));
      setModels([]);
    } finally {
      setModelsBusy(false);
    }
  }, [persist, settings]);

  const switchEngine = async (engine: Qwen21PromptUpsampleEngine) => {
    if (!settings) return;
    const defaults = await getPromptAssistDefaults();
    persist({
      ...settings,
      engine,
      model: engine === "official" ? "" : settings.model,
      base_url: engine === "ollama" ? defaults.ollama_base_url : defaults.lm_studio_base_url,
    });
    setModels([]);
  };

  const rewrite = async () => {
    if (!settings) return;
    if (settings.engine !== "official" && !settings.model) {
      setError("Select a local LLM model first.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const encoded = await Promise.all(images.map(imageSourceToDataUrl));
      const response = await transformQwen21Prompt({
        ...settings,
        api_key: "",
        prompt,
        mode,
        images: encoded,
      });
      setResult(response.prompt);
      setRatio(response.ratio_follow || response.wh_ratio);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded border border-cyan-500/30 bg-gray-900/70">
      <button type="button" onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs text-gray-200 hover:bg-gray-800/70">
        <Sparkles size={14} className="text-cyan-300" />
        <span className="font-medium">Qwen 2.1 Prompt Upsample</span>
        <span className="rounded bg-cyan-500/15 px-1.5 py-0.5 text-[10px] uppercase text-cyan-200">{mode}</span>
        <span className="ml-auto text-gray-400">{open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</span>
      </button>
      {open && settings && (
        <div className="space-y-2 border-t border-gray-700/70 p-2.5" data-prompt-assist-open="true">
          <div className="flex flex-wrap items-center gap-3 rounded bg-gray-800/70 px-2 py-1.5 text-xs">
            <label className="flex items-center gap-1.5 text-gray-200">
              <input type="checkbox" checked={settings.enabled}
                onChange={(event) => persist({ ...settings, enabled: event.target.checked })} />
              Upsample automatically on Generate
            </label>
            <span className="text-gray-500">The suggested ratio is shown only; canvas size is not changed.</span>
          </div>
          <div className="grid gap-2 sm:grid-cols-[150px_minmax(180px,1fr)_auto]">
            <label className="text-xs text-gray-400">Engine
              <select value={settings.engine} onChange={(event) => switchEngine(event.target.value as Qwen21PromptUpsampleEngine)}
                className="mt-1 w-full rounded border border-gray-600 bg-gray-800 px-2 py-1.5 text-xs text-white">
                <option value="official">Official PE</option>
                <option value="lm_studio">LM Studio</option>
                <option value="ollama">Ollama</option>
              </select>
            </label>
            {settings.engine !== "official" && <label className="text-xs text-gray-400">Local model
              <select value={settings.model} onChange={(event) => persist({ ...settings, model: event.target.value })}
                className="mt-1 w-full rounded border border-gray-600 bg-gray-800 px-2 py-1.5 text-xs text-white">
                <option value="">Select model…</option>
                {models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </select>
            </label>}
            {settings.engine !== "official" && <button type="button" onClick={refreshModels}
              className="self-end rounded bg-gray-700 px-2.5 py-1.5 text-xs hover:bg-gray-600 disabled:opacity-50" disabled={modelsBusy}>
              {modelsBusy ? "Checking…" : "Refresh"}
            </button>}
          </div>
          <button type="button" onClick={rewrite} disabled={busy || !prompt.trim()}
            className="rounded bg-cyan-700 px-2.5 py-1.5 text-xs font-medium hover:bg-cyan-600 disabled:opacity-50">
            {busy ? "Upsampling…" : "Preview upsample"}
          </button>
          {error && <p className="text-xs text-red-300">{error}</p>}
          {result && <>
            <Textarea label="Upsampled prompt" rows={6} value={result} onChange={(event) => setResult(event.target.value)} />
            <div className="flex items-center gap-2">
              <button type="button" onClick={() => onApply(result)} className="rounded bg-emerald-700 px-2.5 py-1.5 text-xs hover:bg-emerald-600">Apply</button>
              {ratio && <span className="text-xs text-cyan-200">Suggested ratio: {ratio}</span>}
            </div>
          </>}
        </div>
      )}
    </div>
  );
}
