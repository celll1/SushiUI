"use client";

import { useCallback, useEffect, useState } from "react";
import { Check, ChevronDown, ChevronUp, RotateCcw, Sparkles, X } from "lucide-react";
import Textarea from "./Textarea";
import {
  createH3PromptTemplate,
  getPromptAssistDefaults,
  listPromptAssistModels,
  transformH3Prompt,
} from "@/utils/api";
import type {
  H3PromptMode,
  PromptAssistModel,
  PromptAssistReference,
  PromptAssistSettings,
} from "@/utils/api";
import {
  readH3PromptEditingMode,
  resolvePromptAssistSettings,
  saveH3PromptEditingMode,
  savePromptAssistSettings,
} from "@/utils/h3PromptAssist";
import type { H3PromptEditingMode } from "@/utils/h3PromptAssist";

interface H3PromptAssistProps {
  prompt: string;
  onApply: (prompt: string) => void;
  suggestedMode: H3PromptMode;
  durationSeconds: number;
  references?: PromptAssistReference[];
}

function errorMessage(error: any): string {
  return error?.response?.data?.detail || error?.message || "Prompt Assist failed";
}

function modelSize(size?: number | null): string {
  if (!size) return "";
  return ` · ${(size / 1024 ** 3).toFixed(1)} GB`;
}

export default function H3PromptAssist({
  prompt,
  onApply,
  suggestedMode,
  durationSeconds,
  references = [],
}: H3PromptAssistProps) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<H3PromptMode>(suggestedMode);
  const [settings, setSettings] = useState<PromptAssistSettings | null>(null);
  const [models, setModels] = useState<PromptAssistModel[]>([]);
  const [result, setResult] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [valid, setValid] = useState(false);
  const [cached, setCached] = useState(false);
  const [busy, setBusy] = useState(false);
  const [modelBusy, setModelBusy] = useState(false);
  const [error, setError] = useState("");
  const [lastAppliedSource, setLastAppliedSource] = useState<string | null>(null);
  const [editingMode, setEditingMode] = useState<H3PromptEditingMode>("natural-language");

  useEffect(() => {
    setMode(suggestedMode);
  }, [suggestedMode]);

  useEffect(() => {
    resolvePromptAssistSettings().then(setSettings).catch((reason) => setError(errorMessage(reason)));
    setEditingMode(readH3PromptEditingMode());
    const updateSettings = (event: Event) => {
      setSettings((event as CustomEvent<PromptAssistSettings>).detail);
    };
    window.addEventListener("h3-prompt-assist-settings", updateSettings);
    return () => window.removeEventListener("h3-prompt-assist-settings", updateSettings);
  }, []);

  const changeEditingMode = (next: H3PromptEditingMode) => {
    setEditingMode(next);
    saveH3PromptEditingMode(next);
  };

  const persist = useCallback((next: PromptAssistSettings) => {
    setSettings(next);
    savePromptAssistSettings({ ...next, api_key: "" });
  }, []);

  const refreshModels = useCallback(async () => {
    if (!settings) return;
    setModelBusy(true);
    setError("");
    try {
      const found = await listPromptAssistModels(settings.provider, settings.base_url);
      setModels(found);
      if (!settings.model && found.length === 1) {
        persist({ ...settings, model: found[0].id });
      }
    } catch (reason) {
      setModels([]);
      setError(errorMessage(reason));
    } finally {
      setModelBusy(false);
    }
  }, [persist, settings]);

  useEffect(() => {
    if (open && settings && models.length === 0) refreshModels();
  }, [open, settings, models.length, refreshModels]);

  const switchProvider = async (provider: "lm_studio" | "ollama") => {
    if (!settings) return;
    const defaults = await getPromptAssistDefaults();
    persist({
      ...settings,
      provider,
      base_url: provider === "lm_studio"
        ? defaults.lm_studio_base_url
        : defaults.ollama_base_url,
      model: "",
    });
    setModels([]);
  };

  const createTemplate = async () => {
    setBusy(true);
    setError("");
    try {
      const response = await createH3PromptTemplate(prompt, mode, durationSeconds);
      setResult(response.prompt);
      setWarnings(response.warnings);
      setValid(response.valid);
      setCached(false);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const rewrite = async (forceRefresh = false) => {
    if (!settings?.model) {
      setError("Select a local LLM model first.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const response = await transformH3Prompt({
        ...settings,
        api_key: "",
        prompt,
        mode,
        duration_seconds: durationSeconds,
        references,
        force_refresh: forceRefresh,
      });
      setResult(response.prompt);
      setWarnings(response.warnings);
      setValid(response.valid);
      setCached(!!response.cached);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const apply = () => {
    if (!result) return;
    setLastAppliedSource(prompt);
    onApply(result);
  };

  return (
    <div className="rounded border border-violet-500/30 bg-gray-900/70">
      <div className="flex items-center text-xs text-gray-200 hover:bg-gray-800/70">
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          className="flex min-w-0 flex-1 items-center gap-2 px-2.5 py-1.5 text-left"
          aria-expanded={open}
        >
          <Sparkles size={14} className="text-violet-300" />
          <span className="font-medium">MiniMax H3 Prompt Assist</span>
          <span className="rounded bg-violet-500/15 px-1.5 py-0.5 text-[10px] uppercase text-violet-200">
            {mode}
          </span>
          {cached && <span className="text-[10px] text-emerald-300">cached</span>}
          <span className="ml-auto text-gray-400">{open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</span>
        </button>
        <select
          value={editingMode}
          onChange={(event) => changeEditingMode(event.target.value as H3PromptEditingMode)}
          className="mr-2 rounded border border-gray-600 bg-gray-800 px-1.5 py-0.5 text-[10px] text-gray-200"
          title="Choose standard natural-language editing or Danbooru tag suggestions and shortcuts"
        >
          <option value="natural-language">Natural</option>
          <option value="tags">Tags</option>
        </select>
      </div>

      {open && (
        <div className="space-y-2 border-t border-gray-700/70 p-2.5" data-prompt-assist-open="true">
          <div className="grid gap-2 sm:grid-cols-[140px_140px_minmax(180px,1fr)_auto]">
            <label className="text-xs text-gray-400">
              H3 workflow
              <select
                value={mode}
                onChange={(event) => setMode(event.target.value as H3PromptMode)}
                className="mt-1 w-full rounded border border-gray-600 bg-gray-800 px-2 py-1.5 text-xs text-white"
              >
                <option value="t2va">T2VA</option>
                <option value="i2va">I2VA</option>
                <option value="fl2va">FL2VA</option>
                <option value="l2va">L2VA</option>
                <option value="ref2va">REF2VA</option>
              </select>
            </label>
            <label className="text-xs text-gray-400">
              Provider
              <select
                value={settings?.provider ?? "lm_studio"}
                onChange={(event) => switchProvider(event.target.value as "lm_studio" | "ollama")}
                className="mt-1 w-full rounded border border-gray-600 bg-gray-800 px-2 py-1.5 text-xs text-white"
                disabled={!settings}
              >
                <option value="lm_studio">LM Studio</option>
                <option value="ollama">Ollama</option>
              </select>
            </label>
            <label className="text-xs text-gray-400">
              Local model
              <select
                value={settings?.model ?? ""}
                onChange={(event) => settings && persist({ ...settings, model: event.target.value })}
                className="mt-1 w-full rounded border border-gray-600 bg-gray-800 px-2 py-1.5 text-xs text-white"
                disabled={!settings || modelBusy}
              >
                <option value="">Select model…</option>
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}{modelSize(model.size_bytes)}{model.loaded ? " · loaded" : ""}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              onClick={refreshModels}
              className="self-end rounded bg-gray-700 px-2.5 py-1.5 text-xs text-gray-100 hover:bg-gray-600 disabled:opacity-50"
              disabled={!settings || modelBusy}
            >
              {modelBusy ? "Checking…" : "Refresh"}
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-3 rounded bg-gray-800/70 px-2 py-1.5 text-xs">
            <label className="flex items-center gap-1.5 text-gray-300">
              <input
                type="checkbox"
                checked={settings?.auto_on_generate ?? false}
                onChange={(event) => settings && persist({ ...settings, auto_on_generate: event.target.checked })}
                disabled={!settings}
              />
              Rewrite automatically on Generate
            </label>
            <span className="text-gray-500">Cache is checked before model load; an LLM loaded for a rewrite is unloaded afterward.</span>
          </div>

          <div className="flex flex-wrap gap-1.5">
            <button
              type="button"
              onClick={createTemplate}
              className="rounded bg-gray-700 px-2.5 py-1.5 text-xs hover:bg-gray-600 disabled:opacity-50"
              disabled={busy || !prompt.trim()}
              title="Create an editable official-format scaffold without inventing content"
            >
              Structure scaffold
            </button>
            <button
              type="button"
              onClick={() => rewrite(false)}
              className="rounded bg-violet-600 px-2.5 py-1.5 text-xs font-medium hover:bg-violet-500 disabled:opacity-50"
              disabled={busy || !prompt.trim() || !settings?.model}
            >
              {busy ? "Rewriting…" : "AI rewrite"}
            </button>
            {result && (
              <button
                type="button"
                onClick={() => rewrite(true)}
                className="flex items-center gap-1 rounded bg-gray-700 px-2.5 py-1.5 text-xs hover:bg-gray-600 disabled:opacity-50"
                disabled={busy || !settings?.model}
                title="Ignore the cached result and run the LLM again"
              >
                <RotateCcw size={12} /> Retry
              </button>
            )}
          </div>

          {error && <p className="rounded bg-red-950/50 px-2 py-1.5 text-xs text-red-300">{error}</p>}

          {result && (
            <div className="space-y-1.5">
              <Textarea
                label={`Preview${cached ? " · reused from cache" : ""}`}
                rows={8}
                value={result}
                onChange={(event) => setResult(event.target.value)}
                className="font-mono text-xs leading-relaxed"
              />
              {warnings.length > 0 && (
                <ul className="space-y-0.5 rounded bg-amber-950/30 px-2 py-1.5 text-xs text-amber-200">
                  {warnings.map((warning, index) => <li key={`${warning}-${index}`}>• {warning}</li>)}
                </ul>
              )}
              <div className="flex flex-wrap items-center gap-1.5">
                <button
                  type="button"
                  onClick={apply}
                  className="flex items-center gap-1 rounded bg-emerald-700 px-2.5 py-1.5 text-xs hover:bg-emerald-600"
                  title={valid ? "Apply the validated result" : "Apply after reviewing the warnings"}
                >
                  <Check size={12} /> Apply to prompt
                </button>
                {lastAppliedSource !== null && (
                  <button
                    type="button"
                    onClick={() => {
                      onApply(lastAppliedSource);
                      setLastAppliedSource(null);
                    }}
                    className="flex items-center gap-1 rounded bg-gray-700 px-2.5 py-1.5 text-xs hover:bg-gray-600"
                  >
                    <RotateCcw size={12} /> Undo apply
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    setResult("");
                    setWarnings([]);
                    setCached(false);
                  }}
                  className="ml-auto flex items-center gap-1 rounded px-2 py-1.5 text-xs text-gray-400 hover:bg-gray-800"
                >
                  <X size={12} /> Clear preview
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
