"use client";
import { useEffect, useRef, useState } from "react";
import Textarea from "./Textarea";
import { getPromptAssistDefaults, listPromptAssistModels, clearMingPromptCache } from "@/utils/api";
import type { MingPromptSettings, PromptAssistModel } from "@/utils/api";
import { mingPromptImageKey, mingPromptSettingsEvent, resolveMingPromptSettings, saveMingPromptSettings, rewriteMingPrompt } from "@/utils/mingPromptUpsample";
import type { MingPromptArgs } from "@/utils/mingPromptUpsample";
interface Props extends MingPromptArgs { onApply: (prompt: string) => void }
const failure = (e: any) => e?.response?.data?.detail || e?.message || "Ming prompt rewriting failed";
export default function MingPromptUpsample({ onApply, ...args }: Props) {
  const [settings, setSettings] = useState<MingPromptSettings | null>(null);
  const [models, setModels] = useState<PromptAssistModel[]>([]);
  const [result, setResult] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const mounted = useRef(false);
  const revision = useRef(0);
  const imageKey = mingPromptImageKey(args.images);
  useEffect(() => {
    mounted.current = true;
    resolveMingPromptSettings().then(s => { if (mounted.current) setSettings(s); }).catch(e => { if (mounted.current) setError(failure(e)); });
    const update = (e: Event) => setSettings((e as CustomEvent<MingPromptSettings>).detail);
    window.addEventListener(mingPromptSettingsEvent, update);
    return () => { mounted.current = false; revision.current++; controller.current?.abort(); window.removeEventListener(mingPromptSettingsEvent, update); };
  }, []);
  useEffect(() => {
    revision.current++;
    controller.current?.abort(); controller.current = null;
    setBusy(false); setResult(""); setError("");
  }, [args.prompt, args.mode, args.width, args.height, args.layers, settings?.engine, settings?.model, settings?.base_url, settings?.temperature, settings?.top_p, settings?.top_k, settings?.max_output_tokens, settings?.context_length, settings?.timeout_seconds, imageKey]);
  const persist = (s: MingPromptSettings) => { setSettings(s); saveMingPromptSettings(s); };
  const refresh = async () => {
    if (!settings) return;
    const stamp = revision.current;
    setLoadingModels(true); setError("");
    try {
      const found = await listPromptAssistModels(settings.engine, settings.base_url);
      if (mounted.current && stamp === revision.current) setModels(found);
    } catch (e) { if (mounted.current && stamp === revision.current) setError(failure(e)); }
    finally { if (mounted.current) setLoadingModels(false); }
  };
  const preview = async () => {
    if (!settings) return;
    const active = new AbortController(); controller.current = active;
    setBusy(true); setResult(""); setError("");
    try {
      const response = await rewriteMingPrompt(args, settings, active.signal);
      if (mounted.current && controller.current === active && !active.signal.aborted) {
        setResult(response.prompt); setError(response.warnings.join("; "));
      }
    } catch (e) { if (mounted.current && controller.current === active && !active.signal.aborted) setError(failure(e)); }
    finally { if (mounted.current && controller.current === active) { controller.current = null; setBusy(false); } }
  };
  return <details className="rounded border border-cyan-500/30 p-2 text-xs">
    <summary>Ming Prompt Enhancement ({args.mode})</summary>
    {error && <p className="text-red-300">{error}</p>}
    {settings && <div className="space-y-2 pt-2">
      <label><input type="checkbox" checked={settings.enabled} onChange={e => persist({ ...settings, enabled: e.target.checked })} /> Rewrite automatically on Generate</label>
      <p>Uses your selected local provider. Canvas size and layer count stay fixed.</p>
      <select value={settings.engine} disabled={busy} onChange={async e => {
        const engine = e.target.value as MingPromptSettings["engine"];
        try { const defaults = await getPromptAssistDefaults(); if (mounted.current) { persist({ ...settings, engine, model: "", base_url: engine === "ollama" ? defaults.ollama_base_url : defaults.lm_studio_base_url }); setModels([]); } }
        catch (reason) { if (mounted.current) setError(failure(reason)); }
      }}><option value="lm_studio">LM Studio</option><option value="ollama">Ollama</option></select>
      <input aria-label="Local provider URL" value={settings.base_url} disabled={busy} onChange={e => persist({ ...settings, base_url: e.target.value, model: "" })} />
      <input aria-label="Local model ID" value={settings.model} disabled={busy} onChange={e => persist({ ...settings, model: e.target.value })} list="ming-prompt-models" />
      <datalist id="ming-prompt-models">{models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</datalist>
      <details><summary>Provider inference settings</summary>
        {(["temperature", "top_p", "top_k", "max_output_tokens", "context_length", "timeout_seconds"] as const).map(field =>
          <label key={field} className="block">{field.replaceAll("_", " ")}
            <input type="number" step={field === "temperature" || field === "top_p" ? "any" : 1}
              value={settings[field]} disabled={busy} onChange={e => {
                const value = Number(e.target.value);
                if (e.target.value && Number.isFinite(value)) persist({ ...settings, [field]: value });
              }} />
          </label>)}
      </details>
      <button type="button" disabled={loadingModels || busy} onClick={refresh}>Refresh models</button>
      <button type="button" disabled={busy || !args.prompt.trim() || !settings.model} onClick={preview}>Preview rewrite</button>
      {busy && <button type="button" onClick={() => { controller.current?.abort(); controller.current = null; setBusy(false); }}>Cancel preview</button>}
      <button type="button" disabled={busy} onClick={async () => { try { await clearMingPromptCache(); if (mounted.current) setResult(""); } catch (e) { if (mounted.current) setError(failure(e)); } }}>Clear cache</button>
      {result && <><Textarea label="Rewritten native prompt" value={result} rows={8} onChange={e => setResult(e.target.value)} /><button type="button" onClick={() => onApply(result)}>Apply</button></>}
    </div>}
  </details>;
}
