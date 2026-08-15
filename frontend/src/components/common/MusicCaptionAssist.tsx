"use client";

import { useCallback, useEffect, useState } from "react";
import { Check, ChevronDown, ChevronUp, RotateCcw, Sparkles, X } from "lucide-react";
import Textarea from "./Textarea";
import {
  getPromptAssistDefaults,
  listPromptAssistModels,
  transformMusic3Caption,
} from "@/utils/api";
import type {
  MusicPromptAssistSettings,
  PromptAssistModel,
} from "@/utils/api";
import {
  resolveMusicPromptAssistSettings,
  saveMusicPromptAssistSettings,
} from "@/utils/musicPromptAssist";

// Sibling of H3PromptAssist for the MiniMax Music 3 caption rewriter. It
// reuses the same local-LLM model listing endpoint (models are not
// domain-specific) but has its own transform endpoint, its own settings
// storage, and no mode/template/reference concepts -- a music caption has
// none of those. See docs/guides/MINIMAX_MUSIC3_DESIGN.md,
// "Caption rewriter (AI rewrite)".
//
// Known duplication, left alone on purpose: the Provider/Local model
// select block below (state, refreshModels, switchProvider) is a near
// copy of the same block in H3PromptAssist.tsx. The design doc's "reuse
// H3's UI" line argues for a shared `usePromptAssistProvider` hook, but
// extracting it means editing H3's shipped component while another
// session is actively using that feature, and that risk isn't worth
// paying for tidiness alone right now. Route a third consumer of this
// block into the hook instead of a third copy.
//
// Base URL defaults are read from `getPromptAssistDefaults()` (H3's), not
// a music-specific copy: the server's `_prompt_assist_base_url()` resolves
// an empty base_url from the same H3 defaults for both rewriters, so a
// second copy here could only drift from what the server actually uses.

interface MusicCaptionAssistProps {
  caption: string;
  lyrics: string;
  onApply: (caption: string) => void;
}

function errorMessage(error: any): string {
  return error?.response?.data?.detail || error?.message || "Caption rewrite failed";
}

function modelSize(size?: number | null): string {
  if (!size) return "";
  return ` · ${(size / 1024 ** 3).toFixed(1)} GB`;
}

export default function MusicCaptionAssist({
  caption,
  lyrics,
  onApply,
}: MusicCaptionAssistProps) {
  const [open, setOpen] = useState(false);
  const [constraints, setConstraints] = useState("");
  const [settings, setSettings] = useState<MusicPromptAssistSettings | null>(null);
  const [models, setModels] = useState<PromptAssistModel[]>([]);
  const [result, setResult] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [valid, setValid] = useState(false);
  const [cached, setCached] = useState(false);
  const [busy, setBusy] = useState(false);
  const [modelBusy, setModelBusy] = useState(false);
  const [error, setError] = useState("");
  const [lastAppliedSource, setLastAppliedSource] = useState<string | null>(null);
  // Revise mode: `instruction` is what to change THIS TIME ("make the drop
  // harder"), distinct from `constraints` above (a standing rule, e.g. "no
  // drums"). Sent as its own field so the LLM never reads it as more
  // caption content to describe. `lastWasRevise` remembers which action
  // produced the current preview, so Retry repeats the same kind of call.
  const [instruction, setInstruction] = useState("");
  const [lastWasRevise, setLastWasRevise] = useState(false);
  const [diffSummary, setDiffSummary] = useState<string | null>(null);

  useEffect(() => {
    resolveMusicPromptAssistSettings().then(setSettings).catch((reason) => setError(errorMessage(reason)));
    const updateSettings = (event: Event) => {
      setSettings((event as CustomEvent<MusicPromptAssistSettings>).detail);
    };
    window.addEventListener("music3-prompt-assist-settings", updateSettings);
    return () => window.removeEventListener("music3-prompt-assist-settings", updateSettings);
  }, []);

  const persist = useCallback((next: MusicPromptAssistSettings) => {
    setSettings(next);
    saveMusicPromptAssistSettings({ ...next, api_key: "" });
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
    // H3's defaults, not a music-specific copy -- see the file-level note.
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

  const rewrite = async (options: { forceRefresh?: boolean; revise?: boolean } = {}) => {
    const forceRefresh = options.forceRefresh ?? false;
    const revise = options.revise ?? false;
    if (!settings?.model) {
      setError("Select a local LLM model first.");
      return;
    }
    if (revise && !instruction.trim()) {
      setError("Enter a revision instruction first.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const response = await transformMusic3Caption({
        ...settings,
        api_key: "",
        caption,
        lyrics,
        constraints,
        instruction: revise ? instruction : "",
        revise,
        force_refresh: forceRefresh,
      });
      setResult(response.prompt);
      setWarnings(response.warnings);
      setValid(response.valid);
      setCached(!!response.cached);
      setDiffSummary(typeof response.diff_summary === "string" ? response.diff_summary : null);
      setLastWasRevise(revise);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const apply = () => {
    if (!result) return;
    setLastAppliedSource(caption);
    onApply(result);
  };

  return (
    <div className="rounded border border-violet-500/30 bg-gray-900/70">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs text-gray-200 hover:bg-gray-800/70"
        aria-expanded={open}
      >
        <Sparkles size={14} className="text-violet-300" />
        <span className="font-medium">MiniMax Music 3 Caption Assist</span>
        {cached && <span className="text-[10px] text-emerald-300">cached</span>}
        <span className="ml-auto text-gray-400">{open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</span>
      </button>

      {open && (
        <div className="space-y-2 border-t border-gray-700/70 p-2.5" data-prompt-assist-open="true">
          <div className="grid gap-2 sm:grid-cols-[140px_minmax(180px,1fr)_auto]">
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

          <Textarea
            label="Constraints (optional)"
            placeholder='e.g. "no drums", "keep it instrumental", "120 bpm"'
            rows={2}
            value={constraints}
            onChange={(e) => setConstraints(e.target.value)}
          />
          <Textarea
            label="Revision instruction (optional — leave empty for a normal AI rewrite)"
            placeholder='e.g. "make the drop harder", "take it to 128 bpm"'
            rows={2}
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
          />
          <p className="text-xs text-gray-500">
            Expands Caption into a Structured Caption (Global Metadata / Vocal Details /
            Arrangement, 250-450 words). Lyrics are read for context only and are never
            rewritten or quoted. Revise applies the revision instruction to the Caption above
            as an edit, treating it as an already-expanded Structured Caption and preserving
            everything the instruction does not mention; AI rewrite ignores the instruction
            field and expands the Caption above as a new short caption, exactly as before.
            Cache is checked before model load; an LLM loaded for a rewrite is unloaded
            afterward. An LM Studio server with authentication enabled is not supported from
            this panel; the provider must accept unauthenticated local requests.
          </p>

          <div className="flex flex-wrap gap-1.5">
            <button
              type="button"
              onClick={() => rewrite({ revise: false })}
              className="rounded bg-violet-600 px-2.5 py-1.5 text-xs font-medium hover:bg-violet-500 disabled:opacity-50"
              disabled={busy || !caption.trim() || !settings?.model}
            >
              {busy && !lastWasRevise ? "Rewriting…" : "AI rewrite"}
            </button>
            <button
              type="button"
              onClick={() => rewrite({ revise: true })}
              className="rounded bg-amber-700 px-2.5 py-1.5 text-xs font-medium hover:bg-amber-600 disabled:opacity-50"
              disabled={busy || !caption.trim() || !instruction.trim() || !settings?.model}
              title="Apply the revision instruction to the Caption above, preserving everything it does not mention"
            >
              {busy && lastWasRevise ? "Revising…" : "Revise"}
            </button>
            {result && (
              <button
                type="button"
                onClick={() => rewrite({ forceRefresh: true, revise: lastWasRevise })}
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
                rows={10}
                value={result}
                onChange={(event) => setResult(event.target.value)}
                className="font-mono text-xs leading-relaxed"
              />
              {warnings.length > 0 && (
                <ul className="space-y-0.5 rounded bg-amber-950/30 px-2 py-1.5 text-xs text-amber-200">
                  {warnings.map((warning, index) => <li key={`${warning}-${index}`}>• {warning}</li>)}
                </ul>
              )}
              {diffSummary !== null && (
                <div className="space-y-1">
                  <p className="text-xs text-gray-400">What changed from the Caption above</p>
                  {diffSummary.trim() ? (
                    <pre className="max-h-40 overflow-auto rounded bg-gray-950/60 px-2 py-1.5 text-[11px] leading-relaxed text-gray-300">
                      {diffSummary}
                    </pre>
                  ) : (
                    <p className="rounded bg-gray-950/60 px-2 py-1.5 text-[11px] text-gray-400">
                      No line changes.
                    </p>
                  )}
                </div>
              )}
              <div className="flex flex-wrap items-center gap-1.5">
                <button
                  type="button"
                  onClick={apply}
                  className="flex items-center gap-1 rounded bg-emerald-700 px-2.5 py-1.5 text-xs hover:bg-emerald-600"
                  title={valid ? "Apply the validated result" : "Apply after reviewing the warnings"}
                >
                  <Check size={12} /> Apply to Caption
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
                    setDiffSummary(null);
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
