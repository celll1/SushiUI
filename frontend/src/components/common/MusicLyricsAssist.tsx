"use client";

import { useCallback, useEffect, useState } from "react";
import { Check, ChevronDown, ChevronUp, RotateCcw, Wand2, X } from "lucide-react";
import Textarea from "./Textarea";
import {
  formatMusic3Lyrics,
  getMusicLyricsAssistDefaults,
  getPromptAssistDefaults,
  listPromptAssistModels,
  transformMusic3Lyrics,
} from "@/utils/api";
import type {
  MusicLyricsAssistMode,
  MusicLyricsAssistSettings,
  PromptAssistModel,
} from "@/utils/api";
import {
  resolveMusicLyricsAssistSettings,
  saveMusicLyricsAssistSettings,
} from "@/utils/musicLyricsAssist";

// Sibling of MusicCaptionAssist for the MiniMax Music 3 lyrics assistant.
// See docs/guides/MINIMAX_MUSIC3_DESIGN.md, "Lyrics assistant".
// The motivating defect: the checkpoint's own lyric normalizer keeps only a
// leading structure tag on a line and silently drops any text sharing that
// line -- the most natural way to type lyrics destroys them, with no error
// anywhere. This panel offers three explicit, opt-in modes; it never
// auto-applies anything, and a user who types lyrics directly is
// unaffected unless they open this panel and press a button.
//
// - "format": deterministic layout fix, no LLM, no network settings.
// - "structure": the LLM writes ONLY the section/tag map for an
//   instrumental piece.
// - "complete": the user gives a theme and/or partial lyrics; the LLM
//   writes or finishes the words, preserving supplied lines verbatim.
//
// Known duplication, left alone on purpose -- same reasoning as
// MusicCaptionAssist.tsx's own note: the Provider/Local model select block
// is a near copy of the same block in both siblings. Route a fourth
// consumer of this block into a shared hook instead of a fourth copy.

interface MusicLyricsAssistProps {
  lyrics: string;
  onApply: (lyrics: string) => void;
}

function errorMessage(error: any): string {
  return error?.response?.data?.detail || error?.message || "Lyrics assist failed";
}

function modelSize(size?: number | null): string {
  if (!size) return "";
  return ` · ${(size / 1024 ** 3).toFixed(1)} GB`;
}

const MODE_LABELS: Record<MusicLyricsAssistMode, string> = {
  format: "Fix layout",
  structure: "Instrumental structure",
  complete: "Write / complete lyrics",
};

export default function MusicLyricsAssist({ lyrics, onApply }: MusicLyricsAssistProps) {
  const [open, setOpen] = useState(false);
  // "format" is only the pre-fetch fallback (a synchronous useState initial
  // value has to be something). The real default is
  // MUSIC_LYRICS_ASSIST_DEFAULTS["mode"] in backend/api/param_defaults.py,
  // fetched below and applied once -- unless the user has already picked a
  // mode themselves, which modeTouched guards against.
  const [mode, setModeState] = useState<MusicLyricsAssistMode>("format");
  const [modeTouched, setModeTouched] = useState(false);
  const setMode = useCallback((next: MusicLyricsAssistMode) => {
    setModeTouched(true);
    setModeState(next);
  }, []);
  const [theme, setTheme] = useState("");
  const [constraints, setConstraints] = useState("");
  const [settings, setSettings] = useState<MusicLyricsAssistSettings | null>(null);
  const [models, setModels] = useState<PromptAssistModel[]>([]);
  const [result, setResult] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [valid, setValid] = useState(true);
  const [cached, setCached] = useState(false);
  const [busy, setBusy] = useState(false);
  const [modelBusy, setModelBusy] = useState(false);
  const [error, setError] = useState("");
  const [lastAppliedSource, setLastAppliedSource] = useState<string | null>(null);

  const needsModel = mode !== "format";

  useEffect(() => {
    // The resolved server default for which tab starts selected -- applied
    // only if the user has not already clicked a mode button themselves.
    getMusicLyricsAssistDefaults()
      .then((defaults) => {
        setModeTouched((touched) => {
          if (!touched) setModeState(defaults.mode);
          return touched;
        });
      })
      .catch((reason) => setError(errorMessage(reason)));
  }, []);

  useEffect(() => {
    resolveMusicLyricsAssistSettings().then(setSettings).catch((reason) => setError(errorMessage(reason)));
    const updateSettings = (event: Event) => {
      setSettings((event as CustomEvent<MusicLyricsAssistSettings>).detail);
    };
    window.addEventListener("music3-lyrics-assist-settings", updateSettings);
    return () => window.removeEventListener("music3-lyrics-assist-settings", updateSettings);
  }, []);

  const persist = useCallback((next: MusicLyricsAssistSettings) => {
    setSettings(next);
    saveMusicLyricsAssistSettings({ ...next, api_key: "" });
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
    if (open && needsModel && settings && models.length === 0) refreshModels();
  }, [open, needsModel, settings, models.length, refreshModels]);

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

  const runFormat = async () => {
    setBusy(true);
    setError("");
    try {
      const response = await formatMusic3Lyrics(lyrics);
      setResult(response.lyrics);
      setWarnings(response.warnings);
      setValid(true);
      setCached(false);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const runTransform = async (forceRefresh = false) => {
    if (!settings?.model) {
      setError("Select a local LLM model first.");
      return;
    }
    if (mode !== "structure" && mode !== "complete") return;
    setBusy(true);
    setError("");
    try {
      const response = await transformMusic3Lyrics({
        ...settings,
        api_key: "",
        mode,
        theme,
        lyrics: mode === "complete" ? lyrics : "",
        constraints,
        force_refresh: forceRefresh,
      });
      setResult(response.lyrics);
      setWarnings(response.warnings);
      setValid(response.valid);
      setCached(!!response.cached);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const run = (forceRefresh = false) => {
    if (mode === "format") {
      runFormat();
    } else {
      runTransform(forceRefresh);
    }
  };

  const apply = () => {
    if (!result) return;
    setLastAppliedSource(lyrics);
    onApply(result);
  };

  const canRun = mode === "format"
    ? !busy && !!lyrics.trim()
    : !busy && !!settings?.model && (mode === "structure" ? !!theme.trim() : !!(theme.trim() || lyrics.trim()));

  return (
    <div className="rounded border border-violet-500/30 bg-gray-900/70">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs text-gray-200 hover:bg-gray-800/70"
        aria-expanded={open}
      >
        <Wand2 size={14} className="text-violet-300" />
        <span className="font-medium">MiniMax Music 3 Lyrics Assist</span>
        {cached && <span className="text-[10px] text-emerald-300">cached</span>}
        <span className="ml-auto text-gray-400">{open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</span>
      </button>

      {open && (
        <div className="space-y-2 border-t border-gray-700/70 p-2.5" data-prompt-assist-open="true">
          <div className="flex flex-wrap gap-1.5">
            {(Object.keys(MODE_LABELS) as MusicLyricsAssistMode[]).map((candidate) => (
              <button
                key={candidate}
                type="button"
                onClick={() => {
                  setMode(candidate);
                  setResult("");
                  setWarnings([]);
                  setError("");
                }}
                className={`rounded px-2.5 py-1 text-xs ${
                  mode === candidate
                    ? "bg-violet-600 font-medium text-white"
                    : "bg-gray-800 text-gray-300 hover:bg-gray-700"
                }`}
              >
                {MODE_LABELS[candidate]}
              </button>
            ))}
          </div>

          {mode === "format" && (
            <p className="text-xs text-gray-500">
              Fixes only the layout of the current Lyrics text: moves any words sharing a line
              with a structure tag onto their own line, puts one tag per line, and lowercases
              tag case. No LLM is used. Every word in Lyrics is preserved exactly.
            </p>
          )}

          {needsModel && (
            <>
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
                label={mode === "structure" ? "Section / arrangement description" : "Theme (optional)"}
                placeholder={
                  mode === "structure"
                    ? 'e.g. "intro, two verses, a big chorus, a breakdown, outro"'
                    : 'e.g. "a bittersweet farewell under city lights"'
                }
                rows={2}
                value={theme}
                onChange={(e) => setTheme(e.target.value)}
              />
              <Textarea
                label="Constraints (optional)"
                placeholder='e.g. "no explicit language", "keep it under a minute of lyrics"'
                rows={2}
                value={constraints}
                onChange={(e) => setConstraints(e.target.value)}
              />
              {mode === "structure" && (
                <p className="text-xs text-gray-500">
                  Emits only structure tags, one per line, no words — the control surface for an
                  instrumental track.
                </p>
              )}
              {mode === "complete" && (
                <p className="text-xs text-gray-500">
                  Any lines already in the Lyrics field are treated as partial lyrics and are
                  preserved verbatim; new sections are written around them. Leave Lyrics empty to
                  write from the theme alone.
                </p>
              )}
            </>
          )}

          <div className="flex flex-wrap gap-1.5">
            <button
              type="button"
              onClick={() => run(false)}
              className="rounded bg-violet-600 px-2.5 py-1.5 text-xs font-medium hover:bg-violet-500 disabled:opacity-50"
              disabled={!canRun}
            >
              {busy ? "Working…" : MODE_LABELS[mode]}
            </button>
            {result && mode !== "format" && (
              <button
                type="button"
                onClick={() => run(true)}
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
              <div className="flex flex-wrap items-center gap-1.5">
                <button
                  type="button"
                  onClick={apply}
                  className="flex items-center gap-1 rounded bg-emerald-700 px-2.5 py-1.5 text-xs hover:bg-emerald-600"
                  title={valid ? "Apply the validated result" : "Apply after reviewing the warnings"}
                >
                  <Check size={12} /> Apply to Lyrics
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
