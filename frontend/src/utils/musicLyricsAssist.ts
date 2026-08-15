import {
  getMusicLyricsAssistDefaults,
  getPromptAssistDefaults,
} from "@/utils/api";
import type {
  MusicLyricsAssistDefaults,
  MusicLyricsAssistSettings,
  PromptAssistDefaults,
} from "@/utils/api";

// Sibling of musicPromptAssist.ts (the caption rewriter's settings helper):
// same local-LLM provider/settings shape, its own storage key so a saved
// caption-rewriter provider/model choice and a saved lyrics-assistant one
// never overwrite each other. See docs/guides/MINIMAX_MUSIC3_DESIGN.md.
export const MUSIC3_LYRICS_ASSIST_STORAGE_KEY = "minimax_music3_lyrics_assist_settings_v1";

let defaultsPromise: Promise<MusicLyricsAssistDefaults> | null = null;
// H3's defaults, not a music-specific copy -- same reasoning as
// musicPromptAssist.ts: the server resolves an empty base_url from H3's
// PROMPT_ASSIST_DEFAULTS regardless of which rewriter/assistant is calling.
let baseUrlDefaultsPromise: Promise<PromptAssistDefaults> | null = null;

export function readMusicLyricsAssistSettings(): Partial<MusicLyricsAssistSettings> {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(localStorage.getItem(MUSIC3_LYRICS_ASSIST_STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

export function saveMusicLyricsAssistSettings(settings: MusicLyricsAssistSettings): void {
  localStorage.setItem(MUSIC3_LYRICS_ASSIST_STORAGE_KEY, JSON.stringify(settings));
  window.dispatchEvent(new CustomEvent("music3-lyrics-assist-settings", { detail: settings }));
}

export async function resolveMusicLyricsAssistSettings(): Promise<MusicLyricsAssistSettings> {
  defaultsPromise ??= getMusicLyricsAssistDefaults();
  baseUrlDefaultsPromise ??= getPromptAssistDefaults();
  const [defaults, baseUrlDefaults] = await Promise.all([defaultsPromise, baseUrlDefaultsPromise]);
  const saved = readMusicLyricsAssistSettings();
  const provider = saved.provider ?? defaults.provider;
  return {
    provider,
    base_url: saved.base_url ?? (
      provider === "lm_studio" ? baseUrlDefaults.lm_studio_base_url : baseUrlDefaults.ollama_base_url
    ),
    model: saved.model ?? "",
    api_key: saved.api_key ?? "",
    temperature: saved.temperature ?? defaults.temperature,
    top_p: saved.top_p ?? defaults.top_p,
    max_output_tokens: saved.max_output_tokens ?? defaults.max_output_tokens,
    context_length: saved.context_length ?? defaults.context_length,
    timeout_seconds: saved.timeout_seconds ?? defaults.timeout_seconds,
  };
}
