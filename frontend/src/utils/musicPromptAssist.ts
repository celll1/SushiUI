import {
  getMusicPromptAssistDefaults,
  getPromptAssistDefaults,
} from "@/utils/api";
import type {
  MusicPromptAssistDefaults,
  MusicPromptAssistSettings,
  PromptAssistDefaults,
} from "@/utils/api";

// Sibling of h3PromptAssist.ts: same local-LLM provider/settings shape, no
// mode/reference/auto-on-generate concepts, because a music caption has
// none of those. See docs/guides/MINIMAX_MUSIC3_DESIGN.md,
// "Caption rewriter (AI rewrite)".
export const MUSIC3_PROMPT_ASSIST_STORAGE_KEY = "minimax_music3_prompt_assist_settings_v1";

let defaultsPromise: Promise<MusicPromptAssistDefaults> | null = null;
// H3's defaults, not a music-specific copy: the server resolves an empty
// base_url from PROMPT_ASSIST_DEFAULTS (H3's) regardless of which rewriter
// is calling it, so the base-URL fallback shown here has to read the same
// source or it can silently show a host the server would not actually use.
let baseUrlDefaultsPromise: Promise<PromptAssistDefaults> | null = null;

export function readMusicPromptAssistSettings(): Partial<MusicPromptAssistSettings> {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(localStorage.getItem(MUSIC3_PROMPT_ASSIST_STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

export function saveMusicPromptAssistSettings(settings: MusicPromptAssistSettings): void {
  localStorage.setItem(MUSIC3_PROMPT_ASSIST_STORAGE_KEY, JSON.stringify(settings));
  window.dispatchEvent(new CustomEvent("music3-prompt-assist-settings", { detail: settings }));
}

export async function resolveMusicPromptAssistSettings(): Promise<MusicPromptAssistSettings> {
  defaultsPromise ??= getMusicPromptAssistDefaults();
  baseUrlDefaultsPromise ??= getPromptAssistDefaults();
  const [defaults, baseUrlDefaults] = await Promise.all([defaultsPromise, baseUrlDefaultsPromise]);
  const saved = readMusicPromptAssistSettings();
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
