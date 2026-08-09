import {
  getPromptAssistDefaults,
  transformH3Prompt,
} from "@/utils/api";
import type {
  H3PromptMode,
  PromptAssistDefaults,
  PromptAssistReference,
  PromptAssistSettings,
} from "@/utils/api";

export const H3_PROMPT_ASSIST_STORAGE_KEY = "minimax_h3_prompt_assist_settings_v1";
export const H3_PROMPT_EDITING_MODE_KEY = "minimax_h3_prompt_editing_mode";
export type H3PromptEditingMode = "natural-language" | "tags";

let defaultsPromise: Promise<PromptAssistDefaults> | null = null;

export function readH3PromptEditingMode(): H3PromptEditingMode {
  if (typeof window === "undefined") return "natural-language";
  return localStorage.getItem(H3_PROMPT_EDITING_MODE_KEY) === "tags"
    ? "tags"
    : "natural-language";
}

export function saveH3PromptEditingMode(mode: H3PromptEditingMode): void {
  localStorage.setItem(H3_PROMPT_EDITING_MODE_KEY, mode);
  window.dispatchEvent(new CustomEvent("h3-prompt-editing-mode", { detail: mode }));
}

export function createH3ReferenceInventory(counts: {
  pictures?: number;
  videos?: number;
  audios?: number;
}): PromptAssistReference[] {
  const references: PromptAssistReference[] = [];
  for (let index = 1; index <= (counts.pictures ?? 0); index += 1) {
    references.push({ token: `<Picture ${index}>`, kind: "picture", role: "reference" });
  }
  for (let index = 1; index <= (counts.videos ?? 0); index += 1) {
    references.push({ token: `<Video ${index}>`, kind: "video", role: "reference" });
  }
  for (let index = 1; index <= (counts.audios ?? 0); index += 1) {
    references.push({ token: `<Audio ${index}>`, kind: "audio", role: "reference" });
  }
  return references;
}

export function isStructuredH3Prompt(prompt: string): boolean {
  const base = [
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
  ];
  const fullReference = [
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
  ];
  const normalized = `\n${prompt.trim().toLowerCase()}`;
  return base.every((section) => normalized.includes(`\n${section}`))
    || fullReference.every((section) => normalized.includes(`\n${section}`));
}

export function readPromptAssistSettings(): Partial<PromptAssistSettings> {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(localStorage.getItem(H3_PROMPT_ASSIST_STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

export function savePromptAssistSettings(settings: PromptAssistSettings): void {
  localStorage.setItem(H3_PROMPT_ASSIST_STORAGE_KEY, JSON.stringify(settings));
  window.dispatchEvent(new CustomEvent("h3-prompt-assist-settings", { detail: settings }));
}

export async function resolvePromptAssistSettings(): Promise<PromptAssistSettings> {
  defaultsPromise ??= getPromptAssistDefaults();
  const defaults = await defaultsPromise;
  const saved = readPromptAssistSettings();
  const provider = saved.provider ?? defaults.provider;
  return {
    provider,
    base_url: saved.base_url ?? (
      provider === "lm_studio" ? defaults.lm_studio_base_url : defaults.ollama_base_url
    ),
    model: saved.model ?? "",
    api_key: saved.api_key ?? "",
    temperature: saved.temperature ?? defaults.temperature,
    top_p: saved.top_p ?? defaults.top_p,
    max_output_tokens: saved.max_output_tokens ?? defaults.max_output_tokens,
    context_length: saved.context_length ?? defaults.context_length,
    timeout_seconds: saved.timeout_seconds ?? defaults.timeout_seconds,
    auto_on_generate: saved.auto_on_generate ?? defaults.auto_on_generate,
  };
}

export async function maybeTransformH3PromptForGeneration(args: {
  prompt: string;
  mode: H3PromptMode;
  durationSeconds: number;
  references?: PromptAssistReference[];
}): Promise<{ prompt: string; cached: boolean; transformed: boolean }> {
  if (isStructuredH3Prompt(args.prompt)) {
    return { prompt: args.prompt, cached: false, transformed: false };
  }
  const settings = await resolvePromptAssistSettings();
  if (!settings.auto_on_generate) {
    return { prompt: args.prompt, cached: false, transformed: false };
  }
  if (!settings.model) {
    throw new Error("Prompt Assist is set to automatic, but no local LLM model is selected.");
  }
  const result = await transformH3Prompt({
    ...settings,
    prompt: args.prompt,
    mode: args.mode,
    duration_seconds: args.durationSeconds,
    references: args.references ?? [],
  });
  if (!result.valid) {
    throw new Error(`Prompt Assist validation failed: ${result.warnings.join(" ")}`);
  }
  return { prompt: result.prompt, cached: !!result.cached, transformed: true };
}
