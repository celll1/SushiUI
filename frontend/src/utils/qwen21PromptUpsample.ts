import {
  getPromptAssistDefaults,
  getQwen21PromptUpsampleDefaults,
  transformQwen21Prompt,
} from "@/utils/api";
import type {
  Qwen21PromptUpsampleSettings,
  Qwen21PromptUpsampleResponse,
} from "@/utils/api";

export const QWEN21_UPSAMPLE_STORAGE_KEY = "qwen_image_21_prompt_upsample_v1";
let defaultsPromise: Promise<Qwen21PromptUpsampleSettings> | null = null;

export async function imageSourceToDataUrl(source: File | string): Promise<string> {
  if (typeof source === "string") {
    if (source.startsWith("data:image/")) return source;
    const response = await fetch(source);
    if (!response.ok) throw new Error(`Could not read reference image (${response.status})`);
    source = new File([await response.blob()], "reference-image");
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error || new Error("Could not read reference image"));
    reader.readAsDataURL(source);
  });
}

export function saveQwen21PromptUpsampleSettings(settings: Qwen21PromptUpsampleSettings): void {
  localStorage.setItem(QWEN21_UPSAMPLE_STORAGE_KEY, JSON.stringify({ ...settings, api_key: "" }));
  window.dispatchEvent(new CustomEvent("qwen21-prompt-upsample-settings", { detail: settings }));
}

export async function resolveQwen21PromptUpsampleSettings(): Promise<Qwen21PromptUpsampleSettings> {
  defaultsPromise ??= Promise.all([getQwen21PromptUpsampleDefaults(), getPromptAssistDefaults()]).then(
    ([defaults, local]) => ({
      ...defaults,
      base_url: defaults.base_url || local.lm_studio_base_url,
    }),
  );
  const defaults = await defaultsPromise;
  let saved: Partial<Qwen21PromptUpsampleSettings> = {};
  try {
    saved = JSON.parse(localStorage.getItem(QWEN21_UPSAMPLE_STORAGE_KEY) || "{}");
  } catch {}
  return { ...defaults, ...saved, api_key: "" };
}

export async function maybeUpsampleQwen21Prompt(args: {
  prompt: string;
  mode: "t2i" | "i2i";
  images?: Array<File | string>;
}): Promise<Qwen21PromptUpsampleResponse | null> {
  const settings = await resolveQwen21PromptUpsampleSettings();
  if (!settings.enabled) return null;
  if (settings.engine !== "official" && !settings.model) {
    throw new Error("Qwen prompt upsampling is enabled, but no local LLM model is selected.");
  }
  const images = await Promise.all((args.images ?? []).map(imageSourceToDataUrl));
  return transformQwen21Prompt({ ...settings, prompt: args.prompt, mode: args.mode, images });
}
