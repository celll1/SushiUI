import { getMingPromptDefaults, getPromptAssistDefaults, transformMingPrompt, fetchGenerationDefaults } from "@/utils/api";
import type { MingPromptSettings, MingPromptRequest } from "@/utils/api";
import { imageSourceToDataUrl } from "@/utils/qwen21PromptUpsample";
const storageKey = "ming_image_01_prompt_upsample_v1";
export const mingPromptSettingsEvent = "ming-prompt-upsample-settings";
export async function resolveMingPromptSettings(): Promise<MingPromptSettings> {
  const [defaults, local] = await Promise.all([getMingPromptDefaults(), getPromptAssistDefaults()]);
  let saved: Partial<MingPromptSettings> = {};
  try { saved = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch {}
  const settings = { ...defaults, ...saved, api_key: "" };
  if (settings.engine !== "lm_studio" && settings.engine !== "ollama") settings.engine = defaults.engine;
  settings.base_url ||= settings.engine === "ollama" ? local.ollama_base_url : local.lm_studio_base_url;
  return settings;
}
export function saveMingPromptSettings(settings: MingPromptSettings): void {
  localStorage.setItem(storageKey, JSON.stringify({ ...settings, api_key: "" }));
  window.dispatchEvent(new CustomEvent(mingPromptSettingsEvent, { detail: settings }));
}
export type MingPromptArgs = Pick<MingPromptRequest, "prompt" | "mode" | "width" | "height"> & {
  layers?: number;
  images?: Array<File | string>;
};
export async function rewriteMingPrompt(args: MingPromptArgs, settings: MingPromptSettings, signal?: AbortSignal) {
  if (!settings.model) throw new Error("Select a local Ming prompt model first.");
  const layers = args.layers ?? (await fetchGenerationDefaults()).txt2img.ming_image_01_layers;
  if (typeof layers !== "number") throw new Error("Ming layer defaults are not available.");
  const images = await Promise.all((args.images || []).map(imageSourceToDataUrl));
  signal?.throwIfAborted();
  return transformMingPrompt({ ...settings, ...args, layers, images, api_key: "" }, signal);
}
export async function maybeUpsampleMingPrompt(args: MingPromptArgs) {
  const settings = await resolveMingPromptSettings();
  return settings.enabled ? rewriteMingPrompt(args, settings) : null;
}

const fileIds = new WeakMap<File, number>();
let nextFileId = 0;
export function mingPromptImageKey(images: Array<File | string> = []): string {
  return JSON.stringify(images.map(image => {
    if (typeof image === "string") return image;
    if (!fileIds.has(image)) fileIds.set(image, ++nextFileId);
    return fileIds.get(image);
  }));
}
