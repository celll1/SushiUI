import { loadImportedMedia, saveImportedMedia } from "./studioStorage";
import type { StudioAssetKind } from "./types";
import { newId } from "@/utils/id";

const TRANSFER_KEY = "sushiui_studio_transfer_v1";

export interface StudioTransferMedia {
  galleryId?: number;
  name?: string;
  kind: StudioAssetKind;
  url: string;
  masterUrl?: string;
  thumbnailUrl?: string;
  duration?: number;
  width?: number;
  height?: number;
  createdAt?: string;
  generationType?: string;
  modelName?: string;
  seed?: number;
}

export interface StudioTransferPayload {
  id: string;
  source: "generate" | "gallery";
  createdAt: string;
  media?: StudioTransferMedia & { blobKey?: string };
  prompt?: string;
  negativePrompt?: string;
  parameters?: Record<string, unknown>;
}

interface QueueStudioTransferOptions {
  source: StudioTransferPayload["source"];
  media?: StudioTransferMedia;
  prompt?: string;
  negativePrompt?: string;
  parameters?: object;
}

const STUDIO_PARAMETER_KEYS = [
  "prompt",
  "negative_prompt",
  "width",
  "height",
  "num_frames",
  "frame_rate",
  "fps",
  "num_inference_steps",
  "inference_steps",
  "steps",
  "guidance_scale",
  "cfg_scale",
  "seed",
  "audio_enable",
  "sampler",
  "schedule_type",
] as const;

const compactParameters = (parameters?: object): Record<string, unknown> | undefined => {
  if (!parameters) return undefined;
  const values = parameters as Record<string, unknown>;
  return Object.fromEntries(STUDIO_PARAMETER_KEYS.flatMap((key) =>
    values[key] === undefined ? [] : [[key, values[key]]],
  ));
};

const transientUrl = (url: string) => url.startsWith("blob:") || url.startsWith("data:");

export const queueStudioTransfer = async ({
  source,
  media,
  prompt,
  negativePrompt,
  parameters,
}: QueueStudioTransferOptions): Promise<void> => {
  let transferredMedia: StudioTransferPayload["media"] = media;
  if (media && transientUrl(media.url)) {
    const response = await fetch(media.url);
    const blob = await response.blob();
    const blobKey = `transfer-${newId()}`;
    await saveImportedMedia(blobKey, blob);
    transferredMedia = { ...media, url: "", masterUrl: undefined, thumbnailUrl: undefined, blobKey };
  }

  const payload: StudioTransferPayload = {
    id: newId(),
    source,
    createdAt: new Date().toISOString(),
    media: transferredMedia,
    prompt,
    negativePrompt,
    parameters: compactParameters(parameters),
  };
  localStorage.setItem(TRANSFER_KEY, JSON.stringify(payload));
};

export const takeStudioTransfer = (): StudioTransferPayload | null => {
  const raw = localStorage.getItem(TRANSFER_KEY);
  if (!raw) return null;
  localStorage.removeItem(TRANSFER_KEY);
  try {
    const parsed = JSON.parse(raw) as StudioTransferPayload;
    return parsed.id && parsed.source ? parsed : null;
  } catch {
    return null;
  }
};

export const resolveStudioTransferUrl = async (
  media: StudioTransferPayload["media"],
): Promise<string> => {
  if (!media) return "";
  if (media.url) return media.url;
  if (!media.blobKey) return "";
  const blob = await loadImportedMedia(media.blobKey);
  return blob ? URL.createObjectURL(blob) : "";
};
