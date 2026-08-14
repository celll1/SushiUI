import type { StudioAsset, StudioClip, StudioRange } from "./types";
import { clipEnd, frameIndexAt } from "./studioTimeline";

const numberValue = (value: unknown): number | undefined => {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

export const frameTimeForClip = (clip: StudioClip, timelineTime: number): number =>
  Math.max(0, clip.sourceIn + Math.max(0, Math.min(clip.duration, timelineTime - clip.start)));

export const frameIndexForClipTime = (clip: StudioClip, timelineTime: number, fps: number): number =>
  frameIndexAt(frameTimeForClip(clip, timelineTime), fps);

export const sourceTrimFrames = (clip: StudioClip, sourceDuration: number, fps: number) => ({
  start: frameIndexAt(clip.sourceIn, fps),
  end: Math.max(0, frameIndexAt(Math.max(0, sourceDuration - clip.sourceIn - clip.duration), fps)),
});

export const videoInpaintFrames = (
  clip: StudioClip,
  inpaintRange: StudioRange,
  fps: number,
) => ({
  start: frameIndexAt(Math.max(0, inpaintRange.start - clip.start), fps),
  end: frameIndexAt(Math.max(0, inpaintRange.end - clip.start), fps),
});

/** Whether an outpaint plan is the one row `/generate/outpaint/video` reads
 * explicit references on, so a ref2va request that has them stays an extend
 * instead of becoming a fresh reference-composed clip. The endpoint takes
 * image references only, and ref2va serves only extend_forward -- offset 0 in
 * `videoOutpaintPlacement`'s own frame arithmetic, mirrored here so the two
 * cannot drift (backend: resolve_minimax_h3_outpaint_reference_gate). Any
 * other placement, or any non-image reference, keeps ref2v, where the
 * reference is actually honoured.
 */
export const outpaintReadsReferences = (
  mode: string,
  clip: StudioClip | null | undefined,
  outputRange: StudioRange,
  fps: number,
  referenceKinds: Array<StudioAsset["kind"] | undefined>,
): boolean =>
  mode === "outpaint" && !!clip
  && Math.max(0, frameIndexAt(clip.start - outputRange.start, fps)) === 0
  && referenceKinds.every((kind) => kind === "image");

export const videoOutpaintPlacement = (
  clip: StudioClip,
  outputRange: StudioRange,
  sourceDuration: number,
  fps: number,
) => ({
  totalFrames: Math.max(1, frameIndexAt(outputRange.end - outputRange.start, fps)),
  inputOffsetFrames: Math.max(0, frameIndexAt(clip.start - outputRange.start, fps)),
  inputTrimStartFrames: frameIndexAt(clip.sourceIn, fps),
  inputTrimEndFrames: Math.max(0, frameIndexAt(Math.max(0, sourceDuration - clip.sourceIn - clip.duration), fps)),
});

export interface GenerationAssetFallback {
  id: string;
  filename: string;
  kind: StudioAsset["kind"];
  url: string;
  masterUrl?: string;
  thumbnailUrl?: string;
  duration: number;
  width?: number;
  height?: number;
  source: StudioAsset["source"];
  prompt?: string;
  negativePrompt?: string;
  generationType?: string;
  modelName?: string;
  seed?: number;
  parameters?: Record<string, unknown>;
}

/** Preserve server-resolved Gallery metadata whenever a generation returned it. */
export const studioAssetFromGeneration = (
  result: any,
  fallback: GenerationAssetFallback,
): StudioAsset => {
  const image = result?.image || {};
  const filename = image.filename || fallback.filename;
  const playbackFilename = image.preview_filename || filename;
  const baseName = playbackFilename.replace(/\.[^/.]+$/, "");
  const galleryId = numberValue(image.id);
  return {
    id: galleryId != null ? `gallery-${galleryId}` : fallback.id,
    galleryId,
    name: filename,
    kind: fallback.kind,
    url: fallback.kind === "image" ? `/outputs/${filename}` : `/outputs/${playbackFilename}`,
    masterUrl: `/outputs/${filename}`,
    thumbnailUrl: fallback.kind === "audio" ? undefined : `/thumbnails/${baseName}.png`,
    duration: numberValue(image.duration) ?? fallback.duration,
    width: numberValue(image.width) ?? fallback.width,
    height: numberValue(image.height) ?? fallback.height,
    source: "generation",
    prompt: image.prompt ?? fallback.prompt,
    negativePrompt: image.negative_prompt ?? fallback.negativePrompt,
    createdAt: image.created_at,
    generationType: image.generation_type ?? fallback.generationType,
    modelName: image.model_name ?? fallback.modelName,
    seed: numberValue(image.seed) ?? fallback.seed,
    parameters: image.parameters ?? fallback.parameters,
  };
};

export const clipDurationInside = (clip: StudioClip, range: StudioRange): number =>
  Math.max(0, Math.min(clipEnd(clip), range.end) - Math.max(clip.start, range.start));
