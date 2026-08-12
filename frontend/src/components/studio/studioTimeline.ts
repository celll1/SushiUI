import type { StudioAsset, StudioClip, StudioGenerationMode, StudioRange } from "./types";

export const frameDuration = (fps: number): number => 1 / Math.max(1, fps || 1);

export const frameIndexAt = (seconds: number, fps: number): number =>
  Math.max(0, Math.round(seconds * Math.max(1, fps || 1)));

export const normalizedRange = (range: StudioRange | null | undefined, duration: number): StudioRange | null => {
  if (!range) return null;
  const start = Math.max(0, Math.min(duration, Math.min(range.start, range.end)));
  const end = Math.max(start, Math.min(duration, Math.max(range.start, range.end)));
  return end - start > 0 ? { start, end } : null;
};

export const clipEnd = (clip: StudioClip): number => clip.start + clip.duration;

export const clipOverlapsRange = (clip: StudioClip, range: StudioRange): boolean =>
  clip.start < range.end && clipEnd(clip) > range.start;

export const clipContainsTime = (clip: StudioClip, time: number): boolean =>
  time >= clip.start && time < clipEnd(clip);

export const sourceEnd = (clip: StudioClip): number => clip.sourceIn + clip.duration;

export const maxTimelineDuration = (
  clip: StudioClip,
  asset: StudioAsset,
  projectDuration: number,
): number => {
  if (asset.kind === "image" && clip.presentation === "hold") {
    return Math.max(0, projectDuration - clip.start);
  }
  const available = Math.max(0, (clip.sourceDuration ?? asset.duration) - clip.sourceIn);
  return Math.max(0, Math.min(projectDuration - clip.start, available));
};

export interface StudioGenerationPlan {
  mode: StudioGenerationMode;
  outputRange: StudioRange;
  inpaintRange: StudioRange | null;
  videoClip: StudioClip | null;
  imageClip: StudioClip | null;
  hasVideoInput: boolean;
  hasImageInput: boolean;
  outputExtendsVideo: boolean;
}

export interface StudioGenerationPlanInput {
  isVideoModel: boolean;
  fps: number;
  projectDuration: number;
  playhead: number;
  outputRange?: StudioRange | null;
  inpaintRange?: StudioRange | null;
  selectedClipId?: string | null;
  clips: StudioClip[];
  assets: StudioAsset[];
}

/**
 * Converts the timeline's explicit selections into one endpoint family. The
 * caller still validates architecture-specific capabilities before sending.
 */
export const planStudioGeneration = ({
  isVideoModel,
  fps,
  projectDuration,
  playhead,
  outputRange: requestedOutput,
  inpaintRange: requestedInpaint,
  selectedClipId,
  clips,
  assets,
}: StudioGenerationPlanInput): StudioGenerationPlan => {
  const outputStart = Math.max(0, Math.min(Math.max(0, projectDuration - frameDuration(fps)), playhead));
  const outputRange = normalizedRange(requestedOutput, projectDuration) || {
    start: outputStart,
    end: Math.min(projectDuration, outputStart + frameDuration(fps)),
  };
  const normalizedInpaint = normalizedRange(requestedInpaint, projectDuration);
  const inpaintRange = normalizedInpaint
    ? normalizedRange({
      start: Math.max(outputRange.start, normalizedInpaint.start),
      end: Math.min(outputRange.end, normalizedInpaint.end),
    }, projectDuration)
    : null;
  const activeClips = clips.filter((clip) => clip.activeTake !== false);
  const assetFor = (clip: StudioClip) => assets.find((asset) => asset.id === clip.assetId);
  const selected = selectedClipId ? activeClips.find((clip) => clip.id === selectedClipId) : null;
  const clipsInOutput = activeClips.filter((clip) => clipOverlapsRange(clip, outputRange));
  const videoClip =
    (selected && clipsInOutput.includes(selected) && assetFor(selected)?.kind === "video" ? selected : null) ||
    clipsInOutput.find((clip) => assetFor(clip)?.kind === "video") ||
    null;
  const imageClip =
    (selected && clipsInOutput.includes(selected) && assetFor(selected)?.kind === "image" ? selected : null) ||
    clipsInOutput.find((clip) => assetFor(clip)?.kind === "image") ||
    null;
  const effectiveInpaintRange = inpaintRange && videoClip
    ? normalizedRange({
      start: Math.max(inpaintRange.start, videoClip.start),
      end: Math.min(inpaintRange.end, clipEnd(videoClip)),
    }, projectDuration)
    : inpaintRange;
  const hasVideoInput = !!videoClip;
  const hasImageInput = !!imageClip;
  const outputExtendsVideo = !!videoClip && (
    outputRange.start < videoClip.start || outputRange.end > clipEnd(videoClip)
  );

  let mode: StudioGenerationMode;
  if (!isVideoModel) {
    mode = hasImageInput ? (effectiveInpaintRange ? "image-inpaint" : "i2i") : "t2i";
  } else if (hasVideoInput && outputExtendsVideo) {
    mode = "outpaint";
  } else if (hasVideoInput && effectiveInpaintRange) {
    mode = "inpaint";
  } else if (hasVideoInput) {
    // A video source cannot be silently downgraded to text/image-to-video;
    // the user must mark the temporal region to regenerate.
    mode = "inpaint";
  } else if (hasImageInput) {
    mode = "i2v";
  } else {
    mode = "t2v";
  }

  return { mode, outputRange, inpaintRange: effectiveInpaintRange, videoClip, imageClip, hasVideoInput, hasImageInput, outputExtendsVideo };
};
