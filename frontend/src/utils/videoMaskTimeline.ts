export type MaskInterpolation = "hold" | "affine" | "sdf";

/** All mask coordinates are pixels in the generated output canvas. */
export type MaskCoordinateSpace = "output_canvas";
export type MaskPolarity = "white_generate";

/**
 * Prevent accidental near-zero or runaway affine transforms in the API.
 * Must match `MIN_MASK_TRANSFORM_SCALE`/`MAX_MASK_TRANSFORM_SCALE` in
 * `backend/core/inference/video_mask_timeline.py` -- the frontend validates
 * this bound only so a bad manual entry fails locally instead of round-
 * tripping to the backend for the same 400.
 */
export const MIN_MASK_SCALE = 0.01;
export const MAX_MASK_SCALE = 100;
export const MAX_MASK_KEYFRAMES = 128;
export const MAX_MASK_ASSETS = 64;
export const MAX_COMPOSITE_FEATHER_PX = 128;

export interface MaskTransform {
  // Pivot note: the backend applies scale/rotation around the CANVAS
  // center (`source_center_xy` in video_mask_timeline.py is the output
  // canvas's own width/height midpoint), not the mask's own centroid. A
  // shape drawn off-center swings/scales around a point outside itself.
  /** Pixel translation in output_canvas coordinates; rotation uses the canvas center as pivot. */
  x: number;
  y: number;
  /** Positive, bounded scale factors applied around the canvas center. */
  scaleX: number;
  scaleY: number;
  /** Clockwise rotation in degrees around the canvas center. */
  rotation: number;
}

export interface VideoMaskCanvas {
  width: number;
  height: number;
}

export interface VideoMaskKeyframe {
  id: string;
  frame: number;
  /** Stable multipart mapping key; it must match VideoMaskAsset.id. */
  maskId: string;
  interpolationToNext: MaskInterpolation;
  transform: MaskTransform;
}

export interface VideoMaskManifest {
  version: number;
  coordinateSpace: MaskCoordinateSpace;
  canvas: VideoMaskCanvas;
  polarity: MaskPolarity;
  keyframes: VideoMaskKeyframe[];
  compositeFeatherPx: number;
  /** Optional local assets used to validate maskId references before upload. */
  assets?: VideoMaskAsset[];
}

export interface VideoMaskAsset {
  /** Send this value as the multipart part's `mask_id`; do not rely on array order. */
  id: string;
  dataUrl: string;
  /**
   * The output-canvas size (in pixels) this asset's PNG was actually
   * rendered at when it was saved. Optional because assets loaded from an
   * older manifest (or a "send to inpaint" round-trip predating this field)
   * may not carry it; callers that need a strict per-asset size check
   * should treat a missing value as "unknown, do not assume it matches".
   * Not part of the wire DTO -- assets travel as separate multipart PNG
   * parts, and the backend reads their real dimensions from the PNG itself
   * (see `_decode_png` in `video_mask_timeline.py`), so this field never
   * needs to be serialized.
   */
  width?: number;
  height?: number;
}

export interface VideoMaskValidationResult {
  valid: boolean;
  errors: string[];
  manifest?: VideoMaskManifest;
}

/** Backend DTO. The multipart asset part uses the same `mask_id` value. */
export interface VideoMaskTransformDto {
  x: number;
  y: number;
  scale_x: number;
  scale_y: number;
  rotation: number;
}

export interface VideoMaskKeyframeDto {
  id: string;
  frame: number;
  mask_id: string;
  interpolation_to_next: MaskInterpolation;
  transform: VideoMaskTransformDto;
}

export interface VideoMaskManifestDto {
  version: 1;
  coordinate_space: MaskCoordinateSpace;
  canvas: VideoMaskCanvas;
  polarity: MaskPolarity;
  keyframes: VideoMaskKeyframeDto[];
  composite_feather_px: number;
}

export const DEFAULT_MASK_INTERPOLATION: MaskInterpolation = "hold";

export function createDefaultMaskTransform(): MaskTransform {
  return { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0 };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isMaskInterpolation(value: unknown): value is MaskInterpolation {
  return value === "hold" || value === "affine" || value === "sdf";
}

function isCoordinateSpace(value: unknown): value is MaskCoordinateSpace {
  return value === "output_canvas";
}

function isMaskPolarity(value: unknown): value is MaskPolarity {
  return value === "white_generate";
}

export function validateMaskTransform(transform: unknown, path = "transform"): string[] {
  if (!isRecord(transform)) return [`${path} must be an object.`];

  const errors: string[] = [];
  for (const field of ["x", "y", "rotation"] as const) {
    if (!isFiniteNumber(transform[field])) errors.push(`${path}.${field} must be finite.`);
  }
  for (const field of ["scaleX", "scaleY"] as const) {
    if (
      !isFiniteNumber(transform[field]) ||
      transform[field] < MIN_MASK_SCALE ||
      transform[field] > MAX_MASK_SCALE
    ) {
      errors.push(
        `${path}.${field} must be finite and between ${MIN_MASK_SCALE} and ${MAX_MASK_SCALE}.`,
      );
    }
  }
  return errors;
}

export function validateVideoMaskKeyframe(keyframe: unknown, index?: number): string[] {
  const path = index === undefined ? "keyframe" : `keyframes[${index}]`;
  if (!isRecord(keyframe)) return [`${path} must be an object.`];

  const errors: string[] = [];
  if (typeof keyframe.id !== "string" || keyframe.id.trim() === "") {
    errors.push(`${path}.id must be a non-empty string.`);
  }
  if (!Number.isInteger(keyframe.frame) || (keyframe.frame as number) < 0) {
    errors.push(`${path}.frame must be a non-negative integer.`);
  }
  if (typeof keyframe.maskId !== "string" || keyframe.maskId.trim() === "") {
    errors.push(`${path}.maskId must be a non-empty string.`);
  }
  if (keyframe.interpolationToNext !== undefined && !isMaskInterpolation(keyframe.interpolationToNext)) {
    errors.push(`${path}.interpolationToNext must be hold, affine, or sdf.`);
  }
  errors.push(...validateMaskTransform(keyframe.transform, `${path}.transform`));
  return errors;
}

export function validateVideoMaskManifest(value: unknown): VideoMaskValidationResult {
  if (!isRecord(value)) return { valid: false, errors: ["Manifest must be an object."] };

  const errors: string[] = [];
  if (value.version !== 1) {
    errors.push("version must be exactly 1.");
  }
  if (!isCoordinateSpace(value.coordinateSpace)) {
    errors.push("coordinateSpace must be output_canvas.");
  }
  if (!isRecord(value.canvas)) {
    errors.push("canvas must be an object.");
  } else {
    if (!Number.isInteger(value.canvas.width) || (value.canvas.width as number) <= 0) {
      errors.push("canvas.width must be a positive integer.");
    }
    if (!Number.isInteger(value.canvas.height) || (value.canvas.height as number) <= 0) {
      errors.push("canvas.height must be a positive integer.");
    }
  }
  if (!isMaskPolarity(value.polarity)) {
    errors.push("polarity must be white_generate.");
  }
  if (!isFiniteNumber(value.compositeFeatherPx) || value.compositeFeatherPx < 0) {
    errors.push("compositeFeatherPx must be a non-negative finite number.");
  } else if (value.compositeFeatherPx > MAX_COMPOSITE_FEATHER_PX) {
    errors.push(`compositeFeatherPx must be at most ${MAX_COMPOSITE_FEATHER_PX}.`);
  }
  if (!Array.isArray(value.keyframes)) {
    errors.push("keyframes must be an array.");
  } else {
    if (value.keyframes.length > MAX_MASK_KEYFRAMES) {
      errors.push(`keyframes may contain at most ${MAX_MASK_KEYFRAMES} entries.`);
    }
    const seenFrames = new Set<number>();
    const seenIds = new Set<string>();
    let previousFrame: number | null = null;
    value.keyframes.forEach((keyframe, index) => {
      errors.push(...validateVideoMaskKeyframe(keyframe, index));
      if (!isRecord(keyframe)) return;
      if (typeof keyframe.id === "string" && seenIds.has(keyframe.id)) {
        errors.push(`keyframes contains duplicate id ${keyframe.id}.`);
      }
      if (typeof keyframe.id === "string") seenIds.add(keyframe.id);
      if (!Number.isInteger(keyframe.frame)) return;

      const frame = keyframe.frame as number;
      if (seenFrames.has(frame)) errors.push(`keyframes contains duplicate frame ${frame}.`);
      if (previousFrame !== null && frame < previousFrame) {
        errors.push(`keyframes must be sorted by frame at index ${index}.`);
      }
      seenFrames.add(frame);
      previousFrame = frame;
    });
    value.keyframes.forEach((keyframe, index) => {
      if (!isRecord(keyframe) || index >= value.keyframes.length - 1) return;
      const nextKeyframe = value.keyframes[index + 1];
      if (
        isRecord(nextKeyframe) &&
        keyframe.interpolationToNext === "affine" &&
        keyframe.maskId !== nextKeyframe.maskId
      ) {
        errors.push(`keyframes[${index}].affine requires the same maskId as the next keyframe.`);
      }
    });
  }

  if (value.assets !== undefined) {
    errors.push(...validateVideoMaskAssets(value.assets));
    if (Array.isArray(value.assets) && Array.isArray(value.keyframes)) {
      const assetIds = new Set(value.assets.map((asset) => (isRecord(asset) ? asset.id : undefined)));
      value.keyframes.forEach((keyframe, index) => {
        if (!isRecord(keyframe) || typeof keyframe.maskId !== "string") return;
        if (!assetIds.has(keyframe.maskId)) {
          errors.push(`keyframes[${index}].maskId has no matching asset: ${keyframe.maskId}.`);
        }
      });
    }
  }

  if (errors.length > 0) return { valid: false, errors };
  return { valid: true, errors: [], manifest: value as unknown as VideoMaskManifest };
}

export function validateVideoMaskAssets(assets: unknown): string[] {
  if (!Array.isArray(assets)) return ["assets must be an array."];
  const errors: string[] = [];
  if (assets.length > MAX_MASK_ASSETS) {
    errors.push(`assets may contain at most ${MAX_MASK_ASSETS} entries.`);
  }
  const seenIds = new Set<string>();
  assets.forEach((asset, index) => {
    if (!isRecord(asset)) {
      errors.push(`assets[${index}] must be an object.`);
      return;
    }
    if (typeof asset.id !== "string" || asset.id.trim() === "") {
      errors.push(`assets[${index}].id must be a non-empty string.`);
    } else if (seenIds.has(asset.id)) {
      errors.push(`assets contains duplicate id ${asset.id}.`);
    } else {
      seenIds.add(asset.id);
    }
    if (typeof asset.dataUrl !== "string" || asset.dataUrl.trim() === "") {
      errors.push(`assets[${index}].dataUrl must be a non-empty string.`);
    }
  });
  return errors;
}

export function videoMaskManifestToDto(
  manifest: VideoMaskManifest,
  assets?: VideoMaskAsset[],
): VideoMaskManifestDto {
  const assetsToValidate = assets ?? manifest.assets;
  const manifestForValidation = assetsToValidate
    ? { ...manifest, assets: assetsToValidate }
    : manifest;
  const validation = validateVideoMaskManifest(manifestForValidation);
  if (!validation.valid) throw new Error(`Invalid video mask manifest: ${validation.errors.join(" ")}`);
  return {
    version: 1,
    coordinate_space: "output_canvas",
    canvas: { ...manifest.canvas },
    polarity: manifest.polarity,
    keyframes: manifest.keyframes.map((keyframe) => ({
      id: keyframe.id,
      frame: keyframe.frame,
      mask_id: keyframe.maskId,
      interpolation_to_next: keyframe.interpolationToNext ?? DEFAULT_MASK_INTERPOLATION,
      transform: {
        x: keyframe.transform.x,
        y: keyframe.transform.y,
        scale_x: keyframe.transform.scaleX,
        scale_y: keyframe.transform.scaleY,
        rotation: keyframe.transform.rotation,
      },
    })),
    composite_feather_px: manifest.compositeFeatherPx,
  };
}

/** Serialize the internal camelCase model into the stable snake_case API DTO. */
export function serializeVideoMaskManifestForApi(
  manifest: VideoMaskManifest,
  assets?: VideoMaskAsset[],
): string {
  return JSON.stringify(videoMaskManifestToDto(manifest, assets));
}

/** Backwards-compatible alias for callers that used the original serializer name. */
export const serializeVideoMaskManifest = serializeVideoMaskManifestForApi;
export const videoMaskManifestToJSON = serializeVideoMaskManifestForApi;

export function videoMaskManifestFromDto(value: unknown): VideoMaskValidationResult {
  if (!isRecord(value)) return { valid: false, errors: ["Manifest DTO must be an object."] };
  const keyframes = Array.isArray(value.keyframes)
    ? value.keyframes.map((keyframe) => {
        if (!isRecord(keyframe)) return keyframe;
        const transform = keyframe.transform;
        return {
          id: keyframe.id,
          frame: keyframe.frame,
          maskId: keyframe.mask_id,
          interpolationToNext: keyframe.interpolation_to_next ?? DEFAULT_MASK_INTERPOLATION,
          transform: isRecord(transform)
            ? {
                x: transform.x,
                y: transform.y,
                scaleX: transform.scale_x,
                scaleY: transform.scale_y,
                rotation: transform.rotation,
              }
            : transform,
        };
      })
    : value.keyframes;
  return validateVideoMaskManifest({
    version: value.version,
    coordinateSpace: value.coordinate_space,
    canvas: value.canvas,
    polarity: value.polarity,
    keyframes,
    compositeFeatherPx: value.composite_feather_px,
  });
}

export function parseVideoMaskManifest(json: string): VideoMaskValidationResult {
  try {
    return videoMaskManifestFromDto(JSON.parse(json) as unknown);
  } catch {
    return { valid: false, errors: ["Manifest JSON is invalid."] };
  }
}

export const videoMaskManifestFromJSON = parseVideoMaskManifest;

export function sortKeyframes(keyframes: VideoMaskKeyframe[]): VideoMaskKeyframe[] {
  return [...keyframes].sort((a, b) => a.frame - b.frame || a.id.localeCompare(b.id));
}

/** Upsert is keyed by id; callers can validate duplicate frame positions separately. */
export function upsertKeyframe(
  keyframes: VideoMaskKeyframe[],
  keyframe: VideoMaskKeyframe,
): VideoMaskKeyframe[] {
  const replaced = keyframes.some((existing) => existing.id === keyframe.id);
  const next = replaced
    ? keyframes.map((existing) => (existing.id === keyframe.id ? keyframe : existing))
    : [...keyframes, keyframe];
  return sortKeyframes(next);
}

export function removeKeyframe(
  keyframes: VideoMaskKeyframe[],
  keyframeId: string,
): VideoMaskKeyframe[] {
  return keyframes.filter((keyframe) => keyframe.id !== keyframeId);
}

export function keyframeAtOrBefore(
  keyframes: VideoMaskKeyframe[],
  frame: number,
): VideoMaskKeyframe | null {
  let result: VideoMaskKeyframe | null = null;
  for (const keyframe of sortKeyframes(keyframes)) {
    if (keyframe.frame > frame) break;
    result = keyframe;
  }
  return result;
}

export function clampFrame(frame: number, minFrame: number, maxFrame: number): number {
  const min = Math.min(Math.round(minFrame), Math.round(maxFrame));
  const max = Math.max(Math.round(minFrame), Math.round(maxFrame));
  const candidate = Number.isFinite(frame) ? Math.round(frame) : min;
  return Math.max(min, Math.min(max, candidate));
}

export function clampKeyframe(
  keyframe: VideoMaskKeyframe,
  minFrame: number,
  maxFrame: number,
): VideoMaskKeyframe {
  return {
    ...keyframe,
    frame: clampFrame(keyframe.frame, minFrame, maxFrame),
    transform: { ...keyframe.transform },
  };
}

export function validateKeyframes(keyframes: VideoMaskKeyframe[]): string[] {
  const errors: string[] = [];
  const seenFrames = new Set<number>();
  const seenIds = new Set<string>();
  let previousFrame: number | null = null;
  keyframes.forEach((keyframe, index) => {
    errors.push(...validateVideoMaskKeyframe(keyframe, index));
    if (seenFrames.has(keyframe.frame)) errors.push(`keyframes contains duplicate frame ${keyframe.frame}.`);
    if (seenIds.has(keyframe.id)) errors.push(`keyframes contains duplicate id ${keyframe.id}.`);
    if (previousFrame !== null && keyframe.frame < previousFrame) {
      errors.push(`keyframes must be sorted by frame at index ${index}.`);
    }
    seenFrames.add(keyframe.frame);
    seenIds.add(keyframe.id);
    previousFrame = keyframe.frame;
  });
  return errors;
}

/** UI helper: drop malformed, out-of-range, duplicate-id, and duplicate-frame entries. */
export function pruneKeyframesToFrameRange(
  keyframes: VideoMaskKeyframe[],
  minFrame: number,
  maxFrame: number,
): VideoMaskKeyframe[] {
  const min = Math.min(Math.round(minFrame), Math.round(maxFrame));
  const max = Math.max(Math.round(minFrame), Math.round(maxFrame));
  const seenIds = new Set<string>();
  const seenFrames = new Set<number>();
  return sortKeyframes(
    keyframes.filter((keyframe) => {
      if (!Number.isInteger(keyframe.frame) || keyframe.frame < min || keyframe.frame > max) return false;
      if (seenIds.has(keyframe.id) || seenFrames.has(keyframe.frame)) return false;
      seenIds.add(keyframe.id);
      seenFrames.add(keyframe.frame);
      return true;
    }),
  );
}
