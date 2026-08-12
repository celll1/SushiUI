/**
 * Client-side rasterization for `hold`/`affine` video mask interpolation
 * (L0 layer, dependency-free except for the DOM `<canvas>` 2D context).
 *
 * This is a from-scratch port of the SAME two branches of
 * `backend/core/inference/video_mask_timeline.py`'s `rasterize_mask_timeline`
 * (line 651) and `apply_mask_transform` (line 455) -- not an approximation of
 * a different algorithm. `sdf` interpolation (the third branch, a signed-
 * distance-field morph between two DIFFERENT mask assets) is intentionally
 * NOT implemented here; callers must detect it with
 * {@link canRasterizeMaskManifestClientSide} and fall back to the backend
 * `/video-mask/preview` endpoint for any manifest where it applies.
 *
 * Pixel-exactness with the backend's `scipy.ndimage.affine_transform(...,
 * order=1)` is not attempted or required -- Canvas 2D's own resampling filter
 * differs, and the design this module implements only promises "visually
 * equivalent" resampling, not a byte-identical filter. Compositing (what
 * fills the area outside the transformed mask) is a separate concern, and
 * is the caller's responsibility to match -- see `apply_mask_transform`'s
 * `cval=0.0` (opaque black outside the transformed source footprint).
 */

import {
  DEFAULT_MASK_INTERPOLATION,
  sortKeyframes,
  type MaskTransform,
  type VideoMaskKeyframe,
} from "./videoMaskTimeline";

/** The 6 numbers `CanvasRenderingContext2D.setTransform` takes, in its own (a, b, c, d, e, f) order. */
export interface CanvasAffineMatrix {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
}

/**
 * Linear interpolation of all five transform scalars. Verbatim port of
 * `_interpolate_transform` (video_mask_timeline.py line 521): a plain lerp
 * on `rotation`, not a shortest-arc/quaternion-style interpolation, so a
 * transform pair spanning e.g. 350deg -> 10deg sweeps the LONG way around
 * (340deg of travel), exactly like the backend.
 */
export function interpolateMaskTransform(
  left: MaskTransform,
  right: MaskTransform,
  amount: number,
): MaskTransform {
  return {
    x: left.x + (right.x - left.x) * amount,
    y: left.y + (right.y - left.y) * amount,
    scaleX: left.scaleX + (right.scaleX - left.scaleX) * amount,
    scaleY: left.scaleY + (right.scaleY - left.scaleY) * amount,
    rotation: left.rotation + (right.rotation - left.rotation) * amount,
  };
}

/**
 * Builds the `setTransform(a, b, c, d, e, f)` matrix that maps a mask image
 * of `sourceWidth` x `sourceHeight` onto a `targetWidth` x `targetHeight`
 * canvas the same way `apply_mask_transform` (video_mask_timeline.py line
 * 455) maps it onto its `output_shape`, so that `drawImage(maskImage, 0, 0)`
 * under this transform lands the mask exactly where the backend would.
 *
 * `M = R(rotation) . diag(scaleX, scaleY)`, pivoted at each canvas's own
 * center. scipy's pivot is `((w-1)/2, (h-1)/2)` on integer pixel INDICES;
 * Canvas 2D coordinates are continuous with a pixel's center at
 * `index + 0.5`, so the same pivot is exactly `(w/2, h/2)` in canvas space --
 * this is a coordinate-space translation of the SAME pivot, not a different
 * one.
 */
export function maskTransformToCanvasMatrix(
  transform: MaskTransform,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): CanvasAffineMatrix {
  const angle = (transform.rotation * Math.PI) / 180;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);

  const a = transform.scaleX * cosine;
  const b = transform.scaleX * sine;
  const c = -transform.scaleY * sine;
  const d = transform.scaleY * cosine;
  const e = -a * (sourceWidth / 2) - c * (sourceHeight / 2) + targetWidth / 2 + transform.x;
  const f = -b * (sourceWidth / 2) - d * (sourceHeight / 2) + targetHeight / 2 + transform.y;

  return { a, b, c, d, e, f };
}

/** A frame resolves to one mask asset shown verbatim, or a transform-interpolated draw of one mask asset. */
export interface ResolvedMaskFrame {
  maskId: string;
  transform: MaskTransform;
}

/** The governing segment uses `sdf`; the caller must ask the backend for this frame instead. */
export interface ResolvedMaskNeedsServer {
  needsServer: true;
}

/**
 * Resolves which mask asset + transform to draw for `frame`, following the
 * EXACT same keyframe-pair search and branch selection as
 * `rasterize_mask_timeline`'s per-frame loop (video_mask_timeline.py line
 * 723): `right_index` is the first keyframe strictly after `frame`,
 * `left_index = right_index - 1`, and:
 *   - before the first keyframe -> that keyframe's own mask, verbatim
 *   - after the last keyframe -> that keyframe's own mask, verbatim
 *   - otherwise, `left.interpolationToNext` selects `hold` (left mask,
 *     verbatim), `affine` (left mask, transform lerped toward right's), or
 *     `sdf` (not handled here -- see {@link ResolvedMaskNeedsServer}).
 *
 * Returns `null` when there are no keyframes at all (nothing to draw).
 */
export function resolveMaskAtFrame(
  keyframes: VideoMaskKeyframe[],
  frame: number,
): ResolvedMaskFrame | ResolvedMaskNeedsServer | null {
  if (keyframes.length === 0) return null;
  const sorted = sortKeyframes(keyframes);

  let rightIndex = 0;
  while (rightIndex < sorted.length && sorted[rightIndex].frame <= frame) rightIndex++;
  const leftIndex = rightIndex - 1;

  if (leftIndex < 0) {
    const first = sorted[0];
    return { maskId: first.maskId, transform: first.transform };
  }
  if (rightIndex >= sorted.length) {
    const last = sorted[sorted.length - 1];
    return { maskId: last.maskId, transform: last.transform };
  }

  const left = sorted[leftIndex];
  const right = sorted[rightIndex];
  const mode = left.interpolationToNext ?? DEFAULT_MASK_INTERPOLATION;

  if (mode === "sdf") return { needsServer: true };
  if (mode === "hold") return { maskId: left.maskId, transform: left.transform };

  // affine -- the parser (and validateVideoMaskManifest) already enforce
  // left.maskId === right.maskId for this mode, so this is unreachable for
  // any manifest that went through validation. But this function also takes
  // a raw `keyframes` array straight from live panel state, which could in
  // principle be mid-edit and not yet re-validated; mirror the backend's own
  // refusal (`apply_mask_transform`'s "affine interpolation requires the
  // same mask_id" raise, video_mask_timeline.py line 755) rather than
  // silently blending toward the wrong asset -- returning `null` here (no
  // draw) is the safest analogue available to a function that isn't allowed
  // to throw into a render effect.
  if (left.maskId !== right.maskId) return null;
  const amount = (frame - left.frame) / (right.frame - left.frame);
  return { maskId: left.maskId, transform: interpolateMaskTransform(left.transform, right.transform, amount) };
}

/**
 * True when every governing keyframe-pair segment in `keyframes` uses
 * `hold` or `affine` (never `sdf`), i.e. every frame that
 * {@link resolveMaskAtFrame} could be asked to resolve for this manifest can
 * be rasterized client-side without ever hitting {@link ResolvedMaskNeedsServer}.
 *
 * Only `interpolationToNext` on all but the LAST keyframe governs a
 * displayable segment (the last keyframe has no "next", so its own
 * `interpolationToNext` -- if present -- never selects a branch in
 * `resolveMaskAtFrame`); mirrors the loop bound `rasterize_mask_timeline`
 * itself uses (segments only exist between consecutive keyframes).
 */
export function canRasterizeMaskManifestClientSide(keyframes: VideoMaskKeyframe[]): boolean {
  const sorted = sortKeyframes(keyframes);
  for (let index = 0; index < sorted.length - 1; index++) {
    const mode = sorted[index].interpolationToNext ?? DEFAULT_MASK_INTERPOLATION;
    if (mode === "sdf") return false;
  }
  return true;
}
