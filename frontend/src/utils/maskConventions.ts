/**
 * Mask conventions (L0 layer).
 *
 * Single source of truth for cross-cutting mask conventions that were
 * previously duplicated across ImageEditor (canvas compositing + PNG
 * encoding), InpaintPanel (CSS overlay preview), and videoMaskTimeline
 * (wire-format polarity type). Downstream consumers must import from here
 * instead of redefining these constants/types locally.
 *
 * This module is intentionally dependency-free (no React, no complex DOM
 * APIs) except for `encodeMaskLayerToPng`, which takes an HTMLCanvasElement
 * because the mask-layer PNG encoding logic itself lives here now.
 */

// ---------------------------------------------------------------------------
// Mask polarity
// ---------------------------------------------------------------------------

/**
 * "white_generate": white/opaque mask pixels mark the region to
 * generate/inpaint; black/transparent pixels are preserved from the source
 * image. This is the only polarity currently supported end-to-end (canvas
 * drawing, PNG encoding here, and the backend wire format consumed in
 * backend/core/inference/video_mask_timeline.py).
 */
export const MASK_POLARITY = "white_generate" as const;

/**
 * Re-exported under the same name videoMaskTimeline.ts used to declare
 * locally, so the wire-format DTOs there keep compiling unchanged while
 * pointing at this single definition.
 */
export type MaskPolarity = typeof MASK_POLARITY;

// ---------------------------------------------------------------------------
// Mask overlay compositing (on-screen preview only; never encoded into
// saved output pixels)
// ---------------------------------------------------------------------------

/** Blend mode used to preview the mask layer as a semi-transparent white overlay. */
export const MASK_OVERLAY_BLEND_MODE = "screen" as const;

/** Opacity applied to the mask overlay preview. */
export const MASK_OVERLAY_ALPHA = 0.5;

/**
 * Luminance threshold (0-255, R channel of an already-grayscale
 * `MASK_POLARITY` PNG) above which a pixel counts as "white" (marked for
 * generation) rather than "black" (preserved). Matches the backend's own
 * `>= 0.5` on a [0, 1]-normalized mask (`0.5 * 255 = 127.5`); frontend
 * callers that need a binary white/black check on decoded mask pixel data
 * (e.g. "does this mask have anything drawn at all") should compare against
 * this constant rather than re-deriving their own copy of the same number.
 */
export const MASK_WHITE_LUMINANCE_THRESHOLD = 127;

/**
 * Canvas 2D context and CSS use different vocabularies for the same blend
 * concept; both consumers should read from these so the two preview paths
 * (ImageEditor's <canvas> compositing and InpaintPanel's CSS <img> overlay)
 * cannot drift apart again.
 */
export const MASK_OVERLAY_CANVAS_COMPOSITE_OPERATION: GlobalCompositeOperation =
  MASK_OVERLAY_BLEND_MODE;
export const MASK_OVERLAY_CSS_MIX_BLEND_MODE = MASK_OVERLAY_BLEND_MODE;

// ---------------------------------------------------------------------------
// Mask-layer PNG encoding
// ---------------------------------------------------------------------------

/**
 * Encode a mask layer canvas (RGBA, alpha-based drawing) into an opaque
 * grayscale PNG data URL matching MASK_POLARITY ("white_generate"):
 *   - alpha > 0 (drawn) pixels -> grayscale taken from the R channel, opaque
 *   - alpha == 0 (undrawn) pixels -> black, opaque
 *
 * Extracted verbatim from ImageEditor.handleSave's inline mask conversion;
 * the pixel math is unchanged. Returns null if a 2D context could not be
 * obtained for either canvas, mirroring the original inline early-return
 * (the caller should skip saving in that case, same as before extraction).
 */
export function encodeMaskLayerToPng(canvas: HTMLCanvasElement): string | null {
  const maskCtx = canvas.getContext("2d");
  if (!maskCtx) return null;

  const maskImageData = maskCtx.getImageData(0, 0, canvas.width, canvas.height);
  const maskData = maskImageData.data;

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;
  const tempCtx = tempCanvas.getContext("2d");
  if (!tempCtx) return null;

  const tempImageData = tempCtx.createImageData(tempCanvas.width, tempCanvas.height);
  const tempData = tempImageData.data;

  // Convert: alpha channel -> grayscale, where white = area to inpaint
  for (let i = 0; i < maskData.length; i += 4) {
    const a = maskData[i + 3];
    if (a > 0) {
      // Drawn pixel: use RGB value as grayscale (should be white 255)
      const gray = maskData[i]; // R channel
      tempData[i] = gray;
      tempData[i + 1] = gray;
      tempData[i + 2] = gray;
      tempData[i + 3] = 255;
    } else {
      // Transparent pixel: black in mask (area NOT to inpaint)
      tempData[i] = 0;
      tempData[i + 1] = 0;
      tempData[i + 2] = 0;
      tempData[i + 3] = 255;
    }
  }

  tempCtx.putImageData(tempImageData, 0, 0);

  return tempCanvas.toDataURL("image/png");
}

// ---------------------------------------------------------------------------
// Device pixel ratio - deliberately unhandled
// ---------------------------------------------------------------------------

/**
 * ImageEditor sizes its canvases to the source image's native pixel
 * dimensions; CSS-side zoom/pan scales the *display* of that canvas without
 * resampling it. Because devicePixelRatio only changes how many physical
 * screen pixels back one CSS pixel (i.e. how the browser rasterizes an
 * already fixed-resolution canvas for display), it has no effect on the
 * pixels read back via getImageData/toDataURL. DPR handling is therefore
 * intentionally out of scope for this L0 layer: with the current sizing
 * strategy it could only ever change on-screen sharpness, never mask or
 * output correctness. Revisit only if canvas sizing itself changes to be
 * CSS-pixel-based (e.g. width = cssWidth * devicePixelRatio).
 */
