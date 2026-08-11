/**
 * Fitting an arbitrary-aspect source image into a fixed-size output canvas
 * (L0 layer, dependency-free except for the DOM canvas/Image APIs).
 *
 * There is exactly ONE rule for "how does a source image map onto the
 * output canvas" in this project, and it is decided by the backend, not the
 * frontend: `backend/core/inference/outpaint_utils.py`'s
 * `center_crop_resize_frames` maps an input clip onto the output canvas by
 * scaling to cover (preserving aspect ratio, cropping the overflow) and
 * center-cropping -- never by stretching (ignoring aspect ratio). A video
 * mask PNG must additionally match the output canvas's pixel dimensions
 * exactly (`backend/core/inference/video_mask_timeline.py`'s `_decode_png`
 * rejects a mismatch), but "same dimensions" says nothing about how a
 * *source image of a different aspect ratio* should have been mapped onto
 * those dimensions in the first place -- that mapping still has to be
 * center-crop-cover, or a position the user drew relative to the on-screen
 * frame stops matching what the backend does with the same pixel offset.
 *
 * Before this module existed, InpaintPanel's video mask editor called two
 * different rules for the same "onto the output canvas" mapping depending on
 * whether the source was the underlying video frame (cover) or the user's
 * drawn mask (stretch/no-op resize) -- see the "video inpaint mask timeline"
 * audit. The stretch call never actually distorted anything in practice
 * (the mask is always drawn on a canvas already sized to the output canvas,
 * so source dims == target dims there and a stretch degenerates to an
 * identity copy), but having two rules for one concept was a latent
 * incorrectness the next phase (in-editor frame repositioning) would have
 * made easy to trip over. This module is the single named rule both call
 * sites use now.
 */

/**
 * Render `dataUrl` onto a `width` x `height` canvas using center-crop-cover:
 * scale so the source fully covers the target box, then crop off whatever
 * hangs outside (never letterboxing, never stretching). Matches the
 * backend's `center_crop_resize_frames`.
 *
 * Returns a PNG data URL sized exactly `width` x `height`.
 */
export function centerCropToCanvas(
  dataUrl: string,
  width: number,
  height: number,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      if (!image.naturalWidth || !image.naturalHeight) {
        reject(new Error("The source image has no usable dimensions."));
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d");
      if (!context) {
        reject(new Error("The browser could not create a canvas."));
        return;
      }
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      const scale = Math.max(width / image.naturalWidth, height / image.naturalHeight);
      const drawWidth = image.naturalWidth * scale;
      const drawHeight = image.naturalHeight * scale;
      context.drawImage(image, (width - drawWidth) / 2, (height - drawHeight) / 2, drawWidth, drawHeight);
      resolve(canvas.toDataURL("image/png"));
    };
    image.onerror = () => reject(new Error("The source image could not be decoded."));
    image.src = dataUrl;
  });
}

export interface DisplayRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * The on-screen rectangle, in CSS pixels relative to an `object-contain`
 * `<video>` element's OWN box (`containerWidth` x `containerHeight`), that
 * corresponds EXACTLY to the `outputWidth` x `outputHeight` canvas the
 * backend's center-crop-cover mapping (`centerCropToCanvas` above;
 * `center_crop_resize_frames` server-side) produces from the
 * `nativeWidth` x `nativeHeight` source.
 *
 * `object-contain` shows the FULL native frame, letterboxed. The backend
 * instead COVERS the output canvas and crops off whatever hangs outside, so
 * a spatial mask timeline (drawn in output-canvas pixel coordinates) must be
 * overlaid onto a sub-rectangle of the on-screen video that is usually
 * SMALLER than the full letterboxed frame -- the cropped-away edges of
 * `object-contain`'s display have no output-canvas counterpart at all, and
 * drawing the mask over them would misalign it relative to what the mask
 * actually affects once generated.
 *
 * Derivation: invert `centerCropToCanvas`'s own forward mapping (native ->
 * output canvas) to find which native pixel range survives the crop, then
 * map that native range through the video element's own `object-contain`
 * (native -> on-screen) mapping. Returns null for any non-positive input
 * dimension (nothing sensible to compute).
 */
export function computeCoverCropDisplayRect(
  containerWidth: number,
  containerHeight: number,
  nativeWidth: number,
  nativeHeight: number,
  outputWidth: number,
  outputHeight: number,
): DisplayRect | null {
  if (
    !(containerWidth > 0) || !(containerHeight > 0) ||
    !(nativeWidth > 0) || !(nativeHeight > 0) ||
    !(outputWidth > 0) || !(outputHeight > 0)
  ) {
    return null;
  }

  // Forward mapping (matches centerCropToCanvas's own math exactly): cover-
  // scale the native frame onto the output canvas, centered.
  const coverScale = Math.max(outputWidth / nativeWidth, outputHeight / nativeHeight);
  const drawWidth = nativeWidth * coverScale;
  const drawHeight = nativeHeight * coverScale;
  const canvasOffsetX = (outputWidth - drawWidth) / 2;
  const canvasOffsetY = (outputHeight - drawHeight) / 2;

  // Invert it: which native pixel range maps into [0, outputWidth] x
  // [0, outputHeight] (i.e. survives the crop) rather than being cut off.
  const nativeVisibleX0 = -canvasOffsetX / coverScale;
  const nativeVisibleX1 = (outputWidth - canvasOffsetX) / coverScale;
  const nativeVisibleY0 = -canvasOffsetY / coverScale;
  const nativeVisibleY1 = (outputHeight - canvasOffsetY) / coverScale;

  // The <video>'s own object-contain display: the full native frame,
  // letterboxed inside its container.
  const containScale = Math.min(containerWidth / nativeWidth, containerHeight / nativeHeight);
  const videoOffsetX = (containerWidth - nativeWidth * containScale) / 2;
  const videoOffsetY = (containerHeight - nativeHeight * containScale) / 2;

  // Map the visible-native rectangle into on-screen coordinates.
  return {
    x: videoOffsetX + nativeVisibleX0 * containScale,
    y: videoOffsetY + nativeVisibleY0 * containScale,
    width: (nativeVisibleX1 - nativeVisibleX0) * containScale,
    height: (nativeVisibleY1 - nativeVisibleY0) * containScale,
  };
}
