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
