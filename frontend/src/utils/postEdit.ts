// Client-side post-edit adjustments for generated images.
//
// These adjustments (brightness + saturation) are applied purely on the client:
//   - As a live CSS filter on the preview <img> (see buildFilterString).
//   - Baked into the pixels when the user downloads an edited image (see applyPostEdit).
//
// Output-folder files on disk are NEVER modified by this feature. The baking path
// re-encodes a fresh PNG from a canvas, which means any embedded generation metadata
// (PNG text chunks written by the backend) is LOST in the edited download. This is
// acceptable: the unedited download path still goes through /api/download with the
// include_metadata flag and preserves metadata bit-for-bit. Callers MUST download the
// original blob unchanged when isNeutral(state) is true.

export interface PostEditState {
  /** Brightness in percent; 100 = neutral (no change). */
  brightness: number;
  /** Saturation in percent; 100 = neutral (no change). */
  saturation: number;
}

export const NEUTRAL_POST_EDIT: PostEditState = { brightness: 100, saturation: 100 };

export function isNeutral(state: PostEditState): boolean {
  return state.brightness === 100 && state.saturation === 100;
}

/**
 * Single source of truth for the filter string. Used both for the CSS preview
 * (style.filter) and for the canvas bake (ctx.filter) so preview and downloaded
 * pixels are always consistent. Order: brightness() then saturate().
 * Returns undefined when neutral so the preview DOM stays identical to today.
 */
export function buildFilterString(state: PostEditState): string | undefined {
  if (isNeutral(state)) return undefined;
  return `brightness(${state.brightness}%) saturate(${state.saturation}%)`;
}

/**
 * Bakes the post-edit adjustments into a new PNG blob.
 * NOTE: re-encodes the image; embedded PNG metadata is lost (see file header).
 * Callers should skip this and use the original blob when isNeutral(state).
 */
export async function applyPostEdit(blob: Blob, state: PostEditState): Promise<Blob> {
  const filter = buildFilterString(state);
  if (!filter) return blob; // neutral safety net: return original untouched

  const { bitmap, width, height, cleanup } = await decodeBlob(blob);
  try {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Failed to get 2D canvas context");

    // Honesty guard: browsers without canvas ctx.filter support (Safari < 18)
    // silently IGNORE the assignment, which would bake nothing and hand the
    // user an unedited file under an "_edited" name. Probe support and fail
    // loudly instead - callers catch and surface an alert.
    ctx.filter = "brightness(50%)";
    if (ctx.filter === "none" || ctx.filter === "") {
      throw new Error(
        "This browser does not support canvas filters; cannot bake the post-edit into the download."
      );
    }

    // Must match buildFilterString ordering exactly.
    ctx.filter = filter;
    ctx.drawImage(bitmap as CanvasImageSource, 0, 0, width, height);

    const outBlob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/png")
    );
    if (!outBlob) throw new Error("canvas.toBlob returned null");
    return outBlob;
  } finally {
    cleanup();
  }
}

interface DecodedImage {
  bitmap: ImageBitmap | HTMLImageElement;
  width: number;
  height: number;
  cleanup: () => void;
}

async function decodeBlob(blob: Blob): Promise<DecodedImage> {
  // Prefer createImageBitmap (fast, off-DOM). Fall back to Image + object URL.
  if (typeof createImageBitmap === "function") {
    const bitmap = await createImageBitmap(blob);
    return {
      bitmap,
      width: bitmap.width,
      height: bitmap.height,
      cleanup: () => bitmap.close(),
    };
  }

  const objectUrl = URL.createObjectURL(blob);
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("Failed to decode image blob"));
    img.src = objectUrl;
  });
  return {
    bitmap: img,
    width: img.naturalWidth,
    height: img.naturalHeight,
    cleanup: () => URL.revokeObjectURL(objectUrl),
  };
}

/**
 * Inserts "_edited" before the file extension.
 * "txt2img_001.png" -> "txt2img_001_edited.png"
 */
export function editedFilename(filename: string): string {
  const dot = filename.lastIndexOf(".");
  if (dot <= 0) return `${filename}_edited`;
  return `${filename.slice(0, dot)}_edited${filename.slice(dot)}`;
}
