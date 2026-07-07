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
  /**
   * Color-flatten (chroma smoothing / 色ムラ除去) strength, 0-100; 0 = neutral
   * (no change). Unlike brightness/saturation this is NOT a CSS filter: it is a
   * pixel pass (see flattenChroma) applied to the image data itself, with
   * brightness/saturation layered on top as a CSS filter.
   */
  flatten: number;
}

export const NEUTRAL_POST_EDIT: PostEditState = { brightness: 100, saturation: 100, flatten: 0 };

export function isNeutral(state: PostEditState): boolean {
  return state.brightness === 100 && state.saturation === 100 && state.flatten === 0;
}

/**
 * Single source of truth for the brightness/saturation filter string. Used both
 * for the CSS preview (style.filter) and for the canvas bake (ctx.filter) so
 * preview and downloaded pixels are always consistent. Order: brightness() then
 * saturate(). Returns undefined when brightness AND saturation are neutral so
 * the preview DOM stays identical to today. NOTE: flatten is intentionally not
 * encoded here (it is a pixel pass, not CSS) - a pure-flatten edit therefore
 * still yields undefined and never touches the ctx.filter path.
 */
export function buildFilterString(state: PostEditState): string | undefined {
  if (state.brightness === 100 && state.saturation === 100) return undefined;
  return `brightness(${state.brightness}%) saturate(${state.saturation}%)`;
}

// ---------------------------------------------------------------------------
// Color flatten (chroma smoothing) - color-guided guided filter on YCoCg chroma
// ---------------------------------------------------------------------------
//
// Pure function operating in-place on an ImageData. Structured so it could later
// be moved to a Web Worker unchanged (it only touches the passed ImageData and
// scratch typed arrays; no DOM). Ported from the reference prototype
// (cand_guided_color): a guided filter whose guide is the image's own RGB and
// whose filtered signals are the two chroma channels (Co, Cg). The luma (Y)
// channel is left untouched, so edges and detail carried by luminance survive
// while low-frequency chroma mottling is smoothed away.
//
// Domain note: matches the prototype exactly - it operates on the sRGB-coded
// [0,1] values directly (no linearization / gamma conversion), because the
// prototype did not linearize either.

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Separable box filter with a clamped window (running-sum via prefix sums),
 * O(W*H) per axis. For each output index the average is taken over the valid
 * clamped window [i-r, i+r] and divided by the actual (clamped) sample count -
 * identical to the prototype's boxfilter (normalize by window count, not 2r+1).
 */
function boxFilter(src: Float32Array, W: number, H: number, r: number): Float32Array {
  const tmp = new Float32Array(W * H);
  const dst = new Float32Array(W * H);

  // Horizontal pass: src -> tmp
  for (let y = 0; y < H; y++) {
    const row = y * W;
    // prefix sum along the row (cs[x] = sum of src[0..x-1]); length W+1
    let acc = 0;
    // Use a small rolling technique without allocating a prefix array per row:
    // compute cumulative into a reused buffer.
    // (Allocate once outside loop would be marginally faster; kept simple/clear.)
    const cs = new Float32Array(W + 1);
    for (let x = 0; x < W; x++) {
      acc += src[row + x];
      cs[x + 1] = acc;
    }
    for (let x = 0; x < W; x++) {
      const lo = x - r > 0 ? x - r : 0;
      const hi = x + r + 1 < W ? x + r + 1 : W;
      tmp[row + x] = (cs[hi] - cs[lo]) / (hi - lo);
    }
  }

  // Vertical pass: tmp -> dst
  const cs = new Float32Array(H + 1);
  for (let x = 0; x < W; x++) {
    let acc = 0;
    for (let y = 0; y < H; y++) {
      acc += tmp[y * W + x];
      cs[y + 1] = acc;
    }
    for (let y = 0; y < H; y++) {
      const lo = y - r > 0 ? y - r : 0;
      const hi = y + r + 1 < H ? y + r + 1 : H;
      dst[y * W + x] = (cs[hi] - cs[lo]) / (hi - lo);
    }
  }

  return dst;
}

export function flattenChroma(imageData: ImageData, strength: number): void {
  const f = strength / 100;
  if (f <= 0) return; // hard no-op

  const W = imageData.width;
  const H = imageData.height;
  const N = W * H;
  const data = imageData.data; // RGBA Uint8ClampedArray

  const longSide = Math.max(W, H);
  let radius = Math.round(lerp(12, 40, f) * (longSide / 1024));
  if (radius < 4) radius = 4;
  const eps = lerp(1.5e-3, 8e-3, f);
  const blend = lerp(0.4, 1.0, f);

  // Decode RGB (linear-in-code [0,1]) and chroma planes.
  const R = new Float32Array(N);
  const G = new Float32Array(N);
  const B = new Float32Array(N);
  const Y = new Float32Array(N);
  const Co = new Float32Array(N);
  const Cg = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    R[i] = r;
    G[i] = g;
    B[i] = b;
    Y[i] = 0.25 * r + 0.5 * g + 0.25 * b;
    Co[i] = 0.5 * r - 0.5 * b;
    Cg[i] = -0.25 * r + 0.5 * g - 0.25 * b;
  }

  // Guide (RGB) means.
  const mR = boxFilter(R, W, H, radius);
  const mG = boxFilter(G, W, H, radius);
  const mB = boxFilter(B, W, H, radius);

  // Guide covariance entries (Sigma + eps*I), symmetric 3x3 per pixel.
  const RR = new Float32Array(N);
  const RG = new Float32Array(N);
  const RB = new Float32Array(N);
  const GG = new Float32Array(N);
  const GB = new Float32Array(N);
  const BB = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    RR[i] = R[i] * R[i];
    RG[i] = R[i] * G[i];
    RB[i] = R[i] * B[i];
    GG[i] = G[i] * G[i];
    GB[i] = G[i] * B[i];
    BB[i] = B[i] * B[i];
  }
  const mRR = boxFilter(RR, W, H, radius);
  const mRG = boxFilter(RG, W, H, radius);
  const mRB = boxFilter(RB, W, H, radius);
  const mGG = boxFilter(GG, W, H, radius);
  const mGB = boxFilter(GB, W, H, radius);
  const mBB = boxFilter(BB, W, H, radius);

  // Per-pixel: closed-form inverse of the symmetric 3x3 (Sigma + eps*I).
  // Store the 6 unique inverse entries per pixel so both chroma channels reuse them.
  const i00 = new Float32Array(N);
  const i01 = new Float32Array(N);
  const i02 = new Float32Array(N);
  const i11 = new Float32Array(N);
  const i12 = new Float32Array(N);
  const i22 = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const a = mRR[i] - mR[i] * mR[i] + eps; // vrr
    const b = mRG[i] - mR[i] * mG[i]; // vrg
    const c = mRB[i] - mR[i] * mB[i]; // vrb
    const d = mGG[i] - mG[i] * mG[i] + eps; // vgg
    const e = mGB[i] - mG[i] * mB[i]; // vgb
    const g = mBB[i] - mB[i] * mB[i] + eps; // vbb

    const co00 = d * g - e * e;
    const co01 = c * e - b * g;
    const co02 = b * e - c * d;
    const co11 = a * g - c * c;
    const co12 = b * c - a * e;
    const co22 = a * d - b * b;
    let det = a * co00 + b * co01 + c * co02;
    if (det > -1e-12 && det < 1e-12) {
      // Singular guide window -> a=0 (q collapses to box(mean(p))). Set inverse
      // to zero so a-vector is zero regardless of covIp.
      i00[i] = 0; i01[i] = 0; i02[i] = 0; i11[i] = 0; i12[i] = 0; i22[i] = 0;
      continue;
    }
    const invDet = 1 / det;
    i00[i] = co00 * invDet;
    i01[i] = co01 * invDet;
    i02[i] = co02 * invDet;
    i11[i] = co11 * invDet;
    i12[i] = co12 * invDet;
    i22[i] = co22 * invDet;
  }

  // Process one chroma channel through the guided filter, blend in place.
  const processChannel = (p: Float32Array) => {
    const mp = boxFilter(p, W, H, radius);
    const Rp = new Float32Array(N);
    const Gp = new Float32Array(N);
    const Bp = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      Rp[i] = R[i] * p[i];
      Gp[i] = G[i] * p[i];
      Bp[i] = B[i] * p[i];
    }
    const mRp = boxFilter(Rp, W, H, radius);
    const mGp = boxFilter(Gp, W, H, radius);
    const mBp = boxFilter(Bp, W, H, radius);

    const aR = new Float32Array(N);
    const aG = new Float32Array(N);
    const aB = new Float32Array(N);
    const bb = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const covR = mRp[i] - mR[i] * mp[i];
      const covG = mGp[i] - mG[i] * mp[i];
      const covB = mBp[i] - mB[i] * mp[i];
      const ar = i00[i] * covR + i01[i] * covG + i02[i] * covB;
      const ag = i01[i] * covR + i11[i] * covG + i12[i] * covB;
      const ab = i02[i] * covR + i12[i] * covG + i22[i] * covB;
      aR[i] = ar;
      aG[i] = ag;
      aB[i] = ab;
      bb[i] = mp[i] - (ar * mR[i] + ag * mG[i] + ab * mB[i]);
    }
    const maR = boxFilter(aR, W, H, radius);
    const maG = boxFilter(aG, W, H, radius);
    const maB = boxFilter(aB, W, H, radius);
    const mb = boxFilter(bb, W, H, radius);

    for (let i = 0; i < N; i++) {
      const q = maR[i] * R[i] + maG[i] * G[i] + maB[i] * B[i] + mb[i];
      p[i] = p[i] * (1 - blend) + q * blend;
    }
  };

  processChannel(Co);
  processChannel(Cg);

  // Reconstruct RGB from Y + smoothed chroma, write back (clamp handled by Uint8ClampedArray).
  for (let i = 0; i < N; i++) {
    const y = Y[i];
    const co = Co[i];
    const cg = Cg[i];
    const r = y + co - cg;
    const g = y + cg;
    const b = y - co - cg;
    data[i * 4] = r * 255;
    data[i * 4 + 1] = g * 255;
    data[i * 4 + 2] = b * 255;
    // alpha (data[i*4+3]) untouched
  }
}

/**
 * Bakes the post-edit adjustments into a new PNG blob.
 * NOTE: re-encodes the image; embedded PNG metadata is lost (see file header).
 * Callers should skip this and use the original blob when isNeutral(state).
 */
export async function applyPostEdit(blob: Blob, state: PostEditState): Promise<Blob> {
  if (isNeutral(state)) return blob; // neutral safety net: return original untouched

  const filter = buildFilterString(state); // brightness/saturation only (undefined when b/s neutral)
  const flatten = state.flatten;

  const { bitmap, width, height, cleanup } = await decodeBlob(blob);
  try {
    // Stage 1 (optional): color-flatten pixel pass at FULL resolution. Uses
    // getImageData/putImageData (universally supported), so a pure-flatten edit
    // never touches the ctx.filter honesty probe below.
    let source: CanvasImageSource = bitmap as CanvasImageSource;
    if (flatten > 0) {
      const flatCanvas = document.createElement("canvas");
      flatCanvas.width = width;
      flatCanvas.height = height;
      const fctx = flatCanvas.getContext("2d");
      if (!fctx) throw new Error("Failed to get 2D canvas context");
      fctx.drawImage(bitmap as CanvasImageSource, 0, 0, width, height);
      const imgData = fctx.getImageData(0, 0, width, height);
      flattenChroma(imgData, flatten);
      fctx.putImageData(imgData, 0, 0);
      source = flatCanvas;

      // Flatten-only: no brightness/saturation to bake, output directly.
      if (!filter) {
        const outBlob = await new Promise<Blob | null>((resolve) =>
          flatCanvas.toBlob(resolve, "image/png")
        );
        if (!outBlob) throw new Error("canvas.toBlob returned null");
        return outBlob;
      }
    }

    // Stage 2: brightness/saturation via ctx.filter (only reached when b/s are
    // non-neutral). Operates on `source` (the flattened canvas if flatten>0,
    // else the original bitmap).
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
    ctx.drawImage(source, 0, 0, width, height);

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
