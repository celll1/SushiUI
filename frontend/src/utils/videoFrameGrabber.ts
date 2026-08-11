// Off-DOM frame-grab utility shared by the inpaint/outpaint timelines' hover
// and drag previews. One hidden <video> + <canvas> pair is kept per source
// URL (not per call), and every grab for that source shares them, so a drag
// producing dozens of requests per second does not spawn dozens of decoders.
//
// Seeks against a single <video> element are inherently serial (setting
// `.currentTime` again before the previous `seeked` fires just abandons the
// previous seek), so requests for the same source run through an explicit
// FIFO chain rather than firing concurrently. A request that is no longer
// the latest one asked for by the time its turn comes up is skipped -- it
// resolves from whatever is cached for a nearby time instead of performing
// its own now-stale seek, and reports `exact: false` so a caller pairing the
// image with a label can tell the two do not describe the same instant.
// Both the metadata load and the seek race a timeout with full listener
// teardown, since some browsers (Firefox) never fire `seeked` when the
// target time already equals the current time, and a decode can simply
// stall -- so a call always settles instead of wedging the source's FIFO
// chain for the rest of the page's life. An `ensureVideo` failure is cached
// per source rather than retried on every subsequent grab.

interface GrabberState {
  video: HTMLVideoElement | null;
  canvas: HTMLCanvasElement | null;
  cache: Map<string, string>;
  cacheOrder: string[];
  chain: Promise<void>;
  latestKey: string;
  /** Set once `ensureVideo` fails for this source; further grabs return null immediately instead of creating a fresh detached `<video>` per call. */
  failed: boolean;
}

export interface FrameGrabResult {
  /** The quantized time key `dataUrl` actually corresponds to. */
  key: string;
  dataUrl: string;
  /** False when this request was superseded before its turn in the FIFO chain and `dataUrl`/`key` were substituted from a nearby cached frame instead of the exact requested time. */
  exact: boolean;
}

const CACHE_LIMIT = 96;
// Quantize requested times to this granularity so a slow drag re-uses the
// same cached thumbnail instead of missing on every sub-pixel move.
const QUANTIZE_SEC = 1 / 30;
const METADATA_TIMEOUT_MS = 8000;
const SEEK_TIMEOUT_MS = 4000;
// Treat "already at the requested time" as a hit rather than awaiting
// `seeked`, since some browsers do not fire it when currentTime is unchanged.
const SEEK_EPSILON_SEC = 1 / 120;

const states = new Map<string, GrabberState>();

function quantize(seconds: number): string {
  return (Math.round(Math.max(0, seconds) / QUANTIZE_SEC) * QUANTIZE_SEC).toFixed(3);
}

function getState(src: string): GrabberState {
  let state = states.get(src);
  if (!state) {
    state = { video: null, canvas: null, cache: new Map(), cacheOrder: [], chain: Promise.resolve(), latestKey: "", failed: false };
    states.set(src, state);
  }
  return state;
}

function remember(state: GrabberState, key: string, dataUrl: string) {
  if (!state.cache.has(key)) state.cacheOrder.push(key);
  state.cache.set(key, dataUrl);
  while (state.cacheOrder.length > CACHE_LIMIT) {
    const evict = state.cacheOrder.shift();
    if (evict) state.cache.delete(evict);
  }
}

async function ensureVideo(state: GrabberState, src: string): Promise<HTMLVideoElement | null> {
  if (state.video) return state.video;
  if (state.failed) return null;
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = src;
  try {
    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const finish = (ok: boolean) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        video.removeEventListener("loadedmetadata", onLoaded);
        video.removeEventListener("error", onError);
        if (ok) resolve();
        else reject(new Error("video metadata load failed or timed out"));
      };
      const onLoaded = () => finish(true);
      const onError = () => finish(false);
      const timer = window.setTimeout(() => finish(false), METADATA_TIMEOUT_MS);
      video.addEventListener("loadedmetadata", onLoaded);
      video.addEventListener("error", onError);
    });
  } catch (error) {
    state.failed = true;
    console.error("[videoFrameGrabber] Failed to open source for frame grabbing (will not retry):", error);
    return null;
  }
  state.video = video;
  return video;
}

async function captureAt(state: GrabberState, video: HTMLVideoElement, time: number, maxWidth: number): Promise<string | null> {
  try {
    const clampedTime = Number.isFinite(video.duration) ? Math.min(video.duration, Math.max(0, time)) : Math.max(0, time);
    const alreadyThere = !video.seeking && Math.abs(video.currentTime - clampedTime) < SEEK_EPSILON_SEC;
    if (!alreadyThere) {
      await new Promise<void>((resolve, reject) => {
        let settled = false;
        const finish = (ok: boolean) => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          video.removeEventListener("seeked", onSeeked);
          video.removeEventListener("error", onError);
          if (ok) resolve();
          else reject(new Error("video seek failed or timed out"));
        };
        const onSeeked = () => finish(true);
        const onError = () => finish(false);
        const timer = window.setTimeout(() => finish(false), SEEK_TIMEOUT_MS);
        video.addEventListener("seeked", onSeeked);
        video.addEventListener("error", onError);
        video.currentTime = clampedTime;
      });
    }
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;
    const scale = Math.min(1, maxWidth / vw);
    const w = Math.max(1, Math.round(vw * scale));
    const h = Math.max(1, Math.round(vh * scale));
    if (!state.canvas) state.canvas = document.createElement("canvas");
    const canvas = state.canvas;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, w, h);
    return canvas.toDataURL("image/jpeg", 0.7);
  } catch (error) {
    console.error("[videoFrameGrabber] Frame capture failed:", error);
    return null;
  }
}

/**
 * Grabs a small thumbnail of `src` (a video object/file URL) at `time`
 * seconds. Cached per (src, time quantized to 1/30s); concurrent/rapid calls
 * for the same source share one decoder and are serialized so none hang, but
 * a call superseded by a newer one before its turn comes up resolves from
 * whatever is cached instead of performing its own (by-then-stale) seek --
 * check `exact` on the result to tell the two cases apart. Returns null if
 * the source cannot be decoded/read, or nothing is cached to fall back to.
 */
export async function grabVideoFrame(
  src: string,
  time: number,
  options?: { maxWidth?: number }
): Promise<FrameGrabResult | null> {
  if (!src || !Number.isFinite(time)) return null;
  const state = getState(src);
  const key = quantize(time);
  state.latestKey = key;

  const cached = state.cache.get(key);
  if (cached) return { key, dataUrl: cached, exact: true };

  const maxWidth = options?.maxWidth ?? 160;
  const run = state.chain.then(async (): Promise<FrameGrabResult | null> => {
    // Re-check the cache: a queued-ahead call for the same key may have
    // already resolved it while this one waited its turn.
    const already = state.cache.get(key);
    if (already) return { key, dataUrl: already, exact: true };
    // Superseded by a newer request before this one's turn -- do not spend a
    // real seek on a position the user has already scrubbed past. Fall back
    // to whatever the latest resolved frame is, if any.
    if (state.latestKey !== key) {
      const fallbackKey = [...state.cacheOrder].reverse().find((k) => state.cache.has(k));
      const fallbackUrl = fallbackKey ? state.cache.get(fallbackKey) : undefined;
      return fallbackKey && fallbackUrl ? { key: fallbackKey, dataUrl: fallbackUrl, exact: false } : null;
    }
    const video = await ensureVideo(state, src);
    if (!video) return null;
    const dataUrl = await captureAt(state, video, time, maxWidth);
    if (!dataUrl) return null;
    remember(state, key, dataUrl);
    return { key, dataUrl, exact: true };
  });

  // Keep the chain alive regardless of this call's outcome, and never let a
  // rejection here propagate into an unrelated later call.
  state.chain = run.then(() => undefined, () => undefined);
  return run;
}

/** Releases the off-DOM <video>/<canvas> and cache kept for `src`. Call on cleanup (e.g. when the input clip is cleared/replaced). */
export function releaseVideoFrameGrabber(src: string | null | undefined): void {
  if (!src) return;
  const state = states.get(src);
  if (!state) return;
  if (state.video) {
    state.video.src = "";
    state.video.load();
  }
  states.delete(src);
}
