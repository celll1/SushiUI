/**
 * Shared persistence for a generation panel's *result preview*.
 *
 * Every generation panel already persists its image result as a bare URL (or
 * data URL) under a `<panel>_preview` localStorage key, so the preview survives
 * a tab switch or a browser restart.  Video and audio results had no equivalent
 * and died with the component state.
 *
 * This module owns all three modalities so the "which one is showing?" rule
 * lives in one place:
 *
 * - The image preview keeps its existing key and existing plain-string format,
 *   so previews written by older builds still load.  It is stored canonically,
 *   without the `?t=` cache-buster panels add when they restore it
 *   (`stripCacheBuster` / `withCacheBuster` below own that pairing, so the stamp
 *   is replaced on each restore instead of accumulating).
 * - The video preview lives under `<panel>_preview_video` and is stored as JSON
 *   ({ url, info, seed }) because a video carries frame/fps/duration metadata
 *   next to its URL.
 * - The audio preview lives under `<panel>_preview_audio`, same JSON shape,
 *   with a duration/sample-rate info line instead.
 * - In every case only the URL is stored -- never the bytes -- so an entry is a
 *   few hundred bytes regardless of clip length.
 * - The three keys are **mutually exclusive**: saving any one of them removes
 *   the other two (`saveExclusive` below is the single place that rule lives).
 *   Whichever result was produced last is the only one in storage, so a restore
 *   can never show a stale image next to a newer clip, and there is no
 *   precedence order for callers to get wrong.
 *
 * Nothing here validates that the referenced file still exists; `outputExists`
 * is provided for callers that want to verify a restored URL once the backend
 * is reachable (`outputs/` can be cleared, or a run deleted from the gallery).
 */

export interface VideoPreviewInfo {
  num_frames?: number;
  fps?: number;
  duration?: number;
}

export interface AudioPreviewInfo {
  duration?: number;
  sample_rate?: number;
}

export interface StoredMediaPreview<TInfo> {
  /** Backend URL of the result, e.g. "/outputs/txt2vid_20260807_070228_0.mp4". */
  url: string;
  info: TInfo | null;
  /** Seed of the run, when the panel exposes a "reuse seed" button. */
  seed?: number | null;
}

export type StoredVideoPreview = StoredMediaPreview<VideoPreviewInfo>;
export type StoredAudioPreview = StoredMediaPreview<AudioPreviewInfo>;

export interface PreviewStorageKeys {
  image: string;
  video: string;
  audio: string;
}

/** The three modalities a panel's result preview can hold. */
export type PreviewMediaKind = keyof PreviewStorageKeys;

/** Derive the trio of keys for a panel from its existing image preview key. */
export function previewStorageKeys(imageKey: string): PreviewStorageKeys {
  return {
    image: imageKey,
    video: `${imageKey}_video`,
    audio: `${imageKey}_audio`,
  };
}

/**
 * The mutual-exclusion rule, in the one place it lives: write `value` under the
 * key for `kind` and remove the keys for the other two modalities.  Every save
 * helper below funnels through this, so adding a modality means extending
 * `PreviewStorageKeys` and nothing else.
 */
function saveExclusive(keys: PreviewStorageKeys, kind: PreviewMediaKind, value: string): void {
  try {
    localStorage.setItem(keys[kind], value);
    (Object.keys(keys) as PreviewMediaKind[])
      .filter((other) => other !== kind)
      .forEach((other) => localStorage.removeItem(keys[other]));
  } catch (error) {
    console.warn(`[previewStorage] Failed to store ${kind} preview:`, error);
  }
}

/** Read and validate a JSON-encoded media preview ({ url, info, seed }). */
function loadMediaPreview<TInfo>(
  key: string,
  label: string,
): StoredMediaPreview<TInfo> | null {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed.url !== "string" || !parsed.url) {
      localStorage.removeItem(key);
      return null;
    }
    return {
      url: parsed.url,
      info: parsed.info ?? null,
      seed: parsed.seed ?? null,
    };
  } catch (error) {
    console.warn(`[previewStorage] Failed to read stored ${label} preview:`, error);
    try {
      localStorage.removeItem(key);
    } catch {
      /* storage unavailable; nothing to clean up */
    }
    return null;
  }
}

const VIDEO_EXTENSIONS = [".mp4", ".webm", ".mov", ".mkv"];
const AUDIO_EXTENSIONS = [".flac", ".wav", ".mp3", ".ogg", ".m4a"];

function extensionOf(url: string): string {
  // Strip any cache-busting query/fragment before looking at the extension.
  const path = url.split(/[?#]/)[0].toLowerCase();
  const dot = path.lastIndexOf(".");
  return dot === -1 ? "" : path.slice(dot);
}

export function isVideoUrl(url: string): boolean {
  return VIDEO_EXTENSIONS.includes(extensionOf(url));
}

export function isAudioUrl(url: string): boolean {
  return AUDIO_EXTENSIONS.includes(extensionOf(url));
}

/**
 * Poster frame the backend writes next to every generated clip (same base name,
 * `.png`).  Used as the `poster` attribute so a thumbnail can render without
 * decoding the video; a missing poster degrades to the browser default.
 *
 * Audio deliberately has no counterpart here.  The backend does write a
 * same-base-name `.png` next to every generated `.flac`, but it is a waveform
 * plot rather than a frame grab and, more to the point, `<audio>` has no
 * `poster` attribute for it to feed -- the PNG exists to seed the gallery
 * thumbnail (see `create_thumbnail` in the audio routes), not the player.  So
 * audio tiles render as a compact `<audio controls>` with no poster lookup;
 * this function returns "" for anything that is not a video URL, so calling it
 * on an audio URL is harmless but pointless.
 */
export function posterUrlForVideo(url: string): string {
  const [path, rest] = splitQuery(url);
  if (!isVideoUrl(path)) return "";
  const dot = path.lastIndexOf(".");
  return `${path.slice(0, dot)}.png${rest}`;
}

function splitQuery(url: string): [string, string] {
  const index = url.search(/[?#]/);
  return index === -1 ? [url, ""] : [url.slice(0, index), url.slice(index)];
}

export function loadVideoPreview(keys: PreviewStorageKeys): StoredVideoPreview | null {
  return loadMediaPreview<VideoPreviewInfo>(keys.video, "video");
}

export function loadAudioPreview(keys: PreviewStorageKeys): StoredAudioPreview | null {
  return loadMediaPreview<AudioPreviewInfo>(keys.audio, "audio");
}

/** Persist a video result and drop any older image/audio preview for the panel. */
export function saveVideoPreview(keys: PreviewStorageKeys, preview: StoredVideoPreview): void {
  saveExclusive(keys, "video", JSON.stringify(preview));
}

/** Persist an audio result and drop any older image/video preview for the panel. */
export function saveAudioPreview(keys: PreviewStorageKeys, preview: StoredAudioPreview): void {
  saveExclusive(keys, "audio", JSON.stringify(preview));
}

/**
 * Persist an image result and drop any older video/audio preview for the panel.
 *
 * The URL is stored *without* its cache-busting `?t=` stamp: panels re-stamp on
 * restore and then write the stamped value straight back through this effect, so
 * storing the stamp verbatim made the query string grow by one `?t=` per reload
 * ("a.png?t=1?t=2?t=3"). Storing the canonical path keeps the stored value the
 * same shape whether it was written by a fresh generation or by a restore.
 */
export function saveImagePreview(keys: PreviewStorageKeys, url: string): void {
  saveExclusive(keys, "image", stripCacheBuster(url));
}

/**
 * Should a failed `<img>` load discard the panel's image preview?
 *
 * Only when the element was actually showing the backend result: a panel may
 * render a derived `blob:` URL instead (the post-edit colour-flatten preview),
 * and that failing is a client-side problem which must not throw away a result
 * that is still on disk. Likewise a `data:` preview, which cannot 404.
 */
export function shouldDiscardImagePreview(
  displayedSrc: string | null | undefined,
  resultUrl: string | null | undefined,
): boolean {
  if (!resultUrl || !resultUrl.startsWith("/outputs/")) return false;
  return !displayedSrc || displayedSrc === resultUrl;
}

/**
 * The `<img>` `onError` backstop, for a result file that disappears while the
 * panel is open.  Resolves true only when the preview is worth discarding
 * (`shouldDiscardImagePreview`) *and* a HEAD confirms the file is really gone.
 *
 * The confirmation matters because an `<img>` error event carries no status: a
 * hot reload or a backend blip fails the load exactly like a 404 would, and
 * throwing the stored preview away on that would lose a result that is still on
 * disk.  `outputExists` already treats an unreachable backend as "keep it", so
 * routing through it applies the same rule the restore path uses instead of
 * inventing a second one.
 */
export async function imagePreviewGone(
  displayedSrc: string | null | undefined,
  resultUrl: string | null | undefined,
): Promise<boolean> {
  if (!shouldDiscardImagePreview(displayedSrc, resultUrl)) return false;
  return !(await outputExists(resultUrl as string));
}

/** Forget the panel's stored image preview (the file it pointed at is gone). */
export function clearImagePreview(keys: PreviewStorageKeys): void {
  clearPreview(keys.image, "image");
}

/**
 * Remove any `t=<ms>` cache-buster from a backend path, tolerating the historical
 * double-stamped form ("a.png?t=1?t=2") and preserving any other query parameter.
 *
 * Only applied to root-relative backend paths: a `data:` or `blob:` preview is
 * returned untouched, so nothing here can corrupt a non-URL preview value.
 */
export function stripCacheBuster(url: string): string {
  if (!url.startsWith("/")) return url;
  const hashAt = url.indexOf("#");
  const hash = hashAt === -1 ? "" : url.slice(hashAt);
  const body = hashAt === -1 ? url : url.slice(0, hashAt);
  const queryAt = body.indexOf("?");
  if (queryAt === -1) return url;
  const path = body.slice(0, queryAt);
  const kept = body
    .slice(queryAt + 1)
    .split(/[?&]/)
    .filter((part) => part && !/^t=\d+$/.test(part));
  return kept.length ? `${path}?${kept.join("&")}${hash}` : `${path}${hash}`;
}

/**
 * Re-stamp a backend path so a restored preview is refetched rather than served
 * from the browser cache (the same filename can be rewritten by a later run).
 * Any previous stamp is replaced, never appended.
 */
export function withCacheBuster(url: string): string {
  const clean = stripCacheBuster(url);
  if (!clean.startsWith("/")) return clean;
  return `${clean}${clean.includes("?") ? "&" : "?"}t=${Date.now()}`;
}

function clearPreview(key: string, label: string): void {
  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.warn(`[previewStorage] Failed to clear ${label} preview:`, error);
  }
}

export function clearVideoPreview(keys: PreviewStorageKeys): void {
  clearPreview(keys.video, "video");
}

export function clearAudioPreview(keys: PreviewStorageKeys): void {
  clearPreview(keys.audio, "audio");
}

/**
 * HEAD-check a backend URL.  Returns true for anything that is not a backend
 * path (data URLs, blob URLs) so callers only ever discard results they can
 * actually prove are gone; a network failure is likewise treated as "unknown"
 * and reported as missing only for real 4xx/5xx responses.
 */
export async function outputExists(url: string): Promise<boolean> {
  const [path] = splitQuery(url);
  if (!path.startsWith("/outputs/")) return true;
  try {
    const response = await fetch(path, { method: "HEAD", cache: "no-store" });
    return response.ok;
  } catch (error) {
    // Backend unreachable / transient network error: keep the preview rather
    // than throwing away a pointer to a file that is probably still there.
    console.warn("[previewStorage] Could not verify preview URL:", path, error);
    return true;
  }
}
