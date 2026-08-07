/**
 * Shared persistence for a generation panel's *result preview*.
 *
 * Every generation panel already persists its image result as a bare URL (or
 * data URL) under a `<panel>_preview` localStorage key, so the preview survives
 * a tab switch or a browser restart.  Video results had no equivalent and died
 * with the component state.
 *
 * This module owns both halves so the "which one is showing?" rule lives in one
 * place:
 *
 * - The image preview keeps its existing key and existing plain-string format,
 *   so previews written by older builds still load.
 * - The video preview lives under `<panel>_preview_video` and is stored as JSON
 *   ({ url, info, seed }) because a video carries frame/fps/duration metadata
 *   next to its URL.  Only the URL is stored -- never the bytes -- so this is a
 *   few hundred bytes regardless of clip length.
 * - The two keys are **mutually exclusive**: saving one removes the other.
 *   Whichever result was produced last is the only one in storage, so a restore
 *   can never show a stale image next to a newer video (or vice versa).
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

export interface StoredVideoPreview {
  /** Backend URL of the clip, e.g. "/outputs/txt2vid_20260807_070228_0.mp4". */
  url: string;
  info: VideoPreviewInfo | null;
  /** Seed of the run, when the panel exposes a "reuse seed" button. */
  seed?: number | null;
}

export interface PreviewStorageKeys {
  image: string;
  video: string;
}

/** Derive the pair of keys for a panel from its existing image preview key. */
export function previewStorageKeys(imageKey: string): PreviewStorageKeys {
  return { image: imageKey, video: `${imageKey}_video` };
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
  try {
    const raw = localStorage.getItem(keys.video);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed.url !== "string" || !parsed.url) {
      localStorage.removeItem(keys.video);
      return null;
    }
    return {
      url: parsed.url,
      info: parsed.info ?? null,
      seed: parsed.seed ?? null,
    };
  } catch (error) {
    console.warn("[previewStorage] Failed to read stored video preview:", error);
    try {
      localStorage.removeItem(keys.video);
    } catch {
      /* storage unavailable; nothing to clean up */
    }
    return null;
  }
}

/** Persist a video result and drop any older image preview for the panel. */
export function saveVideoPreview(keys: PreviewStorageKeys, preview: StoredVideoPreview): void {
  try {
    localStorage.setItem(keys.video, JSON.stringify(preview));
    localStorage.removeItem(keys.image);
  } catch (error) {
    console.warn("[previewStorage] Failed to store video preview:", error);
  }
}

/** Persist an image result and drop any older video preview for the panel. */
export function saveImagePreview(keys: PreviewStorageKeys, url: string): void {
  try {
    localStorage.setItem(keys.image, url);
    localStorage.removeItem(keys.video);
  } catch (error) {
    console.warn("[previewStorage] Failed to store image preview:", error);
  }
}

export function clearVideoPreview(keys: PreviewStorageKeys): void {
  try {
    localStorage.removeItem(keys.video);
  } catch (error) {
    console.warn("[previewStorage] Failed to clear video preview:", error);
  }
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
