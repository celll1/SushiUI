// Global user preference for the video frame count new generations start
// from, in the same localStorage-style store as `attentionSettings.ts`
// (Attention Type / Attention Implementation). That precedent is followed
// deliberately rather than the backend-persisted `UserSettings` row
// (`DirectorySettings.tsx` / `POST /settings`): a frame-count starting point
// is per-browser UI state consumed entirely on the frontend at seed time --
// unlike a directory path, the backend has no reason to know it, validate
// it, or make it available to a second frontend client. It is resolved into
// a concrete `num_frames` value client-side and only THAT value is ever
// sent in a request.
//
// Storing the raw preference here does NOT snap it onto any architecture's
// frame grid -- a single global number cannot be valid for every
// architecture at once (MiniMax-H3 is `17n+5`, LTX-2.3 is `8k+1`, each with
// its own floor/ceiling). Snapping happens once, at the point of use, via
// the same `normalizeVideoFrames` helper (api.ts) the panels already use to
// re-snap a persisted `num_frames` when the loaded architecture changes --
// no second copy of that arithmetic lives here.

const STORAGE_KEY = "default_video_frame_count";

const readStorage = (): string | null => {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
};

/**
 * The raw stored preference, unsnapped. `null` means "unset" -- use the
 * architecture default -- which is also the state before the user has ever
 * opened Settings, so nothing changes for anyone who never visits that page.
 */
export const readGlobalVideoFrameCount = (): number | null => {
  const raw = readStorage();
  if (raw == null) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
};

/** `null` clears the preference (removes the key) rather than storing it. */
export const writeGlobalVideoFrameCount = (value: number | null): void => {
  if (typeof window === "undefined") return;
  try {
    if (value == null) {
      window.localStorage.removeItem(STORAGE_KEY);
    } else {
      window.localStorage.setItem(STORAGE_KEY, String(value));
    }
  } catch {
    // Best-effort, same as every other localStorage write in this app.
  }
};
