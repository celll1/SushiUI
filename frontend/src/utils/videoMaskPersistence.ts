/**
 * Reload-persistence for the video-inpaint mask timeline (InpaintPanel's
 * `videoMaskManifest` + `videoMaskAssets`). This is orchestration only: the
 * actual PNG storage mechanism is the existing `tempImageStorage.ts` (backend
 * temp directory, not localStorage/IndexedDB) -- the same mechanism the
 * static inpaint mask already uses. Only small JSON (keyframes, feather,
 * canvas size, and a `temp_img://` ref per asset) goes into localStorage.
 *
 * Keyframes and asset refs are always written together in one
 * `JSON.stringify` per call to `persistVideoMaskManifest`, so a reload can
 * never see one half of a SINGLE call's write updated without the other.
 * That guarantee does NOT extend across separate, concurrently-issued calls
 * to `persistVideoMaskManifest`/`clearVideoMaskPersistence` -- this module
 * has no lock of its own. Callers that can fire more than one of these in
 * flight at a time (InpaintPanel's manifest/asset-driven persist effect) MUST
 * serialize their own calls (see InpaintPanel's `videoMaskPersistChainRef`)
 * and should pass `isCurrent` so a call that is superseded before it reaches
 * its own write skips that write instead of clobbering a newer one.
 */

import { saveTempImage, loadTempImage, deleteTempImageRef, isTempImageRef } from "./tempImageStorage";
import {
  validateVideoMaskManifest,
  type VideoMaskAsset,
  type VideoMaskKeyframe,
  type VideoMaskManifest,
} from "./videoMaskTimeline";
import type { MaskPolarity } from "./maskConventions";

export const VIDEO_MASK_STORAGE_KEY = "inpaint_video_mask_manifest";

/**
 * Identifies the clip a persisted manifest was drawn against. The input clip
 * itself lives in a single-slot IndexedDB record (`mediaInputStorage.ts` --
 * one fixed key, not content-addressed), so there is no existing content
 * hash to compare against. name + size + lastModified is the same cheap,
 * already-available File identity most browser apps use to tell "this is not
 * the file I had before" apart from "the same file was re-selected", without
 * reading/hashing the (potentially large) video content.
 */
export interface VideoMaskClipSignature {
  name: string;
  size: number;
  lastModified: number;
}

export function clipSignatureOf(file: File | null): VideoMaskClipSignature | null {
  if (!file) return null;
  return { name: file.name, size: file.size, lastModified: file.lastModified };
}

export function clipSignaturesMatch(
  a: VideoMaskClipSignature | null | undefined,
  b: VideoMaskClipSignature | null | undefined,
): boolean {
  if (!a || !b) return false;
  return a.name === b.name && a.size === b.size && a.lastModified === b.lastModified;
}

/** In-memory record of what this session has already uploaded, so repeated
 * saves for an unchanged asset do not re-upload it. Callers own this map
 * (create one per panel instance) and pass it into every
 * `persistVideoMaskManifest` call; `loadVideoMaskManifest` returns a fresh
 * one seeded from a successful restore. */
export type VideoMaskAssetRefMap = Map<string, { dataUrl: string; ref: string }>;

/**
 * Thrown by `persistVideoMaskManifest` when `saveTempImage` silently fell
 * back to returning the raw base64 dataURL as the "ref" (backend temp
 * storage unreachable, image small enough for the fallback -- see
 * `tempImageStorage.ts`). A video mask asset's ref MUST be a
 * `temp_img://` backend reference: writing an inline dataURL into the
 * localStorage record would (a) grow with every keyframe instead of storing
 * a fixed-size reference, risking `QuotaExceededError` on write, and (b)
 * self-lock once written, because a later successful upload is only
 * triggered by `existing.dataUrl !== asset.dataUrl` -- an inline ref already
 * equals the asset's own `dataUrl`, so the diff would never see it as stale
 * again even after the backend comes back. The whole call is aborted before
 * any of its results (refs, localStorage) are committed, matching
 * `loadVideoMaskManifest`'s all-or-nothing restore.
 */
export class VideoMaskTempStorageUnavailableError extends Error {
  constructor() {
    super(
      "A video mask asset could not be uploaded to backend temp storage " +
      "(the upload fell back to inline storage, which the video mask " +
      "timeline does not persist).",
    );
    this.name = "VideoMaskTempStorageUnavailableError";
  }
}

interface StoredVideoMaskAsset {
  id: string;
  ref: string;
  width?: number;
  height?: number;
}

interface StoredVideoMaskManifest {
  version: 1;
  clip: VideoMaskClipSignature | null;
  manifest: {
    version: 1;
    coordinateSpace: "output_canvas";
    canvas: { width: number; height: number };
    polarity: MaskPolarity;
    keyframes: VideoMaskKeyframe[];
    compositeFeatherPx: number;
  };
  assets: StoredVideoMaskAsset[];
}

/**
 * Diffs `assets` against `previousRefs` (this session's own record of what it
 * already uploaded) and uploads/deletes only what changed, then writes the
 * manifest + resulting ref list to localStorage in one `JSON.stringify`.
 *
 * An empty manifest (no keyframes and no assets) clears the persisted record
 * entirely instead of writing an empty shell -- there is nothing left worth
 * restoring, and leaving a stale record around would outlive the PNGs it
 * once referenced once cleanup below runs.
 *
 * `previousRefs` is mutated in place so it always reflects what is currently
 * uploaded, for correct diffing on the next call.
 *
 * `isCurrent`, if provided, is re-checked immediately before every localStorage
 * write (including the "clear" write for an empty manifest). If it returns
 * false the write is skipped -- a newer call for the same clip has already
 * been enqueued behind this one and will perform the write that actually
 * reflects the latest state, so writing here would either be redundant or,
 * worse, momentarily overwrite that newer state with this stale one. The
 * asset uploads/deletes already performed before the check still stand
 * (they keep `previousRefs` correct for the next call's diff); only the
 * localStorage write itself is skipped.
 *
 * Throws `VideoMaskTempStorageUnavailableError` (without writing anything to
 * `previousRefs` or localStorage) if any asset upload fell back to inline
 * base64 storage -- see that class's doc comment.
 */
export async function persistVideoMaskManifest(
  manifest: VideoMaskManifest,
  assets: VideoMaskAsset[],
  clip: VideoMaskClipSignature,
  previousRefs: VideoMaskAssetRefMap,
  isCurrent?: () => boolean,
): Promise<void> {
  if (manifest.keyframes.length === 0 && assets.length === 0) {
    if (isCurrent && !isCurrent()) return;
    await clearVideoMaskPersistence();
    previousRefs.clear();
    return;
  }

  const nextRefs: VideoMaskAssetRefMap = new Map();
  const storedAssets: StoredVideoMaskAsset[] = [];

  for (const asset of assets) {
    const existing = previousRefs.get(asset.id);
    if (existing && existing.dataUrl === asset.dataUrl) {
      nextRefs.set(asset.id, existing);
    } else {
      // New asset, or the same id redrawn in place with different content:
      // release the stale backend temp file (if any) before uploading the
      // replacement so it does not leak.
      if (existing) {
        await deleteTempImageRef(existing.ref).catch((error) =>
          console.error("[videoMaskPersistence] Failed to release a stale mask asset:", error),
        );
      }
      const ref = await saveTempImage(asset.dataUrl);
      if (!isTempImageRef(ref)) {
        // Backend temp storage is unreachable and saveTempImage fell back to
        // handing back the inline dataURL. Abort the WHOLE call now, before
        // touching previousRefs or localStorage -- see
        // VideoMaskTempStorageUnavailableError's doc comment. Any assets
        // uploaded earlier in this same loop iteration are left as
        // successfully-uploaded backend temp files that this call never
        // recorded; they are released the same way any other orphaned temp
        // file is (manual "Clear temp images" / the backend's 24h sweep).
        throw new VideoMaskTempStorageUnavailableError();
      }
      nextRefs.set(asset.id, { dataUrl: asset.dataUrl, ref });
    }
    const stored = nextRefs.get(asset.id)!;
    storedAssets.push({ id: asset.id, ref: stored.ref, width: asset.width, height: asset.height });
  }

  // Any id previousRefs still has but nextRefs does not is an asset that is
  // no longer in state (deleted keyframe/duplicate GC) -- release it too.
  for (const [id, entry] of previousRefs) {
    if (!nextRefs.has(id)) {
      await deleteTempImageRef(entry.ref).catch((error) =>
        console.error("[videoMaskPersistence] Failed to release a removed mask asset:", error),
      );
    }
  }

  previousRefs.clear();
  for (const [id, entry] of nextRefs) previousRefs.set(id, entry);

  if (isCurrent && !isCurrent()) return;

  const payload: StoredVideoMaskManifest = {
    version: 1,
    clip,
    manifest: {
      version: 1,
      coordinateSpace: "output_canvas",
      canvas: { ...manifest.canvas },
      polarity: manifest.polarity,
      keyframes: manifest.keyframes,
      compositeFeatherPx: manifest.compositeFeatherPx,
    },
    assets: storedAssets,
  };
  localStorage.setItem(VIDEO_MASK_STORAGE_KEY, JSON.stringify(payload));
}

/**
 * Releases every backend temp file a persisted record references, then
 * removes the record itself. Reads directly from localStorage (rather than
 * requiring the caller's in-memory ref map) so it works correctly even when
 * called before any restore/save happened this session -- e.g. on mount with
 * no video clip loaded at all, or when a clip is replaced/cleared before its
 * mask manifest was ever touched this session.
 */
export async function clearVideoMaskPersistence(): Promise<void> {
  const raw = localStorage.getItem(VIDEO_MASK_STORAGE_KEY);
  localStorage.removeItem(VIDEO_MASK_STORAGE_KEY);
  if (!raw) return;
  try {
    const parsed = JSON.parse(raw) as StoredVideoMaskManifest;
    for (const asset of parsed.assets ?? []) {
      await deleteTempImageRef(asset.ref).catch((error) =>
        console.error("[videoMaskPersistence] Failed to release a mask asset during cleanup:", error),
      );
    }
  } catch {
    // Malformed record; the key is already removed above, nothing else to clean up.
  }
}

/**
 * Releases every asset a session has EVER uploaded for a clip -- everything
 * still in `previousRefs` (in-memory), not just whatever last made it into
 * the committed localStorage record -- then clears both the record and the
 * map itself.
 *
 * Callers whose clip is being replaced/removed while a `persistVideoMaskManifest`
 * call for it may still be queued/in-flight (see InpaintPanel's
 * `videoMaskPersistChainRef`) MUST use this INSTEAD OF calling
 * `clearVideoMaskPersistence()` alone: a queued call whose write got skipped
 * because a newer call had already been enqueued behind it (its `isCurrent`
 * check -- see `persistVideoMaskManifest`) still uploads its assets and
 * records them in `previousRefs` before skipping only the localStorage write.
 * Those uploads are real backend temp files; `clearVideoMaskPersistence()`
 * alone, which only ever reads the last COMMITTED localStorage record, would
 * never learn about them and would leak them.
 *
 * The caller must pass the SAME map object instance every earlier-queued
 * call for this clip received as `previousRefs` (do not `.clear()` it or
 * reassign the ref to a new Map before this call finishes -- hand this
 * function the old object and point the ref at a NEW empty Map for whatever
 * comes next instead, so a still-queued call for the old clip and this
 * cleanup keep operating on the one object consistently).
 */
export async function releaseAllTrackedMaskAssets(previousRefs: VideoMaskAssetRefMap): Promise<void> {
  for (const entry of previousRefs.values()) {
    await deleteTempImageRef(entry.ref).catch((error) =>
      console.error("[videoMaskPersistence] Failed to release a tracked mask asset:", error),
    );
  }
  await clearVideoMaskPersistence();
  previousRefs.clear();
}

/**
 * Structural validation of a `JSON.parse`d localStorage record, run BEFORE
 * any asset is restored. `parsed.version` was declared as the literal `1`
 * but nothing ever checked it at read time, and `stored.ref` could be
 * missing/non-string without this catching it -- both fell through to
 * `loadTempImage(undefined)` -> `""` -> `{status: "aborted"}`, i.e. treated
 * as a transient, retryable failure instead of the doc-promised `"none"`
 * (discard, no retry) for genuine corruption. Deliberately loose about
 * `manifest`/`clip` shape beyond "is an object" -- the field-level
 * `validateVideoMaskManifest` call further down still runs and is the
 * authority on manifest content.
 */
function isValidStoredManifest(value: unknown): value is StoredVideoMaskManifest {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  if (record.version !== 1) return false;
  if (record.clip !== null && (typeof record.clip !== "object" || record.clip === undefined)) return false;
  if (!record.manifest || typeof record.manifest !== "object") return false;
  if (!Array.isArray(record.assets)) return false;
  for (const asset of record.assets) {
    if (!asset || typeof asset !== "object") return false;
    const a = asset as Record<string, unknown>;
    if (typeof a.id !== "string" || a.id.length === 0) return false;
    if (typeof a.ref !== "string" || a.ref.length === 0) return false;
  }
  return true;
}

export type VideoMaskRestoreOutcome =
  | { status: "none" }
  | {
      status: "ok";
      manifest: VideoMaskManifest;
      assets: VideoMaskAsset[];
      refs: VideoMaskAssetRefMap;
    }
  | { status: "aborted" };

/**
 * Loads a persisted manifest ONLY if its recorded clip signature matches
 * `currentClip` -- a mask drawn against a different clip's frames is
 * meaningless composited onto this one.
 *
 * `loadTempImage` collapses every failure (the temp file genuinely expired
 * on the backend vs. the backend simply not being reachable yet) into an
 * empty string, so there is no reliable way to tell "permanently gone" apart
 * from "transiently unavailable" here. Rather than guess and risk silently
 * dropping keyframes that were only temporarily unreachable, ANY asset load
 * failure aborts the WHOLE restore (`status: "aborted"`) -- callers must
 * leave the manifest untouched (default/empty) and must NOT write anything
 * back to localStorage in that case, so the persisted record survives for a
 * later retry once the backend is reachable.
 *
 * A structurally invalid record -- wrong/missing `version`, an `assets`
 * entry missing its `id`/`ref` (checked by `isValidStoredManifest` before
 * any asset is touched), or a keyframe referencing an asset id that was
 * never in `assets` to begin with (checked by `validateVideoMaskManifest`
 * after assets are restored) -- is corruption, not a network issue, and is
 * discarded outright (`status: "none"`): unlike a transient failure, there
 * is no reason to expect a retry would recover it.
 */
export async function loadVideoMaskManifest(
  currentClip: VideoMaskClipSignature | null,
): Promise<VideoMaskRestoreOutcome> {
  const raw = localStorage.getItem(VIDEO_MASK_STORAGE_KEY);
  if (!raw) return { status: "none" };

  let parsed: StoredVideoMaskManifest;
  try {
    const rawParsed = JSON.parse(raw);
    if (!isValidStoredManifest(rawParsed)) return { status: "none" };
    parsed = rawParsed;
  } catch {
    return { status: "none" };
  }

  if (!clipSignaturesMatch(parsed.clip, currentClip)) return { status: "none" };

  const refs: VideoMaskAssetRefMap = new Map();
  const assets: VideoMaskAsset[] = [];
  for (const stored of parsed.assets ?? []) {
    let dataUrl = "";
    try {
      dataUrl = await loadTempImage(stored.ref);
    } catch {
      dataUrl = "";
    }
    if (!dataUrl) {
      return { status: "aborted" };
    }
    assets.push({ id: stored.id, dataUrl, width: stored.width, height: stored.height });
    refs.set(stored.id, { dataUrl, ref: stored.ref });
  }

  const manifest: VideoMaskManifest = {
    version: 1,
    coordinateSpace: "output_canvas",
    canvas: parsed.manifest?.canvas,
    polarity: parsed.manifest?.polarity,
    keyframes: parsed.manifest?.keyframes ?? [],
    compositeFeatherPx: parsed.manifest?.compositeFeatherPx,
  };

  const validation = validateVideoMaskManifest({ ...manifest, assets });
  if (!validation.valid) {
    console.error("[videoMaskPersistence] Discarding an unrestorable video mask record:", validation.errors);
    return { status: "none" };
  }

  return { status: "ok", manifest, assets, refs };
}
