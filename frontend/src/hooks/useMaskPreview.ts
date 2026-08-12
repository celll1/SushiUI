"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  previewVideoMask,
  VideoMaskRefUnresolvedError,
  type VideoMaskPreviewResult,
} from "@/utils/api";
import {
  serializeVideoMaskManifestForApi,
  type VideoMaskAsset,
  type VideoMaskManifest,
} from "@/utils/videoMaskTimeline";
import type { VideoMaskAssetRefMap } from "@/utils/videoMaskPersistence";

const DEBOUNCE_MS = 500;

export interface MaskPreviewState {
  /** The rasterized sprite + its geometry, or null before the first successful fetch. */
  result: VideoMaskPreviewResult | null;
  /** True while a request for the CURRENT input is debounced or in flight. */
  isPending: boolean;
  /** The last request's failure message, or null. `result` (if any) still holds the last SUCCESSFUL fetch, which `isStale` below says whether to trust. */
  error: string | null;
  /**
   * True whenever `result` was fetched for different input (manifest, asset
   * PNGs, frame list, or maxSize) than what is currently passed in. A caller
   * must not present a stale `result` as an accurate preview of the CURRENT
   * mask -- e.g. render it dimmed, or not at all, until this clears.
   */
  isStale: boolean;
}

async function assetToFilePart(id: string, dataUrl: string): Promise<{ id: string; file: File }> {
  const response = await fetch(dataUrl);
  if (!response.ok) throw new Error(`Could not read mask asset ${id}.`);
  const blob = await response.blob();
  return { id, file: new File([blob], `${id}.png`, { type: "image/png" }) };
}

/**
 * Splits `referencedIds` into upload parts and ref parts for one
 * `previewVideoMask` call. An id goes to `refParts` only when `refs` holds an
 * entry for it whose `dataUrl` still matches the CURRENT asset's `dataUrl`
 * (per-asset freshness proof) and is not in `forceUploadIds` -- everything
 * else is read from its data URL and uploaded, exactly as before this ref
 * path existed. `forceUploadIds` is how the 409-retry below downgrades just
 * the asset(s) the backend could not resolve, without discarding a fresh ref
 * for every OTHER asset in the same request.
 */
async function buildRequestParts(
  referencedIds: string[],
  assetsById: Map<string, VideoMaskAsset>,
  refs: VideoMaskAssetRefMap | undefined,
  forceUploadIds: ReadonlySet<string>,
): Promise<{ fileParts: Array<{ id: string; file: File }>; refParts: Array<{ id: string; ref: string }> }> {
  const fileParts: Array<{ id: string; file: File }> = [];
  const refParts: Array<{ id: string; ref: string }> = [];
  for (const id of referencedIds) {
    const asset = assetsById.get(id);
    if (!asset) throw new Error(`No saved mask image for id ${id}.`);
    const refEntry = refs?.get(id);
    if (!forceUploadIds.has(id) && refEntry && refEntry.dataUrl === asset.dataUrl) {
      refParts.push({ id, ref: refEntry.ref });
    } else {
      fileParts.push(await assetToFilePart(id, asset.dataUrl));
    }
  }
  return { fileParts, refParts };
}

/**
 * Debounces a spatial mask manifest + its PNG assets + a requested frame
 * list into ONE `/video-mask/preview` fetch, and reports whether the held
 * `result` still corresponds to the CURRENT input via `isStale` -- so a
 * caller never draws an old rasterization as if it were the current one.
 *
 * Rasterization itself (including `sdf`'s distance-transform morph) is never
 * reimplemented here: this hook is purely network plumbing, request
 * de-duplication, and staleness bookkeeping around the backend response.
 *
 * `frames` is the caller's OWN choice of which frames to sample (e.g.
 * `computeMaskPreviewSampleFrames` in `videoMaskTimeline.ts`); this hook does
 * not decide that -- it only fetches exactly the frames it is given, once,
 * per debounced input change.
 *
 * `refs`, if given, is a caller's live record of which assets are already
 * uploaded to backend temp storage (`videoMaskPersistence.ts`'s
 * `VideoMaskAssetRefMap`) -- an asset whose ref is still fresh (see
 * `buildRequestParts`) is sent as a few-hundred-byte ref instead of
 * re-uploading its full PNG on every debounced request. Omit it (or pass
 * `undefined`) to always upload, unchanged from before this existed.
 */
export function useMaskPreview(
  manifest: VideoMaskManifest,
  assets: VideoMaskAsset[],
  frames: number[],
  maxSize = 256,
  refs?: VideoMaskAssetRefMap,
): MaskPreviewState {
  const [held, setHeld] = useState<{ result: VideoMaskPreviewResult; key: string } | null>(null);
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const versionRef = useRef(0);

  // Recomputed only when `manifest`/`assets`/`frames`/`maxSize`/`refs`
  // actually change reference -- NOT on every render. `VideoMaskPreviewOverlay`
  // passes `currentFrame` through this hook's caller at up to video frame
  // rate during playback (it re-renders on every playhead tick), and without
  // this memo, `assetDigest` below would re-concatenate every referenced
  // asset's FULL data-URL (each up to `maxSize`x`maxSize` px of PNG-as-
  // base64, several hundred KB) on every one of those re-renders even though
  // none of manifest/assets/frames/maxSize/refs changed.
  const { referencedIds, assetsById, dedupedFrames, key } = useMemo(() => {
    const ids = Array.from(new Set(manifest.keyframes.map((keyframe) => keyframe.maskId))).sort();
    const byId = new Map(assets.map((asset) => [asset.id, asset]));
    // Folds every referenced asset's actual PNG content into the key (not
    // just its id): editing a mask's pixels in place (same id, new dataUrl)
    // must count as a distinct input, or a stale sprite would keep being
    // treated as current after the user redraws a mask without changing any
    // id.
    const digest = ids.map((id) => `${id}:${byId.get(id)?.dataUrl ?? ""}`).join("|");
    // A fresh ref (matching the CURRENT dataUrl) is folded in too, keyed
    // separately from `digest` above: the rasterized OUTPUT never depends on
    // whether an asset was fetched by ref or by upload, only its bytes do
    // (already covered by `digest`). This is kept as its own key field
    // anyway so a ref transitioning between fresh/stale/absent still forces
    // `buildRequestParts` to run for the new `key`, rather than being read
    // only from a `held` result computed under a different ref state.
    const refDigest = ids
      .map((id) => {
        const entry = refs?.get(id);
        const asset = byId.get(id);
        return entry && asset && entry.dataUrl === asset.dataUrl ? `${id}:${entry.ref}` : `${id}:`;
      })
      .join("|");
    const dedupedFrameList = Array.from(new Set(frames)).sort((a, b) => a - b);
    const computedKey = JSON.stringify({
      manifest: manifestKeyPart(manifest),
      assetDigest: digest,
      refDigest,
      frames: dedupedFrameList,
      maxSize,
    });
    return { referencedIds: ids, assetsById: byId, dedupedFrames: dedupedFrameList, key: computedKey };
  }, [manifest, assets, frames, maxSize, refs]);

  // Revokes the PREVIOUS held sprite's blob object URL once `held` moves on
  // to a new one (or to null) -- `previewVideoMask` hands back a `blob:` URL
  // per fetch (see api.ts), and nothing else in this hook's contract ever
  // reads an old one again once a newer `held` has replaced it. Runs as a
  // cleanup (not inline in the state setter) so the revoke happens after
  // this render has committed and any consumer reading the new `held.result`
  // this render has already picked it up.
  const heldUrlRef = useRef<string | null>(null);
  useEffect(() => {
    const previousUrl = heldUrlRef.current;
    const nextUrl = held?.result.strip_png ?? null;
    heldUrlRef.current = nextUrl;
    return () => {
      if (previousUrl && previousUrl !== nextUrl) URL.revokeObjectURL(previousUrl);
    };
  }, [held]);
  useEffect(() => {
    return () => {
      if (heldUrlRef.current) URL.revokeObjectURL(heldUrlRef.current);
    };
    // Unmount-only cleanup; the effect above already revokes every
    // superseded URL, this just catches the LAST one.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (referencedIds.length === 0) {
      // No keyframes reference any mask asset (e.g. every keyframe was just
      // deleted, or undone away). Any held sprite was rasterized for a
      // manifest that no longer exists -- it must not be handed back as
      // `result` at all, since `isStale` alone is not a strong enough
      // contract for a caller that only checks "do I have a result" (rather
      // than also checking `isStale`) before drawing it.
      setHeld(null);
      setIsPending(false);
      setError(null);
      return;
    }
    if (dedupedFrames.length === 0) {
      // Masks still exist, but the caller is not currently requesting any
      // frames (e.g. the overlay is toggled off). Leave `held` alone --
      // `isStale` will already read true once frames are requested again
      // with a different key, and there is nothing wrong to redraw here in
      // the meantime since nothing is being drawn (frames.length === 0).
      setIsPending(false);
      setError(null);
      return;
    }
    const missingAssetId = referencedIds.find((id) => !assetsById.get(id));
    if (missingAssetId) {
      setIsPending(false);
      setError(`No saved mask image for id ${missingAssetId}.`);
      return;
    }

    let cancelled = false;
    versionRef.current += 1;
    const myVersion = versionRef.current;
    setIsPending(true);
    setError(null);

    const timer = setTimeout(() => {
      (async () => {
        try {
          const manifestJson = serializeVideoMaskManifestForApi(manifest, assets);
          const noForce: ReadonlySet<string> = new Set();
          const parts = await buildRequestParts(referencedIds, assetsById, refs, noForce);
          if (cancelled || versionRef.current !== myVersion) return;

          let result: VideoMaskPreviewResult;
          try {
            result = await previewVideoMask(manifestJson, parts.fileParts, parts.refParts, dedupedFrames, maxSize);
          } catch (err) {
            // A ref the freshness check above trusted turned out to be gone
            // server-side (e.g. swept by the temp-file cleanup) -- retry
            // ONCE with exactly those asset(s) uploaded as bytes instead of
            // guessing, or looping if the retry itself somehow reports the
            // same ids again.
            if (err instanceof VideoMaskRefUnresolvedError && err.unresolvedRefIds.length > 0) {
              const forceUploadIds = new Set(err.unresolvedRefIds);
              // These ids' refs are dead (e.g. swept). Drop them from `refs`
              // BEFORE retrying, not just for this one request via
              // `forceUploadIds` -- `refs` is the caller's live map
              // (`videoMaskPersistence.ts`), read again by every LATER
              // debounced fetch for this same unchanged asset. Leaving the
              // dead entry in place would make every later fetch pay a full
              // 409-then-retry round trip forever instead of uploading
              // directly.
              if (refs) {
                for (const id of forceUploadIds) refs.delete(id);
              }
              const retryParts = await buildRequestParts(referencedIds, assetsById, refs, forceUploadIds);
              if (cancelled || versionRef.current !== myVersion) return;
              result = await previewVideoMask(
                manifestJson, retryParts.fileParts, retryParts.refParts, dedupedFrames, maxSize,
              );
            } else {
              throw err;
            }
          }
          if (cancelled || versionRef.current !== myVersion) {
            // This fetch's blob sprite (from either the initial or the
            // 409-retry attempt above) was never handed to `setHeld`, so
            // nothing else will ever revoke it -- the cleanup effect that
            // owns `held`'s blob lifecycle only tracks URLs that actually
            // made it into `held`.
            URL.revokeObjectURL(result.strip_png);
            return;
          }
          setHeld({ result, key });
          setError(null);
        } catch (err) {
          if (cancelled || versionRef.current !== myVersion) return;
          setError(err instanceof Error ? err.message : "Video mask preview failed.");
        } finally {
          if (!cancelled && versionRef.current === myVersion) setIsPending(false);
        }
      })();
    }, DEBOUNCE_MS);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // key alone determines whether a new request is needed; manifest/assets/
    // frames/maxSize/refs are all folded into it above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const isStale = held == null || held.key !== key;
  return { result: held?.result ?? null, isPending, error, isStale };
}

/** manifest, minus `assets` (already folded into assetDigest by the caller,
 * via each referenced id's own dataUrl) -- so the same PNG bytes are not
 * counted twice under two different key fields. */
function manifestKeyPart(manifest: VideoMaskManifest): unknown {
  const { assets: _omit, ...rest } = manifest;
  return rest;
}
