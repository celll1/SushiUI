"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { previewVideoMask, type VideoMaskPreviewResult } from "@/utils/api";
import {
  serializeVideoMaskManifestForApi,
  type VideoMaskAsset,
  type VideoMaskManifest,
} from "@/utils/videoMaskTimeline";

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
 */
export function useMaskPreview(
  manifest: VideoMaskManifest,
  assets: VideoMaskAsset[],
  frames: number[],
  maxSize = 256,
): MaskPreviewState {
  const [held, setHeld] = useState<{ result: VideoMaskPreviewResult; key: string } | null>(null);
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const versionRef = useRef(0);

  // Recomputed only when `manifest`/`assets`/`frames`/`maxSize` actually
  // change reference -- NOT on every render. `VideoMaskPreviewOverlay`
  // passes `currentFrame` through this hook's caller at up to video frame
  // rate during playback (it re-renders on every playhead tick), and without
  // this memo, `assetDigest` below would re-concatenate every referenced
  // asset's FULL data-URL (each up to `maxSize`x`maxSize` px of PNG-as-
  // base64, several hundred KB) on every one of those re-renders even though
  // none of manifest/assets/frames/maxSize changed.
  const { referencedIds, assetsById, dedupedFrames, key } = useMemo(() => {
    const ids = Array.from(new Set(manifest.keyframes.map((keyframe) => keyframe.maskId))).sort();
    const byId = new Map(assets.map((asset) => [asset.id, asset]));
    // Folds every referenced asset's actual PNG content into the key (not
    // just its id): editing a mask's pixels in place (same id, new dataUrl)
    // must count as a distinct input, or a stale sprite would keep being
    // treated as current after the user redraws a mask without changing any
    // id.
    const digest = ids.map((id) => `${id}:${byId.get(id)?.dataUrl ?? ""}`).join("|");
    const dedupedFrameList = Array.from(new Set(frames)).sort((a, b) => a - b);
    const computedKey = JSON.stringify({
      manifest: manifestKeyPart(manifest),
      assetDigest: digest,
      frames: dedupedFrameList,
      maxSize,
    });
    return { referencedIds: ids, assetsById: byId, assetDigest: digest, dedupedFrames: dedupedFrameList, key: computedKey };
  }, [manifest, assets, frames, maxSize]);

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
          const parts = await Promise.all(
            referencedIds.map(async (id) => {
              const asset = assetsById.get(id);
              if (!asset) throw new Error(`No saved mask image for id ${id}.`);
              const response = await fetch(asset.dataUrl);
              if (!response.ok) throw new Error(`Could not read mask asset ${id}.`);
              const blob = await response.blob();
              return { id, file: new File([blob], `${id}.png`, { type: "image/png" }) };
            }),
          );
          if (cancelled || versionRef.current !== myVersion) return;
          const result = await previewVideoMask(manifestJson, parts, dedupedFrames, maxSize);
          if (cancelled || versionRef.current !== myVersion) return;
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
    // frames/maxSize are all folded into it above.
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
