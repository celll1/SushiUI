"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useMaskPreview } from "@/hooks/useMaskPreview";
import { computeCoverCropDisplayRect } from "@/utils/canvasFit";
import {
  MASK_OVERLAY_ALPHA,
  MASK_OVERLAY_CANVAS_COMPOSITE_OPERATION,
} from "@/utils/maskConventions";
import {
  canRasterizeMaskManifestClientSide,
  maskTransformToCanvasMatrix,
  resolveMaskAtFrame,
} from "@/utils/videoMaskRaster";
import {
  computeMaskPreviewSampleFrames,
  type VideoMaskAsset,
  type VideoMaskManifest,
} from "@/utils/videoMaskTimeline";
import type { VideoMaskAssetRefMap } from "@/utils/videoMaskPersistence";

const PREVIEW_MAX_SIZE = 256;

export interface VideoMaskPreviewOverlayProps {
  /** The SAME <video> element the panel renders for preview/scrub. */
  videoRef: React.RefObject<HTMLVideoElement | null>;
  /** The uploaded clip's own native pixel size (its intrinsic decode size, not its on-screen box). Null before metadata loads -- nothing is drawn until then. */
  nativeSize: { width: number; height: number } | null;
  /** The mask timeline's own canvas size (the output canvas the manifest's coordinates are defined against). */
  outputWidth: number;
  outputHeight: number;
  manifest: VideoMaskManifest;
  assets: VideoMaskAsset[];
  /**
   * What the panel has already uploaded each asset under, so the `sdf`
   * fallback can send a reference instead of the PNG bytes. Only used on
   * that path, and only for an asset whose recorded dataUrl still matches
   * the live one -- see useMaskPreview.
   */
  assetRefs?: VideoMaskAssetRefMap;
  /** The regenerate range (pixel frames of the trimmed clip), used only to choose sample frames for the `sdf` backend fallback -- see computeMaskPreviewSampleFrames. */
  rangeStart: number;
  rangeEnd: number;
  /**
   * The live playhead frame, in TRIMMED-clip coordinates -- the SAME
   * coordinate space as `rangeStart`/`rangeEnd` and every keyframe's own
   * `frame` (see `VideoInpaintTimeline`'s `- trimStart`). This is NOT the
   * raw `<video>` element's own frame number (`useVideoPlayhead`'s
   * `currentFrame`) unless `input_trim_start_frames` is 0 -- the caller must
   * subtract the trim-start offset before passing it in here, or both the
   * client rasterizer and the `sdf` "nearest sample" fallback below silently
   * pick the wrong frame's mask.
   * May be null: when there is no live playhead yet, this falls back to the
   * manifest's own first keyframe (its "before the first keyframe" hold
   * behavior), matching the `sdf` fallback path's previous "nearest is the
   * earliest sample" behavior for the same case.
   */
  currentFrame: number | null;
  enabled: boolean;
  /** 0..1. Independent of MASK_OVERLAY_ALPHA, which is the fixed blend alpha baked into the highlight itself; this is a user-facing overall opacity on top of it. */
  opacity: number;
}

/**
 * Overlays a spatial mask timeline's rasterized preview onto the input clip
 * `<video>`, aligned for `object-contain` display vs. the backend's own
 * center-crop-cover mapping (see `computeCoverCropDisplayRect`).
 *
 * Rasterization for `hold` and `affine` interpolation happens IN THE BROWSER
 * (`videoMaskRaster.ts`), for the EXACT current playhead frame, with no
 * network request at all -- `useMaskPreview` is called with an empty frame
 * list in that case, which it treats as "request nothing" (see its own
 * early-return on `dedupedFrames.length === 0`). `sdf` interpolation is a
 * signed-distance-field morph between two DIFFERENT mask assets that this
 * component does not reimplement; whenever any keyframe pair that could
 * govern a displayed frame uses it (`canRasterizeMaskManifestClientSide`
 * returns false), this falls back to the ORIGINAL behavior in full: request
 * the backend's sampled sprite strip (`computeMaskPreviewSampleFrames` /
 * `/video-mask/preview`) and draw whichever sample is nearest the live
 * playhead.
 */
export default function VideoMaskPreviewOverlay({
  videoRef,
  nativeSize,
  outputWidth,
  outputHeight,
  manifest,
  assets,
  assetRefs,
  rangeStart,
  rangeEnd,
  currentFrame,
  enabled,
  opacity,
}: VideoMaskPreviewOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [containerSize, setContainerSize] = useState<{ width: number; height: number } | null>(null);
  const spriteImageRef = useRef<{ src: string; image: HTMLImageElement } | null>(null);

  // Decoded HTMLImageElements for client-rasterized assets, memoized by
  // dataUrl so a scrub tick that keeps resolving to the same asset never
  // redecodes it. Pruned (not just left to grow) whenever `assets` changes,
  // since assets are already capped at MAX_MASK_ASSETS (64) but can still
  // churn across a long editing session -- an unpruned cache would otherwise
  // keep every mask PNG a user ever drew, for the lifetime of the panel.
  const assetImageCacheRef = useRef<Map<string, HTMLImageElement>>(new Map());
  const pendingDecodesRef = useRef<Set<string>>(new Set());
  // dataUrls whose decode has already failed once, so a scrub tick that keeps
  // resolving to the same broken asset doesn't start a fresh Image() every
  // frame -- see L10.
  const failedDecodesRef = useRef<Set<string>>(new Set());
  const [decodeTick, setDecodeTick] = useState(0);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  // Mirrors the `sdf` fallback's own error badge, but for failures specific
  // to the client-rasterized path: a decode that failed, or a mask asset
  // whose own pixel size no longer matches the canvas (see H3) -- both of
  // which would otherwise either draw nothing with no explanation, or draw a
  // confidently wrong preview.
  const [clientRasterError, setClientRasterError] = useState(false);

  const clientRasterizable = useMemo(
    () => canRasterizeMaskManifestClientSide(manifest.keyframes),
    [manifest.keyframes],
  );

  const keyframeFrames = useMemo(
    () => manifest.keyframes.map((keyframe) => keyframe.frame),
    [manifest.keyframes],
  );
  const sampleFrames = useMemo(
    () => computeMaskPreviewSampleFrames(keyframeFrames, rangeStart, rangeEnd),
    [keyframeFrames, rangeStart, rangeEnd],
  );

  // Only the `sdf` fallback path ever needs a server sprite: an empty frame
  // list here means useMaskPreview issues no request at all.
  const preview = useMaskPreview(
    manifest,
    assets,
    enabled && !clientRasterizable ? sampleFrames.frames : [],
    PREVIEW_MAX_SIZE,
    assetRefs,
  );

  // Once a fetch has ever produced a sprite, remember that -- so the
  // "Updating…" badge can still fire after every keyframe is deleted (the
  // mask that WAS there is now gone, which is worth surfacing) without also
  // firing permanently on a pristine timeline that has never had a mask at
  // all (where `preview.isStale` is trivially true forever, since `held`
  // starts and stays null until a first successful fetch). Only meaningful
  // on the `sdf` fallback path -- the client-rasterized path never fetches.
  const everHadResultRef = useRef(false);
  if (preview.result != null) everHadResultRef.current = true;

  // Track the video element's own rendered box (its client size IS the
  // display container: the panel renders it with w-full h-full).
  useEffect(() => {
    const video = videoRef.current;
    if (!video || typeof ResizeObserver === "undefined") return;
    const update = () => setContainerSize({ width: video.clientWidth, height: video.clientHeight });
    update();
    const observer = new ResizeObserver(update);
    observer.observe(video);
    return () => observer.disconnect();
  }, [videoRef]);

  // What `assets` currently contains, kept in a ref so the async decode
  // callbacks below (which close over whatever `assets` was at decode-start
  // time) can instead check the LATEST list before inserting into the
  // shared cache -- see L11.
  const liveDataUrlsRef = useRef<Set<string>>(new Set());

  // Evict decoded images for assets that are no longer referenced by the
  // current manifest/asset list (deleted or replaced), rather than only ever
  // growing this cache for the lifetime of the panel. Also drops any
  // recorded decode failure for a since-removed asset, so a later re-add
  // (same dataUrl reused) gets a fresh attempt rather than staying
  // permanently blacklisted.
  useEffect(() => {
    const cache = assetImageCacheRef.current;
    const liveDataUrls = new Set(assets.map((asset) => asset.dataUrl));
    liveDataUrlsRef.current = liveDataUrls;
    for (const dataUrl of Array.from(cache.keys())) {
      if (!liveDataUrls.has(dataUrl)) cache.delete(dataUrl);
    }
    for (const dataUrl of Array.from(failedDecodesRef.current)) {
      if (!liveDataUrls.has(dataUrl)) failedDecodesRef.current.delete(dataUrl);
    }
  }, [assets]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const clear = () => {
      canvas.width = 0;
      canvas.height = 0;
    };

    if (!enabled || !containerSize || !nativeSize) {
      clear();
      return;
    }

    const displayRect = computeCoverCropDisplayRect(
      containerSize.width, containerSize.height,
      nativeSize.width, nativeSize.height,
      outputWidth, outputHeight,
    );
    if (!displayRect || displayRect.width <= 0 || displayRect.height <= 0) {
      clear();
      setClientRasterError(false);
      return;
    }

    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;

    if (clientRasterizable) {
      // `currentFrame == null` mirrors the sdf fallback's own "nothing
      // scrubbed yet" behavior below: resolve as if before the timeline's
      // first keyframe, which resolveMaskAtFrame does for any frame less
      // than every keyframe's own frame (keyframe.frame is validated >= 0).
      const resolved = resolveMaskAtFrame(manifest.keyframes, currentFrame ?? -1);
      if (!resolved || "needsServer" in resolved) {
        clear();
        setClientRasterError(false);
        return;
      }
      const asset = assets.find((candidate) => candidate.id === resolved.maskId);
      if (!asset) {
        clear();
        setClientRasterError(false);
        return;
      }

      if (failedDecodesRef.current.has(asset.dataUrl)) {
        // Already tried and failed once -- surface the badge without
        // starting another Image() this tick (see L10).
        clear();
        setClientRasterError(true);
        return;
      }

      const cache = assetImageCacheRef.current;
      let image = cache.get(asset.dataUrl);
      if (!image) {
        if (!pendingDecodesRef.current.has(asset.dataUrl)) {
          pendingDecodesRef.current.add(asset.dataUrl);
          const decoding = new Image();
          decoding.onload = () => {
            pendingDecodesRef.current.delete(asset.dataUrl);
            // The asset list may have moved on while this decode was in
            // flight (deleted or replaced mid-decode); only cache it if it's
            // still live, so a stale decode can't repopulate an entry the
            // eviction effect already dropped (see L11).
            if (liveDataUrlsRef.current.has(asset.dataUrl)) {
              assetImageCacheRef.current.set(asset.dataUrl, decoding);
              setDecodeTick((tick) => tick + 1);
            }
          };
          decoding.onerror = () => {
            pendingDecodesRef.current.delete(asset.dataUrl);
            if (liveDataUrlsRef.current.has(asset.dataUrl)) {
              failedDecodesRef.current.add(asset.dataUrl);
              setDecodeTick((tick) => tick + 1);
            }
          };
          decoding.src = asset.dataUrl;
        }
        clear();
        return;
      }

      if (image.naturalWidth !== outputWidth || image.naturalHeight !== outputHeight) {
        // The backend hard-rejects any mask PNG whose size differs from the
        // canvas (video_mask_timeline.py `_decode_png`), so `source_center
        // === target_center` is an invariant there. Pivoting at the asset's
        // own (mismatched) center here would silently produce a confidently
        // wrong preview instead -- refuse and surface the same badge the
        // fallback path uses for its own errors (see H3).
        clear();
        setClientRasterError(true);
        return;
      }
      setClientRasterError(false);

      // This runs on every playhead tick, so the mask is rasterized at the
      // size it will actually be SHOWN at rather than at the output canvas's
      // own resolution -- a 1024x1024 mask displayed in a 400px-wide preview
      // would otherwise cost a full-resolution clear + transformed draw
      // ~30 times a second for pixels that are immediately downscaled away.
      // Never upscales (capped at 1), so this only ever removes work.
      const rasterScale = Math.min(
        1,
        Math.max(
          (displayRect.width * dpr) / outputWidth,
          (displayRect.height * dpr) / outputHeight,
        ),
      );
      const rasterWidth = Math.max(1, Math.round(outputWidth * rasterScale));
      const rasterHeight = Math.max(1, Math.round(outputHeight * rasterScale));

      let offscreen = offscreenRef.current;
      if (!offscreen) {
        offscreen = document.createElement("canvas");
        offscreenRef.current = offscreen;
      }
      if (offscreen.width !== rasterWidth || offscreen.height !== rasterHeight) {
        offscreen.width = rasterWidth;
        offscreen.height = rasterHeight;
      }
      const offscreenCtx = offscreen.getContext("2d");
      if (!offscreenCtx) {
        clear();
        return;
      }
      offscreenCtx.setTransform(1, 0, 0, 1, 0, 0);
      // The backend's own `apply_mask_transform` fills everything outside
      // the transformed source footprint with opaque black (`cval=0.0`), not
      // transparency -- a mask asset PNG is itself fully opaque (see
      // `encodeMaskLayerToPng`), so within the transformed footprint the two
      // paths already agree, but any non-identity transform (scale < 1, a
      // translate) would otherwise leave the OUTSIDE transparent here vs.
      // opaque black on the backend, making the same manifest look
      // materially different depending on which path served it (see M4).
      offscreenCtx.clearRect(0, 0, rasterWidth, rasterHeight);
      offscreenCtx.fillStyle = "#000000";
      offscreenCtx.fillRect(0, 0, rasterWidth, rasterHeight);
      const matrix = maskTransformToCanvasMatrix(
        resolved.transform,
        image.naturalWidth,
        image.naturalHeight,
        outputWidth,
        outputHeight,
      );
      // The mask's own transform is defined in OUTPUT-canvas pixels, so the
      // preview downscale is composed on top of it rather than folded into
      // the transform's own scale factors.
      offscreenCtx.setTransform(
        matrix.a * rasterScale, matrix.b * rasterScale,
        matrix.c * rasterScale, matrix.d * rasterScale,
        matrix.e * rasterScale, matrix.f * rasterScale,
      );
      offscreenCtx.drawImage(image, 0, 0);
      offscreenCtx.setTransform(1, 0, 0, 1, 0, 0);

      canvas.width = Math.max(1, Math.round(containerSize.width * dpr));
      canvas.height = Math.max(1, Math.round(containerSize.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, containerSize.width, containerSize.height);
      ctx.save();
      ctx.globalAlpha = MASK_OVERLAY_ALPHA * Math.max(0, Math.min(1, opacity));
      ctx.globalCompositeOperation = MASK_OVERLAY_CANVAS_COMPOSITE_OPERATION;
      ctx.drawImage(
        offscreen,
        0, 0, rasterWidth, rasterHeight,
        displayRect.x, displayRect.y, displayRect.width, displayRect.height,
      );
      ctx.restore();
      return;
    }

    // sdf fallback: unchanged backend-sprite behavior.
    if (!preview.result || preview.isStale) {
      // `isStale` covers both "the input changed since this sprite was
      // fetched" AND "there is no longer any mask to show at all" (see
      // `useMaskPreview`'s early-return when every keyframe is deleted,
      // which clears `held` so `result` is null and `isStale` is true) --
      // never draw a sprite the hook itself says no longer matches the
      // current mask.
      clear();
      return;
    }

    canvas.width = Math.max(1, Math.round(containerSize.width * dpr));
    canvas.height = Math.max(1, Math.round(containerSize.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, containerSize.width, containerSize.height);

    const { result } = preview;
    const drawTile = (image: HTMLImageElement) => {
      const frames = result.frames;
      if (frames.length === 0) return;
      // Nearest already-fetched sample to the live playhead -- never a fresh
      // request, and never an interpolation between two samples.
      let nearest = frames[0];
      if (currentFrame != null) {
        let bestDistance = Infinity;
        for (const entry of frames) {
          const distance = Math.abs(entry.frame - currentFrame);
          if (distance < bestDistance) {
            bestDistance = distance;
            nearest = entry;
          }
        }
      }
      ctx.save();
      ctx.globalAlpha = MASK_OVERLAY_ALPHA * Math.max(0, Math.min(1, opacity));
      ctx.globalCompositeOperation = MASK_OVERLAY_CANVAS_COMPOSITE_OPERATION;
      ctx.drawImage(
        image,
        nearest.x_offset, 0, result.frame_width, result.frame_height,
        displayRect.x, displayRect.y, displayRect.width, displayRect.height,
      );
      ctx.restore();
    };

    const cached = spriteImageRef.current;
    if (cached && cached.src === result.strip_png) {
      drawTile(cached.image);
    } else {
      const image = new Image();
      image.onload = () => {
        spriteImageRef.current = { src: result.strip_png, image };
        drawTile(image);
      };
      image.src = result.strip_png;
    }
  }, [
    enabled, containerSize, nativeSize, outputWidth, outputHeight,
    clientRasterizable, manifest.keyframes, assets, decodeTick,
    preview.result, preview.isStale, currentFrame, opacity,
  ]);

  if (!enabled) return null;

  return (
    <>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{ width: "100%", height: "100%" }}
      />
      {!clientRasterizable &&
        (preview.isPending || preview.isStale) &&
        (manifest.keyframes.length > 0 || everHadResultRef.current) &&
        !preview.error && (
          <div className="absolute top-1 right-1 pointer-events-none rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-gray-300">
            Updating mask preview…
          </div>
        )}
      {!clientRasterizable && preview.error && (
        <div className="absolute top-1 right-1 pointer-events-none rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-amber-400">
          Mask preview unavailable
        </div>
      )}
      {clientRasterizable && clientRasterError && (
        <div className="absolute top-1 right-1 pointer-events-none rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-amber-400">
          Mask preview unavailable
        </div>
      )}
      {!clientRasterizable && sampleFrames.keyframesOmitted > 0 && (
        <div className="absolute bottom-1 right-1 pointer-events-none rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-amber-400">
          {sampleFrames.keyframesOmitted} keyframe{sampleFrames.keyframesOmitted === 1 ? "" : "s"} not previewed
        </div>
      )}
    </>
  );
}
