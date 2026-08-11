"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useMaskPreview } from "@/hooks/useMaskPreview";
import { computeCoverCropDisplayRect } from "@/utils/canvasFit";
import {
  MASK_OVERLAY_ALPHA,
  MASK_OVERLAY_CANVAS_COMPOSITE_OPERATION,
} from "@/utils/maskConventions";
import {
  computeMaskPreviewSampleFrames,
  type VideoMaskAsset,
  type VideoMaskManifest,
} from "@/utils/videoMaskTimeline";

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
  /** The regenerate range (pixel frames of the trimmed clip), used only to choose sample frames -- see computeMaskPreviewSampleFrames. */
  rangeStart: number;
  rangeEnd: number;
  /** The live playhead frame (from useVideoPlayhead), or null. Used only to pick the nearest already-fetched sample to draw -- never sent as a fresh request by itself. */
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
 * Rasterization happens ONLY on the backend (`useMaskPreview` /
 * `/video-mask/preview`); this component never reimplements `hold`/`affine`/
 * `sdf` interpolation. What it draws is always an EXACT backend rasterization
 * of one of a small set of sampled frames (`computeMaskPreviewSampleFrames`),
 * picking whichever sampled frame is nearest the live playhead rather than
 * refetching on every animation frame -- see the design note on
 * `useMaskPreview` for why. Because the preview is fetched at a downscaled
 * resolution (`PREVIEW_MAX_SIZE`, well below the token granularity a real
 * generation call resolves at), this must not be read as a claim about the
 * exact per-pixel boundary that call would produce.
 */
export default function VideoMaskPreviewOverlay({
  videoRef,
  nativeSize,
  outputWidth,
  outputHeight,
  manifest,
  assets,
  rangeStart,
  rangeEnd,
  currentFrame,
  enabled,
  opacity,
}: VideoMaskPreviewOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [containerSize, setContainerSize] = useState<{ width: number; height: number } | null>(null);
  const spriteImageRef = useRef<{ src: string; image: HTMLImageElement } | null>(null);

  const keyframeFrames = useMemo(
    () => manifest.keyframes.map((keyframe) => keyframe.frame),
    [manifest.keyframes],
  );
  const sampleFrames = useMemo(
    () => computeMaskPreviewSampleFrames(keyframeFrames, rangeStart, rangeEnd),
    [keyframeFrames, rangeStart, rangeEnd],
  );

  const preview = useMaskPreview(
    manifest,
    assets,
    enabled ? sampleFrames : [],
    PREVIEW_MAX_SIZE,
  );

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

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const clear = () => {
      canvas.width = 0;
      canvas.height = 0;
    };

    if (!enabled || !containerSize || !nativeSize || !preview.result) {
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
      return;
    }

    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
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
  }, [enabled, containerSize, nativeSize, outputWidth, outputHeight, preview.result, currentFrame, opacity]);

  if (!enabled) return null;

  return (
    <>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{ width: "100%", height: "100%" }}
      />
      {(preview.isPending || preview.isStale) && manifest.keyframes.length > 0 && (
        <div className="absolute top-1 right-1 pointer-events-none rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-gray-300">
          Updating mask preview…
        </div>
      )}
      {preview.error && (
        <div className="absolute top-1 right-1 pointer-events-none rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-amber-400">
          Mask preview unavailable
        </div>
      )}
    </>
  );
}
