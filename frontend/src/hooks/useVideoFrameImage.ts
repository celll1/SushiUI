"use client";

import { useCallback } from "react";
import { grabVideoFrame } from "@/utils/videoFrameGrabber";
import { centerCropToCanvas } from "@/utils/canvasFit";

export interface VideoFrameImageResult {
  imageUrl: string;
}

export interface VideoFrameImageError {
  error: string;
}

/**
 * Grabs a video frame and maps it onto the output canvas via
 * `centerCropToCanvas` (see canvasFit.ts).
 *
 * `grabVideoFrame` reports `exact: false` when a request was superseded by a
 * newer one before its turn and it substituted a nearby cached frame
 * instead of performing the real seek. This retries once against the same
 * time; if the retry also comes back inexact, it returns an error rather
 * than silently showing the wrong frame under a mask meant for a different
 * one.
 */
export function useVideoFrameImage() {
  const requestFrameImage = useCallback(
    async (
      sourceUrl: string,
      timeSec: number,
      canvasWidth: number,
      canvasHeight: number,
      maxGrabWidth?: number,
    ): Promise<VideoFrameImageResult | VideoFrameImageError> => {
      const grabMaxWidth = Math.max(maxGrabWidth ?? 0, canvasWidth, 1024);
      let frameResult = await grabVideoFrame(sourceUrl, timeSec, { maxWidth: grabMaxWidth });
      if (frameResult && !frameResult.exact) {
        frameResult = await grabVideoFrame(sourceUrl, timeSec, { maxWidth: grabMaxWidth });
      }
      if (!frameResult?.dataUrl) {
        return { error: "Could not capture that video frame for mask editing." };
      }
      if (!frameResult.exact) {
        return {
          error: "Could not capture the exact frame for mask editing (the video was still seeking). Try again.",
        };
      }
      try {
        const imageUrl = await centerCropToCanvas(frameResult.dataUrl, canvasWidth, canvasHeight);
        return { imageUrl };
      } catch (error) {
        console.error("[useVideoFrameImage] Failed to map frame onto output canvas:", error);
        return { error: "Could not prepare that video frame for mask editing." };
      }
    },
    [],
  );

  return { requestFrameImage };
}
