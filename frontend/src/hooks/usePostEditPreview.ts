"use client";

import { useEffect, useRef, useState } from "react";
import { flattenChroma } from "@/utils/postEdit";

// Preview-only downscale cap (long side). The flatten pixel pass runs on the
// main thread; capping the preview keeps it to ~100ms. Downloads bake at full
// resolution via applyPostEdit (a separate path), so this cap is preview-only.
const PREVIEW_MAX_LONG_SIDE = 1024;
// Debounce so dragging the flatten slider does not re-run the pixel pass on
// every intermediate value.
const DEBOUNCE_MS = 200;

/**
 * Returns the effective <img> src for a given original image src and color
 * flatten strength (0-100).
 *
 * - flatten <= 0 (or no src): returns the original src unchanged (DOM identical
 *   to today; no processing, no object URLs).
 * - flatten > 0: asynchronously fetches the original (same-origin), decodes,
 *   downscales to PREVIEW_MAX_LONG_SIDE, runs flattenChroma, and returns an
 *   object URL of the result. Until that result is ready (debounce + decode +
 *   pixel pass) it returns the original src, so there is never a stale flatten
 *   applied to the wrong image.
 *
 * Brightness/saturation are intentionally NOT handled here - callers keep those
 * as a CSS filter layered on top of the returned src, so b/s changes never
 * re-run the pixel pass.
 *
 * Caching: only re-runs when (imageSrc, flatten) changes. The previous object
 * URL is revoked on each new result and on unmount.
 */
export function usePostEditPreview(
  imageSrc: string | null | undefined,
  flatten: number
): string | null | undefined {
  const [processed, setProcessed] = useState<{ url: string; key: string } | null>(null);
  const urlRef = useRef<string | null>(null);

  useEffect(() => {
    if (!imageSrc || flatten <= 0) return;
    const key = `${imageSrc}|${flatten}`;
    if (processed && processed.key === key) return; // already have this result

    let cancelled = false;
    const timer = setTimeout(async () => {
      try {
        const resp = await fetch(imageSrc);
        if (!resp.ok) throw new Error(`fetch failed: ${resp.status}`);
        const blob = await resp.blob();

        // Decode + measure. Prefer createImageBitmap; fall back to <img>.
        let drawSource: CanvasImageSource;
        let srcW: number;
        let srcH: number;
        let bmp: ImageBitmap | null = null;
        let fallbackUrl: string | null = null;
        let fallbackImg: HTMLImageElement | null = null;
        if (typeof createImageBitmap === "function") {
          bmp = await createImageBitmap(blob);
          drawSource = bmp;
          srcW = bmp.width;
          srcH = bmp.height;
        } else {
          fallbackUrl = URL.createObjectURL(blob);
          fallbackImg = new Image();
          const img = fallbackImg;
          await new Promise<void>((resolve, reject) => {
            img.onload = () => resolve();
            img.onerror = () => reject(new Error("image decode failed"));
            img.src = fallbackUrl as string;
          });
          drawSource = fallbackImg;
          srcW = fallbackImg.naturalWidth;
          srcH = fallbackImg.naturalHeight;
        }

        const longSide = Math.max(srcW, srcH);
        const scale = longSide > PREVIEW_MAX_LONG_SIDE ? PREVIEW_MAX_LONG_SIDE / longSide : 1;
        const dw = Math.max(1, Math.round(srcW * scale));
        const dh = Math.max(1, Math.round(srcH * scale));

        const canvas = document.createElement("canvas");
        canvas.width = dw;
        canvas.height = dh;
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Failed to get 2D canvas context");
        ctx.drawImage(drawSource, 0, 0, dw, dh);
        if (bmp) bmp.close();
        if (fallbackUrl) URL.revokeObjectURL(fallbackUrl);

        const imgData = ctx.getImageData(0, 0, dw, dh);
        flattenChroma(imgData, flatten);
        ctx.putImageData(imgData, 0, 0);

        const outBlob = await new Promise<Blob | null>((resolve) =>
          canvas.toBlob(resolve, "image/png")
        );
        if (cancelled) return;
        if (!outBlob) throw new Error("canvas.toBlob returned null");

        const objUrl = URL.createObjectURL(outBlob);
        if (cancelled) {
          URL.revokeObjectURL(objUrl);
          return;
        }
        if (urlRef.current) URL.revokeObjectURL(urlRef.current);
        urlRef.current = objUrl;
        setProcessed({ url: objUrl, key });
      } catch (err) {
        // On failure, silently fall back to the original src (returned below).
        // eslint-disable-next-line no-console
        console.error("[usePostEditPreview] flatten preview failed:", err);
      }
    }, DEBOUNCE_MS);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [imageSrc, flatten, processed]);

  // Revoke the last object URL on unmount.
  useEffect(() => {
    return () => {
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current);
        urlRef.current = null;
      }
    };
  }, []);

  if (!imageSrc || flatten <= 0) return imageSrc;
  const key = `${imageSrc}|${flatten}`;
  if (processed && processed.key === key) return processed.url;
  return imageSrc; // fallback until the processed result for this key is ready
}
