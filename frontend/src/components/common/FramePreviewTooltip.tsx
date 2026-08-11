"use client";

import { useEffect, useState } from "react";
import { grabVideoFrame } from "@/utils/videoFrameGrabber";

export interface FramePreviewTooltipProps {
  /** Object/file URL of the video to grab a frame from. Null hides the tooltip -- there is no video to preview (e.g. an audio-only timeline mount, or no clip loaded yet). */
  videoSrc: string | null;
  /** Time to preview, in seconds, within `videoSrc`'s own timeline. */
  timeSec: number;
  /** Horizontal position within the positioning container (a `relative` ancestor), 0-100. */
  leftPercent: number;
  /** Text drawn under the thumbnail, e.g. "frame 42 · 00:00:01:18". */
  label: string;
  visible: boolean;
}

/**
 * A floating thumbnail that follows the pointer along a timeline track,
 * showing the input clip's frame at the hovered/dragged position. Distinct
 * from `InlineHelp` (an anchored `<details>` popover for static text) --
 * this tracks a moving pointer position and renders an async-loaded image.
 *
 * Host this in a `relative` wrapper that is NOT `overflow-hidden` (the
 * thumbnail sits above the track, so an ancestor that clips the track's own
 * contents would clip the tooltip away too) and whose width matches the
 * element `leftPercent` is measured against. Horizontal placement is clamped
 * to the wrapper's own bounds so it stays fully on screen even when
 * `leftPercent` is 0 or 100.
 */
export default function FramePreviewTooltip({ videoSrc, timeSec, leftPercent, label, visible }: FramePreviewTooltipProps) {
  // The image and its label are only ever updated together, from an `exact`
  // grab -- a superseded grab can resolve with a nearby-but-different
  // frame, and showing that next to the current pointer's label would
  // caption the wrong image. A non-exact result just leaves the last
  // confirmed pair on screen.
  const [confirmed, setConfirmed] = useState<{ dataUrl: string; label: string } | null>(null);

  useEffect(() => {
    setConfirmed(null);
  }, [videoSrc]);

  useEffect(() => {
    if (!visible || !videoSrc) return;
    let cancelled = false;
    const requestedLabel = label;
    grabVideoFrame(videoSrc, timeSec).then((result) => {
      if (cancelled || !result || !result.exact) return;
      setConfirmed({ dataUrl: result.dataUrl, label: requestedLabel });
    });
    return () => {
      cancelled = true;
    };
  }, [visible, videoSrc, timeSec, label]);

  if (!visible || !videoSrc) return null;

  return (
    <div
      className="pointer-events-none absolute -top-[5.5rem] z-50 flex flex-col items-center"
      style={{ left: `clamp(3.5rem, ${leftPercent}%, calc(100% - 3.5rem))`, transform: "translateX(-50%)" }}
    >
      <div className="h-16 w-28 rounded border border-gray-600 bg-gray-900 overflow-hidden flex items-center justify-center">
        {confirmed ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={confirmed.dataUrl} alt="" className="h-full w-full object-cover" />
        ) : (
          <span className="text-[10px] text-gray-500">…</span>
        )}
      </div>
      <span className="mt-0.5 text-[10px] leading-3 text-gray-200 bg-gray-900/90 px-1 rounded whitespace-nowrap">
        {confirmed ? confirmed.label : label}
      </span>
    </div>
  );
}
