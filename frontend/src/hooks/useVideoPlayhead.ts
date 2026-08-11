"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type VideoWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (callback: (now: number, metadata: { mediaTime: number }) => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

export interface VideoPlayheadState {
  /** The attached video's current playback position, in seconds. Null until it has one (no video attached, or it has not loaded metadata yet). */
  currentTimeSec: number | null;
  /** currentTimeSec expressed in whole frames at the caller's frameRate. Null under the same conditions, or when frameRate <= 0. */
  currentFrame: number | null;
  isPlaying: boolean;
  /** Whether a loop range is currently armed (see `setLoopRange`). */
  isLooping: boolean;
  /** Seeks the attached video to an exact second. No-op if nothing is attached. */
  seekToSeconds: (seconds: number) => void;
  /** Seeks the attached video to the given frame at the hook's frameRate. No-op if nothing is attached or frameRate <= 0. */
  seekToFrame: (frame: number) => void;
  play: () => void;
  pause: () => void;
  /**
   * Loops playback within [startSec, endSec): once attached, playback past
   * `endSec` seeks back to `startSec`. `endSec` is clamped to the video's
   * own duration, so a caller-computed end past the true length still
   * re-triggers instead of silently never looping. Pass null to clear it.
   * Does not itself start playback -- call `play()` too. Automatically
   * cleared when the attached source changes or the hook unmounts.
   */
  setLoopRange: (range: { startSec: number; endSec: number } | null) => void;
}

/**
 * Tracks a `<video>` element's live playhead and exposes seek/play/loop
 * controls, so a timeline can draw a position synced with the SAME player
 * the panel already renders for preview, and can scrub it.
 *
 * `videoRef` must point at the panel's own preview `<video>` (this hook adds
 * listeners only; it does not create an element and does not touch the
 * preview-URL lifecycle the panel already owns). `attachKey` should be the
 * value that changes whenever the panel swaps in a new clip (its preview
 * object-URL is the natural choice) -- refs are not reactive, so without an
 * explicit key the hook would not know to re-attach after a new clip loads
 * into the same `<video>` node.
 *
 * While playing, the reported position is synced via
 * `requestVideoFrameCallback` (falling back to `requestAnimationFrame` where
 * unsupported) rather than the coarser `timeupdate` event, so frame-exact
 * reads (e.g. an "at the playhead" button) are not off by several frames.
 * React state only updates when the whole-frame value actually changes, so
 * this does not re-render on every callback -- at most once per real frame.
 */
export function useVideoPlayhead(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  frameRate: number,
  attachKey: string | null
): VideoPlayheadState {
  const [currentTimeSec, setCurrentTimeSec] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const loopRangeRef = useRef<{ startSec: number; endSec: number } | null>(null);
  const lastFrameRef = useRef<number | null>(null);

  const commitTime = useCallback((t: number, force = false) => {
    const fps = frameRate > 0 ? frameRate : null;
    const frame = fps ? Math.round(t * fps) : null;
    if (!force && fps != null && lastFrameRef.current === frame) return;
    lastFrameRef.current = frame;
    setCurrentTimeSec(t);
  }, [frameRate]);

  useEffect(() => {
    // Attaching a new source (or detaching) invalidates any loop armed for
    // whatever was previously attached.
    loopRangeRef.current = null;
    setIsLooping(false);
    lastFrameRef.current = null;

    const video = videoRef.current;
    if (!video || !attachKey) {
      setCurrentTimeSec(null);
      setIsPlaying(false);
      return;
    }

    const rvfcVideo = video as VideoWithFrameCallback;
    const supportsRvfc = typeof rvfcVideo.requestVideoFrameCallback === "function";
    let frameHandle: number | null = null;

    const scheduleFrameSync = () => {
      if (supportsRvfc) {
        frameHandle = rvfcVideo.requestVideoFrameCallback!((_now, metadata) => {
          commitTime(metadata.mediaTime);
          if (!video.paused && !video.ended) scheduleFrameSync();
        });
      } else {
        frameHandle = requestAnimationFrame(() => {
          commitTime(video.currentTime);
          if (!video.paused && !video.ended) scheduleFrameSync();
        });
      }
    };
    const cancelFrameSync = () => {
      if (frameHandle == null) return;
      if (supportsRvfc) rvfcVideo.cancelVideoFrameCallback?.(frameHandle);
      else cancelAnimationFrame(frameHandle);
      frameHandle = null;
    };

    // Drives the loop clamp (does not need frame-callback precision) and
    // acts as the position source while paused/buffering, when no frame
    // callback is scheduled.
    const handleTimeUpdate = () => {
      commitTime(video.currentTime);
      const loop = loopRangeRef.current;
      if (loop && video.currentTime >= loop.endSec - 0.001) {
        video.currentTime = loop.startSec;
      }
    };
    // A user-initiated seek should always report the exact landed position,
    // never deduped against the last frame, and must never itself clamp
    // into the loop range -- otherwise a loop-enabled user could not scrub
    // past the selection with the native controls at all.
    const syncTime = () => commitTime(video.currentTime, true);
    const onPlay = () => {
      setIsPlaying(true);
      scheduleFrameSync();
    };
    const onPauseOrEnded = () => {
      setIsPlaying(!video.paused && !video.ended);
      cancelFrameSync();
    };

    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("seeking", syncTime);
    video.addEventListener("seeked", syncTime);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPauseOrEnded);
    video.addEventListener("ended", onPauseOrEnded);
    if (Number.isFinite(video.currentTime)) syncTime();
    if (!video.paused) scheduleFrameSync();

    return () => {
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("seeking", syncTime);
      video.removeEventListener("seeked", syncTime);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPauseOrEnded);
      video.removeEventListener("ended", onPauseOrEnded);
      cancelFrameSync();
      loopRangeRef.current = null;
      setIsLooping(false);
    };
  }, [videoRef, attachKey, commitTime]);

  const seekToSeconds = useCallback((seconds: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, seconds);
  }, [videoRef]);

  const seekToFrame = useCallback((frame: number) => {
    if (frameRate > 0) seekToSeconds(frame / frameRate);
  }, [frameRate, seekToSeconds]);

  const play = useCallback(() => {
    videoRef.current?.play().catch(() => {
      // Autoplay can be refused by the browser (e.g. no user gesture yet);
      // the loop toggle just stays paused rather than throwing.
    });
  }, [videoRef]);

  const pause = useCallback(() => {
    videoRef.current?.pause();
  }, [videoRef]);

  const setLoopRange = useCallback((range: { startSec: number; endSec: number } | null) => {
    if (!range) {
      loopRangeRef.current = null;
      setIsLooping(false);
      return;
    }
    const video = videoRef.current;
    const duration = video && Number.isFinite(video.duration) ? video.duration : null;
    const endSec = duration != null ? Math.min(range.endSec, duration) : range.endSec;
    loopRangeRef.current = { startSec: range.startSec, endSec };
    setIsLooping(true);
  }, [videoRef]);

  const safeFps = frameRate > 0 ? frameRate : null;
  const currentFrame = currentTimeSec != null && safeFps ? Math.round(currentTimeSec * safeFps) : null;

  return { currentTimeSec, currentFrame, isPlaying, isLooping, seekToSeconds, seekToFrame, play, pause, setLoopRange };
}
