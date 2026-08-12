"use client";

import { useEffect, useRef, useState } from "react";
import ImageEditor, { type ImageEditorHandle } from "../common/ImageEditor";
import Button from "../common/Button";
import { useVideoFrameImage } from "@/hooks/useVideoFrameImage";
import {
  clampFrame,
  keyframeAtOrBefore,
  nextKeyframeAfter,
  previousKeyframeBefore,
  type VideoMaskAsset,
  type VideoMaskKeyframe,
} from "@/utils/videoMaskTimeline";

// P4: in-editor frame navigation for the video mask timeline. A shell around
// ImageEditor that keeps one instance mounted for the whole session (so
// zoom/pan/rotation survive a frame change) and drives it via `imageUrl` +
// the exportMask/loadMask/hasUnsavedMaskEdits imperative handle.
//
// State ownership: the keyframe/asset LIST lives in the parent
// (InpaintPanel's videoMaskManifest/videoMaskAssets, undo-tracked there).
// This component only reads them as props. Undo/redo for strokes on the
// currently open frame is ImageEditor's own history, reset per frame by
// loadMask -- i.e. frame-scoped, separate from the manifest-level undo
// stack in InpaintPanel.

interface VideoMaskFrameEditorProps {
  /** Object/file URL of the input clip (same source VideoInpaintTimeline previews from). */
  videoUrl: string;
  /** params.input_trim_start_frames ?? 0 -- converts a trimmed-clip frame number to a raw-clip time. */
  trimStartFrames: number;
  frameRate: number;
  minFrame: number;
  maxFrame: number;
  canvasWidth: number;
  canvasHeight: number;
  /** The frame this editing session was opened for (an "Add at playhead"/"Edit" click). */
  initialFrame: number;
  /** Read-only: the live keyframe list, owned by InpaintPanel. */
  keyframes: VideoMaskKeyframe[];
  /** Read-only: the live asset list, owned by InpaintPanel. */
  assets: VideoMaskAsset[];
  /**
   * Persist a drawn mask for `frame`. Looks up whether a keyframe already
   * exists at that exact frame (fork/new-asset/MAX_MASK_ASSETS handling all
   * live in the parent, alongside the manifest-level undo push) and returns
   * either non-blocking warnings (already saved) or a hard error (nothing
   * was saved -- e.g. the asset cap was hit).
   */
  onSaveFrame: (
    frame: number,
    maskDataUrl: string,
  ) => Promise<
    | { warnings: string[]; keyframes: VideoMaskKeyframe[]; assets: VideoMaskAsset[] }
    | { error: string }
  >;
  onClose: () => void;
}

interface LoadedFrame {
  frame: number;
  imageUrl: string;
  /** The keyframe already saved at exactly this frame, if any. */
  existingKeyframe: VideoMaskKeyframe | null;
  /**
   * True when the mask currently shown was copied from the nearest EARLIER
   * keyframe (keyframeAtOrBefore) rather than one saved for this exact
   * frame -- drives the "not yet a keyframe" banner. Also true (with no
   * mask shown at all) when this frame is before the first keyframe.
   */
  isFallbackMask: boolean;
  /** Whether a fallback keyframe (before this frame) actually existed to copy from. False + isFallbackMask true means this frame is before the first keyframe (blank mask). */
  hasFallbackKeyframe: boolean;
}

// Frame-image cache: avoids re-running `centerCropToCanvas` on every
// back-and-forth between two frames in one editing session. The cached
// value is a PNG data URL (uncompressed-ish, unlike grabVideoFrame's own
// JPEG cache), so the limit is kept small. Cleared whenever the mapping
// inputs change (source/canvas size/frame rate), since a cached entry keyed
// only by frame number would otherwise silently outlive the mapping it was
// rendered for.
const FRAME_IMAGE_CACHE_LIMIT = 10;

export default function VideoMaskFrameEditor({
  videoUrl,
  trimStartFrames,
  frameRate,
  minFrame,
  maxFrame,
  canvasWidth,
  canvasHeight,
  initialFrame,
  keyframes,
  assets,
  onSaveFrame,
  onClose,
}: VideoMaskFrameEditorProps) {
  const editorRef = useRef<ImageEditorHandle>(null);
  const { requestFrameImage } = useVideoFrameImage();

  const [loaded, setLoaded] = useState<LoadedFrame | null>(null);
  // Frozen at the FIRST successful load only: ImageEditor only reads
  // `initialMaskUrl` once, at mount. Later frames' masks are pushed via the
  // `loadMask` imperative handle instead.
  const [initialMaskUrlForMount, setInitialMaskUrlForMount] = useState<string | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);

  // Guards against overlapping navigate() calls (e.g. double-clicking a nav
  // button before the previous frame finished loading) racing each other's
  // async grab/save work.
  const navigatingRef = useRef(false);
  const frameImageCacheRef = useRef<Map<number, string>>(new Map());
  // Retry target for the "could not load a frame" overlay below.
  const lastAttemptedFrameRef = useRef(initialFrame);

  // The mapping from source frame to output canvas is a function of these
  // five inputs (frameRate included: it converts frame -> time in
  // getFrameImage below); if any changes mid-session, previously cached
  // images no longer describe the current mapping.
  useEffect(() => {
    frameImageCacheRef.current.clear();
  }, [videoUrl, trimStartFrames, frameRate, canvasWidth, canvasHeight]);

  // ImageEditor itself sets this flag while mounted (InpaintPanel's
  // Ctrl+Enter queue shortcut checks it to avoid firing while a mask editor
  // is open), but this component shows its own overlay -- not ImageEditor --
  // during the initial frame grab, so the flag must be set independently
  // for the whole lifetime of this component, not just while ImageEditor is
  // mounted.
  useEffect(() => {
    document.body.dataset.imageEditorOpen = "true";
    return () => {
      delete document.body.dataset.imageEditorOpen;
    };
  }, []);

  // Takes `keyframesList`/`assetsList` as parameters rather than closing
  // over the `keyframes`/`assets` props directly: navigate() below needs to
  // resolve against the keyframes/assets a save JUST persisted, which are
  // not yet visible in this component's props (the parent has not
  // re-rendered yet within the same async call).
  const resolveMaskForFrame = (
    frame: number,
    keyframesList: VideoMaskKeyframe[],
    assetsList: VideoMaskAsset[],
  ): { existingKeyframe: VideoMaskKeyframe | null; maskUrl: string | null; isFallback: boolean; hasFallbackKeyframe: boolean } => {
    const existingKeyframe = keyframesList.find((k) => k.frame === frame) ?? null;
    if (existingKeyframe) {
      const asset = assetsList.find((a) => a.id === existingKeyframe.maskId);
      return { existingKeyframe, maskUrl: asset?.dataUrl ?? null, isFallback: false, hasFallbackKeyframe: false };
    }
    const fallback = keyframeAtOrBefore(keyframesList, frame);
    if (!fallback) {
      return { existingKeyframe: null, maskUrl: null, isFallback: true, hasFallbackKeyframe: false };
    }
    const asset = assetsList.find((a) => a.id === fallback.maskId);
    return { existingKeyframe: null, maskUrl: asset?.dataUrl ?? null, isFallback: true, hasFallbackKeyframe: true };
  };

  const getFrameImage = async (frame: number): Promise<{ imageUrl: string } | { error: string }> => {
    const cached = frameImageCacheRef.current.get(frame);
    if (cached) return { imageUrl: cached };
    const targetTimeSec = (trimStartFrames + frame) / frameRate;
    const result = await requestFrameImage(videoUrl, targetTimeSec, canvasWidth, canvasHeight);
    if ("error" in result) return result;
    const cache = frameImageCacheRef.current;
    cache.set(frame, result.imageUrl);
    if (cache.size > FRAME_IMAGE_CACHE_LIMIT) {
      const oldestKey = cache.keys().next().value;
      if (oldestKey !== undefined) cache.delete(oldestKey);
    }
    return { imageUrl: result.imageUrl };
  };

  /**
   * Move to `targetFrame`: auto-save the CURRENTLY open frame first (only if
   * it actually has unsaved strokes -- see ImageEditor's hasUnsavedMaskEdits
   * doc comment for why this is a flag, not a PNG diff), then load the new
   * frame's base image and mask.
   */
  const navigate = async (targetFrame: number) => {
    if (navigatingRef.current) return;
    navigatingRef.current = true;
    setIsBusy(true);
    setError(null);
    try {
      const clamped = clampFrame(targetFrame, minFrame, maxFrame);
      lastAttemptedFrameRef.current = clamped;

      // Resolve against these unless a save below returns fresher ones.
      let keyframesForResolve = keyframes;
      let assetsForResolve = assets;

      if (loaded && editorRef.current?.hasUnsavedMaskEdits()) {
        const maskDataUrl = editorRef.current.exportMask();
        if (maskDataUrl) {
          const result = await onSaveFrame(loaded.frame, maskDataUrl);
          if ("error" in result) {
            // Do not navigate away from a frame whose edits failed to save
            // (e.g. the asset cap was hit) -- the user would otherwise lose
            // track of which frame still holds the unsaved stroke.
            setError(result.error);
            return;
          }
          if (result.warnings.length > 0) setError(result.warnings.join(" "));
          // The keyframes/assets props on THIS render still predate the
          // save above (the parent has not re-rendered yet), so resolving
          // the frame we are moving to against them would see a stale
          // world -- use what the save just confirmed instead.
          keyframesForResolve = result.keyframes;
          assetsForResolve = result.assets;
          // Persisted successfully: the pixels just saved are now the
          // record of truth for this frame, so the editor's own dirty flag
          // (strokes since the last loadMask) must be cleared here too --
          // otherwise every later navigate() re-saves this frame again
          // (harmless but wasteful) and keeps growing the undo stack.
          await editorRef.current.loadMask(maskDataUrl);
        }
      }

      if (loaded && clamped === loaded.frame) {
        // Re-saved in place (or nothing to save); nothing else to do.
        return;
      }

      const imageResult = await getFrameImage(clamped);
      if ("error" in imageResult) {
        setError(imageResult.error);
        return;
      }
      const { existingKeyframe, maskUrl, isFallback, hasFallbackKeyframe } = resolveMaskForFrame(
        clamped,
        keyframesForResolve,
        assetsForResolve,
      );
      const nextLoaded: LoadedFrame = {
        frame: clamped,
        imageUrl: imageResult.imageUrl,
        existingKeyframe,
        isFallbackMask: isFallback,
        hasFallbackKeyframe,
      };
      if (loaded === null) {
        // First load for this session: seed ImageEditor's initialMaskUrl at
        // mount time instead of calling loadMask (which requires the editor
        // to already be mounted).
        setInitialMaskUrlForMount(maskUrl ?? undefined);
        setLoaded(nextLoaded);
      } else {
        setLoaded(nextLoaded);
        const maskLoaded = await editorRef.current?.loadMask(maskUrl);
        if (maskLoaded === false) {
          // The layer was cleared rather than left holding the previous
          // frame's pixels, so this frame now starts blank; say so instead
          // of letting the banner claim an earlier mask was carried over.
          setError("Could not load this frame's saved mask. The mask layer was cleared.");
        }
      }
    } finally {
      navigatingRef.current = false;
      setIsBusy(false);
    }
  };

  // Kick off the session's first frame load.
  useEffect(() => {
    void navigate(initialFrame);
    // Intentionally run once per mount; initialFrame is a session-open
    // parameter, not something that re-triggers this effect if the parent
    // happens to re-render with a new value for an unrelated reason.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const goToPreviousKeyframe = () => {
    if (!loaded) return;
    const target = previousKeyframeBefore(keyframes, loaded.frame);
    if (target) void navigate(target.frame);
  };
  const goToNextKeyframe = () => {
    if (!loaded) return;
    const target = nextKeyframeAfter(keyframes, loaded.frame);
    if (target) void navigate(target.frame);
  };
  const stepFrame = (delta: number) => {
    if (!loaded) return;
    void navigate(loaded.frame + delta);
  };

  const handleClose = () => {
    // Cancel discards the CURRENTLY open frame's unsaved stroke, matching
    // the static (single-frame) mask editor's existing Cancel behavior --
    // every frame already navigated away from was already auto-saved.
    onClose();
  };

  const handleSaveAndClose = async (maskDataUrl: string) => {
    if (!loaded) return;
    const result = await onSaveFrame(loaded.frame, maskDataUrl);
    if ("error" in result) {
      setError(result.error);
      return;
    }
    if (result.warnings.length > 0) setError(result.warnings.join(" "));
    onClose();
  };

  const hasPrevKeyframe = loaded ? previousKeyframeBefore(keyframes, loaded.frame) !== null : false;
  const hasNextKeyframe = loaded ? nextKeyframeAfter(keyframes, loaded.frame) !== null : false;

  const auxiliaryControls = loaded ? (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-gray-300">Video Frame</h3>
      <div className="text-xs text-gray-400">
        Frame {loaded.frame} / {maxFrame}
        {loaded.existingKeyframe ? " (keyframe)" : ""}
      </div>
      <div className="flex gap-2">
        <Button
          onClick={goToPreviousKeyframe}
          disabled={isBusy || !hasPrevKeyframe}
          variant="secondary"
          size="sm"
          className="flex-1"
          title="Jump to the previous mask keyframe"
        >
          ⏮ Prev KF
        </Button>
        <Button
          onClick={goToNextKeyframe}
          disabled={isBusy || !hasNextKeyframe}
          variant="secondary"
          size="sm"
          className="flex-1"
          title="Jump to the next mask keyframe"
        >
          Next KF ⏭
        </Button>
      </div>
      <div className="flex gap-2">
        <Button
          onClick={() => stepFrame(-1)}
          disabled={isBusy || loaded.frame <= minFrame}
          variant="secondary"
          size="sm"
          className="flex-1"
        >
          ◀ -1
        </Button>
        <Button
          onClick={() => stepFrame(1)}
          disabled={isBusy || loaded.frame >= maxFrame}
          variant="secondary"
          size="sm"
          className="flex-1"
        >
          +1 ▶
        </Button>
      </div>
      {loaded.isFallbackMask && (
        <p className="text-xs text-amber-400">
          {loaded.hasFallbackKeyframe
            ? "This frame has no keyframe yet; it is showing the nearest earlier keyframe's mask. Drawing here creates a new keyframe starting from that mask."
            : "This frame has no keyframe yet and there is no earlier keyframe to copy from. Drawing here creates a new keyframe starting from a blank mask."}
        </p>
      )}
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  ) : null;

  // Escape/backdrop-click close for the "not loaded yet" overlay below --
  // ImageEditor is not mounted yet, so its own keyboard/close handling
  // cannot cover this window.
  useEffect(() => {
    if (loaded) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [loaded, onClose]);

  if (!loaded) {
    return (
      <div
        className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center"
        onClick={onClose}
      >
        <div
          className="bg-gray-800 rounded-lg p-6 text-center space-y-4 max-w-sm"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="text-white text-sm">
            {error ?? "Preparing video frame..."}
          </div>
          <div className="flex gap-2 justify-center">
            {error && (
              <Button
                onClick={() => void navigate(lastAttemptedFrameRef.current)}
                variant="secondary"
                size="sm"
              >
                Retry
              </Button>
            )}
            <Button onClick={onClose} variant="secondary" size="sm">
              Cancel
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <ImageEditor
      ref={editorRef}
      imageUrl={loaded.imageUrl}
      onSave={() => undefined}
      onClose={handleClose}
      onSaveMask={handleSaveAndClose}
      mode="inpaint"
      initialMaskUrl={initialMaskUrlForMount}
      auxiliaryControls={auxiliaryControls}
    />
  );
}
