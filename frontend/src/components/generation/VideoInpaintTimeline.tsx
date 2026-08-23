"use client";

import { useEffect, useRef, useState } from "react";
import Button from "../common/Button";
import NumberInput from "../common/NumberInput";
import InlineHelp from "../common/InlineHelp";
import Select from "../common/Select";
import FramePreviewTooltip from "../common/FramePreviewTooltip";
import Timeline, { useTimelineContext, type TimelineDomain } from "../timeline/Timeline";
import { cn } from "@/lib/utils";
import { latentGroupSpans, snapRangeToLatentGroups } from "@/utils/api";
import { formatFrameLabel, formatTimecode } from "@/utils/timecode";
import {
  clampFrame,
  DEFAULT_MASK_INTERPOLATION,
  MAX_COMPOSITE_FEATHER_PX,
  MAX_MASK_KEYFRAMES,
  MAX_MASK_SCALE,
  MIN_MASK_SCALE,
  removeKeyframe,
  sortKeyframes,
  upsertKeyframe,
  type MaskInterpolation,
  type MaskTransform,
  type VideoMaskAsset,
  type VideoMaskKeyframe,
} from "@/utils/videoMaskTimeline";
import type { VideoPlayheadState } from "@/hooks/useVideoPlayhead";

// ---------------------------------------------------------------------------
// The SINGLE video-inpaint timeline for POST /generate/inpaint/video: one
// shared ruler/playhead (via `../timeline/Timeline`) with two stacked
// tracks, replacing the previously separate VideoInpaintRangeTimeline and
// VideoInpaintMaskTimeline (each of which drew its own ruler/playhead on
// the same time axis).
//
//   Track 1 "Regenerate range": two-handle drag, latent-group-boundary
//     snapping, hover/drag frame-thumbnail preview -- ported unchanged from
//     VideoInpaintRangeTimeline.
//   Track 2 "Mask keyframes": keyframe markers (now including ones outside
//     the current regenerate range, previously hidden), each with a saved-
//     asset thumbnail, drag-to-move or numeric frame reassignment,
//     interpolation/transform editing, and a composite feather control.
//
// State ownership is unchanged from the two predecessor components: this
// component is fully controlled by InpaintPanel, which keeps
// `videoMaskManifest`/`videoMaskAssets`/`videoMaskEditor`/`videoMaskError`
// as its own state. Undo/redo (`canUndo`/`canRedo`/`onUndo`/`onRedo`) is
// ALSO owned by InpaintPanel, not here: InpaintPanel is the only place that
// sees every way the keyframe/asset list can change -- this component's own
// controls (duplicate, delete, transform, interpolation, frame move,
// composite feather) AND drawing a new mask (which this component never
// sees; InpaintPanel opens ImageEditor and commits that save directly). A
// history stack that only covered the first group would let undo replace
// the keyframe list wholesale with a snapshot that predates a mask added by
// drawing, silently deleting that keyframe AND its saved PNG on undo.
// ---------------------------------------------------------------------------

const interpolationOptions: Array<{ value: MaskInterpolation; label: string }> = [
  { value: "hold", label: "Hold" },
  { value: "affine", label: "Affine" },
  { value: "sdf", label: "SDF morph" },
];

const transformFields: Array<{
  key: keyof MaskTransform;
  label: string;
  step: string;
  min?: string;
  max?: string;
}> = [
  { key: "x", label: "X", step: "1" },
  { key: "y", label: "Y", step: "1" },
  // step "any", not 0.05: a numeric step is the browser's validity grid
  // (min + n*step), and MIN_MASK_SCALE=0.01 puts identity scale 1.0 off it,
  // so the spinner rewrote a typed 1.0 to 1.01.
  { key: "scaleX", label: "Scale X", step: "any", min: String(MIN_MASK_SCALE), max: String(MAX_MASK_SCALE) },
  { key: "scaleY", label: "Scale Y", step: "any", min: String(MIN_MASK_SCALE), max: String(MAX_MASK_SCALE) },
  { key: "rotation", label: "Rotation (degrees)", step: "1" },
];

// Above this many latent groups, per-boundary hairlines are decimated so the
// track does not turn into an undifferentiated grey wall on a long clip.
const DENSE_GROUP_THRESHOLD = 60;

export interface VideoInpaintTimelineProps {
  // ---- Shared clip geometry / Regenerate range track --------------------
  /** Frames of the uploaded clip before trim. */
  rawFrames: number;
  trimStart: number;
  trimEnd: number;
  /** Pixel frames per latent frame, cycled (empty = arch declares no chunking). */
  latentChunkPattern: number[];
  start: number;
  end: number;
  onRangeChange: (start: number, end: number) => void;
  frameRate: number;
  /** Gates the Regenerate Range track and the shared seek surface (ruler/track background). */
  disabled?: boolean;
  /** Object/file URL of the uploaded clip, for hover/drag frame previews and seek-on-click. Null/omitted = no preview, no playhead, no seek. */
  videoSrc?: string | null;
  /** The SAME input <video>'s live playhead (from `useVideoPlayhead`), in RAW clip frames/seconds. Omitted = degrades the same as videoSrc absent. */
  player?: VideoPlayheadState;

  // ---- Mask keyframes track ----------------------------------------------
  keyframes: VideoMaskKeyframe[];
  onChange: (keyframes: VideoMaskKeyframe[]) => void;
  onEditKeyframe: (keyframe: VideoMaskKeyframe) => void;
  /** The parent owns mask creation; this callback receives the requested frame (trimmed-clip space). */
  onAddKeyframe: (frame: number) => void;
  compositeFeatherPx: number;
  onCompositeFeatherPxChange: (value: number) => void;
  /** Saved mask assets, for keyframe thumbnails. */
  assets: VideoMaskAsset[];
  /**
   * Extra gate for the Mask track ONLY, ORed with `disabled` -- the old
   * VideoInpaintMaskTimeline was disabled by isGenerating/invalid-clip-
   * length conditions the Range track never gated on; kept as a separate
   * prop rather than folding into `disabled` so the Range track's own gate
   * does not change.
   */
  maskDisabled?: boolean;
  /**
   * Why `maskDisabled` is set, shown next to the inert controls. Omitted
   * when the reason is self-evident (a generation running).
   */
  maskDisabledReason?: string;
  /**
   * Undo/redo for the keyframe/asset manifest, owned by InpaintPanel (see
   * the module comment above for why). This component only renders the
   * buttons and their enabled state.
   */
  canUndo?: boolean;
  canRedo?: boolean;
  onUndo?: () => void;
  onRedo?: () => void;
}

function frameDescription(frame: number): string {
  return `Frame ${frame}`;
}

// ---------------------------------------------------------------------------
// Mode-coloured mask spans, drawn under the diamond markers in `MaskTrack`.
// Mirrors the per-frame governing logic of the backend's
// `rasterize_mask_timeline` (backend/core/inference/video_mask_timeline.py,
// the `for frame in range(start_frame, end_frame)` loop): a frame before the
// first keyframe holds that keyframe's mask, a frame at/after the last
// keyframe holds ITS mask, and every frame in between belongs to the segment
// [L.frame, R.frame) whose character is `L.interpolationToNext`. Segments
// are clipped to [rangeStart, rangeEnd) -- the same latent-group-snapped
// span the "Inpaint range" band already highlights -- so a keyframe sitting
// outside that span still contributes a correctly-bounded terminal segment
// (it governs frames right up to the edge of the visible range) rather than
// a bar drawn off the track or spanning the wrong frames.
// ---------------------------------------------------------------------------

interface MaskSegment {
  key: string;
  /** Trimmed-clip frame, inclusive. */
  start: number;
  /** Trimmed-clip frame, exclusive. */
  end: number;
  mode: MaskInterpolation;
  title: string;
}

const MASK_SEGMENT_MODE_LABEL: Record<MaskInterpolation, string> = {
  hold: "Hold",
  affine: "Affine",
  sdf: "SDF morph",
};

function computeMaskSegments(sortedKeyframes: VideoMaskKeyframe[], rangeStart: number, rangeEnd: number): MaskSegment[] {
  if (rangeEnd <= rangeStart || sortedKeyframes.length === 0) return [];

  // Defensive de-duplication by frame: `moveKeyframe` above already rejects
  // a frame collision in the committed manifest, but this function must not
  // produce a zero-width or overlapping pair of segments if it is ever
  // handed a transient duplicate. Mirrors the backend's tie-break
  // (`rasterize_mask_timeline`'s `while frame_numbers[right] <= frame`,
  // backend/core/inference/video_mask_timeline.py): when several keyframes
  // share a frame, that loop advances past all of them, so the LAST one in
  // sorted order ends up governing -- keep the last occurrence here too,
  // not the first.
  const unique: VideoMaskKeyframe[] = [];
  const seenFrames = new Map<number, number>();
  for (const keyframe of sortedKeyframes) {
    const existingIndex = seenFrames.get(keyframe.frame);
    if (existingIndex !== undefined) {
      unique[existingIndex] = keyframe;
      continue;
    }
    seenFrames.set(keyframe.frame, unique.length);
    unique.push(keyframe);
  }

  const segments: MaskSegment[] = [];
  const first = unique[0];
  const last = unique[unique.length - 1];

  if (rangeStart < first.frame) {
    const start = rangeStart;
    const end = Math.min(rangeEnd, first.frame);
    if (end > start) {
      segments.push({
        key: `before-${first.id}`,
        start,
        end,
        mode: "hold",
        title: `Hold: frame ${first.frame}'s mask, held back over frames [${start}, ${end})`,
      });
    }
  }

  for (let i = 0; i < unique.length - 1; i += 1) {
    const left = unique[i];
    const right = unique[i + 1];
    const start = Math.max(rangeStart, left.frame);
    const end = Math.min(rangeEnd, right.frame);
    if (end <= start) continue;
    const mode: MaskInterpolation = left.interpolationToNext || DEFAULT_MASK_INTERPOLATION;
    segments.push({
      key: `${left.id}-${right.id}`,
      start,
      end,
      mode,
      title: `${MASK_SEGMENT_MODE_LABEL[mode]}: frame ${left.frame} to frame ${right.frame}, frames [${start}, ${end})`,
    });
  }

  if (rangeEnd > last.frame) {
    const start = Math.max(rangeStart, last.frame);
    const end = rangeEnd;
    if (end > start) {
      segments.push({
        key: `after-${last.id}`,
        start,
        end,
        mode: "hold",
        title: `Hold: frame ${last.frame}'s mask, held forward over frames [${start}, ${end})`,
      });
    }
  }

  return segments;
}

// Colour + border-style pair per mode, so the encoding is not colour-alone.
// Teal/sky/fuchsia are chosen to stay
// clear of this file's other timeline colours: amber (the "Inpaint range"
// band), violet (the diamond markers/selection), and emerald (the shared
// `Timeline` playhead).
const MASK_SEGMENT_STYLE: Record<MaskInterpolation, string> = {
  hold: "border-teal-400/70 bg-teal-500/35",
  affine: "border-sky-400/70 bg-sky-500/35 border-dashed",
  sdf: "border-fuchsia-400/70 bg-fuchsia-500/35 border-dotted",
};

function uniqueCopyId(source: VideoMaskKeyframe, keyframes: VideoMaskKeyframe[]): string {
  const ids = new Set(keyframes.map((keyframe) => keyframe.id));
  const base = `${source.id}-copy`;
  let candidate = base;
  let suffix = 2;
  while (ids.has(candidate)) candidate = `${base}-${suffix++}`;
  return candidate;
}

function findFreeFrame(
  source: VideoMaskKeyframe,
  keyframes: VideoMaskKeyframe[],
  minFrame: number,
  maxFrame: number,
): number | null {
  const occupied = new Set(keyframes.map((keyframe) => keyframe.frame));
  occupied.delete(source.frame);
  const sourceFrame = clampFrame(source.frame, minFrame, maxFrame);
  for (let distance = 1; distance <= maxFrame - minFrame; distance += 1) {
    const candidates = [sourceFrame + distance, sourceFrame - distance];
    for (const candidate of candidates) {
      if (candidate >= minFrame && candidate <= maxFrame && !occupied.has(candidate)) return candidate;
    }
  }
  return null;
}

export default function VideoInpaintTimeline({
  rawFrames,
  trimStart,
  trimEnd,
  latentChunkPattern,
  start,
  end,
  onRangeChange,
  frameRate,
  disabled = false,
  videoSrc = null,
  player,
  keyframes,
  onChange,
  onEditKeyframe,
  onAddKeyframe,
  compositeFeatherPx,
  onCompositeFeatherPxChange,
  assets,
  maskDisabled = false,
  maskDisabledReason,
  canUndo = false,
  canRedo = false,
  onUndo,
  onRedo,
}: VideoInpaintTimelineProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const safeRaw = Math.max(1, rawFrames);
  const trimmedFrames = Math.max(0, rawFrames - trimStart - trimEnd);
  const lastFrame = Math.max(0, trimmedFrames - 1);
  const groups = latentGroupSpans(latentChunkPattern, trimmedFrames);
  const bounds = groups.length ? [0, ...groups.map(([, hi]) => hi)] : [];
  const effective = snapRangeToLatentGroups(groups, start, end);
  const selectedGroups = groups.filter(([lo, hi]) => lo < effective.end && hi > effective.start).length;
  const rangeStart = clampFrame(effective.start, 0, lastFrame);
  const rangeEnd = clampFrame(effective.end, 0, trimmedFrames);

  const nearestBound = (frame: number): number =>
    bounds.reduce((best, b) => (Math.abs(frame - b) < Math.abs(frame - best) ? b : best), bounds[0]);
  const lastBound = bounds.length ? bounds[bounds.length - 1] : 0;
  const wholeClip = (s: number, e: number) => bounds.length > 2 && s === 0 && e === lastBound;
  const commitStart = (raw: number) => {
    if (!bounds.length) return;
    let s = Math.min(nearestBound(raw), bounds[bounds.length - 2]);
    const e = Math.max(effective.end, bounds[bounds.indexOf(s) + 1]);
    if (wholeClip(s, e)) s = bounds[1];
    onRangeChange(s, e);
  };
  const commitEnd = (raw: number) => {
    if (!bounds.length) return;
    let e = Math.max(nearestBound(raw), bounds[1]);
    const s = Math.min(effective.start, bounds[bounds.indexOf(e) - 1]);
    if (wholeClip(s, e)) e = bounds[bounds.length - 2];
    onRangeChange(s, e);
  };
  const adjacentBound = (frame: number, dir: -1 | 1, stride: number): number => {
    if (!bounds.length) return frame;
    const idx = bounds.indexOf(nearestBound(frame));
    const next = bounds[Math.max(0, Math.min(bounds.length - 1, idx + dir * stride))];
    return next ?? frame;
  };
  const toggleLoop = () => {
    if (!player) return;
    if (player.isLooping) {
      player.setLoopRange(null);
      return;
    }
    const fps = frameRate > 0 ? frameRate : 24;
    player.setLoopRange({ startSec: (trimStart + effective.start) / fps, endSec: (trimStart + effective.end) / fps });
    player.seekToSeconds((trimStart + effective.start) / fps);
    player.play();
  };

  // Dragging a handle (or typing a new "Regenerate to") while the loop is
  // armed must not leave the player looping the OLD span while the UI shows
  // the new one -- ported from the predecessor VideoInpaintRangeTimeline,
  // which had this same effect for the same reason.
  useEffect(() => {
    if (!player || !player.isLooping) return;
    const fps = frameRate > 0 ? frameRate : 24;
    player.setLoopRange({ startSec: (trimStart + effective.start) / fps, endSec: (trimStart + effective.end) / fps });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [player?.isLooping, player?.setLoopRange, trimStart, effective.start, effective.end, frameRate]);

  const seconds = (frame: number) => (frameRate > 0 ? frame / frameRate : 0);

  // ---- Mask-keyframe-list edits made by this component's own controls
  // (duplicate, delete, transform, interpolation, frame move, composite
  // feather) commit straight through `onChange`/`onCompositeFeatherPxChange`.
  // Undo/redo lives in InpaintPanel (see the module comment above); it sees
  // every one of these commits AND drawing a new mask, so a single history
  // stack there stays consistent with the keyframe list no matter which of
  // the two paths produced the edit.
  const orderedKeyframes = sortKeyframes(keyframes);
  // At least one full latent group must stay outside the regenerate range
  // (enforced elsewhere), so `rangeEnd > rangeStart` always holds once a
  // clip is loaded; `hasRange` still gates the UI explicitly rather than
  // relying on that invariant, matching the predecessor VideoInpaintMaskTimeline.
  const hasRange = bounds.length > 0 && rangeEnd > rangeStart;
  const lastFrameInRange = Math.max(rangeStart, rangeEnd - 1);
  const addFrame = hasRange
    ? clampFrame(Math.round((player?.currentFrame ?? 0) - trimStart), rangeStart, lastFrameInRange)
    : clampFrame(Math.round((player?.currentFrame ?? 0) - trimStart), 0, lastFrame);
  const existingAtAddFrame = orderedKeyframes.find((keyframe) => keyframe.frame === addFrame);
  const atKeyframeCap = !existingAtAddFrame && keyframes.length >= MAX_MASK_KEYFRAMES;
  const assetById = new Map(assets.map((asset) => [asset.id, asset]));

  const addAtPlayhead = () => {
    if (maskDisabled || disabled || trimmedFrames <= 0 || !hasRange) return;
    if (existingAtAddFrame) {
      setNotice(null);
      setSelectedId(existingAtAddFrame.id);
      onEditKeyframe(existingAtAddFrame);
      return;
    }
    if (atKeyframeCap) {
      setNotice(`This clip already has the maximum of ${MAX_MASK_KEYFRAMES} mask keyframes. Delete one before adding another.`);
      return;
    }
    setNotice(null);
    onAddKeyframe(addFrame);
  };

  // These mutate against the raw `keyframes` prop (the whole manifest) --
  // every keyframe is shown and editable now, in or out of the current
  // regenerate range, so there is no pruned view left to build from.
  const changeInterpolation = (keyframe: VideoMaskKeyframe, interpolation: MaskInterpolation, isFinalKeyframe: boolean) => {
    if (maskDisabled || disabled || isFinalKeyframe) return;
    onChange(upsertKeyframe(keyframes, { ...keyframe, interpolationToNext: interpolation || DEFAULT_MASK_INTERPOLATION }));
  };

  const changeTransform = (keyframe: VideoMaskKeyframe, field: keyof MaskTransform, value: number) => {
    if (maskDisabled || disabled) return;
    if (!Number.isFinite(value)) return;
    onChange(upsertKeyframe(keyframes, { ...keyframe, transform: { ...keyframe.transform, [field]: value } }));
  };

  const duplicateKeyframe = (source: VideoMaskKeyframe) => {
    if (maskDisabled || disabled) return;
    if (keyframes.length >= MAX_MASK_KEYFRAMES) {
      setNotice(`This clip already has the maximum of ${MAX_MASK_KEYFRAMES} mask keyframes. Delete one before duplicating another.`);
      return;
    }
    // Every keyframe is visible/editable now, so a duplicate may land
    // anywhere in the clip, not just inside the current regenerate range.
    const frame = findFreeFrame(source, orderedKeyframes, 0, lastFrame);
    if (frame === null) {
      setNotice("There is no free frame anywhere in this clip for a duplicate.");
      return;
    }
    const duplicate: VideoMaskKeyframe = { ...source, id: uniqueCopyId(source, keyframes), frame, transform: { ...source.transform } };
    setNotice(null);
    setSelectedId(duplicate.id);
    onChange(upsertKeyframe(keyframes, duplicate));
  };

  const deleteKeyframe = (keyframe: VideoMaskKeyframe) => {
    if (maskDisabled || disabled) return;
    setSelectedId((selected) => (selected === keyframe.id ? null : selected));
    onChange(removeKeyframe(keyframes, keyframe.id));
  };

  // Shared by the drag-to-move marker and the numeric frame field. Rejects
  // (rather than silently overwriting) a target frame already occupied by
  // a DIFFERENT keyframe -- the manifest requires unique frames.
  const moveKeyframe = (keyframe: VideoMaskKeyframe, targetFrame: number): boolean => {
    if (maskDisabled || disabled) return false;
    const clamped = clampFrame(targetFrame, 0, lastFrame);
    if (clamped === keyframe.frame) return true;
    const occupied = keyframes.some((other) => other.id !== keyframe.id && other.frame === clamped);
    if (occupied) {
      setNotice(`Frame ${clamped} already has a mask keyframe. Choose a different frame, or delete the one there first.`);
      return false;
    }
    setNotice(null);
    onChange(upsertKeyframe(keyframes, { ...keyframe, frame: clamped }));
    return true;
  };

  const changeFeather = (value: number) => {
    if (maskDisabled || disabled) return;
    const clamped = Math.max(0, Math.min(MAX_COMPOSITE_FEATHER_PX, Math.round(value)));
    onCompositeFeatherPxChange(clamped);
  };

  const domain: TimelineDomain = { min: 0, max: safeRaw };
  const denseGroups = groups.length > DENSE_GROUP_THRESHOLD;
  const decimateStep = denseGroups ? Math.ceil(groups.length / DENSE_GROUP_THRESHOLD) : 1;

  const outOfRangeCount = keyframes.filter((k) => k.frame < rangeStart || k.frame >= rangeEnd).length;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Regenerate from (frame)</label>
          <div className="flex gap-1">
            <NumberInput
              label="Regenerate from"
              value={effective.start}
              onCommit={commitStart}
              min={0}
              max={Math.max(0, trimmedFrames - 1)}
              step={1}
              parse="int"
              className="w-full"
              disabled={disabled || !bounds.length}
            />
            {player && (
              <Button
                variant="secondary"
                size="sm"
                disabled={disabled || !bounds.length || player.currentFrame == null}
                onClick={() => {
                  if (player.currentFrame != null) commitStart(player.currentFrame - trimStart);
                }}
                title="Set to the input video's current playhead position"
              >
                ↓ Playhead
              </Button>
            )}
          </div>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Regenerate to (frame, exclusive)</label>
          <div className="flex gap-1">
            <NumberInput
              label="Regenerate to"
              value={effective.end}
              onCommit={commitEnd}
              min={1}
              max={trimmedFrames}
              step={1}
              parse="int"
              className="w-full"
              disabled={disabled || !bounds.length}
            />
            {player && (
              <Button
                variant="secondary"
                size="sm"
                disabled={disabled || !bounds.length || player.currentFrame == null}
                onClick={() => {
                  if (player.currentFrame != null) commitEnd(player.currentFrame - trimStart);
                }}
                title="Set to the input video's current playhead position"
              >
                ↓ Playhead
              </Button>
            )}
          </div>
        </div>
      </div>

      <Timeline
        domain={domain}
        frameRate={frameRate}
        playheadFrame={player?.currentFrame ?? null}
        onSeek={player ? player.seekToFrame : undefined}
        disabled={disabled}
      >
        <RangeTrack
          trimStart={trimStart}
          trimEnd={trimEnd}
          safeRaw={safeRaw}
          groups={groups}
          bounds={bounds}
          effective={effective}
          disabled={disabled}
          videoSrc={videoSrc}
          frameRate={frameRate}
          decimateStep={decimateStep}
          startDragHandlers={{ commitStart, commitEnd, adjacentBound }}
        />
        <MaskTrack
          trimStart={trimStart}
          keyframes={orderedKeyframes}
          rangeStart={rangeStart}
          rangeEnd={rangeEnd}
          lastFrame={lastFrame}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onMoveKeyframe={moveKeyframe}
          disabled={disabled || maskDisabled}
        />
      </Timeline>
      {/* Outside `Timeline`'s click-to-seek surface (that container's
          `onPointerDown` seeks/scrubs on any press in empty track space) so
          clicking the legend cannot jump the playhead, and outside the
          shared playhead line's `absolute inset-y-0` stacking context so
          that line no longer stretches over the legend row. Gated on there
          being segments -- with zero keyframes, or while the mask track is
          disabled, there is nothing on the track for the legend to explain. */}
      {!maskDisabled && hasRange && orderedKeyframes.length > 0 && <MaskSegmentLegend />}

      {bounds.length > 0 ? (
        <p className="text-xs text-gray-500">
          Regenerate frames {effective.start} ({formatTimecode(seconds(effective.start), frameRate)}) to{" "}
          {effective.end} ({formatTimecode(seconds(effective.end), frameRate)}) of the trimmed clip —{" "}
          {effective.end - effective.start} frame(s), {selectedGroups} of {groups.length} latent groups.
          Preserved: {trimmedFrames - (effective.end - effective.start)} frame(s).
          {denseGroups && ` Group boundary lines are shown every ${decimateStep} groups on this clip.`}
        </p>
      ) : (
        <p className="text-xs text-gray-500">Load a clip to choose a range.</p>
      )}
      <div className="flex items-center gap-2 text-xs text-gray-500 flex-wrap">
        <span>
          Handles snap to latent-group boundaries
          {latentChunkPattern.length > 0 && ` (pattern repeats every ${latentChunkPattern.length} group(s): ${latentChunkPattern.join(", ")} frame(s) each)`}
        </span>
        <InlineHelp label="Temporal inpaint range details">
          <p>The video VAE processes groups of up to four frames, so each group is regenerated or preserved as a unit.</p>
          <p>Preserved pixels are pasted back after decode while their re-encoded latents condition the selected range. A boundary seam may remain visible depending on the clip.</p>
          <p>The control keeps at least one group preserved; replacing the full clip is a text-to-video request.</p>
        </InlineHelp>
        {player && (
          <Button
            variant={player.isLooping ? "primary" : "secondary"}
            size="sm"
            disabled={disabled || !bounds.length}
            onClick={toggleLoop}
            title="Loop the input video over the currently selected regenerate range"
          >
            {player.isLooping ? "Stop loop" : "Loop selection"}
          </Button>
        )}
      </div>

      <section aria-label="Video inpaint mask timeline" className="mt-2 border-t border-gray-700 pt-3 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-medium text-gray-300">Mask keyframes</h3>
            <p className="text-xs text-gray-500">Draw a mask at keyframes and choose how it applies until the next one.</p>
            <p className="text-xs text-gray-500">
              Inpaint range: [{rangeStart}, {rangeEnd}) (trimmed-clip frames). The ruler above and each
              keyframe's Frame field also use trimmed-clip frames; add {trimStart} to a trimmed-clip frame
              number to find its position on the ruler, which is drawn in the uploaded clip's own frame numbers.
            </p>
          </div>
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="secondary"
              size="xs"
              disabled={disabled || maskDisabled || !canUndo}
              onClick={onUndo}
              title="Undo the last mask-keyframe or asset edit, including a keyframe added by drawing a new mask"
            >
              Undo
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="xs"
              disabled={disabled || maskDisabled || !canRedo}
              onClick={onRedo}
              title="Redo the last undone mask-keyframe or asset edit"
            >
              Redo
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              disabled={disabled || maskDisabled || trimmedFrames <= 0 || !hasRange || atKeyframeCap}
              onClick={addAtPlayhead}
              aria-label={`Add or edit mask keyframe at frame ${addFrame}`}
              title={atKeyframeCap ? `This clip already has the maximum of ${MAX_MASK_KEYFRAMES} mask keyframes.` : undefined}
            >
              {existingAtAddFrame ? "Edit at playhead" : "Add at playhead"} ({addFrame})
            </Button>
          </div>
        </div>

        {maskDisabled && maskDisabledReason && (
          // Every mask control above and below is inert while maskDisabled
          // is set, and a disabled button that says nothing reads as the
          // feature being broken rather than as a condition to fix.
          <p className="text-xs text-amber-400">{maskDisabledReason}</p>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <label className="text-xs text-gray-500" htmlFor="composite-feather-px">
            Composite feather (px)
          </label>
          <BlurCommitNumberField
            id="composite-feather-px"
            ariaLabel="Composite feather (px)"
            value={compositeFeatherPx}
            onCommit={changeFeather}
            min={0}
            max={MAX_COMPOSITE_FEATHER_PX}
            step={1}
            parse="int"
            className="w-20"
            disabled={disabled || maskDisabled}
          />
          <InlineHelp label="Composite feather details">
            <p>Softens the edge where each regenerated mask blends back into the preserved pixels. 0 keeps a hard edge.</p>
          </InlineHelp>
        </div>

        {notice && <p className="text-xs text-amber-300" role="status">{notice}</p>}

        {outOfRangeCount > 0 && (
          <p className="text-xs text-amber-400" role="alert">
            {outOfRangeCount} mask keyframe{outOfRangeCount === 1 ? "" : "s"} outside the current regenerate
            range [{rangeStart}, {rangeEnd}). They are shown below (marked "outside range") and can still be
            edited, moved, or deleted, but will not be composited unless the regenerate range is widened to
            include them.
          </p>
        )}

        {orderedKeyframes.length === 0 ? (
          <p className="rounded border border-dashed border-gray-700 px-3 py-4 text-xs text-gray-500">
            No mask keyframes yet. Add one at the playhead to begin.
          </p>
        ) : (
          <div className="space-y-2" role="list" aria-label="Mask keyframes">
            {orderedKeyframes.map((keyframe, index) => {
              const asset = assetById.get(keyframe.maskId);
              const outOfRange = keyframe.frame < rangeStart || keyframe.frame >= rangeEnd;
              const isFinal = index === orderedKeyframes.length - 1;
              return (
                <div
                  key={keyframe.id}
                  role="listitem"
                  className={`rounded border p-2 ${
                    selectedId === keyframe.id ? "border-violet-400/70 bg-gray-800" : "border-gray-700 bg-gray-900/40"
                  }`}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    {/* Thumbnail: the SAVED asset's own pixels, not the
                        interpolated composite the backend produces between
                        keyframes -- interpolation is not reproduced client-
                        side, so this is not a preview of the actual output
                        at intermediate frames. */}
                    <button
                      type="button"
                      disabled={disabled || maskDisabled}
                      className="h-10 w-10 shrink-0 overflow-hidden rounded border border-gray-700 bg-gray-950 focus:outline-none focus:ring-1 focus:ring-violet-400"
                      onClick={() => {
                        if (disabled || maskDisabled) return;
                        setSelectedId(keyframe.id);
                        onEditKeyframe(keyframe);
                      }}
                      title="Edit this mask"
                      aria-label={`Edit mask at frame ${keyframe.frame}`}
                    >
                      {asset ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={asset.dataUrl} alt="" className="h-full w-full object-cover" />
                      ) : (
                        <span className="flex h-full w-full items-center justify-center text-[9px] text-gray-600">?</span>
                      )}
                    </button>
                    <div className="flex items-center gap-1">
                      <span className="text-[10px] text-gray-500">Frame</span>
                      <BlurCommitNumberField
                        ariaLabel={`Frame for keyframe ${keyframe.id}`}
                        value={keyframe.frame}
                        onCommit={(value) => moveKeyframe(keyframe, value)}
                        min={0}
                        max={lastFrame}
                        step={1}
                        parse="int"
                        className="w-16"
                        disabled={disabled || maskDisabled}
                      />
                    </div>
                    <span className="text-gray-500 text-xs">{keyframe.maskId}</span>
                    {outOfRange && (
                      <span className="rounded bg-amber-900/40 px-1.5 py-0.5 text-[10px] text-amber-300">
                        outside range
                      </span>
                    )}
                    <span className="text-[10px] text-gray-600">{isFinal ? "last (no interpolation)" : "to next"}</span>
                    <Select
                      title={isFinal ? "Final keyframe has no next segment" : undefined}
                      className="min-w-[7rem]"
                      options={interpolationOptions}
                      value={keyframe.interpolationToNext || DEFAULT_MASK_INTERPOLATION}
                      onChange={(event) => changeInterpolation(keyframe, event.target.value as MaskInterpolation, isFinal)}
                      disabled={disabled || maskDisabled || isFinal}
                      aria-label={`Interpolation after frame ${keyframe.frame}`}
                    />
                    <details className="w-full rounded border border-gray-800 bg-gray-950/40 px-2 py-1">
                      <summary className="cursor-pointer text-[10px] text-gray-500">Transform (canvas center pivot)</summary>
                      <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-5">
                        {transformFields.map((field) => (
                          <label key={field.key} className="text-[10px] text-gray-500">
                            {field.label}
                            <BlurCommitNumberField
                              step={field.step}
                              min={field.min !== undefined ? Number(field.min) : undefined}
                              max={field.max !== undefined ? Number(field.max) : undefined}
                              parse="float"
                              value={keyframe.transform[field.key]}
                              disabled={disabled || maskDisabled}
                              onCommit={(value) => changeTransform(keyframe, field.key, value)}
                              className="mt-1 w-full"
                              ariaLabel={`${field.label} for frame ${keyframe.frame}`}
                            />
                          </label>
                        ))}
                      </div>
                    </details>
                    <div className="ml-auto flex items-center gap-1">
                      <Button
                        type="button"
                        variant="secondary"
                        size="xs"
                        disabled={disabled || maskDisabled}
                        onClick={() => {
                          setSelectedId(keyframe.id);
                          onEditKeyframe(keyframe);
                        }}
                        aria-label={`Edit mask at frame ${keyframe.frame}`}
                      >
                        Edit mask
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        size="xs"
                        disabled={disabled || maskDisabled || keyframes.length >= MAX_MASK_KEYFRAMES}
                        onClick={() => duplicateKeyframe(keyframe)}
                        aria-label={`Duplicate mask keyframe at frame ${keyframe.frame}`}
                        title={keyframes.length >= MAX_MASK_KEYFRAMES ? `This clip already has the maximum of ${MAX_MASK_KEYFRAMES} mask keyframes.` : undefined}
                      >
                        Duplicate
                      </Button>
                      <Button
                        type="button"
                        variant="danger"
                        size="xs"
                        disabled={disabled || maskDisabled}
                        onClick={() => deleteKeyframe(keyframe)}
                        aria-label={`Delete mask keyframe at frame ${keyframe.frame}`}
                      >
                        Delete
                      </Button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// A number field that commits only on blur/Enter, for the fields in this
// component that feed InpaintPanel's undo/redo history (frame reassignment,
// transform, composite feather). Deliberately NOT the shared `NumberInput`
// (which commits on every keystroke, by design, for callers like the
// Regenerate from/to fields above that do not push each keystroke onto a
// history stack): a caller of THIS field expects one history entry per
// edit, and a live-commit-while-typing field here would (a) reject a value
// the user has not finished typing yet whenever an intermediate digit
// happens to collide with another keyframe's frame, and (b) push one
// history entry per keystroke, so a single undo would not undo the whole
// edit. `NumberInput` itself is left untouched -- it is used across many
// other panels for live-update UX that is unrelated to this history.
// ---------------------------------------------------------------------------

interface BlurCommitNumberFieldProps {
  id?: string;
  ariaLabel?: string;
  value: number;
  /** Called once, on blur/Enter. A `false` return (e.g. the target frame is
   * already occupied) snaps the draft back to `value` instead of leaving
   * the rejected text sitting in the field. */
  onCommit: (value: number) => boolean | void;
  min?: number;
  max?: number;
  step?: string | number;
  parse?: "int" | "float";
  className?: string;
  disabled?: boolean;
}

function BlurCommitNumberField({
  id,
  ariaLabel,
  value,
  onCommit,
  min,
  max,
  step,
  parse = "int",
  className,
  disabled,
}: BlurCommitNumberFieldProps) {
  const [draft, setDraft] = useState<string>(String(value));
  const focusedRef = useRef(false);
  // Enter commits by calling `.blur()` (which synchronously fires the
  // native blur event, running `onBlur` below before `.blur()` itself
  // returns) rather than calling `commit()` directly -- so `onBlur` must
  // NOT commit a second time. Escape needs the opposite: it resets `draft`
  // then blurs, but the `onBlur` closure still sees the PRE-reset `draft`
  // (the `setDraft` call has not re-rendered yet), so committing there
  // would parse the abandoned text instead of reverting -- this flag skips
  // that commit entirely for Escape.
  const skipNextBlurCommitRef = useRef(false);

  useEffect(() => {
    if (!focusedRef.current) setDraft(String(value));
  }, [value]);

  const parseValue = (text: string): number | null => {
    if (text.trim() === "") return null;
    const parsed = parseFloat(text);
    if (isNaN(parsed)) return null;
    return parse === "int" ? Math.round(parsed) : parsed;
  };
  const clamp = (num: number): number => {
    let clamped = num;
    if (min !== undefined && clamped < min) clamped = min;
    if (max !== undefined && clamped > max) clamped = max;
    return clamped;
  };

  const commit = () => {
    const parsed = parseValue(draft);
    if (parsed === null) {
      setDraft(String(value));
      return;
    }
    const normalized = clamp(parsed);
    const accepted = onCommit(normalized);
    if (accepted === false) {
      // Rejected (e.g. frame already occupied) -- the manifest did not
      // change, so the field must not keep showing the rejected text.
      setDraft(String(value));
    } else {
      setDraft(String(normalized));
    }
  };

  return (
    <input
      id={id}
      type="number"
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      aria-label={ariaLabel}
      className={cn(
        "h-7 rounded-md border border-gray-700 bg-gray-800 px-2 text-xs text-gray-100 focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500",
        className,
      )}
      value={draft}
      onFocus={() => {
        focusedRef.current = true;
      }}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => {
        focusedRef.current = false;
        if (skipNextBlurCommitRef.current) {
          skipNextBlurCommitRef.current = false;
          return;
        }
        commit();
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          // `.blur()` alone commits (via `onBlur` above) -- calling
          // `commit()` here too would push a second, redundant history
          // entry for the same edit.
          (e.currentTarget as HTMLInputElement).blur();
        } else if (e.key === "Escape") {
          skipNextBlurCommitRef.current = true;
          setDraft(String(value));
          (e.currentTarget as HTMLInputElement).blur();
        }
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Track 1: Regenerate range. Ported from VideoInpaintRangeTimeline; the only
// change is reading pointer<->frame conversion, hover, and the ruler/
// playhead from the shared `Timeline` (via `useTimelineContext`) instead of
// owning its own container ref and pointer handlers for those.
// ---------------------------------------------------------------------------

interface RangeTrackProps {
  trimStart: number;
  trimEnd: number;
  safeRaw: number;
  groups: Array<[number, number]>;
  bounds: number[];
  effective: { start: number; end: number };
  disabled: boolean;
  videoSrc: string | null;
  frameRate: number;
  decimateStep: number;
  startDragHandlers: {
    commitStart: (raw: number) => void;
    commitEnd: (raw: number) => void;
    adjacentBound: (frame: number, dir: -1 | 1, stride: number) => number;
  };
}

type DragMode = "start" | "end" | null;

function RangeTrack({
  trimStart,
  trimEnd,
  safeRaw,
  groups,
  bounds,
  effective,
  disabled,
  videoSrc,
  frameRate,
  decimateStep,
  startDragHandlers,
}: RangeTrackProps) {
  const ctx = useTimelineContext();
  const [dragMode, setDragMode] = useState<DragMode>(null);
  const [dragPreviewRawFrame, setDragPreviewRawFrame] = useState<number | null>(null);
  const { commitStart, commitEnd, adjacentBound } = startDragHandlers;

  const pct = (frameTrimmed: number) => ctx.percentForFrame(trimStart + frameTrimmed);
  const rawPct = (rawFrame: number) => ctx.percentForFrame(rawFrame);
  const seconds = (frame: number) => (frameRate > 0 ? frame / frameRate : 0);
  const fmt = (frame: number) => `${frame} (${formatTimecode(seconds(frame), frameRate)})`;

  const startDrag = (mode: DragMode) => (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled || !bounds.length) return;
    e.preventDefault();
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    setDragMode(mode);
  };
  const onHandlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragMode) return;
    const rawFrame = ctx.frameAtClientX(e.clientX);
    setDragPreviewRawFrame(Math.max(0, Math.min(safeRaw - 1, rawFrame)));
    const trimmedFrame = rawFrame - trimStart;
    if (dragMode === "start") commitStart(trimmedFrame);
    else commitEnd(trimmedFrame);
  };
  const onHandlePointerUp = () => {
    setDragMode(null);
    setDragPreviewRawFrame(null);
  };

  const previewRawFrame = dragPreviewRawFrame ?? ctx.hoverFrame;

  return (
    <div className="relative">
      <div className="relative h-16 bg-gray-800 border border-gray-600 rounded overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          {groups.map(([lo], index) => {
            if (index !== 0 && index % decimateStep !== 0) return null;
            return (
              <div key={index} className="absolute top-0 bottom-0 border-l border-gray-700/70" style={{ left: `${pct(lo)}%` }} />
            );
          })}
          {groups.length > 0 && (
            <div
              className="absolute top-0 bottom-0 bg-sky-500/10 border-r border-sky-500/50"
              style={{ left: `${pct(groups[0][0])}%`, width: `${Math.max(0.3, ((groups[0][1] - groups[0][0]) / safeRaw) * 100)}%` }}
              title={`First latent group: ${groups[0][1] - groups[0][0]} frame(s)`}
            />
          )}
        </div>

        {trimStart > 0 && (
          <div
            className="absolute top-0 bottom-0 left-0 bg-gray-900/80 border-r border-gray-600"
            style={{ width: `${(trimStart / safeRaw) * 100}%` }}
            title="Trimmed off the uploaded clip before anything else"
          />
        )}
        {trimEnd > 0 && (
          <div
            className="absolute top-0 bottom-0 right-0 bg-gray-900/80 border-l border-gray-600"
            style={{ width: `${(trimEnd / safeRaw) * 100}%` }}
            title="Trimmed off the uploaded clip before anything else"
          />
        )}

        <span className="absolute left-1 top-1 text-[10px] text-gray-400 pointer-events-none">Regenerate range</span>

        {bounds.length > 0 && (
          <div
            className="absolute top-4 bottom-4 border-2 border-amber-500 bg-amber-500/25 rounded"
            style={{ left: `${pct(effective.start)}%`, width: `${Math.max(0.5, ((effective.end - effective.start) / safeRaw) * 100)}%` }}
            title="Regenerate: this span is generated; everything else is the input's own pixels"
          >
            <span className="absolute left-1 top-0 text-[10px] text-amber-200 pointer-events-none">Regenerate</span>
            <div
              role="slider"
              tabIndex={disabled || !bounds.length ? -1 : 0}
              aria-label="Regenerate range start"
              aria-valuemin={0}
              aria-valuemax={bounds.length ? bounds[bounds.length - 2] : 0}
              aria-valuenow={effective.start}
              aria-valuetext={fmt(effective.start)}
              onPointerDown={startDrag("start")}
              onPointerMove={onHandlePointerMove}
              onPointerUp={onHandlePointerUp}
              onPointerCancel={onHandlePointerUp}
              onKeyDown={(e) => {
                if (disabled || !bounds.length) return;
                if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                  e.preventDefault();
                  const dir = e.key === "ArrowLeft" ? -1 : 1;
                  commitStart(adjacentBound(effective.start, dir, e.shiftKey ? 4 : 1));
                } else if (e.key === "PageDown" || e.key === "PageUp") {
                  e.preventDefault();
                  commitStart(adjacentBound(effective.start, e.key === "PageDown" ? -1 : 1, 4));
                } else if (e.key === "Home") {
                  e.preventDefault();
                  commitStart(bounds[0]);
                } else if (e.key === "End") {
                  e.preventDefault();
                  commitStart(bounds[bounds.length - 2]);
                }
              }}
              className={`absolute top-0 bottom-0 left-0 w-2 -ml-1 bg-amber-400 hover:bg-amber-300 focus:outline-none focus:ring-2 focus:ring-amber-300 ${
                disabled ? "cursor-not-allowed" : "cursor-ew-resize"
              }`}
              title="Drag, or use the arrow keys, to move the start of the regenerated range (snaps to latent-group boundaries; Shift jumps 4 groups)"
            />
            <div
              role="slider"
              tabIndex={disabled || !bounds.length ? -1 : 0}
              aria-label="Regenerate range end"
              aria-valuemin={bounds.length > 1 ? bounds[1] : 1}
              aria-valuemax={groups.length ? groups[groups.length - 1][1] : 0}
              aria-valuenow={effective.end}
              aria-valuetext={fmt(effective.end)}
              onPointerDown={startDrag("end")}
              onPointerMove={onHandlePointerMove}
              onPointerUp={onHandlePointerUp}
              onPointerCancel={onHandlePointerUp}
              onKeyDown={(e) => {
                if (disabled || !bounds.length) return;
                if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                  e.preventDefault();
                  const dir = e.key === "ArrowLeft" ? -1 : 1;
                  commitEnd(adjacentBound(effective.end, dir, e.shiftKey ? 4 : 1));
                } else if (e.key === "PageDown" || e.key === "PageUp") {
                  e.preventDefault();
                  commitEnd(adjacentBound(effective.end, e.key === "PageDown" ? -1 : 1, 4));
                } else if (e.key === "Home") {
                  e.preventDefault();
                  commitEnd(bounds[1]);
                } else if (e.key === "End") {
                  e.preventDefault();
                  commitEnd(bounds[bounds.length - 1]);
                }
              }}
              className={`absolute top-0 bottom-0 right-0 w-2 -mr-1 bg-amber-400 hover:bg-amber-300 focus:outline-none focus:ring-2 focus:ring-amber-300 ${
                disabled ? "cursor-not-allowed" : "cursor-ew-resize"
              }`}
              title="Drag, or use the arrow keys, to move the end of the regenerated range (snaps to latent-group boundaries; Shift jumps 4 groups)"
            />
          </div>
        )}
      </div>

      {previewRawFrame != null && (
        <FramePreviewTooltip
          videoSrc={videoSrc}
          timeSec={seconds(previewRawFrame)}
          leftPercent={rawPct(previewRawFrame)}
          label={formatFrameLabel(previewRawFrame, frameRate)}
          visible
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Track 2: Mask keyframes. Ported from VideoInpaintMaskTimeline, extended
// with: every keyframe drawn (not just ones inside the current regenerate
// range), a saved-asset thumbnail per marker, and drag-to-move (clamped to
// the clip, rejecting a frame already used by another keyframe). Selection,
// interpolation/transform editing, duplicate/delete stay in the detail list
// rendered by the parent below this track.
// ---------------------------------------------------------------------------

interface MaskTrackProps {
  trimStart: number;
  keyframes: VideoMaskKeyframe[];
  rangeStart: number;
  rangeEnd: number;
  lastFrame: number;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onMoveKeyframe: (keyframe: VideoMaskKeyframe, targetFrame: number) => boolean;
  disabled: boolean;
}

function MaskTrack({
  trimStart,
  keyframes,
  rangeStart,
  rangeEnd,
  lastFrame,
  selectedId,
  onSelect,
  onMoveKeyframe,
  disabled,
}: MaskTrackProps) {
  const ctx = useTimelineContext();
  const [dragId, setDragId] = useState<string | null>(null);
  const [dragGhostFrame, setDragGhostFrame] = useState<number | null>(null);

  // Clamped against the SHARED RAW domain, not `lastFrame` (trimmed-space):
  // a valid `rangeEnd` (exclusive) can legitimately sit one frame past
  // `lastFrame` while still being inside the raw domain, so clamping in
  // trimmed-space first would needlessly pull the range highlight's right
  // edge inward. Only a raw position genuinely outside the domain -- e.g. a
  // keyframe's stored `frame` left stale after the user later increased the
  // trim -- gets pulled back to the domain edge here.
  const pctForFrame = (frameTrimmed: number) => {
    const raw = trimStart + frameTrimmed;
    const clampedRaw = Math.max(ctx.domain.min, Math.min(ctx.domain.max, raw));
    return ctx.percentForFrame(clampedRaw);
  };

  const startDrag = (keyframe: VideoMaskKeyframe) => (e: React.PointerEvent<HTMLButtonElement>) => {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragId(keyframe.id);
    setDragGhostFrame(keyframe.frame);
    onSelect(keyframe.id);
  };
  const onDragMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (!dragId) return;
    const rawFrame = ctx.frameAtClientX(e.clientX);
    setDragGhostFrame(clampFrame(rawFrame - trimStart, 0, lastFrame));
  };
  const onDragEnd = (keyframe: VideoMaskKeyframe) => () => {
    if (dragId === keyframe.id && dragGhostFrame != null) {
      onMoveKeyframe(keyframe, dragGhostFrame);
    }
    setDragId(null);
    setDragGhostFrame(null);
  };
  // A pointercancel is an INTERRUPTION (e.g. the browser takes over the
  // gesture for something else), not a completed drag -- unlike
  // `onDragEnd`, this must discard the in-progress move rather than commit
  // whatever frame the ghost marker last showed.
  const onDragCancel = () => {
    setDragId(null);
    setDragGhostFrame(null);
  };

  // Keyframes actually being DRAGGED report their live ghost frame here too,
  // so the segment bars re-flow with the marker instead of lagging behind it
  // until the drag commits.
  const segmentSourceKeyframes = keyframes.map((keyframe) =>
    dragId === keyframe.id && dragGhostFrame != null ? { ...keyframe, frame: dragGhostFrame } : keyframe,
  );
  const segments = computeMaskSegments(sortKeyframes(segmentSourceKeyframes), rangeStart, rangeEnd);

  return (
    <div className="relative h-16 overflow-hidden rounded border border-gray-700 bg-gray-800">
      <div
        className="absolute inset-y-0 border-x border-amber-500/70 bg-amber-500/20"
        style={{ left: `${pctForFrame(rangeStart)}%`, width: `${Math.max(0.5, pctForFrame(rangeEnd) - pctForFrame(rangeStart))}%` }}
        title="Inpaint range"
      >
        <span className="absolute left-1 top-1 text-[10px] text-amber-200 pointer-events-none">Inpaint range</span>
      </div>
      <div className="absolute inset-x-0 top-1/2 h-px bg-gray-600" />

      {/* Mode-coloured spans, drawn below the diamond markers (earlier in
          DOM order, so the markers stack on top) and `pointer-events-none`
          so they never intercept the markers' own drag/click handlers. */}
      <div className="absolute inset-x-0 bottom-1 h-3" aria-hidden="true">
        {segments.map((segment) => {
          const left = pctForFrame(segment.start);
          const width = Math.max(0.4, pctForFrame(segment.end) - left);
          return (
            <div
              key={segment.key}
              className={cn("absolute inset-y-0 rounded-sm border pointer-events-none", MASK_SEGMENT_STYLE[segment.mode])}
              style={{ left: `${left}%`, width: `${width}%` }}
              title={segment.title}
            />
          );
        })}
      </div>

      <div className="absolute inset-x-0 top-0 h-full" role="list" aria-label="Mask keyframe markers">
        {keyframes.map((keyframe) => {
          const isDragging = dragId === keyframe.id;
          const displayFrame = isDragging && dragGhostFrame != null ? dragGhostFrame : keyframe.frame;
          const outOfRange = displayFrame < rangeStart || displayFrame >= rangeEnd;
          const selected = selectedId === keyframe.id;
          const left = pctForFrame(displayFrame);
          return (
            <span key={keyframe.id} role="listitem">
              <button
                type="button"
                disabled={disabled}
                className={`absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rotate-45 border focus:outline-none focus:ring-2 focus:ring-violet-300 ${
                  outOfRange
                    ? selected
                      ? "border-white bg-gray-500"
                      : "border-gray-400 bg-gray-600"
                    : selected
                    ? "border-white bg-violet-400"
                    : "border-violet-300 bg-violet-600"
                }`}
                style={{ left: `${left}%`, cursor: disabled ? "default" : "grab" }}
                onPointerDown={startDrag(keyframe)}
                onPointerMove={onDragMove}
                onPointerUp={onDragEnd(keyframe)}
                onPointerCancel={onDragCancel}
                onClick={() => {
                  if (disabled || isDragging) return;
                  onSelect(keyframe.id);
                }}
                aria-label={`${frameDescription(keyframe.frame)} mask ${keyframe.maskId}${outOfRange ? " (outside regenerate range)" : ""}`}
                aria-pressed={selected}
                title={`${frameDescription(keyframe.frame)} - ${keyframe.maskId}${outOfRange ? " (outside regenerate range)" : ""} - drag to move`}
              />
            </span>
          );
        })}
      </div>
    </div>
  );
}

// Maps the mode colours drawn on the track back to their names, since a
// colour alone is not discoverable -- matches this file's `interpolationOptions`
// naming (the per-keyframe interpolation `<Select>` above).
function MaskSegmentLegend() {
  const entries: MaskInterpolation[] = ["hold", "affine", "sdf"];
  return (
    <div className="mt-1 flex flex-wrap items-center gap-3 text-[10px] text-gray-500">
      <span>Mask span:</span>
      {entries.map((mode) => (
        <span key={mode} className="flex items-center gap-1">
          <span className={cn("inline-block h-2 w-4 rounded-sm border", MASK_SEGMENT_STYLE[mode])} />
          {MASK_SEGMENT_MODE_LABEL[mode]}
        </span>
      ))}
    </div>
  );
}
