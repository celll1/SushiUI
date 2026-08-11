"use client";

import { useEffect, useRef, useState } from "react";
import NumberInput from "../common/NumberInput";
import Button from "../common/Button";
import FramePreviewTooltip from "../common/FramePreviewTooltip";
import { formatTimecode } from "@/utils/timecode";
import { percentForValue, valueAtClientX } from "@/utils/timelineScale";
import type { VideoPlayheadState } from "@/hooks/useVideoPlayhead";

// ---------------------------------------------------------------------------
// Generic horizontal timeline widget for TEMPORAL outpaint placement.
//
// Modality-neutral by design (reused by the video branch here, and by the
// audio branch): everything is expressed in an abstract "unit" space (video:
// frames; audio: seconds) plus an optional `unitRate` (units-per-second)
// purely to render a seconds ruler.
//
// Model:
//   - `totalUnits`      : length of the OUTPUT timeline, in units.
//   - `rawSegmentLength`: length of the UPLOADED input clip, in units, BEFORE
//                          trim (e.g. total decoded video frames).
//   - `trimStart`/`trimEnd`: units trimmed off the input clip's start/end
//                          BEFORE placement. The placed segment length is
//                          `rawSegmentLength - trimStart - trimEnd` (floored
//                          at `minSegmentLength`).
//   - `offset`          : where the (trimmed) segment's first unit lands on
//                          the OUTPUT timeline. Independent of trim.
//
// Interaction (mirrors standard NLE trim-handle semantics):
//   - Dragging the BODY of the block moves `offset` only (segment length
//     unchanged).
//   - Dragging the RIGHT handle changes `trimEnd` only; the right edge of the
//     block follows the pointer, the left edge (offset) stays fixed.
//   - Dragging the LEFT handle changes `trimStart`; to keep it a true
//     direct-manipulation handle (the edge follows the pointer), `offset` is
//     shifted by the same delta as `trimStart` so the block's RIGHT edge
//     stays fixed while its left edge moves with the handle -- exactly how
//     trimming the head of a clip behaves in a video timeline. This is a UI
//     convenience only; the two params remain independently meaningful to
//     the backend (trimStart trims the source clip, offset places it).
//
// Snapping: a generic `gridSize` (e.g. 8 for LTX-2.3's 8-frame temporal
// compression) is applied to every value on drag release / numeric commit.
// An optional stricter `offsetSnapFn` (e.g. the LTX-2.3 valid-latent-index
// rule {0, 1, 9, 17, ..., 8k+1}) is applied to `offset` specifically, on top
// of the grid snap. The backend re-validates/snaps regardless (this is a UX
// nicety, not the source of truth).
//
// `player`/`videoSrc` (both optional; video mount only -- the audio mount
// omits them and the track degrades to exactly its prior behavior) wire this
// track to the panel's own input <video>: a moving marker inside the placed
// block synced to its live position, click-to-seek within the placed block
// (there is no source content to seek to outside it), and a frame preview
// while hovering/dragging within the block. The preview maps a pointer
// position within the block back to the RAW clip frame it plays -- content
// does not otherwise change while dragging the block to move it, so that
// frame is shown for a body-drag too, not only the trim handles.
//
// Click vs. drag: the block body is both the "move" drag handle AND the only
// element that ever receives a pointer down within the placed segment (it
// sits on top of the track for that whole span), so seeking is implemented
// as a SHORT-CIRCUITED drag rather than a separate track-level handler: a
// pointer-down/up on the block with less than CLICK_THRESHOLD_PX of movement
// in between is treated as a click (seeks, commits nothing) instead of a
// move (commits the new offset). Actual movement past the threshold still
// moves/trims exactly as before, from the drag's original start values.
// ---------------------------------------------------------------------------

export interface OutpaintTimelineProps {
  totalUnits: number;
  onTotalUnitsChange: (v: number) => void;
  /** Optional hard constraint applied to totalUnits on commit (e.g. LTX-2.3's (n-1)%8==0). */
  totalUnitsSnapFn?: (v: number) => number;
  totalUnitsMin?: number;
  /**
   * Optional hard ceiling on totalUnits, applied AFTER totalUnitsSnapFn so a
   * snap-up rule cannot push the committed value back past it. Absent = no
   * ceiling (the historical behavior). Also passed to the NumberInput's own
   * `max`, so a wildly unservable value is not even typeable.
   */
  totalUnitsMax?: number;
  totalUnitsStep?: number;

  /** Full length of the uploaded input clip, in units, before trim. */
  rawSegmentLength: number;
  trimStart: number;
  onTrimStartChange: (v: number) => void;
  trimEnd: number;
  onTrimEndChange: (v: number) => void;

  offset: number;
  onOffsetChange: (v: number) => void;
  /** Optional stricter snap for offset only (e.g. LTX-2.3 valid latent index rule). */
  offsetSnapFn?: (v: number) => number;

  /** Generic snap grid applied to offset/trimStart/trimEnd (default 1 = no snap). */
  gridSize?: number;
  /** Floor for the placed segment length after trim (default 1). */
  minSegmentLength?: number;
  /** units-per-second, for a seconds ruler under the bar (omit to label in raw units). */
  unitRate?: number;
  unitLabel?: string;
  /**
   * NumberInput parse mode for the four numeric fields (total/offset/trim).
   * Default "int" (units are whole numbers, e.g. video frames). Pass "float"
   * when units are already fractional seconds (e.g. audio, gridSize<1) so
   * typed values aren't rounded to whole numbers.
   */
  unitParse?: "int" | "float";
  disabled?: boolean;
  /** Object/file URL of the uploaded clip, for the body-drag/hover frame preview. Video mount only. */
  videoSrc?: string | null;
  /** The SAME input <video>'s live playhead (from `useVideoPlayhead`), in RAW clip units. Video mount only. */
  player?: VideoPlayheadState;
}

type DragMode = "move" | "trim-left" | "trim-right" | null;

interface DragStart {
  mouseX: number;
  offset: number;
  trimStart: number;
  trimEnd: number;
  widthPx: number;
}

const roundToGrid = (value: number, grid: number): number =>
  grid > 0 ? Math.round(value / grid) * grid : Math.round(value);

const clamp = (value: number, min: number, max: number): number => Math.min(Math.max(value, min), max);

// Below this many pixels of pointer movement, a block-body pointer-down/up
// pair is a click (seeks, commits no parameter change) rather than a drag
// (moves the segment). See the "Click vs. drag" note above.
const CLICK_THRESHOLD_PX = 4;

// Multiplier applied to the grid step for the Shift-modified arrow-key step
// on the trim handles, so the unmodified step is always the fine (grid)
// step and Shift is always the coarser one -- regardless of whether the
// grid itself is coarser or finer than 1 unit (it is finer for the audio
// mount's 1/25s grid). Matches VideoInpaintTimeline's Regenerate-range
// track's own dir*4 step.
const SHIFT_STEP_MULTIPLIER = 4;

export default function OutpaintTimeline({
  totalUnits,
  onTotalUnitsChange,
  totalUnitsSnapFn,
  totalUnitsMin = 1,
  totalUnitsMax,
  totalUnitsStep = 1,
  rawSegmentLength,
  trimStart,
  onTrimStartChange,
  trimEnd,
  onTrimEndChange,
  offset,
  onOffsetChange,
  offsetSnapFn,
  gridSize = 1,
  minSegmentLength = 1,
  unitRate,
  unitLabel = "units",
  unitParse = "int",
  disabled = false,
  videoSrc = null,
  player,
}: OutpaintTimelineProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dragMode, setDragMode] = useState<DragMode>(null);
  const dragStartRef = useRef<DragStart | null>(null);
  // Whether the pointer has moved past CLICK_THRESHOLD_PX since the current
  // drag started -- distinguishes a click (seek, no commit) from a drag
  // (move/trim, commits on release). Reset on every pointer-down.
  const dragMovedRef = useRef(false);
  const [previewUnit, setPreviewUnit] = useState<number | null>(null);
  const [loopEnabled, setLoopEnabled] = useState(false);

  const safeTotalUnits = Math.max(1, totalUnits);
  const segmentLength = Math.max(minSegmentLength, rawSegmentLength - trimStart - trimEnd);

  const applyOffsetSnap = (v: number): number => {
    let snapped = roundToGrid(v, gridSize);
    if (offsetSnapFn) snapped = offsetSnapFn(snapped);
    return clamp(snapped, 0, Math.max(0, safeTotalUnits - segmentLength));
  };

  const startDrag = (mode: DragMode) => (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const rect = containerRef.current?.getBoundingClientRect();
    setDragMode(mode);
    dragMovedRef.current = false;
    // Stored directly on the ref (not state) so the very first pointermove in
    // the same drag can read it synchronously -- a state setter's update
    // wouldn't be visible until the next render, causing the first pixel of
    // movement to be silently dropped.
    dragStartRef.current = { mouseX: e.clientX, offset, trimStart, trimEnd, widthPx: rect?.width || 1 };
  };

  // The raw input-clip unit currently under a pointer position within the
  // OUTPUT timeline, given `offset`/`trimStart` -- null outside the placed
  // segment (there is no source content to preview there).
  const rawUnitAtOutputUnit = (outputUnit: number): number | null => {
    if (outputUnit < offset || outputUnit >= offset + segmentLength) return null;
    return trimStart + (outputUnit - offset);
  };
  const outputUnitAtClientX = (clientX: number): number => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return 0;
    return valueAtClientX(clientX, rect, { min: 0, max: safeTotalUnits });
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragMode || !dragStartRef.current) return;
    const start = dragStartRef.current;
    const dxPx = e.clientX - start.mouseX;
    if (!dragMovedRef.current && Math.abs(dxPx) > CLICK_THRESHOLD_PX) {
      dragMovedRef.current = true;
    }
    if (!dragMovedRef.current) {
      // Still within the click threshold: don't commit any position/trim
      // change yet -- only a real drag (see below) does that. Do show the
      // hover frame under the pointer for the "move" mode so the pending
      // seek target is visible before release.
      if (dragMode === "move") {
        setPreviewUnit(rawUnitAtOutputUnit(clamp(outputUnitAtClientX(e.clientX), 0, safeTotalUnits)));
      }
      return;
    }
    const dxUnits = (dxPx / start.widthPx) * safeTotalUnits;

    if (dragMode === "move") {
      const newOffset = clamp(
        Math.round(start.offset + dxUnits),
        0,
        Math.max(0, safeTotalUnits - segmentLength)
      );
      onOffsetChange(newOffset);
      setPreviewUnit(rawUnitAtOutputUnit(clamp(outputUnitAtClientX(e.clientX), 0, safeTotalUnits)));
      return;
    }

    if (dragMode === "trim-right") {
      // Right edge follows the pointer: growing dx (moving right) REDUCES trimEnd,
      // shrinking dx (moving left) INCREASES it. Left edge/offset unchanged.
      const maxTrimEnd = Math.max(0, rawSegmentLength - start.trimStart - minSegmentLength);
      const newTrimEnd = clamp(Math.round(start.trimEnd - dxUnits), 0, maxTrimEnd);
      onTrimEndChange(newTrimEnd);
      setPreviewUnit(Math.max(0, rawSegmentLength - newTrimEnd - 1));
      return;
    }

    if (dragMode === "trim-left") {
      // Left edge follows the pointer: moving right INCREASES trimStart (cuts
      // more off the head); offset shifts by the same delta so the segment's
      // RIGHT edge on the timeline stays fixed (standard trim-handle feel).
      const maxTrimStart = Math.max(0, rawSegmentLength - start.trimEnd - minSegmentLength);
      const newTrimStart = clamp(Math.round(start.trimStart + dxUnits), 0, maxTrimStart);
      const appliedDelta = newTrimStart - start.trimStart;
      const newOffset = clamp(Math.round(start.offset + appliedDelta), 0, Math.max(0, safeTotalUnits));
      onTrimStartChange(newTrimStart);
      onOffsetChange(newOffset);
      setPreviewUnit(newTrimStart);
      return;
    }
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragMode) return;
    const mode = dragMode;
    const moved = dragMovedRef.current;
    setDragMode(null);
    dragStartRef.current = null;
    dragMovedRef.current = false;
    setPreviewUnit(null);

    if (!moved) {
      // A click, not a drag: commit nothing. For the block body specifically,
      // this is how seeking is reachable -- the block covers the entire
      // placed segment, so it is the only element that ever receives a
      // pointer-down there; a plain click on it seeks the input video to the
      // raw clip frame under the pointer instead of moving the segment.
      if (mode === "move" && player && unitRate && unitRate > 0) {
        const outputUnit = outputUnitAtClientX(e.clientX);
        const rawUnit = rawUnitAtOutputUnit(outputUnit);
        if (rawUnit != null) player.seekToSeconds(rawUnit / unitRate);
      }
      return;
    }

    // Snap on release (grid + optional stricter offset rule).
    onOffsetChange(applyOffsetSnap(offset));
    onTrimStartChange(roundToGrid(trimStart, gridSize));
    onTrimEndChange(roundToGrid(trimEnd, gridSize));
  };

  // Hover preview outside of any drag: only meaningful when there is a video
  // to grab a frame from (the audio mount passes neither `videoSrc` nor
  // `player`) -- otherwise this would re-render the whole timeline on every
  // pointer move for a tooltip that FramePreviewTooltip itself never shows.
  const onTrackPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (dragMode || !videoSrc) return;
    const outputUnit = outputUnitAtClientX(e.clientX);
    setPreviewUnit(rawUnitAtOutputUnit(outputUnit));
  };
  const onTrackPointerLeave = () => {
    if (!dragMode) setPreviewUnit(null);
  };

  const toggleLoop = () => {
    if (!player || !unitRate || unitRate <= 0) return;
    if (loopEnabled) {
      player.setLoopRange(null);
      setLoopEnabled(false);
      return;
    }
    // The range itself is (re-)pushed by the effect below, which also keeps
    // it in sync afterwards -- this only seeks/starts playback.
    player.seekToSeconds(trimStart / unitRate);
    player.play();
    setLoopEnabled(true);
  };

  // Keep the player's loop range in sync with the placed segment while a
  // loop is active. Without this, dragging the block or a trim handle (or
  // committing a new trim value from the numeric fields) changed `trimStart`
  // and `segmentLength` but left the player looping the OLD span until the
  // user toggled the loop off and back on.
  useEffect(() => {
    if (!loopEnabled || !player || !unitRate || unitRate <= 0) return;
    player.setLoopRange({ startSec: trimStart / unitRate, endSec: (trimStart + segmentLength) / unitRate });
  }, [loopEnabled, trimStart, segmentLength, unitRate, player?.setLoopRange]);

  const leftPct = percentForValue(offset, { min: 0, max: safeTotalUnits });
  const widthPct = percentForValue(segmentLength, { min: 0, max: safeTotalUnits });

  const formatUnit = (u: number): string => {
    if (unitRate && unitRate > 0) return `${(u / unitRate).toFixed(2)}s`;
    return `${Math.round(u)} ${unitLabel}`;
  };

  // Ruler labels. The four numeric fields above this track are in FRAMES, so a
  // seconds-only ruler made the two halves of the same control speak different
  // units; a frame-only ruler is no better next to a video the user is
  // scrubbing in seconds. Video ticks therefore carry both, matching
  // VideoInpaintTimeline's shared ruler "frame · timecode". The audio mount has no
  // frames to name (unitRate is 1 and its grid is 1/25 s), so it keeps the
  // plain seconds label rather than a timecode whose frames field is always 00.
  const isFrameRuler = !!videoSrc && !!unitRate && unitRate > 0;
  const formatTick = (u: number): string =>
    isFrameRuler
      ? `${Math.round(u)} · ${formatTimecode(u / unitRate!, unitRate!)}`
      : formatUnit(u);

  // Fewer ticks on the wider two-part label, so six of them cannot overlap on a
  // narrow panel column. Same count as the inpaint ruler.
  const tickCount = isFrameRuler ? 3 : 5;
  // Value-positioned (not the prior `justify-between` row): each label sits
  // at its own tick's actual percentage of the track, so it lines up with
  // the ruler line it labels instead of being evenly spaced regardless of
  // where `totalUnits` actually falls.
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => (safeTotalUnits * i) / tickCount);

  // Playhead: the input video's own live position, mapped into the placed
  // segment. Hidden while the player is outside the segment's raw range --
  // there is no corresponding pixel on this timeline to draw it on.
  const playheadOutputUnit =
    player?.currentTimeSec != null && unitRate && unitRate > 0
      ? player.currentTimeSec * unitRate
      : null;
  const playheadVisible =
    playheadOutputUnit != null && playheadOutputUnit >= trimStart && playheadOutputUnit < trimStart + segmentLength;
  const playheadPctInBlock = playheadVisible && playheadOutputUnit != null
    ? ((playheadOutputUnit - trimStart) / segmentLength) * 100
    : null;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Total ({unitLabel})</label>
          <NumberInput
            value={totalUnits}
            onCommit={(v) => {
              const snapped = totalUnitsSnapFn ? totalUnitsSnapFn(v) : Math.max(totalUnitsMin, v);
              // Re-clamp AFTER the snap: a snap-up rule (e.g. rounding to the
              // next 8n+1) can otherwise push the committed value back past
              // the ceiling that was just enforced by NumberInput's own `max`.
              onTotalUnitsChange(totalUnitsMax != null ? Math.min(snapped, totalUnitsMax) : snapped);
            }}
            min={totalUnitsMin}
            max={totalUnitsMax}
            step={totalUnitsStep}
            parse={unitParse}
            className="w-full"
            disabled={disabled}
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Offset ({unitLabel})</label>
          <NumberInput
            value={offset}
            onCommit={(v) => onOffsetChange(applyOffsetSnap(v))}
            min={0}
            max={Math.max(0, safeTotalUnits - segmentLength)}
            step={gridSize}
            parse={unitParse}
            className="w-full"
            disabled={disabled}
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Trim start ({unitLabel})</label>
          <div className="flex gap-1">
            <NumberInput
              value={trimStart}
              onCommit={(v) => onTrimStartChange(clamp(roundToGrid(v, gridSize), 0, Math.max(0, rawSegmentLength - trimEnd - minSegmentLength)))}
              min={0}
              max={Math.max(0, rawSegmentLength - trimEnd - minSegmentLength)}
              step={gridSize}
              parse={unitParse}
              className="w-full"
              disabled={disabled}
            />
            {player && (
              <Button
                variant="secondary"
                size="sm"
                disabled={disabled || player.currentTimeSec == null || !unitRate}
                onClick={() => {
                  if (player.currentTimeSec != null && unitRate) {
                    const maxTrimStart = Math.max(0, rawSegmentLength - trimEnd - minSegmentLength);
                    const newTrimStart = clamp(roundToGrid(player.currentTimeSec * unitRate, gridSize), 0, maxTrimStart);
                    onTrimStartChange(newTrimStart);
                  }
                }}
                title="Set trim start to the input video's current playhead position"
              >
                ↓
              </Button>
            )}
          </div>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Trim end ({unitLabel})</label>
          <NumberInput
            value={trimEnd}
            onCommit={(v) => onTrimEndChange(clamp(roundToGrid(v, gridSize), 0, Math.max(0, rawSegmentLength - trimStart - minSegmentLength)))}
            min={0}
            max={Math.max(0, rawSegmentLength - trimStart - minSegmentLength)}
            step={gridSize}
            parse={unitParse}
            className="w-full"
            disabled={disabled}
          />
        </div>
      </div>

      <p className="text-xs text-gray-500">
        Placed segment: {formatUnit(segmentLength)} of {formatUnit(rawSegmentLength)} input, at {formatUnit(offset)}
        {" "}of {formatUnit(safeTotalUnits)} total. Drag the block to move it, or its edges to trim.
      </p>

      {/*
        Outer host for FramePreviewTooltip: it positions itself with a
        negative top offset to float above the track, so it needs a
        `relative` ancestor that is NOT `overflow-hidden` -- the track div
        itself is `overflow-hidden` (it clips the tick marks/block to its
        rounded corners), which clipped the tooltip away entirely. This
        wrapper is the same width as the track (no padding/border of its
        own), so `leftPercent` still lines up with the track's own pixels.
      */}
      <div className="relative">
        <div
          ref={containerRef}
          className="relative h-14 bg-gray-800 border border-gray-600 rounded overflow-hidden touch-none select-none"
          onPointerMove={onTrackPointerMove}
          onPointerLeave={onTrackPointerLeave}
        >
          {/* Tick marks -- positioned at each tick's own percentage of the
              track (not spread evenly with `justify-between`), so it lines
              up with the ruler line it labels instead of being evenly spaced
              regardless of where `totalUnits` actually falls. The last tick
              sits at 100% and is anchored by its RIGHT edge (not left) so
              its label stays inside the track instead of being clipped by
              `overflow-hidden` above. */}
          <div className="absolute inset-0 pointer-events-none">
            {ticks.map((t, i) => {
              const isLast = i === ticks.length - 1;
              return (
                <div key={i} className="absolute top-0 bottom-0 border-l border-gray-700/60" style={{ left: `${percentForValue(t, { min: 0, max: safeTotalUnits })}%` }}>
                  <span
                    className={`absolute bottom-0 text-[10px] text-gray-500 whitespace-nowrap ${isLast ? "right-0.5" : "left-0.5"}`}
                  >
                    {formatTick(t)}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Placed segment block */}
          <div
            onPointerDown={startDrag("move")}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            className={`absolute top-2 bottom-2 border-2 border-blue-500 bg-blue-500/20 rounded ${
              disabled ? "cursor-not-allowed" : dragMode === "move" ? "cursor-grabbing" : "cursor-grab"
            }`}
            style={{ left: `${leftPct}%`, width: `${Math.max(0.5, widthPct)}%` }}
            title="Click to seek the input video here; drag to move the placed segment"
          >
            {/* Left trim handle */}
            <div
              role="slider"
              tabIndex={disabled ? -1 : 0}
              aria-label="Trim start"
              aria-valuemin={0}
              aria-valuemax={Math.max(0, rawSegmentLength - trimEnd - minSegmentLength)}
              aria-valuenow={trimStart}
              onPointerDown={startDrag("trim-left")}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              onKeyDown={(e) => {
                if (disabled) return;
                if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
                e.preventDefault();
                const step = (e.key === "ArrowRight" ? 1 : -1) * gridSize * (e.shiftKey ? SHIFT_STEP_MULTIPLIER : 1);
                const maxTrimStart = Math.max(0, rawSegmentLength - trimEnd - minSegmentLength);
                const newTrimStart = clamp(trimStart + step, 0, maxTrimStart);
                const delta = newTrimStart - trimStart;
                onTrimStartChange(newTrimStart);
                onOffsetChange(clamp(offset + delta, 0, Math.max(0, safeTotalUnits)));
              }}
              className="absolute top-0 bottom-0 left-0 w-2 -ml-1 bg-blue-400 hover:bg-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-300 cursor-ew-resize"
              title="Drag, or use the arrow keys, to trim the start of the input clip (Shift steps 4x the grid)"
            />
            {/* Right trim handle */}
            <div
              role="slider"
              tabIndex={disabled ? -1 : 0}
              aria-label="Trim end"
              aria-valuemin={0}
              aria-valuemax={Math.max(0, rawSegmentLength - trimStart - minSegmentLength)}
              aria-valuenow={trimEnd}
              onPointerDown={startDrag("trim-right")}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              onKeyDown={(e) => {
                if (disabled) return;
                if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
                e.preventDefault();
                // Dragging the right handle right REDUCES trimEnd (mirrors the pointer mapping above).
                const step = (e.key === "ArrowRight" ? -1 : 1) * gridSize * (e.shiftKey ? SHIFT_STEP_MULTIPLIER : 1);
                const maxTrimEnd = Math.max(0, rawSegmentLength - trimStart - minSegmentLength);
                onTrimEndChange(clamp(trimEnd + step, 0, maxTrimEnd));
              }}
              className="absolute top-0 bottom-0 right-0 w-2 -mr-1 bg-blue-400 hover:bg-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-300 cursor-ew-resize"
              title="Drag, or use the arrow keys, to trim the end of the input clip (Shift steps 4x the grid)"
            />

            {/* Playhead: the input video's own live position, mapped into this block. */}
            {playheadPctInBlock != null && (
              <div
                className="absolute top-0 bottom-0 w-[2px] bg-emerald-400 pointer-events-none"
                style={{ left: `${playheadPctInBlock}%` }}
              />
            )}
          </div>
        </div>

        {previewUnit != null && (
          <FramePreviewTooltip
            videoSrc={videoSrc}
            timeSec={unitRate && unitRate > 0 ? previewUnit / unitRate : 0}
            leftPercent={clamp(((offset + (previewUnit - trimStart)) / safeTotalUnits) * 100, 0, 100)}
            label={unitRate && unitRate > 0 ? `${Math.round(previewUnit)} ${unitLabel} · ${formatTimecode(previewUnit / unitRate, unitRate)}` : `${Math.round(previewUnit)} ${unitLabel}`}
            visible={previewUnit != null}
          />
        )}
      </div>

      {player && unitRate && unitRate > 0 && (
        <div className="flex items-center gap-2">
          <Button
            variant={loopEnabled ? "primary" : "secondary"}
            size="sm"
            disabled={disabled}
            onClick={toggleLoop}
            title="Loop the input video over the placed (trimmed) segment"
          >
            {loopEnabled ? "Stop loop" : "Loop placed segment"}
          </Button>
        </div>
      )}
    </div>
  );
}
