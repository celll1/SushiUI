"use client";

import { useRef, useState } from "react";
import NumberInput from "../common/NumberInput";

// ---------------------------------------------------------------------------
// Generic horizontal timeline widget for TEMPORAL outpaint placement.
//
// Modality-neutral by design (reused by the video branch here, and intended
// for the audio branch in Phase 3): everything is expressed in an abstract
// "unit" space (video: frames; audio would be seconds or 1/25s ticks) plus an
// optional `unitRate` (units-per-second) purely to render a seconds ruler.
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
// ---------------------------------------------------------------------------

export interface OutpaintTimelineProps {
  totalUnits: number;
  onTotalUnitsChange: (v: number) => void;
  /** Optional hard constraint applied to totalUnits on commit (e.g. LTX-2.3's (n-1)%8==0). */
  totalUnitsSnapFn?: (v: number) => number;
  totalUnitsMin?: number;
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

export default function OutpaintTimeline({
  totalUnits,
  onTotalUnitsChange,
  totalUnitsSnapFn,
  totalUnitsMin = 1,
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
}: OutpaintTimelineProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dragMode, setDragMode] = useState<DragMode>(null);
  const dragStartRef = useRef<DragStart | null>(null);

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
    // Stored directly on the ref (not state) so the very first pointermove in
    // the same drag can read it synchronously -- a state setter's update
    // wouldn't be visible until the next render, causing the first pixel of
    // movement to be silently dropped.
    dragStartRef.current = { mouseX: e.clientX, offset, trimStart, trimEnd, widthPx: rect?.width || 1 };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragMode || !dragStartRef.current) return;
    const start = dragStartRef.current;
    const dxUnits = ((e.clientX - start.mouseX) / start.widthPx) * safeTotalUnits;

    if (dragMode === "move") {
      const newOffset = clamp(
        Math.round(start.offset + dxUnits),
        0,
        Math.max(0, safeTotalUnits - segmentLength)
      );
      onOffsetChange(newOffset);
      return;
    }

    if (dragMode === "trim-right") {
      // Right edge follows the pointer: growing dx (moving right) REDUCES trimEnd,
      // shrinking dx (moving left) INCREASES it. Left edge/offset unchanged.
      const maxTrimEnd = Math.max(0, rawSegmentLength - start.trimStart - minSegmentLength);
      const newTrimEnd = clamp(Math.round(start.trimEnd - dxUnits), 0, maxTrimEnd);
      onTrimEndChange(newTrimEnd);
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
      return;
    }
  };

  const onPointerUp = () => {
    if (!dragMode) return;
    setDragMode(null);
    dragStartRef.current = null;
    // Snap on release (grid + optional stricter offset rule).
    onOffsetChange(applyOffsetSnap(offset));
    onTrimStartChange(roundToGrid(trimStart, gridSize));
    onTrimEndChange(roundToGrid(trimEnd, gridSize));
  };

  const leftPct = (offset / safeTotalUnits) * 100;
  const widthPct = (segmentLength / safeTotalUnits) * 100;

  const formatUnit = (u: number): string => {
    if (unitRate && unitRate > 0) return `${(u / unitRate).toFixed(2)}s`;
    return `${Math.round(u)} ${unitLabel}`;
  };

  const tickCount = 5;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => (safeTotalUnits * i) / tickCount);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Total ({unitLabel})</label>
          <NumberInput
            value={totalUnits}
            onCommit={(v) => onTotalUnitsChange(totalUnitsSnapFn ? totalUnitsSnapFn(v) : Math.max(totalUnitsMin, v))}
            min={totalUnitsMin}
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

      <div
        ref={containerRef}
        className="relative h-14 bg-gray-800 border border-gray-600 rounded overflow-hidden touch-none select-none"
      >
        {/* Tick marks */}
        <div className="absolute inset-0 flex justify-between pointer-events-none">
          {ticks.map((t, i) => (
            <div key={i} className="h-full border-l border-gray-700/60 flex flex-col justify-end">
              <span className="text-[10px] text-gray-500 pl-0.5">{formatUnit(t)}</span>
            </div>
          ))}
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
          title="Drag to move the placed segment"
        >
          {/* Left trim handle */}
          <div
            onPointerDown={startDrag("trim-left")}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            className="absolute top-0 bottom-0 left-0 w-2 -ml-1 bg-blue-400 hover:bg-blue-300 cursor-ew-resize"
            title="Drag to trim the start of the input clip"
          />
          {/* Right trim handle */}
          <div
            onPointerDown={startDrag("trim-right")}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            className="absolute top-0 bottom-0 right-0 w-2 -mr-1 bg-blue-400 hover:bg-blue-300 cursor-ew-resize"
            title="Drag to trim the end of the input clip"
          />
        </div>
      </div>
    </div>
  );
}
