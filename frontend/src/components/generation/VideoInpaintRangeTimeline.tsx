"use client";

import { useRef, useState } from "react";
import NumberInput from "../common/NumberInput";
import InlineHelp from "../common/InlineHelp";
import { latentGroupSpans, snapRangeToLatentGroups } from "@/utils/api";

// ---------------------------------------------------------------------------
// Range control for VIDEO TEMPORAL INPAINT (POST /generate/inpaint/video).
//
// Deliberately NOT MiniMaxH3KeyframeTimeline and NOT OutpaintTimeline. A
// keyframe chip addresses one pixel frame, which is honest for an anchor; this
// control selects a range whose unit is a LATENT GROUP of 1 or 4 pixel frames
// (`video_constraints[arch].latent_chunk_pattern`), because a latent frame is
// regenerated or preserved as a whole. Both handles therefore sit on group
// boundaries, the groups are drawn, and the readout is the span the server will
// run -- so the backend's outward expansion never fires for a request built
// here. OutpaintTimeline's model is "place/trim a segment in a longer
// timeline" with a uniform grid, which is neither the geometry nor the snap.
//
// Coordinates: `start`/`end` are pixel frames of the TRIMMED clip (start
// inclusive, end exclusive), which is what the route takes. The track draws the
// whole upload, with the trimmed head/tail greyed out, so what the trim removes
// is visible rather than implied.
// ---------------------------------------------------------------------------

export interface VideoInpaintRangeTimelineProps {
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
  disabled?: boolean;
}

type DragMode = "start" | "end" | null;

export default function VideoInpaintRangeTimeline({
  rawFrames,
  trimStart,
  trimEnd,
  latentChunkPattern,
  start,
  end,
  onRangeChange,
  frameRate,
  disabled = false,
}: VideoInpaintRangeTimelineProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dragMode, setDragMode] = useState<DragMode>(null);

  const safeRaw = Math.max(1, rawFrames);
  const trimmedFrames = Math.max(0, rawFrames - trimStart - trimEnd);
  const groups = latentGroupSpans(latentChunkPattern, trimmedFrames);
  // Every legal handle position: the start of the clip plus every group end.
  const bounds = groups.length ? [0, ...groups.map(([, hi]) => hi)] : [];

  const effective = snapRangeToLatentGroups(groups, start, end);
  const selectedGroups = groups.filter(([lo, hi]) => lo < effective.end && hi > effective.start).length;

  const nearestBound = (frame: number): number =>
    bounds.reduce((best, b) => (Math.abs(frame - b) < Math.abs(frame - best) ? b : best), bounds[0]);

  // Both handles stop one group short of covering the clip: a range that
  // preserves nothing is refused by the route (that request is /generate/txt2vid),
  // so the control cannot express it rather than discovering it at generate time.
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

  const frameAtPointer = (clientX: number): number => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return 0;
    const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    // Pointer space is the WHOLE upload; the range is in trimmed-clip frames.
    return Math.round(fraction * safeRaw) - trimStart;
  };

  const startDrag = (mode: DragMode) => (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled || !bounds.length) return;
    e.preventDefault();
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    setDragMode(mode);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragMode) return;
    const frame = frameAtPointer(e.clientX);
    if (dragMode === "start") commitStart(frame);
    else commitEnd(frame);
  };

  const onPointerUp = () => setDragMode(null);

  const pct = (frame: number) => ((trimStart + frame) / safeRaw) * 100;
  const seconds = (frame: number) => (frameRate > 0 ? frame / frameRate : 0);
  const fmt = (frame: number) => `${frame} (${seconds(frame).toFixed(2)}s)`;

  const tickCount = 5;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => Math.round((safeRaw * i) / tickCount));

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Regenerate from (frame)</label>
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
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Regenerate to (frame, exclusive)</label>
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
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative h-16 bg-gray-800 border border-gray-600 rounded overflow-hidden touch-none select-none"
      >
        {/* Latent-group boundaries: the only positions the handles can take. */}
        <div className="absolute inset-0 pointer-events-none">
          {groups.map(([lo], index) => (
            <div
              key={index}
              className="absolute top-0 bottom-0 border-l border-gray-700/70"
              style={{ left: `${pct(lo)}%` }}
            />
          ))}
        </div>

        {/* Trimmed head/tail: part of the upload, not part of the request. */}
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

        {/* Preserved region label (the whole trimmed clip minus the range). */}
        <span className="absolute left-1 top-1 text-[10px] text-gray-400 pointer-events-none">
          Preserved
        </span>

        {bounds.length > 0 && (
          <div
            className="absolute top-4 bottom-4 border-2 border-amber-500 bg-amber-500/25 rounded"
            style={{
              left: `${pct(effective.start)}%`,
              width: `${Math.max(0.5, ((effective.end - effective.start) / safeRaw) * 100)}%`,
            }}
            title="Regenerate: this span is generated; everything else is the input's own pixels"
          >
            <span className="absolute left-1 top-0 text-[10px] text-amber-200 pointer-events-none">
              Regenerate
            </span>
            <div
              onPointerDown={startDrag("start")}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              className={`absolute top-0 bottom-0 left-0 w-2 -ml-1 bg-amber-400 hover:bg-amber-300 ${
                disabled ? "cursor-not-allowed" : "cursor-ew-resize"
              }`}
              title="Drag to move the start of the regenerated range (snaps to latent-group boundaries)"
            />
            <div
              onPointerDown={startDrag("end")}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              className={`absolute top-0 bottom-0 right-0 w-2 -mr-1 bg-amber-400 hover:bg-amber-300 ${
                disabled ? "cursor-not-allowed" : "cursor-ew-resize"
              }`}
              title="Drag to move the end of the regenerated range (snaps to latent-group boundaries)"
            />
          </div>
        )}

        <div className="absolute inset-x-0 bottom-0 flex justify-between pointer-events-none">
          {ticks.map((t, i) => (
            <span key={i} className="text-[10px] text-gray-500 px-0.5">
              {t}
            </span>
          ))}
        </div>
      </div>

      {bounds.length > 0 ? (
        <p className="text-xs text-gray-500">
          Regenerate frames {fmt(effective.start)} to {fmt(effective.end)} of the trimmed clip —{" "}
          {effective.end - effective.start} frame(s), {selectedGroups} of {groups.length} latent groups.
          Preserved: {trimmedFrames - (effective.end - effective.start)} frame(s).
        </p>
      ) : (
        <p className="text-xs text-gray-500">
          Load a clip to choose a range.
        </p>
      )}
      <div className="flex items-center gap-1 text-xs text-gray-500">
        <span>Handles snap to latent-group boundaries</span>
        <InlineHelp label="Temporal inpaint range details">
          <p>
            The video VAE processes groups of up to four frames, so each group is regenerated or preserved as a unit.
          </p>
          <p>
            Preserved pixels are pasted back after decode while their re-encoded latents condition the selected range. A boundary seam may remain visible depending on the clip.
          </p>
          <p>The control keeps at least one group preserved; replacing the full clip is a text-to-video request.</p>
        </InlineHelp>
      </div>
    </div>
  );
}
