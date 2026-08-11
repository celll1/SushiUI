"use client";

import { useEffect, useRef, useState } from "react";
import Button from "../common/Button";
import NumberInput from "../common/NumberInput";
import InlineHelp from "../common/InlineHelp";
import FramePreviewTooltip from "../common/FramePreviewTooltip";
import { latentGroupSpans, snapRangeToLatentGroups } from "@/utils/api";
import { formatFrameLabel, formatTimecode } from "@/utils/timecode";
import type { VideoPlayheadState } from "@/hooks/useVideoPlayhead";

// ---------------------------------------------------------------------------
// Range control for VIDEO TEMPORAL INPAINT (POST /generate/inpaint/video).
//
// Deliberately NOT MiniMaxH3KeyframeTimeline and NOT OutpaintTimeline: this
// control's unit is a LATENT GROUP of 1 or 4 pixel frames
// (`video_constraints[arch].latent_chunk_pattern`), since a latent frame is
// regenerated or preserved as a whole -- both handles sit on group
// boundaries and the readout is exactly the span the server will run.
//
// Coordinates: `start`/`end` are pixel frames of the TRIMMED clip (start
// inclusive, end exclusive), which is what the route takes. The track draws
// the whole upload, with the trimmed head/tail greyed out.
//
// `player`/`videoSrc` (both optional) wire this track to the panel's own
// input <video>: a playhead line, click/drag-to-seek on the empty track, and
// a frame-thumbnail preview while hovering or dragging a handle. Omitting
// them degrades to the prior range-only behavior.
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
  /** Object/file URL of the uploaded clip, for hover/drag frame previews and seek-on-click. Null/omitted = no preview, no playhead, no seek (e.g. before a clip is loaded). */
  videoSrc?: string | null;
  /** The SAME input <video>'s live playhead (from `useVideoPlayhead`), in RAW clip frames/seconds. Omitted = degrades the same as videoSrc absent. */
  player?: VideoPlayheadState;
}

type DragMode = "start" | "end" | null;

// Above this many latent groups, per-boundary hairlines are decimated so the
// track does not turn into an undifferentiated grey wall on a long clip.
const DENSE_GROUP_THRESHOLD = 60;

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
  videoSrc = null,
  player,
}: VideoInpaintRangeTimelineProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dragMode, setDragMode] = useState<DragMode>(null);
  const [trackScrubbing, setTrackScrubbing] = useState(false);
  const [previewRawFrame, setPreviewRawFrame] = useState<number | null>(null);

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
  // The adjacent legal boundary in `dir` (-1/+1) from a currently-on-grid
  // value -- the finest step a handle can actually take, since it may only
  // ever sit on a group boundary. `stride` jumps that many boundaries at
  // once (the Shift-modified step). `frame` and the return value are both in
  // TRIMMED-clip space, matching `bounds` -- callers must not add/subtract
  // `trimStart` around this call.
  const adjacentBound = (frame: number, dir: -1 | 1, stride: number): number => {
    if (!bounds.length) return frame;
    const idx = bounds.indexOf(nearestBound(frame));
    const next = bounds[Math.max(0, Math.min(bounds.length - 1, idx + dir * stride))];
    return next ?? frame;
  };

  const frameAtPointer = (clientX: number): number => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return 0;
    const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    // Pointer space is the WHOLE upload; the range is in trimmed-clip frames.
    return Math.round(fraction * safeRaw) - trimStart;
  };
  const rawFrameAtPointer = (clientX: number): number => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return 0;
    const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    return Math.round(fraction * safeRaw);
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
    setPreviewRawFrame(Math.max(0, Math.min(safeRaw - 1, frame + trimStart)));
    if (dragMode === "start") commitStart(frame);
    else commitEnd(frame);
  };

  const onPointerUp = () => {
    setDragMode(null);
    setPreviewRawFrame(null);
  };

  // Click/drag on the EMPTY track (not a handle -- handles stopPropagation
  // above) seeks the input video. A plain hover (no button down) only shows
  // the frame preview; it does not move the player.
  const onTrackPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled || !player || !bounds.length) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const raw = rawFrameAtPointer(e.clientX);
    setTrackScrubbing(true);
    setPreviewRawFrame(raw);
    player.seekToFrame(raw);
  };
  const onTrackPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (dragMode) return;
    const raw = rawFrameAtPointer(e.clientX);
    setPreviewRawFrame(raw);
    if (trackScrubbing && player) player.seekToFrame(raw);
  };
  const onTrackPointerUp = () => {
    setTrackScrubbing(false);
    if (!dragMode) setPreviewRawFrame(null);
  };
  const onTrackPointerLeave = () => {
    if (!dragMode && !trackScrubbing) setPreviewRawFrame(null);
  };

  const toggleLoop = () => {
    if (!player) return;
    if (player.isLooping) {
      player.setLoopRange(null);
      return;
    }
    const fps = frameRate > 0 ? frameRate : 24;
    const startSec = (trimStart + effective.start) / fps;
    const endSec = (trimStart + effective.end) / fps;
    player.setLoopRange({ startSec, endSec });
    player.seekToSeconds(startSec);
    player.play();
  };

  // Dragging a handle while the loop is armed must not leave the player
  // looping the OLD span while the UI shows the new one.
  useEffect(() => {
    if (!player || !player.isLooping) return;
    const fps = frameRate > 0 ? frameRate : 24;
    player.setLoopRange({ startSec: (trimStart + effective.start) / fps, endSec: (trimStart + effective.end) / fps });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [player?.isLooping, player?.setLoopRange, trimStart, effective.start, effective.end, frameRate]);

  const pct = (frame: number) => ((trimStart + frame) / safeRaw) * 100;
  const rawPct = (rawFrame: number) => (rawFrame / safeRaw) * 100;
  const seconds = (frame: number) => (frameRate > 0 ? frame / frameRate : 0);
  const fmt = (frame: number) => `${frame} (${formatTimecode(seconds(frame), frameRate)})`;

  // Fewer, absolutely-positioned ticks: at tickCount+1=6 labels (the prior
  // count) with the "frame · timecode" text, adjacent labels overlapped on a
  // narrow track. 4 leaves enough room for the widened label.
  const tickCount = 3;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => Math.round((safeRaw * i) / tickCount));

  const denseGroups = groups.length > DENSE_GROUP_THRESHOLD;
  const decimateStep = denseGroups ? Math.ceil(groups.length / DENSE_GROUP_THRESHOLD) : 1;

  const playheadVisible = player?.currentFrame != null && player.currentFrame >= 0 && player.currentFrame < safeRaw;

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

      {/* Outer wrapper is `relative` but NOT `overflow-hidden` -- it hosts
          the frame-preview tooltip, which floats above the track. The track
          itself stays `overflow-hidden` (it clips the group hairlines/fill
          bars to its rounded corners) but must not also host the tooltip,
          since that would clip it away too (it lies above the track's own
          box). Both share the same width, so percentages line up. */}
      <div className="relative">
      <div
        ref={containerRef}
        className="relative h-16 bg-gray-800 border border-gray-600 rounded overflow-hidden touch-none select-none"
        onPointerDown={onTrackPointerDown}
        onPointerMove={onTrackPointerMove}
        onPointerUp={onTrackPointerUp}
        onPointerCancel={onTrackPointerUp}
        onPointerLeave={onTrackPointerLeave}
      >
        {/* Latent-group boundaries: the only positions the handles can take.
            Decimated past DENSE_GROUP_THRESHOLD groups so a long clip does not
            render as a wall of identical hairlines; the first group (often a
            different width from the rest, e.g. MiniMax-H3's leading 1-frame
            group) is always drawn and shaded so it stays visually distinct. */}
        <div className="absolute inset-0 pointer-events-none">
          {groups.map(([lo], index) => {
            if (index !== 0 && index % decimateStep !== 0) return null;
            return (
              <div
                key={index}
                className="absolute top-0 bottom-0 border-l border-gray-700/70"
                style={{ left: `${pct(lo)}%` }}
              />
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
              role="slider"
              tabIndex={disabled || !bounds.length ? -1 : 0}
              aria-label="Regenerate range start"
              aria-valuemin={0}
              aria-valuemax={bounds.length ? bounds[bounds.length - 2] : 0}
              aria-valuenow={effective.start}
              aria-valuetext={fmt(effective.start)}
              onPointerDown={startDrag("start")}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              onKeyDown={(e) => {
                if (disabled || !bounds.length) return;
                if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                  e.preventDefault();
                  const dir = e.key === "ArrowLeft" ? -1 : 1;
                  commitStart(adjacentBound(effective.start, dir, e.shiftKey ? 4 : 1));
                } else if (e.key === "PageDown" || e.key === "PageUp") {
                  e.preventDefault();
                  const dir = e.key === "PageDown" ? -1 : 1;
                  commitStart(adjacentBound(effective.start, dir, 4));
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
              aria-valuemax={trimmedFrames}
              aria-valuenow={effective.end}
              aria-valuetext={fmt(effective.end)}
              onPointerDown={startDrag("end")}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              onKeyDown={(e) => {
                if (disabled || !bounds.length) return;
                if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                  e.preventDefault();
                  const dir = e.key === "ArrowLeft" ? -1 : 1;
                  commitEnd(adjacentBound(effective.end, dir, e.shiftKey ? 4 : 1));
                } else if (e.key === "PageDown" || e.key === "PageUp") {
                  e.preventDefault();
                  const dir = e.key === "PageDown" ? -1 : 1;
                  commitEnd(adjacentBound(effective.end, dir, 4));
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

        {/* Playhead: the input video's own live position, in raw-clip space. */}
        {playheadVisible && player?.currentFrame != null && (
          <div
            className="absolute top-0 bottom-0 w-[2px] bg-emerald-400 pointer-events-none"
            style={{ left: `${rawPct(player.currentFrame)}%` }}
            title={`Player: ${formatFrameLabel(player.currentFrame, frameRate)}`}
          />
        )}

        {/* Positioned by each tick's own raw-frame value (not evenly spaced)
            so a label's position on the track actually indicates its value.
            Ticks are RAW-clip frames throughout -- the timecode uses the
            same raw second as the frame number, matching the playhead. */}
        <div className="absolute inset-x-0 bottom-0 h-3 pointer-events-none">
          {ticks.map((t, i) => {
            const isFirst = i === 0;
            const isLast = i === ticks.length - 1;
            return (
              <span
                key={i}
                className="absolute text-[10px] text-gray-500 px-0.5 whitespace-nowrap"
                style={
                  isFirst
                    ? { left: 0 }
                    : isLast
                    ? { right: 0 }
                    : { left: `${rawPct(t)}%`, transform: "translateX(-50%)" }
                }
              >
                {t} · {formatTimecode(seconds(t), frameRate)}
              </span>
            );
          })}
        </div>
      </div>

      {previewRawFrame != null && (
        <FramePreviewTooltip
          videoSrc={videoSrc}
          timeSec={seconds(previewRawFrame)}
          leftPercent={rawPct(previewRawFrame)}
          label={formatFrameLabel(previewRawFrame, frameRate)}
          visible={previewRawFrame != null}
        />
      )}
      </div>

      {bounds.length > 0 ? (
        <p className="text-xs text-gray-500">
          Regenerate frames {fmt(effective.start)} to {fmt(effective.end)} of the trimmed clip —{" "}
          {effective.end - effective.start} frame(s), {selectedGroups} of {groups.length} latent groups.
          Preserved: {trimmedFrames - (effective.end - effective.start)} frame(s).
          {denseGroups && ` Group boundary lines are shown every ${decimateStep} groups on this clip.`}
        </p>
      ) : (
        <p className="text-xs text-gray-500">
          Load a clip to choose a range.
        </p>
      )}
      <div className="flex items-center gap-2 text-xs text-gray-500 flex-wrap">
        <span>
          Handles snap to latent-group boundaries
          {latentChunkPattern.length > 0 && ` (pattern repeats every ${latentChunkPattern.length} group(s): ${latentChunkPattern.join(", ")} frame(s) each)`}
        </span>
        <InlineHelp label="Temporal inpaint range details">
          <p>
            The video VAE processes groups of up to four frames, so each group is regenerated or preserved as a unit.
          </p>
          <p>
            Preserved pixels are pasted back after decode while their re-encoded latents condition the selected range. A boundary seam may remain visible depending on the clip.
          </p>
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
    </div>
  );
}
