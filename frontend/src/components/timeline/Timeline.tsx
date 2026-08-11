"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { cn } from "@/lib/utils";
import { percentForValue, roundedValueAtClientX } from "@/utils/timelineScale";
import { formatFrameLabel, formatTimecode } from "@/utils/timecode";

// ---------------------------------------------------------------------------
// Minimal shared shell for a horizontal, frame-indexed timeline made of one
// or more stacked TRACKS (plain children, not a plugin/registry abstraction
// -- each track owns its own markup and interaction, e.g. drag handles or
// keyframe markers). This module owns exactly three things, all shared
// across every track stacked inside one <Timeline>:
//
//   1. The ruler (frame/timecode tick labels along the shared domain).
//   2. The playhead line (a single vertical line spanning every track).
//   3. Click/drag-to-seek on empty track space, plus the pointer<->frame
//      conversion every track needs to position its own content -- exposed
//      via `useTimelineContext()` so a track never re-derives clientX<->frame
//      math independently (that arithmetic itself still lives in
//      `utils/timelineScale.ts`; this module only wires it to one pointer
//      surface shared by every stacked track).
//
// Deliberately NOT here: per-track drag behavior (handles, markers,
// keyframes), thumbnails/tooltips, zoom/scroll (the domain is always the
// whole clip; `timelineScale`'s `[min,max]` already leaves room for a future
// zoomed view without this module changing), and anything about what a
// track's VALUE means (pixel frames vs. latent groups vs. seconds is each
// track's own business -- this module only ever deals in "frame" units of
// the single domain passed to it).
// ---------------------------------------------------------------------------

export interface TimelineDomain {
  /** The frame at the LEFT edge of every stacked track and the ruler. */
  min: number;
  /** The frame at the RIGHT edge of every stacked track and the ruler. */
  max: number;
}

export interface TimelineContextValue {
  domain: TimelineDomain;
  frameRate: number;
  /**
   * The frame currently under the pointer while it is hovering (button up)
   * or scrubbing (button down, dragging on empty track space) the shared
   * container -- null when the pointer is elsewhere, or while a track's own
   * interactive element (a handle, a marker) has captured the pointer and
   * called `stopPropagation()` on it. Tracks that want a hover preview (e.g.
   * a frame thumbnail) read this instead of tracking their own pointer
   * state for the shared surface; a track dragging its OWN element (which
   * stopPropagation()s before this can update) still owns its own preview
   * state for that interaction, same as before.
   */
  hoverFrame: number | null;
  /**
   * clientX (from ANY pointer event, in ANY stacked track, since they all
   * share one coordinate system) -> the frame under it, rounded and clamped
   * to `domain`, using the shared container's own rect. A track calls this
   * from its own drag handlers (e.g. a range handle, a keyframe marker) so
   * that math is always taken from the SAME rect the ruler/playhead use.
   */
  frameAtClientX: (clientX: number) => number;
  /** frame -> percent (0-100) along `domain`, for a track's own `left: X%`. */
  percentForFrame: (frame: number) => number;
}

const TimelineContext = createContext<TimelineContextValue | null>(null);

/** Call from any track rendered as a child of `<Timeline>`. */
export function useTimelineContext(): TimelineContextValue {
  const ctx = useContext(TimelineContext);
  if (!ctx) {
    throw new Error("useTimelineContext() must be called from a track rendered inside <Timeline>.");
  }
  return ctx;
}

export interface TimelineProps {
  domain: TimelineDomain;
  frameRate: number;
  /** The live playhead position, in the SAME domain as every stacked track. Null/omitted = no playhead line. */
  playheadFrame?: number | null;
  /**
   * Called while the pointer is down over empty track/ruler space (i.e. not
   * stopPropagation()'d by a track's own interactive element) as it moves,
   * and once immediately on press. Omit to disable click/drag-to-seek
   * entirely (e.g. before a clip is loaded).
   */
  onSeek?: (frame: number) => void;
  disabled?: boolean;
  /** Ticks drawn along the ruler, in addition to both edges. Default 3 (4 labels total, matching the prior single-track ruler). */
  tickCount?: number;
  className?: string;
  /** Stacked tracks (and any other markup) sharing this timeline's domain/ruler/playhead. */
  children: ReactNode;
}

export default function Timeline({
  domain,
  frameRate,
  playheadFrame = null,
  onSeek,
  disabled = false,
  tickCount = 3,
  className,
  children,
}: TimelineProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoverFrame, setHoverFrame] = useState<number | null>(null);
  const scrubbingRef = useRef(false);

  const frameAtClientX = useCallback(
    (clientX: number): number => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect || rect.width <= 0) return domain.min;
      return roundedValueAtClientX(clientX, rect, domain);
    },
    [domain],
  );
  const percentForFrame = useCallback((frame: number) => percentForValue(frame, domain), [domain]);

  // Memoized so a `hoverFrame` update (on every pointermove while hovering
  // or scrubbing) does not hand every stacked track a NEW context object --
  // without this, each track's own keyframe/marker list re-renders on every
  // pointermove even when nothing about ITS markers changed, which scales
  // with marker count on a dense timeline (e.g. a 128-keyframe mask track).
  const ctx: TimelineContextValue = useMemo(
    () => ({ domain, frameRate, hoverFrame, frameAtClientX, percentForFrame }),
    [domain, frameRate, hoverFrame, frameAtClientX, percentForFrame],
  );

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled || !onSeek) return;
    // A track's own interactive element (handle/marker) stopPropagation()s
    // its own pointerdown before this fires, so this only ever runs for a
    // press on genuinely empty track/ruler space.
    e.currentTarget.setPointerCapture(e.pointerId);
    scrubbingRef.current = true;
    const frame = frameAtClientX(e.clientX);
    setHoverFrame(frame);
    onSeek(frame);
  };
  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const frame = frameAtClientX(e.clientX);
    setHoverFrame(frame);
    if (scrubbingRef.current && onSeek) onSeek(frame);
  };
  const stopScrub = () => {
    scrubbingRef.current = false;
    // A pointer released (or cancelled) outside the container never fires
    // `onPointerLeave` on it (the capture set by `handlePointerDown`
    // delivers the up/cancel event here regardless of where the pointer
    // physically is), so without this a frame-thumbnail tooltip driven by
    // `hoverFrame` stays pinned to the last scrubbed frame indefinitely.
    setHoverFrame(null);
  };
  const handlePointerLeave = () => {
    if (!scrubbingRef.current) setHoverFrame(null);
  };

  const seconds = (frame: number) => (frameRate > 0 ? frame / frameRate : 0);
  const span = domain.max - domain.min;
  const ticks =
    span > 0
      ? Array.from({ length: tickCount + 1 }, (_, i) => Math.round(domain.min + (span * i) / tickCount))
      : [domain.min];
  const playheadVisible =
    playheadFrame != null && playheadFrame >= domain.min && playheadFrame <= domain.max;

  return (
    <TimelineContext.Provider value={ctx}>
      <div className={cn("space-y-1", className)}>
        <div
          ref={containerRef}
          className={cn("relative touch-none select-none", disabled && "pointer-events-none")}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={stopScrub}
          onPointerCancel={stopScrub}
          onPointerLeave={handlePointerLeave}
        >
          <div className="space-y-1">{children}</div>

          {/* One playhead line spanning every stacked track above. */}
          {playheadVisible && (
            <div
              className="pointer-events-none absolute inset-y-0 z-10 w-[2px] bg-emerald-400"
              style={{ left: `${percentForFrame(playheadFrame!)}%` }}
              title={`Player: ${formatFrameLabel(playheadFrame!, frameRate)}`}
            />
          )}
        </div>

        {/* Shared ruler: one row of frame/timecode labels below every track,
            positioned by each tick's own frame value (not evenly spaced) so
            a label's position actually indicates its value. */}
        <div className="relative h-3">
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
                    : { left: `${percentForFrame(t)}%`, transform: "translateX(-50%)" }
                }
              >
                {t} · {formatTimecode(seconds(t), frameRate)}
              </span>
            );
          })}
        </div>
      </div>
    </TimelineContext.Provider>
  );
}
