"use client";

import { ChangeEvent, WheelEvent } from "react";
import Select from "./Select";
import NumberInput from "./NumberInput";
import {
  ArchCapabilities,
  isValidVideoFrameCount,
  largestValidVideoFrameCount,
  nearestValidVideoFrameCount,
  videoFrameLabel,
  videoFrameOptions,
} from "@/utils/api";

interface VideoFrameCountSliderProps {
  caps: ArchCapabilities | null | undefined;
  arch: string | null | undefined;
  /** Currently held clip length, in frames. */
  value: number;
  onChange: (frames: number) => void;
  /**
   * Frame rate to use for the derived seconds readout when the loaded arch
   * does not fix its own (`fps_fixed`). Callers pass `params.frame_rate`.
   */
  fallbackFps: number;
  disabled?: boolean;
  className?: string;
}

// How far the SLIDER TRACK reaches on an architecture that declares no
// `max_frames` (LTX-2.3). A range input has to end somewhere; that is a
// drawing constraint, not an architecture fact -- the backend imposes no
// upper bound there beyond the frame grid. So this bounds the track only:
// the number box stays unbounded on such an architecture, and typing past
// the track extends it (see `rawCeiling`). Same convention as api.ts's
// `UNCAPPED_VIDEO_EDGE` for the canvas sliders.
const UNCAPPED_FRAME_SLIDER_CEILING = 241;

/**
 * Any valid clip length on the loaded architecture's frame grid, not just the
 * `suggested_frames` a `<Select>` can offer: a slider dragged to any point,
 * plus a paired numeric box, both snapping onto the grid (`frame_multiple *
 * n + frame_offset`) only at commit time -- release for the slider, blur/
 * live-parse for the number box -- never mid-drag-tick or mid-keystroke in a
 * way that fights what the user is doing. The 17n+5-style arithmetic itself
 * lives once, in `nearestValidVideoFrameCount`/`largestValidVideoFrameCount`
 * (api.ts); this component only calls those, it does not restate the grid.
 *
 * When the loaded architecture is not known yet (no model loaded, or the
 * capability matrix has not loaded), this falls back to the historical
 * `<Select>` + `videoFrameOptions()` control rather than guessing bounds for
 * an unconstrained slider -- the same "assume supported, let the backend
 * re-validate" posture as `archSupportsFeature`, applied to keep today's
 * behaviour rather than showing a slider with no real grid to snap to.
 */
export default function VideoFrameCountSlider({
  caps,
  arch,
  value,
  onChange,
  fallbackFps,
  disabled,
  className = "",
}: VideoFrameCountSliderProps) {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;

  if (!c) {
    return (
      <Select
        label={videoFrameLabel(caps, arch)}
        value={String(value)}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
        options={videoFrameOptions(caps, arch, value ?? null)}
        disabled={disabled}
        className={className}
      />
    );
  }

  const fps = c.fps_fixed ?? fallbackFps;
  const min = c.min_frames;
  const rawCeiling = c.max_frames ?? Math.max(UNCAPPED_FRAME_SLIDER_CEILING, value);
  const max = largestValidVideoFrameCount(caps, arch, rawCeiling) ?? rawCeiling;
  const step = c.frame_multiple;
  // Clamp only what the native <input type="range"> needs to stay in
  // [min, max] to render without a browser warning; the readout and the
  // number box always show the true `value`, off-grid or not.
  const sliderValue = Math.min(Math.max(value, min), max);

  const snap = (frames: number): number => nearestValidVideoFrameCount(caps, arch, frames) ?? frames;

  const handleRangeChange = (e: ChangeEvent<HTMLInputElement>) => {
    const raw = parseInt(e.target.value, 10);
    if (!Number.isFinite(raw)) return;
    onChange(snap(raw));
  };

  const handleWheel = (e: WheelEvent<HTMLInputElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const delta = e.deltaY < 0 ? step : -step;
    const ceiling = c.max_frames ?? Number.POSITIVE_INFINITY;
    onChange(snap(Math.max(min, Math.min(ceiling, value + delta))));
  };

  const isOnGrid = isValidVideoFrameCount(caps, arch, value);

  return (
    <div className={className}>
      <div className="mb-1 flex items-center justify-between">
        <label className="block text-xs font-medium text-gray-400">
          {videoFrameLabel(caps, arch)}
        </label>
        <span className="text-xs font-mono text-gray-400">
          {value} frames{fps > 0 ? ` · ${(value / fps).toFixed(2)}s` : ""}
        </span>
      </div>
      <div className="flex items-center space-x-2">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={sliderValue}
          onChange={handleRangeChange}
          onWheel={handleWheel}
          disabled={disabled}
          className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-violet-500
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:hover:bg-violet-400
            [&::-moz-range-thumb]:w-4
            [&::-moz-range-thumb]:h-4
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:bg-violet-500
            [&::-moz-range-thumb]:cursor-pointer
            [&::-moz-range-thumb]:hover:bg-violet-400
            [&::-moz-range-thumb]:border-0"
        />
        <NumberInput
          label={videoFrameLabel(caps, arch)}
          value={value}
          onCommit={(v) => onChange(snap(v))}
          min={min}
          // The architecture's own cap, NOT the slider track's end: on an
          // uncapped architecture the box must not inherit a ceiling the UI
          // invented for drawing purposes.
          max={c.max_frames ?? undefined}
          step={step}
          parse="int"
          disabled={disabled}
          className="w-20"
        />
      </div>
      {!isOnGrid && (
        <p className="text-xs text-amber-400 mt-1">
          {value} is not a length this model generates; it is kept as set and
          will be snapped when the slider or number box is next used.
        </p>
      )}
    </div>
  );
}
