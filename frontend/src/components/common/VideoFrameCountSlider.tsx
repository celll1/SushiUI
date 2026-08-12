"use client";

import { ChangeEvent, WheelEvent } from "react";
import Select from "./Select";
import NumberInput from "./NumberInput";
import {
  ArchCapabilities,
  isValidVideoFrameCount,
  largestValidVideoFrameCount,
  nearestValidVideoFrameCount,
  planVideoChain,
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
  /**
   * Whether the number box may accept a value ABOVE the single-inference
   * cap (`c.max_frames`), the opt-in entry point for the chain feature on
   * the main `num_frames` control. Callers using this component for a
   * value that is itself a single-inference length -- e.g. a chain's own
   * segment length -- pass `false` so the field stays within the cap and
   * the "exceeds the single-inference limit" / chain-plan messaging below
   * (which talks about splitting the CURRENT field, not the one this
   * control represents) does not render. Defaults to `true`, preserving
   * the original `num_frames` behaviour.
   */
  allowOverCap?: boolean;
  /**
   * User-configured upper bound for the slider TRACK (Settings ->
   * `UserSettings.video_frame_slider_max`, threaded down from
   * `useStartup().videoFrameSliderMax`). Bounds the track only -- see
   * `rawCeiling` below, and the number box's own comment, for why it is
   * never applied to the number box. `null`/`undefined` (unset) keeps this
   * component's own built-in track reach (`UNCAPPED_FRAME_SLIDER_CEILING` /
   * `TRAINED_RANGE_SLIDER_HEADROOM`), so a user who never opens Settings
   * sees no change. Applies identically to both call sites of this
   * component (`num_frames` and a chain segment length): both compute the
   * track ceiling through the same `rawCeiling` expression below, and this
   * setting is about how far a slider TRACK reaches, a property of the
   * track, not of which quantity it happens to be editing.
   */
  sliderMaxOverride?: number | null;
}

// How far the SLIDER TRACK reaches on an architecture that declares no
// `max_frames` AND no `trained_max_frames` either. A range input has to end
// somewhere; that is a drawing constraint, not an architecture fact -- the
// backend imposes no upper bound there beyond the frame grid. So this bounds
// the track only: the number box stays unbounded on such an architecture,
// and typing past the track extends it (see `rawCeiling`). Same convention
// as api.ts's `UNCAPPED_VIDEO_EDGE` for the canvas sliders.
//
// When `trained_max_frames` IS known (MiniMax-H3: 362), the track derives
// its reach from that instead -- 241 is below MiniMax-H3's own 124-frame
// FLOOR, which would render a slider whose entire track sits in territory
// the architecture cannot even produce.
const UNCAPPED_FRAME_SLIDER_CEILING = 241;
// Headroom multiplier applied above `trained_max_frames` so the track still
// reaches usefully past the documented range (the amber note below is what
// tells the user that territory is untested, not the track's own bound).
const TRAINED_RANGE_SLIDER_HEADROOM = 1.5;

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
  allowOverCap = true,
  sliderMaxOverride = null,
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
  const rawCeiling = c.max_frames ?? (
    c.trained_max_frames != null
      ? Math.max(sliderMaxOverride ?? Math.round(c.trained_max_frames * TRAINED_RANGE_SLIDER_HEADROOM), value)
      : Math.max(sliderMaxOverride ?? UNCAPPED_FRAME_SLIDER_CEILING, value)
  );
  const max = largestValidVideoFrameCount(caps, arch, rawCeiling) ?? rawCeiling;
  const step = c.frame_multiple;
  // Clamp only what the native <input type="range"> needs to stay in
  // [min, max] to render without a browser warning; the readout and the
  // number box always show the true `value`, off-grid or not.
  const sliderValue = Math.min(Math.max(value, min), max);

  const snap = (frames: number): number => nearestValidVideoFrameCount(caps, arch, frames) ?? frames;

  // The number box only (never the slider, whose native `max` already stops
  // it at the cap): a value ABOVE the single-inference cap is an opt-in
  // request for a length only the chain feature can reach, so it is rounded
  // onto the frame grid but NOT clamped back down to `max_frames` the way
  // `snap` clamps every other value. See `planVideoChain` (api.ts).
  const snapAllowingOverCap = (frames: number): number => {
    if (c.max_frames != null && frames > c.max_frames) {
      const k = Math.max(0, Math.round((frames - c.frame_offset) / c.frame_multiple));
      return k * c.frame_multiple + c.frame_offset;
    }
    return snap(frames);
  };

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
  // The threshold the "over cap" warning keys off: the architecture's real
  // single-inference wall (`max_frames`) when it still has one (LTX-2.3), or
  // its documented-trained advisory ceiling (`trained_max_frames`) when it
  // does not (MiniMax-H3) -- the latter is no longer enforced by the backend,
  // so a value past it is untested rather than rejected/auto-chained.
  const overCapThreshold = c.max_frames ?? c.trained_max_frames;
  // isValidVideoFrameCount is also false for any value past max_frames, so an
  // over-cap value falls into the same "off-grid" bucket by that check alone
  // — this splits it out so the two get separate, non-contradictory messages
  // (the off-grid one says the value "will be snapped", which is not true of
  // an over-cap value: it is deliberately left alone for the chain feature).
  const overCap = allowOverCap && overCapThreshold != null && value > overCapThreshold;
  const chainPlan = overCap ? planVideoChain(caps, arch, value) : null;
  const thresholdSeconds = overCapThreshold != null && fps > 0 ? (overCapThreshold / fps).toFixed(2) : null;

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
          onCommit={(v) => onChange(allowOverCap ? snapAllowingOverCap(v) : snap(v))}
          min={min}
          // No `max`: the number box (unlike the slider track, whose native
          // `max` stays at the cap below) must accept a value ABOVE the
          // single-inference cap -- that is the opt-in entry point for the
          // chain feature. `snapAllowingOverCap` is what still keeps a
          // within-cap value on the frame grid. Callers with `allowOverCap`
          // false (a length that is itself a single-inference cap, e.g. a
          // chain segment length) use the plain, clamping `snap` instead.
          step={step}
          parse="int"
          disabled={disabled}
          className="w-20"
        />
      </div>
      {overCap ? (
        <p className="text-xs text-amber-400 mt-1">
          {c.max_frames != null ? (
            <>
              {value} frames exceeds the single-inference limit of {c.max_frames} frames
              {thresholdSeconds != null ? ` (${thresholdSeconds}s at ${fps} fps)` : ""}.
              {chainPlan != null
                ? ` Reaching it takes ${chainPlan.segments} generation requests (actually reaches ${chainPlan.finalFrames} frames` +
                  `${chainPlan.finalFrames !== value ? `, ${chainPlan.finalFrames - value} more than this` : ""}); segments after` +
                  ` the first are conditioned on the boundary frame of the previous segment, not the rest of its content, while the` +
                  ` same full-length prompt is resent unchanged on every segment. Generate will ask you to choose between a single` +
                  ` inference at the cap and the chain.`
                : ""}
            </>
          ) : (
            <>
              {value} frames is beyond the model's documented trained range ({c.trained_max_frames} frames
              {thresholdSeconds != null ? ` / ${thresholdSeconds}s` : ""}); longer is untested.
              {chainPlan != null
                ? ` A chain segment length is set, so reaching it takes ${chainPlan.segments} generation requests (actually reaches` +
                  ` ${chainPlan.finalFrames} frames${chainPlan.finalFrames !== value ? `, ${chainPlan.finalFrames - value} more than this` : ""});` +
                  ` segments after the first are conditioned on the boundary frame of the previous segment, not the rest of its` +
                  ` content, while the same full-length prompt is resent unchanged on every segment.`
                : ""}
            </>
          )}
        </p>
      ) : !isOnGrid && (
        <p className="text-xs text-amber-400 mt-1">
          {value} is not a length this model generates; it is kept as set and
          will be snapped when the slider or number box is next used.
        </p>
      )}
    </div>
  );
}
