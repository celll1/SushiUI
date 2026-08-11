/**
 * Pure value<->pixel conversion for the project's horizontal timeline tracks
 * (VideoInpaintTimeline -- via components/timeline/Timeline.tsx --,
 * OutpaintTimeline, MiniMaxH3KeyframeTimeline). Extracted so the same
 * clientX->value and value->CSS-left arithmetic isn't re-derived (and
 * re-drifted) independently in each track's own component file.
 *
 * Deliberately NOT a generic "track" abstraction: each track's VALUE DOMAIN
 * differs (pixel frames, latent groups, output-timeline units, seconds) and
 * so does its interaction model (continuous drag vs. discrete-frame
 * snapping vs. latent-group snapping) -- unifying those belongs to each
 * component, not here. This module only owns the arithmetic of "given a
 * clientX (or a value) and a [min, max] domain mapped onto a track's own
 * box (optionally inset by a fixed pixel margin at both edges), what is the
 * corresponding value (or CSS position)".
 *
 * The domain is currently always the FULL visible track (`[0, max]`) in
 * every caller -- there is no zoom/scroll yet. `TimelineValueScale.min`/
 * `.max` are still exposed as the domain's own edges (not hardcoded to 0) so
 * a future zoomed/scrolled view can pass a narrower `[min, max]` window
 * without this module changing.
 */

export interface TimelineValueScale {
  /** The value at the LEFT edge of the track's usable (post-inset) width. */
  min: number;
  /** The value at the RIGHT edge of the track's usable (post-inset) width. */
  max: number;
  /**
   * Pixels of margin reserved at BOTH edges of the track's own box before the
   * domain starts mapping to pixels -- e.g. half a marker's width, so an
   * anchor at `min` or `max` is fully visible instead of hanging outside the
   * track (MiniMaxH3KeyframeTimeline's `MARKER_HALF_PX`). 0 (default) = the
   * domain maps to the track's full width, which is every OTHER existing
   * track.
   */
  insetPx?: number;
}

/**
 * Minimal shape this module needs from a `DOMRect`, so it never calls
 * `getBoundingClientRect()` itself -- callers own that (it is the only
 * DOM-touching part of the calculation, and it is only safe to call while a
 * ref is mounted, which differs per caller).
 */
export interface TimelineRect {
  left: number;
  width: number;
}

const clamp01 = (v: number): number => Math.min(1, Math.max(0, v));

/**
 * clientX (a pointer event's own coordinate) -> the CONTINUOUS value under
 * it, per `scale`. Clamped to `[scale.min, scale.max]`; NOT rounded (see
 * `roundedValueAtClientX` for that) -- OutpaintTimeline intentionally keeps
 * the fractional value mid-drag (delta math, snap only on release).
 *
 * Returns `scale.min` if the track's usable width (`rect.width - 2 *
 * insetPx`) is not positive, matching every existing implementation's own
 * degenerate-width guard.
 */
export function valueAtClientX(clientX: number, rect: TimelineRect, scale: TimelineValueScale): number {
  const inset = scale.insetPx ?? 0;
  const usable = rect.width - inset * 2;
  if (usable <= 0) return scale.min;
  const fraction = clamp01((clientX - rect.left - inset) / usable);
  return scale.min + fraction * (scale.max - scale.min);
}

/**
 * Same as `valueAtClientX`, rounded to the nearest integer -- for callers
 * (components/timeline/Timeline.tsx, MiniMaxH3KeyframeTimeline) whose value
 * domain is a whole frame/group index. Re-clamped to `[min, max]` after
 * rounding as a defensive measure (rounding a value already inside
 * `[min, max]` cannot actually leave that range when `min`/`max` are
 * themselves integers, which holds for both current callers).
 */
export function roundedValueAtClientX(clientX: number, rect: TimelineRect, scale: TimelineValueScale): number {
  const value = Math.round(valueAtClientX(clientX, rect, scale));
  return Math.min(scale.max, Math.max(scale.min, value));
}

/**
 * value -> percent (0-100) of the DOMAIN alone (`[scale.min, scale.max]`),
 * ignoring `insetPx` -- what every existing track without an inset uses
 * directly for a plain `left: X%`/`width: X%` style. Returns 0 if the
 * domain is degenerate (`max === min`), matching each existing call site's
 * own guard against a divide-by-zero.
 */
export function percentForValue(value: number, scale: TimelineValueScale): number {
  const span = scale.max - scale.min;
  if (span === 0) return 0;
  return ((value - scale.min) / span) * 100;
}

/**
 * value -> a CSS `left` expression that also accounts for `insetPx`: a plain
 * `"X%"` when insetPx is 0 (every track but one), else the same
 * `calc(insetPx + (100% - 2*insetPx) * X%)` MiniMaxH3KeyframeTimeline already
 * used to keep an anchor at the domain's own edge fully visible inside the
 * track's box.
 */
export function cssLeftForValue(value: number, scale: TimelineValueScale): string {
  const pct = percentForValue(value, scale);
  const inset = scale.insetPx ?? 0;
  if (inset === 0) return `${pct}%`;
  return `calc(${inset}px + (100% - ${inset * 2}px) * ${pct / 100})`;
}
