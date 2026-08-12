/**
 * Shared resolver for user-overridable slider/number-input UPPER BOUNDS.
 *
 * Single place where the three-way precedence for a bound is enforced -- no
 * panel computes this inline, so the precedence can never drift between call
 * sites. Backed by `backend/api/param_defaults.py`'s `PARAM_BOUNDS` registry
 * (served via `GET /schema/generation-defaults`'s `param_bounds` field) and
 * `UserSettings.slider_bounds` (`GET/POST /settings/generation`, threaded
 * through `StartupContext`'s `sliderBounds`/`setSliderBounds`).
 *
 * THE RULE this mechanism exists to serve (see PARAM_BOUNDS's own docstring
 * for the full statement): a bound is user-overridable IFF exceeding it can
 * only produce a worse or slower result, never a wrong or refused one. That
 * is a backend-registry decision, not a frontend one -- this module only
 * resolves a value against whatever the registry + user settings say; it
 * never decides eligibility itself.
 */

import type { ParamBoundsRegistry } from "./api";

export type SliderBoundsOverrides = Record<string, number>;

/**
 * Resolve the effective UPPER BOUND for one slider/number-input track.
 *
 * Precedence (highest wins first), enforced in exactly this order:
 *   1. `archLimit` -- a real architecture capability ceiling (e.g.
 *      `max_pixel_hw`), when the caller has one. This ALWAYS wins: it is a
 *      hard wall the backend will actually reject past, never a convenience,
 *      so no user override or built-in default may raise the control above
 *      it.
 *   2. The user's own override (`sliderBounds[boundName]`), when set.
 *   3. The registry's `builtin` value (today's literal), when the user has
 *      no override on file.
 *
 * Finally, `Math.max(resolved, currentValue)`: a value already loaded (from
 * a saved generation, a loop-generation step, or typed directly into the
 * paired number box) must never be stranded outside the track it is
 * rendered against -- the same property `VideoFrameCountSlider.tsx`'s own
 * `rawCeiling` expression already has (`Math.max(sliderMaxOverride ?? ...,
 * value)`), generalized here to every registered bound.
 *
 * `paramBounds`/`sliderBounds` are nullable so a caller can pass
 * `generationDefaults?.param_bounds` / `useStartup().sliderBounds` directly
 * without waiting on the startup fetch -- before it resolves, this falls
 * back to `currentValue` itself (i.e. the control's native/JSX literal is
 * unaffected until the registry arrives).
 */
export function resolveBound(
  boundName: string,
  paramBounds: ParamBoundsRegistry | null | undefined,
  sliderBounds: SliderBoundsOverrides | null | undefined,
  currentValue: number,
  archLimit?: number | null,
): number {
  const spec = paramBounds?.[boundName];
  const builtin = spec?.builtin ?? currentValue;
  const override = sliderBounds?.[boundName];
  let resolved = override ?? builtin;

  if (archLimit != null) {
    resolved = Math.min(resolved, archLimit);
  }

  return Math.max(resolved, currentValue);
}

/**
 * Whether `currentValue` sits above this bound's BUILTIN (not the resolved
 * value) -- i.e. only a user override (or a stranded loaded/typed value) put
 * it there. Mirrors `VideoFrameCountSlider`'s beyond-trained-range note:
 * informative, never blocking. Callers render this next to a control whose
 * `max` is `resolveBound(...)`, so a user understands WHY the track reaches
 * further than the shipped default without implying anything is wrong.
 */
export function isAboveBuiltin(
  boundName: string,
  paramBounds: ParamBoundsRegistry | null | undefined,
  currentValue: number,
): boolean {
  const builtin = paramBounds?.[boundName]?.builtin;
  return builtin != null && currentValue > builtin;
}
