import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/**
 * Smooths the generation progress bar *within* a single step.
 *
 * A step of a video model can take minutes, during which the backend's integer
 * `step` counter does not move and the bar looks frozen. Two sub-step sources
 * feed this hook, in priority order:
 *
 *   1. `sub_progress` from the backend `progress` message (currently only
 *      minimax_h3 emits it). Authoritative — used verbatim when present.
 *   2. Otherwise, a time-based estimate: elapsed / EMA(step duration), capped
 *      below 1 so the bar never claims a step finished that has not.
 *
 * The caller keeps owning `progress` / `totalSteps` (the textual "3/7 steps"
 * stays integer); this hook only produces the bar's width.
 */

const TICK_MS = 200;
/** Weight of the newest step duration in the EMA. Runs are short (often <30
 *  steps), so the estimate has to adapt fast. */
const EMA_ALPHA = 0.4;
/** Interpolation ceiling: an overrunning step must not fill the segment. */
const MAX_INTERPOLATED = 0.95;

interface ServerSample {
  /** Completed-step count the sample belongs to; guards against stale values. */
  step: number;
  value: number;
}

interface TimeSample {
  step: number;
  at: number;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export interface SmoothProgress {
  /** Bar width in percent, 0..100, including the sub-step fraction. */
  percent: number;
  /** Sub-step fraction currently in use, 0..1 (exposed for debugging/tests). */
  subProgress: number;
  /**
   * Feed the backend's `sub_progress` field. Call from the panel's WS progress
   * handler with the same `step` the message carried; pass `undefined` when the
   * message has no `sub_progress` (architectures other than minimax_h3).
   */
  reportSubProgress: (step: number, subProgress?: number) => void;
}

export function useSmoothProgress(
  progress: number,
  totalSteps: number,
  isActive: boolean
): SmoothProgress {
  const [serverSub, setServerSub] = useState<ServerSample | null>(null);
  const [interpolated, setInterpolated] = useState(0);

  /** Seconds per step, EMA over the current run. null until measurable. */
  const emaRef = useRef<number | null>(null);
  const lastSampleRef = useRef<TimeSample | null>(null);

  const reportSubProgress = useCallback((step: number, subProgress?: number) => {
    setServerSub(
      typeof subProgress === "number" && Number.isFinite(subProgress)
        ? { step, value: clamp(subProgress, 0, 1) }
        : null
    );
  }, []);

  // Track step arrival times and maintain the duration EMA. Re-runs on every
  // progress change, so every panel-side setProgress(0) also resets the
  // sub-step state without the call sites needing to know about it.
  useEffect(() => {
    if (!isActive) {
      emaRef.current = null;
      lastSampleRef.current = null;
      setServerSub(null);
      setInterpolated(0);
      return;
    }

    const now = Date.now();
    const prev = lastSampleRef.current;

    if (progress <= 0) {
      emaRef.current = null;
      lastSampleRef.current = null;
      setInterpolated(0);
      return;
    }

    if (!prev || progress < prev.step) {
      // Run start, or the counter restarted (batch / loop generation).
      emaRef.current = null;
    } else if (progress > prev.step && prev.step >= 1) {
      // Step 1's window also contains model load and text encoding, so it is
      // never used as a duration sample.
      const perStep = (now - prev.at) / 1000 / (progress - prev.step);
      if (perStep > 0) {
        emaRef.current =
          emaRef.current === null
            ? perStep
            : EMA_ALPHA * perStep + (1 - EMA_ALPHA) * emaRef.current;
      }
    }

    lastSampleRef.current = { step: progress, at: now };
    setInterpolated(0);
  }, [progress, isActive]);

  const hasServerSub = serverSub !== null && serverSub.step === progress;

  useEffect(() => {
    if (!isActive || hasServerSub) return;
    if (totalSteps <= 0 || progress <= 0 || progress >= totalSteps) return;
    if (emaRef.current === null) return; // no duration estimate yet

    const id = setInterval(() => {
      const ema = emaRef.current;
      const last = lastSampleRef.current;
      if (ema === null || !last) return;
      const elapsed = (Date.now() - last.at) / 1000;
      setInterpolated(Math.min(MAX_INTERPOLATED, elapsed / ema));
    }, TICK_MS);

    return () => clearInterval(id);
  }, [isActive, hasServerSub, totalSteps, progress]);

  return useMemo(() => {
    const sub = !isActive || progress >= totalSteps
      ? 0
      : hasServerSub
        ? serverSub!.value
        : interpolated;
    const percent =
      totalSteps > 0 ? clamp(((progress + sub) / totalSteps) * 100, 0, 100) : 0;
    return { percent, subProgress: sub, reportSubProgress };
  }, [progress, totalSteps, isActive, hasServerSub, serverSub, interpolated, reportSubProgress]);
}
