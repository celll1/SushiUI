"use client";

import { useEffect, useRef, useState } from "react";
import { getActiveTraining, ActiveTrainingInfo } from "@/utils/api";

/** Polls /training/active periodically and returns the current state.
 *
 * Returns ``null`` when no LoRA / Full-FT training is running.  Polling
 * stops when the consumer unmounts; the next mount starts a fresh poll
 * loop.  Default cadence (10 s) matches the training monitor's status
 * polling so the backend isn't hammered by every generate panel mounting.
 */
export function useActiveTraining(intervalMs: number = 10_000): ActiveTrainingInfo | null {
  const [active, setActive] = useState<ActiveTrainingInfo | null>(null);
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;
    const refresh = async () => {
      try {
        const info = await getActiveTraining();
        if (!cancelled.current) setActive(info);
      } catch (e) {
        // getActiveTraining already swallows 404 / network errors; we
        // only land here on a programming bug.  Reset to null so the
        // toggle becomes inert rather than stuck on stale data.
        if (!cancelled.current) setActive(null);
      }
    };
    // Fire immediately on mount so the UI doesn't wait intervalMs to
    // populate the indicator.
    void refresh();
    const id = setInterval(refresh, intervalMs);
    return () => {
      cancelled.current = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return active;
}
