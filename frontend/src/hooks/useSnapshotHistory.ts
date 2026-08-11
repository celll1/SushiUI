"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Generic undo/redo for a CONTROLLED value whose actual state lives in a
// parent component (e.g. InpaintPanel's `videoMaskManifest`), reached only
// through a `commit` callback -- this hook holds no copy of "the current
// value" itself, only the past/future STACKS of prior snapshots. Every
// mutating action calls `push(currentValueFromProps, nextValue)`; `undo`/
// `redo` are handed the current value from props at call time (not cached),
// so there is no chance of the hook's own idea of "current" drifting from
// what the parent actually holds.
//
// Scope: this is the mechanism for VideoInpaintTimeline's mask-keyframe
// editing (add/delete/duplicate/transform/frame-move/interpolation/feather)
// ONLY. Drawing inside the mask editor itself has its own undo, scoped to
// ImageEditor's canvas -- that is a different history stack for a different
// kind of edit and is not touched by or related to this hook.
// ---------------------------------------------------------------------------

export interface SnapshotHistory<T> {
  /** Record `current` (the value BEFORE the mutation) onto the undo stack, then commit `next`. Clears the redo stack (a fresh edit invalidates any redo path). */
  push: (current: T, next: T) => void;
  /** Commit the most recent entry on the undo stack, given the CURRENT value (from props) to push onto the redo stack. No-op if the undo stack is empty. */
  undo: (current: T) => void;
  /** Commit the most recent entry on the redo stack, given the CURRENT value (from props) to push back onto the undo stack. No-op if the redo stack is empty. */
  redo: (current: T) => void;
  canUndo: boolean;
  canRedo: boolean;
}

export interface SnapshotHistoryOptions {
  /** Max entries kept per stack. Default 100 -- generous relative to the 128-keyframe manifest cap this was built for, since each entry is just numbers and id strings. */
  limit?: number;
  /**
   * Changing this value clears both stacks without touching the live value
   * (e.g. the identity of the clip/session this history belongs to -- a new
   * clip's keyframes are not undo-continuous with the previous one's).
   */
  resetKey?: unknown;
}

export function useSnapshotHistory<T>(
  commit: (value: T) => void,
  options?: SnapshotHistoryOptions,
): SnapshotHistory<T> {
  const limit = options?.limit ?? 100;
  const pastRef = useRef<T[]>([]);
  const futureRef = useRef<T[]>([]);
  // Only used to force a re-render when the stacks change shape (so
  // `canUndo`/`canRedo` reflect the latest push/undo/redo); the stacks
  // themselves live in refs, not state, since their CONTENTS never drive
  // rendering directly.
  const [, bumpRenderTick] = useState(0);

  useEffect(() => {
    pastRef.current = [];
    futureRef.current = [];
    bumpRenderTick((n) => n + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options?.resetKey]);

  const push = useCallback(
    (current: T, next: T) => {
      pastRef.current = [...pastRef.current, current].slice(-limit);
      futureRef.current = [];
      bumpRenderTick((n) => n + 1);
      commit(next);
    },
    [commit, limit],
  );

  const undo = useCallback(
    (current: T) => {
      const past = pastRef.current;
      if (past.length === 0) return;
      const previous = past[past.length - 1];
      pastRef.current = past.slice(0, -1);
      futureRef.current = [current, ...futureRef.current].slice(0, limit);
      bumpRenderTick((n) => n + 1);
      commit(previous);
    },
    [commit, limit],
  );

  const redo = useCallback(
    (current: T) => {
      const future = futureRef.current;
      if (future.length === 0) return;
      const next = future[0];
      futureRef.current = future.slice(1);
      pastRef.current = [...pastRef.current, current].slice(-limit);
      bumpRenderTick((n) => n + 1);
      commit(next);
    },
    [commit, limit],
  );

  return {
    push,
    undo,
    redo,
    canUndo: pastRef.current.length > 0,
    canRedo: futureRef.current.length > 0,
  };
}
