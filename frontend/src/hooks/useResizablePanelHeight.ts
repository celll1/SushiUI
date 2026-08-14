"use client";

import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

const MIN_HEIGHT = 112;
const DEFAULT_HEIGHT = 144;
const MAX_HEIGHT = 560;
// Tolerance for fit-to-content: absorbs a fractional content height that would
// otherwise round down into a clipped last row, and keeps a repeated fit from
// walking the panel upwards two pixels at a time.
const FIT_SLACK = 2;

function clamp(value: number): number {
  return Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, Math.round(value)));
}

export function useResizablePanelHeight(storageKey: string) {
  const [height, setHeight] = useState(DEFAULT_HEIGHT);
  // The height in force before a fit-to-content, kept so it can be restored.
  // null = nothing to go back to.
  const [userHeight, setUserHeight] = useState<number | null>(null);
  const [mounted, setMounted] = useState(false);
  const dragStart = useRef<{ y: number; height: number } | null>(null);
  // Written after commit, not during render: a render that is discarded must
  // not leave fitToContent reading a height that never reached the DOM.
  const heightRef = useRef(height);
  useEffect(() => {
    heightRef.current = height;
  }, [height]);
  const userHeightKey = `${storageKey}_user`;

  useEffect(() => {
    const saved = Number(localStorage.getItem(storageKey));
    if (Number.isFinite(saved) && saved > 0) setHeight(clamp(saved));
    const savedUser = Number(localStorage.getItem(userHeightKey));
    setUserHeight(Number.isFinite(savedUser) && savedUser > 0 ? clamp(savedUser) : null);
    setMounted(true);
  }, [storageKey, userHeightKey]);

  useEffect(() => {
    if (!mounted) return;
    localStorage.setItem(storageKey, String(clamp(height)));
  }, [height, mounted, storageKey]);

  // Persisted next to the height itself, so a fit survives a reload with its
  // restore target intact.
  useEffect(() => {
    if (!mounted) return;
    if (userHeight === null) localStorage.removeItem(userHeightKey);
    else localStorage.setItem(userHeightKey, String(userHeight));
  }, [userHeight, mounted, userHeightKey]);

  const reset = useCallback(() => {
    setUserHeight(null);
    setHeight(clamp(DEFAULT_HEIGHT));
  }, []);
  // A hand-set height is the new size to keep, so it replaces whatever a
  // restore would have gone back to.
  const setClampedHeight = useCallback((value: number) => {
    setUserHeight(null);
    setHeight(clamp(value));
  }, []);

  // Sizes the panel to the measured content, shrinking as well as growing, and
  // remembers the height it replaced. Clamped, so content taller than
  // MAX_HEIGHT still scrolls. The restore target is only taken when there is
  // none: a second fit must not overwrite the size the user set by hand.
  const fitToContent = useCallback((contentHeight: number) => {
    const target = clamp(contentHeight + FIT_SLACK);
    const previous = heightRef.current;
    if (Math.abs(target - previous) <= FIT_SLACK) return;
    setUserHeight((remembered) => (remembered === null ? previous : remembered));
    setHeight(target);
  }, []);
  const restoreUserHeight = useCallback(() => {
    if (userHeight === null) return;
    setHeight(clamp(userHeight));
    setUserHeight(null);
  }, [userHeight]);

  const onPointerDown = useCallback((event: PointerEvent<HTMLDivElement>) => {
    dragStart.current = { y: event.clientY, height };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }, [height]);
  const onPointerMove = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (!dragStart.current) return;
    setClampedHeight(dragStart.current.height + event.clientY - dragStart.current.y);
  }, [setClampedHeight]);
  const onPointerUp = useCallback((event: PointerEvent<HTMLDivElement>) => {
    dragStart.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);
  const onKeyDown = useCallback((event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 32 : 8;
    if (event.key === "ArrowUp") setClampedHeight(height - step);
    else if (event.key === "ArrowDown") setClampedHeight(height + step);
    else if (event.key === "Home") setClampedHeight(MIN_HEIGHT);
    else if (event.key === "End") setClampedHeight(MAX_HEIGHT);
    else if (event.key === "Enter" || event.key === " ") reset();
    else return;
    event.preventDefault();
  }, [height, reset, setClampedHeight]);

  return {
    height,
    minHeight: MIN_HEIGHT,
    maxHeight: MAX_HEIGHT,
    userHeight,
    reset,
    fitToContent,
    restoreUserHeight,
    separatorProps: {
      onPointerDown,
      onPointerMove,
      onPointerUp,
      onPointerCancel: onPointerUp,
      onDoubleClick: reset,
      onKeyDown,
    },
  };
}
