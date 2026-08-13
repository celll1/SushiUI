"use client";

import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

const MIN_HEIGHT = 112;
const DEFAULT_HEIGHT = 144;
const MAX_HEIGHT = 560;

function clamp(value: number): number {
  return Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, Math.round(value)));
}

export function useResizablePanelHeight(storageKey: string) {
  const [height, setHeight] = useState(DEFAULT_HEIGHT);
  const [mounted, setMounted] = useState(false);
  const dragStart = useRef<{ y: number; height: number } | null>(null);

  useEffect(() => {
    const saved = Number(localStorage.getItem(storageKey));
    if (Number.isFinite(saved) && saved > 0) setHeight(clamp(saved));
    setMounted(true);
  }, [storageKey]);

  useEffect(() => {
    if (!mounted) return;
    localStorage.setItem(storageKey, String(clamp(height)));
  }, [height, mounted, storageKey]);

  const reset = useCallback(() => setHeight(clamp(DEFAULT_HEIGHT)), []);
  const setClampedHeight = useCallback((value: number) => setHeight(clamp(value)), []);

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
    reset,
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
