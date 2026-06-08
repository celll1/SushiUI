"use client";

import { useCallback, useRef } from "react";

/**
 * Returns a mousedown handler for a drag divider between two panels.
 * containerRef: the flex container holding both panels.
 * storageKey: localStorage key to persist the split (optional).
 * direction: "horizontal" (left/right) | "vertical" (top/bottom).
 * minPx: minimum size in px for the primary panel.
 * maxRatio: maximum ratio (0–1) for the primary panel.
 * onResize(primaryPx): called each frame with the new primary panel size in px.
 */
export function usePanelResize({
  containerRef,
  direction = "horizontal",
  minPx = 120,
  maxRatio = 0.85,
  onResize,
}: {
  containerRef: React.RefObject<HTMLElement | null>;
  direction?: "horizontal" | "vertical";
  minPx?: number;
  maxRatio?: number;
  onResize: (primaryPx: number) => void;
}) {
  const dragging = useRef(false);
  const startPos = useRef(0);
  const startSize = useRef(0);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragging.current = true;
      startPos.current = direction === "horizontal" ? e.clientX : e.clientY;

      const el = containerRef.current;
      if (el) {
        const rect = el.getBoundingClientRect();
        const totalSize =
          direction === "horizontal" ? rect.width : rect.height;
        // Measure the first child's current size
        const firstChild = el.firstElementChild as HTMLElement | null;
        if (firstChild) {
          const childRect = firstChild.getBoundingClientRect();
          startSize.current =
            direction === "horizontal" ? childRect.width : childRect.height;
        } else {
          startSize.current = totalSize * 0.33;
        }
      }

      const onMove = (me: MouseEvent) => {
        if (!dragging.current) return;
        const el = containerRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const totalSize =
          direction === "horizontal" ? rect.width : rect.height;
        const delta =
          (direction === "horizontal" ? me.clientX : me.clientY) -
          startPos.current;
        const newSize = Math.min(
          Math.max(minPx, startSize.current + delta),
          totalSize * maxRatio
        );
        onResize(newSize);
      };

      const onUp = () => {
        dragging.current = false;
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };

      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [containerRef, direction, minPx, maxRatio, onResize]
  );

  return { onMouseDown };
}
