"use client";

import {
  Children,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

interface ResizableColumnsProps {
  children: ReactNode;
  storageKey: string;
  label: string;
  defaultPrimaryPercent?: number;
  minPrimaryPercent?: number;
  maxPrimaryPercent?: number;
  minPrimaryPx?: number;
  minSecondaryPx?: number;
  className?: string;
}

interface DragState {
  pointerId: number;
  startX: number;
  startPercent: number;
  containerWidth: number;
  minPercent: number;
  maxPercent: number;
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

export const GENERATION_WORKSPACE_SPLIT_KEY = "generation_workspace_split";
export const GENERATION_PREVIEW_QUEUE_SPLIT_KEY = "generation_preview_queue_split";

export default function ResizableColumns({
  children,
  storageKey,
  label,
  defaultPrimaryPercent = 50,
  minPrimaryPercent = 30,
  maxPrimaryPercent = 70,
  minPrimaryPx = 0,
  minSecondaryPx = 0,
  className = "",
}: ResizableColumnsProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const [preferredPercent, setPreferredPercent] = useState(() =>
    clamp(defaultPrimaryPercent, minPrimaryPercent, maxPrimaryPercent),
  );
  const [isDragging, setIsDragging] = useState(false);
  const [preferenceLoaded, setPreferenceLoaded] = useState(false);
  const [containerWidth, setContainerWidth] = useState(0);
  const [isHorizontal, setIsHorizontal] = useState(false);
  // Overlay siblings may follow the two columns; keep them outside the flex row.
  const [primary, secondary, ...afterColumns] = Children.toArray(children);
  const effectiveBounds = useCallback((width: number) => {
    let min = minPrimaryPercent;
    let max = maxPrimaryPercent;
    if (isHorizontal && width > 0) {
      min = Math.max(min, (minPrimaryPx / width) * 100);
      max = Math.min(max, ((width - 8 - minSecondaryPx) / width) * 100);
    }
    if (min <= max) return { min, max };

    const totalMinimum = minPrimaryPx + minSecondaryPx;
    const compromise = totalMinimum > 0
      ? (minPrimaryPx / totalMinimum) * (Math.max(1, width - 8) / width) * 100
      : defaultPrimaryPercent;
    const fixed = clamp(compromise, minPrimaryPercent, maxPrimaryPercent);
    return { min: fixed, max: fixed };
  }, [
    defaultPrimaryPercent,
    isHorizontal,
    maxPrimaryPercent,
    minPrimaryPercent,
    minPrimaryPx,
    minSecondaryPx,
  ]);
  const bounds = effectiveBounds(containerWidth);
  const primaryPercent = clamp(preferredPercent, bounds.min, bounds.max);

  useEffect(() => {
    const stored = Number(localStorage.getItem(storageKey));
    if (Number.isFinite(stored) && stored > 0) {
      setPreferredPercent(clamp(stored, minPrimaryPercent, maxPrimaryPercent));
    }
    setPreferenceLoaded(true);
  }, [maxPrimaryPercent, minPrimaryPercent, storageKey]);

  useEffect(() => {
    const media = window.matchMedia("(min-width: 1024px)");
    const update = () => setIsHorizontal(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(([entry]) => {
      setContainerWidth(entry.contentRect.width);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!preferenceLoaded) return;
    localStorage.setItem(storageKey, preferredPercent.toFixed(2));
  }, [preferenceLoaded, preferredPercent, storageKey]);

  const reset = () => {
    setPreferredPercent(clamp(defaultPrimaryPercent, minPrimaryPercent, maxPrimaryPercent));
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const dragBounds = effectiveBounds(rect.width);

    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startPercent: primaryPercent,
      containerWidth: rect.width,
      minPercent: dragBounds.min,
      maxPercent: dragBounds.max,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsDragging(true);
    event.preventDefault();
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const deltaPercent = ((event.clientX - drag.startX) / drag.containerWidth) * 100;
    setPreferredPercent(clamp(
      drag.startPercent + deltaPercent,
      drag.minPercent,
      drag.maxPercent,
    ));
  };

  const stopDragging = (event: PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    setIsDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 5 : 2;
    let next: number | null = null;
    if (event.key === "ArrowLeft") next = primaryPercent - step;
    if (event.key === "ArrowRight") next = primaryPercent + step;
    if (event.key === "Home") next = bounds.min;
    if (event.key === "End") next = bounds.max;
    if (event.key === "Enter") next = defaultPrimaryPercent;
    if (next === null) return;
    event.preventDefault();
    setPreferredPercent(clamp(next, bounds.min, bounds.max));
  };

  if (primary === undefined || secondary === undefined) {
    return <div className={className}>{children}</div>;
  }

  return (
    <>
      <div
        ref={containerRef}
        className={`flex flex-col gap-2.5 lg:flex-row lg:gap-0 ${className}`}
        style={{ "--split-primary": `${primaryPercent}%` } as CSSProperties}
      >
        <div className="flex min-w-0 flex-col lg:flex-none lg:[flex-basis:var(--split-primary)]">
          {primary}
        </div>
        <div
          role="separator"
          aria-label={label}
          aria-orientation="vertical"
          aria-valuemin={Math.round(bounds.min)}
          aria-valuemax={Math.round(bounds.max)}
          aria-valuenow={Math.round(primaryPercent)}
          aria-valuetext={`${Math.round(primaryPercent)}% / ${Math.round(100 - primaryPercent)}%`}
          tabIndex={0}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={stopDragging}
          onPointerCancel={stopDragging}
          onLostPointerCapture={() => {
            dragRef.current = null;
            setIsDragging(false);
          }}
          onKeyDown={handleKeyDown}
          onDoubleClick={reset}
          className={`group hidden w-2 shrink-0 cursor-col-resize touch-none select-none items-center justify-center rounded-sm transition-colors lg:flex focus:outline-none focus:ring-2 focus:ring-violet-500/70 ${
            isDragging ? "bg-violet-500/20" : "hover:bg-violet-500/10"
          }`}
          title="Drag to resize. Double-click or press Enter to reset."
        >
          <span className={`h-10 w-0.5 rounded-full transition-colors ${
            isDragging ? "bg-violet-400" : "bg-gray-700 group-hover:bg-violet-400"
          }`} />
        </div>
        <div className="flex min-w-0 flex-col lg:flex-1">
          {secondary}
        </div>
      </div>
      {afterColumns}
    </>
  );
}
