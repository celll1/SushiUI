"use client";

/**
 * Two chart panes with a draggable split and a shared draggable height.
 *
 * Per-pane sizing was considered and rejected: two panes that can differ in
 * height stop being comparable at a glance, which is the whole reason there are
 * two of them. So the vertical handle moves the boundary BETWEEN them and the
 * horizontal handle at the bottom sizes BOTH.
 *
 * The split only exists at the two-column breakpoint. Below it the panes stack,
 * the vertical handle is not rendered, and the width is whatever the column is.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const KEY = "sushi.trainingChart.v1.layout";
const TWO_COL = "(min-width: 1800px)";

const MIN_PCT = 25;
const MAX_PCT = 75;
const MIN_H = 120;
const MAX_H = 640;
const DEFAULT_PCT = 50;
const DEFAULT_H = 160;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export default function ResizableChartPair(
  { left, right }: { left: (height: number) => React.ReactNode; right: (height: number) => React.ReactNode },
) {
  const [pct, setPct] = useState(DEFAULT_PCT);
  const [height, setHeight] = useState(DEFAULT_H);
  const [twoCol, setTwoCol] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const rowRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ kind: "x" | "y"; startY: number; startH: number } | null>(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) {
        const v = JSON.parse(raw) as { pct?: number; height?: number };
        if (typeof v.pct === "number") setPct(clamp(v.pct, MIN_PCT, MAX_PCT));
        if (typeof v.height === "number") setHeight(clamp(v.height, MIN_H, MAX_H));
      }
    } catch { /* a malformed or older shape falls back to the defaults */ }
    setLoaded(true);
  }, []);

  useEffect(() => {
    if (!loaded) return;
    const t = setTimeout(() => {
      try { localStorage.setItem(KEY, JSON.stringify({ pct, height })); } catch { /* quota / private mode */ }
    }, 400);
    return () => clearTimeout(t);
  }, [loaded, pct, height]);

  useEffect(() => {
    const mq = window.matchMedia(TWO_COL);
    const apply = () => setTwoCol(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  // Pointer capture on the handle, so a fast drag that leaves the element still
  // tracks -- the same reason SharedMetricChart's brush captures.
  const onMove = useCallback((e: PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    if (d.kind === "x") {
      const row = rowRef.current;
      if (!row) return;
      const r = row.getBoundingClientRect();
      if (r.width > 0) setPct(clamp(((e.clientX - r.left) / r.width) * 100, MIN_PCT, MAX_PCT));
    } else {
      setHeight(clamp(d.startH + (e.clientY - d.startY), MIN_H, MAX_H));
    }
  }, []);

  useEffect(() => {
    const up = () => { drag.current = null; document.body.style.userSelect = ""; };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", up);
    };
  }, [onMove]);

  const start = (kind: "x" | "y") => (e: React.PointerEvent) => {
    e.preventDefault();
    drag.current = { kind, startY: e.clientY, startH: height };
    document.body.style.userSelect = "none";
  };

  const reset = () => { setPct(DEFAULT_PCT); setHeight(DEFAULT_H); };

  return (
    <div className="mb-3">
      <div
        ref={rowRef}
        className={twoCol ? "flex items-stretch" : "flex flex-col gap-3"}
      >
        <div className="min-w-0" style={twoCol ? { width: `${pct}%` } : undefined}>
          {left(height)}
        </div>
        {twoCol && (
          <div
            onPointerDown={start("x")}
            onDoubleClick={reset}
            title="Drag to resize · double-click to reset"
            className="w-3 shrink-0 cursor-col-resize flex items-center justify-center group"
          >
            <div className="w-px h-full bg-gray-700 group-hover:bg-blue-500 transition-colors" />
          </div>
        )}
        <div className="min-w-0 flex-1">{right(height)}</div>
      </div>
      <div
        onPointerDown={start("y")}
        onDoubleClick={reset}
        title="Drag to resize both panes · double-click to reset"
        className="h-3 cursor-row-resize flex items-center justify-center group"
      >
        <div className="h-px w-16 bg-gray-700 group-hover:bg-blue-500 transition-colors" />
      </div>
    </div>
  );
}
