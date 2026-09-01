"use client";

/**
 * N chart panes on one row, with draggable boundaries and one shared height.
 *
 * Per-pane heights were considered and rejected: panes that can differ in height
 * stop being comparable at a glance, which is the reason there is more than one.
 * So the vertical handles move the boundaries BETWEEN panes and the horizontal
 * handle at the bottom sizes ALL of them.
 *
 * The row only splits at the two-column breakpoint. Below it the panes stack,
 * the vertical handles are not rendered, and each pane is the column's width.
 * At one pane there is nothing to split and no handle is drawn either.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const KEY = "sushi.trainingChart.v1.layout";
const WIDE = "(min-width: 1800px)";

export const MIN_PANES = 1;
export const MAX_PANES = 3;
const MIN_H = 120;
const MAX_H = 640;
const DEFAULT_H = 160;
/** Smallest share of the row a pane may be squeezed to. */
const MIN_SHARE = 0.15;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const evenShares = (n: number) => Array.from({ length: n }, () => 1 / n);

export interface ChartLayout {
  panes: number;
  height: number;
  /** One share per pane, summing to 1. Only meaningful at the wide breakpoint. */
  shares: number[];
}

export const DEFAULT_LAYOUT: ChartLayout = { panes: 2, height: DEFAULT_H, shares: evenShares(2) };

/** Layout state + persistence, lifted so the toolbar can live outside the row. */
export function useChartLayout() {
  const [layout, setLayout] = useState<ChartLayout>(DEFAULT_LAYOUT);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) {
        const v = JSON.parse(raw) as Partial<ChartLayout>;
        const panes = clamp(Math.round(Number(v.panes) || DEFAULT_LAYOUT.panes), MIN_PANES, MAX_PANES);
        const height = clamp(Number(v.height) || DEFAULT_H, MIN_H, MAX_H);
        const shares = Array.isArray(v.shares) && v.shares.length === panes
          && v.shares.every((x) => typeof x === "number" && x > 0)
          ? normalize(v.shares) : evenShares(panes);
        setLayout({ panes, height, shares });
      }
    } catch { /* a malformed or older shape falls back to the defaults */ }
    setLoaded(true);
  }, []);

  useEffect(() => {
    if (!loaded) return;
    const t = setTimeout(() => {
      try { localStorage.setItem(KEY, JSON.stringify(layout)); } catch { /* quota / private mode */ }
    }, 400);
    return () => clearTimeout(t);
  }, [loaded, layout]);

  const setPanes = useCallback((n: number) => setLayout((l) => {
    const panes = clamp(n, MIN_PANES, MAX_PANES);
    // Re-even rather than trying to preserve shares across a count change: a
    // dropped pane's share has no obvious owner and a new one has no share.
    return panes === l.panes ? l : { ...l, panes, shares: evenShares(panes) };
  }), []);

  return { layout, setLayout, setPanes };
}

function normalize(shares: number[]): number[] {
  const total = shares.reduce((a, b) => a + b, 0);
  return total > 0 ? shares.map((s) => s / total) : evenShares(shares.length);
}

export function ChartPaneCount(
  { panes, onChange }: { panes: number; onChange: (n: number) => void },
) {
  return (
    <div className="flex items-center gap-1">
      <span className="text-[10px] text-gray-500">Charts</span>
      {Array.from({ length: MAX_PANES }, (_, i) => i + 1).map((n) => (
        <button
          key={n}
          onClick={() => onChange(n)}
          title={`Show ${n} chart${n > 1 ? "s" : ""}`}
          className={`text-[10px] w-5 py-0.5 rounded transition-colors ${
            panes === n ? "bg-blue-700 text-blue-100" : "bg-gray-700 hover:bg-gray-600 text-gray-300"
          }`}
        >{n}</button>
      ))}
    </div>
  );
}

export default function ResizableChartRow(
  {
    layout, onLayoutChange, renderPane,
  }: {
    layout: ChartLayout;
    onLayoutChange: (next: ChartLayout) => void;
    renderPane: (index: number, height: number) => React.ReactNode;
  },
) {
  const [wide, setWide] = useState(false);
  const rowRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ kind: "x"; edge: number } | { kind: "y"; startY: number; startH: number } | null>(null);
  const latest = useRef(layout);
  latest.current = layout;

  useEffect(() => {
    const mq = window.matchMedia(WIDE);
    const apply = () => setWide(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  const onMove = useCallback((e: PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const l = latest.current;
    if (d.kind === "y") {
      onLayoutChange({ ...l, height: clamp(d.startH + (e.clientY - d.startY), MIN_H, MAX_H) });
      return;
    }
    const row = rowRef.current;
    if (!row) return;
    const r = row.getBoundingClientRect();
    if (r.width <= 0) return;
    // Only the two panes either side of the dragged edge change; everything
    // beyond it keeps its share, so one drag cannot reflow the whole row.
    const before = l.shares.slice(0, d.edge).reduce((a, b) => a + b, 0);
    const pair = l.shares[d.edge] + l.shares[d.edge + 1];
    const want = (e.clientX - r.left) / r.width - before;
    const first = clamp(want, MIN_SHARE, pair - MIN_SHARE);
    const shares = [...l.shares];
    shares[d.edge] = first;
    shares[d.edge + 1] = pair - first;
    onLayoutChange({ ...l, shares });
  }, [onLayoutChange]);

  useEffect(() => {
    const up = () => { drag.current = null; document.body.style.userSelect = ""; };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", up);
    };
  }, [onMove]);

  const startX = (edge: number) => (e: React.PointerEvent) => {
    e.preventDefault();
    drag.current = { kind: "x", edge };
    document.body.style.userSelect = "none";
  };
  const startY = (e: React.PointerEvent) => {
    e.preventDefault();
    drag.current = { kind: "y", startY: e.clientY, startH: latest.current.height };
    document.body.style.userSelect = "none";
  };

  const resetShares = () => onLayoutChange({ ...layout, shares: evenShares(layout.panes) });
  const resetHeight = () => onLayoutChange({ ...layout, height: DEFAULT_H });

  const split = wide && layout.panes > 1;

  return (
    <div className="mb-3">
      <div ref={rowRef} className={split ? "flex items-stretch" : "flex flex-col gap-3"}>
        {Array.from({ length: layout.panes }, (_, i) => (
          <div key={i} className="contents">
            {split && i > 0 && (
              <div
                onPointerDown={startX(i - 1)}
                onDoubleClick={resetShares}
                title="Drag to resize · double-click to even out"
                className="w-3 shrink-0 cursor-col-resize flex items-center justify-center group"
              >
                <div className="w-px h-full bg-gray-700 group-hover:bg-blue-500 transition-colors" />
              </div>
            )}
            <div
              className="min-w-0"
              style={split ? { width: `${(layout.shares[i] ?? 1 / layout.panes) * 100}%` } : undefined}
            >
              {renderPane(i, layout.height)}
            </div>
          </div>
        ))}
      </div>
      <div
        onPointerDown={startY}
        onDoubleClick={resetHeight}
        title="Drag to resize every chart · double-click to reset"
        className="h-3 cursor-row-resize flex items-center justify-center group"
      >
        <div className="h-px w-16 bg-gray-700 group-hover:bg-blue-500 transition-colors" />
      </div>
    </div>
  );
}
