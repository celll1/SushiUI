"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { BrowserImageEntry, browserImageUrl } from "@/utils/api";

interface ThumbnailGridProps {
  images: BrowserImageEntry[];
  selectedIds: Set<string>;
  primaryId: string | null;
  onSelectMulti: (rel_path: string, mods: { ctrl: boolean; shift: boolean }) => void;
  taggedSet?: Set<string>;
}

const CARD_H = 128;
const BUFFER_ROWS = 4;

export default function ThumbnailGrid({
  images,
  selectedIds,
  primaryId,
  onSelectMulti,
  taggedSet,
}: ThumbnailGridProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [cols, setCols] = useState(4);

  const [visibleRange, setVisibleRange] = useState({ startRow: 0, endRow: 20 });

  const scrollTopRef = useRef(0);
  const viewHRef = useRef(600);
  const colsRef = useRef(4);
  const rafRef = useRef<number | null>(null);

  const computeRange = useCallback(
    (st: number, vh: number, c: number, total: number) => {
      const totalRows = Math.ceil(total / c);
      const start = Math.max(0, Math.floor(st / CARD_H) - BUFFER_ROWS);
      const end = Math.min(
        totalRows,
        Math.ceil((st + vh) / CARD_H) + BUFFER_ROWS
      );
      return { startRow: start, endRow: end };
    },
    []
  );

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      viewHRef.current = el.clientHeight;
      const w = el.clientWidth;
      const c = w < 480 ? 3 : w < 768 ? 4 : 5;
      colsRef.current = c;
      setCols(c);
      const range = computeRange(scrollTopRef.current, el.clientHeight, c, images.length);
      setVisibleRange(range);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [images.length, computeRange]);

  useEffect(() => {
    const range = computeRange(scrollTopRef.current, viewHRef.current, colsRef.current, images.length);
    setVisibleRange(range);
  }, [images.length, computeRange]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    scrollTopRef.current = el.scrollTop;

    if (rafRef.current !== null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const next = computeRange(scrollTopRef.current, viewHRef.current, colsRef.current, images.length);
      setVisibleRange((prev) => {
        if (prev.startRow === next.startRow && prev.endRow === next.endRow) return prev;
        return next;
      });
    });
  }, [images.length, computeRange]);

  useEffect(
    () => () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    },
    []
  );

  if (images.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
        画像がありません
      </div>
    );
  }

  const { startRow, endRow } = visibleRange;
  const totalRows = Math.ceil(images.length / cols);
  const totalH = totalRows * CARD_H;
  const visibleStart = startRow * cols;
  const visibleEnd = Math.min(images.length, endRow * cols);
  const visibleItems = images.slice(visibleStart, visibleEnd);
  const topPad = startRow * CARD_H;

  const gridCols =
    cols === 3 ? "grid-cols-3" : cols === 4 ? "grid-cols-4" : "grid-cols-5";

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto min-h-0"
      onScroll={handleScroll}
    >
      <div style={{ height: totalH, position: "relative" }}>
        <div
          style={{ position: "absolute", top: topPad, left: 0, right: 0 }}
          className={`grid ${gridCols} gap-1 p-1`}
        >
          {visibleItems.map((img) => {
            const isPrimary = img.rel_path === primaryId;
            const isSelected = selectedIds.has(img.rel_path);
            const hasTagsNow =
              taggedSet !== undefined ? taggedSet.has(img.rel_path) : img.has_tags;
            return (
              <div
                key={img.rel_path}
                onClick={(e) =>
                  onSelectMulti(img.rel_path, { ctrl: e.ctrlKey || e.metaKey, shift: e.shiftKey })
                }
                className={`relative cursor-pointer rounded overflow-hidden border-2 transition-colors ${
                  isPrimary
                    ? "border-blue-500"
                    : isSelected
                    ? "border-blue-300"
                    : "border-transparent hover:border-gray-500"
                }`}
                style={{ height: CARD_H - 4 }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={browserImageUrl(img.rel_path, 160)}
                  alt={img.rel_path}
                  className="w-full object-cover"
                  style={{ height: CARD_H - 24 }}
                  loading="lazy"
                  decoding="async"
                  fetchPriority="low"
                />
                <div className="text-xs truncate px-1 text-gray-400 leading-5">
                  {img.rel_path.split(/[\\/]/).pop()}
                </div>
                {hasTagsNow && (
                  <div className="absolute top-1 right-1 w-2 h-2 bg-green-500 rounded-full" />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
