"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { BrowserImageEntry, browserImageUrl } from "@/utils/api";

interface ThumbnailGridProps {
  images: BrowserImageEntry[];
  selectedIdx: number | null;
  onSelect: (idx: number) => void;
  /** Called when has_tags state changes (e.g. after auto-save) */
  taggedSet?: Set<string>;
}

const CARD_H = 128; // px per row
const BUFFER_ROWS = 4;

export default function ThumbnailGrid({
  images,
  selectedIdx,
  onSelect,
  taggedSet,
}: ThumbnailGridProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewH, setViewH] = useState(600);
  const [cols, setCols] = useState(4);

  // Measure container dimensions
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setViewH(el.clientHeight);
      const w = el.clientWidth;
      // Responsive cols: <480 → 3, <768 → 4, else → 5
      setCols(w < 480 ? 3 : w < 768 ? 4 : 5);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const handleScroll = useCallback(() => {
    setScrollTop(scrollRef.current?.scrollTop ?? 0);
  }, []);

  // Virtual scroll calculations
  const totalRows = Math.ceil(images.length / cols);
  const totalH = totalRows * CARD_H;
  const startRow = Math.max(0, Math.floor(scrollTop / CARD_H) - BUFFER_ROWS);
  const endRow = Math.min(
    totalRows,
    Math.ceil((scrollTop + viewH) / CARD_H) + BUFFER_ROWS
  );
  const visibleStart = startRow * cols;
  const visibleEnd = Math.min(images.length, endRow * cols);
  const visibleItems = images.slice(visibleStart, visibleEnd);
  const topPad = startRow * CARD_H;
  const bottomPad = Math.max(0, (totalRows - endRow) * CARD_H);

  if (images.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
        画像がありません
      </div>
    );
  }

  const gridCols =
    cols === 3
      ? "grid-cols-3"
      : cols === 4
        ? "grid-cols-4"
        : "grid-cols-5";

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto min-h-0"
      onScroll={handleScroll}
    >
      <div style={{ height: totalH, position: "relative" }}>
        <div
          style={{
            position: "absolute",
            top: topPad,
            left: 0,
            right: 0,
          }}
          className={`grid ${gridCols} gap-1 p-1`}
        >
          {visibleItems.map((img, localIdx) => {
            const idx = visibleStart + localIdx;
            const selected = idx === selectedIdx;
            const hasTagsNow =
              taggedSet !== undefined
                ? taggedSet.has(img.path)
                : img.has_tags;
            return (
              <div
                key={img.path}
                onClick={() => onSelect(idx)}
                className={`relative cursor-pointer rounded overflow-hidden border-2 transition-colors ${
                  selected
                    ? "border-blue-500"
                    : "border-transparent hover:border-gray-500"
                }`}
                style={{ height: CARD_H - 4 }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={browserImageUrl(img.path, 160)}
                  alt={img.rel_path}
                  className="w-full object-cover"
                  style={{ height: CARD_H - 24 }}
                  loading="lazy"
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
        {/* Bottom spacer */}
        <div style={{ height: bottomPad }} />
      </div>
    </div>
  );
}
