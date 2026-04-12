"use client";

import React from "react";
import { SigLIP2TagResult } from "@/utils/api";

// ─── Category colour mapping ─────────────────────────────────────────────────

const CATEGORY_BAR_COLOR: Record<string, string> = {
  Quality:   "bg-yellow-500",
  Rating:    "bg-orange-500",
  Character: "bg-blue-500",
  Copyright: "bg-purple-500",
  General:   "bg-green-600",
  Artist:    "bg-pink-500",
  Meta:      "bg-gray-400",
  Unknown:   "bg-gray-600",
};

const CATEGORY_TEXT_COLOR: Record<string, string> = {
  Quality:   "text-yellow-400",
  Rating:    "text-orange-400",
  Character: "text-blue-400",
  Copyright: "text-purple-400",
  General:   "text-green-400",
  Artist:    "text-pink-400",
  Meta:      "text-gray-400",
  Unknown:   "text-gray-500",
};

function barColor(category: string): string {
  return CATEGORY_BAR_COLOR[category] ?? "bg-gray-500";
}

function textColor(category: string): string {
  return CATEGORY_TEXT_COLOR[category] ?? "text-gray-400";
}

// ─── Single bar row ──────────────────────────────────────────────────────────

interface BarRowProps {
  tag: string;
  prob: number;
  category: string;
  selected: boolean;
  onToggle: (tag: string) => void;
}

function BarRow({ tag, prob, category, selected, onToggle }: BarRowProps) {
  const pct = `${(prob * 100).toFixed(1)}%`;

  return (
    <button
      onClick={() => onToggle(tag)}
      className={`w-full flex items-center gap-2 px-2 py-0.5 rounded text-left hover:bg-gray-700 transition-colors ${
        selected ? "bg-gray-700 ring-1 ring-blue-500" : ""
      }`}
    >
      {/* Category badge */}
      <span className={`text-[9px] w-14 shrink-0 ${textColor(category)}`}>
        {category}
      </span>

      {/* Tag name */}
      <span className="text-xs text-gray-200 w-44 shrink-0 truncate" title={tag}>
        {tag}
      </span>

      {/* Bar */}
      <div className="flex-1 h-3 bg-gray-800 rounded overflow-hidden">
        <div
          className={`h-full rounded transition-all ${barColor(category)}`}
          style={{ width: pct }}
        />
      </div>

      {/* Probability */}
      <span className="text-[10px] text-gray-400 w-10 text-right shrink-0">{pct}</span>

      {/* Checkbox indicator */}
      <span className={`text-[10px] w-4 shrink-0 ${selected ? "text-blue-400" : "text-gray-600"}`}>
        {selected ? "✓" : "○"}
      </span>
    </button>
  );
}

// ─── Main chart component ────────────────────────────────────────────────────

export interface TagResultsChartProps {
  tags: SigLIP2TagResult[];
  qualityTop: SigLIP2TagResult | null;
  ratingTop: SigLIP2TagResult | null;
  threshold: number;
  selectedTags: Set<string>;
  onTagToggle: (tag: string) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
}

export default function TagResultsChart({
  tags,
  qualityTop,
  ratingTop,
  threshold,
  selectedTags,
  onTagToggle,
  onSelectAll,
  onDeselectAll,
}: TagResultsChartProps) {
  if (tags.length === 0 && !qualityTop && !ratingTop) {
    return (
      <div className="flex items-center justify-center h-40 text-gray-500 text-sm">
        No tags predicted above threshold {threshold.toFixed(2)}
      </div>
    );
  }

  const pinnedItems = [qualityTop, ratingTop].filter(Boolean) as SigLIP2TagResult[];

  return (
    <div className="space-y-1">
      {/* Legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 px-2 pb-1">
        {Object.entries(CATEGORY_TEXT_COLOR).map(([cat, cls]) => (
          <span key={cat} className={`text-[9px] ${cls}`}>● {cat}</span>
        ))}
      </div>

      {/* Select-all / Deselect-all */}
      <div className="flex gap-2 px-2 pb-1">
        <button
          onClick={onSelectAll}
          className="text-[10px] text-blue-400 hover:text-blue-300 underline"
        >
          Select all
        </button>
        <button
          onClick={onDeselectAll}
          className="text-[10px] text-gray-400 hover:text-gray-300 underline"
        >
          Deselect all
        </button>
        <span className="text-[10px] text-gray-500 ml-auto">
          {selectedTags.size} selected / {tags.length + pinnedItems.length} total
        </span>
      </div>

      {/* Pinned: Quality & Rating (always shown, above threshold line) */}
      {pinnedItems.length > 0 && (
        <div className="border border-gray-700 rounded p-1 mb-1">
          <p className="text-[9px] text-gray-500 px-2 pb-0.5">Quality / Rating (always shown)</p>
          {pinnedItems.map((item) => (
            <BarRow
              key={item.tag}
              tag={item.tag}
              prob={item.prob}
              category={item.category}
              selected={selectedTags.has(item.tag)}
              onToggle={onTagToggle}
            />
          ))}
        </div>
      )}

      {/* Threshold separator */}
      {tags.length > 0 && (
        <div className="flex items-center gap-2 px-2 py-0.5">
          <div className="h-px flex-1 bg-yellow-600 opacity-60" />
          <span className="text-[9px] text-yellow-600">threshold {threshold.toFixed(2)}</span>
          <div className="h-px flex-1 bg-yellow-600 opacity-60" />
        </div>
      )}

      {/* Threshold-filtered tags */}
      <div className="space-y-0.5 max-h-[60vh] overflow-y-auto">
        {tags.map((item) => (
          <BarRow
            key={item.tag}
            tag={item.tag}
            prob={item.prob}
            category={item.category}
            selected={selectedTags.has(item.tag)}
            onToggle={onTagToggle}
          />
        ))}
      </div>
    </div>
  );
}
