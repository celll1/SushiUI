"use client";

import React, { useState } from "react";
import { SigLIP2TagResult } from "@/utils/api";

// ─── Category colour mapping ─────────────────────────────────────────────────

const CATEGORY_ORDER = [
  "Quality", "Rating", "Character", "Copyright", "General", "Artist", "Meta", "Unknown",
];

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
  showCategory?: boolean;
}

function BarRow({ tag, prob, category, selected, onToggle, showCategory = true }: BarRowProps) {
  const pct = `${(prob * 100).toFixed(1)}%`;

  return (
    <button
      onClick={() => onToggle(tag)}
      className={`w-full flex items-center gap-2 px-2 py-1 rounded text-left hover:bg-gray-700 transition-colors ${
        selected ? "bg-gray-700 ring-1 ring-blue-500" : ""
      }`}
    >
      {/* Category badge */}
      {showCategory && (
        <span className={`text-xs w-16 shrink-0 ${textColor(category)}`}>
          {category}
        </span>
      )}

      {/* Tag name */}
      <span className="text-sm text-gray-200 flex-1 truncate" title={tag}>
        {tag}
      </span>

      {/* Bar */}
      <div className="w-32 h-3.5 bg-gray-800 rounded overflow-hidden shrink-0">
        <div
          className={`h-full rounded transition-all ${barColor(category)}`}
          style={{ width: pct }}
        />
      </div>

      {/* Probability */}
      <span className="text-xs text-gray-400 w-12 text-right shrink-0">{pct}</span>

      {/* Checkbox indicator */}
      <span className={`text-sm w-4 shrink-0 ${selected ? "text-blue-400" : "text-gray-600"}`}>
        {selected ? "✓" : "○"}
      </span>
    </button>
  );
}

// ─── Category group section ───────────────────────────────────────────────────

interface CategoryGroupProps {
  category: string;
  items: SigLIP2TagResult[];
  selectedTags: Set<string>;
  onToggle: (tag: string) => void;
  onSelectGroup: (tags: string[]) => void;
  onDeselectGroup: (tags: string[]) => void;
}

function CategoryGroup({ category, items, selectedTags, onToggle, onSelectGroup, onDeselectGroup }: CategoryGroupProps) {
  const [collapsed, setCollapsed] = useState(false);
  const tagNames = items.map(t => t.tag);
  const selectedCount = tagNames.filter(t => selectedTags.has(t)).length;

  return (
    <div className="border border-gray-700 rounded mb-2">
      <div className="flex items-center gap-2 px-2 py-1.5 bg-gray-800 rounded-t">
        <button onClick={() => setCollapsed(c => !c)} className="text-gray-400 hover:text-white text-xs">
          {collapsed ? "▶" : "▼"}
        </button>
        <span className={`text-sm font-medium ${textColor(category)}`}>{category}</span>
        <span className="text-xs text-gray-500 ml-1">({selectedCount}/{items.length})</span>
        <div className="ml-auto flex gap-2">
          <button
            onClick={() => onSelectGroup(tagNames)}
            className="text-xs text-blue-400 hover:text-blue-300 underline"
          >
            All
          </button>
          <button
            onClick={() => onDeselectGroup(tagNames)}
            className="text-xs text-gray-400 hover:text-gray-300 underline"
          >
            None
          </button>
        </div>
      </div>
      {!collapsed && (
        <div className="p-1 space-y-0.5">
          {items.map(item => (
            <BarRow
              key={item.tag}
              tag={item.tag}
              prob={item.prob}
              category={item.category}
              selected={selectedTags.has(item.tag)}
              onToggle={onToggle}
              showCategory={false}
            />
          ))}
        </div>
      )}
    </div>
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
  const [viewMode, setViewMode] = useState<"flat" | "grouped">("flat");

  const handleSelectGroup = (tagNames: string[]) => {
    tagNames.forEach(t => { if (!selectedTags.has(t)) onTagToggle(t); });
  };
  const handleDeselectGroup = (tagNames: string[]) => {
    tagNames.forEach(t => { if (selectedTags.has(t)) onTagToggle(t); });
  };

  const pinnedItems = [qualityTop, ratingTop].filter(Boolean) as SigLIP2TagResult[];
  const totalCount = tags.length + pinnedItems.length;

  if (totalCount === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-gray-400 text-sm">
        No tags predicted above threshold {threshold.toFixed(2)}
      </div>
    );
  }

  // Group tags by category
  const grouped: Record<string, SigLIP2TagResult[]> = {};
  for (const t of tags) {
    (grouped[t.category] ??= []).push(t);
  }

  return (
    <div className="space-y-1">
      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 px-2 pb-1">
        {CATEGORY_ORDER.filter(c => CATEGORY_TEXT_COLOR[c]).map((cat) => (
          <span key={cat} className={`text-xs ${textColor(cat)}`}>● {cat}</span>
        ))}
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-3 px-2 pb-1">
        <button onClick={onSelectAll} className="text-xs text-blue-400 hover:text-blue-300 underline">
          Select all
        </button>
        <button onClick={onDeselectAll} className="text-xs text-gray-400 hover:text-gray-300 underline">
          Deselect all
        </button>
        <span className="text-xs text-gray-500 ml-auto mr-2">
          {selectedTags.size} / {totalCount}
        </span>
        {/* View mode toggle */}
        <div className="flex rounded overflow-hidden border border-gray-600 text-xs">
          <button
            onClick={() => setViewMode("flat")}
            className={`px-2 py-0.5 ${viewMode === "flat" ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
          >
            Flat
          </button>
          <button
            onClick={() => setViewMode("grouped")}
            className={`px-2 py-0.5 ${viewMode === "grouped" ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
          >
            Grouped
          </button>
        </div>
      </div>

      {/* Pinned: Quality & Rating (always shown) */}
      {pinnedItems.length > 0 && viewMode === "flat" && (
        <div className="border border-gray-700 rounded p-1 mb-1">
          <p className="text-xs text-gray-500 px-2 pb-0.5">Quality / Rating (always shown)</p>
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

      {/* Threshold separator (flat mode only) */}
      {viewMode === "flat" && tags.length > 0 && (
        <div className="flex items-center gap-2 px-2 py-0.5">
          <div className="h-px flex-1 bg-yellow-600 opacity-60" />
          <span className="text-xs text-yellow-600">threshold {threshold.toFixed(2)}</span>
          <div className="h-px flex-1 bg-yellow-600 opacity-60" />
        </div>
      )}

      {/* Flat view */}
      {viewMode === "flat" && (
        <div className="space-y-0.5">
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
      )}

      {/* Grouped view */}
      {viewMode === "grouped" && (
        <div>
          {/* Quality / Rating in their own groups */}
          {pinnedItems.length > 0 && (
            <>
              {qualityTop && (
                <CategoryGroup
                  category="Quality"
                  items={[qualityTop]}
                  selectedTags={selectedTags}
                  onToggle={onTagToggle}
                  onSelectGroup={handleSelectGroup}
                  onDeselectGroup={handleDeselectGroup}
                />
              )}
              {ratingTop && (
                <CategoryGroup
                  category="Rating"
                  items={[ratingTop]}
                  selectedTags={selectedTags}
                  onToggle={onTagToggle}
                  onSelectGroup={handleSelectGroup}
                  onDeselectGroup={handleDeselectGroup}
                />
              )}
            </>
          )}
          {/* Other categories in order */}
          {CATEGORY_ORDER.filter(c => c !== "Quality" && c !== "Rating" && grouped[c]?.length > 0).map(cat => (
            <CategoryGroup
              key={cat}
              category={cat}
              items={grouped[cat]}
              selectedTags={selectedTags}
              onToggle={onTagToggle}
              onSelectGroup={handleSelectGroup}
              onDeselectGroup={handleDeselectGroup}
            />
          ))}
          {/* Any unlisted categories */}
          {Object.keys(grouped)
            .filter(c => !CATEGORY_ORDER.includes(c))
            .map(cat => (
              <CategoryGroup
                key={cat}
                category={cat}
                items={grouped[cat]}
                selectedTags={selectedTags}
                onToggle={onTagToggle}
                onSelectGroup={handleSelectGroup}
                onDeselectGroup={handleDeselectGroup}
              />
            ))}
        </div>
      )}
    </div>
  );
}
