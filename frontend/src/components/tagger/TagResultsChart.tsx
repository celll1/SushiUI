"use client";

import React, { useState, useEffect } from "react";
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
      className={`w-full flex items-center gap-1.5 px-2 py-0.5 rounded text-left hover:bg-gray-700 transition-colors ${
        selected ? "bg-gray-700 ring-1 ring-blue-500" : ""
      }`}
    >
      {showCategory && (
        <span className={`text-xs w-14 shrink-0 ${textColor(category)}`}>
          {category}
        </span>
      )}
      <span className="text-sm text-gray-200 flex-1 truncate min-w-0" title={tag}>
        {tag}
      </span>
      {/* Bar — flexible width */}
      <div className="w-20 h-3 bg-gray-800 rounded overflow-hidden shrink-0">
        <div
          className={`h-full rounded transition-all ${barColor(category)}`}
          style={{ width: pct }}
        />
      </div>
      <span className="text-xs text-gray-400 w-10 text-right shrink-0">{pct}</span>
      <span className={`text-xs w-3.5 shrink-0 ${selected ? "text-blue-400" : "text-gray-600"}`}>
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
  displayProb: (item: SigLIP2TagResult) => number;
}

function CategoryGroup({ category, items, selectedTags, onToggle, onSelectGroup, onDeselectGroup, displayProb }: CategoryGroupProps) {
  const [collapsed, setCollapsed] = useState(false);
  const tagNames = items.map(t => t.tag);
  const selectedCount = tagNames.filter(t => selectedTags.has(t)).length;

  return (
    <div className="border border-gray-700 rounded mb-2">
      <div className="flex items-center gap-2 px-2 py-1 bg-gray-800 rounded-t">
        <button onClick={() => setCollapsed(c => !c)} className="text-gray-400 hover:text-white text-xs">
          {collapsed ? "▶" : "▼"}
        </button>
        <span className={`text-xs font-medium ${textColor(category)}`}>{category}</span>
        <span className="text-xs text-gray-500 ml-1">({selectedCount}/{items.length})</span>
        <div className="ml-auto flex gap-2">
          <button onClick={() => onSelectGroup(tagNames)} className="text-xs text-blue-400 hover:text-blue-300 underline">All</button>
          <button onClick={() => onDeselectGroup(tagNames)} className="text-xs text-gray-400 hover:text-gray-300 underline">None</button>
        </div>
      </div>
      {!collapsed && (
        <div className="p-1 space-y-0.5">
          {items.map(item => (
            <BarRow
              key={item.tag}
              tag={item.tag}
              prob={displayProb(item)}
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
  hasCalibration?: boolean;
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
  hasCalibration = false,
}: TagResultsChartProps) {
  const [viewMode, setViewMode] = useState<"flat" | "grouped">(() => {
    try { return (localStorage.getItem("tagger_view_mode") as "flat" | "grouped") || "grouped"; }
    catch { return "grouped"; }
  });
  const [numCols, setNumCols] = useState<1 | 2>(() => {
    try { return (parseInt(localStorage.getItem("tagger_num_cols") ?? "2") as 1 | 2); }
    catch { return 2; }
  });
  const [showCal, setShowCal] = useState<boolean>(() => {
    try { return localStorage.getItem("tagger_show_cal") === "1"; }
    catch { return false; }
  });
  useEffect(() => { localStorage.setItem("tagger_view_mode", viewMode); }, [viewMode]);
  useEffect(() => { localStorage.setItem("tagger_num_cols", String(numCols)); }, [numCols]);
  useEffect(() => { localStorage.setItem("tagger_show_cal", showCal ? "1" : "0"); }, [showCal]);

  // Resolve display probability: cal_prob when showCal and available, else prob
  const displayProb = (item: SigLIP2TagResult) =>
    showCal && item.cal_prob != null ? item.cal_prob : item.prob;

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

  const orderedCategories = [
    ...CATEGORY_ORDER.filter(c => c !== "Quality" && c !== "Rating" && grouped[c]?.length > 0),
    ...Object.keys(grouped).filter(c => !CATEGORY_ORDER.includes(c)),
  ];

  return (
    <div className="space-y-1">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-2 pb-1 flex-wrap">
        <button onClick={onSelectAll} className="text-xs text-blue-400 hover:text-blue-300 underline">
          Select all
        </button>
        <button onClick={onDeselectAll} className="text-xs text-gray-400 hover:text-gray-300 underline">
          Deselect all
        </button>
        <span className="text-xs text-gray-500">
          {selectedTags.size} / {totalCount}
        </span>
        <div className="ml-auto flex gap-1">
          {/* Raw / Cal toggle (only when calibration is available) */}
          {hasCalibration && (
            <div className="flex rounded overflow-hidden border border-gray-600 text-xs">
              <button
                onClick={() => setShowCal(false)}
                className={`px-2 py-0.5 ${!showCal ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
                title="Raw sigmoid probabilities"
              >
                Raw
              </button>
              <button
                onClick={() => setShowCal(true)}
                className={`px-2 py-0.5 ${showCal ? "bg-blue-700 text-white" : "text-gray-400 hover:bg-gray-700"}`}
                title="Jeffreys-calibrated probabilities"
              >
                Cal
              </button>
            </div>
          )}
          {/* View mode */}
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
          {/* Column toggle */}
          <div className="flex rounded overflow-hidden border border-gray-600 text-xs">
            <button
              onClick={() => setNumCols(1)}
              className={`px-2 py-0.5 ${numCols === 1 ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
              title="1 column"
            >
              ▌
            </button>
            <button
              onClick={() => setNumCols(2)}
              className={`px-2 py-0.5 ${numCols === 2 ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
              title="2 columns"
            >
              ▌▌
            </button>
          </div>
        </div>
      </div>

      {/* Flat view */}
      {viewMode === "flat" && (
        <>
          {pinnedItems.length > 0 && (
            <div className="border border-gray-700 rounded p-1 mb-1">
              <p className="text-xs text-gray-500 px-2 pb-0.5">Quality / Rating</p>
              {pinnedItems.map((item) => (
                <BarRow key={item.tag} tag={item.tag} prob={displayProb(item)} category={item.category}
                  selected={selectedTags.has(item.tag)} onToggle={onTagToggle} />
              ))}
            </div>
          )}
          <div className="flex items-center gap-2 px-2 py-0.5">
            <div className="h-px flex-1 bg-yellow-600 opacity-60" />
            <span className="text-xs text-yellow-600">threshold {threshold.toFixed(2)}</span>
            <div className="h-px flex-1 bg-yellow-600 opacity-60" />
          </div>
          {numCols === 1 ? (
            <div className="space-y-0.5">
              {tags.map(item => (
                <BarRow key={item.tag} tag={item.tag} prob={displayProb(item)} category={item.category}
                  selected={selectedTags.has(item.tag)} onToggle={onTagToggle} />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-x-2">
              {[tags.slice(0, Math.ceil(tags.length / 2)), tags.slice(Math.ceil(tags.length / 2))].map((half, ci) => (
                <div key={ci} className="space-y-0.5 min-w-0">
                  {half.map(item => (
                    <BarRow key={item.tag} tag={item.tag} prob={displayProb(item)} category={item.category}
                      selected={selectedTags.has(item.tag)} onToggle={onTagToggle} />
                  ))}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Grouped view */}
      {viewMode === "grouped" && (
        <>
          {/* Quality / Rating always single row */}
          {pinnedItems.length > 0 && (
            <div className={numCols === 2 ? "grid grid-cols-2 gap-2" : ""}>
              {qualityTop && (
                <CategoryGroup category="Quality" items={[qualityTop]} selectedTags={selectedTags}
                  onToggle={onTagToggle} onSelectGroup={handleSelectGroup} onDeselectGroup={handleDeselectGroup}
                  displayProb={displayProb} />
              )}
              {ratingTop && (
                <CategoryGroup category="Rating" items={[ratingTop]} selectedTags={selectedTags}
                  onToggle={onTagToggle} onSelectGroup={handleSelectGroup} onDeselectGroup={handleDeselectGroup}
                  displayProb={displayProb} />
              )}
            </div>
          )}
          {numCols === 1 ? (
            orderedCategories.map(cat => (
              <CategoryGroup key={cat} category={cat} items={grouped[cat]} selectedTags={selectedTags}
                onToggle={onTagToggle} onSelectGroup={handleSelectGroup} onDeselectGroup={handleDeselectGroup}
                displayProb={displayProb} />
            ))
          ) : (
            <div className="grid grid-cols-2 gap-x-2 items-start">
              {orderedCategories.map(cat => (
                <CategoryGroup key={cat} category={cat} items={grouped[cat]} selectedTags={selectedTags}
                  onToggle={onTagToggle} onSelectGroup={handleSelectGroup} onDeselectGroup={handleDeselectGroup}
                  displayProb={displayProb} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
