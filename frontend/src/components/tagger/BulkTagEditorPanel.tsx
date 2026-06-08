"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import {
  BrowserImageEntry,
  browserImageUrl,
  browserGetTags,
  browserSaveTags,
} from "@/utils/api";
import InputWithTagSuggestions from "@/components/common/InputWithTagSuggestions";
import { useTagSuggestions } from "@/contexts/TagSuggestionsContext";

interface BulkTagEditorPanelProps {
  selectedImages: BrowserImageEntry[];
  onTagsSaved: (updates: Array<{ relPath: string; hasTags: boolean }>) => void;
  onDeselectAll: () => void;
}

const CATEGORY_ORDER = [
  "Copyright",
  "Character",
  "Artist",
  "General",
  "Meta",
  "Quality",
  "Rating",
  "Unknown",
] as const;

type CategoryName = typeof CATEGORY_ORDER[number];

const CATEGORY_COLORS: Record<string, string> = {
  Copyright: "#c084fc",
  Character: "#60a5fa",
  Artist: "#f472b6",
  General: "#4ade80",
  Meta: "#9ca3af",
  Quality: "#facc15",
  Rating: "#fb923c",
  Unknown: "#6b7280",
};

export default function BulkTagEditorPanel({
  selectedImages,
  onTagsSaved,
  onDeselectAll,
}: BulkTagEditorPanelProps) {
  // rel_path → tags loaded from disk
  const [loadedTags, setLoadedTags] = useState<Map<string, string[]>>(new Map());
  const [loading, setLoading] = useState(false);
  const [inputValue, setInputValue] = useState("");
  // Pending bulk operations (staged, not yet applied)
  const [bulkAdd, setBulkAdd] = useState<Map<string, string>>(new Map()); // tag → category
  const [bulkRemove, setBulkRemove] = useState<Set<string>>(new Set());
  const [applying, setApplying] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [applyError, setApplyError] = useState<string | null>(null);

  const tagSuggestionsCtx = useTagSuggestions();

  // Fetch tags for all selected images in parallel
  useEffect(() => {
    setLoading(true);
    setBulkAdd(new Map());
    setBulkRemove(new Set());
    setApplyError(null);
    setInputValue("");
    Promise.all(
      selectedImages.map((img) =>
        browserGetTags(img.rel_path)
          .then(({ tags }) => [img.rel_path, tags] as const)
          .catch(() => [img.rel_path, []] as const)
      )
    ).then((results) => {
      const map = new Map<string, string[]>();
      for (const [rp, tags] of results) map.set(rp, tags);
      setLoadedTags(map);
      setLoading(false);
    });
  }, [selectedImages.map((i) => i.rel_path).join("\0")]); // eslint-disable-line react-hooks/exhaustive-deps

  // Union of all tags with coverage count
  const tagCoverage = useMemo(() => {
    const total = selectedImages.length;
    const counts = new Map<string, number>();
    for (const tags of loadedTags.values()) {
      for (const t of tags) counts.set(t, (counts.get(t) ?? 0) + 1);
    }
    // Also include bulkAdd tags (coverage 0 initially)
    for (const t of bulkAdd.keys()) {
      if (!counts.has(t)) counts.set(t, 0);
    }
    return { counts, total };
  }, [loadedTags, bulkAdd, selectedImages.length]);

  // Resolve categories for all union tags
  const [tagCategories, setTagCategories] = useState<Map<string, string>>(new Map());
  useEffect(() => {
    const allTags = [...tagCoverage.counts.keys()];
    if (allTags.length === 0) return;
    tagSuggestionsCtx.getCategoriesForTags(allTags)
      .then(setTagCategories)
      .catch(() => {});
  }, [tagCoverage.counts.size]); // eslint-disable-line react-hooks/exhaustive-deps

  // Group tags by category
  const groupedTags = useMemo(() => {
    const groups = new Map<CategoryName, string[]>(CATEGORY_ORDER.map((c) => [c, []]));
    for (const tag of tagCoverage.counts.keys()) {
      const raw = tagCategories.get(tag) ?? "Unknown";
      const cat: CategoryName = CATEGORY_ORDER.includes(raw as CategoryName)
        ? (raw as CategoryName)
        : "Unknown";
      groups.get(cat)!.push(tag);
    }
    for (const arr of groups.values()) arr.sort();
    return groups;
  }, [tagCoverage.counts, tagCategories]);

  const handleTagClick = useCallback((tag: string) => {
    if (bulkRemove.has(tag)) {
      // Cancel remove
      setBulkRemove((prev) => { const n = new Set(prev); n.delete(tag); return n; });
    } else if (bulkAdd.has(tag)) {
      // Cancel add
      setBulkAdd((prev) => { const n = new Map(prev); n.delete(tag); return n; });
    } else {
      const { counts, total } = tagCoverage;
      const count = counts.get(tag) ?? 0;
      if (count === total) {
        // All have it → stage remove
        setBulkRemove((prev) => new Set([...prev, tag]));
      } else {
        // Partial or none → stage add
        setBulkAdd((prev) => new Map([...prev, [tag, tagCategories.get(tag) ?? ""]]));
      }
    }
  }, [bulkAdd, bulkRemove, tagCoverage, tagCategories]);

  const handleAddNew = useCallback((tag: string, category?: string) => {
    const normalized = tag.trim().replace(/ /g, "_");
    if (!normalized) return;
    setBulkAdd((prev) => new Map([...prev, [normalized, category ?? ""]]));
    setBulkRemove((prev) => { const n = new Set(prev); n.delete(normalized); return n; });
    setInputValue("");
  }, []);

  const handleApply = useCallback(async () => {
    if (applying || selectedImages.length === 0) return;
    setApplying(true);
    setApplyError(null);
    setProgress({ done: 0, total: selectedImages.length });

    const updates: Array<{ relPath: string; hasTags: boolean }> = [];
    let errorCount = 0;

    for (const img of selectedImages) {
      const current = loadedTags.get(img.rel_path) ?? [];
      // Apply: remove first, then add
      let next = current.filter((t) => !bulkRemove.has(t));
      for (const t of bulkAdd.keys()) {
        if (!next.includes(t)) next = [...next, t];
      }
      try {
        await browserSaveTags(img.rel_path, next);
        updates.push({ relPath: img.rel_path, hasTags: next.length > 0 });
        // Update loadedTags in-place for coverage recalc
        setLoadedTags((prev) => new Map([...prev, [img.rel_path, next]]));
      } catch {
        errorCount++;
      }
      setProgress((p) => p ? { ...p, done: p.done + 1 } : p);
    }

    setBulkAdd(new Map());
    setBulkRemove(new Set());
    setApplying(false);
    setProgress(null);

    if (errorCount > 0) setApplyError(`${errorCount} 件の保存に失敗しました`);
    if (updates.length > 0) onTagsSaved(updates);
  }, [applying, selectedImages, loadedTags, bulkAdd, bulkRemove, onTagsSaved]);

  const hasPending = bulkAdd.size > 0 || bulkRemove.size > 0;

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header: thumbnail strip + deselect */}
      <div className="flex items-center gap-2 px-2 py-1.5 border-b border-gray-700 flex-shrink-0">
        <div className="flex gap-1 overflow-x-auto flex-1 min-w-0">
          {selectedImages.map((img) => (
            <img
              key={img.rel_path}
              src={browserImageUrl(img.rel_path, 48)}
              alt={img.rel_path}
              className="w-10 h-10 object-cover rounded flex-shrink-0 border border-gray-600"
              // eslint-disable-next-line @next/next/no-img-element
              loading="lazy"
              decoding="async"
            />
          ))}
        </div>
        <span className="text-xs text-gray-400 flex-shrink-0">
          {selectedImages.length} 枚選択中
        </span>
        <button
          onClick={onDeselectAll}
          className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded flex-shrink-0"
        >
          解除
        </button>
      </div>

      {/* Tag input */}
      <div className="px-2 pt-2 flex-shrink-0">
        <InputWithTagSuggestions
          value={inputValue}
          onChange={setInputValue}
          onTagAdd={handleAddNew}
          placeholder="追加するタグを入力..."
          showSuggestionsAbove={false}
          className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-600 rounded text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
        />
      </div>

      {/* Apply bar */}
      <div className="px-2 py-1.5 flex-shrink-0 flex items-center gap-2">
        <button
          onClick={handleApply}
          disabled={!hasPending || applying}
          className="px-3 py-1 text-sm bg-blue-600 hover:bg-blue-500 disabled:opacity-40 rounded"
        >
          {applying ? "適用中..." : "適用"}
        </button>
        {hasPending && (
          <span className="text-xs text-gray-400">
            {bulkAdd.size > 0 && `+${bulkAdd.size} 追加`}
            {bulkAdd.size > 0 && bulkRemove.size > 0 && " / "}
            {bulkRemove.size > 0 && `−${bulkRemove.size} 削除`}
          </span>
        )}
        {progress && (
          <div className="flex-1 min-w-0">
            <div className="w-full bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full transition-all"
                style={{ width: `${(progress.done / progress.total) * 100}%` }}
              />
            </div>
          </div>
        )}
        {applyError && <span className="text-red-400 text-xs">{applyError}</span>}
      </div>

      {/* Category-grouped union tags (scrollable) */}
      <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-2">
        {loading ? (
          <span className="text-gray-500 text-sm">読込中...</span>
        ) : tagCoverage.counts.size === 0 ? (
          <span className="text-gray-600 text-sm">タグなし</span>
        ) : (
          CATEGORY_ORDER.map((cat) => {
            const catTags = groupedTags.get(cat) ?? [];
            if (catTags.length === 0) return null;
            const color = CATEGORY_COLORS[cat];
            return (
              <div key={cat} className="mb-3">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                  <span className="text-xs font-semibold text-gray-300">{cat}</span>
                  <span className="text-xs text-gray-600">({catTags.length})</span>
                </div>
                <div className="flex flex-wrap gap-1.5 pl-4">
                  {catTags.map((tag) => {
                    const count = tagCoverage.counts.get(tag) ?? 0;
                    const total = tagCoverage.total;
                    const isStagedAdd = bulkAdd.has(tag);
                    const isStagedRemove = bulkRemove.has(tag);

                    let chipStyle: React.CSSProperties = {
                      backgroundColor: "#374151",
                      borderLeft: `3px solid ${color}`,
                    };
                    if (isStagedAdd) chipStyle = { ...chipStyle, backgroundColor: "#14532d", outline: "1px solid #4ade80" };
                    if (isStagedRemove) chipStyle = { ...chipStyle, backgroundColor: "#450a0a", outline: "1px solid #f87171", textDecoration: "line-through" };

                    return (
                      <button
                        key={tag}
                        onClick={() => handleTagClick(tag)}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-sm text-white hover:brightness-125 transition-all"
                        style={chipStyle}
                        title={
                          isStagedAdd ? "追加待ち（クリックでキャンセル）" :
                          isStagedRemove ? "削除待ち（クリックでキャンセル）" :
                          count === total ? "全画像に存在（クリックで削除待ちに）" :
                          `${count}/${total} 枚に存在（クリックで追加待ちに）`
                        }
                      >
                        {tag}
                        {count < total && !isStagedAdd && !isStagedRemove && (
                          <span className="text-xs text-gray-400 ml-0.5">
                            {count}/{total}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
