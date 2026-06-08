"use client";

import { useState, useCallback, useRef, useMemo, useEffect } from "react";
import {
  BrowserImageEntry,
  browserSetDirectory,
  browserListImages,
  browserPickDirectory,
  browserBatchInfer,
  browserImageUrl,
  BrowserBatchEvent,
} from "@/utils/api";
import ThumbnailGrid from "./ThumbnailGrid";
import TagEditorPanel from "./TagEditorPanel";
import BulkTagEditorPanel from "./BulkTagEditorPanel";
import ChipInput from "@/components/common/ChipInput";
import { usePanelResize } from "@/hooks/usePanelResize";
import { useTagSuggestions } from "@/contexts/TagSuggestionsContext";
import {
  FilterQuery,
  EMPTY_FILTER,
  isFilterActive,
  needsTagsLoaded,
  compileFilter,
} from "@/utils/browserFilter";

interface DatasetBrowserPanelProps {
  modelLoaded: boolean;
}

type FilterMode = "all" | "tagged" | "untagged";

export default function DatasetBrowserPanel({
  modelLoaded,
}: DatasetBrowserPanelProps) {
  const [dirPath, setDirPath] = useState("");
  const [displayName, setDisplayName] = useState<string | null>(null);
  const [recursive, setRecursive] = useState(false);
  const [images, setImages] = useState<BrowserImageEntry[]>([]);
  const [taggedSet, setTaggedSet] = useState<Set<string>>(new Set());
  // Multi-select state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [primaryId, setPrimaryId] = useState<string | null>(null);
  const [rangeAnchorId, setRangeAnchorId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [filter, setFilter] = useState<FilterMode>("all");
  const [overwriteMode, setOverwriteMode] = useState(false);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState<{
    done: number;
    total: number;
    errors: number;
  } | null>(null);
  const batchCtrlRef = useRef<AbortController | null>(null);
  const [picking, setPicking] = useState(false);

  // Advanced filter
  const [filterQuery, setFilterQuery] = useState<FilterQuery>(EMPTY_FILTER);
  const [filterPanelOpen, setFilterPanelOpen] = useState(false);
  // category map for compileFilter — populated lazily when tags are loaded
  const [filterCategoryMap, setFilterCategoryMap] = useState<Map<string, string>>(new Map());
  const filterDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const tagSuggestionsCtx = useTagSuggestions();

  // Resizable split between grid and editor
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const [gridWidthPx, setGridWidthPx] = useState<number | null>(null);
  const { onMouseDown: onDividerMouseDown } = usePanelResize({
    containerRef: splitContainerRef,
    direction: "horizontal",
    minPx: 160,
    maxRatio: 0.7,
    onResize: setGridWidthPx,
  });

  // --- helpers ---

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
    setPrimaryId(null);
    setRangeAnchorId(null);
  }, []);

  const loadImages = useCallback(
    async (includeTags = false) => {
      setLoading(true);
      setLoadError(null);
      clearSelection();
      try {
        const { images: imgs } = await browserListImages(recursive, includeTags);
        setImages(imgs);
        const tagged = new Set<string>();
        imgs.forEach((img) => {
          if (img.has_tags) tagged.add(img.rel_path);
        });
        setTaggedSet(tagged);

        // Resolve categories for all unique tags (for filter category matching)
        if (includeTags) {
          const allTags = new Set<string>();
          for (const img of imgs) {
            for (const t of img.tags ?? []) allTags.add(t);
          }
          if (allTags.size > 0) {
            tagSuggestionsCtx
              .getCategoriesForTags([...allTags])
              .then((map) => setFilterCategoryMap(map))
              .catch(() => {});
          }
        }
      } catch (e) {
        setLoadError(String(e));
        setImages([]);
      } finally {
        setLoading(false);
      }
    },
    [recursive, clearSelection, tagSuggestionsCtx]
  );

  const handleLoad = useCallback(async () => {
    if (!dirPath.trim()) return;
    setLoading(true);
    setLoadError(null);
    try {
      const res = await browserSetDirectory(dirPath.trim());
      if (!res.ok) {
        setLoadError("ディレクトリの設定に失敗しました");
        setLoading(false);
        return;
      }
      setDisplayName(res.display_name);
    } catch (e) {
      setLoadError(String(e));
      setLoading(false);
      return;
    }
    await loadImages(needsTagsLoaded(filterQuery));
  }, [dirPath, loadImages, filterQuery]);

  const handlePickDirectory = useCallback(async () => {
    setPicking(true);
    try {
      const res = await browserPickDirectory();
      if (!res.ok || !res.display_name) return;
      setDisplayName(res.display_name);
      setDirPath("");
      await loadImages(needsTagsLoaded(filterQuery));
    } catch (e) {
      setLoadError(String(e));
    } finally {
      setPicking(false);
    }
  }, [loadImages, filterQuery]);

  // Re-fetch when filter conditions change (debounced 300ms)
  useEffect(() => {
    if (images.length === 0) return; // not loaded yet
    if (filterDebounceRef.current) clearTimeout(filterDebounceRef.current);
    filterDebounceRef.current = setTimeout(() => {
      loadImages(needsTagsLoaded(filterQuery));
    }, 300);
    return () => {
      if (filterDebounceRef.current) clearTimeout(filterDebounceRef.current);
    };
  }, [filterQuery]); // eslint-disable-line react-hooks/exhaustive-deps

  // Compile filter function
  const filterFn = useMemo(
    () => compileFilter(filterQuery, filterCategoryMap),
    [filterQuery, filterCategoryMap]
  );

  const filteredImages = useMemo(() => {
    let result = images;
    // tagged/untagged mode filter
    if (filter !== "all") {
      result = result.filter((img) => {
        const has = taggedSet.has(img.rel_path) || img.has_tags;
        return filter === "tagged" ? has : !has;
      });
    }
    // advanced filter
    if (isFilterActive(filterQuery)) {
      result = result.filter(filterFn);
    }
    return result;
  }, [images, filter, taggedSet, filterQuery, filterFn]);

  const handleTagsSaved = useCallback((relPath: string, hasTags: boolean) => {
    setTaggedSet((prev) => {
      const next = new Set(prev);
      if (hasTags) next.add(relPath);
      else next.delete(relPath);
      return next;
    });
  }, []);

  const handleBulkTagsSaved = useCallback(
    (updates: Array<{ relPath: string; hasTags: boolean }>) => {
      setTaggedSet((prev) => {
        const next = new Set(prev);
        for (const { relPath, hasTags } of updates) {
          if (hasTags) next.add(relPath);
          else next.delete(relPath);
        }
        return next;
      });
    },
    []
  );

  // Multi-select handler
  const handleSelectMulti = useCallback(
    (rel_path: string, { ctrl, shift }: { ctrl: boolean; shift: boolean }) => {
      if (shift && rangeAnchorId) {
        const anchorIdx = filteredImages.findIndex((i) => i.rel_path === rangeAnchorId);
        const clickIdx = filteredImages.findIndex((i) => i.rel_path === rel_path);
        if (anchorIdx >= 0 && clickIdx >= 0) {
          const lo = Math.min(anchorIdx, clickIdx);
          const hi = Math.max(anchorIdx, clickIdx);
          const rangeIds = new Set(filteredImages.slice(lo, hi + 1).map((i) => i.rel_path));
          setSelectedIds(rangeIds);
          setPrimaryId(rel_path);
          return;
        }
      }

      if (ctrl) {
        setSelectedIds((prev) => {
          const next = new Set(prev);
          if (next.has(rel_path)) next.delete(rel_path);
          else next.add(rel_path);
          return next;
        });
        setPrimaryId(rel_path);
        setRangeAnchorId(rel_path);
      } else {
        setSelectedIds(new Set([rel_path]));
        setPrimaryId(rel_path);
        setRangeAnchorId(rel_path);
      }
    },
    [filteredImages, rangeAnchorId]
  );

  // Navigation (single-image mode)
  const primaryIdx = primaryId
    ? filteredImages.findIndex((i) => i.rel_path === primaryId)
    : -1;

  const handlePrev = useCallback(() => {
    if (primaryIdx <= 0) return;
    const newId = filteredImages[primaryIdx - 1].rel_path;
    setSelectedIds(new Set([newId]));
    setPrimaryId(newId);
    setRangeAnchorId(newId);
  }, [filteredImages, primaryIdx]);

  const handleNext = useCallback(() => {
    if (primaryIdx < 0 || primaryIdx >= filteredImages.length - 1) return;
    const newId = filteredImages[primaryIdx + 1].rel_path;
    setSelectedIds(new Set([newId]));
    setPrimaryId(newId);
    setRangeAnchorId(newId);
  }, [filteredImages, primaryIdx]);

  // Batch inference
  const handleBatchInfer = useCallback(() => {
    if (!modelLoaded || batchRunning) return;
    const rel_paths = filteredImages.map((img) => img.rel_path);
    if (rel_paths.length === 0) return;
    setBatchRunning(true);
    setBatchProgress({ done: 0, total: rel_paths.length, errors: 0 });

    const ctrl = browserBatchInfer(
      rel_paths,
      { overwrite: overwriteMode },
      (ev: BrowserBatchEvent) => {
        if (ev.type === "done") {
          setTaggedSet((prev) => {
            const next = new Set(prev);
            next.add(ev.rel_path);
            return next;
          });
          setBatchProgress((p) => (p ? { ...p, done: p.done + 1 } : p));
        } else if (ev.type === "skip") {
          setBatchProgress((p) => (p ? { ...p, done: p.done + 1 } : p));
        } else if (ev.type === "error") {
          setBatchProgress((p) =>
            p ? { ...p, done: p.done + 1, errors: p.errors + 1 } : p
          );
        } else if (ev.type === "complete") {
          setBatchRunning(false);
        }
      }
    );
    batchCtrlRef.current = ctrl;
  }, [modelLoaded, batchRunning, filteredImages, overwriteMode]);

  const handleBatchAbort = useCallback(() => {
    batchCtrlRef.current?.abort();
    setBatchRunning(false);
  }, []);

  // Prefetch prev/next images (size=1200) when primary selection changes
  useEffect(() => {
    if (primaryIdx < 0) return;
    const rps: string[] = [];
    if (primaryIdx > 0) rps.push(filteredImages[primaryIdx - 1].rel_path);
    if (primaryIdx < filteredImages.length - 1) rps.push(filteredImages[primaryIdx + 1].rel_path);
    for (const rp of rps) {
      const img = new window.Image();
      img.src = browserImageUrl(rp, 1200);
    }
  }, [primaryIdx, filteredImages]); // eslint-disable-line react-hooks/exhaustive-deps

  // Keyboard fallback: fires only when no image is selected
  useEffect(() => {
    if (primaryId !== null) return;
    const handler = (e: globalThis.KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === "PageDown" || e.key === "j") {
        e.preventDefault();
        if (filteredImages.length > 0) {
          const newId = filteredImages[0].rel_path;
          setSelectedIds(new Set([newId]));
          setPrimaryId(newId);
          setRangeAnchorId(newId);
        }
      } else if (e.key === "PageUp" || e.key === "k") {
        e.preventDefault();
        if (filteredImages.length > 0) {
          const newId = filteredImages[0].rel_path;
          setSelectedIds(new Set([newId]));
          setPrimaryId(newId);
          setRangeAnchorId(newId);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [primaryId, filteredImages]);

  const taggedCount = taggedSet.size;
  const totalCount = images.length;
  const filteredCount = filteredImages.length;

  const selectedImageObjects = useMemo(
    () => filteredImages.filter((img) => selectedIds.has(img.rel_path)),
    [filteredImages, selectedIds]
  );
  const primaryImage = primaryId
    ? filteredImages.find((i) => i.rel_path === primaryId) ?? null
    : null;

  const activeFilterCount = useMemo(() => {
    let n = 0;
    if (filterQuery.includeTags.length > 0) n++;
    if (filterQuery.excludeTags.length > 0) n++;
    if (filterQuery.tagCountMin !== null) n++;
    if (filterQuery.tagCountMax !== null) n++;
    if (filterQuery.missingCopyright) n++;
    if (filterQuery.missingCharacter) n++;
    return n;
  }, [filterQuery]);

  const resetFilter = useCallback(() => setFilterQuery(EMPTY_FILTER), []);

  return (
    <div
      ref={splitContainerRef}
      className="flex flex-col lg:flex-row h-full min-h-0 gap-0"
    >
      {/* Left: Grid panel */}
      <div
        className="flex flex-col min-h-0 border-r border-gray-700 flex-shrink-0"
        style={gridWidthPx !== null ? { width: gridWidthPx } : { width: "33.333%" }}
      >
        {/* Toolbar */}
        <div className="p-2 border-b border-gray-700 flex flex-col gap-2 flex-shrink-0">
          {/* Directory input row */}
          <div className="flex gap-1">
            <input
              type="text"
              value={dirPath}
              onChange={(e) => setDirPath(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLoad()}
              placeholder={displayName ? `現在: ${displayName}` : "ディレクトリパス..."}
              className="flex-1 px-2 py-1 text-sm bg-gray-800 border border-gray-600 rounded text-white placeholder-gray-500 min-w-0"
            />
            <button
              onClick={handlePickDirectory}
              disabled={picking || loading}
              title="フォルダを選択（OS標準ダイアログ）"
              className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:opacity-40 rounded flex-shrink-0"
            >
              {picking ? (
                <span className="text-xs">...</span>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V7z" />
                </svg>
              )}
            </button>
            <button
              onClick={handleLoad}
              disabled={loading || !dirPath.trim()}
              className="px-3 py-1 text-sm bg-blue-600 hover:bg-blue-500 disabled:opacity-40 rounded flex-shrink-0"
            >
              {loading ? "読込中..." : "読込"}
            </button>
          </div>

          {/* Options row */}
          <div className="flex items-center gap-3 flex-wrap">
            <label className="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
              <input
                type="checkbox"
                checked={recursive}
                onChange={(e) => setRecursive(e.target.checked)}
                className="accent-blue-500"
              />
              サブフォルダ含む
            </label>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value as FilterMode)}
              className="text-xs bg-gray-800 border border-gray-600 rounded px-1 py-0.5 text-white"
            >
              <option value="all">全て</option>
              <option value="tagged">タグあり</option>
              <option value="untagged">タグなし</option>
            </select>
            {/* Advanced filter toggle */}
            <button
              onClick={() => setFilterPanelOpen((v) => !v)}
              className={`text-xs px-2 py-0.5 rounded flex items-center gap-1 ${
                filterPanelOpen || activeFilterCount > 0
                  ? "bg-blue-700 text-blue-100"
                  : "bg-gray-700 hover:bg-gray-600 text-gray-300"
              }`}
            >
              フィルタ
              {activeFilterCount > 0 && (
                <span className="bg-blue-500 text-white rounded-full w-4 h-4 text-[10px] flex items-center justify-center">
                  {activeFilterCount}
                </span>
              )}
            </button>
            <span className="text-xs text-gray-500 ml-auto">
              {filteredCount !== totalCount
                ? `${filteredCount} / ${totalCount} 件`
                : `${totalCount} 件`}
              {totalCount > 0 && (
                <span className="ml-1 text-green-600">({taggedCount} タグ済)</span>
              )}
            </span>
          </div>

          {/* Advanced filter panel (collapsible) */}
          {filterPanelOpen && (
            <div className="flex flex-col gap-2 pt-1 border-t border-gray-700">
              <div>
                <label className="text-xs text-gray-400 block mb-0.5">
                  含むタグ (AND、*ワイルドカード、&lt;category&gt;)
                </label>
                <ChipInput
                  chips={filterQuery.includeTags}
                  onChange={(chips) => setFilterQuery((q) => ({ ...q, includeTags: chips }))}
                  placeholder="例: *hair, <character>"
                  chipColor="#14532d"
                />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-0.5">
                  除外タグ (OR)
                </label>
                <ChipInput
                  chips={filterQuery.excludeTags}
                  onChange={(chips) => setFilterQuery((q) => ({ ...q, excludeTags: chips }))}
                  placeholder="例: watermark"
                  chipColor="#450a0a"
                />
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-400">タグ数:</label>
                <input
                  type="number"
                  min={0}
                  value={filterQuery.tagCountMin ?? ""}
                  onChange={(e) =>
                    setFilterQuery((q) => ({
                      ...q,
                      tagCountMin: e.target.value === "" ? null : parseInt(e.target.value, 10),
                    }))
                  }
                  placeholder="min"
                  className="w-14 px-1 py-0.5 text-xs bg-gray-800 border border-gray-600 rounded text-white"
                />
                <span className="text-xs text-gray-500">〜</span>
                <input
                  type="number"
                  min={0}
                  value={filterQuery.tagCountMax ?? ""}
                  onChange={(e) =>
                    setFilterQuery((q) => ({
                      ...q,
                      tagCountMax: e.target.value === "" ? null : parseInt(e.target.value, 10),
                    }))
                  }
                  placeholder="max"
                  className="w-14 px-1 py-0.5 text-xs bg-gray-800 border border-gray-600 rounded text-white"
                />
              </div>
              <div className="flex items-center gap-3 flex-wrap">
                <label className="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filterQuery.missingCopyright}
                    onChange={(e) =>
                      setFilterQuery((q) => ({ ...q, missingCopyright: e.target.checked }))
                    }
                    className="accent-blue-500"
                  />
                  版権タグなし
                </label>
                <label className="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filterQuery.missingCharacter}
                    onChange={(e) =>
                      setFilterQuery((q) => ({ ...q, missingCharacter: e.target.checked }))
                    }
                    className="accent-blue-500"
                  />
                  キャラタグなし
                </label>
                {activeFilterCount > 0 && (
                  <button
                    onClick={resetFilter}
                    className="text-xs px-2 py-0.5 bg-gray-700 hover:bg-gray-600 rounded ml-auto"
                  >
                    リセット
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Batch inference controls */}
          {images.length > 0 && (
            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={batchRunning ? handleBatchAbort : handleBatchInfer}
                disabled={!modelLoaded && !batchRunning}
                className={`px-3 py-1 text-xs rounded flex-shrink-0 ${
                  batchRunning
                    ? "bg-red-700 hover:bg-red-600"
                    : "bg-green-700 hover:bg-green-600 disabled:opacity-40"
                }`}
              >
                {batchRunning ? "中止" : "バッチ推論"}
              </button>
              <label className="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={overwriteMode}
                  onChange={(e) => setOverwriteMode(e.target.checked)}
                  className="accent-blue-500"
                  disabled={batchRunning}
                />
                上書き
              </label>
              {batchProgress && (
                <div className="flex-1 min-w-0">
                  <div className="flex justify-between text-xs text-gray-400 mb-0.5">
                    <span>
                      {batchProgress.done}/{batchProgress.total}
                      {batchProgress.errors > 0 && (
                        <span className="text-red-400 ml-1">({batchProgress.errors} エラー)</span>
                      )}
                    </span>
                    <span>{Math.round((batchProgress.done / batchProgress.total) * 100)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-1.5">
                    <div
                      className="bg-blue-500 h-1.5 rounded-full transition-all"
                      style={{ width: `${(batchProgress.done / batchProgress.total) * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {loadError && <p className="text-red-400 text-xs">{loadError}</p>}
        </div>

        {/* Thumbnail grid */}
        <ThumbnailGrid
          images={filteredImages}
          selectedIds={selectedIds}
          primaryId={primaryId}
          onSelectMulti={handleSelectMulti}
          taggedSet={taggedSet}
        />
      </div>

      {/* Drag divider */}
      <div
        onMouseDown={onDividerMouseDown}
        className="hidden lg:flex w-1.5 flex-shrink-0 cursor-col-resize items-center justify-center bg-gray-700 hover:bg-blue-600 transition-colors group"
        title="ドラッグして幅を調整"
      >
        <div className="w-0.5 h-8 bg-gray-500 rounded group-hover:bg-blue-300 transition-colors" />
      </div>

      {/* Right: Tag editor */}
      <div className="flex-1 min-w-0 flex flex-col min-h-0 overflow-hidden">
        {selectedIds.size === 0 ? (
          <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
            {images.length === 0
              ? "ディレクトリを読み込んでください"
              : "画像を選択してください"}
          </div>
        ) : selectedIds.size === 1 && primaryImage ? (
          <TagEditorPanel
            key={primaryImage.rel_path}
            image={primaryImage}
            modelLoaded={modelLoaded}
            onPrev={handlePrev}
            onNext={handleNext}
            hasPrev={primaryIdx > 0}
            hasNext={primaryIdx < filteredImages.length - 1}
            onTagsSaved={handleTagsSaved}
          />
        ) : (
          <BulkTagEditorPanel
            selectedImages={selectedImageObjects}
            onTagsSaved={handleBulkTagsSaved}
            onDeselectAll={clearSelection}
          />
        )}
      </div>
    </div>
  );
}
