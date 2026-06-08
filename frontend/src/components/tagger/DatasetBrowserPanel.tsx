"use client";

import { useState, useCallback, useRef, useMemo, useEffect } from "react";
import {
  BrowserImageEntry,
  browserSetDirectory,
  browserListImages,
  browserPickDirectory,
  browserBatchInfer,
  BrowserBatchEvent,
} from "@/utils/api";
import ThumbnailGrid from "./ThumbnailGrid";
import TagEditorPanel from "./TagEditorPanel";
import { usePanelResize } from "@/hooks/usePanelResize";

interface DatasetBrowserPanelProps {
  modelLoaded: boolean;
}

type FilterMode = "all" | "tagged" | "untagged";

export default function DatasetBrowserPanel({
  modelLoaded,
}: DatasetBrowserPanelProps) {
  // dirPath is shown in the input for user convenience, but absolute path is
  // never sent back from the server in API responses.
  const [dirPath, setDirPath] = useState("");
  // displayName is the folder basename returned by the server (not full path).
  const [displayName, setDisplayName] = useState<string | null>(null);
  const [recursive, setRecursive] = useState(false);
  const [images, setImages] = useState<BrowserImageEntry[]>([]);
  // taggedSet uses rel_path as keys (no absolute paths).
  const [taggedSet, setTaggedSet] = useState<Set<string>>(new Set());
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
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

  const loadImages = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    setSelectedIdx(null);
    try {
      const { images: imgs } = await browserListImages(recursive);
      setImages(imgs);
      const tagged = new Set<string>();
      imgs.forEach((img) => {
        if (img.has_tags) tagged.add(img.rel_path);
      });
      setTaggedSet(tagged);
    } catch (e) {
      setLoadError(String(e));
      setImages([]);
    } finally {
      setLoading(false);
    }
  }, [recursive]);

  // Set directory via typed path, then load
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
    await loadImages();
  }, [dirPath, loadImages]);

  // Open native folder picker, then load
  const handlePickDirectory = useCallback(async () => {
    setPicking(true);
    try {
      const res = await browserPickDirectory();
      if (!res.ok || !res.display_name) return;
      setDisplayName(res.display_name);
      setDirPath(""); // clear typed path — actual path is server-side only
      await loadImages();
    } catch (e) {
      setLoadError(String(e));
    } finally {
      setPicking(false);
    }
  }, [loadImages]);

  // Filtered image list (uses rel_path for taggedSet lookup)
  const filteredImages = useMemo(() => {
    if (filter === "all") return images;
    return images.filter((img) => {
      const has = taggedSet.has(img.rel_path) || img.has_tags;
      return filter === "tagged" ? has : !has;
    });
  }, [images, filter, taggedSet]);

  // Called by TagEditorPanel after auto-save
  const handleTagsSaved = useCallback((relPath: string, hasTags: boolean) => {
    setTaggedSet((prev) => {
      const next = new Set(prev);
      if (hasTags) next.add(relPath);
      else next.delete(relPath);
      return next;
    });
  }, []);

  // Navigation
  const handleSelect = useCallback((idx: number) => setSelectedIdx(idx), []);
  const handlePrev = useCallback(
    () => setSelectedIdx((i) => (i !== null && i > 0 ? i - 1 : i)),
    []
  );
  const handleNext = useCallback(
    () =>
      setSelectedIdx((i) =>
        i !== null && i < filteredImages.length - 1 ? i + 1 : i
      ),
    [filteredImages.length]
  );

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

  // Keyboard fallback: fires only when no image is selected (TagEditorPanel not mounted)
  useEffect(() => {
    if (selectedIdx !== null) return;
    const handler = (e: globalThis.KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === "PageDown" || e.key === "j") {
        e.preventDefault();
        setSelectedIdx((i) =>
          i === null
            ? filteredImages.length > 0 ? 0 : null
            : i < filteredImages.length - 1 ? i + 1 : i
        );
      } else if (e.key === "PageUp" || e.key === "k") {
        e.preventDefault();
        setSelectedIdx((i) =>
          i === null
            ? filteredImages.length > 0 ? 0 : null
            : i > 0 ? i - 1 : i
        );
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedIdx, filteredImages.length]);

  const taggedCount = taggedSet.size;
  const totalCount = images.length;
  const filteredCount = filteredImages.length;

  return (
    <div
      ref={splitContainerRef}
      className="flex flex-col lg:flex-row h-full min-h-0 gap-0"
    >
      {/* Left: Grid panel */}
      <div
        className="flex flex-col min-h-0 border-r border-gray-700 flex-shrink-0"
        style={
          gridWidthPx !== null
            ? { width: gridWidthPx }
            : { width: "33.333%" }
        }
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
              placeholder={
                displayName ? `現在: ${displayName}` : "ディレクトリパス..."
              }
              className="flex-1 px-2 py-1 text-sm bg-gray-800 border border-gray-600 rounded text-white placeholder-gray-500 min-w-0"
            />
            {/* Native folder picker */}
            <button
              onClick={handlePickDirectory}
              disabled={picking || loading}
              title="フォルダを選択（OS標準ダイアログ）"
              className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:opacity-40 rounded flex-shrink-0"
            >
              {picking ? (
                <span className="text-xs">...</span>
              ) : (
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V7z"
                  />
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
            <span className="text-xs text-gray-500 ml-auto">
              {filteredCount !== totalCount
                ? `${filteredCount} / ${totalCount} 件`
                : `${totalCount} 件`}
              {totalCount > 0 && (
                <span className="ml-1 text-green-600">
                  ({taggedCount} タグ済)
                </span>
              )}
            </span>
          </div>

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
                        <span className="text-red-400 ml-1">
                          ({batchProgress.errors} エラー)
                        </span>
                      )}
                    </span>
                    <span>
                      {Math.round(
                        (batchProgress.done / batchProgress.total) * 100
                      )}
                      %
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-1.5">
                    <div
                      className="bg-blue-500 h-1.5 rounded-full transition-all"
                      style={{
                        width: `${(batchProgress.done / batchProgress.total) * 100}%`,
                      }}
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
          selectedIdx={selectedIdx}
          onSelect={handleSelect}
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
        {selectedIdx !== null && filteredImages[selectedIdx] ? (
          <TagEditorPanel
            key={filteredImages[selectedIdx].rel_path}
            image={filteredImages[selectedIdx]}
            modelLoaded={modelLoaded}
            onPrev={handlePrev}
            onNext={handleNext}
            hasPrev={selectedIdx > 0}
            hasNext={selectedIdx < filteredImages.length - 1}
            onTagsSaved={handleTagsSaved}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
            {images.length === 0
              ? "ディレクトリを読み込んでください"
              : "画像を選択してください"}
          </div>
        )}
      </div>
    </div>
  );
}
