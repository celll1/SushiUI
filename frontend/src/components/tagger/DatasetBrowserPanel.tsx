"use client";

import { useState, useCallback, useRef, useMemo } from "react";
import {
  BrowserImageEntry,
  browserListImages,
  browserBatchInfer,
  BrowserBatchEvent,
} from "@/utils/api";
import ThumbnailGrid from "./ThumbnailGrid";
import TagEditorPanel from "./TagEditorPanel";

interface DatasetBrowserPanelProps {
  modelLoaded: boolean;
}

type FilterMode = "all" | "tagged" | "untagged";

export default function DatasetBrowserPanel({
  modelLoaded,
}: DatasetBrowserPanelProps) {
  const [dirPath, setDirPath] = useState("");
  const [recursive, setRecursive] = useState(false);
  const [images, setImages] = useState<BrowserImageEntry[]>([]);
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

  // Load directory
  const handleLoad = useCallback(async () => {
    if (!dirPath.trim()) return;
    setLoading(true);
    setLoadError(null);
    setSelectedIdx(null);
    try {
      const { images: imgs } = await browserListImages(dirPath.trim(), recursive);
      setImages(imgs);
      const tagged = new Set<string>();
      imgs.forEach((img) => {
        if (img.has_tags) tagged.add(img.path);
      });
      setTaggedSet(tagged);
    } catch (e) {
      setLoadError(String(e));
      setImages([]);
    } finally {
      setLoading(false);
    }
  }, [dirPath, recursive]);

  // Filtered image list
  const filteredImages = useMemo(() => {
    if (filter === "all") return images;
    return images.filter((img) => {
      const has = taggedSet.has(img.path) || img.has_tags;
      return filter === "tagged" ? has : !has;
    });
  }, [images, filter, taggedSet]);

  // Update taggedSet when tags are saved
  const handleTagsSaved = useCallback((path: string, hasTags: boolean) => {
    setTaggedSet((prev) => {
      const next = new Set(prev);
      if (hasTags) next.add(path);
      else next.delete(path);
      return next;
    });
  }, []);

  // Navigation
  const handleSelect = useCallback(
    (idx: number) => setSelectedIdx(idx),
    []
  );
  const handlePrev = useCallback(() => {
    setSelectedIdx((i) => (i !== null && i > 0 ? i - 1 : i));
  }, []);
  const handleNext = useCallback(() => {
    setSelectedIdx((i) =>
      i !== null && i < filteredImages.length - 1 ? i + 1 : i
    );
  }, [filteredImages.length]);

  // Batch inference
  const handleBatchInfer = useCallback(() => {
    if (!modelLoaded || batchRunning) return;
    const paths = filteredImages.map((img) => img.path);
    if (paths.length === 0) return;
    setBatchRunning(true);
    setBatchProgress({ done: 0, total: paths.length, errors: 0 });

    const ctrl = browserBatchInfer(
      paths,
      { overwrite: overwriteMode },
      (ev: BrowserBatchEvent) => {
        if (ev.type === "done") {
          setTaggedSet((prev) => {
            const next = new Set(prev);
            next.add(ev.path);
            return next;
          });
          setBatchProgress((p) =>
            p ? { ...p, done: p.done + 1 } : p
          );
        } else if (ev.type === "skip") {
          setBatchProgress((p) =>
            p ? { ...p, done: p.done + 1 } : p
          );
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

  const taggedCount = taggedSet.size;
  const totalCount = images.length;
  const filteredCount = filteredImages.length;

  return (
    <div className="flex flex-col lg:flex-row h-full min-h-0 gap-0">
      {/* Left: Grid panel */}
      <div className="lg:w-2/5 flex flex-col min-h-0 border-r border-gray-700">
        {/* Toolbar */}
        <div className="p-2 border-b border-gray-700 flex flex-col gap-2 flex-shrink-0">
          {/* Directory input */}
          <div className="flex gap-1">
            <input
              type="text"
              value={dirPath}
              onChange={(e) => setDirPath(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLoad()}
              placeholder="ディレクトリパス..."
              className="flex-1 px-2 py-1 text-sm bg-gray-800 border border-gray-600 rounded text-white placeholder-gray-500 min-w-0"
            />
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

          {loadError && (
            <p className="text-red-400 text-xs">{loadError}</p>
          )}
        </div>

        {/* Thumbnail grid */}
        <ThumbnailGrid
          images={filteredImages}
          selectedIdx={selectedIdx}
          onSelect={handleSelect}
          taggedSet={taggedSet}
        />
      </div>

      {/* Right: Tag editor */}
      <div className="lg:w-3/5 flex flex-col min-h-0 overflow-hidden">
        {selectedIdx !== null && filteredImages[selectedIdx] ? (
          <TagEditorPanel
            key={filteredImages[selectedIdx].path}
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
