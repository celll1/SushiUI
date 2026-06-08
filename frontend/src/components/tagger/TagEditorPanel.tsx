"use client";

import {
  useState,
  useEffect,
  useCallback,
  useRef,
  useMemo,
  KeyboardEvent,
} from "react";
import {
  BrowserImageEntry,
  browserImageUrl,
  browserGetTags,
  browserSaveTags,
  predictSigLIP2Tags,
} from "@/utils/api";
import InputWithTagSuggestions from "@/components/common/InputWithTagSuggestions";
import { useTagSuggestions } from "@/contexts/TagSuggestionsContext";
import { usePanelResize } from "@/hooks/usePanelResize";

interface TagEditorPanelProps {
  image: BrowserImageEntry;
  modelLoaded: boolean;
  onPrev: () => void;
  onNext: () => void;
  hasPrev: boolean;
  hasNext: boolean;
  onTagsSaved?: (relPath: string, hasTags: boolean) => void;
}

// Category display order and colors (matching image-tag-helper)
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

export default function TagEditorPanel({
  image,
  modelLoaded,
  onPrev,
  onNext,
  hasPrev,
  hasNext,
  onTagsSaved,
}: TagEditorPanelProps) {
  const [tags, setTags] = useState<string[]>([]);
  const [tagCategories, setTagCategories] = useState<Map<string, string>>(new Map());
  const [inputValue, setInputValue] = useState("");
  const [tagSearch, setTagSearch] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [inferring, setInferring] = useState(false);
  const [inferError, setInferError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Undo/Redo
  const [history, setHistory] = useState<string[][]>([[]]);
  const [historyIdx, setHistoryIdx] = useState(0);

  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tagSuggestionsCtx = useTagSuggestions();

  // Resizable split between image and tag editor
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const [tagPanelWidthPx, setTagPanelWidthPx] = useState<number | null>(null);
  // usePanelResize measures the FIRST child (image side), so we invert for the tag panel
  const { onMouseDown: onDividerMouseDown } = usePanelResize({
    containerRef: splitContainerRef,
    direction: "horizontal",
    minPx: 160,
    maxRatio: 0.75,
    onResize: (imagePx) => {
      // Compute container width then derive tag panel width
      const el = splitContainerRef.current;
      if (!el) return;
      const total = el.getBoundingClientRect().width;
      // 6px for the divider itself
      setTagPanelWidthPx(Math.max(160, total - imagePx - 6));
    },
  });

  const resolveCategories = useCallback(
    async (tagList: string[]) => {
      if (tagList.length === 0) return;
      try {
        const map = await tagSuggestionsCtx.getCategoriesForTags(tagList);
        setTagCategories(map);
      } catch {
        // category lookup failure is non-fatal
      }
    },
    [tagSuggestionsCtx]
  );

  // Load tags when image changes
  useEffect(() => {
    setInputValue("");
    setTagSearch("");
    setDirty(false);
    setInferError(null);
    setLoadError(null);
    setHistory([[]]);
    setHistoryIdx(0);
    setTagCategories(new Map());

    browserGetTags(image.rel_path)
      .then(({ tags: loaded }) => {
        setTags(loaded);
        setHistory([loaded]);
        setHistoryIdx(0);
        resolveCategories(loaded);
      })
      .catch((e) => setLoadError(String(e)));
  }, [image.rel_path]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-save with debounce
  useEffect(() => {
    if (!dirty) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(async () => {
      setSaving(true);
      try {
        await browserSaveTags(image.rel_path, tags);
        setDirty(false);
        onTagsSaved?.(image.rel_path, tags.length > 0);
      } finally {
        setSaving(false);
      }
    }, 500);
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, [tags, dirty, image.rel_path, onTagsSaved]);

  const pushHistory = useCallback(
    (newTags: string[]) => {
      setHistory((h) => {
        const next = h.slice(0, historyIdx + 1);
        next.push([...newTags]);
        return next.slice(-30);
      });
      setHistoryIdx((i) => Math.min(i + 1, 29));
    },
    [historyIdx]
  );

  const applyTags = useCallback(
    (newTags: string[], pushHist = true) => {
      setTags(newTags);
      setDirty(true);
      if (pushHist) pushHistory(newTags);
    },
    [pushHistory]
  );

  const addTag = useCallback(
    (tag: string, category?: string) => {
      const normalized = tag.trim().replace(/ /g, "_");
      if (!normalized) return;
      if (tags.includes(normalized)) {
        setInputValue("");
        return;
      }
      applyTags([...tags, normalized]);
      if (category) {
        setTagCategories((prev) => {
          const next = new Map(prev);
          next.set(normalized, category);
          return next;
        });
      }
      setInputValue("");
    },
    [tags, applyTags]
  );

  const removeTag = useCallback(
    (tag: string) => applyTags(tags.filter((t) => t !== tag)),
    [tags, applyTags]
  );

  const undo = useCallback(() => {
    if (historyIdx <= 0) return;
    const newIdx = historyIdx - 1;
    setHistoryIdx(newIdx);
    setTags(history[newIdx]);
    setDirty(true);
  }, [history, historyIdx]);

  const redo = useCallback(() => {
    if (historyIdx >= history.length - 1) return;
    const newIdx = historyIdx + 1;
    setHistoryIdx(newIdx);
    setTags(history[newIdx]);
    setDirty(true);
  }, [history, historyIdx]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.key === "z") { e.preventDefault(); undo(); }
        else if (e.key === "y") { e.preventDefault(); redo(); }
      } else if (!e.target || (e.target as HTMLElement).tagName !== "INPUT") {
        if (e.key === "ArrowLeft") onPrev();
        else if (e.key === "ArrowRight") onNext();
      }
    },
    [undo, redo, onPrev, onNext]
  );

  const handleInfer = useCallback(async () => {
    if (!modelLoaded || inferring) return;
    setInferring(true);
    setInferError(null);
    try {
      const imgRes = await fetch(browserImageUrl(image.rel_path, 0));
      const blob = await imgRes.blob();
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () =>
          resolve((reader.result as string).split(",")[1] ?? "");
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
      const result = await predictSigLIP2Tags(base64, 0.5);
      const inferred: string[] = result.tags ?? [];
      if (tags.length === 0 || confirm(`既存の ${tags.length} タグを上書きしますか？`)) {
        applyTags(inferred);
        resolveCategories(inferred);
      }
    } catch (e) {
      setInferError(String(e));
    } finally {
      setInferring(false);
    }
  }, [modelLoaded, inferring, image.rel_path, tags.length, applyTags, resolveCategories]);

  // Group tags by category, filtered by search, sorted within group
  const groupedTags = useMemo(() => {
    const search = tagSearch.toLowerCase();
    const filtered = search
      ? tags.filter((t) => t.toLowerCase().includes(search))
      : tags;

    const groups = new Map<CategoryName, string[]>(
      CATEGORY_ORDER.map((c) => [c, []])
    );
    for (const tag of filtered) {
      const raw = tagCategories.get(tag) ?? "Unknown";
      const cat: CategoryName = CATEGORY_ORDER.includes(raw as CategoryName)
        ? (raw as CategoryName)
        : "Unknown";
      groups.get(cat)!.push(tag);
    }
    for (const arr of groups.values()) arr.sort();
    return groups;
  }, [tags, tagCategories, tagSearch]);

  const totalGrouped = useMemo(
    () => [...groupedTags.values()].reduce((s, a) => s + a.length, 0),
    [groupedTags]
  );

  return (
    <div
      ref={splitContainerRef}
      className="flex flex-row h-full min-h-0 outline-none"
      tabIndex={-1}
      onKeyDown={handleKeyDown}
    >
      {/* Left: image + navigation */}
      <div className="flex flex-col min-h-0 flex-1 min-w-0 border-r border-gray-700">
        {/* Navigation */}
        <div className="flex items-center gap-2 px-2 py-1.5 flex-shrink-0 border-b border-gray-700">
          <button
            onClick={onPrev}
            disabled={!hasPrev}
            className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:opacity-40 rounded"
          >
            ← 前
          </button>
          <span className="text-xs text-gray-400 truncate flex-1 text-center">
            {image.rel_path}
          </span>
          <button
            onClick={onNext}
            disabled={!hasNext}
            className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:opacity-40 rounded"
          >
            次 →
          </button>
        </div>

        {/* Image */}
        <div className="flex-1 min-h-0 flex justify-center bg-gray-900 overflow-hidden">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={browserImageUrl(image.rel_path, 0)}
            alt={image.rel_path}
            className="object-contain w-full h-full"
            fetchPriority="high"
            decoding="async"
          />
        </div>

        {loadError && (
          <p className="text-red-400 text-xs px-2 pb-1 flex-shrink-0">
            読込エラー: {loadError}
          </p>
        )}
      </div>

      {/* Drag divider */}
      <div
        onMouseDown={onDividerMouseDown}
        className="w-1.5 flex-shrink-0 cursor-col-resize flex items-center justify-center bg-gray-700 hover:bg-blue-600 transition-colors group"
        title="ドラッグして幅を調整"
      >
        <div className="w-0.5 h-8 bg-gray-500 rounded group-hover:bg-blue-300 transition-colors" />
      </div>

      {/* Right: tag editor panel */}
      <div
        className="flex flex-col min-h-0 flex-shrink-0"
        style={
          tagPanelWidthPx !== null
            ? { width: tagPanelWidthPx }
            : { width: "20rem" /* 320px = w-80 */ }
        }
      >
        {/* Action bar */}
        <div className="flex items-center gap-1.5 px-2 py-1.5 flex-shrink-0 border-b border-gray-700 flex-wrap">
          <button
            onClick={handleInfer}
            disabled={!modelLoaded || inferring}
            className="px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-500 disabled:opacity-40 rounded"
          >
            {inferring ? "推論中..." : "推論"}
          </button>
          <button
            onClick={undo}
            disabled={historyIdx <= 0}
            className="px-2 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 disabled:opacity-40 rounded"
            title="Ctrl+Z"
          >
            元に戻す
          </button>
          <button
            onClick={redo}
            disabled={historyIdx >= history.length - 1}
            className="px-2 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 disabled:opacity-40 rounded"
            title="Ctrl+Y"
          >
            やり直し
          </button>
          <span className="ml-auto text-xs text-gray-500">
            {saving ? "保存中..." : dirty ? "未保存" : `${tags.length} タグ`}
          </span>
        </div>

        {inferError && (
          <p className="text-red-400 text-xs px-2 pt-1 flex-shrink-0">
            推論エラー: {inferError}
          </p>
        )}

        {/* Tag input */}
        <div className="px-2 pt-2 flex-shrink-0">
          <InputWithTagSuggestions
            value={inputValue}
            onChange={setInputValue}
            onTagAdd={(tag, category) => addTag(tag, category)}
            placeholder="タグを追加..."
            showSuggestionsAbove={false}
            className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-600 rounded text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Tag search */}
        <div className="px-2 pt-1.5 pb-2 flex-shrink-0">
          <input
            type="text"
            value={tagSearch}
            onChange={(e) => setTagSearch(e.target.value)}
            placeholder="タグを検索..."
            className="w-full px-2 py-1 text-xs bg-gray-800 border border-gray-600 rounded text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Category-grouped tags (scrollable) */}
        <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-2">
          {tags.length === 0 ? (
            <span className="text-gray-600 text-sm">タグなし</span>
          ) : totalGrouped === 0 ? (
            <span className="text-gray-600 text-sm">一致なし</span>
          ) : (
            CATEGORY_ORDER.map((cat) => {
              const catTags = groupedTags.get(cat) ?? [];
              if (catTags.length === 0) return null;
              const color = CATEGORY_COLORS[cat];
              return (
                <div key={cat} className="mb-3">
                  {/* Category header */}
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <span
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-xs font-semibold text-gray-300">
                      {cat}
                    </span>
                    <span className="text-xs text-gray-600">
                      ({catTags.length})
                    </span>
                  </div>
                  {/* Tag chips */}
                  <div className="flex flex-wrap gap-1.5 pl-4">
                    {catTags.map((tag) => (
                      <span
                        key={tag}
                        className="inline-flex items-center gap-1 px-2.5 py-1 rounded text-sm text-white"
                        style={{
                          backgroundColor: "#374151",
                          borderLeft: `3px solid ${color}`,
                        }}
                      >
                        {tag}
                        <button
                          onClick={() => removeTag(tag)}
                          className="text-gray-400 hover:text-white ml-0.5 text-base leading-none"
                          title="削除"
                        >
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
