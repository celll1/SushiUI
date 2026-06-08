"use client";

import {
  useState,
  useEffect,
  useCallback,
  useRef,
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

interface TagEditorPanelProps {
  image: BrowserImageEntry;
  modelLoaded: boolean;
  onPrev: () => void;
  onNext: () => void;
  hasPrev: boolean;
  hasNext: boolean;
  onTagsSaved?: (relPath: string, hasTags: boolean) => void;
}

const CATEGORY_COLORS: Record<string, string> = {
  General: "#4ade80",
  Character: "#60a5fa",
  Copyright: "#c084fc",
  Meta: "#9ca3af",
  Quality: "#facc15",
  Rating: "#fb923c",
  Artist: "#f472b6",
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
  const [tagCategories, setTagCategories] = useState<Record<string, string>>({});
  const [inputValue, setInputValue] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [inferring, setInferring] = useState(false);
  const [inferError, setInferError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Undo/Redo
  const [history, setHistory] = useState<string[][]>([[]]);
  const [historyIdx, setHistoryIdx] = useState(0);

  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load tags when image changes
  useEffect(() => {
    setInputValue("");
    setDirty(false);
    setInferError(null);
    setLoadError(null);
    setHistory([[]]);
    setHistoryIdx(0);

    browserGetTags(image.rel_path)
      .then(({ tags: loaded }) => {
        setTags(loaded);
        setHistory([loaded]);
        setHistoryIdx(0);
      })
      .catch((e) => setLoadError(String(e)));
  }, [image.rel_path]);

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
      const newTags = [...tags, normalized];
      applyTags(newTags);
      if (category) {
        setTagCategories((c) => ({ ...c, [normalized]: category }));
      }
      setInputValue("");
    },
    [tags, applyTags]
  );

  const removeTag = useCallback(
    (tag: string) => {
      applyTags(tags.filter((t) => t !== tag));
    },
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

  // Keyboard shortcuts
  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.key === "z") {
          e.preventDefault();
          undo();
        } else if (e.key === "y") {
          e.preventDefault();
          redo();
        }
      } else if (!e.target || (e.target as HTMLElement).tagName !== "INPUT") {
        if (e.key === "ArrowLeft") onPrev();
        else if (e.key === "ArrowRight") onNext();
      }
    },
    [undo, redo, onPrev, onNext]
  );

  // Single-image inference
  const handleInfer = useCallback(async () => {
    if (!modelLoaded || inferring) return;
    setInferring(true);
    setInferError(null);
    try {
      // Fetch image as base64
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
      const inferred = result.tags ?? [];
      if (
        tags.length === 0 ||
        confirm(`既存の ${tags.length} タグを上書きしますか？`)
      ) {
        applyTags(inferred);
      }
    } catch (e) {
      setInferError(String(e));
    } finally {
      setInferring(false);
    }
  }, [modelLoaded, inferring, image.rel_path, tags.length, applyTags]);

  return (
    <div
      className="flex flex-col h-full min-h-0 p-3 gap-3 outline-none"
      tabIndex={-1}
      onKeyDown={handleKeyDown}
    >
      {/* Navigation */}
      <div className="flex items-center gap-2 flex-shrink-0">
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

      {/* Image preview — flex-1 so it takes all available vertical space */}
      <div className="flex-1 min-h-0 flex justify-center bg-gray-900 rounded overflow-hidden">
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
        <p className="text-red-400 text-xs">読込エラー: {loadError}</p>
      )}

      {/* Tag chips */}
      <div className="flex flex-wrap gap-1 min-h-8 max-h-32 overflow-y-auto flex-shrink-0">
        {tags.map((tag) => {
          const cat = tagCategories[tag] ?? "Unknown";
          const color = CATEGORY_COLORS[cat] ?? CATEGORY_COLORS.Unknown;
          return (
            <span
              key={tag}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs text-white"
              style={{ backgroundColor: "#374151", borderLeft: `3px solid ${color}` }}
            >
              {tag}
              <button
                onClick={() => removeTag(tag)}
                className="text-gray-400 hover:text-white leading-none"
              >
                ×
              </button>
            </span>
          );
        })}
        {tags.length === 0 && (
          <span className="text-gray-600 text-xs">タグなし</span>
        )}
      </div>

      {/* Tag input */}
      <div className="flex-shrink-0">
        <InputWithTagSuggestions
          value={inputValue}
          onChange={setInputValue}
          onTagAdd={(tag, category) => addTag(tag, category)}
          placeholder="タグを入力..."
          showSuggestionsAbove={true}
          className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-600 rounded text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
        />
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 flex-shrink-0 flex-wrap">
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
        <p className="text-red-400 text-xs">推論エラー: {inferError}</p>
      )}
    </div>
  );
}
