"use client";

import { useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { predictSigLIP2Tags, SigLIP2PredictResponse, SigLIP2TagResult } from "@/utils/api";
import { sendBase64ImageToImg2Img, sendBase64ImageToInpaint } from "@/utils/sendHelpers";
import TagResultsChart from "./TagResultsChart";

interface InferencePanelProps {
  modelLoaded: boolean;
}

// Categories that have their own threshold slider
const THRESHOLD_CATEGORIES = [
  "General", "Character", "Copyright", "Artist", "Meta",
];

interface CategoryThresholds {
  [category: string]: number;
}

export default function InferencePanel({ modelLoaded }: InferencePanelProps) {
  const router = useRouter();

  // Image state
  const [imageBase64, setImageBase64]   = useState<string | null>(null);
  const [imageSrc,    setImageSrc]      = useState<string | null>(null);
  const [dragging,    setDragging]      = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Threshold state
  const [globalThreshold,  setGlobalThreshold]  = useState(0.55);
  const [thresholdMode,    setThresholdMode]     = useState<"global" | "per-category">("global");
  const [categoryThresholds, setCategoryThresholds] = useState<CategoryThresholds>(() =>
    Object.fromEntries(THRESHOLD_CATEGORIES.map(c => [c, 0.55]))
  );

  // Inference state
  const [running,      setRunning]      = useState(false);
  const [error,        setError]        = useState<string | null>(null);
  const [result,       setResult]       = useState<SigLIP2PredictResponse | null>(null);
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());

  // ── Image loading ─────────────────────────────────────────────────────────

  const loadFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setImageBase64(dataUrl);
      setImageSrc(dataUrl);
      setResult(null);
      setSelectedTags(new Set());
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) loadFile(file);
  }, []);

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const item = Array.from(e.clipboardData.items).find((i) => i.type.startsWith("image/"));
    if (item) loadFile(item.getAsFile()!);
  }, []);

  // ── Threshold helpers ─────────────────────────────────────────────────────

  // Returns the effective threshold for a given category
  const thresholdFor = (category: string): number => {
    if (thresholdMode === "per-category") {
      return categoryThresholds[category] ?? globalThreshold;
    }
    return globalThreshold;
  };

  // Filter tags based on current threshold settings
  const filterTags = (allTags: SigLIP2TagResult[]): SigLIP2TagResult[] => {
    return allTags.filter(t => t.prob >= thresholdFor(t.category));
  };

  // Effective threshold passed to chart (for display in flat mode)
  const effectiveThreshold = globalThreshold;

  // ── Inference ─────────────────────────────────────────────────────────────

  const handlePredict = async () => {
    if (!imageBase64) return;
    setRunning(true);
    setError(null);
    try {
      const b64 = imageBase64.replace(/^data:[^;]+;base64,/, "");
      // Use global threshold as baseline; client-side per-category filter applied to result
      const res  = await predictSigLIP2Tags(b64, Math.min(...Object.values(
        thresholdMode === "per-category" ? categoryThresholds : { g: globalThreshold }
      )));
      setResult(res);
      const allTags = new Set<string>([
        ...res.tags.map((t) => t.tag),
        ...(res.quality_top ? [res.quality_top.tag] : []),
        ...(res.rating_top  ? [res.rating_top.tag]  : []),
      ]);
      setSelectedTags(allTags);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? "Prediction failed");
    } finally {
      setRunning(false);
    }
  };

  // ── Tag selection ─────────────────────────────────────────────────────────

  const handleTagToggle = (tag: string) => {
    setSelectedTags((prev) => {
      const next = new Set(prev);
      if (next.has(tag)) next.delete(tag); else next.add(tag);
      return next;
    });
  };

  const visibleTags = result ? filterTags(result.tags) : [];

  const handleSelectAll = () => {
    if (!result) return;
    setSelectedTags(new Set([
      ...visibleTags.map((t) => t.tag),
      ...(result.quality_top ? [result.quality_top.tag] : []),
      ...(result.rating_top  ? [result.rating_top.tag]  : []),
    ]));
  };

  const handleDeselectAll = () => setSelectedTags(new Set());

  // ── Send to generation panels ─────────────────────────────────────────────

  const tagString = () => Array.from(selectedTags).join(", ");

  const sendTagsTo = (storageKey: string) => {
    const tags = tagString();
    if (!tags) return;
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
    saved.prompt = saved.prompt ? saved.prompt + ", " + tags : tags;
    localStorage.setItem(storageKey, JSON.stringify(saved));
    router.push("/generate");
  };

  const sendImageTo = async (target: "img2img" | "inpaint") => {
    if (!imageBase64) return;
    try {
      if (target === "img2img") {
        await sendBase64ImageToImg2Img(imageBase64);
      } else {
        await sendBase64ImageToInpaint(imageBase64);
      }
      router.push(`/generate?tab=${target}`);
    } catch (e) {
      console.error("[TaggerInference] Failed to send image:", e);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-4 p-4 h-full">

      {/* ── Top row: image + controls ── */}
      <div className="flex gap-4">

        {/* Drop zone — larger */}
        <div
          className={`w-80 h-72 shrink-0 border-2 border-dashed rounded-lg flex items-center justify-center cursor-pointer transition-colors ${
            dragging
              ? "border-blue-400 bg-blue-900/20"
              : "border-gray-600 hover:border-gray-500"
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onPaste={handlePaste}
          onClick={() => fileInputRef.current?.click()}
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
        >
          {imageSrc ? (
            <img
              src={imageSrc}
              alt="input"
              className="max-w-full max-h-full object-contain rounded"
            />
          ) : (
            <div className="text-center text-gray-500 text-sm px-4">
              <div className="text-4xl mb-2">📷</div>
              Drop image here<br />or click to browse<br />or paste (Ctrl+V)
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) loadFile(f); }}
          />
        </div>

        {/* Controls */}
        <div className="flex flex-col gap-3 flex-1 min-w-0">

          {/* Threshold mode toggle */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm text-gray-400">Threshold</span>
              <div className="flex rounded overflow-hidden border border-gray-600 text-xs ml-auto">
                <button
                  onClick={() => setThresholdMode("global")}
                  className={`px-2 py-0.5 ${thresholdMode === "global" ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
                >
                  Global
                </button>
                <button
                  onClick={() => setThresholdMode("per-category")}
                  className={`px-2 py-0.5 ${thresholdMode === "per-category" ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
                >
                  Per-category
                </button>
              </div>
            </div>

            {thresholdMode === "global" ? (
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Threshold: <span className="text-white font-mono">{globalThreshold.toFixed(2)}</span>
                  <span className="text-gray-500 ml-2 text-[11px]">(Quality / Rating always shown)</span>
                </label>
                <input
                  type="range" min={0.01} max={0.99} step={0.01}
                  value={globalThreshold}
                  onChange={(e) => setGlobalThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            ) : (
              <div className="space-y-1.5">
                <p className="text-xs text-gray-500">Quality / Rating always shown. Other categories:</p>
                {THRESHOLD_CATEGORIES.map(cat => (
                  <div key={cat} className="flex items-center gap-2">
                    <span className={`text-xs w-20 shrink-0 ${
                      { General: "text-green-400", Character: "text-blue-400", Copyright: "text-purple-400",
                        Artist: "text-pink-400", Meta: "text-gray-400" }[cat] ?? "text-gray-400"
                    }`}>{cat}</span>
                    <input
                      type="range" min={0.01} max={0.99} step={0.01}
                      value={categoryThresholds[cat] ?? 0.5}
                      onChange={(e) => setCategoryThresholds(prev => ({ ...prev, [cat]: parseFloat(e.target.value) }))}
                      className="flex-1"
                    />
                    <span className="text-xs text-gray-400 font-mono w-9 text-right shrink-0">
                      {(categoryThresholds[cat] ?? 0.5).toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Predict button */}
          <button
            onClick={handlePredict}
            disabled={!imageBase64 || !modelLoaded || running}
            className="py-2 rounded text-sm font-medium bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
          >
            {running ? "Running…" : "Predict Tags"}
          </button>

          {!modelLoaded && (
            <p className="text-sm text-yellow-500">Load a model first (left panel)</p>
          )}

          {error && (
            <p className="text-sm text-red-400 break-all">{error}</p>
          )}

          {/* Send-to-panel buttons */}
          {result && (
            <div className="space-y-2">
              <p className="text-sm text-gray-400">Send selected tags to prompt:</p>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => sendTagsTo("txt2img_params")}
                  disabled={selectedTags.size === 0}
                  className="px-3 py-1.5 text-sm rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white transition-colors"
                >
                  → Txt2Img
                </button>
                <button
                  onClick={() => sendTagsTo("img2img_params")}
                  disabled={selectedTags.size === 0}
                  className="px-3 py-1.5 text-sm rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white transition-colors"
                >
                  → Img2Img
                </button>
                <button
                  onClick={() => sendTagsTo("inpaint_params")}
                  disabled={selectedTags.size === 0}
                  className="px-3 py-1.5 text-sm rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white transition-colors"
                >
                  → Inpaint
                </button>
              </div>
              {imageSrc && (
                <>
                  <p className="text-sm text-gray-400">Send image to panel:</p>
                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={() => sendImageTo("img2img")}
                      className="px-3 py-1.5 text-sm rounded bg-indigo-700 hover:bg-indigo-600 text-white transition-colors"
                    >
                      → Img2Img (image)
                    </button>
                    <button
                      onClick={() => sendImageTo("inpaint")}
                      className="px-3 py-1.5 text-sm rounded bg-indigo-700 hover:bg-indigo-600 text-white transition-colors"
                    >
                      → Inpaint (image)
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Results chart ── */}
      {result && (
        <div className="flex-1 overflow-y-auto">
          <TagResultsChart
            tags={visibleTags}
            qualityTop={result.quality_top}
            ratingTop={result.rating_top}
            threshold={effectiveThreshold}
            selectedTags={selectedTags}
            onTagToggle={handleTagToggle}
            onSelectAll={handleSelectAll}
            onDeselectAll={handleDeselectAll}
          />
        </div>
      )}
    </div>
  );
}
