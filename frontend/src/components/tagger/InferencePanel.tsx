"use client";

import { useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { predictSigLIP2Tags, SigLIP2PredictResponse, SigLIP2TagResult } from "@/utils/api";
import { sendBase64ImageToImg2Img, sendBase64ImageToInpaint } from "@/utils/sendHelpers";
import TagResultsChart from "./TagResultsChart";

interface InferencePanelProps {
  modelLoaded: boolean;
}

export default function InferencePanel({ modelLoaded }: InferencePanelProps) {
  const router = useRouter();

  // Image state
  const [imageBase64, setImageBase64]   = useState<string | null>(null);  // full data-URL
  const [imageSrc,    setImageSrc]      = useState<string | null>(null);  // preview src
  const [dragging,    setDragging]      = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Inference state
  const [threshold,    setThreshold]    = useState(0.35);
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

  // ── Inference ─────────────────────────────────────────────────────────────

  const handlePredict = async () => {
    if (!imageBase64) return;
    setRunning(true);
    setError(null);
    try {
      // Strip data-URL prefix to get raw base64
      const b64 = imageBase64.replace(/^data:[^;]+;base64,/, "");
      const res  = await predictSigLIP2Tags(b64, threshold);
      setResult(res);
      // Auto-select all predicted tags
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

  const handleSelectAll = () => {
    if (!result) return;
    setSelectedTags(new Set([
      ...result.tags.map((t) => t.tag),
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
      {/* Top row: image drop + threshold */}
      <div className="flex gap-4">
        {/* Drop zone */}
        <div
          className={`w-56 h-48 shrink-0 border-2 border-dashed rounded-lg flex items-center justify-center cursor-pointer transition-colors ${
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
            <div className="text-center text-gray-500 text-xs px-4">
              <div className="text-3xl mb-2">📷</div>
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
        <div className="flex flex-col gap-3 flex-1">
          {/* Threshold */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">
              Threshold: <span className="text-white font-mono">{threshold.toFixed(2)}</span>
              <span className="text-gray-500 ml-1 text-[10px]">(Quality / Rating always shown)</span>
            </label>
            <input
              type="range"
              min={0.01}
              max={0.99}
              step={0.01}
              value={threshold}
              onChange={(e) => {
                setThreshold(parseFloat(e.target.value));
                // Re-run if we already have a result — cheap client-side re-filter handled in chart
              }}
              className="w-full"
            />
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
            <p className="text-xs text-yellow-500">Load a model first (left panel)</p>
          )}

          {error && (
            <p className="text-xs text-red-400 break-all">{error}</p>
          )}

          {/* Send-to-panel buttons */}
          {result && (
            <div className="space-y-2">
              <p className="text-xs text-gray-400">Send selected tags to prompt:</p>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => sendTagsTo("txt2img_params")}
                  disabled={selectedTags.size === 0}
                  className="px-3 py-1 text-xs rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white transition-colors"
                >
                  → Txt2Img
                </button>
                <button
                  onClick={() => sendTagsTo("img2img_params")}
                  disabled={selectedTags.size === 0}
                  className="px-3 py-1 text-xs rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white transition-colors"
                >
                  → Img2Img
                </button>
                <button
                  onClick={() => sendTagsTo("inpaint_params")}
                  disabled={selectedTags.size === 0}
                  className="px-3 py-1 text-xs rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white transition-colors"
                >
                  → Inpaint
                </button>
              </div>
              {imageSrc && (
                <>
                  <p className="text-xs text-gray-400">Send image to panel:</p>
                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={() => sendImageTo("img2img")}
                      className="px-3 py-1 text-xs rounded bg-indigo-700 hover:bg-indigo-600 text-white transition-colors"
                    >
                      → Img2Img (image)
                    </button>
                    <button
                      onClick={() => sendImageTo("inpaint")}
                      className="px-3 py-1 text-xs rounded bg-indigo-700 hover:bg-indigo-600 text-white transition-colors"
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

      {/* Results chart */}
      {result && (
        <div className="flex-1 overflow-y-auto">
          <TagResultsChart
            tags={result.tags}
            qualityTop={result.quality_top}
            ratingTop={result.rating_top}
            threshold={threshold}
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
