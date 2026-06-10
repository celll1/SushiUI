"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  predictSigLIP2Tags,
  getSigLIP2Status,
  fetchTagMetrics,
  getCalibrationSettings,
  setCalibrationSettings,
  buildSigLIP2OodReference,
  SigLIP2PredictResponse,
  SigLIP2TagResult,
  SigLIP2ContextMethod,
  SigLIP2CalibrationSettings,
  TagMetricsData,
} from "@/utils/api";
import { sendBase64ImageToImg2Img, sendBase64ImageToInpaint } from "@/utils/sendHelpers";
import InputWithTagSuggestions from "@/components/common/InputWithTagSuggestions";
import TagResultsChart from "./TagResultsChart";
import TagMetricsAnalysis from "./TagMetricsAnalysis";

interface InferencePanelProps {
  modelLoaded: boolean;
}

const THRESHOLD_CATEGORIES = [
  "General", "Character", "Copyright", "Artist", "Meta",
];

interface CategoryThresholds {
  [category: string]: number;
}

// ─── OOD score badge ─────────────────────────────────────────────────────────

function OodBadge({ distance, p50, p95 }: { distance: number; p50: number | null; p95: number | null }) {
  let label = `${distance.toFixed(2)}`;
  let cls = "bg-gray-700 text-gray-300";
  if (p50 != null && p95 != null) {
    if (distance <= p50) {
      cls = "bg-green-900 text-green-300 border border-green-700";
      label += " In-dist";
    } else if (distance <= p95) {
      cls = "bg-yellow-900 text-yellow-300 border border-yellow-700";
      label += " Borderline";
    } else {
      cls = "bg-orange-900 text-orange-300 border border-orange-700";
      label += " OOD ⚠";
    }
    label += ` (p50=${p50.toFixed(1)}, p95=${p95.toFixed(1)})`;
  }
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono ${cls}`}>
      OOD {label}
    </span>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function InferencePanel({ modelLoaded }: InferencePanelProps) {
  const router = useRouter();

  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageSrc,    setImageSrc]    = useState<string | null>(null);
  const [dragging,    setDragging]    = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [globalThreshold,     setGlobalThreshold]     = useState(0.55);
  const [thresholdMode,       setThresholdMode]       = useState<"global" | "per-category">("global");
  const [categoryThresholds,  setCategoryThresholds]  = useState<CategoryThresholds>(() =>
    Object.fromEntries(THRESHOLD_CATEGORIES.map(c => [c, 0.55]))
  );

  const [running,      setRunning]      = useState(false);
  const [error,        setError]        = useState<string | null>(null);
  const [result,       setResult]       = useState<SigLIP2PredictResponse | null>(null);
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());

  const [useTrainingModel, setUseTrainingModel] = useState(false);

  const [activeTab,     setActiveTab]     = useState<"inference" | "analysis">("inference");
  const [hasTagMetrics, setHasTagMetrics] = useState(false);
  const [tagMetrics,    setTagMetrics]    = useState<TagMetricsData | null>(null);
  const [metricsLoading,setMetricsLoading] = useState(false);
  const [metricsError,  setMetricsError]  = useState<string | null>(null);

  const [hasOodReference, setHasOodReference] = useState(false);
  const [oodP50,          setOodP50]          = useState<number | null>(null);
  const [oodP95,          setOodP95]          = useState<number | null>(null);
  const [useOodDetection, setUseOodDetection] = useState(false);
  const [oodBuilding,     setOodBuilding]     = useState(false);
  const [oodBuildError,   setOodBuildError]   = useState<string | null>(null);
  const [oodBuildResult,  setOodBuildResult]  = useState<{ p50: number; p95: number; n_images: number } | null>(null);

  useEffect(() => {
    getSigLIP2Status()
      .then((s) => {
        const hasMet = s.has_tag_metrics ?? false;
        setHasTagMetrics(hasMet);
        if (hasMet) setInferMode("best_thr");
        if (s.calib_method) setCalibMethod(s.calib_method as "jeffreys" | "beta_bb");
        if (typeof s.calib_eps === "number") setCalibEps(s.calib_eps);
        if (typeof s.calib_prior_strength === "number") setCalibPriorStrength(s.calib_prior_strength);
        setHasOodReference(s.has_ood_reference ?? false);
        setOodP50(typeof s.ood_p50 === "number" ? s.ood_p50 : null);
        setOodP95(typeof s.ood_p95 === "number" ? s.ood_p95 : null);
      })
      .catch(() => {});
    if (!modelLoaded) {
      setActiveTab("inference");
      setTagMetrics(null);
    }
  }, [modelLoaded]);

  const handleApplyCalibration = async () => {
    setCalibApplying(true);
    try {
      await setCalibrationSettings({ method: calibMethod, eps: calibEps, prior_strength: calibPriorStrength });
    } catch { /* silently ignore */ }
    finally { setCalibApplying(false); }
  };

  const handleTabChange = async (tab: "inference" | "analysis") => {
    setActiveTab(tab);
    if (tab === "analysis" && tagMetrics === null && !metricsLoading) {
      setMetricsLoading(true);
      setMetricsError(null);
      try {
        const data = await fetchTagMetrics();
        setTagMetrics(data);
      } catch (e: any) {
        setMetricsError(e?.response?.data?.detail ?? "Failed to load tag metrics");
      } finally {
        setMetricsLoading(false);
      }
    }
  };

  const [inferMode,         setInferMode]         = useState<"best_thr" | "fixed">("fixed");
  const [displayCalibrated, setDisplayCalibrated] = useState(false);
  const [minBestThr,        setMinBestThr]        = useState(0.30);
  const [minBestF1,         setMinBestF1]         = useState(0.05);
  const [calibMethod,       setCalibMethod]       = useState<"jeffreys" | "beta_bb">("jeffreys");
  const [calibEps,          setCalibEps]          = useState(0.5);
  const [calibPriorStrength,setCalibPriorStrength]= useState(10.0);
  const [calibApplying,     setCalibApplying]     = useState(false);
  const [useCalibration,    setUseCalibration]    = useState(false);

  const [contextMethod, setContextMethod] = useState<SigLIP2ContextMethod>("none");
  const [contextLambda, setContextLambda] = useState(0.5);
  const [knownTagsPos,  setKnownTagsPos]  = useState<string[]>([]);
  const [knownTagsNeg,  setKnownTagsNeg]  = useState<string[]>([]);
  const [posTagInput,   setPosTagInput]   = useState("");
  const [negTagInput,   setNegTagInput]   = useState("");
  const [showNegInput,  setShowNegInput]  = useState(false);

  // ── Image loading ───────────────────────────────────────────────────────────

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

  // ── Threshold helpers ───────────────────────────────────────────────────────

  const thresholdFor = (category: string): number =>
    thresholdMode === "per-category" ? (categoryThresholds[category] ?? globalThreshold) : globalThreshold;

  const filterTags = (allTags: SigLIP2TagResult[]): SigLIP2TagResult[] => {
    if (result?.used_best_thr) return allTags;
    return allTags.filter(t => (t.raw_prob ?? t.prob) >= thresholdFor(t.category));
  };

  // ── Inference ───────────────────────────────────────────────────────────────

  const handlePredict = async () => {
    if (!imageBase64) return;
    setRunning(true);
    setError(null);
    try {
      const b64 = imageBase64.replace(/^data:[^;]+;base64,/, "");
      const baselineThr = Math.min(...Object.values(
        thresholdMode === "per-category" ? categoryThresholds : { g: globalThreshold }
      ));
      const res = await predictSigLIP2Tags(b64, baselineThr, {
        known_tags_pos: knownTagsPos,
        known_tags_neg: knownTagsNeg,
        context_method: contextMethod,
        context_lambda: contextLambda,
        use_training_model: useTrainingModel,
        use_per_tag_threshold: inferMode === "best_thr" && hasTagMetrics,
        min_best_thr: minBestThr,
        min_best_f1: minBestF1,
        display_calibration: displayCalibrated && hasTagMetrics,
        use_calibration: inferMode === "fixed" && useCalibration && hasTagMetrics,
        use_ood_detection: useOodDetection && hasOodReference && inferMode === "best_thr",
      });
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

  // ── Known tag handlers ──────────────────────────────────────────────────────

  const addKnownTag = (which: "pos" | "neg") => (tag: string) => {
    const t = tag.trim();
    if (!t) return;
    if (which === "pos") {
      setKnownTagsPos((cur) => (cur.includes(t) ? cur : [...cur, t]));
      if (contextMethod === "none") setContextMethod("head_sim");
    } else {
      setKnownTagsNeg((cur) => (cur.includes(t) ? cur : [...cur, t]));
      if (contextMethod === "none") setContextMethod("head_sim");
    }
  };

  const removeKnownTag = (which: "pos" | "neg", tag: string) => {
    if (which === "pos") setKnownTagsPos((cur) => cur.filter((t) => t !== tag));
    else setKnownTagsNeg((cur) => cur.filter((t) => t !== tag));
  };

  // ── Tag selection ───────────────────────────────────────────────────────────

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

  // ── Send to panels ──────────────────────────────────────────────────────────

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
      if (target === "img2img") await sendBase64ImageToImg2Img(imageBase64);
      else await sendBase64ImageToInpaint(imageBase64);
      router.push(`/generate?tab=${target}`);
    } catch (e) {
      console.error("[TaggerInference] Failed to send image:", e);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-3 p-4 h-full">

      {/* ── Tab bar ── */}
      {modelLoaded && (
        <div className="flex gap-1 border-b border-gray-700 flex-shrink-0 -mb-1">
          <button
            onClick={() => handleTabChange("inference")}
            className={`px-3 py-1.5 text-sm transition-colors ${
              activeTab === "inference" ? "text-white border-b-2 border-blue-500 -mb-px" : "text-gray-400 hover:text-gray-200"
            }`}
          >
            推論
          </button>
          {hasTagMetrics && (
            <button
              onClick={() => handleTabChange("analysis")}
              className={`px-3 py-1.5 text-sm transition-colors ${
                activeTab === "analysis" ? "text-white border-b-2 border-blue-500 -mb-px" : "text-gray-400 hover:text-gray-200"
              }`}
            >
              分析
            </button>
          )}
        </div>
      )}

      {/* ── Analysis tab ── */}
      {activeTab === "analysis" && (
        <TagMetricsAnalysis data={tagMetrics} loading={metricsLoading} error={metricsError} />
      )}

      {/* ── Inference tab ── */}
      <div className={`flex flex-col gap-3 flex-1 min-h-0 ${activeTab !== "inference" ? "hidden" : ""}`}>

        {/* ── Top section: image + 2-column options ── */}
        <div className="flex gap-4 shrink-0">

          {/* Drop zone */}
          <div
            className={`w-64 h-56 shrink-0 border-2 border-dashed rounded-lg flex items-center justify-center cursor-pointer transition-colors ${
              dragging ? "border-blue-400 bg-blue-900/20" : "border-gray-600 hover:border-gray-500"
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
              <img src={imageSrc} alt="input" className="max-w-full max-h-full object-contain rounded" />
            ) : (
              <div className="text-center text-gray-500 text-sm px-4">
                <div className="text-4xl mb-2">📷</div>
                Drop image here<br />or click to browse<br />or paste (Ctrl+V)
              </div>
            )}
            <input ref={fileInputRef} type="file" accept="image/*" className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) loadFile(f); }} />
          </div>

          {/* Options: 2-column grid */}
          <div className="flex-1 flex flex-col gap-2 min-w-0">
            <div className="grid grid-cols-2 gap-3">

              {/* ── Column 1: Threshold + Inference mode ── */}
              <div className="flex flex-col gap-2">

                {/* Threshold */}
                <div className="border border-gray-700 rounded p-2 space-y-1.5">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">Threshold</span>
                    <div className="flex rounded overflow-hidden border border-gray-600 text-xs ml-auto">
                      <button onClick={() => setThresholdMode("global")}
                        className={`px-2 py-0.5 ${thresholdMode === "global" ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}>
                        Global
                      </button>
                      <button onClick={() => setThresholdMode("per-category")}
                        className={`px-2 py-0.5 ${thresholdMode === "per-category" ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}>
                        Per-cat
                      </button>
                    </div>
                  </div>

                  {thresholdMode === "global" ? (
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">
                        <span className="font-mono text-white">{globalThreshold.toFixed(2)}</span>
                        <span className="text-gray-600 ml-1 text-[10px]">(Quality/Rating always shown)</span>
                      </label>
                      <input type="range" min={0.01} max={0.99} step={0.01}
                        value={globalThreshold}
                        onChange={(e) => setGlobalThreshold(parseFloat(e.target.value))}
                        className="w-full" />
                    </div>
                  ) : (
                    <div className="space-y-1">
                      {THRESHOLD_CATEGORIES.map(cat => (
                        <div key={cat} className="flex items-center gap-1.5">
                          <span className={`text-[10px] w-14 shrink-0 ${
                            { General: "text-green-400", Character: "text-blue-400", Copyright: "text-purple-400",
                              Artist: "text-pink-400", Meta: "text-gray-400" }[cat] ?? "text-gray-400"
                          }`}>{cat}</span>
                          <input type="range" min={0.01} max={0.99} step={0.01}
                            value={categoryThresholds[cat] ?? 0.5}
                            onChange={(e) => setCategoryThresholds(prev => ({ ...prev, [cat]: parseFloat(e.target.value) }))}
                            className="flex-1" />
                          <span className="text-[10px] text-gray-400 font-mono w-7 text-right shrink-0">
                            {(categoryThresholds[cat] ?? 0.5).toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Inference mode */}
                {hasTagMetrics && (
                  <div className="border border-gray-700 rounded p-2 space-y-1.5">
                    <div className="flex items-center gap-1">
                      <span className="text-xs text-gray-400">Inference</span>
                      <div className="flex gap-1 ml-auto">
                        {(["best_thr", "fixed"] as const).map((m) => (
                          <button key={m} onClick={() => setInferMode(m)}
                            className={`px-2 py-0.5 text-xs rounded ${inferMode === m ? "bg-blue-700 text-white" : "text-gray-400 hover:bg-gray-700"}`}>
                            {m === "best_thr" ? "Best-thr" : "固定"}
                          </button>
                        ))}
                      </div>
                    </div>

                    {inferMode === "best_thr" && (
                      <>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-gray-500 w-16 shrink-0">min best_thr</span>
                          <input type="range" min={0.10} max={0.50} step={0.01}
                            value={minBestThr}
                            onChange={(e) => setMinBestThr(parseFloat(e.target.value))}
                            className="flex-1 accent-blue-500" />
                          <span className="text-[10px] text-gray-400 w-7 text-right">{minBestThr.toFixed(2)}</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-gray-500 w-16 shrink-0">min best_F1</span>
                          <input type="range" min={0.00} max={0.30} step={0.01}
                            value={minBestF1}
                            onChange={(e) => setMinBestF1(parseFloat(e.target.value))}
                            className="flex-1 accent-blue-500" />
                          <span className="text-[10px] text-gray-400 w-7 text-right">{minBestF1.toFixed(2)}</span>
                        </div>
                      </>
                    )}

                    {inferMode === "fixed" && (
                      <label className="flex items-center gap-1.5 cursor-pointer select-none">
                        <input type="checkbox" checked={useCalibration}
                          onChange={(e) => setUseCalibration(e.target.checked)}
                          className="w-3 h-3 rounded accent-blue-500" />
                        <span className="text-xs text-gray-400">Calibration (legacy)</span>
                      </label>
                    )}
                    {inferMode === "fixed" && useCalibration && (
                      <>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-gray-500 w-12">Method</span>
                          <div className="flex gap-1">
                            {(["jeffreys", "beta_bb"] as const).map((m) => (
                              <button key={m} onClick={() => setCalibMethod(m)}
                                className={`px-1.5 py-0.5 text-xs rounded ${calibMethod === m ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}>
                                {m === "jeffreys" ? "Jeffreys" : "Beta-BB"}
                              </button>
                            ))}
                          </div>
                        </div>
                        {calibMethod === "jeffreys" && (
                          <div className="flex items-center gap-1.5">
                            <span className="text-[10px] text-gray-500 w-12">ε</span>
                            <input type="range" min={0.1} max={2.0} step={0.1} value={calibEps}
                              onChange={(e) => setCalibEps(parseFloat(e.target.value))} className="flex-1 accent-blue-500" />
                            <span className="text-[10px] text-gray-400 w-6">{calibEps.toFixed(1)}</span>
                          </div>
                        )}
                        {calibMethod === "beta_bb" && (
                          <div className="flex items-center gap-1.5">
                            <span className="text-[10px] text-gray-500 w-12">Prior</span>
                            <input type="range" min={0.5} max={50} step={0.5} value={calibPriorStrength}
                              onChange={(e) => setCalibPriorStrength(parseFloat(e.target.value))} className="flex-1 accent-blue-500" />
                            <span className="text-[10px] text-gray-400 w-6">{calibPriorStrength.toFixed(1)}</span>
                          </div>
                        )}
                        <button onClick={handleApplyCalibration} disabled={calibApplying}
                          className="w-full py-0.5 text-xs rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-gray-200 transition-colors">
                          {calibApplying ? "Applying…" : "Apply"}
                        </button>
                      </>
                    )}
                  </div>
                )}
              </div>

              {/* ── Column 2: Conditional + OOD + toggles ── */}
              <div className="flex flex-col gap-2">

                {/* Conditional inference */}
                <div className="border border-gray-700 rounded p-2 space-y-1.5">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">Conditional</span>
                    <div className="flex rounded overflow-hidden border border-gray-600 text-xs ml-auto">
                      {(["none", "head_sim", "lr_matrix"] as const).map((m) => (
                        <button key={m} onClick={() => setContextMethod(m)}
                          className={`px-1.5 py-0.5 ${contextMethod === m ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
                          title={m === "none" ? "No correction" : m === "head_sim" ? "Head weight cosine sim" : "Likelihood-Ratio matrix"}>
                          {m === "none" ? "Off" : m === "head_sim" ? "HeadSim" : "LR"}
                        </button>
                      ))}
                    </div>
                  </div>

                  {contextMethod !== "none" && (
                    <>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] text-gray-400 w-4 shrink-0">λ</span>
                        <input type="range" min={0.0} max={2.0} step={0.05} value={contextLambda}
                          onChange={(e) => setContextLambda(parseFloat(e.target.value))} className="flex-1" />
                        <span className="text-[10px] text-gray-400 font-mono w-7 text-right shrink-0">
                          {contextLambda.toFixed(2)}
                        </span>
                      </div>
                      {/* Known + tags */}
                      <div>
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className="text-[10px] text-green-400">Known +</span>
                        </div>
                        {knownTagsPos.length > 0 && (
                          <div className="flex flex-wrap gap-1 mb-1">
                            {knownTagsPos.map((t) => (
                              <span key={t} className="inline-flex items-center gap-1 px-1 py-0.5 rounded bg-green-900/40 border border-green-700 text-[10px] text-green-200">
                                {t}
                                <button onClick={() => removeKnownTag("pos", t)} className="text-green-400 hover:text-red-400 leading-none">×</button>
                              </span>
                            ))}
                          </div>
                        )}
                        <InputWithTagSuggestions
                          value={posTagInput} onChange={setPosTagInput}
                          onTagAdd={(tag) => { addKnownTag("pos")(tag); setPosTagInput(""); }}
                          placeholder="add positive tag…"
                          className="w-full px-2 py-0.5 bg-gray-800 border border-gray-600 rounded text-[10px] focus:outline-none focus:border-blue-500"
                          showSuggestionsAbove={true}
                        />
                      </div>
                      {/* Known − tags (collapsible) */}
                      <div>
                        <button onClick={() => setShowNegInput((v) => !v)}
                          className="text-[10px] text-gray-500 hover:text-gray-300 transition-colors">
                          {showNegInput ? "▾" : "▸"} Known − ({knownTagsNeg.length})
                        </button>
                        {showNegInput && (
                          <div className="mt-1">
                            {knownTagsNeg.length > 0 && (
                              <div className="flex flex-wrap gap-1 mb-1">
                                {knownTagsNeg.map((t) => (
                                  <span key={t} className="inline-flex items-center gap-1 px-1 py-0.5 rounded bg-red-900/40 border border-red-700 text-[10px] text-red-200">
                                    {t}
                                    <button onClick={() => removeKnownTag("neg", t)} className="text-red-400 hover:text-red-200 leading-none">×</button>
                                  </span>
                                ))}
                              </div>
                            )}
                            <InputWithTagSuggestions
                              value={negTagInput} onChange={setNegTagInput}
                              onTagAdd={(tag) => { addKnownTag("neg")(tag); setNegTagInput(""); }}
                              placeholder="add negative tag…"
                              className="w-full px-2 py-0.5 bg-gray-800 border border-gray-600 rounded text-[10px] focus:outline-none focus:border-blue-500"
                              showSuggestionsAbove={true}
                            />
                          </div>
                        )}
                      </div>
                    </>
                  )}
                </div>

                {/* OOD detection (best_thr only) */}
                {hasTagMetrics && inferMode === "best_thr" && (
                  <div className="border border-gray-700 rounded p-2 space-y-1.5">
                    <label className="flex items-center gap-1.5 cursor-pointer select-none">
                      <input type="checkbox" checked={useOodDetection}
                        onChange={(e) => setUseOodDetection(e.target.checked)}
                        disabled={!hasOodReference}
                        className="w-3 h-3 rounded accent-purple-500 disabled:opacity-40" />
                      <span className={`text-xs ${hasOodReference ? "text-gray-300" : "text-gray-500"}`}>
                        OOD検出{hasOodReference ? "" : " (参照未構築)"}
                      </span>
                      {hasOodReference && oodP50 != null && (
                        <span className="text-[10px] text-gray-500 ml-1">p50={oodP50.toFixed(1)}</span>
                      )}
                    </label>
                    <OodReferenceBuilder
                      building={oodBuilding} buildResult={oodBuildResult} buildError={oodBuildError}
                      onBuild={async (dir, maxImages) => {
                        setOodBuilding(true); setOodBuildError(null); setOodBuildResult(null);
                        try {
                          const res = await buildSigLIP2OodReference(dir, maxImages);
                          setOodBuildResult(res);
                          setHasOodReference(true);
                          setOodP50(res.p50);
                          setOodP95(res.p95);
                        } catch (e: any) {
                          setOodBuildError(e?.response?.data?.detail ?? e?.message ?? "Failed");
                        } finally { setOodBuilding(false); }
                      }}
                    />
                  </div>
                )}

                {/* Display calibration + misc toggles */}
                <div className="flex flex-col gap-1.5">
                  {hasTagMetrics && (
                    <label className="flex items-center gap-1.5 cursor-pointer select-none">
                      <input type="checkbox" checked={displayCalibrated}
                        onChange={(e) => setDisplayCalibrated(e.target.checked)}
                        className="w-3 h-3 rounded accent-blue-500" />
                      <span className="text-xs text-gray-400">確率表示: 校正後</span>
                    </label>
                  )}
                  <label className="flex items-center gap-1.5 cursor-pointer select-none"
                    title="Use the currently-training model for inference.">
                    <input type="checkbox" checked={useTrainingModel}
                      onChange={(e) => setUseTrainingModel(e.target.checked)}
                      className="w-3 h-3 rounded accent-blue-500" />
                    <span className="text-xs text-gray-400">Use training model</span>
                  </label>
                </div>
              </div>
            </div>

            {/* ── Predict button + status ── */}
            <button
              onClick={handlePredict}
              disabled={!imageBase64 || (!modelLoaded && !useTrainingModel) || running}
              className="py-2 rounded text-sm font-medium bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
            >
              {running ? "Running…" : "Predict Tags"}
            </button>

            {!modelLoaded && !useTrainingModel && (
              <p className="text-xs text-yellow-500">Load a model first (left panel)</p>
            )}
            {error && <p className="text-xs text-red-400 break-all">{error}</p>}

            {result && (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-gray-500">
                  {result.used_best_thr
                    ? (result.display_calibrated ? `校正後 / Best-thr` : "Raw / Best-thr")
                    : (result.calibrated
                        ? `後験確率 (${calibMethod === "jeffreys" ? `Jeffreys ε=${calibEps.toFixed(1)}` : `Beta-BB S=${calibPriorStrength.toFixed(1)}`})`
                        : "Raw / 固定閾値")}
                </span>
                {result.source === "training_model" && (
                  <span className="text-xs text-blue-400">
                    training model{result.run_id && ` (${result.run_id.slice(0, 8)})`}
                  </span>
                )}
                {result.ood_distance != null && (
                  <OodBadge distance={result.ood_distance} p50={oodP50} p95={oodP95} />
                )}
              </div>
            )}

            {/* Send-to buttons */}
            {result && (
              <div className="flex flex-wrap gap-2">
                {(["txt2img_params", "img2img_params", "inpaint_params"] as const).map((key, i) => (
                  <button key={key} onClick={() => sendTagsTo(key)} disabled={selectedTags.size === 0}
                    className="px-2.5 py-1 text-xs rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white transition-colors">
                    → {["Txt2Img", "Img2Img", "Inpaint"][i]}
                  </button>
                ))}
                {imageSrc && (
                  <>
                    <button onClick={() => sendImageTo("img2img")}
                      className="px-2.5 py-1 text-xs rounded bg-indigo-700 hover:bg-indigo-600 text-white transition-colors">
                      → Img2Img (img)
                    </button>
                    <button onClick={() => sendImageTo("inpaint")}
                      className="px-2.5 py-1 text-xs rounded bg-indigo-700 hover:bg-indigo-600 text-white transition-colors">
                      → Inpaint (img)
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ── Results chart (multi-column, fills remaining space) ── */}
        {result && (
          <div className="flex-1 overflow-y-auto min-h-0">
            <TagResultsChart
              tags={visibleTags}
              qualityTop={result.quality_top}
              ratingTop={result.rating_top}
              threshold={globalThreshold}
              selectedTags={selectedTags}
              onTagToggle={handleTagToggle}
              onSelectAll={handleSelectAll}
              onDeselectAll={handleDeselectAll}
            />
          </div>
        )}
      </div>
    </div>
  );
}

// ─── OOD Reference Builder ────────────────────────────────────────────────────

interface OodReferenceBuilderProps {
  building: boolean;
  buildResult: { p50: number; p95: number; n_images: number } | null;
  buildError: string | null;
  onBuild: (dir: string, maxImages: number) => Promise<void>;
}

function OodReferenceBuilder({ building, buildResult, buildError, onBuild }: OodReferenceBuilderProps) {
  const [expanded, setExpanded] = useState(false);
  const [dir,       setDir]       = useState("");
  const [maxImages, setMaxImages] = useState(2000);

  return (
    <div className="text-xs">
      <button onClick={() => setExpanded((v) => !v)}
        className="text-gray-500 hover:text-gray-300 underline">
        {expanded ? "▲ OOD参照構築を閉じる" : "▼ OOD参照を構築…"}
      </button>
      {expanded && (
        <div className="mt-1 space-y-1.5 p-2 bg-gray-800 rounded">
          <div>
            <label className="text-gray-400 text-[10px]">学習内データディレクトリ</label>
            <input type="text" value={dir} onChange={(e) => setDir(e.target.value)}
              placeholder="M:/dataset_07/..."
              className="w-full mt-0.5 px-2 py-1 bg-gray-700 rounded text-gray-200 text-[10px]" />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-gray-400 text-[10px] w-14 shrink-0">最大枚数</label>
            <input type="number" min={100} max={10000} step={100} value={maxImages}
              onChange={(e) => setMaxImages(parseInt(e.target.value) || 2000)}
              className="w-20 px-2 py-0.5 bg-gray-700 rounded text-gray-200 text-[10px]" />
          </div>
          <button onClick={() => onBuild(dir, maxImages)} disabled={building || !dir.trim()}
            className="w-full py-1 rounded bg-purple-700 hover:bg-purple-600 disabled:opacity-40 text-white transition-colors text-[10px]">
            {building ? "構築中…" : "OOD参照を構築"}
          </button>
          {buildError && <p className="text-red-400 text-[10px]">{buildError}</p>}
          {buildResult && (
            <p className="text-green-400 text-[10px]">
              完了: {buildResult.n_images} 枚 | p50={buildResult.p50.toFixed(1)} | p95={buildResult.p95.toFixed(1)}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
