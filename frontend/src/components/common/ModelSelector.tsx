"use client";

import { useState, useEffect } from "react";
import Card from "./Card";
import Button from "./Button";
import Select from "./Select";
import MiniMaxH3TextEncoderSelector from "./MiniMaxH3TextEncoderSelector";
import MiniMaxH3HybridSelector, { HYBRID_CHECK_PENDING } from "./MiniMaxH3HybridSelector";
import MiniMaxH3ReferenceBankPanel from "./MiniMaxH3ReferenceBankPanel";
import { ChevronDown, ChevronUp, Folder } from "lucide-react";
import { useStartup } from "@/contexts/StartupContext";
import {
  getCurrentModel,
  getModels,
  loadModel,
  MiniMaxH3HybridLoadRequest,
} from "@/utils/api";

interface Model {
  name: string;
  path: string;
  type: string;
  source_type: string;
  size_gb?: number;
  source_dir?: string;
  // Detected model architecture (sd15/sdxl/zimage/flux2/...). Falls back to
  // `type` (diffusers/safetensors, a file-format label, not an arch) when
  // the registry couldn't classify the arch for some reason.
  architecture?: string;
}

interface ModelSelectorProps {
  onModelLoad?: (modelInfo: any) => void;
  embedded?: boolean;
}

// The merged pair a LOADED MiniMax-H3 DiT was built from, read from
// current_model_info rather than from the selection above it: the two disagree
// the moment the user edits the recipe without pressing Load.
const hybridSummary = (modelInfo: any): string | null => {
  const hybrid = modelInfo?.hybrid;
  if (!hybrid || modelInfo?.variant !== "hybrid") return null;
  const recipe = hybrid.hybrid_recipe || {};
  const base = hybrid.base_variant || hybrid.base_file || "base";
  const overlay = hybrid.overlay_variant || hybrid.overlay_file || "overlay";
  const range = `blocks ${recipe.block_range_start}..${recipe.block_range_end}`;
  const final = recipe.final_adaln_from_overlay ? " + final AdaLN" : "";
  return `${base} + ${overlay} / ${range}${final}`;
};

export default function ModelSelector({ onModelLoad, embedded = false }: ModelSelectorProps) {
  const { modelInfoVersion, refreshModelInfo } = useStartup();
  const [models, setModels] = useState<Model[]>([]);
  const [currentModel, setCurrentModel] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [selectedModelPath, setSelectedModelPath] = useState<string>("");
  const [selectedArchitecture, setSelectedArchitecture] = useState<string>("all");
  const [selectedSourceDir, setSelectedSourceDir] = useState<string>("all");
  const [showDirectoryFilter, setShowDirectoryFilter] = useState(false);
  const [loadError, setLoadError] = useState("");
  // MiniMax-H3 only: the load-time text encoder / projection choice, remembered
  // per model path. null on either means "let the loader decide".
  const [h3TextEncoder, setH3TextEncoder] = useState<string | null>(null);
  const [h3ClipProjection, setH3ClipProjection] = useState<string | null>(null);
  // MiniMax-H3 only: the overlay checkpoint and recipe merged into the selected
  // base, remembered per base path. null means the single-checkpoint load.
  const [h3Hybrid, setH3Hybrid] = useState<MiniMaxH3HybridLoadRequest | null>(null);
  // Why the current hybrid selection must not be sent, or null when it may be.
  const [h3HybridBlocked, setH3HybridBlocked] = useState<string | null>(null);

  const h3StorageKey = selectedModelPath ? `minimax_h3_te_choice_${selectedModelPath}` : "";
  const h3HybridStorageKey = selectedModelPath
    ? `minimax_h3_hybrid_choice_${selectedModelPath}`
    : "";

  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = h3StorageKey ? localStorage.getItem(h3StorageKey) : null;
    if (!saved) {
      setH3TextEncoder(null);
      setH3ClipProjection(null);
      return;
    }
    try {
      const parsed = JSON.parse(saved);
      setH3TextEncoder(typeof parsed?.text_encoder === "string" ? parsed.text_encoder : null);
      setH3ClipProjection(typeof parsed?.clip_projection === "string" ? parsed.clip_projection : null);
    } catch {
      setH3TextEncoder(null);
      setH3ClipProjection(null);
    }
  }, [h3StorageKey]);

  // Written here rather than in an effect: an effect would fire once with the
  // previous model's choice already under the new model's key.
  const handleH3Change = (textEncoder: string | null, clipProjection: string | null) => {
    setH3TextEncoder(textEncoder);
    setH3ClipProjection(clipProjection);
    if (typeof window === "undefined" || !h3StorageKey) return;
    if (!textEncoder && !clipProjection) {
      localStorage.removeItem(h3StorageKey);
    } else {
      localStorage.setItem(
        h3StorageKey,
        JSON.stringify({ text_encoder: textEncoder, clip_projection: clipProjection })
      );
    }
  };

  // A restored overlay starts BLOCKED: the selector has not answered for it
  // yet, and until it does the load button must not offer an unchecked pair.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = h3HybridStorageKey ? localStorage.getItem(h3HybridStorageKey) : null;
    let restored: MiniMaxH3HybridLoadRequest | null = null;
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (typeof parsed?.overlay_file === "string" && parsed.overlay_file) {
          restored = parsed as MiniMaxH3HybridLoadRequest;
        }
      } catch {
        restored = null;
      }
    }
    setH3Hybrid(restored);
    setH3HybridBlocked(restored ? HYBRID_CHECK_PENDING : null);
  }, [h3HybridStorageKey]);

  const handleH3HybridChange = (next: MiniMaxH3HybridLoadRequest | null) => {
    setH3Hybrid(next);
    if (typeof window === "undefined" || !h3HybridStorageKey) return;
    if (!next?.overlay_file) {
      localStorage.removeItem(h3HybridStorageKey);
    } else {
      localStorage.setItem(h3HybridStorageKey, JSON.stringify(next));
    }
  };

  useEffect(() => {
    loadModels();
    loadCurrentModel();
  }, []);

  // Re-read the loaded model whenever the shared source says it changed --
  // including changes this page did not make (API call, backend restart,
  // another tab), which previously left the header showing the wrong model.
  useEffect(() => {
    loadCurrentModel();
  }, [modelInfoVersion]);

  const loadModels = async () => {
    try {
      const data = await getModels();
      setModels(data.models || []);
    } catch (error) {
      console.error("Failed to load models:", error);
    }
  };

  const loadCurrentModel = async () => {
    try {
      const data = await getCurrentModel();
      if (data.loaded) {
        setCurrentModel(data.model_info);
        if (data.model_info.source) {
          setSelectedModelPath(data.model_info.source);
        }
      } else {
        setCurrentModel(null);
      }
    } catch (error) {
      console.error("Failed to load current model:", error);
    }
  };

  // Loading the model that is ALREADY loaded sends force=true: without it the
  // backend early-returns and the click does nothing at all. That reload is the
  // documented (and only) way to undo per-session mutations of the loaded
  // components — above all the one-way in-place INT8 conversion
  // (unet_quantization="int8" on anima/krea2/flux2/ideogram4), whose warnings tell the user to
  // load the model again.
  const handleLoadModel = async (sourceType: string, source: string) => {
    setLoading(true);
    setLoadError("");
    // Same path => force, which is also what makes a changed text encoder or
    // projection take effect: the backend early-returns on an identical
    // model_id, and the encoder choice is not part of that id.
    const isReload = currentModel?.source === source;
    const model = models.find(m => m.path === source);
    const isMiniMaxH3 = (model?.architecture || model?.type) === "minimax_h3";
    try {
      const data = await loadModel(
        sourceType,
        source,
        undefined,
        isReload,
        isMiniMaxH3 ? h3TextEncoder : null,
        isMiniMaxH3 ? h3ClipProjection : null,
        isMiniMaxH3 ? h3Hybrid : null
      );
      if (!data.success) {
        throw new Error(data.detail || data.message || "The model could not be loaded.");
      }
      await refreshModelInfo();
      await loadCurrentModel();
      if (onModelLoad) {
        await onModelLoad(data.model_info);
      }
      const merged = hybridSummary(data.model_info);
      alert(
        (isReload ? "Model reloaded successfully!" : "Model loaded successfully!") +
          (merged
            ? `\n\nMerged checkpoint: ${merged}\nEvery generation endpoint refuses a merged ` +
              "checkpoint; it can be loaded and inspected only."
            : "")
      );
    } catch (error: any) {
      console.error("Failed to load model:", error);
      await refreshModelInfo();
      await loadCurrentModel();
      const detail = error?.response?.data?.detail;
      setLoadError(
        (typeof detail === "string" && detail) ||
        error?.message ||
        "The model could not be loaded."
      );
    } finally {
      setLoading(false);
    }
  };

  // Architecture is the PRIMARY filter (which model family). Directory is
  // SECONDARY (where it's stored) and only narrows within the arch-filtered
  // set, so its options stay relevant to the current architecture selection.
  const archOf = (m: Model) => m.architecture || m.type || "Unknown";
  const uniqueArchitectures = Array.from(new Set(models.map(archOf))).sort();
  const archFilteredModels = models.filter(
    m => selectedArchitecture === "all" || archOf(m) === selectedArchitecture
  );
  const uniqueDirs = Array.from(new Set(archFilteredModels.map(m => m.source_dir || "Unknown")));
  const filteredModels = archFilteredModels.filter(
    m => selectedSourceDir === "all" || m.source_dir === selectedSourceDir
  );
  const selectedModel = models.find(m => m.path === selectedModelPath);
  const selectedIsMiniMaxH3 = !!selectedModel && archOf(selectedModel) === "minimax_h3";
  // Only a chosen overlay can block: without one the request is the
  // single-checkpoint load, which no compatibility check applies to.
  const hybridLoadBlocked =
    selectedIsMiniMaxH3 && !!h3Hybrid?.overlay_file ? h3HybridBlocked : null;
  const loadedHybridSummary = hybridSummary(currentModel);

  const content = (
    <div
      className="grid gap-2"
      style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 17rem), 1fr))" }}
    >
        {/* Current Model Display */}
        {currentModel && (
          <div className="h-full rounded-md border border-gray-700 bg-gray-800/70 p-2.5">
            <p className="app-kicker">Active model</p>
            <p className="mt-1 truncate text-xs font-medium text-white" title={currentModel.source}>
              {currentModel.source}
            </p>
            <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
              <p className="rounded bg-gray-900 px-2 py-0.5 text-[10px] text-gray-400">{currentModel.type || "Unknown"}</p>
              {currentModel.is_v_prediction && (
                <span className="rounded bg-violet-600/80 px-2 py-0.5 text-[10px] text-white">
                  v-prediction
                </span>
              )}
              {loadedHybridSummary && (
                <span className="rounded bg-amber-600/80 px-2 py-0.5 text-[10px] text-white">
                  merged
                </span>
              )}
            </div>
            {loadedHybridSummary && (
              <div className="mt-1.5 space-y-1 text-[11px] leading-relaxed">
                <p className="text-gray-300">{loadedHybridSummary}</p>
                <p className="text-amber-300">
                  Every generation endpoint refuses a merged checkpoint, so a generation request
                  from any panel returns an error until the A/B measurement releases it. Loading
                  and inspecting the model works.
                </p>
              </div>
            )}
          </div>
        )}

        <div className={`grid gap-2 sm:grid-cols-2 ${currentModel ? "" : "lg:col-span-2"}`}>
          {models.length === 0 ? (
            <p className="text-gray-500 text-sm">No local models found. Place models in the models/ directory.</p>
          ) : (
            <>
              {/* Architecture Filter — PRIMARY (only shown when >1 architecture present) */}
              {uniqueArchitectures.length > 1 && (
                <Select
                  label="Architecture"
                  value={selectedArchitecture}
                  onChange={(e) => {
                    setSelectedArchitecture(e.target.value);
                    // Reset the secondary filter: its options are scoped to
                    // the arch selection, so a stale dir choice may no longer
                    // exist in the new set.
                    setSelectedSourceDir("all");
                  }}
                  options={[
                    { value: "all", label: "All architectures" },
                    ...uniqueArchitectures.map(arch => ({ value: arch, label: arch }))
                  ]}
                />
              )}

              {/* Model Dropdown */}
              <Select
                label="Model"
                className={uniqueArchitectures.length > 1 ? "" : "sm:col-span-2"}
                value={selectedModelPath}
                onChange={(e) => {
                  setSelectedModelPath(e.target.value);
                  setLoadError("");
                }}
                options={[
                  { value: "", label: "-- Select a model --" },
                  ...filteredModels.map(model => ({
                    value: model.path,
                    label: `${model.name} (${archOf(model)}${model.size_gb ? ` • ${model.size_gb} GB` : ''})`
                  }))
                ]}
              />

              {/* Directory Filter — secondary to architecture and model. */}
              {uniqueDirs.length > 1 && (
                <div className="sm:col-span-2">
                  <button
                    type="button"
                    onClick={() => setShowDirectoryFilter(!showDirectoryFilter)}
                    className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-300"
                    aria-expanded={showDirectoryFilter}
                  >
                    <Folder className="h-3 w-3" />
                    More filters (directory)
                    {showDirectoryFilter ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                  </button>
                  {showDirectoryFilter && (
                    <Select
                      className="mt-2"
                      label="Directory"
                      value={selectedSourceDir}
                      onChange={(e) => setSelectedSourceDir(e.target.value)}
                      options={[
                        { value: "all", label: "All Directories" },
                        ...uniqueDirs.map(dir => ({ value: dir, label: dir }))
                      ]}
                    />
                  )}
                </div>
              )}

              {/* Load-time text encoder choice (MiniMax-H3 only) */}
              {selectedModel && selectedIsMiniMaxH3 && (
                <MiniMaxH3TextEncoderSelector
                  className="sm:col-span-2"
                  modelPath={selectedModel.path}
                  textEncoderPath={h3TextEncoder}
                  clipProjectionPath={h3ClipProjection}
                  onChange={handleH3Change}
                  disabled={loading}
                />
              )}

              {/* Load-time overlay merge (MiniMax-H3 only). The model dropdown
                  above picks the base; this picks the second checkpoint. */}
              {selectedModel && selectedIsMiniMaxH3 && (
                <MiniMaxH3HybridSelector
                  className="sm:col-span-2"
                  modelPath={selectedModel.path}
                  value={h3Hybrid}
                  onChange={handleH3HybridChange}
                  onBlockedChange={setH3HybridBlocked}
                  disabled={loading}
                />
              )}

              {/* The reference bank is about the encoder that is LOADED, not the
                  one selected above, so it follows currentModel. */}
              {currentModel?.type === "minimax_h3" && (
                <MiniMaxH3ReferenceBankPanel className="sm:col-span-2" modelVersion={modelInfoVersion} />
              )}

              {/* Model Details */}
              {selectedModelPath && (() => {
                if (!selectedModel) return null;
                return (
                  <div className="rounded-md border border-gray-700 bg-gray-800/70 p-2 text-xs sm:col-span-2">
                    <div className="grid gap-2 sm:grid-cols-[auto_auto_minmax(0,1fr)_auto] sm:items-center">
                      <div>
                        <span className="text-gray-500">Architecture</span>
                        <span className="ml-1.5 text-white">{archOf(selectedModel)}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Size</span>
                        <span className="ml-1.5 text-white">{selectedModel.size_gb ? `${selectedModel.size_gb} GB` : "N/A"}</span>
                      </div>
                      <p className="min-w-0 truncate font-mono text-[10px] text-gray-400" title={selectedModel.path}>{selectedModel.path}</p>
                      <Button
                        onClick={() => handleLoadModel(selectedModel.source_type, selectedModel.path)}
                        disabled={loading || !!hybridLoadBlocked}
                        title={hybridLoadBlocked || undefined}
                        className="w-full sm:w-auto"
                      >
                        {loading
                          ? "Loading..."
                          : currentModel?.source === selectedModelPath
                            ? "Reload Model"
                            : "Load Model"}
                      </Button>
                    </div>
                    {hybridLoadBlocked && (
                      <p className="mt-1.5 text-[11px] leading-relaxed text-amber-300">
                        {hybridLoadBlocked}
                      </p>
                    )}
                  </div>
                );
              })()}
              {loadError && (
                <div
                  role="alert"
                  aria-live="assertive"
                  className="max-h-28 overflow-auto rounded border border-red-500/40 bg-red-950/40 px-2.5 py-2 text-[11px] leading-relaxed text-red-200 sm:col-span-2"
                >
                  <span className="font-semibold">Model load failed: </span>
                  <span className="break-words">{loadError}</span>
                </div>
              )}
            </>
          )}
        </div>
    </div>
  );

  if (embedded) return content;

  return (
    <Card
      title="Model Selection"
      collapsible={true}
      defaultCollapsed={false}
      storageKey="model_selection_collapsed"
      collapsedPreview={
        currentModel && (
          <div className="py-1 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Currently Loaded:</span>
              <span className="text-white font-medium truncate ml-2">{currentModel.source}</span>
            </div>
            {loadedHybridSummary && (
              <p className="mt-1 text-right text-[11px] text-amber-300">
                merged: {loadedHybridSummary} — generation endpoints refuse it
              </p>
            )}
          </div>
        )
      }
    >
      {content}
    </Card>
  );
}
