"use client";

import { useState, useEffect } from "react";
import Card from "./Card";
import Button from "./Button";
import Select from "./Select";
import { ChevronDown, ChevronUp, Folder } from "lucide-react";
import { useStartup } from "@/contexts/StartupContext";

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

export default function ModelSelector({ onModelLoad, embedded = false }: ModelSelectorProps) {
  const { modelLoaded, modelInfoVersion } = useStartup();
  const [models, setModels] = useState<Model[]>([]);
  const [currentModel, setCurrentModel] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [selectedModelPath, setSelectedModelPath] = useState<string>("");
  const [selectedArchitecture, setSelectedArchitecture] = useState<string>("all");
  const [selectedSourceDir, setSelectedSourceDir] = useState<string>("all");
  const [showDirectoryFilter, setShowDirectoryFilter] = useState(false);

  useEffect(() => {
    loadModels();
    loadCurrentModel();
  }, []);

  // Re-read the loaded model whenever the shared source says it changed --
  // including changes this page did not make (API call, backend restart,
  // another tab), which previously left the header showing the wrong model.
  useEffect(() => {
    if (modelLoaded) {
      loadCurrentModel();
    }
  }, [modelLoaded, modelInfoVersion]);

  const loadModels = async () => {
    try {
      const response = await fetch("/api/models");
      const data = await response.json();
      setModels(data.models || []);
    } catch (error) {
      console.error("Failed to load models:", error);
    }
  };

  const loadCurrentModel = async () => {
    try {
      const response = await fetch("/api/models/current");
      const data = await response.json();
      if (data.loaded) {
        setCurrentModel(data.model_info);
        if (data.model_info.source) {
          setSelectedModelPath(data.model_info.source);
        }
      } else {
        setCurrentModel(null);
        setSelectedModelPath("");
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
    const isReload = currentModel?.source === source;
    try {
      const formData = new FormData();
      formData.append("source_type", sourceType);
      formData.append("source", source);
      if (isReload) {
        formData.append("force", "true");
      }

      const response = await fetch("/api/models/load", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        await loadCurrentModel();
        if (onModelLoad) {
          onModelLoad(data.model_info);
        }
        alert(isReload ? "Model reloaded successfully!" : "Model loaded successfully!");
      }
    } catch (error) {
      console.error("Failed to load model:", error);
      alert("Failed to load model");
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

  const content = (
    <div className="grid gap-3 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        {/* Current Model Display */}
        {currentModel && (
          <div className="h-full rounded-md border border-gray-700 bg-gray-800/70 p-3">
            <p className="app-kicker">Active model</p>
            <p className="mt-1.5 truncate text-xs font-medium text-white" title={currentModel.source}>
              {currentModel.source}
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <p className="rounded bg-gray-900 px-2 py-0.5 text-[10px] text-gray-400">{currentModel.type || "Unknown"}</p>
              {currentModel.is_v_prediction && (
                <span className="rounded bg-violet-600/80 px-2 py-0.5 text-[10px] text-white">
                  v-prediction
                </span>
              )}
            </div>
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
                onChange={(e) => setSelectedModelPath(e.target.value)}
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

              {/* Model Details */}
              {selectedModelPath && (() => {
                if (!selectedModel) return null;
                return (
                  <div className="rounded-md border border-gray-700 bg-gray-800/70 p-2.5 text-xs sm:col-span-2">
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
                        disabled={loading}
                        className="w-full sm:w-auto"
                      >
                        {loading
                          ? "Loading..."
                          : currentModel?.source === selectedModelPath
                            ? "Reload Model"
                            : "Load Model"}
                      </Button>
                    </div>
                  </div>
                );
              })()}
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
          <div className="flex items-center justify-between text-sm py-1">
            <span className="text-gray-400">Currently Loaded:</span>
            <span className="text-white font-medium truncate ml-2">{currentModel.source}</span>
          </div>
        )
      }
    >
      {content}
    </Card>
  );
}
