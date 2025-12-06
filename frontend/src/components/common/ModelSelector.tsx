"use client";

import { useState, useEffect } from "react";
import Card from "./Card";
import Button from "./Button";
import Input from "./Input";
import Select from "./Select";
import { Folder, Globe } from "lucide-react";
import { useStartup } from "@/contexts/StartupContext";

interface Model {
  name: string;
  path: string;
  type: string;
  source_type: string;
  size_gb?: number;
}

interface ModelSelectorProps {
  onModelLoad?: (modelInfo: any) => void;
}

export default function ModelSelector({ onModelLoad }: ModelSelectorProps) {
  const { modelLoaded } = useStartup();
  const [models, setModels] = useState<Model[]>([]);
  const [currentModel, setCurrentModel] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [selectedTab, setSelectedTab] = useState<"local" | "huggingface">("local");

  // Form states
  const [huggingfaceRepo, setHuggingfaceRepo] = useState("");
  const [huggingfaceRevision, setHuggingfaceRevision] = useState("");

  // Filter states
  const [selectedModelPath, setSelectedModelPath] = useState<string>("");
  const [selectedSourceDir, setSelectedSourceDir] = useState<string>("all");

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    // Load current model on startup
    if (modelLoaded) {
      loadCurrentModel();
    }

    // Also poll current model periodically (in case model was loaded to CPU)
    const interval = setInterval(() => {
      loadCurrentModel();
    }, 5000); // Check every 5 seconds

    return () => clearInterval(interval);
  }, [modelLoaded]);

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
        // Sync selectedModelPath with current model
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

  const handleLoadModel = async (sourceType: string, source: string, revision?: string) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("source_type", sourceType);
      formData.append("source", source);
      if (revision) {
        formData.append("revision", revision);
      }

      const response = await fetch("/api/models/load", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setCurrentModel(data.model_info);
        if (onModelLoad) {
          onModelLoad(data.model_info);
        }
        alert("Model loaded successfully!");
      }
    } catch (error) {
      console.error("Failed to load model:", error);
      alert("Failed to load model");
    } finally {
      setLoading(false);
    }
  };

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
      <div className="space-y-4">
        {/* Current Model Display */}
        {currentModel && (
          <div className="bg-gray-800 p-3 rounded-lg">
            <p className="text-sm text-gray-400">Currently Loaded:</p>
            <p className="text-white font-medium">{currentModel.source}</p>
            <div className="flex items-center gap-2 mt-1">
              <p className="text-xs text-gray-500">Type: {currentModel.type || "Unknown"}</p>
              {currentModel.is_v_prediction && (
                <span className="text-xs bg-purple-600 text-white px-2 py-0.5 rounded">
                  v-prediction
                </span>
              )}
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="flex space-x-2 border-b border-gray-700">
          <button
            onClick={() => setSelectedTab("local")}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              selectedTab === "local"
                ? "border-b-2 border-blue-500 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            <Folder className="inline w-4 h-4 mr-2" />
            Local Files
          </button>
          <button
            onClick={() => setSelectedTab("huggingface")}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              selectedTab === "huggingface"
                ? "border-b-2 border-blue-500 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            <Globe className="inline w-4 h-4 mr-2" />
            HuggingFace
          </button>
        </div>

        {/* Local Files Tab */}
        {selectedTab === "local" && (
          <div className="space-y-4">
            {models.length === 0 ? (
              <p className="text-gray-500 text-sm">No local models found. Place models in the models/ directory.</p>
            ) : (
              <div className="space-y-3">
                {/* Directory Filter */}
                {(() => {
                  const uniqueDirs = Array.from(new Set(models.map(m => m.source_dir || "Unknown")));
                  if (uniqueDirs.length > 1) {
                    return (
                      <Select
                        label="Filter by Directory"
                        value={selectedSourceDir}
                        onChange={(e) => setSelectedSourceDir(e.target.value)}
                        options={[
                          { value: "all", label: "All Directories" },
                          ...uniqueDirs.map(dir => ({
                            value: dir,
                            label: dir
                          }))
                        ]}
                      />
                    );
                  }
                  return null;
                })()}

                {/* Model Dropdown */}
                <div className="space-y-2">
                  <Select
                    label="Select Model"
                    value={selectedModelPath}
                    onChange={(e) => setSelectedModelPath(e.target.value)}
                    options={[
                      { value: "", label: "-- Select a model --" },
                      ...models
                        .filter(model =>
                          selectedSourceDir === "all" ||
                          model.source_dir === selectedSourceDir
                        )
                        .map(model => ({
                          value: model.path,
                          label: `${model.name} (${model.type}${model.size_gb ? ` • ${model.size_gb} GB` : ''})`
                        }))
                    ]}
                  />

                  {/* Model Details */}
                  {selectedModelPath && (() => {
                    const selectedModel = models.find(m => m.path === selectedModelPath);
                    if (selectedModel) {
                      return (
                        <div className="bg-gray-800 p-3 rounded-lg text-sm">
                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <span className="text-gray-400">Type:</span>
                              <span className="ml-2 text-white">{selectedModel.type}</span>
                            </div>
                            {selectedModel.size_gb && (
                              <div>
                                <span className="text-gray-400">Size:</span>
                                <span className="ml-2 text-white">{selectedModel.size_gb} GB</span>
                              </div>
                            )}
                            <div className="col-span-2">
                              <span className="text-gray-400">Path:</span>
                              <div className="mt-1 text-xs text-white break-all bg-gray-900 p-2 rounded">
                                {selectedModel.path}
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  })()}

                  <Button
                    onClick={() => {
                      const selectedModel = models.find(m => m.path === selectedModelPath);
                      if (selectedModel) {
                        handleLoadModel(selectedModel.source_type, selectedModel.path);
                      }
                    }}
                    disabled={!selectedModelPath || loading}
                    className="w-full"
                  >
                    {loading ? "Loading..." : "Load Selected Model"}
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* HuggingFace Tab */}
        {selectedTab === "huggingface" && (
          <div className="space-y-4">
            <Input
              label="Repository ID"
              placeholder="runwayml/stable-diffusion-v1-5"
              value={huggingfaceRepo}
              onChange={(e) => setHuggingfaceRepo(e.target.value)}
            />
            <Input
              label="Revision (optional)"
              placeholder="main"
              value={huggingfaceRevision}
              onChange={(e) => setHuggingfaceRevision(e.target.value)}
            />
            <Button
              onClick={() => handleLoadModel("huggingface", huggingfaceRepo, huggingfaceRevision || undefined)}
              disabled={!huggingfaceRepo || loading}
              className="w-full"
            >
              {loading ? "Loading..." : "Load from HuggingFace"}
            </Button>

            <div className="mt-4 p-3 bg-gray-800 rounded-lg">
              <p className="text-xs text-gray-400">Popular models:</p>
              <div className="mt-2 space-y-1">
                {[
                  "runwayml/stable-diffusion-v1-5",
                  "stabilityai/stable-diffusion-xl-base-1.0",
                  "stabilityai/stable-diffusion-2-1",
                ].map((repo) => (
                  <button
                    key={repo}
                    onClick={() => setHuggingfaceRepo(repo)}
                    className="block text-xs text-blue-400 hover:text-blue-300"
                  >
                    {repo}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
