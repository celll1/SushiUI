"use client";

import { useState, useEffect } from "react";
import Card from "./Card";
import Button from "./Button";
import Input from "./Input";
import Select from "./Select";
import { Folder, Globe } from "lucide-react";

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
  const [models, setModels] = useState<Model[]>([]);
  const [currentModel, setCurrentModel] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [selectedTab, setSelectedTab] = useState<"local" | "huggingface">("local");

  // Form states
  const [huggingfaceRepo, setHuggingfaceRepo] = useState("");
  const [huggingfaceRevision, setHuggingfaceRevision] = useState("");
  const [localModelPath, setLocalModelPath] = useState("");
  const [browseFile, setBrowseFile] = useState<File | null>(null);

  // VRAM Optimization states
  const [precision, setPrecision] = useState<"fp32" | "fp16" | "fp8">("fp16");
  const [textEncoderOffload, setTextEncoderOffload] = useState<"gpu" | "cpu" | "auto">("auto");
  const [vaeOffload, setVaeOffload] = useState<"gpu" | "cpu" | "auto">("auto");

  useEffect(() => {
    loadModels();
    loadCurrentModel();
  }, []);

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

      // Add VRAM optimization settings
      formData.append("precision", precision);
      formData.append("text_encoder_offload", textEncoderOffload);
      formData.append("vae_offload", vaeOffload);

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

  const handleBrowseAndLoad = async () => {
    if (!browseFile) return;

    setLoading(true);
    try {
      // First, upload the file
      const formData = new FormData();
      formData.append("file", browseFile);

      const uploadResponse = await fetch("/api/models/upload", {
        method: "POST",
        body: formData,
      });

      const uploadData = await uploadResponse.json();
      if (!uploadData.success) {
        throw new Error("Upload failed");
      }

      // Reload models list
      await loadModels();

      // Then load the uploaded model
      const loadFormData = new FormData();
      loadFormData.append("source_type", "safetensors");
      loadFormData.append("source", uploadData.path);

      const loadResponse = await fetch("/api/models/load", {
        method: "POST",
        body: loadFormData,
      });

      const loadData = await loadResponse.json();
      if (loadData.success) {
        setCurrentModel(loadData.model_info);
        if (onModelLoad) {
          onModelLoad(loadData.model_info);
        }
        alert("Model uploaded and loaded successfully!");
        setBrowseFile(null);
      }
    } catch (error) {
      console.error("Failed to upload and load model:", error);
      alert("Failed to upload and load model");
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
            <p className="text-xs text-gray-500">Type: {currentModel.type || "Unknown"}</p>
          </div>
        )}

        {/* VRAM Optimization Settings */}
        <div className="bg-gray-800 p-3 rounded-lg space-y-3">
          <h4 className="text-sm font-semibold text-gray-300">VRAM Optimization</h4>

          <div className="grid grid-cols-3 gap-3">
            <Select
              label="Precision"
              value={precision}
              onChange={(e) => setPrecision(e.target.value as "fp32" | "fp16" | "fp8")}
              options={[
                { value: "fp32", label: "FP32 (Full)" },
                { value: "fp16", label: "FP16 (Half)" },
                { value: "fp8", label: "FP8 (Experimental)" },
              ]}
            />

            <Select
              label="Text Encoder"
              value={textEncoderOffload}
              onChange={(e) => setTextEncoderOffload(e.target.value as "gpu" | "cpu" | "auto")}
              options={[
                { value: "gpu", label: "GPU" },
                { value: "cpu", label: "CPU" },
                { value: "auto", label: "Auto Offload" },
              ]}
            />

            <Select
              label="VAE"
              value={vaeOffload}
              onChange={(e) => setVaeOffload(e.target.value as "gpu" | "cpu" | "auto")}
              options={[
                { value: "gpu", label: "GPU" },
                { value: "cpu", label: "CPU" },
                { value: "auto", label: "Auto Offload" },
              ]}
            />
          </div>

          <p className="text-xs text-gray-500">
            <strong>Precision:</strong> Lower precision uses less VRAM. FP8 is experimental.<br/>
            <strong>CPU:</strong> Component stays on CPU/RAM (slower).<br/>
            <strong>Auto Offload:</strong> Moves to CPU after use (saves VRAM).
          </p>
        </div>

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
          <div className="space-y-2">
            {models.length === 0 ? (
              <p className="text-gray-500 text-sm">No local models found. Place models in the models/ directory.</p>
            ) : (
              <div className="space-y-2">
                {models.map((model, idx) => (
                  <div key={idx} className="flex items-center justify-between bg-gray-800 p-3 rounded-lg">
                    <div>
                      <p className="text-white">{model.name}</p>
                      <p className="text-xs text-gray-500">
                        {model.type} {model.size_gb && `• ${model.size_gb} GB`}
                      </p>
                    </div>
                    <Button
                      onClick={() => handleLoadModel(model.source_type, model.path)}
                      disabled={loading}
                      size="sm"
                    >
                      Load
                    </Button>
                  </div>
                ))}
              </div>
            )}

            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Or select a local file
              </label>
              <div className="space-y-2">
                <div className="flex space-x-2">
                  <Input
                    placeholder="Select .safetensors file"
                    value={browseFile?.name || ""}
                    readOnly
                    className="flex-1"
                  />
                  <input
                    type="file"
                    accept=".safetensors"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        setBrowseFile(file);
                      }
                    }}
                    className="hidden"
                    id="local-file-browse"
                  />
                  <Button
                    size="sm"
                    onClick={() => document.getElementById('local-file-browse')?.click()}
                  >
                    Browse
                  </Button>
                </div>
                {browseFile && (
                  <div className="bg-gray-800 p-2 rounded-lg">
                    <p className="text-xs text-gray-400">
                      {(browseFile.size / (1024 ** 3)).toFixed(2)} GB
                    </p>
                  </div>
                )}
                <Button
                  onClick={handleBrowseAndLoad}
                  disabled={!browseFile || loading}
                  className="w-full"
                >
                  {loading ? "Uploading and Loading..." : "Upload and Load"}
                </Button>
              </div>
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Or enter full path manually
              </label>
              <Input
                placeholder="/path/to/model or model.safetensors"
                value={localModelPath}
                onChange={(e) => setLocalModelPath(e.target.value)}
              />
              <Button
                onClick={() => {
                  const sourceType = localModelPath.endsWith('.safetensors') ? 'safetensors' : 'diffusers';
                  handleLoadModel(sourceType, localModelPath);
                }}
                disabled={!localModelPath || loading}
                className="mt-2 w-full"
              >
                Load Path
              </Button>
            </div>
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
