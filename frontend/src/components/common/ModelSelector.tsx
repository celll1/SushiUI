"use client";

import { useState, useEffect } from "react";
import Card from "./Card";
import Button from "./Button";
import Input from "./Input";
import { Upload, Folder, Globe } from "lucide-react";

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
  const [selectedTab, setSelectedTab] = useState<"local" | "huggingface" | "upload">("local");

  // Form states
  const [huggingfaceRepo, setHuggingfaceRepo] = useState("");
  const [huggingfaceRevision, setHuggingfaceRevision] = useState("");
  const [localModelPath, setLocalModelPath] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);

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

  const handleUpload = async () => {
    if (!uploadFile) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);

      const response = await fetch("/api/models/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        alert("Model uploaded successfully!");
        loadModels();
        setUploadFile(null);
      }
    } catch (error) {
      console.error("Failed to upload model:", error);
      alert("Failed to upload model");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="Model Selection">
      <div className="space-y-4">
        {/* Current Model Display */}
        {currentModel && (
          <div className="bg-gray-800 p-3 rounded-lg">
            <p className="text-sm text-gray-400">Currently Loaded:</p>
            <p className="text-white font-medium">{currentModel.source}</p>
            <p className="text-xs text-gray-500">Type: {currentModel.type || "Unknown"}</p>
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
          <button
            onClick={() => setSelectedTab("upload")}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              selectedTab === "upload"
                ? "border-b-2 border-blue-500 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            <Upload className="inline w-4 h-4 mr-2" />
            Upload
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
              <Input
                label="Or enter path manually"
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

        {/* Upload Tab */}
        {selectedTab === "upload" && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Upload .safetensors file
              </label>
              <input
                type="file"
                accept=".safetensors"
                onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                className="block w-full text-sm text-gray-400
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-medium
                  file:bg-blue-600 file:text-white
                  hover:file:bg-blue-700
                  file:cursor-pointer cursor-pointer"
              />
            </div>
            {uploadFile && (
              <div className="bg-gray-800 p-3 rounded-lg">
                <p className="text-sm text-white">{uploadFile.name}</p>
                <p className="text-xs text-gray-500">
                  {(uploadFile.size / (1024 ** 3)).toFixed(2)} GB
                </p>
              </div>
            )}
            <Button
              onClick={handleUpload}
              disabled={!uploadFile || loading}
              className="w-full"
            >
              {loading ? "Uploading..." : "Upload Model"}
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
}
