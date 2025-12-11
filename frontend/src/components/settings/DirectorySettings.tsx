"use client";

import { useState, useEffect } from "react";
import Button from "@/components/common/Button";

interface DirectorySettingsData {
  model_dirs: string[];
  lora_dirs: string[];
  controlnet_dirs: string[];
  cache_dir: string | null;
}

export default function DirectorySettings() {
  const [modelDirs, setModelDirs] = useState("");
  const [loraDirs, setLoraDirs] = useState("");
  const [controlnetDirs, setControlnetDirs] = useState("");
  const [cacheDir, setCacheDir] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Load settings on mount
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await fetch("/api/settings/directories");
      if (!response.ok) {
        throw new Error("Failed to load directory settings");
      }

      const data: DirectorySettingsData = await response.json();

      // Convert arrays to newline-separated strings
      setModelDirs((data.model_dirs || []).join("\n"));
      setLoraDirs((data.lora_dirs || []).join("\n"));
      setControlnetDirs((data.controlnet_dirs || []).join("\n"));
      setCacheDir(data.cache_dir || "");
    } catch (error) {
      console.error("Error loading directory settings:", error);
      setMessage({ type: "error", text: "Failed to load directory settings" });
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    setMessage(null);

    try {
      // Split by newlines and filter empty lines
      const modelDirsArray = modelDirs.split("\n").map(d => d.trim()).filter(d => d);
      const loraDirsArray = loraDirs.split("\n").map(d => d.trim()).filter(d => d);
      const controlnetDirsArray = controlnetDirs.split("\n").map(d => d.trim()).filter(d => d);

      const response = await fetch("/api/settings/directories", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_dirs: modelDirsArray,
          lora_dirs: loraDirsArray,
          controlnet_dirs: controlnetDirsArray,
          cache_dir: cacheDir.trim() || null,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to save directory settings");
      }

      const result = await response.json();
      console.log("Save result:", result);

      setMessage({ type: "success", text: "Directory settings saved successfully! Restart the backend to apply changes." });
    } catch (error) {
      console.error("Error saving directory settings:", error);
      setMessage({ type: "error", text: "Failed to save directory settings. Please check the console." });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="text-gray-400 text-sm">
          Add additional directories to scan for models, LoRAs, and ControlNets. Enter one absolute path per line.
          These directories are scanned in addition to the default directories.
        </p>
        <div className="p-4 bg-gray-800 rounded-lg">
          <h3 className="text-sm font-semibold mb-2 text-yellow-400">Important Notes:</h3>
          <ul className="text-xs text-gray-400 space-y-1 list-disc list-inside">
            <li>Paths must be <strong>absolute paths</strong> (e.g., C:\Models or /home/user/models)</li>
            <li>Each directory is scanned recursively for model files</li>
            <li>Changes require a backend restart to take effect</li>
            <li>Invalid or non-existent paths will be logged but won&apos;t break scanning</li>
          </ul>
        </div>
      </div>

      {message && (
        <div className={`p-3 rounded ${message.type === "success" ? "bg-green-900/30 text-green-400" : "bg-red-900/30 text-red-400"}`}>
          {message.text}
        </div>
      )}

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Base Model Directories
          </label>
          <textarea
            value={modelDirs}
            onChange={(e) => setModelDirs(e.target.value)}
            placeholder="C:\StableDiffusion\models&#10;D:\AI\checkpoints"
            rows={4}
            className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm font-mono"
          />
          <p className="text-xs text-gray-500 mt-1">
            Directories containing Stable Diffusion checkpoints (.safetensors) or diffusers format models
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            LoRA Directories
          </label>
          <textarea
            value={loraDirs}
            onChange={(e) => setLoraDirs(e.target.value)}
            placeholder="C:\StableDiffusion\loras&#10;D:\AI\lora_models"
            rows={4}
            className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm font-mono"
          />
          <p className="text-xs text-gray-500 mt-1">
            Directories containing LoRA model files (.safetensors, .pt, .bin)
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            ControlNet Directories
          </label>
          <textarea
            value={controlnetDirs}
            onChange={(e) => setControlnetDirs(e.target.value)}
            placeholder="C:\StableDiffusion\controlnet&#10;D:\AI\controlnet_models"
            rows={4}
            className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm font-mono"
          />
          <p className="text-xs text-gray-500 mt-1">
            Directories containing ControlNet model files (.safetensors, .pth, .pt, .bin)
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Cache Directory
          </label>
          <input
            type="text"
            value={cacheDir}
            onChange={(e) => setCacheDir(e.target.value)}
            placeholder="D:\cache (leave empty for default: backend/cache)"
            className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm font-mono"
          />
          <p className="text-xs text-gray-500 mt-1">
            Custom directory for training caches (latents, text embeddings). Leave empty to use default (backend/cache).
          </p>
        </div>
      </div>

      <div className="flex gap-3">
        <Button onClick={handleSave} disabled={isSaving}>
          {isSaving ? "Saving..." : "Save Settings"}
        </Button>
        <Button onClick={loadSettings} variant="secondary" disabled={isSaving}>
          Reload
        </Button>
      </div>
    </div>
  );
}
