"use client";

import { useState, useEffect } from "react";
import Button from "@/components/common/Button";

// The video-frame-slider-max field used to live here too, but it is now an
// immediate-apply control (Settings page's "Generation Behavior" card,
// frontend/src/app/settings/page.tsx) instead of a save-button field: it
// needed to behave like the other generation controls there (Resolution
// slider step size, Attention Type), and this component's Save button covers
// only inpaint_use_dedicated_model.
interface GenerationSettingsData {
  inpaint_use_dedicated_model: boolean;
}

export default function GenerationSettings() {
  const [inpaintUseDedicatedModel, setInpaintUseDedicatedModel] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Load settings on mount
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await fetch("/api/settings/generation");
      if (!response.ok) {
        throw new Error("Failed to load generation settings");
      }

      const data: GenerationSettingsData = await response.json();
      setInpaintUseDedicatedModel(data.inpaint_use_dedicated_model || false);
    } catch (error) {
      console.error("Error loading generation settings:", error);
      setMessage({ type: "error", text: "Failed to load generation settings" });
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    setMessage(null);

    try {
      const response = await fetch("/api/settings/generation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          inpaint_use_dedicated_model: inpaintUseDedicatedModel,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to save generation settings");
      }

      const result = await response.json();
      console.log("Save result:", result);

      setMessage({ type: "success", text: "Generation settings saved successfully!" });
    } catch (error) {
      console.error("Error saving generation settings:", error);
      setMessage({ type: "error", text: "Failed to save generation settings. Please check the console." });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="text-gray-400 text-sm">
          Configure image generation behavior and optimization settings.
        </p>
      </div>

      {message && (
        <div className={`p-3 rounded ${message.type === "success" ? "bg-green-900/30 text-green-400" : "bg-red-900/30 text-red-400"}`}>
          {message.text}
        </div>
      )}

      <div className="space-y-4">
        <div className="p-4 bg-gray-800 rounded-lg">
          <h3 className="text-sm font-semibold mb-3 text-white">Inpaint Method</h3>

          <div className="space-y-3">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="radio"
                name="inpaint_method"
                checked={!inpaintUseDedicatedModel}
                onChange={() => setInpaintUseDedicatedModel(false)}
                className="mt-1"
              />
              <div>
                <span className="text-white font-medium">Mask Blending (Default)</span>
                <p className="text-xs text-gray-400 mt-1">
                  Use the same method as Z-Image/FLUX.2. Works with any model.
                  Blends the original image with the generated result using the mask.
                  This is the recommended method for most use cases.
                </p>
              </div>
            </label>

            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="radio"
                name="inpaint_method"
                checked={inpaintUseDedicatedModel}
                onChange={() => setInpaintUseDedicatedModel(true)}
                className="mt-1"
              />
              <div>
                <span className="text-white font-medium">Dedicated Inpaint Model (Legacy)</span>
                <p className="text-xs text-gray-400 mt-1">
                  Use dedicated 9-channel inpaint UNet if available.
                  This is the legacy SD/SDXL method. Only use this if you have a
                  specifically trained inpaint model (e.g., SD 1.5 Inpaint).
                  Regular models will fall back to mask blending.
                </p>
              </div>
            </label>
          </div>
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
