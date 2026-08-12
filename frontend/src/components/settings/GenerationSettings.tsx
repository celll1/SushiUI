"use client";

import { useState, useEffect } from "react";
import Button from "@/components/common/Button";
import NumberInput from "@/components/common/NumberInput";
import { useStartup } from "@/contexts/StartupContext";

interface GenerationSettingsData {
  inpaint_use_dedicated_model: boolean;
  video_frame_slider_max?: number | null;
}

export default function GenerationSettings() {
  const { generationDefaults } = useStartup();
  // backend/api/param_defaults.py VIDEO_GEN_DEFAULTS.video_frame_slider_max_seed
  // (fetched via generationDefaults.txt2vid, which resolves it like every other
  // video-only key). 241 mirrors that same value as the pre-fetch fallback, per
  // this repo's convention for DEFAULT_PARAMS-style fallbacks (see param_defaults.py's
  // comment on that constant for why 241 specifically).
  const videoFrameSliderMaxSeed =
    (generationDefaults?.txt2vid?.video_frame_slider_max_seed as number | undefined) ?? 241;
  const [inpaintUseDedicatedModel, setInpaintUseDedicatedModel] = useState(false);
  // Upper bound for the video frame-count SLIDER TRACK (VideoFrameCountSlider),
  // not a value cap -- see the field's help text below for the exact
  // distinction. `null` = unset = the slider's own built-in track reach.
  const [videoFrameSliderMax, setVideoFrameSliderMax] = useState<number | null>(null);
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
      setVideoFrameSliderMax(data.video_frame_slider_max ?? null);
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
          video_frame_slider_max: videoFrameSliderMax,
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

        <div className="p-4 bg-gray-800 rounded-lg">
          <h3 className="text-sm font-semibold mb-3 text-white">Video Frame Count Slider</h3>
          <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-2">
            <input
              type="checkbox"
              checked={videoFrameSliderMax != null}
              onChange={(e) => {
                setVideoFrameSliderMax(e.target.checked ? (videoFrameSliderMax ?? videoFrameSliderMaxSeed) : null);
              }}
              className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
            />
            Video Frame Count Slider Track Max
          </label>
          {videoFrameSliderMax != null && (
            <NumberInput
              label="Video Frame Count Slider Track Max"
              value={videoFrameSliderMax}
              onCommit={(v) => setVideoFrameSliderMax(v)}
              min={1}
              parse="int"
              className="w-28"
            />
          )}
          <p className="text-xs text-gray-400 mt-2">
            Sets how far the video frame-count slider&apos;s track reaches on an
            architecture that does not impose a hard per-request frame limit.
            The number box next to the slider is not bounded by this setting
            and always accepts a value above it. Unchecked uses the
            slider&apos;s own built-in track reach. The value is snapped onto
            the loaded architecture&apos;s frame grid where the slider is used.
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
