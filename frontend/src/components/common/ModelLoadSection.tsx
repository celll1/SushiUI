"use client";

import { useState, useEffect } from "react";
import Card from "./Card";
import ModelSelector from "./ModelSelector";
import VisionEncoderSelector from "./VisionEncoderSelector";
import VaeOverrideSelector from "./VaeOverrideSelector";
import TextEncoderOverrideSelector from "./TextEncoderOverrideSelector";
import Select from "./Select";
import { useStartup } from "@/contexts/StartupContext";

// Fallback arch -> latent_channels map, used only when the loaded model's
// latent_channels isn't exposed by GET /models/current (pipeline_manager's
// current_model_info carries "type" but not latent_channels today). Mirrors
// backend/core/models/components/wiring.py's per-arch ComponentWiringSpec
// (kept minimal + commented, update alongside wiring.py if it changes).
const ARCH_LATENT_CHANNELS: Record<string, number> = {
  sd15: 4,
  sdxl: 4,
  zimage: 16,
  anima: 16,
  lens: 32,
  ideogram4: 32,
  minit2i: 0, // pixel-space, no VAE
  krea2: 16,
  flux2: 32,
  ltx2: 128,
  acestep: 64,
};

interface ModelLoadSectionProps {
  // Panel callback fired after a model (re)load (receives model_info).
  onModelLoad?: (modelInfo: any) => void;

  // Vision encoder override (SDXL/SD1.5 reference conditioning).
  showVisionEncoder?: boolean;
  visionEncoderPath?: string | null;
  onVisionEncoderChange?: (path: string | null) => void;

  // VAE override (empty = model default).
  vaePath: string | null;
  onVaePathChange: (path: string | null) => void;

  // Text encoder override (empty = model default).
  textEncoderPath: string | null;
  onTextEncoderChange: (path: string | null) => void;

  // PiD (Pixel Diffusion Decoder) options — only relevant when vaePath
  // selects a PiD checkpoint (VaeOverrideSelector reports kind="pid_decoder");
  // hidden otherwise.
  pidSrOutput?: string;
  onPidSrOutputChange?: (value: string) => void;
  pidUseGemma?: boolean;
  onPidUseGemmaChange?: (value: boolean) => void;
  pidLowVram?: boolean;
  onPidLowVramChange?: (value: boolean) => void;

  // Storage key suffix so per-panel collapse state stays independent.
  storageKeyPrefix?: string;
}

// Shared model-load section: model selection + component overrides.
// Rendered by every generation panel except Upscale. modelInfo/isVideo come
// from StartupContext (the single source of truth).
export default function ModelLoadSection({
  onModelLoad,
  showVisionEncoder = true,
  visionEncoderPath = null,
  onVisionEncoderChange,
  vaePath,
  onVaePathChange,
  textEncoderPath,
  onTextEncoderChange,
  pidSrOutput = "4x",
  onPidSrOutputChange,
  pidUseGemma = false,
  onPidUseGemmaChange,
  pidLowVram = false,
  onPidLowVramChange,
  storageKeyPrefix = "model_load",
}: ModelLoadSectionProps) {
  const { modelInfo, refreshModelInfo } = useStartup();
  const modelType = modelInfo?.type;
  const [selectedVaeKind, setSelectedVaeKind] = useState<string | null>(null);
  const isPidDecoder = selectedVaeKind === "pid_decoder";

  // TE override is sound only for SD1.5/SDXL server-side; disable (do not
  // block) for other archs as a cosmetic hint. Unknown type stays enabled.
  const teDisabled = !!modelType && modelType !== "sd15" && modelType !== "sdxl";

  // "Show only compatible with loaded model" toggle for the VAE/TE override
  // lists (default checked). Persisted per-panel like the Card's own
  // collapse state.
  const compatibleOnlyStorageKey = `${storageKeyPrefix}_compatible_only`;
  const [compatibleOnly, setCompatibleOnly] = useState(true);
  const [compatibleOnlyMounted, setCompatibleOnlyMounted] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = localStorage.getItem(compatibleOnlyStorageKey);
    if (saved !== null) setCompatibleOnly(saved === "true");
    setCompatibleOnlyMounted(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compatibleOnlyStorageKey]);
  useEffect(() => {
    if (!compatibleOnlyMounted || typeof window === "undefined") return;
    localStorage.setItem(compatibleOnlyStorageKey, compatibleOnly.toString());
  }, [compatibleOnly, compatibleOnlyStorageKey, compatibleOnlyMounted]);

  const loadedArch = modelType ?? null;
  // modelInfo doesn't carry latent_channels today (see ARCH_LATENT_CHANNELS
  // comment above) — fall back to the arch map when it's missing.
  const loadedLatentChannels =
    (modelInfo?.latent_channels as number | undefined) ??
    (loadedArch != null ? ARCH_LATENT_CHANNELS[loadedArch] : undefined) ??
    null;

  const handleModelLoad = async (mi: any) => {
    // Keep the shared model-info source in sync on every (re)load.
    await refreshModelInfo();
    onModelLoad?.(mi);
  };

  const showVE = showVisionEncoder && modelType !== "flux2" && !!onVisionEncoderChange;

  return (
    <>
      <ModelSelector onModelLoad={handleModelLoad} />

      <Card
        title="Component overrides"
        collapsible={true}
        defaultCollapsed={true}
        storageKey={`${storageKeyPrefix}_overrides_collapsed`}
      >
        <div className="space-y-3">
          {showVE && (
            <VisionEncoderSelector
              value={visionEncoderPath ?? null}
              onChange={onVisionEncoderChange!}
            />
          )}

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={compatibleOnly}
              onChange={(e) => setCompatibleOnly(e.target.checked)}
              className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-2 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-300">
              Show only VAE / text encoder compatible with loaded model
            </span>
          </label>

          <VaeOverrideSelector
            value={vaePath}
            onChange={onVaePathChange}
            onKindChange={setSelectedVaeKind}
            compatibleOnly={compatibleOnly}
            loadedArch={loadedArch}
            loadedLatentChannels={loadedLatentChannels}
          />
          {isPidDecoder && (
            <div className="space-y-2 pl-2 border-l-2 border-gray-700">
              <Select
                label="PiD Output Size"
                value={pidSrOutput}
                onChange={(e) => onPidSrOutputChange?.(e.target.value)}
                options={[
                  { value: "4x", label: "4x super-resolution (native)" },
                  { value: "original", label: "Original size (downscale)" },
                ]}
              />
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={pidUseGemma}
                  onChange={(e) => onPidUseGemmaChange?.(e.target.checked)}
                  className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-2 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-300">PiD: use prompt captioning (Gemma)</span>
              </label>
              <p className="text-xs text-gray-500">
                Loads a text encoder (Gemma) on first use to condition the PiD decode on your prompt. Opt-in.
              </p>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={pidLowVram}
                  onChange={(e) => onPidLowVramChange?.(e.target.checked)}
                  className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-2 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-300">PiD: low VRAM decode</span>
              </label>
              <p className="text-xs text-gray-500">
                Row-chunks the PiD decode to cut peak VRAM at high resolution (~42% less at 4096px, measured). Not bit-identical to the unchunked decode, but visually indistinguishable.
              </p>
            </div>
          )}
          <TextEncoderOverrideSelector
            value={textEncoderPath}
            onChange={onTextEncoderChange}
            disabled={teDisabled}
            compatibleOnly={compatibleOnly}
            loadedArch={loadedArch}
          />
        </div>
      </Card>
    </>
  );
}
