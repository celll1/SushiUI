"use client";

import { useState } from "react";
import Card from "./Card";
import ModelSelector from "./ModelSelector";
import VisionEncoderSelector from "./VisionEncoderSelector";
import VaeOverrideSelector from "./VaeOverrideSelector";
import TextEncoderOverrideSelector from "./TextEncoderOverrideSelector";
import Select from "./Select";
import { useStartup } from "@/contexts/StartupContext";

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
  storageKeyPrefix = "model_load",
}: ModelLoadSectionProps) {
  const { modelInfo, refreshModelInfo } = useStartup();
  const modelType = modelInfo?.type;
  const [selectedVaeKind, setSelectedVaeKind] = useState<string | null>(null);
  const isPidDecoder = selectedVaeKind === "pid_decoder";

  // TE override is sound only for SD1.5/SDXL server-side; disable (do not
  // block) for other archs as a cosmetic hint. Unknown type stays enabled.
  const teDisabled = !!modelType && modelType !== "sd15" && modelType !== "sdxl";

  const handleModelLoad = async (mi: any) => {
    // Keep the shared model-info source in sync on every (re)load.
    await refreshModelInfo();
    onModelLoad?.(mi);
  };

  return (
    <>
      <ModelSelector onModelLoad={handleModelLoad} />

      {showVisionEncoder && modelType !== "flux2" && onVisionEncoderChange && (
        <VisionEncoderSelector
          value={visionEncoderPath ?? null}
          onChange={onVisionEncoderChange}
        />
      )}

      <Card
        title="Component overrides"
        collapsible={true}
        defaultCollapsed={true}
        storageKey={`${storageKeyPrefix}_overrides_collapsed`}
      >
        <div className="space-y-3">
          <VaeOverrideSelector
            value={vaePath}
            onChange={onVaePathChange}
            onKindChange={setSelectedVaeKind}
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
            </div>
          )}
          <TextEncoderOverrideSelector
            value={textEncoderPath}
            onChange={onTextEncoderChange}
            disabled={teDisabled}
          />
        </div>
      </Card>
    </>
  );
}
