"use client";

import Card from "./Card";
import ModelSelector from "./ModelSelector";
import VisionEncoderSelector from "./VisionEncoderSelector";
import VaeOverrideSelector from "./VaeOverrideSelector";
import TextEncoderOverrideSelector from "./TextEncoderOverrideSelector";
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
  storageKeyPrefix = "model_load",
}: ModelLoadSectionProps) {
  const { modelInfo, refreshModelInfo } = useStartup();
  const modelType = modelInfo?.type;

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
          <VaeOverrideSelector value={vaePath} onChange={onVaePathChange} />
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
