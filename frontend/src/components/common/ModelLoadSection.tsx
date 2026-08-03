"use client";

import { useState, useEffect } from "react";
import Card from "./Card";
import ModelSelector from "./ModelSelector";
import VisionEncoderSelector from "./VisionEncoderSelector";
import VaeOverrideSelector from "./VaeOverrideSelector";
import QuantizedExportSection from "./QuantizedExportSection";
import TextEncoderOverrideSelector from "./TextEncoderOverrideSelector";
import Select from "./Select";
import NumberInput from "./NumberInput";
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
  // PiD large-output (>4096px) decode controls: default = tiled true
  // super-resolution; pidFastLargeDecode = true switches to a faster
  // cap+bicubic path (lower quality).
  pidTileNative?: number;
  onPidTileNativeChange?: (value: number) => void;
  pidTileOverlapRatio?: number;
  onPidTileOverlapRatioChange?: (value: number) => void;
  pidFastLargeDecode?: boolean;
  onPidFastLargeDecodeChange?: (value: boolean) => void;

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
  pidTileNative = 512,
  onPidTileNativeChange,
  pidTileOverlapRatio = 0.25,
  onPidTileOverlapRatioChange,
  pidFastLargeDecode = false,
  onPidFastLargeDecodeChange,
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

  // Concise summary of the active component overrides, shown in the Card header
  // even while collapsed so a set override is visible at a glance.
  const _basename = (p?: string | null): string | null => {
    if (!p) return null;
    const parts = p.split(/[\\/]/).filter(Boolean);
    return parts.length ? parts[parts.length - 1] : p;
  };
  const _vaeName = _basename(vaePath);
  // Recognize a PiD checkpoint from the path too, so its flags show in the
  // collapsed summary even before VaeOverrideSelector has mounted to report kind.
  const _looksPid = isPidDecoder || (!!_vaeName && /^PiD_.*\.pth$/i.test(_vaeName));
  const overrideSummary: string[] = [];
  if (_vaeName) {
    let v = `VAE: ${_vaeName}`;
    if (_looksPid) {
      const flags: string[] = [];
      if (pidSrOutput && pidSrOutput !== "4x") flags.push(pidSrOutput);
      if (pidUseGemma) flags.push("Gemma");
      if (pidLowVram) flags.push("low-VRAM");
      if (pidFastLargeDecode) flags.push("fast large-decode");
      if (flags.length) v += ` (${flags.join(", ")})`;
    }
    overrideSummary.push(v);
  }
  if (showVE && visionEncoderPath) overrideSummary.push(`Vision: ${_basename(visionEncoderPath)}`);
  if (textEncoderPath) overrideSummary.push(`TE: ${_basename(textEncoderPath)}`);
  const collapsedPreview =
    overrideSummary.length > 0 ? (
      <p className="text-xs text-gray-400 break-words">{overrideSummary.join(" · ")}</p>
    ) : undefined;

  return (
    <>
      <ModelSelector onModelLoad={handleModelLoad} />

      {/* Renders itself only when the loaded transformer owns quantized
          Linear layers (checkpoint-loaded or converted in place this session). */}
      <QuantizedExportSection
        arch={modelType ?? null}
        storageKeyPrefix={storageKeyPrefix}
      />

      <Card
        title="Component overrides"
        collapsible={true}
        defaultCollapsed={true}
        collapsedPreview={collapsedPreview}
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
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={pidFastLargeDecode}
                  onChange={(e) => onPidFastLargeDecodeChange?.(e.target.checked)}
                  className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-2 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-300">PiD: fast large-decode (lower quality, cap+bicubic)</span>
              </label>
              <p className="text-xs text-gray-500">
                Default off = tiled true super-resolution for &gt;4096px outputs; on = faster but softer.
              </p>
              <details className="text-xs text-gray-500">
                <summary className="cursor-pointer select-none">PiD large-decode tiling (advanced)</summary>
                <div className="flex items-center gap-3 mt-2">
                  <label className="flex items-center gap-2">
                    <span>Tile size</span>
                    <NumberInput
                      value={pidTileNative}
                      onCommit={(v) => onPidTileNativeChange?.(v)}
                      min={64}
                      max={2048}
                      step={64}
                      parse="int"
                      className="w-20"
                    />
                  </label>
                  <label className="flex items-center gap-2">
                    <span>Overlap ratio</span>
                    <NumberInput
                      value={pidTileOverlapRatio}
                      onCommit={(v) => onPidTileOverlapRatioChange?.(v)}
                      min={0}
                      max={0.9}
                      step={0.05}
                      parse="float"
                      className="w-20"
                    />
                  </label>
                </div>
              </details>
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
