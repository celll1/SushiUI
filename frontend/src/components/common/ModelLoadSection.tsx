"use client";

import { useCallback, useState, useEffect, type KeyboardEvent } from "react";
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

type ModelWorkspaceTab = "model" | "components" | "quantization";

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
  const { modelInfo, modelInfoVersion, refreshModelInfo } = useStartup();
  const modelType = modelInfo?.type;
  const [selectedVaeKind, setSelectedVaeKind] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ModelWorkspaceTab>("model");
  const [quantizationAvailable, setQuantizationAvailable] = useState(false);
  const [quantizationResolved, setQuantizationResolved] = useState(false);
  const [tabStateMounted, setTabStateMounted] = useState(false);
  const [modelLoadRevision, setModelLoadRevision] = useState(0);
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
  useEffect(() => {
    const saved = localStorage.getItem(`${storageKeyPrefix}_workspace_tab`);
    if (saved === "model" || saved === "components" || saved === "quantization") {
      setActiveTab(saved);
    }
    setTabStateMounted(true);
  }, [storageKeyPrefix]);
  useEffect(() => {
    if (!tabStateMounted) return;
    localStorage.setItem(`${storageKeyPrefix}_workspace_tab`, activeTab);
  }, [activeTab, storageKeyPrefix, tabStateMounted]);
  useEffect(() => {
    if (activeTab === "quantization" && quantizationResolved && !quantizationAvailable) {
      setActiveTab("model");
    }
  }, [activeTab, quantizationAvailable, quantizationResolved]);

  const loadedArch = modelType ?? null;
  // modelInfo doesn't carry latent_channels today (see ARCH_LATENT_CHANNELS
  // comment above) — fall back to the arch map when it's missing.
  const loadedLatentChannels =
    (modelInfo?.latent_channels as number | undefined) ??
    (loadedArch != null ? ARCH_LATENT_CHANNELS[loadedArch] : undefined) ??
    null;

  const handleModelLoad = async (mi: any) => {
    // Keep the shared model-info source in sync on every (re)load.
    setModelLoadRevision((revision) => revision + 1);
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
  const overrideSummaryText = overrideSummary.length > 0
    ? overrideSummary.join(" · ")
    : "No component overrides";
  const showQuantizationTab = quantizationAvailable
    || (activeTab === "quantization" && !quantizationResolved);
  const availableTabs: ModelWorkspaceTab[] = showQuantizationTab
    ? ["model", "components", "quantization"]
    : ["model", "components"];
  const tabId = (tab: ModelWorkspaceTab) => `${storageKeyPrefix}_${tab}_tab`;
  const panelId = (tab: ModelWorkspaceTab) => `${storageKeyPrefix}_${tab}_panel`;
  const handleTabKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    currentTab: ModelWorkspaceTab,
  ) => {
    const currentIndex = availableTabs.indexOf(currentTab);
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % availableTabs.length;
    if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + availableTabs.length) % availableTabs.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = availableTabs.length - 1;
    if (nextIndex === null) return;

    event.preventDefault();
    const nextTab = availableTabs[nextIndex];
    setActiveTab(nextTab);
    requestAnimationFrame(() => document.getElementById(tabId(nextTab))?.focus());
  };
  const handleQuantizationAvailability = useCallback((available: boolean, resolved: boolean) => {
    setQuantizationAvailable(available);
    setQuantizationResolved(resolved);
  }, []);

  return (
    <Card>
      <div className="app-tabs" role="tablist" aria-label="Model workspace">
        <button
          type="button"
          id={tabId("model")}
          role="tab"
          aria-selected={activeTab === "model"}
          aria-controls={panelId("model")}
          tabIndex={activeTab === "model" ? 0 : -1}
          onClick={() => setActiveTab("model")}
          onKeyDown={(event) => handleTabKeyDown(event, "model")}
          className={`app-tab ${activeTab === "model" ? "app-tab-active" : ""}`}
        >
          Model
        </button>
        <button
          type="button"
          id={tabId("components")}
          role="tab"
          aria-selected={activeTab === "components"}
          aria-controls={panelId("components")}
          tabIndex={activeTab === "components" ? 0 : -1}
          onClick={() => setActiveTab("components")}
          onKeyDown={(event) => handleTabKeyDown(event, "components")}
          className={`app-tab flex items-center gap-1.5 ${activeTab === "components" ? "app-tab-active" : ""}`}
          title={overrideSummaryText}
        >
          Components
          {overrideSummary.length > 0 && (
            <span className="rounded-full bg-violet-500/20 px-1.5 text-[9px] text-violet-300">
              {overrideSummary.length}
            </span>
          )}
        </button>
        {showQuantizationTab && (
          <button
            type="button"
            id={tabId("quantization")}
            role="tab"
            aria-selected={activeTab === "quantization"}
            aria-controls={panelId("quantization")}
            tabIndex={activeTab === "quantization" ? 0 : -1}
            onClick={() => setActiveTab("quantization")}
            onKeyDown={(event) => handleTabKeyDown(event, "quantization")}
            className={`app-tab ${activeTab === "quantization" ? "app-tab-active" : ""}`}
          >
            Quantization
          </button>
        )}
      </div>

      <div className="h-[clamp(7rem,15vh,8.5rem)] overflow-y-auto overscroll-contain pr-1">
        <div
          id={panelId("model")}
          role="tabpanel"
          aria-labelledby={tabId("model")}
          className={activeTab === "model" ? "" : "hidden"}
        >
          <ModelSelector embedded onModelLoad={handleModelLoad} />
        </div>

        <div
          id={panelId("components")}
          role="tabpanel"
          aria-labelledby={tabId("components")}
          className={activeTab === "components" ? "space-y-2" : "hidden"}
        >
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

        <div
          id={panelId("quantization")}
          role="tabpanel"
          aria-labelledby={tabId("quantization")}
          className={activeTab === "quantization" ? "" : "hidden"}
        >
          {!quantizationResolved && (
            <p className="text-xs text-gray-500">Checking quantization status…</p>
          )}
          <QuantizedExportSection
            embedded
            arch={modelType ?? null}
            modelInfoVersion={modelInfoVersion}
            modelLoadRevision={modelLoadRevision}
            storageKeyPrefix={storageKeyPrefix}
            onAvailabilityChange={handleQuantizationAvailability}
          />
        </div>
      </div>
    </Card>
  );
}
