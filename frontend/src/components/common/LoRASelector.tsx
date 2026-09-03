"use client";

import { useState, useEffect, useRef } from "react";
import Card from "./Card";
import Button from "./Button";
import Slider from "./Slider";
import RangeSlider from "./RangeSlider";
import LayerWeightGraph from "./LayerWeightGraph";
import {
  ADAPTER_TYPE_AUTO,
  AdapterTypeAssertion,
  ArchAdapterFamilies,
  DetectedAdapterType,
  LoRAConfig,
  LoRAInfo,
  LoRAListEntry,
  getLoras,
  getLoraInfo,
  archDisplayName,
} from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";

// Display spelling only; the VALUES come from the backend detector
// (GET /loras -> adapter_type). An unmapped value renders as itself, so a
// family added backend-side shows up instead of disappearing.
const ADAPTER_TYPE_LABELS: Record<string, string> = {
  lora: "LoRA",
  loha: "LoHa",
  lokr: "LoKr",
  dora: "DoRA",
  doha: "DoHa",
  dokr: "DoKr",
  unknown: "Unknown",
};

// Assertion choices offered under Advanced. "auto" is the default and stays
// first; the rest must MATCH the file (a mismatch is refused, not applied).
const ADAPTER_TYPE_ASSERTIONS: AdapterTypeAssertion[] = [
  "auto", "lora", "loha", "lokr", "dora", "doha", "dokr",
];

function adapterTypeLabel(value: string): string {
  return ADAPTER_TYPE_LABELS[value] || value;
}

// Display order for LoRA architecture groups. "unknown" is a first-class
// value (files whose key structure doesn't match any recognized signature),
// so it always gets its own group rather than being hidden or merged.
// Mirrors the architectures classify_lora_keys can return
// (backend/core/extensions/lora_manager.py). "unknown" stays last.
const LORA_ARCH_GROUP_ORDER = [
  "sd15", "sdxl", "zimage", "flux2", "anima", "lens", "ideogram4",
  "minit2i", "krea2", "sensenova", "ltx2", "minimax_h3", "acestep", "unknown",
];

interface LoRASelectorProps {
  value: LoRAConfig[];
  onChange: (loras: LoRAConfig[]) => void;
  disabled?: boolean;
  storageKey?: string;
  /**
   * When true, renders a reduced UI (LoRA name + single Strength slider +
   * enable/remove only) without the Text Encoder/U-Net split toggle or the
   * per-block LoRALayerWeights graph. Intended for modalities (e.g. audio
   * LoRAs) where the underlying pipeline applies a single uniform strength
   * and has no per-block/TE-vs-UNet concept.
   */
  simpleMode?: boolean;
  /**
   * Architecture of the currently loaded model (e.g. "sdxl", "minimax_h3").
   * Orders the LoRA list -- the group matching this arch is listed
   * first/expanded -- and selects the adapter-family capability entry used to
   * warn that a detected family would be refused. A LoRA whose detected arch
   * does not match stays selectable; a wrong or unrecognized arch sniff must
   * never make a LoRA unreachable.
   */
  loadedArch?: string | null;
  /**
   * Applies a LoRA's declared recommended settings (parsed from the file's
   * own metadata) to the generation params, like any ordinary user edit.
   * Returns the list of setting keys the current panel/modality has no
   * param for and therefore skipped (e.g. audio has no fbcache concept);
   * an empty array/undefined means everything was applied.
   */
  onApplyRecommended?: (settings: Record<string, unknown>) => string[] | void;
}

interface LoRALayerWeightsProps {
  loraPath: string;
  weights: { [layerName: string]: number };
  onChange: (weights: { [layerName: string]: number }) => void;
  disabled?: boolean;
  loadLoraInfo: (loraPath: string) => Promise<LoRAInfo | null>;
}

interface LoRAAdapterNoteProps {
  entry?: LoRAListEntry;
  asserted: AdapterTypeAssertion;
  onAssert: (adapter_type: AdapterTypeAssertion) => void;
  families?: ArchAdapterFamilies;
  disabled?: boolean;
}

// What the FILE is (detected by the backend), and -- separately -- whether the
// LOADED architecture can apply it. Two different questions, two different
// sources: `GET /loras`'s adapter_type and `adapter_families` of
// `GET /schema/arch-capabilities`. The reason shown for a family the
// architecture refuses is the backend's own sentence, so what the user reads
// here is what the request would answer with.
function LoRAAdapterNote({ entry, asserted, onAssert, families, disabled }: LoRAAdapterNoteProps) {
  const detected = (entry?.adapter_type ?? undefined) as DetectedAdapterType | undefined;
  if (!detected) return null;

  const unsupportedReason =
    detected !== "unknown" && families
      ? families.unsupported?.[detected as Exclude<DetectedAdapterType, "unknown">]
      : undefined;
  const invalidReason =
    entry?.adapter_state === "invalid" ? entry.adapter_state_reason : null;

  return (
    <div className="mt-1 space-y-1 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded bg-gray-700 px-1.5 py-0.5 text-gray-200">
          {adapterTypeLabel(detected)}
        </span>
        {entry?.adapter_format && entry.adapter_format !== "unknown" && (
          <span className="text-gray-500">{entry.adapter_format}</span>
        )}
        {entry?.adapter_rank != null && (
          <span className="text-gray-500">
            rank {entry.adapter_rank}
            {entry.adapter_alpha != null ? ` / alpha ${entry.adapter_alpha}` : ""}
          </span>
        )}
        {detected === "unknown" && (
          <span className="text-gray-500">
            detection could not name this file&apos;s adapter algebra
          </span>
        )}
        {asserted !== ADAPTER_TYPE_AUTO && (
          <span className="text-gray-400">asserted: {adapterTypeLabel(asserted)}</span>
        )}
      </div>

      {invalidReason && <div className="text-amber-400">{invalidReason}</div>}

      {unsupportedReason && (
        <div className="text-amber-400">
          The loaded model does not accept this adapter family -- {unsupportedReason}
        </div>
      )}

      <details>
        <summary className="cursor-pointer text-gray-500">Advanced</summary>
        <label className="mt-1 flex flex-wrap items-center gap-2 text-gray-400">
          <span>Adapter type</span>
          <select
            value={asserted}
            onChange={(e) => onAssert(e.target.value as AdapterTypeAssertion)}
            disabled={disabled}
            className="bg-gray-700 text-white px-2 py-1 rounded text-xs"
          >
            {ADAPTER_TYPE_ASSERTIONS.map((value) => (
              <option key={value} value={value}>
                {value === ADAPTER_TYPE_AUTO ? "Auto (detect)" : adapterTypeLabel(value)}
              </option>
            ))}
          </select>
          <span className="text-gray-500">
            An assertion about the file, not a conversion: any value but Auto
            must match what was detected, or the request is refused.
          </span>
        </label>
      </details>
    </div>
  );
}

interface LoRARecommendedNoteProps {
  loraPath: string;
  loadLoraInfo: (loraPath: string) => Promise<LoRAInfo | null>;
  onApplyRecommended?: (settings: Record<string, unknown>) => string[] | void;
}

// Shows a LoRA's own declared step-distillation recommendation (from the
// file's safetensors metadata) and an explicit, one-click control to apply
// it to the generation params. Renders nothing when the LoRA declares no
// recommendation -- never a guessed/invented one.
function LoRARecommendedNote({ loraPath, loadLoraInfo, onApplyRecommended }: LoRARecommendedNoteProps) {
  const [recommended, setRecommended] = useState<LoRAInfo["recommended"] | undefined>(undefined);
  const [applyResult, setApplyResult] = useState<"applied" | string[] | null>(null);

  useEffect(() => {
    setApplyResult(null);
    let cancelled = false;
    loadLoraInfo(loraPath).then((info) => {
      if (!cancelled) setRecommended(info?.recommended ?? null);
    });
    return () => {
      cancelled = true;
    };
  }, [loraPath]);

  if (!recommended) return null;

  return (
    <div className="mt-1 flex flex-wrap items-center gap-2 rounded bg-gray-900/60 px-2 py-1.5 text-xs text-gray-400">
      <span>
        This LoRA&apos;s file metadata declares {recommended.num_inference_steps} inference steps
        {!recommended.fbcache_enable && !recommended.spectrum_enable
          ? ", First Block Cache off, and Spectrum forecasting off"
          : ""}
        {" "}(source: {recommended.source}).
      </span>
      {onApplyRecommended && (
        <Button
          onClick={() => {
            const skipped = onApplyRecommended({
              num_inference_steps: recommended.num_inference_steps,
              fbcache_enable: recommended.fbcache_enable,
              spectrum_enable: recommended.spectrum_enable,
            });
            setApplyResult(skipped && skipped.length > 0 ? skipped : "applied");
          }}
          variant="secondary"
          size="sm"
        >
          {applyResult ? "Applied" : "Apply to params"}
        </Button>
      )}
      {Array.isArray(applyResult) && (
        <span className="text-amber-400">
          This panel has no param for: {applyResult.join(", ")} -- not applied.
        </span>
      )}
    </div>
  );
}

function LoRALayerWeights({ loraPath, weights, onChange, disabled, loadLoraInfo }: LoRALayerWeightsProps) {
  const [layers, setLayers] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadLayers();
  }, [loraPath]);

  const loadLayers = async () => {
    setIsLoading(true);
    try {
      const info = await loadLoraInfo(loraPath);
      if (info && info.layers) {
        setLayers(info.layers);
      }
    } catch (error) {
      console.error("Failed to load layers:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="p-4 bg-gray-800 rounded text-gray-400 text-center text-sm">
        Loading layer information...
      </div>
    );
  }

  if (layers.length === 0) {
    return (
      <div className="p-4 bg-gray-800 rounded text-gray-400 text-center text-sm">
        No layer information available
      </div>
    );
  }

  return (
    <LayerWeightGraph
      layers={layers}
      weights={weights}
      onChange={onChange}
      disabled={disabled}
    />
  );
}

export default function LoRASelector({ value, onChange, disabled = false, storageKey = "lora_panel_collapsed", simpleMode = false, loadedArch = null, onApplyRecommended }: LoRASelectorProps) {
  const { isBackendReady, modelLoaded, archCapabilities, generationDefaults } = useStartup();
  const [availableLoras, setAvailableLoras] = useState<Array<LoRAListEntry>>([]);
  const [loraInfoCache, setLoraInfoCache] = useState<Map<string, LoRAInfo>>(new Map());
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    if (!isBackendReady) {
      return () => {
        mountedRef.current = false;
      };
    }
    // A LoRA directory scan is not needed for the first paint or state restore.
    // Start it only after current-model confirmation has settled.
    const timer = window.setTimeout(() => { void loadAvailableLoras(); }, 1000);
    return () => {
      mountedRef.current = false;
      window.clearTimeout(timer);
    };
  }, [isBackendReady]);

  const loadAvailableLoras = async () => {
    try {
      const response = await getLoras();
      if (mountedRef.current) setAvailableLoras(response.loras);
    } catch (error) {
      console.error("Failed to load LoRAs:", error);
    }
  };

  const loadLoraInfo = async (loraPath: string): Promise<LoRAInfo | null> => {
    // Check cache first
    if (loraInfoCache.has(loraPath)) {
      return loraInfoCache.get(loraPath)!;
    }

    try {
      const info = await getLoraInfo(loraPath);
      if (mountedRef.current) setLoraInfoCache((prev) => new Map(prev).set(loraPath, info));
      return info;
    } catch (error) {
      console.error("Failed to load LoRA info:", error);
      return null;
    }
  };

  // Backend LORA_ITEM_DEFAULTS; the literal is the not-yet-started fallback.
  const adapterTypeDefault: AdapterTypeAssertion =
    (generationDefaults?.lora_item?.adapter_type as AdapterTypeAssertion | undefined) ??
    ADAPTER_TYPE_AUTO;
  const adapterFamilies = loadedArch
    ? archCapabilities?.adapter_families?.[loadedArch]
    : undefined;
  const detectionByPath = new Map(availableLoras.map((e) => [e.path, e]));

  const addLoRA = () => {
    if (availableLoras.length === 0) return;

    const newLora: LoRAConfig = {
      path: availableLoras[0].path,
      strength: 1.0,
      apply_to_text_encoder: true,
      apply_to_unet: true,
      unet_layer_weights: {},
      step_range: [0, 1000],
      adapter_type: adapterTypeDefault,
    };

    onChange([...value, newLora]);
  };

  const removeLora = (index: number) => {
    const newLoras = value.filter((_, i) => i !== index);
    onChange(newLoras);
  };

  const updateLora = (index: number, updates: Partial<LoRAConfig>) => {
    const newLoras = value.map((lora, i) =>
      i === index ? { ...lora, ...updates } : lora
    );
    onChange(newLoras);
  };

  // Group the flat LoRA list by detected arch (grouping, never filtering --
  // a wrong/unknown arch sniff must not make a LoRA unreachable). The group
  // matching the loaded model's architecture, if any, is listed first.
  const archGroups = new Map<string, LoRAListEntry[]>();
  for (const lora of availableLoras) {
    const arch = lora.arch || "unknown";
    if (!archGroups.has(arch)) archGroups.set(arch, []);
    archGroups.get(arch)!.push(lora);
  }
  const orderedArchKeys = Array.from(archGroups.keys()).sort((a, b) => {
    if (loadedArch) {
      if (a === loadedArch && b !== loadedArch) return -1;
      if (b === loadedArch && a !== loadedArch) return 1;
    }
    const ia = LORA_ARCH_GROUP_ORDER.indexOf(a);
    const ib = LORA_ARCH_GROUP_ORDER.indexOf(b);
    return (ia === -1 ? LORA_ARCH_GROUP_ORDER.length : ia) - (ib === -1 ? LORA_ARCH_GROUP_ORDER.length : ib);
  });

  return (
    <Card
      title={`LoRA (${value.length})`}
      collapsible={true}
      defaultCollapsed={false}
      storageKey={storageKey}
      collapsedPreview={
        value.length > 0 ? (
          <div className="text-xs text-gray-400 truncate">
            {value.map((l) => l.path.split("/").pop()).join(", ")}
          </div>
        ) : undefined
      }
    >
      <div className="space-y-4">
        {value.map((lora, index) => (
          <div key={index} className="p-3 bg-gray-800 rounded-lg">
            {/* LoRA Selection */}
            <div className="flex gap-2 mb-3">
              <select
                value={lora.path}
                onChange={(e) => updateLora(index, { path: e.target.value })}
                disabled={disabled}
                className="flex-1 bg-gray-700 text-white px-3 py-2 rounded text-sm"
              >
                {orderedArchKeys.map((archKey) => (
                  <optgroup
                    key={archKey}
                    label={`${archKey === "unknown" ? "Unknown" : (archDisplayName(archKey) || archKey)}${archKey === loadedArch ? " (loaded)" : ""}`}
                  >
                    {archGroups.get(archKey)!.map((availLora) => (
                      <option key={availLora.path} value={availLora.path}>
                        {availLora.name}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
              <Button
                onClick={() => removeLora(index)}
                disabled={disabled}
                variant="secondary"
                size="sm"
              >
                Remove
              </Button>
            </div>

            <LoRAAdapterNote
              entry={detectionByPath.get(lora.path)}
              asserted={lora.adapter_type ?? adapterTypeDefault}
              onAssert={(adapter_type) => updateLora(index, { adapter_type })}
              families={adapterFamilies}
              disabled={disabled}
            />

            <LoRARecommendedNote
              loraPath={lora.path}
              loadLoraInfo={loadLoraInfo}
              onApplyRecommended={onApplyRecommended}
            />

            {simpleMode ? (
              /* Simple Mode: Single uniform strength only (no TE/U-Net split, no block graph) */
              <div className="space-y-3">
                <Slider
                  label="Strength"
                  min={-2}
                  max={2}
                  step={0.05}
                  value={lora.strength}
                  onChange={(e) => updateLora(index, { strength: parseFloat(e.target.value) })}
                  disabled={disabled}
                />
              </div>
            ) : (
              /* 2-Column Layout: Settings on left, Graph on right */
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Left Column: Settings */}
                <div className="space-y-3">
                  {/* Strength Slider */}
                  <Slider
                    label="Strength"
                    min={-2}
                    max={2}
                    step={0.05}
                    value={lora.strength}
                    onChange={(e) => updateLora(index, { strength: parseFloat(e.target.value) })}
                    disabled={disabled}
                  />

                  {/* Text Encoder / U-Net Toggles */}
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 text-sm cursor-pointer">
                      <input
                        type="checkbox"
                        checked={lora.apply_to_text_encoder}
                        onChange={(e) =>
                          updateLora(index, { apply_to_text_encoder: e.target.checked })
                        }
                        disabled={disabled}
                        className="w-4 h-4"
                      />
                      <span className="text-gray-300">Text Encoder</span>
                    </label>
                    <label className="flex items-center gap-2 text-sm cursor-pointer">
                      <input
                        type="checkbox"
                        checked={lora.apply_to_unet}
                        onChange={(e) =>
                          updateLora(index, { apply_to_unet: e.target.checked })
                        }
                        disabled={disabled}
                        className="w-4 h-4"
                      />
                      <span className="text-gray-300">U-Net</span>
                    </label>
                  </div>

                  {/* Step Range */}
                  <RangeSlider
                    label="Step Range"
                    min={0}
                    max={1000}
                    step={10}
                    value={lora.step_range}
                    onChange={(step_range) => updateLora(index, { step_range })}
                    disabled={disabled}
                  />
                </div>

                {/* Right Column: Block Weights Graph */}
                <div>
                  {lora.apply_to_unet && (
                    <LoRALayerWeights
                      loraPath={lora.path}
                      weights={lora.unet_layer_weights}
                      onChange={(unet_layer_weights) => updateLora(index, { unet_layer_weights })}
                      disabled={disabled}
                      loadLoraInfo={loadLoraInfo}
                    />
                  )}
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Add LoRA Button */}
        <Button
          onClick={addLoRA}
          disabled={disabled || availableLoras.length === 0}
          variant="secondary"
          className="w-full"
        >
          + Add LoRA
        </Button>

        {availableLoras.length === 0 && (
          <div className="text-xs text-gray-500 text-center">
            No LoRA files found in lora directory
          </div>
        )}
      </div>
    </Card>
  );
}
