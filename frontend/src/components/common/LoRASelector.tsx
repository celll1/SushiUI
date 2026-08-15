"use client";

import { useState, useEffect, useRef } from "react";
import Card from "./Card";
import Button from "./Button";
import Slider from "./Slider";
import RangeSlider from "./RangeSlider";
import LayerWeightGraph from "./LayerWeightGraph";
import { LoRAConfig, LoRAInfo, LoRAListEntry, getLoras, getLoraInfo, archDisplayName } from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";

// Display order for LoRA architecture groups. "unknown" is a first-class
// value (files whose key structure doesn't match any recognized signature),
// so it always gets its own group rather than being hidden or merged.
const LORA_ARCH_GROUP_ORDER = ["sd15", "sdxl", "zimage", "flux2", "minimax_h3", "unknown"];

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
   * Used only to order the LoRA list -- the group matching this arch is
   * listed first/expanded. A LoRA whose detected arch does not match stays
   * selectable; a wrong or unrecognized arch sniff must never make a LoRA
   * unreachable.
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
  const { isBackendReady, modelLoaded } = useStartup();
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

  const addLoRA = () => {
    if (availableLoras.length === 0) return;

    const newLora: LoRAConfig = {
      path: availableLoras[0].path,
      strength: 1.0,
      apply_to_text_encoder: true,
      apply_to_unet: true,
      unet_layer_weights: {},
      step_range: [0, 1000],
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
