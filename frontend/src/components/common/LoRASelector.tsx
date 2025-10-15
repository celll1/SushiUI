"use client";

import { useState, useEffect } from "react";
import Card from "./Card";
import Button from "./Button";
import Slider from "./Slider";
import RangeSlider from "./RangeSlider";
import LayerWeightGraph from "./LayerWeightGraph";
import { LoRAConfig, LoRAInfo, getLoras, getLoraInfo } from "@/utils/api";

interface LoRASelectorProps {
  value: LoRAConfig[];
  onChange: (loras: LoRAConfig[]) => void;
  disabled?: boolean;
}

interface LoRALayerWeightsProps {
  loraPath: string;
  weights: { [layerName: string]: number };
  onChange: (weights: { [layerName: string]: number }) => void;
  disabled?: boolean;
  loadLoraInfo: (loraPath: string) => Promise<LoRAInfo | null>;
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

export default function LoRASelector({ value, onChange, disabled = false }: LoRASelectorProps) {
  const [availableLoras, setAvailableLoras] = useState<Array<{ path: string; name: string }>>([]);
  const [loraInfoCache, setLoraInfoCache] = useState<Map<string, LoRAInfo>>(new Map());
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    loadAvailableLoras();
  }, []);

  const loadAvailableLoras = async () => {
    try {
      const response = await getLoras();
      setAvailableLoras(response.loras);
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
      setLoraInfoCache((prev) => new Map(prev).set(loraPath, info));
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

  return (
    <Card
      title={`LoRA (${value.length})`}
      collapsible={true}
      collapsed={!isExpanded}
      onToggle={() => setIsExpanded(!isExpanded)}
      preview={
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
                {availableLoras.map((availLora) => (
                  <option key={availLora.path} value={availLora.path}>
                    {availLora.name}
                  </option>
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

            {/* 2-Column Layout: Settings on left, Graph on right */}
            <div className="grid grid-cols-2 gap-4">
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
