"use client";

import { useState, useEffect } from "react";
import Card from "./Card";
import Button from "./Button";
import Slider from "./Slider";
import RangeSlider from "./RangeSlider";
import { LoRAConfig, getLoras } from "@/utils/api";

interface LoRASelectorProps {
  value: LoRAConfig[];
  onChange: (loras: LoRAConfig[]) => void;
  disabled?: boolean;
}

export default function LoRASelector({ value, onChange, disabled = false }: LoRASelectorProps) {
  const [availableLoras, setAvailableLoras] = useState<Array<{ path: string; name: string }>>([]);
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

  const addLoRA = () => {
    if (availableLoras.length === 0) return;

    const newLora: LoRAConfig = {
      path: availableLoras[0].path,
      strength: 1.0,
      apply_to_text_encoder: true,
      apply_to_unet: true,
      unet_layer_weights: {
        down: 1.0,
        mid: 1.0,
        up: 1.0,
      },
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
          <div key={index} className="p-3 bg-gray-800 rounded-lg space-y-3">
            {/* LoRA Selection */}
            <div className="flex gap-2">
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

            {/* Strength Slider */}
            <Slider
              label="Strength"
              min={-2}
              max={2}
              step={0.05}
              value={lora.strength}
              onChange={(strength) => updateLora(index, { strength })}
              disabled={disabled}
            />

            {/* Text Encoder / U-Net Toggles */}
            <div className="grid grid-cols-2 gap-2">
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

            {/* U-Net Layer Weights */}
            {lora.apply_to_unet && (
              <div className="space-y-2 pl-4 border-l-2 border-gray-700">
                <div className="text-xs text-gray-400 mb-2">U-Net Layer Weights</div>
                <Slider
                  label="Down Blocks"
                  min={0}
                  max={2}
                  step={0.1}
                  value={lora.unet_layer_weights.down}
                  onChange={(down) =>
                    updateLora(index, {
                      unet_layer_weights: { ...lora.unet_layer_weights, down },
                    })
                  }
                  disabled={disabled}
                />
                <Slider
                  label="Mid Block"
                  min={0}
                  max={2}
                  step={0.1}
                  value={lora.unet_layer_weights.mid}
                  onChange={(mid) =>
                    updateLora(index, {
                      unet_layer_weights: { ...lora.unet_layer_weights, mid },
                    })
                  }
                  disabled={disabled}
                />
                <Slider
                  label="Up Blocks"
                  min={0}
                  max={2}
                  step={0.1}
                  value={lora.unet_layer_weights.up}
                  onChange={(up) =>
                    updateLora(index, {
                      unet_layer_weights: { ...lora.unet_layer_weights, up },
                    })
                  }
                  disabled={disabled}
                />
              </div>
            )}

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
            No LoRA files found in models/lora directory
          </div>
        )}
      </div>
    </Card>
  );
}
