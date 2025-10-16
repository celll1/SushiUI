"use client";

import { useState, useEffect } from "react";
import Card from "./Card";
import Select from "./Select";
import Slider from "./Slider";
import Button from "./Button";
import { getControlNets } from "@/utils/api";

export interface ControlNetConfig {
  model_path: string;
  image_base64?: string;
  strength: number;
  start_step: number;
  end_step: number;
  layer_weights?: { down: number; mid: number; up: number };
  prompt?: string;
  is_lllite: boolean;
  use_input_image: boolean;
}

interface ControlNetSelectorProps {
  value: ControlNetConfig[];
  onChange: (controlnets: ControlNetConfig[]) => void;
  disabled?: boolean;
}

export default function ControlNetSelector({ value, onChange, disabled }: ControlNetSelectorProps) {
  const [availableControlNets, setAvailableControlNets] = useState<Array<{ path: string; name: string }>>([]);
  const [collapsed, setCollapsed] = useState(true);

  useEffect(() => {
    loadControlNets();
  }, []);

  const loadControlNets = async () => {
    try {
      const data = await getControlNets();
      setAvailableControlNets(data.controlnets);
    } catch (error) {
      console.error("Failed to load ControlNets:", error);
    }
  };

  const addControlNet = () => {
    const newControlNet: ControlNetConfig = {
      model_path: availableControlNets[0]?.path || "",
      strength: 1.0,
      start_step: 0.0,
      end_step: 1.0,
      layer_weights: { down: 1.0, mid: 1.0, up: 1.0 },
      is_lllite: false,
      use_input_image: false,
    };
    onChange([...value, newControlNet]);
  };

  const removeControlNet = (index: number) => {
    const newValue = [...value];
    newValue.splice(index, 1);
    onChange(newValue);
  };

  const updateControlNet = (index: number, updates: Partial<ControlNetConfig>) => {
    const newValue = [...value];
    newValue[index] = { ...newValue[index], ...updates };
    onChange(newValue);
  };

  const handleImageUpload = (index: number, file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const base64 = e.target?.result as string;
      // Remove data:image/...;base64, prefix
      const base64Data = base64.split(",")[1];
      updateControlNet(index, { image_base64: base64Data });
    };
    reader.readAsDataURL(file);
  };

  if (availableControlNets.length === 0) {
    return null;
  }

  return (
    <Card
      title={`ControlNet (${value.length})`}
      collapsible
      collapsed={collapsed}
      onToggle={() => setCollapsed(!collapsed)}
    >
      <div className="space-y-4">
        {value.map((cn, index) => (
          <div key={index} className="p-3 bg-gray-800 rounded-lg space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm font-semibold text-gray-300">
                ControlNet {index + 1}
              </span>
              <Button
                onClick={() => removeControlNet(index)}
                variant="danger"
                size="sm"
                disabled={disabled}
              >
                Remove
              </Button>
            </div>

            <Select
              label="Model"
              value={cn.model_path}
              onChange={(e) => updateControlNet(index, { model_path: e.target.value })}
              options={availableControlNets.map((cn) => ({ value: cn.path, label: cn.name }))}
              disabled={disabled}
            />

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Control Image
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleImageUpload(index, e.target.files[0]);
                  }
                }}
                className="block w-full text-sm text-gray-300
                         file:mr-4 file:py-2 file:px-4
                         file:rounded-md file:border-0
                         file:text-sm file:font-semibold
                         file:bg-blue-600 file:text-white
                         hover:file:bg-blue-700
                         disabled:opacity-50"
                disabled={disabled}
              />
              {cn.image_base64 && (
                <div className="mt-2">
                  <img
                    src={`data:image/png;base64,${cn.image_base64}`}
                    alt="Control"
                    className="max-w-full h-32 object-contain rounded border border-gray-700"
                  />
                </div>
              )}
            </div>

            <Slider
              label="Strength"
              min={0}
              max={2}
              step={0.05}
              value={cn.strength}
              onChange={(e) => updateControlNet(index, { strength: parseFloat(e.target.value) })}
              disabled={disabled}
            />

            <div className="grid grid-cols-2 gap-3">
              <Slider
                label="Start Step"
                min={0}
                max={1}
                step={0.01}
                value={cn.start_step}
                onChange={(e) => updateControlNet(index, { start_step: parseFloat(e.target.value) })}
                disabled={disabled}
              />
              <Slider
                label="End Step"
                min={0}
                max={1}
                step={0.01}
                value={cn.end_step}
                onChange={(e) => updateControlNet(index, { end_step: parseFloat(e.target.value) })}
                disabled={disabled}
              />
            </div>
          </div>
        ))}

        <Button
          onClick={addControlNet}
          variant="primary"
          disabled={disabled}
        >
          Add ControlNet
        </Button>
      </div>
    </Card>
  );
}
