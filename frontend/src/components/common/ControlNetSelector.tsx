"use client";

import { useState, useEffect } from "react";
import Card from "./Card";
import Select from "./Select";
import Slider from "./Slider";
import Button from "./Button";
import ImageEditor from "./ImageEditor";
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

interface ControlNetSelectorPropsWithStorage extends ControlNetSelectorProps {
  storageKey?: string;
}

export default function ControlNetSelector({ value, onChange, disabled, storageKey }: ControlNetSelectorPropsWithStorage) {
  const [availableControlNets, setAvailableControlNets] = useState<Array<{ path: string; name: string }>>([]);
  const [editingImageIndex, setEditingImageIndex] = useState<number | null>(null);
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);

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
    if (availableControlNets.length === 0) return;

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

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    e.stopPropagation();
    setDraggingIndex(index);
  };

  const handleDragLeave = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    e.stopPropagation();
    setDraggingIndex(null);
  };

  const handleDrop = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    e.stopPropagation();
    setDraggingIndex(null);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleImageUpload(index, file);
    }
  };

  const handleEditImage = (index: number) => {
    if (value[index].image_base64) {
      setEditingImageIndex(index);
    }
  };

  const handleSaveEditedImage = (editedImageUrl: string) => {
    if (editingImageIndex !== null) {
      const base64Data = editedImageUrl.split(",")[1];
      updateControlNet(editingImageIndex, { image_base64: base64Data });
      setEditingImageIndex(null);
    }
  };

  return (
    <Card
      title={`ControlNet (${value.length})`}
      collapsible={true}
      defaultCollapsed={false}
      storageKey={storageKey}
      collapsedPreview={
        value.length > 0 ? (
          <div className="text-xs text-gray-400 truncate">
            {value.map((cn) => cn.model_path.split("/").pop()).join(", ")}
          </div>
        ) : undefined
      }
    >
      <div className="space-y-4">
        {availableControlNets.length === 0 && (
          <div className="text-sm text-gray-400 p-2">
            No ControlNet models found in the controlnet directory.
          </div>
        )}
        {value.map((cn, index) => (
          <div key={index} className="p-3 bg-gray-800 rounded-lg">
            {/* Model Selection */}
            <div className="flex gap-2 mb-3">
              <select
                value={cn.model_path}
                onChange={(e) => updateControlNet(index, { model_path: e.target.value })}
                disabled={disabled}
                className="flex-1 bg-gray-700 text-white px-3 py-2 rounded text-sm"
              >
                {availableControlNets.map((availCn) => (
                  <option key={availCn.path} value={availCn.path}>
                    {availCn.name}
                  </option>
                ))}
              </select>
              <Button
                onClick={() => removeControlNet(index)}
                disabled={disabled}
                variant="secondary"
                size="sm"
              >
                Remove
              </Button>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Control Image
              </label>
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg,image/webp"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleImageUpload(index, e.target.files[0]);
                  }
                }}
                className="block w-full text-sm text-gray-400
                         file:mr-4 file:py-2 file:px-4
                         file:rounded-lg file:border-0
                         file:text-sm file:font-medium
                         file:bg-blue-600 file:text-white
                         hover:file:bg-blue-700
                         file:cursor-pointer cursor-pointer
                         disabled:opacity-50"
                disabled={disabled}
              />
              <div
                onDragOver={(e) => handleDragOver(e, index)}
                onDragLeave={(e) => handleDragLeave(e, index)}
                onDrop={(e) => handleDrop(e, index)}
                onDoubleClick={() => handleEditImage(index)}
                className={`mt-2 aspect-square bg-gray-800 rounded-lg overflow-hidden border-2 border-dashed transition-colors ${
                  draggingIndex === index
                    ? 'border-blue-500 bg-gray-700'
                    : 'border-gray-600'
                } ${cn.image_base64 ? 'cursor-pointer' : ''}`}
                title={cn.image_base64 ? "Double-click to edit image" : ""}
              >
                {cn.image_base64 ? (
                  <img
                    src={`data:image/png;base64,${cn.image_base64}`}
                    alt="Control"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <p className="text-gray-500 text-center px-4 text-sm">
                      {draggingIndex === index
                        ? 'Drop image here'
                        : 'Drag and drop an image here or use the file picker above'}
                    </p>
                  </div>
                )}
              </div>
              {cn.image_base64 && (
                <p className="text-xs text-gray-500 text-center mt-1">
                  💡 Double-click the image to edit with built-in paint tool
                </p>
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

        {/* Add ControlNet Button */}
        <Button
          onClick={addControlNet}
          disabled={disabled || availableControlNets.length === 0}
          variant="secondary"
          className="w-full"
        >
          + Add ControlNet
        </Button>
      </div>

      {/* Image Editor Overlay */}
      {editingImageIndex !== null && value[editingImageIndex]?.image_base64 && (
        <ImageEditor
          imageUrl={`data:image/png;base64,${value[editingImageIndex].image_base64}`}
          onSave={handleSaveEditedImage}
          onClose={() => setEditingImageIndex(null)}
        />
      )}
    </Card>
  );
}
