"use client";

import { useState, useEffect } from "react";
import { createPortal } from "react-dom";
import Card from "./Card";
import Select from "./Select";
import Slider from "./Slider";
import Button from "./Button";
import ImageEditor from "./ImageEditor";
import LayerWeightGraph from "./LayerWeightGraph";
import { getControlNets, getControlNetInfo, ControlNetInfo } from "@/utils/api";
import RangeSlider from "./RangeSlider";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";

export interface ControlNetConfig {
  model_path: string;
  image_base64?: string;
  strength: number;
  start_step: number;  // 0-1000 step range (same as LoRA)
  end_step: number;    // 0-1000 step range (same as LoRA)
  layer_weights?: { [layerName: string]: number };  // Changed to support per-layer weights
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

interface ControlNetLayerWeightsProps {
  controlnetPath: string;
  weights: { [layerName: string]: number };
  onChange: (weights: { [layerName: string]: number }) => void;
  disabled?: boolean;
  loadControlNetInfo: (path: string) => Promise<ControlNetInfo | null>;
}

function ControlNetLayerWeights({ controlnetPath, weights, onChange, disabled, loadControlNetInfo }: ControlNetLayerWeightsProps) {
  const [layers, setLayers] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLllite, setIsLllite] = useState(false);

  useEffect(() => {
    loadLayers();
  }, [controlnetPath]);

  const loadLayers = async () => {
    setIsLoading(true);
    try {
      const info = await loadControlNetInfo(controlnetPath);
      if (info) {
        setIsLllite(info.is_lllite);
        if (info.layers && !info.is_lllite) {
          setLayers(info.layers);
        } else {
          setLayers([]);
        }
      }
    } catch (error) {
      console.error("Failed to load ControlNet layers:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="p-4 bg-gray-800 rounded text-gray-400 text-center text-sm">
        Loading model information...
      </div>
    );
  }

  if (isLllite) {
    return (
      <div className="p-4 bg-gray-800 rounded border border-blue-500">
        <div className="text-sm text-blue-400 font-medium mb-1">
          ✓ ControlNet-LLLite Detected
        </div>
        <div className="text-xs text-gray-400">
          This is a lightweight ControlNet model. Layer weights are not applicable.
        </div>
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

export default function ControlNetSelector({ value, onChange, disabled, storageKey }: ControlNetSelectorPropsWithStorage) {
  const [availableControlNets, setAvailableControlNets] = useState<Array<{ path: string; name: string }>>([]);
  const [editingImageIndex, setEditingImageIndex] = useState<number | null>(null);
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
  const [controlnetInfoCache, setControlnetInfoCache] = useState<Map<string, ControlNetInfo>>(new Map());
  const [modelTypes, setModelTypes] = useState<Map<number, string>>(new Map());

  // Use panel-specific storage key for images, fallback to generic key
  const IMAGE_STORAGE_KEY = storageKey ? `${storageKey}_images` : "controlnet_images";

  useEffect(() => {
    loadControlNets();
    loadPersistedImages();
  }, []);

  useEffect(() => {
    // Detect model types for all loaded ControlNets
    value.forEach((cn, index) => {
      if (cn.model_path && !modelTypes.has(index)) {
        detectModelType(cn.model_path, index);
      }
    });
  }, [value]);

  const loadControlNets = async () => {
    try {
      const data = await getControlNets();
      setAvailableControlNets(data.controlnets);
    } catch (error) {
      console.error("Failed to load ControlNets:", error);
    }
  };

  const loadPersistedImages = async () => {
    try {
      const stored = localStorage.getItem(IMAGE_STORAGE_KEY);
      if (!stored) return;

      const imageRefs: { [index: number]: string } = JSON.parse(stored);
      const loadedImages: { [index: number]: string } = {};

      // Load all temp images in parallel
      await Promise.all(
        Object.entries(imageRefs).map(async ([index, ref]) => {
          try {
            const imageData = await loadTempImage(ref);
            if (imageData) {
              loadedImages[parseInt(index)] = imageData;
            }
          } catch (error) {
            console.error(`Failed to load ControlNet image at index ${index}:`, error);
          }
        })
      );

      // Apply loaded images to the current ControlNet configs
      if (Object.keys(loadedImages).length > 0) {
        const updatedValue = value.map((cn, index) => {
          if (loadedImages[index]) {
            // Remove data URL prefix if present
            const base64Data = loadedImages[index].includes(",")
              ? loadedImages[index].split(",")[1]
              : loadedImages[index];
            return { ...cn, image_base64: base64Data };
          }
          return cn;
        });
        onChange(updatedValue);
      }
    } catch (error) {
      console.error("Failed to load persisted ControlNet images:", error);
    }
  };

  const saveImageReference = async (index: number, imageBase64: string) => {
    try {
      // Add data URL prefix if not present
      const fullImageData = imageBase64.startsWith("data:")
        ? imageBase64
        : `data:image/png;base64,${imageBase64}`;

      const imageRef = await saveTempImage(fullImageData);

      // Update stored references
      const stored = localStorage.getItem(IMAGE_STORAGE_KEY);
      const imageRefs: { [index: number]: string } = stored ? JSON.parse(stored) : {};

      // Delete old reference if exists
      if (imageRefs[index]) {
        await deleteTempImageRef(imageRefs[index]);
      }

      imageRefs[index] = imageRef;
      localStorage.setItem(IMAGE_STORAGE_KEY, JSON.stringify(imageRefs));
    } catch (error) {
      console.error("Failed to save ControlNet image reference:", error);
    }
  };

  const deleteImageReference = async (index: number) => {
    try {
      const stored = localStorage.getItem(IMAGE_STORAGE_KEY);
      if (!stored) return;

      const imageRefs: { [index: number]: string } = JSON.parse(stored);

      if (imageRefs[index]) {
        await deleteTempImageRef(imageRefs[index]);
        delete imageRefs[index];

        // Re-index remaining images
        const reindexed: { [index: number]: string } = {};
        Object.entries(imageRefs).forEach(([idx, ref]) => {
          const numIdx = parseInt(idx);
          if (numIdx > index) {
            reindexed[numIdx - 1] = ref;
          } else {
            reindexed[numIdx] = ref;
          }
        });

        localStorage.setItem(IMAGE_STORAGE_KEY, JSON.stringify(reindexed));
      }
    } catch (error) {
      console.error("Failed to delete ControlNet image reference:", error);
    }
  };

  const loadControlNetInfo = async (controlnetPath: string): Promise<ControlNetInfo | null> => {
    // Check cache first
    if (controlnetInfoCache.has(controlnetPath)) {
      return controlnetInfoCache.get(controlnetPath)!;
    }

    try {
      const info = await getControlNetInfo(controlnetPath);
      setControlnetInfoCache((prev) => new Map(prev).set(controlnetPath, info));
      return info;
    } catch (error) {
      console.error("Failed to load ControlNet info:", error);
      return null;
    }
  };

  const detectModelType = async (controlnetPath: string, index: number) => {
    try {
      const info = await loadControlNetInfo(controlnetPath);
      if (info) {
        const modelType = info.is_lllite ? "LLLite" : "Standard";
        setModelTypes((prev) => new Map(prev).set(index, modelType));
      }
    } catch (error) {
      console.error("Failed to detect model type:", error);
    }
  };

  const addControlNet = () => {
    if (availableControlNets.length === 0) return;

    const newControlNet: ControlNetConfig = {
      model_path: availableControlNets[0]?.path || "",
      strength: 1.0,
      start_step: 0,
      end_step: 1000,
      layer_weights: {},  // Will be initialized by LayerWeightGraph
      is_lllite: false,
      use_input_image: false,
    };
    onChange([...value, newControlNet]);
  };

  const removeControlNet = async (index: number) => {
    // Delete the image reference before removing the ControlNet
    await deleteImageReference(index);

    const newValue = [...value];
    newValue.splice(index, 1);
    onChange(newValue);
  };

  const updateControlNet = (index: number, updates: Partial<ControlNetConfig>) => {
    const newValue = [...value];
    newValue[index] = { ...newValue[index], ...updates };
    onChange(newValue);

    // Re-detect model type if model_path changed
    if (updates.model_path) {
      detectModelType(updates.model_path, index);
    }
  };

  const handleImageUpload = (index: number, file: File) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      const base64 = e.target?.result as string;
      // Remove data:image/...;base64, prefix
      const base64Data = base64.split(",")[1];
      updateControlNet(index, { image_base64: base64Data });

      // Save to temp storage
      await saveImageReference(index, base64Data);
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

  const handleSaveEditedImage = async (editedImageUrl: string) => {
    if (editingImageIndex !== null) {
      const base64Data = editedImageUrl.split(",")[1];
      updateControlNet(editingImageIndex, { image_base64: base64Data });

      // Save to temp storage
      await saveImageReference(editingImageIndex, base64Data);

      setEditingImageIndex(null);
    }
  };

  return (
    <>
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
            <div className="flex gap-2 mb-3 items-start">
              <div className="flex-1">
                <select
                  value={cn.model_path}
                  onChange={(e) => updateControlNet(index, { model_path: e.target.value })}
                  disabled={disabled}
                  className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                >
                  {availableControlNets.map((availCn) => (
                    <option key={availCn.path} value={availCn.path}>
                      {availCn.name}
                    </option>
                  ))}
                </select>
                {/* Model Type Badge */}
                {modelTypes.has(index) && (
                  <div className="mt-1">
                    <span
                      className={`inline-block text-xs px-2 py-0.5 rounded ${
                        modelTypes.get(index) === "LLLite"
                          ? "bg-blue-600 text-blue-100"
                          : "bg-gray-600 text-gray-100"
                      }`}
                    >
                      {modelTypes.get(index) === "LLLite" ? "⚡ ControlNet-LLLite" : "🎯 Standard ControlNet"}
                    </span>
                  </div>
                )}
              </div>
              <Button
                onClick={() => removeControlNet(index)}
                disabled={disabled}
                variant="secondary"
                size="sm"
              >
                Remove
              </Button>
            </div>

            {/* 2-column layout: Image on left, settings on right */}
            <div className="grid grid-cols-2 gap-4">
              {/* Left column: Control Image */}
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
                    💡 Double-click to edit
                  </p>
                )}
              </div>

              {/* Right column: Settings */}
              <div className="space-y-3">
                <Slider
                  label="Strength"
                  min={0}
                  max={2}
                  step={0.05}
                  value={cn.strength}
                  onChange={(e) => updateControlNet(index, { strength: parseFloat(e.target.value) })}
                  disabled={disabled}
                />

                <RangeSlider
                  label="Step Range"
                  min={0}
                  max={1000}
                  step={1}
                  value={[cn.start_step, cn.end_step]}
                  onChange={(values) => updateControlNet(index, {
                    start_step: values[0],
                    end_step: values[1]
                  })}
                  disabled={disabled}
                />

                {/* U-Net Block Weights */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    U-Net Block Weights
                  </label>
                  <ControlNetLayerWeights
                    controlnetPath={cn.model_path}
                    weights={cn.layer_weights || {}}
                    onChange={(layer_weights) => updateControlNet(index, { layer_weights })}
                    disabled={disabled}
                    loadControlNetInfo={loadControlNetInfo}
                  />
                </div>
              </div>
            </div>

            {/* Optional Prompt for this ControlNet */}
            <div className="mt-3">
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Optional Prompt (for this ControlNet only)
              </label>
              <input
                type="text"
                value={cn.prompt || ""}
                onChange={(e) => updateControlNet(index, { prompt: e.target.value })}
                disabled={disabled}
                placeholder="Leave empty to use main prompt"
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              />
            </div>

            {/* Use Input Image Toggle */}
            <div className="mt-3">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={cn.use_input_image}
                  onChange={(e) => updateControlNet(index, { use_input_image: e.target.checked })}
                  disabled={disabled}
                  className="rounded"
                />
                <span className="text-sm text-gray-300">Use input image as control (img2img/inpaint only)</span>
              </label>
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
    </Card>

    {/* Image Editor Overlay - Use portal to render at document body level */}
    {editingImageIndex !== null && value[editingImageIndex]?.image_base64 && typeof document !== 'undefined' &&
      createPortal(
        <ImageEditor
          imageUrl={`data:image/png;base64,${value[editingImageIndex].image_base64}`}
          onSave={handleSaveEditedImage}
          onClose={() => setEditingImageIndex(null)}
        />,
        document.body
      )
    }
    </>
  );
}
