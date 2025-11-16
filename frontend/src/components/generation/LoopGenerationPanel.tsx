"use client";

import { useState } from "react";
import Button from "../common/Button";
import Slider from "../common/Slider";
import Select from "../common/Select";
import Input from "../common/Input";
import ControlNetSelector from "../common/ControlNetSelector";

export interface LoopGenerationStep {
  id: string;
  enabled: boolean;

  // Size settings
  sizeMode: "absolute" | "scale";
  width?: number;
  height?: number;
  scale?: number;
  linkAspectRatio: boolean;
  resizeMode: "image" | "latent";
  resamplingMethod: "lanczos" | "bicubic" | "bilinear" | "nearest";

  // Generation settings
  denoisingStrength: number;
  doFullSteps: boolean;
  useMainSettings: boolean;
  useMainLoRAs: boolean; // Inherit LoRAs from main generation
  useMainControlNets: boolean; // Inherit ControlNets from main generation
  steps?: number;
  cfgScale?: number;
  sampler?: string;
  scheduleType?: string;
  seed?: number;
  ancestralSeed?: number;

  // ControlNet
  controlnets: Array<{
    model_path: string;
    image_base64?: string;
    useLoopImage: boolean; // Use generated image from loop vs custom image
    strength: number;
    start_step: number;
    end_step: number;
    layer_weights?: { [layerName: string]: number };
    prompt?: string;
    is_lllite: boolean;
    preprocessor?: string;
    enable_preprocessor: boolean;
  }>;

  // Inpaint specific
  keepMask?: boolean;
}

export interface LoopGenerationConfig {
  enabled: boolean;
  steps: LoopGenerationStep[];
}

interface LoopGenerationPanelProps {
  config: LoopGenerationConfig;
  onChange: (config: LoopGenerationConfig) => void;
  mode: "txt2img" | "img2img" | "inpaint";
  mainWidth: number;
  mainHeight: number;
  samplers?: Array<{ id: string; name: string }>;
  scheduleTypes?: Array<{ id: string; name: string }>;
}

export default function LoopGenerationPanel({
  config,
  onChange,
  mode,
  mainWidth,
  mainHeight,
  samplers = [],
  scheduleTypes = [],
}: LoopGenerationPanelProps) {
  const [expandedStep, setExpandedStep] = useState<string | null>(null);

  const addStep = () => {
    const newStep: LoopGenerationStep = {
      id: `step_${Date.now()}`,
      enabled: true,
      sizeMode: "absolute",
      scale: 1.0,
      linkAspectRatio: true,
      resizeMode: "latent",
      resamplingMethod: "lanczos",
      denoisingStrength: 0.5,
      doFullSteps: true,  // Default ON
      useMainSettings: true,
      useMainLoRAs: true, // Default: inherit LoRAs
      useMainControlNets: false, // Default: don't inherit ControlNets (use loop image instead)
      controlnets: [],
      keepMask: mode === "inpaint",
    };

    onChange({
      ...config,
      steps: [...config.steps, newStep],
    });
    setExpandedStep(newStep.id);
  };

  const removeStep = (id: string) => {
    onChange({
      ...config,
      steps: config.steps.filter(s => s.id !== id),
    });
  };

  const duplicateStep = (step: LoopGenerationStep) => {
    const newStep = {
      ...step,
      id: `step_${Date.now()}`,
    };
    onChange({
      ...config,
      steps: [...config.steps, newStep],
    });
  };

  const updateStep = (id: string, updates: Partial<LoopGenerationStep>) => {
    onChange({
      ...config,
      steps: config.steps.map(s => s.id === id ? { ...s, ...updates } : s),
    });
  };

  const updateStepWithAspectRatio = (id: string, field: "width" | "height", e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    const step = config.steps.find(s => s.id === id);
    if (!step) return;

    if (step.linkAspectRatio) {
      const aspectRatio = mainWidth / mainHeight;
      if (field === "width") {
        const newHeight = Math.round(value / aspectRatio / 8) * 8;
        updateStep(id, { width: value, height: newHeight });
      } else {
        const newWidth = Math.round(value * aspectRatio / 8) * 8;
        updateStep(id, { width: newWidth, height: value });
      }
    } else {
      updateStep(id, { [field]: value });
    }
  };

  const updateStepScale = (id: string, e: React.ChangeEvent<HTMLInputElement>) => {
    const newScale = parseFloat(e.target.value);
    const scaledWidth = Math.round(mainWidth * newScale / 8) * 8;
    const scaledHeight = Math.round(mainHeight * newScale / 8) * 8;
    updateStep(id, { scale: newScale, width: scaledWidth, height: scaledHeight });
  };

  const updateStepSizeMode = (id: string, newMode: "absolute" | "scale") => {
    const step = config.steps.find(s => s.id === id);
    if (!step) return;

    if (newMode === "scale") {
      // Switch to scale mode - calculate scale based on current dimensions or use default
      const currentScale = step.scale || 1.0;
      const scaledWidth = Math.round(mainWidth * currentScale / 8) * 8;
      const scaledHeight = Math.round(mainHeight * currentScale / 8) * 8;
      updateStep(id, { sizeMode: newMode, width: scaledWidth, height: scaledHeight });
    } else {
      updateStep(id, { sizeMode: newMode });
    }
  };

  const moveStep = (id: string, direction: "up" | "down") => {
    const index = config.steps.findIndex(s => s.id === id);
    if (index === -1) return;
    if (direction === "up" && index === 0) return;
    if (direction === "down" && index === config.steps.length - 1) return;

    const newSteps = [...config.steps];
    const targetIndex = direction === "up" ? index - 1 : index + 1;
    [newSteps[index], newSteps[targetIndex]] = [newSteps[targetIndex], newSteps[index]];

    onChange({
      ...config,
      steps: newSteps,
    });
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={config.enabled}
            onChange={(e) => onChange({ ...config, enabled: e.target.checked })}
            className="cursor-pointer"
          />
          <h3 className="text-sm font-semibold text-gray-200">Loop Generation</h3>
        </div>
        <Button
          onClick={addStep}
          variant="secondary"
          className="text-xs px-2 py-1"
          disabled={!config.enabled}
        >
          + Add Step
        </Button>
      </div>

      {config.enabled && config.steps.length === 0 && (
        <div className="text-xs text-gray-400 text-center py-4">
          No loop steps configured. Click &quot;Add Step&quot; to create one.
        </div>
      )}

      {config.enabled && config.steps.map((step, index) => (
        <div
          key={step.id}
          className={`border rounded-lg transition-colors ${
            step.enabled ? "border-gray-600 bg-gray-750" : "border-gray-700 bg-gray-800 opacity-60"
          }`}
        >
          {/* Step Header */}
          <div className="flex items-center justify-between p-2 border-b border-gray-700">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={step.enabled}
                onChange={(e) => updateStep(step.id, { enabled: e.target.checked })}
                className="cursor-pointer"
              />
              <button
                onClick={() => setExpandedStep(expandedStep === step.id ? null : step.id)}
                className="text-xs font-medium text-gray-200 hover:text-white"
              >
                Step {index + 1} {expandedStep === step.id ? "▼" : "▶"}
              </button>
            </div>
            <div className="flex items-center gap-1">
              <Button
                onClick={() => moveStep(step.id, "up")}
                variant="secondary"
                className="text-xs px-1 py-0.5"
                disabled={index === 0}
              >
                ↑
              </Button>
              <Button
                onClick={() => moveStep(step.id, "down")}
                variant="secondary"
                className="text-xs px-1 py-0.5"
                disabled={index === config.steps.length - 1}
              >
                ↓
              </Button>
              <Button
                onClick={() => duplicateStep(step)}
                variant="secondary"
                className="text-xs px-2 py-0.5"
              >
                Copy
              </Button>
              <Button
                onClick={() => removeStep(step.id)}
                variant="secondary"
                className="text-xs px-2 py-0.5"
              >
                Remove
              </Button>
            </div>
          </div>

          {/* Step Settings (Expanded) */}
          {expandedStep === step.id && (
            <div className="p-3 space-y-3">
              {/* Size Settings */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-xs font-semibold text-gray-300">Size</h4>
                  <div className="flex gap-1">
                    <Button
                      onClick={() => updateStepSizeMode(step.id, "absolute")}
                      variant={step.sizeMode === "absolute" ? "primary" : "secondary"}
                      size="sm"
                      className="text-xs px-2 py-0.5"
                    >
                      Absolute
                    </Button>
                    <Button
                      onClick={() => updateStepSizeMode(step.id, "scale")}
                      variant={step.sizeMode === "scale" ? "primary" : "secondary"}
                      size="sm"
                      className="text-xs px-2 py-0.5"
                    >
                      Scale
                    </Button>
                  </div>
                </div>

                {step.sizeMode === "absolute" ? (
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-4">
                      <Slider
                        label="Width"
                        value={step.width || mainWidth}
                        onChange={(e) => updateStepWithAspectRatio(step.id, "width", e)}
                        min={64}
                        max={2048}
                        step={8}
                      />
                      <Slider
                        label="Height"
                        value={step.height || mainHeight}
                        onChange={(e) => updateStepWithAspectRatio(step.id, "height", e)}
                        min={64}
                        max={2048}
                        step={8}
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={step.linkAspectRatio}
                        onChange={(e) => updateStep(step.id, { linkAspectRatio: e.target.checked })}
                        className="cursor-pointer"
                      />
                      <label className="text-xs text-gray-400">Link to main aspect ratio</label>
                    </div>
                  </div>
                ) : (
                  <div>
                    <Slider
                      label={`Scale (${step.width || mainWidth}x${step.height || mainHeight})`}
                      value={step.scale || 1.0}
                      onChange={(e) => updateStepScale(step.id, e)}
                      min={0.25}
                      max={4.0}
                      step={0.25}
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Base: {mainWidth}x{mainHeight}
                    </p>
                  </div>
                )}

                <div className="mt-2 grid grid-cols-2 gap-4">
                  <Select
                    label="Resize Mode"
                    options={[
                      { value: "image", label: "Resize Image" },
                      { value: "latent", label: "Resize Latent" },
                    ]}
                    value={step.resizeMode}
                    onChange={(e) => updateStep(step.id, { resizeMode: e.target.value as "image" | "latent" })}
                  />
                  <Select
                    label="Resampling Method"
                    options={[
                      { value: "lanczos", label: "Lanczos (High Quality)" },
                      { value: "bicubic", label: "Bicubic" },
                      { value: "bilinear", label: "Bilinear" },
                      { value: "nearest", label: "Nearest (Pixelated)" },
                    ]}
                    value={step.resamplingMethod}
                    onChange={(e) => updateStep(step.id, { resamplingMethod: e.target.value as "lanczos" | "bicubic" | "bilinear" | "nearest" })}
                  />
                </div>
              </div>

              {/* Generation Settings */}
              <div>
                <h4 className="text-xs font-semibold text-gray-300 mb-2">Generation</h4>
                <div className="space-y-2">
                  <Slider
                    label="Denoising Strength"
                    value={step.denoisingStrength}
                    onChange={(e) => updateStep(step.id, { denoisingStrength: parseFloat(e.target.value) })}
                    min={0}
                    max={1}
                    step={0.05}
                  />
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={step.doFullSteps}
                      onChange={(e) => updateStep(step.id, { doFullSteps: e.target.checked })}
                      className="cursor-pointer"
                    />
                    <label className="text-xs text-gray-400">Do Full Steps (ignore denoising for step count)</label>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={step.useMainSettings}
                        onChange={(e) => updateStep(step.id, { useMainSettings: e.target.checked })}
                        className="cursor-pointer"
                      />
                      <label className="text-xs text-gray-400">Use Main Settings (steps, CFG, sampler, scheduler, seed)</label>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={step.useMainLoRAs}
                        onChange={(e) => updateStep(step.id, { useMainLoRAs: e.target.checked })}
                        className="cursor-pointer"
                      />
                      <label className="text-xs text-gray-400">Use Main LoRAs</label>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={step.useMainControlNets}
                        onChange={(e) => updateStep(step.id, { useMainControlNets: e.target.checked })}
                        className="cursor-pointer"
                      />
                      <label className="text-xs text-gray-400">Use Main ControlNets</label>
                    </div>
                  </div>

                  {!step.useMainSettings && (
                    <div className="space-y-3 mt-2">
                      <div className="grid grid-cols-2 gap-4">
                        <Slider
                          label="Steps"
                          value={step.steps || 20}
                          onChange={(e) => updateStep(step.id, { steps: parseInt(e.target.value) })}
                          min={1}
                          max={150}
                          step={1}
                        />
                        <Slider
                          label="CFG Scale"
                          value={step.cfgScale || 7}
                          onChange={(e) => updateStep(step.id, { cfgScale: parseFloat(e.target.value) })}
                          min={1}
                          max={30}
                          step={0.5}
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <Select
                          label="Sampler"
                          value={step.sampler || ""}
                          onChange={(e) => updateStep(step.id, { sampler: e.target.value })}
                          options={samplers.map(s => ({ value: s.id, label: s.name }))}
                        />
                        <Select
                          label="Scheduler"
                          value={step.scheduleType || ""}
                          onChange={(e) => updateStep(step.id, { scheduleType: e.target.value })}
                          options={scheduleTypes.map(s => ({ value: s.id, label: s.name }))}
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-300 mb-1">
                            Seed
                          </label>
                          <div className="flex gap-2">
                            <Input
                              type="number"
                              value={step.seed ?? -1}
                              onChange={(e) => updateStep(step.id, { seed: parseInt(e.target.value) })}
                              className="flex-1"
                            />
                            <Button
                              onClick={() => updateStep(step.id, { seed: Math.floor(Math.random() * 2147483647) })}
                              variant="secondary"
                              size="sm"
                              title="Random seed"
                            >
                              🎲
                            </Button>
                            <Button
                              onClick={() => updateStep(step.id, { seed: -1 })}
                              variant="secondary"
                              size="sm"
                              title="Reset to random (-1)"
                            >
                              -1
                            </Button>
                          </div>
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-300 mb-1">
                            Ancestral Seed
                          </label>
                          <div className="flex gap-2">
                            <Input
                              type="number"
                              value={step.ancestralSeed ?? -1}
                              onChange={(e) => updateStep(step.id, { ancestralSeed: parseInt(e.target.value) })}
                              className="flex-1"
                              placeholder="-1 (use main)"
                            />
                            <Button
                              onClick={() => updateStep(step.id, { ancestralSeed: Math.floor(Math.random() * 2147483647) })}
                              variant="secondary"
                              size="sm"
                              title="Random ancestral seed"
                            >
                              🎲
                            </Button>
                            <Button
                              onClick={() => updateStep(step.id, { ancestralSeed: -1 })}
                              variant="secondary"
                              size="sm"
                              title="Use main seed (-1)"
                            >
                              -1
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* ControlNet (only shown when not using main ControlNets) */}
              {!step.useMainControlNets && (
                <div>
                  <h4 className="text-xs font-semibold text-gray-300 mb-2">ControlNet for Loop Step</h4>
                  <div className="space-y-2">
                    <ControlNetSelector
                      value={step.controlnets.map(cn => ({
                        model_path: cn.model_path,
                        image_base64: cn.image_base64,
                        strength: cn.strength,
                        start_step: cn.start_step,
                        end_step: cn.end_step,
                        layer_weights: cn.layer_weights,
                        prompt: cn.prompt,
                        is_lllite: cn.is_lllite,
                        preprocessor: cn.preprocessor,
                        enable_preprocessor: cn.enable_preprocessor,
                      }))}
                      onChange={(controlnets) => {
                        // Map ControlNets from selector and preserve useLoopImage flag by model_path
                        const useLoopImageMap = new Map<string, boolean>();
                        step.controlnets.forEach(cn => {
                          useLoopImageMap.set(cn.model_path, cn.useLoopImage);
                        });

                        updateStep(step.id, {
                          controlnets: controlnets.map((cn) => ({
                            ...cn,
                            // Use existing useLoopImage for same model, otherwise default to true
                            useLoopImage: useLoopImageMap.get(cn.model_path) ?? true,
                          }))
                        });
                      }}
                      storageKey={`loop_controlnet_${step.id}`}
                    />

                    {step.controlnets.length > 0 && (
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={step.controlnets.every(cn => cn.useLoopImage ?? true)}
                            onChange={(e) => {
                              updateStep(step.id, {
                                controlnets: step.controlnets.map(cn => ({
                                  ...cn,
                                  useLoopImage: e.target.checked,
                                }))
                              });
                            }}
                            className="cursor-pointer"
                          />
                          <label className="text-xs text-gray-400 cursor-pointer">
                            Use loop output as ControlNet reference (uncheck to use custom images)
                          </label>
                        </div>
                        <p className="text-xs text-gray-500">
                          Note: Loop output image will be used at its original resolution (no upscaling). Applies to all {step.controlnets.length} ControlNet(s).
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Inpaint Specific */}
              {mode === "inpaint" && (
                <div>
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={step.keepMask ?? true}
                      onChange={(e) => updateStep(step.id, { keepMask: e.target.checked })}
                      className="cursor-pointer"
                    />
                    <label className="text-xs text-gray-400">Keep and scale mask from main generation</label>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
