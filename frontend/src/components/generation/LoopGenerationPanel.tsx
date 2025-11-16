"use client";

import { useState } from "react";
import Button from "../common/Button";

export interface LoopGenerationStep {
  id: string;
  enabled: boolean;

  // Size settings
  width?: number;
  height?: number;
  scale?: number;
  linkAspectRatio: boolean;
  upscaleMethod: "image" | "latent";

  // Generation settings
  denoisingStrength: number;
  doFullSteps: boolean;
  useMainSettings: boolean;
  steps?: number;
  cfgScale?: number;
  seed?: number;

  // ControlNet
  controlnets: any[]; // TODO: Define proper type

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
}

export default function LoopGenerationPanel({
  config,
  onChange,
  mode,
  mainWidth,
  mainHeight,
}: LoopGenerationPanelProps) {
  const [expandedStep, setExpandedStep] = useState<string | null>(null);

  const addStep = () => {
    const newStep: LoopGenerationStep = {
      id: `step_${Date.now()}`,
      enabled: true,
      linkAspectRatio: true,
      upscaleMethod: "latent",
      denoisingStrength: 0.5,
      doFullSteps: false,
      useMainSettings: true,
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
          No loop steps configured. Click "Add Step" to create one.
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
                <h4 className="text-xs font-semibold text-gray-300 mb-2">Size</h4>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Width</label>
                    <input
                      type="number"
                      value={step.width || mainWidth}
                      onChange={(e) => updateStep(step.id, { width: parseInt(e.target.value) })}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs"
                      step={8}
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Height</label>
                    <input
                      type="number"
                      value={step.height || mainHeight}
                      onChange={(e) => updateStep(step.id, { height: parseInt(e.target.value) })}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs"
                      step={8}
                    />
                  </div>
                </div>
                <div className="flex items-center gap-2 mt-2">
                  <input
                    type="checkbox"
                    checked={step.linkAspectRatio}
                    onChange={(e) => updateStep(step.id, { linkAspectRatio: e.target.checked })}
                    className="cursor-pointer"
                  />
                  <label className="text-xs text-gray-400">Link to main aspect ratio</label>
                </div>
                <div className="mt-2">
                  <label className="block text-xs text-gray-400 mb-1">Upscale Method</label>
                  <select
                    value={step.upscaleMethod}
                    onChange={(e) => updateStep(step.id, { upscaleMethod: e.target.value as "image" | "latent" })}
                    className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs"
                  >
                    <option value="latent">Latent (faster, reuses latent)</option>
                    <option value="image">Image (decode → upscale → encode)</option>
                  </select>
                </div>
              </div>

              {/* Generation Settings */}
              <div>
                <h4 className="text-xs font-semibold text-gray-300 mb-2">Generation</h4>
                <div className="space-y-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Denoising Strength: {step.denoisingStrength.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={step.denoisingStrength}
                      onChange={(e) => updateStep(step.id, { denoisingStrength: parseFloat(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={step.doFullSteps}
                      onChange={(e) => updateStep(step.id, { doFullSteps: e.target.checked })}
                      className="cursor-pointer"
                    />
                    <label className="text-xs text-gray-400">Do Full Steps (ignore denoising for step count)</label>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={step.useMainSettings}
                      onChange={(e) => updateStep(step.id, { useMainSettings: e.target.checked })}
                      className="cursor-pointer"
                    />
                    <label className="text-xs text-gray-400">Use Main Settings (steps, CFG, seed)</label>
                  </div>

                  {!step.useMainSettings && (
                    <div className="grid grid-cols-2 gap-2 mt-2">
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Steps</label>
                        <input
                          type="number"
                          value={step.steps || 20}
                          onChange={(e) => updateStep(step.id, { steps: parseInt(e.target.value) })}
                          className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">CFG Scale</label>
                        <input
                          type="number"
                          value={step.cfgScale || 7}
                          onChange={(e) => updateStep(step.id, { cfgScale: parseFloat(e.target.value) })}
                          className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs"
                          step="0.5"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Seed (-1 = random)</label>
                        <input
                          type="number"
                          value={step.seed ?? -1}
                          onChange={(e) => updateStep(step.id, { seed: parseInt(e.target.value) })}
                          className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs"
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>

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

              {/* ControlNet Placeholder */}
              <div>
                <h4 className="text-xs font-semibold text-gray-300 mb-2">ControlNet</h4>
                <div className="text-xs text-gray-500">
                  ControlNet configuration (TODO)
                </div>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
