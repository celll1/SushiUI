"use client";

import { useState, useEffect } from "react";
import Card from "../common/Card";
import Input from "../common/Input";
import Textarea from "../common/Textarea";
import Button from "../common/Button";
import Slider from "../common/Slider";
import Select from "../common/Select";
import ModelSelector from "../common/ModelSelector";
import { generateTxt2Img, GenerationParams, getSamplers, getScheduleTypes } from "@/utils/api";

const DEFAULT_PARAMS: GenerationParams = {
  prompt: "",
  negative_prompt: "",
  steps: 20,
  cfg_scale: 7.0,
  sampler: "euler",
  schedule_type: "uniform",
  seed: -1,
  width: 1024,
  height: 1024,
};

const STORAGE_KEY = "txt2img_params";
const PREVIEW_STORAGE_KEY = "txt2img_preview";

export default function Txt2ImgPanel() {
  const [params, setParams] = useState<GenerationParams>(DEFAULT_PARAMS);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);
  const [isMounted, setIsMounted] = useState(false);

  // Load from localStorage after component mounts (client-side only)
  useEffect(() => {
    setIsMounted(true);

    // Load params
    const saved = localStorage.getItem(STORAGE_KEY);
    console.log("Loading params from localStorage:", saved);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        console.log("Parsed params:", parsed);
        const merged = { ...DEFAULT_PARAMS, ...parsed };
        console.log("Merged with defaults:", merged);
        setParams(merged);
      } catch (error) {
        console.error("Failed to load saved params:", error);
      }
    }

    // Load preview image
    const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
    if (savedPreview) {
      setGeneratedImage(savedPreview);
    }

    loadSamplers();
    loadScheduleTypes();
  }, []);

  // Save params to localStorage whenever they change (but only after mounted)
  useEffect(() => {
    if (isMounted) {
      console.log("Saving params to localStorage:", params);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
    }
  }, [params, isMounted]);

  // Save preview image to localStorage whenever it changes
  useEffect(() => {
    if (isMounted && generatedImage) {
      localStorage.setItem(PREVIEW_STORAGE_KEY, generatedImage);
    }
  }, [generatedImage, isMounted]);

  const resetToDefault = () => {
    setParams(DEFAULT_PARAMS);
    localStorage.removeItem(STORAGE_KEY);
  };

  const importFromImage = (imageData: any) => {
    const imported: GenerationParams = {
      prompt: imageData.prompt || "",
      negative_prompt: imageData.negative_prompt || "",
      steps: imageData.steps || DEFAULT_PARAMS.steps,
      cfg_scale: imageData.cfg_scale || DEFAULT_PARAMS.cfg_scale,
      sampler: imageData.parameters?.sampler || DEFAULT_PARAMS.sampler,
      schedule_type: imageData.parameters?.schedule_type || DEFAULT_PARAMS.schedule_type,
      seed: imageData.seed || -1,
      width: imageData.width || DEFAULT_PARAMS.width,
      height: imageData.height || DEFAULT_PARAMS.height,
    };
    setParams(imported);
  };

  const loadSamplers = async () => {
    try {
      const data = await getSamplers();
      setSamplers(data.samplers);
    } catch (error) {
      console.error("Failed to load samplers:", error);
    }
  };

  const loadScheduleTypes = async () => {
    try {
      const data = await getScheduleTypes();
      setScheduleTypes(data.schedule_types);
    } catch (error) {
      console.error("Failed to load schedule types:", error);
    }
  };

  const handleGenerate = async () => {
    if (!params.prompt) {
      alert("Please enter a prompt");
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    setTotalSteps(params.steps || 20);

    // Simulate progress (since we don't have real-time updates from backend yet)
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev < totalSteps) {
          return prev + 1;
        }
        return prev;
      });
    }, 200); // Rough estimation: 200ms per step

    try {
      const result = await generateTxt2Img(params);
      setGeneratedImage(`/outputs/${result.image.filename}`);

      // Update seed if it was random
      if (result.actual_seed && params.seed === -1) {
        setParams(prev => ({ ...prev, seed: result.actual_seed }));
      }
    } catch (error) {
      console.error("Generation failed:", error);
      alert("Generation failed. Please check console for details.");
    } finally {
      clearInterval(progressInterval);
      setProgress(totalSteps);
      setTimeout(() => {
        setIsGenerating(false);
        setProgress(0);
      }, 500);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Parameters Panel */}
      <div className="space-y-4">
        <ModelSelector />

        <Card title="Prompt">
          <Textarea
            label="Positive Prompt"
            placeholder="Enter your prompt here..."
            rows={4}
            value={params.prompt}
            onChange={(e) => setParams({ ...params, prompt: e.target.value })}
          />
          <Textarea
            label="Negative Prompt"
            placeholder="Enter negative prompt..."
            rows={3}
            value={params.negative_prompt}
            onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
          />
        </Card>

        <Card title="Parameters">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Slider
                label="Steps"
                min={1}
                max={150}
                step={1}
                value={params.steps}
                onChange={(e) => setParams({ ...params, steps: parseInt(e.target.value) })}
              />
              <Slider
                label="CFG Scale"
                min={1}
                max={30}
                step={0.5}
                value={params.cfg_scale}
                onChange={(e) => setParams({ ...params, cfg_scale: parseFloat(e.target.value) })}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Slider
                label="Width"
                min={64}
                max={2048}
                step={64}
                value={params.width}
                onChange={(e) => setParams({ ...params, width: parseInt(e.target.value) })}
              />
              <Slider
                label="Height"
                min={64}
                max={2048}
                step={64}
                value={params.height}
                onChange={(e) => setParams({ ...params, height: parseInt(e.target.value) })}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Select
                label="Sampler"
                options={samplers.map(s => ({ value: s.id, label: s.name }))}
                value={params.sampler}
                onChange={(e) => setParams({ ...params, sampler: e.target.value })}
              />
              <Select
                label="Schedule Type"
                options={scheduleTypes.map(s => ({ value: s.id, label: s.name }))}
                value={params.schedule_type}
                onChange={(e) => setParams({ ...params, schedule_type: e.target.value })}
              />
            </div>
            <Input
              label="Seed"
              type="number"
              value={params.seed}
              onChange={(e) => setParams({ ...params, seed: parseInt(e.target.value) })}
            />
          </div>
        </Card>

        <div className="flex gap-2">
          <Button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="flex-1"
            size="lg"
          >
            {isGenerating ? "Generating..." : "Generate"}
          </Button>
          <Button
            onClick={resetToDefault}
            disabled={isGenerating}
            variant="secondary"
            size="lg"
          >
            Reset
          </Button>
        </div>
      </div>

      {/* Preview Panel */}
      <div>
        <Card title="Preview">
          <div className="space-y-2">
            {isGenerating && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-gray-400">
                  <span>Generating...</span>
                  <span>{progress}/{totalSteps} steps</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-200"
                    style={{ width: `${(progress / totalSteps) * 100}%` }}
                  />
                </div>
              </div>
            )}
            <div className="aspect-square bg-gray-800 rounded-lg flex items-center justify-center">
              {generatedImage ? (
                <img
                  src={generatedImage}
                  alt="Generated"
                  className="max-w-full max-h-full rounded-lg"
                />
              ) : (
                <p className="text-gray-500">No image generated yet</p>
              )}
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
