"use client";

import { useState, useEffect } from "react";
import Card from "../common/Card";
import Input from "../common/Input";
import Textarea from "../common/Textarea";
import Button from "../common/Button";
import Slider from "../common/Slider";
import Select from "../common/Select";
import ModelSelector from "../common/ModelSelector";
import ImageEditor from "../common/ImageEditor";
import { getSamplers, getScheduleTypes, generateInpaint, InpaintParams as ApiInpaintParams } from "@/utils/api";

interface InpaintParams {
  prompt: string;
  negative_prompt?: string;
  steps?: number;
  cfg_scale?: number;
  sampler?: string;
  schedule_type?: string;
  seed?: number;
  width?: number;
  height?: number;
  denoising_strength?: number;
  mask_blur?: number;
  inpaint_full_res?: boolean;
  inpaint_full_res_padding?: number;
}

const DEFAULT_PARAMS: InpaintParams = {
  prompt: "",
  negative_prompt: "",
  steps: 20,
  cfg_scale: 7.0,
  sampler: "euler",
  schedule_type: "uniform",
  seed: -1,
  width: 1024,
  height: 1024,
  denoising_strength: 0.75,
  mask_blur: 4,
  inpaint_full_res: false,
  inpaint_full_res_padding: 32,
};

const STORAGE_KEY = "inpaint_params";
const PREVIEW_STORAGE_KEY = "inpaint_preview";
const INPUT_IMAGE_STORAGE_KEY = "inpaint_input_image";

export default function InpaintPanel() {
  const [params, setParams] = useState<InpaintParams>(DEFAULT_PARAMS);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [generatedImageSeed, setGeneratedImageSeed] = useState<number | null>(null);
  const [inputImage, setInputImage] = useState<File | null>(null);
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null);
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);
  const [isMounted, setIsMounted] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [showImageEditor, setShowImageEditor] = useState(false);
  const [editingImageUrl, setEditingImageUrl] = useState<string | null>(null);

  // Load from localStorage after component mounts (client-side only)
  useEffect(() => {
    setIsMounted(true);

    // Load params
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        const merged = { ...DEFAULT_PARAMS, ...parsed };
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

    // Load input image preview
    const savedInputPreview = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
    if (savedInputPreview) {
      setInputImagePreview(savedInputPreview);
    }

    loadSamplers();
    loadScheduleTypes();

    // Listen for input image updates from txt2img or img2img
    const handleInputUpdate = () => {
      const newInput = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (newInput) {
        setInputImagePreview(newInput);
      }
    };

    window.addEventListener("inpaint_input_updated", handleInputUpdate);

    return () => {
      window.removeEventListener("inpaint_input_updated", handleInputUpdate);
    };
  }, []);

  // Save params to localStorage whenever they change (but only after mounted)
  useEffect(() => {
    if (isMounted) {
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

  const processImageFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file');
      return;
    }

    setInputImage(file);
    const reader = new FileReader();
    reader.onload = (event) => {
      const preview = event.target?.result as string;
      setInputImagePreview(preview);
      if (isMounted) {
        localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, preview);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processImageFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      processImageFile(file);
    }
  };

  const handleInputImageDoubleClick = () => {
    if (inputImagePreview) {
      setEditingImageUrl(inputImagePreview);
      setShowImageEditor(true);
    }
  };

  const handleEditorSave = (editedImageUrl: string) => {
    setInputImagePreview(editedImageUrl);
    if (isMounted) {
      localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, editedImageUrl);
    }
    setShowImageEditor(false);
  };

  const handleEditorSaveMask = (maskUrl: string) => {
    setMaskImage(maskUrl);
  };

  const handleEditorClose = () => {
    setShowImageEditor(false);
  };

  const handleGenerate = async () => {
    if (!params.prompt) {
      alert("Please enter a prompt");
      return;
    }

    if (!inputImagePreview) {
      alert("Please upload an input image");
      return;
    }

    if (!maskImage) {
      alert("Please draw a mask by double-clicking the input image");
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    const currentSteps = params.steps || 20;
    setTotalSteps(currentSteps);

    try {
      const apiParams: ApiInpaintParams = {
        prompt: params.prompt,
        negative_prompt: params.negative_prompt,
        steps: params.steps,
        cfg_scale: params.cfg_scale,
        sampler: params.sampler,
        schedule_type: params.schedule_type,
        seed: params.seed,
        width: params.width,
        height: params.height,
        denoising_strength: params.denoising_strength,
        mask_blur: params.mask_blur,
        inpaint_full_res: params.inpaint_full_res,
        inpaint_full_res_padding: params.inpaint_full_res_padding,
      };

      const result = await generateInpaint(apiParams, inputImagePreview, maskImage);

      if (result.success) {
        const imageUrl = `/api/images/${result.image.id}/file`;
        setGeneratedImage(imageUrl);
        setGeneratedImageSeed(result.actual_seed);

        if (isMounted) {
          localStorage.setItem(PREVIEW_STORAGE_KEY, imageUrl);
        }
      } else {
        alert("Generation failed");
      }
    } catch (error) {
      console.error("Generation error:", error);
      alert("Generation failed: " + (error instanceof Error ? error.message : String(error)));
    } finally {
      setIsGenerating(false);
    }
  };

  // Handle Ctrl+Enter keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === 'Enter' && !isGenerating) {
        e.preventDefault();
        handleGenerate();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [params, isGenerating, inputImage, inputImagePreview, maskImage]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Parameters Panel */}
      <div className="space-y-4">
        <ModelSelector />

        <Card title="Input Image">
          <div className="space-y-4">
            <input
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/webp"
              onChange={handleImageUpload}
              className="block w-full text-sm text-gray-400
                file:mr-4 file:py-2 file:px-4
                file:rounded-lg file:border-0
                file:text-sm file:font-medium
                file:bg-blue-600 file:text-white
                hover:file:bg-blue-700
                file:cursor-pointer cursor-pointer"
            />
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onDoubleClick={handleInputImageDoubleClick}
              className={`aspect-square bg-gray-800 rounded-lg overflow-hidden border-2 border-dashed transition-colors relative ${
                isDragging
                  ? 'border-blue-500 bg-gray-700'
                  : inputImagePreview
                  ? 'border-gray-600 cursor-pointer hover:border-blue-500'
                  : 'border-gray-600'
              }`}
              title={inputImagePreview ? 'Double-click to edit and add inpaint mask' : ''}
            >
              {inputImagePreview ? (
                <>
                  <img
                    src={inputImagePreview}
                    alt="Input"
                    className="w-full h-full object-contain"
                  />
                  {maskImage && (
                    <img
                      src={maskImage}
                      alt="Mask overlay"
                      className="absolute inset-0 w-full h-full object-contain opacity-50 mix-blend-screen"
                      style={{ pointerEvents: 'none' }}
                    />
                  )}
                </>
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <p className="text-gray-500 text-center px-4">
                    {isDragging
                      ? 'Drop image here'
                      : 'Drag and drop an image here or use the file picker above'}
                  </p>
                </div>
              )}
            </div>
            {inputImagePreview && (
              <p className="text-xs text-gray-500 text-center">
                💡 Double-click image to edit and draw inpaint mask
              </p>
            )}
          </div>
        </Card>

        <Card title="Prompt">
          <Textarea
            label="Positive Prompt"
            placeholder="Enter your prompt here..."
            rows={4}
            value={params.prompt}
            onChange={(e) => setParams({ ...params, prompt: e.target.value })}
            enableWeightControl={true}
          />
          <Textarea
            label="Negative Prompt"
            placeholder="Enter negative prompt..."
            rows={3}
            value={params.negative_prompt}
            onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
            enableWeightControl={true}
          />
        </Card>

        <Card title="Parameters">
          <div className="space-y-4">
            <Slider
              label="Denoising Strength"
              min={0}
              max={1}
              step={0.05}
              value={params.denoising_strength}
              onChange={(e) => setParams({ ...params, denoising_strength: parseFloat(e.target.value) })}
            />

            <Slider
              label="Mask Blur"
              min={0}
              max={64}
              step={1}
              value={params.mask_blur}
              onChange={(e) => setParams({ ...params, mask_blur: parseInt(e.target.value) })}
            />

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

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Seed
              </label>
              <div className="flex gap-2">
                <Input
                  type="number"
                  value={params.seed}
                  onChange={(e) => setParams({ ...params, seed: parseInt(e.target.value) })}
                  className="flex-1"
                />
                <Button
                  onClick={() => setParams({ ...params, seed: Math.floor(Math.random() * 2147483647) })}
                  variant="secondary"
                  size="sm"
                  title="Random seed"
                >
                  🎲
                </Button>
                <Button
                  onClick={() => setParams({ ...params, seed: -1 })}
                  variant="secondary"
                  size="sm"
                  title="Reset to random (-1)"
                >
                  -1
                </Button>
                <Button
                  onClick={() => generatedImageSeed !== null && setParams({ ...params, seed: generatedImageSeed })}
                  variant="secondary"
                  size="sm"
                  title="Use seed from preview image"
                  disabled={generatedImageSeed === null}
                >
                  ♻️
                </Button>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="inpaint_full_res"
                checked={params.inpaint_full_res || false}
                onChange={(e) => setParams({ ...params, inpaint_full_res: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="inpaint_full_res" className="text-sm">
                Inpaint at full resolution
              </label>
            </div>

            {params.inpaint_full_res && (
              <Slider
                label="Only masked padding"
                min={0}
                max={256}
                step={4}
                value={params.inpaint_full_res_padding}
                onChange={(e) => setParams({ ...params, inpaint_full_res_padding: parseInt(e.target.value) })}
              />
            )}
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

      {/* Image Editor Modal */}
      {showImageEditor && editingImageUrl && (
        <ImageEditor
          imageUrl={editingImageUrl}
          onSave={handleEditorSave}
          onClose={handleEditorClose}
          onSaveMask={handleEditorSaveMask}
          mode="inpaint"
        />
      )}
    </div>
  );
}
