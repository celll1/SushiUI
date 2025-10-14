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
import { getSamplers, getScheduleTypes, generateImg2Img } from "@/utils/api";

interface Img2ImgParams {
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
}

const DEFAULT_PARAMS: Img2ImgParams = {
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
  resize_mode: "image",
  resampling_method: "lanczos",
};

const STORAGE_KEY = "img2img_params";
const PREVIEW_STORAGE_KEY = "img2img_preview";
const INPUT_IMAGE_STORAGE_KEY = "img2img_input_image";

interface Img2ImgPanelProps {
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint") => void;
}

export default function Img2ImgPanel({ onTabChange }: Img2ImgPanelProps = {}) {
  const [params, setParams] = useState<Img2ImgParams>(DEFAULT_PARAMS);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [generatedImageSeed, setGeneratedImageSeed] = useState<number | null>(null);
  const [inputImage, setInputImage] = useState<File | null>(null);
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null);
  const [inputImageSize, setInputImageSize] = useState<{ width: number; height: number } | null>(null);
  const [sizeMode, setSizeMode] = useState<"absolute" | "scale">("absolute");
  const [scale, setScale] = useState<number>(1.0);
  const [progress, setProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);
  const [isMounted, setIsMounted] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isEditingImage, setIsEditingImage] = useState(false);
  const [sendImage, setSendImage] = useState(true);
  const [sendPrompt, setSendPrompt] = useState(true);
  const [sendParameters, setSendParameters] = useState(true);

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
      // Load image dimensions
      const img = new Image();
      img.onload = () => {
        setInputImageSize({ width: img.width, height: img.height });
      };
      img.src = savedInputPreview;
    }

    loadSamplers();
    loadScheduleTypes();

    // Listen for input image updates from txt2img or gallery
    const handleInputUpdate = () => {
      const newInput = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (newInput) {
        setInputImagePreview(newInput);
        // Load image dimensions
        const img = new Image();
        img.onload = () => {
          setInputImageSize({ width: img.width, height: img.height });
        };
        img.src = newInput;
      }
    };

    window.addEventListener("img2img_input_updated", handleInputUpdate);

    return () => {
      window.removeEventListener("img2img_input_updated", handleInputUpdate);
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

      // Load image to get dimensions
      const img = new Image();
      img.onload = () => {
        setInputImageSize({ width: img.width, height: img.height });
        // If in scale mode, update width/height based on scale
        if (sizeMode === "scale") {
          const scaledWidth = Math.round(img.width * scale / 64) * 64;
          const scaledHeight = Math.round(img.height * scale / 64) * 64;
          setParams({ ...params, width: scaledWidth, height: scaledHeight });
        }
      };
      img.src = preview;
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

  const handleScaleChange = (newScale: number) => {
    setScale(newScale);
    if (inputImageSize && sizeMode === "scale") {
      const scaledWidth = Math.round(inputImageSize.width * newScale / 64) * 64;
      const scaledHeight = Math.round(inputImageSize.height * newScale / 64) * 64;
      setParams({ ...params, width: scaledWidth, height: scaledHeight });
    }
  };

  const handleSizeModeChange = (newMode: "absolute" | "scale") => {
    setSizeMode(newMode);
    if (newMode === "scale" && inputImageSize) {
      // Switch to scale mode - update dimensions based on current scale
      const scaledWidth = Math.round(inputImageSize.width * scale / 64) * 64;
      const scaledHeight = Math.round(inputImageSize.height * scale / 64) * 64;
      setParams({ ...params, width: scaledWidth, height: scaledHeight });
    }
  };

  const handleEditImage = () => {
    if (inputImagePreview) {
      setIsEditingImage(true);
    }
  };

  const handleClearInputImage = () => {
    setInputImage(null);
    setInputImagePreview(null);
    setInputImageSize(null);
    if (isMounted) {
      localStorage.removeItem(INPUT_IMAGE_STORAGE_KEY);
    }
  };

  const handleSaveEditedImage = (editedImageUrl: string) => {
    setInputImagePreview(editedImageUrl);
    if (isMounted) {
      localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, editedImageUrl);
    }

    // Update image dimensions
    const img = new Image();
    img.onload = () => {
      setInputImageSize({ width: img.width, height: img.height });
    };
    img.src = editedImageUrl;

    setIsEditingImage(false);
    setInputImage(null); // Clear File object, use data URL instead
  };

  const sendToImg2Img = () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }

    // Send image if checked - already in img2img, use generated image as new input
    if (sendImage) {
      localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, generatedImage);
      window.dispatchEvent(new Event("img2img_input_updated"));
    }

    // Send prompt if checked
    if (sendPrompt) {
      const img2imgParams = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      img2imgParams.prompt = params.prompt;
      img2imgParams.negative_prompt = params.negative_prompt;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(img2imgParams));
    }

    // Send parameters if checked
    if (sendParameters) {
      const img2imgParams = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      img2imgParams.steps = params.steps;
      img2imgParams.cfg_scale = params.cfg_scale;
      img2imgParams.sampler = params.sampler;
      img2imgParams.schedule_type = params.schedule_type;
      img2imgParams.seed = params.seed;
      img2imgParams.width = params.width;
      img2imgParams.height = params.height;
      img2imgParams.denoising_strength = params.denoising_strength;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(img2imgParams));
    }

    // Reload current panel to reflect changes if image was sent
    if (sendImage) {
      setInputImagePreview(generatedImage);
    }
  };

  const sendToInpaint = () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }

    // Send image if checked
    if (sendImage) {
      localStorage.setItem("inpaint_input_image", generatedImage);
      localStorage.removeItem("inpaint_mask_image");
      window.dispatchEvent(new Event("inpaint_input_updated"));
    }

    // Send prompt if checked
    if (sendPrompt) {
      const inpaintParams = JSON.parse(localStorage.getItem("inpaint_params") || "{}");
      inpaintParams.prompt = params.prompt;
      inpaintParams.negative_prompt = params.negative_prompt;
      localStorage.setItem("inpaint_params", JSON.stringify(inpaintParams));
    }

    // Send parameters if checked
    if (sendParameters) {
      const inpaintParams = JSON.parse(localStorage.getItem("inpaint_params") || "{}");
      inpaintParams.steps = params.steps;
      inpaintParams.cfg_scale = params.cfg_scale;
      inpaintParams.sampler = params.sampler;
      inpaintParams.schedule_type = params.schedule_type;
      inpaintParams.seed = params.seed;
      inpaintParams.width = params.width;
      inpaintParams.height = params.height;
      inpaintParams.denoising_strength = params.denoising_strength;
      localStorage.setItem("inpaint_params", JSON.stringify(inpaintParams));
    }

    // Navigate to inpaint tab
    if (onTabChange) {
      onTabChange("inpaint");
    }
  };

  const handleGenerate = async () => {
    if (!params.prompt) {
      alert("Please enter a prompt");
      return;
    }

    if (!inputImage && !inputImagePreview) {
      alert("Please upload an input image");
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    const currentSteps = params.steps || 20;
    setTotalSteps(currentSteps);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev < currentSteps) {
          return prev + 1;
        }
        return prev;
      });
    }, 200);

    try {
      // Use inputImage if available, otherwise use inputImagePreview (for images sent from gallery/txt2img)
      const imageSource = inputImage || inputImagePreview;
      if (!imageSource) {
        alert("No input image available");
        return;
      }

      const result = await generateImg2Img(params, imageSource);
      setGeneratedImage(`/outputs/${result.image.filename}`);
      setGeneratedImageSeed(result.image.seed);

      // Don't update seed parameter to keep -1 for continuous random generation
      // The actual seed is saved in the database/metadata
    } catch (error) {
      console.error("Generation failed:", error);
      alert("Generation failed. Please check console for details.");
    } finally {
      clearInterval(progressInterval);
      setProgress(currentSteps);
      setTimeout(() => {
        setIsGenerating(false);
        setProgress(0);
      }, 500);
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
  }, [params, isGenerating, inputImage, inputImagePreview]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Parameters Panel */}
      <div className="space-y-4">
        <ModelSelector />

        <Card title="Input Image">
          <div className="space-y-4">
            <div className="flex gap-2">
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg,image/webp"
                onChange={handleImageUpload}
                className="flex-1 block w-full text-sm text-gray-400
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-medium
                  file:bg-blue-600 file:text-white
                  hover:file:bg-blue-700
                  file:cursor-pointer cursor-pointer"
              />
              {inputImagePreview && (
                <Button
                  onClick={handleClearInputImage}
                  variant="secondary"
                  size="sm"
                  title="Clear input image"
                >
                  Clear
                </Button>
              )}
            </div>
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onDoubleClick={handleEditImage}
              className={`aspect-square bg-gray-800 rounded-lg overflow-hidden border-2 border-dashed transition-colors ${
                isDragging
                  ? 'border-blue-500 bg-gray-700'
                  : 'border-gray-600'
              } ${inputImagePreview ? 'cursor-pointer' : ''}`}
              title={inputImagePreview ? "Double-click to edit image" : ""}
            >
              {inputImagePreview ? (
                <img
                  src={inputImagePreview}
                  alt="Input"
                  className="w-full h-full object-contain"
                />
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
                💡 Double-click the image to edit with built-in paint tool
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
            <div className="grid grid-cols-2 gap-4">
              <Select
                label="Resize Mode"
                options={[
                  { value: "image", label: "Resize Image" },
                  { value: "latent", label: "Resize Latent" },
                ]}
                value={params.resize_mode}
                onChange={(e) => setParams({ ...params, resize_mode: e.target.value })}
              />
              <Select
                label="Resampling Method"
                options={[
                  { value: "lanczos", label: "Lanczos (High Quality)" },
                  { value: "bicubic", label: "Bicubic" },
                  { value: "bilinear", label: "Bilinear" },
                  { value: "nearest", label: "Nearest (Pixelated)" },
                ]}
                value={params.resampling_method}
                onChange={(e) => setParams({ ...params, resampling_method: e.target.value })}
              />
            </div>
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
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-sm font-medium text-gray-300">
                  Size Mode
                </label>
                <div className="flex gap-2">
                  <Button
                    onClick={() => handleSizeModeChange("absolute")}
                    variant={sizeMode === "absolute" ? "primary" : "secondary"}
                    size="sm"
                  >
                    Absolute
                  </Button>
                  <Button
                    onClick={() => handleSizeModeChange("scale")}
                    variant={sizeMode === "scale" ? "primary" : "secondary"}
                    size="sm"
                    disabled={!inputImageSize}
                    title={!inputImageSize ? "Load an image first" : ""}
                  >
                    Scale
                  </Button>
                </div>
              </div>

              {sizeMode === "absolute" ? (
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
              ) : (
                <div>
                  <Slider
                    label={`Scale (${params.width}x${params.height})`}
                    min={0.25}
                    max={4.0}
                    step={0.25}
                    value={scale}
                    onChange={(e) => handleScaleChange(parseFloat(e.target.value))}
                  />
                  {inputImageSize && (
                    <p className="text-xs text-gray-500 mt-1">
                      Original: {inputImageSize.width}x{inputImageSize.height}
                    </p>
                  )}
                </div>
              )}
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
            {generatedImage && (
              <div className="space-y-3 mt-4">
                <div className="flex flex-wrap gap-2 text-sm">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={sendImage}
                      onChange={(e) => setSendImage(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-gray-300">Send image</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={sendPrompt}
                      onChange={(e) => setSendPrompt(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-gray-300">Send prompt</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={sendParameters}
                      onChange={(e) => setSendParameters(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-gray-300">Send parameters</span>
                  </label>
                </div>
                <div className="flex gap-2">
                  <Button
                    onClick={sendToImg2Img}
                    variant="secondary"
                    size="sm"
                    disabled={!sendImage && !sendPrompt && !sendParameters}
                  >
                    Send to img2img
                  </Button>
                  <Button
                    onClick={sendToInpaint}
                    variant="secondary"
                    size="sm"
                    disabled={!sendImage && !sendPrompt && !sendParameters}
                  >
                    Send to inpaint
                  </Button>
                </div>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Image Editor Overlay */}
      {isEditingImage && inputImagePreview && (
        <ImageEditor
          imageUrl={inputImagePreview}
          onSave={handleSaveEditedImage}
          onClose={() => setIsEditingImage(false)}
        />
      )}
    </div>
  );
}
