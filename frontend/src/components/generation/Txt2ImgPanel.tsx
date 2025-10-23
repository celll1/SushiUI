"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Card from "../common/Card";
import Input from "../common/Input";
import Textarea from "../common/Textarea";
import TextareaWithTagSuggestions from "../common/TextareaWithTagSuggestions";
import Button from "../common/Button";
import Slider from "../common/Slider";
import Select from "../common/Select";
import ModelSelector from "../common/ModelSelector";
import LoRASelector from "../common/LoRASelector";
import ControlNetSelector from "../common/ControlNetSelector";
import TIPODialog, { TIPOSettings } from "../common/TIPODialog";
import FloatingGallery from "../common/FloatingGallery";
import ImageViewer from "../common/ImageViewer";
import { generateTxt2Img, GenerationParams, getSamplers, getScheduleTypes, tokenizePrompt, generateTIPOPrompt } from "@/utils/api";
import { wsClient } from "@/utils/websocket";
import { saveTempImage, loadTempImage } from "@/utils/tempImageStorage";
import { useStartup } from "@/contexts/StartupContext";

const DEFAULT_PARAMS: GenerationParams = {
  prompt: "",
  negative_prompt: "",
  steps: 20,
  cfg_scale: 7.0,
  sampler: "euler",
  schedule_type: "uniform",
  seed: -1,
  ancestral_seed: -1,
  width: 1024,
  height: 1024,
  loras: [],
  prompt_chunking_mode: "a1111",
  max_prompt_chunks: 0,
  controlnets: [],
};

const STORAGE_KEY = "txt2img_params";
const PREVIEW_STORAGE_KEY = "txt2img_preview";

interface Txt2ImgPanelProps {
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint") => void;
}

export default function Txt2ImgPanel({ onTabChange }: Txt2ImgPanelProps = {}) {
  const { modelLoaded } = useStartup();
  const [params, setParams] = useState<GenerationParams>(DEFAULT_PARAMS);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [generatedImageSeed, setGeneratedImageSeed] = useState<number | null>(null);
  const [generatedImageAncestralSeed, setGeneratedImageAncestralSeed] = useState<number | null>(null);
  const [progress, setProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);
  const [isMounted, setIsMounted] = useState(false);
  const [sendImage, setSendImage] = useState(true);
  const [sendPrompt, setSendPrompt] = useState(true);
  const [sendParameters, setSendParameters] = useState(true);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [promptTokenCount, setPromptTokenCount] = useState<number>(0);
  const [negativePromptTokenCount, setNegativePromptTokenCount] = useState<number>(0);
  const [isTIPODialogOpen, setIsTIPODialogOpen] = useState(false);
  const [tipoSettings, setTipoSettings] = useState<TIPOSettings>({
    model_name: "KBlueLeaf/TIPO-500M",
    tag_length: "short",
    nl_length: "short",
    temperature: 1.0,
    top_p: 0.95,
    top_k: 50,
    max_new_tokens: 256,
    categories: [
      { id: 'rating', label: 'Rating', enabled: true },
      { id: 'quality', label: 'Quality', enabled: true },
      { id: 'special', label: 'Special', enabled: true },
      { id: 'copyright', label: 'Copyright', enabled: true },
      { id: 'characters', label: 'Characters', enabled: true },
      { id: 'artist', label: 'Artist', enabled: true },
      { id: 'general', label: 'General', enabled: true },
      { id: 'meta', label: 'Meta', enabled: true },
      { id: 'short_nl', label: 'Short NL', enabled: false },
      { id: 'long_nl', label: 'Long NL', enabled: false }
    ]
  });
  const [isGeneratingTIPO, setIsGeneratingTIPO] = useState(false);
  const [galleryImages, setGalleryImages] = useState<Array<{ url: string; timestamp: number }>>([]);
  const [maxGalleryImages, setMaxGalleryImages] = useState(30);
  const [previewViewerOpen, setPreviewViewerOpen] = useState(false);

  const tokenizePromptTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const tokenizeNegativeTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const promptTextareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Tokenize prompts using backend tokenizer (debounced)
  useEffect(() => {
    if (tokenizePromptTimeoutRef.current) {
      clearTimeout(tokenizePromptTimeoutRef.current);
    }

    tokenizePromptTimeoutRef.current = setTimeout(async () => {
      try {
        if (params.prompt) {
          const result = await tokenizePrompt(params.prompt);
          setPromptTokenCount(result.total_count);
        } else {
          setPromptTokenCount(0);
        }
      } catch (error) {
        // Silently fail, keep previous count
        console.error("Failed to tokenize prompt:", error);
      }
    }, 300);

    return () => {
      if (tokenizePromptTimeoutRef.current) {
        clearTimeout(tokenizePromptTimeoutRef.current);
      }
    };
  }, [params.prompt]);

  useEffect(() => {
    if (tokenizeNegativeTimeoutRef.current) {
      clearTimeout(tokenizeNegativeTimeoutRef.current);
    }

    tokenizeNegativeTimeoutRef.current = setTimeout(async () => {
      try {
        if (params.negative_prompt) {
          const result = await tokenizePrompt(params.negative_prompt);
          setNegativePromptTokenCount(result.total_count);
        } else {
          setNegativePromptTokenCount(0);
        }
      } catch (error) {
        // Silently fail, keep previous count
        console.error("Failed to tokenize negative prompt:", error);
      }
    }, 300);

    return () => {
      if (tokenizeNegativeTimeoutRef.current) {
        clearTimeout(tokenizeNegativeTimeoutRef.current);
      }
    };
  }, [params.negative_prompt]);

  // WebSocket progress callback
  const handleProgress = useCallback((step: number, totalSteps: number, message: string, preview?: string) => {
    if (isGenerating) {
      setProgress(step);
      setTotalSteps(totalSteps);
      if (preview) {
        setPreviewImage(preview);
      }
    }
  }, [isGenerating]);

  // Setup WebSocket connection
  useEffect(() => {
    wsClient.connect();
    wsClient.subscribe(handleProgress);

    return () => {
      wsClient.unsubscribe(handleProgress);
    };
  }, [handleProgress]);

  // Load from localStorage after component mounts (client-side only)
  useEffect(() => {
    console.clear();
    console.log("=== Txt2ImgPanel mounted ===");
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

    // Load max gallery images setting
    const savedMaxImages = localStorage.getItem('floating_gallery_max_images');
    if (savedMaxImages) {
      setMaxGalleryImages(parseInt(savedMaxImages));
    }

  }, []);

  // Load samplers and schedule types when model is loaded
  useEffect(() => {
    if (modelLoaded) {
      loadSamplers();
      loadScheduleTypes();
    }
  }, [modelLoaded]);

  // Save params to localStorage whenever they change (but only after mounted)
  useEffect(() => {
    if (isMounted) {
      // ControlNet images are now managed by ControlNetSelector via tempImageStorage
      // We don't need to remove image_base64 here anymore, as it's no longer stored in params
      console.log("[Txt2Img] Saving params to localStorage:", {
        loras: params.loras?.length || 0,
        controlnets: params.controlnets?.length || 0,
        fullParams: params
      });
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

  const sendToTxt2Img = () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }

    // Note: Send image is not applicable for txt2img (no input image)

    // Send prompt if checked
    if (sendPrompt) {
      const txt2imgParams = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      txt2imgParams.prompt = params.prompt;
      txt2imgParams.negative_prompt = params.negative_prompt;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(txt2imgParams));
    }

    // Send parameters if checked
    if (sendParameters) {
      const txt2imgParams = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      txt2imgParams.steps = params.steps;
      txt2imgParams.cfg_scale = params.cfg_scale;
      txt2imgParams.sampler = params.sampler;
      txt2imgParams.schedule_type = params.schedule_type;
      txt2imgParams.seed = params.seed;
      txt2imgParams.width = params.width;
      txt2imgParams.height = params.height;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(txt2imgParams));
    }

    // Already in txt2img, just reload params
    // No tab change needed
  };

  const sendToImg2Img = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }

    // Send image if checked
    if (sendImage) {
      try {
        const tempRef = await saveTempImage(generatedImage);
        localStorage.setItem("img2img_input_image", tempRef);
        window.dispatchEvent(new Event("img2img_input_updated"));
      } catch (error) {
        console.error("[Txt2Img] Failed to send image to img2img:", error);
      }
    }

    // Send prompt if checked
    if (sendPrompt) {
      const img2imgParams = JSON.parse(localStorage.getItem("img2img_params") || "{}");
      img2imgParams.prompt = params.prompt;
      img2imgParams.negative_prompt = params.negative_prompt;
      localStorage.setItem("img2img_params", JSON.stringify(img2imgParams));
    }

    // Send parameters if checked
    if (sendParameters) {
      const img2imgParams = JSON.parse(localStorage.getItem("img2img_params") || "{}");
      img2imgParams.steps = params.steps;
      img2imgParams.cfg_scale = params.cfg_scale;
      img2imgParams.sampler = params.sampler;
      img2imgParams.schedule_type = params.schedule_type;
      img2imgParams.seed = params.seed;
      img2imgParams.width = params.width;
      img2imgParams.height = params.height;
      localStorage.setItem("img2img_params", JSON.stringify(img2imgParams));
    }

    // Navigate to img2img tab
    if (onTabChange) {
      onTabChange("img2img");
    }
  };

  const sendToInpaint = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }

    // Send image if checked
    if (sendImage) {
      try {
        const tempRef = await saveTempImage(generatedImage);
        localStorage.setItem("inpaint_input_image", tempRef);
        localStorage.removeItem("inpaint_mask_image");
        window.dispatchEvent(new Event("inpaint_input_updated"));
      } catch (error) {
        console.error("[Txt2Img] Failed to send image to inpaint:", error);
      }
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
      localStorage.setItem("inpaint_params", JSON.stringify(inpaintParams));
    }

    // Navigate to inpaint tab
    if (onTabChange) {
      onTabChange("inpaint");
    }
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
      console.log("[Txt2Img] Calling getSamplers()...");
      const data = await getSamplers();
      console.log("[Txt2Img] Received samplers:", data.samplers);
      setSamplers(data.samplers);
      console.log("[Txt2Img] setSamplers called");
    } catch (error) {
      console.error("Failed to load samplers:", error);
    }
  };

  const loadScheduleTypes = async () => {
    try {
      console.log("[Txt2Img] Calling getScheduleTypes()...");
      const data = await getScheduleTypes();
      console.log("[Txt2Img] Received schedule types:", data.schedule_types);
      setScheduleTypes(data.schedule_types);
      console.log("[Txt2Img] setScheduleTypes called");
    } catch (error) {
      console.error("Failed to load schedule types:", error);
    }
  };

  const handleGenerateTIPO = async () => {
    const textarea = promptTextareaRef.current;
    if (!textarea) return;

    const selectionStart = textarea.selectionStart;
    const selectionEnd = textarea.selectionEnd;
    const hasSelection = selectionStart !== selectionEnd;

    const inputPrompt = hasSelection
      ? params.prompt.substring(selectionStart, selectionEnd)
      : params.prompt;

    if (!inputPrompt.trim()) {
      alert("Please enter a prompt or select text to enhance");
      return;
    }

    setIsGeneratingTIPO(true);
    try {
      // Build category order and enabled map from settings
      const categoryOrder = tipoSettings.categories.map(c => c.id);
      const enabledCategories: Record<string, boolean> = {};
      tipoSettings.categories.forEach(c => {
        enabledCategories[c.id] = c.enabled;
      });

      const result = await generateTIPOPrompt({
        input_prompt: inputPrompt,
        tag_length: tipoSettings.tag_length,
        nl_length: tipoSettings.nl_length,
        temperature: tipoSettings.temperature,
        top_p: tipoSettings.top_p,
        top_k: tipoSettings.top_k,
        max_new_tokens: tipoSettings.max_new_tokens,
        category_order: categoryOrder,
        enabled_categories: enabledCategories
      });

      // Replace with generated prompt
      // If selection exists, only the selected portion is used as input
      // The entire prompt is replaced with the generated result
      setParams({ ...params, prompt: result.generated_prompt });
    } catch (error) {
      console.error("TIPO generation failed:", error);
      alert("TIPO generation failed. Make sure the model is loaded in settings.");
    } finally {
      setIsGeneratingTIPO(false);
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
    setPreviewImage(null);
    setGeneratedImage(null);

    try {
      const result = await generateTxt2Img(params);
      const imageUrl = `/outputs/${result.image.filename}`;
      setGeneratedImage(imageUrl);
      setGeneratedImageSeed(result.image.seed);
      setGeneratedImageAncestralSeed(result.image.ancestral_seed || null);

      // Add to gallery
      setGalleryImages(prev => [...prev, { url: imageUrl, timestamp: Date.now() }]);

      // Don't update seed parameter to keep -1 for continuous random generation
      // The actual seed is saved in the database/metadata
    } catch (error) {
      console.error("Generation failed:", error);
      alert("Generation failed. Please check console for details.");
    } finally {
      setTimeout(() => {
        setIsGenerating(false);
        setProgress(0);
        setPreviewImage(null);
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
  }, [params, isGenerating]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Parameters Panel */}
      <div className="space-y-4">
        <ModelSelector />

        <LoRASelector
          value={params.loras || []}
          onChange={(loras) => {
            console.log("[Txt2Img] LoRA onChange called with:", loras);
            setParams({ ...params, loras });
          }}
          disabled={isGenerating}
          storageKey="txt2img_lora_collapsed"
        />

        <ControlNetSelector
          value={params.controlnets || []}
          onChange={(controlnets) => {
            console.log("[Txt2Img] ControlNet onChange called with:", controlnets);
            setParams({ ...params, controlnets });
          }}
          disabled={isGenerating}
          storageKey="txt2img_controlnet_collapsed"
        />

        <Card title="Prompt">
          <div className="relative">
            <TextareaWithTagSuggestions
              label="Positive Prompt"
              placeholder="Enter your prompt here..."
              rows={4}
              value={params.prompt}
              onChange={(e) => {
                setParams({ ...params, prompt: e.target.value });
                // Capture textarea ref from the event
                if (e.target) {
                  promptTextareaRef.current = e.target as HTMLTextAreaElement;
                }
              }}
              enableWeightControl={true}
            />
            <div className="absolute top-0 right-0 flex items-center gap-1 px-2 py-1">
              <button
                onClick={handleGenerateTIPO}
                disabled={isGeneratingTIPO || isGenerating}
                className="p-1 hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Generate enhanced prompt with TIPO"
              >
                {isGeneratingTIPO ? "⏳" : "✨"}
              </button>
              <button
                onClick={() => setIsTIPODialogOpen(true)}
                disabled={isGenerating}
                className="p-1 hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="TIPO Settings"
              >
                ⚙️
              </button>
              <span className="text-xs text-gray-400 pointer-events-none ml-1">
                {promptTokenCount} tokens
              </span>
            </div>
          </div>
          <div className="relative">
            <TextareaWithTagSuggestions
              label="Negative Prompt"
              placeholder="Enter negative prompt..."
              rows={3}
              value={params.negative_prompt}
              onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
              enableWeightControl={true}
            />
            <div className="absolute top-0 right-0 text-xs text-gray-400 px-2 py-1 pointer-events-none">
              {negativePromptTokenCount} tokens
            </div>
          </div>
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
            <div className="grid grid-cols-2 gap-4">
              <Select
                label="Prompt Chunking Mode"
                options={[
                  { value: "a1111", label: "A1111 (Separate chunks)" },
                  { value: "sd_scripts", label: "sd-scripts (Single BOS/EOS)" },
                  { value: "nobos", label: "No BOS/EOS" },
                ]}
                value={params.prompt_chunking_mode || "a1111"}
                onChange={(e) => setParams({ ...params, prompt_chunking_mode: e.target.value })}
              />
              <Select
                label="Max Chunks"
                options={[
                  { value: "0", label: "Unlimited" },
                  { value: "1", label: "1 chunk (75 tokens)" },
                  { value: "2", label: "2 chunks (150 tokens)" },
                  { value: "3", label: "3 chunks (225 tokens)" },
                  { value: "4", label: "4 chunks (300 tokens)" },
                ]}
                value={params.max_prompt_chunks?.toString() || "0"}
                onChange={(e) => setParams({ ...params, max_prompt_chunks: parseInt(e.target.value) })}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
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
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Ancestral Seed
                <span className="text-xs text-gray-500 ml-2">(for Euler a, DPM2 a, etc.)</span>
              </label>
              <div className="flex gap-2">
                <Input
                  type="number"
                  value={params.ancestral_seed}
                  onChange={(e) => setParams({ ...params, ancestral_seed: parseInt(e.target.value) })}
                  className="flex-1"
                  placeholder="-1 (use main seed)"
                />
                <Button
                  onClick={() => setParams({ ...params, ancestral_seed: Math.floor(Math.random() * 2147483647) })}
                  variant="secondary"
                  size="sm"
                  title="Random ancestral seed"
                >
                  🎲
                </Button>
                <Button
                  onClick={() => setParams({ ...params, ancestral_seed: -1 })}
                  variant="secondary"
                  size="sm"
                  title="Use main seed (-1)"
                >
                  -1
                </Button>
                <Button
                  onClick={() => generatedImageAncestralSeed !== null && setParams({ ...params, ancestral_seed: generatedImageAncestralSeed })}
                  variant="secondary"
                  size="sm"
                  title="Use ancestral seed from preview image"
                  disabled={generatedImageAncestralSeed === null}
                >
                  ♻️
                </Button>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                -1 = use main seed (default). Set a different value to vary details while keeping composition.
              </p>
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
            <div
              className="aspect-square bg-gray-800 rounded-lg flex items-center justify-center cursor-pointer"
              onDoubleClick={() => {
                if (generatedImage) {
                  setPreviewViewerOpen(true);
                }
              }}
            >
              {generatedImage ? (
                <img
                  src={generatedImage}
                  alt="Generated"
                  className="max-w-full max-h-full rounded-lg"
                />
              ) : previewImage ? (
                <img
                  src={`data:image/jpeg;base64,${previewImage}`}
                  alt="Preview"
                  className="max-w-full max-h-full rounded-lg opacity-80"
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
                <div className="grid grid-cols-3 gap-2">
                  <Button
                    onClick={sendToTxt2Img}
                    variant="secondary"
                    size="sm"
                    disabled={!sendPrompt && !sendParameters}
                    title="Send image not applicable for txt2img"
                  >
                    Send to txt2img
                  </Button>
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

      {/* Floating Gallery */}
      <FloatingGallery images={galleryImages} maxImages={maxGalleryImages} />

      {/* Preview Image Viewer */}
      {previewViewerOpen && generatedImage && (
        <ImageViewer
          imageUrl={generatedImage}
          onClose={() => setPreviewViewerOpen(false)}
        />
      )}

      {/* TIPO Dialog */}
      <TIPODialog
        isOpen={isTIPODialogOpen}
        onClose={() => setIsTIPODialogOpen(false)}
        settings={tipoSettings}
        onSettingsChange={setTipoSettings}
      />
    </div>
  );
}
