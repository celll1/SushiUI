"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Card from "../common/Card";
import Button from "../common/Button";
import Slider from "../common/Slider";
import Select from "../common/Select";
import NumberInput from "../common/NumberInput";
import Textarea from "../common/Textarea";
import Input from "../common/Input";
import GenerationQueue from "../common/GenerationQueue";
import ResizableColumns, {
  GENERATION_PREVIEW_QUEUE_SPLIT_KEY,
  GENERATION_WORKSPACE_SPLIT_KEY,
} from "../common/ResizableColumns";
import { fixFloatingPointParams } from "@/utils/numberUtils";
import {
  generateUpscale,
  fetchUpscalerModels,
  getSamplers,
  getScheduleTypes,
  UpscaleParams,
  UpscalerModelInfo,
  isGenerationStalledError,
} from "@/utils/api";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";
import { previewStorageKeys, saveImagePreview, clearImagePreview, outputExists, stripCacheBuster, withCacheBuster, imagePreviewGone } from "@/utils/previewStorage";
import { sendImageToImg2Img, sendImageToInpaint, sendImageToOutpaint } from "@/utils/sendHelpers";
import { useStartup } from "@/contexts/StartupContext";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";
import { wsClient, CFGMetrics } from "@/utils/websocket";

const DEFAULT_PARAMS: UpscaleParams = {
  upscaler_backend: "spandrel",
  upscaler_model: null,
  scale_factor: 2.0,
  pil_resample: "lanczos",
  tile_size: 512,
  tile_overlap: 32,
  rtx_vsr_quality: "high",
  unsharp_enable: false,
  unsharp_radius: 2.0,
  unsharp_percent: 100,
  unsharp_threshold: 3,
  // Diffusion tile upscale (upscaler_backend === "diffusion")
  prompt: "",
  negative_prompt: "",
  diffusion_denoising_strength: 0.3,
  steps: 20,
  cfg_scale: 7.0,
  sampler: "euler",
  schedule_type: "uniform",
  seed: -1,
  diffusion_pre_upscale_mode: "pil",
};

const STORAGE_KEY = "upscale_params";
const INPUT_IMAGE_STORAGE_KEY = "upscale_input_image";
const PREVIEW_STORAGE_KEY = "upscale_preview";
// Upscale is image-only (it takes an image and returns an image, with no
// video/audio route), so this panel only ever writes the image key. It still
// goes through the shared helper so every result-preview write in the app obeys
// the same mutual-exclusion rule (see utils/previewStorage.ts).
const PREVIEW_KEYS = previewStorageKeys(PREVIEW_STORAGE_KEY);

interface UpscalePanelProps {
  onImageGenerated?: (imageUrl: string) => void;
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint" | "outpaint" | "upscale") => void;
}

export default function UpscalePanel({ onTabChange, onImageGenerated }: UpscalePanelProps = {}) {
  const { isBackendReady, generationDefaults } = useStartup();
  const [params, setParams] = useState<UpscaleParams>(DEFAULT_PARAMS);
  const [isMounted, setIsMounted] = useState(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  const [inputImage, setInputImage] = useState<File | null>(null);
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null);
  const [inputImageSize, setInputImageSize] = useState<{ width: number; height: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);

  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [generatedImageInfo, setGeneratedImageInfo] = useState<{ width: number; height: number } | null>(null);
  const [generatedImageParams, setGeneratedImageParams] = useState<UpscaleParams | null>(null);

  const [upscalerModels, setUpscalerModels] = useState<UpscalerModelInfo[]>([]);
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);

  const [sendImage, setSendImage] = useState(true);

  const isGeneratingRef = useRef(isGenerating);
  useEffect(() => {
    isGeneratingRef.current = isGenerating;
  }, [isGenerating]);

  const handleProgress = useCallback((step: number, total: number, message: string, preview?: string, metrics?: CFGMetrics) => {
    if (isGeneratingRef.current) {
      setProgress(step);
      setTotalSteps(total);
    }
  }, []);

  useEffect(() => {
    wsClient.connect();
    wsClient.subscribe(handleProgress);
    return () => {
      wsClient.unsubscribe(handleProgress);
    };
  }, [handleProgress]);

  // Initial load from localStorage
  useEffect(() => {
    setIsMounted(true);

    const loadInitialData = async () => {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          setParams(fixFloatingPointParams(merged));
        } catch (error) {
          console.error("[Upscale] Failed to load saved params:", error);
        }
      }

      const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
      if (savedPreview) {
        setGeneratedImage(savedPreview);
      }

      const savedInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (savedInputRef) {
        try {
          const imageData = await loadTempImage(savedInputRef);
          if (imageData) {
            setInputImagePreview(imageData);
            const img = new Image();
            img.onload = () => {
              setInputImageSize({ width: img.width, height: img.height });
            };
            img.src = imageData;
          }
        } catch (error) {
          console.error("[Upscale] Failed to load input image:", error);
        }
      }

      setIsInitialLoad(false);
    };

    loadInitialData();
  }, []);

  // Fetch upscaler models
  const loadUpscalerModels = useCallback(async () => {
    try {
      const data = await fetchUpscalerModels();
      setUpscalerModels(data.models || []);
    } catch (error) {
      console.error("[Upscale] Failed to load upscaler models:", error);
    }
  }, []);

  useEffect(() => {
    loadUpscalerModels();
  }, [loadUpscalerModels]);

  // Fetch samplers / schedule types (diffusion backend, reused from Img2ImgPanel source)
  useEffect(() => {
    const loadSamplers = async () => {
      try {
        const data = await getSamplers();
        setSamplers(data.samplers);
      } catch (error) {
        console.error("[Upscale] Failed to load samplers:", error);
        setSamplers([
          { id: "euler", name: "Euler" },
          { id: "euler_ancestral", name: "Euler Ancestral" },
          { id: "heun", name: "Heun" },
          { id: "dpm_2", name: "DPM2" },
          { id: "dpm_2_ancestral", name: "DPM2 Ancestral" },
          { id: "lms", name: "LMS" },
          { id: "dpm_pp_2s_ancestral", name: "DPM++ 2S Ancestral" },
          { id: "dpm_pp_sde", name: "DPM++ SDE" },
          { id: "dpm_pp_2m", name: "DPM++ 2M" },
          { id: "dpm_pp_2m_sde", name: "DPM++ 2M SDE" },
          { id: "dpm_pp_3m_sde", name: "DPM++ 3M SDE" },
        ]);
      }
    };
    const loadScheduleTypes = async () => {
      try {
        const data = await getScheduleTypes();
        setScheduleTypes(data.schedule_types);
      } catch (error) {
        console.error("[Upscale] Failed to load schedule types:", error);
      }
    };
    loadSamplers();
    loadScheduleTypes();
  }, []);

  // Reload input image when notified from other panels / gallery
  useEffect(() => {
    const handleInputUpdate = async () => {
      const newInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (newInputRef) {
        try {
          const imageData = await loadTempImage(newInputRef);
          if (imageData) {
            setInputImagePreview(imageData);
            const img = new Image();
            img.onload = () => {
              setInputImageSize({ width: img.width, height: img.height });
            };
            img.src = imageData;
          }
        } catch (error) {
          console.error("[Upscale] Failed to reload input image:", error);
        }
      }
    };

    window.addEventListener("upscale_input_updated", handleInputUpdate);
    return () => {
      window.removeEventListener("upscale_input_updated", handleInputUpdate);
    };
  }, []);

  // Reload params when notified from other panels / gallery
  useEffect(() => {
    const handleParamsUpdate = () => {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          setParams(fixFloatingPointParams(merged));
        } catch (error) {
          console.error("[Upscale] Failed to parse params update:", error);
        }
      }
    };

    window.addEventListener("upscale_params_updated", handleParamsUpdate);
    return () => {
      window.removeEventListener("upscale_params_updated", handleParamsUpdate);
    };
  }, []);

  // Save params to localStorage
  useEffect(() => {
    if (isMounted && !isInitialLoad) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
    }
  }, [params, isMounted, isInitialLoad]);

  // Save preview to localStorage
  useEffect(() => {
    if (isMounted && generatedImage) {
      saveImagePreview(PREVIEW_KEYS, generatedImage);
    }
  }, [generatedImage, isMounted]);

  // Apply backend-fetched defaults when they arrive (only if no localStorage value exists)
  useEffect(() => {
    if (!generationDefaults) return;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      setParams(prev => ({ ...DEFAULT_PARAMS, ...(generationDefaults.upscale as Partial<UpscaleParams>) }));
    }
  }, [generationDefaults]);

  // Reload preview when backend becomes ready, verifying it is still there
  // first (outputs/ can be cleared, or the run deleted from the gallery) --
  // the same rule the video/audio previews follow in the panels that have
  // them. Non-`/outputs/` values (a data: URL, a blob:, a path served from
  // elsewhere) are left untouched: they cannot go missing server-side and must
  // never be stamped or discarded. The stamp is applied only to a URL that
  // verified, and it replaces any earlier stamp rather than appending.
  useEffect(() => {
    if (!isBackendReady) return;
    const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
    if (savedPreview && savedPreview.startsWith('/outputs/')) {
      const previewPath = stripCacheBuster(savedPreview);
      outputExists(previewPath).then((exists) => {
        if (!exists) {
          console.log("[Upscale] Stored preview image is gone, clearing:", previewPath);
          clearImagePreview(PREVIEW_KEYS);
          setGeneratedImage(null);
          return;
        }
        setGeneratedImage(withCacheBuster(previewPath));
      });
    }
  }, [isBackendReady]);

  const processImageFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file');
      return;
    }

    setInputImage(file);
    const reader = new FileReader();
    reader.onload = async (event) => {
      const preview = event.target?.result as string;
      setInputImagePreview(preview);
      if (isMounted) {
        try {
          const ref = await saveTempImage(preview);
          localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, ref);
        } catch (error) {
          console.error("[Upscale] Failed to save temp image:", error);
        }
      }

      const img = new Image();
      img.onload = () => {
        setInputImageSize({ width: img.width, height: img.height });
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

  const handleClearInputImage = async () => {
    setInputImage(null);
    setInputImagePreview(null);
    setInputImageSize(null);
    if (isMounted) {
      const ref = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (ref) {
        await deleteTempImageRef(ref);
        localStorage.removeItem(INPUT_IMAGE_STORAGE_KEY);
      }
    }
  };

  const outputDimensions = inputImageSize && params.scale_factor
    ? {
        width: Math.round(inputImageSize.width * params.scale_factor),
        height: Math.round(inputImageSize.height * params.scale_factor),
      }
    : null;

  const { addToQueue, startNextInQueue, completeCurrentItem, failCurrentItem, currentItem, queue } = useGenerationQueue();

  const handleAddToQueue = async () => {
    if (!inputImage && !inputImagePreview) {
      alert("Please upload an input image");
      return;
    }

    if (params.upscaler_backend === "spandrel" && !params.upscaler_model) {
      alert("Please select an upscaler model");
      return;
    }

    let imageBase64: string;
    const imageSource = inputImage || inputImagePreview;
    if (typeof imageSource === 'string') {
      imageBase64 = imageSource;
    } else if (imageSource instanceof File) {
      imageBase64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.readAsDataURL(imageSource);
      });
    } else {
      alert("Invalid input image");
      return;
    }

    addToQueue({
      type: "upscale",
      params: { ...params },
      inputImage: imageBase64,
      prompt: "Upscale",
    });
  };

  const processQueueRef = useRef<() => Promise<void>>();

  const processQueue = useCallback(async () => {
    if (isGenerating) return;

    const nextItem = startNextInQueue();
    if (!nextItem || nextItem.type !== "upscale") return;

    setIsGenerating(true);
    setProgress(0);
    setTotalSteps(0);
    setGeneratedImage(null);

    try {
      const inputImageToUse = nextItem.inputImage;
      if (!inputImageToUse) {
        throw new Error("No input image available for upscale generation");
      }

      const result = await generateUpscale(nextItem.params as UpscaleParams, inputImageToUse);
      const imageUrl = `/outputs/${result.image.filename}`;
      setGeneratedImage(imageUrl);
      setGeneratedImageInfo({ width: result.image.width, height: result.image.height });
      setGeneratedImageParams(nextItem.params as UpscaleParams);

      // Notify the page so the result reaches the shared top-right strip, the
      // same way every other generation panel does. An upscale result is a
      // plain image URL under /outputs/, so the strip renders it as a tile and
      // the full-screen viewer's prev/next walks it along with the rest.
      if (onImageGenerated) {
        onImageGenerated(imageUrl);
      }

      setIsGenerating(false);
      setProgress(0);
      completeCurrentItem();

      setTimeout(() => {
        if (processQueueRef.current) {
          processQueueRef.current();
        }
      }, 100);
    } catch (error: any) {
      console.error("[Upscale] Generation failed:", error);
      alert(isGenerationStalledError(error) ? error.message : "Upscale generation failed. Please check console for details.");

      setIsGenerating(false);
      setProgress(0);
      failCurrentItem();

      setTimeout(() => {
        if (processQueueRef.current) {
          processQueueRef.current();
        }
      }, 100);
    }
  }, [isGenerating, startNextInQueue, completeCurrentItem, failCurrentItem, onImageGenerated]);

  processQueueRef.current = processQueue;

  useEffect(() => {
    const hasPendingItems = queue.some(item => item.status === "pending" && item.type === "upscale");
    const isCurrentItemNull = currentItem === null;

    if (hasPendingItems && isCurrentItemNull && !isGenerating) {
      processQueue();
    }
  }, [queue, currentItem, isGenerating, processQueue]);

  const sendToTxt2Img = () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }
    if (onTabChange) {
      onTabChange("txt2img");
    }
  };

  const sendToImg2Img = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }
    if (sendImage) {
      try {
        await sendImageToImg2Img(generatedImage);
      } catch (error) {
        console.error("[Upscale] Failed to send image to img2img:", error);
      }
    }
    if (onTabChange) {
      onTabChange("img2img");
    }
  };

  const sendToInpaint = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }
    if (sendImage) {
      try {
        await sendImageToInpaint(generatedImage);
      } catch (error) {
        console.error("[Upscale] Failed to send image to inpaint:", error);
      }
    }
    if (onTabChange) {
      onTabChange("inpaint");
    }
  };

  const sendToOutpaint = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }
    if (sendImage) {
      try {
        await sendImageToOutpaint(generatedImage);
      } catch (error) {
        console.error("[Upscale] Failed to send image to outpaint:", error);
      }
    }
    if (onTabChange) {
      onTabChange("outpaint");
    }
  };

  const backendOptions = [
    { value: "pil", label: "PIL resize (no model)" },
    { value: "spandrel", label: "GAN/transformer model (.pth/.safetensors)" },
    { value: "rtx_vsr", label: "NVIDIA Video Effects SDK" },
    { value: "diffusion", label: "Diffusion (img2img tile)" },
  ];

  const diffusionPreUpscaleModeOptions = [
    { value: "pil", label: "PIL resize (uses Resample Filter below)" },
    { value: "model", label: "Model (uses Upscaler Model)" },
  ];

  const pilResampleOptions = [
    { value: "lanczos", label: "Lanczos" },
    { value: "bicubic", label: "Bicubic" },
    { value: "nearest", label: "Nearest" },
  ];

  const rtxQualityOptions = [
    { value: "low", label: "Low" },
    { value: "medium", label: "Medium" },
    { value: "high", label: "High" },
    { value: "ultra", label: "Ultra" },
  ];

  return (
    <ResizableColumns
      storageKey={GENERATION_WORKSPACE_SPLIT_KEY}
      label="Settings and output width"
      defaultPrimaryPercent={46}
      minPrimaryPercent={34}
      maxPrimaryPercent={66}
      minPrimaryPx={360}
      minSecondaryPx={540}
    >
      {/* Parameters Panel */}
      <div className="space-y-2.5">
        <Card title="Input Image">
          <div
            className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
              isDragging ? "border-blue-500 bg-blue-500/10" : "border-gray-700"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById("upscale-image-input")?.click()}
          >
            {inputImagePreview ? (
              <div className="space-y-2">
                <img src={inputImagePreview} alt="Input" className="max-h-64 mx-auto rounded" />
                {inputImageSize && (
                  <div className="text-xs text-gray-400">
                    {inputImageSize.width} x {inputImageSize.height}
                  </div>
                )}
                <Button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleClearInputImage();
                  }}
                  variant="secondary"
                  size="sm"
                >
                  Clear
                </Button>
              </div>
            ) : (
              <div className="text-gray-400 py-8">
                Drop image here or click to upload
              </div>
            )}
            <input
              id="upscale-image-input"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
          </div>
        </Card>

        <Card title="Upscaler">
          <Select
            label="Backend"
            value={params.upscaler_backend || "spandrel"}
            onChange={(e) => setParams({ ...params, upscaler_backend: e.target.value })}
            options={backendOptions}
          />

          {(params.upscaler_backend === "spandrel" ||
            (params.upscaler_backend === "diffusion" && params.diffusion_pre_upscale_mode === "model")) && (
            <Select
              label="Model"
              value={params.upscaler_model || ""}
              onChange={(e) => setParams({ ...params, upscaler_model: e.target.value })}
              options={[
                { value: "", label: "Select a model..." },
                ...upscalerModels.map((m) => ({ value: m.name, label: `${m.name} (${m.size_mb.toFixed(1)} MB)` })),
              ]}
            />
          )}

          <Slider
            label="Scale Factor"
            min={1.0}
            max={8.0}
            step={0.05}
            value={params.scale_factor ?? 2.0}
            onChange={(e) => setParams({ ...params, scale_factor: parseFloat(e.target.value) })}
          />
          {outputDimensions && (
            <div className="text-xs text-gray-400">
              Output: {outputDimensions.width} x {outputDimensions.height}
            </div>
          )}

          {(params.upscaler_backend === "pil" ||
            (params.upscaler_backend === "diffusion" && params.diffusion_pre_upscale_mode === "pil")) && (
            <Select
              label="Resample Filter"
              value={params.pil_resample || "lanczos"}
              onChange={(e) => setParams({ ...params, pil_resample: e.target.value })}
              options={pilResampleOptions}
            />
          )}

          {(params.upscaler_backend === "spandrel" || params.upscaler_backend === "diffusion") && (
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Tile Size</label>
                <NumberInput
                  label="Tile Size"
                  value={params.tile_size ?? 512}
                  onCommit={(v) => setParams({ ...params, tile_size: v })}
                  min={0}
                  max={4096}
                  step={64}
                  parse="int"
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Tile Overlap</label>
                <NumberInput
                  label="Tile Overlap"
                  value={params.tile_overlap ?? 32}
                  onCommit={(v) => setParams({ ...params, tile_overlap: v })}
                  min={0}
                  max={512}
                  step={8}
                  parse="int"
                  className="w-full"
                />
              </div>
            </div>
          )}

          {params.upscaler_backend === "rtx_vsr" && (
            <Select
              label="RTX VSR Quality"
              value={params.rtx_vsr_quality || "high"}
              onChange={(e) => setParams({ ...params, rtx_vsr_quality: e.target.value })}
              options={rtxQualityOptions}
            />
          )}
        </Card>

        {params.upscaler_backend === "diffusion" && (
          <Card title="Diffusion Tile Upscale">
            <p className="text-xs text-gray-500 mb-2">
              Uses the currently loaded checkpoint for per-tile img2img.
            </p>
            <Select
              label="Pre-upscale Mode"
              value={params.diffusion_pre_upscale_mode || "pil"}
              onChange={(e) => setParams({ ...params, diffusion_pre_upscale_mode: e.target.value })}
              options={diffusionPreUpscaleModeOptions}
            />
            <Textarea
              label="Prompt"
              value={params.prompt || ""}
              onChange={(e) => setParams({ ...params, prompt: e.target.value })}
              rows={3}
            />
            <Textarea
              label="Negative Prompt"
              value={params.negative_prompt || ""}
              onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
              rows={2}
            />
            <Slider
              label="Denoising Strength"
              min={0.05}
              max={0.9}
              step={0.01}
              value={params.diffusion_denoising_strength ?? 0.3}
              onChange={(e) => setParams({ ...params, diffusion_denoising_strength: parseFloat(e.target.value) })}
            />
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Select
                label="Sampler"
                options={samplers.map(s => ({ value: s.id, label: s.name }))}
                value={params.sampler || "euler"}
                onChange={(e) => setParams({ ...params, sampler: e.target.value })}
              />
              <Select
                label="Schedule Type"
                options={scheduleTypes.map(s => ({ value: s.id, label: s.name }))}
                value={params.schedule_type || "uniform"}
                onChange={(e) => setParams({ ...params, schedule_type: e.target.value })}
              />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Steps</label>
                <NumberInput
                  label="Steps"
                  value={params.steps ?? 20}
                  onCommit={(v) => setParams({ ...params, steps: v })}
                  min={1}
                  max={150}
                  step={1}
                  parse="int"
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">CFG Scale</label>
                <NumberInput
                  label="CFG Scale"
                  value={params.cfg_scale ?? 7.0}
                  onCommit={(v) => setParams({ ...params, cfg_scale: v })}
                  min={1.0}
                  max={30.0}
                  step={0.1}
                  parse="float"
                  className="w-full"
                />
              </div>
              <Input
                type="number"
                label="Seed"
                value={params.seed ?? -1}
                onChange={(e) => {
                  const parsed = parseInt(e.target.value);
                  setParams({ ...params, seed: Number.isNaN(parsed) ? -1 : parsed });
                }}
              />
            </div>
          </Card>
        )}

        <Card title="Unsharp Mask" collapsible defaultCollapsed={!params.unsharp_enable}>
          <label className="flex items-center gap-2 cursor-pointer mb-2">
            <input
              type="checkbox"
              checked={params.unsharp_enable ?? false}
              onChange={(e) => setParams({ ...params, unsharp_enable: e.target.checked })}
              className="rounded"
            />
            <span className="text-gray-300 text-sm">Enable</span>
          </label>
          {params.unsharp_enable && (
            <div className="space-y-2">
              <Slider
                label="Radius"
                min={0.1}
                max={10.0}
                step={0.1}
                value={params.unsharp_radius ?? 2.0}
                onChange={(e) => setParams({ ...params, unsharp_radius: parseFloat(e.target.value) })}
              />
              <Slider
                label="Percent"
                min={0}
                max={500}
                step={1}
                value={params.unsharp_percent ?? 100}
                onChange={(e) => setParams({ ...params, unsharp_percent: parseInt(e.target.value) })}
              />
              <Slider
                label="Threshold"
                min={0}
                max={255}
                step={1}
                value={params.unsharp_threshold ?? 3}
                onChange={(e) => setParams({ ...params, unsharp_threshold: parseInt(e.target.value) })}
              />
            </div>
          )}
        </Card>

        <Button
          onClick={handleAddToQueue}
          variant="primary"
          size="lg"
          className="w-full"
          disabled={!inputImage && !inputImagePreview}
        >
          Add to Queue
        </Button>
      </div>

      {/* Output Panel */}
      <ResizableColumns
        storageKey={GENERATION_PREVIEW_QUEUE_SPLIT_KEY}
        label="Output and queue width"
        defaultPrimaryPercent={68}
        minPrimaryPercent={55}
        maxPrimaryPercent={82}
        minPrimaryPx={300}
        minSecondaryPx={200}
        className="lg:h-[800px]"
      >
        <Card title="Output">
          {isGenerating && (
            <div className="mb-3">
              <div className="text-sm text-gray-400 mb-1">
                {totalSteps > 0 ? `Tile ${progress} / ${totalSteps}` : "Processing..."}
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: totalSteps > 0 ? `${(progress / totalSteps) * 100}%` : "0%" }}
                />
              </div>
            </div>
          )}

          {generatedImage ? (
            <div className="space-y-3">
              <img
                src={generatedImage}
                alt="Upscaled result"
                className="w-full rounded"
                onError={() => {
                  // The file went away while the panel was open -- show an
                  // empty preview rather than a broken image, the same
                  // backstop the video/audio players have elsewhere. Confirmed
                  // with a HEAD first, so a hot reload or a backend blip cannot
                  // discard a result that is still on disk (see helper).
                  imagePreviewGone(generatedImage, generatedImage).then((gone) => {
                    if (!gone) return;
                    console.warn("[Upscale] Preview image failed to load, clearing:", generatedImage);
                    clearImagePreview(PREVIEW_KEYS);
                    setGeneratedImage(null);
                    setGeneratedImageInfo(null);
                  });
                }}
              />
              {generatedImageInfo && (
                <div className="text-xs text-gray-400">
                  {generatedImageInfo.width} x {generatedImageInfo.height}
                </div>
              )}

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
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <Button onClick={sendToTxt2Img} variant="secondary" size="sm">
                  Send to txt2img
                </Button>
                <Button onClick={sendToImg2Img} variant="secondary" size="sm" disabled={!sendImage}>
                  Send to img2img
                </Button>
                <Button onClick={sendToInpaint} variant="secondary" size="sm" disabled={!sendImage}>
                  Send to inpaint
                </Button>
                <Button onClick={sendToOutpaint} variant="secondary" size="sm" disabled={!sendImage}>
                  Send to outpaint
                </Button>
              </div>
            </div>
          ) : (
            <div className="text-gray-500 text-sm py-8 text-center">
              No output yet
            </div>
          )}
        </Card>

        <div className="w-full">
          <GenerationQueue currentStep={progress} />
        </div>
      </ResizableColumns>
    </ResizableColumns>
  );
}
