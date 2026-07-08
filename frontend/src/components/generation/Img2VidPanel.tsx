"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Card from "../common/Card";
import Button from "../common/Button";
import Select from "../common/Select";
import NumberInput from "../common/NumberInput";
import Textarea from "../common/Textarea";
import Input from "../common/Input";
import GenerationQueue from "../common/GenerationQueue";
import { fixFloatingPointParams } from "@/utils/numberUtils";
import { generateImg2Vid, Img2VidParams } from "@/utils/api";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";
import { useStartup } from "@/contexts/StartupContext";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";
import { wsClient, CFGMetrics } from "@/utils/websocket";

const DEFAULT_PARAMS: Img2VidParams = {
  prompt: "",
  negative_prompt: "",
  width: 768,
  height: 512,
  num_frames: 121,
  frame_rate: 24.0,
  num_inference_steps: 8,
  guidance_scale: 1.0,
  seed: -1,
  num_videos_per_prompt: 1,
  max_sequence_length: 1024,
  audio_enable: true,
};

const STORAGE_KEY = "img2vid_params";
const PREVIEW_STORAGE_KEY = "img2vid_preview";
// P3b gallery frame-grab writes a temp-image ref here and dispatches
// "img2vid_input_updated" (see sendImageToImg2Vid in sendHelpers.ts).
const INPUT_IMAGE_STORAGE_KEY = "img2vid_input_image";

// num_frames must be 8k+1 (LTX-2.3). Offer common lengths.
const FRAME_OPTIONS = [9, 17, 25, 33, 49, 65, 81, 97, 121].map((n) => ({
  value: String(n),
  label: String(n),
}));

interface Img2VidPanelProps {
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint" | "upscale" | "txt2vid" | "img2vid") => void;
}

export default function Img2VidPanel({ onTabChange }: Img2VidPanelProps = {}) {
  const { isBackendReady, generationDefaults } = useStartup();
  const [params, setParams] = useState<Img2VidParams>(DEFAULT_PARAMS);
  const [isMounted, setIsMounted] = useState(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  const [inputImage, setInputImage] = useState<File | null>(null);
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null);
  const [inputImageSize, setInputImageSize] = useState<{ width: number; height: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);

  const [generatedVideo, setGeneratedVideo] = useState<string | null>(null);
  const [generatedInfo, setGeneratedInfo] = useState<{ num_frames?: number; fps?: number; duration?: number } | null>(null);

  const isGeneratingRef = useRef(isGenerating);
  useEffect(() => {
    isGeneratingRef.current = isGenerating;
  }, [isGenerating]);

  const handleProgress = useCallback((step: number, total: number, _message: string, _preview?: string, _metrics?: CFGMetrics) => {
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
          setParams(fixFloatingPointParams(merged) as Img2VidParams);
        } catch (error) {
          console.error("[Img2Vid] Failed to load saved params:", error);
        }
      }

      const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
      if (savedPreview) {
        setGeneratedVideo(savedPreview);
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
          console.error("[Img2Vid] Failed to load input image:", error);
        }
      }

      setIsInitialLoad(false);
    };

    loadInitialData();
  }, []);

  // Reload keyframe when notified from other panels / gallery (P3b frame-grab)
  useEffect(() => {
    const handleInputUpdate = async () => {
      const newInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (newInputRef) {
        try {
          const imageData = await loadTempImage(newInputRef);
          if (imageData) {
            setInputImage(null);
            setInputImagePreview(imageData);
            const img = new Image();
            img.onload = () => {
              setInputImageSize({ width: img.width, height: img.height });
            };
            img.src = imageData;
          }
        } catch (error) {
          console.error("[Img2Vid] Failed to reload input image:", error);
        }
      }
    };

    window.addEventListener("img2vid_input_updated", handleInputUpdate);
    return () => {
      window.removeEventListener("img2vid_input_updated", handleInputUpdate);
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
    if (isMounted && generatedVideo) {
      localStorage.setItem(PREVIEW_STORAGE_KEY, generatedVideo);
    }
  }, [generatedVideo, isMounted]);

  // Apply backend-fetched defaults when they arrive (only if no localStorage value exists)
  useEffect(() => {
    if (!generationDefaults) return;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      setParams((prev) => ({ ...DEFAULT_PARAMS, ...(generationDefaults.img2vid as Partial<Img2VidParams>) }));
    }
  }, [generationDefaults]);

  // Reload preview when backend becomes ready
  useEffect(() => {
    if (!isBackendReady) return;
    const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
    if (savedPreview && savedPreview.startsWith("/outputs/")) {
      setGeneratedVideo(`${savedPreview}?t=${Date.now()}`);
    }
  }, [isBackendReady]);

  const processImageFile = (file: File) => {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file");
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
          console.error("[Img2Vid] Failed to save temp image:", error);
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

  const { addToQueue, startNextInQueue, completeCurrentItem, failCurrentItem, currentItem, queue } = useGenerationQueue();

  const handleAddToQueue = async () => {
    if (!params.prompt || params.prompt.trim() === "") {
      alert("Please enter a prompt");
      return;
    }
    if (!inputImage && !inputImagePreview) {
      alert("Please upload a keyframe image");
      return;
    }

    let imageBase64: string;
    const imageSource = inputImage || inputImagePreview;
    if (typeof imageSource === "string") {
      imageBase64 = imageSource;
    } else if (imageSource instanceof File) {
      imageBase64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.readAsDataURL(imageSource);
      });
    } else {
      alert("Invalid keyframe image");
      return;
    }

    addToQueue({
      type: "img2vid",
      params: { ...params },
      inputImage: imageBase64,
      prompt: params.prompt,
    });
  };

  const processQueueRef = useRef<() => Promise<void>>();

  const processQueue = useCallback(async () => {
    if (isGenerating) return;

    const nextItem = startNextInQueue();
    if (!nextItem || nextItem.type !== "img2vid") return;

    setIsGenerating(true);
    setProgress(0);
    setTotalSteps(0);
    setGeneratedVideo(null);

    try {
      const inputImageToUse = nextItem.inputImage;
      if (!inputImageToUse) {
        throw new Error("No keyframe image available for img2vid generation");
      }

      const result = await generateImg2Vid(nextItem.params as Img2VidParams, inputImageToUse);
      const videoUrl = `/outputs/${result.image.filename}`;
      setGeneratedVideo(videoUrl);
      setGeneratedInfo({
        num_frames: result.image.num_frames,
        fps: result.image.fps,
        duration: result.image.duration,
      });

      setIsGenerating(false);
      setProgress(0);
      completeCurrentItem();

      setTimeout(() => {
        if (processQueueRef.current) {
          processQueueRef.current();
        }
      }, 100);
    } catch (error: any) {
      console.error("[Img2Vid] Generation failed:", error);
      alert("img2vid generation failed. Please check console for details.");

      setIsGenerating(false);
      setProgress(0);
      failCurrentItem();

      setTimeout(() => {
        if (processQueueRef.current) {
          processQueueRef.current();
        }
      }, 100);
    }
  }, [isGenerating, startNextInQueue, completeCurrentItem, failCurrentItem]);

  processQueueRef.current = processQueue;

  useEffect(() => {
    const hasPendingItems = queue.some((item) => item.status === "pending" && item.type === "img2vid");
    const isCurrentItemNull = currentItem === null;

    if (hasPendingItems && isCurrentItemNull && !isGenerating) {
      processQueue();
    }
  }, [queue, currentItem, isGenerating, processQueue]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Parameters Panel */}
      <div className="space-y-4">
        <Card title="Keyframe Image">
          <div
            className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
              isDragging ? "border-blue-500 bg-blue-500/10" : "border-gray-700"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById("img2vid-image-input")?.click()}
          >
            {inputImagePreview ? (
              <div className="space-y-2">
                <img src={inputImagePreview} alt="Keyframe" className="max-h-64 mx-auto rounded" />
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
              id="img2vid-image-input"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
          </div>
        </Card>

        <Card title="Prompt">
          <Textarea
            label="Prompt"
            value={params.prompt || ""}
            onChange={(e) => setParams({ ...params, prompt: e.target.value })}
            rows={4}
          />
          <Textarea
            label="Negative Prompt"
            value={params.negative_prompt || ""}
            onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
            rows={2}
          />
        </Card>

        <Card title="Video">
          <div className="grid grid-cols-2 gap-2">
            <NumberInput
              label="Width (÷32)"
              value={params.width ?? 768}
              onCommit={(v) => setParams({ ...params, width: v })}
              min={32}
              max={2048}
              step={32}
              parse="int"
            />
            <NumberInput
              label="Height (÷32)"
              value={params.height ?? 512}
              onCommit={(v) => setParams({ ...params, height: v })}
              min={32}
              max={2048}
              step={32}
              parse="int"
            />
          </div>

          <Select
            label="Frames (8k+1)"
            value={String(params.num_frames ?? 121)}
            onChange={(e) => setParams({ ...params, num_frames: parseInt(e.target.value) })}
            options={FRAME_OPTIONS}
          />

          <NumberInput
            label="Frame Rate (fps)"
            value={params.frame_rate ?? 24.0}
            onCommit={(v) => setParams({ ...params, frame_rate: v })}
            min={1}
            max={60}
            step={1}
            parse="float"
          />

          <label className="flex items-center gap-2 cursor-pointer mt-2">
            <input
              type="checkbox"
              checked={params.audio_enable ?? true}
              onChange={(e) => setParams({ ...params, audio_enable: e.target.checked })}
              className="rounded"
            />
            <span className="text-gray-300 text-sm">Audio</span>
          </label>
        </Card>

        <Card title="Sampling">
          <p className="text-xs text-gray-500 mb-2">Distilled: 8 steps</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <NumberInput
              label="Steps"
              value={params.num_inference_steps ?? 8}
              onCommit={(v) => setParams({ ...params, num_inference_steps: v })}
              min={1}
              max={100}
              step={1}
              parse="int"
            />
            <NumberInput
              label="Guidance Scale"
              value={params.guidance_scale ?? 1.0}
              onCommit={(v) => setParams({ ...params, guidance_scale: v })}
              min={0}
              max={20}
              step={0.1}
              parse="float"
            />
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
          <div className="grid grid-cols-2 gap-2 mt-2">
            <NumberInput
              label="Videos per Prompt"
              value={params.num_videos_per_prompt ?? 1}
              onCommit={(v) => setParams({ ...params, num_videos_per_prompt: v })}
              min={1}
              max={8}
              step={1}
              parse="int"
            />
            <NumberInput
              label="Max Sequence Length"
              value={params.max_sequence_length ?? 1024}
              onCommit={(v) => setParams({ ...params, max_sequence_length: v })}
              min={128}
              max={4096}
              step={128}
              parse="int"
            />
          </div>
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
      <div className="space-y-4">
        <Card title="Output">
          {isGenerating && (
            <div className="mb-3">
              <div className="text-sm text-gray-400 mb-1">
                {totalSteps > 0 ? `Step ${progress} / ${totalSteps}` : "Processing..."}
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: totalSteps > 0 ? `${(progress / totalSteps) * 100}%` : "0%" }}
                />
              </div>
            </div>
          )}

          {generatedVideo ? (
            <div className="space-y-3">
              <video
                src={generatedVideo}
                className="w-full rounded"
                controls
                loop
                muted
                autoPlay
                playsInline
              />
              {generatedInfo && (
                <div className="text-xs text-gray-400">
                  {generatedInfo.num_frames != null && <span>{generatedInfo.num_frames} frames</span>}
                  {generatedInfo.fps != null && <span> · {generatedInfo.fps} fps</span>}
                  {generatedInfo.duration != null && <span> · {generatedInfo.duration.toFixed(2)}s</span>}
                </div>
              )}
            </div>
          ) : (
            <div className="text-gray-500 text-sm py-8 text-center">
              No output yet
            </div>
          )}
        </Card>

        <GenerationQueue currentStep={progress} />
      </div>
    </div>
  );
}
