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
import { generateTxt2Vid, Txt2VidParams } from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";
import { wsClient, CFGMetrics } from "@/utils/websocket";

const DEFAULT_PARAMS: Txt2VidParams = {
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

const STORAGE_KEY = "txt2vid_params";
const PREVIEW_STORAGE_KEY = "txt2vid_preview";

// num_frames must be 8k+1 (LTX-2.3). Offer common lengths.
const FRAME_OPTIONS = [9, 17, 25, 33, 49, 65, 81, 97, 121].map((n) => ({
  value: String(n),
  label: String(n),
}));

interface Txt2VidPanelProps {
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint" | "upscale" | "txt2vid" | "img2vid") => void;
}

export default function Txt2VidPanel({ onTabChange }: Txt2VidPanelProps = {}) {
  const { isBackendReady, generationDefaults } = useStartup();
  const [params, setParams] = useState<Txt2VidParams>(DEFAULT_PARAMS);
  const [isMounted, setIsMounted] = useState(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);

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

    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        const merged = { ...DEFAULT_PARAMS, ...parsed };
        setParams(fixFloatingPointParams(merged) as Txt2VidParams);
      } catch (error) {
        console.error("[Txt2Vid] Failed to load saved params:", error);
      }
    }

    const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
    if (savedPreview) {
      setGeneratedVideo(savedPreview);
    }

    setIsInitialLoad(false);
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
      setParams((prev) => ({ ...DEFAULT_PARAMS, ...(generationDefaults.txt2vid as Partial<Txt2VidParams>) }));
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

  const { addToQueue, startNextInQueue, completeCurrentItem, failCurrentItem, currentItem, queue } = useGenerationQueue();

  const handleAddToQueue = () => {
    if (!params.prompt || params.prompt.trim() === "") {
      alert("Please enter a prompt");
      return;
    }
    addToQueue({
      type: "txt2vid",
      params: { ...params },
      prompt: params.prompt,
    });
  };

  const processQueueRef = useRef<() => Promise<void>>();

  const processQueue = useCallback(async () => {
    if (isGenerating) return;

    const nextItem = startNextInQueue();
    if (!nextItem || nextItem.type !== "txt2vid") return;

    setIsGenerating(true);
    setProgress(0);
    setTotalSteps(0);
    setGeneratedVideo(null);

    try {
      const result = await generateTxt2Vid(nextItem.params as Txt2VidParams);
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
      console.error("[Txt2Vid] Generation failed:", error);
      alert("txt2vid generation failed. Please check console for details.");

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
    const hasPendingItems = queue.some((item) => item.status === "pending" && item.type === "txt2vid");
    const isCurrentItemNull = currentItem === null;

    if (hasPendingItems && isCurrentItemNull && !isGenerating) {
      processQueue();
    }
  }, [queue, currentItem, isGenerating, processQueue]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Parameters Panel */}
      <div className="space-y-4">
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
                  {generatedInfo.duration != null && Number.isFinite(Number(generatedInfo.duration)) && <span> · {Number(generatedInfo.duration).toFixed(2)}s</span>}
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
