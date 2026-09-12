"use client";

import { useEffect, useRef, type MutableRefObject } from "react";
import {
  queueItemBelongsToPanel,
  type GenerationFailureSnapshot,
  type GenerationPanelId,
  type GenerationProgressSnapshot,
  type QueueItem,
} from "@/contexts/GenerationQueueContext";

interface ProgressSetters {
  setIsGenerating: (value: boolean) => void;
  setProgress: (value: number) => void;
  setTotalSteps: (value: number) => void;
  setProgressMessage: (value: string) => void;
  setPreviewImage: (value: string | null) => void;
}

interface GenerationPanelProgressOptions extends ProgressSetters {
  panel: GenerationPanelId;
  currentItem: QueueItem | null;
  progressSnapshot: GenerationProgressSnapshot | null;
  reportSubProgress: (step: number, subProgress?: number) => void;
  onItemStart: (item: QueueItem) => void;
  isOwnRunRef?: MutableRefObject<boolean>;
}

export function initialGenerationTotalSteps(
  item: QueueItem,
  panel: GenerationPanelId,
): number {
  const params = item.params as {
    steps?: number;
    denoising_strength?: number;
    num_inference_steps?: number;
    inference_steps?: number;
  };
  if (item.type === "txt2aud" || item.type === "aud2aud" || item.type === "outpaint_aud") {
    return params.num_inference_steps || params.inference_steps || 8;
  }
  if (item.type === "ref2vid") {
    return params.num_inference_steps || (panel === "img2img" ? 20 : 8);
  }
  if (["txt2vid", "img2vid", "chain_vid", "inpaint_vid", "outpaint_vid"].includes(item.type)) {
    return params.num_inference_steps || 8;
  }
  if (item.type === "txt2img") return params.steps || 20;
  const denoisingStrength = panel === "outpaint"
    ? (params.denoising_strength ?? 1.0)
    : (params.denoising_strength || 0.75);
  return Math.ceil((params.steps || 20) * denoisingStrength);
}

export function useGenerationPanelProgress({
  panel,
  currentItem,
  progressSnapshot,
  reportSubProgress,
  onItemStart,
  isOwnRunRef,
  setIsGenerating,
  setProgress,
  setTotalSteps,
  setProgressMessage,
  setPreviewImage,
}: GenerationPanelProgressOptions): void {
  const clearedForItemRef = useRef<string | null>(null);
  const onItemStartRef = useRef(onItemStart);
  onItemStartRef.current = onItemStart;

  useEffect(() => {
    if (!queueItemBelongsToPanel(currentItem, panel)) {
      if (isOwnRunRef) isOwnRunRef.current = false;
      setIsGenerating(false);
      return;
    }
    if (isOwnRunRef) isOwnRunRef.current = true;
    setIsGenerating(true);
    if (clearedForItemRef.current !== currentItem.id) {
      clearedForItemRef.current = currentItem.id;
      onItemStartRef.current(currentItem);
      setProgress(0);
      setProgressMessage("");
      setPreviewImage(null);
      setTotalSteps(initialGenerationTotalSteps(currentItem, panel));
    }
    if (progressSnapshot?.itemId !== currentItem.id) return;
    setProgress(progressSnapshot.step);
    setTotalSteps(progressSnapshot.totalSteps);
    setProgressMessage(progressSnapshot.message);
    reportSubProgress(progressSnapshot.step, progressSnapshot.subProgress);
    if (progressSnapshot.previewImage) {
      setPreviewImage(progressSnapshot.previewImage);
    }
  }, [
    currentItem,
    isOwnRunRef,
    panel,
    progressSnapshot,
    reportSubProgress,
    setIsGenerating,
    setPreviewImage,
    setProgress,
    setProgressMessage,
    setTotalSteps,
  ]);
}

interface RestoreImageOnCancelOptions {
  panel: GenerationPanelId;
  lastFailure: GenerationFailureSnapshot | null;
  previousImageRef: MutableRefObject<string | null>;
  setGeneratedImage: (value: string | null) => void;
  setPreviewImage: (value: string | null) => void;
}

export function useRestoreImageOnCancel({
  panel,
  lastFailure,
  previousImageRef,
  setGeneratedImage,
  setPreviewImage,
}: RestoreImageOnCancelOptions): void {
  const lastFailureRevisionRef = useRef(0);

  useEffect(() => {
    if (!lastFailure || lastFailure.panel !== panel) return;
    if (lastFailure.revision === lastFailureRevisionRef.current) return;
    lastFailureRevisionRef.current = lastFailure.revision;
    if (!lastFailure.cancelled) return;
    if (localStorage.getItem("restore_image_on_cancel") !== "true") return;
    if (previousImageRef.current) {
      setGeneratedImage(previousImageRef.current);
      setPreviewImage(null);
    }
  }, [lastFailure, panel, previousImageRef, setGeneratedImage, setPreviewImage]);
}
