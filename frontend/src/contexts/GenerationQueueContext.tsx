"use client";

import { createContext, useContext, useState, useCallback, ReactNode, useEffect, useRef } from "react";
import { GenerationParams, Img2ImgParams, InpaintParams, InpaintVideoParams, OutpaintParams, OutpaintVideoParams, OutpaintAudioParams, UpscaleParams, Txt2VidParams, Img2VidParams, Ref2VidParams, MiniMaxH3References, Txt2AudParams, Aud2AudParams } from "@/utils/api";
import { CFGMetrics, wsClient } from "@/utils/websocket";

export type GenerationPanelId = "txt2img" | "img2img" | "inpaint" | "outpaint" | "upscale";

export interface GenerationProgressSnapshot {
  itemId: string;
  step: number;
  totalSteps: number;
  message: string;
  previewImage?: string;
  cfgMetrics?: CFGMetrics;
  subProgress?: number;
}

export interface GenerationResultSnapshot {
  panel: GenerationPanelId;
  kind: "image" | "video" | "audio";
  url: string;
  playbackUrl?: string;
  info?: Record<string, number | string | undefined> | null;
  seed?: number | null;
  ancestralSeed?: number | null;
  params?: unknown;
  revision: number;
}

export interface QueueItem {
  id: string;
  type: "txt2img" | "img2img" | "inpaint" | "inpaint_vid" | "outpaint" | "outpaint_vid" | "outpaint_aud" | "upscale" | "txt2vid" | "img2vid" | "ref2vid" | "txt2aud" | "aud2aud";
  params: GenerationParams | Img2ImgParams | InpaintParams | InpaintVideoParams | OutpaintParams | OutpaintVideoParams | OutpaintAudioParams | UpscaleParams | Txt2VidParams | Img2VidParams | Ref2VidParams | Txt2AudParams | Aud2AudParams;
  inputImage?: string; // For img2img, inpaint, and outpaint
  // Server-cached latent to chain from instead of an image (loop-generation
  // decodeMode "final-only" latent passthrough; img2img only — set by the
  // previous step's response.latent_id when its loop_decode was "none").
  inputLatentId?: string;
  inputAudio?: File; // For aud2aud / outpaint_aud (reference clip; a File, unlike inputImage's base64 string)
  inputVideo?: File; // For outpaint_vid / inpaint_vid (uploaded clip; a File, mirrors inputAudio -- avoids a giant base64 string)
  // For outpaint_vid BRIDGE placement only: the second clip, preserved at the
  // END of the timeline, with the generated span between the two. Only an
  // architecture whose video_constraints.outpaint_placements contains "bridge"
  // (MiniMax-H3, which conditions on boundary frames and can anchor both ends
  // of a gap) accepts it; the backend refuses it on any other.
  bridgeVideo?: File;
  // For ref2vid (MiniMax-H3 ref2va) only: the reference uploads, IN THE ORDER
  // THE MODEL READS THEM. They ride on the item -- like inputAudio/inputVideo,
  // and unlike inputImage's base64 string -- so a queued request keeps the
  // references it was built with even after the panel's inputs change.
  references?: MiniMaxH3References;
  // For outpaint_vid (MiniMax-H3 ref2va, extend_forward only): optional
  // image references on top of the automatic source-clip video reference.
  // Images only -- this endpoint has no reference_videos/reference_audios
  // field, so this is a plain File[] rather than MiniMaxH3References.
  referenceImages?: File[];
  maskImage?: string; // For inpaint only
  status: "pending" | "generating" | "completed" | "failed";
  addedAt: number;
  prompt: string; // For display purposes
  loopGroupId?: string; // ID to group loop steps together
  loopStepIndex?: number; // Index of this step in the loop sequence
  isLoopStep?: boolean; // Whether this is a loop step (vs main generation)
  // Generate with the in-training model (training-preview). Captured per item at enqueue
  // time so loop steps (which may be processed by a different panel whose own
  // "use training model" checkbox is off) keep the base generation's choice.
  useTrainingModel?: boolean;
  trainingRunId?: number;
  startTime?: number; // When generation started (for timing)
  endTime?: number; // When generation completed (for timing)
}

interface GenerationQueueContextType {
  queue: QueueItem[];
  currentItem: QueueItem | null;
  addToQueue: (item: Omit<QueueItem, "id" | "status" | "addedAt">) => void;
  removeFromQueue: (id: string) => void;
  updateQueueItem: (id: string, updates: Partial<QueueItem>) => void;
  updateQueueItemByLoop: (loopGroupId: string, loopStepIndex: number, updates: Partial<QueueItem> | ((item: QueueItem) => Partial<QueueItem>)) => void;
  cancelLoopGroup: (loopGroupId: string) => void;
  cancelRelatedItems: (itemId: string) => void;
  startNextInQueue: (allowedTypes?: readonly QueueItem["type"][]) => QueueItem | null;
  completeCurrentItem: () => void;
  failCurrentItem: () => void;
  clearQueue: () => void;
  generateForever: boolean;
  setGenerateForever: (enabled: boolean) => void;
  progressSnapshot: GenerationProgressSnapshot | null;
  completedResults: Partial<Record<GenerationPanelId, GenerationResultSnapshot>>;
  publishCompletedResult: (result: Omit<GenerationResultSnapshot, "revision">) => void;
}

const GenerationQueueContext = createContext<GenerationQueueContextType | undefined>(undefined);

export function GenerationQueueProvider({ children }: { children: ReactNode }) {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [currentItem, setCurrentItem] = useState<QueueItem | null>(null);
  const [generateForever, setGenerateForever] = useState<boolean>(false);
  const [progressSnapshot, setProgressSnapshot] = useState<GenerationProgressSnapshot | null>(null);
  const [completedResults, setCompletedResults] = useState<Partial<Record<GenerationPanelId, GenerationResultSnapshot>>>({});

  // Use refs that are synchronously updated alongside state
  const queueRef = useRef<QueueItem[]>(queue);
  const currentItemRef = useRef<QueueItem | null>(currentItem);

  // Synchronously update refs whenever state changes
  queueRef.current = queue;
  currentItemRef.current = currentItem;

  useEffect(() => {
    const handleProgress = (
      step: number,
      totalSteps: number,
      message: string,
      previewImage?: string,
      cfgMetrics?: CFGMetrics,
      subProgress?: number,
    ) => {
      const item = currentItemRef.current;
      if (!item) return;
      setProgressSnapshot((previous) => ({
        itemId: item.id,
        step,
        totalSteps,
        message: message || "",
        previewImage: previewImage || (previous?.itemId === item.id ? previous.previewImage : undefined),
        cfgMetrics,
        subProgress,
      }));
    };

    wsClient.connect();
    wsClient.subscribe(handleProgress);
    return () => wsClient.unsubscribe(handleProgress);
  }, []);

  const publishCompletedResult = useCallback((result: Omit<GenerationResultSnapshot, "revision">) => {
    setCompletedResults((previous) => ({
      ...previous,
      [result.panel]: { ...result, revision: (previous[result.panel]?.revision ?? 0) + 1 },
    }));
  }, []);

  const addToQueue = useCallback((item: Omit<QueueItem, "id" | "status" | "addedAt">) => {
    const newItem: QueueItem = {
      ...item,
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      status: "pending",
      addedAt: Date.now(),
    };
    setQueue((prev) => [...prev, newItem]);
  }, []);

  const removeFromQueue = useCallback((id: string) => {
    setQueue((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const updateQueueItem = useCallback((id: string, updates: Partial<QueueItem>) => {
    setQueue((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, ...updates } : item
      )
    );
  }, []);

  const updateQueueItemByLoop = useCallback((loopGroupId: string, loopStepIndex: number, updates: Partial<QueueItem> | ((item: QueueItem) => Partial<QueueItem>)) => {
    setQueue((prev) => {
      const item = prev.find((item) =>
        item.loopGroupId === loopGroupId &&
        item.loopStepIndex === loopStepIndex
      );

      if (item) {
        console.log(`[QueueContext] updateQueueItemByLoop: Found item with loopGroupId=${loopGroupId}, loopStepIndex=${loopStepIndex}`);
        const actualUpdates = typeof updates === 'function' ? updates(item) : updates;
        return prev.map((i) =>
          i.id === item.id ? { ...i, ...actualUpdates} : i
        );
      } else {
        console.log(`[QueueContext] updateQueueItemByLoop: Item not found with loopGroupId=${loopGroupId}, loopStepIndex=${loopStepIndex}`);
        return prev;
      }
    });
  }, []);

  const cancelLoopGroup = useCallback((loopGroupId: string) => {
    console.log(`[QueueContext] Cancelling all pending items in loop group: ${loopGroupId}`);

    // Remove all pending items with this loopGroupId
    setQueue((prev) => prev.filter((item) =>
      !(item.loopGroupId === loopGroupId && item.status === "pending")
    ));

    // If currentItem is part of this loop group, clear its loopGroupId
    // to prevent trying to continue the cancelled loop sequence
    setCurrentItem((prev) => {
      if (prev && prev.loopGroupId === loopGroupId) {
        console.log(`[QueueContext] Clearing loopGroupId from currentItem to break loop sequence`);
        return { ...prev, loopGroupId: undefined, loopStepIndex: undefined };
      }
      return prev;
    });
  }, []);

  // Smart cancel: determine what to cancel based on item type and position
  const cancelRelatedItems = useCallback((itemId: string) => {
    const item = queueRef.current.find((i) => i.id === itemId);
    if (!item || !item.loopGroupId) {
      // No loop group - just remove this item
      console.log(`[QueueContext] Removing single item: ${itemId}`);
      removeFromQueue(itemId);
      return;
    }

    const { loopGroupId, isLoopStep, loopStepIndex } = item;
    const allItemsInGroup = queueRef.current.filter((i) => i.loopGroupId === loopGroupId);

    // Case 1: Cancelling main generation (Base)
    if (!isLoopStep) {
      console.log(`[QueueContext] Cancelling main generation and all related loop steps for group: ${loopGroupId}`);
      // Remove all items in this loop group (main + all loop steps)
      setQueue((prev) => prev.filter((i) => i.loopGroupId !== loopGroupId));

      // Clear loopGroupId from currentItem if it belongs to this group
      setCurrentItem((prev) => {
        if (prev && prev.loopGroupId === loopGroupId) {
          return { ...prev, loopGroupId: undefined, loopStepIndex: undefined };
        }
        return prev;
      });
      return;
    }

    // Case 2 & 3: Cancelling loop step
    const maxLoopStepIndex = Math.max(
      ...allItemsInGroup
        .filter((i) => i.isLoopStep && i.loopStepIndex !== undefined)
        .map((i) => i.loopStepIndex!)
    );

    // Case 2: Cancelling first or middle loop step - cancel this and all following steps
    if (loopStepIndex !== undefined && loopStepIndex < maxLoopStepIndex) {
      console.log(`[QueueContext] Cancelling loop step ${loopStepIndex} and all following steps in group: ${loopGroupId}`);
      setQueue((prev) => prev.filter((i) =>
        !(i.loopGroupId === loopGroupId &&
          i.isLoopStep === true &&
          i.loopStepIndex !== undefined &&
          i.loopStepIndex >= loopStepIndex)
      ));

      // Clear loopGroupId from currentItem if it's a later step
      setCurrentItem((prev) => {
        if (prev &&
            prev.loopGroupId === loopGroupId &&
            prev.isLoopStep === true &&
            prev.loopStepIndex !== undefined &&
            prev.loopStepIndex >= loopStepIndex) {
          return { ...prev, loopGroupId: undefined, loopStepIndex: undefined };
        }
        return prev;
      });
      return;
    }

    // Case 3: Cancelling last loop step - only cancel this step
    console.log(`[QueueContext] Cancelling only the last loop step ${loopStepIndex} in group: ${loopGroupId}`);
    removeFromQueue(itemId);
  }, [removeFromQueue]);

  const startNextInQueue = useCallback((allowedTypes?: readonly QueueItem["type"][]) => {
    // Access latest values from refs (synchronously updated)
    const currentQueue = queueRef.current;
    const currentItemValue = currentItemRef.current;

    console.log("[QueueContext] startNextInQueue - current queue:", currentQueue);
    console.log("[QueueContext] currentItem:", currentItemValue);

    if (currentItemValue?.status === "generating") {
      console.log("[QueueContext] A generation is already active, not starting another item");
      return null;
    }

    let nextItem: QueueItem | undefined;

    // If current item is part of a loop group, prioritize next step in same group
    if (currentItemValue?.loopGroupId) {
      const currentLoopGroupId = currentItemValue.loopGroupId;
      const currentLoopStepIndex = currentItemValue.loopStepIndex ?? -1;

      // Find next step in the same loop group
      nextItem = currentQueue.find((item) =>
        item.status === "pending" &&
        (!allowedTypes || allowedTypes.includes(item.type)) &&
        item.loopGroupId === currentLoopGroupId &&
        (item.loopStepIndex ?? 0) === currentLoopStepIndex + 1
      );

      console.log("[QueueContext] Looking for next loop step in group:", currentLoopGroupId, "after index:", currentLoopStepIndex);
      console.log("[QueueContext] Found loop step:", nextItem);
    }

    // If no loop step found, get next pending item in queue order
    if (!nextItem) {
      nextItem = currentQueue.find((item) =>
        item.status === "pending" && (!allowedTypes || allowedTypes.includes(item.type)));
      console.log("[QueueContext] Found next pending item:", nextItem);
    }

    if (nextItem) {
      // Auto-enable keep_models_hot whenever another pending item remains in the
      // queue after this one. The backend safely invalidates/offloads GPU-resident
      // components if the next dispatched item turns out to use a different model,
      // so "is there a next queued item at all" is a sufficient signal here -- the
      // frontend does not need to compare models itself. Only the last item of a
      // back-to-back run is dispatched with keep_models_hot=false, which tells the
      // backend to release VRAM once that generation completes.
      const hasNext = currentQueue.some(
        (item) => item.id !== nextItem!.id && item.status === "pending"
      );
      const supportsKeepModelsHot =
        nextItem.type === "txt2img" || nextItem.type === "img2img" || nextItem.type === "inpaint";

      const updatedItem = {
        ...nextItem,
        status: "generating" as const,
        startTime: Date.now(),
        params: supportsKeepModelsHot
          ? { ...nextItem.params, keep_models_hot: hasNext }
          : nextItem.params,
      };
      currentItemRef.current = updatedItem;
      setCurrentItem(updatedItem);
      setProgressSnapshot(null);

      // Update the item in queue to generating status
      setQueue((prev) =>
        prev.map((item) =>
          item.id === nextItem!.id ? updatedItem : item
        )
      );

      console.log("[QueueContext] Returning updated item:", updatedItem);
      return updatedItem;
    }

    console.log("[QueueContext] No pending items found, setting currentItem to null");
    setCurrentItem(null);
    return null;
  }, []); // Empty deps - uses refs for latest values

  const completeCurrentItem = useCallback(() => {
    const currentItemValue = currentItemRef.current;
    if (!currentItemValue) return;

    console.log("[QueueContext] Completing item:", currentItemValue.id);
    // Mark completion time before removing
    const endTime = Date.now();
    const elapsedMs = currentItemValue.startTime ? endTime - currentItemValue.startTime : 0;
    console.log(`[QueueContext] Generation took ${(elapsedMs / 1000).toFixed(2)}s`);

    // Remove completed item from queue
    setQueue((prev) => prev.filter((item) => item.id !== currentItemValue.id));
    currentItemRef.current = null;
    setCurrentItem(null);
    setProgressSnapshot(null);
  }, []); // Empty deps - uses refs

  const failCurrentItem = useCallback(() => {
    const currentItemValue = currentItemRef.current;
    if (!currentItemValue) return;

    console.log("[QueueContext] Failing item:", currentItemValue.id);
    // Mark as failed but keep in queue for user to see
    setQueue((prev) =>
      prev.map((item) =>
        item.id === currentItemValue.id ? { ...item, status: "failed" as const } : item
      )
    );
    currentItemRef.current = null;
    setCurrentItem(null);
    setProgressSnapshot(null);
  }, []); // Empty deps - uses refs

  const clearQueue = useCallback(() => {
    setQueue([]);
    queueRef.current = [];
    currentItemRef.current = null;
    setCurrentItem(null);
    setProgressSnapshot(null);
  }, []);

  return (
    <GenerationQueueContext.Provider
      value={{
        queue,
        currentItem,
        addToQueue,
        removeFromQueue,
        updateQueueItem,
        updateQueueItemByLoop,
        cancelLoopGroup,
        cancelRelatedItems,
        startNextInQueue,
        completeCurrentItem,
        failCurrentItem,
        clearQueue,
        generateForever,
        setGenerateForever,
        progressSnapshot,
        completedResults,
        publishCompletedResult,
      }}
    >
      {children}
    </GenerationQueueContext.Provider>
  );
}

export function useGenerationQueue() {
  const context = useContext(GenerationQueueContext);
  if (!context) {
    throw new Error("useGenerationQueue must be used within a GenerationQueueProvider");
  }
  return context;
}
