"use client";

import { createContext, useContext, useState, useCallback, ReactNode, useEffect, useRef } from "react";
import { GenerationParams, Img2ImgParams, InpaintParams, InpaintVideoParams, OutpaintParams, OutpaintVideoParams, OutpaintAudioParams, UpscaleParams, Txt2VidParams, Img2VidParams, Ref2VidParams, MiniMaxH3References, Txt2AudParams, Aud2AudParams } from "@/utils/api";
// Type-only (videoChain.ts imports QueueItem from here the same way), so this
// mutual reference is erased at compile time and creates no runtime cycle.
import type { ChainDriftPause } from "@/utils/videoChain";
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
  type: "txt2img" | "img2img" | "inpaint" | "inpaint_vid" | "outpaint" | "outpaint_vid" | "outpaint_aud" | "upscale" | "txt2vid" | "img2vid" | "ref2vid" | "txt2aud" | "aud2aud" | "chain_vid";
  params: GenerationParams | Img2ImgParams | InpaintParams | InpaintVideoParams | OutpaintParams | OutpaintVideoParams | OutpaintAudioParams | UpscaleParams | Txt2VidParams | Img2VidParams | Ref2VidParams | Txt2AudParams | Aud2AudParams;
  inputImage?: string; // For img2img, inpaint, and outpaint
  // Server-cached latent to chain from instead of an image (loop-generation
  // decodeMode "final-only" latent passthrough; img2img only — set by the
  // previous step's response.latent_id when its loop_decode was "none").
  inputLatentId?: string;
  inputAudio?: File; // For aud2aud / outpaint_aud (reference clip; a File, unlike inputImage's base64 string)
  inputVideo?: File; // For outpaint_vid / inpaint_vid (uploaded clip; a File, mirrors inputAudio -- avoids a giant base64 string)
  // For inpaint_vid spatial mask timelines: the manifest stays in params and
  // the referenced PNG files ride on the queued item by stable id.
  spatialMaskFiles?: Array<{ id: string; file: File }>;
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
  // Opt-in video-length chaining (Txt2Img/Img2Img "chain_vid" loop steps,
  // videoChain.ts): the overall clip length this chain is working toward, and
  // the accumulated frame count the PREVIOUS segment in the chain actually
  // reported (not the planned value -- this is what a "did this segment make
  // forward progress" check compares the segment's own reported frame count
  // against). Present on every item of a chain (main segment + chain_vid
  // steps) once video-length chaining has claimed the loop group; absent on
  // every other queue item.
  chainTargetFrames?: number;
  chainPreviousFrames?: number;
  // The per-segment length the chain was built with (chain_segment_frames at
  // enqueue time; see api.ts's chainSegmentCap). Persisted on the item, not
  // re-read from panel state, because `advanceVideoChain` (videoChain.ts)
  // computes each continuation's `total_frames` well after enqueue -- a chain
  // already enqueued is frozen at enqueue time and must not be retargeted by
  // a later change to the panel's segment-length control.
  chainSegmentFrames?: number | null;
  // Chain Manifest provenance for a segment planned by POST /video-chain/plan:
  // the chain the segment belongs to, the hash of the plan it was compiled
  // from, and which manifest segment this item is (0 = the main item, so a
  // `chain_vid` step at loopStepIndex n is segment n+1). Absent on a chain run
  // without a manifest (legacy repeat) and on every non-chain item.
  chainManifestId?: string;
  chainPlanHash?: string;
  chainSegmentIndex?: number;
  // Design §4.1 (scratchpad/video_chain_context_design.md): the manifest's
  // planned accumulated frame count at the END of THIS item's own segment
  // (`VideoChainSegment.owned_end_frame` for `chainSegmentIndex`), and the
  // drift tolerance the manifest was planned with
  // (`VideoChainManifest.chain_drift_tolerance_frames`, itself sourced from
  // `backend/api/param_defaults.py`'s `VIDEO_CHAIN_DEFAULTS`). Both frozen at
  // enqueue time, same as every other chain field on this item. Present only
  // when this chain has a manifest (absent in legacy-repeat mode, where §4.1
  // does not apply because there is no per-segment plan to drift from).
  // `advanceVideoChain` (videoChain.ts) compares this against the segment's
  // ACTUAL reported accumulated frame count when it finishes, before feeding
  // the next continuation.
  chainPlannedAccumulatedFrames?: number;
  chainDriftToleranceFrames?: number;
  // Design §7.2c: the frames THIS item's own segment was planned to add
  // (`owned_end_frame - owned_start_frame`). Set only for a `shot_aligned`
  // manifest, whose per-segment lengths vary and therefore cannot be re-derived
  // from `chainSegmentFrames`; `advanceVideoChain` rebases it onto the previous
  // segment's actual length. Absent for every fixed-length and legacy chain,
  // which is how that path keeps its existing arithmetic.
  chainPlannedNewOutputFrames?: number;
  // The |actual - planned| drift measured when THIS item's own segment
  // finished, recorded (never used to gate anything) once the chain
  // continues past it within tolerance -- design §4.1 "許容内: そのまま続行
  // し、drift 値を記録する". Undefined if this item's segment had no manifest
  // to drift from, or has not finished yet.
  chainLastDriftFrames?: number;
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
  // Video-chain drift pause (videoChain.ts design §4.1). Held HERE, not in the
  // panel that observed it: `chain_vid` is claimed by both Txt2Img and Img2Img,
  // which are mounted exclusively per tab, so panel-local pause state would be
  // destroyed (dialog and already-fetched clip with it) by a tab switch while
  // the queue kept the unpatched item pending. Living on the queue also lets
  // `startNextInQueue` refuse to dispatch the paused group, which is the only
  // place that refusal actually holds -- suppressing one panel's explicit
  // `processQueue()` call does not, because the auto-start effect re-enters it.
  chainPause: ChainDriftPause | null;
  pauseChain: (pause: ChainDriftPause) => void;
  clearChainPause: () => void;
}

const GenerationQueueContext = createContext<GenerationQueueContextType | undefined>(undefined);

export function GenerationQueueProvider({ children }: { children: ReactNode }) {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [currentItem, setCurrentItem] = useState<QueueItem | null>(null);
  const [generateForever, setGenerateForever] = useState<boolean>(false);
  const [progressSnapshot, setProgressSnapshot] = useState<GenerationProgressSnapshot | null>(null);
  const [completedResults, setCompletedResults] = useState<Partial<Record<GenerationPanelId, GenerationResultSnapshot>>>({});
  const [chainPause, setChainPause] = useState<ChainDriftPause | null>(null);

  // Use refs that are synchronously updated alongside state
  const queueRef = useRef<QueueItem[]>(queue);
  const currentItemRef = useRef<QueueItem | null>(currentItem);
  // Written synchronously by pauseChain/clearChainPause so a pause raised in
  // the same React batch as completeCurrentItem() is already visible to the
  // startNextInQueue that batch's re-render triggers.
  const chainPauseRef = useRef<ChainDriftPause | null>(chainPause);

  // Synchronously update refs whenever state changes
  queueRef.current = queue;
  currentItemRef.current = currentItem;

  const pauseChain = useCallback((pause: ChainDriftPause) => {
    chainPauseRef.current = pause;
    setChainPause(pause);
  }, []);

  const clearChainPause = useCallback(() => {
    chainPauseRef.current = null;
    setChainPause(null);
  }, []);

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

    // A drift pause is a held decision about THIS group; cancelling the group
    // settles it, whether the cancel came from the pause dialog's "stop" or
    // from the segment-failure cascade.
    if (chainPauseRef.current?.loopGroupId === loopGroupId) {
      chainPauseRef.current = null;
      setChainPause(null);
    }

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

    // Removing the item a drift pause is holding settles that pause: case 1
    // (main) drops the whole group, case 2 drops every step from
    // `loopStepIndex` on, case 3 drops only that (last) step.
    const pause = chainPauseRef.current;
    if (
      pause?.loopGroupId === loopGroupId &&
      (!isLoopStep || (loopStepIndex !== undefined && pause.nextStepIndex >= loopStepIndex))
    ) {
      chainPauseRef.current = null;
      setChainPause(null);
    }

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

    // The gate for a video-chain drift pause: the paused group's next segment
    // is deliberately still unpatched (no `inputVideo`), so dispatching it
    // would fail with an unrelated "no input video" error and the failure
    // cascade would then cancel the rest of the group the user is being asked
    // about. Items outside the paused group are unaffected.
    const pausedGroupId = chainPauseRef.current?.loopGroupId;
    const isDispatchable = (item: QueueItem) =>
      item.status === "pending" &&
      (!allowedTypes || allowedTypes.includes(item.type)) &&
      !(pausedGroupId !== undefined && item.loopGroupId === pausedGroupId);

    let nextItem: QueueItem | undefined;

    // If current item is part of a loop group, prioritize next step in same group
    if (currentItemValue?.loopGroupId) {
      const currentLoopGroupId = currentItemValue.loopGroupId;
      const currentLoopStepIndex = currentItemValue.loopStepIndex ?? -1;

      // Find next step in the same loop group
      nextItem = currentQueue.find((item) =>
        isDispatchable(item) &&
        item.loopGroupId === currentLoopGroupId &&
        (item.loopStepIndex ?? 0) === currentLoopStepIndex + 1
      );

      console.log("[QueueContext] Looking for next loop step in group:", currentLoopGroupId, "after index:", currentLoopStepIndex);
      console.log("[QueueContext] Found loop step:", nextItem);
    }

    // If no loop step found, get next pending item in queue order
    if (!nextItem) {
      nextItem = currentQueue.find(isDispatchable);
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
        (item) =>
          item.id !== nextItem!.id &&
          item.status === "pending" &&
          !(pausedGroupId !== undefined && item.loopGroupId === pausedGroupId)
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
    // Nothing is left for a pause to hold.
    chainPauseRef.current = null;
    setChainPause(null);
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
        chainPause,
        pauseChain,
        clearChainPause,
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
