"use client";

// The generation queue's dispatch loop. Headless and mounted once at the root,
// because /generate truly unmounts a panel on tab switch: while each panel
// owned its own processQueue, switching away from the panel that claimed the
// running item's type left it with no dispatcher and it stalled forever.
//
// Everything this needs is either on the QueueItem (frozen at enqueue time --
// see freezeDispatchState in the panels) or in a context; it never reads panel
// state. Panels render `progressSnapshot` and `completedResults` instead.

import { useCallback, useEffect, useRef } from "react";
import { QueueItem, useGenerationQueue } from "@/contexts/GenerationQueueContext";
import { useStartup } from "@/contexts/StartupContext";
import {
  generateOutpaint,
  generateOutpaintAudio,
  generateOutpaintVideo,
  generateUpscale,
  getResultFilename,
  getResultPlaybackFilename,
  getResultSeed,
  getResultAncestralSeed,
  isGenerationStalledError,
  OutpaintParams,
  OutpaintAudioParams,
  OutpaintVideoParams,
  UpscaleParams,
} from "@/utils/api";

// Types this processor has taken over from the panels. Panels still claim the
// rest until their migration phase lands.
const CLAIMED_TYPES: readonly QueueItem["type"][] = [
  "upscale",
  "outpaint",
  "outpaint_vid",
  "outpaint_aud",
];

const resultWarnings = (result: any): string[] =>
  (result?.warnings || []).map((w: any) => (typeof w === "string" ? w : w?.message)).filter(Boolean);

const errorDetail = (error: any) =>
  error?.response?.data?.detail || error?.message || "Unknown error";

// Types that do not need a loaded diffusion model: an upscale can run on a
// spandrel checkpoint alone, and UpscalePanel never held its queue on
// `modelLoaded`. Everything else waits, since a queue can outlive a backend
// restart and dispatching then earns a 400 about the wrong thing.
const NO_MODEL_REQUIRED: readonly QueueItem["type"][] = ["upscale"];

const RESCHEDULE_MS = 100;

export default function GenerationQueueProcessor() {
  const {
    queue,
    currentItem,
    startNextInQueue,
    completeCurrentItem,
    failCurrentItem,
    publishCompletedResult,
    appendResult,
    chainPause,
  } = useGenerationQueue();
  const { modelLoaded } = useStartup();

  // Re-entrancy guard, synchronous where `currentItem` is a render behind.
  const busyRef = useRef(false);
  const processRef = useRef<() => Promise<void>>();

  const scheduleNext = useCallback(() => {
    setTimeout(() => { void processRef.current?.(); }, RESCHEDULE_MS);
  }, []);

  const runUpscale = useCallback(async (item: QueueItem) => {
    try {
      const inputImage = item.inputImage;
      if (!inputImage) {
        throw new Error("No input image available for upscale generation");
      }

      const result = await generateUpscale(item.params as UpscaleParams, inputImage);
      const url = `/outputs/${result.image.filename}`;
      const info = { width: result.image.width, height: result.image.height };
      publishCompletedResult({ panel: "upscale", kind: "image", url, info, params: item.params });
      appendResult({ url, kind: "image" });

      busyRef.current = false;
      completeCurrentItem();
      scheduleNext();
    } catch (error: any) {
      console.error("[Queue] Upscale generation failed:", error);
      // alert() blocks the JS thread; release the guard and requeue before
      // showing it, or the auto-start effect sees a stale busy flag until the
      // dialog closes.
      busyRef.current = false;
      failCurrentItem();
      scheduleNext();
      alert(isGenerationStalledError(error) ? error.message : "Upscale generation failed. Please check console for details.");
    }
  }, [publishCompletedResult, appendResult, completeCurrentItem, failCurrentItem, scheduleNext]);

  const runOutpaintVideo = useCallback(async (item: QueueItem) => {
    try {
      const clip = item.inputVideo;
      if (!clip) {
        throw new Error("No input video available for video outpaint generation");
      }
      const result = await generateOutpaintVideo(
        item.params as OutpaintVideoParams, clip, item.bridgeVideo, item.referenceImages);
      const url = `/outputs/${getResultFilename(result)}`;
      const playback = `/outputs/${getResultPlaybackFilename(result)}`;
      const playbackUrl = playback !== url ? playback : undefined;
      const info = { num_frames: result.image?.num_frames, fps: result.image?.fps, duration: result.image?.duration };
      publishCompletedResult({
        panel: "outpaint", kind: "video", url, playbackUrl, info,
        seed: getResultSeed(result), params: item.params, warnings: resultWarnings(result),
      });
      appendResult({ url, kind: "video", playbackUrl });

      busyRef.current = false;
      completeCurrentItem();
      scheduleNext();
    } catch (error: any) {
      console.error("[Queue] Video outpaint generation failed:", error);
      busyRef.current = false;
      failCurrentItem();
      scheduleNext();
      alert(isGenerationStalledError(error)
        ? error.message
        : `Video outpaint generation failed: ${errorDetail(error)}`);
    }
  }, [publishCompletedResult, appendResult, completeCurrentItem, failCurrentItem, scheduleNext]);

  const runOutpaintAudio = useCallback(async (item: QueueItem) => {
    try {
      const referenceAudio = item.inputAudio;
      if (!referenceAudio) {
        throw new Error("No reference audio available for audio outpaint generation");
      }
      const result = await generateOutpaintAudio(item.params as OutpaintAudioParams, referenceAudio);
      const url = `/outputs/${result.image.filename}`;
      const info = { duration: result.image?.duration, sample_rate: result.image?.sample_rate };
      publishCompletedResult({
        panel: "outpaint", kind: "audio", url, info,
        seed: getResultSeed(result), params: item.params, warnings: resultWarnings(result),
      });
      appendResult({ url, kind: "audio" });

      busyRef.current = false;
      completeCurrentItem();
      scheduleNext();
    } catch (error: any) {
      console.error("[Queue] Audio outpaint generation failed:", error);
      busyRef.current = false;
      failCurrentItem();
      scheduleNext();
      alert(isGenerationStalledError(error)
        ? error.message
        : `Audio outpaint generation failed: ${errorDetail(error)}`);
    }
  }, [publishCompletedResult, appendResult, completeCurrentItem, failCurrentItem, scheduleNext]);

  const runOutpaintImage = useCallback(async (item: QueueItem) => {
    try {
      const itemParams = item.params as OutpaintParams;
      const result = await generateOutpaint(itemParams, item.inputImage!);
      if (!result.success) {
        throw new Error("Outpaint generation did not succeed");
      }
      const url = `/outputs/${getResultFilename(result)}`;
      const seed = getResultSeed(result);
      const ancestralSeed = getResultAncestralSeed(result);
      publishCompletedResult({
        panel: "outpaint",
        kind: "image",
        url,
        seed,
        ancestralSeed,
        params: { ...itemParams, seed, ancestral_seed: ancestralSeed ?? -1 },
        warnings: resultWarnings(result),
      });
      appendResult({ url, kind: "image" });

      busyRef.current = false;
      completeCurrentItem();
      scheduleNext();
    } catch (error: any) {
      console.error("[Queue] Outpaint generation failed:", error);
      busyRef.current = false;
      failCurrentItem();
      scheduleNext();
      alert(isGenerationStalledError(error)
        ? error.message
        : `Outpaint generation failed: ${errorDetail(error)}`);
    }
  }, [publishCompletedResult, appendResult, completeCurrentItem, failCurrentItem, scheduleNext]);

  const process = useCallback(async () => {
    if (busyRef.current) return;

    const claimable = modelLoaded
      ? CLAIMED_TYPES
      : CLAIMED_TYPES.filter((type) => NO_MODEL_REQUIRED.includes(type));
    const nextItem = startNextInQueue(claimable);
    if (!nextItem) return;

    busyRef.current = true;
    switch (nextItem.type) {
      case "upscale":
        await runUpscale(nextItem);
        return;
      case "outpaint":
        await runOutpaintImage(nextItem);
        return;
      case "outpaint_vid":
        await runOutpaintVideo(nextItem);
        return;
      case "outpaint_aud":
        await runOutpaintAudio(nextItem);
        return;
      default:
        // Unreachable: `claimable` is exactly the set handled above.
        busyRef.current = false;
        console.error("[Queue] No dispatch branch for claimed type:", nextItem.type);
        failCurrentItem();
        return;
    }
  }, [modelLoaded, startNextInQueue, failCurrentItem, runUpscale, runOutpaintImage, runOutpaintVideo, runOutpaintAudio]);

  processRef.current = process;

  useEffect(() => {
    // Items held by a video-chain drift pause are not pending WORK: the queue
    // refuses to dispatch them until the user answers the dialog, so counting
    // them here would re-enter process() on every render for an item that
    // cannot start.
    const pausedGroupId = chainPause?.loopGroupId;
    const hasPending = queue.some((item) =>
      item.status === "pending" &&
      CLAIMED_TYPES.includes(item.type) &&
      (modelLoaded || NO_MODEL_REQUIRED.includes(item.type)) &&
      !(pausedGroupId !== undefined && item.loopGroupId === pausedGroupId));

    if (hasPending && currentItem === null && !busyRef.current) {
      void process();
    }
  }, [queue, currentItem, process, modelLoaded, chainPause]);

  return null;
}
