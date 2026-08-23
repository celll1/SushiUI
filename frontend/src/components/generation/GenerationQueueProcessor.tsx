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
import { useActiveTraining } from "@/hooks/useActiveTraining";
import {
  generateInpaint,
  generateInpaintTrainingPreview,
  generateInpaintVideo,
  imageSourceToBase64,
  isLatentOnlyResult,
  InpaintParams,
  InpaintVideoParams,
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
  "inpaint",
  "inpaint_vid",
];

const resultWarnings = (result: any): string[] =>
  (result?.warnings || []).map((w: any) => (typeof w === "string" ? w : w?.message)).filter(Boolean);

const errorDetail = (error: any) =>
  error?.response?.data?.detail || error?.message || "Unknown error";

// A deliberate cancelGeneration() surfaces as the backend's own RuntimeError,
// not a distinct error type, so it has to be recognised by its text.
const isCancelledError = (error: any) => {
  const message = String(error?.message || "").toLowerCase();
  const detail = String(error?.response?.data?.detail || "").toLowerCase();
  return message.includes("cancel")
    || detail.includes("cancel")
    || JSON.stringify(error).toLowerCase().includes("cancel");
};

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
    updateQueueItemByLoop,
    getLoopGroupItems,
    publishCompletedResult,
    publishFailure,
    appendResult,
    chainPause,
  } = useGenerationQueue();
  const { modelLoaded } = useStartup();

  const activeTraining = useActiveTraining();

  // Re-entrancy guard, synchronous where `currentItem` is a render behind.
  const busyRef = useRef(false);
  const processRef = useRef<() => Promise<void>>();
  // The object URL of the last training preview that the backend did NOT
  // persist. Owned here rather than by a panel so it is not revoked by a tab
  // switch, which would blank a preview that is still the current result.
  const previewBlobUrlRef = useRef<string | null>(null);

  const scheduleNext = useCallback(() => {
    setTimeout(() => { void processRef.current?.(); }, RESCHEDULE_MS);
  }, []);

  // Prefer the stable /outputs/ URL when the preview was saved; fall back to a
  // transient blob URL.
  const trainingPreviewUrl = useCallback((preview: { filename?: string; blob: Blob }) => {
    if (preview.filename) return `/outputs/${preview.filename}`;
    if (previewBlobUrlRef.current) URL.revokeObjectURL(previewBlobUrlRef.current);
    previewBlobUrlRef.current = URL.createObjectURL(preview.blob);
    return previewBlobUrlRef.current;
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

  // Feed the loop group's NEXT step from the step that just finished: its input
  // (image or cached latent), the TIPO-expanded prompt, the compounded scale
  // size, and a fresh base64 for any per-step ControlNet set to reuse the loop
  // image. The step config read here was frozen at enqueue time.
  const advanceLoopGroup = useCallback(async (
    item: QueueItem,
    result: any,
    imageUrl: string | undefined,
    opts: { latentPassthrough: boolean },
  ) => {
    const groupId = item.loopGroupId;
    if (groupId === undefined) return;
    const nextIndex = (item.loopStepIndex ?? -1) + 1;
    const stepConfigOf = (index: number) =>
      getLoopGroupItems(groupId).find((queued) => queued.loopStepIndex === index)?.loopStepConfig;

    if (opts.latentPassthrough && isLatentOnlyResult(result)) {
      // No decoded image for this step; chain the cached latent instead, and
      // compound scale off THIS step's own target size (every step's initial
      // size was computed from the main params at enqueue time, so without
      // this a chain of scale steps would never compound). TIPO/ControlNet
      // are skipped: both need a decoded image, and a ControlNet-conditioned
      // step is forced to resize_mode "image" at enqueue, so never lands here.
      updateQueueItemByLoop(groupId, nextIndex, {
        inputLatentId: result.latent_id,
        inputImage: undefined,
      });
      const nextStepConfig = stepConfigOf(nextIndex);
      const currentWidth = (item.params as any).width;
      const currentHeight = (item.params as any).height;
      if (nextStepConfig?.sizeMode === "scale" && currentWidth && currentHeight) {
        const scale = nextStepConfig.scale || 1.0;
        updateQueueItemByLoop(groupId, nextIndex, (queued) => ({
          params: {
            ...queued.params,
            width: Math.round(currentWidth * scale),
            height: Math.round(currentHeight * scale),
          } as any,
        }));
      }
      return;
    }

    updateQueueItemByLoop(groupId, nextIndex, opts.latentPassthrough
      ? { inputImage: imageUrl, inputLatentId: undefined }
      : { inputImage: imageUrl });

    if (item.loopStepIndex === -1 && (item.params as any).use_tipo && result.image?.prompt) {
      for (const stepItem of getLoopGroupItems(groupId)) {
        if (!stepItem.isLoopStep || stepItem.loopStepIndex === undefined) continue;
        updateQueueItemByLoop(groupId, stepItem.loopStepIndex, (queued) => ({
          params: { ...queued.params, prompt: result.image.prompt } as any,
        }));
      }
    }

    const stepConfig = stepConfigOf(nextIndex);
    const needsImageData = !!stepConfig && (
      (!stepConfig.useMainControlNets && stepConfig.controlnets && stepConfig.controlnets.length > 0) ||
      stepConfig.sizeMode === "scale");
    if (!needsImageData || !imageUrl) return;

    const imageBlob = await (await fetch(imageUrl)).blob();

    if (stepConfig!.sizeMode === "scale" && stepConfig!.scale) {
      const objectUrl = URL.createObjectURL(imageBlob);
      const size = await new Promise<{ width: number; height: number }>((resolve) => {
        const img = new Image();
        img.onload = () => resolve({ width: img.width, height: img.height });
        img.src = objectUrl;
      });
      URL.revokeObjectURL(objectUrl);
      if (size.width && size.height) {
        updateQueueItemByLoop(groupId, nextIndex, (queued) => ({
          params: {
            ...queued.params,
            width: Math.round(size.width * stepConfig!.scale!),
            height: Math.round(size.height * stepConfig!.scale!),
          } as any,
        }));
      }
    }

    if (!stepConfig!.useMainControlNets && stepConfig!.controlnets?.length) {
      const imageBase64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(String(reader.result).split(",")[1]);
        reader.readAsDataURL(imageBlob);
      });
      updateQueueItemByLoop(groupId, nextIndex, (queued) => ({
        params: {
          ...queued.params,
          controlnets: stepConfig!.controlnets.map((cn) =>
            cn.useLoopImage ? { ...cn, image_base64: imageBase64 } : cn),
        } as any,
      }));
    }
  }, [getLoopGroupItems, updateQueueItemByLoop]);

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

  const runInpaintVideo = useCallback(async (item: QueueItem) => {
    try {
      const clip = item.inputVideo;
      if (!clip) throw new Error("No input video available for video inpaint generation");
      const result = await generateInpaintVideo(
        item.params as InpaintVideoParams, clip, item.spatialMaskFiles, item.references);
      const url = `/outputs/${getResultFilename(result)}`;
      const playback = `/outputs/${getResultPlaybackFilename(result)}`;
      const playbackUrl = playback !== url ? playback : undefined;
      const info = {
        num_frames: result.image?.num_frames,
        fps: result.image?.fps,
        duration: result.image?.duration,
      };
      publishCompletedResult({
        panel: "inpaint", kind: "video", url, playbackUrl, info,
        seed: getResultSeed(result), params: item.params, warnings: resultWarnings(result),
      });
      appendResult({ url, kind: "video", playbackUrl });

      busyRef.current = false;
      completeCurrentItem();
      scheduleNext();
    } catch (error: any) {
      console.error("[Queue] Video inpaint generation failed:", error);
      busyRef.current = false;
      publishFailure({ panel: "inpaint", itemId: item.id, cancelled: false });
      failCurrentItem();
      scheduleNext();
      alert(isGenerationStalledError(error)
        ? error.message
        : `Video inpaint generation failed: ${errorDetail(error)}`);
    }
  }, [publishCompletedResult, appendResult, completeCurrentItem, failCurrentItem, publishFailure, scheduleNext]);

  const runInpaint = useCallback(async (item: QueueItem) => {
    // Explicit allowlist rather than a spread: `item.params` also carries
    // panel-only keys (TIPO config among them) that inpaint must not send.
    const p = item.params as any;
    let apiParams: InpaintParams = {
      prompt: p.prompt,
      negative_prompt: p.negative_prompt,
      steps: p.steps,
      cfg_scale: p.cfg_scale,
      timestep_shift: p.timestep_shift,
      img_cfg_scale: p.img_cfg_scale,
      cfg_norm: p.cfg_norm,
      sensenova_mot_phase_eviction: p.sensenova_mot_phase_eviction,
      sensenova_kv_cache_streaming: p.sensenova_kv_cache_streaming,
      sampler: p.sampler,
      schedule_type: p.schedule_type,
      seed: p.seed,
      width: p.width,
      height: p.height,
      denoising_strength: p.denoising_strength,
      vae_drift_correction: p.vae_drift_correction,
      mask_blur: p.mask_blur,
      inpaint_full_res: p.inpaint_full_res,
      inpaint_full_res_padding: p.inpaint_full_res_padding,
      inpaint_fill_mode: p.inpaint_fill_mode,
      inpaint_fill_strength: p.inpaint_fill_strength,
      inpaint_blur_strength: p.inpaint_blur_strength,
      region_prompt: p.region_prompt,
      region_negative_prompt: p.region_negative_prompt,
      region_prompt_strength: p.region_prompt_strength,
      region_prompt_method: p.region_prompt_method,
      region_mask_feather: p.region_mask_feather,
      seam_structure_strength: p.seam_structure_strength,
      seam_structure_depth: p.seam_structure_depth,
      seam_structure_end: p.seam_structure_end,
      seam_structure_saliency: p.seam_structure_saliency,
      seam_structure_max_area: p.seam_structure_max_area,
      boundary_relax_strength: p.boundary_relax_strength,
      boundary_relax_width: p.boundary_relax_width,
      boundary_relax_noise: p.boundary_relax_noise,
      boundary_relax_full_until: p.boundary_relax_full_until,
      boundary_relax_end: p.boundary_relax_end,
      boundary_relax_paste: p.boundary_relax_paste,
      resize_mode: p.resize_mode,
      resampling_method: p.resampling_method,
      loras: p.loras,
      controlnets: p.controlnets,
      developer_mode: p.developer_mode,
      cfg_schedule_type: p.cfg_schedule_type,
      cfg_rescale_snr_alpha: p.cfg_rescale_snr_alpha,
      dynamic_threshold_percentile: p.dynamic_threshold_percentile,
      nag_enable: p.nag_enable,
      nag_scale: p.nag_scale,
      nag_tau: p.nag_tau,
      nag_alpha: p.nag_alpha,
      nag_sigma_end: p.nag_sigma_end,
      nag_negative_prompt: p.nag_negative_prompt,
      unet_quantization: p.unet_quantization,
      quantized_gemm_mode: p.quantized_gemm_mode,
      cpu_text_encoding: p.cpu_text_encoding,
      text_encoder_quantization: p.text_encoder_quantization,
      original_size_w: p.original_size_w,
      original_size_h: p.original_size_h,
      original_size_scale: p.original_size_scale,
      attention_type: p.attention_type,
      vision_encoder_path: p.vision_encoder_path,
      vae_path: p.vae_path,
      text_encoder_path: p.text_encoder_path,
      pid_sr_output: p.pid_sr_output,
      pid_use_gemma: p.pid_use_gemma,
      pid_low_vram: p.pid_low_vram,
      pid_tile_native: p.pid_tile_native,
      pid_tile_overlap_ratio: p.pid_tile_overlap_ratio,
      pid_fast_large_decode: p.pid_fast_large_decode,
      spectrum_enable: p.spectrum_enable,
      fbcache_enable: p.fbcache_enable,
      fbcache_threshold: p.fbcache_threshold,
      fbcache_warmup_steps: p.fbcache_warmup_steps,
      spectrum_w: p.spectrum_w,
      spectrum_w_decay: p.spectrum_w_decay,
      spectrum_delta_cap: p.spectrum_delta_cap,
      spectrum_m: p.spectrum_m,
      spectrum_lam: p.spectrum_lam,
      spectrum_warmup_steps: p.spectrum_warmup_steps,
      spectrum_window_size: p.spectrum_window_size,
      spectrum_flex_window: p.spectrum_flex_window,
      spectrum_tail: p.spectrum_tail,
      spectrum_feature_mode: p.spectrum_feature_mode,
      spectrum_cache_branch: p.spectrum_cache_branch,
      spectrum_max_cache: p.spectrum_max_cache,
      vae_tiling: p.vae_tiling,
      vae_tile_threshold: p.vae_tile_threshold,
      vae_tile_mode: p.vae_tile_mode,
      vae_tile_global_norm: p.vae_tile_global_norm,
      color_flatten_strength: p.color_flatten_strength,
      flatten_in_loop: p.flatten_in_loop,
      flatten_in_loop_last_steps: p.flatten_in_loop_last_steps,
      flatten_in_loop_min_region: p.flatten_in_loop_min_region,
      enable_block_swap: p.enable_block_swap,
      blocks_to_swap: p.blocks_to_swap,
      use_pinned_memory: p.use_pinned_memory,
      block_swap_h2d_only: p.block_swap_h2d_only,
      block_swap_ring_size: p.block_swap_ring_size,
      keep_models_hot: p.keep_models_hot,
      // Inpaint never uses loop_decode "none" (the backend rejects it);
      // intermediate loop steps fall back to "cheap" + skip_gallery.
      loop_decode: p.loop_decode,
      skip_gallery: p.skip_gallery,
    };
    if (p.ref_images && p.ref_images.length > 0) {
      apiParams = { ...apiParams, ref_images: p.ref_images };
    }

    try {
      let result: any;
      let imageUrl: string;

      if (item.useTrainingModel && (item.trainingRunId ?? activeTraining?.run_id)) {
        const preview = await generateInpaintTrainingPreview({
          ...(apiParams as any),
          init_image_base64: await imageSourceToBase64(item.inputImage!),
          mask_image_base64: await imageSourceToBase64(item.maskImage!),
          denoising_strength: apiParams.denoising_strength ?? 0.75,
          run_id: (item.trainingRunId ?? activeTraining!.run_id),
          save_to_gallery: item.savePreviewToGallery ?? false,
        });
        imageUrl = trainingPreviewUrl(preview);
        result = {
          success: true,
          actual_seed: preview.seed ? Number(preview.seed) : -1,
          image: {
            filename: preview.filename ?? `preview_${preview.requestId ?? "training"}.png`,
            filepath: imageUrl,
            seed: preview.seed ? Number(preview.seed) : -1,
            ancestral_seed: -1,
            prompt: apiParams.prompt,
            negative_prompt: apiParams.negative_prompt,
            width: apiParams.width,
            height: apiParams.height,
          },
        };
      } else {
        result = await generateInpaint(apiParams, item.inputImage!, item.maskImage!);
        // skip_gallery=true (a "cheap" intermediate loop step) answers with a
        // top-level filename and no nested `image` object.
        imageUrl = result.success ? `/outputs/${getResultFilename(result)}` : "";
      }

      if (!result.success) {
        busyRef.current = false;
        publishFailure({ panel: "inpaint", itemId: item.id, cancelled: false });
        failCurrentItem();
        scheduleNext();
        alert("Generation failed");
        return;
      }

      const seed = getResultSeed(result);
      const ancestralSeed = getResultAncestralSeed(result);
      if (imageUrl) {
        publishCompletedResult({
          panel: "inpaint",
          kind: "image",
          url: imageUrl,
          seed,
          ancestralSeed,
          params: {
            ...item.params,
            seed,
            ancestral_seed: ancestralSeed ?? -1,
            width: result.image?.width ?? p.width,
            height: result.image?.height ?? p.height,
          },
          ephemeral: isLatentOnlyResult(result) || !result.image,
        });
        appendResult({ url: imageUrl, kind: "image" });
      }

      await advanceLoopGroup(item, result, imageUrl, { latentPassthrough: false });

      busyRef.current = false;
      completeCurrentItem();
      scheduleNext();
    } catch (error: any) {
      console.error("[Queue] Inpaint generation failed:", error);
      const cancelled = isCancelledError(error);
      busyRef.current = false;
      publishFailure({ panel: "inpaint", itemId: item.id, cancelled });
      failCurrentItem();
      scheduleNext();
      if (!cancelled) {
        alert(isGenerationStalledError(error)
          ? error.message
          : "Generation failed: " + (error instanceof Error ? error.message : String(error)));
      }
    }
  }, [activeTraining, advanceLoopGroup, appendResult, completeCurrentItem, failCurrentItem, publishCompletedResult, publishFailure, scheduleNext, trainingPreviewUrl]);

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
      case "inpaint":
        await runInpaint(nextItem);
        return;
      case "inpaint_vid":
        await runInpaintVideo(nextItem);
        return;
      default:
        // Unreachable: `claimable` is exactly the set handled above.
        busyRef.current = false;
        console.error("[Queue] No dispatch branch for claimed type:", nextItem.type);
        failCurrentItem();
        return;
    }
  }, [modelLoaded, startNextInQueue, failCurrentItem, runUpscale, runOutpaintImage, runOutpaintVideo, runOutpaintAudio, runInpaint, runInpaintVideo]);

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
