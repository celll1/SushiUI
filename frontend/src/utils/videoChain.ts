// Opt-in video-length chaining: reaching a clip length above the loaded
// architecture's single-inference cap by running several generation requests,
// each continuation a temporal-outpaint `extend_forward` from the previous
// segment's own output.
//
// This runs ON GenerationQueueContext, as ordinary queue items (a "main"
// video item at the cap, followed by one or more `chain_vid` loop steps) --
// NOT as a promise chain awaited outside the queue. An earlier version of
// this feature bypassed the queue (awaiting each segment directly inside the
// panel), which meant the queue's own serialization gate
// (`startNextInQueue`'s `currentItemValue?.status === "generating"` check)
// never saw the chain running: switching tabs mid-chain let the OTHER panel's
// processQueue dispatch a second, fully concurrent generation, with nothing
// on the backend to stop it (gpu_coordinator's generation_slot is refcounted,
// not exclusive; pipeline_manager has no generation lock). Putting the chain
// on the queue closes that gap for free -- it is the same lock every other
// queued generation already relies on -- and also buys back queue visibility,
// progress-snapshot/completedResults persistence across a remount, and
// correct group cancellation (cancelLoopGroup), none of which the bypassed
// version had.
//
// `chain_vid` is claimed by Txt2ImgPanel and Img2ImgPanel only -- the two
// panels that can ever enqueue it (mirrors how `img2img`/`ref2vid` are
// claimed by exactly the panels that can produce THOSE types, not every
// panel). A chain survives switching between Txt2Img and Img2Img while it
// runs, the same as any other loop group; it does not survive switching to
// Outpaint/Inpaint/Upscale, which have no video-chain entry point and would
// gain a third dispatch branch for a type they can never enqueue.
//
// All N segments are enqueued UP FRONT as a loop group (main item at
// `loopStepIndex: -1`, continuations at `0..N-2`), rather than one at a time
// as each prior segment finishes: the point of putting this on the queue is
// that the user can SEE the whole plan waiting, the same as a loop-generation
// group with 4 configured steps shows all 4 immediately. Each continuation's
// `total_frames` is only a planning ESTIMATE at enqueue time
// (`planVideoChainSegments`); `advanceVideoChain` below re-derives it from
// the ACTUAL previous segment's reported frame count right before that step
// runs (a real generation can snap slightly differently than the pre-flight
// estimate), and patches it onto the already-queued item with
// `updateQueueItemByLoop` -- the same mechanism img2img loop steps already
// use to inherit the previous step's output image.
import {
  ArchCapabilities,
  LoRAConfig,
  OutpaintVideoParams,
  QuantizedGemmMode,
  nextVideoChainTotalFrames,
  planVideoChainSegments,
} from "./api";
import type { QueueItem } from "@/contexts/GenerationQueueContext";

// The fields every one of this app's video-generation request shapes
// (Txt2VidParams / Img2VidParams / Ref2VidParams) shares with
// OutpaintVideoParams, i.e. everything a continuation segment carries over
// from the request that started the chain. Kept as its own interface (rather
// than accepting one of those request types directly) so this module does not
// need to import every video request shape just to read a handful of fields
// from it -- structural typing means any of them satisfies this already.
export interface ChainContinuationBase {
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  frame_rate?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  seed?: number;
  num_videos_per_prompt?: number;
  max_sequence_length?: number;
  audio_enable?: boolean;
  vae_path?: string | null;
  text_encoder_path?: string | null;
  unet_quantization?: string | null;
  quantized_gemm_mode?: QuantizedGemmMode;
  // Generation-time LoRA (MiniMax-H3 only; see Txt2VidParams.loras). Carried
  // to every continuation segment so a LoRA applied to segment 1 stays applied
  // for the whole chain, not just the capped first request.
  loras?: LoRAConfig[];
  // Block-swap CPU offload (Txt2VidParams/Img2VidParams/Ref2VidParams'
  // `blocks_to_swap`, itself sourced from the panel's video-specific
  // `video_blocks_to_swap` state). Carried to every continuation segment: a
  // user who enabled it because their card cannot hold the model resident
  // needs EVERY segment to run with it, not just the capped first request --
  // dropping it on segments 2..N is a real OOM on the exact machine that
  // needed the setting, not just a slowdown.
  blocks_to_swap?: number;
  // Output-tail head fusion (MiniMax-H3 only, not bit-exact -- see
  // Txt2VidParams.fuse_output_proj). Carried for the same reason
  // `blocks_to_swap` is: a VRAM setting the user picked for segment 1 has to
  // hold for every continuation segment.
  fuse_output_proj?: boolean;
  // FBCache/Spectrum acceleration (see VideoAccelerationControls, shared by
  // every video-capable panel). Carried to every continuation segment for the
  // same reason `blocks_to_swap` is: a user who enabled one because their
  // card cannot hold the model resident at full speed needs EVERY segment to
  // run with it, not just the capped first request.
  fbcache_enable?: boolean;
  fbcache_threshold?: number;
  fbcache_warmup_steps?: number;
  spectrum_enable?: boolean;
  spectrum_w?: number;
  spectrum_w_decay?: number;
  spectrum_delta_cap?: number;
  spectrum_m?: number;
  spectrum_lam?: number;
  spectrum_warmup_steps?: number;
  spectrum_window_size?: number;
  spectrum_flex_window?: number;
  spectrum_tail?: number;
  spectrum_max_cache?: number;
}

// The Txt2VidParams/Img2VidParams/Ref2VidParams -> OutpaintVideoParams
// mapping a continuation segment sends, in ONE place (previously duplicated,
// near-verbatim, in both Txt2ImgPanel and Img2ImgPanel). `total_frames` is
// the only field that changes segment to segment; everything else -- content
// (prompt/geometry/steps/guidance/seed) AND execution/acceleration
// (blocks_to_swap, quantization, LoRAs) -- replays the request that started
// the chain unchanged. Segments 2..N are one clip to the user, not N
// independent requests, so an execution setting the user picked for a real
// reason (most concretely: block swap because the card cannot hold the model
// resident) has to hold for every segment or a later one can OOM on the exact
// machine that needed it.
//
// `attention_type` is NOT part of `ChainContinuationBase` even though the
// endpoint accepts it: `generateOutpaintVideo` (api.ts) resolves it itself at
// SEND time via `resolveGlobalAttentionType`, the same as every other video
// route, from the current global localStorage setting -- not from whatever
// value happened to be on the object passed in. Every segment of a chain already
// reads that same global at its own send time, so it is already consistent
// across the whole chain with no field needed here; adding one would just be
// a second, stale source fighting the resolver.
//
// `fbcache_enable`/`spectrum_*` ARE part of `ChainContinuationBase`: both
// Txt2ImgPanel and Img2ImgPanel's video-mode Acceleration section
// (`VideoAccelerationControls`) now sets them on the SAME `params` fields the
// still-image Acceleration tab uses, and `Txt2VidParams`/`generateTxt2Vid`
// (and the img2vid/ref2vid equivalents) read and send them on segment 1 --
// so segment 1 has a real setting to replay, and every continuation replays
// it exactly like `blocks_to_swap`.
//
// Deliberately NOT carried over (each is a real, disclosed limitation of
// what a continuation segment can condition on -- see the chain-choice
// dialog, which states them):
//   - MiniMax-H3 ref2va VIDEO/AUDIO references: this endpoint accepts image
//     references only (`referenceImages`, a plain File[]); the video/audio
//     reference tracks a ref2va request can carry condition segment 1 only.
//   - img2vid's ia2v `input_audio` track: no equivalent field exists on this
//     endpoint; it conditions segment 1 only.
//   - Keyframe anchors: this endpoint has no `keyframes` field; a
//     continuation is conditioned only on the boundary frame the placed clip
//     itself provides via `extend_forward`, not on any anchor from the
//     original request.
export function buildChainContinuationParams(
  base: ChainContinuationBase,
  totalFrames: number,
  referenceImageSize?: "max" | "match"
): OutpaintVideoParams {
  return {
    prompt: base.prompt,
    negative_prompt: base.negative_prompt,
    width: base.width,
    height: base.height,
    frame_rate: base.frame_rate,
    num_inference_steps: base.num_inference_steps,
    guidance_scale: base.guidance_scale,
    seed: base.seed,
    num_videos_per_prompt: base.num_videos_per_prompt,
    max_sequence_length: base.max_sequence_length,
    audio_enable: base.audio_enable,
    total_frames: totalFrames,
    input_offset_frames: 0,
    input_trim_start_frames: 0,
    input_trim_end_frames: 0,
    vae_path: base.vae_path,
    text_encoder_path: base.text_encoder_path,
    unet_quantization: base.unet_quantization,
    quantized_gemm_mode: base.quantized_gemm_mode,
    reference_image_size: referenceImageSize,
    loras: base.loras,
    blocks_to_swap: base.blocks_to_swap,
    fuse_output_proj: base.fuse_output_proj,
    fbcache_enable: base.fbcache_enable,
    fbcache_threshold: base.fbcache_threshold,
    fbcache_warmup_steps: base.fbcache_warmup_steps,
    spectrum_enable: base.spectrum_enable,
    spectrum_w: base.spectrum_w,
    spectrum_w_decay: base.spectrum_w_decay,
    spectrum_delta_cap: base.spectrum_delta_cap,
    spectrum_m: base.spectrum_m,
    spectrum_lam: base.spectrum_lam,
    spectrum_warmup_steps: base.spectrum_warmup_steps,
    spectrum_window_size: base.spectrum_window_size,
    spectrum_flex_window: base.spectrum_flex_window,
    spectrum_tail: base.spectrum_tail,
    spectrum_max_cache: base.spectrum_max_cache,
  };
}

// The `chain_vid` loop-step items (segments 2..N) to enqueue alongside the
// main segment-1 item (a plain `txt2vid`/`img2vid`/`ref2vid` item the caller
// still builds and enqueues itself, at `loopStepIndex: -1`, exactly as it
// would for an unchained request except capped at `capFrames`). Each item's
// `inputVideo` is left unset -- `advanceVideoChain` fills it in once the
// segment before it actually finishes, since it is a File built from that
// segment's OWN output, which does not exist yet at enqueue time.
export function buildChainContinuationQueueItems(args: {
  caps: ArchCapabilities | null | undefined;
  arch: string | null | undefined;
  targetFrames: number;
  capFrames: number;
  loopGroupId: string;
  continuationBase: ChainContinuationBase;
  referenceImageSize?: "max" | "match";
  // MiniMax-H3 ref2va only: the ORIGINAL image references, carried onto
  // every continuation segment (identity continuity is what ref2va is for),
  // on top of the boundary-frame anchor the placed clip itself provides.
  referenceImages?: File[];
}): Array<Omit<QueueItem, "id" | "status" | "addedAt">> {
  const plannedTotals = planVideoChainSegments(args.caps, args.arch, args.targetFrames) ?? [];
  const items: Array<Omit<QueueItem, "id" | "status" | "addedAt">> = [];
  let previous = args.capFrames;
  plannedTotals.forEach((total, index) => {
    items.push({
      type: "chain_vid",
      params: buildChainContinuationParams(args.continuationBase, total, args.referenceImageSize),
      referenceImages: args.referenceImages,
      prompt: args.continuationBase.prompt,
      loopGroupId: args.loopGroupId,
      loopStepIndex: index,
      isLoopStep: true,
      chainTargetFrames: args.targetFrames,
      chainPreviousFrames: previous,
    });
    previous = total;
  });
  return items;
}

// Fetches a completed segment's own output and wraps it as the File the next
// segment's `inputVideo` needs (POST /generate/outpaint/video takes the clip
// to extend as a multipart upload, not a URL). Mirrors the equivalent
// inline fetch-and-wrap the img2img loop-step path already does for images.
export async function fetchVideoAsFile(url: string, filename = "chain_segment.mp4"): Promise<File> {
  const response = await fetch(url);
  const blob = await response.blob();
  return new File([blob], filename, { type: blob.type || "video/mp4" });
}

export interface ChainAdvanceResult {
  /** Set only when the chain stopped for a reason worth telling the user
   *  about (no forward progress, or the architecture could not produce a
   *  further continuation) -- undefined for a normal on-plan finish, whether
   *  or not this was the chain's last segment. */
  message?: string;
}

// Called after EVERY segment of a chain finishes (the main item as well as
// each `chain_vid` step) with that segment's own reported result. Advances
// the chain: patches the next queued step's `inputVideo` + `total_frames`
// from the segment that just actually ran (not the enqueue-time estimate),
// or stops the chain (`cancelLoopGroup`) when it has reached its target, the
// architecture cannot produce a further continuation, or the segment made no
// forward progress at all (S8: a re-decoded clip reporting no more frames
// than it was fed, e.g. mp4 frame-count drift on re-upload -- without this
// check a stall like that would keep re-requesting the same length until the
// 500-segment guard in `nextVideoChainTotalFrames`/`planVideoChain`).
// A no-op (`{}`, no queue mutation) for a queue item that never belonged to a
// chain in the first place.
export async function advanceVideoChain(args: {
  caps: ArchCapabilities | null | undefined;
  arch: string | null | undefined;
  queue: QueueItem[];
  completedItem: QueueItem;
  resultFrames: number | undefined;
  resultVideoUrl: string;
  updateQueueItemByLoop: (
    loopGroupId: string,
    loopStepIndex: number,
    updates: Partial<QueueItem> | ((item: QueueItem) => Partial<QueueItem>)
  ) => void;
  cancelLoopGroup: (loopGroupId: string) => void;
}): Promise<ChainAdvanceResult> {
  const item = args.completedItem;
  if (!item.loopGroupId || item.chainTargetFrames == null) return {};

  const target = item.chainTargetFrames;
  const segmentsCompleted = (item.loopStepIndex ?? -1) + 2; // main (-1) -> 1

  if (
    item.type === "chain_vid" &&
    item.chainPreviousFrames != null &&
    args.resultFrames != null &&
    args.resultFrames <= item.chainPreviousFrames
  ) {
    args.cancelLoopGroup(item.loopGroupId);
    return {
      message:
        `Video chain stopped after segment ${segmentsCompleted}: that segment reported ` +
        `${args.resultFrames} frames, no more than the ${item.chainPreviousFrames} frames it continued from. ` +
        `${segmentsCompleted} segment(s) already completed are saved to the gallery.`,
    };
  }

  const nextStepIndex = (item.loopStepIndex ?? -1) + 1;
  const nextChainItem = args.queue.find(
    (q) => q.loopGroupId === item.loopGroupId && q.loopStepIndex === nextStepIndex && q.type === "chain_vid"
  );
  if (!nextChainItem || args.resultFrames == null) return {};

  const nextTotal = nextVideoChainTotalFrames(args.caps, args.arch, args.resultFrames, target);
  if (nextTotal == null) {
    // Reached target (normal) or the architecture cannot produce a further
    // continuation (stuck) -- either way nothing more should run; drop any
    // surplus planned step(s) left over from the enqueue-time estimate.
    args.cancelLoopGroup(item.loopGroupId);
    if (args.resultFrames < target) {
      return {
        message:
          `Video chain stopped after segment ${segmentsCompleted}: the architecture could not generate a ` +
          `further continuation. Reached ${args.resultFrames} of the requested ${target} frames. ` +
          `${segmentsCompleted} segment(s) are saved to the gallery.`,
      };
    }
    return {};
  }

  const file = await fetchVideoAsFile(args.resultVideoUrl);
  args.updateQueueItemByLoop(item.loopGroupId, nextStepIndex, (queued) => ({
    inputVideo: file,
    chainPreviousFrames: args.resultFrames,
    params: { ...(queued.params as OutpaintVideoParams), total_frames: nextTotal },
  }));
  return {};
}
