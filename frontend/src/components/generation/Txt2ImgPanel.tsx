"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { usePathname } from "next/navigation";
import { ChevronLeft, ChevronRight, X, RotateCcw } from "lucide-react";
import Card from "../common/Card";
import TabbedOptions from "../common/TabbedOptions";
import Input from "../common/Input";
import NumberInput from "../common/NumberInput";
import TextareaWithTagSuggestions from "../common/TextareaWithTagSuggestions";
import Textarea, {
  GENERATION_LYRICS_HEIGHT_KEY,
  GENERATION_NEGATIVE_PROMPT_HEIGHT_KEY,
  GENERATION_PROMPT_HEIGHT_KEY,
} from "../common/Textarea";
import Button from "../common/Button";
import Slider from "../common/Slider";
import Select from "../common/Select";
import { resolveBound } from "@/utils/paramBounds";
import ModelLoadSection from "../common/ModelLoadSection";
import LoRASelector from "../common/LoRASelector";
import ControlNetSelector from "../common/ControlNetSelector";
import MiniMaxH3ReferenceSelector, { EMPTY_MINIMAX_H3_REFERENCES, countMiniMaxH3References, MAX_VIDEOS, MAX_TOTAL } from "../common/MiniMaxH3ReferenceSelector";
import MiniMaxH3Ref2VidKeyframeSelector from "../common/MiniMaxH3Ref2VidKeyframeSelector";
import TIPODialog, { TIPOSettings } from "../common/TIPODialog";
import { fixFloatingPointParams } from "@/utils/numberUtils";
import ImageViewer from "../common/ImageViewer";
import PostEditControls from "../common/PostEditControls";
import VideoAccelerationControls from "../common/VideoAccelerationControls";
import { PostEditState, NEUTRAL_POST_EDIT, buildFilterString } from "@/utils/postEdit";
import { usePostEditPreview } from "@/hooks/usePostEditPreview";
import GenerationQueue from "../common/GenerationQueue";
import GenerationLeadGrid from "../common/GenerationLeadGrid";
import InlineHelp from "../common/InlineHelp";
import H3PromptAssist from "../common/H3PromptAssist";
import SendToStudioButton from "../studio/SendToStudioButton";
import ResizableColumns, {
  GENERATION_PREVIEW_QUEUE_SPLIT_KEY,
  GENERATION_WORKSPACE_SPLIT_KEY,
} from "../common/ResizableColumns";
import LoopGenerationPanel, { LoopGenerationConfig } from "./LoopGenerationPanel";
import QuantizedGemmSelect from "./QuantizedGemmSelect";
import VideoFrameCountSlider from "../common/VideoFrameCountSlider";
import VideoChainConfirmDialog, { VideoChainPlanInput } from "../common/VideoChainConfirmDialog";
import {
  buildChainContinuationQueueItems,
  buildChainImageReferenceInventory,
  segmentChainReferenceImages,
  segmentChainText,
  advanceVideoChain,
} from "@/utils/videoChain";
import { migrateLoopGenerationConfig, computeLoopDecodeDirective } from "@/utils/loopGenerationInheritance";
import { generateTxt2Img, generateImg2Img, generateTxt2Vid, Txt2VidParams, generateRef2Vid, Ref2VidParams, generateOutpaintVideo, OutpaintVideoParams, MiniMaxH3References, MiniMaxH3Keyframe, generateTxt2Aud, Txt2AudParams, generateTxt2ImgTrainingPreview, GenerationParams, getSamplers, getScheduleTypes, tokenizePrompt, generateTIPOPrompt, cancelGeneration, getCurrentModel, isLatentOnlyResult, getResultFilename, getResultPlaybackFilename, getResultSeed, getResultAncestralSeed, unetQuantizationOptions, normalizeUnetQuantization, transformerQuantizationLabel, archSupportsFeature, archDisplayName, normalizeVideoFrames, videoCanvasRule, videoCanvasAxisBounds, videoMinInferenceSteps, videoCanvasExceedsEnvelope, isGenerationStalledError, planVideoChain, effectiveSegmentFrames, VideoChainManifest, VIDEO_BLOCK_SWAP_MAX } from "@/utils/api";
import { useActiveTraining } from "@/hooks/useActiveTraining";
import { useSmoothProgress } from "@/hooks/useSmoothProgress";
import { wsClient, CFGMetrics } from "@/utils/websocket";
import CFGMetricsGraph from "../common/CFGMetricsGraph";
import VramInspector from "../common/VramInspector";
import { saveTempImage, loadTempImage } from "@/utils/tempImageStorage";
import { previewStorageKeys, loadVideoPreview, saveVideoPreview, loadAudioPreview, saveAudioPreview, saveImagePreview, clearVideoPreview, clearAudioPreview, clearImagePreview, outputExists, stripCacheBuster, withCacheBuster, imagePreviewGone } from "@/utils/previewStorage";
import { sendToPanel, sendImageToImg2Img, sendBase64ImageToInpaint, sendBase64ImageToUpscale, sendBase64ImageToOutpaint, sendVideoToOutpaint, sendVideoToInpaint, sendVideoToReference, sendAudioToOutpaint, sendAudioToImg2Img, fetchUrlToFile } from "@/utils/sendHelpers";
import { useStartup } from "@/contexts/StartupContext";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";
import { createH3ReferenceInventory, maybeTransformH3PromptForGeneration } from "@/utils/h3PromptAssist";
import { readGlobalAttentionType } from "@/utils/attentionSettings";

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
  cfg_schedule_type: "constant",
  cfg_schedule_min: 1.0,
  cfg_schedule_max: undefined,
  cfg_schedule_power: 2.0,
  cfg_rescale_snr_alpha: 0.0,
  dynamic_threshold_percentile: 0.0,
  dynamic_threshold_mimic_scale: 7.0,
  nag_enable: false,
  nag_scale: 5.0,
  nag_tau: 3.5,
  nag_alpha: 0.25,
  nag_sigma_end: 3.0,
  nag_negative_prompt: "",
  unet_quantization: null,
  quantized_gemm_mode: null,
  original_size_w: 0,
  original_size_h: 0,
  original_size_scale: 1.0,
  text_encoder_quantization: null,
  cpu_text_encoding: false,
  use_torch_compile: false,
  keep_models_hot: false,
  vae_tiling: false,
  vae_tile_threshold: 0,
  vae_tile_mode: "blend",
  vae_tile_global_norm: false,
  color_flatten_strength: 0,
  flatten_in_loop: false,
  flatten_in_loop_last_steps: 3,
  flatten_in_loop_min_region: 0.02,
  spectrum_enable: false,
  fbcache_enable: false,
  fbcache_threshold: 0.12,
  fbcache_warmup_steps: 1,
  spectrum_w: 0.5,
  spectrum_w_decay: 0.0,
  spectrum_delta_cap: 0.0,
  spectrum_m: 4,
  spectrum_lam: 0.1,
  spectrum_warmup_steps: 3,
  spectrum_window_size: 4,
  spectrum_flex_window: 0.75,
  spectrum_tail: 0.12,
  spectrum_feature_mode: "output",
  spectrum_cache_branch: 1,
  spectrum_max_cache: 0,
  preview_predicted_x0: false,
  preview_decoder: "matrix",
  use_tipo: false,
  enable_block_swap: false,
  blocks_to_swap: 20,
  use_pinned_memory: false,
  block_swap_h2d_only: false,
  block_swap_ring_size: 2,
  attention_type: "normal",
  vision_encoder_path: null,
  vae_path: null,
  text_encoder_path: null,
  pid_sr_output: "4x",
  pid_use_gemma: false,
  pid_low_vram: false,
  pid_tile_native: 512,
  pid_tile_overlap_ratio: 0.25,
  pid_fast_large_decode: false,
  // Video generation fields (used when a video model is loaded; the panel maps
  // these into Txt2VidParams for txt2vid requests). Carried alongside the image
  // params so a single params object drives both modes.
  num_frames: 121,
  frame_rate: 24.0,
  num_inference_steps: 8,
  guidance_scale: 1.0,
  num_videos_per_prompt: 1,
  audio_enable: true,
  max_sequence_length: 1024,
  // Video route's block swap (see GenerationParams.video_blocks_to_swap).
  // 0 = disabled, this endpoint's own default (opt-in).
  video_blocks_to_swap: 0,
  fuse_output_proj: false,
  // Music generation fields (used when an audio model (ACE-Step) is loaded;
  // the panel maps these into Txt2AudParams for txt2aud requests).
  lyrics: "",
  audio_duration: 30.0,
  inference_steps: 8,
  shift: 3.0,
  sampler_mode: "euler",
  vocal_language: "en",
};

// The valid clip lengths differ per video architecture (LTX-2.3: 8k+1;
// MiniMax-H3: 17n+5 within 124-345), so the option list comes from the
// backend's own `video_constraints` payload via videoFrameOptions() below
// rather than from a list kept here. See frontend/src/utils/api.ts.

// Txt2Img's secondary options are grouped into a single-open tabbed accordion
// (see the "Txt2Img Options" Card below, shared chrome via
// frontend/src/components/common/TabbedOptions.tsx — ported from
// OutpaintPanel/InpaintPanel's *_OPTIONS_TABS pattern). Every tab owns a
// disjoint set of param keys, used both by its "reset to default" button and
// by its active-highlight predicate (isTxt2ImgOptionsTabActive below).
// LoRA/ControlNet are left outside the tabs (they're full component
// selectors, not param groups); Sampler/Steps/CFG Scale/Seed/Width/Height
// stay outside as core fields, matching Outpaint/Inpaint. Image-only (image
// generation vs video/audio, gated by !isVideo && !isAudio at the call site).
type Txt2ImgOptionsTabId =
  | "cfg"
  | "acceleration"
  | "post_process"
  | "prompt_chunking"
  | "environment";

const TXT2IMG_OPTIONS_TABS: { id: Txt2ImgOptionsTabId; label: string }[] = [
  { id: "cfg", label: "CFG / NAG" },
  { id: "acceleration", label: "Acceleration（高速化）" },
  { id: "post_process", label: "Post-process（色補正）" },
  { id: "prompt_chunking", label: "Prompt Chunking" },
  { id: "environment", label: "Environment" },
];

const TXT2IMG_OPTIONS_TAB_KEYS: Record<Txt2ImgOptionsTabId, (keyof GenerationParams)[]> = {
  cfg: [
    "cfg_schedule_type",
    "cfg_schedule_min",
    "cfg_schedule_max",
    "cfg_schedule_power",
    "cfg_rescale_snr_alpha",
    "dynamic_threshold_percentile",
    "dynamic_threshold_mimic_scale",
    "nag_enable",
    "nag_scale",
    "nag_tau",
    "nag_alpha",
    "nag_sigma_end",
    "original_size_w",
    "original_size_h",
    "original_size_scale",
  ],
  acceleration: [
    "spectrum_enable",
    "spectrum_feature_mode",
    "spectrum_cache_branch",
    "spectrum_w",
    "spectrum_w_decay",
    "spectrum_delta_cap",
    "spectrum_m",
    "spectrum_lam",
    "spectrum_warmup_steps",
    "spectrum_window_size",
    "spectrum_flex_window",
    "spectrum_tail",
    "fbcache_enable",
    "fbcache_threshold",
    "fbcache_warmup_steps",
  ],
  post_process: [
    "color_flatten_strength",
    "flatten_in_loop",
    "flatten_in_loop_last_steps",
    "flatten_in_loop_min_region",
  ],
  prompt_chunking: [
    "prompt_chunking_mode",
    "max_prompt_chunks",
  ],
  environment: [
    "unet_quantization",
    "quantized_gemm_mode",
    "text_encoder_quantization",
    "cpu_text_encoding",
    "vae_tiling",
    "vae_tile_threshold",
    "vae_tile_mode",
    "vae_tile_global_norm",
    "use_torch_compile",
    "enable_block_swap",
    "blocks_to_swap",
    "use_pinned_memory",
    "block_swap_h2d_only",
    "block_swap_ring_size",
  ],
};

// "Active" means the group is currently doing something to the generation
// (enabled / non-neutral), not just "differs from DEFAULT_PARAMS" -- mirrors
// isOutpaintOptionsTabActive/isInpaintOptionsTabActive's rationale.
function isTxt2ImgOptionsTabActive(tabId: Txt2ImgOptionsTabId, params: GenerationParams): boolean {
  switch (tabId) {
    case "cfg":
      return (
        (params.cfg_schedule_type ?? "constant") !== "constant" ||
        (params.dynamic_threshold_percentile ?? 0) > 0 ||
        !!params.nag_enable ||
        (params.original_size_w ?? 0) > 0 ||
        (params.original_size_scale ?? 1.0) !== 1.0
      );
    case "acceleration":
      return !!params.spectrum_enable || !!params.fbcache_enable;
    case "post_process":
      return (
        (params.color_flatten_strength ?? 0) > 0 ||
        !!params.flatten_in_loop
      );
    case "prompt_chunking":
      return (
        (params.prompt_chunking_mode ?? "a1111") !== "a1111" ||
        (params.max_prompt_chunks ?? 0) > 0
      );
    case "environment":
      return (
        !!(params.unet_quantization && params.unet_quantization !== "none") ||
        !!params.quantized_gemm_mode ||
        !!(params.text_encoder_quantization && params.text_encoder_quantization !== "none") ||
        !!params.cpu_text_encoding ||
        !!params.vae_tiling ||
        !!params.use_torch_compile ||
        !!params.enable_block_swap
      );
    default:
      return false;
  }
}

const STORAGE_KEY = "txt2img_params";
const PREVIEW_STORAGE_KEY = "txt2img_preview";
// Image + video + audio preview keys for this panel. The three are mutually
// exclusive in storage (see utils/previewStorage.ts), so the newest result is
// the only one that can be restored.
const PREVIEW_KEYS = previewStorageKeys(PREVIEW_STORAGE_KEY);
const LOOP_GENERATION_STORAGE_KEY = "txt2img_loop_generation";
const REF_IMAGES_STORAGE_KEY = "txt2img_ref_images";
// Chain segment length (`chainSegmentFrames`) round-trips separately from
// `params` -- see the state declaration below for why it is never part of
// the `params` blob (it is client-side chain orchestration only, never sent
// to the backend). `null` must round-trip as `null` ("never split"), which
// JSON already does correctly (`JSON.stringify(null) === "null"`); the
// restore/persist effects only need to distinguish "key absent" (never
// persisted, also defaults to null) from "key present".
const CHAIN_SEGMENT_FRAMES_STORAGE_KEY = "txt2img_chain_segment_frames";

interface Txt2ImgPanelProps {
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint" | "outpaint" | "upscale") => void;
  // opts.kind/playbackUrl let the shared top-right strip (FloatingGallery)
  // render video/audio results correctly instead of guessing from the URL
  // extension and falling back to a non-playable master URL.
  onImageGenerated?: (imageUrl: string, opts?: { kind?: "image" | "video" | "audio"; playbackUrl?: string }) => void;
}

export default function Txt2ImgPanel({ onTabChange, onImageGenerated }: Txt2ImgPanelProps = {}) {
  const { modelLoaded, isBackendReady, generationDefaults, isVideo, isAudio, archCapabilities, resolveModality, modelInfoVersion, videoFrameSliderMax, sliderBounds } = useStartup();
  const pathname = usePathname();
  const [params, setParams] = useState<GenerationParams>(DEFAULT_PARAMS);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  // Video output (produced when a video model is loaded / txt2vid queue item).
  const [generatedVideo, setGeneratedVideo] = useState<string | null>(null);
  // MiniMax-H3 ref2va references. Kept as local UI state (they are file
  // uploads, like the aud2aud reference clip, not generation parameters) and
  // carried on the queue item so a queued request keeps the references it was
  // built with.
  const [h3References, setH3References] = useState<MiniMaxH3References>(
    EMPTY_MINIMAX_H3_REFERENCES);
  const [h3ReferenceImageSize, setH3ReferenceImageSize] = useState<"max" | "match">("max");
  // C5: optional keyframe anchors on a ref2vid request, laid out AFTER the
  // reference blocks. A separate track from `h3References` -- content
  // conditioning vs placement conditioning -- kept as its own local state for
  // the same reason (file uploads, not a generation parameter).
  const [h3Keyframes, setH3Keyframes] = useState<MiniMaxH3Keyframe[]>([]);
  const [generatedVideoInfo, setGeneratedVideoInfo] = useState<{ num_frames?: number; fps?: number; duration?: number } | null>(null);
  // A video result's `warnings[]`, rendered under the player. This panel
  // discarded them for every video item; `outpaint_video_total_frames_adjusted`
  // in particular carries a chain segment's EFFECTIVE output length, which is
  // the one number the user cannot infer from anything else on screen.
  const [generatedVideoWarnings, setGeneratedVideoWarnings] = useState<string[]>([]);
  // Seed the last video result actually ran with, so the video card's seed
  // control has the same "reuse the seed from the preview" button the image
  // path has (StoredVideoPreview carries it, as it does in OutpaintPanel).
  const [generatedVideoSeed, setGeneratedVideoSeed] = useState<number | null>(null);
  const [generatedVideoParams, setGeneratedVideoParams] = useState<GenerationParams | null>(null);
  // Opt-in video-length chaining (CLAUDE.md "opt-in long-clip feature"): set
  // when Generate is pressed with a video length above the loaded
  // architecture's single-inference cap, holding what the dialog needs to
  // offer BOTH choices (single inference at the cap, or the chain) --
  // cleared as soon as either choice is made. `null` = dialog closed.
  const [videoChainPrompt, setVideoChainPrompt] = useState<{
    videoParams: Txt2VidParams;
    isRef2Va: boolean;
    references: MiniMaxH3References;
    targetFrames: number;
    capFrames: number;
    segmentFrames: number | null;
    // Loaded variant at Generate time. The planner decides the available
    // continuation modes from the architecture/variant pair, never from a
    // checkpoint name, so it is captured here rather than re-read later.
    variant: string | null;
  } | null>(null);
  // Any-segment-of-a-chain reason the chain stopped short of its target
  // (no forward progress / architecture could not continue further) --
  // set by advanceVideoChain via processQueue, shown next to the frame
  // slider until the user starts a new chain or dismisses it.
  const [videoChainStoppedMessage, setVideoChainStoppedMessage] = useState<string | null>(null);
  // User-settable chain segment length (`chain_segment_frames`, client-side
  // orchestration only -- NEVER sent to the backend). `null` = unset = never
  // split: raising the total frame count alone splits nothing, regardless of
  // whether the loaded architecture still has a hard `max_frames` wall
  // (`planVideoChain`'s `chainSegmentCap` falls back to that wall on its own
  // when this is unset, so a still-capped architecture keeps chaining
  // automatically with no action needed here). Setting this is what turns
  // chaining into a deliberate choice, including on an architecture with no
  // hard wall at all (MiniMax-H3), e.g. to keep every segment within the
  // documented trained range even though the backend would accept one huge
  // request. A chain already enqueued is frozen at enqueue time (the value
  // is copied onto each queue item's own `chainSegmentFrames` field) and is
  // never retargeted by a later change here. Persisted separately under
  // `CHAIN_SEGMENT_FRAMES_STORAGE_KEY` (see the mount-restore/persist
  // effects below); restoring it never touches an already-enqueued item's
  // own frozen copy.
  const [chainSegmentFrames, setChainSegmentFrames] = useState<number | null>(null);
  // Set when the architecture-grid snap (below) actually changes a held
  // `chainSegmentFrames` -- including one just restored from localStorage
  // under an architecture that no longer accepts it -- so the replacement
  // is reported instead of silently applied. Same shape as InpaintPanel's
  // `regenerateRangeReplacedNotice`.
  const [chainSegmentFramesReplacedNotice, setChainSegmentFramesReplacedNotice] = useState<string | null>(null);
  // Audio output (produced when an audio model is loaded / txt2aud queue item).
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [generatedAudioInfo, setGeneratedAudioInfo] = useState<{ duration?: number; sample_rate?: number } | null>(null);
  const [generatedAudioParams, setGeneratedAudioParams] = useState<GenerationParams | null>(null);
  const [generatedImageSeed, setGeneratedImageSeed] = useState<number | null>(null);
  const [generatedImageAncestralSeed, setGeneratedImageAncestralSeed] = useState<number | null>(null);
  const [generatedImageParams, setGeneratedImageParams] = useState<GenerationParams | null>(null);
  // Client-side post-edit (brightness/saturation) for the current preview image.
  // Never sent to the backend; reset to neutral on each new generated image.
  const [postEdit, setPostEdit] = useState<PostEditState>({ ...NEUTRAL_POST_EDIT });
  // Color-flatten preview for the inline result image (b/s stay as CSS filter).
  const effectiveGeneratedImage = usePostEditPreview(generatedImage, postEdit.flatten);
  useEffect(() => {
    setPostEdit({ ...NEUTRAL_POST_EDIT });
  }, [generatedImage]);
  const [progress, setProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  // Streamed progress-phase label (e.g. "Step 12/28" or "PiD decode (tile 3/9)").
  // Rendered in place of the hardcoded "Generating..." text so decode-phase
  // status is visible; reset alongside every setProgress(0) site.
  const [progressMessage, setProgressMessage] = useState("");
  // Sub-step smoothing for the bar only; the "n/total steps" text stays integer.
  const { percent: progressPercent, reportSubProgress } = useSmoothProgress(progress, totalSteps, isGenerating);
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);
  const [isMounted, setIsMounted] = useState(false);
  const [sendImage, setSendImage] = useState(true);
  const [sendPrompt, setSendPrompt] = useState(true);
  const [sendParameters, setSendParameters] = useState(true);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  // ── Training-preview integration ────────────────────────────────────
  // When toggle ON, generation requests are routed to the in-training
  // model via /generate/txt2img/training-preview.  Result is a blob
  // (no /outputs/ file, no gallery sync — preview only).  Disabled
  // automatically when no LoRA/Full-FT training is active.
  const [useTrainingModel, setUseTrainingModel] = useState(false);
  // Sub-option: when true, the backend additionally persists the
  // preview to ``outputs/`` and inserts a GeneratedImage row so it
  // appears in the gallery (tagged as ``training-preview:...``).
  const [savePreviewToGallery, setSavePreviewToGallery] = useState(false);
  const activeTraining = useActiveTraining();
  const previewBlobUrlRef = useRef<string | null>(null);
  // Auto-untoggle when no training is active (otherwise the next
  // generate would fail with a confusing 409).
  useEffect(() => {
    if (!activeTraining && useTrainingModel) setUseTrainingModel(false);
  }, [activeTraining, useTrainingModel]);
  // Release previous blob URL when generating a new preview (or on unmount)
  useEffect(() => {
    return () => {
      if (previewBlobUrlRef.current) {
        URL.revokeObjectURL(previewBlobUrlRef.current);
        previewBlobUrlRef.current = null;
      }
    };
  }, []);
  const [currentModelInfo, setCurrentModelInfo] = useState<any>(null);
  // Keep this panel's copy of GET /models/current in step with the shared one.
  // modelInfoVersion only changes when the loaded model's identity actually
  // changes, so this costs one request per model change -- including changes
  // this page did not make (API, backend restart, another tab).
  useEffect(() => {
    if (modelInfoVersion === 0) return; // initial fetch happens on mount below
    getCurrentModel()
      .then(setCurrentModelInfo)
      .catch((error) => console.warn("[Txt2Img] Failed to refresh model info", error));
  }, [modelInfoVersion]);
  // Drop a persisted unet_quantization the loaded architecture does not offer
  // (e.g. fp8_e4m3fn carried over onto a krea2 model): otherwise the <select>
  // holds a value absent from its options and renders blank, while the panel
  // keeps sending that value.
  useEffect(() => {
    const arch = currentModelInfo?.model_info?.type as string | undefined;
    if (!archCapabilities || !arch) return;
    setParams((prev) => {
      const normalized = normalizeUnetQuantization(archCapabilities, arch, prev.unet_quantization ?? null);
      return normalized === (prev.unet_quantization ?? null) ? prev : { ...prev, unet_quantization: normalized };
    });
  }, [archCapabilities, currentModelInfo?.model_info?.type]);
  // The loaded architecture and the two capability gates the VIDEO controls
  // read. `archSupportsFeature` treats an unknown arch (or a capability matrix
  // that has not loaded) as supporting the feature, so a control is never
  // hidden merely because the matrix was unavailable.
  const loadedArch = currentModelInfo?.model_info?.type as string | undefined;
  const loadedArchName = archDisplayName(loadedArch);
  // Applies a LoRA's own declared recommended settings (from its file
  // metadata) to params, like any ordinary user edit -- writes through the
  // normal params state so it flows through the same request/DB/metadata
  // path. num_inference_steps maps to whichever step field this modality
  // actually has; audio has no fbcache/spectrum concept, so those are
  // reported back as skipped rather than silently applied.
  const applyLoraRecommended = (settings: Record<string, unknown>): string[] => {
    const skipped: string[] = [];
    const updates: Partial<GenerationParams> = {};
    if (typeof settings.num_inference_steps === "number") {
      if (isVideo) updates.num_inference_steps = settings.num_inference_steps;
      else if (isAudio) updates.inference_steps = settings.num_inference_steps;
      else updates.steps = settings.num_inference_steps;
    }
    if (typeof settings.fbcache_enable === "boolean") {
      if (isAudio) skipped.push("fbcache_enable");
      else updates.fbcache_enable = settings.fbcache_enable;
    }
    if (typeof settings.spectrum_enable === "boolean") {
      if (isAudio) skipped.push("spectrum_enable");
      else updates.spectrum_enable = settings.spectrum_enable;
    }
    setParams({ ...params, ...updates });
    return skipped;
  };
  // MiniMax-H3 ships TWO transformer partitions that share every other
  // component: `fl2va` (txt2vid / img2vid / video outpaint) and `ref2va`
  // (omni-reference). Which one is loaded IS the workflow, so the reference
  // inputs appear only for the one that was trained to read reference rows —
  // the backend refuses the other by name rather than running it.
  const isRef2Va =
    loadedArch === "minimax_h3" &&
    (currentModelInfo?.model_info?.variant as string | undefined) === "ref2va";
  const supportsCfg = archSupportsFeature(archCapabilities, loadedArch, "cfg");
  const supportsNegativePrompt = !isAudio
    && archSupportsFeature(archCapabilities, loadedArch, "negative_prompt");
  // Hide Spectrum/FBCache when the loaded sampler does not consume them; H3 now
  // supports both. This matches the other capability-gated leaf controls.
  const supportsSpectrum = archSupportsFeature(archCapabilities, loadedArch, "spectrum");
  const supportsFbcache = archSupportsFeature(archCapabilities, loadedArch, "fbcache");
  const supportsFuseOutputProj = archSupportsFeature(archCapabilities, loadedArch, "fuse_output_proj");
  // The value the video Block Swap checkbox writes when turned ON (backend
  // SSOT: param_defaults.VIDEO_GEN_DEFAULTS["blocks_to_swap_enabled_default"],
  // identical across txt2vid/img2vid/ref2vid since there is no per-arch
  // overlay for it). The `?? 40` fallback only matters before
  // /schema/generation-defaults answers.
  const videoBlocksToSwapEnabledDefault =
    (generationDefaults?.txt2vid as Record<string, unknown> | undefined)
      ?.blocks_to_swap_enabled_default as number ?? 40;
  // Snap a persisted clip length the LOADED video architecture does not accept
  // (LTX-2.3's 121 carried onto MiniMax-H3, whose grid starts at 124). Same
  // shape and same reason as the unet_quantization normaliser above: otherwise
  // the <select> holds a value absent from its options and the panel keeps
  // sending it, only for the backend to snap it and warn.
  useEffect(() => {
    if (!archCapabilities || !loadedArch) return;
    setParams((prev) => {
      const normalized = normalizeVideoFrames(archCapabilities, loadedArch, prev.num_frames ?? null);
      return normalized == null || normalized === prev.num_frames
        ? prev
        : { ...prev, num_frames: normalized };
    });
  }, [archCapabilities, loadedArch]);
  // Same snap for a held `chainSegmentFrames`: a non-null value carried over
  // from another architecture's grid -- including one just restored from
  // localStorage -- is moved to the nearest length THIS architecture
  // accepts, the same as `num_frames` above. A null value (the default --
  // "never split") is left alone; there is nothing to snap. Unlike the
  // `num_frames` effect above, a real change here is reported via
  // `chainSegmentFramesReplacedNotice` rather than applied silently -- the
  // restored/held value came from the user, not from a schema default.
  useEffect(() => {
    if (!archCapabilities || !loadedArch) return;
    if (chainSegmentFrames == null) return;
    const normalized = normalizeVideoFrames(archCapabilities, loadedArch, chainSegmentFrames);
    if (normalized == null || normalized === chainSegmentFrames) return;
    const previous = chainSegmentFrames;
    setChainSegmentFrames(normalized);
    setChainSegmentFramesReplacedNotice(
      `The chain segment length (${previous} frames) does not fit this architecture's frame grid and was replaced with ${normalized} frames.`
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [archCapabilities, loadedArch]);
  const [promptTokenCount, setPromptTokenCount] = useState<number>(0);
  const [negativePromptTokenCount, setNegativePromptTokenCount] = useState<number>(0);
  const [isTIPODialogOpen, setIsTIPODialogOpen] = useState(false);
  const [tipoSettings, setTipoSettings] = useState<TIPOSettings>({
    model_name: "KBlueLeaf/TIPO-500M",
    tag_length: "short",
    nl_length: "short",
    temperature: 0.5,
    top_p: 0.9,
    top_k: 40,
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
  const [previewViewerOpen, setPreviewViewerOpen] = useState(false);
  const [showAdvancedCFG, setShowAdvancedCFG] = useState(false);

  // FLUX.2 Image Edit: Reference images
  const [refImages, setRefImages] = useState<File[]>([]);
  const [refImagePreviews, setRefImagePreviews] = useState<string[]>([]);
  const [isRefImageDragging, setIsRefImageDragging] = useState(false);

  const [loopGenerationConfig, setLoopGenerationConfig] = useState<LoopGenerationConfig>({
    enabled: false,
    steps: [],
    decodeMode: "every",
  });
  const [isMobileControlsOpen, setIsMobileControlsOpen] = useState(true);
  const [cfgMetrics, setCfgMetrics] = useState<CFGMetrics[]>([]);
  const [developerMode, setDeveloperMode] = useState(false);

  // Cache TIPO-generated prompts for loop groups
  const tipoPromptCache = useRef<Map<string, string>>(new Map());

  const tokenizePromptTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const tokenizeNegativeTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const promptTextareaRef = useRef<HTMLTextAreaElement | null>(null);

  // TIPO: Treat as Natural Language (local state, not persisted)
  const [treatAsNL, setTreatAsNL] = useState(false);

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

  // Load from localStorage after component mounts (client-side only)
  useEffect(() => {
    // console.clear(); // Temporarily disabled for debugging
    console.log("=== Txt2ImgPanel mounted ===");
    setIsMounted(true);

    // Load current model info
    getCurrentModel().then((modelInfo) => {
      setCurrentModelInfo(modelInfo);
      console.log("[Txt2Img] Current model info:", modelInfo);
    }).catch((error) => {
      console.error("Failed to load model info:", error);
    });

    // Load params
    const saved = localStorage.getItem(STORAGE_KEY);
    const savedLength = saved ? saved.length : 0;
    console.log(`[Txt2Img] Loading params from localStorage (${savedLength} chars)`);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        console.log("[Txt2Img] Parsed params:", {
          loras: parsed.loras?.length || 0,
          controlnets: parsed.controlnets?.length || 0,
          prompt_length: parsed.prompt?.length || 0,
        });
        const merged = { ...DEFAULT_PARAMS, ...parsed };
        // Fix floating point precision issues
        const fixed = fixFloatingPointParams(merged);
        setParams(fixed);
      } catch (error) {
        console.error("Failed to load saved params:", error);
      }
    }

    // Load chain segment length. Kept out of `params` (see the state
    // declaration), so it is restored from its own key. `getItem` returning
    // `null` means the key was never written (default: unset/never-split);
    // a written `"null"` (from `JSON.stringify(null)`) parses back to the
    // JS `null` a set-but-then-unchecked box would have written, so both
    // cases land on `null` without a fallback default masking either one.
    const savedChainSegmentFrames = localStorage.getItem(CHAIN_SEGMENT_FRAMES_STORAGE_KEY);
    if (savedChainSegmentFrames !== null) {
      try {
        const parsedChainSegmentFrames = JSON.parse(savedChainSegmentFrames);
        setChainSegmentFrames(typeof parsedChainSegmentFrames === "number" ? parsedChainSegmentFrames : null);
      } catch (error) {
        console.error("Failed to load saved chain segment length:", error);
      }
    }

    // Load preview image

    const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
    if (savedPreview) {
      setGeneratedImage(savedPreview);
    }

    // Load preview video (txt2vid / ref2vid result). Restored unconditionally:
    // the player itself is gated on `isVideo`, which arrives asynchronously from
    // useStartup(), so nothing renders until the loaded arch is known to be a
    // video arch. The URL is verified once the backend is ready (below).
    const savedVideo = loadVideoPreview(PREVIEW_KEYS);
    if (savedVideo) {
      setGeneratedVideo(savedVideo.url);
      setGeneratedVideoInfo(savedVideo.info);
      setGeneratedVideoSeed(savedVideo.seed ?? null);
    }

    // Load preview audio (txt2aud result). Same reasoning as the video above:
    // restored unconditionally because the <audio> render site is gated on
    // `isAudio` from useStartup(), which arrives asynchronously, so nothing
    // plays until the loaded arch is known to be an audio arch.
    const savedAudio = loadAudioPreview(PREVIEW_KEYS);
    if (savedAudio) {
      setGeneratedAudio(savedAudio.url);
      setGeneratedAudioInfo(savedAudio.info);
    }

    // Load resolution step setting
    const savedResolutionStep = localStorage.getItem('resolution_step');
    if (savedResolutionStep) {
      setResolutionStep(parseInt(savedResolutionStep));
    }

    // Load developer mode
    const savedDeveloperMode = localStorage.getItem('developer_mode');
    if (savedDeveloperMode === 'true') {
      setDeveloperMode(true);
    }

    // Load advanced CFG settings visibility
    const savedShowAdvancedCFG = localStorage.getItem('show_advanced_cfg');
    if (savedShowAdvancedCFG === 'true') {
      setShowAdvancedCFG(true);
    }

    // Load attention type from global settings
    const savedAttentionType = readGlobalAttentionType();
    if (savedAttentionType) {
      setParams(prev => ({ ...prev, attention_type: savedAttentionType }));
    }

    // Load custom presets
    const savedAspectRatioPresets = localStorage.getItem('aspect_ratio_presets');
    if (savedAspectRatioPresets) {
      try {
        setAspectRatioPresets(JSON.parse(savedAspectRatioPresets));
      } catch (e) {
        console.error('Failed to parse aspect ratio presets:', e);
      }
    }

    const savedFixedResolutionPresets = localStorage.getItem('fixed_resolution_presets');
    if (savedFixedResolutionPresets) {
      try {
        setFixedResolutionPresets(JSON.parse(savedFixedResolutionPresets));
      } catch (e) {
        console.error('Failed to parse fixed resolution presets:', e);
      }
    }

    // Load panel visibility settings
    const savedVisibility = localStorage.getItem('txt2img_visibility');
    if (savedVisibility) {
      try {
        setVisibility(JSON.parse(savedVisibility));
      } catch (e) {
        console.error('Failed to parse txt2img visibility:', e);
      }
    }

    // Load loop generation config
    const savedLoopGen = localStorage.getItem(LOOP_GENERATION_STORAGE_KEY);
    if (savedLoopGen) {
      try {
        setLoopGenerationConfig(migrateLoopGenerationConfig(JSON.parse(savedLoopGen)));
      } catch (e) {
        console.error('Failed to parse loop generation config:', e);
      }
    }

    // Load reference images (FLUX.2 Image Edit)
    const loadRefImages = async () => {
      const savedRefImageRefs = localStorage.getItem(REF_IMAGES_STORAGE_KEY);
      if (savedRefImageRefs) {
        try {
          const refRefs: string[] = JSON.parse(savedRefImageRefs);
          console.log(`[Txt2Img] Loading ${refRefs.length} reference images from storage`);

          const loadedPreviews: string[] = [];
          for (const ref of refRefs) {
            try {
              const imageData = await loadTempImage(ref);
              if (imageData) {
                loadedPreviews.push(imageData);
              }
            } catch (error) {
              console.error(`[Txt2Img] Failed to load reference image ${ref}:`, error);
            }
          }

          if (loadedPreviews.length > 0) {
            setRefImagePreviews(loadedPreviews);
            console.log(`[Txt2Img] Restored ${loadedPreviews.length} reference images`);
          }
        } catch (error) {
          console.error('[Txt2Img] Failed to parse reference images storage:', error);
        }
      }
    };
    loadRefImages();

  }, []);

  // Listen for localStorage changes from Gallery/Preview (send to feature)
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      // Only react to changes in our storage key from other tabs/windows
      if (e.key === STORAGE_KEY && e.newValue) {
        try {
          const parsed = JSON.parse(e.newValue);
          console.log("[Txt2Img] Received params from Gallery via storage event:", {
            prompt_length: parsed.prompt?.length || 0,
            steps: parsed.steps,
            cfg_scale: parsed.cfg_scale,
          });
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          const fixed = fixFloatingPointParams(merged);
          setParams(fixed);
        } catch (error) {
          console.error("[Txt2Img] Failed to parse storage change:", error);
        }
      }
    };

    // Custom event for same-tab localStorage changes (Gallery -> Generate panel)
    const handleCustomStorageChange = () => {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          console.log("[Txt2Img] Received params from Gallery via custom event:", {
            prompt_length: parsed.prompt?.length || 0,
            steps: parsed.steps,
            cfg_scale: parsed.cfg_scale,
          });
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          const fixed = fixFloatingPointParams(merged);
          setParams(fixed);
        } catch (error) {
          console.error("[Txt2Img] Failed to parse custom storage change:", error);
        }
      }
    };

    // storage event only fires for changes from OTHER tabs/windows
    window.addEventListener('storage', handleStorageChange);
    // Custom event for same-tab changes (triggered by ImageGrid)
    window.addEventListener('txt2img_params_updated', handleCustomStorageChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('txt2img_params_updated', handleCustomStorageChange);
    };
  }, []);

  // "Use as reference video" (gallery, video results): appends the clip to the
  // ref2va reference track. Whole-clip content conditioning, not a placement
  // anchor -- see sendVideoToReference in sendHelpers.ts.
  useEffect(() => {
    const handleReferenceVideoUpdate = async () => {
      const url = localStorage.getItem("h3_reference_video");
      if (!url) return;
      try {
        const file = await fetchUrlToFile(url);
        setH3References(prev => {
          if (prev.videos.length >= MAX_VIDEOS || countMiniMaxH3References(prev) >= MAX_TOTAL) {
            console.warn("[Txt2Img] Reference video not added: track is full");
            return prev;
          }
          return { ...prev, videos: [...prev.videos, file], videoAudios: [...prev.videoAudios, null] };
        });
      } catch (error) {
        console.error("[Txt2Img] Failed to load sent reference video:", error);
      } finally {
        localStorage.removeItem("h3_reference_video");
      }
    };
    window.addEventListener("h3_reference_video_updated", handleReferenceVideoUpdate);
    return () => window.removeEventListener("h3_reference_video_updated", handleReferenceVideoUpdate);
  }, []);

  // Reload params from localStorage when navigating to /generate (from Gallery)
  useEffect(() => {
    if (pathname === "/generate" && isMounted) {
      console.log("[Txt2Img] Page navigated to /generate, reloading params from localStorage");
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          const fixed = fixFloatingPointParams(merged);
          setParams(fixed);
          console.log("[Txt2Img] Params reloaded:", {
            prompt_length: fixed.prompt?.length || 0,
            steps: fixed.steps,
            cfg_scale: fixed.cfg_scale,
          });
        } catch (error) {
          console.error("[Txt2Img] Failed to reload params on navigation:", error);
        }
      }
    }
  }, [pathname, isMounted]);

  // Reload images when backend becomes ready
  useEffect(() => {
    if (!isBackendReady) return;

    console.log("[Txt2Img] Backend ready, reloading preview image if needed");

    // Reload the preview image if it's a backend URL, and verify it is still
    // there first (outputs/ can be cleared, or the run deleted from the
    // gallery) -- exactly what the video and audio branches below do.
    // Non-`/outputs/` values (a data: URL, a blob:, a path served from
    // elsewhere) are left untouched: they cannot go missing server-side and
    // must never be stamped or discarded. The stamp is applied only to a URL
    // that verified, and it replaces any earlier stamp rather than appending.
    const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
    if (savedPreview && savedPreview.startsWith('/outputs/')) {
      const previewPath = stripCacheBuster(savedPreview);
      outputExists(previewPath).then((exists) => {
        if (!exists) {
          console.log("[Txt2Img] Stored preview image is gone, clearing:", previewPath);
          clearImagePreview(PREVIEW_KEYS);
          setGeneratedImage(null);
          return;
        }
        console.log("[Txt2Img] Reloading preview image from backend:", previewPath);
        setGeneratedImage(withCacheBuster(previewPath));
      });
    }

    // Verify the restored preview video still exists (outputs/ can be cleared,
    // or the run deleted from the gallery). No cache-busting timestamp here --
    // an .mp4 is large and its URL is stable.
    const savedVideo = loadVideoPreview(PREVIEW_KEYS);
    if (savedVideo) {
      outputExists(savedVideo.url).then((exists) => {
        if (!exists) {
          console.log("[Txt2Img] Stored preview video is gone, clearing:", savedVideo.url);
          clearVideoPreview(PREVIEW_KEYS);
          setGeneratedVideo(null);
          setGeneratedVideoInfo(null);
          setGeneratedVideoSeed(null);
        }
      });
    }

    // Same verification for a restored preview audio clip.
    const savedAudio = loadAudioPreview(PREVIEW_KEYS);
    if (savedAudio) {
      outputExists(savedAudio.url).then((exists) => {
        if (!exists) {
          console.log("[Txt2Img] Stored preview audio is gone, clearing:", savedAudio.url);
          clearAudioPreview(PREVIEW_KEYS);
          setGeneratedAudio(null);
          setGeneratedAudioInfo(null);
        }
      });
    }
  }, [isBackendReady]);

  // Reset torch.compile when developer mode is disabled
  useEffect(() => {
    if (!developerMode) {
      setParams(prev => {
        if (prev.use_torch_compile) {
          return { ...prev, use_torch_compile: false };
        }
        return prev;
      });
    }
  }, [developerMode]);

  // Load samplers and schedule types immediately on mount (don't wait for model)
  useEffect(() => {
    loadSamplers();
    loadScheduleTypes();
  }, []); // Empty deps - load once on mount

  // Save params to localStorage whenever they change (but only after mounted)
  useEffect(() => {
    if (isMounted) {
      // ControlNet images are now managed by ControlNetSelector via tempImageStorage
      // We don't need to remove image_base64 here anymore, as it's no longer stored in params
      console.log("[Txt2Img] Saving params to localStorage:", {
        loras: params.loras?.length || 0,
        controlnets: params.controlnets?.length || 0,
        prompt_length: params.prompt?.length || 0,
        // Don't log full params to avoid base64 spam
      });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
    }
  }, [params, isMounted]);

  // Save preview image to localStorage whenever it changes
  useEffect(() => {
    if (isMounted && generatedImage) {
      saveImagePreview(PREVIEW_KEYS, generatedImage);
    }
  }, [generatedImage, isMounted]);

  // Save preview video to localStorage whenever it changes. Only the URL and
  // the frame/fps/duration line are stored -- never the clip bytes.
  useEffect(() => {
    if (isMounted && generatedVideo) {
      saveVideoPreview(PREVIEW_KEYS, {
        url: generatedVideo,
        info: generatedVideoInfo,
        seed: generatedVideoSeed,
      });
    }
  }, [generatedVideo, generatedVideoInfo, generatedVideoSeed, isMounted]);

  // Save preview audio to localStorage whenever it changes. Only the URL and
  // the duration/sample-rate line are stored -- never the audio bytes.
  useEffect(() => {
    if (isMounted && generatedAudio) {
      saveAudioPreview(PREVIEW_KEYS, { url: generatedAudio, info: generatedAudioInfo });
    }
  }, [generatedAudio, generatedAudioInfo, isMounted]);

  // Save loop generation config to localStorage whenever it changes
  useEffect(() => {
    if (isMounted) {
      localStorage.setItem(LOOP_GENERATION_STORAGE_KEY, JSON.stringify(loopGenerationConfig));
    }
  }, [loopGenerationConfig, isMounted]);

  // Save chain segment length whenever it changes. `null` is written
  // explicitly (`JSON.stringify(null) === "null"`) rather than removing the
  // key, so a reload can tell "explicitly unset" apart from "never saved" --
  // both currently mean the same default, but writing `null` keeps that an
  // intentional round trip rather than an accident of `JSON.stringify`
  // dropping `undefined` keys.
  useEffect(() => {
    if (isMounted) {
      localStorage.setItem(CHAIN_SEGMENT_FRAMES_STORAGE_KEY, JSON.stringify(chainSegmentFrames));
    }
  }, [chainSegmentFrames, isMounted]);

  // Apply backend-fetched defaults when they arrive (only if no localStorage value exists)
  useEffect(() => {
    if (!generationDefaults) return;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      setParams(prev => ({ ...DEFAULT_PARAMS, ...(generationDefaults.txt2img as Partial<typeof DEFAULT_PARAMS>) }));
    }
  }, [generationDefaults]);

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

    // Use generated image params if available, otherwise fall back to current UI params
    const sourceParams = generatedImageParams || params;

    console.log("[Txt2Img] sendToImg2Img - sendPrompt:", sendPrompt, "sendParameters:", sendParameters);
    console.log("[Txt2Img] sendToImg2Img - sourceParams.prompt:", sourceParams.prompt);

    // Send prompt and/or parameters
    sendToPanel(sourceParams, "img2img_params", {
      sendPrompt,
      sendParameters,
      includeDenoising: false,
      dispatchEvent: "img2img_params_updated"
    });

    console.log("[Txt2Img] sendToImg2Img - Sent to panel");

    // Navigate to img2img tab
    if (onTabChange) {
      onTabChange("img2img");
    }
  };

  const sendToUpscale = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }

    try {
      await sendBase64ImageToUpscale(generatedImage);
    } catch (error) {
      console.error("[Txt2Img] Failed to send image to upscale:", error);
    }

    if (onTabChange) {
      onTabChange("upscale");
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
        await sendBase64ImageToInpaint(generatedImage);
      } catch (error) {
        console.error("[Txt2Img] Failed to send image to inpaint:", error);
      }
    }

    // Use generated image params if available, otherwise fall back to current UI params
    const sourceParams = generatedImageParams || params;

    console.log("[Txt2Img] sendToInpaint - sendPrompt:", sendPrompt, "sendParameters:", sendParameters);
    console.log("[Txt2Img] sendToInpaint - sourceParams.prompt:", sourceParams.prompt);

    // Send prompt and/or parameters
    sendToPanel(sourceParams, "inpaint_params", {
      sendPrompt,
      sendParameters,
      includeDenoising: false,
      dispatchEvent: "inpaint_params_updated"
    });

    console.log("[Txt2Img] sendToInpaint - Sent to panel");

    // Navigate to inpaint tab
    if (onTabChange) {
      onTabChange("inpaint");
    }
  };

  const sendToOutpaint = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }

    // Send image if checked
    if (sendImage) {
      try {
        await sendBase64ImageToOutpaint(generatedImage);
      } catch (error) {
        console.error("[Txt2Img] Failed to send image to outpaint:", error);
      }
    }

    // Use generated image params if available, otherwise fall back to current UI params
    const sourceParams = generatedImageParams || params;

    // Send prompt and/or parameters
    sendToPanel(sourceParams, "outpaint_params", {
      sendPrompt,
      sendParameters,
      includeDenoising: false,
      dispatchEvent: "outpaint_params_updated"
    });

    // Navigate to outpaint tab
    if (onTabChange) {
      onTabChange("outpaint");
    }
  };

  // generatedVideo (Txt2Vid) result -> Outpaint's outpaint_vid clip input.
  const sendVideoResultToOutpaint = async () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    try {
      await sendVideoToOutpaint(generatedVideo);
    } catch (error) {
      console.error("[Txt2Img] Failed to send video to outpaint:", error);
      alert("Failed to send the video to outpaint");
      return;
    }
    if (onTabChange) onTabChange("outpaint");
  };

  // generatedVideo (Txt2Vid) result -> Inpaint's temporal inpaint clip input.
  const sendVideoResultToInpaint = async () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    try {
      await sendVideoToInpaint(generatedVideo);
    } catch (error) {
      console.error("[Txt2Img] Failed to send video to inpaint:", error);
      alert("Failed to send the video to inpaint");
      return;
    }
    if (onTabChange) onTabChange("inpaint");
  };

  // generatedVideo (Txt2Vid) result -> the ref2va reference track (whole-clip
  // conditioning, not a placement anchor -- see sendVideoToReference).
  const sendVideoResultToReference = () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    sendVideoToReference(generatedVideo);
  };

  // generatedAudio (Txt2Aud) result -> Outpaint's outpaint_aud clip input.
  const sendAudioResultToOutpaint = () => {
    if (!generatedAudio) {
      alert("No audio to send");
      return;
    }
    sendAudioToOutpaint(generatedAudio);
    if (onTabChange) onTabChange("outpaint");
  };

  // generatedAudio (Txt2Aud) result -> Img2Img as the aud2aud reference clip.
  const sendAudioResultToImg2Img = () => {
    if (!generatedAudio) {
      alert("No audio to send");
      return;
    }
    sendAudioToImg2Img(generatedAudio);
    if (onTabChange) onTabChange("img2img");
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
      // Fallback: set common samplers if API fails
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
      console.log("[Txt2Img] Calling getScheduleTypes()...");
      const data = await getScheduleTypes();
      console.log("[Txt2Img] Received schedule types:", data.schedule_types);
      setScheduleTypes(data.schedule_types);
      console.log("[Txt2Img] setScheduleTypes called");
    } catch (error) {
      console.error("Failed to load schedule types:", error);
      // Fallback: set common schedule types if API fails
      setScheduleTypes([
        { id: "uniform", name: "Uniform" },
        { id: "karras", name: "Karras" },
        { id: "exponential", name: "Exponential" },
        { id: "sgm_uniform", name: "SGM Uniform" },
        { id: "simple", name: "Simple" },
        { id: "ddim_uniform", name: "DDIM Uniform" },
      ]);
    }
  };

  // FLUX.2 Image Edit: Reference image handlers
  const handleRefImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const newFiles = Array.from(files).slice(0, 10 - refImagePreviews.length); // Max 10 total
    const newPreviews: string[] = [];
    const newRefs: string[] = [];

    for (const file of newFiles) {
      const reader = new FileReader();
      reader.onload = async (event) => {
        if (event.target?.result) {
          const base64Data = event.target.result as string;
          newPreviews.push(base64Data);

          // Save to tempImageStorage
          try {
            const ref = await saveTempImage(base64Data);
            newRefs.push(ref);
          } catch (error) {
            console.error("[Txt2Img] Failed to save reference image to temp storage:", error);
          }

          if (newPreviews.length === newFiles.length) {
            // Use functional setState to get the latest state
            setRefImagePreviews((prevPreviews) => [...prevPreviews, ...newPreviews]);

            // Update localStorage with refs
            const savedRefImageRefs = localStorage.getItem(REF_IMAGES_STORAGE_KEY);
            const existingRefs = savedRefImageRefs ? JSON.parse(savedRefImageRefs) : [];
            const allRefs = [...existingRefs, ...newRefs];
            localStorage.setItem(REF_IMAGES_STORAGE_KEY, JSON.stringify(allRefs));
            console.log(`[Txt2Img] Saved ${newRefs.length} reference images to storage`);
          }
        }
      };
      reader.readAsDataURL(file);
    }

    // Use functional setState to get the latest state
    setRefImages((prevFiles) => [...prevFiles, ...newFiles]);
  };

  const handleRemoveRefImage = (index: number) => {
    setRefImages(refImages.filter((_, i) => i !== index));
    setRefImagePreviews(refImagePreviews.filter((_, i) => i !== index));

    // Remove from localStorage
    const savedRefImageRefs = localStorage.getItem(REF_IMAGES_STORAGE_KEY);
    if (savedRefImageRefs) {
      try {
        const refRefs: string[] = JSON.parse(savedRefImageRefs);
        const updatedRefs = refRefs.filter((_, i) => i !== index);
        localStorage.setItem(REF_IMAGES_STORAGE_KEY, JSON.stringify(updatedRefs));
        console.log(`[Txt2Img] Removed reference image ${index} from storage`);
      } catch (error) {
        console.error("[Txt2Img] Failed to update reference images storage:", error);
      }
    }
  };

  const handleClearAllRefImages = () => {
    setRefImages([]);
    setRefImagePreviews([]);

    // Clear localStorage
    localStorage.removeItem(REF_IMAGES_STORAGE_KEY);
    console.log("[Txt2Img] Cleared all reference images from storage");
  };

  const handleRefImageDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsRefImageDragging(false);

    const files = e.dataTransfer.files;
    if (!files || files.length === 0) return;

    const imageFiles = Array.from(files)
      .filter(file => file.type.startsWith('image/'))
      .slice(0, 10 - refImagePreviews.length); // Max 10 total

    if (imageFiles.length === 0) return;

    const newPreviews: string[] = [];
    const newRefs: string[] = [];

    for (const file of imageFiles) {
      const reader = new FileReader();
      reader.onload = async (event) => {
        if (event.target?.result) {
          const base64Data = event.target.result as string;
          newPreviews.push(base64Data);

          // Save to tempImageStorage
          try {
            const ref = await saveTempImage(base64Data);
            newRefs.push(ref);
          } catch (error) {
            console.error("[Txt2Img] Failed to save reference image to temp storage:", error);
          }

          if (newPreviews.length === imageFiles.length) {
            // Use functional setState to get the latest state
            setRefImagePreviews((prevPreviews) => [...prevPreviews, ...newPreviews]);

            // Update localStorage with refs
            const savedRefImageRefs = localStorage.getItem(REF_IMAGES_STORAGE_KEY);
            const existingRefs = savedRefImageRefs ? JSON.parse(savedRefImageRefs) : [];
            const allRefs = [...existingRefs, ...newRefs];
            localStorage.setItem(REF_IMAGES_STORAGE_KEY, JSON.stringify(allRefs));
            console.log(`[Txt2Img] Saved ${newRefs.length} reference images to storage (D&D)`);
          }
        }
      };
      reader.readAsDataURL(file);
    }

    // Use functional setState to get the latest state
    setRefImages((prevFiles) => [...prevFiles, ...imageFiles]);
  };

  const handleRefImageDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsRefImageDragging(true);
  };

  const handleRefImageDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    // Only set to false if leaving the drop area entirely (not entering child elements)
    if (e.currentTarget.contains(e.relatedTarget as Node)) {
      return;
    }
    setIsRefImageDragging(false);
  };

  const handleGenerateTIPO = async () => {
    // Use params.prompt directly, or selection if user has selected text
    const textarea = promptTextareaRef.current;
    let inputPrompt = params.prompt;

    // If textarea is available and user has selected text, use only the selection
    if (textarea) {
      const selectionStart = textarea.selectionStart;
      const selectionEnd = textarea.selectionEnd;
      const hasSelection = selectionStart !== selectionEnd;

      if (hasSelection) {
        inputPrompt = params.prompt.substring(selectionStart, selectionEnd);
      }
    }

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
        model_name: tipoSettings.model_name,
        tag_length: tipoSettings.tag_length,
        nl_length: tipoSettings.nl_length,
        temperature: tipoSettings.temperature,
        top_p: tipoSettings.top_p,
        top_k: tipoSettings.top_k,
        max_new_tokens: tipoSettings.max_new_tokens,
        category_order: categoryOrder,
        enabled_categories: enabledCategories,
        treat_as_nl: treatAsNL
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

  const { addToQueue, updateQueueItem, updateQueueItemByLoop, cancelLoopGroup, startNextInQueue, completeCurrentItem, failCurrentItem, currentItem, queue, generateForever, setGenerateForever, progressSnapshot, completedResults, publishCompletedResult } = useGenerationQueue();

  // Use refs for WebSocket callback to prevent recreations
  const isGeneratingRef = useRef(isGenerating);
  const developerModeRef = useRef(developerMode);

  useEffect(() => {
    isGeneratingRef.current = isGenerating;
  }, [isGenerating]);

  useEffect(() => {
    developerModeRef.current = developerMode;
  }, [developerMode]);

  useEffect(() => {
    if (!currentItem || !["txt2img", "img2img", "txt2vid", "ref2vid", "txt2aud", "chain_vid"].includes(currentItem.type)) {
      isGeneratingRef.current = false;
      setIsGenerating(false);
      return;
    }
    isGeneratingRef.current = true;
    setIsGenerating(true);
    if (progressSnapshot?.itemId !== currentItem.id) return;
    setProgress(progressSnapshot.step);
    setTotalSteps(progressSnapshot.totalSteps);
    setProgressMessage(progressSnapshot.message);
    reportSubProgress(progressSnapshot.step, progressSnapshot.subProgress);
    if (progressSnapshot.previewImage) setPreviewImage(progressSnapshot.previewImage);
  }, [currentItem, progressSnapshot, reportSubProgress]);

  useEffect(() => {
    const result = completedResults.txt2img;
    if (!result || (currentItem && ["txt2img", "img2img", "txt2vid", "ref2vid", "txt2aud", "chain_vid"].includes(currentItem.type))) return;
    setPreviewImage(null);
    if (result.kind === "image") {
      setGeneratedImage(result.url);
      setGeneratedImageSeed(result.seed ?? null);
      setGeneratedImageAncestralSeed(result.ancestralSeed ?? null);
      setGeneratedImageParams(result.params as GenerationParams);
      setGeneratedVideo(null);
      setGeneratedAudio(null);
    } else if (result.kind === "video") {
      setGeneratedVideo(result.url);
      setGeneratedVideoInfo(result.info as typeof generatedVideoInfo);
      setGeneratedVideoSeed(result.seed ?? null);
      setGeneratedVideoParams(result.params as GenerationParams);
      setGeneratedImage(null);
      setGeneratedAudio(null);
    } else {
      setGeneratedAudio(result.url);
      setGeneratedAudioInfo(result.info as typeof generatedAudioInfo);
      setGeneratedAudioParams(result.params as GenerationParams);
      setGeneratedImage(null);
      setGeneratedVideo(null);
    }
  }, [completedResults.txt2img, currentItem]);

  // WebSocket progress callback - stable reference
  const handleProgress = useCallback((step: number, totalSteps: number, message: string, preview?: string, metrics?: CFGMetrics, subProgress?: number) => {
    if (isGeneratingRef.current) {
      setProgress(step);
      setTotalSteps(totalSteps);
      setProgressMessage(message || "");
      reportSubProgress(step, subProgress);
      if (preview) {
        setPreviewImage(preview);
      }
      if (metrics && developerModeRef.current) {
        setCfgMetrics(prev => [...prev, metrics]);
      }
    }
  }, [reportSubProgress]); // reportSubProgress is stable

  // Setup WebSocket connection - runs once
  useEffect(() => {
    wsClient.connect();
    wsClient.subscribe(handleProgress);

    return () => {
      wsClient.unsubscribe(handleProgress);
    };
  }, [handleProgress]); // handleProgress is now stable

  const [showForeverMenu, setShowForeverMenu] = useState(false);
  const [menuPosition, setMenuPosition] = useState({ x: 0, y: 0 });
  const longPressTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isLongPressTriggeredRef = useRef(false);
  const longPressPositionRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const [resolutionStep, setResolutionStep] = useState(64);
  const [aspectRatioPresets, setAspectRatioPresets] = useState<Array<{ label: string; ratio: number }>>([
    { label: "1:1", ratio: 1 / 1 },
    { label: "4:3", ratio: 4 / 3 },
    { label: "3:4", ratio: 3 / 4 },
    { label: "16:9", ratio: 16 / 9 },
    { label: "9:16", ratio: 9 / 16 },
    { label: "21:9", ratio: 21 / 9 },
    { label: "9:21", ratio: 9 / 21 },
    { label: "3:2", ratio: 3 / 2 },
    { label: "2:3", ratio: 2 / 3 },
    { label: "5:4", ratio: 5 / 4 },
  ]);
  const [fixedResolutionPresets, setFixedResolutionPresets] = useState<Array<{ width: number; height: number }>>([
    { width: 768, height: 1152 },
    { width: 1152, height: 768 },
    { width: 1248, height: 720 },
    { width: 720, height: 1248 },
    { width: 960, height: 1344 },
    { width: 1344, height: 960 },
    { width: 1024, height: 1152 },
    { width: 1152, height: 1024 },
    { width: 1024, height: 1024 },
    { width: 896, height: 1152 },
    { width: 1152, height: 896 },
    { width: 832, height: 1216 },
    { width: 1216, height: 832 },
    { width: 640, height: 1536 },
    { width: 1536, height: 640 },
    { width: 512, height: 512 },
  ]);

  // Panel visibility settings
  const [visibility, setVisibility] = useState({
    lora: true,
    controlnet: true,
    aspectRatioPresets: true,
    fixedResolutionPresets: true,
    advancedSettings: true,
  });

  const clearLongPressTimer = () => {
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
  };

  const toFixedViewportPosition = (clientX: number, clientY: number) => {
    const visualViewport = window.visualViewport;
    return {
      x: clientX + (visualViewport?.offsetLeft ?? 0),
      y: clientY + (visualViewport?.offsetTop ?? 0),
    };
  };

  const handleGenerateTouchStart = (e: React.TouchEvent<HTMLButtonElement>) => {
    isLongPressTriggeredRef.current = false;
    const touch = e.touches[0];
    if (touch) {
      const pos = toFixedViewportPosition(touch.clientX, touch.clientY);
      longPressPositionRef.current = pos;
    }
    clearLongPressTimer();
    longPressTimerRef.current = setTimeout(() => {
      isLongPressTriggeredRef.current = true;
      setMenuPosition({
        x: Math.max(16, longPressPositionRef.current.x - 80),
        y: Math.max(16, longPressPositionRef.current.y - 56),
      });
      setShowForeverMenu(true);
    }, 500);
  };

  const handleGenerateTouchEnd = () => {
    clearLongPressTimer();
  };

  // Add generation request to queue
  const handleAddToQueue = async () => {
    if (!params.prompt) {
      alert("Please enter a prompt");
      return;
    }

    // Which endpoint this request goes to is decided from a FRESH read of
    // GET /models/current, not from the cached isVideo/isAudio render flags:
    // the model can change under an open page (API call, backend restart,
    // second tab), and routing a still-image request at a video model costs a
    // 400 whose message is about the wrong thing. The cached flags remain the
    // render-time hint; only the dispatch decision is re-verified.
    const modality = await resolveModality();
    const videoMode = modality.isVideo;
    const audioMode = modality.isAudio;

    // Import wildcard replacement function dynamically
    const { replaceWildcardsInPrompt } = await import("@/utils/wildcardStorage");

    // Replace wildcards in prompts
    let processedPrompt = await replaceWildcardsInPrompt(params.prompt);
    const processedNegativePrompt = supportsNegativePrompt
      ? await replaceWildcardsInPrompt(params.negative_prompt)
      : "";

    if (videoMode && modality.modelInfo?.type === "minimax_h3") {
      const promptMode = modality.modelInfo?.variant === "ref2va" && countMiniMaxH3References(h3References) > 0
        ? "ref2va"
        : "t2va";
      try {
        const assisted = await maybeTransformH3PromptForGeneration({
          prompt: processedPrompt,
          mode: promptMode,
          // S9: no single generation request this app ever sends is longer
          // than the architecture's single-inference cap (a chain's segment 1
          // is capped, same as an unchained request) -- Prompt Assist must be
          // told that duration, never a held value above the cap, or the
          // resulting prompt (reused verbatim by every segment when chaining)
          // describes an arc far longer than any one segment actually spans.
          durationSeconds: effectiveSegmentFrames(archCapabilities, loadedArch, params.num_frames ?? 121, chainSegmentFrames) / (params.frame_rate ?? 24),
          references: createH3ReferenceInventory({
            pictures: h3References.images.length + h3Keyframes.length,
            videos: h3References.videos.length,
            audios: h3References.audios.length + h3References.videoAudios.filter(Boolean).length,
          }),
        });
        processedPrompt = assisted.prompt;
      } catch (error: any) {
        alert(error?.message || "MiniMax H3 Prompt Assist failed");
        return;
      }
    }

    // Prepare TIPO config if use_tipo is enabled
    let tipo_config = undefined;
    if (params.use_tipo) {
      const categoryOrder = tipoSettings.categories.map((c: any) => c.id);
      const enabledCategories: Record<string, boolean> = {};
      tipoSettings.categories.forEach((c: any) => {
        enabledCategories[c.id] = c.enabled;
      });

      tipo_config = {
        model_name: tipoSettings.model_name,
        tag_length: tipoSettings.tag_length,
        nl_length: tipoSettings.nl_length,
        temperature: tipoSettings.temperature,
        top_p: tipoSettings.top_p,
        top_k: tipoSettings.top_k,
        max_new_tokens: tipoSettings.max_new_tokens,
        category_order: categoryOrder,
        enabled_categories: enabledCategories,
        treat_as_nl: treatAsNL  // Add local state
      };
    }

    // Create loop group ID if loop generation is enabled
    const loopGroupId = loopGenerationConfig.enabled ? `loop_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` : undefined;
    const hasEnabledLoopSteps = loopGenerationConfig.enabled && loopGenerationConfig.steps.some(s => s.enabled);
    // Main step decode directive: resizeMode is moot for the main step (it has
    // none of its own) — passing "latent" correctly forces loop_decode="none"
    // for decodeMode "final-only" when loop steps follow (txt2img/img2img
    // support latent passthrough for their main step).
    const mainDecodeDirective = computeLoopDecodeDirective({
      decodeMode: loopGenerationConfig.decodeMode ?? "every",
      isFinalStep: !hasEnabledLoopSteps,
      resizeMode: "latent",
      supportsLatentPassthrough: true,
    });

    // Audio mode: an audio model (ACE-Step) is loaded -> enqueue a txt2aud item
    // built from the shared params. Audio loop-generation is out of scope;
    // enqueue one item. Checked before the video branch (mutually exclusive).
    if (audioMode) {
      const audioParams: Txt2AudParams = {
        prompt: processedPrompt,
        lyrics: params.lyrics,
        audio_duration: params.audio_duration,
        seed: params.seed,
        inference_steps: params.inference_steps,
        guidance_scale: params.guidance_scale,
        shift: params.shift,
        sampler_mode: params.sampler_mode,
        vocal_language: params.vocal_language,
        loras: params.loras,
        // Weight-only quantization (both axes). The panel controls are rendered
        // from arch capabilities, and `acestep` is now in runtime_int8_archs +
        // quantized_linear_archs, so these must be carried into the audio
        // params or the UI value is silently dropped.
        unet_quantization: params.unet_quantization,
        quantized_gemm_mode: params.quantized_gemm_mode,
      };
      addToQueue({
        type: "txt2aud",
        params: audioParams as any,
        prompt: processedPrompt,
      });
      return;
    }

    // Video mode: a video model is loaded -> enqueue a txt2vid item built from
    // the shared params. Video loop-generation is out of scope; enqueue one item.
    if (videoMode) {
      const videoParams: Txt2VidParams = {
        prompt: processedPrompt,
        negative_prompt: processedNegativePrompt,
        width: params.width,
        height: params.height,
        num_frames: params.num_frames,
        frame_rate: params.frame_rate,
        num_inference_steps: params.num_inference_steps,
        guidance_scale: params.guidance_scale,
        seed: params.seed,
        num_videos_per_prompt: params.num_videos_per_prompt,
        max_sequence_length: params.max_sequence_length,
        audio_enable: params.audio_enable,
        vae_path: params.vae_path,
        text_encoder_path: params.text_encoder_path,
        // Only "int8" is applied on LTX-2.3 (one-time in-place conversion of the
        // video DiT); other values warn and are ignored server-side.
        unet_quantization: params.unet_quantization,
        // ltx2 is in quantized_linear_archs, so the QuantizedGemmSelect control
        // is rendered for a loaded LTX-2.3 model and must actually be sent.
        quantized_gemm_mode: params.quantized_gemm_mode,
        // Applied by MiniMax-H3; accepted-and-warned by LTX-2.3 (no video LoRA
        // loader). Same selector/list as image generation's `params.loras`.
        loras: params.loras,
        // Distinct from the image mode's model-global `params.blocks_to_swap`
        // (see GenerationParams.video_blocks_to_swap's own comment).
        blocks_to_swap: params.video_blocks_to_swap,
        // MiniMax-H3 only, not bit-exact -- see Txt2VidParams.fuse_output_proj.
        fuse_output_proj: params.fuse_output_proj,
        // Acceleration: FBCache/Spectrum share the same params fields as image
        // mode (see VideoAccelerationControls' mutual-exclusion note above).
        fbcache_enable: params.fbcache_enable,
        fbcache_threshold: params.fbcache_threshold,
        fbcache_warmup_steps: params.fbcache_warmup_steps,
        spectrum_enable: params.spectrum_enable,
        spectrum_w: params.spectrum_w,
        spectrum_w_decay: params.spectrum_w_decay,
        spectrum_delta_cap: params.spectrum_delta_cap,
        spectrum_m: params.spectrum_m,
        spectrum_lam: params.spectrum_lam,
        spectrum_warmup_steps: params.spectrum_warmup_steps,
        spectrum_window_size: params.spectrum_window_size,
        spectrum_flex_window: params.spectrum_flex_window,
        spectrum_tail: params.spectrum_tail,
        spectrum_max_cache: params.spectrum_max_cache,
      };
      // MiniMax-H3 ref2va with at least one reference goes to the dedicated
      // omni-reference endpoint instead: it is a different request (12
      // heterogeneous files, whose order is semantic), and it is the only thing
      // the loaded transformer partition was trained for. With no references
      // the same partition still serves a plain text-to-video request.
      // ref2va-ness is read from the same fresh fetch as the modality, not from
      // this panel's own copy: if the model changed under the page, the copy
      // can still be a render behind at this point.
      const freshIsRef2Va =
        modality.modelInfo?.type === "minimax_h3" && modality.modelInfo?.variant === "ref2va";
      const isRef2VaRequest = freshIsRef2Va && countMiniMaxH3References(h3References) > 0;
      const fullVideoParams: Txt2VidParams = isRef2VaRequest
        ? ({
            ...videoParams,
            reference_image_size: h3ReferenceImageSize,
            // C5: anchors ride along on the same request when any are set.
            keyframes: h3Keyframes.length > 0 ? h3Keyframes : undefined,
          } as Ref2VidParams)
        : videoParams;

      // Opt-in video-length chaining (CLAUDE.md "opt-in long-clip feature"):
      // a held length above the loaded architecture's single-inference cap
      // is never enqueued (chained or clamped) without the user picking one
      // of the two choices explicitly -- see VideoChainConfirmDialog.
      const chainPlan = planVideoChain(archCapabilities, loadedArch, params.num_frames ?? 0, chainSegmentFrames);
      if (chainPlan != null) {
        setVideoChainPrompt({
          videoParams: fullVideoParams,
          isRef2Va: isRef2VaRequest,
          references: h3References,
          targetFrames: params.num_frames ?? 0,
          capFrames: chainPlan.capFrames,
          segmentFrames: chainSegmentFrames,
          variant: modality.modelInfo?.variant ?? null,
        });
        return;
      }

      if (isRef2VaRequest) {
        addToQueue({
          type: "ref2vid",
          params: fullVideoParams as any,
          references: h3References,
          prompt: processedPrompt,
        });
        return;
      }
      addToQueue({
        type: "txt2vid",
        params: videoParams as any,
        prompt: processedPrompt,
      });
      return;
    }

    // Add main generation to queue
    // Debug log for quantization
    if (params.unet_quantization) {
      console.log('[Txt2Img] Adding to queue with quantization:', params.unet_quantization);
    }

    addToQueue({
      type: "txt2img",
      params: {
        ...params,
        prompt: processedPrompt,
        negative_prompt: processedNegativePrompt,
        tipo_config: tipo_config,  // TIPO config will be sent to backend
        loop_decode: mainDecodeDirective.loop_decode,
        skip_gallery: mainDecodeDirective.skip_gallery,
      },
      prompt: processedPrompt,
      loopGroupId,
      loopStepIndex: loopGroupId ? -1 : undefined, // -1 indicates main generation
      isLoopStep: false,
      useTrainingModel,
      trainingRunId: activeTraining?.run_id,
    });

    // If loop generation is enabled, add all loop steps immediately
    // Use the processed (and potentially TIPO-generated) prompt for all loop steps
    if (loopGenerationConfig.enabled && loopGroupId) {
      await addLoopStepsToQueueImmediate({
        ...params,
        prompt: processedPrompt,  // Keep the same prompt for all loop steps
        negative_prompt: processedNegativePrompt,
      } as GenerationParams, loopGroupId);
    }
  };

  // Opt-in video-length chain, Choice 1 (DEFAULT): a single inference at the
  // architecture's cap, snapped -- exactly the pre-chain-feature enqueue
  // path, just with num_frames clamped down to what one request can produce.
  const handleVideoChainGenerateAtCap = () => {
    if (!videoChainPrompt) return;
    const { videoParams, isRef2Va, references, capFrames } = videoChainPrompt;
    const cappedParams: Txt2VidParams = { ...videoParams, num_frames: capFrames };
    if (isRef2Va) {
      addToQueue({
        type: "ref2vid",
        params: cappedParams as any,
        references,
        prompt: cappedParams.prompt,
      });
    } else {
      addToQueue({
        type: "txt2vid",
        params: cappedParams as any,
        prompt: cappedParams.prompt,
      });
    }
    setVideoChainPrompt(null);
  };

  // Opt-in video-length chain, Choice 2 (explicit, never the default):
  // enqueue the whole chain as a loop group -- a main segment at the cap
  // (loopStepIndex -1) plus one `chain_vid` loop step per continuation
  // (loopStepIndex 0..N-2). No isGenerating gate: this only enqueues, exactly
  // like `handleVideoChainGenerateAtCap` and every other Add-to-Queue path,
  // so the two buttons in the dialog behave identically regardless of whether
  // a generation is already running (see videoChain.ts for why this runs on
  // the queue at all).
  // `manifest` is the plan the user approved in the dialog; `null` is the
  // legacy repeat mode they picked by name. Everything a segment says (prompt)
  // and is conditioned on (which image references) is fixed onto the queue
  // items HERE, so a later change to this panel's prompt, segment length or
  // references cannot reach a chain that is already enqueued.
  const handleVideoChainStart = (manifest: VideoChainManifest | null) => {
    if (!videoChainPrompt) return;
    const { videoParams, isRef2Va, references, targetFrames, capFrames, segmentFrames } = videoChainPrompt;
    setVideoChainPrompt(null);

    const loopGroupId = `chain_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const referenceImages = references.images && references.images.length > 0 ? references.images : undefined;
    const mainText = segmentChainText(manifest, 0, {
      prompt: videoParams.prompt,
      negative_prompt: videoParams.negative_prompt,
    });
    const cappedParams: Txt2VidParams = {
      ...videoParams,
      num_frames: capFrames,
      prompt: mainText.prompt,
      negative_prompt: mainText.negative_prompt,
    };
    // Segment 0 obeys the same binding as every other segment: a reference the
    // user unbound from it is not sent, and the manifest's prompt tokens were
    // renumbered for exactly the set that IS sent.
    const mainReferenceImages = segmentChainReferenceImages(manifest, 0, referenceImages);
    const mainReferences: MiniMaxH3References | undefined = isRef2Va
      ? { ...references, images: mainReferenceImages ?? [] }
      : undefined;

    addToQueue({
      type: isRef2Va ? "ref2vid" : "txt2vid",
      params: cappedParams as any,
      references: mainReferences,
      prompt: cappedParams.prompt,
      loopGroupId,
      loopStepIndex: -1,
      isLoopStep: false,
      chainTargetFrames: targetFrames,
      // Frozen at enqueue time -- see `chainSegmentFrames` state's own
      // comment. A later change to that control never retargets this chain.
      chainSegmentFrames: segmentFrames,
      chainManifestId: manifest?.chain_id,
      chainPlanHash: manifest?.plan_hash,
      chainSegmentIndex: manifest ? 0 : undefined,
    });

    const continuationItems = buildChainContinuationQueueItems({
      caps: archCapabilities,
      arch: loadedArch,
      targetFrames,
      capFrames,
      segmentFrames,
      loopGroupId,
      continuationBase: videoParams,
      referenceImageSize: isRef2Va ? (videoParams as Ref2VidParams).reference_image_size : undefined,
      referenceImages: isRef2Va ? referenceImages : undefined,
      manifest,
    });
    continuationItems.forEach((item) => addToQueue(item));
  };

  // Add loop generation steps to queue immediately (without base image URL)
  const addLoopStepsToQueueImmediate = useCallback(async (mainParams: GenerationParams, loopGroupId: string) => {
    if (!loopGenerationConfig.enabled || loopGenerationConfig.steps.length === 0) {
      return;
    }

    console.log('[Txt2Img] Adding loop steps with mainParams.unet_quantization:', mainParams.unet_quantization);

    const { replaceWildcardsInPrompt } = await import("@/utils/wildcardStorage");
    const enabledSteps = loopGenerationConfig.steps.filter(step => step.enabled);

    for (let i = 0; i < enabledSteps.length; i++) {
      const step = enabledSteps[i];

      // Calculate size based on mode
      let stepWidth: number;
      let stepHeight: number;

      if (step.sizeMode === "scale") {
        // Scale mode: calculate from main params
        stepWidth = Math.round(mainParams.width * (step.scale || 1.0));
        stepHeight = Math.round(mainParams.height * (step.scale || 1.0));
      } else {
        // Absolute mode: use step's dimensions or fallback to main params
        stepWidth = step.width || mainParams.width;
        stepHeight = step.height || mainParams.height;
      }

      // Prepare params for this loop step
      const stepParams: any = {
        prompt: mainParams.prompt,
        negative_prompt: mainParams.negative_prompt,
        width: stepWidth,
        height: stepHeight,
        denoising_strength: step.denoisingStrength,
        img2img_fix_steps: step.doFullSteps,
        resize_mode: step.resizeMode,
        resampling_method: step.resamplingMethod,
        unet_quantization: mainParams.unet_quantization, // Inherit quantization from main
        quantized_gemm_mode: mainParams.quantized_gemm_mode, // Inherit quantized GEMM path from main
        original_size_w: mainParams.original_size_w,
        original_size_h: mainParams.original_size_h,
        original_size_scale: mainParams.original_size_scale,
        cpu_text_encoding: mainParams.cpu_text_encoding, // Inherit CPU text encoding setting
        use_torch_compile: mainParams.use_torch_compile, // Inherit torch.compile setting
        keep_models_hot: mainParams.keep_models_hot, // Inherited default; queue dispatch overrides based on hasNext
        vae_tiling: mainParams.vae_tiling, // Inherit VAE tiling setting
        vae_tile_threshold: mainParams.vae_tile_threshold, // Inherit VAE tile threshold
        vae_tile_mode: mainParams.vae_tile_mode, // Inherit VAE tile join mode
        vae_tile_global_norm: mainParams.vae_tile_global_norm, // Inherit two-pass global GroupNorm stats
        color_flatten_strength: mainParams.color_flatten_strength, // Inherit Color Flatten setting
        flatten_in_loop: mainParams.flatten_in_loop, // Inherit in-loop background flatten setting
        flatten_in_loop_last_steps: mainParams.flatten_in_loop_last_steps,
        flatten_in_loop_min_region: mainParams.flatten_in_loop_min_region,
        spectrum_enable: mainParams.spectrum_enable, // Inherit Spectrum acceleration
        fbcache_enable: mainParams.fbcache_enable, // Inherit First Block Cache
        fbcache_threshold: mainParams.fbcache_threshold,
        fbcache_warmup_steps: mainParams.fbcache_warmup_steps,
        spectrum_w: mainParams.spectrum_w,
        spectrum_w_decay: mainParams.spectrum_w_decay,
        spectrum_delta_cap: mainParams.spectrum_delta_cap,
        spectrum_m: mainParams.spectrum_m,
        spectrum_lam: mainParams.spectrum_lam,
        spectrum_warmup_steps: mainParams.spectrum_warmup_steps,
        spectrum_window_size: mainParams.spectrum_window_size,
        spectrum_flex_window: mainParams.spectrum_flex_window,
        spectrum_tail: mainParams.spectrum_tail,
        spectrum_feature_mode: mainParams.spectrum_feature_mode,
        spectrum_cache_branch: mainParams.spectrum_cache_branch,
        spectrum_max_cache: mainParams.spectrum_max_cache,
        attention_type: mainParams.attention_type, // Inherit attention backend (NAG/NegPip)
        preview_predicted_x0: mainParams.preview_predicted_x0, // Inherit preview mode
        preview_decoder: mainParams.preview_decoder, // Inherit preview decoder
        // Model/Environment (model-global) — always inherited from main. Gap fixes:
        // text_encoder_quantization + block swap group were previously never copied.
        text_encoder_quantization: mainParams.text_encoder_quantization,
        enable_block_swap: mainParams.enable_block_swap,
        blocks_to_swap: mainParams.blocks_to_swap,
        block_swap_h2d_only: mainParams.block_swap_h2d_only,
        block_swap_ring_size: mainParams.block_swap_ring_size,
        vae_path: mainParams.vae_path, // Inherit VAE override (model-global)
        text_encoder_path: mainParams.text_encoder_path, // Inherit TE override (model-global)
        pid_sr_output: mainParams.pid_sr_output, // Inherit PiD decoder options (model-global)
        pid_use_gemma: mainParams.pid_use_gemma,
        pid_low_vram: mainParams.pid_low_vram,
        pid_tile_native: mainParams.pid_tile_native,
        pid_tile_overlap_ratio: mainParams.pid_tile_overlap_ratio,
        pid_fast_large_decode: mainParams.pid_fast_large_decode,
      };

      // Genre-based inheritance. Each genre toggle defaults to the legacy combined
      // flag (then true). When OFF, per-step fields win, falling back to MAIN values
      // (never hardcoded literals). See utils/loopGenerationInheritance.ts.
      const useMainSampling = step.use_main_sampling ?? step.useMainSettings ?? true;
      const useMainCfgSchedule = step.use_main_cfg_schedule ?? step.useMainSettings ?? true;
      const useMainNag = step.use_main_nag ?? step.useMainSettings ?? true;

      // Sampling genre
      if (useMainSampling) {
        stepParams.steps = mainParams.steps;
        stepParams.cfg_scale = mainParams.cfg_scale;
        stepParams.sampler = mainParams.sampler;
        stepParams.schedule_type = mainParams.schedule_type;
        stepParams.seed = mainParams.seed;
        stepParams.ancestral_seed = mainParams.ancestral_seed;
      } else {
        stepParams.steps = step.steps ?? mainParams.steps;
        stepParams.cfg_scale = step.cfgScale ?? mainParams.cfg_scale;
        stepParams.sampler = step.sampler || mainParams.sampler;
        stepParams.schedule_type = step.scheduleType || mainParams.schedule_type;
        stepParams.seed = step.seed ?? mainParams.seed;
        stepParams.ancestral_seed = step.ancestralSeed ?? mainParams.ancestral_seed;
      }

      // Advanced CFG genre
      if (useMainCfgSchedule) {
        stepParams.cfg_schedule_type = mainParams.cfg_schedule_type;
        stepParams.cfg_schedule_min = mainParams.cfg_schedule_min;
        stepParams.cfg_schedule_max = mainParams.cfg_schedule_max;
        stepParams.cfg_schedule_power = mainParams.cfg_schedule_power;
        stepParams.cfg_rescale_snr_alpha = mainParams.cfg_rescale_snr_alpha;
        stepParams.dynamic_threshold_percentile = mainParams.dynamic_threshold_percentile;
        stepParams.dynamic_threshold_mimic_scale = mainParams.dynamic_threshold_mimic_scale;
      } else {
        stepParams.cfg_schedule_type = step.cfg_schedule_type ?? mainParams.cfg_schedule_type;
        stepParams.cfg_schedule_min = step.cfg_schedule_min ?? mainParams.cfg_schedule_min;
        stepParams.cfg_schedule_max = step.cfg_schedule_max ?? mainParams.cfg_schedule_max;
        stepParams.cfg_schedule_power = step.cfg_schedule_power ?? mainParams.cfg_schedule_power;
        stepParams.cfg_rescale_snr_alpha = step.cfg_rescale_snr_alpha ?? mainParams.cfg_rescale_snr_alpha;
        stepParams.dynamic_threshold_percentile = step.dynamic_threshold_percentile ?? mainParams.dynamic_threshold_percentile;
        stepParams.dynamic_threshold_mimic_scale = step.dynamic_threshold_mimic_scale ?? mainParams.dynamic_threshold_mimic_scale;
      }

      // NAG genre
      if (useMainNag) {
        stepParams.nag_enable = mainParams.nag_enable;
        stepParams.nag_scale = mainParams.nag_scale;
        stepParams.nag_tau = mainParams.nag_tau;
        stepParams.nag_alpha = mainParams.nag_alpha;
        stepParams.nag_sigma_end = mainParams.nag_sigma_end;
        stepParams.nag_negative_prompt = mainParams.nag_negative_prompt;
      } else {
        stepParams.nag_enable = step.nag_enable ?? mainParams.nag_enable;
        stepParams.nag_scale = step.nag_scale ?? mainParams.nag_scale;
        stepParams.nag_tau = step.nag_tau ?? mainParams.nag_tau;
        stepParams.nag_alpha = step.nag_alpha ?? mainParams.nag_alpha;
        stepParams.nag_sigma_end = step.nag_sigma_end ?? mainParams.nag_sigma_end;
        stepParams.nag_negative_prompt = step.nag_negative_prompt ?? mainParams.nag_negative_prompt;
      }

      // Apply LoRA inheritance
      stepParams.loras = step.useMainLoRAs ? (mainParams.loras || []) : [];

      // Apply reference images inheritance
      if (step.useMainRefImages ?? true) {
        stepParams.ref_images = refImages.length > 0 ? refImages : undefined;
      }

      // Apply ControlNet inheritance
      if (step.useMainControlNets) {
        stepParams.controlnets = mainParams.controlnets || [];
      } else {
        // Use step's custom ControlNets, but filter out image_base64 for useLoopImage
        console.log(`[Txt2Img] Loop step ${i}: Processing ${step.controlnets?.length || 0} ControlNets`);
        stepParams.controlnets = (step.controlnets || []).map((cn, idx) => {
          console.log(`[Txt2Img] ControlNet ${idx}: model=${cn.model_path}, useLoopImage=${cn.useLoopImage}, has_image=${!!cn.image_base64}`);
          return {
            ...cn,
            // If useLoopImage is true, set image_base64 to empty (will be filled after generation)
            image_base64: cn.useLoopImage ? "" : cn.image_base64,
          };
        });
      }

      // Force image resize mode if ControlNet is present
      if (stepParams.controlnets.length > 0) {
        stepParams.resize_mode = "image";
      }

      // Decode directive: computed AFTER the ControlNet resize_mode force above,
      // since a ControlNet-conditioned step always needs a decoded image
      // regardless of the user's selected upscale resize mode.
      const isFinalStep = i === enabledSteps.length - 1;
      const decodeDirective = computeLoopDecodeDirective({
        decodeMode: loopGenerationConfig.decodeMode ?? "every",
        isFinalStep,
        resizeMode: stepParams.resize_mode as "image" | "latent",
        supportsLatentPassthrough: true, // txt2img loop steps are always img2img (latent passthrough supported)
      });
      stepParams.loop_decode = decodeDirective.loop_decode;
      stepParams.skip_gallery = decodeDirective.skip_gallery;

      stepParams.prompt_chunking_mode = mainParams.prompt_chunking_mode;
      stepParams.max_prompt_chunks = mainParams.max_prompt_chunks;

      const processedPrompt = await replaceWildcardsInPrompt(stepParams.prompt);
      const processedNegativePrompt = await replaceWildcardsInPrompt(stepParams.negative_prompt);

      addToQueue({
        type: "img2img",
        params: {
          ...stepParams,
          prompt: processedPrompt,
          negative_prompt: processedNegativePrompt,
        },
        inputImage: "", // Will be set when main generation completes
        prompt: `[Loop ${i + 1}/${enabledSteps.length}] ${processedPrompt.substring(0, 50)}...`,
        loopGroupId,
        loopStepIndex: i,
        isLoopStep: true,
        useTrainingModel,
        trainingRunId: activeTraining?.run_id,
      });
    }

    console.log(`[Txt2Img] Added ${enabledSteps.length} loop steps to queue with group ID: ${loopGroupId}`);
  }, [loopGenerationConfig, addToQueue, refImages, useTrainingModel, activeTraining]);

  // Add loop generation steps to queue after main generation completes (legacy - not used anymore)
  const addLoopStepsToQueue = useCallback(async (baseImageUrl: string, mainParams: GenerationParams, loopGroupId: string) => {
    if (!loopGenerationConfig.enabled || loopGenerationConfig.steps.length === 0) {
      return;
    }

    const { replaceWildcardsInPrompt } = await import("@/utils/wildcardStorage");
    const enabledSteps = loopGenerationConfig.steps.filter(step => step.enabled);

    for (let i = 0; i < enabledSteps.length; i++) {
      const step = enabledSteps[i];
      const previousImageUrl = i === 0 ? baseImageUrl : null; // First step uses main output, others will chain

      // Prepare params for this loop step
      const stepParams: any = {
        prompt: mainParams.prompt,
        negative_prompt: mainParams.negative_prompt,
        width: step.width || mainParams.width,
        height: step.height || mainParams.height,
        denoising_strength: step.denoisingStrength,
        img2img_fix_steps: step.doFullSteps,
        resize_mode: step.resizeMode,
        resampling_method: step.resamplingMethod,
      };

      // Use custom settings or inherit from main
      if (step.useMainSettings) {
        stepParams.steps = mainParams.steps;
        stepParams.cfg_scale = mainParams.cfg_scale;
        stepParams.sampler = mainParams.sampler;
        stepParams.schedule_type = mainParams.schedule_type;
        stepParams.seed = mainParams.seed;
        stepParams.ancestral_seed = mainParams.ancestral_seed;
      } else {
        stepParams.steps = step.steps || 20;
        stepParams.cfg_scale = step.cfgScale || 7;
        stepParams.sampler = step.sampler || mainParams.sampler;
        stepParams.schedule_type = step.scheduleType || mainParams.schedule_type;
        stepParams.seed = step.seed ?? -1;
        stepParams.ancestral_seed = step.ancestralSeed ?? -1;
        // Use step's Advanced CFG or defaults
        stepParams.cfg_schedule_type = step.cfg_schedule_type || "constant";
        stepParams.cfg_schedule_min = step.cfg_schedule_min ?? 1.0;
        stepParams.cfg_schedule_max = step.cfg_schedule_max;
        stepParams.cfg_schedule_power = step.cfg_schedule_power ?? 2.0;
        stepParams.cfg_rescale_snr_alpha = step.cfg_rescale_snr_alpha ?? 0.0;
        stepParams.dynamic_threshold_percentile = step.dynamic_threshold_percentile ?? 0.0;
        stepParams.dynamic_threshold_mimic_scale = step.dynamic_threshold_mimic_scale ?? 7.0;
        // Use step's NAG or defaults
        stepParams.nag_enable = step.nag_enable ?? false;
        stepParams.nag_scale = step.nag_scale ?? 5.0;
        stepParams.nag_tau = step.nag_tau ?? 3.5;
        stepParams.nag_alpha = step.nag_alpha ?? 0.25;
        stepParams.nag_sigma_end = step.nag_sigma_end ?? 3.0;
        stepParams.nag_negative_prompt = step.nag_negative_prompt ?? "";
      }

      stepParams.loras = mainParams.loras || [];
      stepParams.controlnets = mainParams.controlnets || [];
      stepParams.prompt_chunking_mode = mainParams.prompt_chunking_mode;
      stepParams.max_prompt_chunks = mainParams.max_prompt_chunks;
      stepParams.unet_quantization = mainParams.unet_quantization;
      stepParams.quantized_gemm_mode = mainParams.quantized_gemm_mode;
      stepParams.original_size_w = mainParams.original_size_w;
      stepParams.original_size_h = mainParams.original_size_h;
      stepParams.original_size_scale = mainParams.original_size_scale;
      stepParams.cpu_text_encoding = mainParams.cpu_text_encoding;
      stepParams.vision_encoder_path = mainParams.vision_encoder_path;
      stepParams.vae_path = mainParams.vae_path;
      stepParams.text_encoder_path = mainParams.text_encoder_path;
      stepParams.pid_sr_output = mainParams.pid_sr_output;
      stepParams.pid_use_gemma = mainParams.pid_use_gemma;
      stepParams.pid_low_vram = mainParams.pid_low_vram;
      stepParams.pid_tile_native = mainParams.pid_tile_native;
      stepParams.pid_tile_overlap_ratio = mainParams.pid_tile_overlap_ratio;
      stepParams.pid_fast_large_decode = mainParams.pid_fast_large_decode;

      const processedPrompt = await replaceWildcardsInPrompt(stepParams.prompt);
      const processedNegativePrompt = await replaceWildcardsInPrompt(stepParams.negative_prompt);

      addToQueue({
        type: "img2img",
        params: {
          ...stepParams,
          prompt: processedPrompt,
          negative_prompt: processedNegativePrompt,
        },
        inputImage: previousImageUrl || "", // Will be updated to previous output for chained steps
        prompt: `[Loop ${i + 1}/${enabledSteps.length}] ${processedPrompt.substring(0, 50)}...`,
        loopGroupId,
        loopStepIndex: i,
        isLoopStep: true,
        useTrainingModel,
        trainingRunId: activeTraining?.run_id,
      });
    }

    console.log(`[Txt2Img] Added ${enabledSteps.length} loop steps to queue with group ID: ${loopGroupId}`);
  }, [loopGenerationConfig, addToQueue, useTrainingModel, activeTraining]);

  // Process queue - automatically start next item
  const processQueueRef = useRef<() => Promise<void>>();

  const processQueue = useCallback(async () => {
    console.log("[Txt2Img] processQueue called, isGenerating:", isGeneratingRef.current);
    if (isGeneratingRef.current) {
      console.log("[Txt2Img] Already generating, skipping");
      return;
    }

    const nextItem = startNextInQueue(["txt2img", "img2img", "txt2vid", "ref2vid", "txt2aud", "chain_vid"]);
    console.log("[Txt2Img] Next item from queue:", nextItem);
    if (!nextItem) {
      console.log("[Txt2Img] No items in queue");
      return;
    }

    // Audio branch: txt2aud item (an audio model is loaded). Produces a .flac
    // and renders an <audio> instead of an <img>. No loop-generation handling.
    if (nextItem.type === "txt2aud") {
      isGeneratingRef.current = true;
      setIsGenerating(true);
      setProgress(0);
      setProgressMessage("");
      setTotalSteps((nextItem.params as any).inference_steps || 8);
      setPreviewImage(null);
      setGeneratedImage(null);
      // An audio run supersedes any image/video result still on screen; the
      // stored preview is only replaced once this run actually succeeds.
      setGeneratedAudio(null);
      setGeneratedAudioInfo(null);
      setGeneratedVideo(null);
      setGeneratedVideoInfo(null);
      try {
        const result = await generateTxt2Aud(nextItem.params as Txt2AudParams);
        const audioUrl = `/outputs/${result.image.filename}`;
        const audioInfo = {
          duration: result.image.duration,
          sample_rate: result.image.sample_rate,
        };
        const audioParams = {
          ...(nextItem.params as GenerationParams),
          seed: getResultSeed(result) ?? (nextItem.params as GenerationParams).seed,
        };
        setGeneratedAudio(audioUrl);
        setGeneratedAudioInfo(audioInfo);
        setGeneratedAudioParams(audioParams);
        publishCompletedResult({ panel: "txt2img", kind: "audio", url: audioUrl, info: audioInfo, params: audioParams });
        if (onImageGenerated) onImageGenerated(audioUrl, { kind: "audio" });
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        completeCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
      } catch (error: any) {
        console.error("[Txt2Img] txt2aud generation failed:", error);
        // alert() blocks the JS thread; reset state and requeue before showing it,
        // otherwise the queue effect sees a stale isGenerating until the dialog closes.
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        failCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
        alert(isGenerationStalledError(error) ? error.message : "txt2aud generation failed. Please check console for details.");
      }
      return;
    }

    // Video branch: txt2vid, or ref2vid when the loaded MiniMax-H3 partition is
    // ref2va and the request carries references. Both produce an .mp4 and
    // render a <video> instead of an <img>; no loop-generation handling.
    if (nextItem.type === "txt2vid" || nextItem.type === "ref2vid") {
      isGeneratingRef.current = true;
      setIsGenerating(true);
      setProgress(0);
      setProgressMessage("");
      setTotalSteps((nextItem.params as any).num_inference_steps || 8);
      setPreviewImage(null);
      setGeneratedImage(null);
      setGeneratedVideo(null);
      setGeneratedVideoInfo(null);
      setGeneratedVideoSeed(null);
      setGeneratedVideoWarnings([]);
      setGeneratedAudio(null);
      setGeneratedAudioInfo(null);
      try {
        const result = nextItem.type === "ref2vid"
          ? await generateRef2Vid(
              nextItem.params as Ref2VidParams,
              nextItem.references ?? EMPTY_MINIMAX_H3_REFERENCES)
          : await generateTxt2Vid(nextItem.params as Txt2VidParams);
        const videoUrl = `/outputs/${getResultFilename(result)}`;
        const videoPlaybackFilename = getResultPlaybackFilename(result);
        const videoPlaybackUrl = videoPlaybackFilename ? `/outputs/${videoPlaybackFilename}` : videoUrl;
        const videoInfo = {
          num_frames: result.image.num_frames,
          fps: result.image.fps,
          duration: result.image.duration,
        };
        const videoSeed = getResultSeed(result);
        setGeneratedVideo(videoUrl);
        setGeneratedVideoInfo(videoInfo);
        // The seed the run actually used (-1 in the request means "pick one"),
        // so the seed control's reuse button can pin it for the next run.
        setGeneratedVideoSeed(videoSeed);
        setGeneratedVideoParams(nextItem.params as GenerationParams);
        setGeneratedVideoWarnings(
          (result.warnings || []).map((w: any) => (typeof w === "string" ? w : w?.message)).filter(Boolean));
        publishCompletedResult({
          panel: "txt2img", kind: "video", url: videoUrl,
          playbackUrl: videoPlaybackUrl !== videoUrl ? videoPlaybackUrl : undefined,
          info: videoInfo, seed: videoSeed, params: nextItem.params,
        });
        if (onImageGenerated) {
          onImageGenerated(videoUrl, {
            kind: "video",
            playbackUrl: videoPlaybackUrl !== videoUrl ? videoPlaybackUrl : undefined,
          });
        }

        // Video-length chain (this segment may be segment 1 of one): advance
        // to the next queued step, or stop the chain with a reason. A no-op
        // for a plain, unchained txt2vid/ref2vid item.
        const chainOutcome = await advanceVideoChain({
          caps: archCapabilities,
          arch: loadedArch,
          queue,
          completedItem: nextItem,
          resultFrames: result.image?.num_frames,
          resultVideoUrl: videoUrl,
          updateQueueItemByLoop,
          cancelLoopGroup,
        });
        setVideoChainStoppedMessage(chainOutcome.message ?? null);

        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        completeCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
      } catch (error: any) {
        console.error(`[Txt2Img] ${nextItem.type} generation failed:`, error);
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        // A deliberate cancel (cancelGeneration()) surfaces here as the
        // backend's RuntimeError, not as a distinct error type -- read it as
        // a cancel, not a generic failure, and say how many chain segments
        // (if any) already completed and are in the gallery.
        const isCancelled = String(error?.message || error?.response?.data?.detail || "").toLowerCase().includes("cancel");
        const isChainSegment = !!nextItem.loopGroupId && nextItem.chainTargetFrames != null;
        failCurrentItem();
        // A chain segment failing or being cancelled must not let the
        // remaining pending "chain_vid" steps of the SAME loop group dispatch
        // next -- each of them would otherwise fail its own generic "no
        // input video" error (their `inputVideo` is only filled in by a
        // segment that finishes successfully), cascading into one alert per
        // remaining segment. Drop them all up front instead; already
        // completed segments stay in the gallery.
        if (isChainSegment) {
          cancelLoopGroup(nextItem.loopGroupId!);
        }
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
        if (isChainSegment) {
          const completedSegments = (nextItem.loopStepIndex ?? -1) + 1;
          const reason = isCancelled ? "cancelled" : "stopped: a segment failed";
          alert(completedSegments > 0
            ? `Video chain ${reason}. ${completedSegments} segment(s) completed before this are saved to the gallery.`
            : `Video chain ${reason} before any segment completed.`);
        } else if (!isCancelled) {
          alert(isGenerationStalledError(error)
            ? error.message
            : `${nextItem.type} generation failed: ${error?.response?.data?.detail || error?.response?.data?.error || "see the console for details."}`);
        }
      }
      return;
    }

    // Video branch: chain_vid item (a video-length chain continuation
    // segment, from either panel's "Start chain" -- see videoChain.ts).
    // Structurally identical to OutpaintPanel's own outpaint_vid dispatch
    // (same endpoint, same request shape); the only addition is
    // advanceVideoChain, which feeds this chain's NEXT step or stops it.
    if (nextItem.type === "chain_vid") {
      isGeneratingRef.current = true;
      setIsGenerating(true);
      setProgress(0);
      setProgressMessage("");
      setTotalSteps((nextItem.params as OutpaintVideoParams).num_inference_steps || 8);
      setPreviewImage(null);
      setGeneratedImage(null);
      setGeneratedVideo(null);
      setGeneratedVideoInfo(null);
      setGeneratedVideoSeed(null);
      setGeneratedVideoWarnings([]);
      setGeneratedAudio(null);
      setGeneratedAudioInfo(null);
      try {
        const clip = nextItem.inputVideo;
        if (!clip) {
          throw new Error("No input video available for this chain segment (the previous segment has not finished yet)");
        }
        const result = await generateOutpaintVideo(
          nextItem.params as OutpaintVideoParams, clip, undefined, nextItem.referenceImages);
        const videoUrl = `/outputs/${getResultFilename(result)}`;
        const videoPlaybackFilename = getResultPlaybackFilename(result);
        const videoPlaybackUrl = videoPlaybackFilename ? `/outputs/${videoPlaybackFilename}` : videoUrl;
        const videoInfo = { num_frames: result.image?.num_frames, fps: result.image?.fps, duration: result.image?.duration };
        const videoSeed = getResultSeed(result);
        setGeneratedVideo(videoUrl);
        setGeneratedVideoInfo(videoInfo);
        setGeneratedVideoSeed(videoSeed);
        setGeneratedVideoParams(nextItem.params as unknown as GenerationParams);
        setGeneratedVideoWarnings(
          (result.warnings || []).map((w: any) => (typeof w === "string" ? w : w?.message)).filter(Boolean));
        publishCompletedResult({
          panel: "txt2img", kind: "video", url: videoUrl,
          playbackUrl: videoPlaybackUrl !== videoUrl ? videoPlaybackUrl : undefined,
          info: videoInfo, seed: videoSeed, params: nextItem.params,
        });
        if (onImageGenerated) {
          onImageGenerated(videoUrl, {
            kind: "video",
            playbackUrl: videoPlaybackUrl !== videoUrl ? videoPlaybackUrl : undefined,
          });
        }

        const chainOutcome = await advanceVideoChain({
          caps: archCapabilities,
          arch: loadedArch,
          queue,
          completedItem: nextItem,
          resultFrames: result.image?.num_frames,
          resultVideoUrl: videoUrl,
          updateQueueItemByLoop,
          cancelLoopGroup,
        });
        setVideoChainStoppedMessage(chainOutcome.message ?? null);

        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        completeCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
      } catch (error: any) {
        console.error("[Txt2Img] chain_vid generation failed:", error);
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        const isCancelled = String(error?.message || error?.response?.data?.detail || "").toLowerCase().includes("cancel");
        failCurrentItem();
        // Drop the remaining pending "chain_vid" steps of this loop group up
        // front. Without this, each of them dispatches next with no
        // `inputVideo` (only a successful predecessor fills that in) and
        // throws its own generic error, cascading into one alert per
        // remaining segment. Already completed segments stay in the gallery.
        if (nextItem.loopGroupId) {
          cancelLoopGroup(nextItem.loopGroupId);
        }
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
        const completedSegments = (nextItem.loopStepIndex ?? -1) + 1;
        if (!isCancelled) {
          const detail = isGenerationStalledError(error)
            ? error.message
            : (error?.response?.data?.detail || error?.message || "see the console for details.");
          alert(completedSegments > 0
            ? `Video chain stopped: segment failed (${detail}). ${completedSegments} segment(s) completed before this are saved to the gallery.`
            : `Video chain stopped: segment failed (${detail}).`);
        } else {
          alert(completedSegments > 0
            ? `Video chain cancelled. ${completedSegments} segment(s) completed before the cancel are saved to the gallery.`
            : "Video chain cancelled before any segment completed.");
        }
      }
      return;
    }

    // Save current image before starting new generation
    const previousImage = generatedImage;

    isGeneratingRef.current = true;
    setIsGenerating(true);
    setProgress(0);
    setProgressMessage("");
    setTotalSteps(nextItem.params.steps || 20);
    setPreviewImage(null);
    setGeneratedImage(null);
    // An image run supersedes any video/audio result still on screen. The
    // stored previews are left alone until the image actually succeeds (see the
    // save effects), so a failed run does not throw away the last good result.
    setGeneratedVideo(null);
    setGeneratedVideoInfo(null);
    setGeneratedAudio(null);
    setGeneratedAudioInfo(null);
    setCfgMetrics([]); // Clear previous metrics

    try {
      let result;
      let imageUrl;

      // Generate based on type
      if (nextItem.type === "txt2img") {
        // Add developer_mode flag and reset advanced CFG params if disabled
        let paramsWithDevMode = { ...nextItem.params, developer_mode: developerMode };
        if (!showAdvancedCFG) {
          paramsWithDevMode = {
            ...paramsWithDevMode,
            cfg_schedule_type: "constant",
            cfg_rescale_snr_alpha: 0.0,
            dynamic_threshold_percentile: 0.0,
          };
        }
        // Add FLUX.2 Image Edit reference images
        if (refImages.length > 0) {
          paramsWithDevMode = {
            ...paramsWithDevMode,
            ref_images: refImages,
          };
        }
        // Training-preview branch: route to in-training model when the
        // toggle is on and a LoRA/Full-FT run is active.  Returns a blob;
        // we wrap it in an object-URL and reuse the same display path.
        // Skips gallery save (preview only).
        // Per-item flag (set at enqueue) so queued items keep the model choice
        // regardless of the live checkbox state.
        if ((nextItem?.useTrainingModel ?? useTrainingModel) && (nextItem?.trainingRunId ?? activeTraining?.run_id)) {
          const preview = await generateTxt2ImgTrainingPreview({
            ...(paramsWithDevMode as GenerationParams),
            run_id: nextItem?.trainingRunId ?? activeTraining!.run_id,
            save_to_gallery: savePreviewToGallery,
          });
          // Prefer the stable /outputs/<filename> URL when the backend
          // persisted it; fall back to a transient blob URL otherwise.
          if (preview.filename) {
            imageUrl = `/outputs/${preview.filename}`;
          } else {
            if (previewBlobUrlRef.current) URL.revokeObjectURL(previewBlobUrlRef.current);
            const objectUrl = URL.createObjectURL(preview.blob);
            previewBlobUrlRef.current = objectUrl;
            imageUrl = objectUrl;
          }
          // Synthesise a minimal result shape so downstream code that
          // reads result.* doesn't crash.
          result = {
            image: {
              filename: preview.filename
                ?? `preview_${preview.requestId ?? "training"}.png`,
              filepath: imageUrl,
              prompt: paramsWithDevMode.prompt,
              negative_prompt: paramsWithDevMode.negative_prompt,
              metadata: {},
              size_bytes: preview.blob.size,
            },
            actual_seed: preview.seed ? Number(preview.seed) : -1,
            actual_ancestral_seed: -1,
          } as any;
        } else {
          result = await generateTxt2Img(paramsWithDevMode as GenerationParams);
          // loop_decode="none" (decodeMode "final-only" main step, when loop
          // steps follow) returns { latent_id, actual_seed } with NO image.
          imageUrl = isLatentOnlyResult(result) ? undefined : `/outputs/${getResultFilename(result)}`;
        }
      } else if (nextItem.type === "img2img") {
        console.log(`[Txt2Img] Starting img2img generation with prompt:`, nextItem.params.prompt?.substring(0, 100));

        // Add developer_mode flag and reset advanced CFG params if disabled
        let paramsWithDevMode = { ...nextItem.params, developer_mode: developerMode };
        if (!showAdvancedCFG) {
          paramsWithDevMode = {
            ...paramsWithDevMode,
            cfg_schedule_type: "constant",
            cfg_rescale_snr_alpha: 0.0,
            dynamic_threshold_percentile: 0.0,
          };
        }

        let file: File | undefined;
        if (nextItem.inputLatentId) {
          // Latent passthrough chaining (decodeMode "final-only"): the previous
          // step returned a cached latent_id instead of an image; no PNG to fetch.
          console.log(`[Txt2Img] Loop step chaining via cached latent: ${nextItem.inputLatentId}`);
        } else {
          // For loop steps after the first, use the previous output as input
          const inputImageToUse = nextItem.inputImage || previousImage;
          if (!inputImageToUse) {
            throw new Error("No input image available for img2img loop step");
          }
          // Fetch the input image and convert to File
          const response = await fetch(inputImageToUse);
          const blob = await response.blob();
          file = new File([blob], "input.png", { type: "image/png" });
        }

        result = await generateImg2Img(paramsWithDevMode, file, nextItem.inputLatentId);
        // loop_decode="none" (this step's resize_mode is "latent" under
        // decodeMode "final-only") returns { latent_id, actual_seed } with NO image.
        imageUrl = isLatentOnlyResult(result) ? undefined : `/outputs/${getResultFilename(result)}`;
      } else {
        throw new Error(`Unsupported generation type: ${nextItem.type}`);
      }

      // Latent-only steps (loop_decode="none") produce no displayable image —
      // leave the preview/gallery display untouched (already cleared to null
      // above) rather than pointing it at an undefined imageUrl.
      const resultSeed = getResultSeed(result);
      const resultAncestralSeed = getResultAncestralSeed(result);
      if (imageUrl) {
        setGeneratedImage(imageUrl);
      }
      setGeneratedImageSeed(resultSeed);
      setGeneratedImageAncestralSeed(resultAncestralSeed);
      // Save the params used for this generation (with actual result values)
      const completedParams: GenerationParams = {
        ...nextItem.params,
        seed: resultSeed,
        ancestral_seed: resultAncestralSeed ?? -1,
        width: result.image?.width ?? nextItem.params.width,
        height: result.image?.height ?? nextItem.params.height,
      };
      setGeneratedImageParams(completedParams);
      if (imageUrl) {
        publishCompletedResult({
          panel: "txt2img",
          kind: "image",
          url: imageUrl,
          seed: resultSeed,
          ancestralSeed: resultAncestralSeed,
          params: completedParams,
        });
      }
      setPreviewImage(null);

      // Notify parent component (skip for latent-only steps — nothing to show)
      if (onImageGenerated && imageUrl) {
        onImageGenerated(imageUrl, { kind: "image" });
      }

      // If this item has a loop group, update the next loop step's input image, prompt, and ControlNets
      // Use nextItem (not currentItem from context) to avoid timing issues
      console.log(`[Txt2Img] Checking loop group:`, {
        hasNextItem: !!nextItem,
        nextItemType: nextItem?.type,
        loopGroupId: nextItem?.loopGroupId,
        loopStepIndex: nextItem?.loopStepIndex,
      });

      if (nextItem?.loopGroupId !== undefined) {
        const nextLoopStepIndex = (nextItem.loopStepIndex ?? -1) + 1;

        console.log(`[Txt2Img] Processing loop step completion:`, {
          loopGroupId: nextItem.loopGroupId,
          currentStepIndex: nextItem.loopStepIndex,
          nextLoopStepIndex,
        });

        if (isLatentOnlyResult(result)) {
          // Latent passthrough chaining (decodeMode "final-only", resize_mode
          // "latent"): no decoded image exists for this step — chain the next
          // loop step via the cached latent_id instead of an image URL. Skip
          // TIPO-prompt/ControlNet-recompute below (both need a decoded image;
          // ControlNet-conditioned steps always force resize_mode="image" at
          // enqueue time, so they never land here).
          console.log(`[Txt2Img] Updating loop step ${nextLoopStepIndex} with cached latent:`, result.latent_id);
          updateQueueItemByLoop(nextItem.loopGroupId, nextLoopStepIndex, {
            inputLatentId: result.latent_id,
            inputImage: undefined,
          });

          // Scale-mode compounding: there's no decoded image to measure, but
          // this step's OWN target width/height (nextItem.params) is already
          // known — the next step's scale must compound off THAT, not off the
          // constant mainParams size baked in at enqueue time (addLoopStepsToQueueImmediate
          // computes every step's initial width/height from mainParams, so
          // without this recompute a chain of scale steps would never compound).
          const enabledStepsForScale = loopGenerationConfig.steps.filter(step => step.enabled);
          const nextStepConfig = enabledStepsForScale[nextLoopStepIndex];
          const currentWidth = nextItem.params.width;
          const currentHeight = nextItem.params.height;
          if (nextStepConfig && nextStepConfig.sizeMode === "scale" && currentWidth && currentHeight) {
            const scale = nextStepConfig.scale || 1.0;
            const scaledWidth = Math.round(currentWidth * scale);
            const scaledHeight = Math.round(currentHeight * scale);
            console.log(`[Txt2Img] Scale mode (latent passthrough): ${currentWidth}x${currentHeight} * ${scale} = ${scaledWidth}x${scaledHeight}`);
            updateQueueItemByLoop(nextItem.loopGroupId!, nextLoopStepIndex, (item) => ({
              params: {
                ...item.params,
                width: scaledWidth,
                height: scaledHeight,
              } as any,
            }));
          }
        } else {
        // Update input image first
        console.log(`[Txt2Img] Updating loop step ${nextLoopStepIndex} with input image:`, imageUrl);
        updateQueueItemByLoop(nextItem.loopGroupId, nextLoopStepIndex, { inputImage: imageUrl, inputLatentId: undefined });

        // If TIPO was used for base generation, update loop steps with TIPO-generated prompt
        console.log(`[Txt2Img] TIPO inheritance check:`, {
          loopStepIndex: nextItem.loopStepIndex,
          use_tipo: nextItem.params.use_tipo,
          hasResultPrompt: !!result.image?.prompt,
          resultPrompt: result.image?.prompt?.substring(0, 100)
        });

        if (nextItem.loopStepIndex === -1 && nextItem.params.use_tipo && result.image?.prompt) {
          console.log(`[Txt2Img] Base generation used TIPO, updating all loop steps with TIPO prompt`);
          console.log(`[Txt2Img] Original prompt: ${nextItem.params.prompt?.substring(0, 100)}...`);
          console.log(`[Txt2Img] TIPO prompt: ${result.image.prompt?.substring(0, 100)}...`);

          // Update all loop steps (not just the next one) with TIPO-generated prompt
          const enabledSteps = loopGenerationConfig.steps.filter(step => step.enabled);
          for (let i = 0; i < enabledSteps.length; i++) {
            updateQueueItemByLoop(nextItem.loopGroupId, i, (item) => ({
              params: {
                ...item.params,
                prompt: result.image.prompt,
              } as any,
            }));
          }
        }

        // Find step config to check if ControlNet processing is needed
        const enabledSteps = loopGenerationConfig.steps.filter(step => step.enabled);
        const stepConfig = enabledSteps[nextLoopStepIndex];

        console.log(`[Txt2Img] Step config:`, {
          hasStepConfig: !!stepConfig,
          useMainControlNets: stepConfig?.useMainControlNets,
          controlnetsCount: stepConfig?.controlnets?.length,
          sizeMode: stepConfig?.sizeMode,
          scale: stepConfig?.scale,
        });

        // Fetch the generated image for ControlNet or size calculation
        let imageBlob: Blob | null = null;
        let imageWidth: number | null = null;
        let imageHeight: number | null = null;

        const needsImageData = stepConfig && (
          (!stepConfig.useMainControlNets && stepConfig.controlnets && stepConfig.controlnets.length > 0) ||
          stepConfig.sizeMode === "scale"
        );

        if (needsImageData) {
          const response = await fetch(imageUrl);
          imageBlob = await response.blob();

          // Load image to get dimensions for scale mode
          if (stepConfig.sizeMode === "scale") {
            const img = new Image();
            const imageLoadPromise = new Promise<void>((resolve) => {
              img.onload = () => {
                imageWidth = img.width;
                imageHeight = img.height;
                resolve();
              };
            });
            img.src = URL.createObjectURL(imageBlob);
            await imageLoadPromise;
            URL.revokeObjectURL(img.src);

            // Update size based on scale
            if (imageWidth && imageHeight && stepConfig.scale) {
              const scaledWidth = Math.round(imageWidth * stepConfig.scale);
              const scaledHeight = Math.round(imageHeight * stepConfig.scale);
              console.log(`[Txt2Img] Scale mode: ${imageWidth}x${imageHeight} * ${stepConfig.scale} = ${scaledWidth}x${scaledHeight}`);

              updateQueueItemByLoop(nextItem.loopGroupId!, nextLoopStepIndex, (item) => ({
                params: {
                  ...item.params,
                  width: scaledWidth,
                  height: scaledHeight,
                } as any,
              }));
            }
          }
        }

        // Update ControlNet images if needed
        if (stepConfig && !stepConfig.useMainControlNets && stepConfig.controlnets && stepConfig.controlnets.length > 0 && imageBlob) {
          console.log(`[Txt2Img] Processing ${stepConfig.controlnets.length} ControlNet(s) for loop step ${nextLoopStepIndex}`);

          // Convert to base64
          const reader = new FileReader();

          const imageBase64 = await new Promise<string>((resolve) => {
            reader.onloadend = () => {
              const base64 = reader.result as string;
              // Remove data URL prefix to get just the base64 string
              const base64String = base64.split(',')[1];
              resolve(base64String);
            };
            reader.readAsDataURL(imageBlob);
          });

          console.log(`[Txt2Img] Converted image to base64, length: ${imageBase64.length}`);

          // Update ControlNets with useLoopImage enabled using callback to preserve existing params
          updateQueueItemByLoop(nextItem.loopGroupId!, nextLoopStepIndex, (item) => {
            const updatedControlnets = stepConfig.controlnets.map((cnConfig, idx) => {
              console.log(`[Txt2Img] ControlNet ${idx}: useLoopImage=${cnConfig.useLoopImage}`);
              if (cnConfig.useLoopImage) {
                console.log(`[Txt2Img] Setting image_base64 for ControlNet ${idx}`);
                return { ...cnConfig, image_base64: imageBase64 };
              }
              return cnConfig;
            });

            return {
              params: {
                ...item.params,
                controlnets: updatedControlnets,
              } as any,
            };
          });

          console.log(`[Txt2Img] ControlNet images updated for loop step ${nextLoopStepIndex}`);
        }
        }
      }

      // Reset state first, then complete item
      console.log("[Txt2Img] Generation complete, resetting state and completing item");
      isGeneratingRef.current = false;
      setIsGenerating(false);
      setProgress(0);
      setProgressMessage("");
      completeCurrentItem();

      // Wait briefly for state to propagate, then trigger next
      setTimeout(() => {
        console.log("[Txt2Img] Triggering next queue item");
        if (processQueueRef.current) {
          processQueueRef.current();
        }
      }, 100);
    } catch (error: any) {
      console.error("Generation failed:", error);
      console.log("Error details:", {
        message: error?.message,
        responseData: error?.response?.data,
        responseDetail: error?.response?.data?.detail,
      });

      // Check if cancelled
      const errorStr = JSON.stringify(error);
      const errorMessage = error?.message || "";
      const errorDetail = error?.response?.data?.detail || "";
      const isCancelled =
        errorMessage.toLowerCase().includes("cancel") ||
        errorDetail.toLowerCase().includes("cancel") ||
        errorStr.toLowerCase().includes("cancel");

      // alert() blocks the JS thread; decide what to show but do not call it
      // yet -- reset state and requeue first, or the queue effect sees a
      // stale isGenerating until the dialog closes.
      let alertMessage: string | null = null;
      if (isCancelled) {
        const shouldRestore = localStorage.getItem('restore_image_on_cancel') === 'true';
        if (shouldRestore && previousImage) {
          setGeneratedImage(previousImage);
          setPreviewImage(null);
        }
      } else if (isGenerationStalledError(error)) {
        alertMessage = error.message;
      } else {
        alertMessage = "Generation failed. Please check console for details.";
      }

      // Reset state first, then fail item
      console.log("[Txt2Img] Generation failed, resetting state and failing item");
      isGeneratingRef.current = false;
      setIsGenerating(false);
      setProgress(0);
      setProgressMessage("");
      failCurrentItem();

      // Wait briefly for state to propagate, then trigger next
      setTimeout(() => {
        console.log("[Txt2Img] Triggering next queue item after failure");
        if (processQueueRef.current) {
          processQueueRef.current();
        }
      }, 100);

      if (alertMessage) {
        alert(alertMessage);
      }
    }
  }, [isGenerating, generatedImage, onImageGenerated, startNextInQueue, completeCurrentItem, failCurrentItem, updateQueueItem, updateQueueItemByLoop, cancelLoopGroup, queue, publishCompletedResult, archCapabilities, loadedArch]);

  processQueueRef.current = processQueue;

  // Auto-start queue processing when queue has pending items and not currently generating
  useEffect(() => {
    const hasPendingItems = queue.some(item =>
      item.status === "pending" && ["txt2img", "img2img", "txt2vid", "ref2vid", "txt2aud", "chain_vid"].includes(item.type));
    const isCurrentItemNull = currentItem === null;

    console.log("[Txt2Img] Queue effect:", {
      hasPendingItems,
      isCurrentItemNull,
      isGenerating,
      queueLength: queue.length,
      queue: queue,
      currentItem: currentItem,
      generateForever
    });

    // If generate forever is enabled and queue is empty, add new item
    if (generateForever && !hasPendingItems && isCurrentItemNull && !isGenerating && params.prompt) {
      console.log("[Txt2Img] Generate forever: Adding new item to queue");
      handleAddToQueue();
      return;
    }

    // A queue survives a page reload and a backend restart, so on mount there
    // can be pending items with no model loaded yet. Dispatching then earns an
    // immediate 400 and the item is marked failed for a reason that has nothing
    // to do with the item. Hold instead: `modelLoaded` is a dependency, so the
    // queue starts by itself once a model is up.
    if (hasPendingItems && isCurrentItemNull && !isGenerating && !modelLoaded) {
      console.log("[Txt2Img] Queue held: no model loaded yet");
      return;
    }

    if (hasPendingItems && isCurrentItemNull && !isGenerating) {
      console.log("[Txt2Img] Auto-starting queue processing");
      processQueue();
    }
  }, [queue, currentItem, isGenerating, processQueue, generateForever, params, modelLoaded]);

  // Handle Ctrl+Enter keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if Prompt Editor or Image Editor is open
      if (document.body.dataset.imageEditorOpen || document.querySelector('[data-prompt-assist-open="true"]')) return;

      if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        handleAddToQueue();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [params]);

  // Render functions for each Txt2Img Options tab (see TXT2IMG_OPTIONS_TABS /
  // TXT2IMG_OPTIONS_TAB_KEYS / isTxt2ImgOptionsTabActive above). Every control
  // below is unchanged from its original in-Card location -- same param
  // binding / handler / conditional reveal -- ported from Outpaint/Inpaint's
  // *OptionsTabRender pattern.
  const txt2imgOptionsTabRender: Record<Txt2ImgOptionsTabId, () => JSX.Element> = {
    cfg: () => (
      <div className="space-y-4">
        {/* Advanced CFG Settings */}
        {showAdvancedCFG && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Dynamic CFG Scheduling */}
            <div className="space-y-3">
              <label className="block text-sm font-medium text-gray-300">
                Dynamic CFG Schedule
              </label>
              <select
                value={params.cfg_schedule_type || "constant"}
                onChange={(e) => setParams({ ...params, cfg_schedule_type: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="constant">Constant (no scheduling)</option>
                <option value="linear">Linear (sigma-based)</option>
                <option value="quadratic">Quadratic (sigma-based)</option>
                <option value="cosine">Cosine (sigma-based)</option>
                <option value="snr_based">SNR-Based Adaptive</option>
              </select>

              {params.cfg_schedule_type && params.cfg_schedule_type !== "constant" && params.cfg_schedule_type !== "snr_based" && (
                <>
                  <Slider
                    label="CFG Min (end of generation)"
                    min={1}
                    max={15}
                    step={0.5}
                    value={params.cfg_schedule_min || 1.0}
                    onChange={(e) => setParams({ ...params, cfg_schedule_min: parseFloat(e.target.value) })}
                  />
                  <Slider
                    label="CFG Max (start of generation)"
                    min={1}
                    max={30}
                    step={0.5}
                    value={params.cfg_schedule_max || params.cfg_scale}
                    onChange={(e) => setParams({ ...params, cfg_schedule_max: parseFloat(e.target.value) })}
                  />
                  {params.cfg_schedule_type === "quadratic" && (
                    <Slider
                      label="Power (curve steepness)"
                      min={0.5}
                      max={4.0}
                      step={0.1}
                      value={params.cfg_schedule_power || 2.0}
                      onChange={(e) => setParams({ ...params, cfg_schedule_power: parseFloat(e.target.value) })}
                    />
                  )}
                </>
              )}
              {params.cfg_schedule_type === "snr_based" && (
                <Slider
                  label="SNR Alpha (0=off, 0.1-0.5 typical)"
                  min={0}
                  max={1.0}
                  step={0.05}
                  value={params.cfg_rescale_snr_alpha || 0.0}
                  onChange={(e) => setParams({ ...params, cfg_rescale_snr_alpha: parseFloat(e.target.value) })}
                />
              )}
            </div>

            {/* Dynamic Thresholding */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={params.dynamic_threshold_percentile !== undefined && params.dynamic_threshold_percentile > 0}
                  onChange={(e) => setParams({
                    ...params,
                    dynamic_threshold_percentile: e.target.checked ? 99.5 : 0.0
                  })}
                  className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
                />
                <label className="text-sm font-medium text-gray-300">
                  Dynamic Thresholding
                </label>
              </div>
              {params.dynamic_threshold_percentile !== undefined && params.dynamic_threshold_percentile > 0 && (
                <>
                  <Slider
                    label="Threshold Percentile"
                    min={90}
                    max={100}
                    step={0.5}
                    value={params.dynamic_threshold_percentile || 99.5}
                    onChange={(e) => setParams({ ...params, dynamic_threshold_percentile: parseFloat(e.target.value) })}
                  />
                  <Slider
                    label="Mimic Scale (static clamp)"
                    min={1}
                    max={30}
                    step={0.5}
                    value={params.dynamic_threshold_mimic_scale || 7.0}
                    onChange={(e) => setParams({ ...params, dynamic_threshold_mimic_scale: parseFloat(e.target.value) })}
                  />
                </>
              )}
            </div>
          </div>
        )}

        {/* NAG (Normalized Attention Guidance) */}
        {showAdvancedCFG && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={params.nag_enable || false}
                onChange={(e) => setParams({
                  ...params,
                  nag_enable: e.target.checked
                })}
                className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
              />
              <label className="text-sm font-medium text-gray-300">
                NAG (Normalized Attention Guidance)
              </label>
            </div>
            {params.nag_enable && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Slider
                  label="NAG Scale"
                  min={1}
                  max={10}
                  step={0.5}
                  value={params.nag_scale || 5.0}
                  onChange={(e) => setParams({ ...params, nag_scale: parseFloat(e.target.value) })}
                />
                <Slider
                  label="NAG Tau (normalization threshold)"
                  min={1.0}
                  max={5.0}
                  step={0.1}
                  value={params.nag_tau || 3.5}
                  onChange={(e) => setParams({ ...params, nag_tau: parseFloat(e.target.value) })}
                />
                <Slider
                  label="NAG Alpha (blending factor)"
                  min={0.05}
                  max={1.0}
                  step={0.05}
                  value={params.nag_alpha || 0.25}
                  onChange={(e) => setParams({ ...params, nag_alpha: parseFloat(e.target.value) })}
                />
                <Slider
                  label="NAG Sigma End"
                  min={0.0}
                  max={5.0}
                  step={0.1}
                  value={params.nag_sigma_end ?? 3.0}
                  onChange={(e) => setParams({ ...params, nag_sigma_end: parseFloat(e.target.value) })}
                />
              </div>
            )}
          </div>
        )}

        {/* SDXL micro-conditioning: original_size override.
            Sets original_size in SDXL time_ids separately from the output size.
            Absolute = explicit W/H (both > 0); Scale = output size × scale. */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-gray-300">Original Size (SDXL micro-conditioning)</div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-400">Source</span>
            <div className="flex gap-1">
              <Button
                onClick={() => setParams({ ...params, original_size_w: params.width, original_size_h: params.height })}
                variant={(params.original_size_w ?? 0) > 0 && (params.original_size_h ?? 0) > 0 ? "primary" : "secondary"}
                size="sm"
                className="text-xs px-2 py-0.5"
              >
                Absolute
              </Button>
              <Button
                onClick={() => setParams({ ...params, original_size_w: 0, original_size_h: 0 })}
                variant={!((params.original_size_w ?? 0) > 0 && (params.original_size_h ?? 0) > 0) ? "primary" : "secondary"}
                size="sm"
                className="text-xs px-2 py-0.5"
              >
                Scale
              </Button>
            </div>
          </div>
          {(params.original_size_w ?? 0) > 0 && (params.original_size_h ?? 0) > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Slider
                label="Original Width"
                min={64}
                max={4096}
                step={resolutionStep}
                value={params.original_size_w || params.width}
                onChange={(e) => setParams({ ...params, original_size_w: parseInt(e.target.value) })}
              />
              <Slider
                label="Original Height"
                min={64}
                max={4096}
                step={resolutionStep}
                value={params.original_size_h || params.height}
                onChange={(e) => setParams({ ...params, original_size_h: parseInt(e.target.value) })}
              />
            </div>
          ) : (
            <Slider
              label={`Scale (${Math.round(params.width * (params.original_size_scale ?? 1.0))}x${Math.round(params.height * (params.original_size_scale ?? 1.0))})`}
              min={0.25}
              max={4.0}
              step={0.05}
              value={params.original_size_scale ?? 1.0}
              onChange={(e) => setParams({ ...params, original_size_scale: parseFloat(e.target.value) })}
            />
          )}
          <p className="text-xs text-gray-500">
            SDXL only. Sets original_size in time_ids separately from the output size. Absolute uses explicit W/H; Scale uses output size × scale.
          </p>
        </div>
      </div>
    ),

    acceleration: () => (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
        {supportsSpectrum && (
        <div className="space-y-2">
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="spectrum_enable"
            checked={params.spectrum_enable || false}
            onChange={(e) => setParams({ ...params, spectrum_enable: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="spectrum_enable" className="text-sm text-gray-300">
            Spectrum (Spectral Feature Forecasting)
          </label>
          <span className="text-xs text-gray-500">(skips U-Net steps via Chebyshev forecast; best at high step counts)</span>
        </div>
        {params.spectrum_enable && (
          <div className="ml-6 mt-1 flex items-center gap-2">
            <label className="text-xs text-gray-400">Mode</label>
            <select
              value={params.spectrum_feature_mode ?? "output"}
              onChange={(e) => setParams({ ...params, spectrum_feature_mode: e.target.value })}
              className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
            >
              <option value="output">output (black-box, max speed)</option>
              <option value="block">block (deep-feature, higher quality)</option>
            </select>
            {params.spectrum_feature_mode === "block" && (
              <label className="text-xs text-gray-400 flex items-center gap-1" title="down_blocks[branch:] + mid are forecast; lower skips more deep blocks.">
                Branch
                <input type="number" min={1} max={3} step={1}
                  value={params.spectrum_cache_branch ?? 1}
                  onChange={(e) => setParams({ ...params, spectrum_cache_branch: parseInt(e.target.value) || 1 })}
                  className="w-16 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
              </label>
            )}
          </div>
        )}
        {params.spectrum_enable && (
          <div className="ml-6 mt-1 grid grid-cols-2 gap-2">
            <label className="text-xs text-gray-400 flex items-center gap-1">
              Mix w
              <input type="number" min={0} max={1} step={0.05}
                value={params.spectrum_w ?? 1.0}
                onChange={(e) => setParams({ ...params, spectrum_w: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">
              Mix w decay
              <input type="number" min={0} step={0.25}
                value={params.spectrum_w_decay ?? 0.0}
                onChange={(e) => setParams({ ...params, spectrum_w_decay: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1" title="Limits how far a forecast may advance past the last real pass, relative to the observed trajectory speed. 0 disables the cap.">
              Delta cap
              <input type="number" step={0.25}
                value={params.spectrum_delta_cap ?? 0.0}
                onChange={(e) => setParams({ ...params, spectrum_delta_cap: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">
              Basis m
              <input type="number" min={1} max={8} step={1}
                value={params.spectrum_m ?? 4}
                onChange={(e) => setParams({ ...params, spectrum_m: parseInt(e.target.value) || 4 })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">
              Ridge λ
              <input type="number" min={0} step={0.01}
                value={params.spectrum_lam ?? 0.1}
                onChange={(e) => setParams({ ...params, spectrum_lam: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">
              Warmup
              <input type="number" min={1} step={1}
                value={params.spectrum_warmup_steps ?? 3}
                onChange={(e) => setParams({ ...params, spectrum_warmup_steps: parseInt(e.target.value) || 3 })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">
              Window
              <input type="number" min={1} step={1}
                value={params.spectrum_window_size ?? 4}
                onChange={(e) => setParams({ ...params, spectrum_window_size: parseInt(e.target.value) || 4 })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">
              Flex
              <input type="number" min={0} max={1} step={0.05}
                value={params.spectrum_flex_window ?? 0.75}
                onChange={(e) => setParams({ ...params, spectrum_flex_window: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1" title="Fraction of final steps forced to real forwards (preserves detail). Higher = sharper/slower.">
              Tail
              <input type="number" min={0} max={0.5} step={0.02}
                value={params.spectrum_tail ?? 0.12}
                onChange={(e) => setParams({ ...params, spectrum_tail: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
          </div>
        )}
        </div>
        )}

        {supportsFbcache && (
        <div className="space-y-2">
        <div className="flex items-center gap-2 mt-2">
          <input
            type="checkbox"
            id="fbcache_enable"
            checked={params.fbcache_enable || false}
            onChange={(e) => setParams({ ...params, fbcache_enable: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="fbcache_enable" className="text-sm text-gray-300">
            First Block Cache (dynamic caching)
          </label>
          <span className="text-xs text-gray-500">(mutually exclusive with Spectrum)</span>
        </div>
        {params.fbcache_enable && (
          <div className="ml-6 mt-1 grid grid-cols-2 gap-2">
            <label className="text-xs text-gray-400 flex items-center gap-1">
              Residual threshold (higher = more skips)
              <NumberInput min={0} step={0.01} parse="float"
                value={params.fbcache_threshold ?? 0.12}
                defaultValue={0.12}
                placeholder="0.12"
                onCommit={(v) => setParams({ ...params, fbcache_threshold: v })}
                className="w-20" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">
              Warmup steps
              <NumberInput min={0} step={1}
                value={params.fbcache_warmup_steps ?? 1}
                defaultValue={1}
                placeholder="1"
                onCommit={(v) => setParams({ ...params, fbcache_warmup_steps: v })}
                className="w-20" />
            </label>
          </div>
        )}
        </div>
        )}
      </div>
    ),

    post_process: () => (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
        <div title="Applies the same chroma-smoothing as the post-edit Color Flatten at generation time, baked into the saved image; 0 = off.">
          <Slider
            label="Color Flatten（色ムラ除去）"
            min={0}
            max={100}
            step={1}
            value={params.color_flatten_strength ?? 0}
            onChange={(e) => setParams({ ...params, color_flatten_strength: parseInt(e.target.value) })}
          />
        </div>

        <div className="lg:col-span-2">
        <div className="flex items-center gap-2 mt-2">
          <input
            type="checkbox"
            id="flatten_in_loop"
            checked={params.flatten_in_loop || false}
            onChange={(e) => setParams({ ...params, flatten_in_loop: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="flatten_in_loop" className="text-sm text-gray-300" title="During the final denoise steps, detects the flat background region and replaces it with its solid dominant color (both luma and chroma become uniform - stronger than Color Flatten); no-op when no confident flat region is found; SD/SDXL only for now.">
            In-loop background flatten（背景ベタ塗り化）
          </label>
        </div>
        {params.flatten_in_loop && (
          <div className="ml-6 mt-1 grid grid-cols-2 gap-2">
            <label className="text-xs text-gray-400 flex items-center gap-1" title="Number of final denoise steps to apply the correction on; more = flatter background but more subject-detail change and +decode/encode cost per step.">
              Flatten last N steps
              <NumberInput min={1} max={16} step={1}
                value={params.flatten_in_loop_last_steps ?? 3}
                defaultValue={3}
                placeholder="3"
                onCommit={(v) => setParams({ ...params, flatten_in_loop_last_steps: v })}
                className="w-20" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1" title="Minimum fraction of the image the detected flat region must cover; below it the feature is a no-op (protects textured backgrounds).">
              Min region fraction
              <NumberInput min={0.005} max={0.5} step={0.005} parse="float"
                value={params.flatten_in_loop_min_region ?? 0.02}
                defaultValue={0.02}
                placeholder="0.02"
                onCommit={(v) => setParams({ ...params, flatten_in_loop_min_region: v })}
                className="w-20" />
            </label>
          </div>
        )}
        </div>
      </div>
    ),

    prompt_chunking: () => (
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
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
    ),

    environment: () => (
      <div className="space-y-3">
        <p className="text-xs text-gray-500">
          モデル全体に適用され、Loop Generationにも常に引き継がれます。
        </p>

        {/* Quantization: Z-Image/FLUX.2 uses 2-column layout (Transformer + Text Encoder), SD/SDXL uses 1-column (U-Net) */}
        {(currentModelInfo?.model_info?.type === "zimage" || currentModelInfo?.model_info?.type === "flux2" || currentModelInfo?.model_info?.type === "anima") ? (
          <>
            {/* Z-Image/FLUX.2: 2-column layout */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Select
                label={`Transformer Quantization (${currentModelInfo?.model_info?.type === "flux2" ? "FLUX.2" : "Z-Image"})`}
                value={params.unet_quantization || "none"}
                onChange={(e) => setParams({
                  ...params,
                  unet_quantization: e.target.value === "none" ? null : e.target.value
                })}
                options={unetQuantizationOptions(archCapabilities, currentModelInfo?.model_info?.type as string | undefined)}
              />
              <Select
                label={`Text Encoder Quantization (${currentModelInfo?.model_info?.type === "flux2" ? "Qwen3" : "Gemma2"})`}
                value={params.text_encoder_quantization || "none"}
                onChange={(e) => setParams({
                  ...params,
                  text_encoder_quantization: e.target.value === "none" ? null : e.target.value
                })}
                options={[
                  { value: "none", label: "None" },
                  { value: "fp8_e4m3fn", label: "FP8 E4M3 (Recommended)" },
                  { value: "fp8_e5m2", label: "FP8 E5M2" },
                ]}
              />
            </div>
            {(params.unet_quantization && params.unet_quantization !== "none") || (params.text_encoder_quantization && params.text_encoder_quantization !== "none") ? (
              <div className="bg-blue-900/20 border border-blue-600/30 rounded-lg p-3 space-y-1">
                <p className="text-xs text-blue-200">
                  💡 Quantization can reduce VRAM significantly. Text encoder ({currentModelInfo?.model_info?.type === "flux2" ? "Qwen3" : "Gemma2"}) is particularly large.
                </p>
                {params.unet_quantization && params.unet_quantization !== "none" && params.unet_quantization !== "int8" && (
                  <p className="text-xs text-blue-200">
                    Transformer FP8 weights are dequantized back to full precision per operation during inference, so generation is slower than without quantization.
                  </p>
                )}
                {params.unet_quantization === "int8" && (
                  <p className="text-xs text-blue-200">
                    INT8 converts the transformer in place the first time you generate after loading the model, and keeps it for the session. Layers where INT8 measures worse than FP8 E4M3 are stored as E4M3 instead. The conversion is one-way: reload the model to return to the checkpoint&apos;s original precision.
                  </p>
                )}
              </div>
            ) : null}
          </>
        ) : (
          <>
            {/* Every other architecture: the transformer/U-Net control, plus the
                text-encoder one only where the backend applies it. */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Select
                label={transformerQuantizationLabel(currentModelInfo?.model_info?.type as string | undefined)}
                value={params.unet_quantization || "none"}
                onChange={(e) => setParams({
                  ...params,
                  unet_quantization: e.target.value === "none" ? null : e.target.value
                })}
                options={unetQuantizationOptions(archCapabilities, currentModelInfo?.model_info?.type as string | undefined)}
              />
              {archSupportsFeature(archCapabilities, currentModelInfo?.model_info?.type as string | undefined, "text_encoder_quantization") && (
                <Select
                  label="Text Encoder Quantization"
                  value={params.text_encoder_quantization || "none"}
                  onChange={(e) => setParams({
                    ...params,
                    text_encoder_quantization: e.target.value === "none" ? null : e.target.value
                  })}
                  options={[
                    { value: "none", label: "None" },
                    { value: "fp8_e4m3fn", label: "FP8 E4M3 (Recommended)" },
                    { value: "fp8_e5m2", label: "FP8 E5M2" },
                  ]}
                />
              )}
            </div>
            {(params.unet_quantization && params.unet_quantization !== "none") || (params.text_encoder_quantization && params.text_encoder_quantization !== "none") ? (
              <div className="bg-blue-900/20 border border-blue-600/30 rounded-lg p-3 space-y-1">
                <p className="text-xs text-blue-200">
                  💡 Quantization reduces the VRAM the model&apos;s weights occupy.
                </p>
                {params.unet_quantization && params.unet_quantization !== "none" && params.unet_quantization !== "int8" && (
                  <p className="text-xs text-blue-200">
                    FP8 weights are dequantized back to full precision per operation during inference, so generation is slower than without quantization.
                  </p>
                )}
                {params.unet_quantization === "int8" && (
                  <p className="text-xs text-blue-200">
                    INT8 converts the transformer in place the first time you generate after loading the model, and keeps it for the session. Layers where INT8 measures worse than FP8 E4M3 are stored as E4M3 instead. The conversion is one-way: reload the model to return to the checkpoint&apos;s original precision.
                  </p>
                )}
              </div>
            ) : null}
          </>
        )}

        <QuantizedGemmSelect
          arch={currentModelInfo?.model_info?.type as string | undefined}
          value={params.quantized_gemm_mode ?? null}
          onChange={(v) => setParams({ ...params, quantized_gemm_mode: v })}
        />

        {/* CPU Text Encoding — applies to all model types */}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={params.cpu_text_encoding ?? false}
            onChange={(e) => setParams({ ...params, cpu_text_encoding: e.target.checked })}
            className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500"
          />
          <span className="text-sm text-gray-300">CPU Text Encoding</span>
          <span className="text-xs text-gray-500">(saves VRAM, slower)</span>
        </label>

        <div className="flex items-center gap-2 mt-2">
          <input
            type="checkbox"
            id="vae_tiling"
            checked={params.vae_tiling || false}
            onChange={(e) => setParams({ ...params, vae_tiling: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="vae_tiling" className="text-sm text-gray-300">
            VAE Tiling
          </label>
          <span className="text-xs text-gray-500">(tiled decode for large images, saves VRAM)</span>
        </div>
        {params.vae_tiling && (
          <>
          <div className="flex items-center gap-2 mt-1 ml-6">
            <label htmlFor="vae_tile_threshold" className="text-xs text-gray-400">Tile threshold (px)</label>
            <NumberInput
              id="vae_tile_threshold"
              min={0}
              step={128}
              value={params.vae_tile_threshold ?? 0}
              defaultValue={0}
              placeholder="0"
              onCommit={(v) => setParams({ ...params, vae_tile_threshold: v })}
              className="w-24"
            />
            <span className="text-xs text-gray-500">0 = auto (per-VAE default; 256px on Anima/Krea2)</span>
          </div>
          <div className="flex items-center gap-2 mt-1 ml-6">
            <label htmlFor="vae_tile_mode" className="text-xs text-gray-400">Tile join</label>
            <select
              id="vae_tile_mode"
              value={params.vae_tile_mode ?? "blend"}
              onChange={(e) => setParams({ ...params, vae_tile_mode: e.target.value })}
              className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
            >
              <option value="blend">Blend (overlapping tiles, cross-faded together)</option>
              <option value="context">Context margin (16 latent cells of real neighbouring context, discarded after decode)</option>
            </select>
            <span className="text-xs text-gray-500">
              blend: tiles overlap and are cross-faded. context: tiles join without a cross-fade;
              lower decode memory peak at the same threshold, more decoder calls at small thresholds
            </span>
          </div>
          <div className="flex items-center gap-2 mt-1 ml-6">
            <input
              type="checkbox"
              id="vae_tile_global_norm"
              checked={params.vae_tile_global_norm || false}
              onChange={(e) => setParams({ ...params, vae_tile_global_norm: e.target.checked })}
              className="w-4 h-4"
            />
            <label htmlFor="vae_tile_global_norm" className="text-xs text-gray-400">Global GroupNorm statistics</label>
            <span className="text-xs text-gray-500">
              decodes twice: the first pass measures the decoder&apos;s GroupNorm statistics across
              all tiles, the second re-decodes using the whole-image values, so no tile is
              normalized against itself. Measured on SDXL (blend, 512px tiles): per-tile offset
              1.32 &rarr; 0.037 /255, decode wall time x2, peak decode memory unchanged. The x2
              applies to every VAE decode in the request, including the in-loop decodes of
              In-Loop Flatten and VAE Drift Correction when those are on. No effect on
              Anima/Krea2 (their decoder contains no GroupNorm)
            </span>
          </div>
        </>
        )}

        {developerMode && (
          <>
            <div className="flex items-center gap-2 mt-2">
              <input
                type="checkbox"
                id="use_torch_compile"
                checked={params.use_torch_compile || false}
                onChange={(e) => setParams({ ...params, use_torch_compile: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="use_torch_compile" className="text-sm text-gray-300">
                ⚠️ torch.compile (Experimental, slow first run)
              </label>
            </div>
            {params.use_torch_compile && (
              <div className="bg-orange-900/20 border border-orange-600/30 rounded-lg p-3 mt-2">
                <p className="text-xs text-orange-200">
                  ⚠️ <strong>Experimental feature:</strong> torch.compile takes several minutes on first run for compilation.
                  Subsequent runs will be 1.3-2x faster. May fail on some GPU/Windows configurations.
                </p>
              </div>
            )}

            {/* Block Swap (Z-Image only) */}
            <div className="flex items-center gap-2 mt-4">
              <input
                type="checkbox"
                id="enable_block_swap"
                checked={params.enable_block_swap || false}
                onChange={(e) => setParams({ ...params, enable_block_swap: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="enable_block_swap" className="text-sm text-gray-300">
                Block Swap (Z-Image Transformer offloading)
              </label>
            </div>
            {params.enable_block_swap && (
              <div className="space-y-3 mt-2 p-3 bg-blue-900/20 border border-blue-600/30 rounded-lg">
                <Slider
                  label="Blocks to Swap"
                  min={1}
                  max={29}
                  step={1}
                  value={params.blocks_to_swap || 20}
                  onChange={(e) => setParams({ ...params, blocks_to_swap: parseInt(e.target.value) })}
                />
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="use_pinned_memory"
                    checked={params.use_pinned_memory || false}
                    onChange={(e) => setParams({ ...params, use_pinned_memory: e.target.checked })}
                    className="rounded"
                  />
                  <label htmlFor="use_pinned_memory" className="text-xs text-gray-300">
                    Use Pinned Memory (faster transfer, more RAM)
                  </label>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="block_swap_h2d_only"
                    checked={params.block_swap_h2d_only || false}
                    onChange={(e) => setParams({ ...params, block_swap_h2d_only: e.target.checked })}
                    className="rounded"
                  />
                  <label htmlFor="block_swap_h2d_only" className="text-xs text-gray-300">
                    H2D-only (no device-to-host eviction of read-only weights)
                  </label>
                </div>
                {params.block_swap_h2d_only && (
                  <Slider
                    label="Ring Size (GPU weight buffer slots)"
                    min={1}
                    max={4}
                    step={1}
                    value={params.block_swap_ring_size || 2}
                    onChange={(e) => setParams({ ...params, block_swap_ring_size: parseInt(e.target.value) })}
                  />
                )}
                <div className="text-xs text-blue-200">
                  <p>
                    <strong>Block Swap:</strong> Offloads Z-Image Transformer blocks between CPU and GPU to reduce VRAM usage.
                  </p>
                  <p className="mt-1">
                    <strong>Blocks to Swap:</strong> Higher = more VRAM reduction, but slower generation.
                  </p>
                  <p className="mt-1">
                    <strong>H2D-only:</strong> Keeps a CPU master copy and only transfers host-to-device (inference / read-only weights). Ring Size 1 = minimum VRAM; 2+ = next block loads during current block compute.
                  </p>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    ),
  };

  // ── What the video Absolute sliders are allowed to reach ─────────────────
  //
  // Same rule as Img2Img/Inpaint/Outpaint's video Size sliders (see
  // videoCanvasAxisBounds): the envelope is on the short/long edges, not on
  // width/height, so each slider's ceiling depends on where the other one
  // sits. txt2vid has no input clip, so there is no Scale mode here -- only
  // the Absolute bounds/warning apply.
  const videoCanvasWidth = params.width ?? 768;
  const videoCanvasHeight = params.height ?? 512;
  const videoWidthBounds = videoCanvasAxisBounds(archCapabilities, loadedArch, videoCanvasHeight);
  const videoHeightBounds = videoCanvasAxisBounds(archCapabilities, loadedArch, videoCanvasWidth);
  const videoCanvasOverEnvelope = videoCanvasExceedsEnvelope(
    archCapabilities, loadedArch, videoCanvasWidth, videoCanvasHeight);
  const hasLeadConditioning = (isVideo && isRef2Va)
    || currentModelInfo?.model_info?.type === "flux2"
    || !!params.vision_encoder_path;
  const promptPanel = isAudio ? (
    <Card title="Prompt">
      <Textarea
        label="Caption"
        placeholder="Describe the music (genre, mood, instruments)..."
        rows={3}
        resizeStorageKey={GENERATION_PROMPT_HEIGHT_KEY}
        value={params.prompt}
        onChange={(e) => setParams({ ...params, prompt: e.target.value })}
      />
      <Textarea
        label="Lyrics"
        placeholder="Enter lyrics (optional)..."
        rows={3}
        resizeStorageKey={GENERATION_LYRICS_HEIGHT_KEY}
        value={params.lyrics ?? ""}
        onChange={(e) => setParams({ ...params, lyrics: e.target.value })}
      />
      <Textarea
        label="Negative Prompt"
        placeholder="Negative prompting is unavailable for this model"
        rows={2}
        resizeStorageKey={GENERATION_NEGATIVE_PROMPT_HEIGHT_KEY}
        value={params.negative_prompt}
        onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
        disabled
        title="Audio generation does not accept negative-prompt conditioning."
      />
      <p className="text-xs text-gray-500">Unavailable for audio generation; the saved value is preserved.</p>
    </Card>
  ) : (
    <Card title="Prompt">
      <div className="relative">
        <TextareaWithTagSuggestions
          label="Positive Prompt"
          placeholder="Enter your prompt here..."
          rows={3}
          resizeStorageKey={GENERATION_PROMPT_HEIGHT_KEY}
          value={params.prompt}
          onChange={(e) => {
            setParams({ ...params, prompt: e.target.value });
            if (e.target) promptTextareaRef.current = e.target as HTMLTextAreaElement;
          }}
          suggestionMode={loadedArch === "minimax_h3" ? "h3" : "tags"}
          enableWeightControl
        />
        <div className="absolute -top-1 right-0 px-2 py-1 text-xs text-gray-400 pointer-events-none">
          {promptTokenCount} tokens
        </div>
      </div>
      {loadedArch === "minimax_h3" && (
        <H3PromptAssist
          prompt={params.prompt}
          onApply={(prompt) => setParams((previous) => ({ ...previous, prompt }))}
          suggestedMode={isRef2Va && countMiniMaxH3References(h3References) > 0 ? "ref2va" : "t2va"}
          durationSeconds={effectiveSegmentFrames(archCapabilities, loadedArch, params.num_frames ?? 121, chainSegmentFrames) / (params.frame_rate ?? 24)}
          references={createH3ReferenceInventory({
            pictures: h3References.images.length + h3Keyframes.length,
            videos: h3References.videos.length,
            audios: h3References.audios.length + h3References.videoAudios.filter(Boolean).length,
          })}
        />
      )}
      {!isVideo && <div className="flex flex-wrap items-center gap-1.5 rounded bg-gray-800 px-2 py-1.5">
        <label className="flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            checked={params.use_tipo || false}
            onChange={(e) => setParams({ ...params, use_tipo: e.target.checked })}
            className="h-4 w-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-2 focus:ring-blue-500"
          />
          <span className="text-sm text-gray-300">✨ Feeling Lucky (TIPO)</span>
        </label>
        <label className="flex cursor-pointer items-center gap-1.5">
          <input
            type="checkbox"
            checked={treatAsNL}
            onChange={(e) => setTreatAsNL(e.target.checked)}
            className="h-4 w-4 rounded border-gray-600 bg-gray-700 text-green-500 focus:ring-2 focus:ring-green-500"
            title="Treat input as natural language instead of tags"
          />
          <span className="text-xs text-gray-400">NL</span>
        </label>
        <button
          onClick={() => setIsTIPODialogOpen(true)}
          className="ml-auto rounded bg-gray-700 px-2 py-1 text-xs hover:bg-gray-600"
          title="Configure TIPO settings"
        >
          ⚙️ Settings
        </button>
      </div>}
      <div className="relative">
        <TextareaWithTagSuggestions
          label="Negative Prompt"
          placeholder={supportsNegativePrompt ? "Enter negative prompt..." : "Negative prompting is unavailable for this model"}
          rows={2}
          resizeStorageKey={GENERATION_NEGATIVE_PROMPT_HEIGHT_KEY}
          value={params.negative_prompt}
          onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
          suggestionMode={loadedArch === "minimax_h3" ? "h3" : "tags"}
          enableWeightControl
          disabled={!supportsNegativePrompt}
          title={!supportsNegativePrompt ? "The loaded model does not accept negative-prompt conditioning." : undefined}
        />
        <div className="absolute top-0 right-0 px-2 py-1 text-xs text-gray-400 pointer-events-none">
          {negativePromptTokenCount} tokens
        </div>
        {!supportsNegativePrompt && (
          <p className="mt-1 text-xs text-gray-500">Unavailable for the loaded model; the saved value is preserved.</p>
        )}
      </div>
    </Card>
  );

  // Derived, once per render, for VideoChainConfirmDialog: the plan (segments
  // + the length the chain actually reaches) and any conditioning-drop
  // disclosures specific to what THIS request would carry into a chain.
  const videoChainPlan = videoChainPrompt
    ? planVideoChain(archCapabilities, loadedArch, videoChainPrompt.targetFrames, videoChainPrompt.segmentFrames)
    : null;
  const videoChainFinalSeconds = videoChainPlan && videoChainPrompt
    ? (videoChainPlan.finalFrames / (videoChainPrompt.videoParams.frame_rate ?? 24)).toFixed(2)
    : null;
  const videoChainNotes: string[] = [];
  if (videoChainPrompt) {
    const refs = videoChainPrompt.references;
    const hasNonImageRefs =
      (refs?.videos?.length ?? 0) > 0 ||
      (refs?.audios?.length ?? 0) > 0 ||
      (refs?.videoAudios?.filter(Boolean).length ?? 0) > 0;
    if (videoChainPrompt.isRef2Va && hasNonImageRefs) {
      videoChainNotes.push(
        "Video/audio references condition segment 1 only: the temporal-outpaint continuation request accepts image references only."
      );
    }
    if (((videoChainPrompt.videoParams as Ref2VidParams).keyframes?.length ?? 0) > 0) {
      videoChainNotes.push(
        "Keyframe anchors (including a frame_index of -1, which pins to the end of segment 1, not the end of the finished chain) apply to segment 1 only; continuation segments carry no keyframes."
      );
    }
  }
  // What POST /video-chain/plan is asked for. Only IMAGE references are listed:
  // they are the only kind a continuation segment can be given, so they are the
  // only kind a per-segment binding could apply to.
  const videoChainPlanInput: VideoChainPlanInput | null =
    videoChainPrompt && loadedArch
      ? {
          architecture: loadedArch,
          variant: videoChainPrompt.variant,
          rootPrompt: videoChainPrompt.videoParams.prompt,
          negativePrompt: videoChainPrompt.videoParams.negative_prompt,
          targetFrames: videoChainPrompt.targetFrames,
          fps: videoChainPrompt.videoParams.frame_rate ?? 24,
          requestedSegmentFrames: videoChainPrompt.segmentFrames,
          rootSeed: videoChainPrompt.videoParams.seed,
          references: videoChainPrompt.isRef2Va
            ? buildChainImageReferenceInventory(videoChainPrompt.references?.images)
            : [],
        }
      : null;

  return (
    <ResizableColumns
      storageKey={GENERATION_WORKSPACE_SPLIT_KEY}
      label="Settings and preview width"
      defaultPrimaryPercent={46}
      minPrimaryPercent={34}
      maxPrimaryPercent={66}
      minPrimaryPx={360}
      minSecondaryPx={540}
    >
      {/* Parameters Panel */}
      <div className="generation-settings space-y-2">
        <ModelLoadSection
          onModelLoad={async () => {
            // Reload model info when model changes
            const modelInfo = await getCurrentModel();
            setCurrentModelInfo(modelInfo);
            console.log("[Txt2Img] Model changed, updated currentModelInfo:", modelInfo);

            // Auto-adjust sampler/schedule for Flow Matching models (Z-Image, FLUX.2)
            const modelType = modelInfo?.model_info?.type;
            if (modelType === "zimage" || modelType === "flux2" || modelType === "anima") {
              // Flow Matching models: use Euler with flow schedule
              setParams(prev => ({
                ...prev,
                sampler: "euler",
                schedule_type: "flow"
              }));
              console.log("[Txt2Img] Auto-set sampler=euler, schedule_type=flow for Flow Matching model");
            }
          }}
          visionEncoderPath={params.vision_encoder_path ?? null}
          onVisionEncoderChange={(path) => setParams({ ...params, vision_encoder_path: path })}
          vaePath={params.vae_path ?? null}
          onVaePathChange={(path) => setParams({ ...params, vae_path: path })}
          textEncoderPath={params.text_encoder_path ?? null}
          onTextEncoderChange={(path) => setParams({ ...params, text_encoder_path: path })}
          pidSrOutput={params.pid_sr_output ?? "4x"}
          onPidSrOutputChange={(value) => setParams({ ...params, pid_sr_output: value })}
          pidUseGemma={params.pid_use_gemma ?? false}
          onPidUseGemmaChange={(value) => setParams({ ...params, pid_use_gemma: value })}
          pidLowVram={params.pid_low_vram ?? false}
          onPidLowVramChange={(value) => setParams({ ...params, pid_low_vram: value })}
          pidTileNative={params.pid_tile_native ?? 512}
          onPidTileNativeChange={(value) => setParams({ ...params, pid_tile_native: value })}
          pidTileOverlapRatio={params.pid_tile_overlap_ratio ?? 0.25}
          onPidTileOverlapRatioChange={(value) => setParams({ ...params, pid_tile_overlap_ratio: value })}
          pidFastLargeDecode={params.pid_fast_large_decode ?? false}
          onPidFastLargeDecodeChange={(value) => setParams({ ...params, pid_fast_large_decode: value })}
          storageKeyPrefix="txt2img"
        />

        <GenerationLeadGrid
          conditioning={hasLeadConditioning ? (
            <>
        {/* Omni references are content conditioning. */}
        {isVideo && isRef2Va && (
          <>
            <details className="group -mb-1 rounded-md border border-gray-800/80 bg-gray-900/35 px-3 py-1.5 text-xs text-gray-500">
              <summary className="cursor-pointer select-none text-gray-400 marker:text-gray-600 hover:text-gray-300">
                MiniMax reference behavior
              </summary>
              <p className="mt-2 leading-relaxed">
                Adding a video reference uses MiniMax&apos;s documented video
                continuation task. The reference and generated span are laid
                out frame-contiguously, and the entire output is regenerated.
                Combining it with an image anchor performs video continuation
                plus keyframe completion.
              </p>
            </details>
            <MiniMaxH3ReferenceSelector
              value={h3References}
              onChange={setH3References}
              referenceImageSize={h3ReferenceImageSize}
              onReferenceImageSizeChange={setH3ReferenceImageSize}
              disabled={isGenerating}
            />
          </>
        )}

        {/* FLUX.2 Image Edit / Vision Encoder: Reference Images */}
        {(currentModelInfo?.model_info?.type === "flux2" || params.vision_encoder_path) && (
          <Card
            title={currentModelInfo?.model_info?.type === "flux2" ? "FLUX.2 Image Edit (Reference Images)" : "Vision Encoder (Reference Images)"}

            collapsible={true}
            defaultCollapsed={true}
            storageKey="txt2img_ref_images_collapsed"
            collapsedPreview={
              refImages.length > 0 ? (
                <span className="text-green-400 text-sm">✓ {refImages.length} image{refImages.length > 1 ? 's' : ''}</span>
              ) : (
                <span className="text-gray-500 text-sm">No reference images</span>
              )
            }
          >
            <div className="space-y-3">
              {/* Upload section */}
              <div className="flex gap-2">
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg,image/webp"
                  multiple
                  onChange={handleRefImageUpload}
                  disabled={refImages.length >= 10}
                  className="flex-1 block w-full text-sm text-gray-400
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-lg file:border-0
                    file:text-sm file:font-medium
                    file:bg-blue-600 file:text-white
                    hover:file:bg-blue-700
                    file:cursor-pointer cursor-pointer
                    disabled:opacity-50 disabled:cursor-not-allowed"
                />
                {refImagePreviews.length > 0 && (
                  <Button
                    onClick={handleClearAllRefImages}
                    variant="secondary"
                    size="sm"
                    title="Clear all reference images"
                  >
                    Clear All
                  </Button>
                )}
              </div>

              {/* Thumbnails and drag & drop area (side by side) */}
              {refImagePreviews.length === 0 ? (
                <div
                  onDragOver={handleRefImageDragOver}
                  onDragLeave={handleRefImageDragLeave}
                  onDrop={handleRefImageDrop}
                  className={`h-32 bg-gray-800 rounded-lg border-2 border-dashed transition-colors flex items-center justify-center ${
                    isRefImageDragging
                      ? 'border-blue-500 bg-gray-700'
                      : 'border-gray-600'
                  }`}
                >
                  <p className="text-gray-500 text-center text-sm px-4">
                    {isRefImageDragging
                      ? 'Drop images here (max 10)'
                      : 'Drag and drop images here or use the file picker above (max 10)'}
                  </p>
                </div>
              ) : (
                <div>
                  {/* Thumbnail grid with integrated D&D area */}
                  <div className="grid grid-cols-3 sm:grid-cols-5 auto-rows-fr gap-2">
                    {refImagePreviews.map((preview, index) => (
                      <div
                        key={index}
                        className="relative aspect-square bg-gray-800 rounded-lg overflow-hidden border border-gray-700 group"
                      >
                        <img
                          src={preview}
                          alt={`Reference ${index + 1}`}
                          className="w-full h-full object-cover"
                        />
                        <button
                          onClick={() => handleRemoveRefImage(index)}
                          className="absolute top-1 right-1 bg-red-600 hover:bg-red-700 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                          title={`Remove image ${index + 1}`}
                        >
                          <X className="w-3 h-3" />
                        </button>
                        <span className="absolute bottom-1 left-1 bg-black/70 text-white text-xs px-1.5 py-0.5 rounded">
                          {index + 1}
                        </span>
                      </div>
                    ))}
                    {/* Drag & drop area fills remaining grid cells */}
                    {refImagePreviews.length < 10 && (
                      <div
                        onDragOver={handleRefImageDragOver}
                        onDragLeave={handleRefImageDragLeave}
                        onDrop={handleRefImageDrop}
                        className={`bg-gray-800 rounded-lg border-2 border-dashed transition-colors flex items-center justify-center ${
                          isRefImageDragging
                            ? 'border-blue-500 bg-gray-700'
                            : 'border-gray-600'
                        }`}
                        title="Drop more images here"
                        style={{
                          gridColumn: refImagePreviews.length % 5 === 0 ? 'span 5' : `span ${5 - (refImagePreviews.length % 5)}`,
                          gridRow: 'span 1'
                        }}
                      >
                        <p className="text-gray-400 text-center text-sm px-2">
                          {isRefImageDragging ? 'Drop images here' : 'Drop more images here'}
                        </p>
                      </div>
                    )}
                  </div>
                  {/* Info text */}
                  <p className="text-xs text-gray-400 mt-2">
                    💡 {refImages.length}/10 images. {refImages.length < 10 ? 'Drop more images in the area above' : 'Max reached'}
                  </p>
                </div>
              )}
            </div>
          </Card>
        )}

            </>
          ) : undefined}
          prompt={promptPanel}
          primaryDetails={(isVideo || isAudio) ? (
            <>

        {isAudio && (
          <Card title="Audio Settings">

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Duration (seconds)</label>
                <NumberInput
                  label="Duration (seconds)"
                  value={params.audio_duration ?? 30.0}
                  onCommit={(v) => setParams({ ...params, audio_duration: v })}
                  min={1}
                  max={600}
                  step={1}
                  parse="float"
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Steps</label>
                <NumberInput
                  label="Steps"
                  value={params.inference_steps ?? 8}
                  onCommit={(v) => setParams({ ...params, inference_steps: v })}
                  min={1}
                  max={100}
                  step={1}
                  parse="int"
                  className="w-full"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Shift</label>
                <NumberInput
                  label="Shift"
                  value={params.shift ?? 3.0}
                  onCommit={(v) => setParams({ ...params, shift: v })}
                  min={0}
                  max={20}
                  step={0.1}
                  parse="float"
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Guidance Scale</label>
                <NumberInput
                  label="Guidance Scale"
                  value={params.guidance_scale ?? 1.0}
                  onCommit={(v) => setParams({ ...params, guidance_scale: v })}
                  min={0}
                  max={20}
                  step={0.1}
                  parse="float"
                  className="w-full"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
              <Input
                type="number"
                label="Seed"
                value={params.seed ?? -1}
                onChange={(e) => {
                  const parsed = parseInt(e.target.value);
                  setParams({ ...params, seed: Number.isNaN(parsed) ? -1 : parsed });
                }}
              />
              <Select
                label="Vocal Language"
                value={params.vocal_language ?? "en"}
                onChange={(e) => setParams({ ...params, vocal_language: e.target.value })}
                options={[
                  { value: "en", label: "English" },
                  { value: "zh", label: "Chinese" },
                  { value: "ja", label: "Japanese" },
                  { value: "ko", label: "Korean" },
                  { value: "es", label: "Spanish" },
                  { value: "fr", label: "French" },
                  { value: "de", label: "German" },
                  { value: "ru", label: "Russian" },
                  { value: "it", label: "Italian" },
                  { value: "pt", label: "Portuguese" },
                ]}
              />
            </div>
          </Card>
        )}

        {isAudio && visibility.lora && (
          <LoRASelector
            value={params.loras || []}
            onChange={(loras) => {
              console.log("[Txt2Img] Audio LoRA onChange called with:", loras);
              setParams({ ...params, loras });
            }}
            disabled={isGenerating}
            storageKey="txt2img_audio_lora_collapsed"
            simpleMode
            loadedArch={loadedArch}
            onApplyRecommended={applyLoraRecommended}
          />
        )}

        {isVideo && (
          <Card title={`Video${loadedArchName ? ` (${loadedArchName})` : ""}`}>
            {/* Resolution / steps / frame rate / seed in the image models'
                Parameters-card shape: labelled sliders with a numeric entry
                beside them (common/Slider) in the same two-column grid, and
                the image path's seed control rather than a bare number box.
                There is no Scale size mode here: txt2vid has no input image to
                derive a size from (Img2Img/Outpaint, which do, have one). */}
            <div className="space-y-4">
              <div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <Slider
                    label={`Width (÷${videoWidthBounds.step})`}
                    min={videoWidthBounds.min}
                    max={videoWidthBounds.max}
                    step={videoWidthBounds.step}
                    value={videoCanvasWidth}
                    onChange={(e) => setParams({ ...params, width: parseInt(e.target.value) })}
                  />
                  <Slider
                    label={`Height (÷${videoHeightBounds.step})`}
                    min={videoHeightBounds.min}
                    max={videoHeightBounds.max}
                    step={videoHeightBounds.step}
                    value={videoCanvasHeight}
                    onChange={(e) => setParams({ ...params, height: parseInt(e.target.value) })}
                  />
                </div>
                {/* Why a slider stops where it does. Only rendered for an
                    architecture that HAS an envelope -- LTX-2.3 declares
                    none, so it keeps its full range and says nothing about
                    a cap. */}
                {videoWidthBounds.capped && (
                  <p className="text-xs text-gray-500 mt-1">
                    {videoCanvasRule(archCapabilities, loadedArch)}. The cap is on the
                    short and long edges rather than on width and height, so each
                    slider stops at the largest edge the other axis currently allows.
                  </p>
                )}
                {videoCanvasOverEnvelope && (
                  <p className="text-xs text-amber-400 mt-1">
                    The canvas is {videoCanvasWidth}x{videoCanvasHeight}, which is
                    outside this model&apos;s envelope. The value is kept as set — it
                    is not moved for you — and this model refuses it, so change it
                    before generating.
                  </p>
                )}
              </div>

              <VideoFrameCountSlider
                caps={archCapabilities}
                arch={loadedArch}
                value={params.num_frames ?? 121}
                onChange={(frames) => setParams({ ...params, num_frames: frames })}
                fallbackFps={params.frame_rate ?? 24.0}
                sliderMaxOverride={videoFrameSliderMax}
              />
              {/* Opt-in video-length chaining, segment length: unset (default)
                  means "never split" -- raising the frame count above does not
                  by itself trigger a chain unless the architecture still has a
                  hard `max_frames` wall (chainSegmentCap in api.ts falls back
                  to that automatically). Setting this lets the user split
                  DELIBERATELY, even on an architecture with no hard wall, e.g.
                  to keep each segment within the documented trained range. */}
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-1.5 text-xs text-gray-400">
                  <input
                    type="checkbox"
                    checked={chainSegmentFrames != null}
                    onChange={(e) => {
                      setChainSegmentFramesReplacedNotice(null);
                      if (!e.target.checked) {
                        setChainSegmentFrames(null);
                        return;
                      }
                      const c = loadedArch ? archCapabilities?.video_constraints?.[loadedArch] : undefined;
                      const seed = c?.max_frames ?? c?.trained_max_frames ?? params.num_frames ?? 121;
                      setChainSegmentFrames(normalizeVideoFrames(archCapabilities, loadedArch, seed) ?? seed);
                    }}
                  />
                  Chain segment length
                </label>
              </div>
              {chainSegmentFrames != null && (
                <VideoFrameCountSlider
                  caps={archCapabilities}
                  arch={loadedArch}
                  value={chainSegmentFrames}
                  onChange={(frames) => {
                    setChainSegmentFramesReplacedNotice(null);
                    setChainSegmentFrames(frames);
                  }}
                  fallbackFps={params.frame_rate ?? 24.0}
                  allowOverCap={false}
                  disabled={(params.num_frames ?? 0) <= chainSegmentFrames}
                  sliderMaxOverride={videoFrameSliderMax}
                />
              )}
              {chainSegmentFramesReplacedNotice && (
                <p className="text-xs text-amber-400">{chainSegmentFramesReplacedNotice}</p>
              )}
              {chainSegmentFrames == null ? (
                <p className="text-xs text-gray-500">
                  A chained generation is never split by default; check the box to split into requests of a fixed length.
                </p>
              ) : (params.num_frames ?? 0) <= chainSegmentFrames ? (
                <p className="text-xs text-gray-500">
                  Chain segment length has no effect while the total frame count ({params.num_frames ?? 0}) is at or below it.
                </p>
              ) : (
                <p className="text-xs text-gray-500">
                  A chained generation splits into requests of at most {chainSegmentFrames} frames each.
                </p>
              )}
              {currentItem?.loopGroupId && currentItem.chainTargetFrames != null && (
                <p className="text-xs text-amber-400">
                  Video chain: segment {(currentItem.loopStepIndex ?? -1) + 2}
                  {" "}running (target {currentItem.chainTargetFrames} frames)
                  {currentItem.chainPlanHash
                    ? `, plan ${currentItem.chainPlanHash.slice(0, 12)}.`
                    : ". Legacy repeat: every segment is sent the same full-length prompt."}
                </p>
              )}
              {videoChainStoppedMessage && (
                <p className="text-xs text-amber-400">{videoChainStoppedMessage}</p>
              )}

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Slider
                  label="Steps"
                  // Arch floor, not 1: `validate_video_steps` answers 400 below
                  // it (MiniMax-H3 declares 2 -- its steps are sigma grid
                  // points, so 1 evaluates nothing).
                  min={videoMinInferenceSteps(archCapabilities, loadedArch)}
                  max={100}
                  step={1}
                  value={params.num_inference_steps ?? 8}
                  onChange={(e) => setParams({ ...params, num_inference_steps: parseInt(e.target.value) })}
                />
                <Slider
                  label="Frame Rate (fps)"
                  min={1}
                  max={resolveBound("video_frame_rate_max", generationDefaults?.param_bounds, sliderBounds, params.frame_rate ?? 24.0)}
                  step={1}
                  value={params.frame_rate ?? 24.0}
                  onChange={(e) => setParams({ ...params, frame_rate: parseFloat(e.target.value) })}
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Guidance: hidden on an architecture that declares it
                    unsupported (MiniMax-H3 is guidance-distilled — it has no
                    guider and no unconditional branch, so the sampler takes no
                    scale at all). Driven by the capability matrix, not by an
                    arch name kept here. */}
                {supportsCfg && (
                  <Slider
                    label="Guidance Scale"
                    min={0}
                    max={20}
                    step={0.1}
                    value={params.guidance_scale ?? 1.0}
                    onChange={(e) => setParams({ ...params, guidance_scale: parseFloat(e.target.value) })}
                  />
                )}
                {/* Seed: the image path's control, verbatim -- a labelled
                    number field plus randomise / reset-to--1 / reuse-the-
                    preview's-seed. */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Seed
                  </label>
                  <div className="flex gap-2">
                    <Input
                      type="number"
                      value={params.seed ?? -1}
                      onChange={(e) => {
                        const parsed = parseInt(e.target.value);
                        setParams({ ...params, seed: Number.isNaN(parsed) ? -1 : parsed });
                      }}
                      className="flex-1 min-w-0"
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
                      onClick={() => generatedVideoSeed !== null && setParams({ ...params, seed: generatedVideoSeed })}
                      variant="secondary"
                      size="sm"
                      title="Use seed from the result video"
                      disabled={generatedVideoSeed === null}
                    >
                      ♻️
                    </Button>
                  </div>
                </div>
              </div>
            </div>

            <label className="flex items-center gap-2 cursor-pointer mt-2">
              <input
                type="checkbox"
                checked={params.audio_enable ?? true}
                onChange={(e) => setParams({ ...params, audio_enable: e.target.checked })}
                className="rounded"
              />
              <span className="text-gray-300 text-sm">Audio</span>
            </label>

            <VideoAccelerationControls
              idPrefix="txt2vid"
              values={params}
              onChange={(patch) => setParams({ ...params, ...patch })}
              supportsSpectrum={supportsSpectrum}
              supportsFbcache={supportsFbcache}
              supportsFuseOutputProj={supportsFuseOutputProj}
              blocksToSwapEnabledDefault={videoBlocksToSwapEnabledDefault}
              blockSwapMax={VIDEO_BLOCK_SWAP_MAX}
            />

            {/* Factual notes from MiniMax's own release documentation (README +
                prompt-writing guide + the reproducible request scripts). No
                quality claims: the prompt shape below is the output format of
                MiniMax's H3-Context-IR stage, which is not open-sourced. */}
            {loadedArch === "minimax_h3" && (
              <div className="mt-2 flex items-center gap-1 text-xs text-gray-500">
                <span>Structured prompt recommended; steps are sigma points</span>
                <InlineHelp label="MiniMax-H3 generation details">
                  <p>
                    Use a structured block with integrated multimodal description, overall soundscape, and non-diegetic music sections.
                  </p>
                  <p>
                    Video and audio are generated jointly. Turning Audio off skips decoding and muxing, while audio still participates in generation.
                  </p>
                  <p>N steps run N-1 model evaluations (minimum 2). MiniMax does not publish a recommended step count.</p>
                </InlineHelp>
              </div>
            )}
          </Card>
        )}

        {isVideo && visibility.lora && (
          <LoRASelector
            value={params.loras || []}
            onChange={(loras) => {
              console.log("[Txt2Img] Video LoRA onChange called with:", loras);
              setParams({ ...params, loras });
            }}
            disabled={isGenerating}
            storageKey="txt2img_video_lora_collapsed"
            loadedArch={loadedArch}
            onApplyRecommended={applyLoraRecommended}
          />
        )}
            </>
          ) : undefined}
        />

        {/* C5: keyframe anchors, a track separate from the references above --
            content conditioning (references) vs placement conditioning
            (anchors). Only where the loaded arch declares placement support;
            ref2va's endpoint accepts it regardless of whether any reference is
            set, but it is only useful once at least one is. */}
        {isVideo && isRef2Va && archSupportsFeature(archCapabilities, loadedArch, "keyframe_placement") && (
          <MiniMaxH3Ref2VidKeyframeSelector
            value={h3Keyframes}
            onChange={setH3Keyframes}
            disabled={isGenerating}
          />
        )}

        {!isVideo && !isAudio && (<>
        {/* Txt2Img Options: a single-open tabbed accordion (chrome shared via
            frontend/src/components/common/TabbedOptions.tsx). Every control
            below is unchanged from its original location (same param
            binding / handler / conditional reveal) -- only the container
            changed. See TXT2IMG_OPTIONS_TAB_KEYS / isTxt2ImgOptionsTabActive /
            txt2imgOptionsTabRender above. */}
        <TabbedOptions<GenerationParams>
          cardTitle="Txt2Img Options"
          params={params}
          setParams={setParams}
          defaultParams={DEFAULT_PARAMS}
          tabs={TXT2IMG_OPTIONS_TABS.map((tab) => ({
            id: tab.id,
            label: tab.label,
            keys: TXT2IMG_OPTIONS_TAB_KEYS[tab.id],
            isActive: (p: GenerationParams) => isTxt2ImgOptionsTabActive(tab.id, p),
            render: txt2imgOptionsTabRender[tab.id],
          }))}
        />

        <Card title="Parameters">
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Slider
                label="Steps"
                min={1}
                max={resolveBound("steps_max", generationDefaults?.param_bounds, sliderBounds, params.steps)}
                step={1}
                value={params.steps}
                onChange={(e) => setParams({ ...params, steps: parseInt(e.target.value) })}
              />
              <Slider
                label="CFG Scale"
                min={0}
                max={resolveBound("cfg_scale_max", generationDefaults?.param_bounds, sliderBounds, params.cfg_scale)}
                step={0.5}
                value={params.cfg_scale}
                onChange={(e) => setParams({ ...params, cfg_scale: parseFloat(e.target.value) })}
              />
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Slider
                  label="Width"
                  min={64}
                  max={resolveBound("image_width_max", generationDefaults?.param_bounds, sliderBounds, params.width)}
                  step={resolutionStep}
                  value={params.width}
                  onChange={(e) => setParams({ ...params, width: parseInt(e.target.value) })}
                />
                <Slider
                  label="Height"
                  min={64}
                  max={resolveBound("image_height_max", generationDefaults?.param_bounds, sliderBounds, params.height)}
                  step={resolutionStep}
                  value={params.height}
                  onChange={(e) => setParams({ ...params, height: parseInt(e.target.value) })}
                />
              </div>

              {visibility.aspectRatioPresets && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="block text-sm font-medium text-gray-300">Aspect Ratio Presets</label>
                    <div className="flex gap-2">
                      <span className="text-xs text-gray-400">Base on:</span>
                      <label className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="radio"
                          name="aspect_base_txt2img"
                          value="width"
                          defaultChecked
                          className="w-3 h-3"
                        />
                        <span className="text-xs text-gray-300">Width</span>
                      </label>
                      <label className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="radio"
                          name="aspect_base_txt2img"
                          value="height"
                          className="w-3 h-3"
                        />
                        <span className="text-xs text-gray-300">Height</span>
                      </label>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
                    {aspectRatioPresets.map((preset) => (
                      <button
                        key={preset.label}
                        onClick={() => {
                          const baseOn = (document.querySelector('input[name="aspect_base_txt2img"]:checked') as HTMLInputElement)?.value || 'width';
                          let newWidth: number, newHeight: number;

                          if (baseOn === 'width') {
                            newWidth = params.width;
                            newHeight = Math.round(params.width / preset.ratio / 8) * 8;
                          } else {
                            newHeight = params.height;
                            newWidth = Math.round(params.height * preset.ratio / 8) * 8;
                          }

                          setParams({ ...params, width: newWidth, height: newHeight });
                        }}
                        className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded transition-colors"
                        title={`Aspect ratio ${preset.label}`}
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {visibility.fixedResolutionPresets && (
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-gray-300">Fixed Resolution Presets</label>
                  <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
                    {fixedResolutionPresets.map((preset) => (
                      <button
                        key={`${preset.width}x${preset.height}`}
                        onClick={() => setParams({ ...params, width: preset.width, height: preset.height })}
                        className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded transition-colors"
                        title={`${preset.width}×${preset.height}`}
                      >
                        {preset.width}×{preset.height}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
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
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
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
                  onClick={() => generatedImageAncestralSeed !== null && generatedImageAncestralSeed !== -1 && setParams({ ...params, ancestral_seed: generatedImageAncestralSeed })}
                  variant="secondary"
                  size="sm"
                  title="Use ancestral seed from preview image"
                  disabled={generatedImageAncestralSeed === null || generatedImageAncestralSeed === -1}
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

        {visibility.lora && (
          <LoRASelector
            value={params.loras || []}
            onChange={(loras) => {
              console.log("[Txt2Img] LoRA onChange called with:", loras);
              setParams({ ...params, loras });
            }}
            disabled={isGenerating}
            storageKey="txt2img_lora_collapsed"
            loadedArch={loadedArch}
            onApplyRecommended={applyLoraRecommended}
          />
        )}

        {visibility.controlnet && (
          <ControlNetSelector
            value={params.controlnets || []}
            onChange={(controlnets) => {
              console.log("[Txt2Img] ControlNet onChange called with:", controlnets);
              setParams({ ...params, controlnets });
            }}
            disabled={isGenerating}
            storageKey="txt2img_controlnet_collapsed"
          />
        )}

        {/* Loop Generation */}
        <LoopGenerationPanel
          config={loopGenerationConfig}
          onChange={setLoopGenerationConfig}
          mode="txt2img"
          mainWidth={params.width}
          mainHeight={params.height}
          samplers={samplers}
          scheduleTypes={scheduleTypes}
        />
        </>)}
      </div>

      {/* Preview Panel */}
      <div className="pb-16 lg:pb-0">
        <Card title="Preview">
          <ResizableColumns
            storageKey={GENERATION_PREVIEW_QUEUE_SPLIT_KEY}
            label="Preview and queue width"
            defaultPrimaryPercent={68}
            minPrimaryPercent={55}
            maxPrimaryPercent={82}
            minPrimaryPx={300}
            minSecondaryPx={200}
            className="lg:h-[800px]"
          >
            {/* Left: Preview and Controls */}
            <div className="flex-1 flex flex-col space-y-2 min-w-0">
              {/* Action Buttons - Desktop only (hidden on mobile) */}
              <div className="hidden lg:flex gap-2 relative">
              <Button
                onClick={() => {
                  if (isLongPressTriggeredRef.current) {
                    isLongPressTriggeredRef.current = false;
                    return;
                  }
                  handleAddToQueue();
                }}
                onContextMenu={(e) => {
                  e.preventDefault();
                  const pos = toFixedViewportPosition(e.clientX, e.clientY);
                  setMenuPosition({ x: pos.x, y: pos.y });
                  setShowForeverMenu(true);
                }}
                className="flex-1"
                size="lg"
              >
                {isGenerating ? "Add to Queue" : generateForever ? "Generate Forever ∞" : "Generate"}
              </Button>

              {isGenerating && (
                <Button
                  onClick={async () => {
                    try {
                      await cancelGeneration();
                      setGenerateForever(false);
                      // Cancel all pending loop steps if this is part of a loop group
                      if (currentItem?.loopGroupId) {
                        cancelLoopGroup(currentItem.loopGroupId);
                      }
                      // Don't call processQueue() here - let the error handler handle it
                      // to avoid race condition with reset_cancel_flag()
                    } catch (error) {
                      console.error("Failed to cancel generation:", error);
                    }
                  }}
                  variant="secondary"
                  size="lg"
                  title="Cancel generation and move to next"
                >
                  Cancel
                </Button>
              )}
              <Button
                onClick={resetToDefault}
                disabled={isGenerating}
                variant="secondary"
                size="lg"
              >
                Reset
              </Button>
            </div>

              {/* Action Buttons - Mobile only (fixed bar at bottom with inline toggle) */}
              <div className={`lg:hidden fixed bottom-0 z-40 bg-gray-900 border-t transition-all ${isMobileControlsOpen ? 'left-0 right-0 border-gray-700' : 'left-auto right-0 border-l border-gray-700'}`}>
                <div className="flex gap-2 p-3 items-center">
                  {/* Buttons (conditionally visible) */}
                  {isMobileControlsOpen && (
                    <>
                      <Button
                        onClick={() => {
                          if (isLongPressTriggeredRef.current) {
                            isLongPressTriggeredRef.current = false;
                            return;
                          }
                          handleAddToQueue();
                        }}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          const pos = toFixedViewportPosition(e.clientX, e.clientY);
                          setMenuPosition({ x: pos.x, y: pos.y });
                          setShowForeverMenu(true);
                        }}
                        onTouchStart={handleGenerateTouchStart}
                        onTouchEnd={handleGenerateTouchEnd}
                        onTouchCancel={handleGenerateTouchEnd}
                        className="flex-1"
                        size="lg"
                      >
                        {isGenerating ? "Add Queue" : generateForever ? "Generate Forever ∞" : "Generate"}
                      </Button>
                      {isGenerating && (
                        <button
                          onClick={async () => {
                            try {
                              await cancelGeneration();
                              setGenerateForever(false);
                              // Cancel all pending loop steps if this is part of a loop group
                              if (currentItem?.loopGroupId) {
                                cancelLoopGroup(currentItem.loopGroupId);
                              }
                              // Don't call processQueue() here - let the error handler handle it
                              // to avoid race condition with reset_cancel_flag()
                            } catch (error) {
                              console.error("Failed to cancel generation:", error);
                            }
                          }}
                          className="p-3 bg-gray-800 hover:bg-gray-700 text-white rounded transition-colors"
                          title="Cancel generation"
                        >
                          <X className="h-6 w-6" />
                        </button>
                      )}
                      <button
                        onClick={resetToDefault}
                        disabled={isGenerating}
                        className="p-3 bg-gray-800 hover:bg-gray-700 text-white rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Reset to default"
                      >
                        <RotateCcw className="h-6 w-6" />
                      </button>
                    </>
                  )}

                  {/* Toggle button (always visible on the right) */}
                  <button
                    onClick={() => setIsMobileControlsOpen(!isMobileControlsOpen)}
                    className="p-2 text-gray-400 hover:text-white transition-colors flex-shrink-0"
                  >
                    {isMobileControlsOpen ? <ChevronRight className="h-6 w-6" /> : <ChevronLeft className="h-6 w-6" />}
                  </button>
                </div>
              </div>

              {/* Context/long-press menu for generate forever */}
              {showForeverMenu && (
                <>
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setShowForeverMenu(false)}
                  />
                  <div
                    className="fixed z-50 bg-gray-800 border border-gray-600 rounded shadow-lg py-1"
                    style={{ left: menuPosition.x, top: menuPosition.y }}
                  >
                    <button
                      onClick={() => {
                        setGenerateForever(!generateForever);
                        setShowForeverMenu(false);
                      }}
                      className="w-full px-4 py-2 text-left hover:bg-gray-700 flex items-center gap-2"
                    >
                      <span className="w-4">{generateForever ? "✓" : ""}</span>
                      <span>Generate Forever</span>
                    </button>
                  </div>
                </>
              )}

            {/* Preview Predicted x0 toggle */}
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="preview_predicted_x0"
                checked={params.preview_predicted_x0 || false}
                onChange={(e) => setParams({ ...params, preview_predicted_x0: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="preview_predicted_x0" className="text-sm text-gray-300">
                Preview Predicted x0
              </label>
            </div>

            {/* Live-preview decoder — only meaningful for AutoencoderKLFlux2-latent
                models (FLUX.2 / Lens / Ideogram 4). Other architectures ignore the
                preview_decoder value (SD/SDXL: TAESD, Z-Image/Anima: latent-direct,
                MiniT2I: pixel-space RGB-direct), so the selector is hidden for them. */}
            {(currentModelInfo?.model_info?.type === "flux2"
              || currentModelInfo?.model_info?.type === "lens"
              || currentModelInfo?.model_info?.type === "ideogram4") && (
              <div className="flex items-center gap-2">
                <label htmlFor="preview_decoder" className="text-sm text-gray-300">
                  Preview Decoder
                </label>
                <select
                  id="preview_decoder"
                  value={params.preview_decoder || "matrix"}
                  onChange={(e) => setParams({ ...params, preview_decoder: e.target.value })}
                  className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1"
                  title="FLUX.2 / Lens / Ideogram 4 のライブプレビュー方式。matrix=線形変換（軽量）、TAEF2=tiny decoder（高精度）"
                >
                  <option value="matrix">Matrix (linear, light)</option>
                  <option value="taef2">TAEF2 (FLUX.2 VAE)</option>
                </select>
              </div>
            )}

            {/* Use training model toggle (only enabled when an LoRA/Full-FT
                training is active).  When ON, generate calls the
                /generate/txt2img/training-preview endpoint and renders
                a transient preview using the in-training model state. */}
            <div className="flex items-center gap-2"
                 title={activeTraining
                   ? `Active: ${activeTraining.run_name ?? `run #${activeTraining.run_id}`} (step ${activeTraining.current_step ?? "?"})`
                   : "No active LoRA/Full-FT training"}>
              <input
                type="checkbox"
                id="use_training_model"
                checked={useTrainingModel}
                disabled={!activeTraining}
                onChange={(e) => setUseTrainingModel(e.target.checked)}
                className="rounded disabled:opacity-50"
              />
              <label htmlFor="use_training_model"
                     className={`text-sm ${activeTraining ? "text-gray-300" : "text-gray-500"}`}>
                Use training model
                {useTrainingModel && activeTraining && (
                  <span className="ml-1 text-xs text-emerald-400">
                    · {activeTraining.run_name ?? `run #${activeTraining.run_id}`} (step {activeTraining.current_step ?? "?"})
                  </span>
                )}
              </label>
            </div>

            {/* Sub-toggle: persist preview to gallery (only meaningful
                when Use training model is on).  Indented to visually
                indicate the dependency. */}
            {useTrainingModel && (
              <div className="flex items-center gap-2 ml-6"
                   title="Save preview PNG to outputs/ and the gallery (tagged as training-preview)">
                <input
                  type="checkbox"
                  id="save_preview_to_gallery"
                  checked={savePreviewToGallery}
                  onChange={(e) => setSavePreviewToGallery(e.target.checked)}
                  className="rounded"
                />
                <label htmlFor="save_preview_to_gallery" className="text-sm text-gray-300">
                  Save preview to gallery
                </label>
              </div>
            )}

            {isGenerating && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-gray-400">
                  <span>{progressMessage || "Generating..."}</span>
                  <span>{progress}/{totalSteps} steps</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-200"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>
            )}
            <div
              className="w-full aspect-square max-h-[500px] lg:max-h-none bg-gray-800 rounded-lg flex items-center justify-center cursor-pointer"
              onDoubleClick={() => {
                if (generatedImage) {
                  setPreviewViewerOpen(true);
                }
              }}
            >
              {isVideo && generatedVideo ? (
                <div className="w-full space-y-2">
                  <video
                    src={generatedVideo}
                    className="w-full rounded-lg"
                    controls
                    loop
                    muted
                    autoPlay
                    playsInline
                    onError={() => {
                      // The file is gone (outputs/ cleared, run deleted) --
                      // show an empty preview rather than a dead player.
                      console.warn("[Txt2Img] Preview video failed to load, clearing:", generatedVideo);
                      clearVideoPreview(PREVIEW_KEYS);
                      setGeneratedVideo(null);
                      setGeneratedVideoInfo(null);
                    }}
                  />
                  {generatedVideoInfo && (
                    <div className="text-xs text-gray-400">
                      {generatedVideoInfo.num_frames != null && <span>{generatedVideoInfo.num_frames} frames</span>}
                      {generatedVideoInfo.fps != null && <span> · {generatedVideoInfo.fps} fps</span>}
                      {generatedVideoInfo.duration != null && Number.isFinite(Number(generatedVideoInfo.duration)) && <span> · {Number(generatedVideoInfo.duration).toFixed(2)}s</span>}
                    </div>
                  )}
                  {generatedVideoWarnings.length > 0 && (
                    <ul className="text-xs text-amber-400 list-disc pl-4 space-y-1">
                      {generatedVideoWarnings.map((w, i) => <li key={i}>{w}</li>)}
                    </ul>
                  )}
                </div>
              ) : isAudio && generatedAudio ? (
                <div className="w-full space-y-2">
                  <audio
                    src={generatedAudio}
                    className="w-full"
                    controls
                    onError={() => {
                      // The file is gone (outputs/ cleared, run deleted) --
                      // show an empty preview rather than a dead player.
                      console.warn("[Txt2Img] Preview audio failed to load, clearing:", generatedAudio);
                      clearAudioPreview(PREVIEW_KEYS);
                      setGeneratedAudio(null);
                      setGeneratedAudioInfo(null);
                    }}
                  />
                  {generatedAudioInfo && (
                    <div className="text-xs text-gray-400">
                      {generatedAudioInfo.duration != null && Number.isFinite(Number(generatedAudioInfo.duration)) && <span>{Number(generatedAudioInfo.duration).toFixed(2)}s</span>}
                      {generatedAudioInfo.sample_rate != null && <span> · {generatedAudioInfo.sample_rate} Hz</span>}
                    </div>
                  )}
                </div>
              ) : generatedImage ? (
                <img
                  src={effectiveGeneratedImage ?? generatedImage}
                  alt="Generated"
                  className="max-w-full max-h-full rounded-lg"
                  style={{ filter: buildFilterString(postEdit) }}
                  onError={() => {
                    // The file went away while the panel was open -- show an
                    // empty preview rather than a broken image, same as the
                    // video/audio players above. Confirmed with a HEAD first,
                    // so a hot reload or a backend blip cannot discard a result
                    // that is still on disk (see helper).
                    imagePreviewGone(effectiveGeneratedImage ?? generatedImage, generatedImage).then((gone) => {
                      if (!gone) return;
                      console.warn("[Txt2Img] Preview image failed to load, clearing:", generatedImage);
                      clearImagePreview(PREVIEW_KEYS);
                      setGeneratedImage(null);
                    });
                  }}
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

            {/* Post-Edit controls (client-side brightness/saturation) */}
            {generatedImage && (
              <div className="mt-3">
                <PostEditControls value={postEdit} onChange={setPostEdit} />
              </div>
            )}

            {/* CFG Metrics Graph (Developer Mode) */}
            {developerMode && cfgMetrics.length > 0 && (
              <div className="mt-4">
                <div className="text-sm text-gray-400 mb-2">CFG Metrics (Developer Mode)</div>
                <CFGMetricsGraph metrics={cfgMetrics} />
              </div>
            )}

            {/* VRAM Inspector (Developer Mode) */}
            {developerMode && <VramInspector />}

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
                <div className="grid grid-cols-1 sm:grid-cols-6 gap-2">
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
                  <Button
                    onClick={sendToOutpaint}
                    variant="secondary"
                    size="sm"
                    disabled={!sendImage && !sendPrompt && !sendParameters}
                  >
                    Send to outpaint
                  </Button>
                  <Button
                    onClick={sendToUpscale}
                    variant="secondary"
                    size="sm"
                    disabled={!generatedImage}
                  >
                    Send to Upscale
                  </Button>
                  <SendToStudioButton
                    media={{ kind: "image", url: generatedImage, masterUrl: stripCacheBuster(generatedImage), width: generatedImageParams?.width, height: generatedImageParams?.height }}
                    parameters={generatedImageParams || params}
                    sendMedia={sendImage}
                    sendPrompt={sendPrompt}
                    sendParameters={sendParameters}
                  />
                </div>
              </div>
            )}

            {isVideo && generatedVideo && (
              <div className="space-y-3 mt-4">
                <div className="grid grid-cols-2 gap-2">
                  <Button onClick={sendVideoResultToInpaint} variant="secondary" size="sm">
                    Send to inpaint
                  </Button>
                  <Button onClick={sendVideoResultToOutpaint} variant="secondary" size="sm">
                    Send to outpaint
                  </Button>
                  <Button
                    onClick={sendVideoResultToReference}
                    variant="secondary"
                    size="sm"
                    className="col-span-2"
                    title="Condition a new generation on this whole clip (MiniMax-H3 ref2va). Regenerates everything; use Send to outpaint to extend the clip in place instead."
                  >
                    Use as reference video
                  </Button>
                  <SendToStudioButton
                    media={{ kind: "video", url: generatedVideo, duration: generatedVideoInfo?.duration, width: params.width, height: params.height }}
                    parameters={{
                      ...(generatedVideoParams || params),
                      num_frames: generatedVideoInfo?.num_frames ?? generatedVideoParams?.num_frames ?? params.num_frames,
                      frame_rate: generatedVideoInfo?.fps ?? generatedVideoParams?.frame_rate ?? params.frame_rate,
                      seed: generatedVideoSeed ?? generatedVideoParams?.seed ?? params.seed,
                    }}
                    className="col-span-2"
                  />
                </div>
              </div>
            )}

            {isAudio && generatedAudio && (
              <div className="space-y-3 mt-4">
                <div className="grid grid-cols-2 gap-2">
                  <Button onClick={sendAudioResultToOutpaint} variant="secondary" size="sm">
                    Send to outpaint
                  </Button>
                  <Button onClick={sendAudioResultToImg2Img} variant="secondary" size="sm">
                    Send to img2img
                  </Button>
                  <SendToStudioButton
                    media={{ kind: "audio", url: generatedAudio, duration: generatedAudioInfo?.duration }}
                    parameters={generatedAudioParams || params}
                    className="col-span-2"
                  />
                </div>
              </div>
            )}
            </div>

            {/* Right: Generation Queue */}
            <div className="w-full">
              <GenerationQueue currentStep={progress} />
            </div>
          </ResizableColumns>
        </Card>
      </div>

      {/* Preview Image Viewer */}
      {previewViewerOpen && generatedImage && (
        <ImageViewer
          imageUrl={generatedImage}
          onClose={() => setPreviewViewerOpen(false)}
          postEdit={postEdit}
          onPostEditChange={setPostEdit}
        />
      )}

      {/* TIPO Dialog */}
      <TIPODialog
        isOpen={isTIPODialogOpen}
        onClose={() => setIsTIPODialogOpen(false)}
        settings={tipoSettings}
        onSettingsChange={setTipoSettings}
      />

      {/* Opt-in video-length chain choice: never auto-chain, see CLAUDE.md */}
      <VideoChainConfirmDialog
        isOpen={videoChainPrompt != null}
        requestedFrames={videoChainPrompt?.targetFrames ?? 0}
        capFrames={videoChainPrompt?.capFrames ?? 0}
        capSeconds={
          videoChainPrompt
            ? (videoChainPrompt.capFrames / (videoChainPrompt.videoParams.frame_rate ?? 24)).toFixed(2)
            : null
        }
        finalSeconds={videoChainFinalSeconds}
        plan={videoChainPlan}
        planInput={videoChainPlanInput}
        notes={videoChainNotes}
        onCancel={() => setVideoChainPrompt(null)}
        onGenerateAtCap={handleVideoChainGenerateAtCap}
        onStartChain={handleVideoChainStart}
      />

    </ResizableColumns>
  );
}
