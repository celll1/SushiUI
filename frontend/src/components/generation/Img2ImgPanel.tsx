"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import { ChevronLeft, ChevronRight, X, RotateCcw } from "lucide-react";
import Card from "../common/Card";
import TabbedOptions from "../common/TabbedOptions";
import Input from "../common/Input";
import NumberInput from "../common/NumberInput";
import Textarea, {
  GENERATION_LYRICS_HEIGHT_KEY,
  GENERATION_NEGATIVE_PROMPT_HEIGHT_KEY,
  GENERATION_PROMPT_HEIGHT_KEY,
} from "../common/Textarea";
import TextareaWithTagSuggestions from "../common/TextareaWithTagSuggestions";
import Button from "../common/Button";
import Slider from "../common/Slider";
import Select from "../common/Select";
import ModelLoadSection from "../common/ModelLoadSection";
import LoRASelector from "../common/LoRASelector";
import ControlNetSelector from "../common/ControlNetSelector";
import ImageEditor from "../common/ImageEditor";
import TIPODialog, { TIPOSettings } from "../common/TIPODialog";
import { fixFloatingPointParams } from "@/utils/numberUtils";
import ImageViewer from "../common/ImageViewer";
import PostEditControls from "../common/PostEditControls";
import { PostEditState, NEUTRAL_POST_EDIT, buildFilterString } from "@/utils/postEdit";
import { usePostEditPreview } from "@/hooks/usePostEditPreview";
import GenerationQueue from "../common/GenerationQueue";
import GenerationLeadGrid from "../common/GenerationLeadGrid";
import InlineHelp from "../common/InlineHelp";
import SendToStudioButton from "../studio/SendToStudioButton";
import ResizableColumns, {
  GENERATION_PREVIEW_QUEUE_SPLIT_KEY,
  GENERATION_WORKSPACE_SPLIT_KEY,
} from "../common/ResizableColumns";
import H3PromptAssist from "../common/H3PromptAssist";
import LoopGenerationPanel, { LoopGenerationConfig } from "./LoopGenerationPanel";
import QuantizedGemmSelect from "./QuantizedGemmSelect";
import MiniMaxH3KeyframeTimeline from "../common/MiniMaxH3KeyframeTimeline";
import MiniMaxH3ReferenceSelector, { EMPTY_MINIMAX_H3_REFERENCES, countMiniMaxH3References, MAX_VIDEOS, MAX_TOTAL } from "../common/MiniMaxH3ReferenceSelector";
import { migrateLoopGenerationConfig, computeLoopDecodeDirective } from "@/utils/loopGenerationInheritance";
import { getSamplers, getScheduleTypes, generateImg2Img, generateImg2Vid, Img2VidParams, MiniMaxH3Keyframe, MiniMaxH3References, generateRef2Vid, Ref2VidParams, generateAud2Aud, Aud2AudParams, generateImg2ImgTrainingPreview, toBase64, LoRAConfig, ControlNetConfig, generateTIPOPrompt, cancelGeneration, getCurrentModel, isLatentOnlyResult, getResultFilename, getResultSeed, getResultAncestralSeed, unetQuantizationOptions, normalizeUnetQuantization, transformerQuantizationLabel, archSupportsFeature, videoFrameOptions, videoFrameLabel, archDisplayName, normalizeVideoFrames, fitVideoCanvas, videoCanvasRule, videoCanvasAxisBounds, videoCanvasExceedsEnvelope, isGenerationStalledError } from "@/utils/api";
import { useActiveTraining } from "@/hooks/useActiveTraining";
import { useSmoothProgress } from "@/hooks/useSmoothProgress";
import { wsClient, CFGMetrics } from "@/utils/websocket";
import CFGMetricsGraph from "../common/CFGMetricsGraph";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";
import { previewStorageKeys, loadVideoPreview, saveVideoPreview, loadAudioPreview, saveAudioPreview, saveImagePreview, clearVideoPreview, clearAudioPreview, clearImagePreview, outputExists, stripCacheBuster, withCacheBuster, imagePreviewGone } from "@/utils/previewStorage";
import { sendToPanel, sendImageToImg2Img, sendImageToInpaint, sendImageToUpscale, sendImageToOutpaint, fetchUrlToFile, sendVideoToOutpaint, sendVideoToInpaint, sendVideoToReference, sendAudioToOutpaint, sendAudioToImg2Img } from "@/utils/sendHelpers";
import { useStartup } from "@/contexts/StartupContext";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";
import { createH3ReferenceInventory, maybeTransformH3PromptForGeneration } from "@/utils/h3PromptAssist";
import { readGlobalAttentionType } from "@/utils/attentionSettings";

interface Img2ImgParams {
  prompt: string;
  negative_prompt?: string;
  steps?: number;
  cfg_scale?: number;
  sampler?: string;
  schedule_type?: string;
  seed?: number;
  ancestral_seed?: number;
  width?: number;
  height?: number;
  denoising_strength?: number;
  img2img_fix_steps?: boolean;
  resize_mode?: string;
  resampling_method?: string;
  prompt_chunking_mode?: string;
  max_prompt_chunks?: number;
  loras?: LoRAConfig[];
  controlnets?: ControlNetConfig[];
  // Advanced CFG parameters
  cfg_schedule_type?: string;
  cfg_schedule_min?: number;
  cfg_schedule_max?: number;
  cfg_schedule_power?: number;
  cfg_rescale_snr_alpha?: number;
  dynamic_threshold_percentile?: number;
  dynamic_threshold_mimic_scale?: number;
  // NAG parameters
  nag_enable?: boolean;
  nag_scale?: number;
  nag_tau?: number;
  nag_alpha?: number;
  nag_sigma_end?: number;
  nag_negative_prompt?: string;
  // SDXL micro-conditioning override (inference)
  original_size_w?: number;
  original_size_h?: number;
  original_size_scale?: number;
  // U-Net Quantization
  unet_quantization?: string | null;
  // Quantized GEMM path for already-quantized checkpoints (ideogram4/krea2/anima).
  // null = leave the process-level setting alone.
  quantized_gemm_mode?: "w8a8" | "dequant" | null;
  // Text Encoder Quantization (Z-Image only)
  text_encoder_quantization?: string | null;
  // Attention type
  attention_type?: string;
  // SigLIP2 Vision Encoder path
  vision_encoder_path?: string | null;
  // Component overrides (model-global)
  vae_path?: string | null;
  text_encoder_path?: string | null;
  // PiD (Pixel Diffusion Decoder) options — only relevant when vae_path
  // selects a PiD checkpoint; ignored otherwise.
  pid_sr_output?: string | null;
  pid_use_gemma?: boolean;
  pid_low_vram?: boolean;
  pid_tile_native?: number;
  pid_tile_overlap_ratio?: number;
  pid_fast_large_decode?: boolean;
  // Block swap (model-global)
  enable_block_swap?: boolean;
  blocks_to_swap?: number;
  use_pinned_memory?: boolean;
  block_swap_h2d_only?: boolean;
  block_swap_ring_size?: number;
  // Video generation fields (used when a video model is loaded; the panel maps
  // these into Img2VidParams for img2vid requests, with the input image as the keyframe).
  num_frames?: number;
  frame_rate?: number;
  // OPTIONAL last-frame keyframe, as a data URL. Equivalent to a `keyframes`
  // entry at frame index -1 (MiniMax-H3 `fl2va`); it stays a field of its own
  // because it is the shipped alias the endpoint, the gallery send-to and this
  // panel's own persistence already use. null = no end anchor.
  last_frame_image?: string | null;
  // Keyframe PLACEMENT (MiniMax-H3): which frame the uploaded input image
  // anchors, and any additional anchors with their own frames. -1 means the
  // clip's last frame, resolved server-side after num_frames is snapped.
  input_image_frame_index?: number;
  keyframes?: MiniMaxH3Keyframe[];
  num_inference_steps?: number;
  guidance_scale?: number;
  num_videos_per_prompt?: number;
  audio_enable?: boolean;
  max_sequence_length?: number;
  // Music cover fields (used when an audio model (ACE-Step) is loaded; the panel
  // maps these into Aud2AudParams for aud2aud requests, with the uploaded
  // reference clip as the cover source).
  lyrics?: string;
  inference_steps?: number;
  shift?: number;
  cover_strength?: number;
  vocal_language?: string;
  mode?: "cover" | "repaint";
  repaint_start?: number;
  repaint_end?: number;
  // Loop-generation decode mode (heavy-decoder aware; see loopGenerationInheritance.ts)
  loop_decode?: "full" | "cheap" | "none";
  skip_gallery?: boolean;
  // Start from a server-cached latent instead of an uploaded image (loop
  // passthrough chaining; mutually exclusive with the uploaded image).
  input_latent_id?: string | null;
}

const DEFAULT_PARAMS: Img2ImgParams = {
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
  denoising_strength: 0.75,
  img2img_fix_steps: true,
  vae_drift_correction: false,
  resize_mode: "image",
  resampling_method: "lanczos",
  prompt_chunking_mode: "a1111",
  max_prompt_chunks: 0,
  loras: [],
  controlnets: [],
  cfg_schedule_type: "constant",
  cfg_schedule_min: 1.0,
  cfg_schedule_max: undefined,
  cfg_schedule_power: 2.0,
  cfg_rescale_snr_alpha: 0.0,
  dynamic_threshold_percentile: 0.0,
  dynamic_threshold_mimic_scale: 7.0,
  nag_enable: false,
  unet_quantization: null,
  quantized_gemm_mode: null,
  original_size_w: 0,
  original_size_h: 0,
  original_size_scale: 1.0,
  text_encoder_quantization: null,
  cpu_text_encoding: false,
  nag_scale: 5.0,
  nag_tau: 3.5,
  nag_alpha: 0.25,
  feeling_lucky: false,
  nag_sigma_end: 3.0,
  nag_negative_prompt: "",
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
  // Block swap (model-global; inherited by loop generation). Panel UI pending the Model/Environment section (phase 2).
  enable_block_swap: false,
  blocks_to_swap: 20,
  use_pinned_memory: false,
  block_swap_h2d_only: false,
  block_swap_ring_size: 2,
  // Video generation fields (used when a video model is loaded; the panel maps
  // these into Img2VidParams for img2vid requests, with the input image as the keyframe).
  num_frames: 121,
  frame_rate: 24.0,
  last_frame_image: null,
  input_image_frame_index: 0,
  keyframes: [],
  num_inference_steps: 8,
  guidance_scale: 1.0,
  num_videos_per_prompt: 1,
  audio_enable: true,
  max_sequence_length: 1024,
  // Music cover fields (used when an audio model (ACE-Step) is loaded; the panel
  // maps these into Aud2AudParams for aud2aud requests, with the uploaded
  // reference clip as the cover source). inference_steps/guidance_scale are
  // shared with the video fields above (same defaults, 8 / 1.0).
  lyrics: "",
  shift: 3.0,
  cover_strength: 1.0,
  vocal_language: "en",
  mode: "cover",
  repaint_start: 0,
  repaint_end: 0,
};

// Img2Img's secondary options are grouped into a single-open tabbed accordion
// (see the "Img2Img Options" Card below, shared chrome via
// frontend/src/components/common/TabbedOptions.tsx — ported from
// OutpaintPanel's OUTPAINT_OPTIONS_TABS pattern, same as InpaintPanel). Every
// tab owns a disjoint set of param keys, used both by its "reset to default"
// button and by its active-highlight predicate (isImg2ImgOptionsTabActive
// below). LoRA/ControlNet are left outside the tabs (they're full component
// selectors, not param groups); Sampler/Schedule Type/Steps/CFG Scale/Seed/
// Ancestral Seed/Width/Height stay outside as core fields. Only rendered for
// still-image generation (gated `!isVideo && !isAudio`, matching the
// existing Parameters Card).
type Img2ImgOptionsTabId =
  | "img2img"
  | "cfg"
  | "acceleration"
  | "post_process"
  | "prompt_chunking"
  | "environment";

const IMG2IMG_OPTIONS_TABS: { id: Img2ImgOptionsTabId; label: string }[] = [
  { id: "img2img", label: "Img2Img" },
  { id: "cfg", label: "CFG / NAG" },
  { id: "acceleration", label: "Acceleration（高速化）" },
  { id: "post_process", label: "Post-process（色補正）" },
  { id: "prompt_chunking", label: "Prompt Chunking" },
  { id: "environment", label: "Environment" },
];

const IMG2IMG_OPTIONS_TAB_KEYS: Record<Img2ImgOptionsTabId, (keyof Img2ImgParams)[]> = {
  img2img: [
    "denoising_strength",
    "img2img_fix_steps",
    "resize_mode",
    "resampling_method",
  ],
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
    "vae_drift_correction",
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
// isInpaintOptionsTabActive's rationale.
function isImg2ImgOptionsTabActive(tabId: Img2ImgOptionsTabId, params: Img2ImgParams): boolean {
  switch (tabId) {
    case "img2img":
      return (
        (params.resize_mode ?? "image") !== "image" ||
        (params.resampling_method ?? "lanczos") !== "lanczos" ||
        !(params.img2img_fix_steps ?? true)
      );
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
        !!params.flatten_in_loop ||
        !!params.vae_drift_correction
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

// The valid clip lengths differ per video architecture (LTX-2.3: 8k+1;
// MiniMax-H3: 17n+5 within 124-345), so the option list comes from the
// backend's own `video_constraints` payload via videoFrameOptions() below
// rather than from a list kept here. See frontend/src/utils/api.ts.

const STORAGE_KEY = "img2img_params";
const LOOP_GENERATION_STORAGE_KEY = "img2img_loop_generation";
const PREVIEW_STORAGE_KEY = "img2img_preview";
// Image + video + audio preview keys for this panel. The three are mutually
// exclusive in storage (see utils/previewStorage.ts), so the newest result is
// the only one that can be restored.
const PREVIEW_KEYS = previewStorageKeys(PREVIEW_STORAGE_KEY);
const INPUT_IMAGE_STORAGE_KEY = "img2img_input_image";
const REF_IMAGES_STORAGE_KEY = "img2img_ref_images";
// The optional last-frame keyframe (a data URL) gets its OWN key, exactly like
// the input image above: it is an IMAGE, so it must not ride the params blob
// into the ~5 MB localStorage quota -- but keeping it out of that blob without
// storing it anywhere made it the one parameter that silently vanished on
// reload while every other one came back.
const LAST_FRAME_STORAGE_KEY = "img2img_last_frame_image";

/**
 * A conditioning image OTHER than the uploaded input image, addressed by the
 * slot it lives in. `keyframe` is `params.keyframes[index]`; `last` is the
 * `last_frame_image` alias (an anchor at the clip's last frame). The input
 * image itself is not an ExtraAnchor: it has its own File/preview/temp-storage
 * plumbing and keeps it.
 */
type ExtraAnchor = { kind: "keyframe"; index: number } | { kind: "last" };

interface Img2ImgPanelProps {
  onImageGenerated?: (imageUrl: string) => void;
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint" | "outpaint" | "upscale") => void;
}

export default function Img2ImgPanel({ onTabChange, onImageGenerated }: Img2ImgPanelProps = {}) {
  const { modelLoaded, isBackendReady, generationDefaults, isVideo, isAudio, archCapabilities, resolveModality, modelInfoVersion } = useStartup();
  const [params, setParams] = useState<Img2ImgParams>(DEFAULT_PARAMS);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  // Video output (produced when a video model is loaded / img2vid queue item).
  const [generatedVideo, setGeneratedVideo] = useState<string | null>(null);
  const [generatedVideoInfo, setGeneratedVideoInfo] = useState<{ num_frames?: number; fps?: number; duration?: number } | null>(null);
  // Seed the last video result actually ran with, so the video card's seed
  // control has the same "reuse the seed from the preview" button the image
  // path has (StoredVideoPreview carries it, as it does in OutpaintPanel).
  const [generatedVideoSeed, setGeneratedVideoSeed] = useState<number | null>(null);
  const [generatedVideoParams, setGeneratedVideoParams] = useState<Img2ImgParams | null>(null);
  // Audio output (produced when an audio model is loaded / aud2aud queue item).
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [generatedAudioInfo, setGeneratedAudioInfo] = useState<{ duration?: number; sample_rate?: number } | null>(null);
  const [generatedAudioParams, setGeneratedAudioParams] = useState<Img2ImgParams | null>(null);
  // Reference audio clip (the aud2aud "input image" equivalent) -- kept as a
  // File (not base64) so it can carry through the queue as `inputAudio`.
  const [referenceAudioFile, setReferenceAudioFile] = useState<File | null>(null);
  const [referenceAudioPreview, setReferenceAudioPreview] = useState<string | null>(null);
  // Revoke the reference-audio blob URL on unmount (mirrors previewBlobUrlRef cleanup).
  useEffect(() => {
    return () => {
      if (referenceAudioPreview) URL.revokeObjectURL(referenceAudioPreview);
    };
  }, [referenceAudioPreview]);
  // Client-side post-edit (brightness/saturation) for the current preview image.
  // Never sent to the backend; reset to neutral on each new generated image.
  const [postEdit, setPostEdit] = useState<PostEditState>({ ...NEUTRAL_POST_EDIT });
  // Color-flatten preview for the inline result image (b/s stay as CSS filter).
  const effectiveGeneratedImage = usePostEditPreview(generatedImage, postEdit.flatten);
  useEffect(() => {
    setPostEdit({ ...NEUTRAL_POST_EDIT });
  }, [generatedImage]);
  const [generatedImageSeed, setGeneratedImageSeed] = useState<number | null>(null);
  const [generatedImageAncestralSeed, setGeneratedImageAncestralSeed] = useState<number | null>(null);
  const [generatedImageParams, setGeneratedImageParams] = useState<Img2ImgParams | null>(null);
  const [inputImage, setInputImage] = useState<File | null>(null);
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null);
  const [inputImageSize, setInputImageSize] = useState<{ width: number; height: number } | null>(null);
  const [sizeMode, setSizeMode] = useState<"absolute" | "scale">("absolute");
  const [scale, setScale] = useState<number>(1.0);
  // Grid a scale-derived resolution is rounded onto on the IMAGE path (a video
  // model goes through deriveScaledSize below instead, because its canvas has
  // an architecture-specific alignment and envelope rather than one constant).
  const sizeSnap = 64;
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
  const [currentModelInfo, setCurrentModelInfo] = useState<any>(null);
  // Keep this panel's copy of GET /models/current in step with the shared one.
  // modelInfoVersion only changes when the loaded model's identity actually
  // changes, so this costs one request per model change -- including changes
  // this page did not make (API, backend restart, another tab).
  useEffect(() => {
    if (modelInfoVersion === 0) return; // initial fetch is done in loadInitialData
    getCurrentModel()
      .then(setCurrentModelInfo)
      .catch((error) => console.warn("[Img2Img] Failed to refresh model info", error));
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
  // The loaded architecture and the capability gates the VIDEO controls read.
  // `archSupportsFeature` treats an unknown arch (or a capability matrix that
  // has not loaded) as supporting the feature, so a control is never hidden
  // merely because the matrix was unavailable.
  const loadedArch = currentModelInfo?.model_info?.type as string | undefined;
  const loadedArchName = archDisplayName(loadedArch);
  const supportsCfg = archSupportsFeature(archCapabilities, loadedArch, "cfg");
  const supportsNegativePrompt = !isAudio
    && archSupportsFeature(archCapabilities, loadedArch, "negative_prompt");
  // Spectrum/FBCache: accepted-but-inert on an architecture whose sampler never
  // reads spectrum_enable/fbcache_enable. Hidden rather than shown-disabled,
  // the same convention as supportsCfg/supportsNegativePrompt above.
  const supportsSpectrum = archSupportsFeature(archCapabilities, loadedArch, "spectrum");
  const supportsFbcache = archSupportsFeature(archCapabilities, loadedArch, "fbcache");
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
  const supportsLastFrame = archSupportsFeature(archCapabilities, loadedArch, "last_frame_image");
  // Keyframe PLACEMENT is a separate capability from "there is a last-frame
  // slot": an architecture can have the second without the first. When it is
  // supported the timeline SUBSUMES the last-frame box (it renders that anchor
  // as its own chip), so exactly one of the two controls is shown.
  const supportsKeyframePlacement = archSupportsFeature(
    archCapabilities, loadedArch, "keyframe_placement");
  // ia2v: an uploaded track the video is generated against (MiniMax-H3). A
  // third, independent capability -- an architecture can place image keyframes
  // without being able to read an audio track at all.
  const supportsAudioConditioning = archSupportsFeature(
    archCapabilities, loadedArch, "audio_conditioning");
  // MiniMax-H3 ships two transformer partitions (see Txt2ImgPanel's isRef2Va):
  // `fl2va` serves this panel's img2vid keyframe path, `ref2va` was trained to
  // read reference rows instead. Direct variant check, matching Txt2ImgPanel --
  // there is no per-variant capability key, since keyframe_placement/
  // audio_conditioning are true for minimax_h3 regardless of which is loaded.
  const isRef2Va =
    loadedArch === "minimax_h3" &&
    (currentModelInfo?.model_info?.variant as string | undefined) === "ref2va";
  // The input card takes MORE THAN ONE image exactly where the loaded
  // architecture has somewhere to put a second one -- keyframe placement, or
  // the last-frame slot on its own. On an image model, and on a video model
  // that conditions on the first frame only (LTX-2.3), it stays the
  // single-image card it has always been rather than growing a tab strip with
  // one tab in it.
  const multiImageInput = isVideo && (supportsKeyframePlacement || supportsLastFrame);

  /**
   * Width/height for "the input image at Nx".
   *
   * On the image path this is the historical round-to-64. On a VIDEO model the
   * canvas is not free: both axes align to the architecture's `pixel_align`
   * and it may cap the envelope (MiniMax-H3: short edge 768, long edge 1344),
   * so the size is resolved by `fitVideoCanvas` from the capability matrix.
   * Without that, Scale 4x on a 1024x1024 image would send a 4096x4096 canvas
   * the backend refuses.
   */
  const deriveScaledSize = (srcWidth: number, srcHeight: number, scaleValue: number) => {
    if (isVideo) {
      const fitted = fitVideoCanvas(archCapabilities, loadedArch, srcWidth, srcHeight, scaleValue);
      return { width: fitted.width, height: fitted.height };
    }
    return {
      width: Math.round(srcWidth * scaleValue / sizeSnap) * sizeSnap,
      height: Math.round(srcHeight * scaleValue / sizeSnap) * sizeSnap,
    };
  };
  // The track itself, as a File and NOT in `params`: it is an upload, so it
  // rides on the queue item the way aud2aud's reference clip and outpaint_vid's
  // source clip do, and it never reaches the persisted params blob.
  const [inputAudioTrack, setInputAudioTrack] = useState<File | null>(null);
  // MiniMax-H3 ref2va references, mirroring Txt2ImgPanel's h3References: file
  // uploads, kept out of `params` and carried on the queue item (QueueItem.references)
  // so a queued request keeps the references it was built with.
  const [h3References, setH3References] = useState<MiniMaxH3References>(
    EMPTY_MINIMAX_H3_REFERENCES
  );
  const [h3ReferenceImageSize, setH3ReferenceImageSize] = useState<"max" | "match">("max");
  const [isDragging, setIsDragging] = useState(false);
  const [isEditingImage, setIsEditingImage] = useState(false);
  // ── Multi-image (tabbed) input, for a video architecture that takes more
  // than one conditioning image. `input` is the uploaded input image (the
  // img2vid keyframe); the other tabs are the extra anchors, which are the
  // same two storage slots the timeline draws: `params.keyframes[i]` and the
  // `last_frame_image` alias. On an image model none of this renders and the
  // card is the single-image one it has always been.
  const [activeInputTab, setActiveInputTab] = useState<string>("input");
  // The extra anchor currently open in the paint editor. The input image keeps
  // its own `isEditingImage` flag (untouched image-model path); the two are
  // mutually exclusive because only one tab is on screen at a time.
  const [editingExtraAnchor, setEditingExtraAnchor] = useState<ExtraAnchor | null>(null);
  const addAnchorInputRef = useRef<HTMLInputElement>(null);
  const [sendImage, setSendImage] = useState(true);
  const [sendPrompt, setSendPrompt] = useState(true);
  const [sendParameters, setSendParameters] = useState(true);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  // ── Training-preview integration (mirrors Txt2ImgPanel) ──────────────
  const [useTrainingModel, setUseTrainingModel] = useState(false);
  const [savePreviewToGallery, setSavePreviewToGallery] = useState(false);
  const activeTraining = useActiveTraining();
  const previewBlobUrlRef = useRef<string | null>(null);
  useEffect(() => {
    if (!activeTraining && useTrainingModel) setUseTrainingModel(false);
  }, [activeTraining, useTrainingModel]);
  useEffect(() => {
    return () => {
      if (previewBlobUrlRef.current) {
        URL.revokeObjectURL(previewBlobUrlRef.current);
        previewBlobUrlRef.current = null;
      }
    };
  }, []);
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
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  const pathname = usePathname();
  const searchParams = useSearchParams();

  const promptTextareaRef = useRef<HTMLTextAreaElement | null>(null);

  // TIPO: Treat as Natural Language (local state, not persisted)
  const [treatAsNL, setTreatAsNL] = useState(false);

  // Use refs for WebSocket callback to prevent recreations
  const isGeneratingRef = useRef(isGenerating);
  const developerModeRef = useRef(developerMode);

  useEffect(() => {
    isGeneratingRef.current = isGenerating;
  }, [isGenerating]);

  useEffect(() => {
    developerModeRef.current = developerMode;
  }, [developerMode]);

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

  // Load from localStorage after component mounts (client-side only)
  useEffect(() => {
    // console.clear(); // Temporarily disabled for debugging
    console.log("=== Img2ImgPanel mounted ===");
    setIsMounted(true);

    const loadInitialData = async () => {
      // Load current model info
      try {
        const modelInfo = await getCurrentModel();
        setCurrentModelInfo(modelInfo);
        console.log("[Img2Img] Current model info:", modelInfo);
      } catch (error) {
        console.error("Failed to load model info:", error);
      }

      // Load params
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          // Fix floating point precision issues
          const fixed = fixFloatingPointParams(merged);
          setParams(fixed);
        } catch (error) {
          console.error("Failed to load saved params:", error);
        }
      }

      // The last-frame keyframe lives under its own key (it is an image, so it
      // is kept out of the params blob), and is restored whether or not any
      // params were persisted.
      const savedLastFrame = localStorage.getItem(LAST_FRAME_STORAGE_KEY);
      if (savedLastFrame) {
        setParams((prev) => ({ ...prev, last_frame_image: savedLastFrame }));
      }

      // Load preview image
      const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
      if (savedPreview) {
        setGeneratedImage(savedPreview);
      }

      // Load preview video (img2vid result). Restored unconditionally: the
      // player is gated on `isVideo`, which arrives asynchronously from
      // useStartup(), so nothing renders until the loaded arch is known to be a
      // video arch. The URL is verified once the backend is ready (below).
      const savedVideo = loadVideoPreview(PREVIEW_KEYS);
      if (savedVideo) {
        setGeneratedVideo(savedVideo.url);
        setGeneratedVideoInfo(savedVideo.info);
        setGeneratedVideoSeed(savedVideo.seed ?? null);
      }

      // Load preview audio (aud2aud result). Same reasoning as the video above:
      // restored unconditionally because the <audio> render site is gated on
      // `isAudio` from useStartup(), which arrives asynchronously, so nothing
      // plays until the loaded arch is known to be an audio arch.
      const savedAudio = loadAudioPreview(PREVIEW_KEYS);
      if (savedAudio) {
        setGeneratedAudio(savedAudio.url);
        setGeneratedAudioInfo(savedAudio.info);
      }

      // Load input image preview
      const savedInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      console.log("[Img2Img] Initial load - input image ref:", savedInputRef);
      if (savedInputRef) {
        // NOTE: Allow old-style references (direct URLs) for now
        // // Check if it's an old-style reference (direct URL like /outputs/... or http://...)
        // if (savedInputRef.startsWith('/outputs/') || savedInputRef.startsWith('http://') || savedInputRef.startsWith('https://')) {
        //   console.log("[Img2Img] Detected old-style input image reference, clearing storage");
        //   localStorage.removeItem(INPUT_IMAGE_STORAGE_KEY);
        // } else if (savedInputRef.startsWith('temp_img://') || savedInputRef.startsWith('data:')) {
        try {
          const imageData = await loadTempImage(savedInputRef);
          console.log("[Img2Img] Input image loaded successfully:", imageData ? "yes" : "no");
          if (imageData) {
            setInputImagePreview(imageData);
            // Load image dimensions
            const img = new Image();
            img.onload = () => {
              console.log("[Img2Img] Input image dimensions set:", img.width, "x", img.height);
              setInputImageSize({ width: img.width, height: img.height });
            };
            img.src = imageData;
          }
          // } else {
          //   console.warn("[Img2Img] Invalid input image data, clearing storage");
          //   localStorage.removeItem(INPUT_IMAGE_STORAGE_KEY);
          // }
        } catch (error) {
          console.error("[Img2Img] Failed to load input image:", error);
        }
        // } else {
        //   console.warn("[Img2Img] Unknown input image reference format, clearing storage");
        //   localStorage.removeItem(INPUT_IMAGE_STORAGE_KEY);
        // }
      }

      // Load resolution step and aspect ratio presets settings
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
      const savedVisibility = localStorage.getItem('img2img_visibility');
      if (savedVisibility) {
        try {
          setVisibility(JSON.parse(savedVisibility));
        } catch (e) {
          console.error('Failed to parse img2img visibility:', e);
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
      const savedRefImageRefs = localStorage.getItem(REF_IMAGES_STORAGE_KEY);
      if (savedRefImageRefs) {
        try {
          const refRefs: string[] = JSON.parse(savedRefImageRefs);
          console.log(`[Img2Img] Loading ${refRefs.length} reference images from storage`);

          const loadedPreviews: string[] = [];
          for (const ref of refRefs) {
            try {
              const imageData = await loadTempImage(ref);
              if (imageData) {
                loadedPreviews.push(imageData);
              }
            } catch (error) {
              console.error(`[Img2Img] Failed to load reference image ${ref}:`, error);
            }
          }

          if (loadedPreviews.length > 0) {
            setRefImagePreviews(loadedPreviews);
            console.log(`[Img2Img] Restored ${loadedPreviews.length} reference images`);
          }
        } catch (error) {
          console.error('[Img2Img] Failed to parse reference images storage:', error);
        }
      }

      // Mark initial load as complete
      setIsInitialLoad(false);
      console.log("[Img2Img] Initial load complete");
    };

    loadInitialData();
  }, []);

  // Reload images when backend becomes ready
  useEffect(() => {
    if (!isBackendReady) return;

    const reloadImages = async () => {
      console.log("[Img2Img] Backend ready, reloading images if needed");

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
        const previewStillThere = await outputExists(previewPath);
        if (!previewStillThere) {
          console.log("[Img2Img] Stored preview image is gone, clearing:", previewPath);
          clearImagePreview(PREVIEW_KEYS);
          setGeneratedImage(null);
        } else {
          console.log("[Img2Img] Reloading preview image from backend:", previewPath);
          setGeneratedImage(withCacheBuster(previewPath));
        }
      }

      // Verify the restored preview video still exists (outputs/ can be
      // cleared, or the run deleted from the gallery). No cache-busting
      // timestamp -- an .mp4 is large and its URL is stable.
      const savedVideo = loadVideoPreview(PREVIEW_KEYS);
      if (savedVideo) {
        const exists = await outputExists(savedVideo.url);
        if (!exists) {
          console.log("[Img2Img] Stored preview video is gone, clearing:", savedVideo.url);
          clearVideoPreview(PREVIEW_KEYS);
          setGeneratedVideo(null);
          setGeneratedVideoInfo(null);
          setGeneratedVideoSeed(null);
        }
      }

      // Same verification for a restored preview audio clip.
      const savedAudio = loadAudioPreview(PREVIEW_KEYS);
      if (savedAudio) {
        const exists = await outputExists(savedAudio.url);
        if (!exists) {
          console.log("[Img2Img] Stored preview audio is gone, clearing:", savedAudio.url);
          clearAudioPreview(PREVIEW_KEYS);
          setGeneratedAudio(null);
          setGeneratedAudioInfo(null);
        }
      }

      // Reload input image if needed
      const savedInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (savedInputRef) {
        try {
          const imageData = await loadTempImage(savedInputRef);
          if (imageData) {
            setInputImagePreview(imageData);
            // Update dimensions
            const img = new Image();
            img.onload = () => {
              setInputImageSize({ width: img.width, height: img.height });
            };
            img.src = imageData;
          }
        } catch (error) {
          console.error("[Img2Img] Failed to reload input image:", error);
        }
      }
    };

    reloadImages();
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

  // When backend becomes ready, reload temp image if not already loaded
  useEffect(() => {
    if (isBackendReady && !inputImagePreview) {
      const reloadImage = async () => {
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
            console.error("[Img2Img] Failed to reload input image after backend ready:", error);
          }
        }
      };

      reloadImage();
    }
  }, [isBackendReady]);

  useEffect(() => {
    // Listen for input image updates from txt2img or gallery
    const handleInputUpdate = async () => {
      const newInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (newInputRef) {
        try {
          const imageData = await loadTempImage(newInputRef);
          if (imageData) {
            setInputImagePreview(imageData);
            // Load image dimensions
            const img = new Image();
            img.onload = () => {
              setInputImageSize({ width: img.width, height: img.height });

              // Apply global send size mode settings
              const sendSizeMode = localStorage.getItem('send_size_mode') as "absolute" | "scale" | null;
              if (sendSizeMode === 'scale') {
                setSizeMode('scale');
                const sendDefaultScale = parseFloat(localStorage.getItem('send_default_scale') || '1.0');
                setScale(sendDefaultScale);
                // Update dimensions based on scale
                const scaledWidth = Math.round(img.width * sendDefaultScale / 64) * 64;
                const scaledHeight = Math.round(img.height * sendDefaultScale / 64) * 64;
                setParams(prev => ({ ...prev, width: scaledWidth, height: scaledHeight }));
              } else {
                // Absolute mode - use image dimensions as-is
                setSizeMode('absolute');
                setScale(1.0);
                setParams(prev => ({ ...prev, width: img.width, height: img.height }));
              }
            };
            img.src = imageData;
          }
        } catch (error) {
          console.error("Failed to load input image:", error);
        }
      }
    };

    window.addEventListener("img2img_input_updated", handleInputUpdate);

    return () => {
      window.removeEventListener("img2img_input_updated", handleInputUpdate);
    };
  }, []);

  // Also accept keyframes sent to the (now-merged) img2vid target: the gallery
  // frame-grab / send-to-img2vid writes to "img2vid_input_image" and dispatches
  // "img2vid_input_updated". Load it as the input image so a video model can use it.
  useEffect(() => {
    const handleVidInputUpdate = async () => {
      const newInputRef = localStorage.getItem("img2vid_input_image");
      if (!newInputRef) return;
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
        console.error("[Img2Img] Failed to load img2vid keyframe:", error);
      }
    };

    window.addEventListener("img2vid_input_updated", handleVidInputUpdate);
    return () => {
      window.removeEventListener("img2vid_input_updated", handleVidInputUpdate);
    };
  }, []);

  // Accept an audio reference clip sent from a result's "Send to Img2Img"
  // (aud2aud reference, e.g. Txt2Img/Outpaint's generatedAudio). Transport is
  // the plain `/outputs/<filename>` URL (too large for base64/localStorage) --
  // fetch it into a real File so it flows through the same referenceAudioFile
  // path an upload does (mirrors handleReferenceAudioUpload).
  useEffect(() => {
    const handleAudioInputUpdate = async () => {
      const url = localStorage.getItem("img2img_input_audio");
      if (!url) return;
      try {
        const file = await fetchUrlToFile(url);
        setReferenceAudioPreview(prev => {
          if (prev) URL.revokeObjectURL(prev);
          return URL.createObjectURL(file);
        });
        setReferenceAudioFile(file);
      } catch (error) {
        console.error("[Img2Img] Failed to load sent audio:", error);
      } finally {
        localStorage.removeItem("img2img_input_audio");
      }
    };
    window.addEventListener("img2img_input_audio_updated", handleAudioInputUpdate);
    return () => {
      window.removeEventListener("img2img_input_audio_updated", handleAudioInputUpdate);
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
            console.warn("[Img2Img] Reference video not added: track is full");
            return prev;
          }
          return { ...prev, videos: [...prev.videos, file], videoAudios: [...prev.videoAudios, null] };
        });
      } catch (error) {
        console.error("[Img2Img] Failed to load sent reference video:", error);
      } finally {
        localStorage.removeItem("h3_reference_video");
      }
    };
    window.addEventListener("h3_reference_video_updated", handleReferenceVideoUpdate);
    return () => window.removeEventListener("h3_reference_video_updated", handleReferenceVideoUpdate);
  }, []);

  // Save params to localStorage whenever they change (but only after mounted and initial load complete)
  useEffect(() => {
    if (isMounted && !isInitialLoad) {
      // Only save if params are different from what's in localStorage
      // This prevents overwriting params sent from Gallery/other panels
      const saved = localStorage.getItem(STORAGE_KEY);
      const savedParams = saved ? JSON.parse(saved) : null;
      // `last_frame_image` is a data URL, i.e. the whole IMAGE. It is kept out
      // of the persisted copy for the same reason the input image has its own
      // key (INPUT_IMAGE_STORAGE_KEY): a base64 image in the params blob eats
      // the ~5 MB localStorage quota and, once it overflows, takes every OTHER
      // persisted parameter down with it.
      // `keyframes` carries data URLs too, for the same quota reason. It is
      // dropped rather than given its own key: an arbitrary number of full
      // images is not something to push into a 5 MB store.
      const { last_frame_image: _lastFrame, keyframes: _keyframes, ...persistableParams } = params;
      const currentParamsStr = JSON.stringify(persistableParams);
      const savedParamsStr = savedParams ? JSON.stringify(savedParams) : null;

      if (currentParamsStr !== savedParamsStr) {
        console.log("[Img2Img] Params changed by user, saving to localStorage:", {
          loras: params.loras?.length || 0,
          controlnets: params.controlnets?.length || 0,
          prompt_length: params.prompt?.length || 0,
        });
        localStorage.setItem(STORAGE_KEY, currentParamsStr);
      }
    }
  }, [params, isMounted, isInitialLoad]);

  // The last-frame keyframe, under its own key (see LAST_FRAME_STORAGE_KEY).
  // Written best-effort: a large upload can exceed the quota, and when it does
  // only this one value is lost instead of every persisted parameter.
  useEffect(() => {
    if (!isMounted || isInitialLoad) return;
    try {
      if (params.last_frame_image) {
        localStorage.setItem(LAST_FRAME_STORAGE_KEY, params.last_frame_image);
      } else {
        localStorage.removeItem(LAST_FRAME_STORAGE_KEY);
      }
    } catch (error) {
      console.warn("[Img2Img] Could not persist the last-frame keyframe:", error);
    }
  }, [params.last_frame_image, isMounted, isInitialLoad]);

  // Listen for localStorage changes from Gallery/Preview (send to feature)
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && e.newValue) {
        try {
          const parsed = JSON.parse(e.newValue);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          const fixed = fixFloatingPointParams(merged);
          setParams(fixed);
          console.log("[Img2Img] Params updated from storage event (cross-tab)");
        } catch (error) {
          console.error("[Img2Img] Failed to parse storage change:", error);
        }
      }
    };

    const handleCustomStorageChange = () => {
      const saved = localStorage.getItem(STORAGE_KEY);
      console.log("[Img2Img] handleCustomStorageChange - saved:", saved);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          console.log("[Img2Img] handleCustomStorageChange - parsed:", parsed);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          const fixed = fixFloatingPointParams(merged);
          console.log("[Img2Img] handleCustomStorageChange - merged prompt:", merged.prompt);
          setParams(fixed);
          console.log("[Img2Img] Params updated from custom storage event (same-tab)");
        } catch (error) {
          console.error("[Img2Img] Failed to parse custom storage change:", error);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('img2img_params_updated', handleCustomStorageChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('img2img_params_updated', handleCustomStorageChange);
    };
  }, []);

  // Reload params from localStorage when navigating to /generate?tab=img2img (from Gallery)
  useEffect(() => {
    if (pathname === "/generate" && searchParams.get('tab') === 'img2img' && isMounted) {
      console.log("[Img2Img] Page navigated to img2img tab, reloading params from localStorage");
      const saved = localStorage.getItem(STORAGE_KEY);
      console.log("[Img2Img] Navigation reload - saved:", saved);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          console.log("[Img2Img] Navigation reload - parsed:", parsed);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          const fixed = fixFloatingPointParams(merged);
          console.log("[Img2Img] Navigation reload - merged prompt:", merged.prompt);
          setParams(fixed);
          console.log("[Img2Img] Params reloaded:", {
            prompt_length: fixed.prompt?.length || 0,
            prompt: fixed.prompt,
            steps: fixed.steps,
            cfg_scale: fixed.cfg_scale,
          });
        } catch (error) {
          console.error("[Img2Img] Failed to reload params on navigation:", error);
        }
      }
    }
  }, [pathname, searchParams, isMounted]);

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

  // Apply backend-fetched defaults when they arrive (only if no localStorage value exists).
  // aud2aud defaults are merged on top so the audio fields (lyrics, cover_strength,
  // inference_steps, shift, guidance_scale, vocal_language) reflect param_defaults.py
  // even though this panel's primary shape is img2img.
  useEffect(() => {
    if (!generationDefaults) return;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      setParams(prev => ({
        ...DEFAULT_PARAMS,
        ...(generationDefaults.img2img as Partial<typeof DEFAULT_PARAMS>),
        ...(generationDefaults.aud2aud as Partial<typeof DEFAULT_PARAMS>),
      }));
    }
  }, [generationDefaults]);

  const resetToDefault = () => {
    setParams(DEFAULT_PARAMS);
    localStorage.removeItem(STORAGE_KEY);
  };

  const loadSamplers = async () => {
    try {
      const data = await getSamplers();
      setSamplers(data.samplers);
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
      const data = await getScheduleTypes();
      setScheduleTypes(data.schedule_types);
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
          console.error("Failed to save temp image:", error);
        }
      }

      // Load image to get dimensions
      const img = new Image();
      img.onload = () => {
        setInputImageSize({ width: img.width, height: img.height });
        // If in scale mode, update width/height based on scale
        if (sizeMode === "scale") {
          const scaled = deriveScaledSize(img.width, img.height, scale);
          setParams({ ...params, width: scaled.width, height: scaled.height });
        }
      };
      img.src = preview;
    };
    reader.readAsDataURL(file);
  };

  // NOTE: the input card's file-picker and drop handlers now live in
  // renderImageDropZone (one per tab) and call processImageFile / the anchor
  // setters directly, so the old single-image handleImageUpload/handleDrop
  // wrappers are gone. handleDragOver/handleDragLeave are still shared.
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

  const handleScaleChange =(newScale: number) => {
    setScale(newScale);
    if (inputImageSize && sizeMode === "scale") {
      const scaled = deriveScaledSize(inputImageSize.width, inputImageSize.height, newScale);
      setParams({ ...params, width: scaled.width, height: scaled.height });
    }
  };

  const handleSizeModeChange = (newMode: "absolute" | "scale") => {
    setSizeMode(newMode);
    if (newMode === "scale" && inputImageSize) {
      // Switch to scale mode - update dimensions based on current scale
      const scaled = deriveScaledSize(inputImageSize.width, inputImageSize.height, scale);
      setParams({ ...params, width: scaled.width, height: scaled.height });
    }
  };

  const handleEditImage = () => {
    if (inputImagePreview) {
      setIsEditingImage(true);
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

  // ── Extra conditioning images (video architectures) ──────────────────────
  // The tabbed INPUT IMAGES card edits the same two slots the keyframe
  // timeline draws, so both controls stay in sync by construction: there is no
  // third copy of an image anywhere. Data URLs live in `params.keyframes` /
  // `params.last_frame_image`; `keyframes` is deliberately excluded from the
  // persisted params blob (see the save effect) so image bytes never enter the
  // ~5 MB localStorage quota.
  const anchorImage = (anchor: ExtraAnchor): string | null => {
    if (anchor.kind === "last") return params.last_frame_image ?? null;
    const image = params.keyframes?.[anchor.index]?.image;
    return typeof image === "string" ? image : null;
  };

  const setAnchorImage = (anchor: ExtraAnchor, dataUrl: string) => {
    if (anchor.kind === "last") {
      setParams((prev) => ({ ...prev, last_frame_image: dataUrl }));
      return;
    }
    setParams((prev) => ({
      ...prev,
      keyframes: (prev.keyframes ?? []).map((keyframe, index) =>
        index === anchor.index ? { ...keyframe, image: dataUrl } : keyframe,
      ),
    }));
  };

  const removeAnchor = (anchor: ExtraAnchor) => {
    if (anchor.kind === "last") {
      setParams((prev) => ({ ...prev, last_frame_image: null }));
    } else {
      setParams((prev) => ({
        ...prev,
        keyframes: (prev.keyframes ?? []).filter((_k, index) => index !== anchor.index),
      }));
    }
    setActiveInputTab("input");
  };

  /**
   * Add an image as a new anchor and open its tab.
   *
   * On an architecture with keyframe PLACEMENT it becomes a `keyframes` entry
   * on a free frame near the middle (the same free-frame search the timeline's
   * own "Add keyframe" does, so two adds never collide); on one that only has
   * the last-frame slot it fills that slot instead, because a `keyframes`
   * entry would be accepted and dropped there.
   */
  const addAnchorImage = (dataUrl: string) => {
    const lastIndex = Math.max(0, (params.num_frames ?? 124) - 1);
    if (!supportsKeyframePlacement) {
      setParams((prev) => ({ ...prev, last_frame_image: dataUrl }));
      setActiveInputTab("last");
      return;
    }
    const resolve = (requested: number) => (requested === -1 ? lastIndex : requested);
    const taken = new Set<number>([
      resolve(params.input_image_frame_index ?? 0),
      ...(params.keyframes ?? []).map((keyframe) => resolve(keyframe.frame_index)),
      ...(params.last_frame_image ? [lastIndex] : []),
    ]);
    let frame = Math.floor(lastIndex / 2);
    while (frame < lastIndex && taken.has(frame)) frame += 1;
    while (frame > 0 && taken.has(frame)) frame -= 1;
    const nextIndex = (params.keyframes ?? []).length;
    setParams((prev) => ({
      ...prev,
      keyframes: [...(prev.keyframes ?? []), { image: dataUrl, frame_index: frame }],
    }));
    setActiveInputTab(`kf-${nextIndex}`);
  };

  // Reference audio clip (aud2aud cover source). Kept in-memory only (a blob
  // URL preview + the File itself); not persisted across reloads like
  // inputImage/tempImageStorage -- audio models are a later phase and don't
  // need loop-generation/reload survival yet.
  const handleReferenceAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (referenceAudioPreview) URL.revokeObjectURL(referenceAudioPreview);
      setReferenceAudioFile(file);
      setReferenceAudioPreview(URL.createObjectURL(file));
    }
  };

  const handleClearReferenceAudio = () => {
    if (referenceAudioPreview) URL.revokeObjectURL(referenceAudioPreview);
    setReferenceAudioFile(null);
    setReferenceAudioPreview(null);
  };

  const handleSaveEditedImage = async (editedImageUrl: string) => {
    setInputImagePreview(editedImageUrl);
    if (isMounted) {
      try {
        // Delete old reference and save new one
        const oldRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
        if (oldRef) {
          await deleteTempImageRef(oldRef);
        }
        const ref = await saveTempImage(editedImageUrl);
        localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, ref);
      } catch (error) {
        console.error("Failed to save edited image:", error);
      }
    }

    // Update image dimensions
    const img = new Image();
    img.onload = () => {
      setInputImageSize({ width: img.width, height: img.height });
    };
    img.src = editedImageUrl;

    setIsEditingImage(false);
    setInputImage(null); // Clear File object, use data URL instead
  };

  const sendToTxt2Img = () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }

    // Note: Send image is not applicable for txt2img (no input image)

    // Send prompt if checked
    if (sendPrompt) {
      const txt2imgParams = JSON.parse(localStorage.getItem("txt2img_params") || "{}");
      txt2imgParams.prompt = params.prompt;
      txt2imgParams.negative_prompt = params.negative_prompt;
      localStorage.setItem("txt2img_params", JSON.stringify(txt2imgParams));
    }

    // Send parameters if checked
    if (sendParameters) {
      const txt2imgParams = JSON.parse(localStorage.getItem("txt2img_params") || "{}");
      txt2imgParams.steps = params.steps;
      txt2imgParams.cfg_scale = params.cfg_scale;
      txt2imgParams.sampler = params.sampler;
      txt2imgParams.schedule_type = params.schedule_type;
      txt2imgParams.seed = params.seed;
      txt2imgParams.width = params.width;
      txt2imgParams.height = params.height;
      localStorage.setItem("txt2img_params", JSON.stringify(txt2imgParams));
    }

    // Navigate to txt2img tab
    if (onTabChange) {
      onTabChange("txt2img");
    }
  };

  const sendToImg2Img = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }

    // Use generated image params if available, otherwise fall back to current UI params
    const sourceParams = generatedImageParams || params;

    // Send image if checked - already in img2img, use generated image as new input
    if (sendImage) {
      try {
        await sendImageToImg2Img(generatedImage, INPUT_IMAGE_STORAGE_KEY);
        setInputImagePreview(generatedImage);
      } catch (error) {
        console.error("Failed to send image to img2img:", error);
      }
    }

    console.log("[Img2Img] sendToTxt2Img - sendPrompt:", sendPrompt, "sendParameters:", sendParameters);
    console.log("[Img2Img] sendToTxt2Img - sourceParams.prompt:", sourceParams.prompt);

    // Send prompt and/or parameters
    sendToPanel(sourceParams, STORAGE_KEY, {
      sendPrompt,
      sendParameters,
      includeDenoising: true,
      dispatchEvent: "txt2img_params_updated"
    });

    console.log("[Img2Img] sendToTxt2Img - Sent to panel");
  };

  const sendToUpscale = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }

    try {
      await sendImageToUpscale(generatedImage);
    } catch (error) {
      console.error("[Img2Img] Failed to send image to upscale:", error);
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

    // Use generated image params if available, otherwise fall back to current UI params
    const sourceParams = generatedImageParams || params;

    // Send image if checked
    if (sendImage) {
      try {
        await sendImageToInpaint(generatedImage);
      } catch (error) {
        console.error("Failed to send image to inpaint:", error);
      }
    }

    console.log("[Img2Img] sendToInpaint - sendPrompt:", sendPrompt, "sendParameters:", sendParameters);
    console.log("[Img2Img] sendToInpaint - sourceParams.prompt:", sourceParams.prompt);

    // Send prompt and/or parameters
    sendToPanel(sourceParams, "inpaint_params", {
      sendPrompt,
      sendParameters,
      includeDenoising: true,
      dispatchEvent: "inpaint_params_updated"
    });

    console.log("[Img2Img] sendToInpaint - Sent to panel");

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

    // Use generated image params if available, otherwise fall back to current UI params
    const sourceParams = generatedImageParams || params;

    // Send image if checked
    if (sendImage) {
      try {
        await sendImageToOutpaint(generatedImage);
      } catch (error) {
        console.error("Failed to send image to outpaint:", error);
      }
    }

    // Send prompt and/or parameters
    sendToPanel(sourceParams, "outpaint_params", {
      sendPrompt,
      sendParameters,
      includeDenoising: true,
      dispatchEvent: "outpaint_params_updated"
    });

    // Navigate to outpaint tab
    if (onTabChange) {
      onTabChange("outpaint");
    }
  };

  // generatedVideo (Img2Vid) result -> Outpaint's outpaint_vid clip input.
  const sendVideoResultToOutpaint = async () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    try {
      await sendVideoToOutpaint(generatedVideo);
    } catch (error) {
      console.error("[Img2Img] Failed to send video to outpaint:", error);
      alert("Failed to send the video to outpaint");
      return;
    }
    if (onTabChange) onTabChange("outpaint");
  };

  // generatedVideo (Img2Vid) result -> Inpaint's temporal inpaint clip input.
  const sendVideoResultToInpaint = async () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    try {
      await sendVideoToInpaint(generatedVideo);
    } catch (error) {
      console.error("[Img2Img] Failed to send video to inpaint:", error);
      alert("Failed to send the video to inpaint");
      return;
    }
    if (onTabChange) onTabChange("inpaint");
  };

  // generatedVideo (Img2Vid) result -> the ref2va reference track (whole-clip
  // conditioning, not a placement anchor -- see sendVideoToReference).
  const sendVideoResultToReference = () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    sendVideoToReference(generatedVideo);
  };

  // generatedAudio (aud2aud) result -> Outpaint's outpaint_aud clip input.
  const sendAudioResultToOutpaint = () => {
    if (!generatedAudio) {
      alert("No audio to send");
      return;
    }
    sendAudioToOutpaint(generatedAudio);
    if (onTabChange) onTabChange("outpaint");
  };

  // generatedAudio (aud2aud) result -> Img2Img again as a new reference clip
  // (self-send = iterate the cover/reference further).
  const sendAudioResultToImg2Img = () => {
    if (!generatedAudio) {
      alert("No audio to send");
      return;
    }
    sendAudioToImg2Img(generatedAudio);
    if (onTabChange) onTabChange("img2img");
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
            console.error("[Img2Img] Failed to save reference image to temp storage:", error);
          }

          if (newPreviews.length === newFiles.length) {
            // Use functional setState to get the latest state
            setRefImagePreviews((prevPreviews) => [...prevPreviews, ...newPreviews]);

            // Update localStorage with refs
            const savedRefImageRefs = localStorage.getItem(REF_IMAGES_STORAGE_KEY);
            const existingRefs = savedRefImageRefs ? JSON.parse(savedRefImageRefs) : [];
            const allRefs = [...existingRefs, ...newRefs];
            localStorage.setItem(REF_IMAGES_STORAGE_KEY, JSON.stringify(allRefs));
            console.log(`[Img2Img] Saved ${newRefs.length} reference images to storage`);
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
        console.log(`[Img2Img] Removed reference image ${index} from storage`);
      } catch (error) {
        console.error("[Img2Img] Failed to update reference images storage:", error);
      }
    }
  };

  const handleClearAllRefImages = () => {
    setRefImages([]);
    setRefImagePreviews([]);

    // Clear localStorage
    localStorage.removeItem(REF_IMAGES_STORAGE_KEY);
    console.log("[Img2Img] Cleared all reference images from storage");
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
            console.error("[Img2Img] Failed to save reference image to temp storage:", error);
          }

          if (newPreviews.length === imageFiles.length) {
            // Use functional setState to get the latest state
            setRefImagePreviews((prevPreviews) => [...prevPreviews, ...newPreviews]);

            // Update localStorage with refs
            const savedRefImageRefs = localStorage.getItem(REF_IMAGES_STORAGE_KEY);
            const existingRefs = savedRefImageRefs ? JSON.parse(savedRefImageRefs) : [];
            const allRefs = [...existingRefs, ...newRefs];
            localStorage.setItem(REF_IMAGES_STORAGE_KEY, JSON.stringify(allRefs));
            console.log(`[Img2Img] Saved ${newRefs.length} reference images to storage (D&D)`);
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
        enabled_categories: enabledCategories
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

  useEffect(() => {
    if (!currentItem || !["img2img", "img2vid", "ref2vid", "aud2aud"].includes(currentItem.type)) {
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
    const result = completedResults.img2img;
    if (!result || (currentItem && ["img2img", "img2vid", "ref2vid", "aud2aud"].includes(currentItem.type))) return;
    setPreviewImage(null);
    if (result.kind === "image") {
      setGeneratedImage(result.url);
      setGeneratedImageSeed(result.seed ?? null);
      setGeneratedImageAncestralSeed(result.ancestralSeed ?? null);
      setGeneratedImageParams(result.params as Img2ImgParams);
      setGeneratedVideo(null);
      setGeneratedAudio(null);
    } else if (result.kind === "video") {
      setGeneratedVideo(result.url);
      setGeneratedVideoInfo(result.info as typeof generatedVideoInfo);
      setGeneratedVideoSeed(result.seed ?? null);
      setGeneratedVideoParams(result.params as Img2ImgParams);
      setGeneratedImage(null);
      setGeneratedAudio(null);
    } else {
      setGeneratedAudio(result.url);
      setGeneratedAudioInfo(result.info as typeof generatedAudioInfo);
      setGeneratedAudioParams(result.params as Img2ImgParams);
      setGeneratedImage(null);
      setGeneratedVideo(null);
    }
  }, [completedResults.img2img, currentItem]);
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
  });

  // Add generation request to queue
  const handleAddToQueue = async () => {
    if (!params.prompt) {
      alert("Please enter a prompt");
      return;
    }

    // Which endpoint this request goes to is decided from a FRESH read of
    // GET /models/current, not from the cached isVideo/isAudio render flags:
    // the model can change under an open page (API call, backend restart,
    // second tab), and routing an image request at a video model costs a 400
    // whose message is about the wrong thing. The cached flags remain the
    // render-time hint; only the dispatch decision is re-verified.
    const modality = await resolveModality();
    const videoMode = modality.isVideo;
    const audioMode = modality.isAudio;

    // Audio mode (ACE-Step) uses an uploaded reference clip instead of an
    // input image; skip the image-required check and base64 conversion below.
    if (audioMode) {
      if (!referenceAudioFile) {
        alert("Please select a reference audio clip");
        return;
      }
    } else if (!inputImage && !inputImagePreview) {
      alert("Please upload an input image");
      return;
    }

    // Convert image to base64 for queue storage (img2img/img2vid only)
    let imageBase64: string = "";
    if (!audioMode) {
      const imageSource = inputImage || inputImagePreview;

      if (typeof imageSource === 'string') {
        // Already a base64 or URL
        imageBase64 = imageSource;
      } else if (imageSource instanceof File) {
        // Convert File to base64
        imageBase64 = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.readAsDataURL(imageSource);
        });
      } else {
        alert("Invalid input image");
        return;
      }
    }

    // Import wildcard replacement function dynamically
    const { replaceWildcardsInPrompt } = await import("@/utils/wildcardStorage");

    // Replace wildcards in prompts
    let processedPrompt = await replaceWildcardsInPrompt(params.prompt);
    const processedNegativePrompt = supportsNegativePrompt
      ? await replaceWildcardsInPrompt(params.negative_prompt)
      : "";

    // Feeling Lucky mode: Generate prompt with TIPO before queueing
    if (params.feeling_lucky && !videoMode) {
      try {
        // Use panel's TIPO settings (not localStorage)
        // Build category order and enabled map from settings
        const categoryOrder = tipoSettings.categories.map((c: any) => c.id);
        const enabledCategories: Record<string, boolean> = {};
        tipoSettings.categories.forEach((c: any) => {
          enabledCategories[c.id] = c.enabled;
        });

        console.log('[Img2Img] Feeling Lucky: Generating prompt with TIPO...');
        const result = await generateTIPOPrompt({
          input_prompt: processedPrompt,
          model_name: tipoSettings.model_name,
          tag_length: tipoSettings.tag_length,
          nl_length: tipoSettings.nl_length,
          temperature: tipoSettings.temperature,
          top_p: tipoSettings.top_p,
          top_k: tipoSettings.top_k,
          max_new_tokens: tipoSettings.max_new_tokens,
          treat_as_nl: treatAsNL,
          category_order: categoryOrder,
          enabled_categories: enabledCategories
        });

        processedPrompt = result.generated_prompt;
        console.log('[Img2Img] Feeling Lucky: Generated prompt:', processedPrompt.substring(0, 100) + '...');
      } catch (error) {
        console.error("TIPO generation failed in Feeling Lucky mode:", error);
        alert("Failed to generate prompt with TIPO. Using original prompt.");
      }
    }

    if (videoMode && modality.modelInfo?.type === "minimax_h3") {
      const refMode = modality.modelInfo?.variant === "ref2va" && countMiniMaxH3References(h3References) > 0;
      const promptMode = refMode ? "ref2va" : params.last_frame_image ? "fl2va" : "i2va";
      try {
        const assisted = await maybeTransformH3PromptForGeneration({
          prompt: processedPrompt,
          mode: promptMode,
          durationSeconds: (params.num_frames ?? 121) / (params.frame_rate ?? 24),
          references: createH3ReferenceInventory({
            pictures: 1 + (params.last_frame_image ? 1 : 0) + (params.keyframes?.length ?? 0) + h3References.images.length,
            videos: h3References.videos.length,
            audios: h3References.audios.length + h3References.videoAudios.filter(Boolean).length + (inputAudioTrack ? 1 : 0),
          }),
        });
        processedPrompt = assisted.prompt;
      } catch (error: any) {
        alert(error?.message || "MiniMax H3 Prompt Assist failed");
        return;
      }
    }

    // Audio mode: an audio model (ACE-Step) is loaded -> enqueue an aud2aud item
    // built from the shared params + the uploaded reference clip. Checked before
    // the video branch (mutually exclusive). Audio loop-generation is out of scope.
    if (audioMode) {
      const audioParams: Aud2AudParams = {
        prompt: processedPrompt,
        lyrics: params.lyrics,
        seed: params.seed,
        inference_steps: params.inference_steps,
        guidance_scale: params.guidance_scale,
        shift: params.shift,
        cover_strength: params.cover_strength,
        vocal_language: params.vocal_language,
        loras: params.loras,
        mode: params.mode,
        repaint_start: params.repaint_start,
        repaint_end: params.repaint_end,
        // Weight-only quantization (both axes). The panel controls are rendered
        // from arch capabilities, and `acestep` is now in runtime_int8_archs +
        // quantized_linear_archs, so these must be carried into the audio
        // params or the UI value is silently dropped.
        unet_quantization: params.unet_quantization,
        quantized_gemm_mode: params.quantized_gemm_mode,
      };
      addToQueue({
        type: "aud2aud",
        params: audioParams as any,
        inputAudio: referenceAudioFile!,
        prompt: processedPrompt,
      });
      return;
    }

    // Video mode: a video model is loaded -> enqueue an img2vid item using the
    // input image as the keyframe. Video loop-generation is out of scope.
    //
    // MiniMax-H3 ref2va with at least one reference is checked FIRST and
    // returns early: "img2vid with references" IS "ref2vid with anchors"
    // (/generate/img2vid refuses a ref2va checkpoint by name and points
    // here). The input image and any extra keyframes/last-frame anchor
    // become the ref2vid request's `keyframes` list; the timeline UI
    // (params.keyframes/last_frame_image/input_image_frame_index) is
    // unchanged, only where it is sent differs. ref2va-ness is read from the
    // fresh fetch, matching Txt2ImgPanel.
    if (videoMode) {
      const freshIsRef2Va =
        modality.modelInfo?.type === "minimax_h3" && modality.modelInfo?.variant === "ref2va";
      if (freshIsRef2Va && countMiniMaxH3References(h3References) > 0) {
        const ref2vidKeyframes: MiniMaxH3Keyframe[] = [
          { image: imageBase64, frame_index: params.input_image_frame_index ?? 0 },
          ...(params.last_frame_image ? [{ image: params.last_frame_image, frame_index: -1 }] : []),
          ...(params.keyframes ?? []),
        ];
        const refParams: Ref2VidParams = {
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
          unet_quantization: params.unet_quantization,
          quantized_gemm_mode: params.quantized_gemm_mode,
          reference_image_size: h3ReferenceImageSize,
          keyframes: ref2vidKeyframes,
        };
        addToQueue({
          type: "ref2vid",
          params: refParams as any,
          references: h3References,
          prompt: processedPrompt,
        });
        return;
      }
      const videoParams: Img2VidParams = {
        prompt: processedPrompt,
        negative_prompt: processedNegativePrompt,
        width: params.width,
        height: params.height,
        num_frames: params.num_frames,
        frame_rate: params.frame_rate,
        // The optional END keyframe (MiniMax-H3 fl2va) and the placement of
        // every anchor. Carried into the queued item's params, because the
        // queue is what the sender is handed -- a value left in `params` alone
        // never reaches generateImg2Vid.
        last_frame_image: params.last_frame_image ?? null,
        input_image_frame_index: params.input_image_frame_index ?? 0,
        keyframes: params.keyframes ?? [],
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
      };
      addToQueue({
        type: "img2vid",
        params: videoParams as any,
        inputImage: imageBase64,
        // The ia2v track rides on the ITEM, like every other upload the queue
        // carries, so a queued request keeps the track it was built with after
        // the panel's own picker changes. Only sent where the loaded
        // architecture reads it.
        inputAudio: (supportsAudioConditioning && inputAudioTrack) ? inputAudioTrack : undefined,
        prompt: processedPrompt,
      });
      return;
    }

    // Create loop group ID if loop generation is enabled
    const loopGroupId = loopGenerationConfig.enabled ? `loop_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` : undefined;
    const hasEnabledLoopSteps = loopGenerationConfig.enabled && loopGenerationConfig.steps.some(s => s.enabled);
    // Main step decode directive: resizeMode is moot for the main step (it has
    // none of its own) — passing "latent" correctly forces loop_decode="none"
    // for decodeMode "final-only" when loop steps follow (img2img supports
    // latent passthrough for its main step).
    const mainDecodeDirective = computeLoopDecodeDirective({
      decodeMode: loopGenerationConfig.decodeMode ?? "every",
      isFinalStep: !hasEnabledLoopSteps,
      resizeMode: "latent",
      supportsLatentPassthrough: true,
    });

    addToQueue({
      type: "img2img",
      params: {
        ...params,
        prompt: processedPrompt,
        negative_prompt: processedNegativePrompt,
        loop_decode: mainDecodeDirective.loop_decode,
        skip_gallery: mainDecodeDirective.skip_gallery,
      },
      inputImage: imageBase64,
      prompt: processedPrompt,
      loopGroupId,
      loopStepIndex: loopGroupId ? -1 : undefined,
      isLoopStep: false,
      useTrainingModel,
      trainingRunId: activeTraining?.run_id,
    });

    // If loop generation is enabled, add all loop steps immediately
    // Use the processed (and potentially TIPO-generated) prompt for all loop steps
    if (loopGenerationConfig.enabled && loopGroupId) {
      await addLoopStepsToQueueImmediate({
        ...params,
        prompt: processedPrompt,
        negative_prompt: processedNegativePrompt,
      } as Img2ImgParams, loopGroupId);
    }
  };

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

  // Add loop generation steps to queue immediately (without base image URL)
  const addLoopStepsToQueueImmediate = useCallback(async (mainParams: Img2ImgParams, loopGroupId: string) => {
    if (!loopGenerationConfig.enabled || loopGenerationConfig.steps.length === 0) {
      return;
    }

    console.log('[Img2Img] Adding loop steps with mainParams.unet_quantization:', mainParams.unet_quantization);

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
        vae_drift_correction: mainParams.vae_drift_correction, // Inherit VAE drift correction setting
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
        use_pinned_memory: mainParams.use_pinned_memory,
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
        stepParams.controlnets = (step.controlnets || []).map(cn => ({
          ...cn,
          // If useLoopImage is true, set image_base64 to empty (will be filled after generation)
          image_base64: cn.useLoopImage ? "" : cn.image_base64,
        }));
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
        supportsLatentPassthrough: true, // img2img loop steps support latent passthrough
      });
      stepParams.loop_decode = decodeDirective.loop_decode;
      stepParams.skip_gallery = decodeDirective.skip_gallery;

      stepParams.prompt_chunking_mode = mainParams.prompt_chunking_mode;
      stepParams.max_prompt_chunks = mainParams.max_prompt_chunks;
      stepParams.unet_quantization = mainParams.unet_quantization;
      stepParams.quantized_gemm_mode = mainParams.quantized_gemm_mode;
      stepParams.original_size_w = mainParams.original_size_w;
      stepParams.original_size_h = mainParams.original_size_h;
      stepParams.original_size_scale = mainParams.original_size_scale;
      stepParams.cpu_text_encoding = mainParams.cpu_text_encoding;
      stepParams.vision_encoder_path = mainParams.vision_encoder_path;

      const processedPrompt = await replaceWildcardsInPrompt(stepParams.prompt);
      const processedNegativePrompt = await replaceWildcardsInPrompt(stepParams.negative_prompt);

      addToQueue({
        type: "img2img",
        params: {
          ...stepParams,
          prompt: processedPrompt,
          negative_prompt: processedNegativePrompt,
        },
        inputImage: "", // Will be set when previous step completes
        prompt: `[Loop ${i + 1}/${enabledSteps.length}] ${processedPrompt.substring(0, 50)}...`,
        loopGroupId,
        loopStepIndex: i,
        isLoopStep: true,
        useTrainingModel,
        trainingRunId: activeTraining?.run_id,
      });
    }

    console.log(`[Img2Img] Added ${enabledSteps.length} loop steps to queue with group ID: ${loopGroupId}`);
  }, [loopGenerationConfig, addToQueue, refImages, useTrainingModel, activeTraining]);

  // Process queue - automatically start next item
  const processQueueRef = useRef<() => Promise<void>>();

  const processQueue = useCallback(async () => {
    console.log("[Img2Img] processQueue called, isGenerating:", isGeneratingRef.current);
    if (isGeneratingRef.current) {
      console.log("[Img2Img] Already generating, skipping");
      return;
    }

    const nextItem = startNextInQueue(["img2img", "img2vid", "ref2vid", "aud2aud"]);
    console.log("[Img2Img] Next item from queue:", nextItem);
    if (!nextItem || (nextItem.type !== "img2img" && nextItem.type !== "img2vid" && nextItem.type !== "ref2vid" && nextItem.type !== "aud2aud")) {
      console.log("[Img2Img] No img2img/img2vid/ref2vid/aud2aud items in queue");
      return;
    }

    // Audio branch: aud2aud item (an audio model is loaded). The queued
    // reference clip (a File) is the cover source. Produces a .flac and
    // renders an <audio> instead of an <img>. No loop-generation handling.
    if (nextItem.type === "aud2aud") {
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
        const referenceAudio = nextItem.inputAudio;
        if (!referenceAudio) {
          throw new Error("No reference audio available for aud2aud generation");
        }
        const result = await generateAud2Aud(nextItem.params as Aud2AudParams, referenceAudio);
        const audioUrl = `/outputs/${result.image.filename}`;
        const audioInfo = { duration: result.image.duration, sample_rate: result.image.sample_rate };
        const audioParams = {
          ...(nextItem.params as Img2ImgParams),
          seed: getResultSeed(result) ?? (nextItem.params as Img2ImgParams).seed,
        };
        setGeneratedAudio(audioUrl);
        setGeneratedAudioInfo(audioInfo);
        setGeneratedAudioParams(audioParams);
        publishCompletedResult({ panel: "img2img", kind: "audio", url: audioUrl, info: audioInfo, params: audioParams });
        if (onImageGenerated) onImageGenerated(audioUrl);
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        completeCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
      } catch (error: any) {
        console.error("[Img2Img] aud2aud generation failed:", error);
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
        alert(isGenerationStalledError(error) ? error.message : "aud2aud generation failed. Please check console for details.");
      }
      return;
    }

    // Video branch: img2vid item (a video model is loaded). The queued input
    // image is the keyframe. Produces an .mp4 and renders a <video>.
    if (nextItem.type === "img2vid") {
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
      setGeneratedAudio(null);
      setGeneratedAudioInfo(null);
      try {
        const keyframe = nextItem.inputImage;
        if (!keyframe) {
          throw new Error("No keyframe image available for img2vid generation");
        }
        // The uploaded track lives on the item (it is a File); the sender reads
        // it from the params object, so it is merged in HERE -- the same
        // dequeue-time site every other upload-carrying request rebuilds.
        const apiParams: Img2VidParams = {
          ...(nextItem.params as Img2VidParams),
          input_audio: nextItem.inputAudio ?? null,
        };
        const result = await generateImg2Vid(apiParams, keyframe);
        const videoUrl = `/outputs/${result.image.filename}`;
        const videoInfo = { num_frames: result.image.num_frames, fps: result.image.fps, duration: result.image.duration };
        const videoSeed = getResultSeed(result);
        setGeneratedVideo(videoUrl);
        setGeneratedVideoInfo(videoInfo);
        // The seed the run actually used (-1 in the request means "pick one"),
        // so the seed control's reuse button can pin it for the next run.
        setGeneratedVideoSeed(videoSeed);
        setGeneratedVideoParams(nextItem.params as Img2ImgParams);
        publishCompletedResult({ panel: "img2img", kind: "video", url: videoUrl, info: videoInfo, seed: videoSeed, params: nextItem.params });
        if (onImageGenerated) onImageGenerated(videoUrl);
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        completeCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
      } catch (error: any) {
        console.error("[Img2Img] img2vid generation failed:", error);
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        failCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
        alert(isGenerationStalledError(error) ? error.message : "img2vid generation failed. Please check console for details.");
      }
      return;
    }

    // Video branch: ref2vid item -- img2vid-with-references, routed to
    // /generate/ref2vid because that is the endpoint the ref2va checkpoint was
    // trained for (/generate/img2vid refuses this checkpoint by name). The
    // input image already rides in nextItem.params.keyframes (built at
    // enqueue time); references ride on nextItem.references, same as
    // Txt2ImgPanel's ref2vid item.
    if (nextItem.type === "ref2vid") {
      isGeneratingRef.current = true;
      setIsGenerating(true);
      setProgress(0);
      setProgressMessage("");
      setTotalSteps((nextItem.params as any).num_inference_steps || 20);
      setPreviewImage(null);
      setGeneratedImage(null);
      setGeneratedVideo(null);
      setGeneratedVideoInfo(null);
      setGeneratedVideoSeed(null);
      setGeneratedAudio(null);
      setGeneratedAudioInfo(null);
      try {
        const result = await generateRef2Vid(
          nextItem.params as Ref2VidParams,
          nextItem.references ?? EMPTY_MINIMAX_H3_REFERENCES);
        const videoUrl = `/outputs/${result.image.filename}`;
        const videoInfo = { num_frames: result.image.num_frames, fps: result.image.fps, duration: result.image.duration };
        const videoSeed = getResultSeed(result);
        setGeneratedVideo(videoUrl);
        setGeneratedVideoInfo(videoInfo);
        setGeneratedVideoSeed(videoSeed);
        setGeneratedVideoParams(nextItem.params as Img2ImgParams);
        publishCompletedResult({ panel: "img2img", kind: "video", url: videoUrl, info: videoInfo, seed: videoSeed, params: nextItem.params });
        if (onImageGenerated) onImageGenerated(videoUrl);
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        completeCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
      } catch (error: any) {
        console.error("[Img2Img] ref2vid generation failed:", error);
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        failCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
        alert(isGenerationStalledError(error)
          ? error.message
          : `ref2vid generation failed: ${error?.response?.data?.detail || error?.response?.data?.error || "see the console for details."}`);
      }
      return;
    }

    // Save current image before starting new generation
    const previousImage = generatedImage;

    isGeneratingRef.current = true;
    setIsGenerating(true);
    setProgress(0);
    setProgressMessage("");
    const denoisingStrength = nextItem.params.denoising_strength || 0.75;
    const actualSteps = Math.ceil((nextItem.params.steps || 20) * denoisingStrength);
    setTotalSteps(actualSteps);
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
      // For loop steps, use the input image or fall back to previous image.
      // Latent passthrough chaining (decodeMode "final-only"): when the
      // previous step returned a cached latent_id, there is no image to fetch.
      const inputImageToUse = nextItem.inputLatentId ? undefined : (nextItem.inputImage || previousImage);
      if (!nextItem.inputLatentId && !inputImageToUse) {
        throw new Error("No input image available for img2img generation");
      }

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

      // Debug log for quantization
      console.log('[Img2Img] Generating with params.unet_quantization:', paramsWithDevMode.unet_quantization);

      let imageUrl: string | undefined;
      let result: any;
      // Use the per-item flag (set at enqueue time) so loop steps queued under the
      // training model keep using it even though this panel's own checkbox may be off.
      const itemUseTraining = (nextItem?.useTrainingModel ?? useTrainingModel);
      const itemRunId = nextItem?.trainingRunId ?? activeTraining?.run_id;
      if (itemUseTraining && itemRunId) {
        // Training-preview branch: encode init image as base64 and route
        // to /generate/img2img/training-preview.  Result is a blob;
        // we wrap it in an object-URL for display (no gallery save).
        // Not supported with latent passthrough (a separate, simpler preview
        // flow that doesn't know about loop_decode/input_latent_id).
        if (!inputImageToUse) {
          throw new Error("Training-preview generation requires an input image (latent passthrough is not supported)");
        }
        const initImageBase64 = await toBase64(inputImageToUse);
        const preview = await generateImg2ImgTrainingPreview({
          ...(paramsWithDevMode as any),
          init_image_base64: initImageBase64,
          denoising_strength: paramsWithDevMode.denoising_strength ?? 0.75,
          run_id: itemRunId,
          save_to_gallery: savePreviewToGallery,
        });
        if (preview.filename) {
          imageUrl = `/outputs/${preview.filename}`;
        } else {
          if (previewBlobUrlRef.current) URL.revokeObjectURL(previewBlobUrlRef.current);
          const objectUrl = URL.createObjectURL(preview.blob);
          previewBlobUrlRef.current = objectUrl;
          imageUrl = objectUrl;
        }
        result = {
          image: {
            filename: preview.filename
              ?? `preview_${preview.requestId ?? "training"}.png`,
            filepath: imageUrl,
            seed: preview.seed ? Number(preview.seed) : -1,
            ancestral_seed: -1,
            prompt: paramsWithDevMode.prompt,
            negative_prompt: paramsWithDevMode.negative_prompt,
            width: paramsWithDevMode.width,
            height: paramsWithDevMode.height,
          },
        };
      } else {
        result = await generateImg2Img(paramsWithDevMode, inputImageToUse, nextItem.inputLatentId);
        // loop_decode="none" (decodeMode "final-only", resize_mode "latent")
        // returns { latent_id, actual_seed } with NO image.
        imageUrl = isLatentOnlyResult(result) ? undefined : `/outputs/${getResultFilename(result)}`;
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
      const completedParams: Img2ImgParams = {
        ...nextItem.params,
        seed: resultSeed,
        ancestral_seed: resultAncestralSeed ?? -1,
        width: result.image?.width ?? nextItem.params.width,
        height: result.image?.height ?? nextItem.params.height,
      };
      setGeneratedImageParams(completedParams);
      if (imageUrl) {
        publishCompletedResult({
          panel: "img2img",
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
        onImageGenerated(imageUrl);
      }

      // If this item has a loop group, update the next loop step's input image, prompt, and ControlNets
      // Use nextItem (not currentItem from context) to avoid timing issues
      if (nextItem?.loopGroupId !== undefined) {
        const nextLoopStepIndex = (nextItem.loopStepIndex ?? -1) + 1;

        console.log(`[Img2Img] Processing loop step completion:`, {
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
          console.log(`[Img2Img] Updating loop step ${nextLoopStepIndex} with cached latent:`, result.latent_id);
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
            console.log(`[Img2Img] Scale mode (latent passthrough): ${currentWidth}x${currentHeight} * ${scale} = ${scaledWidth}x${scaledHeight}`);
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
        console.log(`[Img2Img] Updating loop step ${nextLoopStepIndex} with input image:`, imageUrl);
        updateQueueItemByLoop(nextItem.loopGroupId, nextLoopStepIndex, { inputImage: imageUrl, inputLatentId: undefined });

        // If TIPO was used for base generation, update loop steps with TIPO-generated prompt
        if (nextItem.loopStepIndex === -1 && nextItem.params.use_tipo && result.image?.prompt) {
          console.log(`[Img2Img] Base generation used TIPO, updating all loop steps with TIPO prompt`);
          console.log(`[Img2Img] Original prompt: ${nextItem.params.prompt?.substring(0, 100)}...`);
          console.log(`[Img2Img] TIPO prompt: ${result.image.prompt?.substring(0, 100)}...`);

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

        console.log(`[Img2Img] Step config:`, {
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
              console.log(`[Img2Img] Scale mode: ${imageWidth}x${imageHeight} * ${stepConfig.scale} = ${scaledWidth}x${scaledHeight}`);

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
          console.log(`[Img2Img] Processing ${stepConfig.controlnets.length} ControlNet(s) for loop step ${nextLoopStepIndex}`);

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

          console.log(`[Img2Img] Converted image to base64, length: ${imageBase64.length}`);

          // Update ControlNets with useLoopImage enabled using callback to preserve existing params
          updateQueueItemByLoop(nextItem.loopGroupId!, nextLoopStepIndex, (item) => {
            const updatedControlnets = stepConfig.controlnets.map((cnConfig, idx) => {
              console.log(`[Img2Img] ControlNet ${idx}: useLoopImage=${cnConfig.useLoopImage}`);
              if (cnConfig.useLoopImage) {
                console.log(`[Img2Img] Setting image_base64 for ControlNet ${idx}`);
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

          console.log(`[Img2Img] ControlNet images updated for loop step ${nextLoopStepIndex}`);
        }
        }
      }

      // Reset state first, then complete item
      console.log("[Img2Img] Generation complete, resetting state and completing item");
      isGeneratingRef.current = false;
      setIsGenerating(false);
      setProgress(0);
      setProgressMessage("");
      completeCurrentItem();

      // Wait briefly for state to propagate, then trigger next
      setTimeout(() => {
        console.log("[Img2Img] Triggering next queue item");
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
      console.log("[Img2Img] Generation failed, resetting state and failing item");
      isGeneratingRef.current = false;
      setIsGenerating(false);
      setProgress(0);
      setProgressMessage("");
      failCurrentItem();

      // Wait briefly for state to propagate, then trigger next
      setTimeout(() => {
        console.log("[Img2Img] Triggering next queue item after failure");
        if (processQueueRef.current) {
          processQueueRef.current();
        }
      }, 100);

      if (alertMessage) {
        alert(alertMessage);
      }
    }
  }, [isGenerating, generatedImage, onImageGenerated, startNextInQueue, completeCurrentItem, failCurrentItem, updateQueueItem, queue, publishCompletedResult]);

  processQueueRef.current = processQueue;

  // Auto-start queue processing when queue has pending items and not currently generating
  useEffect(() => {
    const hasPendingItems = queue.some(item => item.status === "pending" && (item.type === "img2img" || item.type === "img2vid" || item.type === "ref2vid" || item.type === "aud2aud"));
    const isCurrentItemNull = currentItem === null;

    console.log("[Img2Img] Queue effect:", {
      hasPendingItems,
      isCurrentItemNull,
      isGenerating,
      queueLength: queue.length,
      queue: queue,
      currentItem: currentItem,
      generateForever
    });

    // If generate forever is enabled and queue is empty, add new item
    if (generateForever && !hasPendingItems && isCurrentItemNull && !isGenerating && params.prompt && (inputImage || inputImagePreview)) {
      console.log("[Img2Img] Generate forever: Adding new item to queue");
      handleAddToQueue();
      return;
    }

    // A queue survives a page reload and a backend restart, so on mount there
    // can be pending items with no model loaded yet. Dispatching then earns an
    // immediate 400 ("No video model loaded") and the item is marked failed for
    // a reason that has nothing to do with the item. Hold instead: `modelLoaded`
    // is a dependency, so the queue starts by itself once a model is up.
    if (hasPendingItems && isCurrentItemNull && !isGenerating && !modelLoaded) {
      console.log("[Img2Img] Queue held: no model loaded yet");
      return;
    }

    if (hasPendingItems && isCurrentItemNull && !isGenerating) {
      console.log("[Img2Img] Auto-starting queue processing");
      processQueue();
    }
  }, [queue, currentItem, isGenerating, processQueue, generateForever, params, inputImage, inputImagePreview, modelLoaded]);

  // Handle Ctrl+Enter keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if Image Editor is open (global check for all Image Editors)
      if (document.body.dataset.imageEditorOpen || document.querySelector('[data-prompt-assist-open="true"]')) return;

      if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        handleAddToQueue();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [params, inputImage, inputImagePreview]);

  // Render functions for each Img2Img Options tab (see IMG2IMG_OPTIONS_TABS /
  // IMG2IMG_OPTIONS_TAB_KEYS / isImg2ImgOptionsTabActive above). Every control
  // below is unchanged from its original in-Card location -- same param
  // binding / handler / conditional reveal -- ported from InpaintPanel's
  // inpaintOptionsTabRender pattern.
  const img2imgOptionsTabRender: Record<Img2ImgOptionsTabId, () => JSX.Element> = {
    img2img: () => (
      <div className="space-y-4">
        <Slider
          label="Denoising Strength"
          min={0}
          max={1}
          step={0.05}
          value={params.denoising_strength}
          onChange={(e) => setParams({ ...params, denoising_strength: parseFloat(e.target.value) })}
        />
        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="img2img_fix_steps"
            checked={params.img2img_fix_steps ?? true}
            onChange={(e) => setParams({ ...params, img2img_fix_steps: e.target.checked })}
            className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
          />
          <label htmlFor="img2img_fix_steps" className="text-sm text-gray-300">
            Do full steps (ensures complete denoising regardless of strength)
          </label>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Select
            label="Resize Mode"
            options={[
              { value: "image", label: "Resize Image" },
              { value: "latent", label: "Resize Latent" },
            ]}
            value={params.resize_mode}
            onChange={(e) => setParams({ ...params, resize_mode: e.target.value })}
          />
          <Select
            label="Resampling Method"
            options={[
              { value: "lanczos", label: "Lanczos (High Quality)" },
              { value: "bicubic", label: "Bicubic" },
              { value: "bilinear", label: "Bilinear" },
              { value: "nearest", label: "Nearest (Pixelated)" },
            ]}
            value={params.resampling_method}
            onChange={(e) => setParams({ ...params, resampling_method: e.target.value })}
          />
        </div>
      </div>
    ),

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
            id="spectrum_enable_i2i"
            checked={params.spectrum_enable || false}
            onChange={(e) => setParams({ ...params, spectrum_enable: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="spectrum_enable_i2i" className="text-sm text-gray-300">
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
              <label className="text-xs text-gray-400 flex items-center gap-1">
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
            <label className="text-xs text-gray-400 flex items-center gap-1">Mix w
              <input type="number" min={0} max={1} step={0.05} value={params.spectrum_w ?? 0.5}
                onChange={(e) => setParams({ ...params, spectrum_w: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">Mix w decay
              <input type="number" min={0} step={0.25} value={params.spectrum_w_decay ?? 0.0}
                onChange={(e) => setParams({ ...params, spectrum_w_decay: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1" title="Limits how far a forecast may advance past the last real pass, relative to the observed trajectory speed. 0 disables the cap.">Delta cap
              <input type="number" step={0.25} value={params.spectrum_delta_cap ?? 0.0}
                onChange={(e) => setParams({ ...params, spectrum_delta_cap: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">Basis m
              <input type="number" min={1} max={8} step={1} value={params.spectrum_m ?? 4}
                onChange={(e) => setParams({ ...params, spectrum_m: parseInt(e.target.value) || 4 })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">Ridge λ
              <input type="number" min={0} step={0.01} value={params.spectrum_lam ?? 0.1}
                onChange={(e) => setParams({ ...params, spectrum_lam: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">Warmup
              <input type="number" min={1} step={1} value={params.spectrum_warmup_steps ?? 3}
                onChange={(e) => setParams({ ...params, spectrum_warmup_steps: parseInt(e.target.value) || 3 })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">Window
              <input type="number" min={1} step={1} value={params.spectrum_window_size ?? 4}
                onChange={(e) => setParams({ ...params, spectrum_window_size: parseInt(e.target.value) || 4 })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">Flex
              <input type="number" min={0} max={1} step={0.05} value={params.spectrum_flex_window ?? 0.75}
                onChange={(e) => setParams({ ...params, spectrum_flex_window: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">Tail
              <input type="number" min={0} max={0.5} step={0.02} value={params.spectrum_tail ?? 0.12}
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
            id="fbcache_enable_i2i"
            checked={params.fbcache_enable || false}
            onChange={(e) => setParams({ ...params, fbcache_enable: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="fbcache_enable_i2i" className="text-sm text-gray-300">
            First Block Cache (dynamic caching)
          </label>
          <span className="text-xs text-gray-500">(mutually exclusive with Spectrum)</span>
        </div>
        {params.fbcache_enable && (
          <div className="ml-6 mt-1 grid grid-cols-2 gap-2">
            <label className="text-xs text-gray-400 flex items-center gap-1">Residual threshold (higher = more skips)
              <NumberInput min={0} step={0.01} parse="float" value={params.fbcache_threshold ?? 0.12}
                defaultValue={0.12}
                placeholder="0.12"
                onCommit={(v) => setParams({ ...params, fbcache_threshold: v })}
                className="w-20" />
            </label>
            <label className="text-xs text-gray-400 flex items-center gap-1">Warmup steps
              <NumberInput min={0} step={1} value={params.fbcache_warmup_steps ?? 1}
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
            id="flatten_in_loop_i2i"
            checked={params.flatten_in_loop || false}
            onChange={(e) => setParams({ ...params, flatten_in_loop: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="flatten_in_loop_i2i" className="text-sm text-gray-300" title="During the final denoise steps, detects the flat background region and replaces it with its solid dominant color (both luma and chroma become uniform - stronger than Color Flatten); no-op when no confident flat region is found; SD/SDXL only for now.">
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

        <div className="flex items-center space-x-2" title="Subtracts the VAE encode/decode round-trip color bias (measured per image) from the output; independent of denoising strength.">
          <input
            type="checkbox"
            id="vae_drift_correction"
            checked={params.vae_drift_correction ?? false}
            onChange={(e) => setParams({ ...params, vae_drift_correction: e.target.checked })}
            className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
          />
          <label htmlFor="vae_drift_correction" className="text-sm text-gray-300">
            VAE drift correction
          </label>
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
          value={String(params.max_prompt_chunks || 0)}
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
                  💡 {currentModelInfo?.model_info?.type === "flux2" ? "FLUX.2" : "Z-Image"} quantization can reduce VRAM significantly. Text encoder ({currentModelInfo?.model_info?.type === "flux2" ? "Qwen3" : "Gemma2 3.4B"}) is particularly large.
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
            {/* Every other architecture: 1-column layout. Only SD1.5/SDXL have a
                U-Net, so the label is arch-aware (see transformerQuantizationLabel). */}
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
            </div>
            {params.unet_quantization && params.unet_quantization !== "none" && params.unet_quantization !== "int8" && (
              <div className="bg-yellow-900/20 border border-yellow-600/30 rounded-lg p-3">
                <p className="text-xs text-yellow-200">
                  ⚠️ Quantization reduces VRAM but may affect quality. FP8 weights are dequantized back to full precision per operation during inference, so generation is slower than without quantization. Original model kept on CPU.
                </p>
              </div>
            )}
            {params.unet_quantization === "int8" && (
              <div className="bg-blue-900/20 border border-blue-600/30 rounded-lg p-3">
                <p className="text-xs text-blue-200">
                  INT8 converts the transformer in place the first time you generate after loading the model, and keeps it for the session. Layers where INT8 measures worse than FP8 E4M3 are stored as E4M3 instead. The conversion is one-way: reload the model to return to the checkpoint&apos;s original precision.
                </p>
              </div>
            )}
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
                id="enable_block_swap_i2i"
                checked={params.enable_block_swap || false}
                onChange={(e) => setParams({ ...params, enable_block_swap: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="enable_block_swap_i2i" className="text-sm text-gray-300">
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
                    id="use_pinned_memory_i2i"
                    checked={params.use_pinned_memory || false}
                    onChange={(e) => setParams({ ...params, use_pinned_memory: e.target.checked })}
                    className="rounded"
                  />
                  <label htmlFor="use_pinned_memory_i2i" className="text-xs text-gray-300">
                    Use Pinned Memory (faster transfer, more RAM)
                  </label>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="block_swap_h2d_only_i2i"
                    checked={params.block_swap_h2d_only || false}
                    onChange={(e) => setParams({ ...params, block_swap_h2d_only: e.target.checked })}
                    className="rounded"
                  />
                  <label htmlFor="block_swap_h2d_only_i2i" className="text-xs text-gray-300">
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

  // ── INPUT IMAGES tabs ────────────────────────────────────────────────────
  // One tab per conditioning image, in timeline order of the slots they live
  // in: the input image first, then each `keyframes` entry, then the
  // last-frame alias when it holds an image. Labels carry the frame the anchor
  // is on so the tab strip and the timeline agree without cross-referencing.
  const videoLastIndex = Math.max(0, (params.num_frames ?? 124) - 1);
  const inputTabs: Array<{ id: string; label: string; anchor: ExtraAnchor | null }> = [
    {
      id: "input",
      label: supportsKeyframePlacement
        ? `Input · f${(params.input_image_frame_index ?? 0) === -1 ? videoLastIndex : (params.input_image_frame_index ?? 0)}`
        : "Input",
      anchor: null,
    },
    ...(params.keyframes ?? []).map((keyframe, index) => ({
      id: `kf-${index}`,
      label: `KF ${index + 1} · f${keyframe.frame_index === -1 ? videoLastIndex : keyframe.frame_index}`,
      anchor: { kind: "keyframe" as const, index },
    })),
    ...(params.last_frame_image
      ? [{ id: "last", label: "Last frame", anchor: { kind: "last" as const } }]
      : []),
  ];
  // A removed tab must not leave the card blank: fall back to the input image.
  const activeInputTabId = inputTabs.some((tab) => tab.id === activeInputTab)
    ? activeInputTab
    : "input";
  const activeInputAnchor =
    inputTabs.find((tab) => tab.id === activeInputTabId)?.anchor ?? null;
  const loadedInputImageCount =
    (inputImagePreview ? 1 : 0) +
    (params.keyframes ?? []).filter((keyframe) => typeof keyframe.image === "string").length +
    (params.last_frame_image ? 1 : 0);

  /**
   * The image drop zone, with every affordance the single input image has: a
   * file picker, a Clear button, drag-and-drop, and DOUBLE-CLICK THROUGH TO
   * THE PAINT EDITOR. Rendered once per tab so each anchor gets all of them --
   * this is the same markup the single-image card used, parameterised by which
   * slot it reads and writes.
   */
  const renderImageDropZone = (options: {
    preview: string | null;
    onFile: (file: File) => void;
    onClear: () => void;
    onEdit: () => void;
    clearTitle: string;
    emptyText: string;
  }) => (
    <div className="space-y-2">
      <div className="flex gap-2">
        <input
          type="file"
          accept="image/png,image/jpeg,image/jpg,image/webp"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) options.onFile(file);
            e.target.value = "";
          }}
          className="flex-1 block w-full text-sm text-gray-400
            file:mr-4 file:py-2 file:px-4
            file:rounded-lg file:border-0
            file:text-sm file:font-medium
            file:bg-blue-600 file:text-white
            hover:file:bg-blue-700
            file:cursor-pointer cursor-pointer"
        />
        {options.preview && (
          <Button
            onClick={options.onClear}
            variant="secondary"
            size="sm"
            title={options.clearTitle}
          >
            Clear
          </Button>
        )}
      </div>
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file) options.onFile(file);
        }}
        onDoubleClick={() => {
          if (options.preview) options.onEdit();
        }}
        className={`h-[clamp(10rem,22vh,13rem)] bg-gray-800 rounded-lg overflow-hidden border-2 border-dashed transition-colors ${
          isDragging
            ? 'border-blue-500 bg-gray-700'
            : 'border-gray-600'
        } ${options.preview ? 'cursor-pointer' : ''}`}
        title={options.preview ? "Double-click to edit image" : ""}
      >
        {options.preview ? (
          <img
            src={options.preview}
            alt="Input"
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <p className="text-gray-500 text-center px-4">
              {isDragging ? 'Drop image here' : options.emptyText}
            </p>
          </div>
        )}
      </div>
      {options.preview && (
        <p className="text-xs text-gray-500 text-center">
          💡 Double-click the image to edit with built-in paint tool
        </p>
      )}
    </div>
  );

  // ── What the video Absolute sliders are allowed to reach ─────────────────
  //
  // The envelope is on the SHORT and LONG edges, not on width and height, so
  // each slider's ceiling depends on where the other one sits (see
  // videoCanvasAxisBounds). Passing the OTHER axis is what keeps 1344x768 and
  // 768x1344 both reachable while 1344x1344 is not.
  //
  // The bounds constrain the CONTROL, never the stored value: a width/height
  // carried over from an architecture with a wider envelope is left exactly as
  // the user set it (a canvas is a hard 400 server-side precisely because it is
  // not something to change under a caller), and `videoCanvasOverEnvelope`
  // states the mismatch instead.
  const videoCanvasWidth = params.width ?? 768;
  const videoCanvasHeight = params.height ?? 512;
  const videoWidthBounds = videoCanvasAxisBounds(archCapabilities, loadedArch, videoCanvasHeight);
  const videoHeightBounds = videoCanvasAxisBounds(archCapabilities, loadedArch, videoCanvasWidth);
  const videoCanvasOverEnvelope = videoCanvasExceedsEnvelope(
    archCapabilities, loadedArch, videoCanvasWidth, videoCanvasHeight);
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
      </div>
      {loadedArch === "minimax_h3" && (
        <H3PromptAssist
          prompt={params.prompt}
          onApply={(prompt) => setParams((previous) => ({ ...previous, prompt }))}
          suggestedMode={isRef2Va && countMiniMaxH3References(h3References) > 0
            ? "ref2va"
            : params.last_frame_image ? "fl2va" : "i2va"}
          durationSeconds={(params.num_frames ?? 121) / (params.frame_rate ?? 24)}
          references={createH3ReferenceInventory({
            pictures: 1 + (params.last_frame_image ? 1 : 0) + (params.keyframes?.length ?? 0) + h3References.images.length,
            videos: h3References.videos.length,
            audios: h3References.audios.length + h3References.videoAudios.filter(Boolean).length + (inputAudioTrack ? 1 : 0),
          })}
        />
      )}
      {!isVideo && <div className="flex flex-wrap items-center gap-1.5 rounded bg-gray-800 px-2 py-1.5">
        <label className="flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            checked={params.feeling_lucky || false}
            onChange={(e) => setParams({ ...params, feeling_lucky: e.target.checked })}
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
      {!supportsNegativePrompt && (
        <p className="text-xs text-gray-500">Unavailable for the loaded model; the saved value is preserved.</p>
      )}
    </Card>
  );

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
            console.log("[Img2Img] Model changed, updated currentModelInfo:", modelInfo);

            // Auto-adjust sampler/schedule for Flow Matching models (Z-Image, FLUX.2)
            const modelType = modelInfo?.model_info?.type;
            if (modelType === "zimage" || modelType === "flux2" || modelType === "anima") {
              // Flow Matching models: use Euler with flow schedule
              setParams(prev => ({
                ...prev,
                sampler: "euler",
                schedule_type: "flow"
              }));
              console.log("[Img2Img] Auto-set sampler=euler, schedule_type=flow for Flow Matching model");
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
          storageKeyPrefix="img2img"
        />

        <GenerationLeadGrid
          conditioning={(
            <>
        {!isAudio && (
        <Card
          title={multiImageInput ? "Input Images" : "Input Image"}
          collapsible={true}
          defaultCollapsed={false}
          storageKey="img2img_input_collapsed_v2"
          collapsedPreview={
            multiImageInput ? (
              loadedInputImageCount > 0 ? (
                <span className="text-green-400 text-sm">
                  ✓ {loadedInputImageCount} image{loadedInputImageCount > 1 ? "s" : ""}
                </span>
              ) : (
                <span className="text-gray-500 text-sm">No images</span>
              )
            ) : inputImagePreview ? (
              <span className="text-green-400 text-sm">✓ Image loaded</span>
            ) : (
              <span className="text-gray-500 text-sm">No image</span>
            )
          }
        >
          {/* TAB STRIP -- only where the loaded architecture takes more than
              one conditioning image (see multiImageInput). Each tab is one
              anchor and gets the full set of affordances below. */}
          {multiImageInput && (
            <div className="mb-2 space-y-1.5">
              <div className="flex flex-wrap items-center gap-1">
                {inputTabs.map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveInputTab(tab.id)}
                    className={`px-2 py-1 text-xs rounded transition-colors ${
                      tab.id === activeInputTabId
                        ? "bg-blue-600 text-white"
                        : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
                <button
                  type="button"
                  disabled={!supportsKeyframePlacement && !!params.last_frame_image}
                  onClick={() => addAnchorInputRef.current?.click()}
                  className="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  title={
                    supportsKeyframePlacement
                      ? "Add another conditioning image (placed on a free frame; move it on the timeline)"
                      : params.last_frame_image
                        ? "This architecture conditions on the first and last frame only, and the last-frame slot is taken"
                        : "Add the last-frame conditioning image"
                  }
                >
                  + Image
                </button>
                <input
                  ref={addAnchorInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/jpg,image/webp"
                  className="hidden"
                  onChange={async (e) => {
                    const file = e.target.files?.[0];
                    e.target.value = "";
                    if (!file) return;
                    addAnchorImage(await toBase64(file));
                  }}
                />
              </div>
              <p className="text-xs text-gray-500">
                {supportsKeyframePlacement
                  ? "Every image here is one conditioning anchor pinned to one exact frame. Placement is on the Keyframes timeline below."
                  : "The input image conditions the first frame; the last-frame image conditions the end of the clip."}
              </p>
            </div>
          )}

          {activeInputAnchor === null
            ? renderImageDropZone({
                preview: inputImagePreview,
                onFile: processImageFile,
                onClear: handleClearInputImage,
                onEdit: handleEditImage,
                clearTitle: "Clear input image",
                emptyText: "Drag and drop an image here or use the file picker above",
              })
            : renderImageDropZone({
                preview: anchorImage(activeInputAnchor),
                onFile: async (file) => setAnchorImage(activeInputAnchor, await toBase64(file)),
                onClear: () => removeAnchor(activeInputAnchor),
                onEdit: () => setEditingExtraAnchor(activeInputAnchor),
                clearTitle:
                  activeInputAnchor.kind === "last"
                    ? "Remove the last-frame image"
                    : "Remove this keyframe",
                emptyText: "Drag and drop an image here or use the file picker above",
              })}
        </Card>
        )}

        {/* KEYFRAME PLACEMENT + the audio-conditioning lane, immediately after
            INPUT IMAGES and above the prompt: these say WHERE the images that
            were just uploaded land (and what soundtrack they are generated
            against), so they belong with them rather than in the Video card's
            sampler settings. Clip length and frame rate stay in the Video card
            -- the timeline reads them and reports the resulting placement. */}
        {isVideo && supportsKeyframePlacement && (
          <Card title="Keyframes">
            <MiniMaxH3KeyframeTimeline
              numFrames={params.num_frames ?? 124}
              frameRate={params.frame_rate ?? 24}
              inputImage={inputImagePreview}
              inputImageFrameIndex={params.input_image_frame_index ?? 0}
              onInputImageFrameIndexChange={(frameIndex) =>
                setParams({ ...params, input_image_frame_index: frameIndex })
              }
              keyframes={params.keyframes ?? []}
              onKeyframesChange={(keyframes) => setParams({ ...params, keyframes })}
              lastFrameImage={params.last_frame_image ?? null}
              onLastFrameImageChange={(dataUrl) =>
                setParams({ ...params, last_frame_image: dataUrl })
              }
              // The audio lane appears only where the architecture reads a
              // track; passing no handler is what hides it.
              inputAudio={supportsAudioConditioning ? inputAudioTrack : null}
              onInputAudioChange={
                supportsAudioConditioning ? setInputAudioTrack : undefined
              }
              // With the Audio toggle in the Video card off, nothing is muxed at
              // all; the lane says so rather than describing an output file that
              // will have no audio track.
              audioEnabled={params.audio_enable !== false}
              disabled={isGenerating}
            />
          </Card>
        )}

        {/* MiniMax-H3 ref2va references: with at least one set, submit routes
            the input image + timeline anchors to /generate/ref2vid instead of
            /generate/img2vid (that endpoint refuses this checkpoint by name).
            Same component and semantics as Txt2ImgPanel's -- order-is-semantic
            file lists, no strength/schedule knobs (see
            scratchpad/minimax_h3_conditioning_design.md §3.3 for why this is
            not a ControlNet-shaped UI). This panel is where both continuation
            routes are reachable, on different checkpoint variants -- the
            keyframe timeline above (fl2va) and the reference video slot below
            (ref2va) can look like the same feature; they are not. */}
        {isVideo && isRef2Va && (
          <>
            <div className="-mb-1 flex items-center gap-1 text-xs text-gray-500">
              <span>Whole-clip reference; the output is regenerated</span>
              <InlineHelp label="Video reference details">
                <p>
                  This reference conditions a whole clip rather than one boundary frame. It is laid out frame-contiguously with the generated span, so the full output is regenerated rather than preserved.
                </p>
                <p>
                  MiniMax calls this video continuation. It can be combined with an image anchor as keyframe completion.
                </p>
              </InlineHelp>
            </div>
            <MiniMaxH3ReferenceSelector
              value={h3References}
              onChange={setH3References}
              referenceImageSize={h3ReferenceImageSize}
              onReferenceImageSizeChange={setH3ReferenceImageSize}
              disabled={isGenerating}
            />
          </>
        )}

        {isAudio && (
          <Card
            title="Reference Audio"
            collapsible={true}
            defaultCollapsed={true}
            storageKey="img2img_reference_audio_collapsed"
            collapsedPreview={
              referenceAudioPreview ? (
                <span className="text-green-400 text-sm">✓ Audio loaded</span>
              ) : (
                <span className="text-gray-500 text-sm">No audio</span>
              )
            }
          >
            <div className="space-y-4">
              <div className="flex gap-2">
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleReferenceAudioUpload}
                  className="flex-1 block w-full text-sm text-gray-400
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-lg file:border-0
                    file:text-sm file:font-medium
                    file:bg-blue-600 file:text-white
                    hover:file:bg-blue-700
                    file:cursor-pointer cursor-pointer"
                />
                {referenceAudioPreview && (
                  <Button
                    onClick={handleClearReferenceAudio}
                    variant="secondary"
                    size="sm"
                    title="Clear reference audio"
                  >
                    Clear
                  </Button>
                )}
              </div>
              {referenceAudioPreview ? (
                <audio src={referenceAudioPreview} className="w-full" controls />
              ) : (
                <div className="bg-gray-800 rounded-lg border-2 border-dashed border-gray-600 py-6">
                  <p className="text-gray-500 text-center text-sm px-4">
                    Select a reference audio clip to cover. Duration is derived from the clip itself.
                  </p>
                </div>
              )}
            </div>
          </Card>
        )}

        {/* FLUX.2 Image Edit / Vision Encoder: Reference Images */}
        {(currentModelInfo?.model_info?.type === "flux2" || params.vision_encoder_path) && (
          <Card
            title={currentModelInfo?.model_info?.type === "flux2" ? "FLUX.2 Image Edit (Reference Images)" : "Vision Encoder (Reference Images)"}
            collapsible={true}
            defaultCollapsed={true}
            storageKey="img2img_ref_images_collapsed"
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
                    💡 {refImagePreviews.length}/10 images. {refImagePreviews.length < 10 ? 'Drop more images in the area above' : 'Max reached'}
                  </p>
                </div>
              )}
            </div>
          </Card>
        )}
            </>
          )}
          prompt={promptPanel}
          primaryDetails={(isVideo || isAudio) ? (
            <>

        {isVideo && (
          <Card title={`Video${loadedArchName ? ` (${loadedArchName})` : ""}`}>
            {/* Resolution, in the image models' Parameters-card shape: a
                labelled slider with a numeric entry beside it (common/Slider),
                laid out in the same two-column grid, and the same
                Absolute/Scale size mode -- scale derives width/height from the
                uploaded image's own dimensions, fitted to a canvas this
                architecture accepts (see deriveScaledSize / fitVideoCanvas:
                pixel_align, plus the max_pixel_hw envelope where there is
                one). */}
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-gray-300">
                    Size Mode
                  </label>
                  <div className="flex gap-2">
                    <Button
                      onClick={() => handleSizeModeChange("absolute")}
                      variant={sizeMode === "absolute" ? "primary" : "secondary"}
                      size="sm"
                    >
                      Absolute
                    </Button>
                    <Button
                      onClick={() => handleSizeModeChange("scale")}
                      variant={sizeMode === "scale" ? "primary" : "secondary"}
                      size="sm"
                      disabled={!inputImageSize}
                      title={!inputImageSize ? "Load an input image first" : ""}
                    >
                      Scale
                    </Button>
                  </div>
                </div>

                {sizeMode === "absolute" ? (
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
                ) : (
                  <div>
                    <Slider
                      label={`Scale (${params.width}x${params.height})`}
                      min={0.25}
                      max={4.0}
                      step={0.25}
                      value={scale}
                      onChange={(e) => handleScaleChange(parseFloat(e.target.value))}
                    />
                    {inputImageSize && (
                      <p className="text-xs text-gray-500 mt-1">
                        Input image: {inputImageSize.width}x{inputImageSize.height} ·{" "}
                        {videoCanvasRule(archCapabilities, loadedArch)}, so the scaled size
                        is fitted to the nearest canvas the model accepts.
                      </p>
                    )}
                  </div>
                )}
              </div>

              <Select
                label={videoFrameLabel(archCapabilities, loadedArch)}
                value={String(params.num_frames ?? 121)}
                onChange={(e) => setParams({ ...params, num_frames: parseInt(e.target.value) })}
                options={videoFrameOptions(archCapabilities, loadedArch, params.num_frames ?? null)}
              />

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Slider
                  label="Steps"
                  min={1}
                  max={100}
                  step={1}
                  value={params.num_inference_steps ?? 8}
                  onChange={(e) => setParams({ ...params, num_inference_steps: parseInt(e.target.value) })}
                />
                <Slider
                  label="Frame Rate (fps)"
                  min={1}
                  max={60}
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

            {/* The keyframe timeline and the optional last-frame image used to
                live here. Both are now anchors of the INPUT IMAGES card and
                its Keyframes timeline, directly above the prompt: the images
                and where they land belong together, and the last-frame slot is
                one of those anchors (the timeline's "last" chip, or the "Last
                frame" tab on an architecture that has the slot without
                per-anchor placement). This card keeps the clip settings the
                timeline reads: resolution, length, frame rate, steps. */}

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
                  <p>A keyframe is an exact-frame conditioning anchor retained at keyframe noise level throughout generation.</p>
                  <p>N steps run N-1 model evaluations (minimum 2). MiniMax does not publish a recommended step count.</p>
                </InlineHelp>
              </div>
            )}
          </Card>
        )}

        {isAudio && (
          <Card title="Audio Settings">
            <Select
              label="Mode"
              value={params.mode ?? "cover"}
              onChange={(e) => setParams({ ...params, mode: e.target.value as "cover" | "repaint" })}
              options={[
                { value: "cover", label: "Cover" },
                { value: "repaint", label: "Repaint" },
              ]}
            />

            {(params.mode ?? "cover") === "cover" ? (
              <Slider
                label="Cover Strength"
                min={0}
                max={1}
                step={0.05}
                value={params.cover_strength ?? 1.0}
                onChange={(e) => setParams({ ...params, cover_strength: parseFloat(e.target.value) })}
              />
            ) : (
              <div className="mt-2">
                <p className="text-xs text-gray-400 mb-2">
                  Only the [start, end) range of the reference audio is regenerated; the rest is kept unchanged.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Repaint start (s)</label>
                    <NumberInput
                      label="Repaint start (s)"
                      value={params.repaint_start ?? 0}
                      onCommit={(v) => setParams({ ...params, repaint_start: v })}
                      min={0}
                      step={0.1}
                      parse="float"
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Repaint end (s)</label>
                    <NumberInput
                      label="Repaint end (s)"
                      value={params.repaint_end ?? 0}
                      onCommit={(v) => {
                        const start = params.repaint_start ?? 0;
                        setParams({ ...params, repaint_end: v < start ? start : v });
                      }}
                      min={0}
                      step={0.1}
                      parse="float"
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
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
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
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
          </Card>
        )}
            </>
          ) : undefined}
        />

        {isAudio && visibility.lora && (
          <LoRASelector
            value={params.loras || []}
            onChange={(loras) => {
              console.log("[Img2Img] Audio LoRA onChange called with:", loras);
              setParams({ ...params, loras });
            }}
            disabled={isGenerating}
            storageKey="img2img_audio_lora_collapsed"
            simpleMode
          />
        )}

        {!isVideo && !isAudio && (<>
        {/* Img2Img Options: a single-open tabbed accordion (chrome shared via
            frontend/src/components/common/TabbedOptions.tsx). Every control
            below is unchanged from its original location (same param
            binding / handler / conditional reveal) -- only the container
            changed. See IMG2IMG_OPTIONS_TAB_KEYS / isImg2ImgOptionsTabActive /
            img2imgOptionsTabRender above. */}
        <TabbedOptions<Img2ImgParams>
          cardTitle="Img2Img Options"
          params={params}
          setParams={setParams}
          defaultParams={DEFAULT_PARAMS}
          tabs={IMG2IMG_OPTIONS_TABS.map((tab) => ({
            id: tab.id,
            label: tab.label,
            keys: IMG2IMG_OPTIONS_TAB_KEYS[tab.id],
            isActive: (p: Img2ImgParams) => isImg2ImgOptionsTabActive(tab.id, p),
            render: img2imgOptionsTabRender[tab.id],
          }))}
        />

        <Card title="Parameters">
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Slider
                label="Steps"
                min={1}
                max={150}
                step={1}
                value={params.steps}
                onChange={(e) => setParams({ ...params, steps: parseInt(e.target.value) })}
              />
              <Slider
                label="CFG Scale"
                min={0}
                max={30}
                step={0.5}
                value={params.cfg_scale}
                onChange={(e) => setParams({ ...params, cfg_scale: parseFloat(e.target.value) })}
              />
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-sm font-medium text-gray-300">
                  Size Mode
                </label>
                <div className="flex gap-2">
                  <Button
                    onClick={() => handleSizeModeChange("absolute")}
                    variant={sizeMode === "absolute" ? "primary" : "secondary"}
                    size="sm"
                  >
                    Absolute
                  </Button>
                  <Button
                    onClick={() => handleSizeModeChange("scale")}
                    variant={sizeMode === "scale" ? "primary" : "secondary"}
                    size="sm"
                    disabled={!inputImageSize}
                    title={!inputImageSize ? "Load an image first" : ""}
                  >
                    Scale
                  </Button>
                </div>
              </div>

              {sizeMode === "absolute" ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <Slider
                      label="Width"
                      min={64}
                      max={2048}
                      step={resolutionStep}
                      value={params.width}
                      onChange={(e) => setParams({ ...params, width: parseInt(e.target.value) })}
                    />
                    <Slider
                      label="Height"
                      min={64}
                      max={2048}
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
                              name="aspect_base_img2img"
                              value="width"
                              defaultChecked
                              className="w-3 h-3"
                            />
                            <span className="text-xs text-gray-300">Width</span>
                          </label>
                          <label className="flex items-center gap-1 cursor-pointer">
                            <input
                              type="radio"
                              name="aspect_base_img2img"
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
                              const baseOn = (document.querySelector('input[name="aspect_base_img2img"]:checked') as HTMLInputElement)?.value || 'width';
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
              ) : (
                <div>
                  <Slider
                    label={`Scale (${params.width}x${params.height})`}
                    min={0.25}
                    max={4.0}
                    step={0.25}
                    value={scale}
                    onChange={(e) => handleScaleChange(parseFloat(e.target.value))}
                  />
                  {inputImageSize && (
                    <p className="text-xs text-gray-500 mt-1">
                      Original: {inputImageSize.width}x{inputImageSize.height}
                    </p>
                  )}
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
                </label>
                <div className="flex gap-2">
                  <Input
                    type="number"
                    value={params.ancestral_seed}
                    onChange={(e) => setParams({ ...params, ancestral_seed: parseInt(e.target.value) })}
                    className="flex-1"
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
                    title="Reset to use main seed (-1)"
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

          </div>
        </Card>

        {visibility.lora && (
          <LoRASelector
            value={params.loras || []}
            onChange={(loras) => {
              console.log("[Img2Img] LoRA onChange called with:", loras);
              setParams({ ...params, loras });
            }}
            disabled={isGenerating}
            storageKey="img2img_lora_collapsed"
          />
        )}

        {visibility.controlnet && (
          <ControlNetSelector
            value={params.controlnets || []}
            onChange={(controlnets) => {
              console.log("[Img2Img] ControlNet onChange called with:", controlnets);
              setParams({ ...params, controlnets });
            }}
            disabled={isGenerating}
            storageKey="img2img_controlnet_collapsed"
            inputImagePreview={inputImagePreview}
          />
        )}

        {/* Loop Generation */}
        <LoopGenerationPanel
          config={loopGenerationConfig}
          onChange={setLoopGenerationConfig}
          mode="img2img"
          mainWidth={params.width || 1024}
          mainHeight={params.height || 1024}
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
                  id="preview_predicted_x0_img2img"
                  checked={params.preview_predicted_x0 || false}
                  onChange={(e) => setParams({ ...params, preview_predicted_x0: e.target.checked })}
                  className="rounded"
                />
                <label htmlFor="preview_predicted_x0_img2img" className="text-sm text-gray-300">
                  Preview Predicted x0
                </label>
              </div>

              {/* Live-preview decoder — only meaningful for AutoencoderKLFlux2-latent
                  models (FLUX.2 / Lens / Ideogram 4); hidden for architectures that
                  ignore preview_decoder (SD/SDXL, Z-Image, Anima, MiniT2I). */}
              {(currentModelInfo?.model_info?.type === "flux2"
                || currentModelInfo?.model_info?.type === "lens"
                || currentModelInfo?.model_info?.type === "ideogram4") && (
                <div className="flex items-center gap-2">
                  <label htmlFor="preview_decoder_img2img" className="text-sm text-gray-300">
                    Preview Decoder
                  </label>
                  <select
                    id="preview_decoder_img2img"
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

              {/* Use training model toggle (mirrors Txt2ImgPanel) */}
              <div className="flex items-center gap-2"
                   title={activeTraining
                     ? `Active: ${activeTraining.run_name ?? `run #${activeTraining.run_id}`} (step ${activeTraining.current_step ?? "?"})`
                     : "No active LoRA/Full-FT training"}>
                <input
                  type="checkbox"
                  id="use_training_model_img2img"
                  checked={useTrainingModel}
                  disabled={!activeTraining}
                  onChange={(e) => setUseTrainingModel(e.target.checked)}
                  className="rounded disabled:opacity-50"
                />
                <label htmlFor="use_training_model_img2img"
                       className={`text-sm ${activeTraining ? "text-gray-300" : "text-gray-500"}`}>
                  Use training model
                  {useTrainingModel && activeTraining && (
                    <span className="ml-1 text-xs text-emerald-400">
                      · {activeTraining.run_name ?? `run #${activeTraining.run_id}`} (step {activeTraining.current_step ?? "?"})
                    </span>
                  )}
                </label>
              </div>

              {useTrainingModel && (
                <div className="flex items-center gap-2 ml-6"
                     title="Save preview PNG to outputs/ and the gallery (tagged as training-preview)">
                  <input
                    type="checkbox"
                    id="save_preview_to_gallery_img2img"
                    checked={savePreviewToGallery}
                    onChange={(e) => setSavePreviewToGallery(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="save_preview_to_gallery_img2img" className="text-sm text-gray-300">
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
                        console.warn("[Img2Img] Preview video failed to load, clearing:", generatedVideo);
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
                        console.warn("[Img2Img] Preview audio failed to load, clearing:", generatedAudio);
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
                      // so a hot reload or a backend blip cannot discard a
                      // result that is still on disk (see helper).
                      imagePreviewGone(effectiveGeneratedImage ?? generatedImage, generatedImage).then((gone) => {
                        if (!gone) return;
                        console.warn("[Img2Img] Preview image failed to load, clearing:", generatedImage);
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

      {/* Image Editor Overlay */}
      {isEditingImage && inputImagePreview && (
        <ImageEditor
          imageUrl={inputImagePreview}
          onSave={handleSaveEditedImage}
          onClose={() => setIsEditingImage(false)}
        />
      )}

      {/* Image Editor Overlay for the EXTRA conditioning images (keyframes /
          last frame). Same editor and the same double-click entry point the
          input image has; it writes the edited data URL straight back into the
          slot the tab is bound to. */}
      {editingExtraAnchor && anchorImage(editingExtraAnchor) && (
        <ImageEditor
          imageUrl={anchorImage(editingExtraAnchor)!}
          onSave={(editedImageUrl) => {
            setAnchorImage(editingExtraAnchor, editedImageUrl);
            setEditingExtraAnchor(null);
          }}
          onClose={() => setEditingExtraAnchor(null)}
        />
      )}


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
    </ResizableColumns>
  );
}
