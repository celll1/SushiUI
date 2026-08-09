"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { usePathname, useSearchParams } from "next/navigation";
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
import ModelLoadSection from "../common/ModelLoadSection";
import LoRASelector from "../common/LoRASelector";
import ControlNetSelector from "../common/ControlNetSelector";
import GenerationQueue from "../common/GenerationQueue";
import GenerationLeadGrid from "../common/GenerationLeadGrid";
import ResizableColumns, {
  GENERATION_PREVIEW_QUEUE_SPLIT_KEY,
  GENERATION_WORKSPACE_SPLIT_KEY,
} from "../common/ResizableColumns";
import OutpaintPlacementCanvas, { OutpaintPlacementParams } from "./OutpaintPlacementCanvas";
import OutpaintTimeline from "./OutpaintTimeline";
import QuantizedGemmSelect from "./QuantizedGemmSelect";
import MiniMaxH3ReferenceSelector from "../common/MiniMaxH3ReferenceSelector";
import H3PromptAssist from "../common/H3PromptAssist";
import ImageViewer from "../common/ImageViewer";
import PostEditControls from "../common/PostEditControls";
import { PostEditState, NEUTRAL_POST_EDIT, buildFilterString } from "@/utils/postEdit";
import { usePostEditPreview } from "@/hooks/usePostEditPreview";
import {
  getSamplers,
  getScheduleTypes,
  getControlNets,
  generateOutpaint,
  generateOutpaintVideo,
  generateOutpaintAudio,
  getCurrentModel,
  cancelGeneration,
  getResultFilename,
  getResultPlaybackFilename,
  getResultSeed,
  getResultAncestralSeed,
  OutpaintParams as ApiOutpaintParams,
  OutpaintVideoParams,
  OutpaintAudioParams,
  LoRAConfig,
  ControlNetConfig,
  unetQuantizationOptions,
  normalizeUnetQuantization,
  transformerQuantizationLabel,
  videoOutpaintPlacements,
  outpaintVideoDefaultsForArch,
  fitVideoCanvas,
  videoCanvasRule,
  videoCanvasAxisBounds,
  videoCanvasExceedsEnvelope,
  isGenerationStalledError,
  archSupportsFeature,
} from "@/utils/api";
import { createH3ReferenceInventory, maybeTransformH3PromptForGeneration } from "@/utils/h3PromptAssist";
import { wsClient, CFGMetrics } from "@/utils/websocket";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";
import { previewStorageKeys, loadVideoPreview, saveVideoPreview, playbackUrlOf, loadAudioPreview, saveAudioPreview, saveImagePreview, clearVideoPreview, clearAudioPreview, clearImagePreview, outputExists, stripCacheBuster, withCacheBuster, imagePreviewGone } from "@/utils/previewStorage";
import { sendToPanel, sendImageToImg2Img, sendImageToInpaint, sendImageToUpscale, fetchUrlToFile, sendVideoToOutpaint, sendVideoToInpaint, sendVideoToReference, sendAudioToOutpaint, sendAudioToImg2Img } from "@/utils/sendHelpers";
import { fixFloatingPointParams } from "@/utils/numberUtils";
import { useStartup } from "@/contexts/StartupContext";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";
import SendToStudioButton from "../studio/SendToStudioButton";

// Extends the image OutpaintParams with the video (outpaint_vid, LTX-2.3)
// AND audio (outpaint_aud, ACE-Step 1.5 extend) fields. A single unified
// `params` object is used for all three modalities -- same convention as
// Txt2ImgPanel/Img2ImgPanel's isVideo/isAudio branches -- so most fields
// (prompt/negative_prompt/width/height/seed/vae_path/text_encoder_path/
// fbcache_*/spectrum_*) are reused as-is: their numeric defaults are
// IDENTICAL between OUTPAINT_DEFAULTS (derived from INPAINT_DEFAULTS),
// OUTPAINT_VIDEO_DEFAULTS (derived from VIDEO_GEN_DEFAULTS), and
// OUTPAINT_AUDIO_DEFAULTS (derived from AUD2AUD_DEFAULTS), and mean the same
// thing across whichever routes accept them (e.g. `guidance_scale` defaults
// to 1.0 in both the video and audio dicts).
//
// `blocks_to_swap` is the one exception: the IMAGE route gates it behind a
// separate `enable_block_swap` boolean (default magnitude 20, ignored unless
// the flag is set), while the VIDEO route has no such flag -- `blocks_to_swap`
// alone is the enable signal (0 = off) with a different default (0). Sharing
// the same field across modes would silently carry the image mode's value
// into a video request (or vice versa) on a tab switch, so video keeps its
// own `video_blocks_to_swap` field instead.
interface OutpaintPanelParams extends ApiOutpaintParams {
  // --- Video temporal outpaint (outpaint_vid, LTX-2.3) ---
  frame_rate?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  num_videos_per_prompt?: number;
  max_sequence_length?: number;
  audio_enable?: boolean;
  total_frames?: number;
  input_offset_frames?: number;
  input_trim_start_frames?: number;
  input_trim_end_frames?: number;
  outpaint_video_audio_mode?: "regenerate" | "preserve_input";
  // PANEL-LOCAL, never sent: which architecture `outpaint_video_audio_mode`
  // above was last resolved FOR. The backend default for that field is
  // per-architecture (`outpaint_video_arch_overlays`), so a value stored from
  // a session with a different model loaded is not a preference, it is a
  // stale default -- this marker is what tells the two apart, so switching
  // architecture re-resolves while a deliberate choice on the CURRENT
  // architecture survives reloads.
  outpaint_video_audio_mode_arch?: string | null;
  video_lossless?: boolean;
  video_blocks_to_swap?: number;
  // --- Audio temporal outpaint (outpaint_aud, ACE-Step 1.5 extend) ---
  lyrics?: string;
  inference_steps?: number;
  shift?: number;
  vocal_language?: string;
  total_duration?: number;
  input_offset_sec?: number;
  input_trim_start_sec?: number;
  input_trim_end_sec?: number;
}

// Mirrors backend OUTPAINT_DEFAULTS (backend/api/param_defaults.py) --
// derived from the full inpaint parameter set + the placement fields --
// plus OUTPAINT_VIDEO_DEFAULTS (derived from VIDEO_GEN_DEFAULTS) for the
// video branch and OUTPAINT_AUDIO_DEFAULTS (derived from AUD2AUD_DEFAULTS)
// for the audio branch. This object is a fallback only; on mount it is
// overridden by generationDefaults.outpaint / .outpaint_vid / .outpaint_aud
// fetched from GET /schema/generation-defaults (single source of truth),
// unless the user already has localStorage state.
const DEFAULT_PARAMS: OutpaintPanelParams = {
  prompt: "",
  negative_prompt: "",
  steps: 20,
  cfg_scale: 7.0,
  sampler: "euler",
  schedule_type: "uniform",
  seed: -1,
  ancestral_seed: -1,
  // Outpaint's default is full-strength generation of the surrounding canvas
  // (the placed region is preserved via the final pixel paste regardless).
  denoising_strength: 1.0,
  img2img_fix_steps: true,
  vae_drift_correction: false,
  mask_blur: 4,
  inpaint_full_res: false,
  inpaint_full_res_padding: 32,
  inpaint_fill_mode: "original",
  inpaint_fill_strength: 1.0,
  inpaint_blur_strength: 1.0,
  region_prompt: "",
  region_negative_prompt: "",
  region_prompt_strength: 1.0,
  region_prompt_method: "cfg",
  region_mask_feather: 0.0,
  seam_structure_strength: 0.0,
  seam_structure_depth: 6.0,
  seam_structure_end: 0.70,
  seam_structure_saliency: 2.0,
  seam_structure_max_area: 0.25,
  boundary_relax_strength: 0.0,
  boundary_relax_width: 3.0,
  boundary_relax_noise: 0.35,
  boundary_relax_full_until: 0.37,
  boundary_relax_end: 0.55,
  boundary_relax_paste: "feather",
  outpaint_controlnet_enable: false,
  outpaint_controlnet_mode: "crop_mask",
  outpaint_controlnet_model: "",
  outpaint_controlnet_detector: "canny",
  outpaint_controlnet_scale: 0.6,
  outpaint_controlnet_guidance_start: 0.0,
  outpaint_controlnet_guidance_end: 0.55,
  outpaint_controlnet_depth: 160,
  outpaint_controlnet_taper: 2.0,
  outpaint_controlnet_corner_radius_px: 0.0,
  outpaint_controlnet_corner_gate_radius_px: 0.0,
  outpaint_controlnet_corner_gate_min: 1.0,
  outpaint_pin_corner_relax_radius_px: 0.0,
  outpaint_pin_corner_relax_min: 1.0,
  outpaint_seam_membrane: false,
  outpaint_seam_membrane_band: 0,
  outpaint_seam_tone_strength: 0.0,
  outpaint_seam_tone_band: 0,
  outpaint_seam_offset_prop: 0.0,
  outpaint_boundary_color_strength: 0.25,
  outpaint_resample_count: 1,
  outpaint_jump_length: 4,
  outpaint_reference_strength: 0.0,
  outpaint_paste_feather_px: 24,
  outpaint_preserve_mode: "exact",
  outpaint_preview_unpinned_x0: false,
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
  nag_scale: 5.0,
  nag_tau: 3.5,
  nag_alpha: 0.25,
  nag_sigma_end: 3.0,
  nag_negative_prompt: "",
  attention_type: "normal",
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
  fbcache_cache_branch: 1,
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
  vision_encoder_path: null,
  vae_path: null,
  text_encoder_path: null,
  pid_sr_output: "4x",
  pid_use_gemma: false,
  pid_low_vram: false,
  pid_tile_native: 512,
  pid_tile_overlap_ratio: 0.25,
  pid_fast_large_decode: false,
  enable_block_swap: false,
  blocks_to_swap: 20,
  use_pinned_memory: false,
  block_swap_h2d_only: false,
  block_swap_ring_size: 2,
  loop_decode: "full",
  skip_gallery: false,
  // --- Placement (outpaint-only) ---
  canvas_width: 1536,
  canvas_height: 1536,
  place_x: 0,
  place_y: 0,
  place_width: 0,
  place_height: 0,
  input_crop_x: 0,
  input_crop_y: 0,
  input_crop_w: 0,
  input_crop_h: 0,
  outpaint_fill_mode: "replicate",
  // --- Video temporal outpaint (outpaint_vid, LTX-2.3) ---
  width: 768,
  height: 512,
  frame_rate: 24.0,
  num_inference_steps: 8,
  guidance_scale: 1.0,
  num_videos_per_prompt: 1,
  max_sequence_length: 1024,
  audio_enable: true,
  total_frames: 121,
  input_offset_frames: 0,
  input_trim_start_frames: 0,
  input_trim_end_frames: 0,
  outpaint_video_audio_mode: "regenerate",
  outpaint_video_audio_mode_arch: null,
  video_lossless: false,
  video_blocks_to_swap: 0,
  // --- Audio temporal outpaint (outpaint_aud, ACE-Step 1.5 extend) ---
  lyrics: "",
  inference_steps: 8,
  shift: 3.0,
  vocal_language: "en",
  total_duration: 60.0,
  input_offset_sec: 0.0,
  input_trim_start_sec: 0.0,
  input_trim_end_sec: 0.0,
};

// Image-tab outpaint options are grouped into a single-open tabbed accordion
// (see the "Outpaint Options" Card below) instead of a stack of <details>
// elements. Each tab owns a disjoint set of param keys, used both by its
// "reset to default" button and by its active-highlight predicate (which
// tabs currently have an enabled/non-neutral option, not merely "differs
// from default" -- see isOutpaintOptionsTabActive below).
type OutpaintOptionsTabId =
  | "controlnet"
  | "regional_prompt"
  | "seam"
  | "continuity"
  | "acceleration"
  | "post_process";

const OUTPAINT_OPTIONS_TABS: { id: OutpaintOptionsTabId; label: string }[] = [
  { id: "controlnet", label: "ControlNet" },
  { id: "regional_prompt", label: "Regional Prompt" },
  { id: "seam", label: "Seam（継ぎ目）" },
  { id: "continuity", label: "Continuity（連続性）" },
  { id: "acceleration", label: "Acceleration（高速化）" },
  { id: "post_process", label: "Post-process（色補正）" },
];

const OUTPAINT_OPTIONS_TAB_KEYS: Record<OutpaintOptionsTabId, (keyof OutpaintPanelParams)[]> = {
  controlnet: [
    "outpaint_controlnet_enable",
    "outpaint_controlnet_mode",
    "outpaint_controlnet_model",
    "outpaint_controlnet_detector",
    "outpaint_controlnet_scale",
    "outpaint_controlnet_guidance_start",
    "outpaint_controlnet_guidance_end",
    "outpaint_controlnet_depth",
    "outpaint_controlnet_taper",
    "outpaint_controlnet_corner_radius_px",
    "outpaint_controlnet_corner_gate_radius_px",
    "outpaint_controlnet_corner_gate_min",
    "outpaint_pin_corner_relax_radius_px",
    "outpaint_pin_corner_relax_min",
  ],
  regional_prompt: [
    "region_prompt",
    "region_negative_prompt",
    "region_prompt_strength",
    "region_prompt_method",
    "region_mask_feather",
  ],
  seam: [
    "outpaint_seam_membrane",
    "outpaint_seam_membrane_band",
    "outpaint_seam_tone_strength",
    "outpaint_seam_tone_band",
    "outpaint_seam_offset_prop",
    "outpaint_paste_feather_px",
    "outpaint_preserve_mode",
  ],
  continuity: [
    "seam_structure_strength",
    "seam_structure_depth",
    "seam_structure_end",
    "seam_structure_saliency",
    "seam_structure_max_area",
    "boundary_relax_strength",
    "boundary_relax_width",
    "boundary_relax_noise",
    "boundary_relax_full_until",
    "boundary_relax_end",
    "boundary_relax_paste",
    "outpaint_boundary_color_strength",
    "outpaint_resample_count",
    "outpaint_jump_length",
    "outpaint_reference_strength",
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
};

// "Active" means the group is currently doing something to the generation,
// not just "differs from DEFAULT_PARAMS". Two tabs are legitimately active
// out of the box because their own default is already "on":
// outpaint_paste_feather_px defaults to 24 (the tiled-VAE-style paste-band
// blend that removes the hard seam paste-line is enabled by default) and
// outpaint_boundary_color_strength defaults to 0.25 (In-loop Continuity's B1
// correction is enabled by default).
function isOutpaintOptionsTabActive(tabId: OutpaintOptionsTabId, params: OutpaintPanelParams): boolean {
  switch (tabId) {
    case "controlnet":
      return !!params.outpaint_controlnet_enable;
    case "regional_prompt":
      return !!(params.region_prompt?.trim() || params.region_negative_prompt?.trim());
    case "seam":
      return (
        (params.outpaint_seam_offset_prop ?? 0.0) > 0 ||
        (params.outpaint_seam_tone_strength ?? 0) > 0 ||
        !!params.outpaint_seam_membrane ||
        (params.outpaint_paste_feather_px ?? 24) > 0 ||
        (params.outpaint_preserve_mode ?? "exact") !== "exact"
      );
    case "continuity":
      return (
        (params.seam_structure_strength ?? 0) > 0 ||
        (params.boundary_relax_strength ?? 0) > 0 ||
        (params.outpaint_boundary_color_strength ?? 0.25) !== 0 ||
        (params.outpaint_resample_count ?? 1) > 1 ||
        (params.outpaint_reference_strength ?? 0) > 0
      );
    case "acceleration":
      return !!params.spectrum_enable || !!params.fbcache_enable;
    case "post_process":
      return (
        (params.color_flatten_strength ?? 0) > 0 ||
        !!params.flatten_in_loop ||
        !!params.vae_drift_correction
      );
    default:
      return false;
  }
}

const STORAGE_KEY = "outpaint_params";
const PREVIEW_STORAGE_KEY = "outpaint_preview";
// Image + video + audio preview keys for this panel. The three are mutually
// exclusive in storage (see utils/previewStorage.ts), so the newest result is
// the only one that can be restored.
const PREVIEW_KEYS = previewStorageKeys(PREVIEW_STORAGE_KEY);
const INPUT_IMAGE_STORAGE_KEY = "outpaint_input_image";

interface OutpaintPanelProps {
  onImageGenerated?: (imageUrl: string) => void;
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint" | "outpaint" | "upscale") => void;
}

export default function OutpaintPanel({ onTabChange, onImageGenerated }: OutpaintPanelProps = {}) {
  const { isBackendReady, generationDefaults, isVideo, isAudio, archCapabilities, resolveModality, modelInfoVersion } = useStartup();
  const [params, setParams] = useState<OutpaintPanelParams>(DEFAULT_PARAMS);
  const [generatedImageParams, setGeneratedImageParams] = useState<OutpaintPanelParams | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [generatedImageSeed, setGeneratedImageSeed] = useState<number | null>(null);
  const [generatedImageAncestralSeed, setGeneratedImageAncestralSeed] = useState<number | null>(null);
  // Preview zoom (image result only), mirrors Txt2ImgPanel/Img2ImgPanel/InpaintPanel.
  const [previewViewerOpen, setPreviewViewerOpen] = useState(false);
  // Client-side post-edit (brightness/saturation/flatten) for the current preview image.
  // Never sent to the backend; reset to neutral on each new generated image.
  const [postEdit, setPostEdit] = useState<PostEditState>({ ...NEUTRAL_POST_EDIT });
  // Color-flatten preview for the inline result image (b/s stay as CSS filter).
  const effectiveGeneratedImage = usePostEditPreview(generatedImage, postEdit.flatten);
  useEffect(() => {
    setPostEdit({ ...NEUTRAL_POST_EDIT });
  }, [generatedImage]);

  const [inputImage, setInputImage] = useState<File | null>(null);
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null);
  const [inputImageSize, setInputImageSize] = useState<{ width: number; height: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Video temporal outpaint (outpaint_vid) input clip + result. Not persisted
  // across reloads (unlike the image input, which round-trips through
  // IndexedDB via tempImageStorage.ts) -- an uploaded video File cannot be
  // cheaply stored the same way, and no existing modality (aud2aud's
  // reference clip either) persists its upload across a refresh.
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [videoDurationSec, setVideoDurationSec] = useState<number | null>(null);
  // The uploaded clip's own pixel size, read off the <video> element's
  // metadata. Only used by the video card's Scale size mode -- the same
  // "derive the output size from the input's dimensions" the image panels'
  // img2img control has; the request itself still carries width/height.
  const [inputVideoSize, setInputVideoSize] = useState<{ width: number; height: number } | null>(null);
  const [videoSizeMode, setVideoSizeMode] = useState<"absolute" | "scale">("absolute");
  const [videoScale, setVideoScale] = useState<number>(1.0);
  // BRIDGE placement only (an architecture whose video_constraints
  // .outpaint_placements contains "bridge"): the second clip, preserved at the
  // END of the timeline, with the generated span between the two.
  const [bridgeVideoFile, setBridgeVideoFile] = useState<File | null>(null);
  const [bridgeVideoPreviewUrl, setBridgeVideoPreviewUrl] = useState<string | null>(null);
  const [bridgeVideoDurationSec, setBridgeVideoDurationSec] = useState<number | null>(null);
  // MiniMax-H3 ref2va, extend_forward only: optional image references on top
  // of the automatic source-clip video reference the backend always adds.
  // Images only -- this endpoint has no reference_videos/reference_audios
  // field (the preserved clip IS the video reference).
  const [h3ReferenceImages, setH3ReferenceImages] = useState<File[]>([]);
  const [h3ReferenceImageSize, setH3ReferenceImageSize] = useState<"max" | "match">("max");
  const [generatedVideo, setGeneratedVideo] = useState<string | null>(null);
  // Playback source for the <video> element, when it differs from
  // generatedVideo (a video_lossless FFV1-in-mkv master no browser can
  // decode): its H.264 mp4 proxy. generatedVideo itself stays the master
  // for send-to/reference actions. Falls back to generatedVideo when null.
  const [generatedVideoPlaybackUrl, setGeneratedVideoPlaybackUrl] = useState<string | null>(null);
  const [generatedVideoInfo, setGeneratedVideoInfo] = useState<{ num_frames?: number; fps?: number; duration?: number } | null>(null);
  const [generatedVideoSeed, setGeneratedVideoSeed] = useState<number | null>(null);
  const [generatedVideoParams, setGeneratedVideoParams] = useState<OutpaintPanelParams | null>(null);

  // Audio temporal outpaint (outpaint_aud) input clip + result. The INPUT clip
  // is not persisted across reloads -- mirrors videoFile's rationale (an
  // uploaded File can't be cheaply round-tripped through localStorage/IndexedDB
  // the way the image input is). The RESULT is persisted, as a URL, under the
  // panel's audio preview key (see utils/previewStorage.ts).
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [audioDurationSec, setAudioDurationSec] = useState<number | null>(null);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [generatedAudioInfo, setGeneratedAudioInfo] = useState<{ duration?: number; sample_rate?: number } | null>(null);
  const [generatedAudioSeed, setGeneratedAudioSeed] = useState<number | null>(null);
  const [generatedAudioParams, setGeneratedAudioParams] = useState<OutpaintPanelParams | null>(null);

  const [progress, setProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);
  // Model list for the "Outpaint ControlNet (structure continuity)" section --
  // reuses the same /controlnets endpoint as ControlNetSelector (single
  // source of truth for available ControlNet checkpoints).
  const [outpaintControlNetModels, setOutpaintControlNetModels] = useState<Array<{ path: string; name: string }>>([]);
  const [isMounted, setIsMounted] = useState(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [currentModelInfo, setCurrentModelInfo] = useState<any>(null);
  // Keep this panel's copy of GET /models/current in step with the shared one.
  // modelInfoVersion only changes when the loaded model's identity actually
  // changes, so this costs one request per model change -- including changes
  // this page did not make (API, backend restart, another tab).
  useEffect(() => {
    if (modelInfoVersion === 0) return; // initial fetch happens on mount below
    getCurrentModel()
      .then(setCurrentModelInfo)
      .catch((error) => console.warn("[Outpaint] Failed to refresh model info", error));
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

  const [sendImage, setSendImage] = useState(true);
  const [sendPrompt, setSendPrompt] = useState(true);
  const [sendParameters, setSendParameters] = useState(true);

  const [developerMode, setDeveloperMode] = useState(false);
  const [showAdvancedCFG, setShowAdvancedCFG] = useState(false);
  // Mobile bottom-fixed generate bar expand/collapse (mirrors
  // Txt2ImgPanel/Img2ImgPanel/InpaintPanel's isMobileControlsOpen).
  const [isMobileControlsOpen, setIsMobileControlsOpen] = useState(true);

  // Lifted from OutpaintPlacementCanvas (was local child state) so panel-level
  // handlers (Send-to-Outpaint / new-image reset, "Reset Placement" button)
  // can also drive it -- see initializePlacementForImage / handleInputUpdate /
  // processImageFile. Deliberately NOT part of `params` -- must not persist to
  // the outpaint_params localStorage blob (it's a UI/session-only toggle).
  const [maintainAspect, setMaintainAspect] = useState(true);

  const pathname = usePathname();
  const searchParams = useSearchParams();
  const promptTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  // Set synchronously (before any async image decode) by any "new input
  // image" entry point (Send-to-Outpaint's outpaint_input_updated listener,
  // local upload/drop via processImageFile) so that if a same-tick
  // outpaint_params_updated event also fires (the Gallery sender dispatches
  // both), handleParamsUpdate knows to force the placement-reset sentinel
  // fields onto the incoming params instead of trusting whatever place_*
  // values happened to be saved under STORAGE_KEY. Cleared by
  // initializePlacementForImage once it actually re-initializes placement
  // for the new image (i.e. once the reset has "landed").
  const placementInitPendingRef = useRef(false);

  const isGeneratingRef = useRef(isGenerating);
  useEffect(() => {
    isGeneratingRef.current = isGenerating;
  }, [isGenerating]);

  // Revoke the uploaded clip's object URL on unmount / replacement to avoid
  // leaking blob URLs (createObjectURL persists until explicitly revoked).
  useEffect(() => {
    return () => {
      if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoPreviewUrl]);

  // Same rationale as above, for the audio branch's uploaded clip.
  useEffect(() => {
    return () => {
      if (audioPreviewUrl) URL.revokeObjectURL(audioPreviewUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioPreviewUrl]);

  const handleProgress = useCallback((step: number, total: number, message: string, preview?: string, _metrics?: CFGMetrics) => {
    if (isGeneratingRef.current) {
      setProgress(step);
      setTotalSteps(total);
      setProgressMessage(message || "");
      if (preview) {
        setPreviewImage(preview);
      }
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
      try {
        const modelInfo = await getCurrentModel();
        setCurrentModelInfo(modelInfo);
      } catch (error) {
        console.error("[Outpaint] Failed to load model info:", error);
      }

      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          setParams(fixFloatingPointParams(merged));
        } catch (error) {
          console.error("[Outpaint] Failed to load saved params:", error);
        }
      }

      const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
      if (savedPreview) {
        setGeneratedImage(savedPreview);
      }

      // Preview video (outpaint_vid result). Restored unconditionally: the
      // player is gated on `isVideo`, which arrives asynchronously from
      // useStartup(), so nothing renders until the loaded arch is known to be a
      // video arch. The URL is verified once the backend is ready (below).
      const savedVideo = loadVideoPreview(PREVIEW_KEYS);
      if (savedVideo) {
        setGeneratedVideo(savedVideo.url);
        setGeneratedVideoPlaybackUrl(playbackUrlOf(savedVideo));
        setGeneratedVideoInfo(savedVideo.info);
        setGeneratedVideoSeed(savedVideo.seed ?? null);
      }

      // Preview audio (outpaint_aud result). Same reasoning as the video above:
      // restored unconditionally because the <audio> render site is gated on
      // `isAudio` from useStartup(), which arrives asynchronously, so nothing
      // plays until the loaded arch is known to be an audio arch.
      const savedAudio = loadAudioPreview(PREVIEW_KEYS);
      if (savedAudio) {
        setGeneratedAudio(savedAudio.url);
        setGeneratedAudioInfo(savedAudio.info);
        setGeneratedAudioSeed(savedAudio.seed ?? null);
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
          console.error("[Outpaint] Failed to load input image:", error);
        }
      }

      const savedDeveloperMode = localStorage.getItem('developer_mode');
      if (savedDeveloperMode === 'true') {
        setDeveloperMode(true);
      }

      const savedShowAdvancedCFG = localStorage.getItem('show_advanced_cfg');
      if (savedShowAdvancedCFG === 'true') {
        setShowAdvancedCFG(true);
      }

      const savedAttentionType = localStorage.getItem('attention_type');
      if (savedAttentionType && (savedAttentionType === 'normal' || savedAttentionType === 'sage' || savedAttentionType === 'flash')) {
        setParams(prev => ({ ...prev, attention_type: savedAttentionType }));
      }

      setIsInitialLoad(false);
    };

    loadInitialData();
  }, []);

  // Reset torch.compile when developer mode is disabled
  useEffect(() => {
    if (!developerMode) {
      setParams(prev => (prev.use_torch_compile ? { ...prev, use_torch_compile: false } : prev));
    }
  }, [developerMode]);

  useEffect(() => {
    loadSamplers();
    loadScheduleTypes();
    loadOutpaintControlNetModels();
  }, []);

  const loadSamplers = async () => {
    try {
      const data = await getSamplers();
      setSamplers(data.samplers);
    } catch (error) {
      console.error("[Outpaint] Failed to load samplers:", error);
      setSamplers([
        { id: "euler", name: "Euler" },
        { id: "euler_ancestral", name: "Euler Ancestral" },
        { id: "dpm_pp_2m", name: "DPM++ 2M" },
        { id: "dpm_pp_2m_sde", name: "DPM++ 2M SDE" },
      ]);
    }
  };

  const loadScheduleTypes = async () => {
    try {
      const data = await getScheduleTypes();
      setScheduleTypes(data.schedule_types);
    } catch (error) {
      console.error("[Outpaint] Failed to load schedule types:", error);
      setScheduleTypes([
        { id: "uniform", name: "Uniform" },
        { id: "karras", name: "Karras" },
      ]);
    }
  };

  const loadOutpaintControlNetModels = async () => {
    try {
      const data = await getControlNets();
      setOutpaintControlNetModels(data.controlnets);
    } catch (error) {
      console.error("[Outpaint] Failed to load ControlNet models:", error);
    }
  };

  // Reload temp images once the backend is ready
  useEffect(() => {
    if (!isBackendReady) return;
    // Verify the restored preview image still exists before showing it, the
    // same way the video and audio branches below do. Non-`/outputs/` values (a
    // data: URL, a blob:, a path served from elsewhere) are left untouched:
    // they cannot go missing server-side and must never be stamped or
    // discarded. The cache-busting stamp is applied only to a URL that
    // verified, and it replaces any earlier stamp rather than appending.
    const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
    if (savedPreview && savedPreview.startsWith('/outputs/')) {
      const previewPath = stripCacheBuster(savedPreview);
      outputExists(previewPath).then((exists) => {
        if (!exists) {
          console.log("[Outpaint] Stored preview image is gone, clearing:", previewPath);
          clearImagePreview(PREVIEW_KEYS);
          setGeneratedImage(null);
          return;
        }
        setGeneratedImage(withCacheBuster(previewPath));
      });
    }
    // Verify the restored preview video still exists (outputs/ can be cleared,
    // or the run deleted from the gallery). No cache-busting timestamp -- an
    // .mp4 is large and its URL is stable.
    const savedVideo = loadVideoPreview(PREVIEW_KEYS);
    if (savedVideo) {
      outputExists(savedVideo.url).then((exists) => {
        if (!exists) {
          console.log("[Outpaint] Stored preview video is gone, clearing:", savedVideo.url);
          clearVideoPreview(PREVIEW_KEYS);
          setGeneratedVideo(null);
          setGeneratedVideoPlaybackUrl(null);
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
          console.log("[Outpaint] Stored preview audio is gone, clearing:", savedAudio.url);
          clearAudioPreview(PREVIEW_KEYS);
          setGeneratedAudio(null);
          setGeneratedAudioInfo(null);
          setGeneratedAudioSeed(null);
        }
      });
    }
    if (!inputImagePreview) {
      const savedInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (savedInputRef) {
        loadTempImage(savedInputRef).then((imageData) => {
          if (imageData) {
            setInputImagePreview(imageData);
            const img = new Image();
            img.onload = () => setInputImageSize({ width: img.width, height: img.height });
            img.src = imageData;
          }
        }).catch((error) => console.error("[Outpaint] Failed to reload input image:", error));
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isBackendReady]);

  // Listen for input image updates from the Gallery's "Send to Outpaint"
  // (fires while this panel is already mounted). Mirrors processImageFile's
  // reset so a brand-new image re-centers/re-sizes the placement rect
  // instead of keeping the previous image's stale place_*/canvas_* geometry
  // (initializePlacementForImage bails once place_width>0).
  useEffect(() => {
    const handleInputUpdate = () => {
      const newInput = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (newInput) {
        // Set synchronously, BEFORE the async loadTempImage() below, so that if
        // the sender's paired outpaint_params_updated event fires in the same
        // tick (see handleParamsUpdate), it reliably sees the flag set and
        // preserves this reset instead of letting the incoming saved params
        // clobber it with stale place_*/crop_* geometry from a previous image.
        // Gated on `newInput` so a missing/failed temp image can't leave the ref
        // stuck-true and later wrongly zero placement on a params-only send.
        placementInitPendingRef.current = true;
        loadTempImage(newInput).then((imageData) => {
          if (imageData) {
            setInputImagePreview(imageData);
            setParams(prev => ({
              ...prev,
              place_width: 0,
              place_height: 0,
              input_crop_x: 0,
              input_crop_y: 0,
              input_crop_w: 0,
              input_crop_h: 0,
            }));
            const img = new Image();
            img.onload = () => {
              setInputImageSize({ width: img.width, height: img.height });
            };
            img.src = imageData;
          }
        }).catch((error) => console.error("[Outpaint] Failed to load updated input image:", error));
      }
    };
    window.addEventListener("outpaint_input_updated", handleInputUpdate);
    return () => window.removeEventListener("outpaint_input_updated", handleInputUpdate);
  }, []);

  // Listen for a video clip sent from a result's "Send to Outpaint" (e.g.
  // Txt2Img/Img2Img/Outpaint's own outpaint_vid result). Transport is the
  // plain `/outputs/<filename>` URL (too large for base64/localStorage) --
  // fetch it into a real File so it flows through the same videoFile path an
  // upload does (mirrors processVideoFile's reset of the trim/offset fields).
  useEffect(() => {
    const handleVideoInputUpdate = async () => {
      const url = localStorage.getItem("outpaint_input_video");
      if (!url) return;
      try {
        const file = await fetchUrlToFile(url);
        setVideoPreviewUrl(prev => {
          if (prev) URL.revokeObjectURL(prev);
          return URL.createObjectURL(file);
        });
        setVideoFile(file);
        setVideoDurationSec(null);
        setParams(prev => ({ ...prev, input_offset_frames: 0, input_trim_start_frames: 0, input_trim_end_frames: 0 }));
      } catch (error) {
        console.error("[Outpaint] Failed to load sent video:", error);
      } finally {
        localStorage.removeItem("outpaint_input_video");
      }
    };
    window.addEventListener("outpaint_input_video_updated", handleVideoInputUpdate);
    return () => window.removeEventListener("outpaint_input_video_updated", handleVideoInputUpdate);
  }, []);

  // Listen for an audio clip sent from a result's "Send to Outpaint" (e.g.
  // Txt2Img/Img2Img/Outpaint's own outpaint_aud result). Mirrors the video
  // listener above (mirrors processAudioFile's reset of the trim/offset fields).
  useEffect(() => {
    const handleAudioInputUpdate = async () => {
      const url = localStorage.getItem("outpaint_input_audio");
      if (!url) return;
      try {
        const file = await fetchUrlToFile(url);
        setAudioPreviewUrl(prev => {
          if (prev) URL.revokeObjectURL(prev);
          return URL.createObjectURL(file);
        });
        setAudioFile(file);
        setAudioDurationSec(null);
        setParams(prev => ({ ...prev, input_offset_sec: 0, input_trim_start_sec: 0, input_trim_end_sec: 0 }));
      } catch (error) {
        console.error("[Outpaint] Failed to load sent audio:", error);
      } finally {
        localStorage.removeItem("outpaint_input_audio");
      }
    };
    window.addEventListener("outpaint_input_audio_updated", handleAudioInputUpdate);
    return () => window.removeEventListener("outpaint_input_audio_updated", handleAudioInputUpdate);
  }, []);

  // Listen for param updates dispatched from the Gallery / other panels
  useEffect(() => {
    const handleParamsUpdate = () => {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          // If a new-input-image event (Send-to-Outpaint / upload / drop) is
          // pending in this same dispatch batch, force the placement-reset
          // sentinel fields regardless of what the sender's saved params
          // blob contains -- otherwise a "params + image" send would restore
          // the SOURCE panel's stale place_*/crop_* geometry (wrong aspect
          // for the new image) and initializePlacementForImage's
          // place_width>0 guard would then never re-run.
          if (placementInitPendingRef.current) {
            merged.place_width = 0;
            merged.place_height = 0;
            merged.place_x = 0;
            merged.place_y = 0;
            merged.input_crop_x = 0;
            merged.input_crop_y = 0;
            merged.input_crop_w = 0;
            merged.input_crop_h = 0;
          }
          setParams(fixFloatingPointParams(merged));
        } catch (error) {
          console.error("[Outpaint] Failed to parse params update:", error);
        }
      }
    };
    window.addEventListener('outpaint_params_updated', handleParamsUpdate);
    return () => window.removeEventListener('outpaint_params_updated', handleParamsUpdate);
  }, []);

  // Initialize placement (canvas + centered rect) whenever a NEW input image
  // is loaded and the placement hasn't been customized yet (place_width===0,
  // the backend's "use native size" sentinel).
  const initializePlacementForImage = useCallback((width: number, height: number) => {
    setParams(prev => {
      if ((prev.place_width ?? 0) > 0) return prev; // already customized -- don't clobber
      const roundTo16 = (v: number) => Math.max(64, Math.round(v / 16) * 16);
      const canvasW = roundTo16(width * 1.5);
      const canvasH = roundTo16(height * 1.5);
      // This branch is the actual "landing" of a pending new-image reset
      // (see placementInitPendingRef) -- clear the flag now that placement
      // has been re-initialized for the new image.
      placementInitPendingRef.current = false;
      return {
        ...prev,
        canvas_width: canvasW,
        canvas_height: canvasH,
        place_width: width,
        place_height: height,
        place_x: Math.max(0, Math.round((canvasW - width) / 2)),
        place_y: Math.max(0, Math.round((canvasH - height) / 2)),
      };
    });
  }, []);

  useEffect(() => {
    if (inputImageSize) {
      initializePlacementForImage(inputImageSize.width, inputImageSize.height);
    }
  }, [inputImageSize, initializePlacementForImage]);

  // Save params to localStorage whenever they change
  useEffect(() => {
    if (isMounted && !isInitialLoad) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
    }
  }, [params, isMounted, isInitialLoad]);

  useEffect(() => {
    if (isMounted && generatedImage) {
      saveImagePreview(PREVIEW_KEYS, generatedImage);
    }
  }, [generatedImage, isMounted]);

  // Save preview video to localStorage whenever it changes. Only the URL, the
  // frame/fps/duration line and the seed are stored -- never the clip bytes.
  useEffect(() => {
    if (isMounted && generatedVideo) {
      saveVideoPreview(PREVIEW_KEYS, {
        url: generatedVideo,
        playbackUrl: generatedVideoPlaybackUrl || undefined,
        info: generatedVideoInfo,
        seed: generatedVideoSeed,
      });
    }
  }, [generatedVideo, generatedVideoPlaybackUrl, generatedVideoInfo, generatedVideoSeed, isMounted]);

  // Save preview audio to localStorage whenever it changes. Only the URL, the
  // duration/sample-rate line and the seed are stored -- never the audio bytes.
  useEffect(() => {
    if (isMounted && generatedAudio) {
      saveAudioPreview(PREVIEW_KEYS, {
        url: generatedAudio,
        info: generatedAudioInfo,
        seed: generatedAudioSeed,
      });
    }
  }, [generatedAudio, generatedAudioInfo, generatedAudioSeed, isMounted]);

  // Apply backend-fetched defaults when they arrive (only if no localStorage value exists).
  // Merges the image (outpaint), video (outpaint_vid), AND audio
  // (outpaint_aud) default dicts -- they share the unified `params` object
  // (see OutpaintPanelParams). Fields whose numeric defaults are IDENTICAL
  // across dicts (prompt/negative_prompt/seed/vae_path/text_encoder_path/
  // fbcache_*/spectrum_*/loras) are already covered by the `outpaint`
  // spread, so only the video/audio-SPECIFIC fields are pulled from
  // `outpaint_vid`/`outpaint_aud` here -- notably `blocks_to_swap`, which is
  // remapped to `video_blocks_to_swap` (see OutpaintPanelParams' doc comment
  // for why a blind spread of the raw backend dict would silently clobber
  // the image mode's `blocks_to_swap` default with the video route's
  // different one). `guidance_scale` is shared verbatim between the video
  // and audio dicts (both default 1.0), so either can supply it.
  useEffect(() => {
    if (!generationDefaults) return;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      const vidDefaults = (generationDefaults.outpaint_vid || {}) as Record<string, unknown>;
      const audDefaults = (generationDefaults.outpaint_aud || {}) as Record<string, unknown>;
      setParams(prev => ({
        ...DEFAULT_PARAMS,
        ...(generationDefaults.outpaint as Partial<ApiOutpaintParams>),
        width: vidDefaults.width as number ?? DEFAULT_PARAMS.width,
        height: vidDefaults.height as number ?? DEFAULT_PARAMS.height,
        frame_rate: vidDefaults.frame_rate as number ?? DEFAULT_PARAMS.frame_rate,
        num_inference_steps: vidDefaults.num_inference_steps as number ?? DEFAULT_PARAMS.num_inference_steps,
        guidance_scale: (vidDefaults.guidance_scale ?? audDefaults.guidance_scale) as number ?? DEFAULT_PARAMS.guidance_scale,
        num_videos_per_prompt: vidDefaults.num_videos_per_prompt as number ?? DEFAULT_PARAMS.num_videos_per_prompt,
        max_sequence_length: vidDefaults.max_sequence_length as number ?? DEFAULT_PARAMS.max_sequence_length,
        audio_enable: (vidDefaults.audio_enable as boolean) ?? DEFAULT_PARAMS.audio_enable,
        total_frames: vidDefaults.total_frames as number ?? DEFAULT_PARAMS.total_frames,
        input_offset_frames: vidDefaults.input_offset_frames as number ?? DEFAULT_PARAMS.input_offset_frames,
        input_trim_start_frames: vidDefaults.input_trim_start_frames as number ?? DEFAULT_PARAMS.input_trim_start_frames,
        input_trim_end_frames: vidDefaults.input_trim_end_frames as number ?? DEFAULT_PARAMS.input_trim_end_frames,
        outpaint_video_audio_mode: (vidDefaults.outpaint_video_audio_mode as "regenerate" | "preserve_input") ?? DEFAULT_PARAMS.outpaint_video_audio_mode,
        video_lossless: (vidDefaults.video_lossless as boolean) ?? DEFAULT_PARAMS.video_lossless,
        video_blocks_to_swap: vidDefaults.blocks_to_swap as number ?? DEFAULT_PARAMS.video_blocks_to_swap,
        // --- Audio temporal outpaint (outpaint_aud) ---
        lyrics: (audDefaults.lyrics as string) ?? DEFAULT_PARAMS.lyrics,
        inference_steps: audDefaults.inference_steps as number ?? DEFAULT_PARAMS.inference_steps,
        shift: audDefaults.shift as number ?? DEFAULT_PARAMS.shift,
        vocal_language: (audDefaults.vocal_language as string) ?? DEFAULT_PARAMS.vocal_language,
        total_duration: audDefaults.total_duration as number ?? DEFAULT_PARAMS.total_duration,
        input_offset_sec: audDefaults.input_offset_sec as number ?? DEFAULT_PARAMS.input_offset_sec,
        input_trim_start_sec: audDefaults.input_trim_start_sec as number ?? DEFAULT_PARAMS.input_trim_start_sec,
        input_trim_end_sec: audDefaults.input_trim_end_sec as number ?? DEFAULT_PARAMS.input_trim_end_sec,
      }));
    }
  }, [generationDefaults]);

  // Reload params/preview when navigating to /generate?tab=outpaint
  useEffect(() => {
    if (pathname === "/generate" && searchParams.get('tab') === 'outpaint' && isMounted) {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          setParams(fixFloatingPointParams(merged));
        } catch (error) {
          console.error("[Outpaint] Failed to reload params on navigation:", error);
        }
      }
    }
  }, [pathname, searchParams, isMounted]);

  const resetToDefault = () => {
    setParams(DEFAULT_PARAMS);
    localStorage.removeItem(STORAGE_KEY);
  };

  const processImageFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file');
      return;
    }

    setInputImage(file);
    // A brand-new image resets the placement so it re-centers at native size
    // (mirrors handleInputUpdate's reset for the Send-to-Outpaint path).
    placementInitPendingRef.current = true;
    setParams(prev => ({
      ...prev,
      place_width: 0,
      place_height: 0,
      input_crop_x: 0,
      input_crop_y: 0,
      input_crop_w: 0,
      input_crop_h: 0,
    }));

    const reader = new FileReader();
    reader.onload = async (event) => {
      const preview = event.target?.result as string;
      setInputImagePreview(preview);

      if (isMounted) {
        const oldInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
        if (oldInputRef) {
          await deleteTempImageRef(oldInputRef).catch(console.error);
        }
        try {
          const imageRef = await saveTempImage(preview);
          localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, imageRef);
        } catch (error) {
          console.error("[Outpaint] Failed to save input image:", error);
          localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, preview);
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
    if (file) processImageFile(file);
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
    if (file) processImageFile(file);
  };

  const handleClearInputImage = async () => {
    setInputImage(null);
    setInputImagePreview(null);
    setInputImageSize(null);
    setParams(prev => ({ ...prev, place_width: 0, place_height: 0 }));
    if (isMounted) {
      const inputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (inputRef) {
        await deleteTempImageRef(inputRef).catch(console.error);
      }
      localStorage.removeItem(INPUT_IMAGE_STORAGE_KEY);
    }
  };

  const handlePlacementChange = (patch: Partial<OutpaintPlacementParams>) => {
    setParams(prev => ({ ...prev, ...patch }));
  };

  // --- Video temporal outpaint (outpaint_vid) input clip handling ---

  const processVideoFile = (file: File) => {
    if (!file.type.startsWith('video/')) {
      alert('Please upload a valid video file');
      return;
    }
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl);
    }
    setVideoFile(file);
    setVideoDurationSec(null);
    // Forget the previous clip's dimensions so the metadata handler below
    // treats this upload as a new clip and re-defaults the canvas to 1x of it.
    setInputVideoSize(null);
    // A brand-new clip resets the placement so it doesn't inherit the
    // previous clip's stale offset/trim (mirrors processImageFile's
    // place_width=0 reset).
    setParams(prev => ({ ...prev, input_offset_frames: 0, input_trim_start_frames: 0, input_trim_end_frames: 0 }));
    setVideoPreviewUrl(URL.createObjectURL(file));
  };

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processVideoFile(file);
  };

  // Scale is relative to the input clip, and 1x is THE DEFAULT: temporal
  // outpaint extends a clip at its own resolution as a rule. `fitVideoCanvas`
  // resolves "the clip's own resolution" into a canvas the loaded architecture
  // actually accepts (pixel_align, and the max_pixel_hw envelope where it has
  // one) — 1x is not always literally reachable, and what the user sees when it
  // is not is the note under the control.
  const fitCanvas = (srcWidth: number, srcHeight: number, scaleValue: number) =>
    fitVideoCanvas(archCapabilities, loadedArchType, srcWidth, srcHeight, scaleValue);

  const handleVideoScaleChange = (newScale: number) => {
    setVideoScale(newScale);
    if (inputVideoSize) {
      const fitted = fitCanvas(inputVideoSize.width, inputVideoSize.height, newScale);
      setParams(prev => ({ ...prev, width: fitted.width, height: fitted.height }));
    }
  };

  const handleVideoSizeModeChange = (newMode: "absolute" | "scale") => {
    setVideoSizeMode(newMode);
    if (newMode === "scale" && inputVideoSize) {
      const fitted = fitCanvas(inputVideoSize.width, inputVideoSize.height, videoScale);
      setParams(prev => ({ ...prev, width: fitted.width, height: fitted.height }));
    }
  };

  const handleVideoLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const duration = e.currentTarget.duration;
    if (Number.isFinite(duration) && duration > 0) {
      setVideoDurationSec(duration);
    }
    const { videoWidth, videoHeight } = e.currentTarget;
    if (videoWidth > 0 && videoHeight > 0) {
      // Only a NEW clip re-defaults the canvas. `loadedmetadata` fires again
      // whenever the <video> remounts (tab switch, collapse/expand), and a
      // canvas the user chose must survive that -- processVideoFile clears
      // inputVideoSize, so an actual upload always counts as new.
      const isNewClip =
        !inputVideoSize
        || inputVideoSize.width !== videoWidth
        || inputVideoSize.height !== videoHeight;
      setInputVideoSize({ width: videoWidth, height: videoHeight });
      if (isNewClip) {
        // 1x ON THE CLIP THAT WAS JUST LOADED is the default canvas, whatever
        // width/height were carried over from an earlier run or another clip:
        // temporal outpaint extends at the source resolution as a rule, and any
        // other canvas has to be asked for. The note under the size control
        // states the resolved canvas either way.
        setVideoScale(1.0);
        const fitted = fitCanvas(videoWidth, videoHeight, 1.0);
        setParams(prev => ({ ...prev, width: fitted.width, height: fitted.height }));
      }
    }
  };

  const handleClearVideo = () => {
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl);
    }
    setVideoFile(null);
    setVideoPreviewUrl(null);
    setVideoDurationSec(null);
    setInputVideoSize(null);
    // Scale mode has nothing to scale from once the clip is gone.
    setVideoSizeMode("absolute");
    setParams(prev => ({ ...prev, input_offset_frames: 0, input_trim_start_frames: 0, input_trim_end_frames: 0 }));
  };

  const handleBridgeVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('video/')) {
      alert('Please upload a valid video file');
      return;
    }
    if (bridgeVideoPreviewUrl) URL.revokeObjectURL(bridgeVideoPreviewUrl);
    setBridgeVideoFile(file);
    setBridgeVideoDurationSec(null);
    setBridgeVideoPreviewUrl(URL.createObjectURL(file));
  };

  const handleClearBridgeVideo = () => {
    if (bridgeVideoPreviewUrl) URL.revokeObjectURL(bridgeVideoPreviewUrl);
    setBridgeVideoFile(null);
    setBridgeVideoPreviewUrl(null);
    setBridgeVideoDurationSec(null);
  };

  // --- Placement, from the loaded architecture's own conditioning rule ---
  //
  // `outpaint_placements` (GET /schema/arch-capabilities -> video_constraints)
  // is the single source: ["free"] means the clip can sit anywhere in the
  // timeline (LTX-2.3 conditions on an arbitrary latent index), while a
  // boundary-conditioned architecture (MiniMax-H3 conditions on the first
  // and/or last frame of the span it generates) lists only the placements it
  // can anchor. No arch name appears here.
  const loadedArchType = currentModelInfo?.model_info?.type as string | undefined;
  // Spectrum/FBCache: accepted-but-inert on an architecture whose sampler never
  // reads spectrum_enable/fbcache_enable (e.g. MiniMax-H3's FBCache was measured
  // and dropped rather than shipped). Hidden rather than shown-disabled, the
  // same leaf-control convention the other generation panels use for this pair.
  const supportsSpectrum = archSupportsFeature(archCapabilities, loadedArchType, "spectrum");
  const supportsFbcache = archSupportsFeature(archCapabilities, loadedArchType, "fbcache");
  const supportsNegativePrompt = !isAudio
    && archSupportsFeature(archCapabilities, loadedArchType, "negative_prompt");
  const outpaintPlacements = videoOutpaintPlacements(archCapabilities, loadedArchType);
  const boundaryPlacementOnly = !outpaintPlacements.includes("free");
  // MiniMax-H3 ref2va: direct variant check, matching Txt2ImgPanel/Img2ImgPanel
  // (there is no per-variant capability key). Reference conditioning on this
  // panel is offered only on extend_forward -- the ONLY row the backend's
  // ref2va partition/placement gate allows (see the gate in routes.py).
  const isRef2Va =
    loadedArchType === "minimax_h3" &&
    (currentModelInfo?.model_info?.variant as string | undefined) === "ref2va";

  // Backend rule (LTX-2.3): total_frames must satisfy (n-1) % 8 == 0, minimum 9.
  // On a boundary-conditioned architecture the grid binds the GENERATED span,
  // not this number (the preserved frames are pasted, never sampled), so the
  // total is left alone and the backend reports the effective length it
  // resolved to.
  const snapTotalFrames = (n: number): number =>
    boundaryPlacementOnly ? Math.max(2, Math.round(n)) : Math.max(9, n - (n % 8) + 1);

  // Nearest valid LTX-2.3 latent-frame pixel start: {0, 1, 9, 17, ..., 8k+1}.
  // A UX nicety only -- the backend re-snaps (and warns) server-side
  // regardless (see OUTPAINT_VIDEO_DEFAULTS.input_offset_frames).
  const snapLtxOffset = (raw: number): number => {
    const r = Math.round(raw);
    if (r <= 0) return 0;
    const k = Math.max(1, Math.round((r - 1) / 8));
    const candidates = [0, 1, 8 * k + 1];
    return candidates.reduce((best, c) => (Math.abs(r - c) < Math.abs(r - best) ? c : best), candidates[0]);
  };

  // Full length (frames) of the uploaded clip at the current frame_rate,
  // before trim -- the timeline's rawSegmentLength.
  const videoRawFrames = videoDurationSec != null
    ? Math.max(1, Math.round(videoDurationSec * (params.frame_rate ?? 24.0)))
    : 0;
  const videoPlacedFrames = Math.max(
    1,
    videoRawFrames - (params.input_trim_start_frames ?? 0) - (params.input_trim_end_frames ?? 0)
  );
  // The only two offsets a boundary-conditioned architecture accepts: flush
  // with the start of the timeline, or flush with its end.
  const boundaryEndOffset = Math.max(0, (params.total_frames ?? 0) - videoPlacedFrames);
  const snapBoundaryOffset = (raw: number): number =>
    Math.abs(raw - 0) <= Math.abs(raw - boundaryEndOffset) ? 0 : boundaryEndOffset;
  // The placement is DERIVED from the offset + whether a bridge clip is
  // present, so there is one source of truth for it (the same derivation the
  // backend does) rather than a second selector state that can disagree.
  const videoPlacement: "extend_forward" | "extend_backward" | "bridge" =
    bridgeVideoFile ? "bridge"
      : (params.input_offset_frames ?? 0) === 0 ? "extend_forward"
        : "extend_backward";
  const setVideoPlacement = (next: "extend_forward" | "extend_backward" | "bridge") => {
    if (next !== "bridge" && bridgeVideoFile) handleClearBridgeVideo();
    setParams(prev => ({
      ...prev,
      input_offset_frames: next === "extend_backward" ? boundaryEndOffset : 0,
    }));
  };

  // A `total_frames` the loaded architecture cannot serve at all -- below the
  // shortest span it can generate -- is replaced by that architecture's own
  // default from the SAME overlay chain the backend resolves from. Mirrors
  // Txt2ImgPanel's normalizeVideoFrames effect: otherwise a value carried over
  // from another architecture sits in the control and is sent anyway, only to
  // be bumped server-side with a warning.
  useEffect(() => {
    if (!archCapabilities || !loadedArchType || !generationDefaults) return;
    const constraints = archCapabilities.video_constraints?.[loadedArchType];
    if (!constraints) return;
    const archDefault = outpaintVideoDefaultsForArch(generationDefaults, loadedArchType)
      .total_frames as number | undefined;
    setParams(prev => (
      (prev.total_frames ?? 0) < constraints.min_frames && archDefault != null
        ? { ...prev, total_frames: archDefault }
        : prev
    ));
  }, [archCapabilities, generationDefaults, loadedArchType]);

  // The audio mode's DEFAULT is per-architecture too, and unlike total_frames
  // there is no invalid value to detect: "regenerate" is selectable
  // everywhere, it just means something different per architecture (on one
  // that generates audio only for the frames it generates, it leaves the
  // preserved span silent, which is why that architecture's default is
  // "preserve_input"). So the trigger is the ARCHITECTURE changing rather than
  // the value being out of range: re-resolve from the same overlay chain the
  // backend resolves from, and record which arch the answer belongs to. A
  // choice the user makes afterwards leaves the marker matching and is
  // therefore never overwritten, including across reloads.
  const archAudioMode =
    (outpaintVideoDefaultsForArch(generationDefaults, loadedArchType)
      .outpaint_video_audio_mode as "regenerate" | "preserve_input" | undefined)
    ?? DEFAULT_PARAMS.outpaint_video_audio_mode!;
  useEffect(() => {
    if (!generationDefaults || !loadedArchType) return;
    const resolved = outpaintVideoDefaultsForArch(generationDefaults, loadedArchType)
      .outpaint_video_audio_mode as "regenerate" | "preserve_input" | undefined;
    setParams(prev => (
      prev.outpaint_video_audio_mode_arch === loadedArchType
        ? prev
        : {
          ...prev,
          outpaint_video_audio_mode: resolved ?? prev.outpaint_video_audio_mode,
          outpaint_video_audio_mode_arch: loadedArchType,
        }
    ));
  }, [generationDefaults, loadedArchType]);

  // --- Audio temporal outpaint (outpaint_aud) input clip handling ---

  const processAudioFile = (file: File) => {
    if (!file.type.startsWith('audio/')) {
      alert('Please upload a valid audio file');
      return;
    }
    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl);
    }
    setAudioFile(file);
    setAudioDurationSec(null);
    // A brand-new clip resets the placement so it doesn't inherit the
    // previous clip's stale offset/trim (mirrors processVideoFile's reset).
    setParams(prev => ({ ...prev, input_offset_sec: 0, input_trim_start_sec: 0, input_trim_end_sec: 0 }));
    setAudioPreviewUrl(URL.createObjectURL(file));
  };

  const handleAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processAudioFile(file);
  };

  const handleAudioLoadedMetadata = (e: React.SyntheticEvent<HTMLAudioElement>) => {
    const duration = e.currentTarget.duration;
    if (Number.isFinite(duration) && duration > 0) {
      setAudioDurationSec(duration);
    }
  };

  const handleClearAudio = () => {
    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl);
    }
    setAudioFile(null);
    setAudioPreviewUrl(null);
    setAudioDurationSec(null);
    setParams(prev => ({ ...prev, input_offset_sec: 0, input_trim_start_sec: 0, input_trim_end_sec: 0 }));
  };

  // Backend rule: total_duration must be in (0, 240] seconds. A UX nicety
  // only -- the backend re-validates/clamps regardless (see
  // OUTPAINT_AUDIO_DEFAULTS.total_duration).
  const clampAudioTotalDuration = (n: number): number => Math.min(240, Math.max(0.1, n));

  const sendToTxt2Img = () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }
    if (sendPrompt) {
      const txt2imgParams = JSON.parse(localStorage.getItem("txt2img_params") || "{}");
      txt2imgParams.prompt = params.prompt;
      txt2imgParams.negative_prompt = params.negative_prompt;
      localStorage.setItem("txt2img_params", JSON.stringify(txt2imgParams));
    }
    if (sendParameters) {
      const txt2imgParams = JSON.parse(localStorage.getItem("txt2img_params") || "{}");
      txt2imgParams.steps = params.steps;
      txt2imgParams.cfg_scale = params.cfg_scale;
      txt2imgParams.sampler = params.sampler;
      txt2imgParams.schedule_type = params.schedule_type;
      txt2imgParams.seed = params.seed;
      txt2imgParams.width = params.canvas_width;
      txt2imgParams.height = params.canvas_height;
      localStorage.setItem("txt2img_params", JSON.stringify(txt2imgParams));
    }
    if (onTabChange) onTabChange("txt2img");
  };

  const sendToImg2Img = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }
    const sourceParams = generatedImageParams || params;
    if (sendImage) {
      try {
        await sendImageToImg2Img(generatedImage);
      } catch (error) {
        console.error("[Outpaint] Failed to send image to img2img:", error);
      }
    }
    sendToPanel(sourceParams as any, "img2img_params", {
      sendPrompt,
      sendParameters,
      includeDenoising: true,
      dispatchEvent: "img2img_params_updated",
    });
    if (onTabChange) onTabChange("img2img");
  };

  const sendToInpaintPanel = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }
    if (sendImage) {
      try {
        await sendImageToInpaint(generatedImage);
      } catch (error) {
        console.error("[Outpaint] Failed to send image to inpaint:", error);
      }
    }
    if (onTabChange) onTabChange("inpaint");
  };

  const sendToUpscale = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }
    if (sendImage) {
      try {
        await sendImageToUpscale(generatedImage);
      } catch (error) {
        console.error("[Outpaint] Failed to send image to upscale:", error);
      }
    }
    if (onTabChange) onTabChange("upscale");
  };

  // Outpaint's own outpaint_vid result -> Outpaint again (self-send = iterate
  // an extend, e.g. keep pushing the clip further out).
  const sendVideoResultToOutpaint = () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    sendVideoToOutpaint(generatedVideo);
    if (onTabChange) onTabChange("outpaint");
  };

  // Outpaint's own outpaint_vid result -> Inpaint's temporal inpaint clip input.
  const sendVideoResultToInpaint = () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    sendVideoToInpaint(generatedVideo);
    if (onTabChange) onTabChange("inpaint");
  };

  // Outpaint's own outpaint_vid result -> the ref2va reference track
  // (whole-clip conditioning, not a placement anchor -- see sendVideoToReference).
  const sendVideoResultToReference = () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    sendVideoToReference(generatedVideo);
    if (onTabChange) onTabChange("txt2img");
  };

  // Outpaint's own outpaint_aud result -> Outpaint again (self-send = iterate
  // an extend).
  const sendAudioResultToOutpaint = () => {
    if (!generatedAudio) {
      alert("No audio to send");
      return;
    }
    sendAudioToOutpaint(generatedAudio);
    if (onTabChange) onTabChange("outpaint");
  };

  // Outpaint's own outpaint_aud result -> Img2Img as the aud2aud reference clip.
  const sendAudioResultToImg2Img = () => {
    if (!generatedAudio) {
      alert("No audio to send");
      return;
    }
    sendAudioToImg2Img(generatedAudio);
    if (onTabChange) onTabChange("img2img");
  };

  const { addToQueue, startNextInQueue, completeCurrentItem, failCurrentItem, currentItem, queue } = useGenerationQueue();

  const [visibility] = useState({ lora: true, controlnet: true });

  // Add generation request to queue. Three modality branches: image
  // (outpaint), video (outpaint_vid, LTX-2.3), and audio (outpaint_aud,
  // ACE-Step 1.5 extend) -- mutually exclusive on the loaded model's
  // modality, matching Txt2ImgPanel/Img2ImgPanel's isVideo/isAudio dispatch.
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

    const { replaceWildcardsInPrompt } = await import("@/utils/wildcardStorage");
    let processedPrompt = await replaceWildcardsInPrompt(params.prompt);
    const processedNegativePrompt = supportsNegativePrompt
      ? await replaceWildcardsInPrompt(params.negative_prompt || "")
      : "";

    if (videoMode && modality.modelInfo?.type === "minimax_h3") {
      try {
        const assisted = await maybeTransformH3PromptForGeneration({
          prompt: processedPrompt,
          mode: modality.modelInfo?.variant === "ref2va" ? "ref2va" : "t2va",
          durationSeconds: (params.total_frames ?? 121) / (params.frame_rate ?? 24),
          references: createH3ReferenceInventory({
            pictures: h3ReferenceImages.length,
            videos: 1 + (bridgeVideoFile ? 1 : 0),
          }),
        });
        processedPrompt = assisted.prompt;
      } catch (error: any) {
        alert(error?.message || "MiniMax H3 Prompt Assist failed");
        return;
      }
    }

    // Video mode: a video model (LTX-2.3) is loaded -> enqueue an
    // outpaint_vid item using the uploaded clip. No loop-generation (matches
    // Upscale + the video/audio branches of the merged txt2img/img2img panels).
    if (videoMode) {
      if (!videoFile) {
        alert("Please upload an input video clip");
        return;
      }
      const videoParams: OutpaintVideoParams = {
        prompt: processedPrompt,
        negative_prompt: processedNegativePrompt,
        width: params.width,
        height: params.height,
        total_frames: params.total_frames,
        frame_rate: params.frame_rate,
        num_inference_steps: params.num_inference_steps,
        guidance_scale: params.guidance_scale,
        seed: params.seed,
        num_videos_per_prompt: params.num_videos_per_prompt,
        max_sequence_length: params.max_sequence_length,
        audio_enable: params.audio_enable,
        input_offset_frames: params.input_offset_frames,
        input_trim_start_frames: params.input_trim_start_frames,
        input_trim_end_frames: params.input_trim_end_frames,
        outpaint_video_audio_mode: params.outpaint_video_audio_mode,
        video_lossless: params.video_lossless,
        blocks_to_swap: params.video_blocks_to_swap,
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
        vae_path: params.vae_path,
        text_encoder_path: params.text_encoder_path,
        // Only "int8" is applied on LTX-2.3 (one-time in-place conversion of the
        // video DiT); other values warn and are ignored server-side.
        unet_quantization: params.unet_quantization,
        // ltx2 is in quantized_linear_archs, so the QuantizedGemmSelect control
        // is rendered for a loaded LTX-2.3 model and must actually be sent.
        quantized_gemm_mode: params.quantized_gemm_mode,
        // MiniMax-H3 ref2va, extend_forward only. Sent unconditionally (the
        // backend ignores it when there is nothing to size); the images
        // themselves ride on the queue item like inputVideo/bridgeVideo.
        reference_image_size: h3ReferenceImageSize,
      };
      addToQueue({
        type: "outpaint_vid",
        params: videoParams as any,
        inputVideo: videoFile,
        // Bridge placement only; undefined otherwise, and the backend refuses
        // it on an architecture that has no bridge placement.
        bridgeVideo: bridgeVideoFile || undefined,
        // MiniMax-H3 ref2va, extend_forward only; empty otherwise (the
        // backend gate refuses reference_images on every other row, so an
        // empty list is a no-op request there rather than a wrong one).
        referenceImages: isRef2Va && videoPlacement === "extend_forward" ? h3ReferenceImages : undefined,
        prompt: processedPrompt,
      });
      return;
    }

    // Audio mode: an audio model (ACE-Step 1.5) is loaded -> enqueue an
    // outpaint_aud item using the uploaded reference clip. No loop-generation
    // (matches Upscale + the video/audio branches of the merged txt2img/img2img
    // panels). No negative_prompt (the audio route has no such field).
    if (audioMode) {
      if (!audioFile) {
        alert("Please upload a reference audio clip");
        return;
      }
      const audioParams: OutpaintAudioParams = {
        prompt: processedPrompt,
        lyrics: params.lyrics,
        seed: params.seed,
        inference_steps: params.inference_steps,
        guidance_scale: params.guidance_scale,
        shift: params.shift,
        vocal_language: params.vocal_language,
        loras: params.loras,
        total_duration: params.total_duration,
        input_offset_sec: params.input_offset_sec,
        input_trim_start_sec: params.input_trim_start_sec,
        input_trim_end_sec: params.input_trim_end_sec,
        // Weight-only quantization (both axes). The panel controls are rendered
        // from arch capabilities, and `acestep` is now in runtime_int8_archs +
        // quantized_linear_archs, so these must be carried into the audio
        // params or the UI value is silently dropped.
        unet_quantization: params.unet_quantization,
        quantized_gemm_mode: params.quantized_gemm_mode,
      };
      addToQueue({
        type: "outpaint_aud",
        params: audioParams as any,
        inputAudio: audioFile,
        prompt: processedPrompt,
      });
      return;
    }

    if (!inputImagePreview) {
      alert("Please upload an input image");
      return;
    }

    // NOTE: Loop Generation is intentionally out of scope for the Outpaint
    // tab (all phases) -- matches Upscale + the video/audio branches of the
    // merged txt2img/img2img panels. No stepParams / loop group wiring here.
    addToQueue({
      type: "outpaint",
      params: {
        ...params,
        prompt: processedPrompt,
        negative_prompt: processedNegativePrompt,
      },
      inputImage: inputImagePreview,
      prompt: processedPrompt,
    });
  };

  const processQueueRef = useRef<() => Promise<void>>();

  const processQueue = useCallback(async () => {
    if (isGenerating) return;

    const nextItem = startNextInQueue();
    if (!nextItem || (nextItem.type !== "outpaint" && nextItem.type !== "outpaint_vid" && nextItem.type !== "outpaint_aud")) return;

    // Video branch: outpaint_vid item (LTX-2.3). The queued input clip is a
    // File (see inputVideo on QueueItem). Produces an .mp4 and renders a
    // <video> instead of an <img>. No loop-generation handling.
    if (nextItem.type === "outpaint_vid") {
      setIsGenerating(true);
      setProgress(0);
      setProgressMessage("");
      setTotalSteps((nextItem.params as OutpaintVideoParams).num_inference_steps || 8);
      setPreviewImage(null);
      setGeneratedImage(null);
      setGeneratedVideo(null);
      setGeneratedVideoPlaybackUrl(null);
      setGeneratedVideoInfo(null);
      setGeneratedVideoSeed(null);
      setGeneratedAudio(null);
      setGeneratedAudioInfo(null);
      setGeneratedAudioSeed(null);
      try {
        const clip = nextItem.inputVideo;
        if (!clip) {
          throw new Error("No input video available for video outpaint generation");
        }
        const result = await generateOutpaintVideo(
          nextItem.params as OutpaintVideoParams, clip, nextItem.bridgeVideo, nextItem.referenceImages);
        const videoUrl = `/outputs/${getResultFilename(result)}`;
        const videoPlaybackUrl = `/outputs/${getResultPlaybackFilename(result)}`;
        setGeneratedVideo(videoUrl);
        setGeneratedVideoPlaybackUrl(videoPlaybackUrl !== videoUrl ? videoPlaybackUrl : null);
        setGeneratedVideoSeed(getResultSeed(result));
        setGeneratedVideoParams(nextItem.params as OutpaintPanelParams);
        setGeneratedVideoInfo({
          num_frames: result.image?.num_frames,
          fps: result.image?.fps,
          duration: result.image?.duration,
        });
        if (onImageGenerated) onImageGenerated(videoUrl);
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        completeCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
      } catch (error: any) {
        console.error("[Outpaint] Video generation failed:", error);
        alert(isGenerationStalledError(error)
          ? error.message
          : `Video outpaint generation failed: ${error?.response?.data?.detail || error?.message || "Unknown error"}`);
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        failCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
      }
      return;
    }

    // Audio branch: outpaint_aud item (ACE-Step 1.5 extend). The queued
    // input clip is a File (see inputAudio on QueueItem). Produces a .flac
    // and renders an <audio> instead of an <img>. No loop-generation handling.
    if (nextItem.type === "outpaint_aud") {
      setIsGenerating(true);
      setProgress(0);
      setProgressMessage("");
      setTotalSteps((nextItem.params as OutpaintAudioParams).inference_steps || 8);
      setPreviewImage(null);
      setGeneratedImage(null);
      // An audio run supersedes any image/video result still on screen; the
      // stored preview is only replaced once this run actually succeeds.
      setGeneratedAudio(null);
      setGeneratedAudioInfo(null);
      setGeneratedAudioSeed(null);
      setGeneratedVideo(null);
      setGeneratedVideoPlaybackUrl(null);
      setGeneratedVideoInfo(null);
      setGeneratedVideoSeed(null);
      try {
        const referenceAudio = nextItem.inputAudio;
        if (!referenceAudio) {
          throw new Error("No reference audio available for audio outpaint generation");
        }
        const result = await generateOutpaintAudio(nextItem.params as OutpaintAudioParams, referenceAudio);
        const audioUrl = `/outputs/${result.image.filename}`;
        setGeneratedAudio(audioUrl);
        setGeneratedAudioSeed(getResultSeed(result));
        setGeneratedAudioParams(nextItem.params as OutpaintPanelParams);
        setGeneratedAudioInfo({
          duration: result.image?.duration,
          sample_rate: result.image?.sample_rate,
        });
        if (onImageGenerated) onImageGenerated(audioUrl);
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        completeCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
      } catch (error: any) {
        console.error("[Outpaint] Audio generation failed:", error);
        alert(isGenerationStalledError(error)
          ? error.message
          : `Audio outpaint generation failed: ${error?.response?.data?.detail || error?.message || "Unknown error"}`);
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        failCurrentItem();
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
      }
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    setProgressMessage("");
    const denoisingStrength = (nextItem.params as ApiOutpaintParams).denoising_strength ?? 1.0;
    const actualSteps = Math.ceil(((nextItem.params as ApiOutpaintParams).steps || 20) * denoisingStrength);
    setTotalSteps(actualSteps);
    setPreviewImage(null);
    setGeneratedImage(null);
    // An image run supersedes any video/audio result still on screen. The
    // stored previews are left alone until the image actually succeeds (see the
    // save effects), so a failed run does not throw away the last good result.
    setGeneratedVideo(null);
    setGeneratedVideoPlaybackUrl(null);
    setGeneratedVideoInfo(null);
    setGeneratedVideoSeed(null);
    setGeneratedAudio(null);
    setGeneratedAudioInfo(null);
    setGeneratedAudioSeed(null);

    try {
      const itemParams = nextItem.params as ApiOutpaintParams;
      const apiParams: ApiOutpaintParams = {
        ...itemParams,
        developer_mode: developerMode,
        // Reset advanced CFG params if the section is collapsed (mirrors
        // InpaintPanel/Img2ImgPanel behavior).
        cfg_schedule_type: !showAdvancedCFG ? "constant" : itemParams.cfg_schedule_type,
        cfg_rescale_snr_alpha: !showAdvancedCFG ? 0.0 : itemParams.cfg_rescale_snr_alpha,
        dynamic_threshold_percentile: !showAdvancedCFG ? 0.0 : itemParams.dynamic_threshold_percentile,
      };

      const result = await generateOutpaint(apiParams, nextItem.inputImage!);
      const imageUrl = result.success ? `/outputs/${getResultFilename(result)}` : "";

      if (result.success) {
        const resultSeed = getResultSeed(result);
        const resultAncestralSeed = getResultAncestralSeed(result);
        setGeneratedImage(imageUrl);
        setGeneratedImageSeed(resultSeed);
        setGeneratedImageAncestralSeed(resultAncestralSeed);
        setPreviewImage(null);
        setGeneratedImageParams({
          ...itemParams,
          seed: resultSeed,
          ancestral_seed: resultAncestralSeed ?? -1,
        });

        if (onImageGenerated) {
          onImageGenerated(imageUrl);
        }
        if (isMounted) {
          saveImagePreview(PREVIEW_KEYS, imageUrl);
        }

        setIsGenerating(false);
        setProgress(0);
        completeCurrentItem();
      } else {
        throw new Error("Outpaint generation did not succeed");
      }

      setTimeout(() => {
        if (processQueueRef.current) processQueueRef.current();
      }, 100);
    } catch (error: any) {
      console.error("[Outpaint] Generation failed:", error);
      alert(isGenerationStalledError(error)
        ? error.message
        : `Outpaint generation failed: ${error?.response?.data?.detail || error?.message || "Unknown error"}`);

      setIsGenerating(false);
      setProgress(0);
      failCurrentItem();

      setTimeout(() => {
        if (processQueueRef.current) processQueueRef.current();
      }, 100);
    }
  }, [isGenerating, startNextInQueue, completeCurrentItem, failCurrentItem, developerMode, showAdvancedCFG, isMounted, onImageGenerated]);

  processQueueRef.current = processQueue;

  useEffect(() => {
    const hasPendingItems = queue.some(item => item.status === "pending" && (item.type === "outpaint" || item.type === "outpaint_vid" || item.type === "outpaint_aud"));
    const isCurrentItemNull = currentItem === null;
    if (hasPendingItems && isCurrentItemNull && !isGenerating) {
      processQueue();
    }
  }, [queue, currentItem, isGenerating, processQueue]);

  const placementParams: OutpaintPlacementParams = {
    canvas_width: params.canvas_width ?? 1536,
    canvas_height: params.canvas_height ?? 1536,
    place_x: params.place_x ?? 0,
    place_y: params.place_y ?? 0,
    place_width: params.place_width || inputImageSize?.width || 1024,
    place_height: params.place_height || inputImageSize?.height || 1024,
    input_crop_x: params.input_crop_x ?? 0,
    input_crop_y: params.input_crop_y ?? 0,
    input_crop_w: params.input_crop_w ?? 0,
    input_crop_h: params.input_crop_h ?? 0,
    outpaint_fill_mode: params.outpaint_fill_mode || "replicate",
    mask_blur: params.mask_blur ?? 4,
  };

  // Per-tab body content for the "Outpaint Options" TabbedOptions instance
  // below. Every control here is unchanged from its original inline
  // location (same param binding / handler / conditional reveal) -- only
  // the container changed (see OUTPAINT_OPTIONS_TABS / OUTPAINT_OPTIONS_TAB_KEYS
  // / isOutpaintOptionsTabActive above, and frontend/src/components/common/
  // TabbedOptions.tsx for the shared chrome).
  const outpaintOptionsTabRender: Record<OutpaintOptionsTabId, () => JSX.Element> = {
    controlnet: () => (
      <div className="space-y-3">
        {/* Outpaint ControlNet (structure continuity): synthesizes an
            edge-extrapolation control image (canny/lineart) from the
            placed region and conditions the generated surround with it,
            tapering out with distance/schedule progress. See
            backend/api/routes.py generate_outpaint outpaint_controlnet_*
            Form params. */}
        <p className="text-xs text-gray-500">
          SD/SDXL only. Extrapolates edges from the placed region into the generated surround using a ControlNet, tapering out with distance from the seam.
          Mutually exclusive with a user-supplied ControlNet/LLLite above; forces Boundary Relax Paste Mode to Exact and disables Seam Structure Continuity while active.
        </p>
        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="outpaint_controlnet_enable"
            checked={params.outpaint_controlnet_enable ?? false}
            onChange={(e) => setParams({ ...params, outpaint_controlnet_enable: e.target.checked })}
            disabled={isGenerating}
            className="w-4 h-4"
          />
          <label htmlFor="outpaint_controlnet_enable" className="text-sm text-gray-300">
            Enable
          </label>
        </div>
        {params.outpaint_controlnet_enable && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Select
              label="Mode"
              options={[
                { value: "edge_extrapolate", label: "Edge extrapolate (anytest)" },
                { value: "crop_mask", label: "Crop mask (trained outpaint CN)" },
              ]}
              value={params.outpaint_controlnet_mode || "edge_extrapolate"}
              onChange={(e) => setParams({ ...params, outpaint_controlnet_mode: e.target.value })}
            />
            <p className="text-xs text-gray-500 sm:col-span-2">
              {params.outpaint_controlnet_mode === "crop_mask"
                ? "Crop mask: builds the trained 4-channel crop+mask conditioning. Requires a ControlNet trained with conditioning_mode=outpaint (4-ch diffusers directory). Detector/depth/taper do not apply."
                : "Edge extrapolate: detects and extrapolates boundary-crossing edges over a guessed geometry (any structure ControlNet)."}
            </p>
            <Select
              label="ControlNet Model"
              options={
                outpaintControlNetModels.length > 0
                  ? outpaintControlNetModels.map((m) => ({ value: m.path, label: m.name }))
                  : [{ value: "", label: "No ControlNet models found" }]
              }
              value={params.outpaint_controlnet_model || ""}
              onChange={(e) => setParams({ ...params, outpaint_controlnet_model: e.target.value })}
              disabled={isGenerating || outpaintControlNetModels.length === 0}
            />
            <Select
              label="Detector"
              options={[
                { value: "canny", label: "Canny" },
                { value: "lineart", label: "Lineart" },
                { value: "lineart_anime", label: "Lineart (Anime)" },
              ]}
              value={params.outpaint_controlnet_detector || "canny"}
              onChange={(e) => setParams({ ...params, outpaint_controlnet_detector: e.target.value })}
            />
            <Slider
              label="Conditioning Scale"
              min={0.0}
              max={1.5}
              step={0.05}
              value={params.outpaint_controlnet_scale ?? 0.6}
              onChange={(e) => setParams({ ...params, outpaint_controlnet_scale: parseFloat(e.target.value) })}
            />
            <Slider
              label="Guidance Start (schedule progress)"
              min={0}
              max={1}
              step={0.05}
              value={params.outpaint_controlnet_guidance_start ?? 0.0}
              onChange={(e) => setParams({ ...params, outpaint_controlnet_guidance_start: parseFloat(e.target.value) })}
            />
            <Slider
              label="Guidance End (schedule progress)"
              min={0}
              max={1}
              step={0.05}
              value={params.outpaint_controlnet_guidance_end ?? 0.55}
              onChange={(e) => setParams({ ...params, outpaint_controlnet_guidance_end: parseFloat(e.target.value) })}
            />
            <Slider
              label="Extrapolation Depth (px)"
              min={32}
              max={320}
              step={16}
              value={params.outpaint_controlnet_depth ?? 160}
              onChange={(e) => setParams({ ...params, outpaint_controlnet_depth: parseFloat(e.target.value) })}
            />
            <Slider
              label="Confidence Taper"
              min={0.5}
              max={4.0}
              step={0.25}
              value={params.outpaint_controlnet_taper ?? 2.0}
              onChange={(e) => setParams({ ...params, outpaint_controlnet_taper: parseFloat(e.target.value) })}
            />
            {params.outpaint_controlnet_mode === "crop_mask" && (
              <>
                <Slider
                  label="Corner Residual Gate Radius (px)"
                  min={0}
                  max={64}
                  step={2}
                  value={params.outpaint_controlnet_corner_gate_radius_px ?? 0.0}
                  onChange={(e) => setParams({ ...params, outpaint_controlnet_corner_gate_radius_px: parseFloat(e.target.value) })}
                />
                <Slider
                  label="Corner Residual Gate Min"
                  min={0}
                  max={1}
                  step={0.05}
                  value={params.outpaint_controlnet_corner_gate_min ?? 1.0}
                  onChange={(e) => setParams({ ...params, outpaint_controlnet_corner_gate_min: parseFloat(e.target.value) })}
                />
                <p className="text-xs text-gray-500 sm:col-span-2">
                  Attenuates the ControlNet residual in a disk of this radius around each of the 4 placed-rect corners, down to the min value at the corner center. Edges away from corners keep full residual strength. Radius 0 = disabled.
                </p>
                <Slider
                  label="Corner Conditioning Radius (px)"
                  min={0}
                  max={64}
                  step={2}
                  value={params.outpaint_controlnet_corner_radius_px ?? 0.0}
                  onChange={(e) => setParams({ ...params, outpaint_controlnet_corner_radius_px: parseFloat(e.target.value) })}
                />
                <p className="text-xs text-gray-500 sm:col-span-2">
                  Rounds the ControlNet conditioning's known/unknown boundary at each corner instead of a sharp 90-degree vertex. Only takes effect together with a nonzero edge feather. 0 = disabled.
                </p>
                <Slider
                  label="Pin Corner Relax Radius (px)"
                  min={0}
                  max={64}
                  step={2}
                  value={params.outpaint_pin_corner_relax_radius_px ?? 0.0}
                  onChange={(e) => setParams({ ...params, outpaint_pin_corner_relax_radius_px: parseFloat(e.target.value) })}
                />
                <Slider
                  label="Pin Corner Relax Min"
                  min={0}
                  max={1}
                  step={0.05}
                  value={params.outpaint_pin_corner_relax_min ?? 1.0}
                  onChange={(e) => setParams({ ...params, outpaint_pin_corner_relax_min: parseFloat(e.target.value) })}
                />
                <p className="text-xs text-gray-500 sm:col-span-2">
                  Softens the per-step known-region pin in a disk of this radius around each of the 4 placed-rect corners, down to the min value at the corner center. Edges away from corners keep the full pin. The preserved rect stays byte-exact via the final paste. Radius 0 = disabled.
                </p>
              </>
            )}
          </div>
        )}
      </div>
    ),

    regional_prompt: () => (
      <div className="space-y-3">
        {/* Regional additional prompt (image outpaint only, SD/SDXL): conditions
            ONLY the generated region, leaving the main prompt + the placed
            (preserved) region untouched. See backend/api/routes.py
            generate_outpaint region_* Form params. */}
        <p className="text-xs text-gray-500">
          Conditions only the generated region — the main prompt above and the placed (preserved) input pixels are unaffected.
          Cost: "cfg" runs an extra regional denoise branch (up to ~2x U-Net forwards, more with outpaint's resampling passes). "attention" adds no extra forward pass.
        </p>
        <TextareaWithTagSuggestions
          label="Generated-region positive prompt"
          placeholder="Additional prompt applied only in the generated region..."
          rows={2}
          value={params.region_prompt || ""}
          onChange={(e) => setParams({ ...params, region_prompt: e.target.value })}
          enableWeightControl={true}
        />
        <div className="relative">
          <TextareaWithTagSuggestions
            label="Generated-region negative prompt"
            placeholder="Additional negative prompt applied only in the generated region..."
            rows={2}
            value={params.region_negative_prompt || ""}
            onChange={(e) => setParams({ ...params, region_negative_prompt: e.target.value })}
            enableWeightControl={true}
          />
          <button
            type="button"
            onClick={() => setParams({
              ...params,
              region_negative_prompt: "ui, hud, frame, border, text, watermark, logo, letterbox, game screenshot, game ui, health bar, speech bubble, dialogue box",
            })}
            className="mt-1 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
          >
            Fill with chrome-suppression default
          </button>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Slider
            label="Regional Prompt Strength"
            min={0}
            max={2}
            step={0.05}
            value={params.region_prompt_strength ?? 1.0}
            onChange={(e) => setParams({ ...params, region_prompt_strength: parseFloat(e.target.value) })}
          />
          <Select
            label="Regional Prompt Method"
            options={[
              { value: "cfg", label: "Spatial CFG (stronger, ~2x slower)" },
              { value: "attention", label: "Attention (free, softer)" },
            ]}
            value={params.region_prompt_method || "cfg"}
            onChange={(e) => setParams({ ...params, region_prompt_method: e.target.value })}
          />
          <Slider
            label="Region Mask Feather (latent px)"
            min={0}
            max={8}
            step={0.5}
            value={params.region_mask_feather ?? 0.0}
            onChange={(e) => setParams({ ...params, region_mask_feather: parseFloat(e.target.value) })}
          />
        </div>
      </div>
    ),

    seam: () => (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
        {/* Seam Membrane: post-decode harmonic boundary-offset blend.
            Adjusts generated pixels to meet the preserved boundary
            exactly; the preserved region remains byte-identical. See
            backend/core/inference/seam_membrane.py +
            backend/api/routes.py generate_outpaint outpaint_seam_membrane*
            Form params. */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-gray-300">Seam Membrane</div>
          <p className="text-xs text-gray-500">
            Solves a smooth per-channel offset field over the generated region, equal to the preserved rectangle&apos;s own pixels at the seam and diffused smoothly away from it, tapering to zero over a fixed band.
            Runs after the exposure harmonizer above and before the final unconditional paste; the preserved rectangle stays byte-identical.
          </p>
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="outpaint_seam_membrane"
              checked={params.outpaint_seam_membrane ?? false}
              onChange={(e) => setParams({ ...params, outpaint_seam_membrane: e.target.checked })}
              disabled={isGenerating}
              className="w-4 h-4"
            />
            <label htmlFor="outpaint_seam_membrane" className="text-sm text-gray-300">
              Enable
            </label>
          </div>
          {params.outpaint_seam_membrane && developerMode && (
            <Slider
              label="Taper Band (px, 0 = auto)"
              min={0}
              max={256}
              step={8}
              value={params.outpaint_seam_membrane_band ?? 0}
              onChange={(e) => setParams({ ...params, outpaint_seam_membrane_band: parseInt(e.target.value, 10) })}
            />
          )}
        </div>

        {/* Cross-Seam Tone Membrane ("R2"): post-decode low-frequency
            tone correction, distinct from the harmonic membrane above.
            Measures the tone step between the preserved rectangle's own
            pixels and the decoded generated pixels immediately across the
            seam and writes a decaying offset into the generated side
            only. See backend/core/inference/seam_membrane.py
            apply_cross_seam_tone + backend/api/routes.py generate_outpaint
            outpaint_seam_tone_* Form params. */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-gray-300">Cross-Seam Tone Membrane</div>
          <p className="text-xs text-gray-500">
            Measures the tone step between the preserved rectangle&apos;s own pixels and the decoded generated pixels immediately across the seam, subtracts the local content gradient estimated from the preserved side, and writes a decaying offset into the generated side only, within the decay band.
            Runs after the harmonic membrane above and before the final unconditional paste; the preserved rectangle stays byte-identical.
          </p>
          <Slider
            label="Strength (0 = off)"
            min={0}
            max={2.0}
            step={0.05}
            value={params.outpaint_seam_tone_strength ?? 0.0}
            onChange={(e) => setParams({ ...params, outpaint_seam_tone_strength: parseFloat(e.target.value) })}
          />
          {(params.outpaint_seam_tone_strength ?? 0) > 0 && developerMode && (
            <Slider
              label="Decay Band (px, 0 = auto)"
              min={0}
              max={64}
              step={2}
              value={params.outpaint_seam_tone_band ?? 0}
              onChange={(e) => setParams({ ...params, outpaint_seam_tone_band: parseInt(e.target.value, 10) })}
            />
          )}
        </div>

        {/* Boundary-offset propagation ("G_prop16"): post-decode, a third
            seam mechanism distinct from both membranes above. Measures the
            same offset the harmonic membrane measures (preserved pixels
            vs the decoded reconstruction of that same region, not the
            cross-seam comparison the tone membrane uses), and writes it
            directly into the generated pixels adjacent to the seam.
            Writes only generated-side pixels; the preserved region is
            unaffected. See backend/core/inference/seam_membrane.py
            apply_seam_offset_propagation + backend/api/routes.py
            generate_outpaint outpaint_seam_offset_prop Form param. */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-gray-300">Boundary-Offset Propagation</div>
          <p className="text-xs text-gray-500">
            Measures the offset between the preserved rectangle&apos;s own pixels and the decoded reconstruction of that same region, and writes it directly into the generated pixels adjacent to the seam as a low-frequency term plus a short high-frequency residual term, each tapered to zero moving away from the seam.
            Writes only generated-side pixels; the preserved rectangle stays byte-identical. Runs after the cross-seam tone membrane above and before the final unconditional paste.
          </p>
          <Slider
            label="Strength (0 = off)"
            min={0}
            max={2.0}
            step={0.05}
            value={params.outpaint_seam_offset_prop ?? 0.0}
            onChange={(e) => setParams({ ...params, outpaint_seam_offset_prop: parseFloat(e.target.value) })}
          />
        </div>

        {/* Paste-band reconciliation feather ("Option E"): at the final
            preserved-rectangle paste, the last N rows/columns of the
            preserved rectangle at its generate-adjacent edges are blended
            from the exact input toward the decoded canvas underneath
            instead of pasted byte-exact. Independent of boundary relaxation's
            own feather paste and takes precedence over it when both are
            active. See backend/core/inference/outpaint_utils.py
            reconcile_and_paste's paste_feather_px + backend/api/routes.py
            generate_outpaint outpaint_paste_feather_px Form param. */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-gray-300">Paste-Band Reconciliation Feather</div>
          <p className="text-xs text-gray-500">
            At the final preserved-rectangle paste, blends the last N rows/columns of the preserved rectangle at its generate-adjacent edges from the exact input toward the decoded canvas already underneath them, instead of pasting byte-exact.
            Independent of the boundary relaxation feather paste and takes precedence over it when both are active; only the N-row/column band loses byte-exactness.
          </p>
          <Slider
            label="Feather Width (px, 0 = off; default 24 removes the hard seam paste-line)"
            min={0}
            max={64}
            step={1}
            value={params.outpaint_paste_feather_px ?? 24}
            onChange={(e) => setParams({ ...params, outpaint_paste_feather_px: parseInt(e.target.value, 10) })}
          />
        </div>

        {/* Preserved-region compositing mode: "exact" (default) is the
            current byte-exact paste, unchanged. "vae_reconstruct" outputs
            a single uniform VAE decode of the whole canvas with no paste
            at all -- the preserved region becomes a VAE reconstruction of
            the input rather than byte-identical to it, removing the hard
            raw/decoded pixel discontinuity at the boundary.
            "vae_reconstruct_hf" additionally restores the preserved
            region's own high-frequency detail on top of that uniform
            decode, tapering to zero at the boundary; implemented for
            SD1.5/SDXL, falls back to "vae_reconstruct" on other
            architectures. See backend/core/inference/outpaint_utils.py
            reconcile_and_paste's outpaint_preserve_mode + backend/api/
            routes.py generate_outpaint outpaint_preserve_mode Form param. */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-gray-300">Preserved-Region Compositing Mode</div>
          <p className="text-xs text-gray-500">
            Controls how the preserved (placed input) rectangle is combined with the generated surroundings.
            &quot;Exact&quot; keeps the input byte-for-byte identical. The other two modes replace it with a VAE
            reconstruction of the input (not byte-identical) in exchange for a boundary with no hard raw/decoded
            pixel discontinuity; &quot;VAE reconstruct + HF restore&quot; additionally restores the input&apos;s own
            high-frequency detail on top of that reconstruction (implemented for SD1.5/SDXL only).
          </p>
          <Select
            label="Preserve Mode"
            options={[
              { value: "exact", label: "Exact (byte-identical input, default)" },
              { value: "vae_reconstruct", label: "VAE reconstruct (uniform decode, not byte-identical)" },
              { value: "vae_reconstruct_hf", label: "VAE reconstruct + HF restore (not byte-identical, SD1.5/SDXL)" },
            ]}
            value={params.outpaint_preserve_mode || "exact"}
            onChange={(e) => setParams({ ...params, outpaint_preserve_mode: e.target.value as "exact" | "vae_reconstruct" | "vae_reconstruct_hf" })}
          />
        </div>
      </div>
    ),

    continuity: () => (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
        {/* Seam Structure Continuity (SSC): continues thin structures that
            cross the region boundary (a held rod/staff, limb, torso, lines)
            into the generated region. See backend/api/routes.py
            generate_outpaint seam_structure_* Form params. */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-gray-300">Seam Structure Continuity</div>
          <p className="text-xs text-gray-500">
            SD/SDXL only. Continues thin structures that cross the region boundary (a held rod/staff, limb, torso, lines) into the generated region.
            x0-space, no extra U-Net forwards. 0 = off.
          </p>
          <Slider
            label="Seam Structure Strength"
            min={0}
            max={1.5}
            step={0.05}
            value={params.seam_structure_strength ?? 0.0}
            onChange={(e) => setParams({ ...params, seam_structure_strength: parseFloat(e.target.value) })}
          />
          {developerMode && (
            <>
              <Slider
                label="Seam Structure Depth (latent cells)"
                min={1}
                max={24}
                step={1}
                value={params.seam_structure_depth ?? 6.0}
                onChange={(e) => setParams({ ...params, seam_structure_depth: parseFloat(e.target.value) })}
              />
              <Slider
                label="Seam Structure End (schedule progress)"
                min={0.45}
                max={1.0}
                step={0.05}
                value={params.seam_structure_end ?? 0.70}
                onChange={(e) => setParams({ ...params, seam_structure_end: parseFloat(e.target.value) })}
              />
              <Slider
                label="Seam Structure Saliency"
                min={0}
                max={6}
                step={0.5}
                value={params.seam_structure_saliency ?? 2.0}
                onChange={(e) => setParams({ ...params, seam_structure_saliency: parseFloat(e.target.value) })}
              />
              <Slider
                label="Seam Structure Max Area"
                min={0.05}
                max={1.0}
                step={0.05}
                value={params.seam_structure_max_area ?? 0.25}
                onChange={(e) => setParams({ ...params, seam_structure_max_area: parseFloat(e.target.value) })}
              />
            </>
          )}
        </div>

        {/* Boundary Determinism Relaxation (BDR): soft-pins a narrow
            saliency-gated seam band (annealed soft->hard) so the known-side
            latent can bend to meet the continuation. See backend/api/routes.py
            generate_outpaint boundary_relax_* Form params. */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-gray-300">Boundary Relaxation</div>
          <p className="text-xs text-gray-500">
            SD/SDXL only. Soft-pins a narrow saliency-gated seam band (annealed soft-&gt;hard) so the known-side latent can bend to meet the continuation.
            Most effective with Seam Structure Continuity &gt; 0. 0 = off.
          </p>
          <Slider
            label="Boundary Relax Strength"
            min={0.0}
            max={0.5}
            step={0.05}
            value={params.boundary_relax_strength ?? 0.0}
            onChange={(e) => setParams({ ...params, boundary_relax_strength: parseFloat(e.target.value) })}
          />
          {developerMode && (
            <>
              <Slider
                label="Boundary Relax Width (latent px)"
                min={1}
                max={6}
                step={1}
                value={params.boundary_relax_width ?? 3.0}
                onChange={(e) => setParams({ ...params, boundary_relax_width: parseFloat(e.target.value) })}
              />
              <Slider
                label="Boundary Relax Noise"
                min={0}
                max={1}
                step={0.05}
                value={params.boundary_relax_noise ?? 0.35}
                onChange={(e) => setParams({ ...params, boundary_relax_noise: parseFloat(e.target.value) })}
              />
              <Slider
                label="Boundary Relax Full Until (schedule progress)"
                min={0}
                max={1}
                step={0.05}
                value={params.boundary_relax_full_until ?? 0.37}
                onChange={(e) => setParams({ ...params, boundary_relax_full_until: parseFloat(e.target.value) })}
              />
              <Slider
                label="Boundary Relax End (schedule progress)"
                min={0}
                max={1}
                step={0.05}
                value={params.boundary_relax_end ?? 0.55}
                onChange={(e) => setParams({ ...params, boundary_relax_end: parseFloat(e.target.value) })}
              />
              <Select
                label="Boundary Relax Paste Mode"
                options={[
                  { value: "feather", label: "Feather (thin model-rendered seam strip)" },
                  { value: "exact", label: "Exact (full byte-exact input)" },
                ]}
                value={params.boundary_relax_paste || "feather"}
                onChange={(e) => setParams({ ...params, boundary_relax_paste: e.target.value })}
              />
            </>
          )}
        </div>

        {/* In-loop continuity fixes B1/B2/B3 (SD/SDXL only; core.inference.
            custom_sampling's outpaint_noise_init-gated mechanisms). Unlike
            the post-decode seam mechanisms above, these run inside the
            denoise loop itself. See backend/api/param_defaults.py
            OUTPAINT_DEFAULTS outpaint_boundary_color_strength /
            outpaint_resample_count / outpaint_jump_length /
            outpaint_reference_strength. */}
        <div className="space-y-4 lg:col-span-2">
          <div className="text-sm font-medium text-gray-300">In-loop Continuity（境界連続性）</div>
          <div className="space-y-2">
            <p className="text-xs text-gray-500">
              B1: a weak low-frequency color/illumination correction applied to the generate region only, within a narrow collar near the preserved rectangle&apos;s boundary, active mid/late in the schedule. 0 = off.
            </p>
            <Slider
              label="Boundary Color Strength (0 = off)"
              min={0}
              max={1.0}
              step={0.05}
              value={params.outpaint_boundary_color_strength ?? 0.25}
              onChange={(e) => setParams({ ...params, outpaint_boundary_color_strength: parseFloat(e.target.value) })}
            />
          </div>

          <div className="space-y-2">
            <p className="text-xs text-gray-500">
              B2: after a denoise step inside a mid-schedule band, jumps back the jump-back length in step indices by re-noising the whole latent (keep + generate together) and re-denoising, repeated the resample count times per band segment. 1 = off (B1 only). Values above 1 multiply the number of denoise passes actually run -- roughly 1.5-2x the requested step count at a resample count of 2. Only takes effect with a resample-compatible sampler (Euler, Euler Ancestral, DDIM, DDPM).
            </p>
            <div className="flex items-center gap-2">
              <label htmlFor="outpaint_resample_count" className="text-xs text-gray-400">Resample Count (1 = off)</label>
              <NumberInput
                id="outpaint_resample_count"
                min={1}
                max={8}
                step={1}
                parse="int"
                value={params.outpaint_resample_count ?? 1}
                defaultValue={1}
                onCommit={(v) => setParams({ ...params, outpaint_resample_count: v })}
                className="w-20"
              />
            </div>
            {(params.outpaint_resample_count ?? 1) > 1 && (
              <div className="flex items-center gap-2 ml-6">
                <label htmlFor="outpaint_jump_length" className="text-xs text-gray-400">Jump-Back Length (steps)</label>
                <NumberInput
                  id="outpaint_jump_length"
                  min={1}
                  max={32}
                  step={1}
                  parse="int"
                  value={params.outpaint_jump_length ?? 4}
                  defaultValue={4}
                  onCommit={(v) => setParams({ ...params, outpaint_jump_length: v })}
                  className="w-20"
                />
              </div>
            )}
          </div>

          <div className="space-y-2">
            <p className="text-xs text-gray-500">
              B3: masked self-attention KV injection -- a noise-matched reference composite built from the preserved rectangle&apos;s own clean latents, restricted to known-region tokens via spatial masking, so generate-region self-attention queries can attend to the input&apos;s own clean features. 0 = off.
            </p>
            <Slider
              label="Reference Strength (0 = off)"
              min={0}
              max={1.0}
              step={0.05}
              value={params.outpaint_reference_strength ?? 0.0}
              onChange={(e) => setParams({ ...params, outpaint_reference_strength: parseFloat(e.target.value) })}
            />
          </div>
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
            id="spectrum_enable_outpaint"
            checked={params.spectrum_enable || false}
            onChange={(e) => setParams({ ...params, spectrum_enable: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="spectrum_enable_outpaint" className="text-sm text-gray-300">
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
            id="fbcache_enable_outpaint"
            checked={params.fbcache_enable || false}
            onChange={(e) => setParams({ ...params, fbcache_enable: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="fbcache_enable_outpaint" className="text-sm text-gray-300">
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
            id="flatten_in_loop_outpaint"
            checked={params.flatten_in_loop || false}
            onChange={(e) => setParams({ ...params, flatten_in_loop: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="flatten_in_loop_outpaint" className="text-sm text-gray-300" title="During the final denoise steps, detects the flat background region and replaces it with its solid dominant color (both luma and chroma become uniform - stronger than Color Flatten); no-op when no confident flat region is found; SD/SDXL only for now.">
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

        <div className="flex items-center space-x-2 mt-2" title="Subtracts the VAE encode/decode round-trip color bias (measured per image) from the output; independent of denoising strength.">
          <input
            type="checkbox"
            id="vae_drift_correction_outpaint"
            checked={params.vae_drift_correction ?? false}
            onChange={(e) => setParams({ ...params, vae_drift_correction: e.target.checked })}
            className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
          />
          <label htmlFor="vae_drift_correction_outpaint" className="text-sm text-gray-300">
            VAE drift correction
          </label>
        </div>
      </div>
    ),
  };

  // ── What the chosen canvas does to the input clip ────────────────────────
  //
  // The backend fits the upload to width x height with
  // `center_crop_resize_frames` and the RESULT of that preprocessing -- not the
  // raw upload -- is what the exact-preservation guarantee is about. So the
  // panel states, factually:
  //   * whether the canvas is the clip's own resolution (preserved frames ARE
  //     the uploaded frames), and
  //   * when it is not, that the clip is centre-cropped and resized to it, and
  //     which edges that discards.
  // `videoCanvasAt1x` is the nearest canvas this architecture accepts to the
  // clip's own size; when that already differs from the clip, 1x is simply not
  // reachable here and the rule that prevents it is quoted.
  const videoCanvasAt1x = inputVideoSize
    ? fitVideoCanvas(archCapabilities, loadedArchType, inputVideoSize.width, inputVideoSize.height, 1)
    : null;
  const canvasWidth = params.width ?? 768;
  const canvasHeight = params.height ?? 512;
  const canvasIsSourceSize = !!inputVideoSize
    && canvasWidth === inputVideoSize.width
    && canvasHeight === inputVideoSize.height;
  const sourceAspect = inputVideoSize ? inputVideoSize.width / inputVideoSize.height : 0;
  const canvasAspect = canvasHeight > 0 ? canvasWidth / canvasHeight : 0;
  const canvasCropsSource = !!inputVideoSize && Math.abs(sourceAspect - canvasAspect) > 1e-3;
  const croppedEdges = sourceAspect > canvasAspect ? "left and right" : "top and bottom";

  // ── What the Absolute sliders are allowed to reach ───────────────────────
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
  const videoWidthBounds = videoCanvasAxisBounds(archCapabilities, loadedArchType, canvasHeight);
  const videoHeightBounds = videoCanvasAxisBounds(archCapabilities, loadedArchType, canvasWidth);
  const videoCanvasOverEnvelope = videoCanvasExceedsEnvelope(
    archCapabilities, loadedArchType, canvasWidth, canvasHeight);
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
        value={params.negative_prompt || ""}
        onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
        disabled
        title="Audio generation does not accept negative-prompt conditioning."
      />
      <p className="text-xs text-gray-500">Unavailable for audio generation; the saved value is preserved.</p>
    </Card>
  ) : (
    <Card title="Prompt">
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
        suggestionMode={loadedArchType === "minimax_h3" ? "h3" : "tags"}
        enableWeightControl
      />
      {loadedArchType === "minimax_h3" && (
        <H3PromptAssist
          prompt={params.prompt}
          onApply={(prompt) => setParams((previous) => ({ ...previous, prompt }))}
          suggestedMode={isRef2Va ? "ref2va" : "t2va"}
          durationSeconds={(params.total_frames ?? 121) / (params.frame_rate ?? 24)}
          references={createH3ReferenceInventory({
            pictures: h3ReferenceImages.length,
            videos: 1 + (bridgeVideoFile ? 1 : 0),
          })}
        />
      )}
      <TextareaWithTagSuggestions
        label="Negative Prompt"
        placeholder={supportsNegativePrompt ? "Enter negative prompt..." : "Negative prompting is unavailable for this model"}
        rows={2}
        resizeStorageKey={GENERATION_NEGATIVE_PROMPT_HEIGHT_KEY}
        value={params.negative_prompt || ""}
        onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
        suggestionMode={loadedArchType === "minimax_h3" ? "h3" : "tags"}
        enableWeightControl
        disabled={!supportsNegativePrompt}
        title={!supportsNegativePrompt ? "The loaded model does not accept negative-prompt conditioning." : undefined}
      />
      {!supportsNegativePrompt && (
        <p className="text-xs text-gray-500">Unavailable for the loaded model; the saved value is preserved.</p>
      )}
      {!isVideo && developerMode && (
        <div className="mt-3 flex items-center space-x-2">
          <input
            type="checkbox"
            id="outpaint_preview_unpinned_x0"
            checked={params.outpaint_preview_unpinned_x0 ?? false}
            onChange={(e) => setParams({ ...params, outpaint_preview_unpinned_x0: e.target.checked })}
            disabled={isGenerating}
            className="h-4 w-4"
          />
          <label htmlFor="outpaint_preview_unpinned_x0" className="text-sm text-gray-300">
            Preview: show unpinned prediction
          </label>
        </div>
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
            const modelInfo = await getCurrentModel();
            setCurrentModelInfo(modelInfo);
            const modelType = modelInfo?.model_info?.type;
            if (modelType === "zimage" || modelType === "flux2" || modelType === "anima") {
              setParams(prev => ({ ...prev, sampler: "euler", schedule_type: "flow" }));
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
          storageKeyPrefix="outpaint"
        />

        <GenerationLeadGrid
          prompt={promptPanel}
          conditioning={(
            <>
        {!isVideo && !isAudio && (
        <Card
          title="Input Image"
          collapsible={true}
          defaultCollapsed={false}
          storageKey="outpaint_input_collapsed"
          collapsedPreview={
            inputImagePreview ? (
              <span className="text-green-400 text-sm">✓ Image loaded</span>
            ) : (
              <span className="text-gray-500 text-sm">No image</span>
            )
          }
        >
          <div className="space-y-2">
            <div className="flex gap-2">
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg,image/webp"
                onChange={handleImageUpload}
                className="block w-full text-sm text-gray-400
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-medium
                  file:bg-blue-600 file:text-white
                  hover:file:bg-blue-700
                  file:cursor-pointer cursor-pointer"
              />
              {inputImagePreview && (
                <Button onClick={handleClearInputImage} variant="secondary" size="sm" title="Clear input image">
                  Clear
                </Button>
              )}
            </div>
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`h-[clamp(10rem,22vh,13rem)] bg-gray-800 rounded-lg overflow-hidden border-2 border-dashed transition-colors relative ${
                isDragging ? 'border-blue-500 bg-gray-700' : 'border-gray-600'
              }`}
            >
              {inputImagePreview ? (
                <img src={inputImagePreview} alt="Input" className="w-full h-full object-contain" />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <p className="text-gray-500 text-center px-4">
                    {isDragging ? 'Drop image here' : 'Drag and drop an image here or use the file picker above'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </Card>
        )}

        {isVideo && (
        <Card
          title="Input Video"
          collapsible={true}
          defaultCollapsed={false}
          storageKey="outpaint_video_input_collapsed"
          collapsedPreview={
            videoPreviewUrl ? (
              <span className="text-green-400 text-sm">✓ Clip loaded</span>
            ) : (
              <span className="text-gray-500 text-sm">No clip</span>
            )
          }
        >
          <div className="space-y-2">
            <div className="flex gap-2">
              <input
                type="file"
                accept="video/mp4,video/webm"
                onChange={handleVideoUpload}
                className="block w-full text-sm text-gray-400
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-medium
                  file:bg-blue-600 file:text-white
                  hover:file:bg-blue-700
                  file:cursor-pointer cursor-pointer"
              />
              {videoPreviewUrl && (
                <Button onClick={handleClearVideo} variant="secondary" size="sm" title="Clear input video">
                  Clear
                </Button>
              )}
            </div>
            <div className="h-[clamp(10rem,22vh,13rem)] bg-gray-800 rounded-lg overflow-hidden border-2 border-dashed border-gray-600">
              {videoPreviewUrl ? (
                <video
                  src={videoPreviewUrl}
                  onLoadedMetadata={handleVideoLoadedMetadata}
                  className="w-full h-full object-contain"
                  controls
                  muted
                  playsInline
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <p className="text-gray-500 text-center px-4">Use the file picker above to select an mp4/webm clip</p>
                </div>
              )}
            </div>
            {videoDurationSec != null && (
              <p className="text-xs text-gray-500">
                Clip length: {videoDurationSec.toFixed(2)}s (~{videoRawFrames} frames at {params.frame_rate ?? 24.0} fps).
                Non-÷32 resolutions are center-cropped/resized once to width×height; the resized frames become the exact-preserved content.
              </p>
            )}
          </div>
        </Card>
        )}

        {isAudio && (
        <Card
          title="Input Audio"
          collapsible={true}
          defaultCollapsed={false}
          storageKey="outpaint_audio_input_collapsed"
          collapsedPreview={
            audioPreviewUrl ? (
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
                onChange={handleAudioUpload}
                className="block w-full text-sm text-gray-400
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-medium
                  file:bg-blue-600 file:text-white
                  hover:file:bg-blue-700
                  file:cursor-pointer cursor-pointer"
              />
              {audioPreviewUrl && (
                <Button onClick={handleClearAudio} variant="secondary" size="sm" title="Clear input audio">
                  Clear
                </Button>
              )}
            </div>
            {audioPreviewUrl ? (
              <audio
                src={audioPreviewUrl}
                onLoadedMetadata={handleAudioLoadedMetadata}
                className="w-full"
                controls
              />
            ) : (
              <div className="bg-gray-800 rounded-lg border-2 border-dashed border-gray-600 py-6">
                <p className="text-gray-500 text-center text-sm px-4">
                  Use the file picker above to select an audio clip to extend
                </p>
              </div>
            )}
            {audioDurationSec != null && (
              <p className="text-xs text-gray-500">
                Clip length: {audioDurationSec.toFixed(2)}s. Uploads not already 48kHz/16-bit stereo are
                resampled/requantized once during normalization; the placed span is otherwise sample-exact.
              </p>
            )}
          </div>
        </Card>
        )}
            </>
          )}
        />

        {/* Outpaint Options: a single-open tabbed accordion (chrome shared
            via frontend/src/components/common/TabbedOptions.tsx). Every
            control below is unchanged from its original location (same
            param binding / handler / conditional reveal) -- only the
            container changed. See OUTPAINT_OPTIONS_TAB_KEYS /
            isOutpaintOptionsTabActive / outpaintOptionsTabRender above. */}
        {!isVideo && !isAudio && (
          <TabbedOptions<OutpaintPanelParams>
            cardTitle="Outpaint Options"
            params={params}
            setParams={setParams}
            defaultParams={DEFAULT_PARAMS}
            tabs={OUTPAINT_OPTIONS_TABS.map((tab) => ({
              id: tab.id,
              label: tab.label,
              keys: OUTPAINT_OPTIONS_TAB_KEYS[tab.id],
              isActive: (p: OutpaintPanelParams) => isOutpaintOptionsTabActive(tab.id, p),
              render: outpaintOptionsTabRender[tab.id],
            }))}
          />
        )}

        {!isVideo && !isAudio && (
        <Card title="Placement">
          <OutpaintPlacementCanvas
            inputImagePreview={inputImagePreview}
            inputImageSize={inputImageSize}
            params={placementParams}
            onChange={handlePlacementChange}
            maintainAspect={maintainAspect}
            onMaintainAspectChange={setMaintainAspect}
          />
        </Card>
        )}

        {isVideo && (
        <Card title="Temporal Placement">
          {boundaryPlacementOnly && (
            <div className="mb-4 space-y-3">
              <Select
                label="Placement"
                value={videoPlacement}
                onChange={(e) => setVideoPlacement(e.target.value as "extend_forward" | "extend_backward" | "bridge")}
                options={[
                  ...(outpaintPlacements.includes("extend_forward")
                    ? [{ value: "extend_forward", label: "Extend forward (generate after the clip)" }] : []),
                  ...(outpaintPlacements.includes("extend_backward")
                    ? [{ value: "extend_backward", label: "Extend backward (generate before the clip)" }] : []),
                  ...(outpaintPlacements.includes("bridge")
                    ? [{ value: "bridge", label: "Bridge (generate between two clips)" }] : []),
                ]}
              />
              <p className="text-xs text-gray-500">
                This model conditions on the first and/or last frame of the span it generates, so the
                input clip must sit at the start or the end of the timeline, or at both ends of a
                bridge. Interior source motion is not provided to the model. The generated span is
                what has to be a length the model can produce, so the effective output length is
                reported with the result.
              </p>
              {videoPlacement === "bridge" && (
                <div className="space-y-2">
                  <label className="block text-xs text-gray-400">Second clip (preserved at the end)</label>
                  {bridgeVideoPreviewUrl ? (
                    <div className="space-y-2">
                      <video
                        src={bridgeVideoPreviewUrl}
                        className="w-full max-h-40 rounded border border-gray-700"
                        controls
                        onLoadedMetadata={(e) => {
                          const d = e.currentTarget.duration;
                          if (Number.isFinite(d) && d > 0) setBridgeVideoDurationSec(d);
                        }}
                      />
                      <button
                        type="button"
                        onClick={handleClearBridgeVideo}
                        className="text-xs text-gray-400 hover:text-gray-200 underline"
                      >
                        Remove second clip
                      </button>
                    </div>
                  ) : (
                    <input type="file" accept="video/*" onChange={handleBridgeVideoUpload}
                           className="block w-full text-xs text-gray-300" />
                  )}
                  {bridgeVideoDurationSec != null && (
                    <p className="text-xs text-gray-500">
                      {bridgeVideoDurationSec.toFixed(2)}s (~{Math.round(bridgeVideoDurationSec * (params.frame_rate ?? 24.0))} frames) preserved at the end.
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
          <OutpaintTimeline
            totalUnits={params.total_frames ?? 121}
            onTotalUnitsChange={(v) => setParams(prev => ({ ...prev, total_frames: v }))}
            totalUnitsSnapFn={snapTotalFrames}
            totalUnitsMin={9}
            totalUnitsStep={8}
            rawSegmentLength={videoRawFrames}
            trimStart={params.input_trim_start_frames ?? 0}
            onTrimStartChange={(v) => setParams(prev => ({ ...prev, input_trim_start_frames: v }))}
            trimEnd={params.input_trim_end_frames ?? 0}
            onTrimEndChange={(v) => setParams(prev => ({ ...prev, input_trim_end_frames: v }))}
            offset={params.input_offset_frames ?? 0}
            onOffsetChange={(v) => setParams(prev => ({ ...prev, input_offset_frames: v }))}
            offsetSnapFn={boundaryPlacementOnly ? snapBoundaryOffset : snapLtxOffset}
            gridSize={boundaryPlacementOnly ? 1 : 8}
            minSegmentLength={1}
            unitRate={params.frame_rate ?? 24.0}
            unitLabel="frames"
            disabled={!videoPreviewUrl}
          />
          <p className="text-xs text-gray-500 mt-2">
            {boundaryPlacementOnly
              ? "Dragging the clip snaps it to the start or the end of the timeline -- the only two offsets this model can anchor. A mid-timeline offset is refused by the backend rather than approximated."
              : "Offset is snapped to the nearest valid LTX-2.3 latent frame index (0, 1, 9, 17, ...); the backend re-snaps and warns if it differs."}
          </p>
        </Card>
        )}

        {/* MiniMax-H3 ref2va, extend_forward only -- the ONLY row the backend's
            partition/placement gate allows references on (extend_backward and
            bridge are refused outright there). The preserved clip is ALWAYS
            the video reference; this adds optional images on top. Does NOT
            hold identity across the join -- that is what A-V8 measures, and
            no claim of it is made here. */}
        {isVideo && isRef2Va && videoPlacement === "extend_forward" && (
          <MiniMaxH3ReferenceSelector
            value={{ images: h3ReferenceImages, videos: [], videoAudios: [], audios: [] }}
            onChange={(next) => setH3ReferenceImages(next.images)}
            referenceImageSize={h3ReferenceImageSize}
            onReferenceImageSizeChange={setH3ReferenceImageSize}
            disabled={isGenerating}
            imagesOnly
          />
        )}
        {isVideo && isRef2Va && videoPlacement !== "extend_forward" && (
          <p className="text-xs text-gray-500 -mt-2">
            Reference images are offered only for the extend-forward
            placement on this checkpoint (extend-backward and bridge are
            refused on ref2va -- unmeasured, not a UI limitation).
          </p>
        )}

        {isAudio && (
        <Card title="Temporal Placement">
          <OutpaintTimeline
            totalUnits={params.total_duration ?? 60.0}
            onTotalUnitsChange={(v) => setParams(prev => ({ ...prev, total_duration: v }))}
            totalUnitsSnapFn={clampAudioTotalDuration}
            totalUnitsMin={0.1}
            totalUnitsStep={1}
            rawSegmentLength={audioDurationSec ?? 0}
            trimStart={params.input_trim_start_sec ?? 0}
            onTrimStartChange={(v) => setParams(prev => ({ ...prev, input_trim_start_sec: v }))}
            trimEnd={params.input_trim_end_sec ?? 0}
            onTrimEndChange={(v) => setParams(prev => ({ ...prev, input_trim_end_sec: v }))}
            offset={params.input_offset_sec ?? 0}
            onOffsetChange={(v) => setParams(prev => ({ ...prev, input_offset_sec: v }))}
            gridSize={1 / 25}
            minSegmentLength={1 / 25}
            unitRate={1}
            unitLabel="s"
            unitParse="float"
            disabled={!audioPreviewUrl}
          />
          <p className="text-xs text-gray-500 mt-2">
            Offset/trim are snapped to the ACE-Step VAE's latent frame rate (1/25s); the backend re-snaps and clamps regardless.
          </p>
        </Card>
        )}

        {!isVideo && !isAudio && (
        <Card title="Parameters">
          <div className="space-y-4">
            <Slider
              label="Denoising Strength"
              min={0}
              max={1}
              step={0.05}
              value={params.denoising_strength ?? 1.0}
              onChange={(e) => setParams({ ...params, denoising_strength: parseFloat(e.target.value) })}
            />
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="outpaint_fix_steps"
                checked={params.img2img_fix_steps ?? true}
                onChange={(e) => setParams({ ...params, img2img_fix_steps: e.target.checked })}
                className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
              />
              <label htmlFor="outpaint_fix_steps" className="text-sm text-gray-300">
                Do full steps (ensures complete denoising regardless of strength)
              </label>
            </div>

            <Select
              label="Masked Content Fill (generated region)"
              options={[
                { value: "original", label: "Original" },
                { value: "blur", label: "Blur" },
                { value: "noise", label: "Latent Noise" },
                { value: "erase", label: "Latent Nothing" },
              ]}
              value={params.inpaint_fill_mode || "original"}
              onChange={(e) => setParams({ ...params, inpaint_fill_mode: e.target.value })}
            />
            {params.inpaint_fill_mode && params.inpaint_fill_mode !== "original" && (
              <>
                <Slider
                  label="Fill Strength"
                  min={0}
                  max={1}
                  step={0.05}
                  value={params.inpaint_fill_strength ?? 1.0}
                  onChange={(e) => setParams({ ...params, inpaint_fill_strength: parseFloat(e.target.value) })}
                />
                {params.inpaint_fill_mode === "blur" && (
                  <Slider
                    label="Blur Strength"
                    min={0.1}
                    max={5.0}
                    step={0.1}
                    value={params.inpaint_blur_strength ?? 1.0}
                    onChange={(e) => setParams({ ...params, inpaint_blur_strength: parseFloat(e.target.value) })}
                  />
                )}
              </>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Slider
                label="Steps"
                min={1}
                max={150}
                step={1}
                value={params.steps ?? 20}
                onChange={(e) => setParams({ ...params, steps: parseInt(e.target.value) })}
              />
              <Slider
                label="CFG Scale"
                min={0}
                max={30}
                step={0.5}
                value={params.cfg_scale ?? 7.0}
                onChange={(e) => setParams({ ...params, cfg_scale: parseFloat(e.target.value) })}
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Select
                label="Sampler"
                options={samplers.map(s => ({ value: s.id, label: s.name }))}
                value={params.sampler || "euler"}
                onChange={(e) => setParams({ ...params, sampler: e.target.value })}
              />
              <Select
                label="Schedule Type"
                options={scheduleTypes.map(s => ({ value: s.id, label: s.name }))}
                value={params.schedule_type || "uniform"}
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
                    value={params.seed ?? -1}
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
                    value={params.ancestral_seed ?? -1}
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

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="outpaint_show_advanced_cfg"
                checked={showAdvancedCFG}
                onChange={(e) => {
                  setShowAdvancedCFG(e.target.checked);
                  localStorage.setItem('show_advanced_cfg', String(e.target.checked));
                }}
                className="rounded"
              />
              <label htmlFor="outpaint_show_advanced_cfg" className="text-sm text-gray-300">
                Show Advanced CFG / NAG
              </label>
            </div>

            {showAdvancedCFG && (
              <div className="space-y-3">
                <label className="block text-sm font-medium text-gray-300">Dynamic CFG Schedule</label>
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
                      value={params.cfg_schedule_min ?? 1.0}
                      onChange={(e) => setParams({ ...params, cfg_schedule_min: parseFloat(e.target.value) })}
                    />
                    <Slider
                      label="CFG Max (start of generation)"
                      min={1}
                      max={30}
                      step={0.5}
                      value={params.cfg_schedule_max || params.cfg_scale || 7.0}
                      onChange={(e) => setParams({ ...params, cfg_schedule_max: parseFloat(e.target.value) })}
                    />
                    {params.cfg_schedule_type === "quadratic" && (
                      <Slider
                        label="Power (curve steepness)"
                        min={0.5}
                        max={4.0}
                        step={0.1}
                        value={params.cfg_schedule_power ?? 2.0}
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
                    value={params.cfg_rescale_snr_alpha ?? 0.0}
                    onChange={(e) => setParams({ ...params, cfg_rescale_snr_alpha: parseFloat(e.target.value) })}
                  />
                )}

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={(params.dynamic_threshold_percentile ?? 0) > 0}
                    onChange={(e) => setParams({ ...params, dynamic_threshold_percentile: e.target.checked ? 99.5 : 0.0 })}
                    className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
                  />
                  <label className="text-sm font-medium text-gray-300">Dynamic Thresholding</label>
                </div>
                {(params.dynamic_threshold_percentile ?? 0) > 0 && (
                  <>
                    <Slider
                      label="Threshold Percentile"
                      min={90}
                      max={100}
                      step={0.5}
                      value={params.dynamic_threshold_percentile ?? 99.5}
                      onChange={(e) => setParams({ ...params, dynamic_threshold_percentile: parseFloat(e.target.value) })}
                    />
                    <Slider
                      label="Mimic Scale (static clamp)"
                      min={1}
                      max={30}
                      step={0.5}
                      value={params.dynamic_threshold_mimic_scale ?? 7.0}
                      onChange={(e) => setParams({ ...params, dynamic_threshold_mimic_scale: parseFloat(e.target.value) })}
                    />
                  </>
                )}

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={params.nag_enable || false}
                    onChange={(e) => setParams({ ...params, nag_enable: e.target.checked })}
                    className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
                  />
                  <label className="text-sm font-medium text-gray-300">NAG (Normalized Attention Guidance)</label>
                </div>
                {params.nag_enable && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <Slider
                      label="NAG Scale"
                      min={1}
                      max={10}
                      step={0.5}
                      value={params.nag_scale ?? 5.0}
                      onChange={(e) => setParams({ ...params, nag_scale: parseFloat(e.target.value) })}
                    />
                    <Slider
                      label="NAG Tau"
                      min={1.0}
                      max={5.0}
                      step={0.1}
                      value={params.nag_tau ?? 3.5}
                      onChange={(e) => setParams({ ...params, nag_tau: parseFloat(e.target.value) })}
                    />
                    <Slider
                      label="NAG Alpha"
                      min={0.05}
                      max={1.0}
                      step={0.05}
                      value={params.nag_alpha ?? 0.25}
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

            {/* Model / Environment — pipeline-global settings, applied last */}
            <details className="bg-gray-800/40 border border-gray-700 rounded-lg p-3 mt-4">
              <summary className="text-sm font-semibold text-gray-300 cursor-pointer select-none">
                Model / Environment
              </summary>
              <div className="mt-3 space-y-3">
                {(currentModelInfo?.model_info?.type === "zimage" || currentModelInfo?.model_info?.type === "flux2" || currentModelInfo?.model_info?.type === "anima") ? (
                  <>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <Select
                        label={`Transformer Quantization (${currentModelInfo?.model_info?.type === "flux2" ? "FLUX.2" : "Z-Image"})`}
                        value={params.unet_quantization || "none"}
                        onChange={(e) => setParams({ ...params, unet_quantization: e.target.value === "none" ? null : e.target.value })}
                        options={unetQuantizationOptions(archCapabilities, currentModelInfo?.model_info?.type as string | undefined)}
                      />
                      <Select
                        label={`Text Encoder Quantization (${currentModelInfo?.model_info?.type === "flux2" ? "Qwen3" : "Gemma2"})`}
                        value={params.text_encoder_quantization || "none"}
                        onChange={(e) => setParams({ ...params, text_encoder_quantization: e.target.value === "none" ? null : e.target.value })}
                        options={[
                          { value: "none", label: "None" },
                          { value: "fp8_e4m3fn", label: "FP8 E4M3 (Recommended)" },
                          { value: "fp8_e5m2", label: "FP8 E5M2" },
                        ]}
                      />
                    </div>
                    {params.unet_quantization && params.unet_quantization !== "none" && params.unet_quantization !== "int8" && (
                      <div className="bg-blue-900/20 border border-blue-600/30 rounded-lg p-3">
                        <p className="text-xs text-blue-200">
                          Transformer FP8 reduces VRAM. Weights are dequantized back to full precision per operation during inference, so generation is slower than without quantization.
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
                ) : (
                  <>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {/* Only SD1.5/SDXL have a U-Net; every other arch here is a
                          DiT, so the label follows the loaded architecture. */}
                      <Select
                        label={transformerQuantizationLabel(currentModelInfo?.model_info?.type as string | undefined)}
                        value={params.unet_quantization || "none"}
                        onChange={(e) => setParams({ ...params, unet_quantization: e.target.value === "none" ? null : e.target.value })}
                        options={unetQuantizationOptions(archCapabilities, currentModelInfo?.model_info?.type as string | undefined)}
                      />
                    </div>
                    {params.unet_quantization && params.unet_quantization !== "none" && params.unet_quantization !== "int8" && (
                      <div className="bg-yellow-900/20 border border-yellow-600/30 rounded-lg p-3">
                        <p className="text-xs text-yellow-200">
                          Quantization reduces VRAM but may affect quality. FP8 weights are dequantized back to full precision per operation during inference, so generation is slower than without quantization. Original model kept on CPU.
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

                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={params.cpu_text_encoding ?? false}
                    onChange={(e) => setParams({ ...params, cpu_text_encoding: e.target.checked })}
                    className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-300">CPU Text Encoding</span>
                </label>

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="outpaint_vae_tiling"
                    checked={params.vae_tiling || false}
                    onChange={(e) => setParams({ ...params, vae_tiling: e.target.checked })}
                    className="rounded"
                  />
                  <label htmlFor="outpaint_vae_tiling" className="text-sm text-gray-300">VAE Tiling</label>
                  <span className="text-xs text-gray-500">(tiled decode for large canvases, saves VRAM)</span>
                </div>
                {params.vae_tiling && (
                  <>
                  <div className="flex items-center gap-2 ml-6">
                    <label htmlFor="outpaint_vae_tile_threshold" className="text-xs text-gray-400">Tile threshold (px)</label>
                    <NumberInput
                      id="outpaint_vae_tile_threshold"
                      min={0}
                      step={128}
                      value={params.vae_tile_threshold ?? 0}
                      defaultValue={0}
                      onCommit={(v) => setParams({ ...params, vae_tile_threshold: v })}
                      className="w-24"
                    />
                  </div>
                  <div className="flex items-center gap-2 ml-6 mt-1">
                    <label htmlFor="outpaint_vae_tile_mode" className="text-xs text-gray-400">Tile join</label>
                    <select
                      id="outpaint_vae_tile_mode"
                      value={params.vae_tile_mode ?? "blend"}
                      onChange={(e) => setParams({ ...params, vae_tile_mode: e.target.value })}
                      className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                    >
                      <option value="blend">Blend (overlapping tiles, cross-faded together)</option>
                      <option value="context">Context margin (16 latent cells of real neighbouring context, discarded after decode)</option>
                    </select>
                  </div>
                  <div className="flex items-center gap-2 ml-6 mt-1">
                    <input
                      type="checkbox"
                      id="outpaint_vae_tile_global_norm"
                      checked={params.vae_tile_global_norm || false}
                      onChange={(e) => setParams({ ...params, vae_tile_global_norm: e.target.checked })}
                      className="rounded"
                    />
                    <label htmlFor="outpaint_vae_tile_global_norm" className="text-xs text-gray-400">Global GroupNorm statistics</label>
                    <span className="text-xs text-gray-500">
                      decodes twice (whole-image GroupNorm statistics instead of per-tile);
                      measured on SDXL: per-tile offset 1.32 &rarr; 0.037 /255, decode time x2,
                      peak memory unchanged. The x2 applies to every VAE decode in the request,
                      including the in-loop decodes of In-Loop Flatten and VAE Drift Correction.
                      No effect on Anima/Krea2
                    </span>
                  </div>
                  </>
                )}

                {developerMode && (
                  <>
                    <div className="flex items-center gap-2 mt-2">
                      <input
                        type="checkbox"
                        id="outpaint_use_torch_compile"
                        checked={params.use_torch_compile || false}
                        onChange={(e) => setParams({ ...params, use_torch_compile: e.target.checked })}
                        className="rounded"
                      />
                      <label htmlFor="outpaint_use_torch_compile" className="text-sm text-gray-300">
                        torch.compile (Experimental, slow first run)
                      </label>
                    </div>

                    <div className="flex items-center gap-2 mt-4">
                      <input
                        type="checkbox"
                        id="outpaint_enable_block_swap"
                        checked={params.enable_block_swap || false}
                        onChange={(e) => setParams({ ...params, enable_block_swap: e.target.checked })}
                        className="rounded"
                      />
                      <label htmlFor="outpaint_enable_block_swap" className="text-sm text-gray-300">
                        Block Swap (Transformer offloading)
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
                            id="outpaint_use_pinned_memory"
                            checked={params.use_pinned_memory || false}
                            onChange={(e) => setParams({ ...params, use_pinned_memory: e.target.checked })}
                            className="rounded"
                          />
                          <label htmlFor="outpaint_use_pinned_memory" className="text-xs text-gray-300">
                            Use Pinned Memory (faster transfer, more RAM)
                          </label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            id="outpaint_block_swap_h2d_only"
                            checked={params.block_swap_h2d_only || false}
                            onChange={(e) => setParams({ ...params, block_swap_h2d_only: e.target.checked })}
                            className="rounded"
                          />
                          <label htmlFor="outpaint_block_swap_h2d_only" className="text-xs text-gray-300">
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
                      </div>
                    )}
                  </>
                )}
              </div>
            </details>

          </div>
        </Card>
        )}

        {isVideo && (
        <Card title="Video">
          {/* Resolution in the image models' Parameters-card shape: labelled
              sliders with a numeric entry beside them (common/Slider) in the
              same two-column grid, plus the Absolute/Scale size mode -- Scale
              derives width/height from the uploaded clip's own dimensions,
              rounded onto the ÷32 grid these controls advertise. */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-300">
                Size Mode
              </label>
              <div className="flex gap-2">
                <Button
                  onClick={() => handleVideoSizeModeChange("absolute")}
                  variant={videoSizeMode === "absolute" ? "primary" : "secondary"}
                  size="sm"
                >
                  Absolute
                </Button>
                <Button
                  onClick={() => handleVideoSizeModeChange("scale")}
                  variant={videoSizeMode === "scale" ? "primary" : "secondary"}
                  size="sm"
                  disabled={!inputVideoSize}
                  title={!inputVideoSize ? "Load an input clip first" : ""}
                >
                  Scale
                </Button>
              </div>
            </div>

            {videoSizeMode === "absolute" ? (
              <div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <Slider
                    label={`Width (÷${videoWidthBounds.step})`}
                    min={videoWidthBounds.min}
                    max={videoWidthBounds.max}
                    step={videoWidthBounds.step}
                    value={canvasWidth}
                    onChange={(e) => setParams({ ...params, width: parseInt(e.target.value) })}
                  />
                  <Slider
                    label={`Height (÷${videoHeightBounds.step})`}
                    min={videoHeightBounds.min}
                    max={videoHeightBounds.max}
                    step={videoHeightBounds.step}
                    value={canvasHeight}
                    onChange={(e) => setParams({ ...params, height: parseInt(e.target.value) })}
                  />
                </div>
                {/* Why a slider stops where it does. Only rendered for an
                    architecture that HAS an envelope -- LTX-2.3 declares none,
                    so it keeps its full range and says nothing about a cap. */}
                {videoWidthBounds.capped && (
                  <p className="text-xs text-gray-500 mt-1">
                    {videoCanvasRule(archCapabilities, loadedArchType)}. The cap is on the
                    short and long edges rather than on width and height, so each slider
                    stops at the largest edge the other axis currently allows.
                  </p>
                )}
                {videoCanvasOverEnvelope && (
                  <p className="text-xs text-amber-400 mt-1">
                    The canvas is {canvasWidth}x{canvasHeight}, which is outside this
                    model&apos;s envelope. The value is kept as set — it is not moved for
                    you — and this model refuses it, so change it before generating.
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
                  value={videoScale}
                  onChange={(e) => handleVideoScaleChange(parseFloat(e.target.value))}
                />
                {inputVideoSize && (
                  <p className="text-xs text-gray-500 mt-1">
                    Input clip: {inputVideoSize.width}x{inputVideoSize.height} ·{" "}
                    {videoCanvasRule(archCapabilities, loadedArchType)}
                  </p>
                )}
              </div>
            )}

            {/* WHAT THE CANVAS DOES TO THE INPUT CLIP. The backend fits the
                upload to width x height (centre-crop to the canvas aspect
                ratio, then resize) and the preserved span is that fitted
                result, so a canvas that is not the clip's own size changes
                both what "preserved" means and, at a different aspect ratio,
                how much of the frame survives. Stated, not advised. */}
            {inputVideoSize && (
              <div className="mt-2 space-y-1 text-xs">
                {canvasIsSourceSize ? (
                  <p className="text-gray-500">
                    The canvas is the input clip&apos;s own resolution, so the preserved
                    frames are the uploaded frames.
                  </p>
                ) : (
                  <p className="text-amber-400">
                    The canvas is {canvasWidth}x{canvasHeight}; the input clip is{" "}
                    {inputVideoSize.width}x{inputVideoSize.height}. The clip is fitted to
                    the canvas once — centre-cropped to the canvas aspect ratio, then
                    resized — and it is that fitted version, not the upload, that is
                    preserved frame for frame.
                    {canvasCropsSource && (
                      <>
                        {" "}The two aspect ratios differ, so the {croppedEdges} edges of
                        the clip are discarded by the crop.
                      </>
                    )}
                  </p>
                )}
                {videoCanvasAt1x && !videoCanvasAt1x.matchesSource && (
                  <p className="text-gray-500">
                    1x is not reachable for this clip on this model:{" "}
                    {videoCanvasRule(archCapabilities, loadedArchType)}. The nearest
                    canvas is {videoCanvasAt1x.width}x{videoCanvasAt1x.height}.
                  </p>
                )}
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
            <Slider
              label="Steps"
              min={1}
              max={100}
              step={1}
              value={params.num_inference_steps ?? 8}
              onChange={(e) => setParams({ ...params, num_inference_steps: parseInt(e.target.value) })}
            />
            <Slider
              label="Guidance Scale"
              min={0}
              max={20}
              step={0.1}
              value={params.guidance_scale ?? 1.0}
              onChange={(e) => setParams({ ...params, guidance_scale: parseFloat(e.target.value) })}
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
            <Slider
              label="Frame Rate (fps)"
              min={1}
              max={60}
              step={1}
              value={params.frame_rate ?? 24.0}
              onChange={(e) => setParams({ ...params, frame_rate: parseFloat(e.target.value) })}
            />
            {/* Seed: the image path's control (same label styling, same
                randomise / reset-to--1 / reuse-the-result's-seed buttons). */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Seed</label>
              <div className="flex gap-2">
                <NumberInput
                  value={params.seed ?? -1}
                  onCommit={(v) => setParams({ ...params, seed: v })}
                  parse="int"
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
                  title="Use seed from result video"
                  disabled={generatedVideoSeed === null}
                >
                  ♻️
                </Button>
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

          {params.audio_enable && (
            <div className="ml-6 mt-1">
              <Select
                label="Audio mode"
                value={params.outpaint_video_audio_mode || archAudioMode}
                onChange={(e) => setParams({ ...params, outpaint_video_audio_mode: e.target.value as "regenerate" | "preserve_input" })}
                options={[
                  { value: "regenerate", label: boundaryPlacementOnly ? "Regenerate (generated span only)" : "Regenerate whole track" },
                  { value: "preserve_input", label: "Preserve input clip audio" },
                ]}
              />
              <p className="text-xs text-gray-500 mt-1">
                "Preserve input clip audio" mutes the generated track over the placed span and mixes the uploaded clip's own audio in instead; falls back to "regenerate" (with a warning) if the clip has no audio stream.
              </p>
              {boundaryPlacementOnly && (
                <p className="text-xs text-gray-500 mt-1">
                  This model generates audio jointly with video, so it produces audio only for the
                  frames it generates. Under "Regenerate" the preserved span carries no audio and is
                  silent; "Preserve input clip audio" is what fills it
                  {archAudioMode === "preserve_input" && ", and is this model's default for that reason"}.
                </p>
              )}
            </div>
          )}

          <div className="flex items-center gap-2 mt-3">
            <input
              type="checkbox"
              id="outpaint_video_lossless"
              checked={params.video_lossless ?? false}
              onChange={(e) => setParams({ ...params, video_lossless: e.target.checked })}
              className="rounded"
            />
            <label htmlFor="outpaint_video_lossless" className="text-sm text-gray-300">Lossless (FFV1)</label>
          </div>
          {params.video_lossless && (
            <p className="text-xs text-gray-500 ml-6">
              Bit-exact frames, much larger file size, and generally not playable in a browser's native video element (FFV1 has no mainstream browser decoder).
            </p>
          )}

          <div className="text-sm font-semibold text-gray-400 mt-4 mb-1">Acceleration</div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="outpaint_vid_block_swap_enable"
              checked={(params.video_blocks_to_swap ?? 0) > 0}
              onChange={(e) => setParams({ ...params, video_blocks_to_swap: e.target.checked ? 10 : 0 })}
              className="rounded"
            />
            <label htmlFor="outpaint_vid_block_swap_enable" className="text-sm text-gray-300">
              Block Swap (Transformer offloading)
            </label>
          </div>
          {(params.video_blocks_to_swap ?? 0) > 0 && (
            <div className="ml-6 mt-1">
              {/* NumberInput puts `label` on aria-label only and renders no
                  visible text, so the caption is drawn here (the same way the
                  image panels' number fields are wrapped). */}
              <label className="block text-xs text-gray-400 mb-1">Blocks to swap</label>
              <NumberInput
                label="Blocks to swap"
                value={params.video_blocks_to_swap ?? 10}
                onCommit={(v) => setParams({ ...params, video_blocks_to_swap: Math.max(1, v) })}
                min={1}
                max={48}
                step={1}
                parse="int"
                className="w-24"
              />
            </div>
          )}

          {supportsSpectrum && (
          <div className="flex items-center gap-2 mt-2">
            <input
              type="checkbox"
              id="outpaint_vid_spectrum_enable"
              checked={params.spectrum_enable || false}
              onChange={(e) => setParams({ ...params, spectrum_enable: e.target.checked })}
              className="rounded"
            />
            <label htmlFor="outpaint_vid_spectrum_enable" className="text-sm text-gray-300">
              Spectrum (Spectral Feature Forecasting)
            </label>
            <span className="text-xs text-gray-500">(mutually exclusive with FBCache; disabled if Block Swap is on)</span>
          </div>
          )}
          {supportsSpectrum && params.spectrum_enable && (
            <div className="ml-6 mt-1 grid grid-cols-2 gap-2">
              <label className="text-xs text-gray-400 flex items-center gap-1">Mix w
                <input type="number" min={0} max={1} step={0.05} value={params.spectrum_w ?? 0.5}
                  onChange={(e) => setParams({ ...params, spectrum_w: parseFloat(e.target.value) })}
                  className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
              </label>
              <label className="text-xs text-gray-400 flex items-center gap-1">Warmup
                <input type="number" min={1} step={1} value={params.spectrum_warmup_steps ?? 3}
                  onChange={(e) => setParams({ ...params, spectrum_warmup_steps: parseInt(e.target.value) || 3 })}
                  className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs" />
              </label>
            </div>
          )}

          {supportsFbcache && (
          <div className="flex items-center gap-2 mt-2">
            <input
              type="checkbox"
              id="outpaint_vid_fbcache_enable"
              checked={params.fbcache_enable || false}
              onChange={(e) => setParams({ ...params, fbcache_enable: e.target.checked })}
              className="rounded"
            />
            <label htmlFor="outpaint_vid_fbcache_enable" className="text-sm text-gray-300">
              First Block Cache (dynamic caching)
            </label>
            <span className="text-xs text-gray-500">(mutually exclusive with Spectrum)</span>
          </div>
          )}
          {supportsFbcache && params.fbcache_enable && (
            <div className="ml-6 mt-1 grid grid-cols-2 gap-2">
              <label className="text-xs text-gray-400 flex items-center gap-1">Residual threshold
                <NumberInput min={0} step={0.01} parse="float" value={params.fbcache_threshold ?? 0.12}
                  defaultValue={0.12}
                  onCommit={(v) => setParams({ ...params, fbcache_threshold: v })}
                  className="w-20" />
              </label>
              <label className="text-xs text-gray-400 flex items-center gap-1">Warmup steps
                <NumberInput min={0} step={1} value={params.fbcache_warmup_steps ?? 1}
                  defaultValue={1}
                  onCommit={(v) => setParams({ ...params, fbcache_warmup_steps: v })}
                  className="w-20" />
              </label>
            </div>
          )}
        </Card>
        )}

        {isAudio && (
        <Card title="Audio Settings">

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
            <div>
              <label className="block text-xs text-gray-400 mb-1">Seed</label>
              <div className="flex gap-1">
                <NumberInput
                  value={params.seed ?? -1}
                  onCommit={(v) => setParams({ ...params, seed: v })}
                  parse="int"
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
                  onClick={() => generatedAudioSeed !== null && setParams({ ...params, seed: generatedAudioSeed })}
                  variant="secondary"
                  size="sm"
                  title="Use seed from result audio"
                  disabled={generatedAudioSeed === null}
                >
                  ♻️
                </Button>
              </div>
            </div>
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

        {!isVideo && visibility.lora && (
          <LoRASelector
            value={params.loras || []}
            onChange={(loras: LoRAConfig[]) => setParams({ ...params, loras })}
            disabled={isGenerating}
            storageKey="outpaint_lora_collapsed"
          />
        )}

        {!isVideo && !isAudio && visibility.controlnet && (
          <ControlNetSelector
            value={params.controlnets || []}
            onChange={(controlnets: ControlNetConfig[]) => setParams({ ...params, controlnets })}
            disabled={isGenerating}
            storageKey="outpaint_controlnet_collapsed"
            inputImagePreview={inputImagePreview}
          />
        )}

        {/* Loop Generation is intentionally NOT implemented for Outpaint
            (all phases) -- see the design doc §3.3 / §7 decision 9. */}
      </div>

      {/* Preview Panel */}
      <div className="space-y-4 pb-16 lg:pb-0">
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
            <div className="flex-1 flex flex-col space-y-2 min-w-0">
              {/* Action Buttons - Desktop only (hidden on mobile, which uses the
                  fixed bottom bar below instead, mirrors
                  Txt2ImgPanel/Img2ImgPanel/InpaintPanel). */}
              <div className="hidden lg:flex gap-2">
                <Button
                  onClick={handleAddToQueue}
                  variant="primary"
                  size="lg"
                  className="flex-1"
                  disabled={isVideo ? !videoFile : isAudio ? !audioFile : !inputImagePreview}
                >
                  {isGenerating ? "Add to Queue" : "Generate"}
                </Button>
                {isGenerating && (
                  <Button
                    onClick={async () => {
                      try {
                        await cancelGeneration();
                      } catch (error) {
                        console.error("[Outpaint] Failed to cancel generation:", error);
                      }
                    }}
                    variant="secondary"
                    size="lg"
                    title="Cancel generation and move to next"
                  >
                    Cancel
                  </Button>
                )}
                <Button onClick={resetToDefault} disabled={isGenerating} variant="secondary" size="lg">
                  Reset
                </Button>
              </div>

              {/* Action Buttons - Mobile only (fixed bar at bottom with inline toggle).
                  Mirrors Txt2ImgPanel/Img2ImgPanel/InpaintPanel's mobile bottom bar --
                  shares handleAddToQueue/cancelGeneration/resetToDefault with the
                  desktop button above (no generateForever/long-press here since
                  Outpaint doesn't implement that feature). */}
              <div className={`lg:hidden fixed bottom-0 z-40 bg-gray-900 border-t transition-all ${isMobileControlsOpen ? 'left-0 right-0 border-gray-700' : 'left-auto right-0 border-l border-gray-700'}`}>
                <div className="flex gap-2 p-3 items-center">
                  {isMobileControlsOpen && (
                    <>
                      <Button
                        onClick={handleAddToQueue}
                        disabled={isVideo ? !videoFile : isAudio ? !audioFile : !inputImagePreview}
                        className="flex-1"
                        size="lg"
                      >
                        {isGenerating ? "Add Queue" : "Generate"}
                      </Button>
                      {isGenerating && (
                        <button
                          onClick={async () => {
                            try {
                              await cancelGeneration();
                            } catch (error) {
                              console.error("[Outpaint] Failed to cancel generation:", error);
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

              {isGenerating && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-gray-400">
                    <span>{progressMessage || "Generating..."}</span>
                    <span>{Math.min(progress, totalSteps)}/{totalSteps} steps</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-200"
                      style={{ width: `${Math.min(100, Math.max(0, totalSteps > 0 ? (progress / totalSteps) * 100 : 0))}%` }}
                    />
                  </div>
                </div>
              )}

              <div
                className="w-full aspect-square max-h-[500px] lg:max-h-none bg-gray-800 rounded-lg flex items-center justify-center cursor-pointer"
                onDoubleClick={() => {
                  if (!isVideo && !isAudio && generatedImage) {
                    setPreviewViewerOpen(true);
                  }
                }}
              >
                {isVideo && generatedVideo ? (
                  <div className="w-full space-y-2">
                    <video
                      src={generatedVideoPlaybackUrl || generatedVideo}
                      className="w-full rounded-lg"
                      controls
                      loop
                      muted
                      autoPlay
                      playsInline
                      onError={() => {
                        // The file is gone (outputs/ cleared, run deleted) --
                        // show an empty preview rather than a dead player.
                        console.warn("[Outpaint] Preview video failed to load, clearing:", generatedVideo);
                        clearVideoPreview(PREVIEW_KEYS);
                        setGeneratedVideo(null);
                        setGeneratedVideoPlaybackUrl(null);
                        setGeneratedVideoInfo(null);
                        setGeneratedVideoSeed(null);
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
                        console.warn("[Outpaint] Preview audio failed to load, clearing:", generatedAudio);
                        clearAudioPreview(PREVIEW_KEYS);
                        setGeneratedAudio(null);
                        setGeneratedAudioInfo(null);
                        setGeneratedAudioSeed(null);
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
                        console.warn("[Outpaint] Preview image failed to load, clearing:", generatedImage);
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
                  <p className="text-gray-500">No image/video/audio generated yet</p>
                )}
              </div>

              {/* Post-Edit controls (client-side brightness/saturation/flatten) */}
              {!isVideo && !isAudio && generatedImage && (
                <PostEditControls value={postEdit} onChange={setPostEdit} />
              )}

              {!isVideo && !isAudio && generatedImage && (
                <div className="space-y-3 mt-4">
                  <div className="flex flex-wrap gap-2 text-sm">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" checked={sendImage} onChange={(e) => setSendImage(e.target.checked)} className="rounded" />
                      <span className="text-gray-300">Send image</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" checked={sendPrompt} onChange={(e) => setSendPrompt(e.target.checked)} className="rounded" />
                      <span className="text-gray-300">Send prompt</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" checked={sendParameters} onChange={(e) => setSendParameters(e.target.checked)} className="rounded" />
                      <span className="text-gray-300">Send parameters</span>
                    </label>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                    <Button onClick={sendToTxt2Img} variant="secondary" size="sm" disabled={!sendPrompt && !sendParameters}>
                      Send to txt2img
                    </Button>
                    <Button onClick={sendToImg2Img} variant="secondary" size="sm" disabled={!sendImage && !sendPrompt && !sendParameters}>
                      Send to img2img
                    </Button>
                    <Button onClick={sendToInpaintPanel} variant="secondary" size="sm" disabled={!sendImage && !sendPrompt && !sendParameters}>
                      Send to inpaint
                    </Button>
                    <Button onClick={sendToUpscale} variant="secondary" size="sm" disabled={!generatedImage}>
                      Send to Upscale
                    </Button>
                    <SendToStudioButton
                      media={{
                        kind: "image",
                        url: stripCacheBuster(generatedImage),
                        masterUrl: stripCacheBuster(generatedImage),
                        name: stripCacheBuster(generatedImage).split("/").pop() || "Generated image",
                        width: generatedImageParams?.width,
                        height: generatedImageParams?.height,
                      }}
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
                      className="col-span-2"
                      media={{
                        kind: "video",
                        url: generatedVideoPlaybackUrl || generatedVideo,
                        masterUrl: generatedVideo,
                        name: generatedVideo.split("/").pop() || "Generated video",
                        width: params.width,
                        height: params.height,
                        duration: generatedVideoInfo?.duration,
                      }}
                      parameters={{
                        ...(generatedVideoParams || params),
                        num_frames: generatedVideoInfo?.num_frames ?? generatedVideoParams?.total_frames ?? params.total_frames,
                        frame_rate: generatedVideoInfo?.fps ?? generatedVideoParams?.frame_rate ?? params.frame_rate,
                        seed: generatedVideoSeed ?? generatedVideoParams?.seed ?? params.seed,
                      }}
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
                      className="col-span-2"
                      media={{
                        kind: "audio",
                        url: generatedAudio,
                        masterUrl: generatedAudio,
                        name: generatedAudio.split("/").pop() || "Generated audio",
                        duration: generatedAudioInfo?.duration,
                      }}
                      parameters={{ ...(generatedAudioParams || params), seed: generatedAudioSeed ?? generatedAudioParams?.seed ?? params.seed }}
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="w-full">
              <GenerationQueue currentStep={progress} />
            </div>
          </ResizableColumns>
        </Card>
      </div>

      {/* Preview Image Viewer (image result only) */}
      {previewViewerOpen && generatedImage && (
        <ImageViewer
          imageUrl={generatedImage}
          onClose={() => setPreviewViewerOpen(false)}
          postEdit={postEdit}
          onPostEditChange={setPostEdit}
        />
      )}
    </ResizableColumns>
  );
}
