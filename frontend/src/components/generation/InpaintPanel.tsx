"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import { ChevronLeft, ChevronRight, X, RotateCcw, Maximize2, Minimize2 } from "lucide-react";
import Card from "../common/Card";
import TabbedOptions from "../common/TabbedOptions";
import Input from "../common/Input";
import NumberInput from "../common/NumberInput";
import Textarea, {
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
import FloatingGallery from "../common/FloatingGallery";
import ImageViewer from "../common/ImageViewer";
import PostEditControls from "../common/PostEditControls";
import VideoAccelerationControls from "../common/VideoAccelerationControls";
import { PostEditState, NEUTRAL_POST_EDIT, buildFilterString } from "@/utils/postEdit";
import { usePostEditPreview } from "@/hooks/usePostEditPreview";
import GenerationQueue from "../common/GenerationQueue";
import GenerationLeadGrid from "../common/GenerationLeadGrid";
import InlineHelp from "../common/InlineHelp";
import H3PromptAssist from "../common/H3PromptAssist";
import ResizableColumns, {
  GENERATION_PREVIEW_QUEUE_SPLIT_KEY,
  GENERATION_WORKSPACE_SPLIT_KEY,
} from "../common/ResizableColumns";
import LoopGenerationPanel, { LoopGenerationConfig } from "./LoopGenerationPanel";
import QuantizedGemmSelect from "./QuantizedGemmSelect";
import {
  MASK_OVERLAY_ALPHA,
  MASK_OVERLAY_CSS_MIX_BLEND_MODE,
  MASK_POLARITY,
  MASK_WHITE_LUMINANCE_THRESHOLD,
} from "@/utils/maskConventions";
import { migrateLoopGenerationConfig, computeLoopDecodeDirective } from "@/utils/loopGenerationInheritance";
import { getSamplers, getScheduleTypes, generateInpaint, generateInpaintVideo, generateInpaintTrainingPreview, toBase64, InpaintParams as ApiInpaintParams, InpaintVideoParams, LoRAConfig, ControlNetConfig, MiniMaxH3References, generateTIPOPrompt, cancelGeneration, getResultFilename, getResultPlaybackFilename, getResultSeed, getResultAncestralSeed, isLatentOnlyResult, unetQuantizationOptions, normalizeUnetQuantization, transformerQuantizationLabel, archSupportsFeature, archDisplayName, inpaintVideoDefaultsForArch, fitVideoCanvas, videoCanvasRule, videoCanvasAxisBounds, videoCanvasExceedsEnvelope, largestValidVideoFrameCount, isValidVideoFrameCount, latentGroupSpans, snapRangeToLatentGroups, isGenerationStalledError, VIDEO_BLOCK_SWAP_MAX } from "@/utils/api";
import MiniMaxH3ReferenceSelector, { EMPTY_MINIMAX_H3_REFERENCES, countMiniMaxH3References } from "../common/MiniMaxH3ReferenceSelector";
import VideoInpaintTimeline from "./VideoInpaintTimeline";
import VideoMaskPreviewOverlay from "./VideoMaskPreviewOverlay";
import VideoMaskFrameEditor from "./VideoMaskFrameEditor";
import { useActiveTraining } from "@/hooks/useActiveTraining";
import { useSnapshotHistory } from "@/hooks/useSnapshotHistory";
import { useSmoothProgress } from "@/hooks/useSmoothProgress";
import { useVideoPlayhead } from "@/hooks/useVideoPlayhead";
import { releaseVideoFrameGrabber } from "@/utils/videoFrameGrabber";
import { centerCropToCanvas } from "@/utils/canvasFit";
import {
  createDefaultMaskTransform,
  MAX_MASK_ASSETS,
  MAX_MASK_KEYFRAMES,
  serializeVideoMaskManifestForApi,
  sortKeyframes,
  upsertKeyframe,
  validateVideoMaskManifest,
  type VideoMaskAsset,
  type VideoMaskKeyframe,
  type VideoMaskManifest,
} from "@/utils/videoMaskTimeline";
import { wsClient, CFGMetrics } from "@/utils/websocket";
import CFGMetricsGraph from "../common/CFGMetricsGraph";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";
import {
  clipSignatureOf,
  clipSignaturesMatch,
  clearVideoMaskPersistence,
  loadVideoMaskManifest,
  persistVideoMaskManifest,
  releaseAllTrackedMaskAssets,
  VideoMaskTempStorageUnavailableError,
  type VideoMaskAssetRefMap,
  type VideoMaskClipSignature,
} from "@/utils/videoMaskPersistence";
import {
  deleteMediaInput,
  INPAINT_VIDEO_INPUT_KEY,
  INPAINT_VIDEO_PENDING_KEY,
  loadMediaInput,
  saveMediaInput,
} from "@/utils/mediaInputStorage";
import { previewStorageKeys, saveImagePreview, clearImagePreview, loadVideoPreview, saveVideoPreview, playbackUrlOf, clearVideoPreview, outputExists, stripCacheBuster, withCacheBuster, imagePreviewGone } from "@/utils/previewStorage";
import { sendToPanel, sendImageToImg2Img, sendImageToUpscale, sendImageToOutpaint, sendVideoToOutpaint, sendVideoToInpaint, sendVideoToReference, fetchUrlToFile } from "@/utils/sendHelpers";
import { fixFloatingPointParams } from "@/utils/numberUtils";
import { readGlobalAttentionType } from "@/utils/attentionSettings";
import { newId } from "@/utils/id";
import { useStartup } from "@/contexts/StartupContext";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";
import SendToStudioButton from "../studio/SendToStudioButton";
import { createH3ReferenceInventory, maybeTransformH3PromptForGeneration } from "@/utils/h3PromptAssist";

interface InpaintParams {
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
  mask_blur?: number;
  inpaint_full_res?: boolean;
  inpaint_full_res_padding?: number;
  inpaint_fill_mode?: string;
  inpaint_fill_strength?: number;
  inpaint_blur_strength?: number;
  resize_mode?: string;
  resampling_method?: string;
  // Regional additional prompt (SD/SDXL only): conditions ONLY the repaint
  // mask region, leaving the main prompt + preserved region untouched.
  region_prompt?: string;
  region_negative_prompt?: string;
  region_prompt_strength?: number;
  region_prompt_method?: string;
  region_mask_feather?: number;
  // Seam Structure Continuity (SSC, SD/SDXL only): continues thin structures
  // that cross the region boundary (a held rod/staff, limb, torso, lines)
  // into the generated/repainted region. x0-space, no extra U-Net forwards.
  // 0 = off.
  seam_structure_strength?: number;
  seam_structure_depth?: number;
  seam_structure_end?: number;
  seam_structure_saliency?: number;
  seam_structure_max_area?: number;
  // Boundary Determinism Relaxation (BDR, SD/SDXL only): soft-pins a narrow
  // saliency-gated seam band (annealed soft->hard) so the known-side latent
  // can bend to meet the continuation. Most effective with Seam Structure
  // Continuity > 0. 0 = off.
  boundary_relax_strength?: number;
  boundary_relax_width?: number;
  boundary_relax_noise?: number;
  boundary_relax_full_until?: number;
  boundary_relax_end?: number;
  boundary_relax_paste?: string;
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
  // Attention backend
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
  // Loop-generation decode mode (heavy-decoder aware; see loopGenerationInheritance.ts).
  // NOTE: inpaint does NOT support loop_decode="none" / input_latent_id — the
  // backend rejects it; intermediate loop steps fall back to "cheap"+skip_gallery.
  loop_decode?: "full" | "cheap" | "none";
  skip_gallery?: boolean;
  // Keys DEFAULT_PARAMS and the controls below already set, declared here so the
  // literal matches this type (they were being carried untyped).
  vae_drift_correction?: boolean;
  cpu_text_encoding?: boolean;
  use_torch_compile?: boolean;
  keep_models_hot?: boolean;
  vae_tiling?: boolean;
  vae_tile_threshold?: number;
  vae_tile_mode?: string;
  vae_tile_global_norm?: boolean;
  color_flatten_strength?: number;
  flatten_in_loop?: boolean;
  flatten_in_loop_last_steps?: number;
  flatten_in_loop_min_region?: number;
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
  spectrum_feature_mode?: string;
  spectrum_cache_branch?: number;
  spectrum_max_cache?: number;
  preview_predicted_x0?: boolean;
  preview_decoder?: string;
  feeling_lucky?: boolean;
  // ── Video temporal inpaint (POST /generate/inpaint/video) ────────────────
  // Kept in the same params object (and so in the same localStorage blob) as
  // the image fields, the way OutpaintPanel carries its three modalities.
  // Shared with the image path: prompt/negative_prompt/seed/width/height and
  // the spectrum/fbcache keys. Distinct where the routes differ:
  // `num_inference_steps` vs `steps`, `guidance_scale` vs `cfg_scale`, and
  // `video_blocks_to_swap` vs the image path's model-global `blocks_to_swap`.
  num_inference_steps?: number;
  guidance_scale?: number;
  frame_rate?: number;
  num_videos_per_prompt?: number;
  max_sequence_length?: number;
  audio_enable?: boolean;
  regenerate_start_frame?: number;
  regenerate_end_frame?: number;
  input_trim_start_frames?: number;
  input_trim_end_frames?: number;
  inpaint_video_audio_mode?: "regenerate" | "preserve_input" | "regenerate_range";
  // Which architecture the audio mode above was resolved for, so a per-arch
  // default is re-applied on a model change but a user's own choice is not
  // (the OutpaintPanel pattern; not sent to the backend).
  inpaint_video_audio_mode_arch?: string;
  video_lossless?: boolean;
  video_blocks_to_swap?: number;
}

const DEFAULT_PARAMS: InpaintParams = {
  prompt: "",
  negative_prompt: "",
  steps: 20,
  cfg_scale: 7.0,
  // SenseNova U1.5 flow-matching time-shift; every other architecture ignores it.
  timestep_shift: 3.0,
  // SenseNova U1.5 second CFG scale; inert without ref_images, ignored elsewhere.
  img_cfg_scale: 1.0,
  // SenseNova U1.5 per-phase weight-half CPU eviction; every other architecture ignores it.
  sensenova_mot_phase_eviction: false,
  // SenseNova U1.5 per-layer prefix KV cache CPU streaming; every other architecture ignores it.
  sensenova_kv_cache_streaming: false,
  sampler: "euler",
  schedule_type: "uniform",
  seed: -1,
  ancestral_seed: -1,
  width: 1024,
  height: 1024,
  denoising_strength: 0.75,
  img2img_fix_steps: true,
  vae_drift_correction: false,
  mask_blur: 4,
  inpaint_full_res: false,
  inpaint_full_res_padding: 32,
  inpaint_fill_mode: "original",
  inpaint_fill_strength: 1.0,
  inpaint_blur_strength: 1.0,
  resize_mode: "image",
  resampling_method: "lanczos",
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
  feeling_lucky: false,
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
  // Video temporal inpaint. These are the fallbacks for a backend that has not
  // answered yet; the real values come from /schema/generation-defaults
  // (`inpaint_vid` + the two overlay maps) in the effect below.
  num_inference_steps: 8,
  guidance_scale: 1.0,
  frame_rate: 24.0,
  num_videos_per_prompt: 1,
  max_sequence_length: 1024,
  audio_enable: true,
  // No default range is defensible (the route requires both fields); a clip
  // load picks the middle third and the user moves it from there.
  regenerate_start_frame: 0,
  regenerate_end_frame: 0,
  input_trim_start_frames: 0,
  input_trim_end_frames: 0,
  inpaint_video_audio_mode: "preserve_input",
  video_lossless: false,
  video_blocks_to_swap: 0,
  fuse_output_proj: false,
};

function createDefaultVideoMaskManifest(width?: number, height?: number): VideoMaskManifest {
  return {
    version: 1,
    coordinateSpace: "output_canvas",
    polarity: MASK_POLARITY,
    canvas: {
      width: Math.max(1, Math.round(width ?? 768)),
      height: Math.max(1, Math.round(height ?? 512)),
    },
    keyframes: [],
    compositeFeatherPx: 0,
  };
}

/** Undo/redo snapshot for the video-mask manifest (see `videoMaskHistory`
 * below, where this is owned) -- `keyframes`/`compositeFeatherPx`/`assets`
 * together, so undo/redo restores all three atomically and can never leave
 * a keyframe pointing at an asset that got garbage-collected (or vice
 * versa). */
interface MaskHistorySnapshot {
  keyframes: VideoMaskKeyframe[];
  compositeFeatherPx: number;
  assets: VideoMaskAsset[];
}

/** True if any pixel is past mid-grey. Mask polarity is white_generate, and
 * brush edges are anti-aliased, so this is checked against luminance rather
 * than requiring pure 0/255. */
function dataUrlHasWhitePixel(dataUrl: string): Promise<boolean> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      if (!image.naturalWidth || !image.naturalHeight) {
        resolve(false);
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d");
      if (!context) {
        resolve(false);
        return;
      }
      context.drawImage(image, 0, 0);
      const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] > MASK_WHITE_LUMINANCE_THRESHOLD) {
          resolve(true);
          return;
        }
      }
      resolve(false);
    };
    image.onerror = () => reject(new Error("The mask image could not be decoded."));
    image.src = dataUrl;
  });
}

// Inpaint's secondary options are grouped into a single-open tabbed accordion
// (see the "Inpaint Options" Card below, shared chrome via
// frontend/src/components/common/TabbedOptions.tsx — ported from
// OutpaintPanel's OUTPAINT_OPTIONS_TABS pattern). Every tab owns a disjoint
// set of param keys, used both by its "reset to default" button and by its
// active-highlight predicate (isInpaintOptionsTabActive below). LoRA/
// ControlNet are left outside the tabs (they're full component selectors,
// not param groups); Sampler/Steps/CFG Scale/Seed/Width/Height stay outside
// as core fields, matching OutpaintPanel.
type InpaintOptionsTabId =
  | "inpaint"
  | "regional_prompt"
  | "seam"
  | "cfg"
  | "acceleration"
  | "post_process"
  | "prompt_chunking"
  | "environment";

const INPAINT_OPTIONS_TABS: { id: InpaintOptionsTabId; label: string }[] = [
  { id: "inpaint", label: "Inpaint" },
  { id: "regional_prompt", label: "Regional Prompt" },
  { id: "seam", label: "Seam Continuity（継ぎ目・連続性）" },
  { id: "cfg", label: "CFG / NAG" },
  { id: "acceleration", label: "Acceleration（高速化）" },
  { id: "post_process", label: "Post-process（色補正）" },
  { id: "prompt_chunking", label: "Prompt Chunking" },
  { id: "environment", label: "Environment" },
];

const INPAINT_OPTIONS_TAB_KEYS: Record<InpaintOptionsTabId, (keyof InpaintParams)[]> = {
  inpaint: [
    "denoising_strength",
    "img2img_fix_steps",
    "mask_blur",
    "inpaint_fill_mode",
    "inpaint_fill_strength",
    "inpaint_blur_strength",
    "resize_mode",
    "resampling_method",
  ],
  regional_prompt: [
    "region_prompt",
    "region_negative_prompt",
    "region_prompt_strength",
    "region_prompt_method",
    "region_mask_feather",
  ],
  seam: [
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
// isOutpaintOptionsTabActive's rationale.
function isInpaintOptionsTabActive(tabId: InpaintOptionsTabId, params: InpaintParams): boolean {
  switch (tabId) {
    case "inpaint":
      return (
        (params.inpaint_fill_mode ?? "original") !== "original" ||
        (params.resize_mode ?? "image") !== "image" ||
        (params.resampling_method ?? "lanczos") !== "lanczos" ||
        !(params.img2img_fix_steps ?? true)
      );
    case "regional_prompt":
      return !!(params.region_prompt?.trim() || params.region_negative_prompt?.trim());
    case "seam":
      return (
        (params.seam_structure_strength ?? 0) > 0 ||
        (params.boundary_relax_strength ?? 0) > 0
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

const STORAGE_KEY = "inpaint_params";
// The hand-corrected clip frame COUNT, stored next to the clip signature it
// was entered for. Not part of `params` (the route derives the length from
// the trims, and never receives this), but it feeds videoRawFrames -- so
// losing it on a panel switch changed videoTrimmedFrames underneath a
// restored regenerate range and got the range discarded as "no longer fits".
const CLIP_FRAMES_OVERRIDE_STORAGE_KEY = "inpaint_clip_frames_override";
const PREVIEW_STORAGE_KEY = "inpaint_preview";
// Image and video results are mutually exclusive in storage: the panel writes
// whichever modality it just produced and the helper clears the other (see
// utils/previewStorage.ts). No audio key -- no architecture routed here
// produces audio.
const PREVIEW_KEYS = previewStorageKeys(PREVIEW_STORAGE_KEY);
const LOOP_GENERATION_STORAGE_KEY = "inpaint_loop_generation";
const INPUT_IMAGE_STORAGE_KEY = "inpaint_input_image";
const MASK_IMAGE_STORAGE_KEY = "inpaint_mask_image";
const REF_IMAGES_STORAGE_KEY = "inpaint_ref_images";

interface InpaintPanelProps {
  // opts.kind/playbackUrl let the shared top-right strip (FloatingGallery)
  // render video/audio results correctly instead of guessing from the URL
  // extension and falling back to a non-playable master URL.
  onImageGenerated?: (imageUrl: string, opts?: { kind?: "image" | "video" | "audio"; playbackUrl?: string }) => void;
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint" | "outpaint" | "upscale") => void;
}

export default function InpaintPanel({ onTabChange, onImageGenerated }: InpaintPanelProps = {}) {
  const { modelLoaded, modelInfo, isBackendReady, generationDefaults, archCapabilities, isVideo, resolveModality } = useStartup();
  const [params, setParams] = useState<InpaintParams>(DEFAULT_PARAMS);
  const [generatedImageParams, setGeneratedImageParams] = useState<InpaintParams | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
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
  const [inputImage, setInputImage] = useState<File | null>(null);
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null);
  const [inputImageSize, setInputImageSize] = useState<{ width: number; height: number } | null>(null);
  const [sizeMode, setSizeMode] = useState<"absolute" | "scale">("absolute");
  const [scale, setScale] = useState<number>(1.0);
  const [maskImage, setMaskImage] = useState<string | null>(null);
  // ── Video temporal inpaint (inpaint_vid) input clip + result ────────────
  // The input File is persisted in IndexedDB so it survives panel navigation
  // and browser reloads. The result remains a URL in preview storage.
  const [videoFile, setVideoFile] = useState<File | null>(null);
  // MiniMax-H3 ref2va temporal-inpaint references (images/videos/audios),
  // same shape and ordering rules as /generate/ref2vid's own h3References.
  // Only reachable when the loaded transformer is confirmed ref2va
  // (isH3Ref2VaInpaint below); fl2va and hybrid never render the control.
  const [h3References, setH3References] = useState<MiniMaxH3References>(
    EMPTY_MINIMAX_H3_REFERENCES,
  );
  const [h3ReferenceImageSize, setH3ReferenceImageSize] = useState<"max" | "match">("max");
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [videoDurationSec, setVideoDurationSec] = useState<number | null>(null);
  const [inputVideoSize, setInputVideoSize] = useState<{ width: number; height: number } | null>(null);
  const [videoMaskManifest, setVideoMaskManifest] = useState<VideoMaskManifest>(() =>
    createDefaultVideoMaskManifest(DEFAULT_PARAMS.width, DEFAULT_PARAMS.height),
  );
  const [videoMaskAssets, setVideoMaskAssets] = useState<VideoMaskAsset[]>([]);
  // Edit-time validation warnings only (e.g. affine-link demotion below).
  // Restore/persistence status is a SEPARATE channel (videoMaskPersistenceNotice)
  // so an unrelated edit can no longer silently clear a persistence notice the
  // user has not acted on -- see videoMaskPersistence.ts's module doc comment.
  const [videoMaskError, setVideoMaskError] = useState<string | null>(null);
  // ── Video mask reload-persistence (P5) ──────────────────────────────────
  // See videoMaskPersistence.ts's module doc comment for the storage model.
  // `videoMaskUploadedRefsRef` mirrors, for the CURRENT clip, which asset ids
  // this session has already uploaded and under what ref. `videoMaskHydrated`
  // gates the persist effect below until the mount-time restore attempt has
  // run at least once, so it never fires against the transient empty-default
  // manifest that exists for one render before restore applies.
  // `videoMaskRestoreAbortedRef` is set when a restore found a matching
  // persisted record but could not load one of its PNGs (transient, NOT
  // "permanently gone" -- see loadVideoMaskManifest's doc comment); while set,
  // the persist effect must not write anything. It is cleared -- via
  // `dismissAbortedVideoMaskRestore` -- the moment the user performs any
  // explicit edit on the mask timeline, since that supersedes whatever
  // restore was pending.
  const videoMaskUploadedRefsRef = useRef<VideoMaskAssetRefMap>(new Map());
  const videoMaskRestoreAbortedRef = useRef(false);
  // Bumped every time `videoMaskRestoreAbortedRef.current` is freshly set to
  // true. A plain ref mutation triggers no re-render, so the dedicated retry
  // effect below (keyed on `[isBackendReady, videoFile, videoMaskAbortEpoch]`)
  // would otherwise never get a chance to re-evaluate when the abort happens
  // in a tick where NEITHER `isBackendReady` NOR `videoFile` also changes
  // (e.g. the backend was already marked ready but the one temp-storage
  // fetch during this restore hit a transient failure) -- this epoch exists
  // purely to give that effect a dependency to react to in that case.
  const [videoMaskAbortEpoch, setVideoMaskAbortEpoch] = useState(0);
  const [videoMaskHydrated, setVideoMaskHydrated] = useState(false);
  // Restore/persistence status shown to the user: an aborted restore, a
  // discarded pending restore (superseded by an edit), or a persist attempt
  // that fell back to inline storage and was refused (see
  // VideoMaskTempStorageUnavailableError). Deliberately not `videoMaskError`
  // (see that state's comment).
  const [videoMaskPersistenceNotice, setVideoMaskPersistenceNotice] = useState<string | null>(null);
  // Serializes every persistVideoMaskManifest/clearVideoMaskPersistence call
  // this panel instance issues into a single FIFO chain: neither function has
  // its own lock, and firing them concurrently (fast keyframe edits, a clip
  // replacement landing mid-upload, etc.) can race on the shared
  // `videoMaskUploadedRefsRef` map / localStorage record. See
  // videoMaskPersistence.ts's module doc comment. Errors are swallowed here
  // (each op logs/handles its own) so one failure never wedges the chain for
  // later ops.
  const videoMaskPersistChainRef = useRef<Promise<void>>(Promise.resolve());
  // Monotonic counter handed to persistVideoMaskManifest as its `isCurrent`
  // check: only the MOST RECENTLY enqueued persist op is allowed to perform
  // its localStorage write, so an op that becomes stale while queued behind
  // others (state changed again before it got a turn) skips a write that
  // would otherwise clobber a newer one written by the op ahead of/behind it.
  const videoMaskPersistGenerationRef = useRef(0);
  const enqueueVideoMaskPersistOp = useCallback((op: () => Promise<void>) => {
    videoMaskPersistChainRef.current = videoMaskPersistChainRef.current.then(op, op).catch((error) => {
      console.error("[Inpaint] A queued video mask persistence operation failed:", error);
    });
  }, []);
  // Distinguishes overlapping invocations of the `inpaint_input_video_updated`
  // handler (mount + a sender's replace event firing again before the first
  // invocation's awaits have resolved) from one another, since the `cancelled`
  // flag below only distinguishes "still mounted" from "unmounted" and cannot
  // tell two in-flight invocations of the SAME still-mounted effect apart.
  const videoInputRunSeqRef = useRef(0);
  // Local UI state (not a generation param): whether the rasterized mask
  // preview overlay is drawn on top of the input clip, and its opacity.
  // Defaults on -- the whole point of the overlay is to be visible by
  // default while editing keyframes.
  const [videoMaskPreviewEnabled, setVideoMaskPreviewEnabled] = useState(true);
  const [videoMaskPreviewOpacity, setVideoMaskPreviewOpacity] = useState(0.6);
  // The input clip's own <video>, ref'd so the Regenerate Range timeline can
  // read/drive its live playhead (useVideoPlayhead below) -- this is the same
  // element the panel already renders for preview, not a second one.
  const inputVideoRef = useRef<HTMLVideoElement | null>(null);
  // The wrapper the <video> and its mask overlay canvas are siblings inside.
  // Fullscreen is requested on THIS element, not the <video> itself: the
  // native video fullscreen path promotes only the <video> into the top
  // layer, leaving the overlay canvas behind (and unsized, since it would
  // still be reading its small non-fullscreen CSS box) -- see
  // ImageEditor.tsx's own container-fullscreen pattern.
  const videoContainerRef = useRef<HTMLDivElement | null>(null);
  const [isVideoFullscreen, setIsVideoFullscreen] = useState(false);
  // Feature-detected once, not just "does `requestFullscreen` exist": iOS
  // Safari has no element-level fullscreen for an arbitrary <div> at all
  // (only `HTMLVideoElement.webkitEnterFullscreen`, which promotes the
  // native player and would leave this component's own mask-overlay canvas
  // behind exactly like the bug the container-fullscreen pattern above
  // exists to avoid) -- there is nothing this button can correctly do there,
  // so it does not render rather than being clickable and silently failing.
  const canFullscreenContainer = useMemo(() => {
    if (typeof document === "undefined") return false;
    const doc = document as Document & { webkitFullscreenEnabled?: boolean };
    const docEl = document.documentElement as HTMLElement & { webkitRequestFullscreen?: () => void };
    const enabled = doc.fullscreenEnabled ?? doc.webkitFullscreenEnabled ?? false;
    const hasRequest = typeof docEl.requestFullscreen === "function" || typeof docEl.webkitRequestFullscreen === "function";
    return !!enabled && hasRequest;
  }, []);
  const toggleVideoFullscreen = useCallback(async () => {
    const container = videoContainerRef.current as (HTMLDivElement & { webkitRequestFullscreen?: () => Promise<void> | void }) | null;
    if (!container) return;
    const doc = document as Document & { webkitFullscreenElement?: Element | null; webkitExitFullscreen?: () => Promise<void> | void };
    try {
      const fullscreenElement = doc.fullscreenElement ?? doc.webkitFullscreenElement ?? null;
      if (fullscreenElement === container) {
        if (document.exitFullscreen) await document.exitFullscreen();
        else if (doc.webkitExitFullscreen) await doc.webkitExitFullscreen();
      } else if (container.requestFullscreen) {
        await container.requestFullscreen();
      } else if (container.webkitRequestFullscreen) {
        await container.webkitRequestFullscreen();
      }
    } catch (error) {
      console.error("[Inpaint] Failed to toggle input clip fullscreen", error);
    }
  }, []);
  useEffect(() => {
    const handleFullscreenChange = () => {
      const doc = document as Document & { webkitFullscreenElement?: Element | null };
      const fullscreenElement = doc.fullscreenElement ?? doc.webkitFullscreenElement ?? null;
      setIsVideoFullscreen(fullscreenElement === videoContainerRef.current);
    };
    // Chromium/Firefox fire `fullscreenchange`; older WebKit (Safari) fires
    // the prefixed `webkitfullscreenchange` instead -- both are registered
    // so `isVideoFullscreen` (and therefore the Exit-fullscreen affordance
    // in `handleClearVideo` below) stays correct on either.
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
      document.removeEventListener("webkitfullscreenchange", handleFullscreenChange);
    };
  }, []);
  const [videoSizeMode, setVideoSizeMode] = useState<"absolute" | "scale">("absolute");
  const [videoScale, setVideoScale] = useState<number>(1.0);
  // The clip's frame COUNT, which the browser does not report: it is estimated
  // from duration x frame rate and can be corrected by hand, because the
  // trimmed length has to be exactly on the architecture's grid and an estimate
  // one frame out would otherwise only surface as the route's 400.
  const [clipFramesOverride, setClipFramesOverride] = useState<number | null>(null);
  // The clip source `applyClipLength` was last run for -- see
  // handleVideoLoadedMetadata.
  const lastClipLengthAppliedSrcRef = useRef<string | null>(null);
  const [generatedVideo, setGeneratedVideo] = useState<string | null>(null);
  // Playback source for the <video> element, when it differs from
  // generatedVideo (a video_lossless FFV1-in-mkv master no browser can
  // decode): its H.264 mp4 proxy. generatedVideo itself stays the master
  // for send-to/reference actions. Falls back to generatedVideo when null.
  const [generatedVideoPlaybackUrl, setGeneratedVideoPlaybackUrl] = useState<string | null>(null);
  const [generatedVideoInfo, setGeneratedVideoInfo] = useState<{ num_frames?: number; fps?: number; duration?: number } | null>(null);
  const [generatedVideoSeed, setGeneratedVideoSeed] = useState<number | null>(null);
  const [generatedVideoParams, setGeneratedVideoParams] = useState<InpaintParams | null>(null);
  // The run's `warnings[]`, shown under the result. The panel snaps the range to
  // latent-group boundaries itself, so the range-snap warning should not fire
  // for a request built here -- which is exactly why it is worth showing if it
  // does. Session-only: the gallery row keeps its own copy.
  const [generatedVideoWarnings, setGeneratedVideoWarnings] = useState<string[]>([]);
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
  // Keep the legacy response shape locally while reading the shared SSOT. This
  // panel no longer issues its own GET /models/current on mount or model change.
  const currentModelInfo = modelInfo ? { loaded: true, model_info: modelInfo } : null;
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

  // ── Video temporal inpaint: the loaded architecture's own rules ──────────
  const loadedArchType = currentModelInfo?.model_info?.type as string | undefined;
  const loadedArchName = archDisplayName(loadedArchType);
  // Applies a LoRA's own declared recommended settings (from its file
  // metadata) to params, like any ordinary user edit -- see Txt2ImgPanel's
  // twin of this function for the full rationale. Inpaint has no audio
  // modality, so the only mapping is image steps vs video steps.
  const applyLoraRecommended = (settings: Record<string, unknown>): string[] => {
    const skipped: string[] = [];
    const updates: Partial<InpaintParams> = {};
    if (typeof settings.num_inference_steps === "number") {
      if (isVideo) updates.num_inference_steps = settings.num_inference_steps;
      else updates.steps = settings.num_inference_steps;
    }
    if (typeof settings.fbcache_enable === "boolean") {
      updates.fbcache_enable = settings.fbcache_enable;
    }
    if (typeof settings.spectrum_enable === "boolean") {
      updates.spectrum_enable = settings.spectrum_enable;
    }
    setParams({ ...params, ...updates });
    return skipped;
  };
  // The capability the video surface is gated on. `archSupportsFeature` treats
  // an unknown arch (or a matrix that has not loaded) as supporting it, so the
  // surface is never hidden merely because the matrix was unavailable.
  const archSupportsTemporalInpaint = archSupportsFeature(archCapabilities, loadedArchType, "temporal_inpaint");
  // MiniMax-H3's gate is per-PARTITION, and archSupportsFeature cannot express
  // it: it is keyed on architecture, not on the loaded transformer file.
  // `/generate/inpaint/video` now serves fl2va (no references) AND ref2va
  // (references optional, including audio-only -- the interior pin already
  // supplies vision conditioning), refuses `hybrid` unconditionally, and
  // refuses an UNIDENTIFIED variant only when a reference is attached
  // (generation_utils.resolve_minimax_h3_inpaint_reference_gate). Only
  // `hybrid` hides the whole surface here; ref2va's own reference limits are
  // enforced by the selector/route, not by hiding the tab.
  const h3Variant = loadedArchType === "minimax_h3"
    ? (currentModelInfo?.model_info?.variant as string | undefined)
    : undefined;
  const h3TemporalInpaintRefused = h3Variant === "hybrid";
  const supportsTemporalInpaint = archSupportsTemporalInpaint && !h3TemporalInpaintRefused;
  // Renders the References card only for a confirmed ref2va checkpoint --
  // fl2va was never trained to read reference rows (fl2va + a reference is a
  // 400), and an unidentified variant's reference support cannot be told
  // apart from fl2va's absence of it.
  const isH3Ref2VaInpaint = h3Variant === "ref2va";
  // M3 fix: the References card only exists while isH3Ref2VaInpaint is true.
  // Without this, attaching references then switching to an fl2va checkpoint
  // unmounted the card but left h3References populated, so every later
  // Generate click hit the freshH3Variant guard below and alerted "clear the
  // references" while pointing at a control no longer on screen. Clearing
  // the state the moment the surface goes away keeps state and UI in sync
  // (the request itself was already safe either way -- hasH3References at
  // enqueue time already dropped them when !isH3Ref2VaInpaint).
  useEffect(() => {
    if (!isH3Ref2VaInpaint) {
      setH3References(EMPTY_MINIMAX_H3_REFERENCES);
    }
  }, [isH3Ref2VaInpaint]);
  const supportsNegativePrompt = archSupportsFeature(archCapabilities, loadedArchType, "negative_prompt");
  // Hide Spectrum/FBCache when the loaded sampler does not consume them; H3 now
  // supports both. This matches the other panels' leaf-control convention.
  const supportsSpectrum = archSupportsFeature(archCapabilities, loadedArchType, "spectrum");
  const supportsFbcache = archSupportsFeature(archCapabilities, loadedArchType, "fbcache");
  const supportsFuseOutputProj = archSupportsFeature(archCapabilities, loadedArchType, "fuse_output_proj");
  const supportsTimestepShift = archSupportsFeature(archCapabilities, loadedArchType, "timestep_shift");
  const supportsImgCfgScale = archSupportsFeature(archCapabilities, loadedArchType, "img_cfg_scale");
  const supportsSensenovaMotPhaseEviction = archSupportsFeature(archCapabilities, loadedArchType, "sensenova_mot_phase_eviction");
  const supportsSensenovaKvCacheStreaming = archSupportsFeature(archCapabilities, loadedArchType, "sensenova_kv_cache_streaming");
  // The value the Block Swap checkbox writes when turned ON (backend SSOT:
  // param_defaults.VIDEO_GEN_DEFAULTS["blocks_to_swap_enabled_default"]). The
  // `?? 40` fallback only matters before /schema/generation-defaults answers.
  const videoBlocksToSwapEnabledDefault =
    (generationDefaults?.inpaint_vid as Record<string, unknown> | undefined)
      ?.blocks_to_swap_enabled_default as number ?? 40;
  // Mirrors the substance of the route's own 400 text for the one variant
  // this endpoint still refuses outright. Only reached when h3Variant ===
  // "hybrid" -- fl2va and ref2va both serve this endpoint now (fl2va without
  // references, ref2va with or without), and an unidentified variant is only
  // refused if a reference is actually attached, which the References card
  // never renders for it in the first place.
  const h3TemporalInpaintReason =
    "The loaded MiniMax-H3 transformer is the hybrid variant. A merged checkpoint is "
    + "released for text-to-video only; temporal inpaint was not part of the comparison "
    + "that released it and is refused.";
  const temporalInpaintReason = !archSupportsTemporalInpaint
    ? (loadedArchType ? archCapabilities?.unsupported?.[loadedArchType]?.temporal_inpaint : undefined)
    : (h3TemporalInpaintRefused ? h3TemporalInpaintReason : undefined);
  const videoConstraints = loadedArchType ? archCapabilities?.video_constraints?.[loadedArchType] : undefined;
  const latentChunkPattern = videoConstraints?.latent_chunk_pattern ?? [];
  // Frames of the upload. The browser exposes duration, not a frame count, so
  // this is duration x the architecture's own rate (or the requested one, for
  // an arch with no fixed rate) unless the user corrected it.
  const clipFrameRate = videoConstraints?.fps_fixed ?? params.frame_rate ?? 24.0;
  const estimatedRawFrames = videoDurationSec != null
    ? Math.max(1, Math.round(videoDurationSec * clipFrameRate))
    : 0;
  const videoRawFrames = clipFramesOverride ?? estimatedRawFrames;
  // The Regenerate Range timeline's synced playhead/seek/loop, against the
  // SAME <video> rendered above -- attachKey is the preview URL so the hook
  // re-attaches whenever a new clip is loaded into that element.
  const inputVideoPlayer = useVideoPlayhead(inputVideoRef, clipFrameRate, videoPreviewUrl);
  const videoTrimmedFrames = Math.max(
    0,
    videoRawFrames - (params.input_trim_start_frames ?? 0) - (params.input_trim_end_frames ?? 0)
  );
  // What the backend's `plan_video_inpaint_span` actually regenerates: the
  // requested range expanded OUTWARD to latent-group boundaries, never
  // shrunk (mirrors VideoInpaintTimeline's own `effective` range). The mask
  // preview overlay must clamp to THIS span, not the raw requested one --
  // the backend applies the held first/last keyframe mask across the whole
  // snapped span, including the frames the snap added.
  const videoMaskOverlaySpan = snapRangeToLatentGroups(
    latentGroupSpans(latentChunkPattern, videoTrimmedFrames),
    params.regenerate_start_frame ?? 0,
    params.regenerate_end_frame ?? 0,
  );
  // The clip length itself has to be on the grid here (temporal inpaint samples
  // the whole clip), and the route refuses an off-grid length rather than
  // snapping it, so the panel computes the trim that reaches a valid one.
  const videoTrimmedLengthValid = isValidVideoFrameCount(archCapabilities, loadedArchType, videoTrimmedFrames);
  const videoTargetLength = videoRawFrames > 0
    ? largestValidVideoFrameCount(archCapabilities, loadedArchType, videoRawFrames)
    : null;
  // Canvas: the envelope is on the short/long edge rather than per axis, so each
  // slider's ceiling depends on where the other axis sits.
  const videoWidthBounds = videoCanvasAxisBounds(archCapabilities, loadedArchType, params.height ?? 0);
  const videoHeightBounds = videoCanvasAxisBounds(archCapabilities, loadedArchType, params.width ?? 0);
  const videoCanvasOverEnvelope = videoCanvasExceedsEnvelope(
    archCapabilities, loadedArchType, params.width ?? 0, params.height ?? 0);
  const videoCanvasIsSourceSize = !!inputVideoSize
    && params.width === inputVideoSize.width
    && params.height === inputVideoSize.height;

  useEffect(() => {
    return () => {
      if (videoPreviewUrl) {
        releaseVideoFrameGrabber(videoPreviewUrl);
        URL.revokeObjectURL(videoPreviewUrl);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoPreviewUrl]);

  // A range that no longer fits the trimmed clip (the trim moved, or the frame
  // count was corrected) is re-centred rather than sent as an out-of-range
  // request the route would refuse. `lastValidRegenerateRangeRef` remembers the
  // last range that DID fit, purely so the replacement can be reported: without
  // it, the very first run of this effect (against the 0/0 default, before the
  // user has picked anything) would be reported the same way as a real
  // discard of a range the user actually selected.
  const lastValidRegenerateRangeRef = useRef<{ start: number; end: number } | null>(null);
  const [regenerateRangeReplacedNotice, setRegenerateRangeReplacedNotice] = useState<string | null>(null);
  useEffect(() => {
    if (!isVideo || videoTrimmedFrames <= 0) return;
    const start = params.regenerate_start_frame ?? 0;
    const end = params.regenerate_end_frame ?? 0;
    if (start < end && end <= videoTrimmedFrames) {
      lastValidRegenerateRangeRef.current = { start, end };
      return;
    }
    const spans = latentGroupSpans(latentChunkPattern, videoTrimmedFrames);
    const snapped = snapRangeToLatentGroups(
      spans, Math.floor(videoTrimmedFrames / 3), Math.ceil((2 * videoTrimmedFrames) / 3));
    const previous = lastValidRegenerateRangeRef.current;
    lastValidRegenerateRangeRef.current = { start: snapped.start, end: snapped.end };
    setParams(prev => ({ ...prev, regenerate_start_frame: snapped.start, regenerate_end_frame: snapped.end }));
    setRegenerateRangeReplacedNotice(
      previous
        ? `The previously selected regenerate range (${previous.start} to ${previous.end}) no longer fits the trimmed clip (${videoTrimmedFrames} frames) and was replaced with ${snapped.start} to ${snapped.end}.`
        : null
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isVideo, videoTrimmedFrames]);

  // Mask keyframes outside the regenerate range are deliberately NOT pruned
  // here: this effect (and VideoInpaintTimeline's onRangeChange, which fires
  // continuously while a handle is dragged) used to delete them, which
  // meant dragging a handle across a keyframe and back discarded it and its
  // PNG asset before the pointer was ever released. Out-of-range keyframes
  // are kept and surfaced (still editable) inside VideoInpaintTimeline
  // itself; generation already refuses to submit
  // while any exist (see the `outOfRangeKeyframe` check near the submit
  // handler).

  // The audio mode's default is per-architecture (MiniMax-H3 overlays
  // "preserve_input"), and every value is selectable everywhere, so the trigger
  // is the ARCHITECTURE changing rather than a value being out of range -- the
  // OutpaintPanel pattern: re-resolve from the same overlay chain the backend
  // resolves from and record which arch the answer belongs to, so a choice the
  // user makes afterwards is never overwritten.
  const archAudioMode =
    (inpaintVideoDefaultsForArch(generationDefaults, loadedArchType)
      .inpaint_video_audio_mode as "regenerate" | "preserve_input" | "regenerate_range" | undefined)
    ?? DEFAULT_PARAMS.inpaint_video_audio_mode!;
  useEffect(() => {
    if (!generationDefaults || !loadedArchType) return;
    setParams(prev => (
      prev.inpaint_video_audio_mode_arch === loadedArchType
        ? prev
        : { ...prev, inpaint_video_audio_mode: archAudioMode, inpaint_video_audio_mode_arch: loadedArchType }
    ));
  }, [generationDefaults, loadedArchType, archAudioMode]);

  const [isDragging, setIsDragging] = useState(false);
  const [showImageEditor, setShowImageEditor] = useState(false);
  const [editingImageUrl, setEditingImageUrl] = useState<string | null>(null);
  // The static (single-frame) mask editor and the video mask editor are now
  // separate mounts (VideoMaskFrameEditor owns frame navigation/grabbing/
  // caching itself, P4) -- this only records which frame a video-mask
  // editing SESSION was opened for; everything else (which frame is
  // currently open, its base image, its mask) lives inside
  // VideoMaskFrameEditor and is re-derived from `videoMaskManifest`/
  // `videoMaskAssets` (passed down as props) on every navigation.
  const [videoMaskEditorSession, setVideoMaskEditorSession] = useState<{ initialFrame: number } | null>(null);
  const videoMaskCanvasWidth = Math.max(1, Math.round(params.width ?? 768));
  const videoMaskCanvasHeight = Math.max(1, Math.round(params.height ?? 512));
  // `videoMaskManifest.canvas` is intentionally NOT kept in sync with the
  // live output size here. It records the canvas size the existing
  // keyframes/assets were actually drawn for (set in
  // persistVideoMaskFrame whenever a mask is saved); comparing that
  // stored value against the current output canvas is what lets
  // `videoMaskCanvasMismatch` (below) and the submit-time check flag a stale
  // mask instead of either silently reusing pixels drawn for a different
  // resolution or discarding every keyframe and its PNG the instant a size
  // slider is touched.
  // Per-asset, not per-manifest: `videoMaskManifest.canvas` only ever records
  // the size at the LAST save (persistVideoMaskFrame overwrites it
  // every time), so after a resize a fresh save on one keyframe makes the
  // manifest-level canvas match again even though older sibling keyframes'
  // PNGs are still sized for the pre-resize canvas. Each asset carries its
  // own width/height (set at save time) precisely so this check can catch
  // that case; an asset missing those fields (nothing in this session
  // predates them) falls back to the manifest-level comparison.
  const referencedVideoMaskAssetIds = new Set(videoMaskManifest.keyframes.map((keyframe) => keyframe.maskId));
  const staleVideoMaskAssets = videoMaskAssets.filter((asset) => {
    if (!referencedVideoMaskAssetIds.has(asset.id)) return false;
    if (asset.width !== undefined && asset.height !== undefined) {
      return asset.width !== videoMaskCanvasWidth || asset.height !== videoMaskCanvasHeight;
    }
    return videoMaskManifest.canvas.width !== videoMaskCanvasWidth
      || videoMaskManifest.canvas.height !== videoMaskCanvasHeight;
  });
  const videoMaskCanvasMismatch = videoMaskManifest.keyframes.length > 0
    && staleVideoMaskAssets.length > 0;
  const [sendImage, setSendImage] = useState(true);
  const [sendPrompt, setSendPrompt] = useState(true);
  const [sendParameters, setSendParameters] = useState(true);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  // ── Training-preview integration (mirrors Txt2Img / Img2Img panels) ──
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
  const [galleryImages, setGalleryImages] = useState<Array<{ url: string; timestamp: number }>>([]);
  const [maxGalleryImages, setMaxGalleryImages] = useState(30);
  const [previewViewerOpen, setPreviewViewerOpen] = useState(false);
  const [showAdvancedCFG, setShowAdvancedCFG] = useState(false);

  // FLUX.2 Image Edit / Vision Encoder / SenseNova U1.5: Reference images
  const [refImages, setRefImages] = useState<File[]>([]);
  const [refImagePreviews, setRefImagePreviews] = useState<string[]>([]);
  const [isRefImageDragging, setIsRefImageDragging] = useState(false);
  // SenseNova's reference-image count cap mirrors the backend's
  // SENSENOVA_MAX_REFERENCE_IMAGES (backend/core/pipeline_backends/sensenova.py).
  // FLUX.2 has no backend-enforced cap; 10 is this UI's own upload-grid limit.
  const isSenseNovaModel = currentModelInfo?.model_info?.type === "sensenova";
  const maxRefImages = isSenseNovaModel ? 5 : 10;

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
  const preserveVideoSettingsRef = useRef(false);

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
    let cancelled = false;
    // console.clear(); // Temporarily disabled for debugging
    console.log("=== InpaintPanel mounted ===");
    setIsMounted(true);

    const loadInitialData = async () => {
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

      // Load preview image
      const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
      if (savedPreview) {
        setGeneratedImage(savedPreview);
      }

      // Preview video (inpaint_vid result). Restored unconditionally: the
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

      // Load input image preview
      const savedInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      console.log("[Inpaint] Initial load - input image ref:", savedInputRef);
      if (savedInputRef) {
        // NOTE: Allow old-style references (direct URLs) for now
        // // Check if it's an old-style reference (direct URL like /outputs/... or http://...)
        // if (savedInputRef.startsWith('/outputs/') || savedInputRef.startsWith('http://') || savedInputRef.startsWith('https://')) {
        //   console.log("[Inpaint] Detected old-style input image reference, clearing storage");
        //   localStorage.removeItem(INPUT_IMAGE_STORAGE_KEY);
        // } else if (savedInputRef.startsWith('temp_img://') || savedInputRef.startsWith('data:')) {
        try {
          const imageData = await loadTempImage(savedInputRef);
          if (cancelled) return;
          console.log("[Inpaint] Input image loaded successfully:", imageData ? "yes" : "no");
          if (imageData) {
            setInputImagePreview(imageData);
            // Load image dimensions
            const img = new Image();
            img.onload = () => {
              console.log("[Inpaint] Input image dimensions set:", img.width, "x", img.height);
              setInputImageSize({ width: img.width, height: img.height });
            };
            img.src = imageData;
          }
          // } else {
          //   console.warn("[Inpaint] Invalid input image data, clearing storage");
          //   localStorage.removeItem(INPUT_IMAGE_STORAGE_KEY);
          // }
        } catch (error) {
          console.error("[Inpaint] Failed to load input image:", error);
        }
        // } else {
        //   console.warn("[Inpaint] Unknown input image reference format, clearing storage");
        //   localStorage.removeItem(INPUT_IMAGE_STORAGE_KEY);
        // }
      }

      // Load mask image preview
      const savedMaskRef = localStorage.getItem(MASK_IMAGE_STORAGE_KEY);
      console.log("[Inpaint] Initial load - mask image ref:", savedMaskRef);
      if (savedMaskRef) {
        // NOTE: Allow old-style references (direct URLs) for now
        // // Check if it's an old-style reference
        // if (savedMaskRef.startsWith('/outputs/') || savedMaskRef.startsWith('http://') || savedMaskRef.startsWith('https://')) {
        //   console.log("[Inpaint] Detected old-style mask image reference, clearing storage");
        //   localStorage.removeItem(MASK_IMAGE_STORAGE_KEY);
        // } else if (savedMaskRef.startsWith('temp_img://') || savedMaskRef.startsWith('data:')) {
        try {
          const imageData = await loadTempImage(savedMaskRef);
          if (cancelled) return;
          console.log("[Inpaint] Mask image loaded successfully:", imageData ? "yes" : "no");
          if (imageData) {
            setMaskImage(imageData);
          }
          // } else {
          //   console.warn("[Inpaint] Invalid mask image data, clearing storage");
          //   localStorage.removeItem(MASK_IMAGE_STORAGE_KEY);
          // }
        } catch (error) {
          console.error("[Inpaint] Failed to load mask image:", error);
        }
        // } else {
        //   console.warn("[Inpaint] Unknown mask image reference format, clearing storage");
        //   localStorage.removeItem(MASK_IMAGE_STORAGE_KEY);
        // }
      }

      // Load max gallery images setting
      const savedMaxImages = localStorage.getItem('floating_gallery_max_images');
      if (savedMaxImages) {
        setMaxGalleryImages(parseInt(savedMaxImages));
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
      const savedVisibility = localStorage.getItem('inpaint_visibility');
      if (savedVisibility) {
        try {
          setVisibility(JSON.parse(savedVisibility));
        } catch (e) {
          console.error('Failed to parse inpaint visibility:', e);
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
          console.log(`[Inpaint] Loading ${refRefs.length} reference images from storage`);

          const restored = await Promise.all(refRefs.map(async (ref) => {
            try {
              const imageData = await loadTempImage(ref);
              return imageData || null;
            } catch (error) {
              console.error(`[Inpaint] Failed to load reference image ${ref}:`, error);
              return null;
            }
          }));
          if (cancelled) return;
          const loadedPreviews = restored.filter((value): value is string => value !== null);

          if (loadedPreviews.length > 0) {
            setRefImagePreviews(loadedPreviews);
            console.log(`[Inpaint] Restored ${loadedPreviews.length} reference images`);
          }
        } catch (error) {
          console.error('[Inpaint] Failed to parse reference images storage:', error);
        }
      }

      // Mark initial load as complete
      if (!cancelled) setIsInitialLoad(false);
      console.log("[Inpaint] Initial load complete");
    };

    void loadInitialData();
    return () => {
      cancelled = true;
    };
  }, []);

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

  // When backend becomes ready, reload temp images if not already loaded
  useEffect(() => {
    if (isBackendReady) {
      let cancelled = false;
      const reloadImages = async () => {
        console.log("[Inpaint] Backend ready, reloading images if needed");

        // Reload the preview image if it's a backend URL, and verify it is
        // still there first (outputs/ can be cleared, or the run deleted from
        // the gallery) -- the same rule the video/audio previews follow in the
        // panels that have them. Non-`/outputs/` values (a data: URL, a blob:,
        // a path served from elsewhere) are left untouched: they cannot go
        // missing server-side and must never be stamped or discarded. The
        // stamp is applied only to a URL that verified, and it replaces any
        // earlier stamp rather than appending.
        const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
        if (savedPreview && savedPreview.startsWith('/outputs/')) {
          const previewPath = stripCacheBuster(savedPreview);
          const previewStillThere = await outputExists(previewPath);
          if (cancelled) return;
          if (!previewStillThere) {
            console.log("[Inpaint] Stored preview image is gone, clearing:", previewPath);
            clearImagePreview(PREVIEW_KEYS);
            setGeneratedImage(null);
          } else {
            console.log("[Inpaint] Reloading preview image from backend:", previewPath);
            setGeneratedImage(withCacheBuster(previewPath));
          }
        }

        // Same verification for a restored preview video. No cache-busting
        // stamp -- an .mp4 is large and its URL is stable.
        const savedVideo = loadVideoPreview(PREVIEW_KEYS);
        if (savedVideo && !(await outputExists(savedVideo.url))) {
          if (cancelled) return;
          console.log("[Inpaint] Stored preview video is gone, clearing:", savedVideo.url);
          clearVideoPreview(PREVIEW_KEYS);
          setGeneratedVideo(null);
          setGeneratedVideoPlaybackUrl(null);
          setGeneratedVideoInfo(null);
          setGeneratedVideoSeed(null);
        }

        // Reload input image if not loaded
        if (!inputImagePreview) {
          const savedInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
          if (savedInputRef) {
            try {
              const imageData = await loadTempImage(savedInputRef);
              if (cancelled) return;
              if (imageData) {
                setInputImagePreview(imageData);
                const img = new Image();
                img.onload = () => {
                  setInputImageSize({ width: img.width, height: img.height });
                };
                img.src = imageData;
              }
            } catch (error) {
              console.error("[Inpaint] Failed to reload input image after backend ready:", error);
            }
          }
        }

        // Reload mask image if not loaded
        if (!maskImage) {
          const savedMaskRef = localStorage.getItem(MASK_IMAGE_STORAGE_KEY);
          if (savedMaskRef) {
            try {
              const imageData = await loadTempImage(savedMaskRef);
              if (cancelled) return;
              if (imageData) {
                setMaskImage(imageData);
              }
            } catch (error) {
              console.error("[Inpaint] Failed to reload mask image after backend ready:", error);
            }
          }
        }

      };

      void reloadImages();
      return () => {
        cancelled = true;
      };
    }
  }, [isBackendReady]);

  // Retry the video mask manifest restore if the mount-time attempt aborted
  // (temp storage unreachable at the time) -- mirrors the input/mask image
  // retry above, but as its OWN effect keyed on `videoFile` (and
  // `videoMaskAbortEpoch`) too: the mount-time restore attempt runs while
  // `videoFile` is still null (the clip is loaded asynchronously by the
  // separate `inpaint_input_video_updated` handler), so folding this into the
  // `[isBackendReady]`-only effect meant it fired exactly once, at a point
  // `videoFile` was always null, and then never again for the lifetime of the
  // panel -- the one case that most needs a retry (a persisted record whose
  // PNGs could not be loaded) never got one. `videoMaskAbortEpoch` closes the
  // remaining gap where the abort itself happens without `isBackendReady` or
  // `videoFile` also changing (see its declaration).
  useEffect(() => {
    if (!isBackendReady || !videoMaskRestoreAbortedRef.current || !videoFile) return;
    let cancelled = false;
    (async () => {
      try {
        const outcome = await loadVideoMaskManifest(clipSignatureOf(videoFile));
        if (cancelled) return;
        if (outcome.status === "ok") {
          setVideoMaskManifest(outcome.manifest);
          setVideoMaskAssets(outcome.assets);
          videoMaskUploadedRefsRef.current = outcome.refs;
          videoMaskRestoreAbortedRef.current = false;
          setVideoMaskPersistenceNotice(null);
        } else if (outcome.status === "none") {
          // The record no longer matches this clip (or was cleared/
          // superseded meanwhile) -- stop retrying.
          videoMaskRestoreAbortedRef.current = false;
          setVideoMaskPersistenceNotice(null);
        }
        // "aborted" again: leave the flag (and notice) set; this effect
        // re-runs on the next `videoFile`/`isBackendReady` change. Does NOT
        // bump `videoMaskAbortEpoch` itself -- that would make this retry
        // fire itself again immediately in a tight loop on a persistently
        // unreachable backend.
      } catch (error) {
        console.error("[Inpaint] Retry of video mask manifest restore failed:", error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isBackendReady, videoFile, videoMaskAbortEpoch]);

  // Persist the video mask timeline (keyframes + feather + per-keyframe PNG
  // assets) so a reload does not lose keyframes drawn for a clip that is
  // still loaded. Gated on `videoMaskHydrated` (see its declaration) so this
  // never runs before the mount-time restore attempt above has had a chance
  // to apply whatever it found, and skipped entirely while
  // `videoMaskRestoreAbortedRef` is set (an unresolved restore for a
  // matching persisted record must not be overwritten by the empty state
  // that restore left in place).
  //
  // Every call is enqueued through `enqueueVideoMaskPersistOp` (FIFO chain)
  // instead of fired directly: this effect can run again -- with a newer
  // manifest/assets snapshot -- before a previous call's uploads/deletes
  // have finished, and without serialization the two calls' awaits interleave
  // arbitrarily on the SAME `videoMaskUploadedRefsRef` map and localStorage
  // key (orphaned uploads, a stale write clobbering a newer one, a deleted
  // keyframe's PNG being deleted out from under a still-in-flight upload of
  // it). `videoMaskPersistGenerationRef` is bumped for every enqueued call so
  // `persistVideoMaskManifest` can skip its own write if a newer call has
  // already been enqueued behind it by the time it gets its turn.
  useEffect(() => {
    if (!isMounted || !videoMaskHydrated) return;
    if (videoMaskRestoreAbortedRef.current) return;
    const signature = clipSignatureOf(videoFile);
    if (!signature) return; // no clip loaded -- nothing meaningful to persist against
    const manifestSnapshot = videoMaskManifest;
    const assetsSnapshot = videoMaskAssets;
    const generation = ++videoMaskPersistGenerationRef.current;
    // Captured NOW (not read lazily as `videoMaskUploadedRefsRef.current`
    // inside the queued closure below): a clip replacement/reset landing
    // before this op's turn comes reassigns `videoMaskUploadedRefsRef.current`
    // to a brand-new Map for the NEXT clip (see resetVideoMaskTimeline and
    // the mount handler) rather than mutating this one in place, precisely
    // so an op that captures the map object eagerly, like this one, keeps
    // operating on the map it was actually enqueued for.
    const refsMap = videoMaskUploadedRefsRef.current;
    enqueueVideoMaskPersistOp(() =>
      persistVideoMaskManifest(
        manifestSnapshot,
        assetsSnapshot,
        signature,
        refsMap,
        () => videoMaskPersistGenerationRef.current === generation,
      ).catch((error) => {
        if (error instanceof VideoMaskTempStorageUnavailableError) {
          setVideoMaskPersistenceNotice(
            "The video mask timeline could not be saved for reload right now (backend temp storage " +
            "unavailable). Your current keyframes are unaffected in this session; saving will retry " +
            "the next time the mask timeline changes.",
          );
          return;
        }
        console.error("[Inpaint] Failed to persist video mask manifest:", error);
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoMaskManifest, videoMaskAssets, videoFile, isMounted, videoMaskHydrated]);

  useEffect(() => {
    // Listen for input image updates from txt2img or img2img
    const handleInputUpdate = () => {
      const newInput = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (newInput) {
        loadTempImage(newInput).then((imageData) => {
          if (imageData) {
            setInputImagePreview(imageData);
          }
        }).catch((error) => {
          console.error("Failed to load updated input image:", error);
        });
      }
    };

    window.addEventListener("inpaint_input_updated", handleInputUpdate);

    return () => {
      window.removeEventListener("inpaint_input_updated", handleInputUpdate);
    };
  }, []);

  // Restore the persisted clip on mount and reload it when a sender replaces
  // the IndexedDB record. The legacy URL path is retained for older tabs.
  useEffect(() => {
    let cancelled = false;
    const handleVideoInputUpdate = async () => {
      // A fresh run id for THIS invocation: mount fires this handler once,
      // and a sender's replace event can fire it again before that first
      // invocation's `await`s (loadMediaInput/loadVideoMaskManifest, etc.)
      // have resolved. `isStale()` lets every await-resumption below tell
      // "a newer invocation has since started" apart from "I am still the
      // most recent invocation" -- `cancelled` alone only ever catches
      // unmount, not this overlap.
      const runId = ++videoInputRunSeqRef.current;
      const isStale = () => cancelled || runId !== videoInputRunSeqRef.current;
      const url = localStorage.getItem("inpaint_input_video");
      const isReplacement = url !== null || localStorage.getItem(INPAINT_VIDEO_PENDING_KEY) === "1";
      try {
        const file = url
          ? await fetchUrlToFile(url)
          : await loadMediaInput(INPAINT_VIDEO_INPUT_KEY);
        if (!file || isStale()) {
          if (!file && !isStale()) {
            // No clip at all -- nothing a persisted mask manifest could
            // still apply to. Release any leftover backend temp PNGs from a
            // previous session/clip rather than let them leak indefinitely.
            // The ref is REASSIGNED to a fresh Map (not `.clear()`ed in
            // place) so a still-queued persist call for the old clip -- which
            // captured the old Map object, not this ref, at its own enqueue
            // time -- keeps operating on that unaffected object; the stale
            // object is handed to releaseAllTrackedMaskAssets so uploads it
            // makes are still released even if its own write gets skipped.
            const staleRefs = videoMaskUploadedRefsRef.current;
            videoMaskUploadedRefsRef.current = new Map();
            videoMaskRestoreAbortedRef.current = false;
            setVideoMaskPersistenceNotice(null);
            enqueueVideoMaskPersistOp(() =>
              releaseAllTrackedMaskAssets(staleRefs).catch((error) =>
                console.error("[Inpaint] Failed to clear a stale video mask record:", error),
              ),
            );
          }
          return;
        }
        if (url) await saveMediaInput(INPAINT_VIDEO_INPUT_KEY, file);
        if (isStale()) return;
        // Everything that describes the clip must be settled BEFORE its src
        // is attached below: `loadedmetadata` fires once and never again for
        // an unchanged src, so anything landing after it (these used to sit
        // past the `await` further down) is never reconciled.
        if (!isReplacement) {
          // Same clip, so a frame count corrected by hand for it still
          // applies. The two branches are mutually exclusive with the
          // `setClipFramesOverride(null)` below.
          try {
            const storedOverride = localStorage.getItem(CLIP_FRAMES_OVERRIDE_STORAGE_KEY);
            if (storedOverride) {
              const parsedOverride = JSON.parse(storedOverride) as { clip?: unknown; frames?: unknown };
              const frames = parsedOverride.frames;
              if (
                typeof frames === "number"
                && Number.isFinite(frames)
                && frames > 0
                && clipSignaturesMatch(
                  parsedOverride.clip as VideoMaskClipSignature | null,
                  clipSignatureOf(file),
                )
              ) {
                setClipFramesOverride(Math.round(frames));
              }
            }
          } catch (error) {
            console.error("[Inpaint] Failed to restore the corrected clip frame count:", error);
          }
        }
        setVideoDurationSec(null);
        setInputVideoSize(null);
        if (isReplacement) setClipFramesOverride(null);
        preserveVideoSettingsRef.current = !isReplacement;
        setVideoPreviewUrl(prev => {
          if (prev) URL.revokeObjectURL(prev);
          return URL.createObjectURL(file);
        });
        setVideoFile(file);
        if (isReplacement) {
          setVideoMaskManifest((previous) =>
            createDefaultVideoMaskManifest(previous.canvas.width, previous.canvas.height),
          );
          setVideoMaskAssets([]);
          setVideoMaskEditorSession(null);
          setVideoMaskError(null);
          setVideoMaskPersistenceNotice(null);
          // A replacement clip invalidates any mask timeline drawn for the
          // old one -- release its backend temp PNGs instead of leaking them.
          // See the "no clip" branch above on why the ref is reassigned
          // rather than cleared in place.
          {
            const staleRefs = videoMaskUploadedRefsRef.current;
            videoMaskUploadedRefsRef.current = new Map();
            videoMaskRestoreAbortedRef.current = false;
            enqueueVideoMaskPersistOp(() =>
              releaseAllTrackedMaskAssets(staleRefs).catch((error) =>
                console.error("[Inpaint] Failed to clear the replaced clip's video mask record:", error),
              ),
            );
          }
        } else {
          // Mount-time restore of the SAME clip (mediaInputStorage.ts is a
          // single-slot record, not content-addressed, so this is the one
          // path where a persisted manifest could still apply): try to
          // reload the mask timeline saved for this clip's signature.
          try {
            const outcome = await loadVideoMaskManifest(clipSignatureOf(file));
            if (!isStale()) {
              if (outcome.status === "ok") {
                setVideoMaskManifest(outcome.manifest);
                setVideoMaskAssets(outcome.assets);
                videoMaskUploadedRefsRef.current = outcome.refs;
                videoMaskRestoreAbortedRef.current = false;
                setVideoMaskPersistenceNotice(null);
              } else if (outcome.status === "aborted") {
                videoMaskRestoreAbortedRef.current = true;
                setVideoMaskAbortEpoch((epoch) => epoch + 1);
                setVideoMaskPersistenceNotice(
                  "Saved video mask keyframes for this clip could not be loaded and have not been restored. " +
                  "This will retry automatically; use Discard below if it does not recover.",
                );
              }
              // "none": no persisted record for this clip -- nothing to restore.
            }
          } catch (error) {
            console.error("[Inpaint] Failed to restore video mask manifest:", error);
          }
        }
      } catch (error) {
        console.error("[Inpaint] Failed to restore input video:", error);
      } finally {
        if (url) localStorage.removeItem("inpaint_input_video");
        localStorage.removeItem(INPAINT_VIDEO_PENDING_KEY);
        // Always flips, including on the "no clip"/error paths above -- the
        // persist effect below must never fire before this first restore
        // attempt (successful or not) has had its chance to run. Skipped
        // when stale: a newer invocation is already in flight and will flip
        // this itself once IT completes.
        if (!isStale()) setVideoMaskHydrated(true);
      }
    };
    window.addEventListener("inpaint_input_video_updated", handleVideoInputUpdate);
    void handleVideoInputUpdate();
    return () => {
      cancelled = true;
      window.removeEventListener("inpaint_input_video_updated", handleVideoInputUpdate);
    };
  }, []);

  // Load image dimensions when inputImagePreview changes
  useEffect(() => {
    if (inputImagePreview) {
      const img = new Image();
      img.onload = () => {
        setInputImageSize({ width: img.width, height: img.height });

        // Apply global send size mode settings when image is loaded from send
        const sendSizeMode = localStorage.getItem('send_size_mode') as "absolute" | "scale" | null;
        if (sendSizeMode === 'scale') {
          setSizeMode('scale');
          const sendDefaultScale = parseFloat(localStorage.getItem('send_default_scale') || '1.0');
          setScale(sendDefaultScale);
          // Update dimensions based on scale
          const scaledWidth = Math.round(img.width * sendDefaultScale / 64) * 64;
          const scaledHeight = Math.round(img.height * sendDefaultScale / 64) * 64;
          setParams(prev => ({ ...prev, width: scaledWidth, height: scaledHeight }));
        } else if (sendSizeMode === 'absolute') {
          // Absolute mode - use image dimensions as-is
          setSizeMode('absolute');
          setScale(1.0);
          setParams(prev => ({ ...prev, width: img.width, height: img.height }));
        } else if (sizeMode === "scale") {
          // No global setting, but already in scale mode - use current scale
          const scaledWidth = Math.round(img.width * scale / 64) * 64;
          const scaledHeight = Math.round(img.height * scale / 64) * 64;
          setParams((prev) => ({ ...prev, width: scaledWidth, height: scaledHeight }));
        }
      };
      img.src = inputImagePreview;
    }
  }, [inputImagePreview]);

  // Save params to localStorage whenever they change (but only after mounted and initial load complete)
  useEffect(() => {
    if (isMounted && !isInitialLoad) {
      // Only save if params are different from what's in localStorage
      // This prevents overwriting params sent from Gallery/other panels
      const saved = localStorage.getItem(STORAGE_KEY);
      const savedParams = saved ? JSON.parse(saved) : null;
      const currentParamsStr = JSON.stringify(params);
      const savedParamsStr = savedParams ? JSON.stringify(savedParams) : null;

      if (currentParamsStr !== savedParamsStr) {
        console.log('[Inpaint] Params changed by user, saving to localStorage:', {
          loras: params.loras?.length || 0,
          controlnets: params.controlnets?.length || 0,
          prompt_length: params.prompt?.length || 0,
        });
        localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
      }
    }
  }, [params, isMounted, isInitialLoad]);

  // Persist the hand-corrected clip frame count alongside the clip it was
  // entered for. Gated on `videoMaskHydrated` for the same reason the mask
  // manifest persist effect is: the mount restore has to have had its turn
  // first, or this would write the pre-restore `null` over the stored value.
  useEffect(() => {
    if (!isMounted || !videoMaskHydrated) return;
    const signature = clipSignatureOf(videoFile);
    if (clipFramesOverride == null || !signature) {
      localStorage.removeItem(CLIP_FRAMES_OVERRIDE_STORAGE_KEY);
      return;
    }
    localStorage.setItem(
      CLIP_FRAMES_OVERRIDE_STORAGE_KEY,
      JSON.stringify({ clip: signature, frames: clipFramesOverride }),
    );
  }, [clipFramesOverride, videoFile, isMounted, videoMaskHydrated]);

  // Listen for localStorage changes from Gallery/Preview (send to feature)
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && e.newValue) {
        try {
          const parsed = JSON.parse(e.newValue);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          const fixed = fixFloatingPointParams(merged);
          setParams(fixed);
          console.log("[Inpaint] Params updated from storage event (cross-tab)");
        } catch (error) {
          console.error("[Inpaint] Failed to parse storage change:", error);
        }
      }
    };

    const handleCustomStorageChange = () => {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          const fixed = fixFloatingPointParams(merged);
          setParams(fixed);
          console.log("[Inpaint] Params updated from custom storage event (same-tab)");
        } catch (error) {
          console.error("[Inpaint] Failed to parse custom storage change:", error);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('inpaint_params_updated', handleCustomStorageChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('inpaint_params_updated', handleCustomStorageChange);
    };
  }, []);

  // Reload params from localStorage when navigating to /generate?tab=inpaint (from Gallery)
  useEffect(() => {
    if (pathname === "/generate" && searchParams.get('tab') === 'inpaint' && isMounted) {
      console.log("[Inpaint] Page navigated to inpaint tab, reloading params from localStorage");
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          const fixed = fixFloatingPointParams(merged);
          setParams(fixed);
          console.log("[Inpaint] Params reloaded:", {
            prompt_length: fixed.prompt?.length || 0,
            steps: fixed.steps,
            cfg_scale: fixed.cfg_scale,
          });
        } catch (error) {
          console.error("[Inpaint] Failed to reload params on navigation:", error);
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

  // Save the preview video (inpaint_vid result) the same way. Only the URL, the
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

  // Save loop generation config to localStorage whenever it changes
  useEffect(() => {
    if (isMounted) {
      localStorage.setItem(LOOP_GENERATION_STORAGE_KEY, JSON.stringify(loopGenerationConfig));
    }
  }, [loopGenerationConfig, isMounted]);

  // Apply backend-fetched defaults when they arrive (only if no localStorage
  // value exists). Both dicts are merged: the image ones (`inpaint`) and the
  // video-inpaint ones (`inpaint_vid`). Only the keys the VIDEO route owns are
  // pulled from the second -- a blind spread would clobber the image mode's
  // `blocks_to_swap` with the video route's, which is why that one is carried
  // under `video_blocks_to_swap` (the OutpaintPanel precedent).
  useEffect(() => {
    if (!generationDefaults) return;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) return;
    const vid = (generationDefaults.inpaint_vid || {}) as Record<string, unknown>;
    setParams(prev => ({
      ...DEFAULT_PARAMS,
      ...(generationDefaults.inpaint as Partial<typeof DEFAULT_PARAMS>),
      frame_rate: (vid.frame_rate as number) ?? DEFAULT_PARAMS.frame_rate,
      num_inference_steps: (vid.num_inference_steps as number) ?? DEFAULT_PARAMS.num_inference_steps,
      guidance_scale: (vid.guidance_scale as number) ?? DEFAULT_PARAMS.guidance_scale,
      num_videos_per_prompt: (vid.num_videos_per_prompt as number) ?? DEFAULT_PARAMS.num_videos_per_prompt,
      max_sequence_length: (vid.max_sequence_length as number) ?? DEFAULT_PARAMS.max_sequence_length,
      audio_enable: (vid.audio_enable as boolean) ?? DEFAULT_PARAMS.audio_enable,
      input_trim_start_frames: (vid.input_trim_start_frames as number) ?? DEFAULT_PARAMS.input_trim_start_frames,
      input_trim_end_frames: (vid.input_trim_end_frames as number) ?? DEFAULT_PARAMS.input_trim_end_frames,
      inpaint_video_audio_mode:
        (vid.inpaint_video_audio_mode as "regenerate" | "preserve_input" | "regenerate_range")
        ?? DEFAULT_PARAMS.inpaint_video_audio_mode,
      video_lossless: (vid.video_lossless as boolean) ?? DEFAULT_PARAMS.video_lossless,
      video_blocks_to_swap: (vid.blocks_to_swap as number) ?? DEFAULT_PARAMS.video_blocks_to_swap,
      fuse_output_proj: (vid.fuse_output_proj as boolean) ?? DEFAULT_PARAMS.fuse_output_proj,
    }));
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
    // Clear mask when new image is loaded
    setMaskImage(null);
    if (isMounted) {
      // Delete old mask reference
      const oldMaskRef = localStorage.getItem(MASK_IMAGE_STORAGE_KEY);
      if (oldMaskRef) {
        deleteTempImageRef(oldMaskRef).catch(console.error);
      }
      localStorage.removeItem(MASK_IMAGE_STORAGE_KEY);
    }

    const reader = new FileReader();
    reader.onload = async (event) => {
      const preview = event.target?.result as string;
      setInputImagePreview(preview);

      if (isMounted) {
        // Delete old input image reference
        const oldInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
        if (oldInputRef) {
          await deleteTempImageRef(oldInputRef).catch(console.error);
        }

        // Save new image and store reference
        try {
          const imageRef = await saveTempImage(preview);
          localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, imageRef);
        } catch (error) {
          console.error("Failed to save input image:", error);
          // Fallback to direct storage for small images
          localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, preview);
        }
      }

      // Load image to get dimensions
      const img = new Image();
      img.onload = () => {
        setInputImageSize({ width: img.width, height: img.height });
        // If in scale mode, update width/height based on scale
        if (sizeMode === "scale") {
          const scaledWidth = Math.round(img.width * scale / 64) * 64;
          const scaledHeight = Math.round(img.height * scale / 64) * 64;
          setParams({ ...params, width: scaledWidth, height: scaledHeight });
        }
      };
      img.src = preview;
    };
    reader.readAsDataURL(file);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processImageFile(file);
    }
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
    if (file) {
      processImageFile(file);
    }
  };

  const handleInputImageDoubleClick = () => {
    if (inputImagePreview) {
      setEditingImageUrl(inputImagePreview);
      setShowImageEditor(true);
    }
  };

  const handleEditorSave = async (editedImageUrl: string) => {
    setInputImagePreview(editedImageUrl);
    if (isMounted) {
      try {
        // Delete old reference and save new one
        const oldRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
        if (oldRef) {
          await deleteTempImageRef(oldRef);
        }
        const imageRef = await saveTempImage(editedImageUrl);
        localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, imageRef);
      } catch (error) {
        console.error("Failed to save edited input image:", error);
        // Fallback to direct storage
        localStorage.setItem(INPUT_IMAGE_STORAGE_KEY, editedImageUrl);
      }
    }
    setShowImageEditor(false);
  };

  const handleEditorSaveMask = async (maskUrl: string) => {
    setMaskImage(maskUrl);
    if (isMounted) {
      try {
        // Delete old reference and save new one
        const oldRef = localStorage.getItem(MASK_IMAGE_STORAGE_KEY);
        if (oldRef) {
          await deleteTempImageRef(oldRef);
        }
        const imageRef = await saveTempImage(maskUrl);
        localStorage.setItem(MASK_IMAGE_STORAGE_KEY, imageRef);
      } catch (error) {
        console.error("Failed to save mask image:", error);
        // Fallback to direct storage
        localStorage.setItem(MASK_IMAGE_STORAGE_KEY, maskUrl);
      }
    }
  };

  const handleEditorClose = () => {
    setShowImageEditor(false);
  };

  // Opens a video-mask editing session at `frame`. Grabbing/cropping the
  // frame image and resolving which mask to show now happen INSIDE
  // VideoMaskFrameEditor (its own `navigate` effect, keyed off
  // `initialFrame`) rather than here, so this is just a synchronous state
  // flip -- the async work, and its double-click/race guard, moved with it.
  const openVideoMaskEditor = (frame: number) => {
    if (!videoPreviewUrl || videoTrimmedFrames <= 0) return;
    setVideoMaskError(null);
    setVideoMaskEditorSession({ initialFrame: Math.max(0, Math.round(frame)) });
  };

  const handleVideoMaskEditorClose = () => {
    setVideoMaskEditorSession(null);
  };

  const handleAddVideoMaskKeyframe = (frame: number) => {
    openVideoMaskEditor(frame);
  };

  const handleEditVideoMaskKeyframe = (keyframe: VideoMaskKeyframe) => {
    // No longer passed through to the editor: VideoMaskFrameEditor looks up
    // whichever keyframe/asset currently sits at `keyframe.frame` from the
    // live `keyframes`/`assets` props itself (frame is the stable lookup
    // key end to end now, not a keyframeId captured at open time).
    openVideoMaskEditor(keyframe.frame);
  };

  // Shared by handleVideoMaskKeyframesChange and persistVideoMaskFrame
  // so both paths enforce the same affine invariant (an "affine" link
  // requires identical maskId on both ends) instead of only one of them
  // catching a mismatch it itself introduced.
  const demoteMismatchedAffineLinks = (
    keyframes: VideoMaskKeyframe[],
  ): { keyframes: VideoMaskKeyframe[]; changed: boolean } => {
    const ordered = sortKeyframes(keyframes);
    let changed = false;
    const next = ordered.map((keyframe, index) => {
      const nextKeyframe = ordered[index + 1];
      if (
        nextKeyframe
        && keyframe.interpolationToNext === "affine"
        && keyframe.maskId !== nextKeyframe.maskId
      ) {
        changed = true;
        return { ...keyframe, interpolationToNext: "hold" as const };
      }
      return keyframe;
    });
    return { keyframes: next, changed };
  };

  // ---------------------------------------------------------------------
  // Undo/redo for the video-mask manifest, owned HERE (not inside
  // VideoInpaintTimeline) because this panel is the only place that sees
  // EVERY way the keyframe/asset list changes: VideoInpaintTimeline's own
  // controls (duplicate/delete/transform/interpolation/frame-move/composite
  // feather, via `handleVideoMaskKeyframesChange`/`handleVideoMaskFeatherChange`
  // below) AND drawing a brand-new mask (`persistVideoMaskFrame`, called from
  // VideoMaskFrameEditor's `onSaveFrame` prop, which VideoInpaintTimeline
  // never sees). A history stack that only covered the
  // first group -- as a prior version of this undo/redo did -- lets undo
  // replace the keyframe list wholesale with a snapshot that predates a
  // mask added by drawing, silently deleting that keyframe AND its saved
  // PNG asset. Keeping `assets` inside every snapshot (not just
  // keyframes/compositeFeatherPx) closes the other half of that gap: the
  // asset garbage-collection below is computed ONCE per edit, folded into
  // the snapshot that gets pushed, and restored verbatim by undo/redo --
  // there is no separate GC pass on the restored value that could diverge
  // from what was actually saved.
  const restoreVideoMaskSnapshot = (snapshot: MaskHistorySnapshot) => {
    setVideoMaskManifest((previous) => ({
      ...previous,
      keyframes: snapshot.keyframes,
      compositeFeatherPx: snapshot.compositeFeatherPx,
    }));
    setVideoMaskAssets(snapshot.assets);
  };
  const videoMaskHistory = useSnapshotHistory<MaskHistorySnapshot>(restoreVideoMaskSnapshot, {
    // A different clip's keyframes are not undo-continuous with the
    // previous one's; `videoPreviewUrl` changes on every upload/replace/
    // clear (see processVideoFile/handleClearVideo/the video-input-updated
    // handler above), so it doubles as the reset signal here too.
    resetKey: videoPreviewUrl,
    limit: 100,
  });
  const currentVideoMaskSnapshot = (): MaskHistorySnapshot => ({
    keyframes: videoMaskManifest.keyframes,
    compositeFeatherPx: videoMaskManifest.compositeFeatherPx,
    assets: videoMaskAssets,
  });

  // An explicit edit supersedes any pending (aborted) restore -- the persist
  // effect must resume writing from here on regardless. If a restore WAS
  // still pending (its persisted PNGs never got read this session), it is
  // about to be overwritten by whatever state this edit leaves behind, so
  // that outcome is surfaced as its own notice rather than silently clearing
  // `videoMaskPersistenceNotice` to null the way an ordinary "restore
  // resolved" transition would -- otherwise the user sees the "could not be
  // restored yet" banner disappear with no indication the unread saved
  // keyframes are now gone for good.
  const dismissAbortedVideoMaskRestore = () => {
    if (videoMaskRestoreAbortedRef.current) {
      videoMaskRestoreAbortedRef.current = false;
      setVideoMaskPersistenceNotice(
        "Previously saved video mask keyframes for this clip could not be loaded and have now been " +
        "discarded by this edit.",
      );
    } else {
      setVideoMaskPersistenceNotice(null);
    }
  };

  // Explicit user action for a restore that never recovers: the backend
  // temp files themselves can go away independently of this panel (Settings'
  // "Clear temp images", or the backend's own 24h sweep) without this session
  // ever finding out, in which case retries never succeed and the "not
  // restored yet" notice would otherwise persist across every future reload
  // with no way for the user to get the timeline back to a clean, editable
  // state. Discarding here releases the (now inevitably orphaned) localStorage
  // record instead of leaving it to accumulate retries indefinitely.
  const handleDiscardStuckVideoMaskRestore = () => {
    const staleRefs = videoMaskUploadedRefsRef.current;
    videoMaskUploadedRefsRef.current = new Map();
    videoMaskRestoreAbortedRef.current = false;
    setVideoMaskPersistenceNotice(null);
    enqueueVideoMaskPersistOp(() =>
      releaseAllTrackedMaskAssets(staleRefs).catch((error) =>
        console.error("[Inpaint] Failed to discard the unrestorable video mask record:", error),
      ),
    );
  };

  const handleVideoMaskKeyframesChange = (keyframes: VideoMaskKeyframe[]) => {
    dismissAbortedVideoMaskRestore();
    const { keyframes: normalized, changed: normalizedAffine } = demoteMismatchedAffineLinks(keyframes);
    const referencedMaskIds = new Set(normalized.map((keyframe) => keyframe.maskId));
    const nextAssets = videoMaskAssets.filter((asset) => referencedMaskIds.has(asset.id));
    videoMaskHistory.push(currentVideoMaskSnapshot(), {
      keyframes: normalized,
      compositeFeatherPx: videoMaskManifest.compositeFeatherPx,
      assets: nextAssets,
    });
    setVideoMaskError(
      normalizedAffine
        ? "Affine interpolation needs the same mask asset on both keyframes; changed to Hold."
        : null,
    );
  };

  // Wired to VideoInpaintTimeline's composite-feather control. Already part
  // of the manifest's wire format (`composite_feather_px`); only the UI to
  // change it was missing.
  const handleVideoMaskFeatherChange = (value: number) => {
    dismissAbortedVideoMaskRestore();
    videoMaskHistory.push(currentVideoMaskSnapshot(), {
      keyframes: videoMaskManifest.keyframes,
      compositeFeatherPx: value,
      assets: videoMaskAssets,
    });
  };

  /**
   * Persists a drawn mask for `frame` -- the fork/new-asset/MAX_MASK_ASSETS
   * handling and the manifest-level undo push that used to live inline in
   * `handleVideoMaskEditorSaveMask` (single-frame editor, closed the modal
   * itself on success). Extracted (P4) so both VideoMaskFrameEditor's
   * per-frame auto-save (on navigating away from a dirty frame) and its
   * "Save & Use" button share this ONE persistence path instead of two
   * near-duplicate copies; closing the editor is now the CALLER's decision
   * (auto-save must NOT close it), so this only returns a result.
   *
   * Keyed by `frame` (not a keyframeId captured when the editor was opened):
   * VideoMaskFrameEditor can call this for any frame in the manifest at any
   * time during one open session, so the keyframe this session started on
   * is no longer necessarily the one being saved.
   */
  const persistVideoMaskFrame = async (
    frame: number,
    maskUrl: string,
  ): Promise<
    | { warnings: string[]; keyframes: VideoMaskKeyframe[]; assets: VideoMaskAsset[] }
    | { error: string }
  > => {
    dismissAbortedVideoMaskRestore();
    try {
      // The mask canvas ImageEditor hands back is already sized to
      // videoMaskCanvasWidth x videoMaskCanvasHeight (its base layer was
      // initialized from a frame image that centerCropToCanvas already
      // rendered at that exact size), so this call is normally an identity
      // copy -- it exists to apply the SAME mapping rule as the frame image
      // rather than to actually resize anything, so the two stay provably
      // consistent instead of merely coincidentally equal.
      const normalizedMaskUrl = await centerCropToCanvas(
        maskUrl,
        videoMaskCanvasWidth,
        videoMaskCanvasHeight,
      );
      // The backend rejects an all-black (nothing to regenerate) mask at
      // generation time; catching it here, at save, surfaces the mistake
      // immediately instead of after the rest of the queue has run.
      const isEmptyMask = !(await dataUrlHasWhitePixel(normalizedMaskUrl).catch(() => true));
      const existingKeyframe = videoMaskManifest.keyframes.find(
        (keyframe) => keyframe.frame === frame,
      );
      // Mirrors VideoInpaintTimeline's own MAX_MASK_KEYFRAMES gate on its
      // Add/Duplicate buttons -- those only guarded the button click, but
      // VideoMaskFrameEditor's in-place frame navigation is a second entry
      // point that creates a brand-new keyframe (drawing on a frame with no
      // keyframe yet) without going through either button.
      if (!existingKeyframe && videoMaskManifest.keyframes.length >= MAX_MASK_KEYFRAMES) {
        const message = `This clip already has the maximum of ${MAX_MASK_KEYFRAMES} mask keyframes. Delete one before adding another.`;
        setVideoMaskError(message);
        alert(message);
        return { error: message };
      }
      // Duplicate (in VideoInpaintTimeline) intentionally shares a
      // maskId across keyframes so affine interpolation has identical
      // source pixels on both ends. Repainting that asset in place would
      // silently change every keyframe still referencing it; fork onto a
      // fresh, keyframe-private maskId unless this keyframe is the only
      // one left referencing it.
      const priorMaskId = existingKeyframe?.maskId;
      const sharerCount = priorMaskId
        ? videoMaskManifest.keyframes.filter((keyframe) => keyframe.maskId === priorMaskId).length
        : 0;
      const isFork = !!priorMaskId && sharerCount > 1;
      const maskId = isFork ? newId() : (priorMaskId ?? newId());
      const isNewAsset = !videoMaskAssets.some((asset) => asset.id === maskId);
      if (isNewAsset && videoMaskAssets.length >= MAX_MASK_ASSETS) {
        // Medium-4 (final audit): this return happens while the mask editor
        // overlay is still open (unlike every other branch below, which
        // closes it before returning), so `setVideoMaskError` alone renders
        // its text behind that overlay -- the user sees the Save button do
        // nothing. `alert()` matches this panel's existing convention for
        // errors that must interrupt an open modal (see the submit-time
        // mask checks above).
        const message = `This clip already has the maximum of ${MAX_MASK_ASSETS} saved mask images. Delete a keyframe (or reuse Duplicate instead of drawing a new mask) before adding another.`;
        setVideoMaskError(message);
        alert(message);
        return { error: message };
      }
      const keyframe: VideoMaskKeyframe = {
        id: existingKeyframe?.id ?? newId(),
        frame,
        maskId,
        interpolationToNext: existingKeyframe?.interpolationToNext ?? "hold",
        transform: existingKeyframe?.transform
          ? { ...existingKeyframe.transform }
          : createDefaultMaskTransform(),
      };
      // Computed synchronously off the CURRENT `videoMaskAssets`/
      // `videoMaskManifest` (the same reads that decided `isFork`/`maskId`
      // above), not inside a `setVideoMaskAssets` updater: this needs to
      // become one atomic history entry alongside the keyframe change
      // below, pushed together via `videoMaskHistory.push`, rather than two
      // independent state writes that undo/redo could otherwise see torn
      // apart from each other.
      const nextAsset: VideoMaskAsset = {
        id: maskId,
        dataUrl: normalizedMaskUrl,
        // Recorded per-asset, not just on the manifest: `centerCropToCanvas`
        // above rendered THIS PNG at the current output size, but sibling
        // assets saved before a since-changed width/height slider still hold
        // pixels sized for whatever canvas was live when THEY were saved. The
        // submit-time check below (and the mismatch banner) need per-asset
        // truth, not "the size of whichever asset was saved most recently".
        width: videoMaskCanvasWidth,
        height: videoMaskCanvasHeight,
      };
      const assetReplaced = videoMaskAssets.some((asset) => asset.id === maskId);
      const nextAssets = assetReplaced
        ? videoMaskAssets.map((asset) => (asset.id === maskId ? nextAsset : asset))
        : [...videoMaskAssets, nextAsset];
      // Merged against the same `videoMaskManifest` read that decided the fork
      // above, so the warning below reflects what is actually stored. Deriving
      // it inside the updater would not work: the updater runs during a later
      // render, after the warnings have already been assembled.
      const merged = demoteMismatchedAffineLinks(
        upsertKeyframe(videoMaskManifest.keyframes, keyframe),
      );
      const normalizedAffine = merged.changed;
      // `canvas` is NOT part of the undo/redo snapshot (it tracks the
      // current output width/height, not a manual edit -- see the
      // `videoMaskManifest.canvas` comment elsewhere in this file), so it
      // is written directly. It touches a disjoint field from the
      // `keyframes`/`compositeFeatherPx` `restoreVideoMaskSnapshot` below
      // writes, so the two updates compose regardless of call order.
      setVideoMaskManifest((previous) => ({
        ...previous,
        canvas: { width: videoMaskCanvasWidth, height: videoMaskCanvasHeight },
      }));
      videoMaskHistory.push(currentVideoMaskSnapshot(), {
        keyframes: merged.keyframes,
        compositeFeatherPx: videoMaskManifest.compositeFeatherPx,
        assets: nextAssets,
      });
      const warnings = [
        isEmptyMask
          ? "This mask has no white (generate) area. It was saved, but generation will refuse an empty mask."
          : null,
        isFork
          ? "This mask was shared with another duplicated keyframe; the edit was saved as a separate copy so the other keyframe is unaffected."
          : null,
        normalizedAffine
          ? "Affine interpolation needs the same mask asset on both keyframes; changed to Hold."
          : null,
      ].filter((message): message is string => message !== null);
      // Closing the editor (or not) on success is the CALLER's decision now
      // (see the doc comment above) -- auto-save-on-navigate must leave it
      // open, while the "Save & Use" button closes it. Both callers still
      // surface these warnings via `setVideoMaskError`, same as before.
      setVideoMaskError(warnings.length > 0 ? warnings.join(" ") : null);
      // Returned alongside `warnings` so VideoMaskFrameEditor's navigate()
      // can resolve the mask for the frame it is moving TO using the
      // just-confirmed keyframes/assets instead of the `keyframes`/`assets`
      // props it closed over when navigate() was invoked -- those props
      // only reflect this update after InpaintPanel re-renders, which has
      // not happened yet inside this same async call.
      return { warnings, keyframes: merged.keyframes, assets: nextAssets };
    } catch (error) {
      console.error("[Inpaint] Failed to save video mask:", error);
      const message = "Could not save the video mask. Please try again.";
      setVideoMaskError(message);
      return { error: message };
    }
  };

  /** VideoMaskFrameEditor's `onSaveFrame` prop -- thin pass-through, kept as its own function so the prop identity/name at the call site reads as "the video-frame-editor integration point" rather than the lower-level persistence helper. */
  const handleVideoMaskFrameSave = (frame: number, maskDataUrl: string) =>
    persistVideoMaskFrame(frame, maskDataUrl);

  const handleScaleChange = (newScale: number) => {
    setScale(newScale);
    if (inputImageSize && sizeMode === "scale") {
      const scaledWidth = Math.round(inputImageSize.width * newScale / 64) * 64;
      const scaledHeight = Math.round(inputImageSize.height * newScale / 64) * 64;
      setParams({ ...params, width: scaledWidth, height: scaledHeight });
    }
  };

  const handleSizeModeChange = (newMode: "absolute" | "scale") => {
    setSizeMode(newMode);
    if (newMode === "scale" && inputImageSize) {
      // Switch to scale mode - update dimensions based on current scale
      const scaledWidth = Math.round(inputImageSize.width * scale / 64) * 64;
      const scaledHeight = Math.round(inputImageSize.height * scale / 64) * 64;
      setParams({ ...params, width: scaledWidth, height: scaledHeight });
    }
  };

  // ── Video temporal inpaint: clip input, canvas and range handling ────────

  const fitVideoCanvasFor = (srcWidth: number, srcHeight: number, scaleValue: number) =>
    fitVideoCanvas(archCapabilities, loadedArchType, srcWidth, srcHeight, scaleValue);

  // Trim (start + end) that brings a clip of `raw` frames to `target`, and the
  // range that fits inside the result. Applied on load and by the "Fit" button,
  // and it mirrors the route's rule so the panel and the route cannot disagree.
  const applyClipLength = (raw: number, target: number | null) => {
    const trimEnd = target != null ? Math.max(0, raw - target) : 0;
    const trimmed = target ?? raw;
    // A default range: the middle third, snapped to latent-group boundaries.
    const spans = latentGroupSpans(latentChunkPattern, trimmed);
    const snapped = snapRangeToLatentGroups(
      spans, Math.floor(trimmed / 3), Math.ceil((2 * trimmed) / 3));
    setParams(prev => ({
      ...prev,
      input_trim_start_frames: 0,
      input_trim_end_frames: trimEnd,
      regenerate_start_frame: snapped.start,
      regenerate_end_frame: snapped.end,
    }));
  };

  const resetVideoMaskTimeline = () => {
    setVideoMaskManifest(createDefaultVideoMaskManifest(params.width, params.height));
    setVideoMaskAssets([]);
    setVideoMaskEditorSession(null);
    setVideoMaskError(null);
    setVideoMaskPersistenceNotice(null);
    // Both call sites (processVideoFile, handleClearVideo) reset the
    // timeline because the clip it was drawn for is going away (a new
    // upload, or none at all) -- the persisted record can never apply
    // afterwards, so release its backend temp PNGs here too instead of
    // leaking them. Enqueued (not fired directly) so it cannot race a
    // still-in-flight persist call for the clip that is being replaced; the
    // ref is reassigned rather than cleared in place for the same reason
    // (see the mount handler's "no clip" branch).
    const staleRefs = videoMaskUploadedRefsRef.current;
    videoMaskUploadedRefsRef.current = new Map();
    videoMaskRestoreAbortedRef.current = false;
    enqueueVideoMaskPersistOp(() =>
      releaseAllTrackedMaskAssets(staleRefs).catch((error) =>
        console.error("[Inpaint] Failed to clear the video mask record:", error),
      ),
    );
  };

  const processVideoFile = (file: File) => {
    if (!file.type.startsWith('video/')) {
      alert('Please upload a valid video file');
      return;
    }
    preserveVideoSettingsRef.current = false;
    resetVideoMaskTimeline();
    setVideoFile(file);
    setVideoDurationSec(null);
    setInputVideoSize(null);
    setClipFramesOverride(null);
    setVideoPreviewUrl(prev => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
    void saveMediaInput(INPAINT_VIDEO_INPUT_KEY, file).catch((error) => {
      console.error("[Inpaint] Failed to persist input video:", error);
    });
  };

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processVideoFile(file);
  };

  const handleVideoLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const preserveSettings = preserveVideoSettingsRef.current;
    const duration = e.currentTarget.duration;
    // `loadedmetadata` fires again every time the <video> remounts
    // (collapsing/expanding the Input Video card, the temporal-inpaint
    // subtree reappearing after a model reload), and `preserveSettings` is a
    // one-shot ref the FIRST of those consumes -- so without a source test
    // every later remount re-ran applyClipLength and silently replaced the
    // regenerate range with the middle third. Keyed on the resolved source
    // rather than on "have we measured a duration yet" so a metadata event
    // still in flight for the PREVIOUS clip cannot be mistaken for this one.
    const clipSource = e.currentTarget.currentSrc || videoPreviewUrl || "";
    const isNewClipSource = lastClipLengthAppliedSrcRef.current !== clipSource;
    if (Number.isFinite(duration) && duration > 0) {
      lastClipLengthAppliedSrcRef.current = clipSource;
      setVideoDurationSec(duration);
      const raw = Math.max(1, Math.round(duration * clipFrameRate));
      if (isNewClipSource && !preserveSettings) {
        applyClipLength(raw, largestValidVideoFrameCount(archCapabilities, loadedArchType, raw));
      }
    }
    const { videoWidth, videoHeight } = e.currentTarget;
    if (videoWidth > 0 && videoHeight > 0) {
      // Only a NEW clip re-defaults the canvas: `loadedmetadata` fires again
      // whenever the <video> remounts (tab switch, collapse/expand), and a
      // canvas the user chose must survive that. processVideoFile clears
      // inputVideoSize, so an actual upload always counts as new.
      const isNewClip =
        !inputVideoSize
        || inputVideoSize.width !== videoWidth
        || inputVideoSize.height !== videoHeight;
      setInputVideoSize({ width: videoWidth, height: videoHeight });
      if (isNewClip && !preserveSettings) {
        setVideoScale(1.0);
        const fitted = fitVideoCanvasFor(videoWidth, videoHeight, 1.0);
        setParams(prev => ({ ...prev, width: fitted.width, height: fitted.height }));
      }
    }
    preserveVideoSettingsRef.current = false;
  };

  const handleClearVideo = () => {
    // The Exit-fullscreen button lives in the `videoPreviewUrl ? ... : ...`
    // branch below, but the fullscreened container itself does not -- so
    // clearing the clip while fullscreen would otherwise swap that branch to
    // the empty-state placeholder while still occupying the whole screen,
    // exitable only by Esc. Exit fullscreen here instead of also rendering
    // the button outside the ternary.
    const doc = document as Document & { webkitFullscreenElement?: Element | null; webkitExitFullscreen?: () => Promise<void> | void };
    if ((doc.fullscreenElement ?? doc.webkitFullscreenElement ?? null) === videoContainerRef.current) {
      if (document.exitFullscreen) void document.exitFullscreen().catch(() => {});
      else if (doc.webkitExitFullscreen) void doc.webkitExitFullscreen();
    }
    if (videoPreviewUrl) {
      releaseVideoFrameGrabber(videoPreviewUrl);
      URL.revokeObjectURL(videoPreviewUrl);
    }
    setVideoFile(null);
    setVideoPreviewUrl(null);
    setVideoDurationSec(null);
    setInputVideoSize(null);
    setClipFramesOverride(null);
    resetVideoMaskTimeline();
    setVideoSizeMode("absolute");
    setParams(prev => ({
      ...prev,
      input_trim_start_frames: 0,
      input_trim_end_frames: 0,
      regenerate_start_frame: 0,
      regenerate_end_frame: 0,
    }));
    void deleteMediaInput(INPAINT_VIDEO_INPUT_KEY).catch((error) => {
      console.error("[Inpaint] Failed to clear persisted input video:", error);
    });
  };

  const handleVideoScaleChange = (newScale: number) => {
    setVideoScale(newScale);
    if (inputVideoSize) {
      const fitted = fitVideoCanvasFor(inputVideoSize.width, inputVideoSize.height, newScale);
      setParams(prev => ({ ...prev, width: fitted.width, height: fitted.height }));
    }
  };

  const handleVideoSizeModeChange = (newMode: "absolute" | "scale") => {
    setVideoSizeMode(newMode);
    if (newMode === "scale" && inputVideoSize) {
      const fitted = fitVideoCanvasFor(inputVideoSize.width, inputVideoSize.height, videoScale);
      setParams(prev => ({ ...prev, width: fitted.width, height: fitted.height }));
    }
  };

  const handleClearInputImage = async () => {
    setInputImage(null);
    setInputImagePreview(null);
    setInputImageSize(null);
    setMaskImage(null);
    if (isMounted) {
      // Delete temp image references
      const inputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
      if (inputRef) {
        await deleteTempImageRef(inputRef).catch(console.error);
      }
      const maskRef = localStorage.getItem(MASK_IMAGE_STORAGE_KEY);
      if (maskRef) {
        await deleteTempImageRef(maskRef).catch(console.error);
      }
      localStorage.removeItem(INPUT_IMAGE_STORAGE_KEY);
      localStorage.removeItem(MASK_IMAGE_STORAGE_KEY);
    }
  };

  const handleClearMask = async () => {
    setMaskImage(null);
    if (isMounted) {
      // Delete temp mask reference
      const maskRef = localStorage.getItem(MASK_IMAGE_STORAGE_KEY);
      if (maskRef) {
        await deleteTempImageRef(maskRef).catch(console.error);
      }
      localStorage.removeItem(MASK_IMAGE_STORAGE_KEY);
    }
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

    // Send image if checked
    if (sendImage) {
      try {
        await sendImageToImg2Img(generatedImage);
      } catch (error) {
        console.error("Failed to send image to img2img:", error);
      }
    }

    console.log("[Inpaint] sendToImg2Img - sendPrompt:", sendPrompt, "sendParameters:", sendParameters);
    console.log("[Inpaint] sendToImg2Img - sourceParams.prompt:", sourceParams.prompt);

    // Send prompt and/or parameters
    sendToPanel(sourceParams, "img2img_params", {
      sendPrompt,
      sendParameters,
      includeDenoising: true,
      dispatchEvent: "img2img_params_updated"
    });

    console.log("[Inpaint] sendToImg2Img - Sent to panel");

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
      await sendImageToUpscale(generatedImage);
    } catch (error) {
      console.error("[Inpaint] Failed to send image to upscale:", error);
    }

    if (onTabChange) {
      onTabChange("upscale");
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
        console.error("[Inpaint] Failed to send image to outpaint:", error);
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

  // generatedVideo (inpaint_vid) result -> Outpaint's outpaint_vid clip input.
  const sendVideoResultToOutpaint = async () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    try {
      await sendVideoToOutpaint(generatedVideo);
    } catch (error) {
      console.error("[Inpaint] Failed to send video to outpaint:", error);
      alert("Failed to send the video to outpaint");
      return;
    }
    if (onTabChange) onTabChange("outpaint");
  };

  // Inpaint's own inpaint_vid result -> Inpaint again (self-send = iterate a
  // temporal inpaint further, e.g. regenerate a different range next).
  const sendVideoResultToInpaint = async () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    try {
      await sendVideoToInpaint(generatedVideo);
    } catch (error) {
      console.error("[Inpaint] Failed to reuse video as inpaint input:", error);
      alert("Failed to send the video to inpaint");
      return;
    }
    if (onTabChange) onTabChange("inpaint");
  };

  // generatedVideo (inpaint_vid) result -> the ref2va reference track
  // (whole-clip conditioning, not a placement anchor -- see sendVideoToReference).
  const sendVideoResultToReference = () => {
    if (!generatedVideo) {
      alert("No video to send");
      return;
    }
    sendVideoToReference(generatedVideo);
    if (onTabChange) onTabChange("txt2img");
  };

  const sendToInpaint = async () => {
    if (!generatedImage) {
      alert("No image to send");
      return;
    }

    // Send image if checked (use generated image as new input, clear mask)
    if (sendImage) {
      try {
        // Fetch the generated image and convert to base64
        const response = await fetch(generatedImage);
        const blob = await response.blob();
        const reader = new FileReader();
        reader.onloadend = async () => {
          const base64data = reader.result as string;
          // Delete old input and mask references
          const oldInputRef = localStorage.getItem(INPUT_IMAGE_STORAGE_KEY);
          if (oldInputRef) {
            await deleteTempImageRef(oldInputRef).catch(console.error);
          }
          const oldMaskRef = localStorage.getItem(MASK_IMAGE_STORAGE_KEY);
          if (oldMaskRef) {
            await deleteTempImageRef(oldMaskRef).catch(console.error);
          }
          const ref = await saveTempImage(base64data);
          localStorage.setItem("inpaint_input_image", ref);
          localStorage.removeItem(MASK_IMAGE_STORAGE_KEY);
          window.dispatchEvent(new Event("inpaint_input_updated"));
        };
        reader.readAsDataURL(blob);
      } catch (error) {
        console.error("Failed to send image to inpaint:", error);
      }
    }

    // Send prompt if checked
    if (sendPrompt) {
      const inpaintParams = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      inpaintParams.prompt = params.prompt;
      inpaintParams.negative_prompt = params.negative_prompt;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(inpaintParams));
    }

    // Send parameters if checked
    if (sendParameters) {
      const inpaintParams = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      inpaintParams.steps = params.steps;
      inpaintParams.cfg_scale = params.cfg_scale;
      inpaintParams.sampler = params.sampler;
      inpaintParams.schedule_type = params.schedule_type;
      inpaintParams.seed = params.seed;
      inpaintParams.width = params.width;
      inpaintParams.height = params.height;
      inpaintParams.denoising_strength = params.denoising_strength;
      inpaintParams.mask_blur = params.mask_blur;
      inpaintParams.inpaint_full_res = params.inpaint_full_res;
      inpaintParams.inpaint_full_res_padding = params.inpaint_full_res_padding;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(inpaintParams));
    }

    // Reload current panel to reflect changes if image was sent
    if (sendImage) {
      // The preview will be updated by the event listener after loading from temp storage
      setMaskImage(null);
    }
  };

  // FLUX.2 Image Edit: Reference image handlers
  const handleRefImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const newFiles = Array.from(files).slice(0, maxRefImages - refImagePreviews.length); // Max total
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
            console.error("[Inpaint] Failed to save reference image to temp storage:", error);
          }

          if (newPreviews.length === newFiles.length) {
            // Use functional setState to get the latest state
            setRefImagePreviews((prevPreviews) => [...prevPreviews, ...newPreviews]);

            // Update localStorage with refs
            const savedRefImageRefs = localStorage.getItem(REF_IMAGES_STORAGE_KEY);
            const existingRefs = savedRefImageRefs ? JSON.parse(savedRefImageRefs) : [];
            const allRefs = [...existingRefs, ...newRefs];
            localStorage.setItem(REF_IMAGES_STORAGE_KEY, JSON.stringify(allRefs));
            console.log(`[Inpaint] Saved ${newRefs.length} reference images to storage`);
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
        console.log(`[Inpaint] Removed reference image ${index} from storage`);
      } catch (error) {
        console.error("[Inpaint] Failed to update reference images storage:", error);
      }
    }
  };

  const handleClearAllRefImages = () => {
    setRefImages([]);
    setRefImagePreviews([]);

    // Clear localStorage
    localStorage.removeItem(REF_IMAGES_STORAGE_KEY);
    console.log("[Inpaint] Cleared all reference images from storage");
  };

  const handleRefImageDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsRefImageDragging(false);

    const files = e.dataTransfer.files;
    if (!files || files.length === 0) return;

    const imageFiles = Array.from(files)
      .filter(file => file.type.startsWith('image/'))
      .slice(0, maxRefImages - refImagePreviews.length); // Max total

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
            console.error("[Inpaint] Failed to save reference image to temp storage:", error);
          }

          if (newPreviews.length === imageFiles.length) {
            // Use functional setState to get the latest state
            setRefImagePreviews((prevPreviews) => [...prevPreviews, ...newPreviews]);

            // Update localStorage with refs
            const savedRefImageRefs = localStorage.getItem(REF_IMAGES_STORAGE_KEY);
            const existingRefs = savedRefImageRefs ? JSON.parse(savedRefImageRefs) : [];
            const allRefs = [...existingRefs, ...newRefs];
            localStorage.setItem(REF_IMAGES_STORAGE_KEY, JSON.stringify(allRefs));
            console.log(`[Inpaint] Saved ${newRefs.length} reference images to storage (D&D)`);
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
    if (!currentItem || !["inpaint", "inpaint_vid"].includes(currentItem.type)) {
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
    const result = completedResults.inpaint;
    if (!result || (currentItem && ["inpaint", "inpaint_vid"].includes(currentItem.type))) return;
    setPreviewImage(null);
    if (result.kind === "video") {
      setGeneratedVideo(result.url);
      setGeneratedVideoPlaybackUrl(result.playbackUrl || null);
      setGeneratedVideoInfo(result.info as typeof generatedVideoInfo);
      setGeneratedVideoSeed(result.seed ?? null);
      setGeneratedVideoParams(result.params as InpaintParams);
      setGeneratedImage(null);
    } else if (result.kind === "image") {
      setGeneratedImage(result.url);
      setGeneratedImageSeed(result.seed ?? null);
      setGeneratedImageAncestralSeed(result.ancestralSeed ?? null);
      setGeneratedImageParams(result.params as InpaintParams);
      setGeneratedVideo(null);
    }
  }, [completedResults.inpaint, currentItem]);
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

  // Add generation request to queue. Two modality branches: image (inpaint,
  // mask-driven) and video (inpaint_vid, range-driven), mutually exclusive on
  // the loaded model's modality.
  const handleAddToQueue = async () => {
    if (!params.prompt) {
      alert("Please enter a prompt");
      return;
    }

    // Which endpoint this goes to is decided from a FRESH read of
    // GET /models/current rather than the cached `isVideo` render flag: the
    // model can change under an open page (API call, backend restart, second
    // tab), and routing an image request at a video model costs a 400 about the
    // wrong thing. The cached flag stays the render-time hint.
    const modality = await resolveModality();
    if (modality.isVideo) {
      if (modality.modelInfo?.type === "minimax_h3" && modality.modelInfo?.variant === "hybrid") {
        alert("A merged MiniMax-H3 checkpoint is released for text-to-video only, which is the Txt2Img tab with this model loaded. Temporal inpaint is refused: it was not part of the comparison that released the merge.");
        return;
      }
      // fl2va and ref2va both serve this endpoint now; only hybrid is refused
      // outright, and that was already caught by the fresh read above. A
      // reference attached to a non-ref2va variant is refused by the route
      // (fl2va: "never trained to read reference rows"; unidentified: cannot
      // be told apart from fl2va's absence of it) -- checked just below, from
      // the SAME fresh variant read, since the References card is built off
      // the cached `currentModelInfo` and a backbone switch under an open
      // page could otherwise let a stale reference set reach the route.
      const freshH3Variant = modality.modelInfo?.type === "minimax_h3"
        ? (modality.modelInfo?.variant as string | undefined) : undefined;
      if (freshH3Variant !== "ref2va" && countMiniMaxH3References(h3References) > 0) {
        alert(`The loaded MiniMax-H3 transformer is the ${freshH3Variant || "unidentified"} `
          + "variant. Reference conditioning on temporal inpaint requires the ref2va "
          + "checkpoint (e.g. diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors); "
          + "clear the references, or load that checkpoint.");
        return;
      }
      if (!supportsTemporalInpaint) {
        alert(temporalInpaintReason
          || `${loadedArchName} has no temporal inpaint; load a MiniMax-H3 fl2va model.`);
        return;
      }
      if (!videoFile) {
        alert("Please upload an input video clip");
        return;
      }
      if (!videoTrimmedLengthValid) {
        alert("The trimmed clip is not a length this model can generate — use the trim controls "
          + "(or 'Fit to a valid length') first.");
        return;
      }
      if (!((params.regenerate_start_frame ?? 0) < (params.regenerate_end_frame ?? 0))) {
        alert("Please choose a range to regenerate");
        return;
      }
      const { replaceWildcardsInPrompt } = await import("@/utils/wildcardStorage");
      let videoPrompt = await replaceWildcardsInPrompt(params.prompt);
      if (modality.modelInfo?.type === "minimax_h3") {
        try {
          // The inventory has to describe the ACTUAL request's reference
          // rows -- the input clip being inpainted is a pin, not a
          // reference, and gets no <Video k> label of its own, so it must
          // not be counted here (M5 fix: this used to hardcode `videos: 1`
          // for the pinned clip, which could point the assisted prompt at
          // a <Video 1> that is absent or is someone else's reference).
          const assisted = await maybeTransformH3PromptForGeneration({
            prompt: videoPrompt,
            mode: modality.modelInfo?.variant === "ref2va" && countMiniMaxH3References(h3References) > 0
              ? "ref2va"
              : "t2va",
            durationSeconds: videoDurationSec ?? Math.max(1, estimatedRawFrames) / clipFrameRate,
            references: createH3ReferenceInventory({
              pictures: h3References.images.length,
              videos: h3References.videos.length,
              audios: h3References.audios.length + h3References.videoAudios.filter(Boolean).length,
            }),
          });
          videoPrompt = assisted.prompt;
        } catch (error: any) {
          alert(error?.message || "MiniMax H3 Prompt Assist failed");
          return;
        }
      }
      let spatialMaskManifest: string | undefined;
      let spatialMaskFiles: Array<{ id: string; file: File }> | undefined;
      if (videoMaskManifest.keyframes.length > 0) {
        // A spatial mask pins individual latent rows, so the free rows no
        // longer divide into whole latent frames and First Block Cache's
        // per-frame reuse indicator cannot be computed. The backend rejects
        // the pair; stopping here keeps that from surfacing only after the
        // clip has been encoded.
        if (params.fbcache_enable) {
          const message = "First Block Cache cannot be used with a spatial mask timeline. Turn off First Block Cache or delete the mask keyframes.";
          setVideoMaskError(message);
          alert(message);
          return;
        }
        const maskRangeStart = params.regenerate_start_frame ?? 0;
        const maskRangeEnd = params.regenerate_end_frame ?? 0;
        const outOfRangeKeyframe = videoMaskManifest.keyframes.find(
          (keyframe) => keyframe.frame < maskRangeStart || keyframe.frame >= maskRangeEnd,
        );
        if (outOfRangeKeyframe) {
          const message = `Mask keyframe frame ${outOfRangeKeyframe.frame} must be inside the regenerate range [${maskRangeStart}, ${maskRangeEnd}).`;
          setVideoMaskError(message);
          alert(message);
          return;
        }
        const validation = validateVideoMaskManifest({
          ...videoMaskManifest,
          assets: videoMaskAssets,
        });
        if (!validation.valid) {
          const message = `Video mask timeline is invalid: ${validation.errors.join(" ")}`;
          setVideoMaskError(message);
          alert(message);
          return;
        }
        if (videoMaskCanvasMismatch) {
          const message = "One or more video mask assets do not match the current output canvas. Recreate the affected masks.";
          setVideoMaskError(message);
          alert(message);
          return;
        }
        try {
          spatialMaskManifest = serializeVideoMaskManifestForApi(videoMaskManifest, videoMaskAssets);
          const referencedIds = [...new Set(videoMaskManifest.keyframes.map((keyframe) => keyframe.maskId))];
          const assetsById = new Map(videoMaskAssets.map((asset) => [asset.id, asset]));
          const referencedAssets = referencedIds.map((id) => assetsById.get(id));
          if (referencedAssets.some((asset) => !asset) || referencedAssets.length !== referencedIds.length) {
            throw new Error("Every video mask keyframe must have a saved PNG asset.");
          }
          spatialMaskFiles = await Promise.all(
            referencedAssets.map(async (asset) => {
              if (!asset) throw new Error("A referenced video mask asset is missing.");
              const response = await fetch(asset.dataUrl);
              if (!response.ok) throw new Error(`Could not read mask asset ${asset.id}.`);
              const blob = await response.blob();
              return {
                // This File's own name is never sent to the backend --
                // generateInpaintVideo (api.ts) passes an explicit filename
                // to formData.append that overrides it. Kept simple/human-
                // readable for local debugging only.
                id: asset.id,
                file: new File([blob], `${asset.id}.png`, { type: "image/png" }),
              };
            }),
          );
          if (spatialMaskFiles.length !== referencedIds.length) {
            throw new Error("Video mask asset pairing is incomplete.");
          }
        } catch (error: any) {
          const message = error?.message || "Could not prepare the video mask files.";
          setVideoMaskError(message);
          alert(message);
          return;
        }
      }
      const videoParams: InpaintVideoParams = {
        prompt: videoPrompt,
        negative_prompt: await replaceWildcardsInPrompt(params.negative_prompt || ""),
        width: params.width,
        height: params.height,
        frame_rate: params.frame_rate,
        num_inference_steps: params.num_inference_steps,
        guidance_scale: params.guidance_scale,
        seed: params.seed,
        num_videos_per_prompt: params.num_videos_per_prompt,
        max_sequence_length: params.max_sequence_length,
        audio_enable: params.audio_enable,
        regenerate_start_frame: params.regenerate_start_frame ?? 0,
        regenerate_end_frame: params.regenerate_end_frame ?? 0,
        input_trim_start_frames: params.input_trim_start_frames,
        input_trim_end_frames: params.input_trim_end_frames,
        inpaint_video_audio_mode: params.inpaint_video_audio_mode,
        spatial_mask_manifest: spatialMaskManifest,
        video_lossless: params.video_lossless,
        blocks_to_swap: params.video_blocks_to_swap,
        fuse_output_proj: params.fuse_output_proj,
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
        unet_quantization: params.unet_quantization,
        quantized_gemm_mode: params.quantized_gemm_mode,
        // Applied by MiniMax-H3 (the only architecture this endpoint serves).
        // Same selector/list as image generation's `params.loras`.
        loras: params.loras,
        // ref2va only: harmless to send on fl2va too (the References card
        // never populates h3References there, so this stays "max" unread).
        reference_image_size: h3ReferenceImageSize,
      };
      const hasH3References = isH3Ref2VaInpaint && countMiniMaxH3References(h3References) > 0;
      if (hasH3References && spatialMaskManifest) {
        alert("A spatial mask timeline cannot be combined with reference conditioning: the "
          + "ref2va layout builder carries a frame-level temporal-inpaint pin alongside "
          + "references, not a row-level spatial-mask pin. Drop the spatial mask, or clear "
          + "the references.");
        return;
      }
      addToQueue({
        type: "inpaint_vid",
        params: videoParams as any,
        inputVideo: videoFile,
        ...(spatialMaskFiles ? { spatialMaskFiles } : {}),
        ...(hasH3References ? { references: h3References } : {}),
        prompt: videoParams.prompt,
      });
      return;
    }

    if (!inputImagePreview) {
      alert("Please upload an input image");
      return;
    }

    if (!maskImage) {
      alert("Please draw a mask by double-clicking the input image");
      return;
    }

    // Import wildcard replacement function dynamically
    const { replaceWildcardsInPrompt } = await import("@/utils/wildcardStorage");

    // Replace wildcards in prompts
    let processedPrompt = await replaceWildcardsInPrompt(params.prompt);
    const processedNegativePrompt = supportsNegativePrompt
      ? await replaceWildcardsInPrompt(params.negative_prompt)
      : "";

    // Feeling Lucky mode: Generate prompt with TIPO before queueing
    if (params.feeling_lucky) {
      try {
        // Use panel's TIPO settings (not localStorage)
        // Build category order and enabled map from settings
        const categoryOrder = tipoSettings.categories.map((c: any) => c.id);
        const enabledCategories: Record<string, boolean> = {};
        tipoSettings.categories.forEach((c: any) => {
          enabledCategories[c.id] = c.enabled;
        });

        console.log('[Inpaint] Feeling Lucky: Generating prompt with TIPO...');
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
        console.log('[Inpaint] Feeling Lucky: Generated prompt:', processedPrompt.substring(0, 100) + '...');
      } catch (error) {
        console.error("TIPO generation failed in Feeling Lucky mode:", error);
        alert("Failed to generate prompt with TIPO. Using original prompt.");
      }
    }

    // Create loop group ID if loop generation is enabled
    const loopGroupId = loopGenerationConfig.enabled ? `loop_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` : undefined;
    const hasEnabledLoopSteps = loopGenerationConfig.enabled && loopGenerationConfig.steps.some(s => s.enabled);
    // Main step decode directive. Inpaint never supports latent passthrough
    // (backend rejects loop_decode="none"/input_latent_id for inpaint), so an
    // intermediate main step falls back to "cheap"+skip_gallery, never "none".
    const mainDecodeDirective = computeLoopDecodeDirective({
      decodeMode: loopGenerationConfig.decodeMode ?? "every",
      isFinalStep: !hasEnabledLoopSteps,
      resizeMode: "image",
      supportsLatentPassthrough: false,
    });

    addToQueue({
      type: "inpaint",
      params: {
        ...params,
        prompt: processedPrompt,
        negative_prompt: processedNegativePrompt,
        loop_decode: mainDecodeDirective.loop_decode,
        skip_gallery: mainDecodeDirective.skip_gallery,
      },
      inputImage: inputImagePreview,
      maskImage: maskImage,
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
      } as InpaintParams, loopGroupId, maskImage);
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
  const addLoopStepsToQueueImmediate = useCallback(async (mainParams: InpaintParams, loopGroupId: string, maskImageData: string) => {
    if (!loopGenerationConfig.enabled || loopGenerationConfig.steps.length === 0) {
      return;
    }

    console.log('[Inpaint] Adding loop steps with mainParams.unet_quantization:', mainParams.unet_quantization);

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
        mask_blur: mainParams.mask_blur,
        inpaint_full_res: mainParams.inpaint_full_res,
        inpaint_full_res_padding: mainParams.inpaint_full_res_padding,
        inpaint_fill_mode: mainParams.inpaint_fill_mode,
        inpaint_fill_strength: mainParams.inpaint_fill_strength,
        inpaint_blur_strength: mainParams.inpaint_blur_strength,
        // Regional additional prompt: inherit from main (region strings fixed in v1, no per-step override)
        region_prompt: mainParams.region_prompt,
        region_negative_prompt: mainParams.region_negative_prompt,
        region_prompt_strength: mainParams.region_prompt_strength,
        region_prompt_method: mainParams.region_prompt_method,
        region_mask_feather: mainParams.region_mask_feather,
        seam_structure_strength: mainParams.seam_structure_strength,
        seam_structure_depth: mainParams.seam_structure_depth,
        seam_structure_end: mainParams.seam_structure_end,
        seam_structure_saliency: mainParams.seam_structure_saliency,
        seam_structure_max_area: mainParams.seam_structure_max_area,
        boundary_relax_strength: mainParams.boundary_relax_strength,
        boundary_relax_width: mainParams.boundary_relax_width,
        boundary_relax_noise: mainParams.boundary_relax_noise,
        boundary_relax_full_until: mainParams.boundary_relax_full_until,
        boundary_relax_end: mainParams.boundary_relax_end,
        boundary_relax_paste: mainParams.boundary_relax_paste,
        unet_quantization: mainParams.unet_quantization, // Inherit quantization from main
        quantized_gemm_mode: mainParams.quantized_gemm_mode, // Inherit quantized GEMM path from main
        timestep_shift: mainParams.timestep_shift, // Inherit SenseNova U1.5 time-shift (no per-step override)
        img_cfg_scale: mainParams.img_cfg_scale, // Inherit SenseNova U1.5 second CFG scale
        sensenova_mot_phase_eviction: mainParams.sensenova_mot_phase_eviction, // Inherit SenseNova U1.5 per-phase weight-half CPU eviction
        sensenova_kv_cache_streaming: mainParams.sensenova_kv_cache_streaming, // Inherit SenseNova U1.5 per-layer KV cache CPU streaming
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
        preview_predicted_x0: mainParams.preview_predicted_x0, // Inherit preview mode
        preview_decoder: mainParams.preview_decoder, // Inherit preview decoder
        attention_type: mainParams.attention_type, // Inherit attention backend from main
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

      // Decode directive. Inpaint never supports latent passthrough (backend
      // rejects loop_decode="none"/input_latent_id for inpaint) — intermediate
      // steps fall back to "cheap"+skip_gallery regardless of resize_mode.
      const isFinalStep = i === enabledSteps.length - 1;
      const decodeDirective = computeLoopDecodeDirective({
        decodeMode: loopGenerationConfig.decodeMode ?? "every",
        isFinalStep,
        resizeMode: stepParams.resize_mode as "image" | "latent",
        supportsLatentPassthrough: false,
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
        type: "inpaint",
        params: {
          ...stepParams,
          prompt: processedPrompt,
          negative_prompt: processedNegativePrompt,
        },
        inputImage: "", // Will be set when previous step completes
        maskImage: step.keepMask ? maskImageData : "", // Use same mask if keepMask is enabled
        prompt: `[Loop ${i + 1}/${enabledSteps.length}] ${processedPrompt.substring(0, 50)}...`,
        loopGroupId,
        loopStepIndex: i,
        isLoopStep: true,
        useTrainingModel,
        trainingRunId: activeTraining?.run_id,
      });
    }

    console.log(`[Inpaint] Added ${enabledSteps.length} loop steps to queue with group ID: ${loopGroupId}`);
  }, [loopGenerationConfig, addToQueue, refImages, useTrainingModel, activeTraining]);

  // Process queue - automatically start next item
  const processQueueRef = useRef<() => Promise<void>>();

  const processQueue = useCallback(async () => {
    console.log("[Inpaint] processQueue called, isGenerating:", isGeneratingRef.current);
    if (isGeneratingRef.current) {
      console.log("[Inpaint] Already generating, skipping");
      return;
    }

    const nextItem = startNextInQueue(["inpaint", "inpaint_vid"]);
    console.log("[Inpaint] Next item from queue:", nextItem);
    if (!nextItem || (nextItem.type !== "inpaint" && nextItem.type !== "inpaint_vid")) {
      console.log("[Inpaint] No inpaint items in queue");
      return;
    }

    // Video branch: inpaint_vid item. The queued clip is a File (inputVideo on
    // QueueItem), the result is a clip, and there is no loop-generation
    // handling -- matching the video branches of the other panels.
    if (nextItem.type === "inpaint_vid") {
      const videoParams = nextItem.params as InpaintVideoParams;
      isGeneratingRef.current = true;
      setIsGenerating(true);
      setProgress(0);
      setProgressMessage("");
      setTotalSteps(videoParams.num_inference_steps || 8);
      setPreviewImage(null);
      setGeneratedImage(null);
      setGeneratedVideo(null);
      setGeneratedVideoPlaybackUrl(null);
      setGeneratedVideoInfo(null);
      setGeneratedVideoSeed(null);
      setGeneratedVideoWarnings([]);
      setCfgMetrics([]);
      try {
        const clip = nextItem.inputVideo;
        if (!clip) throw new Error("No input video available for video inpaint generation");
        const result = await generateInpaintVideo(
          videoParams, clip, nextItem.spatialMaskFiles, nextItem.references);
        const videoUrl = `/outputs/${getResultFilename(result)}`;
        const videoPlaybackUrl = `/outputs/${getResultPlaybackFilename(result)}`;
        const playbackUrl = videoPlaybackUrl !== videoUrl ? videoPlaybackUrl : undefined;
        const videoSeed = getResultSeed(result);
        const videoInfo = {
          num_frames: result.image?.num_frames,
          fps: result.image?.fps,
          duration: result.image?.duration,
        };
        setGeneratedVideoWarnings(
          (result.warnings || []).map((w: any) => (typeof w === "string" ? w : w?.message)).filter(Boolean));
        setGeneratedVideo(videoUrl);
        setGeneratedVideoPlaybackUrl(playbackUrl || null);
        setGeneratedVideoSeed(videoSeed);
        setGeneratedVideoParams(nextItem.params as InpaintParams);
        setGeneratedVideoInfo(videoInfo);
        publishCompletedResult({ panel: "inpaint", kind: "video", url: videoUrl, playbackUrl, info: videoInfo, seed: videoSeed, params: nextItem.params });
        if (onImageGenerated) onImageGenerated(videoUrl, { kind: "video", playbackUrl });
        completeCurrentItem();
      } catch (error: any) {
        console.error("[Inpaint] Video generation failed:", error);
        failCurrentItem();
        // alert() blocks the JS thread; reset state and requeue before showing
        // it, otherwise the queue effect sees a stale isGenerating until the
        // dialog closes.
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        setTimeout(() => {
          if (processQueueRef.current) processQueueRef.current();
        }, 100);
        alert(isGenerationStalledError(error)
          ? error.message
          : `Video inpaint generation failed: ${error?.response?.data?.detail || error?.message || "Unknown error"}`);
        return;
      }
      isGeneratingRef.current = false;
      setIsGenerating(false);
      setProgress(0);
      setProgressMessage("");
      setTimeout(() => {
        if (processQueueRef.current) processQueueRef.current();
      }, 100);
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
    setCfgMetrics([]); // Clear previous metrics

    try {
      let apiParams: ApiInpaintParams = {
        prompt: nextItem.params.prompt,
        negative_prompt: nextItem.params.negative_prompt,
        steps: nextItem.params.steps,
        cfg_scale: nextItem.params.cfg_scale,
        timestep_shift: nextItem.params.timestep_shift, // SenseNova U1.5 flow-matching time-shift
        img_cfg_scale: nextItem.params.img_cfg_scale, // SenseNova U1.5 second CFG scale
        sensenova_mot_phase_eviction: nextItem.params.sensenova_mot_phase_eviction, // SenseNova U1.5 per-phase weight-half CPU eviction
        sensenova_kv_cache_streaming: nextItem.params.sensenova_kv_cache_streaming, // SenseNova U1.5 per-layer KV cache CPU streaming
        sampler: nextItem.params.sampler,
        schedule_type: nextItem.params.schedule_type,
        seed: nextItem.params.seed,
        width: nextItem.params.width,
        height: nextItem.params.height,
        denoising_strength: nextItem.params.denoising_strength,
        vae_drift_correction: nextItem.params.vae_drift_correction,
        mask_blur: nextItem.params.mask_blur,
        inpaint_full_res: nextItem.params.inpaint_full_res,
        inpaint_full_res_padding: nextItem.params.inpaint_full_res_padding,
        inpaint_fill_mode: nextItem.params.inpaint_fill_mode,
        inpaint_fill_strength: nextItem.params.inpaint_fill_strength,
        inpaint_blur_strength: nextItem.params.inpaint_blur_strength,
        // Regional additional prompt (SD/SDXL only, generated/repaint region)
        region_prompt: nextItem.params.region_prompt,
        region_negative_prompt: nextItem.params.region_negative_prompt,
        region_prompt_strength: nextItem.params.region_prompt_strength,
        region_prompt_method: nextItem.params.region_prompt_method,
        region_mask_feather: nextItem.params.region_mask_feather,
        seam_structure_strength: nextItem.params.seam_structure_strength,
        seam_structure_depth: nextItem.params.seam_structure_depth,
        seam_structure_end: nextItem.params.seam_structure_end,
        seam_structure_saliency: nextItem.params.seam_structure_saliency,
        seam_structure_max_area: nextItem.params.seam_structure_max_area,
        boundary_relax_strength: nextItem.params.boundary_relax_strength,
        boundary_relax_width: nextItem.params.boundary_relax_width,
        boundary_relax_noise: nextItem.params.boundary_relax_noise,
        boundary_relax_full_until: nextItem.params.boundary_relax_full_until,
        boundary_relax_end: nextItem.params.boundary_relax_end,
        boundary_relax_paste: nextItem.params.boundary_relax_paste,
        resize_mode: nextItem.params.resize_mode,
        resampling_method: nextItem.params.resampling_method,
        loras: nextItem.params.loras,
        controlnets: nextItem.params.controlnets,
        developer_mode: developerMode,
        // Reset advanced CFG params if disabled
        cfg_schedule_type: !showAdvancedCFG ? "constant" : nextItem.params.cfg_schedule_type,
        cfg_rescale_snr_alpha: !showAdvancedCFG ? 0.0 : nextItem.params.cfg_rescale_snr_alpha,
        dynamic_threshold_percentile: !showAdvancedCFG ? 0.0 : nextItem.params.dynamic_threshold_percentile,
        // NAG params
        nag_enable: nextItem.params.nag_enable,
        nag_scale: nextItem.params.nag_scale,
        nag_tau: nextItem.params.nag_tau,
        nag_alpha: nextItem.params.nag_alpha,
        nag_sigma_end: nextItem.params.nag_sigma_end,
        nag_negative_prompt: nextItem.params.nag_negative_prompt,
        unet_quantization: nextItem.params.unet_quantization,
        quantized_gemm_mode: nextItem.params.quantized_gemm_mode,
        original_size_w: nextItem.params.original_size_w,
        original_size_h: nextItem.params.original_size_h,
        original_size_scale: nextItem.params.original_size_scale,
        attention_type: nextItem.params.attention_type,
        vision_encoder_path: nextItem.params.vision_encoder_path,
        vae_path: nextItem.params.vae_path,
        text_encoder_path: nextItem.params.text_encoder_path,
        pid_sr_output: nextItem.params.pid_sr_output,
        pid_use_gemma: nextItem.params.pid_use_gemma,
        pid_low_vram: nextItem.params.pid_low_vram,
        pid_tile_native: nextItem.params.pid_tile_native,
        pid_tile_overlap_ratio: nextItem.params.pid_tile_overlap_ratio,
        pid_fast_large_decode: nextItem.params.pid_fast_large_decode,
        // Spectrum acceleration
        spectrum_enable: nextItem.params.spectrum_enable,
        // First Block Cache
        fbcache_enable: nextItem.params.fbcache_enable,
        fbcache_threshold: nextItem.params.fbcache_threshold,
        fbcache_warmup_steps: nextItem.params.fbcache_warmup_steps,
        spectrum_w: nextItem.params.spectrum_w,
        spectrum_w_decay: nextItem.params.spectrum_w_decay,
        spectrum_delta_cap: nextItem.params.spectrum_delta_cap,
        spectrum_m: nextItem.params.spectrum_m,
        spectrum_lam: nextItem.params.spectrum_lam,
        spectrum_warmup_steps: nextItem.params.spectrum_warmup_steps,
        spectrum_window_size: nextItem.params.spectrum_window_size,
        spectrum_flex_window: nextItem.params.spectrum_flex_window,
        spectrum_tail: nextItem.params.spectrum_tail,
        spectrum_feature_mode: nextItem.params.spectrum_feature_mode,
        spectrum_cache_branch: nextItem.params.spectrum_cache_branch,
        spectrum_max_cache: nextItem.params.spectrum_max_cache,
        // VAE tiling (model-global). vae_tiling / vae_tile_threshold were
        // missing from this object, so a queued inpaint always fell back to
        // the api.ts defaults regardless of the panel setting.
        vae_tiling: nextItem.params.vae_tiling,
        vae_tile_threshold: nextItem.params.vae_tile_threshold,
        vae_tile_mode: nextItem.params.vae_tile_mode,
        vae_tile_global_norm: nextItem.params.vae_tile_global_norm,
        color_flatten_strength: nextItem.params.color_flatten_strength,
        flatten_in_loop: nextItem.params.flatten_in_loop,
        flatten_in_loop_last_steps: nextItem.params.flatten_in_loop_last_steps,
        flatten_in_loop_min_region: nextItem.params.flatten_in_loop_min_region,
        // Block swap (model-global)
        enable_block_swap: nextItem.params.enable_block_swap,
        blocks_to_swap: nextItem.params.blocks_to_swap,
        use_pinned_memory: nextItem.params.use_pinned_memory,
        block_swap_h2d_only: nextItem.params.block_swap_h2d_only,
        block_swap_ring_size: nextItem.params.block_swap_ring_size,
        // Keep model components GPU-resident for the next queued generation
        // (value is set by the queue dispatcher's hasNext check)
        keep_models_hot: nextItem.params.keep_models_hot,
        // Loop-generation decode mode (heavy-decoder aware). Inpaint never
        // uses loop_decode="none" (backend rejects it) — intermediate loop
        // steps fall back to "cheap"+skip_gallery instead (see addLoopStepsToQueueImmediate).
        loop_decode: nextItem.params.loop_decode,
        skip_gallery: nextItem.params.skip_gallery,
      };

      // Add FLUX.2 Image Edit / Vision Encoder reference images
      if (refImages.length > 0) {
        apiParams = {
          ...apiParams,
          ref_images: refImages,
        };
      }

      console.log('[Inpaint] Generating with params:', {
        loras: apiParams.loras?.length || 0,
        controlnets: apiParams.controlnets?.length || 0,
        unet_quantization: apiParams.unet_quantization,
      });

      let result: any;
      let imageUrl: string;
      // Per-item flag (set at enqueue) so loop steps keep the model choice.
      if ((nextItem?.useTrainingModel ?? useTrainingModel) && (nextItem?.trainingRunId ?? activeTraining?.run_id)) {
        // Training-preview branch: encode init+mask, route to
        // /generate/inpaint/training-preview; result is a blob URL.
        const initImageBase64 = await toBase64(nextItem.inputImage!);
        const maskImageBase64 = await toBase64(nextItem.maskImage!);
        const preview = await generateInpaintTrainingPreview({
          ...(apiParams as any),
          init_image_base64: initImageBase64,
          mask_image_base64: maskImageBase64,
          denoising_strength: apiParams.denoising_strength ?? 0.75,
          run_id: nextItem?.trainingRunId ?? activeTraining!.run_id,
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
          success: true,
          actual_seed: preview.seed ? Number(preview.seed) : -1,
          image: {
            filename: preview.filename
              ?? `preview_${preview.requestId ?? "training"}.png`,
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
        result = await generateInpaint(apiParams, nextItem.inputImage!, nextItem.maskImage!);
        // skip_gallery=true (loop_decode="cheap" intermediate step) returns a
        // top-level filename with NO nested `image` object.
        imageUrl = result.success ? `/outputs/${getResultFilename(result)}` : "";
      }

      if (result.success) {
        const resultSeed = getResultSeed(result);
        const resultAncestralSeed = getResultAncestralSeed(result);
        setGeneratedImage(imageUrl);
        setGeneratedImageSeed(resultSeed);
        setGeneratedImageAncestralSeed(resultAncestralSeed);
        setPreviewImage(null);

        // Save the params used for this generation (with actual result values)
        const completedParams: InpaintParams = {
          ...nextItem.params,
          seed: resultSeed,
          ancestral_seed: resultAncestralSeed ?? -1,
          width: result.image?.width ?? nextItem.params.width,
          height: result.image?.height ?? nextItem.params.height,
        };
        setGeneratedImageParams(completedParams);
        if (imageUrl) {
          publishCompletedResult({
            panel: "inpaint",
            kind: "image",
            url: imageUrl,
            seed: resultSeed,
            ancestralSeed: resultAncestralSeed,
            params: completedParams,
          });
        }

        // Add to the client-side session gallery — but NOT for latent-only or
        // skip_gallery intermediate loop steps (decodeMode "final-only"
        // fallback for inpaint, which never uses loop_decode="none"): the
        // server intentionally skipped the DB record for these, so they'd
        // show transiently then vanish on refresh. Only the final (fully
        // galleried) step should populate it.
        const isEphemeralStep = isLatentOnlyResult(result) || !result.image;
        if (!isEphemeralStep) {
          setGalleryImages(prev => [...prev, { url: imageUrl, timestamp: Date.now() }]);
        }

        // Notify parent component
        if (onImageGenerated) {
          onImageGenerated(imageUrl, { kind: "image" });
        }

        if (isMounted) {
          saveImagePreview(PREVIEW_KEYS, imageUrl);
        }

        // If this item has a loop group, update the next loop step's input image, prompt, and ControlNets
        // Use nextItem (not currentItem from context) to avoid timing issues
        if (nextItem?.loopGroupId !== undefined) {
          const nextLoopStepIndex = (nextItem.loopStepIndex ?? -1) + 1;

          console.log(`[Inpaint] Processing loop step completion:`, {
            loopGroupId: nextItem.loopGroupId,
            currentStepIndex: nextItem.loopStepIndex,
            nextLoopStepIndex,
          });

          // Update input image first
          console.log(`[Inpaint] Updating loop step ${nextLoopStepIndex} with input image:`, imageUrl);
          updateQueueItemByLoop(nextItem.loopGroupId, nextLoopStepIndex, { inputImage: imageUrl });

          // If TIPO was used for base generation, update loop steps with TIPO-generated prompt.
          // Guarded with optional chaining: skip_gallery=true responses have no nested `image`.
          if (nextItem.loopStepIndex === -1 && nextItem.params.use_tipo && result.image?.prompt) {
            console.log(`[Inpaint] Base generation used TIPO, updating all loop steps with TIPO prompt`);
            console.log(`[Inpaint] Original prompt: ${nextItem.params.prompt?.substring(0, 100)}...`);
            console.log(`[Inpaint] TIPO prompt: ${result.image.prompt?.substring(0, 100)}...`);

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

          console.log(`[Inpaint] Step config:`, {
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
                console.log(`[Inpaint] Scale mode: ${imageWidth}x${imageHeight} * ${stepConfig.scale} = ${scaledWidth}x${scaledHeight}`);

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
            console.log(`[Inpaint] Processing ${stepConfig.controlnets.length} ControlNet(s) for loop step ${nextLoopStepIndex}`);

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

            console.log(`[Inpaint] Converted image to base64, length: ${imageBase64.length}`);

            // Update ControlNets with useLoopImage enabled using callback to preserve existing params
            updateQueueItemByLoop(nextItem.loopGroupId!, nextLoopStepIndex, (item) => {
              const updatedControlnets = stepConfig.controlnets.map((cnConfig, idx) => {
                console.log(`[Inpaint] ControlNet ${idx}: useLoopImage=${cnConfig.useLoopImage}`);
                if (cnConfig.useLoopImage) {
                  console.log(`[Inpaint] Setting image_base64 for ControlNet ${idx}`);
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

            console.log(`[Inpaint] ControlNet images updated for loop step ${nextLoopStepIndex}`);
          }
        }

        // Reset state first, then complete item
        console.log("[Inpaint] Generation complete, resetting state and completing item");
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        completeCurrentItem();

        // Wait briefly for state to propagate, then trigger next
        setTimeout(() => {
          console.log("[Inpaint] Triggering next queue item");
          if (processQueueRef.current) {
            processQueueRef.current();
          }
        }, 100);
      } else {
        // alert() blocks the JS thread; reset state and requeue before
        // showing it, otherwise the queue effect sees a stale isGenerating
        // until the dialog closes.
        isGeneratingRef.current = false;
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        failCurrentItem();

        setTimeout(() => {
          if (processQueueRef.current) {
            processQueueRef.current();
          }
        }, 100);
        alert("Generation failed");
      }
    } catch (error: any) {
      console.error("Generation error:", error);
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
        alertMessage = "Generation failed: " + (error instanceof Error ? error.message : String(error));
      }

      // Reset state first, then fail item
      console.log("[Inpaint] Generation failed, resetting state and failing item");
      isGeneratingRef.current = false;
      setIsGenerating(false);
      setProgress(0);
      setProgressMessage("");
      failCurrentItem();

      // Wait briefly for state to propagate, then trigger next
      setTimeout(() => {
        console.log("[Inpaint] Triggering next queue item after failure");
        if (processQueueRef.current) {
          processQueueRef.current();
        }
      }, 100);

      if (alertMessage) {
        alert(alertMessage);
      }
    }
  }, [isGenerating, generatedImage, onImageGenerated, isMounted, startNextInQueue, completeCurrentItem, failCurrentItem, updateQueueItem, queue, publishCompletedResult]);

  processQueueRef.current = processQueue;

  // Auto-start queue processing when queue has pending items and not currently generating
  useEffect(() => {
    const hasPendingItems = queue.some(item =>
      item.status === "pending" && (item.type === "inpaint" || item.type === "inpaint_vid"));
    const isCurrentItemNull = currentItem === null;

    console.log("[Inpaint] Queue effect:", {
      hasPendingItems,
      isCurrentItemNull,
      isGenerating,
      queueLength: queue.length,
      queue: queue,
      currentItem: currentItem,
      generateForever
    });

    // If generate forever is enabled and queue is empty, add new item. Image
    // mode only -- the video branch has no mask and one clip per request.
    if (generateForever && !isVideo && !hasPendingItems && isCurrentItemNull && !isGenerating && params.prompt && inputImagePreview && maskImage) {
      console.log("[Inpaint] Generate forever: Adding new item to queue");
      handleAddToQueue();
      return;
    }

    // A queue survives a page reload and a backend restart, so on mount there
    // can be pending items with no model loaded yet. Dispatching then earns an
    // immediate 400 and the item is marked failed for a reason that has nothing
    // to do with the item. Hold instead: `modelLoaded` is a dependency, so the
    // queue starts by itself once a model is up.
    if (hasPendingItems && isCurrentItemNull && !isGenerating && !modelLoaded) {
      console.log("[Inpaint] Queue held: no model loaded yet");
      return;
    }

    if (hasPendingItems && isCurrentItemNull && !isGenerating) {
      console.log("[Inpaint] Auto-starting queue processing");
      processQueue();
    }
  }, [queue, currentItem, isGenerating, processQueue, generateForever, params, inputImagePreview, maskImage, modelLoaded]);

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
    // `videoFile` is a dependency for the same reason the image inputs are:
    // handleAddToQueue closes over whichever modality's input it will send.
  }, [params, inputImage, inputImagePreview, maskImage, videoFile]);

  // Render functions for each Inpaint Options tab (see INPAINT_OPTIONS_TABS /
  // INPAINT_OPTIONS_TAB_KEYS / isInpaintOptionsTabActive above). Every control
  // below is unchanged from its original in-Card location -- same param
  // binding / handler / conditional reveal -- ported from OutpaintPanel's
  // outpaintOptionsTabRender pattern.
  const inpaintOptionsTabRender: Record<InpaintOptionsTabId, () => JSX.Element> = {
    inpaint: () => (
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
            id="inpaint_fix_steps"
            checked={params.img2img_fix_steps ?? true}
            onChange={(e) => setParams({ ...params, img2img_fix_steps: e.target.checked })}
            className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
          />
          <label htmlFor="inpaint_fix_steps" className="text-sm text-gray-300">
            Do full steps (ensures complete denoising regardless of strength)
          </label>
        </div>
        <Slider
          label="Mask Blur"
          min={0}
          max={64}
          step={1}
          value={params.mask_blur}
          onChange={(e) => setParams({ ...params, mask_blur: parseInt(e.target.value) })}
        />

        <Select
          label="Masked Content Fill"
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
              value={params.inpaint_fill_strength || 1.0}
              onChange={(e) => setParams({ ...params, inpaint_fill_strength: parseFloat(e.target.value) })}
            />
            {params.inpaint_fill_mode === "blur" && (
              <Slider
                label="Blur Strength"
                min={0.1}
                max={5.0}
                step={0.1}
                value={params.inpaint_blur_strength || 1.0}
                onChange={(e) => setParams({ ...params, inpaint_blur_strength: parseFloat(e.target.value) })}
              />
            )}
          </>
        )}

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

    regional_prompt: () => (
      <div className="space-y-3">
        {/* Regional additional prompt: conditions ONLY the repaint mask
            region, leaving the main prompt + preserved region untouched.
            See backend/api/routes.py generate_inpaint region_* Form params. */}
        <p className="text-xs text-gray-500">
          Conditions only the repaint mask region — the main prompt above and the preserved (unmasked) pixels are unaffected.
          Cost: "cfg" runs an extra regional denoise branch (up to ~2x U-Net forwards). "attention" adds no extra forward pass.
        </p>
        <TextareaWithTagSuggestions
          label="Generated-region positive prompt"
          placeholder="Additional prompt applied only inside the mask..."
          rows={2}
          value={params.region_prompt || ""}
          onChange={(e) => setParams({ ...params, region_prompt: e.target.value })}
          enableWeightControl={true}
        />
        <div className="relative">
          <TextareaWithTagSuggestions
            label="Generated-region negative prompt"
            placeholder="Additional negative prompt applied only inside the mask..."
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
        {/* Seam Structure Continuity (SSC): continues thin structures that
            cross the region boundary (a held rod/staff, limb, torso, lines)
            into the generated/repainted region. See backend/api/routes.py
            generate_inpaint seam_structure_* Form params. */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-gray-300">Seam Structure Continuity</div>
          <p className="text-xs text-gray-500">
            SD/SDXL only. Continues thin structures that cross the region boundary (a held rod/staff, limb, torso, lines) into the generated/repainted region.
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
            generate_inpaint boundary_relax_* Form params. */}
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
            </>
          )}
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
            id="spectrum_enable_inpaint"
            checked={params.spectrum_enable || false}
            onChange={(e) => setParams({ ...params, spectrum_enable: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="spectrum_enable_inpaint" className="text-sm text-gray-300">
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
            id="fbcache_enable_inpaint"
            checked={params.fbcache_enable || false}
            onChange={(e) => setParams({ ...params, fbcache_enable: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="fbcache_enable_inpaint" className="text-sm text-gray-300">
            First Block Cache (dynamic caching)
          </label>
          <span className="text-xs text-gray-500">
            (mutually exclusive with Spectrum)
          </span>
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
            id="flatten_in_loop_inpaint"
            checked={params.flatten_in_loop || false}
            onChange={(e) => setParams({ ...params, flatten_in_loop: e.target.checked })}
            className="rounded"
          />
          <label htmlFor="flatten_in_loop_inpaint" className="text-sm text-gray-300" title="During the final denoise steps, detects the flat background region and replaces it with its solid dominant color (both luma and chroma become uniform - stronger than Color Flatten); no-op when no confident flat region is found; SD/SDXL only for now.">
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
                id="enable_block_swap_inpaint"
                checked={params.enable_block_swap || false}
                onChange={(e) => setParams({ ...params, enable_block_swap: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="enable_block_swap_inpaint" className="text-sm text-gray-300">
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
                    id="use_pinned_memory_inpaint"
                    checked={params.use_pinned_memory || false}
                    onChange={(e) => setParams({ ...params, use_pinned_memory: e.target.checked })}
                    className="rounded"
                  />
                  <label htmlFor="use_pinned_memory_inpaint" className="text-xs text-gray-300">
                    Use Pinned Memory (faster transfer, more RAM)
                  </label>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="block_swap_h2d_only_inpaint"
                    checked={params.block_swap_h2d_only || false}
                    onChange={(e) => setParams({ ...params, block_swap_h2d_only: e.target.checked })}
                    className="rounded"
                  />
                  <label htmlFor="block_swap_h2d_only_inpaint" className="text-xs text-gray-300">
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
  const promptPanel = (
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
          suggestionMode={loadedArchType === "minimax_h3" ? "h3" : "tags"}
          enableWeightControl
        />
      </div>
      {loadedArchType === "minimax_h3" && (
        <H3PromptAssist
          prompt={params.prompt}
          onApply={(prompt) => setParams((previous) => ({ ...previous, prompt }))}
          suggestedMode={
            isH3Ref2VaInpaint && countMiniMaxH3References(h3References) > 0 ? "ref2va" : "t2va"
          }
          durationSeconds={videoDurationSec ?? Math.max(1, estimatedRawFrames) / clipFrameRate}
          // Built from the actual reference set (M5 fix): the input clip is a
          // pin, not a labeled reference, so it must not be counted here.
          references={createH3ReferenceInventory({
            pictures: h3References.images.length,
            videos: h3References.videos.length,
            audios: h3References.audios.length + h3References.videoAudios.filter(Boolean).length,
          })}
        />
      )}
      {!isVideo && (
        <div className="flex flex-wrap items-center gap-1.5 rounded bg-gray-800 px-2 py-1.5">
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
        </div>
      )}
      <TextareaWithTagSuggestions
        label="Negative Prompt"
        placeholder={supportsNegativePrompt ? "Enter negative prompt..." : "Negative prompting is unavailable for this model"}
        rows={2}
        resizeStorageKey={GENERATION_NEGATIVE_PROMPT_HEIGHT_KEY}
        value={params.negative_prompt}
        onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
        suggestionMode={loadedArchType === "minimax_h3" ? "h3" : "tags"}
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
          onModelLoad={(loadedModelInfo) => {
            // Auto-adjust sampler/schedule for Flow Matching models (Z-Image, FLUX.2)
            const modelType = loadedModelInfo?.type;
            if (modelType === "zimage" || modelType === "flux2" || modelType === "anima") {
              // Flow Matching models: use Euler with flow schedule
              setParams(prev => ({
                ...prev,
                sampler: "euler",
                schedule_type: "flow"
              }));
              console.log("[Inpaint] Auto-set sampler=euler, schedule_type=flow for Flow Matching model");
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
          storageKeyPrefix="inpaint"
        />

        <GenerationLeadGrid
          prompt={promptPanel}
          conditioning={(
            <>
        {/* ── Video temporal inpaint: the clip, the range, the video params.
            These replace the image input + mask surface when a video model is
            loaded; the image mode below is unchanged. ── */}
        {isVideo && !supportsTemporalInpaint && (
          <Card title="Video Inpaint">
            <p className="text-sm text-amber-400">
              {temporalInpaintReason
                || `${loadedArchName} does not implement temporal inpaint.`}
            </p>
            <p className="text-xs text-gray-500 mt-2">
              To extend a clip instead of regenerating part of it, use the Outpaint tab.
            </p>
          </Card>
        )}

        {isVideo && supportsTemporalInpaint && isH3Ref2VaInpaint && (
          <>
            <details className="group -mb-1 rounded-md border border-gray-800/80 bg-gray-900/35 px-3 py-1.5 text-xs text-gray-500">
              <summary className="cursor-pointer select-none text-gray-400 marker:text-gray-600 hover:text-gray-300">
                MiniMax reference behavior on temporal inpaint
              </summary>
              <p className="mt-2 leading-relaxed">
                The preserved frames outside the regenerate range already
                condition the vision stream, so a reference set can be
                audio only here (unlike /generate/ref2vid, where an audio
                reference has to be paired with an image or video
                reference). An audio reference contributes audio rows to the
                packed sequence ahead of the generated span; whether the
                model reads them while the clip&apos;s frames are pinned has
                not been measured. The response carries a warning saying so.
              </p>
            </details>
            <MiniMaxH3ReferenceSelector
              value={h3References}
              onChange={setH3References}
              referenceImageSize={h3ReferenceImageSize}
              onReferenceImageSizeChange={setH3ReferenceImageSize}
              disabled={isGenerating}
              storageKey="inpaint_h3_references"
              allowAudioAlone
            />
          </>
        )}

        {isVideo && supportsTemporalInpaint && (
        <Card
          title="Input Video"
          collapsible={true}
          defaultCollapsed={false}
          storageKey="inpaint_video_input_collapsed"
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
                <Button onClick={handleClearVideo} variant="secondary" size="sm" title="Clear input clip">
                  Clear
                </Button>
              )}
            </div>
            <div
              ref={videoContainerRef}
              className={
                isVideoFullscreen
                  ? "relative w-full h-full bg-gray-800"
                  : "relative h-[clamp(10rem,22vh,13rem)] bg-gray-800 rounded-lg overflow-hidden border-2 border-dashed border-gray-600"
              }
            >
              {videoPreviewUrl ? (
                <>
                  <video
                    ref={inputVideoRef}
                    src={videoPreviewUrl}
                    onLoadedMetadata={handleVideoLoadedMetadata}
                    className="w-full h-full object-contain"
                    controls
                    controlsList="nofullscreen"
                    disablePictureInPicture
                    muted
                    playsInline
                  />
                  <VideoMaskPreviewOverlay
                    videoRef={inputVideoRef}
                    nativeSize={inputVideoSize}
                    outputWidth={videoMaskManifest.canvas.width}
                    outputHeight={videoMaskManifest.canvas.height}
                    manifest={videoMaskManifest}
                    assets={videoMaskAssets}
                    assetRefs={videoMaskUploadedRefsRef.current}
                    rangeStart={videoMaskOverlaySpan.start}
                    rangeEnd={videoMaskOverlaySpan.end}
                    // inputVideoPlayer.currentFrame is the <video> element's
                    // RAW frame number; sampleFrames/keyframe.frame/rangeStart/
                    // End are all in TRIMMED-clip coordinates (see
                    // VideoInpaintTimeline's own `- trimStart`), so the raw
                    // frame must be shifted into that same coordinate space
                    // here or the overlay picks the wrong sample whenever
                    // input_trim_start_frames > 0.
                    currentFrame={
                      inputVideoPlayer.currentFrame != null
                        ? inputVideoPlayer.currentFrame - (params.input_trim_start_frames ?? 0)
                        : null
                    }
                    enabled={videoMaskPreviewEnabled}
                    opacity={videoMaskPreviewOpacity}
                  />
                  {/* Pushed down below the overlay's own top-1 badges (Updating
                      mask preview.../Mask preview unavailable/Playhead outside
                      regenerate range) instead of sharing their corner -- both
                      used to sit at top-*-2/top-1 right-1 and occluded each
                      other. Still clear of the native <video> control bar,
                      which is pinned to the bottom of this container. */}
                  {/* isMounted as well: the feature detection below reads
                      `document`, so it is false during SSR and would
                      otherwise mismatch on hydration. */}
                  {isMounted && canFullscreenContainer && (
                    <button
                      type="button"
                      onClick={toggleVideoFullscreen}
                      className="absolute top-8 right-2 z-10 p-1.5 rounded-lg bg-gray-900/70 text-white hover:bg-gray-900/90"
                      aria-label={isVideoFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                      title={isVideoFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                    >
                      {isVideoFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                    </button>
                  )}
                </>
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <p className="text-gray-500 text-center px-4">Use the file picker above to select an mp4/webm clip</p>
                </div>
              )}
            </div>
            {videoPreviewUrl && videoMaskManifest.keyframes.length > 0 && (
              <div className="flex items-center gap-3 text-xs text-gray-400">
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={videoMaskPreviewEnabled}
                    onChange={(e) => setVideoMaskPreviewEnabled(e.target.checked)}
                    className="w-3.5 h-3.5 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
                  />
                  Mask preview overlay
                </label>
                {videoMaskPreviewEnabled && (
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={videoMaskPreviewOpacity}
                    onChange={(e) => setVideoMaskPreviewOpacity(parseFloat(e.target.value))}
                    className="w-24"
                    aria-label="Mask preview overlay opacity"
                  />
                )}
                <InlineHelp label="Mask preview overlay details">
                  <p>
                    Rasterized on the backend from the exact keyframe timeline (the same
                    hold/affine/sdf mechanism generation uses), downscaled for preview and shown at
                    the nearest sampled frame to the playhead. It is not a live per-frame render, and
                    a sharp edge here is not a claim about the exact per-pixel boundary generation
                    produces.
                  </p>
                  <p>
                    Only shown while the playhead sits inside the regenerate range -- outside it
                    (including right after upload, before the range is widened or the playhead is
                    moved into it) a badge on the video says so instead of drawing nothing
                    unexplained.
                  </p>
                </InlineHelp>
              </div>
            )}

            {videoDurationSec != null && (
              <div className="space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Clip frames</label>
                    <NumberInput
                      label="Clip frames"
                      value={videoRawFrames}
                      onCommit={(v) => setClipFramesOverride(Math.max(1, v))}
                      min={1}
                      step={1}
                      parse="int"
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Trim start (frames)</label>
                    <NumberInput
                      label="Trim start"
                      value={params.input_trim_start_frames ?? 0}
                      onCommit={(v) => setParams(prev => ({ ...prev, input_trim_start_frames: Math.max(0, v) }))}
                      min={0}
                      step={1}
                      parse="int"
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Trim end (frames)</label>
                    <NumberInput
                      label="Trim end"
                      value={params.input_trim_end_frames ?? 0}
                      onCommit={(v) => setParams(prev => ({ ...prev, input_trim_end_frames: Math.max(0, v) }))}
                      min={0}
                      step={1}
                      parse="int"
                      className="w-full"
                    />
                  </div>
                </div>
                <div className="flex items-center gap-1 text-xs text-gray-500">
                  <span>{videoDurationSec.toFixed(2)}s · about {estimatedRawFrames} frames at {clipFrameRate} fps</span>
                  <InlineHelp label="Clip length and trim details">
                    <p>The browser reports duration rather than an exact frame count. Correct Clip frames when needed.</p>
                    <p>The trimmed length must match a length supported by the model. Generation never trims automatically because that would remove frames selected for preservation.</p>
                  </InlineHelp>
                </div>
                {videoTrimmedLengthValid ? (
                  <p className="text-xs text-green-400">
                    Trimmed clip: {videoTrimmedFrames} frames — a length this model generates.
                  </p>
                ) : (
                  <div className="space-y-2">
                    <p className="text-xs text-amber-400">
                      Trimmed clip: {videoTrimmedFrames} frames, which this model cannot generate
                      {videoConstraints && (
                        <> (valid lengths are {videoConstraints.frame_multiple}n
                          {videoConstraints.frame_offset ? `+${videoConstraints.frame_offset}` : ""},
                          {" "}{videoConstraints.min_frames}
                          {videoConstraints.max_frames != null ? `-${videoConstraints.max_frames}` : ""})</>
                      )}
                      {videoTargetLength != null
                        ? `. Trimming ${videoRawFrames - videoTargetLength} frame(s) off the upload reaches ${videoTargetLength}.`
                        : ". No trim of this clip reaches a valid length; it is shorter than this model's shortest clip."}
                    </p>
                    {videoTargetLength != null && (
                      <Button
                        onClick={() => applyClipLength(videoRawFrames, videoTargetLength)}
                        variant="secondary"
                        size="sm"
                      >
                        Fit to a valid length ({videoTargetLength} frames)
                      </Button>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </Card>
        )}

        {isVideo && supportsTemporalInpaint && (
        <Card title="Regenerate Range & Mask Keyframes">
          {/* `frameRate` is the CLIP's rate, so the seconds in the readout line
              up with the input player the range is picked against. One
              timeline now hosts both the regenerate-range track and the
              mask-keyframe track on a single shared ruler/playhead. */}
          <VideoInpaintTimeline
            rawFrames={videoRawFrames}
            trimStart={params.input_trim_start_frames ?? 0}
            trimEnd={params.input_trim_end_frames ?? 0}
            latentChunkPattern={latentChunkPattern}
            start={params.regenerate_start_frame ?? 0}
            end={params.regenerate_end_frame ?? 0}
            onRangeChange={(start, end) => {
              // Does NOT touch videoMaskManifest/videoMaskAssets. This fires on
              // every pointer-move while a handle is dragged (not just on
              // release), so pruning mask keyframes here used to delete any
              // keyframe the handle passed over -- and its PNG asset -- even if
              // the drag ended back on a range that still contained it.
              // Out-of-range keyframes are kept and surfaced via a read-only
              // count computed inside VideoInpaintTimeline; submission already
              // refuses to proceed while any exist.
              lastValidRegenerateRangeRef.current = { start, end };
              setRegenerateRangeReplacedNotice(null);
              setParams(prev => ({
                ...prev,
                regenerate_start_frame: start,
                regenerate_end_frame: end,
              }));
            }}
            frameRate={clipFrameRate}
            disabled={!videoPreviewUrl}
            videoSrc={videoPreviewUrl}
            player={inputVideoPlayer}
            keyframes={videoMaskManifest.keyframes}
            onChange={handleVideoMaskKeyframesChange}
            onAddKeyframe={handleAddVideoMaskKeyframe}
            onEditKeyframe={handleEditVideoMaskKeyframe}
            compositeFeatherPx={videoMaskManifest.compositeFeatherPx}
            onCompositeFeatherPxChange={handleVideoMaskFeatherChange}
            assets={videoMaskAssets}
            maskDisabled={isGenerating || !videoTrimmedLengthValid || videoTrimmedFrames <= 0}
            maskDisabledReason={
              isGenerating
                ? undefined
                : videoDurationSec == null && clipFramesOverride == null
                  ? "Mask editing is unavailable until a clip's length has been read."
                  : videoTrimmedFrames <= 0
                    ? "Mask editing is unavailable because the trim removes the whole clip. Reduce Trim start/end above."
                    : !videoTrimmedLengthValid
                      ? "Mask editing is unavailable while the trimmed clip length is off this architecture's frame grid. Use Fit, or adjust the trim, first."
                      : undefined
            }
            canUndo={videoMaskHistory.canUndo}
            canRedo={videoMaskHistory.canRedo}
            onUndo={() => videoMaskHistory.undo(currentVideoMaskSnapshot())}
            onRedo={() => videoMaskHistory.redo(currentVideoMaskSnapshot())}
          />
          {regenerateRangeReplacedNotice && (
            <p className="mt-2 text-xs text-amber-400">{regenerateRangeReplacedNotice}</p>
          )}
          {videoPreviewUrl && (
            <p className={`mt-2 text-xs ${
              (params.regenerate_end_frame ?? 0) >= videoTrimmedFrames && videoTrimmedFrames > 0
                ? "text-amber-400"
                : "text-gray-500"
            }`}>
              This replaces frames {params.regenerate_start_frame ?? 0} to{" "}
              {Math.max((params.regenerate_end_frame ?? 0) - 1, params.regenerate_start_frame ?? 0)}
              {" "}of the trimmed clip ({videoTrimmedFrames} frames). Output length equals input length
              ({videoTrimmedFrames} frames); no frames are added.
              {(params.regenerate_end_frame ?? 0) >= videoTrimmedFrames && videoTrimmedFrames > 0
                && " The selected range reaches the end of the clip, so this overwrites its current tail rather than extending it."}
            </p>
          )}
          {(videoMaskError || videoMaskPersistenceNotice || videoMaskCanvasMismatch) && (
            <div className="mt-4 border-t border-gray-700 pt-4">
              {videoMaskError && (
                <p className="mt-2 text-xs text-amber-400" role="alert">{videoMaskError}</p>
              )}
              {videoMaskPersistenceNotice && (
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <p className="text-xs text-amber-400" role="alert">{videoMaskPersistenceNotice}</p>
                  {videoMaskRestoreAbortedRef.current && (
                    <button
                      type="button"
                      onClick={handleDiscardStuckVideoMaskRestore}
                      className="rounded border border-gray-600 px-2 py-0.5 text-xs text-gray-300 hover:bg-gray-700"
                    >
                      Discard saved keyframes
                    </button>
                  )}
                </div>
              )}
              {videoMaskCanvasMismatch && (
                <p className="mt-2 text-xs text-amber-400" role="alert">
                  {staleVideoMaskAssets.length} mask asset{staleVideoMaskAssets.length === 1 ? "" : "s"}{" "}
                  {staleVideoMaskAssets.length === 1 ? "was" : "were"} drawn for an output canvas other than the
                  current {videoMaskCanvasWidth}x{videoMaskCanvasHeight}. Recreate the affected keyframes before
                  generating.
                </p>
              )}
            </div>
          )}
          <div className="mt-2 flex items-center gap-1 text-xs text-gray-500">
            <span>Interior range conditioning is experimental</span>
            <InlineHelp label="Interior range support details">
              <p>
                MiniMax documents first- and last-frame conditioning with up to two images. Interior range anchors use the same mechanism but are not covered by its model card.
              </p>
            </InlineHelp>
          </div>
          <p className="mt-2 text-xs text-gray-500">
            To add frames and increase the clip's length, use the Outpaint tab instead.
          </p>
        </Card>
        )}

        {!isVideo && (
        <Card
          title="Input Image"
          collapsible={true}
          defaultCollapsed={false}
          storageKey="inpaint_input_collapsed_v2"
          collapsedPreview={
            inputImagePreview ? (
              <span className="flex items-center gap-2 text-sm">
                <span className="text-green-400">✓ Image loaded</span>
                {maskImage && <span className="text-blue-400">| Mask set</span>}
              </span>
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
                <>
                  <Button
                    onClick={handleClearInputImage}
                    variant="secondary"
                    size="sm"
                    title="Clear input image and mask"
                  >
                    Clear
                  </Button>
                  {maskImage && (
                    <Button
                      onClick={handleClearMask}
                      variant="secondary"
                      size="sm"
                      title="Clear mask only"
                    >
                      Clear Mask
                    </Button>
                  )}
                </>
              )}
            </div>
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onDoubleClick={handleInputImageDoubleClick}
              className={`h-[clamp(11rem,24vh,15rem)] bg-gray-800 rounded-lg overflow-hidden border-2 border-dashed transition-colors relative ${
                isDragging
                  ? 'border-blue-500 bg-gray-700'
                  : inputImagePreview
                  ? 'border-gray-600 cursor-pointer hover:border-blue-500'
                  : 'border-gray-600'
              }`}
              title={inputImagePreview ? 'Double-click to edit and add inpaint mask' : ''}
            >
              {inputImagePreview ? (
                <>
                  <img
                    src={inputImagePreview}
                    alt="Input"
                    className="w-full h-full object-contain"
                  />
                  {maskImage && (
                    <img
                      src={maskImage}
                      alt="Mask overlay"
                      className="absolute inset-0 w-full h-full object-contain"
                      style={{
                        pointerEvents: 'none',
                        mixBlendMode: MASK_OVERLAY_CSS_MIX_BLEND_MODE,
                        opacity: MASK_OVERLAY_ALPHA
                      }}
                      title="Mask overlay - highlighted areas will be inpainted"
                    />
                  )}
                </>
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <p className="text-gray-500 text-center px-4">
                    {isDragging
                      ? 'Drop image here'
                      : 'Drag and drop an image here or use the file picker above'}
                  </p>
                </div>
              )}
            </div>
            {inputImagePreview && (
              <p className="text-xs text-gray-500 text-center">
                💡 Double-click image to edit and draw inpaint mask
              </p>
            )}
          </div>
        </Card>
        )}


        {/* FLUX.2 Image Edit / SenseNova U1.5 / Vision Encoder: Reference Images */}
        {!isVideo && (currentModelInfo?.model_info?.type === "flux2" || isSenseNovaModel || params.vision_encoder_path) && (
          <Card
            title={
              currentModelInfo?.model_info?.type === "flux2"
                ? "FLUX.2 Image Edit (Reference Images)"
                : isSenseNovaModel
                ? "SenseNova U1.5 (Reference Images)"
                : "Vision Encoder (Reference Images)"
            }
            collapsible={true}
            defaultCollapsed={true}
            storageKey="inpaint_ref_images_collapsed"
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
                  disabled={refImages.length >= maxRefImages}
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
                      ? `Drop images here (max ${maxRefImages})`
                      : `Drag and drop images here or use the file picker above (max ${maxRefImages})`}
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
                    {refImagePreviews.length < maxRefImages && (
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
                    💡 {refImagePreviews.length}/{maxRefImages} images. {refImagePreviews.length < maxRefImages ? 'Drop more images in the area above' : 'Max reached'}
                  </p>
                </div>
              )}
            </div>
          </Card>
        )}
            </>
          )}
        />

        {/* Inpaint Options: a single-open tabbed accordion (chrome shared via
            frontend/src/components/common/TabbedOptions.tsx). Every control
            below is unchanged from its original location (same param
            binding / handler / conditional reveal) -- only the container
            changed. See INPAINT_OPTIONS_TAB_KEYS / isInpaintOptionsTabActive /
            inpaintOptionsTabRender above. */}
        {!isVideo && (
        <TabbedOptions<InpaintParams>
          cardTitle="Inpaint Options"
          params={params}
          setParams={setParams}
          defaultParams={DEFAULT_PARAMS}
          tabs={INPAINT_OPTIONS_TABS.map((tab) => ({
            id: tab.id,
            label: tab.label,
            keys: INPAINT_OPTIONS_TAB_KEYS[tab.id],
            isActive: (p: InpaintParams) => isInpaintOptionsTabActive(tab.id, p),
            render: inpaintOptionsTabRender[tab.id],
          }))}
        />
        )}

        {!isVideo && (
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
              {supportsTimestepShift && (
                <Slider
                  label="Timestep Shift"
                  min={0.1}
                  max={10.0}
                  step={0.1}
                  value={params.timestep_shift ?? generationDefaults?.inpaint?.timestep_shift ?? 3.0}
                  onChange={(e) => setParams({ ...params, timestep_shift: parseFloat(e.target.value) })}
                />
              )}
              {supportsImgCfgScale && (
                <Slider
                  label="Image CFG Scale"
                  min={0}
                  max={10.0}
                  step={0.1}
                  value={params.img_cfg_scale ?? generationDefaults?.inpaint?.img_cfg_scale ?? 1.0}
                  onChange={(e) => setParams({ ...params, img_cfg_scale: parseFloat(e.target.value) })}
                  title="SenseNova U1.5 second CFG scale, applied alongside CFG Scale only when reference images are supplied. At 1.0, sampling uses the reference-conditioned branch as the guidance baseline."
                />
              )}
            </div>

            {supportsSensenovaMotPhaseEviction && (
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={params.sensenova_mot_phase_eviction || false}
                  onChange={(e) => setParams({ ...params, sensenova_mot_phase_eviction: e.target.checked })}
                  className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
                />
                <label
                  className="text-sm font-medium text-gray-300"
                  title="Moves the generation-branch weights to pinned CPU memory at the start of the prefix phase; at the start of the denoise phase, moves the understanding-branch weights to pinned CPU memory first, then moves the generation-branch weights back to GPU (three half-transfers per generation, roughly 22.6 GiB of PCIe traffic). Measured host-RAM cost: process resident memory rises by about 21.7 GiB once eviction first engages (15.11 GiB pinned - 7.55 GiB live + 7.55 GiB pooled - plus pageable staging copies), stays flat across subsequent generations, and is not returned to the OS; it persists for the process lifetime, including for later generations run with this toggle off."
                >
                  SenseNova Phase-Eviction (CPU offload of unused weight half)
                </label>
              </div>
            )}

            {supportsSensenovaKvCacheStreaming && (
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={params.sensenova_kv_cache_streaming || false}
                  onChange={(e) => setParams({ ...params, sensenova_kv_cache_streaming: e.target.checked })}
                  className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
                />
                <label
                  className="text-sm font-medium text-gray-300"
                  title="Streams the prefix phase's KV cache from pinned host memory per layer, through a 2-slot GPU ring shared across layers and branches, instead of keeping all 42 layers' KV buffers GPU-resident. Independent of SenseNova Phase-Eviction above; the two toggles may be combined or used alone."
                >
                  SenseNova KV Cache Streaming (streams prefix KV cache from pinned CPU per layer)
                </label>
              </div>
            )}

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
                              name="aspect_base_inpaint"
                              value="width"
                              defaultChecked
                              className="w-3 h-3"
                            />
                            <span className="text-xs text-gray-300">Width</span>
                          </label>
                          <label className="flex items-center gap-1 cursor-pointer">
                            <input
                              type="radio"
                              name="aspect_base_inpaint"
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
                              const baseOn = (document.querySelector('input[name="aspect_base_inpaint"]:checked') as HTMLInputElement)?.value || 'width';
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

            {/* Commented out: Not implemented in backend
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="inpaint_full_res"
                checked={params.inpaint_full_res || false}
                onChange={(e) => setParams({ ...params, inpaint_full_res: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="inpaint_full_res" className="text-sm">
                Inpaint at full resolution
              </label>
            </div>

            {params.inpaint_full_res && (
              <Slider
                label="Only masked padding"
                min={0}
                max={256}
                step={4}
                value={params.inpaint_full_res_padding}
                onChange={(e) => setParams({ ...params, inpaint_full_res_padding: parseInt(e.target.value) })}
              />
            )}
            */}

          </div>
        </Card>
        )}

        {isVideo && supportsTemporalInpaint && (
        <Card title="Video">
          {/* Canvas, in the image panels' Parameters-card shape (labelled
              sliders, Absolute/Scale size mode). Scale derives width/height
              from the clip's own dimensions through fitVideoCanvas, which is
              what resolves "the clip's own resolution" into a canvas this
              architecture accepts. */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-300">Size Mode</label>
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
                    value={params.width ?? videoWidthBounds.min}
                    onChange={(e) => setParams({ ...params, width: parseInt(e.target.value) })}
                  />
                  <Slider
                    label={`Height (÷${videoHeightBounds.step})`}
                    min={videoHeightBounds.min}
                    max={videoHeightBounds.max}
                    step={videoHeightBounds.step}
                    value={params.height ?? videoHeightBounds.min}
                    onChange={(e) => setParams({ ...params, height: parseInt(e.target.value) })}
                  />
                </div>
                {videoWidthBounds.capped && (
                  <p className="text-xs text-gray-500 mt-1">
                    {videoCanvasRule(archCapabilities, loadedArchType)}. The cap is on the short and
                    long edges rather than on width and height, so each slider stops at the largest
                    edge the other axis currently allows.
                  </p>
                )}
                {videoCanvasOverEnvelope && (
                  <p className="text-xs text-amber-400 mt-1">
                    The canvas is {params.width}x{params.height}, which is outside this model&apos;s
                    envelope. The value is kept as set — it is not moved for you — and this model
                    refuses it, so change it before generating.
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

            {/* WHAT THE CANVAS DOES TO THE CLIP. The backend fits the upload to
                width x height (centre-crop, then resize) and it is that fitted
                result which is preserved, so a canvas that is not the clip's own
                size changes what "preserved" means. Stated, not advised. */}
            {inputVideoSize && (
              <p className={`text-xs mt-2 ${videoCanvasIsSourceSize ? "text-gray-500" : "text-amber-400"}`}>
                {videoCanvasIsSourceSize ? (
                  <>The canvas is the input clip&apos;s own resolution, so the preserved frames are the
                    uploaded frames.</>
                ) : (
                  <>The canvas is {params.width}x{params.height}; the input clip is{" "}
                    {inputVideoSize.width}x{inputVideoSize.height}. The clip is fitted to the canvas
                    once — centre-cropped to the canvas aspect ratio, then resized — and it is that
                    fitted version, not the upload, whose frames are preserved.</>
                )}
              </p>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
            <Slider
              label="Steps"
              min={videoConstraints?.min_inference_steps ?? 1}
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

          <div className="ml-6 mt-1">
            <div className="flex items-center gap-1">
              <Select
                label="Audio mode"
                value={params.inpaint_video_audio_mode || archAudioMode}
                onChange={(e) => setParams({ ...params, inpaint_video_audio_mode: e.target.value as "regenerate" | "preserve_input" | "regenerate_range" })}
                options={[
                  { value: "preserve_input", label: "Preserve the clip's own track" },
                  { value: "regenerate", label: "Regenerate the whole track" },
                  { value: "regenerate_range", label: "Regenerate inside the range, condition on and keep the input's audio elsewhere" },
                ]}
              />
              <InlineHelp label="Audio mode details">
                <p>
                  &quot;Regenerate inside the range, condition on and keep the input&apos;s audio
                  elsewhere&quot; pins the audio outside the regenerate range as conditioning, the
                  same mechanism &quot;Preserve the clip&apos;s own track&quot; uses, so the audio
                  inside the range is generated with the surrounding original track as context.
                  The frames outside the regenerate range are then spliced back from the input
                  clip&apos;s own audio after decode, with a short crossfade at the two boundaries.
                  The pinned/free boundary snaps to the audio latent grid, which is finer than the
                  video latent-group grid, so it can differ from the pixel range by up to one audio
                  latent per side. It falls back to &quot;Regenerate&quot; (with a warning) if the
                  clip has no audio stream or if the regenerate range covers the whole clip, since
                  there is then nothing to pin.
                </p>
              </InlineHelp>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              &quot;Preserve the clip&apos;s own track&quot; conditions the generation on the source
              audio across the whole clip and muxes that track back verbatim; it falls back to
              regenerating (with a warning) if the clip has no audio stream. &quot;Regenerate&quot;
              generates a soundtrack for the whole clip, so the preserved video frames carry generated
              audio that need not match them. With Audio off nothing is muxed either way, and under
              &quot;Preserve&quot; the source track still conditions the generation.
            </p>
          </div>

          <div className="flex items-center gap-2 mt-3">
            <input
              type="checkbox"
              id="inpaint_video_lossless"
              checked={params.video_lossless ?? false}
              onChange={(e) => setParams({ ...params, video_lossless: e.target.checked })}
              className="rounded"
            />
            <label htmlFor="inpaint_video_lossless" className="text-sm text-gray-300">Lossless (FFV1)</label>
          </div>
          <p className="text-xs text-gray-500 ml-6">
            The preserved frames are the input&apos;s own pixels, exact at the frames handoff either
            way. FFV1 carries that exactness into the FILE; the default H.264 re-encodes preserved and
            generated frames alike. FFV1 files are much larger and generally do not play in a
            browser&apos;s native video element.
          </p>

          <VideoAccelerationControls
            idPrefix="inpaint_vid"
            values={params}
            onChange={(patch) => setParams({ ...params, ...patch })}
            supportsSpectrum={supportsSpectrum}
            supportsFbcache={supportsFbcache}
            supportsFuseOutputProj={supportsFuseOutputProj}
            blocksToSwapEnabledDefault={videoBlocksToSwapEnabledDefault}
            blockSwapMax={VIDEO_BLOCK_SWAP_MAX}
            fbcacheLockedReason={
              videoMaskManifest.keyframes.length > 0
                ? "unavailable while the mask timeline has keyframes"
                : undefined
            }
          />
        </Card>
        )}

        {isVideo && supportsTemporalInpaint && visibility.lora && (
          <LoRASelector
            value={params.loras || []}
            onChange={(loras) => setParams({ ...params, loras })}
            disabled={isGenerating}
            storageKey="inpaint_video_lora_collapsed"
            loadedArch={loadedArchType}
            onApplyRecommended={applyLoraRecommended}
          />
        )}

        {!isVideo && visibility.lora && (
          <LoRASelector
            value={params.loras || []}
            onChange={(loras) => setParams({ ...params, loras })}
            disabled={isGenerating}
            storageKey="inpaint_lora_collapsed"
            loadedArch={loadedArchType}
            onApplyRecommended={applyLoraRecommended}
          />
        )}

        {!isVideo && visibility.controlnet && (
          <ControlNetSelector
            value={params.controlnets || []}
            onChange={(controlnets) => setParams({ ...params, controlnets })}
            disabled={isGenerating}
            storageKey="inpaint_controlnet_collapsed"
            inputImagePreview={inputImagePreview}
          />
        )}

        {/* Loop Generation. Image mode only: the video branch enqueues one
            inpaint_vid item, matching the video branches of the other panels. */}
        {!isVideo && (
        <LoopGenerationPanel
          config={loopGenerationConfig}
          onChange={setLoopGenerationConfig}
          mode="inpaint"
          mainWidth={params.width || 1024}
          mainHeight={params.height || 1024}
          samplers={samplers}
          scheduleTypes={scheduleTypes}
        />
        )}
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
                  disabled={isVideo && (!supportsTemporalInpaint || !videoFile)}
                  title={isVideo && !supportsTemporalInpaint ? (temporalInpaintReason || "") : ""}
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
                        disabled={isVideo && (!supportsTemporalInpaint || !videoFile)}
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
              {!isVideo && (
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="preview_predicted_x0_inpaint"
                  checked={params.preview_predicted_x0 || false}
                  onChange={(e) => setParams({ ...params, preview_predicted_x0: e.target.checked })}
                  className="rounded"
                />
                <label htmlFor="preview_predicted_x0_inpaint" className="text-sm text-gray-300">
                  Preview Predicted x0
                </label>
              </div>
              )}

              {/* Live-preview decoder — only meaningful for AutoencoderKLFlux2-latent
                  models (FLUX.2 / Lens / Ideogram 4); hidden for architectures that
                  ignore preview_decoder (SD/SDXL, Z-Image, Anima, MiniT2I). */}
              {!isVideo && (currentModelInfo?.model_info?.type === "flux2"
                || currentModelInfo?.model_info?.type === "lens"
                || currentModelInfo?.model_info?.type === "ideogram4") && (
                <div className="flex items-center gap-2">
                  <label htmlFor="preview_decoder_inpaint" className="text-sm text-gray-300">
                    Preview Decoder
                  </label>
                  <select
                    id="preview_decoder_inpaint"
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

              {/* Use training model toggle (mirrors Txt2Img / Img2Img panels).
                  Image path only: the training-preview route takes an init
                  image and a mask. */}
              {!isVideo && (
              <div className="flex items-center gap-2"
                   title={activeTraining
                     ? `Active: ${activeTraining.run_name ?? `run #${activeTraining.run_id}`} (step ${activeTraining.current_step ?? "?"})`
                     : "No active LoRA/Full-FT training"}>
                <input
                  type="checkbox"
                  id="use_training_model_inpaint"
                  checked={useTrainingModel}
                  disabled={!activeTraining}
                  onChange={(e) => setUseTrainingModel(e.target.checked)}
                  className="rounded disabled:opacity-50"
                />
                <label htmlFor="use_training_model_inpaint"
                       className={`text-sm ${activeTraining ? "text-gray-300" : "text-gray-500"}`}>
                  Use training model
                  {useTrainingModel && activeTraining && (
                    <span className="ml-1 text-xs text-emerald-400">
                      · {activeTraining.run_name ?? `run #${activeTraining.run_id}`} (step {activeTraining.current_step ?? "?"})
                    </span>
                  )}
                </label>
              </div>
              )}

              {!isVideo && useTrainingModel && (
                <div className="flex items-center gap-2 ml-6"
                     title="Save preview PNG to outputs/ and the gallery (tagged as training-preview)">
                  <input
                    type="checkbox"
                    id="save_preview_to_gallery_inpaint"
                    checked={savePreviewToGallery}
                    onChange={(e) => setSavePreviewToGallery(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="save_preview_to_gallery_inpaint" className="text-sm text-gray-300">
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
                  if (!isVideo && generatedImage) {
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
                        console.warn("[Inpaint] Preview video failed to load, clearing:", generatedVideo);
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
                    {generatedVideoWarnings.length > 0 && (
                      <ul className="text-xs text-amber-400 list-disc pl-4 space-y-1">
                        {generatedVideoWarnings.map((w, i) => <li key={i}>{w}</li>)}
                      </ul>
                    )}
                  </div>
                ) : isVideo ? (
                  <p className="text-gray-500">No video generated yet</p>
                ) : generatedImage ? (
                  <img
                    src={effectiveGeneratedImage ?? generatedImage}
                    alt="Generated"
                    className="max-w-full max-h-full rounded-lg"
                    style={{ filter: buildFilterString(postEdit) }}
                    onError={() => {
                      // The file went away while the panel was open -- show an
                      // empty preview rather than a broken image, the backstop
                      // the video/audio players have elsewhere. Confirmed with
                      // a HEAD first, so a hot reload or a backend blip cannot
                      // discard a result that is still on disk (see helper).
                      imagePreviewGone(effectiveGeneratedImage ?? generatedImage, generatedImage).then((gone) => {
                        if (!gone) return;
                        console.warn("[Inpaint] Preview image failed to load, clearing:", generatedImage);
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
              {!isVideo && generatedImage && (
                <div className="mt-3">
                  <PostEditControls value={postEdit} onChange={setPostEdit} />
                </div>
              )}

              {isVideo && generatedVideo && (
                <div className="grid grid-cols-2 gap-2 mt-4">
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
                      num_frames: generatedVideoInfo?.num_frames,
                      frame_rate: generatedVideoInfo?.fps ?? generatedVideoParams?.frame_rate ?? params.frame_rate,
                      seed: generatedVideoSeed ?? generatedVideoParams?.seed ?? params.seed,
                    }}
                  />
                </div>
              )}

              {/* CFG Metrics Graph (Developer Mode) */}
              {developerMode && cfgMetrics.length > 0 && (
                <div className="mt-4">
                  <div className="text-sm text-gray-400 mb-2">CFG Metrics (Developer Mode)</div>
                  <CFGMetricsGraph metrics={cfgMetrics} />
                </div>
              )}

            {!isVideo && generatedImage && (
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
            </div>

            {/* Right: Generation Queue */}
            <div className="w-full">
              <GenerationQueue currentStep={progress} />
            </div>
          </ResizableColumns>
        </Card>
      </div>

      {/* Image Editor Modal (static input-image mask editing only -- the
          video mask editor below is now a separate mount, P4) */}
      {showImageEditor && editingImageUrl && (
        <ImageEditor
          imageUrl={editingImageUrl}
          onSave={handleEditorSave}
          onClose={handleEditorClose}
          onSaveMask={handleEditorSaveMask}
          mode="inpaint"
          initialMaskUrl={maskImage ?? undefined}
        />
      )}

      {/* Video Mask Frame Editor: one ImageEditor instance stays mounted for
          the whole session and navigates frames in place (P4) -- see
          VideoMaskFrameEditor's own header comment. */}
      {videoMaskEditorSession && videoPreviewUrl && (
        <VideoMaskFrameEditor
          videoUrl={videoPreviewUrl}
          trimStartFrames={params.input_trim_start_frames ?? 0}
          frameRate={clipFrameRate}
          minFrame={0}
          maxFrame={Math.max(0, videoTrimmedFrames - 1)}
          canvasWidth={videoMaskCanvasWidth}
          canvasHeight={videoMaskCanvasHeight}
          initialFrame={videoMaskEditorSession.initialFrame}
          keyframes={videoMaskManifest.keyframes}
          assets={videoMaskAssets}
          onSaveFrame={handleVideoMaskFrameSave}
          onClose={handleVideoMaskEditorClose}
        />
      )}

      {/* Floating Gallery */}
      <FloatingGallery images={galleryImages} maxImages={maxGalleryImages} />

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
