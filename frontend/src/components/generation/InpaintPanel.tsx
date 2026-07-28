"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import { ChevronLeft, ChevronRight, X, RotateCcw } from "lucide-react";
import Card from "../common/Card";
import TabbedOptions from "../common/TabbedOptions";
import Input from "../common/Input";
import NumberInput from "../common/NumberInput";
import Textarea from "../common/Textarea";
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
import { PostEditState, NEUTRAL_POST_EDIT, buildFilterString } from "@/utils/postEdit";
import { usePostEditPreview } from "@/hooks/usePostEditPreview";
import GenerationQueue from "../common/GenerationQueue";
import LoopGenerationPanel, { LoopGenerationConfig } from "./LoopGenerationPanel";
import { migrateLoopGenerationConfig, computeLoopDecodeDirective } from "@/utils/loopGenerationInheritance";
import { getSamplers, getScheduleTypes, generateInpaint, generateInpaintTrainingPreview, toBase64, InpaintParams as ApiInpaintParams, LoRAConfig, ControlNetConfig, generateTIPOPrompt, cancelGeneration, getCurrentModel, getResultFilename, getResultSeed, getResultAncestralSeed, isLatentOnlyResult } from "@/utils/api";
import { useActiveTraining } from "@/hooks/useActiveTraining";
import { wsClient, CFGMetrics } from "@/utils/websocket";
import CFGMetricsGraph from "../common/CFGMetricsGraph";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";
import { sendToPanel, sendImageToImg2Img, sendImageToUpscale, sendImageToOutpaint } from "@/utils/sendHelpers";
import { fixFloatingPointParams } from "@/utils/numberUtils";
import { useStartup } from "@/contexts/StartupContext";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";

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
}

const DEFAULT_PARAMS: InpaintParams = {
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
};

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
    "text_encoder_quantization",
    "cpu_text_encoding",
    "vae_tiling",
    "vae_tile_threshold",
    "vae_tile_mode",
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
const PREVIEW_STORAGE_KEY = "inpaint_preview";
const LOOP_GENERATION_STORAGE_KEY = "inpaint_loop_generation";
const INPUT_IMAGE_STORAGE_KEY = "inpaint_input_image";
const MASK_IMAGE_STORAGE_KEY = "inpaint_mask_image";
const REF_IMAGES_STORAGE_KEY = "inpaint_ref_images";

interface InpaintPanelProps {
  onImageGenerated?: (imageUrl: string) => void;
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint" | "outpaint" | "upscale") => void;
}

export default function InpaintPanel({ onTabChange, onImageGenerated }: InpaintPanelProps = {}) {
  const { modelLoaded, isBackendReady, generationDefaults } = useStartup();
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
  const [progress, setProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  // Streamed progress-phase label (e.g. "Step 12/28" or "PiD decode (tile 3/9)").
  // Rendered in place of the hardcoded "Generating..." text so decode-phase
  // status is visible; reset alongside every setProgress(0) site.
  const [progressMessage, setProgressMessage] = useState("");
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);
  const [isMounted, setIsMounted] = useState(false);
  const [currentModelInfo, setCurrentModelInfo] = useState<any>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showImageEditor, setShowImageEditor] = useState(false);
  const [editingImageUrl, setEditingImageUrl] = useState<string | null>(null);
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
  const handleProgress = useCallback((step: number, totalSteps: number, message: string, preview?: string, metrics?: CFGMetrics) => {
    if (isGeneratingRef.current) {
      setProgress(step);
      setTotalSteps(totalSteps);
      setProgressMessage(message || "");
      if (preview) {
        setPreviewImage(preview);
      }
      if (metrics && developerModeRef.current) {
        setCfgMetrics(prev => [...prev, metrics]);
      }
    }
  }, []); // Empty deps - stable callback

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
    console.log("=== InpaintPanel mounted ===");
    setIsMounted(true);

    const loadInitialData = async () => {
      // Load current model info
      try {
        const modelInfo = await getCurrentModel();
        setCurrentModelInfo(modelInfo);
        console.log("[Inpaint] Current model info:", modelInfo);
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

      // Load preview image
      const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
      if (savedPreview) {
        setGeneratedImage(savedPreview);
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
      const savedAttentionType = localStorage.getItem('attention_type');
      if (savedAttentionType && (savedAttentionType === 'normal' || savedAttentionType === 'sage' || savedAttentionType === 'flash')) {
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

          const loadedPreviews: string[] = [];
          for (const ref of refRefs) {
            try {
              const imageData = await loadTempImage(ref);
              if (imageData) {
                loadedPreviews.push(imageData);
              }
            } catch (error) {
              console.error(`[Inpaint] Failed to load reference image ${ref}:`, error);
            }
          }

          if (loadedPreviews.length > 0) {
            setRefImagePreviews(loadedPreviews);
            console.log(`[Inpaint] Restored ${loadedPreviews.length} reference images`);
          }
        } catch (error) {
          console.error('[Inpaint] Failed to parse reference images storage:', error);
        }
      }

      // Mark initial load as complete
      setIsInitialLoad(false);
      console.log("[Inpaint] Initial load complete");
    };

    loadInitialData();
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
      const reloadImages = async () => {
        console.log("[Inpaint] Backend ready, reloading images if needed");

        // Reload preview image if it's a backend URL
        const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
        if (savedPreview && savedPreview.startsWith('/outputs/')) {
          console.log("[Inpaint] Reloading preview image from backend:", savedPreview);
          // Force reload by adding timestamp
          setGeneratedImage(`${savedPreview}?t=${Date.now()}`);
        }

        // Reload input image if not loaded
        if (!inputImagePreview) {
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
              if (imageData) {
                setMaskImage(imageData);
              }
            } catch (error) {
              console.error("[Inpaint] Failed to reload mask image after backend ready:", error);
            }
          }
        }
      };

      reloadImages();
    }
  }, [isBackendReady]);

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
      localStorage.setItem(PREVIEW_STORAGE_KEY, generatedImage);
    }
  }, [generatedImage, isMounted]);

  // Save loop generation config to localStorage whenever it changes
  useEffect(() => {
    if (isMounted) {
      localStorage.setItem(LOOP_GENERATION_STORAGE_KEY, JSON.stringify(loopGenerationConfig));
    }
  }, [loopGenerationConfig, isMounted]);

  // Apply backend-fetched defaults when they arrive (only if no localStorage value exists)
  useEffect(() => {
    if (!generationDefaults) return;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      setParams(prev => ({ ...DEFAULT_PARAMS, ...(generationDefaults.inpaint as Partial<typeof DEFAULT_PARAMS>) }));
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

  const { addToQueue, updateQueueItem, updateQueueItemByLoop, cancelLoopGroup, startNextInQueue, completeCurrentItem, failCurrentItem, currentItem, queue, generateForever, setGenerateForever } = useGenerationQueue();
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
    const processedNegativePrompt = await replaceWildcardsInPrompt(params.negative_prompt);

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
        original_size_w: mainParams.original_size_w,
        original_size_h: mainParams.original_size_h,
        original_size_scale: mainParams.original_size_scale,
        cpu_text_encoding: mainParams.cpu_text_encoding, // Inherit CPU text encoding setting
        use_torch_compile: mainParams.use_torch_compile, // Inherit torch.compile setting
        keep_models_hot: mainParams.keep_models_hot, // Inherited default; queue dispatch overrides based on hasNext
        vae_tiling: mainParams.vae_tiling, // Inherit VAE tiling setting
        vae_tile_threshold: mainParams.vae_tile_threshold, // Inherit VAE tile threshold
        vae_tile_mode: mainParams.vae_tile_mode, // Inherit VAE tile join mode
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
    console.log("[Inpaint] processQueue called, isGenerating:", isGenerating);
    if (isGenerating) {
      console.log("[Inpaint] Already generating, skipping");
      return;
    }

    const nextItem = startNextInQueue();
    console.log("[Inpaint] Next item from queue:", nextItem);
    if (!nextItem || nextItem.type !== "inpaint") {
      console.log("[Inpaint] No inpaint items in queue");
      return;
    }

    // Save current image before starting new generation
    const previousImage = generatedImage;

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
        setGeneratedImageParams({
          ...nextItem.params,
          seed: resultSeed,
          ancestral_seed: resultAncestralSeed ?? -1,
          width: result.image?.width ?? nextItem.params.width,
          height: result.image?.height ?? nextItem.params.height,
        });

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
          onImageGenerated(imageUrl);
        }

        if (isMounted) {
          localStorage.setItem(PREVIEW_STORAGE_KEY, imageUrl);
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
        alert("Generation failed");
        // Reset state first, then fail item
        setIsGenerating(false);
        setProgress(0);
        setProgressMessage("");
        failCurrentItem();

        setTimeout(() => {
          if (processQueueRef.current) {
            processQueueRef.current();
          }
        }, 100);
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

      if (isCancelled) {
        const shouldRestore = localStorage.getItem('restore_image_on_cancel') === 'true';
        if (shouldRestore && previousImage) {
          setGeneratedImage(previousImage);
          setPreviewImage(null);
        }
      } else {
        alert("Generation failed: " + (error instanceof Error ? error.message : String(error)));
      }

      // Reset state first, then fail item
      console.log("[Inpaint] Generation failed, resetting state and failing item");
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
    }
  }, [isGenerating, generatedImage, onImageGenerated, isMounted, startNextInQueue, completeCurrentItem, failCurrentItem, updateQueueItem, queue]);

  processQueueRef.current = processQueue;

  // Auto-start queue processing when queue has pending items and not currently generating
  useEffect(() => {
    const hasPendingItems = queue.some(item => item.status === "pending" && item.type === "inpaint");
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

    // If generate forever is enabled and queue is empty, add new item
    if (generateForever && !hasPendingItems && isCurrentItemNull && !isGenerating && params.prompt && inputImagePreview && maskImage) {
      console.log("[Inpaint] Generate forever: Adding new item to queue");
      handleAddToQueue();
      return;
    }

    if (hasPendingItems && isCurrentItemNull && !isGenerating) {
      console.log("[Inpaint] Auto-starting queue processing");
      processQueue();
    }
  }, [queue, currentItem, isGenerating, processQueue, generateForever, params, inputImagePreview, maskImage]);

  // Handle Ctrl+Enter keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if Image Editor is open (global check for all Image Editors)
      if (document.body.dataset.imageEditorOpen) return;

      if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        handleAddToQueue();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [params, inputImage, inputImagePreview, maskImage]);

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
                options={[
                  { value: "none", label: "None" },
                  { value: "fp8_e4m3fn", label: "FP8 E4M3 (Recommended)" },
                  { value: "fp8_e5m2", label: "FP8 E5M2" },
                ]}
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
              <div className="bg-blue-900/20 border border-blue-600/30 rounded-lg p-3">
                <p className="text-xs text-blue-200">
                  💡 {currentModelInfo?.model_info?.type === "flux2" ? "FLUX.2" : "Z-Image"} quantization can reduce VRAM significantly. Text encoder ({currentModelInfo?.model_info?.type === "flux2" ? "Qwen3" : "Gemma2 3.4B"}) is particularly large.
                </p>
              </div>
            ) : null}
          </>
        ) : (
          <>
            {/* SD/SDXL: 1-column layout */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Select
                label="U-Net Quantization"
                value={params.unet_quantization || "none"}
                onChange={(e) => setParams({
                  ...params,
                  unet_quantization: e.target.value === "none" ? null : e.target.value
                })}
                options={[
                  { value: "none", label: "None" },
                  { value: "fp8_e4m3fn", label: "FP8 E4M3 (Recommended)" },
                  { value: "fp8_e5m2", label: "FP8 E5M2" },
                ]}
              />
            </div>
            {params.unet_quantization && params.unet_quantization !== "none" && (
              <div className="bg-yellow-900/20 border border-yellow-600/30 rounded-lg p-3">
                <p className="text-xs text-yellow-200">
                  ⚠️ Quantization reduces VRAM but may affect quality. Original model kept on CPU.
                </p>
              </div>
            )}
          </>
        )}

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

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Parameters Panel */}
      <div className="space-y-4">
        <ModelLoadSection
          onModelLoad={async () => {
            // Reload model info when model changes
            const modelInfo = await getCurrentModel();
            setCurrentModelInfo(modelInfo);
            console.log("[Inpaint] Model changed, updated currentModelInfo:", modelInfo);

            // Auto-adjust sampler/schedule for Flow Matching models (Z-Image, FLUX.2)
            const modelType = modelInfo?.model_info?.type;
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

        <Card
          title="Input Image"
          collapsible={true}
          defaultCollapsed={true}
          storageKey="inpaint_input_collapsed"
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
          <div className="space-y-4">
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
              className={`aspect-square bg-gray-800 rounded-lg overflow-hidden border-2 border-dashed transition-colors relative ${
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
                        mixBlendMode: 'screen',
                        opacity: 0.5
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


        {/* FLUX.2 Image Edit / Vision Encoder: Reference Images */}
        {(currentModelInfo?.model_info?.type === "flux2" || params.vision_encoder_path) && (
          <Card
            title={currentModelInfo?.model_info?.type === "flux2" ? "FLUX.2 Image Edit (Reference Images)" : "Vision Encoder (Reference Images)"}
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

        <Card title="Prompt">
          <div className="relative">
            <TextareaWithTagSuggestions
              label="Positive Prompt"
              placeholder="Enter your prompt here..."
              rows={4}
              value={params.prompt}
              onChange={(e) => {
                setParams({ ...params, prompt: e.target.value });
                if (e.target) {
                  promptTextareaRef.current = e.target as HTMLTextAreaElement;
                }
              }}
              enableWeightControl={true}
            />
          </div>

          {/* Feeling Lucky Mode */}
          <div className="flex items-center gap-2 px-2 py-2 bg-gray-800 rounded">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={params.feeling_lucky || false}
                onChange={(e) => setParams({ ...params, feeling_lucky: e.target.checked })}
                className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-2 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-300">✨ Feeling Lucky (TIPO)</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer ml-4">
              <input
                type="checkbox"
                checked={treatAsNL}
                onChange={(e) => setTreatAsNL(e.target.checked)}
                className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-green-500 focus:ring-2 focus:ring-green-500"
                title="Treat input as natural language instead of tags"
              />
              <span className="text-xs text-gray-400">NL</span>
            </label>
            <button
              onClick={() => setIsTIPODialogOpen(true)}
              className="ml-auto px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
              title="Configure TIPO settings"
            >
              ⚙️ Settings
            </button>
          </div>

          <TextareaWithTagSuggestions
            label="Negative Prompt"
            placeholder="Enter negative prompt..."
            rows={3}
            value={params.negative_prompt}
            onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
            enableWeightControl={true}
          />
        </Card>

        {/* Inpaint Options: a single-open tabbed accordion (chrome shared via
            frontend/src/components/common/TabbedOptions.tsx). Every control
            below is unchanged from its original location (same param
            binding / handler / conditional reveal) -- only the container
            changed. See INPAINT_OPTIONS_TAB_KEYS / isInpaintOptionsTabActive /
            inpaintOptionsTabRender above. */}
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

        {visibility.lora && (
          <LoRASelector
            value={params.loras || []}
            onChange={(loras) => setParams({ ...params, loras })}
            disabled={isGenerating}
            storageKey="inpaint_lora_collapsed"
          />
        )}

        {visibility.controlnet && (
          <ControlNetSelector
            value={params.controlnets || []}
            onChange={(controlnets) => setParams({ ...params, controlnets })}
            disabled={isGenerating}
            storageKey="inpaint_controlnet_collapsed"
            inputImagePreview={inputImagePreview}
          />
        )}

        {/* Loop Generation */}
        <LoopGenerationPanel
          config={loopGenerationConfig}
          onChange={setLoopGenerationConfig}
          mode="inpaint"
          mainWidth={params.width || 1024}
          mainHeight={params.height || 1024}
          samplers={samplers}
          scheduleTypes={scheduleTypes}
        />
      </div>

      {/* Preview Panel */}
      <div className="pb-16 lg:pb-0">
        <Card title="Preview">
          <div className="flex flex-col lg:flex-row gap-2 lg:h-[800px]">
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
                  id="preview_predicted_x0_inpaint"
                  checked={params.preview_predicted_x0 || false}
                  onChange={(e) => setParams({ ...params, preview_predicted_x0: e.target.checked })}
                  className="rounded"
                />
                <label htmlFor="preview_predicted_x0_inpaint" className="text-sm text-gray-300">
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

              {/* Use training model toggle (mirrors Txt2Img / Img2Img panels) */}
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

              {useTrainingModel && (
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
                      style={{ width: `${(progress / totalSteps) * 100}%` }}
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
                {generatedImage ? (
                  <img
                    src={effectiveGeneratedImage ?? generatedImage}
                    alt="Generated"
                    className="max-w-full max-h-full rounded-lg"
                    style={{ filter: buildFilterString(postEdit) }}
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
                <div className="grid grid-cols-1 sm:grid-cols-5 gap-2">
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
                </div>
              </div>
            )}
            </div>

            {/* Right: Generation Queue */}
            <div className="w-full lg:w-60 lg:flex-shrink-0">
              <GenerationQueue currentStep={progress} />
            </div>
          </div>
        </Card>
      </div>

      {/* Image Editor Modal */}
      {showImageEditor && editingImageUrl && (
        <ImageEditor
          imageUrl={editingImageUrl}
          onSave={handleEditorSave}
          onClose={handleEditorClose}
          onSaveMask={handleEditorSaveMask}
          mode="inpaint"
          initialMaskUrl={maskImage || undefined}
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
    </div>
  );
}
