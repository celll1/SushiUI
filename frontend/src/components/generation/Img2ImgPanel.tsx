"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import { ChevronLeft, ChevronRight, X, RotateCcw } from "lucide-react";
import Card from "../common/Card";
import Input from "../common/Input";
import Textarea from "../common/Textarea";
import TextareaWithTagSuggestions from "../common/TextareaWithTagSuggestions";
import Button from "../common/Button";
import Slider from "../common/Slider";
import Select from "../common/Select";
import ModelSelector from "../common/ModelSelector";
import VisionEncoderSelector from "../common/VisionEncoderSelector";
import LoRASelector from "../common/LoRASelector";
import ControlNetSelector from "../common/ControlNetSelector";
import ImageEditor from "../common/ImageEditor";
import TIPODialog, { TIPOSettings } from "../common/TIPODialog";
import { fixFloatingPointParams } from "@/utils/numberUtils";
import ImageViewer from "../common/ImageViewer";
import GenerationQueue from "../common/GenerationQueue";
import PromptEditor from "../common/PromptEditor";
import LoopGenerationPanel, { LoopGenerationConfig } from "./LoopGenerationPanel";
import { getSamplers, getScheduleTypes, generateImg2Img, generateImg2ImgTrainingPreview, toBase64, LoRAConfig, ControlNetConfig, generateTIPOPrompt, cancelGeneration, getCurrentModel } from "@/utils/api";
import { useActiveTraining } from "@/hooks/useActiveTraining";
import { wsClient, CFGMetrics } from "@/utils/websocket";
import CFGMetricsGraph from "../common/CFGMetricsGraph";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";
import { sendToPanel, sendImageToImg2Img, sendImageToInpaint } from "@/utils/sendHelpers";
import { useStartup } from "@/contexts/StartupContext";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";

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
  // U-Net Quantization
  unet_quantization?: string | null;
  // Text Encoder Quantization (Z-Image only)
  text_encoder_quantization?: string | null;
  // Attention type
  attention_type?: string;
  // SigLIP2 Vision Encoder path
  vision_encoder_path?: string | null;
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
  text_encoder_quantization: null,
  cpu_text_encoding: false,
  nag_scale: 5.0,
  nag_tau: 3.5,
  nag_alpha: 0.25,
  feeling_lucky: false,
  nag_sigma_end: 3.0,
  nag_negative_prompt: "",
  use_torch_compile: false,
  preview_predicted_x0: false,
  preview_decoder: "matrix",
  attention_type: "normal",
  vision_encoder_path: null,
};

const STORAGE_KEY = "img2img_params";
const LOOP_GENERATION_STORAGE_KEY = "img2img_loop_generation";
const PREVIEW_STORAGE_KEY = "img2img_preview";
const INPUT_IMAGE_STORAGE_KEY = "img2img_input_image";
const REF_IMAGES_STORAGE_KEY = "img2img_ref_images";

interface Img2ImgPanelProps {
  onImageGenerated?: (imageUrl: string) => void;
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint") => void;
}

export default function Img2ImgPanel({ onTabChange, onImageGenerated }: Img2ImgPanelProps = {}) {
  const { modelLoaded, isBackendReady, generationDefaults } = useStartup();
  const [params, setParams] = useState<Img2ImgParams>(DEFAULT_PARAMS);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [generatedImageSeed, setGeneratedImageSeed] = useState<number | null>(null);
  const [generatedImageAncestralSeed, setGeneratedImageAncestralSeed] = useState<number | null>(null);
  const [generatedImageParams, setGeneratedImageParams] = useState<Img2ImgParams | null>(null);
  const [inputImage, setInputImage] = useState<File | null>(null);
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null);
  const [inputImageSize, setInputImageSize] = useState<{ width: number; height: number } | null>(null);
  const [sizeMode, setSizeMode] = useState<"absolute" | "scale">("absolute");
  const [scale, setScale] = useState<number>(1.0);
  const [progress, setProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);
  const [isMounted, setIsMounted] = useState(false);
  const [currentModelInfo, setCurrentModelInfo] = useState<any>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isEditingImage, setIsEditingImage] = useState(false);
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
    steps: []
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

      // Load preview image
      const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
      if (savedPreview) {
        setGeneratedImage(savedPreview);
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
          setLoopGenerationConfig(JSON.parse(savedLoopGen));
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

      // Reload preview image if it's a backend URL
      const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
      if (savedPreview && savedPreview.startsWith('/outputs/')) {
        console.log("[Img2Img] Reloading preview image from backend:", savedPreview);
        // Force reload by adding timestamp
        setGeneratedImage(`${savedPreview}?t=${Date.now()}`);
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
        console.log("[Img2Img] Params changed by user, saving to localStorage:", {
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
      setParams(prev => ({ ...DEFAULT_PARAMS, ...(generationDefaults.img2img as Partial<typeof DEFAULT_PARAMS>) }));
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

    if (!inputImage && !inputImagePreview) {
      alert("Please upload an input image");
      return;
    }

    // Convert image to base64 for queue storage
    let imageBase64: string;
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

    // Create loop group ID if loop generation is enabled
    const loopGroupId = loopGenerationConfig.enabled ? `loop_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` : undefined;

    addToQueue({
      type: "img2img",
      params: {
        ...params,
        prompt: processedPrompt,
        negative_prompt: processedNegativePrompt,
      },
      inputImage: imageBase64,
      prompt: processedPrompt,
      loopGroupId,
      loopStepIndex: loopGroupId ? -1 : undefined,
      isLoopStep: false,
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
        resize_mode: step.resizeMode,
        resampling_method: step.resamplingMethod,
        unet_quantization: mainParams.unet_quantization, // Inherit quantization from main
        cpu_text_encoding: mainParams.cpu_text_encoding, // Inherit CPU text encoding setting
        use_torch_compile: mainParams.use_torch_compile, // Inherit torch.compile setting
        preview_predicted_x0: mainParams.preview_predicted_x0, // Inherit preview mode
        preview_decoder: mainParams.preview_decoder, // Inherit preview decoder
      };

      // Use custom settings or inherit from main
      if (step.useMainSettings) {
        stepParams.steps = mainParams.steps;
        stepParams.cfg_scale = mainParams.cfg_scale;
        stepParams.sampler = mainParams.sampler;
        stepParams.schedule_type = mainParams.schedule_type;
        stepParams.seed = mainParams.seed;
        stepParams.ancestral_seed = mainParams.ancestral_seed;
        // Inherit Advanced CFG from main
        stepParams.cfg_schedule_type = mainParams.cfg_schedule_type;
        stepParams.cfg_schedule_min = mainParams.cfg_schedule_min;
        stepParams.cfg_schedule_max = mainParams.cfg_schedule_max;
        stepParams.cfg_schedule_power = mainParams.cfg_schedule_power;
        stepParams.cfg_rescale_snr_alpha = mainParams.cfg_rescale_snr_alpha;
        stepParams.dynamic_threshold_percentile = mainParams.dynamic_threshold_percentile;
        stepParams.dynamic_threshold_mimic_scale = mainParams.dynamic_threshold_mimic_scale;
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

      stepParams.prompt_chunking_mode = mainParams.prompt_chunking_mode;
      stepParams.max_prompt_chunks = mainParams.max_prompt_chunks;
      stepParams.unet_quantization = mainParams.unet_quantization;
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
      });
    }

    console.log(`[Img2Img] Added ${enabledSteps.length} loop steps to queue with group ID: ${loopGroupId}`);
  }, [loopGenerationConfig, addToQueue, refImages]);

  // Process queue - automatically start next item
  const processQueueRef = useRef<() => Promise<void>>();

  const processQueue = useCallback(async () => {
    console.log("[Img2Img] processQueue called, isGenerating:", isGenerating);
    if (isGenerating) {
      console.log("[Img2Img] Already generating, skipping");
      return;
    }

    const nextItem = startNextInQueue();
    console.log("[Img2Img] Next item from queue:", nextItem);
    if (!nextItem || nextItem.type !== "img2img") {
      console.log("[Img2Img] No img2img items in queue");
      return;
    }

    // Save current image before starting new generation
    const previousImage = generatedImage;

    setIsGenerating(true);
    setProgress(0);
    const denoisingStrength = nextItem.params.denoising_strength || 0.75;
    const actualSteps = Math.ceil((nextItem.params.steps || 20) * denoisingStrength);
    setTotalSteps(actualSteps);
    setPreviewImage(null);
    setGeneratedImage(null);
    setCfgMetrics([]); // Clear previous metrics

    try {
      // For loop steps, use the input image or fall back to previous image
      const inputImageToUse = nextItem.inputImage || previousImage;
      if (!inputImageToUse) {
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

      let imageUrl: string;
      let result: any;
      if (useTrainingModel && activeTraining) {
        // Training-preview branch: encode init image as base64 and route
        // to /generate/img2img/training-preview.  Result is a blob;
        // we wrap it in an object-URL for display (no gallery save).
        const initImageBase64 = await toBase64(inputImageToUse);
        const preview = await generateImg2ImgTrainingPreview({
          ...(paramsWithDevMode as any),
          init_image_base64: initImageBase64,
          denoising_strength: paramsWithDevMode.denoising_strength ?? 0.75,
          run_id: activeTraining.run_id,
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
        result = await generateImg2Img(paramsWithDevMode, inputImageToUse);
        imageUrl = `/outputs/${result.image.filename}`;
      }
      setGeneratedImage(imageUrl);
      setGeneratedImageSeed(result.image.seed);
      setGeneratedImageAncestralSeed(result.image.ancestral_seed || null);
      // Save the params used for this generation (with actual result values)
      setGeneratedImageParams({
        ...nextItem.params,
        seed: result.image.seed,
        ancestral_seed: result.image.ancestral_seed || -1,
        width: result.image.width,
        height: result.image.height,
      });
      setPreviewImage(null);

      if (onImageGenerated) {
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

        // Update input image first
        console.log(`[Img2Img] Updating loop step ${nextLoopStepIndex} with input image:`, imageUrl);
        updateQueueItemByLoop(nextItem.loopGroupId, nextLoopStepIndex, { inputImage: imageUrl });

        // If TIPO was used for base generation, update loop steps with TIPO-generated prompt
        if (nextItem.loopStepIndex === -1 && nextItem.params.use_tipo && result.image.prompt) {
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

      // Reset state first, then complete item
      console.log("[Img2Img] Generation complete, resetting state and completing item");
      setIsGenerating(false);
      setProgress(0);
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

      if (isCancelled) {
        const shouldRestore = localStorage.getItem('restore_image_on_cancel') === 'true';
        if (shouldRestore && previousImage) {
          setGeneratedImage(previousImage);
          setPreviewImage(null);
        }
      } else {
        alert("Generation failed. Please check console for details.");
      }

      // Reset state first, then fail item
      console.log("[Img2Img] Generation failed, resetting state and failing item");
      setIsGenerating(false);
      setProgress(0);
      failCurrentItem();

      // Wait briefly for state to propagate, then trigger next
      setTimeout(() => {
        console.log("[Img2Img] Triggering next queue item after failure");
        if (processQueueRef.current) {
          processQueueRef.current();
        }
      }, 100);
    }
  }, [isGenerating, generatedImage, onImageGenerated, startNextInQueue, completeCurrentItem, failCurrentItem, updateQueueItem, queue]);

  processQueueRef.current = processQueue;

  // Auto-start queue processing when queue has pending items and not currently generating
  useEffect(() => {
    const hasPendingItems = queue.some(item => item.status === "pending" && item.type === "img2img");
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

    if (hasPendingItems && isCurrentItemNull && !isGenerating) {
      console.log("[Img2Img] Auto-starting queue processing");
      processQueue();
    }
  }, [queue, currentItem, isGenerating, processQueue, generateForever, params, inputImage, inputImagePreview]);

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
  }, [params, inputImage, inputImagePreview]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Parameters Panel */}
      <div className="space-y-4">
        <ModelSelector onModelLoad={async () => {
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
        }} />

        <Card
          title="Input Image"
          collapsible={true}
          defaultCollapsed={true}
          storageKey="img2img_input_collapsed"
          collapsedPreview={
            inputImagePreview ? (
              <span className="text-green-400 text-sm">✓ Image loaded</span>
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
                className="flex-1 block w-full text-sm text-gray-400
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-medium
                  file:bg-blue-600 file:text-white
                  hover:file:bg-blue-700
                  file:cursor-pointer cursor-pointer"
              />
              {inputImagePreview && (
                <Button
                  onClick={handleClearInputImage}
                  variant="secondary"
                  size="sm"
                  title="Clear input image"
                >
                  Clear
                </Button>
              )}
            </div>
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onDoubleClick={handleEditImage}
              className={`aspect-square bg-gray-800 rounded-lg overflow-hidden border-2 border-dashed transition-colors ${
                isDragging
                  ? 'border-blue-500 bg-gray-700'
                  : 'border-gray-600'
              } ${inputImagePreview ? 'cursor-pointer' : ''}`}
              title={inputImagePreview ? "Double-click to edit image" : ""}
            >
              {inputImagePreview ? (
                <img
                  src={inputImagePreview}
                  alt="Input"
                  className="w-full h-full object-contain"
                />
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
                💡 Double-click the image to edit with built-in paint tool
              </p>
            )}
          </div>
        </Card>

        {/* SigLIP2 Vision Encoder (SDXL/SD1.5 reference image conditioning) */}
        {currentModelInfo?.model_info?.type !== "flux2" && (
          <VisionEncoderSelector
            value={params.vision_encoder_path ?? null}
            onChange={(path) => setParams({ ...params, vision_encoder_path: path })}
          />
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
                  <div className="grid grid-cols-5 auto-rows-fr gap-2">
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

        <Card title="Parameters">
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

              {/* Advanced CFG Settings */}
              {showAdvancedCFG && (
                <>
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
              </>
              )}
            </div>

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
              </>
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
                  id="preview_predicted_x0_img2img"
                  checked={params.preview_predicted_x0 || false}
                  onChange={(e) => setParams({ ...params, preview_predicted_x0: e.target.checked })}
                  className="rounded"
                />
                <label htmlFor="preview_predicted_x0_img2img" className="text-sm text-gray-300">
                  Preview Predicted x0
                </label>
              </div>

              {/* Live-preview decoder (FLUX.2 / Lens / Ideogram 4 — AutoencoderKLFlux2 latent) */}
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
                    <span>Generating...</span>
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
                    src={generatedImage}
                    alt="Generated"
                    className="max-w-full max-h-full rounded-lg"
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
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
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

      {/* Image Editor Overlay */}
      {isEditingImage && inputImagePreview && (
        <ImageEditor
          imageUrl={inputImagePreview}
          onSave={handleSaveEditedImage}
          onClose={() => setIsEditingImage(false)}
        />
      )}


      {/* Preview Image Viewer */}
      {previewViewerOpen && generatedImage && (
        <ImageViewer
          imageUrl={generatedImage}
          onClose={() => setPreviewViewerOpen(false)}
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
