"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { ChevronLeft, ChevronRight, X, RotateCcw } from "lucide-react";
import Card from "../common/Card";
import Input from "../common/Input";
import TextareaWithTagSuggestions from "../common/TextareaWithTagSuggestions";
import Button from "../common/Button";
import Slider from "../common/Slider";
import Select from "../common/Select";
import ModelSelector from "../common/ModelSelector";
import LoRASelector from "../common/LoRASelector";
import ControlNetSelector from "../common/ControlNetSelector";
import TIPODialog, { TIPOSettings } from "../common/TIPODialog";
import { fixFloatingPointParams } from "@/utils/numberUtils";
import ImageViewer from "../common/ImageViewer";
import GenerationQueue from "../common/GenerationQueue";
import PromptEditor from "../common/PromptEditor";
import LoopGenerationPanel, { LoopGenerationConfig } from "./LoopGenerationPanel";
import { generateTxt2Img, generateImg2Img, GenerationParams, getSamplers, getScheduleTypes, tokenizePrompt, generateTIPOPrompt, cancelGeneration } from "@/utils/api";
import { wsClient, CFGMetrics } from "@/utils/websocket";
import CFGMetricsGraph from "../common/CFGMetricsGraph";
import { saveTempImage, loadTempImage } from "@/utils/tempImageStorage";
import { sendPromptToPanel, sendParametersToPanel, sendImageToImg2Img, sendBase64ImageToInpaint } from "@/utils/sendHelpers";
import { useStartup } from "@/contexts/StartupContext";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";

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
  use_torch_compile: false,
  feeling_lucky: false,
};

const STORAGE_KEY = "txt2img_params";
const PREVIEW_STORAGE_KEY = "txt2img_preview";
const LOOP_GENERATION_STORAGE_KEY = "txt2img_loop_generation";

interface Txt2ImgPanelProps {
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint") => void;
  onImageGenerated?: (imageUrl: string) => void;
}

export default function Txt2ImgPanel({ onTabChange, onImageGenerated }: Txt2ImgPanelProps = {}) {
  const { modelLoaded } = useStartup();
  const [params, setParams] = useState<GenerationParams>(DEFAULT_PARAMS);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [generatedImageSeed, setGeneratedImageSeed] = useState<number | null>(null);
  const [generatedImageAncestralSeed, setGeneratedImageAncestralSeed] = useState<number | null>(null);
  const [generatedImageParams, setGeneratedImageParams] = useState<GenerationParams | null>(null);
  const [progress, setProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);
  const [isMounted, setIsMounted] = useState(false);
  const [sendImage, setSendImage] = useState(true);
  const [sendPrompt, setSendPrompt] = useState(true);
  const [sendParameters, setSendParameters] = useState(true);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
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
  const [isPromptEditorOpen, setIsPromptEditorOpen] = useState(false);
  const [loopGenerationConfig, setLoopGenerationConfig] = useState<LoopGenerationConfig>({
    enabled: false,
    steps: []
  });
  const [isMobileControlsOpen, setIsMobileControlsOpen] = useState(true);
  const [cfgMetrics, setCfgMetrics] = useState<CFGMetrics[]>([]);
  const [developerMode, setDeveloperMode] = useState(false);

  const tokenizePromptTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const tokenizeNegativeTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const promptTextareaRef = useRef<HTMLTextAreaElement | null>(null);

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
    console.clear();
    console.log("=== Txt2ImgPanel mounted ===");
    setIsMounted(true);

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

    // Load preview image
    const savedPreview = localStorage.getItem(PREVIEW_STORAGE_KEY);
    if (savedPreview) {
      setGeneratedImage(savedPreview);
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
        setLoopGenerationConfig(JSON.parse(savedLoopGen));
      } catch (e) {
        console.error('Failed to parse loop generation config:', e);
      }
    }

  }, []);

  // Reset torch.compile when developer mode is disabled
  useEffect(() => {
    if (!developerMode && params.use_torch_compile) {
      setParams({ ...params, use_torch_compile: false });
    }
  }, [developerMode]);

  // Load samplers and schedule types when model is loaded
  useEffect(() => {
    if (modelLoaded) {
      loadSamplers();
      loadScheduleTypes();
    }
  }, [modelLoaded]);

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
      localStorage.setItem(PREVIEW_STORAGE_KEY, generatedImage);
    }
  }, [generatedImage, isMounted]);

  // Save loop generation config to localStorage whenever it changes
  useEffect(() => {
    if (isMounted) {
      localStorage.setItem(LOOP_GENERATION_STORAGE_KEY, JSON.stringify(loopGenerationConfig));
    }
  }, [loopGenerationConfig, isMounted]);

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

    // Send prompt if checked
    if (sendPrompt) {
      sendPromptToPanel(sourceParams, "img2img_params");
    }

    // Send parameters if checked
    if (sendParameters) {
      sendParametersToPanel(sourceParams, "img2img_params");
    }

    // Navigate to img2img tab
    if (onTabChange) {
      onTabChange("img2img");
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

    // Send prompt if checked
    if (sendPrompt) {
      sendPromptToPanel(sourceParams, "inpaint_params");
    }

    // Send parameters if checked
    if (sendParameters) {
      sendParametersToPanel(sourceParams, "inpaint_params");
    }

    // Navigate to inpaint tab
    if (onTabChange) {
      onTabChange("inpaint");
    }
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
    }
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

  // WebSocket progress callback
  const handleProgress = useCallback((step: number, totalSteps: number, message: string, preview?: string, metrics?: CFGMetrics) => {
    if (isGenerating) {
      setProgress(step);
      setTotalSteps(totalSteps);
      if (preview) {
        setPreviewImage(preview);
      }
      if (metrics && developerMode) {
        setCfgMetrics(prev => [...prev, metrics]);
      }
    }
  }, [isGenerating, developerMode]);

  // Setup WebSocket connection
  useEffect(() => {
    wsClient.connect();
    wsClient.subscribe(handleProgress);

    return () => {
      wsClient.unsubscribe(handleProgress);
    };
  }, [handleProgress]);

  const [showForeverMenu, setShowForeverMenu] = useState(false);
  const [menuPosition, setMenuPosition] = useState({ x: 0, y: 0 });
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

  // Open prompt editor
  const handleOpenPromptEditor = () => {
    setIsPromptEditorOpen(true);
  };

  // Save prompts from editor
  const handleSavePrompts = (prompt: string, negativePrompt: string) => {
    setParams({ ...params, prompt, negative_prompt: negativePrompt });
  };

  // Add generation request to queue
  const handleAddToQueue = async () => {
    if (!params.prompt) {
      alert("Please enter a prompt");
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
        // Load shared TIPO settings from localStorage (same as Prompt Editor)
        const saved = localStorage.getItem("tipo_settings");
        const sharedTipoSettings = saved ? JSON.parse(saved) : tipoSettings;

        // Build category order and enabled map from settings
        const categoryOrder = sharedTipoSettings.categories.map((c: any) => c.id);
        const enabledCategories: Record<string, boolean> = {};
        sharedTipoSettings.categories.forEach((c: any) => {
          enabledCategories[c.id] = c.enabled;
        });

        console.log('[Txt2Img] Feeling Lucky: Generating prompt with TIPO...');
        const result = await generateTIPOPrompt({
          input_prompt: processedPrompt,
          model_name: sharedTipoSettings.model_name,
          tag_length: sharedTipoSettings.tag_length,
          nl_length: sharedTipoSettings.nl_length,
          temperature: sharedTipoSettings.temperature,
          top_p: sharedTipoSettings.top_p,
          top_k: sharedTipoSettings.top_k,
          max_new_tokens: sharedTipoSettings.max_new_tokens,
          category_order: categoryOrder,
          enabled_categories: enabledCategories
        });

        processedPrompt = result.generated_prompt;
        console.log('[Txt2Img] Feeling Lucky: Generated prompt:', processedPrompt.substring(0, 100) + '...');
      } catch (error) {
        console.error("TIPO generation failed in Feeling Lucky mode:", error);
        alert("Failed to generate prompt with TIPO. Using original prompt.");
      }
    }

    // Create loop group ID if loop generation is enabled
    const loopGroupId = loopGenerationConfig.enabled ? `loop_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` : undefined;

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
      },
      prompt: processedPrompt,
      loopGroupId,
      loopStepIndex: loopGroupId ? -1 : undefined, // -1 indicates main generation
      isLoopStep: false,
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
        use_torch_compile: mainParams.use_torch_compile, // Inherit torch.compile setting
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
      });
    }

    console.log(`[Txt2Img] Added ${enabledSteps.length} loop steps to queue with group ID: ${loopGroupId}`);
  }, [loopGenerationConfig, addToQueue]);

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
      });
    }

    console.log(`[Txt2Img] Added ${enabledSteps.length} loop steps to queue with group ID: ${loopGroupId}`);
  }, [loopGenerationConfig, addToQueue]);

  // Process queue - automatically start next item
  const processQueueRef = useRef<() => Promise<void>>();

  const processQueue = useCallback(async () => {
    console.log("[Txt2Img] processQueue called, isGenerating:", isGenerating);
    if (isGenerating) {
      console.log("[Txt2Img] Already generating, skipping");
      return;
    }

    const nextItem = startNextInQueue();
    console.log("[Txt2Img] Next item from queue:", nextItem);
    if (!nextItem) {
      console.log("[Txt2Img] No items in queue");
      return;
    }

    // Save current image before starting new generation
    const previousImage = generatedImage;

    setIsGenerating(true);
    setProgress(0);
    setTotalSteps(nextItem.params.steps || 20);
    setPreviewImage(null);
    setGeneratedImage(null);
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
        result = await generateTxt2Img(paramsWithDevMode as GenerationParams);
        imageUrl = `/outputs/${result.image.filename}`;
      } else if (nextItem.type === "img2img") {
        // For loop steps after the first, use the previous output as input
        const inputImageToUse = nextItem.inputImage || previousImage;
        if (!inputImageToUse) {
          throw new Error("No input image available for img2img loop step");
        }

        // Fetch the input image and convert to File
        const response = await fetch(inputImageToUse);
        const blob = await response.blob();
        const file = new File([blob], "input.png", { type: "image/png" });

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
        result = await generateImg2Img(paramsWithDevMode, file);
        imageUrl = `/outputs/${result.image.filename}`;
      } else {
        throw new Error(`Unsupported generation type: ${nextItem.type}`);
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

      // Notify parent component
      if (onImageGenerated) {
        onImageGenerated(imageUrl);
      }

      // If this item has a loop group, update the next loop step's input image and ControlNets
      // Use currentItem instead of nextItem to respect cancellation
      if (currentItem?.loopGroupId !== undefined) {
        const nextLoopStepIndex = (currentItem.loopStepIndex ?? -1) + 1;

        console.log(`[Txt2Img] Processing loop step completion:`, {
          loopGroupId: currentItem.loopGroupId,
          currentStepIndex: currentItem.loopStepIndex,
          nextLoopStepIndex,
        });

        // Update input image first
        console.log(`[Txt2Img] Updating loop step ${nextLoopStepIndex} with input image:`, imageUrl);
        updateQueueItemByLoop(currentItem.loopGroupId, nextLoopStepIndex, { inputImage: imageUrl });

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

              updateQueueItemByLoop(currentItem.loopGroupId!, nextLoopStepIndex, (item) => ({
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
          updateQueueItemByLoop(currentItem.loopGroupId!, nextLoopStepIndex, (item) => {
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

      // Reset state first, then complete item
      console.log("[Txt2Img] Generation complete, resetting state and completing item");
      setIsGenerating(false);
      setProgress(0);
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
      console.log("[Txt2Img] Generation failed, resetting state and failing item");
      setIsGenerating(false);
      setProgress(0);
      failCurrentItem();

      // Wait briefly for state to propagate, then trigger next
      setTimeout(() => {
        console.log("[Txt2Img] Triggering next queue item after failure");
        if (processQueueRef.current) {
          processQueueRef.current();
        }
      }, 100);
    }
  }, [isGenerating, generatedImage, onImageGenerated, startNextInQueue, completeCurrentItem, failCurrentItem, updateQueueItem, queue]);

  processQueueRef.current = processQueue;

  // Auto-start queue processing when queue has pending items and not currently generating
  useEffect(() => {
    const hasPendingItems = queue.some(item => item.status === "pending");
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

    if (hasPendingItems && isCurrentItemNull && !isGenerating) {
      console.log("[Txt2Img] Auto-starting queue processing");
      processQueue();
    }
  }, [queue, currentItem, isGenerating, processQueue, generateForever, params]);

  // Handle Ctrl+Enter keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if Prompt Editor or Image Editor is open
      if (isPromptEditorOpen || document.body.dataset.imageEditorOpen) return;

      if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        handleAddToQueue();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [params, isPromptEditorOpen]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Parameters Panel */}
      <div className="space-y-4">
        <ModelSelector />

        <Card title="Prompt">
          <div className="relative">
            <TextareaWithTagSuggestions
              label="Positive Prompt"
              placeholder="Enter your prompt here..."
              rows={4}
              value={params.prompt}
              onChange={(e) => {
                setParams({ ...params, prompt: e.target.value });
                // Capture textarea ref from the event
                if (e.target) {
                  promptTextareaRef.current = e.target as HTMLTextAreaElement;
                }
              }}
              onDoubleClick={handleOpenPromptEditor}
              enableWeightControl={true}
            />
            <div className="absolute -top-1 right-0 flex items-center gap-1 px-2 py-1">
              <span className="text-xs text-gray-400 pointer-events-none">
                {promptTokenCount} tokens
              </span>
            </div>
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
            <button
              onClick={() => setIsTIPODialogOpen(true)}
              className="ml-auto px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
              title="Configure TIPO settings"
            >
              ⚙️ Settings
            </button>
          </div>

          <div className="relative">
            <TextareaWithTagSuggestions
              label="Negative Prompt"
              placeholder="Enter negative prompt..."
              rows={3}
              value={params.negative_prompt}
              onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
              onDoubleClick={handleOpenPromptEditor}
              enableWeightControl={true}
            />
            <div className="absolute top-0 right-0 text-xs text-gray-400 px-2 py-1 pointer-events-none">
              {negativePromptTokenCount} tokens
            </div>
          </div>
        </Card>

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
                min={1}
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
                  onClick={() => generatedImageAncestralSeed !== null && setParams({ ...params, ancestral_seed: generatedImageAncestralSeed })}
                  variant="secondary"
                  size="sm"
                  title="Use ancestral seed from preview image"
                  disabled={generatedImageAncestralSeed === null}
                >
                  ♻️
                </Button>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                -1 = use main seed (default). Set a different value to vary details while keeping composition.
              </p>
            </div>
          </div>

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
                { value: "uint8", label: "UINT8" },
                { value: "uint7", label: "UINT7" },
                { value: "uint6", label: "UINT6" },
                { value: "uint5", label: "UINT5" },
                { value: "uint4", label: "UINT4" },
                { value: "uint3", label: "UINT3" },
                { value: "uint2", label: "UINT2" },
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
              value={params.max_prompt_chunks?.toString() || "0"}
              onChange={(e) => setParams({ ...params, max_prompt_chunks: parseInt(e.target.value) })}
            />
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
                onClick={handleAddToQueue}
                onContextMenu={(e) => {
                  e.preventDefault();
                  setMenuPosition({ x: e.clientX, y: e.clientY });
                  setShowForeverMenu(true);
                }}
                className="flex-1"
                size="lg"
              >
                {isGenerating ? "Add to Queue" : generateForever ? "Generate Forever ∞" : "Generate"}
              </Button>

              {/* Right-click menu for generate forever */}
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
                        onClick={handleAddToQueue}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          setMenuPosition({ x: e.clientX, y: e.clientY });
                          setShowForeverMenu(true);
                        }}
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

      {/* Prompt Editor */}
      {isPromptEditorOpen && (
        <PromptEditor
          initialPrompt={params.prompt}
          initialNegativePrompt={params.negative_prompt}
          onSave={handleSavePrompts}
          onClose={() => setIsPromptEditorOpen(false)}
        />
      )}
    </div>
  );
}
