"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Card from "../common/Card";
import Input from "../common/Input";
import Textarea from "../common/Textarea";
import TextareaWithTagSuggestions from "../common/TextareaWithTagSuggestions";
import Button from "../common/Button";
import Slider from "../common/Slider";
import Select from "../common/Select";
import ModelSelector from "../common/ModelSelector";
import LoRASelector from "../common/LoRASelector";
import ControlNetSelector from "../common/ControlNetSelector";
import ImageEditor from "../common/ImageEditor";
import TIPODialog, { TIPOSettings } from "../common/TIPODialog";
import { getSamplers, getScheduleTypes, generateInpaint, InpaintParams as ApiInpaintParams, LoRAConfig, ControlNetConfig, generateTIPOPrompt } from "@/utils/api";
import { wsClient } from "@/utils/websocket";
import { saveTempImage, loadTempImage, deleteTempImageRef } from "@/utils/tempImageStorage";
import { useStartup } from "@/contexts/StartupContext";

interface InpaintParams {
  prompt: string;
  negative_prompt?: string;
  steps?: number;
  cfg_scale?: number;
  sampler?: string;
  schedule_type?: string;
  seed?: number;
  width?: number;
  height?: number;
  denoising_strength?: number;
  img2img_fix_steps?: boolean;
  mask_blur?: number;
  inpaint_full_res?: boolean;
  inpaint_full_res_padding?: number;
  inpaint_fill_mode?: string;
  inpaint_fill_strength?: number;
  resize_mode?: string;
  resampling_method?: string;
  prompt_chunking_mode?: string;
  max_prompt_chunks?: number;
  loras?: LoRAConfig[];
  controlnets?: ControlNetConfig[];
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
  mask_blur: 4,
  inpaint_full_res: false,
  inpaint_full_res_padding: 32,
  inpaint_fill_mode: "original",
  inpaint_fill_strength: 1.0,
  resize_mode: "image",
  resampling_method: "lanczos",
  prompt_chunking_mode: "a1111",
  max_prompt_chunks: 0,
  loras: [],
  controlnets: [],
};

const STORAGE_KEY = "inpaint_params";
const PREVIEW_STORAGE_KEY = "inpaint_preview";
const INPUT_IMAGE_STORAGE_KEY = "inpaint_input_image";
const MASK_IMAGE_STORAGE_KEY = "inpaint_mask_image";

interface InpaintPanelProps {
  onTabChange?: (tab: "txt2img" | "img2img" | "inpaint") => void;
}

export default function InpaintPanel({ onTabChange }: InpaintPanelProps = {}) {
  const { modelLoaded, isBackendReady } = useStartup();
  const [params, setParams] = useState<InpaintParams>(DEFAULT_PARAMS);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
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
  const [samplers, setSamplers] = useState<Array<{ id: string; name: string }>>([]);
  const [scheduleTypes, setScheduleTypes] = useState<Array<{ id: string; name: string }>>([]);
  const [isMounted, setIsMounted] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [showImageEditor, setShowImageEditor] = useState(false);
  const [editingImageUrl, setEditingImageUrl] = useState<string | null>(null);
  const [sendImage, setSendImage] = useState(true);
  const [sendPrompt, setSendPrompt] = useState(true);
  const [sendParameters, setSendParameters] = useState(true);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [isTIPODialogOpen, setIsTIPODialogOpen] = useState(false);
  const [tipoSettings, setTipoSettings] = useState<TIPOSettings>({
    model_name: "KBlueLeaf/TIPO-500M",
    tag_length: "short",
    nl_length: "short",
    temperature: 1.0,
    top_p: 0.95,
    top_k: 50,
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

  const promptTextareaRef = useRef<HTMLTextAreaElement | null>(null);

  // WebSocket progress callback
  const handleProgress = useCallback((step: number, totalSteps: number, message: string, preview?: string) => {
    if (isGenerating) {
      setProgress(step);
      setTotalSteps(totalSteps);
      if (preview) {
        setPreviewImage(preview);
      }
    }
  }, [isGenerating]);

  // Setup WebSocket connection
  useEffect(() => {
    wsClient.connect();
    wsClient.subscribe(handleProgress);

    return () => {
      wsClient.unsubscribe(handleProgress);
    };
  }, [handleProgress]);

  // Load from localStorage after component mounts (client-side only)
  useEffect(() => {
    console.clear();
    console.log("=== InpaintPanel mounted ===");
    setIsMounted(true);

    const loadInitialData = async () => {
      // Load params
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const merged = { ...DEFAULT_PARAMS, ...parsed };
          setParams(merged);
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
    };

    loadInitialData();
  }, []);

  // When model loads on startup, load samplers and schedule types
  useEffect(() => {
    if (modelLoaded) {
      loadSamplers();
      loadScheduleTypes();
    }
  }, [modelLoaded]);

  // When backend becomes ready, reload temp images if not already loaded
  useEffect(() => {
    if (isBackendReady) {
      const reloadImages = async () => {
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
        // If in scale mode, update width/height based on scale
        if (sizeMode === "scale") {
          const scaledWidth = Math.round(img.width * scale / 64) * 64;
          const scaledHeight = Math.round(img.height * scale / 64) * 64;
          setParams((prev) => ({ ...prev, width: scaledWidth, height: scaledHeight }));
        }
      };
      img.src = inputImagePreview;
    }
  }, [inputImagePreview]);

  // Save params to localStorage whenever they change (but only after mounted)
  useEffect(() => {
    if (isMounted) {
      // ControlNet images are now managed by ControlNetSelector via tempImageStorage
      localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
      console.log('[Inpaint] Saving params to localStorage:', {
        loras: params.loras?.length || 0,
        controlnets: params.controlnets?.length || 0,
        fullParams: params
      });
    }
  }, [params, isMounted]);

  // Save preview image to localStorage whenever it changes
  useEffect(() => {
    if (isMounted && generatedImage) {
      localStorage.setItem(PREVIEW_STORAGE_KEY, generatedImage);
    }
  }, [generatedImage, isMounted]);

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
    }
  };

  const loadScheduleTypes = async () => {
    try {
      const data = await getScheduleTypes();
      setScheduleTypes(data.schedule_types);
    } catch (error) {
      console.error("Failed to load schedule types:", error);
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

    // Send image if checked
    if (sendImage) {
      try {
        // Fetch the generated image and convert to base64
        const response = await fetch(generatedImage);
        const blob = await response.blob();
        const reader = new FileReader();
        reader.onloadend = async () => {
          const base64data = reader.result as string;
          const ref = await saveTempImage(base64data);
          localStorage.setItem("img2img_input_image", ref);
          window.dispatchEvent(new Event("img2img_input_updated"));
        };
        reader.readAsDataURL(blob);
      } catch (error) {
        console.error("Failed to send image to img2img:", error);
      }
    }

    // Send prompt if checked
    if (sendPrompt) {
      const img2imgParams = JSON.parse(localStorage.getItem("img2img_params") || "{}");
      img2imgParams.prompt = params.prompt;
      img2imgParams.negative_prompt = params.negative_prompt;
      localStorage.setItem("img2img_params", JSON.stringify(img2imgParams));
    }

    // Send parameters if checked
    if (sendParameters) {
      const img2imgParams = JSON.parse(localStorage.getItem("img2img_params") || "{}");
      img2imgParams.steps = params.steps;
      img2imgParams.cfg_scale = params.cfg_scale;
      img2imgParams.sampler = params.sampler;
      img2imgParams.schedule_type = params.schedule_type;
      img2imgParams.seed = params.seed;
      img2imgParams.width = params.width;
      img2imgParams.height = params.height;
      img2imgParams.denoising_strength = params.denoising_strength;
      localStorage.setItem("img2img_params", JSON.stringify(img2imgParams));
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

  const handleGenerateTIPO = async () => {
    const textarea = promptTextareaRef.current;
    if (!textarea) return;

    const selectionStart = textarea.selectionStart;
    const selectionEnd = textarea.selectionEnd;
    const hasSelection = selectionStart !== selectionEnd;

    const inputPrompt = hasSelection
      ? params.prompt.substring(selectionStart, selectionEnd)
      : params.prompt;

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

  const handleGenerate = async () => {
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

    setIsGenerating(true);
    setProgress(0);
    const denoisingStrength = params.denoising_strength || 0.75;
    const actualSteps = Math.ceil((params.steps || 20) * denoisingStrength);
    setTotalSteps(actualSteps);
    setPreviewImage(null);
    setGeneratedImage(null);

    try {
      const apiParams: ApiInpaintParams = {
        prompt: params.prompt,
        negative_prompt: params.negative_prompt,
        steps: params.steps,
        cfg_scale: params.cfg_scale,
        sampler: params.sampler,
        schedule_type: params.schedule_type,
        seed: params.seed,
        width: params.width,
        height: params.height,
        denoising_strength: params.denoising_strength,
        mask_blur: params.mask_blur,
        inpaint_full_res: params.inpaint_full_res,
        inpaint_full_res_padding: params.inpaint_full_res_padding,
        inpaint_fill_mode: params.inpaint_fill_mode,
        inpaint_fill_strength: params.inpaint_fill_strength,
        resize_mode: params.resize_mode,
        resampling_method: params.resampling_method,
        loras: params.loras,
        controlnets: params.controlnets,
      };

      console.log('[Inpaint] Generating with params:', {
        loras: apiParams.loras?.length || 0,
        controlnets: apiParams.controlnets?.length || 0,
      });

      const result = await generateInpaint(apiParams, inputImagePreview, maskImage);

      if (result.success) {
        const imageUrl = `/outputs/${result.image.filename}`;
        setGeneratedImage(imageUrl);
        setGeneratedImageSeed(result.actual_seed);
        setGeneratedImageAncestralSeed(result.image.ancestral_seed || null);

        if (isMounted) {
          localStorage.setItem(PREVIEW_STORAGE_KEY, imageUrl);
        }
      } else {
        alert("Generation failed");
      }
    } catch (error) {
      console.error("Generation error:", error);
      alert("Generation failed: " + (error instanceof Error ? error.message : String(error)));
    } finally {
      setTimeout(() => {
        setIsGenerating(false);
        setProgress(0);
        setPreviewImage(null);
      }, 500);
    }
  };

  // Handle Ctrl+Enter keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === 'Enter' && !isGenerating) {
        e.preventDefault();
        handleGenerate();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [params, isGenerating, inputImage, inputImagePreview, maskImage]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Parameters Panel */}
      <div className="space-y-4">
        <ModelSelector />

        <LoRASelector
          value={params.loras || []}
          onChange={(loras) => setParams({ ...params, loras })}
          disabled={isGenerating}
          storageKey="inpaint_lora_collapsed"
        />

        <ControlNetSelector
          value={params.controlnets || []}
          onChange={(controlnets) => setParams({ ...params, controlnets })}
          disabled={isGenerating}
          storageKey="inpaint_controlnet_collapsed"
          inputImagePreview={inputImagePreview}
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
            <div className="absolute top-0 right-0 flex items-center gap-1 px-2 py-1">
              <button
                onClick={handleGenerateTIPO}
                disabled={isGeneratingTIPO || isGenerating}
                className="p-1 hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Generate enhanced prompt with TIPO"
              >
                {isGeneratingTIPO ? "⏳" : "✨"}
              </button>
              <button
                onClick={() => setIsTIPODialogOpen(true)}
                disabled={isGenerating}
                className="p-1 hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="TIPO Settings"
              >
                ⚙️
              </button>
            </div>
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

        <TIPODialog
          isOpen={isTIPODialogOpen}
          onClose={() => setIsTIPODialogOpen(false)}
          onSave={(settings) => setTipoSettings(settings)}
          currentSettings={tipoSettings}
        />

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
              <Slider
                label="Fill Strength"
                min={0}
                max={1}
                step={0.05}
                value={params.inpaint_fill_strength || 1.0}
                onChange={(e) => setParams({ ...params, inpaint_fill_strength: parseFloat(e.target.value) })}
              />
            )}

            <div className="grid grid-cols-2 gap-4">
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
                <div className="grid grid-cols-2 gap-4">
                  <Slider
                    label="Width"
                    min={64}
                    max={2048}
                    step={64}
                    value={params.width}
                    onChange={(e) => setParams({ ...params, width: parseInt(e.target.value) })}
                  />
                  <Slider
                    label="Height"
                    min={64}
                    max={2048}
                    step={64}
                    value={params.height}
                    onChange={(e) => setParams({ ...params, height: parseInt(e.target.value) })}
                  />
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

            <div className="grid grid-cols-2 gap-4">
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

            <div className="grid grid-cols-2 gap-4">
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

            <div className="grid grid-cols-2 gap-4">
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
            <div className="grid grid-cols-2 gap-4">
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

        <div className="flex gap-2">
          <Button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="flex-1"
            size="lg"
          >
            {isGenerating ? "Generating..." : "Generate"}
          </Button>
          <Button
            onClick={resetToDefault}
            disabled={isGenerating}
            variant="secondary"
            size="lg"
          >
            Reset
          </Button>
        </div>
      </div>

      {/* Preview Panel */}
      <div>
        <Card title="Preview">
          <div className="space-y-2">
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
            <div className="aspect-square bg-gray-800 rounded-lg flex items-center justify-center">
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
                <div className="grid grid-cols-3 gap-2">
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
    </div>
  );
}
