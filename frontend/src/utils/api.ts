import axios from "axios";

const api = axios.create({
  baseURL: "/api/v1",
  headers: {
    "Content-Type": "application/json",
  },
  // Set a very long timeout for generation requests (10 minutes)
  // Set to 0 to disable timeout completely
  timeout: 600000, // 10 minutes in milliseconds
});

// Add auth token to requests if available (session storage - cleared on browser close)
api.interceptors.request.use(
  (config) => {
    const token = sessionStorage.getItem("auth_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle 401 errors (unauthorized)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token is invalid or expired
      sessionStorage.removeItem("auth_token");
      // Only redirect if not already on login page
      if (typeof window !== "undefined" && !window.location.pathname.includes("/login")) {
        window.location.href = "/login";
      }
    }
    return Promise.reject(error);
  }
);

// Helper function to load ControlNet images from temp storage
const loadControlNetImages = async (
  controlnets: ControlNetConfig[] | undefined,
  storageKey: string
): Promise<ControlNetConfig[]> => {
  if (!controlnets || controlnets.length === 0) {
    return controlnets || [];
  }

  console.log(`[API] Loading ControlNet images from temp storage (${storageKey})...`);
  const { loadTempImage } = await import('./tempImageStorage');

  const IMAGE_STORAGE_KEY = `${storageKey}_images`;
  const stored = localStorage.getItem(IMAGE_STORAGE_KEY);
  const storedLength = stored ? stored.length : 0;
  console.log(`[API] localStorage key: ${IMAGE_STORAGE_KEY} (${storedLength} chars)`);
  const imageRefs: { [index: number]: string } = stored ? JSON.parse(stored) : {};
  console.log('[API] imageRefs count:', Object.keys(imageRefs).length);

  const loadedControlnets = await Promise.all(
    controlnets.map(async (cn, index) => {
      // If use_input_image is true, don't need to load image (backend will use input image)
      if (cn.use_input_image) {
        console.log(`[API] ControlNet ${index}: use_input_image=true, skipping image load`);
        return cn;
      }

      // If image_base64 is already set (e.g., from loop generation), use it directly
      if (cn.image_base64) {
        console.log(`[API] ControlNet ${index}: image_base64 already set (length: ${cn.image_base64.length}), skipping localStorage load`);
        return cn;
      }

      const imageRef = imageRefs[index];
      console.log(`[API] ControlNet ${index}: imageRef = ${imageRef ? 'exists' : 'none'}`);
      if (imageRef) {
        const imageData = await loadTempImage(imageRef);
        const imageDataLength = imageData?.length || 0;
        console.log(`[API] ControlNet ${index}: loaded image data (${imageDataLength} chars)`);
        const base64Data = imageData.startsWith('data:')
          ? imageData.split(',')[1]
          : imageData;
        const base64Length = base64Data?.length || 0;
        console.log(`[API] ControlNet ${index}: base64 (${base64Length} chars)`);
        return {
          ...cn,
          image_base64: base64Data,
        };
      }
      console.log(`[API] ControlNet ${index}: No imageRef, using fallback`);
      return {
        ...cn,
        image_base64: cn.image_base64,
      };
    })
  );

  console.log('[API] Final controlnets:', loadedControlnets.map((cn, i) => ({
    index: i,
    has_image: !!cn.image_base64,
    length: cn.image_base64?.length
  })));

  return loadedControlnets;
};

export interface LoRAConfig {
  path: string;
  strength: number;
  apply_to_text_encoder: boolean;
  apply_to_unet: boolean;
  unet_layer_weights: {
    [layerName: string]: number;
  };
  step_range: [number, number];
}

export interface LoRAInfo {
  name: string;
  path: string;
  size: number;
  exists: boolean;
  layers: string[];
}

export interface ControlNetConfig {
  model_path: string;
  image_base64?: string;
  strength: number;
  start_step: number;
  end_step: number;
  layer_weights?: { down: number; mid: number; up: number };
  prompt?: string;
  is_lllite: boolean;
  preprocessor?: string;
  enable_preprocessor: boolean;
}

export interface GenerationParams {
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
  model?: string;
  loras?: LoRAConfig[];
  prompt_chunking_mode?: string;
  max_prompt_chunks?: number;
  controlnets?: ControlNetConfig[];
  // Dynamic CFG scheduling
  cfg_schedule_type?: string;
  cfg_schedule_min?: number;
  cfg_schedule_max?: number;
  cfg_schedule_power?: number;
  cfg_rescale_snr_alpha?: number;
  // Dynamic thresholding
  dynamic_threshold_percentile?: number;
  dynamic_threshold_mimic_scale?: number;
  // NAG (Normalized Attention Guidance)
  nag_enable?: boolean;
  nag_scale?: number;
  nag_tau?: number;
  nag_alpha?: number;
  nag_sigma_end?: number;
  nag_negative_prompt?: string;
  // U-Net Quantization
  unet_quantization?: string | null;
  // torch.compile optimization
  use_torch_compile?: boolean;
  // TIPO prompt upsampling
  use_tipo?: boolean;
  tipo_config?: any;  // TIPO configuration object
}

export interface Img2ImgParams extends GenerationParams {
  denoising_strength?: number;
  img2img_fix_steps?: boolean;
  resize_mode?: string;
  resampling_method?: string;
}

export interface InpaintParams extends GenerationParams {
  denoising_strength?: number;
  img2img_fix_steps?: boolean;
  mask_blur?: number;
  inpaint_full_res?: boolean;
  inpaint_full_res_padding?: number;
  inpaint_fill_mode?: string;
  inpaint_fill_strength?: number;
  resize_mode?: string;
  resampling_method?: string;
}

export interface GeneratedImage {
  id: number;
  filename: string;
  prompt: string;
  negative_prompt: string;
  model_name: string;
  sampler: string;
  steps: number;
  cfg_scale: number;
  seed: number;
  ancestral_seed?: number;
  width: number;
  height: number;
  generation_type: string;
  parameters: any;
  created_at: string;
  is_favorite: boolean;
  image_hash?: string;
  source_image_hash?: string;
  mask_data?: string;
  lora_names?: string;
  model_hash?: string;
  unet_quantization?: string;
  // Advanced CFG parameters
  cfg_schedule_type?: string;
  cfg_schedule_min?: string;
  cfg_schedule_max?: string;
  cfg_schedule_power?: string;
  cfg_rescale_snr_alpha?: string;
  dynamic_threshold_percentile?: string;
  dynamic_threshold_mimic_scale?: string;
  // NAG parameters
  nag_enable?: string;
  nag_scale?: string;
  nag_tau?: string;
  nag_alpha?: string;
  nag_sigma_end?: string;
}

export const generateTxt2Img = async (params: GenerationParams) => {
  // Get attention_type from localStorage
  const attentionType = typeof window !== 'undefined' ? localStorage.getItem('attention_type') : null;

  const paramsWithImages = {
    ...params,
    attention_type: attentionType || 'normal',
    controlnets: await loadControlNetImages(params.controlnets, "txt2img_controlnet_collapsed"),
  };

  const formData = new FormData();

  formData.append("prompt", paramsWithImages.prompt);
  formData.append("negative_prompt", paramsWithImages.negative_prompt || "");
  formData.append("steps", String(paramsWithImages.steps || 20));
  formData.append("cfg_scale", String(paramsWithImages.cfg_scale || 7.0));
  formData.append("sampler", paramsWithImages.sampler || "euler");
  formData.append("schedule_type", paramsWithImages.schedule_type || "uniform");
  formData.append("seed", String(paramsWithImages.seed || -1));
  formData.append("ancestral_seed", String(paramsWithImages.ancestral_seed ?? -1));
  formData.append("width", String(paramsWithImages.width || 1024));
  formData.append("height", String(paramsWithImages.height || 1024));
  formData.append("batch_size", String(paramsWithImages.batch_size || 1));
  formData.append("loras", JSON.stringify(paramsWithImages.loras || []));
  formData.append("controlnets", JSON.stringify(paramsWithImages.controlnets || []));
  formData.append("prompt_chunking_mode", paramsWithImages.prompt_chunking_mode || "a1111");
  formData.append("max_prompt_chunks", String(paramsWithImages.max_prompt_chunks ?? 0));
  formData.append("developer_mode", String(paramsWithImages.developer_mode ?? false));
  formData.append("cfg_schedule_type", paramsWithImages.cfg_schedule_type || "constant");
  formData.append("cfg_schedule_min", String(paramsWithImages.cfg_schedule_min ?? 1.0));
  formData.append("cfg_schedule_max", String(paramsWithImages.cfg_schedule_max ?? ""));
  formData.append("cfg_schedule_power", String(paramsWithImages.cfg_schedule_power ?? 2.0));
  formData.append("cfg_rescale_snr_alpha", String(paramsWithImages.cfg_rescale_snr_alpha ?? 0.0));
  formData.append("dynamic_threshold_percentile", String(paramsWithImages.dynamic_threshold_percentile ?? 0.0));
  formData.append("dynamic_threshold_mimic_scale", String(paramsWithImages.dynamic_threshold_mimic_scale ?? 7.0));
  formData.append("nag_enable", String(paramsWithImages.nag_enable ?? false));
  formData.append("nag_scale", String(paramsWithImages.nag_scale ?? 5.0));
  formData.append("nag_tau", String(paramsWithImages.nag_tau ?? 3.5));
  formData.append("nag_alpha", String(paramsWithImages.nag_alpha ?? 0.25));
  formData.append("nag_sigma_end", String(paramsWithImages.nag_sigma_end ?? 3.0));
  formData.append("nag_negative_prompt", paramsWithImages.nag_negative_prompt || "");
  formData.append("attention_type", paramsWithImages.attention_type || "normal");

  // Debug log for quantization
  console.log('[API] txt2img unet_quantization:', paramsWithImages.unet_quantization);
  if (paramsWithImages.unet_quantization && paramsWithImages.unet_quantization !== "none") {
    formData.append("unet_quantization", paramsWithImages.unet_quantization);
    console.log('[API] Added unet_quantization to FormData:', paramsWithImages.unet_quantization);
  } else {
    console.log('[API] No quantization or "none" selected');
  }

  // torch.compile optimization
  formData.append("use_torch_compile", String(paramsWithImages.use_torch_compile ?? false));

  // TIPO prompt upsampling
  formData.append("use_tipo", String(paramsWithImages.use_tipo ?? false));
  formData.append("tipo_config", JSON.stringify(paramsWithImages.tipo_config || {}));

  const response = await api.post("/generate/txt2img", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const generateImg2Img = async (params: Img2ImgParams, image: File | string) => {
  // Get attention_type from localStorage
  const attentionType = typeof window !== 'undefined' ? localStorage.getItem('attention_type') : null;

  const paramsWithImages = {
    ...params,
    attention_type: attentionType || 'normal',
    controlnets: await loadControlNetImages(params.controlnets, "img2img_controlnet_collapsed"),
  };

  const formData = new FormData();

  // Handle both File objects and data URLs
  if (typeof image === 'string') {
    // Convert data URL or URL to blob
    const response = await fetch(image);
    const blob = await response.blob();
    formData.append("image", blob, "input.png");
  } else {
    formData.append("image", image);
  }

  formData.append("prompt", paramsWithImages.prompt);
  formData.append("negative_prompt", paramsWithImages.negative_prompt || "");
  formData.append("steps", String(paramsWithImages.steps || 20));
  formData.append("cfg_scale", String(paramsWithImages.cfg_scale || 7.0));
  formData.append("denoising_strength", String(paramsWithImages.denoising_strength || 0.75));
  formData.append("img2img_fix_steps", String(paramsWithImages.img2img_fix_steps ?? true));
  formData.append("sampler", paramsWithImages.sampler || "euler");
  formData.append("schedule_type", paramsWithImages.schedule_type || "uniform");
  formData.append("seed", String(paramsWithImages.seed || -1));
  formData.append("ancestral_seed", String(paramsWithImages.ancestral_seed ?? -1));
  formData.append("width", String(paramsWithImages.width || 1024));
  formData.append("height", String(paramsWithImages.height || 1024));
  formData.append("resize_mode", paramsWithImages.resize_mode || "image");
  formData.append("resampling_method", paramsWithImages.resampling_method || "lanczos");
  formData.append("loras", JSON.stringify(paramsWithImages.loras || []));
  formData.append("controlnets", JSON.stringify(paramsWithImages.controlnets || []));
  formData.append("prompt_chunking_mode", paramsWithImages.prompt_chunking_mode || "a1111");
  formData.append("max_prompt_chunks", String(paramsWithImages.max_prompt_chunks ?? 0));
  formData.append("developer_mode", String(paramsWithImages.developer_mode ?? false));
  formData.append("cfg_schedule_type", paramsWithImages.cfg_schedule_type || "constant");
  formData.append("cfg_schedule_min", String(paramsWithImages.cfg_schedule_min ?? 1.0));
  formData.append("cfg_schedule_max", String(paramsWithImages.cfg_schedule_max ?? ""));
  formData.append("cfg_schedule_power", String(paramsWithImages.cfg_schedule_power ?? 2.0));
  formData.append("cfg_rescale_snr_alpha", String(paramsWithImages.cfg_rescale_snr_alpha ?? 0.0));
  formData.append("dynamic_threshold_percentile", String(paramsWithImages.dynamic_threshold_percentile ?? 0.0));
  formData.append("dynamic_threshold_mimic_scale", String(paramsWithImages.dynamic_threshold_mimic_scale ?? 7.0));
  formData.append("nag_enable", String(paramsWithImages.nag_enable ?? false));
  formData.append("nag_scale", String(paramsWithImages.nag_scale ?? 5.0));
  formData.append("nag_tau", String(paramsWithImages.nag_tau ?? 3.5));
  formData.append("nag_alpha", String(paramsWithImages.nag_alpha ?? 0.25));
  formData.append("nag_sigma_end", String(paramsWithImages.nag_sigma_end ?? 3.0));
  formData.append("nag_negative_prompt", paramsWithImages.nag_negative_prompt || "");
  formData.append("attention_type", paramsWithImages.attention_type || "normal");

  // Debug log for quantization
  console.log('[API] img2img unet_quantization:', paramsWithImages.unet_quantization);
  if (paramsWithImages.unet_quantization && paramsWithImages.unet_quantization !== "none") {
    formData.append("unet_quantization", paramsWithImages.unet_quantization);
    console.log('[API] Added unet_quantization to FormData:', paramsWithImages.unet_quantization);
  } else {
    console.log('[API] No quantization or "none" selected');
  }

  // torch.compile optimization
  formData.append("use_torch_compile", String(paramsWithImages.use_torch_compile ?? false));

  // TIPO prompt upsampling
  formData.append("use_tipo", String(paramsWithImages.use_tipo ?? false));
  formData.append("tipo_config", JSON.stringify(paramsWithImages.tipo_config || {}));

  const response = await api.post("/generate/img2img", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const generateInpaint = async (params: InpaintParams, image: File | string, mask: File | string) => {
  // Get attention_type from localStorage
  const attentionType = typeof window !== 'undefined' ? localStorage.getItem('attention_type') : null;

  const paramsWithImages = {
    ...params,
    attention_type: attentionType || 'normal',
    controlnets: await loadControlNetImages(params.controlnets, "inpaint_controlnet_collapsed"),
  };

  const formData = new FormData();

  // Handle both File objects and data URLs for image
  if (typeof image === 'string') {
    const response = await fetch(image);
    const blob = await response.blob();
    formData.append("image", blob, "input.png");
  } else {
    formData.append("image", image);
  }

  // Handle both File objects and data URLs for mask
  if (typeof mask === 'string') {
    const response = await fetch(mask);
    const blob = await response.blob();
    formData.append("mask", blob, "mask.png");
  } else {
    formData.append("mask", mask);
  }

  formData.append("prompt", paramsWithImages.prompt);
  formData.append("negative_prompt", paramsWithImages.negative_prompt || "");
  formData.append("steps", String(paramsWithImages.steps || 20));
  formData.append("cfg_scale", String(paramsWithImages.cfg_scale || 7.0));
  formData.append("denoising_strength", String(paramsWithImages.denoising_strength || 0.75));
  formData.append("img2img_fix_steps", String(paramsWithImages.img2img_fix_steps ?? true));
  formData.append("mask_blur", String(paramsWithImages.mask_blur || 4));
  formData.append("sampler", paramsWithImages.sampler || "euler");
  formData.append("schedule_type", paramsWithImages.schedule_type || "uniform");
  formData.append("seed", String(paramsWithImages.seed || -1));
  formData.append("ancestral_seed", String(paramsWithImages.ancestral_seed ?? -1));
  formData.append("width", String(paramsWithImages.width || 1024));
  formData.append("height", String(paramsWithImages.height || 1024));
  formData.append("inpaint_full_res", String(paramsWithImages.inpaint_full_res || false));
  formData.append("inpaint_full_res_padding", String(paramsWithImages.inpaint_full_res_padding || 32));
  formData.append("inpaint_fill_mode", paramsWithImages.inpaint_fill_mode || "original");
  formData.append("inpaint_fill_strength", String(paramsWithImages.inpaint_fill_strength ?? 1.0));
  formData.append("inpaint_blur_strength", String(paramsWithImages.inpaint_blur_strength ?? 1.0));
  formData.append("resize_mode", paramsWithImages.resize_mode || "image");
  formData.append("resampling_method", paramsWithImages.resampling_method || "lanczos");
  formData.append("loras", JSON.stringify(paramsWithImages.loras || []));
  formData.append("controlnets", JSON.stringify(paramsWithImages.controlnets || []));
  formData.append("prompt_chunking_mode", paramsWithImages.prompt_chunking_mode || "a1111");
  formData.append("max_prompt_chunks", String(paramsWithImages.max_prompt_chunks ?? 0));
  formData.append("developer_mode", String(paramsWithImages.developer_mode ?? false));
  formData.append("cfg_schedule_type", paramsWithImages.cfg_schedule_type || "constant");
  formData.append("cfg_schedule_min", String(paramsWithImages.cfg_schedule_min ?? 1.0));
  formData.append("cfg_schedule_max", String(paramsWithImages.cfg_schedule_max ?? ""));
  formData.append("cfg_schedule_power", String(paramsWithImages.cfg_schedule_power ?? 2.0));
  formData.append("cfg_rescale_snr_alpha", String(paramsWithImages.cfg_rescale_snr_alpha ?? 0.0));
  formData.append("dynamic_threshold_percentile", String(paramsWithImages.dynamic_threshold_percentile ?? 0.0));
  formData.append("dynamic_threshold_mimic_scale", String(paramsWithImages.dynamic_threshold_mimic_scale ?? 7.0));
  formData.append("nag_enable", String(paramsWithImages.nag_enable ?? false));
  formData.append("nag_scale", String(paramsWithImages.nag_scale ?? 5.0));
  formData.append("nag_tau", String(paramsWithImages.nag_tau ?? 3.5));
  formData.append("nag_alpha", String(paramsWithImages.nag_alpha ?? 0.25));
  formData.append("nag_sigma_end", String(paramsWithImages.nag_sigma_end ?? 3.0));
  formData.append("nag_negative_prompt", paramsWithImages.nag_negative_prompt || "");
  formData.append("attention_type", paramsWithImages.attention_type || "normal");

  // Debug log for quantization
  console.log('[API] inpaint unet_quantization:', paramsWithImages.unet_quantization);
  if (paramsWithImages.unet_quantization && paramsWithImages.unet_quantization !== "none") {
    formData.append("unet_quantization", paramsWithImages.unet_quantization);
    console.log('[API] Added unet_quantization to FormData:', paramsWithImages.unet_quantization);
  } else {
    console.log('[API] No quantization or "none" selected');
  }

  // torch.compile optimization
  formData.append("use_torch_compile", String(paramsWithImages.use_torch_compile ?? false));

  // TIPO prompt upsampling
  formData.append("use_tipo", String(paramsWithImages.use_tipo ?? false));
  formData.append("tipo_config", JSON.stringify(paramsWithImages.tipo_config || {}));

  const response = await api.post("/generate/inpaint", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export interface ImageFilters {
  skip?: number;
  limit?: number;
  search?: string;
  generation_types?: string;  // Comma-separated: txt2img,img2img,inpaint
  date_from?: string;  // ISO format
  date_to?: string;  // ISO format
  width_min?: number;
  width_max?: number;
  height_min?: number;
  height_max?: number;
}

export const getImages = async (filters: ImageFilters = {}) => {
  const params = new URLSearchParams();
  params.append("skip", String(filters.skip || 0));
  params.append("limit", String(filters.limit || 50));
  if (filters.search) params.append("search", filters.search);
  if (filters.generation_types) params.append("generation_types", filters.generation_types);
  if (filters.date_from) params.append("date_from", filters.date_from);
  if (filters.date_to) params.append("date_to", filters.date_to);
  if (filters.width_min !== undefined) params.append("width_min", String(filters.width_min));
  if (filters.width_max !== undefined) params.append("width_max", String(filters.width_max));
  if (filters.height_min !== undefined) params.append("height_min", String(filters.height_min));
  if (filters.height_max !== undefined) params.append("height_max", String(filters.height_max));

  const response = await api.get(`/images?${params.toString()}`);
  return response.data;
};

export const getImage = async (id: number) => {
  const response = await api.get(`/images/${id}`);
  return response.data;
};

export const deleteImage = async (id: number) => {
  const response = await api.delete(`/images/${id}`);
  return response.data;
};

export const getModels = async () => {
  const response = await api.get("/models");
  return response.data;
};

export const getCurrentModel = async () => {
  const response = await api.get("/models/current");
  return response.data;
};

export const loadModel = async (sourceType: string, source: string, revision?: string) => {
  const formData = new FormData();
  formData.append("source_type", sourceType);
  formData.append("source", source);
  if (revision) {
    formData.append("revision", revision);
  }

  const response = await api.post("/models/load", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const uploadModel = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post("/models/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const getSamplers = async () => {
  const response = await api.get("/samplers");
  return response.data;
};

export const getScheduleTypes = async () => {
  const response = await api.get("/schedule-types");
  return response.data;
};

export const getLoras = async (): Promise<{ loras: Array<{ path: string; name: string }> }> => {
  const response = await api.get("/loras");
  return response.data;
};

export const getLoraInfo = async (loraName: string) => {
  const response = await api.get(`/loras/${loraName}`);
  return response.data;
};

export interface TokenizeResult {
  token_count: number;
  total_count: number;
  chunks: number;
}

export const tokenizePrompt = async (prompt: string): Promise<TokenizeResult> => {
  const formData = new FormData();
  formData.append("prompt", prompt);

  const response = await api.post("/tokenize", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const restartBackend = async () => {
  const response = await api.post("/system/restart-backend");
  return response.data;
};

export const restartFrontend = async () => {
  // Reload the page to restart the frontend
  window.location.reload();
};

export const restartBoth = async () => {
  // First restart backend
  await api.post("/system/restart-backend");
  // Then reload the page after a delay
  setTimeout(() => {
    window.location.reload();
  }, 2000);
};

export const getControlNets = async () => {
  const response = await api.get("/controlnets");
  return response.data;
};

export interface ControlNetInfo {
  name: string;
  path: string;
  layers: string[];
  is_lllite: boolean;
  exists: boolean;
  error?: string;
}

export const getControlNetInfo = async (controlnetPath: string): Promise<ControlNetInfo> => {
  const response = await api.get(`/controlnets/${encodeURIComponent(controlnetPath)}/info`);
  return response.data;
};

// Temp image storage API
export const uploadTempImage = async (imageBase64: string): Promise<string> => {
  const formData = new FormData();
  formData.append("image_base64", imageBase64);

  const response = await api.post("/temp-images/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data.image_id;
};

export const getTempImage = async (imageId: string): Promise<string> => {
  const response = await api.get(`/temp-images/${imageId}`);
  return response.data.image_base64;
};

export const deleteTempImage = async (imageId: string): Promise<void> => {
  await api.delete(`/temp-images/${imageId}`);
};

export const cleanupTempImages = async (maxAgeHours: number = 24): Promise<number> => {
  const response = await api.post("/temp-images/cleanup", null, {
    params: { max_age_hours: maxAgeHours },
  });
  return response.data.deleted_count;
};

// ControlNet Preprocessor API
export interface PreprocessorInfo {
  id: string;
  name: string;
  category: string;
}

export const detectControlNetPreprocessor = async (modelPath: string): Promise<{
  model_path: string;
  preprocessor: string;
  requires_preprocessing: boolean;
}> => {
  const response = await api.get("/controlnet/detect-preprocessor", {
    params: { model_path: modelPath },
  });
  return response.data;
};

export const preprocessControlNetImage = async (
  imageBlob: Blob,
  preprocessor: string,
  options: {
    lowThreshold?: number;
    highThreshold?: number;
    downSamplingRate?: number;
    sharpness?: number;
    blurStrength?: number;
  } = {}
): Promise<{ preprocessed_image: string; preprocessor: string }> => {
  const formData = new FormData();
  formData.append("image", imageBlob);
  formData.append("preprocessor", preprocessor);
  formData.append("low_threshold", (options.lowThreshold ?? 100).toString());
  formData.append("high_threshold", (options.highThreshold ?? 200).toString());

  if (options.downSamplingRate !== undefined) {
    formData.append("down_sampling_rate", options.downSamplingRate.toString());
  }
  if (options.sharpness !== undefined) {
    formData.append("sharpness", options.sharpness.toString());
  }
  if (options.blurStrength !== undefined) {
    formData.append("blur_strength", options.blurStrength.toString());
  }

  const response = await api.post("/controlnet/preprocess-image", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const getAvailablePreprocessors = async (): Promise<{ preprocessors: PreprocessorInfo[] }> => {
  const response = await api.get("/controlnet/preprocessors");
  return response.data;
};

// TIPO API
export interface TIPOGenerateRequest {
  input_prompt: string;
  model_name?: string;  // Model to use (auto-loads if not loaded)
  tag_length?: string;  // very_short, short, long, very_long
  nl_length?: string;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_new_tokens?: number;
  ban_tags?: string;  // Comma-separated tags to exclude from generation
  category_order?: string[];
  enabled_categories?: Record<string, boolean>;
}

export interface TIPOParsedOutput {
  rating: string;
  artist: string;
  copyright: string;
  characters: string;
  target: string;
  short_nl: string;
  long_nl: string;
  tags: string[];
  special_tags: string[];
  quality_tags: string[];
  meta_tags: string[];
  general_tags: string[];
}

export interface TIPOGenerateResponse {
  status: string;
  original_prompt: string;
  raw_output: string;
  parsed: TIPOParsedOutput;
  generated_prompt: string;
}

export interface TIPOStatusResponse {
  loaded: boolean;
  model_name: string | null;
  device: string;
}

export const generateTIPOPrompt = async (request: TIPOGenerateRequest): Promise<TIPOGenerateResponse> => {
  const response = await api.post("/tipo/generate", request);
  return response.data;
};

export const loadTIPOModel = async (model_name: string = "KBlueLeaf/TIPO-500M") => {
  const response = await api.post("/tipo/load-model", { model_name });
  return response.data;
};

export const getTIPOStatus = async (): Promise<TIPOStatusResponse> => {
  const response = await api.get("/tipo/status");
  return response.data;
};

export const unloadTIPOModel = async () => {
  const response = await api.post("/tipo/unload");
  return response.data;
};

export const cancelGeneration = async () => {
  const response = await api.post("/cancel");
  return response.data;
};

// Image Tagger API
export interface TaggerPredictionsResponse {
  status: string;
  predictions: {
    rating: [string, number][];
    general: [string, number][];
    artist: [string, number][];
    character: [string, number][];
    copyright: [string, number][];
    meta: [string, number][];
    quality: [string, number][];
    model: [string, number][];
  };
}

export interface TaggerStatusResponse {
  loaded: boolean;
  model_path: string | null;
  tag_mapping_path: string | null;
  model_version: string | null;
}

export const loadTaggerModel = async (
  model_path?: string,
  tag_mapping_path?: string,
  use_gpu: boolean = true,
  use_huggingface: boolean = true,
  repo_id: string = "cella110n/cl_tagger",
  model_version: string = "cl_tagger_1_02"
) => {
  const response = await api.post("/tagger/load-model", {
    model_path,
    tag_mapping_path,
    use_gpu,
    use_huggingface,
    repo_id,
    model_version,
  });
  return response.data;
};

export const predictTags = async (
  image_base64: string,
  gen_threshold: number = 0.45,
  char_threshold: number = 0.45,
  model_version: string = "cl_tagger_1_02",
  auto_unload: boolean = true,
  thresholds?: { [key: string]: number }
): Promise<TaggerPredictionsResponse> => {
  const response = await api.post("/tagger/predict", {
    image_base64,
    gen_threshold,
    char_threshold,
    model_version,
    auto_unload,
    thresholds,
  });
  return response.data;
};

export const getTaggerStatus = async (): Promise<TaggerStatusResponse> => {
  const response = await api.get("/tagger/status");
  return response.data;
};

export const unloadTaggerModel = async () => {
  const response = await api.post("/tagger/unload");
  return response.data;
};

export interface GPUStats {
  index: number;
  name: string;
  vram_used_gb: number;
  vram_total_gb: number;
  vram_percent: number;
  gpu_utilization: number | null;
  temperature: number | null;
  power_watts: number | null;
}

export interface GPUStatsResponse {
  available: boolean;
  gpus?: GPUStats[];
  error?: string;
}

export const getGPUStats = async (): Promise<GPUStatsResponse> => {
  const response = await api.get("/system/gpu-stats");
  return response.data;
};

export default api;

// ============================================================
// Dataset Management API
// ============================================================

export interface Dataset {
  id: number;
  name: string;
  path: string;
  description?: string;
  recursive: boolean;
  read_exif: boolean;
  total_items: number;
  total_captions: number;
  total_tags: number;
  created_at: string;
  updated_at: string;
  last_scanned_at?: string;
}

export interface DatasetListResponse {
  datasets: Dataset[];
  total: number;
}

export interface DatasetCreateRequest {
  name: string;
  path: string;
  description?: string;
  recursive?: boolean;
  read_exif?: boolean;
}

export const listDatasets = async (): Promise<DatasetListResponse> => {
  const response = await api.get("/datasets");
  return response.data;
};

export const createDataset = async (data: DatasetCreateRequest): Promise<Dataset> => {
  const response = await api.post("/datasets", data);
  return response.data;
};

export const getDataset = async (id: number): Promise<Dataset> => {
  const response = await api.get(`/datasets/${id}`);
  return response.data;
};

export const deleteDataset = async (id: number): Promise<void> => {
  await api.delete(`/datasets/${id}`);
};

export interface DatasetScanResponse {
  items_found: number;
  captions_found: number;
  dataset: Dataset;
}

export const scanDataset = async (id: number): Promise<DatasetScanResponse> => {
  const response = await api.post(`/datasets/${id}/scan`);
  return response.data;
};

export interface TagDictionaryEntry {
  id: number;
  tag: string;
  category: string;
  count: number;
  display_name?: string;
  aliases?: string[];
  description?: string;
  source: string;
  is_official: boolean;
  is_deprecated: boolean;
  replacement_tag?: string;
  created_at: string;
  updated_at: string;
}

export interface TagDictionarySearchResponse {
  tags: TagDictionaryEntry[];
  total: number;
  page: number;
  page_size: number;
}

export interface TagDictionaryStatsResponse {
  total_tags: number;
}

export const searchTagDictionary = async (
  search?: string,
  category?: string,
  page: number = 1,
  page_size: number = 100
): Promise<TagDictionarySearchResponse> => {
  const response = await api.get("/tag-dictionary", {
    params: { search, category, page, page_size },
  });
  return response.data;
};

export const getTagDictionaryStats = async (): Promise<TagDictionaryStatsResponse> => {
  const response = await api.get("/tag-dictionary/stats");
  return response.data;
};

export interface DatasetItem {
  id: number;
  dataset_id: number;
  item_type: string;
  base_name: string;
  image_path: string;
  width: number;
  height: number;
  file_size: number;
  image_hash: string;
  created_at: string;
  updated_at: string;
  captions?: DatasetCaptionData[];
}

export interface DatasetCaptionData {
  id: number;
  item_id: number;
  caption_type: string;
  content: string;
  source: string;
  created_at: string;
  updated_at: string;
}

export interface DatasetItemListResponse {
  items: DatasetItem[];
  total: number;
  page: number;
  page_size: number;
}

export const listDatasetItems = async (
  datasetId: number,
  page: number = 1,
  pageSize: number = 50,
  search?: string
): Promise<DatasetItemListResponse> => {
  const params: any = { page, page_size: pageSize };
  if (search) params.search = search;
  const response = await api.get(`/datasets/${datasetId}/items`, { params });
  return response.data;
};

export const getDatasetItem = async (datasetId: number, itemId: number): Promise<DatasetItem> => {
  const response = await api.get(`/datasets/${datasetId}/items/${itemId}`);
  return response.data;
};

// ============================================================
// Training API
// ============================================================

export interface TrainingRun {
  id: number;
  dataset_id: number;
  run_id: string;  // UUID
  run_name: string;
  training_method: "lora" | "full_finetune";
  base_model_path: string;
  config_yaml?: string;
  status: "pending" | "running" | "paused" | "completed" | "failed" | "starting";
  progress: number;
  current_step: number;
  total_steps: number;
  loss?: number;
  learning_rate?: number;
  output_dir: string;
  checkpoint_paths: string[];
  log_file?: string;
  error_message?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  updated_at: string;
}

export interface TrainingRunCreateRequest {
  dataset_id: number;
  run_name?: string;  // Optional - will use UUID if not provided
  training_method: "lora" | "full_finetune";
  base_model_path: string;
  total_steps?: number;  // Mutually exclusive with epochs
  epochs?: number;  // Mutually exclusive with total_steps
  batch_size?: number;
  learning_rate?: number;
  lr_scheduler?: string;
  optimizer?: string;
  lora_rank?: number;
  lora_alpha?: number;
  network_type?: string;
  save_every?: number;
  sample_every?: number;
  sample_prompts?: string[];
}

export interface TrainingRunListResponse {
  runs: TrainingRun[];
  total: number;
}

export interface TrainingStatus {
  status: string;
  progress: number;
  current_step: number;
  total_steps: number;
  loss?: number;
  learning_rate?: number;
}

export const createTrainingRun = async (data: TrainingRunCreateRequest): Promise<TrainingRun> => {
  const response = await api.post("/training/runs", data);
  return response.data;
};

export const listTrainingRuns = async (): Promise<TrainingRunListResponse> => {
  const response = await api.get("/training/runs");
  return response.data;
};

export const getTrainingRun = async (id: number): Promise<TrainingRun> => {
  const response = await api.get(`/training/runs/${id}`);
  return response.data;
};

export const deleteTrainingRun = async (id: number): Promise<void> => {
  await api.delete(`/training/runs/${id}`);
};

export const startTrainingRun = async (id: number): Promise<{ message: string; run: TrainingRun }> => {
  console.log(`[API] startTrainingRun(${id}): Making POST request to /training/runs/${id}/start`);
  try {
    const response = await api.post(`/training/runs/${id}/start`);
    console.log(`[API] startTrainingRun(${id}): Response received:`, response.data);
    return response.data;
  } catch (error) {
    console.error(`[API] startTrainingRun(${id}): Error:`, error);
    throw error;
  }
};

export const stopTrainingRun = async (id: number): Promise<{ message: string; run: TrainingRun }> => {
  const response = await api.post(`/training/runs/${id}/stop`);
  return response.data;
};

export const getTrainingStatus = async (id: number): Promise<TrainingStatus> => {
  const response = await api.get(`/training/runs/${id}/status`);
  return response.data;
};

// TensorBoard API
export interface TensorBoardStatus {
  is_running: boolean;
  url?: string;
  port?: number;
}

export const startTensorBoard = async (runId: number): Promise<{ status: string; port: number; url: string }> => {
  const response = await api.post(`/training/runs/${runId}/tensorboard/start`);
  return response.data;
};

export const stopTensorBoard = async (runId: number): Promise<{ status: string }> => {
  const response = await api.delete(`/training/runs/${runId}/tensorboard/stop`);
  return response.data;
};

export const getTensorBoardStatus = async (runId: number): Promise<TensorBoardStatus> => {
  const response = await api.get(`/training/runs/${runId}/tensorboard/status`);
  return response.data;
};
