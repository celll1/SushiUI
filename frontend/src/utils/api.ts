import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  headers: {
    "Content-Type": "application/json",
  },
  // Set a very long timeout for generation requests (10 minutes)
  // Set to 0 to disable timeout completely
  timeout: 600000, // 10 minutes in milliseconds
});

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
  console.log('[API] localStorage key:', IMAGE_STORAGE_KEY, 'stored:', stored);
  const imageRefs: { [index: number]: string } = stored ? JSON.parse(stored) : {};
  console.log('[API] imageRefs:', imageRefs);

  const loadedControlnets = await Promise.all(
    controlnets.map(async (cn, index) => {
      // If use_input_image is true, don't need to load image (backend will use input image)
      if (cn.use_input_image) {
        console.log(`[API] ControlNet ${index}: use_input_image=true, skipping image load`);
        return cn;
      }

      const imageRef = imageRefs[index];
      console.log(`[API] ControlNet ${index}: imageRef =`, imageRef);
      if (imageRef) {
        const imageData = await loadTempImage(imageRef);
        console.log(`[API] ControlNet ${index}: loaded image data length =`, imageData?.length);
        const base64Data = imageData.startsWith('data:')
          ? imageData.split(',')[1]
          : imageData;
        console.log(`[API] ControlNet ${index}: base64 length =`, base64Data?.length);
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
}

export const generateTxt2Img = async (params: GenerationParams) => {
  const paramsWithImages = {
    ...params,
    controlnets: await loadControlNetImages(params.controlnets, "txt2img_controlnet_collapsed"),
  };

  const response = await api.post("/generate/txt2img", paramsWithImages);
  return response.data;
};

export const generateImg2Img = async (params: Img2ImgParams, image: File | string) => {
  const paramsWithImages = {
    ...params,
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
  formData.append("width", String(paramsWithImages.width || 1024));
  formData.append("height", String(paramsWithImages.height || 1024));
  formData.append("resize_mode", paramsWithImages.resize_mode || "image");
  formData.append("resampling_method", paramsWithImages.resampling_method || "lanczos");
  formData.append("loras", JSON.stringify(paramsWithImages.loras || []));
  formData.append("controlnets", JSON.stringify(paramsWithImages.controlnets || []));

  const response = await api.post("/generate/img2img", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const generateInpaint = async (params: InpaintParams, image: File | string, mask: File | string) => {
  const paramsWithImages = {
    ...params,
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
  formData.append("width", String(paramsWithImages.width || 1024));
  formData.append("height", String(paramsWithImages.height || 1024));
  formData.append("inpaint_full_res", String(paramsWithImages.inpaint_full_res || false));
  formData.append("inpaint_full_res_padding", String(paramsWithImages.inpaint_full_res_padding || 32));
  formData.append("inpaint_fill_mode", paramsWithImages.inpaint_fill_mode || "original");
  formData.append("inpaint_fill_strength", String(paramsWithImages.inpaint_fill_strength ?? 1.0));
  formData.append("resize_mode", paramsWithImages.resize_mode || "image");
  formData.append("resampling_method", paramsWithImages.resampling_method || "lanczos");
  formData.append("loras", JSON.stringify(paramsWithImages.loras || []));
  formData.append("controlnets", JSON.stringify(paramsWithImages.controlnets || []));

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
  tag_length?: string;  // very_short, short, long, very_long
  nl_length?: string;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_new_tokens?: number;
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

export default api;
