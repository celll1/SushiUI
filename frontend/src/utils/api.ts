import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  headers: {
    "Content-Type": "application/json",
  },
});

export interface LoRAConfig {
  path: string;
  strength: number;
  apply_to_text_encoder: boolean;
  apply_to_unet: boolean;
  unet_layer_weights: {
    down: number;
    mid: number;
    up: number;
  };
  step_range: [number, number];
}

export interface GenerationParams {
  prompt: string;
  negative_prompt?: string;
  steps?: number;
  cfg_scale?: number;
  sampler?: string;
  schedule_type?: string;
  seed?: number;
  width?: number;
  height?: number;
  model?: string;
  loras?: LoRAConfig[];
}

export interface Img2ImgParams extends GenerationParams {
  denoising_strength?: number;
  resize_mode?: string;
  resampling_method?: string;
}

export interface InpaintParams extends GenerationParams {
  denoising_strength?: number;
  mask_blur?: number;
  inpaint_full_res?: boolean;
  inpaint_full_res_padding?: number;
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
  width: number;
  height: number;
  generation_type: string;
  parameters: any;
  created_at: string;
  is_favorite: boolean;
}

export const generateTxt2Img = async (params: GenerationParams) => {
  const response = await api.post("/generate/txt2img", params);
  return response.data;
};

export const generateImg2Img = async (params: Img2ImgParams, image: File | string) => {
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

  formData.append("prompt", params.prompt);
  formData.append("negative_prompt", params.negative_prompt || "");
  formData.append("steps", String(params.steps || 20));
  formData.append("cfg_scale", String(params.cfg_scale || 7.0));
  formData.append("denoising_strength", String(params.denoising_strength || 0.75));
  formData.append("sampler", params.sampler || "euler");
  formData.append("schedule_type", params.schedule_type || "uniform");
  formData.append("seed", String(params.seed || -1));
  formData.append("width", String(params.width || 1024));
  formData.append("height", String(params.height || 1024));
  formData.append("resize_mode", params.resize_mode || "image");
  formData.append("resampling_method", params.resampling_method || "lanczos");
  formData.append("loras", JSON.stringify(params.loras || []));

  const response = await api.post("/generate/img2img", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const generateInpaint = async (params: InpaintParams, image: File | string, mask: File | string) => {
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

  formData.append("prompt", params.prompt);
  formData.append("negative_prompt", params.negative_prompt || "");
  formData.append("steps", String(params.steps || 20));
  formData.append("cfg_scale", String(params.cfg_scale || 7.0));
  formData.append("denoising_strength", String(params.denoising_strength || 0.75));
  formData.append("mask_blur", String(params.mask_blur || 4));
  formData.append("sampler", params.sampler || "euler");
  formData.append("schedule_type", params.schedule_type || "uniform");
  formData.append("seed", String(params.seed || -1));
  formData.append("width", String(params.width || 1024));
  formData.append("height", String(params.height || 1024));
  formData.append("inpaint_full_res", String(params.inpaint_full_res || false));
  formData.append("inpaint_full_res_padding", String(params.inpaint_full_res_padding || 32));
  formData.append("resize_mode", params.resize_mode || "image");
  formData.append("resampling_method", params.resampling_method || "lanczos");
  formData.append("loras", JSON.stringify(params.loras || []));

  const response = await api.post("/generate/inpaint", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const getImages = async (skip = 0, limit = 50, search?: string) => {
  const params = new URLSearchParams();
  params.append("skip", String(skip));
  params.append("limit", String(limit));
  if (search) params.append("search", search);

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

export default api;
