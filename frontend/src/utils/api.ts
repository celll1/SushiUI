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

export interface ModelInfo {
  source_type: string;
  source: string;
  type: "sd15" | "sdxl" | "zimage" | "flux2" | "anima" | "lens" | "ideogram4" | "minit2i";  // DEUS support removed
  is_v_prediction: boolean;
  model_hash: string;
  // Model-list entry fields (from GET /models)
  name?: string;
  path?: string;
  architecture?: string;
  vae_type?: string;  // MiniT2I: "none" (pixel) | "sdxl" | "flux1" (latent)
}

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
  is_reference_guide?: boolean;
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
  // Text Encoder Quantization (Z-Image only)
  text_encoder_quantization?: string | null;
  // CPU Text Encoding: run text encoder on CPU to save VRAM (slower)
  cpu_text_encoding?: boolean;
  // torch.compile optimization
  use_torch_compile?: boolean;
  // TIPO prompt upsampling
  use_tipo?: boolean;
  tipo_config?: any;  // TIPO configuration object
  // Preview mode
  preview_predicted_x0?: boolean;  // Show predicted x0 instead of current latent in preview
  preview_decoder?: string;  // Live-preview decoder for FLUX.2-VAE models: "matrix" | "taef2"
  // Z-Image specific
  max_sequence_length?: number;
  // Block Swap (Z-Image Transformer offloading)
  enable_block_swap?: boolean;
  blocks_to_swap?: number;
  use_pinned_memory?: boolean;
  // FLUX.2 Image Edit (reference images for sequence conditioning)
  ref_images?: File[];
  // SigLIP2 Vision Encoder path (SDXL/SD1.5 reference image conditioning)
  vision_encoder_path?: string | null;
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
  ref_images?: string[]; // FLUX.2 Image Edit: Reference image hashes
  vision_encoder_name?: string;   // SigLIP2 Vision Encoder filename
  vision_encoder_hash?: string;   // SHA256 hash of Vision Encoder model
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

// ---------------------------------------------------------------------------
// Schema defaults — fetched once at startup, backend is source of truth
// ---------------------------------------------------------------------------

export interface GenerationDefaultsResponse {
  txt2img: Partial<GenerationParams> & Record<string, unknown>;
  img2img: Partial<GenerationParams> & Record<string, unknown>;
  inpaint:  Partial<GenerationParams> & Record<string, unknown>;
}

export const fetchGenerationDefaults = async (): Promise<GenerationDefaultsResponse> =>
  (await api.get("/schema/generation-defaults")).data;

export const fetchTrainingDefaults = async (): Promise<Record<string, unknown>> =>
  (await api.get("/schema/training-defaults")).data;

export const fetchTaggerTrainingDefaults = async (): Promise<Record<string, unknown>> =>
  (await api.get("/schema/tagger-training-defaults")).data;

// Per-architecture default timestep_sampling configs (e.g. { _default: {...}, minit2i: {...} }).
// The training UI applies the selected model's entry when the base model changes.
export const fetchTimestepDefaultsByArch = async (): Promise<Record<string, Record<string, unknown>>> =>
  (await api.get("/schema/timestep-defaults-by-arch")).data;

export const generateTxt2Img = async (params: GenerationParams) => {
  // Get attention_type from localStorage
  const attentionType = typeof window !== 'undefined' ? localStorage.getItem('attention_type') : null;

  // Only load ControlNet images if they exist (avoid unnecessary localStorage access)
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "txt2img_controlnet_collapsed")
    : params.controlnets;

  const paramsWithImages = {
    ...params,
    attention_type: attentionType || 'normal',
    controlnets: controlnets,
  };

  const formData = new FormData();

  formData.append("prompt", paramsWithImages.prompt);
  formData.append("negative_prompt", paramsWithImages.negative_prompt || "");
  formData.append("steps", String(paramsWithImages.steps || 20));
  formData.append("cfg_scale", String(paramsWithImages.cfg_scale !== undefined ? paramsWithImages.cfg_scale : 7.0));
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

  // Quantization
  if (paramsWithImages.unet_quantization && paramsWithImages.unet_quantization !== "none") {
    formData.append("unet_quantization", paramsWithImages.unet_quantization);
  }
  if (paramsWithImages.text_encoder_quantization && paramsWithImages.text_encoder_quantization !== "none") {
    formData.append("text_encoder_quantization", paramsWithImages.text_encoder_quantization);
  }
  formData.append("cpu_text_encoding", String(paramsWithImages.cpu_text_encoding ?? false));

  // torch.compile optimization
  formData.append("use_torch_compile", String(paramsWithImages.use_torch_compile ?? false));

  // TIPO prompt upsampling
  formData.append("use_tipo", String(paramsWithImages.use_tipo ?? false));
  formData.append("tipo_config", JSON.stringify(paramsWithImages.tipo_config || {}));

  // Preview mode (predicted x0)
  formData.append("preview_predicted_x0", String(paramsWithImages.preview_predicted_x0 ?? false));
  formData.append("preview_decoder", String(paramsWithImages.preview_decoder ?? "matrix"));

  // Block Swap (Z-Image Transformer offloading)
  formData.append("enable_block_swap", String(paramsWithImages.enable_block_swap ?? false));
  formData.append("blocks_to_swap", String(paramsWithImages.blocks_to_swap ?? 20));
  formData.append("use_pinned_memory", String(paramsWithImages.use_pinned_memory ?? false));

  // FLUX.2 Image Edit / Vision Encoder (reference images)
  if (paramsWithImages.ref_images && paramsWithImages.ref_images.length > 0) {
    for (let i = 0; i < paramsWithImages.ref_images.length; i++) {
      formData.append("ref_images", paramsWithImages.ref_images[i]);
    }
  }

  // SigLIP2 Vision Encoder path
  if (paramsWithImages.vision_encoder_path) {
    formData.append("vision_encoder_path", paramsWithImages.vision_encoder_path);
  }

  const response = await api.post("/generate/txt2img", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};


// ---------------------------------------------------------------------------
// Training-preview generation (in-training LoRA / Full-FT model)
// ---------------------------------------------------------------------------
// Sends a JSON body to the new /generate/txt2img/training-preview endpoint;
// the backend writes a request file in the active training run's output_dir
// and the trainer subprocess picks it up at the next batch boundary.  The
// response is a PNG blob plus a few X-Preview-* headers.

export interface ActiveTrainingInfo {
  run_id: number;
  run_name?: string;
  training_method?: "lora" | "full" | string;
  current_step?: number;
  is_running: boolean;
}

export interface TrainingPreviewParams extends GenerationParams {
  /** Optional explicit run to target.  Backend picks the (sole) active
   *  run if omitted; returns 409 when multiple are active. */
  run_id?: number;
  /** When true, the backend writes the result PNG into outputs/ and
   *  inserts a GeneratedImage row so it appears in the gallery.  The
   *  DB row is tagged with model_name = "training-preview:<run>@step<N>".
   *  Default false (preview blob is transient). */
  save_to_gallery?: boolean;
}

/** Returns { blob, seed, runId } for the rendered image. */
export const generateTxt2ImgTrainingPreview = async (
  params: TrainingPreviewParams,
): Promise<{ blob: Blob; seed?: string; runId?: string; requestId?: string; filename?: string }> => {
  // Attention type honours the local toggle, same as regular generate
  const attentionType = typeof window !== 'undefined'
    ? localStorage.getItem('attention_type') : null;
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "txt2img_controlnet_collapsed")
    : params.controlnets;

  const body = {
    ...params,
    attention_type: attentionType || 'normal',
    controlnets: controlnets || [],
    loras: params.loras || [],
  };

  const response = await api.post("/generate/txt2img/training-preview", body, {
    responseType: "blob",
  });
  return {
    blob: response.data as Blob,
    seed:      response.headers["x-preview-seed"]      as string | undefined,
    runId:     response.headers["x-preview-run-id"]    as string | undefined,
    requestId: response.headers["x-preview-request"]   as string | undefined,
    // Present when save_to_gallery=true and the gallery save succeeded.
    // The URL ``/outputs/<filename>`` is then a stable reference that
    // survives page reload.
    filename:  response.headers["x-preview-filename"]  as string | undefined,
  };
};

/** Lightweight active-training probe.  Returns null when nothing is running.
 *  Used by the generate panel to enable / disable the "Use training model"
 *  toggle. */
export const getActiveTraining = async (): Promise<ActiveTrainingInfo | null> => {
  try {
    const res = await api.get("/training/active");
    return res.data as ActiveTrainingInfo;
  } catch (e: unknown) {
    // 404 / no active run → return null silently
    return null;
  }
};


// img2img and inpaint variants — same JSON-body pattern as txt2img
// but with base64-encoded init / mask images.

export interface Img2ImgTrainingPreviewParams extends TrainingPreviewParams {
  init_image_base64: string;       // raw base64 or data-URL form, both OK
  denoising_strength?: number;
}

export interface InpaintTrainingPreviewParams extends Img2ImgTrainingPreviewParams {
  mask_image_base64: string;
}

/** Helper: turn a File / Blob / data-URL into a raw base64 string for
 *  the training-preview JSON body. */
export const toBase64 = async (src: File | Blob | string): Promise<string> => {
  if (typeof src === "string") {
    // Already a string — strip data-URL prefix if present, else assume raw base64
    return src.startsWith("data:") ? src.split(",", 2)[1] || "" : src;
  }
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => {
      const result = String(reader.result);
      resolve(result.startsWith("data:") ? result.split(",", 2)[1] || "" : result);
    };
    reader.readAsDataURL(src);
  });
};

export const generateImg2ImgTrainingPreview = async (
  params: Img2ImgTrainingPreviewParams,
): Promise<{ blob: Blob; seed?: string; runId?: string; requestId?: string; filename?: string }> => {
  const attentionType = typeof window !== 'undefined'
    ? localStorage.getItem('attention_type') : null;
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "img2img_controlnet_collapsed")
    : params.controlnets;
  const body = {
    ...params,
    attention_type: attentionType || 'normal',
    controlnets: controlnets || [],
    loras: params.loras || [],
  };
  const response = await api.post("/generate/img2img/training-preview", body, {
    responseType: "blob",
  });
  return {
    blob: response.data as Blob,
    seed:      response.headers["x-preview-seed"]      as string | undefined,
    runId:     response.headers["x-preview-run-id"]    as string | undefined,
    requestId: response.headers["x-preview-request"]   as string | undefined,
    // Present when save_to_gallery=true and the gallery save succeeded.
    // The URL ``/outputs/<filename>`` is then a stable reference that
    // survives page reload.
    filename:  response.headers["x-preview-filename"]  as string | undefined,
  };
};

export const generateInpaintTrainingPreview = async (
  params: InpaintTrainingPreviewParams,
): Promise<{ blob: Blob; seed?: string; runId?: string; requestId?: string; filename?: string }> => {
  const attentionType = typeof window !== 'undefined'
    ? localStorage.getItem('attention_type') : null;
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "inpaint_controlnet_collapsed")
    : params.controlnets;
  const body = {
    ...params,
    attention_type: attentionType || 'normal',
    controlnets: controlnets || [],
    loras: params.loras || [],
  };
  const response = await api.post("/generate/inpaint/training-preview", body, {
    responseType: "blob",
  });
  return {
    blob: response.data as Blob,
    seed:      response.headers["x-preview-seed"]      as string | undefined,
    runId:     response.headers["x-preview-run-id"]    as string | undefined,
    requestId: response.headers["x-preview-request"]   as string | undefined,
    // Present when save_to_gallery=true and the gallery save succeeded.
    // The URL ``/outputs/<filename>`` is then a stable reference that
    // survives page reload.
    filename:  response.headers["x-preview-filename"]  as string | undefined,
  };
};


export const generateImg2Img = async (params: Img2ImgParams, image: File | string) => {
  // Get attention_type from localStorage
  const attentionType = typeof window !== 'undefined' ? localStorage.getItem('attention_type') : null;

  // Only load ControlNet images if they exist (avoid unnecessary localStorage access)
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "img2img_controlnet_collapsed")
    : params.controlnets;

  const paramsWithImages = {
    ...params,
    attention_type: attentionType || 'normal',
    controlnets: controlnets,
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
  formData.append("cfg_scale", String(paramsWithImages.cfg_scale !== undefined ? paramsWithImages.cfg_scale : 7.0));
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

  // CPU text encoding
  formData.append("cpu_text_encoding", String(paramsWithImages.cpu_text_encoding ?? false));

  // torch.compile optimization
  formData.append("use_torch_compile", String(paramsWithImages.use_torch_compile ?? false));

  // TIPO prompt upsampling
  formData.append("use_tipo", String(paramsWithImages.use_tipo ?? false));
  formData.append("tipo_config", JSON.stringify(paramsWithImages.tipo_config || {}));

  // Preview mode (predicted x0)
  formData.append("preview_predicted_x0", String(paramsWithImages.preview_predicted_x0 ?? false));
  formData.append("preview_decoder", String(paramsWithImages.preview_decoder ?? "matrix"));

  // FLUX.2 Image Edit / Vision Encoder (reference images)
  if (paramsWithImages.ref_images && paramsWithImages.ref_images.length > 0) {
    for (let i = 0; i < paramsWithImages.ref_images.length; i++) {
      formData.append("ref_images", paramsWithImages.ref_images[i]);
    }
  }

  // SigLIP2 Vision Encoder path
  if (paramsWithImages.vision_encoder_path) {
    formData.append("vision_encoder_path", paramsWithImages.vision_encoder_path);
  }

  const response = await api.post("/generate/img2img", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const generateInpaint = async (params: InpaintParams, image: File | string, mask: File | string) => {
  // Get attention_type from localStorage
  const attentionType = typeof window !== 'undefined' ? localStorage.getItem('attention_type') : null;

  // Only load ControlNet images if they exist (avoid unnecessary localStorage access)
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "inpaint_controlnet_collapsed")
    : params.controlnets;

  const paramsWithImages = {
    ...params,
    attention_type: attentionType || 'normal',
    controlnets: controlnets,
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
  formData.append("cfg_scale", String(paramsWithImages.cfg_scale !== undefined ? paramsWithImages.cfg_scale : 7.0));
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

  // CPU text encoding
  formData.append("cpu_text_encoding", String(paramsWithImages.cpu_text_encoding ?? false));

  // torch.compile optimization
  formData.append("use_torch_compile", String(paramsWithImages.use_torch_compile ?? false));

  // TIPO prompt upsampling
  formData.append("use_tipo", String(paramsWithImages.use_tipo ?? false));
  formData.append("tipo_config", JSON.stringify(paramsWithImages.tipo_config || {}));

  // Preview mode (predicted x0)
  formData.append("preview_predicted_x0", String(paramsWithImages.preview_predicted_x0 ?? false));
  formData.append("preview_decoder", String(paramsWithImages.preview_decoder ?? "matrix"));

  // FLUX.2 Image Edit / Vision Encoder (reference images)
  if (paramsWithImages.ref_images && paramsWithImages.ref_images.length > 0) {
    for (let i = 0; i < paramsWithImages.ref_images.length; i++) {
      formData.append("ref_images", paramsWithImages.ref_images[i]);
    }
  }

  // SigLIP2 Vision Encoder path
  if (paramsWithImages.vision_encoder_path) {
    formData.append("vision_encoder_path", paramsWithImages.vision_encoder_path);
  }

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

// Create a from-scratch MiniT2I model (latent or pixel) for Full-FT training.
// variant: "b16" | "l16"; vaeType: "sdxl" | "flux1" | "none" (none = pixel-space).
export const createScratchMiniT2I = async (
  variant: string,
  vaeType: string,
  name: string,
  targetDir?: string,
) => {
  const response = await api.post("/models/minit2i/create-scratch", {
    variant,
    vae_type: vaeType,
    name,
    target_dir: targetDir ?? null,
  });
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
  treat_as_nl?: boolean;  // Treat input as natural language instead of tags
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

// ─── SigLIP2 Tagger ───────────────────────────────────────────────────────────

export interface SigLIP2LoadRequest {
  checkpoint_path: string;
  vision_encoder_path?: string;
  vocab_path: string;
  lora_rank?: number;
  lora_alpha?: number;
}

export interface SigLIP2TagResult {
  tag: string;
  prob: number;          // raw sigmoid probability (always)
  raw_prob?: number;     // alias for prob (backward compat)
  cal_prob?: number;     // Jeffreys-calibrated probability (present when calibration table available)
  category: string;
}

export interface SigLIP2PredictResponse {
  tags: SigLIP2TagResult[];
  quality_top: SigLIP2TagResult | null;
  rating_top: SigLIP2TagResult | null;
  num_predicted: number;
  source?: string;
  run_id?: string;
  calibrated?: boolean;
  has_calibration?: boolean;    // true when cal_prob fields are present in results
  used_best_thr?: boolean;      // true when per-tag best_thr was used
  ood_distance?: number | null; // Mahalanobis distance (null when OOD detection not used)
}

export interface SigLIP2StatusResponse {
  loaded: boolean;
  checkpoint_path: string;
  vocab_path: string;
  model_type: string;
  num_tags: number;
  lr_matrix_loaded?: boolean;
  has_tag_metrics?: boolean;
  has_ood_reference?: boolean;
  calib_method?: string;
  calib_eps?: number;
  calib_prior_strength?: number;
}

export interface SigLIP2CalibrationSettings {
  method: "jeffreys" | "beta_bb";
  eps: number;
  prior_strength: number;
  has_tag_metrics?: boolean;
}

export const getCalibrationSettings = async (): Promise<SigLIP2CalibrationSettings> => {
  const response = await api.get("/tagger/siglip2/calibration");
  return response.data;
};

export const setCalibrationSettings = async (
  settings: Omit<SigLIP2CalibrationSettings, "has_tag_metrics">
): Promise<SigLIP2CalibrationSettings> => {
  const response = await api.post("/tagger/siglip2/calibration", settings);
  return response.data;
};

export interface TagMetricsData {
  n_tags: number;
  total_images: number;
  hard_lo: number;
  hard_hi: number;
  tag_names: string[];
  categories: string[];
  n_pos: (number | null)[];
  n_neg: (number | null)[];
  global_freq: (number | null)[];
  hard_rate: (number | null)[];
  fp_rate_50: (number | null)[];
  fn_rate_50: (number | null)[];
  best_f1: (number | null)[];
  best_thr: (number | null)[];
}

export const fetchTagMetrics = async (): Promise<TagMetricsData> => {
  const response = await api.get("/tagger/siglip2/tag-metrics");
  return response.data as TagMetricsData;
};

export type SigLIP2ContextMethod = "none" | "head_sim" | "lr_matrix";

export interface SigLIP2PredictOptions {
  known_tags_pos?: string[];
  known_tags_neg?: string[];
  context_method?: SigLIP2ContextMethod;
  context_lambda?: number;
  use_training_model?: boolean;
  use_calibration?: boolean;       // legacy
  use_per_tag_threshold?: boolean; // new: filter by per-tag best_thr
  display_calibration?: boolean;   // new: show calibrated probs in display
  min_best_thr?: number;           // clamp floor for best_thr (default 0.30)
  min_best_f1?: number;            // skip tags with best_f1 below this (default 0.05)
  use_ood_detection?: boolean;     // raise threshold for OOD images (requires OOD reference)
}

export const loadSigLIP2Model = async (req: SigLIP2LoadRequest) => {
  const response = await api.post("/tagger/siglip2/load", req);
  return response.data as { status: string; model_type: string; num_tags: number };
};

export const predictSigLIP2Tags = async (
  image_base64: string,
  threshold: number = 0.5,
  options?: SigLIP2PredictOptions,
): Promise<SigLIP2PredictResponse> => {
  const body: Record<string, unknown> = { image_base64, threshold };
  if (options?.known_tags_pos && options.known_tags_pos.length > 0) {
    body.known_tags_pos = options.known_tags_pos;
  }
  if (options?.known_tags_neg && options.known_tags_neg.length > 0) {
    body.known_tags_neg = options.known_tags_neg;
  }
  if (options?.context_method) {
    body.context_method = options.context_method;
  }
  if (typeof options?.context_lambda === "number") {
    body.context_lambda = options.context_lambda;
  }
  if (options?.use_training_model) {
    body.use_training_model = true;
  }
  if (typeof options?.use_calibration === "boolean") {
    body.use_calibration = options.use_calibration;
  }
  if (options?.use_per_tag_threshold) {
    body.use_per_tag_threshold = true;
  }
  if (options?.display_calibration) {
    body.display_calibration = true;
  }
  if (typeof options?.min_best_thr === "number") {
    body.min_best_thr = options.min_best_thr;
  }
  if (typeof options?.min_best_f1 === "number") {
    body.min_best_f1 = options.min_best_f1;
  }
  if (options?.use_ood_detection) {
    body.use_ood_detection = true;
  }
  const response = await api.post("/tagger/siglip2/predict", body);
  return response.data;
};

export const buildSigLIP2OodReference = async (
  image_dir: string,
  max_images: number = 2000,
): Promise<{ n_images: number; n_errors: number; p50: number; p95: number; save_path: string }> => {
  const response = await api.post("/tagger/siglip2/build-ood-reference", { image_dir, max_images });
  return response.data;
};

export const getSigLIP2Status = async (): Promise<SigLIP2StatusResponse> => {
  const response = await api.get("/tagger/siglip2/status");
  return response.data;
};

export const unloadSigLIP2Model = async () => {
  const response = await api.post("/tagger/siglip2/unload");
  return response.data;
};

export const mergeSigLIP2LoRA = async (output_path: string) => {
  const response = await api.post("/tagger/siglip2/merge-lora", { output_path });
  return response.data as { saved_path: string };
};

export const exportSigLIP2ONNX = async (output_path: string, max_num_patches: number = 256, strip_unknown_tags: boolean = false, also_split: boolean = false, use_model_stem: boolean = false) => {
  const response = await api.post("/tagger/siglip2/export-onnx", { output_path, max_num_patches, strip_unknown_tags, also_split, use_model_stem });
  return response.data as { saved_path: string; vocab_path: string };
};

export interface SigLIP2CheckpointMeta {
  lora_rank?: number;
  lora_alpha?: number;
  num_tags?: number;
  training_method?: string;
  [key: string]: unknown;
}

export const getSigLIP2CheckpointMeta = async (path: string): Promise<SigLIP2CheckpointMeta> => {
  const response = await api.get("/tagger/siglip2/checkpoint-meta", { params: { path } });
  return response.data as SigLIP2CheckpointMeta;
};

export interface SigLIP2ExtractEncoderResponse {
  output_path: string;
  num_params: number;
  hidden_size: number;
  num_layers: number;
}

export const extractSigLIP2Encoder = async (
  repo_id: string,
  output_path: string,
  encoder_type: "vision" | "text" = "vision",
): Promise<SigLIP2ExtractEncoderResponse> => {
  const response = await api.post("/tagger/siglip2/extract-encoder", {
    repo_id,
    output_path,
    encoder_type,
  });
  return response.data;
};

export interface VocabularyData {
  num_tags: number;
  tag_to_idx: Record<string, number>;
  idx_to_tag: Record<string, string>;
  tag_to_category: Record<string, string>;
  categories: Record<string, string[]>;
}

export const getTaggerRunVocabulary = async (runId: string): Promise<VocabularyData> => {
  const response = await api.get(`/tagger-training/runs/${runId}/vocabulary`);
  return response.data as VocabularyData;
};

export const getSigLIP2LoadedVocabulary = async (): Promise<VocabularyData> => {
  const response = await api.get("/tagger/siglip2/vocabulary");
  return response.data as VocabularyData;
};

// ---------------------------------------------------------------------------
// Tagger Browser API
// Security: absolute paths are never sent to/from the client.
// All file operations use rel_path (relative to the server-side browser root).
// ---------------------------------------------------------------------------

/** Image entry returned by /tagger/browser/list — no absolute path field. */
export interface BrowserImageEntry {
  rel_path: string;
  has_tags: boolean;
  mtime: number;
  tags?: string[]; // present when include_tags=true
}

export interface BrowserListResponse {
  images: BrowserImageEntry[];
}

export interface BrowserTagsResponse {
  tags: string[];
  raw: string;
}

export type BrowserBatchEvent =
  | { type: "done"; i: number; total: number; rel_path: string; n_tags: number }
  | { type: "skip"; i: number; total: number; rel_path: string }
  | { type: "error"; i: number; total: number; rel_path: string; error: string }
  | { type: "complete"; total: number };

/** Set browser root by typed path. Returns display_name (folder basename only). */
export const browserSetDirectory = async (
  dir: string
): Promise<{ ok: boolean; display_name: string | null }> => {
  const response = await api.post("/tagger/browser/set-directory", { dir });
  return response.data as { ok: boolean; display_name: string | null };
};

/** Open native OS folder picker. Returns display_name (folder basename only). */
export const browserPickDirectory = async (): Promise<{
  ok: boolean;
  display_name: string | null;
}> => {
  const response = await api.post("/tagger/browser/pick-directory");
  return response.data as { ok: boolean; display_name: string | null };
};

/** List images under the active browser root. */
export const browserListImages = async (
  recursive = false,
  includeTags = false
): Promise<BrowserListResponse> => {
  const response = await api.get("/tagger/browser/list", {
    params: { recursive, include_tags: includeTags },
  });
  return response.data as BrowserListResponse;
};

export const browserGetTags = async (
  rel_path: string
): Promise<BrowserTagsResponse> => {
  const response = await api.get("/tagger/browser/tags", {
    params: { rel_path },
  });
  return response.data as BrowserTagsResponse;
};

export const browserSaveTags = async (
  rel_path: string,
  tags: string[]
): Promise<void> => {
  await api.post("/tagger/browser/tags", { rel_path, tags });
};

/** Build URL for an image served by rel_path. */
export const browserImageUrl = (rel_path: string, size = 0): string => {
  const encoded = encodeURIComponent(rel_path);
  return `/api/v1/tagger/browser/image?rel_path=${encoded}&size=${size}`;
};

export const browserBatchInfer = (
  rel_paths: string[],
  options: { overwrite?: boolean; use_ood_detection?: boolean },
  onProgress: (ev: BrowserBatchEvent) => void
): AbortController => {
  const ctrl = new AbortController();
  (async () => {
    try {
      const res = await fetch("/api/v1/tagger/browser/batch-infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rel_paths, ...options }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) return;
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          if (part.startsWith("data: ")) {
            try {
              onProgress(JSON.parse(part.slice(6)) as BrowserBatchEvent);
            } catch {
              // ignore malformed SSE
            }
          }
        }
      }
    } catch {
      // aborted or network error
    }
  })();
  return ctrl;
};

export const addTagToCategory = async (
  tag: string,
  category: string,
  count: number = 1
): Promise<{
  status: string;
  message: string;
  tag: string;
  category: string;
  count: number;
  json_updated: boolean;
  updated_datasets: number;
}> => {
  const response = await api.post("/tag-category/add", {
    tag,
    category,
    count
  });
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

export interface CaptionProcessingConfig {
  caption_types?: string[];  // Caption types to use for training (e.g., ["tags", "natural_language"]). Empty = auto-select.
  normalize_tags?: boolean;  // Normalize tags to standard format (default: true)
  category_order?: string[];  // Category order (e.g., ["Rating", "Quality", "Character", ...])
  caption_dropout_rate?: number;
  token_dropout_rate?: number;
  keep_tokens?: number;
  shuffle_tokens?: boolean;
  shuffle_per_epoch?: boolean;
  shuffle_keep_first_n?: number;
  shuffle_tag_groups?: string[];  // Tag groups to shuffle (e.g., ["Character", "General"])
  shuffle_groups_together?: boolean;  // Shuffle all groups together vs within each group
  tag_group_dir?: string;  // Directory containing tag group JSON files
  exclude_person_count_from_shuffle?: boolean;  // Exclude person count tags from shuffle
  tag_dropout_rate?: number;
  tag_dropout_per_epoch?: boolean;
  tag_dropout_keep_first_n?: number;
  tag_dropout_category_rates?: Record<string, number>;  // Per-category dropout rates
  tag_dropout_exclude_person_count?: boolean;
}

export interface Dataset {
  id: number;
  unique_id?: string;
  name: string;
  path: string;
  description?: string;
  recursive: boolean;
  read_exif: boolean;
  exif_caption_fields?: string[] | null;
  caption_processing?: CaptionProcessingConfig;
  reference_suffixes?: string[];
  target_suffixes?: string[];
  caption_suffixes_for_reference?: string[];
  total_items: number;
  total_captions: number;
  total_tags: number;
  has_tags_captions?: boolean;
  tag_statistics?: Record<string, { count: number }>;
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

export const updateCaptionProcessing = async (
  id: number,
  captionProcessing: CaptionProcessingConfig
): Promise<Dataset> => {
  const response = await api.patch(`/datasets/${id}/caption-processing`, {
    caption_processing: captionProcessing,
  });
  return response.data;
};

export const updateDatasetSuffixConfig = async (
  id: number,
  config: {
    reference_suffixes?: string[];
    target_suffixes?: string[];
    caption_suffixes_for_reference?: string[];
  }
): Promise<Dataset> => {
  const response = await api.patch(`/datasets/${id}/suffix-config`, config);
  return response.data;
};

export const updateDatasetExifConfig = async (
  id: number,
  config: {
    read_exif?: boolean;
    exif_caption_fields?: string[];
  }
): Promise<Dataset> => {
  const response = await api.patch(`/datasets/${id}/exif-config`, config);
  return response.data;
};

// ============================================================
// TXT File Synchronization API
// ============================================================

export interface SaveToTxtResponse {
  success: boolean;
  message: string;
}

export interface BulkSaveToTxtResponse {
  total: number;
  saved: number;
  skipped: number;
  errors: number;
}

export const saveItemCaptionToTxt = async (itemId: number): Promise<SaveToTxtResponse> => {
  const response = await api.post(`/datasets/items/${itemId}/save-to-txt`);
  return response.data;
};

export const saveAllCaptionsToTxt = async (datasetId: number): Promise<BulkSaveToTxtResponse> => {
  const response = await api.post(`/datasets/${datasetId}/save-all-to-txt`);
  return response.data;
};

export const restoreItemCaptionFromTxt = async (itemId: number): Promise<SaveToTxtResponse> => {
  const response = await api.post(`/datasets/items/${itemId}/restore-from-txt`);
  return response.data;
};

// ============================================================
// Caption Processing Presets API
// ============================================================

export interface CaptionProcessingPreset {
  id: number;
  name: string;
  description?: string;
  config: CaptionProcessingConfig;
  created_at?: string;
  updated_at?: string;
}

export const listCaptionProcessingPresets = async (): Promise<CaptionProcessingPreset[]> => {
  const response = await api.get("/caption-processing-presets");
  return response.data;
};

export const createCaptionProcessingPreset = async (
  name: string,
  description: string | null,
  config: CaptionProcessingConfig
): Promise<CaptionProcessingPreset> => {
  const response = await api.post("/caption-processing-presets", {
    name,
    description,
    config,
  });
  return response.data;
};

export const getCaptionProcessingPreset = async (id: number): Promise<CaptionProcessingPreset> => {
  const response = await api.get(`/caption-processing-presets/${id}`);
  return response.data;
};

export const updateCaptionProcessingPreset = async (
  id: number,
  updates: {
    name?: string;
    description?: string;
    config?: CaptionProcessingConfig;
  }
): Promise<CaptionProcessingPreset> => {
  const response = await api.patch(`/caption-processing-presets/${id}`, updates);
  return response.data;
};

export const deleteCaptionProcessingPreset = async (id: number): Promise<void> => {
  await api.delete(`/caption-processing-presets/${id}`);
};

export interface StructureDetectionResult {
  structure_type: "normal" | "paired";
  reference_suffixes: string[];
  target_suffixes: string[];
  caption_suffixes_for_reference: string[];
  confidence: number;
  unknown_suffixes?: string[];
  stats: {
    total_files_sampled: number;
    suffix_counts: Record<string, number>;
    paired_groups: number;
    unpaired_files: number;
  };
}

export interface ScanFieldStat {
  added: number;
  updated: number;
  images_with?: number;
}

export interface ScanFieldSummary {
  total_images: number;
  tags: ScanFieldStat;
  caption: ScanFieldStat;
  other: ScanFieldStat;
}

export interface DatasetScanResponse {
  items_found: number;
  captions_found: number;
  captions_updated?: number;
  items_purged?: number;
  field_summary?: ScanFieldSummary;
  dataset: Dataset;
  structure_detection?: StructureDetectionResult;
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
  related_images?: {
    reference?: string[];  // Reference image paths for training
    [key: string]: string[] | undefined;  // Extensible for future use
  };
}

export interface DatasetCaptionData {
  id: number;
  item_id: number;
  caption_type: string;
  content: string;
  field_category?: 'training' | 'metadata'; // Field category: training or metadata
  is_tags_format?: boolean; // True if tags format (Danbooru), false if natural language
  tag_match_rate?: number; // Tag match rate (0.0-1.0) for tags format detection
  source_field?: string; // JSON field path (e.g., "metrics.likes", "author")
  source: string;
  created_at: string;
  updated_at: string;
  tag_data?: Array<{ tag: string; category: string }>; // Pre-categorized tags
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
  search?: string,
  tags?: string // Comma-separated tags
): Promise<DatasetItemListResponse> => {
  const params: any = { page, page_size: pageSize };
  if (search) params.search = search;
  if (tags) params.tags = tags;
  const response = await api.get(`/datasets/${datasetId}/items`, { params });
  return response.data;
};

export const getDatasetItem = async (datasetId: number, itemId: number): Promise<DatasetItem> => {
  const response = await api.get(`/datasets/${datasetId}/items/${itemId}`);
  return response.data;
};

export const getAllDatasetItemIds = async (
  datasetId: number,
  search?: string,
  tags?: string
): Promise<{ item_ids: number[]; total: number }> => {
  const params: any = {};
  if (search) params.search = search;
  if (tags) params.tags = tags;
  const response = await api.get(`/datasets/${datasetId}/items/ids`, { params });
  return response.data;
};

export const getDatasetTags = async (datasetId: number): Promise<string[]> => {
  const response = await api.get(`/datasets/${datasetId}/tags`);
  return response.data.tags;
};

export interface CaptionSubtype {
  subtype: string;
  count: number;
}

export interface CaptionTypeInfo {
  caption_type: string;
  total_count: number;
  field_category: 'training' | 'metadata';
  is_tags_format: boolean;
  avg_match_rate: number;
  source_field?: string;
  subtypes: CaptionSubtype[];
}

export interface CaptionTypesResponse {
  caption_types: CaptionTypeInfo[];
}

export const getDatasetCaptionTypes = async (datasetId: number): Promise<CaptionTypesResponse> => {
  const response = await api.get(`/datasets/${datasetId}/caption-types`);
  return response.data;
};

export interface RandomCaptionResponse {
  caption: string;
  caption_type: string;
  caption_subtype?: string;
  item_id: number;
  reference_images?: string[];
}

export const getRandomCaption = async (
  datasetId: number,
  captionTypes?: string[]
): Promise<RandomCaptionResponse> => {
  const params: any = {};
  if (captionTypes && captionTypes.length > 0) {
    params.caption_types = captionTypes.join(",");
  }
  const response = await api.get(`/datasets/${datasetId}/random-caption`, { params });
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
  training_method: "lora" | "relora" | "full_finetune" | "controlnet";
  base_model_path: string;
  config_yaml?: string;
  status: "pending" | "running" | "paused" | "completed" | "failed" | "starting";
  progress: number;
  current_step: number;
  total_steps: number;
  phase?: string;  // "initializing", "latent_cache", "text_encoder_cache", "training"
  phase_progress?: number;  // 0-100
  phase_detail?: string;  // Detailed status message
  loss?: number;
  learning_rate?: number;
  output_dir: string;
  checkpoint_paths: string[];
  log_file?: string;
  error_message?: string;
  created_at: string;
  started_at?: string;
  last_resumed_at?: string;  // Last resume time (for accurate ETA calculation)
  resumed_from_step?: number;  // Step at resume (for accurate ETA calculation)
  completed_at?: string;
  updated_at: string;
}

export interface DatasetConfigItem {
  dataset_id: number;
  caption_types: string[];  // Empty = use all caption types
  filters: Record<string, any>;  // Filter configuration
  ve_reconstruction_mode?: boolean;  // Use training image as its own VE reference (no text conditioning)
}

export interface SamplePrompt {
  positive: string;
  negative: string;
  condition_image_path?: string;
  reference_image_path?: string;
}

export interface TrainingRunCreateRequest {
  dataset_id?: number;  // Deprecated - use dataset_configs instead
  dataset_configs?: DatasetConfigItem[];  // Multiple datasets with filters
  run_name?: string;  // Optional - will use UUID if not provided
  training_method: "lora" | "relora" | "full_finetune" | "controlnet";
  base_model_path: string;
  total_steps?: number;  // Mutually exclusive with epochs
  epochs?: number;  // Mutually exclusive with total_steps
  batch_size?: number;
  gradient_accumulation_steps?: number;
  max_grad_norm?: number;
  learning_rate?: number;
  lr_scheduler?: string;
  lr_warmup_steps?: number;
  optimizer?: string;
  lora_rank?: number;
  lora_alpha?: number;
  network_type?: string;
  save_every?: number;
  save_every_unit?: string;
  max_step_saves_to_keep?: number | null;
  sample_every?: number;
  sample_prompts?: SamplePrompt[];
  resume_from_checkpoint?: string | null;
  sample_width?: number;
  sample_height?: number;
  sample_steps?: number;
  sample_cfg_scale?: number;
  sample_sampler?: string;
  sample_schedule_type?: string;
  sample_seed?: number;
  debug_latents?: boolean;
  debug_latents_every?: number;
  enable_bucketing?: boolean;
  base_resolutions?: number[];
  bucket_strategy?: string;
  multi_resolution_mode?: string;
  train_unet?: boolean;
  train_text_encoder?: boolean;
  unet_lr?: number | null;
  text_encoder_lr?: number | null;
  text_encoder_1_lr?: number | null;
  text_encoder_2_lr?: number | null;
  weight_dtype?: string;
  training_dtype?: string;
  output_dtype?: string;
  vae_dtype?: string;
  mixed_precision?: boolean;
  use_flash_attention?: boolean;
  min_snr_gamma?: number;
  text_encoding_mode?: string;
  text_encoding_swap_interval?: number;
  latent_encoding_mode?: string;
  latent_encoding_swap_interval?: number;
  // MiniT2I
  minit2i_label_drop_rate?: number;
  minit2i_lr_factor?: number;
  minit2i_flan_t5_path?: string;
  minit2i_scratch_init_from?: string;  // from-scratch: inherit weights from this model
  // REPA (Representation Alignment) — MiniT2I only
  repa_enable?: boolean;
  repa_encoder_source?: string;        // "tagger" | "siglip2"
  repa_tagger_model_dir?: string;      // tagger model dir (empty = auto-pick)
  repa_siglip2_repo?: string;          // off-the-shelf SigLIP2 repo
  repa_align_depth?: number;           // -1 = auto (depth//3)
  repa_weight?: number;                // alignment loss weight (lambda)
  repa_proj_lr_factor?: number;        // projector LR multiplier (x unet_lr)
  repa_encoder_resolution?: number;    // 0 = follow encoder native image_size
  // Online Danbooru augmentation (image-generation training)
  danbooru_aug_enable?: boolean;
  danbooru_aug_queries?: string;
  danbooru_aug_weight_static?: number;
  danbooru_aug_deficiency_enable?: boolean;
  danbooru_aug_deficiency_min_count?: number;
  danbooru_aug_deficiency_top_k?: number;
  danbooru_aug_deficiency_manual?: string;
  danbooru_aug_weight_deficiency?: number;
  danbooru_aug_injection_interval?: number;
  danbooru_aug_injection_ratio?: number;
  danbooru_aug_min_score?: number;
  danbooru_aug_max_posts_per_query?: number;
  danbooru_aug_api_interval?: number;
  danbooru_aug_dl_speed_kbps?: number;
  danbooru_speed_check_enable?: boolean;
  danbooru_speed_degraded_kbps?: number;
  danbooru_speed_min_slow_streak?: number;
  danbooru_speed_min_slow_seconds?: number;
  danbooru_speed_cooldown_seconds?: number;
  danbooru_aug_buffer_size?: number | null;
  danbooru_aug_include_rating_tag?: boolean;
  danbooru_aug_max_caption_tags?: number;
  danbooru_quality_tag_enable?: boolean;
  danbooru_quality_tag_thresholds?: string;
  danbooru_quality_tag_attach_negative?: boolean;
  danbooru_aug_shuffle_tags?: boolean;
  danbooru_aug_shuffle_keep_first_n?: number;
  danbooru_aug_tag_dropout_rate?: number;
  danbooru_aug_tag_dropout_keep_first_n?: number;
  danbooru_aug_caption_dropout_rate?: number;
  danbooru_aug_keep_tokens?: number;
  blocks_to_swap?: number;
  use_pinned_memory?: boolean;
  num_optimizer_groups?: number;
  multi_noise_timesteps?: number;
  multi_noise_mode?: string;
  trajectory_blend_alpha?: number;
  timestep_sampling?: {
    distribution: string;
    min_timestep: number;
    max_timestep: number;
    // Distribution-specific parameters
    mean?: number;   // For logit_normal/normal
    std?: number;    // For logit_normal/normal
    alpha?: number;  // For beta
    beta?: number;   // For beta
  };
  cache_latents_to_disk?: boolean;
  force_recache?: boolean;
  use_reference_images?: boolean;
  // Vision Encoder (SigLIP2) — SD/SDXL only
  vision_encoder_path?: string | null;
  train_vision_encoder?: boolean;
  vision_encoder_lr?: number | null;
  gradient_routing_ve?: boolean;
  // Parameter change tracking
  param_tracking?: boolean;
  param_tracking_interval?: number;
  // ReLoRA-specific parameters
  relora_merge_every?: number;
  relora_merge_unit?: "steps" | "epochs";
  restart_warmup_steps?: number;
  optimizer_reset_strategy?: "full_reset" | "magnitude_pruning" | "random_pruning";
  optimizer_pruning_ratio?: number;
  // ControlNet-specific parameters
  controlnet_type?: string;
  controlnet_pretrained_path?: string | null;
  controlnet_init_from_unet?: boolean;
  lllite_conditioning_channels?: number;
  lllite_rank?: number;
  condition_preprocessors?: string[] | null;
  condition_cache_mode?: string;
  // Pre-flight dataset drift check + optional rescan + orphan latent
  // cache cleanup.  Modes:
  //   "off"   — skip
  //   "path"  — detect added/missing files only
  //   "smart" — path drift + caption sidecar mtime
  //   "force" — always rescan
  // Legacy boolean also accepted (true→"path", false→"off") for backwards
  // compatibility with older clients.
  rescan_before_training?: "off" | "path" | "smart" | "force" | boolean;
  // Optimizer hyperparameters
  optimizer_is_paged?: boolean;
  optimizer_cautious?: boolean;
  optimizer_beta1?: number;
  optimizer_beta2?: number;
  optimizer_epsilon?: number;
  optimizer_weight_decay?: number;
  optimizer_schedule_free?: boolean;
  optimizer_schedule_free_r?: number;
  optimizer_schedule_free_weight_lr_power?: number;
  optimizer_use_radam?: boolean;
  optimizer_stochastic_rounding?: boolean;
  // LoRA
  lora_dtype?: "fp32" | "fp16" | "bf16";
  // Component training (image encoder)
  train_image_encoder?: boolean;
  image_encoder_lr?: number | null;
  // Reconstruction loss
  reconstruction_loss_weight?: number;
  // Regularization
  regularization_type?: string | null;
  snr_regularization_weight?: number;
  snr_timestep_adaptive?: boolean;
  snr_penalty_mode?: string;
  energy_regularization_weight?: number;
  energy_timestep_adaptive?: boolean;
  energy_penalty_mode?: string;
  energy_normalize_by_pixels?: boolean;
  // Unified Training Framework
  noise_process?: string;
  prediction_target?: string;
  strict_validation?: boolean;
  // Anima-specific
  anima_lora_scope?: string;
  train_llm_adapter?: boolean;
  anima_attn_mlp_lr_factor?: number;
  anima_mod_lr_factor?: number;
  anima_llm_adapter_lr_factor?: number;
  // Lens-specific
  lens_lora_scope?: string;
  lens_img_lr_factor?: number;
  lens_txt_lr_factor?: number;
  // Priority training
  priority_training?: {
    entries: string[];
    multiplier: number;
  };
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

export const getTrainingRunParams = async (id: number): Promise<TrainingRunCreateRequest & { run_id: number }> => {
  const response = await api.get(`/training/runs/${id}/params`);
  return response.data;
};

export const updateTrainingRun = async (id: number, request: TrainingRunCreateRequest): Promise<TrainingRun> => {
  const response = await api.put(`/training/runs/${id}`, request);
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

/** Skip the dataset currently being rescanned during a LoRA/Full-FT run's
 *  pre-flight. Pass the dataset_id of the in-progress rescan so a stale skip
 *  (for a dataset that already finished) is ignored. */
export const skipTrainingRescan = async (
  id: number,
  datasetId?: number,
): Promise<{ skipped: boolean; current_dataset: number | null }> => {
  const response = await api.post(`/training/runs/${id}/skip-rescan`, { dataset_id: datasetId ?? null });
  return response.data;
};

/** Skip the dataset currently being rescanned during a tagger run's pre-flight. */
export const skipTaggerRescan = async (
  runId: string,
  datasetId?: number,
): Promise<{ skipped: boolean; current_dataset: number | null }> => {
  const response = await api.post(`/tagger-training/runs/${runId}/skip-rescan`, { dataset_id: datasetId ?? null });
  return response.data;
};

export const updateTrainingConfig = async (id: number, configYaml: string): Promise<{ message: string; run: TrainingRun }> => {
  const response = await api.patch(`/training/runs/${id}/config`, { config_yaml: configYaml });
  return response.data;
};

export const reloadTrainingConfig = async (id: number): Promise<{ message: string; run: TrainingRun }> => {
  const response = await api.post(`/training/runs/${id}/config/reload`);
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

// Training Metrics API
export interface MetricPoint {
  step: number;
  value: number;
  wall_time: number;
  /** Resume session this point belongs to (0 = initial run). Carried per-point so
   *  the chart can later split curves per resume; markers use resume_markers below. */
  resume_seq?: number;
}

/** Step at which an epoch ended (for dotted vertical boundary lines). */
export interface EpochBoundary {
  epoch: number;
  step: number;
}

/** First step of a resume session > 0 (for resume boundary markers). */
export interface ResumeMarker {
  resume_seq: number;
  step: number;
}

export interface TrainingMetrics {
  loss: MetricPoint[];
  recon_loss: MetricPoint[];
  repa_loss?: MetricPoint[];
  learning_rate: MetricPoint[];
  grad_norm: MetricPoint[];
  grad_norm_text_encoder: MetricPoint[];
  grad_norm_text_encoder_1: MetricPoint[];
  grad_norm_text_encoder_2: MetricPoint[];
  grad_norm_unet: MetricPoint[];
  grad_norm_vision_encoder: MetricPoint[];
  param_update_norm_unet?: MetricPoint[];
  param_update_norm_te1?: MetricPoint[];
  param_update_norm_te2?: MetricPoint[];
  param_update_norm_ve?: MetricPoint[];
  param_cumulative_drift_unet?: MetricPoint[];
  param_cumulative_drift_te1?: MetricPoint[];
  param_cumulative_drift_te2?: MetricPoint[];
  param_cumulative_drift_ve?: MetricPoint[];
  epoch_boundaries?: EpochBoundary[];
  resume_markers?: ResumeMarker[];
}

export const getTrainingMetrics = async (
  runId: number,
  maxPoints: number = 1000
): Promise<TrainingMetrics> => {
  const params: any = { max_points: maxPoints };
  // Use new DB endpoint with uniform sampling (backend handles sampling)
  const response = await api.get(`/training/runs/${runId}/metrics_db`, { params });
  return response.data;
};

export interface TrainingSampleImage {
  sample_index: number;
  path: string;
  params?: {
    prompt?: string;
    negative_prompt?: string;
    steps?: string;
    cfg_scale?: string;
    seed?: string;
    width?: string;
    height?: string;
    schedule_type?: string;
    condition_image_path?: string;
    reference_image_path?: string;
  };
}

export interface TrainingSampleStep {
  step: number;
  images: TrainingSampleImage[];
}

export interface TrainingSamplesResponse {
  samples: TrainingSampleStep[];
}

export const getTrainingSamples = async (runId: number): Promise<TrainingSamplesResponse> => {
  const response = await api.get(`/training/runs/${runId}/samples`);
  return response.data;
};

export interface DebugLatent {
  step: number;
  timestep: number;
  filename: string;
  path: string;
}

export interface DebugLatentsResponse {
  debug_latents: DebugLatent[];
}

export interface DebugLatentVisualization {
  step: number;
  timestep: number;
  loss: number;
  recon_loss?: number;  // Optional: may not exist in older debug data
  caption?: string;  // Processed caption used during training
  reference_image?: string;  // base64 thumbnail of reference image used in this training batch
  latents_image?: string;  // base64
  noisy_latents_image?: string;  // base64
  predicted_noise_image?: string;  // base64
  predicted_latent_image?: string;  // base64
}

export const getDebugLatents = async (runId: number): Promise<DebugLatentsResponse> => {
  const response = await api.get(`/training/runs/${runId}/debug-latents`);
  return response.data;
};

export const visualizeDebugLatent = async (
  runId: number,
  step: number,
  timestep?: number
): Promise<DebugLatentVisualization> => {
  const params = timestep !== undefined ? { timestep } : {};
  const response = await api.get(`/training/runs/${runId}/debug-latents/${step}/visualize`, { params });
  return response.data;
};

// ============================================================
// Training Presets API
// ============================================================

export interface TrainingPreset {
  id: number;
  name: string;
  description?: string;
  training_method: "lora" | "relora" | "full_finetune" | "controlnet";
  config: Record<string, any>;  // Training parameters (excluding dataset and model path)
  created_at: string;
  updated_at: string;
}

export interface TrainingPresetsResponse {
  presets: TrainingPreset[];
}

export interface TrainingPresetCreateRequest {
  name: string;
  description?: string;
  training_method: "lora" | "relora" | "full_finetune" | "controlnet";
  config: Record<string, any>;
}

export interface TrainingPresetUpdateRequest {
  name?: string;
  description?: string;
  config?: Record<string, any>;
}

export const listTrainingPresets = async (): Promise<TrainingPresetsResponse> => {
  const response = await api.get("/training/presets");
  return response.data;
};

export const getTrainingPreset = async (id: number): Promise<TrainingPreset> => {
  const response = await api.get(`/training/presets/${id}`);
  return response.data;
};

export const createTrainingPreset = async (data: TrainingPresetCreateRequest): Promise<TrainingPreset> => {
  const response = await api.post("/training/presets", data);
  return response.data;
};

export const updateTrainingPreset = async (id: number, data: TrainingPresetUpdateRequest): Promise<TrainingPreset> => {
  const response = await api.patch(`/training/presets/${id}`, data);
  return response.data;
};

export const deleteTrainingPreset = async (id: number): Promise<void> => {
  await api.delete(`/training/presets/${id}`);
};

// ============================================================
// Dataset Caption Update API
// ============================================================

export interface CaptionUpdateRequest {
  caption_type: string;
  content: string;
  tag_data?: Array<{ tag: string; category: string }>;
}

export const updateItemCaption = async (
  itemId: number,
  data: CaptionUpdateRequest
): Promise<{ status: string; caption: DatasetCaptionData }> => {
  const response = await api.patch(`/datasets/items/${itemId}/captions`, data);
  return response.data;
};

// ============================================================
// Reference Images API
// ============================================================

export interface ReferenceImagesResponse {
  status: string;
  item_id: number;
  reference_images: string[];
}

export const updateItemReferenceImages = async (
  itemId: number,
  referenceImages: string[]
): Promise<ReferenceImagesResponse> => {
  const response = await api.patch(`/datasets/items/${itemId}/reference-images`, {
    reference_images: referenceImages
  });
  return response.data;
};

export const addItemReferenceImage = async (
  itemId: number,
  imagePath: string
): Promise<ReferenceImagesResponse> => {
  const formData = new FormData();
  formData.append("image_path", imagePath);
  const response = await api.post(`/datasets/items/${itemId}/reference-images/add`, formData);
  return response.data;
};

export const removeItemReferenceImage = async (
  itemId: number,
  imagePath: string
): Promise<ReferenceImagesResponse> => {
  const response = await api.delete(`/datasets/items/${itemId}/reference-images`, {
    params: { image_path: imagePath }
  });
  return response.data;
};

// ============================================================
// Batch Operations API
// ============================================================

export interface BatchTaggerRequest {
  item_ids: number[];
  gen_threshold?: number;
  char_threshold?: number;
  thresholds?: Record<string, number>;
  model_version?: string;
  remove_below_threshold?: boolean;
  merge_with_existing?: boolean;
}

export interface BatchReorderTagsRequest {
  item_ids: number[];
  category_order: string[];
}

export interface BatchReplaceTagRequest {
  item_ids: number[];
  from_tag: string;
  to_tag: string;
  normalize_match?: boolean;
}

export interface BatchOperationResponse {
  status: string;
  processed_count: number;
  updated_count: number;
  skipped_count: number;
  failed_count: number;
  message: string;
}

export const batchTaggerInference = async (
  datasetId: number,
  request: BatchTaggerRequest
): Promise<BatchOperationResponse> => {
  const response = await api.post(`/datasets/${datasetId}/batch-tagger`, request);
  return response.data;
};

export const batchReorderTags = async (
  datasetId: number,
  request: BatchReorderTagsRequest
): Promise<BatchOperationResponse> => {
  const response = await api.post(`/datasets/${datasetId}/batch-reorder-tags`, request);
  return response.data;
};

export const batchReplaceTag = async (
  datasetId: number,
  request: BatchReplaceTagRequest
): Promise<BatchOperationResponse> => {
  const response = await api.post(`/datasets/${datasetId}/batch-replace-tag`, request);
  return response.data;
};

export const cancelBatchOperation = async (datasetId: number): Promise<{ message: string }> => {
  const response = await api.post(`/datasets/${datasetId}/batch-cancel`);
  return response.data;
};

export const backfillTagData = async (datasetId: number): Promise<BatchOperationResponse> => {
  const response = await api.post(`/datasets/${datasetId}/backfill-tag-data`);
  return response.data;
};

// Tag Dictionary Search API was removed - use tagSuggestions.ts instead

// ==================== Debug VRAM Inspection ====================

// ==================== Dataset Scan Preview ====================

export interface ScanPreviewGroup {
  group_name: string;
  images: Array<{ path: string; role: string }>;
  captions: Array<{ path: string; suffix: string; detected_type: string; content_preview?: string }>;
}

export interface ScanPreviewResult {
  total_groups: number;
  total_images: number;
  total_captions: number;
  detected_suffixes: Record<string, { count: number; sample_types: string[] }>;
  structure_type: string;
  sample_groups: ScanPreviewGroup[];
  dataset_path?: string;
}

export const scanDatasetPreview = async (datasetId: number): Promise<ScanPreviewResult> => {
  const response = await api.post(`/datasets/${datasetId}/scan/preview`);
  return response.data;
};

// ==================== Debug ====================

export const debugVramInspection = async () => {
  const response = await api.get("/debug/vram");
  return response.data;
};

export const debugVramForceRelease = async () => {
  const response = await api.post("/debug/vram/release");
  return response.data;
};

// ==================== Tagger Training ====================

export interface TaggerTrainingRun {
  run_id: string;
  run_name: string;
  status: "idle" | "pending" | "starting" | "running" | "completed" | "failed" | "stopped";
  progress: number;
  current_epoch: number;
  current_step: number;
  total_steps?: number | null;
  training_method: "full" | "lora";
  vision_encoder_path: string;
  dataset_configs: string[];
  config: Record<string, unknown>;
  num_tags: number;
  tag_vocabulary?: Record<string, unknown> | null;
  best_f1: number | null;
  best_threshold: number | null;
  threshold_f1_curve: Record<string, number> | null;
  latest_loss: number | null;
  head_checkpoint_path: string | null;
  lora_checkpoint_path: string | null;
  error_message: string | null;
  status_message: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;        // when the very first run started
  last_resumed_at?: string | null;   // when this session resumed (null if no resume)
  completed_at?: string | null;
}

export interface TaggerDatasetConfig {
  dataset_id: number;
  caption_types?: string[];
}

export interface TaggerTrainingRunCreateRequest {
  run_name: string;
  training_method: "full" | "lora";
  vision_encoder_path: string;
  dataset_configs: TaggerDatasetConfig[];
  lora_rank?: number;
  lora_alpha?: number;
  learning_rate?: number;
  head_lr_multiplier?: number;
  optimizer?: string;
  warmup_steps?: number;
  epochs?: number;
  batch_size?: number;
  num_workers?: number;
  num_workers_override?: number | null;
  save_every_n_steps?: number;
  save_every_n_epochs?: number;
  keep_last_n_checkpoints?: number;
  checkpoint_save_mode?: string;
  mixed_precision?: string;
  use_flash_attention?: boolean;
  gradient_checkpointing?: boolean;
  loss_function?: string;
  loss_gamma_neg?: number;
  loss_gamma_pos?: number;
  loss_gamma0?: number;
  loss_m0?: number;
  loss_rho?: number;
  loss_beta?: number;
  loss_label_weight?: string;
  val_split_mode?: string;
  val_split?: number;
  val_fixed_size?: number;
  validate_every?: number;
  save_best_only?: boolean;
  excluded_categories?: string[];
  ban_tags?: string;
  use_tag_aliases?: boolean;
  save_base_model?: boolean;
  // Loss masking strategy for Quality tags when a sample has at least one
  // quality tag. "intra_group" (default) masks group siblings; "cross_group"
  // trains all non-positive quality tags as negatives (legacy).
  quality_masking_mode?: "intra_group" | "cross_group";
  weight_decay?: number;
  loss_clip?: number;
  vocab_min_count?: number;
  // Resolve "Unknown" tag categories against the Gelbooru taglist supplement
  // (taglist_gel/) in addition to Danbooru. Danbooru takes precedence.
  vocab_use_gelbooru_categories?: boolean;
  cls_dim?: number;
  hidden_proj_dim?: number;
  init_head_from?: string;
  // LR matrix (conditional inference) — built once at training start when enabled.
  build_lr_matrix_on_start?: boolean;
  lr_top_anchors?: number;
  lr_top_targets?: number;
  lr_threshold?: number;
  lr_min_anchor_count?: number;
  // Pre-flight dataset drift check + optional rescan.  Modes:
  // "off" | "path" | "smart" | "force".  Legacy bool accepted.
  rescan_before_training?: "off" | "path" | "smart" | "force" | boolean;
  // Training-time F1 metrics
  train_f1_eval_every_n_steps?: number;
  train_f1_threshold_search_every_n_steps?: number;
  train_f1_initial_threshold?: number;
  train_f1_buffer_batches?: number;
  // Online Danbooru augmentation
  enable_danbooru_augmentation?: boolean;
  // Query mode (first-class collection mode)
  danbooru_query_enable?: boolean;
  danbooru_query_expand_enable?: boolean;
  danbooru_query_new_tag_min_count?: number;
  danbooru_query_resolve_top_k?: number;
  danbooru_query_max_expanded_tags?: number;
  danbooru_query_expand_categories?: number[];
  danbooru_query_resolve_interval?: number;
  // Per-tag per-epoch collection caps (0 = unlimited)
  danbooru_query_collect_per_epoch?: number;
  danbooru_new_tag_collect_per_epoch?: number;
  danbooru_low_f1_collect_per_epoch?: number;
  danbooru_tags?: string;
  danbooru_injection_interval?: number;
  danbooru_injection_batch_size_ratio?: number;
  danbooru_min_score?: number;
  danbooru_max_posts_per_query?: number;
  danbooru_api_interval?: number;
  danbooru_dl_speed_kbps?: number;
  danbooru_buffer_size?: number | null;
  danbooru_vocab_expand?: boolean;
  danbooru_new_tag_min_count?: number;
  danbooru_new_tag_min_count_by_cat?: Record<string, number>;
  danbooru_new_tag_lookback_days?: number;
  danbooru_new_tag_categories?: number[];
  danbooru_new_tag_survey_interval?: number;
  danbooru_max_dynamic_tags?: number;
  // Collection-path weights (weighted selection among available paths)
  danbooru_query_weight_static?: number;
  danbooru_query_weight_new_tag?: number;
  danbooru_query_weight_low_f1?: number;
  danbooru_query_weight_train_count?: number;
  // Low-F1 deficiency collection (existing vocab tags with low per-tag F1)
  danbooru_low_f1_enable?: boolean;
  danbooru_low_f1_threshold?: number;
  danbooru_low_f1_top_k?: number;
  danbooru_low_f1_min_posts?: number;
  // Train-count deficiency collection (exposure balancing)
  danbooru_train_count_enable?: boolean;
  danbooru_train_count_top_k?: number;
  danbooru_train_count_min_deficit_ratio?: number;
  danbooru_train_count_min_per_epoch?: number;
  danbooru_train_count_min_posts?: number;
  danbooru_train_count_collect_per_epoch?: number;
  // Score-based quality tag (label derived from post score)
  danbooru_quality_tag_enable?: boolean;
  danbooru_quality_tag_thresholds?: string;
  danbooru_quality_tag_attach_negative?: boolean;
  // Co-occurrence vocab discovery
  danbooru_cooc_expand_enable?: boolean;
  danbooru_cooc_min_count?: number;
  danbooru_cooc_categories?: number[];
  danbooru_query_weight_cooc?: number;
  danbooru_cooc_collect_per_epoch?: number;
  danbooru_cooc_order_random?: boolean;
}

export interface TaggerTrainingMetric {
  step: number;
  // 0 = initial run, 1+ = nth resume of the same run.  Used by the loss
  // chart to render each resume as its own colored curve.  Optional for
  // backward compatibility with old payloads (treat as 0 when missing).
  resume_seq?: number;
  epoch: number | null;
  loss: number | null;
  f1: number | null;
  train_f1?: number | null;
  threshold: number | null;
  learning_rate: number | null;
  // Macro precision/recall at the current threshold (null for old runs that pre-date this field)
  precision?: number | null;
  recall?: number | null;
  timestamp?: string;
}

export interface TaggerVocabularyPreview {
  num_tags: number;
  category_counts: Record<string, number>;
}

export const createTaggerTrainingRun = async (
  data: TaggerTrainingRunCreateRequest
): Promise<TaggerTrainingRun> => {
  const response = await api.post("/tagger-training/runs", data);
  return response.data;
};

export const updateTaggerTrainingRun = async (
  runId: string,
  data: TaggerTrainingRunCreateRequest
): Promise<TaggerTrainingRun> => {
  const response = await api.patch(`/tagger-training/runs/${runId}`, data);
  return response.data;
};

export const listTaggerTrainingRuns = async (): Promise<TaggerTrainingRun[]> => {
  const response = await api.get("/tagger-training/runs");
  return response.data;
};

export const getTaggerTrainingRun = async (runId: string): Promise<TaggerTrainingRun> => {
  const response = await api.get(`/tagger-training/runs/${runId}`);
  return response.data;
};

export const startTaggerTrainingRun = async (
  runId: string
): Promise<{ message: string; run: TaggerTrainingRun }> => {
  const response = await api.post(`/tagger-training/runs/${runId}/start`);
  return response.data;
};

export const stopTaggerTrainingRun = async (
  runId: string
): Promise<{ message: string; run: TaggerTrainingRun }> => {
  const response = await api.post(`/tagger-training/runs/${runId}/stop`);
  return response.data;
};

export const deleteTaggerTrainingRun = async (runId: string): Promise<void> => {
  await api.delete(`/tagger-training/runs/${runId}`);
};

export interface DanbooruRecentPost {
  post_id: number;
  tags: string[];
  tag_count: number;
  timestamp: number;
}

export interface DanbooruTopTag {
  tag: string;
  count: number;
}

export interface DanbooruAugmentationMetrics {
  enabled: boolean;
  total_collected?: number;
  total_injected_batches?: number;
  buffer_starvation_count?: number;
  buffer_capacity?: number;
  buffer_current?: number;
  unique_tags_seen?: number;
  dynamic_tags_count?: number;
  total_dynamic_collected?: number;
  dynamic_unique_tags_collected?: number;
  low_f1_tags_count?: number;
  low_f1_unavailable_count?: number;
  total_low_f1_collected?: number;
  low_f1_unique_tags_collected?: number;
  cooc_pending_count?: number;
  cooc_promoted_count?: number;
  total_cooc_proposed?: number;
  cooc_proposed_tags?: string[];
  cooc_active_count?: number;
  total_cooc_collected?: number;
  cooc_unique_tags_collected?: number;
  top_cooc_tags?: DanbooruTopTag[];
  top_tags?: DanbooruTopTag[];
  top_dynamic_tags?: DanbooruTopTag[];
  top_low_f1_tags?: DanbooruTopTag[];
  // Query mode (per-tag resolved collection + legacy per-string static)
  query_tags_count?: number;
  query_expanded_count?: number;
  total_query_collected?: number;
  query_unique_tags_collected?: number;
  top_query_tags?: DanbooruTopTag[];
  total_static_collected?: number;
  top_static_queries?: DanbooruTopTag[];
  // Train-count deficiency path (exposure balancing)
  train_count_tags_count?: number;
  train_count_unavailable_count?: number;
  total_train_count_collected?: number;
  train_count_unique_tags_collected?: number;
  top_train_count_tags?: DanbooruTopTag[];
  recent_posts?: DanbooruRecentPost[];
  // Download-speed safety (throttle/ban avoidance)
  dl_speed_check_enabled?: boolean;
  dl_speed_current_kbps?: number;
  dl_speed_avg_kbps?: number;
  dl_cooldown_active?: boolean;
  dl_cooldown_remaining_sec?: number;
  dl_slow_streak?: number;
  dl_cooldown_count?: number;
  dl_cooldown_reason?: string;
  error?: string;
}

export const getTaggerDanbooruMetrics = async (
  runId: string,
): Promise<DanbooruAugmentationMetrics> => {
  const response = await api.get(`/tagger-training/runs/${runId}/danbooru-metrics`);
  return response.data;
};

export const resumeTaggerDanbooru = async (runId: string): Promise<{ success: boolean }> => {
  const response = await api.post(`/tagger-training/runs/${runId}/danbooru/resume`);
  return response.data;
};

// Online Danbooru augmentation metrics for IMAGE-GENERATION training.
// No vocabulary expansion (open-vocab) — collection + interrupt-batch only.
export interface DanbooruImageAugRecentPost {
  post_id: number;
  tag_count: number;
  tags: string[];
  path?: string;  // "static" | "deficiency"
}

export interface DanbooruImageAugMetrics {
  enabled: boolean;
  total_collected?: number;
  total_injected_batches?: number;
  buffer_starvation_count?: number;
  buffer_capacity?: number;
  buffer_current?: number;
  unique_tags_seen?: number;
  static_collected?: number;
  deficiency_collected?: number;
  deficiency_query_count?: number;
  bucket_distribution?: Record<string, number>;
  top_tags?: DanbooruTopTag[];
  recent_posts?: DanbooruImageAugRecentPost[];
  // Download-speed safety (throttle/ban avoidance)
  dl_speed_check_enabled?: boolean;
  dl_speed_current_kbps?: number;
  dl_speed_avg_kbps?: number;
  dl_cooldown_active?: boolean;
  dl_cooldown_remaining_sec?: number;
  dl_slow_streak?: number;
  dl_cooldown_count?: number;
  dl_cooldown_reason?: string;
  error?: string;
}

export const getTrainingDanbooruMetrics = async (
  runId: number,
): Promise<DanbooruImageAugMetrics> => {
  const response = await api.get(`/training/runs/${runId}/danbooru-metrics`);
  return response.data;
};

export const resumeTrainingDanbooru = async (runId: number): Promise<{ success: boolean }> => {
  const response = await api.post(`/training/runs/${runId}/danbooru/resume`);
  return response.data;
};

export const getTaggerTrainingMetrics = async (
  runId: string,
  sinceStep: number = 0,
  maxPoints: number = 2000,
): Promise<TaggerTrainingMetric[]> => {
  const params: Record<string, number> = { max_points: maxPoints };
  if (sinceStep > 0) params.since_step = sinceStep;
  const response = await api.get(`/tagger-training/runs/${runId}/metrics`, { params });
  return response.data;
};

export const getTaggerVocabularyPreview = async (
  datasetIds: number[],
  excludedCategories: string[] = [],
  banTags: string = "",
  useGelbooruCategories: boolean = true,
): Promise<TaggerVocabularyPreview> => {
  const params: Record<string, string> = { dataset_ids: datasetIds.join(",") };
  if (excludedCategories.length > 0) params.excluded_categories = excludedCategories.join(",");
  if (banTags.trim()) params.ban_tags = banTags;
  params.use_gelbooru_categories = String(useGelbooruCategories);
  const response = await api.get("/tagger-training/vocabulary-preview", { params });
  return response.data;
};
