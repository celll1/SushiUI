import axios from "axios";
import { wsClient } from "./websocket";
import { resolveGlobalAttentionImpl, resolveGlobalAttentionType } from "./attentionSettings";

const api = axios.create({
  baseURL: "/api/v1",
  headers: {
    "Content-Type": "application/json",
  },
  // Default timeout for ordinary (non-generation) requests. Generation
  // endpoints call postGenerationRequest() below instead, which is not
  // bound to a fixed ceiling.
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

// ---------------------------------------------------------------------------
// Generation request dispatch: no fixed timeout ceiling
// ---------------------------------------------------------------------------
// A generation's wall-clock time scales with frames/steps/resolution (a
// video run can legitimately take >10 minutes), so a fixed axios timeout
// eventually reports a request as "failed" while the backend is still
// working -- and by then it usually finishes and saves anyway. Instead,
// while the SSE progress channel (websocket.ts) is open, we let the request
// run as long as that channel keeps proving the backend process is alive
// (its 30s "ping" heartbeat continues even through phases, like VAE decode,
// that emit no per-step "progress" message) and only abort if the channel
// itself goes stale or drops. If the channel was never connected in the
// first place, there's no liveness evidence to lean on, so we fall back to
// the previous fixed ceiling.
const GENERATION_NO_CHANNEL_TIMEOUT_MS = 600000;
const GENERATION_STALL_TIMEOUT_MS = 90000; // 3x the server's 30s ping cadence

/** Thrown when the SSE progress channel goes stale mid-generation. Distinct
 *  from a real backend failure (4xx/5xx REST response): the backend may
 *  still be running and will save its result to the gallery when it
 *  finishes -- the client just lost its ability to keep waiting for it. */
export class GenerationStalledError extends Error {
  code = "SUSHIUI_GENERATION_STALLED";
  constructor() {
    super(
      "Lost contact with the server's progress channel mid-generation. " +
      "The backend may still be running and will save its result to the " +
      "gallery if it finishes -- check there before retrying."
    );
    this.name = "GenerationStalledError";
  }
}

/** True for the stall abort thrown by postGenerationRequest() -- callers use
 *  this to show "lost contact, may still finish" instead of "generation
 *  failed" for what is not necessarily a real failure. */
export function isGenerationStalledError(error: any): boolean {
  return error?.code === "SUSHIUI_GENERATION_STALLED";
}

/** True for the 500 the Next dev proxy SYNTHESIZES when it severs the upstream
 *  socket mid-generation -- not a backend failure.
 *
 *  When the dev server's http-proxy aborts the connection to the backend it
 *  answers the browser itself with `res.statusCode = 500; res.end("Internal
 *  Server Error")` (node_modules/next/dist/server/lib/router-utils/proxy-request.js).
 *  The backend never sees this: it keeps generating, keeps emitting progress on
 *  the SSE channel, and saves its result on completion -- so reporting it to
 *  the user as "generation failed" is wrong.
 *
 *  The signature is unambiguous because SushiUI's own 500s never look like
 *  this: every backend error goes through generic_error_handler
 *  (backend/api/error_handlers.py:177), which returns a JSON ErrorResponse
 *  object.  A 500 whose body is the bare STRING "Internal Server Error" can
 *  therefore only have been written by the proxy. */
function isProxySynthesized500(error: any): boolean {
  return (
    error?.response?.status === 500 &&
    typeof error?.response?.data === "string" &&
    error.response.data.trim() === "Internal Server Error"
  );
}

async function postGenerationRequest<T = any>(url: string, data: any, config: any = {}) {
  if (typeof window === "undefined" || !wsClient.isConnected()) {
    return api.post<T>(url, data, { ...config, timeout: GENERATION_NO_CHANNEL_TIMEOUT_MS });
  }

  const controller = new AbortController();
  // Judge staleness by message age alone, not live readyState: the client
  // auto-reconnects a dropped SSE connection after 3s (see websocket.ts), so
  // treating a momentary reconnect as an instant stall would abort a
  // generation over a blip the channel already recovered from.
  const watchdog = setInterval(() => {
    if (wsClient.msSinceLastMessage() > GENERATION_STALL_TIMEOUT_MS) {
      controller.abort();
    }
  }, 5000);

  try {
    return await api.post<T>(url, data, { ...config, timeout: 0, signal: controller.signal });
  } catch (error: any) {
    if (controller.signal.aborted) {
      throw new GenerationStalledError();
    }
    if (isProxySynthesized500(error)) {
      // The dev proxy cut the socket; the backend is still working. Same user
      // situation as a stale progress channel, so report it the same way
      // instead of as a generation failure.
      console.warn(
        "[API] Dev proxy severed the request to " + url +
        " and answered 500 itself; the backend generation is unaffected and " +
        "will still save. See the dev server log for the '[proxy-abort ...]' line."
      );
      throw new GenerationStalledError();
    }
    throw error;
  } finally {
    clearInterval(watchdog);
  }
}

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

// What was taken from a MiniMax-H3 overlay checkpoint, minus the file identities.
export interface MiniMaxH3HybridRecipe {
  preset: "block_range_adaln";
  block_range_start: number;
  /** Inclusive. */
  block_range_end: number;
  final_adaln_from_overlay: boolean;
}

// Which two MiniMax-H3 checkpoints the loaded DiT was merged from. File names
// only, never paths. Read `compatibility_digest`, not `model_hash`, to tell two
// hybrids apart: the hash is the BASE file's alone, so every hybrid on one base
// reports the same one here and in the gallery.
export interface MiniMaxH3HybridProvenance {
  variant: "hybrid";
  base_variant: string | null;
  overlay_variant: string | null;
  base_file: string;
  overlay_file: string;
  hybrid_recipe: MiniMaxH3HybridRecipe;
  compatibility_digest: string;
  quantization_format?: string;
  overlay_key_count?: number;
}

export interface ModelInfo {
  source_type: string;
  source: string;
  type: "sd15" | "sdxl" | "zimage" | "flux2" | "anima" | "lens" | "ideogram4" | "minit2i" | "krea2" | "ltx2" | "acestep" | "minimax_h3" | "minimax_music3";
  is_v_prediction: boolean;
  model_hash: string;
  // Model-list entry fields (from GET /models)
  name?: string;
  path?: string;
  architecture?: string;
  vae_type?: string;  // MiniT2I: "none" (pixel) | "sdxl" | "flux1" (latent)
  // MiniMax-H3: "fl2va" | "ref2va" for a single checkpoint, "hybrid" for a
  // merged pair. Only a hybrid carries the three provenance fields after it.
  variant?: string | null;
  base_variant?: string | null;
  overlay_variant?: string | null;
  hybrid?: MiniMaxH3HybridProvenance | null;
}

export type ComponentSlotId = "text_encoder" | "vision_encoder" | "backbone" | "vae" | "audio_vae";
export type ComponentOrigin = "embedded_checkpoint" | "model_tree" | "architecture_default" | "selected_external" | "unused" | "unavailable";

export interface ComponentCandidate {
  candidate_id: string;
  slot: ComponentSlotId;
  kind: "text_encoder" | "vision_encoder" | "transformer" | "unet" | "vae" | "audio_vae";
  display_name: string;
  origin: ComponentOrigin;
  path_display?: string | null;
  container_size_bytes?: number | null;
  estimated_component_size_bytes?: number | null;
  compatibility: "compatible" | "unknown" | "incompatible";
  compatibility_reason?: string | null;
  switchable: boolean;
  switch_reason?: string | null;
  is_current: boolean;
  load_strategy: "none" | "standalone" | "embedded_extract" | "architecture_resolved" | "unsupported";
  variant?: string | null;
  // MiniMax-H3 text encoders: a converted small encoder conditions only through
  // the trained projection named here. `agreement` is recorded for that exact
  // (encoder, projection) pair, so it is never carried over to another one.
  requires_projection?: boolean;
  projection?: string | null;
  // Every projection declaring this encoder's width. With more than one usable
  // entry the backend refuses to choose, so the switch must name one.
  projection_candidates?: MiniMaxH3ProjectionCandidate[];
  agreement?: MiniMaxH3TeAgreement | null;
}

export interface EffectiveComponent {
  candidate_id: string;
  slot: ComponentSlotId;
  kind: string;
  display_name: string;
  origin: ComponentOrigin;
  residency: "resident" | "configured" | "unloaded" | "unavailable";
  embedded: boolean;
  path_display?: string | null;
  container_size_bytes?: number | null;
}

export interface ComponentSlotState {
  slot: ComponentSlotId;
  visible: boolean;
  current: EffectiveComponent | null;
  runtime_override: EffectiveComponent | null;
  switchable: boolean;
  reason?: string | null;
  candidates: ComponentCandidate[];
}

export interface CurrentComponentsResponse {
  loaded: boolean;
  model_revision: number;
  component_revision: number;
  health: "ready" | "mutating" | "degraded" | "unloaded";
  architecture: string | null;
  operation?: Record<string, unknown> | null;
  slots: ComponentSlotState[];
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

// Architecture detected from a LoRA file's own key names at scan time
// (NOT the currently loaded model). "unknown" is a first-class value.
export type LoRAArch = "sd15" | "sdxl" | "zimage" | "flux2" | "minimax_h3" | "unknown";

export interface LoRAListEntry {
  name: string;
  path: string;
  arch: LoRAArch;
}

// Sampling settings a step-distillation LoRA declares itself distilled for,
// parsed only from fields the LoRA file's own safetensors metadata declares.
export interface LoRARecommendedSettings {
  num_inference_steps: number;
  fbcache_enable: boolean;
  spectrum_enable: boolean;
  source: "student_steps";
}

export interface LoRAInfo {
  name: string;
  path: string;
  arch?: LoRAArch;
  size: number;
  exists: boolean;
  layers: string[];
  // Only present on GET /loras/{lora_name}; null when the file's metadata
  // declares nothing recognized -- never a guessed/invented recommendation.
  recommended?: LoRARecommendedSettings | null;
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
  is_style_transfer?: boolean;
  style_adain_strength?: number;
  style_blocks?: string;
  style_low_scale_end?: number;
  style_beta?: number;
  style_value_mode?: string;
  style_ref_value_mix?: number;
  style_late_release?: number;
  style_rope_offset?: boolean;
  style_combine_mode?: string;  // "stack" | "common_concept" — multi-reference combine mode (N-ref style transfer only)
  style_guidance_scale?: number;  // extra guidance pass strengthening style independently of cfg_scale (0/undefined = off)
}

// Per-generation quantized-GEMM path selection. One axis, not two booleans:
// whether a checkpoint's quantized Linear layers are FP8 or INT8 is fixed by
// the checkpoint format, so the caller only chooses W8A8 vs dequantized.
// `null` (or an absent field) means "do not touch the process-level setting".
export type QuantizedGemmMode = "w8a8" | "dequant" | null;

export interface GenerationParams {
  prompt: string;
  negative_prompt?: string;
  steps?: number;
  cfg_scale?: number;
  // SenseNova U1.5 flow-matching time-shift; every other architecture ignores it.
  timestep_shift?: number;
  // SenseNova U1.5 second CFG scale (it2i_generate reference-image editing path),
  // active only alongside cfg_scale when ref_images is supplied. Every other
  // architecture ignores it.
  img_cfg_scale?: number;
  // SenseNova U1.5 CFG-overshoot clamp applied before the Euler step; every
  // other architecture ignores it.
  cfg_norm?: "none" | "global" | "channel";
  // SenseNova U1.5 per-phase weight-half CPU eviction; every other
  // architecture ignores it.
  sensenova_mot_phase_eviction?: boolean;
  // SenseNova U1.5 per-layer prefix KV cache CPU streaming; every other
  // architecture ignores it.
  sensenova_kv_cache_streaming?: boolean;
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
  attention_type?: string;
  attention_impl?: string;
  // SDXL micro-conditioning override (inference): original_size for time_ids.
  // Explicit w/h (0/undefined = auto), else output size * scale. crop stays (0,0).
  original_size_w?: number;
  original_size_h?: number;
  original_size_scale?: number;
  // U-Net Quantization
  unet_quantization?: string | null;
  // Text Encoder Quantization (Z-Image only)
  text_encoder_quantization?: string | null;
  // Quantized GEMM path for checkpoints whose Linear weights are ALREADY
  // weight-only quantized (ideogram4 = FP8/nf4, krea2 = FP8 or INT8,
  // anima = INT8). A different axis from unet_quantization, which quantizes an
  // unquantized model's weights at load time.
  //   null      = leave the process-level setting alone (the default)
  //   "w8a8"    = force both W8A8 GEMM paths on for this generation
  //   "dequant" = force both off (weights dequantized, normal matmul)
  // Not sent when null, so a raw caller / env opt-in is never overridden.
  quantized_gemm_mode?: QuantizedGemmMode;
  // CPU Text Encoding: run text encoder on CPU to save VRAM (slower)
  cpu_text_encoding?: boolean;
  // torch.compile optimization
  use_torch_compile?: boolean;
  vae_tiling?: boolean;
  vae_tile_threshold?: number;
  // "blend" (diffusers overlap + cross-fade) | "context" (real-context margin,
  // decoded then discarded; tiles abut exactly)
  vae_tile_mode?: string;
  // Two-pass whole-image GroupNorm statistics for a tiled decode (opt-in).
  // Decode runs twice when tiling is actually engaged; no effect on decoders
  // without GroupNorm (Qwen-family VAE used by Anima/Krea2).
  vae_tile_global_norm?: boolean;
  // Color Flatten (chroma-smoothing baked into the saved image at generation time)
  color_flatten_strength?: number;
  // In-loop background hard-flatten (detects flat background region during
  // the final denoise steps and replaces it with its solid dominant color; SD/SDXL only)
  flatten_in_loop?: boolean;
  flatten_in_loop_last_steps?: number;
  flatten_in_loop_min_region?: number;
  // Spectrum (Adaptive Spectral Feature Forecasting) acceleration
  spectrum_enable?: boolean;
  fbcache_enable?: boolean;
  fbcache_threshold?: number;
  fbcache_warmup_steps?: number;
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
  block_swap_h2d_only?: boolean;
  block_swap_ring_size?: number;
  // FLUX.2 Image Edit (reference images for sequence conditioning)
  ref_images?: File[];
  // SigLIP2 Vision Encoder path (SDXL/SD1.5 reference image conditioning)
  vision_encoder_path?: string | null;
  // VAE override: path to a standalone VAE to swap in for this generation
  // (empty/null = use the loaded model's VAE).
  vae_path?: string | null;
  // Text encoder override: path to a standalone text encoder to swap in
  // (empty/null = use the loaded model's text encoder). SD1.5/SDXL only server-side.
  text_encoder_path?: string | null;
  // PiD (Pixel Diffusion Decoder) options: only take effect when vae_path
  // selects a PiD checkpoint (VaeEntry.kind === "pid_decoder"); ignored otherwise.
  pid_sr_output?: string | null;   // "4x" | "original"
  pid_use_gemma?: boolean;
  pid_low_vram?: boolean;
  // PiD large-output (>4096px) decode controls: default = tiled true
  // super-resolution; pid_fast_large_decode = true switches to a faster
  // cap+bicubic path (lower quality).
  pid_tile_native?: number;
  pid_tile_overlap_ratio?: number;
  pid_fast_large_decode?: boolean;
  // Video generation fields (used when a video model is loaded; the merged
  // txt2img/img2img panels carry these and map them into Txt2VidParams/Img2VidParams).
  num_frames?: number;              // 8k+1 (default 121)
  frame_rate?: number;              // default 24.0
  num_inference_steps?: number;     // video sampler steps (default 8, distilled)
  guidance_scale?: number;          // video/audio guidance (default 1.0)
  num_videos_per_prompt?: number;   // default 1
  audio_enable?: boolean;           // default true
  // Video route's block swap (Txt2VidParams/Img2VidParams/Ref2VidParams'
  // `blocks_to_swap`). Kept under its own key rather than reusing the field
  // above, which is the model-global IMAGE block-swap setting -- a blind
  // spread would clobber one with the other. undefined/0 = disabled, which
  // is this endpoint's own default (opt-in; a machine that already fits the
  // model should not pay a swap cost it does not need).
  video_blocks_to_swap?: number;
  // MiniMax-H3 only, not bit-exact -- see Txt2VidParams.fuse_output_proj. No
  // image-mode equivalent, so unlike video_blocks_to_swap this needs no
  // separate key: it maps straight onto the video request's own field.
  fuse_output_proj?: boolean;
  // Music generation fields (used when an audio model (ACE-Step) is loaded;
  // the panel maps these into Txt2AudParams for txt2aud requests). `prompt`
  // above doubles as the caption text.
  lyrics?: string;
  audio_duration?: number;          // seconds. ACE-Step default 30.0; MiniMax Music 3 default 60.0 (an
                                     // UPPER BOUND -- the model may stop earlier -- ceiling 360s).
  inference_steps?: number;         // ACE-Step ONLY sampler steps (default 8, turbo distilled, per-song).
  shift?: number;                   // ACE-Step ONLY; default 3.0.
  sampler_mode?: string;            // ACE-Step ONLY; accepted for forward-compat, currently a no-op.
  vocal_language?: string;          // ACE-Step ONLY; default "en".
  // MiniMax Music 3 ONLY. Deliberately NOT named `num_inference_steps` --
  // that name is already taken by the video fields above (per-song step count
  // there; per-CHUNK, 200-frame-window step count here). The two are
  // mutually exclusive at runtime (only one modality is loaded at a time),
  // but sharing the name would carry the wrong architecture's default across
  // a model switch. Default 30 (design doc "Generation parameter contract"),
  // resolved per-arch via `audioDefaultsForArch`, not hardcoded here.
  music3_num_inference_steps?: number;
  // MiniMax Music 3 ONLY. Flow-stage CFG, distinct from `guidance_scale`
  // above (which video/ACE-Step already share) for the same reason as
  // `music3_num_inference_steps`: the two archs' defaults (1.0 vs 1.7) must
  // not bleed into each other across a model switch. Default 1.7, resolved
  // per-arch via `audioDefaultsForArch`.
  flow_guidance_scale?: number;
  // Tracks which loaded architecture `audio_duration` /
  // `music3_num_inference_steps` / `flow_guidance_scale` were last resolved
  // for (Txt2ImgPanel's arch-aware audio-defaults effect). Not sent to the
  // backend -- purely a UI bookkeeping field, same pattern as
  // OutpaintParams.outpaint_video_audio_mode_arch.
  audio_defaults_arch?: string;
  // Tracks which loaded architecture `steps`/`cfg_scale` were last resolved
  // for via `image_arch_overlays` (currently only SenseNova's 50/4.0). Same
  // bookkeeping pattern as `audio_defaults_arch` just above -- not sent to
  // the backend.
  image_defaults_arch?: string;
  // Keep model components GPU-resident between back-to-back generations
  // (queue sets this automatically based on whether a next item is queued)
  keep_models_hot?: boolean;
  // Loop-generation decode mode (heavy-decoder aware, e.g. PiD): "full" decodes
  // with the active/overridden VAE as usual; "cheap" decodes with the
  // standard/embedded VAE (bypassing a PiD override); "none" skips decode
  // entirely and returns a cached latent_id instead of an image (img2img only;
  // inpaint rejects "none"). Default "full" (current/legacy behavior).
  loop_decode?: "full" | "cheap" | "none";
  // Save the decoded image to disk (so it can still be chained/downloaded)
  // but skip the gallery DB record/thumbnail. Orthogonal to loop_decode.
  skip_gallery?: boolean;
}

export interface Img2ImgParams extends GenerationParams {
  denoising_strength?: number;
  img2img_fix_steps?: boolean;
  resize_mode?: string;
  resampling_method?: string;
  // VAE encode/decode round-trip color-bias correction (img2img/inpaint only)
  vae_drift_correction?: boolean;
  // Start from a server-cached latent (loop_decode="none" from a previous
  // step) instead of an uploaded image. Mutually exclusive with the `image`
  // argument passed to generateImg2Img — exactly one must be provided.
  input_latent_id?: string | null;
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
  // VAE encode/decode round-trip color-bias correction (img2img/inpaint only)
  vae_drift_correction?: boolean;
  // Regional additional prompt (SD/SDXL only): conditions ONLY the
  // generated/repaint region, leaving the main prompt + preserved region
  // untouched. Active iff region_prompt_strength > 0 AND (region_prompt or
  // region_negative_prompt) is non-empty.
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
}

// Outpaint: place a (optionally trimmed/resized) input image inside a LARGER
// canvas and generate everything outside it; the placed region is preserved
// byte-exact regardless of architecture/denoising_strength (see
// core/inference/outpaint_utils.py + PipelineManager.generate_outpaint).
// Shares the ENTIRE inpaint parameter set (feature parity) plus the
// placement fields below. NOTE: "width"/"height" (inherited from
// GenerationParams) are NOT sent by generateOutpaint() -- canvas_width/
// canvas_height fully determine the output size server-side.
export interface OutpaintParams extends GenerationParams {
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
  vae_drift_correction?: boolean;
  // Regional additional prompt (SD/SDXL only): conditions ONLY the generated
  // region, leaving the main prompt + preserved (placed) region untouched.
  // Active iff region_prompt_strength > 0 AND (region_prompt or
  // region_negative_prompt) is non-empty.
  region_prompt?: string;
  region_negative_prompt?: string;
  region_prompt_strength?: number;
  region_prompt_method?: string;
  region_mask_feather?: number;
  // Seam Structure Continuity (SSC, SD/SDXL only): continues thin structures
  // that cross the region boundary (a held rod/staff, limb, torso, lines)
  // into the generated region. x0-space, no extra U-Net forwards. 0 = off.
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
  // Outpaint ControlNet (structure continuity, SD/SDXL only): synthesizes an
  // edge-extrapolation control image (canny/lineart) from the placed region
  // and conditions the generated surround with it, tapering out with
  // distance/schedule progress. false/0 = off (byte-identical).
  outpaint_controlnet_enable?: boolean;
  outpaint_controlnet_mode?: string;
  outpaint_controlnet_model?: string;
  outpaint_controlnet_detector?: string;
  outpaint_controlnet_scale?: number;
  outpaint_controlnet_guidance_start?: number;
  outpaint_controlnet_guidance_end?: number;
  outpaint_controlnet_depth?: number;
  outpaint_controlnet_taper?: number;
  // crop_mask mode only, opt-in. Corner conditioning rounding (Feature #3a,
  // secondary lever) and per-corner residual gate (Feature #2, primary
  // corner-seam fix). Defaults (0.0/0.0/1.0) = disabled = byte-identical.
  outpaint_controlnet_corner_radius_px?: number;
  outpaint_controlnet_corner_gate_radius_px?: number;
  outpaint_controlnet_corner_gate_min?: number;
  // crop_mask mode only, opt-in. L1 four-corner x0-pin softening: softens
  // the per-step x0-pin composite's keep-weight (a different mechanism from
  // the CN residual gate above) in small disks at the 4 rect vertices.
  // Defaults (0.0/1.0) = disabled = byte-identical.
  outpaint_pin_corner_relax_radius_px?: number;
  outpaint_pin_corner_relax_min?: number;
  // Harmonic boundary-offset membrane (post-decode): adjusts generated
  // pixels to meet the preserved boundary exactly; the preserved region
  // remains byte-identical. Distinct from outpaint_seam_fix (a per-edge
  // exposure/tone gain). false = off (byte-identical).
  outpaint_seam_membrane?: boolean;
  outpaint_seam_membrane_band?: number;
  // Cross-seam low-frequency tone membrane ("R2", post-decode): a separate
  // mechanism from outpaint_seam_membrane above. Measures the tone step
  // between the preserved rectangle's own pixels and the decoded generated
  // pixels immediately across the seam, and writes a decaying offset into
  // the generated side only. 0 = off (byte-identical).
  outpaint_seam_tone_strength?: number;
  outpaint_seam_tone_band?: number;
  // Boundary-offset propagation ("G_prop16", post-decode): a third seam
  // mechanism, distinct from both membranes above. Measures the same offset
  // outpaint_seam_membrane measures (preserved pixels vs the decoded
  // reconstruction of that same region, not the cross-seam comparison
  // outpaint_seam_tone_strength uses), and writes it directly into the
  // generated pixels adjacent to the seam. Writes only generated-side
  // pixels; the preserved region is unaffected. 0 = off (byte-identical).
  outpaint_seam_offset_prop?: number;
  // B1 continuity fix (SD/SDXL only): weak low-frequency color/illumination
  // correction applied to the generate region only, within a narrow collar
  // near the preserved rect's boundary, active mid/late in the schedule.
  // 0 = off.
  outpaint_boundary_color_strength?: number;
  // B2 continuity fix (SD/SDXL only): RePaint-style band-limited time-travel
  // resampling -- after a denoise step inside a mid-schedule band, jumps
  // back outpaint_jump_length steps by re-noising the whole latent and
  // re-denoising, repeated outpaint_resample_count times per band segment.
  // 1 = off (B1 only). Values > 1 multiply the number of denoise passes
  // actually run (roughly 1.5-2x the requested step count).
  outpaint_resample_count?: number;
  // B2 jump-back length ("u", in step indices) for each resample cycle.
  outpaint_jump_length?: number;
  // B3 continuity fix (SD/SDXL only): masked self-attention KV injection --
  // a noise-matched reference composite built from the preserved rect's own
  // clean latents, restricted to known-region tokens via spatial masking,
  // so generate-region self-attention queries can attend to the input's own
  // clean features. 0 = off.
  outpaint_reference_strength?: number;
  // Paste-band reconciliation feather ("Option E"): at the final preserved-
  // rectangle paste, the last N rows/columns of the preserved rectangle at
  // its generate-adjacent edges are blended (raised cosine) from the exact
  // input toward the decoded canvas already underneath them, instead of
  // pasted byte-exact. Independent of boundary_relax_strength/
  // boundary_relax_paste's own feather and takes precedence over it when
  // both are active. 0 = off (byte-identical).
  outpaint_paste_feather_px?: number;
  // Preserved-region compositing mode. "exact" (default) is the current
  // byte-exact paste, unchanged. "vae_reconstruct" outputs a single uniform
  // VAE decode of the whole canvas with no paste at all -- the preserved
  // region becomes a VAE reconstruction of the input (not byte-identical),
  // removing the hard raw/decoded pixel discontinuity at the boundary.
  // "vae_reconstruct_hf" additionally restores the preserved region's own
  // high-frequency detail on top of that uniform decode, tapering to zero
  // at the boundary; implemented for SD1.5/SDXL, falls back to
  // "vae_reconstruct" on other architectures. Both non-"exact" modes are
  // NOT byte-identical to the input in the preserved region.
  outpaint_preserve_mode?: "exact" | "vae_reconstruct" | "vae_reconstruct_hf";
  // Display-only preview substitution for outpaint generations: sends the
  // unpinned model x0 prediction to the mid-sampling preview decoder instead
  // of the pinned known/generated composite. Does not affect the sampler's
  // own stepping math or the final saved image. false = off (prior preview
  // behavior).
  outpaint_preview_unpinned_x0?: boolean;
  // --- Placement (outpaint-only) ---
  canvas_width?: number;
  canvas_height?: number;
  place_x?: number;
  place_y?: number;
  place_width?: number;
  place_height?: number;
  input_crop_x?: number;
  input_crop_y?: number;
  input_crop_w?: number;
  input_crop_h?: number;
  outpaint_fill_mode?: string;
}

// Video chain provenance (design §13), carried BY a generation request and
// recorded on the gallery row it produces: which chain a segment belongs to,
// which plan compiled it, which segment it is and what it owns on the chain's
// global frame timeline. Absent on an ordinary, unchained generation.
//
// Every video request shape that can be a chain segment includes these
// (Txt2VidParams -- hence Img2VidParams/Ref2VidParams -- for segment 0, and
// OutpaintVideoParams for every continuation), so ONE interface is what both
// ends of a chain send.
//
// The chain's root prompt and canonical timeline are deliberately not here:
// they live once in the manifest, and the two hashes reference them instead of
// being copied onto every segment. `chain_id`, `chain_plan_hash` and
// `chain_segment_index` are sent together or not at all (the backend answers
// 400 for a partial set).
export interface VideoChainProvenance {
  chain_id?: string;
  chain_manifest_version?: number;
  chain_plan_hash?: string;
  chain_segment_index?: number;
  chain_segment_count?: number;
  /** Half-open [start, end) span this segment owns on the chain timeline. */
  chain_global_frame_start?: number;
  chain_global_frame_end?: number;
  chain_context_mode?: VideoChainContextMode;
  chain_root_prompt_hash?: string;
}

const VIDEO_CHAIN_PROVENANCE_KEYS: Array<keyof VideoChainProvenance> = [
  "chain_id",
  "chain_manifest_version",
  "chain_plan_hash",
  "chain_segment_index",
  "chain_segment_count",
  "chain_global_frame_start",
  "chain_global_frame_end",
  "chain_context_mode",
  "chain_root_prompt_hash",
];

// The provenance fields that are actually set, for a JSON request body. An
// unchained generation contributes nothing: the backend's defaults are all
// null, so an omitted field and an explicit null are the same request, and
// omitting keeps an ordinary video request exactly as it was.
const chainProvenanceBody = (params: VideoChainProvenance): Record<string, unknown> => {
  const body: Record<string, unknown> = {};
  for (const key of VIDEO_CHAIN_PROVENANCE_KEYS) {
    const value = params[key];
    if (value != null) body[key] = value;
  }
  return body;
};

// Same set on a multipart request. Every video route except txt2vid takes the
// provenance as Form fields, and a field that is only in the params object is
// a field the backend never sees (CLAUDE.md failure pattern 1).
const appendChainProvenance = (formData: FormData, params: VideoChainProvenance): void => {
  for (const key of VIDEO_CHAIN_PROVENANCE_KEYS) {
    const value = params[key];
    if (value != null) formData.append(key, String(value));
  }
};

// Video temporal outpaint (LTX-2.3): place a (optionally trimmed) input clip
// at a frame offset inside a LONGER output timeline and generate the frames
// before/after, preserving the placed input frames byte-exact. Mirrors the
// backend OUTPAINT_VIDEO_DEFAULTS (param_defaults.py) + the Form parameters
// of POST /generate/outpaint/video (routes.py). Standalone shape (does not
// extend GenerationParams, matching Txt2VidParams/Img2VidParams -- video has
// no width/height/steps/sampler concept beyond the fields below).
export interface OutpaintVideoParams extends VideoChainProvenance {
  prompt: string;
  negative_prompt?: string;
  width?: number;                  // multiple of 32 (default 768)
  height?: number;                 // multiple of 32 (default 512)
  frame_rate?: number;             // default 24.0
  num_inference_steps?: number;    // default 8 (distilled)
  guidance_scale?: number;         // default 1.0
  seed?: number;                   // default -1
  num_videos_per_prompt?: number;  // default 1
  max_sequence_length?: number;    // default 1024
  audio_enable?: boolean;          // default true
  // --- Placement (outpaint-only) ---
  total_frames?: number;           // output timeline length; (n-1)%8==0, default 121
  input_offset_frames?: number;    // where the (trimmed) clip lands, in pixel frames of the OUTPUT timeline
  input_trim_start_frames?: number; // trim applied to the UPLOADED clip before placement
  input_trim_end_frames?: number;
  // Omit to take the LOADED ARCHITECTURE's default, which differs: "regenerate"
  // on LTX-2.3 (its generated track spans the whole timeline) and
  // "preserve_input" on MiniMax-H3 (which generates audio only for the frames
  // it generates, so "regenerate" leaves the preserved span silent). Resolve it
  // client-side with `outpaintVideoDefaultsForArch`.
  outpaint_video_audio_mode?: "regenerate" | "preserve_input";
  video_lossless?: boolean;        // FFV1 bit-exact encode (not browser-playable)
  // --- Acceleration (same knobs as the image/video GenerationParams schema) ---
  blocks_to_swap?: number;
  // MiniMax-H3 only, not bit-exact -- see Txt2VidParams.fuse_output_proj.
  fuse_output_proj?: boolean;
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
  spectrum_max_cache?: number;
  // Component overrides (same plumbing as image/video gen; empty/null = model default)
  vae_path?: string | null;
  text_encoder_path?: string | null;
  // Transformer quantization (see Txt2VidParams.unet_quantization).
  unet_quantization?: string | null;
  // Quantized-GEMM path (see Txt2VidParams.quantized_gemm_mode).
  quantized_gemm_mode?: QuantizedGemmMode;
  // Attention backend (see Txt2VidParams.attention_type). Filled from the
  // global localStorage setting by the sender, like every other route.
  attention_type?: string;
  // MiniMax-H3 ref2va only (extend_forward). Sizing of each reference image
  // file appended separately by generateOutpaintVideo (not a field here --
  // mirrors how `video`/`bridge_video` are function arguments, not params).
  // Same semantics as Ref2VidParams.reference_image_size.
  reference_image_size?: "max" | "match";
  // Generation-time LoRA (see Txt2VidParams.loras).
  loras?: LoRAConfig[];
  // What THIS continuation is conditioned on, and how many of the preserved
  // clip's tail frames `pinned_tail` pins. Both come from the chain manifest
  // (`buildChainContinuationParams`), never from a panel control: a mode the
  // loaded architecture/variant does not advertise in
  // `chain_context[arch].chain_continuation_modes` is a 400, and an overlap
  // that is not a cumulative sum of `latent_chunk_pattern` is a 400 too --
  // neither is snapped or downgraded server-side.
  continuation_mode?: VideoChainContinuationMode;
  continuation_overlap_frames?: number;
  // `motion_preroll` only: how many of the pre-roll's frames are placed as
  // anchors. Sending it with another mode is a 400 -- a pin and an anchor set
  // claim the same conditioning prefix, so neither is dropped silently.
  continuation_anchor_count?: number;
}

// Video TEMPORAL inpaint (POST /generate/inpaint/video, MiniMax-H3 fl2va):
// regenerate one contiguous frame range of an uploaded clip and preserve the
// rest. There is deliberately NO clip-length field — the output is exactly as
// long as the trimmed input, which is also why it is the TRIMMED INPUT that has
// to be a length the architecture can generate.
export interface InpaintVideoParams {
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  frame_rate?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  seed?: number;
  num_videos_per_prompt?: number;
  max_sequence_length?: number;
  audio_enable?: boolean;
  // --- The range, in PIXEL frames of the TRIMMED clip: start inclusive, end
  // exclusive. Required by the route (there is no defensible default range).
  // The server expands it OUTWARD to latent-group boundaries; a UI that snaps
  // to the same boundaries sends a range that is already effective.
  regenerate_start_frame: number;
  regenerate_end_frame: number;
  input_trim_start_frames?: number;
  input_trim_end_frames?: number;
  // Omit to take the LOADED ARCHITECTURE's default ("preserve_input" on
  // MiniMax-H3); resolve it client-side with `inpaintVideoDefaultsForArch`.
  inpaint_video_audio_mode?: "regenerate" | "preserve_input" | "regenerate_range";
  // Optional spatial mask timeline. Referenced PNG assets are uploaded
  // separately by generateInpaintVideo.
  spatial_mask_manifest?: string;
  video_lossless?: boolean;        // FFV1: carries the preserved frames' exactness into the FILE
  // --- Acceleration (same knobs as the other video routes) ---
  blocks_to_swap?: number;
  // MiniMax-H3 only, not bit-exact -- see Txt2VidParams.fuse_output_proj.
  fuse_output_proj?: boolean;
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
  spectrum_max_cache?: number;
  vae_path?: string | null;
  text_encoder_path?: string | null;
  unet_quantization?: string | null;
  quantized_gemm_mode?: QuantizedGemmMode;
  attention_type?: string;
  // Generation-time LoRA (see Txt2VidParams.loras).
  loras?: LoRAConfig[];
  // MiniMax-H3 ref2va only: how a reference IMAGE is sized before it is
  // packed onto the sequence. Same two values, same meaning, as
  // Ref2VidParams.reference_image_size. Read only when `generateInpaintVideo`
  // is actually given references; harmless to send on fl2va (unread there).
  reference_image_size?: "max" | "match";
}

export interface UpscaleParams {
  upscaler_backend?: string;
  upscaler_model?: string | null;
  scale_factor?: number;
  pil_resample?: string;
  tile_size?: number;
  tile_overlap?: number;
  rtx_vsr_quality?: string;
  unsharp_enable?: boolean;
  unsharp_radius?: number;
  unsharp_percent?: number;
  unsharp_threshold?: number;
  // Diffusion tile upscale (upscaler_backend === "diffusion")
  prompt?: string;
  negative_prompt?: string;
  diffusion_denoising_strength?: number;
  steps?: number;
  cfg_scale?: number;
  sampler?: string;
  schedule_type?: string;
  attention_type?: string;
  attention_impl?: string;
  seed?: number;
  diffusion_pre_upscale_mode?: string;
}

export interface UpscalerModelInfo {
  name: string;
  path: string;
  size_mb: number;
  source_dir: string;
}

// NOTE: `GET /images` (list) returns a slim per-row summary -- only the
// fields the grid cell/page-local filtering actually read (id, filename,
// prompt, negative_prompt, generation_type, width, height, seed, created_at,
// is_favorite, image_hash, is_video, is_audio). Everything else below is
// detail-only and populated by `GET /images/{id}` (see `getImage()`), fetched
// on demand when a gallery cell is opened -- ImageGrid.tsx's `openImageDetail`
// helper. Do not read detail-only fields off list/array items (`images`,
// `filteredImages`); only off `selectedImage` after it resolves.
export interface GeneratedImage {
  id: number;
  filename: string;
  prompt: string;
  negative_prompt: string;
  model_name?: string;
  sampler?: string;
  steps?: number;
  cfg_scale?: number;
  seed: number;
  ancestral_seed?: number;
  width: number;
  height: number;
  generation_type: string;
  parameters?: any;
  created_at: string;
  is_favorite: boolean;
  image_hash?: string;
  source_image_hash?: string;
  mask_data?: string;
  lora_names?: string;
  model_hash?: string;
  // Which MiniMax-H3 checkpoint (fl2va/ref2va) actually ran; absent for every
  // other architecture. The filename alone can't distinguish them once either
  // file is renamed.
  model_variant?: string;
  // Present only when model_variant === "hybrid": which MiniMax-H3 pair and
  // recipe produced the row. Read model_hybrid_digest, not model_hash, to tell
  // two hybrids apart -- the hash is the base file's alone.
  model_hybrid_base?: string;
  model_hybrid_overlay?: string;
  model_hybrid_preset?: string;
  /** Inclusive, formatted "start..end". */
  model_hybrid_block_range?: string;
  model_hybrid_final_adaln_from_overlay?: boolean;
  model_hybrid_digest?: string;
  model_hybrid_quantization?: string;
  unet_quantization?: string;
  // What the request asked for on the quantized-GEMM axis; absent when the
  // generation forced nothing. The path that actually ran is `fp8_gemm`.
  quantized_gemm_mode?: string;
  effective_warnings?: string | { code?: string; message: string }[]; // Feature-degradation notices recorded during generation
  ref_images?: string[]; // FLUX.2 Image Edit: Reference image hashes
  vision_encoder_name?: string;   // SigLIP2 Vision Encoder filename
  vision_encoder_hash?: string;   // SHA256 hash of Vision Encoder model
  // VAE source: dir/repo id, "embedded (checkpoint)", "none (pixel-space)", or
  // "override: <path> (run <name>, step <n>, EMA weights, decoder only)" when a
  // per-generation VAE override decoded the image.
  vae_name?: string;
  vae_hash?: string;              // SHA256 hash of the VAE weight file (when identifiable)
  // FP8 GEMM path used by weight-only FP8 checkpoints (Ideogram 4 / Krea 2).
  // "w8a8_scaled_mm(<mode>)" or "dequant..."; absent for every other checkpoint.
  fp8_gemm?: string;
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
  // Color Flatten / VAE drift correction
  color_flatten_strength?: string;
  vae_drift_correction?: string;
  // In-loop background hard-flatten
  flatten_in_loop?: string;
  flatten_in_loop_last_steps?: string;
  flatten_in_loop_min_region?: string;
  // Upscale parameters (generation_type === 'upscale')
  upscaler_backend?: string;
  upscaler_model?: string;
  upscaler_model_hash?: string;
  scale_factor?: string;
  pil_resample?: string;
  tile_size?: string;
  tile_overlap?: string;
  rtx_vsr_quality?: string;
  diffusion_denoising_strength?: string;
  diffusion_pre_upscale_mode?: string;
  // Video parameters (generation_type === 'txt2vid' / 'img2vid'; filename is an
  // .mp4, UNLESS this row was generated with video_lossless=true, in which case
  // filename is an FFV1-in-mkv master (byte-exact, not browser-playable) and
  // preview_filename -- when present -- is a browser-playable H.264 mp4 proxy
  // of it. Playback UI should prefer preview_filename over filename; download/
  // "send to" actions should always keep using filename (the master).
  is_video?: boolean;
  preview_filename?: string;
  num_frames?: number;
  fps?: number;
  duration?: number;
  audio_enable?: boolean;
  // Video chain provenance (design §13), emitted as a group on a video row
  // that was one segment of a chained long clip and absent on every other row.
  // Stringified by `GeneratedImage.to_dict()` like the other conditional
  // fields. The root prompt and canonical timeline are NOT here: the row's
  // `prompt` is this segment's own compiled prompt, and the two hashes point at
  // the manifest that holds the originals once.
  chain_id?: string;
  chain_manifest_version?: string;
  chain_plan_hash?: string;
  chain_segment_index?: string;
  chain_segment_count?: string;
  chain_global_frame_start?: string;
  chain_global_frame_end?: string;
  chain_context_mode?: string;
  chain_root_prompt_hash?: string;
  // Audio parameters (generation_type === 'txt2aud' / 'aud2aud'; filename is a .flac)
  is_audio?: boolean;
  sample_rate?: number;
  audio_duration?: string;
}

// ---------------------------------------------------------------------------
// Video generation (LTX-2.3) — txt2vid (JSON) / img2vid (multipart keyframe)
// ---------------------------------------------------------------------------

export interface Txt2VidParams extends VideoChainProvenance {
  prompt: string;
  negative_prompt?: string;
  width?: number;             // multiple of 32 (default 768)
  height?: number;            // multiple of 32 (default 512)
  num_frames?: number;        // 8k+1 (default 121)
  frame_rate?: number;        // default 24.0
  num_inference_steps?: number; // default 8 (distilled)
  guidance_scale?: number;    // default 1.0
  seed?: number;              // default -1
  num_videos_per_prompt?: number; // default 1
  max_sequence_length?: number;   // default 1024
  audio_enable?: boolean;     // default true
  // Component overrides (same plumbing as image gen; empty/null = model default)
  vae_path?: string | null;
  text_encoder_path?: string | null;
  // Transformer quantization. Only "int8" is applied on LTX-2.3 (one-time
  // in-place conversion of the video DiT -- NOT the Gemma-3 text encoder);
  // the FP8 values warn and are ignored there.
  unet_quantization?: string | null;
  // Per-generation GEMM path for ALREADY-quantized Linear weights (null =
  // leave the process flags alone). ltx2 is in quantized_linear_archs, so this
  // selects a real path on LTX-2.3 -- its loader swaps in Int8Linear/Fp8Linear
  // for a weight-only quantized transformer component, and unet_quantization
  // "int8" produces the same classes at runtime.
  quantized_gemm_mode?: QuantizedGemmMode;
  attention_type?: string;
  // Generation-time LoRA. Applied by MiniMax-H3 (core.models.minimax_h3.minimax_h3_lora);
  // LTX-2.3 has no LoRA loader on its video path at all -- accepted and
  // ignored, with an unsupported_param warning when non-empty.
  loras?: LoRAConfig[];
  // Number of transformer blocks kept CPU-resident and streamed to GPU during
  // the denoise loop. 0 (the default) disables block swap; the field is
  // opt-in on every video route the same way it is on the image ones.
  blocks_to_swap?: number;
  // MiniMax-H3 only: fuses the output-tail projection heads (proj_out,
  // audio_proj_out) into the sequence-chunked output-norm loop instead of
  // materializing the full (1, S, hidden_size) intermediate first. NOT
  // bit-exact (fusing changes the output-projection GEMM's row count per
  // call, and cuBLAS's tiling/reduction order depends on it) -- see
  // core.models.minimax_h3.adaln_chunking's "Head fusion" note. Off by
  // default for that reason; accepted and ignored on LTX-2.3.
  fuse_output_proj?: boolean;
  // --- Acceleration (same knobs as /generate/outpaint/video and
  // /generate/inpaint/video; see InpaintVideoParams' own comment). Mutually
  // exclusive with Block Swap (each disabled server-side, with a logged
  // reason, whenever blocks_to_swap > 0 -- see
  // core.pipeline_backends.ltx2._ltx2_build_fbcache/_ltx2_build_spectrum and
  // core.models.minimax_h3_block_loop_wrapper.attach_fbcache) and with each
  // other (Spectrum takes precedence over FBCache). ---
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
  spectrum_max_cache?: number;
}

// One ADDITIONAL MiniMax-H3 keyframe anchor: the image and the pixel frame it
// is pinned to. `frame_index` follows the endpoint's own convention — 0 is the
// first frame, -1 the clip's last frame AFTER the server snaps num_frames to
// the model's 17n+5 grid (which is why -1 exists: the client cannot compute the
// last index), and any other value that exact frame.
export interface MiniMaxH3Keyframe {
  image: File | string;   // data URL or File
  frame_index: number;
}

export interface Img2VidParams extends Txt2VidParams {
  // img2vid additionally uploads a keyframe `image` handled by generateImg2Vid()
  //
  // OPTIONAL second keyframe: the LAST frame, as a data URL (or a File).
  // MiniMax-H3's `fl2va` workflow conditions on the ENDS of the clip, so
  // `image` is the first frame and this is the last one. null/undefined =
  // no end anchor. LTX-2.3 declares it unsupported and answers with an
  // `unsupported_param` warning if it is sent. It is exactly equivalent to a
  // `keyframes` entry at frame_index -1 and stays live as that alias.
  last_frame_image?: File | string | null;
  // Where the uploaded `image` sits on the clip (MiniMax-H3). 0 (the default)
  // is the first frame; -1 is the resolved last frame.
  input_image_frame_index?: number;
  // Additional anchors. Sent as two positional lists (keyframe_images /
  // keyframe_frame_indices); the ORDER is not semantic, since the server packs
  // anchors in ascending frame order.
  keyframes?: MiniMaxH3Keyframe[];
  // An audio track the video is generated AGAINST (MiniMax-H3). Its rows are
  // pinned clean across the WHOLE clip -- there is no offset or duration to
  // send, because partial-timeline placement is not supported -- and the muxed
  // output carries this file's samples rather than a decode. A track shorter
  // than the clip is a 400. LTX-2.3 declares it unsupported and answers with an
  // `unsupported_param` warning if it is sent.
  input_audio?: File | null;
}

// ref2vid (MiniMax-H3 `ref2va`): the txt2vid parameter set plus how an image
// reference is sized. The reference FILES travel separately (see
// MiniMaxH3References) because they are uploads, not parameters.
export interface Ref2VidParams extends Txt2VidParams {
  // "max"  — the released recipe: every image reference on a 2048-pixel short
  //          edge of its own, upscaling included and with no area cap.
  // "match" — scale each image reference DOWN to the generation's pixel area.
  //          Fewer reference rows, so a shorter packed sequence; a reference's
  //          rows ride through every sampling step.
  // Video references are unaffected either way: they always follow the canvas
  // rule the generated video follows.
  reference_image_size?: "max" | "match";
  // OPTIONAL keyframe anchors (C5), laid out AFTER every reference block --
  // same shape as Img2VidParams.keyframes. References stay content
  // conditioning (read by the prompt); anchors stay placement conditioning
  // (pinned to a frame). Always emits a warnings[] entry when combined with a
  // reference, since MiniMax's model card does not describe the combination.
  keyframes?: MiniMaxH3Keyframe[];
}

// The reference uploads of one ref2vid request, IN THE ORDER THE MODEL READS
// THEM. That order is semantic: it fixes the <Picture i> / <Audio j> /
// <Video k> labels the prompt refers to and lays the references out on the
// packed sequence's shared rotary clock.
//
// `videoAudios` is POSITIONAL: entry n is the soundtrack of `videos[n]`, and a
// video with no soundtrack holds its slot with null. The model's limits (9 / 3
// / 3, 12 total, and never audio alone) are enforced server-side.
export interface MiniMaxH3References {
  images: File[];
  videos: File[];
  videoAudios: (File | null)[];
  audios: File[];
}

// ---------------------------------------------------------------------------
// Audio generation (ACE-Step 1.5) — txt2aud (JSON). Standalone request shape
// (does not extend GenerationParams -- audio has no width/height/steps/sampler
// concept beyond the fields below).
// ---------------------------------------------------------------------------

export interface Txt2AudParams {
  prompt: string;             // caption text (also the MUSIC DESCRIPTION for MiniMax Music 3 -- distinct from lyrics)
  lyrics?: string;             // ACE-Step: optional. MiniMax Music 3: REQUIRED non-empty (checkpoint contract).
  audio_duration?: number;    // seconds. ACE-Step default 30.0; MiniMax Music 3 default 60.0 -- an UPPER BOUND
                               // (the autoregressive stage may stop earlier), ceiling 360s.
  seed?: number;               // default -1
  inference_steps?: number;   // ACE-Step ONLY (turbo distilled default 8, per-song). MiniMax Music 3 does not
                               // read this field -- see num_inference_steps below.
  guidance_scale?: number;    // ACE-Step ONLY (turbo is CFG-distilled; default 1.0). MiniMax Music 3 does not
                               // read this field -- see flow_guidance_scale below.
  shift?: number;              // ACE-Step ONLY; default 3.0. No MiniMax Music 3 equivalent.
  sampler_mode?: string;       // ACE-Step ONLY; accepted for forward-compat, currently a no-op.
  vocal_language?: string;     // ACE-Step ONLY; default "en". Not a MiniMax Music 3 parameter.
  // MiniMax Music 3 ONLY. Per CHUNK (the flow-matching DiT's 200-frame windows),
  // NOT per song -- distinct from ACE-Step's per-song `inference_steps` above,
  // which is why this is a separate field rather than a shared name. Default 30
  // (design doc "Generation parameter contract"); `undefined` on this field lets
  // the backend resolve MiniMax Music 3's own default from
  // `audio_defaults_for_arch` when omitted.
  num_inference_steps?: number;
  // MiniMax Music 3 ONLY. Flow-stage CFG, distinct from ACE-Step's
  // `guidance_scale` above (autoregressive-stage CFG (1.5) and top-k (50) are
  // fixed by the reference recipe and are not exposed as request parameters at
  // all). Default 1.7.
  flow_guidance_scale?: number;
  loras?: LoRAConfig[];
  // Weight-only quantization of the ACE-Step DiT. Only "int8" is applied on
  // this architecture (a one-time in-place conversion of the audio DiT -- NOT
  // the Oobleck VAE or the Qwen3-Embedding text encoder); the FP8 values warn
  // and are ignored. acestep is in runtime_int8_archs.
  unet_quantization?: string | null;
  // Per-generation GEMM path for ALREADY-quantized Linear weights (null =
  // leave the process flags alone). acestep is in quantized_linear_archs: its
  // loader swaps in Int8Linear/Fp8Linear for a weight-only quantized DiT, and
  // unet_quantization "int8" produces the same classes at runtime.
  quantized_gemm_mode?: QuantizedGemmMode;
}

// aud2aud (cover): multipart -- prompt/lyrics/cover params + an uploaded
// `reference_audio` clip (handled by generateAud2Aud()). No audio_duration
// (derived server-side from the reference clip's length) and no
// sampler_mode (unlike txt2aud); adds cover_strength.
export interface Aud2AudParams {
  prompt: string;              // caption text (the reference is re-rendered under this)
  lyrics?: string;
  seed?: number;                // default -1
  inference_steps?: number;    // turbo distilled default 8
  guidance_scale?: number;     // turbo is CFG-distilled; default 1.0
  shift?: number;               // default 3.0
  cover_strength?: number;      // 0-1 step-count blend toward the reference; default 1.0
  vocal_language?: string;      // default "en"
  loras?: LoRAConfig[];
  mode?: "cover" | "repaint";    // default "cover"; "repaint" regenerates only [repaint_start, repaint_end)
  repaint_start?: number;        // seconds, repaint mode only; default 0.0
  repaint_end?: number;          // seconds, repaint mode only; default 0.0
  // --- MiniMax Music 3 repaint ONLY (mode MUST be "repaint" for this arch;
  // "cover" is refused server-side -- see routes.py's generate_aud2aud
  // docstring). Two honest sub-mechanisms, selected by `music3_repaint_mode`,
  // with DIFFERENT snapping/refusal rules for `repaint_start`/`repaint_end`
  // each -- see `_minimax_music3_repaint_regenerate`/`_minimax_music3_repaint_
  // rerender` in core/pipeline_backends/minimax_music3.py, the source of
  // truth for both:
  //   "regenerate" -- AR-resume with a NEW tail from `repaint_start` onward:
  //     CONTENT changes from that point on, everything before it is preserved
  //     sample-exact. Only `repaint_start` is snapped server-side, to the
  //     nearest chunk-window start that is NOT the song's very first chunk
  //     (so it can land at 0 even for a 0 request); `repaint_end` is used
  //     RAW as an upper bound on the new tail's TOTAL song length, not a
  //     window end and not itself snapped. A source song with fewer than two
  //     chunk windows is refused outright ("requires a longer source song").
  //   "rerender" -- the codes never change, only the flow-matching stage's
  //     rendering of [repaint_start, repaint_end) is redone with a new seed:
  //     timbre/mix change, lyrics/melody/timing do NOT. BOTH `repaint_start`
  //     and `repaint_end` are snapped server-side to chunk-window boundaries
  //     (independently of each other, via different logic per endpoint), so
  //     the effective range may differ from what was requested.
  // `undefined` lets the backend resolve MiniMax Music 3's own default
  // ("regenerate") from `aud2aud_defaults_for_arch` -- see generateAud2Aud's
  // FormData sender for why this must never be sent as an explicit `null`.
  music3_repaint_mode?: "regenerate" | "rerender";
  // Flow-matching steps PER CHUNK for the mechanism above (distinct from
  // ACE-Step's per-song `inference_steps` above; same field/semantics as
  // Txt2AudParams.num_inference_steps / OutpaintAudioParams.num_inference_steps).
  // `undefined` resolves to 30.
  num_inference_steps?: number;
  // Flow-stage CFG for the mechanism above (distinct from ACE-Step's
  // `guidance_scale` above; same field/semantics as
  // Txt2AudParams.flow_guidance_scale). `undefined` resolves to 1.7.
  flow_guidance_scale?: number;
  // Weight-only quantization of the ACE-Step DiT. Only "int8" is applied on
  // this architecture (a one-time in-place conversion of the audio DiT -- NOT
  // the Oobleck VAE or the Qwen3-Embedding text encoder); the FP8 values warn
  // and are ignored. acestep is in runtime_int8_archs.
  unet_quantization?: string | null;
  // Per-generation GEMM path for ALREADY-quantized Linear weights (null =
  // leave the process flags alone). acestep is in quantized_linear_archs: its
  // loader swaps in Int8Linear/Fp8Linear for a weight-only quantized DiT, and
  // unet_quantization "int8" produces the same classes at runtime.
  quantized_gemm_mode?: QuantizedGemmMode;
}

// Audio temporal outpaint (ACE-Step 1.5 extend): place a (optionally
// trimmed) input clip at a time offset inside a LONGER total_duration output
// timeline and generate the audio before/and-or after it, preserving the
// placed input sample-exact (see core/pipeline_backends/acestep.py
// AceStepMixin._generate_audoutpaint_acestep + OUTPAINT_AUDIO_DEFAULTS,
// param_defaults.py). Structurally the INVERSE of Aud2AudParams'
// `mode="repaint"`: repaint holds everything OUTSIDE a window and generates
// INSIDE it; outpaint holds the placed input window itself and generates
// OUTSIDE it. No mode/cover_strength/repaint_start/repaint_end -- outpaint
// has no cover/repaint sub-mode, it always holds the placed span.
export interface OutpaintAudioParams {
  prompt: string;              // caption text (also accepted as "caption"). MiniMax Music 3: IGNORED (sidecar's own is always reused)
  lyrics?: string;              // MiniMax Music 3: IGNORED, same reason as prompt
  seed?: number;                // default -1
  inference_steps?: number;    // ACE-Step ONLY; turbo distilled default 8
  guidance_scale?: number;     // ACE-Step ONLY; turbo is CFG-distilled; default 1.0
  shift?: number;               // ACE-Step ONLY; default 3.0
  vocal_language?: string;      // ACE-Step ONLY; default "en"
  loras?: LoRAConfig[];
  // --- Placement (ACE-Step ONLY), all in SECONDS ---
  total_duration?: number;         // output timeline length; (0, 240], default 60.0
  input_offset_sec?: number;       // where the (trimmed) clip lands, snapped server-side to 1/25s
  input_trim_start_sec?: number;   // trim applied to the UPLOADED clip before placement
  input_trim_end_sec?: number;
  // --- MiniMax Music 3 extend ONLY ---
  // Required when this architecture is loaded -- NO default anywhere in this
  // repo (backend/api/param_defaults.py's OUTPAINT_AUDIO_ARCH_OVERLAYS
  // deliberately omits one). The only value the causal autoregressive stage
  // can ever honor is "extend_forward"; omitting this field is a 400, not a
  // silent fallback -- see `audioOutpaintPlacements()`.
  placement?: "extend_forward";
  // How much MORE audio to generate, in seconds -- an UPPER BOUND, not a
  // target (same "duration is an upper bound" semantics as txt2aud's
  // audio_duration). `undefined` lets the backend resolve MiniMax Music 3's
  // own default (30.0) server-side, from `outpaint_audio_defaults_for_arch`.
  extend_duration_sec?: number;
  // Flow-matching steps PER CHUNK for the new tail (distinct from ACE-Step's
  // per-song `inference_steps` above; same field/semantics as
  // Txt2AudParams.num_inference_steps). `undefined` resolves to 30.
  num_inference_steps?: number;
  // Flow-stage CFG for the new tail (distinct from ACE-Step's
  // `guidance_scale` above; same field/semantics as
  // Txt2AudParams.flow_guidance_scale). `undefined` resolves to 1.7.
  flow_guidance_scale?: number;
  // Weight-only quantization of the ACE-Step DiT. Only "int8" is applied on
  // this architecture (a one-time in-place conversion of the audio DiT -- NOT
  // the Oobleck VAE or the Qwen3-Embedding text encoder); the FP8 values warn
  // and are ignored. acestep is in runtime_int8_archs.
  unet_quantization?: string | null;
  // Per-generation GEMM path for ALREADY-quantized Linear weights (null =
  // leave the process flags alone). acestep is in quantized_linear_archs: its
  // loader swaps in Int8Linear/Fp8Linear for a weight-only quantized DiT, and
  // unet_quantization "int8" produces the same classes at runtime.
  quantized_gemm_mode?: QuantizedGemmMode;
}

// ---------------------------------------------------------------------------
// Schema defaults — fetched once at startup, backend is source of truth
// ---------------------------------------------------------------------------

export interface GenerationDefaultsResponse {
  txt2img: Partial<GenerationParams> & Record<string, unknown>;
  img2img: Partial<GenerationParams> & Record<string, unknown>;
  inpaint:  Partial<InpaintParams> & Record<string, unknown>;
  outpaint: Partial<OutpaintParams> & Record<string, unknown>;
  outpaint_vid: Partial<OutpaintVideoParams> & Record<string, unknown>;
  inpaint_vid: Partial<InpaintVideoParams> & Record<string, unknown>;
  outpaint_aud: Partial<OutpaintAudioParams> & Record<string, unknown>;
  upscale: Partial<UpscaleParams> & Record<string, unknown>;
  txt2vid: Partial<Txt2VidParams> & Record<string, unknown>;
  img2vid: Partial<Img2VidParams> & Record<string, unknown>;
  txt2aud: Partial<Txt2AudParams> & Record<string, unknown>;
  aud2aud: Partial<Aud2AudParams> & Record<string, unknown>;
  // Per-architecture video overrides (backend VIDEO_GEN_ARCH_OVERLAYS /
  // OUTPAINT_VIDEO_ARCH_OVERLAYS). A video default resolves as
  // `base | video_arch_overlays[arch]`, and an outpaint-video one as that
  // again with `outpaint_video_arch_overlays[arch]` on top -- exactly what the
  // routes do server-side for every field a request omits. Optional so an
  // older backend without the keys still type-checks.
  video_arch_overlays?: Record<string, Record<string, unknown>>;
  outpaint_video_arch_overlays?: Record<string, Record<string, unknown>>;
  // Same thing for the keys that exist only on /generate/inpaint/video
  // (currently `inpaint_video_audio_mode`).
  inpaint_video_arch_overlays?: Record<string, Record<string, unknown>>;
  // Per-architecture audio overrides (backend AUDIO_GEN_ARCH_OVERLAYS /
  // param_defaults.audio_defaults_for_arch), the audio equivalent of
  // `video_arch_overlays`. A txt2aud default resolves as
  // `base | audio_arch_overlays[arch]` -- `txt2aud` above is ACE-Step-shaped
  // (audio_duration 30.0, no num_inference_steps/flow_guidance_scale keys at
  // all), and MiniMax Music 3 overlays its own audio_duration (60.0),
  // num_inference_steps (30) and flow_guidance_scale (1.7) on top of it.
  // Optional so an older backend without the key still type-checks.
  audio_arch_overlays?: Record<string, Record<string, unknown>>;
  // Per-architecture overrides for `aud2aud` (backend AUD2AUD_GEN_ARCH_OVERLAYS,
  // empty until MiniMax Music 3 repaint/cover lands) and for the KEYS UNIQUE
  // to `outpaint_aud` (backend OUTPAINT_AUDIO_ARCH_OVERLAYS -- populated by
  // MiniMax Music 3 extend: extend_duration_sec/num_inference_steps/
  // flow_guidance_scale). An `outpaint_aud` default resolves as
  // `base | aud2aud_arch_overlays[arch] | outpaint_audio_arch_overlays[arch]`,
  // the same two-layer composition `outpaint_audio_defaults_for_arch` does
  // server-side -- see `OutpaintPanel.tsx`'s per-arch overlay effect, which
  // reads these two maps directly rather than through a merged helper (so
  // it can tell "this key has no arch-specific default" apart from "this
  // key's default is the base value"). Optional so an older backend without
  // the keys still type-checks.
  aud2aud_arch_overlays?: Record<string, Record<string, unknown>>;
  outpaint_audio_arch_overlays?: Record<string, Record<string, unknown>>;
  // Per-architecture image overrides (backend IMAGE_GEN_ARCH_OVERLAYS /
  // param_defaults.image_defaults_for_arch), the image equivalent of
  // `video_arch_overlays`. A `txt2img`/`img2img`/`inpaint`/`outpaint` default
  // resolves as `base | image_arch_overlays[arch]` -- currently only
  // SenseNova U1.5 has an entry (steps 50, cfg_scale 4.0, vs. the shared
  // 20/7.0). Optional so an older backend without the key still type-checks.
  image_arch_overlays?: Record<string, Record<string, unknown>>;
  // User-overridable slider/number-input UPPER BOUNDS registry (backend
  // PARAM_BOUNDS). Optional so an older backend without the key still
  // type-checks. See frontend/src/utils/paramBounds.ts's resolveBound().
  param_bounds?: ParamBoundsRegistry;
}

// One PARAM_BOUNDS entry (backend/api/param_defaults.py). `builtin` is
// today's literal (what a control falls back to with no user override and no
// architecture clamp); `floor`/`ceiling` bound what a user override may be
// set to (enforced server-side too, in save_generation_settings).
export interface ParamBoundSpec {
  builtin: number;
  floor: number;
  ceiling: number;
  family: string;
  label: string;
}
export type ParamBoundsRegistry = Record<string, ParamBoundSpec>;

// The outpaint-video defaults for one architecture, resolved from the SAME
// three layers the backend resolves them from, in the same order.
export const outpaintVideoDefaultsForArch = (
  defaults: GenerationDefaultsResponse | null | undefined,
  arch: string | null | undefined
): Record<string, unknown> => ({
  ...(defaults?.outpaint_vid || {}),
  ...((arch && defaults?.video_arch_overlays?.[arch]) || {}),
  ...((arch && defaults?.outpaint_video_arch_overlays?.[arch]) || {}),
});

// The txt2aud defaults for one architecture, resolved from the SAME two
// layers `param_defaults.audio_defaults_for_arch` resolves them from: the
// ACE-Step-shaped base (`txt2aud`) with `audio_arch_overlays[arch]` on top.
// An arch with no overlay (ACE-Step, or an unrecognized/unloaded arch)
// resolves to the base unchanged.
export const audioDefaultsForArch = (
  defaults: GenerationDefaultsResponse | null | undefined,
  arch: string | null | undefined
): Record<string, unknown> => ({
  ...(defaults?.txt2aud || {}),
  ...((arch && defaults?.audio_arch_overlays?.[arch]) || {}),
});

// The inpaint-video defaults for one architecture, same three layers in the
// same order (`inpaint_video_defaults_for_arch` server-side).
export const inpaintVideoDefaultsForArch = (
  defaults: GenerationDefaultsResponse | null | undefined,
  arch: string | null | undefined
): Record<string, unknown> => ({
  ...(defaults?.inpaint_vid || {}),
  ...((arch && defaults?.video_arch_overlays?.[arch]) || {}),
  ...((arch && defaults?.inpaint_video_arch_overlays?.[arch]) || {}),
});

export const fetchGenerationDefaults = async (): Promise<GenerationDefaultsResponse> =>
  (await api.get("/schema/generation-defaults")).data;

// User-configured generation settings (backend UserSettings row, GET/POST
// /settings/generation) -- distinct from the schema *defaults* above:
// per-installation, backend-persisted config rather than a request-shape
// default. `video_frame_slider_max` is the upper bound for the video
// frame-count SLIDER TRACK (VideoFrameCountSlider); `null` means unset.
export interface GenerationSettingsResponse {
  inpaint_use_dedicated_model: boolean;
  video_frame_slider_max: number | null;
  // User overrides for slider/number-input UPPER BOUNDS, keyed by bound name
  // (see param_defaults.py's PARAM_BOUNDS for the eligible keys). A key
  // absent means that bound uses its builtin value. Always an object.
  slider_bounds: Record<string, number>;
}

export const fetchGenerationSettings = async (): Promise<GenerationSettingsResponse> =>
  (await api.get("/settings/generation")).data;

// Persists ONE field of GenerationSettingsResponse. POST /settings/generation
// only updates the keys present in the body (see routes.py's
// save_generation_settings), so this does not disturb
// inpaint_use_dedicated_model, which is saved separately by
// GenerationSettings.tsx's own Save button.
//
// This is a commit-time write (call it from a NumberInput's onCommit /a
// checkbox's onChange, never per keystroke). The caller is responsible for
// also updating the live value the panels read (StartupContext's
// videoFrameSliderMax / setVideoFrameSliderMax) -- this function only
// persists to the backend.
export const saveVideoFrameSliderMax = async (
  value: number | null
): Promise<GenerationSettingsResponse> =>
  (await api.post("/settings/generation", { video_frame_slider_max: value })).data.settings;

// Persists a partial update of the `slider_bounds` override map (see
// PARAM_BOUNDS in param_defaults.py). `null` for a key RESETS that one bound
// to its builtin. Backend validates unknown keys / out-of-[floor,ceiling]
// values with a 400 (see save_generation_settings). Same commit-time-write
// contract as saveVideoFrameSliderMax -- the caller updates StartupContext's
// sliderBounds/setSliderBounds itself on success.
export const saveSliderBounds = async (
  overrides: Record<string, number | null>
): Promise<GenerationSettingsResponse> =>
  (await api.post("/settings/generation", { slider_bounds: overrides })).data.settings;

export const fetchTrainingDefaults = async (): Promise<Record<string, unknown>> =>
  (await api.get("/schema/training-defaults")).data;

export const fetchTaggerTrainingDefaults = async (): Promise<Record<string, unknown>> =>
  (await api.get("/schema/tagger-training-defaults")).data;

// Defaults for a decoder-only VAE fine-tune (training_method "vae_decoder").
// Backed by backend/api/param_defaults.py VAE_TRAINING_DEFAULTS.
export const fetchVaeTrainingDefaults = async (): Promise<Record<string, unknown>> =>
  (await api.get("/schema/vae-training-defaults")).data;

// Per-architecture default timestep_sampling configs (e.g. { _default: {...}, minit2i: {...} }).
// The training UI applies the selected model's entry when the base model changes.
export const fetchTimestepDefaultsByArch = async (): Promise<Record<string, Record<string, unknown>>> =>
  (await api.get("/schema/timestep-defaults-by-arch")).data;

// Per-architecture default bundle_vae (full-parameter save VAE embedding);
// e.g. { _default: false, sd15: true, sdxl: true, deus: true }.
export const fetchBundleVaeDefaultsByArch = async (): Promise<Record<string, boolean>> =>
  (await api.get("/schema/bundle-vae-defaults-by-arch")).data;

// Per-architecture capability matrix (GET /schema/arch-capabilities). Mirrors
// backend/api/arch_capabilities.py: `unsupported[arch][feature]` is a factual
// one-line reason the feature has NO effect on that architecture. Panels use it
// to hide a control the loaded architecture would ignore, so the arch list lives
// in exactly one place (the backend table) instead of being duplicated here.
export interface ArchCapabilities {
  unsupported: Record<string, Record<string, string>>;
  // Values of a feature's arming parameter that the arch DOES honor even though
  // the feature is listed in `unsupported` (e.g. unet_quantization="int8" on
  // krea2). Optional so an older backend without the key still type-checks.
  supported_values?: Record<string, Record<string, string[]>>;
  feature_params: Record<string, string[]>;
  feature_labels: Record<string, string>;
  // Architectures whose transformer the in-place weight-only INT8 converter is
  // wired for, i.e. the ones that honor unet_quantization="int8". Served
  // straight from backend RUNTIME_INT8_ARCHS
  // (core/models/common/int8_runtime_quantize.py). Optional so an older backend
  // without the key still type-checks.
  runtime_int8_archs?: string[];
  // Architectures whose LOADERS swap in the weight-only quantized Linear classes
  // (Int8Linear / Fp8Linear), i.e. the ones where quantized_gemm_mode selects
  // anything. Served straight from backend QUANTIZED_LINEAR_ARCHS (same module);
  // a superset of runtime_int8_archs in general, equal to it today. Optional so
  // an older backend without the key still type-checks.
  quantized_linear_archs?: string[];
  // Per-VIDEO-arch TemporalSpec: the same table the video routes validate (and,
  // where the arch says so, snap) against. Present only for video
  // architectures. Optional so an older backend still type-checks.
  video_constraints?: Record<string, VideoConstraints>;
  // arch -> training method -> the factual reason that method is REFUSED.
  //
  // A DIFFERENT axis from `unsupported`, which is about generation parameters
  // that are accepted and ignored. An entry here means the trainer RAISES, so a
  // client must not offer the method: MiniMax-H3 declares `full_finetune`
  // unsupported (33 B dense DiT, weight-only FP8 base). Served straight from the
  // backend's TRAINING_UNSUPPORTED table so the dropdown, the refusal and the
  // reason shown to the user all come from one source. Optional so an older
  // backend without the key still type-checks.
  training_unsupported?: Record<string, Record<string, string>>;
  // arch -> training CONFIG FEATURE -> why the trainer has no such mechanism
  // there (block swap, fused optimizer groups, reference images, text-encoder
  // training, in-training samples, VAE settings).
  //
  // A THIRD axis, next to `unsupported` (generation params ignored) and
  // `training_unsupported` (whole methods refused). ABSENT MEANS SUPPORTED, so
  // an architecture the backend does not know about keeps every control instead
  // of silently losing one. The declaration is the backend's because the fact is
  // a property of the trainer; the form must not re-derive it from arch names.
  training_feature_unsupported?: Record<string, Record<string, TrainingFeatureRefusal>>;
  training_feature_params?: Record<string, string[]>;
  training_feature_labels?: Record<string, string>;
  // arch -> training config parameter -> the value that architecture REQUIRES,
  // with the factual reason and an optional training-method scope.
  //
  // A FOURTH axis: the three above say what is missing, this one says what a
  // parameter must BE. SenseNova implements full fine-tuning under a contract
  // that fixes the optimizer and the batch size, refused before the model
  // loads. ABSENT MEANS UNCONSTRAINED. Optional so an older backend without the
  // key still type-checks.
  training_required_values?: Record<string, Record<string, TrainingRequiredValue>>;
  // arch -> training config FEATURE -> what the backend says about a feature it
  // DOES implement: `{level, reason, methods?}`. A FIFTH axis, and the only one
  // that refuses nothing (see `trainingFeatureAdvisory` below). Optional so an
  // older backend without the key still type-checks.
  training_feature_advisory?: Record<string, Record<string, TrainingFeatureAdvisory>>;
  // Architecture-specific controls inside the training-sample section. Common
  // prompt/size/steps/CFG/seed fields are always handled separately. An empty
  // list means the architecture uses fixed sampling internals.
  training_sample_supported_params?: Record<string, string[]>;
  training_sample_notes?: Record<string, string>;
  // Architecture id -> its user-facing spelling ("sensenova" -> "SenseNova
  // U1.5"), from the backend's ARCH_DISPLAY_NAMES. An id with no entry falls
  // back to the id, so a new architecture shows up (unprettified) rather than
  // disappearing.
  arch_display_names?: Record<string, string>;
  // arch -> at which stage the TRAINER can build that architecture's inference
  // CFG uncond condition: null (it cannot), "collated" or "encode". Mirrors
  // ArchHandler.cfg_null_stage. null is why an explicit cfg_uncond_drop_rate --
  // 0.0 included -- is a 400 there; the reason to show lives in
  // training_feature_unsupported[arch].cfg_uncond_drop. Optional so an older
  // backend without the key still type-checks.
  cfg_null_stage?: Record<string, "collated" | "encode" | null>;
  // arch -> what an OMITTED cfg_uncond_drop_rate resolves to there. Absent
  // arch = the mechanism is not in play for it. Read this instead of keeping a
  // copy of the number in the form.
  cfg_uncond_drop_defaults?: Record<string, number>;
  // arch -> what a long-form video CHAIN's continuation segments receive from
  // their predecessor there (design §7.1). The backend's loaded variant plus
  // this table is the authority on which continuation modes exist — a client
  // must not branch on a checkpoint name. Optional so an older backend without
  // the key still type-checks. Present only for video architectures.
  chain_context?: Record<string, ChainContextCapability>;
  // Audio equivalent of `video_constraints[arch].outpaint_placements`: which
  // `placement` values POST /generate/outpaint/audio accepts for `arch`.
  // MiniMax Music 3: `["extend_forward"]` (its autoregressive stage is a
  // causal language model). ACE-Step is ABSENT -- its placement is a
  // continuous total_duration/input_offset_sec offset, not an enumerated
  // set. Optional so an older backend without the key still type-checks.
  audio_outpaint_placements?: Record<string, string[]>;
  // Sibling table for POST /generate/aud2aud's `music3_repaint_mode` field
  // (repaint mode only): which sub-mechanisms `arch` offers. MiniMax Music 3:
  // `["regenerate", "rerender"]`. ACE-Step is ABSENT -- its own aud2aud has no
  // such sub-mode concept at all (mode=cover/repaint directly, no further
  // choice). Optional so an older backend without the key still type-checks.
  aud2aud_music3_repaint_modes?: Record<string, string[]>;
}

// One architecture's (or one loaded transformer variant's) chain-context
// capability, straight from `CHAIN_CONTEXT` in
// backend/api/arch_capabilities.py. Served by GET /schema/arch-capabilities and
// used by POST /video-chain/plan|validate to refuse an unadvertised
// `continuation_mode` with a 400 — so a mode offered in the UI must come from
// `chain_continuation_modes`, never from a hardcoded list here.
export interface ChainContextVariantCapability {
  // Only IMPLEMENTED modes appear. `VideoChainContinuationMode` is the wider
  // wire vocabulary (it names the not-yet-built candidates so they can be
  // refused by name).
  chain_continuation_modes: VideoChainContinuationMode[];
  chain_default_continuation_mode: VideoChainContinuationMode;
  // The `continuation_overlap_frames` range a mode that PINS an overlap
  // accepts — NOT a description of `boundary_frame`, whose shared anchor is
  // first-frame conditioning and takes no length. Valid lengths sit on
  // video-VAE group boundaries, i.e. the cumulative sums of
  // `video_constraints[arch].latent_chunk_pattern` (MiniMax-H3: 1, 5, 9, 13,
  // 17, 18, ...), which is why no second enumeration is served here.
  // The floor is measured (MiniMax-H3: 5 — a 1-frame pin is a motionless still
  // the model can continue as a static scene), so it filters the list rather
  // than snapping a request up to it.
  chain_context_min_frames: number;
  // null = unbounded: the architecture takes whatever the preserved prefix is
  // (LTX-2.3 places the whole accumulated clip as one video condition).
  chain_context_max_frames: number | null;
  // Several frames of the predecessor placed at chosen positions inside the
  // generated span (`continuation_mode: motion_preroll`). True for MiniMax-H3
  // `fl2va`; it says the placement exists, not that it is better -- the arm is
  // unmeasured and opt-in.
  chain_supports_sparse_motion_anchors: boolean;
  // The pre-roll bounds of `motion_preroll`, null when it is not advertised.
  // NOT `chain_context_min/max_frames`: a pre-roll needs no VAE-group
  // alignment (an anchor names a pixel frame directly), so any integer in
  // range is valid, and its floor is structural rather than measured.
  chain_motion_preroll_min_frames: number | null;
  chain_motion_preroll_max_frames: number | null;
  chain_motion_preroll_min_anchors: number | null;
  chain_motion_preroll_max_anchors: number | null;
  // Part of the preceding clip carried as a REFERENCE video, a separate channel
  // from the boundary anchor. MiniMax-H3 `ref2va` only.
  chain_supports_reference_video: boolean;
  // Earlier segments' frames survive the continuation pixel-exact.
  chain_supports_exact_prefix: boolean;
}

export interface ChainContextCapability extends ChainContextVariantCapability {
  // Loaded transformer variant -> its own capability, for the variants that
  // differ from the architecture-level entry. An absent variant answers with
  // the architecture-level entry (the conservative one).
  variants?: Record<string, ChainContextVariantCapability>;
}

// The chain-context capability for the LOADED arch/variant pair, or undefined
// when the architecture cannot be chained (or the matrix is not loaded). The
// variant is the one the backend reports for the loaded checkpoint
// (`currentModelInfo.model_info.variant`), never a file name.
export const chainContextCapability = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  variant?: string | null
): ChainContextVariantCapability | undefined => {
  const entry = arch ? caps?.chain_context?.[arch] : undefined;
  if (!entry) return undefined;
  const key = (variant || "").trim().toLowerCase();
  return entry.variants?.[key] ?? entry;
};

// The `continuation_overlap_frames` values a `pinned_tail` continuation can be
// given on this arch/variant, ascending. A latent frame is conditioned or
// generated whole, so the candidate lengths are the cumulative sums of the
// arch's `latent_chunk_pattern` — derived from the served pattern through the
// SAME `latentGroupSpans` the inpaint range uses, never a second hardcoded list
// (the pattern CYCLES: MiniMax-H3's [1,4,4,4,4] gives 1, 5, 9, 13, 17, 18, ...
// — 17 is followed by 18, not 33) — kept inside the served
// [min, max] window. Both bounds come from the backend, including the measured
// floor that keeps a one-frame pin off the list; a client that offered a
// shorter one would only earn a 400.
export const chainContinuationOverlapLengths = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  variant?: string | null
): number[] => {
  const capability = chainContextCapability(caps, arch, variant);
  const max = capability?.chain_context_max_frames ?? 0;
  const min = capability?.chain_context_min_frames ?? 1;
  const pattern = arch ? caps?.video_constraints?.[arch]?.latent_chunk_pattern : undefined;
  if (!max || !pattern?.length) return [];
  return latentGroupSpans(pattern, max)
    .map(([, end]) => end)
    .filter((end) => end >= min && end <= max);
};

// One training-config feature's refusal for one architecture.
export interface TrainingFeatureRefusal {
  reason: string;
  // Training methods the refusal applies to; absent = all of them (Z-Image
  // trains no text encoder under LoRA while a full fine-tune does).
  methods?: string[];
}

// The reason `feature` cannot run for `arch` under `method`, or undefined when
// it can. Undefined for an unknown arch or an unloaded matrix: the control stays
// visible and the backend refuses the run, which is recoverable — a control that
// vanishes because the frontend has never heard of the architecture is not.
export const trainingFeatureUnsupportedReason = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  feature: string,
  method?: string | null
): string | undefined => {
  if (!arch) return undefined;
  const entry = caps?.training_feature_unsupported?.[arch]?.[feature];
  if (!entry) return undefined;
  if (entry.methods && method && !entry.methods.includes(method)) return undefined;
  return entry.reason;
};

// Whether `arch`'s training-sample path READS `parameter`, i.e. whether its
// control is worth offering. Same direction as trainingFeatureUnsupportedReason
// above, and for the same reason: an unknown arch or an unloaded matrix answers
// TRUE, so the control stays visible and the value is simply not written for an
// architecture that turns out not to read it. The opposite direction makes every
// sample control vanish on a startup fetch that has not landed yet — including
// the sampler and schedule selects, which were unconditional before this gate
// existed — and a control that disappears because the frontend has not heard
// back from the backend is not recoverable by the user.
//
// Matches api/arch_capabilities.training_sample_key_supported, which gates the
// generated YAML and the sample PNG's metadata on the backend and fails open on
// the same input.
export const trainingSampleParameterSupported = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  parameter: string
): boolean => {
  const table = caps?.training_sample_supported_params;
  if (!arch || !table || !table[arch]) return true;
  return table[arch].includes(parameter);
};

export const trainingSampleNote = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): string | undefined => arch ? caps?.training_sample_notes?.[arch] : undefined;

// What an OMITTED cfg_uncond_drop_rate resolves to on `arch`, or undefined when
// the architecture has no default (the mechanism is not in play there, or the
// matrix has not loaded). Never a literal in a component: the number is the
// backend's, and a second copy of it is what turns an explicit 0 back into 0.1.
export const cfgUncondDropDefault = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): number | undefined => {
  if (!arch) return undefined;
  return caps?.cfg_uncond_drop_defaults?.[arch];
};

// What the backend says ABOUT a training feature it does implement.
export interface TrainingFeatureAdvisory {
  // "high_memory" — the reason carries measured numbers; "experimental" — the
  // path is implemented and thinly measured. Neither is a gate.
  level: "experimental" | "high_memory";
  reason: string;
  methods?: string[];
}

// The advisory for `feature` on `arch` under `method`, or undefined when there
// is none. NEVER a reason to hide or disable a control: the backend accepts and
// runs the feature, so a caller that treats this like a refusal recreates the
// contradiction the axis exists to end.
export const trainingFeatureAdvisory = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  feature: string,
  method?: string | null
): TrainingFeatureAdvisory | undefined => {
  if (!arch) return undefined;
  const entry = caps?.training_feature_advisory?.[arch]?.[feature];
  if (!entry) return undefined;
  if (entry.methods && method && !entry.methods.includes(method)) return undefined;
  return entry;
};

// One training-config parameter's required value for one architecture.
export interface TrainingRequiredValue {
  value: string | number | boolean;
  reason: string;
  // Training methods the requirement applies to; absent = all of them.
  methods?: string[];
  // The full admitted set when the contract admits more than one value; `value`
  // is then its default member. Absent = `value` is the only legal one. A
  // control offers exactly these and leaves a current member alone.
  values?: (string | number | boolean)[];
}

// The config values `arch` requires under `method`, param -> {value, reason}.
// Empty for an unknown arch or an unloaded matrix: unconstrained, so a control
// keeps its own default and the backend refuses the run if that is wrong —
// recoverable, where a control pinned to a value invented here is not.
export const trainingRequiredValues = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  method?: string | null
): Record<string, TrainingRequiredValue> => {
  if (!arch) return {};
  const entries = caps?.training_required_values?.[arch];
  if (!entries) return {};
  const out: Record<string, TrainingRequiredValue> = {};
  for (const [param, entry] of Object.entries(entries)) {
    if (entry.methods && method && !entry.methods.includes(method)) continue;
    out[param] = entry;
  }
  return out;
};

// The reason `method` is refused for `arch`, or undefined when it is offered.
// Used to disable a training-method control AND to title it with the backend's
// own wording rather than a second copy of it in the UI.
export const trainingMethodUnsupportedReason = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  method: string
): string | undefined => {
  if (!arch) return undefined;
  return caps?.training_unsupported?.[arch]?.[method];
};

// One video architecture's temporal/spatial contract, straight from the
// backend's TemporalSpec (core/models/components/wiring.py). Every field is
// something a client cannot derive from the others, which is why they are all
// served rather than reconstructed here.
export interface VideoConstraints {
  frame_multiple: number;          // valid lengths are multiple*n + offset
  frame_offset: number;
  min_frames: number;              // production floor
  max_frames: number | null;
  // Advisory-only: the longest length the architecture was actually trained
  // on (MiniMax-H3: 362), served ONLY when `max_frames` is null for the same
  // arch -- i.e. the backend no longer enforces this as a hard single-
  // inference ceiling, but a length past it is documented-untested rather
  // than a fact the backend can vouch for. Optional so an older backend
  // without the key still type-checks (treated as "no advisory range" then).
  trained_max_frames?: number | null;
  min_decodable_frames: number;    // hard VAE floor, below the production one
  fps_fixed: number | null;        // non-null = the arch generates at this fps only
  // ORIENTATION-AGNOSTIC [short_edge, long_edge], NOT [height, width].
  max_pixel_hw: [number, number] | null;
  pixel_align: number;
  // What an off-grid/out-of-range length does: true = snapped up with a
  // warning (MiniMax-H3), false = 400 (LTX-2.3).
  snap_invalid_length: boolean;
  suggested_frames: number[];
  min_inference_steps: number;
  // true = the step count is a sigma GRID POINT count, so N drives N-1 model
  // evaluations (MiniMax-H3); false = N steps run N evaluations (LTX-2.3).
  steps_are_sigma_grid_points: boolean;
  // Where POST /generate/outpaint/video may place the input clip on this arch.
  // ["free"] (LTX-2.3) = any offset. ["extend_forward","extend_backward",
  // "bridge"] (MiniMax-H3) = boundary placements only: it conditions on the
  // first and/or last frame of the span it generates, so the clip must abut a
  // timeline boundary or bridge two clips. Optional so an older backend
  // without the key still type-checks (treated as "free" by the helper below).
  outpaint_placements?: string[];
  // The video VAE's temporal chunking, in pixel frames per latent frame,
  // cycled from the start of the clip (MiniMax-H3: [1,4,4,4,4]). A latent frame
  // is regenerated or preserved as a whole, so it is the addressable unit of
  // POST /generate/inpaint/video. [] / absent = the arch declares none and
  // temporal inpaint is refused on it.
  latent_chunk_pattern?: number[];
}

// `[pixelStart, pixelEndExclusive]` per latent frame, for a clip of `frames`
// pixel frames. Mirrors the backend's `latent_frame_spans`
// (backend/api/generation_utils.py): the pattern is cycled, and on a valid clip
// length the spans tile the clip exactly. An empty pattern yields no spans,
// which is how "this arch declares no chunking" reaches the UI.
export const latentGroupSpans = (
  pattern: number[] | undefined,
  frames: number
): [number, number][] => {
  const spans: [number, number][] = [];
  if (!pattern || pattern.length === 0 || frames <= 0) return spans;
  let cursor = 0;
  for (let index = 0; cursor < frames; index += 1) {
    const width = pattern[index % pattern.length];
    // `Math.min` here trims a final span that would run past `frames`; the
    // backend's `latent_frame_spans` does not do this trim, so on an
    // off-grid `frames` this span could come out narrower than the
    // backend's. Currently unreachable in practice because
    // `videoTrimmedLengthValid` (InpaintPanel.tsx) blocks submit unless
    // `frames` is already a multiple of the pattern.
    spans.push([cursor, Math.min(frames, cursor + width)]);
    cursor += width;
  }
  return spans;
};

// The range the server would actually regenerate for a requested `[start, end)`:
// expanded OUTWARD to latent-group boundaries, never shrunk — the same rule
// `plan_video_inpaint_span` applies. Returns the request unchanged when the arch
// declares no chunking (the backend refuses that case with its own message).
export const snapRangeToLatentGroups = (
  spans: [number, number][],
  start: number,
  end: number
): { start: number; end: number } => {
  if (!spans.length || !(start < end)) return { start, end };
  const touched = spans.filter(([lo, hi]) => lo < end && hi > start);
  if (!touched.length) return { start, end };
  return { start: touched[0][0], end: touched[touched.length - 1][1] };
};

// The longest clip length this architecture accepts that is <= `frames`, or
// null when even its shortest clip is longer. This is what a temporal-inpaint
// UI trims DOWN to: the clip length itself must be on the grid there, and the
// backend refuses an off-grid length rather than snapping it (snapping would
// delete frames the caller asked to keep).
export const largestValidVideoFrameCount = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number
): number | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || !Number.isFinite(frames)) return null;
  const ceiling = c.max_frames != null ? Math.min(frames, c.max_frames) : frames;
  const k = Math.floor((ceiling - c.frame_offset) / c.frame_multiple);
  const length = k * c.frame_multiple + c.frame_offset;
  return length >= c.min_frames ? length : null;
};

// The valid clip length ON THE GRID (`multiple*n + offset`) closest to
// `frames`, clamped into `[min_frames, max_frames]` first. Unlike
// `largestValidVideoFrameCount` (which only ever snaps DOWN, because a
// temporal-inpaint clip length must not silently grow past what the caller
// trimmed to), this is for a control that lets the user ask for ANY length —
// a slider/number box, not a trim target — so the nearest grid point in
// EITHER direction is the right answer, the same as how a drag handle lands
// on the nearest tick. Returns null on the same "arch unknown / matrix not
// loaded" condition its neighbours do.
export const nearestValidVideoFrameCount = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number
): number | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || !Number.isFinite(frames)) return null;
  const kMin = Math.ceil((c.min_frames - c.frame_offset) / c.frame_multiple);
  const kMax = c.max_frames != null
    ? Math.floor((c.max_frames - c.frame_offset) / c.frame_multiple)
    : Infinity;
  if (kMax < kMin) return null;
  const kRaw = (frames - c.frame_offset) / c.frame_multiple;
  const k = Math.min(kMax, Math.max(kMin, Math.round(kRaw)));
  return k * c.frame_multiple + c.frame_offset;
};

// The valid clip length the BACKEND would produce for a requested one: the
// grid point at or above `frames`, clamped into the arch's producible range.
// This mirrors `TemporalSpec.snap_length` (backend/core/models/components/
// wiring.py) exactly, including its floor of `max(min_frames,
// min_decodable_frames)` and its silent clamp at `max_frames` -- so a panel
// can show the length a request will ACTUALLY come back as, before spending
// the generation to find out. It rounds UP where `nearestValidVideoFrameCount`
// rounds to whichever side is closer, because that is what the backend does;
// do not swap one for the other to save a helper.
export const snapUpValidVideoFrameCount = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number
): number | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || !Number.isFinite(frames)) return null;
  const lo = Math.max(c.min_frames, c.min_decodable_frames);
  const kLo = Math.ceil((lo - c.frame_offset) / c.frame_multiple);
  let k = Math.max(Math.ceil((frames - c.frame_offset) / c.frame_multiple), kLo);
  if (c.max_frames != null) {
    k = Math.min(k, Math.floor((c.max_frames - c.frame_offset) / c.frame_multiple));
  }
  return k * c.frame_multiple + c.frame_offset;
};

// --- Opt-in video length chaining (frontend-only orchestration) ---
//
// A video architecture's `max_frames` is a SINGLE-INFERENCE limit
// (backend/core/models/components/wiring.py's TemporalSpec), not a hard wall:
// a clip longer than it can only be reached by chaining several requests
// together via POST /generate/outpaint/video's `extend_forward` placement,
// each one continuing from the previous segment's own output. This is never
// automatic (see CLAUDE.md / the panels that call it) because a continuation
// segment is conditioned on the BOUNDARY FRAME of the clip it continues from
// (plus, for ref2va/ia2v, original image references and an automatic video
// reference derived from the previous segment's end), not the rest of its
// content, while the SAME full-length prompt is resent unchanged on every
// segment — prompt adherence degrades across segment boundaries in a way a
// single inference does not have.
//
// The arithmetic below mirrors OutpaintPanel's own extend_forward handling
// (`preservedFrames + effectiveGeneratedFrames - sharedAnchorFrames`): the
// GENERATED span (not the request's `total_frames`) is what has to land on
// `max_frames`, because the preserved (already-produced) prefix is placed,
// not regenerated. `sharedAnchorFrames` is 1 for extend_forward with no
// bridge clip (the placement this feature always uses).
const VIDEO_CHAIN_ANCHOR_FRAMES = 1;

// The per-segment length chaining arithmetic should use: the caller-supplied
// `segmentFrames` (`chain_segment_frames`, user-settable, null/undefined =
// unset) when it is a positive finite number, otherwise the architecture's
// own single-inference cap (`max_frames`) when it still has one, otherwise
// no cap at all (`Infinity`, meaning "nothing to chain unless the user opts
// in").
//
// `chain_segment_frames` is independent client-side orchestration state, NOT
// a backend parameter -- the backend only ever sees the resulting
// `total_frames` on each independent request (see videoChain.ts's header).
// Its default (unset) intentionally falls back to `max_frames` rather than
// straight to `Infinity`: that keeps every architecture that still has a
// real single-inference wall (LTX-2.3) chaining automatically exactly as it
// did before this control existed, while fixing the regression this control
// was added to fix -- an architecture whose `max_frames` went null
// (MiniMax-H3; see `trained_max_frames`) no longer has ANY server-enforced
// wall, so with no explicit segment length from the user there is nothing to
// split on, and "raise the total, nothing splits" is correct there by
// default. Setting `chain_segment_frames` is what turns chaining into a
// voluntary choice on an uncapped architecture too (e.g. to keep every
// segment within the documented trained range for quality, even though the
// backend would accept one huge request).
const chainSegmentCap = (
  c: VideoConstraints | undefined,
  segmentFrames: number | null | undefined
): number => {
  if (segmentFrames != null && Number.isFinite(segmentFrames) && segmentFrames > 0) {
    return segmentFrames;
  }
  return c?.max_frames ?? Number.POSITIVE_INFINITY;
};

// The `total_frames` value to send to POST /generate/outpaint/video to
// continue a chain that has produced `accumulatedFrames` so far, aiming for
// `targetFrames` overall. Also, on success, the new accumulated frame count
// (the extend_forward output is exactly this many frames — preserved prefix
// plus the newly generated span, minus the one shared anchor frame). Returns
// null when there is no effective per-segment cap to chain against (see
// `chainSegmentCap`) or the segment would make no forward progress (guards
// against a pathological arch table looping forever).
export const nextVideoChainTotalFrames = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  accumulatedFrames: number,
  targetFrames: number,
  segmentFrames?: number | null
): number | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c) return null;
  const cap = chainSegmentCap(c, segmentFrames);
  if (!Number.isFinite(cap)) return null;
  const remaining = targetFrames - accumulatedFrames;
  if (remaining <= 0) return null;
  const requestedGenerated = Math.min(remaining, cap);
  const generatedSpan = snapUpValidVideoFrameCount(caps, arch, requestedGenerated);
  if (generatedSpan == null || generatedSpan <= VIDEO_CHAIN_ANCHOR_FRAMES) return null;
  return accumulatedFrames + generatedSpan - VIDEO_CHAIN_ANCHOR_FRAMES;
};

// The client-side plan for reaching `targetFrames` when the effective
// per-segment cap (`chainSegmentCap`: `segmentFrames` if the user set one,
// else the architecture's `max_frames`, else uncapped) is below it, using the
// same segment-by-segment arithmetic `nextVideoChainTotalFrames` applies at
// execution time.
//
// Returns null for three DIFFERENT reasons that all mean "nothing to plan",
// kept as separate early-returns (not folded into one condition) so each
// stays independently readable/greppable even though the caller only ever
// sees "no plan":
//   1. the arch/matrix is unknown or `targetFrames` is not a real number;
//   2. there is no effective segment cap at all (uncapped arch, no
//      `segmentFrames` set) -- nothing CAN be chained, by design;
//   3. `targetFrames` already fits inside one segment -- chaining is not
//      NEEDED.
// A caller that must tell these apart (e.g. to phrase "nothing to chain
// automatically -- set a segment length" vs "already fits") calls
// `chainSegmentCap`-derived logic itself; the Generate-time gate
// (`if (chainPlan != null)`) only ever needed the null/non-null distinction,
// which this preserves.
export interface VideoChainPlan {
  capFrames: number;
  segments: number;    // total requests, INCLUDING segment 1
  finalFrames: number;  // the clip length the chain actually reaches
}

export const planVideoChain = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  targetFrames: number,
  segmentFrames?: number | null
): VideoChainPlan | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || !Number.isFinite(targetFrames)) return null;
  const cap = chainSegmentCap(c, segmentFrames);
  if (!Number.isFinite(cap)) return null;
  if (targetFrames <= cap) return null;

  let accumulated = cap; // segment 1: a normal request, at the segment cap
  let segments = 1;
  // Bounded so a pathological arch table can never loop forever; 500
  // segments is far beyond anything this feature would reasonably run.
  for (let guard = 0; guard < 500 && accumulated < targetFrames; guard++) {
    const next = nextVideoChainTotalFrames(caps, arch, accumulated, targetFrames, segmentFrames);
    if (next == null) break;
    accumulated = next;
    segments += 1;
  }
  return { capFrames: cap, segments, finalFrames: accumulated };
};

// Per-CONTINUATION-segment `total_frames` values for reaching `targetFrames`,
// i.e. everything `planVideoChain` above computes except segment 1 itself
// (which is always a plain request at the effective segment cap). Used to
// give the queue items for segments 2..N a real initial `total_frames` at
// enqueue time, so the whole plan is visible in the queue immediately -- each
// item's value is still re-derived from the ACTUAL previous segment's
// reported frame count right before that item runs (see videoChain.ts),
// because a real generation can snap slightly differently than this
// pre-flight estimate. Returns null under the same "nothing to chain"
// conditions as `planVideoChain`.
export const planVideoChainSegments = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  targetFrames: number,
  segmentFrames?: number | null
): number[] | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || !Number.isFinite(targetFrames)) return null;
  const cap = chainSegmentCap(c, segmentFrames);
  if (!Number.isFinite(cap)) return null;
  if (targetFrames <= cap) return null;

  const segments: number[] = [];
  let accumulated = cap;
  for (let guard = 0; guard < 500 && accumulated < targetFrames; guard++) {
    const next = nextVideoChainTotalFrames(caps, arch, accumulated, targetFrames, segmentFrames);
    if (next == null) break;
    segments.push(next);
    accumulated = next;
  }
  return segments;
};

// The length of clip ANY SINGLE generation request in a chain (or a plain,
// unchained request) can actually produce: `requestedFrames` itself when it
// already fits in one segment, otherwise the effective per-segment cap
// (`chainSegmentCap`: the user's `chain_segment_frames` if set, else the
// architecture's single-inference cap, else uncapped). Nothing this feature
// ever sends to a generation endpoint -- segment 1 of a chain, a `Generate at
// cap` single inference, or an unchained request -- is longer than this, so
// it is what a per-segment duration (H3 Prompt Assist) must be computed
// from, never `requestedFrames` itself. Falls back to `requestedFrames` when
// the arch/matrix is unknown, the same "assume supported" convention
// `archSupportsFeature` uses.
export const effectiveSegmentFrames = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  requestedFrames: number,
  segmentFrames?: number | null
): number => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c) return requestedFrames;
  const cap = chainSegmentCap(c, segmentFrames);
  if (!Number.isFinite(cap)) return requestedFrames;
  return Math.min(requestedFrames, cap);
};

// --- Backend video-chain planner (POST /video-chain/plan, /video-chain/validate) ---
//
// Types transcribed from `openapi.yaml`'s `VideoChain*` schemas, which are the
// contract. The chain-length helpers ABOVE stay: they are what the queue is
// still built from, and they are the parity reference the backend planner is
// ported against, so both exist until the migration is finished.
//
// Frame ranges are half-open `[owned_start_frame, owned_end_frame)` in integer
// GLOBAL frames; seconds anywhere in this feature are display-only.

export type VideoChainIssueSeverity = "error" | "warning";

export interface VideoChainIssue {
  code: string;
  severity: VideoChainIssueSeverity;
  message: string;
  segment_index?: number | null;
  event_id?: string | null;
  reference_id?: string | null;
}

export interface VideoChainSourceSpan {
  start_char: number;
  end_char: number;
}

export type VideoChainEventKind =
  | "shot"
  | "visual_action"
  | "camera"
  | "dialogue"
  | "physical_sound"
  | "music_transition"
  | "state_change";

export interface VideoChainEvent {
  id: string;
  kind: VideoChainEventKind;
  start_frame: number;
  end_frame: number;
  description: string;
  subject_ids?: string[];
  one_shot?: boolean;
  must_complete?: boolean;
  resulting_state?: string;
  source_span?: VideoChainSourceSpan | null;
  // The `[Shot N]` number this event was parsed from, in the ROOT prompt's
  // own numbering. Non-null only for `kind: "shot"`; provenance, not what a
  // compiled segment prompt renumbers its own shots to.
  shot_number?: number | null;
  // Quoted spans (dialogue, lyrics, on-screen text) inside `description`.
  // Omit it and the server re-derives it from `description`.
  verbatim?: string[];
}

export interface VideoChainPersistentContext {
  subjects?: string[];
  environment?: string[];
  visual_style?: string[];
  camera_rules?: string[];
  audio_bed?: string[];
  hard_constraints?: string[];
}

export interface VideoChainSegmentState {
  summary?: string;
  subjects?: string[];
  camera?: string;
  lighting?: string;
  ongoing_actions?: string[];
  ongoing_audio?: string[];
}

export type VideoChainReferenceKind = "image" | "video" | "audio";

// A reference and the segments it is bound to. Binding is many-to-many: one
// reference may cover several non-contiguous segments, and one segment may
// carry several references. `VideoChainSegment.reference_ids` is a read-only
// inverse of this, recomputed by the backend on validate.
export type VideoChainReferenceBindingSource = "default_all" | "explicit" | "token_implied";

export interface VideoChainReference {
  id: string;
  kind: VideoChainReferenceKind;
  label: string;
  segment_indices: number[];
  binding_source: VideoChainReferenceBindingSource;
  // The reference token this entry carries in the ROOT prompt, e.g.
  // `<Picture 2>` -- lets the compiler renumber tokens per segment.
  token?: string | null;
}

// Plan-request inventory entry: kind and label only, never file bytes or a path.
export interface VideoChainReferenceInput {
  id: string;
  kind: VideoChainReferenceKind;
  label: string;
  segment_indices?: number[] | null;
  // The token this reference already carries in `root_prompt`. Omit it and
  // the planner derives it from this inventory's order per kind.
  token?: string | null;
}

export type VideoChainContextMode = "timeline" | "manual" | "legacy_repeat";
// Manifest-side seed policy: how `segments[].seed` was actually assigned.
export type VideoChainSeedPolicy = "fixed" | "derived" | "explicit";
// Plan-request-side seed policy: `explicit` cannot be requested here (no field
// on the request carries per-segment seeds); the server answers it with a 400.
// Reach `explicit` by planning with `fixed`/`derived`, then setting
// `segments[].seed` in the editor and re-validating.
export type VideoChainPlanRequestSeedPolicy = "fixed" | "derived";
// How the planner chose the segment boundaries. `fixed` is the shipped
// behaviour: every segment is the cap, the last one is what is left, so a
// client can re-derive a continuation's length from the cap alone.
// `shot_aligned` picks boundaries around the shots, so the lengths differ per
// segment and the manifest is the ONLY source for them (see
// `advanceVideoChain` in videoChain.ts). On a manifest this is the mode that
// was APPLIED, which is `fixed` when there was nothing to align to.
export type VideoChainSegmentLengthMode = "fixed" | "shot_aligned";
// What the planner does with a shot whose frames cross a segment boundary.
// `refuse` is the shipped behaviour and the default: the plan comes back with
// an error naming the shot and the frame it is cut at.
// `assign_to_earlier_segment` gives the whole shot to the earlier segment and
// reports every crossing as a warning.
export type VideoChainBoundaryCrossingPolicy = "refuse" | "assign_to_earlier_segment";
export type VideoChainContinuationMode =
  | "boundary_frame"
  | "pinned_tail"
  | "motion_preroll"
  | "tail_reference_video"
  | "sampler_state";

export interface VideoChainVisualContext {
  mode: "initial" | VideoChainContinuationMode;
  frames?: number | null;
  source_segment_index?: number | null;
  // Frames this segment shares in time with its predecessor: 1 under
  // `boundary_frame`, null for `initial`. `segments[].effective_overlap_frames`
  // stays the authoritative value used in frame arithmetic.
  shared_context_frames?: number | null;
  // `motion_preroll` only: the frames of THIS segment's generated span the
  // anchors sit on (0 = the pre-roll's oldest frame, `shared_context_frames - 1`
  // = the boundary frame), and how many there are. Fixed by the plan so a retry
  // conditions on the same frames.
  anchor_local_frames?: number[] | null;
  anchor_count?: number | null;
}

export interface VideoChainSegment {
  index: number;
  // Global frame this segment's local index 0 reproduces. Null for segment 0.
  anchor_global_frame: number | null;
  owned_start_frame: number;
  owned_end_frame: number;
  // Includes the shared region, so this segment ADDS `generated_span_frames -
  // effective_overlap_frames` new frames (1 under `boundary_frame`).
  generated_span_frames: number;
  // The `total_frames` this segment's generation request asks for. Equals
  // `owned_end_frame` under `boundary_frame`; under a mode that pins a wider
  // overlap the span is rounded up onto the frame grid, so the request asks
  // for one length and `owned_end_frame` states the (longer) one it comes back
  // with. SEND THIS ONE: it is what the plan's arithmetic assumed. Omit it in
  // an edited manifest and the server derives it from `owned_end_frame`.
  requested_total_frames?: number | null;
  prompt: string;
  negative_prompt?: string;
  incoming_state?: VideoChainSegmentState;
  outgoing_state?: VideoChainSegmentState;
  owned_event_ids?: string[];
  reference_ids?: string[];
  seed: number;
  visual_context: VideoChainVisualContext;
  continuation_state_in?: string | null;
  continuation_state_out?: string | null;
  requested_overlap_frames?: number;
  effective_overlap_frames?: number;
  effective_overlap_samples?: number;
  requested_anchor_count?: number;
}

export interface VideoChainManifest {
  manifest_version: number;
  chain_id: string;
  plan_hash: string;
  architecture: string;
  variant?: string | null;
  root_prompt: string;
  root_prompt_hash: string;
  fps: number;
  target_frames: number;
  expected_final_frames: number;
  context_mode: VideoChainContextMode;
  segment_length_mode?: VideoChainSegmentLengthMode;
  continuation_mode: VideoChainContinuationMode;
  seed_policy: VideoChainSeedPolicy;
  root_seed?: number;
  chain_drift_tolerance_frames?: number;
  persistent_context?: VideoChainPersistentContext;
  references?: VideoChainReference[];
  events?: VideoChainEvent[];
  segments: VideoChainSegment[];
  warnings?: VideoChainIssue[];
}

export interface VideoChainPlanRequest {
  architecture: string;
  variant?: string | null;
  root_prompt: string;
  negative_prompt?: string;
  workflow?: string | null;
  target_frames: number;
  fps?: number;
  requested_segment_frames?: number | null;
  // Deliberately has no schema default and is nullable: omitted/null lets the
  // planner resolve it from the canonical timeline (shot-aligned when there is
  // a shot boundary to align to, fixed otherwise). Under `shot_aligned`,
  // `requested_segment_frames` is an upper bound rather than every segment's
  // length. What was applied is `manifest.segment_length_mode`.
  segment_length_mode?: VideoChainSegmentLengthMode | null;
  // Opt-in (default `refuse`). Only ever widens what plans: it decides whether
  // a boundary-crossing shot is an error or a warning, not where the boundaries
  // fall, and it is not part of `plan_hash`.
  boundary_crossing_policy?: VideoChainBoundaryCrossingPolicy;
  // Deliberately has no schema default and is nullable: the server must be
  // able to tell "the caller chose `timeline`" from "the caller chose
  // nothing" -- an omitted/null value is not the same request as `timeline`.
  context_mode?: VideoChainContextMode | null;
  seed_policy?: VideoChainPlanRequestSeedPolicy;
  root_seed?: number;
  continuation_mode?: VideoChainContinuationMode;
  requested_overlap_frames?: number;
  // `motion_preroll` only; a non-zero value with any other mode is a 400.
  requested_anchor_count?: number;
  chain_drift_tolerance_frames?: number;
  references?: VideoChainReferenceInput[];
  canonical_timeline?: {
    persistent_context?: VideoChainPersistentContext;
    events?: VideoChainEvent[];
  } | null;
}

export interface VideoChainSegmentPreview {
  index: number;
  prompt: string;
  negative_prompt?: string;
  global_frame_start: number;
  global_frame_end: number;
  generated_span_frames: number;
  new_output_frames?: number;
  // References bound to this segment. A cost as well as a quality knob:
  // reference rows ride through every denoise step.
  reference_count?: number;
  seed?: number;
}

export interface VideoChainFramePlan {
  target_frames: number;
  expected_final_frames: number;
  overshoot_frames: number;
  segment_frames: number[];
  // The cost side of a shot-aligned plan: how many requests, and how much each
  // one adds to the clip.
  segment_count?: number;
  segment_new_output_frames?: number[];
  segment_length_mode?: VideoChainSegmentLengthMode;
  frame_grid?: string | null;
}

export interface VideoChainPlanResponse {
  success: boolean;
  manifest: VideoChainManifest;
  segments: VideoChainSegmentPreview[];
  frame_plan: VideoChainFramePlan;
  errors: VideoChainIssue[];
  warnings: VideoChainIssue[];
  plan_schema_version: number;
  planner_cache_key?: string | null;
}

export interface VideoChainValidateRequest {
  manifest: VideoChainManifest;
  recompute_plan_hash?: boolean;
}

export interface VideoChainValidateResponse {
  valid: boolean;
  errors: VideoChainIssue[];
  warnings: VideoChainIssue[];
  plan_hash?: string | null;
  manifest?: VideoChainManifest | null;
}

// Build the Chain Manifest. Pure planning: loads no model, generates nothing.
// A 200 with `success: false` is normal — content problems come back as
// `errors` so the plan editor can show them next to the offending segment.
export const planVideoChainRequest = async (
  request: VideoChainPlanRequest
): Promise<VideoChainPlanResponse> =>
  (await api.post("/video-chain/plan", request)).data;

// Re-validate a manifest edited in the plan editor and recompute its
// `plan_hash`. The backend owns that hash; nothing here recomputes it.
export const validateVideoChainManifest = async (
  request: VideoChainValidateRequest
): Promise<VideoChainValidateResponse> =>
  (await api.post("/video-chain/validate", request)).data;

// The temporal-outpaint placements the loaded arch can anchor, from the
// backend's own table. An unknown arch (or a backend that does not serve the
// key) is treated as unconstrained, the same "assume supported" convention as
// archSupportsFeature -- the backend re-validates and answers 400 regardless.
export const videoOutpaintPlacements = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): string[] => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  const placements = c?.outpaint_placements;
  return placements && placements.length ? placements : ["free"];
};

// The audio temporal-outpaint placements the loaded arch can serve
// (POST /generate/outpaint/audio's `placement` field), from the backend's
// own table. UNLIKE `videoOutpaintPlacements`, an unknown/absent arch (e.g.
// ACE-Step, which has no entry at all) returns an EMPTY array rather than a
// fallback placeholder: ACE-Step's placement is a continuous offset, not a
// value from an enumerated set, so there is no "unconstrained enum value" to
// name -- a caller should read an empty array as "no placement selector;
// use the continuous offset/trim controls instead", not as an error.
export const audioOutpaintPlacements = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): string[] => (arch && caps?.audio_outpaint_placements?.[arch]) || [];

// The `music3_repaint_mode` values POST /generate/aud2aud's repaint mode can
// serve for the loaded arch, from the backend's own table. Same "empty, not a
// fallback placeholder" convention as `audioOutpaintPlacements`: ACE-Step has
// no entry at all (its own aud2aud has no such sub-mode), so a caller should
// read an empty array as "no repaint-mode selector for this arch", not as an
// error.
export const aud2audMusic3RepaintModes = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): string[] => (arch && caps?.aud2aud_music3_repaint_modes?.[arch]) || [];

export const fetchArchCapabilities = async (): Promise<ArchCapabilities> =>
  (await api.get("/schema/arch-capabilities")).data;

// True when `frames` is a length the architecture really accepts: on the grid
// (`multiple * n + offset`) and inside the production range. `suggested_frames`
// is only a SUBSET of these (it is capped, and LTX-2.3 omits lengths it accepts
// but does not advertise), so this is what decides whether a value the user
// already holds may stay.
export const isValidVideoFrameCount = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number | null | undefined
): boolean => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || frames == null || !Number.isFinite(frames)) return false;
  if (frames < c.min_frames) return false;
  if (c.max_frames != null && frames > c.max_frames) return false;
  const k = (frames - c.frame_offset) / c.frame_multiple;
  return Number.isInteger(k) && k >= 0;
};

// Same grid test as `isValidVideoFrameCount`, but WITHOUT the `max_frames`
// ceiling -- for callers that must not treat "above the single-inference cap"
// as "invalid", because the video-length chaining feature makes that a
// legitimate value to hold (see the opt-in chaining section below).
// `isValidVideoFrameCount` stays strict (single-inference requests, e.g.
// the temporal-inpaint trim target in InpaintPanel, really do need <= max_frames).
export const isOnGridVideoFrameCount = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number | null | undefined
): boolean => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || frames == null || !Number.isFinite(frames)) return false;
  if (frames < c.min_frames) return false;
  const k = (frames - c.frame_offset) / c.frame_multiple;
  return Number.isInteger(k) && k >= 0;
};

// The clip-length <Select> options for the loaded video arch, from the backend's
// own valid-length rule. Falls back to LTX-2.3's historical hardcoded list only
// when the matrix has not loaded (or the arch is unknown), so the offered
// lengths are never a second copy of a rule the backend owns.
//
// `current` — the value the control is bound to. A <select> renders ONLY the
// options it is handed, so a current value missing from the list makes the
// control render BLANK while the panel keeps sending that value. If it is a
// length this architecture accepts it is merged in (in order); if it is not,
// normalizeVideoFrames() below is what replaces it.
export const videoFrameOptions = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  current?: number | null
): { value: string; label: string }[] => {
  const suggested = arch ? caps?.video_constraints?.[arch]?.suggested_frames : undefined;
  const lengths = suggested?.length ? [...suggested] : [9, 17, 25, 33, 49, 65, 81, 97, 121];
  if (current != null && !lengths.includes(current)) {
    // Unknown arch / matrix not loaded: keep the value rather than blanking the
    // control, the same "assume supported" convention as archSupportsFeature.
    const known = !!(arch && caps?.video_constraints?.[arch]);
    if (!known || isValidVideoFrameCount(caps, arch, current)) {
      lengths.push(current);
      lengths.sort((a, b) => a - b);
    }
  }
  return lengths.map((n) => ({ value: String(n), label: String(n) }));
};

// The clip length to hold after the loaded architecture changed: the current
// value when that architecture accepts it, otherwise the NEAREST offered length
// (ties go up). Mirrors normalizeUnetQuantization: a value carried over from
// another architecture -- LTX-2.3's 121 onto MiniMax-H3, whose grid starts at
// 124 -- would otherwise sit in the control unselectable and be sent anyway,
// only to be snapped server-side with a warning.
// Uses `isOnGridVideoFrameCount`, NOT `isValidVideoFrameCount`: a value ABOVE
// `max_frames` that is still on the frame grid is the opt-in entry point for
// video-length chaining (see below) and must survive a mount / model-change
// pass, not get silently snapped back down to a suggested in-cap length --
// that used to happen on every remount and every model reload, discarding the
// user's chosen target with no notice (see VideoChainConfirmDialog / the
// panels' Generate-time chain prompt for where the choice is actually made).
export const normalizeVideoFrames = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number | null | undefined
): number | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || frames == null) return frames ?? null;
  if (isOnGridVideoFrameCount(caps, arch, frames)) return frames;
  const offered = c.suggested_frames?.length ? c.suggested_frames : null;
  if (!offered) return frames;
  return offered.reduce((best, n) =>
    Math.abs(n - frames) < Math.abs(best - frames) ? n : best, offered[0]);
};

// Label for that control, stating the arch's own rule ("17n+5, 124-362")
// rather than a hardcoded "8k+1". Always states the floor, even when there is
// no hard ceiling: `max_frames == null` used to drop the whole range clause,
// losing the floor hint too. When the arch declares no hard ceiling but does
// have a `trained_max_frames` (advisory-only, e.g. MiniMax-H3's 362), that is
// shown as a "+" open range rather than a bound, since going past it is
// documented-untested, not invalid.
export const videoFrameLabel = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): string => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c) return "Frames";
  const rule = c.frame_offset === 0
    ? `${c.frame_multiple}n`
    : `${c.frame_multiple}n+${c.frame_offset}`;
  let range: string;
  if (c.max_frames != null) {
    range = `, ${c.min_frames}-${c.max_frames}`;
  } else if (c.trained_max_frames != null) {
    range = `, ${c.min_frames}+ (trained to ${c.trained_max_frames})`;
  } else {
    range = `, ${c.min_frames}+`;
  }
  return `Frames (${rule}${range})`;
};

// The alignment both spatial axes must land on for `arch`. An arch the matrix
// does not describe (or a matrix that has not loaded) falls back to 32 — the
// same "assume supported" convention as archSupportsFeature, with the backend
// re-validating regardless. Single definition so the canvas fitter, the rule
// sentence and the slider bounds can never disagree about the grid.
const videoPixelAlign = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): number => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  return c?.pixel_align && c.pixel_align > 0 ? c.pixel_align : 32;
};

/**
 * The NEAREST CANVAS THIS ARCHITECTURE ACCEPTS to `srcWidth x srcHeight`
 * scaled by `scale`, plus why it differs from that when it does.
 *
 * Video generation does not take an arbitrary size: both axes round to
 * `pixel_align`, and an architecture may cap the canvas envelope
 * (`max_pixel_hw` = [short edge, long edge], orientation-agnostic). So
 * "generate at the input clip's own resolution" is often not literally
 * reachable — a 1920x1080 clip cannot be, on MiniMax-H3: 1080 is not a
 * multiple of 32 and 1920 is past the 1344 long-edge policy cap.
 *
 * The aspect ratio is preserved as closely as the grid allows: the cap is
 * applied as a single uniform factor to BOTH axes before rounding, so a capped
 * canvas is a scaled-down clip rather than a squashed one. Whatever aspect
 * mismatch the rounding leaves is resolved by the backend's
 * `center_crop_resize_frames`, which CENTRE-CROPS to the target aspect — it
 * does not letterbox — so callers should surface `cropped` to the user.
 *
 * An unknown architecture (or a capability matrix that has not loaded) gets
 * `pixel_align` 32 and no cap: the same "assume supported" convention as
 * archSupportsFeature, with the backend re-validating regardless.
 */
export const fitVideoCanvas = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  srcWidth: number,
  srcHeight: number,
  scale: number = 1
): { width: number; height: number; matchesSource: boolean; cropped: boolean } => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  const align = videoPixelAlign(caps, arch);
  const cap = c?.max_pixel_hw ?? null;

  let width = Math.max(1, srcWidth) * scale;
  let height = Math.max(1, srcHeight) * scale;

  // Uniform down-scale to fit the envelope (never an up-scale: the cap is a
  // ceiling, not a target).
  if (cap) {
    const [capShort, capLong] = cap;
    const shortEdge = Math.min(width, height);
    const longEdge = Math.max(width, height);
    const factor = Math.min(1, capShort / shortEdge, capLong / longEdge);
    width *= factor;
    height *= factor;
  }

  const round = (v: number) => Math.max(align, Math.round(v / align) * align);
  width = round(width);
  height = round(height);

  // Rounding can push an edge back over the cap (e.g. 756 -> 768 against a 768
  // short-edge cap is fine, but 1350 -> 1344 is not automatic). Step down to
  // the largest multiple of `align` that fits, per axis, using the ORIENTATION
  // OF THE SOURCE so the two caps are not swapped by a rounding tie. A square
  // source is bound by the short-edge cap on both axes, which is what
  // "short <= capShort AND long <= capLong" means for width == height.
  if (cap) {
    const [capShort, capLong] = cap;
    const floorTo = (v: number, limit: number) =>
      v <= limit ? v : Math.max(align, Math.floor(limit / align) * align);
    const widthIsLong = srcWidth > srcHeight;
    const heightIsLong = srcHeight > srcWidth;
    width = floorTo(width, widthIsLong ? capLong : capShort);
    height = floorTo(height, heightIsLong ? capLong : capShort);
  }

  const matchesSource = scale === 1 && width === srcWidth && height === srcHeight;
  // Aspect mismatch = the preprocessing discards content from the edges.
  const cropped =
    srcWidth > 0 && srcHeight > 0 &&
    Math.abs(srcWidth / srcHeight - width / height) > 1e-3;

  return { width, height, matchesSource, cropped };
};

/**
 * The canvas rule of the loaded video architecture, in words, for a UI that has
 * to explain why a requested size was not reachable. Reads the same
 * capability entry `fitVideoCanvas` does, so the two never disagree.
 */
export const videoCanvasRule = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): string => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  const align = videoPixelAlign(caps, arch);
  const cap = c?.max_pixel_hw ?? null;
  const alignRule = `both sides must be a multiple of ${align}`;
  if (!cap) return alignRule;
  return `${alignRule}, the short side is capped at ${cap[0]} and the long side at ${cap[1]}`;
};

// The ceiling of the Absolute width/height sliders where the loaded
// architecture declares NO envelope (LTX-2.3: `max_pixel_hw` null). It is a UI
// range, not an architecture fact -- the backend imposes no upper spatial bound
// there beyond `pixel_align` -- which is why it is a constant here instead of
// something read out of the capability matrix. It is also the historical range
// of those sliders, so an uncapped arch keeps exactly the reach it had.
const UNCAPPED_VIDEO_EDGE = 2048;

// Upper bound offered by the video routes' Block Swap number field
// (Txt2Img/Img2Img/Inpaint/Outpaint panels' video modes). The backend clamps
// `blocks_to_swap` to `num_blocks - 1` for whatever architecture is actually
// loaded (`core.memory_management.transformer_registry`), and there is no
// schema/capability endpoint that reports a loaded architecture's block
// count, so the frontend cannot derive this bound -- it is a defensible
// constant rather than a computed one. MiniMax-H3, the deepest video
// architecture wired today, has 50 transformer blocks (49 swappable, since
// at least one block must stay resident), so 49 is used here: LTX-2.3 has
// fewer blocks and the backend clamp still applies if a value above its own
// count is sent, so this constant only needs to not undershoot the largest
// loaded architecture.
export const VIDEO_BLOCK_SWAP_MAX = 49;

/**
 * The bounds one Absolute canvas slider may offer, given where the OTHER axis
 * currently sits.
 *
 * `max_pixel_hw` is `[short edge, long edge]` and the backend
 * (`validate_video_geometry`) compares it ORIENTATION-AGNOSTICALLY: a canvas is
 * legal when `min(w,h) <= short_cap` AND `max(w,h) <= long_cap`. So there is no
 * such thing as a fixed per-axis maximum. A single cap of `long_cap` on both
 * axes would offer the illegal 1344x1344; a single cap of `short_cap` on both
 * would forbid the perfectly legal 1344x768. The reachable maximum for THIS
 * axis is therefore a function of the other one:
 *
 *   other <= short_cap  -> this axis may be the long edge   -> long_cap
 *   other >  short_cap  -> the other axis is already the long edge, so this one
 *                          must be the short edge            -> short_cap
 *
 * which makes both 1344x768 and 768x1344 reachable and 1344x1345 not. When the
 * other axis is itself past `long_cap` (a value carried over from an uncapped
 * architecture) no value of this axis can make the pair legal; the tightest
 * bound is returned and `videoCanvasExceedsEnvelope` is what tells the user the
 * canvas is out of range.
 *
 * `min`/`step` are the arch's `pixel_align`, so the slider cannot land off-grid
 * either. An unknown arch (or a matrix that has not loaded) gets align 32 and
 * no cap: the same "assume supported" convention as archSupportsFeature.
 */
/**
 * The loaded video architecture's own floor on `num_inference_steps`, for the
 * step slider's `min`.
 *
 * This is a CORRECTNESS bound, not a UI one: `validate_video_steps`
 * (backend/api/generation_utils.py) answers 400 below it. MiniMax-H3 declares
 * 2 -- its step count is a sigma GRID POINT count, so N drives N-1 model
 * evaluations and 1 evaluates nothing -- while LTX-2.3 declares 1. Three of the
 * four video panels hardcoded `min={1}`, which let the user pick a value that
 * could only ever come back as a 400; this exists so the fallback rule lives in
 * one place instead of being re-derived per panel.
 *
 * An unknown arch (or a matrix that has not loaded) gets 1, matching the
 * "assume supported, let the backend re-validate" convention used elsewhere
 * here -- the request is still checked server-side either way.
 */
export const videoMinInferenceSteps = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): number => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  return c?.min_inference_steps ?? 1;
};

export const videoCanvasAxisBounds = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  otherEdge: number | null | undefined
): { min: number; max: number; step: number; capped: boolean } => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  const align = videoPixelAlign(caps, arch);
  const cap = c?.max_pixel_hw ?? null;
  // Floor the ceiling onto the alignment grid so the slider's top value is one
  // the backend actually accepts, the same way fitVideoCanvas steps down.
  const onGrid = (v: number) => Math.max(align, Math.floor(v / align) * align);
  if (!cap) {
    return { min: align, max: onGrid(UNCAPPED_VIDEO_EDGE), step: align, capped: false };
  }
  const capShort = Math.min(cap[0], cap[1]);
  const capLong = Math.max(cap[0], cap[1]);
  const other = otherEdge != null && Number.isFinite(otherEdge) ? otherEdge : 0;
  return {
    min: align,
    max: onGrid(other <= capShort ? capLong : capShort),
    step: align,
    capped: true,
  };
};

// True when `width x height` is outside the loaded arch's envelope — the exact
// comparison validate_video_geometry makes, so the panel's warning and the
// server's 400 agree. No envelope (LTX-2.3) or an unknown arch = never outside.
// The alignment rule is deliberately NOT folded in: the sliders' `step` already
// keeps both axes on the grid, whereas the envelope can be violated by a value
// that was legal on the architecture the user just switched away from.
export const videoCanvasExceedsEnvelope = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  width: number | null | undefined,
  height: number | null | undefined
): boolean => {
  const cap = (arch ? caps?.video_constraints?.[arch] : undefined)?.max_pixel_hw ?? null;
  if (!cap || width == null || height == null) return false;
  if (!Number.isFinite(width) || !Number.isFinite(height)) return false;
  return (
    Math.min(width, height) > Math.min(cap[0], cap[1]) ||
    Math.max(width, height) > Math.max(cap[0], cap[1])
  );
};

// Human-readable architecture names. Used where a model's architecture is shown
// to the user; MiniMax H3's entry also carries its required attribution.
const ARCH_DISPLAY_NAMES: Record<string, string> = {
  sd15: "Stable Diffusion 1.5",
  sdxl: "SDXL",
  zimage: "Z-Image",
  flux2: "FLUX.2",
  krea2: "Krea 2",
  lens: "Lens",
  anima: "Anima",
  minit2i: "MiniT2I",
  ideogram4: "Ideogram 4",
  ltx2: "LTX-2.3",
  acestep: "ACE-Step 1.5",
  minimax_h3: "MiniMax H3",
  minimax_music3: "MiniMax Music 3",
};

export function archDisplayName(arch: string | null | undefined): string;
export function archDisplayName(
  caps: ArchCapabilities | null | undefined,
  arch: string
): string;
export function archDisplayName(
  capsOrArch: ArchCapabilities | string | null | undefined,
  arch?: string
): string {
  if (typeof capsOrArch !== "string") {
    return (arch && capsOrArch?.arch_display_names?.[arch]) || arch || "";
  }
  return ARCH_DISPLAY_NAMES[capsOrArch] || capsOrArch;
}

// True when `arch` honors `feature`. An unknown arch, or capabilities that have
// not loaded yet, are treated as SUPPORTING the feature — the same convention as
// the backend's arch_supports_feature(), so a control is never hidden merely
// because the matrix was unavailable.
// `value` (optional): a value listed in `supported_values` counts as supported
// even when the feature as a whole is unsupported on that arch — the same rule
// as the backend's arch_supports_feature(arch, feature, value).
export const archSupportsFeature = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  feature: string,
  value?: string
): boolean => {
  if (!caps || !arch) return true;
  if (!(caps.unsupported?.[arch] && feature in caps.unsupported[arch])) return true;
  if (value === undefined) return false;
  return (caps.supported_values?.[arch]?.[feature] ?? []).includes(value);
};

// True when `arch`'s transformer can be converted to the weight-only INT8
// layout AT RUNTIME, in place, from an ordinary bf16 checkpoint.
//
// Read from the capability payload's `runtime_int8_archs`, which the backend
// serves straight from RUNTIME_INT8_ARCHS — the tuple the converter itself
// gates on. There is deliberately no fallback list here: a hardcoded copy is
// exactly what went stale as architectures were added. While the matrix has not
// loaded (or on an older backend that does not send the field) the value is not
// offered, the conservative direction for an opt-in control whose backend would
// otherwise refuse the request and warn.
export const archSupportsRuntimeInt8 = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): boolean => !!arch && (caps?.runtime_int8_archs ?? []).includes(arch);

// Label for the weight-quantization selector on the CURRENTLY loaded model.
//
// `unet_quantization` is one request parameter across every architecture, but
// only SD1.5/SDXL have a U-Net: every other architecture in this app is a DiT
// (Z-Image, FLUX.2, Anima, Lens, MiniT2I, Krea 2, Ideogram 4, LTX-2.3) or an
// audio DiT (ACE-Step), and calling the control "U-Net Quantization" there names
// a module the model does not contain. Only the two U-Net architectures are
// listed, because that set cannot grow; anything else, including an arch this
// build has never heard of, gets the neutral both-ways label rather than a
// guess.
const UNET_ARCHS = new Set(["sd15", "sdxl"]);

export const transformerQuantizationLabel = (
  arch: string | null | undefined
): string => {
  if (!arch) return "Transformer / U-Net Quantization";
  return UNET_ARCHS.has(arch) ? "U-Net Quantization" : "Transformer Quantization";
};

// Options for the "Transformer / U-Net Quantization" selector, filtered by what
// the loaded architecture actually applies. When the capability matrix has not
// loaded, every FP8 value is offered (the same "assume supported" convention as
// archSupportsFeature), so a control is never narrowed merely because the matrix
// was unavailable.
export const unetQuantizationOptions = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): { value: string; label: string }[] => {
  const allow = (v: string) =>
    archSupportsFeature(caps, arch, "unet_quantization", v);
  const options = [{ value: "none", label: "None" }];
  if (allow("fp8_e4m3fn")) options.push({ value: "fp8_e4m3fn", label: "FP8 E4M3" });
  if (allow("fp8_e5m2")) options.push({ value: "fp8_e5m2", label: "FP8 E5M2" });
  if (archSupportsRuntimeInt8(caps, arch) && allow("int8")) {
    options.push({
      value: "int8",
      label: "INT8 (in-place, applied once per model load)",
    });
  }
  return options;
};

// A persisted (localStorage) unet_quantization can name a value the CURRENTLY
// loaded architecture does not offer — e.g. `fp8_e4m3fn` carried over onto a
// krea2 model, where only `int8` is applied. Left alone, the <select> holds a
// value that is not among its options (it renders blank) while the panel keeps
// SENDING the value. Returns the value to keep, or null when it is not offered.
export const normalizeUnetQuantization = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  value: string | null | undefined
): string | null => {
  if (!value || value === "none") return null;
  const offered = unetQuantizationOptions(caps, arch).some((o) => o.value === value);
  return offered ? value : null;
};

// ---------------------------------------------------------------------------
// Loop-generation decode-mode response helpers
// ---------------------------------------------------------------------------
// The 3 generation endpoints (txt2img/img2img/inpaint) accept `loop_decode`
// ("full"|"cheap"|"none") + `skip_gallery` (img2img additionally accepts
// `input_latent_id`). Depending on those flags the response shape varies:
//   - normal decode:      { success, image: { filename, seed, ... }, actual_seed, warnings }
//   - loop_decode="none": { success, latent_id, actual_seed, warnings }            (NO image)
//   - skip_gallery=true:  { success, filename, image_path, actual_seed, warnings } (saved file, no DB record)
// These helpers read whichever shape is present so loop-generation chaining
// code doesn't need to special-case every call site.
export const isLatentOnlyResult = (result: any): boolean =>
  !!result?.latent_id && !result?.image;

export const getResultFilename = (result: any): string | undefined =>
  result?.image?.filename ?? result?.filename;

// Playback source for a video result: prefers preview_filename (browser-
// playable H.264 proxy) over filename, which is an FFV1-in-mkv master when
// video_lossless=true. Use for <video src>; getResultFilename stays correct
// for download/"send to" (the master).
export const getResultPlaybackFilename = (result: any): string | undefined =>
  result?.image?.preview_filename ?? getResultFilename(result);

export const getResultSeed = (result: any): number =>
  result?.image?.seed ?? result?.actual_seed ?? -1;

export const getResultAncestralSeed = (result: any): number | null =>
  result?.image?.ancestral_seed ?? result?.actual_ancestral_seed ?? null;

export interface StudioRenderUpload {
  assetId: string;
  file: File;
}

/** Queue a server-side render of a Studio timeline. */
export const renderStudioProject = async (
  manifest: Record<string, unknown>,
  uploads: StudioRenderUpload[] = [],
) => {
  const formData = new FormData();
  formData.append("manifest", JSON.stringify(manifest));
  for (const upload of uploads) {
    formData.append("asset_ids", upload.assetId);
    formData.append("asset_files", upload.file, upload.file.name || "studio-media");
  }
  const response = await api.post("/studio/render-jobs", formData, {
    // The request only stages the files and queues the job. The FFmpeg work is
    // polled separately, so this timeout is only a guard for a stalled upload.
    timeout: 600000,
  });
  return response.data;
};

export const getStudioRenderJob = async (jobId: string, signal?: AbortSignal) => {
  const response = await api.get(`/studio/render-jobs/${encodeURIComponent(jobId)}`, { signal });
  return response.data;
};

export const cancelStudioRenderJob = async (jobId: string) => {
  const response = await api.delete(`/studio/render-jobs/${encodeURIComponent(jobId)}`);
  return response.data;
};

export const generateTxt2Img = async (params: GenerationParams) => {
  // Only load ControlNet images if they exist (avoid unnecessary localStorage access)
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "txt2img_controlnet_collapsed")
    : params.controlnets;

  const paramsWithImages = {
    ...params,
    attention_type: resolveGlobalAttentionType(params.attention_type),
    attention_impl: resolveGlobalAttentionImpl(params.attention_impl),
    controlnets: controlnets,
  };

  const formData = new FormData();

  formData.append("prompt", paramsWithImages.prompt);
  formData.append("negative_prompt", paramsWithImages.negative_prompt || "");
  formData.append("steps", String(paramsWithImages.steps || 20));
  formData.append("cfg_scale", String(paramsWithImages.cfg_scale !== undefined ? paramsWithImages.cfg_scale : 7.0));
  // SenseNova U1.5 flow-matching time-shift; every other architecture ignores it.
  // Module-level helper, no React context here -- mirrors DEFAULT_PARAMS' fallback.
  formData.append("timestep_shift", String(paramsWithImages.timestep_shift ?? 3.0));
  // SenseNova U1.5 second CFG scale; inert without ref_images, ignored elsewhere.
  formData.append("img_cfg_scale", String(paramsWithImages.img_cfg_scale ?? 1.0));
  // SenseNova U1.5 CFG-overshoot clamp; every other architecture ignores it.
  formData.append("cfg_norm", paramsWithImages.cfg_norm ?? "global");
  // SenseNova U1.5 per-phase weight-half CPU eviction; every other architecture ignores it.
  formData.append("sensenova_mot_phase_eviction", String(paramsWithImages.sensenova_mot_phase_eviction ?? false));
  // SenseNova U1.5 per-layer prefix KV cache CPU streaming; every other architecture ignores it.
  formData.append("sensenova_kv_cache_streaming", String(paramsWithImages.sensenova_kv_cache_streaming ?? false));
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
  formData.append("attention_impl", paramsWithImages.attention_impl || "conduit");

  // Quantization
  if (paramsWithImages.unet_quantization && paramsWithImages.unet_quantization !== "none") {
    formData.append("unet_quantization", paramsWithImages.unet_quantization);
  }
  // Quantized GEMM path (already-quantized checkpoints: ideogram4/krea2/anima).
  // Sent ONLY when the user picked an explicit value; omitting it leaves the
  // backend's process-level setting (env var / Settings panel) untouched.
  if (paramsWithImages.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", paramsWithImages.quantized_gemm_mode);
  }
  // SDXL micro-conditioning original_size override
  if (paramsWithImages.original_size_w) formData.append("original_size_w", String(paramsWithImages.original_size_w));
  if (paramsWithImages.original_size_h) formData.append("original_size_h", String(paramsWithImages.original_size_h));
  if (paramsWithImages.original_size_scale !== undefined && paramsWithImages.original_size_scale !== null) {
    formData.append("original_size_scale", String(paramsWithImages.original_size_scale));
  }
  if (paramsWithImages.text_encoder_quantization && paramsWithImages.text_encoder_quantization !== "none") {
    formData.append("text_encoder_quantization", paramsWithImages.text_encoder_quantization);
  }
  formData.append("cpu_text_encoding", String(paramsWithImages.cpu_text_encoding ?? false));

  // torch.compile optimization
  formData.append("use_torch_compile", String(paramsWithImages.use_torch_compile ?? false));
  formData.append("vae_tiling", String(paramsWithImages.vae_tiling ?? false));
  formData.append("vae_tile_threshold", String(paramsWithImages.vae_tile_threshold ?? 0));
  formData.append("vae_tile_mode", String(paramsWithImages.vae_tile_mode ?? "blend"));
  formData.append("vae_tile_global_norm", String(paramsWithImages.vae_tile_global_norm ?? false));
  // Keep model components GPU-resident for the next queued generation (set by the queue dispatcher)
  formData.append("keep_models_hot", String(paramsWithImages.keep_models_hot ?? false));
  // Color Flatten: chroma-smoothing baked into the saved image at generation time
  formData.append("color_flatten_strength", String(paramsWithImages.color_flatten_strength ?? 0));
  // In-loop background hard-flatten (final-step flat-region solid-color replacement)
  formData.append("flatten_in_loop", String(paramsWithImages.flatten_in_loop ?? false));
  formData.append("flatten_in_loop_last_steps", String(paramsWithImages.flatten_in_loop_last_steps ?? 3));
  formData.append("flatten_in_loop_min_region", String(paramsWithImages.flatten_in_loop_min_region ?? 0.02));
  // Spectrum (Adaptive Spectral Feature Forecasting) acceleration (txt2img only in v1;
  // img2img/inpaint backends ignore these until wired)
  formData.append("spectrum_enable", String(paramsWithImages.spectrum_enable ?? false));
  formData.append("fbcache_enable", String(paramsWithImages.fbcache_enable ?? false));
  formData.append("fbcache_threshold", String(paramsWithImages.fbcache_threshold ?? 0.12));
  formData.append("fbcache_warmup_steps", String(paramsWithImages.fbcache_warmup_steps ?? 1));
  formData.append("spectrum_w", String(paramsWithImages.spectrum_w ?? 0.5));
  formData.append("spectrum_w_decay", String(paramsWithImages.spectrum_w_decay ?? 0.0));
  formData.append("spectrum_delta_cap", String(paramsWithImages.spectrum_delta_cap ?? 0.0));
  formData.append("spectrum_m", String(paramsWithImages.spectrum_m ?? 4));
  formData.append("spectrum_lam", String(paramsWithImages.spectrum_lam ?? 0.1));
  formData.append("spectrum_warmup_steps", String(paramsWithImages.spectrum_warmup_steps ?? 3));
  formData.append("spectrum_window_size", String(paramsWithImages.spectrum_window_size ?? 4));
  formData.append("spectrum_flex_window", String(paramsWithImages.spectrum_flex_window ?? 0.75));
  formData.append("spectrum_tail", String(paramsWithImages.spectrum_tail ?? 0.12));
  formData.append("spectrum_feature_mode", String(paramsWithImages.spectrum_feature_mode ?? "output"));
  formData.append("spectrum_cache_branch", String(paramsWithImages.spectrum_cache_branch ?? 1));
  formData.append("spectrum_max_cache", String(paramsWithImages.spectrum_max_cache ?? 0));

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
  formData.append("block_swap_h2d_only", String(paramsWithImages.block_swap_h2d_only ?? false));
  formData.append("block_swap_ring_size", String(paramsWithImages.block_swap_ring_size ?? 2));

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

  // VAE / Text encoder override paths (empty = model default)
  if (paramsWithImages.vae_path) {
    formData.append("vae_path", paramsWithImages.vae_path);
  }
  if (paramsWithImages.text_encoder_path) {
    formData.append("text_encoder_path", paramsWithImages.text_encoder_path);
  }
  // PiD decoder options (only meaningful when vae_path selects a PiD checkpoint;
  // ignored server-side for a normal VAE override / no override)
  formData.append("pid_sr_output", paramsWithImages.pid_sr_output || "4x");
  formData.append("pid_use_gemma", String(paramsWithImages.pid_use_gemma ?? false));
  formData.append("pid_low_vram", String(paramsWithImages.pid_low_vram ?? false));
  formData.append("pid_tile_native", String(paramsWithImages.pid_tile_native ?? 512));
  formData.append("pid_tile_overlap_ratio", String(paramsWithImages.pid_tile_overlap_ratio ?? 0.25));
  formData.append("pid_fast_large_decode", String(paramsWithImages.pid_fast_large_decode ?? false));

  // Loop-generation decode mode (heavy-decoder aware; see loopGenerationInheritance.ts)
  formData.append("loop_decode", paramsWithImages.loop_decode || "full");
  formData.append("skip_gallery", String(paramsWithImages.skip_gallery ?? false));

  const response = await postGenerationRequest("/generate/txt2img", formData, {
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
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "txt2img_controlnet_collapsed")
    : params.controlnets;

  const body = {
    ...params,
    attention_type: resolveGlobalAttentionType(params.attention_type),
    attention_impl: resolveGlobalAttentionImpl(params.attention_impl),
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
    // Idle is a 200 whose body carries is_running=false and run_id=null (see
    // routes.py get_active_training), so the idle case is decided on the BODY,
    // not on the status code.
    const data = res.data as Partial<ActiveTrainingInfo> | null;
    if (!data || !data.is_running || data.run_id == null) {
      return null;
    }
    return data as ActiveTrainingInfo;
  } catch (e: unknown) {
    // Backend unreachable, or an older backend still signalling idle with 404.
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

/** Base64-encode an image reference of unknown shape. A queued input image is a
 *  data: URL when the user uploaded it but an /outputs/ or blob: URL once a loop
 *  step patches in the previous result -- toBase64 hands those straight back
 *  unchanged, so they must be fetched to a Blob first. */
export const imageSourceToBase64 = async (src: string): Promise<string> => {
  // fetch("") resolves to the current document, which would be encoded and sent
  // as if it were an image.
  if (!src) throw new Error("imageSourceToBase64: empty image source");
  return src.startsWith("data:") ? toBase64(src) : toBase64(await (await fetch(src)).blob());
};

export const generateImg2ImgTrainingPreview = async (
  params: Img2ImgTrainingPreviewParams,
): Promise<{ blob: Blob; seed?: string; runId?: string; requestId?: string; filename?: string }> => {
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "img2img_controlnet_collapsed")
    : params.controlnets;
  const body = {
    ...params,
    attention_type: resolveGlobalAttentionType(params.attention_type),
    attention_impl: resolveGlobalAttentionImpl(params.attention_impl),
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
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "inpaint_controlnet_collapsed")
    : params.controlnets;
  const body = {
    ...params,
    attention_type: resolveGlobalAttentionType(params.attention_type),
    attention_impl: resolveGlobalAttentionImpl(params.attention_impl),
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


// `image` is optional when `latentId` (or `params.input_latent_id`) is set —
// the backend requires EXACTLY ONE of `image` / `input_latent_id`. Used for
// loop-generation latent passthrough (decodeMode "final-only", see
// loopGenerationInheritance.ts computeLoopDecodeDirective).
export const generateImg2Img = async (
  params: Img2ImgParams,
  image?: File | string | null,
  latentId?: string | null,
) => {
  // Only load ControlNet images if they exist (avoid unnecessary localStorage access)
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "img2img_controlnet_collapsed")
    : params.controlnets;

  const paramsWithImages = {
    ...params,
    attention_type: resolveGlobalAttentionType(params.attention_type),
    attention_impl: resolveGlobalAttentionImpl(params.attention_impl),
    controlnets: controlnets,
  };

  const formData = new FormData();

  const effectiveLatentId = latentId ?? paramsWithImages.input_latent_id ?? undefined;

  if (effectiveLatentId) {
    // Latent passthrough: skip the image upload entirely, start denoising
    // from the cached latent instead.
    formData.append("input_latent_id", effectiveLatentId);
  } else {
    if (!image) {
      throw new Error("generateImg2Img requires either an image or a latentId");
    }
    // Handle both File objects and data URLs
    if (typeof image === 'string') {
      // Convert data URL or URL to blob
      const response = await fetch(image);
      const blob = await response.blob();
      formData.append("image", blob, "input.png");
    } else {
      formData.append("image", image);
    }
  }

  formData.append("prompt", paramsWithImages.prompt);
  formData.append("negative_prompt", paramsWithImages.negative_prompt || "");
  formData.append("steps", String(paramsWithImages.steps || 20));
  formData.append("cfg_scale", String(paramsWithImages.cfg_scale !== undefined ? paramsWithImages.cfg_scale : 7.0));
  // SenseNova U1.5 flow-matching time-shift; every other architecture ignores it.
  // Module-level helper, no React context here -- mirrors DEFAULT_PARAMS' fallback.
  formData.append("timestep_shift", String(paramsWithImages.timestep_shift ?? 3.0));
  // SenseNova U1.5 second CFG scale; inert without ref_images, ignored elsewhere.
  formData.append("img_cfg_scale", String(paramsWithImages.img_cfg_scale ?? 1.0));
  // SenseNova U1.5 CFG-overshoot clamp; every other architecture ignores it.
  formData.append("cfg_norm", paramsWithImages.cfg_norm ?? "global");
  // SenseNova U1.5 per-phase weight-half CPU eviction; every other architecture ignores it.
  formData.append("sensenova_mot_phase_eviction", String(paramsWithImages.sensenova_mot_phase_eviction ?? false));
  // SenseNova U1.5 per-layer prefix KV cache CPU streaming; every other architecture ignores it.
  formData.append("sensenova_kv_cache_streaming", String(paramsWithImages.sensenova_kv_cache_streaming ?? false));
  formData.append("denoising_strength", String(paramsWithImages.denoising_strength || 0.75));
  formData.append("img2img_fix_steps", String(paramsWithImages.img2img_fix_steps ?? true));
  formData.append("vae_drift_correction", String(paramsWithImages.vae_drift_correction ?? false));
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
  formData.append("attention_impl", paramsWithImages.attention_impl || "conduit");

  // Block swap (CPU offloading)
  formData.append("enable_block_swap", String(paramsWithImages.enable_block_swap ?? false));
  formData.append("blocks_to_swap", String(paramsWithImages.blocks_to_swap ?? 20));
  formData.append("use_pinned_memory", String(paramsWithImages.use_pinned_memory ?? false));
  formData.append("block_swap_h2d_only", String(paramsWithImages.block_swap_h2d_only ?? false));
  formData.append("block_swap_ring_size", String(paramsWithImages.block_swap_ring_size ?? 2));

  // Debug log for quantization
  console.log('[API] img2img unet_quantization:', paramsWithImages.unet_quantization);
  if (paramsWithImages.unet_quantization && paramsWithImages.unet_quantization !== "none") {
    formData.append("unet_quantization", paramsWithImages.unet_quantization);
    console.log('[API] Added unet_quantization to FormData:', paramsWithImages.unet_quantization);
  } else {
    console.log('[API] No quantization or "none" selected');
  }
  // Quantized GEMM path (already-quantized checkpoints: ideogram4/krea2/anima).
  // Sent ONLY when the user picked an explicit value; omitting it leaves the
  // backend's process-level setting (env var / Settings panel) untouched.
  if (paramsWithImages.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", paramsWithImages.quantized_gemm_mode);
  }

  if (paramsWithImages.text_encoder_quantization && paramsWithImages.text_encoder_quantization !== "none") {
    formData.append("text_encoder_quantization", paramsWithImages.text_encoder_quantization);
  }

  // CPU text encoding
  formData.append("cpu_text_encoding", String(paramsWithImages.cpu_text_encoding ?? false));

  // torch.compile optimization
  formData.append("use_torch_compile", String(paramsWithImages.use_torch_compile ?? false));
  formData.append("vae_tiling", String(paramsWithImages.vae_tiling ?? false));
  formData.append("vae_tile_threshold", String(paramsWithImages.vae_tile_threshold ?? 0));
  formData.append("vae_tile_mode", String(paramsWithImages.vae_tile_mode ?? "blend"));
  formData.append("vae_tile_global_norm", String(paramsWithImages.vae_tile_global_norm ?? false));
  // Keep model components GPU-resident for the next queued generation (set by the queue dispatcher)
  formData.append("keep_models_hot", String(paramsWithImages.keep_models_hot ?? false));
  // Color Flatten: chroma-smoothing baked into the saved image at generation time
  formData.append("color_flatten_strength", String(paramsWithImages.color_flatten_strength ?? 0));
  // In-loop background hard-flatten (final-step flat-region solid-color replacement)
  formData.append("flatten_in_loop", String(paramsWithImages.flatten_in_loop ?? false));
  formData.append("flatten_in_loop_last_steps", String(paramsWithImages.flatten_in_loop_last_steps ?? 3));
  formData.append("flatten_in_loop_min_region", String(paramsWithImages.flatten_in_loop_min_region ?? 0.02));
  // Spectrum (Adaptive Spectral Feature Forecasting) acceleration (txt2img only in v1;
  // img2img/inpaint backends ignore these until wired)
  formData.append("spectrum_enable", String(paramsWithImages.spectrum_enable ?? false));
  formData.append("fbcache_enable", String(paramsWithImages.fbcache_enable ?? false));
  formData.append("fbcache_threshold", String(paramsWithImages.fbcache_threshold ?? 0.12));
  formData.append("fbcache_warmup_steps", String(paramsWithImages.fbcache_warmup_steps ?? 1));
  formData.append("spectrum_w", String(paramsWithImages.spectrum_w ?? 0.5));
  formData.append("spectrum_w_decay", String(paramsWithImages.spectrum_w_decay ?? 0.0));
  formData.append("spectrum_delta_cap", String(paramsWithImages.spectrum_delta_cap ?? 0.0));
  formData.append("spectrum_m", String(paramsWithImages.spectrum_m ?? 4));
  formData.append("spectrum_lam", String(paramsWithImages.spectrum_lam ?? 0.1));
  formData.append("spectrum_warmup_steps", String(paramsWithImages.spectrum_warmup_steps ?? 3));
  formData.append("spectrum_window_size", String(paramsWithImages.spectrum_window_size ?? 4));
  formData.append("spectrum_flex_window", String(paramsWithImages.spectrum_flex_window ?? 0.75));
  formData.append("spectrum_tail", String(paramsWithImages.spectrum_tail ?? 0.12));
  formData.append("spectrum_feature_mode", String(paramsWithImages.spectrum_feature_mode ?? "output"));
  formData.append("spectrum_cache_branch", String(paramsWithImages.spectrum_cache_branch ?? 1));
  formData.append("spectrum_max_cache", String(paramsWithImages.spectrum_max_cache ?? 0));

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

  // VAE / Text encoder override paths (empty = model default)
  if (paramsWithImages.vae_path) {
    formData.append("vae_path", paramsWithImages.vae_path);
  }
  if (paramsWithImages.text_encoder_path) {
    formData.append("text_encoder_path", paramsWithImages.text_encoder_path);
  }
  // PiD decoder options (only meaningful when vae_path selects a PiD checkpoint;
  // ignored server-side for a normal VAE override / no override)
  formData.append("pid_sr_output", paramsWithImages.pid_sr_output || "4x");
  formData.append("pid_use_gemma", String(paramsWithImages.pid_use_gemma ?? false));
  formData.append("pid_low_vram", String(paramsWithImages.pid_low_vram ?? false));
  formData.append("pid_tile_native", String(paramsWithImages.pid_tile_native ?? 512));
  formData.append("pid_tile_overlap_ratio", String(paramsWithImages.pid_tile_overlap_ratio ?? 0.25));
  formData.append("pid_fast_large_decode", String(paramsWithImages.pid_fast_large_decode ?? false));

  // Loop-generation decode mode (heavy-decoder aware; see loopGenerationInheritance.ts)
  formData.append("loop_decode", paramsWithImages.loop_decode || "full");
  formData.append("skip_gallery", String(paramsWithImages.skip_gallery ?? false));

  const response = await postGenerationRequest("/generate/img2img", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const generateUpscale = async (params: UpscaleParams, image: File | string) => {
  const formData = new FormData();

  // Handle both File objects and data URLs
  if (typeof image === 'string') {
    const response = await fetch(image);
    const blob = await response.blob();
    formData.append("image", blob, "input.png");
  } else {
    formData.append("image", image);
  }

  formData.append("upscaler_backend", params.upscaler_backend || "spandrel");
  if (params.upscaler_model) {
    formData.append("upscaler_model", params.upscaler_model);
  }
  formData.append("scale_factor", String(params.scale_factor ?? 2.0));
  formData.append("pil_resample", params.pil_resample || "lanczos");
  formData.append("tile_size", String(params.tile_size ?? 512));
  formData.append("tile_overlap", String(params.tile_overlap ?? 32));
  formData.append("rtx_vsr_quality", params.rtx_vsr_quality || "high");
  formData.append("unsharp_enable", String(params.unsharp_enable ?? false));
  formData.append("unsharp_radius", String(params.unsharp_radius ?? 2.0));
  formData.append("unsharp_percent", String(params.unsharp_percent ?? 100));
  formData.append("unsharp_threshold", String(params.unsharp_threshold ?? 3));

  // Diffusion tile upscale
  if (params.upscaler_backend === "diffusion") {
    formData.append("prompt", params.prompt || "");
    formData.append("negative_prompt", params.negative_prompt || "");
    formData.append("diffusion_denoising_strength", String(params.diffusion_denoising_strength ?? 0.3));
    formData.append("steps", String(params.steps ?? 20));
    formData.append("cfg_scale", String(params.cfg_scale ?? 7.0));
    formData.append("sampler", params.sampler || "euler");
    formData.append("schedule_type", params.schedule_type || "uniform");
    formData.append("attention_type", resolveGlobalAttentionType(params.attention_type));
    formData.append("attention_impl", resolveGlobalAttentionImpl(params.attention_impl));
    formData.append("seed", String(params.seed ?? -1));
    formData.append("diffusion_pre_upscale_mode", params.diffusion_pre_upscale_mode || "pil");
  }

  const response = await postGenerationRequest("/generate/upscale", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const fetchUpscalerModels = async (): Promise<{ models: UpscalerModelInfo[] }> => {
  const response = await api.get("/models/upscalers");
  return response.data;
};

// ---------------------------------------------------------------------------
// Standalone VAE / Text-Encoder override candidates (RP3)
// ---------------------------------------------------------------------------

export interface VaeEntry {
  name: string;
  path: string;
  arch?: string | null;
  latent_channels?: number | null;
  vae_class?: string | null;
  scale_spatial?: number | null;
  scale_temporal?: number | null;
  // "autoencoder" (normal VAE) | "pid_decoder" (PiD Pixel Diffusion Decoder checkpoint)
  kind?: string | null;
  // Present only for a VAE exported by a SushiUI VAE fine-tune (read from its
  // sushi_vae_training.json sidecar); absent for every other candidate.
  training?: VaeTrainingProvenance | null;
}

export interface VaeTrainingProvenance {
  produced_by?: string | null;
  run_id?: number | null;
  run_name?: string | null;
  step?: number | null;
  // True when the ENCODER was fine-tuned too: this VAE encodes to a different
  // latent distribution than the base model's VAE, so it is not a drop-in
  // replacement (latent caches / LoRAs built against the base VAE do not match).
  // TRI-STATE: null/undefined means a partial sidecar did not record it —
  // render that as unknown, never as "encoder frozen".
  encoder_trained?: boolean | null;
  // True for the EMA export, false for its "_noema" (live weights) sibling.
  // Tri-state for the same reason as encoder_trained.
  ema_applied?: boolean | null;
  base_vae_path?: string | null;
}

export interface TextEncoderEntry {
  name: string;
  path: string;
  arch?: string | null;
  out_dim?: number | null;
  te_type?: string | null;
}

export const fetchVaes = async (): Promise<{ vaes: VaeEntry[] }> => {
  const response = await api.get("/models/vaes");
  return response.data;
};

export const fetchTextEncoders = async (): Promise<{ text_encoders: TextEncoderEntry[] }> => {
  const response = await api.get("/models/text_encoders");
  return response.data;
};

// A recorded measurement of one (text encoder, projection) pairing against a
// released encoder. Absent from the listing (null) means unmeasured, which is
// not the same as measured-and-equal.
export interface MiniMaxH3TeAgreement {
  reference: string;
  cosine: number;
  rel_rms: number;
  rel_rms_floor: number;
  presentations: number;
  /**
   * The projection this was measured WITH. The record is keyed by the
   * (encoder, projection) pair, so a number measured for one projection says
   * nothing about the encoder driven through another -- callers must check
   * this against the projection actually selected before presenting it.
   */
  projection: string;
}

// One `clip_projections/` file that declares a given encoder's width. `usable`
// is the backend's own pairing gates run with this file named, so an entry with
// a matching d_in but a wrong d_out or tap is listed with `reason` rather than
// dropped.
export interface MiniMaxH3ProjectionCandidate {
  path: string;
  name: string;
  d_in: number;
  d_out: number;
  tap: number;
  usable: boolean;
  reason: string | null;
}

export interface MiniMaxH3TextEncoderEntry {
  path: string;
  name: string;
  size_bytes: number;
  compatible: boolean;
  variant: string | null;
  reason: string;
  // Null for the released 32B encoders, which carry no converted-encoder metadata.
  requires_projection: boolean | null;
  hidden_size: number | null;
  num_hidden_layers: number | null;
  // What auto-discovery would adopt (null when none or several match), the
  // refusal when it is null, and the set to choose from.
  projection?: string | null;
  projection_reason?: string | null;
  projection_candidates?: MiniMaxH3ProjectionCandidate[];
  agreement: MiniMaxH3TeAgreement | null;
}

export interface MiniMaxH3ClipProjectionEntry {
  path: string;
  name: string;
  size_bytes: number;
  d_in: number;
  d_out: number;
  tap: number;
}

export interface MiniMaxH3TextEncodersResponse {
  // What the loader would pick if no text_encoder_file is sent, and why.
  selected: string | null;
  selected_reason: string;
  text_encoders: MiniMaxH3TextEncoderEntry[];
  clip_projections: MiniMaxH3ClipProjectionEntry[];
}

// modelPath is either the DiT file or the model tree root.
export const fetchMiniMaxH3TextEncoders = async (
  modelPath: string
): Promise<MiniMaxH3TextEncodersResponse> => {
  const response = await api.get("/models/minimax-h3/text-encoders", {
    params: { model_path: modelPath },
  });
  return response.data;
};

/** What `loadModel`'s 7th argument sends. Omit it entirely for a plain load. */
export interface MiniMaxH3HybridLoadRequest {
  /** Absolute path of the overlay DiT. Without it nothing hybrid is sent. */
  overlay_file: string;
  preset?: "block_range_adaln";
  block_range_start?: number;
  /** Inclusive. */
  block_range_end?: number;
  final_adaln_from_overlay?: boolean;
}

export interface MiniMaxH3HybridOverlayCandidate {
  path: string;
  name: string;
  variant: string | null;
  size_bytes: number;
  // The loader's own preflight, run against this base over every block, so a
  // compatible entry stays compatible for any range the user then picks. It is
  // taken with final_adaln_from_overlay OFF, and that toggle selects a key the
  // check never looked at, so turning it on can still be refused at load.
  compatible: boolean;
  reason: string | null;
  /** Stable code of that refusal, the same set POST /models/load 400s with. */
  refusal_code: string | null;
  quantization_format: string | null;
  num_blocks: number | null;
}

export interface MiniMaxH3HybridOverlaysResponse {
  base: {
    path: string;
    name: string;
    variant: string | null;
    /** block_range_end is inclusive, so the last valid value is this minus one. */
    num_blocks: number;
  };
  checked_block_range: [number, number];
  overlays: MiniMaxH3HybridOverlayCandidate[];
  // The backend's own defaults for the hybrid load fields. Served here rather
  // than from /schema/* because a block range only means something beside
  // base.num_blocks.
  defaults: {
    preset: string;
    presets: string[];
    block_range_start: number;
    block_range_end: number;
    final_adaln_from_overlay: boolean;
  };
}

// modelPath is the BASE: either the DiT file or the model tree root.
export const fetchMiniMaxH3HybridOverlays = async (
  modelPath: string
): Promise<MiniMaxH3HybridOverlaysResponse> => {
  const response = await api.get("/models/minimax-h3/hybrid-overlays", {
    params: { model_path: modelPath },
  });
  return response.data;
};

/**
 * Reference-bank job document. Before the first build of a backend process it
 * is just `{ state: "idle" }`.
 */
export interface MiniMaxH3ReferenceBankJob {
  job_id?: string;
  state: "idle" | "running" | "completed" | "cancelled" | "failed";
  reference?: string | null;
  /** Presentations encoded so far, and the suite corpus size (0 until the first). */
  processed?: number;
  total?: number;
  message?: string;
  error?: string | null;
  result?: Record<string, any> | null;
  started_at?: number;
  finished_at?: number | null;
}

export interface MiniMaxH3ReferenceBankSummary {
  reference: string;
  suite_version: string;
  presentations: number;
  token_total?: number;
  hidden_size?: number;
  built_at?: string | null;
  /** A bank built from another released encoder does not answer for this one. */
  is_loaded_encoder: boolean;
}

export interface MiniMaxH3StoredMeasurement {
  encoder: string | null;
  projection: string | null;
  reference: string | null;
  cosine: number | null;
  cosine_baseline: number | null;
  rel_rms: number | null;
  rel_rms_baseline: number | null;
  presentations: number | null;
  /** `token_refiner` is the view the packed sequence actually contains. */
  stage: "token_refiner" | "raw";
  suite_version?: string;
  measured_at?: string | null;
}

export interface MiniMaxH3TeAgreementStatus {
  supported: boolean;
  can_build: boolean;
  reason: string | null;
  model_path: string | null;
  loaded: {
    text_encoder: string | null;
    text_encoder_path: string | null;
    projection: string | null;
    is_substitute: boolean;
    /** The backend's one wording for a substituted pairing; null for a released encoder. */
    substitution: string | null;
  } | null;
  suite: { version: string | null; prompts: number; digest: string | null };
  /** Measured cost of one build, for the suite version named here. */
  cost: {
    suite_version: string;
    seconds: number;
    host_ram_gib_min: number;
    host_ram_gib_max: number;
    stored_mb: number;
  };
  bank: MiniMaxH3ReferenceBankSummary | null;
  banks: MiniMaxH3ReferenceBankSummary[];
  measurements: MiniMaxH3StoredMeasurement[];
  measurements_reason: string | null;
  job: MiniMaxH3ReferenceBankJob;
}

export const getMiniMaxH3TeAgreement = async (
  modelPath?: string
): Promise<MiniMaxH3TeAgreementStatus> => {
  const response = await api.get("/models/minimax-h3/te-agreement", {
    params: modelPath ? { model_path: modelPath } : undefined,
  });
  return response.data;
};

/**
 * Start the bank build. Throws on 409 while a generation, a training run, a
 * model-state mutation or another build is in flight, and on 400 when the
 * loaded encoder is a substitute or is not the one named here.
 */
export const startMiniMaxH3ReferenceBank = async (
  textEncoderPath: string
): Promise<MiniMaxH3ReferenceBankJob> => {
  const response = await api.post("/models/minimax-h3/te-agreement/reference-bank", {
    text_encoder_path: textEncoderPath,
  });
  return response.data;
};

export const cancelMiniMaxH3ReferenceBank = async (): Promise<MiniMaxH3ReferenceBankJob> => {
  const response = await api.delete("/models/minimax-h3/te-agreement/reference-bank");
  return response.data;
};

// ---------------------------------------------------------------------------
// Video generation (LTX-2.3)
// ---------------------------------------------------------------------------

// txt2vid: JSON POST /generate/txt2vid. Response is the standard
// GenerationResponse ({ success, image, actual_seed, warnings }); image.filename
// is an .mp4 file under /outputs/.
export const generateTxt2Vid = async (params: Txt2VidParams) => {
  const body = {
    prompt: params.prompt,
    negative_prompt: params.negative_prompt || "",
    width: params.width ?? 768,
    height: params.height ?? 512,
    num_frames: params.num_frames ?? 121,
    frame_rate: params.frame_rate ?? 24.0,
    num_inference_steps: params.num_inference_steps ?? 8,
    guidance_scale: params.guidance_scale ?? 1.0,
    seed: params.seed ?? -1,
    num_videos_per_prompt: params.num_videos_per_prompt ?? 1,
    max_sequence_length: params.max_sequence_length ?? 1024,
    audio_enable: params.audio_enable ?? true,
    vae_path: params.vae_path ?? null,
    text_encoder_path: params.text_encoder_path ?? null,
    // `=== "none" -> null` mirrors img2vid, video outpaint and all four image
    // senders: "none" is the UI's spelling of "no quantization", and a "none"
    // persisted in localStorage by an older build would otherwise be sent as a
    // value and come back as an `unsupported_param` warning on every txt2vid.
    unet_quantization:
      params.unet_quantization && params.unet_quantization !== "none"
        ? params.unet_quantization
        : null,
    quantized_gemm_mode: params.quantized_gemm_mode ?? null,
    attention_type: resolveGlobalAttentionType(params.attention_type),
    loras: params.loras || [],
    // Acceleration (block swap / FBCache / Spectrum) -- same fields/defaults
    // as /generate/outpaint/video and /generate/inpaint/video.
    blocks_to_swap: params.blocks_to_swap ?? 0,
    fuse_output_proj: params.fuse_output_proj ?? false,
    fbcache_enable: params.fbcache_enable ?? false,
    fbcache_threshold: params.fbcache_threshold ?? 0.12,
    fbcache_warmup_steps: params.fbcache_warmup_steps ?? 1,
    spectrum_enable: params.spectrum_enable ?? false,
    spectrum_w: params.spectrum_w ?? 0.5,
    spectrum_w_decay: params.spectrum_w_decay ?? 0.0,
    spectrum_delta_cap: params.spectrum_delta_cap ?? 0.0,
    spectrum_m: params.spectrum_m ?? 4,
    spectrum_lam: params.spectrum_lam ?? 0.1,
    spectrum_warmup_steps: params.spectrum_warmup_steps ?? 3,
    spectrum_window_size: params.spectrum_window_size ?? 4,
    spectrum_flex_window: params.spectrum_flex_window ?? 0.75,
    spectrum_tail: params.spectrum_tail ?? 0.12,
    spectrum_max_cache: params.spectrum_max_cache ?? 0,
    // Video chain provenance -- present only when this request is a chain's
    // first segment (design §13).
    ...chainProvenanceBody(params),
  };

  const response = await postGenerationRequest("/generate/txt2vid", body);
  return response.data;
};

// img2vid: multipart POST /generate/img2vid with an uploaded keyframe `image`.
// Every IMG2VID field is appended explicitly (CLAUDE.md param-threading).
// `image` is nullable because the endpoint's `image` part is optional when an
// `input_audio` track is sent: a pinned track conditions the clip on its own
// (MiniMax-H3). Sending neither is a 400 that points at /generate/txt2vid.
export const generateImg2Vid = async (
  params: Img2VidParams,
  image: File | string | null,
) => {
  const formData = new FormData();

  // Handle both File objects and data URLs
  if (typeof image === "string") {
    const response = await fetch(image);
    const blob = await response.blob();
    formData.append("image", blob, "keyframe.png");
  } else if (image) {
    formData.append("image", image);
  }

  formData.append("prompt", params.prompt);
  formData.append("negative_prompt", params.negative_prompt || "");
  formData.append("width", String(params.width ?? 768));
  formData.append("height", String(params.height ?? 512));
  formData.append("num_frames", String(params.num_frames ?? 121));
  formData.append("frame_rate", String(params.frame_rate ?? 24.0));
  formData.append("num_inference_steps", String(params.num_inference_steps ?? 8));
  formData.append("guidance_scale", String(params.guidance_scale ?? 1.0));
  formData.append("seed", String(params.seed ?? -1));
  formData.append("num_videos_per_prompt", String(params.num_videos_per_prompt ?? 1));
  formData.append("max_sequence_length", String(params.max_sequence_length ?? 1024));
  formData.append("audio_enable", String(params.audio_enable ?? true));
  if (params.vae_path) {
    formData.append("vae_path", params.vae_path);
  }
  if (params.text_encoder_path) {
    formData.append("text_encoder_path", params.text_encoder_path);
  }
  // `&& !== "none"` mirrors the image senders: "none" is the UI's spelling of
  // "no quantization" and must not be sent as a value.
  if (params.unet_quantization && params.unet_quantization !== "none") {
    formData.append("unet_quantization", params.unet_quantization);
  }
  if (params.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", params.quantized_gemm_mode);
  }
  formData.append("attention_type", resolveGlobalAttentionType(params.attention_type));
  // Optional LAST-frame keyframe (MiniMax-H3 fl2va). Same File-or-data-URL
  // handling as `image` above; omitted entirely when null/undefined, which is
  // what makes the backend's `File(None)` sentinel mean "no end anchor".
  if (params.last_frame_image) {
    if (typeof params.last_frame_image === "string") {
      const lastResponse = await fetch(params.last_frame_image);
      const lastBlob = await lastResponse.blob();
      formData.append("last_frame_image", lastBlob, "last_frame.png");
    } else {
      formData.append("last_frame_image", params.last_frame_image);
    }
  }
  // Where the uploaded `image` sits on the clip. Always sent (0 is a real
  // value, not "unset"), so a panel that offers the control never depends on
  // the server default agreeing with its own.
  formData.append("input_image_frame_index", String(params.input_image_frame_index ?? 0));
  // Additional anchors, as two POSITIONAL lists: entry n of
  // keyframe_frame_indices is the placement of entry n of keyframe_images. Both
  // are appended in the same loop so the pairing cannot drift; a mismatch is a
  // 400 server-side.
  for (const keyframe of params.keyframes ?? []) {
    if (!keyframe || !keyframe.image) continue;
    if (typeof keyframe.image === "string") {
      const keyframeResponse = await fetch(keyframe.image);
      const keyframeBlob = await keyframeResponse.blob();
      formData.append("keyframe_images", keyframeBlob, "keyframe.png");
    } else {
      formData.append("keyframe_images", keyframe.image);
    }
    formData.append("keyframe_frame_indices", String(keyframe.frame_index));
  }
  // The ia2v track. Appended only when there is one, so the backend's
  // `File(None)` sentinel keeps meaning "generate the soundtrack jointly".
  // There is no offset or length to send with it: the pin covers the whole
  // clip, and a track shorter than the clip is refused server-side rather than
  // padded.
  if (params.input_audio) {
    formData.append("input_audio", params.input_audio);
  }
  formData.append("loras", JSON.stringify(params.loras || []));
  // Acceleration (block swap / FBCache / Spectrum) -- same fields/defaults as
  // /generate/outpaint/video and /generate/inpaint/video.
  formData.append("blocks_to_swap", String(params.blocks_to_swap ?? 0));
  formData.append("fuse_output_proj", String(params.fuse_output_proj ?? false));
  formData.append("fbcache_enable", String(params.fbcache_enable ?? false));
  formData.append("fbcache_threshold", String(params.fbcache_threshold ?? 0.12));
  formData.append("fbcache_warmup_steps", String(params.fbcache_warmup_steps ?? 1));
  formData.append("spectrum_enable", String(params.spectrum_enable ?? false));
  formData.append("spectrum_w", String(params.spectrum_w ?? 0.5));
  formData.append("spectrum_w_decay", String(params.spectrum_w_decay ?? 0.0));
  formData.append("spectrum_delta_cap", String(params.spectrum_delta_cap ?? 0.0));
  formData.append("spectrum_m", String(params.spectrum_m ?? 4));
  formData.append("spectrum_lam", String(params.spectrum_lam ?? 0.1));
  formData.append("spectrum_warmup_steps", String(params.spectrum_warmup_steps ?? 3));
  formData.append("spectrum_window_size", String(params.spectrum_window_size ?? 4));
  formData.append("spectrum_flex_window", String(params.spectrum_flex_window ?? 0.75));
  formData.append("spectrum_tail", String(params.spectrum_tail ?? 0.12));
  formData.append("spectrum_max_cache", String(params.spectrum_max_cache ?? 0));
  // Chain provenance -- present only when this request is a chain's first
  // segment (design §13).
  appendChainProvenance(formData, params);

  const response = await postGenerationRequest("/generate/img2vid", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

// ref2vid: multipart POST /generate/ref2vid — MiniMax-H3's `ref2va` workflow,
// which needs the ref2va transformer variant loaded (the fl2va one serves
// txt2vid/img2vid/outpaint and is refused here by name).
//
// The ORDER of each list is semantic: it labels the references in the prompt
// presentation (<Picture i> / <Audio j> / <Video k>) and lays them out on the
// packed sequence's rotary clock, so the arrays are sent in the order the user
// arranged them and are never sorted or regrouped here.
export const generateRef2Vid = async (
  params: Ref2VidParams,
  references: MiniMaxH3References,
) => {
  const formData = new FormData();

  formData.append("prompt", params.prompt);
  formData.append("negative_prompt", params.negative_prompt || "");
  formData.append("width", String(params.width ?? 1344));
  formData.append("height", String(params.height ?? 768));
  formData.append("num_frames", String(params.num_frames ?? 124));
  formData.append("frame_rate", String(params.frame_rate ?? 24.0));
  formData.append("num_inference_steps", String(params.num_inference_steps ?? 20));
  formData.append("guidance_scale", String(params.guidance_scale ?? 1.0));
  formData.append("seed", String(params.seed ?? -1));
  formData.append("num_videos_per_prompt", String(params.num_videos_per_prompt ?? 1));
  formData.append("max_sequence_length", String(params.max_sequence_length ?? 1024));
  formData.append("audio_enable", String(params.audio_enable ?? true));
  formData.append("reference_image_size", params.reference_image_size ?? "max");
  if (params.vae_path) {
    formData.append("vae_path", params.vae_path);
  }
  if (params.text_encoder_path) {
    formData.append("text_encoder_path", params.text_encoder_path);
  }
  if (params.unet_quantization && params.unet_quantization !== "none") {
    formData.append("unet_quantization", params.unet_quantization);
  }
  if (params.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", params.quantized_gemm_mode);
  }
  formData.append("attention_type", resolveGlobalAttentionType(params.attention_type));

  // The reference files. Each list keeps its order; a video's soundtrack is
  // positional, so a video with no sound sends an EMPTY part to hold its slot
  // (the backend treats a part with no filename as absent).
  (references.images || []).forEach((file) => formData.append("reference_images", file));
  (references.videos || []).forEach((file) => formData.append("reference_videos", file));
  if ((references.videoAudios || []).some((file) => file)) {
    (references.videos || []).forEach((_video, index) => {
      const soundtrack = (references.videoAudios || [])[index];
      formData.append(
        "reference_video_audios",
        soundtrack ?? new File([], ""),
      );
    });
  }
  (references.audios || []).forEach((file) => formData.append("reference_audios", file));

  // C5: optional keyframe anchors, laid out AFTER the reference blocks --
  // same positional-pair sender as generateImg2Vid's.
  for (const keyframe of params.keyframes ?? []) {
    if (!keyframe || !keyframe.image) continue;
    if (typeof keyframe.image === "string") {
      const keyframeResponse = await fetch(keyframe.image);
      const keyframeBlob = await keyframeResponse.blob();
      formData.append("keyframe_images", keyframeBlob, "keyframe.png");
    } else {
      formData.append("keyframe_images", keyframe.image);
    }
    formData.append("keyframe_frame_indices", String(keyframe.frame_index));
  }
  formData.append("loras", JSON.stringify(params.loras || []));
  // Acceleration (block swap / FBCache / Spectrum) -- same fields/defaults as
  // /generate/outpaint/video and /generate/inpaint/video.
  formData.append("blocks_to_swap", String(params.blocks_to_swap ?? 0));
  formData.append("fuse_output_proj", String(params.fuse_output_proj ?? false));
  formData.append("fbcache_enable", String(params.fbcache_enable ?? false));
  formData.append("fbcache_threshold", String(params.fbcache_threshold ?? 0.12));
  formData.append("fbcache_warmup_steps", String(params.fbcache_warmup_steps ?? 1));
  formData.append("spectrum_enable", String(params.spectrum_enable ?? false));
  formData.append("spectrum_w", String(params.spectrum_w ?? 0.5));
  formData.append("spectrum_w_decay", String(params.spectrum_w_decay ?? 0.0));
  formData.append("spectrum_delta_cap", String(params.spectrum_delta_cap ?? 0.0));
  formData.append("spectrum_m", String(params.spectrum_m ?? 4));
  formData.append("spectrum_lam", String(params.spectrum_lam ?? 0.1));
  formData.append("spectrum_warmup_steps", String(params.spectrum_warmup_steps ?? 3));
  formData.append("spectrum_window_size", String(params.spectrum_window_size ?? 4));
  formData.append("spectrum_flex_window", String(params.spectrum_flex_window ?? 0.75));
  formData.append("spectrum_tail", String(params.spectrum_tail ?? 0.12));
  formData.append("spectrum_max_cache", String(params.spectrum_max_cache ?? 0));
  // Chain provenance -- present only when this request is a chain's first
  // segment (design §13).
  appendChainProvenance(formData, params);

  const response = await postGenerationRequest("/generate/ref2vid", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

// ---------------------------------------------------------------------------
// Audio generation (ACE-Step 1.5)
// ---------------------------------------------------------------------------

// txt2aud: JSON POST /generate/txt2aud. Response is the standard
// GenerationResponse ({ success, image, actual_seed, warnings }); image.filename
// is a .flac file under /outputs/.
export const generateTxt2Aud = async (params: Txt2AudParams) => {
  const body = {
    prompt: params.prompt,
    lyrics: params.lyrics || "",
    audio_duration: params.audio_duration ?? 30.0,
    seed: params.seed ?? -1,
    inference_steps: params.inference_steps ?? 8,
    guidance_scale: params.guidance_scale ?? 1.0,
    shift: params.shift ?? 3.0,
    sampler_mode: params.sampler_mode ?? "euler",
    vocal_language: params.vocal_language ?? "en",
    // MiniMax Music 3 ONLY. `?? undefined`, NOT `?? null`: JSON.stringify
    // drops an `undefined`-valued key entirely, so an omitted value here
    // reaches the backend as an OMITTED field, letting `Txt2AudRequest`'s
    // `model_fields_set` (and therefore `resolve_audio_defaults`) fill it
    // from MiniMax Music 3's own overlay (30 / 1.7) exactly as designed.
    // Sending an explicit `null` would do the opposite of "harmless": it
    // would count as the client having PROVIDED the field (Pydantic still
    // marks an explicit `null` as set), permanently defeating that
    // resolution, AND -- because MiniMax Music 3's pipeline backend raises a
    // ValidationError on either of these being `None` -- would turn a
    // pre-update queued item with no value here (e.g. one persisted across
    // this exact update, before these fields existed) into a hard 400
    // instead of a working generation at the arch's own defaults.
    num_inference_steps: params.num_inference_steps ?? undefined,
    flow_guidance_scale: params.flow_guidance_scale ?? undefined,
    loras: params.loras || [],
    // `=== "none" -> null` mirrors every other sender: "none" is the UI's
    // spelling of "no quantization", and sending it as a value would come back
    // as an `unsupported_param` warning on every txt2aud.
    unet_quantization:
      params.unet_quantization && params.unet_quantization !== "none"
        ? params.unet_quantization
        : null,
    quantized_gemm_mode: params.quantized_gemm_mode ?? null,
  };

  const response = await postGenerationRequest("/generate/txt2aud", body);
  return response.data;
};

// aud2aud (cover): multipart POST /generate/aud2aud with an uploaded
// `reference_audio` clip. Every AUD2AUD field is appended explicitly
// (CLAUDE.md param-threading), mirroring generateImg2Vid's FormData construction.
export const generateAud2Aud = async (params: Aud2AudParams, referenceAudio: File | string) => {
  const formData = new FormData();

  // Handle both File objects and data URLs (mirrors generateImg2Vid's `image` handling).
  if (typeof referenceAudio === "string") {
    const response = await fetch(referenceAudio);
    const blob = await response.blob();
    formData.append("reference_audio", blob, "reference.wav");
  } else {
    formData.append("reference_audio", referenceAudio);
  }

  formData.append("prompt", params.prompt);
  formData.append("lyrics", params.lyrics || "");
  formData.append("seed", String(params.seed ?? -1));
  formData.append("inference_steps", String(params.inference_steps ?? 8));
  formData.append("guidance_scale", String(params.guidance_scale ?? 1.0));
  formData.append("shift", String(params.shift ?? 3.0));
  formData.append("cover_strength", String(params.cover_strength ?? 1.0));
  formData.append("vocal_language", params.vocal_language ?? "en");
  formData.append("loras", JSON.stringify(params.loras || []));
  formData.append("mode", params.mode ?? "cover");
  formData.append("repaint_start", String(params.repaint_start ?? 0.0));
  formData.append("repaint_end", String(params.repaint_end ?? 0.0));
  // MiniMax Music 3 repaint only. Appended ONLY when set (CLAUDE.md
  // "sending null is not omitting it" -- an always-sent `music3_repaint_mode`
  // would land in FastAPI's bound value regardless of what the arch's own
  // overlay would otherwise resolve, and the pipeline backend raises a
  // ValidationError on `num_inference_steps`/`flow_guidance_scale` being
  // `None`, which an unconditional `?? 30`/`?? 1.7` literal here would defeat
  // for a pre-update queued item). Mirrors generateOutpaintAudio's identical
  // convention for its own MiniMax Music 3 -only fields.
  if (params.music3_repaint_mode) {
    formData.append("music3_repaint_mode", params.music3_repaint_mode);
  }
  if (params.num_inference_steps !== undefined) {
    formData.append("num_inference_steps", String(params.num_inference_steps));
  }
  if (params.flow_guidance_scale !== undefined) {
    formData.append("flow_guidance_scale", String(params.flow_guidance_scale));
  }
  // Weight-only quantization (both axes). Appended only when set, so an unset
  // field leaves the backend default (and the process GEMM flags) untouched.
  if (params.unet_quantization && params.unet_quantization !== "none") {
    formData.append("unet_quantization", params.unet_quantization);
  }
  if (params.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", params.quantized_gemm_mode);
  }

  const response = await postGenerationRequest("/generate/aud2aud", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const generateInpaint = async (params: InpaintParams, image: File | string, mask: File | string) => {
  // Only load ControlNet images if they exist (avoid unnecessary localStorage access)
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "inpaint_controlnet_collapsed")
    : params.controlnets;

  const paramsWithImages = {
    ...params,
    attention_type: resolveGlobalAttentionType(params.attention_type),
    attention_impl: resolveGlobalAttentionImpl(params.attention_impl),
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
  // SenseNova U1.5 flow-matching time-shift; every other architecture ignores it.
  // Module-level helper, no React context here -- mirrors DEFAULT_PARAMS' fallback.
  formData.append("timestep_shift", String(paramsWithImages.timestep_shift ?? 3.0));
  // SenseNova U1.5 second CFG scale; inert without ref_images, ignored elsewhere.
  formData.append("img_cfg_scale", String(paramsWithImages.img_cfg_scale ?? 1.0));
  // SenseNova U1.5 CFG-overshoot clamp; every other architecture ignores it.
  formData.append("cfg_norm", paramsWithImages.cfg_norm ?? "global");
  // SenseNova U1.5 per-phase weight-half CPU eviction; every other architecture ignores it.
  formData.append("sensenova_mot_phase_eviction", String(paramsWithImages.sensenova_mot_phase_eviction ?? false));
  // SenseNova U1.5 per-layer prefix KV cache CPU streaming; every other architecture ignores it.
  formData.append("sensenova_kv_cache_streaming", String(paramsWithImages.sensenova_kv_cache_streaming ?? false));
  formData.append("denoising_strength", String(paramsWithImages.denoising_strength || 0.75));
  formData.append("img2img_fix_steps", String(paramsWithImages.img2img_fix_steps ?? true));
  formData.append("vae_drift_correction", String(paramsWithImages.vae_drift_correction ?? false));
  formData.append("mask_blur", String(paramsWithImages.mask_blur || 4));
  // Regional additional prompt (SD/SDXL only): conditions ONLY the repaint mask region
  formData.append("region_prompt", paramsWithImages.region_prompt || "");
  formData.append("region_negative_prompt", paramsWithImages.region_negative_prompt || "");
  formData.append("region_prompt_strength", String(paramsWithImages.region_prompt_strength ?? 1.0));
  formData.append("region_prompt_method", paramsWithImages.region_prompt_method || "cfg");
  formData.append("region_mask_feather", String(paramsWithImages.region_mask_feather ?? 0.0));
  // Seam Structure Continuity (SSC, SD/SDXL only); 0 = off
  formData.append("seam_structure_strength", String(paramsWithImages.seam_structure_strength ?? 0.0));
  formData.append("seam_structure_depth", String(paramsWithImages.seam_structure_depth ?? 6.0));
  formData.append("seam_structure_end", String(paramsWithImages.seam_structure_end ?? 0.70));
  formData.append("seam_structure_saliency", String(paramsWithImages.seam_structure_saliency ?? 2.0));
  formData.append("seam_structure_max_area", String(paramsWithImages.seam_structure_max_area ?? 0.25));
  // Boundary Determinism Relaxation (BDR, SD/SDXL only); 0 = off
  formData.append("boundary_relax_strength", String(paramsWithImages.boundary_relax_strength ?? 0.0));
  formData.append("boundary_relax_width", String(paramsWithImages.boundary_relax_width ?? 3.0));
  formData.append("boundary_relax_noise", String(paramsWithImages.boundary_relax_noise ?? 0.35));
  formData.append("boundary_relax_full_until", String(paramsWithImages.boundary_relax_full_until ?? 0.37));
  formData.append("boundary_relax_end", String(paramsWithImages.boundary_relax_end ?? 0.55));
  formData.append("boundary_relax_paste", String(paramsWithImages.boundary_relax_paste ?? "feather"));
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
  formData.append("attention_impl", paramsWithImages.attention_impl || "conduit");

  // Block swap (CPU offloading)
  formData.append("enable_block_swap", String(paramsWithImages.enable_block_swap ?? false));
  formData.append("blocks_to_swap", String(paramsWithImages.blocks_to_swap ?? 20));
  formData.append("use_pinned_memory", String(paramsWithImages.use_pinned_memory ?? false));
  formData.append("block_swap_h2d_only", String(paramsWithImages.block_swap_h2d_only ?? false));
  formData.append("block_swap_ring_size", String(paramsWithImages.block_swap_ring_size ?? 2));

  // Debug log for quantization
  console.log('[API] inpaint unet_quantization:', paramsWithImages.unet_quantization);
  if (paramsWithImages.unet_quantization && paramsWithImages.unet_quantization !== "none") {
    formData.append("unet_quantization", paramsWithImages.unet_quantization);
    console.log('[API] Added unet_quantization to FormData:', paramsWithImages.unet_quantization);
  } else {
    console.log('[API] No quantization or "none" selected');
  }
  // Quantized GEMM path (already-quantized checkpoints: ideogram4/krea2/anima).
  // Sent ONLY when the user picked an explicit value; omitting it leaves the
  // backend's process-level setting (env var / Settings panel) untouched.
  if (paramsWithImages.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", paramsWithImages.quantized_gemm_mode);
  }

  if (paramsWithImages.text_encoder_quantization && paramsWithImages.text_encoder_quantization !== "none") {
    formData.append("text_encoder_quantization", paramsWithImages.text_encoder_quantization);
  }

  // CPU text encoding
  formData.append("cpu_text_encoding", String(paramsWithImages.cpu_text_encoding ?? false));

  // torch.compile optimization
  formData.append("use_torch_compile", String(paramsWithImages.use_torch_compile ?? false));
  formData.append("vae_tiling", String(paramsWithImages.vae_tiling ?? false));
  formData.append("vae_tile_threshold", String(paramsWithImages.vae_tile_threshold ?? 0));
  formData.append("vae_tile_mode", String(paramsWithImages.vae_tile_mode ?? "blend"));
  formData.append("vae_tile_global_norm", String(paramsWithImages.vae_tile_global_norm ?? false));
  // Keep model components GPU-resident for the next queued generation (set by the queue dispatcher)
  formData.append("keep_models_hot", String(paramsWithImages.keep_models_hot ?? false));
  // Color Flatten: chroma-smoothing baked into the saved image at generation time
  formData.append("color_flatten_strength", String(paramsWithImages.color_flatten_strength ?? 0));
  // In-loop background hard-flatten (final-step flat-region solid-color replacement)
  formData.append("flatten_in_loop", String(paramsWithImages.flatten_in_loop ?? false));
  formData.append("flatten_in_loop_last_steps", String(paramsWithImages.flatten_in_loop_last_steps ?? 3));
  formData.append("flatten_in_loop_min_region", String(paramsWithImages.flatten_in_loop_min_region ?? 0.02));
  // Spectrum (Adaptive Spectral Feature Forecasting) acceleration (txt2img only in v1;
  // img2img/inpaint backends ignore these until wired)
  formData.append("spectrum_enable", String(paramsWithImages.spectrum_enable ?? false));
  formData.append("fbcache_enable", String(paramsWithImages.fbcache_enable ?? false));
  formData.append("fbcache_threshold", String(paramsWithImages.fbcache_threshold ?? 0.12));
  formData.append("fbcache_warmup_steps", String(paramsWithImages.fbcache_warmup_steps ?? 1));
  formData.append("spectrum_w", String(paramsWithImages.spectrum_w ?? 0.5));
  formData.append("spectrum_w_decay", String(paramsWithImages.spectrum_w_decay ?? 0.0));
  formData.append("spectrum_delta_cap", String(paramsWithImages.spectrum_delta_cap ?? 0.0));
  formData.append("spectrum_m", String(paramsWithImages.spectrum_m ?? 4));
  formData.append("spectrum_lam", String(paramsWithImages.spectrum_lam ?? 0.1));
  formData.append("spectrum_warmup_steps", String(paramsWithImages.spectrum_warmup_steps ?? 3));
  formData.append("spectrum_window_size", String(paramsWithImages.spectrum_window_size ?? 4));
  formData.append("spectrum_flex_window", String(paramsWithImages.spectrum_flex_window ?? 0.75));
  formData.append("spectrum_tail", String(paramsWithImages.spectrum_tail ?? 0.12));
  formData.append("spectrum_feature_mode", String(paramsWithImages.spectrum_feature_mode ?? "output"));
  formData.append("spectrum_cache_branch", String(paramsWithImages.spectrum_cache_branch ?? 1));
  formData.append("spectrum_max_cache", String(paramsWithImages.spectrum_max_cache ?? 0));

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

  // VAE / Text encoder override paths (empty = model default)
  if (paramsWithImages.vae_path) {
    formData.append("vae_path", paramsWithImages.vae_path);
  }
  if (paramsWithImages.text_encoder_path) {
    formData.append("text_encoder_path", paramsWithImages.text_encoder_path);
  }
  // PiD decoder options (only meaningful when vae_path selects a PiD checkpoint;
  // ignored server-side for a normal VAE override / no override)
  formData.append("pid_sr_output", paramsWithImages.pid_sr_output || "4x");
  formData.append("pid_use_gemma", String(paramsWithImages.pid_use_gemma ?? false));
  formData.append("pid_low_vram", String(paramsWithImages.pid_low_vram ?? false));
  formData.append("pid_tile_native", String(paramsWithImages.pid_tile_native ?? 512));
  formData.append("pid_tile_overlap_ratio", String(paramsWithImages.pid_tile_overlap_ratio ?? 0.25));
  formData.append("pid_fast_large_decode", String(paramsWithImages.pid_fast_large_decode ?? false));

  // Loop-generation decode mode (heavy-decoder aware; see loopGenerationInheritance.ts).
  // NOTE: inpaint does NOT support loop_decode="none" / input_latent_id (backend
  // rejects it) — loop steps fall back to "cheap"+skip_gallery for intermediates.
  formData.append("loop_decode", paramsWithImages.loop_decode || "full");
  formData.append("skip_gallery", String(paramsWithImages.skip_gallery ?? false));

  const response = await postGenerationRequest("/generate/inpaint", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

// Outpaint: clone of generateInpaint's FormData sender, minus the mask
// upload (the backend builds its own canvas + mask from `image` + the
// placement fields), plus the placement fields themselves. See
// core/inference/outpaint_utils.py + PipelineManager.generate_outpaint.
export const generateOutpaint = async (params: OutpaintParams, image: File | string) => {
  // Only load ControlNet images if they exist (avoid unnecessary localStorage access)
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "outpaint_controlnet_collapsed")
    : params.controlnets;

  const paramsWithImages = {
    ...params,
    attention_type: resolveGlobalAttentionType(params.attention_type),
    attention_impl: resolveGlobalAttentionImpl(params.attention_impl),
    controlnets: controlnets,
  };

  const formData = new FormData();

  // Handle both File objects and data URLs for image (no mask -- outpaint
  // builds its own canvas + mask from the placement fields below).
  if (typeof image === 'string') {
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
  formData.append("denoising_strength", String(paramsWithImages.denoising_strength ?? 1.0));
  formData.append("img2img_fix_steps", String(paramsWithImages.img2img_fix_steps ?? true));
  formData.append("sampler", paramsWithImages.sampler || "euler");
  formData.append("schedule_type", paramsWithImages.schedule_type || "uniform");
  formData.append("seed", String(paramsWithImages.seed || -1));
  formData.append("ancestral_seed", String(paramsWithImages.ancestral_seed ?? -1));

  // Placement (outpaint-only). canvas_width/canvas_height supersede
  // width/height -- those are NOT sent for outpaint.
  formData.append("canvas_width", String(paramsWithImages.canvas_width ?? 1536));
  formData.append("canvas_height", String(paramsWithImages.canvas_height ?? 1536));
  formData.append("place_x", String(paramsWithImages.place_x ?? 0));
  formData.append("place_y", String(paramsWithImages.place_y ?? 0));
  formData.append("place_width", String(paramsWithImages.place_width ?? 0));
  formData.append("place_height", String(paramsWithImages.place_height ?? 0));
  formData.append("input_crop_x", String(paramsWithImages.input_crop_x ?? 0));
  formData.append("input_crop_y", String(paramsWithImages.input_crop_y ?? 0));
  formData.append("input_crop_w", String(paramsWithImages.input_crop_w ?? 0));
  formData.append("input_crop_h", String(paramsWithImages.input_crop_h ?? 0));
  formData.append("outpaint_fill_mode", paramsWithImages.outpaint_fill_mode || "replicate");

  formData.append("mask_blur", String(paramsWithImages.mask_blur ?? 4));
  formData.append("inpaint_full_res", String(paramsWithImages.inpaint_full_res || false));
  formData.append("inpaint_full_res_padding", String(paramsWithImages.inpaint_full_res_padding || 32));
  formData.append("inpaint_fill_mode", paramsWithImages.inpaint_fill_mode || "original");
  formData.append("inpaint_fill_strength", String(paramsWithImages.inpaint_fill_strength ?? 1.0));
  formData.append("inpaint_blur_strength", String(paramsWithImages.inpaint_blur_strength ?? 1.0));
  // Regional additional prompt (SD/SDXL only): conditions ONLY the generated region
  formData.append("region_prompt", paramsWithImages.region_prompt || "");
  formData.append("region_negative_prompt", paramsWithImages.region_negative_prompt || "");
  formData.append("region_prompt_strength", String(paramsWithImages.region_prompt_strength ?? 1.0));
  formData.append("region_prompt_method", paramsWithImages.region_prompt_method || "cfg");
  formData.append("region_mask_feather", String(paramsWithImages.region_mask_feather ?? 0.0));
  // Seam Structure Continuity (SSC, SD/SDXL only); 0 = off
  formData.append("seam_structure_strength", String(paramsWithImages.seam_structure_strength ?? 0.0));
  formData.append("seam_structure_depth", String(paramsWithImages.seam_structure_depth ?? 6.0));
  formData.append("seam_structure_end", String(paramsWithImages.seam_structure_end ?? 0.70));
  formData.append("seam_structure_saliency", String(paramsWithImages.seam_structure_saliency ?? 2.0));
  formData.append("seam_structure_max_area", String(paramsWithImages.seam_structure_max_area ?? 0.25));
  // Boundary Determinism Relaxation (BDR, SD/SDXL only); 0 = off
  formData.append("boundary_relax_strength", String(paramsWithImages.boundary_relax_strength ?? 0.0));
  formData.append("boundary_relax_width", String(paramsWithImages.boundary_relax_width ?? 3.0));
  formData.append("boundary_relax_noise", String(paramsWithImages.boundary_relax_noise ?? 0.35));
  formData.append("boundary_relax_full_until", String(paramsWithImages.boundary_relax_full_until ?? 0.37));
  formData.append("boundary_relax_end", String(paramsWithImages.boundary_relax_end ?? 0.55));
  formData.append("boundary_relax_paste", String(paramsWithImages.boundary_relax_paste ?? "feather"));
  // Outpaint ControlNet (structure continuity, SD/SDXL only); false = off
  formData.append("outpaint_controlnet_enable", String(paramsWithImages.outpaint_controlnet_enable ?? false));
  formData.append("outpaint_controlnet_mode", paramsWithImages.outpaint_controlnet_mode ?? "crop_mask");
  formData.append("outpaint_controlnet_model", paramsWithImages.outpaint_controlnet_model ?? "");
  formData.append("outpaint_controlnet_detector", paramsWithImages.outpaint_controlnet_detector ?? "canny");
  formData.append("outpaint_controlnet_scale", String(paramsWithImages.outpaint_controlnet_scale ?? 0.6));
  formData.append("outpaint_controlnet_guidance_start", String(paramsWithImages.outpaint_controlnet_guidance_start ?? 0.0));
  formData.append("outpaint_controlnet_guidance_end", String(paramsWithImages.outpaint_controlnet_guidance_end ?? 0.55));
  formData.append("outpaint_controlnet_depth", String(paramsWithImages.outpaint_controlnet_depth ?? 160));
  formData.append("outpaint_controlnet_taper", String(paramsWithImages.outpaint_controlnet_taper ?? 2.0));
  formData.append("outpaint_controlnet_corner_radius_px", String(paramsWithImages.outpaint_controlnet_corner_radius_px ?? 0.0));
  formData.append("outpaint_controlnet_corner_gate_radius_px", String(paramsWithImages.outpaint_controlnet_corner_gate_radius_px ?? 0.0));
  formData.append("outpaint_controlnet_corner_gate_min", String(paramsWithImages.outpaint_controlnet_corner_gate_min ?? 1.0));
  formData.append("outpaint_pin_corner_relax_radius_px", String(paramsWithImages.outpaint_pin_corner_relax_radius_px ?? 0.0));
  formData.append("outpaint_pin_corner_relax_min", String(paramsWithImages.outpaint_pin_corner_relax_min ?? 1.0));
  // Harmonic boundary-offset membrane (post-decode); false = off
  formData.append("outpaint_seam_membrane", String(paramsWithImages.outpaint_seam_membrane ?? false));
  formData.append("outpaint_seam_membrane_band", String(paramsWithImages.outpaint_seam_membrane_band ?? 0));
  // Cross-seam low-frequency tone membrane ("R2", post-decode); 0 = off
  formData.append("outpaint_seam_tone_strength", String(paramsWithImages.outpaint_seam_tone_strength ?? 0.0));
  formData.append("outpaint_seam_tone_band", String(paramsWithImages.outpaint_seam_tone_band ?? 0));
  // Boundary-offset propagation ("G_prop16", post-decode); 0 = off (byte-identical, generated-side-only)
  formData.append("outpaint_seam_offset_prop", String(paramsWithImages.outpaint_seam_offset_prop ?? 1.0));
  // In-loop continuity fixes B1/B2/B3 (SD/SDXL only)
  formData.append("outpaint_boundary_color_strength", String(paramsWithImages.outpaint_boundary_color_strength ?? 0.25));
  formData.append("outpaint_resample_count", String(paramsWithImages.outpaint_resample_count ?? 1));
  formData.append("outpaint_jump_length", String(paramsWithImages.outpaint_jump_length ?? 4));
  formData.append("outpaint_reference_strength", String(paramsWithImages.outpaint_reference_strength ?? 0.0));
  // Paste-band reconciliation feather ("Option E"); 0 = off (byte-identical)
  formData.append("outpaint_paste_feather_px", String(paramsWithImages.outpaint_paste_feather_px ?? 0));
  // Preserved-region compositing mode; "exact" = off (byte-identical)
  formData.append("outpaint_preserve_mode", paramsWithImages.outpaint_preserve_mode ?? "exact");
  // Honest outpaint preview (display-only); false = off
  formData.append("outpaint_preview_unpinned_x0", String(paramsWithImages.outpaint_preview_unpinned_x0 ?? false));
  formData.append("prompt_chunking_mode", paramsWithImages.prompt_chunking_mode || "a1111");
  formData.append("max_prompt_chunks", String(paramsWithImages.max_prompt_chunks ?? 0));
  formData.append("loras", JSON.stringify(paramsWithImages.loras || []));
  formData.append("controlnets", JSON.stringify(paramsWithImages.controlnets || []));
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
  formData.append("attention_impl", paramsWithImages.attention_impl || "conduit");

  // Block swap (CPU offloading)
  formData.append("enable_block_swap", String(paramsWithImages.enable_block_swap ?? false));
  formData.append("blocks_to_swap", String(paramsWithImages.blocks_to_swap ?? 20));
  formData.append("use_pinned_memory", String(paramsWithImages.use_pinned_memory ?? false));
  formData.append("block_swap_h2d_only", String(paramsWithImages.block_swap_h2d_only ?? false));
  formData.append("block_swap_ring_size", String(paramsWithImages.block_swap_ring_size ?? 2));

  if (paramsWithImages.unet_quantization && paramsWithImages.unet_quantization !== "none") {
    formData.append("unet_quantization", paramsWithImages.unet_quantization);
  }
  if (paramsWithImages.text_encoder_quantization && paramsWithImages.text_encoder_quantization !== "none") {
    formData.append("text_encoder_quantization", paramsWithImages.text_encoder_quantization);
  }
  // Quantized GEMM path (already-quantized checkpoints: ideogram4/krea2/anima).
  // Sent ONLY when the user picked an explicit value; omitting it leaves the
  // backend's process-level setting (env var / Settings panel) untouched.
  if (paramsWithImages.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", paramsWithImages.quantized_gemm_mode);
  }

  formData.append("cpu_text_encoding", String(paramsWithImages.cpu_text_encoding ?? false));
  formData.append("use_torch_compile", String(paramsWithImages.use_torch_compile ?? false));
  formData.append("vae_tiling", String(paramsWithImages.vae_tiling ?? false));
  formData.append("vae_tile_threshold", String(paramsWithImages.vae_tile_threshold ?? 0));
  formData.append("vae_tile_mode", String(paramsWithImages.vae_tile_mode ?? "blend"));
  formData.append("vae_tile_global_norm", String(paramsWithImages.vae_tile_global_norm ?? false));
  formData.append("keep_models_hot", String(paramsWithImages.keep_models_hot ?? false));
  formData.append("color_flatten_strength", String(paramsWithImages.color_flatten_strength ?? 0));
  formData.append("vae_drift_correction", String(paramsWithImages.vae_drift_correction ?? false));
  formData.append("flatten_in_loop", String(paramsWithImages.flatten_in_loop ?? false));
  formData.append("flatten_in_loop_last_steps", String(paramsWithImages.flatten_in_loop_last_steps ?? 3));
  formData.append("flatten_in_loop_min_region", String(paramsWithImages.flatten_in_loop_min_region ?? 0.02));
  formData.append("spectrum_enable", String(paramsWithImages.spectrum_enable ?? false));
  formData.append("fbcache_enable", String(paramsWithImages.fbcache_enable ?? false));
  formData.append("fbcache_threshold", String(paramsWithImages.fbcache_threshold ?? 0.12));
  formData.append("fbcache_warmup_steps", String(paramsWithImages.fbcache_warmup_steps ?? 1));
  formData.append("fbcache_cache_branch", String(paramsWithImages.fbcache_cache_branch ?? 1));
  formData.append("spectrum_w", String(paramsWithImages.spectrum_w ?? 0.5));
  formData.append("spectrum_w_decay", String(paramsWithImages.spectrum_w_decay ?? 0.0));
  formData.append("spectrum_delta_cap", String(paramsWithImages.spectrum_delta_cap ?? 0.0));
  formData.append("spectrum_m", String(paramsWithImages.spectrum_m ?? 4));
  formData.append("spectrum_lam", String(paramsWithImages.spectrum_lam ?? 0.1));
  formData.append("spectrum_warmup_steps", String(paramsWithImages.spectrum_warmup_steps ?? 3));
  formData.append("spectrum_window_size", String(paramsWithImages.spectrum_window_size ?? 4));
  formData.append("spectrum_flex_window", String(paramsWithImages.spectrum_flex_window ?? 0.75));
  formData.append("spectrum_tail", String(paramsWithImages.spectrum_tail ?? 0.12));
  formData.append("spectrum_feature_mode", String(paramsWithImages.spectrum_feature_mode ?? "output"));
  formData.append("spectrum_cache_branch", String(paramsWithImages.spectrum_cache_branch ?? 1));
  formData.append("spectrum_max_cache", String(paramsWithImages.spectrum_max_cache ?? 0));

  // TIPO prompt upsampling (not exposed in the Outpaint UI in Phase 1; always
  // sent as the disabled default so the Form parameter is satisfied).
  formData.append("use_tipo", String(paramsWithImages.use_tipo ?? false));
  formData.append("tipo_config", JSON.stringify(paramsWithImages.tipo_config || {}));

  formData.append("preview_predicted_x0", String(paramsWithImages.preview_predicted_x0 ?? false));
  formData.append("preview_decoder", String(paramsWithImages.preview_decoder ?? "matrix"));

  // FLUX.2 Image Edit / Vision Encoder (reference images)
  if (paramsWithImages.ref_images && paramsWithImages.ref_images.length > 0) {
    for (let i = 0; i < paramsWithImages.ref_images.length; i++) {
      formData.append("ref_images", paramsWithImages.ref_images[i]);
    }
  }

  if (paramsWithImages.vision_encoder_path) {
    formData.append("vision_encoder_path", paramsWithImages.vision_encoder_path);
  }
  if (paramsWithImages.vae_path) {
    formData.append("vae_path", paramsWithImages.vae_path);
  }
  if (paramsWithImages.text_encoder_path) {
    formData.append("text_encoder_path", paramsWithImages.text_encoder_path);
  }
  formData.append("pid_sr_output", paramsWithImages.pid_sr_output || "4x");
  formData.append("pid_use_gemma", String(paramsWithImages.pid_use_gemma ?? false));
  formData.append("pid_low_vram", String(paramsWithImages.pid_low_vram ?? false));
  formData.append("pid_tile_native", String(paramsWithImages.pid_tile_native ?? 512));
  formData.append("pid_tile_overlap_ratio", String(paramsWithImages.pid_tile_overlap_ratio ?? 0.25));
  formData.append("pid_fast_large_decode", String(paramsWithImages.pid_fast_large_decode ?? false));

  // SDXL micro-conditioning original_size override
  if (paramsWithImages.original_size_w) formData.append("original_size_w", String(paramsWithImages.original_size_w));
  if (paramsWithImages.original_size_h) formData.append("original_size_h", String(paramsWithImages.original_size_h));
  if (paramsWithImages.original_size_scale !== undefined && paramsWithImages.original_size_scale !== null) {
    formData.append("original_size_scale", String(paramsWithImages.original_size_scale));
  }

  // Loop-generation decode mode is out of scope for Outpaint (Phase 1) --
  // always send the "full" default. input_latent_id is never sent (the
  // backend rejects any non-null value for outpaint).
  formData.append("loop_decode", paramsWithImages.loop_decode || "full");
  formData.append("skip_gallery", String(paramsWithImages.skip_gallery ?? false));

  const response = await postGenerationRequest("/generate/outpaint", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

// Video temporal outpaint (LTX-2.3): multipart POST /generate/outpaint/video
// with an uploaded `video` clip. Clone of generateImg2Vid's FormData sender
// (CLAUDE.md param-threading) plus the placement fields; every
// OutpaintVideoParams field is appended explicitly, matching the Form
// parameter names of routes.py's generate_outpaint_video 1:1.
export const generateOutpaintVideo = async (
  params: OutpaintVideoParams,
  video: File | string,
  bridgeVideo?: File | string | null,
  referenceImages?: File[]
) => {
  const formData = new FormData();

  // Handle both File objects and data URLs (mirrors generateImg2Vid's `image` handling).
  if (typeof video === "string") {
    const response = await fetch(video);
    const blob = await response.blob();
    formData.append("video", blob, "input.mp4");
  } else {
    formData.append("video", video);
  }

  // BRIDGE placement: a second clip preserved at the END of the timeline, with
  // the generated span between the two. Appended only when present -- the
  // field's presence is what selects the placement server-side, and an
  // architecture without a bridge placement answers 400 rather than ignoring
  // it (ignoring it would silently produce a one-clip result).
  if (bridgeVideo) {
    if (typeof bridgeVideo === "string") {
      const response = await fetch(bridgeVideo);
      const blob = await response.blob();
      formData.append("bridge_video", blob, "bridge.mp4");
    } else {
      formData.append("bridge_video", bridgeVideo);
    }
  }

  formData.append("prompt", params.prompt);
  formData.append("negative_prompt", params.negative_prompt || "");
  formData.append("width", String(params.width ?? 768));
  formData.append("height", String(params.height ?? 512));
  formData.append("total_frames", String(params.total_frames ?? 121));
  formData.append("frame_rate", String(params.frame_rate ?? 24.0));
  formData.append("num_inference_steps", String(params.num_inference_steps ?? 8));
  formData.append("guidance_scale", String(params.guidance_scale ?? 1.0));
  formData.append("seed", String(params.seed ?? -1));
  formData.append("num_videos_per_prompt", String(params.num_videos_per_prompt ?? 1));
  formData.append("max_sequence_length", String(params.max_sequence_length ?? 1024));
  formData.append("audio_enable", String(params.audio_enable ?? true));

  // Placement (outpaint-only)
  formData.append("input_offset_frames", String(params.input_offset_frames ?? 0));
  formData.append("input_trim_start_frames", String(params.input_trim_start_frames ?? 0));
  formData.append("input_trim_end_frames", String(params.input_trim_end_frames ?? 0));
  // Sent ONLY when the caller actually has a value. The backend field is a
  // sentinel whose default is per-architecture ("regenerate" on LTX-2.3,
  // "preserve_input" on MiniMax-H3, which generates audio only for the frames
  // it generates), so appending a hardcoded fallback here would pin the base
  // value and silently defeat the overlay.
  if (params.outpaint_video_audio_mode) {
    formData.append("outpaint_video_audio_mode", params.outpaint_video_audio_mode);
  }

  // Attention backend: the global setting, exactly as every other sender reads
  // it. Honored by MiniMax-H3 (its transformer runs on SushiUI's conduit),
  // accepted-and-warned by LTX-2.3.
  formData.append("attention_type", resolveGlobalAttentionType(params.attention_type));

  // Acceleration (block swap / FBCache / Spectrum)
  formData.append("blocks_to_swap", String(params.blocks_to_swap ?? 0));
  formData.append("fuse_output_proj", String(params.fuse_output_proj ?? false));
  formData.append("fbcache_enable", String(params.fbcache_enable ?? false));
  formData.append("fbcache_threshold", String(params.fbcache_threshold ?? 0.12));
  formData.append("fbcache_warmup_steps", String(params.fbcache_warmup_steps ?? 1));
  formData.append("spectrum_enable", String(params.spectrum_enable ?? false));
  formData.append("spectrum_w", String(params.spectrum_w ?? 0.5));
  formData.append("spectrum_w_decay", String(params.spectrum_w_decay ?? 0.0));
  formData.append("spectrum_delta_cap", String(params.spectrum_delta_cap ?? 0.0));
  formData.append("spectrum_m", String(params.spectrum_m ?? 4));
  formData.append("spectrum_lam", String(params.spectrum_lam ?? 0.1));
  formData.append("spectrum_warmup_steps", String(params.spectrum_warmup_steps ?? 3));
  formData.append("spectrum_window_size", String(params.spectrum_window_size ?? 4));
  formData.append("spectrum_flex_window", String(params.spectrum_flex_window ?? 0.75));
  formData.append("spectrum_tail", String(params.spectrum_tail ?? 0.12));
  formData.append("spectrum_max_cache", String(params.spectrum_max_cache ?? 0));

  if (params.vae_path) {
    formData.append("vae_path", params.vae_path);
  }
  if (params.text_encoder_path) {
    formData.append("text_encoder_path", params.text_encoder_path);
  }
  // `&& !== "none"` mirrors the image senders: "none" is the UI's spelling of
  // "no quantization" and must not be sent as a value.
  if (params.unet_quantization && params.unet_quantization !== "none") {
    formData.append("unet_quantization", params.unet_quantization);
  }
  if (params.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", params.quantized_gemm_mode);
  }

  formData.append("video_lossless", String(params.video_lossless ?? false));

  // MiniMax-H3 ref2va only (extend_forward): optional image references, IN
  // UPLOAD ORDER (the order is part of the request). Always send
  // reference_image_size -- cheap, and the backend only reads it when there
  // is something to size.
  formData.append("reference_image_size", params.reference_image_size || "max");
  for (const image of referenceImages || []) {
    formData.append("reference_images", image);
  }
  formData.append("loras", JSON.stringify(params.loras || []));
  // Continuation context. Sent only when the caller actually chose one, so an
  // ordinary (unchained) video outpaint is byte-identical to what it was: the
  // backend's own default is `boundary_frame`, today's behaviour.
  if (params.continuation_mode) {
    formData.append("continuation_mode", params.continuation_mode);
    formData.append(
      "continuation_overlap_frames",
      String(params.continuation_overlap_frames ?? 0)
    );
    // Only for the mode that places anchors: sending a count with any other
    // one is a 400 by design, so it is not appended unconditionally.
    if (params.continuation_mode === "motion_preroll") {
      formData.append(
        "continuation_anchor_count",
        String(params.continuation_anchor_count ?? 0)
      );
    }
  }
  // Chain provenance (design §13). This endpoint runs every CONTINUATION
  // segment, so a chained request always carries it here; a plain video
  // outpaint carries nothing.
  appendChainProvenance(formData, params);

  const response = await postGenerationRequest("/generate/outpaint/video", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

// The filename passed to formData.append() below is what actually reaches
// the backend as the multipart part's filename (it overrides whatever name
// the File object itself carries), so this is the one place that needs to
// sanitize `part.id` -- it is presently a crypto.randomUUID() value from
// InpaintPanel, but nothing here should assume that stays true forever.
function sanitizeMaskAssetFilename(id: string): string {
  const safe = id.replace(/[^a-zA-Z0-9_-]/g, "_").replace(/^\.+/, "");
  return `${safe || "mask"}.png`;
}

// Video temporal inpaint (MiniMax-H3 fl2va/ref2va): multipart POST
// /generate/inpaint/video with an uploaded `video` clip. Same explicit-append
// shape as generateOutpaintVideo, matching the Form parameter names of
// routes.py's generate_inpaint_video 1:1. No clip-length field exists: the
// output is as long as the trimmed input.
//
// `references` is optional and carries the SAME field set as
// generateRef2Vid's (reference_images/reference_videos/reference_video_audios
// /reference_audios) -- this endpoint mirrors /generate/ref2vid's reference
// surface, not video outpaint's images-only one. fl2va and an unidentified
// variant refuse a reference-carrying request server-side; the caller is
// expected to have gated the UI on a confirmed ref2va checkpoint before
// populating `references` (see InpaintPanel's isH3Ref2VaInpaint).
export const generateInpaintVideo = async (
  params: InpaintVideoParams,
  video: File | string,
  spatialMaskParts?: Array<{ id: string; file: File }>,
  references?: MiniMaxH3References,
) => {
  const formData = new FormData();

  const manifest = params.spatial_mask_manifest?.trim();
  if (manifest) {
    if (!Array.isArray(spatialMaskParts) || spatialMaskParts.length === 0) {
      throw new Error("A spatial mask manifest requires at least one PNG mask asset.");
    }
    try {
      const parsed = JSON.parse(manifest) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Spatial mask manifest must be a JSON object.");
      }
    } catch (error) {
      throw new Error(`Spatial mask manifest is invalid: ${error instanceof Error ? error.message : "invalid JSON"}`);
    }
    const ids = new Set<string>();
    for (const part of spatialMaskParts) {
      if (!part || typeof part.id !== "string" || part.id.trim() === "") {
        throw new Error("Every spatial mask asset needs a non-empty id.");
      }
      if (ids.has(part.id)) throw new Error(`Duplicate spatial mask asset id: ${part.id}`);
      ids.add(part.id);
      if (typeof File === "undefined" || !(part.file instanceof File) || part.file.size <= 0) {
        throw new Error(`Spatial mask asset ${part.id} is not a valid file.`);
      }
      if (part.file.type && part.file.type !== "image/png") {
        throw new Error(`Spatial mask asset ${part.id} must be a PNG file.`);
      }
    }
    formData.append("spatial_mask_manifest", manifest);
    for (const part of spatialMaskParts) {
      formData.append("spatial_mask_ids", part.id);
      formData.append("spatial_mask_files", part.file, sanitizeMaskAssetFilename(part.id));
    }
  } else if (spatialMaskParts !== undefined && spatialMaskParts.length > 0) {
    throw new Error("Spatial mask assets cannot be uploaded without a manifest.");
  }

  if (typeof video === "string") {
    const response = await fetch(video);
    const blob = await response.blob();
    formData.append("video", blob, "input.mp4");
  } else {
    formData.append("video", video);
  }

  formData.append("prompt", params.prompt);
  formData.append("negative_prompt", params.negative_prompt || "");
  formData.append("width", String(params.width ?? 768));
  formData.append("height", String(params.height ?? 512));
  formData.append("frame_rate", String(params.frame_rate ?? 24.0));
  formData.append("num_inference_steps", String(params.num_inference_steps ?? 8));
  formData.append("guidance_scale", String(params.guidance_scale ?? 1.0));
  formData.append("seed", String(params.seed ?? -1));
  formData.append("num_videos_per_prompt", String(params.num_videos_per_prompt ?? 1));
  formData.append("max_sequence_length", String(params.max_sequence_length ?? 1024));
  formData.append("audio_enable", String(params.audio_enable ?? true));

  // The range, required by the route.
  formData.append("regenerate_start_frame", String(params.regenerate_start_frame));
  formData.append("regenerate_end_frame", String(params.regenerate_end_frame));
  formData.append("input_trim_start_frames", String(params.input_trim_start_frames ?? 0));
  formData.append("input_trim_end_frames", String(params.input_trim_end_frames ?? 0));
  // Sent ONLY when the caller has a value: the backend field is a sentinel
  // whose default is per-architecture, so a hardcoded fallback here would pin
  // the base value and defeat the overlay (see generateOutpaintVideo).
  if (params.inpaint_video_audio_mode) {
    formData.append("inpaint_video_audio_mode", params.inpaint_video_audio_mode);
  }

  formData.append("attention_type", resolveGlobalAttentionType(params.attention_type));

  formData.append("blocks_to_swap", String(params.blocks_to_swap ?? 0));
  formData.append("fuse_output_proj", String(params.fuse_output_proj ?? false));
  formData.append("fbcache_enable", String(params.fbcache_enable ?? false));
  formData.append("fbcache_threshold", String(params.fbcache_threshold ?? 0.12));
  formData.append("fbcache_warmup_steps", String(params.fbcache_warmup_steps ?? 1));
  formData.append("spectrum_enable", String(params.spectrum_enable ?? false));
  formData.append("spectrum_w", String(params.spectrum_w ?? 0.5));
  formData.append("spectrum_w_decay", String(params.spectrum_w_decay ?? 0.0));
  formData.append("spectrum_delta_cap", String(params.spectrum_delta_cap ?? 0.0));
  formData.append("spectrum_m", String(params.spectrum_m ?? 4));
  formData.append("spectrum_lam", String(params.spectrum_lam ?? 0.1));
  formData.append("spectrum_warmup_steps", String(params.spectrum_warmup_steps ?? 3));
  formData.append("spectrum_window_size", String(params.spectrum_window_size ?? 4));
  formData.append("spectrum_flex_window", String(params.spectrum_flex_window ?? 0.75));
  formData.append("spectrum_tail", String(params.spectrum_tail ?? 0.12));
  formData.append("spectrum_max_cache", String(params.spectrum_max_cache ?? 0));

  if (params.vae_path) {
    formData.append("vae_path", params.vae_path);
  }
  if (params.text_encoder_path) {
    formData.append("text_encoder_path", params.text_encoder_path);
  }
  if (params.unet_quantization && params.unet_quantization !== "none") {
    formData.append("unet_quantization", params.unet_quantization);
  }
  if (params.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", params.quantized_gemm_mode);
  }

  formData.append("video_lossless", String(params.video_lossless ?? false));
  formData.append("loras", JSON.stringify(params.loras || []));

  // The reference files, in upload order -- same convention as
  // generateRef2Vid's own append (order is semantic: it fixes the
  // <Picture i>/<Video k>/<Audio j> labels and the packed sequence's shared
  // rotary clock). Only appended when the caller actually has references, so
  // a plain fl2va temporal-inpaint request is byte-identical to what it was
  // before this field set existed.
  if (references) {
    formData.append("reference_image_size", params.reference_image_size ?? "max");
    (references.images || []).forEach((file) => formData.append("reference_images", file));
    (references.videos || []).forEach((file) => formData.append("reference_videos", file));
    if ((references.videoAudios || []).some((file) => file)) {
      (references.videos || []).forEach((_video, index) => {
        const soundtrack = (references.videoAudios || [])[index];
        formData.append(
          "reference_video_audios",
          soundtrack ?? new File([], ""),
        );
      });
    }
    (references.audios || []).forEach((file) => formData.append("reference_audios", file));
  }

  const response = await postGenerationRequest("/generate/inpaint/video", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

// Audio temporal outpaint (ACE-Step 1.5 extend): multipart POST
// /generate/outpaint/audio with an uploaded `reference_audio` clip. Clone of
// generateAud2Aud's FormData sender (CLAUDE.md param-threading) plus the
// placement fields; every OutpaintAudioParams field is appended explicitly,
// matching the Form parameter names of routes.py's generate_outpaint_audio 1:1.
export const generateOutpaintAudio = async (params: OutpaintAudioParams, referenceAudio: File | string) => {
  const formData = new FormData();

  // Handle both File objects and data URLs (mirrors generateAud2Aud's `reference_audio` handling).
  if (typeof referenceAudio === "string") {
    const response = await fetch(referenceAudio);
    const blob = await response.blob();
    formData.append("reference_audio", blob, "reference.wav");
  } else {
    formData.append("reference_audio", referenceAudio);
  }

  formData.append("prompt", params.prompt);
  formData.append("lyrics", params.lyrics || "");
  formData.append("seed", String(params.seed ?? -1));
  formData.append("inference_steps", String(params.inference_steps ?? 8));
  formData.append("guidance_scale", String(params.guidance_scale ?? 1.0));
  formData.append("shift", String(params.shift ?? 3.0));
  formData.append("vocal_language", params.vocal_language ?? "en");
  formData.append("loras", JSON.stringify(params.loras || []));

  // Placement (ACE-Step only), all in seconds.
  formData.append("total_duration", String(params.total_duration ?? 60.0));
  formData.append("input_offset_sec", String(params.input_offset_sec ?? 0.0));
  formData.append("input_trim_start_sec", String(params.input_trim_start_sec ?? 0.0));
  formData.append("input_trim_end_sec", String(params.input_trim_end_sec ?? 0.0));

  // MiniMax Music 3 extend only. Appended ONLY when set, mirroring the
  // quantization fields' "unset leaves the backend default untouched"
  // convention below -- and deliberately NOT `?? "extend_forward"` for
  // `placement`: a key the client always sends lands in FastAPI's bound
  // value regardless of what the user picked, defeating the backend's own
  // "no default assumed" refusal for an actually-omitted placement (the same
  // `?? null` mistake CLAUDE.md's per-arch-default guidance calls out).
  if (params.placement) {
    formData.append("placement", params.placement);
  }
  if (params.extend_duration_sec !== undefined) {
    formData.append("extend_duration_sec", String(params.extend_duration_sec));
  }
  if (params.num_inference_steps !== undefined) {
    formData.append("num_inference_steps", String(params.num_inference_steps));
  }
  if (params.flow_guidance_scale !== undefined) {
    formData.append("flow_guidance_scale", String(params.flow_guidance_scale));
  }

  // Weight-only quantization (both axes). Appended only when set, so an unset
  // field leaves the backend default (and the process GEMM flags) untouched.
  if (params.unet_quantization && params.unet_quantization !== "none") {
    formData.append("unet_quantization", params.unet_quantization);
  }
  if (params.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", params.quantized_gemm_mode);
  }

  const response = await postGenerationRequest("/generate/outpaint/audio", formData, {
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

// Full gallery row response, plus the total number of rows that shared
// `hash` (the returned row is the oldest of those, not the only one).
export interface GeneratedImageByHash extends GeneratedImage {
  match_count: number;
}

// Resolves a hash (image_hash / source_image_hash / source_audio_hash /
// ControlNet reference hash) to the gallery row that produced it, when that
// row is not present in the currently loaded gallery page. Throws (axios
// 404) when no row matches -- callers should catch this and fall back to
// their own "not found" messaging.
export const getImageByHash = async (hash: string): Promise<GeneratedImageByHash> => {
  const response = await api.get(`/images/by-hash/${encodeURIComponent(hash)}`);
  return response.data;
};

// deleteFiles=true (default) removes the DB row and every file it owns;
// deleteFiles=false removes only the DB row and leaves files on disk.
export const deleteImage = async (id: number, deleteFiles: boolean = true) => {
  const response = await api.delete(`/images/${id}`, {
    params: { delete_files: deleteFiles },
  });
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

export const getCurrentModelComponents = async (): Promise<CurrentComponentsResponse> => {
  const response = await api.get("/models/current/components");
  return response.data;
};

export const getCurrentModelComponentCandidates = async (slot: ComponentSlotId) => {
  const response = await api.get("/models/current/components/candidates", { params: { slot } });
  return response.data as {
    model_revision: number;
    component_revision: number;
    slot: ComponentSlotId;
    candidates: ComponentCandidate[];
  };
};

// `projectionPath` (MiniMax-H3 text encoders only) names the projection to pair
// with the new encoder; the backend refuses to pick when several declare that
// encoder's width.
export const switchCurrentModelComponent = async (
  slot: ComponentSlotId,
  candidateId: string,
  expectedModelRevision: number,
  expectedComponentRevision: number,
  projectionPath?: string | null,
) => {
  const response = await api.post("/models/current/components/switch", {
    slot,
    candidate_id: candidateId,
    expected_model_revision: expectedModelRevision,
    expected_component_revision: expectedComponentRevision,
    projection_path: projectionPath || null,
  });
  return response.data as {
    success: boolean;
    operation: Record<string, unknown>;
    components: CurrentComponentsResponse;
  };
};

// `force`: reload even when this model is already the loaded one. Without it the
// backend early-returns, so nothing per-session is reset — which is what makes
// "load the model again" the working recovery for the one-way in-place INT8
// conversion (unet_quantization="int8" on anima/krea2/flux2/ideogram4).
//
// `textEncoderFile`/`clipProjectionFile` (MiniMax-H3 only, absolute paths): omit
// both to get the loader's preference order and its projection auto-discovery.
export const loadModel = async (
  sourceType: string,
  source: string,
  revision?: string,
  force?: boolean,
  textEncoderFile?: string | null,
  clipProjectionFile?: string | null,
  hybrid?: MiniMaxH3HybridLoadRequest | null
) => {
  const formData = new FormData();
  formData.append("source_type", sourceType);
  formData.append("source", source);
  if (revision) {
    formData.append("revision", revision);
  }
  if (force) {
    formData.append("force", "true");
  }
  if (textEncoderFile) {
    formData.append("text_encoder_file", textEncoderFile);
  }
  if (clipProjectionFile) {
    formData.append("clip_projection_file", clipProjectionFile);
  }
  // Nothing is sent without an overlay: the backend then takes the load path it
  // always took. The recipe fields ride along only with one, so a stale range
  // in a caller's state cannot reach a base-only load.
  if (hybrid?.overlay_file) {
    formData.append("overlay_file", hybrid.overlay_file);
    if (hybrid.preset) {
      formData.append("hybrid_preset", hybrid.preset);
    }
    if (hybrid.block_range_start !== undefined) {
      formData.append("hybrid_block_range_start", String(hybrid.block_range_start));
    }
    if (hybrid.block_range_end !== undefined) {
      formData.append("hybrid_block_range_end", String(hybrid.block_range_end));
    }
    if (hybrid.final_adaln_from_overlay !== undefined) {
      formData.append(
        "hybrid_final_adaln_from_overlay",
        hybrid.final_adaln_from_overlay ? "true" : "false"
      );
    }
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

export const getLoras = async (): Promise<{ loras: Array<LoRAListEntry> }> => {
  const response = await api.get("/loras");
  return response.data;
};

export const getLoraInfo = async (loraName: string): Promise<LoRAInfo> => {
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

// --- Process-level quantized GEMM paths -------------------------------------
// These are per-process modes, NOT generation parameters: they are not stored
// with an image and not part of GenerationParams. The path a given image
// actually ran is recorded separately in GeneratedImage.fp8_gemm.

export interface Fp8ScaledMmState {
  enabled: boolean;
  /**
   * Where the current process value came from. "generation" means a generation
   * request carried an explicit `quantized_gemm_mode` and forced the flag for
   * itself; it is distinct from "api" so a manual flip in Settings can be told
   * apart from one a queued generation made.
   */
  origin: "default" | "env" | "api" | "generation";
  /**
   * Probe result per "<device>/<activation dtype>" key. null means no
   * torch._scaled_mm variant worked for that key, so those layers run the
   * dequantized matmul even while `enabled` is true. Empty until an FP8 Linear
   * forward has reached the probe in this process.
   */
  resolved_modes: Record<string, "rowwise_bias" | "rowwise" | "tensorwise" | null>;
}

export interface Int8MmState {
  enabled: boolean;
  /** See Fp8ScaledMmState.origin. */
  origin: "default" | "env" | "api" | "generation";
  /**
   * Probe result per device. "int_mm" means torch._int_mm reproduced an int32
   * reference product; null means it was unusable, so those layers run the
   * dequantized matmul even while `enabled` is true. Empty until an INT8 Linear
   * forward has reached the probe in this process.
   */
  resolved_modes: Record<string, "int_mm" | null>;
}

export const getFp8ScaledMm = async (): Promise<Fp8ScaledMmState> => {
  const response = await api.get("/system/fp8-scaled-mm");
  return response.data;
};

/** Throws on 409 while a generation or training run is active. */
export const setFp8ScaledMm = async (enabled: boolean): Promise<Fp8ScaledMmState> => {
  const response = await api.post("/system/fp8-scaled-mm", { enabled });
  return response.data;
};

export const getInt8Mm = async (): Promise<Int8MmState> => {
  const response = await api.get("/system/int8-mm");
  return response.data;
};

/** Throws on 409 while a generation or training run is active. */
export const setInt8Mm = async (enabled: boolean): Promise<Int8MmState> => {
  const response = await api.post("/system/int8-mm", { enabled });
  return response.data;
};

/** Per-format census of the loaded transformer's Linear modules. */
export interface QuantizedExportInventory {
  /** Int8Linear modules (int8 weight + per-row float32 scale). */
  int8: number;
  /** Fp8Linear modules (float8_e4m3fn weight + per-row float32 scale). */
  e4m3: number;
  /** Unquantized nn.Linear modules. */
  plain: number;
  total: number;
}

/**
 * Export job document. Before the first export of a backend process this is
 * just `{ state: "idle" }`; every other field appears once a job has started.
 */
export interface QuantizedExportJob {
  job_id?: string;
  state: "idle" | "running" | "completed" | "failed";
  arch?: string | null;
  output_path?: string;
  /** The `.safetensors`, or the `.safetensors.index.json` of a sharded export. */
  written_path?: string | null;
  processed?: number;
  total?: number;
  message?: string;
  error?: string | null;
  result?: Record<string, any> | null;
  started_at?: number;
  finished_at?: number | null;
}

export interface QuantizedExportStatus {
  /** True when the loaded transformer owns quantized Linear modules. */
  exportable: boolean;
  /** Why it is not exportable; null when it is. */
  reason: string | null;
  arch: string | null;
  source: string | null;
  inventory: QuantizedExportInventory;
  /**
   * True when this session converted the model in place, so a per-layer audit
   * document exists and is written next to the export.
   */
  has_runtime_audit: boolean;
  suggested_path: string | null;
  job: QuantizedExportJob;
}

export const getQuantizedExportStatus = async (): Promise<QuantizedExportStatus> => {
  const response = await api.get("/models/export-quantized");
  return response.data;
};

/**
 * Start the export job. Throws on 409 while a generation, a training run or
 * another export is in flight, and on 400 for an unexportable model or an
 * invalid destination.
 */
export const startQuantizedExport = async (params: {
  output_path: string;
  link_siblings?: boolean;
  overwrite?: boolean;
}): Promise<QuantizedExportJob> => {
  const response = await api.post("/models/export-quantized", params);
  return response.data;
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

// Video mask timeline preview: rasterizes a spatial mask manifest (the exact
// wire format /generate/inpaint/video's spatial_mask_manifest/_ids/_files use)
// for an explicit list of frames, without loading a model or generating
// anything. See openapi.yaml's /video-mask/preview and
// backend/core/inference/video_mask_preview.py for what this actually does.
export interface VideoMaskPreviewFrame {
  frame: number;
  x_offset: number;
}

export interface VideoMaskPreviewResult {
  canvas_width: number;
  canvas_height: number;
  frame_width: number;
  frame_height: number;
  frames: VideoMaskPreviewFrame[];
  /**
   * Always includes a trailing summary entry (e.g. "3 more warning(s) not
   * shown...") when the backend's `X-Mask-Preview-Meta` header had to drop
   * some warnings to stay under its byte budget -- see
   * `warnings_truncated` in `openapi.yaml`'s `VideoMaskPreviewResponse`.
   * Callers that just render every entry in this array (the common case)
   * therefore never need to read `warnings_truncated` themselves.
   */
  warnings: string[];
  /** Present only when `warnings` above was capped server-side; already folded into the trailing summary entry in `warnings`. */
  warnings_truncated?: number;
  /**
   * A `blob:` object URL for the sprite strip (the route returns raw
   * `image/png` bytes, not base64-in-JSON). The caller is responsible for
   * `URL.revokeObjectURL`-ing the PREVIOUS value once a new result replaces
   * it -- see `useMaskPreview.ts`, which owns that lifecycle.
   */
  strip_png: string;
}

/**
 * Thrown by `previewVideoMask` when the backend answers 409 because one or
 * more `spatial_mask_refs` entries could not be resolved (e.g. the temp file
 * was swept). `unresolvedRefIds` are manifest mask IDs, not the refs
 * themselves -- a caller retries by re-sending the SAME request with just
 * those IDs' assets uploaded as bytes instead of by ref.
 */
export class VideoMaskRefUnresolvedError extends Error {
  code = "SUSHIUI_VIDEO_MASK_REF_UNRESOLVED";
  unresolvedRefIds: string[];
  constructor(unresolvedRefIds: string[]) {
    super(`Video mask ref(s) could not be resolved: ${unresolvedRefIds.join(", ")}`);
    this.name = "VideoMaskRefUnresolvedError";
    this.unresolvedRefIds = unresolvedRefIds;
  }
}

// `responseType: "blob"` means axios hands back an error response's JSON
// body as an unparsed Blob too (it only auto-parses the success path) --
// this reads it back out so the 409 case above can be told apart from any
// other failure.
async function readBlobErrorBody(data: unknown): Promise<{ error?: string; detail?: string } | null> {
  if (typeof Blob === "undefined" || !(data instanceof Blob)) return null;
  try {
    return JSON.parse(await data.text());
  } catch {
    return null;
  }
}

export const previewVideoMask = async (
  manifestJson: string,
  maskParts: Array<{ id: string; file: File }>,
  maskRefParts: Array<{ id: string; ref: string }>,
  frames: number[],
  maxSize = 256,
): Promise<VideoMaskPreviewResult> => {
  const formData = new FormData();
  formData.append("spatial_mask_manifest", manifestJson);
  for (const part of maskParts) {
    formData.append("spatial_mask_ids", part.id);
    formData.append("spatial_mask_files", part.file, sanitizeMaskAssetFilename(part.id));
  }
  for (const part of maskRefParts) {
    formData.append("spatial_mask_ref_ids", part.id);
    formData.append("spatial_mask_refs", part.ref);
  }
  for (const frame of frames) {
    formData.append("frames", String(frame));
  }
  formData.append("max_size", String(maxSize));

  try {
    const response = await api.post("/video-mask/preview", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      responseType: "blob",
    });
    const metaHeader = response.headers["x-mask-preview-meta"] as string | undefined;
    const meta = metaHeader ? JSON.parse(metaHeader) : {};
    const warnings: string[] = Array.isArray(meta.warnings) ? meta.warnings : [];
    if (typeof meta.warnings_truncated === "number" && meta.warnings_truncated > 0) {
      warnings.push(
        `${meta.warnings_truncated} more warning(s) not shown (the server caps how many fit in one response).`,
      );
    }
    return {
      ...meta,
      warnings,
      strip_png: URL.createObjectURL(response.data as Blob),
    } as VideoMaskPreviewResult;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 409) {
      const body = await readBlobErrorBody(err.response.data);
      if (body?.error === "Video mask ref unresolved" && typeof body.detail === "string") {
        try {
          const parsedDetail = JSON.parse(body.detail) as { unresolved_ref_ids?: unknown };
          if (Array.isArray(parsedDetail.unresolved_ref_ids)) {
            throw new VideoMaskRefUnresolvedError(parsedDetail.unresolved_ref_ids as string[]);
          }
        } catch (parseErr) {
          if (parseErr instanceof VideoMaskRefUnresolvedError) throw parseErr;
          // Malformed 409 body: fall through and rethrow the original error.
        }
      }
    }
    throw err;
  }
};

// MiniMax-H3 Prompt Assist API
export type H3PromptMode = "t2va" | "i2va" | "fl2va" | "l2va" | "ref2va";
export type PromptAssistProvider = "lm_studio" | "ollama";

export interface PromptAssistDefaults {
  provider: PromptAssistProvider;
  lm_studio_base_url: string;
  ollama_base_url: string;
  temperature: number;
  top_p: number;
  max_output_tokens: number;
  context_length: number;
  timeout_seconds: number;
  auto_on_generate: boolean;
}

export interface PromptAssistReference {
  token: string;
  kind: "picture" | "video" | "audio" | "subject";
  role: string;
  description?: string;
}

export interface PromptAssistSettings {
  provider: PromptAssistProvider;
  base_url: string;
  model: string;
  api_key?: string;
  temperature: number;
  top_p: number;
  max_output_tokens: number;
  context_length: number;
  timeout_seconds: number;
  auto_on_generate: boolean;
}

export interface PromptAssistTransformRequest extends Omit<PromptAssistSettings, "auto_on_generate"> {
  prompt: string;
  mode: H3PromptMode;
  duration_seconds: number;
  references: PromptAssistReference[];
  // What to change THIS TIME (e.g. "make the drop harder"), used only when
  // `revise` is true. Sent as its own field, never appended into `prompt`
  // -- folding an instruction into the prompt text lets the LLM read it as
  // more content to describe instead of a directive to apply.
  instruction?: string;
  // False (default): `prompt` is freeform intent to expand, exactly as
  // this has always behaved. True: `prompt` is the CURRENT, already-
  // structured prompt -- the base text to preserve -- and `instruction`
  // (required when true) is the edit to apply to it.
  revise?: boolean;
  force_refresh?: boolean;
}

export interface PromptAssistResponse {
  prompt: string;
  warnings: string[];
  valid: boolean;
  cached?: boolean;
  cache_key?: string;
  provider?: string;
  model?: string;
  revise?: boolean;
  // A unified line diff between the revise-mode base text and `prompt`,
  // present only when the request had `revise` true. Lets a caller show
  // the user whether a revise made a targeted edit or rewrote the whole
  // piece -- "only the named parts changed" is not machine-checkable, a
  // diff against the base is.
  diff_summary?: string | null;
}

export interface PromptAssistModel {
  id: string;
  name: string;
  loaded: boolean;
  size_bytes?: number | null;
}

export const getPromptAssistDefaults = async (): Promise<PromptAssistDefaults> => {
  const response = await api.get("/schema/prompt-assist-defaults");
  return response.data;
};

export const listPromptAssistModels = async (
  provider: PromptAssistProvider,
  base_url: string,
  api_key = "",
): Promise<PromptAssistModel[]> => {
  const response = await api.post("/prompt-assist/models", { provider, base_url, api_key });
  return response.data.models;
};

export const createH3PromptTemplate = async (
  prompt: string,
  mode: H3PromptMode,
  duration_seconds: number,
): Promise<PromptAssistResponse> => {
  const response = await api.post("/prompt-assist/template", { prompt, mode, duration_seconds });
  return response.data;
};

export const transformH3Prompt = async (
  request: PromptAssistTransformRequest,
): Promise<PromptAssistResponse> => {
  const response = await api.post("/prompt-assist/transform", request, {
    timeout: Math.max(1000, request.timeout_seconds * 1000 + 5000),
  });
  return response.data;
};

export const clearPromptAssistCache = async (): Promise<number> => {
  const response = await api.post("/prompt-assist/cache/clear");
  return response.data.deleted;
};

// MiniMax Music 3 Caption Rewriter API
//
// Sibling of the MiniMax-H3 prompt assist API above: same provider/model
// listing endpoint and settings shape, different transform contract (no
// mode/duration/references — a music caption has none of those concepts).
//
// Deliberately no lm_studio_base_url/ollama_base_url here: the server's
// `_prompt_assist_base_url()` resolves an empty base_url from H3's
// PROMPT_ASSIST_DEFAULTS regardless of which rewriter is calling it (one
// local LM Studio/Ollama server serves both), so a second, music-only copy
// of those two keys would only be able to drift from what the server
// actually uses. Read them from `getPromptAssistDefaults()` instead.
export interface MusicPromptAssistDefaults {
  provider: PromptAssistProvider;
  temperature: number;
  top_p: number;
  max_output_tokens: number;
  context_length: number;
  timeout_seconds: number;
}

export interface MusicPromptAssistSettings {
  provider: PromptAssistProvider;
  base_url: string;
  model: string;
  api_key?: string;
  temperature: number;
  top_p: number;
  max_output_tokens: number;
  context_length: number;
  timeout_seconds: number;
}

export interface MusicPromptAssistTransformRequest extends MusicPromptAssistSettings {
  // A short caption to expand (default), or -- when `revise` is true --
  // the CURRENT, already-expanded Structured Caption, treated as the base
  // text to preserve.
  caption: string;
  lyrics?: string;
  // Standing rules that hold for the piece regardless of pass. Distinct
  // from `instruction`, which is what to change this time.
  constraints?: string;
  // What to change THIS TIME, used only when `revise` is true. Sent as its
  // own field, never appended into `caption` -- see the same rationale on
  // `PromptAssistTransformRequest.instruction`.
  instruction?: string;
  // False (default): `caption` is a short caption to expand. True:
  // `caption` is the CURRENT Structured Caption -- the base text to
  // preserve -- and `instruction` (required when true) is the edit to
  // apply to it.
  revise?: boolean;
  force_refresh?: boolean;
}

export const getMusicPromptAssistDefaults = async (): Promise<MusicPromptAssistDefaults> => {
  const response = await api.get("/schema/prompt-assist-music-defaults");
  return response.data;
};

export const transformMusic3Caption = async (
  request: MusicPromptAssistTransformRequest,
): Promise<PromptAssistResponse> => {
  const response = await api.post("/prompt-assist/music/transform", request, {
    timeout: Math.max(1000, request.timeout_seconds * 1000 + 5000),
  });
  return response.data;
};

export const clearMusicPromptAssistCache = async (): Promise<number> => {
  const response = await api.post("/prompt-assist/music/cache/clear");
  return response.data.deleted;
};

// MiniMax Music 3 Lyrics Assistant API
//
// Sibling of the caption rewriter above, not an extension of it: its own
// cache, its own defaults endpoint. Three modes (design doc, "Three
// user-selected modes"):
//   - "format" — deterministic, no LLM, no network settings needed.
//   - "structure" / "complete" — LLM-driven, share the caption rewriter's
//     provider/model shape.
export type MusicLyricsAssistMode = "format" | "structure" | "complete";

export interface MusicLyricsAssistDefaults {
  mode: MusicLyricsAssistMode;
  provider: PromptAssistProvider;
  temperature: number;
  top_p: number;
  max_output_tokens: number;
  context_length: number;
  timeout_seconds: number;
}

export interface MusicLyricsAssistSettings {
  provider: PromptAssistProvider;
  base_url: string;
  model: string;
  api_key?: string;
  temperature: number;
  top_p: number;
  max_output_tokens: number;
  context_length: number;
  timeout_seconds: number;
}

export interface MusicLyricsAssistTransformRequest extends MusicLyricsAssistSettings {
  mode: "structure" | "complete";
  // Non-revise "complete" mode: optional creative direction. When `revise`
  // is true (either mode): optional additional direction alongside
  // `instruction`.
  theme?: string;
  // Non-revise "complete" mode: partial lyrics to preserve and complete
  // around. Ignored in non-revise "structure" mode. When `revise` is true
  // (either mode): the CURRENT lyrics or structure/tag map, REQUIRED --
  // the base text to preserve, not a fragment to complete around.
  lyrics?: string;
  // Standing rules that hold for the piece regardless of pass. Distinct
  // from `instruction`, which is what to change this time.
  constraints?: string;
  // What to change THIS TIME, required and used only when `revise` is
  // true. Sent as its own field, never appended into `lyrics` -- see the
  // same rationale on `PromptAssistTransformRequest.instruction`.
  instruction?: string;
  // False (default): behaves exactly as this has always behaved for
  // `mode`. True: `lyrics` is the base text to preserve and `instruction`
  // is the edit to apply -- for "structure", an edit to the tag sequence
  // itself; for "complete", an edit to the written words.
  revise?: boolean;
  force_refresh?: boolean;
}

export interface MusicLyricsAssistResponse {
  lyrics: string;
  warnings: string[];
  valid: boolean;
  cached?: boolean;
  cache_key?: string;
  provider?: string;
  model?: string;
  mode?: string;
  revise?: boolean;
  // A unified line diff between the revise-mode base `lyrics` and the
  // returned `lyrics`, present only when the request had `revise` true.
  diff_summary?: string | null;
}

export interface MusicLyricsFormatResponse {
  lyrics: string;
  warnings: string[];
}

export const getMusicLyricsAssistDefaults = async (): Promise<MusicLyricsAssistDefaults> => {
  const response = await api.get("/schema/prompt-assist-music-lyrics-defaults");
  return response.data;
};

export const formatMusic3Lyrics = async (lyrics: string): Promise<MusicLyricsFormatResponse> => {
  const response = await api.post("/prompt-assist/music/lyrics/format", { lyrics });
  return response.data;
};

export const transformMusic3Lyrics = async (
  request: MusicLyricsAssistTransformRequest,
): Promise<MusicLyricsAssistResponse> => {
  const response = await api.post("/prompt-assist/music/lyrics/transform", request, {
    timeout: Math.max(1000, request.timeout_seconds * 1000 + 5000),
  });
  return response.data;
};

export const clearMusicLyricsAssistCache = async (): Promise<number> => {
  const response = await api.post("/prompt-assist/music/lyrics/cache/clear");
  return response.data.deleted;
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
  index: number; // physical index from nvidia-smi; ignores CUDA_VISIBLE_DEVICES
  cuda_index: number | null; // index torch addresses it by = the value to send as gpu_index; null when not visible to torch
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
  // For item_type="video"/"audio": public URL of the first-frame poster /
  // waveform thumbnail generated at scan time (served via the /thumbnails
  // static mount).
  thumbnail_url?: string | null;
  // For item_type="video": per-clip metadata (fps, num_frames, duration,
  // width, height, codec) captured at scan time via ffprobe.
  video_meta?: {
    video_path?: string;
    fps?: number;
    num_frames?: number;
    duration?: number;
    width?: number;
    height?: number;
    codec?: string | null;
  } | null;
  // For item_type="audio": per-clip metadata captured at scan time via
  // soundfile/ffprobe (mirrors video_meta above).
  audio_meta?: {
    audio_path?: string;
    sample_rate?: number;
    duration?: number;
    channels?: number;
  } | null;
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

/** One structured notice from a training run (a setting overridden or ignored).
 *  Same shape as the `training_log` WebSocket message minus the envelope —
 *  see backend/api/WS_PROTOCOL.md. */
export interface TrainingLogEvent {
  level: "info" | "warning" | "error";
  code?: string | null;
  message: string;
}

export interface TrainingRun {
  id: number;
  dataset_id: number;
  run_id: string;  // UUID
  run_name: string;
  training_method: "lora" | "relora" | "full_finetune" | "controlnet" | "vae_decoder";
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
  /** Detail payload only; absent on the summary rows returned by the run list. */
  warnings?: TrainingLogEvent[];
  /**
   * PUT /training/runs/{id} only: train-section keys carried over because the
   * request model has no field for them (config-channel switches set by
   * hand-editing the YAML).
   */
  preserved_config_keys?: string[];
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
  training_method: "lora" | "relora" | "full_finetune" | "controlnet" | "vae_decoder";
  sensenova_mot_phase_eviction?: boolean;
  // SenseNova full fine-tune only: splits the backward at the prefix KV cache
  // so a TRAINED understanding half can still be evicted. Refused before the
  // model loads unless train_text_encoder and sensenova_mot_phase_eviction are
  // both set; and those two TOGETHER are refused without it.
  sensenova_four_phase_eviction?: boolean;
  // On top of the split, at multi_noise_timesteps > 1: one boundary cut per
  // window instead of per iteration. Changes what the understanding half trains
  // on (one update per window, not N) -- see openapi.yaml.
  sensenova_four_phase_shared_prefix?: boolean;
  sensenova_four_phase_grad_reduction?: "sum" | "mean";
  // SenseNova full fine-tune only: on-disk format of the saved model.
  // "mixed" (default) | "bf16" | "int8". See openapi.yaml for the measured
  // sizes and the int8 requantization loss census.
  sensenova_full_finetune_save_format?: string;  // "mixed" (default) | "bf16" | "int8"
  // SenseNova only, applies to the in-training sample only (not train_step):
  // streams each layer's prefix KV cache from pinned host memory through a
  // 2-slot GPU ring instead of holding the full KV cache resident during the
  // sample's denoise loop. Independent of the phase-eviction flags above.
  sensenova_sample_kv_cache_streaming?: boolean;
  // SenseNova MoT phase eviction only: stage the evicted half to pageable host
  // memory instead of pinned, trading the sticky pinned high-water for host
  // RAM the OS can reclaim, at an unmeasured transfer-time cost. Refused
  // without sensenova_mot_phase_eviction.
  sensenova_mot_pageable_staging?: boolean;
  // SenseNova MoT phase eviction only: run a swap's outgoing and incoming legs
  // concurrently on two CUDA streams instead of back to back. Requires
  // sensenova_mot_phase_eviction; refused together with pageable staging.
  sensenova_mot_overlap_transfer?: boolean;
  base_model_path: string;
  gpu_index?: number | null;  // Physical GPU index to run this training run on; null = backend default device
  // Decoder-only VAE fine-tune options (training_method "vae_decoder" only).
  // Nested so the backend can tell "the caller asked for this" from "a diffusion
  // default happens to have this value": generate_vae_config takes a flat field
  // into account ONLY when the caller explicitly sent it, and a nested
  // vae_config entry always wins. Ignored for every other training_method.
  vae_config?: VaeTrainingConfig | null;
  total_steps?: number;  // Mutually exclusive with epochs
  epochs?: number;  // Mutually exclusive with total_steps
  batch_size?: number;
  gradient_accumulation_steps?: number;
  max_grad_norm?: number;
  learning_rate?: number;
  lr_scheduler?: string;  // Includes "plateau_cosine_floor" (warmup -> plateau -> cosine decay to floor)
  lr_warmup_steps?: number;
  lr_decay_start_ratio?: number;  // plateau_cosine_floor only: fraction of total steps where decay begins (default 0.85)
  lr_floor_ratio?: number;  // plateau_cosine_floor only: floor as a fraction of base LR (default 0.25)
  rewarmup_on_optimizer_reset?: boolean;  // Re-apply lr_warmup_steps when a resume gets a fresh optimizer state (default true)
  use_ema?: boolean;  // Weight EMA (opt-in, default off); saves a separate, loadable "_ema" checkpoint alongside the normal one
  ema_decay?: number;  // EMA decay factor (default 0.9999)
  ema_update_every?: number;  // Apply the EMA update every N optimizer steps (default 1 = every step)
  ema_device?: string;  // Where the EMA shadow lives: "cpu" (default) or "cuda"
  optimizer?: string;
  lora_rank?: number;
  lora_alpha?: number;
  network_type?: string;
  save_every?: number;
  save_every_unit?: string;
  max_step_saves_to_keep?: number | null;
  // Optimizer .pt sidecars to keep (0 = keep all). Pruned independently of the
  // weights; only the newest is ever resumed from.
  max_optimizer_saves_to_keep?: number;
  sample_every?: number;
  sample_prompts?: SamplePrompt[];
  resume_from_checkpoint?: string | null;
  sample_width?: number;
  sample_height?: number;
  sample_steps?: number;
  sample_cfg_scale?: number;
  sample_sampler?: string;
  sample_schedule_type?: string;
  sample_cfg_schedule_type?: string;
  sample_cfg_schedule_min?: number;
  sample_cfg_schedule_max?: number | null;
  sample_cfg_schedule_power?: number;
  sample_cfg_rescale_snr_alpha?: number;
  sample_dynamic_threshold_percentile?: number;
  sample_dynamic_threshold_mimic_scale?: number;
  sample_nag_enable?: boolean;
  sample_nag_scale?: number;
  sample_nag_tau?: number;
  sample_nag_alpha?: number;
  sample_nag_sigma_end?: number;
  sample_nag_negative_prompt?: string;
  sample_seed?: number;
  sensenova_sample_timestep_shift?: number;
  sensenova_sample_img_cfg_scale?: number;
  sensenova_sample_cfg_norm?: "none" | "global";
  debug_latents?: boolean;
  debug_latents_every?: number;
  enable_bucketing?: boolean;
  base_resolutions?: number[];
  bucket_strategy?: string;
  multi_resolution_mode?: string;
  res_curriculum_enable?: boolean;
  res_curriculum_warmup_steps?: number;
  res_curriculum_warmup_scale?: number;
  // Epoch-dynamic crop augmentation (SDXL only)
  crop_augment_enable?: boolean;
  crop_full_image_prob?: number;
  crop_max_bucket_prob?: number;
  crop_min_area_ratio?: number;
  crop_min_short_side_px?: number;
  crop_aspect_mode?: string;
  crop_position_mode?: string;
  crop_smaller_bucket_mode?: string;
  crop_smaller_scale_range?: number[];
  full_crop_position_mode?: string;
  crop_microcond_mode?: string;
  crop_plan_seed?: number;
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
  gradient_checkpointing?: boolean;
  cpu_offload_checkpointing?: boolean;
  async_cpu_offload_checkpointing?: boolean;
  fp8_base_dtype?: string | null;
  torch_compile?: string;
  torch_compile_dynamic?: boolean | null;
  attention_backend?: string;
  attention_impl?: string;  // "conduit" | "diffusers" (training registry selector; SDXL/SD1.5)
  use_flash_attention?: boolean;
  min_snr_gamma?: number;
  text_encoding_mode?: string;
  text_encoding_swap_interval?: number;
  text_encoding_prefetch_depth?: number;
  latent_encoding_mode?: string;
  latent_encoding_swap_interval?: number;
  // Aligned CFG unconditional training (arch-agnostic). TRI-STATE: omit the key
  // for "not supplied" (the backend resolves the per-architecture default), 0
  // for "explicitly disabled". Sending `null` is the same as omitting it.
  // An explicit value -- 0 included -- is a 400 on an architecture whose
  // archCapabilities.cfg_null_stage entry is null, so gate the control on
  // trainingFeatureUnsupportedReason(caps, arch, "cfg_uncond_drop").
  cfg_uncond_drop_rate?: number | null;
  // MiniT2I
  // DEPRECATED spelling of cfg_uncond_drop_rate; sending both is a 400.
  minit2i_label_drop_rate?: number | null;
  minit2i_lr_factor?: number;
  minit2i_flan_t5_path?: string;
  minit2i_lora_scope?: string;
  minit2i_te_lora_scope?: string;
  minit2i_scratch_init_from?: string;  // from-scratch: inherit weights from this model
  minit2i_inherit_final_layer?: boolean;  // from-scratch: also inherit the output head (final_layer.linear)
  // Krea 2 (single-stream flow-matching MMDiT)
  krea2_lora_scope?: string;  // "attn,mlp,text_fusion,proj" tokens; TE always frozen
  krea2_lr_factor?: number;
  krea2_discrete_flow_shift?: number;
  // SDXL high-spec VAE migration (swap VAE + resize U-Net conv_in/out). "none"=standard 4ch.
  sdxl_vae_type?: string;
  // SDXL Text Encoder swap (CLIP -> alt encoder + trainable bridge adapters). "none"=CLIP.
  sdxl_te_type?: string;
  sdxl_te_hidden_layer?: number;   // which TE hidden layer to tap (-2 = penultimate)
  sdxl_te_max_len?: number;        // fixed token length
  sdxl_te_train_encoder?: boolean; // false = adapters only; true = TE body + adapters
  // REPA (Representation Alignment) — MiniT2I only
  repa_enable?: boolean;
  repa_encoder_source?: string;        // "tagger" | "siglip2"
  repa_tagger_model_dir?: string;      // tagger model dir (empty = auto-pick)
  repa_siglip2_repo?: string;          // off-the-shelf SigLIP2 repo
  repa_align_depth?: number;           // -1 = auto (depth//3)
  repa_weight?: number;                // alignment loss weight (lambda)
  repa_proj_lr_factor?: number;        // projector LR multiplier (x unet_lr)
  repa_encoder_resolution?: number;    // 0 = follow encoder native image_size
  // Ideogram4-specific
  ideogram4_lora_scope?: string;
  ideogram4_train_uncond?: boolean;
  ideogram4_uncond_loss_weight?: number;
  ideogram4_lr_factor?: number;
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
  block_swap_h2d_only?: boolean;
  block_swap_ring_size?: number;
  num_optimizer_groups?: number;
  // Full-parameter save: embed the VAE into the single-file checkpoint (default off).
  bundle_vae?: boolean;
  activation_dispatch_enable?: boolean;
  activation_dispatch_margin_gb?: number;
  activation_dispatch_seed_coef?: number;
  activation_dispatch_residual_frac?: number;
  activation_dispatch_threshold_mb?: number;
  multi_noise_timesteps?: number;
  multi_noise_mode?: string;
  stratified_timesteps?: boolean;  // One timestep per equal-probability stratum across the MNT window (default true)
  grad_timestep_cosine_probe?: boolean;  // Diagnostic: cosine between the noisy-half and clean-half gradients of each MNT window
  grad_timestep_cosine_sketch_dim?: number;  // Bilinear sketch width for the above (default 8)
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
  // FLUX.2/SenseNova explicit arm; SD/SDXL mirrors a selected SigLIP2 VE.
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
  // Outpaint-native ControlNet conditioning (PART B): self-supervised
  // crop->full instead of preprocessor-derived condition images.
  conditioning_mode?: "preprocessor" | "outpaint";
  outpaint_crop_min_area?: number;
  outpaint_crop_max_area?: number;
  outpaint_edge_anchor_prob?: number;
  outpaint_corner_anchor_prob?: number;
  outpaint_mask_channel?: boolean;
  outpaint_known_loss_weight?: number;
  outpaint_seam_loss_boost?: number;
  outpaint_seam_ring_width?: number;
  outpaint_seam_grad_lambda?: number;
  outpaint_loss_normalize?: boolean;
  // Pre-flight dataset drift check + optional rescan + orphan latent
  // cache cleanup.  Modes:
  //   "off"   — skip
  //   "path"  — detect added/missing files only
  //   "smart" — path drift + caption sidecar mtime
  //   "force" — always rescan
  // Legacy boolean also accepted (true→"path", false→"off") for backwards
  // compatibility with older clients.
  rescan_before_training?: "off" | "path" | "smart" | "force" | boolean;
  // Optimizer hyperparameters.
  // Paging is part of the optimizer NAME (paged_adamw / paged_adamw8bit /
  // paged_lion8bit), not a boolean; the removed optimizer_is_paged flag was
  // read by no trainer.
  optimizer_cautious?: boolean;
  optimizer_beta1?: number;
  optimizer_beta2?: number;
  optimizer_epsilon?: number;
  optimizer_weight_decay?: number;
  optimizer_schedule_free?: boolean;
  optimizer_schedule_free_r?: number;
  optimizer_schedule_free_weight_lr_power?: number;
  optimizer_use_radam?: boolean;
  // Tri-state: true/false are explicit; null/undefined ("not specified")
  // lets the architecture decide (e.g. some full fine-tune routes force it on).
  optimizer_stochastic_rounding?: boolean | null;
  // Ring-buffer optimizers only: 8-bit state as pinned host memory instead of
  // on the GPU. Required by SenseNova full fine-tuning for those two names.
  optimizer_state_host_resident?: boolean;
  // LoRA
  lora_dtype?: "fp32" | "fp16" | "bf16";
  // Component training (image encoder)
  train_image_encoder?: boolean;
  image_encoder_lr?: number | null;
  // Reconstruction loss
  reconstruction_loss_weight?: number;
  // MiniMax-H3 only: weight of the AUDIO half of its joint objective
  // (loss = video_mean + audio_loss_weight * audio_mean, each modality's
  // velocity MSE averaged over tokens/channels/samples before weighting).
  // 0 trains on the video half only. Ignored by every other architecture.
  audio_loss_weight?: number;
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
  // 1-based for display; null when no metrics logged yet / step-configured run.
  current_epoch?: number | null;
  total_epochs?: number | null;
  loss?: number;
  learning_rate?: number;
  phase?: string | null;
  phase_progress?: number | null;
  phase_detail?: string | null;
  /** Backlog for the `training_log` WebSocket channel, which replays nothing on
   *  connect. Every notice this run has emitted, in order. */
  warnings?: TrainingLogEvent[];
}

// ---------------------------------------------------------------------------
// VAE decoder fine-tune (training_method "vae_decoder")
// ---------------------------------------------------------------------------
// Same key set as backend/api/param_defaults.py VAE_TRAINING_DEFAULTS, served by
// GET /schema/vae-training-defaults (see fetchVaeTrainingDefaults). A VAE run is
// an ordinary TrainingRun row, so listing / status / start / stop / delete /
// checkpoints / metrics all go through the existing training endpoints.

export interface VaeTrainingConfig {
  // Run shape. These live in process.train / process.save in the YAML, but they
  // must still be sent inside vae_config: the backend treats a FLAT field the
  // caller sent as deliberate and lets it override the VAE default.
  batch_size: number;
  total_steps: number;
  gradient_accumulation_steps: number;
  learning_rate: number;
  optimizer: string;
  optimizer_weight_decay: number;
  max_grad_norm: number;
  lr_scheduler: string;
  lr_warmup_steps: number;
  seed: number;
  num_workers: number;
  save_every: number;
  max_step_saves_to_keep: number;
  // Base VAE selection
  vae_source: "model" | "path" | "store";
  vae_path: string;
  // VAE-store key ("" = not stated, the default). Two jobs: it selects the entry
  // to load when vae_source is "store", and for a base VAE that comes from a
  // SINGLE FILE (any source) it is the only statement of which family that file
  // belongs to — such a file has no config.json, so the scaling_factor on the
  // loaded config is a fallback, and save_pretrained bakes it into every export.
  // The backend refuses a single-file base VAE whose vae_arch is unstated or
  // unknown rather than assuming one.
  vae_arch: string;
  // What to train. Encoder training is behind a double gate: the backend
  // refuses train_encoder without acknowledge_latent_space_break, AND refuses
  // acknowledge_latent_space_break without train_encoder.
  train_decoder: boolean;
  decoder_blocks: "all" | "up_blocks" | "mid_block" | "conv_out";
  train_encoder: boolean;
  acknowledge_latent_space_break: boolean;
  encoder_blocks: "all" | "down_blocks" | "mid_block" | "conv_out";
  // Optimisation shape
  resolution: number;
  // How much an image is resampled before the square crop. "downscale" is the
  // historical behaviour (short side -> resolution, which downscales 95.8% of
  // the corpus by a median 2.30x); "native" crops out of the full-size pixels;
  // "mixed" draws the factor per sample. crop_scale_max_downscale bounds the
  // "mixed" draw (0 = unbounded) and is REFUSED by the backend under any other
  // policy, so the panel clears it when leaving "mixed".
  crop_scale_policy: "downscale" | "native" | "mixed";
  crop_scale_max_downscale: number;
  dtype: "bf16" | "fp32";
  ema_enabled: boolean;
  ema_decay: number;
  // Losses
  mse_weight: number;
  l1_weight: number;
  lpips_weight: number;
  lpips_net: "vgg" | "alex" | "squeeze";
  ycbcr_dc_weight: number;
  ycbcr_dc_y_weight: number;
  ycbcr_dc_chroma_weight: number;
  ycbcr_dc_eps: number;
  pattern_weight: number;
  pattern_size: number;
  // Flat-region invented-HF penalty. Penalises high-frequency energy in the
  // decode that a least-squares projection onto the source's own high-frequency
  // content cannot explain, inside plane-fit flat/gradient windows. 0 disables
  // the term; the window geometry and projection constants are fixed inside the
  // backend and are not exposed. NOT a standalone objective: its own minimum
  // inside a flat window is "emit no high frequency at all", so it is meant to
  // run alongside an agreement-with-source term (mse / lpips).
  l_invented_weight: number;
  l_invented_y_weight: number;
  l_invented_chroma_weight: number;
  // Plane-fit residual thresholds (8-bit levels) deciding which windows count
  // as flat. The backend refuses 0 for either WHEN THE TERM IS ON; with
  // l_invented_weight = 0 nothing here is read and nothing is refused.
  l_invented_flat_t_y: number;
  l_invented_flat_t_c: number;
  // Posterior KL. Only constructed when the encoder is trainable; ignored (and
  // reported as ignored by the trainer) under a frozen encoder.
  kl_weight: number;
  // Export. Refused by the backend together with train_encoder.
  export_bare_ldm: boolean;
  // Validation
  validation_every: number;
  validation_num_images: number;
  validation_resolution: number;
}

// What the VAE panel sends. Deliberately narrow: any additional flat field would
// be recorded in the request's model_fields_set and would then override the
// corresponding VAE default. total_steps is the one exception - the create route
// requires total_steps or epochs at the top level for the DB column, so it is
// sent flat AND inside vae_config with the same value.
export interface VaeTrainingRunCreateRequest {
  dataset_configs: DatasetConfigItem[];
  run_name?: string;
  training_method: "vae_decoder";
  base_model_path: string;
  total_steps: number;
  resume_from_checkpoint?: string | null;
  vae_config: VaeTrainingConfig;
}

export const createVaeTrainingRun = async (data: VaeTrainingRunCreateRequest): Promise<TrainingRun> => {
  const response = await api.post("/training/runs", data);
  return response.data;
};

export const updateVaeTrainingRun = async (id: number, data: VaeTrainingRunCreateRequest): Promise<TrainingRun> => {
  const response = await api.put(`/training/runs/${id}`, data);
  return response.data;
};

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

export type MetricFamily =
  | "loss" | "gradient_norm" | "learning_rate" | "bounded_diagnostic"
  | "signed_correlation" | "binary_indicator" | "count" | "duration"
  | "data_volume" | "validation" | "other";

export type MetricRange =
  | { kind: "auto"; floor?: number }
  | { kind: "fixed"; min: number; max: number };

/** Display + semantic metadata for a bespoke extra metric (from the backend
 *  registry, core/training/metric_registry.py). Every semantic field is
 *  optional; unannotated keys fall back to metricCatalog's key heuristics. */
export interface MetricSeriesDef {
  label?: string;
  color?: string;
  dashed?: boolean;
  /** LEGACY hint. "right" renders the series on a separate, independently-scaled
   *  secondary Y-axis instead of pooling it into the primary Y-range (e.g.
   *  learning rate, which lives orders of magnitude below loss). Superseded by
   *  `scale_group`; still read by the older charts. */
  axis?: "right";
  family?: MetricFamily;
  /** Two series may share a Y-axis iff their scale groups are equal. */
  scale_group?: string;
  range?: MetricRange;
  sampling?: "dense" | "periodic" | "event";
}

export interface TrainingMetrics {
  loss: MetricPoint[];
  recon_loss: MetricPoint[];
  // Bespoke, arch/method-specific metrics (REPA, outpaint gen_loss, …) keyed by
  // metric name. Grows without a code change per metric — see extra_metric_defs
  // for display metadata (label/color/dashed) echoed from the backend registry.
  extra_metrics?: Record<string, MetricPoint[]>;
  extra_metric_defs?: Record<string, MetricSeriesDef>;
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
  // SDXL micro-conditioning (crop-augmentation verification), item 0 of the debug batch
  original_size?: [number, number];   // (W, H) full original image
  crop_top_left?: [number, number];   // (left, top) crop point in original pixels
  target_size?: [number, number];     // (W, H) output bucket
  sdxl_time_ids?: number[];           // [oh, ow, ct, cl, th, tw]
  sdxl_time_ids_all?: number[][];     // per-item time_ids for the whole batch
  latents_image?: string;  // base64
  noisy_latents_image?: string;  // base64
  predicted_noise_image?: string;  // base64
  predicted_latent_image?: string;  // base64
  // Video archs (LTX-2.3, MiniMax-H3): the images above tile the clip's leading
  // decodable window along width, window_latent_frames frames wide.
  window_latent_frames?: number;
  clip_latent_frames?: number;
  // ACE-Step / MiniMax-H3 audio: latent channels vs time, no vocoder.
  latent_frames?: number;
  audio_sigma?: number;
  audio_present?: number;
  audio_latents_image?: string;  // base64
  audio_noisy_latents_image?: string;  // base64
  audio_predicted_velocity_image?: string;  // base64
  audio_actual_velocity_image?: string;  // base64
  audio_predicted_latent_image?: string;  // base64
  // Per-channel mean/std of every saved stream, keyed by tensor name.
  channel_stats?: Record<string, { mean: number[]; std: number[] }>;
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
  gpu_index?: number | null;  // Physical GPU index to run this training run on; null = backend default device
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
