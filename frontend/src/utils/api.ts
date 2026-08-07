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
  type: "sd15" | "sdxl" | "zimage" | "flux2" | "anima" | "lens" | "ideogram4" | "minit2i" | "krea2";  // DEUS support removed
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
  is_style_transfer?: boolean;
  style_adain_strength?: number;
  style_blocks?: string;
  style_low_scale_end?: number;
  style_high_scale?: number;
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
  // Music generation fields (used when an audio model (ACE-Step) is loaded;
  // the panel maps these into Txt2AudParams for txt2aud requests). `prompt`
  // above doubles as the caption text.
  lyrics?: string;
  audio_duration?: number;          // seconds, default 30.0
  inference_steps?: number;         // ACE-Step sampler steps (default 8, turbo distilled)
  shift?: number;                   // default 3.0
  sampler_mode?: string;            // accepted for forward-compat; currently a no-op
  vocal_language?: string;          // default "en"
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

// Video temporal outpaint (LTX-2.3): place a (optionally trimmed) input clip
// at a frame offset inside a LONGER output timeline and generate the frames
// before/after, preserving the placed input frames byte-exact. Mirrors the
// backend OUTPAINT_VIDEO_DEFAULTS (param_defaults.py) + the Form parameters
// of POST /generate/outpaint/video (routes.py). Standalone shape (does not
// extend GenerationParams, matching Txt2VidParams/Img2VidParams -- video has
// no width/height/steps/sampler concept beyond the fields below).
export interface OutpaintVideoParams {
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
  // Video parameters (generation_type === 'txt2vid' / 'img2vid'; filename is an .mp4)
  is_video?: boolean;
  num_frames?: number;
  fps?: number;
  duration?: number;
  audio_enable?: boolean;
  // Audio parameters (generation_type === 'txt2aud' / 'aud2aud'; filename is a .flac)
  is_audio?: boolean;
  sample_rate?: number;
  audio_duration?: string;
}

// ---------------------------------------------------------------------------
// Video generation (LTX-2.3) — txt2vid (JSON) / img2vid (multipart keyframe)
// ---------------------------------------------------------------------------

export interface Txt2VidParams {
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
  prompt: string;             // caption text
  lyrics?: string;
  audio_duration?: number;    // seconds, default 30.0
  seed?: number;               // default -1
  inference_steps?: number;   // turbo distilled default 8
  guidance_scale?: number;    // turbo is CFG-distilled; default 1.0
  shift?: number;              // default 3.0
  sampler_mode?: string;       // accepted for forward-compat; currently a no-op
  vocal_language?: string;     // default "en"
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
  prompt: string;              // caption text (also accepted as "caption")
  lyrics?: string;
  seed?: number;                // default -1
  inference_steps?: number;    // turbo distilled default 8
  guidance_scale?: number;     // turbo is CFG-distilled; default 1.0
  shift?: number;               // default 3.0
  vocal_language?: string;      // default "en"
  loras?: LoRAConfig[];
  // --- Placement (outpaint-only), all in SECONDS ---
  total_duration?: number;         // output timeline length; (0, 240], default 60.0
  input_offset_sec?: number;       // where the (trimmed) clip lands, snapped server-side to 1/25s
  input_trim_start_sec?: number;   // trim applied to the UPLOADED clip before placement
  input_trim_end_sec?: number;
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
}

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

export const fetchGenerationDefaults = async (): Promise<GenerationDefaultsResponse> =>
  (await api.get("/schema/generation-defaults")).data;

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
}

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
}

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
export const normalizeVideoFrames = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number | null | undefined
): number | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || frames == null) return frames ?? null;
  if (isValidVideoFrameCount(caps, arch, frames)) return frames;
  const offered = c.suggested_frames?.length ? c.suggested_frames : null;
  if (!offered) return frames;
  return offered.reduce((best, n) =>
    Math.abs(n - frames) < Math.abs(best - frames) ? n : best, offered[0]);
};

// Label for that control, stating the arch's own rule ("17n+5, 124-345") rather
// than a hardcoded "8k+1".
export const videoFrameLabel = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): string => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c) return "Frames";
  const rule = c.frame_offset === 0
    ? `${c.frame_multiple}n`
    : `${c.frame_multiple}n+${c.frame_offset}`;
  const range = c.max_frames != null ? `, ${c.min_frames}-${c.max_frames}` : "";
  return `Frames (${rule}${range})`;
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
  const align = c?.pixel_align && c.pixel_align > 0 ? c.pixel_align : 32;
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
  const align = c?.pixel_align && c.pixel_align > 0 ? c.pixel_align : 32;
  const cap = c?.max_pixel_hw ?? null;
  const alignRule = `both sides must be a multiple of ${align}`;
  if (!cap) return alignRule;
  return `${alignRule}, the short side is capped at ${cap[0]} and the long side at ${cap[1]}`;
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
};

export const archDisplayName = (arch: string | null | undefined): string =>
  (arch && ARCH_DISPLAY_NAMES[arch]) || arch || "";

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

export const getResultSeed = (result: any): number =>
  result?.image?.seed ?? result?.actual_seed ?? -1;

export const getResultAncestralSeed = (result: any): number | null =>
  result?.image?.ancestral_seed ?? result?.actual_ancestral_seed ?? null;

export const generateTxt2Img = async (params: GenerationParams) => {
  // Get attention_type from localStorage
  const attentionType = typeof window !== 'undefined' ? localStorage.getItem('attention_type') : null;
  const attentionImpl = typeof window !== 'undefined' ? localStorage.getItem('attention_impl') : null;

  // Only load ControlNet images if they exist (avoid unnecessary localStorage access)
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "txt2img_controlnet_collapsed")
    : params.controlnets;

  const paramsWithImages = {
    ...params,
    attention_type: attentionType || 'normal',
    attention_impl: attentionImpl || 'conduit',
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
  const attentionImpl = typeof window !== 'undefined'
    ? localStorage.getItem('attention_impl') : null;
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "txt2img_controlnet_collapsed")
    : params.controlnets;

  const body = {
    ...params,
    attention_type: attentionType || 'normal',
    attention_impl: attentionImpl || 'conduit',
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
  const attentionImpl = typeof window !== 'undefined'
    ? localStorage.getItem('attention_impl') : null;
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "img2img_controlnet_collapsed")
    : params.controlnets;
  const body = {
    ...params,
    attention_type: attentionType || 'normal',
    attention_impl: attentionImpl || 'conduit',
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
  const attentionImpl = typeof window !== 'undefined'
    ? localStorage.getItem('attention_impl') : null;
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "inpaint_controlnet_collapsed")
    : params.controlnets;
  const body = {
    ...params,
    attention_type: attentionType || 'normal',
    attention_impl: attentionImpl || 'conduit',
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
  // Get attention_type from localStorage
  const attentionType = typeof window !== 'undefined' ? localStorage.getItem('attention_type') : null;
  const attentionImpl = typeof window !== 'undefined' ? localStorage.getItem('attention_impl') : null;

  // Only load ControlNet images if they exist (avoid unnecessary localStorage access)
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "img2img_controlnet_collapsed")
    : params.controlnets;

  const paramsWithImages = {
    ...params,
    attention_type: attentionType || 'normal',
    attention_impl: attentionImpl || 'conduit',
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

  const response = await api.post("/generate/img2img", formData, {
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
    formData.append("seed", String(params.seed ?? -1));
    formData.append("diffusion_pre_upscale_mode", params.diffusion_pre_upscale_mode || "pil");
  }

  const response = await api.post("/generate/upscale", formData, {
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
  };

  const response = await api.post("/generate/txt2vid", body);
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

  const response = await api.post("/generate/img2vid", formData, {
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

  const response = await api.post("/generate/ref2vid", formData, {
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

  const response = await api.post("/generate/txt2aud", body);
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
  // Weight-only quantization (both axes). Appended only when set, so an unset
  // field leaves the backend default (and the process GEMM flags) untouched.
  if (params.unet_quantization && params.unet_quantization !== "none") {
    formData.append("unet_quantization", params.unet_quantization);
  }
  if (params.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", params.quantized_gemm_mode);
  }

  const response = await api.post("/generate/aud2aud", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const generateInpaint = async (params: InpaintParams, image: File | string, mask: File | string) => {
  // Get attention_type from localStorage
  const attentionType = typeof window !== 'undefined' ? localStorage.getItem('attention_type') : null;
  const attentionImpl = typeof window !== 'undefined' ? localStorage.getItem('attention_impl') : null;

  // Only load ControlNet images if they exist (avoid unnecessary localStorage access)
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "inpaint_controlnet_collapsed")
    : params.controlnets;

  const paramsWithImages = {
    ...params,
    attention_type: attentionType || 'normal',
    attention_impl: attentionImpl || 'conduit',
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

  const response = await api.post("/generate/inpaint", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

// Outpaint: clone of generateInpaint's FormData sender, minus the mask
// upload (the backend builds its own canvas + mask from `image` + the
// placement fields), plus the placement fields themselves. See
// core/inference/outpaint_utils.py + PipelineManager.generate_outpaint.
export const generateOutpaint = async (params: OutpaintParams, image: File | string) => {
  // Get attention_type from localStorage
  const attentionType = typeof window !== 'undefined' ? localStorage.getItem('attention_type') : null;
  const attentionImpl = typeof window !== 'undefined' ? localStorage.getItem('attention_impl') : null;

  // Only load ControlNet images if they exist (avoid unnecessary localStorage access)
  const controlnets = (params.controlnets && params.controlnets.length > 0)
    ? await loadControlNetImages(params.controlnets, "outpaint_controlnet_collapsed")
    : params.controlnets;

  const paramsWithImages = {
    ...params,
    attention_type: attentionType || 'normal',
    attention_impl: attentionImpl || 'conduit',
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

  const response = await api.post("/generate/outpaint", formData, {
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
  bridgeVideo?: File | string | null
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
  const attentionType = typeof window !== 'undefined' ? localStorage.getItem('attention_type') : null;
  formData.append("attention_type", params.attention_type || attentionType || "normal");

  // Acceleration (block swap / FBCache / Spectrum)
  formData.append("blocks_to_swap", String(params.blocks_to_swap ?? 0));
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

  const response = await api.post("/generate/outpaint/video", formData, {
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

  // Placement (outpaint-only), all in seconds.
  formData.append("total_duration", String(params.total_duration ?? 60.0));
  formData.append("input_offset_sec", String(params.input_offset_sec ?? 0.0));
  formData.append("input_trim_start_sec", String(params.input_trim_start_sec ?? 0.0));
  formData.append("input_trim_end_sec", String(params.input_trim_end_sec ?? 0.0));

  // Weight-only quantization (both axes). Appended only when set, so an unset
  // field leaves the backend default (and the process GEMM flags) untouched.
  if (params.unet_quantization && params.unet_quantization !== "none") {
    formData.append("unet_quantization", params.unet_quantization);
  }
  if (params.quantized_gemm_mode) {
    formData.append("quantized_gemm_mode", params.quantized_gemm_mode);
  }

  const response = await api.post("/generate/outpaint/audio", formData, {
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

// `force`: reload even when this model is already the loaded one. Without it the
// backend early-returns, so nothing per-session is reset — which is what makes
// "load the model again" the working recovery for the one-way in-place INT8
// conversion (unet_quantization="int8" on anima/krea2/flux2/ideogram4).
export const loadModel = async (sourceType: string, source: string, revision?: string, force?: boolean) => {
  const formData = new FormData();
  formData.append("source_type", sourceType);
  formData.append("source", source);
  if (revision) {
    formData.append("revision", revision);
  }
  if (force) {
    formData.append("force", "true");
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
  base_model_path: string;
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
  attention_backend?: string;
  attention_impl?: string;  // "conduit" | "diffusers" (training registry selector; SDXL/SD1.5)
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
  optimizer_stochastic_rounding?: boolean;
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
  loss?: number;
  learning_rate?: number;
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

/** Display metadata for a bespoke extra metric (from the backend registry). */
export interface MetricSeriesDef {
  label?: string;
  color?: string;
  dashed?: boolean;
  /** "right" renders the series on a separate, independently-scaled secondary
   *  Y-axis instead of pooling it into the primary Y-range (e.g. learning
   *  rate, which lives orders of magnitude below loss). */
  axis?: "right";
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
