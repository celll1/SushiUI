"""
Single source of truth for all API parameter default values.

Backend Pydantic models and Form() defaults reference this module.
Frontend fetches these via /schema/* endpoints, eliminating manual sync.
"""

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Generation (txt2img / img2img / inpaint)
# ---------------------------------------------------------------------------
# Authoritative source: Form() defaults in generate_* route handlers.
# Pydantic GenerationParams defaults are aligned to these values.

GENERATION_DEFAULTS: Dict[str, Any] = {
    # Core
    "negative_prompt": "",
    "steps": 20,
    "cfg_scale": 7.0,
    "sampler": "euler",
    "schedule_type": "uniform",
    "seed": -1,
    "ancestral_seed": -1,
    "width": 1024,
    "height": 1024,
    "batch_size": 1,
    "prompt_chunking_mode": "a1111",
    "max_prompt_chunks": 0,
    "developer_mode": False,
    # CFG scheduling
    "cfg_schedule_type": "constant",
    "cfg_schedule_min": 1.0,
    "cfg_schedule_max": None,
    "cfg_schedule_power": 2.0,
    "cfg_rescale_snr_alpha": 0.0,
    # Dynamic thresholding
    "dynamic_threshold_percentile": 0.0,
    "dynamic_threshold_mimic_scale": 7.0,   # Form() default; Pydantic had 1.0 (bug)
    # NAG
    "nag_enable": False,
    "nag_scale": 5.0,
    "nag_tau": 3.5,
    "nag_alpha": 0.25,
    "nag_sigma_end": 3.0,                   # Form() default; Pydantic had 0.0 (bug)
    "nag_negative_prompt": "",
    # Attention / quantization
    "attention_type": "normal",
    # Which attention IMPLEMENTATION runs the kernel (orthogonal to attention_type,
    # which selects the backend). "conduit" routes through core/attention (enables
    # conduit-only backends such as tq on FLUX.2); "diffusers" keeps diffusers' own
    # registry (byte-identical legacy path). Consumed by the FLUX.2 inference path;
    # other archs are conduit-only or ignore it.
    "attention_impl": "conduit",
    "unet_quantization": None,
    "text_encoder_quantization": None,
    "cpu_text_encoding": False,
    "use_torch_compile": False,
    # Keep-models-hot (queue optimization, SD1.5/SDXL only in this phase): when
    # generating several items back-to-back on the SAME loaded model (same
    # checkpoint + LoRA set + unet_quantization), skip the CPU offload at the end
    # of a successful generation (and the matching GPU stage at the start of the
    # next one) for components it is safe to keep resident, bounded by a VRAM
    # guard. Off by default (byte-identical staging/offload behavior). See
    # core/keep_hot.py.
    "keep_models_hot": False,
    # VAE tiling: decode the latent in overlapping tiles so the VAE decode peak is
    # bounded by the tile size, not the full image. Lets large images decode
    # without OOM. Off by default (not bit-identical to a full decode; small images
    # below the tile threshold are unaffected).
    "vae_tiling": False,
    # Image size (px) above which VAE tiling kicks in (and the tile size). 0 = auto
    # = VAE sample_size * 1.5 (e.g. ~1536px for SDXL). Below the threshold the decode
    # runs whole (no quality/speed cost); above it, split into threshold-sized tiles.
    "vae_tile_threshold": 0,
    # Color Flatten (chroma smoothing): RGB-guided guided filter applied to the
    # decoded image's YCoCg chroma (luma untouched) right after VAE decode. Removes
    # low-frequency color mottling while preserving luminance detail. 0-100; 0 = off
    # (zero cost). All modes, all architectures.
    "color_flatten_strength": 0,
    # VAE DC-drift correction (img2img/inpaint only): subtract the per-channel DC
    # bias the VAE round-trip introduces (mean(decode(encode(input))) - mean(input))
    # from the final decode. Corrects a VAE property, so it is strength-independent.
    "vae_drift_correction": False,
    # In-loop hard-flatten (SD1.5/SDXL only): on the last N actual denoise steps,
    # decode the x0 prediction, detect the flat background region (largest connected
    # low-gradient component touching the border) and replace it with its dominant
    # colour, then re-encode and inject the correction back into the latents. Gated:
    # if no confident flat region (area < min_region) the step is a no-op, so
    # textured backgrounds are protected. DiT archs accept the flag and warn.
    "flatten_in_loop": False,       # master switch (default off = byte-identical loop)
    # Number of trailing ACTUAL denoise steps to inject on (relative to the real step
    # sequence, not a fixed fraction, so it is stable across step counts / accelerators).
    # Larger N = flatter background but more subject-detail cost (~6-7% wall per injection).
    "flatten_in_loop_last_steps": 3,
    # Flat-region area gate as a fraction of the frame. Below this the step is a no-op.
    "flatten_in_loop_min_region": 0.02,
    # Spectrum: Adaptive Spectral Feature Forecasting (training-free acceleration).
    # Skips U-Net forwards on selected steps by forecasting the output from a Chebyshev
    # fit over actual passes. Most useful at high step counts (>=30); little benefit on
    # low-step/distilled models. Auto-disabled with prompt-editing/ControlNet/DEUS.
    "spectrum_enable": False,
    "spectrum_w": 0.5,             # spectral/linear mix (1.0 = spectral only; lower = more linear/stable)
    "spectrum_w_decay": 0.0,       # OPT-IN per-step decay exponent for spectrum_w (0 = off, default)
    "spectrum_delta_cap": 0.0,     # OPT-IN trajectory speed limiter multiplier K (0 = off, default)
    "spectrum_m": 4,               # number of Chebyshev basis
    "spectrum_lam": 0.1,           # ridge regularization
    "spectrum_warmup_steps": 3,    # leading full-eval steps
    "spectrum_window_size": 4,     # initial skip interval
    "spectrum_flex_window": 0.75,  # skip damping (0 = max skip)
    "spectrum_tail": 0.12,         # fraction of final steps forced to actual passes (detail)
    "spectrum_feature_mode": "output",  # "output" (black-box) or "block" (deep-feature, paper-faithful)
    "spectrum_cache_branch": 1,    # block mode: down_blocks[branch:] + mid are forecast
    "spectrum_max_cache": 0,       # forecaster sliding window (0 = unlimited; block mode -> 6)
    # First Block Cache (FBCache): dynamic per-step caching. Mutually exclusive with Spectrum.
    "fbcache_enable": False,
    "fbcache_threshold": 0.12,     # relative-L1 first-block residual threshold (higher = more skips/faster)
    "fbcache_warmup_steps": 1,     # always compute the first N steps
    "fbcache_cache_branch": 1,     # U-Net block mode: indicator = down[branch]; reused region = down[branch+1:]+mid
    "use_tipo": False,
    "preview_predicted_x0": False,
    # Block swap (Form-only in original code)
    "enable_block_swap": False,
    "blocks_to_swap": 20,
    "use_pinned_memory": False,
    "block_swap_h2d_only": False,   # H2D-only swap (no device->host eviction of read-only weights)
    "block_swap_ring_size": 2,      # GPU weight-buffer ring slots (>=1; 2 double-buffers)
    # Vision encoder
    "vision_encoder_path": None,
    # Per-generation component overrides (RP2b). Both default to None = use the
    # loaded model's own component. VAE override supported on all image archs
    # except ltx2/minit2i; TE override supported on sd15/sdxl only.
    "vae_path": None,
    "text_encoder_path": None,
    # PiD (Pixel Diffusion Decoder) VAE-override options — only consulted when
    # vae_path points at a PiD checkpoint (kind="pid_decoder"; see
    # api.generation_overrides.classify_vae_candidate); no effect otherwise.
    # "4x" = PiD's native super-resolution output (latent_h/w * 8 * 4); PiD's
    # sr_scale=4 is baked into the checkpoint so "original" still runs the full
    # 4x decode and downscales afterward (NOT a cheaper mode).
    "pid_sr_output": "4x",
    # False (default) = use the shipped null-caption embedding (no runtime
    # Gemma). True = load Efficient-Large-Model/gemma-2-2b-it (the ungated
    # mirror PiD was trained against) once per generation and encode the real
    # prompt for sharper, hallucination-reduced output (see phase 1b findings).
    "pid_use_gemma": False,
    # False (default) = PiTBlock/FinalLayer run their exact original,
    # unchunked forward pass (bit-identical output). True opts into a
    # row-chunked activation path that cuts the PiD decoder's per-block VRAM
    # peak (~6.6GB/42% measured at 4096px) at the cost of bf16 GEMM-tiling
    # rounding drift that is NOT bit-identical (verified bit-identical in
    # fp32; the drift is bf16-precision amplification through the 4-step SDE
    # sampler, not an implementation bug — see scratchpad/pid_vram_proposal.md).
    "pid_low_vram": False,
    # F9 — tiled large-output decode (default when native > native_cap; see
    # PidVaeWrapper's module docstring). Each tile's OWN native resolution is
    # capped at pid_tile_native (must stay <= the (currently hardcoded)
    # native_cap, else it is clamped with a warning); pid_tile_overlap_ratio
    # is the feather overlap as a fraction of the tile size (0.25 = R&D-
    # confirmed seam-free on both busy and smooth backgrounds).
    "pid_tile_native": 512,
    "pid_tile_overlap_ratio": 0.25,
    # False (default) = tiled decode (F9, true super-resolution detail at the
    # full requested output size). True opts back into the original
    # whole-latent downscale-then-decode-then-upscale (F7) — cheaper
    # (~6x fewer decode passes) but blurrier at large output sizes.
    "pid_fast_large_decode": False,
    # SDXL micro-conditioning override (inference). original_size for time_ids:
    # explicit w/h (0 = auto), else output size * scale. crop stays (0,0).
    "original_size_w": 0,
    "original_size_h": 0,
    "original_size_scale": 1.0,
    # img2img / inpaint shared
    "denoising_strength": 0.75,
    "img2img_fix_steps": True,
    "resize_mode": "image",
    "resampling_method": "lanczos",
    # inpaint only
    "mask_blur": 4,
    "inpaint_full_res": False,
    "inpaint_full_res_padding": 32,
    "inpaint_fill_mode": "original",
    "inpaint_fill_strength": 1.0,
    "inpaint_blur_strength": 1.0,
    # Regional additional prompt (SD/SDXL only, inpaint/outpaint): an
    # additional positive/negative prompt that conditions ONLY the generated
    # region (outpaint = mask_latent==1; inpaint = the repaint mask), leaving
    # the main whole-image prompt and the preserved region untouched. Active
    # iff region_prompt_strength > 0 AND (region_prompt or
    # region_negative_prompt) is non-empty. See
    # core.inference.custom_sampling.custom_inpaint_sampling_loop's REGIONAL
    # ADDITIONAL PROMPT block.
    "region_prompt": "",
    "region_negative_prompt": "",
    "region_prompt_strength": 1.0,   # 0-2; 0 = feature inactive
    "region_prompt_method": "cfg",   # "cfg" (spatial/masked CFG) | "attention" (not yet implemented -- falls back to "cfg" with a warning)
    "region_mask_feather": 0.0,      # Gaussian sigma (latent cells) for the region mask edge; 0 = hard mask
    # Seam Structure Continuity (SD/SDXL only, inpaint/outpaint): make thin
    # structures that cross the region boundary (a held rod/staff, a limb,
    # torso, architectural lines) CONTINUE coherently into the generated region.
    # Detects oriented structures crossing the boundary in the known latent
    # (structure tensor), affine-extrapolates their band-pass cross-section into
    # a shallow generate-side collar, and pulls the model's per-step x0
    # prediction toward that target -- gated per boundary position by a
    # structure-saliency map so it fires ONLY where a real crossing structure
    # touches the boundary (leaving the rest of the seam untouched). x0-space,
    # no extra U-Net forwards, known region bit-exact. See
    # core.inference.custom_sampling._ssc_precompute / _ssc_apply. 0 = off
    # (byte-identical). Active range ~0.25-0.6.
    "seam_structure_strength": 0.0,  # lambda; 0 = feature inactive
    "seam_structure_depth": 6.0,     # generate-side collar depth (latent cells)
    "seam_structure_end": 0.70,      # schedule progress at which the effect decays to 0 (full <= 0.45)
    "seam_structure_saliency": 2.0,  # saliency-gate midpoint as a multiple of the boundary-ribbon median (0 = whole seam)
    "seam_structure_max_area": 0.25, # safety cap: max fraction of the generate region the gate may cover
    # Boundary Determinism Relaxation (SD/SDXL only, inpaint/outpaint): the
    # "junction render" complement to Seam Structure Continuity. Instead of
    # HARD-pinning the known boundary each step, SOFT-pin a narrow SSC-saliency-
    # gated keep-side seam band (annealed from soft early to a full hard pin
    # late, with a small scheduled noise term), so the immediately-adjacent
    # known-side latent can bend to meet the continuation SSC indicates -- turning
    # a ~1-cell seam offset from a kink into a gentle bend. Deep known content
    # stays hard-pinned throughout. x0-space, no extra U-Net forwards. Most
    # effective WITH seam_structure_strength > 0 (SSC supplies the direction;
    # relaxation grants the permission). See core.inference.custom_sampling.
    # _bdr_precompute / _bdr_apply. 0 = off (byte-identical).
    "boundary_relax_strength": 0.0,   # 0 = feature inactive; active range ~0.2-0.35
    "boundary_relax_width": 3.0,      # keep-side band width (latent cells)
    "boundary_relax_noise": 0.35,     # band-noise fraction of the x0-posterior std (0-1)
    "boundary_relax_full_until": 0.37, # progress up to which the band is fully soft
    "boundary_relax_end": 0.55,       # progress by which the hard pin is fully restored
    "boundary_relax_paste": "feather", # Q3 paste variant: "feather" (thin model-rendered seam strip) | "exact" (full byte-exact rect)
    # Loop-generation decode mode (txt2img/img2img/inpaint, all steps of a
    # client-driven loop). "full" = decode with the active VAE (PiD if
    # overridden) + save + gallery (current/default behavior). "cheap" = if a
    # PidVaeWrapper override is active, decode with its EMBEDDED real SDXL VAE
    # instead of running the PiD student net (no-op -- identical to "full" --
    # when no PiD override is active); still saves + galleries. "none" = skip
    # decode/save/gallery entirely; the final latent is cached server-side
    # (core.inference.latent_cache) and a latent_id is returned instead of an
    # image, for the next loop step's input_latent_id (no VAE round-trip).
    # SD1.5/SDXL only in this phase (the legacy custom_sampling.py pipeline);
    # other architectures accept the field but always behave as "full".
    "loop_decode": "full",
    # img2img/inpaint: start denoising from a cached latent (see
    # core.inference.latent_cache) instead of an uploaded image -- the
    # loop-generation counterpart of loop_decode="none" on the PRODUCING step.
    # None = normal image upload (default). SD1.5/SDXL img2img only; inpaint
    # accepts the field but rejects a non-null value (mask compositing needs a
    # real source image -- see the design doc).
    "input_latent_id": None,
    # Orthogonal to loop_decode: when true, the generated image is still saved
    # to disk (so the loop can chain to the next step via the file path) but
    # the gallery database record (thumbnail + DB row) is skipped. Used for
    # loop-generation intermediate steps that decode cheaply (e.g. image-space
    # upscale via loop_decode="cheap") but shouldn't clutter the gallery.
    # Meaningless combined with loop_decode="none" (nothing is decoded/saved
    # there either way).
    "skip_gallery": False,
}

# Keys present only in img2img/inpaint (not txt2img)
_IMG2IMG_ONLY = frozenset({
    "denoising_strength", "img2img_fix_steps", "resize_mode", "resampling_method",
    # VAE DC-drift correction requires an input image to measure the round-trip
    # bias, so it exists for img2img + inpaint only (excluded from txt2img).
    "vae_drift_correction",
    # Latent-passthrough loop chaining: only meaningful where an input image
    # would otherwise be required.
    "input_latent_id",
})
# Keys present only in inpaint (not txt2img or img2img)
_INPAINT_ONLY = frozenset({
    "mask_blur", "inpaint_full_res", "inpaint_full_res_padding",
    "inpaint_fill_mode", "inpaint_fill_strength", "inpaint_blur_strength",
    # Regional additional prompt: mask-scoped, so it only makes sense where a
    # generate/repaint mask exists (inpaint) -- inherited by OUTPAINT_DEFAULTS
    # below via the same path as mask_blur.
    "region_prompt", "region_negative_prompt", "region_prompt_strength",
    "region_prompt_method", "region_mask_feather",
    # Seam Structure Continuity: mask-scoped (structures crossing the generate/
    # repaint boundary), so inpaint + outpaint only -- inherited by
    # OUTPAINT_DEFAULTS via the same path as region_* / mask_blur.
    "seam_structure_strength", "seam_structure_depth", "seam_structure_end",
    "seam_structure_saliency", "seam_structure_max_area",
    # Boundary Determinism Relaxation: keep-side seam band, inpaint + outpaint only.
    "boundary_relax_strength", "boundary_relax_width", "boundary_relax_noise",
    "boundary_relax_full_until", "boundary_relax_end", "boundary_relax_paste",
})

TXT2IMG_DEFAULTS: Dict[str, Any] = {
    k: v for k, v in GENERATION_DEFAULTS.items()
    if k not in _IMG2IMG_ONLY and k not in _INPAINT_ONLY
}
IMG2IMG_DEFAULTS: Dict[str, Any] = {
    k: v for k, v in GENERATION_DEFAULTS.items()
    if k not in _INPAINT_ONLY
}
INPAINT_DEFAULTS: Dict[str, Any] = dict(GENERATION_DEFAULTS)

# ---------------------------------------------------------------------------
# Outpaint (POST /generate/outpaint)
# ---------------------------------------------------------------------------
# Image spatial outpaint: place a (optionally cropped/resized) input image
# inside a LARGER canvas and generate everything outside it, preserving the
# placed region byte-exact (see core/inference/outpaint_utils.py +
# PipelineManager.generate_outpaint). Pure orchestration over the existing
# all-architecture generate_inpaint -- shares its ENTIRE parameter set
# (feature parity: LoRA/ControlNet/NAG/quant/block-swap/advanced-CFG/etc),
# built here by deriving from INPAINT_DEFAULTS rather than duplicating
# literal values (single source of truth).
OUTPAINT_DEFAULTS: Dict[str, Any] = {
    **INPAINT_DEFAULTS,
    # Outpaint's default is full-strength generation of the surrounding
    # canvas (the placed region is preserved regardless via the final pixel
    # paste, not via denoising_strength). Inpaint defaults to 0.75 (partial
    # repaint of existing content); outpaint has no existing content in the
    # generated region, so full strength is the natural default. strength<1.0
    # remains available for a more input-guided outpaint.
    "denoising_strength": 1.0,
    # NOTE: "width"/"height" are inherited from INPAINT_DEFAULTS above for
    # schema key-parity, but are NOT accepted as request parameters for
    # outpaint -- canvas_width/canvas_height fully determine the output size
    # (PipelineManager.generate_outpaint overwrites params["width"]/["height"]
    # with the resolved canvas size before delegating to generate_inpaint).
    #
    # --- Placement (new for outpaint) ---
    # Output canvas size. Rounded to the loaded architecture's latent-grid
    # alignment (see outpaint_utils.validate_and_snap_placement, align=8).
    "canvas_width": 1536,
    "canvas_height": 1536,
    # Top-left of the placed input rectangle on the canvas. 0/0 is clamped
    # into bounds server-side; the frontend is expected to compute a sensible
    # (e.g. centered) position rather than relying on this literal default.
    "place_x": 0,
    "place_y": 0,
    # Placed size on the canvas (the resize target -- the resized result IS
    # the preserved content). 0 = use the (cropped) input's native size.
    "place_width": 0,
    "place_height": 0,
    # Trim (crop) applied to the input image BEFORE placement. 0 width/height
    # = crop to the input's edge (i.e. "no trim" when x/y are also 0).
    "input_crop_x": 0,
    "input_crop_y": 0,
    "input_crop_w": 0,
    "input_crop_h": 0,
    # How the canvas outside the placed rect is pre-filled before denoising:
    # "replicate" (edge-extend the placed content outward), "reflect"
    # (mirror it outward), "mean" (solid average color of the placed
    # content), "noise" (uniform random RGB).
    "outpaint_fill_mode": "replicate",
    # mask_blur is already inherited from INPAINT_DEFAULTS (4) -- called out
    # here because outpaint's blur is OUTWARD-ONLY (the softened transition
    # band lies entirely outside the preserved rect; see
    # outpaint_utils.build_outpaint_mask), unlike inpaint's symmetric blur.
    # The numeric default is unchanged.
    #
    # Exposure-seam harmonizer (core.inference.outpaint_utils.match_generated_
    # exposure): corrects a tonal mismatch between the generated surroundings
    # and the preserved rect by comparing strips just inside/outside the rect
    # edges. Applied after generation, before the final unconditional paste,
    # so it never touches the preserved rect. On by default; disable to see
    # the model's raw, uncorrected output.
    "outpaint_seam_fix": True,
    # B1 continuity fix (SD/SDXL only; core.inference.custom_sampling's
    # outpaint_noise_init-gated x0-space projection injection): a weak
    # low-frequency color/illumination correction applied to the generate
    # region ONLY within a narrow collar near the preserved rect's boundary,
    # active mid/late in the schedule. Nudges coarse chroma/illumination
    # continuity across the seam without constraining composition/content
    # far from it. 0 disables the correction entirely.
    "outpaint_boundary_color_strength": 0.25,
    # B2 continuity fix (SD/SDXL only; core.inference.custom_sampling's
    # outpaint_noise_init-gated RePaint-style band-limited time-travel
    # resampling): after completing a denoise step inside a mid-schedule band,
    # jump back outpaint_jump_length steps by re-noising the WHOLE latent
    # (keep + generate together, so the keep region becomes a correlated
    # sample instead of an independently-pinned one) and re-denoise, repeating
    # outpaint_resample_count times total per band segment. Re-exposes the
    # generate region to the keep constraint while it is still malleable,
    # fixing content divergence (invented/unrelated composition) that B1
    # alone does not address. 1 disables resampling entirely (B1 only).
    # Costs roughly 1.5-2x the requested step count in actual denoise passes.
    # Only takes effect with a resample-compatible sampler (Euler, Euler
    # Ancestral, DDIM, DDPM); other samplers fall back to B1 only.
    "outpaint_resample_count": 2,
    # B2 jump-back length ("u", in step indices) for each resample cycle.
    "outpaint_jump_length": 4,
    # B3 continuity fix (SD/SDXL only; core.inference.custom_sampling's
    # outpaint_noise_init-gated masked self-attention KV injection): reuses the
    # existing StyleAligned/VSP-style reference-KV-injection machinery
    # (core.inference.reference_style) with a noise-matched reference composite
    # built from the preserved rect's own clean latents (not a user-supplied
    # style image), restricted to KNOWN-region tokens via spatial masking, so
    # the generate region's self-attention queries directly attend to the
    # input's own clean features instead of only following the prompt --
    # addressing content/style/palette divergence that persists even with B1+B2.
    # Strength ("gamma") scales the injected reference Key/Value; 0 disables
    # the mechanism entirely (default, until tuned).
    "outpaint_reference_strength": 0.0,
    # Boundary-outward commitment front (SD/SDXL only; EXPERIMENTAL; AUGMENTS
    # B1's x0-space projection -- core.inference.custom_sampling's
    # outpaint_noise_init-gated commit proximal): a distance-graded,
    # low-frequency, EMA-anchored proximal that makes near-boundary generate
    # cells commit their COARSE structure earlier in the schedule (following
    # the preserved rect outward) than far-interior cells, approximating an
    # autoregressive-in-space factorization of the generate region conditioned
    # on the known content. Only the low-frequency band is committed (high-
    # frequency detail keeps refining every step); this is a biased continuity
    # regularizer, not an exact sampler change. 0 disables the proximal
    # entirely (default; byte-identical to B1 alone).
    "outpaint_commit_strength": 0.0,
    # Schedule progress (0-1) fraction at which boundary-touching generate
    # cells (distance 0) start committing.
    "outpaint_commit_near": 0.35,
    # Schedule progress (0-1) fraction at which the farthest generate cells
    # (distance >= outpaint_commit_distance) start committing.
    "outpaint_commit_far": 0.80,
    # Distance (in latent cells) from the preserved rect boundary at which the
    # commit-fraction schedule saturates to outpaint_commit_far. Cells farther
    # than this all commit at outpaint_commit_far.
    "outpaint_commit_distance": 32.0,
    # Outpaint ControlNet -- PART A edge-extrapolation ControlNet (SD/SDXL
    # only; scratchpad/outpaint_controlnet_synthesis.md; core.inference.
    # outpaint_control.build_outpaint_control_image + PipelineManager.
    # generate_outpaint). Detects structures in the preserved region that
    # cross the rect boundary (a held rod, a limb, architectural lines) and
    # extrapolates them a short, confidence-tapered distance into the
    # generate region as a synthetic ControlNet, driving a general structure
    # ControlNet model (e.g. an "anytest"-style checkpoint). The same
    # confidence field also spatially gates the ControlNet's residual
    # contribution in the denoise loop, so its influence tapers to 0 with
    # distance from the boundary instead of applying uniformly. This is an
    # ENFORCEMENT tool over a GUESSED continuation geometry, not a learned
    # one -- off by default. v1: mutually exclusive with a user-supplied
    # ControlNet/LLLite (never overrides the user's own request) and forces
    # boundary_relax_paste="exact" + disables seam_structure_strength while
    # active (see PipelineManager.generate_outpaint). 0/False = off
    # (byte-identical to before this feature existed).
    "outpaint_controlnet_enable": False,
    # Conditioning mode. "edge_extrapolate" (default, PART A): detect edges in
    # the preserved region and extrapolate them a confidence-tapered distance
    # across the boundary -- drives a general "anytest"-style structure
    # ControlNet over a GUESSED geometry. "crop_mask" (PART B): build the
    # trained outpaint-native 4-channel conditioning (crop RGB + binary
    # known-mask, via core.utils.crop_mask_condition) -- requires a ControlNet
    # trained with conditioning_mode="outpaint" (4-ch), which LEARNED the
    # continuation, so no extrapolation/termination heuristic and a flat
    # (untapered) residual gate over the whole generate region.
    "outpaint_controlnet_mode": "edge_extrapolate",
    # Path to the ControlNet model checkpoint driving this feature. Required
    # (non-empty) for the feature to take effect when enabled. For "crop_mask"
    # this must be an outpaint-trained diffusers directory (4-ch conditioning).
    "outpaint_controlnet_model": "",
    # Edge/structure detector run over the preserved region only (never the
    # synthetic canvas fill): "canny" (cv2, no model download) or
    # "lineart"/"lineart_anime" (controlnet_aux, opt-in; falls back to canny
    # if unavailable offline).
    "outpaint_controlnet_detector": "canny",
    # ControlNet conditioning scale for the synthetic control image.
    "outpaint_controlnet_scale": 0.6,
    # ControlNet guidance window start (schedule progress fraction, 0-1).
    "outpaint_controlnet_guidance_start": 0.0,
    # ControlNet guidance window end (schedule progress fraction, 0-1).
    "outpaint_controlnet_guidance_end": 0.55,
    # Maximum extrapolation depth (canvas pixels) a detected boundary-crossing
    # structure is continued into the generate region before its confidence
    # tapers to 0.
    "outpaint_controlnet_depth": 160,
    # Exponent of the cosine-squared distance taper applied to each
    # extrapolated structure's confidence (higher = sharper falloff near the
    # taper depth).
    "outpaint_controlnet_taper": 2.0,
}

# ---------------------------------------------------------------------------
# Upscale (POST /generate/upscale)
# ---------------------------------------------------------------------------
# Authoritative source: Form() defaults in generate_upscale route handler.
# Upscale-only keys — not part of GENERATION_DEFAULTS (no shared txt2img/
# img2img/inpaint overlap).

UPSCALE_DEFAULTS: Dict[str, Any] = {
    "upscaler_backend": "spandrel",       # "pil" | "spandrel" | "rtx_vsr"
    "upscaler_model": None,               # filename in models/upscalers/ (required for spandrel)
    "scale_factor": 2.0,                  # 1.0-8.0
    "pil_resample": "lanczos",            # "lanczos" | "bicubic" | "nearest" (pil backend only)
    "tile_size": 512,                     # spandrel tiling; 0 = no tiling
    "tile_overlap": 32,                   # spandrel tile overlap px (feather blend)
    "rtx_vsr_quality": "high",            # "low" | "medium" | "high" | "ultra" (rtx_vsr only)
    "unsharp_enable": False,
    "unsharp_radius": 2.0,
    "unsharp_percent": 100,
    "unsharp_threshold": 3,
    # Diffusion tile upscale (upscaler_backend == "diffusion")
    "prompt": "",
    "negative_prompt": "",
    "diffusion_denoising_strength": 0.3,   # 0.05-0.9
    "steps": GENERATION_DEFAULTS["steps"],
    "cfg_scale": GENERATION_DEFAULTS["cfg_scale"],
    "sampler": GENERATION_DEFAULTS["sampler"],
    "schedule_type": GENERATION_DEFAULTS["schedule_type"],
    "seed": -1,                            # -1 = random
    "diffusion_pre_upscale_mode": "pil",   # "pil" | "model"
}

# ---------------------------------------------------------------------------
# Video generation (POST /generate/txt2vid — LTX-2.3)
# ---------------------------------------------------------------------------
# Video-only keys, kept out of GENERATION_DEFAULTS (no overlap with the
# image txt2img/img2img/inpaint parameter set). Distilled LTX-2.3 defaults.
# Constraints enforced server-side: width % 32 == 0, height % 32 == 0,
# num_frames % 8 == 1.

VIDEO_GEN_DEFAULTS: Dict[str, Any] = {
    "prompt": "",
    "negative_prompt": "",
    "width": 768,                    # divisible by 32
    "height": 512,                   # divisible by 32
    "num_frames": 121,               # (num_frames - 1) % 8 == 0
    "frame_rate": 24.0,              # output FPS
    "num_inference_steps": 8,        # distilled default
    "guidance_scale": 1.0,           # distilled default (CFG effectively off)
    "seed": -1,                      # -1 = random
    "num_videos_per_prompt": 1,
    "max_sequence_length": 1024,     # Gemma-3 prompt token budget
    "audio_enable": True,            # mux the generated audio track into the mp4
    # AP1: block-swap generation. Number of transformer_blocks kept CPU-resident
    # (weights streamed to GPU during the denoise loop). 0 = disabled (stock
    # enable_model_cpu_offload path, transformer fully GPU-resident).
    "blocks_to_swap": 0,
    # AP2: First-Block-Cache (dynamic per-step trajectory-redundancy skip). Only
    # available when the transformer runs through Ltx2BlockLoopWrapper (either
    # because blocks_to_swap > 0, or the wrapper is force-attached for FBCache
    # alone). Mutually exclusive with Block Swap (a cache hit skips the block
    # loop, desyncing the per-block swap prefetch rotation) — disabled with a
    # warning if blocks_to_swap > 0.
    "fbcache_enable": False,
    "fbcache_threshold": 0.12,       # relative-L1 first-block-residual threshold
    "fbcache_warmup_steps": 1,       # always-compute the first N steps
    # Spectrum: Adaptive Spectral Feature Forecasting (training-free acceleration),
    # wrapper-hosted (Ltx2BlockLoopWrapper). Forecasts the PRE-CFG per-branch joint
    # (video, audio) transformer output from a Chebyshev fit over actual passes,
    # skipping the whole forward on forecast steps. Mutually exclusive with FBCache
    # (Spectrum takes precedence) and with Block Swap (disabled with a warning if
    # blocks_to_swap > 0). Same knobs as the image GenerationParams schema.
    "spectrum_enable": False,
    "spectrum_w": 0.5,
    "spectrum_w_decay": 0.0,
    "spectrum_delta_cap": 0.0,
    "spectrum_m": 4,
    "spectrum_lam": 0.1,
    "spectrum_warmup_steps": 3,
    "spectrum_window_size": 4,
    "spectrum_flex_window": 0.75,
    "spectrum_tail": 0.12,
    "spectrum_max_cache": 0,
    # Per-generation component overrides (RP2b). Both unsupported on the LTX-2.3
    # video arch (accepted-but-ignored with a warning); kept here so the video
    # request schema carries the same keys as the image routes.
    "vae_path": None,
    "text_encoder_path": None,
}

TXT2VID_DEFAULTS: Dict[str, Any] = dict(VIDEO_GEN_DEFAULTS)

# Image-to-video (POST /generate/img2vid — LTX-2.3). Same parameter set as
# txt2vid plus a first-frame keyframe supplied as an uploaded image (multipart).
# The LTX2ImageToVideoPipeline resizes the keyframe to (width, height) and pins
# it as frame 0 (conditioning_mask[:, :, 0] = 1); no extra first-frame-strength
# knob is exposed since the pipeline __call__ does not accept one.
IMG2VID_DEFAULTS: Dict[str, Any] = dict(VIDEO_GEN_DEFAULTS)

# ---------------------------------------------------------------------------
# Video temporal outpaint (POST /generate/outpaint/video — LTX-2.3)
# ---------------------------------------------------------------------------
# Places a (optionally trimmed) input clip at a latent-frame offset inside a
# LONGER total timeline and generates the frames before/after, preserving the
# placed input frames byte-exact (see core.pipeline_backends.ltx2
# ._generate_vidoutpaint_ltx2 + core.pipeline.generate_vid_outpaint). Pure
# orchestration over the stock diffusers `LTX2ConditionPipeline` -- no new
# denoise loop. Non-÷32 inputs are preprocessed ONCE (center-crop/resize to a
# ÷32 working resolution) and the PREPROCESSED frames become the exact-
# preserved content (mirrors the image outpaint RESIZE convention).
OUTPAINT_VIDEO_DEFAULTS: Dict[str, Any] = {
    **VIDEO_GEN_DEFAULTS,
    # Total output timeline length (frames). Must satisfy (N-1) % 8 == 0,
    # same constraint as txt2vid/img2vid's num_frames.
    "total_frames": 121,
    # Where the (trimmed) input clip is placed within the output timeline, as
    # a PIXEL frame offset. Snapped server-side to the nearest valid latent
    # frame index's pixel start -- {0, 1, 9, 17, ..., 8m+1} -- since
    # LTX2VideoCondition.index addresses a LATENT frame, not a pixel frame.
    "input_offset_frames": 0,
    # Trim applied to the input clip BEFORE placement, in pixel frames. 0/0 =
    # no trim (use the whole clip, subject to fitting inside total_frames).
    "input_trim_start_frames": 0,
    "input_trim_end_frames": 0,
    # "regenerate" (default) = use LTX's own whole-timeline generated audio
    # as-is (the input clip's own audio is NOT preserved). "preserve_input" =
    # after generation, mux the input clip's ORIGINAL audio over its placed
    # span, keeping LTX-generated audio outside that span, with a short
    # crossfade confined to the GENERATED side of each boundary so every
    # input audio sample stays exact. Only relevant when audio_enable is
    # true; falls back to "regenerate" (with a warning) if the uploaded clip
    # has no audio stream.
    "outpaint_video_audio_mode": "regenerate",
    # When true, encode the output with FFV1 (`-pix_fmt rgb24`, no forced
    # colorspace conversion) instead of libx264, so the pasted input frames
    # are BIT-EXACT after decode (empirically verified -- see
    # utils.video_utils.save_video_with_metadata's docstring for why plain
    # libx264 "-qp 0" is NOT actually bit-exact). Audio (when present) uses
    # FLAC instead of AAC in this mode. Trade-offs: much larger file size,
    # and the mp4 will generally NOT play back in a browser's native
    # <video> element (FFV1 has no mainstream browser decoder) -- intended
    # for archival/verification of the exact preserved frames, not casual
    # playback.
    "video_lossless": False,
}

# ---------------------------------------------------------------------------
# Audio generation (POST /generate/txt2aud — ACE-Step 1.5 turbo)
# ---------------------------------------------------------------------------
# Audio-only keys, kept out of GENERATION_DEFAULTS (no overlap with the image
# txt2img/img2img/inpaint parameter set). Authoritative source:
# `core.pipeline_backends.acestep.AceStepMixin._generate_txt2aud_acestep`.

AUDIO_GEN_DEFAULTS: Dict[str, Any] = {
    "prompt": "",              # caption text (also accepted as "caption")
    "lyrics": "",
    "audio_duration": 30.0,    # seconds
    "seed": -1,                # -1 = random
    "inference_steps": 8,      # turbo distilled default
    "guidance_scale": 1.0,     # turbo is CFG-distilled; overridden to 1.0 if != 1.0
    "shift": 3.0,
    "sampler_mode": "euler",   # accepted for forward-compat; currently a no-op
    "vocal_language": "en",
    # Generation-time LoRA (same shape as every other arch's params["loras"]:
    # list of {"path": str, "strength": float, ...}). See
    # `core.pipeline_backends.acestep.AceStepMixin._load_lora_acestep`.
    "loras": [],
    # aud2aud (audio-to-audio / cover) stub keys -- not yet wired to a route
    # (Phase 4+). Kept here now so the schema key-set is stable once aud2aud
    # ships, avoiding a later SSOT migration.
    "reference_audio_path": None,
    "reference_audio_enable": False,
    "is_cover": False,
    "denoising_strength": GENERATION_DEFAULTS["denoising_strength"],
}

TXT2AUD_DEFAULTS: Dict[str, Any] = dict(AUDIO_GEN_DEFAULTS)

# ---------------------------------------------------------------------------
# Audio-to-audio COVER (POST /generate/aud2aud — ACE-Step 1.5 turbo, img2img
# analog). Repaint (inpaint analog) is out of scope -- the vendored
# generate_audio has no repaint kwargs (see
# `core.pipeline_backends.acestep.AceStepMixin._generate_aud2aud_acestep`).
# ---------------------------------------------------------------------------
# Note: distinct from the `reference_audio_path` / `is_cover` /
# `denoising_strength` stub keys left in AUDIO_GEN_DEFAULTS above -- those
# predate this implementation and do not match its actual contract
# (multipart file upload; `cover_strength` is a step-count blend, not a
# denoise-strength/start-timestep knob). Authoritative source:
# `core.pipeline_backends.acestep.AceStepMixin._generate_aud2aud_acestep`.

AUD2AUD_DEFAULTS: Dict[str, Any] = {
    "prompt": "",              # caption text (also accepted as "caption") -- the ref is re-rendered under this
    "lyrics": "",
    "seed": -1,                # -1 = random
    "inference_steps": 8,      # turbo distilled default
    "guidance_scale": 1.0,     # turbo is CFG-distilled; overridden to 1.0 if != 1.0
    "shift": 3.0,
    # Fraction of steps that keep the reference's semantic context before
    # switching to a text2music-style (silence) context (step-count blend,
    # NOT an img2img start-timestep/partial-denoise knob). Higher = closer
    # to the reference.
    "cover_strength": 1.0,
    # aud2aud sub-mode: "cover" (re-render the whole reference under a new
    # caption/lyrics) or "repaint" (regenerate only [repaint_start,
    # repaint_end) seconds of the reference, keeping the rest). See
    # `core.pipeline_backends.acestep.AceStepMixin._generate_aud2aud_acestep`.
    "mode": "cover",
    "repaint_start": 0.0,      # seconds, repaint mode only
    "repaint_end": 0.0,        # seconds, repaint mode only
    "vocal_language": "en",
    # Generation-time LoRA (same shape as every other arch's params["loras"]:
    # list of {"path": str, "strength": float, ...}). See
    # `core.pipeline_backends.acestep.AceStepMixin._load_lora_acestep`.
    "loras": [],
}

# ---------------------------------------------------------------------------
# Audio temporal outpaint (POST /generate/outpaint/audio — ACE-Step 1.5)
# ---------------------------------------------------------------------------
# Places a (optionally trimmed) input clip at a time offset inside a LONGER
# total_duration timeline and generates the audio before/and-or after it,
# preserving the placed input sample-exact to the decoded 48kHz/16-bit
# representation (see core.pipeline_backends.acestep
# ._generate_audoutpaint_acestep + core.pipeline.generate_aud_outpaint).
# This is the structural INVERSE of aud2aud's `mode="repaint"`: repaint holds
# everything OUTSIDE a window and generates INSIDE it; outpaint holds the
# window itself (the placed input) and generates OUTSIDE it (before AND/OR
# after). Derived from AUD2AUD_DEFAULTS (single source of truth) rather than
# duplicating literal values -- shares prompt/lyrics/seed/inference_steps/
# guidance_scale/shift/vocal_language/loras verbatim. `mode`/`cover_strength`/
# `repaint_start`/`repaint_end` are inherited for schema key-parity but are
# NOT accepted request parameters on /generate/outpaint/audio (outpaint has
# no cover/repaint sub-mode -- it always holds the placed span and generates
# outside it), mirroring OUTPAINT_DEFAULTS' inherited-but-unused width/height
# note above.
#
# "Exact" for audio (unlike video, which needed a lossless-encode toggle
# because H.264 is not bit-exact): the app's standard FLAC output is already
# lossless, and the placed input's waveform samples (after normalization to
# stereo/48kHz -- see AceStepMixin._acestep_normalize_stereo_48k) are
# re-spliced verbatim over its span after decoding (see
# AceStepMixin._acestep_apply_outpaint_waveform_splice), so no separate
# "lossless" flag is needed here -- the default output path is sample-exact
# to that decoded 48kHz/16-bit representation end to end. An upload that is
# not already 48kHz/16-bit stereo (e.g. 24-bit or 44.1kHz) is faithfully
# resampled/requantized once during normalization, not byte-identical to
# the original file.
OUTPAINT_AUDIO_DEFAULTS: Dict[str, Any] = {
    **AUD2AUD_DEFAULTS,
    # Total output timeline length, in seconds. Capped at 240s server-side
    # (silence-latent tiling supports arbitrary length, but very long extends
    # drift stylistically without lyrics/captions that cover the full span).
    "total_duration": 60.0,
    # Where the (trimmed) input clip is placed within the output timeline, in
    # seconds. Snapped server-side to the nearest 1/25s (the ACE-Step VAE's
    # latent frame rate) and clamped so the placed input fits inside
    # total_duration -- a warning is added if either adjustment occurs.
    "input_offset_sec": 0.0,
    # Trim applied to the input clip BEFORE placement, in seconds. 0/0 = no
    # trim (use the whole clip, subject to fitting inside total_duration).
    "input_trim_start_sec": 0.0,
    "input_trim_end_sec": 0.0,
}

# ---------------------------------------------------------------------------
# LoRA / Full-FT Training (TrainingRunCreateRequest)
# ---------------------------------------------------------------------------
# Authoritative source: backend Pydantic model.
# Frontend DEFAULT_PARAMS is aligned to these values.

TRAINING_DEFAULTS: Dict[str, Any] = {
    # Dataset
    "dataset_configs": [],
    "run_name": None,
    "training_method": "lora",
    "base_model_path": "",
    # Training parameters
    "total_steps": 1000,
    "epochs": 10,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "learning_rate": 1e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    # Optimizer
    "optimizer": "adamw8bit",
    "optimizer_is_paged": False,
    "optimizer_cautious": False,
    "optimizer_beta1": 0.9,
    "optimizer_beta2": 0.999,
    "optimizer_epsilon": 1e-8,
    "optimizer_weight_decay": 0.01,
    "optimizer_schedule_free": False,
    "optimizer_schedule_free_r": 0.0,
    "optimizer_schedule_free_weight_lr_power": 2.0,
    "optimizer_use_radam": False,
    "optimizer_stochastic_rounding": False,
    # LoRA specific
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dtype": "fp32",
    "network_type": "lora",
    # ReLoRA
    "relora_merge_every": 500,
    "relora_merge_unit": "steps",
    "restart_warmup_steps": 100,
    "optimizer_reset_strategy": "full_reset",
    "optimizer_pruning_ratio": 0.9,
    # Checkpoints & sampling
    "save_every": 100,
    "save_every_unit": "steps",
    "max_step_saves_to_keep": None,
    "sample_every": 100,
    "sample_prompts": [{"positive": "", "negative": ""}],
    "resume_from_checkpoint": "latest",
    "sample_width": 1024,
    "sample_height": 1024,
    "sample_steps": 28,
    "sample_cfg_scale": 7.0,
    "sample_sampler": "euler",
    "sample_schedule_type": "sgm_uniform",  # Fix: frontend had "uniform"
    "sample_seed": -1,
    # Debug
    "debug_latents": False,
    "debug_latents_every": 50,
    # Bucketing
    "enable_bucketing": False,
    "base_resolutions": [1024],
    "bucket_strategy": "resize",
    "multi_resolution_mode": "max",
    # Epoch-dynamic crop augmentation (SDXL only). Per (item, epoch), two independent
    # axes pick how the image is presented: crop (full image vs random crop) and bucket
    # size (largest-fitting vs smaller). Re-bucketed each epoch; forces onthefly_gpu
    # latent encoding (disk/swap caches cannot represent per-epoch crops). Defaults
    # reproduce the previous behavior (disabled). See
    # docs/EPOCH_DYNAMIC_CROP_BUCKETING_DESIGN.md.
    "crop_augment_enable": False,            # master switch
    # Mix proportions (2x2 axes):
    "crop_full_image_prob": 0.7,            # P(full image, minimal crop only)
    "crop_max_bucket_prob": 0.7,            # P(largest-fitting bucket = least downscale)
    # Random-crop controls:
    "crop_min_area_ratio": 0.25,            # crop area >= ratio * original area
    "crop_min_short_side_px": 512,          # crop short side (original px) >= this
    "crop_aspect_mode": "source",           # "source" (keep image aspect) | "free" (any aspect)
    "crop_position_mode": "random",         # "random" (any point) | "corner" (touch a corner)
    # Smaller-bucket controls:
    "crop_smaller_bucket_mode": "base_res",  # "base_res" (use smaller base_resolution) | "scale_range"
    "crop_smaller_scale_range": [0.5, 0.9],  # downscale range when scale_range / single base_res
    # Full-image (minimal crop) position:
    "full_crop_position_mode": "center",    # "center" | "fixed_corner" | "random"
    # Conditioning + seed:
    "crop_microcond_mode": "kohya",         # time_ids semantics: "kohya" = original_size is full image
    "crop_plan_seed": 0,                    # 0 = derive from global training seed
    "cache_latents_to_disk": False,         # Fix: frontend had True
    # Component-specific
    "train_unet": True,
    "train_text_encoder": False,            # Fix: frontend had True
    "unet_lr": 1e-5,
    "text_encoder_lr": 1e-6,
    "text_encoder_1_lr": None,
    "text_encoder_2_lr": None,
    # Frontend-only fields now accepted by backend
    "train_image_encoder": False,
    "image_encoder_lr": None,
    "force_recache": False,
    "reconstruction_loss_weight": 0.0,
    # Precision
    "weight_dtype": "bf16",                 # bf16: works for both LoRA and full-FT
    "training_dtype": "bf16",               # bf16 needs no GradScaler (fp16 full-FT crashes on unscale_)
    "output_dtype": "fp32",
    "vae_dtype": "fp16",                    # Fix: frontend had "fp32"
    # Full-parameter save: also embed the VAE weights into the single-file
    # checkpoint. None = per-arch default (BUNDLE_VAE_DEFAULTS_BY_ARCH below):
    # sd15/sdxl/deus comfy-layout checkpoints are consumed by A1111/ComfyUI which
    # expect first_stage_model.* baked in, so they default True; all other archs
    # default False (their loaders fall back to default VAE resolution when the
    # section is absent). Ignored for LoRA and for pixel-space MiniT2I (no VAE).
    "bundle_vae": None,
    "mixed_precision": True,
    # Attention backend selector for training (single source of truth).
    # Values: "native" (SDPA) | "flash" (FlashAttention). "sage" is refused for
    # training (no backward kernel) and downgraded to native by resolve_backend.
    "attention_backend": "native",
    # DEPRECATED compat mirror of attention_backend: kept so existing YAML/presets
    # that only set the boolean still enable flash. Derived at parse time as
    # (attention_backend != "native"); the string key above is authoritative.
    # Do NOT remove until the deprecation/cleanup phase.
    "use_flash_attention": False,
    # Attention implementation selector for training (which REGISTRY runs the
    # attention kernel; orthogonal to attention_backend, which selects WHICH
    # kernel). "conduit" routes through the unified backend/core/attention
    # dispatch (new default; enables the tq backend in SDXL/SD1.5 training).
    # "diffusers" reproduces the pre-migration set_attention_backend path
    # byte-for-byte. TRAINING-ONLY this pass (FLUX.2/Ideogram4 not yet migrated).
    "attention_impl": "conduit",
    "min_snr_gamma": 5.0,
    # Text / latent encoding
    # text_encoding_mode: "swap_onthefly" | "pre_encoded_cache" | "onthefly_gpu"
    #                   | "cpu_prefetch"
    # cpu_prefetch pins the frozen TE on CPU and runs caption encoding for
    # upcoming batches on a daemon thread, in parallel with GPU train steps.
    # Stalls (queue empty when the trainer pulls the next batch) are logged
    # at epoch end so the user can see whether CPU encode keeps up.
    "text_encoding_mode": "swap_onthefly",
    "text_encoding_swap_interval": 256,
    "text_encoding_prefetch_depth": 4,
    "latent_encoding_mode": "swap_onthefly",
    "latent_encoding_swap_interval": 256,
    # Block swap
    "blocks_to_swap": 0,
    "use_pinned_memory": False,
    "block_swap_h2d_only": False,   # H2D-only swap (FLUX.2 LoRA training: no D2H of frozen base)
    "block_swap_ring_size": 2,      # GPU weight-buffer ring slots (>=1)
    "num_optimizer_groups": 0,
    # Per-bucket activation offload dispatcher. Predicts training peak per bucket
    # (static + coef * bs * latent_area) before the forward and offloads saved
    # activations to CPU only where it fits the VRAM budget. Proactive (no OOM
    # detection), so it works on Windows WDDM which spills instead of raising.
    "activation_dispatch_enable": False,
    "activation_dispatch_margin_gb": 1.0,
    "activation_dispatch_seed_coef": 24.0e-6,
    "activation_dispatch_residual_frac": 0.85,
    "activation_dispatch_threshold_mb": 4,
    # TREAD token routing (arXiv 2501.04765) — training-only acceleration.
    # Routes a random subset of tokens through a span of transformer blocks;
    # dropped tokens bypass the span (identity transport) and are restored at
    # re-entry. Inference/sampling always runs the full network. Currently wired
    # for the Anima DiT (other archs ignore these keys). Default OFF/neutral.
    "tread_enable": False,
    # Fraction of tokens dropped from the routed span, in (0, 1). Paper default
    # selection rate is 50% (drop_ratio 0.5).
    "tread_drop_ratio": 0.5,
    # Route span [start_block, end_block): tokens are dropped at start_block and
    # reintroduced at end_block. Defaults keep the first/last two of Anima's 28
    # blocks on all tokens and route the middle span (paper routes the bulk of a
    # DiT's depth, e.g. DiT-XL uses a single route r_{0,21}).
    "tread_start_block": 2,
    "tread_end_block": 26,
    # Low-rate stochastic depth (per-batch block dropout) — training-only
    # regularization. Each step, eligible transformer blocks (front/back, outside
    # a protected middle span) are independently dropped with prob block_skip_rate;
    # executed eligible blocks rescale their residual by 1/(1-rate) so the expected
    # contribution matches the full eval network. Sampling always runs every block.
    # Currently wired for the Anima DiT (other archs ignore these keys). Default OFF.
    # 0.0 = off; capped to 0.35 (high dropout on a pretrained DiT degrades quality).
    "block_skip_rate": 0.0,
    # Protected middle span [protect_start, protect_end): NEVER dropped. Block-removal
    # studies of pretrained DiTs show middle blocks are semantically critical while
    # early/late blocks tolerate removal — so only [0, protect_start) U [protect_end,
    # num_blocks) are eligible. Defaults protect Anima's middle 16 of 28 blocks
    # (eligible = first 6 + last 6).
    "block_skip_protect_start": 6,
    "block_skip_protect_end": 22,
    # DiT-BlockSkip (arXiv 2603.20755) — training-only MEMORY-REDUCTION feature
    # for LoRA and full fine-tune training. Skips the FIRST `blockskip_front` and
    # LAST `blockskip_back` transformer blocks; only the unskipped MIDDLE blocks
    # train (LoRA injected there, or full parameters when training_method=
    # full_finetune) — the paper's cross-attention masking study shows the middle
    # blocks are semantically critical. The skipped blocks are frozen (LoRA
    # variant: no adapter; full-FT variant: requires_grad_(False), excluded from
    # the optimizer); per step their contribution is captured once by a no_grad
    # full forward as a residual feature Delta (input->output of the skipped span)
    # and re-added during the gradient forward, so backprop only flows through the
    # middle blocks — eliminating the skipped blocks' backward-activation memory
    # (and, for full-FT, their gradient + optimizer-state memory). Currently wired
    # for the Anima DiT (other archs ignore these keys). Default OFF.
    # Mutually exclusive with TREAD, block_skip_rate (stochastic depth) and
    # blocks_to_swap; unsupported for ReLoRA and ControlNet.
    "blockskip_enable": False,
    # Number of leading / trailing blocks to skip (n + m). Defaults skip 4 + 4 of
    # Anima's 28 blocks (~29% skip ratio; the paper evaluates 30/40/50%). The
    # middle (num_blocks - front - back) blocks must be >= 1.
    "blockskip_front": 4,
    "blockskip_back": 4,
    # Resolution curriculum — training-only, arch-agnostic (data-pipeline feature).
    # Warm up at a lower resolution, then switch to the target (base_resolutions) at an
    # epoch boundary. Lower resolution => fewer latent tokens => much cheaper attention
    # per step (tokens scale with area; attention ~ tokens²), so warmup steps are far
    # faster. Default OFF. To take effect, set res_curriculum_warmup_steps > 0.
    "res_curriculum_enable": False,
    # Number of warmup steps at the scaled resolution. Rounds UP to the end of the epoch
    # that contains it (switch happens at an epoch boundary, so per-epoch batch planning
    # stays intact). 0 = no warmup (feature inert even if enabled).
    "res_curriculum_warmup_steps": 0,
    # Warmup resolution = base_resolutions * this scale, each snapped to the /64 grid the
    # bucket table is defined on (reuses the existing bucket-fit logic). 0.5 => half the
    # linear size => 1/4 the tokens. Must be in (0, 1).
    "res_curriculum_warmup_scale": 0.5,
    # MNT
    "multi_noise_timesteps": 1,
    "multi_noise_mode": "independent",
    "trajectory_blend_alpha": 0.7,
    "timestep_sampling": {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0},
    # Regularization
    "regularization_type": None,
    "snr_regularization_weight": 0.1,       # Fix: frontend had 0.0
    "snr_timestep_adaptive": True,
    "snr_penalty_mode": "relu",
    "energy_regularization_weight": 0.05,   # Fix: frontend had 0.0
    "energy_timestep_adaptive": True,
    "energy_penalty_mode": "abs",           # Fix: frontend had "under"
    "energy_normalize_by_pixels": True,
    # Unified framework
    "noise_process": "auto",
    "prediction_target": "auto",
    "strict_validation": False,
    # SDXL micro-conditioning: derive time_ids (original_size / crop_top_left /
    # target_size) from the real source image + bucketing/crop, instead of the legacy
    # all-equal-to-latent-size, crop=(0,0). Default on; off restores legacy behavior.
    "sdxl_micro_conditioning": True,
    # SDXL high-spec VAE migration: swap the VAE + resize U-Net conv_in/out to the new
    # latent channel count. "none"/"sdxl" = standard 4ch (unchanged). e.g. "flux1" (16ch).
    "sdxl_vae_type": "none",
    # SDXL Text Encoder swap: replace CLIP with an alternative encoder + trainable
    # bridge adapters. "none" = standard CLIP. e.g. "siglip2_text".
    "sdxl_te_type": "none",
    "sdxl_te_hidden_layer": -2,      # which TE hidden-states layer to tap (penultimate)
    "sdxl_te_max_len": 256,          # fixed token length the new TE is padded/truncated to
    "sdxl_te_train_encoder": False,  # False = freeze TE, train adapters only; True = train both
    # Vision encoder
    "use_reference_images": False,
    "vision_encoder_path": None,
    "train_vision_encoder": False,
    "vision_encoder_lr": None,
    "gradient_routing_ve": False,
    # Param tracking
    "param_tracking": False,
    "param_tracking_interval": 100,
    # ControlNet
    "controlnet_type": "standard",
    "controlnet_pretrained_path": None,
    "controlnet_init_from_unet": True,
    "lllite_conditioning_channels": 32,
    "lllite_rank": 64,
    "condition_preprocessors": None,
    "condition_cache_mode": "on_the_fly",
    # Outpaint-native conditioning (PART B)
    "conditioning_mode": "preprocessor",
    "outpaint_crop_min_area": 0.15,
    "outpaint_crop_max_area": 0.8,
    "outpaint_edge_anchor_prob": 0.34,
    "outpaint_corner_anchor_prob": 0.33,
    "outpaint_mask_channel": True,
    "outpaint_known_loss_weight": 0.3,
    "outpaint_seam_loss_boost": 0.0,
    # Pre-flight dataset drift check + optional rescan.  4 modes:
    #   "off"   — skip entirely (default)
    #   "path"  — only detect added/missing files
    #   "smart" — path drift + caption sidecar mtime check
    #   "force" — always rescan, no drift detection
    # Also cleans up orphan latent cache files when a rescan happens.
    # See core/training/dataset_drift.py.
    "rescan_before_training": "off",

    # ---- Anima (Cosmos-Predict2 DiT) training ----
    # LoRA targets enumerated by core/models/anima/anima_lora.py.
    # Comma-separated subset of {attention, mlp, mod, llm_adapter}; the
    # AnimaLoRAAdapter dispatch normalises and applies the corresponding
    # scope flags (see core/training/lora_trainer.py:_create_adapter).
    "anima_lora_scope": "attention,mlp,llm_adapter",
    "lens_lora_scope": "img_attn,txt_attn,img_mlp,txt_mlp",
    # Lens full-parameter LR multipliers per stream.
    "lens_img_lr_factor": 1.0,
    "lens_txt_lr_factor": 1.0,
    # Ideogram 4 (flow-matching DiT) LoRA: target scope and options.
    # scope tokens: attn (to_q/to_k/to_v/to_out), mlp (feed_forward), mod (adaln).
    "ideogram4_lora_scope": "attn,mlp",
    # Also LoRA-train the unconditional transformer (asymmetric-CFG branch) with an
    # auxiliary image-only loss. Default False (train the conditional branch only).
    "ideogram4_train_uncond": False,
    "ideogram4_uncond_loss_weight": 1.0,
    # LoRA learning-rate multiplier (applied to unet_lr).
    "ideogram4_lr_factor": 1.0,
    # If False, llm_adapter is dropped from the scope regardless of the
    # csv above. Defaults to True so the LLM Adapter is fine-tuned along
    # with the DiT blocks (Phase C user request).
    "train_llm_adapter": True,

    # ---- MiniT2I (pixel-space MM-JiT, flow matching, x0 prediction) ----
    # LoRA scope tokens: attn (qkv/attn_proj), mlp (w1/w2/w3), txt_embed
    # (txt_embedder/pooled_embedder). Enumerated by core/models/minit2i/minit2i_lora.py.
    "minit2i_lora_scope": "attn,mlp,txt_embed",
    # FLAN-T5 (text encoder) LoRA scope when train_text_encoder is set with LoRA:
    # attn (SelfAttention q/k/v/o), ff (DenseReluDense wi/wi_0/wi_1/wo).
    "minit2i_te_lora_scope": "attn,ff",
    # CFG label-drop rate: per-sample probability of zeroing the text mask so the
    # model sees the mask_token (unconditional) — matches the reference label_drop_rate.
    "minit2i_label_drop_rate": 0.1,
    # LoRA learning-rate multiplier (applied to unet_lr).
    "minit2i_lr_factor": 1.0,
    # Optional override path to FLAN-T5-Large; empty -> resolve next to the model.
    "minit2i_flan_t5_path": "",
    # From-scratch only: inherit compatible weights from an existing MiniT2I model
    # (same variant) instead of pure random init. Body/proj2/embedders copy fully;
    # in/out layers copy overlapping channels when patch is unchanged. Empty = off.
    "minit2i_scratch_init_from": "",
    # From-scratch inheritance: also copy the output head (final_layer.linear) from
    # the source model when shapes match. Default off = relearn the head from scratch.
    "minit2i_inherit_final_layer": False,

    # ---- Krea 2 (single-stream flow-matching MMDiT) training ----
    # LoRA scope tokens: attn (to_q/to_k/to_v/to_gate/to_out), mlp (SwiGLU ff),
    # text_fusion (internal text-fusion sub-transformer + projector, default off),
    # proj (img_in/txt_in/final_layer/time embeds, default off). The Qwen3-VL text
    # encoder is always frozen (no TE LoRA / no TE full-FT). Enumerated by
    # core/models/krea2/krea2_lora.py.
    "krea2_lora_scope": "attn,mlp",
    # LoRA learning-rate multiplier (applied to unet_lr).
    "krea2_lr_factor": 1.0,
    # Discrete flow-matching timestep shift applied to the sampled sigma
    # (sigma' = s*sigma / (1 + (s-1)*sigma)); musubi default 2.5 @1024^2. Set <=1 to disable.
    "krea2_discrete_flow_shift": 2.5,

    # ---- REPA (Representation Alignment) — MiniT2I only ----
    # Aligns a DiT intermediate hidden state with frozen clean-image patch features
    # from a vision encoder (our anime tagger, or off-the-shelf SigLIP2) via a
    # trainable MLP projector + cosine-similarity loss, to speed up convergence
    # (Yu et al., ICLR 2025, arXiv:2410.06940). Training-only; not in the saved model.
    "repa_enable": False,
    # Encoder source: "tagger" (domain-matched anime SigLIP2) or "siglip2".
    "repa_encoder_source": "tagger",
    # Tagger model directory (empty -> auto-pick newest usable under <repo>/tagger_models).
    "repa_tagger_model_dir": "",
    # Off-the-shelf SigLIP2 repo (used when repa_encoder_source == "siglip2").
    "repa_siglip2_repo": "google/siglip2-so400m-patch14-384",
    # DiT block depth to align (0-based). -1 = auto (depth_double // 3).
    "repa_align_depth": -1,
    # Alignment loss weight (lambda) added to the diffusion loss.
    "repa_weight": 0.5,
    # Projector LR multiplier (applied to unet_lr).
    "repa_proj_lr_factor": 1.0,
    # Encoder input square resolution; 0 = follow the encoder's native image_size.
    "repa_encoder_resolution": 0,

    # ---- Anima full-parameter training: per-group LR multipliers ----
    # Applied on top of unet_lr in AnimaFullParameterAdapter. Defaults of
    # 1.0 collapse to a single effective LR; users wanting sd-scripts-style
    # finer control (lower modulation LR etc.) can dial these.
    "anima_attn_mlp_lr_factor": 1.0,
    "anima_mod_lr_factor": 1.0,
    "anima_llm_adapter_lr_factor": 1.0,

    # Gradient checkpointing (activation recompute) on the trainable modules.
    # Default True preserves prior unconditional behavior (lower VRAM, ~slower);
    # set False to trade VRAM for speed. Gated at every enable call site in
    # base_trainer via self.gradient_checkpointing.
    "gradient_checkpointing": True,
    # ---- Anima Phase D: memory optimisations ----
    # Gradient-checkpoint mode for the DiT blocks. Both default to False
    # (i.e. standard GPU-resident checkpointing). When both are True the
    # async variant wins (faster CPU<->GPU overlap).
    "cpu_offload_checkpointing": False,
    "async_cpu_offload_checkpointing": False,
    # Quantise the *frozen* base DiT to FP8 before LoRA wrap. ~50% VRAM
    # reduction on the base. Only meaningful for training_method='lora'
    # (Full FT needs trainable base weights — flag is silently ignored).
    # None | "fp8_e4m3fn" | "fp8_e5m2".
    "fp8_base_dtype": None,
    # ---- torch.compile (opt-in DiT training acceleration) ----
    # Wraps the DiT transformer's forward with torch.compile (Inductor) once,
    # after model/dtype/device + gradient-checkpointing setup, before the loop.
    # Values:
    #   "off"                        (default) — eager, no compile
    #   "default"                    — torch.compile(mode="default")
    #   "reduce-overhead"            — CUDA-graphs mode (higher VRAM)
    #   "max-autotune-no-cudagraphs" — autotuned kernels, no CUDA graphs
    # Gated to DiT archs (transformer is not None) + full-parameter FT;
    # skipped (with a log warning) for LoRA and when block swap is active
    # (blocks_to_swap > 0). Any compile/Inductor failure at runtime falls back
    # to eager. The forward is replaced in-place (module stays the same object),
    # so state_dict keys remain unprefixed and checkpoint saves are unaffected.
    "torch_compile": "off",
    # Dynamic-shape handling for torch.compile. None = auto (Dynamo detects
    # varying shapes and recompiles / promotes to dynamic as needed — best with
    # bucketing). True = force dynamic from the first compile (fewer recompiles,
    # slightly slower kernels). False = assume static (one specialised graph per
    # shape; more recompiles under bucketing).
    "torch_compile_dynamic": None,

    # ---- Online Danbooru augmentation (image-generation training) ----
    # Diffusion-side counterpart of the tagger's Danbooru augmentation. Text
    # conditioning accepts arbitrary tokens, so there is NO vocabulary
    # expansion here — the mechanism only fetches extra training images from
    # Danbooru and injects them as ordinary samples (interrupt-batch).
    # See core/training/danbooru_image_augment.py.
    "danbooru_aug_enable": False,
    # Static path: user-supplied general Danbooru queries (newline-separated;
    # each line is one query, Danbooru tag syntax, e.g. "1girl solo score:>=50").
    "danbooru_aug_queries": "",
    "danbooru_aug_weight_static": 1.0,
    # Deficiency path (auto + manual). Collects images for under-represented
    # tags to rebalance the dataset. No per-tag F1 (diffusion has none) — purely
    # dataset tag-frequency based.
    "danbooru_aug_deficiency_enable": True,      # auto: detect rare tags from dataset captions
    "danbooru_aug_deficiency_min_count": 20,     # tags in fewer than N dataset images are deficient
    "danbooru_aug_deficiency_top_k": 200,        # cap on the number of rarest tags targeted
    "danbooru_aug_deficiency_manual": "",        # manual: extra tags to top up (newline-separated)
    "danbooru_aug_weight_deficiency": 1.0,
    # Injection cadence / sizing.
    "danbooru_aug_injection_interval": 4,        # inject a Danbooru batch every N base batches
    "danbooru_aug_injection_ratio": 1.0,         # injection batch size = ratio x train batch_size
    # Collection / API settings.
    "danbooru_aug_min_score": 0,
    "danbooru_aug_max_posts_per_query": 200,
    "danbooru_aug_api_interval": 1.4,            # Danbooru TOS rate limit (seconds)
    "danbooru_aug_dl_speed_kbps": 500,
    # Download-speed safety: detect sustained throttling (Danbooru throttles before
    # a ban) and pause collection. Robust to transient dips (needs a slow streak
    # AND sustained duration before tripping). Manual resume from the UI.
    "danbooru_speed_check_enable": True,
    "danbooru_speed_degraded_kbps": 250,         # below this = "slow" sample
    "danbooru_speed_min_slow_streak": 8,         # consecutive slow downloads to trip
    "danbooru_speed_min_slow_seconds": 90,       # ...sustained at least this long
    "danbooru_speed_cooldown_seconds": 3600,     # pause duration on trip
    "danbooru_aug_buffer_size": None,            # None -> auto (max(32, 16 x batch_size))
    # Caption construction from the post's per-category tag fields.
    "danbooru_aug_include_rating_tag": False,    # prepend rating word (general/sensitive/...)
    "danbooru_aug_max_caption_tags": 0,          # 0 = keep all tags
    # Score-based quality tag (attach a quality tag derived from the post's
    # Danbooru score). Default thresholds follow the Animagine XL 3.0 convention;
    # override via danbooru_quality_tag_thresholds ("<min_score> <tag>" per line).
    "danbooru_quality_tag_enable": False,
    "danbooru_quality_tag_thresholds": "",       # "" = built-in Animagine XL 3.0 default
    "danbooru_quality_tag_attach_negative": False,  # also attach low/worst-quality tiers
    # Caption tag shuffle / dropout for injected samples. SEPARATE from the
    # per-dataset caption_processing (datasets vary per run by user intent).
    # Applied per-epoch via the same processor the datasets use. All default
    # off = fixed category order, no shuffle/dropout.
    "danbooru_aug_shuffle_tags": False,          # shuffle tags within each category
    "danbooru_aug_shuffle_keep_first_n": 0,      # keep first N tags fixed (no shuffle)
    "danbooru_aug_tag_dropout_rate": 0.0,        # per-tag dropout probability (0-1)
    "danbooru_aug_tag_dropout_keep_first_n": 0,  # keep first N tags fixed (no dropout)
    "danbooru_aug_caption_dropout_rate": 0.0,    # probability to drop the whole caption (0-1)
    "danbooru_aug_keep_tokens": 0,               # first N tokens immune to token dropout
}

# ---------------------------------------------------------------------------
# Per-architecture default timestep_sampling
# ---------------------------------------------------------------------------
# The training timestep distribution is drawn once per step by the shared
# TimestepSampler (core/training/timestep_sampler.py) and is fully user-overridable
# from the UI. Most models default to uniform [0,1]; only MiniT2I defaults to a
# logit-normal that reproduces its reference "lognorm" schedule.
#
# IMPORTANT — convention: the sampler emits t in [0,1]; each model interprets it.
# MiniT2I uses t=1 = data / t=0 = noise (inverted vs the flow models that use
# t=1 = noise). So for MiniT2I, logit_normal(mean=-0.8) concentrates t≈0.31 = the
# NOISE side (matching MiniT2IFlowMatchScheduler's mu=-0.8/sigma=0.8). Do not copy
# this sign to t=1=noise models without flipping it.
#
# Frontend (TrainingConfig.tsx) fetches this map and applies the selected model's
# entry when the base model changes (user edits still win); the backend
# (base_trainer.train) also resolves it when no timestep_sampling is supplied, so
# non-UI/API callers get the right default too. "_default" is the global fallback.
TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH: Dict[str, Any] = {
    "_default": {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0},
    # MiniT2I: t=1=data convention; mean=-0.8 biases toward the noise side.
    "minit2i": {"distribution": "logit_normal", "mean": -0.8, "std": 0.8,
                "min_timestep": 0.0, "max_timestep": 1.0},
    # Krea 2: uniform sigma sampling; the resolution schedule bias comes from the
    # discrete flow shift (krea2_discrete_flow_shift, applied in train_step_krea2).
    "krea2": {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0},
}

# ---------------------------------------------------------------------------
# Per-architecture default bundle_vae (full-parameter save VAE embedding)
# ---------------------------------------------------------------------------
# bundle_vae=None (the TRAINING_DEFAULTS value) resolves per-arch here.
# sd15/sdxl/deus comfy-layout single files are consumed by A1111/ComfyUI, which
# expect first_stage_model.* baked in — a VAE-less save silently produces
# corrupt decodes there, so they default True. All other archs' loaders fall
# back to default VAE resolution when the section is absent, so they default
# False (smaller checkpoints, shared-store VAE reuse).
BUNDLE_VAE_DEFAULTS_BY_ARCH: Dict[str, bool] = {
    "_default": False,
    "sd15": True,
    "sdxl": True,
    "deus": True,
}


def resolve_bundle_vae(value, arch: str) -> bool:
    """Resolve a possibly-None bundle_vae config value to the per-arch default.

    An explicit user boolean always wins; None looks up BUNDLE_VAE_DEFAULTS_BY_ARCH.
    """
    if value is not None:
        return bool(value)
    return bool(BUNDLE_VAE_DEFAULTS_BY_ARCH.get(
        arch, BUNDLE_VAE_DEFAULTS_BY_ARCH["_default"]))

# ---------------------------------------------------------------------------
# Tagger Training (TaggerTrainingRunCreateRequest)
# ---------------------------------------------------------------------------

TAGGER_TRAINING_DEFAULTS: Dict[str, Any] = {
    "run_name": None,
    "vision_encoder_path": "",
    "training_method": "lora",
    "lora_rank": 32,
    "lora_alpha": 16.0,
    "learning_rate": 3e-4,
    "head_lr_multiplier": 10.0,
    "optimizer": "adamw8bit",
    "warmup_steps": 100,
    "epochs": 10,
    "batch_size": 32,
    "num_workers": 4,
    "num_workers_override": None,
    # Live tag-refresh: pick up tag edits made in the UI during training without
    # restarting. Detection runs on a background thread and workers apply changes
    # via a generation-gated mmap, so iteration speed is unaffected.
    "tag_refresh_enable": False,
    "tag_refresh_interval_seconds": 60,
    "save_every_n_steps": 500,
    "save_every_n_epochs": 0,
    "keep_last_n_checkpoints": 3,
    "checkpoint_save_mode": "lora",
    "mixed_precision": "bf16",
    # FlashAttention-2 for the SigLIP2 encoder self-attention. Requires
    # mixed_precision bf16/fp16 (FA2 cannot run in fp32); falls back to SDPA
    # otherwise. Only affects the encoder (pooling head is unaffected).
    "use_flash_attention": False,
    "gradient_checkpointing": True,
    "weight_decay": 1e-4,
    "loss_function": "asl",
    "loss_gamma_neg": 4.0,
    "loss_gamma_pos": 1.0,
    "loss_clip": 0.05,
    "loss_gamma0": 4.0,
    "loss_m0": 0.2,
    "loss_beta": 2.0,
    "loss_rho": 0.5,
    "loss_label_weight": "fisher",
    "validate_every": 1,
    "val_split": 0.05,
    "val_split_mode": "percent",
    "val_fixed_size": None,
    "save_best_only": False,
    "vocab_min_count": 10,
    # When building the vocabulary, resolve tag categories that are absent from
    # the local Danbooru taglist (taglist/) against the Gelbooru supplement
    # (taglist_gel/). Danbooru always takes precedence; Gelbooru only fills tags
    # Danbooru does not know, so enabling this is non-destructive and simply
    # reduces the number of "Unknown" category tags. Requires the taglist_gel/
    # directory to exist (silently skipped if absent).
    "vocab_use_gelbooru_categories": True,
    "excluded_categories": [],
    "ban_tags": "",
    "use_tag_aliases": False,
    "save_base_model": False,               # Fix: frontend had True
    # Quality-tag loss masking when a quality tag is present on a sample:
    #   "intra_group" (default) — mask sibling tags within the same group
    #                              (best/high/normal/medium share gradients;
    #                              low/bad/worst share gradients).  Trains
    #                              the cross-group good-vs-bad distinction.
    #                              Safer when intra-group prevalence imbalance
    #                              or annotation noise is significant.
    #   "cross_group"           — all non-positive quality tags train as
    #                              negatives.  Correct only when intra-group
    #                              labels are truly mutually exclusive.
    "quality_masking_mode": "intra_group",
    "cls_dim": None,
    "hidden_proj_dim": None,
    "init_head_from": None,
    # LR matrix (conditional inference) — built once at training start
    # when enabled.  See backend/core/tagger/lr_matrix_builder.py.
    "build_lr_matrix_on_start": False,
    "lr_top_anchors":            10000,
    "lr_top_targets":            1000,
    "lr_threshold":              1.0,
    "lr_min_anchor_count":       10,
    # Pre-flight dataset drift check + optional rescan.  4 modes:
    #   "off"   — skip entirely (default)
    #   "path"  — only detect added/missing files (~5 min for 3M items)
    #   "smart" — path drift + caption sidecar mtime check (catches
    #             in-place caption edits — adds a stat per sidecar)
    #   "force" — always rescan, no drift detection (most expensive)
    # See core/training/dataset_drift.py.
    "rescan_before_training": "off",
    # Training-time F1 metrics.
    # N2 (eval_every_n_steps) < N1 (threshold_search_every_n_steps).
    # N1 runs _find_best_threshold() on the rolling buffer (27 calls).
    # N2 runs _compute_f1_macro() once with the current threshold (1 call).
    "train_f1_eval_every_n_steps": 100,
    "train_f1_threshold_search_every_n_steps": 500,
    "train_f1_initial_threshold": 0.35,
    # Number of recent training batches kept in the rolling buffer.
    # Memory ≈ buffer_batches × batch_size × num_tags × 3 bytes (fp16 probs + bool labels).
    "train_f1_buffer_batches": 16,
    # Per-tag threshold metrics saved alongside each checkpoint as
    # ``{name}_tag_metrics.npz``.  Set to False to disable.
    # Memory: ≈ 4 × vocab_size × n_bins × 4 bytes (two epoch histogram slots).
    "save_tag_metrics": True,
    # Probability band considered "hard" for hard_rate computation.
    "hard_rate_lo": 0.25,
    "hard_rate_hi": 0.75,
    # Online Danbooru augmentation
    "enable_danbooru_augmentation": False,
    # ---- Query mode (first-class collection mode) ----
    # The static user-query path is now an independent, toggleable mode on equal
    # footing with vocab-expansion (surveyor) and low-F1. When query_expand is on,
    # the queries' tag tokens / wildcards are resolved via the Danbooru tags API
    # (name_matches) into concrete tags: vocab-absent ones are added to the
    # vocabulary, and ALL resolved tags are collected PER-TAG (round-robin within
    # the Query pool, bounded by danbooru_query_weight_static) so a wildcard that
    # resolves to N tags contributes N collection units — not one. With
    # query_expand off, the path keeps the legacy per-query-string image collection.
    "danbooru_query_enable": True,          # default on for backward compat; now independently toggleable
    "danbooru_query_expand_enable": False,  # resolve query tags -> per-tag collection + vocab expansion
    "danbooru_query_new_tag_min_count": 200,   # post_count floor for adding a resolved tag (main guard)
    "danbooru_query_resolve_top_k": 50,        # per-query cap on resolved tags (0 = unlimited)
    "danbooru_query_max_expanded_tags": 0,     # run-wide cap on NEW tags added via query (0 = unlimited)
    "danbooru_query_expand_categories": [0, 3, 4],  # eligible categories (general/copyright/character)
    "danbooru_query_resolve_interval": 3600,   # seconds between wildcard re-resolutions (API throttle)
    # Per-tag per-epoch collection cap (0 = unlimited). Bounds how many posts a
    # single resolved query tag contributes each epoch so a high-post_count tag
    # does not monopolise the injected batches. Reset each epoch (re-collection
    # next epoch is NOT blocked). Mirrors danbooru_cooc_collect_per_epoch.
    "danbooru_query_collect_per_epoch": 0,
    "danbooru_tags": "",              # newline-separated queries (use !tag or -tag to exclude)
    "danbooru_injection_interval": 4, # interrupt-batch every N base steps
    "danbooru_injection_batch_size_ratio": 1.0,  # 1.0 = full batch, 0.5 = half, etc.
    "danbooru_min_score": 0,
    "danbooru_max_posts_per_query": 200,
    "danbooru_api_interval": 1.4,
    "danbooru_dl_speed_kbps": 500,
    # Download-speed safety (throttle/ban avoidance): pause collection on sustained
    # slowdown; robust to transient dips; manual resume from the UI.
    "danbooru_speed_check_enable": True,
    "danbooru_speed_degraded_kbps": 250,
    "danbooru_speed_min_slow_streak": 8,
    "danbooru_speed_min_slow_seconds": 90,
    "danbooru_speed_cooldown_seconds": 3600,
    "danbooru_buffer_size": None,     # None → auto (2 × batch_size)
    "danbooru_vocab_expand": False,
    "danbooru_new_tag_min_count": 200,
    # Per-tag per-epoch collection cap for surveyor-discovered new tags (0 =
    # unlimited). Same rationale/semantics as danbooru_query_collect_per_epoch.
    "danbooru_new_tag_collect_per_epoch": 0,
    # Per-category post_count threshold overrides for the surveyor (category code
    # → min). Empty = use danbooru_new_tag_min_count for all. Lets copyright use a
    # higher bar than character (copyright post counts dwarf individual chars).
    "danbooru_new_tag_min_count_by_cat": {},
    "danbooru_new_tag_lookback_days": 90,
    "danbooru_new_tag_categories": [0, 3, 4],
    "danbooru_new_tag_survey_interval": 3600,
    # LRU cap on the dynamic new-tag query list (surveyor-discovered tags that
    # keep being collected across epochs/resumes). 0 = unlimited. When >0, the
    # least-recently-collected tag is evicted once the cap is exceeded.
    "danbooru_max_dynamic_tags": 0,
    # Collection-path weights (weighted random selection among available paths)
    "danbooru_query_weight_static": 1.0,   # user static queries
    "danbooru_query_weight_new_tag": 1.0,  # surveyor-discovered new tags (vocab expansion)
    "danbooru_query_weight_low_f1": 1.0,   # low-F1 existing vocab tags (deficiency collection)
    # Low-F1 deficiency collection: existing vocab tags whose per-tag F1 is below
    # threshold are collected. Targets recomputed at the train-F1 threshold-search cadence.
    "danbooru_low_f1_enable": False,
    "danbooru_low_f1_threshold": 0.5,  # tags with valid F1 below this are targeted (NaN excluded)
    "danbooru_low_f1_top_k": 500,      # cap on number of worst-F1 tags targeted
    "danbooru_low_f1_min_posts": 50,   # min Danbooru page-1 posts to include a tag (else skipped)
    # Per-tag per-epoch collection cap for low-F1 tags (0 = unlimited). Same
    # rationale/semantics as danbooru_query_collect_per_epoch.
    "danbooru_low_f1_collect_per_epoch": 0,
    # Train-count deficiency collection: rebalances under-exposed tags using the
    # cumulative per-tag training-exposure count (TagMetricsAccumulator.tag_count).
    # A tag is "under-exposed" when its actual cumulative count is well below what
    # its CURRENT per-epoch rate implies for the elapsed epochs — i.e. it joined
    # late or its image set grew mid-run. Catches the "new character went viral
    # and was added mid-training" case; complementary to low-F1 (performance) and
    # distinct from genuinely-rare-but-stable tags. Requires training-F1 metrics
    # cadence OR this flag for tag_count to keep accumulating.
    "danbooru_train_count_enable": False,
    "danbooru_train_count_top_k": 500,           # cap on number of worst-deficit tags targeted
    "danbooru_train_count_min_deficit_ratio": 0.3,  # target tags >= this fraction under-exposed
    "danbooru_train_count_min_per_epoch": 10,    # ignore tags with fewer exposures/epoch (noise floor)
    "danbooru_train_count_min_posts": 50,        # min Danbooru page-1 posts to include a tag (availability)
    "danbooru_train_count_collect_per_epoch": 0, # per-tag per-epoch collection cap (0 = unlimited)
    "danbooru_query_weight_train_count": 1.0,    # weighted-path weight for the train-count path
    # Score-based quality tag (attach a quality tag derived from the post's
    # Danbooru score to the label set; only trains tiers present in the vocab).
    # Default thresholds follow the Animagine XL 3.0 convention; override via
    # danbooru_quality_tag_thresholds ("<min_score> <tag>" per line).
    "danbooru_quality_tag_enable": False,
    "danbooru_quality_tag_thresholds": "",       # "" = built-in Animagine XL 3.0 default
    "danbooru_quality_tag_attach_negative": False,  # also attach low/worst-quality tiers
    # Co-occurrence vocab discovery: add vocab-absent tags that appear in
    # collected posts >= cooc_min_count times (category from the post fields,
    # filtered to danbooru_new_tag_categories). Created-at independent, so it
    # catches old tags missing from the training vocab. Requires vocab_expand.
    "danbooru_cooc_expand_enable": False,
    "danbooru_cooc_min_count": 50,
    # Categories eligible for co-occurrence discovery — independent of the
    # surveyor's categories, so a character-only surveyor net still lets the
    # accompanying copyright/general tags be added when they co-occur.
    "danbooru_cooc_categories": [0, 3, 4],
    # Co-occurrence ACTIVE collection: once a tag is promoted by cooc, also
    # collect it directly (order:random) so it trains across epochs — but only
    # lightly. A low weight + small per-epoch quota keeps it balanced and avoids
    # over-reinforcing the companion co-occurrence. weight 0 disables it (cooc
    # then only expands the vocab, as before).
    "danbooru_query_weight_cooc": 0.1,       # weighted-path weight (vs new_tag 1.0)
    "danbooru_cooc_collect_per_epoch": 50,   # per-tag per-epoch collection quota
    "danbooru_cooc_order_random": True,      # query with order:random for diversity
}
