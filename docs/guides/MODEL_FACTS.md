# MODEL_FACTS

Per-architecture parameterization facts for a generic arch-maintainer agent. Every
cell is derived from code; anchors are relative repo paths. Cells marked
`(verify)` were not fully confirmed from code and must be checked before relying on
them. No subjective performance claims.

## Facts table

| arch | family | text encoder(s) | prediction / noise | latent/pixel + VAE | CFG convention | attention (infer / train) | single-file formats + completion sources | training adapter + constraints |
|------|--------|-----------------|--------------------|--------------------|----------------|----------------------------|-------------------------------------------|--------------------------------|
| sd15 | U-Net | CLIP ViT-L/14 (`CLIPTextModel`, 77 tok, 768) | epsilon / ddpm; v-pred via modelspec/state-dict `v_pred` marker | latent, `AutoencoderKL` 4ch, sf 0.18215 | `abs(cfg-1)>1e-5` else CFG short-circuits to cond-only | conduit; native only (head_dim 40/80/160 refuse flash/sage/tq) / native | `.safetensors` single-file (embedded VAE optional), diffusers dir; size <6GB heuristic | `sd15_adapter.py`; TE trainable optional |
| sdxl | U-Net | CLIP ViT-L/14 (768) + OpenCLIP ViT-bigG (`CLIPTextModelWithProjection`, 1280, penultimate + pooled) | epsilon / ddpm; v-pred via modelspec/`v_pred` | latent, `AutoencoderKL` 4ch, sf 0.13025 | same as sd15 (`abs(cfg-1)>1e-5`) | conduit; tq/flash/sage usable (head_dim 64) infer / native (diffusers dispatch) train | `.safetensors` (>6GB heuristic or `XL` class), diffusers dir | `sdxl_adapter.py`; dual-TE LRs; time_ids micro-cond |
| zimage | DiT | Qwen2.5-1.5B-Instruct causal LM (chat template, penultimate, 1536) | velocity / flow (`FlowMatchEulerDiscreteScheduler`) | latent, FLUX VAE 16ch (SDXL 4ch VAE if in_ch==4), sf 0.3611 | `abs(cfg-1)>1e-5 and abs(cfg)>1e-5`; cfg_truncation can drop late steps to 1.0 | conduit; tq / tq (head_dim 128, GQA) | comfy fused-qkv or sushiUI official split-qkv single-file; base VAE/TE/tokenizer/scheduler from hub `Tongyi-MAI/Z-Image-Turbo` | `zimage_adapter.py`; TE frozen; frame-dim unsqueeze |
| flux2 | MM-DiT | Qwen3 causal LM (`Qwen3ForCausalLM`, 512 tok, hidden states layers 9/18/27 concat) | velocity / flow | latent, `AutoencoderKLFlux2` (from Apache-2.0 FLUX.2-klein-4B `vae`) | `cfg>1.0 and not distilled`; distilled uses guidance vector not CFG | diffusers set_attention_backend dispatch; native / native (no tq entry) | transformer-only `.safetensors` + sharded index; base repo auto-detected by single-layer count (24/36/48 → klein-4B/9B/base-4B); VAE always from 4B store; embedded VAE/TE reattached if present | `flux2_adapter.py`; 8 dual + 48 single blocks; TE (Qwen3) frozen unless `train_text_encoder` |
| ideogram4 | DiT (dual transformer) | Qwen3-VL (`Qwen3VLModel`, weight-only FP8; 13-layer features → 53248-dim) | velocity / flow | latent, `AutoencoderKLFlux2` 32ch (128ch packed flat-seq) | asymmetric CFG: conditional `transformer` + separate `unconditional_transformer` (zeroed-text branch); `guidance_scale` default 7.0 | diffusers dispatch; native / native (head_dim 256; no tq entry) | diffusers-layout dir with `transformer/` + `unconditional_transformer/` + `text_encoder/` (FP8) + `vae/`; combined single-file via `unconditional_transformer.` prefix; fused-qkv → split remap | `ideogram4_adapter.py`; TE frozen; optionally trains unconditional transformer; LoRA over frozen Fp8Linear |
| lens | MM-DiT | GPT-OSS MoE (`LensGptOssEncoder`, 24-layer, multi-layer features, mxfp4) | velocity / flow | latent, `AutoencoderKLFlux2` | standard CFG blend; `guidance_scale` default 4.0 | conduit; tq / tq (head_dim 64) | diffusers dir (`LensPipeline`/`LensTransformer2DModel`); net.* full-FT single-file; own `vae/` else FLUX.2-klein-4B vae store | `lens_adapter.py`; GPT-OSS TE always frozen; VAE frozen |
| minit2i (b16/l16) | MM-DiT (MM-JiT) | FLAN-T5-Large (`T5`, frozen) | sample (x0) / flow (`MiniT2IFlowMatchScheduler`) | pixel-space (no VAE) by default; optional latent VAE variant (`is_latent` when vae_type != none) | plain CFG with `cfg_interval`; `cfg_scale` default 6.0 | conduit; b16 tq (head_dim 64) / l16 native (head_dim 52→56 padded) — same infer/train | diffusers variant dir (`transformer/` + `scheduler/`), single-file (variant auto-detected), `scratch:minit2i:` sentinel; FLAN-T5 from explicit path / sibling `flan-t5-large` dir / hub `google/flan-t5-large` | `minit2i_adapter.py`; TE frozen; pixel variant skips VAE decode |
| anima | DiT (Cosmos-Predict2 style) | Qwen3-0.6B (`Qwen3Model`) + 6-layer LLM Adapter; T5 tokenizer feeds adapter target ids | velocity / flow (rectified flow) | latent, Qwen-Image VAE (`AutoencoderKLQwenImage`-style, 16ch) | (verify) standard CFG via `_anima_encode_nag_neg` | conduit tq infer / native train (`attn_mode` torch/flash blocks tq) | split-files layout (`split_files/diffusion_models|text_encoders|vae`) or single DiT `.safetensors`; Qwen3 + Qwen-Image VAE auto-discovered by filename patterns | `anima_adapter.py`; DiT + optional LLM-Adapter-only training; Qwen3 TE + VAE frozen |
| krea2 | MM-DiT (single-stream) | Qwen3-VL-4B-Instruct (frozen) | velocity / flow (rectified flow) | latent, `AutoencoderKLQwenImage` 16ch (latents_mean/std) | `guidance = cfg_scale - 1` (default cfg 4.5); distilled/turbo disables CFG (guidance 0) | conduit (tq usable, GQA) (verify infer/train head_dim) | diffusers dir (`Krea2Pipeline`), transformer-only dir (auto-complement), single-file (diffusers/raw/comfy/sushiUI TE+DiT combined); TE `Qwen/Qwen3-VL-4B-Instruct`, VAE `Qwen/Qwen-Image` `vae` (env `KREA2_TE_DIR`/`KREA2_VAE_DIR` overrides) | `krea2_adapter.py`; transformer only, Qwen3-VL TE ALWAYS frozen (`train_text_encoder` rejected), VAE frozen; train_runner forces bf16 |
| ltx2 | joint video+audio DiT (`LTX2VideoTransformer3DModel`) | Gemma-3 text encoder (frozen) | velocity / flow (`LTX2Pipeline`/`LTX2ImageToVideoPipeline`, txt2vid + img2vid) | 5D video latent (`[T,H,W]`) via LTX video VAE (tiling enabled) + separate audio VAE/vocoder | plain CFG (`guidance_scale`); img2vid pins frame 0 via `conditioning_mask` | n/a (own pipeline backend, not conduit-routed) | not in the single-file completion matrix above (own loader) | own trainer ops (`ltx2_ops.py`); see row-level notes for AP1-3 speed/lightweight features |

## VRAM management: keep_models_hot (opt-in, all image archs except ltx2)

Post-load CPU offload and generation-time sequential component staging (text
encoder(s) -> GPU -> encode -> CPU, denoiser -> GPU -> denoise -> CPU, VAE ->
GPU -> decode -> CPU, one component on GPU at a time to bound peak VRAM) is
already implemented for every architecture: the SDXL reference path in
`backend/core/vram_optimization.py`, the 7 DiT archs via per-arch `_move`
helpers in `pipeline_backends/*.py`, and LTX-2.3 via diffusers accelerate's
`model_cpu_offload_seq` hooks.

`keep_models_hot` (`GenerationParams`, default `false`) is an additional,
opt-in layer on top of that staging: when a queue runs consecutive
generations on the same model, it skips the ->CPU offload at the end of a
generation and the ->GPU stage at the start of the next one for components
it is safe to leave GPU-resident, cutting re-staging calls between queued
generations. Shared state/logic lives in `backend/core/keep_hot.py` and is
reused verbatim by SD1.5/SDXL (`pipeline.py`) and all 7 DiT image archs
(`pipeline_backends/{flux2,zimage,anima,lens,krea2,ideogram4,minit2i}.py`).

- **Eligibility per component**: text encoder(s) and VAE are eligible
  whenever they are not doing CPU inference; the denoiser (U-Net/transformer)
  is only eligible when there is NO LoRA applied and the arch is NOT
  block-swapped for this generation.
- **model_key** = `(checkpoint, sorted LoRA path+weight set,
  unet_quantization, text_encoder_quantization, cpu_text_encoding,
  weight_dtype)`. Any change invalidates the resident set and forces a full
  offload before staging the new request.
- **VRAM guard**: a component is kept resident only if `free_vram -
  1.5GB headroom >= component_bytes`; otherwise it falls back to a normal
  offload for that component and appends a `warnings[]` entry
  (`keep_hot_vram_guard` / `keep_hot_no_cuda`).
- **Exception path**: any exception during generation always forces a full
  offload and clears residency for every component, regardless of what was
  requested.
- **Per-arch exceptions**: `lens` never keeps its text encoder hot (it frees
  the TE to `None` every generation to reclaim ~9.7GB of untracked mxfp4
  buffers). `minit2i` gates both TE and transformer on no-LoRA (its LoRA
  wraps both) and has no VAE component (pixel-space). `ideogram4` keeps its
  conditional + unconditional transformer pair together as one residency
  unit.
- **ltx2 is excluded**: `keep_models_hot` is not plumbed to the video
  generation request path; LTX-2.3 uses diffusers accelerate's
  `enable_model_cpu_offload` hooks, and its longer denoise loop makes
  per-generation re-staging overhead comparatively small.
- **Frontend**: the generation queue (`GenerationQueueContext`) sets
  `keep_models_hot = true` on every queued item except the last one in a
  back-to-back run on txt2img/img2img/inpaint; the last item sends `false`
  so VRAM is released at queue end. The frontend does not compare models
  itself — the backend's `model_key` invalidation handles a mid-queue model
  change safely.
- **Robustness**: FLUX.2 and Z-Image's post-generation component offload
  (text_encoder/transformer/VAE -> CPU) is now wrapped in a `finally` block
  (previously happy-path only, so an exception during denoise/decode could
  strand the transformer on GPU); this brings them in line with the other
  archs, which already offloaded in a `finally`-guarded cleanup.

## VAE decoder facts per family

Properties that decide which decode-side features apply to an architecture.
Measurements, the non-locality decomposition and the two tiling options are in
`docs/guides/VAE_DECODE_BEHAVIOR.md`; only the structural facts are here.

| VAE class | Used by | latent ndim | `nn.GroupNorm` in decoder | mid-block attention | decoder-only fine-tune (`vae_decoder`) |
|---|---|---|---|---|---|
| `AutoencoderKL` (4ch) | sd15, sdxl, zimage-when-`in_ch==4` | 4-D | **30** | 1 | supported (the exercised path) |
| `AutoencoderKL` (16ch, FLUX.1) | zimage | 4-D | **30** | 1 | supported (untested) |
| `AutoencoderKLFlux2` | flux2, lens, ideogram4 (32ch) | 4-D | **30** | 1 | supported (untested); 2×2 patchify + latent BatchNorm live *outside* `decode` |
| `AutoencoderKLQwenImage` (16ch) | anima, krea2 | **5-D** `(B,C,T,H,W)` | **0** (RMSNorm over channels) | 1 | **not offered** for `vae_source: "model"` (5-D `_encode` vs a 4-D pixel batch) |
| `AutoencoderKLLTXVideo` | ltx2 | **5-D** | n/a (video VAE, not measured) | n/a | **not offered** |
| none | minit2i (default variants) | pixel-space | n/a | n/a | **not offered** (no VAE) |

Consequences:

- `vae_tile_global_norm` is a **bit-exact no-op** on the Qwen-family
  autoencoder (zero GroupNorms) and is gated off there rather than costing 2×
  decode time for an identical image.
- The decoder receptive-field term is extinguished by **14–16 latent cells
  (112–128 px)** of real, discarded context (`vae_tile_mode: "context"` margin
  default 16). For FLUX.2's packed grid the equivalent is inferred, not
  measured; the module carries a runtime scale-mismatch guard.
- Wrapper objects (`SDXLVAEWrapper` / `FluxVAEWrapper` / `PidVaeWrapper`) have
  no `.decoder` of their own; both decode features install on the inner
  autoencoder. `_apply_vae_tiling` walks `vae` **and** `real_vae` for this
  reason (`backend/core/pipeline.py:1506`).

## Style transfer: training-free reference-image KV-injection (opt-in, off by default)

Reference-image style transfer (StyleAligned / Visual-Style-Prompting family):
a style reference image is VAE-encoded and forward-noised to each active
denoise step's sigma, run through the transformer once to CAPTURE the
post-norm/post-RoPE image-token Key/Value (and Query, where wired) per
attention block, then the target's conditional forward INJECTS the
(frequency-scaled + AdaIN-aligned) reference K/V onto its own image-token K/V
before attention. No weights change, no LoRA, no training; the UNCOND branch
is untouched. Default OFF (no reference image selected) is byte-identical to
a generation without style transfer.

- **Arch-agnostic core**: `backend/core/inference/reference_style.py` —
  `StyleTransferConfig`, `inject_kv`, `cross_batch_adain_qk`, `make_ref_value`,
  `frequency_scale_vector`, `StyleContext`. Ports the community
  `ComfyUI-Krea2-StyleTransfer` reference node's math verbatim; per-arch
  wiring lives outside this module (where to hook, how to slice the
  image-token region, how to build the reference forward's position ids).
- **Krea2**: `backend/core/models/krea2/vendor/transformer.py` (attention
  hook) + `backend/core/pipeline_backends/krea2.py` (`krea2_pipeline_ops.py`
  capture/inject orchestration). RoPE present (`axes_dims_rope`), so
  `frequency_scale_vector`'s frequency-content suppression applies.
- **SD1.5 / SDXL**: `backend/core/inference/attention_processors.py`
  (`UnifiedAttnProcessor`, both backends' attention layers) +
  `backend/core/inference/custom_sampling.py` +  `backend/core/pipeline.py`.
  The U-Net has no RoPE, so the frequency-scale vector degenerates to a
  ones-vector and StyleAligned's original self-attention-layer selection is
  used instead of a RoPE axis split.
- **FLUX.2**: `backend/core/inference/style_flux2.py` +
  `backend/core/pipeline_backends/flux2.py`. Injects in both the dual-stream
  blocks (`transformer.transformer_blocks`) and the single-stream blocks
  (`transformer.single_transformer_blocks`). Style transfer is mutually
  exclusive with FLUX.2 Image-Edit `ref_images` conditioning and with
  NAG/NegPip (style wins when both are requested).
- **All 3 generation modes** (txt2img/img2img/inpaint) are wired for every
  arch above.
- **Frontend**: a "+ Style Transfer" sub-mode of the shared
  `ControlNetSelector` component (`frontend/src/components/common/ControlNetSelector.tsx`),
  sibling to the existing "+ Reference Guide" sub-mode; entries carry
  `is_style_transfer: true` inside the same `controlnets[]` array (keeps the
  reference-image upload path) and expose Style Strength (`ref_k_strength`),
  AdaIN strength (`style_adain_strength`/`adain_strength`), block range, and
  a 0-1000 step range (same convention as LoRA step gating).
- **Knobs not yet exposed in the UI** (carried at their reference-node
  defaults in `StyleTransferConfig`): `value_mode`, `ref_value_mix`,
  multi-reference support, `late_release`, `rope_offset`.

## Row-level notes

- **sd15** — Simplest U-Net path. Detection falls through to sd15 when nothing else
  matches; safetensors <6GB → sd15, >6GB → sdxl. v-prediction detected from
  ModelSpec metadata, a `v_pred` state-dict tensor, or legacy `prediction_type`.
- **sdxl** — Dual text encoders concatenated to 2048-dim; TE2 uses penultimate
  hidden state + pooled output; `time_ids` = [orig_h, orig_w, crop_top, crop_left,
  target_h, target_w].
- **zimage** — Flow matching, velocity target, no Min-SNR weighting. Single-file
  loader normalizes both genuine Comfy (fused qkv, single-res embedders) and
  sushiUI full-FT (official split-qkv, multi-res `all_x_embedder`) layouts; in_ch
  auto-detected from `x_embedder` shape selects FLUX (16ch) vs SDXL (4ch) VAE.
- **flux2** — Repo implementation uses a Qwen3 causal LM text encoder (not the
  upstream FLUX text stack); concatenates Qwen3 hidden states from layers 9/18/27.
  Distilled checkpoints inject a guidance vector instead of running CFG. VAE always
  pulled from the Apache-2.0 4B store regardless of transformer variant.
- **ideogram4** — Only architecture bundling two transformers (conditional +
  unconditional); asymmetric CFG zeroes the unconditional text branch. FP8
  weight-only Qwen3-VL and FP8 transformer linears; head_dim 256 keeps it on native
  attention. VAE is `AutoencoderKLFlux2` at 32 latent channels.
- **lens** — GPT-OSS mxfp4 text encoder permanently holds ~9.7 GB VRAM while loaded
  (packed FP4 buffers untracked by PyTorch, cannot be moved to CPU). VAE falls back
  to the shared FLUX.2-klein-4B vae store when the model ships none.
- **minit2i** — Pixel-space MM-JiT: default variants have no VAE and decode is a
  tensor→image passthrough (`is_latent=False`); a latent VAE variant exists. x0
  (sample) prediction under flow matching. b16 head_dim 64 routes tq; l16 head_dim
  52 pads to 56 and stays native. `scratch:minit2i:<variant>:<vae_type>` sentinel
  builds from scratch in memory.
- **anima** — Cosmos-Predict2-style DiT with AdaLN-LoRA; Qwen3-0.6B TE plus a
  6-layer LLM Adapter (T5 tokenizer produces the adapter's target input ids).
  Training can be restricted to the LLM Adapter only; TE and Qwen-Image VAE frozen.
  Inference attention routes tq via conduit; training `attn_mode` (torch/flash)
  blocks tq.
- **krea2** — Single-stream MMDiT with rectified flow. UI `cfg_scale` maps to Krea
  `guidance = cfg_scale - 1`; the distilled/turbo checkpoint sets guidance 0 (no
  CFG). Qwen3-VL-4B TE is always frozen and TE training is explicitly rejected;
  train_runner forces bf16.
- **ltx2** — Video (+ optional audio) generation, not part of the 9-architecture
  image roster; loaded/routed separately from `model_loader.py`'s image-model
  detection. All speed/lightweight features below are opt-in (default OFF) and
  apply to both txt2vid and img2vid.
  - **Generation, block swap** (`blocks_to_swap`, `backend/core/pipeline_backends/ltx2.py`,
    `backend/core/models/ltx2_block_loop_wrapper.py`): streams `transformer_blocks`
    CPU↔GPU during the denoise loop via `Ltx2BlockLoopWrapper` + `TransformerBlockOffloader`
    (`h2d_only=True`, inference-only, `supports_backward=False`). Mutually exclusive
    with FBCache and Spectrum (a block-swap-active transformer cannot take a
    cache-hit/forecast-skip early return without desyncing the swap prefetch
    rotation).
  - **Generation, FBCache** (`fbcache_enable`/`fbcache_threshold`/`fbcache_warmup_steps`,
    `_ltx2_build_fbcache`): first-block-cache over the joint (video, audio)
    residual. Disabled whenever `blocks_to_swap > 0` or `spectrum_enable` is set
    (Spectrum takes precedence, same redundancy target).
  - **Generation, Spectrum/SFF** (`spectrum_enable` + `spectrum_m/lam/w/w_decay/
    delta_cap/warmup_steps/window_size/flex_window/tail/max_cache`,
    `_ltx2_build_spectrum`): Chebyshev output forecasting for both the video and
    audio streams, hosted in `Ltx2BlockLoopWrapper`; takes precedence over FBCache.
    Mutually exclusive with block swap; also disabled if Spatio-Temporal Guidance
    (`stg_scale`/`audio_stg_scale`) is set, since forecasting assumes exactly one
    transformer call per step. `w_decay`/`delta_cap` are separately opt-in (0.0 = off).
  - **Training, block swap + ring-buffer 8-bit optimizers**
    (`backend/core/training/base_trainer.py::_fused_backward_target_module`):
    block swap and `adamw8bit_ringbuffer`/`lion8bit_ringbuffer` now compose for
    LTX-2.3 (and other DiT archs without a `self.unet`); previously the fused
    backward path crashed on `self.unet is None`.
  - **Training, TREAD token routing** (`tread_enable`/`tread_drop_ratio`/
    `tread_start_block`/`tread_end_block`, `backend/core/training/ops/ltx2_ops.py`):
    drops/routes tokens through a reduced-token span (arXiv 2501.04765); the
    LTX-2.3 implementation is exact for its per-sample-scalar timestep (broadcast
    modulation) and gathers only `video_rotary_emb`. Only installs
    `Ltx2BlockLoopWrapper` for training when an AP3 feature (TREAD or BlockSkip)
    is enabled. Composes with block swap; mutually exclusive with BlockSkip.
  - **Training, DiT-BlockSkip** (`blockskip_enable`/`blockskip_front`/`blockskip_back`,
    LoRA and full-parameter trainers only, arXiv 2603.20755): dual-stream
    (video + audio) folded-precompute — a no-grad full pass captures the
    skipped front/back blocks' residual, a grad pass runs only the middle
    blocks. Skipped blocks are gradient-starved (no retained backward
    activations), not optimizer-excluded. Requires `blocks_to_swap == 0`;
    mutually exclusive with TREAD and with stochastic-depth (`block_skip_rate`).

## Anchors used

- `backend/core/model_loader.py` — detection, prediction config, zimage/flux2
  single-file loaders, comfy→official conversion.
- `backend/core/attention/registry.py` — per-arch attention backend routing
  (conduit vs diffusers dispatch, head_dim constraints, tq/flash/sage/native).
- `backend/core/pipeline_backends/{flux2,ideogram4,lens,minit2i,krea2,zimage,anima,ltx2}.py`
  — CFG conventions, text encoding, VAE staging.
- `backend/core/models/ltx2_block_loop_wrapper.py` — `Ltx2BlockLoopWrapper`
  (block swap, FBCache, Spectrum for generation; TREAD/BlockSkip attach points
  for training).
- `backend/core/training/ops/ltx2_ops.py` — LTX-2.3 training forward, TREAD/
  BlockSkip config attach/detach per step.
- `backend/core/inference/custom_sampling.py` — SD/SDXL CFG short-circuit at cfg==1.0.
- `backend/core/models/{lens,ideogram4,minit2i,krea2,anima}/*_loader.py` — component
  classes, completion sources (sibling dirs / hub fallbacks / env overrides).
- `backend/core/training/adapters/{sd15,sdxl,zimage,flux2,ideogram4,lens,minit2i,anima,krea2}_adapter.py`
  — TE-frozen policies, dual-transformer training, LLM-Adapter-only mode.
- `backend/core/training/MODEL_ARCHITECTURES.md` — SD1.5/SDXL/Z-Image component
  specs, forward-pass signatures, schedulers.
- `backend/core/inference/context_tiled_decode.py`,
  `backend/core/inference/global_group_norm.py` — the two tiled-decode options
  (`vae_tile_mode`, `vae_tile_global_norm`); install/uninstall via
  `PipelineManager._apply_vae_tiling`.
- `backend/core/keep_hot.py` — `keep_models_hot` model_key computation, VRAM
  guard, resident-set tracking, shared by `pipeline.py` and the 7 DiT
  `pipeline_backends/*.py` files (not `ltx2.py`).
