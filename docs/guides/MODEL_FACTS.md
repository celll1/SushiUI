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
| anima | DiT (Cosmos-Predict2 style) | Qwen3-0.6B (`Qwen3Model`) + 6-layer LLM Adapter; T5 tokenizer feeds adapter target ids | velocity / flow (rectified flow) | latent, Qwen-Image VAE (`AutoencoderKLQwenImage`-style, 16ch) | (verify) standard CFG via `_anima_encode_nag_neg` | conduit tq infer / native train (`attn_mode` torch/flash blocks tq) | split-files layout (`split_files/diffusion_models` \| `text_encoders` \| `vae`) or single DiT `.safetensors`; Qwen3 + Qwen-Image VAE auto-discovered by filename patterns | `anima_adapter.py`; DiT + optional LLM-Adapter-only training; Qwen3 TE + VAE frozen |
| krea2 | MM-DiT (single-stream) | Qwen3-VL-4B-Instruct (frozen) | velocity / flow (rectified flow) | latent, `AutoencoderKLQwenImage` 16ch (latents_mean/std) | `guidance = cfg_scale - 1` (default cfg 4.5); distilled/turbo disables CFG (guidance 0) | conduit (tq usable, GQA) (verify infer/train head_dim) | diffusers dir (`Krea2Pipeline`), transformer-only dir (auto-complement), single-file (diffusers/raw/comfy/sushiUI TE+DiT combined); TE `Qwen/Qwen3-VL-4B-Instruct`, VAE `Qwen/Qwen-Image` `vae` (env `KREA2_TE_DIR`/`KREA2_VAE_DIR` overrides) | `krea2_adapter.py`; transformer only, Qwen3-VL TE ALWAYS frozen (`train_text_encoder` rejected), VAE frozen; train_runner forces bf16 |
| ltx2 | joint video+audio DiT (`LTX2VideoTransformer3DModel`) | Gemma-3 text encoder (frozen) | velocity / flow (`LTX2Pipeline`/`LTX2ImageToVideoPipeline`, txt2vid + img2vid) | 5D video latent (`[T,H,W]`) via LTX video VAE (tiling enabled) + separate audio VAE/vocoder | plain CFG (`guidance_scale`); img2vid pins frame 0 via `conditioning_mask` | n/a (own pipeline backend, not conduit-routed) | not in the single-file completion matrix above (own loader) | own trainer ops (`ltx2_ops.py`); see row-level notes for AP1-3 speed/lightweight features |
| minimax_h3 | joint video+audio DiT, **single stream, no cross-attention** (vendored `MiniMaxH3Transformer3DModel`, 50 blocks, 33 B dense): one packed sequence of `[text \| conditioning \| audio \| video]` rows scattered by `index_copy`, split back by `index_select` | Qwen3-VL-32B (`Qwen3VLForConditionalGeneration`), truncated to **50 decoder layers**, unnormalised hidden state after layer 50, 5120-dim; frozen, never moved (layer-streamed off the mmap) | velocity / flow with the sign **opposite** to the usual convention, `x0 = x_t + σ·v` (vendored `MiniMaxH3Scheduler`); **two sigma schedules**, video shift 12.0 and audio shift 3.0, each stream stepped on its own grid once per loop iteration | 5-D 24-ch video latent, `AutoencoderKLMiniMaxH3` (16× spatial / 4× temporal, 36-layer ViT decoder, fp16, **pinned tiling policy**), pixels ImageNet-normalised RGB over `[0,1]` (not `[-1,1]`); **separate** 32-ch audio VAE (fp32, 32 kHz stereo) | **none** — no `guidance_scale`, no `negative_prompt`, no unconditional branch; guidance is distilled into the weights, one forward per step. Both keys are accepted and warned on a non-default value | conduit-routed, head_dim 128, equal q/kv heads, no mask → no capability guard fires, so native/flash/sage/tq all really run; sage refused in TRAINING mode by the shared mode guard | ComfyUI-style flat tree (`diffusion_models/` + `text_encoders/` + `vae/`) plus MiniMax's config-only `official/` for geometry, tokenizer and normalization vectors; DiT files are `*_pruned_fp8_scaled` or packed ConvRot `*_pruned_w4a8_mixed` (Comfy-Kitchen 0.2.28), each with `fl2va` and `ref2va` partitions — selecting the FILE selects the workflow | `minimax_h3_adapter.py`, **LoRA only** — full fine-tuning refused in three layers; TE and both VAEs frozen; block swap optional (opt-in) |
| sensenova | Qwen3-8B LLM used directly as the flow-matching denoiser (MoT: every layer carries a `_mot_gen` twin of q/k/v/o_proj + norms + MLP, selected per token by a boolean mask; 42 layers, hidden 4096, GQA 32 q / 8 kv heads, head_dim 128, 3-axis RoPE) | none separate — the prompt goes through the LLM's own tokenizer + chat template, encoded by the same prefix forward pass that builds the KV cache the denoiser consumes | flow matching, `linspace(0,1)` with **t=0 noise, t=1 clean** (opposite of flux2/zimage's convention), Euler forward, `v = (x_pred - z)/(1-t).clamp_min(t_eps)` | pixel-space (no VAE); resolution free (not bucketed), only the structural /32 token grid (patch 16 x merge 2) enforced by snapping, off-range sizes warn and generate | ordinary CFG via two independent prefix KV caches (cond + empty-string uncond by default), or up to three (cond + img_cond + uncond) once `ref_images` is supplied, selected by `img_cfg_scale`; upstream's own `needs_cfg = cfg_scale > 1` collapses to single-branch at `cfg_scale <= 1`, which is how the 8-step distillation LoRA runs — no separate mode flag. **`negative_prompt` IS supported** (empirically verified A/B, not just upstream's own default): the uncond branch is built by the SAME `encode_prompt()` prefix pass as the cond branch, so a caller-supplied string substitutes cleanly for `""` and produces a real, `cfg_scale`-dose-dependent suppression effect (clean up to ~cfg_scale 6, classic CFG-too-high degradation from ~cfg_scale 8); default `cfg_scale` 4.0 is kept for both the plain and negative-prompt cases. Has no effect at `cfg_scale<=1` (warned, `sensenova_negative_prompt_no_cfg`) since no uncond branch is built there at all | conduit-routed (BSHD); sage auto-refused for GQA declaratively (`supports_gqa=False`), no per-arch registry entry / same conduit, training unimplemented | sushiUI single-file shard index only (`transformer.`-prefixed, `sensenova_config` metadata carries geometry); no upstream single-file distribution to complete siblings against | none — training is a deliberately separate future phase; the base is converted UNMERGED from the 8-step distillation LoRA specifically to keep the trainable lineage canonical |
| minimax_music3 | 3-stage: 8B `Qwen3ForCausalLM` autoregressive stage + 0.6B RVQ depth decoder (semantic + 7 residual codes/frame) → condition encoder → 2.4B 1D flow-matching DiT (`MiniMaxMusic3Transformer1DModel`, 36 layers, windowed 200-frame/100-hop denoise) → vocoder decode. No separate text encoder component: `prompt`/`lyrics` are tokenized (`Qwen2Tokenizer`) and consumed directly by the AR stage's own `Qwen3ForCausalLM` | AR stage: categorical, top-k 50 sampling of semantic (vocab 16,384) + 7 residual (vocab 1,024 each) codes per 25 Hz frame, cross-entropy-trained upstream. Flow stage: velocity / flow (`FlowMatchEulerDiscreteScheduler`, `invert_sigmas: true`, `sigmas = linspace(1, 1/steps, steps)`) at 86.13 Hz | 128-ch latent = **two folded 64-ch mono streams** (`vocoder.forward` reshapes `[B,128,L]`→`[2B,64,L]`); `MiniMaxMusic3Vocoder` is **decode-only** (upsamples 512× to 44.1 kHz stereo) — the matching encoder exists in `official/dav.pth` but is not part of the released diffusers component set and is not wired | **two CFGs, no negative prompt anywhere.** AR CFG fixed at 1.5 (unconditional branch = the same prompt with interior tokens replaced by `<\|audio_cfg\|>`); flow CFG exposed as `flow_guidance_scale` (default 1.7), unconditional branch conditions on **zeros** (there is no text/audio unconditional branch to negate, structurally, not by omission) | conduit-routed (attention re-pointed at `backend/core/attention/` during vendoring, replacing upstream's `dispatch_attention_fn`); same conduit for inference and (design-only, unimplemented) training | `official/` 7-component tree (default; every config, including for a flat/GGUF DiT, is sourced from here); flat ComfyUI-repack safetensors (DiT + text encoder, key-remapped — fused QKV split, `.gamma`/`.beta` norm rename, condition encoder unfolded out of the DiT; LM+depth-decoder split apart for the text encoder); GGUF containers (same remap, native reader, no `gguf` pip dependency); Q8_0 packed text encoder (`GGUFQ8_0Linear`, dequantizes once per device move); INT8 ConvRot (`ConvRotInt8Linear`, reused unchanged from MiniMax-H3) for both the flat DiT and the pruned text encoder. `text_encoder_file` on `POST /models/load` selects which of 4 text-encoder builders (non-pruned flat, pruned flat, pruned GGUF dense, pruned GGUF Q8_0) runs, generalising MiniMax-H3's `te_override` field | **none.** Training is out of scope for the phases shipped so far; not in `ARCH_REGISTRY`. Design-forward-compatible only: flow-stage (DiT) LoRA is reachable in principle (needs a from-scratch DAV-encoder reimplementation, since the encoder half is unpublished in the diffusers component set), AR-stage (LM) training is **blocked** — the RVQ tokenizer's own encoder (turns audio into the codes the LM predicts) is not published anywhere in the release |

## VRAM management: keep_models_hot (opt-in, all image archs except sensenova; ltx2 is video)

Post-load CPU offload and generation-time sequential component staging (text
encoder(s) -> GPU -> encode -> CPU, denoiser -> GPU -> denoise -> CPU, VAE ->
GPU -> decode -> CPU, one component on GPU at a time to bound peak VRAM) is
already implemented for every architecture: the SDXL reference path in
`backend/core/vram_optimization.py`, the 8 DiT archs (including sensenova) via
per-arch `_move` helpers in `pipeline_backends/*.py`, and LTX-2.3 via
diffusers accelerate's `model_cpu_offload_seq` hooks.

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
- **minimax_h3 and acestep are excluded too**, for the same reason plus, on
  MiniMax-H3, a structural one: its components are 21 GB (DiT) + 51.5 GB (text
  encoder) + 5.2 GB + 0.6 GB against a 48 GB card, so there is no configuration
  in which two of them are wanted resident at once. `core/keep_hot.py` is not
  imported by `pipeline_backends/{ltx2,acestep,minimax_h3}.py`.
- **sensenova is excluded too**: `core/keep_hot.py` is not imported by
  `pipeline_backends/sensenova.py`. Its transformer is the ONLY component
  (pixel-space, no VAE, no separate text encoder), so keep-hot's per-component
  residency has nothing to partition across. This is orthogonal to
  `sensenova_mot_phase_eviction` (see the sensenova row above): that feature
  moves weight HALVES within one already-resident transformer during a single
  generation, an axis keep-hot's cross-generation per-component residency
  does not have.
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
| `AutoencoderKLMiniMaxH3` (24ch) | minimax_h3 (video) | **5-D** | present (`MiniMaxH3VideoGroupNorm` in the conv encoder; the decoder is a 36-layer ViT with RMSNorm) | ViT decoder is all attention | **not offered** (5-D, and its tiling policy is pinned — see the row-level notes) |
| `AutoencoderKLMiniMaxH3Audio` (32ch) | minimax_h3 (audio) | 3-D waveform latent | n/a (DAC/BigVGAN-derived 1-D stack) | n/a | **not offered** |
| none | minit2i (default variants), sensenova | pixel-space | n/a | n/a | **not offered** (no VAE) |

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
  - **Weight-only INT8**: Z-Image is in `RUNTIME_INT8_ARCHS`, so
    `unet_quantization: "int8"` converts the loaded transformer in place
    (`ZImageMixin._zimage_runtime_int8`, called in all three generate paths
    before the block-swap branch / staging). It runs **after** the LoRA gate
    (`_load_lora_zimage`/`_unload_lora_zimage`) only in txt2img and img2img;
    `_generate_inpaint_zimage` has no LoRA gate at all, so a prior
    txt2img/img2img generation that loaded LoRAs can leave
    `_zimage_lora_wrapped_modules` populated for a following int8 inpaint
    request that carries no LoRAs of its own, which the hook then refuses with
    advice ("remove the LoRAs") that request cannot act on — degradation, not
    corruption, and a symptom of the larger pre-existing gap that inpaint also
    *runs* with the previous generation's LoRAs since it has no gate to unload
    them. The FP8 values are unaffected and still go through
    `move_zimage_transformer_to_gpu`, which is why zimage has **no**
    `unet_quantization` entry in `ARCH_UNSUPPORTED` (unlike
    krea2/ideogram4/ltx2/acestep, which need `ARCH_SUPPORTED_VALUES: ["int8"]`) —
    it honours both axes for every value.
  - **Stale block offloader**: `transformer._block_offloader` is attached per
    block-swap generation and NEVER cleared (`_zimage_cleanup` says so, and the
    transformer's `forward` consults the attribute whenever present). The INT8
    hook therefore passes a `precheck` that TEARS IT DOWN rather than refusing —
    refusing would be the LTX-2.3 false-advice defect, since the offloader is
    stale by construction at that point and a block-swap request overwrites it a
    few lines later anyway. The teardown runs only on a request that really
    converts.
  - **Loader**: `_swap_zimage_quantized_linears` (int8 and e4m3 detected
    independently, both swaps run — an int8 artifact is MIXED) runs on the
    meta-device module before the `strict=True, assign=True` load, verified by
    `verify_quantized_swap`. Because the load is `assign=True`, a pure float8
    cast (float8 weights, no `.weight_scale`) is passed through
    `cast_float8_tensors` first — assignment would otherwise install float8
    parameters. A quantized file in the ComfyUI fused-qkv layout is REFUSED: the
    comfy→official rewrite row-splits `attention.qkv` and would split a per-row
    `.weight_scale` by the same rule.
  - **Export prefix is a requirement, not a convention**: `EXPORT_LAYOUTS["zimage"]`
    uses the empty live prefix because `detect_model_type` recognises a Z-Image
    safetensors by four `startswith` probes (`cap_embedder`, `t_embedder`,
    `context_refiner`, `x_embedder`/`all_x_embedder`), and a
    `model.diffusion_model.` prefix is claimed EARLIER by the SD/SDXL branch.
  - **Offline route is scoped to official-layout sources.** `_zimage_config`
    refuses a fused-qkv source and points at `POST /models/export-quantized`: the
    qkv split is `(n_heads*head_dim, n_kv_heads*head_dim, n_kv_heads*head_dim)`,
    which a per-key `source_transform` cannot see, and it is equal thirds only
    while `n_kv_heads == n_heads`. The runtime export is unaffected — it reads the
    live module, which the loader has already converted to official layout.
  - **CENSUS IS CONFIG-DERIVED, NOT MEASURED.** There is no Z-Image checkpoint on
    this machine, so unlike every other arch here the numbers were produced by
    building `ZImageTransformer2DModel` on the meta device from the published
    `transformer/config.json`, not by reading a safetensors header: **276
    `nn.Linear` / 6.1539 G 2-D parameters / 521 state-dict keys**; every shape
    8-aligned (nothing lost to the GEMM-alignment filter); **37 below the runtime
    min-work gate holding 0.1278 G — 13.4% of layers for 2.08% of parameters**, of
    which 32 are the 256×15360 AdaLN modulation Linears that are the exact shape
    class Anima's roll-up measured as a net loss. Hence
    `skip_below_work_gate: True`. No timing run on Z-Image exists, and no artifact
    of this arch has been produced from a real checkpoint.
- **flux2** — Repo implementation uses a Qwen3 causal LM text encoder (not the
  upstream FLUX text stack); concatenates Qwen3 hidden states from layers 9/18/27.
  Distilled checkpoints inject a guidance vector instead of running CFG. VAE always
  pulled from the Apache-2.0 4B store regardless of transformer variant.
  - **Weight-only INT8**: FLUX.2 is in `RUNTIME_INT8_ARCHS`, so
    `unet_quantization: "int8"` converts the loaded transformer in place
    (`Flux2Mixin._flux2_runtime_int8`, before the block offloader is built and
    before staging — `move_flux2_transformer_to_gpu` is only reached in the
    NO-block-swap branch, so quantizing there would skip every block-swapped
    generation). The FP8 values still go through `move_flux2_transformer_to_gpu`
    /`_quantize_transformer`, which now short-circuits on an already-quantized
    module instead of deep-copying it. `model_loader._swap_flux2_quantized_linears`
    reads a quantized checkpoint back (int8 and e4m3 detected independently, both
    swaps run); only the diffusers key layout is accepted for one. The dtype cast
    is done BEFORE the swap, and the post-load cast skipped, because a later
    `.to(bf16)` would convert the e4m3 weight BUFFERS — not a correctness problem
    (bf16 represents all 256 e4m3 codes exactly and the dequant path still applies
    the scale; measured forward error identical) but it doubles those buffers and
    permanently drops `Fp8Linear`'s `_scaled_mm` fast path, which gates on the
    weight dtype. The swap count is then verified against the checkpoint's own
    quantized-key census (`quantized_checkpoint_guard.verify_quantized_swap`,
    shared with anima): a scale-less INTEGER or path-mismatched quantized file is
    refused instead of falling through to the plain `strict=False` load.
  - **A pure float8 CAST is not a quantized checkpoint** and loads normally on
    all four archs. `quantized_checkpoint_guard.scaled_quantization_report`
    narrows the census before any loader branches on it: float8 `.weight`s with
    NO `.weight_scale` anywhere (the dominant ComfyUI "fp8" release shape) are
    bf16 rounded to 8-bit floats, meant to be read by casting back — which every
    loader already does, exactly (e4m3's range and 3-bit mantissa sit inside
    bf16). So such a file skips the swap, is not refused for its key layout on
    FLUX.2, and loads as it did before the guards existed; Anima additionally
    casts the tensors first because its load is `assign=True`. Scale-less
    INT8/uint8 weights stay refused: those are codes, and casting them measured
    103020% error.
  - **Offline artifact**: `quantize_transformer_fp8.py --arch flux2 --format int8`
    accepts a BFL/Comfy *or* a diffusers source and always emits diffusers keys —
    the arch's `source_transform` runs diffusers'
    `convert_flux2_transformer_checkpoint_to_diffusers` per key (fused
    `img_attn.qkv`/`txt_attn.qkv` fan out to q/k/v; per-row scales make split-then-
    quantize identical to quantize-then-split). Geometry comes from the pinned
    `core/models/flux2/single_file.FLUX2_DEFAULT_CONFIG` (Klein 4B: 5 double + 20
    single blocks), NOT from a hub download; an unrecognised block count is
    refused rather than guessed, and `--config` overrides it.
    Measured on `flux-2-klein-base-4b.safetensors` (149 source tensors → 169
    diffusers keys, exactly the model's state-dict key set): 109 Linears, 3.8755 G
    2-D parameters, **109/109 selected int8, zero e4m3 fallbacks**, geomean
    int8-over-e4m3 weight-error advantage 2.742x, worst layer advantage 1.565,
    highest mean per-row crest 7.38 (`proj_out`) against the 12.0 threshold. Only
    3 Linears sit below the runtime min-work gate (0.04% of the parameters), which
    is why `skip_below_work_gate` is off for flux2.
  - **Runtime export** (`POST /models/export-quantized`): the flux2 metadata block
    propagates `base_model_repo` and `is_distilled` from the LOADED config when it
    has them (the loader puts both there), and omits them otherwise (the offline
    tool's route, which has only the pinned geometry). They are the only metadata
    keys the loader reads back, and `is_distilled` alone flips
    `do_classifier_free_guidance`; a full-FT save exported without them would be
    re-detected as `klein-base-4B` — its 20 single blocks match none of the
    probe's 24/36/48 arms — and silently regain CFG.
- **ideogram4** — Only architecture bundling two transformers (conditional +
  unconditional); asymmetric CFG zeroes the unconditional text branch. FP8
  weight-only Qwen3-VL and FP8 transformer linears; head_dim 256 keeps it on native
  attention. VAE is `AutoencoderKLFlux2` at 32 latent channels.
  - **Weight-only INT8**: Ideogram 4 is in `RUNTIME_INT8_ARCHS`, and it is the
    largest target here — **279 Linears holding 9.2779 G 2-D parameters per
    transformer, times two transformers** = 558 Linears / 18.5559 G parameters,
    all shapes 8-aligned. `unet_quantization: "int8"` converts BOTH branches in
    one call (`Ideogram4Mixin._ideogram4_runtime_int8` ->
    `vram_optimization.apply_runtime_int8_quantization_multi`), from
    `_ideogram4_stage_transformers`, before the per-transformer block offloaders
    are built and before the `->GPU` move (the block-swap branch never performs
    that move at all). The MULTI entry point is not a convenience: the
    `_runtime_int8_converted` latch is per manager, so two single-module calls
    would convert the conditional branch, latch, and silently leave the
    unconditional one bf16 — the two halves of one asymmetric-CFG step at
    different precisions. `arch_capabilities` still lists `unet_quantization` as
    unsupported for ideogram4 (its FP8/nf4 story is checkpoint-format driven) and
    carries `int8` in `ARCH_SUPPORTED_VALUES`, the krea2 treatment.
    `skip_below_work_gate` is ON: 38 of the 279 Linears are below the runtime
    min-work gate and 34 of them are 512x18432 AdaLN modulation Linears — the
    shape class Anima measured as a net loss — for 3.52% of the parameters. That
    is Anima's measurement applied to a matching shape class plus Ideogram 4's own
    census, NOT a timing run on Ideogram 4.
  - **Loader**: `_swap_ideogram4_quantized_linears` now detects int8 and e4m3
    independently and runs both swaps (it used to call only the FP8 half, which
    would have mis-loaded an int8 file), after the nf4 branch — a bitsandbytes
    checkpoint has uint8 weights and `.quant_state` siblings, no `.weight_scale`.
    The swap count is then checked against the checkpoint's own quantized-key
    census (`quantized_checkpoint_guard.verify_quantized_swap`); the same check
    was added to the Krea 2 loader, where a scale-less or path-mismatched
    quantized file also used to fall through to the plain load. Measured on the
    published FP8 checkpoint: 279 scale keys = 279 quantized weights = 279
    swappable Linears, per transformer.
  - **Offline artifact / export**: `EXPORT_LAYOUTS["ideogram4"]` is the only
    multi-component entry — `transformer.` + `unconditional_transformer.` into
    ONE `ShardWriter`, because two files would not be a single-file artifact and a
    conditional-only file would load (the loader skips a missing unconditional
    branch with a print) and then generate with one branch quantized. The offline
    tool plans one pass per component and accepts either a published model ROOT
    (`<root>/transformer/` + `<root>/unconditional_transformer/`) or a combined
    single file; a single component directory is refused. Its `source_transform`
    splits the fused `layers.N.attention.qkv` (13824x4608) into `to_q`/`to_k`/`to_v`
    and renames `attention.o` -> `to_out.0`, delegating to the SAME per-key rule
    `ideogram4_loader._convert_fused_qkv_to_split` uses, because the loader splits
    before the quantized swap so the checkpoint keys are not module paths.
    Measured on `M:/model/ideogram4/ideogram4`: 669 source keys -> 805 canonical
    keys per component, set-EQUAL to the meta-built module's state-dict keys plus
    its 279 scale siblings (0 either way), and the `tensor=None` key pass equals
    the per-tensor pass. Selection: 241 quantized / 38 skipped per transformer,
    482 / 76 in total.
  - **No int8 artifact exists for the local checkpoint**, and cannot: the only
    Ideogram 4 checkpoint here is the FP8 one, and the offline tool now REFUSES an
    already-quantized source (its weights are rounded once already, and its
    `.weight_scale` keys would collide with the new ones). A dry run reports the
    refusal and still prints the selection. The int8 path is for a bf16
    Ideogram 4 — a release or a sushiUI full-FT save.
    The refusal takes the same TWO pieces of evidence the loader-side census
    does: a `.weight_scale` key OR a `.weight` whose stored dtype is
    int8/uint8/float8, read from the safetensors HEADER (zero tensor bytes) with
    the dtype set imported from the guard. The scale test alone missed the
    commonest already-rounded source there is — a scale-less ComfyUI fp8 cast,
    whose keys remap onto module paths like any other, so nothing else refused
    it. Verified against the four real sources (ideogram4 482/76, flux2 109/109,
    krea2 263/1, anima 232/283): all still accepted.
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
  - **Weight-only INT8 (W8A8)**: Anima accepts a mixed int8/e4m3 checkpoint
    produced by `subapps/fp8_quantize/quantize_transformer_fp8.py --arch anima
    --format int8 --skip-below-work-gate`. `anima_loader.load_anima_dit` detects
    int8 and fp8 INDEPENDENTLY and runs both swaps (`Int8Linear` / `Fp8Linear`),
    then loads with `assign=True`, which preserves the stored dtypes. The W8A8
    integer GEMM itself is opt-in per process (`SUSHI_INT8_MM=1`,
    `GET/POST /api/v1/system/int8-mm`) or per generation (`quantized_gemm_mode`).
  - **Runtime INT8 (no pre-built artifact)**: `unet_quantization: "int8"` on an
    ordinary bf16 checkpoint of any arch in `RUNTIME_INT8_ARCHS` (Anima, Krea 2,
    FLUX.2, Ideogram 4, LTX-2.3) converts the loaded transformer IN
    PLACE, once, at the first generation after the model load
    (`vram_optimization.apply_runtime_int8_quantization` ->
    `core/models/common/int8_runtime_quantize.quantize_linears_in_place`). The
    selection rule is the SAME module the offline tool imports, and the layer
    selection is gate-tested against the committed offline audits
    (`tmp/int8_runtime_anima_gate.py`): Anima 232 quantized (231 int8 + 1 e4m3) /
    283 skipped, Krea 2 263 quantized (259 int8 + 4 e4m3) / 1 skipped — identical
    names, identical per-layer format, measured errors matching to <4e-9.
    Measured on an RTX 6000 Ada with the module on CPU and the quantization math
    on GPU: Anima 2.1 s / 0.31 GB GPU peak (3.895 GB -> 2.327 GB of weights),
    Krea 2 11-16 s / 4.22 GB GPU peak (23.879 GB -> 11.948 GB). No second copy of
    the model is made (each source weight is dropped as its replacement is built),
    unlike the SD1.5/SDXL `_original_unet` path.
  - **HOST RAM cost of the runtime conversion is ~1.6x the source module, for
    the session** — the module bytes fall, the process's RSS does not.
    `tmp/int8_runtime_host_memory.py` (20 Hz RSS sampler, real Anima): RSS 0.959
    GB after load -> 6.159 GB peak -> **6.159 GB steady after `gc.collect()`**,
    against 2.327 GB of resulting module bytes and a 3.895 GB source. The
    safetensors mapping of the source stays resident because the 283 skipped
    Linears and every non-Linear parameter still reference it. Extrapolated by
    the Krea 2 figures above, a 23.9 GB bf16 Krea 2 transformer needs roughly
    **36 GB of host RAM** held until the model is reloaded: fine on a 64/96 GB
    box, not viable on 32 GB. Per-layer GPU working set (float32) is small — 0.31
    GB Anima, 4.22 GB Krea 2 — and a layer that hits `torch.cuda.OutOfMemoryError`
    is retried on the weight's own device (CPU) instead of aborting; the fallback
    list is in the returned audit under `oom_fallback_layers`.
  - The conversion is ONE-WAY until the model is reloaded: a later `null`/fp8
    request keeps the quantized transformer and returns a
    `runtime_quantization_persistent` warning. **Recovery is
    `POST /models/load` with `force=true`** (the model selector's Load button
    sends it when the selected model is the loaded one, and reads "Reload
    Selected Model"): without it, `_load_model_locked` early-returns on the same
    model id and resets nothing. `keep_hot.compute_model_key`
    normalises the quantization component to `"int8"` once converted, so the
    resident set does not thrash between generations.
  - An already-quantized CHECKPOINT sets its OWN latch,
    `_runtime_int8_from_checkpoint`, not `_runtime_int8_converted`. Keep-hot keys
    the two identically (both mean "the resident transformer is quantized"), but
    only a real in-place conversion emits `runtime_quantization_persistent` —
    nothing was converted and a reload would produce the same model, so saying
    otherwise is false. It matters most on Ideogram 4, whose published
    checkpoints are ALL FP8/nf4: with a shared latch, one `int8` request made the
    false one-way warning fire on every subsequent generation.
  - Refusals, each leaving the transformer exactly as it was: an already
    weight-only quantized CHECKPOINT (`quantization_superseded`); weights already
    cast to float8 by an FP8 generation earlier in the same session
    (`quantization_superseded` — quantizing e4m3-rounded weights to int8 measured
    0.04400 relative RMS against 0.00394 for a direct conversion, 11.2x, i.e.
    worse than either format alone); a LoRA-wrapped module
    (`quantization_fallback` — wrappers hide Linears and would silently change
    the selection, so both backends call the converter before LoRA application);
    and `int8` on any arch outside `RUNTIME_INT8_ARCHS`, refused before any
    `copy.deepcopy` is paid for.
  - A conversion that dies part-way (CUDA OOM at layer 120/263 is the realistic
    case) leaves the module PARTIALLY converted, which cannot be undone. It is
    NOT latched as converted: the manager gets `_runtime_int8_partial`, the
    response carries `quantization_partial` with the layer count, the keep-hot
    fingerprint becomes `"int8_partial"`, and the next `int8` request RESUMES —
    selection walks `nn.Linear` and a converted layer is no longer one. The
    checkpoint-provenance branch is suppressed while partial so it cannot claim
    those modules came from the file. A conversion that converts ZERO layers also
    does not latch the flag.
  - **PARTIAL conversion + block swap is the real narrow hazard here — not
    `weight_scale` desyncing from the block-swap stream.** Traced: block swap
    ALWAYS moves `weight_scale` correctly, on any arch, converted or not.
    `prepare_block_devices_before_forward` (`block_offloading.py`) does
    `block.to(device)` first — a whole-module move, which carries `weight_scale`
    (an `Int8Linear`/`Fp8Linear` buffer/attribute) with it — and both
    `weighs_to_device` and `swap_weight_devices` afterward touch only `.weight`,
    never `.weight_scale`, so the scale stays wherever the block move put it and
    the pair never desynchronises. The coalesced H2D-only path needs a single
    dtype per swappable block and self-disables (falls back to standard staging
    swap) the moment a block is mixed-dtype — see the LTX-2.3 "MEASURED COST"
    note below for a full-conversion example of that self-disable; it is
    correct, not a hazard, and it is arch-independent.
    The actual hazard was narrower, and is **FIXED** (`DtypeSplitGuardMixin` in
    `block_offloading.py`, inherited by `TransformerBlockOffloader` and
    `FluxBlockOffloader`): a conversion that dies part-way (CUDA OOM at layer N)
    leaves blocks STRUCTURALLY heterogeneous — the same module path is
    `Int8Linear` in one block and still `nn.Linear` in another — and
    `swap_weight_devices` used to pair modules by NAME and SHAPE only; an
    `Int8Linear`'s int8 `.weight` and a same-shaped `nn.Linear`'s bf16 `.weight`
    passed that check identically, and the staging copy then wrote int8 codes
    into bf16-typed storage (or vice versa) with no error and no warning. The
    quantized module kept computing: `Int8Linear._dequant_forward` accepts a bf16
    weight, so the corruption was silent rather than the loud self-disable a full
    conversion gets. **A COMPLETE conversion puts the same split in the
    checkpoint**, because the int8-vs-e4m3 choice is per layer and made from that
    layer's own weights: the shipped audits diverge on `blocks.0.mlp.layer2` for
    Anima (1 e4m3 among 231 int8) and on `transformer_blocks.27.attn.to_v` /
    `.ff.down` for Krea 2 (4 among 259); FLUX.2's 109/109 int8 has no split at
    all. **Neither of those two is reachable today**: Anima's diverging path is in
    block 0, which stays permanently resident in inference (`transformer_registry`
    clamps `blocks_to_swap` to `num_blocks - 1`, and the forward-only rotation
    only touches the last `blocks_to_swap` blocks) — it is pairable only in
    `supports_backward` mode, which swaps blocks `0..blocks_to_swap` — and Krea 2
    has no block-swap streaming at all (`pipeline_backends/krea2.py`). The
    reachable cases are the PARTIAL conversion on any block-swapping arch, and
    plausibly LTX-2.3, whose conversion is mixed by design (unverified: no LTX
    audit on disk). **The partial case alone is why the guard DEFERS the split
    paths rather than refusing the swap** — a partial conversion is a recoverable,
    resumable state, and refusing would turn it into a dead generation. Mixed
    dtypes WITHIN a block are not affected: pairing is per module path.
    The guard resolves, once per offloader and on the FIRST swap (so LoRA
    sub-Linears added after block-swap setup are seen), the set of Linear paths
    whose dtype is not the same in every block of a class; those paths are
    excluded from the paired staging swap and each side is moved to its own
    target device individually, dtype unchanged, while every other path keeps the
    paired swap. It is loud: a `[BlockOffloader]` block naming the paths and
    dtypes, plus a `block_swap_dtype_split` generation warning. A mismatch that
    appears AFTER resolution (module tree changed) raises `RuntimeError` instead.
    Resolving the set ONCE, over all blocks, rather than per pair is load-bearing:
    the cached staging buffers are allocated from the first swap's job list, so a
    job list whose length depended on which two blocks are swapping would be
    zipped against shifted buffers. The "all blocks" it resolves over is only the
    blocks the rotation can actually pair (`pairable_block_indices`), so a
    divergence confined to a permanently resident block — Anima's — is not
    excluded and emits no warning.
    **EXPECTED COST (derived, not measured):** the deferred move allocates a fresh
    PAGEABLE CPU tensor per excluded path per swap and its `.to(cpu)` is a
    host-blocking sync, instead of recycling the pinned/staging storage — it runs
    on the executor worker so it never stalls the model, but it drains that swap's
    overlap. A conversion that stopped BETWEEN blocks splits every path, leaves
    `weight_swap_jobs == []`, and serialises the whole swap into pageable moves;
    that is also why the empty-list guard on `released_pinned_buffer` is
    load-bearing (without it the pinned strategy raises `IndexError` there).
    This is cross-arch: every `RUNTIME_INT8_ARCHS`
    member that also supports block swap (ltx2, flux2, ideogram4, anima, zimage)
    shares those two offloaders, so all of them inherit the guard.
    `backend/tests/block_swap_dtype_split_test.py` holds the functional
    regression (including the pre-fix mechanism), and
    `quantized_capability_parity_test.BlockSwapDtypePairingParityTest` requires
    every offloader class that defines `swap_weight_devices` to inherit the mixin
    so a third offloader cannot re-derive the pairing without it.
  - `--skip-below-work-gate` is **required** for Anima, unlike Krea 2 — a
    per-arch knob that now lives in
    `int8_runtime_quantize.ARCH_QUANT_POLICY` (the CLI flag defaults to it and
    can still override it explicitly with `--skip-below-work-gate` /
    `--no-skip-below-work-gate`). The
    conversion skips **283 of the DiT's 515 Linears** (168 AdaLN modulation
    layers, 56 cross-attention k/v projections, the LLM-adapter projections, the
    timestep and final-layer Linears), leaving 232 quantized. Those 283 can never
    clear the runtime min-work gate at any `m`, so quantizing them would make
    them run `Int8Linear._dequant_forward` forever — slower than the `F.linear`
    the bf16 checkpoint runs. Cost of skipping them: ~369 MB of saving
    (2.4987 GB shipped vs ~2.13 GB fully quantized, against a 4.1822 GB bf16
    source — i.e. −40% instead of −49%).
  - Measured effect, **Linear-only** and to one significant digit. Harness:
    `tmp/anima_int8_rollup_probe.py` (RTX 6000 Ada sm_89, bf16, batch 1, per
    denoise pass; real layer census from the artifact's audit JSON; weights
    rotated over 6 buffers to defeat L2; clock-aware warmup — the card idles at
    210 MHz and a short loop measures fiction). Against the bf16 checkpoint:

    | resolution | 384² | 512² | 640² | 768² | 1024² | 1328² |
    |---|---|---|---|---|---|---|
    | with the flag | ~1.3x | ~1.6x | ~1.8x | ~1.9x | ~2x | ~2x |
    | naive all-int8 | ~0.9x | ~1.2x | ~1.4x | ~1.6x | ~1.8x | ~1.8x |

    What is robust across every harness tried: the filtered artifact is **faster
    than the naive one at every resolution**, and the naive one **regresses**
    below break-even at low resolution. What is *not* robust is the magnitude,
    and at ≤512² not even the sign: a separately written audit harness reported
    the filtered arm at 0.90x/0.98x for 384²/512² where this one measures
    1.3x/1.6x. Treat ≤512² (which includes Anima's **default** 512²,
    `pipeline_backends/anima.py`) as "break-even to modestly positive,
    harness-dependent". The spread is dominated by fixed per-call cost: on this
    host a CUDA launch is ~8 µs, an `F.linear` ~20 µs and an `Int8Linear.forward`
    ~60 µs regardless of size, so the small-`m` layers are dispatch-bound rather
    than arithmetic-bound.
  - These figures are a **Linear-only** roll-up. Attention (SDPA), norms, RoPE,
    the Qwen3 TE and the Qwen-Image VAE are excluded and are identical between
    the arms, so the **end-to-end** speedup is strictly closer to 1.0 than any
    number above.
  - Some quantized layers are permanently on the dequant path and are kept only
    for the VRAM: both tokenizers pad to `max_length=512`, so
    `llm_adapter.blocks.N.mlp.2` sits at mkn = 2.15e9 < the floor at every
    resolution, and both `t_embedder` Linears run at m=1. CFG does not raise `m`
    — Anima runs conditional and unconditional as two separate passes
    (`anima_pipeline_ops.py`).
  - The runtime min-work gate (`_MIN_WORK_K/N/MKN`) is left **unchanged**, but
    that is a scope decision, not a measurement: the constants are Krea-2-derived
    and shared by every arch that uses `int8_linear`. For Anima the floor is
    measurably a little too high. Counterexample (harness
    `tmp/anima_int8_gate_counterexample.py`, same host/method as above), forcing
    `_MIN_WORK_MKN = 0` and comparing against the dequant path each shape is
    actually routed to:

    | shape | mkn | int_mm/dequant | verdict |
    |---|---|---|---|
    | m=576 k=n=2048 (384², the 168-layer group) | 2.42e9 | **1.20x** | refusal costs ~20% on those layers |
    | m=512 k=4096 n=1024 (`llm_adapter…mlp.2`) | 2.15e9 | **1.06x** | marginal loss |
    | m=480 k=n=2048 | 2.01e9 | 0.90x | refusal correct |
    | m=400 k=n=2048 | 1.68e9 | 0.80x | refusal correct |

    Break-even therefore sits near 2.2e9, not 2.5e9. Lowering it is a change to a
    **shared** constant and must not be shipped without re-validating Krea 2
    (whose artifact and gate behaviour are the reason the constant has its
    current value); that re-validation has not been done, so the constant stands.
    Note also that the original Krea 2 sweep predates the clock-aware timing used
    here, which is a further reason any retune must re-measure both arches rather
    than edit the number.
  - Block swap composes with the quantized layers (verified, incl. the pinned
    staging path): the offloader keys on class names ending in `Linear`, moves
    only `weight`, so the float32 `weight_scale` stays GPU-resident. The
    `block_swap_h2d_only` coalesced path needs one dtype per block and therefore
    disables itself on a quantized DiT with a printed message.
  - Training is dequant-only: `training/ops/anima_ops.load_components` calls
    `disable_int8_mm` + `disable_scaled_mm` on the transformer and text encoder.
  - **Full fine-tuning a quantized Anima checkpoint is refused**
    (`AnimaFullParameterAdapter`, `NotImplementedError`). `Int8Linear`/`Fp8Linear`
    hold `weight` as a *buffer*, so `requires_grad_(True)` is a no-op on them and
    `named_parameters()` skips them — a full FT would silently train only the 283
    skipped Linears (measured: 405 M of the DiT's 2.09 B weight elements remain
    trainable, i.e. **80.6% of the weights are frozen**;
    `tmp/anima_int8_fullft_guard_probe.py` ARM 0) while the loss still fell.
    LoRA on the same checkpoint is
    fine and stays allowed (the adapter wraps the quantized module). Same guard
    as `Ideogram4FullParameterAdapter`, but conditional: the bf16 Anima
    checkpoint full-fine-tunes normally.
  - `vram_optimization._anima_quantize_fp8` (the legacy per-call full-weight
    dequant patch, 0.50–0.96x vs 16-bit) is **superseded** and refuses to run on
    an already-quantized checkpoint, emitting a `quantization_superseded` warning.
- **krea2** — Single-stream MMDiT with rectified flow. UI `cfg_scale` maps to Krea
  `guidance = cfg_scale - 1`; the distilled/turbo checkpoint sets guidance 0 (no
  CFG). Qwen3-VL-4B TE is always frozen and TE training is explicitly rejected;
  train_runner forces bf16.
  - `unet_quantization` is honoured for exactly ONE value, `"int8"` (the in-place
    runtime conversion described in the Anima row; `_krea2_runtime_int8` runs it
    before the transformer is staged). The FP8 story stays checkpoint-format
    driven, so `arch_capabilities` still lists `unet_quantization` as unsupported
    for krea2 and carries `int8` in `ARCH_SUPPORTED_VALUES` — the panels read
    that matrix and offer only the values the arch applies (and normalise a
    persisted value the loaded arch does not offer back to null, so the selector
    can never hold a value that is not among its options).
  - Host RAM: a runtime INT8 conversion of the 23.9 GB bf16 transformer holds
    roughly 36 GB of host RAM for the session (source mapping + quantized
    module); see the measured Anima ratio in the Anima row.
- **ltx2** — Video (+ optional audio) generation, not part of the 10-architecture
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
  - **Weight-only INT8**: LTX-2.3 is in `RUNTIME_INT8_ARCHS`, so
    `unet_quantization: "int8"` converts the video DiT in place
    (`LTX2Mixin._ltx2_runtime_int8`, called at the top of ALL THREE generate
    paths — `_generate_txt2vid_ltx2`, `_generate_img2vid_ltx2`,
    `_generate_vidoutpaint_ltx2` — and before `_ensure_ltx2_swap_and_offload`,
    because the block offloader captures the blocks' Linear modules and the
    conversion replaces them; a live offloader raises a `RuntimeError`, not an
    `assert`). The image endpoints already refuse an LTX-2.3 model, so those
    three are the whole generate surface.
  - **Census, and what the DiT actually is.** Enumerated from
    `LTX2VideoTransformer3DModel` on the meta device: **1660 `nn.Linear`
    modules holding 18.9777 G 2-D parameters**, all 8-aligned (nothing lost to
    the GEMM-alignment filter). Selection with the arch policy
    (`skip_below_work_gate: True`): **1360 quantized / 300 skipped**; the 300
    hold 0.0362 G, i.e. **0.19%** of the DiT's Linear parameters, and 288 of
    them are the per-attention `to_gate_logits` projections whose
    `out_features` is 32, so the runtime gate (k>=2048, n>=1024) can never admit
    them at any `m` and they would always run the slower dequant path. The
    filter therefore costs almost nothing here (contrast Ideogram 4's 3.52% and
    Anima's ~9%). PROVENANCE: shape census + Anima's measurement on a matching
    shape class — **not** a timing run on LTX-2.3.
  - **34.33 G is NOT the DiT.** An earlier census over the whole published
    directory reported 34.3396 G of 2-D tensors and 99.1% gate-reachable. Split
    by component: `transformer` 18.9824 G, `text_encoder` 12.1855 G (Gemma-3 —
    `language_model.*` alone is 11.7653 G over **48 decoder layers**, plus a
    0.4158 G vision tower), `connectors` 3.1717 G, both VAEs and the vocoder
    ~0. **Only the DiT is quantized.** The bundled LLM is excluded
    STRUCTURALLY, not by a name heuristic: both enumerations walk the DiT
    module tree (offline: an `init_empty_weights` build; runtime:
    `named_modules()` of `pipeline.transformer`), and neither can reach a text
    encoder that is a different component object in a different directory.
    `text_encoder_quantization` is separately declared unsupported for ltx2.
  - **Keys are module paths.** Verified against the distilled 8-shard index:
    4186 checkpoint keys vs 4186 keys in a meta build from the same
    `config.json`, **zero difference in either direction**. So `EXPORT_LAYOUTS`
    uses the identity `source_transform` and an empty prefix — unlike FLUX.2
    (BFL remap) and Ideogram 4 (fused qkv).
  - **Loader**: LTX-2.3 is diffusers-DIRECTORY only (no single-file variant), so
    `ltx2/loader.py` censuses the `transformer/` component's shard HEADERS first
    (zero tensor bytes — the alternative is materialising 37 GB to find out),
    runs the census through `scaled_quantization_report`, and only for a SCALED
    quantized file rebuilds the DiT (`init_empty_weights` +
    `_swap_ltx2_quantized_linears` + `verify_quantized_swap` +
    `load_state_dict(assign=True)`) and hands it to `from_pretrained` as a
    pre-built component. A plain float8 CAST with no scales takes the untouched
    original path, which reads it correctly; scale-less int8 stays refused.
  - **Export**: `EXPORT_LAYOUTS["ltx2"]` writes `<root>/transformer/` and
    junctions/copies the rest of the pipeline root beside it — including the two
    loose FILES `model_index.json` and `transformer/config.json`, which
    `link_siblings` copies (directories are still junctioned). An `unwrap` hook
    exports the inner DiT when the generation path has left an
    `Ltx2BlockLoopWrapper` in `ltx2_components["transformer"]`.
  - **MEASURED COST: INT8 turns the coalesced block-swap path OFF.**
    `block_offloading._h2d_setup` builds one flat pinned CPU master per
    swappable block, which requires a SINGLE dtype across that block's Linear
    weights; a mixed set falls back to standard staging-buffer swapping
    (`"[BlockOffloader] H2D-only disabled: mixed Linear weight dtypes"`). An
    int8-converted LTX-2.3 block is int8 + e4m3 + bf16 by construction — the
    conversion is mixed on purpose (high-crest layers stay e4m3) and the 300
    below-gate Linears stay bf16 — so `h2d_only` is ALWAYS off after a
    conversion. This is the combination the feature targets (block swap is the
    standard mode for a 37 GB model), so the H2D win and the INT8 win do not
    compose: you get one or the other. Correct, not a bug — but not free, and
    not measured on LTX-2.3 (the mechanism is arch-independent; the dtype-set
    argument above is what is verified).
  - **Capabilities / LoRA / training**: `arch_capabilities` keeps
    `unet_quantization` listed unsupported (the FP8 values genuinely are) and
    carries `int8` in `ARCH_SUPPORTED_VALUES`, the krea2/ideogram4 treatment.
    ltx2 is also in `QUANTIZED_LINEAR_ARCHS`, which grants it `quantized_gemm`,
    so `quantized_gemm_mode` is threaded through all three video routes
    (`/generate/txt2vid`, `/generate/img2vid`, `/generate/outpaint/video`) and
    the resolved path is recorded as `fp8_gemm` in the video's metadata —
    `extract_fp8_gemm_info` now derives `<arch>_components` from that tuple
    instead of a hand-written map that had already gone stale for FLUX.2.
    `ltx2_adapter.iter_ltx2_lora_targets` uses `is_lora_wrappable_linear`, not
    `isinstance(x, nn.Linear)` — `Int8Linear` is not an `nn.Linear` subclass and
    the naive test drops every quantized target silently. `ltx2_ops
    .load_components` calls `disable_scaled_mm` + `disable_int8_mm` on the
    transformer, the text encoder and the connectors: training is dequant-only.
  - **Training, DiT-BlockSkip** (`blockskip_enable`/`blockskip_front`/`blockskip_back`,
    LoRA and full-parameter trainers only, arXiv 2603.20755): dual-stream
    (video + audio) folded-precompute — a no-grad full pass captures the
    skipped front/back blocks' residual, a grad pass runs only the middle
    blocks. Skipped blocks are gradient-starved (no retained backward
    activations), not optimizer-excluded. Requires `blocks_to_swap == 0`;
    mutually exclusive with TREAD and with stochastic-depth (`block_skip_rate`).
- **minimax_h3** — Joint video+audio generation (`t2va` from a prompt, `fl2va`
  from keyframes placed at named frames and/or an audio track the video is
  generated against, `ref2va` from an ordered list of image, video and audio
  references), plus temporal outpaint. Second video arch,
  loaded and routed separately from the image-model detection like ltx2. The
  denoise loop is repo-owned (`core/models/minimax_h3/h3_pipeline_ops.py`) —
  upstream ships a Modular pipeline only — over vendored, frozen model classes
  (`core/models/minimax_h3/vendor/`). Every number below is measured on an
  RTX 6000 Ada 48 GB (sm_89) with 93.6 GB RAM; the protocols and full tables are
  in `scratchpad/minimax_h3_{k0,phase0t,phase2,phase4}_results.md`.
  - **Two schedules, two grids.** The loop steps the video rows on the shift-12.0
    sigma grid and the audio rows on the shift-3.0 grid, once per iteration —
    the diffusers reference shape, exact on each stream's own grid. (ComfyUI
    integrates the video grid alone and scales the audio velocity by
    `dσ_a/dσ_v` because its sampler knows one schedule; the two agree to first
    order, and that slope was verified against fp64 autograd. The design
    document specified the ComfyUI shape; the shipped code follows diffusers,
    and this row follows the code.) Conditioning rows are pinned for every step
    — `max(t_video, 0.999)` for visual anchors, `1.0` for audio references — and
    the loop only ever writes generated rows. Anchors are built slightly noised
    (`keyframe_noise_aug`), because the released model was trained that way and
    an exactly-clean anchor is off-distribution.
  - **A keyframe anchor addresses an exact pixel frame, and that is measured
    rather than documented.** The packed sequence's time axis is pixel-frame
    time: an anchor's rotary coordinate is `num_text_tokens + (5/3)·f` for any
    frame `f`, so `"first"` and `"last"` are two evaluations of one function and
    `build_packed_layout` takes an integer index (`h3_pipeline_ops.
    _anchor_rotary_time`; the two string branches are kept verbatim because
    `"last"` is numpy's pairwise sum and differs from `(5/3)·(T−1)` in the last
    float64 ulp). `POST /generate/img2vid` exposes it as
    `input_image_frame_index` / `keyframe_images` + `keyframe_frame_indices`,
    with `-1` meaning the clip's last frame after the `17n+5` snap;
    `last_frame_image` is a live alias for an anchor at `-1`.
    - **What was measured**, against a pre-registered criterion (argmin within
      ±2 frames of the requested frame, anchored min RMS ≤ 25, no-anchor
      control ≥ 60): an anchor at frame 60 of a 124-frame clip is the per-frame
      RMS argmin exactly, at **640×384** (min 6.47, control ≥ 73) and at the
      production canvas **1344×768** (min 4.69, control 77.19); three anchors at
      {0, 60, 123} each land on their own frame (per-anchor interior RMS 8.50).
      Anchors bind on the `ref2va` partition too when laid out from its
      post-reference rotary origin (min 11.27, control 67.43), with the image
      reference still binding (CLIP-image cosine −0.0179 against the control,
      +0.0004 outside the anchor's ±8-frame neighbourhood). Protocols and
      numbers: `scratchpad/minimax_h3_c0_results.md`.
      **Replicated through the shipped route** at 640×384 × 124 frames, 6
      steps, one seed (12345): first/mid-60/last/3-anchor {0,60,123} all
      reproduce the harness's argmin-exact result (min RMS 6.44–7.05) and the
      no-anchor control stays far away (min 72.89); the `-1` sentinel and
      `keyframe_images`/`keyframe_frame_indices` resolve to the frames the
      design says they do (`keyframe_resolved_frames` in the response).
      Protocols, numbers and the one substitution it required (the no-anchor
      control has no `/generate/img2vid` request shape and used
      `/generate/txt2vid` instead, per that route's own refusal message):
      `scratchpad/minimax_h3_c2_results.md`.
    - **The scope of that measurement**, stated because it is narrower than the
      feature: one prompt family, seeds 12345/4242, **6 sampling steps** at both
      canvases (the shipped default is 20 — the anchor rows are never denoised
      at any step count, so the mechanism has no step dependence, but that is an
      argument, not a measurement), and video only.
    - **None of it is in MiniMax's model card**, which documents `fl2va` for
      zero, one or two input images at the first and last frame. These are
      properties of the released `fl2va`/`ref2va` weights, measured here;
      nothing asserts a future release preserves them. A request that places an
      anchor at an intermediate frame or sends more than two anchors returns one
      `minimax_h3_undocumented_conditioning` entry in `warnings[]` naming that
      scope, and the timeline control states the same sentence once.
    - **Placement is refused, not approximated, where it is unmeasured.**
      `/generate/img2vid` still answers 400 on a loaded `ref2va` checkpoint,
      because this endpoint carries no reference fields to lay anchors out
      after; the combination lives on `/generate/ref2vid`, whose
      `keyframe_images`/`keyframe_frame_indices` lay anchors after every
      reference block (measured to bind — see the anchor-and-reference
      measurement above). And `/generate/outpaint/video` still refuses a
      mid-timeline clip placement —
      its reason is that the outpaint SHAPE (a preserved clip anchored mid-span
      with exact preservation around it) is unmeasured, not that the
      architecture cannot address a frame.
    - **The geometry rule follows the frame, not the list position.** The anchor
      at frame 0 is stretched onto the canvas (it is the model's geometry
      anchor); every other anchor is aspect-preserving centre-cover-cropped.
      That is a change of rule from "the packed-first anchor is stretched", and
      it leaves video outpaint's own anchors untouched because they arrive
      already at `(width, height)` from `center_crop_resize_frames`.
  - **An uploaded audio track can drive the video (ia2v), whole clip only, and
    that is measured rather than documented.** `AUDIO_COND_TIMESTEP` is 1.0 and
    the forward process is `x_t = t·x0 + (1−t)·noise`, so an audio row supplied
    at t = 1 is **exactly clean** — the literal analogue of the outside of an
    i2i mask. The generated clip's own audio rows already sit on the target's
    rotary clock, so pinning the whole track needs **no layout change at all**:
    `build_packed_layout(pin_target_audio=True)` moves
    `num_condition_audio_rows` from 0 to every audio row, `build_row_timesteps`
    pins them, and `denoise` is left an empty slice to write. Exposed as
    `POST /generate/img2vid`'s `input_audio`.
    - **What was measured**, against the pre-registered A2 criterion (own-track
      flow-energy correlation beats the other track's in ≥ 3 of 4 arms;
      transient → nearest contact ≤ 2 frames for ≥ 8 of 12): at **1344×768 ×
      124 frames, 6 steps**, image + pinned track scored **4/4** and **12/12**,
      and pinned track with **no image at all** scored **4/4** and 9/12
      (`scratchpad/minimax_h3_c0_results.md`). Replicated **through the shipped
      route** at **640×384 × 124 frames, 4 steps**, two seeds × two synthetic
      transient tracks: image + track **4/4** own-beats-other (own r
      +0.347…+0.508, other −0.117…+0.014) and 11/12 transients within 2 frames;
      imageless **4/4** (own r +0.230…+0.585, other −0.148…+0.168), where the
      contact metric is reported as PARTIAL rather than as a pass — the blob
      tracker found the ball in under a quarter of the frames in both
      seed-12345 imageless arms, so only 2 of 4 arms produced a number at all
      (4 of 6 transients over those two). Protocols, raw output, scripts and
      artefacts: `scratchpad/minimax_h3_c3_results.md`.
    - **The scope of that measurement**, stated because a user will assume more:
      the tracks were **impulsive** — sharp broadband transients over silence,
      three per track. **Speech, pitch and timbre were not measured at all**,
      and nothing here says the model follows them. Video only; one prompt
      family; seeds 4242/12345; 4 and 6 sampling steps against a shipped default
      of 20; no run at 20 steps and no peak-VRAM figure for the C3 canvas (it
      was not captured — see the results note rather than assuming C0's).
    - **Whole clip only, and that is the row layout rather than a policy.**
      `num_condition_audio_rows` is a PREFIX count and the audio rows are
      CHANNEL-MAJOR, so pinning "half" of them pins one stereo channel's entire
      timeline, not half the clip in both channels. Partial-timeline placement
      needs the counts generalised to index sets (in the layout dict and in
      `build_row_timesteps`) plus its own measurement; it is refused with that
      reason, not approximated. The UI draws the lane full width and gives it no
      offset handles for the same reason.
    - **A short track is a 400, not a pad.** The required length is the clip's
      audio GRID — `round(T/24·40)` latents × 800 samples — or the video's own
      duration, whichever is longer (124 frames: 165 600 samples = 5.175 s
      against 5.167 s of video). Padding the remainder with silence would build
      a half-pinned, half-silent timeline, which is a shape nothing has
      measured. A longer track is trimmed to the clip, head-aligned.
    - **The output is the SOURCE waveform, not a decode.** The pinned rows are
      never written, so decoding them would only round-trip the upload through
      the audio VAE; `trim_audio_to_video` slices the uploaded samples instead.
      That exactness is of the HANDOFF, and it is asserted by test rather than
      measured: the mp4's audio is **AAC** (`utils/video_utils.py`, as it is for
      a generated soundtrack), so A2's "sample-for-sample" clause is not
      measurable in the file and was rescoped. What the file shows instead, over
      the eight C3 runs: alignment exact at lag 0, RMS deviation 0.0010 (track
      B) / 0.0195 (track A, whose deviation sits inside a full-scale broadband
      impulse) against **0.060 versus the other track**, digitally silent
      stretches preserved as exact zeros, and the decoded audio identical across
      seeds and across the imageless/keyed cases — one hash per uploaded file,
      which generated or decoded audio could not be
      (`scratchpad/minimax_h3_c3_results.md` §5).
    - **The noise draw is unchanged.** All three draws happen in the recorded
      K0.6 order and the audio draw is DISCARDED after the fact, so the video
      noise is bit-identical to a free-audio run at the same seed. This is kept
      **by construction**, not measured live (C0 recorded the same disposition):
      `minimax_h3_ia2v_test.test_the_draw_is_structurally_unconditional` reads
      the shipped function's AST and fails if the draw acquires a conditional
      ancestor, if its arguments mention the track, if anything else in the
      function draws from the request generator, or if the substitution stops
      following the draw. It was validated against the mutant it names (skip the
      draw when the track is pinned, draw video/condition noise separately),
      which fails it.
    - **`input_image` is not required with a track.** The design assumed it was
      and the assumption was wrong: the original ia2v probes had no keyframe at
      all, and pure a2v passes the same criterion. `/generate/img2vid` therefore
      accepts `image`, `input_audio`, or both, and refuses only a request that
      uploads no media (which is `/generate/txt2vid`). `image` stays required on
      LTX-2.3, whose image-to-video pipeline has no other conditioning input.
    - **None of it is in MiniMax's model card.** A request that sends
      `input_audio` returns one `minimax_h3_undocumented_conditioning` entry in
      `warnings[]` — the same entry placement uses, folded together so a
      request that does both is warned once — and the control states the scope
      in one sentence, including that richer audio is unmeasured.
  - **No CFG and no negative prompt, structurally.** Guidance is distilled into
    the weights: there is no unconditional branch and the sampler takes no
    guidance scale, so a step is one forward pass. Both keys stay in the shared
    video request schema and are **accepted-and-warned**, never a 400 — and the
    warning fires only on a **non-default** value, because the frontend always
    sends a full parameter object. `check_arch_capabilities` is passed the
    RESOLVED per-arch video defaults for exactly that reason
    (`api/arch_capabilities.py`, `api/param_defaults.video_defaults_for_arch`);
    the same fix retroactively makes LTX-2.3's video-key warnings honest.
  - **Frame geometry is a hard grid, and there are two different floors.** Valid
    lengths are `17n + 5`; `latent_frames(T) = ceil(T/17)*5 − 3` for `T ≥ 2`
    (`1` at `T = 1`, the spatial-only image-conditioning path). `T = 5` is on the
    grid but **cannot be decoded** through the multi-chunk path — that floor is
    **22 frames / 0.917 s**. `T = 1` is the one exception: `_decode` special-cases
    a lone latent frame (mirroring `_encode`'s own `T = 1` branch) and decodes it
    directly, bypassing the chunk-walk — this is the still-image request
    (`TemporalSpec.allows_single_frame`, see below), not a smaller version of an
    ordinary clip. The production floor is **124**; there is **no enforced
    maximum** — `max_frames` is `None`. **362 is `trained_max_frames`, an
    ADVISORY (not enforced) top**: ComfyUI's node states the trained range as
    "~124–362, longer is untested"; 362 = 17·21+5 = 15.083 s is the grid point
    at that stated top (an earlier version of this bound used 345 =
    17·20+5 = 14.375 s — the largest grid point *below* 15 s, read off the
    README's "output 4–15 s" prose as a strict ceiling — which undersold
    ComfyUI's own stated top by one grid step; 362 is the correction, not a
    relaxation). 362 is not a hard limit: RoPE is computed on the fly (no
    learned position table, no mask, no baked sequence literal), so nothing
    structural stops a longer clip — only the 17n+5 grid is structural. A
    request past 362 is accepted and generated; it only gets an unconditional
    `warnings[]` entry stating the length is untested (no env gate needed to
    reach it, unlike the smoke-side `SUSHI_TEMPORAL_SMOKE` floor override,
    which still exists). An
    invalid `num_frames` (off-grid, or below the 124 floor) is snapped UP to
    the next valid length with a `warnings[]` entry naming the rule — which is
    what the model's own `align_num_frames` does, so a snap never drops
    requested content — **except `num_frames = 1`**
    (`TemporalSpec.allows_single_frame = True`), which is left exactly as sent:
    a still-image request, exempt from the 124 floor entirely (still gets an
    unconditional `warnings[]` entry stating it is below the trained range),
    with `audio_enable` forced to `false` server-side. `POST /generate/txt2vid`
    is currently the only route this reaches (`is_still_image_video_request`,
    `api/generation_utils.py`).
    **fps is fixed at 24** and a different `frame_rate` is forced back with a
    warning. All of this is declarative in `MINIMAX_H3_TEMPORAL`
    (`core/models/components/wiring.py`), read by route validation, bucketing,
    the video loader, the clip-cache key and the `video_constraints` block of
    `GET /schema/arch-capabilities` — there is no `if arch ==` in shared code.
  - **The T=1 decode has an optional, higher-fidelity VAE.** The base video
    VAE's own T=1 branch (`AutoencoderKLMiniMaxH3._decode`) is measured 14-18
    dB PSNR below a second, independently fine-tuned checkpoint trained
    specifically for T=1 reconstruction
    (`minimax_h3_t1_image_vae_step1597.safetensors`, from
    `https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE`; measured at 512x512
    fp16, production normalization, encode/decode through the real production
    pipeline (`h3_pipeline_ops.encode_visual_condition`/`decode_video`), 9
    test images — 5 synthetic patterns plus 4 real generated PNGs from
    `outputs/`). Both load through the
    same `AutoencoderKLMiniMaxH3` class. This checkpoint is OPTIONAL and not
    part of the official release: `detect_minimax_h3_layout`/`_layout_from_root`
    resolve it into an `image_vae` layout/component slot
    (`MINIMAX_H3_IMAGE_VAE_PATTERNS`, `backend/core/models/minimax_h3/loader.py`)
    that is never in the required-component check, so every install without
    it keeps loading exactly as before; a PRESENT but malformed file degrades
    to `image_vae = None` (logged) rather than failing the whole model load.
    Adds ~5.2 GB host RAM to every H3 load when the file is present, including
    training and geometry-probe loads that never decode a frame — there is no
    lazy-load, so a host RAM budget that assumes the pre-image-VAE footprint
    needs revisiting once this file is installed. `select_minimax_h3_decode_vae`
    (`backend/core/pipeline_backends/minimax_h3.py`) prefers `image_vae` for a
    `latent_frames == 1` decode when the install has it and its embedded
    contract metadata (`h3_t1_format`, `h3_t1_output_slice`) matches this
    build's `frame_pre_padding`; otherwise it falls back to the video VAE and
    emits ONE `warnings[]` entry (`minimax_h3_still_image_default_vae_fallback`)
    stating the measured gap. Any `latent_frames != 1` (ordinary video) request
    always decodes through the video VAE and never triggers this warning. The
    Mamad8 model card states only "the applicable MiniMax H3 license and
    terms," with no further concrete license text.
  - **`num_inference_steps` counts sigma GRID POINTS, so N steps = N−1 model
    evaluations**, and the vendored scheduler refuses `N = 1`. The minimum is
    validated at the route next to the geometry check, because leaving it to the
    sampler turned a bad request into an HTTP 500 that had already paid for a
    full text encode. **There is no official step count for this model**: MiniMax
    publishes none (their reproducible scripts are HTTP calls to their own server
    and expose no sampler knobs) and the 50 in the diffusers examples is that
    library's generic template default. The shipped 20 is a **community
    baseline** and is described as exactly that everywhere it is user-visible
    (`VIDEO_GEN_ARCH_OVERLAYS["minimax_h3"]`).
  - **Shipped weight formats, and why.** The released checkpoints are the
    "pruned" variant: they carry **no `time_embedder.*` keys at all** and instead
    an `adaln_t_table [1025, 8]` buffer plus per-block `adaln_proj.linear` of
    width 8. Upstream diffusers implements only the full-modulation AdaLN, so the
    curve lookup was **ported from ComfyUI** (Apache-2.0, attributed) into the
    vendored transformer and is **bitwise identical** to ComfyUI's lookup on all
    1025 grid points, all 1024 midpoints and the endpoints (K0.2). Two traps the
    loader honours: the SiLU is **baked into the table** (applying it again is
    silently wrong) and `adaln_proj` runs in fp32. The transformer config is
    therefore **synthesised from the file's own header**: the published
    `transformer/config.json` describes the full-modulation variant and must not
    be applied to a pruned file. The generation file is `*_pruned_fp8_scaled` (21 GB,
    weight-only FP8): its 200 quantized tensors become **300 live `Fp8Linear`
    modules** because the loader splits each fused qkv (both counts are printed
    on load; quote them rather than deriving one from the other).
  - **Two opposite QKV conventions in one distribution.** The DiT single file is
    already `[q|k|v]` CONTIGUOUS and must NOT be de-interleaved (this contradicts
    the upstream conversion script's premise); the video VAE decoder's `to_qkv`
    IS per-head interleaved and MUST be. Both were discriminated by measured RoPE
    row-norm signatures. Getting either backwards produces a model that loads
    perfectly and generates noise. Also at load: SwiGLU halves are swapped
    (`[gate; up]` → `[hidden; gate]`), the audio VAE's pre-folded weight norm is
    removed from 172 modules first (else 268 missing / 134 unexpected keys), and
    a three-rule prefix rewrite maps the text encoder onto transformers'
    `Qwen3VLForConditionalGeneration` with exactly two missing keys
    (`lm_head.weight`, `model.language_model.norm.weight`), neither of which the
    layer-50 read uses. All four component loads pass through
    `models/common/quantized_checkpoint_guard` before any tensor is installed.
    The DiT loader accepts only the released `int8_tensorwise` ConvRot contract
    (groupsize 256); unsupported declarations and quantized TE/VAE files remain
    refused rather than silently mis-loaded.
  - **W8A8 is off for the whole architecture, permanently for this file.** The
    fp8 sidecars are per-tensor SCALARS (broadcast at load, not the
    `(out_features,)` vector `Fp8Linear` expects); 50 of the 200 quantized
    tensors — exactly the 50 `mlp.fc2` — carry
    `{"format":"float8_e4m3fn","full_precision_matrix_mult":true}`, i.e. the
    writer declares their product must not be computed in fp8; and the other 150
    carry an `input_scale` this repo's `Fp8Linear` does not read. The loader
    therefore calls `disable_scaled_mm` over the whole DiT, which outranks the
    `SUSHI_FP8_SCALED_MM` env flag and the `quantized_gemm_mode` request alike, so
    `"w8a8"` is accepted and resolves to dequant with a `quantization_fallback`
    warning naming the resolved path. `minimax_h3` is in `QUANTIZED_LINEAR_ARCHS`
    (hence its LoRA target predicate goes through `is_lora_wrappable_linear`, not
    `isinstance(x, nn.Linear)`, which would silently drop all 300 `Fp8Linear`
    targets) but deliberately **not** in `RUNTIME_INT8_ARCHS`: the shipped DiT is
    already quantized, so there is no unquantized transformer to convert. The
    reason is recorded in `ARCH_QUANT_POLICY["minimax_h3"]` rather than left as an
    absence.
  - **Packed W4A8 checkpoints use a separate runtime contract.** Their 200
    source Linears are packed INT4 plus FP8 group scales, FP32 channel scales,
    a 16-entry codebook and ConvRot metadata. QKV expansion produces 300 live
    `W4A8Linear` modules. The loader applies the same output-row split or
    SwiGLU half-swap to every row-indexed sidecar and executes them through
    Comfy-Kitchen 0.2.28 without expanding the stored weights to BF16. Inference
    uses its packed operator; LoRA backprop dequantizes only the live layer for
    an autograd-visible `F.linear`. ReLoRA is refused because merging a dense
    delta into packed W4A8/FP8 storage requires format-specific requantization.
  - **INT8 ConvRot checkpoints retain rotated weights.** Their 200 source
    Linears carry per-output-row FP32 scales and exact `.comfy_quant` markers;
    QKV expansion produces 300 live `ConvRotInt8Linear` modules. Inference uses
    Comfy-Kitchen's online input rotation without reconstructing a dense BF16
    weight. The autograd path un-rotates only the live layer so LoRA input
    gradients remain correct. The marker stays in module state so the rotation
    contract cannot be silently lost.
  - **Quantized inference, API-verified** (640x384x124, two schedule points =
    one model evaluation, seed 12345, Flash Attention, audio decode off):

    | checkpoint | denoise | denoise peak | recorded operator |
    |---|---:|---:|---|
    | FP8 scaled | 6.901 s | 21.265 GB | `dequant` |
    | W4A8 mixed | 3.923 s | 13.394 GB | `w4a8_int8(comfy-kitchen)` |
    | INT8 ConvRot | 2.351 s | 21.266 GB | `convrot_int8(comfy-kitchen)` |

    W4A8 saves 7.871 GB because it stores 4-bit weights. ConvRot and FP8 both
    store 8-bit weights, so their equal resident footprint is expected; the
    ConvRot win is its fixed online-rotation GEMM rather than lower weight bits.
    All three gallery rows recorded `attention_backend=flash`, proving the
    global/per-request backend reaches the H3 attention conduit.
  - **Quantized ref2va outpaint stays proportional to packed rows.** Extending
    a 124-frame 640x384 clip by a 124-frame generated span produced 19,920
    packed rows, including 9,120 conditioning rows. W4A8 peaked at 15.176 GB,
    only 1.782 GB above its target-only run; FP8 peaked at 23.048 GB, 1.783 GB
    above target-only. The former ~60 GB behaviour is not reproducible after
    keeping the source reference on the requested canvas instead of upscaling
    it to the reference model's maximum canvas.
  - **Generation, measured** (seed 0, the official t2va prompt, `num_frames=124`,
    20 steps = 19 evaluations, fp8 DiT resident, no block swap, one discarded
    warm-up then the median of 3 runs, driven through
    `POST /generate/txt2vid` on **native SDPA** — no attention backend was
    registered yet at that point):

    | point | median wall | text encode | denoise | s/eval | decode | peak VRAM |
    |---|---|---|---|---|---|---|
    | 960×544×124 | **431.5 s (7:12)** | 27.5–33.3 s | ~353 s | 18.6 | ~20 s | **24.25 GB** |
    | 1344×768×124 | **1052.5 s (17:32)** | 33.8–36.8 s | ~939 s | 49.5 | ~38 s | **28.73 GB** |

    The text-encode peak (1.85 GB) is canvas-independent by construction. The
    denoise peak is ~21 GB of resident fp8 weights plus a transient of ~3.2 GB
    and ~7.7 GB respectively. The shipped default canvas is 1344×768×124 and is
    not capped.
  - **Attention conduit, measured** (960×544×124, 20 steps, same protocol, one
    process, the backend that actually ran read off the conduit's own log line):
    flash **−13.7 % wall / −14.9 % denoise**, sage **−18.9 % wall / −23.3 %
    denoise**, both against native SDPA. Peak VRAM is 23.04 GB for all three —
    the peak is resident weights plus packed-sequence activations, not the
    attention workspace. **The default is still `normal` (native)**: sage is an
    INT8 kernel and flash-2 reassociates the softmax accumulation, and that phase
    measured speed, not image quality.
  - **Block swap costs no measurable step time here** (same point, sage
    throughout): `blocks_to_swap` 25/50 → **14.52 GB peak (−37 %)**, 40/50 →
    **9.07 GB (−61 %)**, with every denoise figure — swap and no-swap — landing
    inside the 300.7–313.1 s no-swap spread. At ~15 s per model evaluation the
    PCIe transfer hides completely. It stays **opt-in and off by default**,
    because the DiT fits resident at both registered canvases. `h2d_only` is
    deliberately off: a block mixes `float8_e4m3fn` `Fp8Linear` weights with a
    float32 `adaln_proj.linear`, so the coalesced flat master would be refused for
    mixed dtype anyway. The offloader is built and torn down **per generation**,
    because the DiT leaves the GPU at the end of every generation (the video
    VAE's 36-layer ViT decoder and the DiT do not fit together).
  - **FBCache is opt-in approximate acceleration with H3-specific guards.** The
    first implementation used one global mean over every generated video row,
    allowed unlimited hit chains and protected no tail step. At thresholds
    0.08/0.12/0.20 it skipped 42%/63%/84% of evaluations and changed the
    same-seed trajectory substantially (best LPIPS 0.263 / SSIM 0.656), although
    blind re-review of the 0.08 clips found no subject loss, freeze, black frame
    or structural collapse. The restored path keeps the public threshold and
    warmup controls but adds three fixed safety rules: max-per-latent-frame
    relative-L1 must also pass, at most two consecutive hits are allowed, and
    the final evaluation always runs. The indicator excludes reference/keyframe
    rows; a hit reuses the entire packed-state residual so video and audio share
    one decision. It is disabled with Block Swap and Spectrum. `0.08` is the
    recommended safe starting threshold; the shared UI default `0.12` is more
    aggressive on H3. LPIPS/SSIM are trajectory-distance diagnostics for this
    explicitly approximate option; release evaluation first rejects integrity
    failures, then uses blind prompt/motion/consistency review against speed. A
    640x384x124, 20-step W4A8 + Flash smoke at threshold `0.08` reduced denoise
    from 62.689 s to 46.631 s (25.6%) while retaining SSIM 0.9912, PSNR 39.33 dB
    and LPIPS-Alex 0.0175 mean / 0.0240 max across all 124 decoded frames.
  - **Spectrum/SFF is opt-in approximate acceleration.** The H3-owned denoise
    loop forecasts final video and audio velocities with two forecasters on one
    shared anchor schedule; a forecast skips the whole transformer call while
    both schedulers still advance. At 640x384x124, 20 steps, W4A8 + Flash
    Attention, 11 actual forwards plus 8 forecasts cut denoise from 62.481 s to
    36.884 s (41%). Same-seed LPIPS 0.325 and RGB SSIM 0.671 record substantial
    trajectory divergence, but a visual re-review found the requested red cube,
    dark background and continuous slow rotation intact, with no black frame,
    freeze, subject loss or geometry collapse. The 18-forward/1-forecast arm
    measured LPIPS 0.259 / SSIM 0.770 and saved only 2.468 s, so it is coherent
    but not a useful Pareto point. LPIPS/SSIM remain diagnostics for same-seed
    reproducibility, not a quality veto for an explicitly approximate mode.
    Release evaluation uses hard integrity failures first, then blind prompt
    adherence, subject consistency, temporal coherence and usability
    non-inferiority against the speedup. Spectrum defaults off and is mutually
    exclusive with block swap: a skipped forward cannot service the offloader's
    per-block prefetch rotation.
  - **The video VAE's tiling policy is load-bearing, not a memory knob.** With
    the same weights and the same input, flipping only the shipped tiling flags
    moved the **latents by rel-RMS 0.355** (384×384) / **0.0952** (640×384) and
    the **decode by rel-RMS 0.212**. The policy is therefore PINNED in the loader
    (`MINIMAX_H3_VAE_TILING_POLICY`, 256 px tiles / 64 px overlap), reported in
    the component dict, and is **part of the training clip-cache key**
    (`LatentCache.compute_clip_hash(..., tiling_policy=...)`) so cached latents
    cannot silently disagree with what generation produces. It is **not** the
    user-facing `vae_tiling` generation parameter and must never be wired to it:
    that one trades peak VRAM, this one changes the output. Any A/B on this VAE
    has to hold tiling fixed or it measures tiling instead of the thing under
    test — which is how the fp16-vs-fp32 decode comparison was run: fp16 weights
    are **77.74 dB PSNR / 2.764e-04 rel-RMS** against fp32 weights on a real
    decode (2.2 s / 5.19 GB vs 7.3 s / 13.33 GB), so fp16 is the shipped dtype.
    Still open and not claimed: upstream keeps fp32 weights under
    `torch.autocast(float16)`, which is a different computation from casting the
    weights, and that comparison has not been run.
  - **LoRA training** (`arch/minimax_h3.py` + `ops/minimax_h3_ops.py` +
    `adapters/minimax_h3_adapter.py`). Targets are the per-block attention
    projections (`to_q/to_k/to_v/to_out.0`) and the SwiGLU FFN linears
    (`ff.net.0.proj`, `ff.net.2`) across all 50 blocks — **300 modules /
    83.1 M trainable parameters at rank 16** (measured; that this equals the
    number of live `Fp8Linear` modules is a coincidence of 6 leaves × 50 blocks,
    not the same set). Permanently excluded,
    with reasons, not deferred: the modality I/O heads
    (`proj_in`/`audio_proj_in`/`proj_out`/`audio_proj_out`, structurally
    load-bearing for the packed-sequence split), the 2-layer `token_refiner`
    (its output conditions both modality heads and upstream documents no training
    formulation for it), and AdaLN (a frozen table plus a projection). Measured on
    the registered matrix (batch 1, gradient checkpointing, fp8-resident base with
    dequant inside the forward, AdamW lr 1e-4): **384×640×22 → 5.15 s/step median
    and 23.08 GB peak** over a 100-step run (a single step measured 22.45 GB
    peak / 4.9 s), and all four registered cells fit — 384×640×39 8.63 s /
    24.11 GB, 512×768×22 7.93 s / 23.89 GB, 512×768×39 14.47 s / 25.63 GB. Cost
    is close to linear in packed-sequence length (~2.9 ms per row per step,
    ~+0.53 GB peak per 1000 rows). No memory-escalation rung was needed;
    gradient checkpointing is what makes dequant-in-forward affordable, and the
    two together are the load-bearing interaction. A 100-step numerics smoke on
    identical repeated inputs produced no non-finite value, no dead block group,
    and loss −20.95 %; a saved adapter reloaded onto a freshly built model
    reproduced the pre-save forward **bitwise**.
  - **`audio_loss_weight` (default 1.0) is a joint objective, not a mixing
    knob.** H3 is one stream: every LoRA-targeted weight is shared by the audio
    path, so a video-only objective still moves audio behaviour. The loss is
    `video_mean + audio_loss_weight * audio_mean` with each modality's velocity
    MSE averaged over tokens, channels and samples **before** weighting, so the
    weight's meaning does not depend on the ~20× difference in row counts. 1.0 is
    what the design's pre-registered three-regime experiment selected (200 steps
    per regime, fixed dataset, fixed evaluation draws): joint real-audio loss
    reached a held-out VIDEO loss 0.44 % better than the video-only regime
    (inside its "not worse by 2 %" bar) while its held-out AUDIO loss was 19 %
    lower. `0.0` reproduces a video-only objective and is exposed because a
    dataset whose audio is voiceover, music or editing artefacts is a real case.
  - **The per-modality split of that combined loss is charted, not just the
    total.** `train_step` logs `h3_video_loss`, `h3_audio_loss` and
    `h3_audio_present` (the batch fraction that carried a real audio track
    rather than the zero-weighted fallback) through the extra-metrics
    mechanism (`metric_registry.EXTRA_METRIC_DEFS`, `WS_PROTOCOL.md`), not as
    DB columns — a flat-zero `h3_audio_present` explains a flat-zero
    `h3_audio_loss` instead of leaving it mysterious. A standalone
    `item_type=="audio"` dataset item has no encode path on a video arch (its
    audio comes from a paired video item's own track) and is refused at
    trainer setup, before any GPU work, rather than failing mid-run on a PIL
    "cannot identify image file" error; the guard is generic over both video
    archs.
  - **Full fine-tuning is refused**, in three live layers: the
    `TRAINING_UNSUPPORTED["minimax_h3"]["full_finetune"]` declaration served by
    `GET /schema/arch-capabilities` (which the training UI filters its method
    dropdown from), the deliberate absence of a `MiniMaxH3FullParameterAdapter`
    class, and a hard `ValueError` in `full_parameter_trainer` —
    `_refuse_unsupported_full_finetune` raised before `super().__init__()` loads
    anything, with the adapter-selection branch as a backstop so the missing
    adapter can never fall through to SD1.5's.
    Reason: a 33 B dense DiT's parameters, gradients and optimizer state do not
    fit the single-GPU 48 GB envelope this integration targets.
  - **Training clips use the grid, not the production bounds.** Default clip
    lengths are `(22, 39)`; bucketing validates against the grid plus the 22-frame
    decodable floor, so a short training bucket is not a violation of the
    124-frame production floor (there is no enforced maximum; 124–362 is the
    DOCUMENTED, advisory-only generation range). Clip sampling on this arch is
    **timestamp-based** because
    `fps_fixed` is set — target frame *i* is the source frame nearest
    `start_time + i/24`, and the audio window is cut by the same timestamps, so
    A/V stay aligned by construction. The clip-cache key carries `source_fps`,
    `target_fps`, `resample_policy`, `start_time`, the tiling policy and an audio
    prep version; LTX-2.3's legacy seven-argument keys are byte-identical to
    before.
  - **Image datasets train natively at `T_lat = 1`, and the transfer to video is
    unvalidated.** A still is encoded as a 1-frame clip
    (`ops/minimax_h3_ops.vae_encode`: `[-1,1]` → ImageNet-normalised `[0,1]`,
    posterior **mode**, `[1, 24, 1, H/16, W/16]`) and goes through the same 5-D
    `train_step` as a clip. `_pixel_frames_for` returns 0 pixel frames at
    `T_lat = 1`, so a still carries **0 audio rows** rather than the 74 the
    17n+5 grid inversion produced, and the audio term is the exactly-zero branch.
    Measured facts: a `T = 1` encode and latent frame 0 of a real 22-frame clip
    agree to rel-RMS 0.0005 with per-channel correlation 1.0000 (the encoder is
    causal with `frame_pre_padding = 3` and `temporal_compression_ratio = 4`, so
    latent frame 0 is a function of pixel frame 0 alone); `n_aud = 0` survives
    the DiT forward, with finite loss and finite nonzero gradients in all five
    block groups; 384×640 `T_lat = 1` costs **1.06 s/step and 23.75 GB peak**
    against 4.36 s / 25.01 GB for `T_lat = 7` in the same process — 4.1× cheaper
    in time and 1.25 GB lower, because the resident base and the per-forward FP8
    dequant dominate and neither scales with sequence length. **Not measured:**
    whether a stills-trained LoRA transfers to `t2va`/`fl2va` output, and whether
    it reduces motion. The pre-registered experiment that would answer both
    (`scratchpad/minimax_h3_q1_overturn.md` §8: CLIP-image transfer ≥ 0.05 and a
    Farneback optical-flow floor of 0.80× base) has not been run. At inference the
    video block is always `T_lat ≥ 7`, so a `T_lat = 1` block is a 304-row
    sequence where training on clips is 1818 rows; whether attention behaves
    comparably at that length is untested.
  - **`audio_enable=false` is H3-specific**: it skips the audio VAE decode and
    the mux. The audio rows still ride the packed sequence and still influence
    video through self-attention, and the flag does not perturb the noise draw
    order (one draw per visual condition in packed order, then the video noise
    as a 5-D latent, then the audio noise directly in ROW layout — drawing it
    per-channel and permuting gives different numbers from the same seed) —
    verified by hash for t2va, fl2va with 1 and 2 keyframes, and
    `audio_enable=False`. On
    LTX-2.3 the same flag only discards audio after the pipeline returns it.
    H3's audio is not independently addressable, so `/generate/txt2aud` refuses
    an H3 model with that reason rather than the generic "no ACE-Step model".
  - **The conditioner's vision path owes `Qwen3VLModel.forward` three things**,
    and a hand-written decoder loop drops two of them by default: the merged
    vision rows scattered into the embeddings at the `<|image_pad|>` /
    `<|video_pad|>` placeholders; **deepstack**, three intermediate tower feature
    maps ADDED to the visual rows after each of the first decoder layers; and
    **mrope**, the 3-D `(t, h, w)` positions `get_rope_index` builds for a
    sequence containing a vision block, in place of `arange`. All three are
    implemented in `h3_pipeline_ops.encode_prompt`. **Omitting either of the last
    two produces finite, correctly-shaped and silently wrong conditioning**, so
    the K0.7 probe — which spliced merged vision rows in and ran plain 1-D
    positions — does not stand as verification of this path, which is the thing
    it was meant to rule out. What K0.7 does establish is the layer-streaming
    measurement: `functional_call` off the mmap at **49.82 GB flat RSS and
    13.5 s/prompt**, against **73.08 GB peak, pagefile growth and 46 s/prompt**
    for the `layer.to(cuda)`/`layer.to(cpu)` shape, on this box.
  - **Omni-reference generation (`ref2va`) runs on a SECOND transformer, and
    which file is loaded IS the workflow.** `POST /generate/ref2vid` conditions a
    generation on an ordered list of image, video (each with an optional
    positional soundtrack) and standalone audio references; the request surface,
    the per-modality and total limits and every refusal are in `openapi.yaml`,
    the limits themselves are constants at the top of
    `core/models/minimax_h3/h3_references.py`, and the media normalisation lives
    in that module. It is a dedicated endpoint rather than more files on
    `/generate/img2vid`: that route pins 1–2 keyframes to the ends of the
    generated clip on its canvas and both video archs serve it, while this one
    takes heterogeneous files of three modalities at their own resolutions and
    rates and only one transformer partition implements it.
    - **The variant is read off the filename because nothing else distinguishes
      the two files**: both released DiTs are exactly 20,958,205,608 bytes, both
      have their config synthesised from their own header, and neither carries
      metadata. `detect_minimax_h3_layout` puts `fl2va`/`ref2va` in
      `layout["variant"]`, the loaded components carry it into
      `current_model_info`, `GET /models/current` reports it as
      `model_info.variant`, and `/generate/ref2vid` refuses any other variant by
      naming the file to load instead — a mismatch cannot be detected from the
      weights and would produce a bad video rather than an error. `GET /models`
      expands an H3 tree into one entry per DiT file carrying `variant` (the
      MiniT2I treatment) instead of listing the tree once and silently resolving
      to the first file. **Caveat:** that expansion only runs for a tree sitting
      directly under a configured models directory (`settings.models_dir` plus
      the user's `model_dirs`), and it has not been exercised live here, where
      the tree is outside those directories and the file is loaded by path.
      Detection itself does not need a root `model_index.json`:
      `_looks_like_minimax_h3` also accepts a `diffusion_models/` folder holding
      a DiT with the key signature.
    - **The reference ORDER is semantic, twice over**, so nothing sorts or
      regroups a request: it numbers the `<Picture i>` / `<Audio j>` /
      `<Video k>` labels the prompt refers to, and it lays the references out on
      the packed sequence's shared rotary clock.
    - **Layout**: `[text | reference blocks in request order | target audio |
      target video]`. Each block advances one shared clock — an image by exactly
      1.0 (a single integer slot, not a latent frame's 5/3), a standalone audio
      block by its latent count on the target width grid, and a video reference
      by `max(audio_latents, video_span)`, with its soundtrack rows packed
      immediately BEFORE its own video rows from the same clock origin.
      `build_ref2va_packed_layout` was established the way K0.3 established the
      `t2va`/`fl2va` layout — against an independent port of ComfyUI's
      `PackedLayout`, on seven configurations, comparing sequence length, ORDERED
      index tensors, conditioning row counts and the float64 position grid
      (≤1e-4 after the shipped fp32 cast) — and is pinned as literals in
      `backend/tests/minimax_h3_layout_test.py` alongside reordering and
      soundtrack-placement controls. Because every reference row precedes every
      generated row of its OWN modality, `video_indices`/`audio_indices` still
      lead with their conditioning rows, so **`build_row_timesteps` and the
      denoise loop needed no change at all**.
    - **A reference is not free, and a video reference cannot be made cheap** —
      its rows ride through every sampling step. `reference_image_size` (default
      `max`, the released recipe) governs IMAGES only: `max` puts each image on a
      2048-pixel short edge of its own, upscaling included and with no area cap,
      so a square one contributes 4,096 rows against the 8,880-row target of a
      640×384×124 generation, while `match` scales it down to the generation's
      pixel area. A video reference is put on the 768-short-edge canvas its own
      aspect ratio resolves to (`resolve_canvas_size`) regardless of source size,
      upscaling included, so a 124-frame 16:9 one is 37 latent frames × 1,008
      rows = **37,296 rows, more than the target itself**, and no parameter
      reduces it.
    - **Two refusals rather than approximations**: an audio reference cannot be
      the only kind sent (a standalone soundtrack never reaches the Qwen3-VL
      conditioner, so such a request conditions the vision stream on nothing),
      and a reference video shorter than 22 frames once resampled to 24 fps and
      truncated to the generated length is refused rather than snapped —
      upstream's `max(1, (n−5)//17)*17+5` claims 22 frames while feeding fewer,
      and the floor comes from two directions at once (the video VAE's 17n+5
      chunk grid, and the conditioner needing ≥13 frames to fill one merged
      vision block).
    - **Measured** (four live generations through the endpoint at 640×384×124 on
      the ref2va file, `blocks_to_swap=0`, fp8 DiT on the dequant path, empty
      `warnings[]`; the gallery rows of all four are in
      `scratchpad/h3_ref2vid_run{1..4}.json`, and two were timed): one image
      reference at 6 steps (5 evaluations) **190 s wall / 22.65 GB peak**; one
      image + one video
      reference with its soundtrack, at 4 steps (3 evaluations) and
      `reference_image_size=match`, **177 s / 23.65 GB**. **These two are not a
      like-for-like pair** — different step counts and different image sizing —
      so they bound the observed cost rather than isolating a video reference's.
      Loading the ref2va file took 26 s, with system RAM peaking at 64.1 GB used.
  - **Temporal outpaint conditions on ONE boundary frame**, and the endpoint's
    shape is entirely that fact. H3 has no analogue of
    `LTX2VideoCondition.index` (no index-addressable conditioning) and no
    denoising-strength video-to-video path, so a clip sitting in the MIDDLE of a
    longer timeline cannot condition what is generated around it: the endpoint
    serves exactly **extend-forward, extend-backward and bridge**, and refuses
    any other placement with that reason instead of approximating it. Two
    consequences: the preserved frames are **not generated at all** (the output
    is a concatenation, so exactness is by construction rather than by a
    corrective paste, and the `17n+5` rule binds the generated span only), and
    the anchor frame is not emitted twice (extending a P-frame clip by a G-frame
    span yields `P + G − 1` frames; a bridge drops both ends).
    **Limitation, stated plainly because the measurement says to:** the model
    receives that one boundary frame and never the motion or context behind it,
    so **subject identity across and beyond the join is not guaranteed**, and a
    defocused, motion-blurred or occluded boundary frame is a weak anchor. In the
    registered seam review (3 clips fixed and hashed before any H3 outpaint code
    existed, extend-forward 124→248, seed 0) the **seam itself was invisible in
    all three clips** to an owner reviewing them unmarked; the one drift that was
    noticed happened **inside the generated span** after a focus excursion, which
    the registered optical-flow metric treats as interior baseline and, by
    construction, never tests. That metric has live error in both directions — it
    failed a clip whose seam was fine (a camera event near the boundary raises
    its delta) while being structurally incapable of seeing a subject-identity
    break a few seconds later — so any future seam metric needs an appearance
    term as well as a motion term, and should measure the whole generated span.
    **No seam-hider is added**, and reference conditioning (`ref2va`, now
    implemented — see the omni-reference bullet above) is still not offered as
    the fix for this: what was measured there is that reference conditioning
    reaches the model, not that it holds identity across a join, and it is a
    different feature that would have to earn that claim on its own measurement.
  - **`/generate/outpaint/video` accepts `reference_images` on `ref2va`,
    `extend_forward` only, after a gate that failed and a decision made past
    the failure.** A registered acceptance gate (A-V8: boundary-anchor bind,
    reference bind, K5's seam criterion) was run twice on this exact endpoint.
    Attempt 1 shipped a different mechanism (routed to `/generate/ref2vid`,
    regenerating every frame) and was reverted because two of the three
    criteria had no preserved/generated boundary to read
    (`scratchpad/minimax_h3_av8_results.md`). Attempt 2 ran on the actual
    `extend_forward` surface with every criterion evaluable
    (`scratchpad/minimax_h3_av8_run2_results.md`): boundary-anchor bind passed
    (5.226 RMS against a bar of 25), reference bind passed (CLIP-L/14 mean
    cosine 0.96466 vs the no-reference arm's 0.85961), paste exactness passed
    (max\|diff\| 0), and **the seam criterion failed** — 0.526 against the
    arm's own interior p95 of 0.428. Per the registered disposition the
    feature did not ship on that result, and the design's prior "permanently
    out of scope" declaration (`scratchpad/minimax_h3_design.md` §11) was
    reinstated. **Measured context, not a mitigation**: the same seam metric
    at the same protocol rejects the untouched no-reference extend even
    harder on this clip (0.909 against its own p95 of 0.255) — the metric
    does not discriminate the feature from the baseline it was added to, but
    the registered criterion was against each arm's own interior p95, and the
    reference arm exceeded it, so this does not reverse the fail.
    The repo owner then reviewed three `extend_forward` clips by eye (no
    threshold, `scratchpad/minimax_h3_ab_visual_arms.md`) and judged the
    no-image-reference arm (source clip carried as the sole video reference)
    good; the arm with an added image reference showed the reference acting
    as a keyframe at the head of the extension, which was traced to the image
    reference landing 1.0 rotary-clock unit before the boundary anchor,
    inside its measured ~3.33-unit binding radius, and fixed by packing image
    references away from that instant. A rerun after the fix moved the
    boundary-anchor RMS from 28.679 to 5.0703 and removed the reference-snap
    signature the owner had named. **The owner then accepted the shipped
    surface on that visual review — A-V8 itself remains failed and closed;
    the feature did not ship because a gate passed.** `reference_images` is
    gated to `ref2va` + `extend_forward` only (`extend_backward` and `bridge`
    with references are refused, unmeasured); no UI or doc text claims the
    reference holds character identity across the join — only that the model
    reads it (criterion 2's own scope).
  - **`POST /generate/inpaint/video` regenerates a contiguous mid-clip span and
    pastes the rest of the input back exact after decode** — the complement of
    temporal outpaint's boundary-only shape above. `fl2va` only (same partition
    gate as `/generate/img2vid` and `/generate/outpaint/video`); the requested
    `[regenerate_start_frame, regenerate_end_frame)` range is expanded OUTWARD
    to the video VAE's latent-group boundaries, exposed declaratively as
    `video_constraints.latent_chunk_pattern` (`[1, 4, 4, 4, 4]` on this
    architecture) so a client-built request already matches what the server
    runs. Full parameter surface and behavior: the route's own docstring in
    `backend/api/routes.py` and `openapi.yaml`.
  - **Attribution**: the UI displays this architecture as "MiniMax H3", which the
    model's license requires (`archDisplayName` in `frontend/src/utils/api.ts`,
    `_ARCH_DISPLAY_NAMES` in `int8_runtime_quantize.py`).
- **sensenova** — SenseNova-U1.5-8B-MoT, vendored under `models/sensenova/vendor/`
  the same way as MiniMax-H3 (model classes only; SushiUI owns the denoise
  loop, driving the vendored `NEOChatModel` from `sensenova_pipeline_ops.py`
  rather than calling upstream's `t2i_generate`). Apache-2.0,
  `github.com/OpenSenseNova/SenseNova-U1` branch `feat/u1.5`.
  - **Sampling direction is the trap**: `t=0` is noise and `t=1` is clean —
    the OPPOSITE of Z-Image/FLUX.2's convention in this repo. img2img/inpaint
    follow MiniT2I (the other pixel-space, no-VAE arch), not flux2: img2img is
    SDEdit from `t_start = 1 - denoising_strength`, inpaint is RePaint
    (re-pins the known region each step against noise drawn once).
  - **Ships already int8-quantized**: the conversion script
    (`quantize_transformer_fp8`-adjacent, arch-specific) quantizes exactly the
    588 Linears of both MoT branches — 42 layers x {q,k,v,o}_proj + their
    `_mot_gen` twins + both MLPs — to per-row int8 weight-only; `lm_head`,
    embeddings, all norms, the flow-matching pixel head and both vision patch
    embeds stay bf16. Source 46.8 GiB -> converted 17.6 GiB, 5 shards. In
    `QUANTIZED_LINEAR_ARCHS` but deliberately NOT `RUNTIME_INT8_ARCHS` — no
    unquantized transformer exists for the in-place converter to act on (same
    reasoning as `minimax_h3`); `unet_quantization` is listed unsupported for
    this reason.
  - **8-step distillation LoRA**: applied over the already-int8-quantized base
    via `Int8Linear`-aware LoRA (an `isinstance(x, nn.Linear)` predicate would
    silently drop every target), never merged, restored in the same `finally`
    that tears down the KV caches. Targets exactly the 294 gen-branch Linears
    the conversion quantizes. `cfg_scale <= 1` is what drives the 8-step path
    (upstream's own `needs_cfg = cfg_scale > 1` collapses CFG to a single
    branch); there is no separate mode flag.
  - **Resolution**: free, not bucketed; only the structural /32 token grid
    (patch 16 x merge 2) is enforced by snapping. The 11 upstream ~4 MP
    training-resolution buckets ship as UI presets (starting points, not
    restrictions); an off-bucket size warns (`sensenova_resolution`) and
    generates anyway.
  - **Reference-image editing (`ref_images`) is implemented**, as upstream's
    `it2i_generate`: references are preprocessed by the vendored
    `load_image_native` (**ImageNet** normalization — NOT the 0.5/0.5 the
    generation branch uses), read by the *understanding* vision tower
    (`transformer.vision_model`, distinct from the per-step
    `vision_model_mot_gen`), and spliced into the prompt as
    `<img><IMG_CONTEXT>xN</img>`. It is a prefix-phase concern, so it composes
    with txt2img, img2img (SDEdit) and inpaint (RePaint) alike rather than
    being a separate mode. `img_cfg_scale` selects the branch count: 1 branch
    when both scales are 1, 2 when only one exceeds it, 3 otherwise
    (InstructPix2Pix form). A `negative_prompt` rides the uncond branch when
    one exists and otherwise the img_cond branch, which is the blend's
    baseline at the default `img_cfg_scale=1`
    (`sensenova_negative_prompt_on_img_cond`) — a deliberate deviation from
    upstream, which conditions that branch on the images alone.
    - **Measured** (5 refs, 2048x2048, 3-branch): peak **~39.2 GB** of 48 GB;
      prefix cost **+5.17 s** for 5 references vs no references.
    - **Reference count is capped at 5**, upstream's largest demonstrated
      count, as a typed 4xx.
    - **References are downscaled to 1.05 MP** before encoding
      (`REFERENCE_IMAGE_MAX_PIXELS_CAP`), where upstream would allow up to
      `min(2048², 4096²/n)` (~3.36 MP at n=5). This is a cost knob, not a
      model limit: the 39.2 GB measurement above was taken **with** the cap in
      place, so raising it needs its own measurement rather than an inference
      from that number. The deviation is warned
      (`sensenova_reference_downscaled`) rather than silent.
  - **Refused, with a typed error, not warned**: spatial outpaint
    (`_reject_if_sensenova_unsupported`), deferred because it layers on the
    img2img/inpaint entry points rather than for lacking a mapping. VQA and
    `think_mode` have no route anywhere in this codebase, so their absence is
    a documentation fact, not a refusal; multi-turn conversational editing is
    deferred (re-uploading a result as a new reference covers iteration today).
  - **`negative_prompt` is a real, working feature, not just upstream's
    default**: `encode_prompt()`'s uncond branch is one call through the same
    `_build_t2i_query`/`_build_t2i_text_inputs`/`_t2i_prefix_forward` path as
    the cond branch — upstream always feeds it `""`, but there is nothing
    structurally special about that string. Substituting a caller-supplied
    `negative_prompt` was verified empirically (1024x1024/20-step A/B sweep +
    a 2048x2048/50-step production confirmation, same int8 checkpoint): a
    strong, orthogonal negative concept ("photorealistic, photograph,
    realistic, photography" against an unstyled photo prompt) produces a
    visible, `cfg_scale`-dose-scaling suppression effect (output shifts from
    photographic toward illustrative rendering) at `cfg_scale` 4 and 6, with
    contrast and midtones progressively flattening and visible posterization
    already present at 6, worsening into the classic CFG-too-high
    degradation (background detail loss, contrast burn, broken fine
    structure) at `cfg_scale` 8. A
    weak single-word negative that directly contradicts an explicit, specific
    positive descriptor (`negative_prompt="red"` against "a bright RED
    apple") showed little visible effect across a `cfg_scale` 2-6 sweep of
    its own — expected CFG
    behavior (a strong positive dominating a mild negative), not evidence the
    mechanism is broken. Default `cfg_scale` stays 4.0 for both the
    plain-uncond and real-negative-prompt cases (see
    `SENSENOVA_GENERATION_DEFAULTS` in `backend/api/param_defaults.py`) rather
    than switching the default conditionally on whether `negative_prompt` is
    set, which would reintroduce the same class of prefix/loop `cfg_scale`
    mismatch this integration already guards against elsewhere
    (`SenseNovaPrefix.encode_cfg_scale`, the `sensenova_cfg_mismatch`
    warning). `negative_prompt` has NO effect at `cfg_scale<=1` (no uncond
    branch is built at all there — the single-branch/8-step-distillation-LoRA
    operating point); this is warned, not silently dropped
    (`sensenova_negative_prompt_no_cfg`, in `encode_prompt()`), and is a
    function of `cfg_scale` alone, not of whether a LoRA happens to be
    loaded — a LoRA run at `cfg_scale>1` still gets a real uncond branch and
    a working negative prompt.
  - **Measured, RTX 6000 Ada 48 GB, int8, SDPA**: 2048x2048, 50 steps, CFG
    (two prefix-KV-cache branches) — 5.488 s/step, 275.1 s wall, prefill
    0.74 s, peak 24.51 GiB. 2048x2048, 8 steps, `cfg_scale=1.0`, distillation
    LoRA (294/294 modules applied) — 2.675 s/step, 22.0 s wall, prefill
    0.61 s, peak 24.58 GiB (per-step cost roughly halves because the single
    branch drops the uncond forward). Peak VRAM scales roughly linearly with
    pixel count; 3008x3008 (9.05 MP) at 43.68 GiB is the largest arm that
    completed on the 48 GB card, and 3162x3162 (10.0 MP) failed after the
    checkpoint was resident with no traceback or OOM message — exhaustion
    inferred from the VRAM trend, not read from an error. int8 vs bf16 at a
    fixed seed was visually indistinguishable (same composition, framing,
    lighting; differences confined to fine detail), so plain int8 stands and
    ConvRot was not needed.
  - **Training is a deliberately separate future phase**, not in
    `ARCH_REGISTRY`: the base checkpoint is converted UNMERGED from the
    8-step distillation LoRA specifically to keep the trainable lineage
    canonical.
  - **`sensenova_mot_phase_eviction`** (API boolean, default **off**,
    `SENSENOVA_GENERATION_DEFAULTS`): MoT phase-exclusive half-weight
    eviction. Each of the 42 layers carries two halves — "understanding"
    (plain names) and "generation" (`_mot_gen`), exactly 50/50 at
    386,221,056 bytes/layer (15.11 GiB total, 7.55 GiB/half) — and each
    generation phase uses exactly one: the prefix phase (KV-cache build)
    takes `forward_und`, the denoise phase (Euler loop) takes `forward_gen`.
    When enabled, the idle half is staged to pinned CPU per phase, driven by
    the previously no-op `_notify_layer_offload_phase` hook: three
    half-transfers per generation (~22.6 GiB PCIe) — generation half D2H at
    the start of prefix; understanding half D2H then generation half H2D at
    the start of denoise, in that order (**the eviction must precede the
    load at the denoise transition** — reversing it co-resides both halves
    on GPU for one window and the saving vanishes). Implementation:
    `backend/core/models/sensenova/mot_phase_eviction.py`, wired in
    `backend/core/pipeline_backends/sensenova.py`.
    - **Selection trap**: "owns no `nn.Parameter`" is not a safe
      "not a weight" test for this split — `Int8Linear` (588 of the 588
      quantized Linears) registers weight/weight_scale/bias as BUFFERS and
      owns zero Parameters, so a buffer-only rule silently selected only
      RMSNorm (~0.21 GiB) and made the feature inert with no code-level
      signal; it shipped that way twice before a full GPU measurement gate
      caught it. The real discriminator is persistence (a rotary
      embedding's `inv_freq` is a non-persistent buffer, int8 weight
      buffers are persistent), and a sanity check now warns
      (`sensenova_mot_phase_eviction_selection_suspect`) if either half
      comes in under 1 GiB or the halves differ by more than 2x.
    - **Measured, RTX 6000 Ada 48 GB, fixed seed 424242, 50 steps, cfg 4.0**:
      9.05 MP txt2img (3008², no refs) — off 43.82 GB / 1226.7 s (cold-start
      first arm after a restart, not a clean wall-clock baseline), on
      36.19 GB / 1045.6 s, on (repeat) 36.23 GB / 1046.9 s. 5-ref 2048² it2i
      — off 30.96 GB / 563.4 s, on 23.41 GB / 564.1 s. VRAM: **-7.63 GB**
      (9.05 MP) and **-7.55 GB** (5-ref it2i). Wall-clock: **+0.12%** on the
      it2i pair, the clean comparison (the txt2img pair's apparent speedup
      is a cold-start artifact of run order, not a measurement of the
      feature). Output bit-identical at fixed seed on both workloads (max
      abs channel diff 0, 0.000000% pixels differing).
    - **Host RAM cost is the reason the default is off**: pinned CPU
      tensors are never explicitly un-pinned after a generation — torch's
      caching host allocator pools freed pinned blocks rather than
      returning them to the OS, and an explicit unpin only adds a pageable
      clone on top of the still-reserved pool (measured: a net increase,
      not a release). Measured: RSS rises **~21.7 GiB** once eviction first
      engages (15.11 GiB pinned — 7.55 live + 7.55 pooled — plus pageable
      staging), flat across repeats (no leak), but **never returned to the
      OS**: it persists for the process lifetime, including into later
      generations run with the toggle off. The measured peaks already fit a
      48 GB card without this feature (worst case 43.82 GB); it is for
      operators who are VRAM-constrained and have host RAM to spare.
    - **Generic rolling block-swap was deliberately not built for this
      arch** (`TransformerBlockOffloader`, the mechanism 5 other
      architectures use): its transformer is never registered with it
      (`core.memory_management.transformer_registry` detects it as
      "unknown"), and rewriting the 3-branch denoise loop's
      layer-outer/branch-inner ordering to support it would cost 2-3x more
      PCIe traffic than this phase-exclusive scheme, while activations and
      KV-cache dominate the peak regardless — the marginal ceiling did not
      justify it. `blocks_to_swap` warns rather than being silently inert
      here (also unsupported, same reason, on sd15/sdxl/krea2/
      minimax_music3's generation path; see `arch_capabilities.py`).
      **Reopening condition**: a concrete workload whose measured peak
      exceeds ~44 GB after this feature is active, or an explicit
      requirement to run on a <48 GB card.
  - **`sensenova_kv_cache_streaming`** (API boolean, default **off**,
    `SENSENOVA_GENERATION_DEFAULTS`): collapses the persistent per-layer
    flash-KV prefix cache into a **2-slot GPU ring shared across every layer
    and branch**, streaming each slot's prefix head from a pinned CPU master
    with 1-layer lookahead on its own dedicated CUDA stream and pinned pool
    (deliberately not shared with `sensenova_mot_phase_eviction`'s — that
    feature's phase-boundary GB-scale transfer would head-of-line-block a
    per-layer prefetch). `prepare_flash_kv_cache` normally allocates one
    buffer per layer per branch, shape `(B, prefix_len + current_len, H, D)`;
    collapsing to 2 slots is valid because the denoise loop is
    branch-outer/layer-inner (`_predict_v_branch` is a full 42-layer pass per
    branch), so only one `(branch, layer)` buffer is ever live. `adopt()`
    deliberately bypasses `prepare_flash_kv_cache` entirely and builds the
    pinned master directly from each layer's existing `keys`/`values`, one
    layer at a time — building the full buffer set first and then freeing it
    would hit the very peak this feature exists to remove. Implementation:
    `backend/core/models/sensenova/kv_cache_streaming.py`, wired in
    `backend/core/pipeline_backends/sensenova.py` (3 install + 3 teardown
    sites), with the per-layer hookup in
    `backend/core/models/sensenova/vendor/modeling_qwen3.py` and
    adopt/begin_branch/teardown calls in
    `backend/core/models/sensenova/sensenova_pipeline_ops.py`.
    - **Design trap**: the flash KV buffer is ONE tensor with TWO regions of
      OPPOSITE lifecycle. The head `[:prefix_len]` is write-once/read-many
      (the prompt/reference prefix — immutable, the only part worth
      streaming). The tail `[prefix_len:]` is REWRITTEN IN PLACE by every
      layer on every denoise step (`vendor/modeling_qwen3.py:722-723`, the
      `update_cache=False` live path) immediately before that same layer
      reads it — it is per-step scratch, not a cache, and streaming it back
      would feed a stale tail and change outputs. The saving comes from
      collapsing the buffer COUNT, not from offloading the tail; slot
      reassignment between denoise steps is numerically inert precisely
      because of this write-then-read ordering, which the change leaves
      untouched — that is the load-bearing safety property.
    - **Trap found by audit before the feature ever ran**:
      `layer.is_initialized` is a bool SEPARATE from `keys`/`values` being
      `None`. transformers 5.1.0's `DynamicLayer.get_seq_length()` is
      `if not self.is_initialized or self.keys.numel() == 0: return 0`, so
      nulling the adopted GPU tensors without also clearing that flag makes
      it dereference `None.numel()`. Every denoise step reaches this path,
      because `_t2i_predict_v` calls the LM without `cache_position`
      (`vendor/modeling_qwen3.py:1198-1199` and the identical code in
      `modeling_qwen3_moe.py:431-432`). The fix sets `is_initialized = False`;
      the resulting `prefix_len -> 0` divergence on that call is inert on
      this path because `forward_gen`'s RoPE keys off `indexes`
      (`cache_position` appears only in its signature, never its body) and
      the flash branch's `attention_mask` is always the dict
      `{"full_attention": None}`, which skips mask construction entirely
      (`modeling_qwen3.py:1208`, takes the `else` at `:1231`).
    - **Measured, RTX 6000 Ada 48 GB, fixed seed 424242, `cfg_scale=4.0`,
      10 steps, backend restarted before the run** (peak VRAM is
      allocation-driven and reproduces `sensenova_mot_phase_eviction`'s
      recorded 3008² baseline of 43.82 GB almost exactly at 43.808 GB, so
      the peaks below are comparable across step counts even though
      wall-clock is not): 3008² t2i, 2 branches — off 43.808 GB, on
      40.882 GB (**-2.926 GB**). 2048² t2i, 2 branches — off 24.614 GB, on
      23.246 GB (**-1.368 GB**). 3-ref 2048² it2i, `img_cfg_scale=1.5`, 3
      branches — off 30.784 GB, on 26.803 GB (**-3.981 GB**). Same it2i
      workload with BOTH toggles (this feature + MoT eviction) on — 19.249 GB
      against the 30.784 GB plain baseline (**-11.535 GB**). Output is
      **bit-identical at fixed seed on every pair** (max abs channel diff 0,
      0.000000% pixels differing), including the both-toggles-on cell against
      the plain baseline — independently re-verified from the saved PNGs, not
      only self-reported. **Additivity is near-exact**: the measured combined
      -11.535 GB matches -11.531 GB predicted from MoT eviction's documented
      -7.55 GB plus this feature's own -3.981 GB on the same workload, with
      both features' `_active` warnings present.
    - **Staged pinned-CPU bytes scale with reference count (prefix size), NOT
      with output resolution** — 45.1 MiB for t2i; 1058.7 MiB across 3
      branches (~353 MiB/branch) for 3-reference it2i — whereas the VRAM
      saving scales with resolution via the `(res/32)²` tail. This asymmetry
      is the single most useful predictive fact for an operator deciding
      whether to enable the toggle: it helps most at high resolution and
      costs more host RAM with more references.
    - **Wall-clock is within noise on most cells** (it2i -1.01%, both-on
      +1.28%, 2048² +2.94%). The 3008² `cfg_scale>1` pair reproducibly ran
      ~10% FASTER with streaming on (off: 265.1 s cold, 233.5 s / 240.0 s
      warm; on: 209.8 s / 211.7 s). Since output is bit-identical the
      computation is provably identical, so this is a memory-system effect,
      not skipped work — the plausible mechanism is relief of allocator
      pressure near the card ceiling (43.8 GB on a 49.1 GB card), offered as
      unconfirmed, not as a benchmarked speedup claim.
    - **Host RAM cost, same class as `sensenova_mot_phase_eviction`'s**:
      RSS rose **~1.42 GiB** on first engagement — far more than the
      45.1 MiB staged for t2i — consistent with pinned-pool granularity (the
      same caching-host-allocator behavior already documented for MoT
      eviction). Flat across repeats and across a cancel/recovery cycle (no
      leak), but not returned to the OS; no release is claimed.
      Cancellation was explicitly tested: cancelled at step 12/40, clean
      error transition, backend stayed healthy, RSS flat, and the next
      streaming-on generation succeeded with no stale-state or
      slot-mismatch errors.
    - **Default is off for the same reason as `sensenova_mot_phase_eviction`**:
      the measured peaks already fit the 48 GB reference card, so an
      on-by-default would be an unmeasured default for the sub-48 GB
      population.
    - **Composes with `sensenova_mot_phase_eviction`**: disjoint tensors,
      hook points, cadences, and CUDA streams, with no shared coordinator.
      `enable_block_swap` remains unsupported/warned for this arch, same as
      above.
    - **No applicability to future SenseNova training**: a training step is
      a single-timestep forward/backward with no multi-step denoise loop, so
      there is no persistent read-many KV cache to stream; training-side
      offload belongs to `LayerOffloadConductor`. What DOES transfer is the
      MoT half-eviction CONCEPT: a gen-branch-only fine-tune with the
      understanding branch frozen could CPU-evict that half during training.
      Flag it as a thing to evaluate when training is built, not as planned
      work.
- **minimax_music3** — Full implementation account, including every measured
  number and the audit history behind them, is
  `docs/guides/MINIMAX_MUSIC3_DESIGN.md`; this entry states only the facts an
  arch-maintainer needs at a glance.
  - **Refusals, and why each is a model property, not an unimplemented
    feature** (`arch_capabilities.py`): `negative_prompt` (the flow-stage
    unconditional branch conditions on zeros; there is nothing to negate
    against); `audio_reference_conditioning` — no voice/timbre/instrument
    reference audio can condition generation, because the RVQ tokenizer's
    *encoder* (turns audio into semantic codes) is not published, so no audio
    can be turned into the AR stage's own input alphabet, and the DiT
    conditions on LM hidden states, not on audio directly; `controlnets`;
    `nag`; `advanced_cfg`; generation-time `lora` (the pipeline backend never
    reads `params["loras"]`); `unet_quantization` (checkpoint-format driven,
    like Ideogram 4/Krea 2, not a runtime toggle — see below). `vae_override`
    is refused: the "VAE" here is a decode-only vocoder with no swappable
    counterpart in this repo's override mechanism.
  - **Modality surfaces**: `POST /generate/txt2aud` (text-to-music, real
    audio); `POST /generate/outpaint/audio` with `extend_forward` only
    (forward continuation by AR-resume from the frame-code sidecar; backward
    extension is not possible — the LM is causal); `POST /generate/aud2aud`
    with `mode="repaint"`, two sub-modes (`regenerate`: AR-resume with a new
    tail, content changes, everything before the cut point preserved exactly;
    `rerender`: redraw the flow stage over a kept-codes window with a new
    seed, timbre/mix change, lyrics/melody/timing do not). Mid-span infill
    with a preserved tail is refused (causal LM, no infilling contract) —
    same style as MiniMax-H3's placement enumeration above.
  - **Per-generation state contract**: every `txt2aud` generation writes a
    sidecar (frame codes, sample/frame rates, prompt, lyrics, seed) next to the
    audio file. Extend and repaint both require it — a song generated by an
    older build without one cannot be extended or repainted. Storing the
    sidecar's frame codes rather than the AR stage's `frame_hiddens` is a
    ~4,000x size reduction (144 KB vs. ~590 MB for a six-minute song, bf16) —
    the hidden states are exactly recoverable from the codes by a
    teacher-forced replay.
  - **A seed is not portable across text-encoder sources.** The full-vocabulary
    and pruned-vocabulary text encoders sample DIFFERENT frame codes from the
    same seed even when fed bit-identical restricted logits, because
    `torch.multinomial`'s RNG consumption depends on the sampled category
    count (200,000-wide vs. 16,385-wide). Songs stay reproducible by their
    stored frame codes regardless of which text encoder generated them; a seed
    alone does not reproduce a song across a `text_encoder_file` change.
  - **Quantization residency, per format, measured** (see the design doc for
    the full methodology and audit trail): Q8_0 (`GGUFQ8_0Linear`) keeps a
    dense mirror co-resident once a layer is touched — VRAM saving during the
    AR stage is **zero by construction**, because the language model and depth
    decoder must be dense to compute at the AR loop's per-frame call rate; the
    saving is host RAM / disk only, **~42.7%** (header-only tensor-byte
    arithmetic: 9.589 GB vs. 16.707 GB for the same 328 tensors), not the
    ~49% a since-corrected process-RSS measurement first reported. INT8
    ConvRot (`ConvRotInt8Linear`, reused unchanged from MiniMax-H3) never
    materializes a dense mirror at all — measured on a real 4096×4096 layer,
    resident weight **50.05% smaller** (16.794 MB vs. 33.554 MB) and the
    forward-call peak-memory delta below a single dense weight's own size —
    but that number is layer-local; AR-stage KV cache and activation memory
    were not measured for either format. The flat "FP16" DiT is bit-exact
    under `official.bfloat16().half()`, not `official.half()` directly — it
    carries bf16 precision under an FP16 label, so it gains nothing over
    loading `official/` at this loader's bf16 default. The "BF16" GGUF DiT is,
    at that same bf16 default, the **worse** of the two flat sources for ~40%
    of its tensors (up to 2⁻⁸ extra rounding from a double cast), not a wash.
  - **`MiniMax Music 3` is not in `RUNTIME_INT8_ARCHS` / `QUANTIZED_LINEAR_ARCHS`
    (`core.models.common.int8_runtime_quantize`)**, deliberately: those tables
    advertise a runtime `unet_quantization`/`quantized_gemm_mode` toggle, and
    this architecture's quantized-Linear builders (Q8_0, ConvRot) are all
    reached only by a load-time checkpoint-format choice, with no live
    runtime-conversion path behind them.
  - **The component-switch catalog** (`component_catalog.py`,
    `POST /models/current/components/switch`) was **not** extended for this
    architecture — it needs a per-architecture unload-first adapter this
    architecture does not have. The generic `{slot}_origin` reporting still
    works without one (`selected_external` for a `text_encoder_file`
    override), so component selection works at load time via
    `POST /models/load`'s `text_encoder_file` but not as a post-load hot-swap.
  - **`keep_models_hot` is not wired for this architecture** — see the
    "VRAM management" section above.
  - **Frontend**: txt2aud and extend (`/generate/outpaint/audio`) UI shipped.
    **Repaint's UI branch is not implemented** — blocked on a shared-worktree
    conflict in `Img2ImgPanel.tsx` at the time this phase landed. The backend
    and API (`POST /generate/aud2aud` with `mode="repaint"`) are complete and
    reachable by direct API calls; there is no UI path to reach them today.
    The `text_encoder_file` selection has no frontend surface either, for the
    same reason — it is reachable today only via `POST /models/load`.

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
- `backend/core/models/minimax_h3/loader.py` — MiniMax-H3 component loading:
  header-driven config synthesis, the two QKV conventions, SwiGLU swap, audio
  weight-norm fold, TE prefix rewrite, the pinned VAE tiling policy and the
  fp8 dequant-only quantization policy. `h3_pipeline_ops.py` is the repo-owned
  denoise loop, both packed-layout builders (`t2va`/`fl2va` and `ref2va`), the
  layer-streamed `encode_prompt` (placeholder scatter + deepstack + mrope) and
  both decodes; `core/models/minimax_h3/vendor/` holds the frozen model classes
  (transformer with the ported AdaLN curve, both VAEs, the scheduler).
- `backend/core/models/minimax_h3/h3_references.py` — the `ref2va` media side:
  the released checkpoint's reference limits, the canvas and 2048-short-edge
  rules, the 24 fps / 32 kHz normalisation, the VAE-grid refusal for short
  reference videos, and the labelled `<Picture i>` / `<Audio j>` / `<Video k>`
  presentation the conditioner reads.
- `backend/core/models/minimax_h3_block_loop_wrapper.py` — block-loop
  re-ownership for block swap, guarded FBCache and gradient checkpointing.
- `backend/core/models/components/wiring.py` — `TemporalSpec` /
  `TEMPORAL_SPECS`: the per-video-arch clip-length, frame-rate and canvas
  contract read by route validation, bucketing, the video loader, the clip-cache
  key and `GET /schema/arch-capabilities`.
- `backend/core/training/{arch,ops,adapters}/minimax_h3*.py` — LoRA training
  path (target set, joint video+audio objective, `audio_loss_weight`).
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
- `backend/core/models/minimax_music3/loader.py` — directory/file detection,
  the `official/`-tree load, flat/GGUF DiT and 4-way text-encoder-builder
  dispatch (`detect_minimax_music3_text_encoder_source`), the header-only
  quantization/pruned-vocabulary censuses. `vendor/` holds the ported
  diffusers-PR model classes; `flat_remap.py` / `pruned_text_encoder_remap.py`
  / `convrot_remap.py` are the checkpoint-key remaps; `vocab_view.py` resolves
  full- vs. pruned-vocabulary AR dispatch by the loaded `language_model`'s own
  shape. `backend/core/pipeline_backends/minimax_music3.py` is the AR + flow +
  vocode generation loop (txt2aud, extend, repaint). See
  `docs/guides/MINIMAX_MUSIC3_DESIGN.md` for the full account.
