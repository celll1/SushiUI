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
