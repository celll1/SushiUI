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

## Anchors used

- `backend/core/model_loader.py` — detection, prediction config, zimage/flux2
  single-file loaders, comfy→official conversion.
- `backend/core/attention/registry.py` — per-arch attention backend routing
  (conduit vs diffusers dispatch, head_dim constraints, tq/flash/sage/native).
- `backend/core/pipeline_backends/{flux2,ideogram4,lens,minit2i,krea2,zimage,anima}.py`
  — CFG conventions, text encoding, VAE staging.
- `backend/core/inference/custom_sampling.py` — SD/SDXL CFG short-circuit at cfg==1.0.
- `backend/core/models/{lens,ideogram4,minit2i,krea2,anima}/*_loader.py` — component
  classes, completion sources (sibling dirs / hub fallbacks / env overrides).
- `backend/core/training/adapters/{sd15,sdxl,zimage,flux2,ideogram4,lens,minit2i,anima,krea2}_adapter.py`
  — TE-frozen policies, dual-transformer training, LLM-Adapter-only mode.
- `backend/core/training/MODEL_ARCHITECTURES.md` — SD1.5/SDXL/Z-Image component
  specs, forward-pass signatures, schedulers.
