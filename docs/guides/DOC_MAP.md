# Documentation Map

Every Markdown file tracked in git (`git ls-files '*.md'`), one line each.
Local-only files that may be absent in a fresh clone are listed separately.

## Root / entry points

| Doc | What it covers |
|---|---|
| `README.md` | Project overview, setup, feature list. |
| `AGENTS.md` | Router for coding agents: rules digest + task-to-doc table. |
| `IMPLEMENTATION_PLAN_ORIGINAL_ARCHITECTURE.md` | Historical original-architecture implementation plan. |

## `docs/guides/` (this set)

| Doc | What it covers |
|---|---|
| `docs/guides/DOC_MAP.md` | This file. |
| `docs/guides/ARCHITECTURE_MAP.md` | Directory tree + backend/frontend module responsibilities. |
| `docs/guides/REQUEST_LIFECYCLE.md` | End-to-end request flow (frontend to DB to gallery) for image generation, video generation and model load. |
| `docs/guides/ADD_A_PARAMETER.md` | Ordered checklist for threading a new API parameter through the stack. |
| `docs/guides/ADD_A_MODEL_ARCHITECTURE.md` | Procedure for adding another diffusion architecture (14 exist), including the extra surface a video/audio architecture needs. |
| `docs/guides/API_TESTING.md` | Safe vs. owner-sanctioned endpoints, dry-run convention, restart polling. |
| `docs/guides/MODEL_FACTS.md` | Per-architecture facts reference for all 14 architectures, including the video models' (LTX-2.3, MiniMax-H3) and MiniMax Music 3's measured performance and their accepted/refused feature set. |
| `docs/guides/MINIMAX_MUSIC3_DESIGN.md` | Implementation contract and status for the MiniMax Music 3 audio architecture: why its code is vendored from an unmerged diffusers PR, what the model can and cannot be conditioned on (no reference audio, no negative prompt), the frame-code state contract that extend/repaint depend on, the phase plan, and the weight-format (flat/GGUF/Q8_0/INT8 ConvRot) landing status. Repaint's frontend UI and the component-switch catalog are shipped-backend-only; see the doc's "Current status". |
| `docs/guides/SENSENOVA_TRAINING_DESIGN.md` | Design-only roadmap (no code yet) for making SenseNova U1.5 trainable: gen-branch-only LoRA over the int8 base, a guard-first full fine-tune whose real implementation needs a bf16 gen-branch base, and reference-included datasets mixed with plain ones. Records why the understanding branch stays frozen, why the recorded block-swap refusal does not transfer from generation to training, and what the repo does and does not actually know about quantized-base and bf16-rounding constraints. |
| `docs/guides/VAE_TRAINING.md` | Read when running/changing a `vae_decoder` (decoder-only, encoder-frozen) fine-tune, or before touching its loss defaults or refusal gate. |
| `docs/guides/VAE_DECODE_BEHAVIOR.md` | Read before working on tiled decode, seams, or any claim about what a VAE decoder adds/loses — the measured non-locality decomposition and artifact facts live here. |
| `docs/guides/VAE_TRAINING_RESOLUTION.md` | Read to understand what a VAE fine-tune's crop policy and `resolution` feed the decoder, why bucketing is skipped, and how its memory/time scale — the measured analysis (gradient checkpointing vs ActDispatch vs tiling), which is knowledge rather than a set of `vae_decoder` config keys. The knobs that do exist are in `VAE_TRAINING.md`. |

## `docs/` (design docs / reports / plans)

| Doc | What it covers |
|---|---|
| `docs/API_IMPROVEMENT_PROPOSALS.md` | Proposed API cleanups. |
| `docs/BLOCK_SWAP_AUDIT.md` | Audit of block-swap CPU offload behavior. |
| `docs/CFG_ANALYSIS_ZIMAGE_VS_SDXL.md` | CFG behavior comparison across architectures. |
| `docs/DATABASE_MIGRATION_GUIDE.md` | SQLite schema migration procedure. |
| `docs/DATASET_REQUIREMENTS.md` | Dataset format/requirements for training. |
| `docs/EPOCH_DYNAMIC_CROP_BUCKETING_DESIGN.md` | Dynamic crop/bucketing design for training. |
| `docs/FBCACHE_DESIGN.md` | FBCache quality acceptance and MiniMax-H3 temporal/consecutive-hit safeguards. |
| `docs/FLUX2_NAG_BLOCKSWAP_UNIFY_DESIGN.md` | Unifying NAG + block-swap for Flux2. |
| `docs/H2D_ONLY_TRAINING_DESIGN.md` | Host-to-device-only training memory design. |
| `docs/NAG_DIT_PLAN.md` | NAG (negative-prompt attention guidance) plan for DiT models. |
| `docs/PIPELINE_REFACTOR_PLAN.md` | Pipeline refactor plan. |
| `docs/SDXL_REGULARIZATION_IMPLEMENTATION.md` | SDXL training regularization implementation. |
| `docs/SPECTRUM_DESIGN.md` | Spectrum (frequency-domain guidance) design. |
| `docs/SPECTRUM_DIT_PLAN.md` | Spectrum feature plan for DiT models. |
| `docs/TRAINING_CONFIG_ISSUES.md` | Known training config issues. |
| `docs/TRAINING_FEATURES_PARITY_REPORT.md` | Training feature parity across architectures. |
| `docs/TRAINING_REQUIREMENTS.md` | Training hardware/software requirements. |
| `docs/VRAM_OVERFLOW_PREVENTION_DESIGN.md` | VRAM overflow prevention design. |
| `docs/ZIMAGE_BATCH_VRAM_ANALYSIS.md` | Z-Image batch VRAM analysis. |
| `docs/ZIMAGE_TRAINING_IMPLEMENTATION_PLAN.md` | Z-Image training implementation plan. |

## Backend-adjacent docs

| Doc | What it covers |
|---|---|
| `backend/api/WS_PROTOCOL.md` | WebSocket message types/fields for progress streaming. |
| `backend/backups/README.md` | Backup directory conventions. |
| `backend/core/docs/ATTENTION_PROCESSORS.md` | Attention backend selection and processor details. |
| `backend/core/memory_management/BLOCK_SWAP.md` | Block-swap CPU offload mechanism for training. |
| `backend/core/memory_management/README.md` | Memory management module overview. |
| `backend/core/memory_management/RING_BUFFER_OPTIMIZER.md` | Ring-buffer optimizer state management. |
| `backend/core/training/API_REFERENCE.md` | Training API reference. |
| `backend/core/training/INT8_W8A8_TRAINING_GATE.md` | Measurement gate **G3**: whether a gradient-capable INT8 W8A8 forward path may be built for training at all (the inference gates G1/G2 are in `examples/api/README.md`). **Closed: G3 FAILED** — criterion 2 (no tested workload regresses >3%) was violated at 256px/512px Anima token counts; see the file's "Result" section. Also holds gate **G4** (memory): whether the dequant path may stop handing its dequantized weight to autograd. **Closed: G4 FAILED** on its pre-registered step-time ceiling after passing the bitwise, gradient and memory criteria; records the `gradient_checkpointing: false` + quantized-base cost and the warning that shipped instead. |
| `backend/core/training/MODEL_ARCHITECTURES.md` | Per-architecture training internals (text encoding, conditioning, etc.). |
| `backend/core/training/TRAINING_PARAMS_GUIDE.md` | Training parameter guide. |
| `backend/core/training/adapters/MODEL_ADAPTER_DESIGN.md` | Training model-adapter design pattern. |
| `backend/core/training/optimizers/RINGBUFFER_OPTIMIZERS.md` | Ring-buffer optimizer implementations. |

## Frontend / tooling / examples

| Doc | What it covers |
|---|---|
| `examples/api/README.md` | How to run the frontend-less API example scripts; FP8 scaled-GEMM gate (G1) decision rule. |
| `frontend/src/components/training/SINGLE_STATE_MIGRATION_PLAN.md` | Frontend training-state migration plan. |
| `local/README.md` | The machine-local working area (gitignored contents), and where the tracked equivalents live. |
| `subapps/aesthetic_scorer/README.md` | Aesthetic scorer subapp overview. |
| `subapps/fp8_quantize/README.md` | Weight-only FP8 checkpoint quantization tool (format, layer selection, sibling links). |
| `subapps/aesthetic_scorer/USAGE.md` | Aesthetic scorer usage. |
| `subapps/layer_pruning/README.md` | Layer-pruning subapp overview. |

## Local-only (gitignored, may be absent in a clone)

| Doc | What it covers |
|---|---|
| `CLAUDE.md` | Repo owner's personal, Japanese-language conventions log (paths, rationale, history). |
| `tq-attention/*` | Working notes for the TQ (quantized) attention backend project. |
