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
| `docs/guides/GENERATION_QUEUE_PROCESSOR.md` | Read before changing anything about the generation queue's frontend dispatch: what the headless processor owns vs. the panels, which UI state is frozen onto a QueueItem at enqueue time and why, and the alert-ordering / model-hold / drift-pause traps. |
| `docs/guides/ADD_A_MODEL_ARCHITECTURE.md` | Procedure for adding another diffusion architecture (14 exist), including the extra surface a video/audio architecture needs. |
| `docs/guides/API_TESTING.md` | Safe vs. owner-sanctioned endpoints, dry-run convention, restart polling. |
| `docs/guides/CFG_UNCONDITIONAL_TRAINING.md` | Mathematical and implementation audit of CFG/unconditional training across all 14 generation architectures: shared-model caption dropout, architecture-specific null representations, retained conditions, distilled/no-CFG routes, separate-transformer Ideogram 4, and MiniMax Music 3's inference-only CFG boundary. |
| `docs/guides/MODEL_FACTS.md` | Per-architecture facts reference for all 14 architectures, including the video models' (LTX-2.3, MiniMax-H3) and MiniMax Music 3's measured performance and their accepted/refused feature set. |
| `docs/guides/INT8_CONVROT_TRAINING_DESIGN.md` | Design **plus its 2026-08-26 synthetic measurement pass** for mixed-precision INT8 ConvRot training (nothing shipped, no default changed): confirms current LoRA-over-ConvRot behavior and sizes its train/deploy skew (measured ~1.0e-2 relative forward difference on real ConvRot weights), proposes fused W8A8 forward plus floating `grad_input` for frozen weights only, and carries §3's measured per-shape/rollup/whole-step/VRAM numbers — per-layer forward 1.06×-6.54×, whole synthetic step −15.4% at 64 image tokens and +25.7% at 1024, peak 23.22 → 8.18 GiB with gradient checkpointing off against a 15.83 GiB bf16 equivalent — each labelled synthetic or real. Read §3.1's host-dispatch finding before reasoning about step time (arm B does 29% less GPU work at 64 tokens and is still 15% slower), §5 for the artifact invariant that is **not implemented**, and §4.2 for the unchanged refusal of trainable ConvRot weights. The verdict lives in gate **G5**, not here. |
| `docs/guides/MINIMAX_MUSIC3_DESIGN.md` | Implementation contract and status for the MiniMax Music 3 audio architecture: why its code is vendored from an unmerged diffusers PR, what the model can and cannot be conditioned on (no reference audio, no negative prompt), the frame-code state contract that extend/repaint depend on, the phase plan, and the weight-format (flat/GGUF/Q8_0/INT8 ConvRot) landing status. Repaint's frontend UI and the component-switch catalog are shipped-backend-only; see the doc's "Current status". |
| `docs/guides/SENSENOVA_TRAINING_DESIGN.md` | Design record and DONE/PENDING boundary for SenseNova U1.5 training: Phase 1 gen-branch-only LoRA over the int8 base is **implemented** (arch handler + ops + adapter + `ARCH_REGISTRY`, two-pass prefix/denoise step, opt-in MoT half-eviction with its OFF/ON measurement in §8.3), and **the generation-half full fine-tune is now ACCEPTED end to end** (U-2-2 step 3, 2026-08-25: both refusals removed, and proven by a real 3-step run on the real checkpoint — 294/294 update census, a 25.129 GiB `mixed` checkpoint saved and reloaded byte-identical through the production reader; §13.4's "U-2-2 実測"). Read that box for the per-run contract (bf16, batch 1, no accumulation, no EMA, stochastic rounding forced on and announced, and one of adafactor / either ring-buffer optimizer with `optimizer_state_host_resident`), for **why only `sensenova_full_finetune_save_format='int8'` can be resumed from**, and for the seven acceptance-path defects the run and its audit exposed (five fixed, two recorded). **§8.3.3 is the resolution campaign (2026-08-25)** — read it before quoting any residency number: it corrects the doc's own "no training step has ever exceeded the load high-water" (true only for the four-phase-ON `both` arm), records that every earlier 64px figure was taken at four image tokens, gives the measured step-vs-load separation and which resolutions each branch fits at, and closes the `int8` round trip + resume and generation-from-a-trained-checkpoint. What is still open: 2b-4 (offload composition), the §8.3 half-eviction gate, quality/convergence, `und` at 512/1024 and anything above 1024px. Reference-included datasets mixed with plain ones are **Phase 3** (done). Records why the understanding branch is frozen by default (its LoRA is opt-in via `train_text_encoder`, Phase U-1), why the recorded block-swap refusal does not transfer from generation to training, and what the repo does and does not actually know about quantized-base and bf16-rounding constraints. |
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
| `backend/core/training/INT8_W8A8_TRAINING_GATE.md` | Measurement gate **G3**: whether a gradient-capable INT8 W8A8 forward path may be built for training at all (the inference gates G1/G2 are in `examples/api/README.md`). **Closed: G3 FAILED** — criterion 2 (no tested workload regresses >3%) was violated at 256px/512px Anima token counts; see the file's "Result" section. Also holds gate **G4** (memory): whether the dequant path may stop handing its dequantized weight to autograd. **Closed: G4 FAILED** on its pre-registered step-time ceiling after passing the bitwise, gradient and memory criteria; records the `gradient_checkpointing: false` + quantized-base cost and the warning that shipped instead. Also holds gate **G5** (pre-registered 2026-08-26, **OPEN**): whether a fused ConvRot W8A8 forward with a floating `grad_input` may be built for a frozen ConvRot base — the rotated case G3 put out of scope. Decided by a **real** SenseNova training step (MiniMax-H3 too if a ConvRot/W4A8 base is reachable on its differentiable path), which nobody has taken; the 2026-08-26 ConvRot numbers are labelled inputs and priors, not the verdict. Inherits G3's >=10% bar and 3% regression floor unchanged, registers **no** token-count admission rule (and the conditions a successor gate would need to carry one), and records the release conditions that outrank speed: bf16 **and** fp16 gradient correctness, quality through the deployment path, and the not-yet-implemented base-function/artifact invariant. |
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
