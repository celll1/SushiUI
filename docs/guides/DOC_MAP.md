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
| `docs/guides/REQUEST_LIFECYCLE.md` | End-to-end generation request flow (frontend to DB to gallery). |
| `docs/guides/ADD_A_PARAMETER.md` | Ordered checklist for threading a new API parameter through the stack. |
| `docs/guides/ADD_A_MODEL_ARCHITECTURE.md` | Procedure for adding a 10th diffusion architecture. |
| `docs/guides/API_TESTING.md` | Safe vs. owner-sanctioned endpoints, dry-run convention, restart polling. |
| `docs/guides/MODEL_FACTS.md` | Per-architecture facts reference, including LTX-2.3 video-model speed/lightweight features. |
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
| `backend/core/training/MODEL_ARCHITECTURES.md` | Per-architecture training internals (text encoding, conditioning, etc.). |
| `backend/core/training/TRAINING_PARAMS_GUIDE.md` | Training parameter guide. |
| `backend/core/training/adapters/MODEL_ADAPTER_DESIGN.md` | Training model-adapter design pattern. |
| `backend/core/training/optimizers/RINGBUFFER_OPTIMIZERS.md` | Ring-buffer optimizer implementations. |

## Frontend / tooling / examples

| Doc | What it covers |
|---|---|
| `examples/api/README.md` | How to run the frontend-less API example scripts. |
| `frontend/src/components/training/SINGLE_STATE_MIGRATION_PLAN.md` | Frontend training-state migration plan. |
| `scripts/README.md` | Utility scripts overview. |
| `subapps/aesthetic_scorer/README.md` | Aesthetic scorer subapp overview. |
| `subapps/aesthetic_scorer/USAGE.md` | Aesthetic scorer usage. |
| `subapps/layer_pruning/README.md` | Layer-pruning subapp overview. |
| `tests/README.md` | Test suite overview. |

## Local-only (gitignored, may be absent in a clone)

| Doc | What it covers |
|---|---|
| `CLAUDE.md` | Repo owner's personal, Japanese-language conventions log (paths, rationale, history). |
| `tq-attention/*` | Working notes for the TQ (quantized) attention backend project. |
