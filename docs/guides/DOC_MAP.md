# Documentation map

This index lists the maintained, tracked documentation. See `docs/README.md`
for the tracking policy and the boundary between public documentation and
machine-local working material.

## Entry points

| Document | Purpose |
|---|---|
| `README.md` | Project overview and setup. |
| `AGENTS.md` | Repository rules and task router for coding agents. |
| `docs/README.md` | Documentation taxonomy, privacy rules, and review checklist. |
| `docs/guides/DOC_MAP.md` | This detailed index. |

## Architecture and development guides

| Document | Purpose |
|---|---|
| `docs/guides/ARCHITECTURE_MAP.md` | Backend/frontend ownership map. |
| `docs/reference/architectures/` | Per-architecture structure reference: components, load path, denoiser diagram, tensor contract, hook points. One file per architecture; `README.md` there is the index. |
| `docs/guides/REQUEST_LIFECYCLE.md` | Generation request flow from frontend to persistence. |
| `docs/guides/ADD_A_PARAMETER.md` | End-to-end parameter checklist. |
| `docs/guides/ADD_A_MODEL_ARCHITECTURE.md` | Architecture integration checklist. |
| `docs/guides/API_TESTING.md` | Safe API verification procedures. |
| `docs/guides/DATABASE_MIGRATION.md` | SQLite schema migration procedure. |
| `docs/guides/GENERATION_QUEUE_PROCESSOR.md` | Frontend queue ownership and dispatch invariants. |

## Generation behavior

| Document | Purpose |
|---|---|
| `docs/guides/MODEL_FACTS.md` | Per-architecture generation and training facts, including which adapter families (LoRA/LoHa/LoKr/DoRA) each architecture takes on each axis. |
| `docs/guides/CFG_UNCONDITIONAL_TRAINING.md` | CFG and unconditional-training audit across all generation architectures. |
| `docs/guides/FBCACHE.md` | FBCache acceptance rules and video safeguards. |
| `docs/guides/NAG.md` | Normalized Attention Guidance behavior and architecture hooks. |
| `docs/guides/SPECTRUM.md` | Spectrum forecasting behavior, parameters, and constraints. |
| `docs/guides/MINIMAX_H3_HYBRID_LOADER_DESIGN.md` | Current MiniMax-H3 hybrid-loader contract. |
| `docs/guides/MINIMAX_MUSIC3_DESIGN.md` | Current MiniMax Music 3 integration contract and implemented boundary. |

## Training and data

| Document | Purpose |
|---|---|
| `docs/reference/DATASETS.md` | Dataset formats and validation requirements. |
| `docs/guides/DYNAMIC_CROP_BUCKETING.md` | Epoch-dynamic crop and bucket behavior. |
| `docs/guides/SDXL_REGULARIZATION.md` | SD/SDXL regularization behavior. |
| `docs/guides/SENSENOVA_TRAINING_DESIGN.md` | SenseNova training contract and implemented/pending boundary. |
| `docs/guides/INT8_CONVROT_TRAINING_DESIGN.md` | INT8 ConvRot investigation and measurement status. |
| `docs/guides/LYCORIS_ADAPTER_DESIGN.md` | **The adapter subsystem's durable note.** LyCORIS 4.0.0 assessment, the shared LoRA/LoHa/LoKr/DoRA engine at `backend/core/adapters/` (spec, target topology, tensor grouping, codec registry, `AdapterSession`, execution-backend registry), the two capability axes, and the shipped boundary. Read it before touching anything adapter-, LoRA-variant- or `adapter_type`-related; per-architecture family enablement is summarised in `docs/guides/MODEL_FACTS.md`. |
| `docs/guides/VAE_TRAINING.md` | VAE decoder training contract. |
| `docs/guides/VAE_DECODE_BEHAVIOR.md` | VAE tiled-decode behavior and measured non-locality. |
| `docs/guides/VAE_TRAINING_RESOLUTION.md` | VAE crop/resolution semantics and scaling measurements. |
| `backend/core/training/API_REFERENCE.md` | Training API reference. |
| `backend/core/training/TRAINING_PARAMS_GUIDE.md` | Training parameters and configuration. |
| `backend/core/training/MODEL_ARCHITECTURES.md` | Per-architecture training internals. |
| `backend/core/training/adapters/MODEL_ADAPTER_DESIGN.md` | Model-adapter pattern. |
| `backend/core/training/INT8_W8A8_TRAINING_GATE.md` | Registered quantized-training gates and results. |

## Memory and attention

| Document | Purpose |
|---|---|
| `backend/core/docs/ATTENTION_PROCESSORS.md` | Attention backend selection. |
| `backend/core/memory_management/README.md` | Memory-management subsystem overview. |
| `backend/core/memory_management/BLOCK_SWAP.md` | Current block-swap behavior. |
| `backend/core/memory_management/RING_BUFFER_OPTIMIZER.md` | Optimizer-state residency mechanism. |
| `backend/core/training/optimizers/RINGBUFFER_OPTIMIZERS.md` | Ring-buffer optimizer contracts. |
| `docs/audits/BLOCK_SWAP.md` | Completed block-swap implementation audit. |
| `docs/audits/SENSENOVA_UND_BRANCH_DISCRIMINATION.md` | Measured audit: the SenseNova understanding branch keeps detail-differing prompts apart. |

## API, tools, and subapps

| Document | Purpose |
|---|---|
| `backend/api/WS_PROTOCOL.md` | WebSocket progress protocol. |
| `backend/backups/README.md` | Backup directory conventions. |
| `examples/api/README.md` | API example scripts and measurement gates. |
| `subapps/aesthetic_scorer/README.md` | Aesthetic scorer overview. |
| `subapps/aesthetic_scorer/USAGE.md` | Aesthetic scorer usage. |
| `subapps/fp8_quantize/README.md` | FP8 checkpoint quantizer. |
| `subapps/layer_pruning/README.md` | Layer-pruning utility. |

## Legal and agent support

| Document | Purpose |
|---|---|
| `docs/legal/THIRD_PARTY_PROVENANCE.md` | Vendored/adapted source ledger and redistribution gate. |
| `.claude/agents/arch-maintainer.md` | Architecture-maintenance subagent definition. |
| `.claude/agents/api-tester.md` | API-test subagent definition. |
| `.claude/agents/code-auditor.md` | Code-audit subagent definition. |
| `.claude/agents/docs-maintainer.md` | Documentation-maintenance subagent definition. |
| `.claude/agents/feature-worker.md` | Feature implementation subagent definition. |
| `.claude/agents/orchestrator.md` | Multi-workstream orchestration definition. |
| `.claude/agents/research-integrator.md` | Research assessment and clean-room boundary definition. |
| `local/README.md` | Rules for the ignored machine-local working area. |

Local strategies, research notes, raw measurements, drafts, and historical work
logs are intentionally not enumerated here: they may be absent in a fresh clone
and must not be referenced as current contracts.
