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
| `docs/decisions/SENSENOVA_TEXT_OUTPUT_DESIGN.md` | Accepted design for SenseNova img2txt inference, caption/tag training, and mixed image/text objectives; implementation is pending. |

## Training and data

| Document | Purpose |
|---|---|
| `docs/reference/DATASETS.md` | Dataset formats and validation requirements. |
| `docs/guides/DYNAMIC_CROP_BUCKETING.md` | Epoch-dynamic crop and bucket behavior. |
| `docs/guides/SDXL_REGULARIZATION.md` | SD/SDXL regularization behavior. |
| `docs/guides/SENSENOVA_TRAINING_DESIGN.md` | SenseNova training contract and implemented/pending boundary. |
| `docs/decisions/TRAINING_RUN_STORAGE_V2.md` | Accepted staged migration from central detailed training history to a lightweight catalogue plus per-run databases. |
| `docs/guides/INT8_CONVROT_TRAINING_DESIGN.md` | INT8 ConvRot investigation and measurement status. |
| `docs/guides/TRAINING_DIAGNOSTICS_AND_AUXILIARY_LOSSES.md` | Shipped convergence diagnostics, crop-decode loss, and REPA target-source boundary. |
| `docs/guides/CONVERGENCE_PHASE1_SENSENOVA.md` | Phase 1 diagnostic measurements on SenseNova. |
| `docs/guides/LYCORIS_ADAPTER_DESIGN.md` | Current adapter subsystem contract: capability axes, algebra, topology, codecs, sessions, block swap, and execution backends. |
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
| `docs/audits/ACTIVATION_DISPATCH_ARCH_COVERAGE_REPORT_2026-09.md` | Static activation-dispatch coverage and architecture boundaries. |
| `docs/audits/ACTIVATION_DISPATCH_GPU_VALIDATION_SUMMARY_2026-09.md` | Scope and disposition of the completed GPU validation. |
| `docs/audits/MINIMAX_H3_ACTIVATION_DISPATCH_GPU_RESULT_2026-09.md` | MiniMax-H3 activation-dispatch measurements and limitations. |
| `docs/audits/UNIFIED_OFFLOAD_TRANSFER_VALIDATION_2026-09.md` | Completed common transfer-engine validation and measured claims. |
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
| `docs/audits/DOCUMENTATION_HYGIENE_2026-09.md` | Documentation publication-boundary audit and accepted cleanup plan. |
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
