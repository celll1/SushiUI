# Hybrid U-Net ssDiT plan audit

Reviewed 2026-10-10 against `local/strategy/SDXL_LIKE_UNET_SSDIT_PLAN.md`.
This is a completion audit, not a declaration of completion. The implementation
and compute matrix are substantial; P5 quality acceptance remains incomplete.
Existing owner runs and their artifacts are not migrated implicitly.

## Implemented contracts and evidence

| Requirement | Current evidence | Assessment |
|---|---|---|
| Fixed native width; learned down/up, local residual processing, multiple compressed skips | `core/rewiring/hybrid_spec.py`, `hybrid_modules.py`; `hybrid_rewiring_test.py`, native Anima/Ming/Qwen tests | Implemented; isotropic 2x boundaries, native widths retained |
| Zero output branches preserve the initial residual baseline; internal branches receive gradients after training | `hybrid_anima_test.py`, `hybrid_qwen_test.py`, `hybrid_ming_test.py`; Anima lifecycle records | Verified on CPU and bounded Anima GPU training |
| Text/prefix tokens and Layer frame axes remain separate from image-grid processing | Native residual owners and `hybrid_ming_test.py` | CPU native contract verified; real Ming Layer initializer/generation verified |
| Stable module IDs, exact topology/module/native identities, strict complete payload loading | `spec.py`, `hybrid_spec.py`, native loaders; `hybrid_provenance_test.py`, export tests | Implemented; missing/non-finite/inconsistent payloads refused |
| Full FT save/resume and named optimizer ownership | `hybrid_anima_lifecycle_2026-10.json`, `hybrid_anima_initialized_full_ft_2026-10.json` | Real Anima GPU continuation verified, including structural parameters |
| LoRA complete structural payload and ordinary adapter algebra | `hybrid_adapter_training_2026-10.json`, `hybrid_adapter_generation_payload_2026-10.json` | All three native integrations CPU verified; Anima LoHa/LoKr follow capability tables |
| Official-weight LoRA training/resume/generation | `hybrid_anima_lora_gpu_2026-10.json` | Anima GPU verified; this does not prove Qwen/Ming GPU LoRA acceptance |
| VAE swap rebases native identity without replacing learned modules | Native `vae_alignment.py`; initialized/export and lifecycle tests | Anima/Ming/Qwen CPU contract; Anima combined GPU lifecycle verified |
| EMA, checkpointing and activation ownership | Anima lifecycle and skip/checkpoint tests | Structural parameters owned once; CPU EMA and checkpoint/offload paths verified within recorded scope |
| GPU/CPU/recompute skip policies preserve gradients and random conditions | `hybrid_skip_storage_test.py`, `hybrid_skip_recompute_test.py`, `hybrid_anima_skip_storage_2026-10.json` | CPU correctness and Anima GPU compute evidence; image-quality comparison is not established by compute evidence |
| Mutable Block Swap update completion and replay safety | `hybrid_anima_blockswap_2026-10.json`; conductor regressions | Anima full-frame GPU save/resume verified; unsupported combinations explicitly refused |
| Partition halo, core-only output/loss and full-canvas positions | `hybrid_partition.py`, native training planners, `hybrid_partition_test.py`, native tests | Training CPU contract verified; convolution halo cannot reproduce omitted global attention |
| Qwen hybrid partition/global generation and cleanup | `hybrid_qwen_partition_generation_test.py` | CPU native tile parity and complete adapter loader verified; required halo expansion added during this audit |
| Fresh-model teacher initialization; native teacher frozen; cancellation/failure rollback | `hybrid_fitting_transaction_2026-10.json`, initializer tests | Shared transaction and lifecycle verified; native weights are not silently recalibrated on resume |
| Initialization lineage and complete artifact publication | `hybrid_initialized_export_2026-10.json`, `hybrid_ming_initialized_export_2026-10.json`, provenance tests | Implemented; origin metadata describes initialization, not later FT weights |
| Real public initialization/load/generation and cancellation | `hybrid_initialization_api_2026-10.json`, Qwen and Ming Design/Layer API records | Bounded official-weight GPU functionality verified; each record's cancellation scope remains distinct |
| API defaults/OpenAPI serialization, explicit unsupported-architecture refusal | `param_defaults.py`, `openapi.yaml`, `rewiring_routes.py`, strict hybrid architecture validation | Hybrid is explicitly limited to Anima/Ming/Qwen; legacy runtime membership does not imply hybrid support |
| Editable structure, graphical grids/skips/census/lifetime, readonly inherited structure | `DiTRewiringEditor.tsx`, `HybridDiTModuleEditor.tsx`, graph/summary components | Static wiring reviewed; owner build/interactive confirmation remains separately recorded |
| Edit config round-trip and state/cancellation/error behavior | API-to-YAML adapter tests; editor/API types and initialization panel static review | Static/CPU evidence; no owner build result is invented |
| Final native/hybrid × native/Qwen VAE matrix at 1024, batches 1/2/4 | `hybrid_anima_final_matrix_2026-10.json` | Completed with initialized structural weights, production bridge classes and explicit calibration provenance |
| Forward/backward/update, allocated/reserved VRAM, RSS, preprocessing separation | Final matrix per-update records and guide | Compute measurements complete; end-to-end VAE/TE/IO throughput is explicitly excluded |

Paths beginning with `core/` are under `backend/`; tests are under
`backend/tests/`; JSON evidence is under `docs/audits/data/`; UI components are
under `frontend/src/components/training/`. These entries do not replace the
individual records' limitations.

## Remaining acceptance work

1. **P5 quality:** the retained 3300-update native-teacher initialization plus
   low-LR Full FT preserves and improves selected Anima figures. It is not an
   equal-total-GPU-time legacy/hybrid quality comparison or an initialized
   hybrid + Qwen codec quality acceptance. Run a bounded, fixed general-prompt
   comparison that separates reconstruction, single prediction and free-running
   generation before selecting a production default. The subsequent eight-image
   `hybrid_anima_combined_quality_2026-10.json` probe fails joint free-running
   quality acceptance despite finite outputs and some retained single-figure
   outlines. Native-codec initialization is insufficient evidence for the
   combined candidate; bounded joint adaptation remains necessary.
2. **P0 owner-run diagnosis:** synthetic same-input train/eval parity and native
   LR controls exclude some causes, but do not prove the cause of #174's old
   checkpoint degradation. Keep that limitation explicit; do not modify the
   owner artifact's noise policy to make a diagnostic pass.
3. **GPU scope:** Qwen/Ming LoRA numerical restoration has CPU acceptance, not
   official-weight GPU acceptance. Their three-update initializer probes prove
   publication/load/generation functionality, not quality recovery. Prioritize
   Anima/combined quality work before expanding GPU model downloads.
4. **Owner UI acceptance:** build/type checking and interactive editing remain
   owner work under AGENTS.md. They do not block static review or backend work.

P6 stage-width changes, heavy convolutions, audio/video and Lens GPU quality work
are conditional or outside this initial plan. They are not substituted for the
unfinished quality gate. Existing legacy defaults remain unchanged until the
new candidate has quality acceptance. A successful loss curve, green CPU tests
or a speedup alone is insufficient to close the goal.
