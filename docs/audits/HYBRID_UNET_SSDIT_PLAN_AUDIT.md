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
| Official-weight LoRA training/resume/generation | `hybrid_anima_lora_gpu_2026-10.json`, `hybrid_ming_lora_gpu_2026-10.json`, `hybrid_qwen_lora_gpu_2026-10.json` | Anima BF16 plus Ming/Qwen INT8 ConvRot GPU verified; untested BF16-base and quality scopes remain distinct |
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
| Equal initialization-plus-training compute comparison | `hybrid_anima_equal_total_2026-10.json` | Fresh native/legacy/high16 hybrid, each 240 CUDA-event seconds including hybrid teacher fitting; shared preprocessing/evaluation excluded |
| Recorded phase-specific compressed-skip LR control | `hybrid_skip_rate_public_2026-10.json`; fitting/provenance/API and native owners | API/UI static wiring,85 distinct CPU cases, Anima BF16 two-update GPU transaction and metadata round-trip verified; default1 preserves uniform updates and old records |

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
   combined candidate. A subsequent 200-update combined Full FT probe
   (`hybrid_anima_combined_adaptation_2026-10.json`, LR 5e-6) completed with
   finite gradients but retained strong block texture and did not show consistent
   quality recovery in the two reviewed CFG4 conditions. It is not accepted.
   Separate codec reconstruction, matched-noise single prediction and free-running
   generation before extending training or selecting another structure.
   That separation was subsequently measured on two general images in
   `hybrid_anima_codec_diagnostic_2026-10.json`: direct target reconstruction is
   sharp, while the calibrated native-channel bottleneck introduces artifacts
   without DiT/U-Net processing. Six matched single-prediction errors improve
   after adaptation, despite unaccepted free-running quality. Targeted IO/bridge
   and trajectory consistency work is indicated; this is not proof that hybrid
   topology is harmless or that the quality gate has passed.
   A restricted IO/bridge probe (`hybrid_anima_io_adaptation_2026-10.json`)
   then completed 200 updates with all frozen native/hybrid parameters exactly
   unchanged. It used 54.063 seconds and 4.085 GiB but still failed reliable
   free-running quality recovery; do not promote this initialization recipe.
   Keep native-codec hybrid acceptance/equal-time structural comparison distinct
   from the harder combined codec adaptation question.
   The native-codec 120-second post-initialization comparison
   (`hybrid_anima_equal_time_2026-10.json`) completed native/legacy/hybrid
   263/277/216 updates. Hybrid preserves a selected single figure but fails the
   two-person composition; no consistent quality advantage is established.
   It is also slower at 512px batch 1, unlike the larger-shape compute matrix.
   Historical teacher fitting adds 1046.156 recorded wall seconds, so this does
   not close the equal-total-budget requirement. Test student-state trajectory
   initialization before promoting the current candidate.
   The subsequent three-round student-state probe
   (`hybrid_anima_student_trajectory_2026-10.json`) kept the native parameters
   exact but increased texture/detail degradation in the reviewed outputs.
   It is not promoted into the public initializer. Change high-resolution stage
   allocation as one factor before trying additional fitting recipes.
   The 8/16/4 versus 12/12/4 private placement probe
   (`hybrid_anima_highres_allocation_2026-10.json`) starts with identical retained
   module weights and verifies 200 matched teacher inputs. Selected CFG4 figure
   placement improves at modest fitting cost, but distortions/lettering remain
   and inspected CFG1 still fails. This does not promote the candidate default.
   A further 16/8/4 placement probe
   (`hybrid_anima_highres16_2026-10.json`) reused the 12/12/4 comparator and
   verified all 200 matched exposures plus exact frozen hashes. Reviewed CFG4
   figure placement/outline improves again, but CFG1 fragmentation and remaining
   anatomy/text artifacts prevent acceptance. These are private transfers of
   retained initialization weights, not fresh per-layout acceptance. The new
   allocation has no 1024px compute matrix yet. Test guided-field consistency
   before extending the same fitting objective or promoting a default.
   The no-fit conditional/null probe
   (`hybrid_anima_guidance_diagnostic_2026-10.json`) measures 40 matched
   image/noise cases, verifies frozen hashes and the CFG error identity.
   High16 reduces low/mid-noise errors but worsens high-noise guided error;
   null error alone does not explain the discrepancy. Compare paired field
   anchors with a guided-field fitting objective before adding it publicly.
   This diagnostic does not close free-running or equal-total-budget acceptance.
   The paired fitting comparison (`hybrid_anima_guided_fit_2026-10.json`)
   verified 200 matched updates per arm and exact frozen native hashes. Added
   guided loss gives small field improvements but no clear selected visual
   advantage, so it is not promoted. Paired-anchor images preserve coarse
   figures and are retained for owner evaluation; exact native reproduction is
   not the acceptance goal. The native p28 CFG1 image itself has split/collage
   composition, so earlier CFG1 failures cannot all be assigned to rewiring.
   Native p29 CFG1 remains coherent; compare prompt-specific controls.
   The subsequent candidate matrix
   (`hybrid_anima_candidate_matrix_2026-10.json`) remeasured all four arms
   at 1024px/batches 1/2/4 with five timed samples. High16 alone retains
   1.43–1.54x speedup but no Full FT VRAM reduction; combined batch 4 gives
   6.10x native and 1.42x swap-only speed with 41.7% lower allocated VRAM
   than native. Batch 1/2 has no demonstrated combined advantage over swap.
   Candidate compute is now measured; equal-total-budget and combined-codec
   quality acceptance remain separate work. No owner checkpoint or public
   initializer changed, and coarse-structure owner evaluation is pending.
   The candidate combined probe
   (`hybrid_anima_candidate_combined_quality_2026-10.json`) generated only
   four new images and reused native/candidate/swap controls by hash. Its
   reviewed CFG4 pair still has severe block texture and composition loss;
   swap controls also degrade. High16 does not close combined initialization
   acceptance. Keep native-codec candidate evaluation independent and separate
   codec/IO initialization work from further structural-only fitting.
   A fresh budget-controlled comparison
   (`hybrid_anima_equal_total_2026-10.json`) subsequently completed at
   240.456/240.479/240.453 CUDA-event seconds for native/legacy/hybrid.
   Hybrid includes 297 fresh teacher updates plus 211 Full FT updates;
   retained module weights were not applied. Native/legacy complete 480/459
   Full FT updates, with all 211 common exposures exactly matched. Initialization
   freezes native weights exactly. The inspected hybrid single-figure face/outline
   improves after Full FT, but two-person artifacts remain and no consistent
   advantage over legacy is proved. This closes the bounded equal-model-compute
   comparison requirement, not broad quality/default acceptance. Shared input
   preprocessing/transfers and evaluation are excluded, so it is not total
   end-to-end GPU occupancy or the retained 3700-update recipe's comparison.
   The output-only clean cycle refit (`hybrid_codec_cycle_probe_2026-10.json`)
   retains the input bridge and token geometry, fits on 28 general images and
   evaluates four excluded from that refit. Mean clean reconstruction MSE improves
   by less than 1%, one case worsens, and inspected images retain block texture.
   The single-figure face improves but the two-figure edges deteriorate. It is not
   promoted and does not establish sampler/DiT improvement. Investigate retaining
   more codec information in reduced-grid IO or joint field-aware initialization
   rather than repeating this output-only fit.
   The packed-native codec feasibility probe
   (`hybrid_codec_packed_probe_2026-10.json`) matches all new fits to 28 general
   images and excludes four evaluation images. Packed-64 cycle reconstruction
   reduces mean latent MSE by 18.94% versus a freshly matched pooled-16 cycle,
   retaining the replacement token grid. Block texture remains. An explicit
   experimental Anima IO coefficient transfer passes synthetic linear-boundary
   checks including nonzero patch-constant padding, with one quarter the tokens.
   DiT/RoPE/noise-field and free-running generation remain unverified; no owner
   artifact, production schema or default changes. Verify those before adoption.
   Real packed IO generation (`hybrid_anima_packed_generation_2026-10.json`)
   tests two general prompts at 512px/28 steps/CFG4 with and without hybrid.
   Original per-cell noise complements collapse to purple even without hybrid.
   A head-aware average/repeat noise snapshot passes synthetic oracle/three CPU
   regression checks and clears the purple collapse, but corrected four-image
   outputs retain severe stripes/blocks and fail two-person structure. It is not
   adopted. Packed IO/field initialization needs further fitting before public
   publication or default changes; clean reconstruction gains are insufficient.
   Packed IO-only learning (`hybrid_anima_packed_io_fit_2026-10.json`) completes
   200 updates of six FP32 IO/bridge tensors (1,130,624 parameters) at LR 1e-4,
   with exact frozen body/structural hashes, finite state and counters 200.
   It costs 64.672 training seconds/4.095 GiB allocated; two new held general
   outputs reduce exaggerated eyes but retain strong block artifacts and fail
   two-person composition. No default/adoption or broad quality claim follows.
   Short IO-only learning is insufficient; coherent IO/structural adaptation or
   a different reduced-grid initialization still needs validation.
   A matched IO-only versus IO+structural continuation
   (`hybrid_anima_packed_joint_fit_2026-10.json`) verifies 200 identical item/null/
   sigma/noise exposures and exact frozen-body hashes. All six versus 71 optimizer
   counters reach 200. Joint adaptation costs 81.344s/4.124 GiB versus
   64.562s/4.104 GiB; four held outputs have somewhat smoother joint texture but
   still fail two-person structure and retain severe blocks. This does not close
   initialization/default quality acceptance, despite lower matched training loss.
   A concurrent IRIS GPU generation worker invalidates isolated throughput
   ranking from these wall times; per-worker allocated memory and functional
   matching/frozen-state evidence retain their stated scope.
   Check reduced-grid geometry/position and field mapping before further fitting.
   The no-fit source-span RoPE retrofit
   (`hybrid_anima_source_span_2026-10.json`) passes three GPU-free coordinate tests,
   but four real held general images worsen native facial/composition structure
   and do not fix packed blocks. Existing unit-grid-fitted corrections are not
   silently changed. This rejects that retrofit, not fresh source-span fitting;
   explicit position-policy identity and matched initialization would be required.
   The matched fresh initialization control is now complete
   (`hybrid_anima_source_span_fit_2026-10.json`): both arms start with identical
   new 16/8/4 modules, preserve the same frozen body and match all 300 exposure,
   null, sigma and noise records. All 65 optimizer entries reach300. Four new
   held general outputs show unit-grid retaining clearer figure outlines while
   source-span32 produces overlapping/smeared figures and worse faces. Reject
   source-span32 as the default in this bounded test; neither unit-grid's quality
   gate nor combined-codec acceptance is closed. Longer fits/other geometries
   remain untested, and existing owner artifacts/defaults are unchanged.
   A seven-skip rank64-to256 control
   (`hybrid_anima_wide_skip_fit_2026-10.json`) preserves other BF16 initial weights,
   matches all300 exposure/noise records and has an exactly equal initial loss.
   The native body is unchanged; all65 optimizer entries reach300. At unchanged
   LR0.001 the two new held outputs collapse, while own peak allocation rises
   only0.025GiB. Initial error grows after the first update; a skip-only LR control
   is the next concrete hypothesis. This rejects the tested recipe, not wider-skip
   capacity generally. Previous per-image VAE posterior hashes are unavailable,
   and concurrent GPU work prevents isolated speed ranking. Quality remains open.
   The subsequent skip-only LR0.00025 control
   (`hybrid_anima_wide_skip_lr_fit_2026-10.json`) reuses the exact hashed encoded
   inputs and initial modules from rank256; all300 exposures/noises match and
   the base remains exact. All65 tensors reach300. Single-figure face/outline
   recovers, but two-person composition remains overlapping/fragmented and the
   comparison with rank64 is mixed. Optimization sensitivity is established;
   default quality acceptance is not. Per-phase skip LR is now wired through
   the public API, shared transaction and Anima/Ming/Qwen owners, with explicit
   lineage and old-record preservation (`hybrid_skip_rate_public_2026-10.json`).
   New initialization can reproduce declared rates; private historical weights
   are not retroactively granted public lineage.85 CPU cases and a real Anima
   BF16 two-update transaction pass; owner build/UI and quality remain separate.
   The new factory rank256/skip-LR0.00025 candidate also completes an equal-total
   budget (`hybrid_anima_wide_skip_equal_total_2026-10.json`):319 teacher updates
   plus222 Full FT updates total240.113 CUDA-event seconds. All211 common FT
   item/null/sigma records match historical controls; their old latent/noise
   hashes are unavailable. Six new images preserve initial coarse figures but
   Full FT damages two-person composition, while one face/outline survives.
   Reused native controls remain stronger; no consistent advantage or default
   acceptance. Compare the trained native-body path with the rewired path under
   identical weights before continuing the same fitting/FT recipe. Private
   historical weights do not become public initialized artifacts by this test.
   The trained-routing diagnostic
   (`hybrid_anima_trained_routing_2026-10.json`) matches222 item/null/sigma/noise
   records and first loss, but not later losses or historical image hashes;
   exact historical replay is unproved. Its within-model native/hybrid routing
   comparison preserves the same trained-body hash and codec. Both routes fail
   the two-person composition; both retain a single girl with differing defects.
   Native routing cannot cure this sample, but a body trained for hybrid execution
   is not guaranteed to preserve native behavior. No sole/root cause or#174
   diagnosis is claimed. Compare higher correction-only FT rates before adding
   more structural capacity; capture RNG boundaries in subsequent probes.
   The paired Full FT correction-rate control
   (`hybrid_anima_ft_module_lr_2026-10.json`) now matches initial body/module
   hashes, CPU/CUDA RNG boundaries and all222 item/null/sigma/noise records.
   Both arms use two optimizer groups; native LR stays5e-6 while structural LR
   is5e-6 versus5e-5. All632 optimizer entries reach222; first loss is identical.
   The first updated native-body hashes differ, so bitwise update reproducibility
   remains unproved. Eight held general outputs retain a single face but fail
   two-person composition in both arms; the tenfold rate adds a large horizontal
   split in the CFG4 scene. Reject this bounded recipe as a default remedy.
   Loss means0.169883/0.170308 do not establish quality recovery. Own Full FT
   allocation is11.643/11.644GiB; this is an equal-update, not equal-time test.
   Next isolate body adaptation with frozen-body structural fitting at the same
   correction rate and encoded exposures. That diagnostic does not replace
   Full FT support, establish#174's cause or close the quality gate.
   The frozen-body control (`hybrid_anima_frozen_body_control_2026-10.json`)
   completes222 identical exposures from the same initialization/RNG boundary
   at structural LR5e-5. Native-body hashes remain exact and all65 optimizer
   entries reach222. CFG4 restores two distinct heads/standing figures relative
   to joint Full FT, but torso/hand/scene defects remain; the single girl retains
   exaggerated facial defects. CFG1 still has overlapping/fragmented structures.
   Own allocation4.501GiB and63.508 training seconds describe this diagnostic,
   not Full FT speed. Freezing also changes gradient clipping and stochastic
   rounding draw consumption. Native-body adaptation is a useful intervention
   to investigate, not an established sole cause or permission to replace Full FT.
   Next compare joint adaptation with a frozen-native-teacher field retention
   loss, recording its extra compute/memory separately before any public wiring.
2. **P0 owner-run diagnosis:** synthetic same-input train/eval parity and native
   LR controls exclude some causes, but do not prove the cause of #174's old
   checkpoint degradation. Keep that limitation explicit; do not modify the
   owner artifact's noise policy to make a diagnostic pass.
3. **GPU scope:** a real Ming INT8 ConvRot LoRA probe exposed a missing preview
   autocast before its first update. DiT-only trainer precision is now wired
   through the sampling bridge, with a native BF16/FP32-adapter CPU regression.
   The subsequent real Ming INT8 retry/save/resume/generation passes: 72 structural
   payload tensors match the factory, all 344 optimizer entries restore with
   none fresh and counters 3, and saved-payload generation has no warnings.
   The owned smoke run is removed and owner model restored. This closes Ming
   INT8 functionality, not BF16-base or quality acceptance. Qwen INT8 ConvRot
   now also completes two updates, step-2-to-3 continuation and saved-payload
   generation without warnings. Its 72 factory-matched structural tensors and
   328 optimizer entries restore with none fresh and counters 3. Both owned
   smoke runs are removed and the prior Ming Layer model restored. This closes
   the bounded native-codec LoRA GPU functionality probes on all three supported
   architectures; combined codec/partition and quality scopes remain separate.
   Their three-update initializer probes prove
   publication/load/generation functionality, not quality recovery. Prioritize
   Anima/combined quality work before expanding GPU model downloads.
4. **Owner UI acceptance:** build/type checking and interactive editing remain
   owner work under AGENTS.md. They do not block static review or backend work.

P6 stage-width changes, heavy convolutions, audio/video and Lens GPU quality work
are conditional or outside this initial plan. They are not substituted for the
unfinished quality gate. Existing legacy defaults remain unchanged until the
new candidate has quality acceptance. A successful loss curve, green CPU tests
or a speedup alone is insufficient to close the goal.
