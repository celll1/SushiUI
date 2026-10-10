# Hybrid U-Net ssDiT plan audit

Reviewed 2026-10-10 against `local/strategy/SDXL_LIKE_UNET_SSDIT_PLAN.md`.
This is a completion audit, not a declaration of completion. The implementation
and compute matrix are substantial; P5 quality acceptance remains incomplete.
Existing owner runs and their artifacts are not migrated implicitly.

## Implemented contracts and evidence

Owner evaluation policy update (2026-10-10): VAE-swap quality may require long
training beyond bounded verification. Evaluate matched before/intermediate/after
improvement tendencies, not disappearance of clipping/grid artifacts within a
short probe. Historical unsuccessful quality probes describe their finite
budgets; they do not prove inability to recover through long training. Functional,
numerical and compute acceptance is separate from final appearance quality.
The active1000-update comparison retains500/1000 generation checkpoints to
evaluate this trend against existing initial samples.

Interim evidence (`hybrid_anima_codec_fullft_interim_2026-10.json`): the
independent-spatial arm completed500 Full FT updates after its retained200 IO
updates. Direct comparison of the two held1024px CFG4/seed63 images shows more
continuous clothing/background areas for p28 and cheek/hand/food color areas for
p31. Clipping, grid fragments and colored spots remain. This is a selected
positive tendency, not a claim that the second calibration arm,1000-update
comparison, broader quality or long-training recovery is verified. All638
trainable tensors had finite gradients at the500-update boundary; recorded
compute was359.771 seconds with11.793 GiB peak allocated. VAE/TE encoding,
input preparation and sample evaluation are excluded from that timing.

The first arm subsequently completed1000 updates
(`hybrid_anima_codec_fullft_first_arm_2026-10.json`). p28 large areas improve
further from500 to1000; p31 has stronger clipping and fine fragmentation than
at500 despite retained face/hand/food structure. The trend is prompt-dependent,
not monotonic. All638 optimizer entries reach1000, body weights change and the
frozen caption bridge remains exact. Recorded training compute is718.947
seconds, peak allocated11.793 GiB. The matched native-moment arm is pending.

The native-moment arm now has a500-update checkpoint. Its two reviewed images
also gain more continuous clothing/background and face/hand/food color regions
relative to their own starting samples. Clipping/grid fragments remain; no
clear advantage over the independent500-update control is demonstrated.
The interim record now verifies identical500 item/null/sigma/noise exposures,
initial body and structural hashes between arms. Native-moment training compute
is353.405 seconds and peak allocated11.791 GiB. Its1000-update point is pending.

The matched comparison is now terminal-successful
(`hybrid_anima_codec_fullft_trends_2026-10.json`): both arms finish1000 updates
with exactly matched item/null/sigma/noise sequences and shared initial body/
structural hashes. Both have638 optimizer entries at1000, changed body weights
and exact frozen caption bridges. Native-moment compute is710.857 seconds,
peak11.791 GiB; independent compute is718.947 seconds, peak11.793 GiB.
Direct initial/500/1000 review is prompt-dependent: independent p28 improves
further, native-moment p28 retains a coherent scene but has more fragmented
surface texture than at500, and both p31 outputs have stronger clipping/fine
fragmentation than at500. No clear native-moment advantage is demonstrated;
the production calibration policy is retained. This closes the bounded matched
trend comparison, not long-training quality or broad appearance acceptance.

| Requirement | Current evidence | Assessment |
|---|---|---|
| Fixed native width; learned down/up, local residual processing, multiple compressed skips | `core/rewiring/hybrid_spec.py`, `hybrid_modules.py`; `hybrid_rewiring_test.py`, native Anima/Ming/Qwen tests | Implemented; isotropic 2x boundaries, native widths retained |
| Zero output branches preserve the initial residual baseline; internal branches receive gradients after training | `hybrid_anima_test.py`, `hybrid_qwen_test.py`, `hybrid_ming_test.py`; Anima lifecycle records | Verified on CPU and bounded Anima GPU training |
| Text/prefix tokens and Layer frame axes remain separate from image-grid processing | Native residual owners and `hybrid_ming_test.py` | CPU native contract verified; real Ming Layer initializer/generation verified |
| Stable module IDs, exact topology/module/native identities, strict complete payload loading | `spec.py`, `hybrid_spec.py`, native loaders; `hybrid_provenance_test.py`, export tests | Implemented; missing/non-finite/inconsistent payloads refused |
| Full FT save/resume and named optimizer ownership | `hybrid_anima_lifecycle_2026-10.json`, `hybrid_anima_initialized_full_ft_2026-10.json` | Real Anima GPU continuation verified, including structural parameters |
| Public full-rank boundary fusion + bundled Qwen64ch VAE through the complete training API | `hybrid_fullrank_combined_lifecycle_2026-10.json` | General-only8-item run: two updates, saved-step2 to3 resume, partition/global adapter, activation dispatch and CPU EMA;67 structural tensors,643 optimizer entries restored with none fresh,643/643 EMA restored; saved regular/EMA layouts match, periodic PNG and9 decoded debug images verified; owned DB/disk data removed |
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
| Selectable measured Anima candidate geometry | `hybrid_measured_candidate_api_2026-10.json`; SSOT candidate definitions, inspect/OpenAPI/frontend types | Explicit16/8/4 full-width fusion with rank256 skips; exact measured graph and67 tensors/35,629,056 added params,53 CPU cases, actual API round-trip after restart; standard defaults/inherited artifacts unchanged; fitting/quality not implied |
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

### Current status (2026-10-10)

- P1-P4 implementation and the bounded native integrations are covered by the
  contracts above. The hybrid preset is explicitly selectable through
  `DiTRewiringEditor.tsx` and `inspect.py`; existing owner structures are not
  replaced. Owner build/interactive verification remains separate.
- P5 candidate selection now exposes the measured Anima16/8/4 full-width
  fusion graph as an explicit UI preset for matching28-block/2048-width/patch2
  models. It carries geometry, not retained fitted weights; the standard
  preset/default is not promoted by this comparison.
- The requested1024px batch1/2/4 native/hybrid x native/Qwen-VAE compute matrix
  exists, including the later full-rank feature-fusion candidate in
  `hybrid_anima_public_feature_fusion_matrix_2026-10.json`. Statements below
  that an older candidate still needs its matrix describe that experiment's
  historical state, not absence of the current matrix. Encoding/IO/partition
  remain excluded from its compute scope.
- P5 combined-codec training tendencies are being evaluated. The500-update
  independent-spatial observation is positive for two selected held prompts;
  its1000-update point shows continued improvement for p28 but stronger clipping
  for p31. The native-moment comparator has similar positive500-update
  tendencies without a demonstrated advantage; its1000-update point also
  develops stronger p31 clipping. The matched comparison is complete.
  Neither artifact removal within this budget nor exact native reproduction
  is required. Broader or long-training quality is not established.
- P0 does not establish the exact cause of the owner's old #174 degradation.
  Existing diagnostic exclusions are retained; the owner artifact is untouched.
- P6 variable-width student work is conditional. Negative private prototypes
  are not a requirement to ship a narrower public graph or a reason to expand
  this stage indefinitely.

### Historical experiments and their individual limitations

The entries below preserve the evidence and decisions at each experiment's
time. Their short-budget quality judgments do not supersede the current owner
trend policy or the implementation/matrix evidence above.

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
   The equally weighted native-field anchor
   (`hybrid_anima_native_anchor_2026-10.json`) completes222 matching exposures
   and RNG/start hashes. All632 optimizer counters reach222; the frozen native
   teacher hash remains exact. The supervised initial loss matches the control.
   Final50 mean supervised/anchor losses are0.168214/0.023632, but four new held
   outputs still fail two-person structure and retain a large horizontal band.
   A single face survives with body defects. Reject this bounded recipe as a
   default remedy; small field MSE does not establish free-running quality.
   Teacher forward costs15.410 CUDA seconds in addition to106.549 student update
   seconds; joint peak allocation15.555GiB includes4,182,137,856 teacher parameter
   bytes. This is not an equal-total-time comparison or isolated speed ranking.
   Next test paired conditional/unconditional guided-field retention to separate
   CFG amplification from single-field matching; owner artifacts remain untouched.
   The raw CFG4 paired-field anchor (`hybrid_anima_guided_anchor_2026-10.json`)
   also completes222 matched exposures/RNG/start hashes,632 optimizer counters
   at222 and an unchanged teacher. Conditional and unconditional student forwards
   participate in the guidance loss; dropout exposures reuse the null branch.
   Last50 supervised/guided losses0.170859/0.058254 do not restore composition.
   Four new held outputs retain fragmented figures and a horizontal band; one
   CFG4 face survives, while CFG1 has poor prompt differentiation. Reject this
   bounded recipe. Student/teacher compute174.208/29.909 CUDA seconds and joint
   allocation16.425GiB add cost without the required visual recovery. Neither
   single-field nor guided-field retention establishes a quality remedy; do not
   keep varying these losses without new evidence. Next inspect restored-stage
   intermediate features and evaluate normalized feature alignment during
   structural initialization. Full FT and combined-codec quality gates stay open.
   The same-body feature diagnostic
   (`hybrid_anima_feature_diagnostic_2026-10.json`) covers12 no-training cases:
   two general cached training inputs, three sigmas and conditional/null paths.
   Native-body hashes remain exact across native/zero/fitted319 execution.
   Across28 blocks, teacher coarse grids are average pooled for like-size metrics.
   Mean relative feature error at block11 improves7.526→0.425 and at27 improves
   0.03097→0.01136, but restored full-grid block20 remains0.39293→0.39640.
   This suggests testing intermediate alignment, not a proved generation cause.
   The experimental image-grid alignment helper passes six GPU-free CPU tests
   for pooling, identity, scale normalization, student-only gradients and refusals.
   Next compare matched200-update field-only versus field+0.1 normalized feature
   alignment at16/20/27, starting from the same319 state with a frozen body.
   Held generation and subsequent Full FT remain required before public adoption.
   The matched continuation (`hybrid_anima_feature_fit_2026-10.json`) completes
   200 identical item/null/sigma/noise exposures from equal319 state/RNG hashes.
   Both frozen native bodies remain exact; all65 optimizer counters reach200.
   Field-only versus field+0.1 feature loss yields final50 mean feature errors
   0.46980/0.31268 and field errors0.02949/0.02919. Eight held general outputs
   show feature fitting worsening horizontal stripes and figure coherence;
   field-only retains two heads but has body/face defects. Reject this recipe.
   Compute64.264/69.353 CUDA seconds and allocation4.524/4.527GiB describe the
   equal-update continuation, not quality acceptance or an isolated speed gain.
   These taps observe native block outputs before post-block local refinement.
   In particular16/20 do not represent the complete stage output. The preceding
   diagnostic has the same scope; do not infer a complete decoder mismatch from
   those tap errors. Inspect post-refiner stages before changing fusion. If that
   mismatch remains, test plan3.3's compressed concat fusion rather than more
   output-loss tuning. Private continuation weights have no public fit lineage.
   The post-refiner diagnostic (`hybrid_anima_stage_diagnostic_2026-10.json`)
   covers the same12 cases and exact unchanged native body across zero/319/
   field519/feature519 states. Its observer leaves the old zero/319 field
   predictions unchanged. Restored stage20 relative error remains near0.39–0.40
   after local refinement; feature519's small reductions do not recover held
   generation. This supports a fusion comparison, not a proved degradation cause.
   A compressed concat correction primitive now exists experimentally: current
   and compressed skip are separately normalized, mixed at skip rank and
   projected back, with per-token RMS units and zero final projection.
   Six CPU cases verify exact initial preservation, staged inner/input gradients,
   parameter census, tile equivalence, invalid contracts, zero-input gradients
   and rank1 skip information. It is not installed by the public topology factory
   or given artifact lineage/default acceptance. Next verify integration and
   matched GPU fitting before public API/UI and payload wiring.
   Exclusive private concat integration now passes11 CPU cases for resident/
   CPU/recomputed skips, normal/CPU checkpointing, zero equivalence, gradients,
   dtype alignment and failure cleanup. The first GPU attempt failed before any
   update because new corrections stayed on CPU; context device/dtype alignment
   is fixed and the retry completes200 updates (`hybrid_anima_concat_fit_2026-10.json`).
   All100 optimizer counters reach200, all35 new tensor gradients are nonzero,
   native-body hashes remain exact, and200 exposures/RNG boundaries plus first
   field loss match the reused field-only control. Seven corrections add7,829,248
   parameters. At new-branch LR0.00025, four held general outputs have worse
   magenta color/face/figure defects; reject this recipe. Final50 mean field error
   0.03514 versus0.02949 does not establish benefit. Whole-probe compute73.312s/
   allocation4.549GiB is not isolated overhead: the control retains zero-weight
   feature taps. Test new-branch LR0.000025 with old-group rates held fixed before
   judging the structure. Public topology/payload/default acceptance stays open.
   The tenfold lower new-branch rate also completes200 exactly matched exposures
   (`hybrid_anima_concat_lr_control_2026-10.json`): all100 counters reach200,
   all35 new gradients are positive and the native body remains exact. Final50
   field error improves to0.02812, but both CFG1/4 held prompts retain strong
   blue-magenta coloration and fragmented faces/figures. Reject this recipe too;
   do not infer that every concat design fails. Whole-probe79.653s/4.549GiB is
   not isolated overhead or Full FT evidence. Before more rate/loss variations,
   measure velocity mismatch on native and student-owned free-running trajectories.
   Current public trajectory initialization captures native teacher paths only;
   input-distribution mismatch is a hypothesis, not an established cause.
   The bounded rollout diagnostic (`hybrid_anima_rollout_diagnostic_2026-10.json`)
   now cross-evaluates native/fitted319 on each route-owned CFG4 trajectory for
   one fit and one held general prompt,28 steps and identical seed63/start noise.
   No weights change; native hashes match. Mean guided error is lower on student
   inputs (0.10123/0.05297) than native inputs (0.13153/0.08495), including relative
   error and the last7 steps. This does not support the proposed error-growth
   hypothesis for these prompts. Student last-observed latent RMS is0.55035/
   0.34464 versus native0.73277/0.47225; these are pre-final-update states, not
   decoded quality or a proved cause. Separate same-input velocity amplitude and
   direction before adding on-policy fitting. Full quality/default gates remain open.
   Cached-trajectory velocity geometry (`hybrid_anima_velocity_geometry_2026-10.json`)
   reproduces every prior guided error and latent metric exactly without new
   generation or updates. Global student/native velocity norm ratios average
   0.973–0.990; projection gains0.940–0.962. Orthogonal error explains0.04760–
   0.07564 of total relative error0.04962–0.08226; scalar parallel error is only
   0.00202–0.00773. The projection decomposition is checked numerically.
   Thus smaller generated latent RMS does not establish a simple velocity-scale
   defect. Do not add a global output multiplier on this evidence. The responsible
   topology operation and a quality-preserving replacement remain unproved.
   A private full-width skip control adds zero-output depthwise3x3 restoration
   alongside existing compressed skips, retaining all source channels rather
   than further widening a channel-mixing bottleneck. Twelve CPU cases verify
   storage/checkpoint gradients, exact preservation of nonzero existing skips,
   full-channel identity, dtype alignment and exception restoration.
   Matched200 frozen-body updates (`hybrid_anima_full_width_skip_fit_2026-10.json`)
   add129,024 weights in7 tensors; all72 optimizer counters reach200 and all7
   new gradients are positive. Source body/RNG boundaries/first loss and all
   exposure/noise records match. Final50 field error0.02765 does not recover held
   quality: CFG4 loses a person and washes out cyan/white, while CFG1 has damaged
   limbs and fragments of another composition. Reject this recipe as a default,
   not every full-width fusion. Whole-probe79.381s/4.519GiB is not isolated
   overhead or production Full FT evidence. Public preview omits the extra
   full-channel capture cost; this temporary wrapper must not be publicly exported.
   The next fixed-body comparison (`hybrid_anima_conditioning_fit_2026-10.json`)
   holds319 initialization, both-condition forwards,200 noisy exposures and
   starting RNG fixed. Paired field anchoring versus an added0.1 normalized
   conditional-minus-null error uses a1% teacher-field-energy floor. Six CPU
   analytic/gradient/layout cases pass. All65 optimizer counters reach200,
   native hashes remain exact and first anchor/contrast losses match between arms.
   Contrast last50 error falls0.36484→0.31167, while field anchor rises0.02468→
   0.02835. Held CFG4 face/figure defects worsen and CFG1 scene mixing remains.
   Reject this recipe; do not add the objective to public initialization.
   Whole-probe131.331/126.731s and4.980/4.985GiB are equal-update evidence, not
   a universal speed result. Always paired conditioning differs from the earlier
   single-condition/dropout control. Before further loss/rate variations, inspect
   fusion of encoder detail and coarse update at restored resolution boundaries.
   A private boundary-detail control now adds a channelwise gain on saved minus
   interpolated low-input features, after existing learned up correction. Low
   input includes learned down, so this is not a strict high-pass filter.
   Fifteen new CPU cases plus32 existing runtime/hybrid cases verify identity,
   nested/grouped layouts, gradients/checkpoint/skip policies, optional-hook
   isolation, transfer failure recovery and shape rejection. The public path
   has no callback by default. Both active-gain and same67-tensor zero-rate arms
   finish200 matched exposures from319, identical starting RNG and first field
   loss, with native-body hashes exact (`hybrid_anima_boundary_detail_fit_2026-10.json`,
   `hybrid_anima_boundary_detail_zero_rate_2026-10.json`). Two gains add4,096
   weights; active gains change while zero-rate gains remain exactly zero.
   Final50 field errors0.02808/0.03250 accompany a bounded CFG4 benefit: active
   gains retain two back-facing figures and a coherent girl face, while zero-rate
   loses a person and deforms the face. CFG1 still fragments. Retain this as a
   candidate, not a default or proved general/Full FT improvement. Old65-tensor
   control has different SR ownership; use the matched67-tensor arm for comparison.
   Whole-probe63.021/79.140s and4.509GiB are not isolated boundary overhead.
   Additional held prompts retain some CFG4 shapes but lose hands, pose and
   mouth/burger contact. Turning gains off only at inference preserves the two
   original coarse compositions: direct inference-gain benefit is not proved.
   The matched222-update Full FT comparison then regresses both candidates;
   two-person scenes become strips/figure fragments and the active burger sample
   loses its object/hands. All634 optimizer counters reach222, native bodies
   change and item/null/sigma/noise records match. Last50 losses0.16975/0.16996
   are not quality recovery. Own11.641/11.645GiB and103.571/102.011s are bounded
   probe costs, not a new matrix. Reject this Full FT recipe as a default.
   Holdout/Full FT evidence lives in the two corresponding boundary_detail JSON
   records. The matched native-only control completes the same222 exposures,
   native LR and starting RNG with567 optimizer counters reaching222. CFG4
   retains two figures, external hands and a burger at the mouth, though text
   contamination remains and CFG1 still fragments. Thus the exposure recipe
   alone does not explain converted-model failures. Different parameter sets
   alter clipping/SR; the precise cause is not established. See
   `hybrid_anima_native_matched_fullft_2026-10.json`. Five starting-state
   gradient exposures now exclude structural norm dominance in that bounded
   sample: added tensors contribute0.57–1.37% of squared norm, while mid/deep
   native-block gradient cosines are only0.068–0.101/0.016–0.098. This measures
   gradients, not Adam/SR updates or a cause. Separately fix shared trainer
   clipping across all optimizer groups;49 CPU regression cases pass. Earlier
   private probes already clipped all params, so this fix does not resolve their
   quality regression. See `hybrid_anima_gradient_allocation_2026-10.json`.
   A factory-described boundary-capacity control now widens64→512 with fitted
   coordinates copied and new output columns zero. Ten new CPU cases plus25
   existing hybrid cases pass. Same200 cached exposures/start RNG, native hashes
   exact,65 counters200 and strict saved-module restoration are verified.
   Structural weight count rises10,463,232→17,803,264. BF16 first field loss
   differs6.05e-5; changed SR ownership prevents exact update parity. Last50
   error0.02539 improves on0.02949, but CFG4 faces/hands and two-person structure
   remain broken with color/band defects. Reject this recipe; neither all
   capacity layouts nor longer fitting are ruled out. Record:
   `hybrid_anima_boundary_capacity_fit_2026-10.json`. Down/up/fusion information
   A private spatial-phase down/up branch now separates2×2 positions before
   mixing and expands original coarse delta into four output phases. Fourteen
   new cases plus32 existing runtime/hybrid cases pass. Same200 inputs/RNG and
   first field loss match;77 counters reach200 and the native body stays exact.
   Explicit private-context restoration reproduces77 tensors. It adds2,629,632
   weights, but CFG4 still loses figures/faces and develops cyan bands/lettering.
   Last50 field error0.02833 versus0.02949 is not quality recovery. Reject this
   recipe; pack order is retained but its channel bottleneck remains lossy.
   See `hybrid_anima_phase_boundary_fit_2026-10.json`. Public default promotion,
   practical Full FT/combined-codec acceptance and final matrix remain open.
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

The conditional P6 width investigation now has a private head-aligned Anima
block student with selected QKV/MLP/AdaLN weights and optional learned
encoder/decoder bridges. Sixteen CPU cases and synthetic real-block GPU probes
verify full-width exact BF16/RoPE output, finite narrow backward and gradients
into omitted output channels. Block weights shrink69.207M→24.118M/8.389M with
bridges, but initial synthetic source-relative errors0.702/0.730 require fitting.
Outer stream width stays2048; this is not a variable-width stage graph or public
artifact lineage. See `hybrid_anima_projected_block_2026-10.json`. Private stage
ownership and experimental weight restoration now have ten CPU cases; together
with projection cases26 pass. A real twelve-block width1024 teacher fit reaches
all329 optimizer counters200 with unchanged native fingerprints and explicit
payload reload. Two general held generations remain fragmented/banded; this
recipe is rejected. See `hybrid_anima_projected_stage_2026-10.json`. Alignment
quality, Full FT, public lineage and optimizer resume remain open.
One-block timings are not the required final U-Net±codec matrix. Heavy
convolutions, audio/video and Lens GPU quality work remain conditional or
outside this initial plan. They do not replace the unfinished quality gate.
Existing defaults remain unchanged until quality acceptance; loss, CPU tests
or a speedup alone cannot close the goal.

A private full-rank1×1 boundary correction now mixes saved fine features and
actual coarse input/output, with an optional runtime hook and zero initial output.
Thirteen new CPU cases plus32 runtime/hybrid cases pass; all67 real teacher-fit
weights reload exactly in a fresh private CPU context. Two general CFG4 images
retain coarse figures after200 teacher updates, but matched222 supervised Full
FT updates degrade the two-person composition and face/body. All634 optimizer
counters reach222 and native weights change; no public/default acceptance is
claimed. See `hybrid_anima_boundary_features_2026-10.json`. Full FT stability,
public lineage, combined codec and final matrix are still required.

The follow-up stability controls isolate all-native freezing and tenfold lower
Full FT rates from the same initial fusion/native state. Frozen-body222 updates
improve coarse composition relative to earlier Full FT but retain eye/body
defects. Tenfold-lower rates update all634 tensors to222, change native weights
and retain coarse figures in two held scenes. A fresh private-context reload
matches all755 full-body tensors and preserves an additional held food/person
composition. This bounded candidate does not establish public optimizer resume,
codec/partition parity, equal-time general quality or final performance. See
`hybrid_anima_boundary_features_stability_2026-10.json`. Defaults remain unchanged.

Full-rank boundary fusion is now an explicit optional schema-v2 operator with
factory-owned weights, public export/loader reconstruction, editor selection,
graph markers, parameter counts and logical forward-workspace estimates. CPU
integration:213 passed,16 unsupported adapter algebra combinations skipped.
Real Anima Full FT:755 research tensor values explicitly migrated,634 tensors
updated twice, ordinary adapter checkpoint saved and reloaded,634 named optimizer
states restored with zero fresh entries, then all counters advanced to3. A held
general composition remains coherent. This closes the operator's bounded
save/load/remapping check, not full process resume, broad quality, combined codec
acceptance, stage-width redesign or the final1024px B1/B2/B4 performance matrix.
Frontend checks are static; owner build/UI acceptance remains. Evidence:
`hybrid_anima_public_feature_fusion_2026-10.json`.

The public full-rank candidate now has the requested1024px B1/B2/B4 four-arm
compute measurement. All arms share the trained native body and use fresh
workers, BF16/SR AdamW8bit,24 native block checkpoints and clipping1; one warmup
plus five measured updates. Hybrid-only update speed improves1.42–1.51x but
allocated VRAM rises0.22–0.41GiB. Qwen-only is faster than combined atB1/B2;
combined wins1.35x atB4. Synthetic inputs, VAE/TE/IO/global/partition exclusions
and reuse of40 general calibration records are explicit. This closes this
candidate's compute measurement, not a quality-approved final architecture.
See `hybrid_anima_public_feature_fusion_matrix_2026-10.json` and guide table.

Combined public export/load reconstructs the bundled64ch/16x Qwen codec, ready
spatial/noise bridge and exact fusion module digest. Held general CFG1/4 samples
are still visually negative: gridlike noise, missing figures and changed
composition. A matched swap-only control made by removing just the U-Net from
the same restored artifact also fails, so the common artifact is not evidence
against U-Net alone. Native-codec controls retain figures. Codec/noise alignment
and combined learning quality remain open; throughput and strict restoration
do not close P5. See `hybrid_anima_public_feature_fusion_combined_2026-10.json`.

The subsequent same-token-grid native256 control also exposes a resolution
weakness: no-U-Net retains figures, while the512px-fitted hybrid loses them.
Matched200-update single512 versus mixed512/256 added-weight fitting freezes
the body and restores all67 counters, with identical item/null/sigma/noise seeds
and first loss. Mixed fitting improves small-grid person/hand outlines while
held512px compositions remain coherent. At swapped1024px, mixed weights restore
the missing second figure; strong gridlike texture still remains. This supports
multi-grid initializer work alongside codec/IO adaptation, not a claim that
codec error alone explains every failure. No defaults changed; multi-grid public
recipe, combined Full FT and broad quality are still pending. Evidence:
`hybrid_anima_public_feature_fusion_multigrid_2026-10.json`.

Multi-grid implementation has started with shared resolution validation,
deterministic cycling, shape-independent timestep RNG and optional strict
provenance fields. CPU verification reports40 passing tests, including legacy
recipe/hash preservation and initialization ownership. This does not close
public initializer wiring or GPU acceptance by itself. The existing single-grid
runtime is unchanged.

Public mixed-grid wiring now covers Anima, Qwen and Ming preparation and
trajectory buckets, API defaults/request validation, every-grid geometry
preflight and the frontend settings editor/types. Real tiny Ming Design/Layer
CPU owners verify both grids enter fitting, export/reload lineage and cleanup
after cancellation or publication failure. Qwen input slicing follows the
example's grid. Static frontend review distinguishes output-only curriculum
policy from accepted request fields. Official-weight Anima GPU execution and
combined quality remain open; this does not close P5 or promote a default.

Final CPU initialization/provenance/job/export suite:119 passed,95 warnings;
changed backend files compile and real imports complete without CUDA context
initialization. Owner frontend build/type-check is not run by the agent.

Official Anima mixed-grid owner execution now passes: native512/256 preparation,
four updates of each teacher-field phase, source parameter retention, real
streaming export/native reload and fresh67-parameter gradient updates on both
grids. Own initializer peak4.576GiB is not a Full FT measurement. Export is
relocated to a private directory, not mocked; no full API/dataset resume or
quality acceptance is inferred. Evidence:
`hybrid_anima_public_multigrid_initializer_2026-10.json`.

The pixel-resize versus latent-area calibration diagnostic gives a small2.93%
held reconstruction improvement, while matched combined512/1024 generation
retains clipping/grid defects. Independent input/output fits have measurable
flow-composition error under the current Gaussian/bias algebra; a synthetic
identity fixture closes to numerical zero. Constrained map/bias initialization
is the next hypothesis to test, not a proven universal fix. Combined quality
and P5 remain open. Evidence:
`hybrid_codec_pixel_resize_calibration_2026-10.json`.

The constrained affine prototype now closes source-flow composition numerically
and passes38 CPU Gaussian/pair checks, including BF16 gradients and restoration.
Actual combined-checkpoint added IO weights are zero, so the frozen-head
assumption was checked rather than inferred. Matched fitted/orthogonal-input
generation still has clipping/grid defects; orthogonal input retains coarse
outlines but does not pass quality acceptance. This falsifies closure alone as
a sufficient remedy. It remains experimental, with no default/owner artifact
change; combined quality and P5 remain open. Evidence:
`hybrid_codec_flow_consistent_2026-10.json`.

Spatial output with pointwise input now preserves exact flow closure while
using neighboring native features.48 CPU checks pass, including a planted
spatial-recovery fixture and multiple-frame ordering. Held pixel256 round-trip
MSE improves16.64% over the constrained pointwise variant, but matched frozen
generation retains clipping/grid/color artifacts. Clean native/target moment
statistics and finite output-latent ranges are recorded; neither identifies a
sole cause or proves post-training quality. This remains an experiment, with
short resized-IO/bridge training and combined P5 acceptance still pending.
Evidence: `hybrid_codec_spatial_flow_consistent_2026-10.json`.

Matched short IO/bridge adaptation is now measured for independent-spatial and
flow-constrained-spatial calibration: six tensors/1.131M parameters,200 updates,
identical item/null/sigma/noise exposure digests, finite gradients, exact frozen
body/U-Net hashes and all optimizer counters200. Own peaks are4.154/4.156GiB;
these are not Full FT measurements. Constrained loss and reviewed held1024px
images do not outperform the independent fit. Post-fit Gaussian snapshot drift
is measurable but small (native noise std0.988–1.000), so snapshot staleness is
not established as the main cause. Combined quality remains unaccepted; no
production policy is changed. Evidence:
`hybrid_anima_spatial_codec_io_fit_2026-10.json`.

An experimental input-covariance match now preserves clean native fit moments
and exact affine flow closure while reducing pixel256 inverse norm27.92→14.71.
56 CPU checks pass. Held projected std0.38685 is close to native0.37090, but
paired native projection MSE worsens; different metrics must not be conflated.
Matched frozen1024px generation has less prominent large green/purple regions
but retains clipping and grid texture. This is not combined trained-quality
acceptance or a default change. Evidence:
`hybrid_codec_moment_flow_consistent_2026-10.json`.

The covariance-matched candidate also completes200 matched IO/bridge updates
with finite gradients, exact frozen hashes and counters200. Own peak4.154GiB
and47.016 wall seconds describe restricted fitting, not Full FT. Last50 mean
loss1.5127 improves over fitted-spatial2.2010 but remains above independent1.1510;
reviewed held1024px generation retains clipping/grid defects. Combined quality
remains open; no default or owner artifact is promoted.
