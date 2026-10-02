# Ming-Image 0.1 feature parity audit

Date: 2026-10-02. Scope: source, contracts and recorded CPU milestone evidence in
`codex/ming-image-design` through `456edc8c`, plus the CPU Adafactor follow-up below; no official weights, running API,
frontend build/type-check or actual CUDA execution. This is a completed source
and wiring audit, not official-model acceptance. The historical stage evidence
lives in [the maintained design](../guides/MING_IMAGE_01_DESIGN.md).

The comparison uses Qwen's consumers (`pipeline_backends/qwen_image_21.py`,
native pipeline/transformer, training ops/adapters and capability tables), not
just parameters present in the common API. Generic controls that Qwen's native
run does not consume do not establish a Qwen feature to copy. Ming refuses
unwired nondefault generation controls explicitly.

## Implemented surfaces and evidence

Backend paths below are relative to `backend/`. Test names are relative to
`backend/tests/`. Each row states the validated extent; no row asserts that
small CPU models prove production quality or memory use.

| Surface | Ming implementation | Evidence / limits |
|---|---|---|
| Native Design/Layer discovery and loading | `core/models/ming_image_01/{artifact,discovery,loader}.py`, model registry | Complete six-component native directory/manifest, strict profile/config/header checks, package-local classes, precision/ownership inspection; tiny package/discovery/loader tests |
| Structure and parameter graphics | `inspection.py`; shared loaded model graph; `MingGlobalSummary.tsx`; `MingConditionerTraining.tsx` | Unique parameter ownership and logical bytes separate from serialized storage; global header-only census; no VRAM estimate based on active experts |
| Text generation/edit/references | `generation.py`, `prompting.py`, `sampling.py`, pipeline mixin | Native dual caption streams, ordered RGBA references, deterministic seed, native clock/CFG; real small-model manager/sampler tests; rectangular generation batches require equal reference counts |
| Inpaint/outpaint | `pipeline_backends/ming_image_01.py` | Native Design edit plus protected-pixel RGBA composite; four-operation tests; Layer correctly refuses these tasks |
| Layer output/persistence | `results.py`, `api/ming_image_results.py`, `MingOutputGroups.tsx` | Auxiliary canvas zero followed by ordered RGBA layers; Layer remains image output, not video; primary result is layer 1 |
| Live generation preview | `generation_preview.py`, shared progress factory | Actual scheduler sigma and CFG velocity, no additional DiT forward, primary Layer selection, bounded native VAE, checkerboard JPEG display, interval/disabled demand and finally restoration; 24 affected CPU cases |
| Native Reference Guide | `reference_guide.py` | Native RGBA posterior-mode latents, isolated noise RNG, inclusive windows and actual next sigma; excludes Layer auxiliary direct blend; protected composites; no guide in partition execution |
| Style Transfer | `style_transfer.py`, `style_generation.py`; existing four-handler transport and selector | Native target-only Q/K/V after norms/RoPE, frame-zero exclusion from direct Layer statistics, stack/common-concept, windows/ranges/frequency/Value controls, conditional CFG ownership and cleanup; normal attention required; all scopes/transport tested on CPU |
| Prompt enhancement | `extensions/ming_image_01_prompt_upsampler.py`, `MingPromptUpsample.tsx` | Pinned native Design/Layer templates over LM Studio/Ollama, validation/cache/manual and automatic apply/cancel/stale-response handling; mocked providers only; no automatic external contact or invented Ming official-PE artifact |
| Generation LoRA | `lora.py`, `generation_adapters.py` | Native attention/refiner/caption target mapping, strengths/windows/multiple adapters, quantized base-function gate, topology/global ownership and restoration; ordinary LoRA capability matches Qwen |
| Full DiT / conditioner training | training ops, `training_options.py`, full adapter | Independent DiT/caption and five conditioner-group ownership/LRs, frozen VAE, both live conditioning gradients, trainable query handling; actual native CPU optimizer/update/save tests |
| Floating / quantized LoRA training | LoRA adapter, `quantization.py`, `convrot_training.py` | Frozen packed base, native target map, actual CPU updates/resume; optimized ConvRot helper dispatch/metadata/base-function gates tested with CUDA doubles, not real kernels |
| Flow / reconstruction / guidance / null objectives | `flow.py`, `training.py`, `guidance.py`, shared sigma-null policy | Native `t=1-sigma`, clean-minus-noise velocity, frame supervision, blend/stochastic/ramp/cap and frozen null teacher; independent loss/gradient oracles |
| Dataset / buckets / caches / MNT | `dataset.py`, `training_data.py`, `cache.py`, `training_buffer.py`, `cpu_prefetch.py` | Paired alpha-aware crops/roles/references, H/W/F buckets, native cache fingerprints/ownership, encoding modes, fresh trainable-conditioner graphs and logical noise; recorded CPU dataset/buffer/shared-step tests |
| Memory / precision controls | `block_offload.py`, `training_residency.py`, `checkpointing.py`, shared trainer | Native main/refiner checkpointing, bounded main-block limits, query chunks, training block swap/activation handling and optimizer/accumulation policy; CPU/double lifecycle evidence; official GPU behavior unmeasured |
| Offline DiT/conditioner quantization | `conversion.py`, `conditioner_quantization.py`, conversion CLI | Selected native components, INT8/E4M3/ConvRot, protected tied heads/norms/queries, strict shard preflight, preserved U-Net/global payload and atomic publish/cancel; 28 affected converter CPU cases |
| Full checkpoint and inference export | `checkpoint.py`, full adapter, shared export jobs/UI | Exact floating training bundle versus inference-only packed export; manual/automatic/background stopped-run export, native six-component output, cancellation/collision/source preservation; no checkpoint merge into main |
| Training resume | `training_resume.py`, full/LoRA adapters, shared trainer | Batch-boundary CPU/device RNG and run seed, GAS/pending gradients/scaler ownership, optimizer/scheduler, global branches and U-Net restored; established-state AdamW/Adafactor full/global LoRA continuation exact in CPU tests; rejected resumes preserve source checkpoint |
| Partition inference / training / global summaries | `partition*.py`, `MingPartitionControls.tsx`, shared training controls | Complete coverage/cores/halos/full coordinates, all target frames/full references/captions, fixed/threshold/automatic/manual plans, global train/save/load/LoRA consumption and structural inspection; approximate spatial partitioning is distinct from exact full-KV query chunks |
| Partition diagnostics | `partition_timing.py`, metric registry, shared UI | Host wall time; opt-in synchronized CUDA spans include null/conditional/backward; all-frame target counts and process-wide peak without reset; 12 CPU/mock cases, no CUDA measurement claim |
| Training debug / samples / progress | `training_debug.py`, `training_sampling.py`, shared BaseTrainer/monitor | Native sign/frame-aware core filmstrip, existing-null diagnostics, deferred lossless RGBA VAE previews, ordered roles/weights, native sample persistence and progress; 18 debug CPU cases with exact gradient/RNG invariance |
| Residual U-Net | `residual_unet.py`, shared rewiring editor/metadata | Main stream only, retained references/captions/refiners, no new U-Net parameters; default 30-main-layer two-stage layout outer `[4,26)`, inner `[13,17)` (deepest four), full/LoRA save/reload/resume and graphics; combined style/partition CPU paths validated |
| Public contracts and lifecycle | defaults/schema/routes/OpenAPI, four panels and shared training config | SSOT, profile/task guards, unsupported-control refusals, cancellation/error/retry, output/gallery/metadata wiring and static frontend review; no running API or owner build confirmation |

## Architecture-specific boundaries

- Ming's joint caption/target attention does not admit Qwen's reusable fixed
  prefix K/V cache. It recomputes captions/references per native forward/region;
  full-KV query chunking remains exact. No fake prefix cache is advertised.
- Native Ming negative conditioning is the zero-caption branch, not an encoded
  negative prompt. Nonempty negative prompts are refused. The native CFG formula
  and Layer auxiliary frame are retained rather than changed to Qwen semantics.
- Prompt rewriting uses Ming's upstream templates and existing local transports.
  Qwen's separately trained official PE artifact is Qwen-specific; no compatible
  Ming PE weights are assumed or automatically downloaded.
- Qwen's ordinary-LoRA-only capability/refusals remain: no ReLoRA, full-TE LoRA,
  LoHa/LoKr/DoRA, or invented compatible ControlNet. LoRA on a saved full-global
  base is refused as in Qwen's shipped boundary.
- Style K/V and direct AdaIN capture exclude reference/caption/padding tokens and
  Layer auxiliary frame zero. Added style keys are visible to joint queries;
  unchanged reference/auxiliary hidden states or pixels are not guaranteed.
  Normal attention, partition exclusion and unsupported independent guidance /
  RoPE-offset / late-release guards are explicit.
- Generation block swap, VAE swap, compiled Flex and unvalidated acceleration
  algorithms are not claimed merely because common API/UI fields exist. The
  generic nondefault runtime-option guards are retained. Training block swap is
  a separate implemented surface.

## Remaining acceptance work

No new source/wiring blocker was found after the live-preview gap was implemented
and the undefined training-UI predicate was corrected. This does not establish
production readiness. Required owner validation when resources are available:

1. Owner frontend build/type-check and interactive Design/Layer load, structure /
   parameter / U-Net / global graphics, all generation tasks, style/guide forms,
   preview interval/disabled behavior, outputs/gallery and cancellation/retry.
2. A separate, available environment with official pinned Ming weights: full
   native loading, upstream comparison, generation, one-step full/LoRA training,
   checkpoint inference/restart/resume and U-Net optional execution. Main's
   occupied API/GPU must remain untouched until the owner releases it.
3. Actual BF16/FP16/CUDA attention, training block swap/activation lifetimes,
   optimized ConvRot forward/backward/prefetch/resume and diagnostic timing;
   real peak VRAM, quality and throughput are still unmeasured.
4. Exact recovery inside an unfinished MNT/region, optimizer families beyond the validated AdamW/Adafactor cases
   and GPU optimizer/scaler traces remain unproven. The established CPU exact
   resume boundary is completed logical batch steps with saved pending gradients.

Uncalibrated U-Net image quality recovery is not claimed by structural CPU tests.
The branch has not been merged or pushed; source audit and CPU evidence must not
be represented as official-model GPU or owner UI acceptance.

## CPU-only follow-up: Adafactor continuation

Design and Layer now have additional full/LoRA native shared-loop checks using
transformers Adafactor from the actual optimizer factory, including saved single-stage
U-Net, spatial partitions, global branches, query chunks, guidance and MNT2/GAS4.
The checkpoint is taken at logical step 6 after an optimizer update, with two
pending accumulated backwards. Tests require the serialized optimizer class,
nonempty established state and row/column factored statistics before accepting
continuation evidence. Data/noise/sigma/conditioning/region predictions, all
model weights, tensor and scalar optimizer states, update count and scheduler
are compared exactly against uninterrupted CPU execution. This does not cover
unfinished regions/MNT, CUDA optimizers or scaler trajectories.

Milestone verification: all four new Adafactor cases and four affected existing
AdamW full/global-LoRA cases pass (eight cases). The changed test module compiles
and imports with CUDA initialization forbidden. Owner build/UI and official GPU
acceptance remain pending.


## CPU-only follow-up: prompt UI request ownership

Static review found a delayed provider-default request could overwrite a newer
provider choice or settings edit, and multiple generation panels shared one
model datalist ID. The prompt component now invalidates stale initial/default
responses on settings changes and unmount, aborts a pending rewrite immediately
on shared/local settings updates, clears old provider model candidates, and uses
React `useId` for each panel's model list. Review covers out-of-order provider
responses, cross-panel settings events, edits during initialization, unmount and
React StrictMode cleanup, failed defaults, and the existing preview cancellation
and model-refresh revision guards. This is static evidence only: no frontend
build/type-check, rendered browser verification, live API or local provider call
was performed.

At this check, local `main` is an ancestor of the feature branch (`git merge-base
--is-ancestor main HEAD` succeeds), so committed histories do not currently
diverge. This does not inspect or authorize overwriting main's working changes,
and the feature branch has not been merged or pushed.


## Frontend review follow-up

The owner requested a deeper static frontend review. The resulting
[frontend review and owner checklist](MING_IMAGE_01_FRONTEND_REVIEW.md) records
additional queue/loop/transfer/profile/output findings and fixes. This supersedes
any earlier suggestion that no frontend wiring gap could remain. Compiler,
browser and official-model GPU acceptance are still separate pending evidence.
