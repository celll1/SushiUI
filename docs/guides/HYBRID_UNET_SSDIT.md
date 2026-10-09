# Hybrid U-Net ssDiT foundation

Status: experimental native Anima/Ming/Qwen 2.1 integration; CPU contracts and
actual Anima full-training/save/resume/generation verified. Quality acceptance is pending.
The owner permits incompatibility with the previous residual U-Net system.
Existing runs are not silently converted. This guide records only implemented
behavior; the broader implementation proposal remains under `local/strategy/`.

## Graph and modules

`backend/core/rewiring/hybrid_spec.py` owns schema version 2 with method
`hybrid_unet_ssdit`. It carries explicit stream widths, learned boundary
declarations, local refinement points and compressed skip edges. Base native
configuration and block-layout hashes remain part of its identity. A separate
module-layout hash covers every learned module's shape and placement.

The initial graph has fixed-width native blocks. Learned down/up boundaries
currently require isotropic 2x reduction. Local modules use channel-wise
LayerNorm, depthwise spatial convolution and bottleneck channel mixing. Skip
sources are captured after a declared block and restored before a later block
on the same region grid. Prefix tokens are excluded from all spatial modules.

`hybrid_modules.py` constructs an ordinary registered `nn.Module` tree, with
stable IDs inside down/up/local/skip ModuleDicts. Skip activations are compressed
to the declared rank and owned by one forward execution, without detaching the
encoder graph or retaining activations in module state across calls.

Learned convolutions and both skip projections use non-reentrant checkpointing
when native checkpointing is enabled. Checkpoint calls wrap only pure tensor
modules, never the mutable skip ownership dictionary. Anima CPU/async block
offload modes also save these checkpoint inputs on CPU; new-module transfers
currently use Torch's CPU saved-tensor path, not the native async engine. The
global activation dispatcher can cover the ordinary checkpoint path. Live skip
outputs and residual bypasses still occupy their execution device; checkpointing
does not remove those explicit graph references.

Training Block Swap is supported for Anima full-frame execution without EMA.
Ming/Qwen hybrid training still requires zero swapped blocks. The earlier Anima
failure after five fused updates was quarantined without overwriting consistent
state. Its fix coordinates native checkpoint coverage, internal recomputation
and leaf update completion, as detailed below. Activation offload remains available.

Learned correction outputs start at zero. Initial boundary execution matches
the legacy pool/interpolation, allowing a controlled starting point and an
independent comparison. This initialization does not restrict learned outputs
to legacy behavior. Upsample corrections have no bias or normalization and
preserve a zero update. The existing high-resolution region bypass remains.

The shared residual runtime has optional extension hooks at down/up boundaries
and before/after blocks. Calls without extensions retain legacy execution.
Spatial modules process each ordered Ming layer grid independently, with no
convolution across layer boundaries. Native attention still connects the
declared token streams. Architecture-owned callbacks own native blocks and positions.

## Artifact and inspection boundary

Native Anima, Ming and Qwen inspection returns a separate `hybrid_default_spec`.
The training editor offers an explicit experimental preset button; inherited
graphs and existing defaults remain unchanged. Preset rank, kernel and placement
intervals come from `DIT_HYBRID_DEFAULTS` in `api/param_defaults.py`. Mirror skips
are included only when both endpoints occupy the same nested region grid.
`POST /models/rewiring/hybrid/build` canonicalizes explicit module edits and
recomputes their layout hash. It validates a declaration, not checkpoint identity;
the native loader still checks dimensions and configuration. The training editor
provides a separate draft for region intervals, nested regions, boundary ranks
and kernels, local refiner placements and compressed skip endpoints/ranks.
Validation and application are explicit; cancellation, model changes and late
API responses cannot replace a newer draft. Unapplied or invalid drafts block
run/preset saving, and hybrid graphs require full-parameter training. Inherited
checkpoint graphs remain read-only. Frontend build/type checks and official-weight
GPU acceptance remain pending.

Native artifacts use the existing `sushi.rewire.*` keys with schema version 2.
Standalone module tools also support `sushi.hybrid.spec` and its hash. Their
read/write functions validate the topology and module layout. Module
weights use strict PyTorch state loading; declaration metadata alone is not a
complete checkpoint. Compressed skip storage is estimated from rank, token
count, batch size and element size, excluding autograd's internal saved tensors.
API preview also returns `hybrid_activation_layout`: retained region bypasses,
low inputs and compressed skips, their before/after block capture points and
release points, and a per-stream peak element count. Counts are per image grid
and batch item. The graph provides a collapsible BF16/FP32 capacity table and
skip tooltips. Multiply by batch and ordered layer count. This is logical
forward payload, excluding autograd, aliased storage, prefix tokens, temporaries,
weights and optimizer; it is not an estimate of total peak VRAM. Native runtime
ownership was independently instrumented for nested regions, adjacent region
release/acquisition, prefix exclusion, batch scaling and grouped layers.
For the Anima hybrid preset at 1024px, the API reports 26.53MiB BF16 retained
payload on a 64×64 native token grid and 6.63MiB on a 32×32 Qwen codec grid,
per image/batch item. Both peak after block 12. The CPU hybrid suites passed
44 tests after adding these ownership checks. Frontend contracts, optional-field
handling, legacy fallback, units and per-stream filtering were statically
reviewed; owner build/type checks are still pending.

Anima installs registered modules during model construction, before native
weights load. Its native forward executes them and checkpoint loading rejects
missing/unexpected or nonfinite hybrid parameters. Full-model state saving
includes these modules. Topology reinstallation preserves parameter objects;
codec IO rebasing preserves module declarations. OpenAPI and frontend types
recognize v2. Explicit topology JSON can select it for Anima full-parameter
training, including partition/global-summary execution. Ming uses header-validated
meta construction and shard assignment for hybrid bundles, including packed
native Linear layers; learned convolution weights remain floating. Qwen installs
modules before loading split artifacts and training bundles. Floating Diffusers
directories use header-validated sequential shard assignment; packed native
bases use the split artifact loader. All three native
integrations preserve hybrid declarations when codec IO is resized. Frozen-base training
remains rejected until its artifact contract is implemented.

`hybrid_partition.py` propagates conservative spatial dependency radii through
pooling, interpolation, learned boundary kernels, local kernels and skip edges.
The native planners take the larger of configured and required context, round
it to their partition planner's even unit, and align the expanded real input
region to cumulative U-Net factors. Loss and stitched output use only the
non-overlapping cores. Installed codec bridges add their serial convolution
context, accounting for packing and parallel Gaussian complements. Preview
responses report hybrid-only radii in base token cells; the planner adds codec
context when the actual model is available.
This bounds local-path padding effects; global attention can propagate boundary
differences, so a finite halo does not prove whole-model equivalence with
full-frame attention. Large halos can erase the memory/time benefit of splitting.

The graph shows compressed learned skips with dashed yellow lines and spatial
modules with pink squares. Preview parameter counts cover each registered
boundary, local refiner and skip, independently of native base weights. The
UI adds those counts to its native census; pending hybrid counts are not shown
as zero. Anima, Ming and Qwen have native configuration parameter censuses,
without loading tensors or allocating models. Anima's 2,091,068,928-parameter
base count includes its internal LLM adapter and excludes rotary buffers,
external encoders, VAE and additional spatial/skip modules. Inspection and
loaded-model summaries expose the same census; resized codec IO is counted from
the current native geometry. Other unavailable censuses remain explicit. Hybrid
regions and dependent modules are edited through the explicit validated draft
described above. No existing
run or default preset is converted. Broader native census coverage, full activation-memory
profiling, GPU/native full-trainer
optimizer/EMA/offload acceptance and
LoRA support still require implementation or verification.

## Verification

`backend/tests/hybrid_rewiring_test.py` checks metadata tampering, invalid graph
edges and shapes, nested non-square grids with prefix tokens, zero-init
equivalence in FP32/BF16, gradients through compressed skips and all correction
parameters after training updates, strict module state restoration, optimizer
state loading and compressed activation estimates. These are CPU functional
checks, not image-quality or GPU speed acceptance. `hybrid_anima_test.py`
additionally runs native Anima forward/backward and model updates, serializes
a real safetensors artifact, loads it through the native loader, checks exact
saved-weight restoration and optimizer-state restoration, compares outputs
within FP32 tolerance, and rejects a checkpoint missing learned-module weights.
Bitwise optimizer continuation is not promised across changed tensor layouts
or attention execution kernels.

`hybrid_partition_test.py` compares expanded partition cores with an independent
full-grid execution of pointwise native-block surrogates and nonzero learned
convolutions/skips, and checks actual planner context/coordinate bounds.
The native Anima test compares core-weighted backward with full-frame backward
when the required halo covers the full small test canvas, including the global
adapter and hybrid module gradients. It uses supplied conditioning with a
text-preprocessing stub; it does not verify the caption encoder. These CPU
checks do not establish large-image partition quality or GPU memory savings.

The training-versus-generation behavior observed in the previous Anima
VAE-swap run still needs independent diagnosis. Good single-step debug
reconstruction is not evidence of good free-running generation.

`hybrid_ming_test.py` exercises Design and Layer native forward/backward,
learned branch gradients, independent layer-grid convolution, optimizer
ownership, VAE IO rebasing and strict sharded package restoration. Floating
and int8 native bases retain their exact saved state; full training of packed
bases still requires the existing explicit dequantization option. Missing,
wrong-shaped and nonfinite learned weights are rejected. Native outputs before
and after reload are compared within FP32 tolerance; bitwise output identity
across restored tensor layouts is not promised.

`hybrid_qwen_test.py` checks native branch gradients, full-adapter optimizer
ownership and optimizer-state restoration, single/sharded checkpoint reload,
training-bundle construction and strict Diffusers-directory shard loading.
Partition/global-summary forwards retain full-canvas coordinates after nested
downsampling. Qwen hybrid remains uncached and single-target; legacy topology
keeps its existing partition restrictions. These tests do not validate the
caption encoder or official-weight GPU training.

## Requested Anima performance matrix

2026-10-09 Anima base v1.0 GPU functional probe: BF16, 512px, batch 1,
AdamW8bit, native VAE geometry and the new hybrid preset. Three updates completed
with ordinary and CPU-offloaded checkpointing. After the zero-output branches'
first update, all 65 learned parameter tensors had finite nonzero gradients;
every final correction projection changed. Peak allocated VRAM was 11.47GiB.
Synthetic conditioning/latents were used, without VAE/TE execution or checkpoint
I/O. This proves native backward and module updates, not quality recovery or the
requested four-way timing comparison. CPU FP32/BF16 checkpoint equivalence and
the three native architecture suites passed 40 tests.

The four-way matrix was measured on 2026-10-09 using Anima base v1.0, BF16,
AdamW8bit, 24 checkpointed native blocks, 1024px and batch 1/2/4 on RTX 6000 Ada.
The hybrid preset has two downsampling regions, four deepest blocks, rank 64,
3×3 kernels, seven local refiners and seven compressed skips. Added modules
are checkpointed. All cases use synthetic exposures and 64 conditioning tokens.
One warmup update precedes three measured updates per batch; the table reports
medians. Forward includes loss evaluation; optimizer timing includes its GPU
work, and step time is synchronized wall time. These are compute measurements,
not end-to-end run throughput or a quality comparison.

| Batch | Hybrid | Qwen VAE swap | Forward ms | Backward ms | Optimizer ms | Step ms | Images/sec | Peak allocated GiB | Peak reserved GiB |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | No | No | 250.8 | 957.0 | 85.6 | 1297 | 0.77 | 11.96 | 12.10 |
| 1 | Yes | No | 162.9 | 374.7 | 87.9 | 625 | 1.60 | 12.04 | 12.12 |
| 1 | No | Yes | 110.9 | 234.9 | 73.1 | 422 | 2.37 | 11.46 | 11.56 |
| 1 | Yes | Yes | 101.9 | 218.4 | 67.4 | 391 | 2.56 | 11.48 | 11.59 |
| 2 | No | No | 539.3 | 2026.6 | 84.9 | 2641 | 0.76 | 14.25 | 14.99 |
| 2 | Yes | No | 252.1 | 741.6 | 90.7 | 1078 | 1.86 | 14.09 | 14.48 |
| 2 | No | Yes | 130.9 | 320.7 | 88.1 | 547 | 3.66 | 11.60 | 11.73 |
| 2 | Yes | Yes | 117.1 | 211.7 | 76.5 | 406 | 4.93 | 11.66 | 11.77 |
| 4 | No | No | 1218.6 | 4407.2 | 84.6 | 5703 | 0.70 | 20.71 | 22.14 |
| 4 | Yes | No | 538.2 | 1636.8 | 89.4 | 2266 | 1.77 | 20.37 | 21.07 |
| 4 | No | Yes | 225.4 | 773.2 | 93.1 | 1094 | 3.66 | 11.98 | 12.08 |
| 4 | Yes | Yes | 148.6 | 276.1 | 88.8 | 516 | 7.75 | 12.06 | 12.13 |

Native geometry has 4096 image tokens; Qwen's 64ch/16× codec has 1024. Native
DiT weights remain resident. Model/gradient/optimizer storage therefore explains
much of the approximately 11GiB floor; hybrid downsampling alone does not reduce
weight count. The hybrid adds 4,958,208 parameters (about 0.24%). At batch 4 the
combined condition is 11.05× faster than native and 2.12× faster than swap alone;
at batch 1 its extra benefit over swap alone is only 1.08×. Peak allocated memory
falls by about 42% at batch 4 through the codec's lower token count. Hybrid alone
barely changes peak memory in these checkpointed conditions.

Swap cases reuse the previously fitted 40 rating:general images' paired-spatial
kernels from the native-teacher-grid experiment. Ratings were rechecked against
the dataset API without reading image pixels. Gaussian row-orthogonal noise
coefficients were captured from those fitted kernels using the current production
implementation. Both trainable bridge directions and frozen noise corrections
execute in the measured forward. This does not omit alignment as the old matrix
did. No real-image recalibration was performed.

VAE encode/decode, caption encoding, disk I/O, spatial partition/global adapter
and EMA are excluded. Model/bridge setup took 9.6–10.9s in separate workers;
encoding preparation was not measured. GPU-process peaks exclude the backend's
independent CUDA context and desktop use. Raw per-update data, graph declarations
and calibration digest are tracked in
`docs/audits/data/hybrid_anima_matrix_2026-10.json`; the executable measurement
script remains under `local/verification/hybrid_anima_matrix.py`. Re-measure if
the graph or execution policy changes during remaining quality acceptance.

## Same-input training/inference diagnostic

Anima base v1.0 was also compared in training and evaluation modes on the same
512px synthetic noisy inputs, with native attention, BF16 and 24 checkpointed
blocks. Standard, hybrid, calibrated codec swap and combined paths were checked
at sigma 0.0625/0.5/0.9375 and CFG 1/4. Two synthetic 64-token conditioning
streams were used. Hybrid-only and combined paths additionally used nonzero
final spatial/skip projections (synthetic weights, not a quality-trained model).
All 30 comparisons had exactly zero output difference. The four-way compute
matrix therefore has no detected mode-switch discrepancy under this diagnostic.

This is a conditional same-input test: both modes receive the same exposure
noise. It does not establish equivalence between random training exposures and
the sampler's evolving trajectory, nor explain the old #174 sample degradation.
The trained checkpoint, caption tokenization, partition/global adapter, decoder
and full multi-step sampling are excluded. Raw records are in
`docs/audits/data/hybrid_anima_mode_parity_2026-10.json`; executable probe is
`local/verification/hybrid_anima_mode_parity.py`. Existing runs were not changed.

## Actual Anima training lifecycle

A temporary general-only eight-image dataset exercised full-parameter training
at 512px, batch 1, AdamW8bit, hybrid preset and Qwen 2.1 codec swap. Two-way
partition training, rank-64 global summaries, CPU EMA and activation dispatch
were enabled. The unified adapter-training flag kept the LLM adapter frozen.
Two updates completed with all 18 final spatial/skip projections nonzero. The
checkpoint and optimizer contained all 65 hybrid tensors; EMA owned the same
65 tensors among 641 shadow parameters. Periodic generation and debug decode
also completed.

Resume restored 641 optimizer entries with none fresh, and all 641 EMA entries.
All 65 hybrid optimizer counters advanced to step 3. Loading the saved model
through `/models/load` restored its schema-2 declaration and bundled swapped VAE;
an eight-step, CFG-1 generation completed. These three updates verify lifecycle
ownership, not image quality recovery. The initial Training Block Swap probe failed;
the later verification below covers its corrected path. Anima global summaries without
partition training now return 422 before config is written, rather than 500.

The temporary run, dataset, checkpoints and generated image were removed, and
the prior Ming Layer model was restored. Evidence is tracked in
`docs/audits/data/hybrid_anima_lifecycle_2026-10.json`; full probe logs remain
under `local/verification/hybrid_anima_lifecycle/logs/`.

## Mutable Block Swap integration

The common sequential and branch-qualified conductors retain the original
forward's leaf gradient nodes before weight device rebinding. They wait until
every parameter participating in the current backward has accumulated its gradient
and the fused/group optimizer has cleared it. A missing gradient alone is not an
update-completion signal. Unused parameters are excluded by Torch's graph-task
query, also used by its multi-gradient hook implementation; gradient tensors are
not retained by this bookkeeping. Speculative prefetch waits while all slots are
active. Frozen branch bases also wait for the resident adapters' updates.

Anima's checkpoint lives inside each block and replays an internal function.
It therefore explicitly reacquires mutable weights at that function's replay
entry; module pre-hooks alone cannot cover it. Swapped tail blocks are always
checkpointed, in addition to the configured native prefix. For example, prefix
24 plus two swapped blocks checkpoints 26 of the 28 blocks. CPU/async checkpoint
mode is preserved. Partition training and EMA with Anima Block Swap are refused
at configuration time.

GPU regressions compare exact resident/swapped SGD updates with differentiable
and frozen inputs, external/internal non-reentrant checkpoints, and alternating
full/frozen-base adapter branches. The real Anima probe uses BF16, AdamW8bit,
512px, hybrid preset, Qwen 2.1 codec, two swapped blocks and two slots. Training
completed two updates, VAE-only debug decode, save and periodic generation. Resume
restored all 636 optimizer entries with none fresh; all counters reached step 3,
including 65 hybrid and 40 swapped-tail entries. All 18 final spatial/skip
projections became nonzero. This is functionality evidence, not quality recovery.
An additional resume changed the execution policy to eight swapped blocks
and two slots (28 native blocks checkpointed), reaching step 4 with all 636
optimizer counters advanced, including 160 tail entries. Resume consumed the
regular step-2 and step-3 checkpoints.
Records are in `docs/audits/data/hybrid_anima_blockswap_2026-10.json`.
