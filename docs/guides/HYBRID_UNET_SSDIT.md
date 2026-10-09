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
global activation dispatcher can cover the ordinary checkpoint path. Residual
bypasses still occupy their execution device; checkpointing does not remove
those explicit graph references.

Each compressed skip optionally declares training `storage: gpu | cpu | recompute`.
The schema's `hybrid_defaults.skip_storage` resolves omitted policies; old
declarations retain their exact metadata/hash. GPU remains the default. CPU
storage moves the compressed output synchronously to host without detaching
its graph and restores it on the destination's execution device. The restore
transfer runs inside the projection checkpoint so backward replay does not
keep another live GPU copy. Inference ignores this training storage policy.
This trades transfers and host memory for the compressed tensor's forward
lifetime; region bypasses, low inputs and autograd-saved tensors are unaffected.
Native activation checkpoint/offload policies remain independently necessary.
The topology hash includes an explicit storage choice, while the module-layout
hash excludes it because parameter shapes and ownership do not change. An
existing run/checkpoint is not silently rewritten. Storage is editable in the
draft, survives the canonical API builder and is displayed in the activation
lifetime table and skip tooltip. Its total capacity remains a logical payload
across devices, not a VRAM prediction.

Experimental `recompute` stores one shared stream input on CPU and checkpoints
actual encoder-prefix execution from that input through each skip's source.
The capture runs at the original after-refiner location and tensor layout;
capturing again after token merging can select different convolution kernels.
Each replay creates fresh local activation dictionaries and restores the original
CPU/CUDA RNG in a forked scope. Native callbacks retain the same conditioning,
timestep, grouped-layer geometry and partition coordinates. Encoder gradients
remain connected through the CPU input; no original compressed skip is retained.
The source must precede every skip destination in its stream, avoiding recursive
skip replays and duplicate dependency graphs. Block Swap is rejected at request,
resolved-checkpoint and Anima/Ming runtime boundaries. Inference captures skips
normally on the execution device. This policy trades repeated encoder computation
and a full-width host input for skip storage; it is not a speed optimization and
can cost more DRAM than compressed CPU storage. Region bypasses remain separate.
The preview includes one shared CPU replay-input lifetime and zero live compressed
payload for these edges; replayed element counts are reported separately.

`hybrid_skip_storage_test.py` verifies CPU/CUDA output equality, input and all
learned-parameter gradients, checkpointed and CPU-saved checkpoint inputs,
forward CPU ownership and execution-device inference. CUDA gradients are checked
within FP32 accumulation tolerance; CPU gradients use accumulation tolerances
and all outputs match exactly.
The combined CPU graph/native/partition suites passed 60 tests, and the storage
suite passed 16 CPU/CUDA cases. API checks cover CPU and unchanged legacy
roundtrips and unsupported-policy rejection. These use synthetic tensor grids;
actual full-training CPU-skip memory/speed acceptance is not yet measured.

`hybrid_skip_recompute_test.py` checks actual stochastic prefix call counts in
forward and backward, unchanged post-forward CPU/CUDA RNG, non-square grids,
grouped layers, conditioning/coordinate preservation and all input/native/spatial
gradients. Its CUDA gradient oracle disables TF32 convolution/matmul; no runtime
precision setting is changed. Native Anima (including internal CPU checkpointing),
Ming caption-prefix and Qwen causal-prefix outputs match the resident-skip path
exactly; native gradients use FP32 accumulation tolerances. Anima's partition/global
adapter core-weighted backward is also checked. The final combined CPU suites
passed 78 cases; 11 CUDA transfer/replay/inference cases passed. API validation
roundtrips replay declarations and rejects unsupported policies. Official-weight
replay throughput/peak-memory measurements are recorded below; image-quality comparisons remain pending.

## Skip storage measurement

Anima official base v1.0 was measured at 1024px, BF16, AdamW8bit, 24 native
checkpoint blocks, with all seven compressed skips using each policy. One warmup
preceded three timed updates at batch 1/2/4. VAE/TE and I/O are excluded; swap
cases reuse the same production general-only calibration as the earlier matrix.
Each cell reports synchronized seconds/update and peak allocated GiB.

| Codec | Storage | Batch 1 | Batch 2 | Batch 4 |
|---|---|---:|---:|---:|
| Native | GPU | 0.656 / 12.040 | 1.046 / 14.087 | 2.172 / 20.371 |
| Native | CPU | 0.641 / 12.030 | 1.078 / 14.084 | 2.219 / 20.365 |
| Native | Recompute | 2.219 / 12.251 | 4.422 / 13.992 | 9.656 / 20.179 |
| Qwen 2.1 swap | GPU | 0.469 / 11.476 | 0.484 / 11.658 | 0.578 / 12.055 |
| Qwen 2.1 swap | CPU | 0.469 / 11.479 | 0.484 / 11.658 | 0.578 / 12.061 |
| Qwen 2.1 swap | Recompute | 1.219 / 11.623 | 1.281 / 11.831 | 1.890 / 12.269 |

CPU skip transfers barely change total peak because compressed skips are small
and native activations/gradients/optimizer dominate. At native batch 4, encoder
replay saves about 196MiB but costs 4.45× step time. At native batch 1 and every
swapped-codec batch it increases total peak: removing a retained skip does not
remove the transient recomputed encoder graph. Post-update host RSS at native
batch 4 is 2.71/2.73/3.22GiB for GPU/CPU/replay; these are samples, not peak RSS.
GPU remains the practical default for this preset. None of these measurements
establish quality recovery. Raw timing, loss, memory and declarations are in
`docs/audits/data/hybrid_anima_skip_storage_2026-10.json`.

New hybrid module initialization now forks the construction device's RNG so
adding a topology does not advance the run's exposure generator. Initial weights
remain reproducible; checkpoint loading still restores saved tensors. CPU native
construction/reload suites passed 46 cases and a CUDA-device construction test
preserved both CPU and CUDA RNG state. The benchmark resets exposure RNG explicitly
per update, so this isolation does not change its measured inputs.

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

## Bounded real-caption quality comparison

On 2026-10-10 three temporary runs trained native Anima, legacy residual U-Net
and the new hybrid preset for 200 updates on the same 32 API-revalidated
rating:general images. Here **native means no U-Net and no codec swap**;
all three use Anima's native 16ch/8x VAE. Conditions were 512px, batch 1,
BF16, AdamW8bit, LR 5e-5 constant, normal sigma sampling (mean 0.9, std 0.3),
seed 718, 24 native checkpoint blocks and activation dispatch. The LLM adapter
was frozen; partition, global adapter, EMA and Block Swap were disabled.
Conditioning dropout was 0.1. This is an equal-update comparison, not an
equal-GPU-time comparison or the 1024px compute matrix.

Native Anima produced a recognizable person before training. Legacy and hybrid
produced identical fine-grid/no-figure step-0 PNGs, confirming that zero-output
correction initialization inherits the legacy topology's initial degradation.
At steps 100/200, native generation also developed severe fragmented/block
artifacts. Its sigma-0.6211 single-step debug prediction retained rough color
structure but lost the target's sharp figure. Legacy and hybrid samples remained
fragmented after 200 updates. Increasing CFG from 1 to 4 on all three saved
checkpoints did not establish quality recovery; hybrid CFG-4 retained a strong
fine-grid pattern. Only prompt 0 is compared at step 0; three fixed general tag
prompts were generated at steps 100 and 200, and prompt 0 was checked at CFG 4.

All 567 native/legacy and 632 hybrid optimizer entries reached step 200. The
hybrid's 65 added tensors all advanced; every final spatial/skip projection was
finite and nonzero. Final losses were 0.16454 / 0.16153 / 0.17947 respectively.
These are functionality facts, not evidence of recovered generation quality.

The inherited learning-rate/timestep setup therefore does not provide a stable
native quality control. The follow-up below isolates learning-rate sensitivity; do
not attribute all post-training failure to U-Net topology or promise that longer
training fixes it. Separate the native learning/conditioning issue from the
initial U-Net degradation before selecting a production preset or extending
this experiment. The noising/velocity sign and timestep range match the
[sd-scripts Anima full-training implementation](https://github.com/kohya-ss/sd-scripts/blob/main/anima_train.py#L486-L517);
that source comparison does not validate this optimizer/caption/sampler lifecycle.

Evidence, declarations, sample hashes and optimizer checks are tracked in
`docs/audits/data/hybrid_anima_quality_2026-10.json`. General-only sample/debug
files and logs are retained in ignored `local/verification/hybrid_anima_quality/`.
No images or model weights are committed. The used temporary runs, checkpoints
and dataset are removed after recording evidence; owner run #174 is unchanged.

### Native learning-rate control and teacher initialization

Two further native-only 200-update controls kept the same data and settings.
Enabling stochastic rounding at LR 5e-5 still produced fragmented prompt-0
samples. With rounding enabled and LR reduced to 5e-6, prompt 0 retained a
recognizable person at both steps 100 and 200. All 567 optimizer counters reached
200, and representative embedding/attention weights changed, so this was not
merely an unchanged model. Rounding consumes RNG; the comparison to rounding-off
is not an exact paired-noise experiment. These small controls identify settings
sensitivity, not a universally safe LR or long-run quality acceptance. No
production default or owner run was changed. Records are in
`docs/audits/data/hybrid_anima_native_controls_2026-10.json`.

A separate prototype freezes every native parameter and fits only the hybrid's
65 added tensors (4,958,208 parameters) to the native model's velocity output.
The same model serves as teacher with topology temporarily disabled during
no-grad calls; no second base model is resident. It uses 28 general-only fit
images and four held-out images, 512px, native VAE, batch 1, AdamW8bit with
rounding, LR 1e-3 and 300 updates. Selection/noise generators are independent
of optimizer randomness. Held-out mean MSE at sigma 0.05/0.5/0.95 falls from
0.654/0.101/0.069 to 0.185/0.031/0.019 in 93.3 seconds of fitting; peak allocated
GPU memory is 4.17 GiB. These exclude encoding, validation and sampling time.

CFG-1 generation changes from a fine grid with no figure to partial face, hair
and glasses structure, but severe fragmentation remains; CFG 4 does not repair
it. This is evidence for continuing initialization research, not a production
initialization contract or quality acceptance. The prototype and images remain
ignored local artifacts; only measurements and declarations are tracked in
`docs/audits/data/hybrid_anima_teacher_init_2026-10.json`.

Continuing from those module weights for another 2,000 updates at LR 5e-4
uses a fresh optimizer and resets the private exposure RNG; this is not a
training-resume test. Held-out mean MSE falls further to 0.138/0.022/0.007.
Fitting takes 610.3 seconds with the same 4.17 GiB allocated peak. The native
teacher PNG is unchanged, and the starting hybrid PNG exactly matches the
first probe's final PNG. Prompt 0 now produces a recognizable portrait, with
irregular contours at CFG 1 and substantially less grid noise at CFG 4.

Two held-out prompts were generated through native and hybrid models at CFG
1/4. At CFG 4, one hybrid retains the single-person silhouette, hair and
background colors but distorts facial detail. The other fails to preserve
the native two-person composition and produces a small single figure with
a dark/banded background. This does not establish general quality acceptance
or justify switching the production preset. Measurements and all eight sample
hashes are in `docs/audits/data/hybrid_anima_teacher_init_extended_2026-10.json`.
The native control run DB rows/checkpoints have been removed after archival;
their temporary general-image dataset remains while initialization experiments
use it. Owner run #174 remains unchanged.

A separate 300-update prototype adds a zero-initialized per-channel gain on
each correction's existing depthwise output, alongside its rank-64 branch.
It adds 22,528 parameters. Initial teacher and hybrid PNGs exactly match the
rank-only probe, and a CPU check confirms that a zero gain preserves a learned
correction's output while receiving finite nonzero gradients. After fitting,
held-out MSE is 0.186/0.031/0.015 versus 0.185/0.031/0.019 for rank-only; prompt
0 remains severely fragmented at CFG 1/4. This short experiment establishes no
clear generation benefit, so the branch is not added to production. Optimizer
rounding RNG differs with the extra tensors even though exposure/noise use
private unchanged generators. This does not rule out other training setups.
Evidence is in `docs/audits/data/hybrid_anima_teacher_diagonal_2026-10.json`;
its experimental module state is not loadable by the production factory.

### Shared teacher ownership

`core/rewiring/initialization.py:frozen_native_teacher` borrows a resident
hybrid model as its native teacher without duplicating base weights. Only
`rewiring_modules` parameters become trainable. Teacher calls disable topology
under `no_grad` and restore it even when forward raises. Context exit restores
every parameter flag, per-module train/eval mode, topology and CPU/device RNG;
it clears fit gradients and releases the escaped teacher's model/parameter
references. Existing training gradients, native/shared parameter aliases,
nested ownership and active Block Swap are refused before changing ownership.

Six CPU tests cover update isolation, exception/cancellation restoration,
gradient preservation and ownership refusals. A real Anima GPU check matches
native teacher output exactly, observes added-module gradients with no native
gradients, and restores flags, modes and CPU/CUDA RNG without changing fitted
weights. The caller still owns residency, fit optimizer, inputs, fresh-checkpoint
policy and provenance. This utility does not roll back partially fitted weights
and is not yet a production automatic-initialization option.

### Initialized model: bounded full fine-tuning and resume

After the 2,300-update initialization, a further prototype collects 1,568 raw
native velocity/input pairs along CFG-4 generation trajectories for the 28 fit
prompts (28 steps each, conditional and unconditional calls). Collection takes
101.6 seconds; 1,000 added-module updates at LR 3e-4 take 241.1 seconds with a
4.17 GiB allocated peak. Prompt 0's CFG-1 contours improve; held-out generation
retains more figure structure, but bands/text and facial defects remain.
Data-noising held-out MSE slightly worsens, so this is not a universal improvement.

The 3,300-update module weights were explicitly exported onto unchanged native
Anima weights for a temporary run. The API's step-0 RGB pixels exactly match
the isolated initialized model; PNG bytes differ because training adds metadata.
Full FT then completes 200 updates on the same 32 general images at 512px,
batch 1, LR 5e-6, AdamW8bit with rounding, native VAE and CFG-4 samples.
Prompt 0's face/glasses visibly improve while retaining its figure. Prompt 1's
two-person back-view composition is retained, with rough background/detail.
Prompt 2 is generated but not visually reviewed. Final loss is 0.14186.

Resume from step 200 restores all 632 named optimizer entries with none fresh
and completes step 201. All counters reach 201, including 65 hybrid entries;
all 65 added tensors are finite and the 18 final projections are nonzero.
This establishes a useful initialization/fine-tuning direction and artifact
continuation. It does not establish long-run, joint-VAE-swap, equal-total-GPU-time
quality, or a paired low-LR zero-init comparison. Automatic initialization,
provenance persistence and public API/UI wiring remain to be implemented.
Evidence is in `docs/audits/data/hybrid_anima_initialized_full_ft_2026-10.json`.

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

## Initialization lineage

`core/rewiring/provenance.py` defines `sushi.hybrid.initialization` and its
canonical SHA-256. The version-1 record identifies the native source, calibration
selection fingerprint, original hybrid declaration, seed, sample count, ordered
data-noising/teacher-trajectory phases and the initially fitted module weights.
Hashes record lineage; they are not compared against later fine-tuned weights.
No image pixels or captions are stored in this record.

Anima, Ming and Qwen's full checkpoint metadata writers and native loaders preserve
the record. Ming covers Design/Layer and floating/packed packages; Qwen covers
split artifacts, hybrid Diffusers directories and training bundles.
Codec rebasing and skip residency changes retain lineage when architecture,
native block layout and learned module layout remain the same. Malformed,
incomplete or mismatched records fail before use; shard/index records must agree.
The common frozen teacher context refuses to fit an initialized artifact again.

### Initialization job API

`POST /api/v1/models/rewiring/initialize` creates a process-local, exclusive job
from the loaded native Anima and selected dataset items. GET on the same path
reports preparation, fitting and saving progress; DELETE with its job ID requests
cooperative cancellation. Restart is refused until the job finishes cleanup.
The live native weights are borrowed, frozen and restored; the fitted modules
are published into a new standalone source-sibling artifact, never the source.
Defaults and execution-capable architectures come from `/schema/dit-rewiring`.
The request selects explicit ordered item IDs or the first IDs in a dataset;
the final holdout items are excluded from fitting. Runtime calibration has no
rating filter. Agent-operated acceptance still uses only owner-authorized general
images. Metadata stores the bounded recipe and hashes, without image paths or
caption contents; existing records without a recipe remain readable.

The public execution binding supports floating native Anima and unsharded,
floating native Qwen split packages, plus unsharded floating Ming Design/Layer DiTs.
Ming retains packed conditioner assets if present. Qwen/Ming save a sibling package directory;
`output_name` is its directory basename, while Anima requires a safetensors
filename. Existing topology, VAE swap, adapters and missing source assets are
refused before dispatch.
Training's native Anima topology editor exposes calibration dataset selection,
the API-provided recipe, progress and cooperative cancellation. It polls the
process-local job even after reopening, and leaves running jobs active when the
panel closes. A completed artifact is applied only by an explicit button after
refreshing the model index; this switches topology to inheritance and clears the
old resume checkpoint. Stale polling responses cannot overwrite action results.
UI types, paths and state transitions are statically reviewed; owner build and
interactive acceptance remain pending. Official-weight Ming GPU acceptance remains pending.
Real Anima API acceptance used three general images (two fit, one
holdout), two noising updates and one native-trajectory update at 512px. It
completed export, inspect and reload of the fitted artifact. Cancellation was
verified during source hashing and after optimizer updates; restart was refused
while active. The source model was restored and the temporary artifact removed.
The first GPU attempt exposed FP32 timesteps entering BF16 native modulation,
which disables autocast; noising now preserves the latent/timestep dtype, with
a checkpointed CPU backward regression. Successful jobs report terminal 1/1
model progress. This short test establishes functionality, not image quality.
Evidence: `docs/audits/data/hybrid_initialization_api_2026-10.json`.

The streaming publisher also accepts standalone Qwen transformer components via
`export_initialized_transformer(..., architecture="qwen_image_21")`. It retains
the original config/component metadata and native payload while adding verified
fitted modules and provenance. A tiny native Qwen CPU test performs teacher fit,
export, strict component reload and inspect; every tensor matches exactly, and
forward parity uses the existing Qwen checkpoint tolerance (1e-5/1e-6). Anima's
wrapper delegates to this common publisher and its failure/atomicity tests remain
passing. `export_initialized_qwen_package` publishes a new directory with the
initialized transformer and a verified manifest. Unchanged TE, VAE, processor,
encoder config and scheduler assets are referenced by absolute source paths;
they are not duplicated and must remain available. The package follows the
existing generation/training split loader contract. CPU tests verify manifest
resolution, strict transformer reload and declarations through the package,
existing-destination refusal, early cancellation and cancellation after the
manifest has been written; temporary directories are removed in both cases.
Qwen's execution owner encodes native RGBA latents and native text conditions
with one component on GPU at a time. It captures target-only raw velocities on
native sampler trajectories with KV caching disabled, fits the shared modules,
then publishes the package. The sampler borrows the loaded transformer and
explicitly uses its execution device while TE/VAE are offloaded; schedulers are
private copies. Original attention processors, checkpoint policy, residency,
training modes, gradient flags and Torch RNG are restored on all exits.
CPU tests exercise source-header validation for both architectures and Qwen's
packed conditioning/noising inputs with native-only teacher updates. Public API,
schema and Training panel dispatch are wired for Qwen. Real Qwen GPU acceptance
used the native original split package and three general images (two fit, one
holdout): two noising updates plus one native-trajectory update at 512px and four
teacher steps completed. The new package appeared in the model index and loaded
all components; four-step 512px txt2img generation and VAE decode succeeded.
A second job was cancelled after optimizer updates, returned `cancelled` and
left no output package. After completion/cancellation GPU usage returned to
approximately 2.1GiB including the desktop; the native source was restored, and
the temporary package was removed after restoring the owner's Ming Layer.
This short recipe is functionality evidence, not quality recovery. Evidence:
`docs/audits/data/hybrid_qwen_initialization_api_2026-10.json`.

Ming's initialized package publisher is CPU-verified for both Design and Layer.
`export_initialized_ming_package` streams an unsharded native transformer,
appending fitted modules and lineage, and copies its six native component trees
in 8MiB chunks into a new, self-contained package. Quantization provenance is
retained. Unlike Qwen's external references, Ming's manifest requires assets
inside the package; source assets are neither linked nor modified. Existing
topology, VAE swap and global-adapter sources are refused. Source changes,
early/late cancellation and existing destinations abort without partial output.
Tiny native teacher fits, exact native/module tensor restoration and forward
parity are verified for both profiles. Evidence:
`docs/audits/data/hybrid_ming_initialized_export_2026-10.json`.

Ming's public execution owner now uses native VAE encoding, native MLLM/connector
conditioning and a private scheduler/sampler. Design noising uses encoded images;
Layer first generates an auxiliary canvas and one layer from each fit image as
reference, then uses those native-generated latents for noising. The ordered
frame axis is preserved, without duplicating RGB targets as fictitious layers.
Native raw velocities are captured before CFG mixing/sign inversion; data noising
uses Ming's `time=1-sigma` and its zero-conditioning branch. TE, VAE and DiT reside
separately: Layer reference encoding finishes before the first DiT call moves
the VAE to CPU and restores the DiT. All original component weights, gradient
flags, training modes, attention/checkpoint policy and scheduler configuration
are preserved on completion, cancellation and publication failure.
Real tiny Design/Layer CPU tests execute encoding, native trajectories, fitting,
package publication and strict reload; shared API ownership tests and source
guards also pass. Schema capability and frontend package-name controls are wired
and statically reviewed. Official-weight GPU/dtype acceptance and owner frontend
build remain pending. Evidence:
`docs/audits/data/hybrid_ming_initialization_2026-10.json`.

Missing metadata remains unknown, rather than being upgraded to a success record.
The earlier prototype-only hint is not migrated automatically.

CPU tests exercise small native loaders, subsequent gradient updates and
save/reload, as well as corrupt metadata, module layout changes and shard/index
and directory-wide lineage disagreement. Missing metadata remains unknown.

`core/rewiring/fitting.py` provides the common fitting transaction. Architecture
owners supply native examples, velocity extraction, fresh optimizers, residency
and private input/noise RNG. Each declared phase starts a fresh optimizer over
exactly the added parameters and uses its declared learning rate. Data-noising
examples may call the borrowed teacher directly; trajectory examples require
captured native velocity targets. Progress counts completed updates.
Non-finite loss/gradients/weights, incomplete phases, exceptions and cancellation
roll back added weights in place. Native flags, modes, topology and Torch RNG
are restored by the teacher context. Only successful completion publishes lineage
with the actual fitted module-state digest. The main optimizer/EMA are not used.

CPU tests confirm that weights changed before cancellation and then return
exactly to their original values, with no success record. A seven-block,
width-24 native Anima CUDA probe also completes three fitting updates, preserves
all native weights and CPU/CUDA RNG, and rolls back an actual update when the
progress callback cancels. It uses synthetic inputs and verifies execution rather
than official-weight quality; recorded in
`docs/audits/data/hybrid_fitting_transaction_2026-10.json`.
Automatic preparation, API/UI recipe selection, production residency/cancellation
plumbing and official-weight GPU acceptance of this shared transaction remain
to be wired. This infrastructure alone does not initialize fresh weights or
establish image quality.

### Initialized artifact publication and inspection

`core/rewiring/initialized_export.py` publishes a standalone initialized Anima
beside a native source file. Native tensor payload is copied in bounded chunks;
only fitted module tensors are materialized on CPU. It preserves original
metadata, adds topology and initialization lineage, and verifies the complete
source-file checksum against fitting provenance. Already rewired or codec-swapped
sources are refused. Both plain and `net.` native key layouts are supported.
This initializer export requires the actual fitted module-state checksum to match;
normal training saves continue to preserve lineage after weights have changed.

Publication uses a verified temporary file and an atomic link that never
overwrites another artifact. Cancellation, source changes, checksum disagreement,
changed fitted weights or publication failure leave no partial model or temporary
file. The caller must own model exclusivity throughout fitting and publication;
this function does not acquire the application lifecycle gate itself.

`POST /models/rewiring/inspect` returns `initialization` for Anima, Ming and Qwen.
The Training topology editor shows source-checkpoint sample count, fitting update
count and seed, while explicitly distinguishing lineage from current weights.
Unknown lineage stays unknown. Model-path changes clear prior inspection state,
and cancelled/stale inspections cannot overwrite a later model selection.
Frontend types, response contract and these state transitions were reviewed
statically; owner build/type checking remains pending. The automatic initialization
job and its execution/progress/cancel UI remain unfinished.

A synthetic small Anima was fitted, exported and inspected through the restarted
real REST API: all 150 native tensors remained exact, source checksum was unchanged,
lineage round-tripped and a native source returned null lineage. The loaded Ming
Layer model stayed unchanged and CUDA was untouched. Temporary probe files were
removed. This verifies artifact/API plumbing, not official-weight quality; see
`docs/audits/data/hybrid_initialized_export_2026-10.json`.

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
