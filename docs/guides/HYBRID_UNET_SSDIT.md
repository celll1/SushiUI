# Hybrid U-Net ssDiT foundation

Status: experimental native Anima/Ming/Qwen 2.1 integration; CPU contracts and
actual Anima full-training/save/resume/generation verified. Quality acceptance is pending.
The owner permits incompatibility with the previous residual U-Net system.
Existing runs are not silently converted. This guide records only implemented
behavior; the broader implementation proposal remains under `local/strategy/`.
The requirement/evidence audit and remaining acceptance work are recorded in
`docs/audits/HYBRID_UNET_SSDIT_PLAN_AUDIT.md`.

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

`core/experiments/full_width_skip.py` is a private comparison control: it keeps
the existing compressed branch and additionally retains every source channel,
restored through a zero-initialized depthwise3x3 convolution. It adds9D weights
per edge without D-by-D mixing. Twelve CPU cases cover exact initial preservation,
all-channel information, gradients through resident/CPU/recomputed skips and
checkpointing, dtype moves and failure cleanup. Its concatenated capture includes
rank+D channels, which the public rank-only preview does not account for. Do not
export its temporary wrapper state as a public topology checkpoint or imply that
this experiment is selectable through the UI. GPU quality acceptance is separate.

The runtime also permits an internal `boundary_fusion` callback after learned
up correction and before restoring encoder tokens; absent callbacks preserve
the original path. Callback output must match the restored BCHW grid.
`core/experiments/boundary_detail.py` privately adds zero-initialized per-channel
gains on saved features minus bilinearly interpolated low-input features. This
is a retained-detail control, not a strict high-pass filter: low input includes
the learned down branch. Fifteen CPU cases cover identity, paired-input
gradients, nested/grouped grids, skip storage, checkpointing, dtype moves and
setup/runtime failure cleanup. These gains have no public topology declaration
or payload lineage; do not export the temporary module tree as a public artifact.

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
integrations preserve hybrid declarations when codec IO is resized. Hybrid LoRA
now saves complete structural weights; see Adapter artifacts for its acceptance.

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

## Final initialized Anima compute matrix

The 2026-10-10 final rerun uses the official native Anima, BF16, Flash training
attention, AdamW8bit with stochastic rounding, native checkpoint prefix 24 and
1024px synthetic exposures. Each isolated worker has a fresh model/optimizer;
each batch has one warm-up and three timed updates. Hybrid conditions load the
same retained 3300-update general-only teacher-fitted structural weights before
any codec change, check exact BF16 loading and preserve their digest on rebasing.
This is an explicitly identified prototype initialization, not a public
initializer export. Swap conditions share the same saved general calibration and
production spatial bridge/noise-alignment classes. No image encoding is repeated.

| Batch | Condition | Forward ms | Backward ms | Optimizer ms | Update s | Images/s | Allocated GiB | Reserved GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | native | 227.3 | 766.3 | 159.5 | 1.156 | 0.87 | 12.09 | 12.16 |
| 1 | hybrid | 150.3 | 298.3 | 166.5 | 0.625 | 1.60 | 12.17 | 12.23 |
| 1 | swap | 88.0 | 212.4 | 143.0 | 0.438 | 2.28 | 11.59 | 11.76 |
| 1 | hybrid + swap | 124.0 | 265.4 | 159.5 | 0.562 | 1.78 | 11.61 | 11.82 |
| 2 | native | 489.6 | 1588.2 | 154.8 | 2.234 | 0.90 | 14.37 | 15.13 |
| 2 | hybrid | 236.1 | 584.1 | 162.4 | 0.985 | 2.03 | 14.21 | 14.61 |
| 2 | swap | 111.8 | 294.0 | 154.1 | 0.562 | 3.56 | 11.73 | 11.96 |
| 2 | hybrid + swap | 136.2 | 243.7 | 165.4 | 0.547 | 3.66 | 11.78 | 11.98 |
| 4 | native | 1100.8 | 3364.6 | 156.8 | 4.625 | 0.86 | 20.84 | 22.28 |
| 4 | hybrid | 511.4 | 1308.9 | 163.5 | 1.984 | 2.02 | 20.50 | 21.20 |
| 4 | swap | 215.7 | 702.1 | 160.0 | 1.078 | 3.71 | 12.11 | 12.22 |
| 4 | hybrid + swap | 151.7 | 269.4 | 161.5 | 0.578 | 6.92 | 12.18 | 12.23 |

At batch four, combined execution is 8.00x faster than native and 1.87x faster
than swap alone, with 41.6% lower peak allocated memory than native. At batch one,
combined execution is 28.3% slower than swap alone; batch two differs by only
2.7%, too small for a robust advantage from three samples. Small token grids make
the additional spatial/skip execution significant, while the approximately
160ms full-weight optimizer update remains largely independent of token count.
The codec reduces native image tokens from 4096 to 1024. Hybrid alone preserves
native weights and adds 4,958,208 parameters, so it barely changes allocated peak.

These are compute results, not quality acceptance or end-to-end training speed.
VAE/TE/IO, partition/global adapters, EMA and Block Swap are excluded; no explicit
host-device activation offload runs in the measured interval. Input creation and
allocator clearing occur outside timing. Model/bridge setup takes 2.24-2.52s;
historical teacher fitting and VAE calibration costs are separate and were not
repeated. Process RSS is 2.29-2.88GiB; peaks exclude the independent backend and
desktop context. All workers exit; GPU memory returns to the pre-probe baseline
of about 1931MiB, with the owner's CPU-resident Ming Layer unchanged. Complete
per-update data, configuration, initialization/calibration hashes and limitations
are in `docs/audits/data/hybrid_anima_final_matrix_2026-10.json`. The executable
probe remains `local/verification/hybrid_anima_final_matrix.py` (dry-run by
default; `--arm native|hybrid|swap|hybrid_swap --execute` opts into GPU work).
The earlier zero-initialized matrix above is historical, not the final result.

## Initialized hybrid plus codec quality probe

A bounded 2026-10-10 probe reuses the earlier held-out general prompt indices
28/29 at 512px, seed 63, 28 steps and CFG 1/4. The eight native/native-codec
hybrid controls are checksum-verified and reused. Eight new outputs compare
codec swap alone against the same 3300-update initialized hybrid plus swap,
using the retained general calibration and current production Gaussian spatial
bridge/sampler/normalized codec classes. No images are re-encoded or new training
runs created. Only one of the DiT or VAE occupies the GPU during decode.

All eight outputs are finite, but joint quality acceptance fails. At CFG four,
prompt 28's native control has a clear two-person back-view composition; its
native-codec hybrid control already distorts bodies and introduces text. Both
codec-swap variants lose the two-person composition and retain strong block-like
texture and blurred partial figures. Prompt 29's combined output retains a
single face/body outline but has severe detail distortion; swap alone also has
block texture and scene fragmentation. CFG-one outputs were generated but not
visually reviewed in this audit. This is neither equal-GPU-time training evidence
nor a proof that U-Net alone causes the old owner-run failure. Native-codec
teacher fitting does not provide joint-codec initialization automatically.
No default or existing run is changed. The next quality experiment must include
bounded joint adaptation rather than infer quality from the compute matrix.
Evidence and local image paths are in
`docs/audits/data/hybrid_anima_combined_quality_2026-10.json`; generated pixels
remain under `local/verification/hybrid_anima_combined_quality/`.

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
floating native Qwen split packages, plus floating Ming Design/Layer DiTs including indexed shards.
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
interactive acceptance remain pending. Official-weight Ming Design/Layer GPU
initialization and restored generation are verified below.
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
`export_initialized_ming_package` streams a native transformer, including indexed shards,
appending fitted modules and lineage, and copies its six native component trees
in 8MiB chunks into a new, self-contained package. Quantization provenance is
retained. Unlike Qwen's external references, Ming's manifest requires assets
inside the package; source assets are neither linked nor modified. Existing
topology, VAE swap and global-adapter sources are refused. Source changes,
early/late cancellation and existing destinations abort without partial output.
Tiny native teacher fits, exact native/module tensor restoration and forward
parity are verified for both profiles. Evidence:
`docs/audits/data/hybrid_ming_initialized_export_2026-10.json`.

Official Design/Layer DiTs use five/nine shards, so Ming publication retains
their filenames and native payloads. Added modules occupy only the input-projection
shard; every shard and the index carry identical topology/lineage. The index map
and byte census include the new tensors. Sharded source identity hashes the sorted
filenames and each file SHA-256, including the index; single-file identity keeps
its original whole-file SHA-256. Export re-verifies this identity and each streamed
shard. Unindexed extras or inconsistent declarations are refused. Native root
license/notice files are copied along with the component assets. Both profiles
have exact tensor/asset and strict reload tests with single-file and sharded
sources; official-weight GPU execution remains a separate acceptance gate.

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
and statically reviewed. Owner frontend build remains pending. CPU evidence:
`docs/audits/data/hybrid_ming_initialization_2026-10.json`.

Real public Design GPU acceptance used the official five-shard BF16 DiT with
the previously verified packed conditioner, three authorized general images
(two fit, one held out), 512px, four teacher steps and three fitting updates.
The completed package reloaded through the native loader; four-step CFG=1
generation and VAE decode succeeded (gallery image 11622), with the restored
generation spec exactly matching the initialization spec. A second job was
cancelled after its first optimizer update, terminated at update two with no
error and published no package. GPU usage returned to approximately 2.2GiB;
after restoring the owner's Ming Layer and restarting the API it was 1.9GiB.
Both inventoried temporary model packages were removed and the model index
refreshed, restoring approximately 61GiB of disk space. This proves functionality
on Design, not quality recovery, Layer acceptance or paired throughput. Evidence:
`docs/audits/data/hybrid_ming_design_initialization_api_2026-10.json`.

Real public Layer GPU acceptance used the official nine-shard FP32 saved DiT,
loaded in BF16, with the verified packed conditioner. Three authorized general
images (two fit, one held out), 512px, four teacher steps and three fitting
updates completed through native reference/vision conditioning and ordered
canvas-plus-layer teacher targets. The new package reloaded strictly; a reference
generation produced and decoded `auxiliary_canvas` and `layer_1` (gallery
11623/11624). Both outputs carry exactly the initialized topology. API-reported
generation peak allocated/reserved memory was 19.214/19.451GiB; these are
generation measurements, not initializer peaks. The owner Layer model was
restored, both inventoried temporary packages removed, and the index refreshed,
restoring approximately 84GiB. Functionality is verified; quality recovery and
a separate real-GPU Layer cancellation probe are not claimed. Evidence:
`docs/audits/data/hybrid_ming_layer_initialization_api_2026-10.json`.

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
Automatic preparation, API/UI recipe selection and production residency/cancellation
are implemented by the architecture owners described above. Their real-GPU
acceptance is separate from this synthetic transaction probe. Neither establishes
recovered image quality by itself.

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
job and its execution/progress/cancel UI are wired as described above.

A synthetic small Anima was fitted, exported and inspected through the restarted
real REST API: all 150 native tensors remained exact, source checksum was unchanged,
lineage round-tripped and a native source returned null lineage. The loaded Ming
Layer model stayed unchanged and CUDA was untouched. Temporary probe files were
removed. This verifies artifact/API plumbing, not official-weight quality; see
`docs/audits/data/hybrid_initialized_export_2026-10.json`.

## Adapter artifacts

Hybrid LoRA is wired for Anima, Qwen and Ming. The existing adapter capability
tables still decide the available algebras; Anima also supports LoHa/LoKr.
`core/adapters/full_module.py` supplies the shared full-module tensor
protocol for auxiliary convolution/skip weights, separate from low-rank algebra.
It exports every parameter and persistent buffer, without a synthetic alpha;
the trainer exporter respects this per-module declaration. Loading prepares all
exact keys/shapes and finite dtype conversions before mutation, preserves live
parameter identities, and rejects changed module layouts. Nine CPU regressions
cover round-trip, optimizer ownership, full/ordinary mixed export, invalid keys,
shapes, integer/non-finite tensors and FP16 overflow with no partial mutation.
The combined existing algebra suite reports 106 passed and one unrelated
capability-census failure: its ordinary-LoRA test assumes every table row is
enabled, but the unchanged Chimera row is explicitly empty.

The shared LoRA trainer registers the resolved hybrid modules as one stable
`rewiring_modules` full-weight payload alongside the native low-rank branches.
Only the structural module parameters are unfrozen, subject to `train_unet`,
and registered once in the denoiser LR group; these layers are not low-rank targets.
Save includes their entire tensor scope, topology and initialization metadata.
Resume resolves the selected file before injection, restores its topology on a
native base, and pins that file for weight loading. Incompatible existing base
topologies are refused. Exact keys, layout, finite values and conversions are
validated before loading, and live parameter identities remain stable. Generation
now prepares full-module weights on CPU, validates exact layout and finite dtype
conversion, then installs the payload and native low-rank branches transactionally
for Anima, Qwen and Ming. `sushi.rewiring.full_module=1` declares the complete
`rewiring_modules.*` tensor namespace; undeclared or incomplete payloads are
refused. A hybrid adapter without structural weights cannot initialize a native
base. The common session owns auxiliary cleanup, including component replacement
and model teardown, preserving the original module object and initialization
lineage. Tests compare actual native forward outputs before/after install,
restore exact outputs on unload/release/replacement, and exercise real safetensors
files through all three generation loaders. A full-module payload supplies structural weights rather
than a scalable low-rank delta; its initial contract requires one owner at
strength one with no step-range modulation. Conflicting payload owners must be
refused, not silently summed or overwritten. ReLoRA/ControlNet remain unsupported.

CPU acceptance updates structural weights, checks frozen native weights, unique
optimizer/name ownership, then saves, resumes from native and inherited hybrid
bases, and reloads through each actual generation loader. NaN payloads leave all
weights unchanged. Supported Anima algebras have the same round-trip gate.
API-to-YAML serialization preserves the topology. The real shared Ming trainer
runs both Design and Layer, resumes mid-epoch from a native package with only the
adapter checkpoint naming the topology, and restores every structural optimizer
counter to step two. Continued weights match uninterrupted training at
2e-5/2e-6 tolerance. This is CPU functionality acceptance; official-weight GPU
LoRA acceptance remains separate. UI wording is updated and statically reviewed;
owner build/type checks remain pending.

Official-weight Anima GPU LoRA acceptance now completes two BF16-base updates
with rank-eight FP32 adapters and AdamW8bit, then resumes from step two with
`inherit` topology and no explicit spec. All 625 optimizer entries restore with
zero fresh entries; their counters reach step three, including all 65 structural
parameters. The saved adapter contains the complete finite structural payload.
Loading that file onto the native Anima through the real generation API produces
a 512px four-step image without warnings (gallery 11625). Periodic training
samples also succeed. The probe uses 32 revalidated general images, zero-initialized
hybrid corrections and the native VAE; it establishes functionality, not quality
recovery or joint VAE-swap acceptance. Evidence is recorded in
`docs/audits/data/hybrid_anima_lora_gpu_2026-10.json`. The temporary run's DB and
disk data are removed after retaining logs/sample evidence, and the owner's
Ming Layer model is restored. Qwen/Ming GPU LoRA acceptance remains separate.

Qwen hybrid generation also accepts partition/global-summary payloads, including
combined hybrid LoRA artifacts. Full-canvas position IDs flow through the native
partition executor and are reduced at each hybrid depth. Generation uses the
same conservative convolution/codec halo as training, expands actual neighboring
context and rounds input bounds outward to the cumulative downsampling grid.
Only each core is returned. Auto planning includes the required halo; an expanded
input exceeding its token ceiling is refused before prediction. Incompatible
canvas geometry is refused. KV caching remains disabled with hybrid topology.
References/styles remain excluded, and legacy residual topology remains
full-frame only. CPU tests compare real hybrid/global tile predictions against
explicit native forwards and exercise combined-file backend loading, successful
generation dispatch, failure cleanup and restoration of the original modules.
These tests establish CPU wiring, not official-weight Qwen GPU acceptance.

## Bounded combined-codec adaptation

The initialized hybrid + Qwen 2.1 codec candidate received 200 additional Full FT
updates at LR 5e-6, BF16, AdamW8bit with stochastic rounding, batch 1 and native
24-block checkpointing. The DiT, structural modules, enlarged latent IO and
alignment bridges were trainable; the caption bridge, TE and VAE stayed frozen.
28 revalidated general images supplied training exposures, with independently
drawn noise, logit-normal sigma and 0.1 conditioning dropout. Captions were fixed
in this isolated experiment. Four remaining images were excluded from these
updates, but the earlier initialization/calibration used the same image pool;
this is not an unseen-image generalization test.

The updates took 105.969 seconds and peaked at 11.597 GiB allocated, excluding
caption/latent preprocessing and sample generation. All losses and clipped-gradient
checks were finite. Four 512px outputs reused the prior two prompts, seed 63,
28 steps and CFG 1/4; previous outputs were reused without regeneration.
The reviewed CFG4 pair retained strong block texture: one figure became more
prominent, while the other remained distorted, without consistent quality
recovery. CFG1 outputs were generated but not visually reviewed. This bounded
probe fails quality acceptance and does not establish an equal-GPU-time comparison.
Evidence: `docs/audits/data/hybrid_anima_combined_adaptation_2026-10.json`.

The private experiment state is retained separately from owner checkpoints.
It is not a published initializer artifact or an optimizer-resumable training
checkpoint. The worker ended, the owner's model remained CPU-resident, and GPU
usage returned to the desktop/backend baseline. Reconstruction, matched-noise
single prediction and free-running generation need separate diagnosis before
extending the training budget or changing the preset.

## Codec and matched-noise diagnosis

The next two-general-image probe separates VAE-only reconstruction from one
prediction at sigma 0.1/0.5/0.9, using identical external noise and CFG4 for
swap-only, initialized hybrid+swap and the 200-update adapted candidate. Direct
target VAE reconstruction retains sharp detail. Mapping through the calibrated
bridge, discarding its extra channels to model the initial native 16-channel
head, and mapping back introduces blur/block-like detail loss without any DiT
or U-Net forward. Native latent 2x pooling followed by the calibrated output map
shows similar loss. The two inspected bridge reconstructions retain overall
figures, so this does not account for every free-running composition failure.

All six matched clean-latent prediction errors decreased after the 200 updates.
At sigma 0.5, reductions relative to initialized hybrid+swap are 23.2% and 29.6%
for the two images. The reviewed single prediction still loses facial detail,
and previous free-running outputs did not consistently improve. A small latent
MSE from a low-noise input is not evidence for generation from pure noise.
The result identifies codec bottleneck artifacts before U-Net processing and
learning progress in single prediction, without proving hybrid harmless or
quality accepted. Next isolate latent IO/bridge adaptation and trajectory
consistency while retaining the reduced target token grid.
Evidence: `docs/audits/data/hybrid_anima_codec_diagnostic_2026-10.json`.

## Restricted latent-boundary adaptation

A following 200-update probe trained only `x_embedder.proj.1`,
`final_layer.linear` and the input/output calibration projections: 1,130,624
parameters, LR 1e-4, BF16 AdamW8bit with stochastic rounding. Native DiT,
3300-update hybrid structural parameters, caption bridge, TE and VAE stayed
frozen. Hashes of all frozen parameters matched exactly before and after.
The same 28 general images and four retained adaptation-held-out images were
used. Updates took 54.063 seconds with 4.085 GiB peak allocated, excluding
encoding, hashing and sample generation. Four finite fixed-prompt outputs were
produced. Three inspected outputs still contain strong block texture and
distorted figures; reliable quality recovery is not established.

Both scope and LR differ from the earlier full-parameter probe. Item/null draws
follow the same independent seed, but stochastic rounding consumes global Torch
RNG, so subsequent noise/timestep draws differ between scopes. Equal update
counts are not equal GPU time. This probe supports the feasibility of inexpensive
boundary adaptation, not its quality acceptance, and does not justify promoting
the recipe into the production initializer. Native-codec hybrid quality and
equal-time structural comparison remain separate requirements; a failed combined
codec probe alone does not settle the architecture comparison.
Evidence: `docs/audits/data/hybrid_anima_io_adaptation_2026-10.json`.

## Equal post-initialization update budget

At 512px, native 16ch/8x VAE, batch 1, BF16 AdamW8bit SR, LR 5e-6 and native
24-block checkpointing, each arm received 120 seconds measured by CUDA events
around forward/backward/optimizer. Shared cached general exposures and dedicated
input RNG verify identical item/null/sigma selection for the common 216 updates.
Warm-up is included; encoding, artifact publication and generation are excluded.

| Arm | Updates | CUDA seconds | Median update ms | Peak allocated GiB |
|---|---:|---:|---:|---:|
| Native | 263 | 120.379 | 431.311 | 11.590 |
| Legacy residual U-Net | 277 | 120.399 | 414.810 | 11.580 |
| Initialized hybrid | 216 | 120.105 | 540.576 | 11.604 |

This is not an equal-total-GPU-time comparison: hybrid reused the 3300-update
teacher fit, whose recorded learning and trajectory-collection wall time totals
1046.156 seconds, before encoding/setup. Native and legacy had no such fit.
Initial update overhead contributes to the budget, but hybrid's median is also
slower at this small shape. Do not generalize the previously measured 1024px /
larger-batch speedup to 512px batch 1.

Twelve fixed-seed outputs were generated (two prompts, CFG1/4 per arm); eight
existing native/initialized-hybrid controls were reused. All six CFG4 outputs
were inspected. Native retains coherent figures and the two-person composition.
Legacy retains figures with substantial distortion. Hybrid preserves a selected
single face/body, with cleaner selected lines than legacy, but the two-person
composition and bodies still fail. No consistent quality advantage is established,
and no production default is promoted. The next initialization experiment should
predict the native teacher on states reached by the student, rather than only
native-teacher or data-noising trajectories.
Evidence: `docs/audits/data/hybrid_anima_equal_time_2026-10.json`.

## Student-state trajectory fitting probe

Three further rounds each sampled eight student rollouts (28 steps, CFG4),
captured 448 conditional/unconditional states and queried the frozen native DiT
on those exact states. Each round then fitted only the 65 structural parameters
for 100 updates at LR 3e-4, AdamW8bit SR, refreshing the student-state pool for
the next round. Original DiT and caption-bridge parameter hashes matched exactly
before and after. This general-caption-only probe opened no source image pixels.
Collection took 209.595 seconds and fitting took 78.296 seconds, excluding encoding,
hashing and final generation. Optimizer counters all reached 300.

Four fixed-seed outputs were generated and the CFG4 pair was reviewed against
the existing 3300-update controls. Figures still appear, but composition/body
distortion persists and diffuse/fractured texture increases. Finite teacher-loss
values do not establish improvement: this recipe is not accepted and is not
added to the public initializer or its defaults. The next controlled experiment
should vary high-resolution stage allocation before adding more fitting recipes.
Evidence: `docs/audits/data/hybrid_anima_student_trajectory_2026-10.json`.

## Higher-resolution layer allocation

A one-factor private candidate moved Anima's outer region from blocks [4,24)
to [6,22), keeping the inner [12,16) region. Full/half/quarter-resolution block
counts changed from 8/16/4 to 12/12/4. Module IDs, widths, ranks, kernels,
refiner positions and skip connections stayed fixed. The module layout and tensor
shapes are identical, but execution placement differs: both arms explicitly
received the same retained 3300-update weights, followed by 200 identical
data-noising teacher exposures at LR 1e-4. This is a declared private transfer,
not an implicit owner-checkpoint change or a fresh per-layout initializer test.

Frozen native parameter hashes match before/after for both arms. Fitting took
68.000 / 75.734 seconds and peaked at 4.440 / 4.461 GiB, respectively; these
figures include teacher prediction and structural-only fitting, not full FT.
Eight fixed-seed outputs were generated. The reviewed CFG4 higher-resolution
candidate retains more coherent two-figure/body placement than the fitted
original-layout arm, and some cleaner single-figure outlines. Anatomy/pose
distortion and spurious lettering remain. The inspected higher-resolution CFG1
two-person scene still fragments. This is a positive selected CFG4 trend, not
production acceptance or evidence of a general advantage. Keep the current
default unchanged while examining this allocation tradeoff.
Evidence: `docs/audits/data/hybrid_anima_highres_allocation_2026-10.json`.

### Sixteen full-resolution blocks

A further private placement moved the outer region to [8,20), retaining the
inner [12,16) region: 16/8/4 full/half/quarter-resolution blocks. It reused the
same 3300-update structural weights and matched all 200 subsequent teacher
exposures to the 12/12/4 arm; frozen native parameter hashes remain exact.
The existing 12/12/4 outputs were reused without retraining. Fitting took
68.953 seconds with 4.476 GiB peak allocated VRAM. Separate-run fitting timings
do not establish a throughput ranking, and this placement has not been measured
in the earlier 1024px compute matrix.

In the reviewed CFG4 pair, the two-figure scene has clearer body placement and
the single figure has a clearer face/outline than the 12/12/4 comparator.
Anatomy/pose distortion and spurious lettering remain; the inspected CFG1
two-person scene is heavily fragmented. This supports a stage-allocation effect,
not broad quality acceptance. The default and owner runs remain unchanged.
Further work should separate guided-field error from resolution allocation
before spending more updates on the same teacher-fitting objective.
Evidence: `docs/audits/data/hybrid_anima_highres16_2026-10.json`.

### Conditional/null field diagnostic

A no-fit probe compared the retained 12/12/4 and 16/8/4 candidates with the
same native teacher on four held-out general images, five noise levels and
fixed noise. Both arms retained exact frozen parameter hashes. At sigma about
0.1, mean conditional relative MSE decreases from 5.51% to 4.20%, and CFG4
relative MSE from 6.96% to 5.88%. At sigma about 0.9, CFG4 relative MSE instead
increases from 5.31% to 7.34%. These are mean per-image field-energy ratios,
not image quality scores or free-running trajectory measurements.

Null prediction error is not uniquely dominant. Conditional/null difference
error remains and is amplified by guidance, while low-noise errors share a
substantial component. The recorded CFG error obeys its expected algebraic
identity; this is not evidence of a CFG implementation bug. A controlled paired
conditional/null anchor versus guided-field fitting comparison is the next
candidate experiment. No public objective or default has changed.
Evidence: `docs/audits/data/hybrid_anima_guidance_diagnostic_2026-10.json`.

### Paired conditional/null fitting

Two private 200-update arms started from the identical retained high16 state.
Both predicted conditional and null fields on each identical noisy exposure;
one minimized their mean MSE, and the other added half the CFG4-field MSE
divided by 25. Native parameters remained exact and all 65 optimizer counters
reached 200. Held-out field errors decreased in both arms. The added guided
term lowered selected low/mid-noise guided errors slightly but did not improve
the high-noise case or show a clear visual advantage in the inspected CFG4 pair.
Do not add this term to public initialization defaults on this evidence.

Both retain two-figure outlines and a single figure/face at CFG4, with remaining
anatomy/pose distortion and spurious lettering. The paired-anchor outputs are a
candidate coarse-structure-preserving starting point for owner evaluation, not
an exact native-output reproduction. Their improvement over the previous state
also includes 200 additional updates; this probe does not isolate pairing against
an ordinary single-condition continuation arm. Fit wall times were 204.968 and
128.703 seconds, with peaks 4.953/4.955 GiB. Unequal separate-run wall times do
not establish an objective-specific speed advantage.

Review of the reused native p28 CFG1 control shows split/collage-like composition
already without rewiring. Earlier CFG1 fragmentation observations are valid
descriptions of those hybrid images, but cannot all be attributed to U-Net
conversion. The native p29 CFG1 control is coherent: this is prompt-specific.
Future acceptance should compare each candidate against its actual native
control, prioritizing usable coarse structure for subsequent training.
Evidence: `docs/audits/data/hybrid_anima_guided_fit_2026-10.json`.

`core/experiments/conditioning_alignment.py` supplies a private conditional-minus-
null alignment control. It normalizes each example's difference error by teacher
difference energy, with an explicit floor relative to the mean teacher field
energy. A paired field anchor separately constrains their common component.
Teacher tensors are detached; both student branches retain gradients. Six CPU
cases cover the analytic error, common-offset invariance of the contrast term,
independent example normalization, BF16 accumulation, zero differences and bad
contracts. This objective has no public initializer recipe or artifact lineage;
it is not an API/UI default or a quality claim.

### Compute matrix for the high16 candidate

The retained paired-anchor high16 candidate was remeasured at 1024px, BF16,
AdamW8bit SR, Flash attention and 24 native checkpoint blocks. Each arm used
a fresh worker/model/optimizer, one warmup and five timed updates per batch.
The four arms all ran again; old baseline timings were not substituted. Entries
below are median seconds/update / peak allocated GiB.

| Configuration | Batch 1 | Batch 2 | Batch 4 |
|---|---:|---:|---:|
| Native Anima | 1.187 / 12.09 | 2.281 / 14.37 | 4.765 / 20.84 |
| High16 hybrid | 0.828 / 12.14 | 1.546 / 14.49 | 3.094 / 21.06 |
| Qwen codec swap | 0.516 / 11.59 | 0.578 / 11.73 | 1.109 / 12.11 |
| High16 hybrid + swap | 0.531 / 11.61 | 0.578 / 11.77 | 0.781 / 12.16 |

High16 alone is about 1.43–1.54 times as fast as native, but does not reduce
Full FT peak allocated VRAM here. Combined batch 4 is 6.10 times as fast as
native and 1.42 times as fast as swap alone, with 41.7% less allocated VRAM
than native. At batch 1 combined is slightly slower than swap; batch 2 median
times are equal. Five short measurements do not establish small differences.
Preserving more full-resolution blocks costs speed relative to the previous
8/16/4 hybrid; do not reuse its larger speedup for this quality candidate.

The report retains forward/backward/optimizer CUDA timings, all measurements,
reserved VRAM, process RSS and input/bridge/module hashes. This uses synthetic
exposures and the production calibrated bridge, not a newly accepted combined
generation checkpoint. VAE/TE encoding, IO and global/partition/EMA overhead
are excluded. Formal publication and combined-codec quality remain separate.
Evidence: `docs/audits/data/hybrid_anima_candidate_matrix_2026-10.json`.

### High16 combined-codec generation

Four new outputs combined the retained 3700-update paired-anchor high16 state
with the existing production Gaussian-policy Qwen codec calibration. Native,
native-codec candidate and swap-only controls were reused with verified hashes.
No additional training, owner-run change or calibration-image processing occurred.
The reviewed CFG4 pair still loses composition: p28 retains only a partial,
enlarged figure and p29 an enlarged distorted head/body, both with strong block
texture. The reviewed swap-only controls also have severe texture/composition
loss. Increasing high-resolution blocks does not resolve this codec transition.

Keep the native-codec outline-preserving candidate separate from the unaccepted
combined initialization. The combined compute benefit is measured, but it does
not prove an adequate coarse-structure starting point. The earlier reconstruction
diagnostic establishes losses before DiT processing; neither record assigns all
remaining failure to one subsystem. Do not extend structural-only fitting to
address codec loss without a separate codec/IO initialization intervention.
Evidence: `docs/audits/data/hybrid_anima_candidate_combined_quality_2026-10.json`.

### Equal initialization-plus-training compute budget

A fresh 16/8/4 hybrid used no retained module weights. Its declaration alone
was reused; new modules were seeded at installation. Native and legacy/high16
each spent the full 240-second CUDA-event budget on Full FT. Hybrid spent
120.055 seconds on 297 structural-only native-teacher updates (LR 1e-3), then
120.398 seconds on 211 Full FT updates (LR 5e-6). Native parameters stayed
exact during initialization and all 65 structural optimizer counters advanced.
All arms used native codec, 512px batch 1, BF16, AdamW8bit SR, 24 native
checkpoint blocks, the same 28 general training images and four held-out images.

| Arm | Initializer updates | Full FT updates | Total compute seconds | Full FT peak allocated GiB |
|---|---:|---:|---:|---:|
| Native | 0 | 480 | 240.456 | 11.590 |
| Legacy 16/8/4 | 0 | 459 | 240.479 | 11.578 |
| Fresh hybrid 16/8/4 | 297 | 211 | 240.453 | 11.611 |

All 211 common Full FT exposures match item/dropout/sigma; separate RNGs prevent
optimizer stochastic rounding from changing inputs. Initializer peak was
4.477 GiB. Total here counts teacher/model forward, backward and optimizer CUDA
events, including cold updates; shared preprocessing, input transfers, hashing
and evaluation are excluded. It is not total end-to-end GPU occupancy.

The reviewed native CFG4 pair preserves clear figures. Legacy preserves the
single figure but distorts the two-person composition. Fresh hybrid starts
with coarse silhouettes, pale/glowing detail and facial distortion; subsequent
Full FT improves the single face/outline, while two-person scene division,
grid-like artifacts and lettering remain. This proves a bounded short-training
improvement, not consistent superiority over legacy. It closes the budget-controlled
comparison requirement at this scope without selecting a default. The retained
3700-update candidate uses a different, longer recipe and is not substituted
for this fresh experiment. No large Full checkpoint was written.
Evidence: `docs/audits/data/hybrid_anima_equal_total_2026-10.json`.

### Ming LoRA preview precision

The INT8 ConvRot Ming GPU LoRA probe exposed a preview failure before its first
update: BF16 activations reached FP32 LoRA masters without autocast. The common
adapter reference algebra intentionally relies on caller-owned autocast; Qwen's
trainer preview already provides it. Ming now applies the trainer's precision
context only around each DiT prediction through its sampling bridge, preserving
conditioner/VAE precision and FP32 adapter storage. No adapter algebra or API
default changed. A native CPU BF16/FP32-LoRA preview regression checks the
prediction context, unchanged weights, separate encoder/decoder contexts and
context restoration; training-preview tests pass. The subsequent real INT8
ConvRot retry completed two updates and resumed step 2 to step 3. All 72
structural payload tensors match the factory, and all 344 optimizer states,
including 72 structural entries, were restored with none fresh and counters 3.
Saved-payload API generation at four steps succeeded without warnings. The owner
Ming Layer model was restored and only the owned smoke run was removed from DB
and disk; logs/sample hashes remain. This is INT8-base functionality acceptance,
not BF16-base or quality-recovery acceptance. Qwen GPU LoRA remains separate.
Evidence: `docs/audits/data/hybrid_ming_lora_gpu_2026-10.json`.

### Qwen INT8 LoRA GPU restoration

The Qwen 2.1 INT8 ConvRot native-codec probe completes two rank-eight FP32-LoRA
updates, periodic previews and latest-checkpoint continuation to step three.
The saved complete payload matches all 72 structural factory tensors and shapes;
all values are finite. All 328 named optimizer entries, including 72 structural
entries, restore with zero fresh entries and counters three. Loading the saved
LoRA onto the native package through the generation API succeeds at 512px/four
steps without warnings (gallery 11627). The owner Ming Layer model is restored
and the owned smoke run is removed from DB and disk after retaining evidence.
This verifies INT8-base functionality, not unquantized BF16, teacher-initialized
quality, VAE-swap or partition/global-summary GPU acceptance.
Evidence: `docs/audits/data/hybrid_qwen_lora_gpu_2026-10.json`.

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

## Output-only codec cycle calibration probe

A bounded output-only refit uses the existing calibrated input bridge's actual
first-16-channel clean state, rather than native-codec targets, to fit a new 3x3
16-to-64 output projection on 28 general images. Four images are excluded from
this refit; their absence from the older retained calibration is not established.
Token geometry and the input bridge are unchanged. Mean held reconstruction MSE
changes from 0.615642 to 0.611011, with one of four cases worsening. The inspected
single-figure face/outline improves, while the two-figure scene has stronger edge
artifacts; block texture persists. This is clean codec reconstruction only, not
DiT/noise-field or free-running generation acceptance. It does not justify a
production/default change. The owned worker exited and no API model/run changed.
Evidence: `docs/audits/data/hybrid_codec_cycle_probe_2026-10.json`.

## Reduced-grid packed codec IO experiment

A new codec-only probe packs each native 16-channel 2x2 cell into 64 channels
on the replacement codec's 32x32 grid at 512px. Both directions use 3x3 paired
calibration. An additional output refit reconstructs target clean latents from
the actual incoming projection. All new fits use the same 28 general images,
with four excluded evaluation images. Mean target-latent reconstruction MSE is
0.637771 for the freshly matched pooled-16 cycle, 0.516950 for packed-64 cycle
(18.94% lower), and 0.615642 for the older retained production-geometry control.
The improvement is not attributed to packing alone versus the older differently
fitted control. Pair-only packed fitting gives 0.597980, while direct native
packed-to-target fitting gives 0.659110. The inspected face/outline is more
detailed but strong block/edge artifacts remain. These are codec reconstructions,
not denoising or free-running generation. No default is promoted.

`core/experiments/anima_packed_io.py` provides an explicit coefficient prototype
for native 2x2 Anima IO: average four native input-patch embeddings into one
reduced-grid token, and repeat the native output patch over four coarse cells.
It respects distinct input channel-major and output patch-major orders. Synthetic
FP64 testing, including nonzero patch-constant padding, gives input maximum error
1.78e-14 and exact output agreement. Token count is 24 versus native 96 in that
case, with 64 channels and unchanged patch size two. This is linear-boundary
acceptance only: body/RoPE behavior, calibrated noise covariance and generated
quality are unverified. Nonconstant padding/partition semantics are not accepted.
The prototype is not installed into existing runs or checkpoints. Its next gate
is an isolated DiT/noise-field comparison with explicit codec and IO state,
followed by held general generation before any runtime/schema/default change.
Evidence: `docs/audits/data/hybrid_codec_packed_probe_2026-10.json`.

### Packed IO generation and head-noise check

A real official Anima probe installs the packed IO coefficient transfer with the
retained high16 structural candidate and native-paired/cycle codec output fits.
It generates at 512px/28 steps/CFG4/seed63 for two general prompts, with and without
hybrid topology. Original per-cell noise complements give near-solid purple
outputs, including swap-only; the clean cycle fit is not the sole cause. Packed
input averaging reduces native patch-feature noise covariance to one quarter;
a gain of two restores its covariance algebraically (synthetic FP64 maximum
error 1.71e-13). It does not by itself restore generated quality.

The initial repeated output head predicts a 2x2 average/repeat noise projection,
not independently whitened noise at every coarse cell. Correcting that snapshot
reduces a zero-clean synthetic oracle's mean velocity error from 2.97335 to
8.72e-16. `packed_head_noise` records this experimental boundary projection;
three regression tests cover the IO orders, nonzero patch-constant padding,
covariance, head/noise agreement and invalid geometry. The corrected native-paired
four-image generation clears the purple collapse, but strong stripes/blocks
remain, two-person composition fails and one-person faces are fragmentary or
exaggerated. This does not meet the coarse initialization gate. No production
noise contract/default is changed; the prototype is not published as a checkpoint.
No new DiT fitting occurred. Further work must initialize IO/fields coherently
before claiming a useful packed-codec candidate.
Evidence: `docs/audits/data/hybrid_anima_packed_generation_2026-10.json`.

### Packed IO-only initialization learning

A bounded 200-update probe trains only the FP32 Anima input/output projections
and two codec bridges (six tensors, 1,130,624 parameters), starting from the
head-aware noise snapshot and native-paired output calibration. Native body,
caption bridge and retained 3700-update high16 structural modules stay fixed;
their before/after parameter hashes match exactly. Twenty-eight general images
train, with indices 28–31 excluded from this adaptation. Fixed captions, independent
noise, uniform sigma 0.01–0.99 and 0.1 null conditioning are used. AdamW8bit at
LR 1e-4 completes all six optimizer counters to 200; tensors and losses are finite.
Training takes 64.672 seconds and peaks at 4.095 GiB allocated. This excludes
setup, codec/TE encoding, hashing and evaluation, and is not a Full FT memory claim.

Two new held-prompt CFG4/seed63/28-step outputs reuse the preceding four-image
control by hash. The earlier exaggerated eyes diminish and some figure outlines
change, but strong block texture and failed two-person composition persist.
There is no consistent quality advantage or coarse-initialization acceptance.
The private six-tensor payload is retained for further experiments, not published
as a production artifact or made a default. No owner run or loaded model changes.
The first trial used the frame wrapper for RGB training input and failed before
updates; the corrected trial uses the image wrapper.
Evidence: `docs/audits/data/hybrid_anima_packed_io_fit_2026-10.json`.

### Matched packed IO versus IO/structural adaptation

A second bounded probe starts both arms from the same saved IO200 and structural
3700 states. Both keep native body parameters fixed and use FP32 IO/bridge and
structural storage with BF16 body/compute. Two hundred exposures match exactly
by item, null draw, sigma and noise SHA-256. Fresh AdamW8bit states at LR 1e-4
train either six IO/bridge tensors (1,130,624 parameters) or those plus all 65
structural tensors (6,088,832 parameters). All optimizer counters reach 200 and
both frozen-body hashes match the shared start. IO-only takes 64.562 seconds and
4.104 GiB allocated; joint adaptation takes 81.344 seconds and 4.124 GiB.
This is matched-exposure comparison, not equal-total-compute or optimizer resume. A separate IRIS generation worker was present during
this probe, so wall times are not isolated throughput measurements. Allocated
VRAM is the probe worker allocation, not whole-device usage.

Last-50 matched training mean losses are 1.85723 and 1.52018 respectively; they
are not held-out field scores. Four new held-prompt outputs at the same seed/CFG
show smoother regions with joint adaptation, but two-person composition fails
and severe block artifacts remain. This limited change does not justify default
promotion. Further work must test a concrete geometry/field hypothesis rather
than assuming that more of the same short fitting will fix the initialization.
Evidence: `docs/audits/data/hybrid_anima_packed_joint_fit_2026-10.json`.

## Source-span RoPE retrofit experiment

Current Anima full-frame coarse execution regenerates native unit-grid angles
at each depth. A private alternative averages native linear angles over declared
source-grid cells, retaining their physical span and cell centers without pooling
sine/cosine. `core/experiments/anima_source_span.py` accepts only single-frame,
full-canvas divisible grids; it does not mutate position buffers. Three CPU tests
verify native layout, cell centers, identity, unchanged buffers and rejection.
The real-import test leaves CUDA uninitialized while a separate IRIS job runs.

Four new 512px/28-step/CFG4/seed63 general-prompt images use source grid32x32
with native codec or packed64 codec and the retained unit-grid-fitted structural
candidate. Existing controls are reused by hash; no fitting occurs. Native
outputs worsen: two-person composition splits/fails and one-person facial
structure degrades. Packed outputs retain stripes/blocks and fail figures.
Thus source-span coordinates are not retrofitted or promoted. This experiment
does not reject a freshly source-span-fitted model or establish universally
optimal coordinates. Any future position-policy adoption needs explicit
artifact identity and matched fresh initialization, not silent existing-run edits.
Concurrent IRIS generation means no isolated timing claim is made.
Evidence: `docs/audits/data/hybrid_anima_source_span_2026-10.json`.

## Fresh source-span initialization comparison

The retrofit confound is now separated with two fresh native-codec Anima arms.
Both start from identical new 16/8/4 structural modules and the same frozen
native body; no retained correction weights are loaded. Each receives exactly
300 teacher-fitting updates on 28 general images, AdamW8bit/SR at LR0.001,
uniform sigma0.01..0.99 and null probability0.1. All image choices, null decisions,
sigma values and noise hashes match. Initial module hashes and native body hashes
match across arms, and the body remains unchanged. All 65 optimizer entries
reach step300. Source-span retains source grid32x32 and is identity for the
native teacher's full grid; its coarse angles are cell-center averages.

Four new held 512px/28-step/CFG4/seed63 images exclude these prompts from fitting.
Unit-grid yields two separate back-facing figures and a coherent single girl's
outline, with floating hand-like and distorted-eye details still present.
Fresh source-span fitting produces overlapping contours, smeared regions and
worse facial/composition structure in both prompts. Last50 mean training loss is
0.04618 for unit-grid and 0.05246 for source-span; these are training observations,
not held quality scores. Model-update CUDA times are127.58s/120.17s and own peak
allocation4.476/4.479GiB, excluding encoding, hashing and generation; a single
ordered quality comparison does not establish a throughput advantage.

Source-span32 is therefore not selected as the default by this bounded test.
This also rejects attributing the earlier retrofit failure solely to lack of
matching initialization. It does not prove that a longer fit, another geometry
or another coordinate policy cannot work. Unit-grid's remaining defects and the
combined-codec quality gate remain unresolved. No owner artifact, public policy
or runtime default changes. Images and private module weights remain local;
tracked evidence contains hashes, metadata and observations only.
Evidence: `docs/audits/data/hybrid_anima_source_span_fit_2026-10.json`.

## Wider compressed-skip control

Seven skip ranks increase from64 to256 while the native-codec 16/8/4 layout,
boundary/local branches and 300 teacher-fit exposures stay fixed. Unchanged
module weights are copied at native BF16 precision; each capture retains its
original64 rows and all restore projections start at zero. The existing rank64
control is reused by hash. Item/null/sigma/noise records match all300 updates,
the frozen native body matches exactly, and the first exposure loss is exactly
equal. Posterior samples are recreated by the same seeded preprocessing code;
the earlier control did not retain per-image latent hashes, so their equality
is not independently proved for every exposure. Encoded inputs are now retained
locally for subsequent controlled comparisons.

At unchanged AdamW8bit/SR LR0.001, all65 optimizer entries reach300 with finite
gradients. Structural parameters increase from4,958,208 to10,463,232; own peak
allocation rises from4.476 to4.501GiB. Last50 training loss increases from0.04618
to0.08177 and both new held CFG4 images lose coherent faces/composition with
strong blocks and fragments. This rejects rank256 at the unchanged fitting LR,
not its capacity under other optimization conditions. Error increases immediately
after the initially identical prediction, so test a lower skip-only LR before
concluding that compressed-skip capacity cannot help. A separate IRIS GPU worker
is active; wall times do not establish isolated throughput ranking. No runtime
default or owner artifact changes. Evidence contains no images or tensor payloads:
`docs/audits/data/hybrid_anima_wide_skip_fit_2026-10.json`.

### Skip-only learning-rate control

A fresh rank256 arm lowers only skip LR from0.001 to0.00025 while the other
branches stay at0.001. Initial module/body hashes match the previous rank256
arm, encoded CPU latents and conditioning are reused from the same hashed file,
and all300 exposure/noise records and the first loss match. The frozen native
body remains exact; optimizer groups own all65 tensors once and reach300.
This removes the prior comparison's unrecorded posterior-sample confound when
comparing the two rank256 arms. Against the older rank64 arm that limitation
remains; it is not retroactively erased.

Two new held CFG4 images recover much of the severe wide-skip collapse: the
single girl's face and outline are coherent, while the two-person image retains
overlapping heads/figures and fragmented anatomy. Relative to rank64 the result
is mixed, not a consistent quality advantage. This establishes optimization
sensitivity rather than proving a larger skip is intrinsically harmful or that
rank256 should be the default. At the time of this probe public initialization
enforced uniform phase rates. Its private module weights still have no public
initializer lineage and must not be relabeled as a public initialized artifact.
The subsequent explicit rate control below permits new, recorded initialization;
default adoption still needs further quality evidence.
No runtime defaults or owner artifacts change. Evidence:
`docs/audits/data/hybrid_anima_wide_skip_lr_fit_2026-10.json`.

### Recorded skip-rate initialization option

Each API initialization phase accepts `skip_learning_rate_multiplier` in(0,1],
default1 from `api/param_defaults.py`. For example a phase with learning_rate0.001
and multiplier0.25 fits compressed-skip capture/restore weights at0.00025 while
down/up/local weights stay at0.001. It changes initialization only, not subsequent
training rates. Anima, Ming and Qwen pass the setting through the shared fitting
transaction. Reduced rates require skip modules; invalid/nonfinite/boolean values,
effective-rate underflow and optimizer group overrides are rejected. Parameters
must still have exactly one optimizer owner, with rollback on failure/cancellation.

Phase rates are stored in initialization lineage and included in its hash.
Older records without the optional field retain their exact contents and hash;
absence means uniform rates. New records explicitly describe any requested
multiplier. Default1 preserves numerical updates and optimizer ordering, without
changing topology defaults or migrating owner runs. The UI's advanced phase JSON
receives defaults from the schema, sends the field intact and explains its scope.
Frontend build/type checks remain owner work; static review covers types, request
transport, busy/cancel/error paths and default state.

Eighty-five distinct CPU cases cover transactions, legacy/new lineage, API
defaults/validation, native owners and all three artifact export/reload paths.
Real Anima BF16/AdamW8bit/SR completes two public fitting updates:51 non-skip and
14 skip tensors have rates0.001/0.00025, all65 optimizer entries reach2 and the
native body stays exact. Complete structural tensors and metadata round-trip
through a local safetensors payload. That payload is not a full production model
or quality acceptance. Backend compilation and real GPU-free imports pass;
API restart confirms schema defaults, invalid-zero422 and the prior Ming Layer
model. Evidence: `docs/audits/data/hybrid_skip_rate_public_2026-10.json`.

### Equal-total-budget wide-skip candidate

Fresh factory rank256 skips with initial skip LR0.00025 and other branches0.001
receive120.274 CUDA-event seconds of teacher fitting (319 updates), followed by
119.839s of native-codec Full FT at uniform LR5e-6/SR (222 updates). Total240.113s
includes initialization, matching the historical native/legacy/rank64 budget.
The frozen source body matches the earlier rank64 initializer and remains exact
through teacher fitting; all211 common Full FT item/null/sigma records match.
Encoded inputs reuse the current hashed cache. Historical per-image posterior
and noise hashes were not retained, so those equalities are not independently
proved against the old sessions. Preprocessing, hashing and generation are excluded
from the model-update budget. The new factory does not copy the prior private
rank64 capture rows into larger skips; do not treat the two initializations as
identical. Existing twelve Full FT comparator images are reused by hash.

Six new held general images separate initialization and Full FT. Initial CFG4
outputs retain two distinct back-facing figures and a coherent single girl,
with contour/ghost defects. Full FT retains the single face/outline but the
two-person output develops a headless foreground coat and fails composition.
Both new CFG1 outputs retain overlapping/fragmented contours. The native CFG4
control remains stronger on two-person structure; legacy and rank64 controls
also fail that prompt differently. There is no consistent equal-budget advantage
that would justify promotion. Own Full FT peak allocation is11.644GiB; this
512px ordered quality session is not a new1024px throughput matrix.

Next distinguish the fine-tuned native-body path from the rewired path under the
same weights before extending this recipe: deterioration after initialization
does not prove whether native-body drift or structural mismatch dominates.
No owner checkpoint, runtime default or public initialization lineage changes.
Evidence: `docs/audits/data/hybrid_anima_wide_skip_equal_total_2026-10.json`.

### Fine-tuned native versus hybrid routing

A repeat starts from the saved319-update factory initialization and cached
inputs, then performs222 Full FT updates. All item/null/sigma/noise hashes and
the first loss match the previous probe. Later loss differs by up to0.015653;
none of the four hybrid PNG hashes matches the historical output. This is not
an exact historical replay, and the responsible RNG/backend state has not been
established. Boundary RNG snapshots were not retained in the earlier test.

The new within-model comparison remains valid independently: the trained body
hash is unchanged across both routes, same native codec and conditioning are
used, and only the runtime topology gate is disabled for native execution.
Eight new held general outputs cover both routes at CFG1/4. On the two-person
CFG4 prompt, native routing produces a split/collage scene with repeated or
missing heads; hybrid routing has a background person and a headless foreground
body. Both fail composition. On the one-person prompt both retain a recognizable
girl; hybrid still has facial/anatomy defects. CFG1 outputs retain fragmented
contours and anatomy.

Switching back to native routing therefore does not cure this trained sample;
it is not only a final upsample/display artifact. However the body was optimized
for hybrid execution, so its original native behavior is not guaranteed, and
this does not establish native-body drift as the dominant cause or diagnose
owner run#174. Next test a separate, higher correction-module Full FT LR with
native-body rate/init/data held fixed, and capture boundary RNG state for later
comparisons. Existing owner runs and defaults remain unchanged. Evidence:
`docs/audits/data/hybrid_anima_trained_routing_2026-10.json`.

### Boundary-detail holdout and supervised Full FT control

The private boundary-gain candidate now has four held general prompts at512px,
seed63 and28 sampling steps. On the two additional CFG4 prompts, a kneeling
character and an eating-burger character remain recognizable, but surrounding
hands, mouth/burger contact, pose and monochrome line quality degrade relative
to native. Native CFG1 also fragments on these prompts; conversion is not the
only source of that behavior. With the same fitted65 structural tensors,
turning gains off only at inference preserves the original two CFG4 coarse
compositions. Pixel differences are nonzero (mean absolute0.83–3.34/255 over
the four ablations), but are not a perceptual quality score. Thus the earlier
active-versus-zero-rate fitting result does not establish a direct inference
gain benefit: the fitted structural trajectory also differs.

Both matched candidates subsequently complete222 supervised Full FT updates
without teacher loss. They use identical original native-body hashes, starting
CPU/CUDA RNG, cached inputs and item/null/sigma/noise records. Each optimizer
contains567 native,65 existing structural and2 gain tensors; native LR5e-6,
structural LR5e-5 and gain LR5e-5 or0. All634 counters reach222 and native bodies
change. Last50 losses0.16975/0.16996 and whole-update103.571/102.011s do not prove
quality. Own peaks11.641/11.645GiB are512px probe measurements, not a replacement
for the1024px four-condition matrix.

After Full FT both arms fail the two-person composition with horizontal strips,
spurious lettering and incomplete figures. Single faces remain recognizable
but anatomy degrades; the active eating-burger sample loses the burger/hands.
CFG1 still fragments. Reject this recipe as a default or quality acceptance;
do not infer that every boundary fusion or longer training is impossible.
A matched native-only supervised control distinguishes this exposure recipe
from conversion below. The active private body is retained locally without
optimizer state; it has no public gain/payload lineage and proves no resume
contract. No owner run, checkpoint or runtime default changes.

Evidence: `docs/audits/data/hybrid_anima_boundary_detail_holdout_2026-10.json`
and `docs/audits/data/hybrid_anima_boundary_detail_fullft_2026-10.json`.

The native-only control completes the same222 cached exposures, native LR5e-6,
BF16 AdamW8bit/SR, starting RNG and24 checkpoint blocks. Its567 optimizer
counters reach222 and native-body hashes change. Parameter-set differences
change clipping and SR ownership; this is not bitwise update parity. Last50
loss0.15437,96.247s whole-update compute and11.586GiB own peak are bounded
probe facts, not quality metrics or a new matrix.

Its CFG4 outputs retain two back-facing figures, external hands touching a
kneeling character's head, and a burger held at the mouth. The pointing girl
also has more complete face/body/limbs. Unwanted text/panel layout persists in
the two-person scene; CFG1 still fragments there. This is not broad native
quality acceptance, but disproves an explanation based solely on the exposure
recipe for the hybrid's missing figures/hands/objects. Conversion/optimization
interactions still require improvement before a default is selected. Next
inspect update allocation and retained stage information rather than promoting
the boundary gain from its initialization samples or repeating loss-only fits.
Evidence: `docs/audits/data/hybrid_anima_native_matched_fullft_2026-10.json`.

### Starting-state gradient allocation and shared clipping

Five exact cached general exposures (four conditional, one dropped) compare
native execution with the fitted boundary-detail candidate without updates.
The native-body hash remains exact. Structural tensors contribute only
0.57–1.37% of the total squared gradient norm. Hybrid total norms are smaller
than native on all five exposures, so these measurements do not support
structural gradients dominating the shared clip and starving the native body.
Mid-stage native-block gradient cosines are0.068–0.101 and deepest-stage
cosines0.016–0.098; the output-head cosines remain0.748–0.861. These are grouped
starting-state gradient directions, not actual Adam/SR update magnitudes or a
causal proof. Converted low-resolution processing changes the optimization
direction strongly even when field loss is fairly close. Prior feature fitting
did not establish quality recovery; do not turn these norms into an LR default.

Static review also found a separate production bug: both ordinary optimizer
step paths applied `max_grad_norm` only to group0. Normal GradScaler and
non-GradScaler execution now clip one shared norm over every optimizer group,
counting each parameter once. Existing single-group behavior and fused-path
contracts remain intact. Four analytical CPU cases cover two independently
rated groups and their optimizer update, single-group reference equivalence,
duplicate identity and absent gradients; the existing fused clip/accumulation
tests also pass (49 total). Earlier private Full FT probes already clipped all
parameters, so this fix does not explain or repair their quality regression.
No API parameter/default or owner config changes.

Evidence: `docs/audits/data/hybrid_anima_gradient_allocation_2026-10.json`.

### Boundary capacity extension control

An independent experimental helper widens declared boundary ranks while copying
fitted reduce rows/expand columns and zeroing the new output columns. New reduce
rows keep factory initialization. It returns new factory-described modules,
never edits a live run, and refuses extra private state. Ten new CPU cases cover
fitted-output/input-gradient parity, checkpointing, RNG/source preservation,
staged gradients into new coordinates, strict factory reload/census and bad
capacity requests. With25 existing hybrid cases,35 tests pass. Boundary rank
is already part of the strict declaration; no API/default field is added.

The Anima control starts from the same319-update fitted coordinates and raises
only the two boundary ranks64→512. Skip rank256 and local rank64 remain fixed.
It adds7,340,032 parameters for17,803,264 structural weights total. Native-body
hashes remain exact, all65 optimizer counters reach200, and the new output
coordinates receive gradients before their new reduce coordinates do. Saved
modules reload strictly through the declared factory with every tensor exact.
This proves module restoration, not production optimizer/resume acceptance.

All200 item/null/sigma/noise exposures and starting RNG match the reused rank64
field control. Wider tensors change SR draw ownership; BF16 first field loss
also differs by6.05e-5 despite copied/zero-padded coordinates. No exact GPU
forward or update parity is claimed. Last50 teacher-field loss0.02539 improves
on0.02949, but held CFG4 samples have teal/dark bands, incomplete figures and
deformed faces/hands. CFG1 retains collage fragments. Reject this recipe as a
default; do not infer that every capacity layout or longer fitting must fail.
Whole-update68.464s and4.532GiB own peak are bounded200-update initialization
costs, not isolated convolution overhead or a revised Full FT matrix.

The next architecture question is the information layout at down/up and fusion,
rather than promoting rank512 from its field loss. Owner artifacts and presets
remain unchanged. Evidence:
`docs/audits/data/hybrid_anima_boundary_capacity_fit_2026-10.json`.

### Spatial-phase boundary control

A private branch separates the2×2 positions before channel projection. Down
uses per-token LayerNorm, pixel-unshuffle, rank64 mixing and a zero-output
projection. Up receives the original coarse delta, projects to four output
phases and pixel-shuffles, after existing fine-grid up correction. The pack
itself retains position order; the subsequent bottleneck remains lossy. This
is not an invertible boundary or a quality guarantee. Default execution has
no coarse-up callback and is unchanged.

Fourteen new CPU cases verify independent phase-layout oracles, zero identity,
staged gradients, nested/grouped grids with three checkpoint and three skip
policies, other-root isolation, cleanup/transfer failure and shape rejection.
Together with32 existing runtime/hybrid cases,46 tests pass. Context ownership
is temporary and root-specific; its weights have no public topology lineage.

The Anima candidate adds2,629,632 weights in12 tensors to the same fitted319
initialization. Original51/14 tensor groups keep LR0.001/0.00025; new phase
tensors use0.00025. All77 counters reach200, new output projections receive
initial gradients and all phase parameters receive final gradients. The native
body hash is exact. Same200 cached item/null/sigma/noise records, starting RNG
and first field loss match the field-only control; extra tensors change SR
draw ownership. Last50 field error0.02833 versus0.02949 does not prove quality.
Whole-update66.861s and4.519GiB own peak are initialization probe costs only.

Held CFG4 has missing heads/figures, large cyan bands, spurious lettering and
deformed or absent facial detail; CFG1 retains collage fragments. Reject this
recipe as a default. Explicit rank64 private-context restoration reproduces
all77 saved tensors on CPU, but this is not public payload/optimizer/resume,
Full FT or combined-codec acceptance. No owner config/default is changed.
Evidence: `docs/audits/data/hybrid_anima_phase_boundary_fit_2026-10.json`.

### Head-aligned Anima width-projection foundation

After fixed-width fitting controls fail practical recovery, the conditional
stage-width investigation starts with a private block-level weight transfer.
Keep native head dimension128, select whole heads, and copy corresponding
QKV/output rows/columns, MLP neurons, QK-norm weights and all three AdaLN
shift/scale/gate slices. Cross-attention keeps the full1024 text input width.
Native timestep embedding and shared AdaLN conditioning are selected in the
same channel order; RoPE frequencies keep their head dimension. Quantized/custom
Linear layers, active style context, live block-offload callbacks and extra
per-block positions are refused by this prototype.

The wrapper keeps the outer2048 feature stream, initially updating selected
coordinates. An optional learnable encoder/decoder bridge starts as those
selection/lift matrices, shares input projection with timestep/AdaLN inputs,
and lifts the narrow residual update to every outer channel. CPU tests prove
initial parity and a decoder update into previously unselected channels.
This is a block-working-width foundation, not a completed variable-width
stage graph. Its manifest/weights have no public artifact/factory lineage.
Sixteen CPU cases cover full-width input-gradient/forward parity including
head reorder and both AdaLN forms, checkpoint modes, strict private reload,
conditioning layout, source/RNG preservation and learned bridge gradients.

Two synthetic GPU probes read only actual native block12 weights. At unchanged
width, BF16/RoPE output matches exactly with and without the learned bridges;
native weight hashes stay exact. Width1024/512 forward/backward is finite, and
encoder/decoder gradients—including omitted output-channel rows—are positive.
Block parameters fall69,206,528→19,923,456/6,291,968 without bridges, or
24,117,760/8,389,120 with bridges. Native-relative synthetic output errors are
0.702/0.730 at initialization: this is not an aligned or quality-accepted
student. No images are used, and no optimizer update occurs in GPU timing.

Each probe records just two timed passes after warmup with native checkpointing.
These one-block synthetic timings/allocations cannot replace the requested
whole-model U-Net±codec matrix, nor prove end-to-end speed. Next needs explicit
private stage teacher/student ownership and alignment, then whole generation
and optimizer restoration before public factory/API/UI promotion. Existing
owner runs, topology presets and checkpoint identities remain unchanged.
Evidence: `docs/audits/data/hybrid_anima_projected_block_2026-10.json`.

### Private projected-stage ownership and restoration

`core/experiments/anima_projected_stage.py` temporarily replaces selected
coarse-region blocks with projected students. Native blocks remain frozen and
are reinstalled for each no-grad, eval-mode teacher call with the U-Net topology
disabled. Student blocks and training modes are restored even when the teacher
raises; outer cleanup restores source blocks, flags, modes, topology and RNG.
Existing gradients, block offload and other initialization owners are refused.
The projected wrapper exposes its internal self-attention to Anima's per-forward
style-context stamping without registering a duplicate module.

Private payloads include native parameter fingerprint, exact topology, projection
manifests, structural tensors and student tensors. Identity, keys, shape, dtype,
finite values and integer channel coordinates are checked before any tensor is
loaded. This is experimental weight restoration, not public checkpoint lineage
or optimizer resume. Ten stage CPU cases plus sixteen projection cases pass.
Outer features still have native width; these blocks do not yet implement a
variable-width activation graph. Existing runs and presets are unchanged.

Real Anima coarse blocks8..19 are projected2048→1024 with learned bridges,
starting from the same319-update structural initialization and cached general
exposures. Original51/14 structural tensors use LR0.001/0.00025;264 projected
tensors use0.000005. All329 optimizer counters reach200; each encoder/decoder
has positive first gradients. Native body fingerprints match after cleanup.
The explicit private payload reload succeeds before generation. Initialization
cost is88.890s and5.847GiB own peak, including resident frozen teacher blocks;
this is not student-only memory or the final training matrix.

Field loss starts0.24837 and last50 mean is0.03681. Both held CFG4 generations
have severe figure fragmentation and horizontal bands, with no coherent facial
or two-person structure. Reject this200-update recipe as a default. Width
projection functionality is verified, but alignment/Full FT quality remains
open; these results do not show that every narrower architecture must fail.
Evidence: `docs/audits/data/hybrid_anima_projected_stage_2026-10.json`.

### Full-rank boundary feature fusion probe

`core/experiments/boundary_features.py` exposes saved fine features and actual
coarse input/output independently to a full-rank1×1 projection. The coarse
features are bilinearly restored to the fine grid before concatenation. This
learnable correction is added to existing boundary restoration; zero weights
preserve the starting output exactly. It can learn unrestricted channel mixing
of all three fields, unlike low-rank skips or diagonal gains. This is not a
replacement of all decoder blocks with SDXL residual blocks.

The optional runtime hook receives the actual coarse output, before restoration,
instead of approximating it from the already corrected fine-grid update.
No hook means unchanged execution. Temporary registration is root-specific;
invalid widths, active hooks and concurrent probes are refused. Modes with
checkpointing retain gradients through all three fields, including nested and
grouped grids and encoder replay. Thirteen new CPU cases and32 existing
runtime/hybrid cases pass. Private strict tensor reload is verified; public
factory lineage, optimizer restoration and preset adoption remain conditional
on generation/Full FT acceptance.

The fixed-width Anima probe adds25,165,824 parameters in two fusion weights to
the same319-update initialization. Original51/14 structural tensors keep
LR0.001/0.00025; new feature weights use0.000025. All67 optimizer counters reach
200, first/final new gradients are positive and native fingerprints remain exact.
Same200 cached exposures, start RNG and first loss match the field-only control;
additional SR parameters change draw ownership. Last50 field loss0.02725 versus
0.02949 does not itself prove quality. Whole-update GPU time74.364s and4.690GiB
own peak describe initialization only. Explicit private-context CPU restoration
matches all67 saved tensors exactly.

Two held CFG4 generations retain two back-facing figures and a coherent girl's
face/body, with extra lettering and detail defects. This is a candidate for
Full FT verification, not a default or a general quality guarantee.

The subsequent222 supervised Full FT exposures match the previous boundary-detail
control by item/dropout/sigma/noise hash and starting native/RNG hashes. Native
LR5e-6 and old/new structural LR5e-5 update all634 tensors to counter222; native
body hash changes. Initialization and parameter shapes differ, so SR/backward
updates are not identical. Last50 supervised loss0.17743, GPU update time105.972s
and11.779GiB own peak do not establish quality. The two-person output loses one
coherent figure; the single person's face/body become substantially distorted.
Reject this recipe as a default despite its initially coherent teacher-fit
generation. Public wiring and final performance matrix remain pending quality
recovery. This does not prove unrestricted feature fusion cannot work.
Evidence: `docs/audits/data/hybrid_anima_boundary_features_2026-10.json`.

### Full-rank fusion training stability controls

The same fitted fusion weights are now compared with a frozen native body and
with all Full FT learning rates reduced tenfold. Both use the same222 supervised
item/dropout/sigma/noise records, starting native/RNG hashes and exact first loss
as the previous full-rank Full FT. No teacher loss is added.

Frozen-body learning updates only67 structural tensors at5e-5, with counters222
and identical native fingerprints. The two-person composition is more coherent
than the earlier Full FT, but face/eye and body details remain distorted. Changed
parameter and clipping sets also change SR draw ownership; native updates alone
cannot explain all degradation. Update GPU time56.011s, own peak4.690GiB and
last50 supervised loss0.17073 describe this diagnostic, not Full FT acceptance.

The reduced-rate arm keeps the same634 trainable tensors and parameter ordering:
native LR5e-7, original/new structure5e-6. All counters reach222 and the native
body fingerprint changes. Two held CFG4 samples retain the two-person scene and
single person's body, with detail/hand/face defects; catastrophic fragmentation
seen at the previous rates is absent. Last50 supervised loss0.16469, update GPU
time104.971s and11.779GiB own peak are bounded512px costs, not the final matrix.
This is a candidate rate setting, not a universal Anima or hybrid default.

Explicit private-context full-body restoration in a fresh model matches all755
saved tensors exactly. An additional held prompt31 retains a coherent person,
hands and food near the mouth, with imperfect facial/food detail. This verifies
private weight restoration and one additional general composition, not public
payload lineage, optimizer continuation, partition/codec parity or broad quality.
P5/P6 remain open; public wiring and final U-Net±codec performance verification
are still required. No owner run or learning rate is changed.
Evidence: `docs/audits/data/hybrid_anima_boundary_features_stability_2026-10.json`.

### Declarative full-rank boundary fusion

Hybrid schema v2 boundaries now accept `fusion: "full_rank"` in addition to
`"none"`. Omission uses the central `DIT_HYBRID_DEFAULTS.boundary_fusion`
(`none`); old declarations and hashes are preserved. Full-rank fusion adds a
zero-output, bias-free 1x1 convolution over the saved fine grid, bilinearly
resized coarse input, and actual coarse output. Its residual correction is
added at the boundary exit. It adds `3 * width²` parameters per boundary; it
does not narrow the native stream or replace its blocks.

The factory registers these weights as `rewiring_modules.features.<id>.weight`.
The declaration and layout hash own the operator, so normal generation,
training and checkpoint restoration reconstruct it without experimental hooks.
Nested regions, image-layer grid groups, module checkpointing and activation
offload use the shared execution path. Private research checkpoints are not
automatically migrated or given fabricated initializer provenance.

The training editor offers inherited/none/full-rank selection with the existing
validate/apply workflow. The graph marks fusion exits and displays logical BF16
temporary forward storage: two resized fields, a three-field concatenation and
an output (`6 * fine_tokens * width` elements per image grid and batch item).
This excludes autograd and kernel workspace; it is not a measured VRAM peak.
API types, OpenAPI, server counts and parameter-distribution display include the
new operator. Frontend review is static; owner build and UI acceptance remain.

CPU integration passes 213 cases with 16 intentionally unsupported adapter
algebra combinations skipped. Coverage includes Anima, Ming Design/Layer and
Qwen forward/backward, strict weight restoration and bounded optimizer
continuation. These checks establish wiring, not convergence or the final
U-Net/codec performance matrix. Existing defaults and owner runs are unchanged.

A real Anima BF16 probe explicitly migrates the retained lower-rate research
copy into this declaration (755 tensor values match), updates all634 Full FT
tensors twice, and saves through `AnimaFullParameterAdapter.write_checkpoint`.
The native loader rebuilds the public topology from the saved file. The shared
named optimizer restorer restores634 states with zero fresh tensors; one further
supervised update advances every counter from2 to3. A held general CFG4 image
retains a coherent person, hands and food. This exercises normal full-model
export/load and shared optimizer remapping, not a complete train_runner,
dataset-order or UI resume. Evidence:
`docs/audits/data/hybrid_anima_public_feature_fusion_2026-10.json`.

### Public full-rank candidate: U-Net / Qwen VAE compute matrix

RTX6000 Ada,1024px, BF16, AdamW8bit with SR,24 native blocks checkpointed,
hybrid modules checkpointed, GPU skip storage, gradient clipping1. Each arm runs
in a fresh worker with one warmup and five measured updates per batch. All four
arms start from the same lower-rate Full FT native body (public step2 checkpoint),
not the original pretrained quality baseline. Removing the U-Net deletes only
its structural modules. The Qwen64ch/16x arms reuse40 general-rated calibration
images' fitted spatial/noise bridges; no images are re-encoded for this test.
Inputs and64-token conditioning are synthetic. VAE/TE encoding, IO, partition,
global adapter and EMA are excluded. Backward includes clipping; median columns
need not sum exactly to median wall time. Loss is not a quality metric.

| Structure / codec | Batch | Forward ms | Backward + clip ms | Optimizer ms | Sec/update | Images/sec | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No U-Net / native | 1 | 227.0 | 795.6 | 160.9 | 1.172 | 0.85 | 12.09 | 12.16 |
| No U-Net / native | 2 | 502.4 | 1642.1 | 147.1 | 2.282 | 0.88 | 14.37 | 15.13 |
| No U-Net / native | 4 | 1149.3 | 3484.6 | 149.4 | 4.781 | 0.84 | 20.84 | 22.28 |
| Hybrid / native | 1 | 147.8 | 520.9 | 152.7 | 0.828 | 1.21 | 12.31 | 12.39 |
| Hybrid / native | 2 | 328.5 | 1061.5 | 152.9 | 1.547 | 1.29 | 14.65 | 15.31 |
| Hybrid / native | 4 | 756.9 | 2243.6 | 165.6 | 3.156 | 1.27 | 21.24 | 22.50 |
| No U-Net / Qwen | 1 | 87.7 | 220.9 | 138.4 | 0.453 | 2.21 | 11.59 | 11.76 |
| No U-Net / Qwen | 2 | 108.6 | 317.7 | 144.1 | 0.578 | 3.46 | 11.73 | 11.96 |
| No U-Net / Qwen | 4 | 202.7 | 748.9 | 151.8 | 1.094 | 3.66 | 12.11 | 12.22 |
| Hybrid / Qwen | 1 | 128.4 | 304.4 | 171.9 | 0.594 | 1.68 | 11.79 | 11.99 |
| Hybrid / Qwen | 2 | 146.9 | 295.0 | 183.8 | 0.625 | 3.20 | 11.94 | 12.17 |
| Hybrid / Qwen | 4 | 180.7 | 443.0 | 179.0 | 0.812 | 4.93 | 12.34 | 12.44 |

Hybrid alone gives1.42/1.48/1.51x update speed atB1/B2/B4 but increases peak
allocated storage by0.22/0.27/0.41GiB. The same native weights and optimizer
remain, with35.63M added structural parameters and saved fine-resolution
activations: reducing coarse-token work does not imply lower measured peak.
Qwen's base patch-token grid is1024 tokens versus4096 with the native codec.
Combined versus Qwen-only is31%/8% slower atB1/B2, but1.35x faster atB4: the
extra fusion/checkpoint/optimizer work outweighs the coarse-block saving at
small batch, while atB4 backward falls749→443ms. Combined is1.97/3.65/5.89x
faster than no-U-Net/native for this compute probe. ItsB1→B4 peak11.79→12.34GiB
is dominated by fixed parameter/gradient/optimizer storage, with a smaller
batch-dependent activation contribution. These are measured costs, not quality
acceptance or end-to-end training speed guarantees. Raw samples and host RSS:
`docs/audits/data/hybrid_anima_public_feature_fusion_matrix_2026-10.json`.

The candidate's Qwen VAE/fusion combination also saves and reloads through the
ordinary Anima Full FT adapter and component loader:1006 tensors, bundled
64ch/16x codec, ready spatial/noise bridge and exact hybrid-module digest. The
extracted `model:` codec must be bundled under the existing save contract.
An initial probe incorrectly selected the older frame facade; using the same
`NormalizedImageVAE` as production restores raw codec keys correctly. This was
a probe configuration error, not a backend fix or an owner run change.

Matched seed63/28-step512px generation from the same native body remains a
negative quality result. Native-codec CFG4 retains two back-facing figures and
a separate person/hand/food composition; CFG1 has detail defects. Combined CFG4
has strong gridlike noise and missing/distorted figures. Removing only the
U-Net from the same restored combined model preserves the64ch bridge/codec,
yet swap-only also shows gridlike noise and changed composition (partial
person/food outlines remain). The shared defect therefore cannot be attributed
solely to U-Net; this does not exclude extra damage from its smaller grids.
Prioritize codec/noise alignment before declaring combined quality acceptance.
No combined fine-tuning or equal-time quality comparison was done in this probe.
General-only image hashes and restoration evidence:
`docs/audits/data/hybrid_anima_public_feature_fusion_combined_2026-10.json`.

### Separating small-grid initialization from codec error

Native-codec256px uses a16x16 patch-token grid, the same as Qwen-codec512px.
From the same public step2 native body, no-U-Net native256 generation retains
two back-facing figures and a separate person/hand/food outline. Hybrid256
produces missing figures, stripes and cropped/deformed facial structure. This
separates a small-grid weakness in the512px-fitted U-Net from codec error, but
does not equate pixel size or latent channel representation across codecs.

An initialization comparison starts both arms from exactly the same67 added
tensors and freezes the native body. Single512 and alternating512/256 receive200
updates with identical item/null/sigma/noise seeds, parameter groups and initial
field loss. Sigma is drawn before shape-dependent noise. Native256 inputs are
encoded once from28 general images;512 inputs/conditioning are reused. Held
indices28/31 never enter the fitting data. Rates are1e-4 for51 original non-skip
tensors,2.5e-5 for14 skips and1e-5 for2 full-rank fusion tensors. Both67 optimizer
counters reach200 and frozen native fingerprints match before/after.

Update GPU time is74.694s for single512 and76.934s for mixed grids; own peak is
4.690/4.692GiB. These exclude setup, hashing and evaluation and are not Full FT
memory claims. Last50 teacher-field mean is0.02031 at512 for single-grid;
mixed-grid has0.01987 at512 and0.03440 at256. Differing geometries' loss magnitudes
are not a quality comparison. The mixed arm improves small-grid person/hand
outlines versus single512 and preserves held512 compositions. The256 two-person
scene still fails; neither universal resolution robustness nor default promotion
is established. This is equal-update, not explicitly equal-GPU-time fitting.

Replacing only the U-Net weights in the retained combined artifact restores
more person outline at swapped512px, but gridlike codec texture remains. At
swapped1024px (32x32 patch grid, matching native512px), a matched before/after
comparison restores the left figure in the two-person scene with mixed fitting.
Person/food outlines are recognizable in both1024px arms; substantial gridlike
texture remains. Small-grid initialization is therefore a separate improvement
from codec/noise alignment, not its replacement. No combined Full FT was done
here and no owner artifact/default changed. Next gates are a reusable multi-grid
initialization recipe and codec/IO adaptation with held generation.
Evidence: `docs/audits/data/hybrid_anima_public_feature_fusion_multigrid_2026-10.json`.

### Multi-grid initializer foundations (2026-10-10)

`core/rewiring/resolution_schedule.py` now owns bounded grid validation and
the `round_robin_v1` curriculum: anchor resolution first, then at most three
distinct additional square resolutions, each divisible by64 within64–1024.
Each phase starts a fresh cycle. Grid selection consumes no exposure-selection
RNG; mixed-grid noising accepts a separate timestep generator so changing a
noise tensor's shape does not change subsequent timestep draws.

Initialization lineage accepts the optional paired fields
`additional_resolutions` and `resolution_sampling`. It rejects incomplete,
unknown or empty curricula. Legacy recipes and calibration hashes remain
unchanged; the helper emits these fields only for a real multi-grid plan.
The40-test CPU check covers shape-independent timestep draws, historical
single-grid RNG order, lineage round trips and initializer ownership.

Anima/Ming/Qwen preparation and the public API now consume this infrastructure
as described below. No GPU quality claim, default promotion or owner-run change
follows from the CPU check.

### Public multi-grid initializer wiring (2026-10-10)

`additional_resolutions` is exposed by the initialization request and schema
defaults, with an empty SSOT default preserving single-grid behavior. For
example, `resolution:512, additional_resolutions:[256]` cycles512 then256 on
each fitting phase, without multiplying its update count. Geometry preflight
checks every grid before a job acquires heavy resources. Captured trajectory
targets are partitioned by grid; selection cannot overweight a grid merely
because its sampler captured more forwards.

Anima and Qwen encode captions once and encode images at each requested size.
Qwen predicts and crops by the current example's native image grid, rather
than a fixed anchor size. Ming keeps conditioning, pixels, reference latents
and native-generated Layer canvas/layers associated with each resolution.
Its Design and Layer CPU owners both verify actual mixed-grid fitting and
strict checkpoint restoration, including cancellation/failure rollback.
The timestep stream uses seed+1 only for mixed plans; noise and exposure
selection keep their own seed streams. Single-grid draw order remains intact.

The frontend settings editor receives the field through schema defaults and
submits it with the existing JSON request. Its help explains the ordered grid
cycle and added preparation/host-memory cost. Recorded `resolution_sampling`
is output lineage, not an accepted request option. Frontend validation/errors,
pending actions, polling, cancellation and using the result retain their
existing paths; owner build/type-check remains pending. Official-weight Anima
GPU owner execution is verified below; held-image quality remains a separate
acceptance gate.

### Official Anima multi-grid owner execution (2026-10-10)

The actual `initialize_loaded_anima` owner prepares four verified general
images, excludes the final two from fitting, captures native512/256 trajectories
and executes four data-noising plus four trajectory updates in BF16 with
AdamW8bit/stochastic rounding. The67 public added parameters include full-rank
boundary fusion. All eight hybrid forwards alternate latent64/32, losses stay
finite and the original parameter digest is unchanged. The owner restores
component devices, flags and training modes and removes its temporary modules.

The verified streaming exporter writes4,253,493,072bytes/752tensors into a
private verification directory. Its destination is relocated for this probe;
preparation, fitting and source-byte validation are unchanged. Native loading
reconstructs the mixed recipe and exactly the initialized67 modules. A separate
CPU reload confirms the native parameter digest too. Fresh structural-optimizer
updates on both grids give all67 parameters finite gradients and advance their
counters to2. This is functional acceptance, not optimizer/dataset process
resume, convergence or default promotion. Initializer own peak is4.576GiB,
not Full FT peak. No owner model, DB run or artifact is overwritten.
Evidence: `docs/audits/data/hybrid_anima_public_multigrid_initializer_2026-10.json`.

### Pixel-resize calibration and flow composition diagnostic (2026-10-10)

An equal-data codec diagnostic compares native512 latents area-pooled onto the
target32x32 grid against native256 image encodes on that grid. Both3x3 fits use
the same28 general images and exclude four held items; existing56 native train
encodes are reused. New target32/held native8 encodes are retained locally for
subsequent experiments. Mean held target-latent pair MSE drops0.66309→0.64366
(2.93%); cycle refit drops0.63777→0.61312. Neither is DiT loss or quality proof.

Replacing only bridge kernels and their Gaussian snapshots in the same retained
mixed-grid combined model does not fix generation. Reused controls and new
512/1024px general CFG4 samples use the same body/U-Net, seed63 and28 steps.
The1024px two-person/person-food outlines remain recognizable, but clipped
tones and gridlike texture remain;512px still loses composition. The retained
older calibration is not claimed disjoint from every held item. No default or
artifact is promoted.

For frozen original-channel heads, linear input/output maps A/B, clean biases
a/b and fixed exposure noise e, the current corrected flow has
`z=A*x+(1-sigma)*a+sigma*(G-A)*e` and
`v_x=B*v_z-b+e-B*G*e`. Differentiation gives
`dz/dsigma=A*B*v_z+(I-A*B)*G*e-A*b-a`.
Matching `v_z` requires operator composition and bias closure, in addition to
clean-image reconstruction. Independent fits do not enforce those conditions.

A synthetic FP64 interior-grid check measures source-flow consistency MSE
1.04436 for the retained40-image calibration,1.56858 for the new latent-area
fit and1.15807 for pixel256, against unit source-velocity variance1.00250.
An identity-map positive control gives5.57e-33. This diagnoses a concrete
mapping-consistency gap, not sole causality for visual artifacts or a trained
model's loss. The next bounded prototype should constrain A/B and bias closure
and verify generation before any production promotion; pixel resizing alone
does not meet that gate.
Evidence: `docs/audits/data/hybrid_codec_pixel_resize_calibration_2026-10.json`.

### Constrained affine flow initialization (2026-10-10)

`core/experiments/flow_consistent_alignment.py` implements a bounded CPU
prototype, not a production calibration policy. Streaming paired covariances
fit a pointwise input map A and output map B, then enforce `A*B=I` and
`A*b+a=0`. An optional row-orthogonal input limits inverse amplification.
The existing spatial bridge/pair containers carry these center-only kernels;
their Gaussian noise snapshots and ordinary save/gradient paths are retained.
Channel counts, finite data, grid equality and full row rank are validated.

On the same28 general fit items/four held items, fitted input reduces synthetic
source-flow consistency MSE to approximately1e-14. Its input condition numbers
are22.83 (latent-area) and11.40 (pixel256), with inverse norms46.25/27.92.
Row-orthogonal input has condition1/inverse norm4 and likewise closes the
algebra, but increases clean reconstruction error. CPU pair tests cover
FP32/BF16, image/frame shapes, boundaries, gradients and strict restoration:
38 tests pass together with the Gaussian-alignment suite. These are numerical
and contract checks, not image-quality acceptance.

Actual combined-checkpoint inspection confirms added input/output channels
16:64 are zero; the input padding-mask channel is excluded. Thus the frozen
original-head assumption holds for these generation comparisons. The same
body/mixed U-Net, held items, CFG4, seed63 and28 steps are used at512/1024px.
Fitted-input generation has strong colored clipping/fragmentation. The
orthogonal pixel256 variant retains two-person/person-food outlines at1024px,
but gridlike fragmentation and clipped tones remain. Exact flow composition
does not establish that the projected clean/noisy states lie on the native
or target codec manifold. Neither variant is promoted, and no owner artifact
or production default changes. Further quality work must assess that state
distribution and trainability rather than treating closure alone as success.
Evidence: `docs/audits/data/hybrid_codec_flow_consistent_2026-10.json`.

### Spatial output under exact flow closure (2026-10-10)

Held clean-latent statistics distinguish the previous variants: native mean
channel standard deviation is0.37090; fitted/orthogonal projected inputs are
0.33361/0.75127. Target is0.94181, while fitted/orthogonal round trips are
0.40109/0.72945. Thus the orthogonal-input variant changes clean-state scale
substantially; exact noisy-flow closure alone does not preserve it.

The experimental calibrator now accepts `output_kernel_size=3`. Input remains
pointwise. Output uses unfolded native neighborhoods and constrains each output
kernel slice: `A*B_center=I`, `A*B_off_center=0`; bias closure remains unchanged.
This permits neighboring information in the target-space null directions
without source-flow leakage, including padding boundaries. Existing bridge
containers, parameter counts and runtime operators are unchanged. A planted
neighbor-feature fixture verifies spatial recovery and multiple-frame ordering;
Gaussian/pair tests cover both kernel choices and FP32/BF16 gradients/reload.
48 tests pass, with compile and real import verification and no CUDA context.

Pixel256 held round-trip MSE drops0.75180→0.62672 (16.64%) versus the pointwise
constrained variant; direct native-to-target MSE remains0.86306, worse than
the independently fitted spatial baseline0.68722. Round-trip channel standard
deviation rises to0.52143, still below target0.94181. Improvements are not
universal across these different measures.

Matched frozen-body512/1024px generation retains the coarse two-person and
person/food outlines at1024px, but clipping, fragmentation and colored spots
remain. Output latents are finite (RMS0.97–1.16, max absolute4.38–4.75), which
does not prove decoder-manifold compatibility. No production default is
changed. The next quality comparison must include short training of the
resized IO/bridges rather than infer post-training behavior from this frozen
head; use equal exposure/GPU-time controls and keep native-body stability
separate from codec adaptation. Evidence:
`docs/audits/data/hybrid_codec_spatial_flow_consistent_2026-10.json`.

### Matched IO/bridge adaptation after spatial calibration (2026-10-10)

Two arms start from the same retained public-step2 body and mixed-grid U-Net;
only the28-image pixel256 calibration differs (independent spatial versus
flow-constrained spatial). Existing target512 latents and TE conditions are
reused. Both train the same six IO/bridge tensors (1,130,624 parameters) for200
updates at1e-4 using FP32 storage, BF16 frozen body/compute and AdamW8bit.
All six receive finite gradients every update; clipping uses norm1. Body and
U-Net parameter hashes remain exact, all six optimizer counters reach200,
and item/null/sigma/noise exposure digests match. Null probability is0.1;
sigma sampling is uniform over0.01–0.99 before BF16 rounding.

Independent/constrained fitting uses49.531/44.968 wall seconds with own peak
4.154/4.156GiB. These include probe synchronizations and exclude encoding,
checkpoint export and evaluation, so they are not end-to-end training speeds
or Full FT VRAM. First25/last50 mean losses are1.3275/1.1510 versus
2.3879/2.2010. Shared inputs permit loss comparison; lower loss alone does
not prove held generation quality. Both1024px held conditions retain outlines
but strong fragmentation/clipping remains. The constrained arm retains or
increases colored spots and does not show a quality advantage. Four512px
outputs are retained locally; only one independent512px image was inspected.

A post-fit interior white-noise operator check quantifies mutable bridges
against their fixed Gaussian snapshots. Independent/constrained native-channel
noise standard deviations span0.99365–0.99951 /0.98798–0.99358; maximum
covariance errors are0.01265/0.02390. Output residual drift mean variances
are0.000261/0.000181. The drift exists but is small in this bounded probe;
it is not evidence that snapshot staleness alone causes the severe artifacts.
Do not change the production Gaussian policy on that assumption. Further
codec work should control inverse amplification and generated-state mismatch;
short IO-only adaptation did not establish combined quality. No owner run,
checkpoint or default changed. Evidence:
`docs/audits/data/hybrid_anima_spatial_codec_io_fit_2026-10.json`.
