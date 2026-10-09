# Hybrid U-Net ssDiT foundation

Status: experimental native Anima/Ming/Qwen 2.1 integration; CPU contracts and
synthetic official-weight Anima backward verified. Quality acceptance is pending.
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
run or default preset is converted. Broader native census coverage, activation-memory
accounting, GPU/native full-trainer
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
