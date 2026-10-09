# Hybrid U-Net ssDiT foundation

Status: experimental native Anima/Ming/Qwen 2.1 integration; CPU acceptance only.
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
as zero. An unavailable native census remains explicitly unavailable. Hybrid
region editing is temporarily disabled to preserve dependent module settings;
the complete hybrid editor and preset selection remain pending. No existing
run or default preset is converted. Native census coverage, activation-memory
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

After implementation, rerun the owner's four-way comparison: native/hybrid
U-Net crossed with native/Qwen-Image 2.1 VAE. The previous batch comparison used
Anima base v1.0, BF16, AdamW8bit, 24 checkpointed native blocks, 1024px and batch
1/2/4 on RTX 6000 Ada. It used synthetic latents and excluded codec/TE/IO and
partitioning; its VAE swap did not include real calibration. Keep comparable
compute measurements, but include the production bridge in the new swapped
conditions and state calibration provenance. Report forward, backward, update,
images/sec and peak allocated/reserved VRAM; separate preprocessing timings
from the training step. Results are pending and no speedup is assumed.
