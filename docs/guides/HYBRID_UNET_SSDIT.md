# Hybrid U-Net ssDiT foundation

Status: experimental native Anima integration; CPU acceptance only.
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
Hybrid execution currently accepts single image grids only; temporal grouping
is rejected. Architecture-owned callbacks still own native blocks and positions.

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
recognize v2. Explicit topology JSON can select it for Anima full-parameter,
full-frame training. Frozen-base training and partition/global training are
rejected until their artifact and convolution-halo contracts are implemented.

The graph shows compressed learned skips with dashed yellow lines. Hybrid
region editing is temporarily disabled to preserve dependent module settings;
the complete hybrid editor and preset selection remain pending. No existing
run or default preset is converted. Parameter census, GPU/native full-trainer
optimizer/EMA/offload acceptance, partition halo, Ming/Qwen integration and
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

The training-versus-generation behavior observed in the previous Anima
VAE-swap run still needs independent diagnosis. Good single-step debug
reconstruction is not evidence of good free-running generation.
