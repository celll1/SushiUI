# DiT residual rewiring contract

Status: accepted implementation contract. Canonical topology validation and
side-effect-free preview APIs are implemented. Qwen-Image 2.1 supports native
checkpointed residual execution, metadata-driven generation, LoRA/full training,
checkpoint restore and adapter-stack compatibility checks. Actual 1024px API
workflows verify generation, training samples, saved artifacts and resume, both
resident and with block swap. The graphical editor is connected and statically
reviewed; the owner will perform consolidated frontend checks. Krea 2 has native
execution/artifact/training/LoRA generation integration and CPU contract checks.
GPU acceptance is restricted to Qwen-Image 2.1 by owner instruction; other models
receive provisional implementation and CPU verification. Further architectures
still require native execution/artifact adapters and capability gates.

## Default topology

A new user-selected U-Net conversion defaults to two nested regions with four
blocks at the deepest level. This does not automatically convert ordinary models
or replace the topology declared by an existing checkpoint.

For Qwen-Image 2.1's 32 main blocks, block indices are zero-based half-open:

| Region | Block interval | Spatial downsample relative to parent |
|---|---|---|
| Outer | `[4, 28)` | 2 by 2 |
| Inner | `[14, 18)` | 2 by 2 |

At 1024px, eight blocks use 4096 image tokens, twenty use 1024 tokens, and four
use 256 tokens. The latent input/output grid and text tokens retain their native
meaning. The default block allocation is:

| Zero-based blocks | Image token grid at 1024px | Tokens per block | Blocks |
|---|---|---|---|
| `[0, 4)`, `[28, 32)` | 64 by 64 | 4096 | 8 |
| `[4, 14)`, `[18, 28)` | 32 by 32 | 1024 | 20 |
| `[14, 18)` | 16 by 16 | 256 | 4 |

```text
blocks    0..3    4..13    14..17    18..27    28..31
tokens    4096    1024       256      1024      4096
outer skip        [----------------------)
inner skip                  [-----)
```

Each skip connects its region entry to its exit, restoring that region's residual
update. The outer region is `[4, 28)` and the inner region is `[14, 18)`.
The 1024 values above describe image token counts, not image pixels.

This allocation retains more blocks at 1024 tokens than the sixteen-deepest-block
probe. Its image-token-linear work is approximately 41.40625% of dense execution;
this is not a full-model FLOP or throughput estimate. Fixed pooling and upsampling
add no parameters, so checkpoint parameter size does not shrink. The measured
quality improvement is partial and still requires calibration.

Other architectures must resolve valid model-specific intervals before
registering a default; do not reuse these indices without validation.

## Execution

Each region preserves its input image hidden states and executes its blocks on
a pooled copy. Restore only the update:

```text
low_in = pool(hidden)
low_out = execute_region(low_in)
hidden = saved_hidden + upsample(low_out - low_in)
```

Use fixed average pooling, bilinear upsampling with `align_corners=False`, and
identity delta addition initially. Each nested region performs its own subtraction
and restoration. Preserve text updates, native block order, masks, positions,
conditioning, gradient checkpointing, and offload ownership. Region context is
per forward and must not depend on a mutable global skip stack.

Allow nested or disjoint regions; refuse crossing regions, cycles, reordered
blocks, unsupported stream transitions, and indivisible grids. Rebuild coarse
positions using the architecture's coordinate contract. Restore the original
grid and context before final output projection.

## Configuration and artifacts

All API defaults and presets belong to `backend/api/param_defaults.py`.
Configuration is architecture-neutral, with per-architecture generation and
training capabilities. Do not add rewired variants to ModelType/ARCH_REGISTRY.

Store a versioned canonical JSON topology, its SHA256, architecture, stream block
counts, block-layout/config identities, intervals, parent regions, operator and
position rules, and any learned topology tensors. Safetensors metadata values
are strings. Single files, shard indices, and directory artifacts must agree.
Unknown versions, incompatible layouts, missing tensors, or conflicting metadata
must fail before mutating the active model. Metadata cannot execute Python.

Generation restores the checkpoint topology automatically and shows it read-only.
Training can inherit a topology or configure a new conversion. Resume requires
agreement between run configuration and checkpoint topology. LoRA artifacts must
carry their required topology and base identity; incompatible adapter stacks must
be refused atomically. Preserve existing adapter capability tables.

## Editor and acceptance

Provide a shared graphical topology renderer: block index on the horizontal axis,
resolution levels on the vertical axis, draggable interval endpoints, and explicit
skip edges. Show image pixels, latent grid, token grid, per-level counts, and
validation errors. Backend-resolved plans are authoritative for editor previews.
Keep presets, YAML editing, clone, export, and queues consistent.

The uncalibrated Qwen probes show blur and granular artifacts. Shortening the
deepest region from sixteen to four blocks improves some shapes but does not
establish acceptable quality or universal performance gains. Calibration and
fine-tuning quality, full/LoRA save-load-resume, production dtype gradients,
checkpointing/offload, API round trips, and per-architecture gates remain required.
Do not infer reduced checkpoint/optimizer size from reduced token computation.

The method was assessed from
[lodestone-rock/residual-to-unet](https://github.com/lodestone-rock/residual-to-unet/tree/4834458ea1e2dbede89e776ebccbe23233d10fd0).
Its seven-layer linear reference is not a real DiT quality or speed benchmark.
The SushiUI probe is independently written from the residual-update formulation.

## Current verification boundary

`GET /api/v1/schema/dit-rewiring` supplies presets and
`POST /api/v1/models/rewiring/validate` returns a validated grid/skip plan.
These endpoints do not enable model execution; runtime capabilities remain empty
until model loading and training requests enforce the complete artifact contract.

The internal Qwen forward currently accepts a single uncached 2D target grid
with native causal modulation. References, cached execution, partition positions,
input residuals, and style contexts are refused. Native block execution and its
gradient-checkpointing call remain in charge of attention and recomputation.
CPU verification covers an independent nested residual output/gradient oracle,
FP32/BF16/FP16 identity regions, local context restoration, actual Qwen blocks,
and the full Qwen forward with backward. Production GPU dtype/offload and artifact
save-load-resume verification remain acceptance gates.

Full Qwen transformer artifacts and training bundles now restore validated topology
metadata; full and LoRA writers record it. Native config identity excludes loader
annotations and includes every Qwen constructor setting with resolved defaults.
Shard index and all referenced shard headers must declare the same topology.
Tiny actual Qwen full-writer/load-transformer round trips cover single and sharded
files, exact parameter restoration, numerically matching forward output, config
mismatch, and a tampered shard. LoRA stack application, full-size GPU artifacts,
training-bundle companion restoration, and training resume remain unverified.

Training accepts `dit_rewiring_mode` (`inherit` or `configure`) and an optional
`dit_rewiring_spec`, with defaults in `TRAINING_DEFAULTS`. Configure with no spec
resolves the native-layout preset after loading; inherit with a spec asserts exact
agreement. API requests and YAML preflight validate the structure. Qwen component
loading installs the resolved topology, and LoRA resume checks its metadata before
loading weights. Full resume prohibits topology edits. Sampling a rewired training
model disables KV caching. Current gates refuse partition training/global adapters,
TREAD, REPA, and block skipping. CPU contracts cover request/YAML parity, default
resolution, inheritance, resume comparison, and refusal without model mutation.
Actual trainer steps, persisted run/resume state and UI round trips remain pending.

The training editor uses `POST /api/v1/models/rewiring/inspect` to obtain native
config/layout identities, saved topology, and the resolved preset from file headers.
It previews pixel/latent dimensions and per-level block/token counts through the
validation API. The shared SVG renderer draws the resolution sequence and residual
skip arcs; draggable boundaries, numeric endpoints, per-axis factors, parent
selection, nested-region addition and subtree removal edit the same request spec.
`PARAM_KEYS` carries both settings through clone, presets and run restoration.
The backend inspection API is tested without tensor loading or model construction.
Frontend build/type checking remain owner-run; browser/editor validation is pending.

Generation preflights the entire Qwen LoRA stack before branch installation.
Declared native config and topology must agree across every file and with a
rewired full checkpoint. A legacy adapter with no topology declaration denotes
the native sequence and cannot be mixed with rewired adapters. On a dense base,
a consistent rewired stack temporarily installs its topology; unload restores
the base using model-specific weakref ownership. Invalid structure/shape stacks
leave the model untouched. Generation exceptions also unload branches/topology.
Rewired generation excludes partition/reference/style workflows, validates the
actual grid, and disables KV caching. PNG/gallery parameters record the effective
spec/hash and cache policy. Loaded full-model topology is shown read-only through
the shared graph in the model selector. Real AdapterSession CPU contracts cover
these installation/refusal/lifetime paths; denoising in this test is a stub.
Production-path Qwen GPU generation is verified below; frontend rendering is pending.

## GPU verification (2026-09-30)

The formal metadata-driven Qwen path generated 1024px/40-step output through actual
in-process FastAPI routes on an RTX 6000 Ada. Its output is pixel-identical to the
prior deepest-four-block probe. PNG spec/hash/cache metadata and topology cleanup
were verified. Full request time was 19.27s; peak allocated memory was 9.21 GiB.
This is a single measurement including prompt encoding and VAE decode.

The actual transformer completed BF16 LoRA steps at 4096 image/16 text tokens,
rank 4, 128 attention adapters, with all 32 blocks checkpointed and bounded INT8
prefetch backward. The production checkpoint writer and LoRATrainer loader
restored saved parameters exactly. With deterministic CUDA/math SDPA, checkpoint
plus optimizer resume reproduced the next loss, gradients, and updated parameters
bit-for-bit. Default automatic attention yielded matching resumed loss but small
update differences; its bitwise determinism is not established. Warm synthetic
steps took 1.23–1.30s with automatic attention and 2.38–2.45s with deterministic
math attention; these exclude dataset/encoder/decoder/scheduler workflow.

LoRA resume now delays topology installation until weight validation/loading
succeeds. A regression test rejects an incomplete checkpoint while preserving
native topology and configuration.
Broader dataset/resume matrices, calibration quality, frontend acceptance,
and additional architecture adapters remain outstanding. The baseline dataset/resume workflow is verified below.

## Diffusers directory restoration verification

Native Qwen Diffusers directory loading reads transformer safetensors headers
before allocating weights, validates the topology against the native config,
and restores it after the transformer loads. Inspection and loading share the
same directory declaration check. Conflicting files or shards are refused.
A small native transformer round trip verifies exact parameter restoration,
topology restoration, and config mismatch refusal before loading companions.
The residual rewiring CPU suite now passes 17 tests.

## Actual dataset/API training verification

The isolated worktree API completed a one-image 1024px LoRA run at two steps,
then updated the same run and resumed its latest checkpoint to step three.
FP32 rank-4 adapters, BF16 compute/model weights, native attention, 32 checkpointed
blocks, ConvRot prefetch and VAE/text caching were exercised. Step-zero and
periodic training samples completed. Configuration retrieval preserved the
conversion selection; all three saved checkpoint topologies agree, their tensors
are finite, and the resumed step changes adapter weights.

The real resume restored all 256 optimizer tensors without fresh states, the
training position, and RNG state. Extending the target steps issued the expected
scheduler-continuity notice. Both subprocesses exited successfully and the API
reported completion at the requested step. This is a baseline workflow check,
not a calibration or bitwise resume-equivalence result.

## Block-swap verification

The actual dataset/API LoRA workflow also completed two steps and resumed to
step three with 30 swapped blocks and two GPU ring slots. Both processes
registered 90 conductor lifecycle hooks, used all 32 gradient-checkpointed
blocks and fused AdamW8bit ringbuffer updates, and completed training samples.
ConvRot used transient BF16 backward; its cached/prefetch backward mode remains
incompatible with swap by the existing Qwen contract. Checkpoint tensors are
finite and topology-consistent, optimizer state is restored, and resumed updates
change adapter weights. This does not establish bitwise swap/resident equality.

The probe exposed a pre-existing LoRA runner gap: swap settings were saved in
YAML but not passed to the trainer constructor. The runner now forwards swap
count, pinned-memory selection and optimizer-group count using API defaults.

## Full-parameter and saved-model inference verification

Actual isolated API workflows completed full-size Qwen training at step one and
resumed to step two. All 297 trainable tensors (7,115,124,736 parameters) used
BF16 after explicit INT8 materialization, native attention, 32 checkpointed
blocks, 30 swap layers/two GPU slots and fused AdamW8bit ringbuffer updates.
TE/VAE remained frozen. Both processes exited zero and their training samples
completed. Each saved model comprises four shards and an index, about 14.23 GB
of weight files. All topology declarations agree and all saved tensors are
finite; the resumed step changes 235 tensors. All 297 optimizer tensors restore
without fresh states, and training/RNG position restoration is exercised.

The step-two full checkpoint also completes actual 1024px generation through
API loading with no LoRA or explicit topology request. Loaded-model info and
PNG metadata report the restored spec/hash; KV caching is disabled and the base
topology remains installed afterwards. Missing Qwen file dispatch and single-file
metadata detection in ModelLoader were fixed; regression tests use real saved
single/sharded artifacts. The schema API now reports Qwen as an implemented
backend architecture through a shared capability declaration. Seventeen CPU
contract tests, real imports and OpenAPI capability checks pass.

These short runs prove workflow connectivity; they do not establish quality
calibration, TE joint training or bitwise equivalence under automatic attention.

## Consolidated frontend checks for the owner

Frontend build/type checking and browser checks are deferred to the owner at the
end, per the owner's instruction; agent verification is static review.

- Create a Qwen LoRA/full run and select Configure / Model preset. Confirm the
  8/20/4 block allocation at 4096/1024/256 image tokens for a 1024px preview.
- Drag skip endpoints and edit numeric boundaries/shrink factors. Confirm the
  graph, counts and interval labels update after validation and dragging stops
  on pointer release.
- Enter crossing intervals, a parent cycle or incompatible dimensions. Confirm
  the error appears and invalid graphs disappear; restore the preset to recover.
- Save, reopen, copy and preset a configured run. Confirm mode/spec survive.
  Open an older run without topology keys and confirm API defaults populate
  without overwriting its saved values.
- Change the base model and confirm config/layout mismatch is reported until a
  compatible topology is selected.
- Load a rewired full checkpoint and confirm the model selector's read-only
  topology graph matches its metadata. Generate and inspect effective topology
  in gallery/PNG parameters, including automatic KV-cache disablement.
- Apply compatible rewired LoRA files and confirm their temporary topology is
  removed on unload; incompatible stacks must report a useful refusal.
- Confirm ordinary models retain their native sequence and unsupported
  architectures do not offer an unverified conversion option.

## Krea native execution foundation

The Krea 2 native single-stream forward has an internal residual adapter. It
validates the config identity and origin-based row-major image coordinates,
preserves the text prefix/padding mask, rebuilds coarse RoPE and masks, and
calls native blocks through their existing checkpointing path. Style injection
and REPA taps are refused. No parameters are added, and ordinary forwards retain
the original block loop.

Three CPU contracts exercise a seven-block native GQA model with nested regions:
image/text token counts and mask preservation, checkpointed output/gradient
equivalence, and incompatible coordinates/config/REPA refusal. This is a native
execution implementation with artifact persistence. Krea is exposed through the
runtime capability list following native CPU contracts; it has no GPU acceptance
claim. Qwen is the sole GPU-verified conversion architecture.

Krea full checkpoints now save the canonical topology and restore it from single
files, shard indices and transformer directories. Header/config validation runs
before tensor loading. Export metadata cannot replace the installed topology;
LoRA checkpoints also carry the installed topology. Two additional CPU contracts
use actual writers/loaders to verify bit-identical weights and output after
single/sharded round trips, dense compatibility, directory restoration and
pre-allocation refusal of mismatched config.

The shared training resolver now owns inherit/configure/resume agreement and
side-effect-free validation for both Qwen and Krea. Krea component loading and
LoRA resume use it; topology installs only after all adapter tensors validate.
Generation LoRA sessions validate the entire stack before mutation, install its
required topology temporarily, and restore the base topology on unload/cleanup.
The shared stack validator retains Qwen's existing partition-global refusal.
All three Krea generation entry points validate grid divisibility, refuse styles
and attach effective topology metadata.

Two additional CPU contracts exercise actual Krea training adapter checkpoint
loading and generation AdapterSession behavior: failed resume leaves tensors and
config unchanged, successful resume restores tensors/spec, incompatible edits
and features fail, dense/rewired stacks fail atomically, and base topology returns
on unload. Krea configure/resume tests now use the actual public capability gate.
These are not full Krea GPU workflow or performance/quality results.

Krea's 28-block preset is outer `[4, 24)` and inner `[12, 16)`: eight blocks at
4096 image tokens, sixteen at 1024 and four at 256 for native 1024px output.
Its image-token-linear work is 43.75% of dense; no measured speed or quality is
claimed. The preset is owned by param_defaults.py. Model inspection supports
Krea metadata files/shards and native full/transformer-only directories without
tensor allocation. The training editor uses the backend capability list instead
of a Qwen-only condition; Krea loaded-model info includes restored topology.
Two further CPU checks verify header-only inspection through actual API routes,
malformed config refusal, the 8/16/4 preset and request/YAML/runtime resolution.
Twenty-six aggregate CPU contracts pass. Per owner instruction, further GPU
checks use Qwen only; Krea UI receives static review and consolidated owner checks.

## GPU verification scope

The owner explicitly limits GPU verification to Qwen-Image 2.1. Krea and subsequent
models receive provisional integration and CPU contracts only. Do not run their
GPU generation or training to promote a capability. Record any absent real-model
GPU/performance/quality evidence explicitly instead of inferring it from Qwen.

## Qwen regression after shared resolution changes

A fresh actual 1024px/40-step generation through isolated FastAPI routes passes
with the shared training/adapter resolvers and expanded inspection/capability
implementation. Its metadata-driven default topology disables KV caching,
executes all forty forwards and restores the dense base after LoRA unload. The
output is pixel-identical to the previous accepted deepest-four-block probe
(MAE zero). The cold request took 30.67 seconds; this is a connectivity/regression
check, not a new steady-state throughput benchmark. No Krea GPU run was performed.
