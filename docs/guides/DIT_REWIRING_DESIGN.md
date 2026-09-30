# DiT residual rewiring contract

Status: accepted implementation contract. Canonical topology validation and
side-effect-free preview APIs are implemented. Qwen-Image 2.1 supports native
checkpointed residual execution, metadata-driven generation, LoRA/full training,
checkpoint restore and adapter-stack compatibility checks. Actual 1024px API
workflows verify generation, training samples, saved artifacts and resume, both
resident and with block swap. The graphical editor is connected and statically
reviewed; the owner will perform consolidated frontend checks. Krea 2, Anima, Z-Image and Flux2
have native execution/artifact/training/LoRA generation integration, model/editor
inspection and CPU contract checks.
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

## Graphical custom topology construction

Inspection now returns an explicit `initial_spec`, a single editable region whose
edge-block margin and shrink factors come from param_defaults.py. This is separate
from `default_spec`: configure-with-null retains the accepted two-stage preset,
and nonstandard block counts still require the user to select an explicit layout.
A custom seven-block artifact is inspected, resolved for training and previewed
through actual API routes without model allocation; all 27 CPU contracts pass.

The editor can create the first region without a model preset, reset to a custom
single region, and add independent or nested regions into available noncrossing
intervals. Region deletion keeps at least one region. Model/parent/grid validation
still determines whether the edited layout is usable. Graph rows show depth when
same-depth regions have different token counts; tooltips and counts retain the
actual grids. Block widths remain positive for large declared block counts.

Owner frontend checks should also cover custom layouts without presets, adding
independent regions with different shrink factors, resetting after a base-model
change, and removing a parent while retaining another region. These controls
receive static review; frontend build/type and browser checks remain deferred to
the owner's consolidated verification. No additional GPU check is needed for
this header/editor change, and no other model is run on the GPU.

## Anima native execution foundation

Anima now has an internal image-only residual adapter for its native five-axis
hidden grid. The shared runtime pools the image stream; native 3D RoPE is rebuilt
for each coarse grid, cross-attention context and AdaLN conditioning remain
unchanged, and the full grid is restored before the native output head/unpatchify.
Each native Block owns its existing checkpointing mode. The inference offloader's
wait/submit order is preserved; no parameters are added. Native config identity
covers constructor geometry, including the explicitly retained mlp_ratio, while
attention backend selection remains independent.

The initial boundary refuses multiple frames, temporal patches, extra absolute
positions, style injection, FBCache, TREAD, block skipping/dropout and REPA. Three
CPU contracts use a randomized seven-block native model to check grid/RoPE shape,
unchanged cross-context, offloader ordering, finite checkpointed output/gradient
equivalence and feature/config refusals. CUDA initialization is stubbed for the
whole CPU test, including native forward's late imports. All 30 aggregate contracts
pass; no Anima model weights or compute run on the GPU.

Anima's execution foundation is connected to artifact restoration and the
training/generation scopes and editor discovery described below. It carries no
real-model GPU, performance, quality or activation-offload equivalence claim.

## Anima artifact restoration

Full-parameter saves now contain canonical native geometry and residual topology;
LoRA saves include topology metadata. The loader validates header declarations and
config identity before constructing the DiT, and component loading preflights
before allocating checkpoint tensors. Native geometry retains cross-attention
width as well as MLP ratio, allowing nonstandard CPU fixture models to round-trip.
Legacy files without topology/config retain the shipped native geometry;
programmatic in-memory state-dict callers without a backing file remain supported.

The actual full writer/native loader round trip preserves all tensors and inference
output exactly on a randomized seven-block CPU model. Shared sharded export and
native loader also restore topology/weights, including the sibling-index fallback.
Modified config is refused before model construction. All 31 rewiring contracts
pass, with the expanded four-test Anima suite also passing separately. Existing
Anima scaled/scaleless quantized-loader matrix regression passes, and changed modules
compile/import without CUDA initialization. Training/generation API topology scopes
and public discovery are connected as described below. No Anima GPU inference/training
is run.


## Anima training and generation topology scopes

The native model now exposes constructor geometry to the shared training and
adapter resolvers. Component loading resolves inherited topology after latent
space selection; LoRA resume validates topology before tensor mutation and
installs it only after the complete checkpoint validates. The 28-block
preset is outer `[4, 24)` and inner `[12, 16)`, retaining eight full-grid,
sixteen middle-grid and four deepest-grid blocks. Presets remain owned by
param_defaults.py; an available preset does not by itself advertise runtime
support.

Generation AdapterSession validates the selected topology stack before applying
weights, temporarily installs its topology and restores the base topology on
unload or cleanup. All three Anima generation entry points attach effective
spec/hash metadata, validate grid divisibility and refuse style injection and
active FBCache before denoising. Loaded-model information includes the restored
topology. These scopes have CPU coverage; Anima is now exposed through the public
configure capability and model/editor inspection described below. Configure and
resume contracts use the public capability without a test-only override.

Three additional CPU contracts cover the actual training adapter resume path,
atomic failure, generation LoRA apply/unload and incompatible stacks, effective
metadata/feature gates, and the preset allocation. All 34 rewiring contracts
pass. Existing Anima LoRA and refusal regressions yield 29 passes and one known
baseline failure: generate_img2txt lacks an error-context attachment. The same
failure is present in the unchanged base commit; this conversion change does not
edit that route. Changed backend modules compile and import with CUDA untouched.
No Anima GPU inference, training, speed or quality claim is made. All further
real-model GPU acceptance remains limited to Qwen-Image 2.1 by owner instruction.


## Anima discovery and public CPU integration

Native constructor defaults live in a lightweight config module shared by the
model and topology adapter. Header/key and split-path discovery also lives outside
the model loader. Inspection reads Anima DiT files, shard indices (including the
sibling-index fallback), and official split-files directories without allocating
a model or reading tensor contents. Legacy native files use the shipped constructor
geometry; saved full checkpoints use their declared geometry. Adapter-only files
are refused. Anima joins the runtime capability list based on CPU contracts only,
following the owner's provisional-integration boundary.

Actual API inspection covers files, split directories, shards, restored metadata
and custom seven-block initial layouts. The public training request/YAML resolver
selects the standard 8/16/4 preset. A separate fresh process preloads API defaults
with CUDA initialization blocked, then permits only header methods while forbidding
GPU discovery/initialization; inspection passes without importing Anima's native
model or loader. Nonstandard patch geometry reports its actual token stride.

Loaded-model information includes dit_rewiring_token_stride. The read-only graph
uses this native geometry instead of assuming sixteen pixels per token; the
training editor already consumes inspection geometry and the public capability
list. Frontend receives static review only. Owner checks should include Anima
preset selection, reopening a custom run, and loaded-checkpoint previews with
nonstandard patch size. GPU inference/training and performance/quality remain
unverified for Anima.

All 36 rewiring contracts pass, as do eighteen existing Anima LoRA/VAE-resume
checks and the Anima scaled/scaleless quantized-loader matrix. Changed modules
compile/import without CUDA initialization, and OpenAPI parses. Qwen remains the
sole architecture with actual GPU generation/training/resume acceptance; no
additional GPU run is needed for header discovery or read-only preview geometry.

Positive stochastic block dropout is now refused by the shared request/YAML/runtime
feature gate, before model loading or adapter mutation. This closes the gap between
the planned feature boundary and Anima's native forward refusal.


## Z-Image native execution foundation

Z-Image has an internal residual adapter for its main joint block stack. Native
noise/context refiners retain their full sequences. The actual image prefix is
pooled; learned image-padding slots, caption-padding slots, caption tokens and
ragged-batch padding remain outside the spatial grid and evolve through every
joint block. The adapter transports the image prefix to the shared runtime's
image-tail layout, restores native order for each block, and returns the full
native sequence before the output head. Coarse RoPE retains each caption's native
temporal offset and uses integer spatial positions; auxiliary frequencies/masks
remain unchanged.

The provisional boundary supports single-frame images with one grid per batch,
including different caption lengths and image padding. It refuses NAG/NegPip,
style injection, FBCache, TREAD, stochastic block skipping and REPA contexts.
Native main-block checkpointing and offloader wait/submit ordering are retained.
Constructor geometry is exposed without adding parameters. Absolute helper
imports and checks against the actual attention class also support the loader's
standalone module identity.

Rewired training through BatchedZImageWrapperOptimized delegates to the native
list forward, giving generation and training identical learned-padding and
position semantics. Its ordinary dense execution retains the optimized path.
A fixed caption-width expansion in that dense path now uses the input's actual
width, allowing the nonstandard CPU fixture geometry; the standard 2560-wide
model's expression remains equivalent. This conversion does not establish GPU
activation-offload equivalence or throughput.

Six new CPU contracts use a randomized seven-block native model: image and
auxiliary token counts/masks/frequencies (with ragged captions and image padding),
checkpointed output/gradient equivalence, wrapper/native equality and gradients,
feature/grid/config refusals, file-loaded alias compatibility, and an independent
constant-residual update that preserves every token's order. All 42 aggregate
rewiring contracts pass. Five existing Z-Image LoRA round-trip checks and the
scaled/scaleless loader and quantized Comfy-layout refusal checks pass; changed
files compile/import with CUDA untouched.

This remains an internal foundation with artifact restoration described below.
Training configuration/resume, generation LoRA topology scopes and model/editor
discovery must be connected before Z-Image joins the public runtime capability list. No
Z-Image GPU generation/training, performance or quality evidence is claimed; GPU
acceptance remains restricted to Qwen-Image 2.1.


## Z-Image artifact restoration

Full-parameter saves now persist canonical constructor geometry and the installed
residual topology; LoRA saves carry the topology metadata. Saved full geometry
supersedes the companion repository's transformer config on load. Header-only
preflight validates topology/config before companion downloads or tensor reads,
and the native builder validates the resolved layer/channel geometry before
constructing a model, then restores topology after strict weight loading.

Single files, shard indices and the sibling-index fallback use the shared reader
and topology declaration agreement checks. Training's file dispatcher also accepts
the index path. Native embedding-bank keys determine spatial and temporal patch
factors when inferring channels; the previous two-element-list heuristic could
misread a multi-bank geometry. Legacy files without topology/config retain their
native dense path and companion-config/shape reconstruction.

An additional CPU contract drives the actual full writer, native loader and
component-loader config selection. Single/sharded round trips preserve every
weight exactly and FP32 inference within rtol 1e-5 / atol 2e-7; the first single
round-trip comparison differed by at most 1.49e-7. The component-loader check
stubs downloads and stops before companion allocation, confirming saved geometry
is selected even when the companion config differs. A real two-bank fixture
also preserves its channel count. Invalid/missing topology config is refused
before downloads or model allocation. LoRA metadata and dense legacy compatibility
are covered. All 43 aggregate contracts pass, as do the existing Z-Image quantized
loader matrix/refusal checks and changed-module compile/import checks without CUDA
initialization. Public training/generation scopes and editor discovery are still
pending for Z-Image; it carries no GPU inference/training or quality claim.


## Z-Image checkpoint training inheritance

The component loader resolves topology after latent-space changes and before
attention/checkpointing setup and weight freezing. It unwraps the batched training
wrapper, so the native inference and training model share one installed topology.
The LoRA resume path validates metadata without mutation before reading/applying
branch weights, then installs topology only after every branch succeeds.

Internal native resolvers may inherit a topology from a restored model or a
validated resume artifact even before public configuration support is advertised.
This exception applies only to `inherit`; request-level validation and new
`configure` requests retain the runtime capability gate. An explicit inherited
spec still has to match the model/resume declaration, constructor identity and
feature constraints. It cannot create a new topology on a dense model.

CPU contracts cover repeated wrapper inheritance, incompatible block dropping,
actual component-loader ordering and freezing, and 36 real LoRA branches restored
from a saved file. Missing branch weights leave weights/config/topology untouched;
a dense resume declaration against a rewired base is rejected. Public Z-Image
configuration/defaults, generation LoRA topology lifetime and editor discovery
remain pending. All 45 aggregate CPU contracts and changed-file compile/real
imports pass with CUDA untouched. No Z-Image GPU workflow or performance/quality
claim is made.


## Z-Image generation adapter scopes

Z-Image resolves every selected LoRA's topology against the native constructor
and the loaded base before the shared adapter session mutates any module. Missing
metadata declares a dense topology; dense/rewired mixtures and disagreeing specs
are refused. A weak reference and a copied base declaration restore topology on
unload, including model swaps; a replacement model cannot inherit the old model's
restore snapshot.

Txt2img, img2img and inpaint now unload the previous session, apply this request's
files, validate the topology/grid/features, and restore adapters/topology in their
finally cleanup. Inpaint previously omitted adapter selection and could run with
LoRAs left by an earlier request. CPU offloading still executes if adapter restore
raises, and adapter restore precedes CPU offloading. The loaded checkpoint's own
rewired topology survives cleanup.

The actual spec/hash is recorded in generation params for PNG metadata. Native
pixel stride follows the selected VAE's downscale and the generation 2-1 patch
bank; incompatible grids are rejected. Style injection, active FBCache/NAG and
negative-weight NegPip prompts are refused before text encoding/model staging.
Dense generations retain their native path and omit residual metadata.

CPU checks cover real adapter installation/native forward, mixed declarations,
shape-refusal atomicity, metadata, feature/grid gates, base preservation, model
swaps and cleanup failures. All three real generation entry methods are exercised
up to a deliberate stop before GPU staging, then their finally restoration is
checked. This is not a full Z-Image generation/API/GPU acceptance result. All 48
aggregate CPU contracts and 17 existing Z-Image LoRA/DoRA quantization guard tests
pass; changed modules compile/import without CUDA initialization. Public Z-Image
configuration/defaults and model/editor discovery remain pending.


## Z-Image public topology configuration and inspection

Z-Image now joins the public residual runtime capability list after its native
CPU execution, artifact, training/resume and generation-adapter contracts. This
is a provisional integration under the owner's Qwen-only GPU acceptance scope.
The SSOT preset for the standard 30 main blocks has outer [4,26) and inner
[13,17): eight full-resolution blocks, eighteen half-axis blocks and four
quarter-axis blocks. Both refiner stacks retain native resolution. Main-stream
image-token-linear work is 42.5% of dense; this excludes refiners, captions,
padding and attention and is not measured speed or total FLOPs.

File/shard inspection reads metadata and keys only, accepts full Z-Image model
artifacts and rejects adapter-only keys. Files require declared native constructor
config; legacy files without it retain existing dense loading but cannot yet seed
the graphical editor. Native config directories are inspectable, but SushiUI's
existing Z-Image directory loader remains unsupported; full saves use files or
shard indices. Nonstandard block counts get an explicit editable initial region
and no guessed standard preset. Generation remains bound to the native 2-1 bank.

The capability response automatically enables the existing dynamic training
editor; no additional frontend defaults are hardcoded. Loaded model info supplies
the installed spec and the actual VAE-based pixel/token stride for the read-only
summary. CPU checks exercise the public schema/inspect routes, single/sharded/
sibling-index/directory inspection with tensor reads forbidden, LoRA-only refusal,
standard preset and public request/YAML configuration. The frontend is reviewed
statically; build/type checking and browser checks remain with the owner.

All 50 aggregate CPU contracts pass, including API 422 refusal for malformed
native array config. Changed files compile/import and OpenAPI parses with CUDA
untouched. No further GPU run was performed for this provisional Z-Image surface.


## Multiple-stream graphical editing

The graph renders each declared stream separately with its own block count,
resolution rows, layer path, skip arcs and drag coordinate scale. Streams without
regions display their native sequence. The editor's stream selector controls the
region list, independent/nested region placement and input boundary limits;
selecting a skip on another graph selects that stream too. Parent choices remain
within the selected stream and exclude the selected subtree. Resetting to a custom
layout creates one region in the selected stream while retaining the full declared
stream layout; the reset still replaces the topology as its tooltip states.

Grid summaries group by stream and both spatial dimensions, so equal token counts
in different aspect ratios or streams are not combined. The checkpoint summary
labels every region with its stream. The model's template supplies operators,
shrink factors and edge placement; no frontend default topology is introduced.

A CPU contract exercises a 7/11/5-block preview with independently rewired dual/
single streams and an untouched native stream, rectangular asymmetric shrink,
nesting, metadata round trip and actual preview API responses. Cross-stream
parent connections are refused with 422. All six topology contracts pass and the
changed test compiles/imports without CUDA initialization. Frontend changes are
statically reviewed against api.ts and the shared graph/summary/editor call sites;
no build/type check or browser rendering was run. The owner should include stream
selection, cross-graph selection, independent regions on a previously dense
stream, stream-specific endpoint dragging and parent choices in consolidated UI
verification once a multiple-stream native integration is available. This editor
work does not claim that a new architecture's native runtime is implemented.


## Flux2 native execution foundation

Flux2 now has an internal double/single native residual adapter. Each stream
restores its full image grid before handing text/image states to the next stream;
only the contiguous image tail is pooled, and evolving text states are retained.
Native timestep/guidance modulation, projections, normalization and output heads
are used. Per-batch timesteps/guidance are preserved rather than truncating to the
first item. The block-swap wrapper routes rewired models through this native path
and keeps the existing dense branch unchanged.

Image positions require one shared row-major 2D grid in four axes (T,H,W,L).
SushiUI's constant T/linear L layout and diffusers' constant T/L layout are
accepted. Coarse grids preserve T, rebuild integer H/W, and either rebuild linear
L or preserve its constant value. Text positions/rotary frequencies remain
unchanged. Reference KV caching, reference token grids, extra attention arguments,
NAG/NegPip/style processors, FBCache, token/block skipping and REPA are refused.
The native guidance vector itself remains supported.

Installing a topology replaces only the model instance's forward binding; no
installed diffusers file or parameter is changed. Removing it restores the exact
prior binding/descriptor. A deep-copied rewired model owns its own forward and can
be restored independently. The shared offloader uses double indices followed by
single indices, and native checkpointing receives the actual block and tensors.

CPU contracts use the installed real Flux2Transformer2DModel with three double
and seven single blocks, two heads/eight-dimensional heads, four-channel packed
inputs, five text tokens and a rectangular 8x12 image grid. They verify double
image lengths 96/24/96, single lengths 96/24/6/6/6/24/96, text-frequency identity,
all ten wait/submit pairs, unchanged parameter identities, wrapper/root output
identity, checkpointed/noncheckpointed finite gradient equivalence, independent
clone restoration, dense forward restoration and feature/grid refusals. All 54
aggregate CPU contracts and changed-file compile/real imports pass with CUDA
untouched.

Public configuration, presets and model/editor discovery are connected below. Legacy dense
wrapper paths are not accepted by this new-path CPU check. No Flux2 GPU workflow,
performance or quality evidence is claimed. Only Qwen-Image 2.1 receives actual
GPU acceptance under the owner's instruction.

## Flux2 checkpoint persistence

LoRA saves now carry the installed topology declaration. Full saves additionally
carry the canonical native transformer config, including both stream counts and
RoPE geometry. The writer returns the actual single-file or shard-index path.
The native loader validates topology/config headers before reading tensors or
resolving companions; missing config or an incompatible stream layout is refused.
Saved transformer config takes precedence over the companion repository config.
Declared VAE channel changes must still agree with the topology identity before
model construction. Rewired checkpoints require complete matching transformer
weights; the topology is restored after loading and dtype conversion.

CPU contracts drive the actual full writer and native Flux2 loader with tiny
three-double/seven-single models. Only external companion resolution/loading is
stubbed. Single files, shard indices and missing stems resolved to indices restore
every weight exactly and recover the topology/config, with output agreement at
`rtol=1e-5, atol=2e-7`. A deliberately different tiny companion config verifies
that the checkpoint config governs reconstruction. Missing/mismatched config
fails before tensor reading/downloads, and a legacy dense header remains accepted.
Four Flux2 contracts and 50 existing LoRA/VAE regression cases pass; changed files
compile and import with CUDA untouched. This is CPU artifact evidence only;
public configuration and inspection are connected below.

## Flux2 training inheritance

Both normal component loading and full-checkpoint-as-base loading now resolve
the installed topology after latent-space application and before checkpointing,
attention configuration or freezing. This catches incompatible latent edits and
training features before the model proceeds. The shared resolver supplies the
same inherit-versus-resume agreement rules as Qwen/Krea/Anima/Z-Image. Public
configuration is connected below, using the same resolver.

LoRA resume validates topology without installing it, then validates/loads branch
tensors before committing the topology and run config. CPU tests use the real
Flux2 adapter (50 branches over both streams) and LoRATrainer loader: incomplete
weights leave every branch, topology and config unchanged; valid weights restore
exactly and inherit the saved layout; a dense declaration cannot replace an
already rewired base. Native checkpointing plus the real `flux2_ops.train_step`
produces finite losses and finite nonzero LoRA gradients on CPU. Reference image
grids are rejected before training packing or sample setup. Both load entry
points are exercised up to a deliberate stop after topology resolution. These
are CPU contracts only, not real-model GPU training, optimizer resume or sample
quality acceptance.

All 57 aggregate rewiring contracts and 50 existing Flux2 LoRA/VAE regression
cases pass after this change. Changed backend files compile and import with CUDA
untouched; frontend checks remain consolidated owner work.

## Flux2 generation adapter scopes

Generation parses every selected adapter declaration and checks agreement with
the native model identity and the loaded base topology before installing any
branch. Mixed dense/rewired adapters or different residual layouts are refused.
The adapter session installs the complete branch plan atomically, then the
native forward binding restores the declared double/single topology. A weak
model reference and copied base spec govern reversal, including model swaps.
Every generation entry selects an adapter stack even when empty, so text-only
adapter scopes cannot leave a topology behind for a subsequent dense request.

Rewired adapter scopes unload both transformer and text encoder branches in
generation cleanup and restore the base topology. Dense adapter lifetime rules
remain as before. Cleanup still attempts CPU offload if adapter restoration fails;
an offload failure does not prevent the prior topology restore. Actual generation
metadata records the spec/hash and `kv_cache=false`; reference image grids,
style injection, FBCache, NAG and NegPip are refused before GPU staging, and native
packed-grid divisibility is checked. Img2img/inpaint preview uses the entry's
existing 16-pixel rounded image dimensions.

CPU contracts verify native forward with an applied rewired branch, invalid
stack/shape refusals, full-base topology preservation, model-swap restoration and
cleanup failure ordering. All three real generation entries are exercised up to
a deliberate stop before staging and checked for finally restoration. This
does not prove full GPU generation or performance/quality. All 60 aggregate
rewiring contracts and 50 existing Flux2 LoRA/VAE cases pass; changed backend
files compile and import with CUDA untouched. Public presets/model inspection
and model-info connection are described below.

## Flux2 public configuration and editor discovery

Flux2 joins runtime capabilities as a provisional CPU-verified integration.
The authoritative preset in `api/param_defaults.py` targets the pinned native
five-double/twenty-single layout. All five double blocks stay full resolution;
single blocks use outer `[4,16)` and inner `[8,12)` regions with 2x2 pooling each.
At 1024px this gives thirteen full-grid blocks, eight 1024-image-token blocks
and four 256-image-token blocks. The image-token-times-block sum is 61% of dense;
this is neither a FLOP nor a speed estimate, since streams have different work
and auxiliary text, attention and pooling remain. No real Flux2 quality or
calibration claim is made.

Other native block-count pairs have no guessed preset. They require an explicit
layout; inspection provides a one-region initial template and both stream
counts, and custom configuration is accepted after native identity validation.
The existing editor fetches runtime capabilities dynamically and already renders
one graph per stream, including the dense double stream. Its stream selection,
region bounds and parent choices use the selected stream's own count. No separate
frontend defaults are introduced. Owner build/type/browser checks remain pending.

File/index inspection reads config and safe tensor headers only, validates native
double/single block counts, and rejects adapter-only files. Native constructor
config metadata is required for Flux2 files; legacy files without that declaration
cannot seed the editor. Native-class directories are inspected through their
constructor config. Single files, shard indices and missing stems resolved to
indices expose identical topology/config hashes and both streams. The public
inspect and preview endpoints return these layouts without model allocation or
tensor loading; invalid declarations return 422. OpenAPI describes this surface.

Both Pipeline load branches report the saved topology and packed token stride.
The native-config fallback now reaches Flux2 instead of being swallowed by the
generic Z-Image transformer branch. CPU tests use a real tiny five-double/twenty-
single model for preset execution, and a three-double/seven-single model for
custom configuration, header/ASGI inspection and both Pipeline load branches.
Only model loading is stubbed in the latter Pipeline checks; all native artifact
round trips were verified separately above. GPU acceptance remains Qwen-only.

All 63 aggregate rewiring contracts and 50 existing Flux2 LoRA/VAE cases pass.
Changed backend files compile and import with CUDA untouched; the OpenAPI runtime
capability example matches the actual schema response. Frontend capability and
per-stream graph wiring were statically checked; owner build/type/browser review
is still required.

## Lens native execution foundation

Lens now has an internal native residual adapter for its evolving image/text
joint blocks. Only packed image tokens are pooled; text states pass through each
block and survive both nested-region exits. Native image/text normalization,
multilayer text feature projection, timestep embedding, output normalization and
projection remain in use. Single-feature text input is also supported. Native
block checkpointing owns recompute and CPU-offload behavior; the residual loop
does not add another checkpoint layer. Native offloader wait/submit calls retain
the original block index sequence, and no learned parameter is added.

Lens centers image RoPE coordinates and offsets text positions from the image
grid size. Coarse blocks use the native centered coarse image coordinates, while
text rotary frequencies stay at their original full-grid values. Joint masks
keep native image-prefix/text-tail ordering: image keys stay unmasked and the
text padding mask stays unchanged. The transport to the shared executor uses
text-prefix/image-tail ordering and converts back only at each native block.
One frame and one shared rectangular grid are required; inconsistent batches,
reference/video grids and out-of-table coordinates are refused. Style injection,
NAG/NegPip, FBCache, REPA, token/block skipping and extra attention arguments are
refused for rewired execution. Dense execution still takes the original branch.

Native config identity is derived from the constructor's own fields/defaults;
RoPE axes and selected text layers are canonical lists. Effective `inner_dim`
comes from native head count times head dimension, matching the constructor.
Removing the topology returns the model to the dense branch without rebinding
forward or changing modules/parameters.

CPU contracts use seven real Lens blocks with 24 hidden channels, two heads,
two twelve-dimensional text feature layers, five text tokens and an 8x12 image
grid. Image lengths are 96/24/6/6/6/24/96. Tests verify native image coordinates,
exact text-frequency/mask preservation, consecutive evolving text states, all
seven offloader wait/submit pairs, unchanged parameter identities, finite outputs
and dense restoration. No checkpointing, standard checkpointing, CPU-offload and
async CPU-offload modes agree in output and finite gradients on CPU; this does
not establish their GPU offload equivalence. All 66 aggregate rewiring contracts
and 69 existing Lens LoRA/REPA/text-mask cases pass, with changed-file compile and
real imports leaving CUDA untouched.

Lens public configuration and model/editor inspection are connected in the
public integration milestone below. No Lens GPU inference/training,
performance or quality evidence is claimed; GPU acceptance stays Qwen-only.


## Lens checkpoint artifact integration

Full-parameter saves now include the canonical native transformer config and
validated residual topology, and use the shared single-file/sharded writer. The
writer returns the actual file or shard-index path so REPA sidecars retain their
existing pairing. LoRA saves include the installed topology metadata; generation LoRA topology
agreement and installation are connected below.

Single-file and indexed-shard loads validate topology headers before reading
weights or resolving companion components. Rewired files require a native
transformer config whose stream identity matches the topology. The saved config
constructs the native DiT directly, preserving layer counts, RoPE configuration
and selected text feature layers even when the companion base has a different
DiT config. Rewired weights must be complete with no unexpected keys before
companion text encoder/VAE loading. Existing quantized-weight refusal and declared
latent/VAE restoration remain in the path. Legacy files without saved native
config retain the base-model construction route. Missing single-file stems
resolve to their sibling shard index.

CPU contracts exercise the actual full writer, ModelLoader dispatch, native DiT
construction and forward for single-file, shard-index and missing-stem paths.
External text encoder/tokenizer/VAE/scheduler loaders are stubbed; a base DiT
from_pretrained call is forbidden. Every restored weight and native output agrees
with the saved model, and text layer selection comes from saved config. Missing
config, incompatible config and incomplete weights are refused before companion
loading. No Lens GPU performance, quality or training acceptance is claimed.

All 67 aggregate rewiring contracts plus 104 existing Lens LoRA/REPA/text-mask
and VAE regression cases pass (171 total). Changed backend files compile and
import with CUDA untouched. GPU validation remains limited to Qwen 2.1.


## Lens training topology inheritance and resume

Lens training now resolves inherited residual topology after latent-space setup
and before native checkpointing, freezing or base quantization. The full-parameter
resume branch uses this same loader. Base/resume/explicit inherited declarations
must agree, and residual runs refuse REPA, TREAD, block skipping and partition
training/global adapters through the shared training gates. New public configure
requests and the default topology are connected in the public integration
milestone below.

LoRA resume validates checkpoint topology before reading or changing adapter
weights, and installs the resolved topology/config only after successful weight
restoration. Incomplete branch tensors leave the dense model, config and all
existing branch weights untouched; a later dense checkpoint cannot erase an
installed residual topology. CPU tests use real Lens LoRA branches and the
production flow-matching train_step with native checkpointing, confirming finite
loss and finite, nonzero adapter gradients. Both normal and full-resume loaders
reach topology resolution before checkpointing; the normal loader also completes
checkpoint setup, freezing and CPU placement.

All 69 aggregate residual contracts plus 104 existing Lens/VAE cases pass (173
total). The final loader check also passes separately. Changed files compile and
import with CUDA untouched. This is CPU compatibility evidence only; GPU training,
quality, performance and calibration acceptance remain Qwen 2.1 only.


## Lens generation adapter scopes and preflight

Lens adapter loading now checks every selected LoRA's topology against the native
model config and the base checkpoint. Mixed dense/residual declarations or
incompatible residual graphs are refused before installing any branch. The shared
AdapterSession still validates every target and installs the request atomically;
only a successful load installs the topology. Unload restores the base topology
through a weak model reference, so a model replacement cannot apply the previous
model's graph to the new model. An empty adapter request also clears a previous
LoRA-derived graph while retaining any graph carried by the base checkpoint.

All three generation entries preflight the selected adapters and aligned grid
before text encoding or component staging. Parsed files are reused for the later
post-staging adapter install. Rewired runs refuse reference grids, style injection,
FBCache, NAG and NegPip, and record the resolved spec/hash with KV caching disabled
in dit_rewiring_actual. Dense requests remove stale residual result metadata.
Native denoise loops consume the restored topology through the Lens forward path.

CPU tests verify real trained-LoRA/native forward agreement, mixed-topology and
zero-target atomic refusals, dense and residual base restoration, empty requests,
model replacement, and pre-staging refusal/metadata behavior for txt2img,
img2img and inpaint. These are CPU compatibility checks; no Lens GPU generation,
image-quality or throughput evidence is claimed. Public presets/inspection and
graphical configuration are connected below.

All 71 aggregate rewiring contracts plus 104 existing Lens/VAE cases pass (175
total). Changed backend files and the composed Pipeline compile/import with CUDA
untouched. GPU acceptance remains Qwen 2.1 only.


## Lens local directory artifact restoration

Local diffusers directories now read and reconcile topology headers from every
transformer safetensors file/index before native weight loading or construction.
A declared topology must match the canonical native transformer/config.json.
Rewired directories require diffusion_pytorch_model.safetensors or its native
index and reconstruct the native DiT directly, refusing quantized codes and
missing/unexpected weights before loading text encoder, tokenizer, VAE or
scheduler. The graph is restored before companion loading. Dense local artifacts
and Hub identifiers retain the existing from_pretrained path; the new header
contract covers local checkpoint directories. A validated transformer override
from a single-file save bypasses the companion base's directory topology.

CPU tests drive the actual ModelLoader and native construction/loading for both
single and indexed directory weights. All weights, native output and selected
text layers agree with the saved model. Config/layer disagreement and shard
metadata disagreement refuse before tensor loading/model construction; incomplete
weights refuse before companion loading. Dense headers preserve legacy dispatch.
External companion loaders are stubbed throughout; no Lens GPU is used.

All 72 aggregate rewiring contracts and 104 existing Lens/VAE regression cases
pass (176 total); changed-file compile and real imports leave CUDA untouched.
GPU acceptance remains Qwen 2.1 only.


## Lens public configuration and model inspection

Lens joins runtime_architectures after native execution, artifact persistence,
training inheritance/resume and generation adapter scopes are connected. Its
48-block default is centralized in backend/api/param_defaults.py: outer region
[4,44), deepest region [22,26), both downsampling image axes by two. At 1024px
with native stride 16, eight blocks keep 4096 image tokens, 36 use 1024 and four
use 256. The image-token linear estimate is 35.9375% of dense; this is neither a
FLOP estimate nor a speed/quality measurement, and excludes evolving text tokens.
Other layer counts require an explicit graph instead of guessing a preset.

Header-only inspection accepts Lens single-file, shard-index/missing-stem and
local diffusers artifacts, canonicalizes native config and verifies contiguous
native block counts against headers. Adapter-only files and config/header
count disagreement are refused. Native packed 2x2 patch geometry is required by
the editor. Inspection returns config/layout hashes, restored topology, default
or initial graph, stride and alignment without reading tensors or allocating a
model. The existing generic editor obtains Lens support through the schema and
renders region depth/downsampling/skip edges using this result.

Generation model info includes the restored topology and dit_rewiring_token_stride
for the existing read-only common graph. CPU tests exercise actual configure
request/train-section resolution, a real tiny 48-block preset forward, explicit
custom topology and resume-edit refusal, actual ASGI schema/inspection/preview
responses, and the composed Pipeline Lens loading branch. Inspection tests forbid
tensor loading and model allocation. Frontend API types, capability gating,
training editor and generation summary wiring were statically checked; owner
build/type/browser validation remains pending as requested. Lens GPU behavior,
performance and quality remain unverified; GPU acceptance is Qwen 2.1 only.

All 75 aggregate rewiring contracts and 104 existing Lens/VAE regression cases
pass (179 total). Changed files compile/import with CUDA untouched, and the
OpenAPI runtime capability example matches the actual schema response.


## MiniT2I native execution foundation

MiniT2I now has an internal residual adapter at the MMJiT net forward entry,
which is also used by the wrapper's pred_velocity. The wrapper/net share the
validated topology; clearing it restores the original dense branch without
rebinding forward or changing any learned module/parameter. Native config is
canonicalized from MMJiTConfig fields, including pixel/latent patch geometry,
text preamble depth, CFG interval and denoiser/sampling declarations. Native
absolute-position width and rotary head width must be divisible by four; image
and text hidden widths must agree because the native final layer concatenates
them. These are native execution constraints, not resizing rules.

The native learned mask token replaces masked text features before both text
projection and pooled conditioning. Text preamble blocks run at original text
length, with native attention backend/mode propagation and optional non-reentrant
checkpointing. Full-grid patch embedding and absolute sin/cos positions are added
once; their hidden representation is pooled with the image states. Joint blocks
use native text RoPE and native coarse integer image RoPE for each region grid.
Evolving text states survive region exits, while image states restore through the
shared FP32 identity-delta path. Native final projection and unpatchify run after
the full image grid has been restored, retaining x0 output and the wrapper's
existing velocity conversion. Block-swap wait/submit callbacks keep native block
indices; the text preamble is outside the swapped/resized stream.

REPA, FBCache, TREAD, block skipping, active style and NAG/NegPip hooks are refused
for rewired execution. Inputs require complete native patches, valid batch/text
mask/timestep geometry and divisible region grids. Pixel-space and latent-space
models use the same residual transport over their actual patch grid.

CPU contracts use seven real native double blocks, two text preamble blocks,
16 hidden channels, two twelve-dimensional heads, five text tokens and an 8x12
image patch grid. Image lengths are 96/24/6/6/6/24/96. They verify the first
block's full-grid absolute embedding, native per-block grid arguments, consecutive
evolving text states, all seven offloader wait/submit pairs, unchanged parameter
identities, x0/velocity and BCHW/BHWC behavior, and dense restoration. Native
checkpointed and direct outputs/finite gradients agree. Both pixel (patch16 RGB)
and latent (patch2 four-channel) models execute, and the published l16 head width
52 follows the existing native attention padding route. This does not establish
GPU offload, throughput or quality.

MiniT2I remains outside public runtime capabilities. Public training
configuration, generation adapter scopes, preset/model
inspection and editor exposure remain to be connected. GPU validation remains
Qwen 2.1 only.

All 78 aggregate residual contracts and 39 existing MiniT2I LoRA/CFG cases pass
(117 total). Changed files compile and import with CUDA untouched.


## MiniT2I checkpoint artifact integration

MiniT2I full saves validate the installed topology before copying tensors, then
persist its metadata and complete canonical MMJiTConfig. Dense saves retain the
legacy variant/I/O config format. The shared writer returns the actual single-file
or shard-index path; the full training adapter returns that path for correct REPA
sidecar pairing. LoRA metadata also carries the installed native topology.

Single-file loads preflight topology declarations and their native config before
weight reads/construction. A rewired file requires valid mmjit_config JSON whose
canonical identity matches the graph; it cannot silently fall back to variant
shape inference after a malformed/missing declaration. Native quantized-code
refusal and complete-key checks remain in place. The wrapper/net topology is
restored only after a successful native state load. Missing single-file stems
resolve to their sibling index at the component-loader entry.

Resolved local diffusers transformer directories reconcile every file/index
header declaration, validate transformer/config.json against the topology, and
load complete native single/indexed weights before resolving FLAN-T5, scheduler
or VAE. Dense directories retain native from_pretrained. Rewired directory weights
with missing/unexpected keys are refused instead of accepting a partially
initialized native model. Native VAE declaration/bundling behavior is retained.

CPU contracts exercise the actual full training writer, low-threshold shared
sharding, ModelLoader component dispatch, native model construction, weight load
and forward for single-file, index, missing-stem, directory-single and
directory-sharded paths. External FLAN-T5 and VAE loading is stubbed. All restored
weights/config/topology/output agree with the saved model. Missing/incompatible
file config is refused before tensors/allocation; incompatible directory config
is refused before tensors/allocation; incomplete file/directory weights are
refused before companion loading. These checks do not claim MiniT2I GPU inference,
quality, throughput or training acceptance. GPU validation remains Qwen 2.1 only.

All 79 aggregate residual contracts and 74 existing MiniT2I LoRA/CFG/VAE cases
pass (153 total), with changed-file compile/import leaving CUDA untouched.


## MiniT2I training topology inheritance and resume

MiniT2I training now resolves checkpoint topology after native latent-space setup
and before gradient checkpointing, transformer freeze/train mode or device
placement. The full checkpoint resume branch reuses this same loader. Inherited
explicit/base/resume declarations must agree, and residual runs refuse REPA,
TREAD, block skipping and partition training/global adapters through the common
gates. Public configure requests/presets are still pending generation/editor
integration; this milestone connects inherited saved graphs and LoRA resume.

LoRA resume validates topology before reading or mutating branch tensors. Only a
successful restore installs the resolved config and graph on both wrapper and
MMJiT net. Missing branch tensors leave the original graph/config/weights intact;
a dense resume file cannot erase the residual graph on a loaded checkpoint.

CPU tests use real MiniT2I attention/MLP LoRA branches and the production
flow-matching x0-to-velocity train_step with native gradient checkpointing. Loss,
prediction/reconstruction monitoring and adapter gradients are finite, with
nonzero gradients reaching trainable branches. Both normal and full-resume
loaders reach topology resolution before checkpointing. The normal loader also
completes native checkpoint setup, freezing, train-mode restoration and CPU
placement. External component loading/latent swap are stubbed only for loader
ordering checks; native execution is exercised independently. No MiniT2I GPU
training, quality or speed acceptance is claimed; GPU validation is Qwen 2.1 only.

All 81 aggregate residual contracts and 74 existing MiniT2I LoRA/CFG/VAE cases
pass (155 total), with changed-file compile and real imports leaving CUDA
untouched.

## MiniT2I generation topology scopes

Generation checks base/LoRA topology agreement before residency setup or either
adapter pass in txt2img, img2img and inpaint. Metadata-free dense adapters cannot
silently run against a residual base. Text-encoder branches are still applied
before encoding, and transformer branches install topology only after their
atomic weight restore succeeds. Cleanup restores both wrapper and MMJiT net;
a weak reference keeps model replacement from changing the new model graph.

Preflight derives the image token stride from native patch size and the actual
VAE scale (or one for pixel models), checks complete/divisible nested grids, and
records the resolved graph/hash with KV cache disabled. Residual runs refuse
reference grids, style injection, FBCache, NAG and NegPip before encoding.
CPU tests cover real adapter output, conflicting declarations, graph cleanup,
model replacement, pixel/latent grids and all three generation entry points.
GPU validation remains **Qwen-Image 2.1 only**; other architectures have CPU
implementation checks and no quality, speed or GPU training acceptance.
MiniT2I public presets, inspection and editor exposure remain pending.

The MiniT2I residual and existing adapter round-trip suites pass all 20 CPU
cases; changed-file compile and real imports leave CUDA uninitialized.

## MiniT2I public topology configuration and inspection

MiniT2I is now included in `/schema/dit-rewiring` runtime capabilities. Existing
frontend capability gating exposes its training graph editor; loaded model info
supplies checkpoint topology and native patch/VAE token stride to the generation
summary. Frontend verification is static only; owner build/type/browser checks
remain deferred to the consolidated manual pass.

`param_defaults.py` owns separate native-depth presets:

| Layout | Outer interval | Deep interval | Full / half / quarter-grid blocks |
| --- | --- | --- | --- |
| B16, 17 double blocks | [4,13) | [6,10) | 8 / 5 / 4 |
| L16, 23 double blocks | [4,19) | [9,13) | 8 / 11 / 4 |

Both retain four blocks at the deepest grid; text preamble blocks remain native
and outside the image transport stream. Other depths require an explicit graph.
CPU native forwards execute both presets and public training configuration
resolves the same graph as the editor inspection API.

Inspection supports native config directories and single/index/missing-stem
SushiUI files using headers only. It requires native image embedding keys and
contiguous actual double-block indices agreeing with saved config, rejects LoRA
files presented as base models, validates topology identity, and derives pixel
alignment/token stride from patch size and the existing native VAE registry.
Pixel, SDXL and Flux1 declarations are covered. Dense files without explicit
native config/block count are refused instead of guessing a variant.

CPU ASGI tests exercise schema, inspection and graphical-plan validation without
allocating models or reading tensors. Public MiniT2I support remains provisional:
no MiniT2I GPU throughput, image-quality or GPU training acceptance is claimed.
GPU acceptance remains restricted to Qwen-Image 2.1.

Validation: 123 aggregate residual/MiniT2I LoRA/CFG tests passed; the sole
remaining assertion expected the previous capability list. After updating that
API assertion, its targeted rerun passed (124 covered cases in total).
Changed-file compile, real imports and OpenAPI capability agreement pass, with
CUDA uninitialized throughout.

## Ideogram 4 native residual transport foundation

The vendored forward uses a residual branch only when topology is installed;
removing it restores the existing dense path. Native input/text/role projections,
per-sample timestep AdaLN and final packed projection stay unchanged. No weights
or parameters are added, copied or rebound. The graph applies to one backbone;
Ideogram 4's conditional and unconditional backbones use the same runner, with a
text/padding prefix or an image-only sequence respectively.

The runner validates a single complete row-major image tail with one image
segment per sample. It pools only the evolving image hidden states; text and
padding hidden states continue through all native blocks at their original
lengths. At each downsample boundary it pools native three-axis position ids to
cell centers, rebuilds MRoPE and the native segment equality mask, and preserves
prefix positions/segment ids. Nested exits restore the saved full-grid context
and add the upsampled image residual delta. Multi-image/irregular grids and
per-token timestep conditioning are refused because they require separate
transport/conditioning rules.

Each native block keeps its checkpoint function and block-swap wait/submit at
its original index. Residual execution refuses REPA taps, FBCache, active
style/NAG/NegPip/TREAD/block-skip hooks and non-native attention processors.
CPU native tests cover left-padded conditional batches, evolving text, exact
segment masks, coarse/deep RoPE centers, the unconditional image-only layout,
unchanged parameter identities, checkpoint output/finite gradients and bit-exact
dense restoration. Existing Ideogram 4 LoRA and REPA cases are included.

Checkpoint artifacts, inference adapter topology scopes, training resume and
public inspection/editor presets remain pending for Ideogram 4. It is not yet
advertised in public residual runtime capabilities. GPU validation remains
Qwen-Image 2.1 only; this foundation makes no Ideogram GPU performance or quality
claim.

The 3 new native residual tests and 42 existing Ideogram LoRA/REPA tests pass
(45 total). Compile and real imports pass with CUDA uninitialized.

## Ideogram 4 dual-backbone checkpoint contracts

The combined inference export helper writes both transformer states with canonical
native configs and the resolved topology. It validates the pair before copying
weights, restores shared-storage tensor keys rather than dropping a backbone,
and returns the actual file or shard-index path. Every shard and its index carries
the same graph. LoRA saves add the conditional graph and, when an unconditional
model is loaded, require that model's graph to agree.

Combined-file loading checks all topology headers before tensor/base-directory
loading, requires native configs for each saved branch, binds the graph to both,
and restores it only after native weights load successfully. Generation requires
both branches; conditional-only training loading may skip unconditional weights.
A split directory checks conditional/unconditional graph agreement and native
config identity before reading either state. Dense legacy loading remains on the
existing path; an absent dense config may still use companion config as before.

CPU tests use the actual native export and ModelLoader for single-file, shard
index, missing-stem index, split sharded directory and shared-storage export.
Both restored backbones reproduce every weight and the original packed native
output exactly. Invalid/missing graph configs are refused before tensor loading;
missing native tensors are refused before TE/VAE companions. Directory headers
with conflicting declarations are refused before either branch's weights load.
Only external TE/tokenizer/VAE/scheduler companions are stubbed.

The 4 Ideogram residual tests plus 42 existing Ideogram LoRA/REPA tests pass (46
total), with compile/real imports leaving CUDA uninitialized. This connects
artifact restoration and LoRA save metadata. Training graph setup/resume,
inference LoRA graph scopes and public presets/inspection/editor remain pending.
The existing Ideogram full-finetune refusal is retained; the combined export
helper does not claim to enable that baseline workflow. GPU validation remains
Qwen-Image 2.1 only.

## Ideogram 4 training inheritance and dual LoRA resume

The training component loader accepts existing directories and combined native
files/indexes (including a missing stem with its index). It resolves inherited
checkpoint topology on both loaded backbones before checkpointing, freezing or
device placement. Conditional-only training leaves the unconditional model
unloaded. When `ideogram4_train_uncond` is requested, a missing unconditional
model is refused instead of silently dropping that loss branch.

The shared training resolver validates the pair without changing either graph or
the config; installation occurs only after both native identities and inherited
base/explicit/resume declarations agree. The common residual feature gates apply
to inherited runs. Ideogram's public configure mode/presets remain pending.

LoRA resume checks topology and the saved `train_uncond` scope before reading
branch tensors. It uses the existing atomic branch restore and installs topology
on both models only after that restore succeeds. A missing branch tensor leaves
all weights, config and both graphs unchanged; a dense resume cannot erase an
installed residual graph. A mismatch in the second backbone's native config is
refused during dry validation.

CPU tests run real conditional/unconditional native LoRA branches through the
production velocity train_step with checkpointing and the reconstruction mix.
Loss/monitoring values and gradients are finite; nonzero adapter gradients reach
both backbones. Loader ordering is checked for directory and combined-file
entry points, with external component loading stubbed. No Ideogram GPU training,
quality or speed acceptance is claimed; GPU verification remains Qwen 2.1 only.
Inference LoRA graph scopes and public inspection/editor exposure remain pending.

All aggregate residual contracts and the existing Ideogram LoRA/REPA cases pass
(133 total). Changed-file compile and real imports pass with CUDA uninitialized.

## Ideogram 4 generation adapter topology scopes

All generation entry points parse requested LoRAs once and validate both loaded
backbones plus every adapter declaration before prompt cleaning, text encoding,
residency changes or transformer staging. They validate the normalized native
image grid (including the existing 256px minimum) against nested downsampling.
Residual requests reject reference grids, style injection, FBCache, NAG and
NegPip before those workflows can install incompatible processors. Resolved
spec/hash and disabled KV-cache status are recorded in generation metadata;
dense requests clear stale residual metadata.

The existing dual-component AdapterSession still owns atomic weight binding.
Only after it succeeds does the backend install the required graph on both
models. Cleanup restores both base graphs even if adapter unloading raises;
weak references restrict cleanup to the original models during a partial reload.
Metadata-free dense LoRAs cannot run against a residual checkpoint, nor can
selected LoRAs require different graphs. Residual inference requires both native
transformers, even for a LoRA trained on the conditional branch only.

CPU tests use real trained native branches, compare both restored outputs, refuse
mixed graph declarations and invalid unconditional targets atomically, verify
parse-once behavior, empty-request cleanup, saved-base retention and partial
model replacement. All three generation entries are checked before residency;
unsupported features and indivisible normalized grids refuse at that boundary.
Public presets, artifact inspection and editor exposure are still pending.
GPU verification remains restricted to Qwen-Image 2.1.

Validation covers 8 Ideogram residual contracts and 42 existing LoRA/REPA
cases (50 total): 49 passed initially; after correcting the minimum-resolution
assumption, the generation-entry contract passed its targeted rerun. The
dual-graph unload-failure/retry contract also passes. Changed-file compile and
real imports leave CUDA uninitialized.

## Ideogram 4 public topology configuration and inspection

Ideogram 4 is included in public residual runtime capabilities. The existing
frontend schema gate exposes its training graph editor, and loaded model info
provides the restored graph and 16px token stride to the generation summary.
The owner retains build/type/browser verification; frontend review here is static.

The native 34-block preset lives only in `param_defaults.py`: outer [4,30),
deepest [15,19), giving 8 full-grid, 22 half-grid and 4 quarter-grid blocks on
each backbone. Public configure resolves both models together; a custom block
count requires an explicit graph. CPU forwards execute the actual native preset
with evolving packed text and both nested transitions.

Header inspection supports combined single/index/missing-stem files and paired
native directories, including a transformer-subdirectory selection with its
unconditional sibling. It reads no tensors and allocates no model. Both branches
must have explicit native block counts, native embedding keys and contiguous
actual layer indices agreeing with config. Their canonical configs and installed
graphs must agree, since the editor expresses one graph shared by the two
backbones. LoRA-as-base files, mismatched counts/configs and conflicting graph
headers refuse with 422. Dense native declarations are inspected without adding
a graph; absent native configs are refused rather than guessed from a companion.

CPU ASGI tests exercise schema/inspection/plan endpoints and file/directory
round-trips under guards against model allocation and tensor reads. The pipeline
model-info path is also executed on CPU. Public support is provisional pending
owner UI validation; no Ideogram GPU speed, quality or training acceptance is
claimed. GPU verification remains Qwen-Image 2.1 only. The existing Ideogram
full-finetune refusal remains; LoRA and inherited checkpoint inference are the
connected workflows.

All shared residual contracts and existing Ideogram LoRA/REPA cases pass
(137 total). Changed-file compile, real imports and OpenAPI capability agreement
pass with CUDA uninitialized.
