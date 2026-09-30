# DiT residual rewiring contract

Status: accepted implementation contract. Canonical topology validation and
side-effect-free preview APIs are implemented. Qwen-Image 2.1 supports native
checkpointed residual execution, metadata-driven generation, LoRA/full training,
checkpoint restore and adapter-stack compatibility checks. Actual 1024px API
workflows verify generation, training samples, saved artifacts and resume, both
resident and with block swap. The graphical editor is connected and statically
reviewed; the owner will perform consolidated frontend checks. Krea 2 and Anima
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
