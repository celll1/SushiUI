# DiT residual rewiring contract

Status: accepted implementation contract. Canonical topology validation and
side-effect-free preview APIs are implemented. Qwen-Image 2.1 supports native
checkpointed residual execution, metadata-driven generation, LoRA/full training,
checkpoint restore and adapter-stack compatibility checks. Actual 1024px API
workflows verify generation, training samples, saved artifacts and resume, both
resident and with block swap. The graphical editor is connected and statically
reviewed; the owner will perform consolidated frontend checks. Krea 2, Anima,
Z-Image, Flux2, Lens, MiniT2I, Ideogram 4 and SenseNova have native execution,
artifact, supported training, LoRA generation and model/editor inspection
integration with CPU contract checks. Ideogram 4 retains its LoRA-only training
boundary; SenseNova residual training requires batch_size 1.
ACE-Step has temporal native execution, checkpoint persistence, inherited
training/LoRA resume and generation adapter scopes with CPU contracts. Standalone
component references and full-resume dispatch are connected. Public temporal
configuration, presets and editor/model discovery are connected with CPU checks;
complete audio workflow acceptance and owner frontend checks remain pending.
LTX-2.3 has a frame-local spatial residual block path, standalone transformer
artifact contracts and directory-loader graph restoration with CPU checks. Zero
block swap retains residual execution and its offload ordering. Training retains
the inner transformer reference. Inherited training and LoRA resume restore the
graph with CPU native-training checks. Generation LoRA scopes validate metadata
before staging and restore the base graph on exit. Public configure, presets,
header inspection and graphical editor/model discovery are connected; frontend
validation remains with the owner. Generated video sidecars retain the actual
residual graph and observed native token grids. Full single/shard artifacts carry
validated frozen companion root references and reach normal loading/full-resume
dispatch with CPU checks; complete video workflow acceptance remains pending.
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


## SenseNova U1.5 native generation transport (CPU milestone)

GPU verification remains restricted to Qwen-Image 2.1; every SenseNova check in
this milestone uses reduced native decoder layers on CPU. The published MoT
understanding branch remains dense and produces its original per-layer prefix
KV. Only the generation image-token stream enters the shared nested residual
transport. Native generation attention reads the same prefix at every resolution
with `update_cache=False`; it creates image K/V at the current coarse grid.
Spatial rotary positions follow average-pooled physical cell centers, while the
text-time coordinate stays constant. The final full-grid restoration precedes
`norm_mot_gen` and the existing generation head. No parameters are added.

A native NEOChat config fingerprint binds dense LLM geometry, rotary/norm
configuration and generation patch/channel geometry. Owner and decoder graphs
are installed together after identity validation. Dense understanding forwards
and their prefix caches bypass the residual runner. Both the inference decoder
forward and the differentiable training loop use the same runner, retaining
Module hooks and explicit non-reentrant checkpointing. The training loop keeps
its existing detached/differentiable/four-phase boundary-leaf prefix checks.

This milestone refuses packed multi-image plans, explicit generation masks,
REPA taps, reference-style attention, mixed understanding/generation tokens,
KV-appending calls and irregular spatial indexes before any generation block.
It does not alter the existing dense packed-batch workflow. MoE and sliding
attention are outside this provisional native implementation. Prepared inference
flash buffers retain their prefix and use only the active current-image tail;
immutable training prefixes can carry gradients into the understanding branch.

CPU contracts check 96 -> 24 -> 6 -> 24 -> 96 token transitions, spatial centers,
module hooks and parameter identity; native and explicit-training checkpoint
output/gradient parity; nonzero generation and understanding gradients; exact
zero-delta restoration; unchanged dense understanding outputs/KV; prepared
flash-prefix retention; and atomic identity/layout refusals. Artifact loading,
checkpoint save/resume, generation adapter lifetime, presets, inspection and
public capability exposure remain pending for SenseNova. This native milestone
is not a public acceptance or a SenseNova quality/performance claim.

Validation: the shared residual contract files plus existing SenseNova training
core and packed-batch contracts pass (134 total). The final stricter native
layer-count/KV-shape checks pass a targeted 39-case rerun. Changed backend files
compile and import with CUDA uninitialized. SenseNova remains outside public
runtime capabilities until its remaining persistence and request paths connect.


## SenseNova residual checkpoint persistence (CPU milestone)

The native full-finetune writer and LoRA metadata now persist the installed
owner/decoder graph. Full exports put the same spec/hash on every shard and the
index. Caller-supplied residual metadata cannot override the live graph. Residual
exports validate both the selected native config and the final embedded config,
so an extra metadata override cannot save a graph against different geometry.
Dense legacy exports keep their existing config fallback behavior.

The production loader resolves a missing single-file stem to its sibling index
and checks topology headers before tensor reads, model construction or companion
loads. Index/shard declarations, including absent declarations, must agree.
Residual full artifacts require embedded native config, contiguous native block
indices, generation projection keys and generation-head keys. After native
quantized installation, missing or unexpected model weights are fatal for a
residual artifact; the previous dense warning behavior is unchanged. The graph
is restored on both NEOChat owner and generation decoder before tokenizer/VAE
modules are loaded.

Refiner deltas carry the same graph as their complete base. Header preflight
follows relative/absolute base paths and validates every dependency, refusing
conflicting declarations, cycles and chains over 64 files before tensor reads.
A refiner export validates the referenced base headers/config and refuses a
different graph before publishing a file. These checks concern topology and
serialization; they do not expand the existing refiner training contract.

CPU tests use actual 42-layer NEOChat/Qwen3 modules with reduced hidden widths.
Single and sharded BF16 artifacts reload through ModelLoader with exact saved
weights and exact native decoder/head output. The same production INT8 writer
and loader restore 588 quantized Linears, retain the architecture's W8A8 opt-out
and produce finite residual predictions; no quality-equivalence claim is made
for requantization. Header-only tests refuse malformed hashes, missing/wrong
embedded configs, adapter-as-base payloads, conflicting shard declarations and
refiner dependencies before model/tensor access. Missing native weights refuse
before tokenizer loading. LoRA and refiner exports retain the declared graph;
reserved metadata spoofing and graph/config mismatches cannot publish an
incorrect residual file. Existing full-save, LoRA and refiner contracts pass.

Training configuration/resume, generation adapter lifetime, model-info metadata,
public presets/inspection and editor capability exposure remain pending for
SenseNova. GPU verification remains Qwen-Image 2.1 only. No frontend changes or
build/type checks occur in this milestone.

Validation: all shared residual contract files and existing SenseNova training
core, packed-batch, full-save, LoRA and refiner contracts pass (203 total).
Changed backend files compile and import; CUDA remains uninitialized.


## SenseNova inherited training topology and LoRA resume (CPU milestone)

SenseNova training resolves an installed checkpoint graph after VAE geometry
and refiner selection, before freezing/staging the transformer. The common
training resolver validates requested/inherited/resume agreement and existing
feature refusals. A config view serializes the native NEOChat config for the
common dictionary-based resolver; it does not replace the model config. Dense
legacy configurations remain dense. A geometry change that invalidates an
inherited graph must be corrected explicitly rather than silently reusing it.

LoRA resume validates topology metadata before loading adapter tensors and uses
the existing atomic adapter copy/rollback. Only after every adapter tensor loads
does it install the graph on both NEOChat owner and generation decoder and update
training config. A dense resume cannot erase an installed residual graph; wrong
native identity and explicit inherited mismatches refuse. A failed tensor
validation or partial copy leaves original graph, config and adapter weights.

CPU contracts exercise dry resolution, inherited feature/batch refusals, real
LoRA resume on both MoT branches, injected partial-copy rollback, graph/config
installation after success, and checkpointed nonzero understanding/generation
LoRA gradients. Understanding K/V projections are trainable in that probe to
satisfy the existing differentiable-prefix contract. A loader-order probe checks
geometry/refiner resolution before graph selection and freeze; its quantized
base acceptance is stubbed, so it is not a real-base training acceptance claim.
The native full-save/load and existing SenseNova training contracts remain the
separate evidence for their own paths.

Residual SenseNova training currently requires batch_size 1 because packed
multi-image transport is not implemented. Dense packed training remains
available. Public configure/presets, generation adapter lifetime and editor
capability exposure are still pending; this milestone connects inherited and
resume topology rather than advertising a new public architecture capability.
GPU checks remain restricted to Qwen-Image 2.1, with no SenseNova GPU use.

Validation: shared residual contracts plus existing SenseNova training core,
packed-batch, full-save, LoRA and refiner contracts pass (205 total). Changed
backend files compile and import with CUDA uninitialized.


## SenseNova generation topology lifetime (CPU milestone)

The generation mixin parses LoRA files once before attention/device staging,
validates all adapter/base topology declarations and the actual owner/decoder
geometry, then applies the graph only after the adapter session has installed
its weights. Request cleanup restores the previous graph with a weak reference
to the original model, including when adapter unloading raises. Unexpected
graph-install failure also unloads the newly installed adapters. Model replacement
cannot apply the old graph to the new transformer. Understanding-only adapter
loads share metadata validation while the understanding forward remains dense.
Training dry resolution now also validates the actual decoder before adapter
resume can install weights.

All three image entry points validate the normalized native token grid before
residency. Residual requests refuse style injection, FBCache, NAG and NegPip.
Reference images remain in the immutable understanding-prefix KV path; they do
not join the generation image grid. Request metadata records the actual spec
and hash. Its generic `kv_cache: false` field disables the image-generation
cache option, not SenseNova's immutable prefix-KV reuse.

The native residual transport also accepts the existing inference KV streamer's
host masters in BSHD layout after ordinary cache keys/values have been released.
It validates the active branch, every layer and native batch/head dimensions
before executing a block. The upstream streamer still owns staging and transfer.
A CPU fake of its acquire interface produces exactly the ordinary-cache output
at every residual resolution without modifying host prefixes; this is not a GPU
streaming-performance or real CUDA-ring acceptance claim.

CPU contracts cover both MoT adapter branches, parse-once behavior, arithmetic,
request graph installation/cleanup, inherited graph retention, model replacement,
unload/install exceptions, mixed dense/residual refusal, actual-decoder mismatch,
early grid/feature refusal in all image entry points, and streamed-prefix parity.
Shared residual and related SenseNova regressions pass (213 tests, one existing
img2txt capability expectation deselected). That expectation and seven adapter
key-normalization fixture failures were separately reproduced with the unchanged
pre-milestone mixin; no unrelated fixture fixes are included. Final rollback
changes are additionally checked with the native residual and legacy LoRA suites.
Changed backend files compile and import with CUDA uninitialized.

Public SenseNova configure/defaults, inspection/model-info and editor capability
exposure remain pending. GPU validation stays exclusively Qwen-Image 2.1;
other architectures receive provisional implementation and CPU checks. Frontend
build/type checks remain with the owner for consolidated manual validation.


## SenseNova configure, presets and editor discovery (CPU milestone)

SenseNova joins the public runtime capability list after native execution,
artifact persistence, inherited training/resume and generation adapter lifetime
have connected. Configure-with-null now resolves the backend-owned 42-block
preset: outer [4,38), inner [19,23), eight full-resolution blocks, thirty
half-resolution blocks and four quarter-resolution blocks. Other native layer
counts require an explicit topology; the editor returns a single-region starting
spec for those custom layouts. Canonical native config is idempotent for identity,
preset and preview consumers; artifact declarations still require raw native
NEOChat config so an identity summary cannot masquerade as a reconstructible
model config.

Read-only inspection validates actual block-index headers, generation-head and
projection markers against the embedded native layer count. It supports complete
single/sharded artifacts, missing stems resolved to sibling indices and refiner
deltas whose complete base supplies the geometry. This is a structural header
check, not a full tensor-shape or quantization acceptance check. Native pixel
tokens span 32px; latent artifacts require explicit matching VAE channels and
positive spatial compression. The editor uses generation-patch times VAE-scale
and refuses unknown geometry instead of inventing a preview scale.

Model info now supplies the restored graph and actual pixel stride for both
native and swapped generation grids. The existing frontend capability fetch
opens the shared graphical editor for SenseNova, with the same intervals, skip
connections and per-level token counts. The read-only summary clarifies that
immutable prefix KV remains available during residual execution. No frontend
build or type check is run; owner browser/build/type validation is outstanding.

CPU evidence includes actual 42-layer configuration/default selection and resume
identity checks, explicit seven-layer configuration/forward execution, header-only
single/shard/refiner inspection with tensor/model loading forbidden, real ASGI
schema/inspect/preview requests, malformed/missing artifact metadata refusals,
VAE scale/channel refusals and PipelineManager model-info propagation. Shared
residual and related SenseNova regressions pass (216 tests; the known baseline
img2txt capability expectation is deselected). OpenAPI schema examples match the
actual capability response; changed backend files compile and import without
initializing CUDA. The final raw-config artifact guard is checked again with the
native, artifact, inspection and schema contracts (29 passing tests).

This remains provisional CPU-verified SenseNova support. GPU performance and
quality verification are exclusively Qwen-Image 2.1. Further DiT architectures
and the final full-scope acceptance audit remain unfinished.


## One-axis transport for temporal DiT adapters (CPU milestone)

The common schema and editor now permit a per-axis factor of 1, retaining that
axis while shrinking the other by 2, 4 or 8. A [1,1] region is refused because
it performs no conversion. Existing two-axis presets and checkpoint hashes are
unchanged. OpenAPI records the same supported factors and no-op refusal. The
editor labels factor 1 as keeping the parent size; this does not expose a new
audio/video architecture before its native adapter is connected.

A temporal-only sequence can use the existing residual transport as a T-by-1
logical grid with factors [2,1], or a 1-by-T grid with [1,2]. Average pooling and
bilinear interpolation on that grid equal one-dimensional average pooling and
linear interpolation. Nested residual updates restore the original sequence
length, preserving the same block order, context restoration and parameter
identity; conditioning tokens can remain outside the pooled tail.

CPU tests compare values and every input/weight gradient against an independent
1D oracle over two nested regions, verify both orientations and context transitions,
and check exact zero-delta restoration in FP32, BF16 and FP16. Metadata round-trip
and actual ASGI preview tests exercise a singleton axis and reject [1,1], boolean,
zero and unsupported factors. All shared residual contract files pass (112 tests),
changed backend files compile and import, and CUDA remains uninitialized. The
frontend change is statically reviewed; owner build/type/browser checks remain
pending.

ACE-Step native inspection confirms a temporal Conv1d patch input, a decoder
with separate encoder cross-attention, alternating full/sliding self-attention,
RoPE, and ConvTranspose1d output with original-length cropping. Native integration
must operate after patch projection and restore before the output projection,
retain encoder conditioning, rebuild coarse RoPE and local-window connectivity,
preserve checkpoint/offload hooks, and handle cached cross-attention separately
from the resized self-attention sequence. Its source path currently discards
input masks before constructing bidirectional decoder masks; residual support
must not silently claim a stronger packed/padding contract than that native path.
Native ACE-Step execution, artifact persistence, training/resume, generation
lifetime and temporal editor units remain pending. No ACE-Step or other non-Qwen
GPU model has been loaded.


## ACE-Step native temporal decoder transport (CPU milestone)

A native adapter now validates decoder config, layer count and temporal-only
[factor,1] regions, installs/removes the graph on the wrapper/decoder, and can
produce canonical topology metadata. Execution enters after context/audio
concatenation and Conv1d patch projection and restores before adaptive output
normalization and ConvTranspose1d projection. Original native patch padding and
final-length cropping remain in place; the padded patch grid must still satisfy
all region divisibility constraints.

Coarse temporal RoPE uses physical patch-cell centers. Bidirectional sliding
attention retains its native temporal radius by comparing those coordinates;
full and encoder cross-attention remain unmasked for unpadded inputs. Per-level
RoPE/sliding masks are shared and restored with each region's context. Residual
execution bypasses native full-length mask construction and does not allocate
zero full/cross-attention matrices. Encoder conditioning remains the same
projected sequence at every resolution. Native layer module hooks are preserved;
explicit non-reentrant checkpointing bypasses the Transformers checkpoint wrapper
while retaining decoder checkpoint configuration and trainable conditioning.

Current execution supports eager and SDPA attention. It disables KV cache
creation, rejects supplied caches, attention extraction/early-exit probes and
padded masks before decoder blocks, and refuses irregular temporal positions.
Flash attention and cached encoder-KV reuse need separate native contracts;
they are not advertised by this milestone. Dense execution retains the original
native path when no residual graph is installed.

Reduced-width actual seven-layer ACE-Step decoder tests verify token counts
24/12/6/6/6/12/24, physical centers, sliding connectivity, unchanged conditioning,
module hooks, parameter identity, metadata and exact dense restoration after
graph removal. Direct/checkpointed outputs and every used gradient agree, with
nonzero audio/context/encoder/self/cross/projection gradients; SDPA agrees with
eager within FP32 tolerance. BF16 and FP16 native backward produce finite,
nonzero gradients with native timestep dtype. Cache creation is disabled in an
actual eval forward. Zero block delta restores the pre-head hidden exactly;
the head output can differ by small FP32 rounding because residual restoration
changes tensor strides. Native patch padding retains the exact requested output
length. Invalid feature/grid/axis requests refuse atomically.

The shared residual and legacy ACE-Step LoRA suites passed 125 CPU cases before
the final mask-allocation changes; the final native/legacy ACE-Step subset passes
14 cases. Changed files compile and import with CUDA uninitialized. Artifact
persistence/load, training configuration/resume, generation adapter lifetime,
presets and temporal editor discovery remain pending. ACE-Step is still excluded
from public runtime capabilities. GPU validation remains Qwen-Image 2.1 only.


## ACE-Step checkpoint topology persistence (CPU milestone)

The full-parameter writer embeds reconstructible native wrapper config alongside
its topology metadata; LoRA exports carry the required graph through the existing
adapter metadata hook. Export validates owner/decoder graph agreement, config
identity and actual depth before publishing. Dense legacy test doubles retain
empty residual metadata without requiring a native config.

The DiT loader performs topology/header preflight before model construction or
tensor reads. Residual artifacts require explicit native depth, contiguous decoder
block headers, native input/output projections and the wrapper null condition.
Missing/inconsistent config, bad hashes and adapter-as-model payloads refuse.
Mixed native/net-prefixed weights refuse in preflight and again against loaded
tensors. Full writer net-prefixed checkpoints now load by stripping that uniform
prefix; published unprefixed native weights retain their existing path. Embedded
config reconstructs custom native dimensions; config-less published files retain
the existing architecture template. Strict state loading remains in effect.
The graph is installed on wrapper and decoder only after all weights load.

CPU contracts use an actual reduced-width seven-layer conditional wrapper,
including condition encoder, FSQ tokenizer and detokenizer. Full writer to native
DiT loader preserves every saved tensor, decoder prediction and installed graph
exactly in FP32. An unprefixed artifact also restores exactly. LoRA metadata
retains the same graph. Header-only refusal probes forbid model allocation;
missing individual native weights fail strict loading. These tests do not claim
whole audio-pipeline or quantized-base acceptance.

The companion VAE/text-encoder tree remains the existing loader requirement;
standalone full-checkpoint companion resolution and full training resume still
need connecting. Sharding, generation LoRA lifetime, training configure/resume,
presets and temporal editor discovery also remain pending. ACE-Step stays outside
public runtime capabilities. All checks use CPU with CUDA uninitialized; GPU
verification remains exclusively Qwen-Image 2.1.

Final verification: changed backend files compile and import successfully. The
shared residual suites and legacy ACE-Step LoRA roundtrip suite pass 127 CPU
cases, with 20 subtests. CUDA remains uninitialized throughout verification.

## ACE-Step inherited training and LoRA resume (CPU milestone)

Training component loading now resolves inherited topology after native attention
selection and before freezing/precomputing conditioning. The shared training
resolver receives a canonical config view rather than treating a Transformers
config as an iterable mapping. Dense legacy loads remain valid. Residual loads
require matching owner/decoder geometry, actual depth and eager/SDPA attention;
shared incompatible training-feature checks still apply. Staged validation leaves
both run config and topology untouched on failure.

The real LoRA trainer validates checkpoint topology before reading/applying
adapter tensors, and installs the resolved graph after successful weight loading.
A saved native-decoder LoRA restores its tensors and owner/decoder graph from a
dense base, reproducing the pre-save FP32 decoder prediction exactly. A dense
adapter checkpoint against an already rewired base refuses before changing any
branch tensor. CPU tests also cover dry-run immutability, inherited graph edits,
unsupported attention/features and owner/decoder geometry disagreement.

The shared residual and legacy ACE-Step suites pass 129 CPU cases and 20 subtests.
The final geometry-check refinement passes all 17 native/legacy ACE-Step cases
and 8 subtests. Changed backend files compile and import with CUDA uninitialized.
This milestone does not verify a full audio training run, conditioner padding,
standalone companion resolution or full-parameter resume orchestration. Those
connections, generation adapter lifetime, public configure/presets and temporal
editor discovery remain pending; ACE-Step is still outside public runtime
capabilities. GPU validation remains Qwen-Image 2.1 only.

## ACE-Step padded conditioning and native training step (CPU milestone)

Residual decoder execution now retains the caller's native condition validity
mask. Audio remains an unpadded temporal grid; condition tokens remain at native
length. A shared B/1/1/L additive key mask broadcasts across all coarse/full query
resolutions without allocating a query-by-condition matrix at each region.
Binary B/L masks require one valid token per row and exact input dimensions;
invalid shapes, nonbinary values, NaNs and empty rows refuse before projections.
All-valid conditioning keeps the previous mask-free residual path. The dense
vendored path remains unchanged; its mask handling is not a padding oracle.

CPU contracts observe the same key-mask object at all seven native layers and
three query resolutions. Perturbing masked condition values leaves decoder
predictions exactly unchanged, and masked condition input gradients are zero.
Checkpointed predictions/parameter gradients match the ordinary residual path;
eager and SDPA agree within FP32 tolerance. BF16/FP16 masked backwards remain
finite. Native patch-tail padding still preserves requested audio length.

The real training ops now run against a reduced-width conditional wrapper with
native 64-channel audio, condition encoders, FSQ tokenizer/detokenizer, variable
text/lyric lengths and residual decoder checkpointing. Both full-decoder training
and real injected LoRA produce finite losses/gradients and AdamW parameter
updates; condition encoder gradients remain absent. This is a cached-latent CPU
training-step contract, not acceptance of VAE/audio encoding, a complete training
job, block-swap CUDA transfers or decoded sound quality.

Changed files compile and import. Shared residual and legacy ACE-Step suites pass
131 CPU cases and 27 subtests with CUDA uninitialized. Standalone companion
resolution/full resume orchestration, generation adapter lifetime, public
configure/presets and temporal editor discovery remain pending. ACE-Step remains
outside public runtime capabilities; GPU validation stays Qwen-Image 2.1 only.


## ACE-Step generation adapter scopes and sampler (CPU milestone)

Generation parses selected adapters before installation and resolves their
canonical topology against a native Transformers-config view. Every selected
LoRA must agree, including dense/absent declarations; an already rewired base
must match them. Owner/decoder geometry, actual depth and eager/SDPA support are
validated before adapter application. Inconsistencies report the shared
lora_rewiring_incompatible refusal code. Legacy dense fake models need no config.

A successful adapter load installs topology on wrapper and decoder and retains
a weak reference to the original owner plus its base graph. Unload restores both
weights and topology; the shared request decorator covers success/failure and
zero-target loads that have topology but no wrapped modules. Model replacement
restores the old owner's graph without clearing the new owner's graph or copying
old branches into it. Native trained-LoRA and generation-composite decoder
predictions agree within FP32 tolerance, and unloading recovers the exact dense
base prediction. Incompatible stacks refuse before any module is replaced.

The native generate_audio sampler previously allocated encoder/decoder KV cache
unconditionally, which conflicts with residual decoder preflight. For residual
execution only, initial cache, cover-transition cache and Heun corrector cache
are now absent, and main decoder calls request no cache. Dense generation keeps
its existing cache behavior. Real reduced-width conditional-wrapper CPU tests
run three-step Euler ODE, Heun ODE and Euler SDE with variable-length conditioning,
forbid cache construction throughout residual generation, and verify finite native
64-channel output latents. Dense Euler still passes cache on all three steps.
This does not claim decoded waveform quality or a complete REST audio workflow.

Changed backend files compile and import. Shared residual and legacy ACE-Step
suites pass 133 CPU cases and 30 subtests with CUDA uninitialized. Standalone
companion resolution/full resume orchestration, generation grid preflight and
metadata reporting, public configure/presets and temporal editor discovery remain
pending. ACE-Step remains outside public runtime capabilities; GPU validation
stays exclusively Qwen-Image 2.1.


## ACE-Step standalone full artifact sources and resume dispatch (CPU milestone)

Native component loading records the exact frozen VAE/text-encoder file paths,
original component root and tokenizer source on the DiT owner. Full checkpoint
exports retain these in versioned acestep_components metadata, alongside native
wrapper config and topology. Re-export after standalone loading preserves the
same references. Local paths are absolute; the existing tokenizer hub fallback
remains a source identifier. These are external references, not embedded frozen
component tensors, and their declared paths must remain available.

File and directory resolution prefer explicit sources over nearby flat-tree
heuristics. Invalid schema/missing fields, unavailable root/files, wrong path
kinds and missing absolute tokenizer directories refuse before DiT construction.
Legacy artifacts without declarations keep the existing ancestor-tree behavior;
headerless layout probes also retain that behavior, while actual tensor loading
still checks the safetensors header. No fallback substitutes another nearby VAE
or encoder when an explicit source is broken.

The generic model detector recognizes full ACE-Step metadata only with native
wrapper markers, including the trainer's net prefix. Full-parameter resume now
dispatches the selected artifact into the same native training component loader
instead of falling through the image-model branches. Existing topology inheritance
then applies to the reconstructed owner/decoder.

CPU tests use actual reduced native DiT weights/config/topology and mocked frozen
companion builders/tokenizer to verify exact declared paths, identical restored
DiT tensors and re-exported references. Separate contracts exercise the real
full-resume dispatcher, declared-source refusal before allocation, legacy flat
folders and explicit-reference priority for both file/directory selection. This
does not verify actual Oobleck/Qwen3/tokenizer weights or an entire resumed job.

Changed backend files compile and import. Shared residual, legacy ACE-Step LoRA
and existing quantized-export CPU suites pass 165 cases and 48 subtests with CUDA
uninitialized. Public configure/presets, temporal editor discovery, generation
grid preflight/metadata and complete CPU audio workflow seams remain pending.
ACE-Step is still outside public runtime capabilities; GPU verification remains
Qwen-Image 2.1 only.


## ACE-Step generation grid preflight and audio metadata (CPU milestone)

A shared native generation preflight validates owner/decoder topology and checks
the actual temporal token grid, ceil(input_frames / native patch_size) by one,
against every nested residual region. It produces JSON-safe runtime metadata
with canonical spec/hash, cache-disabled state, native input frame count and
actual token grid. Dense execution returns no residual metadata; the backend
clears any stale entry.

The native sampler checks its source length before preparing conditioning or
noise. Txt2audio retains the existing duration-to-frame rounding and checks
before silence-VAE/text-encoder staging. All three audio generation paths also
check actual source-latent length before DiT staging/block-offload installation.
Backend errors use the normal API ValidationError path. Native patch-tail padding
remains valid when the resulting patch grid divides correctly; arbitrary lengths
that do not divide the selected down factors still refuse. General residual tail
padding and its editor contract remain an explicit pending item before public
ACE-Step presets can make default-duration generation useful.

Audio sidecar JSON now retains dit_rewiring_actual; the existing gallery params
already carry the same mutable request dict. CPU contracts write a real short
FLAC/sidecar and read back exact topology/hash/frame metadata. Native sampler
invalid-grid probes forbid condition preparation, and a real txt2audio entry
probe forbids VAE/model staging and confirms the early API error. These are CPU
control-flow/media-persistence checks, not decoded model sound quality or a live
REST generation acceptance claim.

Changed files compile and import. Shared residual, legacy ACE-Step and existing
quantized-export suites pass 166 CPU cases and 48 subtests with CUDA uninitialized.
Public configure/presets, temporal editor/model discovery, arbitrary-length grid
handling and complete audio workflow/resume seams remain pending. ACE-Step stays
outside public runtime capabilities; GPU verification remains Qwen-Image 2.1 only.


## ACE-Step versioned temporal tail extension (CPU milestone)

New ACE-Step topology identities include sushi_residual_grid_policy with value
zero_token_tail_v1 in the canonical native config. The native decoder computes
the least common multiple of all cumulative region factors, appends zero hidden
tokens after native input patch projection to the smallest aligned grid, and
extends regular native RoPE coordinates over that tail. Residual transport still
requires an exactly divisible execution grid. Restoration removes the synthetic
tail before the native output norm/projection; native patch padding/cropping then
preserves the requested audio-frame length. No parameters are added.

These synthetic tokens participate in native attention/block execution after
zero initialization; this is a defined sequence extension, not an attention
padding mask. Partial boundary pooling and attention can therefore affect nearby
predictions. CPU correctness does not establish sound quality; non-Qwen models
remain provisional under the owner's GPU restriction.

Graph hashes identify the policy. Legacy ACE-Step base-config hashes without the
policy remain accepted and retain require_divisible execution rather than being
silently reinterpreted. Full exports explicitly serialize the resolved policy;
new/legacy native configs are canonical and idempotent. A checkpoint that declares
zero_token_tail_v1 but carries a legacy topology hash refuses before model
allocation. LoRA compatibility still compares the complete graph identity,
including policy differences.

Runtime audio metadata now includes native_token_grid, actual token_grid,
transport_padding_tokens and grid_policy. For the seven-layer CPU fixture with
patch size two and two factor-two regions, 250 audio frames yield 125 native tokens
and 128 execution tokens (three synthetic tokens), with 64/32 middle/deep tokens.
The native three-step generator returns exactly 250 frames without KV cache.

CPU contracts cover lengths 45, 250 and one, exact layer token counts, finite
nonzero audio gradients, checkpointed prediction/all-parameter-gradient agreement,
identity-block restoration, policy roundtrip through actual full writer/native
loader and legacy rejection of nondivisible grids. Shared residual, legacy
ACE-Step and existing quantized-export suites pass 168 cases and 50 subtests.
Changed backend files compile/import with CUDA uninitialized. Public presets,
temporal editor/model discovery and full audio workflow/resume acceptance remain
pending; GPU validation stays Qwen-Image 2.1 only.


## ACE-Step public temporal configuration (CPU milestone)

ACE-Step joins runtime_architectures for explicit configure mode. The preset in
param_defaults uses 24 decoder layers: outer [4,20), inner [10,14), each shrinking
time by two while retaining width one. Eight layers use the full temporal grid,
twelve the middle grid, and four the deepest grid. Other depths expose an editable
single-region initial_spec and require an explicit spec; no preset is guessed.
Ordinary checkpoints retain dense execution unless conversion is selected.

Inspection accepts native full files and the loader's selected flat-tree DiT.
It reads native config and tensor header shapes without loading tensors or
allocating a model. Published files without embedded config must match the known
native depth, input/output patch projections, null conditioning, and each block's
attention/MLP geometry; custom missing-config files and adapter-only headers
refuse. Runtime construction retains the loader's strict full weight checks.

OpenAPI and TypeScript add temporal_geometry to inspection and
dit_rewiring_temporal_geometry to loaded model information. These declare native
frame rate, patch size, default preview frames and grid policy. The editor uses a
clip-frame input and time shrink controls, shows native versus transport token
counts and synthetic tail count, and keeps the existing graphical region/skip
editing. The loaded-model summary previews temporal topology. Image editors keep
spatial inputs. Frontend validation remains static; the owner runs final build,
type and browser checks.

The 30-second API default produces 750 raw frames, 375 native patch tokens and a
376-token execution grid under zero_token_tail_v1: one synthetic tail token,
188 middle tokens and 94 deepest tokens. Legacy hashes keep require_divisible;
inspection exposes the native canonical hash separately from the restored legacy
hash, so previews do not silently pad an older topology. Final native output
cropping retains requested audio length. This arithmetic makes no sound-quality
or GPU-throughput claim. GPU acceptance remains Qwen-Image 2.1 only.

CPU contracts exercise actual FastAPI schema/inspect/preview, new-run preset and
custom configuration, changed-topology resume refusal, a 24-layer reduced-width
native decoder forward/backward, wrong-header/missing-config refusals, flat-tree
selection and legacy-grid inspection. Complete audio workflow/resume acceptance
remains pending; non-Qwen integration is provisional under the owner restriction.

Dense ACE-Step exports with frozen component references also retain their native
config without requiring a residual graph. A real full-writer/native-loader CPU
roundtrip verifies exact dense weights and absence of topology metadata.

Changed backend files compile and import with CUDA uninitialized. The shared
residual/legacy ACE-Step/quantized-export run passes 172 tests and 50 subtests;
after the dense-export correction, its native-source and quantized-export
subset passes 38 tests and 18 subtests. No non-Qwen GPU run or frontend build/type
check was performed.


## LTX-2.3 frame-local residual execution and transformer artifacts (CPU milestone)

LTX-2.3's native blocks jointly update video and audio, so its residual region
shrinks each video's spatial grid independently while retaining every native
frame and the complete audio token sequence. The canonical native config includes
sushi_residual_video_grid_policy=independent_spatial_frames_v1 and all constructor
geometry/modality settings. Topology uses a main stream over the existing shared
block sequence. A frame count is runtime geometry, not a new learned parameter.

Shared residual execution now accepts grid_groups. Pooling and bilinear residual
upsampling reshape each frame into a separate batch grid before transport.
Flattening all frames into one image height would allow interpolation across
frame boundaries even when pooling happened to align; the grouped path prevents
that error. Per-frame spatial divisibility is validated before projections.
Existing image/audio callers use one group. A separate per-frame oracle verifies
values and input/weight gradients for nested regions.

The existing Ltx2BlockLoopWrapper owns the residual loop. Native projections,
time and prompt conditioning, masked text attention, self/cross-modality attention,
modulation gates, output heads, hooks and gradient checkpointing remain in use.
Each coarse grid averages native coordinate boundary pairs and tokenwise video
modulation tensors, recomputes native video/cross RoPE, and preserves time and
full audio positions. Each block returns both updated modalities; video restores
its residual update at region exits while audio continues through every block.
Interleaved/split RoPE and asymmetric spatial factors have native CPU coverage.
Generation offloader wait/submit ordering is tested with a CPU observer; actual
CUDA transfer is unverified. This does not claim time isolation inside native
attention: only transport is frame-local.

Initial execution requires unit native video/audio patches, explicit positive
frame/height/width geometry and rectangular native boundary coordinates. It
refuses STG/perturbation masks, custom attention kwargs, FBCache/Spectrum,
TREAD/BlockSkip and style processors before projections. Dense wrappers continue
to delegate to the native transformer unchanged. Public model loading and
training configuration do not expose this internal foundation yet.

LoRA metadata records the complete graph identity. The real full writer also
records native transformer_config and uses the shared 10-GiB shard writer,
returning the actual file/index path. The internal standalone transformer loader
validates topology, explicit block count, AV projection/attention header shapes,
config agreement and shard ownership before loading tensors; it constructs the
native model on meta, preserves verified quantized linears and applies strict
state loading. It then restores the graph into the wrapper. Frozen companion
components and their source references are separate, pending integration work.

A reduced-width seven-block actual LTX-2.3 transformer verifies three-frame
288/72/18 video-token grids and five audio tokens throughout, nonzero finite
video/audio/text gradients, exact checkpointed predictions and all-parameter
gradients, dense delegation, identity-block output heads, per-token timesteps,
FP16/BF16 execution and graph/geometry refusal before projection. Real single-file
and forced-shard full saves restore exact weights and graph, with FP32 predictions
within rtol 2e-5 / atol 2e-6. A CPU gate Linear probe traced a 2.38e-7 intermediate
difference to weight alignment (original 64-byte aligned, Safetensors view offset
24): inputs/weights/bias were identical and cloning the weight removed the delta.
Avoid claiming bitwise CPU artifact predictions or copying all large weights to
force identical storage alignment.

GPU validation remains Qwen-Image 2.1 only. Video quality, speed, actual offload,
full pipeline/run/resume, public configure/presets and graphical model/editor
integration remain pending. Non-Qwen work stays provisional.

The standalone loader constructs from resolved native kwargs, so Diffusers'
internal _use_default_values bookkeeping cannot replace a saved explicit RoPE
mode with a library default. A split-mode checkpoint with deliberately stale
bookkeeping restores split RoPE and its original graph.

Changed modules compile/import with CUDA uninitialized. Shared residual, legacy
ACE-Step/LTX-2 and quantized-export suites pass 198 cases and 64 subtests; the
final LTX-2/runtime/legacy-LoRA subset passes 30 cases and 14 subtests. No GPU or
frontend build/type checks were run for this milestone.

### LTX-2 directory restoration and wrapper lifetime

The normal directory loader reads every transformer topology declaration before
loading weights or frozen companions. For a rewired component it validates
config.json against the topology, strictly restores the native AV transformer
and passes that native object to Diffusers pipeline assembly. The block-loop
wrapper is installed afterwards and shared by the component map and pipeline.
Conflicting directory geometry refuses before tensor and companion loading.

Zero block swap retains the residual wrapper, excludes the transformer from the
whole-module offload hook and preserves the existing offload-before-wrap order.
Training stores the native inner module in transformer_original so parameter
consumers do not acquire an additional wrapper prefix.

CPU tests verify directory restoration with real tiny native weights, early
geometry refusal and zero-swap offload ordering. Compilation and real imports
pass; the LTX-2/runtime/legacy-LoRA subset passes 32 cases and 14 subtests with
CUDA uninitialized. Companion assembly is mocked in this bounded test. Complete
video generation and training/resume acceptance remain pending. GPU validation
remains exclusively Qwen-Image 2.1; non-Qwen quality and speed are unmeasured.

### LTX-2 inherited training and LoRA resume

Training resolves topology through the shared inheritance/resume contract and
validates the native constructor geometry and actual block depth. Installing a
graph preserves the original inner parameter modules, installs exactly one block
loop and updates the training pipeline reference. A bare transformer without an
execution wrapper or trainer owner refuses graph installation. Incompatible
routing/block-skip features and edited or dense-mismatched resume graphs refuse
before adapter tensors are read.

LoRA load-time preflight and the actual resume loader share artifact selection
for latest, absolute paths and output-directory-relative names. Topology is read
from that selected header during component setup, then validated again before
adapter tensor loading. Successful LoRA restore reinstalls the graph and wrapper
without changing native parameter identities.

Actual reduced-width native train_step calls with mixed per-sample fps exercise
full and LoRA flow losses, reconstruction loss, gradient checkpointing, finite
nonzero gradients and AdamW parameter updates. Video grids follow 288/72/18/18/
18/72/288 tokens while native audio stays unscaled. A real LoRA save and resume
restores exact predictions; dense mismatch refuses before tensor loading and
preserves adapter tensors. Wrapper ownership, no-mutation dry validation and
latest/path selection also have CPU contracts.

Changed modules compile and import with CUDA uninitialized. LTX-2, shared
training, legacy LTX-2 LoRA and ACE-Step contracts pass 49 cases and 39 subtests.
This is CPU provisional coverage, not complete video training/resume workflow
acceptance. GPU validation stays Qwen-Image 2.1 only. LTX-2 public configure,
presets, editor/model discovery, generation LoRA lifetime and full-checkpoint
companion/resume dispatch remain pending.

### LTX-2 generation LoRA topology scopes

All three generation entries parse selected LoRAs and resolve their graph before
quantization, device staging or offload setup. Every selected adapter declaration
must agree, including dense declarations, and match an already-rewired base.
Native spatial VAE factors and every nested region must divide the per-frame
latent grid. Caches, STG and reference style transfer are refused for residual
generation before GPU staging. A graph supplied by an adapter forces the custom
block loop even at zero block swap.

Adapter installation preserves the native block objects and propagates the
wrapper to base, image-to-video and conditioning pipeline references. A weak
reference records the original inner graph. Existing generation finally clauses
remove adapters and restore that graph on success and failure. Model replacement
restores the old owner's graph without changing the replacement's declaration.
A zero-target installation still has a graph scope and is restored correctly.
Cache cleanup now summarizes only an actual FBCache instance, so a residual-only
wrapper does not dereference a missing cache at generation exit.

CPU tests call the actual text-to-video entry with a mock assembled pipeline
that executes the real tiny native AV transformer. Predictions match the saved
training adapter within FP32 tolerance; successful and failing calls both restore
original native Linears and the dense graph. Mixed dense/residual LoRAs, a dense
adapter over a rewired base, invalid grids and unsupported features refuse.
Declared base graphs persist after scope exit, and replacement/zero-target cases
have contracts. Shared training/runtime and LTX-2 legacy suites pass 42 cases and
23 subtests; the common adapter-key gate's LTX-2 subset passes 11 cases. Changed
modules compile/import with CUDA uninitialized. No non-Qwen GPU use occurred.

The complete common adapter-key gate still has seven failures, reproduced at
pre-change commit f4ff6b54 in an isolated temporary archive: five SenseNova
counter/format cases, the Qwen sd-scripts parser case, and the inventory assertion
missing YuE2. These are separate existing acceptance issues, not a green suite.
LTX-2 public configure/presets/editor/model discovery, generated artifact metadata
and full companion/resume dispatch remain pending. Video quality and speed remain
unmeasured; GPU acceptance is Qwen-Image 2.1 only.

### LTX-2 public configuration and graphical discovery

The runtime schema includes LTX-2. A new native 48-block configuration defaults
to outer [4,44) and inner [22,26), each pooling spatially by 2x2. This leaves eight
full-grid, thirty-six middle-grid and four deepest blocks. At a 1024px frame and
the native spatial VAE factor 32, those grids are 32x32, 16x16 and 8x8 tokens.
Other native depths require an explicit graph; presets never guess custom depth.
The canonical preset remains solely in api.param_defaults.

Model inspection recognizes native LTX-2 directories, full single files and
shards. Geometry and block/projection headers are checked without reading tensor
bytes. Ordinary Diffusers components can declare geometry in config.json when
headers have no topology/native-config metadata; rewired files still require
embedded native configuration. Conflicting geometry, incomplete AV headers and
adapter-only files refuse before model allocation.

OpenAPI, loaded-model information and the frontend share video_geometry with
native time, height and width scales and independent_spatial_frames_v1. Axis
scales can differ; the editor uses each native scale to validate pixel dimensions
and calculate per-frame token grids. The existing draggable layer boundaries,
spatial factors and skip arrows apply to these grids. Editor and restored-model
summary explicitly state that native frame counts and audio tokens remain
unscaled and that skips/pooling do not cross frame boundaries. Frontend review is
static; the owner performs consolidated build/type/render checks.

Public schema/inspect/validate ASGI checks and native header contracts verify the
48-block preset, explicit configure, dense config-only directories, single/shard
identity, axis-specific geometry and early invalid-depth refusal. Backend files
compile/import and OpenAPI parses. LTX-2, shared training, legacy LoRA and existing
inspection suites pass 44 cases and 23 subtests with CUDA uninitialized. GPU
validation remains Qwen-Image 2.1 only. Full artifact companions/resume dispatch,
generated metadata and complete video workflow acceptance remain pending.

### LTX-2 generated video metadata

Every residual native forward records its validated frame/height/width grid,
audio token count and graph digest. Generation clears this observation after
LoRA installation, then captures metadata only after the assembled pipeline
returns successfully and before adapter/topology cleanup. Missing or differently
hashed observations refuse capture rather than reusing an earlier forward.
The detached dit_rewiring_actual snapshot retains the graph/digest, native video
grid and total token count, native audio token count, axis-specific geometry,
independent_spatial_frames_v1 and native_unscaled audio policy. Dense generation
removes residual result metadata; preflight also removes stale caller fields.

Text-to-video, image-to-video and temporal outpaint capture through the same
helpers inside their existing try/finally scopes. LoRA teardown restores the base
model graph while leaving the generated snapshot intact. The real video writer
persists this field in the JSON sidecar alongside its actual output dimensions,
frame count, model/seed and timing provenance. No route parameter or default is
introduced; video metadata remains an output artifact.

A CPU test runs the actual text-to-video entry with a mock assembled pipeline
executing the real reduced-width AV transformer, then uses software FFmpeg and
the real save_video_with_metadata writer. JSON readback matches the complete
execution snapshot after the loaded model has reverted to its dense graph.
Observed native grids are three frames by 8x12 video tokens and five unscaled
audio tokens. Success/failure, stale-observation refusal, dense metadata removal
and snapshot detachment have contracts. Generated decoded pixels are mocked;
this validates execution/provenance and file saving, not video quality.

Changed modules compile and import. LTX-2/legacy-LoRA and lossless-video container
suites pass 40 cases and 23 subtests with CUDA uninitialized. No non-Qwen GPU was
used. Full-checkpoint companion references and resume dispatch, remaining model
integrations and consolidated owner frontend/workflow acceptance remain pending.

### LTX-2 standalone full artifacts and training restore

Full native transformer exports retain ltx2_components version 1 with the exact
absolute Diffusers companion root captured by normal loading. This root declares
LTX2Pipeline and supplies frozen text encoder/tokenizer/connectors, video/audio
VAEs, vocoder and scheduler. Their weights are external, not duplicated into the
DiT save; the root and declared component directories must remain available.
Every index/shard reference declaration must agree. No missing source is guessed.

Normal single-file and shard model detection dispatches to load_ltx2_from_path.
The loader validates references and saved/native-root geometry from headers
before reading checkpoint tensors or allocating frozen companions. Native root
geometry can come from config.json or the original transformer's config metadata
without reading its weight tensors. The saved DiT is restored strictly with the
existing quantized-state contract, then passed as a native component override to
Diffusers assembly. Original-root DiT weights are not loaded a second time.
Residual wrappers are installed after assembly; dense saves remain native even
when they carry companion references. BaseTrainer full-resume dispatch loads
through the same component path and shared inherited-topology contract.

Real tiny full saves, both single and forced shards, restore exact weights,
graph/companion references and FP32 predictions within the established storage
alignment tolerance. The actual BaseTrainer checkpoint-as-base dispatcher calls
the actual LTX-2 load_components, configures native attention/checkpointing on CPU
and retains one execution wrapper shared with the training pipeline. Public
safetensors dispatch and model detection have contracts. Root DiT bytes are
intentionally invalid in the geometry-file case, proving those weights are not
read; a header-only source-geometry fallback is also checked. Missing roots,
unsupported reference versions, missing audio VAE, geometry disagreement and
index/shard reference mismatch refuse before tensor/companion loading. Dense
exports do not acquire residual policy markers or a residual graph.

Changed modules compile/import and OpenAPI parses. LTX-2/legacy LoRA and quantized
export contracts pass 64 cases and 38 subtests with CUDA uninitialized. Frozen
companion assembly is mocked with CPU modules; native DiT restoration and training
load dispatch are real. This does not establish complete pretrained video
quality/performance or whole training/optimizer resume acceptance. GPU validation
remains Qwen-Image 2.1 only. Remaining architectures and consolidated owner
frontend/workflow acceptance still require completion.

### MiniMax-H3 packed spatial transport foundation

GPU acceptance remains restricted to Qwen-Image 2.1. MiniMax-H3 work uses CPU
fixtures only; no pretrained model quality or throughput claim is made.

The native transformer packs text, reference/conditioning video, audio and
output video into a single self-attention sequence. Modality tags cannot identify
output video on their own: reference images and vision tokens also have the video
tag, and temporal/spatial pins can permute video_indices. The residual caller
must therefore supply the complete target-video row indices in chronological
frame-major order, together with its explicit frame/height/width geometry. It
must not guess the target from a modality tag or the beginning of video_indices.

packed_residual_layout.py transports these explicit rows into the common runtime's
contiguous spatial tail and retains every other native row as a full-resolution
prefix. The inverse permutation restores the original packed layout before the
native modality heads select rows. Coordinates must form a chronological
rectangular lattice, and target rows must have the native video tag. Coarsening
averages only spatial RoPE coordinates, retains the exact native per-frame clock,
and preserves every prefix position/modality/timestep label. Audio and reference
rows still participate in each native attention block and can update normally.

A pooled cell must contain one AdaLN timestep label. Whole-frame pins survive
spatial pooling, while mixed pinned/free pixels in one cell refuse transport.
Silently averaging their discrete noise labels would change the conditioning
contract. A future public integration must preflight every nested level before
projection or block execution and provide an explicit policy for spatial pins.

Five CPU contracts and four subtests pass. The real reduced-width vendored
seven-block DiT executes a nested two-stage spatial stream with 288/72/18 target
rows and an unchanged 106-row text/reference/audio prefix, restores native row
order, and backpropagates finite gradients to the packed input and all seven
blocks. Native inference's in-place residual adds agree with the autograd path;
shared transport allocations protect the saved fine/coarse residual states.
Arbitrary packed-row permutations, both coarse levels' native clocks and prefix
labels, whole-frame versus spatial pins, and malformed grids are checked.
Changed modules compile and import with CUDA uninitialized.

This is an internal layout foundation, not an enabled MiniMax-H3 capability.
Native wrapper dispatch, generation/training topology installation, artifact
restore, sparse-attention planning and public editor/schema integration remain
pending. Other remaining architectures and consolidated owner acceptance remain
part of the overall task.

### MiniMax-H3 native wrapper execution

The existing MiniMaxH3BlockLoopWrapper now notices an installed inner residual
graph and accepts explicit residual_video_grid as an internal keyword. Graph
identity hashes the vendored constructor's native configuration, including
patch geometry, rotary axes and the pruned/full AdaLN variant, plus the packed
spatial policy. Restoration checks the actual block count before installing a
copy. Clearing the graph restores normal dense delegation.

Residual preflight validates the complete native modality-row partition, input
batch/row/channel shapes, timestep addressing and explicit rectangular target
lattice. Every nested spatial level is checked before any projection, including
mixed pinned/free AdaLN labels that only collide in the deeper level. Custom
attention kwargs, native sparse-attention plans, FBCache, debug skip and residual
probes refuse residual execution rather than consuming stale sequence geometry.
Supporting sparse plans needs an explicit per-level plan rebuild and remains
pending.

The wrapper keeps its native projections, token refiner, timestep embedding and
modality output heads. Its residual loop repacks only the intermediate hidden
state, recomputes native RoPE for each coarse context, and passes the native
block, modulation and positions explicitly through the native gradient
checkpoint function. Block-swap waits/submissions occur once per executed block.
After restoring the original target grid, inverse packing recovers the native
row order before output normalization and both heads.

The combined new layout/wrapper and existing wrapper/FBCache/skip/probe CPU
contracts pass 58 cases and 11 subtests. The actual full forward preserves video
and audio output shapes; gradient checkpointing reproduces exact predictions,
input gradients and parameter gradients. Both AdaLN-curve and unpruned timestep
MLP variants run, including a real backward into the timestep MLP. A pinned-frame
video permutation preserves exact native output ordering. Invalid topology,
attention/cache conflicts and deeper mixed pins refuse before projection.
Compile and real imports pass with CUDA uninitialized.

Generation/training callers do not yet supply the internal geometry, and the
public residual capability list does not yet include MiniMax-H3. Topology
installation through public loading/training, checkpoint persistence/restoration,
editor/schema integration and caller geometry plumbing remain pending. No
MiniMax-H3 GPU generation/training, quality or speed validation was performed.

### MiniMax-H3 native sampler and training caller geometry

Both native packed-layout builders now record target_video_grid with explicit
frame count, spatial token height/width and target start row. This records the
builder's target-video tail before any video-index permutation; it remains
correct when references have different sizes or pinned frames move to the
conditioning prefix. The grid is scalar geometry, not additional weight data.
Native index/tag/position tensors retain their previous contents.

The shared forward_kwargs helper adds residual_video_grid only when a residual
graph is installed. It checks that the native layout identifies exactly the
complete target tail and requires the execution wrapper. Dense callers retain
the original forward kwargs. The denoise loop passes this geometry on every
actual DiT evaluation; the training flow step uses the same builder/helper.
Residual generation refuses Spectrum/FBCache before mutating either scheduler,
so forecasting cannot bypass a graph's execution or reuse incompatible results.

The actual native CPU sampler runs three dual-scheduler evaluations with the
seven-block two-stage residual graph, updates video/audio rows and preserves
conditioning anchors exactly. The actual native train_step runs joint AV
velocity loss with padded-audio loss masking, native gradient checkpointing,
backward and an AdamW update that changes a deepest-block attention weight.
The full AV text/reference/audio prefix and target geometry are supplied by
production callers, rather than a mocked transformer invocation. Ref2va
variable reference banks and a pinned target frame retain correct target rows.

Layout/wrapper, training, native-layout conformance, reference/inpaint, temporal
pin, audio-pin and FBCache CPU suites pass 281 cases and 11 subtests. One existing
latent-regeneration test fixture lacked the current Qwen/SenseNova architecture
flags; the failing test and dispatch method were byte-identical to HEAD before
this change. The fixture now supplies those false flags, exercising the original
MiniMax-H3 recovery assertion again. Changed modules compile and import with
CUDA uninitialized. This does not validate pretrained AV quality or speed.

Public model/training configuration still needs graph installation and
checkpoint persistence/restoration; the public MiniMax-H3 capability and editor
remain pending. GPU acceptance remains exclusively Qwen-Image 2.1.

### MiniMax-H3 LoRA topology persistence and restoration

MiniMax-H3 retains its existing LoRA-only training boundary; full-parameter
training remains explicitly refused by the existing capability/trainer guards.
The native LoRA adapter now saves the versioned residual graph and digest via
model_topology_metadata, validating native configuration and actual block count.
Dense LoRA metadata remains unchanged.

Training component loading resolves the selected resume checkpoint through the
shared exact-path/latest selector and reads its topology before adapter setup
and device staging. Shared inherited-topology resolution distinguishes an absent
resume declaration from an explicitly dense checkpoint. A restored residual
model receives one native execution wrapper; transformer_original retains the
inner model, and the component dictionary points to the same callable wrapper.
Direct LoRATrainer.load_checkpoint first resolves without installation, then
uses the existing prepared/atomic adapter weight loading and installs topology
after successful weight restoration. Incompatible native configurations or
base/resume graphs refuse before checkpoint tensors are loaded.

Generation LoRA loading parses every selected file, resolves their topology
against the base native configuration and requires all selected declarations to
agree, including dense versus residual. Compatible metadata installs the graph
and native wrapper after adapter loading. A weak reference captures the original
inner model/graph; unloading restores the graph even if another component has
replaced the current model. Generation staging retains residual execution at
zero block swap and refuses cache/debug-skip/probe combinations before device
movement. Existing per-generation offloader teardown remains native.

A real tiny LoRA AV training step updates adapter branches, saves through the
actual MiniMaxH3LoRAAdapter, resumes through the actual LoRATrainer checkpoint
method into a fresh model, and restores identical branch weights and exact
native predictions. A dense-metadata resume into a residual base refuses before
load_file and preserves graph/weights. The actual generation LoRA loader reads
that same file, installs residual execution through the actual zero-swap staging
helper, reproduces training predictions within FP32 tolerance (2e-5 relative,
2e-6 absolute), and returns the dense base topology on unload. Mixed dense and
residual LoRAs refuse; cache/debug conflicts refuse before component movement.
Dry resolution preserves config/model state and bare-model installation requires
a trainer owner or an execution wrapper.

New residual/native LoRA tests plus existing cheap LoRA application/roundtrip and
training contracts pass 97 cases and 14 subtests. Changed modules compile and
import with CUDA uninitialized. No non-Qwen GPU was used. Public MiniMax-H3
configuration/preset/schema/editor, output execution metadata and native full
artifact loader integration remain pending; no full-training capability is
implied by this LoRA integration.

### MiniMax-H3 public spatial configuration and native inspection

MiniMax-H3 now belongs to the public residual runtime capability list following
its native CPU execution, train/save/resume and generation-adapter contracts.
The authoritative param_defaults preset is 50 blocks with outer [4,46) and inner
[23,27), each pooling 2x2 spatially: eight full-grid, 38 middle-grid and four
maximum-depth blocks. The normal training default remains inherit; configure
selects this preset only at its exact native depth, otherwise an explicit graph
is required. Existing LoRA-only training capability is unchanged.

The header-only inspector recognizes the distributed Comfy-style DiT keys and
uses the loader's native geometry synthesis with companion official config.
Root, official-directory and exact-file spellings yield identical information.
Noncontiguous block headers and LoRA-only artifacts refuse inspection. It
exposes the actual native patch-dependent spatial token scales and the packed
per-frame policy, permitting the shared graphical layer/scale/skip editor to
configure a model. The existing training panel discovers runtime architectures
from the schema endpoint rather than a frontend hardcoded list.

MiniMax-H3's causal temporal packing has no single constant pixel-to-latent
frame ratio. Its video geometry therefore declares temporal_scale_factor=null
instead of a guessed ratio; height/width retain positive native pixel-per-token
scales. OpenAPI and the frontend video geometry type accept this explicit null
and the packed_target_spatial_frames_v1 policy. The editor/summary use only the
spatial scales for preview and state that time/audio remain native. Static
review found and fixed the summary component's missing videoGeometry property
type, which previously appeared as an unnamed member in its props declaration.
Frontend build/type/render checks remain assigned to the owner.

The existing Comfy loader validates residual metadata against synthesized native
configuration before opening its tensor reader, then restores the native wrapper
after its unchanged strict mapping and quantization guards. Hybrid base/overlay
files must declare the same graph. Full loading and DiT-only reloading update
the restored topology and video geometry in current_model_info; reload computes
geometry before changing the live components. Test doubles now supply native
config attributes for this model-information contract.

CPU ASGI tests read the public schema, inspect native headers and validate the
50-block preset's exact 8/38/4 token distribution. A complete tiny real Comfy
checkpoint goes through the actual header synthesis, contiguous fused-QKV split,
SwiGLU conversion, meta construction and strict native state restoration. Every
weight is identical and residual predictions match within FP32 tolerance.
Tampered graph digests refuse before opening the tensor reader. The LTX-2/shared
video inspection and MiniMax-H3 component-order/LoRA checks pass 49 cases and 40
subtests; after adding native-loader and reload contracts, the MiniMax-H3 native,
DiT-reload and hybrid-lifecycle checks pass 69 cases and 14 subtests. Backend
modules compile/import, OpenAPI parses, and CUDA stays uninitialized.

This remains CPU/provisional support for MiniMax-H3. Additional native full
artifact formats, remaining architectures and consolidated owner frontend/workflow
acceptance require completion. GPU validation remains
restricted to Qwen-Image 2.1.


### MiniMax-H3 successful execution metadata (CPU)

The native wrapper records target frame/height/width and packed prefix, reference
video, audio and text token counts only after both output heads succeed. Failed
forwards clear the prior observation; metadata extraction refuses an unobserved
or changed graph. Captured graph dictionaries are detached from the live model.
The common generation path snapshots this record before LoRA teardown, then
attaches `dit_rewiring_actual` only after successful video/audio decode. Inpaint
and outpaint propagate it from their subrequests after successful compositing.
Both wrapper requests and common generation clear previous caller metadata.
The shared video writer persists the record in its existing JSON sidecar.

The real tiny native CPU forward observes 3x8x12 target tokens, 96 reference
video tokens, 6 audio tokens and 4 text tokens; its packed prefix remains 106
rows at every depth. Tests cover graph changes, failed output heads, detached
snapshots, dense restoration and actual software video encoding/JSON readback.
Together with native load/reload and hybrid lifecycle checks, 71 tests and 14
subtests pass; changed backend modules compile and import with CUDA untouched.
These fixtures validate execution and persistence, not MiniMax-H3 visual quality,
throughput or a full production generation/decode workflow.

GPU measurements and acceptance runs remain limited to Qwen-Image 2.1, per the
owner's resource constraint. Other model implementations use CPU contracts and
static inspection; their quality and throughput are explicitly unmeasured.


### MiniMax Music 3 native temporal flow foundation (CPU)

MiniMax Music 3 contains a noncausal 36-block flow-matching DiT in addition to
its autoregressive semantic-code generator. Only the flow block loop is rewired;
the language model, RVQ depth decoder, condition encoder, scheduler/chunk blend
and vocoder remain native. Generation-only and adapter/training refusals remain
in force. No new training registration is added.

The native flow DiT projects aligned acoustic latents and conditioning first,
then prepends one Fourier timestep token. Temporal residual pooling preserves
that token as the prefix and shrinks only the projected audio timeline. Native
partial RoPE is regenerated at each coarse integer grid. Native checkpointed
blocks, output projection and postprocess convolution stay in use. Arbitrary
chunk lengths gain zero hidden-token tail padding to the graph's required
alignment; the restored timeline is cropped before the native output heads.
This padding changes attention visibility and requires calibration just like
the other uncalibrated residual paths; it is part of the native config hash.

The 36-block preset is owned by param_defaults.py: [4,32) at half length and
[16,20) at quarter length, giving 8 full-length / 24 middle / 4 deepest blocks.
Config identity, actual depth, temporal-axis factors, input batch/conditioning
shape and conflicting execution features are checked before input projection.
Execution metadata is recorded only after successful output heads; failures
invalidate previous observations. Parameter counts and checkpoint weight keys
remain unchanged.

Native component directories restore uniform graph declarations before reading
checkpoint tensors. Their strict native/meta loading and quantization guards
remain in effect. The graph metadata exporter stores canonical native config
and schema/hash fields; real CPU checkpoint round trips preserve every weight
exactly and predictions within FP32 tolerance. Changed config identity and mixed
dense/residual directory declarations refuse before tensor materialization.

The native residual, directory-loader, API-default and capability checks pass
57 tests with CUDA uninitialized. Tests also compare checkpointed gradients,
confirm zero block updates exactly preserve the native heads/time token, and
verify the 8/24/4 preset. An additional existing staged-offload fixture has four
failures because its _MockComponent lacks language_model.model.layers; the same
four failures and seven passes reproduce using the unchanged 12328f37 backend
source in a fresh CPU process. They do not prove a passing offload suite.

This is an internal CPU foundation, not a public capability announcement. Flat
and GGUF artifact declarations, public model inspection/configuration, complete
flow chunk output provenance, production offload interactions and final workflow
acceptance remain to be connected or verified. No Music 3 weights or GPU quality/
performance runs are used. YuE2's separate NAR flow and shared AR KV prefix still
require their own architecture assessment and integration.


### MiniMax Music 3 flat safetensors/GGUF graph restoration (CPU)

Both existing flat flow+condition loaders now restore the residual graph rather
than discarding its metadata. Safetensors use the canonical artifact header
reader; GGUF uses its native string metadata entries with the same three
sushi.rewire schema/spec/hash keys. A missing declaration retains dense native
behavior. Partial declarations and bad hashes refuse before materialization.

For declared residual artifacts, the companion native transformer config is
resolved before tensor reads, checked against the topology's native config hash,
and compared with contiguous flat layer indices and required flow/condition
headers. Flat adapters, missing blocks and mismatched geometry refuse before
the eager safetensors reader or lazy GGUFStateDict is constructed. Safetensors
also compare the materialized metadata with the preflight declaration. Existing
fused-QKV remapping, condition-encoder precision, INT8 ConvRot guards and GGUF
supported-tensor-type restrictions remain in force. Topology is installed only
after both native components load successfully.

Real seven-layer flat fixtures pass through both production builders. Every
flow and condition tensor matches exactly; both restored residual predictions
match the reference within FP32 tolerance. Bad digests, partial declarations,
wrong config and missing layer headers are tested with materializers patched to
raise if reached. Native residual, loader, GGUF loader and flat-remap checks pass
76 tests and 2 subtests in 3.42s, with CUDA uninitialized. Changed backend files
compile and import. No production Music 3 weights or GPU runs are used.

Public model inspection/configuration, whole-song output provenance, production
offload acceptance and YuE2 remain outstanding. This extends the internal
artifact contract; it does not enable Music 3 training or advertise complete
public workflow support.


### MiniMax Music 3 flow-chunk provenance and generation scope (CPU)

The actual denoise_chunks path observes every conditional and unconditional
native DiT evaluation. All evaluations must retain the initial graph digest;
changing the graph during progress callbacks refuses the request. The pipeline
publishes its detached flow record only after every chunk completes, and clears
previous records before a new attempt. Failed flow requests expose no completed
record. Dense requests retain no residual provenance.

The record contains the canonical graph/hash, zero-token-tail policy, preserved
time-token prefix, coarse integer RoPE policy and chunks. Each chunk declares
its input_frame_start/end, native/padded audio tokens, actual inference step
count and flow evaluation count. scope=flow_input_chunks means these are offsets
within the supplied AR frame-hidden window, not global song offsets. Extend and
repaint may supply only a tail/window; existing audio edit provenance describes
where that window is placed. Summing overlapping chunk lengths is not a song
length or a speed measurement.

Fresh generation, extension, regenerate and rerender clear stale caller metadata
at entry and attach the completed flow snapshot only after successful decode and
waveform splicing. The common audio writer persists dit_rewiring_actual in its
existing FLAC JSON sidecar. Returned frame-code sidecars keep their established
AR replay contract; residual metadata does not change their code sampling.

A real native CPU DiT, condition encoder and FlowMatchEulerDiscreteScheduler run
four overlapping test chunks with native lengths 13/13/13/9, padded lengths
16/16/16/12, and six CFG evaluations per chunk over three steps. Tests cover
failed flow, graph changes, detached snapshots, dense absence and actual FLAC/
JSON output. A backend fixture uses this actual flow stack with mocked AR and
waveform decoding, confirming publication after success and absence after a
decode failure. It does not evaluate vocoder quality or real song generation.

Testing also exposed the existing rerender path's unbound transformer reference
at flow staging; it now binds the loaded component alongside the AR components.
The staged-offload mock supplies model.layers and transformer_blocks matching
the production structural contract, resolving its previously recorded four
fixture failures. Native residual, staging/cleanup, extension, repaint and
progress checks pass 74 tests and 2 subtests in 5.44s. Changed modules compile and
import with CUDA uninitialized. Production GPU offload remains unmeasured.

Public model inspection/configuration and model-info display, YuE2 integration
and final owner frontend/workflow acceptance remain outstanding. GPU acceptance
continues to use Qwen-Image 2.1 only.


### MiniMax Music 3 restored model-info and temporal graph display (CPU)

The loaded-model information now includes the validated, detached residual spec
and temporal geometry when the native flow component is rewired. Dense models
report a null spec without inventing a temporal geometry. Geometry comes from
the actual condition encoder's resampling rates/hops; its out_dim must match
the flow DiT condition_dim. Missing conditioning geometry and mismatched channels
refuse instead of displaying a guessed scale.

Preview frames are the condition encoder's exact native CHUNK_FRAMES resampling
calculation, matching its forward arithmetic. The released rates produce
44100/512 = 86.1328125 latent frames/sec and 689 native tokens per 200 AR-frame
chunk; patch_size is 1 because the flow DiT projects each latent frame directly.
The four-deepest-block preset pads this preview to 692 transport tokens. This is
one flow-input chunk, not the complete song, and no throughput claim follows
from the diagram.

The shared restored-topology summary displays Music 3's chunk duration/token
count, preserved timestep token and unchanged autoregressive stage. Its temporal
preview already uses the common zero-tail padding calculation and graph API.
OpenAPI now permits any positive numeric temporal frame_rate, including the
native fractional value; preview_frames can originate from native chunk geometry
as well as generation defaults. Existing frontend frame_rate types are number.
No frontend build/type/render command is run; source and API types are reviewed
statically as required by the owner.

CPU tests exercise the actual DiffusionPipelineManager Music 3 loading branch
with real tiny native flow/condition components and a fixture component loader.
They check detached metadata, native rates/preview frames, channel/geometry
refusals and dense absence. Residual contracts plus encoder selection/reload
fixtures pass 36 tests and 2 subtests; five tests targeting external real-model
headers are excluded. Changed modules compile/import, OpenAPI parses, and CUDA
remains uninitialized. Public artifact inspection/configuration, YuE2 and final
owner workflow acceptance remain outstanding.


### MiniMax Music 3 public artifact inspection (CPU)

The existing models/rewiring/inspect endpoint now resolves Music 3 root,
official, native flow directory/full-file, flat safetensors and GGUF references.
It reads native companion flow/condition configs and checkpoint headers only.
The response contains native config/layout hashes, restored graph/digest,
36-block 8/24/4 preset, a defaults-owned temporal single-region starting graph,
and the actual native flow-chunk temporal geometry. No model or weight
materialization is required to preview layer depths and residual skip edges.

Inspection checks contiguous native/flat block indices, input/output/Fourier
projection dimensions, per-block Q or fused-QKV dimensions, conditioning output
channels and resampling rates. Native directories require consistent topology
across files/indexes and refuse duplicate header tensors or quantized native
component dtypes. GGUF inspection keeps the existing supported-type guard.
Adapter-only files refuse, including those stored inside a native component
directory. Non-object companion configs now yield ValueError and the endpoint's
422 response rather than an AttributeError. Inspection/presets do not register
Music 3 training or expand the training runtime capability list.

Real 36-layer tiny headers yield identical inspection responses for six source
references. In-process ASGI requests validate the 692-token padded preview's
exact 8 full / 24 middle / 4 deepest block distribution and refuse adapter-only,
malformed-config and quantized-native artifacts. Eager/lazy weight readers and
native model construction are patched to raise if reached. Native residual,
shared inspection, Music 3 loader and GGUF loader checks pass 61 tests and 2
subtests; the final nested-adapter classification check also passes independently.
Changed modules compile/import, OpenAPI parses, and CUDA remains uninitialized.

This provides public inspection and restored-generation display for provisional
Music 3 support. A generation-only graph editing/export workflow and production
GPU offload acceptance are not claimed. YuE2's NAR/AR boundary and final owner
frontend/workflow acceptance remain outstanding; GPU verification stays limited
to Qwen-Image 2.1.


### YuE2 acoustic NAR block-loop foundation (CPU)

Residual execution now has an architecture-owned NAR-only stream. The 28-layer
preset uses 8 full-resolution / 16 middle / 4 deepest blocks, with temporal
factors [2,1] at both levels. The preset lives in param_defaults.py. It does not
add YuE2 to the training runtime capability list: Phase-A ABC-planner LoRA
continues to train the AR half only; acoustic NAR training remains unsupported.

CachedNAR still prefills the complete native AR prefix exactly once and retains
its restricted visible KV set unchanged through the midpoint solver. Only the
acoustic hidden-state timeline is pooled. START and END hidden tokens become
two preserved prefix tokens during residual execution and return to their native
head positions afterward. Their absolute rotary positions remain native. Coarse
acoustic positions use integer coordinates; padding positions start after END
so they never alias a boundary. Native learned positions and timestep embeddings
are applied before transport; zero hidden-state tail padding reaches the nested
factor alignment and is removed before the original norm/output head. Padding
that exceeds the native context limit refuses before AR prefill.

Topology identity hashes native configuration and zero_hidden_tail_v1 policy.
Restore/export helpers validate the acoustic stream, actual block depth and
64-channel VAE contract. Each cached engine freezes a detached graph; adding,
removing or changing the model graph after prefill refuses before velocity work.
No parameter tensors, state-dict keys, AR modules or solver arithmetic change.

CPU contracts exercise the actual native tiny MoT, grouped-query attention,
restricted AR cache and midpoint solver. A 13-frame input pads to 16 acoustic
transport tokens: the seven test blocks see 18/10/6/6/6/10/18 tokens including
boundaries. Zero NAR residual updates reproduce the dense native head exactly,
including odd-length padding. Tests check immutable AR values/storage, identical
weights, preserved boundary rotary positions, no duplicate padded positions,
context refusal, configuration mutation and the 8/16/4 public preset.

This is a provisional native execution foundation, not a complete public YuE2
workflow. Loader-driven checkpoint restoration, header-only artifact inspection,
model-info display, generation provenance and generation-only graph export still
need connection. The existing fixed complete single-file/embedded-tokenizer
checkpoint boundary and AR-only training boundary must remain authoritative.
All GPU validation remains limited to Qwen-Image 2.1 by the owner's instruction;
YuE2 quality, throughput and production quantized execution are unmeasured.
