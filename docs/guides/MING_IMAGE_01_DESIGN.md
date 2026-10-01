# Ming-Image 0.1 integration contract

Status: implementation in progress. CPU artifact identity/header preflight is implemented; native source import and tiny DiT execution are verified; native component loading and conditioner training ownership are verified with a small CPU package; official-checkpoint inference, trainer/API and UI wiring remain in progress. Established 2026-10-01 against SushiUI
`2b28c3b4`. This contract preserves the requested Qwen-Image 2.1 feature
scope, including residual U-Net conversion. It is not a claim of measured
Ming support. [Completed source audit](../audits/MING_IMAGE_01_SOURCE_AUDIT.md)
records exact upstream source/model revisions and licensing constraints.

## Verified implementation boundary

The native-package inspector and manifest writer are implemented in `backend/core/models/ming_image_01/artifact.py`, with a CPU-only CLI at `tools/inspect_ming_image_01.py`. The targeted artifact suite passes 22 tests, including profile identity, processor drift, package escape, shard/index errors and malformed weight spans. Both pinned official configuration snapshots pass config-only inspection. No model weights, CUDA context or running API were used. This evidence proves preflight only; it does not prove native state loading or functional generation/training.

Pinned native source is vendored with [file-level provenance](../legal/MING_IMAGE_01_PROVENANCE.md). Thirteen additional CPU tests verify explicit native imports, both padding profiles, reference/ragged input forward/backward, gradients through both conditioning inputs and meta-only transformer parameter accounting. CUDA initialization is forbidden in these tests. This does not establish full MLLM conditioning, full-checkpoint inference or GPU quality.

The native component loader and selectable language/vision/connector/query/projection groups are implemented. Seven conditioner tests cover full-language gradients through both conditioning streams, projection-only training, frozen cache eligibility, projection state mismatch and a complete tiny native-package save/load. Weights and rotary buffers match exactly; condition outputs match within FP32 tolerance (`rtol=1e-5`, `atol=1e-6`). The loader rebuilds custom nonpersistent rotary buffers after low-memory loading. Official-weight loading and GPU quality are not established by this evidence.

Native sampling now has a task/profile adapter, transformer argument bridge, zero CFG branches, ordered Layer result groups and cancellation cleanup. Sixteen targeted CPU sampling tests pass: flow sign/time and Euler endpoint, CFG formula, deterministic RGBA Design output, native reference editing, Layer N+1 output, explicit frame-major regrouping, cancellation/reuse and preflight refusals. A consolidated affected suite also passed 40 tests before the final sampler validation additions. This is small-model functional evidence; the platform API/UI and official-model quality gates remain open.

Ming target-only residual U-Net execution and checkpoint topology restoration are now implemented. Eight CPU tests cover nested transport with ragged auxiliary lengths, independent Layer frame grids, all-main-block gradients, identity reconstruction, gradient checkpointing, profile identity/resume conflicts, the deepest-four-block preset and exact native DiT parameter counts. The seven conditioner tests also exercise a full tiny package with saved topology: strict loading restores it, and both package and manifest inspection report the same topology and parameter census. The affected batch passes 15 tests. Shared inspection/editor contracts remain unchanged; Ming previews describe per-target-layer grids and preserved auxiliary-token overhead. Trainer registration, common trainer/export API integration and advanced generation wiring are still pending.

The native flow objective now accepts explicit sigma, ordered target-frame supervision weights, ragged reference latents and aligned dual-stream CFG dropout. Ten CPU tests cover the native time/sign, target-only weighted loss, both profiles' U-Net/conditioning gradients, malformed supervision refusal and independent component learning-rate ownership. These are objective primitives; the common trainer, datasets, checkpoint/resume and API registration remain unconnected.

Native component-bundle export is implemented in `checkpoint.py`: all six native components plus tokenizer/processor are written to a sibling staging directory, verified, then published without overwriting an existing destination. DiT uses the shared sharded-state writer with identical topology metadata on every shard and index. Export requires explicitly CPU-offloaded components and does not change residency or training mode. The seven conditioner/package tests now also round-trip a forced-sharded full bundle, compare all DiT/conditioner/VAE states and conditioning outputs, and verify collision refusal and mid-export cancellation cleanup. This proves the native inference bundle format; common trainer optimizer/RNG/scaler resume and export API integration remain pending.

Native prompt/vision preparation and generation orchestration now share `prompting.py` and `generation.py`. Official HUMAN/system/assistant templates and processor vision geometry are retained; VAE references separately use the explicitly requested 16-pixel-aligned grid, preserve RGBA channels and remain ordered. Mixed reference counts are retained for training but rejected for a rectangular generation batch. Nine CPU tests cover native chat/image tokens, reference alpha/order, text batches, full language/vision/projection gradients, end-to-end tiny Design image-edit and Layer RGBA generation, early cancellation and caller-owned residency-context release after step failure. Platform generation/residency is connected below; frozen-conditioning caches remain pending.

`inspection.loaded_model_info` now provides the loaded structure snapshot for platform wiring: unique DiT/language/vision/connector/projection/query/VAE parameter ownership, trainable counts, logical dtype/device bytes, separate serialized header storage, refiner/main depth, checkpoint topology and conditioner cache eligibility. It makes no VRAM measurement and explicitly excludes buffers/quantization payloads from loaded parameter-object counts. The seven package tests check JSON serialization, exact full ownership totals, DiT analytical census agreement and trainability/cache transitions. Current-model API/frontend consumption is implemented in the following stage.

Ming native directory/manifest detection, CPU component loading and lifecycle registration now use the common `ModelLoader` and `DiffusionPipelineManager`. The current-model response carries the complete structure snapshot; the model selector renders component parameter distribution, trainable/frozen ownership, dtype/logical bytes and separate serialized storage alongside the restored U-Net graph. Native Layer frames are explicitly classified as image layers, never video. Ten affected CPU tests pass, including real common `load_model`, topology restoration, current-model/wiring fields, scan classification and inventory cleanup. Frontend types/OpenAPI match and have been reviewed statically; owner build/type checking remains pending. Basic native generation is now connected: `generation_available=true` denotes functional dispatch, not official-weight GPU acceptance. No running API or GPU was used.

## Identity and ownership

Use one architecture key, `ming_image_01`, and two explicit artifact profiles:
`generation_edit` (Design) and `layer_decompose` (Layer). Register generation
in `ModelType` and training in `ARCH_REGISTRY`; capabilities depend on the
verified artifact profile. Keep this architecture distinct from Qwen and
Z-Image even when sharing utilities. Never detect solely from a directory name.

Create native model/artifact/loader/conditioning modules under
`backend/core/models/ming_image_01/`, a generation backend under
`backend/core/pipeline_backends/`, and architecture/ops/adapters under
`backend/core/training/`. Reuse `backend/core/adapters/`, shared component
wiring, memory management, partition planning and `backend/core/rewiring/`
through explicit Ming adapters. Algorithm/decomposition acceptance belongs
only in the generation/training tables in `core/adapters/capability.py`.

Manifest identity includes source revisions, profile, component hashes and
configs, tokenizer/processor versions, dtype/quantization per component,
conditioning projection/query configuration, padding policy and multi-frame
contract. Keep DiT, MLLM, vision tower, connector, projections/queries and VAE
independently identifiable. Cataloguing, import/conversion, model selection,
component resolution and checkpoint inspection must handle released packages,
floating full-training bundles, inference exports and LoRA files. Preflight
validates missing shards, index keys, shapes, channels, profile and topology
before GPU allocation. Never load arbitrary remote custom code at runtime.

## Qwen parity obligations

The evidence column points to existing code/contracts, not a promise that all
optional algorithms work on all architectures. Refusals currently made by
Qwen are retained rather than counted as implemented features. Items needing
Ming-specific mathematics remain in scope until verified or explicitly
resolved with the owner; they must not silently become no-ops.

| Surface | Current Qwen evidence | Required Ming behavior and acceptance |
|---|---|---|
| Discovery, artifact conversion, component precision | `core/models/qwen_image_21/{artifact,loader}.py`; Qwen design sections 4–6 | Both profiles round-trip; separate component hashes/precision; CPU preflight rejects wrong/missing components |
| Text generation, deterministic seeds, sizes, steps, CFG | `core/pipeline_backends/qwen_image_21.py` | Design native outputs against pinned upstream; record effective bucket, output size and native CFG formula |
| Native edit and multiple references | Same backend and vendored pipeline | Design references follow both VLM and VAE preprocessing in preserved order; derive supported count from native packing tests, do not copy Qwen's limit of ten |
| Inpaint and outpaint | Same backend; shared generation routes | Native edit plus protected-pixel composite, masks/crops/alpha preserved; Layer cannot masquerade as Design edit |
| RGBA and latent workflows | Qwen pipeline; shared latent cache/output persistence | Four-channel pixels and 16-channel latents through upload, preview, cache, PNG export and gallery; Layer grouping is additional scope |
| Reference Guide | Qwen pipeline and reference-guide configuration | Ming flow-latent blend with its normalization, strength/window and protected channels; oracle for blend endpoints and exclusions |
| Style-KV and style-combine controls | Qwen `reference_guides`/`style_transfers` pipeline paths | New joint-attention style operator, mask/reference isolation, stack/combine/step windows; Qwen's segmented prefix mechanism cannot be copied |
| Prompt enhancement | `core/extensions/qwen_image_21_prompt_upsampler.py`; `Qwen21PromptUpsample.tsx` | Ming templates through existing local provider/cache, preview/apply/cancel/error states; no automatic external service call |
| LoRA load/unload, strengths and windows | Qwen backend `AdapterSession`; `core/models/qwen_image_21/lora.py` | Explicit target map for attention/refiners/FFN as configured; multiple adapters, shape/base mismatch errors; cleanup and topology restoration in finally |
| Full DiT and/or TE training, independent LR | `core/training/ops/qwen_image_21_ops.py`; component wiring; Qwen design 9.1 | DiT and full Ming conditioner independently selectable; trainable MLLM/vision/connector/projections/query ownership explicit; frozen VAE; live graphs when a conditioner component trains |
| LoRA on floating/quantized base | Qwen training adapter and ops | Ordinary supported algorithms through capability tables; exact target/key mapping and merge/export; quantized forward/backward identity recorded |
| Training objectives | Qwen ops guidance selection/loss and sigma-null schedule | Native time/sign oracle, reconstruction loss, guidance-loss schedules/mix/cap and sigma-null conditioning; zero branch replaces encoded empty text; test gradient routing and frozen teacher |
| Data/cache/buckets and diagnostics | Qwen ops prompt/VAE encoding, debug view; shared trainer | Alpha-aware datasets, condition/latent cache, online/swap encoding, dynamic crop, bucket validation, samples, progress and debug latent views; cache invalidated for changing trainable components |
| Training memory/precision | Qwen ops block swap/ConvRot; shared trainer | Gradient checkpointing main/refiner blocks, block swap, activation offload/dispatch, AMP/FP32-master policy, optimizers/accumulation/clipping; frozen and trainable conditioner lifetimes separately managed |
| Quantization and export | Qwen loader/artifact and ConvRot training ops | DiT/conditioner capability-specific BF16 and INT8 ConvRot paths; backward cache/prefetch controls, base-forward metadata, exact floating resume versus inference-only quantized export |
| Partitioned inference/training | `core/inference/qwen_image_21_partition.py`; Qwen ops `train_step_partitioned_backward`; `partition_global.py` | Ming partition adapter with complete target coverage, threshold/fixed splits, halos, deterministic plans, query chunking and checkpoint-block controls; optional global adapter training/save/load; distinguish exact query chunks from approximate changed context |
| K/V and condition caches | Qwen `use_kv_cache` | Cache frozen MLLM/connector outputs safely. Exact denoising prefix K/V cache is mathematically inapplicable; explicit capability refusal and UI explanation, not reuse of stale state |
| Residual U-Net | `core/models/qwen_image_21/residual_unet.py`; `DIT_REWIRING_DESIGN.md` | Ming main-block runtime, topology metadata, load/resume/LoRA reconciliation, generation/training inspection and graph editor; details below |
| Save, resume, train from exported full checkpoint | Qwen ops; shared checkpoint export/training store | All component state, optimizer/RNG/scaler/step/batch-plan state, trainability, caches and topology restored; fresh training from a full U-Net bundle and inference export has explicit format rules |
| UI/API wiring | `api/param_defaults.py`, `openapi.yaml`, frontend `utils/api.ts`, generation/training panels | Defaults/schema/types/request serialization/queue/samples/import/resume paths connected; capability visibility, validation/cancellation/error states and graph/parameter displays reviewed together |

Qwen currently refuses NAG/NegPip/regional prompting, ControlNet, ReLoRA,
TE LoRA, VAE swap, generation block swap, compiled Flex and unvalidated
FBCache/Spectrum/TREAD/BlockSkip. Do not advertise these merely because shared
controls exist. Ming's DiT SDPA and MLLM eager/Flash paths need separate
capabilities; do not expose an attention backend without native mask parity.

## Native generation and Layer output

All defaults originate in `backend/api/param_defaults.py`, exposed by schema
endpoints. Proposed profile defaults follow upstream: 12 steps, Design CFG 1,
Layer CFG 2. Other dimensions/limits must be resolved against pinned upstream
size logic before adding defaults; avoid hardcoded frontend fallbacks.

Preserve the native scheduler, output negation and model clock. True CFG uses
zero tensors for both negative conditioning streams. An arbitrary negative
prompt has no released native path: either implement and validate an explicit
extension or return a capability error when it is nonempty. Label native CFG
semantics in the UI; Qwen's numerical scale convention is different.

A proposed `/api/v1/generate/layer-decompose` operation accepts a verified
Layer checkpoint, RGBA reference, integer requested layer count, optional
instruction, dimensions, seed, sampling and compatible adapters/topology.
Write OpenAPI path/schema/examples first, then route and client. It must use
the existing generation lock, queue, progress, cancellation and persistence
lifecycle, not a second scheduler or server. Validate count versus any layer
count embedded in instruction text; never silently override the typed count.

Return a result-group manifest with task/profile, source image identity,
seed, requested count N, effective target-frame count N+1, effective bucket,
RGBA files and ordered roles: slot zero auxiliary canvas and N layer slots.
Confirm the semantics of slot zero numerically before calling it a flattened
composite. Preserve layer order/alpha, atomic group saving, reloadable gallery
associations, checkerboard preview, individual downloads and group export.
Batch-image count, requested layer count and internal F must be distinct UI
and API fields. Design rejects decomposition; Layer rejects txt2img/edit.

Generation cancellation aborts remaining sampling and output persistence,
cleans adapters/partition runtimes/component ownership in finally, and emits
the normal cancellation status. Upstream's interrupt-loop behavior must not
be adopted without matching SushiUI cancellation semantics.

## Training and component lifecycle

The native training-batch bridge now preserves RGBA target frames independently
through VAE encoding and keeps ragged edit references outside target supervision.
`MingTrainingBatch` slices both conditioning streams, frame weights and references
together for microbatching. Explicit per-frame weights determine slot-zero
supervision. `ming_image_01_ops.train_step` consumes shared MNT noise and timestep
sampling, applies the native clean-minus-noise sign, weighted reconstruction loss
and per-item exposure metrics. Tiny CPU VAE/MLLM/DiT backward and optimizer updates
verify language gradients and independent alpha/frame order. Shared trainer
registration, dataset role manifests, optimizer resume and UI/API wiring remain
pending; this bridge alone does not enable training requests.

The shared full-parameter trainer factory can now select a native Ming adapter
when its loader sets the Ming flag and explicit conditioner groups/rates.
The adapter owns frozen/trainable modes, unique optimizer membership, component
gradient accounting and full six-component bundle export. Save records profile,
step/epoch and named learning-rate groups in an atomically published
`training.json`; it restores component devices after success or failure.
This file describes weight provenance, not optimizer/RNG/scaler resume state.
Native trainer registration is staged; shared dataset/loop and exact-resume orchestration remain pending.

`ming_conditioner_groups` and `ming_conditioner_lrs` now round-trip through
central training defaults, OpenAPI/Pydantic, YAML and edit/preset transport.
Null groups inherit the global TE switch; explicit empty groups freeze every
conditioner group. Missing rates inherit TE/base LR and explicit zero is kept.
Unknown/duplicate groups, coerced booleans/strings, negative and nonfinite rates
are refused. Create/update boundaries reject these options for other models or
methods. The full adapter resolves this contract; the training panel exposes
component checkboxes and per-group rates for Ming full fine-tuning. Frontend
verification is static and end-to-end Ming training remains pending.



The default full-parameter scope can freeze the conditioner, but full TE
training is required parity, not a future optional omission. Provide explicit
component groups and independent learning rates. Frozen conditioning can be
CPU-offloaded or cached; trainable groups keep their graph, disable stale
cache, participate in optimizer/checkpoint state and support exact resume.
Upstream detaches the direct-VLM hidden states before projection; full TE training must explicitly restore the intended graph, with gradient assertions for both conditioning streams. Do not remove detachment blindly in frozen/cache mode.

Stage component residency (MLLM/vision/connector -> DiT -> VAE decode) rather
than leaving every large component resident. Active expert count is not a
resident-weight memory estimate.

Design datasets contain RGBA targets/captions and optional edit references.
Layer datasets additionally require ordered alpha layers plus an input canvas
and consistent instruction/count. Bucket by H/W and layer count, preserve
reference/target roles and use paired crops. Establish an explicit target
for the auxiliary canvas before enabling Layer training; do not invent a
supervision label by flattening layers in an unverified order. Training loss
covers supervised target frames only, excludes appended references and all
padding, and must survive ragged batches. Fail dataset preflight early if
required layer roles or alpha are missing.

Establish model clock, interpolation, prediction sign and loss target in a
CPU scalar/small-tensor oracle; shared Qwen `eps_minus_x0` is not assumed.
Then verify native forward/backward for ordinary, null and guidance objectives.
Trainable TE, LoRA, partition-global and U-Net configurations require explicit
compatibility validation before model loading. Mixed objectives must preserve
the intended gradient routing rather than train the reference branch.

Cache keys include profile/config hashes, tokenizer/processor/instruction
version, both conditioning streams, all reference RGBA contents/order,
bucket/crop/layer count, VAE normalization and active trainable components.
Checkpoint resume restores topology and components before binding adapters,
optimizer slots or cached plans. Explicit UI overrides conflicting with saved
structure must fail with a useful diff. Quantized inference exports must not
be accepted as exact optimizer-resume checkpoints.

## Residual U-Net topology and inspection

Reuse the accepted two-stage default, deepest four main blocks. For 30 Ming
main blocks the proposed half-open regions are outer `[4,26)` and inner
`[13,17)`: eight full-resolution blocks, eighteen middle-resolution blocks,
four deepest blocks. Image and context refiners remain full-resolution.
1024-pixel targets have 4096 -> 1024 -> 256 tokens per layer; 2048 targets
have 16384 -> 4096 -> 1024. Layer's 512 target gives 1024 -> 256 -> 64.
These are token-grid counts, not measured speed/VRAM or total token counts.

Do not reduce caption, reference or learned-padding tokens. Build an explicit
packing map that gathers those tokens as preserved state and each target
spatial grid separately; invert the permutation to native order at the region
exit. Rebuild native positions/masks correctly at each resolution and retain
joint updates to preserved state. Use independent spatial transport per
layer, with F identities intact and skip joins bound to matching layer/grid.
The shared runtime's grouping support is a utility, not evidence that a
Z-Image/Qwen adapter already handles Ming's layout.

Metadata uses shared versioned rewiring schema and structural fingerprint,
including architecture/profile, main-block count, region boundaries/scales,
skip endpoints/merge/transport, frame grouping and padding-layout identity.
Reject unsupported versions, invalid nesting, shape/divisibility mismatches,
wrong profiles, contradictory LoRA requirements and topology/partition-global
combinations before allocation. Preserve Qwen's U-Net/partition-global
exclusion initially; only open coexistence after a separate correctness gate.

A CPU packing oracle must cover learned and masked padding, variable caption
lengths, multiple references/layers, ragged batches, identity transport and
nested down/up/skip reconstruction. A tiny-model backward/save/load test
checks every intended gradient and exact topology restoration. Full-model
U-Net quality and speed acceptance remains a later GPU gate.

The shared graph editor and checkpoint inspection API must display actual
loaded/training structure: main versus refiner blocks, resolution/rate at each
region, skip endpoints, per-layer target tokens, preserved-token overhead,
selected checkpoint topology and proposed override. Provide loaded/pending
states, errors and restoration after checkpoint/model switches. Show a
component parameter census (DiT, refiners, MLLM, vision, connector, projections,
queries, VAE), trainable/frozen counts and storage dtypes separately. U-Net
transport adds no learned parameters; depth and frame count still change
activation/attention cost. Never equate token reduction with proportional
step/sec or resident parameter-memory reduction.

## Generation settings transport

Native Design defaults (12 steps, CFG 1) and Layer defaults (12 steps, CFG 2)
are served from `param_defaults.py`. Profile selection preserves explicit
request values. `ming_image_01_layers` has a central default of 3 and minimum
1, transported through all four image endpoints, queue/loop inheritance and
Studio recipes. Design still produces one image; Layer produces the native
auxiliary canvas plus N ordered RGBA layers. The shared Layer control displays
that distinction and reads the served default rather than duplicating it.

CPU contract tests check JSON rejection of invalid counts, multipart consumers,
profile defaults, explicit overrides and OpenAPI synchronization. Frontend
static review includes model/profile switches, queue allowlists, transfers and
recipe restoration. Settings and rich-result transport are connected;
basic native sampling is connected; advanced parity and official-weight GPU acceptance remain pending.

## Ordered output persistence

`MingGenerationOutput` owns decoded images for one request; model-global last
results are not used. Profile validation requires exact native roles/counts,
RGBA mode and requested dimensions before any write. The grouped persistence
helper stores every frame in batch-major order, keeping a UUID group identity,
frame role/index and profile in PNG and gallery parameters. Its response keeps
the first record in `image` for existing callers and all rows/roles in
`images`/`groups`. Skip-gallery still saves all group filenames without rows.

Explicit group filenames use exclusive creation, preserving older files on
collision. All gallery rows commit together; cancellation or any pre-commit
failure rolls back the rows and removes this request's outputs/thumbnails.
SQLite/PIL CPU tests check byte-identical RGBA recovery, multiple batches,
metadata equality, filename uniqueness, collision refusal and failure cleanup.
All four image routes recognize rich outputs before generic latent/single-image
saving. The envelope deep-copies generation-time model information; final saves
cannot acquire another loaded model's identity after the GPU slot is released.
Route completion preserves source/mask metadata and all ordered groups in the
status response. CPU tests exercise finalization and route branch ordering.

Queue results retain group metadata across panel unmounts, append every frame
to the recent-results feed, and offer a complete per-panel group viewer with
role labels and original RGBA PNG downloads. Studio imports every image into
its assets/result shelf, while the primary output owns the initial timeline
clip. Loop continuation still uses the primary output. Frontend review is
static; owner build/render validation remains pending. Basic native platform sampling is connected and CPU-verified. Thumbnails
may use a white alpha preview, while full PNG files retain native alpha.

## Native platform generation

The shared manager dispatches native Ming text-to-image/edit requests and
ordered Layer decomposition through `MingGenerator`. Profile defaults come
from the central resolver; explicit parameters and seed zero are preserved.
Pixel dimensions must fit the native 16-pixel grid. The processor's native
resolution bucket follows the requested pixel area and never substitutes the
output grid. Native CFG uses zero conditioning, with nonempty negative prompts
explicitly refused until an extension passes its semantic gate.

`MingPhaseResidency` moves the conditioner first, returns it to CPU, then stages
DiT/VAE, without changing storage dtype. Each exit attempts cleanup of every
component; staging/callback/cancellation failures unwind and the next request
can recover. CPU text encoding is honored. This is whole-component offload,
not a low-memory or minimum-VRAM promise; quantization, block swap, frozen cache,
adapters, style and reference-guide integration remain in scope and pending.
Requested unwired runtime options are refused before heavy residency.

Design inpaint supplies a masked-alpha reference and composites all four
channels, preserving unmasked RGBA pixels. Design outpaint shares placement/
canvas geometry, restores the placed RGBA crop without a mask and preserves
those pixels exactly. Layer inpaint/outpaint is refused before model staging.
Restored nested U-Net runs through the same native reference/Layer layout and
records its actual spec/digest. Current-model `generation_available=true`
indicates connected basic dispatch; it does not certify official-model GPU
quality, memory or throughput. Progress currently reports denoising steps
without an incompatible TAESD preview.

Thirty-one affected CPU tests verify the actual manager, deterministic
reference batches for both profiles, native RGBA operations, independent
Layer frames through restored nested U-Net, early invalid-request refusal,
request-owned generation-time model snapshots and cancellation/callback reuse.
All four image routes retain typed API validation errors as 400 responses; unexpected failures remain 500. Native reference selectors are visible for Ming in txt2img/edit panels.
Official weights and live API/GPU were not used. Shared trainer, advanced
parity and owner frontend/GPU acceptance remain pending.

The common trainer's model-load and checkpoint-as-base dispatch now select the
native Ming loader, preserve restored topology and resolve conditioner ownership
before encoding/cache decisions. Its registered handler uses the native 16-channel,
5D layout, independent RGBA VAE frames and the clean-minus-noise sign, and refuses
an unaligned generic context. Registry membership is staged: training methods are
unavailable in capability metadata and normal run preflight refuses before dataset
or weight loading until the common dataset/loop/resume paths are connected.
Loading weights as a base is verified; exact optimizer resume is not implied. The affected native loader/handler suite passes 11 CPU tests. Shared capability/options checks pass; five YuE2/Chimera velocity declaration failures also reproduce with the pre-Ming test and registry from `84f9c93e`, and those model sources are unchanged. These failures are tracked separately from Ming numerical evidence.

## Verification and release gates

Main's running API/GPU are unavailable for this task. Source/config review,
AST/shape reasoning and isolated CPU tests are allowed; weights, CUDA imports,
API requests, server changes and live benchmarks require a later resource
handoff. Do not start or restart services directly.

During implementation, batch checks once per logical milestone. Backend
changes always receive changed-file `py_compile` and real import with the
repository's documented CUDA-trigger stubs. Use targeted meaningful CPU
fixtures for artifact/layout/objective/checkpoint correctness, not repeated
whole-suite runs after documentation or cosmetic changes. Expand tests only
for new changes, failures or unresolved correctness concerns.

Critical gates: licensing/provenance, native mask/RoPE/conditioner parity,
time/sign/CFG, RGBA/layer-role integrity, topology restoration, full TE graph
and checkpoint ownership, partition coverage/loss, safe cancellation and
capability refusals. Static scaffolding can continue while numerical GPU
acceptance is pending, but capabilities remain unavailable until their gate
passes; no production correctness claim is made by a CPU-only import.

Frontend owner-run build/type checking stays pending without stopping
implementation. Agents review `frontend/src/utils/api.ts`, shared defaults,
OpenAPI contracts, generation/training panels, queue and cancellation/error
paths statically. Report that separately from owner build results. Noncritical
presentation issues are collected for the final owner check; actual broken
serialization or destructive state transitions are critical.

When resources become available, run one consolidated native-versus-Sushi
GPU acceptance matrix: Design txt2img/edit/inpaint/outpaint/references/RGBA,
Layer N+1 outputs, ordinary/quantized LoRA and full DiT/TE backward, save/resume,
new training from full checkpoints, U-Net inference/training/restoration,
partition/global adapters and style/reference extensions. Record component
VRAM, startup residency, step/sec, dimensions/tokens, precision and quality
separately. No minimum-memory promise or calibrated U-Net quality promise is
made before measurements. Update model facts, architecture reference,
training/API guides, examples and capability documentation only as support
actually ships.

## Decisions requiring owner review before numerical release

- Exact Qwen prefix-K/V caching cannot transfer; use frozen-conditioning cache
  and an explicit unsupported capability for exact denoising-prefix reuse.
- Style-KV and nonempty negative prompts require new Ming behavior and a
  quality/semantic acceptance gate rather than silent Qwen substitutions.
- Layer training needs verified supervision for slot zero and an ordered RGBA
  dataset contract. Native inference can be developed before that data exists.
- GPU/API verification waits for a separate resource handoff; its absence does
  not block offline implementation, commits or frontend static review.

### Explicit native dataset records

The [training data contract](../reference/architectures/ming_image_01_training_data.md)
now defines ordered RGBA targets, explicit auxiliary-canvas supervision weights,
root-contained references, matching crops/flips and frame-count batch identity.
23 dedicated CPU tests pass, including actual tiny native VAE/conditioner/DiT
backward and optimizer updates through the registered handler. Shared image
encoding retains Ming alpha. Shared discovery and epoch refresh now attach these records; compatible
frame/grid boundaries are enforced before batch ledgers. Content/roles/weights
also enter shared fingerprints, concept digests and resume ledgers. Cache,
optimization-loop and exact optimizer-resume orchestration remain pending; capabilities
stay unavailable until connected. No main API, official weights or GPU was used.


### Shared forward/backward transport

`shared_step_inputs` carries the two conditioning streams as one differentiable
transport tensor with explicit sequence/width layout, zero padding only unused
feature columns. Native dispatch reconstructs both streams, target-frame weights
and ragged references before calling the registered handler. The common
`_execute_forward_backward` now selects this native path; conventional backward,
gradient accumulation, fixed MNT noise and per-item metrics remain shared.
The common two-stage microbatch helper returns the accumulated transport
cotangent to the conditioner once after chunk backwards.

Seven dedicated CPU cases verify transport/layout rejection and both profiles'
real tiny native backward. Full-batch versus three microchunks matches losses,
per-item metrics and gradients across DiT and all five conditioner groups with
CFG-drop masks, explicit noise and gradient accumulation; AdamW changes weights.
An initial test-only component accessor mismatch was corrected and its two
cases pass. This is not a GPU OOM-recovery or precision claim. Common dataset
assembly/cache and optimizer/RNG/scaler resume are still required before normal
training capabilities become available.


### Common assembly and MNT ownership

The shared batch loop now calls native paired record preparation instead of the
single-image generic encode for Ming. Targets/references are independently
VAE-encoded once per batch; native processor input and weights remain batch-owned.
The MNT conditioning branch calls a fresh encoder graph for every iteration when
any conditioner group is trainable. A completely frozen conditioner is encoded
once and reused within that batch; changing trainability clears this reuse.
Native references and weights stay aligned through shared microbatch slicing.

This staged path currently forces on-the-fly pixel/text encoding and derives
prompt resolution from the resolved base resolutions (central defaults when
absent). Persistent native caches remain required; this temporary override must
be removed when their encoding modes are implemented. VAE encoding moves the
VAE onto the trainer device and returns it to CPU even on failure. Frozen
conditioners return to CPU after encoding; trainable ones remain resident through
backward. This is a whole-component baseline, not a verified official-model
memory budget or the final low-memory/offload policy.

47 related native data/shared-step CPU tests pass, including both profiles'
record assembly and two consecutive MNT-style backward/AdamW updates without
reusing a freed encoder graph. The frozen reuse and trainability-change boundary
also passes. These tests drive native preparation and actual shared backward;
they do not claim a complete normal training request, optimizer resume or GPU
acceptance. Normal startup remains staged until persistent caches, checkpoint
resume, sampling and execution-policy wiring are complete.


### Persistent native cache foundation

The native CPU tensor cache stores independent-frame targets and sparse ragged
reference latents, or both frozen conditioning streams, in safetensors. Atomic
replacement and cancellation cleanup preserve an existing published entry.
Reads require exact identity, roles, shapes and dtypes; payload checksums and
finite/layout checks turn corruption into a miss. Conditioning reads/writes
refuse a trainable conditioner so persistent reuse cannot cut its gradient.

Identity builders include paired pixel identities, actual VAE/conditioner weight
content, configs/tokenizer/processor identity, dtype and execution policy;
conditioning additionally includes ordered captions, resolution and topology.
CPU module fingerprints are taken before residency changes and never move a
model implicitly. Callers must supply their encoder attention/tiling/runtime
policy in the execution descriptor. Large weight fingerprints are startup work,
not per-iteration probes.

15 dedicated CPU tests pass for FP32/BF16 roundtrip, sparse references, dual
streams, corruption, incompatible requests, model/content/policy invalidation,
trainability refusal and cancellation preserving an existing entry. This is
the persistence foundation; common encoding-mode selection and producer/consumer
connections are still pending. The temporary on-the-fly override remains until
that integration is complete. No GPU/API or official model weights were used.


### Native cache producers and consumers

Record preparation now accepts an explicit frozen CPU cache-context snapshot.
It binds actual producer weights/config/precision, paired pixel identities and
execution policy to the native keys. Latent hits restore targets and sparse
references without invoking or staging the VAE. Frozen-conditioning hits restore
both native streams without invoking or staging the conditioner. Misses execute
the encoder and atomically publish its CPU result. Caption edits reuse the VAE
entry while selecting new conditioning; trainable conditioning always gets a new
live graph. Parameter-version and dtype drift refuse a stale context.

The shared native preparation/MNT seams consume this context when attached to the
trainer. A fresh model/context with identical weights also reuses the published
entries. CPU integration verifies identical cached/uncached losses, DiT backward,
trainable encoder gradients and no encoder `.to` calls on a complete hit. The
initial affected run passed 29 cases and exposed an in-memory object-reuse defect;
that was corrected with two targeted passes. Residency refinements pass eight
selected cases. Thirteen conditioning/cache/shared-gradient checks additionally
pass after canonical output dtype and transfer-cleanup changes.

Native CUDA query-connector autocast can produce a different dtype from the
direct stream when model weights are FP32. The wrapper now returns both streams
in the module dtype; casts preserve values and gradients. A CPU mixed-output
oracle verifies this boundary without claiming GPU numerical acceptance.

Startup snapshot capture, encoding-mode selection, eager prefill and swap-buffer
policy still need connection before the temporary on-the-fly override can be
removed. This producer/consumer integration does not enable normal training or
prove exact optimizer resume. Persistent cache contexts are never inferred from
file paths or nominal model names alone.

### Cache startup ownership

Native CPU loading now validates encoding-mode names before reading weights and
captures cache producer identity only for requested persistent modes. Frozen
conditioning and VAE snapshots are independent; trainable conditioning never
creates a persistent snapshot. No weight hashing occurs for the default swap
modes. The shared training startup attaches the snapshot under the run's
`ming_cache/` directory and rejects late cache selection, producer precision,
parameter-version, tiling/slicing or target-device changes.

37 affected CPU cache/conditioning tests pass, including a native package reload
through the common trainer loader. Changed backend files compile and real imports
pass with CUDA initialization forbidden. This connects startup identity to the
existing lazy producer/consumer path; eager prefill, swap buffers and CPU prefetch
remain pending. The temporary common on-the-fly dispatch and staged training
refusal remain until those mode contracts are implemented. No live API, official
weights or GPU resources were used.

### Conditioner rotary precision

The language, native vision and HF Qwen2 connector rotary frequencies remain
FP32 across conditioner `.to()` precision/residency changes. The wrapper retains
the unrounded frequencies, including dynamic RoPE's original frequencies, rather
than casting an already rounded BF16/FP16 tensor back to FP32. Low-memory loading
reconstructs all three kinds on the actual parameter device, including buffers
left on meta; connector reconstruction is now covered alongside native buffers.

Nine conditioner/package tests pass after the precision boundary change. Three
focused precision/meta tests pass after adding parameter-device reconstruction,
including exact trigonometric outputs at long position indices. Compile and real
imports pass with CUDA initialization forbidden. GPU numerical acceptance,
advanced attention backends and full common training remain pending.

### Native training encoder phases

Shared native batch preparation now carries a caller-owned residency policy.
For swap and persistent-cache misses, the policy stages the DiT on CPU during
VAE/conditioner encoding, releases the VAE or frozen conditioner, then restores
the DiT to the trainer device. On-the-fly mode leaves the DiT in place. Trainable
conditioning remains resident for its live backward graph. CPU execution does
not transfer the DiT. Phase overlap refuses; failure cleanup restores the DiT
and clears the phase state. Complete cache hits invoke no encoder phase.
The generic loop no longer duplicates VAE offloading for native Ming batches.

53 affected residency/cache/shared-step CPU tests pass. Transfer ordering uses
symbolic devices with spy components; cache reuse and consecutive trainable
conditioning backward use tiny native CPU models. Compile and real imports are
CUDA guarded. These checks do not prove physical GPU residency, official-weight
memory fit or optimizer-state movement. CPU prefetch still explicitly refuses at
the native worker seam, and eager prefill/swap-buffer scheduling remain pending.
The temporary generic on-the-fly dispatch and normal-training refusal remain.

### Replayable native paired transforms

Native shared pixel loading is now separate from encoder execution. Random crop
uses an isolated RNG keyed by resolved run seed, epoch, absolute batch position,
item occurrence, source content and bucket grid. Ahead-of-time calls therefore
produce the same paired pixels and cache keys as consumption without advancing
the common Python augmentation RNG. Targets and references share the crop; an
existing common `CropSpec` remains authoritative. Missing resolved run seed
refuses unplanned random crops. Dataset supervision identity is versioned to
include this paired-transform contract, so older staged resume identities cannot
silently continue under a different crop algorithm.

70 affected dataset/shared-step/cache CPU tests pass; compile and guarded real
import pass. This is the pixel preparation seam for prefill/prefetch, not their
completed scheduling. Epoch prefill, bounded buffers and CPU worker lifecycle
remain pending; normal Ming training remains unavailable. No GPU/API was used.

### Epoch native persistent-cache prefill

The common trainer now invokes native prefill after the epoch's final batch plan
and resume suffix are resolved, before consuming its first batch. Prefill uses
that epoch and absolute batch offset through the replayable pixel-loading seam.
Only requested persistent producers run: latent-only never runs the conditioner,
and conditioner-only never runs the VAE. Dual-stream conditioning resolution is
shared with normal/MNT consumption. Prefill holds only one pixel/prepared batch
at a time; entries are atomically published and a stop before a later batch does
not publish that batch. Progress uses the existing latent/text cache phases.

57 affected native cache/residency/shared-step CPU tests pass, including complete
prefill followed by consumption with both encoder execution and transfers
forbidden. Compile and real imports pass with CUDA initialization forbidden.
GPU memory/performance and the full common train loop remain unverified. Grouped
swap-buffer scheduling, CPU prefetch and force-recache propagation still need
connection before removing the temporary generic dispatch and normal-training
refusal. Resolution/caption changes continue selecting content-bound cache keys.

### Dataset-owned native cache lifecycle foundation

Current Qwen/common cache lifecycle belongs to the dataset and VAE, not the run.
The old `force_recache` run option is removed; rebuilding deletes a dataset cache
namespace through the existing endpoint. Ming must follow that contract.

A native dataset-context factory now creates run-neutral architecture/VAE cache
paths from the dataset's stable unique ID and actual VAE weight/config/precision/
execution identity. Latent safetensors and their metadata live under `vae-*/`,
while conditioning safetensors live in the VAE-independent `text_embeddings/`
folder. The factory performs no file writes and refuses escaping IDs/paths.
Native latent writes add the catalog stamp. The common namespace lister counts
both `.pt` and `.safetensors` entries; OpenAPI reflects the extended format.
Existing frontend numeric listing/delete types remain compatible by static review.

87 affected native-cache/dataset-API CPU tests pass, including run-neutral reuse,
listing bytes and a VAE-namespace removal preserving conditioning. Compile, real
imports and OpenAPI parse pass with CUDA initialization forbidden. Normal Ming
batch producers still use the staged run-folder context: connecting per-item
latent ownership while retaining correct mixed-dataset conditioning remains
pending. The new context factory alone does not complete that migration. GPU/API
were not used. The earlier force-recache TODO is superseded by dataset lifecycle
integration, not by restoring the removed run option.

### Shared batch dataset-cache ownership

Native cache startup now attaches the shared dataset-cache base, without creating
an output-directory cache. Persistent shared latent preparation selects each
item's stable dataset ID and writes one independent target/reference entry under
that dataset's VAE namespace. Rebatching and reordering can reuse those entries.
The conditioner still processes the original full batch, preserving native token
padding and both streams. Mixed-dataset conditioning is stored under the stable
lowest dataset ID in the batch, in its VAE-independent text-embedding directory;
its content key still binds every ordered pixel identity and caption.
Epoch prefill and normal native preparation share this ownership path.

The affected run passed 63 CPU cases and exposed a test fixture assumption:
Layer's three targets plus one reference require four independent VAE calls,
not two. After correcting that expectation, the failed case passes alone.
It verifies mixed-dataset prefill, cached/uncached latent and conditioning
agreement, per-item reuse across rebatching, and deletion rebuilding only the
selected dataset while retaining the batch conditioner entry. Compile and
CUDA-guarded real imports pass. Startup/package tests now assert dataset ownership
rather than the obsolete run-folder path. Explicit isolated test contexts remain
available for native cache primitives.

Per-item cold misses currently use independent encoder phases. Grouped transfer
optimization, bounded swap buffers and CPU prefetch remain pending before normal
training activation. These CPU checks do not measure physical GPU transfer cost
or prove the complete shared training/resume lifecycle. No GPU/live API was used.
