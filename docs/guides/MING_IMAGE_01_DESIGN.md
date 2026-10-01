# Ming-Image 0.1 integration contract

Status: implementation in progress. Native Design/Layer generation, DiT/caption LoRA and full fine-tuning are connected to public API configuration, shared runner and UI capability wiring. CPU tiny-package generation, training, U-Net restoration, batch-boundary resume and previews are verified; advanced Qwen feature parity and official-weight GPU acceptance remain open. Established 2026-10-01 against SushiUI
`2b28c3b4`. This contract preserves the requested Qwen-Image 2.1 feature
scope, including residual U-Net conversion. It is not a claim of measured
Ming support. [Completed source audit](../audits/MING_IMAGE_01_SOURCE_AUDIT.md)
records exact upstream source/model revisions and licensing constraints.

## Implementation evidence by milestone

The entries below record successive implementation stages; later stages supersede earlier staged-boundary notes. The current public training methods are native `full_finetune` and DiT/caption `lora`. Conditioner adapters, ReLoRA and advanced parity are still being implemented.

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

### Grouped native VAE transfer boundary

Dataset-owned cold latent entries now share one lazily opened VAE residency phase
per native batch. The first miss stages the encoder/DiT; remaining misses reuse
that phase, and its exit restores the DiT once. Complete hits never open the
phase. Per-item keys and dataset ownership remain independent. Every item cache
context is constructed before encoding, so invalid later dataset paths refuse
before earlier items can be encoded. A closed/reentered grouped phase refuses,
and exception cleanup uses the same owned transfer policy.

62 affected residency/cache CPU tests pass, including symbolic transfer ordering
for zero/one/multiple misses, cleanup, and a two-dataset tiny-native batch.
Compile and real imports pass with CUDA initialization forbidden. These tests
verify transfer calls, not physical GPU throughput. Cross-batch swap-buffer
scheduling and the CPU-prefetch worker remain pending, as does normal-training
activation and the complete checkpoint/resume acceptance. No GPU/API was used.

### Bounded cross-batch swap windows

The resolved epoch suffix now owns a native `MingBatchBuffer`. Latent and frozen
conditioning windows use their independently selected swap intervals. A refill
holds one encoder residency phase across the window, then retains only CPU
latents/references/prepared inputs or detached CPU conditioning. Consumed entries
are removed; the window size bounds the retained batch count. Trainable
conditioning is never prefetched into these frozen buffers and still constructs
a new live graph for each MNT. Changed batch captions/grids/CropSpec/ownership
invalidate entries. Source/sidecar content is rechecked before consuming a
prefilled entry, so a window cannot hide post-preflight file edits.

Common epoch setup and batch assembly use the buffer; epoch exit and the train
try/finally clear it, including stop/exception paths. Five new CPU buffer tests
pass after fixing the initial identity-descriptor format error; 74 affected
existing dataset/cache cases had already passed. An additional targeted test
passes for buffered native DiT backward plus AdamW update. Changed backend files
compile and real-import with CUDA initialization forbidden. These tests verify
bounded counts and transfer calls, not physical GPU memory/performance or full
common-trainer resume. CPU prefetch remains pending, and normal-training
capability/temporary generic dispatch remain staged until mode integration and
complete shared-loop acceptance are finished. No GPU/live API was used.

### Native frozen CPU prefetch worker

Native batch buffering now supports `cpu_prefetch` through one exclusive CPU
conditioner worker and a bounded output queue. The queue uses the existing
central `text_encoding_prefetch_depth`; shared processor preparation is locked
against latent preparation, while CPU conditioner forward can overlap other
work. The worker owns only frozen eval-mode CPU parameters/buffers, and checks
parameter versions and precision before encoding and consumption. Both streams
are detached CPU tensors. A changed batch identity restarts lookahead at the
current suffix; trainable/changed producer ownership refuses queued results.
Forward guards prevent concurrent main-thread conditioner execution and provide
cancellation checkpoints at native layer boundaries. Worker failures reach the
consumer. Stop joins the thread, removes guards and clears queued payloads.
Common stop/exception handlers close the native buffer before emergency saving,
in addition to epoch/try-finally cleanup.

14 affected CPU worker/buffer cases pass, including exact FP32/BF16 agreement with
synchronous native conditioning, full-queue backpressure/stop, exception delivery,
caption restart and live-conditioning refusal. Four targeted producer-change
cases pass after adding eval-mode validation. Changed backend files compile and
real-import with CUDA initialization forbidden. No official weights, live API or
GPU were used. CPU-only worker execution does not prove physical GPU overlap,
large-model stop latency or the complete common train loop. The generic temporary
encoding-mode override and normal-training refusal remain until common initial
residency/setup paths are routed to these native mode contracts. Sampling/export
must respect exclusive worker ownership when those pending routes are enabled.

### Common encoding-mode routing

The temporary Ming `onthefly_gpu` override is removed. Final selected modes are
resolved after the existing live-TE compatibility adjustment, then attached to
native cache/residency ownership. Native encoders initially remain CPU-resident;
invalid modes and trainable CPU-prefetch ownership refuse before transfer.
Ming skips the generic disk-cache setup, per-caption cache pass, path-keyed swap
buffers and generic CPU worker. Its resolved epoch plan owns persistent prefill,
native swap windows and CPU prefetch. CropSpec-aware native keys preserve cache
and swap modes through crop augmentation rather than forcing uncached encoding.
The generic on-the-fly epoch staging no longer eagerly moves the entire native
conditioner to GPU. Common main-module lookup and CPU/GPU transfer now recognize
the Ming DiT; common GPU text staging preserves CPU-prefetch ownership.

42 affected residency/buffer/prefetch CPU tests pass. Six focused initial-placement
cases additionally pass after ordering mode validation before transfers. Changed
backend files compile and real-import with CUDA initialization forbidden. Symbolic
transfer tests verify dispatch and CPU-only checks verify native encoder behavior;
the complete `train()`/normal runner and optimizer/RNG/scaler resume acceptance
are still pending, so normal training capability remains unavailable. No GPU/live
API was used. Full Qwen parity, sampling/export and advanced surfaces remain
active scope; removal of the temporary mode override alone is not completion.


### Real full-parameter CPU lifecycle milestone

The actual FullParameterTrainer constructor and common train() now run with a
six-component tiny native CPU artifact carrying a saved U-Net topology. The
selected bucket strategy is retained for native paired transforms; previously
this attribute was absent at the real entry point. Checkpoint discovery now
includes Ming manifest directories, enabling latest resume and shared retention.
Unrelated or malformed manifests are excluded without inspecting weight contents.

Five real training configurations verify forward/backward/AdamW updates and
periodic bundle/state/optimizer saving: on-the-fly, swap, persistent cache,
CPU conditioning prefetch, and trainable projections (whose updates are checked).
A latest-resume CPU case reloads exact trained DiT weights and U-Net topology,
restores optimizer history and reaches the second update (all AdamW steps = 2).
The reusable native-package fixture also retains its existing loader/export
round-trip test. Tests use temporary datasets/cache/run outputs, no official
weights, GPU or live API. The staged public capability refusal is overridden
only inside the test, and remains unchanged in production.

This proves a small FP32 Design full-parameter lifecycle, not all acceptance
conditions: production runner/API wiring, Layer profile whole-loop supervision,
mid-epoch/MNT/gradient-accumulation RNG and scaler parity, sampling/worker/export
coordination, other training methods and all remaining Qwen feature parity are
still active work. Normal training capability remains staged pending those gates.

Milestone verification: 11 affected CPU lifecycle/discovery/interrupted-save and
native-package round-trip cases pass. Changed Python files compile and the real
FullParameterTrainer imports with CUDA initialization forbidden.


### Real Layer common-loop supervision milestone

The reusable tiny artifact can now represent the Layer profile and includes
actual native image special tokens plus bounded CPU vision processing. Two
real FullParameterTrainer Layer runs exercise an ordered three-frame RGBA
sidecar with auxiliary-canvas weight zero, layer weights 1 and 0.5, and a canvas
reference. Each native common-loop loss receives [1,16,3,8,8] clean latents and
an independent [16,1,8,8] reference. MNT=2 and accumulation=2 perform two
backward passes and exactly one AdamW update; live vision/projection training
updates vision weights and regenerates conditioning without replay errors.
Periodic saved bundles reload the Layer profile and identical U-Net topology.

Nine affected real-trainer and package round-trip CPU cases pass; changed tests
compile and the real trainer imports with CUDA initialization forbidden. No GPU,
API or official weights were used. This closes the tiny FP32 Layer whole-loop
supervision gap, while exact mid-epoch RNG/augmentation resume, production runner,
sampling/worker coordination and remaining full Qwen feature parity remain open.


### Native shared-trainer previews and publication

Ming's architecture sample handler now uses the same native generator as ordinary
inference. Design supports text generation and single-image edit previews; Layer
requires an explicit reference and preserves every ordered RGBA output. A private
request-owned result group accompanies the legacy primary PIL image through the
shared trainer, keeping TensorBoard and convergence call sites compatible.
The common PNG saver stages all frames, publishes them with rollback, records
profile/frame roles/counts and saved U-Net metadata, and lists every frame in an
on-demand response. The auxiliary canvas is labeled rather than represented as
an assumed composite. Optional unsupported sampler controls remain excluded by
the existing sample capability allowlist.

Preview generation clones the scheduler, restores each submodule's previous
train/eval flag and component device on success/error/cancellation, and uses
caller-owned sequential residency. Native CPU-prefetch workers stop and join
before preview or full checkpoint export; queued conditioning is recomputed
from the next batch cursor after ownership returns. Worker errors propagate,
not silently disappear during suspension. Checkpoint save uses the same gate.

CPU evidence covers native Design text/edit and Layer reference previews, exact
mode/weight/scheduler preservation, progress, cancellation cleanup, worker pause
and next-cursor replay, publication rollback preserving an older frame set, and
real shared-loop step-zero/periodic PNG saving while CPU prefetch is selected.
A real Layer MNT/accumulation run writes all six RGBA images across step zero
and the periodic preview, with role metadata and round-trippable U-Net specs.
The affected suite first passed 23 cases plus one filename-fixture failure;
six focused cases pass after correction/error-propagation work. Two affected
common-loop cases pass after export ownership integration, and the Layer
reference/metadata fixture subsequently passes. Changed backend modules compile
and real-import with CUDA initialization forbidden. No official weights, GPU or
live API were used.

Normal training remains staged until runner/resume gates and remaining full
Qwen parity are completed. Per-preview Layer count currently uses the central
generation default or the trainer config value; its dedicated training UI/API
option and profile-specific sample-default wiring remain pending. Production
memory/offload behavior and exact mid-epoch RNG/scaler replay also remain open.


### Profile-aware preview defaults and public layer-count option

`ming_image_01_sample_layers` is a strict positive-integer per-run training
parameter. Its default comes from param_defaults, and the field is wired through
OpenAPI, the request model, generated train YAML, parameter restore vocabulary,
TypeScript request types, preset/pass-through keys and the Layer sample control.
The native preview consumer reads this public key. Design still emits one image;
Layer adds its separately labeled auxiliary canvas to the requested RGBA layers.
This per-run control is not part of live sample-settings patches yet.

Model listing reads Ming's native config without loading weights and exposes the
Design/Layer profile. The training frontend uses that profile for control
visibility and late-arriving default selection. Central preview overlays now
resolve Design to 12 steps/CFG 1 and Layer to 12 steps/CFG 2. Both generated YAML
and hand-written runner configs resolve the profile; explicit sample fields,
including CFG zero, remain unchanged. Malformed native packages are excluded
from the model list. Unknown/missing profile paths use the architecture overlay
until the native loader enforces the artifact contract.

Nine new offline parameter/profile/model-list cases and the existing related
contracts pass (82 cases in the combined check). The remaining common progress
contract is fixed through the architecture-neutral ops dispatch; it and six
native sample/worker/publication cases pass in a guarded seven-case check.
Changed Python files compile and real-import; frontend review is static only.
Owner build/type-check and production GPU validation remain deferred.

One follow-up verification initially imported the generic sample test before
its Ming CUDA guard, initializing a CUDA context before the CPU tests ran. That
process has exited. `backend/tests/ming_image_01_cpu_verify.py` now installs CUDA
refusals before pytest collection, regardless of selected test ordering, and
asserts no context was initialized. Use the repo venv with this entry point for
Ming CPU checks; explicit test targets avoid repeating the full suite. No HTTP
API, servers or official model weights were used in this milestone.


### Mutable training block swap and phase ownership

Ming's main `transformer.layers` now use the shared mutable
`LayerOffloadConductor`, after adapter ownership is established. Dense and
U-Net sequences retain native layer hooks and require gradient checkpointing
for every main layer. H2D-only mode is refused because trainable weights need
CPU writeback. The existing training block-count, pinned-memory and ring-size
controls use this path; normal public training methods remain staged.

A shared Ming component-move helper flushes completed updates and moves only
unmanaged tensors. It preserves managed parameter identities and CPU master
storage views across encoder phases, BaseTrainer residency, trainer previews
and native bundle export. Execution-device snapshots accept this deliberate
mixed residency without accepting arbitrary mixed-device components.

CPU evidence uses real native layers, checkpoint recomputation, parameter
update hooks, mutable conductor lifecycle hooks and the real transfer engine.
Only CUDA streams/transfers are substituted with CPU slots. Four dense/U-Net
Design/Layer backward cases verify updated masters, and both native trainer
preview/export cases verify exact weight/topology reload and preserved master
storage. Six block-offload cases and related preview/engine checks passed;
the two export cases passed separately. A focused refusal case covers invalid
checkpointing and H2D-only configuration. Imports and compilation are guarded
before any trainer/API import. No HTTP calls, servers, official weights or
physical GPU transfers were used in this milestone.

Frontend static review confirms training controls consume capability refusals
and retain existing block-count/pinned-memory/ring-size API fields. Build and
type checking remain owner-run. CUDA DMA, production memory/performance,
mid-epoch exact resume, adapter families and remaining Qwen feature parity
still require later milestones. Generation block swap is still unavailable,
as in the current Qwen 2.1 generation capability contract.


### Stochastic and pending-gradient training continuation

Ming optimizer sidecars now carry a versioned `_sushi_ming_runtime` payload:
CPU and active execution-device CUDA RNG states, Python/NumPy RNG states,
effective run seed, named pending gradients, accumulation interval and
GradScaler state. The payload is removed before generic optimizer state
conversion/device placement, so unrelated runtime tensors are not copied to
CUDA with the optimizer moments. Old Ming sidecars without this key retain the
existing common resume behavior and cannot promise identical stochastic
continuation.

Restoration occurs after epoch suffix reconstruction, native cache preparation
and worker setup, immediately before the first resumed batch. It preserves the
common saved pre-shuffle Python state while reconstructing data order, then
restores the live post-batch random streams. Pending gradients must retain
parameter ownership, shape, dtype, accumulation interval and scaler mode.
Execution device type changes are refused for this exact continuation path.
An optimizer with no moments before its first update is restored only after
its class and full parameter-name layout match. Resume must not apply runtime
state if optimizer restoration failed. Automatically drawn run seeds reuse the
saved effective seed.

Three actual tiny native FullParameterTrainer comparisons cover Design with a
frozen CPU-prefetch conditioner, Design with trainable projections, and Layer
with references plus weighted auxiliary/RGBA frames. Each runs eight backwards
with MNT=2 and accumulation=4, saves after the first batch (two pending
backwards, before the first AdamW update), and resumes the six remaining
backwards. Clean data, native noise/timesteps, final DiT/conditioner weights,
AdamW moments/steps and scheduler position match uninterrupted CPU training
exactly. The existing U-Net topology is present throughout.

Six additional CPU cases verify real CPU GradScaler state and three random
streams, saved gradients, and refusal of incompatible ownership/shape/device/
accumulation/scaler contracts. Nine cases passed in targeted runs; backend
compilation and real imports remain mandatory. No live API, server, official
weights or physical CUDA execution was used. Exact continuation at a completed
batch seam is proved on CPU; interruption inside an MNT window, CUDA kernel
reproducibility, production mixed precision and optimizer-family changes are
not covered by this evidence. Public training methods remain staged until
runner and remaining capability wiring are reviewed.


### Public native full-training entry point

Ming full fine-tuning is now available in the training capability response;
LoRA/ReLoRA/ControlNet remain refused until their own paths are implemented.
The runner no longer refuses Ming unconditionally. Before dataset scanning it
resolves explicit conditioner groups, normalizes the effective TE scope, rejects
empty training ownership, frozen-worker/live-conditioner conflicts, external
vision encoders and non-native prediction laws. The API write boundary applies
the same native options contract. Native sidecar references are accepted by
the common reference preflight and capability table.

Generated Ming full-training YAML keeps the central `noise_process=auto`
default, resolving to native flow/velocity rather than inheriting the old
full-training `add_noise` fallback. Renamed native Ming packages are recognized
by the runner's BF16-native config detection. Existing native dataset annotation
excludes Layer assets already owned by primary sidecars and repeats annotation
after epoch reload; no separate frame is accidentally trained as a primary.

The frontend uses effective Ming group ownership for the nonempty-scope check
and shared TE rate controls. Explicit groups work when the inheritance checkbox
is off, and an explicit empty list really freezes the conditioner. CPU prefetch
is disabled for live Ming groups and submit validation explains incompatible
restored settings without silently replacing them. Native dataset/reference
instructions accompany the group controls. API types/defaults/state/request
paths were statically reviewed; owner build/type checking remains deferred.

Two CPU cases follow actual request types through generated YAML and runner
preflight into real FullParameterTrainer initialization and a saved native
training step, without overriding the public training capability table. They
cover Design DiT-only and Layer projection-only with native references. Forty-four unique targeted cases passed across staged checks, including
options/sample-default/capability contracts and method refusals using the
authoritative capability reason. Changed Python files compile and real-import
under the CUDA guard, and the updated OpenAPI parses successfully. No running API, server, official weights or physical GPU
was used. Full runner subprocess/DB execution and production CUDA validation
remain unverified; advanced Qwen feature parity is still active work.


### Native DiT attention conduit

Generation `attention_type` and the trainer's resolved backend now install the
common conduit in the native noise/context refiners and main DiT blocks. Native
Q/K normalization, FP32 complex RoPE and output projections remain intact.
Ragged key-only padding packs valid K/V rows while retaining every query row;
query-dependent masks and all-masked samples retain the dense masked path.
Shared dtype/head-dimension/kernel/trainability guards select an available
backend. SLA and legacy diffusers dispatch remain refused. The conditioner
continues to use its native eager implementation. No parameter keys change.

Thirty-nine affected CPU cases pass: both native padding profiles, dense and
nested U-Net output/input/parameter-gradient comparisons, packed backend
contracts, inference-only backend training fallback, generation selection,
trainer resolution, existing pipeline operations and mutable block offload.
Packed Flash/Sage contract tests substitute CPU kernels; they do not execute
those GPU implementations. Python compilation, CUDA-guarded real imports and
OpenAPI parsing are checked at this milestone. Frontend request/settings/queue
paths were reviewed statically; owner build/type checking remains pending.
No running API, server, official weights or physical GPU was used. Advanced
Qwen feature parity remains in progress.


### Native adapter target foundation

`lora.py` enumerates the main/noise-refiner/context-refiner attention Q/K/V/output
slots and optional native caption projection. It reuses the existing reversible
canonical adapter-key codec and architecture-neutral branch builder. Wrapped
slots remain discoverable for generation unloading and layering. Two CPU cases
cover both native profiles, exact target ownership/unique keys, factor-group
round trips, explicit additive deltas and backward after wrapping. Changed files
compile and real-import with CUDA initialization forbidden. This is a foundation
only: Ming adapter capability tables and public LoRA methods remain closed until
trainer, generation session, export and resume paths are connected and verified.
Conditioner adapter groups and the remaining Qwen parity paths remain open.


### Request-scoped native DiT LoRA generation

Ming's generation capability now accepts ordinary LoRA through the shared
AdapterSession. Requests support multiple additive branches, strengths,
apply_to_unet and the common 0–1000 step range. Checkpoints must explicitly
identify model_type=ming_image_01 and the loaded ming_image_01_profile. Native
16-channel latent compatibility and residual topology compatibility are checked
before installation. All selected topology declarations must agree; an existing
U-Net base cannot be silently replaced by a dense or different adapter topology.
Temporary adapter topology and wrapped modules are restored on success,
cancellation and exceptions. Reference/edit/Layer paths use the same session.

Thirty-one unique affected CPU cases pass across staged checks: both profiles'
layering/strength/step contracts, profile refusal, real common-manager generation
and reuse, temporary U-Net topology restoration/conflict refusal, incompatible
latent rejection and separate generation/training capability axes, plus the
existing native pipeline suite. The final seven-case adapter batch passes after
the latent gate addition. Backend compilation, guarded real imports and OpenAPI
parsing pass. Common frontend LoRASelector capability lookup, request fields and
step controls were statically reviewed; owner build/type checking remains pending.
No official weights, GPU, running API or server were used. Conditioner adapters
and public LoRA training/save/resume remain to be connected; this generation
capability does not enable training methods.


### Native DiT/caption LoRA trainer lifecycle foundation

The arch handler now declares MingImage01LoRAAdapter. It freezes the native
base/conditioner/VAE and wraps selected main/refiner attention slots, with an
optional caption-projection-only scope via train_adapter. The common trainer
owns optimizer groups, checkpoint writes and weight restoration; caption groups
read adapter_lr from config, preserve explicit zero, and otherwise use DiT LR.
Saved files carry architecture/profile/rank/alpha and residual U-Net metadata.
Resume checks these identities and exact target tensor scope before mutating
branches. Adapter-owned topology resolution is deferred until resume weights
are validated rather than treating an adapter file as a base checkpoint.

Four CPU cases use real LoRATrainer for one native Design/Layer training step in
attention-only and caption-only scopes. They prove finite parameter updates,
frozen base and conditioner, common safetensors export, exact branch weight
restoration in a new trainer, generation-session output equivalence and atomic
profile/scope mismatch refusal. The Layer tiny-package fixture explicitly
initializes learned padding before serialization: native torch.empty values
are checkpoint-owned and previously left this synthetic fixture undefined.
Official parameter construction/loading is unchanged. Twenty-one related cases
pass; the final four-case rerun additionally checks zero/explicit adapter rates.
Changed files compile and real-import with CUDA initialization forbidden.

These tests temporarily open the internal training-method table; the public
training capability remains closed. API/runner/UI method and projection-scope
wiring, optimizer/RNG/scaler continuation through an actual resumed LoRA loop,
and conditioner adapter ownership remain next work. No live API, server,
official weights or GPU execution was used. Full Qwen parity is still pending.


### Public DiT/caption LoRA and exact CPU continuation

Ordinary Ming LoRA is now enabled on both authoritative adapter capability axes
and the public training method surface. API preflight and the shared runner use
the same native flow/scope contract. Ming LoRA currently requires frozen native
conditioner groups; full fine-tuning retains its existing group behavior. Empty
DiT scope is accepted only when train_adapter=true explicitly selects caption
projection LoRA. Both API and runner refuse incompatible/empty scopes before
weights load. ReLoRA/ControlNet and conditioner adapters remain closed.

The common UI conditioning-adapter control now identifies Ming caption projection
and sends method-gated train_adapter/adapter_lr. Projection-only submission passes
the nonempty-scope check; restored text-encoder training settings produce a clear
Ming LoRA refusal. Existing full-training group controls and non-Ming behavior
remain separate. Types, defaults, restoration/submit paths and callback dependencies
were statically reviewed; owner build/type checking remains deferred.

Twenty-nine affected CPU cases pass after public wiring. Four follow actual public
request models through YAML/runner into real native LoRATrainer Design/Layer
attention/caption updates, safetensors save, weight restore and generation output
equivalence. Four more execute eight MNT backwards with GAS=4, save after two
backwards before the first AdamW update, then resume the remaining six. Data,
noise, timesteps, final native/LoRA/conditioner state, AdamW moments and scheduler
match uninterrupted execution exactly, including CPU prefetch and Layer weighted
reference targets. No internal method-capability override is used. Seven existing
shared scope checks also pass, covering empty and ordinary nonempty runs.
Changed Python files compile and CUDA-guarded API/runner/trainer imports and
OpenAPI parsing pass. No live API, server, official weights or physical GPU
execution was used. Exact CUDA/mixed-precision continuation and full runner
subprocess/DB execution are unverified. Remaining Qwen parity stays in progress.


### Full-training caption bridge ownership and rates

Ming full fine-tuning now consumes the same train_adapter/adapter_lr controls
as its LoRA path. Null train_adapter inherits train_unet; true trains the native
cap_embedder bridge even with the rest of DiT frozen, and false freezes that
bridge while other DiT parameters train. Full scope includes the bridge RMSNorm
and Linear; LoRA scope targets the Linear only. Conditioner groups remain
independent. Explicit adapter_lr separates caption parameters into one unique
optimizer group tagged component=adapter, preserving zero. With no adapter rate
override, existing DiT grouping/order and inherited rate remain unchanged.
Caption-only execution enables native gradient checkpointing when requested.

API/runner nonempty-scope checks and UI controls accept caption-only full training.
Method-gated request fields, preset/restored settings and effective conditioner
scope were reviewed statically; frontend build/type checking remains owner work.
Native full-bundle training metadata records caption scope/rate alongside groups.

Thirty-two affected CPU cases pass. Eight use actual public request types through
YAML/runner into real FullParameterTrainer for both Design and Layer: caption-only
updates, explicit caption freeze, independent rate, and zero-rate preservation.
They verify finite values, unique optimizer ownership, frozen/updated parameter
scopes and exact native bundle reload. Two added batch-seam resume cases prove
caption-only full training with a separate optimizer group, MNT=2, GAS=4 and CPU
prefetch continues exactly from saved pending gradients: data/noise/timesteps,
weights, AdamW state and scheduler match uninterrupted execution. Changed Python
files compile and guarded API/runner/trainer imports plus OpenAPI parsing pass.
No live API, server, official weights or GPU execution was used. The remaining
Qwen feature parity and production CUDA acceptance are still in progress.


### Native DiT runtime INT8 conversion

Ming now accepts unet_quantization=int8 through the common one-time in-place
converter, before LoRA installation and phase residency. Native attention/FFN
Linears are eligible; input/output, timestep, caption and modulation bridges
remain floating point. Shared crest/error selection may retain an e4m3 fallback.
This is a provisional memory-oriented selection policy, not a measured Ming
quality/speed claim. Shared partial-conversion, already-quantized, retry and
one-way persistence rules remain in force. Requesting null/none after conversion
cannot restore discarded weights and emits the common reload warning.

The converter wrapper now optionally preserves a caller's compute dtype; old
callers retain their BF16 behavior. Ming requests its native dtype, and its LoRA
branch builder uses a quantized base's compute dtype so native FP32/FP16 execution
does not silently acquire BF16 factors. Existing layer buffers remain separate
from parameter objects. The current-model census is refreshed after conversion
while native architectural counts and serialized source storage stay separate;
loaded parameter-object accounting explicitly excludes quantization payloads.

Six CPU runtime cases pass: actual Design/Layer RGBA generation, protected bridge
identity/frozen conditioner, audit/census consistency, null/none persistence,
INT8-before-LoRA and quantized-base restoration, unsupported-format refusal, and
both profiles' two-stage nested U-Net generation/topology preservation. Existing
slot/branch and common converter lifecycle checks pass in staged runs; a stale
OpenAPI architecture-list gate was fixed and all three documentation/capability
gates pass. Python compilation, guarded real imports and OpenAPI parsing pass.
Generation panel request fields and capability-derived INT8/GEMM choices were
reviewed statically; owner frontend build/type checking remains deferred.

No live API, server, official weights, GPU allocation or physical GPU kernel was
used. Native quantized bundle export/loading, conditioner quantization, ConvRot
and quantized-base training remain pending. GPU memory, throughput, W8A8 numerics
and official-model image quality are unmeasured. Full Qwen parity is still active.

### Native quantized DiT bundle restoration milestone

The native bundle loader now preflights weight-only INT8 and E4M3 Linear
headers before conditioner construction, builds the declared native transformer
on meta, installs format-specific Linear slots and assigns CPU shards one at a
time. Quantized codes retain their serialized dtype and per-row scales remain
FP32; floating parameters follow the requested compute dtype. Unknown markers,
orphan/missing scales and inconsistent key/shape sets are refused. Payload scales
must be finite and positive. Conditioner quantization remains unsupported.

Existing atomic six-component bundle saving can now round-trip a converted DiT.
Design and Layer CPU cases preserve every serialized tensor bit, two-stage U-Net
topology and native RGBA generation after loading sharded exports. Four malformed
header cases refuse before conditioner loading; two mixed INT8/E4M3 cases retain
codes/scales under FP32/BF16 compute selection; two invalid payload-scale cases
refuse zero and NaN values. The existing ordinary package
loader/export/manager regression also passes. These are tiny native CPU fixtures,
not official-weight GPU acceptance or performance measurements.

The public background quantized-export job still has no Ming layout and refuses
Ming; this stage establishes native package serialization/restoration only.
Public export wiring, conditioner quantization, ConvRot, quantized-base training
and the remaining Qwen parity work are still pending. No main API/GPU was used.

### Public native quantized export milestone

The existing GET/POST `/models/export-quantized` job now resolves native Ming
packages. Status/job documents distinguish `output_kind: native_package` from
single-file layouts. The suggested/editable destination is a new package
directory outside the loaded source; completion returns `manifest.json`.
Ming serializes all six native components instead of linking companions;
`link_siblings` is ignored and overwrite is refused. Package publication remains
atomic. Native progress reports six components and `quantization.json` preserves
the Linear inventory plus the available runtime audit (or its absence).

The worker uses the common exclusive model-lifecycle gate and model-load lock.
After waiting it revalidates architecture, source, transformer identity and all
native component identities, preventing queued exports from saving a replacement
model or conditioner. CPU offload is enclosed in residency restoration for both
success and serialization failure. This is not a GPU residency measurement.

Eight offline native job cases cover both profiles' real background completion
and exact reload, destination/overwrite refusal, architecture/transformer/
conditioner replacement, failure restoration and unavailable-audit recording.
The initial run with the shared exporter passes 33 cases plus 12 subtests;
follow-up checks pass six cases covering new refusals, audit absence, ordinary
bundle regression and the OpenAPI export architecture description. The modified
real-thread cases are checked after identity-gate completion. Compile/guarded
imports and OpenAPI parsing are checked at this milestone.

Frontend status/request types, loaded-model availability probe, package-path
controls, progress units, stale-model error handling and final manifest display
are reviewed statically. Build/type checks and interactive acceptance remain
owner-run. No live HTTP API, server, official weights or CUDA allocation is used.
Conditioner quantization, ConvRot, quantized-base training and the rest of the
Qwen parity contract remain pending; full implementation remains active.

### Native ConvRot DiT checkpoint milestone

Ming DiT package preflight now recognizes the existing shared
`int8_tensorwise`, `convrot: true`, `convrot_groupsize: 256` marker contract.
Only small one-dimensional U8 marker payloads are read during preflight;
weight payloads are still assigned shard-by-shard after geometry/key validation.
Recognized rows install the shared `ConvRotInt8Linear`, never ordinary INT8.
The runtime dependency is checked before conditioner weights are loaded.
Per-row FP32 scales accept `[out]` or `[out, 1]` for marked ConvRot rows;
only the declared ConvRot column scales are normalized to flat buffers.
Unknown markers, incompatible widths/weight types and malformed metadata fail
before conditioner loading. Native package export preserves the markers.

Twelve native CPU cases cover Design/Layer exact packed-state reload, nested
U-Net RGBA generation, the real Comfy-Kitchen eager dequantization/gradient
oracle, public export serialization/restoration and generation GEMM reporting,
six invalid-contract refusals, column-scale normalization, missing-runtime
refusal and both profiles' ConvRot-plus-LoRA generation. The LoRA fixture
explicitly declares its matching U-Net topology: undeclared dense adapters are
correctly refused by the existing checkpoint/adapter topology contract. The
packed ConvRot base and active topology survive adapter cleanup, and LoRA
sampling latents are finite. Ten existing plain INT8/E4M3 package cases pass.

These checks use the installed Comfy-Kitchen CPU eager implementation without
replacing its numerical operators. CUDA initialization is prohibited throughout;
no official weights, main GPU, HTTP API or server are used. Physical GPU kernels,
VRAM, throughput and official-model quality remain unmeasured. This milestone
loads/saves already-converted DiT artifacts; production offline ConvRot
conversion, conditioner quantization, quantized-base training and the remaining
Qwen feature parity are still in progress. Frontend continues to use the common
native-package export controls; owner build/type checks remain deferred.

### Offline native DiT conversion milestone

`subapps/ming_image_01_convert.py` converts a complete floating native Ming
package to `int8` (the shared INT8/E4M3 error-selection policy) or
`int8_convrot` (shared FP32 Hadamard rotation and rowwise INT8 quantization).
CPU is the default; CUDA is used only if explicitly requested. For example,
from a checkout with its repo virtualenv:

```powershell
venv/Scripts/python.exe subapps/ming_image_01_convert.py "SOURCE_PACKAGE" "NEW_PACKAGE" --format int8_convrot --device cpu
```

`--max-shard-mib` optionally changes the shared 4 GiB output threshold. The
source must be floating point and match its native DiT key/shape contract;
already-quantized packages are refused to prevent a second lossy quantization.
The destination must be new and outside the source. Transformer topology is
validated and propagated to every shard/index; unchanged configs and native
processor identity are checked before atomic package publication.

Companion MLLM/connector/projections/VAE/scheduler files are copied verbatim;
no companion model is constructed or loaded. DiT structure is constructed on
meta for selection/shape checks, and source tensors are mapped one file at a
time. The output writer buffers a shard plus the current tensor/quantization
working set; this is not a fixed RAM ceiling. Source files are never changed.
The existing Ming policy excludes input/output, time, caption and modulation
bridges. ConvRot additionally requires input width divisible by 256. The
provisional policy is not an official-model quality or speed claim.

`quantization.json` records selection, exclusions, per-layer plain INT8 audit
or ConvRot declarations and source config identity. Undefined zero-weight audit
metrics use JSON null plus named `undefined_metrics` rather than NaN/Infinity.
Cancellation or conversion failure removes temporary output without publishing
a partial package. Existing training metadata is copied when present.

Nine CPU cases pass: both profiles in both formats produce sharded native
packages restored through the production ModelLoader and finite U-Net RGBA
sampling; protected weights and all companion files retain exact contents.
The converter is tested with companion loaders forbidden. Other cases cover
interruption/failure cleanup, destination/already-quantized refusal, the real CLI
entry's CPU default and zero-weight audit/reload. Compile and guarded real
imports are checked at this milestone. No official weights, HTTP API, server or
CUDA allocation is used. Conditioner quantization, quantized-base training and
the remaining Qwen parity work stay active; owner UI/build acceptance is deferred.
