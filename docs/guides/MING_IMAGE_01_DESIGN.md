# Ming-Image 0.1 integration contract

Status: implementation in progress. Native Design/Layer generation, DiT/caption LoRA and full fine-tuning are connected to public API configuration, shared runner and UI capability wiring. CPU tiny-package generation, training, U-Net restoration, batch-boundary resume and previews are verified; advanced Qwen feature parity and official-weight GPU acceptance remain open. Established 2026-10-01 against SushiUI
`2b28c3b4`. This contract preserves the requested Qwen-Image 2.1 feature
scope, including residual U-Net conversion. It is not a claim of measured
Ming support. [Completed source audit](../audits/MING_IMAGE_01_SOURCE_AUDIT.md)
records exact upstream source/model revisions and licensing constraints.

## Implementation evidence by milestone

The entries below record successive implementation stages; later stages supersede earlier staged-boundary notes. The current public training methods are native `full_finetune` and DiT/caption `lora`. Advanced Qwen parity remains active; ReLoRA and conditioner LoRA retain Qwen's explicit refusals.

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
| LoRA load/unload, strengths and windows | Qwen backend `AdapterSession`; `core/models/qwen_image_21/lora.py` | Explicit target map for attention/refiners and caption projection matching Qwen's shipped target scope; multiple adapters, shape/base mismatch errors; cleanup and topology restoration in finally |
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


### Frozen quantized DiT LoRA training milestone

Ordinary native DiT/caption LoRA training now pins packed INT8, E4M3 fallback
and ConvRot bases to differentiable dequantization before adapter installation.
Inherited inference GEMM flags cannot select nondifferentiable kernels in the
training graph. ConvRot uses the shared eager dequantization path; packed
weights, scales and rotation markers remain frozen. Checkpoint metadata records
`ming_quantized_base_training` execution policy and per-format module counts.
This is descriptive base-forward metadata, not a base-content fingerprint or
a new resume identity gate. Legacy LoRA compatibility rules remain unchanged.

Four public request-model -> YAML -> runner configuration -> real CPU trainer
cases cover both profiles and both conversion formats. They verify finite branch
updates, byte-exact frozen parameters/buffers, saved metadata, weight reload,
and request-scoped generation adapter cleanup over restored nested U-Net topology.
Plain INT8 fixtures deliberately include an E4M3 fallback row. Four additional
shared-loop continuation cases save before the first AdamW update, retaining
pending gradients with MNT=2 and gradient accumulation=4. The six remaining
backwards reproduce input/noise/sigma, all model tensors, optimizer moments and
scheduler/update counters exactly after resume. These tests use actual native
small models and the production converter; CUDA initialization is forbidden.

Frontend static review confirms ordinary Ming LoRA uses the existing public
method/conditioner/adapter controls and request fields. No new API defaults or
training option is introduced. The optimized Qwen ConvRot cache/prefetch controls
remain Qwen-only until a Ming implementation is verified. Owner build/type checks,
interactive UI, CUDA mixed precision and official-weight numerical/performance
acceptance remain pending. Quantized full-parameter materialization, conditioner
quantization/adapters, advanced adapter families, partition/global, style and
other remaining Qwen parity obligations stay in scope.


### Native local-provider prompt enhancement milestone

Ming prompt rewriting uses the pinned upstream Design JSON system template and
README Layer guided template, with MIT provenance recorded separately. Both
profiles share SushiUI's loopback LM Studio/Ollama model-listing, load/release
transport and SQLite cache implementation, but use a separate native protocol
validator and cache. No official prompt-enhancer artifact is invented and no
remote service is automatically contacted. Design/edit yields canvas/layers JSON;
Layer yields the requested numbered front-to-back specification. Layer disables
Ollama's JSON response mode, while existing Qwen requests retain it.

Public defaults, request schema, transform/cache endpoints and frontend client
are connected. Width/height/layer count are explicit request fields. Design
validation checks native keys, normalized in-canvas bbox strings, hex colors
and the exact requested canvas ratio. Layer validation rejects changed counts,
missing/empty/reordered/duplicate layer numbers. One flattened Layer reference
is required; Design edit preserves ordered images. The cache includes prompt,
mode/profile, native-template digest, dimensions, count, image-content digests
and provider/inference settings. Force refresh bypasses lookup without changing
the key; API credentials are neither returned nor cached.

All four image generation panels expose preview/apply and an opt-in automatic
rewrite before generation enqueue. Layer mode is selected only for the Layer
profile; inpaint/outpaint controls are Design-only. Frontend settings are
separate from Qwen and derive defaults from the schema endpoints. Preview
cancellation or prompt/model/profile/dimension/count/reference changes suppress
stale responses. Stable file identities prevent fresh array literals from
canceling a valid preview on each render. Clearing the dedicated cache is wired.
Browser cancellation aborts delivery/application; the backend provider request
finishes or times out and then releases its model through the shared finally
path. It does not promise interruption of inference inside the local provider.

Offline tests verify valid native outputs, malformed schema/ratio/bbox/layer
refusals, profile/reference/loopback guards before provider contact, cache reuse
and refresh/content/geometry isolation, and actual shared request assembly and
load/unload with HTTP replaced by in-process response doubles. Existing Qwen
prompt transport tests pass in the same CUDA-forbidden process: 29 unique
affected checks pass across the main batch and the two added failure cases. Frontend contracts,
defaults, state/cancel/error handling and the four operation paths are reviewed
statically; owner build/type checks and real-provider quality remain pending.
No live API, main server, CUDA context or official weights were used. Remaining
Qwen parity obligations and final interactive/GPU acceptance remain active.


### Native Reference Guide generation milestone

Reference Guide entries now travel through the existing `controlnets` JSON/image
transport into Ming's generation backend. They are distinct from ControlNet
modules and the still-pending style-KV operator. Preflight requires a decoded
image for every requested guide, finite strength in [0,2] and ordered integer
start/end values in [0,1000]. Strength above one extrapolates by the same
positive-strength formula used by Qwen. Negative values are explicitly refused
instead of silently ignored. Missing native-guide strength/range use the central
`REFERENCE_GUIDE_DEFAULTS`; generic ControlNet request-field defaults and other
architectures' runtime interpretation are unchanged.

Guide preparation preserves RGBA pixels, resizes to the requested target grid,
encodes VAE posterior mode and applies Ming's native `(z-shift)*scale` exactly.
Each guide gets independent FP32 CPU-generated noise derived from run seed plus
its ordered index, without consuming the main sampler/global RNG. After each
scheduler update, blend against `(1-next_sigma)*clean + next_sigma*noise` using
the scheduler's actual next sigma, including its dynamic shift and final zero.
Progress windows are inclusive and use index/max(steps-1,1); multiple guides
apply in request order. Guide encoding runs inside the existing denoising VAE/
DiT residency phase, with cancellation checked between guide operations.

Design guides affect its generated target latent. Layer guides broadcast over
independent generated layer frames and exclude slot-zero auxiliary canvas from
the direct blend. The guide does not modify appended reference/caption/padding
streams. Later joint-attention steps may still change the auxiliary canvas;
exclusion from a blend is not a promise of whole-run numerical independence.
Inpaint/outpaint retain their final protected-pixel composite/paste contracts.

The common selector reads initial guide strength/range and flags from the
new `reference_guide` field of generation-defaults, preserving its previous
0.4/0-800 UI behavior without frontend literals. Ming shows its Reference Guide
panel, hides actual ControlNet creation and disables pending style-transfer
creation. Strength range matches the backend. The shared four panel serializers,
queue/loop inheritance and temporary-image transport remain connected and were
reviewed statically; owner build/type checks and interactive acceptance are pending.

CPU verification covers blend endpoints/intermediate sigma/windows/order in
FP32/FP16/BF16, exact native VAE normalization and alpha sensitivity, independent
noise, both native profile/frame layouts, all four manager operations with actual
common base64 decoding, protected pixels, cancellation/recovery and early errors.
Across the milestone, 55 unique affected CPU cases pass (31 guide and 24
pipeline cases), with changed/failed/new cases checked separately. The shifted
scheduler oracle uses three steps to retain a nonzero intermediate sigma; the
native two-step schedule already reaches zero after its first update.
Additional real native FP16/BF16 VAE+DiT generation passes. Design INT8 and Layer
ConvRot tests combine guides with nested U-Net and LoRA, verify the exact final
clean latent, unchanged packed base buffers and adapter/topology cleanup. A stale
older pipeline fixture that still refused now-supported INT8 was changed to
refuse unsupported FP8; runtime INT8 behavior was not narrowed to satisfy it.
No live HTTP API, GPU context or official weights were used. Official quality/
performance and the remaining Qwen parity surfaces remain active work.


### Quantized-base full fine-tuning materialization milestone

Ming's existing `full_finetune_dequantize_int8_base` option now admits native
DiT INT8, E4M3 fallback and declared ConvRot checkpoints. API preflight, generated
YAML, subprocess config and the full trainer share the existing central false
default. Direct YAML requires a boolean. The training UI exposes the option for
Ming full fine-tuning; other methods omit it, and non-full opt-ins are refused.
Qwen's existing opt-in boundary is retained. The separate automatic
`training_export_format=int8_convrot` option is still Qwen-only; Ming's native
quantized export remains the separate export/converter operation.

Before optimizer ownership is established, packed DiT linears are offloaded to
CPU and replaced one at a time with FP16/BF16/FP32 `nn.Linear` parameters.
INT8/E4M3 multiplication follows the shared dequantized forward semantics;
ConvRot uses the actual Comfy eager inverse rotation. Row scales, CPU residency
and geometry are validated first, and nonfinite resulting weights/biases are
refused. There is no recovery of pre-quantization precision. The conversion is
streamed, not transactional: a numerical failure after preflight can leave the
trainer-owned component partially materialized. Construction fails before
optimizer creation and training starts only after the packed-base backstop
passes. Conditioner quantization loading/materialization is separate pending
work and must not be inferred from this DiT option.

Full checkpoints contain floating DiT weights, the existing restored U-Net
metadata and `training.json.ming_full_initialization`: schema/version, floating
execution, counts by source format, materialization dtype and `lossy_source`.
A fresh full run from that floating checkpoint needs no opt-in and preserves
this origin record; resumed checkpoints preserve it too. Strict provenance
parsing refuses malformed records. This record describes origin, not a base
content identity or a promise of restored original floating precision.

Across this stage, 43 unique affected CPU cases pass: six format/dtype actual
forward-and-gradient oracles; four Design/Layer public request-to-train-to-save/
reload cases; refusal and provenance contracts; shared Qwen/other-architecture
API boundaries; existing adapter/runtime/caption resume regressions; and two
packed-origin full resume cases. The latter checkpoint two pending backwards
before the first AdamW update (MNT=2, accumulation=4) and compare six continued
backwards, conditioning, predictions, weights, moments and scheduler. The wide
CPU fixture clones DiT parameter storage on both sides to canonicalize mmap
alignment. Investigation found identical inputs/weights with 64-byte aligned
versus offset-eight-byte CPU storage produced approximately 1e-7 GEMM rounding
differences before the first resumed update. The exact fixture comparison is
not a claim of bitwise equivalence across arbitrary storage/hardware layouts.
Initial saved/restored weights and U-Net structures compare exactly without
that storage condition. Physical GPU transfer, official weights, quality and
performance remain unverified; no GPU context or live API was used.

Frontend static review covered central defaults, visibility, optional request
serialization and shared persisted request types. Owner build/type checks and
interactive acceptance remain pending without blocking continued work.
Conditioner quantization/adapters, optimized ConvRot training, partition/global/
style and remaining objective parity are still active work.


### Run-scoped ConvRot training inference export milestone

`training_export_format=int8_convrot` now supports Ming full fine-tuning as well
as Qwen; this supersedes the previous milestone's Qwen-only gate. The existing
normal-completion hook in `train_runner` calls the shared run export job, so
periodic checkpoints remain floating and stopped/failed runs do not automatically
export. The existing stopped/completed manual export endpoint and monitor allow
Ming and retain the live-process refusal. Shared defaults and YAML serialization
are reused; direct Ming config refuses unsupported formats and non-full exports.
The monitor obtains the run's architecture independently of sample images, so
a run with sample generation disabled still exposes its manual export action.
Qwen's legacy filename fallback is retained. UI status polling, replacement
confirmation, error handling and stopped/completed visibility remain connected.

Latest-checkpoint selection recognizes native `run_name_step_N` directories with
state and optimizer sidecars, then validates the complete native package and
schema-one full-training metadata/profile. Packed or incomplete candidates are
excluded. Sidecar presence is the selection criterion, not deserialization of
optimizer contents; normal training checkpoint writes own that payload contract.
An export invokes the CPU offline converter on this saved package, quantizes
eligible native DiT linears with the real shared ConvRot implementation and copies
all five companion components verbatim. It never constructs a live conditioner
or VAE for conversion. The output is `exports/int8_convrot`, a loadable native
six-component package preserving U-Net metadata and training origin metadata.
Its packed weights require the existing explicit materialization opt-in for a
new full fine-tune; the floating checkpoint remains the resume source.

Replacement converts/preflights a new package before moving the previous export
to a sibling backup, then publishes by rename and rolls back on publication
failure. Linked destinations escaping the run and pre-existing unfinished
backups are refused. Conversion/publication failures report failed status and
retain the previous export and original floating checkpoint. Cleanup failure
can leave a backup for explicit recovery; no claim of crash-proof transactional
filesystem behavior is made. Shared in-process per-run job exclusion/status and
idempotent reuse of an already completed same-source export are preserved.

Twenty unique affected CPU cases pass across this milestone: fourteen Ming
export tests, two existing Qwen export regressions and four shared API-gate
checks. Production request models/YAML/subprocess preflight and direct route
helper invocation with an in-memory database stub exercise Design and Layer
manual background jobs. Both load their actual exported packed package and
produce finite native U-Net RGBA inference; companions and source bytes remain
unchanged. Replacement failure injection covers conversion and final rename,
with rollback/temp cleanup assertions. Additional cases cover invalid/packed
checkpoint selection, config refusals and method/live-process gating. These are
offline tests, not HTTP requests or an actual worker completion/DB integration
run. Automatic wiring was reviewed statically at the existing completion hook.
Backend compile/real imports, spec parsing and frontend static review supplement
the CPU cases; owner build/type/UI checks and official GPU acceptance are pending.
Conditioner quantization/adapters, optimized ConvRot training, partition/global/
style and remaining objective parity remain active.


### Prepared MLLM/connector weight-only quantization milestone

Native loading now accepts packed INT8, E4M3 fallback and declared ConvRot Linear
weights inside `mllm` and `connector` as well as DiT. Their formats are inferred
from validated checkpoint headers/markers; `text_encoder_quantization` remains
an unsupported runtime re-quantization control, matching Qwen's prepared-artifact
precision boundary. Its capability explanation now names the prepared MLLM/
connector path. The separate `mlp` component's packed projections/queries, and
quantized VAE, remain explicitly refused; no support is inferred from this
milestone for arbitrary integer tensors, non-Linear weights or packed tied
embedding/output-head aliases.

The meta slot validator and CPU shard assignment are shared with native DiT.
Both conditioner components are constructed/preflighted on meta before loading
any companion weight payload. Supported markers are read under the existing
bounded CPU marker contract; keys, shapes, Linear ownership, per-row FP32 scales
and declarations are validated before assignment. Scale payloads must be finite
and positive. Packed I8/E4M3 codes keep their dtype, FP32 row scales remain FP32,
and ordinary floating weights/biases use the requested compute dtype. Only the
new floating projection/query modules are cast after conditioner assembly; a
whole-conditioner dtype cast would corrupt FP8 codes and lower scale precision.
Nonpersistent native RoPE is restored with the existing FP32 preservation path.

Declared floating embedding/output-head aliases omitted by HF serialization are
restored by their meta Parameter identity after shard assignment. If both alias
copies are present, disagreement is refused rather than silently selecting one.
Packed tied aliases are refused. Native vision activation-dtype discovery and
the language FlashAttention FP32-recasting fallback read a packed Linear's
`compute_dtype`; ordinary floating fallback behavior is preserved. The upstream
vendor note and legal adaptation ledger record these local compatibility edits.
FlashAttention kernels/GPU execution themselves were not exercised.

Native bundle save/reload preserves packed conditioner payloads. The offline
DiT converter and run-scoped inference export now validate and copy prepared
MLLM/connector components verbatim rather than refusing them or applying another
lossy quantization. This supersedes earlier "all companions floating" statements:
companions retain their stored precision; DiT still must be floating before its
ConvRot conversion. Quantized companion structural preflight uses meta models,
not payload loading or physical model allocation. Full native checkpoint
selection likewise permits preserved packed conditioner components while
requiring floating DiT/projections/VAE. Full fine-tuning of selected packed
conditioner groups still meets the explicit packed-base refusal; conditioner
materialization/training and packed `mlp` support are pending subsequent stages.
Frozen-conditioner ordinary LoRA training is connected through the existing
shared loop and cache lifecycle.

Across this milestone, 46 unique affected CPU cases pass: 24 new prepared-
conditioner tests, 10 existing plain native DiT packed-bundle tests and 12
existing native DiT ConvRot tests. New coverage includes Design/Layer actual
RGBA generation and native bundle roundtrip in FP32/BF16, real Comfy ConvRot
conditioning for both meta-built components, packed vision reference encoding
in FP16/BF16, malformed header refusal before weight payload loading, declared
alias restoration/rejection, DiT ConvRot inference export with byte-identical
prepared conditioner files, and real two-step frozen-conditioner LoRA updates
under swap, CPU prefetch and persistent condition-cache modes. Packed conditioner
bytes remain unchanged through those training runs. No official weights,
physical GPU allocation or live HTTP API was used; the main checkout is untouched.
Compile/real imports and spec parsing supplement the cases. Frontend behavior
continues to use the existing artifact picker/structure display and dynamic
capability explanation; owner build/type/UI verification remains pending.
Conditioner full materialization/packed projections, optimized ConvRot training,
adapters, partition/global/style and remaining objective parity remain active.


### Full conditioner materialization and packed upstream training milestone

The existing explicit `full_finetune_dequantize_int8_base` opt-in now materializes
native conditioner INT8, E4M3 fallback and ConvRot Linears before optimizer
creation when at least one native conditioner group is selected. As for DiT,
conversion streams CPU weights into floating Parameters, then restores component
residency. This converts all packed conditioner Linears, including frozen groups;
trainable ownership and group learning rates are applied afterward. It can
increase memory even when only one group trains. Quantization loss remains in
the initial weights; this is not recovery of the original floating checkpoint.
The DiT path and its existing packed-base guard are retained.

`training.json` records separate strict schema-one `ming_full_initialization`
and `ming_conditioner_full_initialization` lossy-origin reports. Native loading,
full checkpoint writing, fresh training from the floating package and optimizer
resume retain both origin records and U-Net topology. A floating restart requires
no dequantization opt-in. These reports describe preparation, not checkpoint
identity or an additional runtime quantization setting.

Without materialization, selected floating groups such as queries/projections
can train through frozen packed MLLM/connector weights. The guard checks the
selected group's modules rather than refusing any packed upstream module;
selected packed language/vision/connector modules still fail before optimizer
creation unless explicitly materialized. Frozen packed upstream Linears use
existing differentiable dequantization rather than inherited W8A8 flags, with a
`ming_conditioner_quantized_base_training` execution report saved in the full
checkpoint. Execution is configured from the current loaded modules on each
training startup, independently of the saved lossy-origin report.

Across this milestone, 33 unique affected CPU cases pass: twelve new conditioner
full-training/materialization cases, nine existing full-adapter ownership/export
cases, five resume cases and seven existing strict DiT-origin parser regressions.
New public request-model/YAML/runner paths train Design/Layer packed language
and connector weights for two steps, preserve the frozen DiT, and save/reload
floating six-component packages. Additional cases verify a formerly packed
vision weight updates with Layer reference supervision; selected floating
queries/projections update while packed upstream codes/scales stay unchanged;
packed selected groups fail without opt-in; and combined DiT/conditioner
materialization retains both reports and topology. The combined-origin case
checks preparation/save/reload, not simultaneous DiT/conditioner training.

The two new packed-conditioner resume cases checkpoint pending gradients during
MNT2/GAS4, then match uninterrupted predictions, weights, optimizer moments and
scheduler state through the remaining six backward batches and two updates.
Exact CPU comparisons use the previously documented fixture storage-alignment
normalization for both DiT and conditioner Parameters; this does not promise
bitwise equality across arbitrary allocation layouts or hardware. Three existing
floating-conditioner/materialized-DiT resume cases also pass. Backend compile,
real CUDA-guarded imports and OpenAPI default/description validation supplement
the tests. Frontend static review confirms the shared option/type, SSOT false
default and full-training submission gate. Owner build/type/UI checks remain
pending. No live HTTP, official weights, CUDA allocation or main changes occur.

Packed `mlp` projections, optimized ConvRot training, adapter families,
partition/global/style, remaining objective parity and final owner acceptance
remain active stages. This milestone supersedes the previous conditioner
full-materialization pending boundary; it does not close the full parity goal.


### Prepared quantized conditioner projection milestone

The native `mlp` component now accepts INT8, E4M3 fallback and declared ConvRot
Linear weights for `proj_in`, `proj_out` and `proj_directvlm.1`. Its projection
modules are built from the validated language/connector/native projection
configs on meta, then use the same strict slot validator and CPU shard loader
as DiT/MLLM/connector. This preflight precedes companion weight payload loading.
Query-token Parameters and RMSNorm weights remain floating; packed queries,
packed normalization, missing/wrong scales and unknown slots are refused.
Packed payload dtype and FP32 scales are retained while ordinary query/norm/bias
weights use the requested compute dtype. No whole packed-projection dtype cast
occurs during conditioner assembly.

A shared projection constructor now serves both ordinary native assembly and
meta preflight. Native attachment retains the prior Parameter registration
order (queries, connector, then projections) to preserve optimizer ownership and
resume compatibility. Native bundle checkpointing preserves packed projections;
the offline DiT converter and run-scoped ConvRot inference export preflight and
copy all prepared conditioner components verbatim, without requantizing their
stored payloads. VAE quantization remains unsupported. This supersedes the
previous packed-`mlp` refusal and floating-projection export boundary.

Full training of selected packed projections uses the existing explicit one-time
materialization opt-in and separate conditioner lossy-origin report. Without
that opt-in, selected packed projections fail before optimizer creation. Floating
query tokens can still learn through frozen packed projections with the existing
dequantization autograd path. The prepared-artifact precision capability reason
now names projections as well as MLLM/connector; runtime text-encoder
requantization remains unsupported. Existing API types/defaults and UI selectors
are unchanged; no new default or request parameter is introduced.

Forty-four unique affected CPU cases pass in this milestone: twenty new packed-
projection cases, ten existing conditioning cases, nine full-adapter cases and
five resume cases. Actual Design/Layer RGBA inference and bundle roundtrips cover
FP32/BF16 mixed INT8/E4M3 and real Comfy ConvRot projections. Full projection
training updates formerly packed weights and starts fresh from the floating
checkpoint without the opt-in. Floating-query training preserves frozen packed
projection bytes; a combined packed MLLM/connector/projection Layer fixture also
trains queries, preserves all seven packed Linears and restores its full
checkpoint in both plain and ConvRot formats. DiT ConvRot training export keeps
prepared projection bytes unchanged and restores U-Net topology.

Two new materialized-projection MNT2/GAS4 pending-gradient resume cases match
uninterrupted predictions, weights, optimizer moments and scheduler state using
the documented canonicalized CPU storage fixture. Existing packed-language/
connector and floating-projection resume regressions also pass. Compile and real
CUDA-guarded imports supplement the cases; frontend capability/type paths were
reviewed statically. Owner build/type/UI and official-weight GPU acceptance stay
pending. No live HTTP, CUDA allocation, official payload or main modification
was used. Conditioner conversion authoring, optimized ConvRot backward/cache,
adapter families, partition/global/style and remaining objectives still require
subsequent implementation and final acceptance; the full goal remains active.


### Native dual-zero CFG null training milestone and adapter scope audit

Ming now declares the `collated` CFG-null stage in both its training handler and
API capability mirror. Native inference zeros both conditioning streams at the
positive sequence lengths; an empty-caption encoding would be a different
condition. The handler applies the shared collated hook to the current MNT
label, builds an out-of-place dual-stream rewrite, and passes a replaced native
batch to the objective. Cached conditional tensors, query lengths, reference
latents, Layer auxiliary/layer weights and target supervision are retained.
Selected rows have zero conditioning gradients; later MNT labels start from
the original batch. Direct objective callers use the same rewrite helper.

Existing `cfg_uncond_drop_rate` tri-state resolution, request validation/YAML,
shared Bernoulli draws, per-MNT versus per-window selection, caption-dropout
conflict handling and split loss metrics are wired without new request fields.
The only new default is Ming's explicit 0.0 entry in the central per-architecture
SSOT; the shared optional request default remains null. Split metrics consume
the native frame-weighted prediction loss, separately from reconstruction loss.
Frontend capability/default-driven controls now become available and their
Ming-specific explanation says native references are preserved. Existing
non-Ming reference explanatory behavior is unchanged. OpenAPI records the
native dual-zero construction and omitted-rate default.

Sixty-three unique affected CPU cases pass across the staged checks: seven new
Ming CFG-null cases, one new pending-gradient resume case, and fifty-five
existing shared resolver/native objective regressions. Actual Design/Layer full
and ordinary LoRA loops consume public request-model/YAML/runner settings,
execute two MNT backward passes with zero streams, update weights and publish
per-item metrics before the shared loop consumes them. Layer references and
zero-weight auxiliary supervision stay present. Out-of-place masking tests
check independent later labels, exact zero-branch shape and selected-row versus
conditional-row gradients; invalid masks are refused. A Layer MNT2/GAS4 resume
with rate 0.5 and per-MNT redraw matches uninterrupted conditioning, predictions,
weights, optimizer moments and scheduler state after a pending-gradient seam.
The fixed delivered-stage/default test expectations were updated for the new
Ming declaration; the resolver/handler mirror and collated-hook placement gates
pass. Compile/real CUDA-guarded imports, spec/default checks and UI static review
supplement this evidence. Owner build/type/UI and official-weight GPU acceptance
remain pending; no live API, CUDA allocation or main modification occurred.

The adapter scope audit reads the authoritative generation/training pair tables:
both Qwen and Ming currently enable only `(lora, false)`. Their target inventory
is attention plus the explicitly selected caption projection, not configurable
FFN adapters. Qwen's conditioner LoRA, ReLoRA and LoHa/LoKr/DoRA refusals remain
refusals here. Earlier historical notes calling those families or conditioner
adapters pending must not be interpreted as an obligation to enable algorithms
Qwen does not ship. The design introduction/target row now reflect that boundary.
Global partition adapters are a separate shipped Qwen feature and remain in
scope, as do partition inference/training, optimized ConvRot backward/cache,
conditioner conversion authoring, style operators, guidance objectives and final
acceptance. Ordinary adapter capability parity does not complete those features
or the full goal.


### Native guidance-target objective and sigma-null coupling milestone

The existing `qwen_guidance_loss_*` and `qwen_cfg_null_sigma_schedule` API/YAML
keys now serve Ming as well as Qwen. Their historical names remain stable for
saved runs and the shared frontend objective controls. Ming's central
per-architecture default explicitly uses the SSOT ordinary-MSE weight (0.0);
Qwen retains 1.0. Explicit and restored run settings are preserved. The native
Ming objective panel identifies the architecture, dual-zero/reference behavior,
frozen-conditioner requirement and target extrapolation rather than calling its
factor the sampling CFG scale. All new backend defaults still come from
`api/param_defaults.py`; OpenAPI documents both architectures and the schema
example includes Ming's default.

Guided selections first evaluate a no-gradient native dual-zero forward at the
same noise, sigma, target frames and reference latents, retaining its prediction
output dtype. The ordinary conditional forward follows with gradients. The
shared Qwen schedule, stochastic selection, high-noise smoothstep and RMS-cap
math are reused; native loss reduction weights each supervised frame and
excludes zero-weight Layer auxiliary frames. The target is
`u + g * (clean-minus-noise target - u)`, where `u` is the native null prediction.
Reversing both target and null prediction from Qwen's velocity sign preserves
this algebra. The clean-space RMS cap likewise uses the native sigma and
supervision weights. This extrapolation factor is independent of Ming's served
sampling CFG convention. Reconstruction loss continues to reconstruct the
conditional prediction with `noisy + sigma * prediction`.

Null-labelled items retain ordinary MSE. A shared logical-batch/MNT draw chooses
guided loss before the nested ordinary-MSE null test, and shared slicing carries
that draw into microbatches. Ming applies its own collated zero-stream rewrite;
Qwen alone still substitutes a cached empty-prompt encoding. The optional
sigma-null path respects the existing per-MNT contract. A Ming high-noise ramp
with low weight 0 and positive high weight remains active, including logical
batch draws and resume; it must not be mistaken for a disabled constant-weight
objective. Qwen's existing resolution behavior is preserved. Public Ming
preflight refuses a trainable conditioner with an active guidance objective
before loading weight payloads; direct trainer runs retain the shared backward
path guard. Selected guidance without a null prediction is an error.

Native guided/ordinary split, selection fraction, effective scale mean/min,
capped fraction and clean-space null distance travel through the shared extra
metric channel. Per-item guided loss supplies concept weighting and CFG-null
split monitoring. No database field or new request parameter is introduced.
The extra no-gradient forward increases compute for selected guided items;
physical GPU memory/performance and official-model quality are not measured.
Partitioned guidance and native pixel-debug surfaces remain separate parity
work, not inferred from these full-frame tests.

Sixty-seven unique affected CPU cases pass across the staged checks: twenty new
Ming guidance cases, three new pending-gradient resume cases, seven existing
Ming CFG-null cases, thirty-five existing Qwen guidance/sigma-null/cap cases and
two shared guidance-default/YAML regressions. Independent dense oracles cover
constant/sigma scaling, weighted Layer supervision, RMS clipping, gradients and
stochastic null exclusion. Public request-model/YAML/runner paths execute real
Design/Layer full/ordinary-LoRA loops in stochastic and blend modes, update
weights and verify no-gradient-first ordering with identical noise/sigma.
Controlled draws exercise guided, null-MSE and conditional-MSE paths in both
profiles. BF16 native training retains finite prediction precision and updates
weights; trainable-conditioner startup and public preflight refusals are covered.

Three MNT2/GAS4 pending-gradient resumes match uninterrupted conditioning,
predictions, weights, optimizer moments and scheduler state: ordinary stochastic
guidance on Design, coupled sigma-null guidance on Layer, and zero-low-weight
high-noise guidance on Design. Each baseline confirms at least one null forward
was selected. Compile/real CUDA-guarded imports and SSOT/OpenAPI validation
supplement the tests. Frontend static review covers model-change defaults,
objective presets, outgoing weights/sigma-null fields, historical-key types,
conditioner explanation and error/validation paths. Owner build/type/UI and
GPU acceptance remain pending; no live API, CUDA allocation, official payload
or main modification occurred. Partition/global adapters, style operations,
optimized ConvRot backward/cache, conditioner converter authoring and final
complete-parity audit/owner acceptance remain active.


### Exact full-KV query chunk training milestone

The existing `qwen_full_kv_query_chunk_tokens` field now reaches Ming through
request validation, YAML generation, runner preflight, native component loading
and the shared training UI. Its default remains the centralized value 0.
Component loading installs the attention conduit for fresh and checkpoint-based
training; the architecture hook delegates to that same setup routine. This fixes
a missing runtime installation that direct attention-hook tests did not cover.
All main, noise-refiner and context-refiner blocks receive the selected chunk.

Only query rows are divided. Every target Layer frame, reference, caption and
native padding key remains in the joint K/V context; Q/K normalization and RoPE
are applied before splitting. Key-only ragged masks pack K/V once and reuse that
pack across chunks, retaining padded query rows. Query-dependent masks use global
query-row slices, and all-masked rows retain native dense semantics. This does
not add parameters or change topology/checkpoint tensors. The training run YAML
preserves the execution setting; it is not an architectural checkpoint property
or a new generation parameter. Smaller chunks can reduce explicit attention
matrix memory, but add launches and retain full Q/K/V/outputs. FlashAttention
already tiles internally, so no hardware speed or VRAM improvement is claimed.

Thirty unique affected CPU cases pass across staged checks: thirteen new dense
oracle/packed-conduit/native-model forward and gradient cases, four real public
request-model/YAML/runner full/ordinary-LoRA loops across Design and Layer,
three malformed preflight cases, seven existing attention cases, two new
query-chunk guidance pending-gradient resumes, and one existing zero-chunk
CFG-null pending-gradient resume. Native-model oracles cover nested U-Net,
ragged captions/references, multiple Layer frames, learned/zero padding and all
parameter/input gradients. The packed Flash path uses a CPU kernel shim rather
than a physical Flash kernel. MNT2/GAS4 checkpoints retain pending gradients;
continued native predictions, weights, optimizer moments and scheduler match
uninterrupted CPU runs with chunk 17, guidance and Layer sigma-null selection.

Changed backend files compile and import with CUDA initialization forbidden.
OpenAPI parses and agrees with the centralized default. Frontend static review
covers visibility, zero-preserving defaults, input sanitization, existing API
number types and YAML/request serialization. Owner build/type/UI verification
is pending and does not block implementation. No live API, official weight
payload, GPU allocation or main edit occurred. Approximate spatial partition,
global adapters, styles, optimized ConvRot training/cache, conditioner converter
authoring, native pixel-debug/partition guidance and final parity acceptance
remain active work; exact query chunks do not establish those capabilities.


### Native spatial partition coordinate foundation

`MingPartitionAdapter` declares joint caption/target attention and full-canvas
positions. Its complete-coverage planner reuses the shared deterministic core
and halo bookkeeping on the spatial patch grid. These regions are approximate
attention contexts, unlike exact full-KV query chunking. Layer frames share the
same spatial region and retain their native distinct first-axis coordinates.

The native DiT accepts an optional `target_spatial_position_ids` rectangle.
Target tiles and full reference frames are patchified independently, so a tile
can retain every pixel of a larger reference grid. Reference first-axis IDs
follow all target frames; spatial IDs remain native full-reference coordinates.
Alignment padding is added after concatenating both streams. Only target frames
are reconstructed on this explicit path; the ordinary forward still returns its
native Layer target/reference frame layout. A full-canvas explicit rectangle
produces exactly the ordinary target predictions and all parameter/input
gradients in the CPU oracle. No parameter or checkpoint format changes.

Target-only residual U-Net transport accepts a spatial origin. At each coarse
level the origin maps to the corresponding coarse-grid cell by integer division;
local coarse coordinates then advance normally. References/captions keep their
own original rotary context. Default origin zero preserves existing behavior.
Crop dimensions must still satisfy the recorded nested topology's alignment.

Twenty unique affected CPU cases pass in staged checks: twelve new full-box,
cropped-target/full-reference, malformed-coordinate and coverage cases plus
eight existing rewiring cases. Both native padding profiles and dense/nested
U-Net execution are covered. Independent patch oracles verify frame/token order
and full-reference coordinates; gradients reach reference pixels outside the
target tile. Backend compilation and real imports forbid CUDA initialization.
No live API, official model weights, GPU resources or main files are touched.
This is the coordinate/forward foundation: shared training-loop dispatch,
sequential region backward, image-level guidance caps, native debug output,
public partition options/UI and global adapters still require implementation.


### Partition supervision and image-level guidance cap primitives

Native partition supervision now extracts only each region's loss core from its
halo-bearing prediction. It applies the existing independent-frame supervision
weights, so the Layer auxiliary frame contributes neither loss nor cap distance
when its weight is zero. Core fractions are spatial patch area divided by full
canvas patch area; their sum is one. A detached clean-space RMS distance is
accumulated over every disjoint core before resolving one guidance scale for
all regions. Guidance accepts that explicit scale instead of recomputing a
region-local cap. This is necessary for an image-level extrapolation limit.

Fifteen unique targeted CPU checks pass: ten new loss/cap/halo checks and five
existing guidance oracles/selection checks. Independent full-image calculations
match area-weighted sequential core losses and prediction gradients for 2/4
regions, zero/nonzero halo, constant/sigma guidance and weighted Layer frames.
Large auxiliary-frame null predictions do not affect the cap. Halo-only
predictions receive no loss gradient. Incomplete cap core collections are
rejected. These are algebra and autograd checks on prediction tensors, not proof
of native sequential model backward or end-to-end partition training. Changed
backend files compile and import with CUDA forbidden; main API/GPU remain
unused. Shared trainer dispatch, model backward graph lifetime, reconstruction
loss, checkpoint policy, debug/metrics, public UI and global adapters remain
in the partition milestone ahead.


### Native sequential partition backward primitive

`partition_training.backward_regions` executes actual native tile forwards and
backwards sequentially with one supplied full-image noise tensor and sigma draw.
All target frames share each spatial tile, while complete reference grids and
dual caption streams remain present. Selected native-null forwards run without
gradient before conditional tile forwards. Detached null cores establish one
image-level guidance cap. Each region contributes its supervised core area
fraction, frame-weighted prediction objective and clean reconstruction objective
using Ming's clean-minus-noise velocity sign. Shared upstream conditioning graphs
are retained until the last region backward; ordinary per-region model graphs
can then be discarded. The caller supplies MNT/GAS backward scaling. GradScaler
uses the same scaled loss path, but physical CUDA/scaler behavior is not verified.
Per-item prediction/concept losses and area-weighted guidance metrics are retained.

Eight new CPU cases pass on real tiny native Design/Layer models, with dense or
nested U-Net topology and plain or image-capped sigma-guidance objectives. A
separate one-backward oracle sums the same native tile predictions, computes
weighted objectives/cap directly, and matches reported loss, per-item concept
loss and every parameter gradient. Plain cases also match both live conditioning
input gradients across all four regions. This establishes sequential native
backward against a tile oracle, not equivalence to full-image model attention.
Changed backend files compile and import with CUDA forbidden. No API/GPU/main
use occurred. Shared trainer dispatch, MNT/GAS/resume integration, fused-optimizer
refusal, partition checkpoint policy, cancellation/error paths, native debug,
public options/UI and global adapter wiring are still pending.


### Shared trainer partition dispatch milestone

Ming's shared forward/backward entry now dispatches enabled spatial partitions
to native sequential backward before the ordinary full-frame objective. The
existing canonical `dit_partition_*` fields and legacy saved aliases drive the
plan. Threshold mode counts actual target patch tokens across all Layer frames,
not raw latent pixels or reference tokens. Fixed mode uses shared deterministic
split/halo planning keyed by epoch and batch occurrence. Disabled or under-limit
threshold mode retains full-frame execution. No default value is introduced.

The shared wrapper applies collated dual-zero CFG labels once, supplies the
existing MNT noise draw, and divides microbatch loss scaling by GAS before
backward. It flushes the existing metric/concept channels after backward and
refuses fused per-parameter optimizer updates. Ming's architecture declares
partition support and exposes its audited native adapter. The residual-rewiring
training gate now permits spatial partition only for Ming; Qwen's gate and
partition-global/TREAD/REPA/block-skip exclusions remain. Every region's nested
U-Net geometry is checked before any region backward starts. Misaligned crop
plans still fail instead of silently dropping or reshaping a topology.

Eight unique targeted CPU cases pass: four public request-model/YAML/runner
Design/Layer full/ordinary-LoRA training runs, three threshold-token accounting
cases and one architecture-scoped rewiring gate test. The public runs perform
MNT2/GAS2 real training, image guidance and query chunk 17, update weights and
save checkpoints. A preceding GAS1 pass was not counted again. Backend compile
and real imports forbid CUDA initialization. These are offline route helper and
request-model calls, not HTTP; no main API/GPU/official payload was used.
Partition checkpoint block overrides, exact pending-gradient resume, native
debug/cancellation/error acceptance, generation parity, UI exposure and global
adapters remain pending. In particular, the existing partition checkpointing
parameter has not yet been implemented for Ming and must not be reported ready.


### Partial checkpointing configuration milestone

The canonical full-frame and partition checkpoint-block controls now execute on
Ming's main stream. Runner preflight resolves `n_layers` from the native artifact
config without reading weight payloads, instead of relying on a fixed class
constant. Strict limits reject boolean/fractional/out-of-depth values and
conflicting legacy aliases. Fresh/checkpoint component loading installs the
full-frame limit. Dense main-block execution now honors the same prefix limit
as residual U-Net transport. Noise/context refiners retain native global gradient
checkpointing behavior; the main-stream count does not disable their recompute.
Global `gradient_checkpointing=false` still disables all checkpoint execution.

Partition backward temporarily applies its optional block count and restores
both a previous value and a previously absent attribute in `finally`. Null
inherits full-frame execution, preserving memory policy without automatically
spending partition savings. This supersedes the earlier pending block-override
note. OpenAPI explains the main/refiner distinction and restoration semantics.

Sixteen unique targeted CPU cases pass: six actual native dense/nested U-Net
forward/all-gradient checks at main counts 0/2/6, two exception-restoration checks,
four strict bound/type checks, and four public Design/Layer full/ordinary-LoRA
MNT2/GAS2 partition loops with full-frame count 2 and partition count 1. Public
runner preflight accepts native depth 6 rather than assuming official depth;
post-training restoration is asserted. Changed backend files compile and import
with CUDA forbidden. Main API/GPU and official payload remain unused. Physical
GPU memory/recompute performance, pending-gradient partition resume, final public
UI/debug/error paths, global adapters and broader parity acceptance remain work.


### Partition pending-gradient resume and initial training UI exposure

Two new actual shared-loop CPU continuation cases pass on Design and Layer:
MNT2/GAS4, two spatial regions, inherited nested U-Net, stochastic sigma-guidance,
query chunk 17, full-frame checkpoint count 2 and partition count 1. The Layer
case also uses sigma-null/per-MNT labels. A step-2 checkpoint contains two pending
accumulated backwards before the first optimizer update. Resuming it matches the
remaining six logical training traces exactly, including full-image clean/noise/
sigma, both conditioning streams, region input coordinates and each native
conditional tile prediction. Final DiT/conditioner tensors, optimizer moments,
optimizer update count and scheduler state match uninterrupted CPU execution.
Null forwards are confirmed selected. This proves same-config completed-batch
continuation, not a mid-region interruption, changed-plan resume or CUDA bitwise
promise. The existing runtime checkpoint mechanism needed no production changes.

Training UI exposes Ming's verified spatial partition and main checkpoint counts
alongside Qwen. Defaults use the server-provided parameter table, explicit zero
checkpoint limits remain zero, empty limits remain null, and native upper-depth
validation is performed by the server. Threshold text explains all target Layer
patch tokens; references remain complete. Mode-dependent inputs now use the same
fallback as their select state. Qwen-only ConvRot cache, synchronized CUDA timing
and global-adapter controls remain separately gated while Ming wiring is pending.
Ming global/profile settings imported from old Qwen presets are a remaining
transition/preflight concern for that next milestone; hiding those controls is
not evidence they are supported or that complete parity has been delivered.

Static frontend review covers API number/null types, input bounds, fallback and
mode state, existing serialization, and the shared approximate-context warning.
No frontend build/type-check or physical browser interaction was run. Backend
test compilation, real CUDA-guarded imports and SSOT/OpenAPI parsing pass.
No live API/GPU/official payload/main use occurred. Owner frontend build/UI checks
remain for final acceptance. Global adapters, synchronized timing/debug/error
paths, generation partition parity and the other planned Qwen surfaces remain
active work.
