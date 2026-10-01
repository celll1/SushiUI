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

Ming target-only residual U-Net execution and checkpoint topology restoration are now implemented. Eight CPU tests cover nested transport with ragged auxiliary lengths, independent Layer frame grids, all-main-block gradients, identity reconstruction, gradient checkpointing, profile identity/resume conflicts, the deepest-four-block preset and exact native DiT parameter counts. The seven conditioner tests also exercise a full tiny package with saved topology: strict loading restores it, and both package and manifest inspection report the same topology and parameter census. The affected batch passes 15 tests. Shared inspection/editor contracts remain unchanged; Ming previews describe per-target-layer grids and preserved auxiliary-token overhead. Trainer registration, common trainer/export API integration and production UI wiring are still pending.

The native flow objective now accepts explicit sigma, ordered target-frame supervision weights, ragged reference latents and aligned dual-stream CFG dropout. Ten CPU tests cover the native time/sign, target-only weighted loss, both profiles' U-Net/conditioning gradients, malformed supervision refusal and independent component learning-rate ownership. These are objective primitives; the common trainer, datasets, checkpoint/resume and API registration remain unconnected.

Native component-bundle export is implemented in `checkpoint.py`: all six native components plus tokenizer/processor are written to a sibling staging directory, verified, then published without overwriting an existing destination. DiT uses the shared sharded-state writer with identical topology metadata on every shard and index. Export requires explicitly CPU-offloaded components and does not change residency or training mode. The seven conditioner/package tests now also round-trip a forced-sharded full bundle, compare all DiT/conditioner/VAE states and conditioning outputs, and verify collision refusal and mid-export cancellation cleanup. This proves the native inference bundle format; common trainer optimizer/RNG/scaler resume and export API integration remain pending.

Native prompt/vision preparation and generation orchestration now share `prompting.py` and `generation.py`. Official HUMAN/system/assistant templates and processor vision geometry are retained; VAE references separately use the explicitly requested 16-pixel-aligned grid, preserve RGBA channels and remain ordered. Mixed reference counts are retained for training but rejected for a rectangular generation batch. Nine CPU tests cover native chat/image tokens, reference alpha/order, text batches, full language/vision/projection gradients, end-to-end tiny Design image-edit and Layer RGBA generation, early cancellation and caller-owned residency-context release after step failure. This does not yet register platform generation routes, residency policies or frozen-conditioning caches.

`inspection.loaded_model_info` now provides the loaded structure snapshot for platform wiring: unique DiT/language/vision/connector/projection/query/VAE parameter ownership, trainable counts, logical dtype/device bytes, separate serialized header storage, refiner/main depth, checkpoint topology and conditioner cache eligibility. It makes no VRAM measurement and explicitly excludes buffers/quantization payloads from loaded parameter-object counts. The seven package tests check JSON serialization, exact full ownership totals, DiT analytical census agreement and trainability/cache transitions. Current-model API/frontend consumption is implemented in the following stage.

Ming native directory/manifest detection, CPU component loading and lifecycle registration now use the common `ModelLoader` and `DiffusionPipelineManager`. The current-model response carries the complete structure snapshot; the model selector renders component parameter distribution, trainable/frozen ownership, dtype/logical bytes and separate serialized storage alongside the restored U-Net graph. Native Layer frames are explicitly classified as image layers, never video. Ten affected CPU tests pass, including real common `load_model`, topology restoration, current-model/wiring fields, scan classification and inventory cleanup. Frontend types/OpenAPI match and have been reviewed statically; owner build/type checking remains pending. This stage is load/inspect only: `generation_available=false`, and image-generation dispatch explicitly refuses until the platform generation backend is connected. No running API or GPU was used.

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
recipe restoration. This milestone transports settings only: platform
sampling and grouped gallery persistence remain disabled until their contracts
are connected and verified.

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
This helper awaits route and generation-mixin integration; it does not enable
live generation by itself. Thumbnails may use a white alpha preview, while
full PNG files retain native alpha.

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
