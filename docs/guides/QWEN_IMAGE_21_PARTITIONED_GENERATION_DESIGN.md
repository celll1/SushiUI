# Qwen-Image 2.1 partitioned generation and global-adapter contract

Status: **implemented; CPU acceptance and real ConvRot GPU execution passed. Halo-free coherence fails on the tested LoRA; the default halo is 16 latent pixels. Owner frontend build gate remains.**

This extends [partitioned training](QWEN_IMAGE_21_PARTITIONED_TRAINING_DESIGN.md).
The four image generation panels expose Off/Auto/Manual and a shared visual
editor. API capability data advertises partitioning only for Qwen-Image 2.1.

## 1. Scope and current evidence

Qwen training slices one noisy full-canvas latent into complete-coverage target
regions. Every region retains full-canvas RoPE coordinates. An optional
`latent_summary_v1` global adapter computes a low-rank full-canvas residual and
adds it after `img_in` through `target_input_residual`.

Before this change, generation warned about this adapter and used ordinary
LoRA branches only. The generation backend now prepares paired global payloads
and calls a generation-scoped executor for actual multi-region plans. The
vendor pipeline retains the existing full-frame branch and advances one
scheduler over the complete canvas in either mode.

Before this change, only the LoRA training adapter constructed the global
module. Qwen full FT accepted partition settings but did not construct it.
Consequently, old full-FT checkpoints cannot contain learned global weights
from that option. These weights cannot be recovered by enabling it on resume.

Implemented foundation:

- `core/models/qwen_image_21/partition_global.py` owns the shared adapter,
  metadata, payload validation, and training construction.
- LoRA and full FT construct the same module. Full FT includes all five
  trainable tensors in the transformer optimizer group, independently of
  `train_adapter`, which controls `txt_in`.
- Full-FT transformer checkpoints and transformer/TE bundles store global
  tensors under `qwen_partition_global_adapter.*`. Shard indexes retain the
  adapter version, rank, and summary-token count in metadata.
- Floating and INT8 component loaders validate and remove those tensors before
  strict transformer loading. They retain a CPU payload outside the registered
  module tree. INT8 inference export preserves this metadata and keeps the
  global weights floating.
- Full-FT preparation restores that payload before block-swap registration and
  optimizer construction. Disabled adapters, changed dimensions, and attempts
  to resume an old checkpoint with newly initialized global weights are refused.
- Ordinary generation and ordinary training forwards do not consume the
  global adapter. The generation loader does not register or move it to CUDA.

## 2. User controls

Expose one Qwen-specific **Partitioned generation** group in Generate:

| Mode | Meaning |
|---|---|
| Off (default) | Existing complete-frame generation. |
| Auto | Complete-frame below the configured target-token ceiling; otherwise a balanced plan satisfying the ceiling for every halo-expanded input region. |
| Manual | User-defined rectangular partition topology and boundaries. |

Keep the basic controls to mode, target-token ceiling (Auto), a visual layout
(Manual), and optional halo. Global adapters follow the loaded model/LoRAs
automatically when an actual multi-region plan is used; a separate strength
knob is unnecessary initially. Report the active adapter sources and their
effective LoRA strengths.

The token ceiling is a workload limit for Auto, not a hardware fit guarantee.
Do not copy the current training threshold's two/four-region rule: it does not
bound every tile's token count for arbitrarily large canvases. Auto recursively
splits the largest region until every expanded input is within the ceiling,
or refuses when the minimum legal tile or a resource guard prevents progress.
No silent tile-count cap may leave an oversized tile.

### Visual boundary editor

Use a reusable canvas/SVG component in txt2img, img2img, and inpaint/outpaint.
Show a blank aspect-ratio canvas for txt2img, and an available source/last preview
as an optional background. Overlay colored dashed core boundaries and faint
halo bands. Editing changes the layout, never crops or re-encodes the background.

Presets include equal halves, 2x2, three strips, and 2x3. A region may be split
horizontally or vertically, with the divider draggable; undo and reset preserve
complete coverage. Represent the layout as a recursive split tree so moving a
divider cannot leave gaps or create overlapping cores. Three/six-region plans
are new generation topologies, not existing fixed-count training features;
quality on adapters trained with two/four regions needs separate evaluation.

The frontend displays **actual snapped** pixel and latent coordinates, per-core
and halo-expanded token counts, and largest input size. Qwen has one target
token per 16x16 image pixels: 4096 image tokens correspond to 1024x1024 pixels.
Snap boundaries to even latent coordinates (32 image pixels), consistent with
the image-slot geometry and the training planner. This also permits legal
four-token image slots. Integer token fields use `step=1`; 4096 must be valid.

Normalized split ratios persist when the canvas size changes. Recompute the
actual plan, show snapped positions, and refuse illegal collapsed regions;
never send an obsolete plan with a new width/height. The API is authoritative.

## 3. Request and reproducibility contract (proposed)

Use an optional versioned `qwen_image_21_partition` object with:

- `version`: layout protocol version;
- `mode`: `off`, `auto`, or `manual`;
- `max_target_tokens`: positive integer for Auto;
- `layout`: normalized recursive split tree for Manual;
- `halo_tokens`: nonnegative even integer.

Defaults come only from `GENERATION_DEFAULTS`. Absence and `mode=off` resolve
identically. Update OpenAPI before the endpoint/parser changes. Both JSON and
multipart routes parse the same object. Forward it through `api.ts`, all panel
queue/dequeue and loop paths, and `GenerationQueueProcessor`; queued work keeps
an immutable copy of the selected layout.

Server validation precedes adapter installation and denoising:

1. Architecture is Qwen-Image 2.1 and the protocol version is supported.
2. Resolve the actual aligned canvas dimensions and normalized boundaries.
3. Cores cover the full target exactly once. Every input contains its core and
   stays inside the canvas. No empty region; each input has a legal slot count.
4. Enforce layout depth/count guards and the actual expanded-token ceiling for
   Auto. Reject NaN/Infinity coordinates and ambiguous incomplete trees.
5. Validate global weights and initial capability exclusions.

PNG metadata stores mode, aligned canvas, actual latent core/input rectangles,
halo, planner version, and adapter identities/strengths. Seed and these values
reproduce the run. Use one layout for the entire denoising trajectory and both
CFG branches. Random boundary jitter and step-dependent topology are deferred;
they would change neighborhoods over time and complicate cache identity.

## 4. Adapter ownership, loading, and composition

The supported first format embeds global weights with the corresponding LoRA
or full checkpoint. Do not discover arbitrary same-named sibling files. A
future sidecar format must name and hash its exact companion; basename alone
is insufficient.

For LoRA, extend the Qwen `AdapterSession` preparation/commit lifecycle to keep
a validated CPU global payload paired with each `AdapterFile`, before ordinary
key normalization. Do not register the payload as an ordinary attention branch
or count it as an unmatched LoRA target. An otherwise valid global-only payload
may be used for partitioned generation; complete-frame generation reports that
it has no active branches. Loading/unloading or changing LoRA lists must clear
the paired payloads atomically, including on a failed second file.

For full FT, consume the model loader's CPU payload only when the request
actually resolves to more than one region. Ordinary generation preserves it
without module registration, CUDA placement, or adapter forward calls.

Validate `latent_summary_v1`, exact tensor inventory, shapes, channel/hidden
dimensions, rank and token metadata, and floating dtype. A declared adapter
with missing weights is an error, never a newly initialized module. CPU payload
validation occurs at load; construct/cast the small inference module lazily
when partition execution starts, and release its CUDA copy in `finally`.

Composition for tile i at timestep t is:

```text
residual_i = G_fullFT(x_t, input_i)
             + sum_j effective_strength_j(t) * G_LoRA_j(x_t, input_i)
```

The full-FT term is absent unless stored with that model. LoRA strength zero
and inactive step ranges skip that adapter entirely. Use the same step-range
and Unet applicability rules as ordinary branches. Layer-specific LoRA weights
do not have a meaningful mapping onto a full-canvas adapter; keep the global
term tied to the file's overall effective strength and show that policy.
Do not multiply global outputs by ordinary LoRA `alpha/rank`, and do not merge
nonlinear global-adapter parameters. Multiple rank/token sizes are allowed
when all output dimensions match; outputs, not internal weights, are summed.

This additive rule applies to independent LoRA global paths initialized with
zero output. LoRA training directly on a full-FT base that already owns global
weights is refused in this foundation: copying that base module into the LoRA
would save an absolute replacement and later sum the base twice. A follow-up
contract must freeze the base global path and train a separate zero-output
additive path, with base identity recorded, before admitting that training mode.

If no loaded source contains a global adapter, hard-partition generation is
still possible with a visible warning about absent global communication.
There is no automatic initialization and no dense-attention equivalence claim.

## 5. Denoising algorithm

Keep the current vendor full-frame branch intact. Add a separate inference
executor in `core/inference/qwen_image_21_partition.py`, called only for a
multi-region plan. Move/re-export neutral geometry helpers if needed so the
inference executor never imports the trainer stack.

For every scheduler timestep:

1. Freeze one full-canvas `x_t` snapshot. All tiles, references, and global
   adapters read that snapshot; tile order must not feed partially updated
   latents into later tiles.
2. Slice each halo-expanded target region; retain complete text/reference
   conditioning and full-canvas target RoPE IDs. Rebuild target image-slot
   masks and target shape while preserving condition-image slots and layouts.
3. Calculate the global residual from **noisy full target latents**, as in
   training. Pad the residual with zeros for condition-image tokens; their
   projected states must not be modified. Use this same residual policy for
   conditional and negative/null branches.
4. Obtain conditional and, if requested, negative velocity predictions. Copy
   each non-overlapping core into one full-canvas velocity buffer. Halo
   predictions are context only and do not get averaged into neighboring cores.
5. Apply existing true CFG to the assembled velocities and invoke the scheduler
   **once per full image per timestep**. Do not run independent tile samplers.
6. Run reference-guide updates, predicted-x0 previews, callbacks, and progress
   on the complete canvas at their existing logical-step position.

Poll cancellation between regions. A partially completed velocity buffer must
never advance the scheduler. Decode the final full latent through the existing
VAE and retain existing inpaint/outpaint protected-pixel compositing.

The global residual touches only target states and prefix cannot attend to the
target under causal conditioning. Thus prefix K/V can be shared across regions
and timesteps **if** verified against the uncached path. Keep separate cond/
negative caches. Extract from the first tile, use cached mode for later tiles,
and include model/LoRA identity, effective adapter step state, text/reference
content and geometry, dtype, and attention processor in cache identity.
Step-range transitions invalidate prefix caches when ordinary LoRA weights
change. Initial correctness tests use caching off; enable shared caching only
after the numerical gate passes. Do not retain one full cache per tile.

Initial partition capability: native text/reference conditioning, true CFG,
previews and final protected compositing. Reference Guide and segmented Style
Transfer need separate tile-aware parity gates; initially reject them with
partition mode instead of silently dropping or misapplying them. Off mode
retains their current support.

## 6. Ordinary-path cost boundary

When Off or Auto resolves to one region:

- Call the existing complete-frame vendor pipeline and transformer arguments.
- Do not construct partition buffers, a global inference module, or a new cache.
- Do not transfer global weights or calculate global summaries/residuals.
- Do not install forward hooks or wrap the transformer with a partition proxy.
- Preserve existing CFG, reference/style paths, offload and callbacks.

An embedded payload necessarily has a small checkpoint-read/CPU-storage cost;
the contract is no extra denoising computation or CUDA residency for ordinary
generation. Clear CPU payloads with model/LoRA unload. Verify repeated
partition-to-Off requests leave no GPU adapter or hooks behind. One-region
ordinary outputs must match the existing route under the same deterministic
backend; measure memory and timing rather than claiming zero overhead from
static inspection.

## 7. Delivery and acceptance gates

1. **Checkpoint foundation (this change):** full FT construction/optimizer
   membership, floating/sharded and TE-bundle resume, strict malformed-payload
   rejection, and INT8-export preservation. CPU FP32/BF16 tests prove actual
   gradient updates, tensor restoration, and ordinary forward isolation.
2. **Executor:** neutral geometry, manual/Auto planning, CFG/full-canvas Euler,
   halo/core coverage, global composition and cleanup. Tiny CPU integration
   first, then owned GPU gates. Cache optimization follows correctness.
3. **API/UI:** OpenAPI/defaults, transport/queue parity, reusable visual editor,
   metadata/preview overlays and truthful capabilities. The owner runs frontend
   type checking/builds.
4. **Performance/quality:** paired seed Off versus partitioned runs using the
   same model and LoRA. Measure end-to-end seconds/image and peak memory at
   1024 and larger resolutions, seams/coherence, CFG 1 and >1, negative prompts,
   native references, two/four and three/six layouts, and full-FT models.

Required tests include identical Off outputs with/without inactive global
payloads; saved global weights never replaced by random ones; conditional and
negative residual routing; full coverage and per-step scheduler-call count;
tile traversal order invariance; cache-on/off parity and invalidation; adapter
failure rollback; repeated mode switching; cancellation cleanup; UI 4096 input;
and queue/dequeue/loop reproduction of exact snapped boundaries.

### Implemented transport and validation

`qwen_image_21_partition` is an object for JSON callers and a JSON-encoded form
field for all four image multipart routes. Defaults are served from
`param_defaults.py`. The version-1 shape is:

```json
{"version":1,"mode":"auto","max_target_tokens":4096,"layout":null,"halo_tokens":16}
```

Manual layout uses `{axis, ratio, first, second}` split nodes, with null leaves.
The current guards are 64 regions and depth 8. Invalid/collapsed geometry fails
with a validation error. Auto does not silently exceed the input-token ceiling.
Queues and image loops forward the config and KV-cache switch. The editor
offers drag, selected-region splits, 2/3/6 columns, 2x2/2x3, undo, reset and
optional source background in img2img/inpaint. Outpaint shows the sampled canvas
without misplacing its source image. PNG/gallery metadata records requested
configuration and `qwen_image_21_partition_actual` with snapped core/input boxes
and adapter names, strengths and step ranges.

KV caches are scoped to one generation and CFG branch. Within that scope model,
reference images, dtype, geometry and attention processors are fixed. Effective
ordinary-LoRA step transitions and text tensor replacement/in-place changes
invalidate caches. Cancellation polls between regions and raises before any
partial scheduler update; backend `finally` releases caches/global modules and
unloads ordinary LoRAs. No partition executor or global GPU module is created
when Off/Auto/Manual resolves to a complete-frame call.

CPU acceptance covers FP32/BF16 cache parity with native reference-image slots,
region traversal invariance, 3/6 coverage, halo ceilings, lazy weight restore and
RNG isolation, additive global outputs with zero reference residual, separate
CFG caches, one Euler update per timestep, inactive adapters, prompt/step cache
invalidation, failed second-file rollback, and exception/cancellation cleanup.
The combined generation/checkpoint/training/export suite passes 120 CPU tests;
two CUDA tests are excluded. All changed Python modules compile and import
with CUDA initialization stubbed; the assertion confirms no CUDA context.
The full-FT foundation tests also cover real training gradients, sharded/TE
bundle restoration, ordinary-forward isolation and INT8 export preservation.
Browser interaction still requires the owner frontend build gate; CPU tests
establish no speed or quality claim.

### Real GPU checks (2026-09-28)

RTX 6000 Ada, local ConvRot transformer/TE, BF16 inference and model CPU offload:
Run166 step 17801 passed 1024x1024 Off, one-region Auto, 2/6-region execution,
CFG 1/3, native image references and a full-model CPU-payload injection.
The full-model injection tests the inference branch, not a separately trained
full-FT model's quality. Off is pixel-identical with/without the paired global
payload and after partition requests. Every request unloads ordinary adapters.

The owner's `20260926_193229_22397712` checkpoint at step 24411 contains all five
global weights, version `latent_summary_v1`, rank 64, 16 summary tokens and an
output-weight L2 norm of 3.100887. Its 128 ordinary LoRA targets also load.
At its recorded 1472x896 / CFG1 / 40-step / seed611809507 sample settings:

| Route | Wall seconds | Peak allocated GiB | Visual result |
|---|---:|---:|---|
| Off | 42.25 | 9.238 | One coherent subject |
| 2 regions, halo 0 | 31.32 | 9.243 | Face/body mismatch at the central boundary |
| 2 regions, halo 16 (256px) | 48.37 | 9.295 | Central face split resolved in this comparison |

These are single sequential calls with unequal warmup conditions, including
LoRA loading and decode, not a speedup benchmark. A forward pre-hook measured
nonzero global residuals on all 80 halo-enabled region calls (RMS range
0.0007529–0.0659909). Missing/restored weights are therefore not the explanation
for the halo-free failure. The global summary alone does not replace spatial
border context. The generation default is now halo 16, while Off/one-region
plans still create no partition executor. Auto counts the expanded input in
its ceiling. This one comparison does not prove coherence for all seeds, layouts
or checkpoints; halo 0 remains an explicit experimental choice.

No quality equivalence or speedup is assumed. Splitting removes dense target
attention, and a learned summary does not restore it. Global adapter training
quality and benefits for new layouts require measured generated samples.
