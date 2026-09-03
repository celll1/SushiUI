# LyCORIS adapter integration design

Status: investigation and implementation plan. Shipped so far: the additive
LyCORIS algebras (LoHa, LoKr) LOAD AND GENERATE on Z-Image, Krea 2, MiniT2I,
LTX-2.3, Anima, Lens, Ideogram 4, FLUX.2 and ACE-Step (sd-scripts codec only)
— generation only, by file path. Everything else here is plan.

This guide evaluates LyCORIS 4.0.0 for SushiUI and defines the work required to
support LoHa, LoKr, and weight-decomposed adapters (DoRA, DoHa, and DoKr) in both
training and Generate > AddLoRA. It also records defects in the current ordinary
LoRA round trip that must be fixed before adding more algebra.

## Decision

The adapter families and checkpoint conventions are implementable. The reusable
part should be an architecture-neutral adapter engine; each architecture should
only supply target discovery, component policy, and checkpoint-key translation.
LoHa and LoKr are the best first additions because their additive branches can
compose with a dense or quantized base forward. DoRA should start on dense Linear
targets only because it needs the base-weight direction and norm.

LyCORIS 4.0.0's fused runtime is a useful experimental source and a possible
pinned dependency, but it is not ready to become SushiUI's default execution
path. Its published data does not establish an end-to-end SushiUI speedup, its
GPU kernels are not exercised by upstream CI, and several dispatch, scaling, and
benchmark issues require local correctness gates. Integration therefore starts
with an unfused PyTorch oracle and makes fused execution an optional backend.

## Investigated upstream

The reviewed release is LyCORIS `v4.0.0`, commit
[`03270a3839102e63b48578c80e7c024036de74d7`](https://github.com/KohakuBlueleaf/LyCORIS/commit/03270a3839102e63b48578c80e7c024036de74d7),
released 2026-09-01. The release notes and source are the primary references:

- [LyCORIS 4.0.0 release](https://github.com/KohakuBlueleaf/LyCORIS/releases/tag/v4.0.0)
- [kernel coverage](https://github.com/KohakuBlueleaf/LyCORIS/blob/03270a3839102e63b48578c80e7c024036de74d7/docs/kernels/README.md)
- [kernel benchmarks](https://github.com/KohakuBlueleaf/LyCORIS/blob/03270a3839102e63b48578c80e7c024036de74d7/docs/kernels/benchmarks.md)
- [precision contract](https://github.com/KohakuBlueleaf/LyCORIS/blob/03270a3839102e63b48578c80e7c024036de74d7/docs/kernels/precision.md)
- [backend selection](https://github.com/KohakuBlueleaf/LyCORIS/blob/03270a3839102e63b48578c80e7c024036de74d7/lycoris/kernels/select.py)
- [Apache-2.0 license](https://github.com/KohakuBlueleaf/LyCORIS/blob/03270a3839102e63b48578c80e7c024036de74d7/LICENSE.md)

The source and package metadata declare Apache-2.0. There is no upstream
`NOTICE` file at the reviewed commit. As of 2026-09-02, PyPI publishes
`lycoris-lora` 3.4.0 rather than 4.0.0, so an initial dependency would need an
immutable Git commit pin. Vendoring instead requires the Apache license,
attribution, modification notices, and an entry in
`docs/legal/THIRD_PARTY_PROVENANCE.md`.

## What the fused kernels do

LyCORIS exposes merge/rebuild and bypass custom-autograd operations through a
common Triton, TileLang, per-op `torch.compile`, or eager interface. Selection is
per operation in the order Triton, TileLang, compile, eager. Supported floating
dtypes are fp16, bf16, and fp32 with fp32 accumulation; mixing fp16 and bf16 in
one operation is refused.

The relevant mechanisms are:

- LoRA bypass computes the low-rank projection and its backward without
  materializing the full delta weight.
- LoHa bypass generates Hadamard-product weight tiles without writing the full
  delta or its gradient to global memory.
- LoKr bypass evaluates the factored Kronecker transform directly; its rebuild
  path generates Kronecker tiles.
- DoRA/DoHa/DoKr share a weight-decomposition epilogue over an already merged
  weight. DoRA is therefore not a separate base algorithm in the upstream
  checkpoint: it is LoRA, LoHa, or LoKr plus `dora_scale`.

There are shape-specific fallbacks. Examples include no convolution bypass,
LoHa rank limits, LoKr factor-padding limits, and restrictions on Tucker forms.
The exact matrix remains owned by the upstream kernel coverage document; SushiUI
must query capabilities rather than copy those assumptions into architecture
code.

### Runtime hazards found in 4.0.0

These findings block default enablement:

1. The documented runtime-failure fallback is incomplete. Backend compilation
   failure can demote a choice, but a Triton or TileLang launch exception is not
   generally caught and demoted. Some functional entry points also omit rank
   constraints that the selected kernel later enforces.
2. Upstream DoRA multiplier semantics do not match AddLoRA strength. The delta is
   merged before magnitude interpolation, so multiplier zero is not a no-op.
   SushiUI requires `W_eff(s) = W_base + s * (W_adapter - W_base)` so strength
   zero restores the base exactly and negative strengths remain defined.
3. The LoHa/LoKr merge helpers appear to apply scale in both `get_weight()` and
   `get_diff_weight()`. Runtime and merge equivalence at `alpha != rank` must be
   proven against an independent oracle.
4. Upstream CI installs CPU-only Torch and skips CUDA kernel tests. Importability
   is not evidence that a kernel compiles, runs, or returns correct gradients on
   SushiUI's supported Windows/Linux toolchains.
5. The device guard accepts floating dtypes beyond the documented three. SushiUI
   must reject unsupported input rather than letting an undocumented float64 or
   other path reach a kernel.
6. The measured tuner performs a substantial first-use search and writes a user
   cache. Training needs an explicit warm-up/cache policy, concurrency-safe cache
   behavior, and progress reporting so tuning is not mistaken for a stalled first
   step.

## What the published performance data establishes

The upstream table is a synthetic RTX 4090, fp16 shape sweep. Its device-time
metric sums profiler kernel time and excludes Python/autograd dispatch. Its
forward-plus-backward wall metric is more relevant, and includes results where a
fused path is slower than eager: the reported LoHa-like bypass row is `0.91x`
and LoKr merge is `0.73x`. The release includes no end-to-end diffusion training
step, generation latency, bf16 performance table, other GPU, block-swap,
gradient-checkpointing, or SushiUI fused-optimizer measurement.

There is also a benchmark labeling defect. The script named and reported as
`lora` calls LoHa reference and fused functions, so the published `lora` ratios
do not establish a standard LoRA speedup. See the reviewed
[`scripts/bench/kernels/lora.py`](https://github.com/KohakuBlueleaf/LyCORIS/blob/03270a3839102e63b48578c80e7c024036de74d7/scripts/bench/kernels/lora.py).

Consequently, no expected whole-step speedup can currently be bounded for
SushiUI. Adapter work is only one part of a step; base Linear, attention, loss,
optimizer, transfers, and data work remain. Kernel acceptance must be based on
local end-to-end measurements rather than the release headline.

## Current SushiUI state

`ModelType` in `backend/core/model_loader.py` lists 14 generation
architectures. `ARCH_REGISTRY` in `backend/core/training/arch/__init__.py` lists
13 training architectures; MiniMax Music 3 is generation-only.

Training already shares more than the layer class.
`backend/core/training/adapters/base_adapter.py:243` defines the abstract
`BaseLoRAAdapter`, which **all 13 architecture adapters subclass**. That base
class and its module supply:

- injection with component tagging — `register_lora_layer` records each
  wrapped layer against one of the `LORA_COMPONENTS` names (`unet`,
  `text_encoder`, `text_encoder_1`, `text_encoder_2`, `vision_encoder`),
  behind the abstract `apply_lora_to_unet` / `apply_lora_to_text_encoders`;
- quantized-base detection and refusal — `count_quantized_linears`,
  `reject_quantized_base`, `warn_quantized_base_without_checkpointing`;
- branch dtype policy — `lora_branch_dtype`, and `is_lora_wrappable_linear`
  as the wrappability predicate (`Int8Linear`/`Fp8Linear` are not `nn.Linear`
  subclasses, so an `isinstance` test drops every quantized target silently);
- per-component learning-rate resolution — `resolve_component_lr`.

The shared base now owns the bodies of `setup_trainable_parameters` and
`save_checkpoint`, with resume tensor discovery delegated to each adapter leaf.
Training adapter selection has also moved into `ARCH_REGISTRY`:
`ArchHandler.lora_adapter_plan()` supplies the adapter class and
architecture-specific constructor arguments to `LoRATrainer._create_adapter`.
The registry still does not describe generation target topology, checkpoint
codecs, or generation loaders.

A **second adapter algebra already exists**, and the extracted protocol must
accommodate it from day one rather than assuming one forward:
`backend/core/training/adapters/minimax_h3_adapter.py:72` defines
`MiniMaxH3LoRALinearLayer(LoRALinearLayer)` with a different `forward`, which
casts the branch to the ACTIVATION dtype per call because MiniMax-H3's
training forward runs without `torch.autocast` (the vendored transformer owns
its own mixed-precision policy). It is imported by generation, not only by
training.

Generation is split again. SD1.5/SDXL use `LoRAManager` and Diffusers/PEFT.
Most other architectures have a custom loader in their pipeline backend and a
partially shared helper under `backend/core/models/<arch>/`. These paths do not
implement one common strength, target-count, multiple-adapter, step-range, or
atomic restore contract.

### The measured back-edge that places the shared engine

Generation already imports `LoRALinearLayer`, `is_lora_wrappable_linear` and
`lora_branch_dtype` from `core.training.adapters` in **12 modules across 11
architectures** (`acestep.py`, `flux2.py`, `minimax_h3.py`, `zimage.py` under
`core/pipeline_backends/`, plus `anima`, `ideogram4`, `krea2`, `lens`, `ltx2`,
`minimax_h3`, `minit2i`, `sensenova` `_lora.py` helpers under `core/models/`).

That import is not free. `core.training.adapters` is a subpackage of
`core.training`, whose `__init__` imports `base_trainer`, which imports from
`api` at module scope — so the whole API surface, including `api.routes`,
loads, and CUDA is initialised. **Measured in a fresh process** (repo venv
Python, cwd `backend/`, warm filesystem cache, importing
`core.training.adapters.sd15_adapter`): **8.86 s and 9.06 s on two runs, adding
5,803 modules to `sys.modules`**, with `api.routes in sys.modules` and
`torch.cuda.is_initialized()` both `True` afterwards.

That measured back-edge — generation reaching into the training package and
dragging the API and a CUDA context with it — is why the Phase 1 engine must
live in `backend/core/adapters/`, OUTSIDE `core.training`. Without the
measurement, "outside the training package" reads as an arbitrary preference.

The existing pattern these adapters follow is documented in
`backend/core/training/adapters/MODEL_ADAPTER_DESIGN.md`. This document
SUPERSEDES that file's "Base Adapter Interface", "Integration with Trainers"
and "Migration Plan" sections once Phase 1 lands, since the adapter protocol
and the trainer's selection mechanism both move. Its per-architecture notes
(SD1.5, SDXL and Z-Image injection specifics), its learning-rate resolution
section and its "What a resume writes back" section remain the current
description of behaviour and are NOT superseded.

### LoRA round-trip defects found and repaired in Phase 0

Phase 0 audited the generation-time LoRA path of every architecture that
advertises LoRA, then gated it. What it found, and what the repair was, in
sixteen commits (`10f470d5`..`2d7dc86a`):

**A trained LoRA that could not be used at all.** Five architectures could not
complete the trainer-save to fresh-generation round trip:

- Z-Image training writes flattened `lora_transformer_<flattened>` key stems
  while generation searched dotted `transformer.<dotted>` stems, so a
  SushiUI-trained Z-Image LoRA matched none of its 136 targets. Repaired in
  the loader, not on save: the flattened stem is also the trainer's in-memory
  layer identity and its resume key, so renaming on save would strand resume
  for every checkpoint already on disk.
- Krea 2 had parser, apply and restore helpers, but its generation backend
  never read `params["loras"]` — selecting a Krea 2 LoRA was a silent no-op.
- LTX-2.3 had a working training adapter and no generation loader at all. Its
  `arch_capabilities` entry declaring LoRA unimplemented produced a false
  "ignored" warning on a correct request, and is gone.
- MiniMax-H3's loader parsed only the ComfyUI key convention and documented
  dropping the SushiUI one, so a LoRA trained here matched zero targets.
- MiniT2I applied text-encoder LoRAs AFTER the prompt was already encoded and
  removed them before the next generation: such a LoRA reported a nonzero
  applied count and changed nothing.

**A trained scope that was silently dropped.** FLUX.2 saved Qwen3
text-encoder adapters that generation read and discarded; Anima wrapped
attention only, dropping the other 26 of 42 default-scope modules without a
log line; ACE-Step logged its opt-in `mlp` keys as "skipped, not an error";
Lens applied only its default scope, so a mod-scope LoRA lost its modulation
tensors. Each applied scope is now derived from the checkpoint's own keys and
enumerated by the TRAINER's own iterator, so the two sides cannot drift.

**Silent success on a LoRA that did nothing.** A missing file, a loader
exception and a zero-target application returned a successful generation on
every architecture. All thirteen now refuse before denoising under the shared
`lora_not_found` / `lora_load_failed` / `lora_incompatible` codes — the eleven
component backends in `backend/core/pipeline_backends/`, and SD1.5 and SDXL in
`LoRAManager.load_loras` (`backend/core/extensions/lora_manager.py`).

SD1.5 and SDXL were the last two (`2d7dc86a`) and were the hardest, because
diffusers gives the caller nothing to check: `load_lora_into_unet` /
`load_lora_into_text_encoder` return `None`, and after filtering the state dict
by component prefix an empty result makes the whole load a silent no-op with a
log line — which is exactly what a LoRA for another architecture hits. The
count therefore comes from a read-back of what PEFT installed inside the model:
`_count_applied_lora_targets` walks each component for a per-adapter branch
container (`lora_A`, `lora_embedding_A`) holding `adapter_name`, with the
adapter's presence in the component's `peft_config` as a weaker second witness
so an unrecognised PEFT layer class costs a count rather than causing a false
refusal. The zero-target refusal has to run before `set_adapters`, which raises
its own generic error on an empty adapter set and would otherwise mask the
precise one.

**Wrappers surviving a failure.** Several loaders removed wrappers only on the
success path, or held them as process state cleared by the NEXT request's
gate, so a sampling or decode exception carried them into the following
generation. Restore now runs in a `finally` in every entry point, guarded so a
restore failure cannot replace the real error.

**Alpha applied at the wrong scale.** Z-Image records rank/alpha only in
safetensors file metadata and the loader fell back to rank, so every
`alpha != rank` LoRA applied at the wrong scale. Precedence is now per-key
tensor, then file metadata, then rank. MiniMax-H3's metadata tier is
deliberately restricted to its native key path: ComfyUI checkpoints bake their
effective scale into `lora_B` and drop alpha, so a metadata ratio would
attenuate them twice.

**Shape-mismatched branches assigned wholesale**, then failing inside the
denoise loop or the encoder forward (Z-Image, FLUX.2). Partial application now
refuses before denoising under `lora_partial`; it never installs the compatible
subset and continues.

Two findings were structural rather than per-architecture and were deferred to
Phase 1: the original-module bookkeeping surviving a model reload, and additive
multi-LoRA stacking. Both have since landed — all thirteen architectures now sum
two adapters over one module (see Phase 1 below).

**Checkpoint classification collapsed onto `sd15`.** `classify_lora_keys`
(`backend/core/extensions/lora_manager.py`) recognised five architectures and
ended in an sd-scripts `lora_unet_` / `lora_te*` catch-all. Ten architectures
write `lora_unet_*` stems, so the catch-all claimed them: Anima, Lens,
Ideogram 4, MiniT2I, Krea 2, LTX-2.3, ACE-Step and every MiniMax-H3 LoRA
trained in this repo were reported as `sd15` with a single `BASE` block — a
confident wrong answer, not the `unknown` that would have been visible. So
were text-encoder-only FLUX.2 and MiniT2I files. `unknown` remains a
first-class result for a stem matching no signature.

Appending branches would not have fixed it, because ORDER decides: an SD1.5
U-Net stem literally contains the newer spellings as substrings
(`lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q`
contains `transformer_blocks`), so an unanchored or late-ordered DiT check
either is shadowed by the catch-all or steals SD files in return. Each
architecture is now anchored on the key prefix its own training adapter
writes and tested before the catch-all; architectures sharing a root are
separated by leaf name (`d046b894`). Z-Image also classified correctly but
lost its block list, because that branch tested only the dotted spelling while
the trainer writes the flattened one.

**Round-trip gates now exist per architecture.** Phase 0's evidence had lived
in throwaway scripts that no run would execute. Each architecture now has a
checked-in file — `backend/tests/<arch>_lora_roundtrip_cheap_test.py` for
twelve, plus `backend/tests/minimax_h3_lora_apply_cheap_test.py` — that drives
its real training adapter's injection and save, then its real generation
loader on a freshly built tree, and asserts set equality against the adapter's
own iterator rather than a non-empty count, since a partial match is as wrong
as zero and much quieter. `lora_up` initialises to zeros, so the tests
randomise it before comparing forwards; without that a round trip passes even
with the two halves transposed. Alpha is deliberately unequal to rank so a
regression to the rank fallback shows as the wrong scale, and restore is
asserted by object identity rather than tensor equality. Eight gates were
checked by reverting the behaviour they guard. Measured at `a4f1a919`: the
whole suite ran 118 passed, 5 xfailed, 15 s, 1.68 GiB peak; the five xfails
have since been fixed (`b9487812`), and the suite has not been re-measured.

**What the gates found, and why earlier code audits had not.** The gates
caught the stale-module splice still live on three architectures, each for a
different reason: ACE-Step had no ownership key at all, so unloading
re-resolved every stale module path against whichever DiT was live; SenseNova
cleared its map only after restoring and keyed its compensating guard off the
wrapped-key set being empty, which is false in exactly the dangerous ordering;
MiniMax-H3 had a correct weakref accessor that the unload path never called.
Two more architectures raised on a missing file without recording a warning,
so that refusal reached neither the response nor the image metadata
(`b9487812`).

The lesson is about the harness, not the bugs. Every throwaway script written
during the audits unloaded on the OLD model before swapping, which clears the
map and hides precisely the ordering that bites — swapping components while
wrappers are still live. Re-reading the code did not surface it, because each
implementation looks correct under the sequence the reviewer imagines. A gate
is only evidence for the sequence it actually executes, so these tests swap
first and unload second.

**Failure text carried server paths.** The per-architecture fixes had made
every message use the file's basename, but the underlying exception was still
interpolated whole: a corrupt-header error is path-free, a `PermissionError`
is not and carries the absolute path the basename was there to remove. Krea 2
and LTX-2.3 leaked by a second route (the file read sat outside any per-file
`try`, so the raw `OSError` reached the generic handler, which returns its text
as the response detail) and named the configured LoRA directories outright.
Every backend now reports the exception TYPE and the basename, keeping the full
text in the console log and traceback (`0c88406a`). This matters because a
warning is written into a PNG text chunk, returned raw in the response's
`warnings[]` (`get_warnings()` does not redact), and persisted — the existing
`redact_params_for_sharing` helper covers only the chunk
(`backend/utils/image_utils.py`). Restore-failure warnings still interpolate
their exception: those follow an attribute assignment rather than a file read,
so the text carries no path.

## Architecture feasibility

"Additive" below means LoRA, LoHa, and LoKr on the architecture's existing
Linear target set. It states implementation feasibility, not measured quality or
speed. LoCon/Conv targets are separate future scope.

| Architecture | Additive family | Initial DoRA | Required architecture-specific work |
|---|---|---|---|
| SD1.5 | yes | dense Linear | Preserve U-Net and optional CLIP MLP groups |
| SDXL | yes | dense Linear | Preserve U-Net and two text-encoder LR groups |
| Z-Image | yes | dense Linear | Key codec and generation loader repaired (Phase 0) |
| FLUX.2 | yes | dense only | Qwen targets now applied (Phase 0); gate quantized bases |
| Anima | yes | dense only | Inference scope now equals training scope (Phase 0) |
| Lens | yes | dense Linear | Retain fused-QKV path naming and frozen GPT-OSS encoder |
| Ideogram4 | yes | deferred | Dual transformer can be FP8; start with DoRA refused |
| MiniT2I | yes | dense Linear | Preserve transformer and optional FLAN-T5 scopes |
| Krea2 | yes | deferred | Generation call added (Phase 0); INT8/FP8 bases need capability gates |
| LTX-2.3 | yes | dense only | Generation loader added (Phase 0); Gemma-3 remains frozen |
| MiniMax-H3 | yes, later gate | deferred | Preserve custom QKV mapping and FP8/ConvRot dtype policy |
| ACE-Step | yes | dense only | Opt-in MLP scope now round-trips through generation (Phase 0) |
| SenseNova | yes, later gate | deferred | Preserve two MoT halves, phase eviction, and INT8/ConvRot policy |
| MiniMax Music 3 | no training | no | Keep refused until the training input contract exists |

## Shared adapter engine

Create `backend/core/adapters/`, outside the training package, so training and
generation use the same implementation. It should contain five boundaries.

### 1. Adapter specification

`AdapterSpec` is the normalized, versioned description:

- base algorithm: `lora`, `loha`, or `lokr`;
- `weight_decompose: bool` (LoRA + true is DoRA; likewise DoHa/DoKr);
- rank, alpha, factorization and algorithm-specific options;
- architecture and component scope;
- checkpoint format and schema version.

The UI may present LoRA, LoHa, LoKr, and DoRA as simple choices, while the
backend retains the correct two-axis representation. This avoids making DoRA a
mutually exclusive algorithm when upstream represents it as a weight-decompose
epilogue on three algorithms.

### 2. Adapter layer protocol

Each algebra implementation provides:

- `base_module` and base capabilities;
- `trainable_parameters()`;
- `export_tensors()` and `load_tensors()`;
- `forward()` or `forward_delta()`;
- merge support and exact strength semantics;
- an unfused reference path and optional fused backend.

The single-branch wrapper is replaced by `CompositeAdapterLayer`
(`backend/core/adapters/layers.py`, shipped and adopted by the eleven
component-loader architectures; SD1.5 and SDXL stay on diffusers/PEFT). It owns
the base once and holds multiple named branches, allowing AddLoRA
to change strength or step activation without rewrapping. Its output is the base
result plus every active additive branch. DoRA uses the full-difference
interpolation contract rather than being treated as an ordinary additive factor.

The name ends in `Layer`, not `Linear`: every offloader in
`core.memory_management.block_offloading` selects modules by
`__class__.__name__.endswith("Linear")`, so a `*Linear`-named wrapper with a
delegating `.weight` enrols the base weight twice and the paired staging swap,
applied twice, restores the outgoing block's weights. The branch contract is one
method, `forward_delta(x)` (plus `set_adapter_strength(strength)` for a branch
whose strength changes after installation), which both existing algebras
satisfy, so the composite never dispatches on a branch's class. Because the
composite is what hides the base module from `nn.Linear` selection,
`lora_wrapped_count` (`core.models.common.int8_runtime_quantize`) now counts
adapter wrapper ROOTS through `core.adapters.count_adapter_wrapper_roots`
instead of matching `LoRALinearLayer` by name; the INT8 pre-flight in
`core.vram_optimization`, the in-place converter and the Lens FP8 gate all read
that one function, so a composite holding a non-LoRA branch is refused by all
three. `backend/tests/adapter_composite_layer_cheap_test.py` is the gate.

### 3. Target topology

An `AdapterTarget` records module path, parent and attribute/index, component,
scope/block tags, input/output geometry, dtype, quantization, and merge
capability. Existing architecture target iterators should be migrated rather
than rewritten. Architecture code owns only:

- which modules are targets;
- text-encoder/component policy;
- scope names;
- special transforms such as MiniMax-H3 QKV row mapping;
- module-path to checkpoint-stem translation.

Extend each `ARCH_REGISTRY` descriptor with these hooks and a capability matrix.
This removes the adapter factory if-chain and lets a future architecture become
adapter-capable by registering topology instead of copying a loader.

### 4. Checkpoint codec registry

One detector/parser/serializer must serve scanning, details, training resume,
generation preflight, and export. Detection priority is:

1. normalized safetensors metadata;
2. a complete tensor-key signature;
3. `unknown`, which may be listed but is refused on application.

Recognize these LyCORIS tensor groups:

- LoRA/LoCon: `lora_down.weight`, `lora_up.weight`, optional
  `lora_mid.weight`, and `alpha`;
- LoHa: `hada_w1_a`, `hada_w1_b`, `hada_w2_a`, `hada_w2_b`, optional
  `hada_t1`/`hada_t2`, and `alpha`;
- LoKr: full or factored `lokr_w1*` and `lokr_w2*`, optional `lokr_t2`, and
  `alpha`;
- weight decomposition: the base algorithm's complete group plus `dora_scale`.

Support SushiUI canonical, LyCORIS/Kohya, and Diffusers/PEFT codecs. Keep
adapter algebra independent of checkpoint stems. Metadata and key signatures
that disagree are errors; partial tensor groups, shape mismatch, architecture
mismatch, and zero applied targets are also errors.

Canonical saved metadata should include at least:

```text
sushi.adapter.schema_version
sushi.adapter.algorithm
sushi.adapter.weight_decompose
sushi.adapter.format
model_type
target_scope
step
epoch
lora_rank
lora_alpha
```

Existing LyCORIS and Kohya metadata remains readable. Existing down/up-only
SushiUI checkpoints normalize to LoRA.

### 5. Atomic runtime session

`AdapterSession` resolves paths, parses every checkpoint, validates all targets,
and prepares all branches before mutating a model. It then applies the complete
set atomically and restores it in `finally`. It owns strengths, step ranges, and
component switches for every architecture. A failed or partial application must
refuse the request before denoising rather than log and continue.

## Composition with SushiUI fused paths

The systems are complementary but do not become one monolithic kernel:

- Fused optimizer groups and fused backward hooks run after adapter leaf
  gradients accumulate. They can work with custom adapter autograd if
  `trainable_parameters()` returns every LoHa/LoKr factor and DoRA magnitude
  exactly once. Group completion and gradient clearing need integration tests.
- Attention backends are downstream of q/k/v projections. They compose
  sequentially; there is no direct adapter-plus-attention kernel fusion.
- For a frozen quantized Linear, the promising path is
  `fused_base(x) + fused_adapter_bypass(x)`. LyCORIS does not recognize SushiUI's
  custom quantized classes automatically, so the engine must use a capability
  protocol rather than upstream class-name checks.
- Gradient checkpointing is compatible with pure adapter functions, but kernel
  tuning must happen before checkpointed recomputation. Any dropout requires
  reproducible RNG tests.
- Block swap requires adapter parameters to be included before offloader hook
  registration and resident on the same CUDA device at execution. Kernel-saved
  tensors must be measured because they can erode swap savings.
- LoRA-family full-model `torch.compile` is currently disabled in SushiUI.
  LyCORIS per-op compile fallback is therefore not evidence for an additional
  current compile gain.

DoRA over a weight-only quantized base is initially refused. Reconstructing the
base direction can force full dequantization or invalidate the fused base GEMM;
it needs a separately designed cache and correctness/performance gate.

## API, persistence, and frontend

Keep `training_method` as the run lifecycle discriminator:

```text
lora | relora | full_finetune | controlnet | vae_decoder
```

Add a separate canonical training configuration:

```yaml
network:
  type: lora
  adapter_algorithm: loha
  weight_decompose: false
  linear: 16
  linear_alpha: 16
  adapter_config: {}
```

Missing fields in an existing YAML normalize to LoRA without weight
decomposition. Initially, ReLoRA accepts ordinary LoRA only; other combinations
are refused until merge/reinitialize semantics and optimizer reset are tested.

Required backend plumbing includes:

- defaults in `backend/api/param_defaults.py` only;
- strict enums in `TrainingRunCreateRequest` rather than the current free-form
  string and unknown-method fallthrough;
- POST, PUT, YAML generation, `/params` restoration, preset import/export, and
  `train_runner` normalization;
- adapter-family capability data in `backend/api/arch_capabilities.py`;
- corresponding schemas, descriptions, and examples in `openapi.yaml`.

Generation retains the public field name `loras` for compatibility. Each item
adds optional `adapter_type: "auto"`; `auto` uses the checkpoint detector, while
an explicit value is only an assertion and must match the file. The internal
normalized form uses `AdapterSpec`. Existing requests without the field remain
auto-detected as ordinary LoRA when their keys are down/up.

Every multipart `loras` JSON string must be parsed through the same item model as
JSON endpoints. `GET /loras` and `GET /loras/{id}` add detected adapter type,
format, full architecture enum, and validation state. The current filename and
directory identifier remain stable.

Frontend work is concentrated in:

- `frontend/src/utils/api.ts`: adapter types and response/request shapes;
- `frontend/src/components/common/LoRASelector.tsx`: detected variant badge,
  automatic mode by default, manual assertion under advanced controls;
- `frontend/src/components/training/TrainingConfig.tsx`: algorithm and
  weight-decomposition controls, request/restore, and preset round trip.

Existing generation panels already carry the `loras` objects through queue and
loop generation. Tests must ensure the new fields survive that transport rather
than adding architecture-specific panel controls.

**Landed: the generation-side API and frontend surface.** `GET /loras` and
`GET /loras/{id}` report `adapter_type`, `adapter_algorithm`,
`weight_decompose`, `adapter_format`, `adapter_state`, `adapter_state_reason`,
`adapter_rank` and `adapter_alpha`, detected by `CodecRegistry.detect` --
the same detector the generation path uses, not a second one. Each `loras[]`
item accepts `adapter_type` (default `"auto"`, from
`param_defaults.LORA_ITEM_DEFAULTS`), parsed for JSON and multipart alike by
`api/adapter_types.py::parse_lora_items`, which every one of the thirteen
`loras` sites in `routes.py` now calls. `GET /schema/arch-capabilities` gains
`adapter_families`, built by reading `ENABLED_ADAPTER_PAIRS` and
`adapter_refusal_reason` rather than mirroring them. Gate:
`backend/tests/adapter_type_api_cheap_test.py`.

- **The admission test was the dishonest part, in both directions.**
  `_is_valid_lora_file` admitted on a key PREFIX (`has_lora_unet or
  has_lora_te`), so an sd-scripts-spelled LoHa was listed as an ordinary LoRA,
  and Z-Image's flattened `lora_transformer_*` spelling satisfied no arm at all
  and was filtered OUT -- on one of the four architectures where LoHa
  generates. The four historical arms are unchanged; a fifth admits a file the
  detector names `loha`/`lokr`.
- **The listing predicts the engine rather than judging the file, on the
  ALGEBRA axes only.** It mirrors `AdapterSession._refuse_unsupported_algebra`:
  ordinary LoRA is not validated (its rank sniff covers three spellings only)
  and an `unknown` algebra is not either, so neither can be reported `invalid`.
  `unknown` is a report, never a refusal, on the listing path. `invalid` is
  reserved for a named LyCORIS algebra whose own declaration is inconsistent,
  and carries `validate()`'s sentence.
- **`validate()`'s ARCHITECTURE axis is neutralised here, and half-neutralising
  it was a live false positive.** `from_codec(codec, architecture=None)` does
  not ignore the file's `model_type` -- it FALLS BACK to it -- so a kohya LoHa
  whose stems `classify_lora_keys` cannot place and whose metadata says
  `model_type: sdxl_base_v1-0` was reported `invalid` ("declares architecture
  ... which is not a known training architecture") while generating fine on
  every enabled architecture, with the UI rendering that as an amber line under
  a working entry. `AdapterSession` cannot hit the arm at all (it always passes
  the loaded arch, so the fallback never fires), which is exactly why a listing
  that copied the call without the loaded arch inherited a check that is not
  answerable without one. `spec.validate(known_architectures={spec.architecture})`
  passes that one check by construction; the algebra axes are untouched, and a
  Tucker file carrying the same foreign `model_type` is still `invalid`.
- **An explicit `adapter_type` is an assertion, checked at the route.** A
  mismatch answers 400 with `lora_adapter_type_mismatch` before anything loads;
  a file detected `unknown` satisfies no assertion (and `auto` still applies
  it); an unresolvable path is left to the generation path's own
  `lora_not_found`, which names the real cause.
- **Detection is header-only and cached per (path, mtime, size).** `safe_open`
  gives keys, `get_slice(k).get_shape()` gives shapes and only a scalar
  `.alpha` is ever materialized, so no tensor data is read. MEASURED on 200
  synthetic files x 200 keys, OS cache warm, fresh manager per arm: first scan
  194 -> 244 ms, and 119 -> 190 ms on an independent re-run of the same shape,
  so the first scan costs roughly 25-60% more; a subsequent `force_rescan`
  142.9 -> 8.0 ms (independently 119 -> 6.4 ms), ~20x cheaper, because the
  probe cache survives it. Quote this synthetic shape rather than a real LoRA
  directory: the ones configured on the development machine hold too few files
  to measure. The same probe now also backs `get_lora_info`, which used to open
  each file twice more; `_probe_cache` is pruned to the files the latest scan
  saw, so a training run writing checkpoints into a search path cannot grow it
  without bound.
- **Not done here, deliberately:** the SD1.5/SDXL diffusers path still has no
  `lora_incompatible` refusal of its own (the design doc's separate step); the
  selector now WARNS from the capability table instead. Training-side
  `network.adapter_algorithm` and `TrainingConfig.tsx` are untouched.

No database migration is required for the first implementation: runs persist
YAML and presets persist JSON. Derived API responses can expose normalized
adapter fields from those documents.

## Implementation sequence

### Phase 0: repair ordinary LoRA — done

Sixteen commits, `10f470d5`..`2d7dc86a`. The ordinary LoRA round trip is
repaired and gated on every architecture that advertises LoRA: the Z-Image key
codec and alpha precedence; Krea 2, LTX-2.3 and MiniMax-H3 generation wiring;
the FLUX.2 text-encoder, Anima MLP/LLM, ACE-Step MLP and Lens mod scopes; the
MiniT2I text-encoder ordering; uniform refusal, warning and `finally`-restore
behaviour, including SD1.5 and SDXL; weakref-keyed original-module bookkeeping
in all eleven pipeline backends; path-free failure text; architecture
classification anchored per adapter; and a checked-in trainer-save to
fresh-generation gate per architecture. See "LoRA round-trip defects found and
repaired in Phase 0" above for what each of those was.

Deferred to Phase 1, because each is one engine-level change rather than a
per-architecture patch: making multiple adapters additive over one module;
unifying `step_range`, component selection and per-block weights across
backends; carrying a machine-readable `code` on a refusal's error response; and
deciding whether to enumerate the `GenerationWarning.code` taxonomy in
`openapi.yaml`. Three have landed — the codes now ride on the raised exception
onto `ErrorResponse.code`, and the enum was decided AGAINST with the reasoning
recorded below. Only the per-LoRA option unification is outstanding.

### Phase 1: extract the shared engine — partly landed

**Landed.**

- **The adapter leaf mechanism lives outside the training package**
  (`dbcc3047`, `b864934e`). `backend/core/adapters/` holds the
  architecture-neutral leaf names and every importer reads them from there; the
  re-exports at the old paths are gone, and a test scans every backend module
  and fails if a leaf module is imported again. Measured in a fresh process:
  importing `core.adapters` costs 1.22 s and 1,015 modules against 9.22 s and
  5,801 for the old `core.training.adapters` route, and leaves `core.training`,
  `api` and the CUDA context untouched. A Krea 2 LoRA target check that used to
  pull `core.training`, `api.routes` and `api.param_defaults` on first call
  (3.09 s, 876 modules) now adds three modules and no measurable time.
  MiniMax-H3, LTX-2.3 and ACE-Step still reach into `core.training` for OTHER
  names on their native key paths, so they keep paying the edge until their
  codecs move. `core/models/krea2/__init__.py` remains expensive on its own
  (about 5 s, initialises CUDA) — a package-init problem, not a
  training-package edge.
- **The layer answers for its own tensors, and the shared training bodies are
  lifted** (`d82fa1d1`). `branch_tensors` is the single extension point and the
  other four tensor methods derive from it; `alpha` is deliberately not among
  them, being a spec constant the saving adapter owns. All thirteen adapters'
  `setup_trainable_parameters` and `save_checkpoint` bodies are lifted onto the
  base, with the learning-rate policy staying per-architecture as a thunk called
  only for a component that received layers. Checkpoints are byte-identical
  against a pristine HEAD tree — header, tensor table, dtypes, offsets and data
  blob deterministic and identical, and the whole file identical once metadata
  keys are sorted (safetensors randomises metadata key ORDER per process, so two
  runs of unchanged code already differ in raw bytes). Parameter groups match by
  count, order, rate and per-parameter checksum. Resume slices by the layer's own
  tensor names. Left deliberately: SD1.5, SDXL, FLUX.2 and Z-Image read the
  trainer's learning-rate attributes directly where nine others use the resolver;
  converging them changes what reaches the optimizer when a component rate is
  unset, which is a numerics decision.
- **`CompositeAdapterLayer`** (`4d6e9824`) — see "Adapter layer protocol" above
  for its contract.
- **Adoption across all thirteen architectures** (`7bc6baf3`, `326d41fc`,
  `50f66562`, `ed15cba2`, `7f727107`, `baf04fd5`, `2de43ee5`). The eleven
  component-loader architectures cover each target Linear with one composite and
  add one named branch (`"<request index>:<file basename>"`) per selected LoRA,
  so two LoRAs over the same module sum, each at its own strength, in either
  selection order; a single LoRA is bit-identical to the pre-composite loader
  (`torch.equal`) on each. SD1.5 and SDXL are additive through diffusers/PEFT
  instead and are deliberately NOT on the composite: putting them there would
  replace the production diffusers path with a reimplementation. `2de43ee5` also
  fixed the defect that made only the LAST selected SD1.5/SDXL LoRA active
  (`set_adapters` REPLACES the active set) and the block-weight loss that hid
  behind it. Per-architecture detail is in `docs/guides/MODEL_FACTS.md`.
- **Training adapter resolution through the architecture registry.** Every
  `ArchHandler` now declares its LoRA adapter class and constructor arguments;
  `LoRATrainer._create_adapter` builds that plan rather than maintaining a
  second architecture if-chain.
- **`AdapterSession` has landed across eleven generation backends.** It owns
  resolve/parse, pre-mutation accounting, atomic install/rollback, per-component
  original-module bookkeeping, and restore for Z-Image, Anima, Lens, Krea 2,
  Ideogram 4, MiniT2I, FLUX.2, SenseNova, ACE-Step, LTX-2.3, and MiniMax-H3.
  MiniT2I uses its split-session contract so one parsed file can cover the text
  encoder before prompt encoding and the transformer after staging. SD1.5/SDXL
  remain on the Diffusers/PEFT runtime rather than the component session. MiniMax
  Music 3 does not load adapters.
- **Session-level component options and step_range contract unification.**
  `AdapterSession` now uniformly enforces `apply_to_unet` and `apply_to_text_encoder`
  component kind filtering, warns `lora_no_targets` without refusal when all
  components are user-disabled, and drives dynamic branch activation over
  `step_range` [0, 1000] via `set_step()` across all session-managed backends.
- **LyCORIS adapter layer variants: LoHa, LoKr, and DoRA.**
  `core.adapters` now provides reference `LoHaLinearLayer` (Hadamard-product),
  `LoKrLinearLayer` (Kronecker-product), and `DoRALinearLayer` (weight decomposition
  with exact strength-zero identity and magnitude vector scaling). All satisfy
  the composite branch protocol and can be combined additively over shared modules.
  Their conventions were then checked against upstream v4.0.0
  (`03270a38`) and five silent, shape-compatible mismatches corrected: LoKr's
  scale is derived from WHICH operands are factored (`rank_scale`) and is
  exactly 1 for the full/full form rather than the stored `alpha`; `dora_scale`
  picks its norm axis from its own shape, so a `wd_on_out=False` `(1, in)`
  vector is no longer read as row magnitudes on a square projection (measured
  47.6% off on the delta weight, and 45-68% off on the layer output depending
  on the probe input); `factorization()` matches
  upstream for both the default and an explicit `factor`, which applies to both
  dimensions; `lokr_w1_a`/`lokr_w1_b` (`decompose_both`) is implemented; and the
  trained `scalar` of `use_scalar` layers is modelled. LoHa needed no change,
  and upstream's doubled scale in `get_diff_weight` is deliberately NOT
  reproduced: a checkpoint is trained under `forward`/`_rebuild_forward`, which
  applies it once. `backend/tests/adapter_oracle_gate_cheap_test.py` is the
  gate; its `ALGEBRAS` table now carries eleven rows.

  Three constraints on the loader and exporter that this work fixed the algebra
  for but did NOT make live, since no LyCORIS load or save path exists yet:

  - **Export must FOLD `scalar`, not emit it.** `scalar` is a training-side
    tensor. Upstream's `LohaModule`/`LokrModule.custom_state_dict` multiply it
    into the saved `hada_w1_a` / `lokr_w1` (or `lokr_w1_a`) and write no
    `scalar` key, and `load_weight_hook` forces `scalar := 1` after any load --
    so no real file carries one, and our read path is right to treat its
    absence as 1. `branch_tensors()` DOES emit it, because it is the live
    optimizer/resume view; a serializer built on it must fold and drop the key,
    or every other reader gets an adapter `1/scalar` too strong.
  - **Tucker refusal is groundwork, not an operative gate.** `hada_t1` /
    `hada_t2` / `lokr_t2` are detected by `CodecRegistry.detect` and refused by
    `AdapterSpec.validate()` and by a branch's `load_tensors`, but neither
    refusal is on a live path today: `AdapterSpec` has no production caller and
    `load_tensors` only ever sees a `LoRALinearLayer` on training resume. It
    becomes live when a LyCORIS load path lands. A real conv Tucker file is
    still refused today, by the per-architecture zero-target message.
  - **One of the four LoKr stored forms is unrepresentable here.**
    `compute_delta_weight` keys the form on `rank`/`decompose_both` while
    `scale` keys it on tensor presence, and `decompose_both` requires
    `rank > 0`, so **w1-factored + w2-full** cannot be built --- upstream
    reaches it via `decompose_both=True` with `lora_dim >= max(...)/2`, or
    `full_matrix`. Relatedly, upstream's `use_w1`/`use_w2` auto-fallbacks mean
    the same YAML config can select a DIFFERENT stored form there than here.
    Training-side only; no LoKr training path exists yet.

  One convention was deliberately aligned without a behaviour change: the LoKr
  scale consults `w1_a` before `w2_a`, as upstream's `rank_scale` does. No
  representable checkpoint can tell the orders apart (one `lora_dim` serves
  both operands, and upstream's own `make_module_from_state_dict` would fail to
  load an asymmetric pair), but a future asymmetric writer would diverge
  silently.
- **Checkpoint codec registry and foreign format normalization.**
  `core.adapters.codec` provides `CodecRegistry`, `detect_adapter_codec()`, and
  `normalize_adapter_keys()`. It identifies algorithm (`lora`, `loha`, `lokr`),
  weight decomposition (`dora`), and container format (`sushiui_canonical`,
  `lycoris_kohya`, `diffusers_peft`). Detection always runs during
  `AdapterSession` parsing; rewriting Hugging Face PEFT keys into canonical
  down/up stems is opt-in per session and off by default, because six
  architectures parse the `lora_A`/`lora_B` spelling themselves: ACE-Step
  discriminates on it directly, the others reach it inside a branch selected by
  PREFIX (`diffusion_model.` vs `lora_unet_`), so a rewrite leaves that branch
  matching nothing. Rewriting unconditionally refused five of them. Detection
  itself is advisory and guarded: it indexes shapes it has not validated, and a
  failed sniff must not replace an architecture's 400 with a 500.
- **`AdapterSpec`, `AdapterTarget` and a per-architecture capability matrix.**
  `core.adapters.spec.AdapterSpec` is the normalized two-axis description, and
  it OWNS the `sushi.adapter.*` key names: `codec.py` imports those constants
  rather than repeating the literals, and `to_metadata()` / `from_metadata()`
  are the single encode/decode pair. A block with no `sushi.adapter.*` key
  normalizes to ordinary LoRA, but a Kohya/LyCORIS one is REFUSED there rather
  than defaulted, because its algorithm lives in `ss_*` keys that entry point
  does not read; `AdapterSpec.from_codec(detect_adapter_codec(...))` is the
  entry point for an arbitrary file. `validate()` refuses an unknown algorithm
  or format, a bad or missing rank where the algebra scales by `alpha/rank`, an
  alpha with no rank, an unknown architecture and a newer schema version.
  `core.adapters.targets.AdapterTarget` is the section-3 topology record
  (path, parent and attribute/index slot, component, scope/block tags,
  geometry, base and branch dtype, quantization kind, merge capability), with
  `enumerate_adapter_targets()` beside it. Every `ArchHandler` declares an
  `AdapterCapability` carrying the feasibility table's verdicts AND, separately,
  what round-trips today: ordinary LoRA alone on all thirteen, with a factual
  reason on every other `(algorithm, weight_decompose)` pair, so no LoHa/LoKr/
  DoRA run can start. Nothing consumes the matrix yet and it is deliberately
  absent from `api/arch_capabilities.py` and every HTTP response, per the
  shipped-boundary rule.
  `backend/tests/adapter_spec_targets_cheap_test.py` is the gate.

**Not landed.**

- Foreign LoHa/LoKr on the four architectures still refusing them: Tier 3 with
  its own gate (MiniMax-H3, SenseNova), ACE-Step's diffusers/PEFT branch, and
  SD1.5/SDXL, which never reach `AdapterSession` at all.
- The architecture-registry hooks that would carry generation TOPOLOGY -- target
  discovery, component policy, scope names, checkpoint-stem translation -- and
  the migration of the thirteen existing target iterators onto `AdapterTarget`.
- Checkpoint codec parsing and training adapter integration for foreign LoHa/LoKr/DoRA.
- The generation `AdapterSpec` API, codec-derived adapter metadata, and the
  corresponding frontend selector/training controls.

**Findings that must be fixed once in the engine, never patched
per-architecture:**

- **Original-module bookkeeping surviving a model reload.** Each backend kept
  a map from module key to the pre-LoRA `nn.Module`, and the map outlived the
  model it described; the next unload then spliced the PREVIOUS model's Linear
  modules into the new tree. FLUX.2 did it across all 252 wrapped targets, and
  its text-encoder restore runs every generation, so it fired on the first
  generation after a switch; SenseNova carried 588 stale entries. Eight
  independent implementations of the same bookkeeping produced the same silent
  defect. The uniform fix is a `weakref.ref`-keyed reset, now present in all
  eleven pipeline backends (`acestep`, `anima`, `flux2`, `ideogram4`, `krea2`,
  `lens`, `ltx2`, `minimax_h3`, `minit2i`, `sensenova`, `zimage`), consulted on
  both the load and the unload path and before the empty-config exit, with
  restore discarding each key as it lands so a restore that raises part way
  leaves only what it still owes. `id()` is unsafe here because a freed object's id is REUSABLE,
  and a reload allocating at the dead model's address is exactly the case the
  key must survive. An engine-level session owns this once.
- **Refusal warnings were write-only — fixed.** Every refusal path calls
  `add_warning` before raising, but the routes read `get_warnings()` only on the
  success path, so `lora_incompatible`, `lora_not_found` and the rest reached the
  client as a generic 400/500 with the taxonomy discarded. `APIError` now carries
  `code` and `warnings`, `create_error_response` emits both, and every generation
  route copies them onto the error it answers with (`error_context` /
  `attach_error_context` in `backend/api/generation_status.py`).

  The code rides on the RAISED EXCEPTION, not on a read-back of the warning
  bucket, and the raised TYPE is unchanged: `api.error_handlers.with_error_code`
  tags the plain `FileNotFoundError`/`RuntimeError` the component backends raise,
  the `APIError` subclasses take a `code=` argument, and `AdapterRefusal`
  (`core.adapters.session`) carries one as a class attribute. Inference from the
  bucket was rejected on evidence: ACE-Step and LTX-2.3 deliberately raise a
  shorter sentence than they warn, so message matching reports no code for them,
  and last-warning-wins would pin a bystander advisory code on an unrelated
  later failure. The HTTP split has now been removed without changing raised types:
  `error_handlers.is_lora_refusal_code` names the refusal subset, a wrapped
  `GenerationError` selects 400 from that code, and `attach_error_context`
  normalizes an already-raised `APIError` the same way. Missing, unreadable,
  incompatible, partial, and architecture-specific compatibility refusals
  therefore answer 400 on every generation route; untagged exceptions remain
  500 and warning-only degradations remain successful.
  `backend/tests/refusal_error_code_cheap_test.py` is the gate.
- **Additive multi-LoRA stacking was blocked repo-wide by the layer class —
  fixed.** `LoRALinearLayer.__init__` reads `original_module.in_features` /
  `out_features` into LOCALS and never exposes them on `self`, so the wrapper
  cannot wrap a wrapper. That is why every architecture was first-wins or a
  refusal rather than summing branches. `CompositeAdapterLayer` is the fix (a
  per-architecture re-wrap was not), and every loader has migrated:
  `lora_stacking_unsupported` is emitted nowhere in `backend/core`,
  `backend/api` or the frontend.
- **A legacy FP8 generation path can cast the adapter's own branches.** Four
  architectures deep-copy a component and cast every `isinstance(m, nn.Linear)`
  weight to FP8, which over a wrapped tree includes each branch's `lora_down` /
  `lora_up`: Z-Image and FLUX.2 through `vram_optimization._quantize_transformer`
  (and `_quantize_text_encoder` on FLUX.2), Anima and Lens through
  `vram_optimization._anima_quantize_fp8`. Only two gates exist, and neither
  covers all of it — `_lens_quantization_with_lora` drops the Lens TRANSFORMER's
  quantization while wrappers are live, and `_flux2_te_quantization_with_lora`
  drops the FLUX.2 TEXT ENCODER's; both warn `quantization_fallback` and let the
  LoRA win. Z-Image's transformer, Anima's transformer and FLUX.2's transformer
  have no equivalent. Both gates read `lora_wrapped_count` →
  `count_adapter_wrapper_roots`, which counts a composite as ONE root, so
  adoption did not change what they see. The in-place runtime INT8 conversion is
  gated everywhere it runs by the same function.
- **The four per-LoRA options are each honoured by a different subset of
  architectures.** `LoRAConfig` (`backend/core/extensions/lora_manager.py`)
  parses all four for every request, but:
  - `step_range` is honoured only on the SD1.5/SDXL diffusers path, through
    `LoRAConfig.is_active_at_step` driven by the step callback that
    `load_loras_for_generation` arms when any entry is non-default. No other
    backend reads it.
  - `unet_layer_weights` is honoured on SD1.5/SDXL
    (`LoRAManager._apply_layer_weights`) and on FLUX.2's transformer, where the
    per-block weight multiplies the request strength before the branch is
    built; FLUX.2's text-encoder half deliberately uses the plain strength.
  - `apply_to_unet` and `apply_to_text_encoder` are honoured by FLUX.2 (both:
    they select which component is walked, and a file whose tensors all belong
    to a disabled component is a `lora_no_targets` warning rather than a
    refusal) and `apply_to_unet` alone by LTX-2.3 (a disabled request applies
    nothing and warns `ltx2_lora_unet_disabled`, since an LTX-2.3 LoRA only
    targets the video DiT). **They are NOT honoured on the SD1.5/SDXL path**:
    `LoRAConfig` reads them and `load_loras` prints them, and nothing else
    consults them. Krea 2 treats `apply_to_text_encoder` as vacuously honoured
    because it never touches the text encoder.
  - Everything else is ignored. Krea 2 warns for `apply_to_unet=false`,
    `unet_layer_weights` and a non-default `step_range`; LTX-2.3 warns for the
    latter two (`ltx2_lora_step_range_ignored`,
    `ltx2_lora_layer_weights_ignored`); the remaining backends ignore silently.

  Honouring them uniformly belongs in the engine's target topology and
  session, not in eleven loaders.
- **`GenerationWarning.code` stays a free-form string — decided, not
  deferred.** The taxonomy in use is `lora_not_found`, `lora_load_failed`,
  `lora_incompatible`, `lora_partial`, plus architecture-specific codes
  (`minimax_h3_lora_variant_mismatch`, `ltx2_lora_h2d_disabled`,
  `quantization_fallback`, and others). There are **135 distinct codes** passed
  as `code=` under `backend/api` and `backend/core` today, and each new feature
  adds more, so an `enum` would make almost every commit a spec change and would
  license a strict generated client to REJECT an otherwise valid response over a
  warning it has not heard of — the opposite of what an advisory field is for.
  `VideoChainIssue.code` is already documented as a free-form "stable
  machine-readable identifier" for the same reason. The spec instead names the
  stable cross-architecture subset a client may branch on in the field
  description, and `ErrorResponse.code` points at the same taxonomy. What is
  checkable is the mechanism, not the vocabulary: the gate asserts the codes
  reach the client, not that the set is closed.

### Phase 2: LoHa and LoKr reference paths

- Implement pure PyTorch fp32 oracles and mixed-dtype unfused paths.
- Load, resume, save, and generate using LyCORIS-compatible tensor groups.
- Enable dense targets first, then additive branches over INT8/FP8/W4A8 bases.
- Gate MiniMax-H3 and SenseNova separately because their ConvRot training
  forward and activation dtype policy dominate more of the step.

**Landed: the tensor-group engine, with no production caller.**
`backend/core/adapters/groups.py` owns the step between a checkpoint's keys and
a branch: `ADAPTER_SUFFIXES` (22 spellings into 18 canonical names, covering the
LyCORIS set, the PEFT `lora_A`/`lora_B` names and `.dora_scale`),
`split_adapter_suffix`, `TensorGroup`, `group_adapter_tensors(tensors,
stem_of)`, `split_group_on_out_rows` and `build_adapter_branch`. Alongside it,
`LoRALinearLayer` / `LoHaLinearLayer` / `LoKrLinearLayer` gained `from_tensors`.
**Migrated: all eleven checkpoint-key parsers now run on it.** Every
architecture parses suffixes through `split_adapter_suffix`, groups through
`group_adapter_tensors(tensors, stem_of)` (its own prefix/flattening logic
unchanged, passed in as `stem_of`), and counts declared branches with
`declared_groups` -- the complete groups PLUS the incomplete ones whose algebra
is recognised, so a checkpoint truncated mid-write declares the half pair it
cannot apply and `applied < declared_branches` refuses it. All eleven do that;
the session default (which nothing uses now that Z-Image passes its own) counts
complete groups only, because with no `stem_of` it cannot tell a foreign half
key from a truncated one of its own. Nine builders are untouched -- they still
read `weights["down"]`, which the legacy aliases answer, and each grouper hands
them down/up groups only. Two changed: FLUX.2's now reads the groups
`prepare_file` parsed instead of re-reading raw keys (a spelling its counter
took and its builder did not would declare N and apply 0), and Z-Image's
`_zimage_lora_branch` delegates to a `TensorGroup` probe. Two exceptions, both deliberate: ACE-Step's
diffusers branch keeps its regexes (they bake `lora_A`/`lora_B` into the match,
so no non-pair key can reach a grouper there) and MiniMax-H3's `_split_qkv` /
`_normalise_comfy` are untouched (`split_group_on_out_rows` is phase 8) --
its counter still expands a fused `qkv_proj` stem to 3. The one widening: an
architecture now also accepts the alternate spellings of the shared table under
its own prefixes, so a key it used to drop (and refuse the file over) may now
parse. The capability matrix still refused every LoHa/LoKr/DoRA pair on all
thirteen AT THIS COMMIT (four rows opened one commit later, below), and
ordinary LoRA is byte-unchanged. Gates:
`backend/tests/adapter_tensor_group_cheap_test.py`, 22 LyCORIS rows (11
architectures x LoHa/LoKr) and 11 truncated-file rows in
`adapter_key_normalization_gate_cheap_test.py`, plus new rows in
`adapter_lycoris_variants_cheap_test.py`, `adapter_oracle_gate_cheap_test.py`
and `adapter_composite_layer_cheap_test.py`.

- **`TensorGroup` answers to the legacy aliases on purpose.** `group["down"]` is
  `group["lora_down.weight"]` and `"down" in group` is true, while iteration
  yields canonical names only. The eleven component-loader architectures each
  hand-write this grouping and four of them spell the drop identically
  (`if "down" in v and "up" in v`), so the aliases are what makes their
  migration additive instead of a simultaneous rewrite of eleven branch builders
  — the change shape that cost five architectures a day of broken LoRA loading
  in `22f21078`. **Removal condition: delete them once no branch builder reads
  `weights["down"]`.**
- **`partial` is computed and returned, never raised on.** Every architecture
  drops incomplete groups silently today; making that a refusal is a separate,
  evidence-gated change and not a side effect of the migration.
- **The alpha drop, fixed.** `branch_tensors()["alpha"]` is a freshly built
  tensor for LoHa/LoKr, so the `weight.data.copy_(value)` in `load_tensors`
  mutated a throwaway and the layer's `alpha` — hence its `scale` — never moved.
  The fix is `spec_constants()` / `load_spec_constant()`: a constant is
  ASSIGNED, not copied into. MEASURED: a checkpoint saved at `alpha=8, rank=4`
  loaded into a layer built at `alpha == rank` applied a delta exactly **0.5x**
  the correct one — right shapes, wrong image. `LoRALinearLayer` is unaffected
  (`spec_constants()` is empty, so its load path is the old loop unchanged).
- **Geometry comes from the tensors, which is required rather than tidier.**
  `LoKrLinearLayer.__init__` derives its split from
  `factorization(out_features, factor)` and **no LyCORIS file stores `factor`**,
  so a foreign LoKr written with a different one allocates the wrong
  `lokr_w1`/`lokr_w2` shapes and `copy_` raises. `from_tensors` reads
  `out_l`/`out_k`/`in_m`/`in_n` off the tensors and passes them through a new
  `factors=` argument; LoHa takes its rank from `hada_w1_a.shape[1]`. It also
  skips the kaiming init of factors it is about to overwrite, which keeps a load
  from advancing the GLOBAL rng once per wrapped target.
- **A layer built from a file may not carry `scalar`.** It initialises to
  `zeros(())` and no real file has the key (upstream folds it into
  `hada_w1_a`/`lokr_w1` at save and forces `scalar := 1` at load), so such a
  layer would multiply the whole delta by zero. `from_tensors` refuses
  `use_scalar=True` and ignores a stored `scalar` key, matching upstream's
  reader.
- **`split_group_on_out_rows` refuses a LoKr split it cannot represent.** The
  row slice is exact for `lora` (slice `lora_up`) and `loha` (slice
  `hada_w1_a`/`hada_w2_a`), the `_b` factors being shared. For `lokr`,
  `kron(w1, w2)` puts row `i*K + k` at `w1[i]` times `w2[k]`, so a contiguous
  piece is another Kronecker product **under the parent's own `(out_l, out_k)` /
  `(in_m, in_n)` split** only when it covers whole `i` blocks, i.e. `n` divides
  `w1.shape[0]`. The qualifier is load-bearing: every matrix is a degenerate
  Kronecker product of a 1x1 with itself, which is a different factorization and
  a dense delta, not a slice of this adapter. MEASURED on a 12x12 base (L=3,
  K=4), worst piece against the parent's (3, 4) column split: **0.31** at n=2
  and **0.27** at n=4 — not roundoff. The refusal is also deliberately
  conservative: when `inner` divides `w2.shape[0]` every piece lands inside one
  `i` block and IS representable (measured 2.1e-08 at n=6), and is refused
  anyway. Both facts have pinning tests; implementing the second arm is optional
  future work, never a correctness fix.
- **`build_adapter_branch` returns `SHAPE_MISMATCH`, never raises.** One
  malformed target in a foreign file is a module to skip, as it already is in
  the eleven loaders. Every read of the group's tensors is therefore inside the
  try, and the caught set includes `IndexError` and `ZeroDivisionError`: a 1-D
  `hada_w1_a`, a 0-D `lora_down.weight`, a two-element `.alpha` and a rank-0
  `lora_down.weight` each escaped an earlier draft, which would have turned
  "skip it, warn `lora_partial`" into a 500 out of a generation request.
  `AttributeError` is deliberately not caught — a missing attribute is an engine
  bug, not a file defect. A rank-0 `lora`/`loha` group is refused for the same
  reason `AdapterSpec.validate` refuses it (`RANK_REQUIRED`): its delta is
  exactly zero. LoKr's full/full form is legitimately rank 0 and scales by 1.
- **`TensorGroup.to_spec()` drops the alpha of a rank-0 LoKr.** Upstream
  overrides `alpha = lora_dim` for the full/full form and writes that into the
  file, the layer's `scale` ignores it, and keeping it would make `validate()`
  refuse every legitimate full/full LoKr as "an alpha with no rank". DECIDED,
  with the residual gap recorded: a serializer reconstructing a LyCORIS `.alpha`
  from a spec alone would lose the stored value. It is not parked in `options`
  today, because `options` is written into `sushi.adapter.options` on save and
  that would be a durable format commitment for a value nothing reads; the live
  layer carries it either way via `_alpha_from_tensors`. Revisit when a
  spec-driven serializer exists.

**Phase 2 checklist for the migration itself:**

- `core/training/lora_trainer.py` (`_load_checkpoint`, the resume slice) both
  snapshots and rolls back by iterating `branch_tensors()` and copying, so it
  carries the same throwaway-`alpha` defect on its ROLLBACK path. Unreachable
  today because only `LoRALinearLayer` is ever constructed for training; it
  becomes live the moment a training adapter constructs a LyCORIS layer.
- **ACE-Step did not migrate mechanically, and half of it did not migrate at
  all.** `core/pipeline_backends/acestep.py` seeds each DIFFUSERS group with
  `{"source_prefix": ..., "down": None, "up": None, "alpha": None}` — a
  non-tensor key plus `None` placeholders that `TensorGroup` cannot hold, and
  whose `"down" in v` test is inverted relative to this engine's (always true
  there, presence-based here). That branch is UNCHANGED and still needs its own
  step; its regexes bake `(lora_A|lora_B)` into the key match, so no non-pair
  key can reach a grouper there in the meantime. The sd-scripts branch and the
  declared-branch counter did migrate.

**Landed: the capability gate, enforcing on the generation path.**
`backend/core/adapters/capability.py` owns `ENABLED_ADAPTER_PAIRS`, one explicit
row per architecture in the `ARCH_REGISTRY` spelling, and it is the ONLY place a
family is enabled. `AdapterSession` gained `architecture=` (all eleven
generation backends pass it) and refuses an unenabled algebra in `_parse`.
Nothing was enabled by that step: all thirteen rows were still
`{("lora", False)}`. Four opened next; see "Landed: LoHa and LoKr generate"
below.

- **The table is in `core/adapters/`, and `base_arch` reads it, because the
  dependency only runs one way.** `core.adapters` may not import
  `core.training` (measured back-edge: 8.9 s, 5801 modules and a CUDA context
  in a fresh process; `backend/tests/adapter_layering_test.py` re-measures the
  clean arm every run at ~1.3 s / ~1020 modules). The reverse is fine and
  already pervasive, so `declare_adapter_capability` READS
  `declared_pairs(arch)` instead of hardcoding `frozenset({("lora", False)})`.
  The alternative considered and rejected was mirroring the enabled set into
  `core/adapters/` with a pinned-equality test, the way `KNOWN_ARCHITECTURES`
  mirrors `ARCH_REGISTRY` — a second copy that can drift, for a fact that
  needs no copy.
- **A flip is one table row, and the refusal cannot be dropped separately.**
  Enabling a family used to mean adding the pair to `supported` AND skipping it
  in the refusal loop, with `AdapterCapability.__post_init__` raising if only
  the second half was done. The loop is now driven by the table (`if pair in
  supported: continue`), so both halves move together by construction. The
  `__post_init__` check is kept: it still guards hand-built matrices
  (`NO_ADAPTER_CAPABILITY` and the gates).
- **The refusal fires before the model is mutated, and the gate asserts that,
  not merely that something raised.** `_refuse_unsupported_algebra` runs in
  `_parse`, i.e. before `AdapterFile` is constructed and therefore before
  `_count_declared_branches`, `prepare`, `_plan_file` and `_install`. The test
  installs a recording `build_branch` and asserts `visited == []` plus an
  unchanged slot map; a sibling test shows `visited == list(TARGETS)` when the
  gate deliberately does not fire, which is what gives the first one its
  discriminating power. It is an `AdapterIncompatible`, so `code =
  "lora_incompatible"` answers HTTP 400 through `error_handlers`
  (`is_lora_refusal_code`) rather than 500 — driven end to end over a real ASGI
  round trip in `refusal_error_code_cheap_test.py`.
- **The message is asserted, not just the code, because `validate()`'s
  malformed-file arms answer the same 400.** A user must not be told their file
  is broken when the truth is "LoKr is not implemented yet". Two fixes were
  needed for that to hold: `AdapterSpec.from_codec` now drops an alpha that
  comes with no rank, exactly as `TensorGroup.to_spec()` already did (a
  full/full LoKr has no rank and carries upstream's `lora_dim` alpha, so
  `validate()` refused every legitimate one as "an alpha with no rank"), and
  the codec's rank sniff was widened below. Gates cover both LoKr forms.
- **The codec's rank sniff was wrong for ten real spellings and blind to LoKr.**
  It read `tensor.shape[1]` for anything that was not `.lora_down.weight`, but
  PEFT's `lora_A` is `[rank, in]` exactly like `lora_down` — only the LyCORIS
  `*_a` factors are `[out, rank]`. MEASURED on a rank-4 / 128-wide fixture, the
  interchange or PEFT spelling of **acestep, anima, ideogram4, lens, ltx2,
  minimax_h3, sensenova, sd15 and sdxl** each reported **128** (in_features) as
  the rank; factored LoKr (`lokr_w2_a`) reported `None`. `_sniff_rank` now
  mirrors `TensorGroup.rank`'s per-algebra axes and skips a tensor too small for
  its axis, so a `lora_bias=True` PEFT export's 1-D `.lora_A.bias` no longer
  raises `IndexError` out of detection. This was live, not latent, the moment
  `_parse` started calling `validate()`.
- **`("unknown", *)` is deliberately NOT gated.** `_canonicalize` fabricates an
  `unknown` codec whenever detection raises, and a valid `lora_bias=True` PEFT
  export used to do exactly that; refusing on `unknown` would refuse valid files
  on whichever architectures the sniff misses — the `3271627b` / `22f21078`
  failure shape. An unrecognised algebra is left to the architecture's own
  zero-target refusal. Ordinary LoRA is not validated at all for the same
  reason; widening either is a separate, evidence-gated step.
- **SD1.5 and SDXL never reach `AdapterSession`, so the gate does not cover
  them.** They load through diffusers (`core/extensions/lora_manager.py`).
  A kohya LyCORIS LoHa carries `lora_unet_*` keys, so `has_lora_unet` accepts it
  as a valid LoRA and it is LISTED IN THE UI; it then reaches diffusers' loader,
  which does not understand `hada_*`. There is no `lora_incompatible` refusal on
  that path — just whatever diffusers raises. This is the one place "a refusal
  arrives before the model is mutated" does not hold, and it needs its own step.
- **`AdapterCapability.require()` still has no training caller.** Training-side
  enforcement is not part of this step; the capability check reached generation
  only.

**Landed: LoHa and LoKr generate, on four architectures.**
Z-Image, Krea 2, MiniT2I and LTX-2.3 carry `("loha", False)` and
`("lokr", False)` in `ENABLED_ADAPTER_PAIRS`. Their four branch builders route
through `build_adapter_branch`, so the ALGEBRA is the checkpoint's rather than
the builder's, and each keeps its own alpha precedence and branch dtype
(MiniT2I's is the base weight's dtype, not `lora_branch_dtype`, and is now the
named `minit2i_lora.branch_dtype`). The gate is
`backend/tests/adapter_lycoris_roundtrip_cheap_test.py`: per architecture, a
synthetic LoHa and LoKr file covering exactly its own target iterator's set,
the installed delta against `reference.py`'s fp32 oracle in fp32 AND bf16 at
`alpha = 3 x rank`, `lora_partial` on a bent group, `lora_incompatible` on a
truncated one, and identity restore with the component swap BEFORE the unload;
plus a mixed LoRA + LoHa stacking row, the block-swap rows below, and the
negative rows that drive the seven unflipped sessions' refusals. Nothing else
moved: no decomposed pair is
enabled anywhere, no training adapter constructs a LyCORIS layer,
`AdapterCapability.require()` still has no caller, and `GET /loras`, the
request schema and the frontend selector are untouched.

**What that leaves reachable, measured rather than assumed.** No response
carries an adapter type and the selector has no variant control, but the
scanner's admission test is a key-PREFIX test that says nothing about the
algebra: `LoRAManager._is_valid_lora_file` admits any file carrying
`lora_unet_*` or `lora_te_*`. Driving the real scanner over synthetic LoHa
files in each architecture's own stem spelling:

| stem | LoHa | plain LoRA |
|---|---|---|
| `lora_unet_transformer_blocks_0_attn1_to_q` | listed, `ltx2` | listed, `ltx2` |
| `lora_unet_transformer_blocks__0__attn__to_q` | listed, `krea2` | listed, `krea2` |
| `lora_unet_double_blocks__0__attn__to_q` | listed, `sd15` | listed, `sd15` |
| `lora_transformer_layers_0_attn_to_q` | filtered out | listed, `zimage` |

So a LoHa or LoKr in the sd-scripts spelling is LISTED AND SELECTABLE on
LTX-2.3, Krea 2 and MiniT2I, tagged with the architecture the file's stems name
(MiniT2I's `sd15` is the pre-existing `classify_lora_keys` result and is the
same for its ordinary LoRAs). Z-Image's flattened `lora_transformer_*` spelling
satisfies no arm of the admission test once the down/up keys are gone, so a
Z-Image LyCORIS file is filtered out of the list and is reachable by path only.
What a user still cannot do anywhere is ASSERT an adapter type or see one
reported: `adapter_type` on a request item, and the detected type on
`GET /loras`, are a later phase.

- **`from_tensors` ADOPTS the file's tensors; resume still copies.** A branch
  built from a checkpoint assigns `param.data = value.to(...)` (the new
  `adopt_tensors`), which is what all thirteen generation loaders have always
  done. Copying into a freshly allocated parameter instead is not equivalent:
  the fresh buffer is 64-byte aligned where a safetensors tensor is not, which
  selects a different BLAS kernel and moved an ordinary LoRA delta by 1 ULP
  against the pre-composite reference the per-architecture gates pin with
  `torch.equal`. `load_tensors` keeps copying, because on training resume the
  parameter is already held by an optimizer.
- **The per-key `.alpha` of a LoHa/LoKr file does NOT arrive through
  `_alpha_from_tensors`.** It arrives through `spec_constants()` /
  `load_spec_constant` during the load, so deleting the tensor tier from
  `_alpha_from_tensors` changes nothing for those two algebras and changes
  ordinary LoRA only. MEASURED by reverting that tier: only the mixed
  LoRA + LoHa stacking rows failed. Anyone "unifying" the two alpha paths must
  keep both.
- **A LoHa/LoKr branch is INVISIBLE to the block-swap offloader, and the
  combination is now gated.** Every offloader in
  `core.memory_management.block_offloading` selects by
  `__class__.__name__.endswith("Linear")`, which reaches a LoRA branch's
  `lora_down`/`lora_up` (both `nn.Linear`) and reaches nothing inside a
  `LoHaLinearLayer` or `LoKrLinearLayer`, whose factors are bare
  `nn.Parameter`s. What that costs depends on WHEN the branch is installed
  relative to `prepare_block_devices`, which does `blocks[i].to(device)` (all
  tensors) and only then returns the swapped blocks' LINEAR weights to the
  host. That ordering is a property of each backend's generate function, which
  no session can observe, so it is DECLARED:
  `capability.BLOCK_SWAP_ADAPTER_ORDER`.

  - `AFTER_SPLIT` — **LTX-2.3** (offloader is persistent state on
    `Ltx2BlockLoopWrapper`) and **MiniT2I** (`_minit2i_stage_transformer` runs
    one line before `_load_lora_minit2i`): the branch is built on its base's
    CURRENT device, which is the host for a swapped-out block, and nothing ever
    moves it. **Refused** as `lora_blockswap_unsupported` (a 400 through
    `is_lora_refusal_code`) before a single target is walked, rather than
    raising a device mismatch mid-denoise.
  - `BEFORE_SPLIT` — **Z-Image**: the factors are swept to the device and never
    returned, so the numbers are right and only the block-swap saving is
    smaller. **Advised**, not refused, as `lora_blockswap_not_offloaded`.
  - `NO_BLOCK_SWAP` — **Krea 2** builds no generation-time offloader.

  One mechanism, three questions, and the third is asked of the OBJECT.
  `BLOCK_SWAP_ADAPTER_ORDER` says what the combination costs here;
  `AdapterComponent.block_swap_active` — declared only by the `AFTER_SPLIT`
  backends, since only they have a live offloader to probe at install time —
  says whether one is running now; and
  `layers.branch_survives_block_swap(branch)` says whether THIS branch can ride
  with its block, by asking whether every tensor it owns is the `weight` of an
  `nn.Linear` child. `AdapterSession` refuses when all three hold, immediately
  after planning (which mutates nothing) and before any install;
  `warn_unoffloaded_branches()` applies the same predicate to what is installed
  and emits the advisory from the `BEFORE_SPLIT` offloader build site, the
  first moment the combination is knowable there (at install time the attached
  offloader is the PREVIOUS generation's).

  **The predicate is on the built branch, never on the file's detected
  algorithm, and that is not a stylistic preference.** `CodecRegistry.detect`
  gives metadata priority over keys, so a file of pure `hada_*` tensors
  carrying `ss_network_module: networks.lora` (or
  `sushi.adapter.algorithm: lora`) DETECTS as ordinary LoRA — a label test lets
  it through and the per-group builder then installs `LoHaLinearLayer`s anyway,
  which is exactly the crash the gate exists to prevent, paid for after full
  staging. A label test also mis-refuses in the other direction: every file
  that sniffs `unknown` (a valid `lora_bias=True` PEFT export, or any file
  whose detection raised) would be told "unknown adapters cannot be applied
  while block swap is active", contradicting `_refuse_unsupported_algebra`'s
  deliberate carve-out two functions above. Both directions have gate rows.
  **Ordinary LoRA is exempt by construction** — its branch is two `nn.Linear`s
  — and that is gated too, so LoRA + block swap keeps working unchanged. The
  base module is excluded from the question: its weight is the block's own, and
  a Linear `bias` is not moved by that walk either, so requiring it would
  refuse every LoRA over a biased base. A MiniT2I transformer-pass refusal
  cannot strand its text-encoder half: `_minit2i_cleanup`, in the generate
  function's outer `finally`, unloads both.

  Two consequences of the gate. `ltx2_lora.swappable_block_weight_footprints`
  no longer changes when a LyCORIS adapter covers only some blocks — so
  LTX-2.3's `h2d_only` partial-coverage fallback could not see one — but the
  refusal makes that UNREACHABLE, so it is recorded and left rather than fixed.
  And the underlying engine decision is still UNRESOLVED and deliberately not
  bolted on here: teaching the offloader to carry a branch's bare parameters is
  a feature with its own VRAM measurement and its own gate, and it is what
  would let `AFTER_SPLIT` become `BEFORE_SPLIT` and the refusal go away.
- **The legacy FP8 quantizers cannot reach a LyCORIS branch either**, for the
  same naming reason: `_quantize_transformer` / `_anima_quantize_fp8` cast
  every `isinstance(m, nn.Linear)` weight, which over a wrapped tree includes a
  LoRA branch's factors and excludes a LoHa/LoKr layer's. Better behaviour than
  LoRA gets, but by accident of naming rather than by a gate — do not rely on
  it.
**Landed: LoHa and LoKr generate, on five more architectures.**
Anima, Lens, Ideogram 4, FLUX.2 and ACE-Step carry `("loha", False)` and
`("lokr", False)` in `ENABLED_ADAPTER_PAIRS`, on the same five pieces of
evidence per row and in the same file. Each builder routes through
`build_adapter_branch` and keeps its OWN alpha precedence and branch dtype;
three of the five needed a named `branch_dtype()` because
`core.adapters.lora_branch_dtype` has no bias tier and would send every biased
fp8 target to bfloat16 (`anima_lora.branch_dtype` additionally consults a
declared `compute_dtype` first). FLUX.2 and ACE-Step keep
`lora_branch_dtype`. `NOT_ENABLED` in the gate shrank to the two Tier-3 rows
and the sibling that makes it discriminating grew to nine.

- **ACE-Step is enabled for its sd-scripts codec ALONE, and its other branch
  refuses rather than mis-applies.** `_acestep_prepare_lora_file` classifies a
  file as `sdscripts` (a `lora_unet_decoder_layers_` prefix) or `diffusers` (a
  `.lora_A.`/`.lora_B.` key) and raises `lora_incompatible`
  ("unrecognized key format") for neither. A LyCORIS file in the diffusers
  spelling carries no pair key, so it is neither, and is refused before any
  target is walked — not silently applied at zero targets, and not with the
  capability gate's "not enabled" text, which would be a lie now that the row
  IS enabled. Gate row: `test_acestep_refuses_a_lycoris_file_in_the_diffusers_spelling`.
- **The block-swap classification is the ORDERING in each generate function,
  read rather than assumed.** `AFTER_SPLIT` (refuse): Anima
  (`_anima_stage_transformer` 837 → `_load_lora_anima` 845, and 1126/1134,
  1422/1430), Lens (813/817, 1009/1013, 1214/1218) and Ideogram 4
  (1038/1052, 1227/1237, 1424/1434) each stage — which builds the offloader and
  calls `prepare_block_devices_before_forward` — a handful of lines before the
  LoRA load, on all three generate entry points. `BEFORE_SPLIT` (advise):
  FLUX.2 loads its LoRAs in stage 1 (848, 2415, 3266) and builds
  `create_flux_block_offloader` in stage 3 of the SAME function (1085, 2685,
  3563), so its factors are swept to the device and never returned; the
  advisory is called from all three offloader sites. `NO_BLOCK_SWAP`:
  ACE-Step has no `blocks_to_swap` path in its backend at all.
- **Ideogram 4's probe is not per-component.**
  `_ideogram4_stage_transformers` builds an offloader for BOTH halves or
  neither, so one `block_swap_active` reads `_ideogram4_offloaders` (a list of
  `(component name, offloader)`) and both components declare it. Its two
  transformers carry identical module paths and are told apart by the key
  namespace (`lora_unet_` vs `lora_uncond_`), which is why the branch builder
  keys on `request.component`.
- **FLUX.2's two components keep two different strength rules.**
  `unet_layer_weights` multiplies the request strength for a transformer
  target; the Qwen3 text-encoder half deliberately takes the plain strength.
  Routing the builder through the engine did not change that, because the
  strength is folded by `add_branch(strength=)` after the branch is built.
  `test_flux2_covers_both_components_at_their_own_strength_rules` reads it back
  off the composite.
- **Anima's builder had no shape check at all**, like Krea 2's before
  `aaefb6f4`: a wrong-shape group was assigned wholesale and failed inside the
  denoise loop. It now refuses `lora_partial` before anything is installed.
- **Lens's fused QKV needs no row split.** `img_qkv` / `txt_qkv` are ordinary
  `nn.Linear` targets, so one factor group covers the whole fused stem;
  `split_group_on_out_rows` is MiniMax-H3's problem, not Lens's. The GPT-OSS
  text encoder stays frozen and is not a component.
- **ACE-Step can report `applied > declared`, and always could.**
  `_acestep_count_declared_branches` counts only stems under
  `lora_unet_decoder_layers_`, while `_acestep_lora_slots` also yields the
  lyric-encoder targets the diffusers codec reaches. A hand-written file naming
  lyric stems in the sd-scripts spelling applies more branches than it
  declares. `_account` only refuses on `applied < declared`, so this is silent
  — pre-existing, unreachable from anything this repo saves (the trainer's
  scope is decoder-only), and recorded rather than changed.

### Phase 3: dense DoRA

- Start with SD1.5, SDXL, Z-Image, Lens, and dense MiniT2I targets.
- Enforce strength-zero identity and runtime/merge equivalence.
- Include magnitude tensors in optimizer, fused-hook, save, and resume census.
- Refuse weight-only quantized bases until a separate design is measured.

### Phase 4: experimental fused backend

- Prefer a pinned upstream dependency if 4.0.0 is published with fixes; otherwise
  vendor only the reviewed operations with complete provenance.
- Add an executed per-device/dtype/shape probe against the independent oracle.
- On launch failure, latch the backend off for the process and use the reference
  path on later calls; never switch the base mathematical function mid-training.
- Warm supported shapes before the first measured/training step.
- Start with LoHa/LoKr Linear bypass and the combined quantized-base path.
- Keep standard LoRA and DoRA unfused until their local measurements and strength
  contracts pass.

## Acceptance matrix

Correctness gates:

- forward output, input gradient, and every parameter gradient against the fp32
  oracle for fp32/fp16/bf16;
- strength `0`, `1`, fractional, greater-than-one, and negative;
- runtime versus merge/unmerge equivalence, including `alpha != rank`;
- gradient checkpointing on/off and deterministic dropout recomputation;
- block swap and fused optimizer/backward parameter census;
- dense, INT8, FP8, and W4A8 capability/refusal cases;
- multiple adapters, order independence for additive branches, and exception-safe
  restoration;
- malformed/partial keys, metadata conflict, wrong architecture, wrong shape,
  and zero target all refused;
- training save/resume and fresh generation round trip for every supported
  architecture.

API/frontend gates:

- POST to stored YAML to GET params to PUT round trip;
- preset export/import and legacy missing-field normalization;
- multipart and JSON item validation parity;
- AddLoRA selector, queue, and loop generation preserve detected/asserted type;
- OpenAPI matches all backend enums and defaults.

Performance gates use the same model, batch, bucket, checkpointing, block-swap,
optimizer, and attention settings. Compare current LoRA, unfused variant, fused
variant, and fused quantized base plus fused adapter. Record warmup separately;
measure median/p95 forward, forward+backward, optimizer-inclusive whole step,
generation latency, peak allocated/reserved VRAM, and host memory. A backend is
auto-selected only for measured device/dtype/shape regions where whole-workload
speed improves without violating numerical or memory limits. Kernel-only device
time is diagnostic, not an acceptance result.

## Shipped-boundary rule

Until the phases above land, capability responses and UI labels must continue to
describe only ordinary LoRA where it actually round-trips. A design verdict is
not a supported feature, and an upstream benchmark is not a SushiUI performance
claim.
