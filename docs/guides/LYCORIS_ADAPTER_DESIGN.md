# LyCORIS adapter integration design

Status: investigation and implementation plan. Shipped so far: the additive
LyCORIS algebras (LoHa, LoKr) LOAD AND GENERATE on all eleven architectures
that build an `AdapterSession` — Z-Image, Krea 2, MiniT2I, LTX-2.3, Anima,
Lens, Ideogram 4, FLUX.2, ACE-Step (sd-scripts codec only), MiniMax-H3 and
SenseNova. SD1.5 and SDXL load through diffusers and still take ordinary LoRA
only. Generation by file path. They are also TRAINABLE, on the nine of those
eleven whose training row is open (`network.adapter_algorithm: loha | lokr`,
`training_method: lora`, `blocks_to_swap: 0`); MiniMax-H3 and SenseNova load one
without training one. Dense DoRA (`weight_decompose: true` with
`adapter_algorithm: lora`) now LOADS, GENERATES AND TRAINS on Z-Image, Lens and
MiniT2I; DoHa and DoKr are refused everywhere, and DoRA is refused on the other
ten architectures — SD1.5/SDXL because diffusers drops `dora_scale`, the rest
because their base can be weight-only quantized or their row has no round trip.
Everything else here is plan.

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
| SD1.5 | yes | refused: diffusers drops `dora_scale` | Preserve U-Net and optional CLIP MLP groups |
| SDXL | yes | refused: diffusers drops `dora_scale` | Preserve U-Net and two text-encoder LR groups |
| Z-Image | yes | dense Linear (SHIPPED, phase 3) | Key codec and generation loader repaired (Phase 0) |
| FLUX.2 | yes | dense only | Qwen targets now applied (Phase 0); gate quantized bases |
| Anima | yes | dense only | Inference scope now equals training scope (Phase 0) |
| Lens | yes | dense Linear (SHIPPED, phase 3) | Retain fused-QKV path naming and frozen GPT-OSS encoder |
| Ideogram4 | yes | deferred | Dual transformer can be FP8; start with DoRA refused |
| MiniT2I | yes | dense Linear (SHIPPED, phase 3) | Preserve transformer and optional FLAN-T5 scopes |
| Krea2 | yes | deferred | Generation call added (Phase 0); INT8/FP8 bases need capability gates |
| LTX-2.3 | yes | dense only | Generation loader added (Phase 0); Gemma-3 remains frozen |
| MiniMax-H3 | yes | deferred | Fused-QKV row split landed (phase 2); FP8/ConvRot dtype policy preserved |
| ACE-Step | yes | dense only | Opt-in MLP scope now round-trips through generation (Phase 0) |
| SenseNova | yes | deferred | Two MoT halves, phase eviction and INT8 policy preserved (phase 2) |
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

- Foreign LoHa/LoKr on the two paths still refusing them: ACE-Step's
  diffusers/PEFT branch, and SD1.5/SDXL, which never reach `AdapterSession`
  at all.
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
  (Not separable for MiniMax-H3 and SenseNova, which have no dense
  configuration; their flips are the quantized-base half for their targets --
  see the Landed block below.)
- Gate MiniMax-H3 and SenseNova separately because their ConvRot training
  forward and activation dtype policy dominate more of the step. (Done for
  generation; their TRAINING rows are still ordinary LoRA.)

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

**Landed: LoHa and LoKr generate on the last two, MiniMax-H3 and SenseNova.**
Every architecture that builds an `AdapterSession` now carries `("loha", False)`
and `("lokr", False)`. Both route their branch builders through
`build_adapter_branch` and keep their own alpha precedence and branch dtype
(SenseNova gains a named `sensenova_lora.branch_dtype`, the bias-first chain its
`Int8Linear` targets need since their weight is not floating point at all;
MiniMax-H3 keeps `lora_branch_dtype`, whose no-bias default is right for its
mixed fp8/fp32 tree). No decomposed pair moved, and the training axis
(`TRAINABLE_ADAPTER_PAIRS`) stays ordinary LoRA for both.

- **MiniMax-H3's fused QKV is the actual work, and it is TWO split paths, not
  one.** `split_group_on_out_rows` is wired into `_normalise_comfy` for LoHa and
  LoKr only; ordinary LoRA keeps `_split_qkv`'s compact and shared-down paths
  byte for byte, because the compact one slices `lora_down`'s RANK COLUMNS as
  well as `lora_up`'s rows and the engine function deliberately does not. The
  LyCORIS row split keeps the rank columns, so a piece's own `alpha/rank` is the
  fused stem's and no scale override is needed downstream — which is why the
  `scale_rank` the branch carries is only ever a LoRA correction.
- **The three pieces are given their OWN storage.** `split_group_on_out_rows`
  shares the non-sliced factors by reference and `from_tensors` ADOPTS rather
  than copies (that is what keeps ordinary LoRA bit-identical), so three
  independently strengthened branches would sit on one buffer. Strength alone is
  safe — it is folded into `scale`/`strength`, never into a tensor — but a merge,
  an export, a training resume's `copy_` or a quantizer writing in place would
  reach all three, and `_split_qkv` already clones `lora_down` for exactly this
  reason. `_own_shared_tensors` clones what the split shared, at the cost of two
  extra copies of the `_b`/`w2` factors per fused stem. Gate:
  `test_minimax_h3_qkv_pieces_do_not_share_one_storage`.
- **A LoKr that straddles a block is refused BY NAME.** `w1` rows not divisible
  by three is not a corrupt file, so the message says so: it names the
  `kron(w1, w2)` row map, the actual row count, and "the file is well formed".
  The same conditional at two pieces governs `_swap_fc1_halves`, generalized
  from `lora_up` to any factor carrying the out axis — `hada_w1_a`/`hada_w2_a`
  unconditionally, a LoKr's `w1` only when its rows are even. Both refusals are
  `lora_incompatible` 400s, and both gates assert the wording is neither
  "truncated or corrupt" nor "not enabled".
- **A native LoHa was refused with the right verdict and the wrong sentence.**
  `_normalise_native` tested `"down" not in weights or "up" not in weights`, so a
  COMPLETE LyCORIS group landed in `incomplete` and was reported as a file
  "truncated or corrupt". It now separates on `group_adapter_tensors`' own
  complete/partial split, and the incomplete message names a factor group rather
  than a down/up pair.
- **MiniMax-H3's `prepare_file` ValueErrors now answer 400.** Mixed key
  conventions, an unmappable stem, a truncated group and the two new split
  refusals all mean "this file cannot be applied here", and reached the client as
  an untagged 500 with the sentence in the detail. They are wrapped as
  `lora_incompatible`, which is what makes the split refusals' wording reachable
  at all.
- **Neither architecture was missing a shape check** — the defect Krea 2, Anima,
  Lens and Ideogram 4 had. MiniMax-H3 compared in/out features and the rank
  agreement; SenseNova compared all three. Both now get the same checks from
  `from_tensors`, plus the rank-0 and LoCon-`mid` refusals they did not have.
- **Only MiniMax-H3's LoRA branch needs the per-call activation cast.**
  `MiniMaxH3LoRALinearLayer` exists because this architecture's forward runs
  without `torch.autocast` and the stock LoRA layer's `F.linear` would raise on
  an fp32 master against a bf16 activation. `LoHaLinearLayer` and
  `LoKrLinearLayer` already do `compute_delta_weight().to(x.dtype)` in
  `forward_delta`, so `layer_cls` is passed for the LoRA algebra alone.
- **Block swap, read from the generate functions.** `BEFORE_SPLIT` (advise):
  MiniMax-H3 has ONE generate function — `_generate_minimax_h3`, which all five
  entry points call — and it loads LoRAs at 3184, three lines before
  `_ensure_minimax_h3_swap_and_offload` at 3187 builds the per-generation
  `TransformerBlockOffloader`. There is no keep-models-hot resident branch to
  check: this architecture is excluded from `keep_models_hot` outright, and its
  offloader is per-generation for the same reason (the DiT leaves the GPU before
  every decode). The advisory is called from its single
  `prepare_block_devices_before_forward()` site. `NO_BLOCK_SWAP`: SenseNova's
  `blocks_to_swap` is inert — its backend never reads it — and its MoT phase
  evictor is not a `TransformerBlockOffloader`: `select_mot_weight_modules`
  classifies by `_owns_persistent_tensor` and a `_mot_gen` path substring, and
  `move_non_gen_to_device` / `on_phase` move a module's OWN `_parameters`, so a
  LoHa's bare factors travel with the half they sit under. The
  `.lora_down`/`.lora_up` marker in `_is_adapter` is training-only
  (`require_exact_symmetry`), which generation never sets.
- **SenseNova's opt-in key canonicalization carries LyCORIS unchanged.**
  `normalize_keys` fires only for `FORMAT_PEFT`, which a LyCORIS file reaches
  through the `base_model.model.` prefix rather than through `lora_A`/`lora_B`;
  the suffix rewrites cannot match a `hada_*`/`lokr_*` key, and stripping the
  prefix is exactly what a verbatim-module-path parser needs. Gate:
  `test_sensenova_canonicalizes_a_peft_prefixed_lycoris_file`. The recorded
  collision gap is unchanged and not algebra-specific: a file carrying both
  `base_model.model.X` and a bare `X` still loses one tensor silently.
- **These two flips ARE the quantized-base half of Phase 2, for these targets,
  and the declaration now says so.** The phase order above puts dense targets
  before "additive branches over INT8/FP8/W4A8", and every architecture declared
  `quantized_base_additive_family=False` with `QUANTIZED_ADDITIVE_PENDING`. That
  is not decidable per-phase for these two: SenseNova has **no dense
  configuration at all** — all 294 targets per MoT half are `Int8Linear` — and
  MiniMax-H3's whole DiT block stack is `Fp8Linear`, with only the fp32 AdaLN
  and head projections dense. Enabling LoHa/LoKr on them *is* the quantized-base
  case; declaring it pending alongside would have been a false statement in
  code, not a conservative one. So the flag is `True` for exactly those two,
  carrying `QUANTIZED_ADDITIVE_SHIPPED`, which states what is claimed (the
  branches build and forward correctly over the real quantized layer) and what
  is not (no quality or speed measurement; DoRA still refused, since it needs the
  base weight's direction and norm). The other nine keep `False`: a dense
  checkpoint exists for each, so the flag there is about a configuration nothing
  has evidence for. `test_a_lycoris_branch_installs_over_a_quantized_base` is
  the evidence — the same file over `Int8Linear` and `Fp8Linear`, with and
  without a bias, reaching every target and matching the fp32 oracle. It also
  covers the one thing that changed for these two beyond the algebra: the shape
  check moved to `_base_geometry`, which RAISES on a base exposing neither
  `in_features` nor `out_features`, and a raise there is caught as
  `SHAPE_MISMATCH` — a silently skipped target, then `lora_partial` for the
  file. Both quantized classes set both attributes, and reverting
  `_is_lora_target` to bare `nn.Linear` fails six of the eight rows.
- **`additive_gated` now reads on the TRAINING axis.** The design-doc table's
  third additive value, "yes, later gate", was true of the generation axis until
  these flips and is false there now. It is exactly true on the training axis:
  `TRAINABLE_ADAPTER_PAIRS` leaves both ordinary-LoRA-only, and their gate is
  their own (MiniMax-H3's fused-QKV split has no save side; SenseNova's two MoT
  halves and phase eviction) rather than the general Phase 2 step. Repurposing
  beat dropping because the two-table split had just made that fact expressible,
  and dropping the flag would have discarded it one commit later. The
  per-architecture `additive_reason` those two carried was dead text — with all
  three non-decomposed pairs supported, `declare_adapter_capability` never
  reaches it — so each moved its "gate of their own" sentence into
  `training_reason`, where it renders, and took the shared `PHASE2_PENDING` for
  the unreachable slot like every other enabled row. The pinning test asserts
  the flag both by name and derived from the two axes, so it cannot stay behind
  when either moves.
- **`NOT_ENABLED` is now empty, and the boundary test is not vacuous.** It
  asserts the two things still contingent: every architecture that builds an
  `AdapterSession` is a row in the gate AND carries both families, and the only
  capability rows without a session are SD1.5 and SDXL, still ordinary LoRA
  because they load through diffusers. The discriminating negative moved with
  it — a real session told it is loading for SD1.5 must still refuse, which is
  what keeps the "an enabled architecture does not refuse" sibling from passing
  on a dead check. `refusal_error_code_cheap_test` drives that same session over
  a real ASGI round trip for the same reason.


**Landed: LoHa and LoKr TRAIN, on nine architectures.**
`TRAINABLE_ADAPTER_PAIRS` is a SECOND table beside `ENABLED_ADAPTER_PAIRS` in
`core/adapters/capability.py`, keyed the same way, and
`AdapterCapability.require(algorithm, weight_decompose, axis)` takes the axis as
a MANDATORY argument -- reading the wrong row is the failure the split exists to
prevent, and a generation flip must not open training by omission. The table is
asserted to be a subset of the generation one at import: a checkpoint no loader
accepts is not a trained adapter. zimage, krea2, minit2i, ltx2, anima, lens,
ideogram4, flux2 and acestep carry `("loha", False)` and `("lokr", False)`;
SD1.5/SDXL train through the same adapters but load through diffusers, and
MiniMax-H3 and SenseNova generate a LyCORIS file without training one.

`BaseLoRAAdapter.build_branch()` is the construction seam: all 25 sites across
the thirteen adapters go through it (acestep 1, anima 1, flux2 8, ideogram4 1,
krea2 1, lens 1, ltx2 1, minimax_h3 1, minit2i 2, sd15 2, sdxl 3, sensenova 1,
zimage 2), and it returns
`core.adapters.layers.new_adapter_branch(...)` for the algebra the run asked
for. No per-architecture adapter needed anything else -- MiniMax-H3 declares its
per-call-casting subclass as `LORA_LAYER_CLS` instead of naming it at the call
site, and SDXL's two text-encoder sites pass `dtype=torch.float32` explicitly
because they always did (routing them through `self.lora_dtype` would change
what reaches the optimizer on a bf16 run, which is a numerics decision, not a
migration). Ordinary LoRA is byte-unchanged: `TrainingAdapterSpec.metadata()`
is EMPTY for it, so no `sushi.adapter.*` key appears in the files every
architecture already writes, and all thirteen per-architecture LoRA gates pass
untouched.

Gates: `backend/tests/adapter_lycoris_training_roundtrip_cheap_test.py` (9
architectures x LoHa/LoKr; the trainer builds the algebra, saves, the REAL
generation loader reads it back on a fresh stub, and the installed branch's
`forward_delta` is `torch.equal` to the trained layer's on EVERY wrapped
target, plus metadata and a full resume) and
`backend/tests/adapter_training_algebra_cheap_test.py` (the
architecture-independent half). The stub trees come from each architecture's
own LoRA gate by import rather than by copy.

- **The optimizer census holds on both fused paths, measured rather than
  asserted from a count.** `trainable_parameters()` derives from
  `branch_tensors()` and dedupes by identity, so a LoHa hands over 4 factors and
  a LoKr 3 (full `w1` plus a factored `w2`) per target, each exactly once, and
  the set is checked to EQUAL the layer's own `requires_grad` parameters -- a
  factor the census misses trains nothing while the loss falls normally. Under
  the real `FusedOptimizerGroups`, `parameter_optimizer_map` has one entry per
  factor, `step_incomplete_groups()` returns `[]` (every group completed inside
  the backward), every `.grad` is cleared and every factor moved. Under the real
  `BaseTrainer._setup_fused_backward_pass` with Adafactor, `step_param` is
  called exactly once per factor.
- **Both recorded traps were real, and both bit.** The exporter: `scalar` is
  now folded into `hada_w1_a` / `lokr_w1` / `lokr_w1_a` and the key dropped
  (`layers.fold_scalar_for_export`), because upstream folds at save and forces
  `scalar := 1` at load. Training a `use_scalar` layer is refused outright on
  top of that -- `from_tensors` cannot rebuild one, so the file would not
  resume. The rollback: `lora_trainer.load_checkpoint`'s failure path
  `copy_`-ed into the throwaway `alpha` that `branch_tensors()` rebuilds per
  call, restoring nothing; it now routes a spec constant through
  `load_spec_constant`, and a gate drives a mid-way failure and reads the
  alphas back.
- **Block swap is refused, not silently degraded.** No offloader carries a
  branch whose factors are bare parameters (they select `*Linear` by class
  name), and what that costs a TRAINING step -- residency, and which device a
  factor is on when its fused hook fires -- is UNMEASURED. `blocks_to_swap > 0`
  with a LyCORIS algebra is therefore refused by name, and the UI collapses the
  choice to LoRA while block swap is on. Ordinary LoRA is unaffected.

  **Every refusal is raised from the CONFIG, before the model loads.**
  `refuse_untrainable_algebra` (`training/adapters/base_adapter.py`) is the one
  implementation; `train_runner._assert_adapter_algebra_contract` calls it from
  the process preflight and `lora_trainer.require_trainable_algebra` is the
  backstop for a caller that skipped it. Two checks were late in the first
  draft and are not any more: `blocks_to_swap` lives in `train_config`, which
  the preflight was not passed, and an `adapter_config` key was validated per
  layer -- so both loaded the whole checkpoint and then died. The option check
  runs for `adapter_algorithm: lora` too, where nothing else would ever read
  the key: a stale `{"factor": 8}` left behind by switching algorithm is a
  refusal, not a silent no-op.
- **ReLoRA takes the ordinary branch only**, refused in
  `train_runner._assert_adapter_algebra_contract` before the model loads:
  merge/reinitialize and optimizer reset are not defined for a Hadamard or
  Kronecker factorization. `weight_decompose: true` is accepted as a field and
  refused as a value at three layers (`TrainingAdapterSpec`,
  `new_adapter_branch`, the run contract).
- **Two traps left for whoever opens the next row.**
  `sensenova_adapter.py` filters already-wrapped targets with
  `isinstance(target[3], LoRALinearLayer)`, which does not recognise a LoHa --
  harmless while SenseNova's training row is ordinary-LoRA-only, and it fails
  loudly rather than double-wrapping, but it is wrong the moment that row
  opens. And the ReLoRA trainer construction in `train_runner` never passes
  `adapter_algorithm`, so even a YAML that bypassed the preflight would build
  an ordinary LoRA -- defense in depth by accident, not by design.
- **Config surface**, in the mandated order: `adapter_algorithm`,
  `weight_decompose` and `adapter_config` in `TRAINING_DEFAULTS`; the schema and
  descriptions in `openapi.yaml`; `TrainingRunCreateRequest` (whose
  `training_method` is now a strict enum rather than a free-form string with an
  unknown-method fallthrough to full fine-tuning); the `network` block of
  `generate_lora_config`; `_YAML_FIELD_LOCATIONS` for the `/params` round trip;
  and `TrainingConfig.tsx`. `adapter_config` is `Optional` for one reason: a
  YAML written before the key existed restores as `None` through `/params`, and
  every reader treats `None` as `{}`. `GET /schema/arch-capabilities` gains
  `adapter_families[arch].trainable` / `untrainable`, and the UI's algorithm
  select is built from THAT rather than from `supported` -- the two lists differ
  by two architectures today.
- **The prose for a closed training row lives in `core/adapters/capability.py`
  (`TRAINING_REFUSAL_REASONS`), not on the `ArchHandler`.**
  `api/arch_capabilities.py` may not import the trainer stack (its
  `TRAINING_DECLARED_ARCHS` comment says why), so a sentence declared on the
  handler cannot reach a client: MiniMax-H3's "the fused-QKV row split has no
  save side yet" and SenseNova's "the two MoT halves, phase eviction and the
  INT8/ConvRot policy" rendered as the generic text in every HTTP response and
  in the run's own refusal. `declare_adapter_capability` now READS that table
  the way it already reads `declared_pairs`, so the handler and the payload
  cannot word the same refusal differently.

### Phase 3: dense DoRA — landed for `("lora", True)` on three architectures

- Start with SD1.5, SDXL, Z-Image, Lens, and dense MiniT2I targets.
- Enforce strength-zero identity and runtime/merge equivalence.
- Include magnitude tensors in optimizer, fused-hook, save, and resume census.
- Refuse weight-only quantized bases until a separate design is measured.

**Landed: dense DoRA generates AND trains on Z-Image, Lens and MiniT2I.**
`("lora", True)` is in both `ENABLED_ADAPTER_PAIRS` and
`TRAINABLE_ADAPTER_PAIRS` for those three, and in neither for the other ten.
`new_adapter_branch(weight_decompose=True)` wraps the algebra's own layer in
`DoRALinearLayer`, `build_adapter_branch` already did the load-time half, and
`BaseLoRAAdapter.build_branch` is the one construction seam, so no
per-architecture adapter changed. Gates: the `dora` rows of
`adapter_lycoris_roundtrip_cheap_test.py` (target-set equality, the installed
delta against the fp32 oracle in fp32 and bf16, the magnitude-stripped
comparison, strength-zero identity, and the eight architectures that must still
refuse), the `+wd` rows of `adapter_lycoris_training_roundtrip_cheap_test.py`
(trainer save → real generation loader → `torch.equal` per target, plus
metadata and resume), and the decomposed rows of
`adapter_training_algebra_cheap_test.py` and `adapter_oracle_gate_cheap_test.py`.

- **The SD1.5/SDXL diffusers path refuses a weight-decomposed file itself.**
  This flip WIDENED a gap it did not create: until it landed nothing in the
  repo could produce a DoRA, and now three architectures write one into the
  same directory the SD1.5/SDXL selector lists from — so "train a DoRA on
  Z-Image, switch model, pick the file" reached a successful, numerically wrong
  generation in three clicks. `LoRAManager._refuse_weight_decomposed` refuses it
  as `lora_incompatible` before `pipeline.load_lora_weights` is called, reading
  the SHARED header probe (`_probe_lora_file` → `detect_adapter_fields`, cached
  on path/mtime/size) rather than a second sniffer; `CodecRegistry.detect` ORs
  `weight_decompose` from the key presence, so a mislabelling metadata block
  cannot slip past, and a detection failure reports `False`, keeping "unknown is
  a report, never a refusal".

  MEASURED, driving the real `load_loras` with the refusal removed: a DoRA
  applied to **40 of 40** U-Net targets, returned normally, installed **no**
  `lora_magnitude_vector`, and left weights `torch.equal` to the same file with
  `dora_scale` stripped. That is the silent wrong answer, end to end.

  **LoHa and LoKr are deliberately NOT refused there**, on the same probe: the
  Kohya converter raises `ValueError` ("keys have not been correctly renamed")
  on their unrenamed `hada_*` / `lokr_*` keys, on BOTH mixins, which
  `load_loras` already turns into a `lora_load_failed` 400. Loud is enough, and
  the broader SD1.5/SDXL adapter step stays deferred. Gates:
  `test_{sd15,sdxl}_a_dora_file_refuses_before_diffusers_ever_sees_it` (asserts
  the loader was never reached),
  `..._loha_and_lokr_are_already_refused_by_diffusers_itself`, and
  `..._an_ordinary_lora_is_byte_identical_with_the_gate_neutralised` — an A/B
  against the same load with the new check stubbed out, every PEFT parameter
  compared with `torch.equal`, because `load_loras` is the shipped path for two
  architectures and this series has already had to undo one shared-load-path
  change (`22f21078`).
- **SD1.5 and SDXL are refused for TRAINING and GENERATION, on measurement
  rather than on the "no session" rule.** `StableDiffusion(XL)LoraLoaderMixin.lora_state_dict` (diffusers
  0.38.0, `loaders/lora_pipeline.py`) DROPS every `dora_scale` key with a
  `logger.warning` and nothing else, BEFORE
  `_convert_non_diffusers_lora_to_diffusers` — whose Kohya `dora_scale` →
  `lora_magnitude_vector` branch is therefore dead on this path. A DoRA
  selected for SD1.5/SDXL would apply as an ordinary LoRA at the wrong numbers
  and report success. That is the sentence `DECOMPOSE_REFUSAL_REASONS` carries
  for both, and it is why the training axis could not be opened either: the
  import-time subset check refuses a pair that is trainable and not loadable.
- **Only `("lora", True)`, deliberately.** The engine builds DoHa and DoKr
  identically (`new_adapter_branch` wraps whichever layer the algebra names,
  and the oracle gate has covered all three since Phase 1), so what keeps them
  shut is the capability table plus the absence of a per-architecture round
  trip — which is the enablement rule this series has run on. The refusal says
  exactly that (`PHASE3_DECOMPOSED_PENDING`) rather than claiming decomposition
  is unimplemented, and `test_a_flipped_architecture_takes_dora_and_still_refuses_doha_and_dokr`
  pins the wording. One latent trap for whoever opens them:
  `split_group_on_out_rows` refuses a weight-decomposed group outright, so a
  fused-QKV DoHa is unrepresentable on MiniMax-H3 — irrelevant while that row
  is closed for other reasons, and a silent zero-target file the moment it is not.
- **The quantized-base refusal is enforced at three layers, on the OBJECT.**
  `layers.weight_decompose_refusal(base)` is the one predicate and keys on the
  weight's DTYPE, not on the quantized Linear classes, because the legacy fp8
  pass leaves an ordinary `nn.Linear` holding a float8 weight and a class test
  would miss it. Generation:
  `AdapterSession._refuse_decomposed_over_quantized_base` runs after planning
  (which mutates nothing) and before any install, asking each BUILT branch —
  the same reason `_refuse_stranded_branches` does, since detection gives
  metadata priority over keys and a `dora_scale` file labelled
  `networks.lora` passes a label test. Training: `new_adapter_branch` raises,
  and `refuse_untrainable_algebra` refuses `fp8_base_dtype` from the run's
  CONFIG, before the model loads — that setting quantizes the frozen
  transformer in `prepare_models_for_training`, i.e. before injection, so the
  layer-level refusal would only fire with the whole checkpoint resident.
- **A fourth layer, for the hazard no install-time check can see.** Z-Image
  loads its LoRAs at line 700 of its generate function and reaches
  `_quantize_transformer` at 812, which deep-copies the tree and casts every
  `nn.Linear` weight — including the DoRA wrapper's own base, the weight its
  epilogue divides by. `_refuse_fp8_over_decomposed_adapter` (in
  `vram_optimization`, so `_quantize_transformer`, `_anima_quantize_fp8` and
  every caller are covered at once) drops the quantization and warns
  `quantization_fallback`, the same precedence
  `_lens_quantization_with_lora` already applies. Keyed on the DECOMPOSED
  family alone, so LoRA/LoHa/LoKr quantization behaviour is unchanged — gated
  both ways. MEASURED on the gate's fixture: casting the base to
  `float8_e4m3fn` moves the DoRA delta by **1.3% relative**, which is a quality
  change with no error.
- **The magnitude is in every census, and each half is gated separately.**
  `trainable_parameters()` derives from `branch_tensors()`, which for a
  decomposed branch is the inner algebra's tensors plus `dora_scale`, deduped
  by identity — so a DoRA hands over 3 parameters per target, a DoHa 5 and a
  DoKr 4, each exactly once, checked to EQUAL the layer's own `requires_grad`
  set. Under the real `FusedOptimizerGroups` every factor gets one entry in
  `parameter_optimizer_map`, `step_incomplete_groups()` is empty and every
  factor moves; under the real fused backward pass with Adafactor `step_param`
  fires once per factor. `export_state_dict` writes `<stem>.dora_scale` and
  `LoRATrainer.load_checkpoint` iterates `branch_tensors()`, so a checkpoint
  MISSING the magnitude is refused by name rather than silently restarting it
  at the base's row norms — the specific failure mode, gated by
  `test_a_decomposed_resume_moves_the_magnitude_and_refuses_without_it`.
  `DoRALinearLayer.export_tensors` delegates to the inner branch so a `scalar`
  is still folded away; the inherited default would have emitted it bare.
- **`dora_scale`'s dtype is RESOLVED: it takes the BRANCH's dtype, not the
  base's.** It was initialised in the base's dtype, so an fp16 run trained a
  magnitude with no fp32 master while its factors kept one. `DoRALinearLayer`
  now takes `dtype=`, defaulting to the inner branch's `lora_dtype`;
  `new_adapter_branch` passes the run's, `build_adapter_branch` passes the
  loader's. On ten of the eleven the loader's IS the base weight's own dtype, so
  no generation load moved. **Lens is the exception, and it is a real one**: its
  `branch_dtype` prefers the BIAS (`lens_lora.py`, so a biased fp8 target does
  not fall to the bfloat16 default), so a dense Lens base carrying a bf16 weight
  and an fp32 bias now gets an fp32 magnitude where HEAD gave bf16. No shipped
  Lens checkpoint produces that pairing and the new dtype is the intended one —
  the magnitude belongs with the factors it rides with — but the coincidence is
  not by construction and should not be stated as if it were.
- **The block-swap answer is right rather than accidentally right.**
  `branch_survives_block_swap` returns False for a DoRA over an ordinary LoRA,
  and for a DIFFERENT reason than for a LoHa: the two factors ARE `nn.Linear`
  weights and would ride with their block, and `dora_scale` is the one bare
  parameter left behind. The gate asserts exactly that tensor is the stranded
  one. So Lens and MiniT2I (`AFTER_SPLIT`) refuse `lora_blockswap_unsupported`
  naming "DoRA" from the built class, Z-Image (`BEFORE_SPLIT`) advises, and the
  training-side refusal names `dora_scale` instead of "a lora branch's factors".
- **The magnitude-axis blind spot is half closed, with a third
  implementation.** `layers.py` and `reference.py` both read a `(out, 1)`
  `dora_scale` as per-output-row magnitudes and `reference.py` discloses that
  it SHARES that reading, so their agreement proved nothing about the
  convention. PEFT's `DoraLinearLayer` is installed in this venv, was written
  by neither, norms along `dim=1` and holds one magnitude per output row — and
  diffusers maps a Kohya `dora_scale` straight onto its `lora_magnitude_vector`.
  MEASURED agreement on the forward: **4.1e-7**; a reversed row order is
  **1.04** off. The `(1, in)` column form has no such witness (PEFT does not
  implement it) and remains two mirrors of one reading.
  **FORWARD ONLY, and the difference is a real one:** PEFT DETACHES the weight
  norm from the graph (DoRA paper §4.3, a memory optimization) where this repo
  takes the exact gradient of the stated function, so the two BACKWARDS differ
  by construction. Recorded, not resolved — switching is a numerics decision
  with its own measurement.
- **The decomposition-axis prose moved off the `ArchHandler`**, exactly as the
  training prose did in Phase 2 and for the same reason:
  `api/arch_capabilities.py` may not import the trainer stack, so a `dora_reason`
  declared on a handler could not reach a client, and after this flip the
  handler's refusal map and `adapter_refusal_reason` would have worded the same
  refusal differently. `capability.DECOMPOSE_REFUSAL_REASONS` is the one table
  and `declare_adapter_capability` READS it; the `dora_reason` argument is gone
  from all thirteen handlers.
- **Not done, deliberately:** the SD1.5/SDXL selector still LISTS a DoRA (it
  reports `adapter_type: dora` on `GET /loras`, so a badge is available, but
  nothing filters it) — the refusal is the load-time guarantee, and the
  listing-side step stays deferred; no quality or speed measurement of DoRA
  against ordinary LoRA; `wd_on_out=False` (the `(1, in)` column form) is READ but never
  WRITTEN, so a run here always trains the row form; ReLoRA still takes the
  ordinary branch only; and `_quantize_text_encoder` did not get the guard,
  since no enabled architecture puts a DoRA target on a text encoder it
  quantizes.

### Phase 4: experimental fused backend — the mechanism landed, no backend did

- Prefer a pinned upstream dependency if 4.0.0 is published with fixes; otherwise
  vendor only the reviewed operations with complete provenance.
- Add an executed per-device/dtype/shape probe against the independent oracle.
- On launch failure, latch the backend off for the process and use the reference
  path on later calls; never switch the base mathematical function mid-training.
- Warm supported shapes before the first measured/training step.
- Start with LoHa/LoKr Linear bypass and the combined quantized-base path.
- Keep standard LoRA and DoRA unfused until their local measurements and strength
  contracts pass.

**Landed: the selection mechanism, with `reference` as the only backend.**
`backend/core/adapters/execution/` holds four boundaries, shaped after
`core/attention/` because that package is this repo's existing answer to the
same problem: `registry.py` (frozen `AdapterBackend` descriptors plus the
callable), `probe.py` (the executed per-region check), `dispatch.py` (the
conduit, the latch and warm-up) and `selection.py` (the name vocabulary).
Adding a backend is one `BACKENDS` entry and one callable. Gate:
`backend/tests/adapter_execution_backend_cheap_test.py`, whose every arm runs
against a FAKE backend the test registers — one numerically wrong, one that
raises during its probe, one that raises only after admission — because a gate
that could only be exercised by a backend nobody has written is a gate that
never runs.

- **The dispatch point is `forward_delta`, and no architecture is on it.** Each
  algebra keeps its unfused body as `reference_delta` and inherits one
  `forward_delta` from `_BranchTensorProtocol`, which calls the conduit. The
  eleven component-loader architectures reach the delta only through
  `CompositeAdapterLayer.forward`; TRAINING installs a bare branch, so
  `LoRALinearLayer.forward` and the MiniMax-H3 subclass's now call
  `forward_delta` instead of inlining the same two operations — same ops, same
  order, bit-identical, and it is what stops a fused backend from covering
  generation while missing training. Checked rather than asserted: an AST scan
  over `backend/` finds no module outside `core/adapters` that defines or calls
  `forward_delta` / `reference_delta`, and that scan is a gate.
- **A backend is usable for a region only after an executed comparison.** A
  region is `(algorithm, weight_decompose, device kind and index, activation
  dtype, branch dtype, out_features, in_features)`, and a verdict never
  generalises across regions — the bf16 arm of a shape whose fp32 arm passed is
  still unprobed. The probe holds a candidate to the SAME tolerances the shipped
  algebras are held to (`ORACLE_TOLERANCE`, now defined in `probe.py` and
  imported and pinned by `adapter_oracle_gate_cheap_test`, so loosening it there
  fails that gate); it compares the forward, the input gradient and every
  parameter gradient, and takes gradients with `torch.autograd.grad` so no
  `.grad` buffer is written.
- **The probe copies the branch, and that is load-bearing twice.** Every algebra
  zero-initialises one factor, so a freshly built branch has a delta of exactly
  zero and any backend at all — including one returning zeros — passes against
  it. The copy's zero factors are randomised and `PROBE_MIN_MOVE` refuses a
  verdict that could not have failed. The copy is also what keeps the probe off
  the live run's tensors.
- **The oracle is now reachable from one production module, deferred.**
  `core.adapters.reference` was test-only by contract; the probe is the executed
  admission check phase 4 asks for, and it has to compare against the
  independent oracle rather than against `layers.py`. `adapter_layering_test`
  therefore allows exactly `core/adapters/execution/probe.py`, requires the
  import to stay inside the function that runs it, and asserts in a fresh
  process that importing `core.adapters` still does not load the oracle.
- **The oracle's cost is a real limit on what can be certified.** It is written
  from the definition — `rank` explicit outer products, an explicit Kronecker
  assembly — so its working set is `2 * rank` full delta weights in fp32, not
  `rank`: `_low_rank_product` holds the list of rank-1 terms AND the
  `torch.stack` of them live at once. MEASURED at 512x512 rank 48 without a
  backward graph: a `rank + 3` estimate said 51.0 MiB against a 97.2 MiB peak,
  1.91x optimistic — and an inference-only backend, the fused GENERATION kernel
  case, is exactly the arm that skips the backward doubling. On a 4096x4096
  rank-32 projection this is gigabytes either way, so the probe estimates the
  host bytes BEFORE allocating and reports a region over
  `PROBE_ORACLE_BUDGET_BYTES` (2 GiB) as not admitted, rather than certifying it
  with a check that never ran. Fail-closed, and a residual: certifying the large
  regions needs a chunked oracle or an offline gate, and neither exists.
- **The latch is per PROCESS.** A launch or compile failure is a property of the
  process's toolchain, not of the layer that hit it first. Per LAYER would leave
  one model running two different mathematical functions and would let a layer
  change its own function at step 5000; per RUN would re-arm, for the next run
  in the same process, a backend that has already proved it cannot launch here.
  A process latch still cannot un-compute the steps that ran before it, so what
  makes "never switch the base mathematical function mid-training" true is that
  admission happens BEFORE step 0: a backend that fails warm-up never computes a
  training step. The failing call itself returns the reference result: a
  training step must not die because an experimental kernel did.
- **The latch message is keyed on a live result, not on warm-up having run.**
  It reports whether real work was computed with the backend, and that is
  `_served` — set when `backend.fn` returns successfully from
  `adapter_forward_delta`, never by the probe, which runs on a COPY. Keying it
  on "warm-up ran" conflates the two, and a first live call that latches on an
  unwarmed region then tells the operator that earlier steps used a function
  they did not. A message that can be false about which function computed a run
  is worse than no message. Both arms are gated.
- **Warm-up hooks into `BaseTrainer.train`, one line before
  `_maybe_compile_transformer`.** That window is already this repo's answer to
  the same question — after model device/dtype, gradient checkpointing, adapter
  injection and optimizer setup, after the stop-flag cleanup, before the first
  step. The gate reads the call out of the source, so a hook nobody calls fails
  rather than passing.
- **The run's training dtype is a HINT, unioned with each branch's own, never a
  replacement.** A run dtype is not what every branch sees: MiniMax-H3 runs a
  bf16 block stack with fp32 I/O heads and AdaLN projections and no `autocast`.
  Taking the run dtype alone fabricated a bf16 region for an fp32 head — whose
  probe fails on a genuine `mat1 and mat2 must have the same dtype` — while
  leaving the fp32 region that head really uses UNWARMED, so an operator who
  set `SUSHI_ADAPTER_BACKEND` on such a run got either a hard refusal over a
  region no forward produces or the step-1 stall warm-up exists to prevent.
  `strict` therefore refuses only when a branch has NO admitted region, not when
  one member of its union fails. The set is expected rather than certain — a
  branch's device can still move under block swap — so the conduit probes an
  unwarmed region on first sight and says so in the log. TWO KNOWN GAPS, both
  recorded rather than fixed: a backend admitted at warm-up and latched on a
  later region leaves `usable > 0`, so `strict` does not raise and the run
  continues on the reference path with only the `lora_backend_latched` warning;
  and a run with no adapter branches at all warms nothing and is not refused,
  which is why an env var set on a full-parameter run is inert rather than
  fatal.
- **Selection is explicit, off by default, and has no API surface.**
  `SUSHI_ADAPTER_BACKEND` is the developer entry point, the same shape as
  `SUSHI_FP8_SCALED_MM` and `SUSHI_INT8_MM`. Unlike the attention resolver,
  which absorbs an unknown string into `native` because a downgraded attention
  kernel computes the same function, an unrecognised, unavailable or latched
  adapter backend is REFUSED — `AdapterIncompatible` carrying
  `lora_backend_unavailable`, or the same code as a warning when the caller asks
  for that instead. The trainer refuses. No `openapi.yaml`, `param_defaults` or
  frontend change: only `reference` is registered, so a UI control would offer a
  choice of one, and the shipped-boundary rule says a design verdict is not a
  supported feature.
- **Nothing is auto-selected, and no performance property is claimed anywhere.**
  Auto-selection is defined by the acceptance matrix as a consequence of
  measurement on real device/dtype/shape regions, and this change measured
  nothing.

**Not landed, each blocked on something this step could not supply:**

- **The measurements.** The performance gates need a GPU, with the same model,
  batch, bucket, checkpointing, block-swap, optimizer and attention settings
  across four arms (current LoRA, unfused variant, fused variant, fused
  quantized base plus fused adapter), reporting warm-up separately and
  median/p95 forward, forward+backward, optimizer-inclusive whole step,
  generation latency, peak allocated/reserved VRAM and host memory. Until those
  exist no region may be auto-selected, and the published upstream table is not
  a substitute: its `lora` benchmark calls LoHa functions, and two of its rows
  are slower than eager.
- **The dependency decision.** A real backend needs either an immutable Git
  commit pin (4.0.0 is not on PyPI) or vendoring of only the reviewed operations
  with the Apache-2.0 license text, attribution, modification notices and an
  entry in `docs/legal/THIRD_PARTY_PROVENANCE.md`. That is the repo owner's
  call; nothing here imports, vendors or depends on `lycoris`.
- Generation-side warm-up. `AdapterSession`'s install completion is the
  analogous point, deliberately unwired while nothing can select a backend for
  generation.
- The merge path. `compute_delta_weight` is not on the seam, so a backend's
  rebuild/merge operations would need their own hook and their own probe. One
  qualification: `DoRALinearLayer.branch_delta_weight` reaches the seam through
  `self.branch.forward_delta(eye)` in its fallback arm — dead today, since every
  algebra this repo builds has `compute_delta_weight`, and live the moment an
  inner branch class without one exists.

**What registering a real backend requires, concretely:** a callable
`fn(branch, x) -> Tensor | None` returning the branch contribution alone; one
`AdapterBackend` entry declaring its `(algorithm, weight_decompose)` pairs,
activation dtypes, device kinds, whether it has a backward, whether it refuses a
mixed activation/branch dtype, and an `availability()` that reports a missing
dependency instead of raising; the dependency decision above; and the
measurements above before any region is auto-selected. No architecture, loader,
trainer or API file changes.

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
