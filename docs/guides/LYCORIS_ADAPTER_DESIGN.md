# LyCORIS adapter integration design

Status: investigation and implementation plan; no LyCORIS runtime or new adapter
family is shipped by this document.

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
  `lycoris_kohya`, `diffusers_peft`), normalizing Hugging Face PEFT keys into
  canonical down/up stems seamlessly during `AdapterSession` parsing.
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
