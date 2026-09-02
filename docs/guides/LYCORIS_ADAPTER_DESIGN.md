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

Training shares `LoRALinearLayer` from
`backend/core/training/adapters/sd15_adapter.py`, but does not yet share the
rest of the mechanism:

- `LoRATrainer._create_adapter` selects one of 13 adapters with an if-chain.
- Every adapter directly assumes `lora_down` and `lora_up` when collecting
  optimizer parameters and saving checkpoints.
- `LoRATrainer.load_checkpoint` resumes only those two tensors.
- `ARCH_REGISTRY` selects training ops, not adapter factories, target topology,
  checkpoint codecs, or generation loaders.

Generation is split again. SD1.5/SDXL use `LoRAManager` and Diffusers/PEFT.
Most other architectures have a custom loader in their pipeline backend and a
partially shared helper under `backend/core/models/<arch>/`. These paths do not
implement one common strength, target-count, multiple-adapter, step-range, or
atomic restore contract.

### Existing LoRA round-trip defects

These are Phase 0 blockers, not new-variant limitations:

- Z-Image training writes `lora_transformer_<flattened>` keys while generation
  searches `transformer.<dotted>` keys. Its generation debug line also reads a
  nonexistent `scaling` attribute after replacing the first matching module.
- Krea2 contains parser/apply/restore helpers, but its generation backend never
  applies `params["loras"]`.
- LTX-2.3 has a training adapter but explicitly lacks a generation loader.
- FLUX.2 training can save Qwen text-encoder adapters, but generation applies
  transformer tensors only.
- Anima's default training scope includes attention, MLP, and LLM-adapter
  targets; generation applies only its attention iterator.
- ACE-Step can train an opt-in MLP scope; generation is attention-only.
- `classify_lora_keys` recognizes only a subset of current architectures, so
  several SushiUI-trained adapters appear as `unknown` in AddLoRA.
- Several custom loaders replace an existing wrapper with a new wrapper around
  the original base. Multiple selected adapters therefore become last-wins or
  cause later adapters to match zero targets instead of summing branches.
- Custom paths do not consistently honor `apply_to_text_encoder`,
  `apply_to_unet`, per-block weights, or `step_range`.
- Loader exceptions and zero-target applications can be logged while generation
  still returns success.

The first implementation milestone must close the ordinary LoRA
trainer-save-to-fresh-generation round trip for every advertised architecture.

## Architecture feasibility

"Additive" below means LoRA, LoHa, and LoKr on the architecture's existing
Linear target set. It states implementation feasibility, not measured quality or
speed. LoCon/Conv targets are separate future scope.

| Architecture | Additive family | Initial DoRA | Required architecture-specific work |
|---|---|---|---|
| SD1.5 | yes | dense Linear | Preserve U-Net and optional CLIP MLP groups |
| SDXL | yes | dense Linear | Preserve U-Net and two text-encoder LR groups |
| Z-Image | yes | dense Linear | Repair current key codec and generation loader first |
| FLUX.2 | yes | dense only | Apply saved Qwen targets; gate quantized bases |
| Anima | yes | dense only | Make inference target scope equal training scope |
| Lens | yes | dense Linear | Retain fused-QKV path naming and frozen GPT-OSS encoder |
| Ideogram4 | yes | deferred | Dual transformer can be FP8; start with DoRA refused |
| MiniT2I | yes | dense Linear | Preserve transformer and optional FLAN-T5 scopes |
| Krea2 | yes | deferred | Add generation call; INT8/FP8 bases need capability gates |
| LTX-2.3 | yes | dense only | Add generation loader; Gemma-3 remains frozen |
| MiniMax-H3 | yes, later gate | deferred | Preserve custom QKV mapping and FP8/ConvRot dtype policy |
| ACE-Step | yes | dense only | Make opt-in MLP scope round-trip through generation |
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

Replace the single-branch wrapper with `CompositeAdapterLinear`. It owns the
base once and holds multiple named branches, allowing AddLoRA to change strength
or step activation without rewrapping. Its output is the base result plus every
active additive branch. DoRA uses the full-difference interpolation contract
rather than being treated as an ordinary additive factor.

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

### Phase 0: repair ordinary LoRA

- Fix the Z-Image codec/runtime error.
- Wire Krea2 generation; either wire LTX-2.3 or keep its refusal explicit.
- Complete FLUX.2 text-encoder, Anima MLP/LLM, and ACE-Step MLP scopes.
- Make multiple adapters additive and unify strength, step range, component
  selection, failure, and restore behavior.
- Classify checkpoints from every training architecture.
- Add trainer-save to fresh-generation round-trip tests for all advertised
  architectures.

### Phase 1: extract the shared engine

- Introduce `AdapterSpec`, `AdapterLayer`, `AdapterTarget`, codec registry,
  composite wrapper, and atomic session.
- Migrate ordinary LoRA without numerical changes.
- Replace per-adapter optimizer/save assumptions with protocol methods.
- Move topology and codec hooks into architecture descriptors.

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
