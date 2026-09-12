# Attention architecture and acceleration audit (2026-09)

## Scope and conclusion

This audit covers SushiUI's generation and training attention paths, the installed
attention runtimes, and attention methods published or released through 2026-09-12.
It is a static review: no GPU workload was started.

The current design has a sound core idea—a shared BSHD conduit with explicit
training/inference capability checks—but its capability model and architecture
coverage have drifted behind the installed libraries. The highest-value work is not
a new dense kernel. It is:

1. repair the exact-attention control plane (capability discovery, strict preflight,
   varlen, masks, and missing architecture wiring);
2. add an H3-specific sparse-attention interface which keeps text, reference,
   audio, and mixed boundary tiles dense and sparsifies only target-video to
   target-video edges;
3. prototype that interface with an existing licensed kernel before deciding
whether SushiUI needs its own Ada-optimized CUDA/CUTLASS kernel.

Implementation status (2026-09-12): the dense contract guards, strict training
fallback, boolean-mask preservation, Sage GQA capability, FA2/Sage packed-varlen
registry, LTX-2.3/ACE-Step training selection, and model-local Anima dispatch are
implemented. MiniMax-H3 now has an opt-in `h3_video_window` FlexAttention reference
path on every video endpoint. Its connectivity and lifecycle have CPU coverage;
performance and quality acceptance remain GPU work.

For MiniMax-H3, sparse attention is technically well matched to the released model.
MiniMax states that H3 used native sparse attention in its final training stage,
although the first open-source release exposes full attention only. This removes the
largest architectural objection, but it does not make an arbitrary community mask
numerically equivalent or quality-neutral.

## Current implementation

Installed static environment:

- PyTorch 2.10.0+cu130
- diffusers 0.38.0
- FlashAttention 2.8.3
- SageAttention 2.2.0 build for torch 2.9+
- TQ Attention 0.6.0

The conduit consists of:

- `backend/core/attention/config.py`: names and capability downgrade policy;
- `backend/core/attention/registry.py`: native, FA2, Sage, and TQ descriptors;
- `backend/core/attention/dispatch.py`: layout conversion, GQA handling, kernel
  call, fallback, and packed-varlen handling;
- `backend/core/attention/backends.py`: kernel adapters;
- `backend/core/attention/observed.py`: per-generation observed-backend reporting.

The model-facing coverage is uneven:

| Family | Main route | Important limitation |
|---|---|---|
| SD1.5 / SDXL | conduit by default; legacy diffusers option | custom processor owns regional/style behavior |
| Z-Image | conduit | SLA is only a reserved passthrough string here |
| Flux2 | conduit processors or diffusers | two registries can still diverge |
| Anima | conduit in generation and training, including the LLM adapter | masked adapter calls correctly resolve to native |
| Lens | conduit | fixed inference mode is stamped in one vendor call |
| Krea2 | conduit, GQA | native path materializes repeated K/V heads |
| Ideogram4 | separate diffusers/FA2-varlen dispatcher | head dimension 256 excludes current Sage path |
| MiniT2I | conduit with head-dimension padding | padded dimensions constrain backend choice |
| SenseNova | conduit for generation path, GQA 32/8 | Sage GQA is accepted by the common registry |
| MiniMax-H3 | conduit dense path plus opt-in FlexAttention video window | sparse quality/performance is not yet accepted |
| MiniMax Music 3 | conduit | generation only |
| LTX-2.3 | diffusers dispatcher | training applies native/FA2 and refuses unsupported kernels |
| ACE-Step 1.5 | transformers dispatcher | training applies SDPA/FA2 and refuses unsupported kernels |

## Findings

### P0 — correctness and truthful configuration (implemented)

#### 1. Sage GQA and packed varlen (implemented)

The registry now declares installed SageAttention's divisible GQA support and its
packed-varlen entry point. Capability resolution still gates package availability,
dtype, head dimension, mode, mask, dropout, and layout. Sage remains inference-only;
its varlen BF16 behavior still requires a GPU tolerance measurement.

#### 2. LTX-2.3 and ACE-Step training selection (implemented)

LTX-2.3 now applies native/FA2 through diffusers' dispatcher; ACE-Step applies them
through Transformers' dynamic attention interface. Unsupported kernels are refused
before iteration rather than recorded and ignored.

#### 3. Strict runtime fallback for training (implemented)

The conduit now defaults to these policies:

- `error` for training: capability/kernel failure stops before or at the first
  representative call;
- `warn` for interactive generation, which reports and falls back;
- record requested, resolved, and actually observed kernel independently.

Capability preflight is declarative; a selected training kernel that still fails at
runtime raises instead of silently continuing with different math.

#### 4. Input invariants (implemented)

The common boundary now validates layout, rank, batch/sequence/head dimensions,
device/dtype equality, mask/causal compatibility, and GQA divisibility before
dispatch. Signature caching remains a possible CPU micro-optimization.

### P1 — exact or tolerance-equivalent acceleration

#### 5. Packed-varlen fallback (partially implemented)

FA2 and Sage now use their installed packed kernels. Native/TQ fallback still calls
`.tolist()` on cumulative lengths, loops in Python, launches one SDPA per segment,
optionally repeats K/V, and concatenates results. If offsets are on CUDA, `.tolist()`
also synchronizes the device. Remaining order:

1. diffusers' installed `flash_varlen` / `sage_varlen` adapters where this removes
   duplicate compatibility code;
2. a batched FlexAttention `BlockMask` fallback instead of per-segment SDPA;
3. retain the loop only as a diagnostic reference path.

#### 6. Boolean mask preservation (implemented)

The common conduit now preserves boolean masks and normalizes rank without an
equivalent float allocation. SDPA's `True`-means-allowed convention is covered by
the conduit test.

The regional prompt processor still constructs a dense `[1, 1, Q, K]` bias per
attention call after caching only the regional part. Its spatial rule is an excellent
FlexAttention `score_mod`/`BlockMask` candidate. This changes the implementation, not
the mathematical mask, although floating-point reduction order will differ.

#### 7. Native GQA policy is hard-coded to one old measurement

The conduit materializes K/V from 8 to 32 heads for SenseNova and from 12 to 48 for
Krea2 because a prior shape measured PyTorch `enable_gqa=True` about 9x slower. The
speed result can be valid while the policy is still too broad: explicit repetition
multiplies K/V activation storage and memory traffic by four.

Choose between native GQA, explicit expansion, FA2, Sage, cuDNN SDPA, and TQ using a
cached shape/hardware benchmark or a conservative static table. Include sequence
length and free-memory margin in the policy. The selected route must be stable for a
training run.

#### 8. Avoid needless materialization and per-call bookkeeping

- Native attention always returns a contiguous BSHD tensor; BHSD callers then
  transpose and materialize again. Define output-stride requirements and materialize
  only at the consumer that needs it.
- Sage and TQ call `.contiguous()` on all Q/K/V even when already suitable.
- FP32 input is now refused by capability resolution for FA2/Sage/TQ; the dead
  copy-to-BF16-and-back branches have been removed from all three adapters.
- backend imports, resolution, warning attachment, and `note_backend` run in every
  layer. Resolve a callable once per model/signature and use a one-write-per-forward
  observation fast path.

These are individually small CPU or bandwidth costs; they matter most in short image
models where attention itself is not dominant.

#### 9. The local registry duplicates a much richer diffusers registry

The installed diffusers version already exposes native sub-backends, FlexAttention,
FA2/3/4 adapters, flash/sage varlen, Sage variants, and xFormers. SushiUI maps only
`native`, `flash`, and `sage`; TQ is conduit-only. Keep SushiUI's policy and custom
QKV hooks, but adapt standard dense kernels from diffusers rather than copying their
version and package compatibility logic. The local descriptor must still express
backward, determinism, mask kind, GQA, dtype, device architecture, varlen, and
compile compatibility.

#### 10. Direct attention islands remain

The Anima LLM adapter now uses the common conduit and model-local selection. The two
remaining PiD calls are owned exceptions: they combine context-parallel layouts with
an explicit Flash/Efficient-only SDPA guard that prevents an O(N²) math fallback.
SenseNova's remaining direct helper belongs to its Qwen understanding/prefix path,
not denoising attention. Ideogram4 remains an owned diffusers-varlen dispatcher
because head dimension 256 and segment metadata differ.

### P2 — documentation and API design

`backend/core/docs/ATTENTION_PROCESSORS.md` is materially stale: it lists only three
backends, points at moved files, gives unsupported generic speed claims, and says NAG
does not use custom kernels although current NAG paths enter the conduit. Replace it
with a generated capability matrix plus measured, hardware-labelled results.

Separate the current single selector into:

- **semantic method**: dense, block-sparse, radial, VSA, SLA, Sol-Attn;
- **kernel**: SDPA auto/cuDNN/flash, FA2, Sage, TQ, Flex, custom CUDA;
- **precision**: model dtype, INT8 QK, FP8/FP4 variants;
- **policy**: strict/fallback, deterministic, autotune, compile;
- **method parameters**: block shape, budget, schedule, modality connectivity.

Sparse methods are not fungible kernel backends. Treating `sla` as another spelling
beside `flash` is the design error that the reserved passthrough currently exposes.

## New attention methods and SushiUI fit

Paper speedups below are authors' results on their models and hardware, not expected
SushiUI speedups.

| Method | Exact? | Training | SushiUI fit | Verdict |
|---|---:|---:|---|---|
| PyTorch SDPA / cuDNN | tolerance-equivalent | yes | all dense paths on Ampere/Ada+ | P0 baseline; expose selected native sub-backend for measurement |
| FA2 varlen | tolerance-equivalent | yes | packed batches and block-diagonal documents | implement in common varlen path now |
| FA3 | tolerance-equivalent | yes | Hopper-focused | not a priority for RTX 6000 Ada |
| FA4 | tolerance-equivalent | limited by build/hardware | Blackwell/Hopper paths | future hardware option, not an Ada solution |
| FlexAttention | exact mask semantics | yes | regional masks, structured windows, prototyping | high priority as reference/prototype backend; compile and bucket shapes |
| SageAttention 2/2++ | quantized | inference in current policy | dense H3, SenseNova GQA, varlen | fix capabilities; validate H3 audio as well as video |
| SageAttention 3 | quantized FP4/INT8 | experimental | mainly Blackwell | no Ada priority; research option for training |
| SpargeAttention | approximate sparse + quantized | no | generic inference, long image/video DiTs | useful comparison backend, but H3-specific routing now has better evidence |
| Sliding Tile Attention | approximate 3-D locality | optional fine-tune | LTX/H3 video tokens | viable, but VSA/Sol and H3-native patterns are stronger current candidates |
| Radial Attention | static approximate mask | LoRA adaptation | video models with known `(t,h,w)` | best simple Flex prototype; preserve global modality tokens |
| Sparse VideoGen / HASTE / ScalingAttention | approximate, profiled | no | calibrated per model/head | research branch; requires H3 calibration corpus |
| VSA / VMoBA | learned or routed block sparse | yes | strongest H3 training path | high priority for sparse-distilled H3 checkpoints |
| SLA / SLA2 | sparse + linear residual | yes, extra parameters | only matching converted checkpoints | support as a separate model contract, never dense fallback |
| SpargeAttention2 | sparse-distilled | yes | post-training H3 route | promising but a training method, not a runtime toggle |
| Sol-Attn | approximate on-the-fly with tail correction | inference | H3 and LTX long sequences | highest-priority training-free research candidate |
| MiniMax Sparse Attention (MSA) | model/kernel co-design | model-specific | M3-style language path, not H3 drop-in | learn from routing/kernel design; do not conflate MSA with H3 |

Methods aimed at distributed context parallelism or autoregressive KV-cache decode
do not match SushiUI's main single-GPU bidirectional diffusion workload. They should
not be added merely because their asymptotic complexity is attractive.

## MiniMax-H3 implementation assessment

H3 has 50 DiT blocks plus a token refiner, 56 heads, head dimension 128, and one
packed sequence containing text, conditioning media, audio, and target video. It
uses full self-attention with no mask in the released implementation. The model
already provides the metadata a sparse implementation needs:

- modality tags;
- `(t, h, w)` positions;
- video, audio, and text indices;
- timestep indices;
- a stable per-block attention boundary.

A safe sparse contract should define connectivity, not merely a percentage:

1. text, reference/conditioning, audio, non-video queries, and mixed tiles remain
   dense initially;
2. only target-video-query to target-video-key edges are eligible for sparsity;
3. token reordering must update/restores positions and every index consistently;
4. early denoising steps use a denser budget than middle steps;
5. dense fallback receives the original packed order and complete Q/K/V;
6. a sparse-distilled adapter that introduces routing/gating parameters declares
   that requirement in checkpoint metadata and cannot run with dense attention.

This matches both the official statement that H3 was trained with sparse attention
and recent H3-specific implementations which keep non-video and boundary traffic
dense. Community measurements show that attention savings become much more valuable
as sequence length grows, but those numbers must not be copied into SushiUI's UI
until reproduced on its pipeline and checkpoint formats.

Recommended H3 sequence:

1. **Dense baseline cleanup:** repair Sage GQA; expose SDPA/cuDNN/FA2 selection;
   capture per-layer shapes and attention share of iteration time.
2. **Semantic interface:** add an `AttentionPlan` carrying token roles, coordinates,
   block index, denoise step, connectivity policy, and dense fallback. Keep it
   orthogonal to block swap and activation offload.
3. **Reference sparse path (implemented, not performance-accepted):** target-video
   block masks use FlexAttention. Condition/reference, audio, non-video, and mixed
   boundary blocks stay dense; only pure target-video block pairs are windowed.
4. **Existing-kernel evaluation:** test licensed H3 block-sparse or FastVideo VSA
   code behind the same plan. FastVideo is Apache-2.0; one new H3-specific kernel is
   MIT. Record provenance and avoid importing code whose license is unclear.
5. **Custom kernel decision:** only build if Ada measurements show routing/packing or
   Flex overhead consumes the expected gain.

## If a custom kernel is justified

Do not write another dense FlashAttention kernel. On RTX 6000 Ada, FA2, PyTorch SDPA,
and cuDNN already cover that problem. A SushiUI kernel would be justified for H3's
specific mixed dense/sparse graph and streamed-Q memory contract.

Preferred architecture:

- 64-query by 64/128-key tiles, with block size selected by measured sequence shape;
- full dense prefix/mixed-tile handling fused with sparse video routing;
- online softmax across dense and sparse partitions so normalization is global;
- optional approximate summary of skipped blocks, following the Sol-Attn principle;
- Q streaming with persistent or bounded K/V so it composes with H3's current
  activation-memory work;
- BF16 reference first, then optional INT8 QK after output/audio quality is stable;
- deterministic routing metadata saved or recomputed identically for backward;
- explicit forward-only and forward/backward entry points rather than autograd
  accidentally reaching an inference kernel.

Use Triton for the first auditable BF16 implementation. Use CUDA/CUTLASS/CuTe only
after profiling proves that routing, irregular gathers, or Ada tensor-core occupancy
cannot be made competitive in Triton. Maintain a slow PyTorch/Flex reference for
every connectivity rule. Kernel tests must compare output and Q/K/V gradients across
boundary sequence lengths, partial tiles, modality transitions, and non-contiguous
inputs.

## Composition with existing memory features

- **Gradient checkpointing:** exact dense kernels compose normally. Dynamic sparse
  training must reproduce routing during recomputation or save routing metadata.
- **Activation offload/dispatch:** custom autograd saved tensors and hooks must be
  audited; forward-only kernels cannot be enabled under grad.
- **Block swap/layer offload:** weight movement is orthogonal, but sparse workspace,
  JIT compilation, and prefetch streams must not contend with the transfer stream.
- **LoRA:** exact kernel changes are transparent within tolerance. Sparse VSA/SLA
  adapters may add parameters absent from the base model and require an explicit
  checkpoint contract.
- **Prompt/latent caches:** semantically unaffected, but cached token indices and
  sparse plans must be invalidated when resolution, frame count, conditioning
  modality, or prompt length changes.
- **`torch.compile`:** FlexAttention depends on compilation for performance. Bucket
  shapes and precompile outside the measured iteration; dynamic sparse plans can
  otherwise cause recompilation.

## Remaining GPU validation

No speed, VRAM, or quality claim is made for `h3_video_window` yet. Before changing
its experimental status, run this matrix on the actual target GPU:

- H3 fl2va txt2vid and img2vid, ref2va ref2vid, video outpaint, and video inpaint;
- short/default/long frame counts and at least three aspect ratios;
- dense native/FA2/Sage/TQ baselines versus window radii `(8,4)`, `(16,8)`, and
  `(32,16)`, excluding first-call compilation from steady-state timing;
- peak allocated/reserved VRAM, compile workspace, per-step latency, and complete
  generation latency after warm-up;
- prompt adherence, reference identity, motion, temporal consistency, audio
  intelligibility/synchronization, and failure rate over a fixed prompt/seed suite;
- block swap on/off and output-head fusion on/off.

Training, checkpointing, and activation offload are intentionally refused by the
sparse path until output and Q/K/V-gradient tests exist. Build a custom Triton kernel
only if profiling shows Flex routing or mixed dense/sparse tile handling erases the
expected attention saving. Keep Flex as the connectivity oracle for any such kernel.

## Validation gates

### Exact/tolerance-equivalent paths

- CPU/static contract tests for resolution and refusal;
- output and Q/K/V gradient comparisons to SDPA, using dtype-specific tolerances;
- fixed-seed end-to-end latent comparison;
- peak allocated/reserved VRAM and iteration/generation time after warm-up;
- all supported head dimensions, GQA ratios, mask kinds, and partial varlen segments;
- checkpointing, activation offload, block swap, LoRA, and compile combinations.

### Approximate sparse paths

In addition to performance, compare dense and sparse across prompt adherence,
motion, temporal consistency, reference identity, audio intelligibility/timing, and
failure rate. H3 evaluation must include text-to-video, image/video conditioning,
reference generation, and audio-video conditioning. A fixed seed is useful for
measuring drift, not for asserting bit equality.

Use Amdahl's law before accepting a kernel: if dense attention occupies fraction
`p` of a step and the kernel accelerates it by `s`, maximum step speedup is
`1 / ((1 - p) + p / s)`. Reject methods whose routing, packing, compilation, or
quality-recovery overhead erases the measured end-to-end gain.

## Recommended work packages

1. **P0 exact control plane:** capability redesign, invariant validation, strict
   training mode, truthful LTX/ACE configuration.
2. **P1 dense/varlen performance:** Sage GQA/varlen, bool masks, native GQA policy,
   diffusers adapter, stride/copy cleanup.
3. **P1 H3 sparse contract and Flex reference:** implemented; GPU/quality gate open.
4. **P2 H3 kernel bake-off:** Flex, Sol-Attn-compatible implementation, VSA, and an
   H3-specific block-sparse kernel on the actual Ada target.
5. **P2 sparse training:** only after a checkpoint/adapter format and quality suite
   are defined; evaluate VSA/SLA/SpargeAttention2 independently.

## Primary sources

- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [PyTorch FlexAttention](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html)
- [NVIDIA cuDNN attention support matrix](https://docs.nvidia.com/deeplearning/cudnn/v1.18.0/operations/Attention.html)
- [diffusers attention backends](https://huggingface.co/docs/diffusers/optimization/attention_backends)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [SageAttention](https://github.com/thu-ml/SageAttention)
- [MiniMax-H3 official repository](https://github.com/MiniMax-AI/MiniMax-H3)
- [SpargeAttention](https://arxiv.org/abs/2502.18137)
- [Sliding Tile Attention](https://arxiv.org/abs/2502.04507)
- [Sparse VideoGen](https://arxiv.org/abs/2502.01776)
- [VSA](https://arxiv.org/abs/2505.13389)
- [Radial Attention](https://arxiv.org/abs/2506.19852)
- [VMoBA](https://arxiv.org/abs/2506.23858)
- [SLA](https://arxiv.org/abs/2509.24006)
- [SpargeAttention2](https://arxiv.org/abs/2602.13515)
- [SLA2](https://arxiv.org/abs/2602.12675)
- [HASTE](https://arxiv.org/abs/2605.14513)
- [ScalingAttention](https://arxiv.org/abs/2606.23019)
- [Sol-Attn](https://arxiv.org/abs/2607.24027)
- [FastVideo VSA and H3 integration](https://github.com/hao-ai-lab/FastVideo)
- [H3-specific block-sparse reference](https://github.com/Occipital-Labs/h3-sparse-attn)
