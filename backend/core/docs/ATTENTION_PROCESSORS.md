# Attention architecture

SushiUI separates two choices that are easy to conflate:

- **mechanism** defines which query-to-key connections exist;
- **backend** chooses the kernel used to evaluate those connections.

`attention_method=dense` preserves the model's released connectivity. The
`attention_backend` / generation `attention_type` selector then chooses a dense
kernel. `h3_video_window` is a MiniMax-H3-specific, approximate mechanism; it is
not another spelling of FlashAttention or SageAttention.

## Dense conduit

The implementation is split by responsibility:

| File | Responsibility |
|---|---|
| `core/attention/contracts.py` | tensor, layout, mode, and fallback invariants |
| `core/attention/registry.py` | backend capabilities and callables |
| `core/attention/config.py` | names, aliases, and capability resolution |
| `core/attention/dispatch.py` | dense and packed-varlen dispatch |
| `core/attention/backends.py` | adapters for SDPA, FA2, Sage, and TQ |
| `core/attention/mechanisms.py` | semantic method vocabulary |
| `core/attention/observed.py` | kernels observed during one generation |

Canonical dense backends are:

| Backend | Training | Mask | GQA | Packed varlen | Notes |
|---|---:|---:|---:|---:|---|
| `native` | yes | yes | yes | exact reference | PyTorch SDPA chooses its CUDA sub-backend |
| `flash` | yes | no | yes | yes | explicit FlashAttention 2 |
| `sage` | no | no | yes | yes | quantized inference; no training fallback is permitted |
| `tq` | yes | no | yes | no | conduit-only Triton quantized backend |

Aliases `normal`, `none`, and `sdpa` resolve to `native`. `sla` remains accepted
only for compatibility with SLA-structured checkpoints; it is not a dense
backend and must never silently become one.

Capability resolution also considers dtype, head dimension, mask presence,
dropout, layout, mode, and Q/K head ratio. Inference warns and can fall back to
native. Training is strict when a selected kernel fails at runtime.

## Architecture ownership

Most architectures use `core.attention` directly. The following retain an
external dispatcher because their model implementation owns additional
semantics:

| Architecture | Dispatcher | Training selection |
|---|---|---|
| LTX-2.3 | Diffusers | native / FA2 are applied through `set_attention_backend`; other kernels are refused |
| ACE-Step 1.5 | Transformers | native / FA2 are applied through `set_attn_implementation`; other kernels are refused |
| Ideogram4 | Diffusers varlen path | head dimension 256 limits the local quantized kernels |

An external dispatcher is an owned exception, not permission to accept and
ignore the global setting. Unsupported combinations must fail before the first
training iteration.

## Selection guidance

- Start with `native`; it is the correctness baseline and often selects a fused
  PyTorch CUDA implementation automatically.
- Use `flash` when explicit FA2 is installed and the architecture supports its
  mask/layout contract.
- Treat `sage` as an inference-quality tradeoff, not a tolerance-equivalent
  training backend.
- Use `tq` only on conduit-routed paths. Masked calls resolve to native.

Do not quote generic speedups in UI or documentation. Record model, shape,
dtype, GPU, warm-up, attention share, end-to-end time, and peak allocated and
reserved VRAM for every claim.

## Adding a backend

Add the kernel adapter and one registry descriptor. Declare backward support,
dtype/head constraints, mask, dropout, GQA, and packed-varlen capability. Add
output and Q/K/V-gradient comparisons for every supported training signature.
Do not catch a training failure and continue with a different kernel.

## Adding a mechanism

Add its public name to `AttentionMechanism`, define a connectivity plan from
architecture metadata, and keep dense fallback input order unchanged. An
approximate mechanism needs explicit opt-in, provenance in generation metadata,
and quality gates in addition to timing and VRAM measurements. Training support
also requires deterministic recomputation and Q/K/V-gradient validation under
checkpointing and activation offload.
