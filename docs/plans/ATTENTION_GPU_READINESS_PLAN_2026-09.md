# Attention GPU-readiness implementation plan (2026-09-12)

## Goal

Finish the CPU/static side of the remaining MiniMax-H3 attention work without
initializing CUDA, so the next free-GPU session consists only of running a
documented acceptance matrix. GPU results remain the release gate: unexecuted
kernel code is not described as performance- or quality-accepted.

## Updated implementation decision

The original audit proposed a Sol-Attn-compatible implementation and a possible
new Triton/CUDA kernel. NVIDIA has since released `sol-attn` 0.5.0 under
Apache-2.0, including a portable Triton implementation and an SM89 CuTe kernel.
SushiUI should integrate that public API rather than fork approximately 16,000
lines of hardware-specific code or write another dense/sparse kernel before a
profile justifies it.

Pin the optional source dependency to NVlabs/Sana `sol-engine` commit
`8e0db4fa562d727ea28b8d63015c196db7d97cae`. Keep imports and compilation lazy.
The public kernel contract is contiguous BF16 BSHD, equal Q/K/V shapes,
head-dimension 128, forward-only, and CUDA compute capability 8.0 or newer.

## Work packages

1. **Sol-Attn semantic and kernel adapter**
   - Add an explicit `h3_sol_attn` mechanism; do not disguise it as a dense
     backend.
   - Validate H3's packed `[text | conditioning video | audio | target video]`
     layout and keep the complete prefix exact as both K/V and query rows.
   - Use the official kernel's exact-KV sink in one sparse pass, followed by a
     dense prefix-query replacement. Preserve strict failure for an explicitly
     selected but unavailable/ineligible kernel so measurements cannot be
     silently labelled sparse while running dense.
   - Support the validated `tau`, threshold method, first-dense-step, and
     first-dense-layer policy without process-global state.

2. **API and UI control plane**
   - Put every default in `backend/api/param_defaults.py`.
   - Thread the mechanism and Sol parameters through all five H3 video request
     surfaces and `openapi.yaml`.
   - Expose the method and parameters through the existing global attention
     settings. Keep dense as the default and label Sol-Attn experimental.

3. **CPU oracle and contract tests**
   - Implement a small materialized PyTorch reference for threshold routing,
     approximate skipped-block correction, partial blocks, and exact sinks.
   - Test packed-layout refusal, warm-up policy, lazy dependency loading,
     strict failure, dense prefix-query replacement, and backend observation.
   - Add a GPU-marked kernel suite that compares the official kernel with the
     CPU oracle/full-sink dense SDPA and exercises boundary lengths 63/64/65 and
     production-like H3 sequence shapes. Collection must not initialize CUDA.

4. **Acceptance harness and documentation**
   - Provide one command that runs correctness, latency, allocated/reserved
     VRAM, route-density, and H3 endpoint smoke cases after the GPU is free.
   - Record exact hardware/software metadata and separate first-call compile
     cost from steady-state timing.
   - Update the attention audit and model facts to distinguish implemented,
     CPU-verified, GPU-runnable, and GPU-accepted states.

## Deliberately excluded

- VSA, SLA/SLA2, and SpargeAttention2 are training/checkpoint methods, not
  interchangeable kernels for the released dense checkpoint. Their integration
  remains blocked on a declared checkpoint/adapter format and quality suite.
- A new SushiUI Triton or CUDA/CUTLASS/CuTe kernel is not built unless the
  official Sol-Attn/Flex comparison demonstrates a measured missing capability
  or dominant integration overhead on the target Ada GPU.
- No GPU workload, backend/frontend restart, or frontend build is part of this
  CPU/static preparation pass.

## Verification before GPU availability

- Python compilation and real imports with CUDA initialization stubs.
- Focused CPU tests plus existing attention/H3 contract suites.
- OpenAPI/default/frontend field parity by static inspection.
- Clean test collection of the GPU acceptance suite with no CUDA context.

## GPU acceptance gate

Run only after the owner confirms the GPU is free. Acceptance requires:

- official-kernel output against the materialized oracle and full-sink dense
  attention within declared BF16 tolerances;
- no silent dense fallback and at least one observed sparse call per request;
- all H3 video endpoints, short/default/long durations, three aspect ratios,
  block swap on/off, and output-head fusion on/off;
- peak VRAM and warmed end-to-end latency improvements large enough to justify
  the approximation;
- fixed-seed visual/audio review covering motion, reference identity, temporal
  consistency, speech intelligibility, and synchronization.
