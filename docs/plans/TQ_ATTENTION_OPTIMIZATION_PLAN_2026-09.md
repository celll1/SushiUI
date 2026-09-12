# TQ-Attention optimization plan (2026-09-12)

## Goal

Remove the measured BF16 training regression, let new training runs choose an
explicit TQ backward policy without changing old run semantics, and benchmark
the resulting speed, numerical behavior, and memory use on the RTX 6000 Ada.

## Findings that define the scope

- At `B=1, H=16, S=4096, D=128`, the native TQ BF16 backward spent 7.20 ms in
  dK/dV versus 1.73 ms for FP16. Passing BF16 `dO` into the Triton specialization
  caused the regression even though both main kernels convert it to FP16 before
  tensor-core work.
- Pre-casting `dO` to FP16 while keeping BF16 gradient buffers reduced the
  backward from 9.90 ms to 3.53 ms in the isolation probe. The sampled BF16
  dQ/dK/dV tensors remained bit-identical.
- SushiUI pins `backward_mode="triton"`, bypassing TQ 0.6's faster FA2-hybrid
  selection. This preserves determinism but leaves no per-run speed/VRAM choice.
- Adapter layout and contiguous conversion were not a material regression.
  Forward preprocessing remains a possible optimization, but it must not trade
  away the RHT accuracy advantage merely to improve a microbenchmark.

## Work packages

1. **Native Triton BF16 repair**
   - Preserve the FP32 delta preprocess on the caller's original `dO`.
   - Materialize one FP16 `dO` only for dK/dV and dQ, whose existing kernels
     already cast every loaded value to FP16.
   - Keep dQ/dK/dV buffers in the input gradient dtype.
   - Add BF16/FP16 correctness, bit-exactness, and performance-regression probes.

2. **Per-run backward policy**
   - Add `tq_backward_mode = auto | triton | fa2 | fa2_deterministic` to the
     training configuration, REST schema, preset plumbing, and UI.
   - Keep `triton` as the default. The repaired path is deterministic and keeps
     compact saved activations; FA2 hybrid remains an explicit speed trade-off.
   - Keep explicit FA2 requests strict: missing/ineligible FA2 is an error during
     training, not a silent native or Triton fallback.

3. **Forward-path decision**
   - Measure rotation, quantization, layout/cast, and INT8-kernel shares at
     sequence lengths representative of image and video training.
   - Land a fusion only if it preserves output bits/tolerances and has a useful
     steady-state win. Otherwise record the measured rejection; do not replace
     RHT with `rotation_mode="none"` or internally relabel Sage as TQ.

4. **Documentation and stale-contract cleanup**
   - Correct architecture routing notes for the current conduit defaults.
   - Document speed, determinism, activation footprint, and fallback behavior
     for each TQ backward policy.

## Verification and benchmark gate

- Compile and import every changed backend module with CUDA initialization
  stubbed where appropriate.
- Run focused CPU configuration/dispatch tests and TQ CUDA correctness tests.
- Compare native Triton before/after results for FP16 and BF16, including exact
  equality against the pre-cast reference path.
- Benchmark forward, backward, and total latency for `S=1024/4096/8192`,
  `D=64/128`, BF16/FP16, causal/non-causal, and MHA/GQA representative cells.
- Measure peak allocated and reserved VRAM for Triton, FA2 hybrid, and pure FA2.
- Separate first-call compilation from warmed steady-state timing and report
  the installed torch/Triton/FA2 versions.

No server restart or frontend build is part of this work.
