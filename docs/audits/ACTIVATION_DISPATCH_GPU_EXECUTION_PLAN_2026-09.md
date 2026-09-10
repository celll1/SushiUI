# Activation dispatch GPU validation execution plan (2026-09)

Status: approved for execution. Run 127 must remain stopped throughout this
campaign. No backend/frontend restart is permitted.

## Goal

Close the VRAM-dependent validation left by
`ACTIVATION_DISPATCH_GPU_VALIDATION_BACKLOG_2026-09.md` without treating a
synthetic CUDA check as proof for a real architecture. The campaign separates
three gates so that every claim states exactly what was exercised.

## Safety boundary

- Never start, resume, stop, or alter run 127 during validation.
- Run probes in their own foreground process. Do not terminate an unrelated
  process to recover VRAM.
- Cap each probe's allocator before allocating and record dedicated/free VRAM
  at entry. Abort before model load when the declared bound cannot fit.
- Write probe output only below a dedicated repository results directory or a
  newly created validation-run directory; do not overwrite training artifacts.
- Do not restart either application server. Frontend builds remain owner-run.

## Gate 1: CUDA mechanism

Add a reproducible probe for the production `offload_activations` context. For
3-D audio, 4-D image, and 5-D video-shaped workloads, compare OFF and ON using
the same seed and tensors. Record loss, every input/parameter gradient,
offloaded bytes, allocated/reserved peak, host working set, and warmed
iteration median/p95. Exercise both the default threshold and a deliberately
tight memory decision. A failed equivalence or non-finite gradient blocks all
model claims.

## Gate 2: real checkpoints

For each locally available architecture, use its production loader, adapter,
training step, and a real dataset batch. Compare OFF/ON before the optimizer
step, then run a short optimizer smoke test. Apply the backlog's method and
composition matrix only where the architecture supports the combination.
Results must include the exact checkpoint, dataset, dtype, bucket/clip length,
checkpointing and block-swap state.

The inventory at planning time is:

| Architecture | Local evidence | Planned disposition |
|---|---|---|
| MiniMax-H3 | Complete flat checkpoint tree under `M:\model\minimax_h3` | P0 real-checkpoint test |
| ACE-Step 1.5 | DiT, text encoders and VAE under `M:\model\ace-step` | P0 real-checkpoint test if an audio batch is locally available |
| SenseNova U1.5 | Real checkpoint plus LoRA/full historical runs | P0 flow, instruction and mixed tests |
| SDXL | Real checkpoints and historical LoRA/full/ControlNet runs | P1 supported-method tests |
| Anima | Local model tree and historical SDXL-family training artifacts | P1 real-checkpoint test after loader preflight |
| Krea 2 | Local model tree | P1 dense/quantized preflight and supported tests |
| Z-Image | Historical run references a real checkpoint | P1 preflight and supported tests |
| FLUX.2 | Historical LoRA/full runs reference a real checkpoint | P1 plain/reference tests where data exists |
| MiniT2I | Historical LoRA/full runs reference a model or scratch recipe | P1 preflight and supported tests |
| LTX-2.3 | No checkpoint in the configured model root | Record blocked; do not substitute H3 |
| SD1.5 | No checkpoint found in the configured model root | Record blocked |
| Lens | No checkpoint found in the configured model root | Record blocked |
| Ideogram 4 | No checkpoint found in the configured model root | Record blocked |

Missing assets are a blocked validation row, not a passed row. The final report
must name the missing checkpoint or dataset and retain an executable command or
configuration for completing it later.

## Gate 3: iteration candidates

`TRAINING_ITERATION_VRAM_VALIDATION_PLAN_2026-09.md` contains proposals, not
finished implementations. Handle them separately:

- A, B, C and E require an implementation commit before their A/B campaign.
- D changes sample order and is a quality experiment, not an equivalence
  refactor; leave it opt-in and report it separately.
- Do not use results from activation dispatch to mark any of A-E complete.

Implement only candidates whose static design can preserve the documented
training contract and whose baseline can be measured on an available model.
Each accepted candidate gets its own implementation, tests, measurements and
commit. Rejected candidates remain documented with the measured reason.

## Commit units

1. This execution plan.
2. CUDA probe plus its static tests.
3. CUDA mechanism results and any mechanism fix found by the probe.
4. One commit per independently validated architecture or architecture family.
5. One commit per accepted iteration candidate.
6. Final matrix and explicit residual blockers.

No commit may claim model validation from configuration reachability, a mocked
forward, or a synthetic tensor alone.
