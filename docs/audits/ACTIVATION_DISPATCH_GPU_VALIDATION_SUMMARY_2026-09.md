# Activation dispatch GPU validation summary (2026-09-10)

## Scope decision

The owner approved stopping the long per-architecture real-checkpoint matrix
after the shared CUDA gate and the highest-priority MiniMax-H3 matrix. The
remaining rows retain static coverage but must not be described as GPU
validated.

No training run or application server was started or restarted. Run 127
remained stopped throughout the campaign.

## Completed gates

| Gate | Coverage | Result |
|---|---|---|
| Shared CUDA mechanism | 3-D audio, 4-D image, and 5-D video-shaped forward/backward; exact loss and dtype-tolerant gradients | passed |
| Shared static integration | configuration plumbing, workload-family keys, temporal keys, fused-path recovery boundary, SenseNova text key, and H3 helpers | 146 tests and 356 subtests passed |
| MiniMax-H3 real transformer | short/long, block swap 0/40, dispatch off/on, gradient and AdamW controls | passed with gradient checkpointing |
| MiniMax-H3 capacity | checkpointing off, dispatch off/on, block swap 0/40 | blocked by the 80% allocator safety cap even at the short clip |
| Windows block swap startup | cp932 console output | fixed and covered by regression test |

The H3 measurements and their limitations are in
`MINIMAX_H3_ACTIVATION_DISPATCH_GPU_RESULT_2026-09.md`. The shared mechanism
result is `results/activation_dispatch_cuda_2026-09-10.json`.

## Architecture disposition

| Architecture | Static implementation | Real GPU disposition |
|---|---|---|
| SD1.5 | complete | omitted; checkpoint absent from configured model root |
| SDXL | complete | current matrix omitted; historical SDXL measurements remain workload-specific evidence |
| Z-Image | complete | omitted; only historical run references were inventoried |
| Anima | complete | omitted by reduced-scope decision |
| Lens | complete | omitted; checkpoint absent from configured model root |
| Ideogram 4 | complete | omitted; checkpoint absent from configured model root |
| MiniT2I | complete | omitted; only historical run references were inventoried |
| Krea 2 | complete | omitted by reduced-scope decision; quantized custom-autograd path remains the important execution check |
| FLUX.2 | complete | omitted; plain/reference-latent execution remains unmeasured |
| LTX-2.3 | complete | blocked; checkpoint absent from configured model root |
| MiniMax-H3 | complete | real-transformer matrix completed; dataset/VAE/text-encoder and multi-step resume omitted |
| ACE-Step 1.5 | complete | omitted; no confirmed local audio dataset batch |
| SenseNova U1.5 | complete | omitted; flow/text/mixed real-checkpoint matrix would be long-running |

MiniMax Music 3 has no training handler and is not applicable. Standalone VAE
and tagger trainers do not inherit this integration and remain separate design
work, not missing architecture rows.

## Interpretation

- Shared plumbing is regression-tested across every constructor and workload
  family, but static reachability is not evidence of a model-specific VRAM
  reduction.
- H3 is the only newly expanded architecture for which this campaign supports
  a numerical and VRAM claim. Keep activation dispatch opt-in.
- H3 peak allocated memory fell, while peak reserved memory did not. The
  feature creates reusable in-step headroom; it does not force the CUDA caching
  allocator to return its high-water reservation.
- Candidates A--E in `TRAINING_ITERATION_VRAM_VALIDATION_PLAN_2026-09.md` are
  proposals, not implemented optimizations awaiting a final GPU test. Their
  acceptance campaigns remain future work and are not marked complete here.

## Reopening the omitted matrix

Use the contract and exact rows in
`ACTIVATION_DISPATCH_GPU_VALIDATION_BACKLOG_2026-09.md`. A future result must
name the checkpoint, dataset, dtype, bucket, method, checkpointing state, block
swap state, raw timing samples, allocated/reserved VRAM, host peak, offloaded
bytes, loss/gradient comparison, and optimizer outcome. Missing assets remain
blocked rows rather than implied passes.
