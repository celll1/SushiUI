# Training iteration latency refactor plan

## Goal

Reduce avoidable latency on the diffusion-training iteration path without
changing losses, gradients, optimizer updates, batch order, cache contents, or
checkpoint data. Verification in this pass must not initialize CUDA or restart
the backend/frontend.

## Confirmed hot-path work

1. Architecture `train_step` implementations convert metric tensors to Python
   scalars before the shared backward call. A CUDA scalar read waits for the
   forward stream and delays backward submission.
2. The per-iteration progress callback performs a synchronous `TrainingRun`
   query and database commit even though detailed metric writes already use a
   background worker.
3. When text and latent swap buffers expire together, the main model is moved
   CPU-to-GPU twice: once around each refill.

## Planned commits

1. Defer architecture metric scalar extraction until after the shared backward
   synchronization point. Keep each metric expression and returned value
   unchanged.
2. Move progress persistence off the training thread behind a latest-value
   reporter with an explicit final flush. Preserve phase calculations and final
   state while bounding UI staleness.
3. Coordinate simultaneous text/latent swap refills so the main model is
   offloaded and restored once. Preserve text-before-latent encoding order and
   progress events.

Each implementation unit receives its own tests and commit.

## Deferred work

- Do not change bucket/MNT `torch.cuda.empty_cache()` policy without GPU timing,
  peak-reservation, shared-memory spill, and OOM measurements.
- Do not introduce pinned-memory staging without measuring host-memory pressure
  and transfer overlap on the target GPU.
- Do not replace gradient clipping with a custom reduction in this pass because
  accumulation order can change the clip coefficient.

## CPU-only verification

- Compile every changed backend module.
- Import changed trainer modules with CUDA initialization triggers stubbed.
- Run focused unit tests with CPU tensors and fake database/model objects.
- Run the existing training regression tests that cover affected contracts.
- Confirm the worktree contains only the intended unit before each commit.
