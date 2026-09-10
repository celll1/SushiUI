# Activation dispatch GPU validation backlog (2026-09)

Status: implementation and CPU/static validation complete. MiniMax-H3's
real-transformer cached-input matrix is recorded in
`MINIMAX_H3_ACTIVATION_DISPATCH_GPU_RESULT_2026-09.md`; the other rows and the
remaining H3 end-to-end dataset/encoder coverage are still open.

The 2026-09-10 campaign stopped at this reduced scope with owner approval. See
`ACTIVATION_DISPATCH_GPU_VALIDATION_SUMMARY_2026-09.md`. Open rows below are a
reproducible future backlog, not unfinished claims in the current result.

## Validation contract

For each row, run the same checkpoint, dataset batch, seed, dtype, optimizer,
checkpointing, and physical batch twice with `activation_dispatch_enable` off
and on. Record:

- loss and finite-gradient status before the optimizer step;
- peak allocated and reserved VRAM, plus whether WDDM shared-memory spill occurs;
- host working set and commit peak;
- bytes offloaded and the dispatch decision for each workload bucket;
- warm-step median and p95 iteration time, excluding model load and first-use
  compilation;
- recovery behavior for a deliberately tight bucket, including whether the
  next visit uses the learned decision without repeating the OOM.

The on/off loss and gradients must meet the repository's existing tolerance for
the selected dtype. No run may partially apply an optimizer update after an OOM,
exhaust host commit, or silently spill into WDDM shared GPU memory. Performance
results are to be reported rather than judged against an unmeasured universal
overhead target.

## Required model matrix

| Priority | Architecture | Required paths and workload variation |
|---|---|---|
| P0 | MiniMax-H3 | **Partial:** real LoRA transformer/train-step short/long and block swap off/on passed with gradient checkpointing; checkpointing-off capacity was recorded as blocked OOM. Real dataset/encoder and full trainer recovery remain. |
| P0 | LTX-2.3 | LoRA and full parameter; at least two clip lengths; block swap off/on; joint audio/video loss finite. |
| P0 | ACE-Step 1.5 | LoRA and full parameter; at least two audio durations. Confirm sequence-row keys and audio loss/conditioning equivalence. |
| P0 | SenseNova U1.5 | LoRA and full parameter; flow objective, instruction/text objective, and a mixed-objective run. Confirm independent image/text predictor histories. The current batch-size-1 contract makes micro-splitting inapplicable. |
| P1 | SD1.5 | LoRA, ReLoRA, full parameter, and ControlNet; two spatial buckets. |
| P1 | SDXL | LoRA, ReLoRA, full parameter, and ControlNet; reproduce the historical baseline before comparing new adapter reachability. |
| P1 | Z-Image | LoRA, ReLoRA, and full parameter; two spatial buckets. |
| P1 | Anima | LoRA, ReLoRA, and full parameter; include its auxiliary text-conditioning payload. |
| P1 | Lens | LoRA, ReLoRA, and full parameter; two spatial buckets. |
| P1 | MiniT2I | LoRA, ReLoRA, and full parameter; two spatial buckets. |
| P1 | FLUX.2 | LoRA, ReLoRA, and full parameter; plain and reference-latent batches. |
| P1 | Ideogram 4 | LoRA on its supported quantized base; saved-tensor census for custom autograd linears. |
| P1 | Krea 2 | LoRA and supported dense/full paths; separately measure the quantized LoRA base. |

MiniMax Music 3 is intentionally absent: it has no training handler or training
objective. Activation dispatch cannot be tested for it until training itself is
implemented.

## Composition matrix

Run the applicable model rows with these modifiers after the base on/off pair:

1. gradient checkpointing off/on;
2. block swap off/on, verifying that conductor activation offload is not stacked
   with dispatcher offload;
3. dense and quantized frozen bases where both are supported;
4. fused backward or fused optimizer groups, confirming that escalation never
   micro-splits a partially updating backward;
5. `torch.compile` full-parameter DiT paths, where supported;
6. cached and on-the-fly text/VAE/vision encoders;
7. at least two physical batch sizes where the architecture permits them.

## Calibration decisions deferred until results exist

- family- or architecture-specific cold-start coefficients for video, audio,
  and SenseNova text workloads;
- an offload-first policy for unseen MiniMax-H3 clip lengths;
- any default-on policy;
- pinned-memory or asynchronous prefetch, which changes allocator lifetime and
  stream-ordering risk and is not part of the synchronous implementation;
- extending the mechanism to standalone VAE-decoder and tagger trainers, whose
  loops do not inherit `BaseTrainer`.

Store measurements with checkpoint identity, GPU/driver/PyTorch versions, exact
training configuration, and raw per-step samples so later defaults are based on
reproducible evidence rather than one aggregate number.
