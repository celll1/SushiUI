# MiniMax-H3 activation dispatch GPU result (2026-09-10)

## Verdict

Activation dispatch is numerically valid on the available MiniMax-H3 FL2VA
checkpoint when gradient checkpointing is enabled. It reduced peak allocated
VRAM by 0.84--0.86 GiB for the short clip and 4.46 GiB for the long clip. Keep
the feature opt-in: measured median overhead ranged from 1.12% to 9.97% and the
allocator's peak reserved memory did not fall.

Gradient checkpointing disabled is not a supported configuration on the tested
48 GiB card under the probe's 80% per-process safety cap. Both dispatch arms
OOMed on the short clip, with and without 40-block swap, so no long-clip test
was attempted in that configuration.

## Exercised path

- real `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` transformer;
- production `_build_transformer`, `MiniMaxH3LoRAAdapter`, and
  `minimax_h3_ops.train_step`;
- 300 rank-1 LoRA targets / 600 trainable tensors;
- joint video and audio latents, 384x640 canvas, 32 text tokens;
- short 22-pixel-frame / 7-latent-frame and long 124/37 clips;
- gradient checkpointing on, block swap 0 and 40;
- three forward/backward timing samples per dispatch arm;
- loss, every gradient, and an AdamW update computed from each captured
  gradient set;
- a same-mode OFF/OFF and ON/ON repeatability control.

Inputs use deterministic cached-record-shaped tensors. The test does not load a
dataset, VAE, or text encoder and therefore does not validate cache creation or
on-the-fly encoder residency. The AdamW check executes a real optimizer update
from the captured production-step gradients but is not a multi-step trainer
resume test.

## Gradient-checkpointed results

| Clip | Blocks swapped | Peak allocated off/on | Reduction | Median off/on | Overhead | Offloaded bytes |
|---|---:|---:|---:|---:|---:|---:|
| short | 0 | 21.972 / 21.129 GiB | 0.844 GiB | 4.244 / 4.292 s | 1.12% | 960,153,600 |
| short | 40 | 21.967 / 21.106 GiB | 0.861 GiB | 6.754 / 7.428 s | 9.97% | 960,153,600 |
| long | 0 | 27.985 / 23.522 GiB | 4.463 GiB | 33.983 / 34.883 s | 2.65% | 5,013,657,600 |
| long | 40 | 27.961 / 23.501 GiB | 4.461 GiB | 36.353 / 38.077 s | 4.74% | 5,013,657,600 |

All four conditions had exact scalar loss, finite gradients, no missing
gradient tensors, and gradients within the bf16 tolerance (`atol=1e-3`,
`rtol=1e-2`). Cross-arm maximum gradient error was 1.91e-5--3.92e-5. The
same-mode controls reached up to 6.10e-5, so the cross-arm difference is within
the observed CUDA repeatability band. Cross-arm AdamW parameter error was at
most 2.00e-4; same-mode controls reached the same 2.00e-4 band.

Peak reserved memory was unchanged for swap 0 and increased by at most 0.25
GiB for swap 40. The measured benefit is reusable allocated headroom during
the step, not a lower CUDA allocator high-water mark.

## Safety-cap results without gradient checkpointing

| Blocks swapped | Arm | Peak allocated | Peak reserved | Result |
|---:|---|---:|---:|---|
| 0 | dispatch off | 37.533 GiB | 38.316 GiB | blocked OOM |
| 0 | dispatch on | 37.995 GiB | 38.359 GiB | blocked OOM |
| 40 | dispatch off | 36.972 GiB | 38.385 GiB | blocked OOM |
| 40 | dispatch on | 38.016 GiB | 38.385 GiB | blocked OOM |

The cap was 38.39 GiB. It was not raised to consume the whole card or risk WDDM
spill. These are capacity results, not numerical failures.

## Raw evidence

- `results/minimax_h3_activation_dispatch_short_r3_2026-09-10.json`
- `results/minimax_h3_activation_dispatch_short_swap40_r3_2026-09-10.json`
- `results/minimax_h3_activation_dispatch_long_r3_2026-09-10.json`
- `results/minimax_h3_activation_dispatch_long_swap40_r3_2026-09-10.json`
- `results/minimax_h3_activation_dispatch_short_no_gc_off_oom_2026-09-10.json`
- `results/minimax_h3_activation_dispatch_short_no_gc_on_oom_2026-09-10.json`
- `results/minimax_h3_activation_dispatch_short_swap40_no_gc_off_oom_2026-09-10.json`
- `results/minimax_h3_activation_dispatch_short_swap40_no_gc_on_oom_2026-09-10.json`

Reproduce a passing condition from the repository root:

```powershell
venv\Scripts\python.exe backend\core\training\probes\minimax_h3_activation_dispatch.py `
  --model M:\model\minimax_h3 --clip long --blocks-to-swap 40 --repeats 3 `
  --out docs\audits\results\minimax_h3_activation_dispatch_long_swap40_r3_2026-09-10.json
```
