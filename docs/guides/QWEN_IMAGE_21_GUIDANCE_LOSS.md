# Qwen-Image 2.1 guidance-target loss

Status: experimental, opt-in (`qwen_guidance_loss_weight=0` by default).

## Objective

For a logical image, draw one noise tensor and one flow timestep. With clean
latent `x0`, noise `eps`, and `sigma`, the ordinary velocity target is
`v = eps - x0` at `x_sigma = (1-sigma)*x0 + sigma*eps`. Evaluate the current
transformer once under its inference empty-prompt condition without autograd,
giving `u = stopgrad(model(x_sigma, sigma, ""))`. The conditional prediction is
`c = model(x_sigma, sigma, caption)`.

The effective scale is `g = scale` for `constant`, or
`g = 1 + (scale - 1)*sigma` for `sigma`. Define the guidance target
`v_g = u + g*(v-u)`. For a conditional item, train

`L = (1-weight)*MSE(c,v) + weight*MSE(c,v_g)`.

An item drawn for `cfg_uncond_drop_rate` instead receives only ordinary MSE
under its empty-prompt condition. It does not receive the guidance target.
The two controls can therefore be used together: dropout trains the null
branch; guidance-target loss biases the conditional branch toward a CFG-like
field at CFG 1. This is not an equivalence guarantee for inference at either
CFG 1 or CFG > 1. In particular, larger `weight*scale` may oversteer an
already extrapolated CFG output, so paired fixed-seed sample grids at CFG 1
and higher CFG are required before adopting it for a production run.

## Execution and costs

The empty-prompt text encoding is computed once per run by the frozen Qwen
text encoder and held on CPU. Each logical step moves that small conditioning
to the device and adds one no-gradient transformer forward for each image or
region. It uses exactly the conditional branch's noisy latent, sigma, global
position IDs, and optional partition-global adapter; each branch constructs
its own prefix-length-dependent image mask. Partition loss still covers every
non-overlapping core exactly once and accumulates one optimizer step. The
no-gradient pass avoids retaining teacher activations but increases compute;
the incremental peak VRAM and iteration time require a real-device probe.

`qwen_guidance_loss_weight` accepts 0–1; `qwen_guidance_loss_scale` accepts
1–10; schedule is `constant` or `sigma`. The defaults are 0, 3, and `sigma`
from `backend/api/param_defaults.py`. Loss-chart channels report ordinary
and guided MSE separately when the guidance pass runs. Existing runs are
unchanged at weight 0. The feature is Qwen-only and requires its frozen text
encoder. It works with both full-canvas and complete-coverage partitioned
training; null-drop labels are passed through OOM micro-batch slicing.

## Validation gate

Unit tests assert the exact mixed loss and gradient for full/partitioned
forwards, detached teacher gradients, prefix-mask lengths, and the ordinary
loss on CFG-null items. Before enabling on Run 153's data, compare matched
short runs at weights 0, 0.1, and 0.25 with the same seed, optimizer, tile
policy, and null-drop rate. Record peak allocated/reserved VRAM, seconds per
iter, both loss channels, and fixed-seed CFG 1 / 3 / 7 sample grids. A lower
training loss alone is not an acceptance criterion.
