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
`v_g = u + g*(v-u)`. For a conditional item, define

`L_normal = MSE(c,v)` and `L_guided = MSE(c,v_g)`.

The default `qwen_guidance_loss_mix_mode=stochastic` draws one Bernoulli choice
per logical image, with guided probability given by the configured weight
schedule at that image's sigma. The selected loss applies to every tile of
that image. Images in the same batch can choose different losses; the batch
loss is the mean of the individually selected losses. A batch draws once before
its first forward and reuses that draw across activation-offload retries and
micro-batch splits. The `blend` mode retains
the earlier objective, `L = (1-weight)*L_normal + weight*L_guided`.

At fixed parameters, stochastic selection has the same expected gradient as
blend, but higher gradient variance; it is an experiment in optimizer
trajectory, not a guarantee of learning two separately expressible fields.
Existing saved YAML without this key is interpreted as `blend` when resumed
or opened for editing, preserving its prior guidance-loss behavior. New run
requests materialize the `stochastic` API default into their saved config.

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
text encoder and held on CPU. Guided-selected images (or images with positive
blend weight) use one no-gradient transformer forward per region. It uses
exactly the conditional branch's noisy latent, sigma, global
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

An opt-in `qwen_guidance_loss_weight_schedule=high_noise_smoothstep` changes
the guided probability (or blend fraction) per logical image, not the
guidance-target scale. The existing `qwen_guidance_loss_weight` is its
low-noise value; it transitions
to `qwen_guidance_loss_high_noise_weight` between
`qwen_guidance_loss_ramp_start` and `qwen_guidance_loss_ramp_end` using
`t²(3−2t)` with clamped normalized sigma `t`. Defaults are `constant`, 1.0,
0.5, and 0.8. In blend mode, low=0.25 and high=1.0 reproduces Run 156's mix
below sigma 0.5 and Run 158's guidance-only objective above sigma 0.8;
sigma=1 is fully guided in either mode. The schedule is experimental: the initial
high-noise rollout may remain compositionally constrained even if CFG 1 is
stable. `qwen_guidance_loss_weight=0` still disables the extra forward.

## Validation gate

Unit tests assert the exact mixed loss and gradient for full/partitioned
forwards, detached teacher gradients, prefix-mask lengths, and the ordinary
loss on CFG-null items. Before enabling on Run 153's data, compare matched
short runs at weights 0, 0.1, and 0.25 with the same seed, optimizer, tile
policy, and null-drop rate. Record peak allocated/reserved VRAM, seconds per
iter, both loss channels, and fixed-seed CFG 1 / 3 / 7 sample grids. A lower
training loss alone is not an acceptance criterion.
