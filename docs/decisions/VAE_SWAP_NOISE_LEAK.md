# VAE swap: exposure-noise leak through Gaussian noise alignment

Status: decision, 2026-10-14. Supersedes the noise handling in
[VAE_SWAP_BRIDGE_REDESIGN.md](VAE_SWAP_BRIDGE_REDESIGN.md) for swaps whose backbone
is wider than the source latent.

## Finding

A denoiser must be a function of the noisy latent only: `v = f(x_t, t, c)`.
Gaussian noise alignment (`noise_policy = gaussian_row_orthogonal_v1`) breaks this.

- Input side: the source-channel input is corrected with the exposure noise,
  `incoming(x_t) + sigma * (W eps - incoming(eps))`.
- Output side: a frozen term built from the same `eps` is added to the
  predicted velocity.

When the backbone also receives other channels derived from `x_t` (the legacy
copy channels, or every channel under `passthrough: "full"`), the network can
rebuild `incoming(x_t)` and subtract it. That isolates a projection of the true
noise, and so of the clean latent, at any noise level.

Measured with the trained bridges (CPU, white `x0`/`eps`, linear 3x3 readout of
the 16-dimensional projection `(W - incoming) x0`):

| Bridge | sigma | R2 from `x_t` only | R2 from the bridge input |
|---|---|---|---|
| run 177, full passthrough | 0.6 / 0.9 / 0.97 | ~0 | 1.000 / 1.000 / 0.999 |
| run 176, legacy copy layout | 0.6 / 0.9 / 0.97 | ~0 | 1.000 / 0.998 / 0.983 |

How this explains run 177:

- Training loss and single-step predictions improve, because the clean latent
  is partly given.
- At inference only the initial noise exists, so the leaked channel echoes the
  model's own trajectory instead of new information.
- Run 177's samples keep colour but lose structure from about step 6k–10k, at
  CFG 7 and at CFG 3.5 alike.
- The raw-channel embedding columns grew from zero to about half the
  source-channel norm.
- Blocks 0–1 and the timestep embedding moved most (up to 20% and 7%), which
  fits learning the `1/(1 - sigma)`-scaled readout.
- The text path (LLM adapter) moved about 0.2%.

## Decision

1. A new swap installs Gaussian noise alignment only when the backbone width
   equals the source channel count (`new_channels == old_channels`). Without
   copy or passthrough channels there is no second view of `x_t` to subtract.
   Every wider layout (`new > old`, which includes all full-passthrough swaps)
   is installed without it. The model is then an ordinary flow model on the
   target latent.
2. Training refuses a model whose alignment declaration has a noise policy and
   a backbone wider than the source channels (`leaks_exposure_noise`). This
   covers run 176/177/178-style checkpoints and their resumes. Such
   checkpoints already learned to read the leak. Inference still loads them,
   because they were trained to expect the noise.
3. Patch noise alignment (MiniT2I / SenseNova pixel swaps) is unchanged. It
   replaces the noise of every bridge channel before the patch embedding, so
   the network sees one noisy observation and no second view of `x_t`.

## Consequences

- Step-0 samples of a new wide swap get worse than before. The source channels
  see the target codec's coloured noise `incoming(eps)` instead of white noise,
  and the noise complement is no longer added for free. That complement is
  what the backbone has to learn, and it now learns it from `x_t`.
- Leaked runs (176, 177, 178) must restart from the original base with a fresh
  swap initialization.
- `tools/convert_anima_full_passthrough.py` keeps the legacy noise policy, so
  its output is inference-only.

## Verification

- Unit: for a new wide declaration the bridge input is independent of the
  `noise` argument. The training refusal fires for a leaking declaration and
  not for a clean one.
- The R2 probe above, run on a fresh swap's bridge, must match the `x_t`-only
  column.
- GPU, owner's call: a short run from the base, checking that samples improve
  monotonically past step 10k at the CFG used in production.
