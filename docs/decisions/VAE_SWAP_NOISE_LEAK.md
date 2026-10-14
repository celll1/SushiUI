# VAE swap: exposure-noise leak through Gaussian noise alignment

Status: decision, 2026-10-14; Anima implemented, other architectures refused until ported.
Supersedes the noise handling in [VAE_SWAP_BRIDGE_REDESIGN.md](VAE_SWAP_BRIDGE_REDESIGN.md)
for swaps whose backbone is wider than the source latent.

## Finding

A denoiser's input must be a function of the noisy latent only: `v = f(x_t, t, c)`.
Gaussian noise alignment (`noise_policy = gaussian_row_orthogonal_v1`) breaks this on the
input side. It corrects the source-channel input with the exposure noise:
`incoming(x_t) + sigma * (W eps - incoming(eps))`.

When the backbone also receives other channels derived from `x_t` (the legacy copy
channels, or every channel under `passthrough: "full"`), the network can rebuild
`incoming(x_t)` and subtract it. That isolates a projection of the true noise, and so of
the clean latent, at any noise level.

Evidence:

- Linear 3x3 readout of the 16-dimensional clean projection `(W - incoming) x0` from the
  trained bridges (CPU, white `x0`/`eps`):

  | Bridge | sigma | R2 from `x_t` only | R2 from bridge input |
  |---|---|---|---|
  | run 177, full passthrough | 0.6 / 0.9 / 0.97 | ~0 | 1.000 / 1.000 / 0.999 |
  | run 176, legacy copy layout | 0.6 / 0.9 / 0.97 | ~0 | 1.000 / 0.998 / 0.983 |

- Run 177 (step 39846) relies on it. The x0 error of one forward on real latents
  depends on which noise it is given:

  | sigma | true eps | independent eps | zero |
  |---|---|---|---|
  | 0.5 | 0.060 | 0.630 | 0.349 |
  | 0.8 | 0.198 | 2.120 | 1.293 |
  | 0.95 | 0.396 | 4.014 | 2.873 |

- Training loss and single-step predictions improve, because the clean latent is partly
  given. At inference only the initial noise exists, so the readout echoes the
  trajectory. Run 177's samples keep colour but lose structure from about step 6k–10k.
- In weights, the raw-channel embedding grew from zero to half the source-channel norm.
  Blocks 0–1 and the timestep embedding moved most (up to 20% and 7%). The text path
  moved about 0.2%.

The output side is not a leak. Its residual `eps - outgoing(W eps)` is added after the
network, so the network never reads it.

## Decision

1. A wide Anima swap declares `noise_estimate: gaussian_posterior_v1`.
   - Input side: the backbone receives the alignment computed from
     `eps_hat = sigma / ((1 - sigma)^2 + sigma^2) * x_t`. This is the posterior mean of
     the noise for a standard-normal latent, a function of `x_t` alone. The same applies
     to the partition global canvas.
   - Output side: the residual keeps the exposure noise in training and the initial
     noise at inference, as before.
2. Training refuses a model whose declaration has a noise policy, no noise estimate and a
   backbone wider than the source channels (`leaks_exposure_noise`).
   - This covers checkpoints from runs 176–178 (they already learned the readout) and
     wide swaps of other architectures until they are ported.
   - Inference still loads all of them.
3. Patch noise alignment (MiniT2I / SenseNova pixel swaps) is unchanged. It replaces the
   noise of every bridge channel before the patch embedding, so there is no second view
   of `x_t`.

## Alternatives measured (Anima, run 177 config, GPU)

| Variant | Leak | Step-0 sample |
|---|---|---|
| Old: exposure noise on both sides | yes | character intact, blurred |
| No alignment | no | uniform noise; the noise complement is never removed |
| Estimate on both sides | no | faint silhouette in noise |
| Estimate on input, exposure noise on output (chosen) | no | character, colours and layout present; green blob artefacts |

## Open items

- The standard-normal prior is crude at mid noise, and the blob artefacts trace back to
  it. A per-channel, frequency-aware prior (a Wiener filter from calibration latents)
  remains a function of `x_t` and should close most of the gap.
- Port the input estimate to Z-Image, FLUX.2, Krea 2, Lens, SD1.5/SDXL, Qwen-Image 2.1
  and Ming. Their wide swaps are refused at training start until then.
- Leaked runs restart from the original base with a fresh swap initialization.
