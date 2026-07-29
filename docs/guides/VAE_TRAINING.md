# VAE Fine-Tuning

A fourth training modality alongside LoRA, full-parameter and tagger training:
fine-tune a VAE against raw images from the existing dataset system. By default
only the **decoder** trains, with the **encoder frozen**.

That default is the whole point of the design. The encoder defines the latent
distribution that every diffusion model, every trained LoRA and every cached
latent in this install was built against. Training only the decoder changes how
latents are turned back into pixels and leaves that contract intact, so the
output can be dropped into the existing inference VAE-override slot without
invalidating anything.

Encoder training exists as an opt-in behind a **double gate** and produces a VAE
that is *not* a drop-in replacement — see
[Encoder training](#encoder-training-the-double-gate).

## Where it lives

| Path | Role |
|---|---|
| `backend/core/training/vae/vae_trainer.py` | `VaeTrainer` — load, freeze, train loop, EMA, checkpoints, validation, export. |
| `backend/core/training/vae/vae_config.py` | `resolve_vae_training_config` + the refusal gate. |
| `backend/core/training/vae/vae_losses.py` | `VaeLossBank`, `PatternLoss`, `psnr`, `blockiness`. |
| `backend/core/training/vae/vae_dataset.py` | Raw-pixel dataset (random square crop to `resolution`, `[-1,1]`, `[B,3,H,W]`). |
| `backend/core/training/train_runner.py` | The `network_type == 'vae_decoder'` branch that dispatches to it. |
| `backend/core/training/training_config.py` | `TrainingConfigGenerator.generate_vae_config` — request dict to YAML. |
| `backend/api/param_defaults.py` | `VAE_TRAINING_DEFAULTS` (40 keys, the SSOT). |
| `frontend/src/components/training/vae/VaeTrainingConfig.tsx` | The config panel. |
| `backend/tests/test_vae_refusal_matrix.py` | Executable form of the toggle matrix — every accepted and refused combination. |

### Why it does not subclass `BaseTrainer`

`BaseTrainer` is a *diffusion* spine: noise scheduler, timestep sampler, SNR
weighting, prediction-target detection, text-encoder and latent caches. None of
that exists in a VAE fine-tune. More decisively, `BaseTrainer.encode_image`
(`backend/core/training/base_trainer.py:4374`) wraps its VAE forward in
`torch.no_grad()` — and gradients through the VAE forward are exactly what this
modality needs.

So `backend/core/training/vae/` is a standalone package. It does not edit
`base_trainer.py`, does not add an arch-registry key, and adds no DB tables:
`network.type` inside `config_yaml` is the discriminator, and the run reuses
`TrainingRun`, the subprocess launch, the `.stop_training` sentinel + SIGTERM
handling, checkpoint listing/resume, and the `TrainingMetrics.extra_metrics`
chart channel unchanged.

## Running one

### Config shape

```yaml
config:
  process:
    - network: { type: vae_decoder }        # the discriminator
      model:   { name_or_path: <base model> }
      datasets: [ ... ]                     # the existing dataset_configs shape
      train:                                # shared run-shape section
        batch_size: 1
        steps: 2000
        lr: 1.0e-5
        optimizer: adamw
        ...
      save:  { save_every: 500, max_step_saves_to_keep: 3 }
      vae:                                  # everything VAE-specific
        vae_source: model                   # model | path | store
        train_decoder: true
        decoder_blocks: all                 # all | up_blocks | mid_block | conv_out
        train_encoder: false                # needs acknowledge_latent_space_break too
        acknowledge_latent_space_break: false
        encoder_blocks: all                 # all | down_blocks | mid_block | conv_out
        kl_weight: 1.0e-6                   # only constructed when the encoder trains
        export_bare_ldm: false              # refused together with train_encoder
        resolution: 512
        dtype: bf16
        ema_enabled: true
        ema_decay: 0.999
        mse_weight: 1.0
        lpips_weight: 0.1
        ycbcr_dc_weight: 0.1
        pattern_weight: 0.0
        seed: 42
        num_workers: 2
        validation_every: 100
        ...
```

There is no `sample` section — a VAE fine-tune has no denoiser to sample from.

### API surface

- `POST /training/runs` with `training_method: "vae_decoder"` and the VAE knobs
  nested in the `vae_config` request field (an object with the
  `VaeTrainingDefaults` key set).
- `GET /schema/vae-training-defaults` returns the 40 defaults (fourth sibling of
  `/schema/generation-defaults`, `/schema/training-defaults`,
  `/schema/tagger-training-defaults`).
- Create / start / stop / status / metrics / checkpoints routes are the existing
  ones, unchanged.
- `resume_from_checkpoint` accepts `"latest"`, which resolves to the
  highest-numbered `checkpoints/step_*` directory. `"latest"` with no
  checkpoints starts fresh (matching the diffusion path); a *named* checkpoint
  that does not exist raises and lists what is available, because silently
  restarting a long run from step 0 is a worse failure than an error.

### Output

`save_diffusers_vae` writes `<run_name>_vae/` as a diffusers directory
(`vae.save_pretrained`), which the existing inference VAE-override path loads
unchanged; the compatibility gate passes because `latent_channels`,
`latent_ndim`, class family and spatial scale are all untouched by a
decoder-only fine-tune, and `save_pretrained` preserves `scaling_factor` /
`shift_factor`.

When EMA is on, a **`<run_name>_vae_noema/` sibling** is always written too. The
EMA copy of a short or user-stopped run can be dominated by the base weights,
and the loss/PSNR/blockiness charts cannot show that — they measure the *live*
weights, not the EMA copy that gets exported. Both directories carry a
`sushi_vae_training.json` provenance sidecar (base VAE identity, run id, step,
loss config, `train_decoder` / `decoder_blocks`, `encoder_trained` /
`encoder_blocks` / `kl_weight`, `ema_applied`, `ema_retained_init_fraction`).
The trainer prints that retained fraction and warns loudly above 0.5.

With `export_bare_ldm: true` (off by default, `AutoencoderKL` only) a bare
LDM-format `<run_name>_vae.safetensors` is written alongside the directory, via
`convert_vae_state_dict_to_original`. It is read back from the diffusers export
that was just written, so it always carries the same weights (EMA or live) as
its sibling directory. It has no `config.json`, so whatever loads it supplies
`scaling_factor` / `shift_factor` — correct only while the encoder is frozen,
which is exactly why it is refused for encoder-trained runs.

## Encoder training (the double gate)

`train_encoder: true` alone does nothing but fail: it must be accompanied by
`acknowledge_latent_space_break: true`, and the acknowledgement on its own (with
the encoder frozen) is refused too, so it cannot be left set in a config and
silently authorise a later run. The panel enforces the same pair — the
acknowledgement is a separate, unchecked checkbox that only appears once encoder
training is switched on, and switching it back off clears it.

What changes when the encoder is trainable:

| | Decoder-only (default) | Encoder trained |
|---|---|---|
| Latent used for the decode forward | `latent_dist.mode()`, under `no_grad` | `latent_dist.sample()`, with gradients |
| KL term | not constructed (constant w.r.t. every trainable parameter; `kl_weight` is ignored and the trainer logs that) | constructed at `kl_weight` (default 1e-6 = LDM's balance, applied to a per-element-normalised KL — see [the loss bank](#why-the-kl-is-normalised-before-it-is-weighted)); the weighted contribution is charted as `vae_kl_loss` |
| Export directory | `<run_name>_vae/` (+ `_noema`) | `<run_name>_vae_encoder_trained/` (+ `_noema`) |
| Sidecar | `encoder_trained: false` | `encoder_trained: true`, plus `encoder_blocks` and `kl_weight` |
| `export_bare_ldm` | allowed | refused |
| Granularity | `decoder_blocks` | `decoder_blocks` **and** `encoder_blocks` (`all` / `down_blocks` / `mid_block` / `conv_out`; `all` includes `quant_conv`, the encode-side mirror of `post_quant_conv`) |

**Why the directory name differs.** The structural compatibility gate
(`api/generation_overrides.py:334-403`) still passes for an encoder-trained VAE —
`latent_channels`, `latent_ndim`, class family and spatial scale are unchanged —
so nothing downstream can detect the one thing that *did* change. A `_vae`
directory is therefore reserved for "same latent space as its base model", and
an encoder fine-tune gets a name that says otherwise in a directory listing,
with the sidecar's `encoder_trained` flag as the machine-readable form of the
same fact. The trainer also prints the consequences at selection time and again
at save time.

The consequences, stated once: latents cached with the base VAE do not match
this one and must be re-encoded; LoRAs and diffusion checkpoints trained against
the base VAE were trained on latents this VAE does not produce. Nothing in the
run detects or repairs that — the acknowledgement is the entire mechanism.

`kl_weight` deliberately does **not** count towards the "at least one loss
weight above 0" check: KL regularises the posterior, it is not a reconstruction
signal, and a run with every reconstruction weight at 0 would minimise the total
by collapsing the posterior.

### Charts

`vae_recon_loss`, `vae_lpips_loss`, `vae_dc_loss`, `vae_pattern_loss` and
`vae_kl_loss` (encoder runs only) on the main axis; `vae_val_psnr` and
`vae_val_blockiness` on the right axis. `vae_kl_loss` is the **weighted
contribution**, which shares the magnitude of the other loss components — the
raw KL is 1e4–1e5 and on the right axis would flatten the PSNR/blockiness
curves that are the only quality signal this modality has
(`backend/core/training/metric_registry.py`). Validation runs every
`validation_every` steps on a fixed held-out split. **That chart is the only
signal that a fine-tune is going wrong** — a decoder fine-tune has no sample
images and no obvious loss landmark.

## The loss bank, and why each default is what it is

`L_total = Σ wᵢ·Lᵢ`. The defaults below are **derived from measurement on this
install**, not copied from a paper, and each one has a stated falsification
criterion that a future maintainer can re-run rather than re-argue.

> **Measurement provenance for every number in this section.** A Phase 0
> measurement campaign run 2026-07-28: inference-only, fp32, `torch.no_grad()`,
> single GPU, no training, no backend. Four VAEs — two independent SDXL
> `AutoencoderKL` checkpoints, `AutoencoderKLQwenImage` (the VAE Anima/Krea2
> use), and the 16-channel FLUX.1 `AutoencoderKL` (the one Z-Image uses).
> Sample sizes are **n = 1–3 images per cell** except where stated. The working
> notes were untracked, so the numbers are restated here rather than linked.

| Term | Default | Why that value |
|---|---|---|
| `mse_weight` | **1.0** | The base term of `stabilityai/sd-vae-ft-mse`, the only published, shipped recipe of this exact shape (decoder-only, encoder frozen). |
| `lpips_weight` | **0.1** | ft-MSE's published value. Per its model card, ft-MSE differs from ft-EMA in exactly two ways: L1→MSE and LPIPS 1.0→0.1. LPIPS is the term that *creates* plausible high frequency, so 1.0 would work against the artifact this fine-tune targets; 0.1 is the one published lever for suppressing it. |
| `ycbcr_dc_weight` | **0.1** | Per-pixel Charbonnier on YCbCr (luma down-weighted, `ycbcr_dc_y_weight` 0.25) **plus** a Charbonnier on the per-image per-channel spatial mean, under the same weight. |
| `pattern_weight` | **0.0** (available) | The 8 px latent-grid artifact it targets **measured absent**. |
| `l1_weight` | **0.0** (available) | The LDM / ft-EMA reconstruction term; usable instead of or alongside MSE. Preference, not measurement. |
| `kl_weight` | **1e-6** (encoder runs only) | LDM's value, applied to a **per-element-normalised** KL so that it means the same thing here — see below. Not constructed under a frozen encoder, where it is a constant w.r.t. every trainable parameter. |

### Why the KL is normalised before it is weighted

LDM's `contperceptual.py` pairs `kl_weight` with a reconstruction term that is
**summed** over `C·H·W` per image (`nll_loss = torch.sum(nll) / nll.shape[0]`).
Every reconstruction term in this bank is **mean**-reduced over `B·C·H·W`. The
two conventions differ by exactly `C·H·W` — about 786k at 512 px — so pasting
LDM's 1e-6 onto a per-element reconstruction makes the KL roughly five orders of
magnitude too strong. Measured on an SDXL VAE at 512 px before the fix:
`kl_term = 0.519` against `mse = 0.034`, i.e. **15× the reconstruction term** —
a run that was ~90% "pull the posterior to N(0, I)". It was also
resolution-dependent (4× from 256→512), so the knob did not mean one thing.

`VaeLossBank.forward` therefore divides by the per-image element count before
weighting. After that, `kl_weight = 1e-6` is LDM's balance in LDM's sense and is
resolution-invariant. Measured after the fix, same VAE family, two crops:

| res | `mse` | raw `kl` | weighted `kl_term` | `kl_term / mse` |
|---|---|---|---|---|
| 256 | 0.01149 | 68 983 | 3.51e-7 | 3.1e-5 |
| 512 | 0.00537 | 310 637 | 3.95e-7 | 7.4e-5 |

At LDM's balance the KL is a very weak regulariser by design — that is what
makes an LDM-style VAE's latent nearly deterministic. It is present to stop the
posterior drifting, not to shape it.

The per-step log prints the **raw** KL (comparable with the literature and other
trainers) and the weighted contribution; only the weighted one is charted.

### Why `ycbcr_dc` carries a spatial-mean term the design did not specify

The original design specified only the per-pixel form. The defect actually
measured is a **spatial-mean drift**: over 8 encode/decode roundtrips on a
saturated red field, the mean-colour drift reached **51.1 /255** on one SDXL VAE
and **39.0 /255** on the second (MEASURED, n = 1 stimulus per VAE, max over RGB
channels; Qwen 1.93, FLUX.1 15.42 on the same stimulus). A per-pixel penalty
constrains a mean drift only as one residual among ~250k. With the added term a
pure DC shift now scores **1.997×** a zero-mean perturbation of equal per-pixel
magnitude; under the per-pixel-only form the two scored identically by
construction. The `[-1,1]` clamp on the reconstruction side was also removed,
because it zeroed the gradient precisely on the overshooting pixels most likely
to be drifting.

This measurement also overturned a smaller prior figure (a recorded ~27 /255).
The structure, ordering and SDXL:Qwen ratio (20–26×) reproduced; the absolute
magnitude did not, because the earlier run used a different checkpoint's VAE.
The drift is a per-weights property, not an architecture constant.

### Why `pattern_weight` ships at 0

**Falsification criterion, set in advance:** if edge blockiness is < 1.5 even on
SDXL, the pattern term comes out of the defaults.

**It fired.** The 8 px latent-grid artifact measured at ratio **≈1.0** across
four VAEs, under three independent metric definitions (a cell-boundary-phase
energy ratio, an assumption-free per-phase max/min, and the classic JPEG
across-boundary/within-cell step ratio), on synthetic *and* real *and*
generated-latent inputs. SDXL edge values: 1.05 / 1.40 / 0.99. This is a
**non-reproduction of an earlier internal record of 3.0–5.5**, not a refutation
of it — the original raw artifacts and exact metric definition are gone. Two
mechanisms that produce a spurious factor of 2–3 were identified and documented
in the harness (phase-locked stimulus edges; a bicubic-upsampled noise texture
injecting a period-4 profile).

What *did* reproduce is the other half of that record: SDXL's edge residual is
**3.7×** Qwen's (9.44 vs 2.54 /255 on a line stimulus; 1.7× on real photos).
That is real, but it is broadband softness/ringing at edges, not grid structure,
so a grid-phase penalty is the wrong instrument for it — MSE + LPIPS already
target it. The term stays available as opt-in for anyone who can demonstrate the
artifact.

### What is deliberately NOT built

None of the following exists. They are listed so nobody assumes otherwise.

- **GAN (adversarial) loss.** Every published `disc_start` (e.g. 50001)
  presumes a run 5–25× longer than a single-GPU fine-tune, and it exists
  precisely because a fresh discriminator is a noise gradient source. Such a run
  either never reaches `disc_start` (dead weight) or enables immediately and
  feeds a random discriminator's gradients into the decoder. There is no
  published schedule for this regime. Adding a GAN to a short fine-tune is the
  single most reliable way to make it worse.
- **Crop-consistency loss.** It was gated on a measurement, and the gate closed
  it: the defect it targets is removed by a free, training-free, inference-time
  change (context margin + global GroupNorm statistics), leaving a residual of
  **0.03–0.16 /255** — 6–50× below the design's own >1/255 visibility bar. See
  `docs/guides/VAE_DECODE_BEHAVIOR.md`. It would also double the decoder forward
  per training step.
- **Invented-HF loss.** Its gate *passed* (the decoder does invent high
  frequency — see the decode guide), but it was scoped out of Phase 1. It is the
  one loss term with a live case for being built later.
- **PiD `lq_proj` training** (`network.type: pid_decoder`, design.md §6).
  Designed (Phase 3), not implemented.
- **Encoder-only training with a frozen decoder**, and **`decoder_blocks` /
  `encoder_blocks` as free-form module lists.** The first is refused on purpose
  (see the refusal table); the second is a fixed four-value enum on each side,
  matching ai-toolkit's `blocks_to_train`.

## Refusals

Every refusal fires at **config-resolution time**, before a model load or a
single step, and carries an actionable message. A VAE fine-tune that silently
trains nothing, or silently breaks the latent contract, is far worse than one
that will not start. All live in `vae_config.py::_validate` unless noted.

| Refused | Why |
|---|---|
| `train_encoder: true` without `acknowledge_latent_space_break: true` | Half of the double gate. Encoder training moves the latent distribution; it must be asked for twice. |
| `acknowledge_latent_space_break: true` without `train_encoder: true` | The other half. A left-over acknowledgement must not be able to authorise a run that did not ask for encoder training. |
| `train_encoder: true` with `train_decoder: false` | Encoder-only under a frozen decoder: the only way to reduce a reconstruction loss through a decoder that cannot adapt is to deform the latent distribution to suit it. |
| `train_decoder: false` and `train_encoder: false` | Nothing trainable. |
| `export_bare_ldm: true` with `train_encoder: true` | A bare `.safetensors` has no `config.json`, so the consumer inherits `scaling_factor` / `shift_factor` from the model it is loaded into — precisely what an encoder fine-tune invalidates, with no way for the consumer to notice. Enforced twice: at config resolution, and again on the write itself (`vae_trainer.py::save_bare_ldm_safetensors`). |
| `export_bare_ldm: true` on a non-`AutoencoderKL` VAE | The LDM key mapping in `adapters/state_dict_converter.py` is that architecture's. Raised at save time (the diffusers export is unaffected). |
| A gate key that is not an interpretable boolean (`train_decoder`, `train_encoder`, `acknowledge_latent_space_break`, `export_bare_ldm`, `ema_enabled`) | See [Strict booleans](#strict-booleans-on-the-gate-keys). |
| Resume from a checkpoint that trained a different component set | `vae_trainer.py::_assert_component_set_matches`, before any weight is loaded. A checkpoint holds exactly the parameters that were trainable when it was written, plus optimizer and EMA state indexed by that same set. Both directions were previously silent-ish: a decoder-only checkpoint resumed with the encoder on failed with a message blaming `decoder_blocks`, and the reverse loaded happily (the checkpoint is a superset) and then failed opaquely inside the optimizer state load — or not at all, if `optimizer.pt` was absent. |
| `dtype: fp16` | SD1.5/SDXL-family VAEs overflow fp16 in decoder activations (the documented reason `sdxl-vae-fp16-fix` exists), and a training forward hits it sooner than inference. For every other family there is no `GradScaler` in this trainer, so fp16 gradients would silently underflow instead. `bf16` (default) and `fp32` are allowed. |
| `latent_encoding_mode: pre_encoded_cache` | VAE training is *defined* by a live encode→decode forward on raw pixels; there is no cached latent to consume. Mirrors the existing outpaint-ControlNet refusal. |
| All loss weights 0 | No training signal. |
| `lpips_weight > 0` with `lpips` not importable | Fails before the run starts, never mid-run. |
| Unknown key in `vae_config` / `process.vae` | Caught in `generate_vae_config` and in `resolve_vae_training_config`; a typo must not silently resolve to the default. |
| Out-of-enum `vae_source` / `decoder_blocks` / `encoder_blocks` / `dtype` / `lpips_net`; empty `vae_path`; `vae_source: store` with no `vae_arch`; `resolution` not a multiple of 8 or < 64; `batch_size`/`total_steps`/`gradient_accumulation_steps` < 1; `ema_decay` outside (0,1); negative or non-numeric loss weight or `kl_weight` | Ordinary validation, same fail-early principle. |

### Strict booleans on the gate keys

`bool("false")` is `True`. A YAML that quotes its booleans —
`train_encoder: "false"` — is entirely ordinary (editors, templating and
hand-quoting all produce it), and under a bare cast it would **silently enable
encoder training**, i.e. open the double gate by accident; `export_bare_ldm:
"false"` would silently write the bare file. Every key that decides *what is
trained* or *what is written* therefore goes through
`vae_config.strict_bool()`: real booleans and 0/1 pass through, `true/yes/on/1`
and `false/no/off/0` are accepted in any case with surrounding whitespace, and
anything else raises rather than being guessed at. The same parser runs in
`VaeTrainer.__init__`, which is what protects callers that bypassed the
resolver.

### The tests

Every row above is asserted in **`backend/tests/test_vae_refusal_matrix.py`**
(48 cases), which is the executable form of design.md §4:

```
venv/Scripts/python.exe -m pytest backend/tests/test_vae_refusal_matrix.py -v
```

It also asserts that the matrix rows which are *not* built (PiD, GAN,
crop-consistency, invented-HF) have no config surface at all, so asking for them
lands in the unknown-key refusal rather than being silently ignored. Note the
file needs a `!` negation in `.gitignore` (which ignores `test_*.py` globally) to
stay tracked; a second checked-in test needs the same.

Two properties of that suite are load-bearing and easy to lose:

- **Every case runs at `lpips_weight: 0`.** The default is 0.1, and `_validate`
  imports `lpips` above 0 — so with the default in place, an environment without
  `lpips` turns several *refusal* rows green for the wrong reason (the lpips
  refusal fires first and the substring assertion still matches). One dedicated
  case turns LPIPS back on and asserts both environments.
- **Each refusal row asserts a fragment only its own guard emits.** Matching key
  names is not enough: a mutation sweep found that deleting the
  encoder-only-under-frozen-decoder guard went undetected, because the
  "Nothing to train: `train_decoder=false` and `train_encoder=false`" message of
  a *different* guard happens to contain both key names — and likewise that
  asserting `"fp16"` was satisfied by the out-of-enum `dtype` message that
  follows it. With distinctive fragments, neutralising each of the 11 guards in
  turn is caught 11/11.

## Two structural invariants

Both were found the hard way and will silently corrupt runs if violated.

### 1. A config key may live in `process.train` / `process.save` **only if it is also a `TrainingRunCreateRequest` field**

`GET /training/runs/{id}/params` rebuilds the edit form with a schema-driven
extractor that walks the request schema. A key written into `process.train` that
is not a request field cannot be rebuilt, so it is silently lost on the
create → `/params` → PUT-regenerate round-trip and reverts to its default on
every edit-form save. This is why `seed` and `num_workers` live in
`process.vae` (which `/params` carries verbatim) despite being run-shape-ish.
The rule is written next to `run_shape_keys` in
`training_config.py::generate_vae_config`.

Related: `ema_decay` exists *both* as a flat request field (diffusion full-FT
EMA) and inside `process.vae`. Regeneration was always correct (nested wins),
but `/params` reported two different decays for the same run, so which value a
form saw depended on which field it read. The nested value is now mirrored onto
the flat field for VAE runs.

### 2. `_explicit_fields` / `model_fields_set` — nest run-shape keys inside `vae_config`

`generate_vae_config` resolves each key as:

```
vae_config[key]  >  request[key] (only if the caller explicitly set it)  >  VAE_TRAINING_DEFAULTS[key]
```

The middle tier is gated on `request.model_fields_set`, passed through as
`params_dict["_explicit_fields"]`, because **`request.model_dump()` materialises
every Pydantic default as a non-None value**. Without the gate that tier is not
"what the caller sent" but "the diffusion trainer's defaults, unconditionally" —
which overrode five VAE defaults, including `learning_rate` 1e-5 → 1e-4 and
`optimizer` adamw → adamw8bit. The run would have completed and simply wrecked
the decoder.

**Consequence for any caller (UI, script, agent):** a key sent as a *top-level*
request field is treated as deliberate and overrides the VAE default. Send VAE
run-shape keys **nested inside `vae_config`**. The shipped panel sends exactly
seven top-level keys (`dataset_configs`, `run_name`, `training_method`,
`base_model_path`, `total_steps`, `resume_from_checkpoint`, `vae_config`); the
only one intersecting the 36 VAE keys is `total_steps`, which is safe because
both copies come from one expression and nested wins. `epochs` is never sent —
a nested `vae_config.total_steps` outranks any epoch-derived step count, so an
epochs control would silently do nothing.

## Known limits

- **Only the 4-channel `AutoencoderKL` family has been exercised.** The loader
  is generic over any diffusers `Autoencoder*`, but nothing else has been run.
- **The base-model picker filters when `vae_source: "model"`.** Anima and Krea2
  use `AutoencoderKLQwenImage` and LTX-2 uses `AutoencoderKLLTXVideo` — both
  unpack `(B, C, num_frames, H, W)` in `_encode` while this trainer feeds a 4-D
  pixel batch — and MiniT2I is pixel-space with no VAE at all. It is a
  **deny-list** (`NON_TRAINABLE_VAE_ARCHS` in `VaeTrainingConfig.tsx`), not an
  allow-list, so a future architecture is not silently hidden. With
  `vae_source: "path"` or `"store"` the full model list stays available, so an
  Anima base model with an explicit SDXL VAE remains configurable.
- **`vae_source: "store"` with `vae_arch: "sdxl"` resolves to
  `madebyollin/sdxl-vae-fp16-fix`**, whose fp16 safety comes from a weight
  rescaling that fine-tuning does not preserve. The trainer warns when it
  detects that base; it is not the default (`vae_source` defaults to `"model"`).
- **Scale.** `sd-vae-ft-mse` was batch 192 × ~840k cumulative steps. A single
  GPU at batch 1–4 × 1–2k steps is orders of magnitude below that. What is
  reachable is local adaptation to your data distribution and suppression of
  specific artifact terms — not re-learning the distribution. No prior work
  measures Δ-blockiness or Δ-crop-consistency for a fine-tune of this size, so
  no effect size is promised.
- **Encoder training is verified, not validated.** The smoke run below proves
  the mechanism (encoder tensors move, KL is finite and charted, the artifact is
  labelled, the export refusals fire). Whether an encoder fine-tune of this
  scale *improves* anything is unmeasured, and there is no published recipe of
  this shape to copy — `sd-vae-ft-mse`, the one shipped precedent, froze the
  encoder precisely to avoid the question.
- **Verification level (encoder path, Phase 2).** A 3-step smoke on the sd15
  `vae-ft-mse-840000-ema-pruned` VAE at 256 px, lr 1e-4, EMA off: **248**
  trainable tensors / 83.65M params (106 encoder + 138 decoder + 2 `quant_conv`
  + 2 `post_quant_conv`), max|Δ| 3.008e-4 on the encoder, 3.005e-4 on the
  decoder, 2.839e-4 on `quant_conv`; KL finite (68143 → 71931 → 76472 raw) with
  a weighted contribution of ~3.5e-7 against `mse` ~1.4e-2, i.e. the objective
  is reconstruction; export written to `<run>_vae_encoder_trained/` with
  `encoder_trained: true` in the sidecar and reloadable via
  `AutoencoderKL.from_pretrained`; both bare-LDM refusals fired (config
  resolution and the write itself), both single-gate refusals fired, and the
  resume component-set guard refused in both directions while accepting a
  matching set. The decoder-only control in the same run: encoder max|Δ|
  **exactly 0.0**, `kl_weight` reported as ignored, 140 trainable tensors, and a
  248-tensor bare-LDM file written that reloads with no key mismatches.

  Note the KL *rises* slightly over these three steps. Before the normalisation
  fix it fell 27% in three steps (68143 → 50044), which was not the mechanism
  working — nothing reconstruction-driven moves 27% in three steps — but the
  posterior collapsing under a KL that outweighed the MSE 15:1.
- **Verification level (Phase 1).** Shipped after a 20-step smoke run on a real dataset
  and a real SDXL checkpoint (finite losses; 140/140 trainable tensors moved,
  max|Δ| 1.9e-3; zero encoder/`quant_conv` tensors trainable and encoder
  max|Δ| exactly 0.0 in both exports; checkpoint round-trip and resume; stop
  sentinel honoured; exports load via `AutoencoderKL.from_pretrained`). **No
  convergence run has been done**, consistent with this repo's minimal-training
  verification convention.

## See also

- `docs/guides/VAE_DECODE_BEHAVIOR.md` — what the decoders measurably do at
  inference time (non-locality decomposition, tiling, invented high frequency).
- `docs/guides/ADD_A_PARAMETER.md` — the parameter checklist, including the
  `model_fields_set` and dtype-coverage lessons this work produced.
</content>
</invoke>
