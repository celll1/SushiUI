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
| `backend/core/training/vae/vae_dataset.py` | Raw-pixel dataset (random square crop to `resolution`, `[-1,1]`, `[B,3,H,W]`) + the [crop scale policy](#crop-scale-policy-which-pixels-the-decoder-sees). |
| `backend/core/training/train_runner.py` | The `network_type == 'vae_decoder'` branch that dispatches to it. |
| `backend/core/training/training_config.py` | `TrainingConfigGenerator.generate_vae_config` — request dict to YAML. |
| `backend/api/param_defaults.py` | `VAE_TRAINING_DEFAULTS` (42 keys, the SSOT). |
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
        crop_scale_policy: downscale        # downscale | native | mixed
        crop_scale_max_downscale: 0.0       # bounds the 'mixed' draw; 0 = unbounded
        dtype: bf16
        ema_enabled: true
        ema_decay: 0.999
        mse_weight: 1.0
        lpips_weight: 0.1
        ycbcr_dc_weight: 0.1
        pattern_weight: 0.0
        l_invented_weight: 0.0              # flat-region invented-HF penalty (opt-in)
        seed: 42
        num_workers: 2
        validation_every: 100
        validation_resolution: 1024
        ...
```

There is no `sample` section — a VAE fine-tune has no denoiser to sample from.
Nothing is ever written to the run's `samples/` directory, and the training
monitor's Samples tab says so instead of polling for images that cannot appear;
reconstruction quality is read off the validation PSNR / blockiness chart.

### Dataset loading is pixels-only

`VaeRawImageDataset` reads `image_path` and nothing else, so `train_runner`
loads the datasets for a `vae_decoder` run with `skip_captions=True`
(`train_runner.py::get_dataset_items_fast` / `get_dataset_items_cached`): the
caption table is not joined, no primary caption is selected per item, and the
per-epoch caption-processing pass is skipped entirely. The item dicts keep the
same keys (`raw_caption` is `""`, `tag_data` is `None`), the missing-file skip
and the cache-invalidation key are unchanged, and the pixels-only cache is
stored under a distinct `_nocap` cache key so it can never be picked up by a
text-conditioned run. (The one shape difference: an `item_type == "audio"` item
gets no `lyrics` key, because producing it would mean touching `item.captions`
and reintroducing the N+1 this avoids. VAE training is image-only.) The flag is
set only for `network.type == "vae_decoder"` — the four diffusion methods and
the tagger keep the full caption pipeline.

Because the cache key changed, a config that ran before this existed has
caption-bearing pickles in its own `.dataset_cache/` that can never be read
again (run 113 left 8.4 GB). A pixels-only run therefore prunes every
`dataset_*.pkl` in its OWN run directory that is not a `_nocap` file, once, at
startup (`train_runner.py::_prune_captioned_dataset_caches`); the cache dir
belongs to exactly one run, so nothing else can depend on them.

This matters at scale: measured over three datasets on the run-113 config, the
item-loading phase dropped from 8.20s to 2.38s (43k items), 42.07s to 5.22s
(101k items) and 27.28s to 10.35s (199k items).

### Crop scale policy: which pixels the decoder sees

Every crop is square, at `resolution`, with the aspect ratio preserved (nothing
is squashed) — randomly placed for training, centred for validation.

"Randomly placed" means **re-placed on every visit**: the training crop RNG is
keyed by `(seed, item index, visit)`, where `visit` is the data-pass counter that
`VaeEpochCropSampler` yields alongside the index (`vae_dataset.py`), so the same
image gets a different window — and, under `mixed`, a different scale factor —
each time it comes round. The counter is checkpointed (`data_epoch` in
`train_state.json`) and a resume continues into the next pass, so the stream is
reproducible for a seed and independent of `num_workers`. Until 2026-07-31 the
key was `(seed, item index)` only, which silently gave each image exactly one
crop and one `mixed` factor for the entire run.

`crop_scale_policy` decides how much the image is **resampled before that crop**,
which the crop-geometry study
(`scratchpad/vae_training/results_crop_geometry.md`) measured to be the dominant
control on what the fine-tune learns.

Every figure in the table below is from `results_crop_geometry.md` **§8**
(n = 400 items, `resolution: 512`, dated 2026-07-30, reproducible via
`scratchpad/vae_training/harness/crop_policy_verify.py`; method, sampling rule
and limitations recorded there). Loader cost is §8.4 and is quoted as a **ratio**,
because the absolutes move with machine load.

| policy | geometry | realised downscale factor (§8.2) | loader cost (§8.4) |
|---|---|---|---|
| `downscale` (**default**) | short side scaled to exactly `resolution`, up or down | median **2.17x**, mean 2.51x, p95 5.35x; **95.0%** downscaled, 0.5% already exactly at `resolution` | baseline |
| `native` | crop out of the full-size pixels; upscale only when the short side is genuinely below `resolution` | **95.5% at exactly 1x**; the remaining 4.50% are upscaled | **−42 to −43%** (no LANCZOS pass over a multi-megapixel image) |
| `mixed` | draw the factor per sample **and per visit**, log-uniformly over `[1, f_max]` | median **1.35x**, mean 1.60x, p95 3.13x; the whole range is covered, 1x is a limit of the support | +14 to +16% |
| `mixed` + `crop_scale_max_downscale: 2.0` | as above with `f_max` capped | median 1.28x, max 1.99x | ~+15% |

The corpus-wide census is **§1.2**, not this table: over all 3,842,897 items
95.79% are downscaled by a median 2.30x and 4.21% are upscaled. §8's 95.0% /
2.17x / 4.50% is the same quantity on a 400-item sample and agrees within
sampling noise. The `openapi.yaml` and `param_defaults.py` comments quote §1.2
and §5.2's loader absolutes (30.8 / 17.4 ms, n = 120) — the same measurement on a
different sample and a less loaded machine; its 44% ratio is what §8.4
reproduces.

`crop_scale_max_downscale` bounds `f_max` for `mixed` (`0` = the image's own
`short/resolution`). It is **refused** under any other policy rather than
silently ignored, since only the per-sample draw reads it; the panel clears it
when you switch away from `mixed`.

**Why this is a knob at all.** The historical behaviour downscales 95.79% of the
corpus (3,842,897 items) by a median 2.30x. The original suspicion was that this
starves the decoder of high frequency; measured, it is the **opposite** — a
LANCZOS downscale *concentrates* high frequency, so the production crop carries
**4.06x** the top-octave power of a native crop (n=300, 93.3% of images,
t=+21.6), and the fine-tune consequently softens native content ~8-9x *less*
than training-distribution content (−0.44% vs −3.79% Sobel). The real, measured
cost is **calibration**: the fine-tune's accuracy gain is ~30% smaller on
native-resolution content (edge residual −7.7% vs −12.5%, positive on 19/19,
t=+7.49; PSNR +0.81 vs +1.15 dB), because it never sees any. That is what
`native` / `mixed` address.

**Why the `mixed` draw is log-uniform, not linear.** The dose-response is
monotone in the downscale factor and inference presents ~1x, so the mass belongs
near 1x. Under linear-uniform sampling the realised distribution would depend
strongly on source size — a 5120 px source (`f_max` = 10) would put 90% of its
draws above 2x while a 600 px source puts all of its draws near 1x — dragging
the corpus towards exactly the heavily-resampled regime the knob exists to get
away from. Log-uniform gives equal weight per octave of resampling: the median
draw is `sqrt(f_max)`, which is why the measured `mixed` median lands at 1.35x
against `downscale`'s 2.17x. Nothing about the numbers *forces* log-uniform —
the family (vs linear-uniform, beta, or a two-point mixture) is the one
judgement call the study did not settle, and it is recorded as such in
`results_crop_geometry.md` §8.2.

**Cost.** None: the loader has 8-30x headroom over the GPU either way (measured
loader wait 0.15 ms/step, 0.04%, `results_batchsize.md`), VRAM and step time are
untouched (the tensor shape is identical), and `native` is *cheaper* than the
default. `mixed` is ~15% dearer than `downscale` because it resizes to an
intermediate size *larger* than the crop — `results_crop_geometry.md` §5.3
predicted its cost would sit between the other two rows and that turned out to be
wrong (§8.4) — still ~10x the GPU's demand at 2 workers.

**The default stays `downscale`**, so no existing run changes the *geometry* it
trains on: run 113 has 52k steps of history under that geometry, and `downscale`
is **pixel-identical** to the pre-policy loader (`results_crop_geometry.md` §8.3:
`torch.equal` on 400/400 real dataset images in both random-crop and centre-crop
mode — the branch reuses the same `resolution / min(w, h)` expression, and
`resolve_crop_scale` returns before touching the RNG on both non-`mixed` paths,
so the crop-offset draw sees an unchanged stream).

**That claim is about the resample geometry of a policy, not about which crop
window a given image gets.** The crop-offset *sequence* did change on
2026-07-31, for every policy including `downscale`: the RNG seed derivation went
from `(seed * 1000003) ^ index` to `mix_seed(_DOMAIN_CROP, seed, index, visit)`
in order to make re-visits move the window (see above). Re-keying moves the very
first draw too, so even visit 0 lands elsewhere than it used to: for items 0-4 at
`seed: 7`, with 100 px of slack on both axes, the old key drew the offsets
`(19,91) (81,79) (91,51) (22,89) (99,40)` and the current one draws
`(61,5) (45,21) (69,4) (67,67) (62,93)`. A run resumed across that date
therefore continues on **different crop windows** (the intended fix: the old key
pinned each image to one window for the entire run). What is unchanged is the
resampling each policy applies before the crop, and the validation batch, which
uses no RNG at all.

**Validation ignores the policy, deliberately.** `make_validation_batch` is
pinned to `downscale` and takes no policy argument at all, so:

- it stays deterministic (no RNG at all, no visit counter, centre crop — `mixed`
  would redraw per call and make the held-out series noisy for a reason unrelated
  to the model), and
- `vae_val_psnr` keeps ONE meaning. PSNR is strongly scale-dependent here (the
  same fine-tune measures +1.15 dB on downscaled content and +0.81 dB on
  native), so a validation set that followed the training policy would put a
  step in the chart that no model change caused.

Representativeness is addressed on the other axis instead — see
[validation resolution](#validation-resolution-is-1024-not-512).

No aspect-ratio bucketing, and that is now a measured decision rather than a
statement about batch collation (the old docstring's reasoning was circular:
bucketing exists precisely to avoid forcing a fixed shape). The decoder's only
non-local terms are **one flattened mid-block self-attention** and **30
GroupNorms**; `AttnProcessor2_0` reshapes `[B,C,H,W]` to `[B,H*W,C]` before
attending and GroupNorm reduces over `(C_group, H, W)`, so both observe latent
**area** and never aspect, and everything else is convolution. Constant-area
bucketing would therefore change nothing this architecture can perceive. The
area gap it would not fix is genuinely large (the median generation gives the
decoder 5.06x the training token count) and was measured **harmless**: from
4,096 to 36,864 latent tokens the fine-tune's PSNR advantage held at +0.93 to
+1.20 dB with no significant trend in any sharpness metric.

### API surface

- `POST /training/runs` with `training_method: "vae_decoder"` and the VAE knobs
  nested in the `vae_config` request field (an object with the
  `VaeTrainingDefaults` key set).
- `GET /schema/vae-training-defaults` returns the 42 defaults (fourth sibling of
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

**Selecting the result for inference.** `GET /api/v1/models/vaes` scans the
training base dir on top of the configured model dirs, so a freshly exported
VAE appears in the VAE-override picker without typing a path. The scan is
bounded to two levels (`<training_base>/<run>/<export_dir>`) and accepts a
directory only when it holds both `config.json` and `sushi_vae_training.json`
— per-run `checkpoints/`, `samples/`, `logs/` and `.dataset_cache/` are never
walked (`api/routes.py::_training_vae_export_dirs`). The sidecar is folded into
the candidate as a `training` object (`run_name`, `step`, `ema_applied`,
`encoder_trained`, `base_vae_path`) and rendered in the picker label, so the
EMA export and its `_noema` sibling are told apart. The label shows the base
VAE's filename rather than the candidate's *inferred* `arch`, because a 4-ch
`AutoencoderKL` reads as `sd15` whichever family it came from, and SD1.5/SDXL
VAEs share `latent_channels` but not `scaling_factor` (0.18215 vs 0.13025).
`encoder_trained` / `ema_applied` are tri-state: a partial sidecar yields
`null`, which is shown as "unknown" rather than assumed benign.

An **encoder-trained** VAE additionally raises a `vae_override_warning` on the
generation response's `warnings[]` channel
(`generation_overrides.py::_warn_vae_training_provenance`), because the
structural gate cannot detect it and a dropdown label can be truncated. That
warning is the same channel every other override warning uses. Decoder-only
exports — the default — stay silent, since they leave the latent contract
intact.

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

#### Validation resolution is 1024, not 512

`validation_resolution` defaults to **1024**. Validation is always a
deterministic centre crop under the `downscale` policy (see
[crop scale policy](#crop-scale-policy-which-pixels-the-decoder-sees)), so this
is the only axis on which the held-out metric can be made representative — and at
512 it was not: that is the most flattering and least representative regime
available, the one where the fine-tune's accuracy gain is largest (+1.15 dB,
against +0.81 dB on native content) and where the content carries ~4x the
near-Nyquist energy of anything generation produces. At 1024 the corpus's median
1131 px source is downscaled only ~1.1x, i.e. nearly native, and it reports the
regime generation actually runs in (median generated short side 960 px, >= 1024
in 49.1% of images). It costs nothing in signal quality: native crops from 512 to
1536 showed the PSNR advantage holding at +0.93..+1.20 dB with no significant
trend (n1536 vs n512: t = −1.75, n.s.). All of the above is
`results_crop_geometry.md` §1.2, §1.4, §3.2, §4.2 and §6.6.

**Changing `validation_resolution` mid-run puts a step in the `vae_val_psnr`
chart** — the metric is computed on different content, so the values before and
after are not comparable, and because `global_step != 0` on a resume no fresh
baseline point is emitted to separate the two regimes. It is the same hazard as
changing `lpips_weight` mid-run (which already cost run 113 −0.17 dB across 140
steps, with the correct sign).

The trainer now says so on resume: `VaeTrainer._warn_measurement_changes`
compares the checkpoint's `train_state.json` config against the resumed run and
prints a **non-fatal warning** when `validation_resolution` or
`crop_scale_policy` differs — a warning and not a refusal, because changing
either is a legitimate deliberate act, and the fatal component-set check
(`train_decoder` / `train_encoder` / `decoder_blocks` / `encoder_blocks`) is
untouched. A key the checkpoint never recorded is treated as *absent*, not as
changed, so pre-policy checkpoints do not warn about `crop_scale_policy`.

That warning matters because of exactly one asymmetry: a run created through the
UI or `generate_vae_config` pins every key in its own YAML and is therefore
unaffected by the default moving, but a **hand-written `process.vae` that omits
`validation_resolution` now resolves to 1024 where it used to resolve to 512**.
Run 113 is in the first category — its stored YAML pins 512 and carries no crop
keys, so resuming it is bit-for-bit unaffected.

### Resume across a changed base VAE

A checkpoint contains **only the tensors that were trainable**. Everything else —
the frozen encoder of a decoder-only run, the decoder blocks outside
`decoder_blocks`, the quant convs — comes from the base VAE the *resuming* run
loads. Editing `vae_path` / `vae_source` / `vae_arch` and then resuming therefore
used to produce a **hybrid model**: checkpoint tensors on one base, the rest on
another, with matching tensor names, no error and no warning.

`VaeTrainer._assert_base_vae_matches` now compares the checkpoint's recorded
`base_vae` block against the one this run resolved, immediately after the
component-set check and before any weight is loaded. Three tiers, by how
conclusive the recorded evidence is:

| Axis | Recorded where | Verdict |
|---|---|---|
| **Frozen-weight fingerprint** — a `blake2b` digest over exactly the tensors a resume does *not* restore (`base_vae.frozen_fingerprint`, computed once in `select_trainable`) | `train_state.json` and the export sidecar | Conclusive both ways. Equal digests prove the untouched half is bit-identical; different digests prove a hybrid → **refusal** |
| **Structure** — `class`, `latent_channels` | recorded by every checkpoint since Phase 1 | Read off the loaded model, not off a user string → **refusal**, *unless the digests are equal*: bit-identical weights cannot be a different model, so a renamed `_class_name` after a diffusers upgrade, or a `latent_channels` that used to be unreported (recorded as `-1`) and now reads 16, is demoted to a **warning** rather than stranding a long run. Structure is the fallback axis for checkpoints written before the fingerprint existed |
| **`scaling_factor` / `shift_factor`** | same | Cannot produce a hybrid (training reads neither), but `save_pretrained` bakes both into the exported `config.json` and the provenance sidecar, and the inference VAE-override path reads them back. A change is therefore **always warned about**, matching digests or not |
| **`path` / `format`** | same | Spelling → **warning**, and only when there is no comparable fingerprint to settle the question |

The fingerprint covers the frozen half rather than the whole model on purpose,
in both directions:

- it makes the check indifferent to *spelling* — the same VAE on a moved drive,
  under a relative path, or loaded as a single file instead of a diffusers
  directory all digest identically, so none of those routine cases can block a
  resume (path differences are compared only after `os.path.abspath`, separator
  and — on Windows — case folding, and only when there is no fingerprint);
- and it keeps the legitimate "point the run at an **export of itself** and
  resume" case working: only the trained half differs there, and the checkpoint
  overwrites that half anyway, so the resulting model is identical either way. A
  whole-model hash would refuse it.

Tensors are cast to fp32 before hashing, so the digest does not depend on the
*container* dtype (the same values held as fp16 and as fp32 hash alike). It is
**not** indifferent to a base file that was actually rounded to fp16: those
values differ, the digest differs, and the resume is refused — correctly, since
the frozen half really would be different weights, but it does mean "the fp16
copy of the same VAE" counts as a different base VAE here.

**Older checkpoints are not broken.** A `train_state.json` with no `base_vae`
block at all resumes with an explicit warning that identity could not be
verified; one with `base_vae` but no fingerprint is still checked on structure
(refusal) and on path/format/factors (warning). The algorithm tag is stored next
to the digest, so a future change to how it is computed makes old and new values
*incomparable* (warning) rather than *different* (refusal).

### Resume from an incomplete checkpoint

A checkpoint directory is six files: `vae_decoder.safetensors`, `optimizer.pt`,
`rng_state.pt`, `train_state.json`, plus `ema.safetensors` and
`lr_scheduler.pt` when the run has an EMA / an LR scheduler. A resume used to
load the optional-looking ones *only if the file happened to exist*, while still
adopting the checkpoint's `global_step` — so a directory that was written or
copied only in part resumed at, say, step 10,000 with **freshly-initialised Adam
moments**, or with the LR schedule sitting at position 0, and logged a normal
resume. Nothing downstream distinguishes that from a healthy continuation.

`VaeTrainer._assert_checkpoint_complete` now runs **before any state is
touched** and grades every artifact by the same rule the other resume guards
use: *state that cannot be reconstructed and whose loss is invisible afterwards
is a refusal; state that is re-derivable, and whose loss is announced and
repaired, is a warning.*

| Artifact | Absent, empty or the wrong size | Why |
|---|---|---|
| `train_state.json` | **refusal** | carries the step, the data pass, the component set and the base-VAE identity; without it a resume can neither position itself nor run any other guard |
| `vae_decoder.safetensors` | **refusal** | the trained weights themselves |
| `optimizer.pt` | **refusal** | Adam's moment estimates *are* the run's accumulated history. A fresh optimizer changes the effective step size for thousands of steps, and no log, metric or chart reports it |
| `lr_scheduler.pt` | **refusal** when this run has a scheduler and the file is *established* to have been written (see the conditional-artifact rule below) | the schedule would restart at position 0 while the step counter jumps to the checkpoint's, replaying warmup/decay silently |
| `ema.safetensors` | warning + re-seed | the EMA is a derived average; the re-seed is announced, and it also **resets the EMA warmup counters** (`ema_updates`, `ema_retained_init_fraction`) so the re-seeded average is not immediately damped at full decay. This is the pre-existing partial-EMA behaviour, now applied to the absent and unreadable cases too |
| `rng_state.pt` | warning | only the noise/augmentation draw stream. The data order and per-visit crops are restored separately and exactly by `data_epoch`, so what changes is which random draws are made, not which images are trained on |

Two *config* differences look like missing files and are warnings, not
refusals, because the manifest (below) proves the file was never written and the
run's own config explains why: adding an LR scheduler to a run whose checkpoint
had none (the schedule then starts at position 0 — stated in the warning), and
enabling/disabling EMA across a resume.

**How "complete" is decided.** `save_checkpoint` writes `train_state.json`
*last*, and records in it an `artifacts` manifest of `{filename: byte size}` for
everything it wrote. On resume each artifact is classified as `ok`, `missing`,
`size_mismatch` (present, but empty or not the size it was saved at),
`not_written` (absent from the manifest, i.e. the writing run never produced
it), `absent_unverifiable` (absent, conditionally written, and no manifest to
say which) or `unverified` (present, no manifest at all). Sizes rather than
hashes: the failure being caught is a file cut short — an interrupted save, a
copy still in progress, a full disk — and hashing ~400 MB of optimizer state on
every save and every resume would cost far more than that. A zero-byte artifact
is rejected on both paths, manifest or not. Same-length corruption is not caught
by sizes, so `torch.load` failures on `optimizer.pt` / `lr_scheduler.pt` are
converted into the same refusal rather than being skipped — and those messages
name the *other* cause that lands there, a checkpoint written under a different
`optimizer` / `lr_scheduler` than this run's. Optimizer state normally only
loads back into the same implementation and parameter-group layout; the one
explicit exception is the AdamW -> AdamW8bit migration described below.

### Resume while changing AdamW to AdamW8bit

VAE resume permits exactly one optimizer change: a checkpoint written by torch
`AdamW` may resume with bitsandbytes `AdamW8bit`. `optimizer.pt` records
`_sushi_opt_class` on new checkpoints; older VAE checkpoints can use the
`train_state.json` config's `optimizer: adamw` as the source identity. The state
is loaded on CPU and converted *before* `AdamW8bit.load_state_dict` runs:

- `exp_avg` and `exp_avg_sq` are validated as fp32 tensors with the same shape
  and group/parameter position as the live optimizer;
- parameters at or above the target AdamW8bit `min_8bit_size` are quantized with
  bitsandbytes blockwise quantization (block size 256, signed first moment,
  unsigned second moment), while smaller parameters retain fp32 `state1` and
  `state2`, matching AdamW8bit's own lazy state initialization;
- the target optimizer's parameter groups are retained, and the normal resume
  LR / weight-decay reassertion still applies.

Conversion is mandatory for this pair. A group/shape mismatch, missing moments,
unsupported target configuration, or quantization failure refuses the resume;
the raw foreign dict is never tried. This matters because a foreign state dict
can pass `load_state_dict` and only fail on the first optimizer step. All other
changes (including the reverse AdamW8bit -> AdamW direction, Adafactor, Lion,
paged and ring-buffer variants) remain refused before weights are loaded.

**Conditionally-written artifacts.** `ema.safetensors` and `lr_scheduler.pt`
(`_CKPT_CONDITIONAL`) are written only when the run *has* an EMA / a scheduler.
With a manifest, their absence is proven to be "never written". Without one it
is genuinely ambiguous, and that ambiguity is **not** resolved as damage: the
absence warns (stating that the schedule starts at position 0, or that the EMA
is re-seeded) instead of refusing. Otherwise every checkpoint produced by
`build_optimizer`'s `LR scheduler … unavailable; using constant LR` fallback
would become unresumable once the cause was fixed, with no escape hatch. The
unconditionally-written artifacts have no such ambiguity — the writer always
produces them, so absent means lost, manifest or not.

**A refusal names the way out.** `resume_from_checkpoint` defaults to `latest`,
which picks the highest step *without* inspecting it — and the directory this
guard catches is typically the newest one, so a refusal would otherwise turn a
silent degradation into "the run will not start". The message therefore lists
the intact `step_*` directories in the same `checkpoints/` folder (and says so
explicitly when there are none), the way `resolve_resume_target` already lists
what it found for an unresolvable explicit name. It is a list, not an automatic
fallback: silently resuming an older checkpoint would roll the step counter back
by up to `save_every` steps and re-train that span — the same class of
unannounced surprise this guard exists to remove.

**Existing checkpoints keep resuming.** A checkpoint written before the manifest
existed is checked for presence only, and says so once in the log
(`… predates the artifact manifest …`); an absent conditional artifact warns
there rather than refusing (above). All 11 checkpoint directories on disk from
runs 113–117 verify clean under the new guard, with no warnings.

**There is deliberately no "resume anyway" key** — the parameter surface is
unchanged. A resume means *continue this run*; deliberately restarting
optimisation is a different intent that is already expressible without a new
key: let the run export (or stop it and export), then start a **new** run with
that export as its base VAE. That gets a clean step counter, a clean LR
schedule and a metric series that is not two optimisation regimes spliced into
one line — which is exactly what a "partial resume" would silently produce. The
refusal message says this.

Independently of the resolution: **`vae_val_psnr` is anti-correlated with
sharpness here**, so it must not be used on its own to decide whether an
LPIPS-weight change worked.

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
| `l_invented_weight` | **0.0** (available) | Flat-region invented-HF penalty — the only term in the bank that is **not** an agreement-with-source objective. See below. |
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

### What `l_invented_weight` is for

Every other term in the bank compares the decode with the source, which is the
family the SDXL VAE was already trained on. Measured outcome of a full fine-tune
under that bank (`scratchpad/vae_training/results_flat_region_noise.md`): in
flat regions the *error* fell 21%, while the total high-frequency energy the
decoder emits there moved **+0.4% (not significant)**. The decoder kept
fabricating the same amount of detail and aimed it better. In dark flat windows
**66% of the fine high frequency is fabricated**, and at the exposure gain a
user actually inspects such regions at, the invented luma is **3.3/255** against
a visibility bar of 1/255.

`l_invented_weight` turns on a *conditional non-generation* term instead. Inside
windows where a least-squares **plane fit** on the source says the region is
flat or a smooth gradient (so ramps count, unlike a variance test), it charges
the part of the decode's high frequency that a least-squares projection onto the
source's own high frequency cannot explain:

```
alpha = clamp( <h(recon), h(target)> / (<h(target),h(target)> + eps), 0, 2 )   # DETACHED
L     = mean( (h(recon) - alpha * h(target))^2 )
```

- **alpha carries no gradient.** With it attached, the decoder could reduce the
  loss by raising its correlation with the source instead of emitting less —
  the behaviour the existing bank was measured to reward. Detached, each step is
  a plain MSE toward a fixed target, so the only way down is to emit less
  unexplained energy.
- **eps sits at the measured 8-bit quantisation-floor energy** (`0.2797²` per
  pixel), so in a genuinely empty window alpha goes to 0 smoothly and the term
  becomes "where the source has nothing, emit nothing".
- **Nothing is exempt, and blur is charged less than exact reproduction.** With
  `eps > 0` the fixed point inside a flat window is `d = alpha·s` with
  `alpha < 1` — a shrink, not the identity. For a decode `d = g·s` the charge is
  `sigma²·(g − alpha(g))²`, which rises smoothly from `g = 0` upward — exactly
  as `g²` while alpha tracks. Measured at sigma ≈ 1.5: g=0 (emit nothing) → 0
  exactly, g=0.5 → 0.00223, g=1 (exact reproduction) → 0.00892, g=2 → 0.03567,
  g=3 → 0.62209. The `alpha ≤ 2` clamp caps how much exemption a
  strongly-correlated emission can buy; it does **not** make gains below 2x
  free. Exact reproduction costs `sigma²·eps²/(sigma²+eps)²` per scale, at most
  `eps/4` = 0.0196 levels² (0.140 levels rms per scale, ~0.198 combined), and a
  blur costs less than that at every sigma.
- **Consequence: it is not a standalone objective.** Its own global optimum
  inside the mask is "emit nothing". Measured on 200 real 512² crops at this
  geometry, the systematic under-emission of *transmitted* HF at the fixed point
  is 5.1% of the in-mask transmitted-HF energy amplitude. It is meant to run
  with `mse`/`lpips`, which supply the opposing pull toward `g = 1`; the
  in-mask transmitted-HF gate below (R7) is what keeps that trade honest.
  `l_invented_weight` alone satisfies the config's "at least one training
  signal" check (as `pattern_weight` alone does) — but a run configured that way
  minimises its loss by emitting no high frequency in flat regions at all. Do
  not configure it that way.
- Windows, thresholds, the highpass basis and the photometric weight are
  deliberately different from the frozen evaluation harness's, so that a fall in
  the harness's number is evidence rather than a tautology. Only the five
  weights/thresholds are configurable; the geometry is internal.

Logged per step as `vae_invented_loss` (unweighted, in (8-bit levels)²) and
`vae_invented_cov` (the fraction of candidate windows that passed the flat test
— a value that falls because the term stopped firing looks identical to one that
falls because the decoder stopped inventing, unless both are charted).

`sqrt(vae_invented_loss)` is a **relative trend indicator, not an absolute
level**, and must not be read against the 1/255 visibility bar: the logged value
carries the term's Weber photometric weight (0.16 bright … 0.98 black) and its
channel weights. Injecting exactly 1.0 level of pure uncorrelated invention
gives `sqrt(logged)` = 0.94 (dark window) / 0.52 (mid) / 0.40 (bright), i.e. it
under-reads true invented luma by 1.1–2.5×. Absolute levels come from the frozen
g1flat harness only.

**Gate when running this term (R7).** The design's regression gates R1 (edges)
and R2 (the flat mask's *complement*) cannot see attenuation of transmitted HF
*inside* the mask, and under a blur both the primary success metric (invented
luma) and the secondary (total emitted flat HF) fall — so blur reads as success
on every other instrument. Therefore: in-flat-mask **transmitted** luma HF rms
(`alpha·s`, already computed in the frozen harness,
`results_flat_region_noise.md` §1.3) must be **≥ −5%** against the start arm. A
candidate that reaches the invented-luma target while failing R7 has bought it
with blur and is a failure, not a success.

Behaviour is pinned by `backend/tests/test_l_invented_loss.py`; measured cost of
the term at 512², batch 2, forward+backward: ~18 ms and ~124 MB of transient
VRAM on this install (contended GPU, so the time is an upper bound).

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
| Resume from a checkpoint trained against a **different base VAE** | `vae_trainer.py::_assert_base_vae_matches`, before any weight is loaded. See [Resume across a changed base VAE](#resume-across-a-changed-base-vae). |
| Resume from a checkpoint that trained a different component set | `vae_trainer.py::_assert_component_set_matches`, before any weight is loaded. A checkpoint holds exactly the parameters that were trainable when it was written, plus optimizer and EMA state indexed by that same set. Both directions were previously silent-ish: a decoder-only checkpoint resumed with the encoder on failed with a message blaming `decoder_blocks`, and the reverse loaded happily (the checkpoint is a superset) and then failed opaquely inside the optimizer state load — or not at all, if `optimizer.pt` was absent. |
| Resume from a checkpoint whose `train_state.json`, weights, `optimizer.pt` or (when this run has a scheduler, and the file is established to have been written) `lr_scheduler.pt` is absent, empty, the wrong size or unreadable | `vae_trainer.py::_assert_checkpoint_complete`, before any state is touched. The step counter would be adopted while the missing state was silently re-initialised. `ema.safetensors` and `rng_state.pt` are the warn-and-repair tier instead, as is an absent conditional artifact on a checkpoint with no manifest. The refusal lists the intact sibling checkpoints. See [Resume from an incomplete checkpoint](#resume-from-an-incomplete-checkpoint). |
| Resume while changing optimizer, except AdamW -> AdamW8bit | Optimizer state layouts and moment semantics are implementation-specific. AdamW -> AdamW8bit has an explicit validated conversion; the reverse direction and Adafactor/Lion/paged/ring-buffer changes are refused before weights are loaded. See [Resume while changing AdamW to AdamW8bit](#resume-while-changing-adamw-to-adamw8bit). |
| `dtype: fp16` | SD1.5/SDXL-family VAEs overflow fp16 in decoder activations (the documented reason `sdxl-vae-fp16-fix` exists), and a training forward hits it sooner than inference. For every other family there is no `GradScaler` in this trainer, so fp16 gradients would silently underflow instead. `bf16` (default) and `fp32` are allowed. |
| `latent_encoding_mode: pre_encoded_cache` | VAE training is *defined* by a live encode→decode forward on raw pixels; there is no cached latent to consume. Mirrors the existing outpaint-ControlNet refusal. |
| A single-file base VAE whose `vae_arch` is empty, unknown, contradicts the file's latent-channel count, or contradicts a value the file itself provided | `vae_trainer.repair_single_file_scaling_factor`, at load time. Such a file has no `config.json`, so `vae_arch` is the only statement of which family it is, and that statement is what `save_pretrained` bakes into every export. See [the `vae_arch` matrix](#known-limits). |
| All loss weights 0 | No training signal. |
| `lpips_weight > 0` with `lpips` not importable | Fails before the run starts, never mid-run. |
| Unknown key in `vae_config` / `process.vae` | Caught in `generate_vae_config` and in `resolve_vae_training_config`; a typo must not silently resolve to the default. |
| `crop_scale_max_downscale > 0` with `crop_scale_policy` not `mixed` | The bound is consulted only by the per-sample draw, so under `downscale` / `native` it would be a knob the caller set, the YAML recorded, and nothing read. Refused rather than ignored. |
| `crop_scale_max_downscale` between 0 and 1 | It names a *downscale* factor, so a sub-1 value would mean an upscale bound. Clamping it to 1 silently would train on a distribution nobody asked for; `crop_scale_policy: native` is how "never downscale" is spelled. |
| Out-of-enum `crop_scale_policy` / `vae_source` / `decoder_blocks` / `encoder_blocks` / `dtype` / `lpips_net`; empty `vae_path`; `vae_source: store` with no `vae_arch`; `resolution` not a multiple of 8 or < 64; `batch_size`/`total_steps`/`gradient_accumulation_steps` < 1; `ema_decay` outside (0,1); negative or non-numeric loss weight or `kl_weight` | Ordinary validation, same fail-early principle. |
| `learning_rate <= 0` | At 0 every optimizer step is a no-op (AdamW's decoupled decay is also scaled by the LR), so the run finishes, reports success and exports a copy of the base VAE. Negative ascends the loss. |
| `max_grad_norm < 0` | `clip_grad_norm_` scales by `max_norm / total_norm` and clamps that factor only from *above*, so a negative bound negates every gradient. **0 is accepted and means "no clipping"** — see [Gradient clipping](#gradient-clipping-0-means-off). |
| `optimizer_weight_decay < 0` | Multiplies every weight by more than 1 per step; unbounded growth, unreported until the loss stops being finite. |
| `optimizer` outside `VALID_OPTIMIZERS` | `OptimizerFactory` would raise only after the base VAE is loaded. The enum is exactly what that factory resolves — **including** `adamw8bit_ringbuffer` / `lion8bit_ringbuffer`, which do run here: with no allocator passed they fall back to allocating their 8-bit state on the GPU (verified by a live `step()`), i.e. the same placement as plain `adamw8bit` / `lion8bit`. `build_optimizer` logs that, since the name promises otherwise. |
| `lr_scheduler` outside `VALID_LR_SCHEDULERS` | `build_optimizer` *catches* a `get_scheduler` failure and continues at a constant LR, so an unrunnable name is not an error at run time — it is a silently ignored schedule. `piecewise_constant` is excluded for the same reason (no `step_rules` is ever passed). |
| `lr_scheduler: constant` with `lr_warmup_steps > 0` | `get_scheduler`'s `CONSTANT` branch returns before `num_warmup_steps` is passed to anything, so the run trains at the full LR from step 0 while the YAML, the sidecar and the LR chart all record a warmup. `constant` is the default and both keys are UI-reachable, so this is the likeliest spelling of the mistake. Use `constant_with_warmup`. |
| `lr_warmup_steps >= total_steps` | The whole run would be warmup, so the configured LR is never reached while the YAML, the sidecar and the LR chart all report it. |
| `validation_num_images < 1` | The split is `items[-validation_num_images:]`: 0 leaves the **training** split empty (`items[:-0]` is `items[:0]`) while validating on everything, and -1 trains on `items[:1]`. |
| Negative `validation_every` / `save_every` / `num_workers` / `max_step_saves_to_keep` / `lr_warmup_steps` / `pattern_size` | Each one is guarded downstream by `> 0`, so a negative value *silently* disables validation, disables checkpointing, or keeps every checkpoint instead of pruning. |
| `seed` outside `0 .. 2**32-1` | Not because the value fails — `random.seed(-1)` and `torch.manual_seed(-1)` are legal and the trainer already takes a modulus for numpy. Because of that modulus: python and torch get the literal value while numpy gets `seed % 2**32`, so `-1` reaches numpy as 4294967295 and `2**32+7` as 7, while `train_state.json` and the sidecar record the original — the generators disagree and the recorded seed does not reproduce the run. There is also no `-1 = random` convention here, unlike the generation seeds in `api/param_defaults.py`. |
| `ycbcr_dc_y_weight < 0` or `ycbcr_dc_chroma_weight < 0` | The term is summed over channels, so a negative channel weight *pays* the run for increasing that channel's colour error while the total loss still falls. |
| `ycbcr_dc_y_weight` and `ycbcr_dc_chroma_weight` both 0 while `ycbcr_dc_weight > 0` | Identically zero objective, computed every step — and it passes the "all loss weights 0" check, which only sees the top-level weight. (The `l_invented_*` pair has the same rule.) |
| `ycbcr_dc_eps <= 0` | Charbonnier is `sqrt(d² + eps²) - eps`: at 0 it degenerates to `\|d\|`, whose gradient at an exactly-zero residual is NaN, and a negative value offsets the reported loss instead of subtracting. |
| `pattern_size > resolution` while `pattern_weight > 0` | The term crops to whole cells, so it has none and returns exactly 0 every step while the config says it is active. |
| A non-finite (`NaN` / `inf`) number, a fractional integer count, a boolean where a number is expected, or a non-string `vae_path` / `vae_arch` | `_as_number` / `_as_int` / `_as_text`. `int(2.7)` truncates, `float(True)` is 1.0, and a `NaN` weight is only noticed by the trainer's non-finite-loss abort, one model load later. |

### Gradient clipping: 0 means off

`max_grad_norm: 0` is the way to turn clipping off, which is what the same key
means in the diffusion trainers (`base_trainer`, `optimizers/fused_optimizer_groups`,
the latter documenting it as "0 to disable") and what the UI's `min=0` input has
always implied. Passing 0 straight to `torch.nn.utils.clip_grad_norm_` does
**not** do that: the scale factor is `0 / total_norm`, so every gradient becomes
exactly 0, the optimizer step is a no-op except for AdamW's decoupled weight
decay, and the run reports success while *shrinking* the weights it was asked to
train. `VaeTrainer._clip_gradients` therefore skips the clip at 0 and still
returns the unclipped total norm, which is what the `grad_norm` chart shows.

`max_grad_norm: inf` was another working spelling of "off" (`clip_grad_norm_`
clamps its scale factor at 1.0). It is refused, with a message that names `0`,
so that "no clipping" has one spelling in a config, a chart legend and a
sidecar rather than two.

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

Every row above is asserted in **`backend/tests/test_vae_refusal_matrix.py`**,
which is the executable form of design.md §4 plus, in
`VaeResumeBaseVaeIdentityTest`, the base-VAE identity guard (a changed base
refuses; the same base under a different path/format resumes silently; a
pre-fingerprint and a pre-`base_vae` checkpoint both still resume; a renamed
class or newly-reported `latent_channels` warns instead of refusing when the
digests match, while a changed `scaling_factor` / `shift_factor` warns even then;
`select_trainable` is driven for real so the *recording* wiring is covered, not
just its two ends; the digest ignores the tensors a resume overwrites and reacts
to the ones it does not), in `VaeResumeCompletenessTest`, the checkpoint
completeness guard (a complete checkpoint restores weights, optimizer state and
the EMA counters; missing/truncated/corrupt `optimizer.pt`, `lr_scheduler.pt`,
weights or `train_state.json` each refuse and name the file; missing EMA and RNG
warn, resume, and re-seed the EMA with its ramp reset; adding or removing a
scheduler or the EMA across a resume warns rather than refuses; an absent
*conditional* artifact on a pre-manifest checkpoint warns while an absent
unconditional one still refuses; a zero-byte artifact refuses with and without a
manifest; a refusal lists the intact sibling checkpoints and omits the damaged
ones; a checkpoint written by the real `save_checkpoint` round-trips, carries a
size manifest for every file it wrote, and detects a byte-level truncation of
any of them; and a real-save assertion that every file left in the directory is
one the resume guard verifies) and,
in `VaeCropScalePolicyTest`, the loader side of the crop scale policy — including a
verbatim copy of the pre-policy loader that `downscale` is pixel-compared
against, and an `inspect.signature` assertion that `make_validation_batch` takes
no policy argument:

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
  turn is caught 11/11 — and the 5 crop-scale guards added later were
  mutation-swept the same way, caught 5/5 (the enum row is the important one:
  without its guard an unknown policy resolves cleanly and the *loader* becomes
  the first thing to notice, after the model load).

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
- **`vae_arch` also decides the exported `scaling_factor` for a single-file
  base.** A VAE-only single file carries no `config.json`, and
  `AutoencoderKL.from_single_file` cannot tell an SDXL VAE from an SD1.5 one
  (identical architecture), so diffusers falls back to LDM's `0.18215`. Since
  `save_pretrained` bakes `vae.config` verbatim into the exported
  `config.json` — and the inference-side VAE override trusts a directory's
  `config.json` — an SDXL export carrying `0.18215` would be a silent 1.40×
  latent-scale error, and an SD1.5 export carrying `0.13025` is the same error
  in the other direction. The substituted value is therefore **stated, never
  assumed**: `vae_arch` defaults to the empty string ("not stated"), and the web
  UI exposes the field for every `vae_source`, not only `"store"`. The decision
  matrix in `vae_trainer.repair_single_file_scaling_factor` (canonical numbers
  in `core/models/common/vae_store.VAE_REGISTRY`):

  | Single-file base VAE | Outcome |
  |---|---|
  | Full checkpoint (a backbone key is present) | Left as loaded — `from_single_file` identified the family, which is evidence rather than a guess. A stated scalar `vae_arch` that **contradicts** that value is **refused**, though: one of the two is wrong, and this cross-check is also the backstop for a misclassified file. `vae_arch` left unstated is fine here. |
  | VAE-only `.safetensors`, `vae_arch` stated and scalar (`sdxl` → `0.13025`, `sd15` → `0.18215`, `flux1` → `0.3611`/`0.1159`) | Corrected, with the before/after printed and `base_vae.scaling_factor_source` recorded in the sidecar. |
  | VAE-only `.safetensors`, `vae_arch` empty | **Refused.** Nothing can tell the families apart, and both possible guesses write a wrong number into every export. |
  | `vae_arch` set to something that is not a registry key | **Refused** — the run asked for a correction that cannot be made. |
  | `vae_arch` names a family with no scalar factor (`flux2`, `qwen_image`) | Left as loaded, loudly, and recorded as `UNVERIFIED`: those families normalise with `latents_mean`/`latents_std`, so there is no number to substitute. |
  | `vae_arch`'s latent-channel count contradicts the loaded VAE | **Refused** — that config does not describe this file. |
  | Not classifiable — `.ckpt` / `.pt` / `.bin` (would need unpickling), an unreadable safetensors header, or a key layout that is neither a VAE dump nor a known backbone | Never overwritten, only checked: matching `vae_arch` passes, an empty or contradicting one is **refused**, because whether the loaded value was read from a backbone or is the fallback cannot be determined. |

  All of these fire in `load_base_vae`, before the first training step and
  before anything is written.

  **How "is this a full checkpoint?" is decided.** By looking for a BACKBONE
  (`model.diffusion_model.`, `conditioner.`, `cond_stage_model.`,
  `double_blocks.`, `transformer.`, … — `_BACKBONE_KEY_MARKERS`), not by
  checking that every key looks like a VAE key. The earlier allow-list form of
  that check was an allow-list over a space nobody controls: a stock
  `sdxl_vae.safetensors` also ships a `model_ema.*` block (measured: 250 keys,
  top-level `decoder` / `encoder` / `model_ema` / `post_quant_conv` /
  `quant_conv`), that one unlisted prefix made it classify as a full checkpoint,
  and the repair was skipped for the single most common file this feature
  exists for — exporting 0.18215 for an SDXL VAE while recording "family
  identified" in the sidecar. Backbone names are few and slow-moving, so new VAE
  side-car tensors no longer break the classifier; a file with no backbone
  marker that is *also* not recognisable as a pure VAE dump is reported as
  "unknown" (the check-only row above) rather than being guessed either way.
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
