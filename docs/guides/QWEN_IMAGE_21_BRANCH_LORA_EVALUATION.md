# Qwen-Image 2.1 CFG branch evaluation plan

Status: planned verification; no branch-separated adapter has shipped.
See [QWEN_IMAGE_21_BRANCH_LORA_DESIGN.md](QWEN_IMAGE_21_BRANCH_LORA_DESIGN.md)
for the proposed training/inference contract and
[QWEN_IMAGE_21_GUIDANCE_LOSS.md](QWEN_IMAGE_21_GUIDANCE_LOSS.md) for the
existing loss. Store raw images, latents, logs, prompts tied to a private
dataset, and machine-specific benchmark output under `local/measurements/`,
not in this tracked plan.

## Questions and controls

1. Does the shared adapter trained with no explicit null-drop remain stable
   when its LoRA delta is included on the CFG negative forward? Does using
   the unadapted base instead improve or worsen the same rollout?
2. Does a split adapter trained with zero null-drop and a fixed base-empty
   teacher improve CFG 1 and higher-CFG results relative to the shared mode?
3. How does an actual nonempty negative prompt change either comparison?
   Empty-prompt conclusions must not be generalized to negative tags without
   measurement.

Run 156 is the existing shared-LoRA, `cfg_uncond_drop_rate=0` arm. It starts
from the base model with a fresh optimizer, retains the mixed ordinary/
guidance-target loss (`weight=0.25`, scale 3, sigma schedule), rank 128,
two-region partition and partition-global adapter. This is **not** a
fixed-base-null run: the shared LoRA can change its null response even though
no null-caption item is trained explicitly. In the current partitioned
training path, the global adapter also runs in the detached null reference;
its conditional updates are another possible source of teacher drift. Full-
frame generation ignores that training-only adapter. Do not interrupt or
rewrite Run 156 just to run this probe. Use a safely saved checkpoint and a separate
inference process when GPU availability permits. Run IDs are local examples,
not part of the implementation contract.

For every arm, fix the same base artifact, LoRA checkpoint, positive prompt,
negative prompt, dimensions, scheduler/steps, seed, guide/reference settings,
and KV-cache setting. Keep LoRA strength equal across arms. Record the exact
checkpoint step and a hash of its bytes. Within one checkpoint these paired
rollouts isolate inference branch choice. Cross-run comparisons are less
controlled: Run 156's training seed is `-1`, and a different run may not have
an exactly matched checkpoint step. Report these limitations instead of
attributing every image difference to the branch design.

## Phase A: shared-LoRA Run 156

At each available checkpoint, evaluate CFG 1, 3, and 7 with two negative
prompt classes: empty string and a fixed, nonempty low-quality-tag prompt.
For a given positive condition `p` and negative condition `n`, compare:

```
c       = B(p) + Delta_shared(p)
n_shared= B(n) + Delta_shared(n)   # current generation behavior
n_base  = B(n)                     # diagnostic negative-only LoRA bypass
v_s     = n_shared + s*(c - n_shared)
v_b     = n_base   + s*(c - n_base)
```

The two arms must use the **same conditional prediction**. Implement a
diagnostic request-scoped negative-only bypass if needed; unloading/reloading
the LoRA for the whole request would also change `c` and invalidate the
comparison. Check that bypass works with the pipeline's distinct cond/uncond
KV caches and with cache disabled. CFG 1 has no negative forward, so `v_s`
and `v_b` must be identical. Treat this as an invariant/control, not as a
quality comparison. For CFG 3/7, the same-latent instantaneous difference is
`(1-s)*(n_shared-n_base)`; then compare full free-running rollouts, where
latents diverge and the instantaneous formula no longer predicts endpoints.

Collect at fixed sigma landmarks near 0.95, 0.8, 0.5, and 0.2: conditional
prediction, both negative predictions, guided velocity, predicted clean
latent, delta RMS/norm ratios, and finite-value checks on the **same**
noisy latent. For the rollout, log the latent/clean-prediction trajectory,
first step where arms materially diverge, generated images, runtime, and
peak VRAM. Use more than one seed and prompt; include the character dataset's
intended positive prompt, a different composition, and a neutral prompt.
Keep the exact prompt set in the measurement manifest. Evaluate character
fidelity, composition, anatomical/artifact failures, and overshoot separately;
one attractive endpoint is not evidence of a stable trajectory.

Before relying on the diagnostic, unit-test that its positive prediction is
unchanged, its base-negative prediction equals a no-LoRA base call at the
same latent and text encoding, and CFG 1 does not execute the negative
forward. Failure of any invariant blocks the image comparison.

## Phase B: proposed split `C`/`U` mode, drop 0

Train from the same base model with the same data, ordinary/guidance loss,
optimizer policy, conditional rank, partition policy, and sampling settings
as Run 156, except for the branch-separated adapter. Do not initialize `C`
from Run 156: that would carry the shared adapter's null response into the
experiment. Use a fixed training seed for this and any repeat; because Run
156 used `-1`, comparisons with it remain observational rather than a
perfectly paired training ablation.

At zero null-drop, `U` is absent or frozen at an exact zero delta. The
detached guidance reference is `B("")` evaluated through the same tiled
geometry but without the conditional global adapter. The ordinary and
guidance-target losses update only `C`. Verify at initialization, after an
optimizer step, after save/load, and after resume that the `U` prediction is
equal to the same-geometry unadapted base within the model's numerical
tolerance and that no `U` optimizer state exists.

Repeat Phase A's CFG 1/3/7 and empty/nonempty-negative grid. In the split
mode, comparing `base+U` with base-only for an empty negative is an
**identity test**, not an efficacy arm: both should coincide at drop 0.
The meaningful efficacy comparison is split-mode conditional prediction
against Run 156's conditional prediction and their complete rollouts under
the same negative policy. A nonempty negative defaults to `B(n)` in the
first split prototype. Evaluate `B(n)+C(n)` only as an explicit diagnostic
arm, never silently as the negative default.

Record the training costs as well as image results: step time, positive and
reference forward time, backward time, allocated/reserved peak VRAM,
checkpoint/optimizer sizes, and ordinary/guided losses. No speed or memory
benefit is assumed. If an optional trained `U` is later introduced with
nonzero null-drop, that is a separate experiment and must re-test teacher
drift, null quality, and the checkpoint-size tradeoff.

## Later negative-example experiment

Only after the empty/nonempty-negative results are known, consider either
(a) text-only negative templates as detached **base** reference conditions
for a new guidance-target mixture, or (b) image-backed, explicitly captioned
low-quality examples to train a conditional response to negative tags.
These are different objectives. Text-only prompts provide no ordinary image
MSE target. Image-backed examples must be balanced, labeled by the defects
they actually contain, and checked for leakage of those defects into normal
positive generation. Record negative-template frequencies and dataset
composition; compare against the split drop-0 baseline before adopting.

## Decision gate

- Structural gates: branch-isolation gradients, zero-drop base equality,
  positive-forward invariance during diagnostic bypass, checkpoint roundtrip,
  KV-cache on/off correctness, finite rollout, and unchanged behavior for
  ordinary single-branch LoRA files all pass.
- Empirical gates: report both wins and regressions across seeds/prompts at
  CFG 1/3/7, including nonempty negative prompts and trajectory measures.
  Do not accept a lower training MSE or one improved image as sufficient.
- If base-negative bypass improves high-CFG composition but harms CFG 1,
  recheck the experiment: CFG 1 cannot depend on the negative branch.
  If split mode merely makes `U=base` but does not improve conditional
  results, preserve it as an experiment rather than enabling it by default.
