# Qwen-Image 2.1 branch-separated LoRA design

Status: proposed; not implemented. The existing shared-LoRA guidance-target
objective is documented in [QWEN_IMAGE_21_GUIDANCE_LOSS.md](QWEN_IMAGE_21_GUIDANCE_LOSS.md).
The measurement sequence and acceptance gates are in
[QWEN_IMAGE_21_BRANCH_LORA_EVALUATION.md](QWEN_IMAGE_21_BRANCH_LORA_EVALUATION.md).

## Problem and scope

Today the same attention-projection LoRA weights affect both CFG forwards.
The guidance-target teacher is detached within a step, but a conditional
optimizer update still changes the next step's empty-prompt prediction. A
zero `cfg_uncond_drop_rate` removes explicit null-caption training; it does
**not** freeze the shared LoRA's null response. The proposed mode separates
the adapter deltas by CFG role while retaining one frozen Qwen base. It is an
opt-in Qwen experiment, not a reinterpretation of existing LoRA files.

Let `B(x, sigma, p)` be the frozen base prediction for prompt `p` at a fixed
noisy latent, `C` the conditional adapter delta, and `U` the empty-prompt
adapter delta. The positive prediction is `c = B(p_pos) + C(p_pos)`. The
empty-prompt reference is `u = B("") + U("")`. There is no shared trainable
adapter term. The proposed mode routes the existing partition-global adapter
only through the conditional forward; the reference path must omit it to make
`U=0` mean the unadapted base evaluated
with the same partition geometry. Full-frame inference already ignores that
training-only global adapter. Equality with a full-frame base prediction is
not implied by equality within a tiled training forward.

For positive items, with ordinary flow target `v = eps - x0`, guidance
weight `w`, and the current effective guidance scale `g`, use

```
u_teacher = stop_gradient(B(x_sigma, sigma, "") + U(x_sigma, sigma, ""))
v_guided  = u_teacher + g * (v - u_teacher)
L_cond    = (1-w) * MSE(c, v) + w * MSE(c, v_guided)
```

Only `C` and its conditional global adapter receive gradients from this
loss. A null-drop item, if enabled later, uses only
`L_uncond = MSE(B("") + U(""), v)` and updates `U`, not `C`. An item drawn for
null-drop does not also receive the positive or guidance-target loss. For the
first experiment set drop rate to zero: `U` must be absent or exactly zero,
excluded from the optimizer, weight decay, EMA, and any sidecar update. Thus
the empty-prompt reference is fixed across optimizer steps. Keep the
conditional rank at the comparable run's value; if null-drop is studied
later, a separately configurable smaller `U` rank is preferable to doubling
the full-rank adapter without evidence.

Stopping the teacher's gradient alone is **not** the new mechanism: the
current implementation already evaluates it under `torch.no_grad()`. The
mechanism is that updates to `C` cannot change `U` on later steps. With a
nonzero null-drop rate, `U` can still move through its own supervised loss;
an EMA teacher or sigma-dependent null update would be a later, separately
measured option rather than part of this first prototype.

## Negative prompts are not the empty prompt

Qwen's CFG path encodes a nonempty negative prompt as a real alternate
condition and computes `v_cfg = n + s * (c - n)` for scale `s > 1`.
At `s=1` the negative branch is not evaluated. The first branch-separated
inference policy is role-based, not based on whether words look positive:

| CFG role | First-prototype prediction |
|---|---|
| Positive prompt | `B(p_pos) + C(p_pos)` |
| Empty negative prompt | `B("") + U("")`; with drop 0, exactly base |
| Nonempty negative prompt | `B(p_neg)` by default |

Applying an adapter trained on an empty prompt to arbitrary low-quality tags
would be an out-of-distribution assumption. An optional diagnostic arm may
evaluate `B(p_neg) + C(p_neg)`, but it must not silently become the default.
Training `C` on captioned low-quality images might make that arm meaningful;
text-only negative prompts have no ordinary image-MSE target. Image-backed
negative examples also risk teaching low quality to the shared conditional
adapter unless their labels, sampling ratio, and positive-image regression
are controlled. That is a later experiment, not a dependency of branch
separation.

The initial guidance target still uses the empty prompt so it is comparable
to the existing objective. A follow-on experiment may sample a fixed,
versioned distribution of negative prompts and use detached `B(p_neg)` as
the guidance reference. This changes the optimization target and must be
evaluated separately: training against base-empty alone does not guarantee
stability against every base-negative condition. At CFG `s`, substituting a
different negative prediction changes the guided field by
`(1-s) * (n_new - n_old)` at the same latent, which can be large at CFG 7.

## Integration contract

1. Add a versioned, Qwen-only adapter mode and separate `C`/`U` namespaces
   to the LoRA checkpoint. Preserve ordinary shared-LoRA loading and saving
   unchanged. Metadata must state the branch policy and whether `U` exists;
   an old file must never be guessed to be branch-separated.
2. Route each forward explicitly by CFG role in the training op and Qwen
   pipeline. Do not mutate a global LoRA strength between asynchronous
   requests. Branch identity must survive gradient-checkpoint recomputation;
   a branch switch before backward must not make recomputation use the wrong
   weights. Mixed physical batches require masked/grouped branch forwards or
   an equivalent per-item router, not a single batch-wide mutable flag.
3. Preserve separate conditional and negative KV caches. A cache filled with
   one role's adapter must never be reused under another role, including when
   the negative prompt changes or LoRA strength changes. Validate both cache
   on/off and reference-image paths.
4. In partitioned training, use the same noisy latent, sigma, region, spatial
   positions, and loss core for the detached reference and conditional
   prediction. Disable the conditional global adapter on the reference
   forward. Its checkpoint remains training-only and is ignored at inference.
5. Store and restore optimizer state with explicit branch parameter groups.
   On drop 0, no `U` optimizer group exists. Save/load must preserve this
   contract across resume; do not silently restore a shared-LoRA optimizer
   into the split topology. The future API parameter must be added
   OpenAPI-first and have its sole default in `backend/api/param_defaults.py`.
6. Report ordinary/guided/null losses and gradient norms by branch. A
   positive or guidance-target step with a nonzero `U` gradient is a failure;
   with drop 0, any change in `U` or its empty-prompt prediction is a failure.

## Expected limitations

The fixed empty reference removes one feedback path, not all sources of
instability. The conditional target itself is an extrapolation, and applying
CFG at inference can extrapolate it again. Partitioned training remains
non-equivalent to full-frame inference, including in the reference forward.
Whether a fixed reference improves CFG 1 fidelity, CFG 3/7 composition, or
negative-prompt behavior is empirical and must pass the paired rollout gate
before this mode becomes a default.
