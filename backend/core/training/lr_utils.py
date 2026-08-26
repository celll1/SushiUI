"""Learning-rate re-assertion for trainer resume paths.

Shared by ``BaseTrainer.train()`` (image/video/audio trainers) and
``VaeTrainer.load_checkpoint()``. Lives here rather than inside either trainer
because both need identical semantics and neither should import the other:
``base_trainer`` is the base class of every image trainer, ``vae/vae_trainer``
is a standalone sibling, and this module depends on nothing in the package
(pure duck-typing over ``torch.optim.Optimizer`` / ``LRScheduler``), so it can
never introduce an import cycle.

Why any of this is needed
-------------------------
Two different failure modes converge on the same fix.

1. ``VaeTrainer`` restores BOTH the optimizer and the scheduler from the
   checkpoint, and torch silently re-imports the checkpointed LR in both:

   * ``Optimizer.load_state_dict`` rebuilds each param group taking only
     ``params`` from the live group and *every other key*, ``lr`` included,
     from the SAVED group;
   * ``LRScheduler.load_state_dict`` is ``self.__dict__.update(state_dict)``
     and ``state_dict()`` carries ``base_lrs``, so the scheduler reverts to the
     checkpoint's base LR and re-writes it into the param groups on every
     subsequent ``step()``.

   Net effect without a re-assertion: editing ``train.lr`` and resuming is a
   silent no-op.

2. ``BaseTrainer`` never loads scheduler state (it fast-forwards ``step()``),
   but it *does* call ``load_optimizer_state()``, which imports the
   checkpoint's ``lr`` into the param groups. Re-writing the configured LR
   there **flat** -- with no schedule multiplier -- makes the first optimizer
   step after a resume run at the un-multiplied base LR, because the training
   loop calls ``optimizer.step()`` before ``lr_scheduler.step()`` and a
   mid-epoch resume slices the batch list rather than iterating it. Mid-warmup
   that error is unbounded; in a cosine tail it is 1/floor_ratio.

What is deliberately NOT touched: the schedule *position*
(``last_epoch`` / ``_step_count``) and the optimizer *moments*
(``exp_avg`` / ``exp_avg_sq``, step counters). Those are precisely what a
resume exists to preserve.

Why replacing ``base_lrs`` is schedule-preserving
-------------------------------------------------
Every scheduler this project builds is a ``LambdaLR``: all seven types from
``diffusers.optimization.get_scheduler`` are, and so is the in-house
``plateau_cosine_floor`` (``BaseTrainer._build_plateau_cosine_floor_scheduler``).
For a ``LambdaLR``, ``lr = base_lr * f(last_epoch)`` -- ``base_lrs`` is purely a
scale, so replacing it rescales the schedule without moving along it. No
adaptive scheduler (``ReduceLROnPlateau``-style), whose state would legitimately
carry the LR itself, is constructed anywhere in this codebase. For anything that
is not a ``LambdaLR`` the multiplier is skipped and the plain base LR is written,
which is the pre-existing behaviour.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

__all__ = ["resolve_group_lrs", "reassert_config_lr"]

LrSpec = Union[float, int, Sequence[float]]


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def resolve_group_lrs(
    n_groups: int,
    cfg_lr: LrSpec,
    fallback_lr: Optional[float] = None,
    log_prefix: str = "[Trainer]",
    what: str = "param groups",
) -> List[float]:
    """Expand ``cfg_lr`` to exactly ``n_groups`` base learning rates.

    ``cfg_lr`` is either

    * a **scalar** -- broadcast to every group (the VAE trainer's single-group
      case), or
    * a **sequence** of per-group LRs, in optimizer param-group order
      (``BaseTrainer._configured_component_lr_description()``).

    A scalar is broadcast, so callers with more than one param group must pass a
    sequence: a scalar there assigns every component the same rate. That is what
    ``BaseTrainer`` used to do whenever its component list came out empty.

    Length contract, matching what ``BaseTrainer`` did inline before this was
    hoisted:

    * shorter than ``n_groups`` -> trailing groups take ``fallback_lr``
      (``self.learning_rate``). ``fallback_lr`` is then REQUIRED; omitting it
      is a caller bug and raises ``ValueError``.
    * longer than ``n_groups`` -> the leading ``n_groups`` entries are used and
      a warning is printed. Not fatal on purpose, though unreachable from
      ``BaseTrainer`` today: ``_configured_component_lr_description`` (since
      4e8edb62) only ever returns a list whose length already equals
      ``n_groups``, for exactly this reason -- both consumers write by index,
      so a mismatched length would assign some component another component's
      rate. Kept non-fatal for other callers that build ``cfg_lr`` differently.
    """
    if n_groups <= 0:
        return []

    if _is_scalar(cfg_lr):
        return [float(cfg_lr)] * n_groups

    values = [float(v) for v in cfg_lr]

    if len(values) > n_groups:
        print(f"{log_prefix} WARNING: {len(values)} configured LRs for "
              f"{n_groups} {what}; using the first {n_groups}")
        return values[:n_groups]

    if len(values) < n_groups:
        if fallback_lr is None:
            raise ValueError(
                f"reassert_config_lr: {len(values)} LRs supplied for {n_groups} "
                f"{what} and no fallback_lr was given"
            )
        values = values + [float(fallback_lr)] * (n_groups - len(values))

    return values


def reassert_config_lr(
    optimizer,
    lr_scheduler,
    cfg_lr: LrSpec,
    log_prefix: str = "[Trainer]",
    component_names: Optional[Sequence[str]] = None,
    fallback_lr: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[List[float], List[float]]:
    """Make the CONFIG's learning rate win over whatever a resume restored.

    Args:
        optimizer: the live optimizer, already resumed.
        lr_scheduler: the live scheduler, or ``None``. ``None`` is legitimate --
            ``VaeTrainer.build_optimizer`` falls back to a constant LR when
            ``get_scheduler`` raises -- in which case writing the param groups is
            the whole job.
        cfg_lr: scalar base LR, or a per-param-group sequence (see
            :func:`resolve_group_lrs`).
        log_prefix: trainer log prefix.
        component_names: optional per-group names for the log lines
            (e.g. ``["U-Net", "TE1"]``). Missing entries become ``group{i}``.
        fallback_lr: base LR for groups a short ``cfg_lr`` sequence does not
            cover.
        verbose: print the per-group summary.

    Returns:
        ``(prev_lrs, base_lrs)`` -- the LR each group carried on entry and the
        configured base LR now in force for it. The value actually written to a
        group is ``base_lr * schedule_multiplier(last_epoch)``.
    """
    groups = list(getattr(optimizer, "param_groups", []) or [])
    n = len(groups)
    if n == 0:
        return [], []

    prev_lrs = [float(g["lr"]) for g in groups]
    base_lrs = resolve_group_lrs(n, cfg_lr, fallback_lr, log_prefix)

    # ---- schedule multiplier at the CURRENT position -------------------
    # LambdaLR.get_lr() is `base_lr * lmbda(self.last_epoch)`, so evaluating the
    # lambdas at last_epoch reproduces exactly what the next scheduler.step()
    # would produce for this same position.
    multipliers = [1.0] * n
    lambdas = getattr(lr_scheduler, "lr_lambdas", None) if lr_scheduler is not None else None
    position = getattr(lr_scheduler, "last_epoch", None) if lr_scheduler is not None else None
    if lambdas and position is not None and len(lambdas) == n:
        try:
            multipliers = [float(fn(position)) for fn in lambdas]
        except Exception as exc:  # pragma: no cover - defensive
            print(f"{log_prefix} WARNING: could not evaluate the LR schedule at "
                  f"step {position} ({exc}); applying the base LR unscaled")
            multipliers = [1.0] * n

    # ---- write the param groups ----------------------------------------
    for group, base, mult in zip(groups, base_lrs, multipliers):
        group["lr"] = base * mult
        # LambdaLR re-reads initial_lr whenever a scheduler is rebuilt with
        # last_epoch != -1, and Optimizer.load_state_dict may have just restored
        # the checkpoint's value into the group.
        if "initial_lr" in group:
            group["initial_lr"] = base

    # ---- write the scheduler's base_lrs --------------------------------
    if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
        sched_bases = resolve_group_lrs(
            len(lr_scheduler.base_lrs), cfg_lr, fallback_lr, log_prefix,
            what="scheduler base_lrs",
        )
        for i, value in enumerate(sched_bases):
            lr_scheduler.base_lrs[i] = value
        lr_scheduler._last_lr = [float(g["lr"]) for g in groups]

    # ---- report ---------------------------------------------------------
    if verbose:
        names = list(component_names or [])
        for i, (prev, base, mult) in enumerate(zip(prev_lrs, base_lrs, multipliers)):
            name = names[i] if i < len(names) else f"group{i}"
            applied = groups[i]["lr"]
            scaling = "" if mult == 1.0 else f" = base {base:.3e} x schedule {mult:.4g}"
            if applied != prev:
                print(f"{log_prefix} LR {name}: config overrides the checkpoint's: "
                      f"{prev:.3e} -> {applied:.3e}{scaling} "
                      f"(optimizer moments/EMA/RNG preserved)")
            else:
                print(f"{log_prefix} LR {name}: unchanged at {applied:.3e}{scaling}, "
                      f"re-applied at the schedule's current position")

    return prev_lrs, base_lrs
