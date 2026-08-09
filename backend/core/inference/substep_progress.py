"""Sub-step progress for architectures whose ONE denoise step takes minutes.

MiniMax-H3 runs a 50-block DiT once per step at ~150s/step, so a progress bar
driven by the per-step callback alone sits still for the whole step and reads as
a hang. This module ticks progress from INSIDE a step by watching the block
stack with ``register_forward_hook``: the hook covers every block-loop path an
architecture has (MiniMax-H3 has two -- the block-swap wrapper's re-owned loop
and the vendor fast path) without either loop being rewritten.

The tick is delivered through the SAME ``progress_callback`` the sampler already
has, as an optional third argument ``sub_progress`` (0..1, the fraction of the
step in flight). Callbacks that do not accept it are not called with it -- see
``SubStepReporter._emit``.

ACCURACY LIMIT (deliberate): CUDA is asynchronous, so with block swap disabled
the hooks fire as the blocks are ENQUEUED, not as they finish, and
``sub_progress`` runs ahead of wall-clock. Nothing here synchronises: a
``torch.cuda.synchronize()`` per block would serialise the denoise loop to buy
a smoother bar. With block swap on, the loop's own ``wait_for_block`` keeps the
hooks roughly in step with execution.

Generic on purpose: any module exposing ``transformer_blocks`` (LTX-2.3, the
image DiTs) can be attached the same way. Only MiniMax-H3 is wired today.
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Optional

ProgressCallback = Callable[..., Any]


class SubStepReporter:
    """Emits throttled ``sub_progress`` ticks within one denoise step.

    ``begin_step(i, total)`` at the top of each iteration (``i`` 0-based, so it
    is also the number of COMPLETED steps -- the value the ``step`` field of the
    WS ``progress`` message carries), ``on_block(k, n_blocks)`` per block, and
    ``close()`` to remove the hooks. ``close()`` must run in the caller's
    ``finally``: hooks left on the module would fire again, against a stale
    callback, on the next generation.
    """

    def __init__(
        self,
        progress_callback: Optional[ProgressCallback],
        *,
        min_interval: float = 0.2,
        ticks_per_step: int = 8,
        label: str = "substep",
    ):
        self._callback = progress_callback
        self._min_interval = float(min_interval)
        self._ticks_per_step = max(1, int(ticks_per_step))
        self._label = label
        self._handles: List[Any] = []
        self._step = 0
        self._total = 0
        self._active = False
        self._last_sent = 0.0
        self._reported_error = False

    # -- lifecycle ---------------------------------------------------------

    def track_handle(self, handle: Any) -> None:
        """Take ownership of a ``RemovableHandle`` so ``close()`` removes it."""
        self._handles.append(handle)

    def begin_step(self, step_index: int, total_steps: int) -> None:
        self._step = int(step_index)
        self._total = int(total_steps)
        self._active = self._callback is not None and self._total > 0
        # Start the throttle window at the step boundary: the per-step callback
        # has just fired, so the first mid-step tick is at least
        # ``min_interval`` later rather than immediately after it.
        self._last_sent = time.monotonic()

    def close(self) -> None:
        self._active = False
        handles, self._handles = self._handles, []
        for handle in handles:
            try:
                handle.remove()
            except Exception as exc:  # teardown must never take a generation down
                print(f"[{self._label}] hook removal raised: {exc}")

    # -- ticking -----------------------------------------------------------

    def on_block(self, k: int, n_blocks: int) -> None:
        """Report that block ``k`` of ``n_blocks`` (1-based) just ran."""
        if not self._active or n_blocks <= 0:
            return
        # k == n_blocks is the step boundary, which the sampler's own
        # `progress_callback(i + 1, total)` reports a moment later.
        if k >= n_blocks:
            return
        if k % max(1, n_blocks // self._ticks_per_step) != 0:
            return
        now = time.monotonic()
        if now - self._last_sent < self._min_interval:
            return
        self._last_sent = now
        self._emit(k / n_blocks)

    def _emit(self, fraction: float) -> None:
        try:
            self._callback(self._step, self._total, sub_progress=fraction)
        except TypeError:
            # A callback that predates `sub_progress`. Stop trying: the
            # per-step ticks still go through the sampler's own call.
            self._active = False
        except Exception as exc:  # progress must never take a generation down
            self._active = False
            if not self._reported_error:
                self._reported_error = True
                print(f"[{self._label}] sub-progress callback raised: {exc}")


def attach_block_substep_hooks(
    transformer: Any,
    progress_callback: Optional[ProgressCallback],
    *,
    min_interval: float = 0.2,
    ticks_per_step: int = 8,
    label: str = "substep",
) -> SubStepReporter:
    """Hook every entry of ``transformer.transformer_blocks`` for sub-progress.

    Returns a reporter that is inert (but still safe to ``begin_step`` /
    ``close``) when there is no callback or no block list. Attach AFTER any
    block-swap wrapping: the wrapper delegates ``transformer_blocks`` to the
    same module objects, so either order works for MiniMax-H3, but an
    architecture that replaces the list would not survive the other order.
    """
    reporter = SubStepReporter(
        progress_callback,
        min_interval=min_interval,
        ticks_per_step=ticks_per_step,
        label=label,
    )
    if progress_callback is None:
        return reporter

    blocks = getattr(transformer, "transformer_blocks", None)
    n_blocks = len(blocks) if blocks is not None else 0
    if not n_blocks:
        return reporter

    def make_hook(k: int):
        def hook(_module, _args, _output):
            reporter.on_block(k, n_blocks)
        return hook

    for index, block in enumerate(blocks):
        try:
            reporter.track_handle(block.register_forward_hook(make_hook(index + 1)))
        except Exception as exc:  # a module that cannot be hooked is not fatal
            print(f"[{label}] could not hook block {index}: {exc}")
            break
    return reporter
