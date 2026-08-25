"""Per-step census of which parameters an update actually reached.

Under the fused backward pass ``optimizer.step()`` is never called: every update
is applied from that parameter's own post-accumulate-grad hook. A parameter
whose hook returns early -- or never fires -- is therefore updated by nothing at
all for the whole run, while the loss keeps falling normally. That failure mode
has been found repeatedly in this code base (3a7c9560's silent CPU-skip in the
hooks, the bf16 round-to-nearest defect), and it is invisible to loss curves by
construction.

This module records the count directly: the update sites call ``record`` after
applying an update, and ``assert_complete`` compares the set against the
parameters the optimizer owns and that require a gradient. It is the mechanism
behind G-RB3 in docs/guides/SENSENOVA_TRAINING_DESIGN.md 6.5.

Cost when enabled: one ``set.add(int)`` per parameter per step, plus a set
difference per step. No device work and no synchronisation -- deliberately, so
it can be left on during a real run. Measured at 47.8 us/step over 588
parameters (81 ns/param).

WHAT THIS DOES AND DOES NOT GUARANTEE
-------------------------------------
The expectation set is built from ``optimizer.param_groups``, so what passes is
"every parameter THE OPTIMIZER OWNS received an update". It is NOT "every
trainable parameter of the model was updated": a parameter the optimizer does
not own is invisible here, exactly as it is to hook registration, which is
driven by the same param_groups. That other direction is a separate check and
already exists -- ``fused_backward_registration`` walks the module and refuses a
trainable parameter that is in no param_group. The two together cover both
directions; neither covers both alone.

STRUCTURALLY UNREACHABLE PARAMETERS
-----------------------------------
Some architectures own parameters that no gradient can reach by construction --
not a defect, the model's shape. SenseNova's understanding branch has five: a
prefix forward keeps ``past_key_values`` and discards ``last_hidden_state``, so
the last layer's post-attention half feeds nothing
(``sensenova_lora.und_gradient_unreachable_paths``). They are ``requires_grad``
and owned by the optimizer, so a census that demanded them would raise on every
step of a correct run. ``expect(..., exempt=...)`` takes their names.

Keys are ``id(param)``, following ``fused_grad_norm``: names come from the
expectation set, which is built once.
"""

from typing import Dict, Iterable, List, Optional, Set

import torch.nn as nn

CENSUS_ATTR = "_sushiui_update_census"


class UpdateCensus:
    """``{id(param)}`` updated during the current step, against what was expected."""

    def __init__(self) -> None:
        self._updated: Set[int] = set()
        self._expected: Dict[int, str] = {}
        self.exempt: Set[str] = set()
        self.enabled = False
        self.steps_checked = 0

    # -- expectation set ---------------------------------------------------

    def expect(
        self,
        params: Iterable[nn.Parameter],
        names: Optional[Dict[int, str]] = None,
        exempt: Optional[Iterable[str]] = None,
    ) -> int:
        """Declare the parameters that must be updated every step.

        ``exempt`` names parameters that no gradient can reach by construction
        (see the module docstring). A name matches if it equals, or is a
        dot-separated prefix of, the parameter's name -- so a module path like
        ``...layers.41.self_attn.q_proj`` covers its ``.weight`` and ``.bias``.
        Exempt parameters are dropped from the expectation set rather than
        excused at assert time, so ``expected_count`` reports what is genuinely
        required.

        Replaces any previous expectation. Returns how many are expected.
        """
        names = names or {}
        self.exempt = set(exempt or ())
        expected: Dict[int, str] = {}
        for p in params:
            name = names.get(id(p), f"<unnamed {tuple(p.shape)} {p.dtype}>")
            if self._is_exempt(name):
                continue
            expected[id(p)] = name
        self._expected = expected
        return len(self._expected)

    def _is_exempt(self, name: str) -> bool:
        if name in self.exempt:
            return True
        return any(name.startswith(prefix + ".") for prefix in self.exempt)

    @property
    def expected_count(self) -> int:
        return len(self._expected)

    @property
    def updated_count(self) -> int:
        return len(self._updated)

    # -- per-step ----------------------------------------------------------

    def begin_step(self, enabled: bool = True) -> None:
        self._updated.clear()
        self.enabled = bool(enabled)

    def record(self, param: nn.Parameter) -> None:
        self._updated.add(id(param))

    def missing(self) -> List[str]:
        """Expected parameters that no update reached this step."""
        return sorted(
            name for key, name in self._expected.items() if key not in self._updated
        )

    def unexpected_count(self) -> int:
        """Updates recorded for parameters that are not in the expectation set."""
        return len(self._updated - set(self._expected))

    def assert_complete(self, context: str = "") -> None:
        """Raise unless every expected parameter was updated this step."""
        if not self.enabled:
            return
        self.steps_checked += 1
        missing = self.missing()
        if not missing:
            return
        where = f" ({context})" if context else ""
        raise RuntimeError(
            f"Updated-parameter census failed{where}: {len(missing)} of "
            f"{self.expected_count} trainable parameter(s) received no optimizer "
            f"update this step, e.g. {missing[:5]}. Under the fused backward pass "
            f"optimizer.step() is never called, so a parameter no hook updated is "
            f"updated by nothing for the whole run -- and the loss falls normally "
            f"regardless. Check that every parameter the optimizer owns carries a "
            f"post-accumulate-grad hook and that none of them returns early."
        )


# -- attachment ------------------------------------------------------------


def attach_update_census(optimizer, census: Optional[UpdateCensus]) -> None:
    setattr(optimizer, CENSUS_ATTR, census)


def get_update_census(optimizer) -> Optional[UpdateCensus]:
    return getattr(optimizer, CENSUS_ATTR, None)


def record_param_update(optimizer, param: nn.Parameter) -> None:
    """Record that ``param`` was updated, if a census is attached and armed."""
    census = getattr(optimizer, CENSUS_ATTR, None)
    if census is not None and census.enabled:
        census.record(param)


def trainable_params_of(optimizer) -> List[nn.Parameter]:
    """The parameters the optimizer owns that a gradient can reach.

    The same set ``fused_backward_registration`` hooks: driven by param_groups
    (not by a module walk), deduplicated, ``requires_grad`` only.
    """
    seen: Set[int] = set()
    out: List[nn.Parameter] = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) in seen or not param.requires_grad:
                continue
            seen.add(id(param))
            out.append(param)
    return out


def enable_update_census(
    optimizer,
    module: Optional[nn.Module] = None,
    exempt: Optional[Iterable[str]] = None,
) -> UpdateCensus:
    """Attach an armed census expecting every trainable parameter of ``optimizer``.

    ``module`` supplies parameter names, for the failure message and for
    matching ``exempt`` (see ``UpdateCensus.expect``).
    """
    names = {id(p): n for n, p in module.named_parameters()} if module is not None else {}
    census = get_update_census(optimizer)
    if census is None:
        census = UpdateCensus()
        attach_update_census(optimizer, census)
    census.expect(trainable_params_of(optimizer), names, exempt=exempt)
    census.begin_step(True)
    return census
