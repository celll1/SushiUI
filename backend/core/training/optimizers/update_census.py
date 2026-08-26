"""Track optimizer-owned parameters updated by fused backward hooks.

The optional census checks every expected parameter after a backward. Exempt
parameters are structurally gradient-unreachable; deferred parameters remain
required, but only when their shared MNT window closes. Registration separately
checks the opposite direction: trainable parameters missing from the optimizer.

An always-on process-local ledger counts writes since the current backward
began, allowing OOM recovery to reject a retry after a partial fused step. See
SENSENOVA_TRAINING_DESIGN.md 6.5.
"""

from typing import Dict, Iterable, List, Optional, Set

import torch.nn as nn

CENSUS_ATTR = "_sushiui_update_census"


class UpdateCensus:
    """``{id(param)}`` updated during the current step, against what was expected."""

    def __init__(self) -> None:
        self._updated: Set[int] = set()
        self._expected: Dict[int, str] = {}
        self._deferred: Set[int] = set()
        self._expect_deferred = True
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
        """Replace the expectation set, excluding exact or prefix name matches."""
        names = names or {}
        self.exempt = set(exempt or ())
        expected: Dict[int, str] = {}
        for p in params:
            name = names.get(id(p), f"<unnamed {tuple(p.shape)} {p.dtype}>")
            if self._is_exempt(name):
                continue
            expected[id(p)] = name
        self._expected = expected
        self._deferred = set()
        return len(self._expected)

    def set_deferred(self, params: Iterable[nn.Parameter]) -> int:
        """Defer a nonempty, proper subset of expected parameters to window end."""
        keys = {id(p) for p in params} & set(self._expected)
        if not keys:
            raise ValueError(
                "Updated-parameter census: the deferred group does not intersect "
                "the expectation set, so deferring it would change nothing and "
                "the group it was meant to cover would still fail every step"
            )
        if keys == set(self._expected):
            raise ValueError(
                "Updated-parameter census: the deferred group is the whole "
                "expectation set, which would leave non-final steps checking "
                "nothing at all"
            )
        self._deferred = keys
        return len(keys)

    @property
    def deferred_count(self) -> int:
        return len(self._deferred)

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

    def begin_step(self, enabled: bool = True, expect_deferred: bool = True) -> None:
        self._updated.clear()
        self.enabled = bool(enabled)
        self._expect_deferred = bool(expect_deferred)

    def record(self, param: nn.Parameter) -> None:
        self._updated.add(id(param))

    def missing(self) -> List[str]:
        """Return expected parameters whose update is due but missing."""
        skip = frozenset() if self._expect_deferred else self._deferred
        return sorted(
            name
            for key, name in self._expected.items()
            if key not in self._updated and key not in skip
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
        if self._deferred:
            where += (
                f" [deferral window: {len(self._deferred)} parameter(s) deferred, "
                f"due {'THIS step' if self._expect_deferred else 'at the window end'}]"
            )
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
    note_update_applied()
    census = getattr(optimizer, CENSUS_ATTR, None)
    if census is not None and census.enabled:
        census.record(param)


# -- applied-update ledger (see the module docstring) -----------------------

_applied_updates = 0


def note_update_applied(count: int = 1) -> None:
    """An update has been WRITTEN to ``count`` parameter(s)."""
    global _applied_updates
    _applied_updates += count


def reset_applied_updates() -> None:
    """Open a new window. Belongs immediately before every ``backward()``."""
    global _applied_updates
    _applied_updates = 0


def applied_updates() -> int:
    return _applied_updates


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
