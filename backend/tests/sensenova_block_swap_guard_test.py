"""SenseNova's architecture hook delegates branch block-swap setup."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.arch.sensenova import SenseNovaArchHandler


def test_zero_block_swap_is_a_noop() -> None:
    SenseNovaArchHandler().setup_block_swap(SimpleNamespace(blocks_to_swap=0))


def test_nonzero_block_swap_delegates_to_ops(monkeypatch) -> None:
    from core.training.ops import sensenova_ops

    trainer = SimpleNamespace(blocks_to_swap=1)
    calls = []
    monkeypatch.setattr(sensenova_ops, "setup_block_swap", calls.append)
    SenseNovaArchHandler().setup_block_swap(trainer)
    assert calls == [trainer]
