"""SenseNova's unsupported block swap must not reject its zero-value default."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.arch.sensenova import SenseNovaArchHandler


def test_zero_block_swap_is_a_noop() -> None:
    SenseNovaArchHandler().setup_block_swap(SimpleNamespace(blocks_to_swap=0))


def test_nonzero_block_swap_remains_refused() -> None:
    with pytest.raises(NotImplementedError, match="block swap is not implemented"):
        SenseNovaArchHandler().setup_block_swap(SimpleNamespace(blocks_to_swap=1))
