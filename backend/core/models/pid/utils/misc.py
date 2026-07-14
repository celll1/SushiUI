"""
Minimal stand-ins for the two `pid._ext.imaginaire.utils` helpers referenced by the
vendored inference code (`misc.timer` context manager, standard-library `logging`
in place of the original's `loguru`-based `log` module). Not vendored from NVIDIA
source — these are trivial no-op/pass-through replacements written for this port.
"""

import logging
from contextlib import AbstractContextManager

logger = logging.getLogger("sushiui.pid")


class timer(AbstractContextManager):
    """No-op timing context manager (drop-in replacement for `imaginaire.utils.misc.timer`).

    The original logs elapsed wall-clock time on exit; SushiUI's vendored port has
    no use for that instrumentation, so this is a pure no-op context manager kept
    only so call sites like `with misc.timer("PixelDiTModel: build_net"):` still work.
    """

    def __init__(self, context: str, debug: bool = False):
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def set_random_seed(seed: int, by_rank: bool = False) -> None:
    """Minimal seed helper (torch-only; no distributed rank offset)."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
