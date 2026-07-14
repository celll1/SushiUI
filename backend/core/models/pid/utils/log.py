"""
Minimal stand-in for `pid._ext.imaginaire.utils.log` (the original wraps `loguru`).
Not vendored from NVIDIA source — this is a trivial standard-library `logging`
replacement written for this port, exposing only the functions the vendored
network code (`pid_net.py: init_weights()`) actually calls.
"""

import logging

_logger = logging.getLogger("sushiui.pid")


def debug(message: str, rank0_only: bool = True) -> None:
    _logger.debug(message)


def info(message: str, rank0_only: bool = True) -> None:
    _logger.info(message)


def warning(message: str, rank0_only: bool = True) -> None:
    _logger.warning(message)


def error(message: str, rank0_only: bool = True) -> None:
    _logger.error(message)
