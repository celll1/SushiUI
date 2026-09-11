from __future__ import annotations

import asyncio
import contextvars
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import torch

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from api.routes import _run_generation_in_executor


def test_generation_executor_propagates_context_without_grad():
    request_id = contextvars.ContextVar("request_id")
    token = request_id.set("generation-1")

    async def run():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await _run_generation_in_executor(
                loop,
                executor,
                lambda: (request_id.get(), torch.is_grad_enabled()),
            )

    try:
        assert torch.is_grad_enabled()
        assert asyncio.run(run()) == ("generation-1", False)
        assert torch.is_grad_enabled()
    finally:
        request_id.reset(token)
