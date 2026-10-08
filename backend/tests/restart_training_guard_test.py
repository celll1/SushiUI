"""A restart must not sever ownership of a live or starting training child."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from fastapi import BackgroundTasks, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None
from api import routes
from core.training.training_process import TrainingProcessManager


@pytest.mark.parametrize("child", [None, SimpleNamespace(returncode=None)])
def test_live_and_starting_children_block_restart(child):
    manager = TrainingProcessManager()
    manager.processes[173] = SimpleNamespace(process=child, is_running=False)
    tasks = BackgroundTasks()
    with patch.object(routes, "training_process_manager", manager), patch.object(routes.subprocess, "Popen") as launch:
        with pytest.raises(HTTPException) as error:
            asyncio.run(routes.restart_backend(tasks))
    assert error.value.status_code == 409 and "173" in error.value.detail
    launch.assert_not_called()
    assert not tasks.tasks


def test_terminal_child_allows_restart_without_executing_background_exit():
    manager = TrainingProcessManager()
    manager.processes[173] = SimpleNamespace(process=SimpleNamespace(returncode=0), is_running=True)
    tasks = BackgroundTasks()
    with patch.object(routes, "training_process_manager", manager), patch.object(routes.subprocess, "Popen") as launch:
        result = asyncio.run(routes.restart_backend(tasks))
    assert result["success"] is True and tasks.tasks
    launch.assert_called_once()
