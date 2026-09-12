"""Safe argv construction for locally configured dataset editors."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence


DATASET_PATH_TOKEN = "{dataset}"


def build_editor_argv(
    executable: str,
    arguments: Sequence[str],
    dataset_path: str,
) -> list[str]:
    command = os.path.abspath(executable)
    if not os.path.isfile(command):
        raise ValueError("Configured dataset editor executable does not exist")
    if not os.path.isdir(dataset_path):
        raise ValueError("Dataset directory does not exist")
    if len(arguments) > 32 or any(not isinstance(value, str) for value in arguments):
        raise ValueError("Dataset editor arguments must be a list of at most 32 strings")
    expanded = [value.replace(DATASET_PATH_TOKEN, dataset_path) for value in arguments]
    if not any(DATASET_PATH_TOKEN in value for value in arguments):
        expanded.append(dataset_path)
    return [command, *expanded]


def launch_editor(executable: str, arguments: Sequence[str], dataset_path: str) -> None:
    argv = build_editor_argv(executable, arguments, dataset_path)
    subprocess.Popen(
        argv,
        cwd=dataset_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
