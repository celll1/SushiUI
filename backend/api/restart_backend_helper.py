"""Wait for a Windows backend process to exit, then start its replacement."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import os
import subprocess
from typing import Callable, Sequence


def build_helper_command(
    *,
    python_executable: str,
    helper_path: str,
    parent_pid: int,
    main_path: str,
    backend_dir: str,
) -> list[str]:
    """Build the argv used to start this helper without shell parsing."""
    return [
        os.path.abspath(python_executable),
        os.path.abspath(helper_path),
        "--parent-pid",
        str(parent_pid),
        "--python-executable",
        os.path.abspath(python_executable),
        "--main-path",
        os.path.abspath(main_path),
        "--backend-dir",
        os.path.abspath(backend_dir),
    ]


def helper_creation_flags() -> int:
    """Detach the waiter while leaving the replacement free to own a console."""
    return subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP


def wait_for_process_exit(pid: int) -> None:
    """Wait on the exact Windows process object, avoiding PID-reuse polling."""
    if os.name != "nt":
        raise RuntimeError("The backend restart helper is Windows-only")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    synchronize = 0x00100000
    infinite = 0xFFFFFFFF
    wait_failed = 0xFFFFFFFF
    error_invalid_parameter = 87

    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == error_invalid_parameter:
            return
        raise ctypes.WinError(error)

    try:
        result = kernel32.WaitForSingleObject(handle, infinite)
        if result == wait_failed:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)


def launch_backend_after_parent_exit(
    *,
    parent_pid: int,
    python_executable: str,
    main_path: str,
    backend_dir: str,
    wait_fn: Callable[[int], None] = wait_for_process_exit,
    popen: Callable[..., object] = subprocess.Popen,
) -> object:
    """Start the replacement only after the old process is signalled."""
    wait_fn(parent_pid)
    return popen(
        [os.path.abspath(python_executable), os.path.abspath(main_path)],
        cwd=os.path.abspath(backend_dir),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        close_fds=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--main-path", required=True)
    parser.add_argument("--backend-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    launch_backend_after_parent_exit(
        parent_pid=args.parent_pid,
        python_executable=args.python_executable,
        main_path=args.main_path,
        backend_dir=args.backend_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
