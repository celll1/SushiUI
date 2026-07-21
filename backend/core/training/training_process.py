"""
Training process management for ai-toolkit integration.

Handles subprocess execution, log monitoring, and progress tracking.
"""

import asyncio
import subprocess
import re
import os
import sys
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime


def _is_venv_interpreter(python_path: Path) -> bool:
    """Best-effort check that *python_path* is a venv interpreter.

    A venv's ``Scripts``/``bin`` directory always contains a ``pyvenv.cfg``
    one level up (``<venv_root>/pyvenv.cfg``). This is the same marker
    ``sys.prefix`` vs ``sys.base_prefix`` relies on, but works for an
    arbitrary interpreter path (not just the currently running one).
    """
    try:
        return (python_path.parent.parent / "pyvenv.cfg").is_file()
    except OSError:
        return False


def resolve_venv_python() -> str:
    """Resolve the interpreter to use for spawning the training subprocess.

    Priority:
      1. If the *currently running* backend process is itself inside a venv
         (``sys.prefix != sys.base_prefix``), reuse ``sys.executable``. This
         is the common case and requires no filesystem guessing.
      2. Otherwise (backend was started with a non-venv interpreter, e.g. a
         system Python), fall back to ``<repo_root>/venv/Scripts/python.exe``
         (Windows) or ``<repo_root>/venv/bin/python`` (POSIX), where
         ``repo_root`` is derived from this file's location
         (``backend/core/training/training_process.py`` -> repo root is
         three parents up). No hardcoded absolute path — works regardless
         of where the repo is cloned.

    Raises:
        FileNotFoundError: if neither the running interpreter nor the
            repo-relative venv interpreter can be found. Silently falling
            back to whatever "python" resolves to on PATH would risk
            re-introducing the system-Python bug this function fixes.
    """
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        # Running interpreter is already a venv (or virtualenv) Python.
        return sys.executable

    # Backend itself is not running inside a venv: don't propagate that via
    # sys.executable. Resolve the repo's own venv by path instead.
    repo_root = Path(__file__).resolve().parents[3]  # backend/core/training/ -> repo root
    if sys.platform == "win32":
        candidate = repo_root / "venv" / "Scripts" / "python.exe"
    else:
        candidate = repo_root / "venv" / "bin" / "python"

    if candidate.is_file() and _is_venv_interpreter(candidate):
        return str(candidate)

    raise FileNotFoundError(
        "Could not resolve a venv Python interpreter for the training "
        f"subprocess. sys.executable={sys.executable!r} is not a venv "
        f"interpreter, and no venv was found at {candidate!r}. Refusing to "
        "fall back to a bare 'python' on PATH (see CLAUDE.md: always use "
        "the project venv interpreter)."
    )


class TrainingProcess:
    """Manages a single training process."""

    def __init__(
        self,
        run_id: int,
        config_path: str,
        output_dir: str,
        venv_python: str = None,
    ):
        """
        Initialize training process.

        Args:
            run_id: Training run ID
            config_path: Path to YAML config file
            output_dir: Output directory for checkpoints
            venv_python: Path to venv Python executable. If not given, it is
                resolved via ``resolve_venv_python()`` (prefers the currently
                running interpreter if it is itself a venv Python, otherwise
                falls back to ``<repo_root>/venv/Scripts/python.exe`` /
                ``venv/bin/python`` derived from this file's location).
        """
        self.run_id = run_id
        self.config_path = config_path
        self.output_dir = output_dir
        self.venv_python = venv_python or resolve_venv_python()

        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.is_user_stopped = False  # Track if user requested stop
        self.current_step = 0
        self.current_loss: Optional[float] = None
        self.current_lr: Optional[float] = None

    async def start(
        self,
        progress_callback: Optional[Callable[[int, float, float], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Start training process.

        Args:
            progress_callback: Callback(step, loss, lr) for progress updates
            log_callback: Callback(log_line) for log streaming
        """
        if self.is_running:
            raise RuntimeError("Training process is already running")

        # Construct SushiUI training command
        # Run as script directly instead of module
        backend_dir = Path(__file__).parent.parent.parent
        train_runner_path = backend_dir / "core" / "training" / "train_runner.py"

        cmd = [
            self.venv_python,
            str(train_runner_path),
            self.config_path,
            str(self.run_id),
        ]

        # Set environment variables
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Disable buffering for real-time logs
        # Add backend directory to PYTHONPATH so imports work
        env["PYTHONPATH"] = str(backend_dir) + os.pathsep + env.get("PYTHONPATH", "")

        # Pre-spawn cleanup: a leftover .stop_training flag from a previous run
        # in the same output_dir (e.g. a prior stop request that raced with, or
        # arrived just after, that run's own exit) would otherwise instakill this
        # fresh run's initialization the moment train_runner.py's new init-phase
        # stop checks (_check_init_stop) see it. base_trainer.train() already
        # clears a stale flag once it starts (belt-and-suspenders double-guard,
        # kept as-is), but that happens well after dataset loading/bucketing, i.e.
        # after the new checks run -- so clear it here too, before the process
        # that will actually observe it is even spawned.
        stale_stop_flag = Path(self.output_dir) / ".stop_training"
        try:
            stale_stop_flag.unlink()
            print(f"[Training] Removed stale stop flag before spawn: {stale_stop_flag}")
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"[Training] WARNING: Failed to remove stale stop flag before spawn: {e}")

        # Start asyncio subprocess (non-blocking)
        # Increase buffer limit to handle long tqdm progress bars (default is 64KB)
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(backend_dir),
            limit=1024 * 1024,  # 1MB buffer to handle long progress bars
        )

        self.is_running = True

        # Monitor logs in background
        asyncio.create_task(self._monitor_logs(progress_callback, log_callback))

    async def _monitor_logs(
        self,
        progress_callback: Optional[Callable[[int, float, float], None]],
        log_callback: Optional[Callable[[str], None]],
    ) -> None:
        """
        Monitor training logs and extract progress information.

        Args:
            progress_callback: Callback for progress updates
            log_callback: Callback for log streaming
        """
        if not self.process or not self.process.stdout:
            return

        # Regex patterns for log parsing
        step_pattern = re.compile(r"step:\s*(\d+)")
        loss_pattern = re.compile(r"loss:\s*([\d.]+)")
        lr_pattern = re.compile(r"lr:\s*([\d.e-]+)")

        try:
            # Use async iteration for non-blocking I/O
            while True:
                try:
                    line_bytes = await self.process.stdout.readline()
                except (asyncio.LimitOverrunError, ValueError) as e:
                    # Line too long (exceeds buffer limit)
                    # This can happen with very long progress bars or debug output
                    # ValueError is raised as a wrapper for LimitOverrunError in some Python versions
                    print(f"[Training] Warning: Skipping oversized log line (buffer overflow: {type(e).__name__})")
                    # Read and discard the oversized line in chunks until we find a newline
                    try:
                        while True:
                            chunk = await self.process.stdout.read(8192)
                            if not chunk or b'\n' in chunk:
                                break
                    except Exception as read_error:
                        print(f"[Training] Warning: Error while discarding oversized line: {read_error}")
                        # If we can't even read chunks, skip and continue
                    continue

                if not line_bytes:
                    break

                # Decode with error handling (ignore invalid UTF-8 bytes)
                try:
                    line = line_bytes.decode('utf-8').strip()
                except UnicodeDecodeError as e:
                    # Log the problematic bytes for debugging
                    print(f"[Training] UnicodeDecodeError at position {e.start}-{e.end}: {line_bytes[max(0, e.start-10):e.end+10].hex()}")
                    print(f"[Training] Full line (hex): {line_bytes.hex()}")
                    # Replace invalid bytes and continue
                    line = line_bytes.decode('utf-8', errors='replace').strip()

                # Send log to callback
                if log_callback:
                    log_callback(line)

                # Parse progress information
                step_match = step_pattern.search(line)
                loss_match = loss_pattern.search(line)
                lr_match = lr_pattern.search(line)

                if step_match:
                    self.current_step = int(step_match.group(1))

                if loss_match:
                    self.current_loss = float(loss_match.group(1))

                if lr_match:
                    self.current_lr = float(lr_match.group(1))

                # Trigger progress callback
                if progress_callback and step_match:
                    progress_callback(
                        self.current_step,
                        self.current_loss or 0.0,
                        self.current_lr or 0.0,
                    )

            # Wait for process to complete (async)
            returncode = await self.process.wait()

            # Check if process failed (but distinguish user stop from error)
            if returncode != 0:
                print(f"[Training] Process exited with code {returncode}")
                if self.is_user_stopped:
                    print(f"[Training] Process was stopped by user")
                    # Signal stopped status (step=-2 indicates user stop)
                    if progress_callback:
                        progress_callback(-2, 0.0, 0.0)
                else:
                    print(f"[Training] Process failed")
                    # Signal failure (step=-1 indicates error)
                    if progress_callback:
                        progress_callback(-1, 0.0, 0.0)

        except Exception as e:
            print(f"[Training] Error monitoring logs: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.is_running = False
            print(f"[Training] Process monitoring ended. Final returncode: {self.process.returncode if self.process else 'N/A'}")

    async def stop(self) -> None:
        """Stop training process."""
        if self.process and self.is_running:
            print(f"[Training] Stopping process (user requested)")
            self.is_user_stopped = True  # Mark as user-requested stop

            # Create stop flag file for graceful shutdown (works on Windows)
            stop_flag_file = Path(self.output_dir) / ".stop_training"
            try:
                stop_flag_file.touch()
                print(f"[Training] Created stop flag file: {stop_flag_file}")
            except Exception as e:
                print(f"[Training] WARNING: Failed to create stop flag file: {e}")

            # Wait for graceful shutdown (no timeout - MNT batches can take several minutes)
            # With MNT=32 and large models, a single batch can take 5+ minutes
            # Checkpoint save can also take 60+ seconds for large models (12GB+ safetensors + optimizer state)
            print(f"[Training] Waiting for graceful shutdown (no timeout, press Ctrl+C in terminal to force kill)...")
            await self.process.wait()
            print(f"[Training] Process terminated gracefully")

            self.is_running = False

    def get_status(self) -> Dict[str, Any]:
        """
        Get current training status.

        Returns:
            Dictionary with status information
        """
        return {
            "is_running": self.is_running,
            "current_step": self.current_step,
            "current_loss": self.current_loss,
            "current_lr": self.current_lr,
            "returncode": self.process.returncode if self.process else None,
        }


class TrainingProcessManager:
    """Manages multiple training processes."""

    def __init__(self):
        self.processes: Dict[int, TrainingProcess] = {}

    def create_process(
        self,
        run_id: int,
        config_path: str,
        output_dir: str,
        venv_python: str = None,
    ) -> TrainingProcess:
        """
        Create and register a training process.

        Args:
            run_id: Training run ID
            config_path: Path to YAML config file
            output_dir: Output directory
            venv_python: Path to venv Python executable

        Returns:
            TrainingProcess instance
        """
        process = TrainingProcess(run_id, config_path, output_dir, venv_python)
        self.processes[run_id] = process
        return process

    def get_process(self, run_id: int) -> Optional[TrainingProcess]:
        """Get training process by run ID."""
        return self.processes.get(run_id)

    async def remove_process(self, run_id: int) -> None:
        """Remove training process from registry."""
        if run_id in self.processes:
            process = self.processes[run_id]
            if process.is_running:
                await process.stop()
            del self.processes[run_id]


# Global process manager
training_process_manager = TrainingProcessManager()
