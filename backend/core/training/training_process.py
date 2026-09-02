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

from core.training.training_events import split_training_event


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


def resolve_cuda_visible_devices(
    inherited: Optional[str], gpu_index: Optional[int]
) -> Optional[str]:
    """The child's CUDA_VISIBLE_DEVICES, or None to leave the parent's alone.

    Pinning a run to a GPU goes through the environment rather than a device
    argument because the trainer addresses its device as "cuda"/"cuda:0" in
    ~100 places; hiding the other GPUs makes every one of them resolve to the
    selected card with no code change.

    If the backend itself was launched with CUDA_VISIBLE_DEVICES, indices are
    already remapped, so ``gpu_index`` selects *within that list* rather than
    being a physical index. Composing the two is the only reading that cannot
    silently point at a different card than the one the UI offered.
    """
    if gpu_index is None:
        return None
    visible = [t.strip() for t in (inherited or "").split(",") if t.strip()]
    if not visible:
        return str(gpu_index)
    if gpu_index >= len(visible):
        raise ValueError(
            f"gpu_index={gpu_index} is outside the backend's "
            f"CUDA_VISIBLE_DEVICES={inherited} ({len(visible)} device(s) visible)"
        )
    return visible[gpu_index]


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

    # Distinct structured notices forwarded per run. Notices are "say once" by
    # construction; this only bounds a pathological emitter, and the dedup set
    # below is what a repeating one actually hits.
    MAX_EVENTS_PER_RUN = 200

    def __init__(
        self,
        run_id: int,
        config_path: str,
        output_dir: str,
        venv_python: str = None,
        gpu_index: Optional[int] = None,
    ):
        """
        Initialize training process.

        Args:
            run_id: Training run ID
            config_path: Path to YAML config file
            output_dir: Output directory for checkpoints
            gpu_index: Physical GPU index to pin the child to via
                CUDA_VISIBLE_DEVICES. None inherits the parent's visible
                devices. Because the child sees only the selected GPU, every
                "cuda"/"cuda:0" in the trainer resolves to it unchanged.
            venv_python: Path to venv Python executable. If not given, it is
                resolved via ``resolve_venv_python()`` (prefers the currently
                running interpreter if it is itself a venv Python, otherwise
                falls back to ``<repo_root>/venv/Scripts/python.exe`` /
                ``venv/bin/python`` derived from this file's location).
        """
        self.run_id = run_id
        self.config_path = config_path
        self.output_dir = output_dir
        self.gpu_index = gpu_index
        self.venv_python = venv_python or resolve_venv_python()

        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.is_user_stopped = False  # Track if user requested stop
        self.current_step = 0
        self.current_loss: Optional[float] = None
        self.current_lr: Optional[float] = None
        self._lifecycle_activity = f"training run {run_id}"
        self._lifecycle_activity_active = False
        # The monitor task owns the lifecycle-activity release once it is
        # spawned. asyncio only holds a weak reference to a running task, so
        # dropping this handle lets the GC collect it mid-run -- which would
        # skip its finally: and leak the activity, blocking every model load
        # until the backend restarts.
        self._monitor_task: Optional[asyncio.Task] = None
        # (level, code, message) already forwarded, and whether the cap notice
        # has been sent. Bounded by MAX_EVENTS_PER_RUN.
        self._events_seen: set = set()
        self._events_capped = False

    async def start(
        self,
        progress_callback: Optional[Callable[[int, float, float], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Start training process.

        Args:
            progress_callback: Callback(step, loss, lr) for progress updates
            log_callback: Callback(log_line) for log streaming
            event_callback: Callback(event) for structured notices the trainer
                emitted through ``core.training.training_events``. Deduped and
                capped per run; ordinary log lines never reach it.
        """
        if self.is_running:
            raise RuntimeError("Training process is already running")

        from core.model_state_coordinator import model_state_coordinator
        model_state_coordinator.begin_activity(self._lifecycle_activity)
        self._lifecycle_activity_active = True

        # Everything from here to the monitor-task handoff must release the
        # activity on failure. The activity gates model loads process-wide, so
        # any escape that skips the release blocks loading until restart.
        try:
            await self._spawn(progress_callback, log_callback, event_callback)
        except Exception:
            model_state_coordinator.end_activity(self._lifecycle_activity)
            self._lifecycle_activity_active = False
            raise

    async def _spawn(
        self,
        progress_callback: Optional[Callable[[int, float, float], None]],
        log_callback: Optional[Callable[[str], None]],
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Build the command and hand ownership to the log monitor.

        Split out of start() so a single try there covers every failure path
        between begin_activity and the monitor task taking over the release.
        """
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

        selected_gpu = resolve_cuda_visible_devices(
            env.get("CUDA_VISIBLE_DEVICES"), self.gpu_index
        )
        if selected_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = selected_gpu
            print(f"[Training] Run {self.run_id} pinned to GPU {selected_gpu} (CUDA_VISIBLE_DEVICES)")

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

        # Same reason, for on-demand sample requests: they carry no TTL (one
        # queued during a long caching phase must survive until the first
        # batch), so a request left pending by a stopped or crashed run would
        # otherwise be executed by the next one.
        try:
            from core.training.training_sample_rpc import clear_all as _clear_sample_rpc
            removed = _clear_sample_rpc(self.output_dir)
            if removed:
                print(f"[Training] Removed {removed} stale sample-request/result file(s) before spawn")
        except Exception as e:   # noqa: BLE001
            print(f"[Training] WARNING: Failed to clear stale sample requests before spawn: {e}")

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

        # Monitor logs in background. Keep the handle: see _monitor_task.
        self._monitor_task = asyncio.create_task(
            self._monitor_logs(progress_callback, log_callback, event_callback))

    def _forward_event(
        self,
        event: Dict[str, Any],
        event_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        """Dedup and cap a structured notice, then hand it to the backend.

        Identical notices are forwarded once: every emitter in the tree is a
        "say once" guard already, so a repeat means a loop, not new
        information. The cap bounds the distinct case and says so rather than
        going quiet.
        """
        if event_callback is None:
            return
        key = (event["level"], event.get("code"), event["message"])
        if key in self._events_seen:
            return
        if len(self._events_seen) >= self.MAX_EVENTS_PER_RUN:
            if not self._events_capped:
                self._events_capped = True
                event_callback({
                    "level": "warning",
                    "code": "training_event_cap_reached",
                    "message": (
                        f"This run emitted more than {self.MAX_EVENTS_PER_RUN} distinct "
                        f"notices; the rest are on the backend console only."
                    ),
                })
            return
        self._events_seen.add(key)
        try:
            event_callback(event)
        except Exception as e:
            print(f"[Training] Failed to forward training event: {e}")

    async def _monitor_logs(
        self,
        progress_callback: Optional[Callable[[int, float, float], None]],
        log_callback: Optional[Callable[[str], None]],
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Monitor training logs and extract progress information.

        Args:
            progress_callback: Callback for progress updates
            log_callback: Callback for log streaming
        """
        if not self.process or not self.process.stdout:
            if self._lifecycle_activity_active:
                from core.model_state_coordinator import model_state_coordinator
                model_state_coordinator.end_activity(self._lifecycle_activity)
                self._lifecycle_activity_active = False
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

                # A sentinel line is the machine half of a notice whose human
                # half already went through log_callback; it is lifted off the
                # stream here rather than printed as JSON to the console. It can
                # arrive with a carriage-return-only tqdm write glued in front
                # of it (see split_training_event), so what precedes it is still
                # forwarded to the console.
                line, event = split_training_event(line)
                if event is not None:
                    self._forward_event(event, event_callback)
                    if not line:
                        continue

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
            if self._lifecycle_activity_active:
                from core.model_state_coordinator import model_state_coordinator
                model_state_coordinator.end_activity(self._lifecycle_activity)
                self._lifecycle_activity_active = False
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
        gpu_index: Optional[int] = None,
    ) -> TrainingProcess:
        """
        Create and register a training process.

        Args:
            run_id: Training run ID
            config_path: Path to YAML config file
            output_dir: Output directory
            venv_python: Path to venv Python executable
            gpu_index: Physical GPU index to pin the child to (None: inherit)

        Returns:
            TrainingProcess instance

        Raises:
            RuntimeError: a live child already exists for ``run_id``. Overwriting
                the registry entry would orphan that process — nothing could stop
                it, and two trainers would share the GPU.
        """
        if self.is_live(run_id):
            raise RuntimeError(
                f"Training run {run_id} already has a live training process "
                f"(pid {getattr(self.processes[run_id].process, 'pid', '?')}); "
                f"stop it before starting the run again."
            )
        process = TrainingProcess(run_id, config_path, output_dir, venv_python, gpu_index)
        self.processes[run_id] = process
        return process

    def get_process(self, run_id: int) -> Optional[TrainingProcess]:
        """Get training process by run ID."""
        return self.processes.get(run_id)

    def is_live(self, run_id: int) -> bool:
        """Whether ``run_id`` is claimed by a training process that is not known
        to have finished.

        A REGISTERED-BUT-NOT-YET-SPAWNED entry (``process.process is None``)
        counts as live. create_process and the child spawn are seconds apart —
        pre-flight rescan, the pre-training VRAM release — and reading that
        window as "not live" let a second request reap the first request's
        registry entry and spawn its own child: two trainers on one GPU, the
        first orphaned with nothing left that could stop it. Chosen over an
        asyncio.Lock around register->spawn because create_process consults this
        same predicate, so one rule covers both entry points; the caller that
        registers is responsible for removing the entry if the spawn never
        happens (see start_training_run's failure path).

        Once spawned, ``is_running`` is a flag the monitor task clears only after
        it observes the exit, so the child's returncode is the ground truth.
        """
        process = self.processes.get(run_id)
        if process is None:
            return False
        if process.process is None:
            return True
        return process.process.returncode is None

    async def remove_process(self, run_id: int) -> None:
        """Remove training process from registry."""
        if run_id in self.processes:
            process = self.processes[run_id]
            if process.is_running:
                await process.stop()
            del self.processes[run_id]


# Global process manager
training_process_manager = TrainingProcessManager()
