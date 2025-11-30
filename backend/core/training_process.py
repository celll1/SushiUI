"""
Training process management for ai-toolkit integration.

Handles subprocess execution, log monitoring, and progress tracking.
"""

import asyncio
import subprocess
import re
import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime


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
            venv_python: Path to venv Python executable (defaults to d:\\celll1\\webui_cl\\venv\\Scripts\\python.exe)
        """
        self.run_id = run_id
        self.config_path = config_path
        self.output_dir = output_dir
        self.venv_python = venv_python or r"d:\celll1\webui_cl\venv\Scripts\python.exe"

        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
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
        backend_dir = Path(__file__).parent.parent
        train_runner_path = backend_dir / "core" / "train_runner.py"

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

        # Start asyncio subprocess (non-blocking)
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(backend_dir),
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
                line_bytes = await self.process.stdout.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode('utf-8').strip()

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

            # Check if process failed
            if returncode != 0:
                print(f"[Training] Process exited with code {returncode}")
                # Mark as failed in database via callback
                if progress_callback:
                    # Signal failure (negative step indicates error)
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
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
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
