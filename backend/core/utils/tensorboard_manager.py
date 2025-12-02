"""
TensorBoard Server Manager

Manages TensorBoard server instances for training runs.
"""

import subprocess
import psutil
from pathlib import Path
from typing import Optional, Dict


class TensorBoardManager:
    """Manages TensorBoard server processes."""

    def __init__(self):
        self.processes: Dict[int, subprocess.Popen] = {}  # run_id -> process
        self.ports: Dict[int, int] = {}  # run_id -> port

    def start(self, run_id: int, log_dir: str, port: int = 6006) -> int:
        """
        Start TensorBoard server for a training run.

        Args:
            run_id: Training run ID
            log_dir: Path to tensorboard log directory
            port: Port to run tensorboard on (default: 6006)

        Returns:
            Port number tensorboard is running on

        Raises:
            ValueError: If tensorboard is already running for this run
        """
        if run_id in self.processes:
            if self.is_running(run_id):
                raise ValueError(f"TensorBoard already running for run {run_id}")
            else:
                # Process died, clean up
                self.stop(run_id)

        # Find available port if the requested port is in use
        while port in self.ports.values():
            port += 1

        # Start tensorboard process
        command = [
            "tensorboard",
            "--logdir", log_dir,
            "--port", str(port),
            "--host", "0.0.0.0",  # Allow external access
            "--reload_interval", "30",  # Reload every 30 seconds
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.processes[run_id] = process
        self.ports[run_id] = port

        print(f"[TensorBoard] Started for run {run_id} on port {port}")
        return port

    def stop(self, run_id: int):
        """Stop TensorBoard server for a training run."""
        if run_id not in self.processes:
            return

        process = self.processes[run_id]

        try:
            # Terminate process and its children
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()

            # Wait for process to exit
            parent.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            # Force kill if still running
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass

        del self.processes[run_id]
        if run_id in self.ports:
            del self.ports[run_id]

        print(f"[TensorBoard] Stopped for run {run_id}")

    def is_running(self, run_id: int) -> bool:
        """Check if TensorBoard is running for a training run."""
        if run_id not in self.processes:
            return False

        process = self.processes[run_id]

        try:
            # Check if process is still alive
            return psutil.Process(process.pid).is_running()
        except psutil.NoSuchProcess:
            return False

    def get_port(self, run_id: int) -> Optional[int]:
        """Get the port TensorBoard is running on for a training run."""
        if run_id in self.ports and self.is_running(run_id):
            return self.ports[run_id]
        return None

    def get_url(self, run_id: int) -> Optional[str]:
        """Get the TensorBoard URL for a training run."""
        port = self.get_port(run_id)
        if port:
            return f"http://localhost:{port}"
        return None

    def stop_all(self):
        """Stop all TensorBoard servers."""
        run_ids = list(self.processes.keys())
        for run_id in run_ids:
            self.stop(run_id)


# Global instance
tensorboard_manager = TensorBoardManager()
