"""
Lion 8-bit CUDA Extension Loader

JIT compilation of Lion 8-bit CUDA kernel with compilation logging.
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

_extension = None

def get_extension():
    """
    Get or compile Lion 8-bit CUDA extension (lazy loading).

    Returns:
        Compiled CUDA extension module
    """
    global _extension

    if _extension is not None:
        return _extension

    from torch.utils.cpp_extension import load

    # Kernel source files
    current_dir = Path(__file__).parent
    cuda_dir = current_dir / "cuda"
    kernel_cu = cuda_dir / "lion8bit_kernel.cu"
    schedulefree_kernel_cu = cuda_dir / "lion8bit_schedulefree_kernel.cu"
    schedulefree_launcher_cu = cuda_dir / "lion8bit_schedulefree_launcher.cu"
    wrapper_cpp = cuda_dir / "lion8bit_cuda.cpp"

    if not kernel_cu.exists():
        raise FileNotFoundError(f"CUDA kernel not found: {kernel_cu}")
    if not schedulefree_kernel_cu.exists():
        raise FileNotFoundError(f"Schedule-Free kernel not found: {schedulefree_kernel_cu}")
    if not schedulefree_launcher_cu.exists():
        raise FileNotFoundError(f"Schedule-Free launcher not found: {schedulefree_launcher_cu}")
    if not wrapper_cpp.exists():
        raise FileNotFoundError(f"C++ wrapper not found: {wrapper_cpp}")

    # Build directory
    build_dir = current_dir / "build" / "lion8bit"
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Lion8bit_CUDA] Compiling CUDA extension...")
    print(f"[Lion8bit_CUDA] Kernel: {kernel_cu}")
    print(f"[Lion8bit_CUDA] Schedule-Free Kernel: {schedulefree_kernel_cu}")
    print(f"[Lion8bit_CUDA] Schedule-Free Launcher: {schedulefree_launcher_cu}")
    print(f"[Lion8bit_CUDA] Wrapper: {wrapper_cpp}")
    print(f"[Lion8bit_CUDA] Build directory: {build_dir}")

    # Compilation flags
    extra_cuda_cflags = [
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ]

    extra_cflags = [
        "-std=c++17",
        "-O3",
    ]

    # Setup compilation log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = build_dir / f"compilation_{timestamp}.log"

    class TeeLogger:
        """Logger that writes to both stdout and file."""
        def __init__(self, file_path):
            self.file = open(file_path, 'w', encoding='utf-8')
            self.stdout = sys.stdout
            self.stderr = sys.stderr

        def write(self, message):
            self.stdout.write(message)
            self.file.write(message)
            self.file.flush()

        def flush(self):
            self.stdout.flush()
            self.file.flush()

        def close(self):
            self.file.close()

    logger = TeeLogger(log_file)
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Redirect stdout/stderr to logger
        sys.stdout = logger
        sys.stderr = logger

        # JIT compile CUDA extension
        _extension = load(
            name="lion8bit_cuda",
            sources=[
                str(wrapper_cpp),
                str(kernel_cu),
                # str(schedulefree_kernel_cu),  # Included by launcher
                str(schedulefree_launcher_cu)
            ],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=str(build_dir),
            verbose=True,
        )

        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.close()

        print(f"[Lion8bit_CUDA] Compilation successful!")
        print(f"[Lion8bit_CUDA] Compilation log: {log_file}")

        return _extension

    except Exception as e:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.close()

        print(f"[Lion8bit_CUDA] Compilation failed: {e}")
        print(f"[Lion8bit_CUDA] Check compilation log: {log_file}")

        # Print last 50 lines of log
        print(f"\n[Lion8bit_CUDA] Last 50 lines of compilation log:")
        print("=" * 80)
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line.rstrip())
        print("=" * 80)

        raise RuntimeError(f"Failed to compile Lion8bit CUDA extension: {e}")
