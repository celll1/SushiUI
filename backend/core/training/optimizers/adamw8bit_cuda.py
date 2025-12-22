"""
CUDA Extension Loader for AdamW 8-bit Optimizer

JIT-compiles CUDA kernels on first use.
Supports Windows, Linux, and various CUDA toolkit installations.
"""

import torch
import os
import sys
from pathlib import Path
from datetime import datetime
from torch.utils.cpp_extension import load


_extension = None


def get_extension():
    """
    Get (or compile) the CUDA extension for AdamW 8-bit optimizer.

    Returns:
        Module: Compiled CUDA extension with functions:
            - adamw_8bit_update()
            - init_quantization_maps()

    Raises:
        RuntimeError: If CUDA is not available or compilation fails
    """
    global _extension

    if _extension is not None:
        return _extension

    if not torch.cuda.is_available():
        raise RuntimeError("[AdamW8bit_CUDA] CUDA is not available")

    # Get source directory
    cuda_dir = Path(__file__).parent / "cuda"
    kernel_cu = cuda_dir / "adamw8bit_kernel.cu"
    wrapper_cpp = cuda_dir / "adamw8bit_cuda.cpp"

    if not kernel_cu.exists():
        raise RuntimeError(f"[AdamW8bit_CUDA] Kernel source not found: {kernel_cu}")
    if not wrapper_cpp.exists():
        raise RuntimeError(f"[AdamW8bit_CUDA] Wrapper source not found: {wrapper_cpp}")

    # Create build directory
    build_dir = cuda_dir / "build"
    build_dir.mkdir(exist_ok=True)

    # Setup compilation log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = build_dir / f"compilation_{timestamp}.log"

    print("[AdamW8bit_CUDA] Compiling CUDA extension (this may take a few minutes)...")
    print(f"  Kernel: {kernel_cu}")
    print(f"  Wrapper: {wrapper_cpp}")
    print(f"  Build dir: {build_dir}")
    print(f"  Log file: {log_file}")

    # Redirect stdout/stderr to log file while also printing to console
    class TeeLogger:
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
        # Redirect output
        sys.stdout = logger
        sys.stderr = logger

        # JIT compile
        _extension = load(
            name="adamw8bit_cuda_ext",
            sources=[str(wrapper_cpp), str(kernel_cu)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ],
            build_directory=str(build_dir),
            verbose=True,
        )

        # Restore output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.close()

        print("[AdamW8bit_CUDA] CUDA extension compiled successfully")
        print(f"[AdamW8bit_CUDA] Compilation log saved to: {log_file}")

    except Exception as e:
        # Restore output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.close()

        print(f"[AdamW8bit_CUDA] Compilation failed: {e}")
        print(f"[AdamW8bit_CUDA] Check compilation log for details: {log_file}")

        # Print last 50 lines of log for quick debugging
        print("\n" + "="*60)
        print("Last 50 lines of compilation log:")
        print("="*60)
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(line.rstrip())
        except Exception:
            pass
        print("="*60 + "\n")

        raise RuntimeError(f"Failed to compile CUDA extension: {e}")

    return _extension


if __name__ == "__main__":
    # Test compilation
    print("Testing CUDA extension compilation...")
    ext = get_extension()
    print(f"Extension loaded: {ext}")
    print(f"Available functions: {dir(ext)}")
    print("✓ CUDA extension test PASSED")
