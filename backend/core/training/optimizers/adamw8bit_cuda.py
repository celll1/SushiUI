"""
CUDA Extension Loader for AdamW 8-bit Optimizer

JIT-compiles CUDA kernels on first use.
Supports Windows, Linux, and various CUDA toolkit installations.
"""

import torch
import os
from pathlib import Path
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

    print("[AdamW8bit_CUDA] Compiling CUDA extension (this may take a few minutes)...")
    print(f"  Kernel: {kernel_cu}")
    print(f"  Wrapper: {wrapper_cpp}")
    print(f"  Build dir: {build_dir}")

    try:
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

        print("[AdamW8bit_CUDA] CUDA extension compiled successfully")

    except Exception as e:
        print(f"[AdamW8bit_CUDA] Compilation failed: {e}")
        raise RuntimeError(f"Failed to compile CUDA extension: {e}")

    return _extension


if __name__ == "__main__":
    # Test compilation
    print("Testing CUDA extension compilation...")
    ext = get_extension()
    print(f"Extension loaded: {ext}")
    print(f"Available functions: {dir(ext)}")
    print("✓ CUDA extension test PASSED")
