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


def _ensure_msvc_env():
    """Windows: pre-activate the MSVC x64 build environment in os.environ before
    torch's cpp_extension runs.

    torch's ``_run_ninja_build`` only invokes its own vcvarsall detection when
    ``VSCMD_ARG_TGT_ARCH`` is NOT already in the environment, and setuptools'
    ``_get_vc_env`` honours ``DISTUTILS_USE_SDK``. On some setups torch's automatic
    invocation (``cmd /u /c "<vcvarsall>" x86_amd64 && set``) fails (spaces-in-path
    quoting / cross-arch, exit 255) even though vcvarsall itself works when called
    normally -- which aborts the whole training run when the cached extension is
    invalidated (e.g. after a torch upgrade). Running vcvarsall ourselves via a temp
    .bat (the form that works) and injecting its env makes torch skip its broken
    detection and build with the active compiler. No-op on non-Windows or when the
    MSVC env is already active (e.g. launched from a Developer Command Prompt)."""
    if os.name != "nt":
        return
    if os.environ.get("VSCMD_ARG_TGT_ARCH"):
        return
    import subprocess
    import tempfile

    vcvarsall = None
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    try:
        if vswhere.exists():
            res = subprocess.run(
                [str(vswhere), "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, errors="replace")
            for line in res.stdout.splitlines():
                cand = Path(line.strip()) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                if line.strip() and cand.exists():
                    vcvarsall = cand
                    break
    except Exception:
        pass
    if vcvarsall is None:
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        for base in (Path(pf) / "Microsoft Visual Studio" / "2022",
                     Path(pf86) / "Microsoft Visual Studio" / "2022",
                     Path(pf) / "Microsoft Visual Studio" / "2019",
                     Path(pf86) / "Microsoft Visual Studio" / "2019"):
            for ed in ("Community", "Professional", "Enterprise", "BuildTools"):
                cand = base / ed / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                if cand.exists():
                    vcvarsall = cand
                    break
            if vcvarsall is not None:
                break
    if vcvarsall is None:
        print("[AdamW8bit_CUDA] vcvarsall.bat not found; relying on torch's own MSVC detection")
        return

    bat_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False) as bat:
            bat.write(f'@echo off\r\ncall "{vcvarsall}" x64 >nul\r\nset\r\n')
            bat_path = bat.name
        res = subprocess.run(["cmd", "/c", bat_path], capture_output=True, text=True,
                             errors="replace")
        if res.returncode != 0:
            print(f"[AdamW8bit_CUDA] vcvarsall x64 failed (exit {res.returncode}); "
                  f"relying on torch's own MSVC detection")
            return
        injected = 0
        for line in res.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                if k and not k[0].isspace() and os.environ.get(k) != v:
                    os.environ[k] = v
                    injected += 1
        os.environ.setdefault("VSCMD_ARG_TGT_ARCH", "x64")
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")
        print(f"[AdamW8bit_CUDA] Activated MSVC x64 build environment from "
              f"{vcvarsall} ({injected} vars updated); torch will build with it")
    except Exception as e:
        print(f"[AdamW8bit_CUDA] Could not pre-activate MSVC env ({e}); "
              f"relying on torch's own MSVC detection")
    finally:
        if bat_path:
            try:
                os.unlink(bat_path)
            except Exception:
                pass


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
    schedulefree_kernel_cu = cuda_dir / "adamw8bit_schedulefree_kernel.cu"
    schedulefree_launcher_cu = cuda_dir / "adamw8bit_schedulefree_launcher.cu"
    wrapper_cpp = cuda_dir / "adamw8bit_cuda.cpp"

    if not kernel_cu.exists():
        raise RuntimeError(f"[AdamW8bit_CUDA] Kernel source not found: {kernel_cu}")
    if not schedulefree_kernel_cu.exists():
        raise RuntimeError(f"[AdamW8bit_CUDA] Schedule-Free kernel source not found: {schedulefree_kernel_cu}")
    if not schedulefree_launcher_cu.exists():
        raise RuntimeError(f"[AdamW8bit_CUDA] Schedule-Free launcher source not found: {schedulefree_launcher_cu}")
    if not wrapper_cpp.exists():
        raise RuntimeError(f"[AdamW8bit_CUDA] Wrapper source not found: {wrapper_cpp}")

    # Create build directory
    build_dir = cuda_dir / "build"
    build_dir.mkdir(exist_ok=True)

    # Setup compilation log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = build_dir / f"compilation_{timestamp}.log"

    # Windows: pre-activate the MSVC env so torch skips its own (fragile) vcvarsall
    # detection and builds with the working compiler environment.
    _ensure_msvc_env()

    print("[AdamW8bit_CUDA] Compiling CUDA extension (this may take a few minutes)...")
    print(f"  Kernel: {kernel_cu}")
    print(f"  Schedule-Free Kernel: {schedulefree_kernel_cu}")
    print(f"  Schedule-Free Launcher: {schedulefree_launcher_cu}")
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

        # JIT compile (include Schedule-Free kernels)
        _extension = load(
            name="adamw8bit_cuda_ext",
            sources=[
                str(wrapper_cpp),
                str(kernel_cu),
                # str(schedulefree_kernel_cu),  # Included by launcher
                str(schedulefree_launcher_cu)
            ],
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
    print("OK CUDA extension test PASSED")
