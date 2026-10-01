"""Run Ming CPU checks with CUDA disabled before pytest collects any module."""
import os
from pathlib import Path
import sys
import torch


def forbidden_cuda(*args, **kwargs):
    raise AssertionError("Ming CPU verification attempted CUDA initialization")


def main():
    torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
    torch.cuda.is_available = lambda: False
    torch.cuda._lazy_init = forbidden_cuda
    torch._C._cuda_init = forbidden_cuda
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root / "backend"))
    import pytest
    arguments = sys.argv[1:]
    selected = any(Path(arg.split("::", 1)[0]).is_file() for arg in arguments if not arg.startswith("-"))
    targets = [] if selected else sorted(str(path.relative_to(root))
        for path in (root / "backend/tests").glob("ming_image_01*test.py"))
    result = pytest.main([*targets, *arguments])
    assert not torch.cuda.is_initialized()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
