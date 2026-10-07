"""Expand one Ideogram FP8 component to its BF16 effective weights on CPU."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--work-mib", type=int, default=64)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    import torch
    torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
    torch.cuda._lazy_init = lambda *a, **k: None
    torch._C._cuda_init = lambda *a, **k: None
    torch.cuda.is_available = lambda: False
    from core.models.ideogram4.full_conversion import plan_conversion, convert_checkpoint
    from core.models.common.streaming_safetensors import tensor_bytes
    torch.set_num_threads(8)
    _, specs, converted, _ = plan_conversion(args.source)
    print(json.dumps(dict(matrices=len(converted), output_gib=sum(tensor_bytes(s) for s in specs.values())/2**30,
                          execute=args.execute)), flush=True)
    if not args.execute:
        return
    import psutil
    process = psutil.Process()
    started, last, peak = time.monotonic(), [0.0], [0.0]

    def progress(done, total):
        peak[0] = max(peak[0], process.memory_info().rss/2**30)
        now = time.monotonic()
        if now-last[0] >= 5 or done == total:
            print(json.dumps(dict(percent=100*done/total, elapsed_s=now-started,
                                  observed_peak_rss_gib=peak[0])), flush=True)
            last[0] = now

    convert_checkpoint(args.source, args.output, work_bytes=args.work_mib*1024**2, progress=progress)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(elapsed_s=time.monotonic()-started, observed_peak_rss_gib=peak[0],
                          cuda_initialized=False)), flush=True)


if __name__ == "__main__":
    main()
