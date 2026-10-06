"""Expand official scaled FP8 H3 weights on CPU; refuses output overwrite."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--official-dir")
    parser.add_argument("--work-mib",type=int,default=64)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    import torch
    import os
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    # Import-only CUDA queries must not create a context for this CPU utility.
    torch.cuda.get_device_capability = lambda *a,**k:(8,9)
    torch.cuda._lazy_init = lambda *a,**k:None
    torch._C._cuda_init = lambda *a,**k:None
    from core.models.minimax_h3.full_conversion import plan_conversion,convert_checkpoint
    from core.models.common.streaming_safetensors import tensor_bytes
    _,specs,converted,_ = plan_conversion(args.source,official_dir=args.official_dir)
    print(json.dumps(dict(source=args.source,output=args.output,matrices=len(converted),
        output_gib=sum(tensor_bytes(s) for s in specs.values())/2**30,execute=args.execute)),flush=True)
    if not args.execute:
        return
    import psutil
    process = psutil.Process()
    start, last = time.monotonic(), [0.,0.]
    def progress(done,total):
        last[1] = max(last[1],process.memory_info().rss/2**30)
        now = time.monotonic()
        if now-last[0] >= 5 or done == total:
            print(json.dumps(dict(percent=round(100*done/total,2),elapsed_s=round(now-start,1),
                peak_observed_rss_gib=round(last[1],3))),flush=True)
            last[0] = now
    written = convert_checkpoint(args.source,args.output,work_bytes=args.work_mib*1024**2,
        official_dir=args.official_dir,progress=progress)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(written=str(written),elapsed_s=round(time.monotonic()-start,2),
        peak_observed_rss_gib=round(last[1],3),cuda_initialized=False)),flush=True)


if __name__ == "__main__":
    main()
