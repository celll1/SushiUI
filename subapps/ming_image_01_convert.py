"""Convert a native Ming package to DiT INT8 or ConvRot without loading its MLLM."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from core.models.ming_image_01.conversion import convert_package


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("destination",type=Path,help="New native package directory outside source")
    parser.add_argument("--format",choices=("int8","int8_convrot"),default="int8_convrot")
    parser.add_argument("--device",default="cpu",help="CPU by default; CUDA must be explicitly selected")
    parser.add_argument("--max-shard-mib",type=int,help="Output shard threshold; shared default is 4096 MiB")
    args=parser.parse_args()
    try:
        result=convert_package(args.source,args.destination,format_=args.format,device=args.device,
                               max_shard_bytes=None if args.max_shard_mib is None else args.max_shard_mib*1024**2)
    except (OSError,ValueError,RuntimeError) as exc:
        parser.exit(2,f"Ming conversion failed: {exc}\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return result


if __name__=="__main__":
    main()
