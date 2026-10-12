"""Convert a legacy Anima spatial-swap checkpoint to full passthrough on CPU; refuses overwrite.

Loads the whole checkpoint into host RAM (about its file size). Without --execute it only
reports what would change.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.output+".index.json").exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    import torch
    # Import-only CUDA queries must not create a context for this CPU utility.
    torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
    torch.cuda._lazy_init = lambda *a, **k: None
    torch._C._cuda_init = lambda *a, **k: None
    from core.models.common.single_file_format import read_metadata, read_state_dict, save_single_file_state
    from core.models.components.spatial_swap import META_KEY
    from core.models.anima.passthrough_conversion import convert
    metadata = read_metadata(args.source)
    print(json.dumps(dict(source=args.source, output=args.output, declaration=json.loads(metadata[META_KEY]),
                          execute=args.execute)), flush=True)
    if not args.execute:
        return
    state, metadata = read_state_dict(args.source)
    before = {key: tuple(value.shape) for key, value in state.items()}
    state, metadata = convert(state, metadata)
    changed = {key: [before[key], tuple(value.shape)] for key, value in state.items()
               if before.get(key) != tuple(value.shape)}
    path = save_single_file_state(state, metadata, args.output)
    print(json.dumps(dict(saved=str(path), reshaped=changed,
                          declaration=json.loads(metadata[META_KEY]))), flush=True)


if __name__ == "__main__":
    main()
