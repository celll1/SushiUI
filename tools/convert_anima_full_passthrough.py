"""Convert a legacy Anima spatial-swap checkpoint to full passthrough on CPU.

file SOURCE OUTPUT       write a converted checkpoint for a new run (refuses overwrite)
run RUN_DIR              convert a stopped run's latest checkpoint in place so it can resume:
                         the original checkpoint/optimizer/state files are first moved to a
                         sibling backup directory, then the converted checkpoint and an
                         optimizer state without the reshaped tensors' moments are written
                         under the original names (those tensors restart their moments)

Loads whole files into host RAM (about their size). Nothing is written without --execute.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"backend"))


def latest_step_files(run_dir):
    pattern = re.compile(r"^(?P<run>.+)_step_(?P<step>\d+)\.safetensors$")
    found = [(int(m["step"]), m["run"], path) for path in run_dir.iterdir()
             if (m := pattern.match(path.name))]
    if not found:
        raise SystemExit(f"No single-file step checkpoint in {run_dir}")
    step, run, checkpoint = max(found)
    stem = f"{run}_step_{step:06d}"
    files = dict(checkpoint=checkpoint, optimizer=run_dir/f"{stem}_optimizer.pt", state=run_dir/f"{stem}_state.json")
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing or checkpoint.name != f"{stem}.safetensors":
        raise SystemExit(f"Latest step {step} is incomplete or unexpectedly named: {missing or checkpoint.name}")
    return step, files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    one = sub.add_parser("file")
    one.add_argument("source")
    one.add_argument("output")
    run = sub.add_parser("run")
    run.add_argument("run_dir")
    for item in (one, run):
        item.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    import torch
    # Import-only CUDA queries must not create a context for this CPU utility.
    torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
    torch.cuda._lazy_init = lambda *a, **k: None
    torch._C._cuda_init = lambda *a, **k: None
    from core.models.common.single_file_format import read_metadata, read_state_dict, save_single_file_state
    from core.models.components.spatial_swap import META_KEY, read_declaration
    from core.models.anima.passthrough_conversion import convert, strip_reshaped_moments

    if args.mode == "file":
        source, output = Path(args.source), Path(args.output)
        if output.exists() or Path(str(output)+".index.json").exists():
            raise SystemExit(f"Refusing to overwrite {output}")
    else:
        run_dir = Path(args.run_dir)
        step, files = latest_step_files(run_dir)
        source = output = files["checkpoint"]
        backup = run_dir.parent/f"{run_dir.name}_legacy_passthrough_backup_step_{step}"
        if backup.exists():
            raise SystemExit(f"Backup directory already exists: {backup}")
    declaration = read_declaration(read_metadata(str(source)))
    if declaration is None or declaration.get("passthrough") is not None:
        raise SystemExit(f"{source} is not a legacy index-passthrough spatial swap")
    report = dict(mode=args.mode, source=str(source), declaration=declaration, execute=args.execute)
    if args.mode == "run":
        report.update(step=step, backup=str(backup))
    print(json.dumps(report), flush=True)
    if not args.execute:
        return

    state, metadata = read_state_dict(str(source))
    before = {key: tuple(value.shape) for key, value in state.items()}
    state, metadata = convert(state, metadata)
    reshaped = {key: [before[key], tuple(value.shape)] for key, value in state.items()
                if before.get(key) != tuple(value.shape)}
    declared = json.loads(metadata[META_KEY])
    if args.mode == "file":
        saved = save_single_file_state(state, metadata, str(output))
        print(json.dumps(dict(saved=str(saved), reshaped=reshaped, declaration=declared)), flush=True)
        return
    # Stage everything in the backup directory, release the source mappings, then swap files.
    backup.mkdir()
    staged = Path(save_single_file_state(state, metadata, str(backup/"converted.safetensors")))
    if staged.suffix != ".safetensors" or not staged.is_file():
        raise SystemExit(f"Converted checkpoint was sharded ({staged}); in-place run conversion supports single files")
    del state
    payload = torch.load(files["optimizer"], map_location="cpu", weights_only=False)
    payload, dropped = strip_reshaped_moments(payload)
    torch.save(payload, backup/"converted_optimizer.pt")
    del payload
    import gc
    gc.collect()
    for path in files.values():
        shutil.move(str(path), str(backup/path.name))
    shutil.move(str(staged), str(files["checkpoint"]))
    shutil.move(str(backup/"converted_optimizer.pt"), str(files["optimizer"]))
    shutil.copy2(backup/files["state"].name, files["state"])
    print(json.dumps(dict(converted=str(files["checkpoint"]), dropped_moments=dropped, reshaped=reshaped,
                          backup=str(backup), declaration=declared)), flush=True)


if __name__ == "__main__":
    main()
