"""Inspect a Ming checkpoint package without importing Torch or loading weights."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from core.models.ming_image_01.artifact import inspect_artifact, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--configs-only", action="store_true", help="Do not verify weights; report remains unverified")
    parser.add_argument("--write-manifest", action="store_true", help="Atomically identify a complete native package")
    args = parser.parse_args()
    if args.write_manifest and args.configs_only:
        parser.error("Manifest creation requires verified weight headers")
    try:
        if args.write_manifest:
            write_manifest(args.path)
        result = inspect_artifact(args.path, require_weights=not args.configs_only)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Ming preflight failed: {exc}\n")
    print(json.dumps(result.summary(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
