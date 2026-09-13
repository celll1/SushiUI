"""CLI for creating the canonical complete YuE2 dense training checkpoint."""
from __future__ import annotations

import argparse

from .single_file import repack_official_dense


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official", help="Official m-a-p/YuE2-3B model.safetensors")
    parser.add_argument("assets", help="Complete SushiUI/Comfy YuE2 file providing VAE and tokenizer")
    parser.add_argument("output", help="Output .safetensors path")
    args = parser.parse_args()
    repack_official_dense(args.official, args.assets, args.output)


if __name__ == "__main__":
    main()
