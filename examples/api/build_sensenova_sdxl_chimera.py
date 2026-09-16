"""Build a SenseNova SDXL Chimera artifact through the production API.

The script is dry-run by default because a real request reads multi-gigabyte
sources and writes a full U-Net, bridge, and bundled VAE under the model root.
"""

import argparse
import json

import requests


BASE_URL = "http://localhost:8000/api/v1"


def main():
    parser = argparse.ArgumentParser(description="Build a SenseNova SDXL Chimera artifact")
    parser.add_argument("--understanding-source", required=True)
    parser.add_argument("--sdxl-source", required=True)
    parser.add_argument("--output-name", default="snu1.5_sdxl_chimera")
    parser.add_argument("--target-dir")
    parser.add_argument(
        "--unet-initialization", choices=("scratch", "sdxl_transplant"), default="scratch"
    )
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    args = parser.parse_args()

    url = f"{BASE_URL}/models/sensenova-sdxl-chimera/initialize"
    payload = {
        "output_name": args.output_name,
        "target_dir": args.target_dir,
        "understanding_source": args.understanding_source,
        "sdxl_source": args.sdxl_source,
        "unet_initialization": args.unet_initialization,
        "initialization_seed": args.initialization_seed,
        "context_tokens": 77,
    }
    if args.dry_run:
        print("=== DRY RUN (no request sent) ===")
        print(f"POST {url}")
        print(json.dumps(payload, indent=2))
        print("\nRe-run with --no-dry-run to build the artifact.")
        return

    response = requests.post(url, json=payload, timeout=None)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
