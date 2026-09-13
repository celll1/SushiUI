"""Dry-run-first YuE2 text-to-audio API example.

The endpoint accepts JSON. A real run requires the YuE2 checkpoint to be loaded
and uses the GPU, so this script sends nothing unless ``--no-dry-run`` is set.
"""
import argparse
import json

import requests


BASE_URL = "http://localhost:8000/api/v1"


def build_request(prompt: str, lyrics: str, seed: int) -> dict:
    return {
        "prompt": prompt,
        "lyrics": lyrics,
        "seed": seed,
        "audio_duration": 30.0,
        "yue2_cot": "full",
        "yue2_abc_max_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 100,
        "repetition_penalty": 1.2,
        "vae_decode_mode": "tiled",
        "vae_tile_frames": 1024,
    }


def main():
    parser = argparse.ArgumentParser(description="YuE2 txt2aud example call")
    parser.add_argument("--prompt", default="Japanese city pop, warm female vocal, bright chorus")
    parser.add_argument("--lyrics", default="[Verse]\n夜の街を歩いて\n[Chorus]\n明日へ歌おう")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="POST the request and trigger GPU generation")
    parser.set_defaults(dry_run=True)
    args = parser.parse_args()

    url = f"{BASE_URL}/generate/txt2aud"
    payload = build_request(args.prompt, args.lyrics, args.seed)
    if args.dry_run:
        print("=== DRY RUN (no request sent) ===")
        print("Method:  POST")
        print(f"URL:     {url}")
        print("Headers: {'Content-Type': 'application/json'}")
        print("JSON body:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("\nLoad YuE2, then re-run with --no-dry-run when VRAM is available.")
        return

    response = requests.post(url, json=payload, timeout=1800)
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
    if result.get("success"):
        image = result.get("image", {})
        print(f"\nGenerated audio id={image.get('id')} path={image.get('filename')}")
        print(f"actual_seed={result.get('actual_seed')}")


if __name__ == "__main__":
    main()
