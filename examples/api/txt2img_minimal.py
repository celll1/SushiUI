"""
Example: minimal txt2img generation call.

IMPORTANT — request shape surprise: POST /api/v1/generate/txt2img is NOT a
JSON endpoint. Despite the presence of a `GenerationParams` Pydantic model
elsewhere in routes.py (used only by the training-preview endpoints), the
real /generate/txt2img route signature is built entirely from FastAPI
`Form(...)` parameters (backend/api/routes.py line ~288). That means:
  - The request body must be sent as multipart/form-data (or
    application/x-www-form-urlencoded), NOT JSON.
  - `loras`, `controlnets`, and `tipo_config` are JSON-encoded *strings*
    passed as individual form fields (e.g. loras="[]"), not nested JSON.
  - `prompt` is the only required field; everything else has a server
    default matching backend/api/param_defaults.py.

This script defaults to --dry-run (prints the exact request it WOULD send:
method, url, headers, form fields) and only performs the real POST when you
pass --no-dry-run. Actual generation is a heavy, GPU-mutating, stateful
operation, so dry-run is the safe default.

On real success, the response shape is:
  {"success": true, "image": {<GeneratedImage.to_dict() fields>}, "actual_seed": <int>}
"""
import argparse
import json
import requests

BASE_URL = "http://localhost:8000/api/v1"


def build_form_data(prompt: str) -> dict:
    """Only prompt is set explicitly; every other field is omitted so the
    server applies its own Form(...) defaults (see routes.py generate_txt2img
    signature). This is the minimal valid request."""
    return {"prompt": prompt}


def main():
    parser = argparse.ArgumentParser(description="Minimal txt2img example call")
    parser.add_argument("--prompt", default="1girl, anime, beautiful, masterpiece",
                         help="Prompt to send (only required field)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                         help="Actually POST to the backend and trigger real generation")
    parser.set_defaults(dry_run=True)
    args = parser.parse_args()

    url = f"{BASE_URL}/generate/txt2img"
    form_data = build_form_data(args.prompt)

    if args.dry_run:
        print("=== DRY RUN (no request sent) ===")
        print(f"Method:  POST")
        print(f"URL:     {url}")
        print(f"Headers: {{'Content-Type': 'multipart/form-data'}} (set automatically by requests)")
        print(f"Form data (all other fields fall back to server defaults):")
        print(json.dumps(form_data, indent=2))
        print("\nRe-run with --no-dry-run to actually generate an image.")
        return

    print(f"POST {url} with prompt={args.prompt!r} ...")
    # requests sends dict via `data=` as multipart/form-data-compatible
    # url-encoded form when no `files=` is given -- FastAPI's Form() parses
    # either encoding identically, so this matches the server's expectations.
    #
    # This call blocks until the image is done. To observe progress from a
    # second script/thread while it runs, either connect to the WebSocket
    # (see backend/api/WS_PROTOCOL.md) or, for a frontend-less polling
    # alternative that doesn't need a persistent connection, poll
    # GET /api/v1/generation/status (see openapi.yaml, GenerationStatusResponse).
    resp = requests.post(url, data=form_data, timeout=600)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2)[:2000])

    if result.get("success"):
        image = result.get("image", {})
        print(f"\nGenerated image id={image.get('id')} path={image.get('filename')}")
        print(f"actual_seed={result.get('actual_seed')}")


if __name__ == "__main__":
    main()
