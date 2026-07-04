"""
Example: health check + generation-defaults schema.

Demonstrates the two cheapest read-only GETs against the backend:
  - GET /api/v1/health                       -> {"status": "ok", "version": ...}
  - GET /api/v1/schema/generation-defaults    -> {"txt2img": {...}, "img2img": {...}, "inpaint": {...}}

All backend routes are mounted under the "/api/v1" prefix -- see
backend/main.py: app.include_router(router, prefix="/api/v1").

Safe to run for real; makes no state-changing calls.
"""
import json
import requests

BASE_URL = "http://localhost:8000/api/v1"


def main():
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    resp.raise_for_status()
    print("=== GET /health ===")
    print(json.dumps(resp.json(), indent=2))

    resp = requests.get(f"{BASE_URL}/schema/generation-defaults", timeout=10)
    resp.raise_for_status()
    print("\n=== GET /schema/generation-defaults ===")
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
