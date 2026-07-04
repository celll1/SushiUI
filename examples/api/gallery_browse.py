"""
Example: browse the gallery (read-only, safe to run for real).

Endpoints (backend/api/routes.py):
  GET /api/v1/images
      query params: skip (int, default 0), limit (int, default 50),
                     search, generation_types (comma-separated),
                     date_from, date_to (ISO), width_min/max, height_min/max
      response: {"images": [<GeneratedImage.to_dict()>...], "total": int,
                 "skip": int, "limit": int}

  GET /api/v1/images/{image_id}
      response: <GeneratedImage.to_dict()> (single object, no wrapper)

Note the pagination params are `skip`/`limit`, NOT `page`/`offset` -- verified
directly from the get_images() function signature (line ~1976).

This script makes no state-changing calls and is run for real by default.
"""
import json
import requests

BASE_URL = "http://localhost:8000/api/v1"


def main():
    resp = requests.get(f"{BASE_URL}/images", params={"skip": 0, "limit": 5}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    print("=== GET /images?skip=0&limit=5 ===")
    print(f"total={data.get('total')} skip={data.get('skip')} limit={data.get('limit')}")
    images = data.get("images", [])
    print(f"Returned {len(images)} image(s).")

    if not images:
        print("No images in gallery yet -- nothing to fetch detail for.")
        return

    first_id = images[0]["id"]
    print(f"\nFetching detail for first image id={first_id} ...")
    resp = requests.get(f"{BASE_URL}/images/{first_id}", timeout=10)
    resp.raise_for_status()
    print(f"=== GET /images/{first_id} ===")
    print(json.dumps(resp.json(), indent=2)[:2000])


if __name__ == "__main__":
    main()
