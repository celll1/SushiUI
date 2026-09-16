"""Deterministic image metrics for the SenseNova latent-refiner acceptance run.

The probe is deliberately model-free. It pairs PNG/JPEG files by relative path
and reports the pre-registered grid-period and frequency-band diagnostics from
SENSENOVA_LATENT_REFINER_DESIGN.md. Run it once per checkpoint/sample set; keep
the emitted per-image rows for paired sign tests rather than comparing only the
aggregate means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from PIL import Image


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _images(root: Path) -> Dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    }


def _luma(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _boundary_gradient(image: np.ndarray, period: int) -> float:
    """Mean first-difference magnitude on interior period boundaries."""
    vertical = np.abs(np.diff(image, axis=1))
    horizontal = np.abs(np.diff(image, axis=0))
    x = np.arange(1, image.shape[1]) % period == 0
    y = np.arange(1, image.shape[0]) % period == 0
    values = []
    if x.any():
        values.append(vertical[:, x].reshape(-1))
    if y.any():
        values.append(horizontal[y, :].reshape(-1))
    return float(np.concatenate(values).mean()) if values else float("nan")


def _grid_score(image: np.ndarray, period: int, control: int) -> float:
    boundary = _boundary_gradient(image, period)
    baseline = _boundary_gradient(image, control)
    return boundary / max(baseline, 1e-12)


def _band_energy(image: np.ndarray, wavelength_min: float,
                 wavelength_max: float) -> float:
    centered = image.astype(np.float64) - float(image.mean())
    spectrum = np.fft.rfft2(centered)
    power = spectrum.real.square() + spectrum.imag.square()
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.rfftfreq(image.shape[1])[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    low = 1.0 / wavelength_max
    high = 1.0 / wavelength_min
    mask = (frequency >= low) & (frequency <= high)
    return float(power[mask].mean()) if mask.any() else float("nan")


def image_metrics(image: np.ndarray) -> Dict[str, float]:
    return {
        "grid64": _grid_score(image, 64, 61),
        "grid8": _grid_score(image, 8, 11),
        "band_4_16": _band_energy(image, 4.0, 16.0),
    }


def _paired_rows(candidate: Path, reference: Path) -> Iterable[dict]:
    candidate_files = _images(candidate)
    reference_files = _images(reference)
    names = sorted(set(candidate_files) & set(reference_files))
    if not names:
        raise ValueError("candidate and reference directories have no matched images")
    for name in names:
        candidate_image = _luma(candidate_files[name])
        reference_image = _luma(reference_files[name])
        if candidate_image.shape != reference_image.shape:
            raise ValueError(
                f"paired image {name!r} differs in shape: "
                f"{candidate_image.shape} vs {reference_image.shape}"
            )
        candidate_metrics = image_metrics(candidate_image)
        reference_metrics = image_metrics(reference_image)
        residual = candidate_image - reference_image
        yield {
            "path": name,
            "width": int(candidate_image.shape[1]),
            "height": int(candidate_image.shape[0]),
            "candidate": candidate_metrics,
            "reference": reference_metrics,
            "delta": {
                key: candidate_metrics[key] - reference_metrics[key]
                for key in candidate_metrics
            },
            "residual_band_8_16": _band_energy(residual, 8.0, 16.0),
        }


def compare(candidate: Path, reference: Path) -> dict:
    rows = list(_paired_rows(candidate, reference))
    keys = tuple(rows[0]["delta"])
    aggregate = {}
    for key in keys:
        values = np.asarray([row["delta"][key] for row in rows], dtype=np.float64)
        aggregate[key] = {
            "median_delta": float(np.median(values)),
            "mean_delta": float(values.mean()),
            "candidate_better": int((values < 0).sum()) if key.startswith("grid")
            else int((values > 0).sum()),
            "pairs": int(values.size),
        }
    residual = np.asarray(
        [row["residual_band_8_16"] for row in rows], dtype=np.float64
    )
    return {
        "candidate": str(candidate.resolve()),
        "reference": str(reference.resolve()),
        "aggregate": aggregate,
        "residual_band_8_16_median": float(np.median(residual)),
        "images": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.candidate, args.reference)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
