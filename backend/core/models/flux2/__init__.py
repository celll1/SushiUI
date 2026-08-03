"""FLUX.2 model-side helpers (config pin, checkpoint-key transform).

The FLUX.2 loader itself still lives in ``core/model_loader.py``; this package
holds the pieces that BOTH the loader and the offline quantization tool need, so
neither can drift from the other.
"""
