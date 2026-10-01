# Ming-Image package preflight

The artifact module imports neither Torch nor CUDA. It identifies the two
released native profiles from their configurations, validates component
geometry, and checks safetensors headers/shard maps without loading weights.

From the repository root, using the repository virtualenv:

```powershell
venv/Scripts/python.exe tools/inspect_ming_image_01.py <MODEL_ROOT>
venv/Scripts/python.exe tools/inspect_ming_image_01.py <MODEL_ROOT> --write-manifest
```

`--configs-only` allows an incomplete configuration snapshot and explicitly
reports `weights_verified=false`; it cannot create a manifest. A verified
manifest binds profile and configuration/processor/tokenizer identity to
relative component directories. Header validation verifies storage layout and
index completeness, not compatibility with every native model parameter;
strict native state loading remains the loader's responsibility.

`tensor_elements` includes serialized buffers and is not a trainable-parameter
count. Dtype/storage totals come from tensor headers. Config fingerprints do
not hash the multi-gigabyte tensor payload and must not be treated as a
cryptographic weight identity; component-content hashes belong to exports.

Native loading, selectable conditioner groups and small-model sampling are
implemented. Target-only nested U-Net execution, topology metadata restoration
and shared checkpoint inspection/DiT parameter census are CPU verified.
Platform generation/training registration and full component UI wiring are
subsequent implementation stages. Their capabilities remain
unavailable until the corresponding integration gates pass. See
[contract](../../../../docs/guides/MING_IMAGE_01_DESIGN.md).
