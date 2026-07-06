"""
ops/ — arch-scoped free-function math layer (plan A.1/A.4, pain point #4).

Rule: **handlers are thin orchestrators; the math lives here.** Functions in
this package take no ``self`` and no global state — they are pure free functions
mirroring the generation side's ``*_pipeline_ops``.

REUSE first: most arch math already lives in
``core/models/<arch>/<arch>_pipeline_ops.py`` and
``core/inference/custom_sampling.py`` (which the sampling family already reuses).
``arch/<arch>.py`` handlers import those directly. Training-only math with no
generation analogue lands in this package (populated in P6+).

P0: empty package — the shape is fixed; modules are added as bodies move.
"""
