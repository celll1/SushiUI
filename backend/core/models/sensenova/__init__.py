"""SenseNova-U1.5-8B-MoT (Qwen3-8B-as-flow-matching-denoiser) support for SushiUI.

Apache-2.0 (both code and weights). Model classes vendored from
OpenSenseNova/SenseNova-U1's ``feat/u1.5`` branch (no upstream single-file
distribution exists) -- see ``vendor/__init__.py`` for provenance and the
exact modifications. This package's own ``loader.py`` reads a sushiUI shard
index produced by this repo's own int8 conversion (Unit 1); there is no
directory-completion or sibling-junction layout to support here.
"""

from .loader import is_sensenova_state_dict_keys, load_sensenova_from_path

__all__ = ["is_sensenova_state_dict_keys", "load_sensenova_from_path"]
