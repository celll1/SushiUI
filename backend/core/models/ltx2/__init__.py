"""LTX-2.3 video generation foundation (Phase 0).

Unlike Ideogram4 / Krea2 / Lens / MiniT2I - which vendor their diffusers model
classes because those classes are absent from (or skewed against) the pinned
venv diffusers - the LTX-2 stack is **already fully present and importable in the
venv-resident diffusers 0.38.0**. All eight component classes load without
weights, and ``LTX2Pipeline.from_pretrained`` resolves every entry in the
LTX-2.3 ``model_index.json`` unaided:

  * ``vae``          -> diffusers.AutoencoderKLLTX2Video      (top-level export)
  * ``audio_vae``    -> diffusers.AutoencoderKLLTX2Audio      (top-level export)
  * ``transformer``  -> diffusers.LTX2VideoTransformer3DModel (top-level export)
  * ``connectors``   -> library ``ltx2`` -> diffusers.pipelines.ltx2.LTX2TextConnectors
  * ``vocoder``      -> library ``ltx2`` -> diffusers.pipelines.ltx2.LTX2VocoderWithBWE
  * ``text_encoder`` -> transformers.Gemma3ForConditionalGeneration
  * ``tokenizer``    -> transformers.GemmaTokenizerFast
  * ``scheduler``    -> diffusers.FlowMatchEulerDiscreteScheduler

``LTX2TextConnectors`` and ``LTX2VocoderWithBWE`` are *not* re-exported from the
top-level ``diffusers`` namespace, but the LTX-2.3 ``model_index.json`` tags them
with the library name ``"ltx2"``, which diffusers' component resolver routes to
the ``diffusers.pipelines.ltx2`` submodule (``is_pipeline_module``), where the
lazy ``_import_structure`` exposes them. So no namespace shim is required either.

Therefore this package deliberately does NOT vendor diffusers source: doing so
would duplicate weight-compatible code already shipping in the venv (the only
0.38->0.39 delta in the core stack is a functionally-equivalent refactor of
``LTX2TextConnectors``' register-replacement loop; vocoder / pipeline_output /
utils / latent_upsampler are byte-identical). It only provides a single canonical
import site for later phases.
"""

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    LTX2ConditionPipeline,
    LTX2ImageToVideoPipeline,
    LTX2LatentUpsamplePipeline,
    LTX2Pipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder, LTX2VocoderWithBWE

__all__ = [
    "AutoencoderKLLTX2Audio",
    "AutoencoderKLLTX2Video",
    "LTX2ConditionPipeline",
    "LTX2ImageToVideoPipeline",
    "LTX2LatentUpsamplePipeline",
    "LTX2Pipeline",
    "LTX2VideoTransformer3DModel",
    "LTX2TextConnectors",
    "LTX2Vocoder",
    "LTX2VocoderWithBWE",
]
