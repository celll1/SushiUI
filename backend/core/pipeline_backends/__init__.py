"""Per-architecture backend mixins for DiffusionPipelineManager.

Each mixin holds the generation/LoRA/helper methods for one model family, extracted
verbatim from the former monolithic pipeline.py. They are composed onto the single
DiffusionPipelineManager instance, so `self.*` cross-references resolve via the MRO.
"""

from core.pipeline_backends.zimage import ZImageMixin
from core.pipeline_backends.flux2 import Flux2Mixin
from core.pipeline_backends.anima import AnimaMixin
from core.pipeline_backends.lens import LensMixin
from core.pipeline_backends.ideogram4 import Ideogram4Mixin
from core.pipeline_backends.minit2i import MiniT2IMixin
from core.pipeline_backends.krea2 import Krea2Mixin
from core.pipeline_backends.ltx2 import LTX2Mixin
from core.pipeline_backends.acestep import AceStepMixin
from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin
from core.pipeline_backends.minimax_music3 import MiniMaxMusic3Mixin
from core.pipeline_backends.sensenova import SenseNovaMixin

__all__ = [
    "ZImageMixin",
    "Flux2Mixin",
    "AnimaMixin",
    "LensMixin",
    "Ideogram4Mixin",
    "MiniT2IMixin",
    "Krea2Mixin",
    "LTX2Mixin",
    "AceStepMixin",
    "MiniMaxH3Mixin",
    "MiniMaxMusic3Mixin",
    "SenseNovaMixin",
]
