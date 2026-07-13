"""Vendored ACE-Step 1.5 (turbo) model code (Apache-2.0, ACE Studio / StepFun).

Source: HuggingFace transformers custom-code modeling files shipped alongside
the ``acestep-v15-turbo`` checkpoint (``auto_map`` -> ``AutoModel`` /
``AutoConfig``). Vendored verbatim (only the relative-import fallback in the
original file is exercised) so SushiUI does not depend on ``trust_remote_code``
network fetches at runtime.
"""

from .configuration_acestep_v15 import AceStepConfig
from .modeling_acestep_v15_turbo import (
    AceStepAudioTokenizer,
    AceStepConditionEncoder,
    AceStepConditionGenerationModel,
    AceStepDiTModel,
    AceStepLyricEncoder,
    AceStepPreTrainedModel,
    AceStepTimbreEncoder,
    AttentionPooler,
    AudioTokenDetokenizer,
)

__all__ = [
    "AceStepConfig",
    "AceStepAudioTokenizer",
    "AceStepConditionEncoder",
    "AceStepConditionGenerationModel",
    "AceStepDiTModel",
    "AceStepLyricEncoder",
    "AceStepPreTrainedModel",
    "AceStepTimbreEncoder",
    "AttentionPooler",
    "AudioTokenDetokenizer",
]
