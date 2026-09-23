"""diffusers single-file CLIP loading against transformers with a flat CLIPTextModel.

transformers 5.17 removed the ``text_model`` level from ``CLIPTextModel`` (the
module is now ``embeddings``/``encoder``/``final_layer_norm`` directly), while
diffusers' ``create_diffusers_clip_model_from_ldm`` still reads
``model.text_model`` and emits ``text_model.*`` keys. diffusers silently skips
keys the model does not have, so fixing only the attribute access would leave the
text encoder on meta. The flat class is loaded through a nested wrapper instead,
so the converted keys match, and the inner model is returned.
"""

from transformers import CLIPTextConfig, CLIPTextModel
from transformers.models.clip.modeling_clip import CLIPPreTrainedModel


class _NestedCLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CLIPTextModel(config)


def _clip_text_model_is_flat() -> bool:
    return "text_model" not in CLIPTextModel.__init__.__code__.co_names


def install() -> None:
    if not _clip_text_model_is_flat():
        return

    import diffusers.loaders.single_file as single_file
    import diffusers.loaders.single_file_utils as single_file_utils

    original = single_file_utils.create_diffusers_clip_model_from_ldm
    if getattr(original, "_sushi_flat_clip_compat", False):
        return

    def create_diffusers_clip_model_from_ldm(cls, checkpoint, *args, **kwargs):
        if cls is not CLIPTextModel:
            return original(cls, checkpoint, *args, **kwargs)
        return original(_NestedCLIPTextModel, checkpoint, *args, **kwargs).text_model

    create_diffusers_clip_model_from_ldm._sushi_flat_clip_compat = True
    single_file_utils.create_diffusers_clip_model_from_ldm = create_diffusers_clip_model_from_ldm
    single_file.create_diffusers_clip_model_from_ldm = create_diffusers_clip_model_from_ldm
