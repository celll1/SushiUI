"""Ming native model lifecycle; generation wiring is installed separately."""


class MingImage01Mixin:
    def _install_ming_image_01(self, result, *, model_id, source_type, source):
        from core.models.ming_image_01.inspection import loaded_model_info
        from core.models.components.wiring import MING_IMAGE_01_WIRING
        info = loaded_model_info(result["ming_native"])
        self.ming_image_01_components = result
        self.is_ming_image_01_model = True
        self.current_model = model_id
        self.current_attention_type = "normal"
        self._sushi_wiring = MING_IMAGE_01_WIRING.replace(te_out_dim=result["transformer"].config.cap_feat_dim)
        self.current_model_info = {**info, "source_type": source_type, "source": source,
                                   "is_video": False, "is_audio": False,
                                   "latent_channels": 16, "generation_available": False}

    def _ming_generation_pending(self):
        from api.error_handlers import ValidationError
        raise ValidationError("Ming native model is loaded; platform generation wiring is not yet enabled")
