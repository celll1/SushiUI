"""Chimera text-output routing contracts."""

from types import SimpleNamespace


def test_img2txt_dispatches_to_chimera_understanding_path():
    from core.pipeline import DiffusionPipelineManager

    seen = {}

    def run(params, image, progress_callback=None):
        seen.update(params=params, image=image, callback=progress_callback)
        return "caption", 7, {"generation_seconds": 0.1}

    manager = SimpleNamespace(
        is_sensenova_sdxl_chimera_model=True,
        is_sensenova_model=False,
        _generate_img2txt_sensenova_sdxl_chimera=run,
    )
    callback = object()
    result = DiffusionPipelineManager.generate_img2txt(
        manager, {"instruction": "describe"}, "image", progress_callback=callback
    )
    assert result[0] == "caption"
    assert seen == {
        "params": {"instruction": "describe"},
        "image": "image",
        "callback": callback,
    }
