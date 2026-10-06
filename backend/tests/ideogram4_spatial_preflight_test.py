"""Public configuration gates for the native dual-DiT migration."""
import pytest


def test_public_full_method_and_swap_match_the_native_adapter():
    from api.arch_capabilities import TRAINING_UNSUPPORTED,training_feature_unsupported_reason,training_feature_advisories
    from core.training.vae_swap import preflight_vae_swap,initialization_mode
    assert "full_finetune" not in TRAINING_UNSUPPORTED.get("ideogram4",{})
    assert training_feature_unsupported_reason("ideogram4","vae_swap","full_finetune") is None
    assert training_feature_advisories("ideogram4","full_finetune")["vae_swap"]["level"]=="experimental"
    config={"vae_swap_source":"registry:qwen_image","ideogram4_train_uncond":True}
    assert initialization_mode(config,"ideogram4")=="paired_spatial"
    assert preflight_vae_swap(config,arch="ideogram4",method="full_finetune",bundle_vae_explicit_false=False)==config["vae_swap_source"]
    with pytest.raises(ValueError,match="full_finetune"):
        preflight_vae_swap(config,arch="ideogram4",method="lora",bundle_vae_explicit_false=False)
    with pytest.raises(ValueError,match="ideogram4_train_uncond"):
        preflight_vae_swap({**config,"ideogram4_train_uncond":False},arch="ideogram4",method="full_finetune",bundle_vae_explicit_false=False)
    with pytest.raises(ValueError,match="paired_spatial"):
        preflight_vae_swap({**config,"vae_swap_new_channel_init":"zero"},arch="ideogram4",method="full_finetune",bundle_vae_explicit_false=False)


def test_full_config_preserves_dual_branch_and_calibration_choices():
    from core.training.training_config import _build_train_section
    config={"training_method":"full_finetune","ideogram4_train_uncond":True,
        "ideogram4_uncond_loss_weight":.75,"ideogram4_lr_factor":.4,
        "vae_swap_source":"registry:qwen_image","vae_swap_new_channel_init":"paired_spatial",
        "vae_swap_calibration_samples":12,"vae_swap_calibration_resolution":128}
    train=_build_train_section(config,total_steps=None,epochs=5,train_unet=True,train_text_encoder=False,arch="ideogram4")
    for key,value in config.items():
        if key!="training_method":assert train[key]==value
