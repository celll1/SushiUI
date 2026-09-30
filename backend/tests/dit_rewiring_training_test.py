"""Training request/YAML parity and new-run versus resume topology contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
     patch.object(torch.cuda,"_lazy_init"),patch.object(torch._C,"_cuda_init"):
    from api.routes import TrainingRunCreateRequest
    from core.training.training_config import _build_train_section
    import core.training.train_runner
    import core.training.lora_trainer
    from core.training.ops import qwen_image_21_ops
    from core.models.qwen_image_21.vendor.transformer import QwenImage21Transformer2DModel
    from core.models.qwen_image_21.residual_unet import configure_training_topology
from core.rewiring.training import validate_training_rewiring
from core.rewiring.spec import preview_plan


def model():
    return QwenImage21Transformer2DModel(num_layers=32,num_attention_heads=1,
        attention_head_dim=12,axes_dims_rope=(4,4,4),context_in_dim=12,mlp_ratio=1)


class TrainingContracts(unittest.TestCase):
    def test_request_and_yaml(self):
        request = TrainingRunCreateRequest(training_method="lora",base_model_path="test",
                                           total_steps=1,dit_rewiring_mode="configure")
        section = _build_train_section(request.model_dump(),total_steps=1,epochs=None,
            train_unet=True,train_text_encoder=False,arch="qwen_image_21")
        self.assertEqual(section["dit_rewiring_mode"],"configure")
        self.assertIsNone(section["dit_rewiring_spec"])
        with self.assertRaises(ValueError):
            TrainingRunCreateRequest(training_method="lora",base_model_path="test",
                                     dit_rewiring_spec={"bad":True})
        with self.assertRaises(ValueError):
            TrainingRunCreateRequest(training_method="vae_decoder",base_model_path="test",
                                     dit_rewiring_mode="configure")
        with self.assertRaises(ValueError):
            TrainingRunCreateRequest(training_method="lora",base_model_path="test",
                                     dit_rewiring_mode="configure",dit_partition_training_enabled=True)
        with self.assertRaisesRegex(ValueError,"not yet supported"):
            validate_training_rewiring(section,"sdxl")

    def test_default_resolution_and_inherit(self):
        m = model()
        dense = {}
        self.assertIsNone(configure_training_topology(m,dense))
        config = {"dit_rewiring_mode":"configure"}
        spec = configure_training_topology(m,config)
        self.assertEqual(spec["regions"][1]["start_block"],14)
        self.assertEqual(spec["regions"][1]["end_block"],18)
        plan = preview_plan(spec,64,64)
        self.assertEqual(sum(b["tokens"]==256 for b in plan["blocks"]),4)
        inherited = {"dit_rewiring_mode":"inherit"}
        self.assertEqual(configure_training_topology(m,inherited),spec)
        self.assertEqual(inherited["dit_rewiring_spec"],spec)
        self.assertEqual(configure_training_topology(m,config,resuming=True),spec)

    def test_resume_refusals_and_pending_lora(self):
        m = model()
        config = {"dit_rewiring_mode":"configure"}
        spec = configure_training_topology(m,config)
        changed = deepcopy(spec); changed["regions"][1]["end_block"]=19
        with self.assertRaisesRegex(ValueError,"without edits"):
            configure_training_topology(m,{"dit_rewiring_mode":"configure","dit_rewiring_spec":changed},resuming=True)
        with self.assertRaisesRegex(ValueError,"base topology disagree"):
            configure_training_topology(m,config,resume_spec=None)
        fresh = model()
        resume_config = {"dit_rewiring_mode":"inherit","dit_rewiring_spec":spec}
        configure_training_topology(fresh,resume_config,pending_adapter_resume=True)
        self.assertEqual(configure_training_topology(fresh,resume_config,resume_spec=spec),spec)
        dense_base = model()
        untouched = {}
        self.assertEqual(configure_training_topology(dense_base,untouched,resume_spec=spec,install=False),spec)
        self.assertEqual(untouched,{})
        self.assertFalse(hasattr(dense_base,"_residual_unet_spec"))
        self.assertEqual(configure_training_topology(dense_base,{},resume_spec=spec),spec)

    def test_invalid_resume_keeps_native_topology(self):
        from safetensors.torch import save_file
        from core.training.adapters.qwen_image_21_adapter import QwenImage21LoRAAdapter
        from core.training.lora_trainer import LoRATrainer
        from core.rewiring.spec import checkpoint_metadata
        configured = model()
        required = configure_training_topology(configured,{"dit_rewiring_mode":"configure"})
        native = model()
        trainer = SimpleNamespace(transformer=native,config={},is_qwen_image_21=True,
            log_prefix="[InvalidResumeTest]",lora_rank=2,lora_alpha=2,
            adapter_algorithm="lora",weight_decompose=False,adapter_config={})
        adapter = QwenImage21LoRAAdapter(trainer,2,2)
        layers = {}; adapter.apply_lora_to_unet(layers); trainer.lora_layers=layers
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/"invalid.safetensors"
            save_file({"unrelated":torch.ones(1)},str(checkpoint),
                      metadata={"model_type":"qwen_image_21",**checkpoint_metadata(required)})
            with self.assertRaisesRegex(ValueError,"missing required tensor"):
                LoRATrainer.load_checkpoint(trainer,str(checkpoint))
        self.assertFalse(hasattr(native,"_residual_unet_spec"))
        self.assertEqual(trainer.config,{})

    def test_incompatible_options_do_not_mutate_model(self):
        m = model()
        for key in ("dit_partition_training_enabled","tread_enable","repa_enable","blockskip_enable"):
            with self.assertRaisesRegex(ValueError,"incompatible"):
                configure_training_topology(m,{"dit_rewiring_mode":"configure",key:True})
            self.assertFalse(hasattr(m,"_residual_unet_spec"))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
