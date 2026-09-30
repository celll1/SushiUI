"""CPU contracts for standalone ACE-Step full artifacts and frozen sources."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file


class AceStepSourceContracts(unittest.TestCase):
    def setUp(self):
        for obj,name,value in ((torch.cuda,"get_device_capability",lambda *a,**k:(8,9)),
                (torch.cuda,"_lazy_init",lambda *a,**k:None),(torch._C,"_cuda_init",lambda *a,**k:None)):
            guard=patch.object(obj,name,value);guard.start();self.addCleanup(guard.stop)
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)/"original"
        for name in ("diffusion_models","vae","text_encoders","tokenizer"):
            (self.root/name).mkdir(parents=True)
        self.vae=self.root/"vae"/"ace_1.5_vae.safetensors"
        self.te=self.root/"text_encoders"/"qwen_0.6b_ace15.safetensors"
        for path in (self.vae,self.te): save_file({"marker":torch.ones(1)},str(path))
        self.tokenizer=self.root/"tokenizer"
        self.sources=dict(schema_version=1,root=str(self.root),vae=str(self.vae),
                          text_encoder=str(self.te),tokenizer=str(self.tokenizer))
        from core.models.acestep.vendor import AceStepConfig, AceStepConditionGenerationModel
        from core.models.acestep.residual_unet import native_config, restore_topology
        from core.rewiring.spec import build_spec, OPERATORS
        config=AceStepConfig(hidden_size=16,intermediate_size=24,num_hidden_layers=7,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,patch_size=2,
            in_channels=12,audio_acoustic_hidden_dim=4,encoder_hidden_size=16,
            sliding_window=3,num_lyric_encoder_hidden_layers=1,num_timbre_encoder_hidden_layers=1,
            num_attention_pooler_hidden_layers=1,num_audio_decoder_hidden_layers=1,
            text_hidden_dim=8,timbre_hidden_dim=4,fsq_dim=16,fsq_input_levels=[2,2],pool_window_size=2)
        config._attn_implementation="sdpa"
        self.owner=AceStepConditionGenerationModel(config).eval()
        self.spec=build_spec("acestep",[{"name":"main","block_count":7}],[
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,1],**OPERATORS),
            dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,1],**OPERATORS)],
            native_config(config))
        restore_topology(self.owner,self.spec)
        self.owner._acestep_component_sources=self.sources
        self.path=Path(folder.name)/"outputs"/"full.safetensors"
        from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
        AceStepFullParameterAdapter(SimpleNamespace(transformer=self.owner)).write_checkpoint(3,1,self.path)

    def test_standalone_loader_uses_exact_sources_and_reexports_them(self):
        from core.models.acestep.loader import detect_acestep_layout, load_acestep_from_path
        from core.model_loader import ModelLoader
        from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
        self.assertEqual(ModelLoader.detect_model_type(str(self.path)),"acestep")
        layout=detect_acestep_layout(str(self.path))
        self.assertEqual(layout["vae"],str(self.vae));self.assertEqual(layout["text_encoder"],str(self.te))
        self.assertEqual(layout["dit"],str(self.path));self.assertEqual(layout["tokenizer_source"],str(self.tokenizer))
        vae=torch.nn.Linear(1,1);te=torch.nn.Linear(1,1);tokenizer=object()
        with patch("core.models.acestep.loader._build_vae",return_value=vae) as build_vae, \
                patch("core.models.acestep.loader._build_text_encoder",return_value=(te,{})) as build_te, \
                patch("transformers.AutoTokenizer.from_pretrained",return_value=tokenizer) as build_tokenizer, \
                patch.object(torch.cuda,"is_available",return_value=False):
            loaded=load_acestep_from_path(str(self.path),torch.float32)
        build_vae.assert_called_once_with(str(self.vae),torch.float32)
        build_te.assert_called_once_with(str(self.te),torch.float32)
        build_tokenizer.assert_called_once_with(str(self.tokenizer))
        self.assertIs(loaded["tokenizer"],tokenizer)
        self.assertEqual(loaded["dit"]._acestep_component_sources,self.sources)
        self.assertEqual(loaded["dit"]._residual_unet_spec,self.spec)
        for name,value in self.owner.state_dict().items():
            torch.testing.assert_close(value,loaded["dit"].state_dict()[name],rtol=0,atol=0)
        second=self.path.with_name("next.safetensors")
        AceStepFullParameterAdapter(SimpleNamespace(transformer=loaded["dit"])).write_checkpoint(4,1,second)
        with safe_open(str(second),framework="pt") as handle:
            self.assertEqual(json.loads(handle.metadata()["acestep_components"]),self.sources)
        self.assertFalse(torch.cuda.is_initialized())

    def test_declared_sources_take_priority_and_legacy_tree_still_detects(self):
        from core.models.acestep.loader import detect_acestep_layout
        from core.model_loader import ModelLoader
        legacy=self.root/"diffusion_models"/"acestep_v1.5_turbo.safetensors"
        save_file(self.owner.state_dict(),str(legacy))
        layout=detect_acestep_layout(str(legacy))
        self.assertEqual(layout["vae"],str(self.vae))
        self.assertEqual(layout["text_encoder"],str(self.te))
        self.assertEqual(ModelLoader.detect_model_type(str(self.root)),"acestep")
        self.assertEqual(detect_acestep_layout(str(self.root)),layout)
        with safe_open(str(self.path),framework="pt") as handle: metadata=handle.metadata()
        decoy=self.path.parent/"diffusion_models"
        decoy.mkdir()
        copied=decoy/"full.safetensors"
        save_file(load_file(str(self.path)),str(copied),metadata=metadata)
        layout=detect_acestep_layout(str(copied))
        self.assertEqual(layout["root"],str(self.root))
        self.assertEqual(layout["vae"],str(self.vae))
        self.assertEqual(detect_acestep_layout(str(self.path.parent)),layout)

    def test_declared_sources_refuse_before_model_allocation(self):
        from core.models.acestep.loader import load_acestep_from_path
        with safe_open(str(self.path),framework="pt") as handle: metadata=handle.metadata()
        weights=load_file(str(self.path))
        invalid=[{**self.sources,"schema_version":2},{**self.sources,"vae":str(self.path.parent/"missing")},
                 {**self.sources,"text_encoder":str(self.root)},{**self.sources,"root":"relative"},
                 {**self.sources,"tokenizer":str(self.root/"missing")},{**self.sources,"extra":1}]
        for sources in invalid:
            with self.subTest(sources=sources):
                save_file(weights,str(self.path),metadata={**metadata,"acestep_components":json.dumps(sources)})
                with patch("core.models.acestep.loader._build_dit",side_effect=AssertionError("No allocation")), \
                        self.assertRaises(ValueError):
                    load_acestep_from_path(str(self.path),torch.float32)
        self.assertFalse(torch.cuda.is_initialized())

    def test_legacy_and_new_grid_policies_roundtrip_in_full_config(self):
        from core.models.acestep.residual_unet import native_config, restore_topology, generation_metadata, GRID_POLICY_KEY, GRID_POLICY
        from core.models.acestep.loader import _build_dit
        from core.rewiring.spec import build_spec
        from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
        legacy_config=native_config(self.owner.config);legacy_config.pop(GRID_POLICY_KEY)
        legacy=build_spec("acestep",self.spec["streams"],self.spec["regions"],legacy_config)
        restore_topology(self.owner,legacy)
        AceStepFullParameterAdapter(SimpleNamespace(transformer=self.owner)).write_checkpoint(4,1,self.path)
        with safe_open(str(self.path),framework="pt") as handle: metadata=handle.metadata()
        raw=json.loads(metadata["acestep_config"])
        self.assertEqual(raw[GRID_POLICY_KEY],"require_divisible")
        loaded,config=_build_dit(str(self.path),torch.float32)
        self.assertEqual(loaded._residual_unet_spec,legacy)
        self.assertEqual(native_config(native_config(config)),native_config(config))
        with self.assertRaisesRegex(ValueError,"not divisible"): generation_metadata(loaded,45)
        raw[GRID_POLICY_KEY]=GRID_POLICY
        save_file(load_file(str(self.path)),str(self.path),metadata={**metadata,"acestep_config":json.dumps(raw)})
        with patch("core.models.acestep.vendor.AceStepConditionGenerationModel",side_effect=AssertionError("No allocation")), \
                self.assertRaisesRegex(ValueError,"explicit grid policy"):
            _build_dit(str(self.path),torch.float32)
        restore_topology(self.owner,self.spec)
        AceStepFullParameterAdapter(SimpleNamespace(transformer=self.owner)).write_checkpoint(5,1,self.path)
        loaded,config=_build_dit(str(self.path),torch.float32)
        self.assertEqual(generation_metadata(loaded,250)["transport_padding_tokens"],3)
        self.assertEqual(getattr(config,GRID_POLICY_KEY),GRID_POLICY)
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_resume_dispatches_to_native_component_loader(self):
        from core.training.base_trainer import BaseTrainer
        trainer=SimpleNamespace(model_path="unused",log_prefix="[CPU]")
        paths=[]
        with patch("core.training.ops.acestep_ops.load_components",side_effect=lambda owner:paths.append(owner.model_path)):
            BaseTrainer._load_checkpoint_as_base(trainer,str(self.path))
        self.assertEqual(paths,[str(self.path)])
        self.assertTrue(trainer.is_acestep)
        self.assertEqual(trainer.model_path,str(self.path))
        self.assertFalse(torch.cuda.is_initialized())
