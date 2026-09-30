"""CPU contracts for Lens residual execution and native block checkpoint modes."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import build_spec, OPERATORS, read_checkpoint_metadata


class LensRuntimeContracts(unittest.TestCase):
    def setUp(self):
        for target, name, value in ((torch.cuda, "get_device_capability", lambda *a, **k: (8, 9)),
                (torch.cuda, "_lazy_init", lambda *a, **k: None), (torch._C, "_cuda_init", lambda *a, **k: None)):
            guard = patch.object(target, name, value); guard.start(); self.addCleanup(guard.stop)
        from core.models.lens.vendor.transformer import LensTransformer2DModel
        from core.models.lens.residual_unet import native_config
        torch.manual_seed(64)
        self.model = LensTransformer2DModel(patch_size=2, in_channels=4, out_channels=1, num_layers=7,
            attention_head_dim=12, num_attention_heads=2, inner_dim=24, enc_hidden_dim=12,
            axes_dims_rope=(4,4,4), selected_layer_index=(5,11))
        self.spec = build_spec("lens", [{"name":"main", "block_count":7}], [
            dict(id="outer", stream="main", parent=None, start_block=1, end_block=6, down_factor=[2,2], **OPERATORS),
            dict(id="inner", stream="main", parent="outer", start_block=2, end_block=5, down_factor=[2,2], **OPERATORS)],
            native_config(self.model))
        self.images = torch.randn(2,96,4,requires_grad=True)
        self.features = [torch.randn(2,5,12,requires_grad=True) for _ in range(2)]
        self.mask = torch.tensor([[True,True,True,False,False], [True]*5])
        self.times = torch.tensor([300.0,700.0])

    def forward(self, **options):
        arguments = dict(hidden_states=self.images, encoder_hidden_states=self.features,
            encoder_hidden_states_mask=self.mask, timestep=self.times, img_shapes=[(1,8,12)])
        arguments.update(options)
        return self.model(**arguments)

    def test_native_lengths_text_rotary_masks_and_offloader(self):
        from core.models.lens.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        original = {name:id(value) for name,value in self.model.named_parameters()}
        full_rotary = self.model.pos_embed([(1,8,12)],[5],device=torch.device("cpu"))[1]
        text_mask = self.model._build_joint_attention_mask(self.mask,96)[...,-5:]
        lengths = []; events = []; previous_text = []
        def observe(block,args):
            image,text,modulation,rotary,mask = args
            lengths.append(image.shape[1]); self.assertEqual(text.shape[:2],(2,5))
            if previous_text:torch.testing.assert_close(text,previous_text[-1],rtol=0,atol=0)
            self.assertEqual(rotary[0].shape[0],image.shape[1])
            h,w = {96:(8,12),24:(4,6),6:(2,3)}[image.shape[1]]
            expected_rotary = self.model.pos_embed([(1,h,w)],[5],device=torch.device("cpu"))[0]
            torch.testing.assert_close(rotary[0],expected_rotary,rtol=0,atol=0)
            torch.testing.assert_close(rotary[1],full_rotary,rtol=0,atol=0)
            torch.testing.assert_close(mask[...,-5:],text_mask,rtol=0,atol=0)
            self.assertTrue(torch.equal(mask[...,:image.shape[1]],torch.zeros_like(mask[...,:image.shape[1]])))
        handles = [block.register_forward_pre_hook(observe) for block in self.model.transformer_blocks]
        handles.extend(block.register_forward_hook(lambda block,args,output:previous_text.append(output[0]))
                       for block in self.model.transformer_blocks)
        class Offloader:
            def wait_for_block(owner,index):events.append(("wait",index))
            def submit_move_blocks_forward(owner,index):events.append(("submit",index))
        self.model._block_offloader = Offloader()
        try: output = self.forward()
        finally:
            for handle in handles:handle.remove()
        self.assertEqual(lengths,[96,24,6,6,6,24,96])
        self.assertEqual(events,[(name,i) for i in range(7) for name in ("wait","submit")])
        self.assertEqual(output.shape,self.images.shape);self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(original,{name:id(value) for name,value in self.model.named_parameters()})
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_writer_native_loader_single_shard_and_preflight(self):
        from core.models.lens.residual_unet import restore_topology,native_config,preflight_checkpoint
        from core.training.adapters.lens_adapter import LensFullParameterAdapter,LensLoRAAdapter
        from core.models.common.single_file_format import read_state_dict,save_single_file_state
        from core.model_loader import ModelLoader
        from safetensors.torch import save_file
        restore_topology(self.model,self.spec)
        expected=self.forward().detach()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/"base";(base/"transformer").mkdir(parents=True)
            (base/"vae").mkdir();(base/"vae"/"config.json").write_text("{}")
            config=native_config(self.model)
            companion=dict(config,num_layers=3,selected_layer_index=[1])
            (base/"transformer"/"config.json").write_text(json.dumps(dict(companion,_class_name="LensTransformer2DModel")))
            trainer=SimpleNamespace(transformer=self.model,vae=None,text_encoder=None,config={},bundle_vae=False,lens_base_dir=str(base))
            single=LensFullParameterAdapter(trainer).write_checkpoint(1,0,root/"full.safetensors")
            lora_metadata=LensLoRAAdapter(trainer,2,2).checkpoint_metadata({},1,0)
            self.assertEqual(read_checkpoint_metadata(lora_metadata),self.spec)
            self.assertEqual(preflight_checkpoint(single),(self.spec,config))
            state,metadata=read_state_dict(str(single))
            index=save_single_file_state(state,metadata,str(root/"sharded.safetensors"),max_shard_bytes=20000)
            text_encoder=torch.nn.Linear(1,1)
            text_encoder.set_selected_layers=lambda layers:setattr(text_encoder,"selected",layers)
            tokenizer=SimpleNamespace(pad_token_id=0,padding_side="left",encode=lambda *a,**k:[1])
            vae=torch.nn.Linear(1,1)
            with ExitStack() as stack:
                for target,value in (
                    ("core.models.lens.vendor.LensGptOssEncoder.from_pretrained",text_encoder),
                    ("diffusers.AutoencoderKLFlux2.from_pretrained",vae),
                    ("transformers.AutoTokenizer.from_pretrained",tokenizer),
                    ("diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",object())):
                    stack.enter_context(patch(target,return_value=value))
                stack.enter_context(patch.object(type(self.model),"from_pretrained",side_effect=AssertionError("base DiT allocation")))
                for path in (single,Path(index),root/"sharded.safetensors"):
                    result=ModelLoader.load_lens_from_path(str(path),torch_dtype=torch.float32)
                    restored=result["transformer"]
                    self.assertEqual(restored._residual_unet_spec,self.spec)
                    self.assertEqual(native_config(restored),config)
                    self.assertEqual(text_encoder.selected,(5,11))
                    for key,value in self.model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
                    with patch.object(self,"model",restored):
                        torch.testing.assert_close(self.forward(),expected,rtol=1e-5,atol=2e-7)
            for name,broken in (("missing-config",{k:v for k,v in metadata.items() if k!="transformer_config"}),
                    ("wrong-config",dict(metadata,transformer_config=json.dumps(companion)))):
                path=root/f"{name}.safetensors";save_file(state,str(path),metadata=broken)
                with patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("tensor loading")), \
                     patch("core.models.lens.lens_loader._resolve_lens_base_dir",side_effect=AssertionError("base resolution")):
                    with self.assertRaises(ValueError):ModelLoader.load_lens_from_path(str(path),torch_dtype=torch.float32)
            partial=root/"partial.safetensors"
            save_file({key:value for key,value in state.items() if key!="net.img_in.weight"},str(partial),metadata=metadata)
            with patch("core.models.lens.lens_loader.load_lens_components",side_effect=AssertionError("companion loading")):
                with self.assertRaisesRegex(ValueError,"weights do not match"):
                    ModelLoader.load_lens_from_path(str(partial),torch_dtype=torch.float32)
            dense=root/"dense.safetensors";save_file({"placeholder":torch.ones(1)},str(dense))
            self.assertEqual(preflight_checkpoint(dense),(None,None))
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_checkpoint_modes_preserve_output_and_gradients(self):
        from core.models.lens.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        parameters = [self.images,*self.features,*self.model.parameters()]
        outputs = []; gradients = []
        for mode in ("none","standard","cpu_offload","async_cpu_offload"):
            self.model.disable_gradient_checkpointing()
            if mode != "none":
                self.model.enable_gradient_checkpointing(cpu_offload=mode=="cpu_offload",async_offload=mode=="async_cpu_offload")
            output = self.forward(); outputs.append(output.detach())
            gradients.append(torch.autograd.grad(output.square().mean(),parameters))
        for output,grads in zip(outputs,gradients):
            torch.testing.assert_close(output,outputs[0],rtol=0,atol=0)
            for actual,expected in zip(grads,gradients[0]):
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-7)
        self.assertTrue(any(torch.count_nonzero(value) for value in gradients[0][1:3]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_single_text_features_dense_restore_metadata_and_refusals(self):
        from core.models.lens.residual_unet import restore_topology,model_topology_metadata,native_config
        dense = self.forward().detach()
        restore_topology(self.model,self.spec)
        self.assertEqual(read_checkpoint_metadata(model_topology_metadata(self.model)),self.spec)
        for shapes in ([(2,8,6)],[(1,8,12),(1,12,8)],[(1,8,11)],[(True,8,12)]):
            with self.assertRaises(ValueError):self.forward(img_shapes=shapes)
        for options in (dict(encoder_hidden_states=self.features[:1]),dict(encoder_hidden_states_mask=self.mask[:,:4]),
                        dict(attention_kwargs={"scale":0.5})):
            with self.assertRaises(ValueError):self.forward(**options)
        for name in ("_style_ctx","_fbcache","_repa_tap_depth","_tread_config","_blockskip_config"):
            setattr(self.model,name,object())
            try:
                with self.assertRaises(ValueError):self.forward()
            finally:delattr(self.model,name)
        for name,value in (("_nag_enabled",True),("_negpip_enabled",True),("_style_ctx",object())):
            setattr(self.model.transformer_blocks[0].attn,name,value)
            try:
                with self.assertRaises(ValueError):self.forward()
            finally:delattr(self.model.transformer_blocks[0].attn,name)
        with patch.object(self.model.pos_embed,"scale_rope",False):
            with self.assertRaisesRegex(ValueError,"centered RoPE"):self.forward()
        for shapes in ([(1,8,12)],(1,8,12),[[(1,8,12)]],[(1,8,12)]*2):
            self.assertTrue(torch.isfinite(self.forward(img_shapes=shapes)).all())
        restore_topology(self.model,None)
        torch.testing.assert_close(self.forward(),dense,rtol=0,atol=0)
        self.assertEqual(model_topology_metadata(self.model),{})
        with self.assertRaises(ValueError):native_config(dict(self.model.config,axes_dims_rope=[2,2,2]))
        with self.assertRaises(ValueError):native_config(dict(self.model.config,patch_size=True))
        from core.models.lens.vendor.transformer import LensTransformer2DModel
        model = LensTransformer2DModel(**dict(native_config(self.model),multi_layer_encoder_feature=False))
        restore_topology(model,build_spec("lens",self.spec["streams"],self.spec["regions"],native_config(model)))
        with patch.object(self,"model",model):
            self.assertTrue(torch.isfinite(self.forward(encoder_hidden_states=self.features[0])).all())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":unittest.main()
