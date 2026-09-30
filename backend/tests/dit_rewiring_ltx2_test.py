"""CPU contracts for frame-local residual execution of native LTX-2 AV blocks."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import OPERATORS, build_spec


class Ltx2RuntimeContracts(unittest.TestCase):
    def setUp(self):
        for obj,name,value in ((torch.cuda,"get_device_capability",lambda *a,**k:(8,9)),
                (torch.cuda,"_lazy_init",lambda *a,**k:None),(torch._C,"_cuda_init",lambda *a,**k:None)):
            guard=patch.object(obj,name,value);guard.start();self.addCleanup(guard.stop)
        from diffusers import LTX2VideoTransformer3DModel
        from core.models.ltx2.residual_unet import native_config, wrap_transformer
        torch.manual_seed(103)
        self.inner=LTX2VideoTransformer3DModel(in_channels=8,out_channels=8,
            num_attention_heads=2,attention_head_dim=16,cross_attention_dim=32,
            audio_in_channels=4,audio_out_channels=4,audio_num_attention_heads=2,
            audio_attention_head_dim=8,audio_cross_attention_dim=16,num_layers=7,
            use_prompt_embeddings=False,cross_attn_mod=True,audio_cross_attn_mod=True,
            gated_attn=True,audio_gated_attn=True)
        self.spec=build_spec("ltx2",[{"name":"main","block_count":7}],[
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
            dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)],
            native_config(self.inner.config))
        self.model=wrap_transformer(self.inner,self.spec)
        self.video=torch.randn(2,3*8*12,8,requires_grad=True)
        self.audio=torch.randn(2,5,4,requires_grad=True)
        self.encoder=torch.randn(2,4,32,requires_grad=True)
        self.audio_encoder=torch.randn(2,4,16,requires_grad=True)

    def forward(self,model=None,**overrides):
        args=dict(hidden_states=self.video,audio_hidden_states=self.audio,
            encoder_hidden_states=self.encoder,audio_encoder_hidden_states=self.audio_encoder,
            timestep=torch.tensor([500.,700.]),audio_timestep=torch.tensor([450.,650.]),
            sigma=torch.tensor([.5,.7]),audio_sigma=torch.tensor([.45,.65]),
            encoder_attention_mask=torch.tensor([[1,1,1,1],[1,1,0,0]]),
            audio_encoder_attention_mask=torch.tensor([[1,1,1,1],[1,1,1,0]]),
            num_frames=3,height=8,width=12,fps=24.,audio_num_frames=5,return_dict=False)
        args.update(overrides)
        return (model or self.model)(**args)

    def test_native_joint_layout_coordinates_and_gradients(self):
        observed=[]
        hooks=[layer.register_forward_pre_hook(lambda module,args,kwargs:observed.append(
            (kwargs["hidden_states"].shape[1],kwargs["audio_hidden_states"].shape[1],
             kwargs["video_rotary_emb"][0].shape[1])),with_kwargs=True) for layer in self.inner.transformer_blocks]
        coords=[]
        hook=self.inner.rope.register_forward_pre_hook(lambda module,args:coords.append(args[0].detach().clone()))
        try:video,audio=self.forward()
        finally:
            for handle in [*hooks,hook]:handle.remove()
        self.assertEqual(observed,[(n,5,n) for n in [288,72,18,18,18,72,288]])
        self.assertEqual(video.shape,self.video.shape);self.assertEqual(audio.shape,self.audio.shape)
        for value in (video,audio):self.assertTrue(torch.isfinite(value).all())
        (video.square().mean()+audio.square().mean()).backward()
        for value in (self.video,self.audio,self.encoder,self.audio_encoder):
            self.assertTrue(torch.isfinite(value.grad).all());self.assertGreater(torch.count_nonzero(value.grad).item(),0)
        self.assertEqual(len(coords),3)
        native_time=coords[0][:,0].reshape(2,3,96,2)[:,:,0]
        for coarse in coords[1:]:
            time=coarse[:,0].reshape(2,3,-1,2)
            torch.testing.assert_close(time,native_time[:,:,None].expand_as(time),rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_checkpointed_outputs_and_all_parameter_gradients(self):
        before=self.forward()
        sum(value.square().mean() for value in before).backward()
        expected={name:value.grad.detach().clone() for name,value in self.inner.named_parameters() if value.grad is not None}
        input_grads=[value.grad.clone() for value in [self.video,self.audio,self.encoder,self.audio_encoder]]
        self.inner.zero_grad(set_to_none=True)
        for value in [self.video,self.audio,self.encoder,self.audio_encoder]:value.grad=None
        self.inner.enable_gradient_checkpointing()
        after=self.forward()
        for left,right in zip(before,after):torch.testing.assert_close(left,right,rtol=0,atol=0)
        sum(value.square().mean() for value in after).backward()
        for name,value in self.inner.named_parameters():
            if name in expected:torch.testing.assert_close(value.grad,expected[name],rtol=0,atol=0)
        for value,expected_grad in zip([self.video,self.audio,self.encoder,self.audio_encoder],input_grads):
            torch.testing.assert_close(value.grad,expected_grad,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_split_rope_and_asymmetric_spatial_regions(self):
        from diffusers import LTX2VideoTransformer3DModel
        from core.models.ltx2.residual_unet import native_config,wrap_transformer
        raw=dict(self.inner.config);raw["rope_type"]="split";raw.pop("_use_default_values",None)
        native=LTX2VideoTransformer3DModel.from_config(raw)
        self.assertEqual(native.config.rope_type,"split")
        native.load_state_dict(self.inner.state_dict(),strict=True)
        regions=deepcopy(self.spec["regions"])
        regions[0]["down_factor"]=[2,1];regions[1]["down_factor"]=[1,2]
        spec=build_spec("ltx2",self.spec["streams"],regions,native_config(native.config))
        model=wrap_transformer(native,spec)
        counts=[]
        hooks=[layer.register_forward_pre_hook(lambda module,args,kwargs:counts.append(
            (kwargs["hidden_states"].shape[1],kwargs["video_rotary_emb"][0].shape[2],
             kwargs["audio_hidden_states"].shape[1])),with_kwargs=True) for layer in native.transformer_blocks]
        try:video,audio=self.forward(model=model)
        finally:
            for handle in hooks:handle.remove()
        self.assertEqual(counts,[(n,n,5) for n in [288,144,72,72,72,144,288]])
        self.assertTrue(torch.isfinite(video).all());self.assertTrue(torch.isfinite(audio).all())
        (video.square().mean()+audio.square().mean()).backward()
        self.assertTrue(torch.isfinite(native.proj_in.weight.grad).all())
        self.assertTrue(torch.isfinite(native.audio_proj_in.weight.grad).all())
        import json,tempfile
        from safetensors.torch import save_file
        from core.models.ltx2.residual_unet import export_metadata
        from core.models.ltx2.loader import load_ltx2_transformer_checkpoint
        metadata=export_metadata(model)
        declaration=json.loads(metadata["transformer_config"])
        declaration["_use_default_values"]=["rope_type"]
        metadata["transformer_config"]=json.dumps(declaration)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"split.safetensors"
            save_file({"net."+key:value for key,value in native.state_dict().items()},str(path),metadata=metadata)
            restored=load_ltx2_transformer_checkpoint(str(path),torch.float32)
            self.assertEqual(restored.config.rope_type,"split")
            self.assertEqual(restored._residual_unet_spec,spec)
        self.assertFalse(torch.cuda.is_initialized())

    def test_dense_fast_path_and_identity_native_heads(self):
        from core.models.ltx2.residual_unet import restore_topology
        restore_topology(self.model,None)
        direct=self.forward(model=self.inner);wrapped=self.forward()
        for left,right in zip(direct,wrapped):torch.testing.assert_close(left,right,rtol=0,atol=0)
        restore_topology(self.model,self.spec)
        handles=[patch.object(layer,"forward",side_effect=lambda **kwargs:(kwargs["hidden_states"],kwargs["audio_hidden_states"]))
                 for layer in self.inner.transformer_blocks]
        for handle in handles:handle.start()
        try:
            residual=self.forward()
            restore_topology(self.model,None)
            dense=self.forward(model=self.inner)
            for left,right in zip(residual,dense):torch.testing.assert_close(left,right,rtol=0,atol=0)
        finally:
            for handle in handles:handle.stop()
        self.assertFalse(torch.cuda.is_initialized())

    def test_per_token_timestep_mixed_precision_and_offloader_hooks(self):
        from types import SimpleNamespace
        moves=[]
        self.model._block_offloader=SimpleNamespace(blocks_to_swap=2,
            wait_for_block=lambda i:moves.append(("wait",i)),
            submit_move_blocks_forward=lambda i:moves.append(("submit",i)))
        time=torch.linspace(0.,1000.,288).expand(2,-1)
        video,audio=self.forward(timestep=time)
        self.assertEqual(moves,[(action,i) for i in range(7) for action in ("wait","submit")])
        self.assertTrue(torch.isfinite(video).all());self.assertTrue(torch.isfinite(audio).all())
        self.model._block_offloader=None
        for dtype in (torch.float16,torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.inner.to(dtype)
                video,audio=self.forward(hidden_states=self.video.to(dtype),audio_hidden_states=self.audio.to(dtype),
                    encoder_hidden_states=self.encoder.to(dtype),audio_encoder_hidden_states=self.audio_encoder.to(dtype))
                self.assertEqual(video.dtype,dtype);self.assertEqual(audio.dtype,dtype)
                self.assertTrue(torch.isfinite(video).all());self.assertTrue(torch.isfinite(audio).all())
                sum(value.float().square().mean() for value in (video,audio)).backward()
                self.assertTrue(torch.isfinite(self.inner.proj_in.weight.grad).all())
                self.inner.zero_grad(set_to_none=True)
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_single_and_sharded_transformer_roundtrip_and_lora_metadata(self):
        import tempfile
        from types import SimpleNamespace
        from core.models.ltx2.loader import load_ltx2_transformer_checkpoint
        from core.models.common.single_file_format import save_single_file_state
        from core.training.adapters.ltx2_adapter import Ltx2FullParameterAdapter, Ltx2LoRAAdapter
        from core.rewiring.spec import read_checkpoint_metadata
        trainer=SimpleNamespace(transformer=self.model,config={})
        lo=Ltx2LoRAAdapter(trainer,lora_rank=2,lora_alpha=2)
        self.assertEqual(read_checkpoint_metadata(lo.checkpoint_metadata({},3,1)),self.spec)
        self.model.eval().requires_grad_(False)
        original=self.forward()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for sharded in (False,True):
                path=root/("sharded.safetensors" if sharded else "single.safetensors")
                def write(state,metadata,target):
                    return save_single_file_state(state,metadata,target,max_shard_bytes=32768)
                manager=patch("core.models.common.single_file_format.save_single_file_state",side_effect=write) if sharded else patch("builtins.print")
                with manager:
                    written=Ltx2FullParameterAdapter(trainer).write_checkpoint(3,1,path)
                self.assertEqual(str(written).endswith(".index.json"),sharded)
                restored=load_ltx2_transformer_checkpoint(str(path),torch.float32)
                self.assertEqual(restored._residual_unet_spec,self.spec)
                self.assertEqual(restored.config.num_layers,7)
                for key,value in self.inner.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
                actual=self.forward(model=restored)
                # CPU gate GEMM rounding depends on Safetensors weight alignment.
                for a,b in zip(actual,original):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_checkpoint_preflight_refuses_malformed_identity_before_tensor_loading(self):
        import json,tempfile
        from safetensors.torch import save_file
        from core.models.ltx2.loader import load_ltx2_transformer_checkpoint
        from core.models.ltx2.residual_unet import export_metadata
        from core.rewiring.spec import checkpoint_metadata
        state={"net."+key:value for key,value in self.inner.state_dict().items()}
        metadata=export_metadata(self.model)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.safetensors"
            with patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("No tensor loading")):
                broken=deepcopy(self.spec);broken["base_config_sha256"]="f"*64
                save_file(state,str(path),metadata={**metadata,**checkpoint_metadata(broken)})
                with self.assertRaises(ValueError):load_ltx2_transformer_checkpoint(str(path))
                wrong=dict(state);wrong["net.proj_in.weight"]=torch.ones(31,8)
                save_file(wrong,str(path),metadata=metadata)
                with self.assertRaisesRegex(ValueError,"header geometry"):load_ltx2_transformer_checkpoint(str(path))
                save_file(state,str(path),metadata=checkpoint_metadata(self.spec))
                with self.assertRaisesRegex(ValueError,"transformer_config"):load_ltx2_transformer_checkpoint(str(path))
                save_file({"lora_unet_transformer_blocks_0_attn1_to_q.lora_down.weight":torch.ones(2,32)},str(path),metadata=metadata)
                with self.assertRaisesRegex(ValueError,"native AV transformer"):load_ltx2_transformer_checkpoint(str(path))
        self.assertFalse(torch.cuda.is_initialized())

    def test_directory_loader_restores_graph_before_companions(self):
        import json, tempfile
        from types import SimpleNamespace
        from safetensors.torch import save_file
        from core.models.ltx2.loader import load_ltx2_from_diffusers
        from core.models.ltx2.residual_unet import export_metadata
        from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper
        with tempfile.TemporaryDirectory() as folder:
            component=Path(folder)/"transformer";component.mkdir()
            config=dict(self.inner.config)
            (component/"config.json").write_text(json.dumps(config),encoding="utf-8")
            save_file(self.inner.state_dict(),str(component/"diffusion_pytorch_model.safetensors"),
                      metadata=export_metadata(self.model))
            def assemble(path, **kwargs):
                self.assertIsInstance(kwargs["transformer"],type(self.inner))
                self.assertNotIsInstance(kwargs["transformer"],Ltx2BlockLoopWrapper)
                return SimpleNamespace(transformer=kwargs["transformer"],vae=None)
            with patch("core.models.ltx2.LTX2Pipeline.from_pretrained",side_effect=assemble) as assemble_mock:
                result=load_ltx2_from_diffusers(folder,torch.float32)
                self.assertIs(result["transformer"],result["pipeline"].transformer)
                self.assertIsInstance(result["transformer"],Ltx2BlockLoopWrapper)
                self.assertEqual(result["transformer"]._residual_unet_spec,self.spec)
                for key,value in self.inner.state_dict().items():
                    torch.testing.assert_close(result["transformer"].state_dict()[key],value,rtol=0,atol=0)
                assemble_mock.assert_called_once()
            config["num_layers"]=8
            (component/"config.json").write_text(json.dumps(config),encoding="utf-8")
            with patch("core.models.ltx2.LTX2Pipeline.from_pretrained",side_effect=AssertionError("No companion load")), \
                    patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("No tensors")):
                with self.assertRaises(ValueError):load_ltx2_from_diffusers(folder,torch.float32)
        self.assertFalse(torch.cuda.is_initialized())

    def test_zero_swap_preserves_residual_wrapper_and_offload_order(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from core.pipeline_backends.ltx2 import LTX2Mixin
        manager=LTX2Mixin()
        pipeline=SimpleNamespace(transformer=self.model)
        manager.ltx2_components=dict(pipeline=pipeline,transformer=self.model,
            i2v_pipeline=SimpleNamespace(transformer=self.model),cond_pipeline=SimpleNamespace(transformer=self.model))
        calls=[]
        manager._ensure_ltx2_offload=Mock(side_effect=lambda **kwargs:calls.append(kwargs) or pipeline)
        manager._ensure_ltx2_swap_and_offload(0)
        self.assertEqual(calls,[dict(blocks_to_swap=0,force_block_swap_mode=True)])
        manager._ensure_ltx2_block_swap_wrapper(0)
        self.assertIs(pipeline.transformer,self.model)
        self.assertEqual(pipeline.transformer._residual_unet_spec,self.spec)
        self.assertFalse(torch.cuda.is_initialized())

    def test_invalid_layout_and_conflicting_features_refuse_before_projection(self):
        coords=self.inner.rope.prepare_video_coords(2,3,8,12,self.video.device,fps=24.)
        bad_coords=coords.clone();bad_coords[0,0,1,0]+=1
        with patch.object(self.inner.proj_in,"forward",side_effect=AssertionError("No projection before validation")):
            for kwargs in (dict(height=6),dict(num_frames=None),dict(video_coords=bad_coords),
                           dict(spatio_temporal_guidance_blocks=[1]),dict(perturbation_mask=torch.ones(2)),
                           dict(attention_kwargs={"scale":.5}),dict(sigma=torch.ones(2,288))):
                with self.subTest(kwargs=list(kwargs)),self.assertRaises(ValueError):self.forward(**kwargs)
            for slot in ("_fbcache","_spectrum_video","_tread_router","_blockskip_config","_style_cfg"):
                setattr(self.model,slot,object())
                try:
                    with self.subTest(slot=slot),self.assertRaises(ValueError):self.forward()
                finally:setattr(self.model,slot,None)
        self.assertEqual(self.inner._residual_unet_spec,self.spec)
        self.assertFalse(torch.cuda.is_initialized())
