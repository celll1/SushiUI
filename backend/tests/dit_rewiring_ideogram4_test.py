"""CPU contracts for Ideogram 4 native packed residual transport."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import build_spec, OPERATORS, read_checkpoint_metadata


class Ideogram4RuntimeContracts(unittest.TestCase):
    def setUp(self):
        for target,name,value in ((torch.cuda,"get_device_capability",lambda *a,**k:(8,9)),
                (torch.cuda,"_lazy_init",lambda *a,**k:None),(torch._C,"_cuda_init",lambda *a,**k:None)):
            guard=patch.object(target,name,value);guard.start();self.addCleanup(guard.stop)
        from core.models.ideogram4.vendor.transformer import Ideogram4Transformer2DModel
        from core.models.ideogram4.ideogram4_pipeline_ops import _prepare_ids
        from core.models.ideogram4.residual_unet import native_config
        torch.manual_seed(104)
        self.model=Ideogram4Transformer2DModel(in_channels=8,num_layers=7,attention_head_dim=12,
            num_attention_heads=2,intermediate_size=32,adaln_dim=8,llm_features_dim=12,mrope_section=(2,2,2))
        self.spec=build_spec("ideogram4",[{"name":"main","block_count":7}],[
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
            dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)],native_config(self.model))
        self.positions,self.segments,self.indicator=_prepare_ids([3,5],8,12,5,torch.device("cpu"))
        self.images=torch.randn(2,101,8,requires_grad=True)
        self.features=torch.randn(2,101,12,requires_grad=True)
        self.times=torch.tensor([0.3,0.7])

    def forward(self,**options):
        arguments=dict(hidden_states=self.images,timestep=self.times,encoder_hidden_states=self.features,
            position_ids=self.positions,segment_ids=self.segments,indicator=self.indicator,return_dict=False)
        arguments.update(options);return self.model(**arguments)[0]

    def test_native_masks_positions_evolving_text_and_offloader(self):
        from core.models.ideogram4.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        identities={name:id(value) for name,value in self.model.named_parameters()}
        counts=[];previous=[];events=[];rope_values=[]
        def observe(block,args):
            hidden,mask,rope,adaln,segments=args
            count=hidden.shape[1]-5;counts.append(count)
            self.assertEqual(segments[:,:5].tolist(),self.segments[:,:5].tolist())
            self.assertTrue(torch.equal(mask,(segments[:,:,None]==segments[:,None,:])[:,None]))
            self.assertFalse(mask[0,0,0,-1]);self.assertTrue(mask[0,0,2,-1])
            self.assertEqual(rope[0].shape,(2,count+5,12))
            if previous:torch.testing.assert_close(hidden[:,:5],previous[-1],rtol=0,atol=0)
        handles=[block.register_forward_pre_hook(observe) for block in self.model.layers]
        handles += [block.register_forward_hook(lambda block,args,out:previous.append(out[:,:5])) for block in self.model.layers]
        handle=self.model.rotary_emb.register_forward_pre_hook(lambda module,args:rope_values.append(args[0].detach().clone()))
        class Offloader:
            def wait_for_block(owner,index):events.append(("wait",index))
            def submit_move_blocks_forward(owner,index):events.append(("submit",index))
        self.model._block_offloader=Offloader()
        try:output=self.forward()
        finally:
            for item in handles:item.remove()
            handle.remove();del self.model._block_offloader
        self.assertEqual(counts,[96,24,6,6,6,24,96])
        self.assertEqual(events,[event for i in range(7) for event in (("wait",i),("submit",i))])
        self.assertEqual(identities,{name:id(value) for name,value in self.model.named_parameters()})
        self.assertEqual(output.shape,self.images.shape);self.assertTrue(torch.isfinite(output).all())
        coarse=next(pos for pos in rope_values if pos.shape[1]==29)
        deep=next(pos for pos in rope_values if pos.shape[1]==11)
        torch.testing.assert_close(coarse[0,5],torch.tensor([65536.,65536.5,65536.5]))
        torch.testing.assert_close(deep[0,5],torch.tensor([65536.,65537.5,65537.5]))
        torch.testing.assert_close(coarse[:,:5],self.positions[:,:5].float(),rtol=0,atol=0)
        # The unconditional backbone uses the same native image-only layout.
        value=self.forward(hidden_states=self.images[:,5:],encoder_hidden_states=self.features[:,5:],
            position_ids=self.positions[:,5:],segment_ids=self.segments[:,5:],indicator=self.indicator[:,5:])
        self.assertEqual(value.shape,(2,96,8));self.assertTrue(torch.isfinite(value).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_checkpoint_output_gradients_and_dense_restoration(self):
        from core.models.ideogram4.residual_unet import restore_topology
        dense=self.forward().detach();restore_topology(self.model,self.spec)
        parameters=[self.images,self.features,*self.model.parameters()]
        direct=self.forward();grads=torch.autograd.grad(direct.square().mean(),parameters,allow_unused=True)
        self.model.enable_gradient_checkpointing()
        checkpointed=self.forward();checkpoint_grads=torch.autograd.grad(checkpointed.square().mean(),parameters,allow_unused=True)
        torch.testing.assert_close(direct,checkpointed,rtol=0,atol=0)
        for before,after in zip(grads,checkpoint_grads):
            self.assertEqual(before is None,after is None)
            if before is not None:
                self.assertTrue(torch.isfinite(after).all());torch.testing.assert_close(before,after)
        self.assertTrue(torch.count_nonzero(checkpoint_grads[0]));self.assertTrue(torch.count_nonzero(checkpoint_grads[1]))
        restore_topology(self.model,None);self.model.disable_gradient_checkpointing()
        torch.testing.assert_close(self.forward(),dense,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_metadata_identity_native_grid_and_feature_refusals(self):
        from core.models.ideogram4.residual_unet import restore_topology,model_topology_metadata,native_config
        restore_topology(self.model,self.spec)
        self.assertEqual(read_checkpoint_metadata(model_topology_metadata(self.model)),self.spec)
        for key,value in (("num_layers",True),("norm_eps",float("nan")),("mrope_section",[2,3,3])):
            with self.assertRaises(ValueError):native_config(dict(self.model.config,**{key:value}))
        for key,value in (("_fbcache",object()),("_repa_tap_depth",0),("_nag_active",True)):
            setattr(self.model,key,value)
            try:
                with self.assertRaisesRegex(ValueError,key):self.forward()
            finally:self.model.__dict__.pop(key,None)
        positions=self.positions.clone();positions[0,8,2]+=1
        with self.assertRaisesRegex(ValueError,"row-major"):self.forward(position_ids=positions)
        indicator=self.indicator.clone();indicator[0,7]=3
        with self.assertRaisesRegex(ValueError,"contiguous image tail"):self.forward(indicator=indicator)
        segments=self.segments.clone();segments[0,7]=8
        with self.assertRaisesRegex(ValueError,"one image segment"):self.forward(segment_ids=segments)
        with self.assertRaisesRegex(ValueError,"per-sample timestep"):
            self.forward(timestep=self.times[:,None].expand(-1,101))
        with self.assertRaises(ValueError):restore_topology(self.model,dict(self.spec,base_config_sha256="0"*64))
        restore_topology(self.model,None);self.assertEqual(model_topology_metadata(self.model),{})
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":unittest.main()
