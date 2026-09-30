"""Independent residual oracle and native Qwen checkpointed backward checks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import OPERATORS, build_spec


def topology():
    return build_spec("qwen_image_21", [{"name":"main","block_count":7}], [
        dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,
             down_factor=[2,2],**OPERATORS),
        dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,
             down_factor=[2,2],**OPERATORS)], {})


def linear(weights, value):
    for weight in weights:
        value = value + value @ weight
    return value


def pool(value, h, w):
    return F.avg_pool2d(value.transpose(1,2).reshape(value.shape[0],-1,h,w),2).flatten(2).transpose(1,2)


def up(value, h, w):
    return F.interpolate(value.transpose(1,2).reshape(value.shape[0],-1,h//2,w//2),
                         size=(h,w),mode="bilinear",align_corners=False).flatten(2).transpose(1,2)


class RuntimeContracts(unittest.TestCase):
    def test_nested_values_and_gradients(self):
        torch.manual_seed(42)
        x = torch.randn(2,5+8*12,4,requires_grad=True)
        weights = [(torch.randn(4,4)*0.02).requires_grad_() for _ in range(7)]
        entered = linear(weights[:1],x)
        low = pool(entered[:,5:],8,12)
        pre_inner = linear(weights[1:2],low)
        inner_low = pool(pre_inner,4,6)
        inner = pre_inner + up(linear(weights[2:5],inner_low)-inner_low,4,6)
        image = entered[:,5:] + up(linear(weights[5:6],inner)-low,8,12)
        expected = linear(weights[6:],torch.cat((linear(weights[1:6],entered[:,:5]),image),dim=1))
        actual = run_rewired_stream(x,topology(),"main",8,12,5,None,
            lambda i,h,c: linear(weights[i:i+1],h),lambda c,h,y,w,p: c)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        a = torch.autograd.grad(actual.square().sum(),[x,*weights],retain_graph=True)
        b = torch.autograd.grad(expected.square().sum(),[x,*weights])
        for left,right in zip(a,b):
            torch.testing.assert_close(left,right,rtol=3e-5,atol=3e-5)

    def test_grouped_frames_match_independent_oracle_and_gradients(self):
        torch.manual_seed(102)
        batch,frames,height,width,channels=2,3,8,12,4
        video=torch.randn(batch,frames*height*width,channels,requires_grad=True)
        weights=[(torch.randn(channels,channels)*.03).requires_grad_() for _ in range(7)]
        def execute(i,value,context):return value+torch.tanh(value@weights[i])
        actual=run_rewired_stream(video,topology(),"main",frames*height,width,0,None,
                                 execute,lambda *args:None,grid_groups=frames)
        independent=[]
        for frame in range(frames):
            value=video.reshape(batch,frames,height*width,channels)[:,frame]
            independent.append(run_rewired_stream(value,topology(),"main",height,width,0,None,
                                                 execute,lambda *args:None))
        expected=torch.cat(independent,dim=1)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        left=torch.autograd.grad(actual.square().sum(),[video,*weights],retain_graph=True)
        right=torch.autograd.grad(expected.square().sum(),[video,*weights])
        for a,b in zip(left,right):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-5)
        with self.assertRaises(ValueError):
            run_rewired_stream(video,topology(),"main",24,width,0,None,execute,lambda *args:None,grid_groups=4)
        with self.assertRaises(ValueError):
            run_rewired_stream(video,topology(),"main",24,width,0,None,execute,lambda *args:None,grid_groups=0)

    def test_identity_dtypes_and_local_context(self):
        for dtype in (torch.float32,torch.bfloat16,torch.float16):
            x = torch.randn(1,5+8*12,4).to(dtype)
            seen = []
            def execute(i,h,c):
                seen.append(c)
                return h
            actual = run_rewired_stream(x,topology(),"main",8,12,5,0,execute,
                                        lambda c,h,y,w,p: c+1)
            self.assertTrue(torch.equal(x,actual))
            self.assertEqual(seen,[0,1,2,2,2,1,0])

    def test_temporal_only_nested_transport_matches_linear_oracle(self):
        torch.manual_seed(93)
        regions = [dict(id="outer", stream="main", parent=None, start_block=1,
            end_block=6, down_factor=[2, 1], **OPERATORS),
            dict(id="inner", stream="main", parent="outer", start_block=2,
            end_block=5, down_factor=[2, 1], **OPERATORS)]
        spec = build_spec("temporal_fixture", [{"name": "main", "block_count": 7}], regions, {})
        x = torch.randn(2, 2+24, 4, requires_grad=True)
        weights = [(torch.randn(4,4)*0.02).requires_grad_() for _ in range(7)]
        def temporal_pool(value):
            return F.avg_pool1d(value.transpose(1,2), 2).transpose(1,2)
        def temporal_up(value, length):
            return F.interpolate(value.transpose(1,2), size=length,
                                 mode="linear", align_corners=False).transpose(1,2)
        entered = linear(weights[:1], x)
        low = temporal_pool(entered[:,2:])
        pre_inner = linear(weights[1:2], low)
        inner_low = temporal_pool(pre_inner)
        inner = pre_inner + temporal_up(linear(weights[2:5], inner_low)-inner_low, 12)
        sequence = entered[:,2:] + temporal_up(linear(weights[5:6], inner)-low, 24)
        expected = linear(weights[6:], torch.cat((linear(weights[1:6], entered[:,:2]), sequence), dim=1))
        seen = []
        def execute(index, hidden, context):
            seen.append((index, hidden.shape[1]-2, context))
            return linear(weights[index:index+1], hidden)
        actual = run_rewired_stream(x, spec, "main", 24, 1, 2, 0, execute,
                                   lambda c,h,y,w,p: c+1)
        self.assertEqual(seen, [(i,n,c) for i,(n,c) in enumerate(zip(
            [24,12,6,6,6,12,24], [0,1,2,2,2,1,0]))])
        torch.testing.assert_close(actual, expected)
        a = torch.autograd.grad(actual.square().sum(), [x,*weights], retain_graph=True)
        b = torch.autograd.grad(expected.square().sum(), [x,*weights])
        for left,right in zip(a,b): torch.testing.assert_close(left,right)
        for region in regions: region["down_factor"] = [1,2]
        horizontal = build_spec("temporal_fixture", spec["streams"], regions, {})
        other = run_rewired_stream(x, horizontal, "main", 1, 24, 2, None,
            lambda i,h,c: linear(weights[i:i+1],h), lambda c,h,y,w,p: c)
        torch.testing.assert_close(other, expected)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            value = x.detach().to(dtype)
            identity = run_rewired_stream(value, spec, "main", 24, 1, 2, None,
                lambda i,h,c: h, lambda c,h,y,w,p: c)
            self.assertTrue(torch.equal(identity,value))

    def test_native_qwen_checkpoint_backward(self):
        with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
             patch.object(torch.cuda,"_lazy_init"),patch.object(torch._C,"_cuda_init"):
            from core.models.qwen_image_21.vendor.transformer import QwenImage21Transformer2DModel
            from core.models.qwen_image_21.residual_unet import run_rewired_qwen
        torch.manual_seed(7)
        model = QwenImage21Transformer2DModel(num_layers=7,num_attention_heads=1,
            attention_head_dim=12,axes_dims_rope=(4,4,4),context_in_dim=12,mlp_ratio=1)
        x = torch.randn(1,5+8*12,12,requires_grad=True)
        modulation = torch.randn(2,48,requires_grad=True)
        mask = torch.zeros(x.shape[1],dtype=torch.bool); mask[5:]=True
        rotary = model.pos_embed([(1,8,12)],mask,x.device)
        def run():
            return run_rewired_qwen(model,x,modulation,rotary,None,mask,None,None,(1,8,12),topology())
        dense = run()
        parameters = [x,modulation,*model.parameters()]
        used = [p for p in parameters if p.requires_grad]
        dense_grads = torch.autograd.grad(dense.square().mean(),used,allow_unused=True)
        model.enable_gradient_checkpointing()
        checkpointed = run()
        cp_grads = torch.autograd.grad(checkpointed.square().mean(),used,allow_unused=True)
        torch.testing.assert_close(dense,checkpointed)
        for a,b in zip(dense_grads,cp_grads):
            if a is None:
                self.assertIsNone(b)
            else:
                self.assertTrue(torch.isfinite(b).all())
                torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
        model._residual_unet_spec = topology()
        image = torch.randn(1,8*12,64,requires_grad=True)
        text = torch.randn(1,5,12)
        image_slots = torch.zeros(1,5+(8*12)//4,dtype=torch.bool)
        image_slots[:,5:] = True
        result = model(image,text,torch.tensor([0.5]),[[(1,8,12)]],image_slots).sample
        self.assertEqual(result.shape,(1,5+8*12,64))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
