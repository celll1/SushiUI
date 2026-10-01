"""CPU target-only Ming style capture/injection and request-scoped cleanup."""
from types import SimpleNamespace
import pytest
import torch
from core.inference.reference_style import StyleTransferConfig,StyleContext,inject_kv,make_ref_value
from core.models.ming_image_01.style_transfer import style_attention,style_scope
from core.models.ming_image_01.model_definition import create_transformer
from ming_image_01_rewiring_test import config,spec_for,restore_topology


@pytest.mark.parametrize('auxiliary',[False,True])
@pytest.mark.parametrize('mask_shape',['key','query'])
def test_style_injection_matches_shared_oracle_and_preserves_native_tail(auxiliary,mask_shape):
    torch.manual_seed(17)
    frames=3 if auxiliary else 1;count=frames*4;start=4 if auxiliary else 0
    q,k,v=[torch.randn(2,count+7,3,8) for _ in range(3)]
    cfg=StyleTransferConfig(block_range=(0,0),axes_dims=(2,2,4),adain_strength=.4,ref_k_strength=.7)
    capture=StyleContext('capture',cfg)
    attn=SimpleNamespace(_ming_style_info={'context':capture,'index':0,'count':count,'frames':frames,'exclude_auxiliary':auxiliary})
    ref=[torch.randn_like(tensor) for tensor in (q,k,v)]
    style_attention(attn,*ref,None)
    assert capture.store[0][0].shape[1]==count-start
    assert all(not tensor.requires_grad for tensor in capture.store[0])
    ctx=StyleContext('inject',cfg,store=capture.store,progress=.25)
    attn._ming_style_info['context']=ctx
    mask=torch.ones(2,count+7,dtype=torch.bool);mask[:,-3:]=False
    if mask_shape=='query':mask=mask[:,None,None,:].expand(2,1,count+7,count+7).clone()
    actual_q,actual_k,actual_v,actual_mask=style_attention(attn,q,k,v,mask)
    ref_q,ref_k,ref_v=capture.store[0]
    ref_value=make_ref_value(v[:,start:count],ref_v,cfg.value_mode,cfg.value_adain_strength,cfg.ref_value_mix)
    expected_k,expected_v,expected_q=inject_kv(k,v,ref_k,ref_value,start,count,cfg.ref_k_strength,cfg.get_freq_scale_vector(8,.25,k.device,k.dtype),cfg.adain_strength,q=q,ref_q=ref_q)
    for actual,expected in zip((actual_q,actual_k,actual_v),(expected_q,expected_k,expected_v)):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(actual_q[:,:start],q[:,:start]) and torch.equal(actual_q[:,count:],q[:,count:])
    assert torch.equal(actual_k[:,count:count+7],k[:,count:])
    assert torch.equal(actual_mask[...,:mask.shape[-1]],mask) and actual_mask[...,mask.shape[-1]:].all()


@pytest.mark.parametrize('padding',['zero_masked','learned'])
@pytest.mark.parametrize('rewired',[False,True])
def test_actual_native_capture_per_layer_geometry_injection_and_cleanup(padding,rewired):
    torch.manual_seed(23)
    model=create_transformer(config(padding),device='cpu',dtype=torch.float32).eval()
    if padding=='learned':
        torch.nn.init.constant_(model.x_pad_token,.02);torch.nn.init.constant_(model.cap_pad_token,.03)
    if rewired:restore_topology(model,spec_for(model))
    frames=3 if padding=='learned' else 1
    x=[torch.randn(16,frames,16,16) for _ in range(2)]
    style=[torch.randn_like(item) for item in x]
    caps=[torch.randn(5,8) for _ in x];direct=[torch.randn(3,24),torch.randn(35,24)]
    refs=[torch.randn(16,1,16,16),torch.randn(16,2,16,16)]
    args=dict(t=torch.tensor([.2,.6]),cap_feats=caps,cap_feats_2=direct,ref_x=refs,return_dict=False)
    processors=[layer.attention.processor for layer in model.layers]
    state={key:value.clone() for key,value in model.state_dict().items()}
    cfg=StyleTransferConfig(block_range=(0,5),ref_k_strength=.4,adain_strength=.3)
    capture=StyleContext('capture',cfg)
    with torch.no_grad():
        baseline=model(x=x,**args)[0]
        with style_scope(model,capture,style):model(x=style,**args)
        counts=[64,16,4,4,16,64] if rewired else [64]*6
        factor=frames-1 if frames>1 else 1
        assert [capture.store[index][0].shape[1] for index in range(6)]==[factor*count for count in counts]
        with style_scope(model,StyleContext('inject',cfg,store=capture.store,progress=.2),x):
            actual=model(x=x,**args)[0]
        after=model(x=x,**args)[0]
    assert all(torch.equal(a,b) for a,b in zip(after,baseline))
    assert all(torch.isfinite(value).all() for value in actual)
    assert any(not torch.equal(a[:,:frames],b[:,:frames]) for a,b in zip(actual,baseline))
    assert all(layer.attention.processor is processor for layer,processor in zip(model.layers,processors))
    assert all(not hasattr(layer.attention,'_ming_style_info') for layer in model.layers)
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in state.items())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('combine',['stack','common_concept'])
def test_multi_reference_injection_uses_shared_combiner(combine):
    from core.inference.reference_style import inject_kv_multi
    torch.manual_seed(29)
    q,k,v=[torch.randn(1,8,3,8) for _ in range(3)]
    cfg=StyleTransferConfig(block_range=(0,0),axes_dims=(2,2,4))
    refs=[({0:tuple(torch.randn(1,4,3,8) for _ in range(3))},cfg) for _ in range(2)]
    ctx=StyleContext('inject',cfg,refs=refs,combine_mode=combine,progress=.2)
    attn=SimpleNamespace(_ming_style_info={'context':ctx,'index':0,'count':4,'frames':1,'exclude_auxiliary':False})
    actual=style_attention(attn,q,k,v,None)
    expected_k,expected_v,expected_q=inject_kv_multi(k,v,q,0,4,ctx.collect_block_refs(0,v[:,:4],k.device,k.dtype),combine)
    for a,b in zip(actual[:3],(expected_q,expected_k,expected_v)):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_zero_strength_and_adain_is_exact_noop():
    cfg=StyleTransferConfig(block_range=(0,0),axes_dims=(2,2,4),ref_k_strength=0,adain_strength=0)
    q,k,v=[torch.randn(1,8,3,8) for _ in range(3)];mask=torch.ones(1,8,dtype=torch.bool)
    ctx=StyleContext('inject',cfg,store={0:(q[:,:4],k[:,:4],v[:,:4])})
    attn=SimpleNamespace(_ming_style_info={'context':ctx,'index':0,'count':4,'frames':1,'exclude_auxiliary':False})
    actual=style_attention(attn,q,k,v,mask)
    assert all(a is b for a,b in zip(actual,(q,k,v,mask)))


def test_nested_scope_restores_previous_geometry_on_exception():
    model=create_transformer(config('zero_masked'),device='cpu',dtype=torch.float32)
    x=[torch.randn(16,1,16,16)];cfg=StyleTransferConfig(block_range=(0,5))
    original=[layer.attention.processor for layer in model.layers]
    with torch.no_grad(),style_scope(model,StyleContext('capture',cfg),x):
        outer=[layer.attention._ming_style_info for layer in model.layers]
        with pytest.raises(RuntimeError,match='cancel'):
            with style_scope(model,StyleContext('capture',cfg),x):raise RuntimeError('cancel')
        assert all(layer.attention._ming_style_info is info for layer,info in zip(model.layers,outer))
    assert all(layer.attention.processor is processor for layer,processor in zip(model.layers,original))
    assert all(not hasattr(layer.attention,'_ming_style_info') for layer in model.layers)


def test_partial_scope_failure_restores_earlier_layers():
    from core.models.ming_image_01.attention import MingAttentionProcessor
    model=create_transformer(config('zero_masked'),device='cpu',dtype=torch.float32)
    model.layers[2].attention.set_processor(MingAttentionProcessor('flash',0))
    before=[layer.attention.processor for layer in model.layers]
    with torch.no_grad(),pytest.raises(ValueError,match='native'):
        with style_scope(model,StyleContext('capture',StyleTransferConfig()),[torch.randn(16,1,16,16)]):pytest.fail('flash allowed')
    assert all(layer.attention.processor is processor for layer,processor in zip(model.layers,before))
    assert all(not hasattr(layer.attention,'_ming_style_info') for layer in model.layers)


def test_grad_scope_is_refused_before_mutation():
    model=create_transformer(config('zero_masked'),device='cpu',dtype=torch.float32)
    with pytest.raises(ValueError,match='inference-only'):
        with style_scope(model,StyleContext('capture',StyleTransferConfig()),[torch.randn(16,1,16,16)]):pytest.fail('training allowed')

@pytest.mark.parametrize('padding',['zero_masked','learned'])
def test_style_scope_preserves_full_kv_query_chunking_through_nested_unet(padding):
    from copy import deepcopy
    from core.models.ming_image_01.attention import install_attention
    torch.manual_seed(31)
    native=create_transformer(config(padding),device='cpu',dtype=torch.float32).eval()
    if padding=='learned':
        torch.nn.init.constant_(native.x_pad_token,.02);torch.nn.init.constant_(native.cap_pad_token,.03)
    restore_topology(native,spec_for(native))
    chunked=deepcopy(native)
    install_attention(native,'native',query_chunk_tokens=0);install_attention(chunked,'native',query_chunk_tokens=17)
    frames=3 if padding=='learned' else 1
    x=[torch.randn(16,frames,16,16) for _ in range(2)];style=[torch.randn_like(value) for value in x]
    args=dict(t=torch.tensor([.2,.6]),cap_feats=[torch.randn(5,8) for _ in x],cap_feats_2=[torch.randn(3,24),torch.randn(35,24)],ref_x=[torch.randn(16,1,16,16),torch.randn(16,2,16,16)],return_dict=False)
    results=[]
    with torch.no_grad():
        for model in [native,chunked]:
            cfg=StyleTransferConfig(block_range=(0,5));capture=StyleContext('capture',cfg)
            with style_scope(model,capture,style):model(x=style,**args)
            with style_scope(model,StyleContext('inject',cfg,store=capture.store),x):results.append(model(x=x,**args)[0])
    for full,chunk in zip(*results):torch.testing.assert_close(full,chunk,rtol=2e-5,atol=2e-6)
    assert all(layer.attention.processor._query_chunk_tokens==17 for layer in chunked.layers)
    assert not torch.cuda.is_initialized()



def test_scope_resolves_native_rope_pair_layout_and_invalidates_other_arch_cache():
    model=create_transformer(config('zero_masked'),device='cpu',dtype=torch.float32)
    cfg=StyleTransferConfig(axes_dims=(4,4),rope_layout='rotate_half')
    cfg._freq_cache['old']=torch.ones(8)
    with torch.no_grad(),style_scope(model,StyleContext('capture',cfg),[torch.randn(16,1,16,16)]):
        assert cfg.axes_dims==(2,2,4) and cfg.rope_layout=='interleaved'
        assert cfg._freq_cache=={}
