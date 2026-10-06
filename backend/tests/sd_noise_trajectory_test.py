"""A clean oracle separates actual ancestral noise from sampler history."""
import inspect

import diffusers
import pytest
import torch

from core.inference.sd_codec_boundary import SDNoiseTrajectory,coefficients


@pytest.mark.parametrize("name",["DDIMScheduler","DDPMScheduler","PNDMScheduler",
    "EulerDiscreteScheduler","EulerAncestralDiscreteScheduler","LMSDiscreteScheduler",
    "HeunDiscreteScheduler","KDPM2DiscreteScheduler","KDPM2AncestralDiscreteScheduler",
    "DPMSolverMultistepScheduler","DPMSolverSinglestepScheduler","UniPCMultistepScheduler",
    "FlowMatchEulerDiscreteScheduler","sde-dpmsolver++"])
@pytest.mark.parametrize("prediction",["epsilon","v_prediction"])
def test_noise_history_matches_clean_oracle_without_advancing_owner_rng(name,prediction):
    if name=="PNDMScheduler" and prediction=="v_prediction":
        scheduler=diffusers.PNDMScheduler(prediction_type=prediction)
        scheduler.set_timesteps(8)
        with pytest.raises(ValueError,match="PNDM v-prediction"):
            SDNoiseTrajectory(scheduler,torch.randn(1,8,5,7))
        return
    cls=getattr(diffusers,"DPMSolverMultistepScheduler" if name=="sde-dpmsolver++" else name)
    options=dict(num_train_timesteps=100)
    parameters=inspect.signature(cls.__init__).parameters
    if "prediction_type" in parameters:options["prediction_type"]=prediction
    if "clip_sample" in parameters:options["clip_sample"]=False
    if "thresholding" in parameters:options["thresholding"]=False
    if name=="PNDMScheduler":options["skip_prk_steps"]=True
    if name=="sde-dpmsolver++":options["algorithm_type"]=name
    scheduler=cls(**options);scheduler.set_timesteps(8)
    clean=torch.randn(1,8,5,7,generator=torch.Generator().manual_seed(12))*.3
    noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(13))
    if name=="FlowMatchEulerDiscreteScheduler":
        state=scheduler.scale_noise(clean,scheduler.timesteps[:1],noise)
        noise_state=scheduler.scale_noise(torch.zeros_like(clean),scheduler.timesteps[:1],noise)
    else:
        state=scheduler.add_noise(clean,noise,scheduler.timesteps[:1])
        noise_state=scheduler.add_noise(torch.zeros_like(clean),noise,scheduler.timesteps[:1])
    tracker=SDNoiseTrajectory(scheduler,noise_state)
    generator=torch.Generator().manual_seed(17)
    initial_noise=noise.clone();changed=False
    for t in scheduler.timesteps:
        scaled=state if name=="FlowMatchEulerDiscreteScheduler" else scheduler.scale_model_input(state,t)
        alpha,sigma,process=coefficients(scheduler,t,scaled)
        exposure=tracker.noise(t)
        torch.testing.assert_close(scaled,alpha*clean+sigma*exposure,rtol=2e-5,atol=2e-6)
        changed=changed or not torch.allclose(exposure,initial_noise,rtol=1e-4,atol=1e-4)
        actual=(scaled-alpha*clean)/sigma
        output=(actual-clean if process=="flow" else
            actual if prediction=="epsilon" else alpha*actual-sigma*clean)
        kwargs={"eta":.7} if name=="DDIMScheduler" else {}
        before=generator.get_state().clone()
        tracker.step(t,generator=generator,**kwargs)
        assert torch.equal(generator.get_state(),before)
        if "generator" in inspect.signature(scheduler.step).parameters:
            kwargs["generator"]=generator
        state=scheduler.step(output,t,state,**kwargs).prev_sample
        assert torch.isfinite(state).all() and torch.isfinite(tracker.state).all()
    if name in ("EulerAncestralDiscreteScheduler","KDPM2AncestralDiscreteScheduler",
                "DDIMScheduler","DDPMScheduler","sde-dpmsolver++"):
        assert changed


def test_unsupported_non_linear_or_pndm_v_history_refuses():
    cases=[diffusers.DDPMScheduler(),diffusers.PNDMScheduler(prediction_type="v_prediction"),
        diffusers.DPMSolverMultistepScheduler(thresholding=True),
        diffusers.DDPMScheduler(clip_sample=False,variance_type="learned_range"),
        diffusers.FlowMatchEulerDiscreteScheduler(stochastic_sampling=True)]
    for scheduler in cases:
        scheduler.set_timesteps(4)
        with pytest.raises(ValueError):SDNoiseTrajectory(scheduler,torch.randn(1,8,5,7))


def test_unvalidated_step_options_refuse_before_mutating_history():
    scheduler=diffusers.EulerDiscreteScheduler();scheduler.set_timesteps(4)
    tracker=SDNoiseTrajectory(scheduler,torch.randn(1,8,5,7))
    before=tracker.state.clone()
    for kwargs in ({"s_churn":.1},{"per_token_timesteps":torch.ones(1,5,7)}):
        with pytest.raises(ValueError,match="validated scalar timestep"):
            tracker.step(scheduler.timesteps[0],generator=torch.Generator(),**kwargs)
        assert tracker.scheduler.step_index is None and torch.equal(tracker.state,before)


def test_generator_list_is_independent_and_missing_rng_refuses():
    owners=[torch.Generator().manual_seed(seed) for seed in (12,13)]
    clones=SDNoiseTrajectory.clone_generator(owners)
    for owner,clone in zip(owners,clones):
        before=owner.get_state().clone()
        torch.randn(3,generator=clone)
        assert torch.equal(owner.get_state(),before)
    with pytest.raises(ValueError,match="explicit step generator"):
        SDNoiseTrajectory.clone_generator(None)


@pytest.mark.parametrize("batch",[1,2,3])
def test_session_uses_reference_noise_only_for_the_requested_exposure(batch):
    from core.inference.sd_codec_boundary import SDNoiseSession
    scheduler=diffusers.EulerAncestralDiscreteScheduler();scheduler.set_timesteps(4)
    state=torch.randn(1,8,5,7)*scheduler.init_noise_sigma
    session=SDNoiseSession();session.initialize(scheduler,state,torch.Generator().manual_seed(19))
    t=scheduler.timesteps[0];reference=torch.randn(1,8,5,7)
    baseline=session.exposure(t,torch.zeros(batch,8,5,7))
    overridden=session.exposure(t,torch.zeros(batch,8,5,7),reference)
    torch.testing.assert_close(overridden,reference.repeat(batch,1,1,1),rtol=0,atol=0)
    torch.testing.assert_close(session.exposure(t,torch.zeros(batch,8,5,7)),baseline,rtol=0,atol=0)


@pytest.mark.parametrize("explicit",[False,True])
def test_wired_step_preserves_native_output_and_consumes_rng_once(explicit):
    import copy
    from types import SimpleNamespace
    from core.inference.sd_codec_boundary import (SDNoiseSession,initialize_noise_tracking,
        aligned_scheduler_step,blend_tracked_noise)
    scheduler=diffusers.EulerAncestralDiscreteScheduler();scheduler.set_timesteps(4)
    oracle=copy.deepcopy(scheduler)
    state=torch.randn(1,8,5,7)*scheduler.init_noise_sigma
    model=SimpleNamespace(_sushi_sd_noise_session=SDNoiseSession())
    seed_generator=torch.Generator().manual_seed(19)
    initialize_noise_tracking(model,scheduler,state,seed_generator)
    session=model._sushi_sd_noise_session
    generator=torch.Generator().manual_seed(23) if explicit else None
    owner_rng=generator if explicit else session.generator
    oracle_rng=SDNoiseTrajectory.clone_generator(owner_rng)
    t=scheduler.timesteps[0];prediction=torch.randn_like(state)
    scheduler.scale_model_input(state,t);oracle.scale_model_input(state,t)
    result=aligned_scheduler_step(model,scheduler,prediction,t,state,generator=generator)
    expected=oracle.step(prediction,t,state,generator=oracle_rng)
    torch.testing.assert_close(result.prev_sample,expected.prev_sample,rtol=0,atol=0)
    assert torch.equal(owner_rng.get_state(),oracle_rng.get_state())
    before=session.trajectory.state.clone();anchor=torch.randn_like(before)
    weight=torch.rand(1,1,5,7)
    blend_tracked_noise(model,anchor,weight)
    torch.testing.assert_close(session.trajectory.state,(1-weight)*before+weight*anchor,rtol=0,atol=0)


@pytest.mark.parametrize("name",["EulerDiscreteScheduler","EulerAncestralDiscreteScheduler","DDIMScheduler","DDPMScheduler"])
@pytest.mark.parametrize("explicit",[False,True])
def test_outpaint_jump_uses_same_new_noise_and_preserves_native_rng(name,explicit):
    from types import SimpleNamespace
    from core.inference.sd_codec_boundary import (SDNoiseSession,aligned_noise_jump,
        rewind_tracked_noise,composite_reference_noise)
    from core.inference.custom_sampling import _outpaint_resample_jump
    cls=getattr(diffusers,name)
    options={"clip_sample":False} if name in ("DDIMScheduler","DDPMScheduler") else {}
    scheduler=cls(num_train_timesteps=100,**options);scheduler.set_timesteps(8)
    noise=torch.randn(1,8,5,7);clean=torch.randn_like(noise)
    lo,hi=3,1
    state=scheduler.add_noise(clean,noise,scheduler.timesteps[lo:lo+1])
    raw_noise=scheduler.add_noise(torch.zeros_like(clean),noise,scheduler.timesteps[lo:lo+1])
    session=SDNoiseSession();session.initialize(scheduler,raw_noise,torch.Generator().manual_seed(19))
    model=SimpleNamespace(_sushi_sd_noise_session=session)
    generator=torch.Generator().manual_seed(23) if explicit else None
    actual_rng=generator if explicit else session.generator
    expected_rng=SDNoiseTrajectory.clone_generator(actual_rng)
    expected=_outpaint_resample_jump(scheduler,state,scheduler.timesteps,hi,lo,expected_rng)
    actual=aligned_noise_jump(model,scheduler,state,scheduler.timesteps,hi_index=hi,lo_index=lo,
        generator=generator,fallback_generator=torch.Generator().manual_seed(53),jump=_outpaint_resample_jump)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(actual_rng.get_state(),expected_rng.get_state())
    if name.startswith("Euler"):
        scheduler._step_index=hi;rewind_tracked_noise(model,hi)
    scaled=scheduler.scale_model_input(actual,scheduler.timesteps[hi])
    alpha,sigma,_=coefficients(scheduler,scheduler.timesteps[hi],scaled)
    exposure=session.exposure(scheduler.timesteps[hi],noise)
    torch.testing.assert_close(scaled,alpha*clean+sigma*exposure,rtol=2e-5,atol=2e-6)
    mask=torch.rand(1,1,5,7)
    mixed=composite_reference_noise(model,scheduler.timesteps[hi],noise,mask)
    torch.testing.assert_close(mixed,(1-mask)*noise+mask*exposure,rtol=0,atol=0)
