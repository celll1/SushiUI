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
