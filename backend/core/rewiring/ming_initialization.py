"""Native Design/Layer teacher inputs and exclusive initialization residency."""
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from .fitting import FitExample, FitPhase, InitializationCancelled, fit_teacher_modules
from .initialized_export import export_initialized_ming_package
from .provenance import RECIPE_FIELDS
from .spec import digest


def ming_inputs(value,time,condition,reference=None):
    if value.ndim != 5 or time.shape != (value.shape[0],):
        raise ValueError("Ming initialization requires batch/channel/ordered-frame/height/width inputs")
    return dict(x=list(value.unbind()),t=time,cap_feats=list(condition.queries.unbind()),
                cap_feats_2=list(condition.direct.unbind()),ref_x=reference,return_dict=False)


def ming_prediction(network,args,kwargs):
    frames = kwargs["x"][0].shape[1]
    result = network(*args,**kwargs)[0]
    return torch.stack([value[:,:frames] for value in result])


def initialize_loaded_ming(components,source,spec,items,options,progress,cancelled,*,execution_device="cuda"):
    from core.models.ming_image_01.conditioning import MingConditioning
    from core.models.ming_image_01.flow import flow_training_pair
    from core.models.ming_image_01.prompting import prepare_inputs, encode_inputs
    from core.models.ming_image_01.residual_unet import restore_topology
    from core.models.ming_image_01.sampling import MingSampler, MingTransformerBridge
    from core.models.ming_image_01.training_data import encode_frames
    from core.models.ming_image_01.vendor.diffusion.pipeline import ImageGenerationPipeline
    from .ming_source import source_identity

    native = components["ming_native"]
    model,te,vae = native.transformer,native.conditioner,native.vae
    device = torch.device(execution_device)
    roots = (model,te,vae)
    devices = [(root,next(root.parameters()).device) for root in roots]
    modes = [(module,module.training) for root in roots for module in root.modules()]
    flags = [(p,p.requires_grad) for root in roots for p in root.parameters()]
    absent = object()
    attributes = [(model,key,model.__dict__.get(key,absent)) for key in
                  ("_gradient_checkpointing_func","_training_gradient_checkpointing_blocks")]
    attributes += [(module,"_attention_backend",module.__dict__.get("_attention_backend",absent))
                   for module in model.modules() if hasattr(module,"_attention_backend")]
    checkpointing = model.gradient_checkpointing
    fit_count = len(items)-options["holdout_count"]
    layer = native.artifact.profile == "layer_decompose"
    frames = 2 if layer else 1
    latents,conditioning,pixels,pairs,phases,references = [],[],[],[],[],[]
    side = options["resolution"]

    class TeacherPipeline(ImageGenerationPipeline):
        @property
        def _execution_device(self):
            return device

    private = replace(native,scheduler=deepcopy(native.scheduler))
    sampler = MingSampler(private)
    sampler.pipe = TeacherPipeline(scheduler=private.scheduler,vae=vae,
        transformer=MingTransformerBridge(model),text_encoder=None,tokenizer=None)
    sampler.pipe.set_progress_bar_config(disable=True)

    def check_cancel():
        if cancelled():
            raise InitializationCancelled("Ming hybrid initialization cancelled")

    def report(stage,completed,total,units):
        check_cancel()
        progress(dict(stage=stage,completed=completed,total=total,units=units))

    def sampler_cancelled():
        check_cancel()
        return False

    def checksum(path,stage):
        value = hashlib.sha256()
        completed,total = 0,Path(path).stat().st_size
        with Path(path).open("rb") as stream:
            while chunk := stream.read(8*1024*1024):
                value.update(chunk)
                completed += len(chunk)
                report(stage,completed,total,"bytes")
        return value.hexdigest()

    def condition(index,zero=False):
        value = conditioning[index]
        result = MingConditioning(value.queries.to(device=device,dtype=model.dtype),
                                  value.direct.to(device=device,dtype=model.dtype))
        return result.zero() if zero else result

    def predict(network,args,kwargs):
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type == "cuda"):
            return ming_prediction(network,args,kwargs)

    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
            torch.manual_seed(options["seed"])
            source_hash,_ = source_identity(native.artifact.components["transformer"],cancelled=cancelled,
                progress=lambda completed,total:report("source_verification",completed,total,"bytes"))
            selection = torch.Generator().manual_seed(options["seed"])
            noise_rng = torch.Generator(device=device).manual_seed(options["seed"])
            for root in roots:
                root.to("cpu").eval().requires_grad_(False)
            torch.cuda.empty_cache()
            fingerprints = []
            vae.to(device)
            for index,item in enumerate(items):
                report("encoding_latents",index,len(items),"items")
                fingerprint = checksum(item["image_path"],"calibration_verification")
                with Image.open(item["image_path"]) as original:
                    image = ImageOps.fit(ImageOps.exif_transpose(original).convert("RGBA"),
                                        (side,side),method=Image.Resampling.LANCZOS)
                    value = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/127.5-1
                    pixels.append(value.unsqueeze(0))
                    with torch.no_grad():
                        latent = encode_frames(vae,value[None,:,None].to(device=device,dtype=vae.dtype))
                    latents.append(latent.cpu())
                    references.append(latents[-1][0])
                    del latent,value
                fingerprints.append(dict(image_sha256=fingerprint,
                    caption_sha256=hashlib.sha256(item["caption"].encode()).hexdigest(),
                    role="fit" if index<fit_count else "held_out"))
            vae.to("cpu")
            torch.cuda.empty_cache()
            te.to(device)
            for index,item in enumerate(items):
                report("encoding_captions",index,len(items),"items")
                image = Image.fromarray(((pixels[index][0].permute(1,2,0)+1)*127.5).round().byte().numpy(),"RGBA")
                prepared = prepare_inputs(native.processor,[item["caption"]],references=[[image] if layer else []],
                    width=side,height=side,resolution=side,vae_references=False)
                with torch.no_grad():
                    result = encode_inputs(te,prepared,requires_grad=False)
                if not all(torch.isfinite(value).all() for value in (result.queries,result.direct)):
                    raise ValueError("Ming initializer received nonfinite native conditioning")
                conditioning.append(MingConditioning(result.queries.cpu(),result.direct.cpu()))
                del result,prepared,image
            te.to("cpu")
            torch.cuda.empty_cache()
            for module,key,_ in attributes:
                if key == "_attention_backend":
                    setattr(module,key,options["attention_backend"])
            model.to(device)
            # Layer noising uses native generated canvas/layers, never repeated RGB targets.
            if layer or any(phase["kind"] == "native_trajectory" for phase in options["phases"]):
                current = {}
                def activate(module,args,kwargs):
                    check_cancel()
                    if next(model.parameters()).device != device:
                        vae.to("cpu")
                        torch.cuda.empty_cache()
                        model.to(device)

                def capture(module,args,kwargs,output):
                    check_cancel()
                    if not any(phase["kind"] == "native_trajectory" for phase in options["phases"]):
                        return
                    inputs = kwargs["x"]
                    outputs = output[0]
                    for index,(value,target) in enumerate(zip(inputs,outputs)):
                        query = kwargs["cap_feats"][index]
                        direct = kwargs["cap_feats_2"][index]
                        expected = current["condition"]
                        zero = torch.count_nonzero(query).item() == torch.count_nonzero(direct).item() == 0
                        if not zero and (not torch.equal(query,expected.queries[0]) or not torch.equal(direct,expected.direct[0])):
                            raise ValueError("Native Ming teacher conditioning was unexpectedly transformed")
                        reference = kwargs.get("ref_x")
                        pairs.append(dict(x=value.detach().cpu().unsqueeze(0),
                            time=kwargs["t"][index:index+1].detach().cpu(),
                            target=target[:,:frames].detach().cpu().unsqueeze(0),
                            reference=None if reference is None else reference[index].detach().cpu(),
                            conditioning_index=current["index"],zero=zero))
                activation = model.register_forward_pre_hook(activate,with_kwargs=True)
                hook = model.register_forward_hook(capture,with_kwargs=True)
                try:
                    for index in range(fit_count):
                        report("teacher_trajectory",index,fit_count,"prompts")
                        current.update(index=index,condition=condition(index))
                        if layer:
                            model.to("cpu")
                            torch.cuda.empty_cache()
                            vae.to(device)
                        groups = sampler.run(task="layer-decompose" if layer else "text-to-image",
                            conditioning=current["condition"],width=side,height=side,steps=options["teacher_steps"],
                            cfg=options["teacher_cfg"],seed=options["seed"]+index,requested_layers=1,
                            reference_pixels=pixels[index].to(device) if layer else None,
                            output_type="latent",cancelled=sampler_cancelled)
                        if layer:
                            latents[index] = torch.stack(groups[0].values,dim=1).unsqueeze(0).cpu()
                        del groups
                finally:
                    hook.remove()
                    activation.remove()
                current.clear()
            model.to(device)
            vae.to("cpu")
            restore_topology(model,spec)
            model.enable_gradient_checkpointing()
            model._training_gradient_checkpointing_blocks = len(model.layers)
            model.train()

            def examples(kind):
                while True:
                    check_cancel()
                    if kind == "data_noising":
                        index = int(torch.randint(fit_count,(1,),generator=selection))
                        clean = latents[index].to(device=device,dtype=model.dtype)
                        noise = torch.randn(clean.shape,device=device,dtype=clean.dtype,generator=noise_rng)
                        sigma = .01+.98*torch.rand(1,device=device,generator=noise_rng)
                        value,_,time = flow_training_pair(clean,noise,sigma)
                        zero = float(torch.rand(1,generator=selection))<options["conditioning_dropout"]
                        reference = [references[index].to(device=device,dtype=model.dtype)] if layer else None
                        yield FitExample((),ming_inputs(value.to(model.dtype),time,condition(index,zero),reference))
                    else:
                        pair = pairs[int(torch.randint(len(pairs),(1,),generator=selection))]
                        reference = None if pair["reference"] is None else [pair["reference"].to(device=device,dtype=model.dtype)]
                        yield FitExample((),ming_inputs(pair["x"].to(device),pair["time"].to(device),
                            condition(pair["conditioning_index"],pair["zero"]),reference),pair["target"].to(device))

            for phase in options["phases"]:
                phases.append(FitPhase(phase["kind"],phase["updates"],phase["learning_rate"],examples(phase["kind"])))
            def optimizer(parameters,lr):
                if options["optimizer"] == "adamw8bit":
                    import bitsandbytes as bnb
                    value = bnb.optim.AdamW8bit(parameters,lr=lr,weight_decay=options["weight_decay"])
                else:
                    value = torch.optim.AdamW(parameters,lr=lr,weight_decay=options["weight_decay"])
                if options["stochastic_rounding"]:
                    from core.training.optimizers.stochastic_rounding import attach_stochastic_rounding
                    if not attach_stochastic_rounding(value):
                        raise ValueError("Ming initializer cannot enable stochastic rounding")
                return value
            fit_teacher_modules(model,phases,predict,optimizer,source_sha256=source_hash,
                calibration_sha256=digest(dict(items=fingerprints,resolution=side)),sample_count=len(items),
                seed=options["seed"],clip_norm=options["clip_norm"],
                recipe=dict(schema_version=1,settings={key:options[key] for key in RECIPE_FIELDS}),
                progress=lambda event:progress(dict(event,stage="fitting",units="updates")),cancelled=cancelled)
            report("saving",0,1,"models")
            return export_initialized_ming_package(model,source,native.artifact.root.parent/options["output_name"],cancelled=cancelled)
    finally:
        for phase in phases:
            phase.examples.close()
        pairs.clear(); latents.clear(); conditioning.clear(); pixels.clear(); phases.clear(); references.clear()
        restore_topology(model,None)
        model.gradient_checkpointing = checkpointing
        for module,key,value in attributes:
            if value is absent:
                module.__dict__.pop(key,None)
            else:
                setattr(module,key,value)
        for parameter,requires_grad in flags:
            parameter.requires_grad_(requires_grad)
        for module,training in modes:
            module.training = training
        for module,original in devices:
            module.to(original)
        gc.collect()
        torch.cuda.empty_cache()
