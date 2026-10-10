"""Native Qwen preparation for the shared frozen-teacher fitting transaction."""
from copy import deepcopy
import gc
import hashlib
from pathlib import Path

from PIL import Image, ImageOps
import torch

from .anima_initialization import noising_input
from .fitting import FitExample, FitPhase, InitializationCancelled, fit_teacher_modules
from .initialized_export import export_initialized_qwen_package
from .provenance import RECIPE_FIELDS
from .spec import digest
from .resolution_schedule import (ResolutionSchedule, calibration_identity, exposure_noise,
                                  initialization_resolutions, resolution_items, resolution_recipe_fields,
                                  timestep_generator)


def qwen_inputs(latent, sigma, features, mask, height, width):
    if latent.ndim != 3 or latent.shape[1] != height*width or height%2 or width%2:
        raise ValueError("Qwen initialization requires an even native image grid")
    batch = latent.shape[0]
    return dict(hidden_states=latent,timestep=sigma,encoder_hidden_states=features,
        encoder_hidden_states_mask=mask,img_shapes=[[(1,height,width)]]*batch,
        img_mask=torch.cat((torch.zeros(features.shape[:2],device=latent.device,dtype=torch.bool),
                            torch.ones(batch,height*width//4,device=latent.device,dtype=torch.bool)),dim=1),
        return_dict=False)


def initialize_loaded_qwen(components, source, spec, items, options, progress, cancelled):
    from core.models.qwen_image_21.artifact import load_manifest
    from core.models.qwen_image_21.loader import build_pipeline
    from core.models.qwen_image_21.residual_unet import restore_topology
    from core.models.qwen_image_21.vendor import QwenImage21Pipeline
    from core.training.ops.qwen_image_21_ops import vae_encode
    from types import SimpleNamespace

    class TeacherPipeline(QwenImage21Pipeline):
        @property
        def _execution_device(self):
            return next(self.transformer.parameters()).device

    model,te,vae = (components[key] for key in ("transformer","text_encoder","vae"))
    roots = (model,te,vae)
    devices = [(root,next(root.parameters()).device) for root in roots]
    modes = [(module,module.training) for root in roots for module in root.modules()]
    flags = [(p,p.requires_grad) for root in roots for p in root.parameters()]
    checkpointing = model.gradient_checkpointing
    absent = object()
    attributes = [(model,key,model.__dict__.get(key,absent)) for key in
                  ("_gradient_checkpointing_func","_training_gradient_checkpointing_blocks")]
    processors = [(block.attn.processor,key,getattr(block.attn.processor,key,absent))
                  for block in model.transformer_blocks for key in
                  ("_attention_backend","_target_query_chunk_tokens")]
    manifest = load_manifest(str(source))
    device = torch.device("cuda")
    resolutions = initialization_resolutions(options)
    latents,conditioning,pairs,phases = {},[],{side: [] for side in resolutions},[]
    fit_count = len(items)-options["holdout_count"]
    pipe = build_pipeline({**components,"scheduler":deepcopy(components["scheduler"])})
    teacher_pipe = TeacherPipeline(scheduler=deepcopy(components["scheduler"]),vae=vae,
                                  text_encoder=te,processor=components["processor"],transformer=model)

    def check_cancel():
        if cancelled():
            raise InitializationCancelled("Qwen hybrid initialization cancelled")

    def report(stage,completed,total,units):
        check_cancel()
        progress(dict(stage=stage,completed=completed,total=total,units=units))

    def checksum(path,stage):
        value = hashlib.sha256()
        completed,total = 0,Path(path).stat().st_size
        with Path(path).open("rb") as stream:
            while chunk := stream.read(8*1024*1024):
                value.update(chunk)
                completed += len(chunk)
                report(stage,completed,total,"bytes")
        return value.hexdigest()

    def condition(index):
        return tuple(value.to(device) for value in conditioning[index])

    def inputs(value,sigma,index,side):
        features,mask = condition(index)
        return qwen_inputs(value,sigma,features,mask,side//16,side//16)

    def predict(network,args,kwargs):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            height,width = kwargs["img_shapes"][0][0][1:]
            return network(*args,**kwargs)[0][:,-height*width:]

    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(options["seed"])
            source_hash = checksum(manifest.transformer,"source_verification")
            selection = torch.Generator().manual_seed(options["seed"])
            noise_rng = torch.Generator(device=device).manual_seed(options["seed"])
            sigma_rng = timestep_generator(options,device)
            for root in roots:
                root.to("cpu").eval().requires_grad_(False)
            torch.cuda.empty_cache()
            fingerprints = []
            vae.to(device)
            owner = SimpleNamespace(qwen_image_21_pipeline=pipe,vae=vae,vae_dtype=next(vae.parameters()).dtype)
            for completed,(side,index,item) in enumerate(resolution_items(items,options)):
                report("encoding_latents",completed,len(items)*len(resolutions),"items")
                fingerprint = checksum(item["image_path"],"calibration_verification")
                with Image.open(item["image_path"]) as image:
                    image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGBA"),
                        (side,side),method=Image.Resampling.LANCZOS)
                    with torch.no_grad():
                        latent = vae_encode(owner,None,image=image,width=side,
                                            height=side,vae_device=device)
                    latents[side,index] = latent.cpu()
                    del latent
                if side == resolutions[0]:
                    fingerprints.append(dict(image_sha256=fingerprint,
                        caption_sha256=hashlib.sha256(item["caption"].encode()).hexdigest(),
                        role="fit" if index<fit_count else "held_out"))
            vae.to("cpu")
            torch.cuda.empty_cache()
            te.to(device)
            for index,caption in enumerate([item["caption"] for item in items]+[""]):
                report("encoding_captions",index,len(items)+1,"items")
                with torch.no_grad():
                    features,mask,_ = pipe.encode_prompt(prompt=caption,device=device)
                if mask is None:
                    mask = torch.ones(features.shape[:2],device=device,dtype=torch.bool)
                conditioning.append((features.cpu(),mask.cpu()))
                del features,mask
            te.to("cpu")
            torch.cuda.empty_cache()
            model.to(device)
            for block in model.transformer_blocks:
                block.attn.processor._attention_backend = options["attention_backend"]
                block.attn.processor._target_query_chunk_tokens = 0
            if any(phase["kind"] == "native_trajectory" for phase in options["phases"]):
                current = {}
                pointers = {}

                def capture(module,args,kwargs,output):
                    check_cancel()
                    features = kwargs["encoder_hidden_states"]
                    pointer = features.data_ptr()
                    if pointer not in pointers:
                        if torch.equal(features,current["features"]):
                            pointers[pointer] = current["index"]
                        elif torch.equal(features,current["negative"]):
                            pointers[pointer] = len(items)
                        else:
                            raise ValueError("Native Qwen teacher conditioning was unexpectedly transformed")
                    tokens = (current["side"]//16)**2
                    pairs[current["side"]].append(dict(x=kwargs["hidden_states"].detach().cpu(),
                        sigma=kwargs["timestep"].detach().cpu(),target=output[0][:,-tokens:].detach().cpu(),
                        conditioning_index=pointers[pointer]))

                hook = model.register_forward_hook(capture,with_kwargs=True)
                try:
                    for completed,(side,index,_) in enumerate(resolution_items(items[:fit_count],options)):
                        report("teacher_trajectory",completed,fit_count*len(resolutions),"prompts")
                        features,mask = condition(index)
                        negative,negative_mask = condition(len(items))
                        current.update(index=index,side=side,features=features,negative=negative)
                        pointers.clear()
                        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                            teacher_pipe(prompt_embeds=features,prompt_embeds_mask=mask,
                                negative_prompt_embeds=negative,negative_prompt_embeds_mask=negative_mask,
                                true_cfg_scale=options["teacher_cfg"],height=side,width=side,
                                output_resolution=side,num_inference_steps=options["teacher_steps"],
                                generator=torch.Generator(device=device).manual_seed(options["seed"]+index),
                                output_type="latent",use_kv_cache=False,before_step_callback=lambda *a:check_cancel())
                        del features,mask,negative,negative_mask
                finally:
                    hook.remove()
                current.clear()
                if any(not values for values in pairs.values()):
                    raise ValueError("Native Qwen teacher produced no trajectory targets")
            restore_topology(model,spec)
            model.enable_gradient_checkpointing()
            model._training_gradient_checkpointing_blocks = len(model.transformer_blocks)
            model.train()

            def examples(kind):
                schedule = ResolutionSchedule(options)
                while True:
                    check_cancel()
                    side = schedule.next()
                    if kind == "data_noising":
                        index = int(torch.randint(fit_count,(1,),generator=selection))
                        latent = latents[side,index].to(device=device,dtype=torch.bfloat16)
                        noise,sigma = exposure_noise(latent,noise_rng,sigma_rng)
                        value,sigma = noising_input(latent,noise,sigma)
                        if float(torch.rand(1,generator=selection))<options["conditioning_dropout"]:
                            index = len(items)
                        yield FitExample((),inputs(value,sigma,index,side))
                    else:
                        bucket = pairs[side]
                        pair = bucket[int(torch.randint(len(bucket),(1,),generator=selection))]
                        yield FitExample((),inputs(pair["x"].to(device),pair["sigma"].to(device),pair["conditioning_index"],side),
                                         pair["target"].to(device))

            for phase in options["phases"]:
                phases.append(FitPhase(phase["kind"],phase["updates"],phase["learning_rate"],examples(phase["kind"]),
                                       phase.get("skip_learning_rate_multiplier")))

            def optimizer(parameters,lr):
                if options["optimizer"] == "adamw8bit":
                    import bitsandbytes as bnb
                    value = bnb.optim.AdamW8bit(parameters,lr=lr,weight_decay=options["weight_decay"])
                else:
                    value = torch.optim.AdamW(parameters,lr=lr,weight_decay=options["weight_decay"])
                if options["stochastic_rounding"]:
                    from core.training.optimizers.stochastic_rounding import attach_stochastic_rounding
                    if not attach_stochastic_rounding(value):
                        raise ValueError("Qwen initialization optimizer cannot enable stochastic rounding")
                return value

            fit_teacher_modules(model,phases,predict,optimizer,source_sha256=source_hash,
                calibration_sha256=digest(calibration_identity(fingerprints,options)),
                sample_count=len(items),seed=options["seed"],clip_norm=options["clip_norm"],
                recipe=dict(schema_version=1,settings=dict({key:options[key] for key in RECIPE_FIELDS},
                                                         **resolution_recipe_fields(options))),
                progress=lambda event:progress(dict(event,stage="fitting",units="updates")),cancelled=cancelled)
            report("saving",0,1,"models")
            return export_initialized_qwen_package(model,source,Path(manifest.root).parent/options["output_name"],cancelled=cancelled)
    finally:
        for phase in phases:
            phase.examples.close()
        pairs.clear(); latents.clear(); conditioning.clear(); phases.clear()
        restore_topology(model,None)
        model.gradient_checkpointing = checkpointing
        for module,key,value in attributes+processors:
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
