"""Native Anima preparation for the shared teacher fitting transaction."""
from copy import deepcopy
import gc
import hashlib
from pathlib import Path

from PIL import Image, ImageOps
import torch

from .fitting import FitExample, FitPhase, InitializationCancelled, fit_teacher_modules
from .initialized_export import export_initialized_anima
from .spec import digest
from .provenance import RECIPE_FIELDS


def initialize_loaded_anima(components, source, spec, items, options, progress, cancelled):
    from core.attention import AttentionMode
    from core.models.anima.anima_attention import set_attention_backend
    from core.models.anima.anima_pipeline_ops import encode_prompt, vae_encode_image, sample_txt2img
    from core.models.anima.residual_unet import restore_topology
    model, te, vae = (components[key] for key in ("transformer", "text_encoder", "vae"))
    roots = (model, te, vae)
    devices = [(module, next(module.parameters()).device) for module in roots]
    modes = [(module, module.training) for root in roots for module in root.modules()]
    flags = [(parameter, parameter.requires_grad) for root in roots for parameter in root.parameters()]
    names = ("attn_mode", "_attn_backend", "_attn_mode", "gradient_checkpoint_mode",
             "_style_ctx", "_fbcache", "_fbcache_step")
    absent = object()
    attributes = [(module, name, module.__dict__.get(name, absent))
                  for module in model.modules() for name in names]
    latents, conditioning, pairs, phases = [], [], [], []
    fit_count = len(items)-options["holdout_count"]
    device = torch.device("cuda")

    def check_cancel():
        if cancelled():
            raise InitializationCancelled("Hybrid initialization cancelled")

    def report(stage, completed, total, units):
        check_cancel()
        progress(dict(stage=stage, completed=completed, total=total, units=units))

    def file_hash(path):
        result = hashlib.sha256()
        total, completed = Path(path).stat().st_size, 0
        with Path(path).open("rb") as stream:
            while chunk := stream.read(8*1024*1024):
                result.update(chunk)
                completed += len(chunk)
                report("source_verification", completed, total, "bytes")
        return result.hexdigest()

    def condition(index):
        return {name: value.to(device) for name, value in conditioning[index].items()}

    def predict(network, args, kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return network(*args, **kwargs)

    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(options["seed"])
            source_hash = file_hash(source)
            selection = torch.Generator().manual_seed(options["seed"])
            noise_rng = torch.Generator(device=device).manual_seed(options["seed"])
            for module in roots:
                module.to("cpu").eval().requires_grad_(False)
            torch.cuda.empty_cache()
            fingerprints = []
            vae.to(device)
            for index, item in enumerate(items):
                report("encoding_latents", index, len(items), "items")
                image_hash = hashlib.sha256()
                with Path(item["image_path"]).open("rb") as image_stream:
                    while chunk := image_stream.read(8*1024*1024):
                        check_cancel()
                        image_hash.update(chunk)
                fingerprint = image_hash.hexdigest()
                with Image.open(item["image_path"]) as image:
                    image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"),
                                         (options["resolution"], options["resolution"]),
                                         method=Image.Resampling.LANCZOS)
                    with torch.no_grad():
                        latent = vae_encode_image(vae, image, "cuda", next(vae.parameters()).dtype)
                    latents.append(latent.cpu())
                    del latent
                fingerprints.append(dict(image_sha256=fingerprint,
                    caption_sha256=hashlib.sha256(item["caption"].encode("utf-8")).hexdigest(),
                    role="fit" if index < fit_count else "held_out"))
            vae.to("cpu")
            torch.cuda.empty_cache()
            te.to(device)
            for index, caption in enumerate([item["caption"] for item in items]+[""]):
                report("encoding_captions", index, len(items)+1, "items")
                with torch.no_grad():
                    encoded = encode_prompt(te, components["tokenizer"], components["t5_tokenizer"], caption)
                conditioning.append({name: value.cpu() for name, value in encoded.items()})
                del encoded
            te.to("cpu")
            torch.cuda.empty_cache()
            model.to(device)
            set_attention_backend(model, options["attention_backend"], AttentionMode.TRAINING)
            if any(phase["kind"] == "native_trajectory" for phase in options["phases"]):
                scheduler = deepcopy(components["scheduler"])
                current = {}

                def capture(module, args, kwargs, output):
                    check_cancel()
                    index = current["index"] if kwargs["context"].data_ptr() == current["context"].data_ptr() else len(items)
                    pairs.append(dict(x=kwargs["x"].detach().cpu(), sigma=kwargs["timesteps"].detach().cpu(),
                                      target=output.detach().cpu(), conditioning_index=index))

                hook = model.register_forward_hook(capture, with_kwargs=True)
                try:
                    for index in range(fit_count):
                        report("teacher_trajectory", index, fit_count, "prompts")
                        cond, uncond = condition(index), condition(len(items))
                        current.update(index=index, context=cond["prompt_embeds"])
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            sample_txt2img(model, scheduler, cond, uncond,
                                height=options["resolution"], width=options["resolution"],
                                num_inference_steps=options["teacher_steps"], guidance_scale=options["teacher_cfg"],
                                generator=torch.Generator(device=device).manual_seed(options["seed"]+index),
                                device="cuda", dtype=torch.bfloat16,
                                step_callback=lambda *args: check_cancel(), spectrum_params={})
                        del cond, uncond
                finally:
                    hook.remove()
                current.clear()
                del scheduler
                if not pairs:
                    raise ValueError("Native teacher trajectory produced no velocity pairs")
            restore_topology(model, spec)
            model.enable_gradient_checkpointing()
            model.train()
            if hasattr(model, "llm_adapter"):
                model.llm_adapter.eval()

            def inputs(x, sigma, index):
                cond = condition(index)
                return dict(x=x, timesteps=sigma, context=cond["prompt_embeds"],
                    padding_mask=torch.zeros(x.shape[0],1,*x.shape[-2:],device=device,dtype=x.dtype),
                    target_input_ids=cond["t5_input_ids"], target_attention_mask=cond["t5_attn_mask"],
                    source_attention_mask=cond["source_mask"])

            def examples(kind):
                while True:
                    check_cancel()
                    if kind == "data_noising":
                        index = int(torch.randint(fit_count,(1,),generator=selection))
                        latent = latents[index].to(device=device,dtype=torch.bfloat16)
                        noise = torch.randn(latent.shape,device=device,dtype=latent.dtype,generator=noise_rng)
                        sigma = .01+.98*torch.rand(1,device=device,generator=noise_rng)
                        if float(torch.rand(1,generator=selection)) < options["conditioning_dropout"]:
                            index = len(items)
                        yield FitExample((),inputs((1-sigma)*latent+sigma*noise,sigma,index))
                    else:
                        pair = pairs[int(torch.randint(len(pairs),(1,),generator=selection))]
                        yield FitExample((),inputs(pair["x"].to(device),pair["sigma"].to(device),pair["conditioning_index"]),
                                         pair["target"].to(device))

            for phase in options["phases"]:
                phases.append(FitPhase(phase["kind"],phase["updates"],phase["learning_rate"],examples(phase["kind"])))

            def optimizer(parameters, lr):
                if options["optimizer"] == "adamw8bit":
                    import bitsandbytes as bnb
                    value = bnb.optim.AdamW8bit(parameters,lr=lr,weight_decay=options["weight_decay"])
                else:
                    value = torch.optim.AdamW(parameters,lr=lr,weight_decay=options["weight_decay"])
                if options["stochastic_rounding"]:
                    from core.training.optimizers.stochastic_rounding import attach_stochastic_rounding
                    if not attach_stochastic_rounding(value):
                        raise ValueError("Initialization optimizer cannot enable stochastic rounding")
                return value

            fit_teacher_modules(model,phases,predict,optimizer,source_sha256=source_hash,
                calibration_sha256=digest(dict(items=fingerprints,resolution=options["resolution"])),
                sample_count=len(items),seed=options["seed"],clip_norm=options["clip_norm"],
                recipe=dict(schema_version=1,settings={key:options[key] for key in RECIPE_FIELDS}),
                progress=lambda event: progress(dict(event,stage="fitting",units="updates")),cancelled=cancelled)
            report("saving",0,1,"models")
            return export_initialized_anima(model,source,options["output_name"],cancelled=cancelled)
    finally:
        for phase in phases:
            phase.examples.close()
        phases.clear()
        pairs.clear()
        latents.clear()
        conditioning.clear()
        restore_topology(model,None)
        for module,name,value in attributes:
            if value is absent:
                module.__dict__.pop(name,None)
            else:
                setattr(module,name,value)
        for parameter,requires_grad in flags:
            parameter.requires_grad_(requires_grad)
        for module,training in modes:
            module.training = training
        for module,original in devices:
            module.to(original)
        gc.collect()
        torch.cuda.empty_cache()
