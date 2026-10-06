"""Target-native audio swap setup and bounded paired waveform calibration."""

from dataclasses import replace
import hashlib
from math import ceil

import torch
from torch.nn import functional as F

from core.models.components.audio_latent_adapter import AudioCodecContract, AudioAlignmentPair, PairedAudioCalibration
from core.models.common.vae_source import content_hash_for_state_dict

PENDING = "_acestep_vae_calibration_source"


def calibration_waveform(path, seconds, contract):
    import soundfile
    from core.pipeline_backends.acestep import AceStepMixin
    with soundfile.SoundFile(path) as audio:
        rate = audio.samplerate
        value = torch.from_numpy(audio.read(ceil(seconds * rate), dtype="float32", always_2d=True).T.copy())
    value = AceStepMixin._acestep_normalize_stereo_48k(value, rate)
    value = value[:, :ceil(seconds * contract.sample_rate)].contiguous()
    if value.ndim != 2 or value.shape[0] != contract.waveform_channels or value.shape[1] < 1 or not torch.isfinite(value).all():
        raise ValueError("ACE calibration waveform geometry or values differ")
    return value


def configure_training_swap(trainer):
    from core.training.vae_swap import resolve_vae_swap_source, check_swap_method
    from core.training.ops.training_method import resolve_training_method
    from core.models.common.vae_source import resolve_vae_source, ResolvedVAE
    from core.training.arch.acestep import AceStepArchHandler
    handler = AceStepArchHandler(trainer)
    source = resolve_vae_swap_source(trainer.config)
    check_swap_method(source, resolve_training_method(trainer))
    pair = getattr(trainer.transformer.decoder, "_audio_latent_alignment", None)
    if pair is not None:
        trainer.vae_identity = ResolvedVAE(source="model:"+str(trainer.model_path), form="model", family="custom",
            latent_channels=pair.target.latent_channels, scale_factor=1, scale_temporal=pair.target.hop_length,
            ndim=3, norm="identity", norm_pack=1, vae_class="AutoencoderOobleck",
            config=dict(trainer.vae.config, _class_name="AutoencoderOobleck"),
            content_hash=content_hash_for_state_dict(trainer.vae.state_dict()), provenance="bundled ACE audio codec",
            locator=None, struct_native=False, identity_native=False, scaling_factor=None, shift_factor=None,
            path=str(trainer.model_path), prefix="vae.")
        trainer.base_vae_identity = trainer.vae_identity
        trainer.wiring = handler.resolve_wiring(trainer)
    if not source:
        return
    if not getattr(trainer, "train_unet", True) or trainer.config.get("bundle_vae") is False:
        raise ValueError("ACE audio swap requires train_unet=true and bundled codec export")
    resolved = resolve_vae_source(source, arch="acestep")
    apply_swap(trainer, resolved, handler=handler)
    if hasattr(trainer, PENDING):
        trainer.acestep_source_vae = getattr(trainer, PENDING)


def apply_swap(trainer, resolved, *, handler, module=None):
    from core.models.components.latent_io import ResizeReport
    from core.training.adapters.base_adapter import reject_quantized_base
    from api.param_defaults import TRAINING_DEFAULTS
    from .vae_boundary import bind_encoding_contract, install_decoder_alignment

    compatible, reason = handler.check_vae_compatibility(resolved.facts(), trainer=trainer)
    if not compatible:
        raise ValueError(reason)
    model = trainer.transformer
    pair = getattr(model.decoder, "_audio_latent_alignment", None)
    channels = model.decoder.config.audio_acoustic_hidden_dim
    if pair is not None:
        candidate = module or resolved.load_module(torch_dtype=trainer.vae_dtype)
        if (AudioCodecContract.from_oobleck(candidate) != pair.target
                or dict(candidate.config) != dict(trainer.vae.config)
                or content_hash_for_state_dict(candidate.state_dict()) != content_hash_for_state_dict(trainer.vae.state_dict())):
            raise ValueError("Changing an aligned ACE codec requires a fresh base checkpoint")
        if not pair.ready:
            raise ValueError("Restored ACE audio swap has no calibration")
        trainer.vae_identity = replace(resolved, state_dict=None)
        trainer.wiring = handler.resolve_wiring(trainer)
        return ResizeReport((), channels, channels, channels, 0, 0)
    if module is not None:
        raise ValueError("ACE swapped checkpoints require their saved audio alignment")
    mode = trainer.config.get("vae_swap_new_channel_init", TRAINING_DEFAULTS["vae_swap_new_channel_init"])
    if mode != "auto":
        raise ValueError("ACE audio swap uses auto paired_temporal initialization")
    reject_quantized_base(model, model_label="ACE audio VAE swap")
    old = trainer.vae
    source = AudioCodecContract.from_oobleck(old)
    target = resolved.load_module(torch_dtype=trainer.vae_dtype).requires_grad_(False).eval()
    target_contract = AudioCodecContract.from_oobleck(target)
    source.check_pair(target_contract)
    if source == target_contract and content_hash_for_state_dict(old.state_dict()) == content_hash_for_state_dict(target.state_dict()):
        return ResizeReport((), channels, channels, channels, 0, 0)
    with torch.random.fork_rng(devices=[]):
        pair = AudioAlignmentPair(source, target_contract)
    install_decoder_alignment(model, pair)
    bind_encoding_contract(target, pair)
    trainer.vae = target
    trainer.vae_latent_channels = target_contract.latent_channels
    trainer.acestep_sample_rate = target_contract.sample_rate
    trainer.acestep_latent_frame_rate = target_contract.sample_rate / target_contract.hop_length
    trainer.vae_identity = replace(resolved, state_dict=None, identity_native=False, struct_native=False)
    trainer.wiring = handler.resolve_wiring(trainer)
    if getattr(model, "_residual_unet_spec", None) is not None:
        trainer.config["dit_rewiring_spec"] = model._residual_unet_spec
    setattr(trainer, PENDING, old.to("cpu").requires_grad_(False).eval())
    return ResizeReport(("decoder.proj_in", "decoder.proj_out", "audio_alignment"),
        source.latent_channels, source.latent_channels, pair.backbone_channels,
        0, sum(p.numel() for p in pair.parameters()))


def calibrate(trainer, datasets):
    old = getattr(trainer, PENDING, None)
    if old is None:
        return
    from api.param_defaults import TRAINING_DEFAULTS
    from core.training.optimizer_residency import encoder_refill_residency, park_optimizer_state
    from core.training.base_trainer import all_optimizers
    count = trainer.config.get("vae_swap_calibration_samples", TRAINING_DEFAULTS["vae_swap_calibration_samples"])
    if type(count) is not int or not 2 <= count <= 256:
        raise ValueError("Invalid ACE audio calibration sample count")
    paths = list(dict.fromkeys(item["audio_path"] for dataset in datasets for item in dataset.items if item.get("audio_path")))
    if not paths:
        raise ValueError("ACE VAE calibration requires training audio clips")
    paths = [paths[i] for i in torch.linspace(0, len(paths)-1, min(count, len(paths))).long().tolist()]
    pair = trainer.transformer.decoder._audio_latent_alignment
    frames = trainer.transformer.config.timbre_fix_frame
    if type(frames) is not int or frames < 1:
        raise ValueError("ACE calibration requires a positive native timbre window")
    seconds = frames * pair.source.hop_length / pair.source.sample_rate
    target = trainer.vae
    source_identity = dict(config=dict(old.config), hash=content_hash_for_state_dict(old.state_dict()),
                           dtype=str(next(old.parameters()).dtype))
    fit = PairedAudioCalibration(pair, ridge=1e-2)
    text = getattr(trainer, "text_encoder", None)
    text_device = next(text.parameters()).device if text is not None else None
    parameters = tuple(text.parameters()) if text is not None else ()
    callback = getattr(trainer, "_vae_alignment_progress_callback", None)
    waveform_hashes = {}

    def waveform(path, contract, dtype):
        value = calibration_waveform(path, seconds, contract)
        digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        if waveform_hashes.setdefault(path, digest) != digest:
            raise ValueError("ACE calibration waveform changed between codec passes")
        return F.pad(value[None], (0, (-value.shape[1]) % contract.hop_length)).to(device=trainer.device, dtype=dtype)

    with encoder_refill_residency(trainer, text=False, vae=True), park_optimizer_state(all_optimizers(trainer), parameters):
        if text is not None:
            text.to("cpu")
        try:
            paired = []
            target.to("cpu")
            old.to(trainer.device)
            with torch.no_grad():
                for i, path in enumerate(paths):
                    trainer._check_stop_requested()
                    paired.append(old.encode(waveform(path, pair.source, next(old.parameters()).dtype)).latent_dist.mode().cpu())
                    if callback:
                        callback(phase="latent_cache", step=i+1, total=2*len(paths))
            old.to("cpu")
            if torch.device(trainer.device).type == "cuda":
                torch.cuda.empty_cache()
            target.to(trainer.device)
            with torch.no_grad():
                for i, (path, teacher) in enumerate(zip(paths, paired)):
                    trainer._check_stop_requested()
                    student = target.encode(waveform(path, pair.target, next(target.parameters()).dtype)).latent_dist.mode().cpu()
                    fit.add(teacher, student)
                    if callback:
                        callback(phase="latent_cache", step=len(paths)+i+1, total=2*len(paths))
            fit.finish()
            trainer.transformer._audio_source_codec_identity = source_identity
            delattr(trainer, PENDING)
        finally:
            old.to("cpu")
            target.to("cpu")
            if text is not None:
                text.to(text_device)
