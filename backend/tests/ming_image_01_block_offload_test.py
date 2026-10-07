"""Native dense/U-Net backward against the real mutable engine on CPU slots."""
from types import SimpleNamespace
import pytest
import torch
from core.models.ming_image_01.block_offload import setup_training_block_swap, move_component, execution_device
from core.models.ming_image_01.conditioning import MingConditioning
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.residual_unet import restore_topology
from core.models.ming_image_01.training import flow_loss
from ming_image_01_rewiring_test import config, spec_for


def cpu_transfers(monkeypatch):
    import core.memory_management.layer_offload_conductor as offload
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)) and torch.device(args[0]).type == "cuda":
            args = (torch.device("cpu"), *args[1:])
        if "device" in kwargs and torch.device(kwargs["device"]).type == "cuda":
            kwargs["device"] = torch.device("cpu")
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    class CPUStream:
        def __init__(self, **kwargs):
            pass
    monkeypatch.setattr(torch.cuda, "Stream", CPUStream)
    engine = offload.MutableLruTransferEngine
    def cpu_engine(**kwargs):
        kwargs.update(device=torch.device("cpu"), stream=None)
        return engine(**kwargs)
    monkeypatch.setattr(offload, "MutableLruTransferEngine", cpu_engine)


@pytest.mark.parametrize("padding", ["zero_masked", "learned"])
@pytest.mark.parametrize("rewired", [False, True])
def test_native_mutable_checkpoint_backward_writeback_and_residency(monkeypatch, padding, rewired):
    cpu_transfers(monkeypatch)
    torch.manual_seed(91)
    model = create_transformer(config(padding), device="cpu", dtype=torch.float32)
    if padding == "learned":
        torch.nn.init.constant_(model.x_pad_token, .02)
        torch.nn.init.constant_(model.cap_pad_token, .03)
    if rewired:
        restore_topology(model, spec_for(model))
    model.enable_gradient_checkpointing()
    trainer = SimpleNamespace(transformer=model, blocks_to_swap=2, layer_offload_conductor=None,
        gradient_checkpointing=True, device=torch.device("cuda"), use_pinned_memory=False,
        block_swap_ring_size=2)
    setup_training_block_swap(trainer)
    conductor = trainer.layer_offload_conductor
    assert model._layer_offload_conductor is conductor
    assert conductor.layers is model.layers and conductor.ring_size == 2
    before = {index: {dtype: value.clone() for dtype, value in planes.items()}
              for index, planes in conductor._masters.items()}
    identities = [id(parameter) for parameter in model.parameters()]
    handles = []
    for parameter in model.parameters():
        def update(parameter):
            with torch.no_grad():
                parameter.add_(parameter.grad, alpha=-.001)
            parameter.grad = None
        handles.append(parameter.register_post_accumulate_grad_hook(update))
    conductor.register_optimizer_hooks()
    frames = 3 if padding == "learned" else 1
    clean = torch.randn(1, 16, frames, 16, 16)
    condition = MingConditioning(torch.randn(1, 4, 8), torch.randn(1, 3, 24))
    weights = torch.ones(1, frames)
    if frames > 1:
        weights[:, 0] = 0
    result = flow_loss(model, clean=clean, noise=torch.randn_like(clean), sigma=torch.tensor([.3]),
        conditioning=condition, frame_weights=weights, reference_latents=[torch.randn(16, 1, 16, 16)])
    result.loss.backward()
    conductor.finish_backward()
    conductor.flush()
    assert not conductor.engine.active
    assert all(any(not torch.equal(value, before[index][dtype]) for dtype, value in planes.items())
               for index, planes in conductor._masters.items())
    pointers = {id(value): value.data_ptr() for index in conductor.swappable
                for value in model.layers[index].parameters()}
    move_component(model, "cuda")
    move_component(model, "cpu")
    assert identities == [id(parameter) for parameter in model.parameters()]
    assert pointers == {id(value): value.data_ptr() for index in conductor.swappable
                        for value in model.layers[index].parameters()}
    assert execution_device(model).type == "cpu"
    for handle in handles:
        handle.remove()
    conductor.cleanup()
    assert not torch.cuda.is_initialized()


def test_refuse_noncheckpointed_or_partial_swap_before_allocation():
    model = create_transformer(config("zero_masked"), device="cpu", dtype=torch.float32)
    trainer = SimpleNamespace(transformer=model, blocks_to_swap=2, layer_offload_conductor=None,
        gradient_checkpointing=False)
    with pytest.raises(ValueError, match="gradient checkpointing"):
        setup_training_block_swap(trainer)
    trainer.block_swap_h2d_only = True
    with pytest.raises(ValueError, match="CPU writeback"):
        setup_training_block_swap(trainer)
    trainer.block_swap_h2d_only = False
    trainer.gradient_checkpointing = True
    model._training_gradient_checkpointing_blocks = 1
    with pytest.raises(ValueError, match="every main"):
        setup_training_block_swap(trainer)


def test_managed_storage_never_receives_device_transfer(monkeypatch):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
    model.embedding = torch.nn.Linear(2, 2)
    calls, flushes = [], []
    model._layer_offload_conductor = SimpleNamespace(layers=model.layers, swappable=(1, 2),
        flush=lambda: flushes.append(True))
    protected = {value.data_ptr() for index in (1, 2) for value in model.layers[index].parameters()}
    def traced_to(value, *args, **kwargs):
        calls.append(value.data_ptr())
        return value
    monkeypatch.setattr(torch.Tensor, "to", traced_to)
    move_component(model, "cuda")
    assert flushes == [True]
    assert calls and not protected.intersection(calls)
    assert set(calls) == {value.data_ptr() for value in (*model.layers[0].parameters(), *model.embedding.parameters())}


def test_initial_resident_blocks_do_not_determine_outer_execution_device():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, device="meta"), torch.nn.Linear(2, 2)])
    model.embedding = torch.nn.Linear(2, 2)
    model._layer_offload_conductor = SimpleNamespace(layers=model.layers, swappable=(1,))
    assert execution_device(model).type == "cpu"
    model.extra = torch.nn.Linear(2, 2, device="meta")
    with pytest.raises(ValueError, match="one unmanaged execution device"):
        execution_device(model)


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_native_preview_and_bundle_restore_preserve_master_storage(tmp_path, monkeypatch, profile):
    from PIL import Image
    from ming_image_01_conditioning_test import write_tiny_package
    from core.training.full_parameter_trainer import FullParameterTrainer
    from core.training.arch.base_arch import SampleContext
    from core.models.ming_image_01.loader import load_components
    from api.arch_capabilities import TRAINING_UNSUPPORTED
    root = tmp_path / "native"
    root.mkdir()
    write_tiny_package(root, profile=profile)
    monkeypatch.setitem(TRAINING_UNSUPPORTED, "ming_image_01", {})
    owner = FullParameterTrainer(model_path=str(root), output_dir=str(tmp_path / "run"),
        device="cpu", weight_dtype="fp32", training_dtype="fp32", vae_dtype="fp32",
        mixed_precision=False, attention_backend="native",
        train_config={"ming_conditioner_groups": [], "ming_image_01_sample_layers": 2, "gradient_checkpointing": True})
    cpu_transfers(monkeypatch)
    owner.device = torch.device("cuda")
    owner.blocks_to_swap = 2
    setup_training_block_swap(owner)
    owner.device = torch.device("cpu")
    model = owner.transformer
    conductor = owner.layer_offload_conductor
    original = {key: value.clone() for key, value in model.state_dict().items()}
    pointers = {id(value): value.data_ptr() for index in conductor.swappable
                for value in model.layers[index].parameters()}
    reference = tmp_path / "reference.png"
    Image.new("RGBA", (64, 64), (100, 80, 60, 100)).save(reference)
    sample = owner.arch.sample(owner, SampleContext(prompt="word5", width=64, height=64,
        num_inference_steps=1, guidance_scale=1., seed=17,
        reference_image_path=str(reference) if profile == "layer_decompose" else None))
    assert len(sample._ming_result_group.values) == (3 if profile == "layer_decompose" else 1)
    manifest = owner.adapter.write_checkpoint(1, 0, tmp_path / "export")
    restored = load_components(manifest, dtype=torch.float32, conditioner_attention="eager")
    assert restored.transformer._residual_unet_spec == model._residual_unet_spec
    assert all(torch.equal(restored.transformer.state_dict()[key], value) for key, value in original.items())
    assert pointers == {id(value): value.data_ptr() for index in conductor.swappable
                        for value in model.layers[index].parameters()}
    assert not conductor.engine.active
    conductor.cleanup()
    assert not torch.cuda.is_initialized()
