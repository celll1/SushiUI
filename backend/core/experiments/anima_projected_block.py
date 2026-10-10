"""Private head-aligned Anima block student; outer stream width is retained."""
import torch
from torch import nn

from core.models.anima.anima_models import Block


def _indices(values, size, label):
    if (not isinstance(values, (list, tuple)) or not values
            or any(type(x) is not int or not 0 <= x < size for x in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"Invalid {label} selection")
    return list(values)


class ProjectedAnimaBlock(nn.Module):
    """Apply a narrower native block to selected channels; retain the complement."""
    def __init__(self, student, channels, native_width, trainable_bridges=False):
        super().__init__()
        self.student = student
        self.x_dim = native_width
        self.register_buffer('channels', torch.tensor(channels, dtype=torch.long))
        self.register_buffer('adaln_channels', torch.cat([
            self.channels + offset*native_width for offset in range(3)]))
        self.encoder = self.decoder = None
        if trainable_bridges:
            reference = next(student.parameters())
            encoder = torch.zeros(len(channels), native_width, device=reference.device, dtype=reference.dtype)
            decoder = torch.zeros(native_width, len(channels), device=reference.device, dtype=reference.dtype)
            coordinates = torch.arange(len(channels), device=reference.device)
            selected = self.channels.to(reference.device)
            encoder[coordinates, selected] = 1
            decoder[selected, coordinates] = 1
            self.encoder, self.decoder = nn.Parameter(encoder), nn.Parameter(decoder)

    @property
    def gradient_checkpoint_mode(self):
        return self.student.gradient_checkpoint_mode

    @gradient_checkpoint_mode.setter
    def gradient_checkpoint_mode(self, value):
        self.student.gradient_checkpoint_mode = value

    def forward(self, x, embedding, context, attention, use_fp32=False,
                rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None, extra_per_block_pos_emb=None):
        if (x.shape[-1] != self.x_dim or embedding.shape[-1] != self.x_dim
                or extra_per_block_pos_emb is not None):
            raise ValueError("Projected block requires native features/conditioning without extra positions")
        if self.encoder is None:
            selected = x.index_select(-1, self.channels)
            embedding = embedding.index_select(-1, self.channels)
        else:
            selected = torch.nn.functional.linear(x.to(self.encoder.dtype), self.encoder)
            embedding = torch.nn.functional.linear(embedding.to(self.encoder.dtype), self.encoder)
        if self.student.use_adaln_lora:
            if adaln_lora_B_T_3D is None or adaln_lora_B_T_3D.shape[-1] != 3*self.x_dim:
                raise ValueError("Projected block requires native shift/scale/gate conditioning")
            if self.encoder is None:
                adaln_lora_B_T_3D = adaln_lora_B_T_3D.index_select(-1, self.adaln_channels)
            else:
                adaln_lora_B_T_3D = torch.nn.functional.linear(
                    adaln_lora_B_T_3D.unflatten(-1, (3, self.x_dim)).to(self.encoder.dtype), self.encoder).flatten(-2)
        output = self.student(selected, embedding, context, attention, use_fp32,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        if self.decoder is None:
            return x.to(output.dtype).index_copy(-1, self.channels, output)
        with torch.autocast(device_type=x.device.type, enabled=False):
            correction = torch.nn.functional.linear(output.float()-selected.float(), self.decoder.float())
        return (x.float()+correction).to(output.dtype)


def project_block(source, heads, *, mlp_neurons=None, trainable_bridges=False):
    """Copy selected heads/features, MLP neurons and all three modulation slices."""
    if not isinstance(source, Block):
        raise ValueError("Projection requires a native unquantized Anima block")
    if type(trainable_bridges) is not bool:
        raise ValueError("Trainable projection bridges must be an explicit boolean")
    if (source.self_attn.n_heads != source.cross_attn.n_heads
            or source.x_dim != source.self_attn.n_heads*source.self_attn.head_dim
            or source.self_attn.head_dim != source.cross_attn.head_dim):
        raise ValueError("Projection requires matching head-aligned native attention")
    if any(type(module) is not nn.Linear or not isinstance(getattr(module, 'weight', None), nn.Parameter) for module in source.modules()
           if hasattr(module, 'in_features') and hasattr(module, 'out_features')):
        raise ValueError("Projection refuses quantized or custom Linear layers")
    if source.self_attn._style_ctx is not None:
        raise ValueError("Projection refuses an active reference-style context")
    if getattr(source, '_offload_recompute_callback', None) is not None:
        raise ValueError("Projection refuses a live block-offload callback")
    heads = _indices(heads, source.self_attn.n_heads, 'head')
    channels = [head*source.self_attn.head_dim + offset for head in heads for offset in range(source.self_attn.head_dim)]
    width = len(channels)
    native_hidden = source.mlp.layer1.out_features
    hidden = native_hidden*width//source.x_dim
    if native_hidden*width % source.x_dim:
        raise ValueError("Projected MLP width must be integral")
    if mlp_neurons is None:
        mlp_neurons = [index*native_hidden//hidden for index in range(hidden)]
    neurons = _indices(mlp_neurons, native_hidden, 'MLP neuron')
    if len(neurons) != hidden:
        raise ValueError("MLP selection must match the projected expansion ratio")
    lora_dim = source.adaln_modulation_mlp[1].out_features if source.use_adaln_lora else 256
    reference = next(source.parameters())
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        student = Block(width, source.cross_attn.context_dim, len(heads),
            mlp_ratio=native_hidden/source.x_dim, use_adaln_lora=source.use_adaln_lora,
            adaln_lora_dim=lora_dim)
    student.to(device=reference.device, dtype=reference.dtype)
    indices = torch.tensor(channels, dtype=torch.long, device=reference.device)
    mlp_indices = torch.tensor(neurons, dtype=torch.long, device=reference.device)
    modulation = torch.cat([indices+part*source.x_dim for part in range(3)])
    def copy(destination, original, rows=None, columns=None):
        if rows is not None:
            original = original.index_select(0, rows)
        if columns is not None:
            original = original.index_select(1, columns)
        if destination.shape != original.shape:
            raise ValueError("Projected tensor layout differs from its native source")
        destination.copy_(original)
    with torch.no_grad():
        for name in ('self_attn', 'cross_attn'):
            original, target = getattr(source, name), getattr(student, name)
            copy(target.q_proj.weight, original.q_proj.weight, indices, indices)
            for projection in ('k_proj', 'v_proj'):
                copy(getattr(target, projection).weight, getattr(original, projection).weight,
                    indices, indices if name == 'self_attn' else None)
            copy(target.output_proj.weight, original.output_proj.weight, indices, indices)
            target.q_norm.load_state_dict(original.q_norm.state_dict(), strict=True)
            target.k_norm.load_state_dict(original.k_norm.state_dict(), strict=True)
            target.q_norm.eps, target.k_norm.eps = original.q_norm.eps, original.k_norm.eps
        copy(student.mlp.layer1.weight, source.mlp.layer1.weight, mlp_indices, indices)
        copy(student.mlp.layer2.weight, source.mlp.layer2.weight, indices, mlp_indices)
        for name in ('adaln_modulation_self_attn', 'adaln_modulation_cross_attn', 'adaln_modulation_mlp'):
            original, target = getattr(source, name), getattr(student, name)
            if source.use_adaln_lora:
                copy(target[1].weight, original[1].weight, columns=indices)
                copy(target[2].weight, original[2].weight, rows=modulation)
            else:
                copy(target[1].weight, original[1].weight, modulation, indices)
        for name in ('layer_norm_self_attn', 'layer_norm_cross_attn', 'layer_norm_mlp'):
            getattr(student, name).eps = getattr(source, name).eps
    student.gradient_checkpoint_mode = source.gradient_checkpoint_mode
    student.train(source.training)
    result = ProjectedAnimaBlock(student, channels, source.x_dim, trainable_bridges).to(device=reference.device)
    result.train(source.training)
    result.projection_manifest = dict(native_width=source.x_dim, student_width=width,
        head_dim=source.self_attn.head_dim, heads=heads, mlp_neurons=neurons,
        shared_adaln_rank=lora_dim if source.use_adaln_lora else None,
        outer_stream_width=source.x_dim, trainable_bridges=trainable_bridges, private=True)
    return result
