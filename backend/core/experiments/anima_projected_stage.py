"""Private projected-block stage teacher and explicit experimental payload."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from threading import Lock

import torch
from torch import nn

from core.rewiring.hybrid_spec import validate_hybrid_spec
from .anima_projected_block import project_block

_probe_lock = Lock()


def native_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not name.startswith('rewiring_modules.'):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ProjectedStageTeacher:
    def __init__(self, model, declaration, originals, students, fingerprint):
        self.model, self.declaration = model, declaration
        self.originals, self.students = originals, students
        self.fingerprint = fingerprint
        self.parameters = ()
        self.active = False

    def _check(self):
        if (not self.active or self.model is None
                or getattr(self.model, '_projected_stage_owner', None) is not self
                or getattr(self.model, '_hybrid_teacher_owner', None) is not self):
            raise RuntimeError('Projected stage teacher is outside its active context')
        if self.model._residual_unet_spec != self.declaration or any(
                self.model.blocks[index] is not student for index,student in self.students.items()):
            raise RuntimeError('Projected stage requires exclusive block/topology ownership')

    def __call__(self, *args, **kwargs):
        self._check()
        modules = {id(module):module for module in self.model.modules()}
        for block in self.originals.values():
            modules.update({id(module):module for module in block.modules()})
        modes = [(module, module.training) for module in modules.values()]
        try:
            for index, block in self.originals.items():
                self.model.blocks[index] = block
            self.model._residual_unet_spec = None
            self.model.eval()
            with torch.no_grad():
                return self.model(*args, **kwargs)
        finally:
            for index, block in self.students.items():
                self.model.blocks[index] = block
            self.model._residual_unet_spec = self.declaration
            for module, training in modes:
                module.training = training

    def payload(self):
        self._check()
        def snapshot(module):
            return {name:value.detach().cpu().clone() for name,value in module.state_dict().items()}
        return dict(format='anima_projected_stage_probe', version=1,
            native_sha256=self.fingerprint, declaration=deepcopy(self.declaration),
            manifests={str(index):deepcopy(block.projection_manifest) for index,block in self.students.items()},
            blocks={str(index):snapshot(block) for index,block in self.students.items()},
            structural=snapshot(self.model.rewiring_modules))

    def load_payload(self, payload):
        self._check()
        expected = {'format', 'version', 'native_sha256', 'declaration', 'manifests', 'blocks', 'structural'}
        if (not isinstance(payload, dict) or set(payload) != expected
                or payload['format'] != 'anima_projected_stage_probe' or type(payload['version']) is not int
                or payload['version'] != 1 or payload['native_sha256'] != self.fingerprint
                or payload['declaration'] != self.declaration
                or payload['manifests'] != {str(i):b.projection_manifest for i,b in self.students.items()}
                or not isinstance(payload['blocks'], dict) or set(payload['blocks']) != {str(i) for i in self.students}):
            raise ValueError('Projected stage payload identity/layout mismatch')
        assignments = [(self.model.rewiring_modules, payload['structural'])]
        assignments.extend((block, payload['blocks'][str(index)]) for index,block in self.students.items())
        for module, state in assignments:
            reference = module.state_dict()
            if not isinstance(state, dict) or set(state) != set(reference):
                raise ValueError('Projected stage payload tensor keys mismatch')
            for name, value in state.items():
                target = reference[name]
                if (not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype
                        or not torch.isfinite(value).all()
                        or not value.is_floating_point() and not torch.equal(value.cpu(), target.cpu())):
                    raise ValueError('Projected stage payload tensor layout/value mismatch')
        for module, state in assignments:
            module.load_state_dict(state, strict=True)


@contextmanager
def projected_stage_probe(model, heads_by_block):
    """Borrow frozen originals; restore flags/modes/RNG/topology on every exit."""
    declaration = validate_hybrid_spec(getattr(model, '_residual_unet_spec', None))
    if (declaration['architecture'] != 'anima' or not isinstance(getattr(model, 'blocks', None), nn.ModuleList)
            or declaration['streams'] != [dict(name='main', block_count=len(model.blocks))]
            or not isinstance(heads_by_block, dict) or not heads_by_block):
        raise ValueError('Projected stage probe requires an Anima main-stream hybrid')
    if (getattr(model, '_hybrid_teacher_owner', None) is not None
            or getattr(model, '_projected_stage_owner', None) is not None
            or getattr(model, '_block_offloader', None) is not None
            or getattr(model, '_hybrid_initialization', None) is not None):
        raise ValueError('Projected stage requires a fresh, exclusively owned resident model')
    modules = getattr(model, 'rewiring_modules', None)
    if modules is None or modules.declaration != declaration:
        raise ValueError('Projected stage requires matching structural modules')
    for index in heads_by_block:
        if (type(index) is not int or not 0 <= index < len(model.blocks)
                or not any(region['start_block'] <= index < region['end_block'] for region in declaration['regions'])
                or model.blocks[index].x_dim != declaration['stream_widths']['main']):
            raise ValueError('Projected blocks must belong to a declared coarse region')
    original_parameters = tuple(model.parameters())
    if any(parameter.grad is not None for parameter in original_parameters):
        raise ValueError('Projected stage must not discard existing training gradients')
    learned_ids = {id(parameter) for parameter in modules.parameters()}
    if any(id(p) in learned_ids and not name.startswith('rewiring_modules.')
           for name,p in model.named_parameters(remove_duplicate=False)):
        raise ValueError('Structural parameters alias the native model')
    if not _probe_lock.acquire(blocking=False):
        raise RuntimeError('Only one projected stage probe may execute per process')
    originals = {index:model.blocks[index] for index in sorted(heads_by_block)}
    flags = [(p,p.requires_grad) for p in original_parameters]
    modes = [(module,module.training) for module in model.modules()]
    topology = model._residual_unet_spec
    devices = sorted({value.device.index for value in (*original_parameters, *model.buffers()) if value.is_cuda})
    teacher = None
    try:
        with torch.random.fork_rng(devices=devices):
            students = {index:project_block(block, heads_by_block[index], trainable_bridges=True)
                        for index,block in originals.items()}
            fingerprint = native_fingerprint(model)
            teacher = ProjectedStageTeacher(model, declaration, originals, students, fingerprint)
            try:
                for parameter in original_parameters:
                    parameter.requires_grad_(id(parameter) in learned_ids)
                for index,block in students.items():
                    model.blocks[index] = block
                teacher.parameters = tuple(p for p in model.parameters() if p.requires_grad)
                teacher.active = True
                model._projected_stage_owner = teacher
                model._hybrid_teacher_owner = teacher
                yield teacher
            finally:
                for index,block in originals.items():
                    model.blocks[index] = block
                model._residual_unet_spec = topology
                model.__dict__.pop('_projected_stage_owner', None)
                model.__dict__.pop('_hybrid_teacher_owner', None)
                for parameter in teacher.parameters:
                    parameter.grad = None
                for parameter, requires_grad in flags:
                    parameter.grad = None
                    parameter.requires_grad_(requires_grad)
                for module, training in modes:
                    module.training = training
                teacher.active = False
                teacher.model = None
                teacher.originals = {}
                teacher.students = {}
                teacher.parameters = ()
    finally:
        _probe_lock.release()
