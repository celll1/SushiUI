"""Disk-backed live weights for failure-safe EMA checkpoint export."""

from pathlib import Path
import shutil
import tempfile

import torch


class EmaParameterStash:
    def __init__(self, parameters):
        self.parameters = tuple(parameters)
        self.root = None
        self.ready = False
        self.mutating = False

    def capture(self):
        total = sum(parameter.numel()*parameter.element_size() for _, parameter in self.parameters)
        directory = Path(tempfile.gettempdir())
        if shutil.disk_usage(directory).free < total + 1024**3:
            raise OSError("Insufficient temporary disk space for EMA live-weight rollback")
        self.root = Path(tempfile.mkdtemp(prefix="sushi-ema-stash-", dir=directory))
        try:
            for index, (_, parameter) in enumerate(self.parameters):
                value = parameter.detach().to(device="cpu", copy=True)
                torch.save(value, self.root / f"{index}.pt")
                del value
            self.ready = True
        except BaseException:
            self.close()
            raise

    @torch.no_grad()
    def restore(self):
        errors = []
        for index, (name, parameter) in enumerate(self.parameters):
            try:
                value = torch.load(self.root / f"{index}.pt", map_location="cpu", weights_only=True)
                parameter.copy_(value)
                del value
            except BaseException as exc:
                errors.append((name, exc))
        if errors:
            raise RuntimeError(f"EMA live-weight restoration failed; rollback files retained at {self.root}: "
                               f"{errors[0][0]}: {errors[0][1]}") from errors[0][1]
        self.mutating = False

    def close(self):
        if self.root is not None and not self.mutating:
            root = self.root.resolve()
            parent = Path(tempfile.gettempdir()).resolve()
            if root.parent != parent or not root.name.startswith("sushi-ema-stash-"):
                raise RuntimeError("Invalid EMA temporary stash ownership")
            shutil.rmtree(root)
            self.root = None
