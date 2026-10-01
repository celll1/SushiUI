"""Request-owned Ming images and native ordered layer roles."""

from copy import deepcopy
from dataclasses import dataclass

from PIL import Image


@dataclass
class MingResultGroup:
    values: list
    roles: list[str]
    profile: str
    requested_layers: int
    target_frames: int
    seed: int
    width: int
    height: int
    output_type: str


@dataclass(frozen=True)
class MingGenerationOutput:
    groups: tuple[MingResultGroup, ...]
    model_info: dict | None = None

    def __post_init__(self):
        if self.model_info is not None:
            object.__setattr__(self, "model_info", deepcopy(self.model_info))

    def validate(self):
        if not self.groups:
            raise ValueError("Ming output must contain at least one result group")
        for group in self.groups:
            if type(group.requested_layers) is not int or group.requested_layers < 1:
                raise ValueError("Ming layer count must be a positive integer")
            if group.profile == "generation_edit":
                roles = ["image"]
            elif group.profile == "layer_decompose":
                roles = ["auxiliary_canvas"] + [f"layer_{i}" for i in range(1, group.requested_layers + 1)]
            else:
                raise ValueError("Unknown Ming output profile")
            if group.roles != roles or group.target_frames != len(roles) or len(group.values) != len(roles):
                raise ValueError("Ming output order/count disagrees with its native profile")
            if group.output_type != "pil":
                raise ValueError("Ming gallery output requires decoded RGBA images")
            if type(group.seed) is not int or not 0 <= group.seed < 2**63:
                raise ValueError("Ming group seed must be an integer in [0, 2**63)")
            if any(type(value) is not int or value <= 0 for value in (group.width, group.height)):
                raise ValueError("Ming output dimensions must be positive integers")
            if any(not isinstance(image, Image.Image) or image.mode != "RGBA"
                   or image.size != (group.width, group.height) for image in group.values):
                raise ValueError("Ming output lost RGBA channels or requested dimensions")
        return self

    @property
    def primary(self):
        return self.groups[0].values[0]
