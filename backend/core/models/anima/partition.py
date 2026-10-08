"""Full-canvas patch coordinates for native Anima image tiles."""
import torch

from core.training.partition import DiTPartitionAdapter, PartitionTopology


class AnimaPartitionAdapter(DiTPartitionAdapter):
    topology = PartitionTopology(prefix_attends_target=False, global_position_ids=True,
                                 input_token_multiple=1)

    def build_fixed_plan(self, height, width, **kwargs):
        return super().build_fixed_plan(height, width, allow_odd_sides=True, **kwargs)

    def position_ids(self, full_height, full_width, box, *, device=None):
        if not (0 <= box.top < box.bottom <= full_height
                and 0 <= box.left < box.right <= full_width):
            raise ValueError("Anima partition lies outside the full patch canvas")
        y, x = torch.meshgrid(torch.arange(box.top, box.bottom, device=device),
                             torch.arange(box.left, box.right, device=device), indexing="ij")
        return torch.stack((y.flatten(), x.flatten()), dim=-1)


def region_rotary(embedder, grid, box, *, factor=(1, 1), fps=None):
    ah, aw = factor
    if any(type(value) is not int or value < 1 for value in factor):
        raise ValueError("Anima coarse coordinate factors must be positive integers")
    if box.top % ah or box.left % aw or box.height % ah or box.width % aw:
        raise ValueError("Anima region must align with its cumulative U-Net factors")
    if tuple(grid.shape[2:4]) != (box.height // ah, box.width // aw):
        raise ValueError("Anima region grid disagrees with its global coordinates")
    return embedder.generate_embeddings(grid.shape, fps=fps,
        spatial_origin=(box.top // ah, box.left // aw))
