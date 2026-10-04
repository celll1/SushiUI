"""Native target tile coordinates with full reference-frame context."""
import torch
from core.training.partition import DiTPartitionAdapter, PartitionTopology, PartitionBox


class MingPartitionAdapter(DiTPartitionAdapter):
    topology = PartitionTopology(prefix_attends_target=True, global_position_ids=True,
                                 input_token_multiple=1)

    def build_fixed_plan(self, height, width, **kwargs):
        # These are patch-grid sides; native latent sides are already aligned to patch size 2.
        return super().build_fixed_plan(height, width, allow_odd_sides=True, **kwargs)

    def position_ids(self, full_height, full_width, box, *, device=None):
        if not (0 <= box.top < box.bottom <= full_height and
                0 <= box.left < box.right <= full_width):
            raise ValueError("Ming partition box lies outside the full patch canvas")
        y, x = torch.meshgrid(torch.arange(box.top,box.bottom,device=device),
                             torch.arange(box.left,box.right,device=device),indexing="ij")
        return torch.stack((y.flatten(),x.flatten()),dim=-1)


def patchify_partition_image(model, image, reference, caption_length, patch_size,
                             frame_patch_size, position_ids):
    if frame_patch_size != 1 or patch_size != 2:
        raise ValueError("Ming target partitions require native spatial patches and independent frames")
    channels, frames, height, width = image.shape
    h, w = height // patch_size, width // patch_size
    if height % patch_size or width % patch_size or min(frames,h,w) < 1:
        raise ValueError("Ming partition targets require nonempty aligned native grids")
    if (position_ids.shape != (h*w,2) or position_ids.dtype not in (torch.int32,torch.int64)
            or (position_ids < 0).any()):
        raise ValueError("Ming partition coordinates must be nonnegative integer [H*W,2]")
    ids = position_ids.to(device=image.device,dtype=torch.int32)
    yy,xx = torch.meshgrid(torch.arange(h,device=image.device)+ids[0,0],
                           torch.arange(w,device=image.device)+ids[0,1],indexing="ij")
    expected = torch.stack((yy.flatten(),xx.flatten()),dim=-1)
    if not torch.equal(ids,expected):
        raise ValueError("Ming partition coordinates must form a contiguous row-major rectangle")

    def patches(value):
        c,f,y,x = value.shape
        return value.reshape(c,f,y//2,2,x//2,2).permute(1,2,4,3,5,0).reshape(-1,4*c)

    target_ids = torch.cat((torch.arange(caption_length+1,caption_length+1+frames,
        device=image.device,dtype=torch.int32).repeat_interleave(h*w)[:,None],ids.repeat(frames,1)),dim=1)
    tokens = patches(image)
    if reference is not None:
        if (reference.ndim != 4 or reference.shape[0] != channels or reference.shape[1] < 1
                or reference.shape[2] % 2 or reference.shape[3] % 2):
            raise ValueError("Ming partition references require aligned native C/F/H/W tensors")
        reference_ids = model.create_coordinate_grid(
            (reference.shape[1],reference.shape[2]//2,reference.shape[3]//2),
            start=(caption_length+1+frames,0,0),device=image.device).flatten(0,2)
        tokens = torch.cat((tokens,patches(reference)),dim=0)
        target_ids = torch.cat((target_ids,reference_ids),dim=0)
    return tokens,(frames,height,width),target_ids


def core_result(prediction, target, region, model_time, weights):
    """Only a region's nonoverlapping core contributes supervised flow loss."""
    from .training import MingTrainingResult
    from .guidance import frame_error
    box = region.core_in_input
    prediction = prediction[...,box.top*2:box.bottom*2,box.left*2:box.right*2]
    target = target[...,box.top*2:box.bottom*2,box.left*2:box.right*2]
    expected = (region.core.height*2,region.core.width*2)
    if prediction.shape != target.shape or prediction.shape[-2:] != expected:
        raise ValueError("Ming partition prediction disagrees with its supervised core")
    per_item = frame_error(prediction,target,weights)
    return MingTrainingResult(per_item.mean(),per_item,prediction,target,model_time)


def image_guidance_scale(trainer,sigma,condition,weights,core_pairs,plan):
    """One clean-space RMS cap spans every disjoint core and supervised frame."""
    from .guidance import frame_error
    from core.training.ops.qwen_image_21_ops import _extrapolation_cap,_capped_guidance_scale
    cap = _extrapolation_cap(trainer)
    if cap <= 0:
        return None
    distance = torch.zeros_like(sigma,dtype=torch.float32)
    pairs = list(core_pairs)
    if len(pairs) != len(plan.regions):
        raise ValueError("Ming guidance cap requires every partition core")
    for (target,uncond),region in zip(pairs,plan.regions):
        if target.shape != uncond.shape or target.shape[-2:] != (region.core.height*2,region.core.width*2):
            raise ValueError("Ming guidance cap core geometry disagrees with the partition plan")
        residual = sigma.reshape(-1,1,1,1,1)*(target.float()-uncond.float())
        distance += region.core.tokens/plan.full_tokens * frame_error(residual,torch.zeros_like(residual),weights)
    return _capped_guidance_scale(sigma,condition,distance.detach(),cap)
