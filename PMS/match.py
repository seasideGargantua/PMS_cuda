import torch
import torch.nn.functional as F
from .auxilary import PMSSetting, compute_gradients, compute_gray, plane2disparity, initiate_cost
from .propagation import propagation

def match(
    img_left: torch.Tensor,
    img_right: torch.Tensor,
    setting: PMSSetting
):
    """Cost Volume Matching."""
    assert img_left.shape == img_right.shape, "Left and right images must have the same shape"
    assert len(img_left.shape) == 3, "Input images must be 3D tensors"
    height, width = img_left.shape[:2]

    # Random initiate disparity and normal and plane
    disp_left = torch.rand((height, width), device=img_left.device) * (setting.max_disparity - setting.min_disparity) + setting.min_disparity
    disp_right = -1. * (torch.rand((height, width), device=img_left.device) * (setting.max_disparity - setting.min_disparity) + setting.min_disparity)
    if not setting.is_fource_fpw:
        norm_left = torch.rand((height, width, 3), device=img_left.device) * 2 - 1
        norm_right = torch.rand((height, width, 3), device=img_left.device) * 2 - 1
        norm_left = torch.where(norm_left == 0, torch.tensor(0.1, device=norm_left.device, dtype=norm_left.dtype), norm_left)
        norm_right = torch.where(norm_right == 0, torch.tensor(0.1, device=norm_left.device, dtype=norm_left.dtype), norm_right)
    else:
        norm_left = torch.zeros((height, width, 3), device=img_left.device)
        norm_right = torch.zeros((height, width, 3), device=img_left.device)
        norm_left[..., -1] = 1.
        norm_right[..., -1] = 1.
    # p.x = -n.x / n.z;
    # p.y = -n.y / n.z;
    # p.z = (n.x * x + n.y * y + n.z * d) / n.z;
    xy_plane = torch.stack(torch.meshgrid(torch.arange(height, device=img_left.device), torch.arange(width, device=img_left.device)), dim=-1).float()
    
    plane_left = torch.cat([-norm_left[..., :2]/norm_left[..., 2:3], 
                            (norm_left * torch.cat([xy_plane, disp_left.unsqueeze(-1)], dim=-1)).sum(dim=-1, keepdim=True)], dim=-1)
    plane_right = torch.cat([-norm_right[..., :2]/norm_right[..., 2:3], 
                             (norm_right * torch.cat([xy_plane, disp_right.unsqueeze(-1)], dim=-1)).sum(dim=-1, keepdim=True)], dim=-1)

    # Compute gray
    gray_left = compute_gray(img_left)
    gray_right = compute_gray(img_right)

    # Compute gradients
    grad_left = compute_gradients(gray_left)
    grad_right = compute_gradients(gray_right)

    # Initiate cost
    cost_left, cost_right = initiate_cost(
        setting,
        img_left,
        img_right,
        grad_left,
        grad_right,
        plane_left,
        plane_right
    )

    rand_d = torch.rand((height, width, 2, setting.num_iters), device=img_left.device) * 2 - 1
    rand_n = torch.rand((height, width, 3, 2, setting.num_iters), device=img_left.device) * 2 - 1

    # propagation
    for i in range(setting.num_iters):
        plane_left, plane_right, cost_left, cost_right = propagation(
            i,
            setting,
            img_left,
            img_right,
            grad_left,
            grad_right,
            plane_left,
            plane_right,
            cost_left,
            cost_right,
            rand_d[..., 0, i],
            rand_n[..., 0, i],
            left_img=True
        )
        plane_right, plane_left, cost_right, cost_left = propagation(
            i,
            setting,
            img_left,
            img_right,
            grad_left,
            grad_right,
            plane_left,
            plane_right,
            cost_left,
            cost_right,
            rand_d[..., 1, i],
            rand_n[..., 1, i],
            left_img=False
        )
        __import__('ipdb').set_trace()

    disp_left = plane2disparity(plane_left)
    disp_right = plane2disparity(plane_right)
    return disp_left, disp_right